"""GeoPackage (OGC) features via sqlite3 + KML (PLAN 21g, shared GeoPackage of 21h).

write_layer(gpkg_path, table, geoms, attrs, srs_id, srs_wkt, srs_name)  - creates / replaces a feature table, registers
    it in gpkg_contents / gpkg_geometry_columns, adds an R-tree spatial index when SQLite has the rtree module.
read_layer(gpkg_path, table) -> (geoms, attrs, srs_wkt);  list_layers(gpkg_path)
Geometry blobs: GeoPackageBinary header (magic GP, version 0, little endian, XY envelope) + WKB.
KML: write_kml(path, layers) with lon / lat geometries; read_kml(path) -> (geoms, attrs).
"""
import os
import sqlite3
import struct
import time
import xml.etree.ElementTree as ET

import numpy as np

WKB = {"Point": 1, "LineString": 2, "Polygon": 3, "MultiPoint": 4, "MultiLineString": 5, "MultiPolygon": 6}
WKB_NAMES = {v: k for k, v in WKB.items()}


# ------------------------------------------------------------------------------------------- WKB
def _ring_bytes(p):
    p = np.asarray(p, "<f8").reshape(-1, 2)
    return struct.pack("<I", len(p)) + p.tobytes()


def to_wkb(g):
    t, parts = g["type"], [np.asarray(p, float).reshape(-1, 2) for p in g["parts"]]
    if t == "Point":
        return struct.pack("<BI2d", 1, 1, *parts[0][0])
    if t == "MultiPoint":
        pts = np.concatenate(parts)
        return struct.pack("<BII", 1, 4, len(pts)) + b"".join(struct.pack("<BI2d", 1, 1, *p) for p in pts)
    if t == "LineString":
        if len(parts) == 1:
            return struct.pack("<BI", 1, 2) + _ring_bytes(parts[0])
        return struct.pack("<BII", 1, 5, len(parts)) + b"".join(struct.pack("<BI", 1, 2) + _ring_bytes(p) for p in parts)
    if t == "Polygon":
        from .shapefile import group_rings
        polys = group_rings(parts)
        if len(polys) == 1:
            return struct.pack("<BII", 1, 3, len(polys[0])) + b"".join(_ring_bytes(r) for r in polys[0])
        return struct.pack("<BII", 1, 6, len(polys)) + b"".join(
            struct.pack("<BII", 1, 3, len(pl)) + b"".join(_ring_bytes(r) for r in pl) for pl in polys)
    raise ValueError(t)


def from_wkb(b, pos=0):
    """-> (geometry dict, next position)."""
    bo = "<" if b[pos] == 1 else ">"
    t, = struct.unpack(bo + "I", b[pos + 1:pos + 5])
    t = t % 1000  # ISO Z / M codes
    pos += 5

    def ring(pos):
        n, = struct.unpack(bo + "I", b[pos:pos + 4])
        a = np.frombuffer(b, bo + "f8", 2 * n, pos + 4).reshape(-1, 2).copy()
        return a, pos + 4 + 16 * n

    if t == 1:
        x, y = struct.unpack(bo + "2d", b[pos:pos + 16])
        return {"type": "Point", "parts": [np.array([[x, y]])]}, pos + 16
    if t == 2:
        a, pos = ring(pos)
        return {"type": "LineString", "parts": [a]}, pos
    if t == 3:
        n, = struct.unpack(bo + "I", b[pos:pos + 4])
        pos += 4
        rings = []
        for _ in range(n):
            a, pos = ring(pos)
            rings.append(a)
        return {"type": "Polygon", "parts": rings}, pos
    if t in (4, 5, 6):
        n, = struct.unpack(bo + "I", b[pos:pos + 4])
        pos += 4
        parts = []
        for _ in range(n):
            g, pos = from_wkb(b, pos)
            parts += g["parts"]
        return {"type": {4: "MultiPoint", 5: "LineString", 6: "Polygon"}[t],
                "parts": [np.concatenate(parts)] if t == 4 else parts}, pos
    raise ValueError(f"WKB type {t}")


def gpkg_blob(g, srs_id):
    allp = np.concatenate([np.asarray(p, float).reshape(-1, 2) for p in g["parts"]])
    env = (allp[:, 0].min(), allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max())
    return b"GP" + bytes([0, 0b00000011]) + struct.pack("<i4d", srs_id, *env) + to_wkb(g)


def parse_blob(b):
    if b is None:
        return None
    b = bytes(b)
    flags = b[3]
    env = (flags >> 1) & 0x7
    elen = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(env, 0)
    return from_wkb(b, 8 + elen)[0]


# ------------------------------------------------------------------------------------------- GeoPackage
def _init(con):
    con.execute("PRAGMA application_id = 1196444487")
    con.execute("PRAGMA user_version = 10200")
    con.executescript("""
    CREATE TABLE IF NOT EXISTS gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER NOT NULL PRIMARY KEY,
        organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL, definition TEXT NOT NULL, description TEXT);
    CREATE TABLE IF NOT EXISTS gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL,
        identifier TEXT UNIQUE, description TEXT DEFAULT '', last_change DATETIME NOT NULL DEFAULT
        (strftime('%Y-%m-%dT%H:%M:%fZ','now')), min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER);
    CREATE TABLE IF NOT EXISTS gpkg_geometry_columns (table_name TEXT NOT NULL, column_name TEXT NOT NULL,
        geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL,
        CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name));
    CREATE TABLE IF NOT EXISTS gpkg_extensions (table_name TEXT, column_name TEXT, extension_name TEXT NOT NULL,
        definition TEXT NOT NULL, scope TEXT NOT NULL);
    """)
    for row in ((" Undefined cartesian SRS", -1, "NONE", -1, "undefined", ""),
                ("Undefined geographic SRS", 0, "NONE", 0, "undefined", "")):
        con.execute("INSERT OR IGNORE INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)", row)


def _sqltype(vals):
    nn = [v for v in vals if v is not None]
    if nn and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in nn):
        return "INTEGER"
    if nn and all(isinstance(v, (int, float, np.integer, np.floating)) for v in nn):
        return "REAL"
    return "TEXT"


def _py(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def write_layer(path, table, geoms, attrs=None, srs_id=-1, srs_wkt="undefined", srs_name="", description="",
                geometry_type=None):
    attrs = attrs or {}
    con = sqlite3.connect(path)
    try:
        _init(con)
        if srs_id not in (-1, 0):
            con.execute("INSERT OR REPLACE INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)",
                        (srs_name or f"EPSG:{srs_id}", srs_id, "EPSG" if srs_id < 100000 else "NONE", srs_id, srs_wkt, ""))
        con.execute(f'DROP TABLE IF EXISTS "{table}"')
        con.execute(f'DROP TABLE IF EXISTS "rtree_{table}_geom"')
        con.execute("DELETE FROM gpkg_contents WHERE table_name = ?", (table,))
        con.execute("DELETE FROM gpkg_geometry_columns WHERE table_name = ?", (table,))
        con.execute("DELETE FROM gpkg_extensions WHERE table_name = ?", (table,))
        cols = ", ".join(f'"{n}" {_sqltype(v)}' for n, v in attrs.items())
        con.execute(f'CREATE TABLE "{table}" (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom BLOB' + (", " + cols if cols else "") + ")")
        names = list(attrs)
        q = f'INSERT INTO "{table}" (geom' + "".join(f', "{n}"' for n in names) + ") VALUES (?" + ",?" * len(names) + ")"
        rows = []
        env = [np.inf, np.inf, -np.inf, -np.inf]
        boxes = []
        for i, g in enumerate(geoms):
            blob = gpkg_blob(g, srs_id) if g else None
            rows.append([blob] + [_py(attrs[n][i]) for n in names])
            if g:
                p = np.concatenate([np.asarray(pp, float).reshape(-1, 2) for pp in g["parts"]])
                b = (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max())
                boxes.append((i + 1,) + tuple(float(v) for v in b))
                env = [min(env[0], b[0]), min(env[1], b[2]), max(env[2], b[1]), max(env[3], b[3])]
        con.executemany(q, rows)
        gtype = {"Point": "POINT", "MultiPoint": "MULTIPOINT", "LineString": "GEOMETRY", "Polygon": "GEOMETRY"}.get(
            next((g["type"] for g in geoms if g), "Point"), "GEOMETRY")
        if geometry_type:  # e.g. empty tables the user digitises into (QGIS needs the concrete type)
            gtype = {"Point": "POINT", "LineString": "LINESTRING", "Polygon": "POLYGON"}.get(geometry_type, geometry_type)
        con.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, description, min_x, min_y, max_x, max_y, srs_id)"
                    " VALUES (?,?,?,?,?,?,?,?,?)", (table, "features", table, description,
                                                    *(float(v) if np.isfinite(v) else None for v in env), srs_id))
        con.execute("INSERT INTO gpkg_geometry_columns VALUES (?,?,?,?,0,0)", (table, "geom", gtype, srs_id))
        try:
            con.execute(f'CREATE VIRTUAL TABLE "rtree_{table}_geom" USING rtree(id, minx, maxx, miny, maxy)')
            con.executemany(f'INSERT INTO "rtree_{table}_geom" VALUES (?,?,?,?,?)', boxes)
            con.execute("INSERT INTO gpkg_extensions VALUES (?,?,?,?,?)", (table, "geom", "gpkg_rtree_index",
                                                                         "http://www.geopackage.org/spec120/#extension_rtree",
                                                                         "write-only"))
        except sqlite3.OperationalError:
            pass  # SQLite without rtree: the layer still works, just without the spatial index
        con.commit()
    finally:
        con.close()
    return path


def list_layers(path):
    con = sqlite3.connect(path)
    try:
        return [r[0] for r in con.execute("SELECT table_name FROM gpkg_contents WHERE data_type = 'features'")]
    finally:
        con.close()


def read_layer(path, table):
    con = sqlite3.connect(path)
    try:
        gcol = con.execute("SELECT column_name, srs_id FROM gpkg_geometry_columns WHERE table_name = ?", (table,)).fetchone()
        col, srs = gcol if gcol else ("geom", -1)
        wkt = con.execute("SELECT definition FROM gpkg_spatial_ref_sys WHERE srs_id = ?", (srs,)).fetchone()
        cur = con.execute(f'SELECT * FROM "{table}"')
        names = [d[0] for d in cur.description]
        geoms, attrs = [], {n: [] for n in names if n not in (col, "fid")}
        for row in cur:
            r = dict(zip(names, row))
            geoms.append(parse_blob(r[col]))
            for n in attrs:
                attrs[n].append(r[n])
        return geoms, attrs, (wkt[0] if wkt and wkt[0] != "undefined" else None)
    finally:
        con.close()


def has_rtree(path, table):
    con = sqlite3.connect(path)
    try:
        return bool(con.execute("SELECT count(*) FROM sqlite_master WHERE name = ?", (f"rtree_{table}_geom",)).fetchone()[0])
    finally:
        con.close()


# ------------------------------------------------------------------------------------------- KML
def _coords(a):
    return " ".join(f"{x:.8f},{y:.8f},0" for x, y in np.asarray(a, float).reshape(-1, 2))


def write_kml(path, layers, name="GeoHarmonic"):
    """layers: [(layer name, geoms (lon / lat), attrs, label field or None)]."""
    k = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
         f"<name>{_esc(name)}</name>"]
    for lname, geoms, attrs, label in layers:
        k.append(f"<Folder><name>{_esc(lname)}</name>")
        names = list(attrs or {})
        for i, g in enumerate(geoms):
            if g is None:
                continue
            nm = str(attrs[label][i]) if label and label in attrs else f"{lname} {i + 1}"
            ext = "".join(f'<Data name="{_esc(n)}"><value>{_esc(attrs[n][i])}</value></Data>' for n in names)
            k.append(f"<Placemark><name>{_esc(nm)}</name><ExtendedData>{ext}</ExtendedData>")
            t = g["type"]
            if t in ("Point", "MultiPoint"):
                pts = np.concatenate(g["parts"])
                if len(pts) == 1:
                    k.append(f"<Point><coordinates>{_coords(pts)}</coordinates></Point>")
                else:
                    k.append("<MultiGeometry>" + "".join(f"<Point><coordinates>{_coords([p])}</coordinates></Point>"
                                                        for p in pts) + "</MultiGeometry>")
            elif t == "LineString":
                k.append("<MultiGeometry>" + "".join(f"<LineString><coordinates>{_coords(p)}</coordinates></LineString>"
                                                    for p in g["parts"]) + "</MultiGeometry>")
            else:
                from .shapefile import group_rings
                polys = group_rings(g["parts"])
                k.append("<MultiGeometry>")
                for pl in polys:
                    k.append(f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{_coords(pl[0])}</coordinates></LinearRing>"
                             "</outerBoundaryIs>" + "".join(f"<innerBoundaryIs><LinearRing><coordinates>{_coords(h)}"
                                                            "</coordinates></LinearRing></innerBoundaryIs>" for h in pl[1:])
                             + "</Polygon>")
                k.append("</MultiGeometry>")
            k.append("</Placemark>")
        k.append("</Folder>")
    k.append("</Document></kml>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(k))
    return path


def _esc(v):
    return str(v if v is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def read_kml(path):
    """Placemarks -> (geoms in lon / lat, attrs {"name": [...]})."""
    tree = ET.parse(path)
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    geoms, names = [], []

    def pts(el):
        txt = (el.text or "").strip().split()
        return np.array([[float(v) for v in t.split(",")[:2]] for t in txt]) if txt else np.zeros((0, 2))

    for pm in tree.iter("{http://www.opengis.net/kml/2.2}Placemark"):
        nm = pm.find("k:name", ns)
        names.append(nm.text if nm is not None else "")
        polys = pm.findall(".//k:Polygon", ns)
        lines = pm.findall(".//k:LineString", ns)
        points = pm.findall(".//k:Point", ns)
        if polys:
            parts = []
            for pg_ in polys:
                for c in pg_.findall(".//k:coordinates", ns):
                    parts.append(pts(c))
            geoms.append({"type": "Polygon", "parts": parts})
        elif lines:
            geoms.append({"type": "LineString", "parts": [pts(l.find("k:coordinates", ns)) for l in lines]})
        elif points:
            p = np.concatenate([pts(q.find("k:coordinates", ns)) for q in points])
            geoms.append({"type": "Point" if len(p) == 1 else "MultiPoint", "parts": [p]})
        else:
            geoms.append(None)
    return geoms, {"name": names}


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def remove(path):
    for ext in ("", "-journal", "-wal", "-shm"):
        try:
            os.remove(path + ext)
        except OSError:
            pass
