"""Live link with QGIS (PLAN 21h).

1. Shared GeoPackage <project>_gis.gpkg (+ <project>_gis_grids/*.tif with overviews) written by GeoHarmonic:
   wells (status + petro summary), tops, 2D lines, 3D outline, faults (trace + footprint polygon), contours, prospects /
   polygons, petro values, vector layers; default QGIS styles in the GeoPackage `layer_styles` table (QGIS applies them
   when the layer is loaded); `gh_meta` (project, CRS, time) and `gh_rasters` (grid GeoTIFFs) for the plugin.
   User-editable tables (prospects_edit, leases, section_lines) are created empty and read back with change detection.
2. Messages over localhost TCP (127.0.0.1, JSON lines, stdlib socket): LinkEndpoint("server") in GeoHarmonic, the QGIS
   plugin connects as client.  Types: hello, select_wells {wells}, zoom {bbox}, open_well {well}, section_line {xy},
   polygon {name, xy}, cursor {x, y}, current_section {xy}, refresh {gpkg}.  Off unless enabled; failures are reported,
   never raised into the GUI.
"""
import hashlib
import json
import os
import queue
import socket
import threading
import time

import numpy as np

DEFAULT_PORT = 47561
EDITABLE = {"prospects_edit": "Polygon", "leases": "Polygon", "section_lines": "LineString"}


# ------------------------------------------------------------------------------------------- shared GeoPackage
def shared_paths(project):
    base = project.path[: -len(".seisproj.json")] if project.path.endswith(".seisproj.json") else project.path
    return base + "_gis.gpkg", base + "_gis_grids"


def _qml_vector(kind, color, width=0.6, size=2.4, label=None, fill_alpha=60):
    r, g, b = color[:3]
    if kind == "Point":
        sym = (f'<symbol type="marker" name="0" alpha="1"><layer class="SimpleMarker"><prop k="color" v="{r},{g},{b},255"/>'
               f'<prop k="name" v="circle"/><prop k="size" v="{size}"/><prop k="outline_color" v="255,255,255,255"/></layer></symbol>')
    elif kind == "LineString":
        sym = (f'<symbol type="line" name="0" alpha="1"><layer class="SimpleLine"><prop k="line_color" v="{r},{g},{b},255"/>'
               f'<prop k="line_width" v="{width}"/></layer></symbol>')
    else:
        sym = (f'<symbol type="fill" name="0" alpha="1"><layer class="SimpleFill"><prop k="color" v="{r},{g},{b},{fill_alpha}"/>'
               f'<prop k="outline_color" v="{r},{g},{b},255"/><prop k="outline_width" v="{width}"/></layer></symbol>')
    lab = ""
    if label:
        lab = (f'<labeling type="simple"><settings><text-style fieldName="{label}" fontSize="8" textColor="0,0,0,255">'
               '<text-buffer bufferDraw="1" bufferSize="0.8" bufferColor="255,255,255,255"/></text-style>'
               '<placement placement="1"/></settings></labeling>')
    return ('<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'><qgis version="3.22" styleCategories="AllStyleCategories">'
            f'<renderer-v2 type="singleSymbol"><symbols>{sym}</symbols></renderer-v2>{lab}</qgis>')


def _write_styles(gp, styles):
    import sqlite3
    con = sqlite3.connect(gp)
    try:
        con.execute("""CREATE TABLE IF NOT EXISTS layer_styles (id INTEGER PRIMARY KEY AUTOINCREMENT, f_table_catalog TEXT,
            f_table_schema TEXT, f_table_name TEXT, f_geometry_column TEXT, styleName TEXT, styleQML TEXT, styleSLD TEXT,
            useAsDefault BOOLEAN, description TEXT, owner TEXT, ui TEXT, update_time DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        for table, qml in styles.items():
            con.execute("DELETE FROM layer_styles WHERE f_table_name = ?", (table,))
            con.execute("INSERT INTO layer_styles (f_table_catalog, f_table_schema, f_table_name, f_geometry_column, styleName,"
                        " styleQML, styleSLD, useAsDefault, description, owner) VALUES ('', '', ?, 'geom', ?, ?, '', 1,"
                        " 'GeoHarmonic', 'GeoHarmonic')", (table, table, qml))
        con.commit()
    finally:
        con.close()


def _meta(gp, rows, rasters):
    import sqlite3
    con = sqlite3.connect(gp)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS gh_meta (key TEXT PRIMARY KEY, value TEXT)")
        con.executemany("INSERT OR REPLACE INTO gh_meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in rows.items()])
        con.execute("DROP TABLE IF EXISTS gh_rasters")
        con.execute("CREATE TABLE gh_rasters (name TEXT, path TEXT, ramp TEXT, vmin REAL, vmax REAL)")
        con.executemany("INSERT INTO gh_rasters VALUES (?,?,?,?,?)", rasters)
        con.commit()
    finally:
        con.close()


def _fault_features(project):
    geoms, names = [], []
    polys, pnames = [], []
    for f in getattr(project, "faults", None).items if getattr(project, "faults", None) is not None else []:
        o = f.ordered_sticks()
        if len(o) > 1:
            geoms.append({"type": "LineString", "parts": [np.array([[s[0, 0], s[0, 1]] for s in o])]})
            names.append(f.name)
        pts = np.concatenate([s[:, :2] for s in f.sticks]) if f.sticks else np.zeros((0, 2))
        if len(pts) >= 3:
            hull = _convex_hull(pts)
            if len(hull) >= 3:
                polys.append({"type": "Polygon", "parts": [hull]})
                pnames.append(f.name)
    return (geoms, {"name": names}), (polys, {"name": pnames})


def _convex_hull(p):
    p = np.unique(np.asarray(p, float), axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(tuple(q))
    for q in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(tuple(q))
    h = np.array(lower[:-1] + upper[:-1])
    return np.vstack([h, h[:1]])


def write_shared(project, stack):
    """(Re)write the shared GeoPackage + grid GeoTIFFs.  Returns {"gpkg", "tables", "rasters"}."""
    from .gis import gpkg as GP, geotiff as GT, crs as CRS
    from .gis.export import project_crs, layer_features
    from .maplayers import well_status, display_grid
    from . import maphub as MH
    gp, gdir = shared_paths(project)
    os.makedirs(gdir, exist_ok=True)
    c = project_crs(project)
    srs = int(c.get("epsg") or 99999) if c else -1
    wkt = CRS.to_wkt(c) if c else "undefined"
    cname = c["name"] if c else ""
    tables, styles = [], {}
    # keep the user's edits: read editable tables before rewriting
    user = {}
    if os.path.exists(gp):
        for t in EDITABLE:
            try:
                user[t] = GP.read_layer(gp, t)
            except Exception:
                pass

    def put(name, geoms, attrs, kind, color, label=None, width=0.6):
        if not geoms:
            return
        GP.write_layer(gp, name, geoms, attrs, srs, wkt, cname)
        tables.append(name)
        styles[name] = _qml_vector(kind, color, width=width, label=label)

    # wells with status + petro summary (all zones)
    ws = project.wells()
    if ws:
        status = well_status(project)
        zv = MH.zone_values(project, None)
        cols = sorted({k for d in zv.values() for k in d})
        attrs = {"name": [w[0] for w in ws], "status": [status.get(w[0], "unknown") for w in ws]}
        for col in cols:
            attrs[col[:30]] = [zv.get(w[0], {}).get(col) for w in ws]
        put("wells", [{"type": "Point", "parts": [np.array([[x, y]])]} for _, x, y in ws], attrs, "Point", (0, 0, 0), "name")
        # tops at the well location (vertical-well approximation)
        try:
            with open(project.well_project) as f:
                wj = json.load(f)
            xy = {n: (x, y) for n, x, y in ws}
            tg, ta = [], {"well": [], "formation": [], "md": []}
            for wn, tops in (wj.get("tops") or {}).items():
                if wn not in xy:
                    continue
                for fm, md in tops.items():
                    tg.append({"type": "Point", "parts": [np.array([xy[wn]])]})
                    ta["well"].append(wn)
                    ta["formation"].append(fm)
                    ta["md"].append(md)
            put("tops", tg, ta, "Point", (120, 0, 120))
        except Exception:
            pass
    (fg, fa), (pg_, pa) = _fault_features(project)
    put("faults", fg, fa, "LineString", (200, 0, 0), "name", 0.8)
    put("fault_polygons", pg_, pa, "Polygon", (200, 0, 0))
    rasters = []
    for lay in stack.ordered():
        k = lay["kind"]
        if k in ("wells", "faults", "text"):
            continue
        if k in ("grid", "horizon"):
            g = display_grid(stack, lay["id"])
            if g is None:
                continue
            gx = g.to_xy() if g.kind == "ilxl" else g
            fn = os.path.join(gdir, "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in lay["name"]) + ".tif")
            GT.write(fn, gx.z, gx.x0, gx.y0, gx.dx, gx.dy, epsg=c.get("epsg") if c else None, wkt=wkt if c else None)
            f = gx.z[np.isfinite(gx.z)]
            rasters.append((lay["name"], fn, lay["style"].get("ramp") or "", float(f.min()) if f.size else 0.0,
                            float(f.max()) if f.size else 1.0))
            continue
        feats = layer_features(stack, lay)
        if not feats or not feats[1]:
            continue
        gtype, geoms, attrs = feats
        name = {"outline3d": "survey_3d", "lines2d": "lines_2d", "polygons": "prospects" if "prospect" in lay["name"].lower()
                else None, "contours": None, "petro": None, "vector": None}.get(k)
        name = name or "".join(ch if ch.isalnum() else "_" for ch in lay["name"].lower())[:40]
        color = tuple(lay["style"].get("color") or (60, 60, 60))[:3]
        put(name, geoms, attrs, gtype, color, "name" if "name" in attrs else ("level" if "level" in attrs else None))
    # editable tables (kept if the user already edited them)
    for t, kind in EDITABLE.items():
        if t in user:
            g, a, _ = user[t]
            GP.write_layer(gp, t, [x for x in g], a, srs, wkt, cname, geometry_type=kind)
        else:
            GP.write_layer(gp, t, [], {"name": []}, srs, wkt, cname, geometry_type=kind)
        tables.append(t)
        styles[t] = _qml_vector(kind, (255, 0, 255), width=0.8, label="name")
    _write_styles(gp, styles)
    _meta(gp, {"project": project.path, "written": time.strftime("%Y-%m-%d %H:%M:%S"), "crs_epsg": c.get("epsg") if c else None,
               "app": "GeoHarmonic"}, rasters)
    return {"gpkg": gp, "tables": tables, "rasters": [r[1] for r in rasters]}


def table_hash(gp, table):
    import sqlite3
    con = sqlite3.connect(gp)
    try:
        h = hashlib.sha1()
        for row in con.execute(f'SELECT * FROM "{table}" ORDER BY fid'):
            h.update(repr(row).encode())
        return h.hexdigest()
    except sqlite3.OperationalError:
        return None
    finally:
        con.close()


def read_user_edits(project, last=None):
    """Editable tables whose content changed since `last` ({table: hash}).  Returns (changed {table: (geoms, attrs)},
    new hashes)."""
    from .gis import gpkg as GP
    gp, _ = shared_paths(project)
    last = dict(last or {})
    changed = {}
    if not os.path.exists(gp):
        return changed, last
    for t in EDITABLE:
        h = table_hash(gp, t)
        if h is not None and h != last.get(t):
            last[t] = h
            g, a, _ = GP.read_layer(gp, t)
            changed[t] = (g, a)
    return changed, last


def apply_user_edits(stack, changed):
    """Prospect / lease polygons edited in QGIS -> polygons layers 'Prospects (QGIS)' / 'Leases (QGIS)'; section lines are
    returned for the caller to open.  Returns {"layers": [ids], "section_lines": [(name, xs, ys)]}."""
    out = {"layers": [], "section_lines": []}
    for t, (geoms, attrs) in changed.items():
        names = attrs.get("name") or [f"{t} {i + 1}" for i in range(len(geoms))]
        if t == "section_lines":
            for g, n in zip(geoms, names):
                if g:
                    p = g["parts"][0]
                    out["section_lines"].append((n or "line", p[:, 0], p[:, 1]))
            continue
        lname = "Prospects (QGIS)" if t == "prospects_edit" else "Leases (QGIS)"
        polys = []
        for g, n in zip(geoms, names):
            if g:
                p = g["parts"][0]
                polys.append({"name": n or "", "x": [float(v) for v in p[:, 0]], "y": [float(v) for v in p[:, 1]],
                              "from": "gpkg"})
        lay = next((l for l in stack.layers if l["kind"] == "polygons" and l["name"] == lname), None)
        if lay is None:
            lid = stack.add("polygons", lname, src={"polygons": polys})
        else:  # replace the polygons that came from the GeoPackage table, keep those sent as messages
            lay["src"]["polygons"] = [q for q in lay["src"]["polygons"] if q.get("from") != "gpkg"] + polys
            lid = lay["id"]
        out["layers"].append(lid)
    if out["layers"]:
        stack.save()
    return out


# ------------------------------------------------------------------------------------------- messaging
class LinkEndpoint:
    """JSON-lines over localhost TCP.  role 'server' (GeoHarmonic) or 'client' (QGIS).  Incoming messages go to a queue
    (poll() them from a GUI timer); send() never raises (returns False when not connected)."""

    def __init__(self, role="server", port=DEFAULT_PORT, host="127.0.0.1", name="GeoHarmonic"):
        self.role, self.port, self.host, self.name = role, int(port), host, name
        self.inbox = queue.Queue()
        self._sock = None
        self._conn = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.error = None
        self._thread = None

    # ---- lifecycle
    def start(self, timeout=2.0):
        try:
            if self.role == "server":
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.host, self.port))
                s.listen(1)
                s.settimeout(0.2)
                self._sock = s
                self._thread = threading.Thread(target=self._serve, daemon=True)
            else:
                c = socket.create_connection((self.host, self.port), timeout=timeout)
                c.settimeout(0.2)
                self._conn = c
                self._thread = threading.Thread(target=self._read_loop, args=(c,), daemon=True)
                self.send({"type": "hello", "app": self.name})
            self._thread.start()
            return True
        except OSError as e:
            self.error = f"{type(e).__name__}: {e}"
            return False

    def stop(self):
        self._stop.set()
        for s in (self._conn, self._sock):
            try:
                if s:
                    s.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def connected(self):
        return self._conn is not None

    # ---- io
    def _serve(self):
        while not self._stop.is_set():
            try:
                c, _ = self._sock.accept()
            except (socket.timeout, OSError):
                continue
            c.settimeout(0.2)
            with self._lock:
                if self._conn is not None:
                    try:
                        self._conn.close()
                    except OSError:
                        pass
                self._conn = c
            self.send({"type": "hello", "app": self.name})
            self._read_loop(c)

    def _read_loop(self, c):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = c.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    try:
                        self.inbox.put(json.loads(line.decode("utf-8")))
                    except ValueError:
                        pass
        with self._lock:
            if self._conn is c:
                self._conn = None
        self.inbox.put({"type": "disconnected"})

    def send(self, msg):
        with self._lock:
            c = self._conn
        if c is None:
            return False
        try:
            c.sendall((json.dumps(msg, default=_jd) + "\n").encode("utf-8"))
            return True
        except OSError as e:
            self.error = str(e)
            return False

    def poll(self, max_items=100):
        out = []
        while len(out) < max_items:
            try:
                out.append(self.inbox.get_nowait())
            except queue.Empty:
                break
        return out

    def wait(self, kind, timeout=5.0):
        """Block until a message of `kind` arrives (tests / scripts)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                m = self.inbox.get(timeout=0.05)
            except queue.Empty:
                continue
            if m.get("type") == kind:
                return m
        return None


def _jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


# ------------------------------------------------------------------------------------------- message handling
def handle(msg, app):
    """Apply one message from QGIS to GeoHarmonic.  `app` provides: select_wells(names), zoom(bbox), open_well(name),
    section_line(name, xs, ys), polygon(name, xs, ys), refresh().  Returns a short description (for the status bar)."""
    t = msg.get("type")
    if t == "select_wells":
        app.select_wells(list(msg.get("wells") or []))
        return f"QGIS selected {len(msg.get('wells') or [])} wells"
    if t == "zoom":
        app.zoom(msg.get("bbox"))
        return "QGIS zoom"
    if t == "open_well":
        app.open_well(msg.get("well"))
        return f"QGIS: open {msg.get('well')}"
    if t == "section_line":
        xy = np.asarray(msg.get("xy") or [], float).reshape(-1, 2)
        app.section_line(msg.get("name", "QGIS line"), xy[:, 0], xy[:, 1])
        return "QGIS line -> seismic section"
    if t == "polygon":
        xy = np.asarray(msg.get("xy") or [], float).reshape(-1, 2)
        app.polygon(msg.get("name", "QGIS polygon"), xy[:, 0], xy[:, 1])
        return f"QGIS polygon {msg.get('name', '')} -> prospect"
    if t == "refresh":
        app.refresh()
        return "QGIS asked for a refresh"
    if t in ("hello", "disconnected"):
        return f"QGIS link: {t}"
    return f"QGIS link: ignored {t}"
