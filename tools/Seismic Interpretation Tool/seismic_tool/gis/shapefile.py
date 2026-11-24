"""ESRI shapefile (.shp / .shx / .dbf / .prj) reader and writer (PLAN 21g) - struct + numpy, no dependencies.

Geometry = {"type": "Point" | "MultiPoint" | "LineString" | "Polygon", "parts": [array (n, 2)]}; polygons keep all
rings (outer rings clockwise, holes counter-clockwise - the shapefile rule; ring_roles() tells them apart).
Attributes = {field: list}.  Z / M shapes are read as 2D.
"""
import datetime
import os
import struct

import numpy as np

SHP_TYPES = {0: None, 1: "Point", 3: "LineString", 5: "Polygon", 8: "MultiPoint", 11: "Point", 13: "LineString",
             15: "Polygon", 18: "MultiPoint", 21: "Point", 23: "LineString", 25: "Polygon", 28: "MultiPoint"}
TYPE_CODES = {"Point": 1, "LineString": 3, "Polygon": 5, "MultiPoint": 8}


def signed_area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])


def ring_roles(parts):
    """'outer' / 'hole' per ring by orientation (shapefile: clockwise outer = negative signed area)."""
    return ["outer" if signed_area(p) <= 0 else "hole" for p in parts]


def group_rings(parts):
    """[[outer, hole, ...], ...]: rings with the FIRST ring's orientation are outer rings (works for shapefile CW-outer
    and OGC CCW-outer input alike)."""
    polys = []
    s0 = None
    for p in parts:
        p = np.asarray(p, float).reshape(-1, 2)
        s = np.sign(signed_area(p)) or -1.0
        if s0 is None:
            s0 = s
        if s == s0 or not polys:
            polys.append([p])
        else:
            polys[-1].append(p)
    return polys


# ------------------------------------------------------------------------------------------- read
def read(path):
    """-> (geoms, attrs, prj_wkt or None)."""
    base = os.path.splitext(path)[0]
    with open(base + ".shp", "rb") as f:
        buf = f.read()
    code, = struct.unpack(">i", buf[:4])
    if code != 9994:
        raise ValueError(f"{path}: not a shapefile")
    geoms = []
    pos = 100
    n = len(buf)
    while pos + 8 <= n:
        _, clen = struct.unpack(">ii", buf[pos:pos + 8])
        pos += 8
        rec = buf[pos:pos + 2 * clen]
        pos += 2 * clen
        st, = struct.unpack("<i", rec[:4])
        kind = SHP_TYPES.get(st)
        if kind is None:
            geoms.append(None)
            continue
        if kind == "Point":
            x, y = struct.unpack("<2d", rec[4:20])
            geoms.append({"type": "Point", "parts": [np.array([[x, y]])]})
            continue
        if kind == "MultiPoint":
            npts, = struct.unpack("<i", rec[36:40])
            pts = np.frombuffer(rec, "<f8", 2 * npts, 40).reshape(-1, 2).copy()
            geoms.append({"type": "MultiPoint", "parts": [pts]})
            continue
        nparts, npts = struct.unpack("<2i", rec[36:44])
        starts = np.frombuffer(rec, "<i4", nparts, 44)
        pts = np.frombuffer(rec, "<f8", 2 * npts, 44 + 4 * nparts).reshape(-1, 2).copy()
        bounds = list(starts) + [npts]
        geoms.append({"type": kind, "parts": [pts[bounds[k]:bounds[k + 1]] for k in range(nparts)]})
    attrs = read_dbf(base + ".dbf") if os.path.exists(base + ".dbf") else {}
    prj = None
    if os.path.exists(base + ".prj"):
        with open(base + ".prj", errors="replace") as f:
            prj = f.read()
    return geoms, attrs, prj


def read_dbf(path, encoding="latin-1"):
    with open(path, "rb") as f:
        buf = f.read()
    nrec, hlen, rlen = struct.unpack("<IHH", buf[4:12])
    fields = []
    pos = 32
    while buf[pos] != 0x0D and pos < hlen:
        name = buf[pos:pos + 11].split(b"\x00")[0].decode(encoding, "replace")
        typ = chr(buf[pos + 11])
        ln, dec = buf[pos + 16], buf[pos + 17]
        fields.append((name, typ, ln, dec))
        pos += 32
    out = {f[0]: [] for f in fields}
    for r in range(nrec):
        rec = buf[hlen + r * rlen: hlen + (r + 1) * rlen]
        if not rec or rec[:1] == b"*":
            continue
        p = 1
        for name, typ, ln, dec in fields:
            raw = rec[p:p + ln].decode(encoding, "replace").strip()
            p += ln
            if typ in "NF":
                try:
                    v = float(raw) if raw else None
                    if v is not None and dec == 0 and typ == "N" and float(v).is_integer() and abs(v) < 2 ** 53:
                        v = int(v)
                except ValueError:
                    v = None
            elif typ == "L":
                v = raw.upper() in ("T", "Y") if raw else None
            else:
                v = raw
            out[name].append(v)
    return out


# ------------------------------------------------------------------------------------------- write
def _bbox(parts):
    allp = np.concatenate(parts) if parts else np.zeros((0, 2))
    return (float(allp[:, 0].min()), float(allp[:, 1].min()), float(allp[:, 0].max()), float(allp[:, 1].max())) \
        if len(allp) else (0.0, 0.0, 0.0, 0.0)


def _fix_rings(parts):
    """Close rings and orient them: outer clockwise, holes counter-clockwise (holes = rings inside an earlier ring
    by orientation as given: the first ring and any ring with the first ring's orientation are outer)."""
    out = []
    first_sign = None
    for p in parts:
        p = np.asarray(p, float)
        if len(p) and (p[0] != p[-1]).any():
            p = np.vstack([p, p[:1]])
        a = signed_area(p)
        if first_sign is None:
            first_sign = np.sign(a) or -1.0
        is_outer = np.sign(a) == first_sign
        if (is_outer and a > 0) or (not is_outer and a < 0):
            p = p[::-1]
        out.append(p)
    return out


def write(path, geoms, attrs=None, prj=None, kind=None):
    """Write .shp/.shx/.dbf (+ .prj when prj WKT is given).  All geometries must share one type."""
    base = os.path.splitext(path)[0]
    kind = kind or next((g["type"] for g in geoms if g), "Point")
    st = TYPE_CODES[kind]
    recs = []
    for g in geoms:
        if g is None:
            recs.append(struct.pack("<i", 0))
            continue
        parts = [np.asarray(p, float).reshape(-1, 2) for p in g["parts"]]
        if kind == "Polygon":
            parts = _fix_rings(parts)
        if kind == "Point":
            recs.append(struct.pack("<i2d", 1, *parts[0][0]))
        elif kind == "MultiPoint":
            pts = np.concatenate(parts)
            recs.append(struct.pack("<i4di", 8, *_bbox([pts]), len(pts)) + pts.astype("<f8").tobytes())
        else:
            pts = np.concatenate(parts)
            starts = np.cumsum([0] + [len(p) for p in parts[:-1]]).astype("<i4")
            recs.append(struct.pack("<i4d2i", st, *_bbox(parts), len(parts), len(pts)) + starts.tobytes()
                        + pts.astype("<f8").tobytes())
    allp = [p for g in geoms if g for p in g["parts"]]
    bb = _bbox([np.asarray(p, float).reshape(-1, 2) for p in allp]) if allp else (0.0, 0.0, 0.0, 0.0)

    def header(length_words):
        return struct.pack(">i5ii", 9994, 0, 0, 0, 0, 0, length_words) + struct.pack("<ii8d", 1000, st, *bb, 0, 0, 0, 0)

    body = b""
    index = b""
    off = 50
    for i, r in enumerate(recs):
        body += struct.pack(">ii", i + 1, len(r) // 2) + r
        index += struct.pack(">ii", off, len(r) // 2)
        off += 4 + len(r) // 2
    with open(base + ".shp", "wb") as f:
        f.write(header(50 + len(body) // 2) + body)
    with open(base + ".shx", "wb") as f:
        f.write(header(50 + len(index) // 2) + index)
    write_dbf(base + ".dbf", attrs or {"id": list(range(1, len(geoms) + 1))}, len(geoms))
    if prj:
        with open(base + ".prj", "w") as f:
            f.write(prj)
    return base + ".shp"


def write_dbf(path, attrs, nrec=None, encoding="latin-1"):
    names = list(attrs)
    nrec = nrec if nrec is not None else (len(attrs[names[0]]) if names else 0)
    fields = []
    for n in names:
        vals = list(attrs[n])
        nn = [v for v in vals if v is not None and v != ""]
        if nn and all(isinstance(v, (bool, np.bool_)) for v in nn):
            fields.append((n, "L", 1, 0))
        elif nn and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in nn):
            fields.append((n, "N", max(len(str(int(v))) for v in nn) + 1, 0))
        elif nn and all(isinstance(v, (int, float, np.integer, np.floating)) for v in nn):
            fields.append((n, "F", 19, 8))
        else:
            ln = max([len(str(v).encode(encoding, "replace")) for v in nn] + [1])
            fields.append((n, "C", min(254, ln), 0))
    rlen = 1 + sum(f[2] for f in fields)
    hlen = 32 + 32 * len(fields) + 1
    d = datetime.date.today()
    out = bytearray(struct.pack("<BBBBIHH20x", 3, d.year - 1900, d.month, d.day, nrec, hlen, rlen))
    for name, typ, ln, dec in fields:
        nb = name.encode(encoding, "replace")[:10]
        out += nb + b"\x00" * (11 - len(nb)) + typ.encode() + b"\x00" * 4 + bytes([ln, dec]) + b"\x00" * 14
    out += b"\x0D"
    for i in range(nrec):
        out += b" "
        for name, typ, ln, dec in fields:
            v = attrs[name][i] if i < len(attrs[name]) else None
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                s = ""
            elif typ == "L":
                s = "T" if v else "F"
            elif typ == "N":
                s = str(int(v))
            elif typ == "F":
                s = f"{float(v):.8g}"
                if len(s) > ln:
                    s = f"{float(v):.{max(1, ln - 7)}e}"
            else:
                s = str(v)
            b = s.encode(encoding, "replace")[:ln]
            out += (b.rjust(ln) if typ in "NF" else b.ljust(ln))
    out += b"\x1A"
    with open(path, "wb") as f:
        f.write(bytes(out))
    return path
