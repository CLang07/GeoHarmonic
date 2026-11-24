"""GeoTIFF reader / writer (PLAN 21g, overviews for 21h) - single band, struct + zlib + numpy.

write(path, z, x0, y0, dx, dy, epsg=None, wkt=None, nodata=-9999, compress=True, overviews=(2, 4, 8))
    z rows = +y (south first, like MapGrid "xy"); stored north-up.  Georeferencing: ModelPixelScale + ModelTiepoint,
    GeoKeys with the EPSG code when known, else the WKT goes into a GDAL .aux.xml sidecar (QGIS / ArcGIS read it).
    Overviews = reduced-resolution IFDs (NewSubfileType 1), NaN-aware 2x averaging.
read(path) -> {"z" (rows = +y), "x0", "y0", "dx", "dy", "epsg", "nodata", "overviews"}
    strips or tiles; compression none / deflate (8, 32946); predictor 1 / 2 / 3; uint8 / int16 / uint16 / int32 /
    float32 / float64.  (LZW / JPEG-compressed files: convert with gdal_translate -co COMPRESS=DEFLATE.)
"""
import os
import struct
import zlib

import numpy as np

DT = {(1, 8): "u1", (2, 8): "i1", (1, 16): "u2", (2, 16): "i2", (1, 32): "u4", (2, 32): "i4", (3, 32): "f4",
      (3, 64): "f8"}
TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8}
TYPE_FMT = {1: "B", 2: "s", 3: "H", 4: "I", 5: "II", 6: "b", 7: "B", 8: "h", 9: "i", 10: "ii", 11: "f", 12: "d", 16: "Q"}


def _downsample(z):
    ny, nx = z.shape
    ny2, nx2 = (ny + 1) // 2, (nx + 1) // 2
    pad = np.full((ny2 * 2, nx2 * 2), np.nan, np.float64)
    pad[:ny, :nx] = z
    b = pad.reshape(ny2, 2, nx2, 2)
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(b).sum((1, 3))
        s = np.nansum(b, (1, 3))
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def write(path, z, x0, y0, dx, dy, epsg=None, wkt=None, nodata=-9999.0, compress=True, overviews=(2, 4, 8),
          rows_per_strip=64):
    z = np.asarray(z, np.float64)
    levels = [z]
    for f in overviews or ():
        cur = levels[-1]
        if min(cur.shape) < 8:
            break
        levels.append(_downsample(cur))
    out = bytearray(b"II*\x00\x00\x00\x00\x00")
    ifd_offsets = []
    for li, lev in enumerate(levels):
        data = np.where(np.isfinite(lev), lev, nodata).astype("<f4")[::-1]  # north-up
        ny, nx = data.shape
        strips = []
        for r0 in range(0, ny, rows_per_strip):
            raw = data[r0:r0 + rows_per_strip].tobytes()
            strips.append(zlib.compress(raw, 6) if compress else raw)
        offs = []
        for s in strips:
            if len(out) % 2:
                out += b"\x00"
            offs.append(len(out))
            out += s
        scale = 2 ** li if li else 1
        tags = [(254, 4, [1 if li else 0]), (256, 4, [nx]), (257, 4, [ny]), (258, 3, [32]), (259, 3, [8 if compress else 1]),
                (262, 3, [1]), (273, 4, offs), (277, 3, [1]), (278, 4, [rows_per_strip]), (279, 4, [len(s) for s in strips]),
                (284, 3, [1]), (339, 3, [3])]
        if li == 0:
            top = y0 + (z.shape[0] - 1) * dy + dy / 2.0
            left = x0 - dx / 2.0
            keys = [1, 1, 0, 0]
            kv = []
            if epsg:
                geog = epsg in (4326, 4269, 4267, 4167)
                kv += [(1024, 0, 1, 2 if geog else 1), (1025, 0, 1, 1), (2048 if geog else 3072, 0, 1, int(epsg))]
            else:
                kv += [(1024, 0, 1, 1), (1025, 0, 1, 1)]
            keys[3] = len(kv)
            for k in kv:
                keys += list(k)
            tags += [(33550, 12, [abs(dx) * scale, abs(dy) * scale, 0.0]), (33922, 12, [0.0, 0.0, 0.0, left, top, 0.0]),
                     (34735, 3, keys), (42113, 2, (repr(float(nodata)) + "\x00").encode())]
        ifd_offsets.append(_write_ifd(out, tags))
    # link IFDs: header points to the first; each IFD's next pointer to the following one
    struct.pack_into("<I", out, 4, ifd_offsets[0][0])
    for (off, next_ptr_pos), nxt in zip(ifd_offsets, ifd_offsets[1:] + [(0, None)]):
        struct.pack_into("<I", out, next_ptr_pos, nxt[0])
    with open(path, "wb") as f:
        f.write(bytes(out))
    if wkt and not epsg:
        with open(path + ".aux.xml", "w") as f:
            f.write(f"<PAMDataset>\n  <SRS>{_xml(wkt)}</SRS>\n</PAMDataset>\n")
    return path


def _xml(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _write_ifd(out, tags):
    """Append an IFD (+ out-of-line values); returns (ifd offset, position of its next-IFD pointer)."""
    tags = sorted(tags, key=lambda t: t[0])
    extra = []
    entries = []
    for tag, typ, vals in tags:
        if typ == 2:
            raw = vals if isinstance(vals, bytes) else bytes(vals)
            count = len(raw)
        else:
            fmt = {3: "H", 4: "I", 12: "d", 11: "f"}[typ]
            raw = struct.pack("<" + fmt * len(vals), *vals)
            count = len(vals)
        entries.append([tag, typ, count, raw])
    if len(out) % 2:
        out += b"\x00"
    ifd_off = len(out)
    size = 2 + 12 * len(entries) + 4
    data_pos = ifd_off + size
    blob = bytearray()
    packed = []
    for tag, typ, count, raw in entries:
        if len(raw) <= 4:
            packed.append(struct.pack("<HHI", tag, typ, count) + raw.ljust(4, b"\x00"))
        else:
            if (data_pos + len(blob)) % 2:
                blob += b"\x00"
            packed.append(struct.pack("<HHII", tag, typ, count, data_pos + len(blob)))
            blob += raw
    out += struct.pack("<H", len(entries)) + b"".join(packed)
    next_pos = len(out)
    out += b"\x00\x00\x00\x00" + blob
    return ifd_off, next_pos


def _read_ifds(buf):
    bo = "<" if buf[:2] == b"II" else ">"
    magic, = struct.unpack(bo + "H", buf[2:4])
    if magic != 42:
        raise ValueError("not a (classic) TIFF")
    off, = struct.unpack(bo + "I", buf[4:8])
    ifds = []
    while off:
        n, = struct.unpack(bo + "H", buf[off:off + 2])
        tags = {}
        for k in range(n):
            e = off + 2 + 12 * k
            tag, typ, count = struct.unpack(bo + "HHI", buf[e:e + 8])
            size = TYPE_SIZE.get(typ, 1) * count
            vpos = e + 8 if size <= 4 else struct.unpack(bo + "I", buf[e + 8:e + 12])[0]
            raw = buf[vpos:vpos + size]
            if typ == 2:
                val = raw.split(b"\x00")[0].decode("latin-1")
            else:
                f = TYPE_FMT[typ]
                val = list(struct.unpack(bo + f * count, raw)) if typ not in (5, 10) else \
                    [a / b if b else 0 for a, b in zip(*[iter(struct.unpack(bo + f[0] * 2 * count, raw))] * 2)]
            tags[tag] = val
        ifds.append(tags)
        off, = struct.unpack(bo + "I", buf[off + 2 + 12 * n: off + 6 + 12 * n])
    return bo, ifds


def _band(buf, bo, t):
    nx, ny = t[256][0], t[257][0]
    bps = t.get(258, [8])[0]
    fmt = t.get(339, [1])[0]
    dt = np.dtype(DT[(fmt, bps)]).newbyteorder(bo)
    comp = t.get(259, [1])[0]
    pred = t.get(317, [1])[0]
    if comp not in (1, 8, 32946):
        raise ValueError(f"TIFF compression {comp} not supported (only none / deflate)")

    def dec(raw):
        return zlib.decompress(raw) if comp in (8, 32946) else raw

    if 322 in t:  # tiled
        tw, th = t[322][0], t[323][0]
        offs, cnts = t[324], t[325]
        out = np.zeros((ny, nx), dt)
        ntx = (nx + tw - 1) // tw
        for k, (o, c) in enumerate(zip(offs, cnts)):
            tile = np.frombuffer(dec(buf[o:o + c]), dt, tw * th).reshape(th, tw)
            tile = _unpredict(tile, pred)
            r, cc = (k // ntx) * th, (k % ntx) * tw
            h, w = min(th, ny - r), min(tw, nx - cc)
            out[r:r + h, cc:cc + w] = tile[:h, :w]
        return out
    rps = t.get(278, [ny])[0]
    rows = []
    for o, c in zip(t[273], t[279]):
        raw = dec(buf[o:o + c])
        k = len(raw) // dt.itemsize // nx
        rows.append(_unpredict(np.frombuffer(raw, dt, k * nx).reshape(k, nx), pred))
    return np.concatenate(rows)[:ny]


def _unpredict(a, pred):
    if pred == 2:
        return np.cumsum(a, axis=1, dtype=a.dtype)
    return a


def read(path):
    with open(path, "rb") as f:
        buf = f.read()
    bo, ifds = _read_ifds(buf)
    main = next(t for t in ifds if not (t.get(254, [0])[0] & 1))
    data = _band(buf, bo, main).astype(np.float64)
    nod = main.get(42113)
    nodata = None
    if nod:
        try:
            nodata = float(nod)
        except ValueError:
            nodata = None
    if nodata is not None:
        data[data == nodata] = np.nan
    sx, sy = main.get(33550, [1.0, 1.0, 0.0])[:2]
    tp = main.get(33922, [0, 0, 0, 0.0, 0.0, 0])
    i0, j0, x_left, y_top = tp[0], tp[1], tp[3], tp[4]
    ny = data.shape[0]
    epsg = None
    keys = main.get(34735)
    raster_point = False
    if keys:
        for k in range(keys[3]):
            kid, loc, cnt, val = keys[4 + 4 * k: 8 + 4 * k]
            if kid in (3072, 2048) and loc == 0 and val not in (0, 32767):
                epsg = val if kid == 3072 or epsg is None else epsg
            if kid == 1025 and loc == 0 and val == 2:
                raster_point = True
    off = 0.0 if raster_point else 0.5
    x0 = x_left + (off - i0) * sx
    ytop_c = y_top - (off - j0) * sy
    wkt = None
    if os.path.exists(path + ".aux.xml"):
        import re
        m = re.search(r"<SRS[^>]*>(.*?)</SRS>", open(path + ".aux.xml").read(), re.S)
        if m:
            wkt = m.group(1).replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return {"z": data[::-1].copy(), "x0": x0, "y0": ytop_c - (ny - 1) * sy, "dx": sx, "dy": sy, "epsg": epsg,
            "nodata": nodata, "overviews": sum(1 for t in ifds if t.get(254, [0])[0] & 1), "wkt": wkt}
