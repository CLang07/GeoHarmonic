"""DLIS reader checks: a synthetic DLIS written here byte by byte (multi-segment records, padding, trailing
length, absent attributes, array channels, IBM floats), plus the real Tui-1 / Pateke-2 files when present."""
import glob
import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "data", "nz", "wells"))


# ---------------------------------------------------------------------- minimal RP66 v1 writer
def _ident(s):
    b = s.encode("latin-1")
    return bytes([len(b)]) + b


def _uvari(n):
    if n < 128:
        return bytes([n])
    if n < 16384:
        return struct.pack(">H", 0x8000 | n)
    return struct.pack(">I", 0xC0000000 | n)


def _obname(name, origin=1, copy=0):
    return _uvari(origin) + bytes([copy]) + _ident(name)


def _val(code, v):
    if code == 19:
        return _ident(v)
    if code == 20:
        b = v.encode("latin-1")
        return _uvari(len(b)) + b
    if code == 2:
        return struct.pack(">f", v)
    if code == 7:
        return struct.pack(">d", v)
    if code == 14:
        return struct.pack(">i", v)
    if code == 16:
        return struct.pack(">H", v)
    if code == 15:
        return bytes([v])
    if code == 18:
        return _uvari(v)
    if code == 23:
        return _obname(v)
    if code == 27:
        return _ident(v)
    raise ValueError(code)


def eflr(set_type, template, objects):
    """template: [(label, code, units)]; objects: [(name, {label: value list | None(absent)})]."""
    out = bytes([0xF0]) + _ident(set_type)  # SET with type
    for label, code, units in template:
        d = 0x20 | 0x10 | 0x04 | (0x02 if units else 0)
        out += bytes([d]) + _ident(label) + bytes([code]) + (_ident(units) if units else b"")
    for name, attrs in objects:
        out += bytes([0x70]) + _obname(name)
        for label, code, units in template:
            v = attrs.get(label, "absent")
            if v is None or (isinstance(v, str) and v == "absent"):
                out += bytes([0x00])  # ABSATR
                continue
            vals = v if isinstance(v, list) else [v]
            out += bytes([0x20 | 0x08 | 0x01]) + _uvari(len(vals)) + b"".join(_val(code, x) for x in vals)
    return out


def write_dlis(path, records, max_seg=60, vr_len=200):
    """records: [(is_eflr, type, body)].  Bodies are split into small segments (padding + trailing length
    on alternate segments) and packed into short visible records to exercise the reassembly."""
    segs = []
    for is_eflr, typ, body in records:
        chunks = [body[i:i + max_seg] for i in range(0, len(body), max_seg)] or [b""]
        for k, ch in enumerate(chunks):
            attr = (0x80 if is_eflr else 0) | (0x40 if k > 0 else 0) | (0x20 if k < len(chunks) - 1 else 0)
            trailer = b""
            pad = b""
            if k % 2 == 0:
                attr |= 0x01  # padding: 3 bytes, last one = count
                pad = b"\x00\x00\x03"
            if k % 3 == 1:
                attr |= 0x02  # trailing length
            seg_len = 4 + len(ch) + len(pad) + (2 if attr & 0x02 else 0)
            if seg_len % 2:  # segment lengths are even: one more pad byte
                if attr & 0x01:
                    pad = b"\x00" + pad[:-1] + bytes([pad[-1] + 1])
                else:
                    attr |= 0x01
                    pad = b"\x01"
                seg_len += 1
            if attr & 0x02:
                trailer = struct.pack(">H", seg_len)
            segs.append(struct.pack(">HBB", seg_len, attr, typ) + ch + pad + trailer)
    sul = b"   1V1.00RECORD 8192" + b"Default Storage Set".ljust(60)
    out = bytearray(sul)
    cur = b""
    for sg in segs:
        if len(cur) + len(sg) + 4 > vr_len and cur:
            out += struct.pack(">HBB", len(cur) + 4, 0xFF, 1) + cur
            cur = b""
        cur += sg
    if cur:
        out += struct.pack(">HBB", len(cur) + 4, 0xFF, 1) + cur
    with open(path, "wb") as fh:
        fh.write(bytes(out))


def _ibm(x):
    from .dlis import ibm32  # noqa: F401  (inverse used only for the test file)
    x = float(x)
    if x == 0:
        return b"\x00\x00\x00\x00"
    s = 0x80 if x < 0 else 0
    a = abs(x)
    e = 64
    while a >= 1:
        a /= 16
        e += 1
    while a < 1 / 16:
        a *= 16
        e -= 1
    return bytes([s | e]) + int(round(a * 16777216)).to_bytes(3, "big")


def synthetic_dlis(path, n=120):
    depth01in = (2000.0 + np.arange(n) * 0.1524) / 0.00254  # metres -> 0.1 in, increasing
    depth01in = depth01in[::-1]  # logged upwards (DECREASING)
    gr = 50 + 40 * np.sin(np.arange(n) / 9.0)
    rhob = 2.3 + 0.2 * np.cos(np.arange(n) / 13.0)
    gr[5] = -999.25  # absent value
    recs = []
    recs.append((True, 0, eflr("FILE-HEADER", [("SEQUENCE-NUMBER", 20, ""), ("ID", 20, "")],
                               [("0", {"SEQUENCE-NUMBER": ["1"], "ID": ["SYNTH TEST FILE"]})])))
    recs.append((True, 1, eflr("ORIGIN", [("FILE-ID", 20, ""), ("WELL-NAME", 20, ""), ("COMPANY", 20, ""),
                                          ("FIELD-NAME", 20, "")],
                               [("DLIS_DEFINING_ORIGIN", {"FILE-ID": ["SYNTH"], "WELL-NAME": ["Synth-1  "],
                                                          "COMPANY": ["GeoHarmonic"], "FIELD-NAME": None})])))
    tmpl = [("LONG-NAME", 20, ""), ("REPRESENTATION-CODE", 15, ""), ("UNITS", 27, ""), ("DIMENSION", 18, "")]
    recs.append((True, 3, eflr("CHANNEL", tmpl, [
        ("TDEP", {"LONG-NAME": ["Depth"], "REPRESENTATION-CODE": [2], "UNITS": ["0.1 in"], "DIMENSION": [1]}),
        ("GR", {"LONG-NAME": ["Gamma Ray"], "REPRESENTATION-CODE": [2], "UNITS": ["gAPI"], "DIMENSION": [1]}),
        ("RHOB", {"LONG-NAME": ["Bulk density"], "REPRESENTATION-CODE": [7], "UNITS": ["g/cm3"], "DIMENSION": [1]}),
        ("CNT", {"LONG-NAME": ["Counter"], "REPRESENTATION-CODE": [14], "UNITS": [""], "DIMENSION": [1]}),
        ("WF", {"LONG-NAME": ["Waveform"], "REPRESENTATION-CODE": [16], "UNITS": [""], "DIMENSION": [6]}),
        ("IBMV", {"LONG-NAME": ["IBM value"], "REPRESENTATION-CODE": [5], "UNITS": ["V"], "DIMENSION": [1]}),
    ])))
    recs.append((True, 4, eflr("FRAME", [("INDEX-TYPE", 19, ""), ("DIRECTION", 19, ""), ("CHANNELS", 23, "")],
                               [("60B", {"INDEX-TYPE": ["BOREHOLE-DEPTH"], "DIRECTION": ["DECREASING"],
                                         "CHANNELS": ["TDEP", "GR", "RHOB", "CNT", "WF", "IBMV"]})])))
    recs.append((True, 5, eflr("PARAMETER", [("LONG-NAME", 20, ""), ("VALUES", 2, "m")],
                               [("BS", {"LONG-NAME": ["Bit size"], "VALUES": [0.2159]}),
                                ("EMPTY", {"LONG-NAME": ["absent"], "VALUES": None})])))
    for k in range(n):
        body = _obname("60B") + _uvari(k + 1) + struct.pack(">f", depth01in[k]) + struct.pack(">f", gr[k]) + \
            struct.pack(">d", rhob[k]) + struct.pack(">i", k * 3) + \
            b"".join(struct.pack(">H", k + q) for q in range(6)) + _ibm(-1.5 * k)
        recs.append((False, 0, body))
    write_dlis(path, recs)
    return {"depth_m": depth01in * 0.00254, "gr": gr, "rhob": rhob, "n": n}


def run(check, tmp):
    from .dlis import DlisFile, DlisLogs, checkshot_from_dlis, dlis_to_las
    from .las import LasFile
    p = os.path.join(tmp, "synth.dlis")
    truth = synthetic_dlis(p)
    f = DlisFile(p)
    check(len(f.logical_files) == 1, "DLIS: logical file found (storage unit label, visible records)")
    lf = f.logical_files[0]
    check(lf.origin_info().get("WELL-NAME") == "Synth-1" and "FIELD-NAME" not in lf.origin_info(),
          "DLIS: ORIGIN parsed, absent attribute handled")
    fr = lf.frames.get("60B")
    check(fr is not None and len(fr.records) == truth["n"] and fr.direction == "DECREASING",
          f"DLIS: frame with {len(fr.records) if fr else 0} rows reassembled from multi-segment records")
    d = lf.frame_data("60B")
    check(np.allclose(d["TDEP"] * 0.00254, truth["depth_m"], atol=1e-3) and
          np.allclose(d["RHOB"], truth["rhob"]) and np.array_equal(d["CNT"], np.arange(truth["n"]) * 3),
          "DLIS: FSINGL / FDOUBL / SLONG channel values exact")
    check(d["WF"].shape == (truth["n"], 6) and d["WF"][10, 5] == 15, "DLIS: array channel (dimension 6)")
    check(np.allclose(d["IBMV"], -1.5 * np.arange(truth["n"])), "DLIS: IBM float channel")
    check(len(lf.parameters) == 2 and abs(lf.parameters[0].first("VALUES") - 0.2159) < 1e-6 and
          lf.parameters[0].units("VALUES") == "m", "DLIS: PARAMETER values and units")
    out = dlis_to_las(p, os.path.join(tmp, "las_out"))
    las = LasFile(out[0]["path"])
    gr = next(c for c in las.curves if c.mnemonic == "GR")
    check(len(out) == 1 and out[0]["skipped"] == ["WF"] and np.allclose(las.depth, truth["depth_m"], atol=1e-3)
          and np.isnan(gr.values[5]) and np.allclose(gr.values[6:], truth["gr"][6:], atol=1e-3),
          "DLIS -> LAS 2.0 (depth in m, absent -> NULL, array channel skipped) reads back with the LAS reader")
    out2 = dlis_to_las(p, os.path.join(tmp, "las_arr"), arrays=True)
    check("WF[5]" in out2[0]["curves"], "DLIS -> LAS with --arrays writes array elements")
    logs = DlisLogs(p)
    c = next(c for c in logs.curves if c.mnemonic == "RHOB")
    check(np.all(np.diff(c.depth) > 0) and logs.well_name == "Synth-1", "DLIS as a LAS-like source (increasing depth)")

    # ---- real files (if the data folder is present)
    tui = sorted(glob.glob(os.path.join(DATA, "Tui-1", "dlis", "*.DLIS")))
    if not tui:
        print("SKIP real DLIS checks (data/nz not found)")
        return
    import time
    t0 = time.time()
    f = DlisFile(tui[0])
    lf = f.logical_files[0]
    check(lf.origin_info().get("WELL-NAME") == "Tui-1", f"Tui-1 DLIS parsed in {time.time() - t0:.1f} s")
    ranges = {}
    for fr in lf.frames.values():
        dd = lf.frame_data(fr.name)
        for k, (lo, hi) in {"GR": (0, 200), "RHOZ": (1.5, 3.1), "TNPH": (-0.05, 0.7), "DTCO": (40, 200)}.items():
            if k in dd:
                v = dd[k][(dd[k] > -999) & (np.abs(dd[k]) < 1e29)]
                ranges[k] = (float(np.percentile(v, 1)), float(np.percentile(v, 99)), lo, hi)
    okr = len(ranges) == 4 and all(lo <= a and b <= hi for a, b, lo, hi in ranges.values())
    check(okr, "Tui-1 log values in sane ranges: " + ", ".join(f"{k} {a:.2f}-{b:.2f}" for k, (a, b, _, _) in ranges.items()))
    cs = sorted(glob.glob(os.path.join(DATA, "Tui-1", "checkshot", "*MAIN*.DLIS")))
    if cs:
        t = checkshot_from_dlis(cs[0])
        vavg = t["tvd"] / (t["owt_ms"] / 1000.0)
        check(t is not None and len(t["md"]) >= 5 and np.all(np.diff(t["twt_ms"]) > 0) and
              1500 < vavg.min() and vavg.max() < 4500,
              f"Tui-1 checkshot: {len(t['md'])} levels, TWT {t['twt_ms'].min():.0f}-{t['twt_ms'].max():.0f} ms, "
              f"v_avg {vavg.min():.0f}-{vavg.max():.0f} m/s")
