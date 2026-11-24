"""PLSS (Public Land Survey System) legal descriptions (PLAN 21g): parse "Sec 14, T12N, R3W" style text, find the
section polygon in a PLSS sections layer (BLM CadNSDI or any layer with section / township / range fields) and spot a
well from footage calls ("660 FNL & 1980 FWL").  The footage is measured from the section's bounding lines (exact for
regular sections; irregular sections should be checked against the plat).
"""
import re

import numpy as np

FIELD_GUESS = {
    "section": ("FRSTDIVNO", "SECTION", "SEC", "SECNO", "SECTN", "SECTION_NO"),
    "township": ("TWNSHPNO", "TOWNSHIP", "TWP", "TWPNO", "TOWNSHIPNO"),
    "tdir": ("TWNSHPDIR", "TDIR", "TWPDIR", "NS"),
    "range": ("RANGENO", "RANGE", "RNG", "RNGNO", "RANGE_NO"),
    "rdir": ("RANGEDIR", "RDIR", "RNGDIR", "EW"),
    "meridian": ("PRINMERCD", "MERIDIAN", "PM", "PRINMER"),
}


def parse_legal(text):
    """-> {"section", "township", "tdir", "range", "rdir", "meridian", "calls": [(feet, "N"|"S"|"E"|"W")]}."""
    t = text.upper().replace(",", " ").replace("&", " ")
    out = {"section": None, "township": None, "tdir": None, "range": None, "rdir": None, "meridian": None, "calls": []}
    m = re.search(r"\b(\d{1,2})\s*-\s*(\d{1,3})\s*([NS])\s*-\s*(\d{1,3})\s*([EW])\b", t)  # 14-12N-3W
    if m:
        out.update(section=int(m.group(1)), township=int(m.group(2)), tdir=m.group(3), range=int(m.group(4)),
                   rdir=m.group(5))
    m = re.search(r"\b(?:T|TWP|TOWNSHIP)\s*\.?\s*(\d{1,3})\s*(N|S|NORTH|SOUTH)\b", t)
    if m:
        out.update(township=int(m.group(1)), tdir=m.group(2)[0])
    m = re.search(r"\b(?:R|RNG|RANGE)\s*\.?\s*(\d{1,3})\s*(E|W|EAST|WEST)\b", t)
    if m:
        out.update(range=int(m.group(1)), rdir=m.group(2)[0])
    m = re.search(r"\b(?:S|SEC|SECTION)\s*\.?\s*(\d{1,2})\b", t)
    if m:
        out["section"] = int(m.group(1))
    m = re.search(r"\b(\d{1,2})(?:ST|ND|RD|TH)\s*(?:P\.?\s*M\.?|PRINCIPAL)", t)
    if m:
        out["meridian"] = m.group(1)
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:'|FT|FEET)?\s*F([NSEW])L\b", t):
        out["calls"].append((float(m.group(1)), m.group(2)))
    return out


def _field(attrs, key):
    names = {n.upper(): n for n in attrs}
    return next((names[c] for c in FIELD_GUESS[key] if c in names), None)


def _num(v):
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        m = re.search(r"\d+", str(v or ""))
        return int(m.group()) if m else None


def find_section(vl, legal):
    """Index of the section feature matching a parsed legal description (None if not found)."""
    a = vl.attrs
    fs, ft, fr = _field(a, "section"), _field(a, "township"), _field(a, "range")
    ftd, frd = _field(a, "tdir"), _field(a, "rdir")
    if not (fs and ft and fr):
        raise ValueError("the layer needs section / township / range fields (e.g. CadNSDI FRSTDIVNO, TWNSHPNO, RANGENO)")
    for i in range(len(vl)):
        if _num(a[fs][i]) != legal["section"] or _num(a[ft][i]) != legal["township"] or _num(a[fr][i]) != legal["range"]:
            continue
        if legal["tdir"] and ftd and str(a[ftd][i]).strip().upper()[:1] != legal["tdir"]:
            continue
        if legal["rdir"] and frd and str(a[frd][i]).strip().upper()[:1] != legal["rdir"]:
            continue
        return i
    return None


def spot(vl, text, unit_m=None):
    """(x, y) of a legal description: section centre, or the footage-call position.  unit_m = metres per map unit
    (default from the layer CRS, 1.0)."""
    lg = parse_legal(text)
    i = find_section(vl, lg)
    if i is None:
        return None
    p = np.concatenate(vl.geoms[i]["parts"])
    x0, x1, y0, y1 = p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()
    if unit_m is None:
        unit_m = (vl.crs or {}).get("unit", 1.0) if vl.crs and vl.crs["kind"] != "geographic" else 1.0
    f = 0.3048 / unit_m
    x, y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    for d, side in lg["calls"]:
        if side == "N":
            y = y1 - d * f
        elif side == "S":
            y = y0 + d * f
        elif side == "W":
            x = x0 + d * f
        elif side == "E":
            x = x1 - d * f
    return float(x), float(y), i


def synthetic_township(x0=0.0, y0=0.0, mile=1609.344, township=12, tdir="N", rng=3, rdir="W"):
    """36 one-mile sections of a regular township (numbered 1 at the NE corner, boustrophedon) - for tests."""
    geoms, sec = [], []
    n = 1
    for row in range(6):  # from the north
        cols = range(5, -1, -1) if row % 2 == 0 else range(6)
        for c in cols:
            xa, ya = x0 + c * mile, y0 + (5 - row) * mile
            geoms.append({"type": "Polygon", "parts": [np.array([[xa, ya], [xa, ya + mile], [xa + mile, ya + mile],
                                                                [xa + mile, ya], [xa, ya]])]})
            sec.append(n)
            n += 1
    k = len(geoms)
    return geoms, {"FRSTDIVNO": sec, "TWNSHPNO": [f"{township:03d}"] * k, "TWNSHPDIR": [tdir] * k,
                   "RANGENO": [f"{rng:03d}"] * k, "RANGEDIR": [rdir] * k}
