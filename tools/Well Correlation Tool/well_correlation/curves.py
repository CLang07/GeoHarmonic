"""Curve families, aliases, unit normalisation and compositing (splicing) of runs/passes.

A *family* is what the display uses: GR, RDEEP, RSHAL, RHOB, NPHI, DT, PE, SP, CALI.
Each family has an ordered alias list: earlier = preferred.  Users can add aliases in the
project file (``"aliases": {"GR": ["MYGR"]}``) without touching code.
"""
import re
import numpy as np

DEFAULT_ALIASES = {
    "GR": ["GR", "GR_EDTC", "ECGR", "GRC", "GR_ARC", "GRXS", "GRXB", "SGR", "HSGR", "CGR", "GRD", "GRS", "GAM", "GAPI"],
    "RDEEP": ["RT", "HLLD", "LLD", "AT90", "AHT90", "RLA5", "HDRS", "P40H", "A40H", "RPCEHM", "RD", "ILD", "RILD", "IDPH", "RDEP"],
    "RSHAL": ["HLLS", "LLS", "AT10", "AHT10", "RLA1", "ILM", "RILM", "SFLU", "SFLA", "P16H", "A16H", "LL7", "SN", "RS", "MSFL", "RXOZ"],
    "RHOB": ["RHOZ", "RHOB", "DEN", "ZDEN", "RHO8", "FDC", "BDCFM"],
    "NPHI": ["TNPH", "NPHI", "NPOR", "CNC", "CNCF", "NPRL", "SNPL", "SNP", "TNPL", "NPLFM"],
    "DT": ["DTCO", "DTC", "DT", "DTLN", "DTL", "DT24", "BCSL", "BHC", "SONI", "AC", "DT4P"],
    "PE": ["PEFZ", "PEF", "PE", "PEFL"],
    "SP": ["SP", "SPAR"],
    "CALI": ["HCAL", "CALI", "CALS", "CAL", "C1", "CAL1"],
}

FAMILY_INFO = {
    # family: (display name, default left, default right, log scale)
    "GR": ("GR (API)", 0.0, 150.0, False),
    "RDEEP": ("Deep res (ohm.m)", 0.2, 200.0, True),
    "RSHAL": ("Shallow res (ohm.m)", 0.2, 200.0, True),
    "RHOB": ("RHOB (g/cc)", 1.95, 2.95, False),
    "NPHI": ("NPHI (v/v)", 0.45, -0.15, False),
    "DT": ("DT (us/ft)", 140.0, 40.0, False),
    "PE": ("PE (b/e)", 0.0, 10.0, False),
    "SP": ("SP (mV)", -100.0, 50.0, False),
    "CALI": ("CALI (in)", 6.0, 16.0, False),
    # petrophysics results (not log families; shown in the petro tracks)
    "VSH": ("Vsh (v/v)", 0.0, 1.0, False),
    "PHIE": ("PHIE (v/v)", 0.5, 0.0, False),
    "BVW": ("BVW (v/v)", 0.5, 0.0, False),
    "SW": ("Sw (v/v)", 1.0, 0.0, False),
    "PERM": ("Perm (mD)", 0.01, 10000.0, True),
}

STEP = 0.1524  # metres (half-foot) composite sampling


def family_of(mnemonic, aliases=DEFAULT_ALIASES):
    m = mnemonic.upper()
    for fam, names in aliases.items():
        if m in names:
            return fam, names.index(m)
    return None, None


def normalise_units(family, unit, values):
    """Bring values to the family's standard unit. Returns (values, note)."""
    v = values
    u = (unit or "").upper()
    finite = v[np.isfinite(v)]
    med = float(np.median(finite)) if finite.size else np.nan
    note = ""
    if family == "NPHI":
        if "%" in u or "PU" in u or (np.isfinite(med) and med > 1.5):
            v, note = v / 100.0, "% -> v/v"
    elif family == "RHOB":
        if "KG" in u or (np.isfinite(med) and med > 100):
            v, note = v / 1000.0, "kg/m3 -> g/cc"
    elif family == "DT":
        if "US/M" in u.replace(" ", "") or (np.isfinite(med) and med > 250):
            v, note = v * 0.3048, "us/m -> us/ft"
    elif family == "CALI":
        if u in ("CM",) or (np.isfinite(med) and med > 30 and u not in ("IN",)):
            v, note = v / 2.54, "cm -> in"
        elif u in ("MM",):
            v, note = v / 25.4, "mm -> in"
    elif family in ("RDEEP", "RSHAL"):
        v = np.where(v > 0, v, np.nan)
    return v, note


def remove_flatlines(depth, values, min_len_m=3.0):
    """Blank stretches where a curve repeats the same value (stuck tool, cased hole, padding)."""
    v = np.array(values, float)
    n = len(v)
    if n < 3:
        return v
    same = np.zeros(n, bool)
    same[1:] = np.isfinite(v[1:]) & (v[1:] == v[:-1])
    i = 1
    while i < n:
        if same[i]:
            j = i
            while j < n and same[j]:
                j += 1
            if abs(depth[j - 1] - depth[i - 1]) >= min_len_m:
                v[i - 1:j] = np.nan
            i = j
        else:
            i += 1
    return v


def _resample(depth, values, grid):
    ok = np.isfinite(depth) & np.isfinite(values)
    if ok.sum() < 2:
        return np.full(grid.shape, np.nan)
    d, v = depth[ok], values[ok]
    order = np.argsort(d)
    d, v = d[order], v[order]
    d, idx = np.unique(d, return_index=True)
    v = v[idx]
    out = np.interp(grid, d, v, left=np.nan, right=np.nan)
    # do not bridge gaps bigger than 2 m
    step = np.diff(d)
    big = np.where(step > 2.0)[0]
    for i in big:
        out[(grid > d[i]) & (grid < d[i + 1])] = np.nan
    return out


def composite(las_files, aliases=DEFAULT_ALIASES, overrides=None):
    """Build one curve per family from every LAS file of a well.

    Ranking of candidates for a family: user override first, then alias priority, then
    the longest valid coverage (this is what drops repeat passes - they are short).
    Gaps in the best curve are filled from the next-best candidates.

    Returns (grid, {family: values}, {family: [source descriptions]}, candidates)
    """
    overrides = overrides or {}
    cands = {}
    tmin, tmax = np.inf, -np.inf
    for lf in las_files:
        for c in lf.curves:
            fam, prio = family_of(c.mnemonic, aliases)
            if fam is None or c.n_valid < 20:
                continue
            t, b = c.valid_range()
            tmin, tmax = min(tmin, t), max(tmax, b)
            cands.setdefault(fam, []).append((prio, -(b - t), c))
    if not cands:
        return np.array([]), {}, {}, {}
    grid = np.arange(np.floor(tmin / STEP) * STEP, tmax + STEP, STEP)
    out, src = {}, {}
    for fam, lst in cands.items():
        want = overrides.get(fam)

        def key(item):
            prio, negcov, c = item
            pinned = 0
            if want and (c.mnemonic == want.get("mnemonic") and _same_file(c.file, want.get("file"))):
                pinned = -1
            return (pinned, prio, negcov)

        lst.sort(key=key)
        acc = np.full(grid.shape, np.nan)
        used = []
        for prio, negcov, c in lst:
            vals, note = normalise_units(fam, c.unit, remove_flatlines(c.depth, c.values))
            r = _resample(c.depth, vals, grid)
            fill = np.isnan(acc) & np.isfinite(r)
            if fill.sum() < 10:
                continue
            acc[fill] = r[fill]
            t, b = c.valid_range()
            used.append(f"{c.mnemonic} [{_short(c.file)} col {c.index}] {t:.0f}-{b:.0f} m" + (f" ({note})" if note else ""))
        if np.isfinite(acc).any():
            out[fam], src[fam] = acc, used
    candidates = {f: [(c.mnemonic, _short(c.file), c.index, *c.valid_range()) for _, _, c in l] for f, l in cands.items()}
    return grid, out, src, candidates


def _short(path):
    return re.split(r"[\\/]", str(path))[-1]


def _same_file(a, b):
    return b is None or _short(a).lower() == _short(b).lower()
