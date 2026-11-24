"""Field reports (PLAN 22b): shows, DST / production tests, cores, MICP, mud gas, formation pressures.

Tables (CSV, or XLSX when openpyxl is installed) are read with column aliases into records; free text (well completion
reports pasted as text, or PDF text when pypdf is installed) goes through a keyword extractor whose hits are marked
"auto-extracted, check".  Everything per well is saved to <project>_petro/reports.json and a reports_summary.csv the map
hub can post (Shows, Tests, Cores, Contact_TVDSS ...).

    recs = read_table("cores.csv", "core")               # -> [{"well", "md", "porosity", "perm", "grain_density"}]
    hits = extract_from_text(text)                          # depth + show / test / core phrases
    g = fluid_gradients(tvdss, pressure)                    # two-leg fit: gradients, fluids, contact depth
    w = gas_ratios(C1, C2, C3, iC4, nC4, iC5, nC5)          # Haworth wetness / balance / character
    c = core_log_calibration(depth, phie, core_md, core_phi) # depth shift + regression (core-log cross-plot)
    m = fit_archie_m(phi, FF)                               # cementation exponent from SCAL formation factors
"""
import csv
import json
import os
import re

import numpy as np

from .store import petro_dir

ALIASES = {
    "well": ("well", "well_name", "wellname", "uwi", "api"),
    "md": ("md", "depth", "depth_m", "md_m", "sample_depth", "core_depth", "mid_depth"),
    "top": ("top", "top_md", "from", "depth_top", "top_m", "interval_top"),
    "base": ("base", "base_md", "to", "bottom", "depth_base", "base_m", "interval_base"),
    "tvdss": ("tvdss", "tvdss_m", "tvd_ss", "subsea"),
    "pressure": ("pressure", "pres", "formation_pressure", "pressure_mpa", "pressure_psi", "pressure_bar", "p"),
    "porosity": ("porosity", "phi", "core_porosity", "por", "he_porosity", "helium_porosity"),
    "perm": ("perm", "permeability", "kh", "kair", "k_air", "klinkenberg", "perm_md"),
    "grain_density": ("grain_density", "gd", "rhog", "grain_dens", "rho_grain"),
    "description": ("description", "desc", "show", "remarks", "comment", "lithology"),
    "fluid": ("fluid", "recovery", "result", "fluid_type"),
    "rate": ("rate", "flow_rate", "gas_rate", "oil_rate"),
    "type": ("type", "test_type", "kind"),
    "pc": ("pc", "capillary_pressure", "pc_psi", "pressure_psia"),
    "sw": ("sw", "water_saturation", "swn", "hg_sat_inv"),
}
KIND_FIELDS = {"show": ("well", "top", "base", "description"), "test": ("well", "top", "base", "type", "fluid", "rate"),
               "core": ("well", "md", "porosity", "perm", "grain_density"), "micp": ("well", "md", "pc", "sw"),
               "pressure": ("well", "md", "tvdss", "pressure")}
PRESSURE_UNITS = {"psi": 0.00689475729, "kpa": 0.001, "bar": 0.1, "mpa": 1.0}


def _map_columns(cols):
    low = {re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_"): c for c in cols}
    out = {}
    for key, names in ALIASES.items():
        for n in names:
            if n in low:
                out[key] = low[n]
                break
    return out


def _f(v):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def read_rows(path):
    if path.lower().endswith((".xlsx", ".xls")):
        import pandas as pd  # openpyxl needed for .xlsx
        df = pd.read_excel(path)
        return [dict(zip(df.columns, r)) for r in df.itertuples(index=False)], list(df.columns)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        return rows, list(rd.fieldnames or [])


def read_table(path, kind, well=None, depth_unit="m", pressure_unit=None):
    """Records of one kind (show / test / core / micp / pressure).  well: default well name when there is no column.
    Depths in feet are converted (depth_unit 'ft'); pressures to MPa (unit from the column name or pressure_unit)."""
    rows, cols = read_rows(path)
    cm = _map_columns(cols)
    zf = 0.3048 if depth_unit == "ft" else 1.0
    pu = pressure_unit
    if pu is None and "pressure" in cm:
        nm = cm["pressure"].lower()
        pu = next((u for u in PRESSURE_UNITS if u in nm), "mpa")
    out = []
    for r in rows:
        rec = {"kind": kind, "source": os.path.basename(path)}
        for key in KIND_FIELDS[kind]:
            col = cm.get(key)
            v = r.get(col) if col else None
            if key in ("md", "top", "base", "tvdss"):
                v = _f(v)
                v = v * zf if v is not None else None
            elif key in ("porosity", "perm", "grain_density", "pc", "sw", "rate"):
                v = _f(v)
            elif key == "pressure":
                v = _f(v)
                v = v * PRESSURE_UNITS.get(pu or "mpa", 1.0) if v is not None else None
            rec[key] = v
        if not rec.get("well"):
            rec["well"] = well
        if kind == "core" and rec.get("porosity") is not None and rec["porosity"] > 1.0:
            rec["porosity"] /= 100.0  # percent
        out.append(rec)
    return out


# ------------------------------------------------------------------------------------------- free text
_DEPTH = r"(\d{1,2}[,.]?\d{3}(?:\.\d+)?|\d{3,5}(?:\.\d+)?)\s*(m|mmd|m\s*md|mbrt|mrt|mkb|ft|feet|')"
_RANGE = _DEPTH[:-1] + r")?\s*(?:-|to|–)\s*" + _DEPTH
TEXT_PATTERNS = [
    ("gas show", r"gas\s+show|trace\s+gas|gas\s+peak|elevated\s+gas|background\s+gas"),
    ("oil show", r"oil\s+show|fluorescen|cut\s+fluor|oil\s+stain|bleeding\s+oil|live\s+oil|dead\s+oil"),
    ("test", r"\bdst\b|drill\s*stem\s+test|production\s+test|flowed|recovered|\bmdt\b|\brft\b|well\s+test"),
    ("core", r"\bcore\s*(?:no\.?|#)?\s*\d|cored\s+interval|conventional\s+core|sidewall\s+core|\bswc\b"),
    ("pressure", r"formation\s+pressure|pressure\s+point|pretest|\bpsia\b"),
    ("water", r"water\s+bearing|wet\b|formation\s+water|salt\s+water"),
]


def _to_m(v, unit):
    x = float(v.replace(",", "")) if v else None
    if x is None:
        return None
    return x * 0.3048 if unit and unit.strip().lower() in ("ft", "feet", "'") else x


def extract_from_text(text, well=None):
    """Sentences mentioning shows / tests / cores / pressures with a depth (or depth range) -> records flagged
    'auto-extracted, check'."""
    out = []
    for sent in re.split(r"(?<=[.;\n])\s+(?=[A-Z])|\n+", text):
        s = sent.strip()
        if not s:
            continue
        low = s.lower()
        kinds = [k for k, rx in TEXT_PATTERNS if re.search(rx, low)]
        if not kinds:
            continue
        m = re.search(_RANGE, low)
        top = base = None
        if m:
            top = _to_m(m.group(1), m.group(2) or m.group(4))
            base = _to_m(m.group(3), m.group(4))
        else:
            m = re.search(_DEPTH, low)
            if m:
                top = base = _to_m(m.group(1), m.group(2))
        if top is None:
            continue
        out.append({"kind": kinds[0], "kinds": kinds, "well": well, "top": top, "base": base, "text": s[:300],
                    "flag": "auto-extracted, check"})
    return out


def pdf_text(path):
    """Text of a PDF when pypdf is installed (optional); None otherwise."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)


# ------------------------------------------------------------------------------------------- pressures, gas, cores
FLUID_GRADIENTS = [("gas", 0.0, 0.0045), ("oil", 0.0045, 0.0092), ("water", 0.0092, 0.013)]  # MPa/m


def fluid_of(gradient):
    return next((n for n, lo, hi in FLUID_GRADIENTS if lo <= gradient < hi), "unknown")


def fluid_gradients(tvdss, pressure, min_points=3):
    """Best two-leg straight-line fit of pressure vs TVDSS (exhaustive breakpoint search); single leg when two legs do
    not reduce the misfit by 30 %.  Returns {"legs": [(top, base, gradient MPa/m, intercept, fluid)], "contact" (TVDSS
    where the two lines cross) or None, "rms"}."""
    z, p = np.asarray(tvdss, float), np.asarray(pressure, float)
    ok = np.isfinite(z) & np.isfinite(p)
    z, p = z[ok], p[ok]
    o = np.argsort(z)
    z, p = z[o], p[o]
    n = len(z)

    def fit(a, b):
        A = np.column_stack([z[a:b], np.ones(b - a)])
        c, *_ = np.linalg.lstsq(A, p[a:b], rcond=None)
        r = p[a:b] - A @ c
        return c, float((r ** 2).sum())

    c1, s1 = fit(0, n)
    best = None
    for k in range(min_points, n - min_points + 1):
        ca, sa = fit(0, k)
        cb, sb = fit(k, n)
        if best is None or sa + sb < best[0]:
            best = (sa + sb, k, ca, cb)
    if best is not None:
        g_a, g_b = best[2][0], best[3][0]
    if best is None or best[0] > 0.7 * s1 or abs(g_a - g_b) < 0.0015:  # no real improvement / same fluid
        return {"legs": [(float(z[0]), float(z[-1]), float(c1[0]), float(c1[1]), fluid_of(c1[0]))], "contact": None,
                "rms": float(np.sqrt(s1 / n))}
    _, k, ca, cb = best
    contact = (cb[1] - ca[1]) / (ca[0] - cb[0]) if abs(ca[0] - cb[0]) > 1e-12 else None
    return {"legs": [(float(z[0]), float(z[k - 1]), float(ca[0]), float(ca[1]), fluid_of(ca[0])),
                     (float(z[k]), float(z[-1]), float(cb[0]), float(cb[1]), fluid_of(cb[0]))],
            "contact": float(contact) if contact is not None else None, "rms": float(np.sqrt(best[0] / n))}


def gas_ratios(c1, c2, c3, ic4=0.0, nc4=0.0, ic5=0.0, nc5=0.0):
    """Haworth et al. (1985): wetness Wh %, balance Bh, character Ch and the zone they indicate."""
    c1, c2, c3, c4, c5 = (np.asarray(v, float) for v in (c1, c2, c3, np.asarray(ic4, float) + nc4, np.asarray(ic5, float) + nc5))
    tot = c1 + c2 + c3 + c4 + c5
    with np.errstate(divide="ignore", invalid="ignore"):
        wh = 100.0 * (c2 + c3 + c4 + c5) / tot
        bh = (c1 + c2) / (c3 + c4 + c5)
        ch = (c4 + c5) / c3
    zone = np.where(~np.isfinite(wh) | (tot <= 0), "no gas",
                    np.where(wh < 0.5, "very dry gas",
                             np.where(wh < 17.5, np.where(bh > wh, "gas", "gas / condensate"),
                                      np.where(wh < 40, np.where(bh < wh, "oil", "gas / oil"), "residual oil"))))
    return {"Wh": wh, "Bh": bh, "Ch": ch, "zone": zone}


def read_mudgas(path):
    """Gas chromatograph CSV (Depth, C1, C2, C3, IC4, NC4, IC5, NC5, Tgas; an optional units row) -> dict of arrays."""
    rows, cols = read_rows(path)
    if rows and not _f(rows[0].get(cols[0])):
        rows = rows[1:]  # units row
    out = {c.upper(): np.array([_f(r.get(c)) if _f(r.get(c)) is not None else np.nan for r in rows]) for c in cols}
    return out


def core_log_calibration(depth, log_phi, core_md, core_phi, max_shift=3.0, step=0.1524):
    """Depth shift of the core (within +/- max_shift) that best correlates core porosity with the log, then a linear
    fit core = a + b log.  Returns {"shift", "r", "a", "b", "n", "pairs" (log, core)}."""
    depth, log_phi = np.asarray(depth, float), np.asarray(log_phi, float)
    cm, cp = np.asarray(core_md, float), np.asarray(core_phi, float)
    ok = np.isfinite(log_phi)
    best = None
    for sh in np.arange(-max_shift, max_shift + 1e-9, step):
        lv = np.interp(cm + sh, depth[ok], log_phi[ok], left=np.nan, right=np.nan)
        m = np.isfinite(lv) & np.isfinite(cp)
        if m.sum() < 5:
            continue
        r = float(np.corrcoef(lv[m], cp[m])[0, 1])
        if best is None or r > best[0]:
            best = (r, sh, lv, m)
    if best is None:
        return None
    r, sh, lv, m = best
    b, a = np.polyfit(lv[m], cp[m], 1)
    return {"shift": float(sh), "r": r, "a": float(a), "b": float(b), "n": int(m.sum()),
            "pairs": np.column_stack([lv[m], cp[m]])}


def fit_archie_m(phi, ff, free_a=False):
    """Cementation exponent from formation factors: FF = a / phi^m (a = 1 unless free_a).  Returns (m, a)."""
    phi, ff = np.asarray(phi, float), np.asarray(ff, float)
    ok = (phi > 0) & (ff > 0)
    x, y = np.log10(phi[ok]), np.log10(ff[ok])
    if free_a:
        m_neg, la = np.polyfit(x, y, 1)
        return float(-m_neg), float(10 ** la)
    return float(-np.sum(x * y) / np.sum(x * x)), 1.0


def fit_archie_n(sw, ri):
    """Saturation exponent from resistivity index RI = Sw^-n (through the origin in log-log)."""
    sw, ri = np.asarray(sw, float), np.asarray(ri, float)
    ok = (sw > 0) & (ri > 0)
    x, y = np.log10(sw[ok]), np.log10(ri[ok])
    return float(-np.sum(x * y) / np.sum(x * x))


# ------------------------------------------------------------------------------------------- store
def save_reports(project, records):
    """Merge records into <project>_petro/reports.json (per well) and rewrite reports_summary.csv."""
    d = petro_dir(project, create=True)
    path = os.path.join(d, "reports.json")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    for r in records:
        w = r.get("well") or "_unassigned"
        data.setdefault(w, []).append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()})
    with open(path + ".tmp", "w") as f:
        json.dump(data, f, indent=1)
    os.replace(path + ".tmp", path)
    rows = []
    for w, recs in data.items():
        if w == "_unassigned":
            continue
        kinds = [r.get("kind") for r in recs]
        rows.append({"Well": w, "Shows": sum(k in ("show", "gas show", "oil show") for k in kinds),
                     "Tests": sum(k == "test" for k in kinds), "Cores": sum(k == "core" for k in kinds),
                     "Pressures": sum(k == "pressure" for k in kinds),
                     "Contact_TVDSS": next((r.get("contact") for r in recs if r.get("kind") == "contact"), None)})
    with open(os.path.join(d, "reports_summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["Well", "Shows", "Tests", "Cores", "Pressures", "Contact_TVDSS"])
        wr.writeheader()
        for r in rows:
            wr.writerow({k: ("" if v is None else v) for k, v in r.items()})
    return path


def load_reports(project, well=None):
    p = os.path.join(petro_dir(project) or "", "reports.json")
    if not os.path.exists(p):
        return {} if well is None else []
    with open(p) as f:
        data = json.load(f)
    return data if well is None else data.get(well, [])
