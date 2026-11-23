"""Engineering + production data for analytics: CSV import with column mapping presets and unit detection,
completion features, well spacing / parent-child, monthly production features (cums, IP, GOR, water cut),
Arps decline + EUR, FracFocus estimates, lease allocation, QC flags.  Everything ends up as AttributeTable
columns keyed by normalised API / well name.
"""
import csv
import json

import numpy as np

from .table import AttributeTable, normalise_id

# canonical column -> header names seen in common exports (lower-case, non-alphanumerics removed)
ALIASES = {
    "api": ["api", "api10", "api12", "api14", "uwi", "apinumber", "apiuwi", "wellapi", "api_no"],
    "well": ["well", "wellname", "name", "leasename", "well_name"],
    "x": ["x", "easting", "surfacex", "midpointx", "bhx"],
    "y": ["y", "northing", "surfacey", "midpointy", "bhy"],
    "completion_date": ["completiondate", "compdate", "jobstartdate", "fracdate", "completed"],
    "first_prod": ["firstproddate", "firstproductiondate", "firstprod", "fpdate"],
    "lateral_length": ["laterallength", "perfinterval", "lateral", "lateralft", "lateral_length_ft", "ll"],
    "stages": ["stages", "stagecount", "numstages", "fracstages", "totalstages"],
    "clusters": ["clusters", "clusterspacing", "clustersperstage"],
    "proppant": ["proppant", "totalproppant", "proppantlbs", "proppantlb", "proppantmass", "sandlbs", "proppant_t"],
    "fluid": ["fluid", "totalfluid", "fluidbbl", "totalfluidbbl", "fluidvolume", "fluid_m3", "basewatervolume"],
    "fluid_system": ["fluidsystem", "fluidtype"],
    "landing_zone": ["landingzone", "targetformation", "formation", "zone"],
    "month": ["month", "date", "proddate", "productiondate", "reportdate"],
    "oil": ["oil", "oilbbl", "liquid", "oilprod", "monthlyoil", "oil_m3"],
    "gas": ["gas", "gasmcf", "monthlygas", "gasprod", "gas_e3m3"],
    "water": ["water", "waterbbl", "monthlywater", "waterprod", "water_m3"],
    "days": ["days", "daysonproduction", "producingdays", "dayson", "proddays"],
}

PRESETS = {  # explicit header choices for known exports (tried before the aliases)
    "Enverus": {"api": "API14", "lateral_length": "LateralLength_FT", "proppant": "Proppant_LBS",
                "fluid": "TotalFluidPumped_BBL", "stages": "FracStages", "first_prod": "FirstProdDate"},
    "IHS": {"api": "UWI", "lateral_length": "Perforated Interval", "proppant": "Total Proppant (lbs)",
            "fluid": "Total Fluid (bbl)"},
    "Texas RRC": {"api": "API_NO", "well": "LEASE_NAME", "month": "CYCLE_YEAR_MONTH", "oil": "LEASE_OIL_PROD_VOL",
                  "gas": "LEASE_GAS_PROD_VOL"},
    "FracFocus": {"api": "APINumber", "completion_date": "JobStartDate", "fluid": "TotalBaseWaterVolume"},
    "NDIC": {"api": "API_WELLNO", "month": "ReportDate", "oil": "Oil", "gas": "Gas", "water": "Wtr", "days": "Days"},
    "Colorado ECMC": {"api": "API", "month": "Report Month", "oil": "Oil Produced", "gas": "Gas Produced",
                      "water": "Water Produced", "days": "Days Produced"},
    "NM OCD": {"api": "API", "month": "Production Month", "oil": "Oil (BBLS)", "gas": "Gas (MCF)",
               "water": "Water (BBLS)", "days": "Days Produced"},
    "OCC": {"api": "API Number", "month": "Reporting Month", "oil": "Oil Volume", "gas": "Gas Volume"},
}

LB_PER_TONNE = 2204.62
BBL_PER_M3 = 6.28981
FT_PER_M = 3.28084
MCF_PER_E3M3 = 35.3147


def _key(s):
    return "".join(c for c in str(s).lower() if c.isalnum())


def auto_mapping(headers, preset=None):
    """{canonical: header} from a preset (if its headers exist) then aliases."""
    hk = {_key(h): h for h in headers}
    out = {}
    if preset and preset in PRESETS:
        for k, h in PRESETS[preset].items():
            if _key(h) in hk:
                out[k] = hk[_key(h)]
    for k, names in ALIASES.items():
        if k in out:
            continue
        for n in names:
            if _key(n) in hk:
                out[k] = hk[_key(n)]
                break
    used = set(out.values())
    for k, names in ALIASES.items():  # second chance: header starts with an alias (TotalFluid_M3, Oil_BBL, ...)
        if k in out:
            continue
        for n in sorted(names, key=len, reverse=True):
            nk = _key(n)
            hit = next((h for kk, h in hk.items() if len(nk) >= 3 and kk.startswith(nk) and h not in used), None)
            if hit:
                out[k] = hit
                used.add(hit)
                break
    return out


def save_mapping(mapping, path):
    with open(path, "w") as fh:
        json.dump(mapping, fh, indent=1)


def load_mapping(path):
    with open(path) as fh:
        return json.load(fh)


def _num(v):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return np.nan


def _date(v):
    """'YYYY-MM-DD', 'MM/DD/YYYY', 'YYYYMM' -> months since 1900-01 (float) or NaN."""
    s = str(v or "").strip()
    if not s:
        return np.nan
    try:
        if "-" in s:
            p = s.split(" ")[0].split("-")
            return (int(p[0]) - 1900) * 12 + int(p[1]) - 1 + (int(p[2]) - 1) / 30.0 if len(p) > 2 else \
                (int(p[0]) - 1900) * 12 + int(p[1]) - 1
        if "/" in s:
            p = s.split(" ")[0].split("/")
            return (int(p[2]) - 1900) * 12 + int(p[0]) - 1 + (int(p[1]) - 1) / 30.0
        if len(s) == 6 and s.isdigit():
            return (int(s[:4]) - 1900) * 12 + int(s[4:]) - 1
    except (ValueError, IndexError):
        return np.nan
    return np.nan


def detect_units(header, values):
    """Unit of a column from its header (and magnitude as a fallback): 'lb'|'t', 'bbl'|'m3'|'gal', 'ft'|'m',
    'mcf'|'e3m3'."""
    h = header.lower()
    for u, keys in (("t", ["_t", "(t)", "tonne", "tons"]), ("lb", ["lb", "lbs"]), ("m3", ["m3", "m^3"]),
                    ("gal", ["gal"]), ("bbl", ["bbl"]), ("e3m3", ["e3m3", "103m3"]), ("mcf", ["mcf"]),
                    ("ft", ["ft", "feet", "(ft)"]), ("m", ["_m", "(m)", "metre", "meter"])):
        if any(k in h for k in keys):
            return u
    return None


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


# ====================================================================== completions
def completion_table(rows, mapping=None, preset=None, units=None):
    """Completion rows -> AttributeTable with lateral length (ft), stages, stage spacing (ft), proppant (lb),
    proppant/ft, fluid (bbl), fluid/ft, completion month, first production month, x/y when given.
    units: {canonical: unit} overrides; otherwise detected from the headers."""
    if not rows:
        return AttributeTable()
    mapping = mapping or auto_mapping(rows[0].keys(), preset)
    units = dict(units or {})
    for k in ("lateral_length", "proppant", "fluid"):
        if k in mapping and k not in units:
            units[k] = detect_units(mapping[k], None)
    idk = "api" if "api" in mapping else "well"
    t = AttributeTable([r.get(mapping[idk], "") for r in rows])
    col = lambda k: np.array([_num(r.get(mapping[k])) for r in rows]) if k in mapping else np.full(len(rows), np.nan)  # noqa
    ll = col("lateral_length") * (FT_PER_M if units.get("lateral_length") == "m" else 1.0)
    prop = col("proppant") * (LB_PER_TONNE if units.get("proppant") == "t" else 1.0)
    fl = col("fluid")
    if units.get("fluid") == "m3":
        fl = fl * BBL_PER_M3
    elif units.get("fluid") == "gal":
        fl = fl / 42.0
    st = col("stages")
    t.add("lateral_ft", ll).add("stages", st)
    with np.errstate(invalid="ignore", divide="ignore"):
        t.add("stage_spacing_ft", ll / st).add("proppant_lb", prop).add("proppant_lb_ft", prop / ll)
        t.add("fluid_bbl", fl).add("fluid_bbl_ft", fl / ll)
    if "clusters" in mapping:
        t.add("clusters", col("clusters"))
    for k in ("completion_date", "first_prod"):
        if k in mapping:
            t.add(k + "_month", [_date(r.get(mapping[k])) for r in rows])
    if "x" in mapping and "y" in mapping:
        t.xy = np.column_stack([col("x"), col("y")])
    t.meta["units"] = units
    t.meta["mapping"] = mapping
    return t


def spacing_features(t, same_zone=None, radius_ft=1500.0, m_per_unit=1.0):
    """Nearest-neighbour well spacing (map units) and parent/child flag: child = a neighbour within radius
    started producing earlier (first_prod_month).  same_zone: optional array of zone labels."""
    xy = t.xy
    n = len(t)
    nn = np.full(n, np.nan)
    child = np.zeros(n)
    fp = t.cols.get("first_prod_month", np.full(n, np.nan))
    r = radius_ft / FT_PER_M / m_per_unit
    for i in range(n):
        d = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
        d[i] = np.inf
        if same_zone is not None:
            d[np.asarray(same_zone) != same_zone[i]] = np.inf
        if np.isfinite(d).any():
            nn[i] = d.min()
            close = d <= r
            child[i] = float(np.any(close & (fp < fp[i])))
    t.add("nn_spacing", nn).add("child_well", child)
    return t


def qc_flags(t, max_prop_lb_ft=5000.0, max_fluid_bbl_ft=200.0):
    """Unit-error / outlier flags as a column (1 = suspicious)."""
    flag = np.zeros(len(t))
    for k, lim in (("proppant_lb_ft", max_prop_lb_ft), ("fluid_bbl_ft", max_fluid_bbl_ft)):
        v = t.cols.get(k)
        if v is not None:
            flag = np.maximum(flag, (v > lim).astype(float))
    t.add("qc_flag", flag)
    return t


# ====================================================================== production
def monthly_series(rows, mapping=None, preset=None):
    """{id: dict(month (sorted months since 1900), oil, gas, water, days)} from monthly production rows."""
    mapping = mapping or auto_mapping(rows[0].keys(), preset) if rows else {}
    idk = "api" if "api" in mapping else "well"
    by = {}
    for r in rows:
        k = r.get(mapping.get(idk), "")
        m = _date(r.get(mapping.get("month")))
        if not np.isfinite(m):
            continue
        d = by.setdefault(k, {})
        e = d.setdefault(int(m), [0.0, 0.0, 0.0, np.nan])
        for j, c in enumerate(("oil", "gas", "water")):
            if c in mapping:
                v = _num(r.get(mapping[c]))
                if np.isfinite(v):
                    e[j] += v
        if "days" in mapping:
            e[3] = _num(r.get(mapping["days"]))
    out = {}
    for k, d in by.items():
        ms = np.array(sorted(d))
        a = np.array([d[m] for m in ms], float)
        out[k] = {"month": ms, "oil": a[:, 0], "gas": a[:, 1], "water": a[:, 2], "days": a[:, 3]}
    return out


def arps_fit(q, t=None):
    """Hyperbolic Arps q(t) = qi / (1 + b Di t)^(1/b), t in months from peak.  Grid over b, linearised fit.
    Returns dict(qi, di (1/month), b, r2) or None."""
    q = np.asarray(q, float)
    t = np.arange(len(q), dtype=float) if t is None else np.asarray(t, float)
    ok = np.isfinite(q) & (q > 0)
    if ok.sum() < 4:
        return None
    q, t = q[ok], t[ok]
    best = None
    lq = np.log(q)
    for b in np.r_[0.0, np.linspace(0.05, 2.0, 40)]:
        if b == 0:
            A = np.column_stack([np.ones_like(t), -t])
            c = np.linalg.lstsq(A, lq, rcond=None)[0]
            qi, di = np.exp(c[0]), c[1]
        else:
            y = q ** (-b)
            A = np.column_stack([np.ones_like(t), t])
            c = np.linalg.lstsq(A, y, rcond=None)[0]
            if c[0] <= 0:
                continue
            qi = c[0] ** (-1.0 / b)
            di = c[1] / (b * c[0])
        if di <= 0:
            continue
        pred = arps_rate(qi, di, b, t)
        sse = float(np.sum((np.log(np.maximum(pred, 1e-12)) - lq) ** 2))
        if best is None or sse < best[0]:
            best = (sse, qi, di, b)
    if best is None:
        return None
    sse, qi, di, b = best
    r2 = 1 - sse / max(np.sum((lq - lq.mean()) ** 2), 1e-12)
    return {"qi": float(qi), "di": float(di), "b": float(b), "r2": float(r2)}


def arps_rate(qi, di, b, t):
    t = np.asarray(t, float)
    return qi * np.exp(-di * t) if b == 0 else qi / (1 + b * di * t) ** (1.0 / b)


def eur(series, fit, months_total=360, q_limit=0.0):
    """Produced to date + Arps forecast to months_total (or until the rate drops below q_limit / month)."""
    if fit is None:
        return np.nan
    n = len(series)
    t = np.arange(n, months_total)
    q = arps_rate(fit["qi"], fit["di"], fit["b"], t)
    if q_limit > 0:
        q = q[q >= q_limit]
    return float(np.nansum(series) + q.sum())


def production_features(series, lateral_ft=None, boe_ratio=6.0, skip_partial_first=False, months=(3, 6, 9, 12, 24),
                        fit_arps=True):
    """{id: series} -> AttributeTable of cums per stream + BOE, per-1000-ft versions, peak month, IP30/IP90
    (bbl/d), GOR, water cut, months on production, Arps (qi, di, b, r2) and EUR (BOE).
    lateral_ft: {id: ft} for normalisation.  Wells with fewer than N months get NaN for N-month cums."""
    ids = list(series)
    t = AttributeTable(ids)
    cols = {}

    def put(k, i, v):
        cols.setdefault(k, np.full(len(ids), np.nan))[i] = v

    for i, k in enumerate(ids):
        s = series[k]
        oil, gas, wat, days = s["oil"], s["gas"], s["water"], s["days"]
        boe = oil + gas / boe_ratio
        prod = np.nonzero(boe > 0)[0]
        if not len(prod):
            continue
        f = prod[0]
        if skip_partial_first and np.isfinite(days[f]) and days[f] < 15 and f + 1 < len(boe):
            f += 1
        o, g, w, b = oil[f:], gas[f:], wat[f:], boe[f:]
        dd = days[f:]
        put("months_on", i, len(b))
        for n in months:
            if len(b) >= n:
                put(f"cum{n}_oil", i, o[:n].sum())
                put(f"cum{n}_gas", i, g[:n].sum())
                put(f"cum{n}_water", i, w[:n].sum())
                put(f"cum{n}_boe", i, b[:n].sum())
        put("cum_boe", i, b.sum())
        put("peak_month", i, int(np.argmax(b)) + 1)
        put("peak_boe", i, b.max())
        d0 = dd[0] if np.isfinite(dd[0]) and dd[0] > 0 else 30.4
        put("ip30_boe_d", i, b[0] / d0)
        n3 = min(3, len(b))
        dsum = np.nansum(np.where(np.isfinite(dd[:n3]) & (dd[:n3] > 0), dd[:n3], 30.4))
        put("ip90_boe_d", i, b[:n3].sum() / dsum)
        with np.errstate(invalid="ignore", divide="ignore"):
            put("gor_scf_bbl", i, 1000.0 * g.sum() / o.sum() if o.sum() > 0 else np.nan)
            put("water_cut", i, w.sum() / (w.sum() + o.sum()) if (w.sum() + o.sum()) > 0 else np.nan)
        if fit_arps:
            pk = int(np.argmax(b))
            ft = arps_fit(b[pk:])
            if ft:
                put("arps_qi", i, ft["qi"])
                put("arps_di", i, ft["di"])
                put("arps_b", i, ft["b"])
                put("arps_r2", i, ft["r2"])
                put("eur_boe", i, eur(b, ft, q_limit=0.0) if len(b) else np.nan)
        if lateral_ft and np.isfinite(lateral_ft.get(k, np.nan)) and lateral_ft.get(k, 0) > 0:
            per = 1000.0 / lateral_ft[k]
            for n in months:
                if len(b) >= n:
                    put(f"cum{n}_boe_per1000ft", i, b[:n].sum() * per)
    for k, v in cols.items():
        t.add(k, v)
    return t


# ====================================================================== helpers for specific sources
def fracfocus_estimates(total_water_gal, water_mass_pct, proppant_mass_pct):
    """FracFocus: total water volume (gal) + ingredient mass % -> (water bbl, proppant lb).  Estimate only:
    total fluid mass = water mass / water %, proppant mass = total x proppant %."""
    water_lb = np.asarray(total_water_gal, float) * 8.34
    total_lb = water_lb / (np.asarray(water_mass_pct, float) / 100.0)
    return np.asarray(total_water_gal, float) / 42.0, total_lb * np.asarray(proppant_mass_pct, float) / 100.0


def allocate_lease(lease_volume, wells, method="equal", weights=None, days=None):
    """Texas RRC style: lease volumes are reported per lease - allocate to wells 'equal', 'test' (weights =
    well tests) or 'days' (days on).  Returns {well: volume} (flag these as allocated)."""
    n = len(wells)
    if method == "test" and weights is not None:
        w = np.asarray(weights, float)
    elif method == "days" and days is not None:
        w = np.asarray(days, float)
    else:
        w = np.ones(n)
    w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
    return {wl: float(lease_volume * wi) for wl, wi in zip(wells, w)}


def join_on_api(*tables):
    """Left-join tables on normalised API / well name (first table defines the rows)."""
    out = tables[0]
    for t in tables[1:]:
        out = out.join(t)
    return out


__all__ = ["auto_mapping", "completion_table", "spacing_features", "qc_flags", "monthly_series",
           "production_features", "arps_fit", "arps_rate", "eur", "fracfocus_estimates", "allocate_lease",
           "join_on_api", "normalise_id", "PRESETS", "save_mapping", "load_mapping", "read_csv_rows"]
