"""Prospect risking + volumetrics, Rose & Associates style (PLAN 28).  Pure numpy.

28a  GRV from a top-reservoir depth grid down to a contact (spill / fault-seal / top-seal limited, item 27) with a
     constant thickness or a base grid, area-depth and area-thickness curves; Monte-Carlo volumetrics with lognormal
     (P90 / P10) or user distributions -> P90 / P50 / P10, mean and Swanson's mean.
28b  Chance of geological success = product of factors (source / charge, reservoir, trap, seal, timing) prefilled from
     the evidence of items 18-27; DHI revision of Pg (Bayes with a bounded likelihood ratio, monotonic in the 25d
     DHI index); dependent prospects (shared factors) in a portfolio; CSV export.

Conventions: "P90" is the low case (90 % chance of more), i.e. the 10th percentile of the outcomes.
"""
import csv
import math

import numpy as np

Z90 = 1.2815515655446004
FACTORS = ("source", "reservoir", "trap", "seal", "timing")
M3_TO_BBL = 6.289811
M3_TO_SCF = 35.314667


# ====================================================================== 28a
def lognormal_from_p90_p10(p90, p10):
    """(mu, sigma) of ln X for a lognormal with the given low (P90) and high (P10) values."""
    if p90 <= 0 or p10 <= p90:
        raise ValueError("need 0 < P90 < P10")
    mu = 0.5 * (math.log(p90) + math.log(p10))
    sigma = (math.log(p10) - math.log(p90)) / (2 * Z90)
    return mu, sigma


def lognormal_stats(p90, p10):
    mu, s = lognormal_from_p90_p10(p90, p10)
    return {"p50": math.exp(mu), "mean": math.exp(mu + 0.5 * s * s), "mu": mu, "sigma": s}


def sample(dist, n, rng):
    """dist: number (constant) | ("lognormal", p90, p10) | ("normal", mean, sd[, lo, hi]) | ("uniform", lo, hi) |
    ("triangular", lo, mode, hi) | array of samples (resampled)."""
    if np.ndim(dist) == 0 and not isinstance(dist, (tuple, list)):
        return np.full(n, float(dist))
    if isinstance(dist, np.ndarray):
        return rng.choice(dist, n)
    kind = dist[0]
    if kind == "lognormal":
        mu, s = lognormal_from_p90_p10(dist[1], dist[2])
        return np.exp(rng.normal(mu, s, n))
    if kind == "normal":
        x = rng.normal(dist[1], dist[2], n)
        lo = dist[3] if len(dist) > 3 else -np.inf
        hi = dist[4] if len(dist) > 4 else np.inf
        return np.clip(x, lo, hi)
    if kind == "uniform":
        return rng.uniform(dist[1], dist[2], n)
    if kind == "triangular":
        return rng.triangular(dist[1], dist[2], dist[3], n)
    raise ValueError(f"unknown distribution {dist!r}")


def pvalues(x):
    """{p90, p50, p10, mean, swanson} of outcomes (P90 = 10th percentile)."""
    x = np.asarray(x, float)
    p90, p50, p10 = np.percentile(x, [10, 50, 90])
    return {"p90": float(p90), "p50": float(p50), "p10": float(p10), "mean": float(np.mean(x)),
            "swanson": float(0.3 * p90 + 0.4 * p50 + 0.3 * p10)}


def grv(top, contact, cell_area, thickness=None, base=None, mask=None):
    """Gross rock volume (m3) between a top-reservoir depth grid and the contact depth, limited by a base (grid) or a
    constant thickness (top + thickness); mask restricts to the prospect polygon / closure."""
    z = np.asarray(top, float)
    if base is None:
        base = z + (np.inf if thickness is None else float(thickness))
    b = np.minimum(np.asarray(base, float), float(contact))
    h = np.clip(b - z, 0, None)
    h = np.where(np.isfinite(h), h, 0.0)
    if mask is not None:
        h = h * np.asarray(mask, bool)
    return float(np.sum(h) * cell_area)


def area_depth(top, cell_area, levels=None, mask=None):
    """Area-depth curve: (levels, area shallower than each level)."""
    z = np.asarray(top, float)
    if mask is not None:
        z = np.where(mask, z, np.nan)
    fin = np.isfinite(z)
    levels = np.linspace(np.nanmin(z), np.nanmax(z), 50) if levels is None else np.asarray(levels, float)
    return levels, np.array([np.sum(fin & (z <= L)) * cell_area for L in levels])


def grv_from_area_depth(levels, areas, contact, thickness=None):
    """GRV by integrating the area-depth curve to the contact (thick-reservoir case) or with a constant thickness
    (area(z) - area(z - h) integrated) - the classic hand method, for checking."""
    lv, ar = np.asarray(levels, float), np.asarray(areas, float)
    sel = lv <= contact
    x = np.r_[lv[sel], contact]
    a = np.r_[ar[sel], np.interp(contact, lv, ar)]
    if thickness is None:
        return float(np.trapezoid(a, x) if hasattr(np, "trapezoid") else np.trapz(a, x))
    a2 = np.interp(x - thickness, lv, ar, left=0.0)
    d = a - a2
    return float(np.trapezoid(d, x) if hasattr(np, "trapezoid") else np.trapz(d, x))


def volumetrics(n=20000, grv_m3=None, ntg=0.7, phi=0.2, sw=0.3, fvf=1.2, rf=0.3, fluid="oil", seed=0, grv_scale=None):
    """Monte-Carlo in-place and recoverable volumes.  grv_m3: number / distribution (m3); grv_scale: optional
    multiplier distribution (e.g. ("lognormal", 0.8, 1.25) for map uncertainty).  fvf: Bo (rm3/sm3) for oil, Bg
    (rm3/sm3, ~0.003-0.01) for gas.  Returns {"inplace", "recoverable"} each with pvalues (sm3 and field units)."""
    rng = np.random.default_rng(seed)
    g = sample(grv_m3, n, rng)
    if grv_scale is not None:
        g = g * sample(grv_scale, n, rng)
    hc = g * np.clip(sample(ntg, n, rng), 0, 1) * np.clip(sample(phi, n, rng), 0, 0.5) * \
        (1 - np.clip(sample(sw, n, rng), 0, 1)) / np.maximum(sample(fvf, n, rng), 1e-6)
    rec = hc * np.clip(sample(rf, n, rng), 0, 1)
    f = M3_TO_BBL if fluid == "oil" else M3_TO_SCF
    unit = "bbl" if fluid == "oil" else "scf"
    out = {"inplace_sm3": pvalues(hc), "recoverable_sm3": pvalues(rec), "unit": unit,
           "inplace": {k: v * f for k, v in pvalues(hc).items()}, "recoverable": {k: v * f for k, v in pvalues(rec).items()},
           "samples": rec}
    return out


def contact_depth(crest, spill_depth, fault_column=None, top_column=None):
    """Contact = crest + the limiting column (spill vs fault seal vs top seal) -> (depth, limiting factor)."""
    from .migration import limiting_column
    col, why = limiting_column(spill_depth - crest, fault_column, top_column)
    return crest + col, why


# ====================================================================== 28b
def chance_of_success(factors):
    """Pg = product of the factor probabilities (missing factors = 1)."""
    return float(np.prod([float(factors.get(k, 1.0)) for k in FACTORS]))


def prefill_factors(critical=None, limiting=None, dhi=None, reservoir_evidence=None, source_evidence=None):
    """Rose-style factor suggestions with the evidence: critical = migration.critical_moment result (timing / charge),
    limiting = (column, why) from the seal analysis, dhi = 25d result, reservoir_evidence = e.g. net-to-gross or
    porosity probability from the petro / predictive work, source_evidence = probability of an effective kitchen.
    Returns {factor: (value, reason)} - the interpreter edits these."""
    out = {}
    if source_evidence is not None:
        out["source"] = (float(source_evidence), "basin model / maturity (item 18)")
    if critical is not None:
        f = critical["after_trap"]
        out["timing"] = (float(np.clip(0.3 + 0.65 * f, 0.05, 0.95)), f"{100 * f:.0f} % of the charge after trap "
                         f"formation ({critical['verdict']})")
    if limiting is not None:
        col, why = limiting
        v = 0.8 if why == "spill" else (0.6 if col > 30 else 0.4)
        out["seal"] = (v, f"column {col:.0f} m limited by {why}")
    if reservoir_evidence is not None:
        out["reservoir"] = (float(reservoir_evidence), "petrophysics / predictive modelling (items 20, 24)")
    if dhi is not None:
        out["dhi_index"] = (float(dhi["index"]), "; ".join(dhi.get("reasons", [])[:3]))
    return out


def dhi_revision(pg, dhi_index, strength=4.0, neutral=0.3, quality=1.0):
    """Revised Pg from the DHI index (0..1): Bayes with likelihood ratio LR = exp(strength * quality * (index -
    neutral)), i.e. an index above `neutral` raises Pg, below lowers it.  Bounded in (0, 1), monotonic in the index,
    unchanged at neutral or with quality 0."""
    pg = float(np.clip(pg, 1e-9, 1 - 1e-9))
    lr = math.exp(strength * float(quality) * (float(dhi_index) - neutral))
    return float(pg * lr / (pg * lr + (1 - pg)))


def portfolio(prospects, shared=None, n=20000, seed=0):
    """prospects: [{name, factors {factor: p}, volume (number or samples)}]; shared: {factor: [names]} - a shared
    factor succeeds or fails together for the listed prospects (the first one's probability is used).
    Returns {"p_any", "expected_discoveries", "expected_volume", "per": {name: {pg, mean_volume, risked}},
    "p_all"}."""
    rng = np.random.default_rng(seed)
    names = [p["name"] for p in prospects]
    succ = {nm: np.ones(n, bool) for nm in names}
    shared = shared or {}
    draws = {}
    for f, group in shared.items():
        p0 = next(float(p["factors"].get(f, 1.0)) for p in prospects if p["name"] == group[0])
        draws[f] = (set(group), rng.random(n) < p0)
    for p in prospects:
        for f in FACTORS:
            pf = float(p["factors"].get(f, 1.0))
            if f in draws and p["name"] in draws[f][0]:
                succ[p["name"]] &= draws[f][1]
            else:
                succ[p["name"]] &= rng.random(n) < pf
    vol = np.zeros(n)
    per = {}
    for p in prospects:
        v = p.get("volume", 0.0)
        vs = rng.choice(np.asarray(v, float), n) if np.ndim(v) else np.full(n, float(v))
        vol += np.where(succ[p["name"]], vs, 0.0)
        pg = chance_of_success(p["factors"])
        per[p["name"]] = {"pg": pg, "pg_mc": float(succ[p["name"]].mean()), "mean_volume": float(np.mean(vs)),
                          "risked": float(pg * np.mean(vs))}
    S = np.array([succ[nm] for nm in names])
    return {"p_any": float(S.any(0).mean()), "p_all": float(S.all(0).mean()),
            "expected_discoveries": float(S.sum(0).mean()), "expected_volume": float(vol.mean()), "per": per}


ROSE_COLUMNS = ("Prospect", "Fluid", "P90", "P50", "P10", "Mean", "Swanson", "Unit", "Pg", "P_source", "P_reservoir",
                "P_trap", "P_seal", "P_timing", "DHI_index", "Pg_DHI", "Risked_mean")


def export_csv(path, rows):
    """Portfolio table (Rose-compatible column names where known)."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ROSE_COLUMNS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def prospect_row(name, vol_result, factors, fluid="oil", dhi_index=None):
    rec = vol_result["recoverable"]
    pg = chance_of_success(factors)
    pgd = dhi_revision(pg, dhi_index) if dhi_index is not None else pg
    return {"Prospect": name, "Fluid": fluid, "P90": rec["p90"], "P50": rec["p50"], "P10": rec["p10"],
            "Mean": rec["mean"], "Swanson": rec["swanson"], "Unit": vol_result["unit"], "Pg": pg,
            **{f"P_{k}": factors.get(k, 1.0) for k in FACTORS}, "DHI_index": dhi_index, "Pg_DHI": pgd,
            "Risked_mean": pgd * rec["mean"]}
