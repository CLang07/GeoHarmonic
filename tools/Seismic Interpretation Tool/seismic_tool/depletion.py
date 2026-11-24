"""Depletion, resource left, recovery factor, RTA, frac complexity, M&A screen (PLAN 30).  Pure numpy.

30a  depletion_factor: parent cumulative production inside a child's drainage ellipse at the child's first production
     (parent cum x overlap / parent ellipse area) over the in-place volume of the child's ellipse (SoPhiH x area / B);
     pressure-depletion proxy; child performance vs depletion cross-plot (slope, r).
30b  recovery_factors (EUR / in-place in each ellipse), remaining_map (in-place grid minus what the ellipses drained),
     undrilled_locations (slots at a spacing not covered by existing drainage).
30c  rta_linear (square-root-time linear-flow analysis on pressure-normalised rate or rate-only proxy, end of linear
     flow), material_balance_time, investigation distance; frac_complexity_index -> implied drainage aspect for 29c.
30d  type_curves (P10 / P50 / P90 per month by group), cash_flow / npv (price, royalty, opex, capex, monthly
     discounting), inventory (PDP + PUD value), area_summary (one-page HTML).
"""
import math

import numpy as np


# ====================================================================== 30a
def ellipse_mask(e, X, Y):
    az = math.radians(e["az"])
    dx, dy = X - e["cx"], Y - e["cy"]
    pa = dx * math.sin(az) + dy * math.cos(az)
    pb = dx * math.cos(az) - dy * math.sin(az)
    return (pa / e["a"]) ** 2 + (pb / e["b"]) ** 2 <= 1


def ellipse_area(e):
    return math.pi * e["a"] * e["b"]


def inplace_in(e, sophih, X, Y, cell_area, fvf=1.2):
    """In-place hydrocarbon volume (sm3) inside an ellipse from a SoPhiH grid (m) on nodes X, Y."""
    m = ellipse_mask(e, X, Y)
    return float(np.nansum(np.asarray(sophih, float)[m]) * cell_area / fvf)


def overlap_fraction(e_parent, e_child, X, Y):
    """Fraction of the parent's ellipse that lies inside the child's ellipse (on the grid X, Y)."""
    mp = ellipse_mask(e_parent, X, Y)
    if not mp.any():
        return 0.0
    return float(np.sum(mp & ellipse_mask(e_child, X, Y)) / mp.sum())


def depletion_factor(child, parents, cum_at, sophih, X, Y, cell_area, fvf=1.2, ct_proxy=1.0):
    """child / parents: ellipse dicts with 'name'; cum_at {parent: cumulative production (sm3) at the child's first
    production}.  Returns {df, depleted_sm3, inplace_sm3, pressure_proxy, contributions}."""
    inpl = inplace_in(child, sophih, X, Y, cell_area, fvf)
    dep, contrib = 0.0, {}
    for p in parents:
        f = overlap_fraction(p, child, X, Y)
        v = f * float(cum_at.get(p["name"], 0.0))
        if v > 0:
            contrib[p["name"]] = v
        dep += v
    df = dep / inpl if inpl > 0 else float("nan")
    return {"df": df, "depleted_sm3": dep, "inplace_sm3": inpl, "contributions": contrib,
            "pressure_proxy": float(max(0.0, 1.0 - ct_proxy * df)) if np.isfinite(df) else float("nan")}


def performance_vs_depletion(dfs, perf):
    """Least-squares slope and correlation of child performance (e.g. EUR / 1000 ft) vs depletion factor."""
    x, y = np.asarray(dfs, float), np.asarray(perf, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "r": float("nan"), "n": int(ok.sum())}
    a, b = np.polyfit(x[ok], y[ok], 1)
    return {"slope": float(a), "intercept": float(b), "r": float(np.corrcoef(x[ok], y[ok])[0, 1]), "n": int(ok.sum())}


# ====================================================================== 30b
def recovery_factors(ellipses, eur_sm3, sophih, X, Y, cell_area, fvf=1.2):
    """{well: RF = EUR / in-place inside its drainage ellipse}."""
    return {e["name"]: (float(eur_sm3.get(e["name"], np.nan)) / v if (v := inplace_in(e, sophih, X, Y, cell_area, fvf)) > 0
                        else float("nan")) for e in ellipses}


def remaining_map(sophih, X, Y, cell_area, ellipses, rf, fvf=1.2):
    """In-place sm3 per node minus the fraction each well drained (its RF) inside its ellipse (overlaps compound)."""
    inpl = np.asarray(sophih, float) * cell_area / fvf
    keep = np.ones(inpl.shape)
    for e in ellipses:
        r = float(np.clip(rf.get(e["name"], 0.0), 0, 1)) if np.isfinite(rf.get(e["name"], np.nan)) else 0.0
        keep = np.where(ellipse_mask(e, X, Y), keep * (1 - r), keep)
    return inpl * keep


def undrilled_locations(unit_mask, X, Y, ellipses, spacing, azimuth=0.0, lateral=2000.0, max_cover=0.2):
    """Lateral slots across a unit (mask on nodes X, Y) every `spacing` m perpendicular to the lateral azimuth: a slot
    is a location if the unit is at least half a lateral long along it and existing drainage ellipses cover less than
    max_cover of it.  Returns (n, [slot offsets])."""
    az = math.radians(azimuth)
    along = X * math.sin(az) + Y * math.cos(az)
    across = X * math.cos(az) - Y * math.sin(az)
    um = np.asarray(unit_mask, bool)
    cov = np.zeros(um.shape, bool)
    for e in ellipses:
        cov |= ellipse_mask(e, X, Y)
    lo, hi = float(across[um].min()), float(across[um].max())
    cell = float(np.median(np.abs(np.diff(np.unique(np.round(across[um], 6))))) or spacing / 10)
    slots = []
    for c in np.arange(lo + spacing / 2, hi + 1e-9, spacing):
        band = um & (np.abs(across - c) <= max(cell, 1e-9) / 2 + 1e-9)
        if not band.any():
            continue
        length = float(along[band].max() - along[band].min())
        if length < 0.5 * lateral:
            continue
        if float(np.mean(cov[band])) < max_cover:
            slots.append(float(c))
    return len(slots), slots


# ====================================================================== 30c
def material_balance_time(q, dt_days=30.4375):
    """t_mb = cumulative / rate (days) for a rate series sampled every dt_days."""
    q = np.asarray(q, float)
    cum = np.cumsum(q * dt_days)
    return np.where(q > 0, cum / np.where(q > 0, q, 1), np.nan)


def rta_linear(t_days, q, dp=None, min_points=5, dev=0.1):
    """Square-root-time linear-flow analysis.  y = dp / q (pressure-normalised) or 1 / q (rate-only proxy, assumes
    constant drawdown - flagged) vs sqrt(t): straight line through the early data; end of linear flow = first time
    the data leave the line by more than `dev` (fraction) for 3 points in a row.  Returns {slope, intercept, r2,
    t_elf, proxy}."""
    t = np.asarray(t_days, float)
    q = np.asarray(q, float)
    y = (np.asarray(dp, float) / q) if dp is not None else 1.0 / q
    x = np.sqrt(t)
    ok = np.isfinite(x) & np.isfinite(y) & (q > 0)
    x, y, tt = x[ok], y[ok], t[ok]
    n_fit = max(min_points, len(x) // 4)
    a, b = np.polyfit(x[:n_fit], y[:n_fit], 1)
    for _ in range(3):  # grow the fit window while the data stay on the line
        pred = a * x + b
        off = np.abs(y - pred) / np.maximum(np.abs(pred), 1e-30) > dev
        run = np.convolve(off.astype(int), np.ones(3, int), "valid") >= 3
        k = int(np.argmax(run)) if run.any() else len(x)
        k = max(k, min_points)
        a, b = np.polyfit(x[:k], y[:k], 1)
    # the line is fitted again on the points that stay within dev / 4 of it (excludes the bend), t_elf from k
    pred = a * x + b
    near = np.abs(y - pred) / np.maximum(np.abs(pred), 1e-30) <= dev / 4
    kf = max(min_points, int(np.argmin(near)) if not near.all() else len(x))
    a, b = np.polyfit(x[:kf], y[:kf], 1)
    pred = a * x[:k] + b
    r2 = 1 - np.sum((y[:k] - pred) ** 2) / max(1e-30, np.sum((y[:k] - y[:k].mean()) ** 2))
    return {"slope": float(a), "intercept": float(b), "r2": float(r2),
            "t_elf": float(tt[k]) if k < len(tt) else float("nan"), "n_linear": int(k), "proxy": dp is None}


def investigation_distance(t_days, eta_m2_per_day):
    """Linear-flow distance of investigation sqrt(eta t) (m) - with t_elf gives the half-spacing drained."""
    return math.sqrt(max(0.0, eta_m2_per_day * t_days))


FLUID_COMPLEXITY = {"slickwater": 1.0, "hybrid": 0.6, "crosslinked": 0.25, "gel": 0.2}


def frac_complexity_index(fluid="slickwater", fluid_bbl_ft=None, proppant_lb_ft=None, cluster_spacing_ft=None,
                          stress_anisotropy=None, brittleness=None, net_pressure_mpa=None):
    """0 (planar, bi-wing) .. 1 (complex, wide SRV) from the frac design and the rock (each term 0..1, mean of the
    terms given) and the implied drainage aspect (width / half-length) for spacing.drainage_ellipse.
    stress_anisotropy = (SHmax - Shmin) / Shmin; low anisotropy and net pressure above it favour complexity."""
    terms = {"fluid": FLUID_COMPLEXITY.get(str(fluid).lower(), 0.5)}
    if fluid_bbl_ft is not None:
        terms["fluid volume"] = float(np.clip((fluid_bbl_ft - 20) / 60, 0, 1))
    if proppant_lb_ft is not None:
        terms["proppant"] = float(np.clip((proppant_lb_ft - 500) / 2500, 0, 1))
    if cluster_spacing_ft is not None:
        terms["cluster spacing"] = float(np.clip((60 - cluster_spacing_ft) / 45, 0, 1))
    if stress_anisotropy is not None:
        a = float(np.clip(1 - stress_anisotropy / 0.3, 0, 1))
        if net_pressure_mpa is not None and stress_anisotropy > 0:
            a = max(a, float(np.clip(net_pressure_mpa / 5.0, 0, 1)) * 0.8)
        terms["stress anisotropy"] = a
    if brittleness is not None:
        terms["brittleness"] = float(np.clip(brittleness, 0, 1))
    fci = float(np.mean(list(terms.values())))
    return {"fci": fci, "terms": terms, "aspect": 0.15 + 0.6 * fci,
            "shape": "complex / wide SRV" if fci > 0.6 else ("intermediate" if fci > 0.35 else "planar / narrow")}


# ====================================================================== 30d
def type_curves(series, groups=None, months=None, norm=None):
    """{well: monthly rates} -> {group: {"p10", "p50", "p90", "mean", "n"}} per month (P90 = low = 10th percentile;
    norm {well: divisor, e.g. lateral / 1000 ft})."""
    groups = groups or {w: "all" for w in series}
    months = months or max(len(v) for v in series.values())
    out = {}
    for g in sorted(set(groups.values())):
        ws = [w for w in series if groups.get(w) == g]
        M = np.full((len(ws), months), np.nan)
        for i, w in enumerate(ws):
            v = np.asarray(series[w], float)[:months] / float((norm or {}).get(w, 1.0))
            M[i, :len(v)] = v
        with np.errstate(all="ignore"):
            out[g] = {"p90": np.nanpercentile(M, 10, 0), "p50": np.nanpercentile(M, 50, 0),
                      "p10": np.nanpercentile(M, 90, 0), "mean": np.nanmean(M, 0), "n": len(ws)}
    return out


def cash_flow(q_month, price=70.0, royalty=0.2, opex_fixed=10000.0, opex_var=8.0, capex=0.0, tax=0.0):
    """Monthly net cash flow of a rate series (volume per month): revenue x (1 - royalty) - opex - tax, capex at t0."""
    q = np.asarray(q_month, float)
    rev = q * price * (1 - royalty)
    cf = rev - opex_fixed - opex_var * q
    cf = cf - tax * np.maximum(cf, 0)
    cf = cf.copy()
    if len(cf):
        cf[0] -= capex
    return cf


def npv(cf, rate=0.10, economic_limit=True):
    """NPV at an annual rate with monthly discounting (end of month); truncates at the economic limit (last month
    with positive operating cash flow) when asked."""
    cf = np.asarray(cf, float)
    if economic_limit and len(cf) > 1:
        pos = np.nonzero(cf[1:] > 0)[0]
        cf = cf[:pos[-1] + 2] if len(pos) else cf[:1]
    disc = (1 + rate) ** (-(np.arange(len(cf)) + 1) / 12.0)
    return float(np.sum(cf * disc))


def inventory(pdp_series, pud_count, type_curve, econ):
    """PDP value (each producing well's remaining forecast) + PUD value (count x NPV of the type curve with capex)."""
    pdp = sum(npv(cash_flow(q, **{k: v for k, v in econ.items() if k != "capex"})) for q in pdp_series.values())
    pv_loc = npv(cash_flow(type_curve, **econ))
    return {"pdp_npv": float(pdp), "pv_per_location": float(pv_loc), "pud_npv": float(pud_count * pv_loc),
            "total": float(pdp + pud_count * pv_loc), "pud_count": int(pud_count)}


def area_summary(path, title, tc, inv, notes=()):
    """One-page HTML area summary (type-curve table + inventory)."""
    rows = "".join(f"<tr><td>{g}</td><td>{d['n']}</td><td>{np.nansum(d['p90'][:12]):.0f}</td>"
                   f"<td>{np.nansum(d['p50'][:12]):.0f}</td><td>{np.nansum(d['p10'][:12]):.0f}</td></tr>"
                   for g, d in tc.items())
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head><body>"
            f"<h1>{title}</h1><h2>Type curves (first-12-month cumulative)</h2><table border=1 cellpadding=4>"
            f"<tr><th>Group</th><th>Wells</th><th>P90</th><th>P50</th><th>P10</th></tr>{rows}</table>"
            f"<h2>Inventory</h2><p>PDP NPV10 {inv['pdp_npv']:,.0f}; PUD {inv['pud_count']} locations x "
            f"{inv['pv_per_location']:,.0f} = {inv['pud_npv']:,.0f}; total {inv['total']:,.0f}</p>"
            + "".join(f"<p>{n}</p>" for n in notes) + "</body></html>")
    with open(path, "w") as fh:
        fh.write(html)
    return path
