"""Multi-mineral solver (Multimin / Mineral Solver style) and Monte-Carlo uncertainty (PLAN 20c).

Model: every log is a volume-weighted sum of end-point responses (linear tool response equations)

    log_j = sum_i V_i * E_ij,     sum_i V_i = 1,    V_i >= 0

over minerals (quartz, calcite, dolomite, illite, ...), formation water and hydrocarbon.  Logs used: GR, RHOB, NPHI
(limestone units), DT, U = PEF x RHOB (volumetric photoelectric factor - linear in volumes, PEF is not) and the
resistivity pseudo-log BVWR = (a Rw / Rt)^(1/n), which for m = n equals the bulk volume of water (Archie), so the
resistivity enters linearly: water response 1, everything else 0 (clays get (a Rw / Rsh)^(1/n)).

Solved per sample as weighted least squares with the sum-to-one equality and non-negativity: for <= 9 unknowns every
support set is tried in one batched solve (the KKT matrix depends only on which logs are present, so it is factored
once per support set and log-availability pattern) and the feasible solution with the lowest misfit is kept - exact,
and fast for whole wells (no per-sample Python loop).  Misfit and reconstructed logs are the QC.

Monte Carlo: ``monte_carlo(project, well, dists)`` re-runs the deterministic chain (20b) or the solver with sampled
parameters (Rw multiplier, m, n, a, matrix density, cut-offs, end-points) and returns P90 / P50 / P10 per zone of net
pay, PHIH and SoPhiH (P90 = low case = 10th percentile, P10 = high case = 90th percentile).
"""
import copy
import itertools

import numpy as np

from . import interp as I
from .store import petro_dir, save_curves

LOGS = ["GR", "RHOB", "NPHI", "DT", "U", "BVWR"]
# end points: GR (API), RHOB (g/cc), NPHI (v/v limestone units), DT (us/ft), PEF (b/e) -> U = PEF x RHOB, clay flag
MINERALS = {
    "quartz": dict(GR=10.0, RHOB=2.65, NPHI=-0.02, DT=55.5, PEF=1.81),
    "calcite": dict(GR=10.0, RHOB=2.71, NPHI=0.00, DT=47.5, PEF=5.08),
    "dolomite": dict(GR=10.0, RHOB=2.87, NPHI=0.01, DT=43.5, PEF=3.14),
    "k-feldspar": dict(GR=220.0, RHOB=2.52, NPHI=-0.03, DT=69.0, PEF=2.86),
    "illite": dict(GR=150.0, RHOB=2.52, NPHI=0.30, DT=85.0, PEF=3.45, clay=True),
    "kaolinite": dict(GR=90.0, RHOB=2.41, NPHI=0.37, DT=95.0, PEF=1.83, clay=True),
    "smectite": dict(GR=150.0, RHOB=2.12, NPHI=0.44, DT=110.0, PEF=2.04, clay=True),
    "chlorite": dict(GR=180.0, RHOB=2.76, NPHI=0.52, DT=90.0, PEF=6.30, clay=True),
    "anhydrite": dict(GR=5.0, RHOB=2.98, NPHI=-0.02, DT=50.0, PEF=5.05),
    "halite": dict(GR=0.0, RHOB=2.04, NPHI=-0.03, DT=67.0, PEF=4.65),
    "coal": dict(GR=20.0, RHOB=1.30, NPHI=0.60, DT=120.0, PEF=0.20),
    "pyrite": dict(GR=0.0, RHOB=4.99, NPHI=-0.03, DT=39.0, PEF=16.97),
}
FLUIDS = {
    "water": dict(GR=0.0, RHOB=1.0, NPHI=1.0, DT=189.0, PEF=0.36),
    "oil": dict(GR=0.0, RHOB=0.8, NPHI=0.9, DT=230.0, PEF=0.12),
    "gas": dict(GR=0.0, RHOB=0.25, NPHI=0.35, DT=230.0, PEF=0.10),  # DT of gas sand is not linear: low weight
}
SIGMA = {"GR": 6.0, "RHOB": 0.03, "NPHI": 0.025, "DT": 4.0, "U": 0.6, "BVWR": 0.025}
GAS_DT_SIGMA = 25.0


def endpoint_matrix(components, minerals=None, fluids=None, rw=0.05, rsh=2.0, a=1.0, n=2.0):
    """(n_logs, n_components) response matrix E for the components (mineral names + 'water' + 'oil'/'gas')."""
    minerals = minerals or MINERALS
    fluids = fluids or FLUIDS
    E = np.zeros((len(LOGS), len(components)))
    for j, c in enumerate(components):
        e = fluids[c] if c in fluids else minerals[c]
        E[0, j], E[1, j], E[2, j], E[3, j] = e["GR"], e["RHOB"], e["NPHI"], e["DT"]
        E[4, j] = e["PEF"] * e["RHOB"]
        # BVWR: water 1; minerals (clays too - their conductance is ignored, an Archie-type simplification) 0
        E[5, j] = 1.0 if c == "water" else 0.0
    return E


def solve_volumes(D, E, sigma, max_unknowns=9):
    """Constrained weighted LS for every row of D (n_samples, n_logs; NaN = log missing).
    Returns (V (n_samples, k), misfit chi2 per equation (n_samples,), reconstructed (n_samples, n_logs))."""
    D = np.asarray(D, float)
    ns, nl = D.shape
    k = E.shape[1]
    if k > max_unknowns:
        raise ValueError(f"{k} unknowns: at most {max_unknowns}")
    w = 1.0 / np.asarray(sigma, float)
    V = np.full((ns, k), np.nan)
    best = np.full(ns, np.inf)
    avail = np.isfinite(D)
    patterns, inv = np.unique(avail, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    subsets = [s for r in range(1, k + 1) for s in itertools.combinations(range(k), r)]
    for pi, pat in enumerate(patterns):
        rows = np.where(inv == pi)[0]
        L = np.where(pat)[0]
        if len(L) == 0:
            continue
        b = D[np.ix_(rows, L)] * w[L]  # (nr, nL)
        A = E[L] * w[L, None]  # (nL, k)
        for S in subsets:
            S = list(S)
            if len(S) > len(L) + 1:
                continue
            As = A[:, S]
            # KKT: [2 As^T As, 1; 1^T, 0] [x; lam] = [2 As^T b; 1]
            m = len(S)
            K = np.zeros((m + 1, m + 1))
            K[:m, :m] = 2 * As.T @ As
            K[:m, m] = 1.0
            K[m, :m] = 1.0
            try:
                Ki = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(Ki)) or np.linalg.cond(K) > 1e12:
                continue
            rhs = np.zeros((len(rows), m + 1))
            rhs[:, :m] = 2 * b @ As
            rhs[:, m] = 1.0
            x = rhs @ Ki.T
            xs = x[:, :m]
            feas = np.all(xs >= -1e-10, axis=1)
            if not feas.any():
                continue
            r = b - xs @ As.T
            obj = (r * r).sum(1)
            upd = feas & (obj < best[rows] - 1e-15)
            if upd.any():
                rr = rows[upd]
                best[rr] = obj[upd]
                full = np.zeros((upd.sum(), k))
                full[:, S] = np.maximum(xs[upd], 0.0)
                V[rr] = full
    recon = V @ E.T
    neq = np.maximum(avail.sum(1), 1)
    return V, np.where(np.isfinite(best), best / neq, np.nan), recon


def log_matrix(w, rw, a=1.0, n=2.0):
    """(n_samples, 6) observations GR, RHOB, NPHI, DT, U, BVWR from a loaded well (NaN where missing)."""
    nn = len(w.depth)
    nan = np.full(nn, np.nan)
    L = {k: np.asarray(v, float) for k, v in w.logs.items()}
    rhob, pef, rt = L.get("RHOB", nan), L.get("PE", nan), L.get("RDEEP", nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        bvwr = (a * np.asarray(rw, float) / rt) ** (1.0 / n)
    return np.column_stack([L.get("GR", nan), rhob, L.get("NPHI", nan), L.get("DT", nan), pef * rhob, bvwr])


def multimin_well(project, name, minerals=("quartz", "illite"), fluid="oil", pset=None, endpoints=None, sigma=None,
                  zones=None, save=True, det=None):
    """Solve volumes on a well.  minerals / fluid can also come per zone from the parameter set ("mm_minerals",
    "mm_fluid").  Rw (at temperature), a, n and cut-offs come from the 20b interpretation of the same zone.
    Returns {"well", "depth", "components", "volumes" {comp: array}, "curves" {...}, "zones" [rows], "misfit"}."""
    w = project.load_well(name)
    pset = I.load_params(project) if pset is None else pset
    det = det or I.interpret_well(project, name, pset, save=False)
    z = w.depth
    nn = len(z)
    mins = dict(MINERALS)
    if endpoints:
        for k, v in endpoints.items():
            if k not in FLUIDS:
                mins[k] = dict(mins.get(k, {}), **v)
    flu = {k: dict(v, **(endpoints or {}).get(k, {})) for k, v in FLUIDS.items()}
    sig = dict(SIGMA, **(sigma or {}))
    out_vol = {}
    curves = {"MM_PHIE": np.full(nn, np.nan), "MM_SW": np.full(nn, np.nan), "MM_VCLAY": np.full(nn, np.nan),
              "MM_MISFIT": np.full(nn, np.nan)}
    recon_all = {f"MM_REC_{lg}": np.full(nn, np.nan) for lg in LOGS}
    segs = I.zones_of(project, name, z)
    segs = segs or [(None, float(z[0]), float(z[-1]) + 1e-6)]
    rows = []
    comps_seen = []
    rw_curve = np.full(nn, np.nan)
    for zone, top, base in segs:
        if zones and zone not in zones:
            continue
        idx = np.where((z >= top) & (z < base))[0]
        if not idx.size:
            continue
        P = I.resolve(pset, name, zone)
        zm = list(P.get("mm_minerals") or minerals)
        zf = P.get("mm_fluid") or fluid
        comps = zm + ["water", zf]
        for c in comps:
            if c not in comps_seen:
                comps_seen.append(c)
                out_vol[c] = np.full(nn, np.nan)
        used = det["used"].get(zone, {})
        rw = used.get("rw") or 0.05
        rsh = used.get("rsh") or 2.0
        a, n = P["a"], P["n"]
        E = endpoint_matrix(comps, mins, flu, rw, rsh, a, n)
        D = log_matrix(w, rw, a, n)[idx]
        use = P.get("mm_logs") or LOGS  # e.g. drop "U" in barite mud (PEF reads 4-5 in clean sand)
        for j, lg in enumerate(LOGS):
            if lg not in use:
                D[:, j] = np.nan
        s = np.array([sig[lg] for lg in LOGS])
        if zf == "gas":
            s[3] = max(s[3], GAS_DT_SIGMA)
        V, mis, rec = solve_volumes(D, E, s)
        for j, c in enumerate(comps):
            out_vol[c][idx] = V[:, j]
        jw, jh = comps.index("water"), comps.index(zf)
        phie = V[:, jw] + V[:, jh]
        with np.errstate(invalid="ignore", divide="ignore"):
            sw = np.where(phie > 0.005, V[:, jw] / np.where(phie > 0.005, phie, 1), 1.0)
        clay = [j for j, c in enumerate(comps) if mins.get(c, {}).get("clay")]
        vcl = V[:, clay].sum(1) if clay else np.zeros(len(idx))
        vcl[~np.isfinite(V[:, 0])] = np.nan
        sw[~np.isfinite(phie)] = np.nan
        curves["MM_PHIE"][idx], curves["MM_SW"][idx], curves["MM_VCLAY"][idx], curves["MM_MISFIT"][idx] = phie, sw, vcl, mis
        rw_curve[idx] = rw
        for j, lg in enumerate(LOGS):
            recon_all[f"MM_REC_{lg}"][idx] = rec[:, j]
        if zone is not None:
            tmp = {"VSH": curves["MM_VCLAY"], "PHIE": curves["MM_PHIE"], "SW": curves["MM_SW"],
                   "PERM": np.where(np.isfinite(curves["MM_PHIE"]),
                                    I.perm_timur(curves["MM_PHIE"], np.clip(np.nan_to_num(curves["MM_SW"], nan=1.0),
                                                                            P["swi_min"], 1)), np.nan),
                   "TOC": np.full(nn, np.nan)}
            vc = P["mm_vclay_cut"] if P.get("mm_vclay_cut") is not None else P["vsh_cut"]
            pc, sc = P["phi_cut"], P["sw_cut"]
            row = I.zone_summary(name, zone, top, base, z[idx], det["tvdss"][idx], tmp, idx, cuts=(vc, pc, sc))
            row["Misfit_med"] = round(float(np.nanmedian(mis)), 3) if np.isfinite(mis).any() else None
            row["Components"] = "+".join(comps)
            row.update({f"V_{c}": round(float(np.nanmean(V[:, j])), 3) if np.isfinite(V[:, j]).any() else None
                        for j, c in enumerate(comps)})
            rows.append(row)
    res = {"well": name, "depth": z, "tvdss": det["tvdss"], "components": comps_seen, "volumes": out_vol,
           "curves": dict(curves, **recon_all), "zones": rows, "rw": rw_curve}
    if save and petro_dir(project):
        write_mm_summary(project, name, rows)
        allc = {f"MM_V_{c.upper().replace('-', '_')}": v.astype(np.float32) for c, v in out_vol.items()}
        allc.update({k: v.astype(np.float32) for k, v in res["curves"].items()})
        save_curves(project, name, z, allc, replace_prefix="MM_")
    return res


def write_mm_summary(project, well, rows):
    """Zone rows of the mineral solver (with V_<component> means) in <project>_petro/mm_summary.csv, other wells kept
    (read by the Seismic tool's map hub for mineral pies)."""
    import csv
    import os
    d = petro_dir(project, create=True)
    path = os.path.join(d, "mm_summary.csv")
    old = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            old = [r for r in csv.DictReader(f) if r.get("Well") != well]
    allrows = old + [{k: ("" if v is None else v) for k, v in r.items()} for r in rows]
    cols = list(dict.fromkeys(k for r in allrows for k in r))
    with open(path + ".tmp", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in allrows:
            wr.writerow({k: r.get(k, "") for k in cols})
    os.replace(path + ".tmp", path)
    return path


# ------------------------------------------------------------------------------------------------- Monte Carlo
def _draw(rng, spec, base, size):
    """spec: ("normal", sd) | ("rel_normal", sd_fraction) | ("uniform", lo, hi) | ("triangular", lo, mode, hi) |
    ("lognormal", sd_log10).  Relative / lognormal forms scale the base value."""
    kind = spec[0]
    if kind == "normal":
        return base + spec[1] * rng.standard_normal(size)
    if kind == "rel_normal":
        return base * (1.0 + spec[1] * rng.standard_normal(size))
    if kind == "uniform":
        return rng.uniform(spec[1], spec[2], size)
    if kind == "triangular":
        return rng.triangular(spec[1], spec[2], spec[3], size)
    if kind == "lognormal":
        return base * 10 ** (spec[1] * rng.standard_normal(size))
    raise ValueError(f"unknown distribution {kind}")


MC_DEFAULT = {"rw_mult": ("lognormal", 0.08), "m": ("normal", 0.1), "n": ("normal", 0.15), "rho_ma": ("normal", 0.01),
              "phi_cut": ("normal", 0.01), "sw_cut": ("normal", 0.05), "vsh_cut": ("normal", 0.05)}


def monte_carlo(project, name, dists=None, n=200, seed=1, pset=None, method="deterministic", mm_kwargs=None,
                endpoint_sd=None):
    """P90 / P50 / P10 of NetPay_TVD, PHIH_pay and SoPhiH per zone.

    dists: {param: spec} (see _draw); parameters are 20b parameter names plus "rw_mult" (multiplies the Rw each zone
    uses, auto or given).  Each realisation applies the same draw to every zone (parameters are correlated across
    zones - the usual conservative choice).  method "multimin" also perturbs mineral end-points by endpoint_sd
    ({log: sd}, default RHOB 0.02, NPHI 0.02, DT 3, GR 5, PEF 0.2).
    Returns {"zones": {zone: {metric: {"P90", "P50", "P10", "mean", "values"}}}, "draws": {param: array}, "n"}."""
    pset = I.load_params(project) if pset is None else pset
    dists = MC_DEFAULT if dists is None else dists
    rng = np.random.default_rng(seed)
    base = I.resolve(pset, name, None)
    draws = {k: _draw(rng, spec, base.get(k, 1.0) if k != "rw_mult" else 1.0, n) for k, spec in dists.items()}
    for k in ("m", "n"):
        if k in draws:
            draws[k] = np.clip(draws[k], 1.0, 4.0)
    if "rw_mult" in draws:
        draws["rw_mult"] = np.maximum(draws["rw_mult"], 0.05)
    metrics = ("NetPay_TVD", "PHIH_pay", "SoPhiH")
    acc = {}
    eps = endpoint_sd or {"RHOB": 0.02, "NPHI": 0.02, "DT": 3.0, "GR": 5.0, "PEF": 0.2}
    w = project.load_well(name)  # load once (cached on the well)
    del w
    for i in range(n):
        ps = copy.deepcopy(pset)
        over = {k: float(v[i]) for k, v in draws.items()}
        ps.setdefault("default", {})
        # a realisation's draw overrides every zone: put it at the highest precedence
        for zone_d in [ps["default"]] + list(ps.get("zones", {}).values()) + \
                [zz for wd in ps.get("wells", {}).values() for zz in wd.values()]:
            for k, v in over.items():
                if k in ("m", "n", "a", "rho_ma", "phi_cut", "sw_cut", "vsh_cut", "rw_mult", "gradient"):
                    zone_d[k] = v
        if method == "multimin":
            ep = {}
            for c, e in MINERALS.items():
                ep[c] = {k: e[k] + eps.get(k, 0.0) * rng.standard_normal() for k in ("GR", "RHOB", "NPHI", "DT", "PEF")}
            det = I.interpret_well(project, name, ps, save=False)
            r = multimin_well(project, name, pset=ps, endpoints=ep, save=False, det=det, **(mm_kwargs or {}))
        else:
            r = I.interpret_well(project, name, ps, save=False)
        for row in r["zones"]:
            for mt in metrics:
                acc.setdefault(row["Zone"], {}).setdefault(mt, []).append(row[mt] if row[mt] is not None else np.nan)
    out = {}
    for zone, d in acc.items():
        out[zone] = {}
        for mt, vals in d.items():
            v = np.asarray(vals, float)
            f = v[np.isfinite(v)]
            out[zone][mt] = {"P90": float(np.percentile(f, 10)) if f.size else None,
                             "P50": float(np.percentile(f, 50)) if f.size else None,
                             "P10": float(np.percentile(f, 90)) if f.size else None,
                             "mean": float(f.mean()) if f.size else None, "values": v}
    return {"zones": out, "draws": draws, "n": n, "method": method}


def mc_table(mc):
    """Rows for CSV / display: Well-less (caller adds the well)."""
    rows = []
    for zone, d in mc["zones"].items():
        r = {"Zone": zone}
        for mt, s in d.items():
            for p in ("P90", "P50", "P10"):
                r[f"{mt}_{p}"] = None if s[p] is None else round(s[p], 3)
        rows.append(r)
    return rows
