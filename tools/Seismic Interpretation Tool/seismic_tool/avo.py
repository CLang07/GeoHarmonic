"""AVO, DHI, fizz gas and angle-stack risk (PLAN 25).

25a  zoeppritz_pp (exact), aki_richards / shuey (predictive.py), avo_class (I, II, IIp, III, IV), ig_from_gather,
     background_trend + ig_anomaly (distance from the wet trend in the intercept-gradient plane), fluid_factor,
     interface_avo (curves + class at a well interface, scenarios brine / oil / gas / fizz via Gassmann).
25b  fizz_assessment: commercial gas vs fizz from the density contrast (inverted density) with the density error from
     25c, plus low-frequency shadow / attenuation evidence -> fizz-risk score with reasons.
25c  stack_vs_gather_risk: expected errors of the P-, S- and density reflectivities (Zp, Zs, rho) from 2 / 3 / 5
     angle stacks or the full gather for the actual angle range, noise and fold (linear Aki-Richards, Bayesian with a
     weak prior; analytic covariance + Monte-Carlo check).
25d  DHI checklist (user-editable weights): amplitude strength, conformance to structure, AVO, stack consistency
     computed; flat spot, phase change, edge sharpness, data quality scored by the interpreter -> DHI index + reasons.
"""
import math

import numpy as np

from .predictive import aki_richards, shuey_ab  # noqa: F401  (re-exported)


# ====================================================================== 25a
def zoeppritz_pp(vp1, vs1, rho1, vp2, vs2, rho2, theta_deg):
    """Exact P-P reflection coefficient (real part; complex beyond the critical angle) for angles theta_deg."""
    th = np.radians(np.atleast_1d(np.asarray(theta_deg, float))).astype(complex)
    p = np.sin(th) / vp1
    t1 = th
    t2 = np.arcsin(p * vp2 + 0j)
    f1 = np.arcsin(p * vs1 + 0j)
    f2 = np.arcsin(p * vs2 + 0j)
    out = []
    for a1, a2, b1, b2 in zip(t1, t2, f1, f2):
        M = np.array([
            [-np.sin(a1), -np.cos(b1), np.sin(a2), np.cos(b2)],
            [np.cos(a1), -np.sin(b1), np.cos(a2), -np.sin(b2)],
            [np.sin(2 * a1), vp1 / vs1 * np.cos(2 * b1), rho2 * vs2 ** 2 * vp1 / (rho1 * vs1 ** 2 * vp2) * np.sin(2 * a2),
             rho2 * vs2 * vp1 / (rho1 * vs1 ** 2) * np.cos(2 * b2)],
            [-np.cos(2 * b1), vs1 / vp1 * np.sin(2 * b1), rho2 * vp2 / (rho1 * vp1) * np.cos(2 * b2),
             -rho2 * vs2 / (rho1 * vp1) * np.sin(2 * b2)]], complex)
        N = np.array([np.sin(a1), np.cos(a1), np.sin(2 * a1), np.cos(2 * b1)], complex)
        out.append(np.linalg.solve(M, N)[0])
    r = np.array(out)
    return np.real(r) if np.ndim(theta_deg) else float(np.real(r[0]))


def avo_class(A, B, theta_max=35.0, a_small=0.02):
    """Rutherford-Williams classes (+ Castagna IV) of top-reservoir reflections from intercept A and gradient B:
    I (A > a_small, dims), IIp (A > a_small but changes polarity before theta_max), II (|A| <= a_small),
    III (A < -a_small, B < 0: brightens), IV (A < -a_small, B >= 0: dims with angle)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    s2 = math.sin(math.radians(theta_max)) ** 2
    out = np.full(A.shape, "", dtype=object)
    out[np.abs(A) <= a_small] = "II"
    pos = A > a_small
    out[pos & (A + B * s2 < 0)] = "IIp"
    out[pos & (A + B * s2 >= 0)] = "I"
    neg = A < -a_small
    out[neg & (B < 0)] = "III"
    out[neg & (B >= 0)] = "IV"
    return out.tolist() if out.ndim else str(out)


def ig_from_gather(gather, angles):
    """Least-squares intercept / gradient of R = A + B sin^2 for a gather (n_angles, ...) -> (A, B) arrays."""
    g = np.asarray(gather, float)
    s2 = np.sin(np.radians(np.asarray(angles, float))) ** 2
    G = np.column_stack([np.ones(len(s2)), s2])
    coef = np.linalg.lstsq(G, g.reshape(len(s2), -1), rcond=None)[0]
    return coef[0].reshape(g.shape[1:]), coef[1].reshape(g.shape[1:])


def background_trend(A, B, iters=5, trim=2.0):
    """Robust line B = m A + c through the (wet) background cloud: LS with iterative removal of > trim sd outliers.
    Returns (m, c, sd of the perpendicular distance)."""
    A, B = np.ravel(np.asarray(A, float)), np.ravel(np.asarray(B, float))
    ok = np.isfinite(A) & np.isfinite(B)
    keep = ok.copy()
    m = c = 0.0
    sd = 1.0
    for _ in range(iters):
        m, c = np.polyfit(A[keep], B[keep], 1)
        d = (B - m * A - c) / math.sqrt(1 + m * m)
        sd = float(np.std(d[keep])) or 1e-12
        new = ok & (np.abs(d) <= trim * sd)
        if new.sum() < 10 or np.array_equal(new, keep):
            break
        keep = new
    return float(m), float(c), sd


def ig_anomaly(A, B, trend):
    """Signed distance from the background trend in sd units (negative = below the trend: soft / hydrocarbon side at
    a top-reservoir trough)."""
    m, c, sd = trend
    return (np.asarray(B, float) - m * np.asarray(A, float) - c) / math.sqrt(1 + m * m) / sd


def fluid_factor(A, B, vpvs=2.0):
    """Fluid factor F = Rp - 1.16 (Vs/Vp) Rs (Smith & Gidlow / Fatti) from intercept and gradient: with constant
    density Aki-Richards gives Rp = A and B = Rp - 8 (Vs/Vp)^2 Rs, so Rs = (A - B) vpvs^2 / 8.  Wet rocks on the
    mudrock line give F ~ 0, gas sands F < 0 at their top."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    rs = (A - B) * vpvs ** 2 / 8.0
    return A - 1.16 / vpvs * rs


def interface_avo(upper, lower, angles=np.arange(0, 41, 2.0), theta_max=35.0):
    """AVO of one interface: upper / lower (vp, vs, rho) -> {angles, zoeppritz, aki, A, B, class}."""
    z = zoeppritz_pp(*upper, *lower, angles)
    ak = aki_richards(*upper, *lower, angles)
    A, B = ig_from_gather(ak, angles)
    return {"angles": np.asarray(angles, float), "zoeppritz": z, "aki": ak, "A": float(A), "B": float(B),
            "class": avo_class(float(A), float(B), theta_max)}


def scenario_avo(shale, phi, vsh, sc=None, fluids=("brine", "oil", "gas", "fizz"), angles=np.arange(0, 41, 2.0)):
    """Top-sand AVO for Gassmann fluid scenarios (predictive.sand_elastic) under a shale -> {fluid: interface_avo}."""
    from .predictive import Scenario, sand_elastic
    sc = sc or Scenario()
    return {f: interface_avo(shale, tuple(float(v) for v in sand_elastic(phi, vsh, f, sc)), angles) for f in fluids}


# ====================================================================== 25c
def ar3_rows(angles, vsvp=0.5):
    """Aki-Richards 3-term coefficients for (dVp/Vp, dVs/Vs, drho/rho) per angle."""
    th = np.radians(np.asarray(angles, float))
    k2 = vsvp ** 2
    s2 = np.sin(th) ** 2
    return np.column_stack([0.5 / np.cos(th) ** 2, -4 * k2 * s2, 0.5 * (1 - 4 * k2 * s2)])


def stack_vs_gather_risk(max_angle=35.0, min_angle=2.0, step=1.0, n_stacks=(2, 3, 5), noise=0.02, vsvp=0.5,
                         prior_sd=0.15, n_mc=0, seed=0):
    """Expected 1-sd errors of the reflectivities of Zp, Zs and density from the full gather (one trace per `step`
    degrees, noise sd `noise` each) and from n equal-width angle stacks of the same gather (noise / sqrt(fold)),
    linear Aki-Richards with a weak Gaussian prior (prior_sd on each reflectivity).  Returns {label: {"zp", "zs",
    "rho"} errors} and, with n_mc, Monte-Carlo errors for checking."""
    angles = np.arange(min_angle, max_angle + 1e-9, step)
    G = ar3_rows(angles, vsvp)
    T = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]], float)  # (Vp, Vs, rho) -> (Zp, Zs, rho) reflectivities

    def post(Gm, sig):
        P = np.linalg.inv(Gm.T @ np.diag(1 / sig ** 2) @ Gm + np.eye(3) / prior_sd ** 2)
        C = T @ P @ T.T
        return P, np.sqrt(np.diag(C))

    designs = {"gather": (G, np.full(len(angles), float(noise)))}
    for n in n_stacks:
        edges = np.linspace(0, len(angles), n + 1).round().astype(int)
        rows, sig = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            rows.append(G[a:b].mean(0))
            sig.append(noise / math.sqrt(b - a))
        designs[f"{n} stacks"] = (np.array(rows), np.array(sig))
    out = {}
    rng = np.random.default_rng(seed)
    for lab, (Gm, sig) in designs.items():
        P, e = post(Gm, sig)
        res = {"zp": float(e[0]), "zs": float(e[1]), "rho": float(e[2])}
        if n_mc:
            errs = []
            for _ in range(n_mc):
                x = rng.normal(0, prior_sd, 3)
                d = Gm @ x + rng.normal(0, sig)
                xh = P @ (Gm.T @ (d / sig ** 2))
                errs.append(T @ (xh - x))
            res["mc"] = np.std(np.array(errs), 0).tolist()
        out[lab] = res
    out["angles"] = (float(min_angle), float(max_angle))
    return out


# ====================================================================== 25b
def fizz_assessment(drho_obs, phi, vsh=0.1, rho_err=None, sc=None, shadow=None, attenuation=None, prior_fizz=0.3):
    """Fizz vs commercial gas.  drho_obs: observed relative density change of the sand vs its brine case (e.g. from
    the pre-stack inversion, -0.07 = 7 % lighter); rho_err: 1-sd error of that (25c, default 0.03).  The expected
    density changes of commercial gas and fizz at this porosity come from Gassmann (predictive.sand_elastic).
    shadow / attenuation: optional 0..1 evidence scores (low-frequency shadow, centroid-frequency drop) - they flag
    gas but do not separate fizz from commercial, so they only raise the gas-vs-wet confidence.
    Returns {p_fizz, score (0 low risk .. 1 high), expected, reasons}."""
    from .predictive import Scenario, sand_elastic
    sc = sc or Scenario()
    rho_err = 0.03 if rho_err is None else float(rho_err)
    vb, _, rb = sand_elastic(phi, vsh, "brine", sc)
    vg, _, rg = sand_elastic(phi, vsh, "gas", sc)
    vf, _, rf = sand_elastic(phi, vsh, "fizz", sc)
    e_gas, e_fizz = float(rg / rb - 1), float(rf / rb - 1)
    lg = math.exp(-0.5 * ((drho_obs - e_gas) / rho_err) ** 2)
    lf = math.exp(-0.5 * ((drho_obs - e_fizz) / rho_err) ** 2)
    p_fizz = prior_fizz * lf / max(1e-300, prior_fizz * lf + (1 - prior_fizz) * lg)
    sep = abs(e_gas - e_fizz) / rho_err
    reasons = [f"Vp drop: gas {100 * (vg / vb - 1):.0f} %, fizz {100 * (vf / vb - 1):.0f} % - P-wave amplitude cannot "
               f"tell them apart",
               f"density drop expected: gas {100 * e_gas:.1f} %, fizz {100 * e_fizz:.1f} %; observed {100 * drho_obs:.1f} "
               f"+- {100 * rho_err:.1f} %",
               f"separation {sep:.1f} sd of the density error" + (" - density too uncertain to discriminate (use more / "
                                                                  "farther angles, see angle-stack risk)" if sep < 1.5
                                                                  else "")]
    if shadow is not None:
        reasons.append(f"low-frequency shadow score {shadow:.2f} (gas indicator, also in fizz)")
    if attenuation is not None:
        reasons.append(f"attenuation score {attenuation:.2f} (gas indicator, also in fizz)")
    conf = min(1.0, sep / 3.0)  # how much the density can be trusted
    score = float(p_fizz * conf + prior_fizz * (1 - conf))
    return {"p_fizz": float(p_fizz), "score": score, "separation_sd": float(sep),
            "expected": {"gas": e_gas, "fizz": e_fizz}, "reasons": reasons}


def lf_shadow(section, dt_ms, t_res, below_ms=(10.0, 60.0), above_ms=(60.0, 10.0), f_low=12.0, f_ref=30.0):
    """Low-frequency shadow / attenuation evidence on traces (ns, n) under a reservoir at times t_res (n,):
    ratio of low-frequency (f_low) to reference-frequency (f_ref) spectral amplitude below vs above the reservoir.
    Returns (shadow ratio per trace, centroid-frequency drop Hz per trace)."""
    from .attributes import spectral
    from .inversion.common import amplitude_spectrum
    x = np.asarray(section, float)
    lo = np.asarray(spectral(x.T, dt_ms, f_low), float).T
    hi = np.asarray(spectral(x.T, dt_ms, f_ref), float).T
    ns, n = x.shape
    t = np.arange(ns) * dt_ms
    ratio, drop = np.full(n, np.nan), np.full(n, np.nan)
    for j in range(n):
        b = (t >= t_res[j] + below_ms[0]) & (t <= t_res[j] + below_ms[1])
        a = (t >= t_res[j] - above_ms[0]) & (t <= t_res[j] - above_ms[1])
        if b.sum() > 4 and a.sum() > 4:
            ratio[j] = (lo[b, j].mean() / max(1e-12, hi[b, j].mean())) / max(1e-12, lo[a, j].mean() / max(1e-12, hi[a, j].mean()))
            fa, Aa = amplitude_spectrum(x[a, j][None], dt_ms, 256)
            fb, Ab = amplitude_spectrum(x[b, j][None], dt_ms, 256)
            drop[j] = float(np.sum(fa * Aa) / max(1e-12, Aa.sum()) - np.sum(fb * Ab) / max(1e-12, Ab.sum()))
    return ratio, drop


# ====================================================================== 25d
DHI_CHECKS = {  # key: (default weight, computed?, description)
    "amplitude": (0.20, True, "amplitude strength vs background"),
    "conformance": (0.20, True, "fit of the anomaly to a structural contour (closure)"),
    "flat_spot": (0.15, False, "flat spot (fluid contact)"),
    "avo": (0.15, True, "AVO behaviour consistent with hydrocarbons"),
    "phase": (0.05, False, "phase / polarity change at the contact"),
    "edges": (0.05, False, "sharp edges / terminations"),
    "stacks": (0.10, True, "consistency across angle stacks"),
    "quality": (0.10, False, "data quality"),
}


def amplitude_score(amp_map, inside, background):
    """Ratio of mean |amp| inside the prospect to the background -> score 0 (ratio 1) .. 1 (ratio >= 3)."""
    a = np.abs(np.asarray(amp_map, float))
    r = np.nanmean(a[inside]) / max(1e-12, np.nanmean(a[background]))
    return float(np.clip((r - 1) / 2.0, 0, 1)), float(r)


def conformance_score(anomaly, depth, levels=None):
    """How well the anomaly outline follows a structural contour: best Jaccard index between the anomaly mask and
    'shallower than level' (depth grid, deeper = larger) over contour levels.  Returns (score 0..1, best level)."""
    an = np.asarray(anomaly, bool)
    d = np.asarray(depth, float)
    fin = np.isfinite(d)
    if not an.any():
        return 0.0, float("nan")
    levels = levels if levels is not None else np.linspace(np.nanmin(d[an]), np.nanmax(d), 60)
    best, lev = 0.0, float("nan")
    for L in levels:
        m = fin & (d <= L)
        inter = np.sum(m & an)
        union = np.sum(m | an)
        j = inter / union if union else 0.0
        if j > best:
            best, lev = j, float(L)
    return float(best), lev


def avo_score(A, B, trend=None, classes_hc=("II", "IIp", "III", "IV")):
    """Mean anomaly below the background trend (sd units, 0 -> 0, -3 -> 1) when the class is a hydrocarbon class."""
    A, B = np.ravel(np.asarray(A, float)), np.ravel(np.asarray(B, float))
    cls = avo_class(np.nanmedian(A), np.nanmedian(B))
    if trend is None:
        return (0.6 if cls in classes_hc else 0.2), cls
    z = float(np.nanmedian(ig_anomaly(A, B, trend)))
    return float(np.clip(-z / 3.0, 0, 1)), cls


def stack_consistency_score(maps):
    """Correlation of the anomaly between angle-stack maps (list of 2D arrays over the prospect area): mean pairwise
    correlation -> 0..1."""
    vs = [np.ravel(np.asarray(m, float)) for m in maps]
    ok = np.all([np.isfinite(v) for v in vs], 0)
    cs = []
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            a, b = vs[i][ok], vs[j][ok]
            if a.std() > 0 and b.std() > 0:
                cs.append(abs(float(np.corrcoef(a, b)[0, 1])))
    return float(np.mean(cs)) if cs else 0.0


def dhi_index(scores, weights=None):
    """Weighted DHI index 0..1 from {check: score 0..1} (missing checks are left out and the weights renormalised).
    Returns {index, used, reasons}."""
    w = {k: (weights or {}).get(k, v[0]) for k, v in DHI_CHECKS.items()}
    used = {k: float(s) for k, s in scores.items() if k in w and s is not None and np.isfinite(s)}
    tot = sum(w[k] for k in used)
    idx = sum(w[k] * used[k] for k in used) / tot if tot else 0.0
    reasons = [f"{DHI_CHECKS[k][2]}: {used[k]:.2f} x weight {w[k]:.2f}" for k in sorted(used, key=lambda k: -w[k] * used[k])]
    return {"index": float(idx), "used": used, "weights": {k: w[k] for k in used}, "reasons": reasons}


def assess_prospect(amp_map, depth_map, inside, background, A=None, B=None, trend=None, stack_maps=None, manual=None,
                    weights=None, threshold=None):
    """All computed checks + interpreter scores (manual {check: 0..1}) -> dhi_index result with the details."""
    amp = np.asarray(amp_map, float)
    s_amp, ratio = amplitude_score(amp, inside, background)
    thr = threshold if threshold is not None else np.nanmean(np.abs(amp[background])) * 1.5
    anomaly = np.abs(amp) > thr
    s_conf, lev = conformance_score(anomaly, depth_map)
    scores = {"amplitude": s_amp, "conformance": s_conf}
    det = {"amp_ratio": ratio, "conformance_level": lev}
    if A is not None and B is not None:
        scores["avo"], det["avo_class"] = avo_score(A, B, trend)
    if stack_maps:
        scores["stacks"] = stack_consistency_score(stack_maps)
    scores.update(manual or {})
    out = dhi_index(scores, weights)
    out["details"] = det
    return out
