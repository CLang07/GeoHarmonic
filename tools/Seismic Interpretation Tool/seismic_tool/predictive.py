"""Predictive modelling + amplitude statistics + Bayesian fluid classification (PLAN 24c).

Stochastic pseudo-wells (shale / sand / shale; sand thickness, Vsh, porosity and fluid drawn from distributions; sand
elastic properties from a critical-porosity dry frame + Gassmann with Batzle-Wang fluids from the Well Correlation
Tool's petro.rockphysics), angle synthetics (Aki-Richards, exact interface times), and the amplitude statistics at the
top of the sand: near / far amplitude, intercept A and gradient B (two-term Shuey).  A Gaussian Bayesian classifier
trained on those statistics turns real near / far stacks into probability-of-fluid sections, volumes and maps.

    sim = simulate(400, Scenario(), wavelet, dt_ms=2.0)           # table: fluid, phi, h, near, far, A, B, ...
    clf = BayesClassifier().fit(sim["X"], sim["fluid"])           # features A, B (default)
    P = classify_stacks(near, far, clf, angles=(10, 30))         # {fluid: probability} per sample
    names = classify_volumes(project, "Near", "Far", clf)        # probability volumes '<near> [P(gas)]' ...
"""
import math

import numpy as np

QUARTZ = (36.6, 45.0, 2.65)  # K GPa, mu GPa, rho g/cc
CLAY = (21.0, 7.0, 2.58)


def _rp():
    from .welltie import wc
    wc()
    from well_correlation.petro import rockphysics as RP
    return RP


class Scenario:
    """Distributions of the pseudo-wells (uniform ranges unless noted)."""

    def __init__(self, h_m=(2.0, 40.0), phi=(0.12, 0.30), vsh=(0.0, 0.3), fluids=("brine", "gas"), priors=None,
                 shale=(2900.0, 1300.0, 2.45), shale_sd=0.04, sw_hc=0.2, T=80.0, P=25.0, salinity=50000.0, api=35.0,
                 phic=0.40, fizz_sw=0.93, frame_exp=2.0):
        self.h_m, self.phi, self.vsh = h_m, phi, vsh
        self.fluids = tuple(fluids)
        self.priors = priors or {f: 1.0 / len(self.fluids) for f in self.fluids}
        self.shale, self.shale_sd = shale, shale_sd
        self.sw_hc, self.T, self.P, self.salinity, self.api, self.phic, self.fizz_sw = sw_hc, T, P, salinity, api, phic, fizz_sw
        self.frame_exp = frame_exp  # dry frame = mineral * (1 - phi / phic) ** frame_exp (1 = Nur; 2 = softer, Tertiary)


def sand_elastic(phi, vsh, fluid, sc):
    """Vp, Vs (m/s), rho (g/cc) of a sand: critical-porosity dry frame on a quartz-clay VRH mineral + Gassmann with
    brine and the hydrocarbon (fluid brine | gas | oil | fizz) mixed by Wood at the scenario's Sw."""
    RP = _rp()
    phi = np.asarray(phi, float)
    vsh = np.asarray(vsh, float)
    k_min = RP.vrh([1 - vsh, vsh], [QUARTZ[0], CLAY[0]])
    mu_min = RP.vrh([1 - vsh, vsh], [QUARTZ[1], CLAY[1]])
    rho_min = (1 - vsh) * QUARTZ[2] + vsh * CLAY[2]
    f = np.clip(1 - phi / sc.phic, 0.02, 1) ** getattr(sc, "frame_exp", 1.0)
    k_dry, mu_dry = k_min * f, mu_min * f
    rb, _, kb = RP.brine(sc.T, sc.P, sc.salinity)
    if fluid == "brine":
        k_fl, rho_fl = kb, rb
    else:
        if fluid in ("gas", "fizz"):
            rh, kh = RP.gas(sc.T, sc.P)
        else:
            rh, _, kh = RP.oil(sc.T, sc.P, sc.api)
        sw = sc.fizz_sw if fluid == "fizz" else sc.sw_hc
        k_fl, rho_fl = RP.fluid_mix(sw, kb, kh, rb, rh)
    k_sat = RP.gassmann_sat(k_dry, k_min, k_fl, phi)
    rho = (1 - phi) * rho_min + phi * rho_fl
    vp, vs = RP.velocities(k_sat, mu_dry, rho)
    return np.asarray(vp, float), np.asarray(vs, float), np.asarray(rho, float)


def aki_richards(vp1, vs1, rho1, vp2, vs2, rho2, theta_deg):
    """P-P reflection coefficient (Aki-Richards) at angle(s) theta_deg (array ok)."""
    th = np.radians(np.asarray(theta_deg, float))
    vp, vs, rho = 0.5 * (vp1 + vp2), 0.5 * (vs1 + vs2), 0.5 * (rho1 + rho2)
    dvp, dvs, drho = vp2 - vp1, vs2 - vs1, rho2 - rho1
    k2 = (vs / vp) ** 2
    s2 = np.sin(th) ** 2
    return 0.5 * (1 - 4 * k2 * s2) * drho / rho + 0.5 / np.cos(th) ** 2 * dvp / vp - 4 * k2 * s2 * dvs / vs


def shuey_ab(r_near, r_far, th_near, th_far):
    """Intercept A and gradient B of R = A + B sin^2(theta) from two angles (arrays ok)."""
    s1, s2 = math.sin(math.radians(th_near)) ** 2, math.sin(math.radians(th_far)) ** 2
    B = (np.asarray(r_far, float) - np.asarray(r_near, float)) / (s2 - s1)
    A = np.asarray(r_near, float) - B * s1
    return A, B


def interface_trace(times_ms, t_if, rc, wavelet, dt_ms):
    """Trace on times_ms of reflection coefficients rc at exact (sub-sample) interface times t_if (wavelet centred,
    sampled at dt_ms)."""
    w = np.asarray(wavelet, float)
    tw = (np.arange(len(w)) - len(w) // 2) * dt_ms
    out = np.zeros(len(times_ms))
    for t, r in zip(t_if, rc):
        out += r * np.interp(np.asarray(times_ms, float) - t, tw, w, 0, 0)
    return out


def pseudo_well(rng, sc, fluid=None):
    """One pseudo-well: dict(fluid, h, phi, vsh, layers [(vp, vs, rho)] shale / sand / shale)."""
    fluid = fluid or rng.choice(list(sc.fluids), p=np.array([sc.priors[f] for f in sc.fluids]) /
                                sum(sc.priors.values()))
    h = rng.uniform(*sc.h_m)
    phi = rng.uniform(*sc.phi)
    vsh = rng.uniform(*sc.vsh)
    sh = [tuple(v * (1 + rng.normal(0, sc.shale_sd)) for v in sc.shale) for _ in range(2)]
    sand = tuple(float(v) for v in sand_elastic(phi, vsh, fluid, sc))
    return {"fluid": str(fluid), "h": h, "phi": phi, "vsh": vsh, "layers": [sh[0], sand, sh[1]]}


def pw_traces(pw, angles, wavelet, dt_ms, t_top=100.0, n=None):
    """Angle traces (len(angles), n) of a pseudo-well with the sand top at t_top ms; returns (times, traces, t_base)."""
    sh1, sand, sh2 = pw["layers"]
    t_base = t_top + 2000.0 * pw["h"] / sand[0]
    n = n or int((t_top + 200.0) / dt_ms)
    times = np.arange(n) * dt_ms
    tr = []
    for th in angles:
        r1 = aki_richards(*sh1, *sand, th)
        r2 = aki_richards(*sand, *sh2, th)
        tr.append(interface_trace(times, [t_top, t_base], [r1, r2], wavelet, dt_ms))
    return times, np.array(tr), t_base


def simulate(n, sc, wavelet, dt_ms, angles=(10.0, 30.0), noise=0.0, seed=0, t_top=100.0):
    """n pseudo-wells -> amplitude statistics at the sand top (sample at t_top): near, far, A, B, plus sand AI,
    Vp/Vs and the drawn properties.  Returns dict of arrays + "X" (n, 2) = (A, B) features."""
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ("fluid", "h", "phi", "vsh", "near", "far", "A", "B", "ai", "vpvs")}
    k = int(round(t_top / dt_ms))
    for _ in range(n):
        pw = pseudo_well(rng, sc)
        times, tr, _ = pw_traces(pw, angles, wavelet, dt_ms, t_top)
        if noise:
            tr = tr + rng.normal(0, noise, tr.shape)
        near, far = tr[0, k], tr[-1, k]
        A, B = shuey_ab(near, far, angles[0], angles[-1])
        vp, vs, rho = pw["layers"][1]
        for key, v in (("fluid", pw["fluid"]), ("h", pw["h"]), ("phi", pw["phi"]), ("vsh", pw["vsh"]), ("near", near),
                       ("far", far), ("A", float(A)), ("B", float(B)), ("ai", vp * rho), ("vpvs", vp / vs)):
            rows[key].append(v)
    out = {k: np.array(v) for k, v in rows.items()}
    out["X"] = np.column_stack([out["A"], out["B"]])
    out["angles"] = tuple(angles)
    return out


def wedge_section(sc, fluid, phi, h_range, n, wavelet, dt_ms, angles=(10.0, 30.0), t_top=100.0):
    """2D wedge of the sand (thickness h_range over n traces) with a fixed fluid / porosity: angle sections
    (len(angles), ns, n) and the base times."""
    rng = np.random.default_rng(0)
    out, bases = [], []
    sc0 = Scenario(**{**sc.__dict__, "shale_sd": 0.0}) if hasattr(sc, "__dict__") else sc
    for h in np.linspace(h_range[0], h_range[1], n):
        pw = pseudo_well(rng, sc0, fluid)
        pw["h"], pw["phi"] = h, phi
        pw["layers"][1] = tuple(float(v) for v in sand_elastic(phi, pw["vsh"], fluid, sc0))
        times, tr, tb = pw_traces(pw, angles, wavelet, dt_ms, t_top)
        out.append(tr)
        bases.append(tb)
    return times, np.transpose(np.array(out), (1, 2, 0)), np.array(bases)


class BayesClassifier:
    """Gaussian class-conditional densities (full covariance) + priors -> posterior probabilities."""

    def fit(self, X, labels, priors=None):
        X = np.asarray(X, float)
        labels = np.asarray(labels)
        self.classes = sorted(set(labels.tolist()))
        self.mu, self.icov, self.logdet, self.prior = {}, {}, {}, {}
        for c in self.classes:
            Xc = X[labels == c]
            cov = np.cov(Xc.T) + 1e-12 * np.eye(X.shape[1]) * max(1e-30, float(np.trace(np.cov(X.T))))
            self.mu[c] = Xc.mean(0)
            self.icov[c] = np.linalg.inv(cov)
            self.logdet[c] = float(np.linalg.slogdet(cov)[1])
            self.prior[c] = float((priors or {}).get(c, len(Xc) / len(X)))
        return self

    def log_likelihood(self, X):
        X = np.asarray(X, float)
        out = []
        for c in self.classes:
            d = X - self.mu[c]
            m = np.einsum("...i,ij,...j->...", d, self.icov[c], d)
            out.append(-0.5 * m - 0.5 * self.logdet[c] + math.log(max(self.prior[c], 1e-300)))
        return np.stack(out, -1)

    def predict_proba(self, X):
        L = self.log_likelihood(X)
        L = L - np.max(L, -1, keepdims=True)
        P = np.exp(L)
        return P / P.sum(-1, keepdims=True)

    def predict(self, X):
        return np.array(self.classes)[np.argmax(self.predict_proba(X), -1)]

    def to_dict(self):
        return {"classes": self.classes, "mu": {c: self.mu[c].tolist() for c in self.classes},
                "icov": {c: self.icov[c].tolist() for c in self.classes}, "logdet": self.logdet, "prior": self.prior}


def classify_stacks(near, far, clf, angles=(10.0, 30.0), scale=1.0, min_amp=None):
    """Probability per sample of each class from near / far stacks (same shape; scaled to reflectivity units by
    `scale`, e.g. 1 / wavelet scale from the well tie).  Samples with |near| and |far| < min_amp get NaN (no
    reflection to classify).  Returns {class: probability array}."""
    A, B = shuey_ab(np.asarray(near, float) * scale, np.asarray(far, float) * scale, angles[0], angles[-1])
    P = clf.predict_proba(np.stack([A, B], -1))
    out = {}
    weak = None
    if min_amp is not None:
        weak = (np.abs(near) * scale < min_amp) & (np.abs(far) * scale < min_amp)
    for k, c in enumerate(clf.classes):
        p = P[..., k]
        out[c] = np.where(weak, np.nan, p) if weak is not None else p
    return out


def classify_volumes(project, near_name, far_name, clf, angles=(10.0, 30.0), scale=1.0, min_amp=None, progress=None,
                     cancel=None):
    """Probability volumes '<near> [P(<class>)]' from two angle-stack volumes (same geometry)."""
    from .inversion.common import write_volumes
    vn, vf = project.volume(near_name), project.volume(far_name)
    names = [f"P({c})" for c in clf.classes]

    def compute(a, b):
        n_ = np.asarray(vn.ilxl[a:b], np.float32) * vn.inv
        f_ = np.asarray(vf.ilxl[a:b], np.float32) * vf.inv
        P = classify_stacks(n_, f_, clf, angles, scale, min_amp)
        return {f"P({c})": np.nan_to_num(P[c]).astype(np.float32) for c in clf.classes}
    return write_volumes(project, near_name, names, compute, progress=progress, cancel=cancel,
                         meta_extra={"classifier": clf.to_dict(), "angles": list(angles)})


def probability_map(prob_vol, horizon_grid, window_ms=8.0, stat="max"):
    """Probability on a horizon: max (or mean) of the probability volume in +-window_ms around the horizon time."""
    times = np.asarray(prob_vol.times, float)
    z = np.full(horizon_grid.shape, np.nan)
    for i in range(prob_vol.nil):
        row = np.asarray(prob_vol.ilxl[i], np.float32) * prob_vol.inv
        for j in range(prob_vol.nxl):
            t = horizon_grid[i, j]
            if not np.isfinite(t):
                continue
            m = np.abs(times - t) <= window_ms
            if m.any():
                z[i, j] = float(np.max(row[j, m]) if stat == "max" else np.mean(row[j, m]))
    return z


def stats_table(sim):
    """Per-class summary of the simulation (for the stats plots): mean / sd of near, far, A, B, AI, Vp/Vs."""
    rows = []
    for c in sorted(set(sim["fluid"].tolist())):
        m = sim["fluid"] == c
        r = {"fluid": c, "n": int(m.sum())}
        for k in ("near", "far", "A", "B", "ai", "vpvs"):
            r[f"{k}_mean"] = float(np.mean(sim[k][m]))
            r[f"{k}_sd"] = float(np.std(sim[k][m]))
        rows.append(r)
    return rows
