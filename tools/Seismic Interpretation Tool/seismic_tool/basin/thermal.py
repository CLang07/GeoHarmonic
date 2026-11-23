"""Thermal history (PLAN 18c): 1D transient heat conduction on the moving burial grid, basal heat flow
(constant, piecewise, McKenzie rift transient), calibration to well temperatures (BHT, Horner).

Grid: every depositional unit is split into `n_sub` intervals that move with the unit (Lagrangian - a node
keeps its material, so no re-interpolation between steps); not-yet-deposited / eroded units have zero
thickness at the surface. Below the sediments a basement slab (default 3 km, k 2.6 W/m/K, A 1 uW/m3) gives
the column thermal inertia; the heat flow is applied at its base. Bulk properties per interval: porosity from
Athy at the interval's compaction depth, k = k_m^(1-phi) k_w^phi (geometric), rho c and A volume-weighted.
Implicit (backward Euler) finite volumes, tridiagonal (Thomas) solve vectorised over a batch of columns.
The system is linear in (surface temperature, radiogenic heat, heat flow) -> calibration by superposition.
"""
import os
import re

import numpy as np

from .burial import WATER

SEC_MY = 3.15576e13  # seconds per million years
BASEMENT = {"thickness": 3000.0, "k": 2.6, "A": 1.0, "rho": 2700.0, "cp": 900.0}


# ====================================================================== heat-flow histories
def mckenzie_heat_flow(ages, rift_age, beta, a=125e3, t_m=1333.0, kappa=1e-6, k=3.1, n_terms=30, present=None):
    """Surface heat flow (mW/m2) after instantaneous lithospheric stretching (McKenzie 1978).
    ages (Ma); before the rift the steady value k Tm / a. present: scale so that 0 Ma has this value."""
    ages = np.asarray(ages, float)
    tau = a * a / (np.pi ** 2 * kappa) / SEC_MY  # My
    t = np.clip(rift_age - ages, 0, None)
    n = np.arange(1, n_terms + 1)[:, None]
    s = 2 * np.sum((beta / (n * np.pi)) * np.sin(n * np.pi / beta) * np.exp(-n * n * t[None, :] / tau), axis=0)
    q = k * t_m / a * (1 + s) * 1e3
    q = np.where(ages > rift_age, k * t_m / a * 1e3, q)
    if present is not None:
        q0 = float(mckenzie_heat_flow(np.array([0.0]), rift_age, beta, a, t_m, kappa, k, n_terms)[0])
        q = q * present / q0
    return q


def heat_flow_history(ages, spec):
    """mW/m2 at the ages from spec: number (constant) | [(age, q)] piecewise linear |
    {"kind": "mckenzie", "rift_age", "beta", "present"} | {"kind": "constant", "q"} | {"kind": "piecewise", "points"}."""
    ages = np.asarray(ages, float)
    if isinstance(spec, dict):
        kind = spec.get("kind", "constant")
        if kind == "mckenzie":
            return mckenzie_heat_flow(ages, float(spec["rift_age"]), float(spec["beta"]),
                                      present=spec.get("present"))
        if kind == "piecewise":
            spec = spec["points"]
        else:
            return np.full(ages.shape, float(spec.get("q", 60.0)))
    if np.ndim(spec) == 0:
        return np.full(ages.shape, float(spec))
    arr = np.asarray(spec, float)
    o = np.argsort(arr[:, 0])
    return np.interp(ages, arr[o, 0], arr[o, 1])


def _series(ages, spec, default):
    if spec is None:
        return np.full(len(ages), float(default))
    if callable(spec):
        return np.array([float(spec(a)) for a in ages])
    if np.ndim(spec) == 0:
        return np.full(len(ages), float(spec))
    arr = np.asarray(spec, float)
    o = np.argsort(arr[:, 0])
    return np.interp(ages, arr[o, 0], arr[o, 1])


# ====================================================================== solver
def thomas(a, b, c, d):
    """Tridiagonal solve along the last axis, batched: a sub, b diag, c super (a[..., 0], c[..., -1] unused)."""
    n = b.shape[-1]
    cp = np.zeros_like(b)
    dp = np.zeros_like(b)
    cp[..., 0] = c[..., 0] / b[..., 0]
    dp[..., 0] = d[..., 0] / b[..., 0]
    for i in range(1, n):
        m = b[..., i] - a[..., i] * cp[..., i - 1]
        cp[..., i] = c[..., i] / m
        dp[..., i] = (d[..., i] - a[..., i] * dp[..., i - 1]) / m
    x = np.zeros_like(b)
    x[..., -1] = dp[..., -1]
    for i in range(n - 2, -1, -1):
        x[..., i] = dp[..., i] - cp[..., i] * x[..., i + 1]
    return x


class _Column:
    """Interval geometry + properties of a batch of burial histories at each age (same units / ages)."""

    def __init__(self, hists, n_sub=6, basement=None, n_base=8, props_override=None):
        batch = np.ndim(getattr(hists, "top", None)) == 3  # burial.BurialBatch
        self.h = hists
        h0 = hists if batch else hists[0]
        self.ages = h0.ages
        self.nu = len(h0.units)
        self.m = n_sub
        self.bs = dict(BASEMENT, **(basement or {}))
        self.nb = n_base
        self.N = self.nu * n_sub + n_base
        p = [u["props"] for u in h0.units]
        self.k_m = np.array([q["k"] for q in p])
        self.A_m = np.array([q["A"] for q in p]) * 1e-6
        self.rc_m = np.array([q["rho"] * q["cp"] for q in p])
        self.phi0 = h0.phi0
        self.c = h0.c
        if batch:
            self.top, self.base = hists.top, hists.base
            self.eff = np.where(np.isfinite(hists.top_eff), hists.top_eff - hists.top, 0.0)
            self.water = hists.water
        else:
            self.top = np.array([h.top for h in hists])  # (B, nt, nu)
            self.base = np.array([h.base for h in hists])
            self.eff = np.array([np.where(np.isfinite(h.top_eff), h.top_eff - h.top, 0.0) for h in hists])
            self.water = np.array([h.water for h in hists])  # (B, nt)

    def at(self, it):
        """(dz, k, rho c, A, z nodes) for age index it; shapes (B, N) / (B, N + 1)."""
        th = np.nan_to_num(self.base[:, it] - self.top[:, it])  # (B, nu)
        B = th.shape[0]
        m = self.m
        dz_u = np.repeat(th / m, m, axis=1)  # (B, nu*m)
        z_top = np.nan_to_num(self.top[:, it])
        frac = (np.arange(m) + 0.5) / m
        zmid = (z_top[:, :, None] + th[:, :, None] * frac[None, None, :]).reshape(B, -1)
        off = np.repeat(self.eff[:, it], m, axis=1)
        phi0 = np.repeat(self.phi0, m)[None, :]
        c = np.repeat(self.c, m)[None, :]
        phi = np.clip(phi0 * np.exp(-c * (zmid + off)), 0, 0.95)
        k = np.repeat(self.k_m, m)[None, :] ** (1 - phi) * WATER["k"] ** phi
        rc = (1 - phi) * np.repeat(self.rc_m, m)[None, :] + phi * WATER["rho"] * WATER["cp"]
        A = (1 - phi) * np.repeat(self.A_m, m)[None, :]
        bs = self.bs
        dzb = np.full((B, self.nb), bs["thickness"] / self.nb)
        dz = np.concatenate([dz_u, dzb], axis=1)
        k = np.concatenate([np.broadcast_to(k, dz_u.shape), np.full((B, self.nb), bs["k"])], axis=1)
        rc = np.concatenate([np.broadcast_to(rc, dz_u.shape), np.full((B, self.nb), bs["rho"] * bs["cp"])], axis=1)
        A = np.concatenate([np.broadcast_to(A, dz_u.shape), np.full((B, self.nb), bs["A"] * 1e-6)], axis=1)
        z = np.concatenate([np.zeros((B, 1)), np.cumsum(dz, axis=1)], axis=1)
        self.phi = np.concatenate([phi, np.zeros((B, self.nb))], axis=1)
        return dz, k, rc, A, z


def _step(T_old, dz, k, rc, A, Ts, q, dt_s):
    """One backward-Euler step (dt_s = inf -> steady state). T_old (B, N+1); Ts (B,), q (B,) W/m2."""
    G = k / np.maximum(dz, 1e-3)  # conductance per interval (B, N)
    C = np.zeros(T_old.shape)
    C[:, :-1] += 0.5 * rc * dz
    C[:, 1:] += 0.5 * rc * dz
    S = np.zeros(T_old.shape)
    S[:, :-1] += 0.5 * A * dz
    S[:, 1:] += 0.5 * A * dz
    inv = 0.0 if not np.isfinite(dt_s) else 1.0 / dt_s
    n = T_old.shape[1]
    a = np.zeros(T_old.shape)
    b = np.zeros(T_old.shape)
    c = np.zeros(T_old.shape)
    d = np.zeros(T_old.shape)
    b[:, 0] = 1.0
    d[:, 0] = Ts
    a[:, 1:] = -G
    c[:, 1:-1] = -G[:, 1:]
    b[:, 1:] = C[:, 1:] * inv + G
    b[:, 1:-1] += G[:, 1:]
    d[:, 1:] = C[:, 1:] * inv * T_old[:, 1:] + S[:, 1:]
    d[:, -1] += q
    if n > 1:
        a[:, 1] = 0.0
        d[:, 1] += G[:, 0] * Ts
    return thomas(a, b, c, d)


def solve(hists, heat_flow=60.0, surface_t=10.0, n_sub=6, basement=None, max_dt=0.5, radiogenic=True,
          cancel=None):
    """Temperature history of a batch of burial histories with identical units / ages.
    heat_flow: mW/m2 spec (see heat_flow_history) or array (B, nt); surface_t: deg C number / [(age, T)] /
    array (B, nt). Returns dict: ages (nt,), z (B, nt, N+1) depth below the sediment surface, T (B, nt, N+1),
    q (B, nt) mW/m2, m (nodes per unit)."""
    col = _Column(hists, n_sub, basement)
    ages = col.ages
    nt = len(ages)
    B = col.top.shape[0]
    q = np.asarray(heat_flow, float) if isinstance(heat_flow, np.ndarray) else heat_flow_history(ages, heat_flow)
    q = np.broadcast_to(q, (B, nt)) * 1e-3
    ts = surface_t if isinstance(surface_t, np.ndarray) else _series(ages, surface_t, 10.0)
    ts = np.broadcast_to(np.asarray(ts, float), (B, nt))
    Tz = np.zeros((B, nt, col.N + 1))
    Z = np.zeros((B, nt, col.N + 1))
    PHI = np.zeros((B, nt, col.N), np.float32)
    dz, k, rc, A, z = col.at(0)
    PHI[:, 0] = col.phi
    if not radiogenic:
        A = A * 0
    T = _step(np.zeros((B, col.N + 1)), dz, k, rc, A, ts[:, 0], q[:, 0], np.inf)  # steady start
    Tz[:, 0], Z[:, 0] = T, z
    for it in range(1, nt):
        if cancel is not None and cancel():
            raise RuntimeError("cancelled")
        dz, k, rc, A, z = col.at(it)
        if not radiogenic:
            A = A * 0
        span = ages[it - 1] - ages[it]
        nsub = max(1, int(np.ceil(span / max_dt)))
        for s in range(nsub):
            w = (s + 1) / nsub
            T = _step(T, dz, k, rc, A, ts[:, it - 1] * (1 - w) + ts[:, it] * w, q[:, it - 1] * (1 - w) + q[:, it] * w,
                      span / nsub * SEC_MY)
        Tz[:, it], Z[:, it] = T, z
        PHI[:, it] = col.phi
    return {"ages": ages, "z": Z, "T": Tz, "q": q * 1e3, "m": n_sub, "water": col.water, "nu": col.nu, "phi": PHI}


class ThermalHistory:
    """Thermal history of one column (wraps `solve` with B = 1)."""

    def __init__(self, hist, heat_flow=60.0, surface_t=10.0, **kw):
        self.hist = hist
        self.heat_flow = heat_flow
        self.surface_t = surface_t
        self.kw = kw
        r = solve([hist], heat_flow, surface_t, **kw)
        self.ages = r["ages"]
        self.z = r["z"][0]
        self.T = r["T"][0]
        self.q = r["q"][0]
        self.m = r["m"]

    def present_profile(self):
        """(depth below sea level, T) today."""
        return self.z[-1] + self.hist.water[-1], self.T[-1]

    def temperature_at(self, depth_ss):
        z, T = self.present_profile()
        return np.interp(np.asarray(depth_ss, float), z, T, left=np.nan, right=np.nan)

    def surface_temperatures(self):
        """[(name, age, T (nt,))] temperature history of each dated surface (top node of each unit)."""
        out = []
        for name, age, _ in self.hist.surfaces():
            i = [j for j, u in enumerate(self.hist.units) if not u["ghost"] and u["top_surface"] == name]
            node = i[0] * self.m if i else len(self.hist.units) * self.m
            T = np.where(self.ages <= age + 1e-9, self.T[:, node], np.nan)
            out.append((name, age, T))
        return out

    def unit_temperature(self, i, frac=0.5):
        """Temperature history at a fraction of unit i's thickness (0 = top, 1 = base), NaN before deposition."""
        u = self.hist.units[i]
        node = i * self.m + frac * self.m
        lo = int(np.floor(node))
        w = node - lo
        T = self.T[:, lo] * (1 - w) + self.T[:, min(lo + 1, self.T.shape[1] - 1)] * w
        return np.where(self.ages <= u["age_base"] + 1e-9, T, np.nan)


# ====================================================================== calibration
def fit_heat_flow(hist, obs, heat_flow=60.0, surface_t=10.0, **kw):
    """Scale the heat-flow history to fit observed temperatures obs [(TVDSS m, T degC)] (exact least squares by
    superposition). Returns {"scale", "heat_flow" (fitted spec or mW/m2 array), "present_q", "rms", "residuals",
    "model" (ThermalHistory)}."""
    obs = np.asarray(obs, float).reshape(-1, 2)
    ages = hist.ages
    q = heat_flow_history(ages, heat_flow)
    r0 = solve([hist], np.zeros_like(q), surface_t, **kw)  # surface temperature + radiogenic heat only
    r1 = solve([hist], q, 0.0, radiogenic=False, **kw)  # unit-scaled heat flow only
    zp = r0["z"][0, -1] + hist.water[-1]
    t0 = np.interp(obs[:, 0], zp, r0["T"][0, -1])
    t1 = np.interp(obs[:, 0], zp, r1["T"][0, -1])
    s = float(np.sum((obs[:, 1] - t0) * t1) / max(1e-12, np.sum(t1 * t1)))
    model = ThermalHistory(hist, q * s, surface_t, **kw)
    pred = model.temperature_at(obs[:, 0])
    res = obs[:, 1] - pred
    return {"scale": s, "heat_flow": q * s, "present_q": float(q[-1] * s), "rms": float(np.sqrt(np.mean(res ** 2))),
            "residuals": res, "model": model}


# ====================================================================== BHT
def horner(times_h, temps, circ_h=4.0):
    """Horner extrapolation of BHTs logged at times since circulation stopped (h) -> static temperature."""
    t = np.asarray(times_h, float)
    T = np.asarray(temps, float)
    x = np.log10((circ_h + t) / t)
    if len(t) < 2 or np.ptp(x) < 1e-6:
        raise ValueError("Horner needs two or more BHTs at different times")
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, T, rcond=None)
    return float(coef[0])


def single_bht_correction(T, surface_t=10.0, tsc_h=None, factor=None):
    """Rule-of-thumb correction of one BHT: T + f (T - Ts), f = 0.15 (< 6 h after circulation), 0.10 (6-12 h),
    0.05 (> 12 h) unless given. Use Horner whenever several logging runs exist at the same depth."""
    if factor is None:
        factor = 0.10 if tsc_h is None else (0.15 if tsc_h < 6 else 0.10 if tsc_h < 12 else 0.05)
    return float(T + factor * (T - surface_t))


_MON = {m: i for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}


def _clock(s):
    """'0730 25MAY' / '1100 20/9' / '745' -> (hours, (day, month) or None)."""
    s = str(s).strip().upper()
    m = re.match(r"^(\d{1,2}):(\d\d)\s*(.*)$", s)
    if m:
        h = int(m.group(1)) + int(m.group(2)) / 60.0
    else:
        m = re.match(r"^(\d{1,4})\s*(.*)$", s)
        if not m:
            return None
        hhmm = m.group(1).zfill(4)
        h = int(hhmm[:2]) + int(hhmm[2:]) / 60.0
    rest = m.groups()[-1].strip()
    d = None
    m2 = re.match(r"^(\d{1,2})\s*([A-Z]{3})", rest)
    m3 = re.match(r"^(\d{1,2})[/.-](\d{1,2})", rest)
    if m2 and m2.group(2) in _MON:
        d = (int(m2.group(1)), _MON[m2.group(2)])
    elif m3:
        d = (int(m3.group(1)), int(m3.group(2)))
    return h, d


def elapsed_hours(tcs, tlab):
    a, b = _clock(tcs), _clock(tlab)
    if a is None or b is None:
        return None
    h = b[0] - a[0]
    if a[1] and b[1]:
        import datetime
        try:
            da = datetime.date(2001, a[1][1], a[1][0])
            db = datetime.date(2001, b[1][1], b[1][0])
            days = (db - da).days
            if days < -300:
                days += 365
            h += 24 * days
        except ValueError:
            pass
    elif h < 0:
        h += 24
    return h if h > 0 else None


def las_temperatures(path):
    """BHT records from a LAS header: [{"file", "md", "T", "tsc_h", "kind"}] (MRT preferred over BHT;
    deg F converted; nulls / < 5 degC dropped)."""
    vals = {}
    curves = set()
    with open(path, "r", errors="replace") as f:
        sec = None
        for line in f:
            s = line.strip()
            if s.startswith("~"):
                sec = s[1:2].upper()
                if sec == "A":
                    break
                continue
            if sec == "C" and s and not s.startswith("#"):
                curves.add(re.split(r"[\s.]", s, 1)[0].upper())
                continue
            if sec != "P" or not s or s.startswith("#"):
                continue
            m = re.match(r"^([^.\s]+)\s*\.(\S*)\s+(.*?)\s*:(.*)$", s)
            if m:
                key = m.group(1).upper()
                if key not in vals:
                    vals[key] = (m.group(2).upper(), m.group(3).strip())
    out = []

    def num(key):
        v = vals.get(key)
        if not v:
            return None, ""
        try:
            return float(v[1].split()[0]), v[0]
        except (ValueError, IndexError):
            return None, v[0]

    if curves & TEMP_CURVES:
        out += _curve_temperature(path, curves & TEMP_CURVES)
    md, mu = num("BLI")
    if md is None or md <= 0:
        md, mu = num("TDL")
    if md is None or md <= 0:
        return out
    if mu.startswith("F"):
        md *= 0.3048
    tsc = elapsed_hours(vals.get("TCS", ("", ""))[1], vals.get("TLAB", ("", ""))[1]) if "TCS" in vals else None
    best = None
    for key in ("MRT", "BHT"):
        t, u = num(key)
        if t is None:
            continue
        if "F" in u and "C" not in u:
            t = (t - 32) / 1.8
        if t >= 5.0 and (best is None or t > best[0]):
            best = (t, key)
    if best:
        out.append({"file": os.path.basename(path), "md": md, "T": best[0], "tsc_h": tsc, "kind": best[1]})
    return out


TEMP_CURVES = {"TEMP", "MTEM", "BHT", "ATMP", "TEMPERATURE"}


def _curve_temperature(path, names):
    """Maximum of a borehole temperature curve (logged fluid temperature) as one BHT-like record."""
    from .. import welltie
    welltie.wc()
    from well_correlation.las import LasFile
    lf = LasFile(path)
    out = []
    for c in lf.curves:
        if c.mnemonic.upper() not in names:
            continue
        v = np.asarray(c.values, float)
        u = (c.unit or "").upper()
        if "F" in u and "C" not in u:
            v = (v - 32) / 1.8
        ok = np.isfinite(v) & (v > 5) & (v < 300)
        if not ok.any():
            continue
        d = np.asarray(lf.depth, float)
        if lf.depth_unit.upper().startswith("F"):
            d = d * 0.3048
        i = np.nonzero(ok)[0][np.argmax(v[ok])]
        out.append({"file": os.path.basename(path), "md": float(d[i]), "T": float(v[i]), "tsc_h": None,
                    "kind": f"curve {c.mnemonic} max"})
    return out


def dlis_temperatures(path):
    """BHT records from DLIS PARAMETER objects (BHT / MRT, BLI / TDL, TCS / TLAB)."""
    from .. import welltie
    welltie.wc()
    from well_correlation.dlis import DlisFile, depth_to_m
    out = []
    for lf in DlisFile(path).logical_files:
        p = {o.name.id.upper(): o for o in lf.parameters}

        def val(k):
            o = p.get(k)
            if o is None:
                return None, ""
            v = o.get("VALUES")
            return (v[0] if isinstance(v, list) and v else v), o.units("VALUES")
        md, mu = val("BLI")
        if not isinstance(md, (int, float)) or md <= 0:
            md, mu = val("TDL")
        if not isinstance(md, (int, float)) or md <= 0:
            continue
        md = float(depth_to_m([md], mu)[0][0])
        best = None
        for key in ("MRT", "BHT"):
            t, u = val(key)
            if not isinstance(t, (int, float)):
                continue
            if "F" in str(u).upper() and "C" not in str(u).upper():
                t = (t - 32) / 1.8
            if t >= 5 and (best is None or t > best[0]):
                best = (float(t), key)
        if best:
            tcs, tlab = val("TCS")[0], val("TLAB")[0]
            tsc = elapsed_hours(str(tcs).strip(), str(tlab).strip()) if tcs and tlab else None
            out.append({"file": os.path.basename(path), "md": md, "T": best[0], "tsc_h": tsc, "kind": best[1]})
    return out


def screen_temperatures(recs, tvdss_of, surface_t=10.0, water_depth=0.0, max_grad=70.0, min_grad=10.0):
    """Flag implausible records: apparent gradient below the sea floor outside [min_grad, max_grad] degC/km
    (copied header blocks, unit errors). Adds "tvdss" and "ok" to each record."""
    for r in recs:
        z = float(tvdss_of(r["md"]))
        r["tvdss"] = z
        below = max(1.0, z - water_depth)
        g = (r["T"] - surface_t) / below * 1000.0
        r["gradient"] = g
        r["ok"] = bool(min_grad <= g <= max_grad)
    return recs


def well_temperatures(las_paths, surface_t=10.0, circ_h=4.0, depth_tol=10.0):
    """Corrected temperatures of a well from its LAS headers: runs at the same depth (within depth_tol) ->
    Horner when two or more logging times exist, else the single-BHT correction.
    Returns [{"md", "T" (corrected), "raw", "method", "n"}] sorted by depth."""
    recs = []
    for p in las_paths:
        try:
            if str(p).lower().endswith(".las"):
                recs += las_temperatures(p)
            elif str(p).lower().endswith(".dlis"):
                recs += dlis_temperatures(p)
        except Exception:  # unreadable header: skip the file
            continue
    recs.sort(key=lambda r: r["md"])
    groups = []
    for r in recs:
        if groups and abs(groups[-1][-1]["md"] - r["md"]) <= depth_tol:
            groups[-1].append(r)
        else:
            groups.append([r])
    out = []
    for g in groups:
        timed = {}
        for r in g:
            if r["tsc_h"]:
                timed[round(r["tsc_h"], 2)] = max(timed.get(round(r["tsc_h"], 2), -1e9), r["T"])
        raw = max(r["T"] for r in g)
        md = float(np.mean([r["md"] for r in g]))
        T, method = None, "single BHT"
        if len(timed) >= 2:
            try:
                T = horner(list(timed), list(timed.values()), circ_h)
                method = f"Horner ({len(timed)} runs)"
                if T < raw:
                    T, method = None, "single BHT"
            except ValueError:
                T = None
        if T is None:
            best = max(g, key=lambda r: r["T"])
            T = single_bht_correction(best["T"], surface_t, best["tsc_h"])
        out.append({"md": md, "T": float(T), "raw": float(raw), "method": method, "n": len(g)})
    return out
