"""Well tie toolkit (PLAN item 16): Hampson-Russell-style auto-correlate, event-pair and constrained dynamic-time-
warping stretch/squeeze, drift QC, checkshot-calibrated sonic, wavelet estimation, multi-well batch.

Conventions (same as welltie.py): the synthetic is on a regular TWT grid t_syn (ms, step dt); a tie maps a raw
synthetic time t to seismic time
    t' = t + shift + (stretch - 1) * (t - t_centre)          (bulk shift + linear stretch about an anchor)
    t'' = t' + interp(t', warp_in, warp_out - warp_in)       (optional piecewise warp: event pairs or DTW)
A positive shift means the seismic event is later than the synthetic one.
"""
import json
import math

import numpy as np


# ====================================================================== helpers
def analytic(x):
    """Analytic signal x + i H(x) via FFT (1-D)."""
    x = np.asarray(x, float)
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(X * h)


def rotate_phase(x, deg):
    """Constant phase rotation by deg degrees (a 90 deg rotation of a zero-phase wavelet gives -H)."""
    z = analytic(x)
    th = math.radians(deg)
    return z.real * math.cos(th) - z.imag * math.sin(th)


def _parabolic(y, k):
    """Sub-sample offset (-0.5..0.5) and value of the peak of y at index k."""
    if 0 < k < len(y) - 1:
        a, b, c = y[k - 1], y[k], y[k + 1]
        den = a - 2 * b + c
        if den < 0:
            off = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
            return off, float(b - 0.25 * (a - c) * off)
    return 0.0, float(y[k])


def _grid_dt(t):
    return float(t[1] - t[0]) if len(t) > 1 else 4.0


def nrms(a, b):
    """Normalised RMS difference in % (0 = identical, 141 = uncorrelated, 200 = opposite) after scaling b to a
    by least squares."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return float("nan")
    bb = float((b * b).sum())
    b = b * (float((a * b).sum()) / bb if bb > 0 else 1.0)
    ra, rb = math.sqrt((a * a).mean()), math.sqrt((b * b).mean())
    return 200.0 * math.sqrt(((a - b) ** 2).mean()) / (ra + rb) if ra + rb > 0 else float("nan")


# ====================================================================== 16a: auto-correlate
def xcorr(f, g_ext, nlag):
    """Normalised cross-correlation of f (n) with every window g_ext[k:k+n], k = 0..2*nlag (FFT numerator,
    cumulative-sum window energies). Returns (corr (2*nlag+1,), A = sum f0*g, B = sum H(f0)*g, |f0|, |g window|)."""
    f = np.asarray(f, float)
    g = np.nan_to_num(np.asarray(g_ext, float))
    n = len(f)
    f0 = f - f.mean()
    hf = analytic(f0).imag
    N = 1 << int(math.ceil(math.log2(len(g) + n)))
    G = np.fft.rfft(g, N)
    A = np.fft.irfft(G * np.conj(np.fft.rfft(f0, N)), N)[:2 * nlag + 1]
    B = np.fft.irfft(G * np.conj(np.fft.rfft(hf, N)), N)[:2 * nlag + 1]
    c1 = np.r_[0.0, np.cumsum(g)]
    c2 = np.r_[0.0, np.cumsum(g * g)]
    k = np.arange(2 * nlag + 1)
    s1 = c1[k + n] - c1[k]
    s2 = c2[k + n] - c2[k]
    gn = np.sqrt(np.maximum(s2 - s1 * s1 / n, 0.0))
    fn = math.sqrt(float((f0 * f0).sum()))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(gn > 0, A / (fn * gn), 0.0) if fn > 0 else np.zeros_like(A)
    return corr, A, B, fn, gn


def _seis_on_grid(t_syn, t_seis, seis, nlag):
    dt = _grid_dt(t_syn)
    tg = t_syn[0] + (np.arange(len(t_syn) + 2 * nlag) - nlag) * dt
    return np.interp(tg, t_seis, seis, left=np.nan, right=np.nan)


def window_mask(t, window):
    if window is None:
        return np.ones(len(t), bool)
    return (t >= min(window)) & (t <= max(window))


def correlate(t_syn, syn, t_seis, seis, max_lag_ms=60.0, window=None, phase_step=5.0, second_peak_frac=0.85):
    """Auto-correlate panel data (HRS 'Correlate' style) for a synthetic and one seismic trace.

    Returns dict: lags_ms, corr (correlation vs lag at the synthetic's phase), best_lag_ms (sub-sample),
    best_corr, polarity ('normal' / 'reverse') with corr_normal / corr_reverse, envelope + env_lag_ms (time
    shift independent of phase), phase scan at that lag (phases_deg, phase_corr), best_phase_deg (rotation of
    the synthetic that maximises the correlation = wavelet phase error, 180 = reversed polarity), second_peak
    (lag, corr) + cycle_skip warning when another peak reaches second_peak_frac of the best."""
    t_syn = np.asarray(t_syn, float)
    m = window_mask(t_syn, window)
    ts, f = t_syn[m], np.asarray(syn, float)[m]
    dt = _grid_dt(t_syn)
    nlag = max(1, int(math.ceil(max_lag_ms / dt)))
    g = _seis_on_grid(ts, np.asarray(t_seis, float), np.asarray(seis, float), nlag)
    corr, A, B, fn, gn = xcorr(f, g, nlag)
    lags = (np.arange(2 * nlag + 1) - nlag) * dt
    kp, kn = int(np.argmax(corr)), int(np.argmin(corr))
    pol = "normal" if corr[kp] >= -corr[kn] else "reverse"
    k = kp if pol == "normal" else kn
    y = corr if pol == "normal" else -corr
    off, val = _parabolic(y, k)
    # constant-phase scan at the envelope peak: the envelope of the cross-correlation |A + iB| peaks at the true
    # time shift whatever the phase, and the rotation that maximises cos*A - sin*B there is atan2(-B, A)
    with np.errstate(invalid="ignore", divide="ignore"):
        env = np.where(gn > 0, np.hypot(A, B) / (fn * gn), 0.0) if fn > 0 else np.zeros_like(A)
    ke = int(np.argmax(env))
    eoff, _ = _parabolic(env, ke)
    Ae = float(np.interp(ke + eoff, np.arange(len(A)), A))
    Be = float(np.interp(ke + eoff, np.arange(len(B)), B))
    ge = float(np.interp(ke + eoff, np.arange(len(gn)), gn))
    phases = np.arange(-180.0, 180.0, phase_step)
    th = np.radians(phases)
    pbest = (np.cos(th) * Ae - np.sin(th) * Be) / (fn * ge) if fn > 0 and ge > 0 else np.zeros(len(phases))
    best_phase = math.degrees(math.atan2(-Be, Ae))
    kph = int(np.argmax(pbest))
    # other peaks (cycle skips): local maxima of y at least half a dominant period away
    peaks = [i for i in range(1, len(y) - 1) if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] > 0]
    sep = max(2, int(round(0.5 * _dominant_period(f, dt) / dt)))
    others = sorted((p for p in peaks if abs(p - k) >= sep), key=lambda p: -y[p])
    second = (float(lags[others[0]]), float(y[others[0]])) if others else None
    return {"lags_ms": lags, "corr": corr, "best_lag_ms": float(lags[k] + off * dt), "best_corr": val,
            "polarity": pol, "corr_normal": float(corr[kp]), "corr_reverse": float(-corr[kn]),
            "phases_deg": phases, "phase_corr": pbest, "best_phase_deg": float(best_phase),
            "best_phase_corr": float(max(pbest[kph], math.hypot(Ae, Be) / (fn * ge) if fn > 0 and ge > 0 else 0.0)),
            "envelope": env, "env_lag_ms": float(lags[ke] + eoff * dt), "second_peak": second,
            "cycle_skip": bool(second and second[1] >= second_peak_frac * val), "n": int(m.sum())}


def _dominant_period(x, dt):
    x = np.asarray(x, float) - np.mean(x)
    if len(x) < 4:
        return 8 * dt
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    fr = np.fft.rfftfreq(len(x), dt)
    sp[0] = 0
    fd = float((fr * sp).sum() / max(sp.sum(), 1e-12))
    return 1.0 / fd if fd > 0 else 8 * dt


def best_trace(t_syn, syn, t_seis, traces, max_lag_ms=60.0, window=None, offsets=None):
    """Best-matching trace among traces near the well (HRS 'best trace within N traces').
    Returns dict(index, offset, corr, lag_ms, all=[corr per trace])."""
    traces = np.atleast_2d(traces)
    res = [correlate(t_syn, syn, t_seis, tr, max_lag_ms, window) for tr in traces]
    cs = np.array([r["corr_normal"] for r in res])
    k = int(np.argmax(cs))
    return {"index": k, "offset": None if offsets is None else offsets[k], "corr": float(cs[k]),
            "lag_ms": res[k]["best_lag_ms"], "all": cs}


def stretch_times(t, stretch, anchor):
    return t + (stretch - 1.0) * (t - anchor)


def auto_tie_fft(t_syn, syn, t_seis, seis, max_shift_ms=100.0, stretches=None, window=None, anchor=None,
                 holdout=0.3):
    """Fast auto tie: for every stretch about the anchor (default: window top) the synthetic is resampled on a
    regular grid and FFT-correlated over all shifts; parabolic refinement in shift and stretch.
    The tie is also fitted on the upper (1 - holdout) part of the window and scored on the held-out lower part
    (honest check that the stretch is not just over-fitting). Returns dict(shift_ms, stretch, t_centre, corr,
    nrms, corr_holdout, nrms_holdout, n)."""
    t_syn = np.asarray(t_syn, float)
    syn = np.asarray(syn, float)
    m = window_mask(t_syn, window)
    ts, f = t_syn[m], syn[m]
    anchor = float(ts[0] if anchor is None else anchor)
    stretches = np.linspace(0.95, 1.05, 21) if stretches is None else np.atleast_1d(np.asarray(stretches, float))

    def fit(ts_, f_):
        dt = _grid_dt(ts_)
        best = None
        rows = []
        for st in stretches:
            tc = stretch_times(ts_, st, anchor)
            tr = np.arange(tc[0], tc[-1] + 1e-9, dt)
            fr = np.interp(tr, tc, f_)
            r = correlate(tr, fr, t_seis, seis, max_shift_ms, None, phase_step=90.0)
            k = int(np.argmax(r["corr"]))
            off, val = _parabolic(r["corr"], k)
            rows.append(val)
            if best is None or val > best[0]:
                best = (val, float(r["lags_ms"][k] + off * dt), float(st))
        val, sh, st = best
        rows = np.array(rows)
        k = int(np.argmax(rows))
        if 0 < k < len(stretches) - 1 and len(stretches) > 2:  # refine the stretch, then its shift
            off, _ = _parabolic(rows, k)
            st2 = float(stretches[k] + off * (stretches[1] - stretches[0]))
            tc = stretch_times(ts_, st2, anchor)
            tr = np.arange(tc[0], tc[-1] + 1e-9, dt)
            r = correlate(tr, np.interp(tr, tc, f_), t_seis, seis, max_shift_ms, None, phase_step=90.0)
            kk = int(np.argmax(r["corr"]))
            off2, v2 = _parabolic(r["corr"], kk)
            if v2 >= val - 1e-6:
                val, sh, st = v2, float(r["lags_ms"][kk] + off2 * dt), st2
        return sh, st, val

    sh, st, val = fit(ts, f)
    out = {"shift_ms": sh, "stretch": st, "t_centre": anchor, "corr": val, "n": int(len(ts))}
    seg = np.interp(stretch_times(ts, st, anchor) + sh, t_seis, seis, left=np.nan, right=np.nan)
    out["nrms"] = nrms(seg, f)
    out["corr_holdout"], out["nrms_holdout"] = float("nan"), float("nan")
    nh = int(round(holdout * len(ts)))
    if holdout and nh >= 10 and len(ts) - nh >= 20:
        sh2, st2, _ = fit(ts[:-nh], f[:-nh])
        th = ts[-nh:]
        seg2 = np.interp(stretch_times(th, st2, anchor) + sh2, t_seis, seis, left=np.nan, right=np.nan)
        out["corr_holdout"] = _pearson(seg2, f[-nh:])
        out["nrms_holdout"] = nrms(seg2, f[-nh:])
    return out


def _pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    d = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


# ====================================================================== 16b: stretch / squeeze
class TieState:
    """Bulk shift + stretch + piecewise warp (event pairs or DTW) with undo. apply(t_raw) -> seismic time."""

    MAX_PAIRS = 10

    def __init__(self, shift=0.0, stretch=1.0, t_centre=0.0):
        self.shift, self.stretch, self.t_centre = float(shift), float(stretch), float(t_centre)
        self.pairs = []  # [(t_tied (after shift/stretch), t_seismic)]
        self.warp_in = None
        self.warp_out = None
        self.warp_kind = None  # "pairs" | "dtw"
        self._undo = []

    # -------------------------------------------------- state
    def snapshot(self):
        return (self.shift, self.stretch, self.t_centre, list(self.pairs),
                None if self.warp_in is None else self.warp_in.copy(),
                None if self.warp_out is None else self.warp_out.copy(), self.warp_kind)

    def _push(self):
        self._undo.append(self.snapshot())

    def undo(self):
        if not self._undo:
            return False
        (self.shift, self.stretch, self.t_centre, self.pairs, self.warp_in, self.warp_out,
         self.warp_kind) = self._undo.pop()
        return True

    def set_bulk(self, shift, stretch, t_centre=None):
        self._push()
        self.shift, self.stretch = float(shift), float(stretch)
        if t_centre is not None:
            self.t_centre = float(t_centre)

    def bulk(self, t):
        return np.asarray(t, float) + self.shift + (self.stretch - 1.0) * (np.asarray(t, float) - self.t_centre)

    def apply(self, t):
        tb = self.bulk(t)
        if self.warp_in is None or len(self.warp_in) == 0:
            return tb
        return tb + np.interp(tb, self.warp_in, self.warp_out - self.warp_in)

    # -------------------------------------------------- event pairs
    def add_pair(self, t_tied, t_seis):
        """Match a synthetic event (time after the bulk tie) to a seismic event. Rejected (False) when it would
        fold time (pairs must keep their order in both columns) or when MAX_PAIRS are set."""
        p = sorted(self.pairs + [(float(t_tied), float(t_seis))])
        a = np.array(p)
        if len(p) > self.MAX_PAIRS or (len(p) > 1 and (np.any(np.diff(a[:, 0]) <= 0) or np.any(np.diff(a[:, 1]) <= 0))):
            return False
        self._push()
        self.pairs = p
        self._pairs_warp()
        return True

    def remove_pair(self, k):
        if not 0 <= k < len(self.pairs):
            return False
        self._push()
        self.pairs = self.pairs[:k] + self.pairs[k + 1:]
        self._pairs_warp()
        return True

    def _pairs_warp(self):
        if not self.pairs:
            self.warp_in = self.warp_out = None
            self.warp_kind = None
            return
        a = np.array(self.pairs, float)
        self.warp_in, self.warp_out, self.warp_kind = a[:, 0], a[:, 1], "pairs"  # flat shift beyond the ends

    def set_warp(self, t_in, t_out, kind="dtw"):
        self._push()
        self.pairs = []
        self.warp_in = np.asarray(t_in, float)
        self.warp_out = np.asarray(t_out, float)
        self.warp_kind = kind

    def clear_warp(self):
        self._push()
        self.pairs = []
        self.warp_in = self.warp_out = None
        self.warp_kind = None

    def to_dict(self):
        d = {"shift_ms": self.shift, "stretch": self.stretch, "t_centre": self.t_centre}
        if self.warp_in is not None:
            d.update(warp_in=[round(float(v), 3) for v in self.warp_in],
                     warp_out=[round(float(v), 3) for v in self.warp_out], warp_kind=self.warp_kind,
                     pairs=[list(p) for p in self.pairs])
        return d

    @classmethod
    def from_dict(cls, d):
        s = cls(d.get("shift_ms", 0.0), d.get("stretch", 1.0), d.get("t_centre", 0.0))
        if d.get("warp_in"):
            s.warp_in, s.warp_out = np.array(d["warp_in"], float), np.array(d["warp_out"], float)
            s.warp_kind = d.get("warp_kind", "dtw")
            s.pairs = [tuple(p) for p in d.get("pairs", [])]
        return s


def dtw_lags(f, g_ext, nlag, max_strain=0.1):
    """Constrained dynamic time warping (Hale 2013 'dynamic warping' with a strain limit).

    f: synthetic samples (n); g_ext: seismic on the same grid extended by nlag samples at both ends (n + 2 nlag).
    Lags (samples) are limited to |lag| <= nlag (Sakoe-Chiba band) and may change by at most one sample every
    b = ceil(1 / max_strain) samples, i.e. the local stretch |d lag / dt| <= max_strain (0.1 = 10 % interval
    velocity change). Returns integer lags (n,)."""
    f = np.asarray(f, float)
    g = np.nan_to_num(np.asarray(g_ext, float))
    n, nl = len(f), 2 * nlag + 1
    fs = f / (np.sqrt((f * f).mean()) or 1.0)
    gs = g / (np.sqrt((g * g).mean()) or 1.0)
    idx = np.arange(n)[:, None] + np.arange(nl)[None, :]
    e = (fs[:, None] - gs[np.clip(idx, 0, len(gs) - 1)]) ** 2  # (n, nl): error of lag l - nlag at sample i
    b = max(1, int(math.ceil(1.0 / max(1e-6, float(max_strain)))))
    ce = np.vstack([np.zeros((1, nl)), np.cumsum(e, axis=0)])  # running sums for the b-1 skipped samples
    D = np.empty_like(e)
    move = np.zeros((n, nl), np.int8)  # 0 = same lag from i-1, -1 / +1 = lag changed coming from i-b
    D[:min(b, n)] = np.cumsum(e[:min(b, n)], axis=0)
    INF = np.inf
    for i in range(b, n):
        stay = D[i - 1]
        run = ce[i] - ce[i - b + 1]  # sum e[i-b+1 .. i-1, l]
        prev = D[i - b]
        dn = np.r_[INF, prev[:-1]] + run  # from lag l-1
        up = np.r_[prev[1:], INF] + run  # from lag l+1
        best = np.minimum(stay, np.minimum(dn, up))
        mv = np.where(best == stay, 0, np.where(best == dn, -1, 1))
        D[i] = e[i] + best
        move[i] = mv
    lags = np.empty(n, int)
    l = int(np.argmin(D[-1]))
    i = n - 1
    while i >= 0:
        lags[i] = l
        if i < b:
            i -= 1
            continue
        mv = int(move[i, l])
        if mv == 0:
            i -= 1
        else:
            for k in range(1, b):
                lags[i - k] = l
            i -= b
            l = l + mv
            l = min(max(l, 0), nl - 1)
    return lags - nlag


def dtw_warp(t_tied, syn, t_seis, seis, max_lag_ms=20.0, max_strain=0.1, smooth=None):
    """DTW stretch/squeeze of the (bulk-tied) synthetic to the seismic. Returns (warp_in, warp_out, lags_ms):
    synthetic tied time -> seismic time, smoothed over b samples so the strain limit holds between samples too."""
    t_tied = np.asarray(t_tied, float)
    dt = _grid_dt(t_tied)
    nlag = max(1, int(math.ceil(max_lag_ms / dt)))
    g = _seis_on_grid(t_tied, np.asarray(t_seis, float), np.asarray(seis, float), nlag)
    lags = dtw_lags(syn, g, nlag, max_strain).astype(float)
    b = max(1, int(math.ceil(1.0 / max(1e-6, max_strain))))
    w = smooth or 2 * b + 1
    if w > 1 and len(lags) > w:
        k = np.ones(w) / w
        pad = np.r_[np.full(w // 2, lags[0]), lags, np.full(w // 2, lags[-1])]
        lags = np.convolve(pad, k, mode="valid")[:len(t_tied)]
    lags = _lipschitz(lags, max_strain)  # smoothing a staircase can exceed the cap locally: enforce it exactly
    lag_ms = lags * dt
    return t_tied, t_tied + lag_ms, lag_ms


def _lipschitz(x, eps):
    """Closest-ish sequence with |x[i+1] - x[i]| <= eps: mean of a forward and a backward clamping pass (each
    satisfies the bound, so their mean does too)."""
    x = np.asarray(x, float)
    f = x.copy()
    for i in range(1, len(f)):
        f[i] = min(max(f[i], f[i - 1] - eps), f[i - 1] + eps)
    b = x.copy()
    for i in range(len(b) - 2, -1, -1):
        b[i] = min(max(b[i], b[i + 1] - eps), b[i + 1] + eps)
    return 0.5 * (f + b)


def max_strain_of(warp_in, warp_out):
    """Largest local |d(lag)/dt| of a warp (0.1 = 10 %)."""
    if warp_in is None or len(warp_in) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(np.asarray(warp_out) - np.asarray(warp_in)) / np.diff(warp_in))))


# ====================================================================== 16b: drift + calibration
def interval_velocity(z, t):
    """Interval velocity (m/s) between consecutive (TVDSS m, TWT ms) points; z mid-points."""
    z = np.asarray(z, float)
    t = np.asarray(t, float)
    dz, dtt = np.diff(z), np.diff(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(dtt > 0, 2000.0 * dz / dtt, np.nan)
    return 0.5 * (z[1:] + z[:-1]), v


def drift_table(td_fn, z_cs, t_cs):
    """QC of a time-depth function (e.g. the tied one) against checkshot levels: rows of (z, t_checkshot,
    t_function, drift = t_function - t_checkshot, v_int checkshot, v_int function) - interval velocities between
    this level and the previous one."""
    z_cs = np.asarray(z_cs, float)
    t_cs = np.asarray(t_cs, float)
    tf = np.asarray(td_fn(z_cs), float)
    _, vc = interval_velocity(z_cs, t_cs)
    _, vf = interval_velocity(z_cs, tf)
    rows = []
    for k in range(len(z_cs)):
        rows.append({"z": float(z_cs[k]), "t_cs": float(t_cs[k]), "t_fn": float(tf[k]),
                     "drift_ms": float(tf[k] - t_cs[k]),
                     "v_cs": float(vc[k - 1]) if k else float("nan"), "v_fn": float(vf[k - 1]) if k else float("nan")})
    return rows


def calibrate_sonic(z_son, t_son, z_cs, t_cs):
    """Checkshot-calibrated sonic: the drift (checkshot - sonic time) at each checkshot level inside the sonic is
    interpolated linearly in depth (constant beyond the end levels) and added to the sonic times, so the result
    honours the checkshots and keeps the sonic's detail between them. Returns (t_cal, drift_at_levels (z, d))."""
    z_son = np.asarray(z_son, float)
    t_son = np.asarray(t_son, float)
    z_cs = np.asarray(z_cs, float)
    t_cs = np.asarray(t_cs, float)
    inside = (z_cs >= z_son[0]) & (z_cs <= z_son[-1])
    if not inside.any():
        d = float(np.interp(z_son[0], z_cs, t_cs) - t_son[0])
        return t_son + d, (np.array([z_son[0]]), np.array([d]))
    zc = z_cs[inside]
    d = t_cs[inside] - np.interp(zc, z_son, t_son)
    t_cal = t_son + np.interp(z_son, zc, d)
    # keep time increasing with depth (a noisy level must not fold the function)
    t_cal = np.maximum.accumulate(t_cal + 1e-9 * np.arange(len(t_cal)))
    return t_cal, (zc, d)


def write_td_csv(path, z, t, well="", note=""):
    """Write a time-depth table in the checkshot CSV format the tools read back (TVD_SRD_m, TWT_ms, OWT_ms)."""
    with open(path, "w", newline="") as fh:
        fh.write(f"# time-depth of {well} {note}\n".replace("\n\n", "\n"))
        fh.write("TVD_SRD_m,TWT_ms,OWT_ms\n")
        for a, b in zip(np.asarray(z, float), np.asarray(t, float)):
            fh.write(f"{a:.2f},{b:.3f},{b / 2:.3f}\n")
    return path


def save_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)


# ====================================================================== 16c: wavelets
def _taper(n_half, frac=0.2):
    """Cosine taper on the outer frac of a (2 n_half + 1) wavelet."""
    n = 2 * n_half + 1
    w = np.ones(n)
    m = max(1, int(round(frac * n_half)))
    ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(m) + 0.5) / m))
    w[:m] = ramp
    w[-m:] = ramp[::-1]
    return w


def conv_matrix(r, n_half):
    """R (n, 2 n_half + 1) with (R w)[i] = sum_k w[k] r[i - (k - n_half)] - 'same' convolution with a centred wavelet."""
    r = np.asarray(r, float)
    n = len(r)
    L = 2 * n_half + 1
    R = np.zeros((n, L))
    for k in range(L):
        lag = k - n_half
        if lag >= 0:
            R[lag:, k] = r[:n - lag]
        else:
            R[:n + lag, k] = r[-lag:]
    return R


def deterministic_wavelet(r, s, n_half, lam=0.01, taper=True):
    """Least-squares (Tikhonov-damped) wavelet w minimising |s - r * w|^2 + lam * trace(R'R)/L * |w|^2, given
    the reflectivity r and the seismic s on the same (tied) time samples. Returns dict(wavelet, pep
    (predictability = 1 - |residual|^2 / |s|^2), nmse, phase_deg)."""
    r = np.asarray(r, float)
    s = np.asarray(s, float)
    ok = np.isfinite(r) & np.isfinite(s)
    r, s = np.where(ok, r, 0.0), np.where(ok, s, 0.0)
    R = conv_matrix(r, n_half)
    G = R.T @ R
    L = G.shape[0]
    G[np.diag_indices(L)] += lam * np.trace(G) / L
    w = np.linalg.solve(G, R.T @ s)
    if taper:
        w = w * _taper(n_half)
    res = s - R @ w
    ss = float((s * s).sum()) or 1.0
    nmse = float((res * res).sum()) / ss
    return {"wavelet": w, "pep": 1.0 - nmse, "nmse": nmse, "phase_deg": wavelet_phase(w)}


def _sine_tapers(n, k):
    i = np.arange(1, n + 1)
    return np.array([np.sqrt(2.0 / (n + 1)) * np.sin(np.pi * (j + 1) * i / (n + 1)) for j in range(k)])


def multitaper_wavelet(r, s, n_half, n_tapers=5, dt_ms=4.0):
    """Wavelet from the multitaper cross-spectrum W(f) = sum S R* / sum |R|^2 (sine tapers) - robust to noise -
    with the coherence of seismic and reflectivity per frequency. Returns dict(wavelet, freqs_hz, coherence)."""
    r = np.nan_to_num(np.asarray(r, float))
    s = np.nan_to_num(np.asarray(s, float))
    n = len(r)
    tp = _sine_tapers(n, n_tapers)
    nf = 1 << int(math.ceil(math.log2(2 * n)))
    Rk = np.fft.rfft(tp * r[None], nf, axis=1)
    Sk = np.fft.rfft(tp * s[None], nf, axis=1)
    srx = (Sk * np.conj(Rk)).sum(0)
    rr = (np.abs(Rk) ** 2).sum(0)
    ss = (np.abs(Sk) ** 2).sum(0)
    W = srx / np.maximum(rr, 1e-12 * rr.max())
    coh = np.abs(srx) ** 2 / np.maximum(rr * ss, 1e-30)
    w = np.fft.irfft(W, nf)
    w = np.r_[w[-n_half:], w[:n_half + 1]] * _taper(n_half)
    return {"wavelet": w, "freqs_hz": np.fft.rfftfreq(nf, dt_ms / 1000.0), "coherence": coh, "phase_deg": wavelet_phase(w)}


def zero_phase_of(w):
    """Zero-phase wavelet with the same amplitude spectrum, centred like w."""
    w = np.asarray(w, float)
    n = len(w)
    c = n // 2
    a = np.abs(np.fft.rfft(np.roll(w, -c)))
    return np.roll(np.fft.irfft(a, n), c)


def wavelet_phase(w):
    """Constant-phase estimate (deg) of a centred wavelet: the rotation of its zero-phase equivalent that fits it
    best (atan2 of the projections on the zero-phase wavelet and its Hilbert transform)."""
    w0 = zero_phase_of(w)
    h = analytic(w0).imag
    A, B = float((w * w0).sum()), float((w * h).sum())
    return math.degrees(math.atan2(-B, A))


def wavelet_spectrum(w, dt_ms, nfft=256):
    """(freqs Hz, amplitude normalised to 1, phase deg about the centre sample)."""
    w = np.asarray(w, float)
    c = len(w) // 2
    nf = max(nfft, 1 << int(math.ceil(math.log2(len(w)))))
    buf = np.zeros(nf)
    buf[:len(w) - c] = w[c:]
    buf[nf - c:] = w[:c]
    W = np.fft.rfft(buf)
    a = np.abs(W)
    return np.fft.rfftfreq(nf, dt_ms / 1000.0), a / (a.max() or 1.0), np.degrees(np.angle(W))


def align_wavelets(ws, max_shift=None):
    """Shift wavelets (same length) to best match the first one (integer samples, circular-safe zero padding) and
    normalise to unit RMS - the input for an average wavelet."""
    ws = [np.asarray(w, float) for w in ws]
    ref = ws[0] / (np.sqrt((ws[0] ** 2).mean()) or 1.0)
    n = len(ref)
    ms = n // 4 if max_shift is None else max_shift
    out = []
    for w in ws:
        w = w / (np.sqrt((w ** 2).mean()) or 1.0)
        best = max(range(-ms, ms + 1), key=lambda k: float((ref * _shift0(w, k)).sum()))
        out.append(_shift0(w, best))
    return out


def _shift0(w, k):
    out = np.zeros_like(w)
    if k >= 0:
        out[k:] = w[:len(w) - k]
    else:
        out[:k] = w[-k:]
    return out


def average_wavelet(ws):
    a = np.mean(align_wavelets(ws), axis=0)
    return a / (np.abs(a).max() or 1.0)


class WaveletLibrary:
    """Named wavelets in one JSON file: {name: {dt_ms, samples, centre, source, ...}}."""

    def __init__(self, path=None):
        self.path = path
        self.items = {}
        if path:
            try:
                with open(path) as fh:
                    self.items = json.load(fh)
            except (OSError, ValueError):
                self.items = {}

    def add(self, name, w, dt_ms, **meta):
        w = np.asarray(w, float)
        self.items[name] = {"dt_ms": float(dt_ms), "samples": [float(v) for v in w], "centre": len(w) // 2, **meta}
        return name

    def get(self, name, dt_ms=None):
        """Wavelet samples, resampled to dt_ms when given."""
        d = self.items[name]
        w = np.array(d["samples"], float)
        if dt_ms is None or abs(dt_ms - d["dt_ms"]) < 1e-9:
            return w
        c = d.get("centre", len(w) // 2)
        t = (np.arange(len(w)) - c) * d["dt_ms"]
        nh = int(t[-1] // dt_ms)
        tn = np.arange(-nh, nh + 1) * dt_ms
        return np.interp(tn, t, w)

    def names(self):
        return sorted(self.items)

    def save(self, path=None):
        self.path = path or self.path
        save_json(self.path, self.items)
        return self.path


# ====================================================================== 16c: log conditioning
def running_median(x, win):
    x = np.asarray(x, float)
    h = win // 2
    p = np.r_[np.full(h, x[0]), x, np.full(h, x[-1])]
    v = np.lib.stride_tricks.sliding_window_view(p, 2 * h + 1)
    return np.nanmedian(v, axis=1)


def despike(x, win=11, n_mad=4.0):
    """Replace samples further than n_mad robust sigmas from the running median by the median.
    Returns (cleaned, spike mask)."""
    x = np.asarray(x, float)
    med = running_median(x, win)
    dev = x - med
    mad = 1.4826 * np.nanmedian(np.abs(dev - np.nanmedian(dev)))
    if not mad > 0:  # smooth log: the running median reproduces it exactly, so fall back to the mean deviation
        mad = 1.2533 * float(np.nanmean(np.abs(dev)))
    bad = np.isfinite(x) & (np.abs(dev) > n_mad * (mad if mad > 0 else np.inf))
    return np.where(bad, med, x), bad


def washout_mask(cali, bit_in, tol_in=1.0, drho=None, drho_max=0.15):
    """True where the hole is washed out (caliper > bit + tol, inches) or the density correction is large."""
    m = np.asarray(cali, float) > float(bit_in) + float(tol_in)
    if drho is not None:
        m |= np.abs(np.asarray(drho, float)) > drho_max
    return m


def backus(z, vp, vs, rho, length):
    """Backus average over a running window of `length` (same units as z, regular sampling assumed):
    vertical P and S velocities of the equivalent medium. Returns (vp_b, vs_b, rho_b)."""
    z = np.asarray(z, float)
    dz = float(np.median(np.diff(z))) if len(z) > 1 else 1.0
    n = max(1, int(round(length / dz)))

    def mavg(a):
        a = np.asarray(a, float)
        c = np.r_[0.0, np.cumsum(a)]
        h = n // 2
        i = np.arange(len(a))
        lo, hi = np.clip(i - h, 0, len(a)), np.clip(i + h + 1, 0, len(a))
        return (c[hi] - c[lo]) / (hi - lo)
    rho_b = mavg(rho)
    M = 1.0 / mavg(1.0 / (np.asarray(rho, float) * np.asarray(vp, float) ** 2))
    mu = 1.0 / mavg(1.0 / (np.asarray(rho, float) * np.asarray(vs, float) ** 2))
    return np.sqrt(M / rho_b), np.sqrt(mu / rho_b), rho_b


# ====================================================================== 16c: angle synthetics
def vs_castagna(vp):
    """Mudrock line (Castagna 1985), m/s."""
    return 0.8621 * np.asarray(vp, float) - 1172.4


def rc_aki_richards(vp, vs, rho, angles_deg):
    """P-P reflection coefficients (n-1 interfaces, n_angles) - Aki-Richards linearisation, angle = incidence
    angle in the upper layer (average angle used in the terms)."""
    vp, vs, rho = (np.asarray(a, float) for a in (vp, vs, rho))
    th1 = np.radians(np.atleast_1d(np.asarray(angles_deg, float)))[None, :]
    a1, a2, b1, b2, r1, r2 = vp[:-1, None], vp[1:, None], vs[:-1, None], vs[1:, None], rho[:-1, None], rho[1:, None]
    p = np.sin(th1) / a1
    th2 = np.arcsin(np.clip(p * a2, -1, 1))
    th = 0.5 * (th1 + th2)
    a, b, r = 0.5 * (a1 + a2), 0.5 * (b1 + b2), 0.5 * (r1 + r2)
    da, db, dr = a2 - a1, b2 - b1, r2 - r1
    k = (b / a) ** 2
    return (0.5 * (1 - 4 * k * np.sin(th) ** 2) * dr / r + 0.5 / np.cos(th) ** 2 * da / a
            - 4 * k * np.sin(th) ** 2 * db / b)


def rc_shuey(vp, vs, rho, angles_deg, terms=3):
    """Shuey (1985) form A + B sin^2 + C (tan^2 - sin^2) from the Aki-Richards intercept / gradient / curvature."""
    vp, vs, rho = (np.asarray(a, float) for a in (vp, vs, rho))
    th = np.radians(np.atleast_1d(np.asarray(angles_deg, float)))[None, :]
    a, b, r = 0.5 * (vp[1:] + vp[:-1]), 0.5 * (vs[1:] + vs[:-1]), 0.5 * (rho[1:] + rho[:-1])
    da, db, dr = np.diff(vp), np.diff(vs), np.diff(rho)
    A = 0.5 * (da / a + dr / r)
    B = 0.5 * da / a - 2 * (b / a) ** 2 * (dr / r + 2 * db / b)
    C = 0.5 * da / a
    out = A[:, None] + B[:, None] * np.sin(th) ** 2
    if terms >= 3:
        out = out + C[:, None] * (np.tan(th) ** 2 - np.sin(th) ** 2)
    return out


def angle_synthetics(vp, vs, rho, angles_deg, wavelet, method="aki-richards"):
    """Angle gather (n, n_angles): reflectivity at each sample boundary (assigned to the lower sample, as
    welltie.synthetic) convolved with the wavelet."""
    f = rc_shuey if method == "shuey" else rc_aki_richards
    rc = f(vp, vs, rho, angles_deg)
    r = np.vstack([np.zeros((1, rc.shape[1])), rc])
    return np.column_stack([np.convolve(r[:, k], wavelet, "same") for k in range(r.shape[1])]), r


def elastic_in_time(well, td, dt_ms):
    """Vp, Vs, rho (blocked per output sample) on a regular TWT axis: Vp from DT, Vs from DTS when the well has
    it else the mudrock line, rho from RHOB else Gardner. Returns (t, vp, vs, rho) or None."""
    if not well.logs or well.depth is None or well.logs.get("DT") is None:
        return None
    md = np.asarray(well.depth, float)
    dtl = np.asarray(well.logs["DT"], float)
    vp = 304800.0 / np.where((dtl > 30) & (dtl < 250), dtl, np.nan)
    dts = well.logs.get("DTS")
    vs = 304800.0 / np.where((np.asarray(dts, float) > 50) & (np.asarray(dts, float) < 600), dts, np.nan) \
        if dts is not None else vs_castagna(vp)
    rho = well.logs.get("RHOB")
    rho = np.where((np.asarray(rho, float) > 1.0) & (np.asarray(rho, float) < 3.5), rho, np.nan) if rho is not None \
        else 0.31 * np.power(vp, 0.25)
    ok = np.isfinite(vp) & np.isfinite(vs) & np.isfinite(rho)
    if ok.sum() < 50:
        return None
    t = td.twt_raw(well.md_to_tvdss(md))
    i0, i1 = np.nonzero(ok)[0][[0, -1]]
    sl = slice(i0, i1 + 1)
    t, ok = t[sl], ok[sl]
    tt = np.arange(np.ceil(t[0] / dt_ms) * dt_ms, t[-1], dt_ms)
    edges = np.r_[tt - dt_ms / 2, tt[-1] + dt_ms / 2]
    out = []
    for a in (vp[sl], vs[sl], rho[sl]):
        a = np.interp(t, t[ok], a[ok])
        cum = np.r_[0.0, np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(t))]
        out.append(np.diff(np.interp(edges, t, cum)) / dt_ms)
    return (tt, *out)


# ====================================================================== 16c: deviated wells + multi-well batch
def seismic_along_wellbore(traj, td, vol, t):
    """Seismic sampled along a deviated wellbore at the synthetic times t: for each time the wellbore position at
    that depth (td.depth -> TVDSS -> x, y on the trajectory), nearest trace, sample at t. NaN outside the survey."""
    t = np.asarray(t, float)
    z = td.depth(t)
    order = np.argsort(traj.z)
    zs = traj.z[order]
    x = np.interp(z, zs, traj.x[order])
    y = np.interp(z, zs, traj.y[order])
    il, xl = vol.ilxl_at(x, y)
    ii = np.rint((np.asarray(il) - vol.il0) / vol.il_step).astype(int)
    jj = np.rint((np.asarray(xl) - vol.xl0) / vol.xl_step).astype(int)
    k = np.rint((t - vol.times[0]) / (vol.times[1] - vol.times[0])).astype(int)
    out = np.full(len(t), np.nan)
    ok = (ii >= 0) & (ii < vol.nil) & (jj >= 0) & (jj < vol.nxl) & (k >= 0) & (k < vol.ns)
    if ok.any():
        out[ok] = np.asarray(vol.ilxl[ii[ok], jj[ok], k[ok]], np.float32) * vol.inv
    return out, x, y


def batch_tie(ws, kind, survey, wells=None, wavelet=None, freq=30.0, max_shift_ms=100.0, stretches=None,
              window=None, n_half_ms=64.0, deviated=False):
    """Tie every well with a sonic near the survey: Ricker (or given) wavelet synthetic, FFT auto tie, then a
    deterministic wavelet from the tied reflectivity. Returns [row] with well, x, y, shift_ms, stretch, t_centre,
    corr, nrms, corr_holdout, wavelet_phase_deg, pep, wavelet (array) - or a row with 'error'."""
    from . import welltie as WT
    dt_ms = survey.dt * 1000.0
    rows = []
    for n in (wells or ws.names):
        w = ws.well(n)
        row = {"well": n, "x": w.x, "y": w.y}
        try:
            if w.x is None:
                raise ValueError("no location")
            ws.wcp.load_well(n)
            td = ws.td_for(n)
            r = WT.logs_in_time(w, td, dt_ms)
            if r is None:
                raise ValueError("no sonic")
            t, ai = r
            wav = WT.ricker(freq, dt_ms) if wavelet is None else wavelet
            rc, syn = WT.synthetic(t, ai, wav)
            if deviated and kind == "3D":
                from .laterals import trajectories_from_wellset
                traj = next((tr for tr in trajectories_from_wellset(ws) if tr.name == n), None)
                ts = survey.times
                seis, _, _ = seismic_along_wellbore(traj, td, survey, ts)
            else:
                ts, trs = ws.trace_at_well(kind, survey, n, strict=True)
                seis = trs[0]
            if not np.isfinite(seis).any():
                raise ValueError("outside the survey")
            seis = np.nan_to_num(seis)
            a = auto_tie_fft(t, syn, ts, seis, max_shift_ms, stretches, window)
            row.update({k: a[k] for k in ("shift_ms", "stretch", "t_centre", "corr", "nrms", "corr_holdout")})
            tt = stretch_times(t, a["stretch"], a["t_centre"]) + a["shift_ms"]
            s_t = np.interp(tt, ts, seis)
            nh = max(2, int(round(n_half_ms / dt_ms)))
            dw = deterministic_wavelet(rc, s_t, nh)
            row.update(wavelet=dw["wavelet"], wavelet_phase_deg=dw["phase_deg"], pep=dw["pep"])
        except Exception as e:  # one bad well must not stop the batch
            row["error"] = str(e)
        rows.append(row)
    return rows


def batch_summary(rows):
    """Average wavelet of the good ties, mean/spread of the bulk shift."""
    good = [r for r in rows if "error" not in r and r.get("corr", 0) > 0.3]
    if not good:
        return {"n": 0}
    sh = np.array([r["shift_ms"] for r in good])
    return {"n": len(good), "shift_mean": float(sh.mean()), "shift_std": float(sh.std()),
            "corr_mean": float(np.mean([r["corr"] for r in good])),
            "average_wavelet": average_wavelet([r["wavelet"] for r in good])}


def shift_map(rows, cell=None):
    """Bulk-shift map (maps.MapGrid, IDW) from the batch rows - a trend means a velocity / datum problem, a
    single odd well a bad tie or time-depth."""
    from .maps import MapGrid, idw_grid
    good = [r for r in rows if "error" not in r and r.get("x") is not None]
    if len(good) < 2:
        return None
    x = np.array([r["x"] for r in good], float)
    y = np.array([r["y"] for r in good], float)
    v = np.array([r["shift_ms"] for r in good], float)
    span = max(np.ptp(x), np.ptp(y), 1.0)
    cell = cell or span / 50.0
    gx, gy, z = idw_grid(x, y, v, cell, reach=span, pad=0.1 * span)
    return MapGrid(z, "xy", "ms", "Well tie bulk shift", None, gx[0], gy[0], cell, cell)


def write_batch_csv(path, rows):
    cols = ["well", "x", "y", "shift_ms", "stretch", "t_centre", "corr", "nrms", "corr_holdout", "wavelet_phase_deg",
            "pep", "error"]
    with open(path, "w", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("" if r.get(c) is None else (f"{r[c]:.4g}" if isinstance(r.get(c), float) else str(r[c]))
                              for c in cols) + "\n")
    return path
