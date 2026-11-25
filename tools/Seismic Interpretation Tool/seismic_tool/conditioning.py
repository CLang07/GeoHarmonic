"""Seismic conditioning workflow (PLAN 41) - post-stack, pure numpy (scipy not needed), out-of-core in bricks.

Steps (each a function on a block (n_il, n_xl, ns) in physical units, dt in ms):
    spikes      spike removal: |x| > N x local RMS (3x3 traces x window) -> median of the neighbouring traces
    bandpass    zero-phase Ormsby (f1-f2-f3-f4) or Butterworth band (low / high corners, order); optional
                spectral balancing (whitening) inside the band, off by default
    fxdecon     f-x Wiener prediction in sliding, overlapping windows (time and space), along crosslines, inlines or
                both; filter length, frequency range, mix-back %
    sof         structure-oriented filter: dip-steered neighbours (gradient-structure-tensor dips, linear
                interpolation in time) in an IL x XL window; median / alpha-trimmed mean / mean; edge protection =
                planarity guard (shrink or skip where confidence is low) + Kuwahara sub-window (lowest variance);
                1-3 passes with dips re-estimated; strength 0-100 %
    footprint   acquisition footprint: per time slice 2D FFT (kx-ky), peaks on the acquisition axes found
                automatically (or user notches), tapered notches, optional shallow time gate - a GLOBAL step (whole
                time slices), run in a second pass over time-slice chunks
    agc         display only (never written unless chosen explicitly)
Chains: [{"type", "on", "params"}]; built-in presets Interpretation / Light (for inversion) / Fault-attribute input;
user presets in project.display['conditioning_presets'].  `apply_chain(block, chain, dt_ms)` on a block,
`run_volume(project, vol_name, chain, ...)` on a whole project volume in bricks (halo = the largest support) -> a new
project volume (original untouched) with the chain + parameters in its history; optional difference volume.
QC: `qc_metrics(original, conditioned)` -> RMS removed %, signal-leakage score, spectra, warnings.
Protect geology: `protect_weights(shape, ...)` from map polygons (il / xl) and time gates -> 0..1 strength weights.
"""
import copy
import json
import os
import shutil
import time

import numpy as np

STEP_TYPES = ("spikes", "bandpass", "fxdecon", "sof", "footprint", "agc")
GLOBAL_STEPS = ("footprint",)
DEFAULTS = {
    "spikes": {"n_rms": 6.0, "window_ms": 100.0},
    "bandpass": {"kind": "ormsby", "f": [4.0, 8.0, 60.0, 80.0], "order": 4, "whiten": False, "whiten_ms": 0.0},
    "fxdecon": {"window_traces": 20, "window_ms": 400.0, "filter_len": 4, "fmin": 3.0, "fmax": 80.0, "mix": 0.0,
                "direction": "both", "eps": 0.01},
    "sof": {"half_il": 2, "half_xl": 3, "filter": "median", "alpha": 0.2, "passes": 1, "sigma_d": 1.0,
            "sigma_s": 2.5, "guard": 0.5, "guard_mode": "shrink", "kuwahara": True, "strength": 1.0},
    "footprint": {"directions": "both", "k_min": 0.08, "threshold": 4.0, "width": 1.5, "t0_ms": None,
                  "t1_ms": None, "notches": []},
    "agc": {"window_ms": 500.0},
}
PRESETS = {
    "Interpretation": [{"type": "spikes", "on": True, "params": {}}, {"type": "bandpass", "on": True, "params": {}},
                       {"type": "fxdecon", "on": True, "params": {}}, {"type": "sof", "on": True, "params": {}},
                       {"type": "footprint", "on": True, "params": {}}],
    "Light (for inversion)": [{"type": "spikes", "on": True, "params": {}},
                              {"type": "bandpass", "on": True, "params": {}}],
    "Fault-attribute input": [{"type": "spikes", "on": True, "params": {}},
                              {"type": "sof", "on": True, "params": {"guard": 0.7, "guard_mode": "skip",
                                                                     "kuwahara": True, "half_il": 2,
                                                                     "half_xl": 2}}],
}
KEY = "conditioning_presets"


def params(step):
    p = copy.deepcopy(DEFAULTS[step["type"]])
    p.update(step.get("params") or {})
    return p


# ====================================================================== spectra + bandpass (41b)
def amplitude_spectrum(x, dt_ms, nfft=None):
    """Mean amplitude spectrum over all traces of x (..., ns): (freqs Hz, amplitude)."""
    a = np.asarray(x, float)
    ns = a.shape[-1]
    nfft = nfft or int(2 ** np.ceil(np.log2(max(8, ns))))
    tr = a.reshape(-1, ns)
    tr = tr - tr.mean(axis=1, keepdims=True)
    A = np.abs(np.fft.rfft(tr * np.hanning(ns), nfft, axis=1)).mean(axis=0)
    return np.fft.rfftfreq(nfft, dt_ms / 1000.0), A


def ormsby_response(f, f1, f2, f3, f4):
    f = np.abs(np.asarray(f, float))
    r = np.zeros_like(f)
    up = (f >= f1) & (f < f2)
    r[up] = (f[up] - f1) / max(1e-9, f2 - f1)
    r[(f >= f2) & (f <= f3)] = 1.0
    dn = (f > f3) & (f <= f4)
    r[dn] = (f4 - f[dn]) / max(1e-9, f4 - f3)
    return r


def butterworth_band_response(f, flow, fhigh, order=4):
    f = np.abs(np.asarray(f, float))
    hi = 1.0 / np.sqrt(1.0 + (f / max(1e-9, fhigh)) ** (2 * order))
    with np.errstate(divide="ignore"):
        lo = np.where(f > 0, 1.0 / np.sqrt(1.0 + (max(1e-9, flow) / np.maximum(f, 1e-12)) ** (2 * order)), 0.0)
    return hi * lo


def design_response(p, f):
    if p["kind"] == "butterworth":
        fl, fh = (p["f"][0], p["f"][-1]) if len(p["f"]) != 2 else p["f"]
        if len(p["f"]) == 4:
            fl, fh = p["f"][1], p["f"][2]
        return butterworth_band_response(f, fl, fh, int(p.get("order", 4)))
    return ormsby_response(f, *p["f"][:4])


def bandpass(x, dt_ms, p):
    """Zero-phase band-pass along the last axis (FFT, padded to avoid wrap-around)."""
    a = np.asarray(x, np.float64)
    ns = a.shape[-1]
    nfft = int(2 ** np.ceil(np.log2(2 * ns)))
    f = np.fft.rfftfreq(nfft, dt_ms / 1000.0)
    H = design_response(p, f)
    X = np.fft.rfft(a, nfft, axis=-1)
    if p.get("whiten"):
        amp = np.abs(X)
        k = max(3, int(len(f) * 0.04))
        ker = np.ones(k) / k
        sm = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), -1, amp)
        band = H > 0.05
        ref = np.median(sm[..., band], axis=-1, keepdims=True) if band.any() else 1.0
        X = X * np.where(band, ref / np.maximum(sm, 1e-12 * (ref if np.ndim(ref) else 1.0) + 1e-30), 1.0)
    return np.fft.irfft(X * H, nfft, axis=-1)[..., :ns]


# ====================================================================== spikes (41c)
def _box_mean(a, half, axis):
    """Running mean (window 2 half + 1, edge-replicated) along one axis."""
    if half <= 0:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (half + 1, half)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * half + 1, 2 * half + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * half + 1)


def neighbour_median(a):
    """Median of the 8 neighbouring traces (3 x 3 minus the centre) at each sample, edges replicated."""
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
    ni, nj = a.shape[:2]
    nb = [p[1 + di:1 + di + ni, 1 + dj:1 + dj + nj] for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    return np.median(np.stack(nb), axis=0)


def remove_spikes(block, dt_ms, p, report=None, offset=(0, 0)):
    """Samples above n_rms x local RMS (3 x 3 traces x window) -> median of the neighbouring traces.  report (list)
    receives (i, j, k) of every replaced sample (global indices with offset)."""
    a = np.asarray(block, np.float64)
    half = max(1, int(round(p["window_ms"] / dt_ms / 2)))
    e = _box_mean(a * a, half, 2)
    p3 = np.pad(e, ((1, 1), (1, 1), (0, 0)), mode="edge")
    ni, nj = a.shape[:2]
    ms = sum(p3[1 + di:1 + di + ni, 1 + dj:1 + dj + nj] for di in (-1, 0, 1) for dj in (-1, 0, 1)) / 9.0
    rms = np.sqrt(np.maximum(ms, 1e-30))
    hit = np.abs(a) > p["n_rms"] * rms
    if not hit.any():
        return a
    out = a.copy()
    out[hit] = neighbour_median(a)[hit]
    if report is not None:
        idx = np.argwhere(hit)
        idx[:, 0] += offset[0]
        idx[:, 1] += offset[1]
        report.extend(map(tuple, idx.tolist()))
    return out


# ====================================================================== FX deconvolution (41d)
def _windows(n, w):
    """Window starts (50 % overlap, anchored at 0) covering 0..n-1, and the window length."""
    w = int(max(2, min(w, n)))
    step = max(1, w // 2)
    starts = list(range(0, max(1, n - w + 1), step))
    if starts[-1] + w < n:
        starts.append(n - w)
    return starts, w


def _ar_predict(S, L, eps):
    """Forward + backward linear prediction of complex series S (nf, nx) with an order-L filter per frequency."""
    nf, nx = S.shape
    if nx <= L + 1:
        return S.copy()

    def fwd(s):
        A = np.stack([s[:, L - 1 - k:nx - 1 - k] for k in range(L)], axis=2)  # (nf, nx-L, L)
        b = s[:, L:]
        AH = np.conj(np.transpose(A, (0, 2, 1)))
        R = AH @ A
        lam = eps * np.real(np.trace(R, axis1=1, axis2=2))[:, None, None] / L + 1e-30
        a = np.linalg.solve(R + lam * np.eye(L)[None], (AH @ b[:, :, None]))[:, :, 0]
        pred = np.zeros_like(s)
        pred[:, L:] = np.einsum("fnk,fk->fn", A, a)
        return pred
    pf = fwd(S)
    pb = fwd(S[:, ::-1])[:, ::-1]
    out = np.zeros_like(S)
    w = np.zeros(nx)
    w[L:] += 1
    out[:, L:] += pf[:, L:]
    wb = np.zeros(nx)
    wb[:nx - L] += 1
    out[:, :nx - L] += pb[:, :nx - L]
    return out / np.maximum(w + wb, 1)[None, :]


def fx_section(sec, dt_ms, p, x_offset=0, x_total=None):
    """f-x decon of a section (n_traces, ns) in overlapping windows (Hann-weighted overlap-add).  Spatial windows
    are laid out on the GLOBAL trace axis (x_offset = first trace, x_total = traces in the whole line) so a brick
    with a halo of one window gives exactly the whole-line result."""
    a = np.asarray(sec, np.float64)
    nx, ns = a.shape
    wt = int(max(16, round(p["window_ms"] / dt_ms)))
    t_starts, wt = _windows(ns, wt)
    xt = nx if x_total is None else int(x_total)
    g_starts, wx = _windows(xt, int(p["window_traces"]))
    x_starts = [g - x_offset for g in g_starts if g >= x_offset and g + wx <= x_offset + nx]
    if not x_starts:
        x_starts, wx = _windows(nx, int(p["window_traces"]))
    out = np.zeros_like(a)
    wsum = np.zeros_like(a)
    tw = np.hanning(wt + 2)[1:-1] if len(t_starts) > 1 else np.ones(wt)
    xw = np.hanning(wx + 2)[1:-1] if len(g_starts) > 1 else np.ones(wx)
    W = xw[:, None] * tw[None, :]
    nfft = int(2 ** np.ceil(np.log2(wt)))
    f = np.fft.rfftfreq(nfft, dt_ms / 1000.0)
    band = (f >= p["fmin"]) & (f <= p["fmax"])
    L = int(p["filter_len"])
    for x0 in x_starts:
        for t0 in t_starts:
            blk = a[x0:x0 + wx, t0:t0 + wt]
            D = np.fft.rfft(blk, nfft, axis=1)  # (wx, nf)
            P = D.copy()
            if band.any():
                P[:, band] = _ar_predict(D[:, band].T, L, float(p.get("eps", 0.01))).T
            P[:, ~band] = 0.0
            rec = np.fft.irfft(P, nfft, axis=1)[:, :wt]
            out[x0:x0 + wx, t0:t0 + wt] += rec * W
            wsum[x0:x0 + wx, t0:t0 + wt] += W
    res = out / np.maximum(wsum, 1e-12)
    m = float(p.get("mix", 0.0))
    return (1 - m) * res + m * a


def fxdecon(block, dt_ms, p, il_offset=0, il_total=None):
    a = np.asarray(block, np.float64)
    d = p.get("direction", "both")
    outs = []
    if d in ("crossline", "both", "xline"):
        outs.append(np.stack([fx_section(a[i], dt_ms, p) for i in range(a.shape[0])]))
    if d in ("inline", "both") and (il_total or a.shape[0]) > int(p["filter_len"]) + 2:
        outs.append(np.stack([fx_section(a[:, j], dt_ms, p, il_offset, il_total) for j in range(a.shape[1])], axis=1))
    return np.mean(outs, axis=0) if outs else a


# ====================================================================== dips + SOF (41e, 41f)
def dips(block, sigma_d=1.0, sigma_s=2.5, sigma_t=None):
    """Gradient structure tensor (item 3 rgt.dip): inline / crossline dip (samples per trace) + planarity 0..1."""
    from .rgt.dip import dip_3d_block
    return dip_3d_block(np.asarray(block, np.float32), sigma_d, sigma_s, sigma_t if sigma_t else 2 * sigma_s)


def dip_attributes(pi, pj, dt_ms, il_spacing=25.0, xl_spacing=25.0):
    """Dip azimuth (deg, from +inline axis) and magnitude (ms per m) from the two dips."""
    gi = np.asarray(pi, float) * dt_ms / il_spacing
    gj = np.asarray(pj, float) * dt_ms / xl_spacing
    return np.degrees(np.arctan2(gj, gi)) % 360.0, np.hypot(gi, gj)


def _shifted(a, di, dj, shift):
    """Neighbour trace (i+di, j+dj) (edge-replicated) sampled at t + shift (samples, linear interpolation)."""
    ni, nj, ns = a.shape
    ii = np.clip(np.arange(ni) + di, 0, ni - 1)
    jj = np.clip(np.arange(nj) + dj, 0, nj - 1)
    nb = a[ii][:, jj]
    tk = np.arange(ns)[None, None, :] + shift
    tk = np.clip(tk, 0, ns - 1)
    k0 = np.floor(tk).astype(np.int64)
    k1 = np.minimum(k0 + 1, ns - 1)
    w = tk - k0
    return np.take_along_axis(nb, k0, 2) * (1 - w) + np.take_along_axis(nb, k1, 2) * w


def _reduce(stack, how, alpha):
    if how == "mean":
        return stack.mean(axis=0)
    if how == "alpha":
        s = np.sort(stack, axis=0)
        n = len(s)
        k = int(np.floor(alpha * n))
        return s[k:n - k].mean(axis=0) if n - 2 * k > 0 else np.median(s, axis=0)
    return np.median(stack, axis=0)


SOF_MAX_MB = 256.0


def sof_pass(a, pi, pj, plan, p, max_mb=None):
    """One structure-oriented filter pass on a block (dips at the centre trace steer every neighbour).  The
    neighbour gather is built for a few inlines at a time (memory <= max_mb), so F3-size lines never blow up RAM."""
    hi, hj = int(p["half_il"]), int(p["half_xl"])
    n_off = (2 * hi + 1) * (2 * hj + 1)
    ni = a.shape[0]
    per_row = n_off * a.shape[1] * a.shape[2] * 4 / 1e6
    ch = int(max(1, (max_mb or SOF_MAX_MB) // max(1e-9, per_row) - 2 * hi))
    if ch < ni:
        out = np.empty(a.shape, np.float32)
        for c0 in range(0, ni, ch):
            c1 = min(ni, c0 + ch)
            s0, s1 = max(0, c0 - hi), min(ni, c1 + hi)
            r = _sof_pass(a[s0:s1], pi[s0:s1], pj[s0:s1], plan[s0:s1], p)
            out[c0:c1] = r[c0 - s0:c1 - s0]
        return out
    return _sof_pass(a, pi, pj, plan, p)


def _sof_pass(a, pi, pj, plan, p):
    hi, hj = int(p["half_il"]), int(p["half_xl"])
    offs = [(di, dj) for di in range(-hi, hi + 1) for dj in range(-hj, hj + 1)]
    stack = np.stack([_shifted(a, di, dj, pi * di + pj * dj) for di, dj in offs]).astype(np.float32)
    how, alpha = p.get("filter", "median"), float(p.get("alpha", 0.2))
    if p.get("kuwahara", True):
        # four overlapping quadrant sub-windows (each holds the centre): keep the one with the lowest variance
        quads = [[k for k, (di, dj) in enumerate(offs) if sdi * di >= 0 and sdj * dj >= 0]
                 for sdi in (-1, 1) for sdj in (-1, 1)]
        var = np.stack([stack[q].var(axis=0) for q in quads])
        best = np.argmin(var, axis=0)
        filt = np.zeros(a.shape, np.float32)
        for k, q in enumerate(quads):
            m = best == k
            if m.any():
                filt[m] = _reduce(stack[q][:, m], how, alpha)
    else:
        filt = _reduce(stack, how, alpha)
    g = float(p.get("guard", 0.0))
    if g > 0:
        low = plan < g
        if low.any():
            if p.get("guard_mode", "shrink") == "skip":
                filt[low] = a[low]
            else:  # shrink to the 3 x 3 neighbourhood (still dip-steered)
                small = [k for k, (di, dj) in enumerate(offs) if abs(di) <= 1 and abs(dj) <= 1]
                filt[low] = np.median(stack[small][:, low], axis=0)
    return filt


def sof(block, dt_ms, p, weights=None):
    """Structure-oriented filter: passes with dips re-estimated, strength blend, protect weights (0 = untouched)."""
    a = np.asarray(block, np.float32)
    x = a.copy()
    for _ in range(int(np.clip(p.get("passes", 1), 1, 3))):
        pi, pj, pl = dips(x, p["sigma_d"], p["sigma_s"])
        x = sof_pass(x, pi, pj, pl, p)
    s = float(np.clip(p.get("strength", 1.0), 0, 1))
    if weights is not None:
        s = s * np.asarray(weights, np.float32)
    return a + s * (x - a)


def sof_halo(p):
    """Traces of support one SOF needs on each side (dips + window, per pass)."""
    sup = int(np.ceil(3 * p["sigma_d"])) + int(np.ceil(3 * p["sigma_s"])) + 1
    return int(np.clip(p.get("passes", 1), 1, 3)) * (sup + max(int(p["half_il"]), int(p["half_xl"])))


# ====================================================================== footprint (41g)
def footprint_mask(spec_mag, p, shape):
    """Notch mask (1 = keep) on an fftshifted kx-ky magnitude: peaks on the acquisition axes (kx = 0 / ky = 0) that
    stand above `threshold` x the median along that axis, beyond k_min (cycles / trace), plus user notches
    [(ky, kx)] in cycles / trace.  Gaussian tapers of `width` bins, symmetric."""
    ni, nj = shape
    ky = np.fft.fftshift(np.fft.fftfreq(ni))
    kx = np.fft.fftshift(np.fft.fftfreq(nj))
    cy, cx = ni // 2, nj // 2
    peaks = []
    d = p.get("directions", "both")
    if d in ("both", "inline"):  # stripes repeating along the inline axis -> peaks on the ky axis (kx = 0)
        ax = spec_mag[:, cx]
        bg = np.median(ax) + 1e-30
        for k in np.flatnonzero((ax > p["threshold"] * bg) & (np.abs(ky) >= p["k_min"])):
            peaks.append((ky[k], 0.0))
    if d in ("both", "crossline"):
        ax = spec_mag[cy, :]
        bg = np.median(ax) + 1e-30
        for k in np.flatnonzero((ax > p["threshold"] * bg) & (np.abs(kx) >= p["k_min"])):
            peaks.append((0.0, kx[k]))
    for n in p.get("notches") or []:
        peaks.append((float(n[0]), float(n[1])))
    mask = np.ones(shape)
    Y, X = np.meshgrid((np.arange(ni) - cy), (np.arange(nj) - cx), indexing="ij")
    w = float(p.get("width", 1.5))
    for qy, qx in peaks:
        for sy, sx in ((qy, qx), (-qy, -qx)):
            by, bx = sy * ni, sx * nj
            mask *= 1.0 - np.exp(-0.5 * (((Y - by) / w) ** 2 + ((X - bx) / w) ** 2))
    return mask, peaks


def remove_footprint(cube, dt_ms, p, t0_ms=0.0):
    """cube (n_il, n_xl, ns) holding WHOLE time slices.  Detects the footprint on the mean spectrum of the gate,
    notches every slice in the gate.  Returns (cube, peaks)."""
    a = np.asarray(cube, np.float64)
    ns = a.shape[2]
    t = t0_ms + dt_ms * np.arange(ns)
    g = np.ones(ns, bool)
    if p.get("t0_ms") is not None:
        g &= t >= float(p["t0_ms"])
    if p.get("t1_ms") is not None:
        g &= t <= float(p["t1_ms"])
    ks = np.flatnonzero(g)
    if not len(ks):
        return a, []
    F = np.fft.fftshift(np.fft.fft2(np.moveaxis(a[:, :, ks], 2, 0)), axes=(1, 2))
    mag = np.abs(F).mean(axis=0)
    mask, peaks = footprint_mask(mag, p, a.shape[:2])
    out = a.copy()
    if peaks:
        rec = np.real(np.fft.ifft2(np.fft.ifftshift(F * mask[None], axes=(1, 2))))
        out[:, :, ks] = np.moveaxis(rec, 0, 2)
    return out, peaks


# ====================================================================== AGC (display only, 41h)
def agc(x, dt_ms, window_ms=500.0):
    a = np.asarray(x, np.float64)
    half = max(1, int(round(window_ms / dt_ms / 2)))
    rms = np.sqrt(np.maximum(_box_mean(a * a, half, a.ndim - 1), 1e-30))
    return a / rms


# ====================================================================== protect geology (41j)
def point_in_poly(x, y, px, py):
    x, y = np.asarray(x, float), np.asarray(y, float)
    inside = np.zeros(np.broadcast(x, y).shape, bool)
    n = len(px)
    for k in range(n):
        x1, y1, x2, y2 = px[k], py[k], px[(k + 1) % n], py[(k + 1) % n]
        c = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-30) + x1)
        inside ^= c
    return inside


def protect_weights(shape, il_idx=None, xl_idx=None, t_ms=None, zones=()):
    """0..1 strength weights (n_il, n_xl, ns): zones = [{"polygon": [(il, xl), ...] | None, "t0", "t1", "weight"}]
    (polygon in inline / crossline numbers; None = the whole survey; weight 0 = skip, 0.3 = reduced)."""
    ni, nj, ns = shape
    il_idx = np.arange(ni) if il_idx is None else np.asarray(il_idx, float)
    xl_idx = np.arange(nj) if xl_idx is None else np.asarray(xl_idx, float)
    t_ms = np.arange(ns) if t_ms is None else np.asarray(t_ms, float)
    w = np.ones(shape, np.float32)
    I, J = np.meshgrid(il_idx, xl_idx, indexing="ij")
    for z in zones or ():
        if z.get("polygon"):
            poly = np.asarray(z["polygon"], float)
            m2 = point_in_poly(I, J, poly[:, 0], poly[:, 1])
        else:
            m2 = np.ones((ni, nj), bool)
        mt = (t_ms >= float(z.get("t0", -np.inf))) & (t_ms <= float(z.get("t1", np.inf)))
        w = np.where(m2[:, :, None] & mt[None, None, :], np.minimum(w, float(z.get("weight", 0.0))), w)
    return w


# ====================================================================== chains
def validate_chain(chain):
    out = []
    for s in chain:
        if s.get("type") not in STEP_TYPES:
            raise ValueError(f"unknown conditioning step {s.get('type')!r}")
        out.append({"type": s["type"], "on": bool(s.get("on", True)), "params": dict(s.get("params") or {})})
    return out


def preset(name, project=None):
    user = ((getattr(project, "display", None) or {}).get(KEY) or {}) if project is not None else {}
    if name in user:
        return validate_chain(user[name])
    if name in PRESETS:
        return validate_chain(copy.deepcopy(PRESETS[name]))
    raise KeyError(name)


def preset_names(project=None):
    user = ((getattr(project, "display", None) or {}).get(KEY) or {}) if project is not None else {}
    return list(PRESETS) + [n for n in user if n not in PRESETS]


def save_preset(project, name, chain):
    disp = getattr(project, "display", None)
    if disp is None:
        project.display = disp = {}
    disp.setdefault(KEY, {})[str(name)] = json.loads(json.dumps(validate_chain(chain), default=float))
    return disp[KEY][str(name)]


def chain_halo(chain):
    """Lateral support (traces) the local steps of a chain need around a brick."""
    h = 0
    for s in chain:
        if not s.get("on", True):
            continue
        p = params(s)
        if s["type"] == "spikes":
            h += 1
        elif s["type"] == "fxdecon":
            h += int(p["window_traces"]) if p.get("direction", "both") in ("both", "inline") else 0
        elif s["type"] == "sof":
            h += sof_halo(p)
    return h


def apply_chain(block, chain, dt_ms, weights=None, report=None, offset=(0, 0), t0_ms=0.0, include_global=True,
                progress=None, il_total=None):
    """Run the enabled steps on a block.  Global steps (footprint) run only when include_global (whole slices).
    report: dict collecting spike locations / footprint peaks.  Returns float32."""
    x = np.asarray(block, np.float64)
    rep = report if report is not None else {}
    steps = [s for s in validate_chain(chain) if s["on"]]
    for n, s in enumerate(steps):
        p = params(s)
        t = s["type"]
        if t == "spikes":
            sp = rep.setdefault("spikes", [])
            x = remove_spikes(x, dt_ms, p, sp, offset)
        elif t == "bandpass":
            x = bandpass(x, dt_ms, p)
        elif t == "fxdecon":
            x = fxdecon(x, dt_ms, p, offset[0], il_total)
        elif t == "sof":
            x = sof(x, dt_ms, p, weights)
        elif t == "footprint" and include_global:
            x, peaks = remove_footprint(x, dt_ms, p, t0_ms)
            rep.setdefault("footprint_peaks", []).extend(peaks)
        elif t == "agc" and p.get("write"):
            x = agc(x, dt_ms, p["window_ms"])
        if progress:
            progress((n + 1) / len(steps), t)
    if weights is not None and not any(s["type"] == "sof" for s in steps):
        x = np.asarray(block, np.float64) + np.asarray(weights) * (x - np.asarray(block, np.float64))
    return x.astype(np.float32)


# ====================================================================== QC (41i)
def local_correlation(a, b, half=(2, 2, 12)):
    """Windowed correlation coefficient of two blocks (box windows)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)

    def box(v):
        for ax, h in enumerate(half[:v.ndim]):
            v = _box_mean(v, int(h), ax)
        return v
    ma, mb = box(a), box(b)
    cov = box(a * b) - ma * mb
    va = box(a * a) - ma * ma
    vb = box(b * b) - mb * mb
    return cov / np.sqrt(np.maximum(va * vb, 1e-30))


def qc_metrics(original, conditioned, dt_ms=4.0, rms_warn=35.0, leak_warn=0.35):
    """RMS removed %, signal-leakage score (mean |local correlation| of the difference with the conditioned data -
    high when the difference contains reflectors), before / after spectra, warnings."""
    o = np.asarray(original, np.float64)
    c = np.asarray(conditioned, np.float64)
    d = c - o
    rms_o = float(np.sqrt(np.mean(o * o))) or 1e-30
    removed = 100.0 * float(np.sqrt(np.mean(d * d))) / rms_o
    half = (1, 1, 10) if o.ndim == 3 else (2, 10)
    leak = float(np.mean(np.abs(local_correlation(d, c, half))))
    fo, ao = amplitude_spectrum(o, dt_ms)
    _, ac = amplitude_spectrum(c, dt_ms)
    warns = []
    if removed > rms_warn:
        warns.append(f"{removed:.0f} % of the RMS removed (> {rms_warn:.0f} %): too aggressive?")
    if leak > leak_warn:
        warns.append(f"signal leakage {leak:.2f} (> {leak_warn:.2f}): the difference shows coherent reflectors")
    return {"rms_removed_pct": removed, "leakage": leak, "freqs": fo, "spec_before": ao, "spec_after": ac,
            "warnings": warns}


def snr_db(clean, noisy):
    clean = np.asarray(clean, float)
    n = np.asarray(noisy, float) - clean
    return 10 * np.log10(np.mean(clean ** 2) / max(1e-30, np.mean(n ** 2)))


# ====================================================================== whole volume (41k)
def run_volume(project, vol_name, chain, out_name=None, brick=8, progress=None, cancel=None, diff=False,
               zones=(), max_mb=256):
    """Condition a project volume in inline bricks with halos (local steps), then the global steps over time-slice
    chunks; writes a NEW store + project volume (original never changed) whose entry carries the chain + parameters
    in 'history'.  Returns {name, diff, spikes, footprint_peaks, seconds}."""
    from .volume import STORE_VERSION, Volume3D, unused_store
    t_start = time.time()
    chain = validate_chain(chain)
    vol = project.volume(vol_name)
    dt_ms = vol.dt * 1000.0
    t0_ms = float(vol.times[0])
    local = [s for s in chain if s["type"] not in GLOBAL_STEPS]
    glob = [s for s in chain if s["type"] in GLOBAL_STEPS and s["on"]]
    halo = chain_halo(local)
    # brick + halo held in memory as float64 (+ dips): keep it near max_mb x 4; the SOF gather is chunked itself
    per_il = vol.nxl * vol.ns * 8 * 5 / 1e6
    brick = int(max(1, min(int(brick), max(1, int(4 * max_mb // max(1e-9, per_il)) - 2 * halo))))
    out_name = out_name or f"{vol_name} [conditioned]"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in out_name)
    store = unused_store(os.path.join(os.path.dirname(vol.store), safe))
    os.makedirs(store, exist_ok=True)
    raw = np.lib.format.open_memmap(os.path.join(store, "raw.tmp.npy"), mode="w+", dtype=np.float32,
                                    shape=(vol.nil, vol.nxl, vol.ns))
    report = {"spikes": [], "footprint_peaks": []}
    ils = np.asarray(vol.inlines, float)
    xls = np.asarray(vol.xlines, float)

    def stop():
        nonlocal raw
        del raw
        shutil.rmtree(store, ignore_errors=True)
        raise RuntimeError("cancelled")
    for a in range(0, vol.nil, brick):
        if cancel and cancel():
            stop()
        b = min(vol.nil, a + brick)
        a0, b0 = max(0, a - halo), min(vol.nil, b + halo)
        blk = np.asarray(vol.ilxl[a0:b0], np.float32) * vol.inv
        w = protect_weights(blk.shape, ils[a0:b0], xls, np.asarray(vol.times, float), zones) if zones else None
        res = apply_chain(blk, local, dt_ms, w, report, (a0, 0), t0_ms, include_global=False, il_total=vol.nil)
        raw[a:b] = res[a - a0:a - a0 + (b - a)]
        if progress:
            el = time.time() - t_start
            frac = b / vol.nil * (0.8 if glob else 0.95)
            progress(frac, f"Conditioning: inline {b}/{vol.nil} - about {el / max(1e-9, frac) * (1 - frac):.0f} s left")
    report["spikes"] = [s for s in report["spikes"]]
    if glob:  # whole time slices, in sample chunks
        step = max(1, int(64e6 // max(1, vol.nil * vol.nxl * 8)))
        for k0 in range(0, vol.ns, step):
            if cancel and cancel():
                stop()
            k1 = min(vol.ns, k0 + step)
            cube = np.asarray(raw[:, :, k0:k1], np.float64)
            for s in glob:
                cube, peaks = remove_footprint(cube, dt_ms, params(s), t0_ms + k0 * dt_ms)
                report["footprint_peaks"].extend(peaks)
            raw[:, :, k0:k1] = cube.astype(np.float32)
            if progress:
                progress(0.8 + 0.15 * k1 / vol.ns, f"Footprint: samples {k1}/{vol.ns}")
    raw.flush()
    name = _write_store(project, vol, raw, store, out_name, chain, zones)
    dname = None
    if diff:
        dstore = unused_store(os.path.join(os.path.dirname(vol.store), safe + "_difference"))
        os.makedirs(dstore, exist_ok=True)
        draw = np.lib.format.open_memmap(os.path.join(dstore, "raw.tmp.npy"), mode="w+", dtype=np.float32,
                                         shape=(vol.nil, vol.nxl, vol.ns))
        for a in range(0, vol.nil, 16):
            b = min(vol.nil, a + 16)
            draw[a:b] = np.asarray(raw[a:b]) - np.asarray(vol.ilxl[a:b], np.float32) * vol.inv
        draw.flush()
        dname = _write_store(project, vol, draw, dstore, out_name + " (difference)", chain, zones, difference=True)
        del draw
        os.remove(os.path.join(dstore, "raw.tmp.npy"))
    del raw
    os.remove(os.path.join(store, "raw.tmp.npy"))
    if progress:
        progress(1.0, "Conditioning done")
    Volume3D(store)
    return {"name": name, "diff": dname, "spikes": report["spikes"], "footprint_peaks": report["footprint_peaks"],
            "seconds": time.time() - t_start}


def _write_store(project, vol, raw, store, name, chain, zones, difference=False):
    from .volume import STORE_VERSION
    samples = np.abs(np.asarray(raw[::max(1, vol.nil // 8), ::7, ::5])).ravel()
    p999 = float(np.percentile(samples, 99.9)) if samples.size else 1.0
    scale = 1000.0 / (p999 or 1.0)
    out = np.lib.format.open_memmap(os.path.join(store, "data_ilxl.npy"), mode="w+", dtype=np.float16,
                                    shape=(vol.nil, vol.nxl, vol.ns))
    for a in range(0, vol.nil, 16):
        b = min(vol.nil, a + 16)
        out[a:b] = np.clip(np.asarray(raw[a:b]) * scale, -60000, 60000).astype(np.float16)
    out.flush()
    tm = np.lib.format.open_memmap(os.path.join(store, "data_t.npy"), mode="w+", dtype=np.float16,
                                   shape=(vol.ns, vol.nil, vol.nxl))
    for a in range(0, vol.nil, 16):
        b = min(vol.nil, a + 16)
        tm[:, a:b, :] = np.transpose(np.asarray(out[a:b]), (2, 0, 1))
    tm.flush()
    del tm, out
    filled = os.path.join(vol.store, "filled.npy")
    if os.path.exists(filled):
        shutil.copy(filled, os.path.join(store, "filled.npy"))
    hist = {"step": "conditioning", "source": getattr(vol, "name", None) or os.path.basename(vol.store),
            "chain": json.loads(json.dumps(validate_chain(chain), default=float)),
            "zones": json.loads(json.dumps(list(zones or []), default=float)), "difference": bool(difference),
            "date": time.strftime("%Y-%m-%d %H:%M")}
    meta = dict(vol.meta)
    meta.update({"version": STORE_VERSION, "scale": scale, "conditioning": hist})
    with open(os.path.join(store, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    src = next((v for v in project.volumes if os.path.normpath(v.get("store", "")) ==
                os.path.normpath(vol.store)), {})
    history = list(src.get("history") or []) + [hist]
    project.volumes = [v for v in project.volumes if v["name"] != name]
    getattr(project, "_vol_cache", {}).pop(name, None)
    project.volumes.append({"name": name, "store": store, "segy": "", "source": src.get("name"),
                            "history": history})
    return name


def history(project, name):
    """Conditioning history of a project volume (volume pickers show it)."""
    v = next((v for v in project.volumes if v["name"] == name), None)
    return list((v or {}).get("history") or [])


def history_text(project, name):
    h = history(project, name)
    if not h:
        return ""
    steps = [s["type"] for s in h[-1]["chain"] if s["on"]]
    return " > ".join(steps)


def has_sof(project, name):
    """True when the volume went through a structure-oriented filter (the inversion package warns: use a Light
    volume - SOF changes amplitudes)."""
    return any(s["type"] == "sof" and s["on"] for e in history(project, name) for s in e.get("chain", []))
