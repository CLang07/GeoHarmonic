"""Seismic resolution and single-well inversion (PLAN 24a / 24b).

24a - what frequency does the data have and what does the reservoir need:
    st = spectrum_stats(traces, dt_ms)                 # dominant / centroid frequency, usable band vs noise floor
    g = frequency_map(vol, t0_ms, t1_ms)               # dominant frequency per block of traces (MapGrid, Hz)
    q4, q8 = tuning_thickness(v_int, f_dom)            # lambda / 4, lambda / 8 (m)
    w = wedge_model(np.arange(0, 60, 0.5), wavelet, dt)  # apparent thickness + composite amplitude -> tuning
    beds = bed_thicknesses(depth, flag)                # bed thickness distribution from a log flag (20a cut-offs)
    rec = recommend(st, v_int, beds, dt_ms)            # needed vs available frequency, gain from spectral balancing

24b - can the seismic see this reservoir property at this well?
    r = single_well_inversion(trace, ai_log, wavelet, dt_ms, n_mc=40, prop=phie_log)
    -> P10 / P50 / P90 AI (and property via the well AI-property regression), coverage of the log, correlation gain
       over the low-frequency model alone and a plain-English verdict.
"""
import math

import numpy as np

from .inversion.common import amplitude_spectrum, butter_filter


# ====================================================================== 24a spectrum
def _smooth(a, n=5):
    n = max(1, int(n) | 1)
    return np.convolve(np.pad(a, n // 2, mode="edge"), np.ones(n) / n, "valid")


def spectrum_stats(traces, dt_ms, noise_traces=None, snr=2.0, noise_band=0.8):
    """Average amplitude spectrum of traces (m, ns) and its usable band: frequencies where the spectrum is at least
    snr x the noise floor (noise_traces' spectrum, or the median of the spectrum above noise_band x Nyquist).
    Returns {freqs, amp, noise, f_dom, f_centroid, f_lo, f_hi, octaves, snr_db}."""
    f, A = amplitude_spectrum(traces, dt_ms)
    As = _smooth(A, max(3, len(A) // 64))
    if noise_traces is not None:
        fn, An = amplitude_spectrum(noise_traces, dt_ms, nfft=2 * (len(f) - 1))
        noise = _smooth(np.interp(f, fn, An), max(3, len(A) // 32))
    else:
        tail = f >= noise_band * f[-1]
        noise = np.full(len(f), float(np.median(As[tail])) if tail.any() else 0.0)
    k = int(np.argmax(As * (f > 0)))
    ok = As >= snr * np.maximum(noise, 1e-30)
    lo = k
    while lo > 1 and ok[lo - 1]:
        lo -= 1
    hi = k
    while hi < len(f) - 1 and ok[hi + 1]:
        hi += 1
    band = slice(lo, hi + 1)
    p = As[band] ** 2
    cen = float(np.sum(f[band] * p) / max(1e-30, p.sum()))
    f_lo, f_hi = float(f[lo]), float(f[hi])
    return {"freqs": f, "amp": A, "noise": noise, "f_dom": float(f[k]), "f_centroid": cen, "f_lo": f_lo, "f_hi": f_hi,
            "octaves": float(math.log2(f_hi / max(f_lo, 1e-6))) if f_hi > 0 else 0.0,
            "snr_db": float(20 * math.log10(As[k] / max(1e-30, noise[k])))}


def frequency_map(vol, t0_ms=None, t1_ms=None, block=4, stat="f_dom"):
    """Dominant (or centroid / f_hi) frequency in a time window per block x block traces -> MapGrid on the survey."""
    from .maps import MapGrid
    dt = vol.dt * 1000.0
    times = np.asarray(vol.times, float)
    a = int(np.searchsorted(times, t0_ms)) if t0_ms is not None else 0
    b = int(np.searchsorted(times, t1_ms)) if t1_ms is not None else len(times)
    z = np.full((vol.nil, vol.nxl), np.nan)
    for i in range(0, vol.nil, block):
        for j in range(0, vol.nxl, block):
            tr = np.asarray(vol.ilxl[i:i + block, j:j + block, a:b], np.float32).reshape(-1, b - a) * vol.inv
            tr = tr[np.any(tr != 0, axis=1)]
            if len(tr) and b - a >= 16:
                z[i:i + block, j:j + block] = spectrum_stats(tr, dt)[stat]
    return MapGrid(z, "ilxl", "Hz", f"{stat} {t0_ms}-{t1_ms} ms", vol)


def tuning_thickness(v_int, f_hz):
    """(lambda / 4, lambda / 8) in metres for interval velocity v_int (m/s) and frequency f_hz (arrays ok)."""
    lam = np.asarray(v_int, float) / np.maximum(np.asarray(f_hz, float), 1e-6)
    return lam / 4.0, lam / 8.0


def wavelet_from_spectrum(freqs, amp, dt_ms, n=None):
    """Zero-phase wavelet with the given amplitude spectrum, peak 1, odd length n (default 129)."""
    n = n or 129
    nf = 2 * (len(freqs) - 1)
    w = np.fft.irfft(np.asarray(amp, float), nf)
    w = np.roll(w, n // 2)[:n]
    return w / max(1e-30, np.max(np.abs(w)))


def band_wavelet(f_lo, f_hi, dt_ms, n=129, taper_hz=5.0):
    """Zero-phase band-pass (flat f_lo..f_hi with cosine tapers) - the spectrally balanced wavelet."""
    nf = 1024
    f = np.fft.rfftfreq(nf, dt_ms / 1000.0)
    A = np.zeros(len(f))
    A[(f >= f_lo) & (f <= f_hi)] = 1.0
    lo = (f >= f_lo - taper_hz) & (f < f_lo)
    A[lo] = 0.5 - 0.5 * np.cos(np.pi * (f[lo] - (f_lo - taper_hz)) / taper_hz)
    hi = (f > f_hi) & (f <= f_hi + taper_hz)
    A[hi] = 0.5 + 0.5 * np.cos(np.pi * (f[hi] - f_hi) / taper_hz)
    return wavelet_from_spectrum(f, A, dt_ms, n)


def wedge_model(thick_ms, wavelet, dt_ms, rc_top=0.1, rc_base=-0.1, oversample=8):
    """Wedge of two-way thicknesses thick_ms between a top (rc_top) and base (rc_base) reflector (sub-sample exact:
    the wavelet is interpolated).  Returns {thick_ms, amp (max |trace| per thickness), apparent_ms (top peak to base
    trough), tuning_ms (thickness of the maximum amplitude), amp_ratio (tuning amp / thick-bed amp)}."""
    w = np.asarray(wavelet, float)
    tw = (np.arange(len(w)) - len(w) // 2) * dt_ms
    th = np.asarray(thick_ms, float)
    t = np.arange(-len(w) * dt_ms, (len(w) + 1) * dt_ms + th.max(), dt_ms / oversample)
    amp, app = [], []
    for h in th:
        tr = rc_top * np.interp(t, tw, w, 0, 0) + rc_base * np.interp(t - h, tw, w, 0, 0)
        amp.append(float(np.max(np.abs(tr))))
        k1 = int(np.argmax(tr * np.sign(rc_top)))
        k2 = int(np.argmax(tr * np.sign(rc_base)))
        app.append(float(t[k2] - t[k1]))
    amp, app = np.array(amp), np.array(app)
    k = int(np.argmax(amp))
    return {"thick_ms": th, "amp": amp, "apparent_ms": app, "tuning_ms": float(th[k]),
            "amp_ratio": float(amp[k] / max(1e-30, amp[-1]))}


def bed_thicknesses(depth, flag, min_samples=1):
    """Thicknesses (depth units) of the continuous True runs of flag (e.g. Vsh < cut-off) along depth."""
    d = np.asarray(depth, float)
    f = np.asarray(flag, bool) & np.isfinite(d)
    if len(d) < 2 or not f.any():
        return np.zeros(0)
    step = float(np.median(np.diff(d)))
    edges = np.diff(np.concatenate([[0], f.astype(int), [0]]))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    n = ends - starts
    return (n[n >= min_samples] * step).astype(float)


def spectral_balance(traces, dt_ms, f_lo, f_hi, smooth_hz=8.0, taper_hz=5.0):
    """Whitening inside the usable band: divide each trace's spectrum by the smoothed average spectrum and apply the
    balanced band (band_wavelet's spectrum).  Returns balanced traces scaled to the input RMS."""
    tr = np.atleast_2d(np.asarray(traces, float))
    ns = tr.shape[1]
    nf = int(2 ** np.ceil(np.log2(ns * 2)))
    F = np.fft.rfft(tr, nf, axis=1)
    f = np.fft.rfftfreq(nf, dt_ms / 1000.0)
    avg = np.abs(F).mean(axis=0)
    df = f[1] - f[0]
    avg = _smooth(avg, max(3, int(smooth_hz / df)))
    target = np.zeros(len(f))
    target[(f >= f_lo) & (f <= f_hi)] = 1.0
    lo = (f >= f_lo - taper_hz) & (f < f_lo)
    target[lo] = 0.5 - 0.5 * np.cos(np.pi * (f[lo] - (f_lo - taper_hz)) / taper_hz)
    hi = (f > f_hi) & (f <= f_hi + taper_hz)
    target[hi] = 0.5 + 0.5 * np.cos(np.pi * (f[hi] - f_hi) / taper_hz)
    out = np.fft.irfft(F * (target / np.maximum(avg, 1e-6 * avg.max()))[None], nf, axis=1)[:, :ns]
    r_in, r_out = np.sqrt(np.mean(tr ** 2)), np.sqrt(np.mean(out ** 2))
    return out * (r_in / max(r_out, 1e-30))


def recommend(stats, v_int, beds_m=None, dt_ms=2.0, max_thick_m=None):
    """Needed vs available resolution.  stats: spectrum_stats; v_int: reservoir interval velocity (m/s); beds_m: bed
    thicknesses from the logs.  Tuning thicknesses come from wedge models with the current wavelet (data spectrum) and
    with the spectrally balanced wavelet (flat over the usable band)."""
    v = float(v_int)
    w_now = wavelet_from_spectrum(stats["freqs"], _smooth(stats["amp"], max(3, len(stats["amp"]) // 64)), dt_ms)
    w_bal = band_wavelet(stats["f_lo"], stats["f_hi"], dt_ms)
    t_max = max(4 * 1000.0 / max(stats["f_lo"], 5.0), 50.0)
    th = np.arange(0.0, t_max, dt_ms / 4)
    wn, wb = wedge_model(th, w_now, dt_ms), wedge_model(th, w_bal, dt_ms)
    tun_now = wn["tuning_ms"] / 2000.0 * v  # two-way ms -> m
    tun_bal = wb["tuning_ms"] / 2000.0 * v
    q4, q8 = tuning_thickness(v, stats["f_dom"])
    out = {"f_dom": stats["f_dom"], "f_centroid": stats["f_centroid"], "f_lo": stats["f_lo"], "f_hi": stats["f_hi"],
           "lambda4_m": float(q4), "lambda8_m": float(q8), "tuning_m": float(tun_now), "tuning_balanced_m": float(tun_bal),
           "gain_pct": float(100.0 * (1 - tun_bal / max(tun_now, 1e-9))), "wedge_now": wn, "wedge_balanced": wb}
    if beds_m is not None and len(beds_m):
        b = np.asarray(beds_m, float)
        p50 = float(np.median(b))
        out.update(bed_p10=float(np.percentile(b, 10)), bed_p50=p50, bed_p90=float(np.percentile(b, 90)),
                   f_needed=float(v / (4 * p50)),  # lambda / 4 = median bed
                   resolved_now=float(np.mean(b >= tun_now)), resolved_balanced=float(np.mean(b >= tun_bal)))
        out["verdict"] = (f"median bed {p50:.1f} m needs ~{out['f_needed']:.0f} Hz (lambda/4); data dominant "
                          f"{stats['f_dom']:.0f} Hz, usable to {stats['f_hi']:.0f} Hz; tuning {tun_now:.1f} m now, "
                          f"{tun_bal:.1f} m after spectral balancing ({out['gain_pct']:.0f} % thinner); beds resolved "
                          f"{100 * out['resolved_now']:.0f} % -> {100 * out['resolved_balanced']:.0f} %")
    else:
        out["verdict"] = (f"dominant {stats['f_dom']:.0f} Hz, usable {stats['f_lo']:.0f}-{stats['f_hi']:.0f} Hz; tuning "
                          f"{tun_now:.1f} m now, {tun_bal:.1f} m after spectral balancing")
    return out


# ====================================================================== 24b single-well inversion
def rotate_phase(w, deg):
    """Constant phase rotation of a wavelet (Hilbert transform)."""
    from .attributes import analytic
    w = np.asarray(w, float)
    h = np.imag(analytic(w[None, :], -1))[0]
    a = math.radians(deg)
    return w * math.cos(a) - h * math.sin(a)


def stretch(w, s):
    """Wavelet with its frequencies scaled by s (s > 1 = higher frequency): w(t * s)."""
    w = np.asarray(w, float)
    t = np.arange(len(w)) - len(w) // 2
    return np.interp(t * s, t, w, 0, 0)


def single_well_inversion(trace, ai_log, wavelet, dt_ms, n_mc=40, fc=8.0, lam=0.05, phase_sd=15.0, freq_sd=0.08,
                          scale_sd=0.1, fc_sd=2.0, noise_rms=None, prop=None, fmax=None, seed=0):
    """Model-based inversion of the seismic trace at a well with Monte-Carlo over the wavelet (phase, frequency,
    scale), the low-frequency model cut-off and the noise.  ai_log on the trace's time axis (NaN outside the log).
    prop: optional reservoir property log (same axis) - mapped from AI through the well's linear regression.
    Returns {t index mask, p10, p50, p90 (AI), lfm, coverage, corr_inv, corr_lfm, corr_gain, verdict, [prop_*]}."""
    from .inversion.poststack import invert_traces, synthetic_ln, scale_wavelet
    rng = np.random.default_rng(seed)
    s = np.asarray(trace, float)
    ai = np.asarray(ai_log, float)
    ok = np.isfinite(ai) & (ai > 0)
    if ok.sum() < 3 * len(wavelet) // 2:
        raise ValueError("log too short for a single-well inversion")
    idx = np.arange(len(ai))
    z_log = np.interp(idx, idx[ok], np.log(ai[ok]))
    w0, _ = scale_wavelet(wavelet, s, ai)
    fmax = fmax or 1000.0 / (dt_ms * 4)
    if noise_rms is None:
        z_base = invert_traces(s[None], butter_filter(z_log, dt_ms, fc)[None], w0, lam)
        noise_rms = float(np.std((s - synthetic_ln(z_base, w0)[0])[ok]))
    reals = []
    lfms = []
    for _ in range(n_mc):
        w = rotate_phase(w0, rng.normal(0, phase_sd))
        w = stretch(w, max(0.5, 1 + rng.normal(0, freq_sd))) * max(0.3, 1 + rng.normal(0, scale_sd))
        f_c = max(2.0, fc + rng.normal(0, fc_sd))
        prior = butter_filter(z_log, dt_ms, f_c)
        sn = s + rng.normal(0, noise_rms, len(s))
        reals.append(invert_traces(sn[None], prior[None], w, lam)[0])
        lfms.append(prior)
    Z = np.exp(np.array(reals))
    p10, p50, p90 = np.percentile(Z, [10, 50, 90], axis=0)
    lfm = np.exp(butter_filter(z_log, dt_ms, fc))  # the nominal low-frequency model
    h = len(wavelet) // 2
    m = ok.copy()
    i0, i1 = np.nonzero(ok)[0][[0, -1]]
    m[:i0 + h] = False
    m[i1 - h + 1:] = False  # away from the log ends (no reflection data there)
    band_log = butter_filter(z_log, dt_ms, fmax)

    def cc(a, b):
        a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
        return float(np.sum(a * b) / max(1e-30, math.sqrt(np.sum(a * a) * np.sum(b * b))))
    # correlation of the relative (band) part: inversion vs LFM alone, against the band-limited log
    rel_log = band_log - butter_filter(z_log, dt_ms, fc)
    c_inv = cc(butter_filter(np.log(p50), dt_ms, fmax) - butter_filter(np.log(p50), dt_ms, fc), rel_log)
    c_lfm = cc(np.log(lfm), band_log) if np.std(np.log(lfm)[m]) > 0 else 0.0  # trend only
    c_full = cc(np.log(p50), band_log)
    ai_band = np.exp(band_log)
    cover = float(np.mean((ai_band[m] >= p10[m] * 0.98) & (ai_band[m] <= p90[m] * 1.02)))
    out = {"mask": m, "p10": p10, "p50": p50, "p90": p90, "lfm": lfm, "ai_band": ai_band, "coverage": cover,
           "corr_inv": c_inv, "corr_lfm": c_lfm, "corr_full": c_full, "corr_gain": c_full - c_lfm, "noise_rms": noise_rms,
           "rel_width": float(np.median((p90[m] - p10[m]) / p50[m])), "n_mc": n_mc}
    if prop is not None:
        pr = np.asarray(prop, float)
        okp = m & np.isfinite(pr)
        if okp.sum() > 5:
            A = np.column_stack([np.log(ai[okp] if np.all(np.isfinite(ai[okp])) else np.exp(z_log[okp])), np.ones(okp.sum())])
            coef, *_ = np.linalg.lstsq(A, pr[okp], rcond=None)
            res_sd = float(np.std(pr[okp] - A @ coef))
            conv = lambda x: coef[0] * np.log(x) + coef[1]  # noqa: E731
            q = np.sort(np.stack([conv(p10), conv(p90)]), axis=0)
            contrast = float(np.percentile(pr[okp], 90) - np.percentile(pr[okp], 10))
            width = float(np.median(q[1][okp] - q[0][okp])) + 2.56 * res_sd
            out.update(prop_p50=conv(p50), prop_p10=q[0], prop_p90=q[1], prop_coef=coef.tolist(), prop_resid_sd=res_sd,
                       prop_contrast=contrast, prop_width=width, prop_r2=float(1 - res_sd ** 2 / max(1e-30, np.var(pr[okp]))))
    seen = c_inv >= 0.6 and cover >= 0.5
    msg = (f"seismic-band detail recovered r = {c_inv:.2f}; full log r = {c_full:.2f} vs LFM alone {c_lfm:.2f}; P10-P90 width {100 * out['rel_width']:.0f} % "
           f"of AI; log inside the band {100 * cover:.0f} %")
    if "prop_width" in out:
        seen_p = seen and out["prop_r2"] > 0.4 and out["prop_width"] < out["prop_contrast"]
        msg += (f"; property P10-P90 width {out['prop_width']:.3g} vs its range {out['prop_contrast']:.3g} (AI explains "
                f"{100 * out['prop_r2']:.0f} %)")
        seen = seen_p
    out["verdict"] = ("YES - the seismic resolves it here: " if seen else "NO / weak - ") + msg
    return out
