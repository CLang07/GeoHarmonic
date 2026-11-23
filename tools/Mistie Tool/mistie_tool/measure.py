"""Mistie measurement between two tie traces, and applying corrections.

Convention: a measurement (shift_ms, phase_deg, gain) of trace b against trace a is the
correction that makes b match a:  b_corrected = gain * rotate(delay(b, shift_ms), phase_deg),
where delay() moves the trace DOWN by shift_ms (positive = later) and rotate() is a constant
phase rotation Re{analytic(x) * exp(i*phase)}.
"""
import numpy as np
from scipy.signal import hilbert


def fourier_shift(x, lag_samples):
    """Delay x by lag_samples (fractional allowed)."""
    n = len(x)
    f = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * lag_samples), n)


def phase_rotate(x, deg):
    return np.real(hilbert(x) * np.exp(1j * np.radians(deg)))


def apply_correction(x, dt, shift_ms, phase_deg, gain):
    """Apply a constant correction to one trace or to every trace (rows) of a panel."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n = x.shape[1]
    f = np.fft.rfftfreq(n)
    X = np.fft.rfft(x, axis=1) * np.exp(-2j * np.pi * f * shift_ms / 1000.0 / dt)
    y = np.fft.irfft(X, n, axis=1)
    y = np.real(hilbert(y, axis=1) * np.exp(1j * np.radians(phase_deg))) * gain
    return y if y.shape[0] > 1 else y[0]


def measure_mistie(a, b, dt, window, max_lag_ms=40.0):
    """Return (shift_ms, phase_deg, gain, corr) of b relative to a inside window (t0, t1) seconds."""
    i0, i1 = max(0, int(round(window[0] / dt))), min(len(a), int(round(window[1] / dt)))
    if i1 - i0 < 10:
        raise ValueError("window too short")
    aw = a[i0:i1] - a[i0:i1].mean()
    bw = b[i0:i1] - b[i0:i1].mean()
    L = max(1, int(max_lag_ms / 1000 / dt))
    full = np.correlate(aw, bw, mode="full")
    mid = len(bw) - 1
    seg = np.abs(hilbert(full))[mid - L: mid + L + 1]      # envelope: shift independent of phase
    k = int(np.argmax(seg))
    frac = 0.0
    if 0 < k < len(seg) - 1:
        y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    lag = (k - L) + frac
    bs = fourier_shift(b, lag)[i0:i1]
    bs = bs - bs.mean()
    h = np.imag(hilbert(bs))
    phase = np.degrees(np.arctan2(-np.dot(aw, h), np.dot(aw, bs)))
    br = phase_rotate(bs, phase)
    denom = np.sum(br ** 2)
    gain = float(np.sqrt(np.sum(aw ** 2) / denom)) if denom > 0 else 1.0
    corr = float(np.corrcoef(aw, br)[0, 1]) if np.std(br) > 0 and np.std(aw) > 0 else 0.0
    return float(lag * dt * 1000.0), float(phase), gain, corr


def half_window_difference(a, b, dt, window, max_lag_ms=40.0):
    """Shift measured in the shallow half minus the deep half: a simple stability check."""
    mid = 0.5 * (window[0] + window[1])
    s1 = measure_mistie(a, b, dt, (window[0], mid), max_lag_ms)[0]
    s2 = measure_mistie(a, b, dt, (mid, window[1]), max_lag_ms)[0]
    return abs(s1 - s2)
