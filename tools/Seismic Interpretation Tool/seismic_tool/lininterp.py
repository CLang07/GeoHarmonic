"""Safe linear-interpolation indices (PLAN 43b).

`np.clip(x, 0, n - 1.000001)` looked safe, but on float32 input (numpy 1.26 keeps the array's dtype) n - 1.000001
rounds to n - 1, floor gives n - 1 and `a[k + 1]` reads one past the end (Wheeler on Chris's PC: index 300 out of
bounds for axis 1 with size 300). lin_index works in float64 and clips k to n - 2 so the last sample interpolates
exactly (k = n - 2, w = 1)."""
import numpy as np


def lin_index(pos, n):
    """pos (any shape, fractional sample index; NaN allowed) -> (k, k1, w): a[k] * (1 - w) + a[k1] * w.
    Positions are clamped to [0, n - 1]; NaN -> 0 (mask them yourself). k1 = k + 1 except when n == 1."""
    p = np.asarray(pos)
    wdt = np.float32 if p.dtype == np.float32 else np.float64
    p = np.asarray(p, np.float64)
    n = int(n)
    if n < 2:
        z = np.zeros(p.shape, np.int64)
        return z, z, np.zeros(p.shape, wdt)
    p = np.clip(np.where(np.isfinite(p), p, 0.0), 0.0, n - 1.0)
    k = np.minimum(np.floor(p).astype(np.int64), n - 2)
    w = np.clip(p - k, 0.0, 1.0)
    return k, k + 1, w.astype(wdt)
