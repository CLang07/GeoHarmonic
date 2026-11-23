"""Gridding of well values for the velocity model (PLAN item 17b) and the seismic-velocity guide.

grid_values(X, Y, px, py, v, method) with method:
  idw      inverse distance (power 2) - the item-12 default
  tps      thin-plate spline through the wells (exact unless smooth > 0; reproduces planar trends; < 3 wells -> IDW)
  kriging  simple kriging (mean = well mean) with a spherical / exponential variogram fitted to the wells (or a
           user range); exact at the wells, smooth between, back to the mean far from wells; <= ~500 wells
Seismic-velocity guide: a velocity volume in the project (stacking / RMS velocity, or interval velocity) is sampled
along the horizons; per layer the Dix interval velocity (or the time-averaged interval velocity) is the trend, and the
wells scale it: ratio (well / trend at the well) or difference, gridded with the same method.
"""
import numpy as np

from .velocity import _idw_nodes

METHODS = {"idw": "Inverse distance", "tps": "Thin-plate spline", "kriging": "Kriging (simple)"}


# --------------------------------------------------------------------------- thin-plate spline
def _frame(px, py):
    cx, cy = float(np.mean(px)), float(np.mean(py))
    s = float(max(np.ptp(px), np.ptp(py), 1.0))
    return cx, cy, s


def _tps_kernel(d2):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d2 > 0, 0.5 * d2 * np.log(np.where(d2 > 0, d2, 1.0)), 0.0)


def tps_fit(px, py, v, smooth=0.0):
    """Thin-plate spline coefficients in a centred, scaled frame (well conditioned for map coordinates)."""
    px, py, v = (np.asarray(a, float) for a in (px, py, v))
    cx, cy, s = _frame(px, py)
    x, y = (px - cx) / s, (py - cy) / s
    n = len(x)
    K = _tps_kernel((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    K += smooth * np.eye(n)
    P = np.column_stack([np.ones(n), x, y])
    A = np.zeros((n + 3, n + 3))
    A[:n, :n], A[:n, n:], A[n:, :n] = K, P, P.T
    b = np.concatenate([v, np.zeros(3)])
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    return {"w": sol[:n], "a": sol[n:], "x": x, "y": y, "frame": (cx, cy, s)}


def tps_eval(fit, X, Y):
    cx, cy, s = fit["frame"]
    gx = (np.asarray(X, float) - cx) / s
    gy = (np.asarray(Y, float) - cy) / s
    a = fit["a"]
    out = a[0] + a[1] * gx + a[2] * gy
    for xi, yi, wi in zip(fit["x"], fit["y"], fit["w"]):
        out += wi * _tps_kernel((gx - xi) ** 2 + (gy - yi) ** 2)
    return out


# --------------------------------------------------------------------------- kriging
def _shape(h, a, model):
    """Normalised variogram (0 at h = 0 -> 1 at the sill)."""
    r = np.asarray(h, float) / max(a, 1e-9)
    if model == "exponential":
        return 1.0 - np.exp(-3.0 * r)
    return np.where(r < 1.0, 1.5 * r - 0.5 * r ** 3, 1.0)


def fit_variogram(px, py, v, model="spherical", range_m=None, nbins=10):
    """{model, sill, range, nugget=0, experimental (h, gamma, n)}: experimental semivariogram of the well values,
    sill by least squares for each trial range, best range (or the user's range). Few wells -> sill = variance,
    range = half the largest well distance."""
    px, py, v = (np.asarray(a, float) for a in (px, py, v))
    n = len(v)
    var = float(np.var(v)) if n > 1 else 0.0
    d = np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :])
    iu = np.triu_indices(n, 1)
    h, g = d[iu], 0.5 * (v[iu[0]] - v[iu[1]]) ** 2
    dmax = float(h.max()) if len(h) else 1000.0
    out = {"model": model, "sill": max(var, 1e-12), "range": float(range_m or max(dmax / 2, 1.0)), "nugget": 0.0,
           "experimental": (np.array([]), np.array([]), np.array([]))}
    if n < 4 or var <= 0:
        return out
    edges = np.linspace(0.0, dmax * 0.7, nbins + 1)
    k = np.clip(np.searchsorted(edges, h, "right") - 1, 0, nbins)
    hb, gb, nb = [], [], []
    for b in range(nbins):
        m = k == b
        if m.sum() >= 1:
            hb.append(h[m].mean())
            gb.append(g[m].mean())
            nb.append(m.sum())
    hb, gb, nb = np.array(hb), np.array(gb), np.array(nb, float)
    out["experimental"] = (hb, gb, nb)
    if len(hb) < 2:
        return out
    ranges = [float(range_m)] if range_m else np.geomspace(max(dmax / 50, 1.0), dmax * 2.0, 80)
    best = None
    for a in ranges:
        f = _shape(hb, a, model)
        den = float(np.sum(nb * f * f))
        if den <= 0:
            continue
        sill = float(np.sum(nb * f * gb)) / den
        if sill <= 0:
            continue
        sse = float(np.sum(nb * (sill * f - gb) ** 2))
        if best is None or sse < best[0]:
            best = (sse, a, sill)
    if best is not None:
        out["range"], out["sill"] = float(best[1]), float(best[2])
    return out


def krige_fit(px, py, v, vario):
    """Simple kriging in dual form: alpha = C^-1 (v - mean); estimate = mean + sum C(d_i) alpha_i."""
    px, py, v = (np.asarray(a, float) for a in (px, py, v))
    m = float(np.mean(v))
    d = np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :])
    C = vario["sill"] * (1.0 - _shape(d, vario["range"], vario["model"]))
    C += np.eye(len(v)) * 1e-9 * vario["sill"]
    alpha = np.linalg.solve(C, v - m)
    return {"mean": m, "alpha": alpha, "px": px, "py": py, "vario": vario}


def krige_eval(fit, X, Y):
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    vg = fit["vario"]
    out = np.full(X.shape, fit["mean"])
    for x, y, a in zip(fit["px"], fit["py"], fit["alpha"]):
        out += a * vg["sill"] * (1.0 - _shape(np.hypot(X - x, Y - y), vg["range"], vg["model"]))
    return out


# --------------------------------------------------------------------------- dispatcher
def grid_values(X, Y, px, py, v, method="idw", opts=None):
    """Grid scalar well values v at nodes (X, Y) with the chosen method."""
    opts = opts or {}
    px, py, v = (np.asarray(a, float) for a in (px, py, v))
    if method == "idw" or len(v) < 2:
        return _idw_nodes(X, Y, px, py, v)
    if method == "tps":
        if len(v) < 3 or np.linalg.matrix_rank(np.c_[np.ones(len(v)), px - px.mean(), py - py.mean()]) < 3:
            return _idw_nodes(X, Y, px, py, v)
        return tps_eval(tps_fit(px, py, v, float(opts.get("smooth", 0.0) or 0.0)), X, Y)
    if method == "kriging":
        vg = fit_variogram(px, py, v, opts.get("variogram", "spherical"), opts.get("range"))
        return krige_eval(krige_fit(px, py, v, vg), X, Y)
    raise ValueError(f"unknown gridding method {method!r}")


# --------------------------------------------------------------------------- seismic-velocity guide
def sample_volume(vol, X, Y, times_list):
    """Values of a 3D volume (original units) at map points (X, Y) and times (ms), one array per entry of
    times_list (each X's shape); nearest trace, linear in time; NaN outside the survey. Reads each inline once."""
    shape = np.shape(X)
    x, y = np.ravel(X).astype(float), np.ravel(Y).astype(float)
    il, xl = vol.ilxl_at(x, y)
    ii = np.rint((np.asarray(il) - vol.il0) / vol.il_step).astype(np.int64)
    jj = np.rint((np.asarray(xl) - vol.xl0) / vol.xl_step).astype(np.int64)
    inside = (ii >= 0) & (ii < vol.nil) & (jj >= 0) & (jj < vol.nxl)
    outs = [np.full(x.shape, np.nan) for _ in times_list]
    T = [np.ravel(t).astype(float) for t in times_list]
    for i in np.unique(ii[inside]):
        rows = np.flatnonzero(inside & (ii == i))
        tr = np.asarray(vol.ilxl[i], np.float32)[jj[rows]] * vol.inv  # (nrows, ns)
        for o, t in zip(outs, T):
            s = (t[rows] / 1000.0 - vol.t0) / vol.dt
            ok = np.isfinite(s)
            s = np.clip(np.where(ok, s, 0.0), 0.0, vol.ns - 1.0)
            s0 = np.minimum(np.floor(s).astype(np.int64), vol.ns - 2) if vol.ns > 1 else np.zeros(len(s), np.int64)
            f = s - s0
            r = np.arange(len(rows))
            val = tr[r, s0] * (1 - f) + tr[r, np.minimum(s0 + 1, vol.ns - 1)] * f
            o[rows] = np.where(ok, val, np.nan)
    return [o.reshape(shape) for o in outs]


def dix(v_top, t_top, v_base, t_base):
    """Dix interval velocity between two RMS velocities (times in any common unit); NaN where not physical."""
    with np.errstate(invalid="ignore", divide="ignore"):
        num = v_base ** 2 * t_base - v_top ** 2 * t_top
        den = t_base - t_top
        q = np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1.0), np.nan)
        return np.where(q > 0, np.sqrt(np.where(q > 0, q, 0.0)), np.nan)


def guide_trends(vol, X, Y, time_grids, layers_guided, kind="rms", nsub=12):
    """Trend interval velocity per guided layer index: {k: grid}. kind 'rms' = Dix between the RMS velocities at the
    layer's top and base horizons (layer 0 top = datum); 'interval' = time average of the volume inside the layer."""
    K = len(time_grids)
    tops = [np.zeros(np.shape(X))] + [g.z for g in time_grids[:-1]]
    out = {}
    if kind == "rms":
        need = sorted({k for k in layers_guided} | {k - 1 for k in layers_guided if k > 0})
        vals = sample_volume(vol, X, Y, [time_grids[k].z for k in need])
        vr = dict(zip(need, vals))
        for k in layers_guided:
            tb = time_grids[k].z
            if k == 0:
                out[k] = np.where(tb > 0, vr[0], np.nan)
            else:
                out[k] = dix(vr[k - 1], tops[k], vr[k], tb)
        return out
    for k in layers_guided:
        tt, tb = tops[k], time_grids[k].z
        fr = (np.arange(nsub) + 0.5) / nsub
        vals = sample_volume(vol, X, Y, [tt + f * (tb - tt) for f in fr])
        with np.errstate(invalid="ignore"):
            out[k] = np.mean(vals, axis=0)  # equal time steps: time average = interval velocity
    assert all(k < K for k in out)
    return out


def guided_grid(X, Y, px, py, v, trend, mode="ratio", method="idw", opts=None):
    """Grid well values along a trend grid: ratio (v / trend at the wells, gridded, x trend) or difference.
    Wells where the trend is undefined are skipped; nodes without a trend use the plain gridded values."""
    t_w = _sample_nodes(X, Y, trend, px, py)
    ok = np.isfinite(t_w) & np.isfinite(v) & (t_w > 0 if mode == "ratio" else np.isfinite(t_w))
    plain = grid_values(X, Y, px, py, v, method, opts)
    if not ok.any():
        return plain, None
    q = v[ok] / t_w[ok] if mode == "ratio" else v[ok] - t_w[ok]
    qg = grid_values(X, Y, px[ok], py[ok], q, method, opts)
    g = trend * qg if mode == "ratio" else trend + qg
    return np.where(np.isfinite(g), g, plain), q


def _sample_nodes(X, Y, Z, px, py):
    """Bilinear sample of a node grid (X, Y regular in index space, possibly rotated) at points."""
    X, Y, Z = np.asarray(X, float), np.asarray(Y, float), np.asarray(Z, float)
    ny, nx = Z.shape
    if ny < 2 or nx < 2:
        return np.full(len(px), float(Z.ravel()[0]) if Z.size else np.nan)
    o = np.array([X[0, 0], Y[0, 0]])
    ex = np.array([X[0, 1] - X[0, 0], Y[0, 1] - Y[0, 0]])
    ey = np.array([X[1, 0] - X[0, 0], Y[1, 0] - Y[0, 0]])
    M = np.array([ex, ey]).T
    fj, fi = np.linalg.solve(M, np.array([np.asarray(px, float) - o[0], np.asarray(py, float) - o[1]]))
    i0 = np.clip(np.floor(fi).astype(int), 0, ny - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, nx - 2)
    a, b = np.clip(fi - i0, 0, 1), np.clip(fj - j0, 0, 1)
    out = (Z[i0, j0] * (1 - a) * (1 - b) + Z[i0 + 1, j0] * a * (1 - b) + Z[i0, j0 + 1] * (1 - a) * b
           + Z[i0 + 1, j0 + 1] * a * b)
    far = (fi < -0.5) | (fi > ny - 0.5) | (fj < -0.5) | (fj > nx - 0.5)
    return np.where(far, np.nan, out)
