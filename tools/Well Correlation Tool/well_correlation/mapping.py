"""Maps from tops: structure (TVDSS of a top) and isopach (vertical thickness between two tops).

Gridding (numpy only):
  * thin-plate spline (smooth, honours every well; best for a handful of wells)
  * inverse-distance weighting, power 2 (robust, never overshoots; good for many wells)
Cells far from any well are left blank (more than `reach` x the typical well spacing away),
so the map does not invent structure where there is no control.
Contours by marching squares (line segments; no matplotlib needed in the GUI).
"""
import math
import numpy as np


def map_points(project, kind, top, base=None, wells=None):
    """[(well, x, y, value)] for 'structure' (TVDSS, positive down) or 'isopach' (TVD thickness)."""
    pts = []
    for wn in wells or list(project.wells):
        w = project.wells.get(wn)
        t = project.tops.get(wn, {})
        if w is None or w.x is None or top not in t or (kind == "isopach" and base not in t):
            continue
        project.load_well(wn)
        z0 = float(w.md_to_tvdss([t[top]])[0])
        if kind == "structure":
            pts.append((wn, w.x, w.y, z0))
        else:
            z1 = float(w.md_to_tvdss([t[base]])[0])
            pts.append((wn, w.x, w.y, z1 - z0))
    return pts


def _tps_fit(x, y, v, smooth=0.0):
    n = len(x)
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(d > 0, d * d * np.log(d), 0.0)
    K += smooth * np.eye(n)
    P = np.column_stack([np.ones(n), x, y])
    A = np.zeros((n + 3, n + 3))
    A[:n, :n], A[:n, n:], A[n:, :n] = K, P, P.T
    b = np.concatenate([v, np.zeros(3)])
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    return sol[:n], sol[n:]


def _tps_eval(x, y, w, a, gx, gy):
    out = a[0] + a[1] * gx + a[2] * gy
    for xi, yi, wi in zip(x, y, w):
        d = np.hypot(gx - xi, gy - yi)
        with np.errstate(divide="ignore", invalid="ignore"):
            out += wi * np.where(d > 0, d * d * np.log(d), 0.0)
    return out


def grid_points(pts, method="tps", ncells=160, pad=0.25, reach=1.5, min_pad_m=2000.0):
    """Return dict(x0, y0, dx, dy, nx, ny, z (ny, nx) with NaN outside reach, vmin, vmax)."""
    if len(pts) < 2:
        raise ValueError("Need at least 2 wells with the top(s) to make a map")
    x = np.array([p[1] for p in pts], float)
    y = np.array([p[2] for p in pts], float)
    v = np.array([p[3] for p in pts], float)
    # local coordinates in km for numerical stability
    cx, cy = x.mean(), y.mean()
    xk, yk = (x - cx) / 1000.0, (y - cy) / 1000.0
    w_, h_ = max(np.ptp(xk), 1e-3), max(np.ptp(yk), 1e-3)
    padx = max(pad * max(w_, h_), min_pad_m / 1000.0)
    x0, x1 = xk.min() - padx, xk.max() + padx
    y0, y1 = yk.min() - padx, yk.max() + padx
    step = max(x1 - x0, y1 - y0) / ncells
    gxv = np.arange(x0, x1 + step, step)
    gyv = np.arange(y0, y1 + step, step)
    gx, gy = np.meshgrid(gxv, gyv)
    if method == "tps" and len(pts) >= 3:
        wts, a = _tps_fit(xk, yk, v)
        z = _tps_eval(xk, yk, wts, a, gx, gy)
    else:
        num = np.zeros_like(gx)
        den = np.zeros_like(gx)
        exact = np.full(gx.shape, np.nan)
        for xi, yi, vi in zip(xk, yk, v):
            d2 = (gx - xi) ** 2 + (gy - yi) ** 2
            exact = np.where(d2 < 1e-12, vi, exact)
            wgt = 1.0 / np.maximum(d2, 1e-12)
            num += wgt * vi
            den += wgt
        z = np.where(np.isfinite(exact), exact, num / den)
    # blank cells far from control
    nn = []
    for i in range(len(xk)):
        d = np.hypot(xk - xk[i], yk - yk[i])
        d = d[d > 0]
        if d.size:
            nn.append(d.min())
    spacing = max(np.median(nn) if nn else 2.0, 1.0)
    dmin = np.full(gx.shape, np.inf)
    for xi, yi in zip(xk, yk):
        dmin = np.minimum(dmin, np.hypot(gx - xi, gy - yi))
    z = np.where(dmin <= reach * spacing + 1.0, z, np.nan)
    return {"x0": cx + x0 * 1000.0, "y0": cy + y0 * 1000.0, "dx": step * 1000.0, "dy": step * 1000.0,
            "nx": len(gxv), "ny": len(gyv), "z": z, "vmin": float(np.nanmin(v)), "vmax": float(np.nanmax(v)),
            "points": pts}


def nice_interval(vmin, vmax, target=10):
    span = max(vmax - vmin, 1e-6)
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def contour_segments(g, levels):
    """Marching squares. Returns {level: (xs, ys)} with NaN-separated pairs for connect='pairs'."""
    z = g["z"]
    ny, nx = z.shape
    X = g["x0"] + np.arange(nx) * g["dx"]
    Y = g["y0"] + np.arange(ny) * g["dy"]
    out = {}
    z00, z10, z11, z01 = z[:-1, :-1], z[:-1, 1:], z[1:, 1:], z[1:, :-1]
    valid = np.isfinite(z00) & np.isfinite(z10) & np.isfinite(z11) & np.isfinite(z01)
    J, I = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))
    for lev in levels:
        b = ((z00 > lev).astype(int) | ((z10 > lev).astype(int) << 1) | ((z11 > lev).astype(int) << 2)
             | ((z01 > lev).astype(int) << 3))
        cells = valid & (b != 0) & (b != 15)
        if not cells.any():
            continue
        xs, ys = [], []
        ii, jj = I[cells], J[cells]
        a00, a10, a11, a01 = z00[cells], z10[cells], z11[cells], z01[cells]
        bb = b[cells]

        def edge(e, k):
            # edges: 0 bottom (00-10), 1 right (10-11), 2 top (01-11), 3 left (00-01)
            i, j = ii[k], jj[k]
            if e == 0:
                t = (lev - a00[k]) / (a10[k] - a00[k])
                return X[j] + t * g["dx"], Y[i]
            if e == 1:
                t = (lev - a10[k]) / (a11[k] - a10[k])
                return X[j + 1], Y[i] + t * g["dy"]
            if e == 2:
                t = (lev - a01[k]) / (a11[k] - a01[k])
                return X[j] + t * g["dx"], Y[i + 1]
            t = (lev - a00[k]) / (a01[k] - a00[k])
            return X[j], Y[i] + t * g["dy"]

        table = {1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 5: [(3, 2), (0, 1)], 6: [(0, 2)], 7: [(3, 2)],
                 8: [(2, 3)], 9: [(2, 0)], 10: [(2, 1), (0, 3)], 11: [(2, 1)], 12: [(1, 3)], 13: [(1, 0)], 14: [(0, 3)]}
        for k in range(len(bb)):
            for e1, e2 in table[int(bb[k])]:
                p1, p2 = edge(e1, k), edge(e2, k)
                xs += [p1[0], p2[0]]
                ys += [p1[1], p2[1]]
        out[lev] = (np.array(xs), np.array(ys))
    return out


# five-stop colour ramp (shallow / thick = warm, deep / thin = cool)
_RAMP = np.array([[49, 54, 149], [69, 117, 180], [171, 217, 233], [254, 224, 144], [215, 48, 39]], float)


def colorize(g, reverse=False, alpha=170):
    z = g["z"]
    lo, hi = np.nanmin(z), np.nanmax(z)
    t = (z - lo) / (hi - lo if hi > lo else 1)
    if reverse:
        t = 1 - t
    t = np.clip(np.nan_to_num(t, nan=0), 0, 1) * (len(_RAMP) - 1)
    k = np.clip(np.floor(t).astype(int), 0, len(_RAMP) - 2)
    f = (t - k)[..., None]
    rgb = _RAMP[k] * (1 - f) + _RAMP[k + 1] * f
    img = np.zeros(z.shape + (4,), np.uint8)
    img[..., :3] = rgb.astype(np.uint8)
    img[..., 3] = np.where(np.isfinite(z), alpha, 0)
    return img


def export_map_png(project, g, title, path, unit_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = g["z"]
    X = g["x0"] + np.arange(g["nx"]) * g["dx"]
    Y = g["y0"] + np.arange(g["ny"]) * g["dy"]
    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    lo, hi = np.nanmin(z), np.nanmax(z)
    step = nice_interval(lo, hi)
    levels = np.arange(math.floor(lo / step) * step, hi + step, step)
    cf = ax.contourf(X, Y, z, levels=levels, cmap="RdYlBu" if "Structure" in title else "RdYlBu_r", alpha=0.85)
    cs = ax.contour(X, Y, z, levels=levels, colors="k", linewidths=0.5)
    ax.clabel(cs, fmt="%g", fontsize=7)
    fig.colorbar(cf, ax=ax, label=unit_label, shrink=0.8)
    for wn, x, y, v in g["points"]:
        ax.plot(x, y, "ko", ms=4)
        ax.annotate(f"{wn}\n{v:.0f}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    for w in project.wells.values():
        if w.x is not None and w.name not in [p[0] for p in g["points"]]:
            ax.plot(w.x, w.y, "o", mfc="none", mec="0.5", ms=4)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("Easting NZTM (m)")
    ax.set_ylabel("Northing NZTM (m)")
    ax.ticklabel_format(style="plain", useOffset=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
