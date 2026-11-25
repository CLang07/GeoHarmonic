"""Survey horizon / grid blending (PLAN 42) - adaptive seam blend for unmerged overlapping surveys, pure numpy.

The mistie tool does the bulk tie (shift / phase / amplitude) and item 17 / 37 depth-converts each survey with its own
velocities; this module removes only the RESIDUAL misfit between two horizon grids of the same horizon at their
seam, without smearing geology or removing fault throw:

    m = MasterGrid.covering([vol_a, vol_b])                       # common bin grid (finer survey spacing)
    za, zb = m.resample(vol_a, grid_a), m.resample(vol_b, grid_b)  # NaN outside each survey
    res = blend(m, [Survey("A", za, priority=1), Survey("B", zb, priority=2)], slope=2.0, method="hybrid",
                faults=[polyline_xy, ...], wells=[(x, y, top), ...])
    res.z (merged grid) ; res.qc (residual before / after, weight, width, added slope, seam gradient, histogram,
    well tie table) ; res.readout ("Blend width 0.8-6.2 km, overlap fully used on 87 % of seam")

Core idea (42d): blend width W = |R| / s, R = robust residual A - B along the seam, s = the allowed added slope (one
value Chris sets, m/km or ms/km), clamped to [w_min, w_max] and varying along the seam - small misfits get narrow
blends, big ones wide; the slope added to the surface never exceeds s.  Distances are geodesic inside fault blocks
(fault polylines are barriers), so a blend never crosses a fault and throw is kept.
Methods (42f): 'cut' (priority, hard seam), 'feather' (distance weights), 'residual' (R tapered into the lower-priority
survey), 'hybrid' (residual then a narrow feather; default).  Width > overlap -> that stretch uses the residual
surface (42g); seam points with no overlap are flagged as gaps.  Well tops are re-tied exactly afterwards (42i).
"""
import math
import time

import numpy as np

METHODS = ("cut", "feather", "residual", "hybrid")
UNITS = {"m/km": 1.0, "ms/km": 1.0, "ft/mile": 0.3048 / 1.609344, "ft/kft": 1.0}


def slope_per_m(value, unit="m/km"):
    """Allowed added slope in z units per metre of map distance (m/km -> /1000; ft/mile -> ft per ft)."""
    if unit == "ft/mile":
        return float(value) / 5280.0  # ft per ft
    if unit == "ft/kft":
        return float(value) / 1000.0
    return float(value) / 1000.0


# ====================================================================== geometry
class MasterGrid:
    """Regular map grid: x0, y0 (node 0,0), dx, dy, nx, ny; rows = y."""

    def __init__(self, x0, y0, dx, dy, nx, ny):
        self.x0, self.y0, self.dx, self.dy, self.nx, self.ny = float(x0), float(y0), float(dx), float(dy), int(nx), int(ny)

    @property
    def shape(self):
        return self.ny, self.nx

    def xy(self):
        X, Y = np.meshgrid(self.x0 + self.dx * np.arange(self.nx), self.y0 + self.dy * np.arange(self.ny))
        return X, Y

    @classmethod
    def covering(cls, vols, cell=None, pad=0.0):
        xs, ys, sp = [], [], []
        for v in vols:
            ox, oy = v.outline()
            xs += list(np.ravel(ox))
            ys += list(np.ravel(oy))
            a, b = v.ax, v.ay
            sp += [math.hypot(a[1], b[1]) * abs(v.il_step), math.hypot(a[2], b[2]) * abs(v.xl_step)]
        cell = float(cell or max(1.0, min(sp)))
        x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
        return cls(x0, y0, cell, cell, int(np.ceil((x1 - x0) / cell)) + 1, int(np.ceil((y1 - y0) / cell)) + 1)

    def resample(self, vol, grid):
        """Survey grid (nil, nxl) on its own bins -> master nodes (bilinear in il / xl; NaN outside the survey)."""
        X, Y = self.xy()
        a, b = vol.ax, vol.ay
        M = np.array([[a[1], a[2]], [b[1], b[2]]])
        inv = np.linalg.inv(M)
        dxp, dyp = X - a[0], Y - b[0]
        il = inv[0, 0] * dxp + inv[0, 1] * dyp
        xl = inv[1, 0] * dxp + inv[1, 1] * dyp
        fi = (il - vol.il0) / vol.il_step
        fj = (xl - vol.xl0) / vol.xl_step
        return bilinear(np.asarray(grid, float), fi, fj)

    def index(self, x, y):
        return (np.asarray(y, float) - self.y0) / self.dy, (np.asarray(x, float) - self.x0) / self.dx


def bilinear(g, fi, fj):
    g = np.asarray(g, float)
    ni, nj = g.shape
    fi, fj = np.asarray(fi, float), np.asarray(fj, float)
    ok = (fi >= 0) & (fi <= ni - 1) & (fj >= 0) & (fj <= nj - 1)
    i0 = np.clip(np.floor(fi).astype(int), 0, max(0, ni - 2))
    j0 = np.clip(np.floor(fj).astype(int), 0, max(0, nj - 2))
    u, v = np.clip(fi - i0, 0, 1), np.clip(fj - j0, 0, 1)
    i1, j1 = np.minimum(i0 + 1, ni - 1), np.minimum(j0 + 1, nj - 1)
    z = (g[i0, j0] * (1 - u) * (1 - v) + g[i1, j0] * u * (1 - v) + g[i0, j1] * (1 - u) * v + g[i1, j1] * u * v)
    return np.where(ok, z, np.nan)


def _shift(a, di, dj, fill):
    out = np.full(a.shape, fill, dtype=a.dtype)
    ni, nj = a.shape
    si = slice(max(0, di), ni + min(0, di))
    sj = slice(max(0, dj), nj + min(0, dj))
    ti = slice(max(0, -di), ni + min(0, -di))
    tj = slice(max(0, -dj), nj + min(0, -dj))
    out[si, sj] = a[ti, tj]
    return out


NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def geodesic(seeds, allowed, cell=1.0, max_dist=np.inf, labels=None, barrier=None):
    """Distance (map units) from seed cells through allowed cells, 8-neighbour chamfer (1, sqrt 2), never stepping
    across a barrier (fault) cell; also the label (e.g. seam point index) of the nearest seed.  Iterates until no
    change or max_dist."""
    allowed = np.asarray(allowed, bool) & (True if barrier is None else ~np.asarray(barrier, bool))
    D = np.where(seeds, 0.0, np.inf)
    L = np.where(seeds, labels if labels is not None else 0, -1).astype(np.int64)
    it = 0
    while True:
        it += 1
        changed = False
        for di, dj in NB8:
            c = cell * (math.sqrt(2.0) if di and dj else 1.0)
            nd = _shift(D, di, dj, np.inf) + c
            better = allowed & (nd < D - 1e-9) & (nd <= max_dist)
            if better.any():
                D = np.where(better, nd, D)
                L = np.where(better, _shift(L, di, dj, -1), L)
                changed = True
        if not changed or it > 100000:
            break
    return D, L


def erode(mask, n):
    m = np.asarray(mask, bool).copy()
    for _ in range(int(n)):
        e = m.copy()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            e &= _shift(m, di, dj, False)
        m = e
    return m


def edge(mask):
    """Cells of mask with a 4-neighbour outside it (the map border itself is not an edge)."""
    m = np.asarray(mask, bool)
    inner = m.copy()
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        inner &= _shift(m, di, dj, True)
    return m & ~inner


def rasterize_polylines(grid, lines, width_cells=1):
    """Fault polylines (map x, y) -> barrier cells (8-connected thin line, optionally thickened)."""
    b = np.zeros(grid.shape, bool)
    for ln in lines or ():
        p = np.asarray(ln, float)
        for (x0, y0), (x1, y1) in zip(p[:-1], p[1:]):
            n = int(max(abs(x1 - x0) / grid.dx, abs(y1 - y0) / grid.dy) * 2) + 2
            xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
            fi, fj = grid.index(xs, ys)
            i, j = np.round(fi).astype(int), np.round(fj).astype(int)
            ok = (i >= 0) & (i < grid.ny) & (j >= 0) & (j < grid.nx)
            b[i[ok], j[ok]] = True
    for _ in range(max(0, int(width_cells) - 1)):
        b |= _shift(b, 1, 0, False) | _shift(b, 0, 1, False)
    return b


def fault_blocks(barrier):
    """Connected components (4-connected) of the non-barrier cells -> block label per cell (-1 on barriers)."""
    lab = np.full(barrier.shape, -1, np.int64)
    free = ~barrier
    nb = 0
    todo = free.copy()
    while todo.any():
        seed = np.zeros_like(todo)
        idx = np.argwhere(todo)[0]
        seed[tuple(idx)] = True
        comp = seed
        while True:
            grow = comp.copy()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                grow |= _shift(comp, di, dj, False)
            grow &= free
            if grow.sum() == comp.sum():
                break
            comp = grow
        lab[comp] = nb
        todo &= ~comp
        nb += 1
    return lab


# ====================================================================== robust residual along the seam
def median_filter2(a, half=1):
    """NaN-aware median in a (2h+1)^2 window."""
    a = np.asarray(a, float)
    stack = [_shift(a, di, dj, np.nan) for di in range(-half, half + 1) for dj in range(-half, half + 1)]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(np.stack(stack), axis=0)


def smooth_along(values, order_xy, sigma=3.0):
    """Gaussian smoothing of seam values along the seam (points ordered by nearest-neighbour chaining)."""
    v = np.asarray(values, float)
    n = len(v)
    if n < 3 or sigma <= 0:
        return v
    P = np.asarray(order_xy, float)
    # chain the points: greedy nearest neighbour from one end (an extreme point)
    start = int(np.argmin(P[:, 0] + P[:, 1]))
    order = [start]
    left = np.ones(n, bool)
    left[start] = False
    for _ in range(n - 1):
        last = P[order[-1]]
        d = np.where(left, np.hypot(P[:, 0] - last[0], P[:, 1] - last[1]), np.inf)
        k = int(np.argmin(d))
        order.append(k)
        left[k] = False
    order = np.array(order)
    r = int(np.ceil(3 * sigma))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    vs = v[order]
    fin = np.isfinite(vs)
    num = np.convolve(np.where(fin, vs, 0.0), w, mode="same")
    den = np.convolve(fin.astype(float), w, mode="same")
    out = np.empty(n)
    out[order] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    return out


# ====================================================================== surveys + blend
class Survey:
    def __init__(self, name, z, priority=1, quality=1.0, confidence=None, trim=0):
        self.name, self.z = str(name), np.asarray(z, float)
        self.priority, self.quality = int(priority), float(quality)
        self.confidence = None if confidence is None else np.asarray(confidence, float)
        self.trim = int(trim)

    def valid(self):
        v = np.isfinite(self.z)
        return erode(v, self.trim) if self.trim else v


class BlendResult:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def blend(grid, surveys, slope=2.0, unit="m/km", method="hybrid", w_min=500.0, w_max=10000.0, faults=(),
          wells=(), seam_sigma=3.0, median_half=1, feather_width=None, trim=None, cancel=None, tie_tol=0.1,
          tie_radius=None):
    """Merge two survey grids of one horizon (see module doc).  Returns BlendResult(z, qc, readout, ...)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    t0 = time.time()
    ss = sorted(surveys, key=lambda s: s.priority)
    if len(ss) != 2:
        raise ValueError("blend two surveys at a time (merge the result with the next survey)")
    A, B = ss
    if trim is not None:
        A.trim = B.trim = int(trim)
    va, vb = A.valid(), B.valid()
    za, zb = np.where(va, A.z, np.nan), np.where(vb, B.z, np.nan)
    cell = float(min(abs(grid.dx), abs(grid.dy)))
    s_m = slope_per_m(slope, unit)
    barrier = rasterize_polylines(grid, faults) if faults else np.zeros(grid.shape, bool)
    blocks = fault_blocks(barrier) if faults else np.zeros(grid.shape, np.int64)
    overlap = va & vb & ~barrier
    cut = np.where(va, za, zb)  # (1) priority cut - also the "blending off" result
    # seam = edge of A's coverage where B continues (inside the overlap)
    ea = edge(va) & vb & ~barrier
    seam_idx = np.argwhere(ea)
    qc = {"overlap": overlap, "barrier": barrier, "blocks": blocks}
    if method == "cut" or not len(seam_idx):
        z = cut
        return _finish(grid, z, cut, za, zb, overlap, None, None, None, qc, wells, tie_tol, tie_radius, slope, unit,
                       s_m, t0, method, A, B, [], np.zeros(0, bool))
    # ---- 42c robust residual along the seam (median-filtered, smoothed along the seam), per fault block
    R = np.where(overlap, za - zb, np.nan)
    Rm = median_filter2(R, median_half)
    # the residual at a seam point = median of the overlap residual within a few cells inside A (robust to spikes)
    near, lab = geodesic(ea, overlap, cell, max_dist=3 * cell, labels=np.arange(ea.size).reshape(ea.shape), barrier=barrier)
    seam_flat = np.flatnonzero(ea.ravel())
    pos = {k: n for n, k in enumerate(seam_flat)}
    vals = [[] for _ in seam_flat]
    for c in np.flatnonzero(np.isfinite(near.ravel()) & np.isfinite(Rm.ravel())):
        L = lab.ravel()[c]
        if L in pos:
            vals[pos[L]].append(Rm.ravel()[c])
    r_seam = np.array([np.median(v) if v else np.nan for v in vals])
    X, Y = grid.xy()
    sx, sy = X.ravel()[seam_flat], Y.ravel()[seam_flat]
    # smoothing along the seam stays inside each fault block (throw is not averaged across a fault)
    sb = blocks.ravel()[seam_flat]
    r_s = r_seam.copy()
    for b in np.unique(sb):
        m = sb == b
        r_s[m] = smooth_along(r_seam[m], np.c_[sx[m], sy[m]], seam_sigma)
    r_s = np.where(np.isfinite(r_s), r_s, 0.0)
    # ---- 42d adaptive width
    W = np.clip(np.abs(r_s) / max(1e-12, s_m), w_min, w_max)
    # ---- distances (geodesic, per block): into A from the seam, and out into B beyond A
    lab0 = np.full(grid.shape, -1, np.int64)
    lab0.ravel()[seam_flat] = np.arange(len(seam_flat))
    dmax = float(W.max()) * 1.05 + 2 * cell
    dA, lA = geodesic(ea, va & ~barrier, cell, dmax, lab0, barrier)
    onlyB = vb & ~va & ~barrier
    dB, lB = geodesic(ea, onlyB | ea, cell, dmax, lab0, barrier)
    # ---- 42g width larger than the overlap: the overlap depth available at each seam point
    over_depth = np.zeros(len(seam_flat))
    inov = overlap & (lA >= 0) & np.isfinite(dA)
    np.maximum.at(over_depth, lA[inov], dA[inov])
    too_wide = W > over_depth + cell
    gap = np.zeros(len(seam_flat), bool)  # seam with no B beyond A's edge -> interpolation, not blending
    out_count = np.bincount(lB[onlyB & (lB >= 0)], minlength=len(seam_flat))
    gap = out_count == 0
    z = cut.copy()
    w_map = np.full(grid.shape, np.nan)
    width_map = np.full(grid.shape, np.nan)
    corrB = np.zeros(grid.shape)
    use_res = method in ("residual", "hybrid")
    if method == "feather":
        # (2) weights 1 -> 0 from A's interior to its edge over W (where the overlap allows it), else residual
        m = overlap & (lA >= 0) & np.isfinite(dA)
        Wc = W[lA[m]]
        wa = np.clip(dA[m] / Wc, 0, 1)
        zf = wa * za[m] + (1 - wa) * zb[m]
        z[m] = zf
        w_map[m] = wa
        width_map[m] = Wc
        use_res = bool(too_wide.any())
        res_pts = too_wide
    else:
        res_pts = np.ones(len(seam_flat), bool)
    if use_res:
        # (3) residual surface: R at the nearest seam point, tapered linearly to 0 over W into B's own area -> added
        # to B so B meets A at the seam; A keeps priority everywhere it exists
        m = onlyB & (lB >= 0) & np.isfinite(dB)
        sel = res_pts[lB[m]]
        Wc = W[lB[m]]
        taper = np.clip(1.0 - dB[m] / Wc, 0, 1)
        c = r_s[lB[m]] * taper * sel
        corrB[m] = c
        z[m] = zb[m] + c
        width_map[m] = np.where(sel, Wc, width_map[m])
    if method == "hybrid":
        # then a narrow feather across the seam inside A (removes the last kink; width = w_min)
        fw = float(feather_width or w_min)
        m = overlap & (lA >= 0) & np.isfinite(dA) & (dA <= fw)
        wa = np.clip(dA[m] / fw, 0, 1)
        # B already matches A at the seam within the smoothing error: blend A with the corrected B
        zb_c = zb[m] + r_s[lA[m]]
        z[m] = wa * za[m] + (1 - wa) * zb_c
        w_map[m] = wa
    # added slope ACROSS the seam (along the taper / feather distance): |R| / W <= s by construction; the
    # along-seam part is the residual's own variation (reported separately as added_slope_total)
    sl = []
    for own, base, dist in ((va & ~barrier, za, dA), (onlyB, zb, dB)):
        f = np.where(own, z - base, np.nan)
        gy, gx = np.gradient(f, grid.dy, grid.dx)
        dd = np.where(np.isfinite(dist), dist, np.nan)
        hy, hx = np.gradient(dd, grid.dy, grid.dx)
        hn = np.hypot(hx, hy)
        with np.errstate(all="ignore"):
            sl.append(np.abs(gx * hx + gy * hy) / np.where(hn > 0, hn, np.nan))
    with np.errstate(all="ignore"):
        qc["added_slope"] = np.fmax(sl[0], sl[1])
    qc.update({"seam": ea, "seam_residual": r_seam, "seam_residual_smooth": r_s, "width_seam": W, "too_wide": too_wide,
               "gap": gap, "weight": w_map, "width": width_map, "corr_b": corrB, "seam_xy": np.c_[sx, sy]})
    return _finish(grid, z, cut, za, zb, overlap, W, r_s, over_depth, qc, wells, tie_tol, tie_radius, slope, unit,
                   s_m, t0, method, A, B, seam_flat, too_wide)


def _finish(grid, z, cut, za, zb, overlap, W, r_s, over_depth, qc, wells, tie_tol, tie_radius, slope, unit, s_m, t0,
            method, A, B, seam_flat, too_wide):
    cell = float(min(abs(grid.dx), abs(grid.dy)))
    before = np.where(overlap, za - zb, np.nan)
    after = np.where(overlap, z - za, np.nan)
    ref = np.where(np.isfinite(za), za, zb)
    added = z - ref
    # slope added to EACH survey, measured inside that survey's own area (the A | B reference switch at the seam is
    # not a slope); barrier (fault) cells excluded
    bar = qc.get("barrier", np.zeros(grid.shape, bool))
    slopes = []
    for own, base in ((np.isfinite(za), za), (np.isfinite(zb) & ~np.isfinite(za), zb)):
        f = np.where(own & ~bar, z - base, np.nan)
        gy, gx = np.gradient(f, grid.dy, grid.dx)
        slopes.append(np.hypot(gx, gy))
    with np.errstate(all="ignore"):
        add_slope = np.fmax(slopes[0], slopes[1])
    add_slope[~np.isfinite(z)] = np.nan
    gzy, gzx = np.gradient(np.nan_to_num(z), grid.dy, grid.dx)
    seam_grad = np.where(qc.get("seam", np.zeros(grid.shape, bool)), np.hypot(gzx, gzy), np.nan)
    qc.setdefault("added_slope", np.where(np.isfinite(z), 0.0, np.nan))
    qc.update({"residual_before": before, "residual_after": after, "added": added, "added_slope_total": add_slope,
               "seam_gradient": seam_grad})
    fin = before[np.isfinite(before)]
    qc["histogram"] = np.histogram(fin, bins=30) if fin.size else (np.zeros(0), np.zeros(0))
    # ---- 42i well tops stay honoured
    ties = []
    if wells:
        z, ties = tie_wells(grid, z, wells, tol=tie_tol, radius=tie_radius or (max(W) if W is not None and len(W)
                                                                                 else 20 * cell))
    qc["well_ties"] = ties
    if W is not None and len(W):
        used = 100.0 * float(np.mean(~too_wide)) if len(too_wide) else 100.0
        readout = (f"Blend width {W.min() / 1000:.1f}-{W.max() / 1000:.1f} km, overlap fully used on {used:.0f} % of "
                   f"seam" + (f"; {int(np.sum(qc['gap']))} seam cells without overlap (gap)" if qc.get("gap") is not None
                              and np.any(qc["gap"]) else ""))
    else:
        readout = "Priority cut (no blending)"
    return BlendResult(z=z, cut=cut, qc=qc, readout=readout, method=method, slope=slope, unit=unit, seconds=time.time() - t0,
                       surveys=(A.name, B.name), width=W, residual=r_s)


# ====================================================================== 42i well ties
def wendland(r, R):
    q = np.clip(np.asarray(r, float) / R, 0, 1)
    return (1 - q) ** 4 * (4 * q + 1)


def tie_wells(grid, z, wells, tol=0.1, radius=2000.0):
    """Re-tie every top: residual (top - grid at the well, bilinear) for wells off by more than tol, spread with a
    compact Wendland kernel solved so that the corrected grid (bilinear) honours each top exactly.  Returns (z, table)
    table = [(name, x, y, top, before, after)]."""
    z = np.asarray(z, float).copy()
    rows = []
    W = []
    for w in wells:
        name, x, y, top = (w if len(w) == 4 else ("", *w))
        fi, fj = grid.index(x, y)
        g = float(bilinear(z, fi, fj))
        W.append((name, float(x), float(y), float(top), float(fi), float(fj), g))
    need = [w for w in W if np.isfinite(w[6]) and abs(w[3] - w[6]) > tol]
    if need:
        X, Y = grid.xy()
        use = [w for w in W if np.isfinite(w[6])]  # every well is a kernel centre: wells that tie stay put
        n = len(use)
        phi = [wendland(np.hypot(X - w[1], Y - w[2]), radius) for w in use]
        M = np.array([[float(bilinear(phi[k], wi[4], wi[5])) for k in range(n)] for wi in use])
        r = np.array([(w[3] - w[6]) if w in need else 0.0 for w in use])
        a = np.linalg.lstsq(M, r, rcond=None)[0]
        corr = sum(a[k] * phi[k] for k in range(n))
        z = np.where(np.isfinite(z), z + corr, z)
    for w in W:
        after = float(bilinear(z, w[4], w[5]))
        rows.append((w[0], w[1], w[2], w[3], w[3] - w[6], w[3] - after))
    return z, rows


# ====================================================================== picks, dynamic update, time / depth
def correct_points(grid, result, x, y, z, survey="B"):
    """42j: blend the PICKS (then grid once): apply the residual-surface correction of `result` to scattered picks
    of the lower-priority survey (sampled bilinearly from the correction field)."""
    fi, fj = grid.index(x, y)
    c = bilinear(result.qc.get("corr_b", np.zeros(grid.shape)), fi, fj)
    return np.asarray(z, float) + np.nan_to_num(c) if survey == result.surveys[1] else np.asarray(z, float)


class Blender:
    """Several horizons, each blended once; `update(name, za, zb)` re-blends only that horizon and reports which
    seam stretches changed (42k dynamic update via 17c)."""

    def __init__(self, grid, **kw):
        self.grid, self.kw = grid, kw
        self.results = {}
        self.inputs = {}
        self.log = []

    def add(self, name, za, zb, pa=1, pb=2, **kw):
        self.inputs[name] = (np.asarray(za, float).copy(), np.asarray(zb, float).copy(), pa, pb, kw)
        r = blend(self.grid, [Survey("A", za, pa), Survey("B", zb, pb)], **dict(self.kw, **kw))
        self.results[name] = r
        return r

    def update(self, name, za=None, zb=None, tol=0.05):
        old = self.results.get(name)
        a0, b0, pa, pb, kw = self.inputs[name]
        za = a0 if za is None else np.asarray(za, float)
        zb = b0 if zb is None else np.asarray(zb, float)
        r = self.add(name, za, zb, pa, pb, **kw)
        changed = 0
        if old is not None and old.residual is not None and r.residual is not None and len(old.residual) == len(r.residual):
            changed = int(np.sum(np.abs(old.residual - r.residual) > tol))
        msg = (f"Blend updated: {name} - {changed} of {len(r.residual) if r.residual is not None else 0} seam cells "
               f"changed")
        self.log.append(msg)
        return r, changed, msg
