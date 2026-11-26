"""Cartography shared by the map hub (Seismic tool), this tool's map and PNG / PDF export (PLAN 21f).  numpy only;
grids are plain dicts {"z", "x0", "y0", "dx", "dy"} (rows = y) like mapping.py - the Seismic tool converts MapGrids.

  hillshade(z, dx, dy, ...)            multi-directional hillshade (0..1), vertical exaggeration
  ramp_lut(name, reverse)              perceptual ramps: depth, thickness, property, diverging, terrain, bathymetry
  shaded_rgba(z, lut, vmin, vmax, shade, strength)
  chain_segments / contour_polylines   marching-squares segments chained into polylines (for labels)
  label_candidates / place_labels      labels along contours, rotated with the line, greedy no-overlap placement
  blank_polygons(g, polys, dilate)     NaN inside fault polygons (dilated one node so contours never cross them)
  fault_blocks / grid_by_blocks        node blocks separated by fault traces; grid each block from its own wells
  fault_ticks(line, side, ...)         downthrown ticks along a fault trace
  spill_point(g, crest)                closure: spill depth, spill node, closed area mask (depth grids, + down)
  WELL_SYMBOLS / classify_status       industry well symbols (oil, gas, oil+gas, show, dry, P&A, planned, injector)
  nice_scale / north_arrow_angle       scale bar length + segments; grid convergence for the north arrow
  STYLE_PRESETS                        named style sets (saved per project by the callers)
"""
import math
from collections import deque

import numpy as np

# ------------------------------------------------------------------------------------------- colour ramps
_RAMPS = {
    # perceptual, light-to-dark ramps (stops 0..1, RGB)
    "depth": [(0.0, (255, 255, 204)), (0.2, (161, 218, 180)), (0.45, (65, 182, 196)), (0.7, (44, 127, 184)),
              (0.88, (37, 52, 148)), (1.0, (8, 29, 88))],
    "thickness": [(0.0, (255, 247, 236)), (0.3, (253, 212, 158)), (0.6, (252, 141, 89)), (0.85, (215, 48, 31)),
                  (1.0, (127, 0, 0))],
    "property": [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)), (0.75, (94, 201, 98)),
                 (1.0, (253, 231, 37))],
    "diverging": [(0.0, (5, 48, 97)), (0.25, (67, 147, 195)), (0.5, (247, 247, 247)), (0.75, (214, 96, 77)),
                  (1.0, (103, 0, 31))],
    "terrain": [(0.0, (0, 97, 71)), (0.2, (16, 122, 47)), (0.45, (232, 215, 125)), (0.7, (161, 67, 0)),
                (0.9, (130, 30, 30)), (1.0, (255, 255, 255))],
    "bathymetry": [(0.0, (8, 29, 88)), (0.4, (34, 94, 168)), (0.75, (65, 182, 196)), (1.0, (199, 233, 180))],
}
RAMP_NAMES = list(_RAMPS)


def ramp_lut(name, reverse=False):
    stops = _RAMPS.get(name, _RAMPS["property"])
    pos = np.array([s[0] for s in stops])
    col = np.array([s[1] for s in stops], float)
    x = np.linspace(0, 1, 256)
    out = np.empty((256, 4), np.uint8)
    for k in range(3):
        out[:, k] = np.interp(x, pos, col[:, k]).astype(np.uint8)
    out[:, 3] = 255
    return out[::-1].copy() if reverse else out


def default_ramp(unit="", label=""):
    """Ramp + reverse flag for a grid: depth / time -> 'depth' reversed? (shallow = light); thickness; residual."""
    t = f"{label} {unit}".lower()
    if "resid" in t or "change" in t or "diff" in t:
        return "diverging", False
    if "thick" in t or "isochore" in t or "isochron" in t or "net" in t or "phih" in t or "sophih" in t:
        return "thickness", False
    if unit in ("m", "ms", "ft") or "depth" in t or "time" in t or "tvdss" in t:
        return "depth", False  # small values (shallow) light, deep dark
    return "property", False


# ------------------------------------------------------------------------------------------- hillshade
def hillshade(z, dx, dy, azimuths=(315.0, 270.0, 0.0, 225.0), weights=None, altitude=45.0, ve=1.0):
    """Multi-directional hillshade in 0..1 (NaN where z is NaN).  Rows = +y (north up when dy > 0)."""
    z = np.asarray(z, float) * ve
    zf = np.where(np.isfinite(z), z, np.nanmean(z) if np.isfinite(z).any() else 0.0)
    gy, gx = np.gradient(zf, abs(dy), abs(dx))
    if dy < 0:
        gy = -gy
    if dx < 0:
        gx = -gx
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, -gy)  # direction of steepest descent, clockwise from north
    alt = math.radians(altitude)
    weights = np.ones(len(azimuths)) if weights is None else np.asarray(weights, float)
    acc = np.zeros(z.shape)
    for az, w in zip(azimuths, weights):
        a = math.radians(az)
        acc += w * (math.sin(alt) * np.cos(slope) + math.cos(alt) * np.sin(slope) * np.cos(a - aspect))
    out = np.clip(acc / weights.sum(), 0, 1)
    out[~np.isfinite(z)] = np.nan
    return out


def shaded_rgba(z, lut, vmin=None, vmax=None, shade=None, strength=0.6, alpha=1.0):
    """Colour fill (LUT) darkened by a hillshade: rgb * ((1 - s) + s * 1.25 * shade), clipped."""
    z = np.asarray(z, float)
    fin = np.isfinite(z)
    lo = float(np.nanmin(z)) if vmin is None and fin.any() else (vmin or 0.0)
    hi = float(np.nanmax(z)) if vmax is None and fin.any() else (vmax if vmax is not None else 1.0)
    k = np.clip((z - lo) / max(1e-12, hi - lo) * 255, 0, 255)
    k = np.where(fin, k, 0).astype(int)
    rgba = np.asarray(lut)[k].astype(float)
    if shade is not None:
        f = (1 - strength) + strength * 1.25 * np.nan_to_num(shade, nan=0.8)
        rgba[..., :3] *= f[..., None]
    rgba = np.clip(rgba, 0, 255)
    rgba[..., 3] = np.where(fin, 255 * alpha, 0)
    return rgba.astype(np.uint8)


# ------------------------------------------------------------------------------------------- contours
def chain_segments(xs, ys, tol=1e-6):
    """connect='pairs' segment arrays -> list of polylines (x, y) by joining shared endpoints."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    n = len(xs) // 2
    if n == 0:
        return []
    P = np.stack([np.column_stack([xs[0:2 * n:2], ys[0:2 * n:2]]), np.column_stack([xs[1:2 * n:2], ys[1:2 * n:2]])], 1)
    scale = max(np.ptp(P[..., 0]), np.ptp(P[..., 1]), 1.0)
    q = np.round(P / (scale * tol)).astype(np.int64)
    keep = np.any(q[:, 0] != q[:, 1], axis=1)  # zero-length segments (contour through a grid node) would branch
    P, q = P[keep], q[keep]
    n = len(P)
    if n == 0:
        return []
    ends = {}
    for i in range(n):
        for e in (0, 1):
            ends.setdefault((q[i, e, 0], q[i, e, 1]), []).append((i, e))
    used = np.zeros(n, bool)
    lines = []
    for s in range(n):
        if used[s]:
            continue
        used[s] = True
        pts = deque([P[s, 0], P[s, 1]])
        for direction in (1, 0):  # extend from the tail (end 1) then from the head (end 0)
            cur_seg, cur_end = s, direction
            while True:
                key = tuple(q[cur_seg, cur_end])
                nxt = next(((i, e) for i, e in ends.get(key, []) if not used[i]), None)
                if nxt is None:
                    break
                i, e = nxt
                used[i] = True
                other = 1 - e
                if direction == 1:
                    pts.append(P[i, other])
                else:
                    pts.appendleft(P[i, other])
                cur_seg, cur_end = i, other
        arr = np.array(pts)
        lines.append((arr[:, 0], arr[:, 1]))
    return lines


def contour_polylines(g, levels):
    """{level: [(x, y) polyline]} for a dict grid."""
    from .mapping import contour_segments
    gg = {"z": g["z"], "x0": g["x0"], "y0": g["y0"], "dx": g["dx"], "dy": g["dy"], "nx": g["z"].shape[1],
          "ny": g["z"].shape[0]}
    return {lev: chain_segments(*xy) for lev, xy in contour_segments(gg, levels).items()}


def label_candidates(lines, text, spacing, min_len=None):
    """Positions along polylines every `spacing` (map units): (x, y, angle_deg (readable, -90..90), text)."""
    out = []
    for x, y in lines:
        d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        L = d[-1]
        if L < (min_len or spacing * 0.5):
            continue
        for s in np.arange(spacing / 2, L, spacing):
            k = int(np.clip(np.searchsorted(d, s) - 1, 0, len(x) - 2))
            t = (s - d[k]) / max(d[k + 1] - d[k], 1e-12)
            px, py = x[k] + t * (x[k + 1] - x[k]), y[k] + t * (y[k + 1] - y[k])
            a = math.degrees(math.atan2(y[k + 1] - y[k], x[k + 1] - x[k]))
            if a > 90:
                a -= 180
            elif a < -90:
                a += 180
            out.append((px, py, a, text))
    return out


def label_box(x, y, angle, text, char_w, char_h):
    """Axis-aligned bounds (x0, y0, x1, y1) of a rotated text box centred at (x, y) (map units)."""
    w, h = 0.5 * char_w * max(len(text), 1), 0.5 * char_h
    a = math.radians(angle)
    ca, sa = abs(math.cos(a)), abs(math.sin(a))
    hw, hh = w * ca + h * sa, w * sa + h * ca
    return (x - hw, y - hh, x + hw, y + hh)


def place_labels(cands, char_w, char_h, avoid=None, pad=0.15):
    """Greedy collision avoidance: keep a candidate when its (padded) box overlaps no kept box and no `avoid` box.
    cands: (x, y, angle, text[, priority]); higher priority first.  Returns the kept candidates."""
    order = sorted(range(len(cands)), key=lambda i: -(cands[i][4] if len(cands[i]) > 4 else 0))
    kept, boxes = [], list(avoid or [])
    for i in order:
        c = cands[i]
        b = label_box(c[0], c[1], c[2], c[3], char_w, char_h)
        p = pad * char_h
        b = (b[0] - p, b[1] - p, b[2] + p, b[3] + p)
        if any(not (b[2] < o[0] or o[2] < b[0] or b[3] < o[1] or o[3] < b[1]) for o in boxes):
            continue
        kept.append(c)
        boxes.append(b)
    return kept


def boxes_overlap(boxes):
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]):
                return True
    return False


# ------------------------------------------------------------------------------------------- faults
def point_in_polygon(x, y, px, py):
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    px, py = np.asarray(px, float), np.asarray(py, float)
    inside = np.zeros(x.shape, bool)
    j = len(px) - 1
    for i in range(len(px)):
        cond = (py[i] > y) != (py[j] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i]
        inside ^= cond & (x < xc)
        j = i
    return inside


def blank_polygons(g, polys, dilate=1):
    """Copy of grid g with NaN at nodes inside any polygon, dilated by `dilate` nodes so that no contour cell
    straddles a polygon edge (contours stop at the fault polygon instead of crossing it)."""
    z = np.array(g["z"], float)
    ny, nx = z.shape
    X, Y = np.meshgrid(g["x0"] + g["dx"] * np.arange(nx), g["y0"] + g["dy"] * np.arange(ny))
    m = np.zeros(z.shape, bool)
    for px, py in polys:
        m |= point_in_polygon(X.ravel(), Y.ravel(), px, py).reshape(z.shape)
        # cells crossed by the polygon edges
        xs, ys = np.append(px, px[0]), np.append(py, py[0])
        for k in range(len(xs) - 1):
            L = math.hypot(xs[k + 1] - xs[k], ys[k + 1] - ys[k])
            nstep = max(2, int(L / (0.25 * min(abs(g["dx"]), abs(g["dy"])))) + 1)
            t = np.linspace(0, 1, nstep)
            ex, ey = xs[k] + t * (xs[k + 1] - xs[k]), ys[k] + t * (ys[k + 1] - ys[k])
            jj = np.floor((ex - g["x0"]) / g["dx"]).astype(int)
            ii = np.floor((ey - g["y0"]) / g["dy"]).astype(int)
            for di in (0, 1):
                for dj in (0, 1):
                    a, b = ii + di, jj + dj
                    ok = (a >= 0) & (a < ny) & (b >= 0) & (b < nx)
                    m[a[ok], b[ok]] = True
    for _ in range(max(0, dilate - 1)):
        m2 = m.copy()
        m2[1:] |= m[:-1]
        m2[:-1] |= m[1:]
        m2[:, 1:] |= m[:, :-1]
        m2[:, :-1] |= m[:, 1:]
        m = m2
    z[m] = np.nan
    out = dict(g)
    out["z"] = z
    return out


def _seg_cross(ax, ay, bx, by, cx, cy, dx_, dy_):
    """Vectorised proper intersection of segments AB (arrays) with CD (scalars)."""
    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)
    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx_, dy_)
    o3 = orient(cx, cy, dx_, dy_, ax, ay)
    o4 = orient(cx, cy, dx_, dy_, bx, by)
    return (o1 * o2 < 0) & (o3 * o4 < 0)


def seg_cut(ax, ay, bx, by, cx, cy, dx_, dy_):
    """Half-open segment crossing for barriers (Seismic PLAN 43c): a point exactly on the other segment's line
    counts as the negative side, so a fault vertex / node lying exactly on a grid edge / fault line still cuts
    exactly one edge (the strict test above cut none and the fault leaked: 1 block instead of 2)."""
    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)
    s1 = orient(ax, ay, bx, by, cx, cy) > 0
    s2 = orient(ax, ay, bx, by, dx_, dy_) > 0
    s3 = orient(cx, cy, dx_, dy_, ax, ay) > 0
    s4 = orient(cx, cy, dx_, dy_, bx, by) > 0
    return (s1 != s2) & (s3 != s4)


def fault_blocks(nx, ny, x0, y0, dx, dy, faults):
    """Label grid nodes by fault block: 4-neighbour edges crossing a fault trace (polyline) are cut.
    Returns int array (ny, nx) of block ids."""
    X = x0 + dx * np.arange(nx)
    Y = y0 + dy * np.arange(ny)
    XX, YY = np.meshgrid(X, Y)
    cut_h = np.zeros((ny, nx - 1), bool)  # edge (i, j)-(i, j+1)
    cut_v = np.zeros((ny - 1, nx), bool)  # edge (i, j)-(i+1, j)
    for fx, fy in faults:
        for k in range(len(fx) - 1):
            cut_h |= seg_cut(XX[:, :-1], YY[:, :-1], XX[:, 1:], YY[:, 1:], fx[k], fy[k], fx[k + 1], fy[k + 1])
            cut_v |= seg_cut(XX[:-1], YY[:-1], XX[1:], YY[1:], fx[k], fy[k], fx[k + 1], fy[k + 1])
    lab = np.full((ny, nx), -1, int)
    nb = 0
    for i0 in range(ny):
        for j0 in range(nx):
            if lab[i0, j0] >= 0:
                continue
            q = deque([(i0, j0)])
            lab[i0, j0] = nb
            while q:
                i, j = q.popleft()
                if j + 1 < nx and not cut_h[i, j] and lab[i, j + 1] < 0:
                    lab[i, j + 1] = nb
                    q.append((i, j + 1))
                if j > 0 and not cut_h[i, j - 1] and lab[i, j - 1] < 0:
                    lab[i, j - 1] = nb
                    q.append((i, j - 1))
                if i + 1 < ny and not cut_v[i, j] and lab[i + 1, j] < 0:
                    lab[i + 1, j] = nb
                    q.append((i + 1, j))
                if i > 0 and not cut_v[i - 1, j] and lab[i - 1, j] < 0:
                    lab[i - 1, j] = nb
                    q.append((i - 1, j))
            nb += 1
    return lab


def grid_by_blocks(px, py, v, g, faults, grid_fn):
    """Grid well values separately in each fault block (wells only influence their own block).
    grid_fn(X, Y, px, py, v) -> values; blocks without wells use all wells.  Returns a dict grid."""
    z = np.full(g["z"].shape, np.nan)
    ny, nx = z.shape
    lab = fault_blocks(nx, ny, g["x0"], g["y0"], g["dx"], g["dy"], faults)
    X, Y = np.meshgrid(g["x0"] + g["dx"] * np.arange(nx), g["y0"] + g["dy"] * np.arange(ny))
    px, py, v = (np.asarray(a, float) for a in (px, py, v))
    jw = np.clip(np.round((px - g["x0"]) / g["dx"]).astype(int), 0, nx - 1)
    iw = np.clip(np.round((py - g["y0"]) / g["dy"]).astype(int), 0, ny - 1)
    wb = lab[iw, jw]
    for b in np.unique(lab):
        m = lab == b
        sel = wb == b
        if sel.sum() == 0:
            sel = np.ones(len(v), bool)
        z[m] = np.ravel(grid_fn(X[m], Y[m], px[sel], py[sel], v[sel]))
    out = dict(g)
    out["z"] = z
    out["blocks"] = lab
    return out


def fault_ticks(x, y, side="right", spacing=200.0, length=60.0):
    """Short tick segments on the downthrown side of a fault trace (side relative to the digitising direction).
    Returns NaN-separated (xs, ys) pairs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    xs, ys = [], []
    sgn = -1.0 if side == "right" else 1.0
    for s in np.arange(spacing / 2, d[-1], spacing):
        k = int(np.clip(np.searchsorted(d, s) - 1, 0, len(x) - 2))
        t = (s - d[k]) / max(d[k + 1] - d[k], 1e-12)
        px, py = x[k] + t * (x[k + 1] - x[k]), y[k] + t * (y[k + 1] - y[k])
        tx, ty = x[k + 1] - x[k], y[k + 1] - y[k]
        L = math.hypot(tx, ty) or 1.0
        nx_, ny_ = -ty / L * sgn, tx / L * sgn
        xs += [px, px + length * nx_]
        ys += [py, py + length * ny_]
    return np.array(xs), np.array(ys)


# ------------------------------------------------------------------------------------------- closure / spill
def _component(mask, seed):
    ny, nx = mask.shape
    lab = np.zeros(mask.shape, bool)
    if not mask[seed]:
        return lab
    q = deque([seed])
    lab[seed] = True
    while q:
        i, j = q.popleft()
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < ny and 0 <= b < nx and mask[a, b] and not lab[a, b]:
                lab[a, b] = True
                q.append((a, b))
    return lab


def spill_point(z, crest=None, tol=None):
    """Structural closure on a depth grid (depth positive down; NaN = no data, treated as open).
    crest: (i, j) node (default: the shallowest node).  Returns {"spill_depth", "crest_depth", "relief", "mask"
    (closed area at spill), "spill_node" (i, j), "crest" (i, j)}.  Binary search on the fill level: the closure leaks
    when the crest's connected region (depth <= level) reaches the grid edge or a NaN node."""
    z = np.asarray(z, float)
    if crest is None:
        crest = np.unravel_index(np.nanargmin(z), z.shape)
    zc = float(z[crest])
    ny, nx = z.shape
    border = np.zeros(z.shape, bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    nanb = ~np.isfinite(z)
    # nodes next to NaN count as "open" too
    near_nan = nanb.copy()
    near_nan[1:] |= nanb[:-1]
    near_nan[:-1] |= nanb[1:]
    near_nan[:, 1:] |= nanb[:, :-1]
    near_nan[:, :-1] |= nanb[:, 1:]
    openm = border | near_nan

    def leaks(level):
        comp = _component(np.isfinite(z) & (z <= level), crest)
        return bool((comp & openm).any()), comp

    lo, hi = zc, float(np.nanmax(z))
    if not leaks(hi)[0]:
        hi = hi  # closed everywhere (should not happen with a finite grid edge)
    tol = tol or max((hi - lo) * 1e-4, 1e-6)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if leaks(mid)[0]:
            hi = mid
        else:
            lo = mid
    _, comp = leaks(lo)
    _, comp_hi = leaks(hi)
    newn = comp_hi & ~comp
    leak = newn & openm
    sn = tuple(int(v) for v in np.argwhere(leak if leak.any() else newn)[0]) if newn.any() else tuple(crest)
    return {"spill_depth": lo, "crest_depth": zc, "relief": lo - zc, "mask": comp, "spill_node": sn, "crest": tuple(crest)}


# ------------------------------------------------------------------------------------------- well symbols
# unit-size vertex lists (x, y in -1..1) + fill flags; renderers draw circle + extras
WELL_SYMBOLS = {
    "oil": {"circle": True, "fill": "full", "rays": False, "cross": False, "label": "Oil"},
    "gas": {"circle": True, "fill": "none", "rays": True, "cross": False, "label": "Gas"},
    "oil+gas": {"circle": True, "fill": "full", "rays": True, "cross": False, "label": "Oil and gas"},
    "show": {"circle": True, "fill": "half", "rays": False, "cross": False, "label": "Shows"},
    "dry": {"circle": True, "fill": "none", "rays": False, "cross": False, "label": "Dry hole", "slash": True},
    "p&a": {"circle": True, "fill": "none", "rays": False, "cross": True, "label": "Plugged and abandoned"},
    "planned": {"circle": True, "fill": "none", "rays": False, "cross": False, "label": "Planned / location",
                "dashed": True},
    "injector": {"circle": True, "fill": "none", "rays": False, "cross": False, "label": "Injector", "arrow": True},
    "unknown": {"circle": True, "fill": "dot", "rays": False, "cross": False, "label": "Well"},
}


def classify_status(text):
    t = (text or "").lower()
    if not t:
        return "unknown"
    if "plan" in t or "location" in t or "propos" in t:
        return "planned"
    if "inject" in t or "disposal" in t:
        return "injector"
    oil = "oil" in t or "condensate" in t
    gas = "gas" in t
    if oil and gas:
        return "oil+gas"
    if "p&a" in t or "plugged" in t or "abandon" in t:
        if oil or gas or "show" in t:
            return "show" if "show" in t else ("oil" if oil else "gas")
        return "p&a"
    if "show" in t:
        return "show"
    if "dry" in t or "d&a" in t:
        return "dry"
    if oil:
        return "oil"
    if gas:
        return "gas"
    return "unknown"


def symbol_paths(kind, n=24):
    """Polylines (x, y) of a symbol in unit coordinates (radius 1) and its fill mode - for any renderer."""
    s = WELL_SYMBOLS.get(kind, WELL_SYMBOLS["unknown"])
    a = np.linspace(0, 2 * np.pi, n + 1)
    lines = [(np.cos(a), np.sin(a))]
    if s.get("rays"):
        for k in range(8):
            t = k * np.pi / 4
            lines.append((np.array([np.cos(t), 1.8 * np.cos(t)]), np.array([np.sin(t), 1.8 * np.sin(t)])))
    if s.get("cross"):
        lines += [(np.array([-1.4, 1.4]), np.array([-1.4, 1.4])), (np.array([-1.4, 1.4]), np.array([1.4, -1.4]))]
    if s.get("slash"):
        lines.append((np.array([-1.5, 1.5]), np.array([-1.5, 1.5])))
    if s.get("arrow"):
        lines += [(np.array([0, 0]), np.array([1.0, 2.2])), (np.array([-0.5, 0, 0.5]), np.array([1.6, 1.0, 1.6]))]
    return {"lines": lines, "fill": s["fill"], "dashed": bool(s.get("dashed")), "label": s["label"]}


# ------------------------------------------------------------------------------------------- scale, north, presets
def nice_scale(map_width, fraction=0.25):
    """Scale-bar length (map units) close to fraction x map_width, as 1 / 2 / 2.5 / 5 x 10^k, with 4 or 5 segments."""
    target = max(map_width * fraction, 1e-9)
    mag = 10 ** math.floor(math.log10(target))
    best = min((m * mag for m in (1, 2, 2.5, 5, 10)), key=lambda L: abs(L - target))
    segs = 5 if best / mag in (1, 5, 10) else 4
    unit, div = ("km", 1000.0) if best >= 1000 else ("m", 1.0)
    return {"length": best, "segments": segs, "label": f"{best / div:g} {unit}"}


def north_arrow_angle(lat, lon, lon0):
    """Grid convergence (degrees) of a transverse Mercator projection: true north = grid north rotated by this
    (gamma = atan(tan(lon - lon0) sin(lat)))."""
    return math.degrees(math.atan(math.tan(math.radians(lon - lon0)) * math.sin(math.radians(lat))))


STYLE_PRESETS = {
    "Classic": {"ramp": None, "hillshade": True, "shade_strength": 0.55, "ve": 1.0, "contour_color": (40, 40, 40),
                "index_every": 5, "labels": True, "label_size": 8, "halo": True, "well_size": 7},
    "Print": {"ramp": None, "hillshade": True, "shade_strength": 0.45, "ve": 1.0, "contour_color": (20, 20, 20),
              "index_every": 5, "labels": True, "label_size": 7, "halo": True, "well_size": 6},
    "Flat colour": {"ramp": None, "hillshade": False, "shade_strength": 0.0, "ve": 1.0, "contour_color": (60, 60, 60),
                    "index_every": 5, "labels": True, "label_size": 8, "halo": True, "well_size": 7},
    "Contours only": {"ramp": None, "hillshade": False, "shade_strength": 0.0, "ve": 1.0, "contour_color": (0, 0, 0),
                      "index_every": 5, "labels": True, "label_size": 8, "halo": True, "well_size": 7, "fill": False},
}


def styled_map(g, depth_like=True, ci=None, levels=None, hill=True, strength=0.5, alpha=200, label_every=1,
               char=None):
    """One call for a dict grid: hillshaded colour fill (depth ramp, light = shallow, lit as a surface: elevation =
    -depth), contour segments per level and non-overlapping labels along the lines.
    Returns {"rgba", "segments" {level: (xs, ys)}, "labels" [(x, y, angle, text)], "levels"}."""
    from .mapping import contour_segments
    z = np.asarray(g["z"], float)
    ramp = "depth" if depth_like else "thickness"
    lut = ramp_lut(ramp)
    shade = hillshade(-z if depth_like else z, g["dx"], g["dy"]) if hill else None
    rgba = shaded_rgba(z, lut, shade=shade, strength=strength if hill else 0.0, alpha=alpha / 255.0)
    if levels is None:
        lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
        if ci is None:
            span = max(hi - lo, 1e-6)
            mag = 10 ** math.floor(math.log10(span / 10))
            ci = next((m * mag for m in (1, 2, 2.5, 5, 10) if span / 10 <= m * mag), 10 * mag)
        levels = list(np.arange(math.ceil(lo / ci) * ci, hi, ci))
    gg = {"z": z, "x0": g["x0"], "y0": g["y0"], "dx": g["dx"], "dy": g["dy"], "nx": z.shape[1], "ny": z.shape[0]}
    segs = contour_segments(gg, levels)
    ext = max(z.shape[1] * abs(g["dx"]), z.shape[0] * abs(g["dy"]))
    cw = char or ext / 120.0
    cands = []
    for k, (lev, (xs, ys)) in enumerate(sorted(segs.items())):
        if label_every > 1 and k % label_every:
            continue
        cands += label_candidates(chain_segments(xs, ys), f"{lev:g}", ext / 4.0)
    return {"rgba": rgba, "segments": segs, "labels": place_labels(cands, cw, 1.4 * cw), "levels": levels}
