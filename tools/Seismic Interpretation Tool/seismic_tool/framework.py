"""Structural framework + 3D grid (PLAN 35) - numpy only.

35a Fault modelling: item-2 / item-32 faults (sticks -> triangulated surfaces) with editing helpers (`smooth_sticks`,
    `remove_stick`, `add_stick`) and fault TRACES on a horizon (`fault_trace`: where the fault surface cuts the horizon,
    in map coordinates) - shared with the Live Map gridding as barriers (polygons layer, item 32f / 31g).
35b Horizon modelling: each horizon gridded per fault block (`carto.grid_by_blocks`: data points only influence
    their own block) on one framework geometry; `FrameworkModel.surfaces` keeps the per-block extrapolated grids so
    a cell next to a fault uses its own block's surface (stair-stepped faults).  Rules between horizons
    (`apply_rules`): 'conformable' (no crossing: clipped below the horizon above), 'erosional' (truncates the older
    horizons below it), 'base' (onlaps: clipped above the horizon below); `crossings` reports them.
35c Depth: `to_depth(model, depth_fn)` with a time -> depth function (item-17 VelocityModel / DynamicDepthModel
    depth_of_time or the item-37 velocity volume) - rebuilt on a 'velocities' DataChanged event.
35d 3D grid: `Grid3D` stair-stepped corner-point grid between the depth horizons - areal origin, cell size and
    rotation, zones, layering per zone ('proportional' n layers, 'top' / 'base' = follow top / base with a constant
    layer thickness, 'constant'), fault-block and zone index per cell, `quality()` (cell volumes, zero / negative
    thickness, ACTNUM).  Full corner-point grids with sloping pillars and fault juxtaposition are out of scope.
35e Output: `write_grdecl` (SPECGRID, COORD, ZCORN, ACTNUM + integer properties ZONE, FAULTBLK, FIPNUM), `read_grdecl`
    (round trip), surfaces to ZMAP+ / XYZ.
"""
import math

import numpy as np


# ====================================================================== 35a faults
def smooth_sticks(fault, passes=1):
    """Smooth each stick (3-point moving average of the inner points) and neighbouring sticks along strike."""
    st = [np.asarray(s, float).copy() for s in fault.ordered_sticks()]
    for _ in range(max(1, int(passes))):
        for s in st:
            if len(s) >= 3:
                s[1:-1] = (s[:-2] + 2 * s[1:-1] + s[2:]) / 4.0
    fault.sticks = st
    fault.invalidate()
    return fault


def remove_stick(fault, k):
    st = fault.ordered_sticks()
    fault.sticks = [s for q, s in enumerate(st) if q != k]
    fault.invalidate()
    return fault


def add_stick(fault, pts):
    fault.add_stick(np.asarray(pts, float))
    return fault


def fault_trace(fault, grid):
    """Where a fault surface cuts a horizon: map polyline points [(x, y)] (ordered along strike).  grid = MapGrid of
    horizon times (or depths with fault sticks in the same domain); triangle edges whose ends lie on opposite sides
    of the horizon give the crossing points."""
    tris = fault.triangles()
    if not len(tris):
        return np.zeros((0, 2))
    P = np.asarray(tris, float).reshape(-1, 3)
    h = np.asarray(grid.sample(P[:, 0], P[:, 1]), float)
    bad = ~np.isfinite(h)
    if bad.any():  # vertices on / just outside the grid edge: sample slightly towards the triangle centre
        C = np.repeat(np.asarray(tris, float).reshape(-1, 3, 3).mean(axis=1), 3, axis=0)
        for f_ in (0.02, 0.1, 0.3):
            q = P[bad, :2] + f_ * (C[bad, :2] - P[bad, :2])
            hv = np.asarray(grid.sample(q[:, 0], q[:, 1]), float)
            idx = np.flatnonzero(bad)
            h[idx] = np.where(np.isfinite(hv), hv, h[idx])
            bad = ~np.isfinite(h)
            if not bad.any():
                break
    d = (P[:, 2] - h).reshape(-1, 3)
    P = P.reshape(-1, 3, 3)
    pts = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        da, db = d[:, a], d[:, b]
        m = np.isfinite(da) & np.isfinite(db) & (np.sign(da) != np.sign(db))
        w = da[m] / (da[m] - db[m])
        q = P[m, a, :2] + w[:, None] * (P[m, b, :2] - P[m, a, :2])
        pts.append(q)
    q = np.concatenate(pts) if pts else np.zeros((0, 2))
    if len(q) < 2:
        return q
    # order along the principal direction (strike)
    c = q - q.mean(axis=0)
    u = np.linalg.svd(c, full_matrices=False)[2][0]
    o = np.argsort(c @ u)
    q = q[o]
    keep = np.r_[True, np.hypot(*np.diff(q, axis=0).T) > 1e-6]
    return q[keep]


def extend_trace(tr, dist):
    """Extend a polyline at both ends by dist along its end segments (a fault that ends at the survey edge must cut
    the framework grid completely to separate the blocks)."""
    tr = np.asarray(tr, float)
    if len(tr) < 2:
        return tr
    a = tr[0] - tr[1]
    b = tr[-1] - tr[-2]
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return np.vstack([tr[0] + dist * a, tr, tr[-1] + dist * b])


# ====================================================================== 35b horizons
class Geometry:
    """Areal framework / grid geometry: origin (x0, y0), cell size (dx, dy), size (nx, ny), rotation (deg, counter-
    clockwise from east) of the I axis."""

    def __init__(self, x0, y0, dx, dy, nx, ny, rotation=0.0):
        self.x0, self.y0, self.dx, self.dy = float(x0), float(y0), float(dx), float(dy)
        self.nx, self.ny = int(nx), int(ny)
        self.rotation = float(rotation)

    def nodes(self):
        """Node (corner) coordinates (ny + 1, nx + 1) - the pillars of the 3D grid."""
        I, J = np.meshgrid(np.arange(self.nx + 1) * self.dx, np.arange(self.ny + 1) * self.dy)
        return self._rot(I, J)

    def centres(self):
        I, J = np.meshgrid((np.arange(self.nx) + 0.5) * self.dx, (np.arange(self.ny) + 0.5) * self.dy)
        return self._rot(I, J)

    def _rot(self, u, v):
        a = math.radians(self.rotation)
        return self.x0 + u * math.cos(a) - v * math.sin(a), self.y0 + u * math.sin(a) + v * math.cos(a)

    @classmethod
    def around(cls, x, y, cell, pad=0.0, rotation=0.0):
        x, y = np.asarray(x, float), np.asarray(y, float)
        x0, x1, y0, y1 = x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad
        return cls(x0, y0, cell, cell, max(1, int(math.ceil((x1 - x0) / cell))), max(1, int(math.ceil((y1 - y0) / cell))),
                   rotation)

    def to_dict(self):
        return {"x0": self.x0, "y0": self.y0, "dx": self.dx, "dy": self.dy, "nx": self.nx, "ny": self.ny,
                "rotation": self.rotation}


def fault_blocks(X, Y, faults):
    """Block id per node (ny, nx) from fault polylines [(xs, ys)] cutting the 4-neighbour node graph (any geometry)."""
    from .welltie import wc
    wc()
    from well_correlation.carto import _seg_cross
    ny, nx = X.shape
    cut_h = np.zeros((ny, nx - 1), bool)
    cut_v = np.zeros((ny - 1, nx), bool)
    for fx, fy in faults:
        for k in range(len(fx) - 1):
            cut_h |= _seg_cross(X[:, :-1], Y[:, :-1], X[:, 1:], Y[:, 1:], fx[k], fy[k], fx[k + 1], fy[k + 1])
            cut_v |= _seg_cross(X[:-1], Y[:-1], X[1:], Y[1:], fx[k], fy[k], fx[k + 1], fy[k + 1])
    # union-find over uncut edges (vectorised label propagation)
    idx = np.arange(ny * nx).reshape(ny, nx)
    a = np.concatenate([idx[:, :-1][~cut_h], idx[:-1][~cut_v]])
    b = np.concatenate([idx[:, 1:][~cut_h], idx[1:][~cut_v]])
    from .geobody import _propagate
    lbl = _propagate(np.arange(ny * nx), a, b)
    _, comp = np.unique(lbl, return_inverse=True)
    return comp.reshape(ny, nx)


class FrameworkModel:
    """Horizons [(name, (px, py, t) data points, rule)] top to bottom on one Geometry, faults [(name, (xs, ys) trace)]
    as barriers.  build() -> self.blocks (node block ids), self.surfaces[k] (nb, ny + 1, nx + 1) per-block surfaces,
    self.grids[k] (ny + 1, nx + 1) each node from its own block."""

    def __init__(self, geometry, horizons, faults=(), method="tps", domain="time"):
        self.geom = geometry
        self.horizons = [(n, tuple(np.asarray(a, float) for a in d), r) for n, d, r in horizons]
        self.faults = list(faults)
        self.method = method
        self.domain = domain
        self.report = {}

    def build(self):
        from .vgrid import grid_values
        X, Y = self.geom.nodes()
        self.X, self.Y = X, Y
        traces = [tr for _, tr in self.faults]
        self.blocks = fault_blocks(X, Y, traces) if traces else np.zeros(X.shape, int)
        nb = int(self.blocks.max()) + 1
        self.nblocks = nb
        # block of every data point = block of its nearest node
        self.surfaces, self.grids = [], []
        for name, (px, py, pv), rule in self.horizons:
            ok = np.isfinite(px) & np.isfinite(py) & np.isfinite(pv)
            px, py, pv = px[ok], py[ok], pv[ok]
            pb = self._point_blocks(px, py)
            S = np.full((nb,) + X.shape, np.nan)
            for b in range(nb):
                sel = pb == b
                if sel.sum() < 1:
                    sel = np.ones(len(pv), bool)  # a block without data: all points (no throw information)
                meth = self.method if sel.sum() >= 3 and sel.sum() <= 1500 else "idw"
                S[b] = np.asarray(grid_values(X.ravel(), Y.ravel(), px[sel], py[sel], pv[sel], meth), float).reshape(
                    X.shape)
            self.surfaces.append(S)
            self.grids.append(np.take_along_axis(S, self.blocks[None], 0)[0])
        self.report["crossings_before"] = crossings(self.grids)
        self.apply_rules()
        self.report["crossings_after"] = crossings(self.grids)
        return self

    def _point_blocks(self, px, py):
        g = self.geom
        a = math.radians(g.rotation)
        u = (px - g.x0) * math.cos(a) + (py - g.y0) * math.sin(a)
        v = -(px - g.x0) * math.sin(a) + (py - g.y0) * math.cos(a)
        i = np.clip(np.round(v / g.dy).astype(int), 0, g.ny)
        j = np.clip(np.round(u / g.dx).astype(int), 0, g.nx)
        return self.blocks[i, j]

    def apply_rules(self):
        """Conformable / erosional / base rules applied to the per-block surfaces and the node grids."""
        rules = [r for _, _, r in self.horizons]
        for S in (self.surfaces, None):
            arrs = self.surfaces if S is not None else self.grids
            for k in range(1, len(arrs)):
                if rules[k] in ("conformable", "erosional", None, ""):
                    arrs[k] = np.fmax(arrs[k], arrs[k - 1])  # never above the horizon above
            for k, r in enumerate(rules):
                if r == "erosional":
                    for j in range(k + 1, len(arrs)):
                        arrs[j] = np.fmax(arrs[j], arrs[k])  # older horizons cut by the unconformity
            for k in range(len(arrs) - 2, -1, -1):
                if rules[k] == "base":
                    arrs[k] = np.fmin(arrs[k], arrs[k + 1])  # onlap: clipped above the horizon below
        return self.grids

    def to_depth(self, depth_fn):
        """New FrameworkModel in depth: depth_fn(x, y, t) -> z for arrays (VelocityModel.depth_of_time-like for
        per-point times) applied to every per-block surface; rules re-applied."""
        m = FrameworkModel(self.geom, self.horizons, self.faults, self.method, "depth")
        m.X, m.Y, m.blocks, m.nblocks = self.X, self.Y, self.blocks, self.nblocks
        m.surfaces = [np.stack([np.asarray(depth_fn(self.X, self.Y, S[b]), float) for b in range(len(S))])
                      for S in self.surfaces]
        m.grids = [np.take_along_axis(S, m.blocks[None], 0)[0] for S in m.surfaces]
        m.apply_rules()
        m.report = {"crossings_after": crossings(m.grids)}
        return m

    def horizon_grid(self, k):
        from .maps import MapGrid
        if self.geom.rotation:
            raise ValueError("MapGrid export needs an unrotated framework geometry")
        unit = "ms" if self.domain == "time" else "m"
        return MapGrid(self.grids[k], "xy", unit, f"{self.horizons[k][0]} ({self.domain}, framework)", None,
                       self.geom.x0, self.geom.y0, self.geom.dx, self.geom.dy)


def depth_fn_from_model(vm):
    """time -> depth function of an item-17 VelocityModel / DynamicDepthModel / item-37 VelocityVolumeModel for
    framework surfaces (point-wise: one time per x, y)."""
    model = getattr(vm, "vm", None) or vm

    def fn(X, Y, T):
        X, Y, T = (np.asarray(a, float) for a in (X, Y, T))
        out = np.full(T.shape, np.nan)
        flat = np.isfinite(T.ravel())
        xs, ys, ts = X.ravel()[flat], Y.ravel()[flat], T.ravel()[flat]
        # depth_of_time evaluates many times per point; group by rounded time to keep it vectorised
        z = np.empty(len(ts))
        for a in range(0, len(ts), 512):  # depth_of_time returns (points x times): keep the square small
            b = min(len(ts), a + 512)
            zz = np.asarray(model.depth_of_time(xs[a:b], ys[a:b], ts[a:b]), float)
            z[a:b] = np.diagonal(zz) if zz.ndim == 2 and zz.shape[0] == zz.shape[1] == b - a else np.ravel(zz)[:b - a]
        o = out.ravel()
        o[flat] = z
        return o.reshape(T.shape)
    return fn


def crossings(grids):
    """Nodes where a horizon lies above the one above it (count per horizon pair)."""
    out = []
    for k in range(1, len(grids)):
        a, b = grids[k - 1], grids[k]
        ok = np.isfinite(a) & np.isfinite(b)
        out.append(int(np.sum(b[ok] < a[ok] - 1e-6)))
    return out


# ====================================================================== 35d 3D grid
def cell_blocks(model, Xc, Yc):
    """Fault block of each cell centre (ny, nx): the block of the first corner node reachable without crossing a
    fault trace (a centre-to-nearest-node rule fails when a fault passes between the centre and that node)."""
    X, Y, B = model.X, model.Y, model.blocks
    ny, nx = Xc.shape
    out = np.full((ny, nx), -1, int)
    traces = [tr for _, tr in model.faults]
    if not traces:
        return np.zeros((ny, nx), int) + int(B[0, 0])
    from .welltie import wc
    wc()
    from well_correlation.carto import _seg_cross
    for di, dj in ((0, 0), (0, 1), (1, 0), (1, 1)):
        cx, cy = X[di:di + ny, dj:dj + nx], Y[di:di + ny, dj:dj + nx]
        cross = np.zeros((ny, nx), bool)
        for fx, fy in traces:
            for k in range(len(fx) - 1):
                cross |= _seg_cross(Xc, Yc, cx, cy, fx[k], fy[k], fx[k + 1], fy[k + 1])
        take = (out < 0) & ~cross
        out[take] = B[di:di + ny, dj:dj + nx][take]
    out[out < 0] = B[:ny, :nx][out < 0]
    return out


class Grid3D:
    """Stair-stepped corner-point grid.  zcorn (2 nz, 2 ny, 2 nx) like Eclipse ZCORN; coord pillars vertical at the
    nodes; zone (nz,) per layer; block (ny, nx) per column; actnum (nz, ny, nx)."""

    def __init__(self, model, layering, min_thickness=0.01):
        """layering: per zone ('proportional', n) | ('top', dz) | ('base', dz) | ('constant', dz)."""
        if len(layering) != len(model.grids) - 1:
            raise ValueError(f"{len(model.grids) - 1} zones need {len(model.grids) - 1} layering specs")
        self.model = model
        self.geom = model.geom
        self.layering = [tuple(l) for l in layering]
        self.min_thickness = float(min_thickness)
        self._build()

    def _zone_layers(self, top, base, spec):
        """Layer boundaries (n + 1, ...) between top and base surfaces (arrays of any shape)."""
        kind, val = spec
        th = base - top
        if kind == "proportional":
            n = int(val)
            f = np.linspace(0, 1, n + 1).reshape((-1,) + (1,) * top.ndim)
            return top[None] + f * th[None]
        dz = float(val)
        tmax = float(np.nanmax(th)) if np.isfinite(th).any() else dz
        n = max(1, int(math.ceil(tmax / dz - 1e-9)))
        k = (np.arange(n + 1) * dz).reshape((-1,) + (1,) * top.ndim)
        if kind in ("top", "constant"):
            return np.minimum(top[None] + k, base[None])  # follow top, truncated by the base
        return np.maximum(base[None] - k[::-1], top[None])  # follow base, truncated by the top

    def _build(self):
        m, g = self.model, self.geom
        ny, nx = g.ny, g.nx
        # block of each column = block of a corner node on the same side of every fault as the cell centre
        Xc, Yc = g.centres()
        self.block = cell_blocks(m, Xc, Yc)
        # corner depths of each column from its OWN block's surfaces (stair steps at the faults)
        corners = []  # per horizon: (ny, nx, 4) corner values (sw, se, nw, ne)
        for S in m.surfaces:
            jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
            b = self.block
            c = np.stack([S[b, ii, jj], S[b, ii, jj + 1], S[b, ii + 1, jj], S[b, ii + 1, jj + 1]], -1)
            corners.append(c)
        layers, zone = [], []
        for z, spec in enumerate(self.layering):
            L = self._zone_layers(corners[z], corners[z + 1], spec)  # (n + 1, ny, nx, 4)
            layers.append(L if z == 0 else L[1:])
            zone += [z + 1] * (len(L) - 1)
        B = np.concatenate([layers[0]] + layers[1:], 0)  # (nz + 1, ny, nx, 4) layer boundaries
        nz = len(B) - 1
        self.nz, self.ny, self.nx = nz, ny, nx
        self.zone = np.array(zone, int)
        zc = np.empty((2 * nz, 2 * ny, 2 * nx))
        for k in range(nz):
            for half, Bk in ((0, B[k]), (1, B[k + 1])):
                zc[2 * k + half, 0::2, 0::2] = Bk[..., 0]
                zc[2 * k + half, 0::2, 1::2] = Bk[..., 1]
                zc[2 * k + half, 1::2, 0::2] = Bk[..., 2]
                zc[2 * k + half, 1::2, 1::2] = Bk[..., 3]
        self.zcorn = zc
        th = self.thickness()
        self.actnum = (np.isfinite(th) & (th > self.min_thickness)).astype(np.int8)

    def thickness(self):
        """Mean cell thickness (nz, ny, nx) from the 8 corners."""
        top = self.zcorn[0::2].reshape(self.nz, self.ny, 2, self.nx, 2).mean(axis=(2, 4))
        bot = self.zcorn[1::2].reshape(self.nz, self.ny, 2, self.nx, 2).mean(axis=(2, 4))
        return bot - top

    def cell_volumes(self):
        return self.thickness() * self.geom.dx * self.geom.dy

    def coord(self):
        """Eclipse COORD: one vertical pillar per node (top and bottom points)."""
        X, Y = self.geom.nodes()
        zt = float(np.nanmin(self.zcorn))
        zb = float(np.nanmax(self.zcorn))
        P = np.stack([X, Y, np.full(X.shape, zt), X, Y, np.full(X.shape, zb)], -1)
        return P.reshape(-1, 6)

    def zone_index(self):
        return np.broadcast_to(self.zone[:, None, None], (self.nz, self.ny, self.nx)).astype(int)

    def block_index(self):
        return np.broadcast_to(self.block[None] + 1, (self.nz, self.ny, self.nx)).astype(int)

    def quality(self):
        th = self.thickness()
        v = self.cell_volumes()
        return {"cells": int(self.nz * self.ny * self.nx), "active": int(self.actnum.sum()),
                "zero_thickness": int(np.sum(np.abs(th) <= self.min_thickness)),
                "negative_thickness": int(np.sum(th < -self.min_thickness)),
                "volume_m3": float(np.nansum(v[self.actnum > 0])), "min_thickness": float(np.nanmin(th)),
                "max_thickness": float(np.nanmax(th)), "layers_per_zone": np.bincount(self.zone).tolist()[1:],
                "fault_blocks": int(self.block.max()) + 1}

    def zone_thickness(self, z):
        """Total thickness of zone z (1-based) per column (ny, nx)."""
        th = self.thickness()
        return np.nansum(np.where(self.zone[:, None, None] == z, th, 0.0), axis=0)


# ====================================================================== 35e output
def _fmt(vals, per_line=6, fmt="{:.4f}"):
    out, line = [], []
    for v in np.ravel(vals):
        line.append(fmt.format(v))
        if len(line) == per_line:
            out.append(" ".join(line))
            line = []
    if line:
        out.append(" ".join(line))
    return "\n".join(out)


def _ints(vals, per_line=20):
    """Run-length encoded integers (n*v) as Eclipse allows."""
    v = np.ravel(np.asarray(vals, int))
    out, items = [], []
    k = 0
    while k < len(v):
        j = k
        while j + 1 < len(v) and v[j + 1] == v[k]:
            j += 1
        n = j - k + 1
        items.append(f"{n}*{v[k]}" if n > 1 else str(v[k]))
        k = j + 1
    for a in range(0, len(items), per_line):
        out.append(" ".join(items[a:a + per_line]))
    return "\n".join(out)


def write_grdecl(path, grid, props=None, title="GeoHarmonic structural grid"):
    """Eclipse / OPM Flow GRDECL: SPECGRID, COORD, ZCORN (Eclipse order: i fastest), ACTNUM + integer ZONE,
    FAULTBLK, FIPNUM (= zone) and any extra props {name: (nz, ny, nx) array} (floats)."""
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    with open(path, "w") as f:
        f.write(f"-- {title}\n-- {nx} x {ny} x {nz} cells, stair-stepped faults\n\n")
        f.write(f"SPECGRID\n{nx} {ny} {nz} 1 F /\n\n")
        f.write("COORD\n" + _fmt(grid.coord()) + "\n/\n\n")
        f.write("ZCORN\n" + _fmt(np.nan_to_num(grid.zcorn, nan=0.0)) + "\n/\n\n")
        f.write("ACTNUM\n" + _ints(grid.actnum) + "\n/\n\n")
        f.write("ZONE\n" + _ints(grid.zone_index()) + "\n/\n\n")
        f.write("FAULTBLK\n" + _ints(grid.block_index()) + "\n/\n\n")
        f.write("FIPNUM\n" + _ints(grid.zone_index()) + "\n/\n\n")
        for name, arr in (props or {}).items():
            f.write(f"{name.upper()[:8]}\n" + _fmt(np.nan_to_num(np.asarray(arr, float))) + "\n/\n\n")
    return path


def read_grdecl(path):
    """{keyword: flat numpy array} incl. SPECGRID; expands n*v repeats."""
    out, key, vals = {}, None, []
    with open(path) as f:
        for raw in f:
            line = raw.split("--")[0].strip()
            if not line:
                continue
            if key is None:
                key = line.split()[0]
                vals = []
                rest = line[len(key):].strip()
                if not rest:
                    continue
                line = rest
            end = "/" in line
            line = line.split("/")[0]
            for tok in line.split():
                if "*" in tok:
                    n, v = tok.split("*")
                    vals += [float(v)] * int(n)
                elif tok in ("F", "T"):
                    continue
                else:
                    vals.append(float(tok))
            if end:
                out[key] = np.array(vals)
                key = None
    return out


def write_surfaces(model, folder, fmt="zmap"):
    """Framework horizons as ZMAP+ (or XYZ) files, one per horizon."""
    import os
    os.makedirs(folder, exist_ok=True)
    paths = []
    for k, (name, _, _) in enumerate(model.horizons):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        if fmt == "zmap":
            from .gis.zmap import write_zmap
            paths.append(write_zmap(os.path.join(folder, f"{safe}.zmap"), model.horizon_grid(k), name))
        else:
            X, Y = model.geom.nodes()
            p = os.path.join(folder, f"{safe}.xyz")
            g = model.grids[k]
            ok = np.isfinite(g)
            np.savetxt(p, np.column_stack([X[ok], Y[ok], g[ok]]), fmt="%.3f")
            paths.append(p)
    return paths
