"""Faults: sticks drawn on sections, triangulated fault surfaces, section/time-slice intersections, and the
'blocked' callback that stops the horizon tracker at faults.

Stick points are stored in map coordinates (x, y, t ms) so a fault drawn on a 3D inline shows on 2D lines
and other surveys too.  A fault surface is built by ordering the sticks along strike (principal axis of the
stick centres) and zipping consecutive sticks into triangles.
"""
import itertools

import numpy as np

from .horizons import DEFAULT_COLORS

_SEQ = itertools.count(1)  # shared ordering for undo across horizons and faults


def next_seq():
    return next(_SEQ)


class Fault:
    def __init__(self, name, color=None):
        self.name = name
        self.color = tuple(color) if color else (220, 30, 30)
        self.sticks = []  # list of float arrays (n, 3): x, y, t_ms; points sorted by time
        self.visible = True
        self._tri = None

    def invalidate(self):
        self._tri = None

    def add_stick(self, pts):
        pts = np.asarray(pts, float).reshape(-1, 3)
        pts = pts[np.argsort(pts[:, 2], kind="stable")]
        self.sticks.append(pts)
        self.invalidate()
        return len(self.sticks) - 1

    def ordered_sticks(self):
        st = [s for s in self.sticks if len(s) >= 2]
        if len(st) < 2:
            return st
        cen = np.array([s[:, :2].mean(axis=0) for s in st])
        c = cen - cen.mean(axis=0)
        if np.allclose(c, 0):
            return st
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        order = np.argsort(c @ vt[0])
        return [st[k] for k in order]

    def triangles(self):
        """(n_tri, 3, 3) array of triangle vertices (x, y, t)."""
        if self._tri is None:
            st = self.ordered_sticks()
            tris = []
            for A, B in zip(st[:-1], st[1:]):
                tris.extend(zip_sticks(A, B))
            self._tri = np.array(tris, float).reshape(-1, 3, 3)
        return self._tri


def zip_sticks(A, B):
    """Triangulate the band between two sticks (each sorted by time) - shorter diagonal first."""
    out = []
    i = j = 0
    while i < len(A) - 1 or j < len(B) - 1:
        if j == len(B) - 1 or (i < len(A) - 1 and
                               np.linalg.norm(A[i + 1] - B[j]) <= np.linalg.norm(A[i] - B[j + 1])):
            out.append((A[i], A[i + 1], B[j]))
            i += 1
        else:
            out.append((A[i], B[j + 1], B[j]))
            j += 1
    return out


def plane_cut(tri_u, tri_v, tri_w, level):
    """Cut triangles with the plane u = level. tri_*: (n, 3) per-vertex coordinates.
    Returns (v0, w0, v1, w1) arrays of segment end points for the triangles that straddle the plane.
    A plane through the end stick itself (all vertices on one side or on the plane) is cut just inside."""
    r = _plane_cut(tri_u, tri_v, tri_w, level + 1e-6)
    if not len(r[0]):
        r = _plane_cut(tri_u, tri_v, tri_w, level - 1e-6)
    return r


def _plane_cut(tri_u, tri_v, tri_w, level):
    d = tri_u - level
    segs = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        da, db = d[:, a], d[:, b]
        cross = ((da <= 0) & (db > 0)) | ((da > 0) & (db <= 0))
        f = np.where(cross, da / np.where(da - db == 0, 1, da - db), np.nan)
        segs.append((tri_v[:, a] + f * (tri_v[:, b] - tri_v[:, a]), tri_w[:, a] + f * (tri_w[:, b] - tri_w[:, a])))
    V = np.stack([s[0] for s in segs], axis=1)  # (n, 3) - two finite per straddling triangle
    W = np.stack([s[1] for s in segs], axis=1)
    fin = np.isfinite(V)
    keep = fin.sum(axis=1) >= 2
    V, W, fin = V[keep], W[keep], fin[keep]
    order = np.argsort(~fin, axis=1, kind="stable")[:, :2]
    r = np.arange(len(V))[:, None]
    v2, w2 = V[r, order], W[r, order]
    return v2[:, 0], w2[:, 0], v2[:, 1], w2[:, 1]


# ====================================================================== survey index space
def tri_index_coords(vol, tris):
    """Triangles (n,3,3) x,y,t -> fractional (il index, xl index, t ms) arrays (n,3) each."""
    a, b = vol.ax, vol.ay
    M = np.array([[a[1], a[2]], [b[1], b[2]]])
    Minv = np.linalg.inv(M)
    x = tris[..., 0] - a[0]
    y = tris[..., 1] - b[0]
    il = Minv[0, 0] * x + Minv[0, 1] * y
    xl = Minv[1, 0] * x + Minv[1, 1] * y
    return (il - vol.il0) / vol.il_step, (xl - vol.xl0) / vol.xl_step, tris[..., 2]


def section_segments_3d(fault, vol, mode, number):
    """Fault surface cut by an inline / crossline (segments in (xline or inline number, t ms)) or a time
    slice (segments in (xline number, inline number)). Returns (x0, y0, x1, y1) arrays."""
    tris = fault.triangles()
    if not len(tris):
        return None
    ii, jj, tt = tri_index_coords(vol, tris)
    if mode == "Inline":
        v0, w0, v1, w1 = plane_cut(ii, jj, tt, vol.il_index(number))
        return vol.xl0 + vol.xl_step * v0, w0, vol.xl0 + vol.xl_step * v1, w1
    if mode == "Crossline":
        v0, w0, v1, w1 = plane_cut(jj, ii, tt, vol.xl_index(number))
        return vol.il0 + vol.il_step * v0, w0, vol.il0 + vol.il_step * v1, w1
    v0, w0, v1, w1 = plane_cut(tt, jj, ii, float(number))
    return vol.xl0 + vol.xl_step * v0, vol.il0 + vol.il_step * w0, vol.xl0 + vol.xl_step * v1, \
        vol.il0 + vol.il_step * w1


def section_segments_line(fault, lx, ly, step=25):
    """Fault surface cut by a 2D line (trace positions lx, ly). Returns (k0, t0, k1, t1) with k = fractional
    trace index.  The line is treated as straight pieces of `step` traces."""
    tris = fault.triangles()
    if not len(tris) or len(lx) < 2:
        return None
    n = len(lx)
    verts = np.unique(np.r_[np.arange(0, n, step), n - 1])
    out = [[], [], [], []]
    for a, b in zip(verts[:-1], verts[1:]):
        p0 = np.array([lx[a], ly[a]], float)
        dvec = np.array([lx[b] - lx[a], ly[b] - ly[a]], float)
        L = np.hypot(*dvec)
        if L == 0:
            continue
        u = dvec / L
        rel = tris[..., :2] - p0
        dist = rel[..., 0] * u[1] - rel[..., 1] * u[0]  # signed distance from the vertical plane
        along = rel[..., 0] * u[0] + rel[..., 1] * u[1]
        s0, t0, s1, t1 = plane_cut(dist, along, tris[..., 2], 0.0)
        smid = 0.5 * (s0 + s1)
        keep = (smid >= 0) & (smid <= L)
        k0 = a + (b - a) * s0[keep] / L
        k1 = a + (b - a) * s1[keep] / L
        for lst, arr in zip(out, (k0, t0[keep], k1, t1[keep])):
            lst.append(arr)
    return tuple(np.concatenate(l) if l else np.array([]) for l in out)


def sticks_on_section_3d(fault, vol, mode, number, tol=0.5):
    """Sticks lying on this inline/crossline (within tol traces): list of (x values, t ms)."""
    out = []
    for s in fault.sticks:
        ii, jj, tt = tri_index_coords(vol, s[None])
        ii, jj, tt = ii[0], jj[0], tt[0]
        if mode == "Inline" and np.all(np.abs(ii - vol.il_index(number)) <= tol):
            out.append((vol.xl0 + vol.xl_step * jj, tt))
        elif mode == "Crossline" and np.all(np.abs(jj - vol.xl_index(number)) <= tol):
            out.append((vol.il0 + vol.il_step * ii, tt))
    return out


def sticks_on_line(fault, lx, ly, tol_m=None):
    """Sticks drawn on (or within tol of) a 2D line: list of (fractional trace index, t ms)."""
    lx, ly = np.asarray(lx, float), np.asarray(ly, float)
    if tol_m is None:
        tol_m = 1.5 * float(np.median(np.hypot(np.diff(lx), np.diff(ly)))) if len(lx) > 1 else 1.0
    out = []
    for s in fault.sticks:
        d = np.hypot(lx[None] - s[:, 0:1], ly[None] - s[:, 1:2])
        k = np.argmin(d, axis=1)
        if np.all(d[np.arange(len(s)), k] <= tol_m):
            out.append((k.astype(float), s[:, 2]))
    return out


# ====================================================================== tracker barrier
class EdgeBarrier:
    """Blocks tracker links that cross a fault.  Built from the fault cuts on every inline and crossline (3D)
    or along the line (2D), stored per grid edge as time ranges [tmin, tmax] (sample index units)."""

    def __init__(self, shape, tol=2.0):
        self.ni, self.nj = shape
        self.tol = tol
        self.edges = {0: ([], [], []), 1: ([], [], [])}  # axis -> (key, tmin, tmax)

    def add(self, axis, line_idx, x0, t0, x1, t1):
        """Segments on grid line `line_idx` (an i-row for axis 1 edges, a j-column for axis 0 edges) in
        (fractional position along the line, t) -> mark every unit edge they cross."""
        for a, ta, b, tb in zip(x0, t0, x1, t1):
            if a > b:
                a, b, ta, tb = b, a, tb, ta
            k0, k1 = int(np.floor(a)), int(np.floor(b))
            for k in range(max(k0, 0), min(k1, (self.nj if axis == 1 else self.ni) - 2) + 1):
                lo, hi = max(a, k), min(b, k + 1)
                if hi < lo:
                    continue
                f = lambda x: ta + (tb - ta) * ((x - a) / (b - a) if b > a else 0.0)  # noqa: E731
                key = line_idx * (self.nj if axis == 1 else self.ni) + k
                self.edges[axis][0].append(key)
                self.edges[axis][1].append(min(f(lo), f(hi)))
                self.edges[axis][2].append(max(f(lo), f(hi)))

    def finish(self):
        self._arr = {}
        for ax, (k, lo, hi) in self.edges.items():
            k, lo, hi = np.asarray(k, np.int64), np.asarray(lo, float), np.asarray(hi, float)
            o = np.argsort(k, kind="stable")
            self._arr[ax] = (k[o], lo[o], hi[o])
        return self

    def blocked(self, grid_idx):
        """Returns fn(pi, pj, ci, cj) for track_grid, using the parent's current time from grid_idx."""
        def fn(pi, pj, ci, cj):
            out = np.zeros(len(pi), bool)
            tp = grid_idx[pi, pj]
            for ax in (0, 1):
                keys, lo, hi = self._arr[ax]
                if not len(keys):
                    continue
                sel = (ci != pi) if ax == 0 else (cj != pj)
                if not sel.any():
                    continue
                if ax == 1:
                    line, a, b = pi[sel], np.minimum(pj[sel], cj[sel]), np.maximum(pj[sel], cj[sel])
                    n_along = self.nj
                else:
                    line, a, b = pj[sel], np.minimum(pi[sel], ci[sel]), np.maximum(pi[sel], ci[sel])
                    n_along = self.ni
                hit = np.zeros(sel.sum(), bool)
                t = tp[sel]
                for s in range(int((b - a).max())):
                    k = a + s
                    valid = k < b
                    q = line * n_along + k
                    l_ = np.searchsorted(keys, q, "left")
                    r_ = np.searchsorted(keys, q, "right")
                    has = valid & (r_ > l_)
                    for m in np.nonzero(has)[0]:
                        rng = slice(l_[m], r_[m])
                        if np.any((t[m] >= lo[rng] - self.tol) & (t[m] <= hi[rng] + self.tol)):
                            hit[m] = True
                out[np.nonzero(sel)[0][hit]] = True
            return out
        return fn


def barrier_3d(faults, vol, tol_samples=2.0):
    """EdgeBarrier on the (nil, nxl) grid of a volume; times in sample index units."""
    eb = EdgeBarrier((vol.nil, vol.nxl), tol_samples)
    dt_ms = vol.dt * 1000.0
    t0_ms = vol.t0 * 1000.0
    for f in faults:
        tris = f.triangles()
        if not len(tris):
            continue
        ii, jj, tt = tri_index_coords(vol, tris)
        ts = (tt - t0_ms) / dt_ms
        for i in range(max(0, int(np.floor(ii.min()))), min(vol.nil - 1, int(np.ceil(ii.max()))) + 1):
            x0, t0, x1, t1 = plane_cut(ii, jj, ts, float(i))  # along xl on inline row i -> axis-1 edges
            eb.add(1, i, x0, t0, x1, t1)
        for j in range(max(0, int(np.floor(jj.min()))), min(vol.nxl - 1, int(np.ceil(jj.max()))) + 1):
            x0, t0, x1, t1 = plane_cut(jj, ii, ts, float(j))  # along il on crossline column j -> axis-0 edges
            eb.add(0, j, x0, t0, x1, t1)
    return eb.finish()


def barrier_line(faults, line, tol_samples=2.0):
    eb = EdgeBarrier((1, len(line.cdp)), tol_samples)
    dt_ms, t0_ms = line.dt * 1000.0, line.t0 * 1000.0
    for f in faults:
        seg = section_segments_line(f, line.x, line.y)
        if seg is None or not len(seg[0]):
            continue
        k0, t0, k1, t1 = seg
        eb.add(1, 0, k0, (t0 - t0_ms) / dt_ms, k1, (t1 - t0_ms) / dt_ms)
    return eb.finish()


# ====================================================================== set + persistence + undo
class FaultSet:
    def __init__(self):
        self.items = []
        self.undo_stack = []  # (seq, kind, ...)
        self.max_undo = 50
        self.dirty = False

    def names(self):
        return [f.name for f in self.items]

    def get(self, name):
        return next((f for f in self.items if f.name == name), None)

    def new(self, name=None, color=None):
        base = name or f"Fault {len(self.items) + 1}"
        name, k = base, 2
        while self.get(name):
            name, k = f"{base} ({k})", k + 1
        palette = [(220, 30, 30), (30, 30, 220), (0, 160, 0), (200, 0, 200), (230, 120, 0)] + DEFAULT_COLORS
        f = Fault(name, color or palette[len(self.items) % len(palette)])
        self.items.append(f)
        self.undo_stack.append((next_seq(), "new", f))
        self.dirty = True
        return f

    def delete(self, name):
        f = self.get(name)
        if f:
            pos = self.items.index(f)
            self.items.remove(f)
            self.undo_stack.append((next_seq(), "delete", f, pos))
            self.dirty = True

    def rename(self, old, new):
        f = self.get(old)
        if f and new and not self.get(new):
            f.name = new
            self.dirty = True
            return True
        return False

    def snapshot(self, f):
        self.undo_stack.append((next_seq(), "sticks", f, [s.copy() for s in f.sticks]))
        del self.undo_stack[:-self.max_undo]
        self.dirty = True

    def last_seq(self):
        return self.undo_stack[-1][0] if self.undo_stack else 0

    def undo(self):
        if not self.undo_stack:
            return None
        e = self.undo_stack.pop()
        kind, f = e[1], e[2]
        if kind == "new":
            if f in self.items:
                self.items.remove(f)
        elif kind == "delete":
            if not self.get(f.name):
                self.items.insert(min(e[3], len(self.items)), f)
        else:
            f.sticks = e[3]
            f.invalidate()
        self.dirty = True
        return f.name

    def to_json(self):
        return [{"name": f.name, "color": list(f.color), "visible": f.visible,
                 "sticks": [np.round(s, 3).tolist() for s in f.sticks]} for f in self.items]

    @classmethod
    def from_json(cls, entries):
        s = cls()
        for e in entries or []:
            f = Fault(e["name"], e.get("color"))
            f.visible = e.get("visible", True)
            f.sticks = [np.asarray(p, float).reshape(-1, 3) for p in e.get("sticks", [])]
            s.items.append(f)
        return s

    def visible_faults(self):
        return [f for f in self.items if f.visible]
