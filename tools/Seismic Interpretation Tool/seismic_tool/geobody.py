"""Geobodies and volume rendering helpers (PLAN 34c / 34d) - numpy only.

`label_components(mask)`: 6-connected component labelling (vectorised union-find by min-label propagation with
pointer jumping).  `label_volume(vol, lo, hi, ...)`: out-of-core version - inline blocks labelled one at a time,
labels merged across block faces, written to an int32 memmap; body statistics accumulated per block.
`surface_nets(mask)`: closed triangle surface of a body (naive surface nets: one vertex per boundary cell at the mean of
its edge crossings, one quad per crossing edge) - watertight for closed bodies, smoother than voxel faces, no
256-case marching-cubes table.  `footprint(...)`: map outline (convex hull) of a body for the Live Map.
`render_slabs(...)`: opacity-curve volume rendering as stacked semi-transparent slices (any GL, no 3D textures) with a
voxel budget (`fits_budget`).
"""
import math

import numpy as np


# ====================================================================== labelling
def _propagate(lbl, a, b):
    for _ in range(10000):
        lo = np.minimum(lbl[a], lbl[b])
        new = lbl.copy()
        np.minimum.at(new, a, lo)
        np.minimum.at(new, b, lo)
        new = new[new]
        if np.array_equal(new, lbl):
            return new
        lbl = new
    return lbl


def label_components(mask):
    """Labels (same shape, int64; 0 = background, 1..n by decreasing size) of 6-connected True voxels."""
    m = np.asarray(mask, bool)
    idx = np.flatnonzero(m)
    out = np.zeros(m.shape, np.int64)
    if not len(idx):
        return out, 0
    pos = np.full(m.size, -1, np.int64)
    pos[idx] = np.arange(len(idx))
    ea, eb = [], []
    strides = np.cumprod((1,) + m.shape[::-1][:-1])[::-1]
    coords = np.unravel_index(idx, m.shape)
    for ax in range(m.ndim):
        ok = coords[ax] < m.shape[ax] - 1
        nb = idx[ok] + strides[ax]
        hit = m.ravel()[nb]
        ea.append(pos[idx[ok][hit]])
        eb.append(pos[nb[hit]])
    lbl = np.arange(len(idx))
    if ea:
        lbl = _propagate(lbl, np.concatenate(ea), np.concatenate(eb))
    _, comp, counts = np.unique(lbl, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    out.ravel()[idx] = rank[comp] + 1
    return out, len(counts)


def label_volume(vol, lo=None, hi=None, block=32, out_path=None, min_cells=1, progress=None, cancel=None,
                 velocity=2500.0):
    """Threshold a Volume3D (lo <= value <= hi, None = open) and label connected bodies block by block.
    Returns (labels memmap or array (nil, nxl, ns) int32 with 1..n by decreasing size, bodies [dict])."""
    shape = (vol.nil, vol.nxl, vol.ns)
    if out_path:
        lab = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.int32, shape=shape)
    else:
        lab = np.zeros(shape, np.int32)
    parent = [0]  # global union-find over provisional ids
    nxt = 0
    prev_face = None
    for a in range(0, vol.nil, block):
        if cancel and cancel():
            raise RuntimeError("cancelled")
        b = min(vol.nil, a + block)
        v = np.asarray(vol.ilxl[a:b], np.float32) * vol.inv
        m = np.ones(v.shape, bool)
        if lo is not None:
            m &= v >= lo
        if hi is not None:
            m &= v <= hi
        loc, n = label_components(m)
        g = np.where(loc > 0, loc + nxt, 0).astype(np.int64)
        parent.extend(range(nxt + 1, nxt + n + 1))
        if prev_face is not None:  # merge across the block face (inline a-1 vs a)
            f0, f1 = prev_face, g[0]
            both = (f0 > 0) & (f1 > 0)
            for p_, q_ in set(zip(f0[both].tolist(), f1[both].tolist())):
                _union(parent, p_, q_)
        lab[a:b] = g.astype(np.int32)
        prev_face = g[-1].copy()
        nxt += n
        if progress:
            progress(0.7 * b / vol.nil, f"geobodies: inline {b}/{vol.nil}")
    root = np.array([_find(parent, k) for k in range(nxt + 1)], np.int64)
    # second pass: final ids by size + statistics
    counts = np.zeros(nxt + 1, np.int64)
    for a in range(0, vol.nil, block):
        b = min(vol.nil, a + block)
        r = root[np.asarray(lab[a:b], np.int64)]
        counts += np.bincount(r.ravel(), minlength=nxt + 1)
    counts[0] = 0
    keep = np.flatnonzero(counts >= max(1, min_cells))
    keep = keep[np.argsort(-counts[keep])]
    final = np.zeros(nxt + 1, np.int64)
    final[keep] = np.arange(1, len(keep) + 1)
    lut = final[root]
    bbox = {}
    for a in range(0, vol.nil, block):
        b = min(vol.nil, a + block)
        r = lut[np.asarray(lab[a:b], np.int64)]
        lab[a:b] = r.astype(np.int32)
        for body in np.unique(r[r > 0]):
            I, J, K = np.nonzero(r == body)
            lo_, hi_ = np.array([I.min() + a, J.min(), K.min()]), np.array([I.max() + a, J.max(), K.max()])
            if body in bbox:
                bbox[body] = (np.minimum(bbox[body][0], lo_), np.maximum(bbox[body][1], hi_))
            else:
                bbox[body] = (lo_, hi_)
        if progress:
            progress(0.7 + 0.3 * b / vol.nil, f"geobodies: relabel {b}/{vol.nil}")
    if out_path:
        lab.flush()
    cell = cell_volume(vol, velocity)
    bodies = [{"id": int(k + 1), "cells": int(counts[keep[k]]), "volume_m3": float(counts[keep[k]] * cell),
               "bbox": [bbox[k + 1][0].tolist(), bbox[k + 1][1].tolist()]} for k in range(len(keep))]
    return lab, bodies


def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _union(parent, a, b):
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)


def cell_volume(vol, velocity=2500.0):
    """m3 per voxel: bin area (from the affine) x vertical sample (v / 2 x dt)."""
    from .faultlik import index_frame
    ei, ej = index_frame(vol)
    area = abs(float(ei[0] * ej[1] - ei[1] * ej[0]))
    return area * 0.5 * velocity * float(vol.dt)


# ====================================================================== surfaces
def surface_nets(mask, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """Closed triangle surface around True voxels (padded so bodies touching the edge are closed too).
    Returns (vertices (n, 3) in index units * spacing + origin, triangles (m, 3))."""
    m = np.pad(np.asarray(mask, bool), 1)
    f = m.astype(np.float32)
    nx, ny, nz = m.shape
    # cells = cubes between 8 samples; a cell is on the surface when its corners differ
    c = [f[i:nx - 1 + i, j:ny - 1 + j, k:nz - 1 + k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    s = sum(c)
    surf = (s > 0) & (s < 8)
    cid = np.full(surf.shape, -1, np.int64)
    cells = np.argwhere(surf)
    cid[tuple(cells.T)] = np.arange(len(cells))
    # vertex = cell centre (+0.5) nudged towards the inside corners -> smooth-ish, inside the cell
    corners = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    cval = np.stack([cc[tuple(cells.T)] for cc in c], 1)  # (n, 8)
    w = np.where(cval > 0, 1.0, 0.0)
    inside_mean = (w @ corners) / np.maximum(w.sum(1, keepdims=True), 1)
    verts = cells + 0.5 + 0.35 * (inside_mean - 0.5)
    tris = []
    # an edge between voxels p and p + e (sign change) is surrounded by 4 cells -> one quad
    for ax in range(3):
        e = [0, 0, 0]
        e[ax] = 1
        a = m[:nx - e[0], :ny - e[1], :nz - e[2]]
        b = m[e[0]:, e[1]:, e[2]:]
        cross = a != b
        P = np.argwhere(cross)  # voxel p (padded coords); edge p -> p + e
        o1, o2 = [d for d in range(3) if d != ax]
        # cells sharing the edge: offsets 0 / -1 in the two other axes
        quad = []
        for d1, d2 in ((0, 0), (-1, 0), (-1, -1), (0, -1)):
            q = P.copy()
            q[:, o1] += d1
            q[:, o2] += d2
            ok = (q >= 0).all(1) & (q[:, 0] < cid.shape[0]) & (q[:, 1] < cid.shape[1]) & (q[:, 2] < cid.shape[2])
            ids = np.full(len(q), -1, np.int64)
            ids[ok] = cid[q[ok, 0], q[ok, 1], q[ok, 2]]
            quad.append(ids)
        Q = np.stack(quad, 1)
        Q = Q[(Q >= 0).all(1)]
        flip = a[cross][: len(Q)] if len(Q) == cross.sum() else None
        if flip is not None:  # orientation: normals point out of the body
            Q = np.where(flip[:, None] ^ (ax == 1), Q, Q[:, ::-1])
        tris.append(np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]]))
    T = np.concatenate(tris) if tris else np.zeros((0, 3), int)
    V = (verts - 1.0) * np.asarray(spacing, float) + np.asarray(origin, float)  # undo the padding
    return V, T


def edge_use_counts(tris):
    """How many triangles use each undirected edge (closed surface: every edge exactly 2)."""
    e = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return counts


def body_mesh_world(labels, body, vol, zscale=1.25, ve=1.0, color=(0.9, 0.5, 0.1, 0.9), bbox=None, step=1):
    """scene3d Mesh of one labelled body (sub-box only), in world coordinates."""
    from . import scene3d as S3
    if bbox is None:
        I, J, K = np.nonzero(np.asarray(labels) == body)
        lo = np.array([I.min(), J.min(), K.min()])
        hi = np.array([I.max(), J.max(), K.max()])
    else:
        lo, hi = np.asarray(bbox[0]), np.asarray(bbox[1])
    sub = np.asarray(labels[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]) == body
    sub = sub[::step, ::step, ::step]
    V, T = surface_nets(sub, (step, step, step), lo)
    il = vol.inlines[0] + vol.il_step * V[:, 0]
    xl = vol.xlines[0] + vol.xl_step * V[:, 1]
    x, y = vol.xy(il, xl)
    t = vol.times[0] + vol.dt * 1000.0 * V[:, 2]
    W = np.column_stack([np.ravel(x), np.ravel(y), S3.world_z(t, zscale, ve)])
    pos = W[T.ravel()]
    return S3.Mesh("triangles", pos, np.tile(np.asarray(color, np.float32), (len(pos), 1)), f"Body {body}")


def convex_hull(x, y):
    """Monotone chain convex hull -> (hx, hy) closed polygon (counter-clockwise)."""
    p = np.unique(np.column_stack([x, y]), axis=0)
    if len(p) < 3:
        return p[:, 0], p[:, 1]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(tuple(q))
    for q in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(tuple(q))
    h = np.array(lower[:-1] + upper[:-1])
    return h[:, 0], h[:, 1]


def footprint(labels, body, vol, bbox=None):
    """Map outline of a body (convex hull of its traces) -> {name, x, y} hub polygon item."""
    lab = np.asarray(labels if bbox is None else labels[bbox[0][0]:bbox[1][0] + 1])
    a0 = 0 if bbox is None else bbox[0][0]
    I, J = np.nonzero((lab == body).any(axis=2))
    il = vol.inlines[0] + vol.il_step * (I + a0)
    xl = vol.xlines[0] + vol.xl_step * J
    x, y = vol.xy(il, xl)
    hx, hy = convex_hull(np.ravel(x).astype(float), np.ravel(y).astype(float))
    return {"name": f"Body {body}", "x": [float(v) for v in hx], "y": [float(v) for v in hy]}


# ====================================================================== volume rendering (34c)
def fits_budget(shape, max_voxels=64e6):
    return float(np.prod(shape)) <= max_voxels


def opacity_curve(values, points):
    """Piecewise-linear opacity from [(value, alpha)] points (sorted by value)."""
    pts = sorted(points)
    return np.interp(values, [p[0] for p in pts], [p[1] for p in pts])


def render_slabs(sub, ilxl_lo, vol, lut, clip, curve, n_slabs=48, axis=2, zscale=1.25, ve=1.0, step=1, index_step=1):
    """Volume rendering of a sub-volume (i, j, k) as n_slabs semi-transparent planes along `axis`: colour from the LUT,
    alpha from the opacity curve.  Returns [scene3d.Mesh] (draw them sorted back to front, blending on)."""
    from . import scene3d as S3
    sub = np.asarray(sub, np.float32)[::step, ::step, ::step]
    step = step * int(index_step)  # sub may already be decimated (read with a stride)
    ns = sub.shape[axis]
    ks = np.unique(np.linspace(0, ns - 1, min(n_slabs, ns)).round().astype(int))
    out = []
    i0, j0, k0 = ilxl_lo
    for k in ks:
        sl = np.take(sub, k, axis=axis)
        rgba = S3.lut_rgba(sl.ravel(), lut, clip[0], clip[1])
        rgba[:, 3] *= opacity_curve(sl.ravel(), curve)
        ny, nx = sl.shape
        grids = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        idx = [None, None, None]
        oth = [d for d in range(3) if d != axis]
        idx[axis] = np.full((ny, nx), k * step)
        idx[oth[0]], idx[oth[1]] = grids[0] * step, grids[1] * step
        il = vol.inlines[0] + vol.il_step * (idx[0] + i0)
        xl = vol.xlines[0] + vol.xl_step * (idx[1] + j0)
        x, y = vol.xy(il, xl)
        t = vol.times[0] + vol.dt * 1000.0 * (idx[2] + k0)
        pos = np.column_stack([np.ravel(x), np.ravel(y), S3.world_z(np.ravel(t), zscale, ve)])
        tri = S3.grid_triangles(ny, nx, (rgba[:, 3] > 0.01))
        if len(tri):
            out.append(S3.Mesh("triangles", pos[tri.ravel()], rgba[tri.ravel()], f"slab {k}"))
    return out


def integrated_graphics(renderer_string):
    """True for GPU names that usually share system memory (warn before large volume rendering)."""
    s = (renderer_string or "").lower()
    return any(k in s for k in ("intel", "uhd", "iris", "microsoft basic", "llvmpipe", "swiftshader"))
