"""Automatic fault extraction (PLAN 32c), fault QC / filtering / merging (32d) and fault mapping on horizons (32f).

Fault points = voxels of the thinned fault likelihood (TFL) above a threshold, each with its fault normal.  They are
linked into "skins" (Hale 2013) by connected components over a 26-neighbour graph in which an edge is kept only when
the two normals agree within `max_angle` and the neighbour lies close to the point's plane - so crossing / conjugate
faults stay separate even where they touch.  Skins smaller than `min_size` are dropped.  `skin_to_fault` turns a skin
into the item-2 fault object (editable sticks on every Nth inline or crossline, whichever cuts the fault best), so
manual editing and the triangulated fault surface work as before.

QC (32d): `skin_table` (area, mean strike / dip, likelihood, throw), `stereonet` (lower-hemisphere poles + Kamb
density), `rose` (strike histogram), `filter_skins` (dip / azimuth / size / likelihood / throw ranges, or a brushed
stereonet region), `merge_candidates` (proximity + orientation + throw continuity), `merge_skins`, `split_skin`.
Maps (32f): `horizon_cutoffs` (footwall / hanging-wall cut-off lines of a skin on a horizon) -> `fault_polygons`
(heave polygons with throw, downthrown side) -> hub polygons layer with style fault (= gridding barriers, Live Map).
"""
import math

import numpy as np

from . import faultlik as FL


class Skin:
    def __init__(self, ident, pts, normals, fl):
        self.id = int(ident)
        self.pts = np.asarray(pts, np.float32)  # (n, 3) i, j, k (index space)
        self.normals = np.asarray(normals, np.float32)
        self.fl = np.asarray(fl, np.float32)
        self.name = f"Auto F{self.id}"
        self.throw = None  # (n,) samples, from throw_at_points
        m = self.normals.mean(axis=0)
        self.normal = m / (np.linalg.norm(m) + 1e-12)

    def __len__(self):
        return len(self.pts)

    def centroid(self):
        return self.pts.mean(axis=0)

    def strike_dip(self, ei, ej, dt_ms, velocity=2500.0):
        s, d = FL.geo_orientation(self.normal, ei, ej, dt_ms, velocity)
        return float(s), float(d)


# ====================================================================== 32c extraction
OFFSETS = [(di, dj, dk) for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)
           if (di, dj, dk) > (0, 0, 0)]  # 13 half-neighbourhood offsets


def fault_points(tfl, n, threshold=0.5, fl=None):
    """(idx (m, 3) int, normals (m, 3), values (m,)) of TFL voxels above threshold."""
    m = np.asarray(tfl) > threshold
    idx = np.argwhere(m)
    return idx, np.asarray(n)[m], np.asarray(fl if fl is not None else tfl)[m]


def metric_normals(normals, scale=None):
    """Index-space normals -> unit normals in metric space (scale = metres per (inline, crossline, sample)).  Needed
    to tell conjugate faults apart: in index space (traces >> samples) all steep faults look near-vertical."""
    if scale is None:
        return np.asarray(normals, np.float32)
    nm = np.asarray(normals, np.float64) / np.asarray(scale, float)
    return (nm / np.maximum(np.linalg.norm(nm, axis=-1, keepdims=True), 1e-12)).astype(np.float32)


def default_scale(ei, ej, dt_ms, velocity=2500.0):
    return (float(np.linalg.norm(ei)), float(np.linalg.norm(ej)), 0.5 * velocity * dt_ms / 1000.0)


def link_components(idx, normals, shape, max_angle=20.0, max_plane_dist=1.0, scale=None):
    """Connected components of the compatible-neighbour graph -> label per point (0..n_comp-1).  Orientation
    compatibility is judged in metric space (scale, see metric_normals)."""
    m = len(idx)
    if m == 0:
        return np.zeros(0, np.int64)
    nmet = metric_normals(normals, scale)
    lab = np.full(shape, -1, np.int64)
    lab[idx[:, 0], idx[:, 1], idx[:, 2]] = np.arange(m)
    cmin = math.cos(math.radians(max_angle))
    ea, eb = [], []
    for di, dj, dk in OFFSETS:
        q = idx + np.array([di, dj, dk])
        ok = (q[:, 0] >= 0) & (q[:, 0] < shape[0]) & (q[:, 1] >= 0) & (q[:, 1] < shape[1]) & \
             (q[:, 2] >= 0) & (q[:, 2] < shape[2])
        a = np.flatnonzero(ok)
        b = lab[q[a, 0], q[a, 1], q[a, 2]]
        hit = b >= 0
        a, b = a[hit], b[hit]
        if not len(a):
            continue
        dot = np.abs(np.sum(nmet[a] * nmet[b], axis=1))
        dv = np.array([di, dj, dk], np.float32)
        dist = np.abs(normals[a] @ dv)  # offset of the neighbour from a's plane (index units)
        keep = (dot >= cmin) & (dist <= max_plane_dist)
        ea.append(a[keep])
        eb.append(b[keep])
    if not ea:
        return np.arange(m)
    ea, eb = np.concatenate(ea), np.concatenate(eb)
    lbl = np.arange(m)
    for _ in range(1000):  # min-label propagation with pointer jumping
        lo = np.minimum(lbl[ea], lbl[eb])
        new = lbl.copy()
        np.minimum.at(new, ea, lo)
        np.minimum.at(new, eb, lo)
        new = new[new]
        if np.array_equal(new, lbl):
            break
        lbl = new
    _, comp = np.unique(lbl, return_inverse=True)
    return comp


def grow_components(idx, normals, val, shape, max_angle=20.0, max_dev=30.0, max_plane_dist=1.0, scale=None,
                    min_size=1):
    """Hale-style skin growing: seeds in order of decreasing likelihood; a skin grows to 26-neighbours that are
    compatible with the point they are reached from (angle <= max_angle, close to its plane) AND whose orientation
    stays within max_dev of the seed's - so skins cannot drift through the blended normals where faults cross.
    Vectorised breadth-first fronts.  Returns label per point (-1 = in no skin of >= min_size)."""
    m = len(idx)
    comp = np.full(m, -1, np.int64)
    if m == 0:
        return comp
    nmet = metric_normals(normals, scale)
    lab = np.full(shape, -1, np.int64)
    lab[idx[:, 0], idx[:, 1], idx[:, 2]] = np.arange(m)
    ca, cd = math.cos(math.radians(max_angle)), math.cos(math.radians(max_dev))
    offs = np.array([(di, dj, dk) for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)
                     if (di, dj, dk) != (0, 0, 0)])
    order = np.argsort(-np.asarray(val))
    nxt = 0
    shp = np.array(shape)
    for seed in order:
        if comp[seed] >= 0:
            continue
        ref = nmet[seed]
        cur = nxt
        comp[seed] = cur
        members = [np.array([seed])]
        front = np.array([seed])
        while len(front):
            q = (idx[front][:, None, :] + offs[None]).reshape(-1, 3)
            par = np.repeat(front, len(offs))
            ok = np.all((q >= 0) & (q < shp), axis=1)
            q, par = q[ok], par[ok]
            nb = lab[q[:, 0], q[:, 1], q[:, 2]]
            ok = nb >= 0
            nb, par, q = nb[ok], par[ok], q[ok]
            ok = comp[nb] < 0
            nb, par, q = nb[ok], par[ok], q[ok]
            if not len(nb):
                break
            dot_p = np.abs(np.sum(nmet[nb] * nmet[par], axis=1))
            dot_r = np.abs(nmet[nb] @ ref)
            dist = np.abs(np.sum(normals[par] * (q - idx[par]), axis=1))
            ok = (dot_p >= ca) & (dot_r >= cd) & (dist <= max_plane_dist)
            nb = np.unique(nb[ok])
            comp[nb] = cur
            members.append(nb)
            front = nb
        allm = np.concatenate(members)
        if len(allm) < min_size:
            comp[allm] = -2  # too small: not a seed again, not a skin
        else:
            nxt += 1
    comp[comp < 0] = -1
    return comp


def extract_skins(tfl, n, fl=None, threshold=0.5, min_size=200, max_angle=20.0, max_plane_dist=1.0, scale=None,
                  max_dev=30.0):
    """Skins from a TFL volume (array) and its normals.  Largest first.  scale = metres per (inline, crossline,
    sample) (default_scale) - strongly recommended (conjugate faults look alike in index space)."""
    idx, nn, val = fault_points(tfl, n, threshold, fl)
    comp = grow_components(idx, nn, val, np.asarray(tfl).shape, max_angle, max_dev, max_plane_dist, scale, min_size)
    return _skins_from(idx, nn, val, comp, min_size)


def _skins_from(idx, nn, val, comp, min_size):
    skins = []
    ok = comp >= 0
    if not ok.any():
        return skins
    counts = np.bincount(comp[ok])
    for c in np.argsort(-counts):
        if counts[c] < min_size:
            break
        m = comp == c
        nm = nn[m]
        ref = nm[np.argmax(val[m])]
        nm = np.where((nm @ ref)[:, None] < 0, -nm, nm)  # consistent sign within the skin
        skins.append(Skin(len(skins) + 1, idx[m], nm, val[m]))
    return skins


def strike_axis(skin):
    """'inline' when sticks should be drawn on inlines (the fault strikes more along the inline axis), else
    'crossline'."""
    nh = skin.normal[:2]
    u = np.array([-nh[1], nh[0]])  # strike direction in (i, j)
    return "inline" if abs(u[0]) >= abs(u[1]) else "crossline"


def skin_sticks(skin, every=5, bin_samples=8, min_points=4):
    """Index-space sticks: [(n, 3) (i, j, k)] on every `every`-th slice across the fault, each the median trace
    position per `bin_samples` time bin."""
    ax = 0 if strike_axis(skin) == "inline" else 1
    other = 1 - ax
    p = skin.pts
    c = np.round(p[:, ax]).astype(int)
    out = []
    lo, hi = int(c.min()), int(c.max())
    start = lo + ((every - lo % every) % every)
    for s in range(start, hi + 1, every):
        q = p[c == s]
        if len(q) < min_points:
            continue
        kb = np.floor(q[:, 2] / bin_samples).astype(int)
        pts = []
        for b in np.unique(kb):
            r = q[kb == b]
            pts.append((float(s) if ax == 0 else float(np.median(r[:, 0])),
                        float(np.median(r[:, 1])) if ax == 0 else float(s), float(np.median(r[:, 2]))))
        if len(pts) >= 2:
            a = np.array(pts)
            a[:, ax] = s
            out.append(a[np.argsort(a[:, 2])])
    return out


def skin_to_fault(skin, vol, faultset, every=5, name=None, color=None):
    """Store a skin as an item-2 fault (map x, y, t ms sticks) in a FaultSet.  Returns the Fault."""
    f = faultset.new(name or skin.name, color)
    dt_ms, t0_ms = vol.dt * 1000.0, vol.t0 * 1000.0
    for st in skin_sticks(skin, every):
        il = vol.inlines[0] + vol.il_step * st[:, 0]
        xl = vol.xlines[0] + vol.xl_step * st[:, 1]
        x, y = vol.xy(il, xl)
        f.add_stick(np.column_stack([x, y, t0_ms + dt_ms * st[:, 2]]))
    f.auto = {"skin": skin.id, "points": len(skin), "fl": float(np.mean(skin.fl))}
    return f


def extract_from_volumes(project, vol_name, threshold=0.5, min_size=200, max_angle=20.0, velocity=2500.0,
                         block=32, progress=None, cancel=None):
    """Skins from the stored '<vol> [TFL]', '[Fault strike]', '[Fault dip]' volumes (compute_fault_volumes)."""
    vol = project.volume(vol_name)
    tv, sv, dv = (project.volume(f"{vol_name} [{n}]") for n in ("TFL", "Fault strike", "Fault dip"))
    ei, ej = FL.index_frame(vol)
    dt_ms = vol.dt * 1000.0
    idx, nn, val = [], [], []
    for a in range(0, vol.nil, block):
        if cancel and cancel():
            raise RuntimeError("cancelled")
        b = min(vol.nil, a + block)
        t = np.asarray(tv.ilxl[a:b], np.float32) * tv.inv
        m = t > threshold
        if m.any():
            ii = np.argwhere(m)
            s = (np.asarray(sv.ilxl[a:b], np.float32) * sv.inv)[m]
            d = (np.asarray(dv.ilxl[a:b], np.float32) * dv.inv)[m]
            idx.append(ii + np.array([a, 0, 0]))
            nn.append(index_normals(s, d, ei, ej, dt_ms, velocity))
            val.append(t[m])
        if progress:
            progress(0.6 * b / vol.nil, f"fault points: inline {b}/{vol.nil}")
    if not idx:
        return []
    idx, nn, val = np.concatenate(idx), np.concatenate(nn), np.concatenate(val)
    comp = grow_components(idx, nn, val, (vol.nil, vol.nxl, vol.ns), max_angle, 30.0, 1.0,
                           default_scale(ei, ej, dt_ms, velocity), min_size)
    skins = _skins_from(idx, nn, val, comp, min_size)
    if progress:
        progress(1.0, f"{len(skins)} fault skins")
    return skins


def index_normals(strike, dip, ei, ej, dt_ms, velocity=2500.0):
    """Vectorised plane_normal_index: strike / dip arrays (deg) -> (m, 3) index-space unit normals (pointing up)."""
    dipaz = np.radians(np.asarray(strike, float) + 90.0)
    dr = np.radians(np.asarray(dip, float))
    dz = 0.5 * velocity * dt_ms / 1000.0
    nh = np.column_stack([np.sin(dr) * np.sin(dipaz), np.sin(dr) * np.cos(dipaz)])
    M = np.column_stack([ei, ej])
    ni = nh @ M  # (M^T nh)^T
    n = np.column_stack([ni, -np.cos(dr) * dz])
    return (n / np.linalg.norm(n, axis=1, keepdims=True)).astype(np.float32)


# ====================================================================== 32b / 32e throw
def k_of_tau(T, ii, jj, tau, chunk=20000):
    """Fractional sample where the (monotonic increasing) trace T[ii, jj, :] reaches tau - vectorised."""
    nk = T.shape[2]
    out = np.full(len(tau), np.nan)
    for a in range(0, len(tau), chunk):
        b = min(len(tau), a + chunk)
        tr = T[ii[a:b], jj[a:b]].astype(np.float64)
        t = tau[a:b, None]
        ge = tr >= t
        k1 = np.argmax(ge, axis=1)
        ok = ge.any(axis=1) & (k1 > 0)
        k0 = np.maximum(k1 - 1, 0)
        r = np.arange(b - a)
        v0, v1 = tr[r, k0], tr[r, k1]
        frac = np.where(v1 > v0, (tau[a:b] - v0) / np.where(v1 > v0, v1 - v0, 1.0), 0.0)
        out[a:b] = np.where(ok, k0 + frac, np.nan)
    del nk
    return out


def throw_at_points(skin, T, offset=3.0):
    """Vertical throw (samples) at each skin point from a geologic-time (RGT) volume T (monotonic increasing with
    time on every trace): the iso-T surface through the point is followed on the footwall side (traces at -offset and
    -2 offset along the normal's horizontal part), extrapolated linearly across the fault and compared with where the
    same iso-T surface is on the hanging-wall trace (+offset).  Positive = normal fault (hanging wall down).
    PaleoScan's RGT throw idea; the extrapolation removes the regional dip."""
    p = skin.pts.astype(np.float64)
    h = skin.normals[:, :2].astype(np.float64)
    h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
    ni, nj, nk = T.shape

    def tr_idx(sign, mult):
        return (np.clip(np.round(p[:, 0] + sign * mult * offset * h[:, 0]).astype(int), 0, ni - 1),
                np.clip(np.round(p[:, 1] + sign * mult * offset * h[:, 1]).astype(int), 0, nj - 1))
    f1i, f1j = tr_idx(-1, 1)
    f2i, f2j = tr_idx(-1, 2)
    hi_, hj_ = tr_idx(+1, 1)
    tau = FL.sample3(T, f1i.astype(float), f1j.astype(float), p[:, 2])
    k_f1 = p[:, 2]
    k_f2 = k_of_tau(T, f2i, f2j, tau)
    k_h = k_of_tau(T, hi_, hj_, tau)
    k_lin = k_f1 + 2.0 * (k_f1 - k_f2)
    thr = k_h - np.where(np.isfinite(k_lin), k_lin, k_f1)
    skin.throw = thr.astype(np.float32)
    return skin.throw


def rgt_throw_volume(T, fl, n, threshold=0.3, offset=3.0):
    """Throw cube (samples) from an RGT volume (PaleoScan-style 32b): at voxels with fl > threshold the iso-RGT
    displacement across the fault; elsewhere 0.  Also returns the throw-weighted fault probability
    fl * min(1, |throw| / median |throw|)."""
    m = fl > threshold
    idx = np.argwhere(m)
    sk = Skin(0, idx, n[m], fl[m])
    thr = throw_at_points(sk, T, offset)
    cube = np.zeros(fl.shape, np.float32)
    cube[m] = np.nan_to_num(thr)
    med = float(np.nanmedian(np.abs(thr))) if np.isfinite(thr).any() else 1.0
    w = np.zeros(fl.shape, np.float32)
    w[m] = fl[m] * np.clip(np.abs(np.nan_to_num(thr)) / max(med, 1e-6), 0, 1)
    return cube, w


# ====================================================================== 32d QC
def skin_table(skins, ei, ej, dt_ms, velocity=2500.0):
    """[{name, points, strike, dip, fl, throw_max, extent_il, extent_xl, extent_k}] - the fault table."""
    rows = []
    for s in skins:
        st, dp = s.strike_dip(ei, ej, dt_ms, velocity)
        rows.append({"name": s.name, "id": s.id, "points": len(s), "strike": round(st, 1), "dip": round(dp, 1),
                     "fl": round(float(np.mean(s.fl)), 3),
                     "throw_max": None if s.throw is None or not np.isfinite(s.throw).any()
                     else round(float(np.nanmax(np.abs(s.throw))), 2),
                     "extent_il": float(np.ptp(s.pts[:, 0])), "extent_xl": float(np.ptp(s.pts[:, 1])),
                     "extent_k": float(np.ptp(s.pts[:, 2]))})
    return rows


def poles(strike, dip):
    """Lower-hemisphere poles (trend, plunge deg) of planes (right-hand-rule strike)."""
    strike, dip = np.asarray(strike, float), np.asarray(dip, float)
    return (strike + 270.0) % 360.0, 90.0 - dip


def stereo_xy(trend, plunge):
    """Equal-area (Schmidt) lower-hemisphere projection, unit circle, north up."""
    t, p = np.radians(trend), np.radians(plunge)
    r = math.sqrt(2.0) * np.sin((np.pi / 2 - p) / 2.0)
    return r * np.sin(t), r * np.cos(t)


def stereonet(strikes, dips, weights=None, grid=41, sigma=3.0):
    """Pole points + a Kamb density grid on the equal-area net: {x, y, gx, gy, density (multiples of the uniform
    density, in sigma units of the Kamb count)}."""
    tr, pl = poles(strikes, dips)
    x, y = stereo_xy(tr, pl)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    n = max(1, len(x))
    # Kamb: counting cone with area fraction A = sigma^2 / (n + sigma^2) of the hemisphere
    A = sigma ** 2 / (n + sigma ** 2)
    cosc = 1.0 - A  # cone half-angle from the area fraction of a hemisphere: A = 1 - cos(theta)
    gx, gy = np.meshgrid(np.linspace(-1, 1, grid), np.linspace(-1, 1, grid))
    rr = np.hypot(gx, gy)
    inside = rr <= 1.0
    # back-project grid nodes to unit vectors (lower hemisphere)
    p = np.pi / 2 - 2 * np.arcsin(np.clip(rr / math.sqrt(2.0), 0, 1))
    t = np.arctan2(gx, gy)
    gvec = np.stack([np.cos(p) * np.sin(t), np.cos(p) * np.cos(t), -np.sin(p)], -1)
    tp, pp = np.radians(tr), np.radians(pl)
    v = np.stack([np.cos(pp) * np.sin(tp), np.cos(pp) * np.cos(tp), -np.sin(pp)], -1)
    dots = np.abs(gvec.reshape(-1, 3) @ v.T)  # axial data
    cnt = (dots >= cosc) @ w
    expected = A * w.sum()
    dens = (cnt - expected) / max(math.sqrt(expected * (1 - A)), 1e-9)
    dens = dens.reshape(gx.shape)
    dens[~inside] = np.nan
    return {"x": x, "y": y, "gx": gx, "gy": gy, "density": dens}


def rose(strikes, bins=36, weights=None, axial=True):
    """Strike rose diagram: counts per azimuth bin (axial: strike and strike + 180 folded to 0-180)."""
    s = np.asarray(strikes, float) % (180.0 if axial else 360.0)
    span = 180.0 if axial else 360.0
    edges = np.linspace(0, span, bins + 1)
    return edges, np.histogram(s, edges, weights=weights)[0]


def filter_skins(skins, ei, ej, dt_ms, dip=None, strike=None, size=None, fl=None, throw=None, poly=None,
                 velocity=2500.0):
    """Skins passing ranges (lo, hi) of dip, strike (wraps), points, mean fl, max |throw|, or whose pole falls in
    a polygon [(x, y)] brushed on the stereonet."""
    out = []
    for s in skins:
        st, dp = s.strike_dip(ei, ej, dt_ms, velocity)
        if dip and not (dip[0] <= dp <= dip[1]):
            continue
        if strike:
            lo, hi = strike[0] % 360, strike[1] % 360
            if not ((lo <= st <= hi) if lo <= hi else (st >= lo or st <= hi)):
                continue
        if size and not (size[0] <= len(s) <= (size[1] if len(size) > 1 and size[1] is not None else 1e18)):
            continue
        if fl and not (fl[0] <= float(np.mean(s.fl)) <= fl[1]):
            continue
        if throw:
            tm = float(np.nanmax(np.abs(s.throw))) if s.throw is not None and np.isfinite(s.throw).any() else 0.0
            if not (throw[0] <= tm <= throw[1]):
                continue
        if poly is not None:
            from .maphub import point_in_polygon
            tr, pl = poles(st, dp)
            x, y = stereo_xy(tr, pl)
            if not point_in_polygon(np.array([x]), np.array([y]), [p[0] for p in poly], [p[1] for p in poly])[0]:
                continue
        out.append(s)
    return out


def _min_dist(a, b, cap=4000):
    a = a[:: max(1, len(a) // cap)]
    b = b[:: max(1, len(b) // cap)]
    best = np.inf
    for k in range(0, len(a), 500):
        d = np.sqrt(((a[k:k + 500, None, :] - b[None, :, :]) ** 2).sum(-1)).min()
        best = min(best, float(d))
    return best


def merge_candidates(skins, max_gap=4.0, max_angle=15.0, max_throw_diff=None):
    """Fault Merging Assistant: pairs (a, b, gap, angle) of skins that are close (gap in index units), have the same
    orientation, lie in each other's plane, and (when throws are known) a similar throw where they meet."""
    out = []
    for x in range(len(skins)):
        for y in range(x + 1, len(skins)):
            A, B = skins[x], skins[y]
            ang = math.degrees(math.acos(min(1.0, abs(float(A.normal @ B.normal)))))
            if ang > max_angle:
                continue
            # coplanar: B's centroid near A's plane
            off = abs(float(A.normal @ (B.centroid() - A.centroid())))
            if off > max_gap:
                continue
            gap = _min_dist(A.pts, B.pts)
            if gap > max_gap:
                continue
            if max_throw_diff is not None and A.throw is not None and B.throw is not None:
                ta, tb = float(np.nanmedian(A.throw)), float(np.nanmedian(B.throw))
                if abs(ta - tb) > max_throw_diff:
                    continue
            out.append((A.id, B.id, round(gap, 2), round(ang, 1)))
    out.sort(key=lambda r: (r[2], r[3]))
    return out


def merge_skins(skins, a, b):
    """Merge skin b into skin a (by id); returns the new list."""
    A = next(s for s in skins if s.id == a)
    B = next(s for s in skins if s.id == b)
    nb = np.where((B.normals @ A.normal)[:, None] < 0, -B.normals, B.normals)
    M = Skin(A.id, np.vstack([A.pts, B.pts]), np.vstack([A.normals, nb]), np.concatenate([A.fl, B.fl]))
    M.name = A.name
    if A.throw is not None and B.throw is not None:
        M.throw = np.concatenate([A.throw, B.throw])
    return [M if s.id == a else s for s in skins if s.id != b]


def split_skin(skins, ident, axis=2, value=None):
    """Split a skin into two at index coordinate `value` along axis (0 il, 1 xl, 2 time; default = median)."""
    S = next(s for s in skins if s.id == ident)
    v = float(np.median(S.pts[:, axis])) if value is None else float(value)
    m = S.pts[:, axis] < v
    if m.all() or not m.any():
        return skins
    nid = max(s.id for s in skins) + 1
    A = Skin(S.id, S.pts[m], S.normals[m], S.fl[m])
    A.name = S.name
    B = Skin(nid, S.pts[~m], S.normals[~m], S.fl[~m])
    B.name = f"{S.name} (split)"
    out = []
    for s in skins:
        out += [A, B] if s.id == ident else [s]
    return out


def displacement_length_flag(length_m, max_disp_m, c_range=(0.001, 0.1)):
    """32e QC: displacement / length outside the usual 0.001-0.1 band (Kim & Sanderson 2005) is flagged."""
    if not length_m or max_disp_m is None:
        return None
    r = max_disp_m / length_m
    return "too large for its length" if r > c_range[1] else ("unusually small" if r < c_range[0] else "")


# ====================================================================== 32f fault mapping on horizons
def horizon_cutoffs(skin, surface, dt_ms, t0_ms=0.0, offset=2.0, tol=0.75):
    """Footwall and hanging-wall cut-off points of a skin on a horizon (surface = time grid (ni, nj) ms on the
    survey).  Footwall side = opposite the normal's horizontal part.  Returns dict fw (m, 2) and hw (m, 2) index
    (i, j) points + 'throw' (ms, hanging-wall minus footwall horizon time at the fault) per fw point."""
    p = skin.pts.astype(np.float64)
    h = skin.normals[:, :2].astype(np.float64)
    h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
    S = np.asarray(surface, float)
    k_s = (S - t0_ms) / dt_ms

    def hz(i, j):
        return FL.sample3(np.nan_to_num(k_s, nan=-1e9)[..., None].repeat(2, -1), i, j, np.zeros_like(i))
    kf = hz(p[:, 0] - offset * h[:, 0], p[:, 1] - offset * h[:, 1])
    kh = hz(p[:, 0] + offset * h[:, 0], p[:, 1] + offset * h[:, 1])
    fw = np.abs(p[:, 2] - kf) <= tol
    hw = np.abs(p[:, 2] - kh) <= tol
    return {"fw": p[fw][:, :2], "hw": p[hw][:, :2], "throw": (kh[fw] - kf[fw]) * dt_ms}


def _order_along(pts, direction):
    if len(pts) == 0:
        return pts
    s = pts @ direction
    o = np.argsort(s)
    pts = pts[o]
    # collapse to one point per strike bin (median), 1 trace bins
    sb = np.round(s[o]).astype(int)
    out = [np.median(pts[sb == b], axis=0) for b in np.unique(sb)]
    return np.array(out)


def fault_polygon(skin, surface, dt_ms, t0_ms=0.0):
    """Heave polygon of a skin on a horizon in index space: footwall cut-off line + reversed hanging-wall cut-off line.
    Returns dict {poly (m, 2) i, j, fw, hw, throw_ms (median), down (unit (i, j) towards the hanging wall)} or None."""
    c = horizon_cutoffs(skin, surface, dt_ms, t0_ms)
    if len(c["fw"]) < 2 or len(c["hw"]) < 2:
        return None
    nh = skin.normal[:2] / (np.linalg.norm(skin.normal[:2]) + 1e-12)
    u = np.array([-nh[1], nh[0]])
    fw = _order_along(c["fw"], u)
    hw = _order_along(c["hw"], u)
    poly = np.vstack([fw, hw[::-1]])
    return {"poly": poly, "fw": fw, "hw": hw, "throw_ms": float(np.nanmedian(c["throw"])) if len(c["throw"]) else 0.0,
            "down": nh}


def polygons_layer_items(skins, surface, vol, name_prefix=""):
    """Hub 'polygons' items (map x, y) for every skin cutting the horizon: [{name, x, y, down (compass), throw_ms}]."""
    items = []
    for s in skins:
        fp = fault_polygon(s, surface, vol.dt * 1000.0, vol.t0 * 1000.0)
        if fp is None:
            continue
        il = vol.inlines[0] + vol.il_step * fp["poly"][:, 0]
        xl = vol.xlines[0] + vol.xl_step * fp["poly"][:, 1]
        x, y = vol.xy(il, xl)
        ei, ej = FL.index_frame(vol)
        dvec = fp["down"][0] * ei + fp["down"][1] * ej
        az = math.degrees(math.atan2(dvec[0], dvec[1])) % 360.0
        comp = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int(((az + 22.5) % 360) // 45)]
        items.append({"name": f"{name_prefix}{s.name}", "x": [float(v) for v in np.ravel(x)],
                      "y": [float(v) for v in np.ravel(y)], "down": comp, "throw_ms": round(fp["throw_ms"], 2)})
    return items


def throw_from_rgt(skin, rgt, offset=3.0):
    """throw_at_points with a stored RGTVolume (decodes only the skin's bounding box + margin)."""
    m = int(2 * offset + 3)
    lo = np.maximum(np.floor(skin.pts.min(axis=0)).astype(int) - [m, m, 0], 0)
    hi = np.minimum(np.ceil(skin.pts.max(axis=0)).astype(int) + [m + 1, m + 1, 0], rgt.arr.shape)
    T = rgt.decode(rgt.arr[lo[0]:hi[0], lo[1]:hi[1], :])
    sub = Skin(skin.id, skin.pts - np.array([lo[0], lo[1], 0], np.float32), skin.normals, skin.fl)
    skin.throw = throw_at_points(sub, T, offset)
    return skin.throw


def throw_profile(skin, along="strike", bins=20):
    """32e throw-distance (along strike) or throw-depth (along time) profile: (centres, median throw, max throw)."""
    if skin.throw is None:
        raise ValueError("no throw yet (throw_at_points / throw_from_rgt / horizon cut-offs)")
    nh = skin.normal[:2] / (np.linalg.norm(skin.normal[:2]) + 1e-12)
    u = np.array([-nh[1], nh[0]])
    x = skin.pts[:, :2] @ u if along == "strike" else skin.pts[:, 2]
    ok = np.isfinite(skin.throw)
    if not ok.any():
        return np.zeros(0), np.zeros(0), np.zeros(0)
    edges = np.linspace(x[ok].min(), x[ok].max() + 1e-9, bins + 1)
    k = np.clip(np.searchsorted(edges, x[ok], "right") - 1, 0, bins - 1)
    med = np.full(bins, np.nan)
    mx = np.full(bins, np.nan)
    for b in range(bins):
        v = skin.throw[ok][k == b]
        if len(v):
            med[b], mx[b] = np.median(v), np.max(np.abs(v))
    return 0.5 * (edges[:-1] + edges[1:]), med, mx


def throw_map(skin, bins=(24, 16)):
    """32e throw distribution on the fault plane: grid (strike position x time) of median throw (feeds Allan / SGR)."""
    if skin.throw is None:
        raise ValueError("no throw yet")
    nh = skin.normal[:2] / (np.linalg.norm(skin.normal[:2]) + 1e-12)
    u = np.array([-nh[1], nh[0]])
    s = skin.pts[:, :2] @ u
    t = skin.pts[:, 2]
    ok = np.isfinite(skin.throw)
    H, se, te = np.histogram2d(s[ok], t[ok], bins=bins, weights=skin.throw[ok])
    C, _, _ = np.histogram2d(s[ok], t[ok], bins=(se, te))
    with np.errstate(invalid="ignore"):
        return se, te, np.where(C > 0, H / np.maximum(C, 1), np.nan)


def _xcorr_lag(a, b, max_lag):
    """Lag (samples, sub-sample) maximising the normalised correlation of window rows a, b (m, w): b shifted by
    +lag matches a.  a, b are longer than the window by 2 max_lag (b) - see throw_by_correlation."""
    m, w = a.shape
    best = np.full(m, -np.inf)
    lag = np.zeros(m)
    cs = []
    an = a - a.mean(axis=1, keepdims=True)
    an /= np.linalg.norm(an, axis=1, keepdims=True) + 1e-12
    for L in range(-max_lag, max_lag + 1):
        seg = b[:, max_lag + L:max_lag + L + w]
        sn = seg - seg.mean(axis=1, keepdims=True)
        sn /= np.linalg.norm(sn, axis=1, keepdims=True) + 1e-12
        c = np.sum(an * sn, axis=1)
        cs.append(c)
        up = c > best
        best[up], lag[up] = c[up], L
    C = np.array(cs)  # (2 max_lag + 1, m)
    k = (lag + max_lag).astype(int)
    r = np.arange(m)
    k0, k2 = np.clip(k - 1, 0, len(C) - 1), np.clip(k + 1, 0, len(C) - 1)
    y0, y1, y2 = C[k0, r], C[k, r], C[k2, r]
    den = y0 - 2 * y1 + y2
    frac = np.where((np.abs(den) > 1e-12) & (k > 0) & (k < len(C) - 1), 0.5 * (y0 - y2) / np.where(
        np.abs(den) > 1e-12, den, 1.0), 0.0)
    return lag + frac, best


def throw_by_correlation(skin, cube, offset=3.0, window=12, max_lag=16, min_corr=0.5):
    """32e: throw (samples) without an RGT - Hale-style slip by cross-correlation: the footwall window (trace at
    -offset) is correlated with the hanging-wall trace (+offset); the regional dip is removed with the lag between
    the two footwall traces (-offset vs -2 offset), extrapolated across.  NaN where the correlation is poor."""
    p = skin.pts.astype(np.float64)
    h = skin.normals[:, :2].astype(np.float64)
    h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
    ni, nj, nk = cube.shape

    def tr(sign, mult):
        i = np.clip(np.round(p[:, 0] + sign * mult * offset * h[:, 0]).astype(int), 0, ni - 1)
        j = np.clip(np.round(p[:, 1] + sign * mult * offset * h[:, 1]).astype(int), 0, nj - 1)
        return i, j
    half = window // 2
    k = np.round(p[:, 2]).astype(int)
    ok = (k - half - max_lag >= 0) & (k + half + max_lag < nk)
    out = np.full(len(p), np.nan)
    if not ok.any():
        skin.throw = out.astype(np.float32)
        return skin.throw
    kk = k[ok]
    ar = np.arange(-half, half)
    arl = np.arange(-half - max_lag, half + max_lag)
    (f1i, f1j), (f2i, f2j), (hi, hj) = tr(-1, 1), tr(-1, 2), tr(1, 1)
    sel = np.flatnonzero(ok)
    A = cube[f1i[sel][:, None], f1j[sel][:, None], kk[:, None] + ar[None]]
    B_h = cube[hi[sel][:, None], hj[sel][:, None], kk[:, None] + arl[None]]
    B_f = cube[f2i[sel][:, None], f2j[sel][:, None], kk[:, None] + arl[None]]
    lag_h, c_h = _xcorr_lag(A, B_h, max_lag)
    lag_f, c_f = _xcorr_lag(A, B_f, max_lag)
    # FW2 is one offset further away from the fault: its lag is the dip over one offset (opposite direction)
    thr = lag_h + 2.0 * lag_f
    good = (c_h >= min_corr) & (c_f >= min_corr)
    out[sel] = np.where(good, thr, np.nan)
    skin.throw = out.astype(np.float32)
    return skin.throw


# ====================================================================== 32g 2D lines
class Cut:
    """A fault cut on a 2D line: trace positions + times of a fault segment, map x / y, down-dip azimuth, throw."""

    def __init__(self, line, traces, times, x, y, dipaz, throw=None):
        self.line = line
        self.traces, self.times = np.asarray(traces, float), np.asarray(times, float)
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.dipaz = float(dipaz)  # map azimuth of the down-dip direction along the line (apparent)
        self.throw = throw
        self.group = None


def segments_2d(tfl, n2, threshold=0.5, min_len=12, max_angle=20.0, scale=(25.0, 5.0)):
    """Fault segments on a 2D line from its thinned likelihood: linked ridge points with compatible orientation.
    Returns [(trace (m,), sample (m,), normal (2,))] ordered by sample."""
    pts = np.argwhere(np.asarray(tfl) > threshold)
    if not len(pts):
        return []
    nn = np.asarray(n2)[pts[:, 0], pts[:, 1]]
    idx3 = np.column_stack([np.zeros(len(pts), int), pts])
    n3 = np.column_stack([np.zeros(len(nn)), nn]).astype(np.float32)
    comp = grow_components(idx3, n3, np.asarray(tfl)[pts[:, 0], pts[:, 1]], (1,) + np.asarray(tfl).shape, max_angle,
                           30.0, 1.0, (1.0, scale[0], scale[1]), min_len)
    out = []
    for c in np.unique(comp[comp >= 0]):
        q = pts[comp == c]
        o = np.argsort(q[:, 1])
        q = q[o]
        m = nn[comp == c].mean(axis=0)
        out.append((q[:, 0].astype(float), q[:, 1].astype(float), m / (np.linalg.norm(m) + 1e-12)))
    return out


def cuts_on_line(name, x, y, tfl, n2, dt_ms, t0_ms=0.0, image=None, **kw):
    """Cut objects for a 2D line (x, y per trace).  Throw by correlation across the cut when the image is given."""
    out = []
    x, y = np.asarray(x, float), np.asarray(y, float)
    for tr_, k, nrm in segments_2d(tfl, n2, **kw):
        ti = np.clip(np.round(tr_).astype(int), 0, len(x) - 1)
        # apparent down-dip direction along the line: sign of the normal's trace component
        step = np.sign(nrm[0]) or 1.0
        a = ti[len(ti) // 2]
        b = int(np.clip(a + step * 3, 0, len(x) - 1))
        dipaz = math.degrees(math.atan2(x[b] - x[a], y[b] - y[a])) % 360.0 if b != a else 0.0
        thr = None
        if image is not None:
            sk = Skin(0, np.column_stack([np.zeros(len(tr_)), tr_, k]),
                      np.tile(np.array([0.0, nrm[0], nrm[1]], np.float32), (len(tr_), 1)), np.ones(len(tr_)))
            th = throw_by_correlation(sk, np.asarray(image, np.float32)[None])
            thr = float(np.nanmedian(th)) if np.isfinite(th).any() else None
        out.append(Cut(name, tr_, t0_ms + dt_ms * k, x[ti], y[ti], dipaz, thr))
    return out


def _cut_fit(c):
    """Linear fit of the cut's map position against time: (x0, y0, dxdt, dydt) about t = 0."""
    t = c.times
    if len(t) < 2 or np.ptp(t) <= 0:
        return float(np.mean(c.x)), float(np.mean(c.y)), 0.0, 0.0
    A = np.column_stack([np.ones_like(t), t])
    cx = np.linalg.lstsq(A, c.x, rcond=None)[0]
    cy = np.linalg.lstsq(A, c.y, rcond=None)[0]
    return float(cx[0]), float(cy[0]), float(cx[1]), float(cy[1])


def _pos(fit, t):
    return np.array([fit[0] + fit[2] * t, fit[1] + fit[3] * t])


def cut_link_score(A, B, max_dist=3000.0, strike_tol=15.0, rate_tol=0.5, same_line_tol=40.0):
    """None when cuts A and B cannot be the same planar fault, else a score (smaller = better).
    Same line: B lies on A's extrapolated trace (within same_line_tol m) and dips the same way.
    Different lines: at two times the vectors A(t) -> B(t) are parallel (= the fault strike), both cuts dip to the same
    side of that strike line, and they move perpendicular to strike at the same rate (= same dip)."""
    if abs((A.dipaz - B.dipaz + 180) % 360 - 180) > 90 and A.line == B.line:
        return None
    if A.throw is not None and B.throw is not None and min(abs(A.throw), abs(B.throw)) > 1.0 and \
            np.sign(A.throw) != np.sign(B.throw):
        return None
    fa, fb = _cut_fit(A), _cut_fit(B)
    if A.line == B.line:
        if abs((A.dipaz - B.dipaz + 180) % 360 - 180) > 30:
            return None
        pa = _pos(fa, B.times)
        da = float(np.median(np.hypot(pa[0] - B.x, pa[1] - B.y)))
        pb = _pos(fb, A.times)
        db = float(np.median(np.hypot(pb[0] - A.x, pb[1] - A.y)))
        d = min(da, db)
        return d if d <= same_line_tol else None
    tm = 0.5 * (np.median(A.times) + np.median(B.times))
    t1, t2 = tm - 150.0, tm + 150.0
    v1 = _pos(fb, t1) - _pos(fa, t1)
    v2 = _pos(fb, t2) - _pos(fa, t2)
    L1, L2 = np.hypot(*v1), np.hypot(*v2)
    if max(L1, L2) > max_dist or min(L1, L2) < 1e-6:
        return None
    ang = math.degrees(math.acos(min(1.0, abs(float(v1 @ v2) / (L1 * L2)))))
    if ang > strike_tol:
        return None
    s_ = (v1 + v2) / np.hypot(*(v1 + v2))
    n_ = np.array([s_[1], -s_[0]])  # perpendicular to strike
    da_ = np.array([math.sin(math.radians(A.dipaz)), math.cos(math.radians(A.dipaz))])
    db_ = np.array([math.sin(math.radians(B.dipaz)), math.cos(math.radians(B.dipaz))])
    sa, sb = float(n_ @ da_), float(n_ @ db_)
    if abs(sa) < 0.1 or abs(sb) < 0.1 or np.sign(sa) != np.sign(sb):
        return None
    ra = (fa[2] * n_[0] + fa[3] * n_[1])  # m per ms perpendicular to strike
    rb = (fb[2] * n_[0] + fb[3] * n_[1])
    if np.sign(ra) != np.sign(rb) or min(abs(ra), abs(rb)) < (1 - rate_tol) * max(abs(ra), abs(rb)):
        return None
    return ang + 0.002 * 0.5 * (L1 + L2)


def link_cuts(cuts, max_dist=3000.0, strike_tol=15.0, rate_tol=0.5, same_line_tol=40.0):
    """Fault-cut correlation assistant (Chris's 2D / 3D merge workflow): cuts on the same or on different lines are
    linked when they fit one planar fault (cut_link_score); each cut keeps its best partner per other line (no
    chaining through doubtful pairs).  Union-find -> cut.group; returns [member index lists], largest first."""
    n = len(cuts)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    best = {}
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            sc = cut_link_score(cuts[a], cuts[b], max_dist, strike_tol, rate_tol, same_line_tol)
            if sc is None:
                continue
            key = (a, cuts[b].line)
            if key not in best or sc < best[key][0]:
                best[key] = (sc, b)
    for (a, _line), (sc, b) in best.items():
        if best.get((b, cuts[a].line), (None, None))[1] == a or cuts[a].line == cuts[b].line:  # mutual best
            parent[find(a)] = find(b)
    groups = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    out = []
    for g, (root, members) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        for m in members:
            cuts[m].group = g
        out.append(members)
    return out


def cuts_to_faults(cuts, groups, faultset, prefix="Auto 2D F", min_lines=1, every_ms=None):
    """Linked cuts -> item-2 faults: each cut becomes a stick (map x, y, t ms) of its group's fault."""
    names = []
    for g, members in enumerate(groups):
        if len({cuts[m].line for m in members}) < min_lines:
            continue
        f = faultset.new(f"{prefix}{g + 1}")
        for m in members:
            c = cuts[m]
            o = np.argsort(c.times)
            k = max(1, len(o) // 12)
            sel = np.unique(np.r_[o[::k], o[-1]])
            sel = sel[np.argsort(c.times[sel])]
            f.add_stick(np.column_stack([c.x[sel], c.y[sel], c.times[sel]]))
        f.auto = {"lines": sorted({cuts[m].line for m in members}), "cuts": len(members)}
        names.append(f.name)
    return names


def project_cuts(project, lines=None, threshold=0.5, min_len=15, progress=None, cancel=None):
    """Fault likelihood + cuts on the project's 2D lines (with coordinates).  Returns [Cut]."""
    out = []
    names = lines or [l["name"] for l in project.lines]
    for k, name in enumerate(names):
        if cancel and cancel():
            raise RuntimeError("cancelled")
        line = project.line(name)
        if not line.has_coords:
            continue
        img = np.asarray(line.data(), np.float32)
        dt_ms = float(line.times[1] - line.times[0]) if len(line.times) > 1 else 4.0
        fl2, tfl2, n2 = FL.fault_likelihood_2d(img, dt_ms)
        out += cuts_on_line(name, line.x, line.y, tfl2, n2, dt_ms, float(line.times[0]), image=img,
                            threshold=threshold, min_len=min_len)
        if progress:
            progress((k + 1) / len(names), f"2D fault cuts: {name}")
    return out


# ====================================================================== 32h optional ML
def ml_backend():
    """Name of an importable optional CNN backend ('onnxruntime' / 'torch') or None.  The numpy TFL stays the
    default and the tested path; a FaultSeg-style model file is needed to use a backend."""
    for name in ("onnxruntime", "torch"):
        try:
            __import__(name)
            return name
        except Exception:
            continue
    return None


def cnn_fault_probability(cube, model_path=None):
    """Optional CNN fault probability.  Raises a clear message when no backend / model is available."""
    be = ml_backend()
    if be is None:
        raise RuntimeError("CNN fault probability needs onnxruntime or torch - run 'Install 3D extras (optional).bat'; "
                           "the numpy fault likelihood (Interpret > Automated fault mapping) works without it")
    if not model_path:
        raise RuntimeError(f"{be} is installed but no fault model file was given (e.g. a FaultSeg ONNX export)")
    if be == "onnxruntime":
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path)
        x = np.asarray(cube, np.float32)
        x = (x - x.mean()) / (x.std() + 1e-9)
        out = sess.run(None, {sess.get_inputs()[0].name: x[None, None]})[0]
        return np.asarray(out).reshape(x.shape)
    import torch
    m = torch.jit.load(model_path)
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(cube, np.float32))[None, None]
        return m(x).numpy().reshape(np.asarray(cube).shape)
