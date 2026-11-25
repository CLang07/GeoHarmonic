"""Fault likelihood, fault orientation, thinned fault likelihood and ant-track-style enhancement (PLAN 32a).

numpy only, block-wise over inlines (bounded memory, Cancel + progress), in the spirit of Hale (2013, "Methods to
compute fault images, extract fault surfaces, and estimate fault throws from 3D seismic images"):

1. semblance s of the image along reflectors (3 x 3 traces, steered by the item-3 dip field, a few samples in time);
2. fault orientation: the fault-normal vector n of the thin sheet of low semblance from the structure tensor of
   (1 - s) - its largest eigenvector - refined once from the tensor of the in-plane smoothed likelihood.  This replaces Hale's brute-force scan over strike and dip: the scan picks
   the orientation whose in-plane smoothing keeps the semblance lowest, which is the orientation of the sheet;
3. s smoothed WITHIN that plane (Gaussian along strike and along dip, trilinear sampling), then
   fault likelihood fl = 1 - s_plane ** power (power 8 as Hale: semblance 0.9 -> 0.57, 0.97 -> 0.22);
4. thinned fault likelihood (TFL): fl kept only at maxima across the fault (non-maximum suppression along the
   horizontal part of n), elsewhere 0 - one-voxel-thick fault traces;
5. `enhance` = ant-track-like: repeated in-plane smoothing + sharpening + thinning of ANY edge attribute
   (coherence / variance / TFL).

Index space: axes (inline index i, crossline index j, sample k); n is a unit vector in index units.  Geographic
strike / dip (`geo_orientation`) need the survey geometry (trace spacing + azimuth from the affine) and a velocity
for the time axis (v / 2 metres per second of TWT; default 2500 m/s).

Volumes written (`compute_fault_volumes`): '<vol> [Fault likelihood]', '[TFL]', '[Fault strike]', '[Fault dip]'
(strike azimuth deg 0-360, right-hand rule; dip deg).
"""
import math

import numpy as np

from .rgt.dip import dip_3d_block, gsmooth, gradients


# ====================================================================== helpers
def sample3(a, pi, pj, pk):
    """Trilinear samples of a (ni, nj, nk) at fractional index arrays (clamped to the volume)."""
    ni, nj, nk = a.shape
    pi = np.clip(pi, 0, ni - 1.000001)
    pj = np.clip(pj, 0, nj - 1.000001)
    pk = np.clip(pk, 0, nk - 1.000001)
    i0, j0, k0 = np.floor(pi).astype(np.int64), np.floor(pj).astype(np.int64), np.floor(pk).astype(np.int64)
    wi, wj, wk = pi - i0, pj - j0, pk - k0
    i1, j1, k1 = np.minimum(i0 + 1, ni - 1), np.minimum(j0 + 1, nj - 1), np.minimum(k0 + 1, nk - 1)
    f = a.reshape(-1)
    s_i, s_j = nj * nk, nk

    def g(ii, jj, kk):
        return f[ii * s_i + jj * s_j + kk]
    return ((g(i0, j0, k0) * (1 - wk) + g(i0, j0, k1) * wk) * (1 - wj) +
            (g(i0, j1, k0) * (1 - wk) + g(i0, j1, k1) * wk) * wj) * (1 - wi) + \
        ((g(i1, j0, k0) * (1 - wk) + g(i1, j0, k1) * wk) * (1 - wj) +
         (g(i1, j1, k0) * (1 - wk) + g(i1, j1, k1) * wk) * wj) * wi


def semblance3(block, dt_ms, window_samples=5, dip_i=None, dip_j=None):
    """Dip-steered 3 x 3 semblance for EVERY trace of a (ni, nj, ns) block (edges padded).  dips in samples/trace."""
    from .attributes import semblance
    b = np.pad(np.asarray(block, np.float32), ((1, 1), (1, 1), (0, 0)), mode="edge")
    di = dj = None
    if dip_i is not None:
        di = np.pad(dip_i, ((1, 1), (1, 1), (0, 0)), mode="edge")
        dj = np.pad(dip_j, ((1, 1), (1, 1), (0, 0)), mode="edge")
    return semblance(b, dt_ms, window_ms=window_samples * dt_ms, dip_i=di, dip_j=dj)


def plane_normals(f, sigma=(2.0, 2.0, 4.0), sigma_grad=1.0):
    """Unit normal (ni, nj, ns, 3) of sheet-like features of f from the gradient structure tensor + 'sheetness'
    (lambda1 - lambda2) / lambda1 in [0, 1].  The sign is fixed so that n_k (time component) <= 0 -> the normal
    points up; the horizontal part then points towards the down-dip (hanging-wall) side."""
    g = gradients(np.asarray(f, np.float32), sigma_grad)
    T = np.empty(f.shape + (3, 3), np.float32)
    for a in range(3):
        for b in range(a, 3):
            T[..., a, b] = gsmooth(g[a] * g[b], sigma)
            if a != b:
                T[..., b, a] = T[..., a, b]
    w, v = np.linalg.eigh(T.reshape(-1, 3, 3).astype(np.float64))
    n = v[:, :, 2]  # largest eigenvalue
    n = np.where(n[:, 2:3] > 0, -n, n)
    sheet = (w[:, 2] - w[:, 1]) / np.maximum(w[:, 2], 1e-20)
    return n.reshape(f.shape + (3,)).astype(np.float32), np.clip(sheet, 0, 1).reshape(f.shape).astype(np.float32)


def plane_axes(n):
    """In-plane unit axes: u = strike (horizontal), w = dip direction (in the plane, down-dip)."""
    ez = np.zeros_like(n)
    ez[..., 2] = 1.0
    u = np.cross(n, ez)
    lu = np.linalg.norm(u, axis=-1, keepdims=True)
    u = np.where(lu > 1e-6, u / np.maximum(lu, 1e-12), np.array([0.0, 1.0, 0.0], np.float32))
    w = np.cross(n, u)
    w = np.where(w[..., 2:3] < 0, -w, w)  # down-dip = increasing time
    return u.astype(np.float32), w.astype(np.float32)


def plane_smooth(f, n, half_u=4, half_w=6, sigma_u=2.0, sigma_w=3.0):
    """Gaussian smoothing of f within the local plane of normal n (separable: along strike, then along dip)."""
    u, w = plane_axes(n)
    I, J, K = np.meshgrid(np.arange(f.shape[0], dtype=np.float32), np.arange(f.shape[1], dtype=np.float32),
                          np.arange(f.shape[2], dtype=np.float32), indexing="ij")
    out = np.asarray(f, np.float32)
    for ax, half, sig in ((u, half_u, sigma_u), (w, half_w, sigma_w)):
        acc = np.zeros_like(out)
        wsum = 0.0
        for k in range(-half, half + 1):
            wt = math.exp(-0.5 * (k / sig) ** 2)
            acc += np.float32(wt) * sample3(out, I + k * ax[..., 0], J + k * ax[..., 1], K + k * ax[..., 2])
            wsum += wt
        out = acc / np.float32(wsum)
    return out


def thin(fl, n, step=1.0):
    """Non-maximum suppression across the fault (along the horizontal part of n): fl where it is a ridge, else 0."""
    h = n[..., :2]
    lh = np.linalg.norm(h, axis=-1, keepdims=True)
    h = np.where(lh > 1e-6, h / np.maximum(lh, 1e-12), np.array([1.0, 0.0], np.float32))
    I, J, K = np.meshgrid(np.arange(fl.shape[0], dtype=np.float32), np.arange(fl.shape[1], dtype=np.float32),
                          np.arange(fl.shape[2], dtype=np.float32), indexing="ij")
    a = sample3(fl, I + step * h[..., 0], J + step * h[..., 1], K)
    b = sample3(fl, I - step * h[..., 0], J - step * h[..., 1], K)
    return np.where((fl >= a) & (fl >= b), fl, 0.0).astype(np.float32)


# ====================================================================== main block computation
DEFAULTS = {"window": 5, "sigma_n": (1.5, 1.5, 6.0), "half_u": 4, "half_w": 6, "sigma_u": 2.0, "sigma_w": 3.0,
            "power": 8.0, "dip_steer": True, "passes": 2}


def fault_likelihood_block(block, dt_ms, params=None):
    """(ni, nj, ns) image -> dict fl, tfl, n (unit normals, index space), semblance, sheet.  Edges are reliable
    to ~half_w traces from the block border: use a halo."""
    p = dict(DEFAULTS, **(params or {}))
    blk = np.asarray(block, np.float32)
    blk = blk / (np.percentile(np.abs(blk), 99) + 1e-12)
    di = dj = None
    if p["dip_steer"]:
        di, dj, _ = dip_3d_block(blk)
    s = np.clip(semblance3(blk, dt_ms, int(p["window"]), di, dj), 0, 1)
    n, sheet = plane_normals(1.0 - s, p["sigma_n"])
    for _ in range(max(1, int(p["passes"]))):
        # pass 1 orientation from the raw discontinuity; later passes from the in-plane smoothed likelihood, a
        # cleaner sheet (tests: dip error 15 deg -> 2 deg, strike < 3 deg)
        s_pl = np.clip(plane_smooth(s, n, p["half_u"], p["half_w"], p["sigma_u"], p["sigma_w"]), 0, 1)
        fl = (1.0 - s_pl ** p["power"]).astype(np.float32)
        if _ < int(p["passes"]) - 1:
            n, sheet = plane_normals(fl, p["sigma_n"])
    return {"fl": fl, "tfl": thin(fl, n), "n": n, "semblance": s, "sheet": sheet}


def enhance(attr, n=None, iterations=2, power=1.5, params=None):
    """Ant-track-style enhancement of an edge attribute (high = edge, e.g. 1 - coherence, variance, TFL): repeated
    in-plane smoothing (orientation from the attribute's own structure tensor unless n is given), sharpening
    (power on the 0..1 normalised result) and a final thinning.  Returns (enhanced, normals)."""
    p = dict(DEFAULTS, **(params or {}))
    a = np.asarray(attr, np.float32)
    a = (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + 1e-12)
    for _ in range(max(1, int(iterations))):
        nn = n if n is not None else plane_normals(a, p["sigma_n"])[0]
        a = plane_smooth(a, nn, p["half_u"], p["half_w"], p["sigma_u"], p["sigma_w"])
        a = (a - a.min()) / (a.max() - a.min() + 1e-12)
        a = a ** power
    nn = n if n is not None else plane_normals(a, p["sigma_n"])[0]
    return thin(a, nn), nn


# ====================================================================== geography
def index_frame(vol=None, il_spacing=25.0, xl_spacing=25.0, azimuth_il=0.0):
    """Map direction (unit vectors, x = east, y = north) of +1 inline index and +1 crossline index, and their
    spacings (m).  From a Volume3D's affine when given."""
    if vol is not None:
        x00, y00 = vol.xy(vol.inlines[0], vol.xlines[0])
        x10, y10 = vol.xy(vol.inlines[min(1, vol.nil - 1)], vol.xlines[0])
        x01, y01 = vol.xy(vol.inlines[0], vol.xlines[min(1, vol.nxl - 1)])
        ei = np.array([float(x10 - x00), float(y10 - y00)])
        ej = np.array([float(x01 - x00), float(y01 - y00)])
        return ei, ej
    a = math.radians(azimuth_il)
    ei = il_spacing * np.array([math.sin(a), math.cos(a)])
    ej = xl_spacing * np.array([math.sin(a + math.pi / 2), math.cos(a + math.pi / 2)])
    return ei, ej


def geo_orientation(n, ei, ej, dt_ms, velocity=2500.0):
    """Index-space unit normals (..., 3) -> (strike azimuth deg 0-360 right-hand rule, dip deg 0-90).
    ei / ej = map vectors (m) of one inline / crossline step; time converted with velocity / 2."""
    n = np.asarray(n, float)
    # a plane n_i di + n_j dj + n_k dk = 0; in map metres the normal is the gradient of that form:
    # d(index) = M^-1 d(map), so the map-space normal = M^-T [n_i, n_j] (horizontal) and n_k / dz (vertical)
    M = np.column_stack([ei, ej])  # map = M @ [di, dj]
    Minv_T = np.linalg.inv(M).T
    nh = n[..., :2] @ Minv_T.T  # per metre
    dz = 0.5 * velocity * dt_ms / 1000.0  # metres per sample
    nz = n[..., 2] / dz
    hx, hy = nh[..., 0], nh[..., 1]
    hm = np.hypot(hx, hy)
    dip = np.degrees(np.arctan2(hm, np.abs(nz)))
    # dip direction = horizontal direction in which the plane deepens = direction of -nh * sign(nz)... with the
    # normal pointing up (nz < 0 in time-down coordinates) the horizontal part points down-dip
    s = np.where(nz <= 0, 1.0, -1.0)
    dipaz = np.degrees(np.arctan2(s * hx, s * hy)) % 360.0
    strike = (dipaz - 90.0) % 360.0
    return strike, dip


def plane_normal_index(strike_deg, dip_deg, ei, ej, dt_ms, velocity=2500.0):
    """Inverse of geo_orientation for one plane: unit normal in index space (pointing up)."""
    dipaz = math.radians(strike_deg + 90.0)
    dz = 0.5 * velocity * dt_ms / 1000.0
    nh = math.sin(math.radians(dip_deg)) * np.array([math.sin(dipaz), math.cos(dipaz)])  # unit map normal (down-dip)
    nz = -math.cos(math.radians(dip_deg))
    M = np.column_stack([ei, ej])
    ni = M.T @ nh  # index-space horizontal = M^T * map normal
    n = np.array([ni[0], ni[1], nz * dz])
    return n / np.linalg.norm(n)


# ====================================================================== volumes
def compute_fault_volumes(project, vol_name, params=None, block=12, velocity=2500.0, progress=None, cancel=None):
    """Fault likelihood, TFL, strike and dip volumes for a 3D volume (inline blocks with a halo).  Returns names."""
    from .inversion.common import write_volumes
    p = dict(DEFAULTS, **(params or {}))
    vol = project.volume(vol_name)
    dt_ms = vol.dt * 1000.0
    ei, ej = index_frame(vol)
    halo = int(max(p["half_u"], p["half_w"]) + 3 * max(p["sigma_n"][:2]) + 3)

    def compute(a, b):
        a0, b0 = max(0, a - halo), min(vol.nil, b + halo)
        blk = np.asarray(vol.ilxl[a0:b0], np.float32)
        r = fault_likelihood_block(blk, dt_ms, p)
        sl = slice(a - a0, a - a0 + (b - a))
        st, dp = geo_orientation(r["n"][sl], ei, ej, dt_ms, velocity)
        fl = r["fl"][sl]
        return {"Fault likelihood": fl, "TFL": r["tfl"][sl], "Fault strike": np.where(fl > 0.2, st, 0.0),
                "Fault dip": np.where(fl > 0.2, dp, 0.0)}
    names = ["Fault likelihood", "TFL", "Fault strike", "Fault dip"]
    return write_volumes(project, vol_name, names, compute, block, progress, cancel,
                         {"fault": {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()},
                                    "velocity": velocity}})


def section_likelihood(vol, mode, number, params=None, halo=None):
    """Fault likelihood on one inline / crossline of a volume (for a quick look in the section panel)."""
    p = dict(DEFAULTS, **(params or {}))
    h = halo or int(max(p["half_u"], p["half_w"]) + 4)
    if mode == "Inline":
        k = vol.il_index(number)
        a0, b0 = max(0, k - h), min(vol.nil, k + h + 1)
        r = fault_likelihood_block(np.asarray(vol.ilxl[a0:b0], np.float32), vol.dt * 1000.0, p)
        return r["fl"][k - a0]
    k = vol.xl_index(number)
    a0, b0 = max(0, k - h), min(vol.nxl, k + h + 1)
    blk = np.asarray(vol.ilxl[:, a0:b0], np.float32)
    r = fault_likelihood_block(blk, vol.dt * 1000.0, p)
    return r["fl"][:, k - a0]


# ====================================================================== 2D (PLAN 32g, likelihood part)
def fault_likelihood_2d(section, dt_ms, params=None):
    """(ntr, ns) 2D line -> (fl, tfl, normal (ntr, ns, 2) in (trace, sample) units).  2D analogue of the 3D
    method: 3-trace semblance along dip, normals from the 2D structure tensor, smoothing along the fault trace."""
    from .rgt.dip import dip_2d
    p = dict(DEFAULTS, **(params or {}))
    sec = np.asarray(section, np.float32)
    sec = sec / (np.percentile(np.abs(sec), 99) + 1e-12)
    dip, _ = dip_2d(sec)
    blk = sec[None]
    s = np.clip(semblance3(blk, dt_ms, int(p["window"]), np.zeros_like(blk), dip[None]), 0, 1)[0]
    d = 1.0 - s
    gx, gt = gradients(d, 1.0)
    sx, st = p["sigma_n"][1], p["sigma_n"][2]
    Txx, Txt, Ttt = gsmooth(gx * gx, (sx, st)), gsmooth(gx * gt, (sx, st)), gsmooth(gt * gt, (sx, st))
    tr = Txx + Ttt
    det = Txx * Ttt - Txt * Txt
    l1 = 0.5 * tr + np.sqrt(np.maximum(0.25 * tr * tr - det, 0))
    nx, nt = Txt, l1 - Txx  # eigenvector of l1
    bad = np.hypot(nx, nt) < 1e-12
    nx = np.where(bad, 1.0, nx)
    nt = np.where(bad, 0.0, nt)
    ln = np.hypot(nx, nt)
    nx, nt = nx / ln, nt / ln
    sgn = np.where(nt > 0, -1.0, 1.0)
    nx, nt = nx * sgn, nt * sgn
    # smoothing along the fault trace direction w = (-nt, nx) rotated to point down
    wx, wt = -nt, nx
    flip = wt < 0
    wx, wt = np.where(flip, -wx, wx), np.where(flip, -wt, wt)
    X, T = np.meshgrid(np.arange(sec.shape[0], dtype=np.float32), np.arange(sec.shape[1], dtype=np.float32),
                       indexing="ij")
    acc, wsum = np.zeros_like(s), 0.0
    for k in range(-p["half_w"], p["half_w"] + 1):
        wt_ = math.exp(-0.5 * (k / p["sigma_w"]) ** 2)
        acc += np.float32(wt_) * sample3(s[None], np.zeros_like(X), X + k * wx, T + k * wt)
        wsum += wt_
    s_pl = np.clip(acc / wsum, 0, 1)
    fl = (1.0 - s_pl ** p["power"]).astype(np.float32)
    a = sample3(fl[None], np.zeros_like(X), X + np.sign(nx), T)
    b = sample3(fl[None], np.zeros_like(X), X - np.sign(nx), T)
    tfl = np.where((fl >= a) & (fl >= b), fl, 0.0).astype(np.float32)
    return fl, tfl, np.stack([nx, nt], -1).astype(np.float32)
