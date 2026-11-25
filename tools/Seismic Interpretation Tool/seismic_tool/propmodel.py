"""Property modelling + volumetrics on the item-35 3D grid (PLAN 36) - numpy only.

36a `upscale_well`: log samples (MD -> x, y, TVDSS through the trajectory) into the cells the well passes through
    (arithmetic / harmonic / geometric mean, 'most' for discrete facies), with a raw-vs-upscaled QC table.
36b `variogram_3d` (experimental, along an azimuth with angle tolerance, or vertical) + `fit_variogram` (spherical /
    exponential / gaussian, least squares), `histogram`, `zone_stats`.
36c `sis` (sequential indicator simulation per zone, optional soft probability trend) and `indicator_kriging`.
36d `krige` (simple kriging of a continuous property) and `sgs` (sequential Gaussian simulation with a normal-score
    transform), both per zone, conditioned to upscaled cells and optionally to a facies grid; fixed seeds.
36e `volumetrics` (GRV, NTG, porosity, Sw, Bo / Bg, contacts OWC / GOC, per zone and fault block; P90 / P50 / P10
    over realizations), `block_summary` for the Live Map / WellTable, `write_csv`.
Neighbourhoods use an anisotropic index-space template (ranges in cells along major / minor / vertical after the
azimuth rotation) so a cell's conditioning data are found by table look-ups, not by distance sorts.
"""
import csv
import math

import numpy as np


# ====================================================================== geometry helpers
def cell_centres(grid):
    """(X, Y) (ny, nx) and Z (nz, ny, nx) of the cell centres (depth = mean of the 8 corners)."""
    Xc, Yc = grid.geom.centres()
    top = grid.zcorn[0::2].reshape(grid.nz, grid.ny, 2, grid.nx, 2).mean(axis=(2, 4))
    bot = grid.zcorn[1::2].reshape(grid.nz, grid.ny, 2, grid.nx, 2).mean(axis=(2, 4))
    return Xc, Yc, 0.5 * (top + bot), top, bot


def column_of(grid, x, y):
    g = grid.geom
    a = math.radians(g.rotation)
    u = (np.asarray(x, float) - g.x0) * math.cos(a) + (np.asarray(y, float) - g.y0) * math.sin(a)
    v = -(np.asarray(x, float) - g.x0) * math.sin(a) + (np.asarray(y, float) - g.y0) * math.cos(a)
    j = np.floor(u / g.dx).astype(int)
    i = np.floor(v / g.dy).astype(int)
    ok = (i >= 0) & (i < g.ny) & (j >= 0) & (j < g.nx)
    return i, j, ok


# ====================================================================== 36a upscaling
def upscale_well(grid, x, y, z, values, method="arithmetic", discrete=False):
    """Samples (x, y, z TVDSS down-positive, value) of one well -> {(k, i, j): upscaled value} + QC rows."""
    x, y, z, v = (np.asarray(a, float) for a in (x, y, z, values))
    i, j, ok = column_of(grid, x, y)
    _, _, _, top, bot = cell_centres(grid)
    out, qc = {}, []
    sel = np.flatnonzero(ok & np.isfinite(z) & np.isfinite(v))
    if not len(sel):
        return out, qc
    ks = np.full(len(sel), -1)
    for n, s in enumerate(sel):
        tcol, bcol = top[:, i[s], j[s]], bot[:, i[s], j[s]]
        kk = np.flatnonzero((z[s] >= tcol) & (z[s] < bcol) & grid.actnum[:, i[s], j[s]].astype(bool))
        ks[n] = kk[0] if len(kk) else -1
    groups = {}
    for n, s in enumerate(sel):
        if ks[n] >= 0:
            groups.setdefault((int(ks[n]), int(i[s]), int(j[s])), []).append(v[s])
    for key, vals in groups.items():
        a = np.asarray(vals, float)
        if discrete or method == "most":
            u, c = np.unique(np.round(a).astype(int), return_counts=True)
            r = float(u[np.argmax(c)])
        elif method == "harmonic":
            r = float(len(a) / np.sum(1.0 / np.maximum(a, 1e-12)))
        elif method == "geometric":
            r = float(np.exp(np.mean(np.log(np.maximum(a, 1e-12)))))
        else:
            r = float(np.mean(a))
        out[key] = r
        qc.append({"k": key[0], "i": key[1], "j": key[2], "samples": len(a), "raw_mean": float(np.mean(a)),
                   "raw_min": float(np.min(a)), "raw_max": float(np.max(a)), "upscaled": r})
    return out, qc


def cells_to_array(grid, cells, fill=np.nan):
    a = np.full((grid.nz, grid.ny, grid.nx), fill, float)
    for (k, i, j), v in cells.items():
        a[k, i, j] = v
    return a


# ====================================================================== 36b data analysis
def _model(h, sill, rng_, kind="spherical", nugget=0.0):
    h = np.asarray(h, float) / max(rng_, 1e-9)
    if kind == "exponential":
        g = 1 - np.exp(-3 * h)
    elif kind == "gaussian":
        g = 1 - np.exp(-3 * h * h)
    else:
        g = np.where(h < 1, 1.5 * h - 0.5 * h ** 3, 1.0)
    return nugget + (sill - nugget) * g


def variogram_3d(X, Y, Z, v, azimuth=0.0, tol=22.5, lags=10, lag=None, vertical=False, max_pairs=2_000_000):
    """Experimental semivariogram of scattered points along a map azimuth (deg from north, +- tol) or vertical.
    Returns (lag centres, gamma, pair counts)."""
    X, Y, Z, v = (np.asarray(a, float).ravel() for a in (X, Y, Z, v))
    ok = np.isfinite(v)
    X, Y, Z, v = X[ok], Y[ok], Z[ok], v[ok]
    n = len(v)
    if n < 2:
        return np.zeros(0), np.zeros(0), np.zeros(0, int)
    if n * (n - 1) / 2 > max_pairs:
        k = np.random.default_rng(0).choice(n, int(math.sqrt(2 * max_pairs)), replace=False)
        X, Y, Z, v = X[k], Y[k], Z[k], v[k]
        n = len(v)
    a, b = np.triu_indices(n, 1)
    dx, dy, dz = X[b] - X[a], Y[b] - Y[a], Z[b] - Z[a]
    hh = np.hypot(dx, dy)
    if vertical:
        sel = hh <= 1e-6 + 0.25 * np.abs(dz)
        d = np.abs(dz)
    else:
        ang = np.degrees(np.arctan2(dx, dy)) % 180.0
        diff = np.abs((ang - azimuth % 180.0 + 90.0) % 180.0 - 90.0)
        sel = (diff <= tol) & (np.abs(dz) <= 0.5 * max(np.ptp(Z) / max(len(np.unique(Z)), 1), 1e-6) + 1e-6 + 2.0)
        d = hh
    d, g = d[sel], 0.5 * (v[b][sel] - v[a][sel]) ** 2
    if not len(d):
        return np.zeros(0), np.zeros(0), np.zeros(0, int)
    lag = lag or float(np.max(d)) / (2 * lags)
    k = np.floor(d / lag).astype(int)
    m = k < lags
    gam = np.bincount(k[m], weights=g[m], minlength=lags)
    cnt = np.bincount(k[m], minlength=lags)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.arange(lags) + 0.5) * lag, gam / np.where(cnt > 0, cnt, np.nan), cnt


def fit_variogram(h, gam, counts=None, kind="spherical"):
    """(sill, range, nugget) by least squares on a scan of ranges (pair-count weighted)."""
    ok = np.isfinite(gam)
    h, gam = np.asarray(h)[ok], np.asarray(gam)[ok]
    w = np.ones(len(h)) if counts is None else np.asarray(counts, float)[ok]
    if not len(h):
        return 1.0, 1.0, 0.0
    best = None
    for r in np.linspace(max(h.min(), 1e-6), 2 * h.max(), 60):
        f = _model(h, 1.0, r, kind)
        A = np.column_stack([f, 1 - f])  # sill part + nugget part
        coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], gam * np.sqrt(w), rcond=None)
        sill_minus, nug = max(coef[0], 1e-12), max(coef[1], 0.0)
        res = np.sum(w * (A @ np.array([sill_minus, nug]) - gam) ** 2)
        if best is None or res < best[0]:
            best = (res, sill_minus + 0.0, r, nug)
    _, s, r, nug = best
    return float(s), float(r), float(min(nug, s))


def histogram(values, bins=20):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.zeros(bins + 1), np.zeros(bins, int)
    c, e = np.histogram(v, bins)
    return e, c


def zone_stats(grid, prop):
    out = []
    for z in sorted(set(grid.zone.tolist())):
        v = prop[grid.zone == z]
        v = v[np.isfinite(v)]
        out.append({"zone": int(z), "n": int(len(v)), "mean": float(np.mean(v)) if len(v) else float("nan"),
                    "std": float(np.std(v)) if len(v) else float("nan"),
                    "p10": float(np.percentile(v, 10)) if len(v) else float("nan"),
                    "p90": float(np.percentile(v, 90)) if len(v) else float("nan")})
    return out


# ====================================================================== neighbourhood template
class Vario:
    """Anisotropic variogram in index space: ranges in cells along major (azimuth) / minor / vertical."""

    def __init__(self, sill=1.0, r_major=10.0, r_minor=None, r_vert=3.0, azimuth=0.0, nugget=0.0, kind="spherical"):
        self.sill, self.nugget, self.kind = float(sill), float(nugget), kind
        self.r_major, self.r_minor, self.r_vert = float(r_major), float(r_minor or r_major), float(r_vert)
        self.azimuth = float(azimuth)

    def scaled(self, dk, di, dj):
        """Anisotropic distance (1 = range) of index offsets (layer, row = north, column = east)."""
        a = math.radians(self.azimuth)
        u = dj * math.sin(a) + di * math.cos(a)  # along the major direction
        w = dj * math.cos(a) - di * math.sin(a)
        return np.sqrt((u / self.r_major) ** 2 + (w / self.r_minor) ** 2 + (dk / self.r_vert) ** 2)

    def cov(self, d):
        """Covariance at scaled distance d (sill - gamma)."""
        return self.sill - _model(d, self.sill, 1.0, self.kind, self.nugget) + np.where(np.asarray(d) == 0,
                                                                                         self.nugget, 0.0)

    def template(self, max_n=400):
        rk, ri, rj = int(math.ceil(self.r_vert)), int(math.ceil(max(self.r_major, self.r_minor))), \
            int(math.ceil(max(self.r_major, self.r_minor)))
        K, I, J = np.meshgrid(np.arange(-rk, rk + 1), np.arange(-ri, ri + 1), np.arange(-rj, rj + 1), indexing="ij")
        off = np.stack([K.ravel(), I.ravel(), J.ravel()], 1)
        d = self.scaled(off[:, 0], off[:, 1], off[:, 2])
        keep = (d <= 1.0) & (d > 0)
        off, d = off[keep], d[keep]
        o = np.argsort(d)
        return off[o][:max_n], d[o][:max_n]


def _sk_weights(vario, off_nb, cov_pairs=None):
    """Simple-kriging weights of neighbours (offsets from the target) and the kriging variance."""
    d0 = vario.scaled(off_nb[:, 0], off_nb[:, 1], off_nb[:, 2])
    c0 = vario.cov(d0)
    D = off_nb[:, None, :] - off_nb[None, :, :]
    C = vario.cov(vario.scaled(D[..., 0], D[..., 1], D[..., 2])) + 1e-9 * np.eye(len(off_nb))
    try:
        w = np.linalg.solve(C, c0)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(C, c0, rcond=None)[0]
    return w, max(vario.sill - float(w @ c0), 0.0)


def _zone_mask(grid, zone):
    m = grid.actnum.astype(bool)
    if zone is not None:
        m &= (grid.zone == zone)[:, None, None]
    return m


# ====================================================================== 36d kriging + SGS
def krige(grid, data, vario, zone=None, mean=None, max_nb=16, template=None):
    """Simple kriging of a continuous property; data (nz, ny, nx) with NaN where unknown.  Exact at data cells."""
    dat = np.asarray(data, float)
    mask = _zone_mask(grid, zone)
    known = np.isfinite(dat) & mask
    mu = float(np.nanmean(dat[known])) if mean is None else float(mean)
    out = np.where(known, dat, np.nan)
    off, _ = template if template is not None else vario.template()
    shape = np.array(dat.shape)
    for k, i, j in np.argwhere(mask & ~known):
        q = off + [k, i, j]
        ok = np.all((q >= 0) & (q < shape), axis=1)
        q2, o2 = q[ok], off[ok]
        hit = known[q2[:, 0], q2[:, 1], q2[:, 2]]
        q2, o2 = q2[hit][:max_nb], o2[hit][:max_nb]
        if not len(q2):
            out[k, i, j] = mu
            continue
        w, _ = _sk_weights(vario, o2)
        out[k, i, j] = mu + float(w @ (dat[q2[:, 0], q2[:, 1], q2[:, 2]] - mu))
    return out


def normal_score(v):
    """(transform function, back-transform function) from the data distribution."""
    s = np.sort(np.asarray(v, float)[np.isfinite(v)])
    n = len(s)
    p = (np.arange(n) + 0.5) / n
    from math import erf, sqrt
    # inverse normal CDF by bisection-free rational approximation (Acklam) on the plotting positions
    z = _norm_ppf(p)
    return (lambda x: np.interp(x, s, z)), (lambda y: np.interp(y, z, s)), erf, sqrt


def _norm_ppf(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    q = p - 0.5
    out = np.empty_like(p)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi)
    r = q[mid] * q[mid]
    out[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q[mid] / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    for m, sgn, pp in ((lo, 1.0, p), (hi, -1.0, 1 - p)):
        if m.any():
            t = np.sqrt(-2 * np.log(pp[m]))
            out[m] = sgn * (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / \
                ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1)
    return out


def sgs(grid, data, vario, zone=None, seed=0, max_nb=16, facies=None, facies_code=None, ref=None):
    """Sequential Gaussian simulation of a continuous property (normal-score space, simple kriging with the
    previously simulated cells, random path), honouring data cells; optional restriction to cells of one facies.
    `ref` = values whose distribution defines the normal-score transform (default: the data in the zone)."""
    rng = np.random.default_rng(seed)
    dat = np.asarray(data, float)
    mask = _zone_mask(grid, zone)
    if facies is not None and facies_code is not None:
        mask &= np.asarray(facies) == facies_code
    known = np.isfinite(dat) & mask
    refv = np.asarray(ref if ref is not None else dat[known], float)
    if len(refv[np.isfinite(refv)]) < 2:
        raise ValueError("sgs needs at least two conditioning values")
    fwd, back, _, _ = normal_score(refv)
    ys = np.full(dat.shape, np.nan)
    ys[known] = fwd(dat[known])
    v1 = Vario(1.0, vario.r_major, vario.r_minor, vario.r_vert, vario.azimuth, min(vario.nugget / max(vario.sill,
                                                                                                    1e-12), 0.99),
               vario.kind)
    off, _ = v1.template()
    have = known.copy()
    shape = np.array(dat.shape)
    path = np.argwhere(mask & ~known)
    path = path[rng.permutation(len(path))]
    for k, i, j in path:
        q = off + [k, i, j]
        ok = np.all((q >= 0) & (q < shape), axis=1)
        q2, o2 = q[ok], off[ok]
        hit = have[q2[:, 0], q2[:, 1], q2[:, 2]]
        q2, o2 = q2[hit][:max_nb], o2[hit][:max_nb]
        if len(q2):
            w, var = _sk_weights(v1, o2)
            m = float(w @ ys[q2[:, 0], q2[:, 1], q2[:, 2]])
        else:
            m, var = 0.0, 1.0
        ys[k, i, j] = m + math.sqrt(var) * rng.standard_normal()
        have[k, i, j] = True
    out = np.full(dat.shape, np.nan)
    out[mask] = back(ys[mask])
    out[known] = dat[known]
    return out


# ====================================================================== 36c facies
def indicator_kriging(grid, facies_data, codes, vario, zone=None, proportions=None):
    """Probability (len(codes), nz, ny, nx) of each facies by simple indicator kriging (normalised, clipped)."""
    fd = np.asarray(facies_data, float)
    mask = _zone_mask(grid, zone)
    known = np.isfinite(fd) & mask
    P = []
    for c in codes:
        ind = np.where(known, (fd == c).astype(float), np.nan)
        p0 = float(np.nanmean(ind)) if proportions is None else float(proportions[c])
        P.append(np.clip(krige(grid, ind, vario, zone, mean=p0), 0, 1))
    P = np.array(P)
    s = np.nansum(P, axis=0)
    with np.errstate(invalid="ignore"):
        return np.where(s > 0, P / s, 1.0 / len(codes))


def sis(grid, facies_data, codes, vario, zone=None, seed=0, max_nb=12, soft=None, proportions=None):
    """Sequential indicator simulation: random path, indicator kriging of each facies from data + simulated cells,
    probabilities combined with an optional soft trend (codes x grid probabilities, e.g. from a seismic attribute /
    inversion) by the tau-model with tau = 1 (product of odds ratios); a facies is drawn per cell."""
    rng = np.random.default_rng(seed)
    fd = np.asarray(facies_data, float)
    mask = _zone_mask(grid, zone)
    known = np.isfinite(fd) & mask
    out = np.where(known, fd, np.nan)
    prop = {c: (float(np.mean(fd[known] == c)) if proportions is None else float(proportions[c])) for c in codes}
    off, _ = vario.template()
    shape = np.array(fd.shape)
    path = np.argwhere(mask & ~known)
    path = path[rng.permutation(len(path))]
    have = known.copy()
    for k, i, j in path:
        q = off + [k, i, j]
        ok = np.all((q >= 0) & (q < shape), axis=1)
        q2, o2 = q[ok], off[ok]
        hit = have[q2[:, 0], q2[:, 1], q2[:, 2]]
        q2, o2 = q2[hit][:max_nb], o2[hit][:max_nb]
        p = np.array([prop[c] for c in codes], float)
        if len(q2):
            w, _ = _sk_weights(vario, o2)
            vals = out[q2[:, 0], q2[:, 1], q2[:, 2]]
            p = np.array([prop[c] + float(w @ ((vals == c).astype(float) - prop[c])) for c in codes])
        p = np.clip(p, 1e-6, None)
        if soft is not None:
            s = np.clip(np.asarray(soft)[:, k, i, j], 1e-6, 1 - 1e-6)
            pr = np.clip(np.array([prop[c] for c in codes]), 1e-6, 1 - 1e-6)
            odds = (p / (1 - np.minimum(p, 1 - 1e-6))) * (s / (1 - s)) / (pr / (1 - pr))
            p = odds / (1 + odds)
        p = p / p.sum()
        out[k, i, j] = codes[int(rng.choice(len(codes), p=p))]
        have[k, i, j] = True
    return out


# ====================================================================== 36e volumetrics
def volumetrics(grid, phi, sw=None, ntg=None, owc=None, goc=None, bo=1.2, bg=0.005, facies=None, net_codes=None,
                by=("zone", "block")):
    """In-place volumes per zone and fault block.  Depth domain grid (z down, m).  Cells above GOC: gas (GIIP / Bg),
    between GOC and OWC: oil (STOIIP / Bo); below OWC: water.  Contacts may be per block {block: depth}.
    Returns rows {zone, block, grv_m3, net_m3, pore_m3, hcpv_oil_m3, hcpv_gas_m3, stoiip_m3, giip_m3, stoiip_bbl,
    giip_scf}."""
    _, _, Zc, _, _ = cell_centres(grid)
    vol = np.where(grid.actnum > 0, grid.cell_volumes(), 0.0)
    ntg_a = np.ones(vol.shape) if ntg is None else np.asarray(ntg, float)
    if facies is not None and net_codes is not None:
        ntg_a = ntg_a * np.isin(np.asarray(facies), list(net_codes))
    phi_a = np.nan_to_num(np.asarray(phi, float))
    sw_a = np.ones(vol.shape) * 0.3 if sw is None else np.nan_to_num(np.asarray(sw, float), nan=1.0)
    blocks = grid.block_index()
    zones = grid.zone_index()

    def contact(c, b):
        if c is None:
            return None
        return float(c.get(int(b), np.nan)) if isinstance(c, dict) else float(c)
    rows = []
    for z in np.unique(zones):
        for b in np.unique(blocks):
            m = (zones == z) & (blocks == b)
            if not m.any():
                continue
            ow, gc = contact(owc, b - 1), contact(goc, b - 1)
            zz = Zc[m]
            oil = np.ones(zz.shape, bool) if ow is None or not np.isfinite(ow) else zz < ow
            gas = np.zeros(zz.shape, bool) if gc is None or not np.isfinite(gc) else zz < gc
            oil &= ~gas
            grv = vol[m]
            net = grv * ntg_a[m]
            pore = net * phi_a[m]
            hc = pore * (1 - sw_a[m])
            ho, hg = float(np.sum(hc[oil])), float(np.sum(hc[gas]))
            rows.append({"zone": int(z), "block": int(b), "grv_m3": float(np.sum(grv[oil | gas])),
                         "net_m3": float(np.sum(net[oil | gas])), "pore_m3": float(np.sum(pore[oil | gas])),
                         "hcpv_oil_m3": ho, "hcpv_gas_m3": hg, "stoiip_m3": ho / bo, "giip_m3": hg / bg,
                         "stoiip_bbl": ho / bo * 6.28981, "giip_scf": hg / bg * 35.3147})
    return rows


def totals(rows, key="stoiip_m3"):
    return float(sum(r[key] for r in rows))


def realization_stats(values):
    """P90 / P50 / P10 (exceedance: P90 = 10th percentile) of a list of realization totals."""
    v = np.asarray(values, float)
    return {"P90": float(np.percentile(v, 10)), "P50": float(np.percentile(v, 50)), "P10": float(np.percentile(v, 90)),
            "mean": float(np.mean(v)), "n": int(len(v))}


def block_summary(grid, rows, key="stoiip_m3"):
    """Per fault block total + map position (block centroid) - points for the Live Map / WellTable."""
    Xc, Yc = grid.geom.centres()
    out = []
    for b in sorted({r["block"] for r in rows}):
        m = grid.block == b - 1
        out.append({"block": b, "x": float(Xc[m].mean()) if m.any() else float("nan"),
                    "y": float(Yc[m].mean()) if m.any() else float("nan"),
                    key: float(sum(r[key] for r in rows if r["block"] == b))})
    return out


def write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return path
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path
