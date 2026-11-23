"""Low-frequency model (PLAN 19a): well AI / Vp / Vs / rho in time (from the ties), interpolated through the
structural framework and low-pass filtered.

Framework: horizons (TWT grids) split every trace into zones; inside a zone a sample has a stratigraphic
coordinate u - 'proportional' (0 at the top horizon .. 1 at the base), 'top' (ms below the top: parallel to
the top) or 'base' (ms above the base); above the first / below the last horizon u is parallel to it.
In 'rgt' mode the coordinate is the relative geologic time (RGT volume) - interpolation along stack levels.
A target sample takes, from every well, the well's value at the same (zone, u) and combines them with lateral
weights (IDW or simple-kriging weights, both linear in the well values); interpolation is in ln(value), so
the model is positive and geometric between wells. Optional trend volume (e.g. interval velocity): the wells'
ln(value / trend) is interpolated and multiplied back on the trend (velocity-guided model).
Finally a zero-phase Butterworth low-pass (cut from the seismic spectrum).
"""
from dataclasses import dataclass, field

import numpy as np

from .common import butter_filter

PROPS = ("ai", "vp", "vs", "rho")
MODES = ("proportional", "top", "base")


@dataclass
class WellCurve:
    name: str
    x: float
    y: float
    t: np.ndarray  # (n,) ms, regular (seismic axis)
    values: dict = field(default_factory=dict)  # prop -> (n,) (NaN outside the log)


# ====================================================================== framework
def fill_grid(g, iters=200):
    """Fill NaN holes of a horizon grid by repeated neighbour averaging (edges: nearest)."""
    g = np.asarray(g, float).copy()
    if np.isfinite(g).all():
        return g
    if not np.isfinite(g).any():
        raise ValueError("horizon has no picks")
    for _ in range(iters):
        bad = ~np.isfinite(g)
        if not bad.any():
            break
        p = np.pad(g, 1, mode="edge")
        nb = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        cnt = np.isfinite(nb).sum(0)
        s = np.nansum(nb, 0)
        upd = bad & (cnt > 0)
        g[upd] = s[upd] / cnt[upd]
    g[~np.isfinite(g)] = np.nanmedian(g)
    return g


def strat_coords(t, tops, modes):
    """t (ns,) ms; tops (b, K) horizon times at b traces (monotonic down); modes (K + 1,) per zone.
    Returns zone (b, ns) int in 0..K and u (b, ns)."""
    t = np.asarray(t, float)[None, :]
    tops = np.maximum.accumulate(np.asarray(tops, float), axis=1)  # horizons never cross
    b, K = tops.shape
    zone = np.zeros((b, t.shape[1]), int)
    for k in range(K):
        zone += (t >= tops[:, k:k + 1]).astype(int)
    u = np.zeros(zone.shape)
    for z in range(K + 1):
        m = zone == z
        if not m.any():
            continue
        mode = modes[z] if z < len(modes) else "top"
        if z == 0:
            mode = "base"
        elif z == K:
            mode = "top"
        top = tops[:, z - 1:z] if z > 0 else None
        base = tops[:, z:z + 1] if z < K else None
        if mode == "proportional":
            uu = (t - top) / np.maximum(base - top, 1e-6)
        elif mode == "base":
            uu = t - base
        else:
            uu = t - top
        u = np.where(m, uu, u)
    return zone, u


def lateral_weights(xq, yq, px, py, method="idw", power=2.0, vrange=None):
    """Weights (nq, nw) of the wells at query points, rows sum to 1. idw: 1/d^p (a well at the point takes it
    all); kriging: simple-kriging weights (spherical variogram, range vrange or half the well spread) with the
    remainder spread evenly (the mean term)."""
    xq, yq = np.atleast_1d(np.asarray(xq, float)), np.atleast_1d(np.asarray(yq, float))
    px, py = np.asarray(px, float), np.asarray(py, float)
    nw = len(px)
    if nw == 1:
        return np.ones((len(xq), 1))
    d = np.hypot(xq[:, None] - px[None, :], yq[:, None] - py[None, :])
    if method == "kriging":
        from ..vgrid import _shape
        dw = np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :])
        a = float(vrange) if vrange else max(1.0, 0.5 * float(dw.max()))
        C = 1.0 - _shape(dw, a, "spherical") + np.eye(nw) * 1e-6
        c = 1.0 - _shape(d, a, "spherical")
        lam = np.linalg.solve(C, c.T).T
        return lam + (1.0 - lam.sum(1, keepdims=True)) / nw
    near = d < 1e-6
    w = 1.0 / np.maximum(d, 1e-6) ** power
    w = np.where(near.any(1, keepdims=True), near.astype(float), w)
    return w / w.sum(1, keepdims=True)


class LFM:
    """Interpolator built from well curves; call block(...) for traces."""

    def __init__(self, curves, prop="ai", modes=None, method="idw", power=2.0, vrange=None):
        self.curves = [c for c in curves if prop in c.values and np.isfinite(c.values[prop]).sum() >= 3]
        if not self.curves:
            raise ValueError(f"no well has a usable {prop} curve in time")
        self.prop = prop
        self.modes = list(modes or [])
        self.method, self.power, self.vrange = method, power, vrange
        self.px = np.array([c.x for c in self.curves])
        self.py = np.array([c.y for c in self.curves])
        # global ln-trend in time, used where no well reaches a zone
        tt = np.concatenate([c.t[np.isfinite(c.values[prop])] for c in self.curves])
        vv = np.concatenate([np.log(c.values[prop][np.isfinite(c.values[prop])]) for c in self.curves])
        A = np.c_[np.ones_like(tt), tt]
        self.trend_coef = np.linalg.lstsq(A, vv, rcond=None)[0] if np.ptp(tt) > 0 else np.array([vv.mean(), 0.0])
        self._well_coords = None

    def set_well_coords(self, coords):
        """coords: list per well of (zone (n,), u (n,)) at the well's own samples (from its trace)."""
        self._well_coords = coords

    def _well_values(self, j, zone, u, trend_w=None):
        """ln value of well j at target (zone, u) arrays (any shape); NaN where the well lacks that zone."""
        c = self.curves[j]
        v = np.log(c.values[self.prop])
        if trend_w is not None:
            v = v - np.log(trend_w)
        zw, uw = self._well_coords[j]
        out = np.full(zone.shape, np.nan)
        for z in np.unique(zone):
            m = (zw == z) & np.isfinite(v)
            if m.sum() < 1:
                continue
            o = np.argsort(uw[m])
            xs, ys = uw[m][o], v[m][o]
            sel = zone == z
            out[sel] = np.interp(u[sel], xs, ys)
        return out

    def block(self, xq, yq, t, zone, u, trend=None, trend_wells=None):
        """ln-domain interpolation at traces (b,) with coordinates zone/u (b, ns); trend (b, ns) optional.
        Returns values (b, ns) (not filtered)."""
        W = lateral_weights(xq, yq, self.px, self.py, self.method, self.power, self.vrange)  # (b, nw)
        num = np.zeros(zone.shape)
        den = np.zeros(zone.shape)
        for j in range(len(self.curves)):
            vj = self._well_values(j, zone, u, None if trend_wells is None else trend_wells[j])
            ok = np.isfinite(vj)
            num += np.where(ok, W[:, j:j + 1] * vj, 0.0)
            den += np.where(ok, W[:, j:j + 1], 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            lv = num / den
        if trend is not None:
            lv = lv + np.log(trend)
            lv = np.where(np.isfinite(lv), lv, np.log(trend))
        fallback = self.trend_coef[0] + self.trend_coef[1] * np.broadcast_to(np.asarray(t, float), zone.shape)
        lv = np.where(np.isfinite(lv), lv, fallback)
        return np.exp(lv)


def lowpass(values, dt_ms, fc, order=4):
    """Zero-phase low-pass of model traces (b, ns) in ln domain (keeps the model positive)."""
    if not fc:
        return values
    return np.exp(butter_filter(np.log(values), dt_ms, fc, order, axis=-1))


# ====================================================================== project wrapper (3D)
def well_curves(project, vol_name, wells=None, despike=True):
    """WellCurve per well of the linked well project with logs, on the volume's time axis (tied TWT)."""
    from .common import well_elastic
    ws = project.wellset()
    if ws.wcp is None:
        raise ValueError("link a well project first")
    vol = project.volume(vol_name)
    out = []
    for name in wells or ws.names:
        try:
            w = ws.wcp.load_well(name)
            td = ws.td_for(name)
            el = well_elastic(w, td, vol.dt * 1000.0, t_axis=vol.times, despike=despike)
        except Exception:
            el = None
        if el is None:
            continue
        il, xl = vol.ilxl_at(w.x, w.y)
        if not (vol.inlines.min() - vol.il_step <= il <= vol.inlines.max() + vol.il_step and
                vol.xlines.min() - vol.xl_step <= xl <= vol.xlines.max() + vol.xl_step):
            continue
        out.append(WellCurve(name, float(w.x), float(w.y), el["t"], {k: el[k] for k in PROPS}))
    return out


def horizon_grids(project, vol_name, names):
    key = f"3D:{vol_name}"
    out = []
    for n in names:
        h = project.horizons.get(n)
        g = h.get(key) if h is not None else None
        if g is None:
            raise ValueError(f"horizon {n!r} has no picks on {vol_name}")
        out.append(fill_grid(g))
    order = np.argsort([np.nanmedian(g) for g in out])
    return [out[i] for i in order], [names[i] for i in order]


class VolumeLFM:
    """LFM on a 3D volume: framework from horizons (or the RGT volume), wells from WellCurves."""

    def __init__(self, project, vol_name, curves, prop="ai", horizons=(), modes=None, method="idw", power=2.0,
                 vrange=None, fc=None, use_rgt=False, trend_volume=None):
        self.project, self.vol_name = project, vol_name
        self.vol = vol = project.volume(vol_name)
        self.fc = fc
        self.lfm = LFM(curves, prop, modes, method, power, vrange)
        self.grids, self.hnames = horizon_grids(project, vol_name, list(horizons)) if horizons else ([], [])
        self.rgt = None
        if use_rgt:
            import os

            from ..rgt.solve import RGTVolume
            folder = os.path.join(vol.store, "rgt")
            if not RGTVolume.exists(folder):
                raise ValueError("no RGT for this volume - Interpret > Compute RGT (3D)")
            self.rgt = RGTVolume(folder)
        self.trend = project.volume(trend_volume) if trend_volume else None
        self.ij = []
        coords, tw = [], []
        for c in self.lfm.curves:
            il, xl = vol.ilxl_at(c.x, c.y)
            i, j = vol.il_index(il), vol.xl_index(xl)
            self.ij.append((i, j))
            z, u = self._coords(np.array([i]), np.array([j]))
            coords.append((z[0], u[0]))
            if self.trend is not None:
                tw.append(self._trend_values(np.array([i]), np.array([j]))[0])
        self.lfm.set_well_coords(coords)
        self.trend_wells = tw if self.trend is not None else None

    def _coords(self, ii, jj):
        vol = self.vol
        if self.rgt is not None:
            tau = np.array([self.rgt.decode(self.rgt.arr[i, j]) for i, j in zip(ii, jj)])
            return np.zeros(tau.shape, int), tau
        if not self.grids:
            return np.zeros((len(ii), vol.ns), int), np.broadcast_to(vol.times, (len(ii), vol.ns)).copy()
        tops = np.stack([g[ii, jj] for g in self.grids], axis=1)
        return strat_coords(vol.times, tops, self.lfm.modes or ["proportional"] * (len(self.grids) + 1))

    def _trend_values(self, ii, jj):
        tr = np.asarray(self.trend.ilxl[ii, jj], np.float32) * self.trend.inv
        tr = np.maximum(tr, 100.0)
        if self.lfm.prop == "ai":
            return 0.31 * tr ** 1.25  # Gardner AI from velocity
        if self.lfm.prop == "rho":
            return 0.31 * tr ** 0.25
        return tr

    def traces(self, ii, jj):
        """Filtered model at volume traces (index arrays) -> (b, ns)."""
        vol = self.vol
        ii, jj = np.asarray(ii), np.asarray(jj)
        x, y = vol.xy(vol.inlines[ii], vol.xlines[jj])
        zone, u = self._coords(ii, jj)
        trend = self._trend_values(ii, jj) if self.trend is not None else None
        v = self.lfm.block(x, y, vol.times, zone, u, trend, self.trend_wells)
        return lowpass(v, vol.dt * 1000.0, self.fc)

    def compute(self, a, b):
        vol = self.vol
        I, J = np.meshgrid(np.arange(a, b), np.arange(vol.nxl), indexing="ij")
        return self.traces(I.ravel(), J.ravel()).reshape(b - a, vol.nxl, vol.ns)

    def write(self, name=None, progress=None, cancel=None):
        from .common import write_volume
        name = name or f"LFM {self.lfm.prop.upper()}"
        meta = {"inversion": {"kind": "lfm", "prop": self.lfm.prop, "fc": self.fc, "wells": [c.name for c in
                                                                                             self.lfm.curves],
                              "horizons": self.hnames, "rgt": self.rgt is not None, "method": self.lfm.method}}
        return write_volume(self.project, self.vol_name, name, self.compute, progress=progress, cancel=cancel,
                            meta_extra=meta)
