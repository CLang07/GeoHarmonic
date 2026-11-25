"""Velocity-volume depth conversion (PLAN 37) - the advanced mode of the item-17 dynamic depth conversion.

WELL TOPS ARE THE HARD DATA, THE VELOCITY VOLUME IS THE TREND.

a. Velocities from the well ties: every well's final time-depth function (WellSet.td_for: tie shift / stretch / warp
   on top of checkshot > calibrated sonic > default) sampled on the volume's time axis -> interval velocity per
   sample; despiked, capped (vmin / vmax); QC: tie correlation (display['well_ties']), drift vs the raw checkshot.
b. Framework: the time horizons define zones; each zone is layered proportional / parallel to top / parallel to
   base (item-19 `lfm.strat_coords`).
c. Volume: each well's velocity curve resampled on the zone layers, interpolated along the layers between the wells
   (IDW or simple kriging, `lfm.lateral_weights`), optionally as a RATIO to a background trend (Dix interval
   velocities from a stacking / RMS velocity volume) so the trend carries the lateral change the wells miss.
d. Depth through the volume: z(t) = sum v dt / 2 per trace -> depth horizons, depth sections, depth SEG-Y.
e. Tops tie exactly: residual (top - volume depth) per well and horizon, gridded (TPS / kriging / IDW, exact at the
   wells) and added -> corrected depth maps; then the volume is CALIBRATED: each zone's velocities scaled per trace
   (scale = corrected / volume zone thickness) so integrating the calibrated volume reproduces the corrected maps.
f. Dynamic update (`set_top`, `update_well`): a moved top recomputes residuals + calibration for that horizon and
   below only; a new / edited tie updates that well's curve and only the traces it influences.
g. QC: leave-one-well-out blind test (`blind_test`), residual maps, velocity sections, scenarios + P10 / P50 / P90.
"""
import math
import time

import numpy as np

from .inversion.lfm import lateral_weights, strat_coords
from .maps import MapGrid


class WellCurve:
    def __init__(self, name, x, y, t, z, source="", corr=None):
        self.name, self.x, self.y = name, float(x), float(y)
        self.t = np.asarray(t, float)  # ms (volume axis)
        self.z = np.asarray(z, float)  # TVDSS m at t
        self.source = source
        self.corr = corr
        self.flags = []
        self.v = None  # interval velocity per sample (m/s)

    def interval_velocity(self, vmin=1300.0, vmax=7000.0, despike=True):
        dz = np.diff(self.z)
        dt = np.diff(self.t) / 1000.0
        v = np.r_[2.0 * dz / np.maximum(dt, 1e-9), np.nan]
        v[-1] = v[-2] if len(v) > 1 else 2000.0
        if despike and len(v) > 11:
            from .tie import despike as _ds
            v = _ds(v, 11, 4.0)[0]
        bad = ~np.isfinite(v) | (v < vmin) | (v > vmax)
        if bad.mean() > 0.2:
            self.flags.append(f"{bad.mean():.0%} of the velocities outside {vmin:g}-{vmax:g} m/s")
        v = np.clip(np.where(np.isfinite(v), v, np.nan), vmin, vmax)
        if np.isnan(v).any():
            ok = np.isfinite(v)
            v = np.interp(np.arange(len(v)), np.flatnonzero(ok), v[ok]) if ok.any() else np.full(len(v), 2000.0)
        self.v = v
        return v


def dix(t_ms, vrms):
    """Dix interval velocities from RMS velocities along the last axis (ms, m/s)."""
    t = np.asarray(t_ms, float)
    v = np.asarray(vrms, float)
    a = t * v * v
    num = np.diff(a, axis=-1)
    den = np.diff(t)
    vi = np.sqrt(np.maximum(num / np.maximum(den, 1e-9), 1.0))
    return np.concatenate([v[..., :1], vi], axis=-1)


class VelocityVolumeModel:
    """horizons: [(horizon name, formation name or None)] top to bottom (formation = the well-top name)."""

    def __init__(self, project, vol_name, horizons, wells=None, modes=None, method="idw", power=2.0, vrange=None,
                 residual_method="tps", nu=24, vmin=1300.0, vmax=7000.0, trend=None, td_fn=None, tops_fn=None,
                 exclude=None, min_tie_corr=0.5):
        self.project, self.vol_name = project, vol_name
        self.layers = [(h, f) for h, f in horizons]
        self.vol = project.volume(vol_name)
        self.modes = list(modes or ["proportional"] * (len(self.layers) + 1))
        self.method, self.power, self.vrange = method, power, vrange
        self.residual_method = residual_method
        self.nu = int(nu)
        self.vmin, self.vmax = vmin, vmax
        self.trend = trend  # None or {"volume": name, "kind": "rms" | "interval"} or an array (nil, nxl, ns)
        self.td_fn = td_fn  # callable(name) -> object with depth(twt ms) (default: project WellSet.td_for)
        self.tops_fn = tops_fn  # callable() -> {well: {formation: TVDSS}} (default: linked well project)
        self.wells = wells
        self.exclude = set(exclude or [])
        self.min_tie_corr = min_tie_corr
        self.curves = {}
        self.tops = {}
        self.history = []
        self._built = False

    # ------------------------------------------------------------------ inputs
    def time_grids(self):
        out = []
        for h, _ in self.layers:
            hz = self.project.horizons.get(h)
            g = hz.get(f"3D:{self.vol_name}") if hz is not None else None
            if g is None:
                raise ValueError(f"horizon {h} has no picks on {self.vol_name}")
            out.append(np.asarray(g, float))
        tg = np.stack(out)  # (K, nil, nxl)
        for k in range(len(tg)):  # fill holes so every trace has a framework
            if np.isnan(tg[k]).any():
                from .inversion.lfm import fill_grid
                tg[k] = fill_grid(tg[k])
        return np.maximum.accumulate(tg, axis=0)

    def _wellset(self):
        return self.project.wellset()

    def _well_list(self):
        if self.wells is not None:
            return [w for w in self.wells if w[0] not in self.exclude]
        return [w for w in self.project.wells() if w[0] not in self.exclude]

    def _td(self, name):
        if self.td_fn is not None:
            return self.td_fn(name), "given"
        ws = self._wellset()
        td = ws.td_for(name)
        return td, getattr(td, "source", "")

    def _tops(self):
        if self.tops_fn is not None:
            return self.tops_fn()
        import os
        ws = self._wellset()
        out = {}
        wcp = ws.wcp
        path = getattr(self.project, "well_project", None)
        if path and os.path.exists(path):  # tops change on disk (Well Correlation Tool): read the file again
            mt = os.path.getmtime(path)
            if getattr(self, "_wcp_key", None) != (path, mt):
                from .welltie import wc
                self._wcp = wc().project.Project.open(path)
                self._wcp_key = (path, mt)
            wcp = self._wcp
        for name, _, _ in self.project.wells():
            d = (wcp.tops or {}).get(name, {}) if wcp else {}
            w = wcp.wells.get(name) if wcp else None
            if w is None:
                continue
            try:
                wcp.load_well(name)
            except Exception:
                pass
            out[name] = {fm: float(w.md_to_tvdss(md)) for fm, md in d.items() if md is not None}
        return out

    def well_curves(self):
        """37a: time-depth -> interval velocity per well on the volume time axis, with QC flags."""
        t = np.asarray(self.vol.times, float)
        ties = (self.project.display.get("well_ties") or {}) if hasattr(self.project, "display") else {}
        out = {}
        for name, x, y in self._well_list():
            try:
                td, src = self._td(name)
            except Exception as e:
                continue
            z = np.asarray(td.depth(t), float)
            corr = ties.get(name, {}).get("corr")
            c = WellCurve(name, x, y, t, z, src, corr)
            c.interval_velocity(self.vmin, self.vmax)
            if corr is not None and corr < self.min_tie_corr:
                c.flags.append(f"poor tie (correlation {corr:.2f})")
            out[name] = c
        self.curves = out
        return out

    def qc_table(self):
        return [{"well": c.name, "source": c.source, "corr": c.corr, "v_mean": float(np.mean(c.v)),
                 "flags": "; ".join(c.flags)} for c in self.curves.values()]

    # ------------------------------------------------------------------ framework + curves on layers (37b)
    def _u_grids(self, tg):
        """u axis per zone: proportional 0..1 (nu levels), else ms offsets covering the thickest zone."""
        t = np.asarray(self.vol.times, float)
        dt = float(t[1] - t[0]) if len(t) > 1 else 4.0
        K = len(self.layers)
        out = []
        for z in range(K + 1):
            mode = self.modes[z] if z < len(self.modes) else "top"
            if z == 0:
                mode = "base"
            elif z == K:
                mode = "top"
            if mode == "proportional":
                out.append(("proportional", np.linspace(0.0, 1.0, self.nu)))
            elif mode == "base":
                span = float(np.nanmax(tg[0]) - t[0]) + dt
                out.append(("base", np.arange(-span, dt, dt)))
            else:
                top = tg[z - 1]
                span = float(t[-1] - np.nanmin(top)) + dt
                out.append(("top", np.arange(0.0, span + dt, dt)))
        return out

    def _well_zone_curves(self, tg):
        """{well: [per zone (nu_z,) velocity on the zone's u axis (NaN when the well has no sample there)]}."""
        t = np.asarray(self.vol.times, float)
        ug = self._u_grids(tg)
        out = {}
        for name, c in self.curves.items():
            il, xl = self.vol.ilxl_at(c.x, c.y)
            i = int(np.clip(round((float(il) - self.vol.il0) / self.vol.il_step), 0, self.vol.nil - 1))
            j = int(np.clip(round((float(xl) - self.vol.xl0) / self.vol.xl_step), 0, self.vol.nxl - 1))
            tops = tg[:, i, j][None, :]
            zone, u = strat_coords(t, tops, self.modes)
            zone, u = zone[0], u[0]
            v = c.v
            if self._trend is not None:
                v = v / np.maximum(self._trend_trace(i, j), 1.0)
            per = []
            for z, (_, ux) in enumerate(ug):
                m = zone == z
                if m.sum() < 2:
                    per.append(np.full(len(ux), np.nan))
                    continue
                o = np.argsort(u[m])
                per.append(np.interp(ux, u[m][o], v[m][o]))
            out[name] = per
        self._ug = ug
        return out

    # ------------------------------------------------------------------ trend (Dix)
    def _load_trend(self):
        self._trend = None
        if self.trend is None:
            return None
        if isinstance(self.trend, np.ndarray):  # interval velocities on the volume grid
            arr = np.asarray(self.trend, np.float32)
            kind = "interval"
        else:
            kind = self.trend.get("kind", "rms")
            if self.trend.get("array") is not None:
                arr = np.asarray(self.trend["array"], np.float32)
            else:
                tv = self.project.volume(self.trend["volume"])
                arr = np.asarray(tv.ilxl[:], np.float32) * tv.inv
        if kind == "rms":
            arr = dix(np.asarray(self.vol.times, float), arr).astype(np.float32)
        arr = np.where(np.isfinite(arr) & (arr > 100.0), arr, np.nan)
        if np.isnan(arr).any():  # fill gaps along time with the trace's nearest valid value
            med = np.nanmedian(arr)
            arr = np.where(np.isnan(arr), med, arr)
        self._trend = arr.astype(np.float32)
        return self._trend

    def _trend_trace(self, i, j):
        return self._trend[i, j]

    # ------------------------------------------------------------------ the volume (37c)
    def build(self, progress=None, cancel=None):
        """Curves -> layered well velocities -> lateral weights -> depth grids (raw), then residuals + calibration."""
        t0 = time.perf_counter()
        if not self.curves:
            self.well_curves()
        if len(self.curves) < 1:
            raise ValueError("no well with a time-depth function")
        self._load_trend()
        self.tg = self.time_grids()
        self.zc_curves = self._well_zone_curves(self.tg)
        self.names = list(self.zc_curves)
        vol = self.vol
        I, J = np.meshgrid(vol.inlines, vol.xlines, indexing="ij")
        X, Y = vol.xy(I, J)
        self.X, self.Y = np.asarray(X, float), np.asarray(Y, float)
        px = np.array([self.curves[n].x for n in self.names])
        py = np.array([self.curves[n].y for n in self.names])
        self.W = lateral_weights(self.X.ravel(), self.Y.ravel(), px, py, self.method, self.power,
                                 self.vrange).reshape(vol.nil, vol.nxl, len(px))
        self.z_raw = self._depth_grids(progress=progress, cancel=cancel)
        self.tops = self._tops()
        self.residuals(0)
        self._built = True
        self.seconds = time.perf_counter() - t0
        return self

    def traces(self, ii, jj, calibrated=False):
        """Velocity traces (b, ns) at index arrays ii, jj."""
        t = np.asarray(self.vol.times, float)
        ii, jj = np.atleast_1d(ii), np.atleast_1d(jj)
        tops = self.tg[:, ii, jj].T
        zone, u = strat_coords(t, tops, self.modes)
        W = self.W[ii, jj]  # (b, nw)
        out = np.zeros(zone.shape)
        for z, (_, ux) in enumerate(self._ug):
            m = zone == z
            if not m.any():
                continue
            V = np.stack([self.zc_curves[n][z] for n in self.names])  # (nw, nu)
            f = np.interp(u[m], ux, np.arange(len(ux), dtype=float))
            k0 = np.clip(np.floor(f).astype(int), 0, len(ux) - 1)
            k1 = np.minimum(k0 + 1, len(ux) - 1)
            w1 = f - k0
            vals = V[:, k0] * (1 - w1) + V[:, k1] * w1  # (nw, npos)
            rows = np.nonzero(m)[0]
            Wp = W[rows]  # (npos, nw)
            ok = np.isfinite(vals.T)
            num = np.sum(np.where(ok, Wp * np.nan_to_num(vals.T), 0.0), axis=1)
            den = np.sum(np.where(ok, Wp, 0.0), axis=1)
            vz = num / np.maximum(den, 1e-12)
            # a zone no well samples: the nearest zone velocities above / below fill it (rare)
            vz = np.where(den > 1e-12, vz, np.nan)
            out[m] = vz
        if np.isnan(out).any():
            for r in range(len(out)):
                bad = np.isnan(out[r])
                if bad.any() and (~bad).any():
                    out[r, bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), out[r, ~bad])
        if self._trend is not None:
            out = out * self._trend[ii, jj]
        if calibrated and getattr(self, "scale", None) is not None:
            S = np.ones(zone.shape)
            for z in range(len(self.layers)):
                sz = self.scale[z][ii, jj][:, None]
                S = np.where(zone == z, sz, S)
            out = out * S
        return out

    def depth_traces(self, ii, jj, calibrated=False):
        """TVDSS (b, ns) by integrating the velocity traces from t = 0 (datum = sea level)."""
        t = np.asarray(self.vol.times, float)
        v = self.traces(ii, jj, calibrated)
        z0 = v[:, :1] * t[0] / 2000.0
        dz = v[:, :-1] * np.diff(t)[None, :] / 2000.0
        return np.concatenate([z0, z0 + np.cumsum(dz, axis=1)], axis=1)

    def _depth_grids(self, calibrated=False, mask=None, base=None, progress=None, cancel=None, block=4096):
        """Depth of every horizon at every trace (K, nil, nxl) through the (calibrated) volume.  mask = only those
        traces (the others copied from base)."""
        vol = self.vol
        K = len(self.layers)
        out = np.array(base, float) if base is not None else np.full((K, vol.nil, vol.nxl), np.nan)
        I, J = np.meshgrid(np.arange(vol.nil), np.arange(vol.nxl), indexing="ij")
        sel = np.flatnonzero(np.ones(vol.nil * vol.nxl, bool) if mask is None else np.ravel(mask))
        ii, jj = I.ravel()[sel], J.ravel()[sel]
        t = np.asarray(vol.times, float)
        for a in range(0, len(sel), block):
            if cancel and cancel():
                raise RuntimeError("cancelled")
            b = min(len(sel), a + block)
            zt = self.depth_traces(ii[a:b], jj[a:b], calibrated)
            for k in range(K):
                th = self.tg[k, ii[a:b], jj[a:b]]
                f = np.clip(np.interp(th, t, np.arange(len(t), dtype=float)), 0, len(t) - 1)
                k0 = np.floor(f).astype(int)
                k1 = np.minimum(k0 + 1, len(t) - 1)
                w = f - k0
                r = np.arange(b - a)
                out[k, ii[a:b], jj[a:b]] = zt[r, k0] * (1 - w) + zt[r, k1] * w
            if progress:
                progress(b / len(sel), f"velocity volume: {b}/{len(sel)} traces")
        return out

    # ------------------------------------------------------------------ hard data (37e)
    def _sample(self, grid, x, y):
        g = MapGrid(grid, "ilxl", "m", "", self.vol)
        return float(g.sample([x], [y])[0])

    def residual_points(self, k):
        h, fm = self.layers[k]
        px, py, r, names = [], [], [], []
        for name, d in self.tops.items():
            if fm not in d or name in self.exclude:
                continue
            xy = next(((x, y) for n, x, y in (self.wells or self.project.wells()) if n == name), None)
            if xy is None:
                continue
            zv = self._sample(self.z_raw[k], *xy)
            if not np.isfinite(zv):
                continue
            px.append(xy[0])
            py.append(xy[1])
            r.append(d[fm] - zv)
            names.append(name)
        return np.array(px), np.array(py), np.array(r), names

    def residuals(self, from_layer=0):
        """Residual grids + corrected depth maps + calibration scales for horizon from_layer and below."""
        from .vgrid import grid_values
        K = len(self.layers)
        if not hasattr(self, "R") or from_layer == 0:
            self.R = [None] * K
            self.z_corr = [None] * K
            self.scale = [None] * K
            self.res_points = [None] * K
        for k in range(from_layer, K):
            px, py, r, names = self.residual_points(k)
            R = np.zeros(self.z_raw[k].shape)
            if len(r):
                meth = self.residual_method if len(r) >= 3 else "idw"
                target = np.asarray(r, float)
                misfit = target.copy()
                for _ in range(6):  # wells sit between grid nodes: re-grid the misfit of the bilinear read-back
                    R = R + np.asarray(grid_values(self.X.ravel(), self.Y.ravel(), px, py, misfit, meth),
                                       float).reshape(R.shape)
                    g = MapGrid(R, "ilxl", "m", "", self.vol)
                    misfit = target - np.asarray(g.sample(px, py), float)
                    if np.nanmax(np.abs(misfit)) < 1e-3:
                        break
            self.R[k] = R
            self.z_corr[k] = self.z_raw[k] + R
            self.res_points[k] = (names, r)
        for k in range(from_layer, K):
            top_c = self.z_corr[k - 1] if k > 0 else np.zeros_like(self.z_corr[k])
            top_v = self.z_raw[k - 1] if k > 0 else np.zeros_like(self.z_raw[k])
            th_v = self.z_raw[k] - top_v
            th_c = self.z_corr[k] - top_c
            self.scale[k] = np.where(np.abs(th_v) > 1e-6, th_c / np.where(np.abs(th_v) > 1e-6, th_v, 1.0), 1.0)
        return self.z_corr

    def depth_grid(self, k, corrected=True):
        g = self.z_corr[k] if corrected else self.z_raw[k]
        return MapGrid(g, "ilxl", "m", f"{self.layers[k][0]} depth (velocity volume{'' if corrected else ', raw'})",
                       self.vol)

    def residual_grid(self, k):
        return MapGrid(self.R[k], "ilxl", "m", f"{self.layers[k][0]} residual (top - volume)", self.vol)

    def layer_velocity_grid(self, k, calibrated=True):
        """Average interval velocity of zone k (above horizon k) - map-hub layer."""
        top = (self.z_corr[k - 1] if calibrated else self.z_raw[k - 1]) if k > 0 else 0.0
        base = self.z_corr[k] if calibrated else self.z_raw[k]
        tt = self.tg[k - 1] if k > 0 else np.full(self.tg[k].shape, float(self.vol.times[0]) * 0.0)
        dt = (self.tg[k] - tt) / 2000.0
        return MapGrid((base - top) / np.maximum(dt, 1e-9), "ilxl", "m/s",
                       f"Interval velocity above {self.layers[k][0]} (volume)", self.vol)

    def depth_of_time(self, x, y, times, calibrated=True):
        """TVDSS at map points x, y (arrays) for times (ms, same for all) - section / volume depth conversion."""
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        il, xl = self.vol.ilxl_at(x, y)
        ii = np.clip(np.round((np.asarray(il, float) - self.vol.il0) / self.vol.il_step).astype(int), 0, self.vol.nil - 1)
        jj = np.clip(np.round((np.asarray(xl, float) - self.vol.xl0) / self.vol.xl_step).astype(int), 0, self.vol.nxl - 1)
        zt = self.depth_traces(ii, jj, calibrated)
        t = np.asarray(self.vol.times, float)
        tq = np.asarray(times, float)
        return np.array([np.interp(tq, t, zt[r]) for r in range(len(zt))])

    # ------------------------------------------------------------------ dynamic update (37f)
    def set_top(self, well, formation, tvdss):
        """A new / moved top: residual + calibration for that horizon and below.  Returns the report."""
        t0 = time.perf_counter()
        self.tops.setdefault(well, {})[formation] = float(tvdss)
        ks = [k for k, (_, fm) in enumerate(self.layers) if fm == formation]
        if not ks:
            return None
        k0 = ks[0]
        self.residuals(k0)
        rep = {"changes": [f"{well} {formation}"], "from_layer": k0, "layers": list(range(k0, len(self.layers))),
               "seconds": time.perf_counter() - t0,
               "message": f"{well} {formation} -> residual + calibration rebuilt for "
                          f"{', '.join(h for h, _ in self.layers[k0:])} in {time.perf_counter() - t0:.2f} s"}
        self.history.append(rep)
        return rep

    def refresh_tops(self, tol=0.01):
        """Re-read the tops (linked well project): every new / moved top -> set_top (shallowest layer first).
        Returns the combined report or None when nothing changed (called by the dynamic-update tick)."""
        new = self._tops()
        changes = []
        for w, d in new.items():
            for fm, z in d.items():
                old = self.tops.get(w, {}).get(fm)
                if old is None or abs(old - z) > tol:
                    changes.append((w, fm, z))
        if not changes:
            return None
        order = {fm: k for k, (_, fm) in enumerate(self.layers)}
        changes.sort(key=lambda c: order.get(c[1], 99))
        t0 = time.perf_counter()
        k0 = None
        for w, fm, z in changes:
            self.tops.setdefault(w, {})[fm] = float(z)
            if fm in order:
                k0 = order[fm] if k0 is None else min(k0, order[fm])
        if k0 is None:
            return None
        self.residuals(k0)
        rep = {"changes": [f"{w} {fm}" for w, fm, _ in changes], "from_layer": k0,
               "layers": list(range(k0, len(self.layers))), "seconds": time.perf_counter() - t0,
               "message": f"{', '.join(f'{w} {fm}' for w, fm, _ in changes[:3])}{' ...' if len(changes) > 3 else ''} "
                          f"-> velocity volume maps re-tied from {self.layers[k0][0]} in {time.perf_counter() - t0:.2f} s"}
        self.history.append(rep)
        return rep

    def update_well(self, name, td=None, min_weight=1e-3):
        """A new / edited tie of one well: its curve is recomputed and only traces where it has weight are re-integrated."""
        t0 = time.perf_counter()
        if td is not None:
            old = self.td_fn
            self.td_fn = (lambda n, _o=old, _n=name, _td=td: _td if n == _n else _o(n)) if old else \
                (lambda n, _n=name, _td=td: _td)
        xy = next(((x, y) for n, x, y in (self.wells or self.project.wells()) if n == name), None)
        if xy is None or name not in self.curves:
            raise KeyError(f"{name}: not a model well")
        tdo, src = self._td(name)
        c = WellCurve(name, xy[0], xy[1], self.vol.times, np.asarray(tdo.depth(np.asarray(self.vol.times, float))), src,
                      self.curves[name].corr)
        c.interval_velocity(self.vmin, self.vmax)
        self.curves[name] = c
        new = self._well_zone_curves(self.tg)
        self.zc_curves[name] = new[name]
        w = self.names.index(name)
        mask = self.W[:, :, w] > min_weight
        self.z_raw = self._depth_grids(mask=mask, base=self.z_raw)
        self.residuals(0)
        rep = {"changes": [f"{name} tie"], "traces": int(mask.sum()), "seconds": time.perf_counter() - t0,
               "message": f"{name} tie -> velocities re-integrated at {int(mask.sum())} traces, maps re-tied in "
                          f"{time.perf_counter() - t0:.2f} s"}
        self.history.append(rep)
        return rep

    # ------------------------------------------------------------------ QC (37g)
    def blind_test(self, progress=None, cancel=None):
        """Leave one well out: the volume (velocities AND residual correction) without the well predicts its tops."""
        rows = []
        names = [n for n in self.tops if n in self.curves]
        for q, name in enumerate(names):
            if cancel and cancel():
                raise RuntimeError("cancelled")
            m = self.clone(exclude=self.exclude | {name})
            m.curves = {n: c for n, c in self.curves.items() if n != name}
            try:
                m.build()
            except ValueError:
                continue
            xy = next(((x, y) for n, x, y in (self.wells or self.project.wells()) if n == name))
            others = [(c.x, c.y) for c in m.curves.values()]
            dmin = float(min(math.hypot(xy[0] - a, xy[1] - b) for a, b in others)) if others else float("nan")
            for k, (h, fm) in enumerate(self.layers):
                if fm not in self.tops[name]:
                    continue
                pred = self._sample_grid(m.z_corr[k], *xy)
                act = self.tops[name][fm]
                rows.append({"well": name, "horizon": h, "formation": fm, "actual": act, "predicted": pred,
                             "error": act - pred, "distance": dmin})
            if progress:
                progress((q + 1) / len(names), f"blind test {name}")
        from .scenarios import summarise
        return {"rows": rows, "summary": summarise(rows, [h for h, _ in self.layers])}

    def _sample_grid(self, g, x, y):
        return float(MapGrid(g, "ilxl", "m", "", self.vol).sample([x], [y])[0])

    def clone(self, **kw):
        d = dict(project=self.project, vol_name=self.vol_name, horizons=self.layers, wells=self.wells,
                 modes=self.modes, method=self.method, power=self.power, vrange=self.vrange,
                 residual_method=self.residual_method, nu=self.nu, vmin=self.vmin, vmax=self.vmax, trend=self.trend,
                 td_fn=self.td_fn, tops_fn=self.tops_fn, exclude=self.exclude, min_tie_corr=self.min_tie_corr)
        d.update(kw)
        return VelocityVolumeModel(**d)

    def probability_maps(self, k, bt):
        """P10 (shallow) / P50 / P90 (deep) depth grids from the blind-test error growth with distance to wells."""
        from .scenarios import distance_to_wells
        s = bt["summary"].get(self.layers[k][0]) or {}
        a, b = float(s.get("a") or 0.0), float(s.get("b") or 0.0)
        px = np.array([c.x for c in self.curves.values()])
        py = np.array([c.y for c in self.curves.values()])
        d = distance_to_wells(self.X, self.Y, px, py)
        sig = np.maximum(a + b * d, 0.0)
        p50 = self.z_corr[k]
        return {"P10": p50 - 1.2816 * sig, "P50": p50, "P90": p50 + 1.2816 * sig, "sigma": sig}

    def velocity_section(self, mode, number, calibrated=True):
        """Velocity section (n_traces, ns) on an inline / crossline (with the horizon times for overlays)."""
        vol = self.vol
        if mode == "Inline":
            i = vol.il_index(number)
            ii, jj = np.full(vol.nxl, i), np.arange(vol.nxl)
        else:
            j = vol.xl_index(number)
            ii, jj = np.arange(vol.nil), np.full(vol.nil, j)
        return self.traces(ii, jj, calibrated), self.tg[:, ii, jj]

    # ------------------------------------------------------------------ storage
    def write(self, name="Velocity volume", calibrated=True, block=16, progress=None, cancel=None):
        """Store the (calibrated) interval-velocity volume like any attribute volume (float16 store)."""
        from .inversion.common import write_volume
        vol = self.vol

        def compute(a, b):
            I, J = np.meshgrid(np.arange(a, b), np.arange(vol.nxl), indexing="ij")
            return self.traces(I.ravel(), J.ravel(), calibrated).reshape(b - a, vol.nxl, vol.ns).astype(np.float32)
        return write_volume(self.project, self.vol_name, name, compute, block, progress, cancel,
                            {"velocity_volume": {"calibrated": bool(calibrated), "horizons": [h for h, _ in self.layers]}})

    def to_dict(self):
        return {"mode": "volume", "vol": self.vol_name, "horizons": [list(x) for x in self.layers], "modes": self.modes,
                "method": self.method, "power": self.power, "vrange": self.vrange,
                "residual_method": self.residual_method, "nu": self.nu, "vmin": self.vmin, "vmax": self.vmax,
                "trend": ({k: v for k, v in self.trend.items() if k != "array"} if isinstance(self.trend, dict) and
                          self.trend.get("volume") else None),
                "exclude": sorted(self.exclude)}

    @classmethod
    def from_dict(cls, project, d, **kw):
        return cls(project, d["vol"], [tuple(x) for x in d["horizons"]], modes=d.get("modes"),
                   method=d.get("method", "idw"), power=d.get("power", 2.0), vrange=d.get("vrange"),
                   residual_method=d.get("residual_method", "tps"), nu=d.get("nu", 24), vmin=d.get("vmin", 1300.0),
                   vmax=d.get("vmax", 7000.0), trend=d.get("trend"), exclude=d.get("exclude"), **kw)


def save_scenario(project, name, model):
    """Register a velocity-volume model as a named depth scenario (item 17e list)."""
    sc = project.display.setdefault("velocity_scenarios", {})
    sc[str(name)] = model.to_dict()
    return sorted(sc)


def difference_grids(a, b):
    """Corrected depth difference a - b per horizon (two VelocityVolumeModels, or a model and a VelocityModel)."""
    out = []
    for k, (h, _) in enumerate(a.layers):
        za = a.z_corr[k]
        zb = b.z_corr[k] if hasattr(b, "z_corr") else b.depth_grid(k).z
        out.append(MapGrid(za - zb, "ilxl", "m", f"{h}: volume - other", a.vol))
    return out
