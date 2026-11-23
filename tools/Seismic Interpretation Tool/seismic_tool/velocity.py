"""Velocity model (layer cake) and depth conversion.

Layers are bounded by time horizons (top to bottom; layer 1 starts at the datum, t = 0, z = 0 = MSL).  At every
well inside the map the layer's interval velocity is 2 dz / dt between the horizons, with dz from the linked
formation tops (TVDSS) or, for horizons without a formation, from the well's time-depth function.  Interval
velocities are gridded between wells (inverse distance; constant with one well); depth of horizon k = sum of
V_int x dt / 2 over the layers above.  Optional residual correction grids the remaining misfit at the wells.
Sections (inline / crossline / 2D line) are converted to depth trace by trace with the same model.
"""
import numpy as np

from .maps import MapGrid, horizon_grid


def _idw_nodes(X, Y, px, py, pv, power=2.0):
    if len(pv) == 1:
        return np.full(X.shape, float(pv[0]))
    num = np.zeros(X.shape)
    den = np.zeros(X.shape)
    for x, y, v in zip(px, py, pv):
        w = 1.0 / np.maximum((X - x) ** 2 + (Y - y) ** 2, 1.0) ** (power / 2)
        num += w * v
        den += w
    return num / den


class VelocityModel:
    def __init__(self, project, survey_key, layers, residual_correction=False, cell=100.0, below_velocity=None):
        """layers: [(horizon name, formation name or None)] top to bottom."""
        self.project = project
        self.key = survey_key
        self.layers = list(layers)
        self.residual_correction = residual_correction
        self.cell = cell
        self.below_velocity = below_velocity
        self._grids = None

    # -------------------------------------------------------------- inputs
    def time_grids(self):
        hs = self.project.horizons
        grids = [horizon_grid(self.project, hs.get(n), self.key, self.cell) for n, _ in self.layers]
        ref = grids[0]
        if ref.kind == "xy":  # 2D: put every horizon on the first one's grid
            X, Y = ref.node_xy()
            grids = [ref] + [MapGrid(g.sample(X.ravel(), Y.ravel()).reshape(X.shape), "xy", "ms", g.label, None,
                                     ref.x0, ref.y0, ref.dx, ref.dy) for g in grids[1:]]
        return grids

    def well_table(self):
        """[{well, x, y, t: [..], z: [..], vint: [..], source: [..]}] for wells inside the map."""
        ws = self.project.wellset()
        tg = self.time_grids()
        rows = []
        for n in ws.names:
            w = ws.well(n)
            if w.x is None:
                continue
            ts = [float(g.sample([w.x], [w.y])[0]) for g in tg]
            if not all(np.isfinite(ts)):
                continue
            zs, src = [], []
            tops = (ws.wcp.tops or {}).get(n, {})
            for (hname, fm), t in zip(self.layers, ts):
                if fm and tops.get(fm) is not None:
                    try:
                        ws.wcp.load_well(n)
                    except Exception:
                        pass
                    zs.append(float(w.md_to_tvdss(tops[fm])))
                    src.append(f"top {fm}")
                else:
                    zs.append(float(ws.td_for(n).depth(t)))
                    src.append("time-depth")
            t_prev, z_prev = 0.0, 0.0
            vint = []
            for t, z in zip(ts, zs):
                vint.append(2000.0 * (z - z_prev) / max(t - t_prev, 1e-6))
                t_prev, z_prev = t, z
            rows.append({"well": n, "x": w.x, "y": w.y, "t": ts, "z": zs, "vint": vint, "source": src})
        return rows

    # -------------------------------------------------------------- model
    def build(self):
        tg = self.time_grids()
        wt = self.well_table()
        if not wt:
            raise ValueError("no well inside the map has values for all horizons")
        X, Y = tg[0].node_xy()
        px = np.array([r["x"] for r in wt])
        py = np.array([r["y"] for r in wt])
        vgrids, zgrids = [], []
        z_prev = np.zeros(X.shape)
        t_prev = np.zeros(X.shape)
        for k, g in enumerate(tg):
            v = _idw_nodes(X, Y, px, py, np.array([r["vint"][k] for r in wt]))
            vgrids.append(v)
            dt = np.maximum(g.z - t_prev, 0.0)
            z = z_prev + v * dt / 2000.0
            zgrids.append(z)
            z_prev, t_prev = z, g.z
        if self.residual_correction:
            for k in range(len(tg)):
                dg = MapGrid(zgrids[k], tg[k].kind, "m", "", tg[k].vol, tg[k].x0, tg[k].y0, tg[k].dx, tg[k].dy)
                res = np.array([r["z"][k] for r in wt]) - dg.sample(px, py)
                ok = np.isfinite(res)
                if ok.any():
                    zgrids[k] = zgrids[k] + _idw_nodes(X, Y, px[ok], py[ok], res[ok])
        self._grids = {"t": tg, "v": vgrids, "z": zgrids, "wells": wt}
        return self

    def grids(self):
        if self._grids is None:
            self.build()
        return self._grids

    def depth_grid(self, k):
        g = self.grids()
        t = g["t"][k]
        return MapGrid(g["z"][k], t.kind, "m", f"{self.layers[k][0]} depth (TVDSS)", t.vol, t.x0, t.y0, t.dx, t.dy)

    def velocity_grid(self, k):
        g = self.grids()
        t = g["t"][k]
        return MapGrid(g["v"][k], t.kind, "m/s", f"Interval velocity above {self.layers[k][0]}", t.vol, t.x0, t.y0,
                       t.dx, t.dy)

    def misfit_table(self):
        """[(well, horizon, formation, z_well, z_model, residual)] - residual = well - model (m)."""
        g = self.grids()
        out = []
        for r in g["wells"]:
            for k, (hname, fm) in enumerate(self.layers):
                zm = float(self.depth_grid(k).sample([r["x"]], [r["y"]])[0])
                out.append((r["well"], hname, fm or "", r["z"][k], zm, r["z"][k] - zm))
        return out

    # -------------------------------------------------------------- depth conversion of traces
    def trace_layers(self, x, y):
        """(times (n, K), interval velocities (n, K)) at map points."""
        g = self.grids()
        T = np.array([MapGrid(t.z, t.kind, "", "", t.vol, t.x0, t.y0, t.dx, t.dy).sample(x, y) for t in g["t"]]).T
        V = np.array([MapGrid(v, t.kind, "", "", t.vol, t.x0, t.y0, t.dx, t.dy).sample(x, y)
                      for v, t in zip(g["v"], g["t"])]).T
        return T, V

    def depth_of_time(self, x, y, times):
        """Depth (m TVDSS) of every time sample at map points: (n, ns)."""
        T, V = self.trace_layers(np.atleast_1d(x), np.atleast_1d(y))
        n = T.shape[0]
        times = np.asarray(times, float)
        out = np.zeros((n, len(times)))
        for i in range(n):
            tk = np.r_[0.0, T[i]]
            vk = np.r_[V[i], V[i][-1] if self.below_velocity is None else self.below_velocity]
            if not np.all(np.isfinite(tk)) or not np.all(np.isfinite(vk)):
                out[i] = np.nan
                continue
            tk = np.maximum.accumulate(tk)
            zk = np.r_[0.0, np.cumsum(vk[:-1] * np.diff(tk) / 2000.0)]
            layer = np.clip(np.searchsorted(tk, times, "right") - 1, 0, len(vk) - 1)
            out[i] = zk[np.minimum(layer, len(zk) - 1)] + vk[layer] * (times - tk[np.minimum(layer, len(tk) - 1)]) / 2000.0
        return out

    def section_to_depth(self, arr, x, y, times, dz=None, zmax=None):
        """arr (ns, n) time section at map points (x, y) -> (depth section (nz, n), z axis)."""
        zt = self.depth_of_time(x, y, times)  # (n, ns)
        zt = _monotonic_depths(zt)
        if not np.isfinite(zt).any():  # section outside the model
            return np.full((2, arr.shape[1]), np.nan, np.float32), np.array([0.0, 1.0])
        dz = dz or max(1.0, float(np.nanmedian(np.diff(zt, axis=1))))
        zmax = zmax or float(np.nanmax(zt))
        z = np.arange(0.0, zmax + dz / 2, dz)
        out = np.full((len(z), arr.shape[1]), np.nan, np.float32)
        for j in range(arr.shape[1]):
            if np.all(np.isfinite(zt[j])):
                out[:, j] = np.interp(z, zt[j], arr[:, j], left=np.nan, right=np.nan)
        return out, z

    # -------------------------------------------------------------- persistence
    def to_dict(self):
        return {"survey": self.key, "layers": [list(l) for l in self.layers], "residual_correction":
                self.residual_correction, "cell": self.cell, "below_velocity": self.below_velocity}

    @classmethod
    def from_dict(cls, project, d):
        return cls(project, d["survey"], [tuple(l) for l in d["layers"]], d.get("residual_correction", False),
                   d.get("cell", 100.0), d.get("below_velocity"))


def _monotonic_depths(zt):
    """Depth must increase with time along each trace (a negative interval velocity from inconsistent tops would
    fold the section): running maximum per trace, NaN kept."""
    zt = np.array(zt, float)
    fin = np.isfinite(zt)
    z = np.where(fin, zt, -np.inf)
    z = np.maximum.accumulate(z, axis=-1)
    return np.where(fin, z, np.nan)
