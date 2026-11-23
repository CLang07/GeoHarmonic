"""Velocity model (layer cake) and depth conversion.

Layers are bounded by time horizons (top to bottom; layer 1 starts at the datum, t = 0, z = 0 = MSL).  At every
well inside the map the layer's interval velocity is 2 dz / dt between the horizons, with dz from the linked
formation tops (TVDSS) or, for horizons without a formation, from the well's time-depth function.  Interval
velocities are gridded between wells (inverse distance; constant with one well); depth of horizon k = sum of
V_int x dt / 2 over the layers above.  Optional residual correction grids the remaining misfit at the wells.
Sections (inline / crossline / 2D line) are converted to depth trace by trace with the same model.

Velocity functions per layer (item 17a, Kingdom DDC-style; default "v0" = the item-12 behaviour):
  v0    constant interval velocity per well, gridded
  v0kz  V(z) = V0 + k z  (k fitted over the wells or fixed; V0 per well, gridded).  With one-way time tau in the layer:
        z_base = (z_top + V0/k) exp(k tau) - V0/k
  v0kt  V(t) = V0 + k t  (t one-way seconds below the datum): z_base = z_top + V0 (b - a) + k (b^2 - a^2) / 2
  td    the wells' time-depth functions (checkshot / tied sonic), blended between wells (IDW or nearest well)
Inside a layer, depth at a time follows the layer function scaled so it lands on the (possibly residual-corrected)
depth maps, so depth sections and depth maps agree.
"""
import numpy as np

from .maps import MapGrid, horizon_grid


FUNCTIONS = {"v0": "V0 (constant)", "v0kz": "V0 + kZ", "v0kt": "V0 + kT", "td": "Well T-D function"}


def _idw_nodes(X, Y, px, py, pv, power=2.0):
    """Inverse-distance blend of per-well values; a value may be a scalar or an array of X's shape."""
    if len(pv) == 1:
        return np.broadcast_to(np.asarray(pv[0], float), X.shape).copy()
    num = np.zeros(X.shape)
    den = np.zeros(X.shape)
    for x, y, v in zip(px, py, pv):
        w = 1.0 / np.maximum((X - x) ** 2 + (Y - y) ** 2, 1.0) ** (power / 2)
        num += w * v
        den += w
    return num / den


def _blend_weights(x, y, px, py, mode="idw", power=2.0):
    """(n, nw) normalised weights of the wells at points (x, y): inverse distance or nearest well."""
    d2 = (np.asarray(x, float)[:, None] - px[None, :]) ** 2 + (np.asarray(y, float)[:, None] - py[None, :]) ** 2
    if mode == "nearest" or len(px) == 1:
        w = np.zeros(d2.shape)
        w[np.arange(len(d2)), np.argmin(d2, axis=1)] = 1.0
        return w
    w = 1.0 / np.maximum(d2, 1.0) ** (power / 2)
    return w / w.sum(axis=1, keepdims=True)


def _ek(k, tau):
    """(exp(k tau) - 1) / k with the k -> 0 limit (= tau)."""
    tau = np.asarray(tau, float)
    if abs(k) < 1e-9:
        return tau * (1.0 + 0.5 * k * tau)
    return np.expm1(k * tau) / k


def kz_base(z_top, v0, k, tau):
    """V0 + kZ layer: depth of the base after one-way time tau (s) below a top at z_top."""
    return z_top + np.asarray(z_top, float) * np.expm1(k * np.asarray(tau, float)) + v0 * _ek(k, tau)


def kz_v0(z_top, z_base, k, tau):
    """V0 + kZ: V0 that takes a well from z_top to z_base in one-way time tau."""
    e = _ek(k, tau)
    return (np.asarray(z_base, float) - z_top - np.asarray(z_top, float) * np.expm1(k * np.asarray(tau, float))) / \
        np.where(np.abs(e) > 1e-12, e, np.nan)


def kt_base(z_top, v0, k, a, b):
    """V0 + kT layer (a, b = one-way times of top and base, s)."""
    return z_top + v0 * (b - a) + 0.5 * k * (b * b - a * a)


def kt_v0(z_top, z_base, k, a, b):
    return (np.asarray(z_base, float) - z_top - 0.5 * k * (b * b - a * a)) / np.where(np.abs(b - a) > 1e-12, b - a,
                                                                                       np.nan)


def fit_k_kz(z_top, tau, z_base, lo=-2.0, hi=5.0):
    """Best regional k for V0 + kZ: the k for which one V0 fits all wells best (least-squares in depth).
    For a trial k each well gives V0_i; the depth error of a common V0 is ek_i (V0 - V0_i), so the objective is the
    ek^2-weighted variance of V0_i. Coarse scan then golden-section refinement."""
    z_top, tau, z_base = (np.asarray(a, float) for a in (z_top, tau, z_base))

    def sse(k):
        e = _ek(k, tau)
        v0 = kz_v0(z_top, z_base, k, tau)
        w = e * e
        vb = np.sum(w * v0) / np.sum(w)
        return float(np.sum(w * (v0 - vb) ** 2))

    ks = np.linspace(lo, hi, 281)
    s = np.array([sse(k) for k in ks])
    i = int(np.nanargmin(s))
    a, b = ks[max(i - 1, 0)], ks[min(i + 1, len(ks) - 1)]
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = sse(c), sse(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = sse(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = sse(d)
    return 0.5 * (a + b)


def fit_k_kt(z_top, z_base, a, b):
    """V0 + kT: dz = V0 (b - a) + k (b^2 - a^2)/2 is linear in (V0, k) -> least squares. Returns (k, V0)."""
    A = np.c_[np.asarray(b, float) - a, 0.5 * (np.asarray(b, float) ** 2 - np.asarray(a, float) ** 2)]
    sol = np.linalg.lstsq(A, np.asarray(z_base, float) - z_top, rcond=None)[0]
    return float(sol[1]), float(sol[0])


def fit_line(x, y, nsigma=2.0):
    """Least-squares line y = a + b x: (b, a, r2, residuals, outliers |res| > nsigma * std)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.ptp(x) < 1e-9:
        a = float(np.mean(y)) if len(y) else np.nan
        res = y - a
        return 0.0, a, 0.0, res, np.zeros(len(x), bool)
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(res ** 2)) / sst if sst > 0 else 1.0
    sd = float(np.std(res))
    out = np.abs(res) > nsigma * sd if (len(x) >= 3 and sd > 0) else np.zeros(len(x), bool)
    return float(b), float(a), r2, res, out


class VelocityModel:
    def __init__(self, project, survey_key, layers, residual_correction=False, cell=100.0, below_velocity=None,
                 functions=None):
        """layers: [(horizon name, formation name or None)] top to bottom.
        functions: per layer None or {"func": "v0"|"v0kz"|"v0kt"|"td", "k": float|None (None = fit over the wells),
        "blend": "idw"|"nearest" (td)}; missing = "v0" (item-12 behaviour)."""
        self.project = project
        self.key = survey_key
        self.layers = list(layers)
        self.residual_correction = residual_correction
        self.cell = cell
        self.below_velocity = below_velocity
        fs = list(functions or [])
        self.functions = [dict(fs[i]) if i < len(fs) and fs[i] else {"func": "v0"} for i in range(len(self.layers))]
        for f in self.functions:
            if f.get("func", "v0") not in FUNCTIONS:
                raise ValueError(f"unknown velocity function {f.get('func')!r}")
            f.setdefault("func", "v0")
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
    def _layer_data(self, wt, k):
        """Per-well (z_top, z_base, t_top, t_base) of layer k (layer 0 starts at the datum, t = z = 0)."""
        zt = np.array([r["z"][k - 1] if k else 0.0 for r in wt])
        zb = np.array([r["z"][k] for r in wt])
        tt = np.array([r["t"][k - 1] if k else 0.0 for r in wt])
        tb = np.array([r["t"][k] for r in wt])
        return zt, zb, tt, tb

    def _layer_k(self, wt, k):
        """(k, source) for a V0 + kZ / V0 + kT layer: fixed by the user or fitted over the wells."""
        f = self.functions[k]
        if f.get("k") is not None:
            return float(f["k"]), "fixed"
        zt, zb, tt, tb = self._layer_data(wt, k)
        ok = np.isfinite(zt) & np.isfinite(zb) & (tb - tt > 1e-6)
        zmid = 0.5 * (zt + zb)
        if ok.sum() < 2 or np.ptp(zmid[ok]) < 1.0:
            return 0.0, "k = 0 (fit needs 2+ wells at different depths)"
        if f["func"] == "v0kz":
            return fit_k_kz(zt[ok], (tb[ok] - tt[ok]) / 2000.0, zb[ok]), f"fit over {int(ok.sum())} wells"
        kk, _ = fit_k_kt(zt[ok], zb[ok], tt[ok] / 2000.0, tb[ok] / 2000.0)
        return kk, f"fit over {int(ok.sum())} wells"

    def _td_curves(self, wt):
        ws = self.project.wellset()
        return [ws.td_for(r["well"]) for r in wt]

    def build(self):
        tg = self.time_grids()
        wt = self.well_table()
        if not wt:
            raise ValueError("no well inside the map has values for all horizons")
        X, Y = tg[0].node_xy()
        px = np.array([r["x"] for r in wt])
        py = np.array([r["y"] for r in wt])
        vgrids, zgrids, v0grids, info = [], [], [], []
        z_prev = np.zeros(X.shape)
        t_prev = np.zeros(X.shape)
        tds = None
        for k, g in enumerate(tg):
            f = self.functions[k]
            func = f["func"]
            dt = np.maximum(g.z - t_prev, 0.0)
            zt_w, zb_w, tt_w, tb_w = self._layer_data(wt, k)
            v0 = None
            lay = {"func": func, "k": None, "k_source": ""}
            if func == "v0":
                v = _idw_nodes(X, Y, px, py, np.array([r["vint"][k] for r in wt]))
                z = z_prev + v * dt / 2000.0
            elif func in ("v0kz", "v0kt"):
                kk, src = self._layer_k(wt, k)
                lay.update(k=kk, k_source=src)
                if func == "v0kz":
                    v0w = kz_v0(zt_w, zb_w, kk, (tb_w - tt_w) / 2000.0)
                else:
                    v0w = kt_v0(zt_w, zb_w, kk, tt_w / 2000.0, tb_w / 2000.0)
                ok = np.isfinite(v0w)
                if not ok.any():
                    raise ValueError(f"layer {self.layers[k][0]}: no well with a usable thickness for V0")
                v0 = _idw_nodes(X, Y, px[ok], py[ok], v0w[ok])
                if func == "v0kz":
                    z = kz_base(z_prev, v0, kk, dt / 2000.0)
                    v_top = v0 + kk * z_prev
                else:
                    a = t_prev / 2000.0
                    z = kt_base(z_prev, v0, kk, a, a + dt / 2000.0)
                    v_top = v0 + kk * a
                with np.errstate(invalid="ignore", divide="ignore"):
                    v = np.where(dt > 1e-6, 2000.0 * (z - z_prev) / np.where(dt > 1e-6, dt, 1.0), v_top)
            else:  # td: thickness from the wells' time-depth functions, blended
                if tds is None:
                    tds = self._td_curves(wt)
                mode = f.get("blend", "idw")
                fin = np.isfinite(g.z) & np.isfinite(t_prev)
                tb_safe = np.where(fin, t_prev + dt, 0.0)
                tt_safe = np.where(fin, t_prev, 0.0)
                if mode == "nearest":
                    w = _blend_weights(X.ravel(), Y.ravel(), px, py, "nearest")
                    dz = np.zeros(X.size)
                    for i, td in enumerate(tds):
                        sel = w[:, i] > 0
                        if sel.any():
                            dz[sel] = np.asarray(td.depth(tb_safe.ravel()[sel])) - np.asarray(td.depth(tt_safe.ravel()[sel]))
                    dz = dz.reshape(X.shape)
                else:
                    dz = _idw_nodes(X, Y, px, py, [np.asarray(td.depth(tb_safe)) - np.asarray(td.depth(tt_safe))
                                                   for td in tds])
                z = np.where(fin, z_prev + np.maximum(dz, 0.0), np.nan)
                with np.errstate(invalid="ignore", divide="ignore"):
                    v = np.where(dt > 1e-6, 2000.0 * (z - z_prev) / np.where(dt > 1e-6, dt, 1.0), np.nan)
                lay["blend"] = mode
            vgrids.append(v)
            zgrids.append(z)
            v0grids.append(v0)
            info.append(lay)
            z_prev, t_prev = z, g.z
        if self.residual_correction:
            for k in range(len(tg)):
                dg = MapGrid(zgrids[k], tg[k].kind, "m", "", tg[k].vol, tg[k].x0, tg[k].y0, tg[k].dx, tg[k].dy)
                res = np.array([r["z"][k] for r in wt]) - dg.sample(px, py)
                ok = np.isfinite(res)
                if ok.any():
                    zgrids[k] = zgrids[k] + _idw_nodes(X, Y, px[ok], py[ok], res[ok])
        self._grids = {"t": tg, "v": vgrids, "z": zgrids, "v0": v0grids, "wells": wt, "layers": info,
                       "td": tds, "px": px, "py": py}
        return self

    def layer_info(self):
        """Per layer: {func, k, k_source[, blend]} as used by the last build."""
        return self.grids()["layers"]

    def crossplot(self, k):
        """Cross-plot data of layer k for the dialog: interval velocity at the wells vs mid-layer depth (V0 / V0 + kZ /
        T-D) or vs mid-layer one-way time (V0 + kT), least-squares line, R^2, outliers (> 2 sigma), and the k / V0 the
        model uses."""
        g = self.grids()
        wt = g["wells"]
        zt, zb, tt, tb = self._layer_data(wt, k)
        vint = np.array([r["vint"][k] for r in wt])
        func = self.functions[k]["func"]
        if func == "v0kt":
            x, xlabel = (tt + tb) / 4000.0, "mid-layer one-way time (s)"
        else:
            x, xlabel = 0.5 * (zt + zb), "mid-layer depth TVDSS (m)"
        ok = np.isfinite(x) & np.isfinite(vint)
        slope, icpt, r2, res, out = fit_line(x[ok], vint[ok])
        outl = np.zeros(len(x), bool)
        outl[np.flatnonzero(ok)] = out
        lay = g["layers"][k]
        v0w = None
        if func == "v0kz":
            v0w = kz_v0(zt, zb, lay["k"], (tb - tt) / 2000.0)
        elif func == "v0kt":
            v0w = kt_v0(zt, zb, lay["k"], tt / 2000.0, tb / 2000.0)
        return {"layer": self.layers[k][0], "func": func, "wells": [r["well"] for r in wt], "x": x, "vint": vint,
                "xlabel": xlabel, "slope": slope, "intercept": icpt, "r2": r2, "outlier": outl,
                "k": lay.get("k"), "k_source": lay.get("k_source", ""), "v0_wells": v0w}

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

    def v0_grid(self, k):
        """Gridded V0 of a V0 + kZ / V0 + kT layer (None for other functions)."""
        g = self.grids()
        if g["v0"][k] is None:
            return None
        t = g["t"][k]
        return MapGrid(g["v0"][k], t.kind, "m/s", f"V0 of {self.layers[k][0]}", t.vol, t.x0, t.y0, t.dx, t.dy)

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

    def _sample_layers(self, key, x, y):
        g = self.grids()
        return np.array([MapGrid(a, t.kind, "", "", t.vol, t.x0, t.y0, t.dx, t.dy).sample(x, y)
                         if a is not None else np.full(len(x), np.nan) for a, t in zip(g[key], g["t"])]).T

    def depth_of_time(self, x, y, times):
        """Depth (m TVDSS) of every time sample at map points: (n, ns). Horizon times land on the depth maps; inside a
        layer depth follows the layer's velocity function, scaled to the map depths; below the last horizon the last
        interval velocity (or below_velocity; V0 + kZ / V0 + kT: the instantaneous velocity at the base)."""
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        g = self.grids()
        T, V = self.trace_layers(x, y)
        Z = self._sample_layers("z", x, y)
        V0 = self._sample_layers("v0", x, y)
        n, K = T.shape
        times = np.asarray(times, float)
        tk = np.maximum.accumulate(np.c_[np.zeros(n), T], axis=1)
        zk = np.c_[np.zeros(n), Z]
        bad = ~(np.all(np.isfinite(tk), axis=1) & np.all(np.isfinite(zk), axis=1) & np.all(np.isfinite(V), axis=1))
        tk = np.where(bad[:, None], 0.0, tk)
        zk = np.where(bad[:, None], 0.0, zk)
        # layer index of every sample: 0..K-1 inside, K below the last horizon
        layer = np.zeros((n, len(times)), np.int32)
        for k in range(1, K + 1):
            layer += times[None, :] >= tk[:, k:k + 1]
        layer = np.where(times[None, :] < 0, 0, layer)
        out = np.zeros((n, len(times)))
        tt = np.broadcast_to(times[None, :], out.shape)
        tdw = None
        for k in range(K + 1):
            m = layer == k
            if not m.any():
                continue
            rows = np.nonzero(m)[0]
            t = tt[m]
            t0, z0 = tk[rows, k], zk[rows, k]
            if k == K:  # below the deepest horizon
                lay = g["layers"][K - 1]
                if self.below_velocity is not None:
                    vb = np.full(len(rows), float(self.below_velocity))
                elif lay["func"] == "v0kz":
                    vb = V0[rows, K - 1] + lay["k"] * z0
                elif lay["func"] == "v0kt":
                    vb = V0[rows, K - 1] + lay["k"] * t0 / 2000.0
                else:
                    vb = V[rows, K - 1]
                vb = np.where(np.isfinite(vb), vb, V[rows, K - 1])
                out[m] = z0 + vb * (t - t0) / 2000.0
                continue
            t1, z1 = tk[rows, k + 1], zk[rows, k + 1]
            lay = g["layers"][k]
            func = lay["func"]
            if func == "v0kz":
                kk, v0 = lay["k"], V0[rows, k]
                full = kz_base(z0, v0, kk, (t1 - t0) / 2000.0) - z0
                part = kz_base(z0, v0, kk, (t - t0) / 2000.0) - z0
            elif func == "v0kt":
                kk, v0 = lay["k"], V0[rows, k]
                full = kt_base(z0, v0, kk, t0 / 2000.0, t1 / 2000.0) - z0
                part = kt_base(z0, v0, kk, t0 / 2000.0, t / 2000.0) - z0
            elif func == "td":
                if tdw is None:
                    tdw = _blend_weights(x, y, g["px"], g["py"], lay.get("blend", "idw"))
                w = tdw[rows]
                full = np.zeros(len(rows))
                part = np.zeros(len(rows))
                for i, td in enumerate(g["td"]):
                    wi = w[:, i]
                    if not wi.any():
                        continue
                    d0 = np.asarray(td.depth(t0))
                    full += wi * (np.asarray(td.depth(t1)) - d0)
                    part += wi * (np.asarray(td.depth(t)) - d0)
            else:
                full, part = t1 - t0, t - t0
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(np.abs(full) > 1e-9, part / np.where(np.abs(full) > 1e-9, full, 1.0), 0.0)
            # a zero-thickness layer has no shape: fall back to its interval velocity
            lin = z0 + V[rows, k] * (t - t0) / 2000.0
            out[m] = np.where(np.abs(full) > 1e-9, z0 + (z1 - z0) * frac, lin)
        out[bad] = np.nan
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
        d = {"survey": self.key, "layers": [list(l) for l in self.layers], "residual_correction":
             self.residual_correction, "cell": self.cell, "below_velocity": self.below_velocity}
        if any(f.get("func", "v0") != "v0" for f in self.functions):
            d["functions"] = [dict(f) for f in self.functions]
        return d

    @classmethod
    def from_dict(cls, project, d):
        return cls(project, d["survey"], [tuple(l) for l in d["layers"]], d.get("residual_correction", False),
                   d.get("cell", 100.0), d.get("below_velocity"), d.get("functions"))


def _monotonic_depths(zt):
    """Depth must increase with time along each trace (a negative interval velocity from inconsistent tops would
    fold the section): running maximum per trace, NaN kept."""
    zt = np.array(zt, float)
    fin = np.isfinite(zt)
    z = np.where(fin, zt, -np.inf)
    z = np.maximum.accumulate(z, axis=-1)
    return np.where(fin, z, np.nan)
