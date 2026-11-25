"""Live gridding for the Live Map (PLAN 31g) - Qt-free.

Picks (horizon) and tops (a WellTable column such as 'Top F0' pushed by the Well Correlation Tool) feed a gridder
that runs in a background thread: a COARSE grid right after every change (cell x coarse_factor, while picking) and the
FULL grid once the changes pause (`pause` seconds).  Fault polygons / traces act as barriers (the item-21f
`carto.grid_by_blocks`: wells only influence their own fault block).  Results replace a 'Live: <name>' grid layer
(+ contours, colour fill) in the hub LayerStack; the depth map of the dynamic depth model (item 17) is refreshed the
same way when a 'velocities' DataChanged event arrives (`depth_grids`).  Grid math (isochore, add / subtract, clip
by polygon) is item 21c `propgrid.calc` / `mask_polygon`; `isochore` below is a shortcut.

    lg = LiveGridder(stack, bus.table, project)
    lg.add_source("F0 tops", "column", column="Top F0", cell=100, method="tps", barriers=True)
    lg.changed("F0 tops")        # -> coarse job started now
    lg.tick()                    # call from a timer: starts the full grid after the pause, collects finished jobs
"""
import threading
import time

import numpy as np

from .maps import MapGrid
from . import vgrid


class LiveGridder:
    def __init__(self, stack, table, project=None, pause=1.5, coarse_factor=4, clock=time.monotonic, threaded=True):
        self.stack = stack
        self.table = table
        self.project = project if project is not None else getattr(stack, "project", None)
        self.pause = float(pause)
        self.coarse_factor = max(1, int(coarse_factor))
        self.clock = clock
        self.threaded = threaded
        self.sources = {}
        self._dirty = {}  # name -> time of the last change
        self._gen = {}  # name -> generation (stale results are dropped)
        self._jobs = {}  # name -> (gen, level, thread, box)
        self.done = []  # (name, level, seconds) of applied grids, newest last
        self.errors = []

    # ---------------------------------------------------------------- sources
    def add_source(self, name, kind="column", column=None, horizon=None, key=None, cell=None, method="idw",
                   barriers=False, interval=None, ramp="depth", extent=None):
        """kind 'column' (WellTable column, e.g. 'Top F0') or 'horizon' (picks of a horizon on key '3D:<vol>' / '2D')."""
        self.sources[name] = {"kind": kind, "column": column, "horizon": horizon, "key": key, "cell": cell,
                              "method": method, "barriers": barriers, "interval": interval, "ramp": ramp,
                              "extent": extent, "lid": None, "cid": None}
        self._gen[name] = 0
        return self.run(name, "full", threaded=False)

    def remove_source(self, name):
        src = self.sources.pop(name, None)
        if src and src.get("lid"):
            try:
                self.stack.remove(src["lid"])
            except Exception:
                pass
        self._dirty.pop(name, None)
        return src

    # ---------------------------------------------------------------- data
    def points(self, name):
        s = self.sources[name]
        if s["kind"] == "column":
            x, y = self.table.xy()
            v = self.table.numeric(s["column"])
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
            return x[ok], y[ok], v[ok]
        h = self.project.horizons.get(s["horizon"])
        if h is None:
            raise KeyError(f"no horizon {s['horizon']!r}")
        xs, ys, vs = [], [], []
        for k, a in h.data.items():
            if s.get("key") and k != s["key"] and not (s["key"] == "2D" and k.startswith("2D:")):
                continue
            a = np.asarray(a, float)
            if k.startswith("3D:"):
                vol = self.project.volume(k[3:])
                I, J = np.nonzero(np.isfinite(a))
                step = max(1, int(np.ceil(np.sqrt(len(I) / 4000.0)))) if len(I) else 1
                sel = (I % step == 0) & (J % step == 0)
                I, J = I[sel], J[sel]
                X, Y = vol.xy(vol.inlines[I], vol.xlines[J])
                xs.append(np.ravel(X))
                ys.append(np.ravel(Y))
                vs.append(a[I, J])
            elif k.startswith("2D:"):
                line = self.project.line(k[3:])
                if not line.has_coords:
                    continue
                ok = np.isfinite(a)
                step = max(1, int(ok.sum() // 1000))
                xs.append(np.asarray(line.x, float)[ok][::step])
                ys.append(np.asarray(line.y, float)[ok][::step])
                vs.append(a[ok][::step])
        if not vs:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(vs)

    def barrier_lines(self):
        """Fault traces / polygons from the map hub: polygons layers styled as faults + vector layers named fault."""
        out = []
        for l in self.stack.layers if self.stack is not None else []:
            if l["kind"] == "polygons" and l["style"].get("fault"):
                for p in l["src"].get("polygons", []):
                    xs, ys = list(map(float, p["x"])), list(map(float, p["y"]))
                    if len(xs) >= 2:
                        if len(xs) >= 3 and (xs[0], ys[0]) != (xs[-1], ys[-1]):
                            xs, ys = xs + xs[:1], ys + ys[:1]
                        out.append((np.array(xs), np.array(ys)))
            elif l["kind"] == "vector" and "fault" in l["name"].lower():
                try:
                    from .gis.vector import layer_of
                    vl = layer_of(self.stack, l)
                    for g in vl.geoms if vl is not None else []:
                        for part in (g or {}).get("parts", []):
                            if len(part) >= 2:
                                out.append((part[:, 0].copy(), part[:, 1].copy()))
                except Exception as e:
                    self.errors.append(repr(e))
        return out

    def geometry(self, name, px, py, level):
        s = self.sources[name]
        if s.get("extent"):
            x0, x1, y0, y1 = s["extent"]
        else:
            x0, x1, y0, y1 = float(px.min()), float(px.max()), float(py.min()), float(py.max())
            pad = 0.1 * max(x1 - x0, y1 - y0, 1.0)
            x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
        cell = float(s["cell"] or max(x1 - x0, y1 - y0) / 100.0)
        if level == "coarse":
            cell *= self.coarse_factor
        nx = max(2, int(np.ceil((x1 - x0) / cell)) + 1)
        ny = max(2, int(np.ceil((y1 - y0) / cell)) + 1)
        while nx * ny > 250_000:  # same cap as the hub grids
            cell *= 1.5
            nx = max(2, int(np.ceil((x1 - x0) / cell)) + 1)
            ny = max(2, int(np.ceil((y1 - y0) / cell)) + 1)
        return x0, y0, cell, nx, ny

    def compute(self, name, level="full", barriers=None):
        """The grid itself (pure function of the inputs; runs in the worker thread)."""
        s = self.sources[name]
        t0 = time.perf_counter()
        px, py, v = self.points(name)
        if len(v) == 0:
            raise ValueError(f"{name}: no data points to grid")
        x0, y0, cell, nx, ny = self.geometry(name, px, py, level)
        X, Y = np.meshgrid(x0 + cell * np.arange(nx), y0 + cell * np.arange(ny))
        method = s["method"] if len(v) >= 3 else "idw"
        if len(v) > 600 and method in ("tps", "kriging"):
            method = "idw"  # dense picks: the dense solvers would be slow; IDW is exact enough between picks
        fn = lambda XX, YY, ax, ay, av: vgrid.grid_values(XX, YY, ax, ay, av, method)  # noqa: E731
        bl = barriers if barriers is not None else (self.barrier_lines() if s["barriers"] else [])
        if bl:
            from .welltie import wc
            wc()
            from well_correlation.carto import grid_by_blocks
            g = grid_by_blocks(px, py, v, {"z": np.zeros((ny, nx)), "x0": x0, "y0": y0, "dx": cell, "dy": cell}, bl,
                               fn)
            z = g["z"]
        else:
            z = np.asarray(fn(np.ravel(X), np.ravel(Y), px, py, v), float).reshape(ny, nx)
        unit = "ms" if s["kind"] == "horizon" else (self.table.meta.get(s.get("column"), {}).get("unit") or "")
        grid = MapGrid(z, "xy", unit, f"Live: {name}", None, x0, y0, cell, cell)
        grid.level = level
        grid.n_points = len(v)
        grid.seconds = time.perf_counter() - t0
        return grid

    # ---------------------------------------------------------------- scheduling
    def changed(self, name=None):
        """Data of a source (None = every source) changed: start a coarse grid now, full grid after the pause."""
        names = [name] if name else list(self.sources)
        now = self.clock()
        started = []
        for n in names:
            if n not in self.sources:
                continue
            self._dirty[n] = now
            started.append(self.run(n, "coarse"))
        return started

    def changed_column(self, columns):
        """WellTable columns changed (tops pushed): which sources depend on them -> changed()."""
        cols = set(columns or [])
        hit = [n for n, s in self.sources.items() if s["kind"] == "column" and (not cols or s["column"] in cols)]
        for n in hit:
            self.changed(n)
        return hit

    def changed_horizon(self, names=None):
        hit = [n for n, s in self.sources.items() if s["kind"] == "horizon" and (names is None or s["horizon"] in names)]
        for n in hit:
            self.changed(n)
        return hit

    def tick(self):
        """Timer callback: apply finished jobs; start full grids for sources quiet for `pause` s.  Returns names
        whose layer changed."""
        applied = self.collect()
        now = self.clock()
        for n, t in list(self._dirty.items()):
            if now - t >= self.pause and n not in self._jobs:
                del self._dirty[n]
                self.run(n, "full")
        return applied + (self.collect() if not self.threaded else [])

    def run(self, name, level="full", threaded=None):
        threaded = self.threaded if threaded is None else threaded
        self._gen[name] = self._gen.get(name, 0) + 1
        gen = self._gen[name]
        bl = self.barrier_lines() if self.sources[name]["barriers"] else []  # stack is read in the GUI thread
        box = {}
        if not threaded:
            try:
                box["grid"] = self.compute(name, level, bl)
            except Exception as e:
                box["error"] = e
            self._jobs[name] = (gen, level, None, box)
            self.collect()
            return name

        def work():
            try:
                box["grid"] = self.compute(name, level, bl)
            except Exception as e:  # reported by collect()
                box["error"] = e
        th = threading.Thread(target=work, daemon=True)
        self._jobs[name] = (gen, level, th, box)
        th.start()
        return name

    def busy(self):
        return [n for n, j in self._jobs.items() if j[2] is not None and j[2].is_alive()]

    def wait(self, timeout=30.0):
        for n, (gen, level, th, box) in list(self._jobs.items()):
            if th is not None:
                th.join(timeout)
        return self.collect()

    def collect(self):
        out = []
        for n, (gen, level, th, box) in list(self._jobs.items()):
            if th is not None and th.is_alive():
                continue
            del self._jobs[n]
            if gen != self._gen.get(n) or n not in self.sources:
                continue  # superseded by a newer change
            if "error" in box:
                self.errors.append(f"{n}: {box['error']}")
                continue
            self._apply(n, box["grid"])
            self.done.append((n, level, box["grid"].seconds))
            del self.done[:-50]
            out.append(n)
        return out

    def _apply(self, name, grid):
        """Replace the source's grid layer data in place (keeps the layer, its style, order and contours)."""
        s = self.sources[name]
        st = self.stack
        if s["lid"] is None or st.get(s["lid"]) is None:
            lid, cid = st.add_grid(grid, f"Live: {name}", style={"ramp": s.get("ramp") or "depth"},
                                   contours=True, interval=s.get("interval"))
            s["lid"], s["cid"] = lid, cid
        else:
            lid = s["lid"]
            st._grids[lid] = grid
            if st.project is not None and getattr(st.project, "path", None):
                from .maplayers import save_grid
                st.get(lid)["src"]["file"] = save_grid(st.project, grid, lid)
        st.get(lid)["src"]["live"] = {"name": name, "level": grid.level, "points": grid.n_points}
        return lid


def depth_grids(stack, model, horizons=None, prefix="Live depth: "):
    """Add / refresh one grid layer per horizon of a depth model (VelocityModel or DynamicDepthModel) - called on a
    'velocities' DataChanged event so the depth maps follow the dynamic depth conversion.  Returns layer ids."""
    vm = getattr(model, "vm", None) or getattr(model, "model", None) or model  # DynamicDepthModel.vm
    out = []
    names = [ly[0] for ly in vm.layers]
    for k, hname in enumerate(names):
        if horizons is not None and hname not in horizons:
            continue
        g = vm.depth_grid(k)
        name = f"{prefix}{hname}"
        lay = next((l for l in stack.layers if l["kind"] == "grid" and l["name"] == name), None)
        if lay is None:
            lid, _ = stack.add_grid(g, name, style={"ramp": "depth"}, contours=True)
        else:
            lid = lay["id"]
            stack._grids[lid] = g
            if stack.project is not None and getattr(stack.project, "path", None):
                from .maplayers import save_grid
                lay["src"]["file"] = save_grid(stack.project, g, lid)
        out.append(lid)
    return out


def isochore(top, base, label=None):
    """Thickness grid base - top on the top grid's geometry (both MapGrids, e.g. two depth maps)."""
    from .propgrid import calc
    g = calc("-A + B", {"A": top, "B": base})  # A first: the top grid sets the geometry
    g.label = label or f"{base.label} - {top.label}"
    return g
