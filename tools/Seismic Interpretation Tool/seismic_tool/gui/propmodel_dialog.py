"""Property modelling + volumetrics dialog (PLAN 36) on the 3D grid of the framework dialog: upscale well logs,
variogram, facies (SIS) + porosity / Sw (SGS) realizations, volumetrics per zone and fault block (P90 / P50 / P10),
per-block totals to the Live Map, GRDECL / CSV.  Logic in plain methods (tested under dev/fakeqt)."""
import numpy as np
from PySide6 import QtWidgets

from .. import framework as FW
from .. import propmodel as PM


class PropertyDialog(QtWidgets.QDialog):
    def __init__(self, project, grid, parent=None, bus=None, stack=None):
        super().__init__(parent)
        self.setWindowTitle("Property modelling + volumetrics")
        self.project, self.grid, self.bus, self.stack = project, grid, bus, stack
        self.cells = {}  # property -> upscaled (nz, ny, nx) with NaN
        self.vario = PM.Vario(1.0, 10, 10, 3)
        self.realizations = {}
        self.rows = []
        self.stats = {}
        self.message = ""
        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("Upscale logs", lambda *_: self.upscale_project("PHIE")),
                         ("Variogram", lambda *_: self.variogram("PHIE")),
                         ("Simulate (10 realizations)", lambda *_: self.simulate("PHIE", 10)),
                         ("Volumetrics", lambda *_: self.volumetrics()),
                         ("Export GRDECL / CSV...", lambda *_: self._export_dialog())):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)

    def _say(self, msg):
        self.message = msg
        try:
            self.info.setText(msg)
        except Exception:
            pass

    # ------------------------------------------------------------------ 36a
    def upscale(self, prop, wells, method="arithmetic", discrete=False):
        """wells: [(x (n,), y (n,), z TVDSS (n,), values (n,))] -> upscaled property cells."""
        acc = {}
        for x, y, z, v in wells:
            c, _ = PM.upscale_well(self.grid, x, y, z, v, method, discrete)
            acc.update(c)
        self.cells[prop] = PM.cells_to_array(self.grid, acc)
        self._say(f"{prop}: {len(acc)} cells from {len(wells)} wells")
        return self.cells[prop]

    def project_wells(self, curve):
        """(x, y, z, values) per well of the linked well project: petro curves (or raw logs), trajectory MD -> x, y, z."""
        from ..laterals import trajectories_from_wellset
        ws = self.project.wellset()
        trajs = {t.name: t for t in trajectories_from_wellset(ws)}
        out = []
        for name in ws.names:
            w = ws.well(name)
            try:
                ws.wcp.load_well(name)
            except Exception:
                continue
            vals = None
            try:
                from well_correlation.petro.store import well_curves
                vals = well_curves(ws.wcp, w).get(curve)
            except Exception:
                vals = None
            if vals is None:
                vals = (w.logs or {}).get(curve)
            if vals is None or w.depth is None or name not in trajs:
                continue
            t = trajs[name]
            md = np.asarray(w.depth, float)
            x = np.interp(md, t.md, t.x)
            y = np.interp(md, t.md, t.y)
            z = np.interp(md, t.md, t.z)
            out.append((x, y, z, np.asarray(vals, float)))
        return out

    def upscale_project(self, curve="PHIE", method="arithmetic"):
        wells = self.project_wells(curve)
        if not wells:
            self._say(f"No well with a {curve} curve and a position inside the grid")
            return None
        return self.upscale(curve, wells, method)

    # ------------------------------------------------------------------ 36b
    def variogram(self, prop, azimuth=0.0):
        a = self.cells.get(prop)
        if a is None:
            raise KeyError(prop)
        Xc, Yc, Zc, _, _ = PM.cell_centres(self.grid)
        X = np.broadcast_to(Xc, a.shape)
        Y = np.broadcast_to(Yc, a.shape)
        h, g, n = PM.variogram_3d(X, Y, Zc, a, azimuth)
        hv, gv, nv = PM.variogram_3d(X, Y, Zc, a, vertical=True)
        s, r, nug = PM.fit_variogram(h, g, n)
        sv, rv, _ = PM.fit_variogram(hv, gv, nv) if len(hv) else (s, 3 * float(np.nanmedian(np.diff(Zc, axis=0))), 0)
        dz = float(np.nanmedian(np.abs(np.diff(Zc, axis=0)))) if self.grid.nz > 1 else 1.0
        self.vario = PM.Vario(max(s, 1e-12), max(r / self.grid.geom.dx, 1.0), max(r / self.grid.geom.dy, 1.0),
                              max(rv / max(dz, 1e-6), 1.0), azimuth, nug)
        self._say(f"{prop} variogram: sill {s:.4g}, range {r:.0f} m (vertical {rv:.1f} m)")
        return {"h": h, "gamma": g, "n": n, "sill": s, "range": r, "nugget": nug, "vertical_range": rv}

    # ------------------------------------------------------------------ 36c / 36d
    def simulate(self, prop, n=10, seed0=1, facies=None, facies_code=None):
        a = self.cells.get(prop)
        if a is None:
            raise KeyError(prop)
        self.realizations[prop] = [PM.sgs(self.grid, a, self.vario, seed=seed0 + k, facies=facies,
                                          facies_code=facies_code) for k in range(int(n))]
        self._say(f"{n} {prop} realizations (SGS, seeds {seed0}-{seed0 + n - 1})")
        return self.realizations[prop]

    # ------------------------------------------------------------------ 36e
    def volumetrics(self, phi="PHIE", sw=None, owc=None, goc=None, bo=1.2, bg=0.005):
        if self.grid.model.domain != "depth":
            raise ValueError("volumetrics need the framework in depth (Framework dialog: 2. Depth)")
        reals = self.realizations.get(phi) or ([self.cells[phi]] if phi in self.cells else [])
        if not reals:
            raise KeyError(phi)
        swv = self.realizations.get(sw) if sw else None
        totals, rows_all = [], []
        for k, r in enumerate(reals):
            rows = PM.volumetrics(self.grid, np.nan_to_num(r, nan=0.0), None if swv is None else swv[k % len(swv)],
                                  owc=owc, goc=goc, bo=bo, bg=bg)
            totals.append(PM.totals(rows))
            rows_all.append(rows)
        self.rows = rows_all[int(np.argsort(totals)[len(totals) // 2])]  # the P50-ish realization's breakdown
        self.stats = PM.realization_stats(totals)
        self.publish_blocks()
        self._say(f"STOIIP P90 {self.stats['P90'] / 1e6:.2f}, P50 {self.stats['P50'] / 1e6:.2f}, P10 "
                  f"{self.stats['P10'] / 1e6:.2f} Mm3 ({self.stats['n']} realizations)")
        return self.stats

    def publish_blocks(self):
        """Per fault block STOIIP as labelled points on the Live Map (map bus layer request)."""
        if self.bus is None or not self.rows:
            return None
        from ..livemap import layer_request
        items = [{"x": b["x"], "y": b["y"], "text": f"Block {b['block']}: {b['stoiip_m3'] / 1e6:.2f} Mm3"}
                 for b in PM.block_summary(self.grid, self.rows)]
        return self.bus.notify("layers", source="seismic", layer=layer_request("Volumetrics per block", "points",
                                                                                items, "propmodel"))

    def export(self, grdecl_path, csv_path=None, prop="PHIE"):
        props = {}
        if self.realizations.get(prop):
            props["PORO"] = np.nan_to_num(self.realizations[prop][0])
        elif prop in self.cells:
            props["PORO"] = np.nan_to_num(self.cells[prop])
        FW.write_grdecl(grdecl_path, self.grid, props)
        out = [grdecl_path]
        if csv_path and self.rows:
            out.append(PM.write_csv(csv_path, self.rows))
        return out

    def _export_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "GRDECL with properties", "", "Eclipse grid (*.grdecl)")
        if path:
            self._say("Written " + ", ".join(self.export(path, path.rsplit(".", 1)[0] + "_volumetrics.csv")))
