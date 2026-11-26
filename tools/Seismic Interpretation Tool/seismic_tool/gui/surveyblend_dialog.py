"""Interpret > Survey horizon blending (PLAN 42m): load a horizon from two overlapping surveys > priority > method >
allowed slope (min / max width under Advanced) > fault barriers > preview (seam on / off) > export merged grid + QC
maps (ZMAP+, map-hub layers, Live Map bus).  Maths in `seismic_tool.surveyblend` (tested)."""
import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from .. import surveyblend as SB


class SurveyBlendDialog(QtWidgets.QDialog):
    def __init__(self, project, parent=None, run_job=None, bus=None, stack=None, depth_model_fn=None):
        super().__init__(parent)
        self.setWindowTitle("Survey horizon blending")
        self.project, self.run_job_fn, self.bus, self.stack = project, run_job, bus, stack
        self.depth_model_fn = depth_model_fn
        self.result = None
        self.grid = None
        self.inputs = None
        lay = QtWidgets.QHBoxLayout(self)
        f = QtWidgets.QFormLayout()
        lay.addLayout(f, 0)
        self.hz = QtWidgets.QComboBox()
        self.hz.addItems([h.name for h in project.horizons.items] if getattr(project, "horizons", None) else [])
        vols = [v["name"] for v in project.volumes]
        self.va, self.vb = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.va.addItems(vols)
        self.vb.addItems(vols)
        if len(vols) > 1:
            self.vb.setCurrentIndex(1)
        self.domain = QtWidgets.QComboBox()
        self.domain.addItems(["Depth (velocity model)", "Time"])
        self.method = QtWidgets.QComboBox()
        self.method.addItems(["Hybrid (residual surface + narrow feather)", "Residual surface", "Feather",
                              "Priority cut (no blending)"])
        self.slope = QtWidgets.QDoubleSpinBox()
        self.slope.setRange(0.1, 100.0)
        self.slope.setValue(2.0)
        self.unit = QtWidgets.QComboBox()
        self.unit.addItems(["m/km", "ft/mile", "ms/km"])
        self.wmin, self.wmax = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for w, v in ((self.wmin, 500.0), (self.wmax, 10000.0)):
            w.setRange(0.0, 100000.0)
            w.setSuffix(" m")
            w.setValue(v)
        self.trim = QtWidgets.QSpinBox()
        self.trim.setRange(0, 200)
        self.trim.setValue(5)
        self.trim.setSuffix(" traces")
        self.faults = QtWidgets.QCheckBox("Faults are barriers (no blending across a fault)")
        self.faults.setChecked(True)
        self.wells = QtWidgets.QCheckBox("Re-tie well tops (WellTable column 'Top <horizon>')")
        self.wells.setChecked(True)
        self.seam = QtWidgets.QCheckBox("Show blended (untick = priority cut)")
        self.seam.setChecked(True)
        self.show_combo = QtWidgets.QComboBox()
        self.show_combo.addItems(["Merged grid", "Residual before", "Weight", "Width", "Added slope", "Correction"])
        for lab, w in (("Horizon", self.hz), ("Survey A (priority)", self.va), ("Survey B", self.vb),
                       ("Domain", self.domain), ("Method", self.method), ("Allowed added slope", self.slope),
                       ("Units", self.unit), ("Min width (Advanced)", self.wmin), ("Max width (Advanced)", self.wmax),
                       ("Edge trim", self.trim), ("", self.faults), ("", self.wells), ("", self.seam),
                       ("Show", self.show_combo)):
            f.addRow(lab, w)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("Preview", self.preview), ("Export...", self.export_dialog), ("Close", self.accept)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda *_, fn=fn: fn())
            row.addWidget(b)
        f.addRow(row)
        self.readout = QtWidgets.QLabel("")
        self.readout.setWordWrap(True)
        f.addRow(self.readout)
        self.plot = pg.PlotWidget()
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        self.plot.setAspectLocked(True)
        lay.addWidget(self.plot, 1)
        self.seam.toggled.connect(lambda *_: self.redraw())
        self.show_combo.currentTextChanged.connect(lambda *_: self.redraw())
        self.resize(1250, 760)

    # ------------------------------------------------------------ inputs (42a)
    def load_inputs(self, horizon=None, vol_a=None, vol_b=None, domain=None):
        hz = horizon or self.hz.currentText()
        na, nb = vol_a or self.va.currentText(), vol_b or self.vb.currentText()
        h = self.project.horizons.get(hz)
        if h is None:
            raise ValueError(f"no horizon {hz!r}")
        va, vb = self.project.volume(na), self.project.volume(nb)
        ga, gb = h.data.get(f"3D:{na}"), h.data.get(f"3D:{nb}")
        if ga is None or gb is None:
            raise ValueError(f"{hz} is not picked on both surveys ({na}, {nb})")
        grid = SB.MasterGrid.covering([va, vb])
        za, zb = grid.resample(va, ga), grid.resample(vb, gb)
        dom = domain or self.domain.currentText()
        unit = "ms"
        if dom.startswith("Depth"):
            dm = self.depth_model_fn() if self.depth_model_fn else None
            if dm is not None:
                X, Y = grid.xy()
                try:
                    za = np.asarray(dm.depth_of_time(X.ravel(), Y.ravel(), za.ravel()), float).reshape(za.shape)
                    zb = np.asarray(dm.depth_of_time(X.ravel(), Y.ravel(), zb.ravel()), float).reshape(zb.shape)
                    unit = "m"
                except Exception as e:
                    self.readout.setText(f"Depth conversion failed ({e}) - blending in time")
            else:
                self.readout.setText("No velocity model - blending in time (Interpret > Velocity model)")
        self.grid, self.inputs = grid, {"za": za, "zb": zb, "names": (na, nb), "horizon": hz, "unit": unit}
        return grid, za, zb

    def fault_lines(self):
        """Fault traces on the map from the project's fault sticks (top point of each stick, ordered)."""
        out = []
        for f in getattr(getattr(self.project, "faults", None), "items", []) or []:
            pts = []
            for st in getattr(f, "sticks", []) or []:
                a = np.asarray(st, float)
                if a.ndim == 2 and len(a):
                    pts.append(a[np.argmin(a[:, 2])][:2])
            if len(pts) >= 2:
                p = np.array(pts)
                d = p - p.mean(axis=0)
                u = np.linalg.svd(d, full_matrices=False)[2][0]
                out.append(p[np.argsort(d @ u)].tolist())
        return out

    def well_tops(self, horizon):
        if self.bus is None:
            return []
        t = self.bus.table
        col = f"Top {horizon}"
        if col not in t.columns():
            return []
        xy = t.xy()
        vals = t.numeric(col)
        return [(i, float(x), float(y), float(v)) for i, (x, y), v in zip(t.ids, np.asarray(xy).reshape(-1, 2), vals)
                if np.isfinite(v) and np.isfinite(x) and np.isfinite(y)]

    # ------------------------------------------------------------ run
    def params(self):
        m = {"Hybrid": "hybrid", "Residual": "residual", "Feather": "feather", "Priority": "cut"}
        meth = next((v for k, v in m.items() if self.method.currentText().startswith(k)), "hybrid")
        unit = self.unit.currentText() or "m/km"
        if self.inputs and self.inputs.get("unit") == "ms" and unit != "ms/km":
            unit = "ms/km"
        return dict(slope=float(self.slope.value() or 2.0), unit=unit, method=meth,
                    w_min=float(self.wmin.value() or 500.0), w_max=float(self.wmax.value() or 10000.0),
                    trim=int(self.trim.value() or 0))

    def compute(self, **override):
        if self.inputs is None:
            self.load_inputs()
        p = self.params()
        p.update(override)
        faults = self.fault_lines() if self.faults.isChecked() else []
        wells = self.well_tops(self.inputs["horizon"]) if self.wells.isChecked() else []
        r = SB.blend(self.grid, [SB.Survey(self.inputs["names"][0], self.inputs["za"], 1),
                                 SB.Survey(self.inputs["names"][1], self.inputs["zb"], 2)],
                     faults=faults, wells=wells, **p)
        self.result = r
        return r

    def preview(self, threaded=False, **override):
        try:
            r = self.compute(**override)
        except Exception as e:
            self.readout.setText(f"Blend failed: {e}")
            return None
        self.readout.setText(r.readout)
        self.redraw()
        if self.bus is not None:
            self.bus.notify("grids", ids=[f"{self.inputs['horizon']} (merged)"], source="surveyblend",
                            readout=r.readout)
        return r

    def layer_array(self, what=None):
        r = self.result
        if r is None:
            return None
        what = what or self.show_combo.currentText() or "Merged grid"
        if what.startswith("Merged"):
            return r.z if self.seam.isChecked() else r.cut
        key = {"Residual before": "residual_before", "Weight": "weight", "Width": "width",
               "Added slope": "added_slope", "Correction": "corr_b"}.get(what, "residual_before")
        return r.qc.get(key)

    def redraw(self):
        a = self.layer_array()
        if a is None:
            return None
        arr = np.asarray(a, float)
        fin = arr[np.isfinite(arr)]
        lv = (float(np.percentile(fin, 1)), float(np.percentile(fin, 99))) if fin.size else (0.0, 1.0)
        from .colormaps import lut
        self.img.setImage(np.where(np.isfinite(arr), arr, lv[0]).astype(np.float32), levels=lv)
        self.img.setLookupTable(lut("Viridis-style"))
        return arr

    # ------------------------------------------------------------ export
    def export(self, folder, add_layers=True):
        """Merged grid + QC maps as ZMAP+ files; merged grid (+ width / added slope) as map-hub layers."""
        import os
        from ..gis.zmap import write_zmap
        from ..maps import MapGrid
        r = self.result or self.compute()
        g = self.grid
        os.makedirs(folder, exist_ok=True)
        hz = self.inputs["horizon"]
        files = []
        for lab, arr in (("merged", r.z), ("residual_before", r.qc["residual_before"]), ("width", r.qc.get("width")),
                         ("added_slope", r.qc.get("added_slope")), ("weight", r.qc.get("weight"))):
            if arr is None:
                continue
            mg = MapGrid(np.asarray(arr, float), "xy", self.inputs["unit"] if lab == "merged" else "",
                         f"{hz} {lab}", x0=g.x0, y0=g.y0, dx=g.dx, dy=g.dy)
            pth = os.path.join(folder, f"{hz}_{lab}.zmap".replace(" ", "_"))
            write_zmap(pth, mg, comment=f"survey blend {r.method} {r.slope:g} {r.unit}")
            files.append(pth)
            if add_layers and self.stack is not None and lab in ("merged", "width", "added_slope"):
                try:
                    self.stack.add_grid(mg, f"{hz} ({lab.replace('_', ' ')})", contours=lab == "merged")
                except Exception:
                    pass
        if self.bus is not None:
            self.bus.notify("grids", ids=[f"{hz} (merged)"], source="surveyblend", files=files)
            self.bus.notify("layers", ids=[f"{hz} (merged)"], source="surveyblend")
        return files

    def export_dialog(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Export merged grid + QC maps")
        if folder:
            files = self.export(folder)
            self.readout.setText(self.readout.text() + f"\nWrote {len(files)} ZMAP+ files to {folder}")
            return files
        return None
