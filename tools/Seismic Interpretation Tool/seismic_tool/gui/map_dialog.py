"""Horizon maps window: time structure, isochron, amplitude extraction, depth conversion; contours; export."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtWidgets  # noqa: E402
import numpy as np  # noqa: E402

from .. import maps as M  # noqa: E402

PRODUCTS = ["Time structure", "Isochron", "Amplitude", "Depth (well velocities)"]
AMPS = ["rms", "max_abs", "max", "min", "mean", "at"]


class MapDialog(QtWidgets.QDialog):
    def __init__(self, project, basemap=None, parent=None, source=None):
        super().__init__(parent)
        self.setWindowTitle("Horizon maps")
        self.project, self.basemap = project, basemap
        self.grid = None
        form = QtWidgets.QFormLayout(self)
        self.survey = QtWidgets.QComboBox()
        keys = [f"3D:{v['name']}" for v in project.volumes] + (["2D"] if project.lines else [])
        self.survey.addItems(keys)
        if source:
            want = f"3D:{source[1]}" if source[0] == "3D" else "2D"
            if want in keys:
                self.survey.setCurrentText(want)
        self.product = QtWidgets.QComboBox()
        self.product.addItems(PRODUCTS)
        self.h1 = QtWidgets.QComboBox()
        self.h2 = QtWidgets.QComboBox()
        names = project.horizons.names()
        self.h1.addItems(names)
        self.h2.addItems(names)
        self.amp = QtWidgets.QComboBox()
        self.amp.addItems(AMPS)
        self.above = QtWidgets.QDoubleSpinBox()
        self.below = QtWidgets.QDoubleSpinBox()
        for s_ in (self.above, self.below):
            s_.setRange(0, 500)
            s_.setSuffix(" ms")
        self.below.setValue(20)
        self.fm = QtWidgets.QComboBox()
        self.fm.addItem("(horizon time at the wells)")
        ws = project.wellset() if project.well_project else None
        if ws is not None and ws.wcp is not None:
            self.fm.addItems([f["name"] for f in ws.wcp.formations])
        self.cint = QtWidgets.QDoubleSpinBox()
        self.cint.setRange(0, 10000)
        self.cint.setSpecialValueText("auto")
        self.cell = QtWidgets.QDoubleSpinBox()
        self.cell.setRange(5, 5000)
        self.cell.setValue(100)
        self.cell.setSuffix(" m")
        form.addRow("Survey", self.survey)
        form.addRow("Map", self.product)
        form.addRow("Horizon", self.h1)
        form.addRow("Base horizon (isochron)", self.h2)
        form.addRow("Amplitude", self.amp)
        form.addRow("Window above", self.above)
        form.addRow("Window below", self.below)
        form.addRow("Calibrate depth to top", self.fm)
        form.addRow("Contour interval", self.cint)
        form.addRow("2D grid cell", self.cell)
        row = QtWidgets.QHBoxLayout()
        self.btn = QtWidgets.QPushButton("Compute + show on basemap")
        self.bx = QtWidgets.QPushButton("Export XYZ...")
        self.bc = QtWidgets.QPushButton("Export CPS-3...")
        self.bp = QtWidgets.QPushButton("Export PNG...")
        for b in (self.btn, self.bx, self.bc, self.bp):
            row.addWidget(b)
        form.addRow(row)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        form.addRow(self.info)
        self.btn.clicked.connect(lambda: self.show_map())
        self.bx.clicked.connect(lambda: self._export_dialog("xyz"))
        self.bc.clicked.connect(lambda: self._export_dialog("cps3"))
        self.bp.clicked.connect(lambda: self._export_dialog("png"))

    # ------------------------------------------------------------ logic (testable)
    def settings(self):
        return {"key": self.survey.currentText(), "product": self.product.currentText() or PRODUCTS[0],
                "h1": self.h1.currentText(), "h2": self.h2.currentText(), "amp": self.amp.currentText() or "rms",
                "above": float(self.above.value()), "below": float(self.below.value()),
                "formation": None if self.fm.currentIndex() <= 0 else self.fm.currentText(),
                "cell": float(self.cell.value()) or 100.0}

    def compute(self, **over):
        s = self.settings()
        s.update(over)
        p = self.project
        h = p.horizons.get(s["h1"])
        if h is None:
            raise ValueError("choose a horizon")
        key = s["key"]
        g = M.horizon_grid(p, h, key, s["cell"])
        prod = s["product"]
        if prod == "Isochron":
            h2 = p.horizons.get(s["h2"])
            if h2 is None:
                raise ValueError("choose the base horizon")
            g = M.isochron(g, M.horizon_grid(p, h2, key, s["cell"]))
        elif prod == "Amplitude":
            if not key.startswith("3D:"):
                raise ValueError("amplitude extraction is for 3D volumes")
            vol = p.volume(key[3:])
            z = M.extract_amplitude(vol, g.z, s["above"], s["below"], s["amp"])
            g = M.MapGrid(z, "ilxl", "amp", f"{s['amp'].upper()} amplitude {h.name} "
                          f"(-{s['above']:g}/+{s['below']:g} ms)", vol)
        elif prod.startswith("Depth"):
            ws = p.wellset()
            pts = M.well_velocity_points(ws, g, s["formation"])
            g = M.depth_convert(g, pts)
            g.vel_pts = pts
        self.grid = g
        lo, hi, mean = g.stats()
        txt = f"{g.label}: {lo:.1f} to {hi:.1f} {g.unit} (mean {mean:.1f})"
        if hasattr(g, "vel_pts"):
            txt += "\nWell average velocities: " + ", ".join(f"{n} {v:.0f} m/s" for n, _, _, v, _, _ in g.vel_pts)
        self.info.setText(txt)
        return g

    def levels(self):
        ci = float(self.cint.value())
        if not ci or self.grid is None:
            return None
        f = self.grid.z[np.isfinite(self.grid.z)]
        return list(np.arange(np.ceil(f.min() / ci) * ci, f.max(), ci)) if f.size else []

    def show_map(self):
        try:
            g = self.compute()
        except Exception as e:
            self.info.setText(f"Cannot make the map: {e}")
            return None
        if self.basemap is not None:
            self.basemap.show_grid(g, self.levels())
        return g

    def export(self, kind, path):
        g = self.grid or self.compute()
        if kind == "xyz":
            return M.write_xyz(g, path)
        if kind == "cps3":
            return M.write_cps3(g, path)
        wells = []
        if self.project.well_project:
            ws = self.project.wellset()
            wells = [(n, ws.well(n).x, ws.well(n).y) for n in ws.names if ws.well(n).x is not None]
        return M.export_png(g, path, g.label, wells, self.levels())

    def _export_dialog(self, kind):
        flt = {"xyz": "XYZ (*.xyz *.txt)", "cps3": "CPS-3 grid (*.cps *.grd *.dat)", "png": "PNG (*.png)"}[kind]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export map", self.project.base(), flt)
        if path:
            try:
                self.export(kind, path)
                self.info.setText(f"Wrote {path}")
            except Exception as e:
                self.info.setText(f"Export failed: {e}")
