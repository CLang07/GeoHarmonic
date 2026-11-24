"""Arbitrary seismic section along a polyline (PLAN 21h: a line drawn in QGIS, or a section line from the shared
GeoPackage).  compute() is testable without a display."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtWidgets  # noqa: E402
import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from . import colormaps  # noqa: E402


class ArbitraryDialog(QtWidgets.QDialog):
    def __init__(self, project, xs, ys, name="Arbitrary line", volume=None, parent=None):
        super().__init__(parent)
        self.project = project
        self.xs, self.ys = np.asarray(xs, float), np.asarray(ys, float)
        self.vol_name = volume or (project.volumes[0]["name"] if project.volumes else None)
        self.setWindowTitle(f"{name} - {self.vol_name}")
        self.resize(1000, 600)
        lay = QtWidgets.QVBoxLayout(self)
        self.info = QtWidgets.QLabel("")
        lay.addWidget(self.info)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.invertY(True)
        lay.addWidget(self.plot)
        self.data = None

    def compute(self, spacing=None):
        if self.vol_name is None:
            raise ValueError("the project has no 3D volume")
        vol = self.project.volume(self.vol_name)
        sec, ils, xls = vol.arbitrary(self.xs, self.ys, spacing)
        inside = (ils >= vol.inlines[0]) & (ils <= vol.inlines[-1])
        self.data = {"section": sec, "il": ils, "xl": xls, "times": vol.times, "inside": inside}
        self.info.setText(f"{len(ils)} traces, inlines {ils.min()}-{ils.max()}, crosslines {xls.min()}-{xls.max()}")
        return self.data

    def show_section(self):
        d = self.data or self.compute()
        sec = d["section"]
        lim = float(np.nanpercentile(np.abs(sec), 99)) or 1.0
        img = pg.ImageItem(np.asarray(sec, np.float32).T)
        img.setLookupTable(colormaps.lut("Seismic (red-white-blue)"))
        img.setLevels((-lim, lim))
        t = d["times"]
        img.setRect(0, float(t[0]), sec.shape[0], float(t[-1] - t[0]))
        self.plot.clear()
        self.plot.addItem(img)
        self.plot.setLabel("left", "TWT (ms)")
        self.plot.setLabel("bottom", "trace along the line")
        return img
