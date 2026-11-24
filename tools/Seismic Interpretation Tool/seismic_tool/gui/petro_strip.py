"""Map hub petro views (PLAN 21b): the petro strip of a clicked well (logs + petrophysics results + tops + its zone
rows) and the linked cross-plot of zone values per well (selected wells highlighted)."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtWidgets  # noqa: E402
import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from .. import maphub as MH  # noqa: E402

PROPS = ["SoPhiH", "NetPay_TVD", "PHIH_pay", "PHIE_res", "Sw_pay", "Vsh_avg", "NtG_res", "kh_pay", "Gross_TVD", "TOC_avg"]


class PetroStripDialog(QtWidgets.QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Petro strip")
        self.resize(900, 750)
        self.data = None
        lay = QtWidgets.QVBoxLayout(self)
        self.title = QtWidgets.QLabel("")
        lay.addWidget(self.title)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        lay.addWidget(self.glw, 3)
        self.table = QtWidgets.QTableWidget()
        lay.addWidget(self.table, 1)

    def show_well(self, name):
        self.data = MH.petro_strip_data(self.project, name)
        if self.data is None:
            self.title.setText(f"{name}: not in the linked well project")
            return None
        d = self.data
        self.title.setText(f"<b>{name}</b> - logs and petrophysics vs MD" +
                           ("" if d["has_petro"] else " (no petrophysics results yet: run the Petrophysics window)"))
        self._draw()
        self._fill_table()
        return d

    def _draw(self):
        d = self.data
        self.glw.clear()
        md = d["md"]
        first = None
        for k, (title, curves) in enumerate(d["tracks"].items()):
            p = self.glw.addPlot(row=0, col=k, title=title)
            p.invertY(True)
            p.showGrid(x=True, y=True, alpha=0.2)
            if first is None:
                first = p
                p.setLabel("left", "MD (m)")
            else:
                p.setYLink(first)
                p.getAxis("left").setStyle(showValues=False)
            for label, v, col, log in curves:
                v = np.asarray(v, float)
                x = np.log10(np.where(v > 0, v, np.nan)) if log else v
                p.plot(x, md, pen=pg.mkPen(col, width=1.2), connect="finite", name=label)
            for f, tmd in d["tops"]:
                p.addLine(y=tmd, pen=pg.mkPen((90, 90, 90), style=QtCore.Qt.DashLine))
                if k == 0:
                    t = pg.TextItem(f, color=(60, 60, 60), anchor=(0, 1))
                    t.setPos(0, tmd)
                    p.addItem(t)

    def _fill_table(self):
        rows = self.data["zones"]
        cols = ["Zone", "Gross_TVD", "NetRes_TVD", "NetPay_TVD", "PHIE_res", "Sw_pay", "SoPhiH", "kh_pay"]
        self.table.clear()
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(str(r.get(c, ""))))


class PetroCrossplotDialog(QtWidgets.QDialog):
    """Zone values per well: X vs Y (or a map grid at the wells vs Y); wells selected on the map highlighted."""

    def __init__(self, project, basemap=None, parent=None):
        super().__init__(parent)
        self.project, self.basemap = project, basemap
        self.setWindowTitle("Petro cross-plot (map hub)")
        self.resize(650, 560)
        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.xb, self.yb, self.zb = QtWidgets.QComboBox(), QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.xb.addItems(PROPS + ["grid"])
        self.yb.addItems(PROPS)
        self.yb.setCurrentIndex(3)
        self.zb.addItem("(all zones)")
        for lab, w in (("X", self.xb), ("Y", self.yb), ("Zone", self.zb)):
            row.addWidget(QtWidgets.QLabel(lab))
            row.addWidget(w)
            w.currentIndexChanged.connect(lambda *_: self.redraw())
        lay.addLayout(row)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        lay.addWidget(self.plot)
        self.info = QtWidgets.QLabel("")
        lay.addWidget(self.info)
        self.data = None
        zones = sorted({r.get("Zone") for r in MH._read_csv(os.path.join(MH.petro_folder(project) or "", "summary.csv"))
                        if r.get("Zone")})
        self.zb.addItems(zones)

    def compute(self, prop_x=None, prop_y=None, zone=None, selected=None):
        px = prop_x or self.xb.currentText() or "PHIE_res"
        py = prop_y or self.yb.currentText() or "SoPhiH"
        if zone is None:
            zt = self.zb.currentText()
            zone = None if (not zt or zt.startswith("(")) else zt
        grid = getattr(self.basemap, "_grid", (None, None))[0] if self.basemap is not None else None
        sel = selected if selected is not None else (self.basemap.selected_wells if self.basemap is not None else [])
        self.data = MH.petro_crossplot_data(self.project, px, py, zone, sel, grid)
        return self.data

    def redraw(self, *_):
        d = self.compute()
        self.plot.clear()
        if not d["n"]:
            self.info.setText("no wells with both values - run the Petrophysics window in the Well Correlation Tool")
            return
        s = d["selected"]
        self.plot.addItem(pg.ScatterPlotItem(d["x"][~s], d["y"][~s], size=9, brush=pg.mkBrush(70, 120, 180)))
        if s.any():
            self.plot.addItem(pg.ScatterPlotItem(d["x"][s], d["y"][s], size=12, brush=pg.mkBrush(255, 0, 255),
                                                 pen=pg.mkPen("k")))
        for w, x, y in zip(d["wells"], d["x"], d["y"]):
            t = pg.TextItem(w, color=(60, 60, 60), anchor=(-0.1, 1))
            t.setPos(x, y)
            self.plot.addItem(t)
        self.plot.setLabel("bottom", d["xlabel"])
        self.plot.setLabel("left", d["ylabel"])
        self.info.setText(f"{d['n']} wells, r = {d['r']:.2f}; {int(s.sum())} selected on the map (magenta)")
