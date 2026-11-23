"""Basemap: 3D survey outlines, 2D lines, wells, and the section currently shown."""
import math
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtWidgets  # noqa: E402
import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from .. import horizons as hz  # noqa: E402
from . import colormaps  # noqa: E402

COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75)]


class Basemap(QtWidgets.QWidget):
    openLine = QtCore.Signal(str)  # 2D line name
    openInline = QtCore.Signal(str, int, int)  # volume, inline, crossline (map click inside a survey)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("bottom", "X (m)")
        self.plot.setLabel("left", "Y (m)")
        lay.addWidget(self.plot)
        hint = QtWidgets.QLabel("Click a 2D line to open it; click inside a 3D survey to open the inline there.")
        hint.setStyleSheet("color:#555")
        lay.addWidget(hint)
        self.legend = QtWidgets.QLabel("")
        lay.addWidget(self.legend)
        self.project = None
        self._lines = {}  # name -> (x, y) decimated
        self._current = None
        self._hz_name = None  # horizon shown as a time-structure map
        self._hz_items = []
        self.hz_range = None  # (vmin, vmax) of the last map, for the legend text
        self.plot.scene().sigMouseClicked.connect(self._clicked)

    def set_project(self, project):
        self.project = project
        self.redraw()

    def redraw(self):
        self.plot.clear()
        self._lines = {}
        self._current = None
        p = self.project
        if not p:
            return
        for k, v in enumerate(p.volumes):
            try:
                vol = p.volume(v["name"])
            except Exception:
                continue
            xs, ys = vol.outline()
            c = COLORS[k % len(COLORS)]
            self.plot.addItem(pg.PlotCurveItem(np.asarray(xs), np.asarray(ys), pen=pg.mkPen(c, width=2),
                                               fillLevel=None))
            t = pg.TextItem(f"{v['name']} (3D)", color=c, anchor=(0, 1))
            t.setPos(xs[0], ys[0])
            self.plot.addItem(t, ignoreBounds=True)
        for l in p.lines:
            try:
                line = p.line(l["name"])
            except Exception:
                continue
            if not line.has_coords:
                continue
            step = max(1, len(line.x) // 400)
            x, y = line.x[::step], line.y[::step]
            self._lines[l["name"]] = (x, y)
            self.plot.addItem(pg.PlotCurveItem(x, y, pen=pg.mkPen((90, 90, 90), width=1.5)))
            t = pg.TextItem(l["name"], color=(60, 60, 60), anchor=(0.5, 1.2))
            t.setPos(x[0], y[0])
            self.plot.addItem(t, ignoreBounds=True)
        wells = p.wells()
        if wells:
            self.plot.addItem(pg.ScatterPlotItem([w[1] for w in wells], [w[2] for w in wells], symbol="o", size=8,
                                                 brush=pg.mkBrush(0, 0, 0), pen=pg.mkPen("w")))
            for name, x, y in wells:
                t = pg.TextItem(name, color=(0, 0, 0), anchor=(-0.1, 1.1))
                t.setPos(x, y)
                self.plot.addItem(t, ignoreBounds=True)
        self._hz_items = []
        self._draw_horizon_map()
        self._current = pg.PlotCurveItem(pen=pg.mkPen((220, 0, 0), width=3))
        self.plot.addItem(self._current)
        self.plot.getPlotItem().getViewBox().autoRange(padding=0.05)

    def show_horizon(self, name):
        """Show a horizon's time-structure map (3D grids as colour images, 2D picks as coloured dots)."""
        self._hz_name = name
        self._draw_horizon_map()

    def horizon_map_layers(self, name):
        """Pure-data description of the map for a horizon: list of ("image", rgba, transform6) and
        ("points", x, y, rgba) plus (vmin, vmax) shared by all layers."""
        p = self.project
        h = p.horizons.get(name) if p and name else None
        if h is None:
            return [], None
        vals = [a[np.isfinite(a)] for a in h.data.values()]
        vals = np.concatenate(vals) if vals else np.array([])
        if not vals.size:
            return [], None
        vmin, vmax = float(vals.min()), float(vals.max())
        lut = colormaps.lut("Rainbow (attributes)", reverse=True)
        out = []
        for key, a in h.data.items():
            kind, sname = key.split(":", 1)
            try:
                if kind == "3D":
                    vol = p.volume(sname)
                    rgba, _, _ = hz.structure_rgba(a, lut, vmin, vmax)
                    out.append(("image", rgba, hz.grid_transform(vol)))
                else:
                    line = p.line(sname)
                    fin = np.isfinite(a)
                    if not line.has_coords or not fin.any():
                        continue
                    rgba, _, _ = hz.structure_rgba(a[fin], lut, vmin, vmax)
                    out.append(("points", line.x[fin], line.y[fin], rgba))
            except Exception:
                continue
        return out, (vmin, vmax)

    def _draw_horizon_map(self):
        for it in self._hz_items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._hz_items = []
        layers, self.hz_range = self.horizon_map_layers(self._hz_name)
        from PySide6 import QtGui
        for lay in layers:
            if lay[0] == "image":
                it = pg.ImageItem(lay[1])
                it.setTransform(QtGui.QTransform(*lay[2]))
                it.setZValue(-10)
            else:
                _, x, y, rgba = lay
                it = pg.ScatterPlotItem(x, y, size=5, pen=None, brush=[pg.mkBrush(*c) for c in rgba[:, :3]])
                it.setZValue(-5)
            self.plot.addItem(it)
            self._hz_items.append(it)
        self.legend.setText(f"Time map {self._hz_name}: {self.hz_range[0]:.0f} ms (red, shallow) to "
                            f"{self.hz_range[1]:.0f} ms (purple, deep)" if self.hz_range else "")

    def show_current(self, xy):
        if self._current is None:
            return
        if xy is None:
            self._current.setData([], [])
        else:
            self._current.setData(np.asarray(xy[0], float), np.asarray(xy[1], float))

    def _clicked(self, ev):
        if not self.project or ev.button() != QtCore.Qt.LeftButton:
            return
        vb = self.plot.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        pt = vb.mapSceneToView(ev.scenePos())
        px = abs(vb.viewPixelSize()[0]) * 8  # click tolerance
        best = None
        for name, (x, y) in self._lines.items():
            d = np.min(np.hypot(x - pt.x(), y - pt.y()))
            if d <= px and (best is None or d < best[0]):
                best = (d, name)
        if best:
            self.openLine.emit(best[1])
            return
        for v in self.project.volumes:
            vol = self.project.volume(v["name"])
            il, xl = vol.ilxl_at(pt.x(), pt.y())
            if vol.inlines[0] - 0.5 * vol.il_step <= il <= vol.inlines[-1] + 0.5 * vol.il_step and \
                    vol.xlines[0] - 0.5 * vol.xl_step <= xl <= vol.xlines[-1] + 0.5 * vol.xl_step:
                self.openInline.emit(v["name"], int(round(il)), int(round(xl)))
                return
