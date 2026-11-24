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
    wellClicked = QtCore.Signal(str)  # well name (map hub: petro strip)
    polygonDrawn = QtCore.Signal(object)  # (xs, ys) when a drawn polygon is finished

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
        self._flt_items = []
        self._grid = (None, None)
        self._grid_items = []
        self._tie_items = []
        self.hz_range = None  # (vmin, vmax) of the last map, for the legend text
        self.wells = None  # wellindex.WellLayer
        self._label_items = []
        self._poly = None  # [(x, y)] while drawing a polygon (None = not drawing)
        self._poly_item = None
        self.selected_wells = []
        self._sel_items = []
        self.stack = None  # maplayers.LayerStack: when set, the map is drawn from the layer stack (PLAN 21a)
        self._layer_items = []
        self.layer_errors = []
        self._label_timer = QtCore.QTimer(self)
        self._label_timer.setSingleShot(True)
        self._label_timer.setInterval(150)
        self._label_timer.timeout.connect(self.update_labels)
        self.plot.getPlotItem().getViewBox().sigRangeChanged.connect(lambda *_: self._label_timer.start())
        self.plot.scene().sigMouseClicked.connect(self._clicked)

    def set_project(self, project):
        self.project = project
        self.redraw()

    def set_stack(self, stack):
        """Draw the map from a maplayers.LayerStack (None = the fixed legacy basemap)."""
        self.stack = stack
        self.redraw()

    def _wells_layer_on(self):
        if self.stack is None:
            return True
        return any(l["kind"] == "wells" and l["visible"] and l["style"].get("labels", True) for l in self.stack.layers)

    def draw_layers(self):
        """(Re)draw the layer-stack primitives (keeps the horizon / grid / tie / current-line overlays)."""
        for it in self._layer_items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._layer_items = []
        if self.stack is None:
            return []
        from PySide6 import QtGui
        from ..maplayers import render
        prims = render(self.stack)
        self.layer_errors = [q for q in prims if q[0] == "error"]
        order = {l["id"]: k for k, l in enumerate(self.stack.ordered())}
        for q in prims:
            kind, lid = q[0], q[1]
            zval = -100 + order.get(lid, 0)  # all layers below the overlays (horizon map, grid, ties, current line)
            if kind == "image":
                it = pg.ImageItem(q[2])
                it.setTransform(QtGui.QTransform(*q[3]))
            elif kind == "curve":
                it = pg.PlotCurveItem(q[2], q[3], connect=q[6], pen=pg.mkPen(q[4], width=q[5]))
            elif kind == "points":
                br = [pg.mkBrush(*c) for c in q[4]] if isinstance(q[4], list) else pg.mkBrush(*q[4])
                sz = list(np.asarray(q[5], float)) if np.ndim(q[5]) else q[5]
                it = pg.ScatterPlotItem(q[2], q[3], symbol=q[6], size=sz, brush=br, pen=pg.mkPen("w"))
            elif kind == "text":
                if len(q) > 7:  # contour label: centred, rotated with the line, white box as a halo
                    it = pg.TextItem(q[4], color=q[5][:3], anchor=(0.5, 0.5), angle=float(q[7]), fill=pg.mkBrush(255, 255, 255, 200))
                else:
                    it = pg.TextItem(q[4], color=q[5][:3], anchor=(0, 1))
                it.setPos(q[2], q[3])
            elif kind == "polygon":
                poly = QtGui.QPolygonF([QtCore.QPointF(float(a), float(b)) for a, b in zip(q[2], q[3])])
                it = QtWidgets.QGraphicsPolygonItem(poly)
                if len(q) > 6 and (q[6] or {}).get("hatch"):
                    it.setBrush(QtGui.QBrush(QtGui.QColor(*q[4]), QtCore.Qt.BDiagPattern))
                else:
                    it.setBrush(pg.mkBrush(*q[4]))
                it.setPen(pg.mkPen(q[5], width=0) if q[5] else pg.mkPen(None))
            else:
                continue
            it.setZValue(zval)
            self.plot.addItem(it, ignoreBounds=(kind == "text"))
            self._layer_items.append(it)
        return prims

    def redraw(self):
        self.plot.clear()
        self._lines = {}
        self._current = None
        self._layer_items = []
        p = self.project
        if not p:
            return
        if self.stack is not None:
            self.stack.project = p
            self._lines_for_clicks()
            self.draw_layers()
            from ..wellindex import WellLayer
            self.wells = WellLayer(p.wells() if self._wells_layer_on() else [])
            self._label_items = []
            self._hz_items = []
            self._flt_items = []
            self._grid_items = []
            self._draw_horizon_map()
            self._draw_grid()
            self._current = pg.PlotCurveItem(pen=pg.mkPen((220, 0, 0), width=3))
            self.plot.addItem(self._current)
            self.plot.getPlotItem().getViewBox().autoRange(padding=0.05)
            self.update_labels()
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
        hidden = set(p.display.get("hidden_lines", []) or [])
        for l in p.lines:
            if l["name"] in hidden:
                continue
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
        from ..wellindex import WellLayer
        self.wells = WellLayer(p.wells())
        self._label_items = []
        if len(self.wells):
            wx, wy = self.wells.points()
            big = len(self.wells) > 2000
            self.plot.addItem(pg.ScatterPlotItem(wx, wy, symbol="o", size=5 if big else 8, brush=pg.mkBrush(0, 0, 0),
                                                 pen=None if big else pg.mkPen("w")))
        self._hz_items = []
        self._flt_items = []
        self._grid_items = []
        self._draw_horizon_map()
        self._draw_grid()
        self.draw_faults()
        self._current = pg.PlotCurveItem(pen=pg.mkPen((220, 0, 0), width=3))
        self.plot.addItem(self._current)
        self.plot.getPlotItem().getViewBox().autoRange(padding=0.05)
        self.update_labels()

    def _lines_for_clicks(self):
        """Decimated 2D line coordinates for click picking (also in layer mode)."""
        p = self.project
        hidden = set(p.display.get("hidden_lines", []) or [])
        for l in p.lines:
            if l["name"] in hidden:
                continue
            try:
                line = p.line(l["name"])
            except Exception:
                continue
            if line.has_coords:
                step = max(1, len(line.x) // 400)
                self._lines[l["name"]] = (line.x[::step], line.y[::step])

    def label_view(self):
        """((x0, x1), (y0, y1)) of the map view, or None when it cannot be read."""
        try:
            (x0, x1), (y0, y1) = self.plot.getPlotItem().getViewBox().viewRange()
            return (float(x0), float(x1)), (float(y0), float(y1))
        except Exception:
            return None

    def update_labels(self, view=None):
        """Well name labels only for the wells in view (none when more than WellLayer.max_labels are visible)."""
        for it in self._label_items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._label_items = []
        if not self.wells or not len(self.wells):
            return []
        labs = self.wells.labels(view if view is not None else self.label_view())
        for name, x, y in labs:
            t = pg.TextItem(name, color=(0, 0, 0), anchor=(-0.1, 1.1))
            t.setPos(x, y)
            self.plot.addItem(t, ignoreBounds=True)
            self._label_items.append(t)
        return labs

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

    def show_grid(self, grid, levels=None):
        """Show a maps.MapGrid (structure, isochron, amplitude, depth) with contours; None clears it."""
        self._grid = (grid, levels)
        self._draw_grid()

    def _draw_grid(self):
        for it in self._grid_items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._grid_items = []
        grid, levels = self._grid
        if grid is None:
            return
        from PySide6 import QtGui
        from .. import horizons as hz
        from ..maps import nice_levels
        lut = colormaps.lut("Rainbow (attributes)", reverse=grid.unit in ("ms", "m"))
        rgba, lo, hi = hz.structure_rgba(grid.z, lut)
        it = pg.ImageItem(rgba)
        it.setTransform(QtGui.QTransform(*grid.transform()))
        it.setZValue(-9)
        self.plot.addItem(it)
        self._grid_items.append(it)
        levels = levels if levels is not None else nice_levels(grid.z)
        for lev, (xs, ys) in grid.contours(levels).items():
            c = pg.PlotCurveItem(xs, ys, connect="pairs", pen=pg.mkPen((40, 40, 40), width=1))
            c.setZValue(-8)
            self.plot.addItem(c)
            self._grid_items.append(c)
        self.legend.setText(f"{grid.label}: {lo:.1f} - {hi:.1f} {grid.unit}"
                            + (f", contours every {levels[1] - levels[0]:g}" if len(levels) > 1 else ""))

    def show_ties(self, ties):
        """Mistie residual map: tie positions coloured by |residual shift| (green < 2 ms, orange < 5, red)."""
        for it in getattr(self, "_tie_items", []) or []:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._tie_items = []
        pts = [(t["xy"][0], t["xy"][1], abs(t.get("res_shift_ms", 0.0))) for t in ties or [] if t.get("xy")]
        if not pts:
            return []
        cols = [(0, 170, 0) if r < 2 else (240, 150, 0) if r < 5 else (220, 0, 0) for _, _, r in pts]
        it = pg.ScatterPlotItem([p_[0] for p_ in pts], [p_[1] for p_ in pts], size=11, symbol="d",
                                brush=[pg.mkBrush(*c) for c in cols], pen=pg.mkPen((0, 0, 0)))
        self.plot.addItem(it)
        self._tie_items.append(it)
        return pts

    def fault_map_lines(self):
        """[(name, colour, xs, ys)] - each stick projected on the map plus a line joining the stick tops."""
        p = self.project
        out = []
        if not p or not getattr(p, "faults", None):
            return out
        for f in p.faults.items:
            if not f.visible:
                continue
            for s in f.sticks:
                out.append((f.name, f.color, s[:, 0], s[:, 1]))
            st = f.ordered_sticks()
            if len(st) > 1:
                out.append((f.name, f.color, np.array([s[0, 0] for s in st]), np.array([s[0, 1] for s in st])))
        return out

    def draw_faults(self):
        if self.stack is not None:  # faults are a layer
            self.draw_layers()
            return
        for it in self._flt_items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._flt_items = []
        for _, c, xs, ys in self.fault_map_lines():
            it = pg.PlotCurveItem(np.asarray(xs, float), np.asarray(ys, float), pen=pg.mkPen(c, width=2))
            self.plot.addItem(it)
            self._flt_items.append(it)

    def show_current(self, xy):
        if self._current is None:
            return
        if xy is None:
            self._current.setData([], [])
        else:
            self._current.setData(np.asarray(xy[0], float), np.asarray(xy[1], float))

    # ------------------------------------------------------------------ polygon drawing + well selection (21b)
    def start_polygon(self):
        """Click vertices on the map; double-click (or finish_polygon()) closes it and emits polygonDrawn."""
        self._poly = []
        self._draw_poly()

    def add_polygon_point(self, x, y):
        if self._poly is not None:
            self._poly.append((float(x), float(y)))
            self._draw_poly()

    def finish_polygon(self):
        pts, self._poly = self._poly, None
        self._draw_poly()
        if pts and len(pts) >= 3:
            xs, ys = [p_[0] for p_ in pts], [p_[1] for p_ in pts]
            self.polygonDrawn.emit((xs, ys))
            return xs, ys
        return None

    def _draw_poly(self):
        if self._poly_item is not None:
            try:
                self.plot.removeItem(self._poly_item)
            except Exception:
                pass
            self._poly_item = None
        if self._poly:
            xs = [p_[0] for p_ in self._poly] + [self._poly[0][0]]
            ys = [p_[1] for p_ in self._poly] + [self._poly[0][1]]
            self._poly_item = pg.PlotCurveItem(np.array(xs), np.array(ys), pen=pg.mkPen((200, 0, 200), width=2,
                                                                                         style=QtCore.Qt.DashLine))
            self.plot.addItem(self._poly_item)

    def set_selected_wells(self, names):
        """Highlight wells (rings) - e.g. the wells inside a lasso polygon."""
        for it in self._sel_items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._sel_items = []
        self.selected_wells = list(names or [])
        xy = {n: (x, y) for n, x, y in (self.project.wells() if self.project else [])}
        pts = [xy[n] for n in self.selected_wells if n in xy]
        if pts:
            it = pg.ScatterPlotItem([q[0] for q in pts], [q[1] for q in pts], size=18, symbol="o", brush=None,
                                    pen=pg.mkPen((255, 0, 255), width=2))
            it.setZValue(50)
            self.plot.addItem(it)
            self._sel_items.append(it)
        return self.selected_wells

    def well_at(self, x, y, tol):
        """Nearest well within tol map units, or None."""
        if not self.wells or not len(self.wells):
            return None
        k = self.wells.nearest(x, y)
        if k is None:
            return None
        wx, wy = self.wells.xy[k]
        return self.wells.names[k] if math.hypot(wx - x, wy - y) <= tol else None

    def _clicked(self, ev):
        if not self.project or ev.button() != QtCore.Qt.LeftButton:
            return
        vb = self.plot.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        pt = vb.mapSceneToView(ev.scenePos())
        if self._poly is not None:  # drawing a polygon: clicks add vertices, double-click closes
            self.add_polygon_point(pt.x(), pt.y())
            if ev.double():
                self.finish_polygon()
            return
        px = abs(vb.viewPixelSize()[0]) * 8  # click tolerance
        wname = self.well_at(pt.x(), pt.y(), px)
        if wname:
            self.wellClicked.emit(wname)
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
