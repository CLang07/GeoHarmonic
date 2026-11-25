"""Live Map window (PLAN 31b + 31d): Kingdom-style live base map with Spotfire-style well styling, built on the
item-21 map hub (same LayerStack, renderer and Map layers panel) and driven by the map bus.

Layout (Chris's mockup "GeoHarmonic Live Map mockup", dark theme): left = map layers (+ Add... incl. shapefiles);
centre = map with scale bar, north arrow, X / Y / lat-lon / IL-XL readout, current section line; right = Well style +
Filters; bottom = linked well table, crossplot and histogram.  Selections, filters and the active line are shared
with the seismic window, the Well Correlation Tool and the Mistie Tool through the bus (mapbus.py).

The logic lives in `livemap.py` (tested headless); this file only binds widgets.  Methods with plain arguments
(set_color, add_filter, set_range_filter, select_wells, lasso, ...) are what the tests drive under dev/fakeqt.
"""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import livemap as LM
from .. import mapbus as MB
from .basemap import Basemap

DARK_QSS = """
QMainWindow, QWidget { background: #1e2228; color: #d8dde3; }
QDockWidget::title { background: #2a2f37; padding: 4px; }
QTableWidget, QListWidget, QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox { background: #262b32; color: #d8dde3;
  border: 1px solid #3a414b; selection-background-color: #3d6fb6; }
QHeaderView::section { background: #2a2f37; color: #d8dde3; border: 0; padding: 3px; }
QPushButton, QToolButton { background: #2f3540; border: 1px solid #454d59; padding: 3px 8px; }
QPushButton:checked, QToolButton:checked { background: #3d6fb6; }
QGroupBox { border: 1px solid #3a414b; margin-top: 8px; } QGroupBox::title { left: 6px; top: -6px; }
"""
SEL_PEN = (255, 214, 0)
SECTION_PEN = (255, 214, 0)
WELL_SECTION_PEN = (0, 200, 230)


class LiveMapCanvas(Basemap):
    """The hub Basemap with the Live Map overlays (styled wells, scale bar, north arrow, active section)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.skip_kinds = ("wells",)
        self.dark = True
        self._overlay = []
        self._decor = []
        self._mistie_items = []
        self.misties = None
        self.paths = {}  # well -> (x, y) deviated well paths (31d)
        self._section_item = None
        self._well_section_item = None
        self.styled = None
        self.crs = None
        self.view_override = None  # (x0, x1, y0, y1) for tests / saved view before the first paint
        try:
            self.plot.setBackground("#15181d")
            self.plot.showGrid(x=False, y=False)
        except Exception:
            pass
        self.plot.getPlotItem().getViewBox().sigRangeChanged.connect(lambda *_: self.draw_decorations())

    def view(self):
        if self.view_override is not None:
            return tuple(self.view_override)
        try:
            (x0, x1), (y0, y1) = self.plot.getPlotItem().getViewBox().viewRange()
            return float(x0), float(x1), float(y0), float(y1)
        except Exception:
            return None

    # ---------------------------------------------------------------- wells
    def draw_wells(self, styled):
        """Styled well symbols + selection rings + labels in view (styled = livemap.style_wells output)."""
        for it in self._overlay:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._overlay = []
        self.styled = styled
        if styled is None:
            return 0
        vis = np.flatnonzero(styled["visible"])
        if len(vis):
            spots = [{"pos": (float(styled["x"][k]), float(styled["y"][k])), "size": float(styled["size"][k]),
                      "symbol": styled["symbol"][k], "brush": pg.mkBrush(*[int(v) for v in styled["rgba"][k]]),
                      "pen": pg.mkPen((20, 20, 20) if not self.dark else (230, 230, 230), width=0.6),
                      "data": styled["ids"][k]} for k in vis]
            it = pg.ScatterPlotItem(spots=spots)
            it.setZValue(50)
            self.plot.addItem(it)
            self._overlay.append(it)
        sel = np.flatnonzero(styled["visible"] & styled["selected"])
        if len(sel):
            it = pg.ScatterPlotItem([float(styled["x"][k]) for k in sel], [float(styled["y"][k]) for k in sel],
                                    size=[float(styled["size"][k]) + 10 for k in sel], symbol="o", brush=None,
                                    pen=pg.mkPen(SEL_PEN, width=2.5))
            it.setZValue(55)
            self.plot.addItem(it)
            self._overlay.append(it)
        idx = {i: k for k, i in enumerate(styled["ids"])}
        for w, (px, py) in (self.paths or {}).items():  # deviated paths: thin line in the well colour, toe tick
            k = idx.get(w)
            if k is not None and not styled["visible"][k]:
                continue
            c = tuple(int(v) for v in styled["rgba"][k]) if k is not None else (200, 200, 200, 255)
            it = pg.PlotCurveItem(np.asarray(px, float), np.asarray(py, float), pen=pg.mkPen(c, width=1.5))
            it.setZValue(45)
            self.plot.addItem(it)
            self._overlay.append(it)
        for comp_xy in self._pie_items(styled):
            self._overlay.append(comp_xy)
        v = self.view()
        if v is not None:
            for k in LM.labels_in_view(styled, v):
                if styled["label"][k]:
                    t = pg.TextItem(str(styled["label"][k]), color=(230, 230, 230) if self.dark else (20, 20, 20),
                                    anchor=(-0.15, 1.1))
                    t.setPos(float(styled["x"][k]), float(styled["y"][k]))
                    t.setZValue(60)
                    self.plot.addItem(t, ignoreBounds=True)
                    self._overlay.append(t)
        return len(vis)

    def _pie_items(self, styled):
        out = []
        pies = styled.get("pies") or {}
        if not pies:
            return out
        from ..maphub import pie_polygons
        v = self.view()
        if v is None:
            return out
        r = 0.012 * max(abs(v[1] - v[0]), abs(v[3] - v[2]))
        idx = {i: k for k, i in enumerate(styled["ids"])}
        cols = list(next(iter(pies.values())).keys())
        for w, fr in pies.items():
            k = idx.get(w)
            if k is None or not styled["visible"][k]:
                continue
            for comp, xs, ys in pie_polygons(float(styled["x"][k]), float(styled["y"][k]), fr, r):
                c = LM.hex_rgba(LM.PALETTE[cols.index(comp) % len(LM.PALETTE)])
                poly = QtGui.QPolygonF([QtCore.QPointF(float(a), float(b)) for a, b in zip(xs, ys)])
                it = QtWidgets.QGraphicsPolygonItem(poly)
                it.setBrush(pg.mkBrush(*c))
                it.setPen(pg.mkPen((30, 30, 30), width=0))
                it.setZValue(52)
                self.plot.addItem(it)
                out.append(it)
        return out

    # ---------------------------------------------------------------- decorations (31b)
    def draw_decorations(self):
        for it in self._decor:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._decor = []
        v = self.view()
        if v is None or not all(np.isfinite(v)) or v[1] <= v[0] or v[3] <= v[2]:
            return None
        g = LM.scale_bar_geometry(v)
        col = (235, 235, 235) if self.dark else (20, 20, 20)
        h = g["height"]
        for xa, xb, filled in g["segments"]:
            r = QtWidgets.QGraphicsRectItem(xa, g["y"], xb - xa, h)
            r.setBrush(pg.mkBrush(*col) if filled else pg.mkBrush(None))
            r.setPen(pg.mkPen(col, width=1))
            r.setZValue(900)
            self.plot.addItem(r, ignoreBounds=True)
            self._decor.append(r)
        t = pg.TextItem(g["label"], color=col, anchor=(0, 1))
        t.setPos(g["label_xy"][0] + 0.005 * (v[1] - v[0]), g["label_xy"][1] + h)
        t.setZValue(900)
        self.plot.addItem(t, ignoreBounds=True)
        self._decor.append(t)
        ang = LM.north_angle(self.crs, 0.5 * (v[0] + v[1]), 0.5 * (v[2] + v[3]))
        rx, ry, (lx, ly) = LM.north_arrow_geometry(v, ang)
        poly = QtGui.QPolygonF([QtCore.QPointF(float(a), float(b)) for a, b in zip(rx, ry)])
        a = QtWidgets.QGraphicsPolygonItem(poly)
        a.setBrush(pg.mkBrush(*col))
        a.setPen(pg.mkPen(col))
        a.setZValue(900)
        self.plot.addItem(a, ignoreBounds=True)
        n = pg.TextItem("N", color=col, anchor=(0.5, 0.5))
        n.setPos(lx, ly)
        n.setZValue(900)
        self.plot.addItem(n, ignoreBounds=True)
        self._decor += [a, n]
        return {"scale": g, "north": ang}

    def show_misties(self, markers):
        """Mistie tie markers (livemap.mistie_markers) - diamonds coloured by |residual|."""
        for it in self._mistie_items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._mistie_items = []
        self.misties = markers
        if not markers or not len(markers["x"]):
            return 0
        it = pg.ScatterPlotItem(markers["x"], markers["y"], size=12, symbol="d",
                                brush=[pg.mkBrush(*[int(v) for v in c]) for c in markers["rgba"]],
                                pen=pg.mkPen((10, 10, 10)), data=markers["label"])
        it.setZValue(70)
        self.plot.addItem(it)
        self._mistie_items.append(it)
        return len(markers["x"])

    def show_section(self, xy):
        if self._section_item is None:
            self._section_item = pg.PlotCurveItem([], [], pen=pg.mkPen(SECTION_PEN, width=3, style=QtCore.Qt.DashLine))
            self._section_item.setZValue(80)
            self.plot.addItem(self._section_item)
        if not xy:
            self._section_item.setData([], [])
            return 0
        a = np.asarray(xy, float)
        self._section_item.setData(a[:, 0], a[:, 1])
        return len(a)

    def show_well_section(self, xy):
        """Cross-section line of the Well Correlation Tool (cyan, through the section wells)."""
        if self._well_section_item is None:
            self._well_section_item = pg.PlotCurveItem([], [], pen=pg.mkPen(WELL_SECTION_PEN, width=3))
            self._well_section_item.setZValue(81)
            self.plot.addItem(self._well_section_item)
        if not xy:
            self._well_section_item.setData([], [])
            return 0
        a = np.asarray(xy, float)
        self._well_section_item.setData(a[:, 0], a[:, 1])
        return len(a)

    def redraw(self):
        super().redraw()
        self._overlay, self._decor, self._section_item, self._well_section_item = [], [], None, None
        self._mistie_items = []
        if self.styled is not None:
            self.draw_wells(self.styled)
        if self.misties is not None:
            self.show_misties(self.misties)
        self.draw_decorations()


class LiveMapWindow(QtWidgets.QMainWindow):
    """Top-level Live Map.  project = SeisProject; bus = the seismic window's MapBus (shared in-process)."""

    def __init__(self, project, bus=None, parent=None, open_section=None, open_arbitrary=None, depth_model_fn=None,
                 stack=None):
        super().__init__(parent)
        self.setWindowTitle("Live Map - GeoHarmonic")
        self.setObjectName("LiveMap")
        self.project = project
        self.bus = bus or MB.MapBus()
        self.open_section = open_section  # callable(kind, name, mode, number) -> seismic window (optional)
        self.open_arbitrary = open_arbitrary  # callable(name, xs, ys) -> arbitrary section window (optional)
        self.depth_model_fn = depth_model_fn  # callable() -> DynamicDepthModel | None (live depth maps, 31g)
        self.live_depth = False
        self._draw_mode = None  # 'lasso' | 'arbitrary' while drawing on the map
        self._n_arbitrary = 0
        self.state = LM.MapState.load(project)
        self.styled = None
        self.messages = []
        self._lasso = False
        self._filter_widgets = {}
        self._table_ids = []
        self.setStyleSheet(DARK_QSS)
        self.canvas = LiveMapCanvas()
        self.setCentralWidget(self.canvas)
        try:
            from ..gis.export import project_crs
            self.canvas.crs = project_crs(project)
        except Exception:
            self.canvas.crs = None
        from .layers_dock import LayersPanel
        self.layers = LayersPanel(self.canvas)
        self._dock("Map layers", self.layers, QtCore.Qt.LeftDockWidgetArea)
        self._build_style_panel()
        self._build_filters_panel()
        self._build_bottom()
        self.readout = QtWidgets.QLabel("")
        self.statusBar().addWidget(self.readout, 1)
        tb = self.addToolBar("Live Map")
        tb.setObjectName("LiveMapTools")
        self.act_lasso = QtGui.QAction("Lasso select", self)
        self.act_lasso.setCheckable(True)
        self.act_lasso.triggered.connect(lambda *_: self.start_lasso())
        tb.addAction(self.act_lasso)
        self.act_arb = QtGui.QAction("Arbitrary line", self)
        self.act_arb.setToolTip("Click points along a line, double-click to open the arbitrary section")
        self.act_arb.triggered.connect(lambda *_: self.start_arbitrary())
        tb.addAction(self.act_arb)
        self.act_ws = QtGui.QAction("Well section", self)
        self.act_ws.setToolTip("Draw a line: the wells along it become a cross section in the Well Correlation Tool")
        self.act_ws.triggered.connect(lambda *_: self.start_well_section())
        tb.addAction(self.act_ws)
        a = QtGui.QAction("Section from selected wells", self)
        a.triggered.connect(lambda *_: self.well_section_from_selection())
        tb.addAction(a)
        a = QtGui.QAction("Live grid...", self)
        a.setToolTip("Grid a tops column or a horizon live (coarse while picking, full on pause; faults as barriers)")
        a.triggered.connect(lambda *_: self._live_grid_dialog())
        tb.addAction(a)
        self.act_depth = QtGui.QAction("Live depth maps", self)
        self.act_depth.setCheckable(True)
        self.act_depth.triggered.connect(lambda *_: self.set_live_depth(self.act_depth.isChecked()))
        tb.addAction(self.act_depth)
        self.act_paths = QtGui.QAction("Well paths", self)
        self.act_paths.setCheckable(True)
        self.act_paths.setChecked(True)
        self.act_paths.triggered.connect(lambda *_: self.show_paths(self.act_paths.isChecked()))
        tb.addAction(self.act_paths)
        self.act_sp = QtGui.QAction("Shotpoints", self)
        self.act_sp.setCheckable(True)
        self.act_sp.triggered.connect(lambda *_: self.set_shotpoints(100 if self.act_sp.isChecked() else 0))
        tb.addAction(self.act_sp)
        for text, fn in (("Clear selection", lambda *_: self.select_wells([], "replace")),
                         ("Clear filters", lambda *_: self.clear_filters()),
                         ("Save view", lambda *_: self.save_state()),
                         ("Export map (PNG / PDF)...", lambda *_: self._export_map_dialog()),
                         ("Export GIS (shapefile, GeoJSON, ZMAP+)...", lambda *_: self._export_gis_dialog())):
            a = QtGui.QAction(text, self)
            a.triggered.connect(fn)
            tb.addAction(a)
        # canvas signals
        self.canvas.cursorMoved.connect(lambda m: self.cursor_moved(m.get("x"), m.get("y")))
        self.canvas.wellClicked.connect(lambda n: self.well_clicked(n))
        self.canvas.polygonDrawn.connect(lambda xy: self.drawn(xy[0], xy[1]))
        self.canvas.openLine.connect(lambda n: self._open("2D", n))
        self.canvas.openInline.connect(lambda v, il, xl: self._open("3D", v, "Inline", il))
        self.canvas.mapClicked.connect(lambda x, y, tol: self.map_clicked(x, y, tol))
        self.bus.subscribe("livemap", self._on_event)
        # map content
        from ..maplayers import LayerStack
        # share the seismic window's layer stack (one set of map layers; edits in either window are seen by both)
        self.canvas.stack = stack if stack is not None else LayerStack.load(project)
        self.layers.set_stack(self.canvas.stack)
        self.canvas.set_project(project)
        self.canvas.paths = LM.well_paths(project)
        from ..livegrid import LiveGridder  # 31g: coarse grid while picking, full grid on pause
        self.gridder = LiveGridder(self.canvas.stack, self.bus.table, project)
        self._grid_timer = QtCore.QTimer(self)
        self._grid_timer.setInterval(300)
        self._grid_timer.timeout.connect(lambda *_: self.grid_tick())
        self._grid_timer.start()
        self.restyle()
        sec = self.bus.active.info("section")
        if sec.get("xy"):
            self.canvas.show_section(sec["xy"])
        ws = self.bus.active.info("well_section")
        if ws.get("xy"):
            self.canvas.show_well_section(ws["xy"])
        last = self.bus.last("data", "misties")
        if last is not None:
            self.canvas.show_misties(LM.mistie_markers(last))
        if self.state.view:
            try:
                x0, x1, y0, y1 = self.state.view
                self.canvas.plot.getPlotItem().getViewBox().setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
            except Exception:
                pass
        self.resize(1500, 950)

    def _dock(self, title, w, area):
        d = QtWidgets.QDockWidget(title, self)
        d.setObjectName(title)
        d.setWidget(w)
        self.addDockWidget(area, d)
        return d

    # ---------------------------------------------------------------- panels
    def _build_style_panel(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.c_mode = QtWidgets.QComboBox()
        self.c_mode.addItems(["fixed", "continuous", "categorical", "rules"])
        self.c_col = QtWidgets.QComboBox()
        self.c_ramp = QtWidgets.QComboBox()
        self.c_ramp.addItems(list(LM.RAMPS))
        self.s_col = QtWidgets.QComboBox()
        self.sh_col = QtWidgets.QComboBox()
        self.l_col = QtWidgets.QComboBox()
        self.rules_edit = QtWidgets.QLineEdit()
        self.rules_edit.setPlaceholderText("Mistie_ms > 10 = #ff0000; [Net pay] > 5 = #00aa00")
        for lab, wid in (("Colour by", self.c_mode), ("Column", self.c_col), ("Ramp", self.c_ramp),
                         ("Rules", self.rules_edit), ("Size by", self.s_col), ("Shape by", self.sh_col),
                         ("Label", self.l_col)):
            f.addRow(lab, wid)
        b = QtWidgets.QPushButton("Apply style")
        b.clicked.connect(lambda *_: self.apply_style_widgets())
        f.addRow(b)
        self.legend_label = QtWidgets.QLabel("")
        self.legend_label.setWordWrap(True)
        f.addRow(self.legend_label)
        self.dock_style = self._dock("Well style", w, QtCore.Qt.RightDockWidgetArea)

    def _build_filters_panel(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        self.f_col = QtWidgets.QComboBox()
        bt = QtWidgets.QPushButton("Add filter")
        bt.clicked.connect(lambda *_: self.add_filter(self.f_col.currentText()))
        row.addWidget(self.f_col, 1)
        row.addWidget(bt)
        v.addLayout(row)
        self.filters_box = QtWidgets.QVBoxLayout()
        v.addLayout(self.filters_box)
        v.addStretch(1)
        self.filter_info = QtWidgets.QLabel("")
        v.addWidget(self.filter_info)
        self.dock_filters = self._dock("Filters", w, QtCore.Qt.RightDockWidgetArea)

    def _build_bottom(self):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        self.table = QtWidgets.QTableWidget()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.itemSelectionChanged.connect(lambda *_: self._table_selection())
        h.addWidget(self.table, 3)
        side = QtWidgets.QVBoxLayout()
        row = QtWidgets.QHBoxLayout()
        self.x_col = QtWidgets.QComboBox()
        self.y_col = QtWidgets.QComboBox()
        for c in (self.x_col, self.y_col):
            c.currentTextChanged.connect(lambda *_: self.update_crossplot())
            row.addWidget(c)
        side.addLayout(row)
        self.xplot = pg.PlotWidget()
        self.xplot.setBackground("#15181d")
        side.addWidget(self.xplot, 1)
        self.hist = pg.PlotWidget()
        self.hist.setBackground("#15181d")
        side.addWidget(self.hist, 1)
        h.addLayout(side, 2)
        self.dock_table = self._dock("Wells", w, QtCore.Qt.BottomDockWidgetArea)

    # ---------------------------------------------------------------- styling
    def table_columns(self):
        return list(self.bus.table.cols)

    def _fill_column_combos(self):
        cols = self.table_columns()
        num = self.bus.table.columns("numeric")
        for cb, items in ((self.c_col, cols), (self.s_col, [""] + num), (self.sh_col, [""] + cols),
                          (self.l_col, ["", "Well"] + cols), (self.f_col, cols), (self.x_col, num), (self.y_col, num)):
            cur = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(items)
            if cur in items:
                cb.setCurrentText(cur)
            cb.blockSignals(False)

    def set_color(self, mode, column=None, ramp=None, rules=None, categories=None, fixed=None):
        cs = self.state.well_style["color"]
        cs["mode"] = mode
        if column is not None:
            cs["column"] = column
        if ramp:
            cs["ramp"] = ramp
        if rules is not None:
            cs["rules"] = [dict(r) for r in rules]
        if categories is not None:
            cs["categories"] = dict(categories)
        if fixed:
            cs["fixed"] = fixed
        return self.restyle()

    def set_size(self, column=None, min_px=None, max_px=None):
        ss = self.state.well_style["size"]
        ss["column"] = column or None
        if min_px is not None:
            ss["min_px"] = float(min_px)
        if max_px is not None:
            ss["max_px"] = float(max_px)
        return self.restyle()

    def set_shape(self, column=None):
        self.state.well_style["shape"]["column"] = column or None
        return self.restyle()

    def set_label(self, column):
        self.state.well_style["label"]["column"] = column or None
        return self.restyle()

    def set_pies(self, columns):
        self.state.well_style["pie"]["columns"] = list(columns or [])
        return self.restyle()

    @staticmethod
    def parse_rules(text):
        """'expr = #rrggbb; expr2 = red' -> [{rule, color}] (colour names: red green blue orange grey ...)."""
        names = {"red": "#e15759", "green": "#59a14f", "blue": "#4e79a7", "orange": "#f28e2b", "yellow": "#edc948",
                 "grey": "#bab0ac", "gray": "#bab0ac", "purple": "#b07aa1"}
        out = []
        for part in str(text or "").split(";"):
            if "=" not in part.replace("==", "").replace(">=", "").replace("<=", "").replace("!=", ""):
                continue
            k = part.rfind("=")
            rule, col = part[:k].strip(), part[k + 1:].strip()
            if rule.endswith(("<", ">", "!", "=")):
                continue
            MB.parse_rule(rule)
            out.append({"rule": rule, "color": names.get(col.lower(), col)})
        return out

    def apply_style_widgets(self):
        mode = self.c_mode.currentText()
        rules = self.parse_rules(self.rules_edit.text()) if mode == "rules" else None
        self.state.well_style["size"]["column"] = self.s_col.currentText() or None
        self.state.well_style["shape"]["column"] = self.sh_col.currentText() or None
        self.state.well_style["label"]["column"] = self.l_col.currentText() or None
        return self.set_color(mode, self.c_col.currentText() or None, self.c_ramp.currentText(), rules)

    def restyle(self):
        """Recompute and redraw the styled wells, legend, table, crossplot and histogram."""
        t = self.bus.table
        if not len(t):
            try:
                MB.table_from_wells(self.project.wells(), t)
            except Exception:
                pass
        self._fill_column_combos()
        try:
            self.styled = LM.style_wells(t, self.state.well_style, self.bus.filters, self.bus.selection.get("wells"))
        except Exception as e:  # a style column that no longer exists: fall back to the default style
            self._say(f"Well style: {e} - default style used")
            self.state.well_style = LM.merge_style({})
            self.styled = LM.style_wells(t, self.state.well_style, self.bus.filters, self.bus.selection.get("wells"))
        self.canvas.draw_wells(self.styled)
        self.legend_label.setText(self.legend_html())
        nvis = int(self.styled["visible"].sum())
        self.filter_info.setText(f"{nvis} of {len(t)} wells shown")
        self.update_table()
        self.update_crossplot()
        self.update_histogram()
        return self.styled

    def legend_html(self):
        if self.styled is None:
            return ""
        leg = self.styled["legend"]
        rows = [f"<span style='color:{LM.rgba_hex(c)}'>&#9679;</span> {lab}" for lab, c in leg["color"]]
        if leg["size"]:
            rows.append(f"size: {leg['size'][0]} {leg['size'][1]:g} - {leg['size'][2]:g}")
        rows += [f"{s} = {lab}" for lab, s in leg["shape"]]
        return "<br>".join(rows)

    # ---------------------------------------------------------------- filters (bus FilterService)
    def add_filter(self, column):
        spec = LM.filter_widget_spec(self.bus.table, column)
        if spec is None or column in self._filter_widgets:
            return None
        box = QtWidgets.QGroupBox(column)
        lay = QtWidgets.QVBoxLayout(box)
        if spec[0] == "range":
            lo, hi = spec[1], spec[2]
            row = QtWidgets.QHBoxLayout()
            a, b = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
            for s, v in ((a, lo), (b, hi)):
                s.setDecimals(3)
                s.setRange(lo - abs(hi - lo) - 1, hi + abs(hi - lo) + 1)
                s.setValue(v)
                s.editingFinished.connect(lambda *_, c=column, a=a, b=b: self.set_range_filter(c, a.value(), b.value()))
                row.addWidget(s)
            lay.addLayout(row)
            wid = ("range", a, b)
        else:
            lst = QtWidgets.QListWidget()
            for c in spec[1]:
                it = QtWidgets.QListWidgetItem(c or "(blank)")
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.Checked)
                it.setData(QtCore.Qt.UserRole, c)
                lst.addItem(it)
            lst.itemChanged.connect(lambda *_, c=column, l=lst: self._category_changed(c, l))
            lay.addWidget(lst)
            wid = ("values", lst, list(spec[1]))
        rm = QtWidgets.QPushButton("Remove")
        rm.clicked.connect(lambda *_, c=column: self.remove_filter(c))
        lay.addWidget(rm)
        self.filters_box.addWidget(box)
        self._filter_widgets[column] = (box, wid)
        if column not in self.state.filter_columns:
            self.state.filter_columns.append(column)
        return spec

    def _category_changed(self, column, lst):
        vals = []
        for k in range(lst.count()):
            it = lst.item(k)
            if it.checkState() == QtCore.Qt.Checked:
                vals.append(it.data(QtCore.Qt.UserRole))
        self.set_category_filter(column, vals)

    def set_range_filter(self, column, lo=None, hi=None):
        self.bus.filters.set_range(column, lo, hi, source="livemap")
        return self.restyle()

    def set_category_filter(self, column, values):
        self.bus.filters.set_values(column, values, source="livemap")
        return self.restyle()

    def set_rule_filter(self, name, expr):
        self.bus.filters.set_rule(name, expr, source="livemap")
        return self.restyle()

    def remove_filter(self, column):
        w = self._filter_widgets.pop(column, None)
        if w is not None:
            try:
                w[0].setParent(None)
                w[0].deleteLater()
            except Exception:
                pass
        if column in self.state.filter_columns:
            self.state.filter_columns.remove(column)
        self.bus.filters.remove(column, source="livemap")
        return self.restyle()

    def clear_filters(self):
        for c in list(self._filter_widgets):
            w = self._filter_widgets.pop(c)
            try:
                w[0].setParent(None)
                w[0].deleteLater()
            except Exception:
                pass
        self.state.filter_columns = []
        self.bus.filters.clear(source="livemap")
        return self.restyle()

    # ---------------------------------------------------------------- selection
    def select_wells(self, ids, mode="replace"):
        self.bus.selection.select("wells", list(ids or []), mode=mode, source="livemap")
        return self.restyle()

    def well_clicked(self, name):
        mods = QtWidgets.QApplication.keyboardModifiers() if hasattr(QtWidgets.QApplication, "keyboardModifiers") else 0
        try:
            add = bool(mods & QtCore.Qt.ControlModifier)
        except TypeError:
            add = False
        self.select_wells([name], "toggle" if add else "replace")
        self.bus.active.set("well", name, source="livemap")
        return name

    def start_lasso(self):
        self._lasso = True
        self._draw_mode = "lasso"
        self.canvas.min_poly_points = 3
        self.canvas.start_polygon()
        self.statusBar().showMessage("Lasso: click around the wells, double-click to close", 8000)

    def drawn(self, xs, ys):
        """A polygon / polyline finished on the canvas: lasso selection or arbitrary line."""
        if self._draw_mode == "arbitrary":
            return self.arbitrary(xs, ys)
        if self._draw_mode == "well_section":
            return self.well_section_along(xs, ys)
        return self.lasso(xs, ys)

    def start_well_section(self):
        self._draw_mode = "well_section"
        self.canvas.min_poly_points = 2
        self.canvas.start_polygon()
        self.statusBar().showMessage("Well section: click along the wells, double-click the last point", 8000)

    def well_section_along(self, xs, ys, buffer=None):
        """Wells near the drawn line, in order along it -> cross section (PLAN 31f)."""
        self._draw_mode = None
        self.canvas.min_poly_points = 3
        hits = LM.wells_along_line(self.bus.table, xs, ys, buffer, self.styled["visible"] if self.styled else None)
        return self.publish_well_section([h[0] for h in hits])

    def well_section_from_selection(self):
        return self.publish_well_section(self.bus.selection.get("wells"))

    def publish_well_section(self, wells, name="Live Map"):
        wells = list(wells or [])
        if len(wells) < 1:
            self._say("Well section: no wells on the line")
            return []
        xy = LM.section_xy(self.bus.table, wells)
        self.bus.active.set("well_section", name, source="livemap", wells=wells, xy=xy, force=True)
        self.bus.selection.select("wells", wells, source="livemap")
        self.canvas.show_well_section(xy)
        self.restyle()
        self._say(f"Cross section '{name}': {', '.join(wells)} (opens in the Well Correlation Tool)")
        return wells

    def start_arbitrary(self):
        self._draw_mode = "arbitrary"
        self.canvas.min_poly_points = 2
        self.canvas.start_polygon()
        self.statusBar().showMessage("Arbitrary line: click the points, double-click the last one", 8000)

    def arbitrary(self, xs, ys, name=None):
        """Open an arbitrary section along the drawn line (PLAN 31e) and show it as the active section."""
        self._draw_mode = None
        self.canvas.min_poly_points = 3
        self._n_arbitrary += 1
        name = name or f"Arbitrary {self._n_arbitrary}"
        xy = [[float(a), float(b)] for a, b in zip(xs, ys)]
        self.bus.active.set("section", name, source="livemap", xy=xy, kind2="arbitrary", name=name)
        self.canvas.show_section(xy)
        dlg = None
        if self.open_arbitrary is not None:
            try:
                dlg = self.open_arbitrary(name, list(xs), list(ys))
            except Exception as e:
                self._say(f"Arbitrary line failed: {e}")
        return dlg if dlg is not None else xy

    # ---------------------------------------------------------------- live gridding (31g)
    def add_live_grid(self, name, kind="column", column=None, horizon=None, key=None, cell=None, method="tps",
                      barriers=True, interval=None):
        self.gridder.add_source(name, kind, column=column, horizon=horizon, key=key, cell=cell, method=method,
                                barriers=barriers, interval=interval)
        self._grids_changed()
        return self.gridder.sources[name]["lid"]

    def grid_tick(self):
        applied = self.gridder.tick()
        if applied:
            self._grids_changed()
            n, level, sec = self.gridder.done[-1]
            self._say(f"Live grid {n}: {level} grid in {sec:.2f} s")
        if self.gridder.errors:
            self._say(self.gridder.errors.pop())
        return applied

    def set_live_depth(self, on=True):
        self.live_depth = bool(on)
        return self.refresh_depth() if self.live_depth else []

    def refresh_depth(self):
        """Depth maps of the current dynamic depth model as 'Live depth: <horizon>' layers."""
        dm = self.depth_model_fn() if self.depth_model_fn is not None else None
        if dm is None:
            self._say("Live depth maps: define a depth model first (Interpret > Velocity model)")
            return []
        from ..livegrid import depth_grids
        ids = depth_grids(self.canvas.stack, dm)
        self._grids_changed()
        return ids

    def _grids_changed(self):
        try:
            self.canvas.stack.save()
            self.canvas.draw_layers()
            self.layers.refresh()
        except Exception as e:
            self._say(f"Map redraw: {e}")

    def _live_grid_dialog(self):
        t = self.bus.table
        cols = [c for c in t.columns("numeric") if c not in ("X", "Y")]
        hs = [h.name for h in self.project.horizons.items] if getattr(self.project, "horizons", None) else []
        items = [f"Column: {c}" for c in cols] + [f"Horizon: {h}" for h in hs]
        if not items:
            self._say("Live grid: no numeric WellTable column or horizon yet")
            return None
        it, ok = QtWidgets.QInputDialog.getItem(self, "Live grid", "Grid:", items, 0, False)
        if not ok:
            return None
        if it.startswith("Column: "):
            c = it[8:]
            return self.add_live_grid(c, "column", column=c)
        h = it[9:]
        key = next((k for k in self.project.horizons.get(h).data if k.startswith("3D:")), "2D")
        return self.add_live_grid(h, "horizon", horizon=h, key=key)

    # ---------------------------------------------------------------- other modules (31h) + output (31i)
    def add_layer_request(self, req):
        """A layer another module registered through the map bus (basin / Wheeler section, inversion map ...)."""
        try:
            lid = LM.apply_layer_request(self.canvas.stack, req)
        except Exception as e:
            self._say(f"Layer from {req.get('owner') or '?'} not added: {e}")
            return None
        self._grids_changed()
        self._say(f"Layer '{req.get('name')}' from {req.get('owner') or 'another tool'}")
        return lid

    def map_clicked(self, x, y, tol):
        """Click on a registered section line (basin / Wheeler, arbitrary ...) opens it."""
        hit = LM.section_hit(self.canvas.stack, x, y, tol) if self.canvas.stack is not None else None
        if hit is None:
            return None
        lid, item = hit
        op = item.get("action") if isinstance(item.get("action"), dict) else {}
        self.bus.active.set("section", item.get("name") or "section", source="livemap",
                            xy=[[a, b] for a, b in zip(item["x"], item["y"])], kind2=op.get("kind2", "registered"),
                            name=item.get("name"), open=op)
        if op.get("kind") in ("2D", "3D") and self.open_section is not None:
            self.open_section(op["kind"], op.get("name"), op.get("mode"), op.get("number"))
        elif self.open_arbitrary is not None:
            self.open_arbitrary(item.get("name") or "section", list(item["x"]), list(item["y"]))
        return item

    def export_map(self, path, title="Live Map", **kw):
        self.restyle()
        return LM.export_map(self.canvas.stack, self.styled, path, title=title, **kw)

    def export_gis(self, folder):
        secs = {}
        for k in ("section", "well_section"):
            info = self.bus.active.info(k)
            if info.get("xy"):
                secs[f"{k}: {info.get('value')}"] = info["xy"]
        return LM.export_gis(self.canvas.stack, self.bus.table, folder, self.styled, secs, LM.picks_2d(self.project))

    def _export_map_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export map", "", "PNG (*.png);;PDF (*.pdf)")
        if path:
            try:
                out = self.export_map(path)
                self._say(f"Map written: {out['path']}")
            except Exception as e:
                self._say(f"Export failed: {e}")

    def _export_gis_dialog(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Export GIS layers to folder")
        if folder:
            try:
                out = self.export_gis(folder)
                self._say(f"{len(out['files'])} files written to {folder}" +
                          (f" - {out['warnings'][0]}" if out.get("warnings") else ""))
            except Exception as e:
                self._say(f"Export failed: {e}")

    def show_paths(self, on=True, paths=None):
        """Deviated well paths on / off (paths = {well: (x, y)} to set them directly)."""
        self.canvas.paths = (paths if paths is not None else LM.well_paths(self.project)) if on else {}
        self.canvas.draw_wells(self.styled)
        return len(self.canvas.paths)

    def set_shotpoints(self, every=100):
        """CDP / shotpoint annotation on the 2D lines every N (0 = off)."""
        st = self.canvas.stack
        if st is None:
            return 0
        n = 0
        for l in st.layers:
            if l["kind"] == "lines2d":
                st.set_style(l["id"], shotpoints=int(every) or None)
                n += 1
        st.save()
        self.canvas.draw_layers()
        return n

    def lasso(self, xs, ys, mode="replace"):
        self._lasso = False
        try:
            self.act_lasso.setChecked(False)
        except Exception:
            pass
        ids = LM.lasso_select(self.bus.table, list(zip(xs, ys)), self.styled["visible"] if self.styled else None)
        self.select_wells(ids, mode)
        self._say(f"Lasso: {len(ids)} wells selected")
        return ids

    def _table_selection(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}) if hasattr(self.table, "selectedIndexes") else []
        ids = [self._table_ids[r] for r in rows if 0 <= r < len(self._table_ids)]
        if ids and ids != self.bus.selection.get("wells"):
            self.select_wells(ids)

    # ---------------------------------------------------------------- linked views
    def update_table(self, sort=None):
        cols, rows = LM.table_rows(self.bus.table, self.state.table_columns or None,
                                   self.styled["visible"] if self.styled else None, self.bus.selection.get("wells"),
                                   sort=sort)
        self._table_ids = [r[0] for r in rows]
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setColumnCount(len(cols) + 1)
        self.table.setHorizontalHeaderLabels(["Well"] + cols)
        self.table.setRowCount(len(rows))
        sel = {MB.norm_id(s) for s in self.bus.selection.get("wells")}
        for r, (i, vals) in enumerate(rows):
            for c, v in enumerate([i] + vals):
                it = QtWidgets.QTableWidgetItem(str(v))
                if MB.norm_id(i) in sel:
                    it.setBackground(QtGui.QColor(70, 90, 40))
                self.table.setItem(r, c, it)
        self.table.blockSignals(False)
        return rows

    def update_crossplot(self, xcol=None, ycol=None):
        xcol = xcol or self.x_col.currentText() or self.state.crossplot.get("x")
        ycol = ycol or self.y_col.currentText() or self.state.crossplot.get("y")
        try:
            self.xplot.clear()
        except Exception:
            pass
        if not xcol or not ycol or xcol not in self.bus.table.cols or ycol not in self.bus.table.cols:
            return None
        self.state.crossplot = {"x": xcol, "y": ycol}
        d = LM.crossplot_data(self.bus.table, xcol, ycol, self.styled)
        if len(d["ids"]):
            br = [pg.mkBrush(*[int(v) for v in c]) for c in d["rgba"]] if d["rgba"] is not None else None
            it = pg.ScatterPlotItem(d["x"], d["y"], size=8, brush=br, data=d["ids"])
            it.sigClicked.connect(lambda _it, pts, *_: self.select_wells([p.data() for p in pts]))
            self.xplot.addItem(it)
            s = d["selected"]
            if s.any():
                self.xplot.addItem(pg.ScatterPlotItem(d["x"][s], d["y"][s], size=14, brush=None,
                                                      pen=pg.mkPen(SEL_PEN, width=2)))
        try:
            self.xplot.setLabel("bottom", xcol)
            self.xplot.setLabel("left", f"{ycol}   r = {d['r']:.2f}" if np.isfinite(d["r"]) else ycol)
        except Exception:
            pass
        return d

    def update_histogram(self, column=None):
        col = column or self.state.histogram or (self.state.filter_columns[-1] if self.state.filter_columns else None)
        try:
            self.hist.clear()
        except Exception:
            pass
        if not col or col not in self.bus.table.cols:
            return None
        self.state.histogram = col
        h = LM.histogram_data(self.bus.table, col, self.bus.filters, self.bus.selection.get("wells"))
        if "edges" in h:
            e = h["edges"]
            self.hist.addItem(pg.BarGraphItem(x0=e[:-1], x1=e[1:], height=h["counts"], brush=(78, 121, 167)))
            self.hist.addItem(pg.BarGraphItem(x0=e[:-1], x1=e[1:], height=h["selected"], brush=SEL_PEN))
        else:
            x = np.arange(len(h["counts"]))
            self.hist.addItem(pg.BarGraphItem(x=x, width=0.8, height=h["counts"], brush=(78, 121, 167)))
            self.hist.addItem(pg.BarGraphItem(x=x, width=0.8, height=h["selected"], brush=SEL_PEN))
        return h

    # ---------------------------------------------------------------- bus in
    def _on_event(self, ev):
        t = ev["topic"]
        if t in ("selection", "filter"):
            self.restyle()
        elif t == "data" and ev.get("kind") == "misties":
            n = self.canvas.show_misties(LM.mistie_markers(ev))
            if n:
                self._say(f"{n} mistie ties from {ev.get('role') or ev.get('source')}")
        elif t == "active" and ev.get("kind") == "section":
            self.canvas.show_section(ev.get("xy") or [])
        elif t == "active" and ev.get("kind") == "well_section":
            self.canvas.show_well_section(ev.get("xy") or [])
        elif t == "data" and ev.get("kind") in ("welltable", "tops", "wells", "misties", "velocities"):
            self.restyle()
            self._say(f"{ev.get('kind')} updated by {ev.get('role') or ev.get('source')}")
            if ev.get("kind") in ("welltable", "tops"):
                self.gridder.changed_column(None)  # tops pushed: coarse re-grid now, full on pause
            if ev.get("kind") == "velocities" and self.live_depth:
                self.refresh_depth()
        elif t == "data" and ev.get("kind") == "layers" and ev.get("layer"):
            self.add_layer_request(ev["layer"])
        elif t == "data" and ev.get("kind") == "picks":
            self.gridder.changed_horizon(ev.get("ids") or None)
        elif t == "data" and ev.get("kind") in ("grids", "layers"):
            try:
                self.canvas.draw_layers()
                self.layers.refresh()
            except Exception:
                pass

    def cursor_moved(self, x, y):
        if x is None:
            return ""
        vol = None
        try:
            if self.project.volumes:
                vol = self.project.volume(self.project.volumes[0]["name"])
        except Exception:
            vol = None
        txt = LM.cursor_text(x, y, self.canvas.crs, vol)
        self.readout.setText(txt)
        return txt

    def _open(self, kind, name, mode=None, number=None):
        if kind == "2D":
            self.bus.active.set("line", name, source="livemap")
        else:
            self.bus.active.set(mode.lower() if mode else "inline", number, source="livemap", volume=name)
        if self.open_section is not None:
            try:
                self.open_section(kind, name, mode, number)
            except Exception as e:
                self._say(f"Could not open {name}: {e}")

    def save_state(self):
        v = self.canvas.view()
        if v is not None:
            self.state.view = [float(a) for a in v]
        self.state.save(self.project)
        try:
            if self.project.path:
                self.project.save()
        except Exception:
            pass
        return self.state.to_dict()

    def _say(self, msg):
        self.messages.append(msg)
        del self.messages[:-50]
        try:
            self.statusBar().showMessage(msg, 6000)
        except Exception:
            pass

    def closeEvent(self, ev):
        try:
            self.save_state()
        except Exception:
            pass
        self.bus.unsubscribe("livemap")
        try:
            self._grid_timer.stop()
        except Exception:
            pass
        ev.accept()
