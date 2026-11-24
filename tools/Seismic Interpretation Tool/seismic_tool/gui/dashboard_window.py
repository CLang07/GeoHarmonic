"""Dashboard (PLAN item 14): linked map / histogram / crossplot / table panels over the wells, with laterals drawn
segment-by-segment coloured by an attribute along MD (TVDSS, map grid, seismic amplitude via the well's
time-depth, stage CSV values, in-zone flag).  State lives in a DashboardModel; the non-drawing logic (load,
colour, stats, selection, export) is testable without a display."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtWidgets  # noqa: E402
import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from .. import laterals as L  # noqa: E402
from ..dashboard_model import DashboardModel  # noqa: E402

N_BINS = 16


class DashboardWindow(QtWidgets.QMainWindow):
    MAX_TABLE_ROWS = 3000
    def __init__(self, project=None, basemap=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dashboard - wells and laterals")
        self.project, self.basemap = project, basemap
        self.trajs = []
        self.grids = {}  # name -> MapGrid (attributes, depth surfaces for zones)
        self.stages = {}  # well id -> [(top, base, {col: value})]
        self.color_name, self.color_values, self.vrange = None, {}, (0.0, 1.0)
        self.zone = None  # (top grid name, base grid name)
        self.analytics_window = None  # set by the main window when Analytics is open
        self._ws = None
        self._table_rows = []
        self._log_names = None
        self.background = None  # grid name drawn under the wells
        self.model = DashboardModel()
        self.model.subscribe(self._on_model)
        self._items = []
        self._build_ui()

    # ------------------------------------------------------------ UI
    def _build_ui(self):
        self.map = pg.PlotWidget()
        self.map.setAspectLocked(True)
        self.map.setBackground("w")
        self.hist = pg.PlotWidget()
        self.hist.setBackground("w")
        self.xplot = pg.PlotWidget()
        self.xplot.setBackground("w")
        self.table = QtWidgets.QTableWidget()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        for title, w, area in (("Map", self.map, QtCore.Qt.LeftDockWidgetArea),
                               ("Histogram", self.hist, QtCore.Qt.RightDockWidgetArea),
                               ("Crossplot", self.xplot, QtCore.Qt.RightDockWidgetArea),
                               ("Table", self.table, QtCore.Qt.BottomDockWidgetArea)):
            d = QtWidgets.QDockWidget(title, self)
            d.setObjectName(title)
            d.setWidget(w)
            self.addDockWidget(area, d)
        tb = self.addToolBar("Dashboard")
        self.color_box = QtWidgets.QComboBox()
        self.hist_box = QtWidgets.QComboBox()
        self.x_box = QtWidgets.QComboBox()
        self.y_box = QtWidgets.QComboBox()
        self.bg_box = QtWidgets.QComboBox()
        for lab, w in (("Background", self.bg_box), ("Colour laterals by", self.color_box), ("Histogram", self.hist_box), ("X", self.x_box),
                       ("Y", self.y_box)):
            tb.addWidget(QtWidgets.QLabel(f" {lab} "))
            tb.addWidget(w)
        for text, fn in (("Wells from project", self.load_project_wells), ("Add basemap grid", self._add_basemap_grid),
                         ("Stage CSV...", self._stages_dialog), ("Lateral stats", self.add_lateral_stats), ("Well strip", lambda: self.open_strip()),
                         ("Send to Analytics", self.send_to_analytics), ("Filter by text...", self._text_filter_dialog), ("Clear filters", lambda: self.model.clear_filters()),
                         ("Save layout...", self._save_layout), ("Load layout...", self._load_layout),
                         ("Export HTML...", self._export_dialog),
                         ("Spacing + ellipses...", self._spacing_dialog), ("Gun barrel", lambda: self.gun_barrel_view())):
            a = tb.addAction(text)
            a.triggered.connect(lambda *_, fn=fn: fn())  # never pass 'checked' into optional parameters
        self.color_box.textActivated.connect(lambda s: s and self.color_by(s))  # also re-picking the shown entry
        self.bg_box.currentTextChanged.connect(lambda s: self.set_background(s))
        self.hist_box.currentTextChanged.connect(lambda s: self.draw_histogram())
        self.x_box.currentTextChanged.connect(lambda s: self.draw_crossplot())
        self.y_box.currentTextChanged.connect(lambda s: self.draw_crossplot())
        self.table.itemSelectionChanged.connect(self._table_selected)
        self.map.scene().sigMouseClicked.connect(self._map_clicked)
        self.status = QtWidgets.QLabel("")
        self.statusBar().addWidget(self.status)
        self.resize(1300, 850)

    # ------------------------------------------------------------ data
    def set_trajectories(self, trajs, extra=None):
        """Wells / laterals to show; extra: AttributeTable joined by API / name (engineering, production...)."""
        self.trajs = list(trajs)
        self._log_names = None
        self._pick_key = None
        tab = L.stats_table(self.trajs, {})
        if extra is not None:
            tab = tab.join(extra)
        self.model = DashboardModel(tab)
        self.model.subscribe(self._on_model)
        self._refresh_boxes()
        self.redraw()
        return self.model

    def load_project_wells(self):
        ws = self.project.wellset() if self.project is not None and self.project.well_project else None
        if ws is None or ws.wcp is None:
            self.status.setText("Link a well project first (File > Link well project).")
            return []
        self._ws = ws
        return self.set_trajectories(L.trajectories_from_wellset(ws))

    def add_grid(self, name, grid):
        self.grids[name] = grid
        self._refresh_boxes()

    def _add_basemap_grid(self):
        g = getattr(self.basemap, "_grid", (None, None))[0] if self.basemap is not None else None
        if g is not None:
            self.add_grid(g.label or f"grid {len(self.grids) + 1}", g)

    def load_stages(self, path):
        self.stages = L.read_stages_csv(path)
        self._refresh_boxes()
        return self.stages

    def _stages_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Stage / completion CSV", "", "CSV (*.csv)")
        if path:
            self.load_stages(path)

    def color_sources(self):
        out = ["TVDSS"] + [f"grid: {n}" for n in self.grids]
        cols = sorted({c for st in self.stages.values() for _, _, v in st for c in v})
        out += [f"stage: {c}" for c in cols]
        if self.project is not None and getattr(self.project, "volumes", None) and getattr(self, "_ws", None):
            out += [f"seismic: {v['name']}" for v in self.project.volumes]
        out += [f"log: {c}" for c in self.log_names()]
        if len(self.grids) >= 2:
            out.append("in zone")
        return out

    def well_logs(self, name):
        """{curve: (md, values)} of a project well (loaded on demand, cached by the well project)."""
        if self._ws is None or self._ws.wcp is None or name not in self._ws.names:
            return {}
        try:
            w = self._ws.wcp.load_well(name)
        except Exception:
            return {}
        if w.depth is None or not len(w.depth):
            return {}
        return {c: (w.depth, v) for c, v in w.logs.items()}

    def log_names(self, max_wells=300):
        """Curve names over the project wells (skipped for very large projects - type them in a layout)."""
        if self._ws is None or len(self.trajs) > max_wells:
            return []
        if self._log_names is None:
            names = set()
            for t in self.trajs:
                names |= set(self.well_logs(t.name))
            self._log_names = sorted(names)
        return self._log_names

    def set_zone(self, top, base):
        self.zone = (top, base)

    def values_for(self, source):
        """{well name: values along its stations} for a colour source."""
        out = {}
        for t in self.trajs:
            if source == "TVDSS":
                v = t.z
            elif source.startswith("grid: "):
                v = L.sample_grid(t, self.grids[source[6:]])
            elif source.startswith("stage: "):
                col = source[7:]
                st = self.stages.get(t.api) or self.stages.get(t.name) or []
                v = L.stage_values(t, [(a, b, d.get(col, np.nan)) for a, b, d in st])
            elif source.startswith("seismic: "):
                try:
                    td = self._ws.td_for(t.name)
                    v = L.sample_volume(t, self.project.volume(source[9:]), td.twt)
                except Exception:  # no time-depth for this well
                    v = np.full(len(t.md), np.nan)
            elif source.startswith("log: "):
                lg = self.well_logs(t.name).get(source[5:])
                v = L.sample_log(t, *lg) if lg is not None else np.full(len(t.md), np.nan)
            elif source == "in zone":
                top, base = self.zone or list(self.grids)[:2]
                v = L.zone_along(t, self.grids[top], self.grids[base])
            else:
                raise ValueError(f"unknown colour source {source}")
            out[t.name] = np.asarray(v, float)
        return out

    def color_by(self, source):
        self.color_name = source
        self.color_values = self.values_for(source)
        allv = np.concatenate([v[np.isfinite(v)] for v in self.color_values.values()] or [np.zeros(0)])
        self.vrange = (float(np.percentile(allv, 2)), float(np.percentile(allv, 98))) if len(allv) else (0.0, 1.0)
        if self.vrange[1] <= self.vrange[0]:
            self.vrange = (self.vrange[0] - 0.5, self.vrange[0] + 0.5)
        self.draw_map()
        return self.color_values

    def add_lateral_stats(self):
        """Per-lateral mean/P10/P50/P90 of the current colour attribute (+ % in zone) as dashboard columns."""
        if not self.color_name:
            return None
        zones = None
        if len(self.grids) >= 2:
            top, base = self.zone or list(self.grids)[:2]
            zones = {t.name: L.zone_along(t, self.grids[top], self.grids[base]) for t in self.trajs}
        st = L.stats_table(self.trajs, {self.color_name: self.color_values}, zones)
        for c in st.names:
            if c.startswith(self.color_name) or c == "pct_in_zone":
                self.model.table.cols[c] = st.cols[c]
        self.model._emit("data")
        return st

    def send_to_analytics(self, window=None):
        """Add the dashboard table (lateral stats etc.) to the Analytics window as inputs / targets."""
        win = window or self.analytics_window
        if win is None or not hasattr(win, "add_table"):
            self.status.setText("Open Interpret > Analytics first.")
            return None
        return win.add_table(self.model.table)

    def strip_for(self, k, step=5.0):
        """laterals.strip_data for well row k: its logs, the zone grids (if set), the first 3D volume."""
        t = self.trajs[k]
        top = base = None
        if len(self.grids) >= 2:
            tn, bn = self.zone or list(self.grids)[:2]
            top, base = self.grids[tn], self.grids[bn]
        vol = twt = None
        if self.project is not None and getattr(self.project, "volumes", None) and self._ws is not None:
            try:
                vol = self.project.volume(self.project.volumes[0]["name"])
                twt = self._ws.td_for(t.name).twt
            except Exception:
                vol = twt = None
        return L.strip_data(t, self.well_logs(t.name), top, base, vol, twt, step)

    def open_strip(self, k=None):
        if k is None:
            k = min(self.model.selected) if self.model.selected else None
        if k is None:
            self.status.setText("Select a well first.")
            return None
        from .well_strip import WellStrip
        d = self.strip_for(k)
        self._strip = WellStrip()
        self._strip.setWindowTitle(f"Well strip - {self.trajs[k].name}")
        self._strip.show_data(self.trajs[k].name, d)
        self._strip.show()
        return d

    def map_batches(self, pixel=None):
        """{bin: (xs, ys)} NaN-separated polylines, one per colour bin (one draw call each)."""
        return L.batch_color_bins(self.trajs, self.color_values, self.vrange[0], self.vrange[1], N_BINS, pixel)

    # ------------------------------------------------------------ drawing
    def redraw(self):
        self.draw_map()
        self.draw_histogram()
        self.draw_crossplot()
        self.fill_table()

    def _pixel(self):
        try:
            (x0, x1), _ = self.map.getViewBox().viewRange()
            w = self.map.width() or 800
            return max(0.0, (x1 - x0) / w)
        except Exception:
            return None

    def draw_map(self):
        self.map.clear()
        self._draw_background()
        if not self.trajs:
            return
        vis = self.model.visible() if len(self.model) else np.ones(len(self.trajs), bool)
        sel = self.model.selected
        gx, gy = self.grey_paths(vis, self._pixel())  # all wellbores as one NaN-separated curve
        if len(gx):
            self.map.plot(gx, gy, pen=pg.mkPen((180, 180, 180), width=1), connect="finite")
        if self.color_values:
            show = {t.name: v for k, (t, v) in enumerate(zip(self.trajs, [self.color_values.get(t.name) for t in self.trajs]))
                    if vis[k] and v is not None}
            rgb = L.colormap_rgb(N_BINS)
            batches = L.batch_color_bins([t for t in self.trajs if t.name in show], show, self.vrange[0], self.vrange[1],
                                         N_BINS, self._pixel())
            for b, (xs, ys) in batches.items():
                self.map.plot(xs, ys, pen=pg.mkPen(tuple(int(c) for c in rgb[b]), width=3), connect="finite")
        xs = np.array([t.x[0] for t in self.trajs])
        ys = np.array([t.y[0] for t in self.trajs])
        brushes = [pg.mkBrush(255, 200, 0) if k in sel else pg.mkBrush(60, 60, 60) for k in range(len(self.trajs))]
        self.map.addItem(pg.ScatterPlotItem(xs[vis], ys[vis], size=7,
                                            brush=[b for b, v in zip(brushes, vis) if v]))
        self.status.setText(f"{int(vis.sum())} of {len(self.trajs)} wells, {len(sel)} selected"
                            + (f"; colour: {self.color_name} {self.vrange[0]:.4g} - {self.vrange[1]:.4g}"
                               if self.color_name else ""))

    def grey_paths(self, vis=None, pixel=None):
        """Every visible wellbore (decimated to the pixel size) in one NaN-separated polyline."""
        xs, ys = [], []
        for k, t in enumerate(self.trajs):
            if vis is not None and not vis[k]:
                continue
            idx = L.lod_indices(t.x, t.y, pixel) if pixel else np.arange(len(t.x))
            xs += [t.x[idx], [np.nan]]
            ys += [t.y[idx], [np.nan]]
        return (np.concatenate(xs), np.concatenate(ys)) if xs else (np.zeros(0), np.zeros(0))

    def draw_histogram(self):
        self.hist.clear()
        col = self.hist_box.currentText() if self.hist_box.currentText() in self.model.columns() else None
        if not col or col in self.model.text:
            return
        e, c, cs = self.model.histogram(col)
        self.hist.addItem(pg.BarGraphItem(x0=e[:-1], x1=e[1:], height=c, brush=(120, 150, 200)))
        self.hist.addItem(pg.BarGraphItem(x0=e[:-1], x1=e[1:], height=cs, brush=(255, 200, 0)))
        f = self.model.filters.get(col)
        lo, hi = (f[1], f[2]) if f and f[0] == "range" else (None, None)
        reg = pg.LinearRegionItem([e[0] if lo is None else lo, e[-1] if hi is None else hi], brush=(0, 0, 0, 20))
        reg.sigRegionChangeFinished.connect(lambda r, col=col, e=(e[0], e[-1]): self.filter_from_region(col, *r.getRegion(), e))
        self.hist.addItem(reg)

    def filter_from_region(self, col, lo, hi, full=None):
        """Histogram range handles -> numeric filter (an end dragged to the data limit stays open)."""
        if full is not None:
            lo = None if lo <= full[0] else lo
            hi = None if hi >= full[1] else hi
        self.model.set_filter(col, lo, hi)

    def draw_crossplot(self):
        self.xplot.clear()
        cols = self.model.columns()
        xc, yc = self.x_box.currentText(), self.y_box.currentText()
        if xc not in cols or yc not in cols or xc in self.model.text or yc in self.model.text:
            return
        d = self.model.crossplot(xc, yc)
        br = [pg.mkBrush(255, 200, 0) if s else pg.mkBrush(70, 110, 180) for s in d["selected"]]
        self.xplot.addItem(pg.ScatterPlotItem(d["x"], d["y"], size=8, brush=br))
        if d["fit"]:
            a, b, r = d["fit"]
            xx = np.array([d["x"].min(), d["x"].max()])
            self.xplot.plot(xx, a * xx + b, pen=pg.mkPen((200, 0, 0), style=QtCore.Qt.DashLine))
            self.xplot.setTitle(f"r = {r:.2f}")

    def fill_table(self):
        cols = self.model.columns()
        rows = self.model.table_rows(cols)
        if len(rows) > self.MAX_TABLE_ROWS:  # QTableWidget gets slow beyond a few thousand rows
            sel = [r for r in rows if r[0] in self.model.selected]
            rows = sel + [r for r in rows if r[0] not in self.model.selected][:self.MAX_TABLE_ROWS - len(sel)]
            self.status.setText(f"Table shows {len(rows)} wells - filter to see the others")
        self._table_rows = [r for r, _ in rows]
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        self.table.setColumnCount(len(cols) + 1)
        self.table.setHorizontalHeaderLabels(["well"] + cols)
        for i, (r, vals) in enumerate(rows):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(self.model.ids[r])))
            for j, v in enumerate(vals):
                txt = v if isinstance(v, str) else ("" if not np.isfinite(v) else f"{v:.4g}")
                self.table.setItem(i, j + 1, QtWidgets.QTableWidgetItem(txt))
        self.table.blockSignals(False)
        self._sync_table_selection()

    def set_background(self, name):
        self.background = name if name in self.grids else None
        self.draw_map()

    def text_filter(self, col, text):
        """'A, B' -> keep rows whose text column is A or B; empty text removes the filter."""
        vals = [v.strip() for v in str(text).split(",") if v.strip()]
        if vals:
            self.model.set_filter(col, values=vals)
        else:
            self.model.set_filter(col)

    def _text_filter_dialog(self):
        cols = list(self.model.text)
        if not cols:
            self.status.setText("No text columns (formation, operator...) in the dashboard table.")
            return
        col, ok = QtWidgets.QInputDialog.getItem(self, "Filter by text", "Column", cols, 0, False)
        if not ok:
            return
        present = sorted(set(self.model.text[col]))
        text, ok = QtWidgets.QInputDialog.getText(self, "Filter by text", f"Keep values (comma separated)\n"
                                                  f"present: {', '.join(present[:30])}")
        if ok:
            self.text_filter(col, text)

    def _draw_background(self):
        g = self.grids.get(self.background) if self.background else None
        if g is None:
            return
        from PySide6 import QtGui
        from .. import horizons as hz
        from . import colormaps
        rgba, lo, hi = hz.structure_rgba(g.z, colormaps.lut("Grey", reverse=g.unit in ("ms", "m")))
        it = pg.ImageItem(rgba)
        it.setTransform(QtGui.QTransform(*g.transform()))
        it.setOpacity(0.6)
        it.setZValue(-10)
        self.map.addItem(it)

    def _refresh_boxes(self):
        cols = self.model.columns()
        for box in (self.hist_box, self.x_box, self.y_box):
            cur = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems(cols)
            if cur in cols:
                box.setCurrentText(cur)
            box.blockSignals(False)
        self.bg_box.blockSignals(True)
        cur = self.bg_box.currentText()
        self.bg_box.clear()
        self.bg_box.addItems(["(none)"] + list(self.grids))
        if cur in self.grids:
            self.bg_box.setCurrentText(cur)
        self.bg_box.blockSignals(False)
        self.color_box.blockSignals(True)
        self.color_box.clear()
        srcs = self.color_sources()
        self.color_box.addItems(srcs)
        if self.color_name in srcs:  # keep showing what the map is coloured by
            self.color_box.setCurrentText(self.color_name)
        self.color_box.blockSignals(False)

    # ------------------------------------------------------------ linking
    def _on_model(self, kind):
        if kind == "data":
            self._refresh_boxes()
        if kind == "selection":  # keep the table (and the user's click) as it is, just mirror the selection
            self.draw_map()
            self.draw_histogram()
            self.draw_crossplot()
            self._sync_table_selection()
        else:
            self.redraw()

    def _sync_table_selection(self):
        sel = self.model.selected
        self.table.blockSignals(True)
        mode = self.table.selectionMode()
        self.table.clearSelection()
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for i, r in enumerate(self._table_rows):
            if r in sel:
                self.table.selectRow(i)
        self.table.setSelectionMode(mode)
        self.table.blockSignals(False)

    def _table_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        self.model.select([self._table_rows[i] for i in rows if i < len(self._table_rows)])

    # ------------------------------------------------------------ spacing / drainage / gun barrel (PLAN 29)
    def spacing(self, shmax_az=None, half_length=150.0, width=50.0, zones=None, first_prod=None, draw=True):
        """Inter-well distances (+ parent / child when first_prod {name: date} is given) and drainage ellipses along
        SHmax (default: project display 'shmax_az' or 90) drawn on the dashboard map; the distance table is joined to
        the well table (columns Nearest_m, Nearest_vert_m, Class)."""
        from .. import spacing as SP
        if shmax_az is None:
            shmax_az = float((self.project.display.get("shmax_az") if self.project is not None else None) or 90.0)
        rows = SP.well_distances(self.trajs, zones)
        ell = [dict(SP.drainage_ellipse(t, shmax_az, half_length, width), name=t.name) for t in self.trajs]
        cls = SP.classify_parent_child(first_prod, rows) if first_prod else {}
        per = {}
        for t in self.trajs:
            nb = SP.nearest_neighbours(rows, t.name)
            per[t.name] = {"Nearest_m": nb[0][1] if nb else float("nan"), "Nearest_vert_m": nb[0][2] if nb else float("nan"),
                           "Class": cls.get(t.name, "")}
        self.spacing_result = {"rows": rows, "ellipses": ell, "per_well": per, "shmax_az": shmax_az}
        if draw:
            for it in getattr(self, "_ellipse_items", []):
                try:
                    self.map.removeItem(it)
                except Exception:
                    pass
            self._ellipse_items = []
            for e in ell:
                xs, ys = SP.ellipse_polygon(e)
                it = pg.PlotCurveItem(xs, ys, pen=pg.mkPen((200, 120, 0), width=1))
                self.map.addItem(it)
                self._ellipse_items.append(it)
        self.status.setText(f"{len(rows)} well pairs; SHmax {shmax_az:.0f} deg; ellipses drawn")
        return self.spacing_result

    def _spacing_dialog(self):
        az, ok = QtWidgets.QInputDialog.getDouble(self, "Spacing", "SHmax azimuth (deg)", 90.0, 0.0, 180.0, 0)
        if ok:
            self.spacing(az)

    def gun_barrel_view(self, names=None):
        """Gun-barrel window of the (selected or all) laterals with their ellipse widths."""
        from .. import spacing as SP
        trajs = [t for t in self.trajs if names is None or t.name in names]
        ell = (getattr(self, "spacing_result", None) or {}).get("ellipses")
        gb = SP.gun_barrel(trajs, ellipses=[e for e in (ell or []) if e["name"] in {t.name for t in trajs}])
        win = QtWidgets.QDialog(self)
        win.setWindowTitle("Gun barrel")
        lay = QtWidgets.QVBoxLayout(win)
        pw = pg.PlotWidget()
        pw.setBackground("w")
        pw.invertY(True)
        pw.setLabel("bottom", "Offset across the laterals (m)")
        pw.setLabel("left", "TVD (m)")
        pw.addItem(pg.ScatterPlotItem([w["offset"] for w in gb["wells"]], [w["tvd"] for w in gb["wells"]], size=10,
                                      brush=pg.mkBrush(30, 30, 30)))
        for e in gb["ellipses"]:
            pw.addItem(pg.PlotCurveItem([e["offset"] - e["half_width"], e["offset"] + e["half_width"]],
                                        [e["tvd"], e["tvd"]], pen=pg.mkPen((200, 120, 0), width=3)))
        for w in gb["wells"]:
            t = pg.TextItem(w["name"], color=(0, 0, 0), anchor=(0.5, 1.2))
            t.setPos(w["offset"], w["tvd"])
            pw.addItem(t)
        lay.addWidget(pw)
        win.resize(800, 450)
        win.show()
        self._gun_barrel = (win, gb)
        if self.basemap is not None and hasattr(self.basemap, "show_current"):
            self.basemap.show_current(gb["line"])  # the section line on the main map
        return gb

    def show_cursor(self, msg):
        """Linked cursor (23c): red cross on the dashboard map + nearest wellbore in the status bar."""
        old = getattr(self, "_cursor_item", None)
        if old is not None:
            try:
                self.map.removeItem(old)
            except Exception:
                pass
            self._cursor_item = None
        if msg is None or msg.get("clear") or msg.get("x") is None:
            return None
        it = pg.ScatterPlotItem([float(msg["x"])], [float(msg["y"])], symbol="+", size=18, pen=pg.mkPen((220, 0, 0), width=2))
        it.setZValue(1000)
        self.map.addItem(it)
        self._cursor_item = it
        k = self.nearest_well(float(msg["x"]), float(msg["y"]))
        name = getattr(self.trajs[k], "name", None) if k is not None else None
        if name:
            self.statusBar().showMessage(f"cursor near {name}")
        return name

    def nearest_well(self, x, y):
        if not self.trajs:
            return None
        key = len(self.trajs)
        if getattr(self, "_pick_key", None) != key:  # all stations of all wellbores in one array, built once
            self._pick_xy = np.concatenate([np.column_stack([t.x, t.y]) for t in self.trajs])
            self._pick_owner = np.concatenate([np.full(len(t.x), k) for k, t in enumerate(self.trajs)])
            self._pick_key = key
        d = (self._pick_xy[:, 0] - x) ** 2 + (self._pick_xy[:, 1] - y) ** 2
        return int(self._pick_owner[int(np.argmin(d))])

    def _map_clicked(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        vb = self.map.getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = vb.mapSceneToView(ev.scenePos())
        k = self.nearest_well(p.x(), p.y())
        mods = ev.modifiers()
        tol = 12 * abs(vb.viewPixelSize()[0])  # a click far from every wellbore clears the selection
        if k is None or self.distance_to_well(k, p.x(), p.y()) > tol:
            if not mods & QtCore.Qt.ControlModifier:
                self.model.select([], "replace")
            return
        self.model.select([k], "toggle" if mods & QtCore.Qt.ControlModifier else "replace")

    def distance_to_well(self, k, x, y):
        t = self.trajs[k]
        return float(np.min(np.hypot(t.x - x, t.y - y)))

    # ------------------------------------------------------------ layouts + export
    def _save_layout(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save dashboard layout", "", "JSON (*.json)")
        if path:
            self.model.layout.update(hist=self.hist_box.currentText(), x=self.x_box.currentText(),
                                     y=self.y_box.currentText(), color=self.color_name)
            self.model.save_layout(path)

    def _load_layout(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load dashboard layout", "", "JSON (*.json)")
        if path:
            self.model.load_layout(path)
            lay = self.model.layout
            for box, k in ((self.hist_box, "hist"), (self.x_box, "x"), (self.y_box, "y")):
                if lay.get(k):
                    box.setCurrentText(lay[k])
            if lay.get("color") in self.color_sources():
                self.color_by(lay["color"])

    def export_html(self, path):
        """Standalone HTML with a matplotlib map of the coloured laterals, the histogram and the well table."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        figs = []
        fig, ax = plt.subplots(figsize=(7, 6))
        vis = self.model.visible() if len(self.model) else np.ones(len(self.trajs), bool)
        for k, t in enumerate(self.trajs):
            if not vis[k]:
                continue
            v = self.color_values.get(t.name)
            if v is None:
                ax.plot(t.x, t.y, color="0.6", lw=1)
            else:
                seg = np.stack([np.column_stack([t.x[:-1], t.y[:-1]]), np.column_stack([t.x[1:], t.y[1:]])], 1)
                lc = LineCollection(seg, array=0.5 * (v[:-1] + v[1:]), cmap="viridis", lw=2.5)
                lc.set_clim(*self.vrange)
                ax.add_collection(lc)
            ax.plot(t.x[0], t.y[0], "o", color="gold" if k in self.model.selected else "k", ms=4)
        ax.set_aspect("equal")
        ax.autoscale()
        if self.color_name:
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(*self.vrange))
            fig.colorbar(sm, ax=ax, label=self.color_name)
        figs.append(("Map", fig))
        col = self.hist_box.currentText()
        if col in self.model.columns() and col not in self.model.text:
            f2, a2 = plt.subplots(figsize=(5, 3))
            e, c, cs = self.model.histogram(col)
            a2.bar(e[:-1], c, np.diff(e), align="edge", color="#7896c8")
            a2.bar(e[:-1], cs, np.diff(e), align="edge", color="gold")
            a2.set_xlabel(col)
            figs.append((f"Histogram: {col}", f2))
        out = self.model.export_html(path, "Dashboard", figs)
        for _, f in figs:
            plt.close(f)
        return out

    def _export_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export dashboard", "", "HTML (*.html)")
        if path:
            self.export_html(path)
