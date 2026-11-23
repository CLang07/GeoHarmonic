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
        for lab, w in (("Colour laterals by", self.color_box), ("Histogram", self.hist_box), ("X", self.x_box),
                       ("Y", self.y_box)):
            tb.addWidget(QtWidgets.QLabel(f" {lab} "))
            tb.addWidget(w)
        for text, fn in (("Wells from project", self.load_project_wells), ("Add basemap grid", self._add_basemap_grid),
                         ("Stage CSV...", self._stages_dialog), ("Lateral stats", self.add_lateral_stats),
                         ("Send to Analytics", self.send_to_analytics), ("Clear filters", lambda: self.model.clear_filters()),
                         ("Save layout...", self._save_layout), ("Load layout...", self._load_layout),
                         ("Export HTML...", self._export_dialog)):
            a = tb.addAction(text)
            a.triggered.connect(fn)
        self.color_box.currentTextChanged.connect(lambda s: s and self.color_by(s))
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
        if len(self.grids) >= 2:
            out.append("in zone")
        return out

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
        if not self.trajs:
            return
        vis = self.model.visible() if len(self.model) else np.ones(len(self.trajs), bool)
        sel = self.model.selected
        for k, t in enumerate(self.trajs):  # thin grey paths (also wells without colour values)
            if vis[k]:
                self.map.plot(t.x, t.y, pen=pg.mkPen((180, 180, 180), width=1))
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
        self.color_box.blockSignals(True)
        self.color_box.clear()
        self.color_box.addItems(self.color_sources())
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

    def nearest_well(self, x, y):
        if not self.trajs:
            return None
        d = [np.min(np.hypot(t.x - x, t.y - y)) for t in self.trajs]
        return int(np.argmin(d))

    def _map_clicked(self, ev):
        vb = self.map.getViewBox()
        p = vb.mapSceneToView(ev.scenePos())
        k = self.nearest_well(p.x(), p.y())
        if k is not None:
            mods = ev.modifiers()
            self.model.select([k], "toggle" if mods & QtCore.Qt.ControlModifier else "replace")

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
