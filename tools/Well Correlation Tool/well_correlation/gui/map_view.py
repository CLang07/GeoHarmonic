"""Base map: well spots, labels, section line; build sections by clicking wells or drawing a line."""
import math

import numpy as np
import os
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

GREY = (150, 150, 150)
BLUE = (31, 79, 191)
RED = (200, 22, 27)


class MapView(QtWidgets.QWidget):
    wellClicked = QtCore.Signal(str)  # toggle a well in/out of the section
    wellSelected = QtCore.Signal(str)  # show info
    lineSelected = QtCore.Signal(list)  # ordered wells along a drawn line
    mapMessage = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        self.btn_line = QtWidgets.QPushButton("Draw section line")
        self.btn_line.setCheckable(True)
        self.btn_line.setToolTip("Click points on the map, double-click to finish.\n"
                                 "Wells within the buffer are added in order along the line.")
        self.buffer = QtWidgets.QDoubleSpinBox()
        self.buffer.setRange(0.1, 100.0)
        self.buffer.setValue(2.0)
        self.buffer.setSuffix(" km")
        self.buffer.setToolTip("Distance either side of the drawn line")
        self.chk_labels = QtWidgets.QCheckBox("Labels")
        self.chk_labels.setChecked(True)
        btn_fit = QtWidgets.QPushButton("Fit")
        bar.addWidget(self.btn_line)
        bar.addWidget(QtWidgets.QLabel("Buffer"))
        bar.addWidget(self.buffer)
        bar.addWidget(self.chk_labels)
        bar.addStretch(1)
        bar.addWidget(btn_fit)
        lay.addLayout(bar)
        # ---- map from tops (step 4)
        mbar = QtWidgets.QHBoxLayout()
        self.map_kind = QtWidgets.QComboBox()
        self.map_kind.addItem("No map", "none")
        self.map_kind.addItem("Structure (TVDSS)", "structure")
        self.map_kind.addItem("Isopach (TVD)", "isopach")
        self.map_top = QtWidgets.QComboBox()
        self.map_top.setToolTip("Top (structure) or upper surface (isopach)")
        self.map_base = QtWidgets.QComboBox()
        self.map_base.setToolTip("Lower surface for the isopach")
        self.map_method = QtWidgets.QComboBox()
        self.map_method.addItem("Spline", "tps")
        self.map_method.addItem("IDW", "idw")
        self.map_method.setToolTip("Spline: smooth, honours wells (few wells). IDW: inverse distance, never overshoots.")
        self.map_ci = QtWidgets.QDoubleSpinBox()
        self.map_ci.setRange(0, 1000)
        self.map_ci.setDecimals(1)
        self.map_ci.setSpecialValueText("CI auto")
        self.map_ci.setSuffix(" m")
        self.map_ci.setToolTip("Contour interval (0 = automatic)")
        self.btn_map_png = QtWidgets.QPushButton("Export map...")
        for wdg in (self.map_kind, self.map_top, self.map_base, self.map_method, self.map_ci):
            mbar.addWidget(wdg)
            if isinstance(wdg, QtWidgets.QComboBox):
                wdg.currentIndexChanged.connect(self._map_settings_changed)
        self.map_ci.valueChanged.connect(self._map_settings_changed)
        mbar.addWidget(self.btn_map_png)
        self.btn_map_png.clicked.connect(self.export_map)
        lay.addLayout(mbar)
        self.grid = None
        self._map_items = []

        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "Easting (NZTM, m)")
        self.plot.setLabel("left", "Northing (NZTM, m)")
        lay.addWidget(self.plot, 1)
        self.hint = QtWidgets.QLabel("Click a well to add it to / remove it from the section.")
        self.hint.setStyleSheet("color:#555")
        lay.addWidget(self.hint)

        self.scatter = pg.ScatterPlotItem(size=11, pxMode=True)
        self.scatter.sigClicked.connect(self._spot_clicked)
        self.plot.addItem(self.scatter)
        self.section_curve = pg.PlotCurveItem(pen=pg.mkPen(BLUE, width=2))
        self.plot.addItem(self.section_curve)
        self.draw_curve = pg.PlotCurveItem(pen=pg.mkPen(RED, width=2, style=QtCore.Qt.DashLine))
        self.plot.addItem(self.draw_curve)
        self.labels = []
        self.project = None
        self.section = []
        self.selected = None
        self._draw_pts = []

        self.btn_line.toggled.connect(self._line_mode)
        self.chk_labels.toggled.connect(lambda on: [t.setVisible(on) for _, t in self.labels])
        btn_fit.clicked.connect(self.fit)
        self.plot.scene().sigMouseClicked.connect(self._scene_clicked)

    # ---------------------------------------------------------------- data
    def set_project(self, project):
        self.project = project
        for _, t in self.labels:
            self.plot.removeItem(t)
        self.labels = []
        for w in project.wells.values():
            if w.x is None:
                continue
            t = pg.TextItem(w.name, color=(30, 30, 30), anchor=(-0.1, 1.1))
            t.setPos(w.x, w.y)
            t.setVisible(self.chk_labels.isChecked())
            self.plot.addItem(t, ignoreBounds=True)
            self.labels.append((w.name, t))
        self.refresh()
        self.fit()
        self.set_formations()

    def set_formations(self):
        """Fill the map formation lists from the project (keeps the current choice)."""
        if not self.project:
            return
        d = self.project.display.setdefault("map", {})
        names = [f["name"] for f in self.project.formations]
        for combo, key, default in ((self.map_top, "top", 0), (self.map_base, "base", 1)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            cur = d.get(key)
            combo.setCurrentIndex(names.index(cur) if cur in names else min(default, len(names) - 1))
            combo.blockSignals(False)
        for combo, key in ((self.map_kind, "kind"), (self.map_method, "method")):
            combo.blockSignals(True)
            i = combo.findData(d.get(key))
            combo.setCurrentIndex(i if i >= 0 else 0)
            combo.blockSignals(False)
        self.map_ci.blockSignals(True)
        self.map_ci.setValue(float(d.get("ci", 0) or 0))
        self.map_ci.blockSignals(False)
        self.map_base.setEnabled(self.map_kind.currentData() == "isopach")
        self.refresh_map()

    def _map_settings_changed(self, *_):
        if not self.project:
            return
        d = self.project.display.setdefault("map", {})
        d.update(kind=self.map_kind.currentData(), top=self.map_top.currentText(), base=self.map_base.currentText(),
                 method=self.map_method.currentData(), ci=self.map_ci.value())
        self.map_base.setEnabled(d["kind"] == "isopach")
        self.refresh_map()

    def refresh_map(self):
        """(Re)grid and draw the map from the current tops. Cheap enough to call after every edit."""
        from .. import mapping as M
        for it in self._map_items:
            self.plot.removeItem(it)
        self._map_items = []
        self.grid = None
        d = self.project.display.get("map", {}) if self.project else {}
        kind = d.get("kind", "none")
        if kind == "none" or not self.project or not self.project.formations:
            self._post_values({})
            return
        top, base = d.get("top"), d.get("base")
        try:
            pts = M.map_points(self.project, kind, top, base)
            g = M.grid_points(pts, d.get("method", "tps"))
        except Exception as e:
            self._post_values({})
            self.mapMessage.emit(f"Map: {e}")
            return
        self.grid = g
        self.grid["title"] = (f"Structure: top {top} (TVDSS m)" if kind == "structure"
                              else f"Isopach: {top} to {base} (TVD m)")
        z = g["z"]
        lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
        ci = d.get("ci") or M.nice_interval(lo, hi)
        levels = np.arange(np.ceil(lo / ci) * ci, hi, ci)
        styled = None
        if d.get("carto", True):  # PLAN 21f: hillshaded perceptual fill + labels along the contours
            from ..carto import styled_map
            styled = styled_map(g, depth_like=(kind == "structure"), levels=list(levels), alpha=190)
        img = pg.ImageItem(styled["rgba"] if styled else M.colorize(g, reverse=(kind == "isopach")), levels=(0, 255),
                           autoLevels=False)
        img.setRect(QtCore.QRectF(g["x0"] - g["dx"] / 2, g["y0"] - g["dy"] / 2, g["nx"] * g["dx"], g["ny"] * g["dy"]))
        img.setZValue(-10)
        self.plot.addItem(img, ignoreBounds=True)
        self._map_items.append(img)
        segs = styled["segments"] if styled else M.contour_segments(g, levels)
        xs = np.concatenate([v[0] for v in segs.values()]) if segs else np.array([])
        ys = np.concatenate([v[1] for v in segs.values()]) if segs else np.array([])
        if xs.size:
            c = pg.PlotCurveItem(xs, ys, pen=pg.mkPen((40, 40, 40), width=1), connect="pairs")
            c.setZValue(-5)
            self.plot.addItem(c, ignoreBounds=True)
            self._map_items.append(c)
        if styled:
            for x, y, ang, text in styled["labels"]:
                t = pg.TextItem(text, color=(30, 30, 30), anchor=(0.5, 0.5), angle=ang, fill=pg.mkBrush(255, 255, 255, 190))
                t.setPos(x, y)
                self.plot.addItem(t, ignoreBounds=True)
                self._map_items.append(t)
        else:
            for lev, (lx, ly) in segs.items():
                k = (len(lx) // 2) & ~1
                t = pg.TextItem(f"{lev:g}", color=(30, 30, 30), anchor=(0.5, 0.5))
                t.setPos(lx[k], ly[k])
                self.plot.addItem(t, ignoreBounds=True)
                self._map_items.append(t)
        self._post_values({p[0]: p[3] for p in pts})
        self.mapMessage.emit(f"{self.grid['title']}: {len(pts)} wells, contour interval {ci:g} m")

    def _post_values(self, values):
        for name, t in self.labels:
            t.setText(name + (f"\n{values[name]:.0f}" if name in values else ""))

    def export_map(self):
        from .. import mapping as M
        if self.grid is None:
            self.mapMessage.emit("Choose a structure or isopach map first")
            return
        base = os.path.dirname(self.project.path or "")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export map", os.path.join(base, "map.png"),
                                                        "PNG image (*.png);;PDF (*.pdf)")
        if path:
            unit = "TVDSS (m)" if self.grid["title"].startswith("Structure") else "Thickness (m)"
            M.export_map_png(self.project, self.grid, self.grid["title"], path, unit)
            self.mapMessage.emit(f"Exported {path}")

    def set_section(self, names):
        self.section = list(names)
        self.refresh()

    def set_selected(self, name):
        self.selected = name
        self.refresh()

    def refresh(self):
        if not self.project:
            return
        spots = []
        for w in self.project.wells.values():
            if w.x is None:
                continue
            has_logs = bool(w.las_paths)
            if w.name == self.selected:
                brush, size = RED, 15
            elif w.name in self.section:
                brush, size = BLUE, 13
            else:
                brush, size = GREY, 10
            spots.append({"pos": (w.x, w.y), "data": w.name, "brush": pg.mkBrush(*brush), "size": size,
                          "symbol": "o" if has_logs else "t", "pen": pg.mkPen("k", width=0.8)})
        self.scatter.setData(spots)
        xs = [self.project.wells[n].x for n in self.section if self.project.wells[n].x is not None]
        ys = [self.project.wells[n].y for n in self.section if self.project.wells[n].y is not None]
        self.section_curve.setData(xs, ys)

    def fit(self):
        self.plot.getPlotItem().getViewBox().autoRange(padding=0.08)

    # --------------------------------------------------------------- mouse
    def _spot_clicked(self, item, points, ev=None):
        if self.btn_line.isChecked() or not len(points):
            return
        name = points[0].data()
        self.wellSelected.emit(name)
        self.wellClicked.emit(name)
        if ev is not None:
            ev.accept()

    def _line_mode(self, on):
        self._draw_pts = []
        self.draw_curve.setData([], [])
        self.hint.setText("Click points along the section line; double-click to finish." if on else
                          "Click a well to add it to / remove it from the section.")

    def _scene_clicked(self, ev):
        if not self.btn_line.isChecked():
            return
        vb = self.plot.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = vb.mapSceneToView(ev.scenePos())
        self._draw_pts.append((p.x(), p.y()))
        a = np.array(self._draw_pts)
        self.draw_curve.setData(a[:, 0], a[:, 1])
        if ev.double() and len(self._draw_pts) >= 2:
            names = self.wells_along(self._draw_pts, self.buffer.value() * 1000.0)
            self.btn_line.setChecked(False)
            self.lineSelected.emit(names)

    def wells_along(self, pts, buffer_m):
        """Wells within buffer of a polyline, ordered by distance along it."""
        hits = []
        seglen = [math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:])]
        for w in self.project.wells.values():
            if w.x is None:
                continue
            best = None
            along0 = 0.0
            for (x0, y0), (x1, y1), L in zip(pts[:-1], pts[1:], seglen):
                if L == 0:
                    continue
                t = ((w.x - x0) * (x1 - x0) + (w.y - y0) * (y1 - y0)) / (L * L)
                t = min(1.0, max(0.0, t))
                px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
                d = math.hypot(w.x - px, w.y - py)
                if best is None or d < best[0]:
                    best = (d, along0 + t * L)
                along0 += L
            if best and best[0] <= buffer_m:
                hits.append((best[1], w.name))
        return [n for _, n in sorted(hits)]
