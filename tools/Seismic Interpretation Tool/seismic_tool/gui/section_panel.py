"""Section / time-slice viewer for 3D volumes and 2D lines."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtWidgets  # noqa: E402
import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from . import colormaps  # noqa: E402
from .. import horizons as hz  # noqa: E402

MODES = ["Inline", "Crossline", "Time slice"]


class PickViewBox(pg.ViewBox):
    """ViewBox that hands left-button clicks/drags to the panel while horizon pick mode is on.
    Click = seed pick (snapped), drag = manual line, Shift+click/drag = erase. Right button keeps the menu."""

    def __init__(self, panel, *a, **k):
        super().__init__(*a, **k)
        self.panel = panel
        self._drag = []

    def _picking(self, ev):
        return self.panel.pick_mode and ev.button() == QtCore.Qt.LeftButton

    def mouseClickEvent(self, ev):
        if self._picking(ev):
            pt = self.mapToView(ev.pos())
            erase = bool(ev.modifiers() & QtCore.Qt.ShiftModifier)
            self.panel.pick_at(pt.x(), pt.y(), erase=erase)
            ev.accept()
            return
        super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if self._picking(ev):
            ev.accept()
            pt = self.mapToView(ev.pos())
            if ev.isStart():
                p0 = self.mapToView(ev.buttonDownPos())
                self._drag = [(p0.x(), p0.y())]
            self._drag.append((pt.x(), pt.y()))
            if ev.isFinish():
                erase = bool(ev.modifiers() & QtCore.Qt.ShiftModifier)
                pts, self._drag = self._drag, []
                self.panel.pick_drag(pts, erase=erase)
            return
        super().mouseDragEvent(ev, axis)


class SectionPanel(QtWidgets.QWidget):
    positionChanged = QtCore.Signal()  # current line / slice changed (basemap follows)
    cursorInfo = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.source = None  # ("3D", name) or ("2D", name)
        self.mode = "Inline"
        self.number = None
        self._img = None
        self._data = None  # current 2D array as displayed (rows = y)
        self._xaxis = None
        self._yaxis = None
        # horizon interpretation state (set by the Horizons dock)
        self.pick_mode = False
        self.active_horizon = None  # name
        self.snap_mode = "Peak"
        self.snap_ms = 12.0
        self.track_params = hz.TrackParams()
        self.track_scope = "Whole survey"  # or "This line only"
        self.mix_2d, self.mix_3d = 5, 1  # lateral mix used by the tracker (2D traces, 3D n x n block)
        self.seeds = []  # [(key, i, j)]
        self._hz_items = []
        self.horizonsEdited = None  # optional callback after any edit (dock/basemap refresh)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        self.src_combo = QtWidgets.QComboBox()
        self.src_combo.setMinimumWidth(180)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(MODES)
        self.num_spin = QtWidgets.QSpinBox()
        self.num_spin.setKeyboardTracking(False)
        self.num_spin.setMinimumWidth(90)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimumWidth(160)
        self.step_spin = QtWidgets.QSpinBox()
        self.step_spin.setRange(1, 1000)
        self.step_spin.setValue(1)
        self.step_spin.setPrefix("step ")
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(colormaps.NAMES)
        self.clip_spin = QtWidgets.QDoubleSpinBox()
        self.clip_spin.setRange(80.0, 100.0)
        self.clip_spin.setDecimals(1)
        self.clip_spin.setSingleStep(0.5)
        self.clip_spin.setValue(99.0)
        self.clip_spin.setPrefix("clip ")
        self.clip_spin.setSuffix(" %")
        self.clip_spin.setToolTip("Amplitude percentile mapped to the ends of the colour bar")
        self.rev_chk = QtWidgets.QCheckBox("Reverse polarity")
        for w in (self.src_combo, self.mode_combo, self.num_spin, self.slider, self.step_spin, self.cmap_combo,
                  self.clip_spin, self.rev_chk):
            bar.addWidget(w)
        bar.addStretch(1)
        lay.addLayout(bar)

        self.vb = PickViewBox(self)
        self.plot = pg.PlotWidget(viewBox=self.vb)
        self.plot.setBackground("w")
        self.plot.showGrid(x=False, y=True, alpha=0.2)
        lay.addWidget(self.plot, 1)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)

        self.src_combo.currentIndexChanged.connect(self._source_changed)
        self.mode_combo.currentTextChanged.connect(self._mode_changed)
        self.num_spin.valueChanged.connect(self._num_changed)
        self.slider.valueChanged.connect(self._slider_changed)
        self.cmap_combo.currentTextChanged.connect(lambda *_: self._apply_colors())
        self.clip_spin.valueChanged.connect(lambda *_: self._apply_colors())
        self.rev_chk.toggled.connect(lambda *_: self._apply_colors())
        from PySide6 import QtGui
        for key, d in (("PgUp", -1), ("PgDown", 1), ("Shift+PgUp", -10), ("Shift+PgDown", 10)):
            s = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            s.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            s.activated.connect(lambda d=d: self.step(d))

    # ------------------------------------------------------------ setup
    def set_project(self, project):
        self.project = project
        self.refresh_sources()

    def refresh_sources(self, select=None):
        p = self.project
        self.src_combo.blockSignals(True)
        self.src_combo.clear()
        for v in p.volumes if p else []:
            self.src_combo.addItem(f"3D: {v['name']}", ("3D", v["name"]))
        for l in p.lines if p else []:
            self.src_combo.addItem(f"2D: {l['name']}", ("2D", l["name"]))
        self.src_combo.blockSignals(False)
        d = p.display if p else {}
        self.cmap_combo.setCurrentText(d.get("cmap_name", colormaps.NAMES[0]))
        self.clip_spin.setValue(float(d.get("clip", 99.0)))
        self.rev_chk.setChecked(bool(d.get("reverse", False)))
        want = select or (tuple(d["source"]) if d.get("source") else None)
        idx = next((i for i in range(self.src_combo.count()) if self.src_combo.itemData(i) == want), 0)
        if self.src_combo.count():
            self.src_combo.setCurrentIndex(idx)
            self._source_changed(idx, number=d.get("number"), mode=d.get("mode"))
        else:
            self.plot.clear()

    def show(self, kind, name, mode=None, number=None):
        """Open a 3D volume (with mode/number) or a 2D line."""
        for i in range(self.src_combo.count()):
            if self.src_combo.itemData(i) == (kind, name):
                self.src_combo.blockSignals(True)
                self.src_combo.setCurrentIndex(i)
                self.src_combo.blockSignals(False)
                self._source_changed(i, number=number, mode=mode)
                return

    # ------------------------------------------------------------ navigation
    def _source_changed(self, idx, number=None, mode=None):
        data = self.src_combo.itemData(idx)
        if not data:
            return
        self.source = tuple(data)
        is3d = self.source[0] == "3D"
        for w in (self.mode_combo, self.num_spin, self.slider, self.step_spin):
            w.setEnabled(is3d)
        if is3d:
            if mode in MODES:
                self.mode_combo.blockSignals(True)
                self.mode_combo.setCurrentText(mode)
                self.mode_combo.blockSignals(False)
            self._mode_changed(self.mode_combo.currentText(), number=number)
        else:
            self.mode = "2D"
            self.number = None
            self.redraw(reset_view=True)

    def _mode_changed(self, mode, number=None):
        self.mode = mode
        v = self.project.volume(self.source[1])
        if mode == "Inline":
            lo, hi, st = int(v.inlines[0]), int(v.inlines[-1]), v.il_step
        elif mode == "Crossline":
            lo, hi, st = int(v.xlines[0]), int(v.xlines[-1]), v.xl_step
        else:
            lo, hi, st = int(v.times[0]), int(v.times[-1]), max(1, int(round(v.dt * 1000)))
        num = number if number is not None and lo <= number <= hi else (lo + hi) // 2 // st * st + (lo % st)
        for w in (self.num_spin, self.slider):
            w.blockSignals(True)
        self.num_spin.setRange(lo, hi)
        self.num_spin.setSingleStep(st)
        self.slider.setRange(lo, hi)
        self.slider.setSingleStep(st)
        self.slider.setPageStep(st * 10)
        self.step_spin.setValue(st)
        self.num_spin.setValue(num)
        self.slider.setValue(num)
        for w in (self.num_spin, self.slider):
            w.blockSignals(False)
        self.number = num
        self.redraw(reset_view=True)

    def _num_changed(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(v)
        self.slider.blockSignals(False)
        self.number = v
        self.redraw()

    def _slider_changed(self, v):
        st = self.step_spin.value()
        v = self.num_spin.minimum() + round((v - self.num_spin.minimum()) / st) * st
        self.num_spin.setValue(v)

    def step(self, n):
        if self.source and self.source[0] == "3D":
            self.num_spin.setValue(self.num_spin.value() + n * self.step_spin.value())

    # ------------------------------------------------------------ drawing
    def redraw(self, reset_view=False):
        if not self.project or not self.source:
            return
        kind, name = self.source
        try:
            if kind == "3D":
                v = self.project.volume(name)
                if self.mode == "Inline":
                    arr = v.inline(self.number).T  # (ns, nxl)
                    x, y, xl, yl = v.xlines, v.times, "Crossline", "TWT (ms)"
                elif self.mode == "Crossline":
                    arr = v.xline(self.number).T
                    x, y, xl, yl = v.inlines, v.times, "Inline", "TWT (ms)"
                else:
                    arr = v.time_slice(self.number)  # (nil, nxl)
                    x, y, xl, yl = v.xlines, v.inlines, "Crossline", "Inline"
            else:
                l = self.project.line(name)
                arr = l.data().T  # (ns, ntr)
                x, y, xl, yl = l.cdp, l.times, "CDP", "TWT (ms)"
        except Exception as e:
            self.cursorInfo.emit(f"Could not read data: {e}")
            return
        self._data, self._xaxis, self._yaxis = arr, x, y
        view = None if reset_view else self.vb.viewRange()
        self.plot.clear()
        self._img = pg.ImageItem(arr)
        dx = (x[-1] - x[0]) / max(1, len(x) - 1) if len(x) > 1 else 1
        dy = (y[-1] - y[0]) / max(1, len(y) - 1) if len(y) > 1 else 1
        self._img.setRect(QtCore.QRectF(x[0] - dx / 2, y[0] - dy / 2, dx * len(x), dy * len(y)))
        self.plot.addItem(self._img)
        self.plot.setLabel("bottom", xl)
        self.plot.setLabel("left", yl)
        self.plot.getPlotItem().invertY(self.mode != "Time slice" or kind == "2D")
        self._apply_colors()
        self._hz_items = []
        self.draw_horizons()
        if view and not reset_view:
            self.vb.setRange(xRange=view[0], yRange=view[1], padding=0)
        else:
            self.vb.autoRange(padding=0)
        self._title()
        if self.project:
            self.project.display.update(source=list(self.source), mode=self.mode,
                                        number=None if self.number is None else int(self.number))
        self.positionChanged.emit()

    def _title(self):
        kind, name = self.source
        t = f"{name} - {self.mode} {self.number}" if kind == "3D" else f"2D line {name}"
        if self.mode == "Time slice" and kind == "3D":
            t += " ms"
        self.plot.setTitle(t)

    def _apply_colors(self):
        if self._img is None or self._data is None:
            return
        a = np.abs(self._data[np.isfinite(self._data)])
        clip = float(np.percentile(a, self.clip_spin.value())) if a.size else 1.0
        clip = clip or 1.0
        self._img.setLookupTable(colormaps.lut(self.cmap_combo.currentText(), self.rev_chk.isChecked()))
        self._img.setLevels((-clip, clip))
        if self.project:
            self.project.display.update(cmap_name=str(self.cmap_combo.currentText()), clip=float(self.clip_spin.value()),
                                        reverse=bool(self.rev_chk.isChecked()))

    def _mouse_moved(self, pos):
        if self._data is None or not self.vb.sceneBoundingRect().contains(pos):
            return
        pt = self.vb.mapSceneToView(pos)
        x, y = self._xaxis, self._yaxis
        j = int(np.clip(np.searchsorted(x, pt.x()), 0, len(x) - 1))
        i = int(np.clip(np.searchsorted(y, pt.y()), 0, len(y) - 1))
        amp = self._data[i, j]
        kind, name = self.source
        if kind == "3D":
            v = self.project.volume(name)
            if self.mode == "Inline":
                il, xl, t = self.number, x[j], y[i]
            elif self.mode == "Crossline":
                il, xl, t = x[j], self.number, y[i]
            else:
                il, xl, t = y[i], x[j], self.number
            X, Y = v.xy(il, xl)
            self.cursorInfo.emit(f"IL {il:.0f}  XL {xl:.0f}  t {t:.0f} ms  amp {amp:.4g}   X {X:.1f}  Y {Y:.1f}")
        else:
            l = self.project.line(name)
            self.cursorInfo.emit(f"{name}  CDP {x[j]:.0f}  t {y[i]:.0f} ms  amp {amp:.4g}   X {l.x[j]:.1f}  Y {l.y[j]:.1f}")

    # ------------------------------------------------------------ horizons
    def hset(self):
        return getattr(self.project, "horizons", None) if self.project else None

    def picker(self):
        return hz.Picker(self.project, self.hset()) if self.hset() is not None else None

    def section_key(self):
        if not self.source:
            return None
        return hz.survey_key(*self.source)

    def trace_at(self, xval):
        """(i, j) grid indices of the trace nearest to section x value, or None (time slice)."""
        kind, name = self.source
        if kind == "2D":
            cdp = self.project.line(name).cdp
            return 0, int(np.argmin(np.abs(cdp - xval)))
        v = self.project.volume(name)
        if self.mode == "Inline":
            return v.il_index(self.number), v.xl_index(xval)
        if self.mode == "Crossline":
            return v.il_index(xval), v.xl_index(self.number)
        return None

    def _active(self):
        hs = self.hset()
        h = hs.get(self.active_horizon) if hs is not None and self.active_horizon else None
        if h is None:
            self.cursorInfo.emit("Create or select a horizon in the Horizons panel first.")
        return h

    def _edited(self):
        self.draw_horizons()
        if self.horizonsEdited:
            self.horizonsEdited()

    def pick_at(self, x, t_ms, erase=False):
        """Click in pick mode: snapped seed pick (or erase the pick on that trace)."""
        h = self._active()
        if h is None or not self.source:
            return None
        ij = self.trace_at(x)
        if ij is None:
            self.cursorInfo.emit("Pick on an inline, crossline or 2D line (not on a time slice).")
            return None
        key, pk = self.section_key(), self.picker()
        if erase:
            pk.erase(h, key, [ij[0]], [ij[1]])
            self.seeds = [s for s in self.seeds if s != (key, ij[0], ij[1])]
            self._edited()
            return None
        t = pk.pick_points(h, key, ij[0], ij[1], t_ms, self.snap_mode, self.snap_ms)
        self.seeds.append((key, ij[0], ij[1]))
        self.cursorInfo.emit(f"Seed {len(self.seeds)} on {h.name} at {float(t[0]):.1f} ms - press T to track")
        self._edited()
        return float(t[0])

    def pick_drag(self, pts, erase=False):
        """Drag in pick mode: manual line through the dragged points (snapped lightly), or erase that span."""
        h = self._active()
        if h is None or not self.source or len(pts) < 1:
            return
        ijs = [self.trace_at(x) for x, _ in pts]
        if any(ij is None for ij in ijs):
            return
        key, pk = self.section_key(), self.picker()
        ii = np.array([ij[0] for ij in ijs])
        jj = np.array([ij[1] for ij in ijs])
        tt = np.array([t for _, t in pts], float)
        if erase:
            along = jj if (key.startswith("2D:") or self.mode == "Inline") else ii
            k = np.arange(along.min(), along.max() + 1)
            if key.startswith("2D:") or self.mode == "Inline":
                pk.erase(h, key, np.full(len(k), ii[0]), k)
            else:
                pk.erase(h, key, k, np.full(len(k), jj[0]))
        else:
            pk.manual_line(h, key, ii, jj, tt, self.snap_mode, min(self.snap_ms, 4.0))
        self._edited()

    def track(self):
        """T: autotrack the active horizon from the seeds on this survey (no seeds: from all its picks)."""
        h = self._active()
        if h is None or not self.source:
            return 0
        key, pk = self.section_key(), self.picker()
        seeds = [(i, j) for k, i, j in self.seeds if k == key]
        p = self.track_params
        p.mode = self.snap_mode
        p.smooth = self.mix_2d if key.startswith("2D:") else self.mix_3d
        line_only = self.track_scope == "This line only"
        if key.startswith("2D:"):
            if line_only:
                n = pk.track(h, key, seeds or None, p)
            else:
                res = pk.track_2d_network(h, [self.source[1]], p)
                n = sum(res.values())
                self.cursorInfo.emit(f"Tracked {h.name} on {len(res)} line(s): {n:,} traces")
        else:
            scope = (self.mode, self.number) if line_only and self.mode in ("Inline", "Crossline") else None
            n = pk.track(h, key, seeds or None, p, scope=scope)
        self.seeds = [s for s in self.seeds if s[0] != key]
        if not key.startswith("2D:") or line_only:
            self.cursorInfo.emit(f"Tracked {h.name}: {n:,} new picks")
        self._edited()
        return n

    def undo(self):
        hs = self.hset()
        if hs is not None:
            name = hs.undo()
            self.cursorInfo.emit(f"Undo ({name})" if name else "Nothing to undo")
            self._edited()

    def draw_horizons(self):
        for it in self._hz_items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._hz_items = []
        hs = self.hset()
        if hs is None or not self.source or self._data is None:
            return
        key = self.section_key()
        kind, name = self.source
        try:
            surv = self.project.volume(name) if kind == "3D" else self.project.line(name)
        except Exception:
            return
        for h in hs.items:
            if not h.visible:
                continue
            width = 3 if h.name == self.active_horizon else 2
            if kind == "3D" and self.mode == "Time slice":
                pts = hz.slice_points(h, key, surv, self.number)
                if pts is None or not len(pts[0]):
                    continue
                it = pg.ScatterPlotItem(pts[0], pts[1], size=4, pen=None, brush=pg.mkBrush(*h.color))
            else:
                c = hz.section_curve(h, key, surv, self.mode, self.number)
                if c is None or not np.isfinite(c[1]).any():
                    continue
                it = pg.PlotCurveItem(np.asarray(c[0], float), np.asarray(c[1], float), connect="finite",
                                      pen=pg.mkPen(h.color, width=width))
            self.plot.addItem(it)
            self._hz_items.append(it)
        for k, i, j in self.seeds:
            if k != key or (kind == "3D" and self.mode == "Time slice"):
                continue
            g = hs.get(self.active_horizon).get(k) if hs.get(self.active_horizon) else None
            if g is None:
                continue
            t = g[j] if g.ndim == 1 else g[i, j]
            if kind == "2D":
                xv = surv.cdp[j]
            elif self.mode == "Inline" and i == surv.il_index(self.number):
                xv = surv.xlines[j]
            elif self.mode == "Crossline" and j == surv.xl_index(self.number):
                xv = surv.inlines[i]
            else:
                continue
            it = pg.ScatterPlotItem([xv], [t], symbol="x", size=10, pen=pg.mkPen((0, 0, 0), width=2))
            self.plot.addItem(it)
            self._hz_items.append(it)

    # ------------------------------------------------------------ for the basemap
    def current_trace_line(self):
        """Map polyline (xs, ys) of the section being shown, or None (time slice)."""
        if not self.source:
            return None
        kind, name = self.source
        if kind == "2D":
            l = self.project.line(name)
            return l.x, l.y
        v = self.project.volume(name)
        if self.mode == "Inline":
            return v.xy([self.number, self.number], [v.xlines[0], v.xlines[-1]])
        if self.mode == "Crossline":
            return v.xy([v.inlines[0], v.inlines[-1]], [self.number, self.number])
        return None
