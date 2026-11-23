"""Section / time-slice viewer for 3D volumes and 2D lines."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtWidgets  # noqa: E402
import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from . import colormaps  # noqa: E402

MODES = ["Inline", "Crossline", "Time slice"]


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

        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.showGrid(x=False, y=True, alpha=0.2)
        lay.addWidget(self.plot, 1)
        self.vb = self.plot.getPlotItem().getViewBox()
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
