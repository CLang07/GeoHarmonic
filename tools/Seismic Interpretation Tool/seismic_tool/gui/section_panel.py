"""Section / time-slice viewer for 3D volumes and 2D lines."""
import math
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtWidgets  # noqa: E402
import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from . import colormaps  # noqa: E402
from .. import faults as flt  # noqa: E402
from .. import horizons as hz  # noqa: E402

MODES = ["Inline", "Crossline", "Time slice", "Stratal slice"]
DOMAINS = ["Time", "Wheeler", "Wheeler (age)", "Depth"]
from ..attributes import ATTRIBUTES as _SEIS_ATTRS  # noqa: E402

ATTRS = ["Seismic", "Inline dip", "Crossline dip", "Planarity", "RGT"] + list(_SEIS_ATTRS)
# 2D lines: either dip = dip along the line; seismic attributes are computed on the fly for the section shown


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
        if self.panel.rgt_mode and ev.button() == QtCore.Qt.LeftButton:
            pt = self.mapToView(ev.pos())
            self.panel.rgt_click(pt.x(), pt.y())
            ev.accept()
            return
        if self.panel.fault_mode and ev.button() == QtCore.Qt.LeftButton:
            pt = self.mapToView(ev.pos())
            self.panel.fault_click(pt.x(), pt.y(), erase=bool(ev.modifiers() & QtCore.Qt.ShiftModifier))
            ev.accept()
            return
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


def _click_through(item):
    """Scatter symbols accept left clicks in pyqtgraph, so Shift+click on a seed / stick point would never reach
    the pick ViewBox: let clicks pass through."""
    for it in (item, getattr(item, "scatter", None)):
        if it is not None:
            try:
                it.mouseClickEvent = lambda ev: ev.ignore()
            except Exception:
                pass
    return item


class SectionPanel(QtWidgets.QWidget):
    positionChanged = QtCore.Signal()  # current line / slice changed (basemap follows)
    cursorInfo = QtCore.Signal(str)
    cursorMoved = QtCore.Signal(object)  # {"x", "y", "t", "il", "xl" | "line", "cdp"} for the linked cursor (23c)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.source = None  # ("3D", name) or ("2D", name)
        self.mode = "Inline"
        self.number = None
        self._img = None
        self._rgba = None  # co-blend / overlay / bump image drawn instead of the LUT image (23a / 23b)
        self.blend = None  # {"attrs": [..], "mode"} | {"overlay": attr, "opacity"} | {"bump": True, ...}
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
        self.track_faults = True  # horizon tracker stops at visible faults
        # fault interpretation
        self.fault_mode = False
        self.active_fault = None
        self.stick = []  # points (x, y, t) of the stick being drawn
        self.stick_key = None  # (source, mode, number) the stick is drawn on
        self.faultsEdited = None
        self.rgt_mode = False  # RGT edit: click grabs an iso-RGT level, second click puts it there
        self._rgt_grab = None
        self._rgt_editors = {}
        self.show_contours = True
        self._ctr_items = []
        self._well_items = []
        self._wheeler = None
        self._age = None  # age-axis Wheeler: {"model", "axis", "labels", "blank", "prop"}
        self.basin_colour = None  # {"prop": temperature|ro|tr|porosity, "t_eval": Ma} colours the age Wheeler
        self._ymap = None
        self._stratal_surface = None
        self.modeChanged = []  # callbacks(mode) when pick/fault/rgt mode switches
        self._flt_items = []
        self.glow = None  # GlowManager (PLAN 39), made once the plot exists
        self._hover_t = 0.0  # hover highlight throttle (initialised here: getattr on a mocked QWidget lies)
        self._hover_hit = None

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
        self.attr_combo = QtWidgets.QComboBox()
        self.attr_combo.addItems(ATTRS)
        self.attr_combo.setToolTip("What to display: seismic amplitude or a computed attribute "
                                   "(Interpret > Compute dip field)")
        self.attr = "Seismic"
        self.dom_combo = QtWidgets.QComboBox()
        self.dom_combo.addItems(DOMAINS)
        self.dom_combo.setToolTip("Wheeler: vertical axis = relative geologic time (needs RGT); gaps = hiatus.\n"
                                  "Wheeler (age): the same on an age axis (Ma) from the Basin model age table "
                                  "(needs two dated horizons on the section)")
        self.domain = "Time"
        self.ctr_chk = QtWidgets.QCheckBox("RGT contours")
        self.ctr_chk.setChecked(True)
        self.ctr_chk.setToolTip("Draw the horizon-stack levels (iso-RGT) on sections")
        for w in (self.src_combo, self.mode_combo, self.num_spin, self.slider, self.step_spin, self.dom_combo,
                  self.attr_combo,
                  self.cmap_combo, self.clip_spin, self.rev_chk):
            bar.addWidget(w)
        bar.addStretch(1)
        lay.addLayout(bar)

        self.vb = PickViewBox(self)
        self.plot = pg.PlotWidget(viewBox=self.vb)
        self.plot.setBackground("w")
        self.plot.showGrid(x=False, y=True, alpha=0.2)
        lay.addWidget(self.plot, 1)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        from .glow import GlowManager
        self.glow = GlowManager(self.plot, lut_fn=self._glow_lut,
                                status=lambda m: self.cursorInfo.emit(m))

        self.src_combo.currentIndexChanged.connect(self._source_changed)
        self.mode_combo.currentTextChanged.connect(self._mode_changed)
        self.num_spin.valueChanged.connect(self._num_changed)
        self.slider.valueChanged.connect(self._slider_changed)
        self.cmap_combo.currentTextChanged.connect(lambda *_: self._apply_colors())
        self.clip_spin.valueChanged.connect(lambda *_: self._apply_colors())
        self.rev_chk.toggled.connect(lambda *_: self._apply_colors())
        self.attr_combo.currentTextChanged.connect(self.set_attr)
        self.dom_combo.currentTextChanged.connect(self.set_domain)
        self.ctr_chk.toggled.connect(self._contours_toggled)
        from PySide6 import QtGui
        for key, d in (("PgUp", -1), ("PgDown", 1), ("Shift+PgUp", -10), ("Shift+PgDown", 10)):
            s = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            s.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            s.activated.connect(lambda d=d: self.step(d))
        for key, fn in (("Return", self.end_stick), ("Enter", self.end_stick), ("Escape", self.cancel_stick)):
            # on the plot only: Enter in the line-number box must still jump to the line
            s = QtGui.QShortcut(QtGui.QKeySequence(key), self.plot)
            s.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            s.activated.connect(fn)

    # ------------------------------------------------------------ setup
    def set_project(self, project):
        if project is not self.project:  # nothing of the previous project may survive (stores, editors)
            self.source = None
            self._data = None
            self._img = None
            self._rgt_editors = {}
            self._rgt_grab = None
            self.stick = []
            self.__dict__.pop("_vm_cache", None)
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
        # PySide6 returns the stored tuple as a list: compare as tuples
        idx = next((i for i in range(self.src_combo.count()) if want and tuple(self.src_combo.itemData(i) or ()) == tuple(want)), 0)
        if self.src_combo.count():
            self.src_combo.setCurrentIndex(idx)
            self._source_changed(idx, number=d.get("number"), mode=d.get("mode"))
        else:
            self.source = None
            self._data = None
            self.plot.clear()

    def show(self, kind, name, mode=None, number=None):
        """Open a 3D volume (with mode/number) or a 2D line."""
        for i in range(self.src_combo.count()):
            if tuple(self.src_combo.itemData(i) or ()) == (kind, name):
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
        elif mode == "Stratal slice":
            st3 = self._stack3d()
            lo, hi, st = 0, (len(st3) - 1 if st3 is not None else 0), 1
            if st3 is None:
                self.cursorInfo.emit("No horizon stack yet - Interpret > Build horizon stack (3D)")
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
            if kind == "3D" and self.mode == "Stratal slice":
                arr, x, y, xl, yl = self._stratal()
            elif self.attr != "Seismic":
                arr, x, y, xl, yl = self._attribute_section()
            elif kind == "3D":
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
            self._ymap = None
            if self.wheeler_on():
                arr, y, yl = self._to_wheeler(arr)
            elif self.depth_on():
                arr, y, yl = self._to_depth(arr)
        except Exception as e:
            self.cursorInfo.emit(f"Could not read data: {e}")
            return
        self._data, self._xaxis, self._yaxis = arr, x, y
        self._rgba = None
        if self.blend:
            try:
                self._rgba = self.blend_image(arr)
            except Exception as e:
                self.cursorInfo.emit(f"Co-blend not shown: {e}")
        view = None if reset_view else self.vb.viewRange()
        if self.glow is not None:
            self.glow.clear()  # plot.clear() removes the underlays too
        self.plot.clear()
        self._link_items = []
        # autoDownsample: big sections (long 2D lines, large time slices) render at screen resolution when zoomed out
        self._img = pg.ImageItem(arr if self._rgba is None else self._rgba, autoDownsample=True)
        dx = (x[-1] - x[0]) / max(1, len(x) - 1) if len(x) > 1 else 1
        dy = (y[-1] - y[0]) / max(1, len(y) - 1) if len(y) > 1 else 1
        self._img.setRect(QtCore.QRectF(x[0] - dx / 2, y[0] - dy / 2, dx * len(x), dy * len(y)))
        self.plot.addItem(self._img)
        self.plot.setLabel("bottom", xl)
        self.plot.setLabel("left", yl)
        self.plot.getPlotItem().invertY(self.mode not in ("Time slice", "Stratal slice") or kind == "2D")
        self._apply_colors()
        self._hz_items = []
        self._flt_items = []
        self._ctr_items = []
        self._well_items = []
        self.draw_contours()
        self.draw_wells()
        self.draw_horizons()
        if self.age_on():
            self.draw_age_labels()
        self.draw_faults()
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
        if self.mode == "Stratal slice" and kind == "3D":
            t = f"{name} - Stratal slice (stack level {self.number})"
        if self.attr != "Seismic":
            t += f"  [{self.attr}{' ms/trace' if 'dip' in self.attr else ''}]"
        if self.wheeler_on():
            t += "  [Wheeler, age]" if self.age_on() else "  [Wheeler]"
            if self.age_on() and self.basin_colour:
                t += f"  [{self.basin_colour['prop']} at {float(self.basin_colour.get('t_eval', 0)):g} Ma]"
        if self.depth_on():
            t += "  [Depth]"
        self.plot.setTitle(t)

    # ------------------------------------------------------------ RGT views (Wheeler, stratal slices)
    def set_domain(self, name):
        self.domain = name if name in DOMAINS else "Time"
        self.redraw(reset_view=True)

    def wheeler_on(self):
        return self.domain in ("Wheeler", "Wheeler (age)") and (self.source[0] == "2D" or self.mode in ("Inline", "Crossline"))

    def depth_on(self):
        return self.domain == "Depth" and (self.source[0] == "2D" or self.mode in ("Inline", "Crossline"))

    def velocity_model(self):
        """The project's velocity model (Interpret > Velocity model), cached; None if not defined."""
        d = self.project.display.get("velocity_model") if self.project else None
        if not d:
            return None
        dm = self.depth_model()
        return dm.vm if dm is not None else None

    def depth_model(self):
        """The project's velocity model as a ddc.DynamicDepthModel (cached per model settings); None if undefined."""
        d = self.project.display.get("velocity_model") if self.project else None
        if not d:
            return None
        key = repr(d)
        cache = self.__dict__.setdefault("_vm_cache", {})
        if key not in cache:
            from ..ddc import DynamicDepthModel
            cache.clear()
            cache[key] = DynamicDepthModel(self.project, d).build()
        return cache[key]

    def adopt_depth_model(self, dm):
        """Share a DynamicDepthModel built elsewhere (velocity dialog 'Use') for the stored model settings."""
        d = self.project.display.get("velocity_model") if self.project else None
        if d and dm is not None and dm.vm.to_dict() == d:
            cache = self.__dict__.setdefault("_vm_cache", {})
            cache.clear()
            cache[repr(d)] = dm

    def ddc_refresh(self, force=False):
        """Dynamic depth conversion: rebuild the changed part of the depth model; redraw a depth section.
        Returns the refresh report or None (no model / nothing changed)."""
        dm = self.depth_model()
        if dm is None:
            return None
        rep = dm.refresh(force)
        if rep and rep.get("from_layer") is not None and self.domain == "Depth" and self.project is not None:
            try:
                self.redraw()
            except Exception:
                pass
        return rep

    def section_xy(self):
        """Map positions (x, y) of the traces of the section shown."""
        kind, name = self.source
        if kind == "2D":
            l = self.project.line(name)
            return l.x, l.y
        v = self.project.volume(name)
        if self.mode == "Inline":
            return v.xy(np.full(v.nxl, self.number), v.xlines)
        return v.xy(v.inlines, np.full(v.nil, self.number))

    def _to_depth(self, arr):
        vm = self.velocity_model()
        if vm is None:
            raise RuntimeError("no velocity model - Interpret > Velocity model")
        kind, name = self.source
        times = self.project.volume(name).times if kind == "3D" else self.project.line(name).times
        x, y = self.section_xy()
        from ..velocity import _monotonic_depths
        zt = _monotonic_depths(vm.depth_of_time(x, y, times))
        if not np.isfinite(zt).any():
            raise RuntimeError("this section is outside the velocity model")
        dz = max(1.0, float(np.nanmedian(np.diff(zt, axis=1))))
        z = np.arange(0.0, float(np.nanmax(zt)) + dz / 2, dz)
        out = np.full((len(z), arr.shape[1]), np.nan, np.float32)
        for j in range(arr.shape[1]):
            if np.all(np.isfinite(zt[j])):
                out[:, j] = np.interp(z, zt[j], arr[:, j], left=np.nan, right=np.nan)
        self._ymap = (zt, times)
        return out, z, "Depth TVDSS (m)"

    def _rgt_folder(self, kind, name):
        return os.path.join(self.project.volume(name).store, "rgt") if kind == "3D" else \
            os.path.join(self.project.seismic_dir(), "rgt")

    def _stack3d(self):
        from ..rgt.stack import HorizonStack3D
        if not self.source or self.source[0] != "3D":
            return None
        folder = self._rgt_folder(*self.source)
        return HorizonStack3D(folder) if HorizonStack3D.exists(folder) else None

    def rgt_section(self):
        """(n traces, ns) RGT of the section shown (time domain)."""
        from ..rgt.solve import RGTVolume, load_rgt_line
        kind, name = self.source
        folder = self._rgt_folder(kind, name)
        if kind == "2D":
            tau = load_rgt_line(folder, name)
            if tau is None:
                raise RuntimeError("no RGT for this line - Interpret > Compute RGT (2D lines)")
            return tau
        if not RGTVolume.exists(folder):
            raise RuntimeError("no RGT for this volume - Interpret > Compute RGT (3D)")
        return RGTVolume(folder).section(self.project.volume(name), self.mode, self.number).T

    def wheeler_levels(self, tau):
        from ..rgt.stack import default_levels, load_stack_2d
        kind, name = self.source
        if kind == "3D":
            st = self._stack3d()
            if st is not None:
                return st.levels
        else:
            st = load_stack_2d(self._rgt_folder(kind, name))
            if st is not None:
                return st["levels"]
        return default_levels(tau, 200)

    def _to_wheeler(self, arr):
        """Time-domain display array (ns, n) -> Wheeler (levels, n)."""
        from ..rgt.stack import wheeler
        tau = self.rgt_section()
        kind, name = self.source
        times = self.project.volume(name).times if kind == "3D" else self.project.line(name).times
        levels = self.wheeler_levels(tau)
        amp, hiatus = wheeler(np.asarray(arr, np.float32).T, tau, levels, times)
        self._wheeler = (tau, times, levels)
        self._age = None
        if self.age_on():
            return self._to_age(amp, hiatus, tau, times, levels)
        self._ymap = (tau, times)
        return amp, levels, "RGT (relative geologic time)"

    def age_on(self):
        return self.domain == "Wheeler (age)" and self.wheeler_on()

    def section_horizon_times(self):
        """{horizon name: times (ntr,) ms} of the horizons picked on the displayed section."""
        hs = self.hset()
        out = {}
        if hs is None or not self.source:
            return out
        kind, name = self.source
        surv = self.project.volume(name) if kind == "3D" else self.project.line(name)
        key = self.section_key()
        for h in hs.items:
            c = hz.section_curve(h, key, surv, self.mode, self.number)
            if c is not None and np.isfinite(c[1]).any():
                out[h.name] = np.asarray(c[1], float)
        return out

    def _to_age(self, amp, hiatus, tau, times, levels):
        """Wheeler (levels, n) -> age axis (Ma) using the Basin model age table (PLAN 18a)."""
        from ..basin import chrono as C
        table = C.AgeTable.load(self.project)
        try:
            model, rgts = C.section_age_model(table, tau, times, self.section_horizon_times())
        except ValueError as e:
            raise RuntimeError(f"age axis: {e} - pick horizons named as in Interpret > Basin model "
                               "(age table) or set their 'horizon' names") from None
        la = C.level_ages(levels, model)
        axis = C.age_axis(la)
        W, blank = C.wheeler_age(amp, hiatus, la, axis, model.gaps())
        x = self._section_x()
        prop = None
        if self.basin_colour:
            from ..basin import model as BM
            vm = self.velocity_model()
            if vm is None:
                raise RuntimeError("Wheeler colouring needs a depth model (Interpret > Velocity model, Use)")
            bc = self.basin_colour
            sx, sy = self.section_xy()
            P = BM.section_property(vm, table, sx, sy, axis, bc["prop"], float(bc.get("t_eval", 0.0)),
                                    BM.settings(self.project))
            W = np.where(blank, np.nan, P).astype(np.float32)
            prop = bc["prop"]
        self._age = {"model": model, "axis": axis, "blank": blank, "rgts": rgts, "prop": prop,
                     "labels": C.gap_labels(blank, axis, x, min_my=2.0 * (axis[1] - axis[0]) if len(axis) > 1 else 0)}
        self._ymap = (model(tau), times)
        return W, axis, "Age (Ma)"

    def _section_x(self):
        kind, name = self.source
        if kind == "2D":
            return self.project.line(name).cdp
        v = self.project.volume(name)
        return v.xlines if self.mode == "Inline" else v.inlines

    def age_labels(self, max_labels=12):
        """Hiatus labels [(x, age, text)] of the age Wheeler shown (largest gaps first)."""
        if not self._age:
            return []
        return [(d["x"], d["age"], d["text"]) for d in self._age["labels"][:max_labels]]

    def draw_age_labels(self):
        for x, a, text in self.age_labels():
            it = pg.TextItem(text, color=(120, 0, 0), anchor=(0.5, 0.5))
            it.setPos(float(x), float(a))
            self.plot.addItem(it)
            self._hz_items.append(it)

    def _stratal(self):
        from ..rgt.stack import stratal_slice
        st = self._stack3d()
        if st is None:
            raise RuntimeError("no horizon stack yet - Interpret > Build horizon stack (3D)")
        v = self.project.volume(self.source[1])
        k = int(np.clip(self.number or 0, 0, len(st) - 1))
        surf = st.surface(k)
        self._stratal_surface = surf
        return stratal_slice(v, surf), v.xlines, v.inlines, "Crossline", "Inline"

    # ------------------------------------------------------------ co-blend / overlay / bump (PLAN 23a / 23b)
    def set_blend(self, spec):
        """spec None (off) | {"attrs": [a, b(, c)], "mode": rgb|cmy|hsv, "clip": (lo, hi), "gamma": g}
        | {"overlay": attr, "opacity": 0.5, "lut": name, "threshold": None} | {"bump": True, "azimuth", "elevation", "ve"}"""
        self.blend = dict(spec) if spec else None
        if self.project is not None:
            self.project.display["section_blend"] = self.blend
        self.redraw()

    def blend_image(self, arr):
        """rgba image for the current section according to self.blend (None when not applicable)."""
        from .. import blend as B
        sp = self.blend or {}
        if not sp:
            return None
        if sp.get("bump"):
            return B.bump_section(arr, sp.get("azimuth", 315.0), sp.get("elevation", 45.0), sp.get("ve", 1.0),
                                  sp.get("strength", 0.6))
        kind, name = self.source
        if kind != "3D" or self.mode not in ("Inline", "Crossline", "Time slice") or self.wheeler_on() or self.depth_on():
            raise ValueError("co-blend / overlay work on 3D inlines, crosslines and time slices in time")
        v = self.project.volume(name)
        if sp.get("overlay") or sp.get("volume"):
            if sp.get("volume"):  # 40c: a second project volume (coherence, dip, inversion ...) over the amplitude
                ov = self.project.volume(sp["volume"])
                if self.mode == "Inline":
                    attr = np.asarray(ov.inline(self.number), float).T
                elif self.mode == "Crossline":
                    attr = np.asarray(ov.xline(self.number), float).T
                else:
                    attr = np.asarray(ov.time_slice(self.number), float)
            else:
                from .. import attributes as A
                attr = np.asarray(A.section_attribute(v, sp["overlay"], self.mode, self.number), float)
            if sp.get("mode") in ("multiply", "transparency", "alpha"):
                from .. import aesthetics as AE
                from .. import colortables as CT
                t = CT.get(self.cmap_combo.currentText() or "Seismic (red-white-blue)") or CT.get("Grey")
                t.clip, t.flip = float(self.clip_spin.value() or 99.0), bool(self.rev_chk.isChecked())
                lo, hi = t.levels(arr)
                base = B.lut_rgba(arr, t.lut(), lo, hi)  # the amplitude keeps its own colour table + clip
                if attr.shape != np.shape(arr):
                    raise ValueError(f"overlay volume shape {attr.shape} differs from the section {np.shape(arr)}")
                return AE.corender(base, attr, colormaps.lut(sp.get("lut", "Rainbow (attributes)"), False),
                                   sp["mode"], float(sp.get("opacity", 0.5)), clip=tuple(sp.get("clip", (1.0, 99.0))),
                                   invert_alpha=bool(sp.get("invert", False)))
            base = B.gray_rgba(arr)
            return B.overlay(base, attr, colormaps.lut(sp.get("lut", "Rainbow (attributes)"), False),
                             float(sp.get("opacity", 0.5)), threshold=sp.get("threshold"))
        rgba, _ = B.section_blend(v, sp["attrs"], self.mode, self.number, sp.get("mode", "rgb"),
                                  tuple(sp.get("clip", (2.0, 98.0))), sp.get("gamma", 1.0))
        return rgba

    def scan_frames(self, attr, param, values):
        """Attribute-parameter movie (23c): [(value, array)] of attr on the current 3D section for each parameter
        value, e.g. scan_frames("Spectral 20 Hz", "freq", range(10, 61, 5))."""
        from .. import attributes as A
        from .. import blend as B
        kind, name = self.source
        if kind != "3D" or self.mode not in ("Inline", "Crossline", "Time slice"):
            raise ValueError("parameter scan works on 3D inlines, crosslines and time slices")
        v = self.project.volume(name)
        return list(B.param_scan(lambda p: np.asarray(A.section_attribute(v, attr, self.mode, self.number, {param: p})),
                                 values))

    def show_frame(self, arr):
        """Display one scan frame (keeps the view)."""
        if self._img is None:
            return
        self._data = np.asarray(arr)
        self._rgba = None
        self._img.setImage(self._data, autoLevels=False)
        self._apply_colors()

    def set_attr(self, name):
        self.attr = name if name in ATTRS else "Seismic"
        self.redraw()

    def _attribute_section(self):
        """(array, x, y, xlabel, ylabel) of the selected attribute on the current section."""
        from ..rgt.dip import DipField, line_dip
        from ..rgt.solve import RGTVolume, load_rgt_line
        from .. import attributes as A
        kind, name = self.source
        if self.attr in A.ATTRIBUTES:
            if kind == "2D":
                l = self.project.line(name)
                key = (name, self.attr, id(l))
                cache = self.__dict__.setdefault("_line_attr_cache", {})
                if key not in cache:
                    cache.clear()
                    folder = os.path.join(self.project.seismic_dir(), "rgt") if self.project.path else None
                    cache[key] = A.line_attribute(l, self.attr, None, folder)
                return cache[key].T, l.cdp, l.times, "CDP", "TWT (ms)"
            v = self.project.volume(name)
            arr = A.section_attribute(v, self.attr, self.mode, self.number)
            if self.mode == "Inline":
                return arr, v.xlines, v.times, "Crossline", "TWT (ms)"
            if self.mode == "Crossline":
                return arr, v.inlines, v.times, "Inline", "TWT (ms)"
            return arr, v.xlines, v.inlines, "Crossline", "Inline"
        if self.attr == "RGT":
            if kind == "2D":
                l = self.project.line(name)
                arr = load_rgt_line(os.path.join(self.project.seismic_dir(), "rgt"), name)
                if arr is None:
                    raise RuntimeError("no RGT for this line yet - Interpret > Compute RGT (2D lines)")
                return arr.T, l.cdp, l.times, "CDP", "TWT (ms)"
            v = self.project.volume(name)
            folder = os.path.join(v.store, "rgt")
            if not RGTVolume.exists(folder):
                raise RuntimeError("no RGT for this volume yet - Interpret > Compute RGT (3D)")
            arr = RGTVolume(folder).section(v, self.mode, self.number)
            if self.mode == "Inline":
                return arr, v.xlines, v.times, "Crossline", "TWT (ms)"
            if self.mode == "Crossline":
                return arr, v.inlines, v.times, "Inline", "TWT (ms)"
            return arr, v.xlines, v.inlines, "Crossline", "Inline"
        if kind == "2D":
            l = self.project.line(name)
            dip, plan = line_dip(l, os.path.join(self.project.seismic_dir(), "rgt") if self.project.path else None)
            arr = (plan if self.attr == "Planarity" else dip).T
            return arr, l.cdp, l.times, "CDP", "TWT (ms)"
        v = self.project.volume(name)
        folder = os.path.join(v.store, "rgt")
        if not DipField.exists(folder):
            raise RuntimeError("no dip field for this volume yet - Interpret > Compute dip field")
        df = DipField(folder)
        key = {"Inline dip": "dip_il", "Crossline dip": "dip_xl", "Planarity": "planarity"}[self.attr]
        arr = df.section(key, v, self.mode, self.number)
        if self.mode == "Inline":
            return arr, v.xlines, v.times, "Crossline", "TWT (ms)"
        if self.mode == "Crossline":
            return arr, v.inlines, v.times, "Inline", "TWT (ms)"
        return arr, v.xlines, v.inlines, "Crossline", "Inline"

    def _level_sample(self, limit=1_000_000):
        """Finite values of the shown section, strided down to ~limit values (percentiles of a 60-Mpixel
        2D line cost ~1 s; a regular sample gives the same clip to well under 1 %)."""
        a = np.asarray(self._data).ravel()
        if a.size > limit:
            a = a[::int(math.ceil(a.size / limit))]
        a = a.astype(np.float32, copy=False)
        return a[np.isfinite(a)]

    def color_levels(self):
        """(lo, hi) display levels: symmetric percentile clip, or 0..1 for planarity."""
        if self.attr == "Planarity" or self.attr.startswith("Coherence"):
            return 0.0, 1.0
        age = getattr(self, "_age", None)  # stand-in objects in tests may lack it
        if age and age.get("prop") and self.age_on():
            a = self._data[np.isfinite(self._data)]
            return (float(a.min()), float(max(a.max(), a.min() + 1e-6))) if a.size else (0.0, 1.0)
        if self.attr == "Inst. phase":
            return -180.0, 180.0
        if self.attr == "Seismic" and _positive_volume(self):
            a = self._level_sample()
            a = a[np.isfinite(a)] if a.size else a
            return (float(np.percentile(a, 1)), float(np.percentile(a, 99))) if a.size else (0.0, 1.0)
        if self.attr in ("Envelope", "Inst. frequency", "RMS amplitude") or self.attr.startswith("Spectral"):
            a = self._level_sample()
            return (float(np.percentile(a, 1)), float(np.percentile(a, 99))) if a.size else (0.0, 1.0)
        if self.attr == "RGT":
            a = self._level_sample()
            return (float(a.min()), float(a.max())) if a.size else (0.0, 1.0)
        a = np.abs(self._level_sample())
        clip = float(np.percentile(a, self.clip_spin.value() or 99.0)) if a.size else 1.0
        clip = clip or 1.0
        return -clip, clip

    def _apply_colors(self):
        if self._img is None or self._data is None:
            return
        if self._rgba is not None:  # true-colour image: no LUT / levels
            self._img.setLookupTable(None)
            self._img.setLevels([0, 255])
            return
        cmap = self.cmap_combo.currentText()
        age = getattr(self, "_age", None)
        if (age and age.get("prop") and self.age_on()) or (self.attr == "Seismic" and _positive_volume(self)):
            cmap = "Rainbow (attributes)"
        elif self.attr != "Seismic":
            cmap = "Grey" if (self.attr == "Planarity" or self.attr.startswith("Coherence")) else \
                self.cmap_combo.currentText() if self.attr == "Cosine of phase" else "Rainbow (attributes)"
        self._img.setLookupTable(colormaps.lut(cmap, self.rev_chk.isChecked()))
        self._img.setLevels(self.color_levels())
        if self.project:
            self.project.display.update(cmap_name=str(self.cmap_combo.currentText()), clip=float(self.clip_spin.value()),
                                        reverse=bool(self.rev_chk.isChecked()))

    # ------------------------------------------------------------ glow (PLAN 39)
    def _glow_lut(self):
        try:
            return colormaps.lut(self.cmap_combo.currentText(), self.rev_chk.isChecked())
        except Exception:
            return None

    def set_glow_state(self, state):
        """Share the main window's GlowState (map bus selection / active emphasis, settings)."""
        if self.glow is not None:
            self.glow.set_state(state)

    def glow_curves(self):
        """Interpretation on the current section for the figure export: [{x, y, colour, width, kind, name}]."""
        out = []
        for e in (self.glow.entries.values() if self.glow is not None else []):
            try:
                d = e.parent.getData()
                x, y = np.asarray(d[0], float), np.asarray(d[1], float)
            except Exception:
                continue
            if e.connect == "pairs":  # fault cuts: segment pairs -> NaN-separated polyline
                n = len(x) // 2 * 2
                x = np.insert(x[:n].reshape(-1, 2), 2, np.nan, axis=1).ravel()
                y = np.insert(y[:n].reshape(-1, 2), 2, np.nan, axis=1).ravel()
            out.append({"x": x, "y": y, "colour": e.colour, "width": e.width, "kind": e.kind, "name": e.name,
                        "points": e.symbol_size is not None})
        return out

    def export_figure(self, path, with_glow=True, title=None):
        """File > Export section image: PNG / PDF / SVG of the section with horizons, faults and wells as on screen,
        glow included unless "Export without glow" (PLAN 39g)."""
        from .. import glow as G
        if self._data is None:
            raise RuntimeError("no section shown")
        st = self.glow.settings if self.glow is not None else G.GlowSettings()
        emph = {}
        if self.glow is not None:
            for e in self.glow.entries.values():
                emph[(e.kind, e.name)] = self.glow.state.emphasis(e.kind, e.name)
        x = np.asarray(self._xaxis, float)
        y = np.asarray(self._yaxis, float)
        return G.export_section_png(path, np.asarray(self._data, float).T, x, y, self.glow_curves(),
                                    lut=self._glow_lut(), clip=float(self.clip_spin.value() or 99.0), settings=st,
                                    with_glow=with_glow, title=title if title is not None else self._title_text(),
                                    xlabel="Trace", ylabel="TWT (ms)", emphasis=emph)

    def hover_at(self, xv, yv, tol_px=6.0, now=None):
        """Hover highlight (PLAN 39e): the horizon / fault curve under the cursor glows (throttled; off in the
        settings).  Returns (kind, name) or None."""
        import time as _time
        if self.glow is None or not self.glow.settings["hover"] or not self.glow.settings["enabled"]:
            return None
        now = _time.time() if now is None else now
        if now - self._hover_t < self.glow.settings["hover_ms"] / 1000.0:
            return self._hover_hit
        self._hover_t = now
        try:
            px = self.vb.viewPixelSize()
            tol = float(tol_px) * abs(float(px[1])) if px is not None else 1.0
        except Exception:
            tol = 1.0
        best, hit = tol if tol > 0 else 1.0, None
        for e in self.glow.entries.values():
            if e.kind not in ("horizon", "fault") or e.symbol_size is not None:
                continue
            try:
                d = e.parent.getData()
                xs, ys = np.asarray(d[0], float), np.asarray(d[1], float)
            except Exception:
                continue
            ok = np.isfinite(xs) & np.isfinite(ys)
            if ok.sum() < 2:
                continue
            xs, ys = xs[ok], ys[ok]
            o = np.argsort(xs)
            if not (xs[o][0] <= xv <= xs[o][-1]):
                continue
            dy = abs(float(np.interp(xv, xs[o], ys[o])) - yv)
            if dy <= best:
                best, hit = dy, (e.kind, e.name)
        self._hover_hit = hit
        if hit != self.glow.state.hover:
            self.glow.state.set_hover(*(hit or (None, None)))
        return hit

    def auto_colour(self, kind, name, existing=(), unit=None, age_ma=None):
        """PLAN 40j: maximally distinct colour for a new horizon / fault that avoids the current seismic colour
        table (ICS colour when an age / unit is known); locked colours are kept."""
        from .. import aesthetics as AE
        return AE.ColourAssigner(self.project).colour_for(kind, name, existing, self._glow_lut(), unit, age_ma)

    def section_length_m(self):
        """Horizontal length of the shown section in metres (scale bar of the figure export)."""
        try:
            kind, name = self.source
            if kind == "2D":
                l = self.project.line(name)
                return float(np.hypot(np.diff(l.x), np.diff(l.y)).sum())
            v = self.project.volume(name)
            if self.mode == "Inline":
                a, b = v.xy(float(self.number), float(v.xlines[0])), v.xy(float(self.number), float(v.xlines[-1]))
            elif self.mode == "Crossline":
                a, b = v.xy(float(v.inlines[0]), float(self.number)), v.xy(float(v.inlines[-1]), float(self.number))
            else:
                a, b = v.xy(float(v.inlines[0]), float(v.xlines[0])), v.xy(float(v.inlines[0]), float(v.xlines[-1]))
            return float(np.hypot(b[0] - a[0], b[1] - a[1]))
        except Exception:
            return None

    def export_page(self, path, template=None, with_glow=None):
        """PLAN 40i: one-click figure of the section on a page preset with title block, scale bar, colour bar and
        legend (horizons / faults); glow per the template (39g)."""
        from .. import aesthetics as AE
        from .. import glow as G
        if self._data is None:
            raise RuntimeError("no section shown")
        t = AE.default_template()
        t.update(template or {})
        if not t.get("title"):
            t["title"] = self._title_text()
        if not t.get("project") and self.project is not None:
            t["project"] = getattr(self.project, "name", "")
        st = self.glow.settings if self.glow is not None else G.GlowSettings()
        wg = t.get("glow", True) if with_glow is None else with_glow
        lut = self._glow_lut()
        curves = self.glow_curves()
        box = {}

        def draw(ax):
            n, c = G.draw_section(ax, np.asarray(self._data, float).T, np.asarray(self._xaxis, float),
                                  np.asarray(self._yaxis, float), curves, lut, float(self.clip_spin.value() or 99.0),
                                  st, wg, int(t["dpi"]))
            box["clip"] = c
            ax.set_xlabel("Crossline" if self.mode == "Inline" else "Inline" if self.mode == "Crossline" else "Trace")
            ax.set_ylabel("TWT (ms)" if self.mode != "Time slice" else "Inline")
        seen, legend = set(), []
        for cv in curves:
            if cv["kind"] in ("horizon", "fault") and (cv["kind"], cv["name"]) not in seen:
                seen.add((cv["kind"], cv["name"]))
                legend.append((cv["name"], cv["colour"]))
        res = AE.compose_figure(path, draw=draw, template=t, legend=legend,
                                colourbar=(lut, -1.0, 1.0, "amplitude (relative)") if lut is not None else None,
                                x_extent=self.section_length_m(), kind="section")
        return res

    def _title_text(self):
        if not self.source:
            return ""
        return f"{self.source[1]} - {self.mode} {self.number if self.number is not None else ''}".strip()

    def _mouse_moved(self, pos):
        if self._data is None or not self.vb.sceneBoundingRect().contains(pos):
            return
        pt = self.vb.mapSceneToView(pos)
        self.hover_at(pt.x(), pt.y())
        x, y = self._xaxis, self._yaxis
        j = int(np.clip(np.searchsorted(x, pt.x()), 0, len(x) - 1))
        i = int(np.clip(np.searchsorted(y, pt.y()), 0, len(y) - 1))
        amp = self._data[i, j]
        kind, name = self.source
        if kind == "3D":
            v = self.project.volume(name)
            tl = "RGT" if self.wheeler_on() else "t"
            if self.mode == "Inline":
                il, xl, t = self.number, x[j], y[i]
            elif self.mode == "Crossline":
                il, xl, t = x[j], self.number, y[i]
            elif self.mode == "Stratal slice":
                il, xl = y[i], x[j]
                surf = self._stratal_surface
                t = float(surf[i, j]) if surf is not None else np.nan
                tl = f"level {self.number}  t"
            else:
                il, xl, t = y[i], x[j], self.number
            X, Y = v.xy(il, xl)
            self.cursorMoved.emit({"x": float(X), "y": float(Y), "t": float(t) if self.mode != "Stratal slice" else None,
                                   "il": float(il), "xl": float(xl), "domain": "depth" if self.depth_on() else "time"})
            self.cursorInfo.emit(f"IL {il:.0f}  XL {xl:.0f}  {tl} {t:.0f}{'' if tl == 'RGT' else ' ms'}  amp {amp:.4g}"
                                 f"   X {X:.1f}  Y {Y:.1f}")
        else:
            l = self.project.line(name)
            self.cursorMoved.emit({"x": float(l.x[j]), "y": float(l.y[j]), "t": float(y[i]), "line": name,
                                   "cdp": float(x[j]), "domain": "depth" if self.depth_on() else "time"})
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
        """(i, j) grid indices of the trace nearest to section x value, or None (time/stratal slice, Wheeler)."""
        kind, name = self.source
        if self.wheeler_on() or self.depth_on() or self.mode == "Stratal slice":
            return None
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
                n = pk.track(h, key, seeds or None, p, faults=self._track_faults())
            else:
                res = pk.track_2d_network(h, [self.source[1]], p, faults=self._track_faults())
                n = sum(res.values())
                self.cursorInfo.emit(f"Tracked {h.name} on {len(res)} line(s): {n:,} traces")
        else:
            scope = (self.mode, self.number) if line_only and self.mode in ("Inline", "Crossline") else None
            n = pk.track(h, key, seeds or None, p, scope=scope, faults=self._track_faults())
        self.seeds = [s for s in self.seeds if s[0] != key]
        if not key.startswith("2D:") or line_only:
            self.cursorInfo.emit(f"Tracked {h.name}: {n:,} new picks")
        self._edited()
        return n

    def _track_faults(self):
        fs = self.fset()
        return fs.visible_faults() if (fs is not None and self.track_faults) else None

    def undo(self):
        """Ctrl+Z: undo the most recent horizon or fault edit."""
        if self.stick:
            self.stick = self.stick[:-1]
            self.cursorInfo.emit("Removed last stick point")
            self.draw_faults()
            return
        hs, fs = self.hset(), self.fset()
        hseq = hs.last_seq() if hs is not None else 0
        fseq = fs.last_seq() if fs is not None else 0
        ed = max(self._rgt_editors.values(), key=lambda e: e.last_seq(), default=None)
        rseq = ed.last_seq() if ed is not None else 0
        if not hseq and not fseq and not rseq:
            self.cursorInfo.emit("Nothing to undo")
            return
        if rseq > max(hseq, fseq):
            ed.undo()
            self.cursorInfo.emit("Undo RGT edit")
            self.redraw()
            return
        if fseq > hseq:
            name = fs.undo()
            if self.active_fault and not fs.get(self.active_fault):
                self.active_fault = fs.names()[-1] if fs.items else None
            self.cursorInfo.emit(f"Undo fault edit ({name})")
            self._faults_edited()
        else:
            name = hs.undo()
            self.cursorInfo.emit(f"Undo ({name})")
            self._edited()

    # ------------------------------------------------------------ modes
    def set_mode(self, mode):
        """mode: 'horizon', 'fault' or None. Horizon picking and fault sticks are exclusive."""
        if mode != "fault" and self.stick:
            self.end_stick()
        self.pick_mode = mode == "horizon"
        self.fault_mode = mode == "fault"
        self.rgt_mode = mode == "rgt"
        self._rgt_grab = None
        for cb in self.modeChanged:
            cb(mode)

    # ------------------------------------------------------------ wells
    def well_overlay(self):
        """Pure data: [{name, x, t_top, t_bottom, tops: [(formation, twt, colour)]}] for wells on/near the
        section (3D inline/crossline: within 3 traces; 2D: within 500 m; time slice: map position)."""
        if not self.project or not self.project.well_project or not self.source or self.wheeler_on():
            return []
        kind, name = self.source
        try:
            ws = self.project.wellset()
            surv = self.project.volume(name) if kind == "3D" else self.project.line(name)
            near = ws.near_section(kind, surv, self.mode, self.number)
        except Exception as e:
            self.cursorInfo.emit(f"Wells not shown: {e}")
            return []
        out = []
        for wn, xv, dist in near:
            try:
                td = ws.td_for(wn)
                w = ws.well(wn)
                z_td = float(w.md_to_tvdss(w.td)) if w.td else float(td.z[-1])
                if self.depth_on():  # depth display: tops at their TVDSS
                    md_tops = (ws.wcp.tops or {}).get(wn, {})
                    tops = [(fm, float(w.md_to_tvdss(md)), ws.formation_color(fm)) for fm, md in md_tops.items()
                            if md is not None]
                    bottom = z_td
                else:
                    tops = [(fm, t, ws.formation_color(fm)) for fm, t in ws.tops_twt(wn).items()]
                    bottom = float(td.twt(z_td))
                out.append({"name": wn, "x": xv, "dist": dist, "t_top": 0.0, "t_bottom": bottom,
                            "tops": sorted(tops, key=lambda a: a[1]), "td_source": td.source})
            except Exception:
                continue
        return out

    def draw_wells(self):
        if self.glow is not None:
            self.glow.clear("well")
            self.glow.clear("well_top")
        for it in self._well_items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._well_items = []
        if self._data is None:
            return
        slice_view = self.source and self.source[0] == "3D" and self.mode in ("Time slice", "Stratal slice")
        for w in self.well_overlay():
            if slice_view:
                xv, yv = w["x"]
                items = [pg.ScatterPlotItem([xv], [yv], symbol="o", size=10, brush=pg.mkBrush(0, 0, 0),
                                            pen=pg.mkPen("w"))]
                t = pg.TextItem(w["name"], color=(0, 0, 0), anchor=(-0.1, 1.1))
                t.setPos(xv, yv)
                items.append(t)
            else:
                items = [pg.PlotCurveItem([w["x"], w["x"]], [w["t_top"], w["t_bottom"]], pen=pg.mkPen((0, 0, 0), width=2))]
                t = pg.TextItem(w["name"], color=(0, 0, 0), anchor=(0.5, 1.0))
                t.setPos(w["x"], w["t_top"])
                items.append(t)
                dx = max(1.0, 0.01 * float(np.ptp(self._xaxis))) if self._xaxis is not None else 1.0
                for fm, tw, col in w["tops"]:
                    items.append(pg.PlotCurveItem([w["x"] - dx, w["x"] + dx], [tw, tw], pen=pg.mkPen(col, width=3)))
                    lab = pg.TextItem(fm, color=col, anchor=(0, 0.5))
                    lab.setPos(w["x"] + dx, tw)
                    items.append(lab)
            for it in items:
                self.plot.addItem(it, ignoreBounds=True) if isinstance(it, pg.TextItem) else self.plot.addItem(it)
                self._well_items.append(it)
            if self.glow is not None:  # PLAN 39e: wells on seismic glow (path + tops)
                if slice_view:
                    self.glow.attach(items[0], "well", w["name"], (0, 0, 0), symbol_size=10)
                else:
                    self.glow.attach(items[0], "well", w["name"], (0, 0, 0), 2)
                    for it, (fm, tw, col) in zip([i for i in items[2:] if not isinstance(i, pg.TextItem)], w["tops"]):
                        self.glow.attach(it, "well_top", f"{w['name']}:{fm}", col, 3)

    # ------------------------------------------------------------ RGT editing + contours
    def rgt_editor(self):
        """Editor for the RGT model of the survey shown (cached), or None."""
        from ..rgt.edit import RGTEditor2D, RGTEditor3D
        key = self.section_key()
        if key in self._rgt_editors:
            return self._rgt_editors[key]
        kind, name = self.source
        try:
            ed = RGTEditor3D(self.project.volume(name)) if kind == "3D" else RGTEditor2D(self.project, name)
        except Exception as e:
            self.cursorInfo.emit(f"No RGT model to edit here ({e})")
            return None
        self._rgt_editors[key] = ed
        return ed

    def _contours_toggled(self, on):
        self.show_contours = bool(on)
        self.draw_contours()

    def reset_rgt_editors(self):
        self._rgt_editors = {}

    def rgt_click(self, x, t):
        """RGT edit mode: 1st click grabs the iso-RGT level at the cursor, 2nd click moves it there."""
        ij = self.trace_at(x)
        if ij is None:
            self.cursorInfo.emit("RGT edits work on inlines, crosslines and 2D lines in the Time domain.")
            return None
        ed = self.rgt_editor()
        if ed is None:
            return None
        key = self.section_key()
        if self._rgt_grab is None or self._rgt_grab[0] != key:
            v = ed.value_at(ij[0], ij[1], t) if key.startswith("3D:") else ed.value_at(ij[1], t)
            self._rgt_grab = (key, ij, float(t), v)
            self.cursorInfo.emit(f"Grabbed RGT level {v:.1f} at {t:.0f} ms - click where it should be")
            return None
        _, (i0, j0), t0, v = self._rgt_grab
        self._rgt_grab = None
        fl = self.fset().visible_faults() if self.fset() is not None else None
        if key.startswith("3D:"):
            ed.move_level(i0, j0, t0, t, faults=fl or None, i_to=ij[0], j_to=ij[1])
        else:
            ed.move_level(j0, t0, t, faults=fl or None, j_to=ij[1])
        self.cursorInfo.emit(f"RGT level {v:.1f} moved to {t:.0f} ms (local re-solve) - Ctrl+Z undoes")
        self.redraw()
        return float(t)

    def rgt_contours(self):
        """[(x values, times)] of the horizon-stack levels on the current time-domain section."""
        from ..rgt.stack import load_stack_2d
        if not self.source or self.wheeler_on() or self.depth_on() or self.mode in ("Time slice", "Stratal slice"):
            return []
        kind, name = self.source
        try:
            if kind == "3D":
                st = self._stack3d()
                if st is None:
                    return []
                v = self.project.volume(name)
                sec = st.section(v, self.mode, self.number)  # (N, n)
                x = v.xlines if self.mode == "Inline" else v.inlines
            else:
                st = load_stack_2d(self._rgt_folder(kind, name)) if self.project.path else None
                if st is None or name not in st["lines"]:
                    return []
                sec = st["lines"][name]
                x = self.project.line(name).cdp
        except Exception:
            return []
        step = max(1, len(sec) // 60)
        return [(np.asarray(x, float), np.asarray(sec[k], float)) for k in range(0, len(sec), step)]

    def draw_contours(self):
        for it in self._ctr_items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._ctr_items = []
        if not self.show_contours or self._data is None:
            return
        pen = pg.mkPen((90, 90, 90, 140), width=1)
        for x, t in self.rgt_contours():
            it = pg.PlotCurveItem(x, t, connect="finite", pen=pen)
            self.plot.addItem(it)
            self._ctr_items.append(it)

    # ------------------------------------------------------------ faults
    def fset(self):
        return getattr(self.project, "faults", None) if self.project else None

    def section_point(self, xval, t):
        """Section coordinates -> map (x, y, t), or None on a time slice."""
        kind, name = self.source
        if kind == "2D":
            if self.wheeler_on():
                return None
            l = self.project.line(name)
            return float(np.interp(xval, l.cdp, l.x)), float(np.interp(xval, l.cdp, l.y)), float(t)
        v = self.project.volume(name)
        if self.wheeler_on():
            return None
        if self.mode == "Inline":
            X, Y = v.xy(self.number, xval)
        elif self.mode == "Crossline":
            X, Y = v.xy(xval, self.number)
        else:
            return None
        return float(X), float(Y), float(t)

    def _section_id(self):
        return (tuple(self.source), self.mode, self.number)

    def fault_click(self, x, t, erase=False):
        """Fault mode: click adds a point to the current stick (Enter ends it); Shift+click deletes the
        nearest stick of the active fault on this section."""
        fs = self.fset()
        if fs is None or not self.source:
            return
        if erase:
            return self.delete_stick_near(x, t)
        f = fs.get(self.active_fault) if self.active_fault else None
        if f is None:
            f = fs.new()
            self.active_fault = f.name
            if self.faultsEdited:
                self.faultsEdited()
        p = self.section_point(x, t)
        if p is None:
            self.cursorInfo.emit("Draw fault sticks on an inline, crossline or 2D line.")
            return
        if self.stick and self.stick_key != self._section_id():
            self.end_stick()
        self.stick_key = self._section_id()
        self.stick.append(p)
        self.cursorInfo.emit(f"{f.name}: stick point {len(self.stick)} - Enter ends the stick, Esc cancels")
        self.draw_faults()

    def end_stick(self):
        fs = self.fset()
        pts, self.stick = self.stick, []
        if fs is None or len(pts) < 2:
            if pts:
                self.cursorInfo.emit("A stick needs at least 2 points - discarded")
            self.draw_faults()
            return None
        f = fs.get(self.active_fault)
        if f is None:
            return None
        fs.snapshot(f)
        k = f.add_stick(pts)
        self.cursorInfo.emit(f"{f.name}: stick {k + 1} saved ({len(pts)} points)")
        self._faults_edited()
        return k

    def cancel_stick(self):
        self.stick = []
        self.draw_faults()

    def _sticks_here(self, f):
        """[(stick index, xs, ts)] of the fault's sticks lying on the current section."""
        kind, name = self.source
        out = []
        if kind == "2D":
            l = self.project.line(name)
            for k, s in enumerate(f.sticks):
                r = flt.sticks_on_line(_One(f, s), l.x, l.y)
                if r:
                    out.append((k, np.interp(r[0][0], np.arange(len(l.cdp)), l.cdp), r[0][1]))
            return out
        if self.mode not in ("Inline", "Crossline"):
            return out
        v = self.project.volume(name)
        for k, s in enumerate(f.sticks):
            r = flt.sticks_on_section_3d(_One(f, s), v, self.mode, self.number)
            if r:
                out.append((k, r[0][0], r[0][1]))
        return out

    def delete_stick_near(self, x, t, tol=0.03):
        fs = self.fset()
        f = fs.get(self.active_fault) if fs is not None and self.active_fault else None
        if f is None or self._xaxis is None:
            return None
        xr = max(1e-9, float(np.ptp(self._xaxis)))
        tr = max(1e-9, float(np.ptp(self._yaxis)))
        best = None
        for k, xs, ts in self._sticks_here(f):
            d = np.min(np.hypot((np.asarray(xs) - x) / xr, (np.asarray(ts) - t) / tr))
            if d <= tol and (best is None or d < best[0]):
                best = (d, k)
        if best is None:
            self.cursorInfo.emit("No stick of the active fault near the cursor")
            return None
        fs.snapshot(f)
        del f.sticks[best[1]]
        f.invalidate()
        self.cursorInfo.emit(f"{f.name}: stick deleted")
        self._faults_edited()
        return best[1]

    def _faults_edited(self):
        self.draw_faults()
        if self.faultsEdited:
            self.faultsEdited()

    def fault_lines(self):
        """Pure-data fault overlay for the current section: list of (fault name, kind, xs, ys) where kind is
        'cut' (pairs of points, surface intersection), 'stick' (polyline) or 'new' (stick being drawn)."""
        fs = self.fset()
        out = []
        if fs is None or not self.source or self._data is None or self.wheeler_on() or self.depth_on() or \
                self.mode == "Stratal slice":
            return out
        kind, name = self.source
        for f in fs.items:
            if not f.visible:
                continue
            if kind == "2D":
                l = self.project.line(name)
                seg = flt.section_segments_line(f, l.x, l.y)
                if seg is not None and len(seg[0]):
                    k = np.arange(len(l.cdp))
                    x0, x1 = np.interp(seg[0], k, l.cdp), np.interp(seg[2], k, l.cdp)
                    out.append((f.name, "cut", np.c_[x0, x1].ravel(), np.c_[seg[1], seg[3]].ravel()))
            else:
                seg = flt.section_segments_3d(f, self.project.volume(name), self.mode, self.number)
                if seg is not None and len(seg[0]):
                    out.append((f.name, "cut", np.c_[seg[0], seg[2]].ravel(), np.c_[seg[1], seg[3]].ravel()))
            for _, xs, ts in self._sticks_here(f):
                out.append((f.name, "stick", np.asarray(xs, float), np.asarray(ts, float)))
        if self.stick and self.stick_key == self._section_id():
            xs = []
            for X, Y, t in self.stick:
                if kind == "2D":
                    l = self.project.line(name)
                    xs.append(float(l.cdp[int(np.argmin(np.hypot(l.x - X, l.y - Y)))]))
                else:
                    v = self.project.volume(name)
                    il, xl = v.ilxl_at(X, Y)
                    xs.append(float(xl if self.mode == "Inline" else il))
            out.append((self.active_fault, "new", np.array(xs), np.array([p[2] for p in self.stick])))
        return out

    def draw_faults(self):
        if self.glow is not None:
            self.glow.clear("fault")
        for it in self._flt_items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._flt_items = []
        fs = self.fset()
        cols = {f.name: f.color for f in fs.items} if fs is not None else {}
        for fname, kind, xs, ys in self.fault_lines():
            c = cols.get(fname, (220, 30, 30))
            if kind == "cut":
                it = pg.PlotCurveItem(xs, ys, connect="pairs", pen=pg.mkPen(c, width=2))
            elif kind == "stick":
                it = pg.PlotDataItem(xs, ys, pen=pg.mkPen(c, width=3), symbol="o", symbolSize=6,
                                     symbolBrush=pg.mkBrush(*c), symbolPen=None)
            else:
                it = pg.PlotDataItem(xs, ys, pen=pg.mkPen(c, width=2, style=QtCore.Qt.DashLine), symbol="s",
                                     symbolSize=7, symbolBrush=pg.mkBrush(255, 255, 255), symbolPen=pg.mkPen(c))
            _click_through(it)
            self.plot.addItem(it)
            self._flt_items.append(it)
            if self.glow is not None:  # PLAN 39: faults are a must
                self.glow.attach(it, "fault", fname, c, 3 if kind == "stick" else 2,
                                 connect="pairs" if kind == "cut" else "finite")

    def draw_horizons(self):
        if self.glow is not None:
            self.glow.clear("horizon")
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
        if kind == "3D" and self.mode == "Stratal slice":
            return
        wh = (self.wheeler_on() or self.depth_on()) and self._ymap is not None
        for h in hs.items:
            if not h.visible:
                continue
            width = 3 if h.name == self.active_horizon else 2
            if kind == "3D" and self.mode == "Time slice":
                pts = hz.slice_points(h, key, surv, self.number)
                if pts is None or not len(pts[0]):
                    continue
                it = _click_through(pg.ScatterPlotItem(pts[0], pts[1], size=4, pen=None, brush=pg.mkBrush(*h.color)))
            else:
                c = hz.section_curve(h, key, surv, self.mode, self.number)
                if c is None or not np.isfinite(c[1]).any():
                    continue
                if wh:  # horizon time -> RGT (Wheeler) or depth on each trace
                    tau, times = self._ymap
                    tk = (np.asarray(c[1], float) - times[0]) / (times[1] - times[0])
                    ok = np.isfinite(tk)
                    tk = np.clip(np.where(ok, tk, 0), 0, len(times) - 1.000001)
                    k0 = np.floor(tk).astype(int)
                    w = tk - k0
                    r = np.arange(len(tk))
                    c = (c[0], np.where(ok, tau[r, k0] * (1 - w) + tau[r, k0 + 1] * w, np.nan))
                it = pg.PlotCurveItem(np.asarray(c[0], float), np.asarray(c[1], float), connect="finite",
                                      pen=pg.mkPen(h.color, width=width))
            self.plot.addItem(it)
            self._hz_items.append(it)
            if self.glow is not None:  # PLAN 39: horizons glow; the active horizon stronger
                if kind == "3D" and self.mode == "Time slice":
                    self.glow.attach(it, "horizon", h.name, h.color, symbol_size=4)
                else:
                    self.glow.attach(it, "horizon", h.name, h.color, width)
        for k, i, j in self.seeds:
            if k != key or (kind == "3D" and self.mode == "Time slice") or wh:
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
            it = _click_through(pg.ScatterPlotItem([xv], [t], symbol="x", size=10, pen=pg.mkPen((0, 0, 0), width=2)))
            self.plot.addItem(it)
            self._hz_items.append(it)

    # ------------------------------------------------------------ for the basemap
    def section_horizons(self):
        """{name: t (n traces)} of the visible horizons on the current inline / crossline / 2D line, shallow first
        (PLAN 26 paleo animation / restoration)."""
        hs = self.hset()
        if hs is None or not self.source or self._data is None or self.mode in ("Time slice", "Stratal slice"):
            return {}
        kind, name = self.source
        surv = self.project.volume(name) if kind == "3D" else self.project.line(name)
        key = self.section_key()
        out = {}
        for h in hs.items:
            if not h.visible:
                continue
            c = hz.section_curve(h, key, surv, self.mode, self.number)
            if c is None or not np.isfinite(c[1]).any():
                continue
            t = np.interp(np.asarray(self._xaxis, float), np.asarray(c[0], float), np.asarray(c[1], float),
                          left=np.nan, right=np.nan)
            out[h.name] = t
        return dict(sorted(out.items(), key=lambda kv: float(np.nanmean(kv[1]))))

    # ------------------------------------------------------------ linked cursor (PLAN 23c)
    def linked_position(self, msg, tol=None):
        """Where a cursor message from another view falls on this section: (x axis value, y value or None) in the
        displayed axes, or None when the point is off this line / slice."""
        if not self.source or msg.get("clear") or msg.get("x") is None or self.project is None:
            return None
        kind, name = self.source
        x, y = float(msg["x"]), float(msg["y"])
        t = msg.get("t")
        if kind == "2D":
            l = self.project.line(name)
            d = np.hypot(np.asarray(l.x, float) - x, np.asarray(l.y, float) - y)
            j = int(np.argmin(d))
            step = float(np.median(np.hypot(np.diff(l.x), np.diff(l.y)))) if len(l.x) > 1 else 25.0
            if d[j] > (tol if tol is not None else 3 * step):
                return None
            return float(l.cdp[j]), t
        v = self.project.volume(name)
        il, xl = v.ilxl_at(x, y)
        if not (v.inlines[0] - v.il_step / 2 <= il <= v.inlines[-1] + v.il_step / 2 and
                v.xlines[0] - v.xl_step / 2 <= xl <= v.xlines[-1] + v.xl_step / 2):
            return None
        if self.mode == "Inline":
            return (float(xl), t) if abs(il - self.number) <= (tol or abs(v.il_step) * 1.5) else None
        if self.mode == "Crossline":
            return (float(il), t) if abs(xl - self.number) <= (tol or abs(v.xl_step) * 1.5) else None
        return float(xl), float(il)  # time / stratal slice: a point

    def show_linked_cursor(self, msg):
        pos = self.linked_position(msg)
        items = getattr(self, "_link_items", None) or []
        for it in items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._link_items = []
        self._link_pos = pos
        if pos is None:
            return None
        pen = pg.mkPen((220, 0, 0), width=1, style=QtCore.Qt.DashLine)
        if self.mode in ("Time slice", "Stratal slice") and self.source[0] == "3D":
            it = pg.ScatterPlotItem([pos[0]], [pos[1]], symbol="+", size=16, pen=pg.mkPen((220, 0, 0), width=2))
            self._link_items = [it]
        else:
            self._link_items = [pg.InfiniteLine(pos=pos[0], angle=90, pen=pen)]
            if pos[1] is not None and np.isfinite(pos[1]):
                self._link_items.append(pg.InfiniteLine(pos=pos[1], angle=0, pen=pen))
        for it in self._link_items:
            self.plot.addItem(it)
        return pos

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


class _One:
    """A single stick wrapped as a fault (for the sticks_on_* helpers)."""

    def __init__(self, f, stick):
        self.sticks = [stick]
        self.name = f.name


def _positive_volume(panel):
    """True for a 3D inversion result with positive values (LFM, AI, Zp ... not relative AI)."""
    src = getattr(panel, "source", None)
    if not src or src[0] != "3D" or getattr(panel, "project", None) is None:
        return False
    try:
        inv = panel.project.volume(src[1]).meta.get("inversion") or {}
    except Exception:
        return False
    return inv.get("kind") in ("lfm", "model-based", "simultaneous") and "LambdaRho" not in src[1]
