"""Glow / Halo underlays for pyqtgraph views (PLAN 39a-c, e, h).  Logic (stroke specs, settings, bus) is in
`seismic_tool/glow.py`; this is the thin Qt layer.

    mgr = GlowManager(plot_widget, settings, state, lut_fn=lambda: current_lut)
    mgr.attach(curve_item, "horizon", "Top", colour=(255, 220, 0), width=2)   # underlays created + kept in sync
    mgr.detach(curve_item)        # the parent is removed -> underlays removed
    mgr.refresh()                 # settings / selection changed -> restyle every underlay
    mgr.set_dragging(True)        # decimated underlays while panning / dragging; False = full resolution

Underlays are separate items one z-step below the parent, never interactive (no mouse buttons, no hover, clicks
ignored, flagged `_glow_underlay`), with cosmetic pens (width = screen pixels) and round caps / joins.  They follow
the parent's data (setData is wrapped), visibility (visibleChanged) and pen colour.
"""
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .. import glow as G


def _no_mouse(item):
    """Underlays never take part in picking or hit tests (PLAN 39a)."""
    item._glow_underlay = True
    for it in (item, getattr(item, "scatter", None), getattr(item, "curve", None)):
        if it is None:
            continue
        try:
            it.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        except Exception:
            pass
        try:
            it.setAcceptHoverEvents(False)
        except Exception:
            pass
        try:
            it.mouseClickEvent = lambda ev: ev.ignore()
            it.hoverEvent = lambda ev: None
        except Exception:
            pass
    return item


def glow_pen(width_px, rgba):
    pen = pg.mkPen(QtGui.QColor(*[int(v) for v in rgba]), width=float(width_px))
    try:
        pen.setCosmetic(True)  # width in screen pixels, whatever the zoom
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
    except Exception:
        pass
    return pen


def _xy(item):
    """(x, y) arrays of a PlotCurveItem / PlotDataItem / ScatterPlotItem, or (None, None)."""
    try:
        d = item.getData()
        if d is None or d[0] is None:
            return None, None
        x, y = np.asarray(d[0], float), np.asarray(d[1], float)
        if x.ndim != 1 or y.shape != x.shape:
            return None, None
        return x, y
    except Exception:
        return None, None


class Entry:
    def __init__(self, parent, kind, name, colour, width, symbol_size=None, connect="finite", emphasis=None):
        self.parent, self.kind, self.name = parent, kind, str(name)
        self.emphasis = emphasis  # fixed emphasis (e.g. the Live Map selection ring = 'selected', QC = 'qc')
        self.colour, self.width = tuple(colour), float(width)
        self.symbol_size = symbol_size
        self.connect = connect
        self.under = []  # underlay items, outermost first
        self.mode = None


class GlowManager:
    def __init__(self, plot, settings=None, state=None, lut_fn=None, background_fn=None, status=None):
        self.plot = plot
        self.state = state or G.GlowState(None, settings or G.GlowSettings())
        self.settings = self.state.settings
        self.lut_fn = lut_fn
        self.background_fn = background_fn
        self.status = status  # callable(str) for the auto-limit note
        self.entries = {}  # id(parent) -> Entry
        self.dragging = False
        self.max_drag_points = 3000
        self.created = 0  # underlays created (tests)
        self.state.listeners.append(self.refresh)

    def set_state(self, state):
        """Use a shared GlowState (map bus emphasis + settings of the main window / project)."""
        try:
            self.state.listeners.remove(self.refresh)
        except ValueError:
            pass
        self.state, self.settings = state, state.settings
        state.listeners.append(self.refresh)
        self.refresh()
        return state

    # ------------------------------------------------------------ attach / detach
    def attach(self, item, kind, name, colour=(255, 255, 0), width=2.0, symbol_size=None, connect="finite",
               emphasis=None):
        if item is None:
            return None
        key = id(item)
        if key in self.entries:
            self.detach(item)
        e = Entry(item, kind, name, colour, width, symbol_size, connect, emphasis)
        self.entries[key] = e
        self._hook(e)
        self._build(e)
        return e

    def detach(self, item):
        e = self.entries.pop(id(item), None)
        if e is None:
            return False
        self._drop(e)
        return True

    def clear(self, kind=None):
        for k in [k for k, e in self.entries.items() if kind is None or e.kind == kind]:
            self._drop(self.entries.pop(k))

    def underlays(self, item=None):
        if item is not None:
            e = self.entries.get(id(item))
            return list(e.under) if e else []
        return [u for e in self.entries.values() for u in e.under]

    def _drop(self, e):
        for u in e.under:
            try:
                self.plot.removeItem(u)
            except Exception:
                pass
        e.under = []

    def _hook(self, e):
        it = e.parent
        mgr = self
        orig = getattr(it, "setData", None)
        if callable(orig) and not getattr(orig, "_glow_wrapped", False):
            def setData(*a, **k):
                r = orig(*a, **k)
                mgr.sync(it)
                return r
            setData._glow_wrapped = True
            try:
                it.setData = setData
            except Exception:
                pass
        sig = getattr(it, "visibleChanged", None)
        if sig is not None:
            try:
                sig.connect(lambda *_: mgr.sync_visibility(it))
            except Exception:
                pass

    # ------------------------------------------------------------ build / sync
    def _limited(self):
        n = len(self.entries)
        lim = G.auto_limit(n, self.settings)
        if self.status is not None:
            note = G.limit_note(n, self.settings)
            if note:
                try:
                    self.status(note)
                except Exception:
                    pass
        return lim

    def _strokes(self, e, limited=None):
        emph = e.emphasis or self.state.emphasis(e.kind, e.name)
        lim = self._limited() if limited is None else limited
        mode = self.settings.effective(e.kind, e.name, emph, limited=lim)
        lut = self.lut_fn() if self.lut_fn else None
        bg = self.background_fn() if self.background_fn else None
        ov = self.settings.override(e.kind, e.name)
        e.mode = mode
        if e.symbol_size is not None:
            return mode, G.marker_halo(self.settings, mode, e.symbol_size, e.colour, emph, lut, ov)
        return mode, G.strokes(self.settings, mode, e.colour, e.width, emph, lut, bg, ov)

    def _visible(self, it):
        try:
            return bool(it.isVisible())
        except Exception:
            return True

    def _build(self, e, limited=None):
        self._drop(e)
        mode, sts = self._strokes(e, limited)
        if not sts or not self._visible(e.parent):
            return []
        x, y = _xy(e.parent)
        if x is None or not len(x):
            return []
        if self.dragging:
            x, y = G.decimate_xy(x, y, self.max_drag_points)
        try:
            z0 = float(e.parent.zValue())
        except Exception:
            z0 = 0.0
        n = len(sts)
        for k, (w, c) in enumerate(sts):
            if e.symbol_size is not None:
                u = pg.ScatterPlotItem(x, y, size=float(w), pen=None, brush=pg.mkBrush(*[int(v) for v in c]))
            else:
                u = pg.PlotCurveItem(x, y, connect=e.connect, pen=glow_pen(w, c))
            _no_mouse(u)
            try:
                u.setZValue(z0 - 1.0 + 0.1 * k / max(1, n))  # all below the parent, outermost lowest
            except Exception:
                pass
            self.plot.addItem(u)
            e.under.append(u)
            self.created += 1
        return e.under

    def sync(self, item):
        e = self.entries.get(id(item))
        return self._build(e) if e is not None else []

    def sync_visibility(self, item):
        e = self.entries.get(id(item))
        if e is None:
            return
        if not e.under and self._visible(item):
            self._build(e)
            return
        v = self._visible(item)
        for u in e.under:
            try:
                u.setVisible(v)
            except Exception:
                pass

    def set_colour(self, item, colour):
        e = self.entries.get(id(item))
        if e is not None:
            e.colour = tuple(colour)
            self._build(e)

    def refresh(self):
        lim = self._limited()
        for e in list(self.entries.values()):
            self._build(e, lim)

    def set_dragging(self, on):
        """Decimated underlays while dragging; full resolution on release (PLAN 39h)."""
        on = bool(on)
        if on != self.dragging:
            self.dragging = on
            self.refresh()


# ====================================================================== settings UI (PLAN 39c)
def add_glow_menu(menu, settings, kind, name, on_change=None):
    """Right-click > Glow > On / Off / Use global + Colour... for one object (object trees, docks)."""
    sub = menu.addMenu("Glow")
    cur = settings.override(kind, name)
    acts = {}
    for label, mode in (("On", "on"), ("Off", "off"), ("Use global", "global")):
        a = sub.addAction(label)
        try:
            a.setCheckable(True)
            a.setChecked(cur["mode"] == mode)
        except Exception:
            pass
        a.triggered.connect(lambda *_, m=mode: (settings.set_override(kind, name, m, settings.override(kind, name)
                                                                      .get("colour")),
                                                on_change and on_change()))
        acts[mode] = a

    def pick():
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(*(cur.get("colour") or (255, 255, 160))))
        if c.isValid():
            settings.set_override(kind, name, cur["mode"] if cur["mode"] != "global" else "on",
                                  (c.red(), c.green(), c.blue()))
            on_change and on_change()
    a = sub.addAction("Colour...")
    a.triggered.connect(lambda *_: pick())
    acts["colour"] = a
    return acts


class GlowSettingsDialog(QtWidgets.QDialog):
    """Display > Glow settings: global on/off, mode, width, intensity, falloff layers, colour, active-only, hover."""

    def __init__(self, settings, parent=None, on_apply=None):
        super().__init__(parent)
        self.setWindowTitle("Glow settings")
        self.settings = settings
        self.on_apply = on_apply
        f = QtWidgets.QFormLayout(self)
        self.enabled = QtWidgets.QCheckBox("Glow on (G toggles)")
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Glow", "Halo", "Off"])
        self.width = QtWidgets.QDoubleSpinBox()
        self.width.setRange(0.5, 40.0)
        self.width.setSuffix(" px")
        self.intensity = QtWidgets.QSpinBox()
        self.intensity.setRange(0, 100)
        self.intensity.setSuffix(" %")
        self.layers = QtWidgets.QSpinBox()
        self.layers.setRange(1, 4)
        self.colour = QtWidgets.QComboBox()
        self.colour.addItems(["Match object", "White", "Black", "Custom..."])
        self.halo = QtWidgets.QComboBox()
        self.halo.addItems(["Auto (against the colour table)", "White", "Black", "Custom..."])
        self.halo_px = QtWidgets.QDoubleSpinBox()
        self.halo_px.setRange(0.5, 12.0)
        self.halo_px.setSuffix(" px")
        self.active_only = QtWidgets.QCheckBox("Glow only the active / selected object")
        self.hover = QtWidgets.QCheckBox("Hover highlight")
        self.export = QtWidgets.QCheckBox("Exports include glow (untick = export without glow)")
        for lab, w in (("", self.enabled), ("Mode", self.mode), ("Width", self.width), ("Intensity", self.intensity),
                       ("Falloff layers", self.layers), ("Glow colour", self.colour), ("Halo colour", self.halo),
                       ("Halo width", self.halo_px), ("", self.active_only), ("", self.hover), ("", self.export)):
            f.addRow(lab, w)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Apply
                                        | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(lambda *_: (self.apply(), self.accept()))
        bb.rejected.connect(lambda *_: self.reject())
        try:
            bb.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(lambda *_: self.apply())
        except Exception:
            pass
        f.addRow(bb)
        self.load()

    def load(self):
        s = self.settings
        self.enabled.setChecked(s["enabled"])
        self.mode.setCurrentIndex(G.MODES.index(s["mode"]))
        self.width.setValue(s["width_px"])
        self.intensity.setValue(int(round(100 * s["intensity"])))
        self.layers.setValue(s["layers"])
        self.colour.setCurrentIndex(G.COLOUR_MODES.index(s["colour"]))
        self.halo.setCurrentIndex(("auto", "white", "black", "custom").index(s["halo_colour"]))
        self.halo_px.setValue(s["halo_px"])
        self.active_only.setChecked(s["active_only"])
        self.hover.setChecked(s["hover"])
        self.export.setChecked(s["export_glow"])

    def values(self):
        """Dialog -> settings dict (custom colours picked with a colour dialog)."""
        ci = int(self.colour.currentIndex())
        hi = int(self.halo.currentIndex())
        out = dict(enabled=bool(self.enabled.isChecked()), mode=G.MODES[int(self.mode.currentIndex())],
                   width_px=float(self.width.value()), intensity=float(self.intensity.value()) / 100.0,
                   layers=int(self.layers.value()), colour=G.COLOUR_MODES[ci],
                   halo_colour=("auto", "white", "black", "custom")[hi], halo_px=float(self.halo_px.value()),
                   active_only=bool(self.active_only.isChecked()), hover=bool(self.hover.isChecked()),
                   export_glow=bool(self.export.isChecked()))
        return out

    def apply(self):
        v = self.values()
        for key, which in (("colour", "custom"), ("halo_colour", "halo_custom")):
            if v[key] == "custom" and self.settings[key] != "custom":
                c = QtWidgets.QColorDialog.getColor(QtGui.QColor(*self.settings[which]), self)
                if c.isValid():
                    v[which] = (c.red(), c.green(), c.blue())
        self.settings.update(**v)
        if self.on_apply:
            self.on_apply()
        return v


def install_list_menu(listwidget, settings_fn, kind, on_change=None, project_fn=None, colour_fn=None):
    """Right-click on an object list (Horizons / Faults dock) > Glow > On / Off / Use global / Colour...
    The item's UserRole data is the object name."""
    def show(pos):
        it = listwidget.itemAt(pos)
        name = it.data(QtCore.Qt.UserRole) if it is not None else None
        if not isinstance(name, str):
            return None
        m = QtWidgets.QMenu(listwidget)
        add_glow_menu(m, settings_fn(), kind, name, on_change)
        if project_fn is not None:  # PLAN 40j: lock / unlock the colour against automatic reassignment
            from .. import aesthetics as AE
            ca = AE.ColourAssigner(project_fn())
            locked = ca.is_locked(kind, name)
            a = m.addAction("Unlock colour" if locked else "Lock colour")
            a.triggered.connect(lambda *_: ca.lock(kind, name, colour_fn(name) if colour_fn else None, not locked))
        m.exec(listwidget.viewport().mapToGlobal(pos))
        return m
    listwidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
    listwidget.customContextMenuRequested.connect(show)
    return show
