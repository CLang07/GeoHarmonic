"""Horizons dock: list of horizons, pick mode, snap + tracker settings, track / undo buttons."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from .. import horizons as hz  # noqa: E402

HELP = ("Pick mode (P): click = seed pick snapped to the event, drag = manual line, Shift+click/drag = erase.\n"
        "T = autotrack from the seeds (no seeds: grow from all picks). Ctrl+Z = undo.")


class HorizonDock(QtWidgets.QWidget):
    changed = QtCore.Signal()  # horizon list / visibility / colour changed -> redraw section + basemap

    def __init__(self, section, parent=None):
        super().__init__(parent)
        self.section = section
        self.project = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.list = QtWidgets.QListWidget()
        self.list.setMinimumHeight(90)
        lay.addWidget(self.list, 1)
        row = QtWidgets.QHBoxLayout()
        self.btn_new = QtWidgets.QPushButton("New")
        self.btn_ren = QtWidgets.QPushButton("Rename")
        self.btn_col = QtWidgets.QPushButton("Colour")
        self.btn_del = QtWidgets.QPushButton("Delete")
        for b in (self.btn_new, self.btn_ren, self.btn_col, self.btn_del):
            row.addWidget(b)
        lay.addLayout(row)
        self.pick_btn = QtWidgets.QPushButton("Pick mode (P)")
        self.pick_btn.setCheckable(True)
        lay.addWidget(self.pick_btn)
        form = QtWidgets.QFormLayout()
        self.snap = QtWidgets.QComboBox()
        self.snap.addItems(hz.SNAP_MODES)
        self.snap_ms = QtWidgets.QDoubleSpinBox()
        self.snap_ms.setRange(0, 100)
        self.snap_ms.setValue(12)
        self.snap_ms.setSuffix(" ms")
        self.win = QtWidgets.QDoubleSpinBox()
        self.win.setRange(2, 200)
        self.win.setValue(20)
        self.win.setPrefix("+/- ")
        self.win.setSuffix(" ms")
        self.jump = QtWidgets.QDoubleSpinBox()
        self.jump.setRange(1, 100)
        self.jump.setValue(10)
        self.jump.setSuffix(" ms")
        self.corr = QtWidgets.QDoubleSpinBox()
        self.corr.setRange(0.0, 0.99)
        self.corr.setSingleStep(0.05)
        self.corr.setValue(0.6)
        self.mix2d = QtWidgets.QSpinBox()
        self.mix2d.setRange(1, 21)
        self.mix2d.setSingleStep(2)
        self.mix2d.setValue(5)
        self.mix2d.setSuffix(" traces")
        self.mix2d.setToolTip("Tracker sees a running mean over this many traces (noisy land 2D: 5-9). Picks "
                              "follow the smoothed event.")
        self.mix3d = QtWidgets.QSpinBox()
        self.mix3d.setRange(1, 7)
        self.mix3d.setSingleStep(2)
        self.mix3d.setValue(1)
        self.mix3d.setSuffix(" x n block")
        self.mix3d.setToolTip("3D: mean of an n x n block of traces (3 = 9 traces, slower but more robust)")
        self.gap = QtWidgets.QSpinBox()
        self.gap.setRange(1, 10)
        self.gap.setValue(3)
        self.gap.setSuffix(" traces")
        self.gap.setToolTip("When tracking stalls, try to jump over up to this many bad traces (1 = off)")
        self.scope = QtWidgets.QComboBox()
        self.scope.addItems(["Whole survey", "This line only"])
        self.scope.setToolTip("Whole survey: 3D tracks across inlines and crosslines; 2D follows line ties.")
        form.addRow("Snap to", self.snap)
        form.addRow("Snap search", self.snap_ms)
        form.addRow("Correlation window", self.win)
        form.addRow("Max jump", self.jump)
        form.addRow("Min correlation", self.corr)
        form.addRow("Lateral mix 2D", self.mix2d)
        form.addRow("Lateral mix 3D", self.mix3d)
        form.addRow("Bridge gaps up to", self.gap)
        form.addRow("Track", self.scope)
        lay.addLayout(form)
        row = QtWidgets.QHBoxLayout()
        self.btn_track = QtWidgets.QPushButton("Track (T)")
        self.btn_undo = QtWidgets.QPushButton("Undo (Ctrl+Z)")
        self.btn_clear = QtWidgets.QPushButton("Clear line")
        self.btn_clear.setToolTip("Delete the active horizon's picks on the section shown")
        for b in (self.btn_track, self.btn_undo, self.btn_clear):
            row.addWidget(b)
        lay.addLayout(row)
        self.stop_chk = QtWidgets.QCheckBox("Tracker stops at faults")
        self.stop_chk.setChecked(True)
        self.stop_chk.setToolTip("Links that cross a visible fault surface are not tracked")
        lay.addWidget(self.stop_chk)
        self.map_chk = QtWidgets.QCheckBox("Show time map on basemap")
        self.map_chk.setChecked(True)
        lay.addWidget(self.map_chk)
        tip = QtWidgets.QLabel(HELP)
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#555")
        lay.addWidget(tip)

        self.btn_new.clicked.connect(lambda: self.new_horizon())
        self.btn_ren.clicked.connect(lambda *_: self._rename_dialog())
        self.btn_col.clicked.connect(lambda *_: self._colour_dialog())
        self.btn_del.clicked.connect(lambda *_: self._delete())
        self.pick_btn.toggled.connect(self.set_pick_mode)
        self.btn_track.clicked.connect(lambda *_: self.track())
        self.btn_undo.clicked.connect(lambda *_: self.undo())
        self.btn_clear.clicked.connect(lambda *_: self.clear_line())
        self.list.currentRowChanged.connect(lambda *_: self._active_changed())
        self.list.itemChanged.connect(self._item_changed)
        self.map_chk.toggled.connect(lambda *_: self.changed.emit())
        for w in (self.snap, self.scope):
            w.currentTextChanged.connect(lambda *_: self.apply_settings())
        for w in (self.snap_ms, self.win, self.jump, self.corr, self.mix2d, self.mix3d, self.gap):
            w.valueChanged.connect(lambda *_: self.apply_settings())
        self.stop_chk.toggled.connect(lambda *_: self.apply_settings())
        section.horizonsEdited = self._edited
        section.modeChanged.append(self._mode_changed)

    # ------------------------------------------------------------ state
    def set_project(self, project):
        self.project = project
        self.section.seeds = []
        self.refresh()

    def hset(self):
        return self.project.horizons if self.project else None

    def refresh(self, select=None):
        hs = self.hset()
        cur = select or self.section.active_horizon
        self.list.blockSignals(True)
        self.list.clear()
        for h in hs.items if hs else []:
            it = QtWidgets.QListWidgetItem(f"{h.name}  ({h.count():,} picks)")
            it.setData(QtCore.Qt.UserRole, h.name)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if h.visible else QtCore.Qt.Unchecked)
            pm = QtGui.QPixmap(14, 14)
            pm.fill(QtGui.QColor(*h.color))
            it.setIcon(QtGui.QIcon(pm))
            self.list.addItem(it)
        names = hs.names() if hs else []
        if cur not in names:
            cur = names[0] if names else None
        if cur is not None:
            self.list.setCurrentRow(names.index(cur))
        self.list.blockSignals(False)
        self.section.active_horizon = cur
        self.apply_settings()

    def active(self):
        return self.section.active_horizon

    def apply_settings(self):
        s = self.section
        s.snap_mode = self.snap.currentText() or "Peak"
        s.snap_ms = float(self.snap_ms.value())
        s.track_scope = self.scope.currentText() or "Whole survey"
        p = s.track_params
        p.mode = s.snap_mode
        p.window_ms = float(self.win.value()) or p.window_ms
        p.max_jump_ms = float(self.jump.value()) or p.max_jump_ms
        p.min_corr = float(self.corr.value())
        s.mix_2d = max(1, int(self.mix2d.value()))
        s.mix_3d = max(1, int(self.mix3d.value()))
        p.max_gap = max(1, int(self.gap.value()))
        s.track_faults = bool(self.stop_chk.isChecked())

    def _active_changed(self):
        it = self.list.currentItem()
        self.section.active_horizon = it.data(QtCore.Qt.UserRole) if it else None
        self.changed.emit()

    def _item_changed(self, it):
        h = self.hset().get(it.data(QtCore.Qt.UserRole))
        if h is not None:
            h.visible = it.checkState() == QtCore.Qt.Checked
            self.hset().dirty = True
            self.changed.emit()

    def _edited(self):
        self.refresh()
        self.changed.emit()

    # ------------------------------------------------------------ actions
    def new_horizon(self, name=None):
        if self.hset() is None:
            return None
        if name is None:
            name, ok = QtWidgets.QInputDialog.getText(self, "New horizon", "Name:",
                                                      text=f"Horizon {len(self.hset().items) + 1}")
            if not ok or not str(name).strip():
                return None
        h = self.hset().new(str(name).strip())
        self.section.active_horizon = h.name
        self.refresh(h.name)
        if not self.pick_btn.isChecked():
            self.pick_btn.setChecked(True)
        self.set_pick_mode(True)
        self.changed.emit()
        return h

    def _rename_dialog(self):
        old = self.active()
        if not old:
            return
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename horizon", "Name:", text=old)
        if ok and str(new).strip() and self.hset().rename(old, str(new).strip()):
            self.section.active_horizon = str(new).strip()
            self.refresh(self.section.active_horizon)
            self.changed.emit()

    def _colour_dialog(self):
        h = self.hset().get(self.active()) if self.active() else None
        if h is None:
            return
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(*h.color), self, f"Colour of {h.name}")
        if c.isValid():
            h.color = (c.red(), c.green(), c.blue())
            self.hset().dirty = True
            self.refresh()
            self.changed.emit()

    def _delete(self):
        name = self.active()
        if not name:
            return
        if QtWidgets.QMessageBox.question(self, "Delete horizon", f"Delete {name}? (Ctrl+Z restores it)") \
                != QtWidgets.QMessageBox.Yes:
            return
        self.delete(name)

    def delete(self, name):
        self.hset().delete(name)
        self.section.seeds = []
        self.refresh()
        self.changed.emit()

    def _mode_changed(self, mode):
        on = mode == "horizon"
        if bool(self.pick_btn.isChecked()) != on:
            self.pick_btn.blockSignals(True)
            self.pick_btn.setChecked(on)
            self.pick_btn.blockSignals(False)

    def set_pick_mode(self, on):
        if on:
            self.section.set_mode("horizon")
        elif self.section.pick_mode:
            self.section.set_mode(None)
        self._mode_changed("horizon" if on else None)
        if on and not self.active():
            self.section.cursorInfo.emit("Pick mode: create a horizon first (New).")
        elif on:
            self.section.cursorInfo.emit(f"Picking {self.active()}: click = seed, drag = line, Shift = erase, T = track")

    def toggle_pick_mode(self):
        self.set_pick_mode(not self.section.pick_mode)

    def track(self):
        self.apply_settings()
        app = QtWidgets.QApplication.instance()
        if app:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            return self.section.track()
        finally:
            if app:
                QtWidgets.QApplication.restoreOverrideCursor()

    def undo(self):
        self.section.undo()
        self.refresh()
        self.changed.emit()

    def clear_line(self):
        h = self.hset().get(self.active()) if self.active() else None
        s = self.section
        if h is None or not s.source:
            return
        s.picker().clear_section(h, s.section_key(), s.mode, s.number)
        s._edited()
