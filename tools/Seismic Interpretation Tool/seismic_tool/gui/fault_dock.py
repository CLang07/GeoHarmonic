"""Faults dock: fault list, fault (stick) mode, new / rename / colour / delete."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

HELP = ("Fault mode (F): click points down the fault on an inline, crossline or 2D line, Enter ends the "
        "stick (Esc cancels, Ctrl+Z removes the last point). N = new fault. Shift+click deletes the nearest "
        "stick of the active fault. Sticks of one fault are joined into a surface (shown where it cuts any "
        "section) - draw them on several lines along strike.")


class FaultDock(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, section, parent=None):
        super().__init__(parent)
        self.section = section
        self.project = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.list = QtWidgets.QListWidget()
        self.list.setMinimumHeight(80)
        lay.addWidget(self.list, 1)
        row = QtWidgets.QHBoxLayout()
        self.btn_new = QtWidgets.QPushButton("New (N)")
        self.btn_ren = QtWidgets.QPushButton("Rename")
        self.btn_col = QtWidgets.QPushButton("Colour")
        self.btn_del = QtWidgets.QPushButton("Delete")
        for b in (self.btn_new, self.btn_ren, self.btn_col, self.btn_del):
            row.addWidget(b)
        lay.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.mode_btn = QtWidgets.QPushButton("Fault mode (F)")
        self.mode_btn.setCheckable(True)
        self.end_btn = QtWidgets.QPushButton("End stick (Enter)")
        row.addWidget(self.mode_btn)
        row.addWidget(self.end_btn)
        lay.addLayout(row)
        tip = QtWidgets.QLabel(HELP)
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#555")
        lay.addWidget(tip)
        self.btn_new.clicked.connect(lambda: self.new_fault())
        self.btn_ren.clicked.connect(lambda *_: self._rename_dialog())
        self.btn_col.clicked.connect(lambda *_: self._colour_dialog())
        self.btn_del.clicked.connect(lambda *_: self._delete())
        self.mode_btn.toggled.connect(self.set_fault_mode)
        self.end_btn.clicked.connect(lambda: self.section.end_stick())
        self.list.currentRowChanged.connect(lambda *_: self._active_changed())
        self.list.itemChanged.connect(self._item_changed)
        section.faultsEdited = self._edited
        section.modeChanged.append(self._mode_changed)

    def set_project(self, project):
        self.project = project
        self.section.stick = []
        self.refresh()

    def fset(self):
        return self.project.faults if self.project else None

    def active(self):
        return self.section.active_fault

    def refresh(self, select=None):
        fs = self.fset()
        cur = select or self.section.active_fault
        self.list.blockSignals(True)
        self.list.clear()
        for f in fs.items if fs else []:
            it = QtWidgets.QListWidgetItem(f"{f.name}  ({len(f.sticks)} sticks)")
            it.setData(QtCore.Qt.UserRole, f.name)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if f.visible else QtCore.Qt.Unchecked)
            pm = QtGui.QPixmap(14, 14)
            pm.fill(QtGui.QColor(*f.color))
            it.setIcon(QtGui.QIcon(pm))
            self.list.addItem(it)
        names = fs.names() if fs else []
        if cur not in names:
            cur = names[-1] if names else None
        if cur is not None:
            self.list.setCurrentRow(names.index(cur))
        self.list.blockSignals(False)
        self.section.active_fault = cur

    def _active_changed(self):
        it = self.list.currentItem()
        if self.section.stick:
            self.section.end_stick()
        self.section.active_fault = it.data(QtCore.Qt.UserRole) if it else None
        self.changed.emit()

    def _item_changed(self, it):
        f = self.fset().get(it.data(QtCore.Qt.UserRole))
        if f is not None:
            f.visible = it.checkState() == QtCore.Qt.Checked
            self.fset().dirty = True
            self.changed.emit()

    def _edited(self):
        self.refresh()
        self.changed.emit()

    def new_fault(self, name=None):
        fs = self.fset()
        if fs is None:
            return None
        if self.section.stick:
            self.section.end_stick()
        f = fs.new(name)
        try:  # PLAN 40j
            f.color = self.section.auto_colour("fault", f.name, [x.color for x in fs.items if x is not f])
        except Exception:
            pass
        self.section.active_fault = f.name
        self.refresh(f.name)
        self.set_fault_mode(True)
        self.section.cursorInfo.emit(f"New fault {f.name} - click points down the fault, Enter ends the stick")
        self.changed.emit()
        return f

    def _rename_dialog(self):
        old = self.active()
        if not old:
            return
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename fault", "Name:", text=old)
        if ok and str(new).strip() and self.fset().rename(old, str(new).strip()):
            self.section.active_fault = str(new).strip()
            self.refresh(self.section.active_fault)
            self.changed.emit()

    def _colour_dialog(self):
        f = self.fset().get(self.active()) if self.active() else None
        if f is None:
            return
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(*f.color), self, f"Colour of {f.name}")
        if c.isValid():
            f.color = (c.red(), c.green(), c.blue())
            self.fset().dirty = True
            self.refresh()
            self.changed.emit()

    def _delete(self):
        name = self.active()
        if name and QtWidgets.QMessageBox.question(self, "Delete fault", f"Delete {name}? (Ctrl+Z restores it)") \
                == QtWidgets.QMessageBox.Yes:
            self.delete(name)

    def delete(self, name):
        self.fset().delete(name)
        self.section.active_fault = None
        self.refresh()
        self.changed.emit()

    def _mode_changed(self, mode):
        on = mode == "fault"
        if bool(self.mode_btn.isChecked()) != on:
            self.mode_btn.blockSignals(True)
            self.mode_btn.setChecked(on)
            self.mode_btn.blockSignals(False)

    def set_fault_mode(self, on):
        if on:
            self.section.set_mode("fault")
            self.section.cursorInfo.emit("Fault mode: click points down the fault, Enter ends the stick, N = new fault")
        elif self.section.fault_mode:
            self.section.set_mode(None)
        self._mode_changed("fault" if on else None)

    def toggle_fault_mode(self):
        self.set_fault_mode(not self.section.fault_mode)
