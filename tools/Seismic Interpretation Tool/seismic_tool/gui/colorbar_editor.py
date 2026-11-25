"""Interactive colour-bar editor (PLAN 40b): drag control points along the bar, double-click adds one, right-click
removes one, colour dialog on a point; flip, gamma, clip %, symmetric clip; save to the project, export / import
JSON or CSV.  The model is colortables.ColourTable (tested); this is the thin Qt layer."""
from PySide6 import QtCore, QtGui, QtWidgets

from .. import colortables as CT
from . import colormaps


class ColourBarWidget(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, table, parent=None):
        super().__init__(parent)
        self.table = table
        self.drag = None
        self.setMinimumSize(320, 60)

    # pixel <-> position
    def _x0w(self):
        return 10.0, max(1.0, float(self.width()) - 20.0)

    def pos_at(self, px):
        x0, w = self._x0w()
        return min(1.0, max(0.0, (float(px) - x0) / w))

    def point_near(self, px, tol=6.0):
        x0, w = self._x0w()
        best = None
        for k, (p, _) in enumerate(self.table.points):
            d = abs(x0 + p * w - float(px))
            if d <= tol and (best is None or d < best[0]):
                best = (d, k)
        return None if best is None else best[1]

    # editing (also called by tests with pixel positions)
    def press(self, px, button="left", double=False):
        k = self.point_near(px)
        if double:
            k = self.table.add_point(self.pos_at(px))
        elif button == "right" and k is not None:
            self.table.remove_point(k)
            k = None
        self.drag = k
        self.changed.emit()
        self.update()
        return k

    def move(self, px):
        if self.drag is None:
            return None
        p = self.table.move_point(self.drag, self.pos_at(px))
        self.changed.emit()
        self.update()
        return p

    def release(self):
        self.drag = None

    def mousePressEvent(self, ev):
        b = "right" if ev.button() == QtCore.Qt.RightButton else "left"
        self.press(ev.position().x(), b)

    def mouseDoubleClickEvent(self, ev):
        k = self.point_near(ev.position().x())
        if k is not None:
            c = QtWidgets.QColorDialog.getColor(QtGui.QColor(*self.table.points[k][1]), self)
            if c.isValid():
                self.table.set_colour(k, (c.red(), c.green(), c.blue()))
                self.changed.emit()
                self.update()
            return
        self.press(ev.position().x(), double=True)

    def mouseMoveEvent(self, ev):
        self.move(ev.position().x())

    def mouseReleaseEvent(self, ev):
        self.release()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        try:
            x0, w = self._x0w()
            lut = self.table.lut()
            h = max(10, self.height() - 22)
            for i in range(int(w)):
                c = lut[min(255, int(i / w * 255))]
                p.setPen(QtGui.QColor(int(c[0]), int(c[1]), int(c[2])))
                p.drawLine(int(x0 + i), 4, int(x0 + i), 4 + h)
            p.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20), 1))
            for pos, col in self.table.points:
                x = x0 + pos * w
                p.setBrush(QtGui.QColor(*col))
                p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x, 6 + h), QtCore.QPointF(x - 6, 18 + h),
                                               QtCore.QPointF(x + 6, 18 + h)]))
        finally:
            p.end()


class ColourBarEditor(QtWidgets.QDialog):
    def __init__(self, project=None, name=None, parent=None, on_apply=None):
        super().__init__(parent)
        self.setWindowTitle("Colour bar editor")
        self.project = project
        self.on_apply = on_apply
        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.pick = QtWidgets.QComboBox()
        self.pick.addItems(CT.names(project))
        row.addWidget(QtWidgets.QLabel("Start from"))
        row.addWidget(self.pick, 1)
        lay.addLayout(row)
        self.table = (CT.get(name or CT.names(project)[0], project) or CT.get("Grey")).copy()
        self.bar = ColourBarWidget(self.table, self)
        lay.addWidget(self.bar)
        f = QtWidgets.QFormLayout()
        self.flip = QtWidgets.QCheckBox("Flip")
        self.sym = QtWidgets.QCheckBox("Symmetric clip (about zero)")
        self.gamma = QtWidgets.QDoubleSpinBox()
        self.gamma.setRange(0.2, 5.0)
        self.gamma.setSingleStep(0.1)
        self.clip = QtWidgets.QDoubleSpinBox()
        self.clip.setRange(80.0, 100.0)
        self.clip.setSuffix(" %")
        self.name = QtWidgets.QLineEdit()
        for lab, wd in (("", self.flip), ("", self.sym), ("Gamma", self.gamma), ("Clip", self.clip),
                        ("Save as", self.name)):
            f.addRow(lab, wd)
        lay.addLayout(f)
        bb = QtWidgets.QHBoxLayout()
        for text, fn in (("Save to project", self.save), ("Export...", self.export), ("Import...", self.import_),
                         ("Close", self.accept)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda *_, fn=fn: fn())
            bb.addWidget(b)
        lay.addLayout(bb)
        self.pick.currentTextChanged.connect(lambda n: self.load(n))
        for wd in (self.flip, self.sym):
            wd.toggled.connect(lambda *_: self.apply_widgets())
        for wd in (self.gamma, self.clip):
            wd.valueChanged.connect(lambda *_: self.apply_widgets())
        self.load_widgets()

    def load(self, name):
        t = CT.get(name, self.project)
        if t is not None:
            self.table = t.copy()
            self.bar.table = self.table
            self.load_widgets()
            self.bar.update()
        return self.table

    def load_widgets(self):
        self.flip.setChecked(self.table.flip)
        self.sym.setChecked(self.table.symmetric)
        self.gamma.setValue(self.table.gamma)
        self.clip.setValue(self.table.clip)
        self.name.setText(self.table.name if self.table.custom else f"{self.table.name} (custom)")

    def apply_widgets(self):
        self.table.flip = bool(self.flip.isChecked())
        self.table.symmetric = bool(self.sym.isChecked())
        self.table.gamma = float(self.gamma.value() or 1.0)
        self.table.clip = float(self.clip.value() or 99.0)
        self.bar.update()

    def save(self, name=None):
        self.apply_widgets()
        nm = name or str(self.name.text() or "").strip() or f"{self.table.name} (custom)"
        self.table.name = nm
        CT.save_custom(self.project, self.table)
        colormaps.refresh_names(self.project)
        if self.on_apply:
            self.on_apply(nm)
        return nm

    def export(self, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export colour bar", "", "JSON (*.json);;CSV (*.csv)")
        if not path:
            return None
        return CT.export_csv(self.table, path) if path.lower().endswith(".csv") else CT.export_json(self.table, path)

    def import_(self, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import colour bar", "",
                                                            "Colour bars (*.json *.csv *.txt *.clr *.alut)")
        if not path:
            return None
        self.table = CT.import_file(path)
        self.bar.table = self.table
        self.load_widgets()
        return self.table
