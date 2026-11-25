"""Small sun dial (PLAN 40a): drag the sun around the circle - angle = azimuth (north up), distance from the rim =
elevation (centre = overhead).  Emits sunChanged(Sun); maths in aesthetics.sun_dial_xy / sun_from_dial."""
from PySide6 import QtCore, QtGui, QtWidgets

from .. import aesthetics as AE


class SunDial(QtWidgets.QWidget):
    sunChanged = QtCore.Signal(object)

    def __init__(self, sun=None, parent=None):
        super().__init__(parent)
        self.sun = sun or AE.Sun()
        self.setMinimumSize(110, 110)
        self.setToolTip("Drag the sun: direction = azimuth, nearer the centre = higher elevation")

    def _geom(self):
        w, h = max(1, self.width()), max(1, self.height())
        r = 0.42 * min(w, h)
        return w / 2.0, h / 2.0, r

    def set_from_pixel(self, px, py):
        cx, cy, r = self._geom()
        self.sun = AE.sun_from_dial((px - cx) / r, (cy - py) / r, 1.0, self.sun)
        self.update()
        self.sunChanged.emit(self.sun)
        return self.sun

    def mousePressEvent(self, ev):
        p = ev.position()
        self.set_from_pixel(p.x(), p.y())

    def mouseMoveEvent(self, ev):
        p = ev.position()
        self.set_from_pixel(p.x(), p.y())

    def paintEvent(self, ev):
        cx, cy, r = self._geom()
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.setPen(QtGui.QPen(QtGui.QColor(150, 150, 150), 1))
            p.drawEllipse(QtCore.QPointF(cx, cy), r, r)
            p.drawEllipse(QtCore.QPointF(cx, cy), r / 2, r / 2)
            p.drawText(QtCore.QPointF(cx - 4, cy - r - 2), "N")
            x, y = AE.sun_dial_xy(self.sun, r)
            p.setBrush(QtGui.QColor(255, 200, 0))
            p.setPen(QtGui.QPen(QtGui.QColor(120, 80, 0), 1))
            p.drawEllipse(QtCore.QPointF(cx + x, cy - y), 6, 6)
            p.drawText(QtCore.QPointF(4, self.height() - 4),
                       f"az {self.sun.azimuth:.0f}  el {self.sun.elevation:.0f}")
        finally:
            p.end()


class SunDialog(QtWidgets.QDialog):
    """Live Map / map views: sun dial + vertical exaggeration + blend strength + multi-directional + on/off."""

    def __init__(self, sun=None, parent=None, on_change=None):
        super().__init__(parent)
        self.setWindowTitle("Sun (hillshade)")
        self.on_change = on_change
        f = QtWidgets.QFormLayout(self)
        self.dial = SunDial(sun, self)
        self.on = QtWidgets.QCheckBox("Hillshade on")
        self.on.setChecked(self.dial.sun.on)
        self.ve = QtWidgets.QDoubleSpinBox()
        self.ve.setRange(0.1, 100.0)
        self.ve.setValue(self.dial.sun.ve)
        self.strength = QtWidgets.QSpinBox()
        self.strength.setRange(0, 100)
        self.strength.setSuffix(" %")
        self.strength.setValue(int(round(100 * self.dial.sun.strength)))
        self.multi = QtWidgets.QCheckBox("Multi-directional (softer)")
        self.multi.setChecked(self.dial.sun.multi)
        for lab, w in (("", self.dial), ("", self.on), ("Vertical exaggeration", self.ve),
                       ("Blend with colour", self.strength), ("", self.multi)):
            f.addRow(lab, w)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(lambda *_: self.accept())
        bb.rejected.connect(lambda *_: self.reject())
        f.addRow(bb)
        self.dial.sunChanged.connect(lambda *_: self.changed())

    def value(self):
        s = AE.Sun(**self.dial.sun.to_dict())
        s.on = bool(self.on.isChecked())
        s.ve = float(self.ve.value()) or 1.0
        s.strength = float(self.strength.value()) / 100.0
        s.multi = bool(self.multi.isChecked())
        return s

    def changed(self):
        if self.on_change:
            self.on_change(self.value())
