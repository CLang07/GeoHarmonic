"""Attribute parameter 'movie' scan (PLAN 23c): compute an attribute on the current section for a range of one
parameter (spectral frequency, RMS window, coherence window ...) and step through the frames with a slider."""
import os

os.environ.setdefault("QT_API", "pyside6")
from PySide6 import QtCore, QtWidgets  # noqa: E402
import numpy as np  # noqa: E402

PARAMS = {"Spectral 20 Hz": ("freq", 8.0, 80.0, 4.0), "RMS amplitude": ("window_ms", 8.0, 120.0, 8.0),
          "Coherence": ("window_ms", 8.0, 64.0, 8.0)}


class ScanDialog(QtWidgets.QDialog):
    def __init__(self, section, parent=None):
        super().__init__(parent)
        self.section = section
        self.frames = []
        self.setWindowTitle("Attribute parameter scan")
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.attr = QtWidgets.QComboBox()
        self.attr.addItems(list(PARAMS))
        self.lo, self.hi, self.st = (QtWidgets.QDoubleSpinBox() for _ in range(3))
        for w in (self.lo, self.hi, self.st):
            w.setRange(0.1, 1000.0)
            w.setDecimals(1)
        self.attr.currentTextChanged.connect(self._defaults)
        form.addRow("Attribute", self.attr)
        form.addRow("From", self.lo)
        form.addRow("To", self.hi)
        form.addRow("Step", self.st)
        lay.addLayout(form)
        b = QtWidgets.QPushButton("Compute frames")
        b.clicked.connect(lambda *_: self.compute())
        lay.addWidget(b)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.set_frame)
        lay.addWidget(self.slider)
        self.label = QtWidgets.QLabel("")
        lay.addWidget(self.label)
        self._defaults(self.attr.currentText())

    def _defaults(self, name):
        p = PARAMS.get(name)
        if p:
            self.lo.setValue(p[1])
            self.hi.setValue(p[2])
            self.st.setValue(p[3])

    def values(self):
        lo, hi, st = self.lo.value(), self.hi.value(), max(1e-6, self.st.value())
        return list(np.arange(lo, hi + st / 2, st))

    def compute(self, attr=None, values=None):
        attr = attr or self.attr.currentText()
        param = PARAMS[attr][0]
        try:
            self.frames = self.section.scan_frames(attr, param, values if values is not None else self.values())
        except Exception as e:
            self.label.setText(str(e))
            self.frames = []
            return 0
        self.param = param
        self.slider.setRange(0, max(0, len(self.frames) - 1))
        self.slider.setValue(0)
        self.set_frame(0)
        return len(self.frames)

    def set_frame(self, k):
        if not self.frames:
            return None
        k = int(np.clip(k, 0, len(self.frames) - 1))
        v, arr = self.frames[k]
        self.section.show_frame(arr)
        self.label.setText(f"{self.param} = {v:g}  (frame {k + 1} / {len(self.frames)})")
        return v
