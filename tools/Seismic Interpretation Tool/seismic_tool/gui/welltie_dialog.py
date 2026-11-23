"""Well tie window: AI log, reflectivity, synthetic vs seismic at the well, shift / stretch, auto tie, save."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtWidgets  # noqa: E402
import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from .. import welltie as WT  # noqa: E402


class WellTieDialog(QtWidgets.QDialog):
    def __init__(self, project, source, parent=None, well=None):
        super().__init__(parent)
        self.setWindowTitle("Well tie (synthetic seismogram)")
        self.project = project
        self.source = tuple(source)
        self.ws = project.wellset()
        kind, name = self.source
        self.survey = project.volume(name) if kind == "3D" else project.line(name)
        self.well_name = well
        self.freq = 30.0
        self.wavelet_kind = "Ricker"
        self.shift, self.stretch, self.t_centre, self.corr = 0.0, 1.0, None, None
        self._data = None
        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.well_combo = QtWidgets.QComboBox()
        cands = [n for n in self.ws.names if self._inside(n)]
        self.well_combo.addItems(cands or self.ws.names)
        if well:
            self.well_combo.setCurrentText(well)
        self.wav_combo = QtWidgets.QComboBox()
        self.wav_combo.addItems(["Ricker", "Statistical (from seismic)"])
        self.f_spin = QtWidgets.QDoubleSpinBox()
        self.f_spin.setRange(5, 120)
        self.f_spin.setValue(30)
        self.f_spin.setSuffix(" Hz")
        self.sh_spin = QtWidgets.QDoubleSpinBox()
        self.sh_spin.setRange(-500, 500)
        self.sh_spin.setSingleStep(1)
        self.sh_spin.setSuffix(" ms shift")
        self.st_spin = QtWidgets.QDoubleSpinBox()
        self.st_spin.setRange(0.8, 1.2)
        self.st_spin.setDecimals(3)
        self.st_spin.setSingleStep(0.002)
        self.st_spin.setValue(1.0)
        self.st_spin.setPrefix("stretch ")
        self.btn_auto = QtWidgets.QPushButton("Auto tie")
        self.btn_save = QtWidgets.QPushButton("Save tie")
        self.corr_lbl = QtWidgets.QLabel("")
        for w in (self.well_combo, self.wav_combo, self.f_spin, self.sh_spin, self.st_spin, self.btn_auto,
                  self.btn_save, self.corr_lbl):
            bar.addWidget(w)
        lay.addLayout(bar)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        lay.addWidget(self.glw, 1)
        self.p_ai = self.glw.addPlot(title="AI")
        self.p_rc = self.glw.addPlot(title="Reflectivity")
        self.p_sy = self.glw.addPlot(title="Synthetic (red) vs seismic (black)")
        for pl in (self.p_rc, self.p_sy):
            pl.setYLink(self.p_ai)
        self.p_ai.invertY(True)
        self.info = QtWidgets.QLabel("")
        lay.addWidget(self.info)
        self.resize(1000, 800)
        self.well_combo.currentTextChanged.connect(lambda *_: self._changed())
        self.wav_combo.currentTextChanged.connect(lambda *_: self._changed())
        self.f_spin.valueChanged.connect(lambda *_: self._changed())
        self.sh_spin.valueChanged.connect(lambda *_: self._tie_changed())
        self.st_spin.valueChanged.connect(lambda *_: self._tie_changed())
        self.btn_auto.clicked.connect(lambda: self.auto())
        self.btn_save.clicked.connect(lambda: self.save())
        if self.well_name is None and self.well_combo.count():
            self.well_name = self.well_combo.currentText()

    def _inside(self, n):
        kind = self.source[0]
        w = self.ws.well(n)
        if w.x is None:
            return False
        if kind == "3D":
            il, xl = self.survey.ilxl_at(w.x, w.y)
            s = self.survey
            return s.inlines[0] <= il <= s.inlines[-1] and s.xlines[0] <= xl <= s.xlines[-1]
        return self.survey.has_coords and np.min(np.hypot(self.survey.x - w.x, self.survey.y - w.y)) < 1000

    # ------------------------------------------------------------ logic (testable)
    def _changed(self):
        self.well_name = self.well_combo.currentText() or self.well_name
        self.wavelet_kind = "Statistical" if self.wav_combo.currentText().startswith("Statistical") else "Ricker"
        self.freq = float(self.f_spin.value()) or self.freq
        self.compute()
        self.draw()

    def _tie_changed(self):
        self.shift = float(self.sh_spin.value())
        self.stretch = float(self.st_spin.value()) or 1.0
        self.draw()

    def compute(self):
        """Synthetic + seismic at the well for the current settings (raw time-depth, no saved tie)."""
        if not self.well_name:
            return None
        ws, n = self.ws, self.well_name
        ws.wcp.load_well(n)
        td = ws.td_for(n)
        w = ws.well(n)
        dt_ms = self.survey.dt * 1000.0
        r = WT.logs_in_time(w, td, dt_ms)
        if r is None:
            self.info.setText(f"{n}: no sonic (DT) - cannot make a synthetic")
            self._data = None
            return None
        t, ai = r
        ts, trs = ws.trace_at_well(self.source[0], self.survey, n, n_around=1)
        if self.wavelet_kind == "Statistical":
            wav = WT.statistical_wavelet(trs, dt_ms, 128.0, (t[0], t[-1]), ts)
        else:
            wav = WT.ricker(self.freq, dt_ms)
        rc, syn = WT.synthetic(t, ai, wav)
        centre = trs[len(trs) // 2]
        self._data = {"t": t, "ai": ai, "rc": rc, "syn": syn, "ts": ts, "seis": centre, "wavelet": wav,
                      "td_source": td.source}
        if self.t_centre is None:
            self.t_centre = float(np.median(t))
        tie = (self.project.display.get("well_ties") or {}).get(n)
        if tie and self.corr is None:
            self.shift, self.stretch = tie["shift_ms"], tie["stretch"]
            self.t_centre = tie.get("t_centre", self.t_centre)
        self.corr = WT.tie_correlation(t, syn, ts, centre, self.shift, self.stretch, self.t_centre)
        return self._data

    def auto(self):
        d = self._data or self.compute()
        if d is None:
            return None
        res = WT.auto_tie(d["t"], d["syn"], d["ts"], d["seis"], 100.0)
        self.shift, self.stretch, self.t_centre, self.corr = res["shift_ms"], res["stretch"], res["t_centre"], res["corr"]
        for sp_, v in ((self.sh_spin, self.shift), (self.st_spin, self.stretch)):
            sp_.blockSignals(True)
            sp_.setValue(v)
            sp_.blockSignals(False)
        self.draw()
        return res

    def save(self):
        if not self.well_name or self.t_centre is None:
            return
        self.ws.save_tie(self.well_name, self.shift, self.stretch, self.t_centre, self.corr or 0.0,
                         {"wavelet": self.wavelet_kind, "freq": self.freq, "survey": list(self.source)})
        if self.project.path:
            self.project.save()
        self.info.setText(f"Saved tie for {self.well_name}: shift {self.shift:.1f} ms, stretch {self.stretch:.3f}")

    # ------------------------------------------------------------ drawing
    def draw(self):
        d = self._data
        for pl in (self.p_ai, self.p_rc, self.p_sy):
            pl.clear()
        if d is None:
            return
        t = d["t"]
        tc = t + self.shift + (self.stretch - 1.0) * (t - (self.t_centre or 0.0))
        self.corr = WT.tie_correlation(t, d["syn"], d["ts"], d["seis"], self.shift, self.stretch, self.t_centre or 0.0)
        self.p_ai.plot(d["ai"], tc, pen=pg.mkPen((0, 0, 160)))
        self.p_rc.plot(d["rc"], tc, pen=pg.mkPen((0, 0, 0)))
        m = (d["ts"] >= tc[0] - 100) & (d["ts"] <= tc[-1] + 100)
        s = d["seis"][m]
        sc = np.abs(s).max() or 1.0
        y = d["syn"] / (np.abs(d["syn"]).max() or 1.0)
        for k in range(3):  # three copies side by side like a classic tie panel
            self.p_sy.plot(2.5 * k + y, tc, pen=pg.mkPen((200, 0, 0)))
            self.p_sy.plot(7.5 + 2.5 * k + s / sc, d["ts"][m], pen=pg.mkPen((0, 0, 0)))
        self.corr_lbl.setText(f"corr {self.corr:.2f}")
        self.info.setText(f"{self.well_name}: time-depth from {d['td_source']}; wavelet {self.wavelet_kind}"
                          + (f" {self.freq:.0f} Hz" if self.wavelet_kind == "Ricker" else ""))
