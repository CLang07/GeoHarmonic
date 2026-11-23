"""Well tie window: AI log, reflectivity, synthetic vs seismic at the well, shift / stretch, auto tie, save."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtWidgets  # noqa: E402
import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from .. import tie as TK  # noqa: E402
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
        # row 2 (item 16a/16b): window, auto-correlate, best trace, DTW stretch/squeeze, event pairs, QC, export
        bar2 = QtWidgets.QHBoxLayout()
        self.win_top = QtWidgets.QDoubleSpinBox()
        self.win_base = QtWidgets.QDoubleSpinBox()
        for sp_, pre in ((self.win_top, "window "), (self.win_base, "to ")):
            sp_.setRange(0, 20000)
            sp_.setDecimals(0)
            sp_.setSuffix(" ms")
            sp_.setPrefix(pre)
            sp_.setSpecialValueText(pre + "all")
        self.btn_corr = QtWidgets.QPushButton("Correlate")
        self.btn_corr.setToolTip("Correlation vs lag, polarity, phase scan, cycle-skip warning (Hampson-Russell style)")
        self.nbest_spin = QtWidgets.QSpinBox()
        self.nbest_spin.setRange(0, 10)
        self.nbest_spin.setPrefix("best trace within ")
        self.btn_best = QtWidgets.QPushButton("Find best trace")
        self.lag_spin = QtWidgets.QDoubleSpinBox()
        self.lag_spin.setRange(2, 100)
        self.lag_spin.setValue(20)
        self.lag_spin.setPrefix("DTW max lag ")
        self.lag_spin.setSuffix(" ms")
        self.strain_spin = QtWidgets.QDoubleSpinBox()
        self.strain_spin.setRange(1, 50)
        self.strain_spin.setValue(10)
        self.strain_spin.setPrefix("max dV ")
        self.strain_spin.setSuffix(" %")
        self.btn_dtw = QtWidgets.QPushButton("Auto stretch/squeeze")
        self.btn_pair = QtWidgets.QPushButton("Pick event pair")
        self.btn_pair.setCheckable(True)
        self.btn_pair.setToolTip("Click an event on the red synthetic, then the matching event on the black seismic")
        self.btn_undo = QtWidgets.QPushButton("Undo")
        self.btn_clear = QtWidgets.QPushButton("Clear warp")
        self.btn_drift = QtWidgets.QPushButton("Drift QC")
        self.btn_export = QtWidgets.QPushButton("Export TD CSV")
        for w in (self.win_top, self.win_base, self.btn_corr, self.nbest_spin, self.btn_best, self.lag_spin,
                  self.strain_spin, self.btn_dtw, self.btn_pair, self.btn_undo, self.btn_clear, self.btn_drift,
                  self.btn_export):
            bar2.addWidget(w)
        lay.addLayout(bar2)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        lay.addWidget(self.glw, 1)
        self.p_ai = self.glw.addPlot(title="AI")
        self.p_rc = self.glw.addPlot(title="Reflectivity")
        self.p_sy = self.glw.addPlot(title="Synthetic (red) vs seismic (black)")
        for pl in (self.p_rc, self.p_sy):
            pl.setYLink(self.p_ai)
        self.p_ai.invertY(True)
        self.p_cc = self.glw.addPlot(title="Correlation vs lag (ms)")
        self.glw.nextRow()
        self.p_dr = self.glw.addPlot(title="Lag of the warp (ms)", colspan=2)
        self.p_ph = self.glw.addPlot(title="Correlation vs phase rotation (deg)", colspan=2)
        self.p_sy.scene().sigMouseClicked.connect(self._plot_clicked)
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
        self.win_top.valueChanged.connect(lambda *_: self._win_changed())
        self.win_base.valueChanged.connect(lambda *_: self._win_changed())
        self.btn_corr.clicked.connect(lambda: self.correlate())
        self.btn_best.clicked.connect(lambda: self.find_best_trace())
        self.btn_dtw.clicked.connect(lambda: self.dtw())
        self.btn_undo.clicked.connect(lambda: self.undo())
        self.btn_clear.clicked.connect(lambda: self.clear_warp())
        self.btn_drift.clicked.connect(lambda: self.drift_qc())
        self.btn_export.clicked.connect(lambda: self._export_dialog())
        self.state = TK.TieState()
        self.window = None  # (top, base) ms of raw synthetic time, None = whole log
        self.max_lag_ms, self.max_strain = 20.0, 0.10
        self.trace_offset = 0  # index among the traces around the well used for the tie (best trace)
        self._pending = None  # synthetic time of a half-picked event pair
        self.last_corr = None
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

    def _win_changed(self):
        a, b = float(self.win_top.value()), float(self.win_base.value())
        self.window = (a, b) if b > a > 0 else None
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
        nb = max(1, int(self.nbest_spin.value()) if isinstance(self.nbest_spin.value(), (int, float)) else 1)
        ts, trs = ws.trace_at_well(self.source[0], self.survey, n, n_around=nb)
        if self.wavelet_kind == "Statistical":
            wav = WT.statistical_wavelet(trs, dt_ms, 128.0, (t[0], t[-1]), ts)
        elif self.wavelet_kind == "Deterministic" and getattr(self, "wavelet", None) is not None:
            wav = self.wavelet
        else:
            wav = WT.ricker(self.freq, dt_ms)
        rc, syn = WT.synthetic(t, ai, wav)
        k = int(np.clip(len(trs) // 2 + self.trace_offset, 0, len(trs) - 1))
        self._data = {"t": t, "ai": ai, "rc": rc, "syn": syn, "ts": ts, "seis": trs[k], "traces": trs,
                      "wavelet": wav, "td_source": td.source}
        if self.t_centre is None:
            self.t_centre = float(np.median(t))
        tie = (self.project.display.get("well_ties") or {}).get(n)
        if tie and self.corr is None:
            self.state = TK.TieState.from_dict(tie)
            self.shift, self.stretch = tie["shift_ms"], tie["stretch"]
            self.t_centre = tie.get("t_centre", self.t_centre)
        self.corr = self.tie_corr()
        return self._data

    # -------------------------------------------------- tie state (bulk + warp) and item 16 tools
    def _sync(self):
        """Bulk values from the spin boxes / attributes into the tie state (no undo step)."""
        self.state.shift, self.state.stretch = float(self.shift), float(self.stretch)
        self.state.t_centre = float(self.t_centre if self.t_centre is not None else 0.0)

    def _from_state(self):
        self.shift, self.stretch, self.t_centre = self.state.shift, self.state.stretch, self.state.t_centre
        for sp_, v in ((self.sh_spin, self.shift), (self.st_spin, self.stretch)):
            sp_.blockSignals(True)
            sp_.setValue(v)
            sp_.blockSignals(False)

    def tied_times(self, t=None):
        """Seismic times of the synthetic samples with the current tie (bulk + warp)."""
        self._sync()
        return self.state.apply(self._data["t"] if t is None else t)

    def _win_mask(self):
        t = self._data["t"]
        return TK.window_mask(t, self.window) if self.window else np.ones(len(t), bool)

    def tie_corr(self):
        d = self._data
        if d is None:
            return None
        m = self._win_mask()
        s = np.interp(self.tied_times()[m], d["ts"], d["seis"], left=np.nan, right=np.nan)
        return TK._pearson(s, d["syn"][m])

    def correlate(self):
        """Auto-correlate panel for the current tie: correlation vs lag, polarity, phase scan, cycle skip."""
        d = self._data or self.compute()
        if d is None:
            return None
        tt = self.tied_times()
        m = self._win_mask()
        dt = d["t"][1] - d["t"][0]
        tr = np.arange(tt[m][0], tt[m][-1] + 1e-9, dt)  # tied synthetic on a regular grid
        res = TK.correlate(tr, np.interp(tr, tt[m], d["syn"][m]), d["ts"], d["seis"], 60.0)
        self.last_corr = res
        try:
            self.p_cc.clear()
            self.p_cc.plot(res["lags_ms"], res["corr"], pen=pg.mkPen((0, 0, 160)))
            self.p_cc.plot(res["lags_ms"], res["envelope"], pen=pg.mkPen((150, 150, 150), style=QtCore.Qt.DashLine))
            self.p_ph.clear()
            self.p_ph.plot(res["phases_deg"], res["phase_corr"], pen=pg.mkPen((0, 120, 0)))
        except Exception:
            pass
        msg = (f"Correlate: residual lag {res['best_lag_ms']:+.1f} ms (corr {res['best_corr']:.2f}), polarity "
               f"{res['polarity']}, best phase rotation {res['best_phase_deg']:+.0f} deg (corr {res['best_phase_corr']:.2f})")
        if res["cycle_skip"]:
            msg += f"  WARNING: second peak at {res['second_peak'][0]:+.0f} ms (corr {res['second_peak'][1]:.2f}) - possible cycle skip"
        self.info.setText(msg)
        return res

    def find_best_trace(self):
        """Best-matching trace within N traces of the well (uses the current tie); switches the tie to it."""
        d = self._data or self.compute()
        if d is None:
            return None
        tt = self.tied_times()
        m = self._win_mask()
        dt = d["t"][1] - d["t"][0]
        tr = np.arange(tt[m][0], tt[m][-1] + 1e-9, dt)
        res = TK.best_trace(tr, np.interp(tr, tt[m], d["syn"][m]), d["ts"], d["traces"], 30.0)
        self.trace_offset = int(res["index"]) - len(d["traces"]) // 2
        d["seis"] = d["traces"][res["index"]]
        self.info.setText(f"Best trace: #{res['index'] + 1} of {len(d['traces'])} around the well "
                          f"(corr {res['corr']:.2f}, residual lag {res['lag_ms']:+.1f} ms)")
        self.draw()
        return res

    def dtw(self):
        """Constrained automatic stretch/squeeze (DTW) on top of the bulk tie."""
        d = self._data or self.compute()
        if d is None:
            return None
        try:
            self.max_lag_ms = float(self.lag_spin.value())
            self.max_strain = float(self.strain_spin.value()) / 100.0
        except Exception:
            pass
        self._sync()
        tb = self.state.bulk(d["t"])
        m = self._win_mask()
        dt = d["t"][1] - d["t"][0]
        tr = np.arange(tb[m][0], tb[m][-1] + 1e-9, dt)
        wi, wo, lag = TK.dtw_warp(tr, np.interp(tr, tb[m], d["syn"][m]), d["ts"], d["seis"], self.max_lag_ms,
                                  self.max_strain)
        self.state.set_warp(wi, wo, "dtw")
        self.draw()
        self.info.setText(f"Auto stretch/squeeze: lag {lag.min():+.1f} .. {lag.max():+.1f} ms, max local stretch "
                          f"{100 * TK.max_strain_of(wi, wo):.1f} % (limit {100 * self.max_strain:.0f} %), corr {self.corr:.2f}")
        return lag

    def add_pair(self, t_syn_tied, t_seis):
        self._sync()
        ok = self.state.add_pair(t_syn_tied, t_seis)
        self.draw()
        if not ok:
            self.info.setText("Event pair rejected: pairs must keep their order (no folding) and at most "
                              f"{TK.TieState.MAX_PAIRS} pairs")
        return ok

    def _plot_clicked(self, ev):
        if not self.btn_pair.isChecked() or self._data is None:
            return
        vb = self.p_sy.getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = vb.mapSceneToView(ev.scenePos())
        self.pick_event(p.x(), p.y())

    def pick_event(self, x, t):
        """x < 7.5: a synthetic event (first click); x >= 7.5: the matching seismic event (second click)."""
        if x < 7.5:
            self._pending = float(t)
            self.info.setText(f"Synthetic event at {t:.1f} ms - now click the matching seismic event")
            return None
        if self._pending is None:
            return None
        ok = self.add_pair(self._pending, float(t))
        self._pending = None
        return ok

    def undo(self):
        ok = self.state.undo()
        if ok:
            self._from_state()
            self.draw()
        return ok

    def clear_warp(self):
        self.state.clear_warp()
        self.draw()

    def drift_qc(self):
        """Tied time-depth vs the well's checkshot: drift and interval velocities per level (text table)."""
        if not self.well_name:
            return None
        cs = self.ws.checkshots().get(self.well_name)
        if cs is None:
            self.info.setText(f"{self.well_name}: no checkshot for drift QC")
            return None
        td = self._tied_td()
        rows = TK.drift_table(td.twt, cs[0], cs[1])
        try:
            self.p_dr.clear()
            self.p_dr.plot([r["z"] for r in rows], [r["drift_ms"] for r in rows], pen=pg.mkPen((160, 0, 0)),
                           symbol="o")
            self.p_dr.setTitle("Drift (tied - checkshot, ms) vs TVDSS")
        except Exception:
            pass
        worst = max(rows, key=lambda r: abs(r["drift_ms"]))
        self.info.setText(f"Drift QC: {len(rows)} checkshot levels, max |drift| {abs(worst['drift_ms']):.1f} ms at "
                          f"{worst['z']:.0f} m")
        return rows

    def _tied_td(self):
        """The well's raw time-depth with the dialog's current tie (not saved)."""
        base = self.ws.td_for(self.well_name)
        td = WT.WellTD(self.well_name, base.z, base.t, base.source)
        self._sync()
        td.shift, td.stretch, td.t_centre = self.state.shift, self.state.stretch, self.state.t_centre
        td.warp_in, td.warp_out = self.state.warp_in, self.state.warp_out
        return td

    def export_td(self, path, step_m=10.0):
        """Write the tied time-depth as a checkshot-style CSV (TVD_SRD_m, TWT_ms) - not into the well's
        checkshot folder (it would be merged with the real checkshot)."""
        td = self._tied_td()
        z = np.arange(0.0, float(td.z.max()) + step_m, step_m)
        return TK.write_td_csv(path, z, td.twt(z), self.well_name, "(tied in the Seismic Interpretation Tool)")

    def _export_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export tied time-depth", f"{self.well_name}_tied_td.csv",
                                                        "CSV (*.csv)")
        if path:
            self.export_td(path)
            self.info.setText(f"Wrote {path}")

    def auto(self):
        d = self._data or self.compute()
        if d is None:
            return None
        m = self._win_mask()
        res = TK.auto_tie_fft(d["t"][m], d["syn"][m], d["ts"], d["seis"], 100.0)
        self.state.set_bulk(res["shift_ms"], res["stretch"], res["t_centre"])
        self.state.warp_in = self.state.warp_out = None  # a new bulk tie invalidates the warp (undo restores it)
        self.state.pairs = []
        self._from_state()
        self.corr = res["corr"]
        self.draw()
        self.info.setText(f"Auto tie: shift {res['shift_ms']:+.1f} ms, stretch {res['stretch']:.3f}, corr {res['corr']:.2f}, "
                          f"NRMS {res['nrms']:.0f} %; held-out lower {30}% of the window: corr {res['corr_holdout']:.2f}")
        return res

    def save(self):
        if not self.well_name or self.t_centre is None:
            return
        self._sync()
        extra = {"wavelet": self.wavelet_kind, "freq": self.freq, "survey": list(self.source)}
        wd = self.state.to_dict()
        for k in ("warp_in", "warp_out", "warp_kind", "pairs"):
            if k in wd:
                extra[k] = wd[k]
        self.ws.save_tie(self.well_name, self.shift, self.stretch, self.t_centre, self.corr or 0.0, extra)
        if self.project.path:
            self.project.save()
        self.info.setText(f"Saved tie for {self.well_name}: shift {self.shift:.1f} ms, stretch {self.stretch:.3f}"
                          + (f", {self.state.warp_kind} warp" if self.state.warp_in is not None else ""))

    # ------------------------------------------------------------ drawing
    def draw(self):
        d = self._data
        for pl in (self.p_ai, self.p_rc, self.p_sy):
            pl.clear()
        if d is None:
            return
        tc = self.tied_times()
        self.corr = self.tie_corr()
        self.p_ai.plot(d["ai"], tc, pen=pg.mkPen((0, 0, 160)))
        self.p_rc.plot(d["rc"], tc, pen=pg.mkPen((0, 0, 0)))
        m = (d["ts"] >= tc[0] - 100) & (d["ts"] <= tc[-1] + 100)
        s = d["seis"][m]
        sc = np.abs(s).max() or 1.0
        y = d["syn"] / (np.abs(d["syn"]).max() or 1.0)
        for k in range(3):  # three copies side by side like a classic tie panel
            self.p_sy.plot(2.5 * k + y, tc, pen=pg.mkPen((200, 0, 0)))
            self.p_sy.plot(7.5 + 2.5 * k + s / sc, d["ts"][m], pen=pg.mkPen((0, 0, 0)))
        for a, b in self.state.pairs:
            self.p_sy.plot([2.5, 7.5], [a, b], pen=pg.mkPen((0, 140, 0), width=2))
        if self.state.warp_in is not None:
            self.p_dr.clear()
            self.p_dr.plot(self.state.warp_in, self.state.warp_out - self.state.warp_in, pen=pg.mkPen((0, 0, 160)))
            self.p_dr.setTitle("Lag of the warp (ms) vs TWT")
        self.corr_lbl.setText(f"corr {self.corr:.2f}")
        self.info.setText(f"{self.well_name}: time-depth from {d['td_source']}; wavelet {self.wavelet_kind}"
                          + (f" {self.freq:.0f} Hz" if self.wavelet_kind == "Ricker" else ""))
