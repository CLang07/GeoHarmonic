"""Interpret > Seismic inversion (PLAN 19d): low-frequency model, wavelet, post-stack (coloured / model-based /
blocky) and pre-stack simultaneous inversion, QC (well panel, blind-well test, residual), cross-plots, SEG-Y export.
Logic in plain methods (testable under dev/fakeqt); long runs go through MainWindow.run_job (threaded, Cancel)."""
import os

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtWidgets  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from ..inversion import common as CM  # noqa: E402
from ..inversion import lfm as L  # noqa: E402
from ..inversion import poststack as PS  # noqa: E402
from ..inversion import prestack as PR  # noqa: E402

WAVELETS = ["Ricker", "Statistical (from seismic)", "Library"]
POST_KINDS = ["model-based", "blocky (L1)", "coloured"]


class InversionDialog(QtWidgets.QDialog):
    def __init__(self, project, parent=None, run_job=None, on_volumes=None, volume=None):
        super().__init__(parent)
        self.setWindowTitle("Seismic inversion")
        self.project, self.run_job_fn, self.on_volumes = project, run_job, on_volumes
        self.vol_name = volume or next((v["name"] for v in project.volumes if not v.get("attribute")), None)
        self.curves = None
        self.lfm_names = {}
        self.wavelet = None
        self.wavelet_k = 1.0
        self.stacks = []
        self.blind_rows = []
        self.last = {}
        self._last_text = ""
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.vol_combo = QtWidgets.QComboBox()
        self.vol_combo.addItems([v["name"] for v in project.volumes])
        top.addWidget(QtWidgets.QLabel("Seismic volume"))
        top.addWidget(self.vol_combo, 1)
        lay.addLayout(top)
        tabs = QtWidgets.QTabWidget()
        lay.addWidget(tabs, 1)
        # ---- LFM
        t1 = QtWidgets.QWidget()
        l1 = QtWidgets.QVBoxLayout(t1)
        r = QtWidgets.QHBoxLayout()
        self.hz_list = QtWidgets.QListWidget()
        self.hz_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.hz_list.setToolTip("Horizons of the structural framework (select); none = interpolation along TWT")
        r.addWidget(self.hz_list, 1)
        form = QtWidgets.QFormLayout()
        self.zone_mode = QtWidgets.QComboBox()
        self.zone_mode.addItems(list(L.MODES))
        self.use_rgt = QtWidgets.QCheckBox("along RGT stack levels instead")
        self.method = QtWidgets.QComboBox()
        self.method.addItems(["idw", "kriging"])
        self.fc = QtWidgets.QLineEdit("8")
        self.b_fc = QtWidgets.QPushButton("Pick from seismic spectrum")
        self.props = {p: QtWidgets.QCheckBox(p) for p in L.PROPS}
        self.props["ai"].setChecked(True)
        prow = QtWidgets.QHBoxLayout()
        for c in self.props.values():
            prow.addWidget(c)
        form.addRow("Between horizons", self.zone_mode)
        form.addRow("", self.use_rgt)
        form.addRow("Lateral", self.method)
        form.addRow("Low-pass Hz", self.fc)
        form.addRow("", self.b_fc)
        form.addRow("Properties", prow)
        self.b_lfm = QtWidgets.QPushButton("Build LFM volume(s)")
        form.addRow("", self.b_lfm)
        r.addLayout(form, 1)
        l1.addLayout(r)
        tabs.addTab(t1, "Low-frequency model")
        # ---- post-stack
        t2 = QtWidgets.QWidget()
        l2 = QtWidgets.QVBoxLayout(t2)
        w_row = QtWidgets.QHBoxLayout()
        self.w_kind = QtWidgets.QComboBox()
        self.w_kind.addItems(WAVELETS)
        self.w_freq = QtWidgets.QLineEdit("30")
        self.w_lib = QtWidgets.QComboBox()
        self.w_scale = QtWidgets.QCheckBox("scale at the wells")
        self.w_scale.setChecked(True)
        self.b_wav = QtWidgets.QPushButton("Set wavelet")
        for w_ in (QtWidgets.QLabel("Wavelet"), self.w_kind, QtWidgets.QLabel("Hz"), self.w_freq, self.w_lib,
                   self.w_scale, self.b_wav):
            w_row.addWidget(w_)
        l2.addLayout(w_row)
        p_row = QtWidgets.QHBoxLayout()
        self.p_kind = QtWidgets.QComboBox()
        self.p_kind.addItems(POST_KINDS)
        self.p_lfm = QtWidgets.QComboBox()
        self.lam = QtWidgets.QLineEdit("0.05")
        self.mu = QtWidgets.QLineEdit("0.005")
        self.lateral = QtWidgets.QLineEdit("0")
        self.b_post = QtWidgets.QPushButton("Run inversion")
        self.b_blind = QtWidgets.QPushButton("Blind-well test")
        for w_ in (self.p_kind, QtWidgets.QLabel("LFM"), self.p_lfm, QtWidgets.QLabel("prior weight"), self.lam,
                   QtWidgets.QLabel("blocky weight"), self.mu, QtWidgets.QLabel("lateral"), self.lateral, self.b_post,
                   self.b_blind):
            p_row.addWidget(w_)
        l2.addLayout(p_row)
        q_row = QtWidgets.QHBoxLayout()
        self.qc_well = QtWidgets.QComboBox()
        self.b_qc = QtWidgets.QPushButton("QC at well")
        q_row.addWidget(QtWidgets.QLabel("Well"))
        q_row.addWidget(self.qc_well, 1)
        q_row.addWidget(self.b_qc)
        l2.addLayout(q_row)
        self.qc_plot = pg.PlotWidget(title="Log (black) / LFM (blue) / inversion (red)")
        self.res_plot = pg.PlotWidget(title="Seismic (black) / synthetic of the result (red)")
        pl = QtWidgets.QHBoxLayout()
        pl.addWidget(self.qc_plot)
        pl.addWidget(self.res_plot)
        l2.addLayout(pl, 1)
        tabs.addTab(t2, "Post-stack")
        # ---- pre-stack
        t3 = QtWidgets.QWidget()
        l3 = QtWidgets.QVBoxLayout(t3)
        self.st_table = QtWidgets.QTableWidget(0, 2)
        self.st_table.setHorizontalHeaderLabels(["angle stack volume", "mean angle deg"])
        l3.addWidget(self.st_table, 1)
        s_row = QtWidgets.QHBoxLayout()
        self.st_vol = QtWidgets.QComboBox()
        self.st_vol.addItems([v["name"] for v in project.volumes])
        self.st_ang = QtWidgets.QLineEdit("10")
        self.b_st_add = QtWidgets.QPushButton("Add stack")
        self.b_st_clear = QtWidgets.QPushButton("Clear")
        self.lam3 = QtWidgets.QLineEdit("0.02, 0.1, 1.0")
        self.lam3.setToolTip("prior weights for ln Zp, Zs deviation, density deviation")
        self.b_pre = QtWidgets.QPushButton("Run simultaneous inversion")
        for w_ in (self.st_vol, QtWidgets.QLabel("angle"), self.st_ang, self.b_st_add, self.b_st_clear,
                   QtWidgets.QLabel("prior weights"), self.lam3, self.b_pre):
            s_row.addWidget(w_)
        l3.addLayout(s_row)
        tabs.addTab(t3, "Pre-stack")
        # ---- cross-plot / export
        t4 = QtWidgets.QWidget()
        l4 = QtWidgets.QVBoxLayout(t4)
        c_row = QtWidgets.QHBoxLayout()
        self.cx = QtWidgets.QComboBox()
        self.cy = QtWidgets.QComboBox()
        self.cc = QtWidgets.QComboBox()
        self.b_cross = QtWidgets.QPushButton("Cross-plot")
        self.b_segy = QtWidgets.QPushButton("Export volume as SEG-Y...")
        for w_ in (QtWidgets.QLabel("X"), self.cx, QtWidgets.QLabel("Y"), self.cy, QtWidgets.QLabel("colour"), self.cc,
                   self.b_cross, self.b_segy):
            c_row.addWidget(w_)
        l4.addLayout(c_row)
        self.cross_plot = pg.PlotWidget(title="Cross-plot")
        l4.addWidget(self.cross_plot, 1)
        tabs.addTab(t4, "Cross-plot / export")
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.resize(1100, 620)
        self.refresh_lists()
        self.vol_combo.currentTextChanged.connect(lambda t: self._safe(lambda: self.set_volume(t)))
        self.b_fc.clicked.connect(lambda *_: self._safe(self.auto_fc))
        self.b_lfm.clicked.connect(lambda *_: self._safe(self._build_lfm_clicked))
        self.b_wav.clicked.connect(lambda *_: self._safe(lambda: self.set_wavelet(
            self.w_kind.currentText(), float(self.w_freq.text() or 30), self.w_lib.currentText(),
            self.w_scale.isChecked())))
        self.b_post.clicked.connect(lambda *_: self._safe(lambda: self.run_poststack(
            self.p_kind.currentText(), self.p_lfm.currentText(), float(self.lam.text()), float(self.mu.text()),
            float(self.lateral.text() or 0))))
        self.b_blind.clicked.connect(lambda *_: self._safe(lambda: self.blind_test(float(self.lam.text()))))
        self.b_qc.clicked.connect(lambda *_: self._safe(lambda: self.qc_at_well(self.qc_well.currentText())))
        self.b_st_add.clicked.connect(lambda *_: self._safe(lambda: self.add_stack(self.st_vol.currentText(),
                                                                                   float(self.st_ang.text()))))
        self.b_st_clear.clicked.connect(lambda *_: self._safe(self.clear_stacks))
        self.b_pre.clicked.connect(lambda *_: self._safe(lambda: self.run_prestack(
            tuple(float(v) for v in self.lam3.text().split(",")))))
        self.b_cross.clicked.connect(lambda *_: self._safe(lambda: self.crossplot(
            self.cx.currentText(), self.cy.currentText(), self.cc.currentText())))
        self.b_segy.clicked.connect(lambda *_: self._safe(self._segy_dialog))

    # ------------------------------------------------------------------ helpers
    def _safe(self, fn):
        try:
            return fn()
        except Exception as e:
            self._set_info(f"Inversion: {e}")
            return None

    def _set_info(self, text):
        self._last_text = text
        self.info.setText(text)

    def volume_names(self):
        return [v["name"] for v in self.project.volumes]

    def refresh_lists(self):
        names = self.volume_names()
        for combo in (self.cx, self.cy, self.st_vol, self.p_lfm):
            combo.clear()
        self.cx.addItems(names)
        self.cy.addItems(names)
        self.st_vol.addItems(names)
        self.p_lfm.addItems([n for n in names if "LFM" in n] or names)
        self.cc.clear()
        self.cc.addItems(["(none)"] + names + ["log: GR"])
        self.hz_list.clear()
        key = f"3D:{self.vol_name}"
        self.hz_list.addItems([h.name for h in self.project.horizons.items if h.get(key) is not None])
        self.qc_well.clear()
        if self.curves:
            self.qc_well.addItems([c.name for c in self.curves])
        try:
            from ..tie import WaveletLibrary
            lib = WaveletLibrary(os.path.join(self.project.seismic_dir(), "wavelets.json")) if self.project.path \
                else None
            self.w_lib.clear()
            if lib is not None:
                self.w_lib.addItems(lib.names())
        except Exception:
            pass

    def set_volume(self, name):
        if name not in self.volume_names():
            return
        self.vol_name = name
        self.curves = None
        self.refresh_lists()

    def _done(self, names):
        self.refresh_lists()
        if self.on_volumes is not None:
            self.on_volumes(names)

    def _run(self, label, fn, done):
        """Threaded through MainWindow.run_job when available, else inline."""
        if self.run_job_fn is not None:
            return self.run_job_fn(label, fn, done)
        fn(lambda f, t: None, lambda: False)
        done()
        return True

    def load_curves(self):
        if self.curves is None:
            self.curves = L.well_curves(self.project, self.vol_name)
            self.qc_well.clear()
            self.qc_well.addItems([c.name for c in self.curves])
        if not self.curves:
            raise RuntimeError("no well with sonic (DT) inside the survey - link a well project with logs and ties")
        return self.curves

    # ------------------------------------------------------------------ LFM
    def auto_fc(self):
        vol = self.project.volume(self.vol_name)
        rng = np.random.default_rng(0)
        idx = rng.choice(vol.nil * vol.nxl, min(300, vol.nil * vol.nxl), replace=False)
        tr = np.asarray(vol.ilxl[idx // vol.nxl, idx % vol.nxl], np.float32) * vol.inv
        f, A = CM.amplitude_spectrum(tr, vol.dt * 1000.0)
        fc = CM.pick_lowcut(f, A)
        self.fc.setText(f"{fc:.1f}")
        self._set_info(f"Seismic band starts at about {fc:.1f} Hz - LFM low-pass set to it")
        return fc

    def _build_lfm_clicked(self):
        hz = [it.text() for it in self.hz_list.selectedItems()]
        props = [p for p, c in self.props.items() if c.isChecked()]
        return self.build_lfm(props, hz, self.zone_mode.currentText(), self.method.currentText(),
                              float(self.fc.text() or 0), self.use_rgt.isChecked())

    def build_lfm(self, props=("ai",), horizons=(), mode="proportional", method="idw", fc=8.0, use_rgt=False):
        curves = self.load_curves()
        modes = [mode] * (len(horizons) + 1)
        builders = [L.VolumeLFM(self.project, self.vol_name, curves, p, list(horizons), modes, method, fc=fc or None,
                                use_rgt=bool(use_rgt)) for p in props]
        out = []

        def job(progress, cancel):
            for k, b in enumerate(builders):
                out.append(b.write(progress=lambda f, t, k=k: progress((k + f) / len(builders), t), cancel=cancel))

        def done():
            self.lfm_names.update({p: n for p, n in zip(props, out)})
            self._set_info(f"LFM from {len(curves)} wells ({', '.join(c.name for c in curves)}), "
                           f"{len(horizons)} horizons{' + RGT' if use_rgt else ''}, {fc:g} Hz: " + ", ".join(out))
            self._done(out)
        self._run("LFM", job, done)
        return out

    # ------------------------------------------------------------------ wavelet + post-stack
    def set_wavelet(self, kind="Ricker", freq=30.0, lib_name=None, scale=True):
        from ..tie import WaveletLibrary
        from ..welltie import ricker, statistical_wavelet
        vol = self.project.volume(self.vol_name)
        dt = vol.dt * 1000.0
        if kind == "Library":
            lib = WaveletLibrary(os.path.join(self.project.seismic_dir(), "wavelets.json"))
            if lib_name not in lib.names():
                raise ValueError("pick a wavelet of the project's library (Well tie > Save wavelet)")
            w = lib.get(lib_name, dt)
        elif kind.startswith("Statistical"):
            idx = np.random.default_rng(1).choice(vol.nil * vol.nxl, min(200, vol.nil * vol.nxl), replace=False)
            tr = np.asarray(vol.ilxl[idx // vol.nxl, idx % vol.nxl], np.float32) * vol.inv
            w = statistical_wavelet(tr, dt)
        else:
            w = ricker(float(freq), dt)
        k_all = []
        if scale:
            for c in self.load_curves():
                il, xl = vol.ilxl_at(c.x, c.y)
                s = np.asarray(vol.ilxl[vol.il_index(il), vol.xl_index(xl)], np.float32).astype(float) * vol.inv
                k_all.append(PS.scale_wavelet(w, s, c.values["ai"])[1])
        k = float(np.median(k_all)) if k_all else 1.0
        self.wavelet = np.asarray(w, float) * k
        self.wavelet_k = k
        self._set_info(f"Wavelet: {kind} ({len(w)} samples), scale {k:.4g}" +
                       (f" from {len(k_all)} wells" if k_all else ""))
        return self.wavelet

    def run_poststack(self, kind="model-based", lfm_name=None, lam=0.05, mu=0.005, lateral=0.0):
        if kind == "coloured":
            ci = PS.Coloured(self.project, self.vol_name, self.curves if self.curves else ())
            out = []

            def job(progress, cancel):
                out.append(ci.write(progress=progress, cancel=cancel))

            def done():
                self._set_info(f"Coloured inversion: operator alpha {ci.alpha:.2f}, scale {ci.scale:.3g} -> {out[0]}")
                self._done(out)
            self._run("Coloured inversion", job, done)
            return out
        lfm_name = lfm_name or self.lfm_names.get("ai")
        if not lfm_name or lfm_name not in self.volume_names():
            raise RuntimeError("build an AI low-frequency model first")
        if self.wavelet is None:
            raise RuntimeError("set the wavelet first")
        inv = PS.PostStack(self.project, self.vol_name, lfm_name, self.wavelet, lam, mu if kind.startswith("blocky")
                           else 0.0, lateral)
        out = []

        def job(progress, cancel):
            out.append(inv.write(progress=progress, cancel=cancel))

        def done():
            self.last["poststack"] = out[0]
            self._set_info(f"Model-based inversion ({kind}, prior weight {lam:g}) -> {out[0]}")
            self._done(out)
        self._run("Inversion", job, done)
        return out

    def blind_test(self, lam=0.05, horizons=None, fc=None):
        curves = self.load_curves()
        if self.wavelet is None:
            raise RuntimeError("set the wavelet first")
        hz = horizons if horizons is not None else [it.text() for it in self.hz_list.selectedItems()]
        fc = fc if fc is not None else float(self.fc.text() or 8)
        self.blind_rows = PS.blind_well_test(self.project, self.vol_name, curves, self.wavelet, hz, None, fc, lam)
        self._set_info("Blind-well test (LFM without the well, inverted at its trace, seismic band):\n" + "\n".join(
            f"{r['well']}: r = {r['corr']:.2f}, AI error {r['error_pct']:.1f} % (LFM alone {r['lfm_error_pct']:.1f} %), "
            f"residual NRMS {r['nrms']:.0f} %" for r in self.blind_rows))
        return self.blind_rows

    def qc_at_well(self, well, inv_name=None):
        """Log / LFM / inversion at a well, seismic vs synthetic of the inversion; draws both QC plots."""
        curves = {c.name: c for c in self.load_curves()}
        c = curves[well]
        vol = self.project.volume(self.vol_name)
        il, xl = vol.ilxl_at(c.x, c.y)
        i, j = vol.il_index(il), vol.xl_index(xl)
        inv_name = inv_name or self.last.get("poststack")
        out = {"t": vol.times, "log": c.values["ai"],
               "seismic": np.asarray(vol.ilxl[i, j], np.float32).astype(float) * vol.inv}
        for key, name in (("lfm", self.lfm_names.get("ai")), ("inv", inv_name)):
            if name and name in self.volume_names():
                v = self.project.volume(name)
                out[key] = np.asarray(v.ilxl[i, j], np.float32).astype(float) * v.inv
        if "inv" in out and self.wavelet is not None:
            z = np.log(np.maximum(out["inv"], 1e-6))[None, :]
            out["synthetic"] = PS.synthetic_ln(z, self.wavelet)[0]
            out["nrms"] = float(PS.residual_nrms(out["seismic"][None, :], z, self.wavelet)[0])
            out["error_pct"] = PS.band_error(z[0], c.values["ai"], vol.dt * 1000.0, 60.0)
        self.qc_plot.clear()
        self.qc_plot.plot(out["log"], out["t"], pen=pg.mkPen((0, 0, 0)), connect="finite")
        if "lfm" in out:
            self.qc_plot.plot(out["lfm"], out["t"], pen=pg.mkPen((0, 0, 220), width=2))
        if "inv" in out:
            self.qc_plot.plot(out["inv"], out["t"], pen=pg.mkPen((220, 0, 0)))
        self.qc_plot.getPlotItem().invertY(True)
        self.res_plot.clear()
        self.res_plot.plot(out["seismic"], out["t"], pen=pg.mkPen((0, 0, 0)))
        if "synthetic" in out:
            self.res_plot.plot(out["synthetic"], out["t"], pen=pg.mkPen((220, 0, 0)))
        self.res_plot.getPlotItem().invertY(True)
        if "nrms" in out:
            self._set_info(f"{well}: AI error {out['error_pct']:.1f} % in the seismic band, residual NRMS "
                           f"{out['nrms']:.0f} %")
        self.last["qc"] = out
        return out

    # ------------------------------------------------------------------ pre-stack
    def add_stack(self, vol_name, angle):
        if vol_name not in self.volume_names():
            raise ValueError(f"unknown volume {vol_name!r}")
        self.stacks = [s for s in self.stacks if s[0] != vol_name] + [(vol_name, float(angle))]
        self.stacks.sort(key=lambda s: s[1])
        self.st_table.setRowCount(len(self.stacks))
        for r, (n, a) in enumerate(self.stacks):
            self.st_table.setItem(r, 0, QtWidgets.QTableWidgetItem(n))
            self.st_table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{a:g}"))
        return self.stacks

    def clear_stacks(self):
        self.stacks = []
        self.st_table.setRowCount(0)

    def run_prestack(self, lam=(0.02, 0.1, 1.0), wavelets=None):
        if len(self.stacks) < 2:
            raise RuntimeError("add two or more angle stacks (volume + mean angle)")
        missing = [p for p in ("ai", "vs", "rho") if self.lfm_names.get(p) not in self.volume_names()]
        if missing:
            raise RuntimeError("build LFM volumes for " + ", ".join(missing) + " first (tick ai, vs, rho)")
        if wavelets is None:
            if self.wavelet is None:
                raise RuntimeError("set the wavelet first")
            wavelets = [self.wavelet] * len(self.stacks)
        bg = PR.fit_background(self.load_curves())
        ps = PR.PreStack(self.project, self.stacks, wavelets, {p: self.lfm_names[p] for p in ("ai", "vs", "rho")}, bg,
                         tuple(lam))
        out = []

        def job(progress, cancel):
            out.extend(ps.write(progress=progress, cancel=cancel))

        def done():
            self.last["prestack"] = list(out)
            self._set_info(f"Simultaneous inversion of {len(self.stacks)} stacks (background ln Zs = "
                           f"{bg['k']:.2f} ln Zp + c, ln rho = {bg['m']:.2f} ln Zp + c): " + ", ".join(out))
            self._done(out)
        self._run("Simultaneous inversion", job, done)
        return out

    # ------------------------------------------------------------------ cross-plot + export
    def crossplot(self, xname, yname, colour="(none)", n=20000):
        """Samples of two volumes (random traces, all times with finite values) coloured by a third volume or a
        well log ('log: GR' -> samples along the well traces only). Returns dict x, y, c."""
        vx, vy = self.project.volume(xname), self.project.volume(yname)
        if (vx.nil, vx.nxl, vx.ns) != (vy.nil, vy.nxl, vy.ns):
            raise ValueError("the two volumes must share the grid")
        if str(colour).startswith("log:"):
            log = colour.split(":", 1)[1].strip()
            ii, jj, cc = self._log_samples(vx, log)
            xs = (np.asarray(vx.ilxl[ii, jj], np.float32).astype(float) * vx.inv).ravel()
            ys = (np.asarray(vy.ilxl[ii, jj], np.float32).astype(float) * vy.inv).ravel()
            cs = cc.ravel()
        else:
            rng = np.random.default_rng(2)
            nt = max(1, min(vx.nil * vx.nxl, n // max(1, vx.ns // 4)))
            idx = rng.choice(vx.nil * vx.nxl, nt, replace=False)
            ii, jj = idx // vx.nxl, idx % vx.nxl
            xs = (np.asarray(vx.ilxl[ii, jj], np.float32).astype(float) * vx.inv)[:, ::4].ravel()
            ys = (np.asarray(vy.ilxl[ii, jj], np.float32).astype(float) * vy.inv)[:, ::4].ravel()
            cs = None
            if colour in self.volume_names():
                vc = self.project.volume(colour)
                cs = (np.asarray(vc.ilxl[ii, jj], np.float32).astype(float) * vc.inv)[:, ::4].ravel()
        ok = np.isfinite(xs) & np.isfinite(ys) & ((cs is None) or np.isfinite(cs))
        res = {"x": xs[ok], "y": ys[ok], "c": None if cs is None else cs[ok]}
        self.cross_plot.clear()
        brushes = None
        if res["c"] is not None and len(res["c"]):
            from .colormaps import lut
            lt = lut("Rainbow (attributes)", False)
            lo, hi = np.percentile(res["c"], [2, 98])
            k = np.clip(((res["c"] - lo) / max(1e-12, hi - lo) * (len(lt) - 1)).astype(int), 0, len(lt) - 1)
            brushes = [pg.mkBrush(*lt[q][:3]) for q in k]
        self.cross_plot.addItem(pg.ScatterPlotItem(res["x"], res["y"], size=3, pen=None,
                                                   brush=brushes if brushes else pg.mkBrush(60, 60, 200, 120)))
        self.cross_plot.setLabel("bottom", xname)
        self.cross_plot.setLabel("left", yname)
        self.last["crossplot"] = res
        return res

    def _log_samples(self, vol, log):
        """(il idx (m, 1), xl idx (m, 1), log values (m, ns)) along the wells' traces on the volume's time axis."""
        ws = self.project.wellset()
        II, JJ, CC = [], [], []
        for c in self.load_curves():
            w = ws.wcp.load_well(c.name)
            a = w.logs.get(log)
            if a is None:
                continue
            td = ws.td_for(c.name)
            t = td.twt(w.md_to_tvdss(np.asarray(w.depth, float)))
            ok = np.isfinite(a) & np.isfinite(t)
            if ok.sum() < 5:
                continue
            o = np.argsort(t[ok])
            v = np.interp(vol.times, t[ok][o], np.asarray(a, float)[ok][o], left=np.nan, right=np.nan)
            il, xl = vol.ilxl_at(c.x, c.y)
            II.append(vol.il_index(il))
            JJ.append(vol.xl_index(xl))
            CC.append(v)
        if not CC:
            raise ValueError(f"no well has a {log} log")
        return np.array(II), np.array(JJ), np.array(CC)

    def export_segy(self, name, path, threaded=False):
        vol = self.project.volume(name)
        if threaded and self.run_job_fn is not None:
            return self.run_job_fn("SEG-Y export", lambda p, c: CM.export_segy(vol, path, p, c, name),
                                   lambda: self._set_info(f"Written {path}"))
        CM.export_segy(vol, path, note=name)
        self._set_info(f"Written {path}")
        return path

    def _segy_dialog(self):
        name = self.cx.currentText()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export SEG-Y", f"{name}.sgy", "SEG-Y (*.sgy *.segy)")
        if path:
            self.export_segy(name, path, threaded=True)
