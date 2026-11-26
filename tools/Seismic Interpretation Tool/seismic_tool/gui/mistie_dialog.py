"""Mistie analysis window: 2D-3D and 2D-2D ties, network solution (3D fixed), apply / reset corrections."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtWidgets  # noqa: E402

from .. import mistie as MI  # noqa: E402


class MistieDialog(QtWidgets.QDialog):
    def __init__(self, project, parent=None, on_apply=None, basemap=None):
        super().__init__(parent)
        self.setWindowTitle("Mistie analysis (2D-3D and 2D-2D)")
        self.project = project
        self.on_apply = on_apply
        self.time_window = None  # (t0, t1) ms; None = whole trace
        self.max_lag = 40.0
        self.robust = True
        self.ties, self.sol = [], {}
        self.on_residuals = None  # callable(ties, sol): the main window publishes them on the map bus (PLAN 31e)
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.t0 = QtWidgets.QDoubleSpinBox()
        self.t1 = QtWidgets.QDoubleSpinBox()
        for s_ in (self.t0, self.t1):
            s_.setRange(0, 20000)
            s_.setSuffix(" ms")
        self.t0.setValue(200)
        self.t1.setValue(2500)
        self.lag = QtWidgets.QDoubleSpinBox()
        self.lag.setRange(4, 500)
        self.lag.setValue(40)
        self.lag.setSuffix(" ms")
        self.rob = QtWidgets.QCheckBox("Robust (down-weight ties that disagree with the network)")
        self.rob.setChecked(True)
        self.flex_chk = QtWidgets.QCheckBox("Flex mode: shift varies along each line (after the constant solve)")
        self.smooth = QtWidgets.QDoubleSpinBox()
        self.smooth.setRange(0.01, 1000)
        self.smooth.setValue(1.0)
        self.smooth.setToolTip("Smoothness: small = follow every local tie, large = one constant per line")
        self.knots = QtWidgets.QSpinBox()
        self.knots.setRange(5, 2000)
        self.knots.setValue(50)
        self.knots.setSuffix(" traces")
        self.flex = False
        self.basemap = basemap
        self.hz_combo = QtWidgets.QComboBox()
        self.hz_combo.addItem("(waveform ties)")
        self.hz_combo.addItems(project.horizons.names())
        self.hz_combo.setToolTip("Use the misfit of an interpreted horizon at the ties instead of waveforms")
        self.ov_chk = QtWidgets.QCheckBox("Tie overlapping / duplicate lines along the overlap")
        self.ov_chk.setChecked(True)
        self.tv_chk = QtWidgets.QCheckBox("Time-variant shift for lines crossing a 3D (smooth in TWT)")
        self.groups_edit = QtWidgets.QLineEdit()
        self.groups_edit.setPlaceholderText("survey groups, e.g.  R-2=NPRA76, R-3=NPRA76")
        self.key_edit = QtWidgets.QLineEdit()
        self.key_edit.setPlaceholderText("key / reference lines held fixed, e.g.  R-2")
        self.horizon = None
        self.overlaps = True
        self.time_variant = False
        self.groups = {}
        self.key_lines = []
        form.addRow("Window start", self.t0)
        form.addRow("Window end", self.t1)
        form.addRow("Max shift", self.lag)
        form.addRow(self.rob)
        form.addRow(self.flex_chk)
        form.addRow("Flex smoothness", self.smooth)
        form.addRow("Flex knot spacing", self.knots)
        form.addRow("Ties from", self.hz_combo)
        form.addRow(self.ov_chk)
        form.addRow(self.tv_chk)
        form.addRow("Groups", self.groups_edit)
        form.addRow("Key lines", self.key_edit)
        lay.addLayout(form)
        self.tie_table = QtWidgets.QTableWidget(0, 9)
        self.tie_table.setHorizontalHeaderLabels(["A", "B", "type", "shift ms", "phase deg", "gain", "corr",
                                                  "residual ms", "weight"])
        self.sol_table = QtWidgets.QTableWidget(0, 5)
        self.sol_table.setHorizontalHeaderLabels(["line", "shift ms", "phase deg", "gain", "ties"])
        lay.addWidget(QtWidgets.QLabel("Ties (B relative to A)"))
        lay.addWidget(self.tie_table, 2)
        lay.addWidget(QtWidgets.QLabel("Corrections per line (3D volumes fixed)"))
        lay.addWidget(self.sol_table, 1)
        row = QtWidgets.QHBoxLayout()
        self.b_run = QtWidgets.QPushButton("Measure + solve")
        self.b_apply = QtWidgets.QPushButton("Apply corrections")
        self.b_reset = QtWidgets.QPushButton("Reset corrections")
        self.b_map = QtWidgets.QPushButton("Residuals on basemap")
        self.b_csv = QtWidgets.QPushButton("Export tie table...")
        self.b_sgy = QtWidgets.QPushButton("Export corrected SEG-Y...")
        for b in (self.b_run, self.b_apply, self.b_reset, self.b_map, self.b_csv, self.b_sgy):
            row.addWidget(b)
        lay.addLayout(row)
        self.info = QtWidgets.QLabel("Corrections are applied on the fly to the 2D display and tracking; 2D horizon "
                                     "picks move with them. The SEG-Y files are not changed.")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.resize(900, 650)
        self.b_run.clicked.connect(lambda: self._run())
        self.b_apply.clicked.connect(lambda: self.apply())
        self.b_reset.clicked.connect(lambda: self.reset())
        self.b_map.clicked.connect(lambda: self.show_residuals())
        self.b_csv.clicked.connect(lambda *_: self._export_csv())
        self.b_sgy.clicked.connect(lambda *_: self._export_segy())

    def _read(self):
        t0, t1 = float(self.t0.value()), float(self.t1.value())
        self.time_window = (t0, t1) if t1 > t0 else None
        self.max_lag = float(self.lag.value()) or 40.0
        self.robust = bool(self.rob.isChecked())
        self.flex = bool(self.flex_chk.isChecked())
        self.lam = float(self.smooth.value()) or 1.0
        self.knot_traces = int(self.knots.value()) or 50
        self.horizon = self.hz_combo.currentText() if self.hz_combo.currentIndex() > 0 else None
        self.overlaps = bool(self.ov_chk.isChecked())
        self.time_variant = bool(self.tv_chk.isChecked())
        self.groups = {}
        for part in str(self.groups_edit.text() or "").split(","):
            if "=" in part:
                a, b = part.split("=", 1)
                if a.strip() and b.strip():
                    self.groups[a.strip()] = b.strip()
        self.key_lines = [k.strip() for k in str(self.key_edit.text() or "").split(",") if k.strip()]

    def _run(self):
        self._read()
        try:
            self.compute()
        except Exception as e:
            self.info.setText(f"Mistie analysis failed: {e}")
            return
        self._fill()

    def compute(self):
        names = [l["name"] for l in self.project.lines]
        if self.horizon:
            self.ties = MI.horizon_ties(self.project, self.project.horizons.get(self.horizon))
        else:
            self.ties = MI.all_ties(self.project, self.time_window, self.max_lag)
        if self.overlaps:
            self.ties += MI.overlap_ties(self.project, window_ms=self.time_window, max_lag_ms=self.max_lag)
        if self.groups or self.key_lines:
            self.sol, self.ties = MI.solve_grouped(names, self.ties, self.groups, self.key_lines, robust=self.robust)
        else:
            self.sol, self.ties = MI.solve_network(names, self.ties, robust=self.robust)
        if self.time_variant and self.time_window:
            for v in [v["name"] for v in self.project.volumes]:
                for L in names:
                    try:
                        tv = MI.time_variant_line(self.project, v, L, self.time_window[0], self.time_window[1])
                    except Exception:
                        tv = None
                    if tv and L in self.sol:
                        c = self.sol[L]["shift_ms"]
                        self.sol[L]["tv"] = {"t": tv["t"], "shift": [s_ - c for s_ in tv["shift"]]}
        cl = MI.loop_closures(self.ties)
        worst = max((abs(c) for _, c in cl), default=0.0)
        msg = (f"{len(self.ties)} ties, {len(cl)} loops (worst raw closure {worst:.1f} ms). "
               "Large residuals = cycle skip or navigation error - check those ties.")
        if self.flex:
            lines = [k for k in self.sol if not k.startswith("3D:")]
            fx = MI.solve_flex(self.project, lines, self.ties, self.sol, getattr(self, "lam", 1.0),
                               getattr(self, "knot_traces", 50), self.time_window, self.max_lag)
            big = []
            for L, f in fx.items():
                self.sol[L]["flex"] = f
                if f["max_flex"] > 10.0:
                    big.append(f"{L} ({f['max_flex']:.0f} ms)")
            if big:
                msg += " WARNING large local flex: " + ", ".join(big) + " - likely cycle skip / navigation, not a mistie."
        self.info.setText(msg)
        return self.ties, self.sol

    def _fill(self):
        self.tie_table.setRowCount(len(self.ties))
        for r, t in enumerate(self.ties):
            vals = [t["a"], t["b"], t["kind"], f"{t['shift_ms']:.1f}", f"{t['phase_deg']:.0f}", f"{t['gain']:.2f}",
                    f"{t['corr']:.2f}", f"{t.get('res_shift_ms', float('nan')):.1f}", f"{t.get('weight', 0):.2f}"]
            for c, v in enumerate(vals):
                self.tie_table.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
        lines = [k for k in self.sol if not k.startswith("3D:")]
        self.sol_table.setRowCount(len(lines))
        for r, k in enumerate(lines):
            s = self.sol[k]
            for c, v in enumerate([k, f"{s['shift_ms']:.1f}", f"{s['phase_deg']:.0f}", f"{s['gain']:.2f}", s["n_ties"]]):
                self.sol_table.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))

    def show_residuals(self):
        MI.tie_positions(self.project, self.ties)
        if self.on_residuals is not None:
            try:
                self.on_residuals(self.ties, self.sol)
            except Exception:
                pass
        if self.basemap is not None:
            return self.basemap.show_ties(self.ties)
        return [t.get("xy") for t in self.ties]

    def _export_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Tie table", self.project.base(), "CSV (*.csv)")
        if path:
            MI.write_tie_table(self.ties, self.sol, path)
            self.info.setText(f"Wrote {path}")

    def _export_segy(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for corrected SEG-Y", self.project.base())
        if folder:
            n = self.export_segy(folder)
            self.info.setText(f"Wrote {n} corrected SEG-Y files to {folder}")

    def export_segy(self, folder):
        n = 0
        for l in self.project.lines:
            ln = self.project.line(l["name"])
            if ln.correction or getattr(ln, "time_variant", None):
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in l["name"])
                MI.export_corrected_segy(ln, os.path.join(folder, safe + "_mistie_corrected.sgy"))
                n += 1
        return n

    def apply(self):
        if not self.sol:
            self._read()  # the dialog's window / lag / groups / flex, not the defaults
            try:
                self.compute()
            except Exception as e:
                self.info.setText(f"Mistie analysis failed: {e}")
                return
        MI.apply_solution(self.project, self.sol)
        if self.project.path:
            self.project.save()
        if self.on_apply:
            self.on_apply()

    def reset(self):
        MI.apply_solution(self.project, {})
        if self.project.path:
            self.project.save()
        if self.on_apply:
            self.on_apply()
