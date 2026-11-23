"""Mistie analysis window: 2D-3D and 2D-2D ties, network solution (3D fixed), apply / reset corrections."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtWidgets  # noqa: E402

from .. import mistie as MI  # noqa: E402


class MistieDialog(QtWidgets.QDialog):
    def __init__(self, project, parent=None, on_apply=None):
        super().__init__(parent)
        self.setWindowTitle("Mistie analysis (2D-3D and 2D-2D)")
        self.project = project
        self.on_apply = on_apply
        self.window = None  # (t0, t1) ms; None = whole trace
        self.max_lag = 40.0
        self.robust = True
        self.ties, self.sol = [], {}
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
        form.addRow("Window start", self.t0)
        form.addRow("Window end", self.t1)
        form.addRow("Max shift", self.lag)
        form.addRow(self.rob)
        form.addRow(self.flex_chk)
        form.addRow("Flex smoothness", self.smooth)
        form.addRow("Flex knot spacing", self.knots)
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
        for b in (self.b_run, self.b_apply, self.b_reset):
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

    def _read(self):
        t0, t1 = float(self.t0.value()), float(self.t1.value())
        self.window = (t0, t1) if t1 > t0 else None
        self.max_lag = float(self.lag.value()) or 40.0
        self.robust = bool(self.rob.isChecked())
        self.flex = bool(self.flex_chk.isChecked())
        self.lam = float(self.smooth.value()) or 1.0
        self.knot_traces = int(self.knots.value()) or 50

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
        self.ties = MI.all_ties(self.project, self.window, self.max_lag)
        self.sol, self.ties = MI.solve_network(names, self.ties, robust=self.robust)
        cl = MI.loop_closures(self.ties)
        worst = max((abs(c) for _, c in cl), default=0.0)
        msg = (f"{len(self.ties)} ties, {len(cl)} loops (worst raw closure {worst:.1f} ms). "
               "Large residuals = cycle skip or navigation error - check those ties.")
        if self.flex:
            lines = [k for k in self.sol if not k.startswith("3D:")]
            fx = MI.solve_flex(self.project, lines, self.ties, self.sol, getattr(self, "lam", 1.0),
                               getattr(self, "knot_traces", 50), self.window, self.max_lag)
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

    def apply(self):
        if not self.sol:
            self.compute()
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
