"""Well prognosis / geosteering window (PLAN item 17d): predicted horizon depths along a planned or drilling well,
live with the dynamic depth model (View > Auto-update depth model)."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtWidgets  # noqa: E402

from ..prognosis import Prognosis, vertical_trajectory  # noqa: E402

COLS = ["horizon", "formation", "MD m", "TVDSS m", "X", "Y", "status", "+/- 1 sigma m", "change m"]


class PrognosisDialog(QtWidgets.QDialog):
    def __init__(self, project, depth_model_fn, parent=None, on_change=None):
        """depth_model_fn() -> the ddc.DynamicDepthModel in use (the section's)."""
        super().__init__(parent)
        self.setWindowTitle("Well prognosis / geosteering")
        self.project, self.depth_model_fn, self.on_change = project, depth_model_fn, on_change
        self.prog = None
        self.uncertainty = {}
        self._trajs = {}
        self._last_text = ""
        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.well = QtWidgets.QComboBox()
        row.addWidget(QtWidgets.QLabel("Well"))
        row.addWidget(self.well, 1)
        self.b_load = QtWidgets.QPushButton("Show")
        row.addWidget(self.b_load)
        lay.addLayout(row)
        prow = QtWidgets.QHBoxLayout()
        self.p_x, self.p_y, self.p_td = QtWidgets.QLineEdit(""), QtWidgets.QLineEdit(""), QtWidgets.QLineEdit("")
        self.p_td.setPlaceholderText("TD TVDSS m")
        self.b_planned = QtWidgets.QPushButton("Planned vertical well")
        for w_ in (QtWidgets.QLabel("or planned: X"), self.p_x, QtWidgets.QLabel("Y"), self.p_y, self.p_td,
                   self.b_planned):
            prow.addWidget(w_)
        lay.addLayout(prow)
        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        lay.addWidget(self.table, 1)
        drow = QtWidgets.QHBoxLayout()
        self.bit = QtWidgets.QLineEdit("")
        self.bit.setPlaceholderText("bit MD m")
        self.b_bit = QtWidgets.QPushButton("Set bit depth")
        self.top_h = QtWidgets.QComboBox()
        self.top_md = QtWidgets.QLineEdit("")
        self.top_md.setPlaceholderText("top MD m")
        self.b_top = QtWidgets.QPushButton("Add drilled top")
        self.b_clear = QtWidgets.QPushButton("Clear drilled tops")
        self.b_csv = QtWidgets.QPushButton("Export CSV...")
        for w_ in (self.bit, self.b_bit, QtWidgets.QLabel("top"), self.top_h, self.top_md, self.b_top, self.b_clear,
                   self.b_csv):
            drow.addWidget(w_)
        lay.addLayout(drow)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.resize(900, 480)
        self._fill_wells()
        self.b_load.clicked.connect(lambda *_: self._safe(lambda: self.set_well(self.well.currentText())))
        self.b_planned.clicked.connect(lambda *_: self._safe(lambda: self.set_planned(
            float(self.p_x.text()), float(self.p_y.text()), float(self.p_td.text()))))
        self.b_bit.clicked.connect(lambda *_: self._safe(lambda: self.set_bit(float(self.bit.text()))))
        self.b_top.clicked.connect(lambda *_: self._safe(lambda: self.add_top(self.top_h.currentText(),
                                                                              float(self.top_md.text()))))
        self.b_clear.clicked.connect(lambda *_: self._safe(self.clear_tops))
        self.b_csv.clicked.connect(lambda *_: self._csv())

    def _safe(self, fn):
        try:
            return fn()
        except Exception as e:
            self._set_info(f"Prognosis: {e}")
            return None

    def _set_info(self, text):
        self._last_text = text
        self.info.setText(text)

    def _fill_wells(self):
        self._trajs = {}
        if self.project.well_project:
            from ..laterals import trajectories_from_wellset
            try:
                for t in trajectories_from_wellset(self.project.wellset()):
                    self._trajs[t.name] = t
            except Exception as e:
                self._set_info(f"Wells: {e}")
        self.well.clear()
        self.well.addItems(list(self._trajs))

    def _dm(self):
        dm = self.depth_model_fn()
        if dm is None:
            raise RuntimeError("no depth model - Interpret > Velocity model, then 'Use for depth display'")
        return dm

    def _start(self, traj):
        dm = self._dm()
        bl = (self.project.display or {}).get("ddc_blind") or {}
        if not self.uncertainty and bl.get("survey") == dm.vm.key:
            self.uncertainty = dict(bl.get("sigma") or {})
        self.prog = Prognosis(dm, traj, self.uncertainty)
        self.top_h.clear()
        self.top_h.addItems([h for h, _ in dm.vm.layers])
        self.prog.compute()
        self.refresh_table()
        return self.prog

    def set_well(self, name):
        return self._start(self._trajs[name])

    def set_planned(self, x, y, td_tvdss, kb=0.0, name="Planned"):
        return self._start(vertical_trajectory(name, x, y, td_tvdss, kb))

    def set_uncertainty(self, sigma):
        """{horizon: 1 sigma m} (e.g. from the blind test of item 17e)."""
        self.uncertainty = dict(sigma or {})
        if self.prog is not None:
            self.prog.uncertainty = self.uncertainty
            self.prog.compute()
            self.refresh_table()

    def refresh_table(self):
        rows = self.prog.rows if self.prog else []
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [r["horizon"] + (f" ({r['n'] + 1})" if r["n"] else ""), r["formation"], f"{r['md']:.1f}",
                    f"{r['tvdss']:.1f}", f"{r['x']:.0f}", f"{r['y']:.0f}", r["status"],
                    "" if r["sigma"] is None else f"{r['sigma']:.1f}",
                    "" if r["change"] is None else f"{r['change']:+.1f}"]
            for c, v in enumerate(vals):
                self.table.setItem(i, c, QtWidgets.QTableWidgetItem(str(v)))
        return rows

    def update(self, report=None):
        """The depth model changed (auto-update / drilled top): recompute the prognosis."""
        if self.prog is None:
            return []
        if self.prog.dm is not self.depth_model_fn():  # the model was replaced (dialog 'Use'): follow it
            self.prog.dm = self.depth_model_fn()
            if self.prog.dm is None:
                return []
        moved = self.prog.update()
        self.refresh_table()
        txt = (report or {}).get("message", "")
        if moved:
            txt += ("\n" if txt else "") + "Moved: " + ", ".join(f"{r['horizon']} {r['change']:+.1f} m" for r in moved)
        if txt:
            self._set_info(txt)
        return moved

    def set_bit(self, md):
        self.prog.set_bit(md)
        self.refresh_table()

    def add_top(self, key, md):
        rep = self.prog.add_top(key, md)
        self.refresh_table()
        moved = [r for r in self.prog.rows if r["change"] is not None and abs(r["change"]) > 0.01]
        self._set_info(rep["message"] + ("\nAhead of the bit: " + ", ".join(
            f"{r['horizon']} {r['tvdss']:.1f} m ({r['change']:+.1f})" for r in moved if r["status"] == "ahead")
            if moved else ""))
        if self.on_change:
            self.on_change()
        return rep

    def clear_tops(self):
        rep = self.prog.clear_tops()
        self.refresh_table()
        if self.on_change:
            self.on_change()
        return rep

    def write_csv(self, path):
        return self.prog.write_csv(path)

    def _csv(self):
        if self.prog is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Prognosis", self.project.base(), "CSV (*.csv)")
        if path:
            self.write_csv(path)
