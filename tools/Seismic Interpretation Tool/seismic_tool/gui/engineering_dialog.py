"""Engineering / production CSV import for the Analytics window: pick a preset (Enverus, IHS, Texas RRC, ...),
check / change the column mapping and units, then build completion or production features (AttributeTable
keyed by API / well name).  The logic lives in plain attributes (`rows`, `headers`, `preset`, `kind`, `mapping`,
`units`) so it can be tested without a display."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtWidgets  # noqa: E402

from ..analytics import engineering as E  # noqa: E402

COMPLETION_KEYS = ["api", "well", "x", "y", "completion_date", "first_prod", "lateral_length", "stages", "clusters",
                   "proppant", "fluid", "landing_zone"]
PRODUCTION_KEYS = ["api", "well", "month", "oil", "gas", "water", "days"]
UNIT_CHOICES = {"lateral_length": ["ft", "m"], "proppant": ["lb", "t"], "fluid": ["bbl", "m3", "gal"]}
NONE = "(none)"


class EngineeringImportDialog(QtWidgets.QDialog):
    def __init__(self, path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import engineering / production CSV")
        self.rows, self.headers, self.path = [], [], None
        self.preset, self.kind = "Auto", "completions"
        self.mapping, self.units = {}, {}
        self.lateral_ft = None  # {normalised id: ft} from a completions table, for per-1000-ft production features
        self.table = None
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.b_file = QtWidgets.QPushButton("CSV file...")
        self.preset_box = QtWidgets.QComboBox()
        self.preset_box.addItems(["Auto"] + list(E.PRESETS))
        self.kind_box = QtWidgets.QComboBox()
        self.kind_box.addItems(["completions", "production (monthly)"])
        for w in (self.b_file, QtWidgets.QLabel("Preset"), self.preset_box, QtWidgets.QLabel("Data"), self.kind_box):
            top.addWidget(w)
        lay.addLayout(top)
        self.form = QtWidgets.QFormLayout()
        lay.addLayout(self.form)
        self.boxes, self.unit_boxes = {}, {}
        row = QtWidgets.QHBoxLayout()
        self.b_save = QtWidgets.QPushButton("Save mapping...")
        self.b_load = QtWidgets.QPushButton("Load mapping...")
        self.b_ok = QtWidgets.QPushButton("Import")
        self.b_cancel = QtWidgets.QPushButton("Cancel")
        for w in (self.b_save, self.b_load, self.b_ok, self.b_cancel):
            row.addWidget(w)
        lay.addLayout(row)
        self.info = QtWidgets.QLabel("")
        lay.addWidget(self.info)
        self.b_file.clicked.connect(self._file_dialog)
        self.preset_box.currentTextChanged.connect(lambda s: self.set_preset(s))
        self.kind_box.currentTextChanged.connect(lambda s: self.set_kind(s.split()[0]))
        self.b_save.clicked.connect(self._save_dialog)
        self.b_load.clicked.connect(self._load_dialog)
        self.b_ok.clicked.connect(self._ok)
        self.b_cancel.clicked.connect(self.reject)
        if path:
            self.load(path)

    # ------------------------------------------------------------ logic
    def keys(self):
        return PRODUCTION_KEYS if self.kind == "production" else COMPLETION_KEYS

    def load(self, path):
        self.path = path
        self.rows = E.read_csv_rows(path)
        self.headers = list(self.rows[0].keys()) if self.rows else []
        mp = E.auto_mapping(self.headers)
        self.kind = "production" if "month" in mp and ("oil" in mp or "gas" in mp) and "proppant" not in mp \
            else "completions"
        self._remap()
        return self

    def set_preset(self, preset):
        self.preset = preset
        self._remap()

    def set_kind(self, kind):
        self.kind = kind
        self._remap()

    def _remap(self):
        mp = E.auto_mapping(self.headers, None if self.preset == "Auto" else self.preset)
        self.mapping = {k: v for k, v in mp.items() if k in self.keys()}
        self.units = {k: E.detect_units(self.mapping[k], None) for k in UNIT_CHOICES if k in self.mapping}
        self._refresh_form()

    def set_column(self, key, header):
        if header in (None, "", NONE):
            self.mapping.pop(key, None)
        else:
            if header not in self.headers:
                raise ValueError(f"no column {header!r} in the file")
            self.mapping[key] = header

    def set_unit(self, key, unit):
        self.units[key] = unit

    def problems(self):
        out = []
        if "api" not in self.mapping and "well" not in self.mapping:
            out.append("map an API or well-name column")
        if self.kind == "production" and "month" not in self.mapping:
            out.append("map the production month / date column")
        if self.kind == "production" and not any(k in self.mapping for k in ("oil", "gas", "water")):
            out.append("map at least one of oil / gas / water")
        return out

    def build(self):
        p = self.problems()
        if p:
            raise ValueError("; ".join(p))
        if self.kind == "production":
            ser = E.monthly_series(self.rows, self.mapping)
            lat = {k: self.lateral_ft.get(E.normalise_id(k), float("nan")) for k in ser} if self.lateral_ft else None
            t = E.production_features(ser, lat)
        else:
            t = E.qc_flags(E.completion_table(self.rows, self.mapping, units=self.units))
        t.meta["source"] = os.path.basename(self.path or "")
        self.table = t
        return t

    def summary(self):
        t = self.table
        if t is None:
            return ""
        n_ok = {n: int((v == v).sum()) for n, v in t.cols.items()}
        flagged = int(t.cols["qc_flag"].sum()) if "qc_flag" in t.cols else 0
        return (f"{len(t)} wells, {len(t.names)} columns" + (f", {flagged} flagged (units / outliers)" if flagged else "")
                + "\n" + ", ".join(f"{n} ({c})" for n, c in n_ok.items()))

    # ------------------------------------------------------------ widgets
    def _refresh_form(self):
        try:
            while self.form.rowCount():
                self.form.removeRow(0)
        except Exception:
            pass
        self.boxes, self.unit_boxes = {}, {}
        for k in self.keys():
            cb = QtWidgets.QComboBox()
            cb.addItems([NONE] + self.headers)
            cb.setCurrentText(self.mapping.get(k, NONE))
            cb.currentTextChanged.connect(lambda s, k=k: self.set_column(k, s))
            row = QtWidgets.QHBoxLayout()
            row.addWidget(cb)
            if k in UNIT_CHOICES:
                ub = QtWidgets.QComboBox()
                ub.addItems(UNIT_CHOICES[k])
                ub.setCurrentText(self.units.get(k) or UNIT_CHOICES[k][0])
                ub.currentTextChanged.connect(lambda s, k=k: self.set_unit(k, s))
                row.addWidget(ub)
                self.unit_boxes[k] = ub
            self.form.addRow(k.replace("_", " "), row)
            self.boxes[k] = cb
        self.info.setText(f"{len(self.rows)} rows, {len(self.headers)} columns; " +
                          ("; ".join(self.problems()) or "mapping complete"))

    def _file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Engineering / production CSV", "", "CSV (*.csv *.txt)")
        if path:
            self.load(path)

    def _save_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save column mapping", "", "JSON (*.json)")
        if path:
            E.save_mapping({"kind": self.kind, "mapping": self.mapping, "units": self.units}, path)

    def _load_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load column mapping", "", "JSON (*.json)")
        if path:
            self.apply_saved(E.load_mapping(path))

    def apply_saved(self, d):
        self.kind = d.get("kind", self.kind)
        self.mapping = {k: v for k, v in d.get("mapping", {}).items() if v in self.headers}
        self.units.update(d.get("units", {}))
        self._refresh_form()

    def _ok(self):
        try:
            self.build()
        except ValueError as e:
            self.info.setText(str(e))
            return
        self.accept()
