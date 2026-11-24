"""Map hub dialogs (PLAN 21c): grid a zone property (IDW / TPS / kriging / KED / collocated cokriging with a grid layer
as the secondary, correlation + leave-one-out table, residual map) and the grid calculator (expressions on grid layers,
mask by polygon).  Logic in compute / cross_validate / add_layer / evaluate / mask (testable under the Qt mock)."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtWidgets  # noqa: E402
import numpy as np  # noqa: E402

from .. import propgrid as PG  # noqa: E402
from ..maplayers import petro_values  # noqa: E402

PROPS = ["SoPhiH", "NetPay_TVD", "PHIH_pay", "PHIE_res", "Sw_pay", "Vsh_avg", "NtG_res", "kh_pay", "Gross_TVD", "TOC_avg"]


def grid_layers(stack):
    return [(l["id"], l["name"]) for l in stack.ordered() if l["kind"] in ("grid", "horizon")]


class PropGridDialog(QtWidgets.QDialog):
    def __init__(self, stack, panel=None, parent=None):
        super().__init__(parent)
        self.stack, self.panel = stack, panel
        self.setWindowTitle("Grid a zone property")
        self.resize(620, 520)
        self.grid = None
        self.points = None
        self.corr = None
        self.cv = None
        self.params = {}
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.prop = QtWidgets.QComboBox()
        self.prop.addItems(PROPS)
        self.zone = QtWidgets.QLineEdit()
        self.zone.setPlaceholderText("empty = all zones")
        self.method = QtWidgets.QComboBox()
        for k, v in PG.METHODS.items():
            self.method.addItem(v, k)
        self.sec = QtWidgets.QComboBox()
        self.sec.addItem("(none)", None)
        for lid, name in grid_layers(stack):
            self.sec.addItem(name, lid)
        self.cell = QtWidgets.QDoubleSpinBox()
        self.cell.setRange(0, 1e6)
        self.cell.setSpecialValueText("auto")
        for lab, w in (("Value", self.prop), ("Zone", self.zone), ("Method", self.method),
                       ("Secondary (KED / cokriging)", self.sec), ("Cell (XY grid)", self.cell)):
            form.addRow(lab, w)
        lay.addLayout(form)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("Compute", lambda *_: self._run(self.compute)), ("Cross-validate", lambda *_: self._run(self.cross_validate)),
                         ("Add as layer", lambda *_: self._run(self.add_layer)),
                         ("Add residual map", lambda *_: self._run(self.add_residual_layer))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        self.table = QtWidgets.QTableWidget()
        lay.addWidget(self.table)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)

    def _run(self, fn):
        try:
            fn()
        except Exception as e:
            self.info.setText(f"{type(e).__name__}: {e}")

    def _ui_params(self):
        return {"prop": self.prop.currentText() or "SoPhiH", "zone": self.zone.text().strip() or None,
                "method": self.method.currentData() or "idw", "secondary": self.sec.currentData(),
                "cell": float(self.cell.value()) or None}

    def compute(self, prop=None, zone=None, method=None, secondary=None, cell=None):
        """Grid the zone value; returns the MapGrid (also kept as self.grid)."""
        if prop is None:  # from the form
            ui = self._ui_params()
            prop, zone, method, secondary, cell = ui["prop"], ui["zone"], ui["method"], ui["secondary"], ui["cell"]
        method = method or "idw"
        vals = petro_values(self.stack.project, prop, zone)
        if len(vals) < 2:
            raise ValueError(f"{len(vals)} wells with {prop} - run the Petrophysics window in the Well Correlation Tool")
        names = [q[0] for q in vals]
        px, py, v = (np.array([q[i] for q in vals], float) for i in (1, 2, 3))
        sg = self.stack.grid(secondary) if secondary else None
        self.points = (names, px, py, v)
        self.params = {"prop": prop, "zone": zone, "method": method, "secondary": secondary, "cell": cell}
        self.corr = PG.correlation(px, py, v, sg)[0] if sg is not None else None
        label = f"{prop} {zone or 'all zones'} ({method})"
        self.grid = PG.grid_property(px, py, v, method, sg, cell, label=label)
        self.info.setText(f"{len(v)} wells; " + (f"r with the secondary = {self.corr:.2f}" if self.corr is not None else
                                                "no secondary"))
        return self.grid

    def cross_validate(self, methods=None):
        if self.points is None:
            self.compute()
        names, px, py, v = self.points
        sg = self.stack.grid(self.params["secondary"]) if self.params.get("secondary") else None
        methods = methods or (("idw", "tps", "kriging", "ked", "cck") if sg is not None else ("idw", "tps", "kriging"))
        self.cv = PG.cross_validate(px, py, v, methods, sg)
        rows = [{"method": m, "rmse": c["rmse"], "mae": c["mae"], "bias": c["bias"], "n": c["n"]} for m, c in self.cv.items()]
        t = self.table
        t.clear()
        cols = ["method", "rmse", "mae", "bias", "n"]
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                val = r[c]
                t.setItem(i, j, QtWidgets.QTableWidgetItem(f"{val:.4g}" if isinstance(val, float) else str(val)))
        return rows

    def _after_add(self, msg):
        if self.panel is not None:
            self.panel._apply(msg)
        else:
            self.stack.save()

    def add_layer(self):
        if self.grid is None:
            self.compute()
        lid, _ = self.stack.add_grid(self.grid, self.grid.label)
        self.stack.get(lid)["src"]["recipe"] = dict(self.params)  # re-gridded by "Add new wells" (21d)
        self._after_add(f"added {self.grid.label}")
        return lid

    def add_residual_layer(self, method=None):
        method = method or self.params.get("method", "idw")
        if self.cv is None or method not in self.cv:
            self.cross_validate((method,))
        if self.grid is None:
            self.compute()
        names, px, py, v = self.points
        rm = PG.residual_map(px, py, self.cv[method]["resid"], self.grid)
        rm.label = f"LOO residual {self.params['prop']} ({method})"
        lid, _ = self.stack.add_grid(rm, rm.label, style={"lut": "Red-white-black"})
        self._after_add(rm.label)
        return lid


class GridCalcDialog(QtWidgets.QDialog):
    def __init__(self, stack, panel=None, parent=None):
        super().__init__(parent)
        self.stack, self.panel = stack, panel
        self.setWindowTitle("Grid calculator")
        self.resize(520, 380)
        lay = QtWidgets.QVBoxLayout(self)
        self.legend = QtWidgets.QLabel("")
        self.legend.setWordWrap(True)
        lay.addWidget(self.legend)
        self.expr = QtWidgets.QLineEdit()
        self.expr.setPlaceholderText("e.g. A - B,  clip(A * 0.3048, 0, 500),  max(A, B)")
        lay.addWidget(self.expr)
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("result layer name")
        lay.addWidget(self.name)
        b = QtWidgets.QPushButton("Evaluate and add layer")
        b.clicked.connect(lambda *_: self._safe(lambda: self.evaluate(self.expr.text(), self.name.text() or None)))
        lay.addWidget(b)
        self.info = QtWidgets.QLabel("")
        lay.addWidget(self.info)
        self.refresh()

    def _safe(self, fn):
        try:
            fn()
        except Exception as e:
            self.info.setText(f"{type(e).__name__}: {e}")

    def letters(self):
        """{letter: layer id} for the grid layers (A, B, C ... in list order)."""
        out = {}
        for k, (lid, _) in enumerate(grid_layers(self.stack)):
            out[chr(ord("A") + k) if k < 26 else f"G{k}"] = lid
        return out

    def refresh(self):
        names = dict(grid_layers(self.stack))
        self.legend.setText("Grids: " + ";  ".join(f"{k} = {names[v]}" for k, v in self.letters().items()))

    def evaluate(self, expr, name=None):
        grids = {k: self.stack.grid(v) for k, v in self.letters().items()}
        grids = {k: g for k, g in grids.items() if g is not None}
        g = PG.calc(expr, grids)
        g.label = name or expr
        lid, _ = self.stack.add_grid(g, g.label)
        if self.panel is not None:
            self.panel._apply(f"added {g.label}")
        self.info.setText(f"added {g.label}")
        self.refresh()
        return lid

    def mask(self, lid, px, py, keep="inside", name=None):
        g = PG.mask_polygon(self.stack.grid(lid), px, py, keep)
        g.label = name or g.label
        nid, _ = self.stack.add_grid(g, g.label)
        if self.panel is not None:
            self.panel._apply(f"added {g.label}")
        return nid


class CrossCheckDialog(QtWidgets.QDialog):
    """PLAN 21e: zone value vs grid layers at the wells - cross-plot, regression, residuals, predicted maps with an 80 %
    interval, and 'Send to Analytics' (item-13 models)."""

    def __init__(self, stack, panel=None, parent=None):
        super().__init__(parent)
        import pyqtgraph as pg
        self.stack, self.panel = stack, panel
        self.cc = None
        self.setWindowTitle("Cross-check petrophysics vs maps")
        self.resize(760, 640)
        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.prop = QtWidgets.QComboBox()
        self.prop.addItems(PROPS)
        self.zone = QtWidgets.QLineEdit()
        self.zone.setPlaceholderText("zone (empty = all)")
        self.grids = QtWidgets.QListWidget()
        self.grids.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self._lids = []
        for lid, name in grid_layers(stack):
            self.grids.addItem(name)
            self._lids.append(lid)
        for w in (self.prop, self.zone):
            row.addWidget(w)
        lay.addLayout(row)
        lay.addWidget(QtWidgets.QLabel("Grids (first = x axis of the plot and geometry of the maps):"))
        lay.addWidget(self.grids, 1)
        row2 = QtWidgets.QHBoxLayout()
        for text, fn in (("Compute", lambda *_: self._safe(self.compute)), ("Add predicted maps", lambda *_: self._safe(self.add_maps)),
                         ("Send to Analytics", lambda *_: self._safe(self.send_to_analytics))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row2.addWidget(b)
        lay.addLayout(row2)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        lay.addWidget(self.plot, 3)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)

    def _safe(self, fn):
        try:
            fn()
        except Exception as e:
            self.info.setText(f"{type(e).__name__}: {e}")

    def compute(self, prop=None, zone=None, lids=None):
        from .. import crosscheck as CC
        if prop is None:
            prop = self.prop.currentText() or "PHIE_res"
            zone = self.zone.text().strip() or None
            lids = [self._lids[i.row()] for i in self.grids.selectedIndexes()] if lids is None else lids
        if not lids:
            raise ValueError("select at least one grid layer")
        self.cc = CC.crosscheck(self.stack, prop, zone, lids)
        self._draw()
        c = self.cc
        self.info.setText(f"{c['n']} wells; r = {c['r']:.2f}, residual sigma {c['sigma']:.3g}, leave-one-out RMSE "
                          f"{c['loo_rmse']:.3g}" if c.get("coef") is not None else f"{c['n']} wells: too few for a fit")
        return self.cc

    def _draw(self):
        import pyqtgraph as pg
        c = self.cc
        self.plot.clear()
        if c is None or not c["X"].shape[1]:
            return
        ok = c["ok"]
        x, y = c["X"][ok, 0], c["y"][ok]
        self.plot.addItem(pg.ScatterPlotItem(x, y, size=9, brush=pg.mkBrush(70, 120, 180)))
        if c.get("coef") is not None and len(c["coef"]) == 2 and len(x):
            xs = np.array([x.min(), x.max()])
            self.plot.plot(xs, c["coef"][0] + c["coef"][1] * xs, pen=pg.mkPen("r", width=2))
        self.plot.setLabel("bottom", c["labels"][0])
        self.plot.setLabel("left", c["prop"])

    def add_maps(self, which=("P50", "P90", "P10", "residual", "corrected")):
        from .. import crosscheck as CC
        if self.cc is None:
            self.compute()
        maps = CC.predicted_maps(self.stack, self.cc)
        if maps is None:
            raise ValueError("no regression (too few wells)")
        ids = {}
        for k in which:
            g = maps[k]
            ids[k] = self.stack.add_grid(g, g.label, style={"lut": "Red-white-black"} if k == "residual" else None,
                                         contours=k == "P50")[0]
        if self.panel is not None:
            self.panel._apply(f"added {len(ids)} predicted maps")
        return ids

    def send_to_analytics(self):
        from .. import crosscheck as CC
        if self.cc is None:
            self.compute()
        t = CC.to_attribute_table(self.stack, self.cc["prop"], self.cc["zone"], self.cc["grid_lids"])
        hook = getattr(self.panel, "analytics_hook", None) if self.panel is not None else None
        if callable(hook):
            hook(t)
            self.info.setText(f"sent {len(t)} wells x {len(t.names)} columns to Analytics")
        return t
