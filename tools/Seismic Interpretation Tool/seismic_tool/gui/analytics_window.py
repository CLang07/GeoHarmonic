"""Analytics window: build an attribute table at the wells (map grids, zone statistics, CSV targets /
engineering data), explore, train + validate honestly, explain (SHAP), push predictions back as maps."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtWidgets  # noqa: E402
import numpy as np  # noqa: E402

from ..analytics import explain as XP  # noqa: E402
from ..analytics import models as MD  # noqa: E402
from ..analytics import prep  # noqa: E402
from ..analytics import qc as QC  # noqa: E402
from ..analytics import validation as V  # noqa: E402
from ..analytics.table import AttributeTable  # noqa: E402

MODELS = ["gbm", "rf", "linear", "ridge", "lasso", "knn", "mlp"]
CVS = ["leave one well out", "group 5-fold", "spatial blocks (2 km)", "random 5-fold (optimistic)"]


class AnalyticsWindow(QtWidgets.QDialog):
    def __init__(self, project, basemap=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analytics")
        self.project, self.basemap = project, basemap
        self.grids = {}  # input maps name -> MapGrid
        self.extra = []  # AttributeTables joined by well id (zone stats, CSV)
        self.table = None
        self.result = None
        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.b_grid = QtWidgets.QPushButton("Add basemap grid as input")
        self.b_csv = QtWidgets.QPushButton("Load well CSV (targets / engineering)...")
        self.zone = QtWidgets.QComboBox()
        self.b_zone = QtWidgets.QPushButton("Add zone statistics")
        self.b_eng = QtWidgets.QPushButton("Import engineering / production CSV...")
        for w in (self.b_grid, self.b_csv, self.b_eng, self.zone, self.b_zone):
            row.addWidget(w)
        lay.addLayout(row)
        form = QtWidgets.QFormLayout()
        self.inputs = QtWidgets.QListWidget()
        self.inputs.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.target = QtWidgets.QComboBox()
        self.model = QtWidgets.QComboBox()
        self.model.addItems(MODELS)
        self.cv = QtWidgets.QComboBox()
        self.cv.addItems(CVS)
        form.addRow("Inputs", self.inputs)
        form.addRow("Target", self.target)
        form.addRow("Model", self.model)
        form.addRow("Validation", self.cv)
        lay.addLayout(form)
        row = QtWidgets.QHBoxLayout()
        self.b_explore = QtWidgets.QPushButton("Explore")
        self.b_train = QtWidgets.QPushButton("Train + validate")
        self.b_map = QtWidgets.QPushButton("Predict map")
        self.b_report = QtWidgets.QPushButton("HTML report...")
        self.b_qc = QtWidgets.QPushButton("Model QC...")
        for w in (self.b_explore, self.b_train, self.b_map, self.b_report, self.b_qc):
            row.addWidget(w)
        lay.addLayout(row)
        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        lay.addWidget(self.out, 1)
        self.resize(820, 700)
        ws = project.wellset() if project.well_project else None
        if ws is not None and ws.wcp is not None:
            self.zone.addItems([f["name"] for f in ws.wcp.formations])
        self.b_grid.clicked.connect(self._add_basemap_grid)
        self.b_csv.clicked.connect(self._load_csv_dialog)
        self.b_eng.clicked.connect(self._engineering_dialog)
        self.b_zone.clicked.connect(lambda: self.add_zone(self.zone.currentText()))
        self.b_explore.clicked.connect(lambda: self._show(self.explore()))
        self.b_train.clicked.connect(self._train_clicked)
        self.b_map.clicked.connect(lambda: self.predict_map())
        self.b_report.clicked.connect(self._report_dialog)
        self.b_qc.clicked.connect(self._qc_dialog)

    # ------------------------------------------------------------ data
    def wells(self):
        ws = self.project.wellset()
        return [(n, ws.well(n).x, ws.well(n).y) for n in ws.names if ws.well(n).x is not None]

    def add_grid(self, name, grid):
        self.grids[name] = grid
        self.build_table()

    def _add_basemap_grid(self):
        g = getattr(self.basemap, "_grid", (None, None))[0] if self.basemap is not None else None
        if g is None:
            self._show("Show a map on the basemap first (Interpret > Horizon maps).")
            return
        self.add_grid(g.label or f"map{len(self.grids) + 1}", g)

    def add_csv(self, path, id_col=None):
        self.extra.append(AttributeTable.from_csv(path, id_col))
        self.build_table()

    def add_table(self, t):
        """Join an AttributeTable (engineering / production features, ...) by API / well name."""
        self.extra.append(t)
        self.build_table()
        return self.table

    def lateral_lengths(self):
        """{normalised id: lateral ft} from any imported completions table (for per-1000-ft production)."""
        from ..analytics.table import normalise_id
        out = {}
        for e in self.extra:
            if "lateral_ft" in e.cols:
                out.update({normalise_id(i): float(v) for i, v in zip(e.ids, e.cols["lateral_ft"])})
        return out

    def _engineering_dialog(self):
        from .engineering_dialog import EngineeringImportDialog
        dlg = EngineeringImportDialog(parent=self)
        dlg.lateral_ft = self.lateral_lengths() or None
        if dlg.exec() and dlg.table is not None:
            self.add_table(dlg.table)
            self._show(dlg.summary())

    def _load_csv_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Well CSV", self.project.base(), "CSV (*.csv)")
        if path:
            self.add_csv(path)

    def add_zone(self, zone):
        if not zone:
            return
        ws = self.project.wellset()
        t = AttributeTable.from_zone_stats(ws.wcp, zone)
        t.cols = {f"{zone}:{k}": v for k, v in t.cols.items()}
        self.extra.append(t)
        self.build_table()

    def build_table(self):
        t = AttributeTable.from_maps_at_wells(self.wells(), self.grids) if self.project.well_project else AttributeTable()
        for e in self.extra:
            if len(t) == 0:  # no wells from the well project: the first imported table defines the rows
                t = AttributeTable(e.ids)
                t.xy = e.xy
            t = t.join(e)
        self.table = t
        self.inputs.clear()
        self.target.clear()
        for n in t.names:
            self.inputs.addItem(n)
            self.target.addItem(n)
        return t

    # ------------------------------------------------------------ analysis
    def explore(self, names=None):
        t = self.table or self.build_table()
        names = names or t.names
        X = t.matrix(names)
        lines = ["Missing values: " + ", ".join(f"{n} {m} ({p:.0f}%)" for n, m, p in prep.missing_report(X, names))]
        if X.shape[1] > 1 and len(X) >= 3:
            C = prep.correlation(prep.impute(X))
            lines.append("Correlation:\n" + "\n".join(f"{n[:18]:18s} " + " ".join(f"{c:+.2f}" for c in row)
                                                      for n, row in zip(names, C)))
            v = prep.vif(prep.impute(X))
            lines.append("VIF: " + ", ".join(f"{n} {x:.1f}" for n, x in zip(names, v)) + "  (> 10: strongly collinear)")
        return "\n".join(lines)

    def _splits(self, kind, n, groups, xy):
        if kind.startswith("leave"):
            return list(V.leave_one_group_out(groups))
        if kind.startswith("group"):
            return list(V.group_kfold(groups, min(5, len(np.unique(groups)))))
        if kind.startswith("spatial"):
            return list(V.spatial_blocks(xy, 2000.0, min(5, max(2, n // 3))))
        return list(V.kfold(n, min(5, n)))

    def train(self, inputs, target, model_name="gbm", cv=CVS[0]):
        t = self.table or self.build_table()
        y = t.cols[target]
        ok = np.isfinite(y)
        X = prep.impute(t.matrix(inputs)[ok], "median")
        y = y[ok]
        ids = np.array(t.ids)[ok]
        xy = t.xy[ok] if t.xy is not None else np.zeros((len(y), 2))
        n, p = X.shape
        warn = ""
        if n < max(10, 5 * p):
            warn = (f"WARNING: only {n} wells for {p} inputs - results are fragile (rule of thumb: >= 5-10 wells per "
                    "input). Prefer few inputs, simple models and leave-one-well-out validation.")
        kw = {"gbm": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 2.0},
              "rf": {"n_estimators": 100, "max_depth": 6}, "lasso": {"alpha": 0.05}, "knn": {"k": min(5, n - 1)},
              "mlp": {"hidden": 8, "epochs": 1500}}.get(model_name, {})
        model = MD.make_model(model_name, **kw)
        splits = self._splits(cv, n, ids, xy)
        pred = V.cross_val_predict(model, X, y, splits)
        fold_models = [model.clone().fit(X[tr], y[tr]) for tr, _ in splits[:10]]
        base = V.cross_val_predict(MD.LinearModel(), np.zeros((n, 1)), y, splits)  # mean baseline
        final = model.clone().fit(X, y)
        res = {"model": final, "fold_models": fold_models, "inputs": list(inputs), "target": target, "X": X, "y": y,
               "ids": ids, "xy": xy, "splits": splits, "cv_pred": pred, "r2": V.r2(y, pred), "rmse": V.rmse(y, pred),
               "baseline_r2": V.r2(y, base), "cv": cv, "warning": warn, "model_name": model_name}
        if model_name in ("gbm", "rf"):
            res["shap"], res["shap_ev"] = XP.tree_shap(final, X)
        self.result = res
        return res

    def _train_clicked(self):
        inputs = [it.text() for it in self.inputs.selectedItems()]
        if not inputs or not self.target.currentText():
            self._show("Select inputs and a target.")
            return
        r = self.train(inputs, self.target.currentText(), self.model.currentText(), self.cv.currentText())
        self._show(f"{r['model_name']} / {r['cv']}: CV R2 {r['r2']:.2f}, RMSE {r['rmse']:.3g} "
                   f"(mean baseline R2 {r['baseline_r2']:.2f})\n{r['warning']}")

    def predict_map(self):
        r = self.result
        if r is None or not self.grids:
            return None
        names = r["inputs"]
        if not all(n in self.grids for n in names):
            self._show("Map prediction needs every input to be a map grid.")
            return None
        ref = self.grids[names[0]]
        nodes = AttributeTable.from_map_nodes({n: self.grids[n] for n in names}, ref)
        Xn = nodes.matrix(names)
        good = np.all(np.isfinite(Xn), 1)
        mean, sd = XP.ensemble_map(r["fold_models"], np.where(good[:, None], Xn, 0.0))
        shape = nodes.meta["grid_shape"]
        from ..maps import MapGrid
        z = np.where(good, mean, np.nan).reshape(shape)
        u = np.where(good, sd, np.nan).reshape(shape)
        g = MapGrid(z, ref.kind, "", f"Predicted {r['target']} ({r['model_name']})", ref.vol, ref.x0, ref.y0, ref.dx, ref.dy)
        g.uncertainty = MapGrid(u, ref.kind, "", f"Uncertainty (CV spread) {r['target']}", ref.vol, ref.x0, ref.y0,
                                ref.dx, ref.dy)
        # applicability domain: cells whose inputs lie outside the training wells' input space
        inside = np.zeros(len(Xn), bool)
        if good.any() and r["X"].shape[0] > r["X"].shape[1] + 1:
            inside[good], _, _ = QC.applicability(r["X"], Xn[good])
        else:
            inside[good] = True
        g.inside = inside.reshape(shape)
        g.outside_fraction = float(1 - inside[good].mean()) if good.any() else 0.0
        if self.basemap is not None:
            self.basemap.show_grid(g)
        return g

    def qc(self, path=None):
        """Model QC on the last trained model: baselines, y-randomisation, noise feature, residual spatial
        structure, conformal interval, importance stability.  Each check is skipped (n/a) when there are too few
        wells.  Returns (lines, path of the HTML report or None)."""
        r = self.result
        if r is None:
            return ["Train a model first."], None
        X, y, ids, xy = r["X"], r["y"], r["ids"], r["xy"]
        model = r["model"].clone()
        n = len(y)
        lines, items = [], []

        def step(title, fn):
            try:
                txt = fn()
            except Exception as e:  # too few wells, singular data...
                txt = f"n/a ({type(e).__name__}: {e})"
            lines.append(f"{title}: {txt}")
            items.append((title, txt))

        lines.append(f"{r['model_name']} / {r['cv']}: CV R2 {r['r2']:.2f} on {n} wells")
        k = min(5, len(np.unique(ids)))
        if k < 3:
            lines.append("Too few wells for model QC (need >= 3).")
            return lines, None

        def base():
            b = QC.baselines(X, y, xy, ids, r["inputs"])
            verdict = "beats all" if r["r2"] > max(b.values()) else "DOES NOT beat every baseline"
            return ", ".join(f"{a} {v:.2f}" for a, v in b.items()) + f" -> model {verdict}"
        step("Baselines (CV R2)", base)

        def yrand():
            v = QC.y_randomization(model, X, y, ids, 3)
            return f"shuffled-target R2 max {v.max():.2f}" + (" (OK)" if v.max() < 0.2 else " - LEAKAGE / OVERFIT?")
        step("Y-randomisation", yrand)

        def noise():
            imp, kn = QC.noise_feature_test(model, X, y)
            weak = [nm for nm, v in zip(r["inputs"], imp[:kn]) if v <= imp[kn]]
            return "all inputs out-rank a random column" if not weak else "below random noise: " + ", ".join(weak)
        step("Noise feature", noise)

        def moran():
            I, z = QC.morans_i(xy, y - r["cv_pred"])
            return f"Moran's I {I:.2f} (z {z:.1f})" + (" - spatial structure left: missing input?" if z > 2 else " (OK)")
        step("Residual spatial structure", moran)

        def resid():
            d = QC.residual_diagnostics(y, r["cv_pred"])
            return f"QQ corr {d['qq_corr']:.2f}, residual vs prediction corr {d['resid_vs_pred_corr']:+.2f}"
        step("Residuals", resid)

        def conf():
            lo, hi, q = QC.conformal_intervals(model, X, y, X, 0.2, ids)
            return f"80% split-conformal interval = prediction +/- {q:.3g}"
        step("Uncertainty", conf)

        def stab():
            rho, _ = QC.importance_stability(model, X, y, ids, 4)
            return f"importance rank correlation between bootstraps {rho:.2f}"
        if X.shape[1] > 1:
            step("Importance stability", stab)
        out = None
        if path:
            out = QC.qc_report(path, f"Model QC: {r['target']}", [("Summary", "<br>".join(lines))] + items)
        return lines, out

    def _qc_dialog(self):
        if self.result is None:
            self._show("Train a model first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Model QC report", self.project.base(), "HTML (*.html)")
        lines, _ = self.qc(path or None)
        self._show("\n".join(lines))

    def report(self, path):
        r = self.result
        sections = [("Run", XP.table_html([(r["model_name"], r["cv"], r["r2"], r["rmse"], r["baseline_r2"], len(r["y"]))],
                                          ["model", "validation", "CV R2", "RMSE", "baseline R2", "wells"])),
                    ("Inputs", ", ".join(r["inputs"]) + f" -> {r['target']}<br>{r['warning']}")]
        if "shap" in r:
            sections.append(("SHAP summary", ("png", XP.plot_shap_summary(r["shap"], r["X"], r["inputs"]))))
            sections.append(("Mean |SHAP|", ("png", XP.plot_bar(np.abs(r["shap"]).mean(0), r["inputs"], "importance"))))
        return XP.html_report(path, f"Analytics: {r['target']}", sections)

    def _report_dialog(self):
        if self.result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "HTML report", self.project.base(), "HTML (*.html)")
        if path:
            self.report(path)

    def _show(self, text):
        self.out.setPlainText(str(text))
