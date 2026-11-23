"""Velocity model window: layer-cake from horizons + well tops, depth maps, misfit table, depth display."""
import csv
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtWidgets  # noqa: E402
import numpy as np  # noqa: E402

from ..velocity import FUNCTIONS, VelocityModel  # noqa: E402


class VelocityDialog(QtWidgets.QDialog):
    def __init__(self, project, basemap=None, parent=None, on_change=None):
        super().__init__(parent)
        self.setWindowTitle("Velocity model and depth conversion")
        self.project, self.basemap, self.on_change = project, basemap, on_change
        self.model = None
        self.ddm = None  # ddc.DynamicDepthModel around self.model
        self._shown_layer = None
        self._info_text = ""
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.survey = QtWidgets.QComboBox()
        self.survey.addItems([f"3D:{v['name']}" for v in project.volumes] + (["2D"] if project.lines else []))
        self.resid = QtWidgets.QCheckBox("Correct the remaining misfit at the wells (grid residuals)")
        form.addRow("Survey", self.survey)
        form.addRow(self.resid)
        grow = QtWidgets.QHBoxLayout()
        self.g_method = QtWidgets.QComboBox()
        for key_, label in (("idw", "Inverse distance"), ("tps", "Thin-plate spline"), ("kriging", "Kriging")):
            self.g_method.addItem(label, key_)
        self.g_vario = QtWidgets.QComboBox()
        self.g_vario.addItems(["spherical", "exponential"])
        self.g_range = QtWidgets.QLineEdit("")
        self.g_range.setPlaceholderText("range m (blank = fit)")
        self.g_kind = QtWidgets.QComboBox()
        self.g_kind.addItem("RMS / stacking velocity (Dix)", "rms")
        self.g_kind.addItem("interval velocity", "interval")
        self.g_mode = QtWidgets.QComboBox()
        self.g_mode.addItem("scale by ratio at the wells", "ratio")
        self.g_mode.addItem("shift by difference at the wells", "difference")
        for w_ in (self.g_method, QtWidgets.QLabel("variogram"), self.g_vario, self.g_range):
            grow.addWidget(w_)
        form.addRow("Gridding of well values", grow)
        hrow = QtWidgets.QHBoxLayout()
        hrow.addWidget(self.g_kind)
        hrow.addWidget(self.g_mode)
        form.addRow("Seismic velocity guide volume is", hrow)
        # plain state (the Qt mock's widgets hold no values)
        self._gridding = {"method": "idw"}
        self._guide_opts = {"kind": "rms", "mode": "ratio"}
        self.g_method.currentIndexChanged.connect(lambda *_: self._grid_widgets_changed())
        self.g_vario.currentIndexChanged.connect(lambda *_: self._grid_widgets_changed())
        self.g_range.textChanged.connect(lambda *_: self._grid_widgets_changed())
        self.g_kind.currentIndexChanged.connect(lambda *_: self._grid_widgets_changed())
        self.g_mode.currentIndexChanged.connect(lambda *_: self._grid_widgets_changed())
        lay.addLayout(form)
        lay.addWidget(QtWidgets.QLabel("Layers (top to bottom): tick the horizons to use and link each to a well top"))
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["use", "horizon", "formation top at the wells", "velocity function",
                                              "k (blank = fit)", "seismic velocity guide"])
        lay.addWidget(self.table, 1)
        self.fms = ["(time-depth at the wells)"]
        if project.well_project:
            ws = project.wellset()
            if ws.wcp is not None:
                self.fms += [f["name"] for f in ws.wcp.formations]
        self._fill_layers()
        self.misfit = QtWidgets.QTableWidget(0, 6)
        self.misfit.setHorizontalHeaderLabels(["well", "horizon", "formation", "z well", "z model", "residual m"])
        lay.addWidget(QtWidgets.QLabel("Well misfit"))
        lay.addWidget(self.misfit, 1)
        row = QtWidgets.QHBoxLayout()
        self.b_build = QtWidgets.QPushButton("Build model")
        self.b_map = QtWidgets.QPushButton("Depth map of selected layer")
        self.b_use = QtWidgets.QPushButton("Use for depth display")
        self.b_csv = QtWidgets.QPushButton("Export misfit CSV...")
        self.b_xplot = QtWidgets.QPushButton("Cross-plot of selected layer")
        for b in (self.b_build, self.b_map, self.b_xplot, self.b_use, self.b_csv):
            row.addWidget(b)
        lay.addLayout(row)
        wrow = QtWidgets.QHBoxLayout()
        wrow.addWidget(QtWidgets.QLabel("What-if well:  X"))
        self.wi_x = QtWidgets.QLineEdit("")
        self.wi_y = QtWidgets.QLineEdit("")
        self.wi_layer = QtWidgets.QComboBox()
        self.wi_layer.addItems([h.name for h in project.horizons.items])
        self.wi_z = QtWidgets.QLineEdit("")
        self.wi_z.setPlaceholderText("TVDSS m")
        self.b_wi_add = QtWidgets.QPushButton("Add pseudo-top")
        self.b_wi_clear = QtWidgets.QPushButton("Clear what-if wells")
        for w_ in (self.wi_x, QtWidgets.QLabel("Y"), self.wi_y, QtWidgets.QLabel("horizon"), self.wi_layer,
                   QtWidgets.QLabel("depth"), self.wi_z, self.b_wi_add, self.b_wi_clear):
            wrow.addWidget(w_)
        lay.addLayout(wrow)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.b_wi_add.clicked.connect(lambda *_: self._safe(lambda: self.add_what_if(
            None, float(self.wi_x.text()), float(self.wi_y.text()), self.wi_layer.currentText(),
            float(self.wi_z.text()))))
        self.b_wi_clear.clicked.connect(lambda *_: self._safe(self.clear_what_if))
        self.resize(820, 640)
        self.b_build.clicked.connect(lambda: self._build())
        self.b_map.clicked.connect(lambda: self._safe(lambda: self.show_depth_map(self.layer_index_of_row(
            max(0, self.table.currentRow())))))
        self.b_use.clicked.connect(lambda: self._safe(self.use))
        self.b_csv.clicked.connect(lambda *_: self._csv())
        self.b_xplot.clicked.connect(lambda *_: self._safe(lambda: self.show_crossplot(self.layer_index_of_row(
            max(0, self.table.currentRow())))))
        self._xplot_win = None

    def _fill_layers(self):
        hs = self.project.horizons.items
        def depth_key(h):  # horizons without picks go last (a NaN key would scramble the sort)
            v = np.concatenate([np.ravel(a) for a in h.data.values()]) if h.data else np.array([])
            v = v[np.isfinite(v)]
            return float(v.mean()) if v.size else np.inf
        order = sorted(hs, key=depth_key)
        self.table.setRowCount(len(order))
        self._combos = []
        self._funcs = {}
        self._guides = {}
        for r, h in enumerate(order):
            chk = QtWidgets.QTableWidgetItem("")
            chk.setFlags(chk.flags() | QtCore.Qt.ItemIsUserCheckable)
            chk.setCheckState(QtCore.Qt.Checked)
            self.table.setItem(r, 0, chk)
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(h.name))
            cb = QtWidgets.QComboBox()
            cb.addItems(self.fms)
            if h.name in self.fms:
                cb.setCurrentText(h.name)
            self.table.setCellWidget(r, 2, cb)
            fcb = QtWidgets.QComboBox()
            for key, label in FUNCTIONS.items():
                fcb.addItem(label, key)
            if "td" in FUNCTIONS:
                fcb.addItem("Well T-D function (nearest well)", "td_nearest")
            self.table.setCellWidget(r, 3, fcb)
            kedit = QtWidgets.QLineEdit("")
            kedit.setPlaceholderText("fit")
            self.table.setCellWidget(r, 4, kedit)
            self._combos.append((h.name, cb, chk))
            gcb = QtWidgets.QComboBox()
            gcb.addItems(["(none)"] + [v["name"] for v in self.project.volumes])
            self.table.setCellWidget(r, 5, gcb)
            gcb.currentIndexChanged.connect(lambda *_, n=h.name, c=gcb: self._guide_changed(n, c))
            self._guides[h.name] = [gcb, None]
            self._funcs[h.name] = [fcb, kedit, "v0", ""]  # widgets + plain state (the mock's widgets hold no values)
            fcb.currentIndexChanged.connect(lambda *_, n=h.name, c=fcb: self._func_changed(n, c))
            kedit.textChanged.connect(lambda txt, n=h.name: self._k_changed(n, txt))

    def layers_from_table(self):
        out = []
        for name, cb, chk in self._combos:
            if chk.checkState() == QtCore.Qt.Checked:
                fm = cb.currentText()
                out.append((name, None if (not fm or fm.startswith("(")) else fm))
        return out

    def _func_changed(self, name, combo):
        d = combo.itemData(combo.currentIndex())
        if isinstance(d, str):
            self._funcs[name][2] = d

    def _k_changed(self, name, txt):
        self._funcs[name][3] = str(txt)

    def set_layer_function(self, name, func, k=None):
        """Set a layer's function ("v0", "v0kz", "v0kt", "td", "td_nearest") and k text (None/"" = fit)."""
        fcb, kedit, _, _ = self._funcs[name]
        self._funcs[name][2] = func
        self._funcs[name][3] = "" if k is None else str(k)
        keys = list(FUNCTIONS) + ["td_nearest"]
        if func in keys:
            fcb.setCurrentIndex(keys.index(func))
        kedit.setText(self._funcs[name][3])

    def _grid_widgets_changed(self):
        m = self.g_method.itemData(self.g_method.currentIndex())
        rng = str(self.g_range.text()).strip()
        try:
            rv = float(rng) if rng else None
        except ValueError:
            rv = None
        self.set_gridding(m if isinstance(m, str) else "idw", str(self.g_vario.currentText()), rv, update_widgets=False)
        k = self.g_kind.itemData(self.g_kind.currentIndex())
        md = self.g_mode.itemData(self.g_mode.currentIndex())
        if isinstance(k, str) and isinstance(md, str):
            self._guide_opts = {"kind": k, "mode": md}

    def set_gridding(self, method="idw", variogram="spherical", range_m=None, update_widgets=True):
        """Gridding of the well values / residuals: idw | tps | kriging (+ variogram model, range or None = fit)."""
        g = {"method": method}
        if method == "kriging":
            g["variogram"] = variogram if variogram in ("spherical", "exponential") else "spherical"
            if range_m:
                g["range"] = float(range_m)
        self._gridding = g
        if update_widgets:
            self.g_method.setCurrentIndex(["idw", "tps", "kriging"].index(method))
        return g

    def _guide_changed(self, name, combo):
        t = str(combo.currentText())
        self._guides[name][1] = None if (not t or t.startswith("(")) else t

    def set_layer_guide(self, name, volume=None, kind=None, mode=None):
        """Seismic velocity guide of a layer: a project volume name (None = no guide); kind rms|interval and mode
        ratio|difference apply to all guided layers."""
        self._guides[name][1] = volume
        if kind:
            self._guide_opts["kind"] = kind
        if mode:
            self._guide_opts["mode"] = mode
        names = [v["name"] for v in self.project.volumes]
        self._guides[name][0].setCurrentIndex(names.index(volume) + 1 if volume in names else 0)

    def functions_from_table(self, names=None):
        """Function dicts of the ticked layers (or of `names`), in layers_from_table() order."""
        names = [n for n, _ in self.layers_from_table()] if names is None else list(names)
        out = []
        for n in names:
            func, ktxt = self._funcs[n][2], self._funcs[n][3].strip()
            if func == "td_nearest":
                out.append({"func": "td", "blend": "nearest"})
                continue
            f = {"func": func}
            if func in ("v0kz", "v0kt"):
                try:
                    f["k"] = float(ktxt) if ktxt else None
                except ValueError:
                    raise ValueError(f"layer {n}: k '{ktxt}' is not a number")
                if f["k"] is None:
                    del f["k"]
            gv = self._guides.get(n, [None, None])[1]
            if gv and func in ("v0", "v0kz", "v0kt"):
                f["guide"] = {"volume": gv, **self._guide_opts}
            out.append(f)
        return out

    # ------------------------------------------------------------ logic (testable)
    def build(self, key=None, layers=None, residual=None, functions=None, gridding=None):
        key = key or self.survey.currentText()
        if layers is None:
            layers = self.layers_from_table()
            if functions is None:
                functions = self.functions_from_table()
        residual = bool(self.resid.isChecked()) if residual is None else residual
        from ..ddc import DynamicDepthModel
        self.model = VelocityModel(self.project, key, layers, residual, functions=functions,
                                   gridding=gridding if gridding is not None else dict(self._gridding))
        self.ddm = DynamicDepthModel(self.project, self.model).build()
        self._fill_misfit()
        return self.model

    def _fill_misfit(self):
        mt = self.model.misfit_table()
        self.misfit.setRowCount(len(mt))
        for r, row in enumerate(mt):
            vals = [row[0], row[1], row[2], f"{row[3]:.1f}", f"{row[4]:.1f}", f"{row[5]:+.1f}"]
            for c, v in enumerate(vals):
                self.misfit.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
        res = np.array([m[5] for m in mt]) if mt else np.array([0.0])
        self._info_text = (f"{len(self.model.grids()['wells'])} wells; misfit RMS {np.sqrt(np.mean(res ** 2)):.1f} m, "
                          f"max {np.max(np.abs(res)):.1f} m\n" + self.info_text() +
                          (f"\n{len(self.ddm.pseudo)} what-if well(s)" if self.ddm is not None and self.ddm.pseudo
                           else ""))
        self.info.setText(self._info_text)

    def model_refreshed(self, report=None):
        """The dynamic model changed (auto-update): refill the misfit table and redraw the depth map shown."""
        if self.model is None:
            return
        self._fill_misfit()
        if report:
            self.info.setText(report.get("message", "") + "\n" + self._info_text)
        if self._shown_layer is not None and self.basemap is not None and self._shown_layer < len(self.model.layers):
            self.basemap.show_grid(self.model.depth_grid(self._shown_layer))

    def add_what_if(self, name, x, y, horizon_or_formation, z):
        """Pseudo-top (TVDSS m) at map x, y; the model rebuilds from that layer down (well project untouched)."""
        if self.ddm is None:
            self.build()
        name = name or f"WI-{len(self.ddm.pseudo) + 1}"
        tops = dict(self.ddm.pseudo.get(name, {}).get("tops", {}))
        tops[horizon_or_formation] = float(z)
        rep = self.ddm.add_what_if(name, x, y, tops)
        self.model_refreshed(rep)
        if self.on_change and self.project.display.get("velocity_model") == self.model.to_dict():
            self.on_change()
        return rep

    def clear_what_if(self):
        if self.ddm is None:
            return None
        rep = self.ddm.remove_what_if(None)
        self.model_refreshed(rep)
        if self.on_change and self.project.display.get("velocity_model") == self.model.to_dict():
            self.on_change()
        return rep

    def info_text(self):
        """One line per layer: function, k and where it came from."""
        if self.model is None:
            return ""
        out = []
        for (name, _), lay in zip(self.model.layers, self.model.layer_info()):
            s = f"{name}: {FUNCTIONS[lay['func']]}"
            if lay.get("k") is not None:
                unit = "1/s" if lay["func"] == "v0kz" else "m/s2"
                s += f", k = {lay['k']:.4g} {unit} ({lay['k_source']})"
            if lay.get("blend"):
                s += f" ({lay['blend']})"
            if self.model.functions[len(out)].get("guide"):
                gd = self.model.functions[len(out)]["guide"]
                s += f", guided by {gd['volume']} ({gd.get('kind', 'rms')}, {gd.get('mode', 'ratio')})"
            out.append(s)
        gm = self.model.gridding.get("method", "idw")
        return "; ".join(out) + f"; gridding: {gm}"

    def crossplot(self, k):
        """Cross-plot data of layer k (see VelocityModel.crossplot)."""
        if self.model is None:
            self.build()
        return self.model.crossplot(k)

    def show_crossplot(self, k):
        cp = self.crossplot(k)
        import pyqtgraph as pg
        win = QtWidgets.QDialog(self)
        win.setWindowTitle(f"Cross-plot: {cp['layer']}")
        lay = QtWidgets.QVBoxLayout(win)
        pw = pg.PlotWidget()
        pw.setLabel("bottom", cp["xlabel"])
        pw.setLabel("left", "interval velocity (m/s)")
        x, v, o = np.asarray(cp["x"]), np.asarray(cp["vint"]), np.asarray(cp["outlier"])
        pw.plot(x[~o], v[~o], pen=None, symbol="o", symbolBrush=(0, 90, 200))
        if o.any():
            pw.plot(x[o], v[o], pen=None, symbol="x", symbolBrush=(220, 0, 0), symbolPen=(220, 0, 0))
        fin = np.isfinite(x)
        if fin.sum() >= 2:
            xl = np.array([np.nanmin(x), np.nanmax(x)])
            pw.plot(xl, cp["intercept"] + cp["slope"] * xl, pen=pg.mkPen((0, 0, 0)))
        for xi, vi, n in zip(x, v, cp["wells"]):
            if np.isfinite(xi) and np.isfinite(vi):
                t = pg.TextItem(n, anchor=(0, 1), color=(60, 60, 60))
                t.setPos(float(xi), float(vi))
                pw.addItem(t)
        lay.addWidget(pw)
        txt = f"line: V = {cp['intercept']:.0f} + {cp['slope']:.4g} x, R2 = {cp['r2']:.3f}"
        if cp["k"] is not None:
            txt += f"; model k = {cp['k']:.4g} ({cp['k_source']})"
        if np.asarray(cp["outlier"]).any():
            txt += "; outliers (> 2 sigma): " + ", ".join(n for n, f in zip(cp["wells"], cp["outlier"]) if f)
        lab = QtWidgets.QLabel(txt)
        lab.setWordWrap(True)
        lay.addWidget(lab)
        win.resize(560, 460)
        win.show()
        self._xplot_win = win
        return cp

    def layer_index_of_row(self, row):
        """Table row -> index among the model's (ticked) layers; unticked rows are not in the model."""
        if self.model is None:
            self.build()
        item = self.table.item(row, 1)
        name = item.text() if item is not None else None
        names = [n for n, _ in self.model.layers]
        if name not in names:
            raise ValueError(f"layer '{name}' is not ticked / not in the model")
        return names.index(name)

    def _safe(self, fn):
        try:
            return fn()
        except Exception as e:
            self.info.setText(f"Velocity model: {e}")
            return None

    def _build(self):
        try:
            self.build()
        except Exception as e:
            self.info.setText(f"Cannot build the model: {e}")

    def show_depth_map(self, k):
        if self.model is None:
            self.build()
        g = self.model.depth_grid(k)
        self._shown_layer = k
        if self.basemap is not None:
            self.basemap.show_grid(g)
        return g

    def use(self):
        if self.model is None:
            self.build()
        self.project.display["velocity_model"] = self.model.to_dict()
        if self.project.path:
            self.project.save()
        if self.on_change:
            self.on_change()

    def write_misfit(self, path):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["well", "horizon", "formation", "z_well_m", "z_model_m", "residual_m"])
            for row in self.model.misfit_table():
                w.writerow([row[0], row[1], row[2], f"{row[3]:.2f}", f"{row[4]:.2f}", f"{row[5]:.2f}"])
        return path

    def _csv(self):
        if self.model is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Misfit table", self.project.base(), "CSV (*.csv)")
        if path:
            self.write_misfit(path)
