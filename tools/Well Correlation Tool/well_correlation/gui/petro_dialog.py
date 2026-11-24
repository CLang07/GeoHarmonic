"""Petrophysics window (PLAN 20e): parameter sets per zone, run QC / interpretation / mineral solver / Monte Carlo /
rock physics for one well or a batch, cross-plots (N-D, M-N, Pickett, Buckles, histogram with discriminators), zone
results table, result tracks in the section (toolbar VSH / PHI / SW / PERM), and "Show on map hub".

All logic lives in plain methods (set_well, set_zone, param_rows, set_param, run_*, crossplot_data, summary_rows,
show_on_hub, export_png) so it is testable under the Qt mock; drawing is in the _draw_* methods.
"""
import json
import os
import time

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from ..petro import interp as I
from ..petro import plotdata as PD
from ..petro.store import petro_dir

SCOPES = ["Project default", "Zone (all wells)", "This well, this zone", "This well, all zones"]
XPLOTS = ["N-D", "M-N", "Pickett", "Buckles", "Histogram"]
PARAM_ORDER = [k for k in I.DEFAULTS]


def parse_value(name, text):
    """Text from the table -> parameter value (None = remove the override)."""
    t = (text or "").strip()
    if t == "" or t.lower() in ("auto", "none", "-"):
        return None
    if name in I.METHOD_CHOICES:
        if t not in I.METHOD_CHOICES[name]:
            raise ValueError(f"{name}: choose one of {', '.join(I.METHOD_CHOICES[name])}")
        return t
    if name in ("mm_minerals", "mm_logs"):
        return [s.strip() for s in t.replace(";", ",").split(",") if s.strip()]
    if name in ("toc", "fit_m"):
        return t.lower() in ("1", "true", "yes", "y", "on")
    if name in ("mm_fluid", "rp_hc", "rp_lith"):
        return t
    try:
        return float(t)
    except ValueError:
        raise ValueError(f"{name}: not a number ({t})")


def fmt_value(v):
    if v is None:
        return "auto"
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


class PetroDialog(QtWidgets.QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.project = win.project
        self.setWindowTitle("Petrophysics")
        self.resize(1250, 800)
        self.pset = I.load_params(self.project)
        self.well = None
        self.zone = None
        self.result = None  # 20b result of the current well
        self.mm = None
        self.mc = None
        self.rp = None
        self.qc = None
        self.xkind = "N-D"
        self.hist_curve = "PHIE"
        self.discriminators = {}
        self.messages = []
        self._build()
        wells = list(self.project.wells)
        sel = getattr(win, "selected_well", None) or (self.project.section_wells() or wells or [None])[0]
        if sel:
            self.set_well(sel)

    # ------------------------------------------------------------------ ui
    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.well_combo = QtWidgets.QComboBox()
        self.well_combo.addItems(list(self.project.wells))
        self.well_combo.currentTextChanged.connect(lambda t: t and self.set_well(t))
        self.zone_combo = QtWidgets.QComboBox()
        self.zone_combo.currentTextChanged.connect(lambda t: self.set_zone(t or None))
        self.scope = QtWidgets.QComboBox()
        self.scope.addItems(SCOPES)
        self.scope.setCurrentIndex(1)
        for lab, wdg in (("Well", self.well_combo), ("Zone", self.zone_combo), ("Edit applies to", self.scope)):
            top.addWidget(QtWidgets.QLabel(lab))
            top.addWidget(wdg)
        top.addStretch(1)
        for text, fn in (("QC report", lambda *_: self.run_qc()), ("Interpret well", lambda *_: self.run_interp()),
                         ("Interpret section wells", lambda *_: self.run_interp(self.project.section_wells())),
                         ("Interpret all wells", lambda *_: self.run_interp(list(self.project.wells))),
                         ("Mineral solver", lambda *_: self.run_multimin()), ("Monte Carlo", lambda *_: self.run_mc()),
                         ("Rock physics", lambda *_: self.run_rockphysics()), ("Show on map hub", lambda *_: self.show_on_hub())):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            top.addWidget(b)
        lay.addLayout(top)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        # parameters
        self.ptable = QtWidgets.QTableWidget()
        self.ptable.setColumnCount(3)
        self.ptable.setHorizontalHeaderLabels(["Parameter", "Value", "From"])
        self.ptable.itemChanged.connect(self._param_edited)
        split.addWidget(self.ptable)
        # plots + results
        right = QtWidgets.QTabWidget()
        xw = QtWidgets.QWidget()
        xl = QtWidgets.QVBoxLayout(xw)
        row = QtWidgets.QHBoxLayout()
        self.xcombo = QtWidgets.QComboBox()
        self.xcombo.addItems(XPLOTS)
        self.xcombo.currentTextChanged.connect(lambda t: t and self.set_crossplot(t))
        self.hcombo = QtWidgets.QComboBox()
        self.hcombo.addItems(["PHIE", "SW", "VSH", "PERM", "GR", "RHOB", "NPHI", "DT", "RDEEP"])
        self.hcombo.currentTextChanged.connect(lambda t: t and self.set_hist_curve(t))
        self.disc = QtWidgets.QLineEdit()
        self.disc.setPlaceholderText("discriminators, e.g. VSH 0 0.4; PHIE 0.08 1")
        self.disc.editingFinished.connect(lambda: self.set_discriminators(self.disc.text()))
        b_png = QtWidgets.QPushButton("Export PNG...")
        b_png.clicked.connect(lambda *_: self._export_dialog())
        for wdg in (self.xcombo, QtWidgets.QLabel("Histogram of"), self.hcombo, self.disc, b_png):
            row.addWidget(wdg)
        xl.addLayout(row)
        self.plot = pg.PlotWidget()
        xl.addWidget(self.plot)
        right.addTab(xw, "Cross-plots")
        self.rtable = QtWidgets.QTableWidget()
        right.addTab(self.rtable, "Zone results")
        self.text = QtWidgets.QTextBrowser()
        right.addTab(self.text, "Reports")
        split.addWidget(right)
        split.setSizes([380, 870])
        lay.addWidget(split)
        self.status = QtWidgets.QLabel("")
        lay.addWidget(self.status)

    def _msg(self, text):
        self.messages.append(text)
        self.status.setText(text)

    # ------------------------------------------------------------------ selection
    def set_well(self, name):
        if name not in self.project.wells:
            return
        self.well = name
        self.result = self.mm = self.mc = self.rp = self.qc = None
        z = self.project.load_well(name).depth
        zones = [zn for zn, _, _ in I.zones_of(self.project, name, z if z is not None else np.array([0.0]))]
        self._zones = zones
        self.zone = zones[0] if zones else None
        try:
            self.zone_combo.blockSignals(True)
            self.zone_combo.clear()
            self.zone_combo.addItems(zones)
        finally:
            self.zone_combo.blockSignals(False)
        self.refresh_params()

    def set_zone(self, zone):
        self.zone = zone
        self.refresh_params()
        self.redraw()

    # ------------------------------------------------------------------ parameters
    def scope_index(self):
        try:
            return int(self.scope.currentIndex())
        except Exception:
            return 1

    def param_rows(self):
        """[(name, effective value, where it comes from)] for the current well + zone."""
        p = self.pset
        zone, well = self.zone, self.well
        eff = I.resolve(p, well, zone)
        rows = []
        for k in PARAM_ORDER:
            src = "built-in"
            if k in p.get("default", {}):
                src = "project"
            if zone and k in p.get("zones", {}).get(zone, {}):
                src = "zone"
            if k in p.get("wells", {}).get(well, {}).get("*", {}):
                src = "well (all zones)"
            if zone and k in p.get("wells", {}).get(well, {}).get(zone, {}):
                src = "well + zone"
            rows.append((k, eff.get(k), src))
        return rows

    def _scope_dict(self, scope):
        p = self.pset
        if scope == 0 or (scope in (1, 2) and not self.zone):
            return p.setdefault("default", {})
        if scope == 1:
            return p.setdefault("zones", {}).setdefault(self.zone, {})
        w = p.setdefault("wells", {}).setdefault(self.well, {})
        return w.setdefault(self.zone if scope == 2 else "*", {})

    def set_param(self, name, text, scope=None):
        """Set (or with '' / 'auto' remove) a parameter at a scope (0-3, see SCOPES) and save params.json."""
        if name not in I.DEFAULTS:
            raise KeyError(name)
        val = parse_value(name, text)
        d = self._scope_dict(self.scope_index() if scope is None else scope)
        if val is None:
            d.pop(name, None)
        else:
            d[name] = val
        I.save_params(self.project, self.pset)
        self.refresh_params()
        return val

    def refresh_params(self):
        rows = self.param_rows()
        t = self.ptable
        try:
            t.blockSignals(True)
            t.setRowCount(len(rows))
            for i, (k, v, src) in enumerate(rows):
                for j, txt in enumerate((k, fmt_value(v), src)):
                    it = QtWidgets.QTableWidgetItem(txt)
                    if j != 1:
                        it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                    t.setItem(i, j, it)
        finally:
            t.blockSignals(False)

    def _param_edited(self, item):
        try:
            if item.column() != 1:
                return
            name = self.ptable.item(item.row(), 0).text()
            self.set_param(name, item.text())
            self._msg(f"{name} set")
        except Exception as e:
            self._msg(f"Not set: {e}")
            self.refresh_params()

    # ------------------------------------------------------------------ runs
    def _busy(self, on):
        try:
            if on:
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            else:
                QtWidgets.QApplication.restoreOverrideCursor()
        except Exception:
            pass

    def run_qc(self):
        from ..petro import qc
        self._busy(True)
        try:
            self.qc = qc.well_qc(self.project, self.well)
        finally:
            self._busy(False)
        self.text.setPlainText(qc.format_report(self.qc))
        self._msg(f"QC report for {self.well}: {len(self.qc['warnings'])} warnings")
        self._refresh_section()
        return self.qc

    def run_interp(self, wells=None):
        """Interpret the current well (wells=None) or a batch; refresh tables, plots and the section tracks."""
        t0 = time.perf_counter()
        self._busy(True)
        try:
            if wells:
                res = I.interpret_project(self.project, wells, self.pset)
                self.result = res.get(self.well) or self.result
                bad = [n for n, r in res.items() if r.get("notes") and not r.get("zones")]
                msg = f"interpreted {len(res)} wells in {time.perf_counter() - t0:.1f} s" + (
                    f"; problems: {', '.join(bad)}" if bad else "")
            else:
                self.result = I.interpret_well(self.project, self.well, self.pset)
                msg = f"{self.well} interpreted in {time.perf_counter() - t0:.2f} s"
        finally:
            self._busy(False)
        if self.well and (self.result is None or self.result.get("well") != self.well):
            self.result = I.interpret_well(self.project, self.well, self.pset)
        notes = (self.result or {}).get("notes", [])
        self._msg(msg + ("; " + "; ".join(notes) if notes else ""))
        self.refresh_results()
        self.redraw()
        self._refresh_section()
        return self.result

    def _need_result(self):
        if self.result is None or self.result.get("well") != self.well:
            self.result = I.interpret_well(self.project, self.well, self.pset)
        return self.result

    def run_multimin(self):
        from ..petro import multimin as M
        det = self._need_result()
        self.mm = M.multimin_well(self.project, self.well, pset=self.pset, det=det)
        self._msg(f"mineral solver: {', '.join(self.mm['components'])}")
        self.refresh_results()
        return self.mm

    def run_mc(self, n=200, method="deterministic"):
        from ..petro import multimin as M
        self._busy(True)
        try:
            self.mc = M.monte_carlo(self.project, self.well, n=n, pset=self.pset, method=method)
        finally:
            self._busy(False)
        self._msg(f"Monte Carlo: {n} realisations ({method})")
        self.refresh_results()
        return self.mc

    def run_rockphysics(self):
        from ..petro import rockphysics as R
        det = self._need_result()
        self.rp = R.rock_physics_well(self.project, self.well, pset=self.pset, det=det)
        self._msg("rock physics: Vs from " + "; ".join(self.rp["vs_source"]))
        self.refresh_results()
        return self.rp

    def _refresh_section(self):
        w = self.project.wells.get(self.well)
        if w is not None and hasattr(w, "_petro_cache"):
            w._petro_cache = None
        sec = getattr(self.win, "section", None)
        if sec is not None:
            try:
                sec.redraw(keep_view=True)
            except Exception:
                pass

    # ------------------------------------------------------------------ results
    def summary_rows(self):
        """Zone rows of the current well (deterministic, + mineral solver and P90/P50/P10 when run)."""
        if self.result is None:
            return []
        rows = []
        mm = {r["Zone"]: r for r in (self.mm or {}).get("zones", [])}
        mc = {}
        if self.mc:
            from ..petro.multimin import mc_table
            mc = {r["Zone"]: r for r in mc_table(self.mc)}
        for r in self.result["zones"]:
            row = {k: r[k] for k in ("Zone", "Gross_TVD", "NetRes_TVD", "NetPay_TVD", "PHIE_res", "Sw_pay", "Vsh_avg",
                                     "PHIH_pay", "SoPhiH", "kh_pay")}
            m = mm.get(r["Zone"])
            if m:
                row.update({"MM_PHIE": m["PHIE_res"], "MM_NetPay": m["NetPay_TVD"], "MM_misfit": m["Misfit_med"]})
            c = mc.get(r["Zone"])
            if c:
                row.update({"SoPhiH_P90": c["SoPhiH_P90"], "SoPhiH_P50": c["SoPhiH_P50"], "SoPhiH_P10": c["SoPhiH_P10"]})
            rows.append(row)
        return rows

    def refresh_results(self):
        rows = self.summary_rows()
        t = self.rtable
        cols = list(dict.fromkeys(k for r in rows for k in r))
        t.clear()
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                v = r.get(c)
                t.setItem(i, j, QtWidgets.QTableWidgetItem("" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))))

    # ------------------------------------------------------------------ cross-plots
    def set_crossplot(self, kind):
        if kind not in XPLOTS:
            raise ValueError(kind)
        self.xkind = kind
        self.redraw()

    def set_hist_curve(self, name):
        self.hist_curve = name
        if self.xkind == "Histogram":
            self.redraw()

    def set_discriminators(self, text):
        """'VSH 0 0.4; PHIE 0.08 1' -> {VSH: (0, 0.4), PHIE: (0.08, 1)}."""
        d = {}
        for part in (text or "").split(";"):
            bits = part.replace(",", " ").split()
            if len(bits) == 3:
                d[bits[0].upper()] = (float(bits[1]), float(bits[2]))
        self.discriminators = d
        if self.xkind == "Histogram":
            self.redraw()
        return d

    def _curves(self):
        res = self._need_result()
        w = self.project.load_well(self.well)
        allc = dict(w.logs)
        allc.update(res["curves"])
        return res, w.logs, allc

    def crossplot_data(self, kind=None):
        kind = kind or self.xkind
        res, logs, allc = self._curves()
        if kind == "N-D":
            return PD.nd(res, logs, self.zone)
        if kind == "M-N":
            return PD.mn(res, logs, self.zone)
        if kind == "Pickett":
            return PD.pickett(res, self.zone)
        if kind == "Buckles":
            return PD.buckles(res, self.zone)
        if kind == "Histogram":
            v = allc.get(self.hist_curve)
            if v is None:
                return None
            mask = PD._zone_mask(res, self.zone)
            return PD.histogram(v, discriminators=self.discriminators, curves=allc, mask=mask,
                                log=self.hist_curve in ("PERM", "RDEEP", "RSHAL"))
        raise ValueError(kind)

    def redraw(self):
        if self.well is None or self.result is None:
            return
        try:
            data = self.crossplot_data()
        except Exception as e:
            self._msg(f"plot: {e}")
            return
        self._draw(data)

    def _draw(self, data):
        pl = self.plot
        pl.clear()
        if data is None:
            pl.setTitle("logs missing for this plot")
            return
        if data["kind"] == "Histogram":
            e = data["edges"]
            pl.addItem(pg.BarGraphItem(x=0.5 * (e[:-1] + e[1:]), height=data["counts"], width=float(np.diff(e)[0]),
                                       brush="#4a78b5"))
            pl.setTitle(f"{self.hist_curve}: n {data['n']}" + (f", P10 {data['p10']:.3g} P50 {data['p50']:.3g} "
                                                              f"P90 {data['p90']:.3g}" if data["n"] else ""))
            return
        logx, logy = bool(data.get("logx")), bool(data.get("logy"))  # drawn as log10 values (axis labels say so)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.log10(data["x"]) if logx else data["x"]
            y = np.log10(data["y"]) if logy else data["y"]
        c = np.clip(np.nan_to_num(data["c"], nan=0.5), 0, 1)
        brushes = pg.mkBrush(70, 120, 180, 120)
        if len(c) < 5000:
            try:
                rgba = np.asarray(pg.colormap.get("viridis").map(c, mode="byte"))
                brushes = [pg.mkBrush(*[int(q) for q in row]) for row in rgba.reshape(-1, 4)]
            except Exception:
                pass
        pl.addItem(pg.ScatterPlotItem(x=x, y=y, size=4, pen=None, brush=brushes))
        for name, ln in data.get("lines", {}).items():
            lx = np.log10(ln[0]) if logx else ln[0]
            ly = np.log10(ln[1]) if logy else ln[1]
            pl.plot(lx, ly, pen=pg.mkPen("k", width=1), name=name)
        for name, (px, py) in data.get("points", {}).items():
            t = pg.TextItem(name, color="k")
            t.setPos(px, py)
            pl.addItem(t)
        pl.setLabel("bottom", ("log10 " if logx else "") + data["xlabel"])
        pl.setLabel("left", ("log10 " if logy else "") + data["ylabel"])
        pl.invertY(bool(data["ylim"][0] > data["ylim"][1]))
        pl.setTitle(f"{data['kind']} - {self.well} {self.zone or ''} (colour {data.get('color', '')})")

    def export_png(self, path, kind=None):
        data = self.crossplot_data(kind)
        if data is None:
            raise ValueError("no data for this plot")
        return PD.render_png(data, path, f"{data['kind']} - {self.well} {self.zone or ''}")

    def _export_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export cross-plot", f"{self.well}_{self.xkind}.png",
                                                        "PNG (*.png)")
        if path:
            try:
                self.export_png(path)
                self._msg(f"wrote {path}")
            except Exception as e:
                self._msg(f"export failed: {e}")

    # ------------------------------------------------------------------ map hub
    def show_on_hub(self, prop="SoPhiH", wells=None):
        """Interpret the wells (default: section wells, else all) so summary.csv is current, then write
        <project>_petro/hub_request.json - the Seismic tool's map hub (PLAN 21) watches it and posts `prop` per well."""
        wells = wells or self.project.section_wells() or list(self.project.wells)
        I.interpret_project(self.project, wells, self.pset)
        d = petro_dir(self.project, create=True)
        req = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "wells": wells, "zone": self.zone, "property": prop,
               "summary": os.path.join(d, "summary.csv"), "project": self.project.path}
        p = os.path.join(d, "hub_request.json")
        with open(p + ".tmp", "w") as f:
            json.dump(req, f, indent=1)
        os.replace(p + ".tmp", p)
        self._msg(f"summary for {len(wells)} wells written; map hub request: {prop} {self.zone or ''}")
        return p
