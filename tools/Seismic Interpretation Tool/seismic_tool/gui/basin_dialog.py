"""Interpret > Basin model (PLAN 18e): age table + lithologies, heat flow + BHT calibration, burial / temperature /
maturity per well, pseudo-well maps (threaded), Wheeler colouring, event chart, CSV / CPS-3 exports.
All logic is in plain methods (testable under dev/fakeqt); widgets only call them."""
import os

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtWidgets  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from ..basin import chrono as C  # noqa: E402
from ..basin import maturity as MT  # noqa: E402
from ..basin import model as BM  # noqa: E402
from ..basin import pseudowells as PW  # noqa: E402

AGE_COLS = ["name", "horizon", "age Ma", "gap My", "eroded m", "role"] + list(C.LITHS)
HEAT_KINDS = ["constant", "piecewise", "mckenzie"]
MAP_OUTPUTS = ["temperature", "ro", "tr", "oil_entry"]
WHEELER_PROPS = ["(seismic)", "temperature", "ro", "tr", "porosity"]
COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75),
          (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]


class BasinDialog(QtWidgets.QDialog):
    def __init__(self, project, basemap=None, parent=None, section=None, run_job=None):
        super().__init__(parent)
        self.setWindowTitle("Basin model - burial, thermal and maturity history")
        self.project, self.basemap, self.section, self.run_job_fn = project, basemap, section, run_job
        self.table = C.AgeTable.load(project)
        self.s = BM.settings(project)
        self.well_result = None
        self.on_wheeler = None  # callable(prop): the main window registers the section on the Live Map (31h)
        self.fit = None
        self.map_result = None
        self._map_args = None
        self.chart = None
        self.plot_data = {}
        self._last_text = ""
        self._filling = False
        self.auto_update = None  # None = follow the checkbox (tests set True / False)
        lay = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        lay.addWidget(tabs, 1)
        # ---- tab 1: age table
        t1 = QtWidgets.QWidget()
        l1 = QtWidgets.QVBoxLayout(t1)
        self.grid = QtWidgets.QTableWidget(0, len(AGE_COLS))
        self.grid.setHorizontalHeaderLabels(AGE_COLS)
        self.grid.setToolTip("One row per dated surface, youngest first. age = onset of the deposits above; gap = "
                             "hiatus below the surface (My); eroded = metres removed at an unconformity; lithology "
                             "fractions describe the zone BELOW the surface; role = source / reservoir / seal / "
                             "overburden (event chart)")
        l1.addWidget(self.grid, 1)
        r1 = QtWidgets.QHBoxLayout()
        self.b_tmpl = QtWidgets.QPushButton("Taranaki template")
        self.b_add = QtWidgets.QPushButton("Add surface")
        self.b_del = QtWidgets.QPushButton("Delete row")
        self.b_lith = QtWidgets.QPushButton("Lithology from wells")
        self.b_save = QtWidgets.QPushButton("Save to project")
        for b in (self.b_tmpl, self.b_add, self.b_del, self.b_lith, self.b_save):
            r1.addWidget(b)
        l1.addLayout(r1)
        tabs.addTab(t1, "Age table")
        # ---- tab 2: heat flow + wells
        t2 = QtWidgets.QWidget()
        l2 = QtWidgets.QVBoxLayout(t2)
        r2 = QtWidgets.QHBoxLayout()
        self.hf_kind = QtWidgets.QComboBox()
        self.hf_kind.addItems(HEAT_KINDS)
        self.hf_q = QtWidgets.QLineEdit("60")
        self.hf_q.setToolTip("constant: mW/m2; piecewise: age:q pairs, e.g. 80:75, 40:65, 0:60")
        self.hf_rift = QtWidgets.QLineEdit("84")
        self.hf_beta = QtWidgets.QLineEdit("1.5")
        self.ts = QtWidgets.QLineEdit("12")
        self.wd = QtWidgets.QLineEdit("110")
        self.kin = QtWidgets.QComboBox()
        self.kin.addItems(list(MT.KINETICS))
        for w_ in (QtWidgets.QLabel("Heat flow"), self.hf_kind, self.hf_q, QtWidgets.QLabel("rift Ma"), self.hf_rift,
                   QtWidgets.QLabel("beta"), self.hf_beta, QtWidgets.QLabel("surface degC"), self.ts,
                   QtWidgets.QLabel("water m"), self.wd, QtWidgets.QLabel("kinetics"), self.kin):
            r2.addWidget(w_)
        l2.addLayout(r2)
        r3 = QtWidgets.QHBoxLayout()
        self.well = QtWidgets.QComboBox()
        self.b_run = QtWidgets.QPushButton("Run well")
        self.b_cal = QtWidgets.QPushButton("Calibrate heat flow to BHT")
        self.b_bcsv = QtWidgets.QPushButton("Burial CSV...")
        for w_ in (QtWidgets.QLabel("Well"), self.well, self.b_run, self.b_cal, self.b_bcsv):
            r3.addWidget(w_)
        l2.addLayout(r3)
        plots = QtWidgets.QHBoxLayout()
        self.p_burial = pg.PlotWidget(title="Burial (depth vs age)")
        self.p_temp = pg.PlotWidget(title="Temperature history")
        self.p_ro = pg.PlotWidget(title="%Ro history")
        self.p_cal = pg.PlotWidget(title="Calibration (T vs depth)")
        for p in (self.p_burial, self.p_temp, self.p_ro, self.p_cal):
            plots.addWidget(p)
        l2.addLayout(plots, 1)
        tabs.addTab(t2, "Wells")
        # ---- tab 3: maps, Wheeler, events
        t3 = QtWidgets.QWidget()
        l3 = QtWidgets.QVBoxLayout(t3)
        r4 = QtWidgets.QHBoxLayout()
        self.m_surf = QtWidgets.QComboBox()
        self.m_out = QtWidgets.QComboBox()
        self.m_out.addItems(MAP_OUTPUTS)
        self.m_stride = QtWidgets.QSpinBox()
        self.m_stride.setRange(1, 100)
        self.m_stride.setValue(int(self.s.get("stride", 10)))
        self.m_stride.setPrefix("every ")
        self.m_stride.setSuffix(" nodes")
        self.b_map = QtWidgets.QPushButton("Run map")
        self.b_show = QtWidgets.QPushButton("Show on basemap")
        self.b_cps = QtWidgets.QPushButton("Export CPS-3...")
        self.auto = QtWidgets.QCheckBox("Re-run when the depth model changes")
        for w_ in (QtWidgets.QLabel("Surface"), self.m_surf, self.m_out, self.m_stride, self.b_map, self.b_show,
                   self.b_cps, self.auto):
            r4.addWidget(w_)
        l3.addLayout(r4)
        r5 = QtWidgets.QHBoxLayout()
        self.w_prop = QtWidgets.QComboBox()
        self.w_prop.addItems(WHEELER_PROPS)
        self.w_age = QtWidgets.QLineEdit("0")
        self.b_wheel = QtWidgets.QPushButton("Colour Wheeler (age)")
        for w_ in (QtWidgets.QLabel("Wheeler colour"), self.w_prop, QtWidgets.QLabel("at Ma"), self.w_age, self.b_wheel):
            r5.addWidget(w_)
        l3.addLayout(r5)
        r6 = QtWidgets.QHBoxLayout()
        self.e_src = QtWidgets.QComboBox()
        self.e_trap0 = QtWidgets.QLineEdit("12")
        self.e_trap1 = QtWidgets.QLineEdit("5")
        self.b_chart = QtWidgets.QPushButton("Event chart...")
        for w_ in (QtWidgets.QLabel("Source below"), self.e_src, QtWidgets.QLabel("trap from Ma"), self.e_trap0,
                   QtWidgets.QLabel("to"), self.e_trap1, self.b_chart):
            r6.addWidget(w_)
        l3.addLayout(r6)
        tabs.addTab(t3, "Maps / Wheeler / events")
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.resize(1200, 640)
        self._settings_to_widgets()
        self.refresh_table()
        self._fill_wells()
        self.b_tmpl.clicked.connect(lambda *_: self._safe(self.use_template))
        self.b_add.clicked.connect(lambda *_: self._safe(lambda: self.add_entry("New surface", 1.0)))
        self.b_del.clicked.connect(lambda *_: self._safe(self._delete_selected))
        self.b_lith.clicked.connect(lambda *_: self._safe(self.lithology_from_wells))
        self.b_save.clicked.connect(lambda *_: self._safe(self.save))
        self.grid.itemChanged.connect(lambda item: self._safe(lambda: self._cell_edited(item.row(), item.column())))
        self.b_run.clicked.connect(lambda *_: self._safe(lambda: self.run_well(self.well.currentText())))
        self.b_cal.clicked.connect(lambda *_: self._safe(lambda: self.calibrate(self.well.currentText())))
        self.b_bcsv.clicked.connect(lambda *_: self._safe(self._burial_csv))
        self.b_map.clicked.connect(lambda *_: self._safe(lambda: self.run_map(
            self.m_surf.currentText(), int(self.m_stride.value()))))
        self.b_show.clicked.connect(lambda *_: self._safe(lambda: self.show_map(self.m_out.currentText())))
        self.b_cps.clicked.connect(lambda *_: self._safe(self._cps3))
        self.b_wheel.clicked.connect(lambda *_: self._safe(lambda: self.colour_wheeler(
            self.w_prop.currentText(), float(self.w_age.text() or 0))))
        self.b_chart.clicked.connect(lambda *_: self._safe(self._chart_dialog))

    # ------------------------------------------------------------------ helpers
    def _safe(self, fn):
        try:
            return fn()
        except Exception as e:
            self._set_info(f"Basin model: {e}")
            return None

    def _set_info(self, text):
        self._last_text = text
        self.info.setText(text)

    def wcp(self):
        ws = self.project.wellset() if self.project.well_project else None
        return ws.wcp if ws is not None else None

    def _fill_wells(self):
        self.well.clear()
        p = self.wcp()
        if p is not None:
            self.well.addItems([w for w in p.wells if p.tops.get(w)])

    def _settings_to_widgets(self):
        hf = self.s["heat_flow"]
        kind = hf.get("kind", "constant") if isinstance(hf, dict) else "constant"
        self.hf_kind.setCurrentText(kind)
        if kind == "piecewise":
            self.hf_q.setText(", ".join(f"{a:g}:{q:g}" for a, q in hf.get("points", [])))
        elif kind == "mckenzie":
            self.hf_q.setText(f"{hf.get('present', 60.0):g}")
            self.hf_rift.setText(f"{hf.get('rift_age', 84.0):g}")
            self.hf_beta.setText(f"{hf.get('beta', 1.5):g}")
        else:
            self.hf_q.setText(f"{hf.get('q', 60.0) if isinstance(hf, dict) else hf:g}")
        self.ts.setText(f"{self.s['surface_t']:g}")
        self.wd.setText(f"{self.s['water_depth']:g}")
        self.kin.setCurrentText(self.s["kinetics"])

    def settings_from_widgets(self):
        """Read the heat-flow / surface widgets into self.s (real Qt only; tests call set_settings)."""
        kind = self.hf_kind.currentText()
        if kind not in HEAT_KINDS:
            return self.s
        txt = str(self.hf_q.text())
        if kind == "piecewise":
            pts = [tuple(float(v) for v in p.split(":")) for p in txt.replace(";", ",").split(",") if ":" in p]
            hf = {"kind": "piecewise", "points": pts}
        elif kind == "mckenzie":
            hf = {"kind": "mckenzie", "rift_age": float(self.hf_rift.text()), "beta": float(self.hf_beta.text()),
                  "present": float(txt)}
        else:
            hf = {"kind": "constant", "q": float(txt)}
        return self.set_settings(heat_flow=hf, surface_t=float(self.ts.text()), water_depth=float(self.wd.text()),
                                 kinetics=self.kin.currentText(), _sync=False)

    def set_settings(self, _sync=True, **kw):
        for k, v in kw.items():
            if k == "kinetics" and v not in MT.KINETICS:
                continue
            self.s[k] = v
        if _sync and self._widgets_live():  # keep the boxes in step, or Save would read the old values back
            try:
                self._settings_to_widgets()
            except Exception:
                pass
        return self.s

    def _widgets_live(self):
        """True with real Qt widgets (combo text is a str), False under the test mock."""
        return isinstance(self.hf_kind.currentText(), str)

    # ------------------------------------------------------------------ age table
    def refresh_table(self):
        self._filling = True
        try:
            self.grid.setRowCount(len(self.table.entries))
            for r, e in enumerate(self.table.entries):
                vals = [e["name"], e.get("horizon", ""), f"{e['age']:g}", f"{e['gap']:g}", f"{e['eroded']:g}",
                        e.get("role", "")] + [f"{e['lith'][k]:.2f}" for k in C.LITHS]
                for c, v in enumerate(vals):
                    self.grid.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
        finally:
            self._filling = False
        names = self.table.names()
        for combo in (self.m_surf, self.e_src):
            combo.clear()
            combo.addItems(names)
        probs = self.table.problems()
        if probs:
            self._set_info("Age table: " + "; ".join(probs))

    def _cell_edited(self, row, col):
        if self._filling or row >= len(self.table.entries):
            return
        it = self.grid.item(row, col)
        txt = str(it.text()).strip() if it is not None else ""
        e = self.table.entries[row]
        name = e["name"]
        key = AGE_COLS[col]
        if key == "name":
            e["name"] = txt
        elif key == "horizon":
            e["horizon"] = txt
        elif key == "role":
            e["role"] = txt.lower()
        elif key in C.LITHS:
            lith = dict(e["lith"])
            lith[key] = float(txt or 0)
            e["lith"] = lith
        else:
            e[{"age Ma": "age", "gap My": "gap", "eroded m": "eroded"}[key]] = float(txt or 0)
        self.table = C.AgeTable(self.table.entries)
        if key in ("age Ma", "name") or key in C.LITHS:
            self.refresh_table()
        return self.table.get(e["name"] or name)

    def set_entry(self, name, **kw):
        e = self.table.set(name, **kw)
        self.refresh_table()
        return e

    def add_entry(self, name, age):
        return self.set_entry(name, age=float(age))

    def remove_entry(self, name):
        self.table.remove(name)
        self.refresh_table()

    def _delete_selected(self):
        r = self.grid.currentRow()
        if isinstance(r, int) and 0 <= r < len(self.table.entries):
            self.remove_entry(self.table.entries[r]["name"])

    def use_template(self):
        self.table = C.AgeTable.taranaki()
        self.refresh_table()
        self._set_info("Taranaki template loaded (approximate ages - check against your stratigraphy)")
        return self.table

    def lithology_from_wells(self):
        p = self.wcp()
        if p is None:
            raise RuntimeError("link a well project first")
        zl = C.zone_lithology(p, self.table)
        C.apply_lithology(self.table, zl)
        self.refresh_table()
        self._set_info("Lithology from logs: " + ", ".join(f"{n} ({d['wells']} wells)" for n, d in zl.items()))
        return zl

    def save(self):
        if self._widgets_live():
            self.settings_from_widgets()
        self.s["stride"] = int(self.m_stride.value()) if isinstance(self.m_stride.value(), int) else self.s["stride"]
        self.table.save(self.project)
        BM.save_settings(self.project, self.s)
        self._set_info("Age table and basin settings saved in the project (File > Save writes the file)")
        return True

    # ------------------------------------------------------------------ wells
    def run_well(self, well):
        p = self.wcp()
        if p is None:
            raise RuntimeError("link a well project first")
        if self._widgets_live():
            self.settings_from_widgets()
        bh, th = BM.well_model(p, well, self.table, self.s)
        rows, _ = BM.well_maturity(th, self.s["kinetics"])
        self.well_result = {"well": well, "burial": bh, "thermal": th, "rows": rows}
        self._plot_well()
        cal = (self.s.get("calibrated") or {}).get(well)
        q = th.q[-1]
        txt = [f"{well}: present heat flow {q:.1f} mW/m2" + (" (calibrated)" if cal else "") +
               f", {len(bh.ages)} time steps"]
        for name, age, T, ro, tr, win, oil in rows:
            txt.append(f"{name} ({age:g} Ma): {T:.0f} degC, {ro:.2f} %Ro, TR {100 * tr:.0f} % - {win}" +
                       (f", oil window from {oil:.1f} Ma" if oil is not None else ""))
        self._set_info("\n".join(txt))
        return self.well_result

    def _plot_well(self):
        r = self.well_result
        bh, th = r["burial"], r["thermal"]
        if self.plot_data.get("well") != r["well"]:
            self.plot_data = {"well": r["well"]}
        self.plot_data.update(burial=bh.surfaces(), temperature=th.surface_temperatures())
        self.plot_data["ro"] = [(n, a, MT.easy_ro(th.ages, T)) for n, a, T in self.plot_data["temperature"]]
        for p in (self.p_burial, self.p_temp, self.p_ro):
            p.clear()
            p.addLegend()
        for k, (name, _, d) in enumerate(self.plot_data["burial"]):
            pen = pg.mkPen(COLORS[k % len(COLORS)], width=2)
            self.p_burial.plot(bh.ages, d, pen=pen, name=name, connect="finite")
            self.p_temp.plot(th.ages, self.plot_data["temperature"][k][2], pen=pen, name=name, connect="finite")
            self.p_ro.plot(th.ages, self.plot_data["ro"][k][2], pen=pen, name=name, connect="finite")
        for p in (self.p_burial, self.p_temp, self.p_ro):
            p.getPlotItem().invertX(True)
            p.setLabel("bottom", "Age (Ma)")
        self.p_burial.getPlotItem().invertY(True)
        self.p_burial.setLabel("left", "Depth below sea level (m)")
        self.p_temp.setLabel("left", "degC")
        self.p_ro.setLabel("left", "%Ro")

    def calibrate(self, well, obs=None):
        """Fit the heat flow to the well's BHTs (LAS / DLIS headers), or to obs [(TVDSS, degC)] when given."""
        p = self.wcp()
        if p is None:
            raise RuntimeError("link a well project first")
        if self._widgets_live():
            self.settings_from_widgets()
        if obs is None:
            obs_recs = BM.well_observations(p, well, self.s)
            obs = [(r["tvdss"], r["T"]) for r in obs_recs if r["ok"]]
        else:
            obs_recs = [{"tvdss": z, "T": t, "ok": True} for z, t in obs]
        fit = BM.calibrate_well(p, well, self.table, self.s, obs if obs else None)
        self.fit = fit
        z, T = fit["model"].present_profile()
        self.plot_data["calibration"] = {"z": z, "T": T, "obs": fit["obs"],
                                         "rejected": [(r["tvdss"], r["T"]) for r in obs_recs if not r["ok"]]}
        self.p_cal.clear()
        self.p_cal.plot(T, z, pen=pg.mkPen((200, 0, 0), width=2))
        self.p_cal.plot(fit["obs"][:, 1], fit["obs"][:, 0], pen=None, symbol="o", symbolBrush=(0, 0, 0))
        rej = self.plot_data["calibration"]["rejected"]
        if rej:
            rj = np.array(rej)
            self.p_cal.plot(rj[:, 1], rj[:, 0], pen=None, symbol="x", symbolBrush=(150, 150, 150))
        self.p_cal.getPlotItem().invertY(True)
        self.p_cal.setLabel("bottom", "degC")
        self.p_cal.setLabel("left", "TVDSS (m)")
        self.run_well(well)
        self._set_info(f"{well}: heat flow fitted to {len(fit['obs'])} temperatures -> {fit['present_q']:.1f} mW/m2 "
                       f"today (rms {fit['rms']:.1f} degC, {len(rej)} implausible BHTs ignored)\n" + self._last_text)
        return fit

    def write_burial_csv(self, path):
        if not self.well_result:
            raise RuntimeError("run a well first")
        self.well_result["burial"].to_csv(path)
        return path

    def _burial_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Burial curves CSV", "burial.csv", "CSV (*.csv)")
        if path:
            self.write_burial_csv(path)

    # ------------------------------------------------------------------ maps
    def depth_grids(self):
        vm = self.section.velocity_model() if self.section is not None else None
        if vm is None:
            raise RuntimeError("needs a depth model: Interpret > Velocity model, Build, 'Use for depth display'")
        g = PW.depth_grids_from_vm(vm, self.table)
        if len(g) < 1:
            raise RuntimeError("no velocity-model layer matches an age-table surface (name or formation)")
        return g

    def run_map(self, surface, stride=None, threaded=True):
        if self._widgets_live():
            self.settings_from_widgets()
        grids = self.depth_grids()
        stride = int(stride or self.s.get("stride", 10))
        args = dict(table=self.table, surface=surface, stride=stride, water_depth=self.s["water_depth"],
                    heat_flow=self.s["heat_flow"], surface_t=self.s["surface_t"], step=self.s["step"],
                    kinetics=self.s["kinetics"])
        self._map_args = (surface, stride)
        state = {}

        def job(progress, cancel):
            state["res"] = PW.map_run(grids, progress=lambda f: progress(f, f"{100 * f:.0f} % of pseudo-wells"),
                                      cancel=cancel, **args)

        def done():
            self.map_result = state["res"]
            info = self.map_result["_info"]
            self._set_info(f"Map at {surface}: {info['valid']} of {info['columns']} pseudo-wells in "
                           f"{info['seconds']:.1f} s")
            self.show_map(self.m_out.currentText() if isinstance(self.m_out.currentText(), str) else "ro")
        if threaded and self.run_job_fn is not None:
            return self.run_job_fn("Basin map", job, done)
        job(lambda f, t: None, lambda: False)
        done()
        return self.map_result

    def show_map(self, output):
        if not self.map_result:
            raise RuntimeError("run a map first")
        g = self.map_result.get(output if output in MAP_OUTPUTS else "ro")
        if self.basemap is not None:
            self.basemap.show_grid(g)
        return g

    def export_cps3(self, path, output="ro"):
        from ..maps import write_cps3
        g = self.show_map(output)
        write_cps3(g, path)
        return path

    def _cps3(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Map as CPS-3", "basin_map.cps3", "CPS-3 (*.cps3 *.grd)")
        if path:
            self.export_cps3(path, self.m_out.currentText())

    def depth_changed(self):
        """Called after the dynamic depth model rebuilt (View > Auto-update depth model): re-runs the last map when
        'Re-run when the depth model changes' is ticked and its depth maps changed. Returns True when re-run."""
        on = self.auto_update if self.auto_update is not None else (self.auto.isChecked() is True)
        if not self.map_result or not self._map_args or not on:
            return False
        grids = self.depth_grids()
        surface, stride = self._map_args
        fp = PW.fingerprint(grids, surface, stride, self.s["heat_flow"], self.s["surface_t"])
        if fp == self.map_result["_info"]["fingerprint"]:
            return False
        self.run_map(surface, stride, threaded=False)
        return True

    # ------------------------------------------------------------------ Wheeler + events
    def colour_wheeler(self, prop, t_eval=0.0):
        if self.section is None:
            raise RuntimeError("no section panel")
        if self._widgets_live():
            self.settings_from_widgets()
        self.table.save(self.project)
        BM.save_settings(self.project, self.s)
        sp = self.section
        sp.basin_colour = None if prop not in WHEELER_PROPS[1:] else {"prop": prop, "t_eval": float(t_eval)}
        if sp.domain != "Wheeler (age)":
            sp.domain = "Wheeler (age)"
        sp.redraw()
        if sp._age is None:
            raise RuntimeError("the section could not be shown on the age axis - see the status message")
        if self.on_wheeler is not None:  # Live Map: Wheeler / basin section location, clickable (PLAN 31h)
            try:
                self.on_wheeler(prop)
            except Exception:
                pass
        return sp._data

    def event_chart(self, source_surface, trap=None, path=None, well=None):
        """Event chart from the age-table roles and the source's TR history (from the well run, else a run at
        the source surface's present depth from the depth model is not attempted)."""
        if source_surface not in self.table.names():
            raise ValueError(f"unknown surface {source_surface!r}")
        e = self.table.get(source_surface)
        e["role"] = "source"
        r = self.well_result if well is None else self.run_well(well)
        tr = None
        if r is not None:
            units = r["burial"].units
            idx = [i for i, u in enumerate(units) if not u["ghost"] and u["top_surface"] == source_surface]
            if idx:
                T = r["thermal"].unit_temperature(idx[0], 0.5)
                tr = MT.transformation_ratio(r["thermal"].ages, T, self.s["kinetics"])
            ages = r["thermal"].ages
        else:
            ages = np.array([0.0])
        self.chart = PW.event_chart(self.table, ages, tr, trap)
        if path:
            PW.plot_event_chart(self.chart, path)
        lines = [f"{d['element']}: {d['label']} {d['start']:.1f}-{d['end']:.1f} Ma" for d in self.chart["events"]]
        cm = self.chart["critical_moment"]
        self._set_info("Event chart" + (f" (critical moment {cm:.1f} Ma)" if cm is not None else
                                        " - run a well to add generation / expulsion") + ": " + "; ".join(lines))
        self.refresh_table()
        return self.chart

    def _chart_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Event chart PNG", "events.png", "PNG (*.png)")
        if path:
            self.event_chart(self.e_src.currentText(), (float(self.e_trap0.text()), float(self.e_trap1.text())), path)
