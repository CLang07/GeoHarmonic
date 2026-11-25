"""Velocity volume (advanced) depth conversion dialog (PLAN 37h): guided steps a-e on top of the item-17 velocity
model.  Logic in plain methods (tested under dev/fakeqt)."""
import numpy as np
from PySide6 import QtWidgets

from .. import velvolume as VV


class VelocityVolumeDialog(QtWidgets.QDialog):
    def __init__(self, project, parent=None, stack=None, on_change=None, run_job=None):
        super().__init__(parent)
        self.setWindowTitle("Velocity volume (advanced) depth conversion")
        self.project, self.stack, self.on_change, self.run_job_fn = project, stack, on_change, run_job
        self.model = None
        self.bt = None
        self.message = ""
        vm = project.display.get("velocity_model") or {}
        self.vol_name = (vm.get("key") or "")[3:] if str(vm.get("key", "")).startswith("3D:") else \
            (project.volumes[0]["name"] if project.volumes else None)
        self.layers = [tuple(x) for x in (vm.get("layers") or [])] or \
            [(h.name, None) for h in project.horizons.items]
        self.method, self.residual_method, self.trend = "idw", "tps", None
        self.mode = "proportional"
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.w_method = QtWidgets.QComboBox()
        self.w_method.addItems(["idw", "kriging"])
        self.w_res = QtWidgets.QComboBox()
        self.w_res.addItems(["tps", "kriging", "idw"])
        self.w_trend = QtWidgets.QComboBox()
        self.w_trend.addItems(["(none)"] + [v["name"] for v in project.volumes])
        self.w_mode = QtWidgets.QComboBox()
        self.w_mode.addItems(["proportional", "top", "base"])
        form.addRow("Interpolation between wells", self.w_method)
        form.addRow("Residual gridding (tops tie)", self.w_res)
        form.addRow("Stacking / RMS velocity trend", self.w_trend)
        form.addRow("Layering in each zone", self.w_mode)
        lay.addLayout(form)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("a. Well velocities", lambda *_: self.well_velocities()),
                         ("b-e. Build + tie to tops", lambda *_: self.build()),
                         ("Store volume", lambda *_: self.store()),
                         ("Blind test vs layer-cake", lambda *_: self.blind_test()),
                         ("Maps to map hub", lambda *_: self.add_maps())):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        self.table = QtWidgets.QTableWidget()
        lay.addWidget(self.table)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)

    def _say(self, msg):
        self.message = msg
        try:
            self.info.setText(msg)
        except Exception:
            pass

    def _read(self):
        for attr, w in (("method", self.w_method), ("residual_method", self.w_res), ("mode", self.w_mode)):
            try:
                v = w.currentText()
            except Exception:
                continue
            if isinstance(v, str) and v:
                setattr(self, attr, v)
        try:
            tv = self.w_trend.currentText()
            if isinstance(tv, str):
                self.trend = None if tv in ("", "(none)") else {"volume": tv, "kind": "rms"}
        except Exception:
            pass

    def make_model(self, **kw):
        self._read()
        if not self.vol_name or not self.layers:
            raise ValueError("needs a 3D volume and horizons (define the layers in Interpret > Velocity model)")
        modes = [self.mode] * (len(self.layers) + 1)
        d = dict(method=self.method, residual_method=self.residual_method, trend=self.trend, modes=modes)
        d.update(kw)
        return VV.VelocityVolumeModel(self.project, self.vol_name, self.layers, **d)

    def well_velocities(self, **kw):
        self.model = self.make_model(**kw)
        self.model.well_curves()
        rows = self.model.qc_table()
        self._fill(["well", "source", "corr", "v_mean", "flags"], rows)
        bad = [r["well"] for r in rows if r["flags"]]
        self._say(f"{len(rows)} wells with time-depth" + (f"; check {', '.join(bad)}" if bad else ""))
        return rows

    def build(self, **kw):
        if self.model is None or kw:
            self.model = self.make_model(**kw)
        self.model.build()
        rows = []
        for k, (h, fm) in enumerate(self.model.layers):
            names, r = self.model.res_points[k]
            rows.append({"horizon": h, "formation": fm or "", "wells": len(names),
                         "max |residual| m": round(float(np.max(np.abs(r))), 2) if len(r) else 0.0})
        self._fill(["horizon", "formation", "wells", "max |residual| m"], rows)
        self.project.display["velocity_volume"] = self.model.to_dict()
        self._say(f"Velocity volume built in {self.model.seconds:.1f} s; depth maps tie every top; calibrated volume "
                  "ready (Store volume / Maps to map hub)")
        if self.on_change is not None:
            self.on_change(self.model)
        return self.model

    def store(self, name="Velocity volume (calibrated)"):
        if self.model is None:
            self.build()
        nm = self.model.write(name)
        if self.project.path:
            self.project.save()
        self._say(f"Stored '{nm}' - show it on sections, export it as SEG-Y like any volume")
        return nm

    def blind_test(self):
        from ..velocity import VelocityModel
        from ..scenarios import blind_test
        if self.model is None:
            self.build()
        self.bt = self.model.blind_test()
        out = {"volume": self.bt["summary"]}
        vm = self.project.display.get("velocity_model")
        if vm:
            lc = VelocityModel.from_dict(self.project, vm)
            lc.build()
            out["layer-cake"] = blind_test(lc)["summary"]
        rows = []
        for mode, summ in out.items():
            for h, s in summ.items():
                rows.append({"mode": mode, "horizon": h, "n": s.get("n"), "rms m": round(float(s.get("rms") or 0), 2)})
        self._fill(["mode", "horizon", "n", "rms m"], rows)
        self._say("Leave-one-well-out: error = actual - predicted top")
        return out

    def add_maps(self):
        if self.model is None or self.stack is None:
            return []
        ids = []
        for k, (h, _) in enumerate(self.model.layers):
            for g in (self.model.depth_grid(k), self.model.layer_velocity_grid(k), self.model.residual_grid(k)):
                name = f"VV: {g.label}"
                old = next((l for l in self.stack.layers if l["kind"] == "grid" and l["name"] == name), None)
                if old is not None:
                    self.stack.remove(old["id"])
                lid, _ = self.stack.add_grid(g, name, contours=False)
                ids.append(lid)
        self.stack.save()
        self._say(f"{len(ids)} grids added to the map hub (depth, interval velocity, residual per horizon)")
        return ids

    def refresh(self):
        """Dynamic update (main window tick): re-tie to new / moved tops."""
        if self.model is None or not getattr(self.model, "_built", False):
            return None
        rep = self.model.refresh_tops()
        if rep:
            self._say(rep["message"])
            if self.on_change is not None:
                self.on_change(self.model)
        return rep

    def _fill(self, cols, rows):
        try:
            self.table.setColumnCount(len(cols))
            self.table.setHorizontalHeaderLabels(cols)
            self.table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for c, k in enumerate(cols):
                    v = row.get(k)
                    self.table.setItem(r, c, QtWidgets.QTableWidgetItem("" if v is None else str(v)))
        except Exception:
            pass
