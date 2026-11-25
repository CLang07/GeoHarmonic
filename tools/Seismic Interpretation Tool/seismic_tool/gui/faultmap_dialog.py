"""Automated fault mapping dialog (PLAN 32): fault likelihood volumes, skin extraction, fault table + filters,
stereonet, merge assistant, faults with editable sticks, fault polygons on a horizon for the map hub / Live Map.
Logic in plain methods (tested under dev/fakeqt); the heavy steps run through the main window's run_job."""
import numpy as np
from PySide6 import QtWidgets

from .. import faultlik as FL
from .. import faultskin as FS


class FaultMappingDialog(QtWidgets.QDialog):
    def __init__(self, project, vol_name, parent=None, run_job=None, on_faults=None, on_layer=None):
        super().__init__(parent)
        self.setWindowTitle(f"Automated fault mapping - {vol_name}")
        self.project, self.vol_name = project, vol_name
        self.run_job_fn = run_job
        self.on_faults = on_faults  # callable(names): fault dock / section redraw / map bus
        self.on_layer = on_layer  # callable(layer id): map hub redraw (+ Live Map)
        self.skins = []
        self.filtered = []
        self.velocity = 2500.0
        self.threshold, self.min_size, self.max_angle = 0.5, 200, 20.0
        self.filters = {}
        self.message = ""
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.w_thr = QtWidgets.QDoubleSpinBox()
        self.w_thr.setRange(0.05, 0.99)
        self.w_thr.setValue(self.threshold)
        self.w_size = QtWidgets.QSpinBox()
        self.w_size.setRange(10, 10_000_000)
        self.w_size.setValue(self.min_size)
        self.w_vel = QtWidgets.QDoubleSpinBox()
        self.w_vel.setRange(1000, 8000)
        self.w_vel.setValue(self.velocity)
        form.addRow("TFL threshold", self.w_thr)
        form.addRow("Minimum fault size (points)", self.w_size)
        form.addRow("Velocity for dips (m/s)", self.w_vel)
        lay.addLayout(form)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("1. Fault likelihood volumes", lambda *_: self.compute()),
                         ("2. Extract faults", lambda *_: self.extract()),
                         ("3. Create faults (sticks)", lambda *_: self.create_faults()),
                         ("Fault polygons on active horizon", lambda *_: self.polygons_dialog()),
                         ("2D lines: fault cuts + correlation", lambda *_: self.lines_2d())):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        self.table = QtWidgets.QTableWidget()
        lay.addWidget(self.table)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)

    def _params(self):
        """Widget values -> attributes (only real numbers: under the Qt mock the attributes set by tests stay)."""
        for attr, w, typ in (("threshold", self.w_thr, float), ("min_size", self.w_size, int),
                             ("velocity", self.w_vel, float)):
            try:
                v = w.value()
            except Exception:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                setattr(self, attr, typ(v))

    def _say(self, msg):
        self.message = msg
        try:
            self.info.setText(msg)
        except Exception:
            pass

    # ------------------------------------------------------------------ steps
    def compute(self, threaded=True):
        names = []

        def fn(progress, cancel):
            names.extend(FL.compute_fault_volumes(self.project, self.vol_name, velocity=self.velocity,
                                                  progress=progress, cancel=cancel))
            return names

        def done():
            if self.project.path:
                self.project.save()
            self._say(f"Fault volumes: {', '.join(names)}")
        if threaded and self.run_job_fn is not None:
            self.run_job_fn("Fault likelihood", fn, done)
            return None
        fn(None, None)
        done()
        return names

    def extract(self):
        self._params()
        self.skins = FS.extract_from_volumes(self.project, self.vol_name, self.threshold, self.min_size,
                                             velocity=self.velocity)
        rgt = self._rgt()
        if rgt is not None:
            for s in self.skins:
                FS.throw_from_rgt(s, rgt)
        self.apply_filters()
        self._say(f"{len(self.skins)} fault skins (TFL > {self.threshold:g}, >= {self.min_size} points)" +
                  (" - throws from the RGT volume" if rgt is not None else ""))
        return self.skins

    def _rgt(self):
        import os
        from ..rgt.solve import RGTVolume
        vol = self.project.volume(self.vol_name)
        folder = os.path.join(vol.store, "rgt")
        return RGTVolume(folder) if RGTVolume.exists(folder) else None

    def frame(self):
        vol = self.project.volume(self.vol_name)
        ei, ej = FL.index_frame(vol)
        return ei, ej, vol.dt * 1000.0

    def apply_filters(self, **filters):
        """filters: dip=(lo, hi), strike=(lo, hi), size=(lo, hi), fl=(lo, hi), throw=(lo, hi), poly=[(x, y)]
        (stereonet brush).  Empty = all skins."""
        if filters:
            self.filters = {k: v for k, v in filters.items() if v is not None}
        ei, ej, dt = self.frame()
        self.filtered = FS.filter_skins(self.skins, ei, ej, dt, velocity=self.velocity, **self.filters) \
            if self.filters else list(self.skins)
        self._fill()
        return self.filtered

    def rows(self):
        ei, ej, dt = self.frame()
        return FS.skin_table(self.filtered, ei, ej, dt, self.velocity)

    def _fill(self):
        rows = self.rows()
        cols = ["name", "points", "strike", "dip", "fl", "throw_max"]
        try:
            self.table.setColumnCount(len(cols))
            self.table.setHorizontalHeaderLabels(cols)
            self.table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for c, k in enumerate(cols):
                    self.table.setItem(r, c, QtWidgets.QTableWidgetItem("" if row[k] is None else str(row[k])))
        except Exception:
            pass
        return rows

    def stereonet(self):
        ei, ej, dt = self.frame()
        sd = np.array([s.strike_dip(ei, ej, dt, self.velocity) for s in self.filtered]) if self.filtered else \
            np.zeros((0, 2))
        return FS.stereonet(sd[:, 0], sd[:, 1], weights=[len(s) for s in self.filtered])

    def stereonet_png(self, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        net = self.stereonet()
        ei, ej, dt = self.frame()
        sts = [s.strike_dip(ei, ej, dt, self.velocity)[0] for s in self.filtered]
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(121)
        ax.contourf(net["gx"], net["gy"], net["density"], levels=8, cmap="Blues")
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False))
        ax.plot(net["x"], net["y"], "k.", ms=3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Poles (lower hemisphere, equal area)")
        ax2 = fig.add_subplot(122, projection="polar")
        edges, cnt = FS.rose(sts, 18, [len(s) for s in self.filtered])
        th = np.radians(np.concatenate([edges[:-1], edges[:-1] + 180.0]))
        ax2.bar(th, np.concatenate([cnt, cnt]), width=np.radians(10.0), color="tab:blue")
        ax2.set_theta_zero_location("N")
        ax2.set_theta_direction(-1)
        ax2.set_title("Strike rose")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    def suggestions(self, max_gap=6.0, max_angle=10.0):
        return FS.merge_candidates(self.skins, max_gap, max_angle)

    def merge(self, a, b):
        self.skins = FS.merge_skins(self.skins, a, b)
        self.apply_filters()
        return self.skins

    def split(self, ident, axis=2, value=None):
        self.skins = FS.split_skin(self.skins, ident, axis, value)
        self.apply_filters()
        return self.skins

    def rename(self, ident, name):
        for s in self.skins:
            if s.id == ident:
                s.name = str(name)
        self._fill()

    def delete_small(self, min_points):
        self.skins = [s for s in self.skins if len(s) >= int(min_points)]
        self.apply_filters()
        return self.skins

    def create_faults(self, ids=None, every=5):
        """Filtered (or the given) skins -> item-2 faults with sticks every `every` lines; replaces older auto faults
        of the same names.  Returns the fault names."""
        vol = self.project.volume(self.vol_name)
        fs = self.project.faults
        pick = [s for s in self.filtered if ids is None or s.id in ids]
        names = []
        for s in pick:
            if s.name in fs.names():
                fs.delete(s.name)
            f = FS.skin_to_fault(s, vol, fs, every)
            names.append(f.name)
        if self.project.path:
            self.project.save()
        if self.on_faults is not None:
            self.on_faults(names)
        self._say(f"{len(names)} faults created with sticks every {every} lines (edit them like manual faults)")
        return names

    def polygons(self, horizon, stack=None):
        """Heave polygons of the filtered skins on a horizon -> hub polygons layer 'Fault polygons: <horizon>' with
        style fault (gridding barriers, downthrown ticks).  Returns (layer id or None, items)."""
        vol = self.project.volume(self.vol_name)
        h = self.project.horizons.get(horizon)
        surf = h.get(f"3D:{self.vol_name}") if h is not None else None
        if surf is None:
            raise ValueError(f"{horizon} has no picks on {self.vol_name}")
        items = FS.polygons_layer_items(self.filtered, surf, vol)
        lid = None
        if stack is not None:
            name = f"Fault polygons: {horizon}"
            lay = next((l for l in stack.layers if l["kind"] == "polygons" and l["name"] == name), None)
            if lay is None:
                lid = stack.add("polygons", name, src={"polygons": items}, style={"fault": True, "color": (180, 0, 0)})
            else:
                lid = lay["id"]
                lay["src"]["polygons"] = items
            stack.save()
            if self.on_layer is not None:
                self.on_layer(lid)
        self._say(f"{len(items)} fault polygons on {horizon}")
        return lid, items

    def lines_2d(self, lines=None, create=True):
        """PLAN 32g: fault cuts on the 2D lines, linked across intersecting lines into named faults."""
        cuts = FS.project_cuts(self.project, lines, self.threshold)
        groups = FS.link_cuts(cuts)
        names = []
        if create and cuts:
            fs = self.project.faults
            for n in [n for n in fs.names() if n.startswith("Auto 2D F")]:
                fs.delete(n)
            names = FS.cuts_to_faults(cuts, groups, fs, min_lines=2)
            if self.project.path:
                self.project.save()
            if self.on_faults is not None:
                self.on_faults(names)
        self._say(f"2D lines: {len(cuts)} fault cuts -> {len(names)} faults seen on 2+ lines")
        return cuts, groups, names

    def polygons_dialog(self):
        hs = [h.name for h in self.project.horizons.items]
        if not hs:
            self._say("No horizons")
            return None
        name, ok = QtWidgets.QInputDialog.getItem(self, "Fault polygons", "Horizon:", hs, 0, False)
        if ok:
            st = getattr(getattr(self.parent(), "basemap", None), "stack", None)
            return self.polygons(name, st)
        return None
