"""Structural framework + 3D grid dialog (PLAN 35): horizons + faults of a 3D survey -> framework per fault block
(time, then depth through the current depth model) -> stair-stepped 3D grid -> GRDECL / ZMAP+.  Logic in plain
methods (tested under dev/fakeqt)."""
import numpy as np
from PySide6 import QtWidgets

from .. import framework as FW


class FrameworkDialog(QtWidgets.QDialog):
    def __init__(self, project, vol_name, parent=None, depth_model_fn=None, bus=None):
        super().__init__(parent)
        self.setWindowTitle(f"Structural framework + 3D grid - {vol_name}")
        self.project, self.vol_name = project, vol_name
        self.vol = project.volume(vol_name)
        self.depth_model_fn = depth_model_fn
        self.bus = bus
        self.horizons = [h.name for h in project.horizons.items if h.data.get(f"3D:{vol_name}") is not None]
        self.faults = [f.name for f in project.faults.items if len(f.sticks) >= 2]
        self.rules = {}
        self.layering = None
        self.cell = None
        self.model_t = self.model = self.grid = None
        self.prop_dialog = None
        self.message = ""
        lay = QtWidgets.QVBoxLayout(self)
        self.info = QtWidgets.QLabel(f"Horizons: {', '.join(self.horizons) or '-'}; faults: {', '.join(self.faults) or '-'}")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("1. Framework (time)", lambda *_: self.build()),
                         ("2. Depth (current depth model)", lambda *_: self.depth()),
                         ("3. 3D grid", lambda *_: self.make_grid()),
                         ("Export GRDECL...", lambda *_: self._export_dialog()),
                         ("4. Properties + volumetrics...", lambda *_: self.properties())):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        if bus is not None:
            bus.subscribe("framework", self._on_event, topics="data")

    def _say(self, msg):
        self.message = msg
        try:
            self.info.setText(msg)
        except Exception:
            pass

    def _points(self, name, max_points=4000):
        g = np.asarray(self.project.horizons.get(name).data[f"3D:{self.vol_name}"], float)
        I, J = np.nonzero(np.isfinite(g))
        step = max(1, int(np.ceil(np.sqrt(len(I) / max_points))))
        sel = (I % step == 0) & (J % step == 0)
        I, J = I[sel], J[sel]
        x, y = self.vol.xy(self.vol.inlines[I], self.vol.xlines[J])
        return np.ravel(x).astype(float), np.ravel(y).astype(float), g[I, J]

    def _traces(self):
        """Map traces of the faults on the horizons (union), used as fault-block barriers."""
        from ..maps import horizon_grid
        out = []
        for fn in self.faults:
            f = self.project.faults.get(fn)
            for hn in self.horizons:
                hg = horizon_grid(self.project, self.project.horizons.get(hn), f"3D:{self.vol_name}")
                tr = FW.fault_trace(f, hg)
                if len(tr) >= 2:
                    tr = FW.extend_trace(tr, 3.0 * (self.cell or 50.0))  # reach past the survey edge / grid nodes
                    out.append((f"{fn} @ {hn}", (tr[:, 0], tr[:, 1])))
                    break
        return out

    def build(self, cell=None):
        if len(self.horizons) < 2:
            raise ValueError("needs two or more horizons picked on the 3D survey")
        xs, ys = self.vol.outline()
        from ..faultlik import index_frame
        ei, ej = index_frame(self.vol)
        self.cell = cell or self.cell or 2.0 * float(min(np.linalg.norm(ei), np.linalg.norm(ej)))
        geom = FW.Geometry.around(xs, ys, self.cell)
        hz = [(n, self._points(n), self.rules.get(n, "conformable")) for n in self.horizons]
        self.model_t = FW.FrameworkModel(geom, hz, self._traces(), "tps", "time").build()
        self.model = self.model_t
        self._say(f"Framework: {len(hz)} horizons, {self.model_t.nblocks} fault blocks, grid {geom.nx} x {geom.ny} "
                  f"({self.cell:g} m); crossings fixed {self.model_t.report['crossings_before']}")
        return self.model_t

    def depth(self):
        if self.model_t is None:
            self.build()
        dm = self.depth_model_fn() if self.depth_model_fn is not None else None
        if dm is None:
            self._say("No depth model yet (Interpret > Velocity model or Velocity volume) - framework stays in time")
            return None
        self.model = self.model_t.to_depth(FW.depth_fn_from_model(dm))
        self._say(f"Framework in depth through the {type(getattr(dm, 'vm', dm)).__name__}")
        return self.model

    def make_grid(self, layering=None):
        if self.model is None:
            self.build()
        nz = len(self.model.grids) - 1
        self.layering = layering or self.layering or [("proportional", 10)] * nz
        self.grid = FW.Grid3D(self.model, self.layering)
        q = self.grid.quality()
        self._say(f"3D grid {self.grid.nx} x {self.grid.ny} x {self.grid.nz} ({self.model.domain}), {q['active']} "
                  f"active cells, {q['fault_blocks']} fault blocks, zero-thickness {q['zero_thickness']}, "
                  f"negative {q['negative_thickness']}")
        return q

    def properties(self):
        """PLAN 36 dialog on this grid."""
        from .propmodel_dialog import PropertyDialog
        if self.grid is None:
            self.make_grid()
        st = getattr(getattr(self.parent(), "basemap", None), "stack", None)
        dlg = PropertyDialog(self.project, self.grid, self, self.bus, st)
        dlg.show()
        self.prop_dialog = dlg
        return dlg

    def export(self, path, surfaces_folder=None):
        if self.grid is None:
            self.make_grid()
        FW.write_grdecl(path, self.grid, title=f"GeoHarmonic {self.project.name} {self.vol_name} ({self.model.domain})")
        out = [path]
        if surfaces_folder and not self.model.geom.rotation:
            out += FW.write_surfaces(self.model, surfaces_folder)
        return out

    def _export_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "GRDECL", "", "Eclipse grid (*.grdecl *.GRDECL)")
        if path:
            self._say(f"Written {', '.join(self.export(path))}")

    def _on_event(self, ev):
        """35c: a new velocity model -> the depth framework and grid are rebuilt."""
        if ev.get("kind") == "velocities" and self.model is not None and self.model.domain == "depth":
            self.depth()
            if self.grid is not None:
                self.make_grid()
            self._say(self.message + " (rebuilt after a velocity-model change)")
