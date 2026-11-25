"""Seismic > Conditioning (PLAN 41): chain of steps + presets, live preview (original | conditioned | difference with
one colour scale and linked zoom, metrics, spectra, warnings), protect zones, full-volume run in the background.
Processing is `seismic_tool.conditioning` (tested); this is the Qt layer."""
import json

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from .. import conditioning as C
from . import colormaps

LABELS = {"spikes": "Spike removal", "bandpass": "Bandpass (+ spectral balancing)", "fxdecon": "FX deconvolution",
          "sof": "Structure-oriented filter", "footprint": "Footprint removal", "agc": "AGC (display only)"}
PROTECT_KEY = "conditioning_protect"


class ConditioningDialog(QtWidgets.QDialog):
    def __init__(self, project, section=None, parent=None, run_job=None, on_volume=None, bus=None):
        super().__init__(parent)
        self.setWindowTitle("Seismic conditioning")
        self.project = project
        self.section = section  # the main window's SectionPanel (current inline / crossline / time slice)
        self.run_job_fn = run_job
        self.on_volume = on_volume
        self.bus = bus
        self.chain = C.preset("Interpretation")
        self.zones = list((getattr(project, "display", {}) or {}).get(PROTECT_KEY) or [])
        self.last = {}
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        lay.addLayout(left, 0)
        f = QtWidgets.QFormLayout()
        self.vol = QtWidgets.QComboBox()
        for v in project.volumes:
            h = C.history_text(project, v["name"])
            self.vol.addItem(v["name"] + (f"   [{h}]" if h else ""), v["name"])
        f.addRow("Input volume", self.vol)
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(C.preset_names(project))
        f.addRow("Preset", self.preset)
        left.addLayout(f)
        self.steps = QtWidgets.QListWidget()
        left.addWidget(self.steps, 1)
        row = QtWidgets.QHBoxLayout()
        self.add_combo = QtWidgets.QComboBox()
        self.add_combo.addItems(list(C.STEP_TYPES))
        for text, fn in (("Add", self.add_step), ("Remove", self.remove_step), ("Up", lambda: self.move_step(-1)),
                         ("Down", lambda: self.move_step(1))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda *_, fn=fn: fn())
            row.addWidget(b)
        row.insertWidget(0, self.add_combo)
        left.addLayout(row)
        left.addWidget(QtWidgets.QLabel("Parameters of the selected step (JSON):"))
        self.params = QtWidgets.QPlainTextEdit()
        self.params.setMaximumHeight(150)
        left.addWidget(self.params)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("Apply parameters", self.apply_params), ("Save preset...", self.save_preset_dialog)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda *_, fn=fn: fn())
            row.addWidget(b)
        left.addLayout(row)
        left.addWidget(QtWidgets.QLabel("Protect geology (reduce / skip filtering):"))
        self.zone_list = QtWidgets.QListWidget()
        self.zone_list.setMaximumHeight(80)
        left.addWidget(self.zone_list)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("Add zone...", self.add_zone_dialog), ("Remove zone", self.remove_zone)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda *_, fn=fn: fn())
            row.addWidget(b)
        left.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.diff_chk = QtWidgets.QCheckBox("Also write the difference volume")
        self.segy_chk = QtWidgets.QCheckBox("Export SEG-Y")
        row.addWidget(self.diff_chk)
        row.addWidget(self.segy_chk)
        left.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("Preview", self.preview), ("Run full volume...", self.run_dialog), ("Close", self.accept)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda *_, fn=fn: fn())
            row.addWidget(b)
        left.addLayout(row)
        right = QtWidgets.QVBoxLayout()
        lay.addLayout(right, 1)
        self.glw = pg.GraphicsLayoutWidget()
        self.views = []
        self.imgs = []
        for k, title in enumerate(("Original", "Conditioned", "Difference")):
            pl = self.glw.addPlot(row=0, col=k, title=title)
            pl.invertY(True)
            im = pg.ImageItem()
            pl.addItem(im)
            if self.views:
                pl.setXLink(self.views[0])
                pl.setYLink(self.views[0])
            self.views.append(pl)
            self.imgs.append(im)
        self.spec = self.glw.addPlot(row=1, col=0, colspan=3, title="Amplitude spectrum (before / after)")
        self.spec.setLabel("bottom", "Hz")
        right.addWidget(self.glw, 1)
        self.metrics = QtWidgets.QLabel("Preview: shows the current line of the seismic window")
        self.metrics.setWordWrap(True)
        right.addWidget(self.metrics)
        self.steps.currentRowChanged.connect(lambda r: self.show_params(r))
        self.steps.itemChanged.connect(lambda it: self._toggle(it))
        self.preset.currentTextChanged.connect(lambda n: self.load_preset(n))
        self.refresh_steps()
        self.refresh_zones()
        self.resize(1400, 820)

    # ------------------------------------------------------------ chain editing
    def refresh_steps(self, select=None):
        self.steps.blockSignals(True)
        self.steps.clear()
        for s in self.chain:
            it = QtWidgets.QListWidgetItem(LABELS.get(s["type"], s["type"]))
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if s["on"] else QtCore.Qt.Unchecked)
            self.steps.addItem(it)
        self.steps.blockSignals(False)
        if select is not None:
            self.steps.setCurrentRow(select)
        return [s["type"] for s in self.chain]

    def _toggle(self, it):
        r = self.steps.row(it)
        if 0 <= r < len(self.chain):
            self.chain[r]["on"] = it.checkState() == QtCore.Qt.Checked

    def load_preset(self, name):
        try:
            self.chain = C.preset(name, self.project)
        except KeyError:
            return None
        return self.refresh_steps()

    def add_step(self, kind=None):
        kind = kind or self.add_combo.currentText() or "sof"
        self.chain.append({"type": kind, "on": True, "params": {}})
        return self.refresh_steps(len(self.chain) - 1)

    def remove_step(self, row=None):
        r = self.steps.currentRow() if row is None else row
        if isinstance(r, int) and 0 <= r < len(self.chain):
            del self.chain[r]
        return self.refresh_steps()

    def move_step(self, d, row=None):
        r = self.steps.currentRow() if row is None else row
        if not isinstance(r, int) or not (0 <= r < len(self.chain)) or not (0 <= r + d < len(self.chain)):
            return None
        self.chain[r], self.chain[r + d] = self.chain[r + d], self.chain[r]
        return self.refresh_steps(r + d)

    def show_params(self, r):
        if isinstance(r, int) and 0 <= r < len(self.chain):
            self.params.setPlainText(json.dumps(C.params(self.chain[r]), indent=1))

    def apply_params(self, row=None, text=None):
        r = self.steps.currentRow() if row is None else row
        if not isinstance(r, int) or not (0 <= r < len(self.chain)):
            return None
        try:
            p = json.loads(text if text is not None else self.params.toPlainText())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Parameters", f"Not valid JSON: {e}")
            return None
        d = C.DEFAULTS[self.chain[r]["type"]]
        self.chain[r]["params"] = {k: v for k, v in p.items() if k in d or k == "write"}
        return self.chain[r]["params"]

    def save_preset_dialog(self, name=None):
        if name is None:
            name, ok = QtWidgets.QInputDialog.getText(self, "Save preset", "Preset name:")
            if not ok or not name:
                return None
        C.save_preset(self.project, name, self.chain)
        if self.preset.findText(name) < 0:
            self.preset.addItem(name)
        return name

    # ------------------------------------------------------------ protect zones (41j)
    def refresh_zones(self):
        self.zone_list.clear()
        for z in self.zones:
            poly = z.get("polygon")
            where = f"IL {poly[0][0]:g}-{poly[2][0]:g}, XL {poly[0][1]:g}-{poly[2][1]:g}" if poly else "whole survey"
            self.zone_list.addItem(f"{where}, {z.get('t0', 0):g}-{z.get('t1', 0):g} ms, strength x {z.get('weight', 0):g}")
        if getattr(self.project, "display", None) is not None:
            self.project.display[PROTECT_KEY] = list(self.zones)

    def add_zone(self, il0, il1, xl0, xl1, t0, t1, weight=0.0):
        il0, il1, xl0, xl1 = float(il0), float(il1), float(xl0), float(xl1)  # JSON-safe (project file)
        z = {"polygon": [[il0, xl0], [il0, xl1], [il1, xl1], [il1, xl0]], "t0": float(t0), "t1": float(t1),
             "weight": float(weight)}
        self.zones.append(z)
        self.refresh_zones()
        self.publish_zones()
        return z

    def add_zone_dialog(self):
        vals = []
        for lab, d in (("First inline", 0), ("Last inline", 0), ("First crossline", 0), ("Last crossline", 0),
                       ("Top (ms)", 1200), ("Base (ms)", 1350), ("Filter strength inside (0 = skip)", 0)):
            v, ok = QtWidgets.QInputDialog.getDouble(self, "Protect zone", lab, d, -1e9, 1e9, 2)
            if not ok:
                return None
            vals.append(v)
        return self.add_zone(*vals)

    def remove_zone(self):
        r = self.zone_list.currentRow()
        if isinstance(r, int) and 0 <= r < len(self.zones):
            del self.zones[r]
            self.refresh_zones()
            self.publish_zones()

    def publish_zones(self):
        """Protect zones as map polygons (Live Map / 3D) through the bus."""
        if self.bus is None:
            return None
        name = self.vol.currentData() or (self.project.volumes[0]["name"] if self.project.volumes else None)
        polys = []
        try:
            v = self.project.volume(name)
            for z in self.zones:
                if z.get("polygon"):
                    xy = [tuple(float(c) for c in v.xy(float(a), float(b))) for a, b in z["polygon"]]
                    polys.append({"xy": xy, "t0": z["t0"], "t1": z["t1"], "weight": z["weight"]})
        except Exception:
            pass
        return self.bus.notify("polygons", ids=["conditioning protect"], source="conditioning", polygons=polys,
                               style="protect")

    # ------------------------------------------------------------ preview (41i)
    def _vol_name(self):
        d = self.vol.currentData()
        if isinstance(d, str) and d:
            return d
        if self.section is not None and self.section.source and self.section.source[0] == "3D":
            return self.section.source[1]
        return self.project.volumes[0]["name"] if self.project.volumes else None

    def preview_block(self, mode=None, number=None):
        """Read the current line with a halo, condition it, return (original 2D, conditioned 2D, dt_ms)."""
        name = self._vol_name()
        v = self.project.volume(name)
        sec = self.section
        mode = mode or (sec.mode if sec is not None and sec.mode in ("Inline", "Crossline", "Time slice") else "Inline")
        number = number if number is not None else (sec.number if sec is not None and sec.number is not None
                                                    else v.inlines[v.nil // 2])
        dt = v.dt * 1000.0
        local = [s for s in self.chain if s["type"] not in C.GLOBAL_STEPS]
        h = min(C.chain_halo(local), 12)  # a small 3D halo is enough for the preview
        if mode == "Time slice":
            k = int(np.argmin(np.abs(np.asarray(v.times) - float(number))))
            k0, k1 = max(0, k - 40), min(v.ns, k + 41)
            blk = np.asarray(v.ilxl[:, :, k0:k1], np.float32) * v.inv
            zones = self.zones
            w = C.protect_weights(blk.shape, v.inlines, v.xlines, np.asarray(v.times)[k0:k1], zones) if zones else None
            out = C.apply_chain(blk, self.chain, dt, w, t0_ms=float(v.times[k0]), include_global=True,
                                il_total=v.nil)
            return blk[:, :, k - k0], out[:, :, k - k0], dt
        if mode == "Inline":
            i = v.il_index(number)
            a0, b0 = max(0, i - h), min(v.nil, i + h + 1)
            blk = np.asarray(v.ilxl[a0:b0], np.float32) * v.inv
            ils = np.asarray(v.inlines)[a0:b0]
            w = C.protect_weights(blk.shape, ils, v.xlines, v.times, self.zones) if self.zones else None
            out = C.apply_chain(blk, local, dt, w, offset=(a0, 0), include_global=False, il_total=v.nil)
            return blk[i - a0], out[i - a0], dt
        j = v.xl_index(number)
        a0, b0 = max(0, j - h), min(v.nxl, j + h + 1)
        blk = np.transpose(np.asarray(v.ilxl[:, a0:b0], np.float32) * v.inv, (1, 0, 2))
        w = C.protect_weights(blk.shape, np.asarray(v.xlines)[a0:b0], v.inlines, v.times,
                              [dict(z, polygon=[(b, a) for a, b in z["polygon"]]) if z.get("polygon") else z
                               for z in self.zones]) if self.zones else None
        out = C.apply_chain(blk, local, dt, w, include_global=False)
        return blk[j - a0], out[j - a0], dt

    def preview(self, mode=None, number=None):
        try:
            o, c, dt = self.preview_block(mode, number)
        except Exception as e:
            self.metrics.setText(f"Preview failed: {e}")
            return None
        d = c - o
        q = C.qc_metrics(o, c, dt)
        lim = float(np.percentile(np.abs(o), 99)) or 1.0
        lut = colormaps.lut(self.section.cmap_combo.currentText() if self.section is not None else
                            "Seismic (red-white-blue)") if self.section is not None else colormaps.lut(colormaps.NAMES[0])
        for im, arr in zip(self.imgs, (o, c, d)):  # same colour scale for all three
            im.setImage(np.asarray(arr, np.float32).T, levels=(-lim, lim))
            im.setLookupTable(lut)
        self.spec.clear()
        self.spec.plot(q["freqs"], q["spec_before"], pen=pg.mkPen((120, 120, 120), width=2), name="before")
        self.spec.plot(q["freqs"], q["spec_after"], pen=pg.mkPen((220, 60, 30), width=2), name="after")
        txt = f"RMS removed {q['rms_removed_pct']:.1f} %   signal leakage {q['leakage']:.2f}"
        if q["warnings"]:
            txt += "   WARNING: " + "; ".join(q["warnings"])
        self.metrics.setText(txt)
        self.last = dict(q, original=o, conditioned=c)
        return q

    # ------------------------------------------------------------ full volume (41k)
    def run_full(self, out_name, diff=False, segy_path=None, threaded=True):
        name = self._vol_name()
        chain, zones = [dict(s) for s in self.chain], list(self.zones)
        result = {}

        def fn(progress=None, cancel=None):
            result.update(C.run_volume(self.project, name, chain, out_name, progress=progress, cancel=cancel,
                                       diff=diff, zones=zones))
            if segy_path:
                from ..inversion.common import export_segy
                export_segy(self.project.volume(result["name"]), segy_path, note=f"conditioned: "
                            f"{C.history_text(self.project, result['name'])}")
            return result

        def done():
            if self.bus is not None:
                self.bus.notify("volumes", ids=[result.get("name")] + ([result["diff"]] if result.get("diff") else []),
                                source="conditioning", history=C.history_text(self.project, result.get("name")))
            if self.on_volume:
                self.on_volume(result.get("name"))
            self.metrics.setText(f"Wrote {result.get('name')} in {result.get('seconds', 0):.0f} s"
                                 f"{' + difference volume' if result.get('diff') else ''}; "
                                 f"{len(result.get('spikes', []))} spikes replaced")
        if threaded and self.run_job_fn is not None:
            self.run_job_fn(f"Conditioning {name}", fn, done)
            return None
        fn()
        done()
        return result

    def run_dialog(self):
        name = self._vol_name()
        out, ok = QtWidgets.QInputDialog.getText(self, "Run full volume", "New volume name:",
                                                 text=f"{name} [{self.preset.currentText() or 'conditioned'}]")
        if not ok or not out:
            return None
        segy = None
        if self.segy_chk.isChecked():
            segy, _ = QtWidgets.QFileDialog.getSaveFileName(self, "SEG-Y", "", "SEG-Y (*.sgy *.segy)")
        return self.run_full(out, bool(self.diff_chk.isChecked()), segy or None)
