"""Main window: data tree + basemap docks, section viewer in the centre."""
import os
import threading
import traceback

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from .. import __version__  # noqa: E402
from ..project import EXT, SeisProject  # noqa: E402
from ..segy import IL_XL_CANDIDATES, SegyFile  # noqa: E402
from ..volume import import_3d  # noqa: E402
from .basemap import Basemap  # noqa: E402
from .fault_dock import FaultDock  # noqa: E402
from .horizon_dock import HorizonDock  # noqa: E402
from .section_panel import SectionPanel  # noqa: E402


class Import3DDialog(QtWidgets.QDialog):
    """Pick a 3D SEG-Y, confirm header bytes, choose a name."""

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import 3D SEG-Y")
        self.path = path
        f = SegyFile(path)
        guess = f.guess_il_xl()
        form = QtWidgets.QFormLayout(self)
        self.name = QtWidgets.QLineEdit(os.path.splitext(os.path.basename(path))[0])
        form.addRow("Volume name", self.name)
        info = (f"{f.ntraces:,} traces, {f.ns} samples @ {f.dt * 1000:g} ms, format {f.fmt}, "
                f"{os.path.getsize(path) / 1e9:.2f} GB")
        form.addRow("File", QtWidgets.QLabel(info))
        self.il = QtWidgets.QSpinBox()
        self.xl = QtWidgets.QSpinBox()
        self.xb = QtWidgets.QSpinBox()
        self.yb = QtWidgets.QSpinBox()
        for s in (self.il, self.xl, self.xb, self.yb):
            s.setRange(1, 237)
        self.il.setValue(guess[0] if guess else 189)
        self.xl.setValue(guess[1] if guess else 193)
        self.xb.setValue(181)
        self.yb.setValue(185)
        form.addRow("Inline byte", self.il)
        form.addRow("Crossline byte", self.xl)
        form.addRow("X coordinate byte", self.xb)
        form.addRow("Y coordinate byte", self.yb)
        note = ("Header bytes found automatically." if guess else
                "Could not detect inline/crossline bytes - common pairs: " +
                ", ".join(f"{a}/{b}" for a, b in IL_XL_CANDIDATES[:4]))
        form.addRow(QtWidgets.QLabel(note))
        hdr = QtWidgets.QPlainTextEdit(f.text_header)
        hdr.setReadOnly(True)
        hdr.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        hdr.setMinimumSize(640, 220)
        form.addRow("Text header", hdr)
        free = self._free_gb(os.path.dirname(path))
        need = 2 * 2 * f.ntraces * f.ns / 1e9
        form.addRow(QtWidgets.QLabel(f"Import writes two fast-access copies (~{need:.1f} GB) next to the project."
                                     + (f"  Free space: {free:.0f} GB." if free else "")))
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        del f

    @staticmethod
    def _free_gb(folder):
        try:
            import shutil
            return shutil.disk_usage(folder).free / 1e9
        except Exception:
            return None


class RGTDialog(QtWidgets.QDialog):
    """Settings for the relative geologic time (RGT) model."""

    def __init__(self, parent=None, is3d=True, n_hz=0, n_flt=0):
        super().__init__(parent)
        self.setWindowTitle("Compute RGT (relative geologic time)")
        form = QtWidgets.QFormLayout(self)
        self.node_tr = QtWidgets.QSpinBox()
        self.node_tr.setRange(1, 64)
        self.node_tr.setValue(4 if is3d else 8)
        self.node_tr.setSuffix(" traces")
        self.node_ms = QtWidgets.QDoubleSpinBox()
        self.node_ms.setRange(2, 100)
        self.node_ms.setValue(8)
        self.node_ms.setSuffix(" ms")
        self.win = QtWidgets.QDoubleSpinBox()
        self.win.setRange(4, 200)
        self.win.setValue(24)
        self.win.setPrefix("+/- ")
        self.win.setSuffix(" ms")
        self.corr = QtWidgets.QDoubleSpinBox()
        self.corr.setRange(0.0, 0.99)
        self.corr.setSingleStep(0.05)
        self.corr.setValue(0.5)
        self.flinks = QtWidgets.QComboBox()
        self.flinks.addItems(["correlate", "none"])
        self.flinks.setToolTip("correlate: tie the blocks across faults by waveform correlation away from the fault;"
                               " none: only horizons tie blocks across faults")
        self.throw = QtWidgets.QDoubleSpinBox()
        self.throw.setRange(0, 1000)
        self.throw.setValue(80)
        self.throw.setSuffix(" ms")
        self.use_hz = QtWidgets.QCheckBox(f"Use visible horizons as constraints ({n_hz})")
        self.use_hz.setChecked(n_hz > 0)
        self.use_flt = QtWidgets.QCheckBox(f"Use visible faults ({n_flt})")
        self.use_flt.setChecked(n_flt > 0)
        form.addRow("Node spacing", self.node_tr)
        form.addRow("Node interval", self.node_ms)
        form.addRow("Correlation window", self.win)
        form.addRow("Min correlation", self.corr)
        form.addRow("Links across faults", self.flinks)
        form.addRow("Max fault throw", self.throw)
        form.addRow(self.use_hz)
        form.addRow(self.use_flt)
        form.addRow(QtWidgets.QLabel("3D: the dip field is computed first if missing. Results: display box 'RGT'."
                                     if is3d else "All 2D lines are solved together and tied at intersections."))
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def params(self):
        from ..rgt.solve import RGTParams
        return RGTParams(node_traces=int(self.node_tr.value()) or 4, node_ms=float(self.node_ms.value()) or 8.0,
                         window_ms=float(self.win.value()) or 24.0, min_corr=float(self.corr.value()),
                         fault_links=self.flinks.currentText() or "correlate",
                         max_throw_ms=float(self.throw.value()) or 80.0)


def job_cancelled(state):
    """True when a job ended because the user pressed Cancel."""
    return bool(state.get("cancel")) and "cancelled" in (state.get("error") or "")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, path=None):
        super().__init__()
        self.project = SeisProject()
        self.section = SectionPanel()
        self.setCentralWidget(self.section)
        self.basemap = Basemap()
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.horizon_dock = HorizonDock(self.section)
        self.fault_dock = FaultDock(self.section)
        for title, w, area in (("Data", self.tree, QtCore.Qt.LeftDockWidgetArea),
                               ("Basemap", self.basemap, QtCore.Qt.LeftDockWidgetArea),
                               ("Horizons", self.horizon_dock, QtCore.Qt.RightDockWidgetArea),
                               ("Faults", self.fault_dock, QtCore.Qt.RightDockWidgetArea)):
            d = QtWidgets.QDockWidget(title, self)
            d.setObjectName(title)
            d.setWidget(w)
            self.addDockWidget(area, d)
            setattr(self, f"dock_{title.lower()}", d)
        from .layers_dock import LayersPanel
        self.layers_panel = LayersPanel(self.basemap)
        self.layers_panel.analytics_hook = lambda t: self.analytics().add_table(t)  # map hub cross-check (21e)
        self.dock_layers = QtWidgets.QDockWidget("Map layers", self)
        self.dock_layers.setObjectName("Map layers")
        self.dock_layers.setWidget(self.layers_panel)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock_layers)
        self.tabifyDockWidget(self.dock_data, self.dock_layers)
        self.dock_data.raise_()
        self.resizeDocks([self.dock_data, self.dock_basemap], [200, 500], QtCore.Qt.Vertical)
        self.resizeDocks([self.dock_basemap], [520], QtCore.Qt.Horizontal)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.b_cancel = QtWidgets.QPushButton("Cancel")
        self.b_cancel.setToolTip("Stop the running job (import, RGT, attributes...) at the next block")
        self.b_cancel.hide()
        self.b_cancel.clicked.connect(lambda *_: self.cancel_job())
        self.statusBar().addPermanentWidget(self.b_cancel)
        self._job = None
        self._velocity_dialog = None
        self._prognosis_dialog = None
        self._basin_dialog = None
        # dynamic depth conversion (View > Auto-update depth model): file watcher + 1 s debounce + 2 s poll
        self.ddc_auto = False
        self._ddc_debounce = QtCore.QTimer(self)
        self._ddc_debounce.setSingleShot(True)
        self._ddc_debounce.setInterval(1000)
        self._ddc_debounce.timeout.connect(lambda *_: self.ddc_tick())
        self._ddc_poll = QtCore.QTimer(self)
        self._ddc_poll.setInterval(2000)
        self._ddc_poll.timeout.connect(lambda *_: self.ddc_tick())
        self._ddc_watch = QtCore.QFileSystemWatcher(self)
        self._ddc_watch.fileChanged.connect(lambda *_: self._ddc_debounce.start())
        self._ddc_watched = []

        self.section.cursorInfo.connect(lambda s: self.statusBar().showMessage(s))
        self.section.positionChanged.connect(lambda: self.basemap.show_current(self.section.current_trace_line()))
        self.section.positionChanged.connect(lambda: self.send_current_section(self.section.current_trace_line()))
        self.basemap.openLine.connect(lambda n: self.section.show("2D", n))
        self.basemap.openInline.connect(lambda v, il, xl: self.section.show("3D", v, "Inline", il))
        self.basemap.wellClicked.connect(lambda n: self.well_clicked(n))
        self.basemap.wellsSelected.connect(lambda names: self.qgis.send({"type": "select_wells", "wells": names})
                                           if self.qgis is not None else None)
        self.basemap.mapClicked.connect(lambda x, y, t: self.layers_panel.identify(x, y, t) if self.layers_panel.stack else None)
        self._petro_strip = None
        self._live_map = None
        # live-linked cursor (PLAN 23c): section <-> basemap, dashboard, petro strip
        from ..linking import CursorBus
        self.cursor_bus = CursorBus(min_interval=0.03)
        self.cursor_bus.subscribe("map", lambda m: self.basemap.show_cursor(m))
        self.cursor_bus.subscribe("section", lambda m: self.section.show_linked_cursor(m))
        self.section.cursorMoved.connect(lambda m: self.cursor_bus.publish("section", **m))
        self.basemap.cursorMoved.connect(lambda m: self.cursor_bus.publish("map", **m))
        self._hub_mtime = None
        self._hub_poll = QtCore.QTimer(self)  # the Well Correlation Tool's "Show on map hub" (PLAN 21b)
        self._hub_poll.setInterval(3000)
        self._hub_poll.timeout.connect(lambda *_: self.hub_tick())
        self._hub_poll.start()
        # live link with QGIS (PLAN 21h): off until View > Live link with QGIS
        self.qgis = None
        self._qgis_edits = {}
        self._qgis_poll = QtCore.QTimer(self)
        self._qgis_poll.setInterval(200)
        self._qgis_poll.timeout.connect(lambda *_: self.qgis_tick())
        self._arbitrary = []
        # Live Map bus (PLAN 31a): selection / active line / data-changed shared with the other tools
        from .maplink import SeismicMapLink
        self.maplink = SeismicMapLink(self)
        self.mapbus = self.maplink.bus
        self.basemap.wellsSelected.connect(lambda names: self.maplink.wells_selected(names))
        self.section.positionChanged.connect(lambda: self.maplink.position_changed())
        self._bus_poll = QtCore.QTimer(self)
        self._bus_poll.setInterval(300)
        self._bus_poll.timeout.connect(lambda *_: self.bus_tick())
        self._bus_poll.start()
        self.tree.itemDoubleClicked.connect(self._tree_open)
        self.horizon_dock.changed.connect(self._horizons_changed)
        self.fault_dock.changed.connect(self._faults_changed)
        self._menus()
        self.resize(1500, 900)
        if path:
            self.open_project(path)
        else:
            self._refresh()

    # ------------------------------------------------------------ menus
    def _menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&File")
        self._act(m, "&New project...", self.new_project, "Ctrl+N")
        self._act(m, "&Open project...", self.open_dialog, "Ctrl+O")
        self._act(m, "&Save project", self.save, "Ctrl+S")
        m.addSeparator()
        self._act(m, "Import &3D SEG-Y...", self.import_3d_dialog, "Ctrl+3")
        self._act(m, "Add &2D SEG-Y lines...", self.add_2d_dialog, "Ctrl+2")
        self._act(m, "Add 2D &navigation CSV...", self.add_nav_dialog)
        self._act(m, "Link &well project...", self.link_wells_dialog)
        m.addSeparator()
        self._act(m, "E&xit", self.close, "Ctrl+Q")
        m = mb.addMenu("&Interpret")
        self._act(m, "&New horizon...", lambda: self.horizon_dock.new_horizon())
        self._act(m, "&Pick mode on/off", self.horizon_dock.toggle_pick_mode, "P")
        self._act(m, "&Track horizon from seeds", self.horizon_dock.track, "T")
        m.addSeparator()
        self._act(m, "New &fault", lambda: self.fault_dock.new_fault(), "N")
        self._act(m, "&Fault mode on/off", self.fault_dock.toggle_fault_mode, "F")
        self._act(m, "&End fault stick (Enter)", lambda: self.section.end_stick())
        m.addSeparator()
        self._act(m, "Fast &crosslines for current 3D volume (adds a crossline copy)...", self.add_xline_store)
        self._act(m, "Compute &dip field (current 3D volume)...", self.compute_dip)
        self._act(m, "Compute attri&bute volume (current 3D volume)...", self.compute_attribute_volume)
        self._act(m, "Well seeds from tied tops (horizon seeds + RGT control)", lambda *_: self.well_seeds())
        self._act(m, "Paleo-structure animation of this section (GIF)...", lambda *_: self._paleo_dialog())
        self._act(m, "Compute &RGT (current 3D volume)...", self.compute_rgt_3d)
        self._act(m, "Compute RGT (all &2D lines)...", self.compute_rgt_2d)
        self._act(m, "Build horizon &stack (3D)...", self.build_stack_3d)
        self._act(m, "Build horizon stack (2D lines)...", self.build_stack_2d)
        self._act(m, "Convert stack &level to horizon...", self.stack_level_to_horizon)
        self._act(m, "RGT &edit mode on/off (click a level, click where it belongs)", self.toggle_rgt_mode, "E")
        self._act(m, "Use active horizon as RGT constraint", self.rgt_horizon_constraint)
        self._act(m, "Re-link RGT around active fault", self.rgt_relink_fault)
        m.addSeparator()
        self._act(m, "&Well tie (synthetic seismogram)...", self.well_tie)
        self._act(m, "Horizon &maps (structure, isochron, amplitude, depth)...", self.horizon_maps, "Ctrl+M")
        self._act(m, "M&istie analysis (2D-3D, 2D-2D)...", self.mistie_analysis)
        self._act(m, "&Velocity model + depth conversion...", self.velocity_model)
        self._act(m, "Well &prognosis / geosteering...", self.prognosis)
        self._act(m, "&Basin model (burial, thermal, maturity, Wheeler)...", self.basin_model)
        self._act(m, "Seismic &inversion (LFM, post-stack, pre-stack)...", self.inversion)
        self._act(m, "&Analytics (attribute table, models, SHAP)...", self.analytics)
        self._act(m, "&Dashboard (wells, laterals coloured by attribute)...", self.dashboard)
        m.addSeparator()
        self._act(m, "&Undo (horizon or fault edit)", self.horizon_dock.undo, "Ctrl+Z")
        m = mb.addMenu("&View")
        self._act(m, "Live &Map (styled wells, filters, linked table + crossplot)...", lambda *_: self.live_map(),
                  "Ctrl+Shift+M")
        self._act(m, "Next line / slice (PgDown)", lambda: self.section.step(1))
        self._act(m, "Previous line / slice (PgUp)", lambda: self.section.step(-1))
        m.addSeparator()
        m.addAction(self.dock_data.toggleViewAction())
        m.addAction(self.dock_basemap.toggleViewAction())
        m.addAction(self.dock_layers.toggleViewAction())
        m.addAction(self.dock_horizons.toggleViewAction())
        m.addAction(self.dock_faults.toggleViewAction())
        m.addSeparator()
        self.act_link = self._act(m, "&Linked cursor (section / map / dashboard / well strip)",
                                  lambda *_: self.set_linked_cursor(self.act_link.isChecked()))
        self.act_link.setCheckable(True)
        self.act_link.setChecked(True)
        self._act(m, "Co-&blend attributes on the section (RGB / CMY / HSV)...", lambda *_: self.coblend_dialog())
        self._act(m, "Attribute &overlay on the seismic...", lambda *_: self.overlay_dialog())
        self.act_bump = self._act(m, "Bump-map the section / slice", lambda *_: self.section.set_blend(
            {"bump": True} if self.act_bump.isChecked() else None))
        self.act_bump.setCheckable(True)
        self._act(m, "Plain section (no blend / overlay / bump)", lambda *_: self.section.set_blend(None))
        self._act(m, "Attribute parameter &scan (movie)...", lambda *_: self.scan_dialog())
        m.addSeparator()
        self._act(m, "Save window la&yout...", lambda *_: self._save_layout_dialog())
        self._act(m, "Restore window layout...", lambda *_: self._load_layout_dialog())
        m.addSeparator()
        self.act_qgis = self._act(m, "Live link with &QGIS (localhost)", lambda *_: self.toggle_qgis())
        self.act_qgis.setCheckable(True)
        self._act(m, "Send map to QGIS (shared GeoPackage)", lambda *_: self.send_map_to_qgis())
        self.act_ddc = self._act(m, "&Auto-update depth model (dynamic depth conversion)", self.toggle_ddc)
        self.act_ddc.setCheckable(True)
        m = mb.addMenu("&Help")
        self._act(m, "About", lambda: QtWidgets.QMessageBox.information(
            self, "Seismic Interpretation Tool",
            f"Seismic Interpretation Tool {__version__}\nGeo Tool Kit\n\n"
            "PgUp/PgDown step through lines (Shift = 10x). Mouse wheel zooms, right-drag stretches axes, "
            "right-click > View all resets.\n\nHorizons: P = pick mode (click = seed, drag = manual line, "
            "Shift = erase), T = track from seeds, Ctrl+Z = undo.\nFaults: F = fault mode (click points, Enter ends "
            "the stick, Esc cancels), N = new fault, Shift+click deletes a stick."))

    # ------------------------------------------------------------ dynamic depth conversion
    def toggle_ddc(self):
        self.set_ddc(not self.ddc_auto)

    def set_ddc(self, on):
        """Auto-update the depth model when well tops (linked .wcproj.json), horizons or what-if wells change."""
        self.ddc_auto = bool(on)
        self.act_ddc.setChecked(self.ddc_auto)
        self._ddc_watch_paths()
        if self.ddc_auto:
            self._ddc_poll.start()
            if self.project.display.get("velocity_model"):
                self.statusBar().showMessage("Auto-update depth model on - watching the well tops and horizons", 6000)
            else:
                self.statusBar().showMessage("Auto-update depth model on - define a model in Interpret > Velocity "
                                             "model (Use for depth display)", 8000)
        else:
            self._ddc_poll.stop()
            self.statusBar().showMessage("Auto-update depth model off", 4000)

    def _ddc_watch_paths(self):
        if self._ddc_watched:
            self._ddc_watch.removePaths(self._ddc_watched)
            self._ddc_watched = []
        p = self.project.well_project
        if self.ddc_auto and p and os.path.exists(p):
            self._ddc_watch.addPath(p)  # editors that replace the file drop the watch: re-added after every tick
            self._ddc_watched = [p]

    def ddc_tick(self):
        """One dynamic-update check (debounced file change, horizon edit or poll). Returns the report or None."""
        if not self.ddc_auto:
            return None
        try:
            rep = self.section.ddc_refresh()
        except Exception as e:
            self.statusBar().showMessage(f"Depth model update failed: {e}", 10000)
            return None
        self._ddc_watch_paths()
        if rep:
            self.statusBar().showMessage(rep["message"], 15000)
            try:
                self.maplink.depth_model_changed(rep)
            except Exception:
                pass
            dlg = self._velocity_dialog
            if dlg is not None and dlg.ddm is not None:
                if dlg.ddm is self.section.depth_model():
                    dlg.model_refreshed(rep)
                elif dlg.ddm.refresh():
                    dlg.model_refreshed(dlg.ddm.last)
            if self._prognosis_dialog is not None:
                self._prognosis_dialog.update(rep)
            if self._basin_dialog is not None:
                try:
                    if self._basin_dialog.depth_changed():
                        self.statusBar().showMessage(rep["message"] + " - basin map re-run", 15000)
                except Exception as e:
                    self.statusBar().showMessage(f"Basin map update failed: {e}", 10000)
        return rep

    def _act(self, menu, text, fn, key=None):
        a = QtGui.QAction(text, self)
        if key:
            a.setShortcut(QtGui.QKeySequence(key))
        a.triggered.connect(lambda *_, fn=fn: fn())  # never pass 'checked' into optional parameters
        menu.addAction(a)
        return a

    # ------------------------------------------------------------ project
    def _refresh(self):
        p = self.project
        self._ddc_watch_paths()
        self.setWindowTitle(f"Seismic Interpretation Tool - {p.name}" + (f"  [{p.path}]" if p.path else ""))
        self.tree.clear()
        r3 = QtWidgets.QTreeWidgetItem(self.tree, [f"3D volumes ({len(p.volumes)})"])
        for v in p.volumes:
            it = QtWidgets.QTreeWidgetItem(r3, [v["name"]])
            it.setData(0, QtCore.Qt.UserRole, ("3D", v["name"]))
        r2 = QtWidgets.QTreeWidgetItem(self.tree, [f"2D lines ({len(p.lines)})"])
        for l in p.lines:
            it = QtWidgets.QTreeWidgetItem(r2, [l["name"]])
            it.setData(0, QtCore.Qt.UserRole, ("2D", l["name"]))
        rw = QtWidgets.QTreeWidgetItem(self.tree, [f"Wells ({len(p.wells())})"])
        for w in p.wells():
            QtWidgets.QTreeWidgetItem(rw, [w[0]])
        self.tree.expandAll()
        from ..maplayers import LayerStack
        self.basemap.stack = LayerStack.load(p)  # map hub layers (PLAN 21a), saved in display["map_layers"]
        self._hub_mtime = None
        self.layers_panel.set_stack(self.basemap.stack)
        self.basemap.set_project(p)
        self.section.set_project(p)
        self.horizon_dock.set_project(p)
        self.fault_dock.set_project(p)
        self._horizons_changed()
        self._faults_changed()
        try:
            self.maplink.attach(p)
        except Exception as e:
            self.statusBar().showMessage(f"Map bus not connected: {e}", 8000)

    def live_map(self):
        """Open (or raise) the Live Map window on the shared map bus (PLAN 31)."""
        from .live_map import LiveMapWindow
        lm = self._live_map
        if lm is None or lm.project is not self.project:
            lm = LiveMapWindow(self.project, self.mapbus, self,
                               open_section=lambda kind, name, mode=None, number=None: self.section.show(kind, name,
                                                                                                         mode, number),
                               open_arbitrary=lambda name, xs, ys: self.open_arbitrary(name, xs, ys),
                               depth_model_fn=lambda: self.section.depth_model())
            self._live_map = lm
        lm.show()
        lm.raise_()
        return lm

    def bus_tick(self):
        """Deliver map bus events from the other tools (polled every 300 ms)."""
        try:
            return self.maplink.poll()
        except Exception as e:
            self.statusBar().showMessage(f"Map bus: {e}", 8000)
            return []

    # ---------------------------------------------------------------- viewer quick wins (PLAN 23)
    def set_linked_cursor(self, on):
        self.cursor_bus.enabled = bool(on)
        if not on:
            self.cursor_bus.clear()
        return self.cursor_bus.enabled

    def coblend_dialog(self):
        from .. import attributes as A
        names = list(A.ATTRIBUTES)
        picked = []
        for k, col in enumerate(("red / cyan / hue", "green / magenta / saturation", "blue / yellow / value")):
            opts = (["(none)"] if k == 2 else []) + names
            default = {0: "Spectral 10 Hz", 1: "Spectral 20 Hz", 2: "Spectral 40 Hz"}[k]
            t, ok = QtWidgets.QInputDialog.getItem(self, "Co-blend", f"Channel {k + 1} ({col})", opts,
                                                   opts.index(default), False)
            if not ok:
                return None
            if t != "(none)":
                picked.append(t)
        mode, ok = QtWidgets.QInputDialog.getItem(self, "Co-blend", "Mode", ["rgb", "cmy", "hsv"], 0, False)
        if not ok:
            return None
        spec = {"attrs": picked, "mode": mode, "clip": (2.0, 98.0), "gamma": 1.0}
        self.section.set_blend(spec)
        return spec

    def overlay_dialog(self):
        from .. import attributes as A
        t, ok = QtWidgets.QInputDialog.getItem(self, "Overlay", "Attribute over the seismic", list(A.ATTRIBUTES), 0, False)
        if not ok:
            return None
        op, ok = QtWidgets.QInputDialog.getDouble(self, "Overlay", "Opacity (0-1)", 0.5, 0.05, 1.0, 2)
        if not ok:
            return None
        spec = {"overlay": t, "opacity": op}
        self.section.set_blend(spec)
        return spec

    def scan_dialog(self):
        from .scan_dialog import ScanDialog
        dlg = ScanDialog(self.section, self)
        dlg.show()
        self._scan_dialog = dlg
        return dlg

    def layout_state(self):
        """Current window layout: docks shown, open tool windows, section state (+ Qt state when available)."""
        import base64
        st = {"docks": {k: bool(getattr(self, f"dock_{k}").isVisible()) for k in
                        ("data", "basemap", "layers", "horizons", "faults")},
              "windows": [n for n, w in (("petro_strip", self._petro_strip), ("dashboard", getattr(self, "_dashboard", None)))
                          if w is not None and w.isVisible()],
              "section": {"source": list(self.section.source) if self.section.source else None, "mode": self.section.mode,
                          "number": self.section.number, "attr": self.section.attr, "blend": self.section.blend}}
        try:
            st["qt"] = base64.b64encode(bytes(self.saveState())).decode("ascii")
        except Exception:
            pass
        return st

    def apply_layout(self, st):
        import base64
        if not st:
            return False
        if st.get("qt"):
            try:
                self.restoreState(QtCore.QByteArray(base64.b64decode(st["qt"])))
            except Exception:
                pass
        for k, vis in (st.get("docks") or {}).items():
            d = getattr(self, f"dock_{k}", None)
            if d is not None:
                d.setVisible(bool(vis))
        sec = st.get("section") or {}
        if sec.get("source") and self.project is not None:
            try:
                self.section.show(sec["source"][0], sec["source"][1], sec.get("mode"), sec.get("number"))
                if sec.get("attr"):
                    self.section.set_attr(sec["attr"])
                self.section.set_blend(sec.get("blend"))
            except Exception as e:
                self.statusBar().showMessage(f"layout: section not restored ({e})")
        if "dashboard" in (st.get("windows") or []) and getattr(self.project, "well_project", None):
            self.dashboard()
        return True

    def save_layout(self, name):
        from ..linking import save_layout
        return save_layout(self.project, name, self.layout_state())

    def restore_layout(self, name):
        from ..linking import load_layout, TEMPLATES
        return self.apply_layout(load_layout(self.project, name) or TEMPLATES.get(name))

    def _save_layout_dialog(self):
        n, ok = QtWidgets.QInputDialog.getText(self, "Save layout", "Layout name")
        if ok and n.strip():
            self.save_layout(n.strip())
            self.statusBar().showMessage(f"layout '{n.strip()}' saved in the project")

    def _load_layout_dialog(self):
        from ..linking import layout_names, TEMPLATES
        names = layout_names(self.project) + [t for t in TEMPLATES if t not in layout_names(self.project)]
        n, ok = QtWidgets.QInputDialog.getItem(self, "Restore layout", "Layout", names, 0, False)
        if ok and n:
            self.restore_layout(n)

    # ---------------------------------------------------------------- map hub (PLAN 21b)
    def well_clicked(self, name):
        """A well clicked on the map: show its petro strip when petrophysics results exist (or the strip is open)."""
        from .. import maphub as MH
        has = any(r.get("Well") == name for r in MH._read_csv(os.path.join(MH.petro_folder(self.project) or "",
                                                                             "summary.csv")))
        if not has and self._petro_strip is None:
            return None
        from .petro_strip import PetroStripDialog
        if self._petro_strip is None:
            self._petro_strip = PetroStripDialog(self.project, self)
            self.cursor_bus.subscribe("petro_strip", lambda m: self._petro_strip.show_cursor(m))
        self._petro_strip.project = self.project
        d = self._petro_strip.show_well(name)
        self._petro_strip.show()
        self._petro_strip.raise_()
        return d

    def hub_tick(self, force=False):
        """Apply a new hub_request.json from the Well Correlation Tool (polled every 3 s)."""
        from .. import maphub as MH
        if not self.project or self.basemap.stack is None:
            return None
        p = os.path.join(MH.petro_folder(self.project) or "", "hub_request.json")
        if not os.path.exists(p):
            return None
        mt = os.path.getmtime(p)
        if not force and mt == self._hub_mtime:
            return None
        first = self._hub_mtime is None and not force
        self._hub_mtime = mt
        if first:  # an old request from before this session: remember it, do not re-apply
            return None
        req = MH.read_hub_request(self.project)
        lid = MH.apply_hub_request(self.basemap.stack, req)
        self.basemap.draw_layers()
        self.layers_panel.refresh()
        self.basemap.set_selected_wells(req.get("wells") or [])
        self.statusBar().showMessage(f"Map hub: {req.get('property')} {req.get('zone') or '(all zones)'} for "
                                     f"{len(req.get('wells') or [])} wells from the Well Correlation Tool", 10000)
        return lid

    # ---------------------------------------------------------------- QGIS link (PLAN 21h)
    def toggle_qgis(self, on=None, port=None):
        from .. import qgislink as QL
        on = (self.qgis is None) if on is None else on
        if on and self.qgis is None:
            ep = QL.LinkEndpoint("server", port or QL.DEFAULT_PORT)
            if not ep.start():
                self.statusBar().showMessage(f"QGIS link could not start: {ep.error}", 10000)
                return False
            self.qgis = ep
            self._qgis_poll.start()
            self.statusBar().showMessage(f"QGIS link listening on 127.0.0.1:{ep.port} - connect from the GeoHarmonic Link "
                                         "plugin in QGIS", 10000)
        elif not on and self.qgis is not None:
            self.qgis.stop()
            self.qgis = None
            self._qgis_poll.stop()
        return self.qgis is not None

    def send_map_to_qgis(self):
        from .. import qgislink as QL
        if not self.project or not self.project.path or self.basemap.stack is None:
            return None
        out = QL.write_shared(self.project, self.basemap.stack)
        if self.qgis is not None:
            self.qgis.send({"type": "refresh", "gpkg": out["gpkg"]})
        self.statusBar().showMessage(f"Shared GeoPackage written: {out['gpkg']} ({len(out['tables'])} layers, "
                                     f"{len(out['rasters'])} grids)", 10000)
        return out

    def qgis_tick(self):
        """Handle queued QGIS messages; send the current section line; pick up edits made in the shared GeoPackage."""
        from .. import qgislink as QL
        done = []
        if self.qgis is not None:
            for msg in self.qgis.poll():
                done.append(QL.handle(msg, _LinkApp(self)))
            if done:
                self.statusBar().showMessage(done[-1], 5000)
        if self.project and self.project.path and self.basemap.stack is not None:
            first = not self._qgis_edits
            changed, self._qgis_edits = QL.read_user_edits(self.project, self._qgis_edits)
            if changed and not first:  # the first read only records the state of the tables
                r = QL.apply_user_edits(self.basemap.stack, changed)
                self.basemap.draw_layers()
                self.layers_panel.refresh()
                for name, xs, ys in r["section_lines"][-1:]:
                    self.open_arbitrary(name, xs, ys)
        return done

    def send_current_section(self, xy):
        if self.qgis is not None and xy is not None:
            x, y = np.asarray(xy[0], float), np.asarray(xy[1], float)
            k = max(1, len(x) // 50)
            self.qgis.send({"type": "current_section", "xy": np.column_stack([x[::k], y[::k]]).tolist()})

    def open_arbitrary(self, name, xs, ys):
        from .arbitrary_dialog import ArbitraryDialog
        if not self.project.volumes:
            self.statusBar().showMessage("Arbitrary line: the project has no 3D volume", 8000)
            return None
        dlg = ArbitraryDialog(self.project, xs, ys, name, parent=self)
        try:
            dlg.compute()
            dlg.show_section()
        except Exception as e:
            self.statusBar().showMessage(f"Arbitrary line failed: {e}", 8000)
            return None
        dlg.show()
        self._arbitrary.append(dlg)
        return dlg

    def _faults_changed(self):
        self.section.draw_faults()
        self.basemap.draw_faults()
        if getattr(self, "maplink", None) is not None:
            self.maplink.faults_changed()

    def _horizons_changed(self):
        if self.ddc_auto:
            self._ddc_debounce.start()
        self.section.draw_horizons()
        self.basemap.show_horizon(self.horizon_dock.active() if self.horizon_dock.map_chk.isChecked() else None)
        self.basemap.show_current(self.section.current_trace_line())
        if getattr(self, "maplink", None) is not None:
            self.maplink.horizons_changed()

    def _tree_open(self, item, _col):
        d = item.data(0, QtCore.Qt.UserRole)
        if d:
            self.section.show(*d)

    def _need_path(self):
        if self.project.path:
            return True
        QtWidgets.QMessageBox.information(self, "Save first", "Save the project first - imported data is stored "
                                                               "in a folder next to the project file.")
        return self.save_as()

    def job_running(self):
        """True (with a message) while an import / computation runs - it writes into the current project."""
        if getattr(self, "_job", None):
            QtWidgets.QMessageBox.information(self, "Job running", "Wait for the running job to finish (or Cancel it) "
                                                                   "before opening another project.")
            return True
        return False

    def new_project(self):
        if self.job_running():
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "New seismic project", os.path.expanduser("~/Documents"),
                                                        f"Seismic project (*{EXT})")
        if not path:
            return
        self.project = SeisProject()
        self.project.path = path if path.endswith(EXT) else path + EXT
        self.project.name = os.path.basename(self.project.path)[:-len(EXT)]
        self.project.save()
        self._refresh()

    def open_dialog(self):
        if self.job_running():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open seismic project", os.path.expanduser("~/Documents"),
                                                        f"Seismic project (*{EXT})")
        if path:
            self.open_project(path)

    def open_project(self, path):
        try:
            self.project = SeisProject.open(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open failed", str(e))
            return
        self._refresh()

    def save(self):
        if not self.project.path:
            return self.save_as()
        self.project.save()
        self.statusBar().showMessage(f"Saved {self.project.path}", 4000)
        return True

    def save_as(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save seismic project", os.path.expanduser("~/Documents"),
                                                        f"Seismic project (*{EXT})")
        if not path:
            return False
        self.project.save(path)
        if self.project.name == "Untitled":
            self.project.name = os.path.basename(self.project.path)[:-len(EXT)]
            self.project.save()
        self._refresh()
        return True

    # ------------------------------------------------------------ data loading
    def add_2d_dialog(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add 2D SEG-Y lines", self.project.base(),
                                                          "SEG-Y (*.sgy *.segy *.SGY *.SEGY);;All files (*)")
        if not paths:
            return
        names = self.project.add_lines(paths)
        bad = []
        for n in names:
            try:
                self.project.line(n)
            except Exception as e:
                bad.append(f"{n}: {e}")
        if bad:
            QtWidgets.QMessageBox.warning(self, "Some lines could not be read", "\n".join(bad))
        no_xy = [n for n in names if n not in [b.split(":")[0] for b in bad] and not self.project.line(n).has_coords]
        if no_xy:
            self.statusBar().showMessage(f"{len(no_xy)} line(s) have no coordinates in the headers - add a "
                                         "navigation CSV (line,cdp,x,y) to place them on the basemap.", 10000)
        if self.project.path:
            self.project.save()
        self._refresh()
        if names:
            self.section.show("2D", names[0])

    def add_nav_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "2D navigation CSV (line,cdp,x,y)", self.project.base(),
                                                        "CSV (*.csv *.txt);;All files (*)")
        if path:
            self.project.add_nav(path)
            if self.project.path:
                self.project.save()
            self._refresh()

    def link_wells_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Well Correlation project", self.project.base(),
                                                        "Well project (*.wcproj.json)")
        if path:
            self.project.well_project = path
            if self.project.path:
                self.project.save()
            self._refresh()

    def import_3d_dialog(self):
        if self._job:
            QtWidgets.QMessageBox.information(self, "Busy", "An import is already running.")
            return
        if not self._need_path():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "3D SEG-Y", self.project.base(),
                                                        "SEG-Y (*.sgy *.segy *.SGY *.SEGY);;All files (*)")
        if not path:
            return
        try:
            dlg = Import3DDialog(path, self)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Not a readable SEG-Y", str(e))
            return
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        name = dlg.name.text().strip() or "Volume"
        store = os.path.join(self.project.base(), os.path.basename(self.project.path)[:-len(EXT)] + "_seismic",
                             "".join(c if c.isalnum() or c in "-_ " else "_" for c in name))
        from ..volume import unused_store
        store = unused_store(store)  # never write over a store that may be memory-mapped (re-import)
        proj = self.project  # the job belongs to this project even if another one is opened meanwhile
        args = dict(il_byte=dlg.il.value(), xl_byte=dlg.xl.value(), x_byte=dlg.xb.value(), y_byte=dlg.yb.value())
        state = {"frac": 0.0, "text": "Starting...", "done": False, "error": None, "cancel": False}

        def work():
            try:
                import_3d(path, store, progress=lambda f, t: state.update(frac=f, text=t),
                          cancel=lambda: state["cancel"], **args)
            except Exception:
                state["error"] = traceback.format_exc()
            state["done"] = True

        th = threading.Thread(target=work, daemon=True)
        timer = QtCore.QTimer(self)
        self._job = (th, timer, state)
        self.progress.setRange(0, 1000)
        self.progress.show()
        self.b_cancel.show()

        def poll():
            self.progress.setValue(int(state["frac"] * 1000))
            self.statusBar().showMessage(f"Importing {name}: {state['text']}")
            if not state["done"]:
                return
            timer.stop()
            self.progress.hide()
            self.b_cancel.hide()
            self._job = None
            if state["error"] and job_cancelled(state):
                self.statusBar().showMessage("Job cancelled", 8000)
                return
            if state["error"]:
                QtWidgets.QMessageBox.critical(self, "Import failed", state["error"][-2000:])
                return
            proj.volumes = [v for v in proj.volumes if v["name"] != name]
            proj._vol_cache.pop(name, None)
            proj.volumes.append({"name": name, "store": store, "segy": path, **args})
            proj.save()
            if proj is not self.project:
                self.statusBar().showMessage(f"Imported {name} into {proj.path}", 8000)
                return
            self._refresh()
            self.section.show("3D", name, "Inline")
            self.statusBar().showMessage(f"Imported {name}", 8000)

        timer.timeout.connect(poll)
        timer.start(250)
        th.start()

    def cancel_job(self):
        """Ask the running job to stop (it checks between blocks and cleans up its partial output)."""
        if self._job:
            self._job[2]["cancel"] = True
            self.statusBar().showMessage("Cancelling...")
            return True
        return False

    def run_job(self, label, fn, done=None):
        """Run fn(progress, cancel) in a thread with the status-bar progress bar; done() on success."""
        if self._job:
            QtWidgets.QMessageBox.information(self, "Busy", "Another job is running.")
            return False
        state = {"frac": 0.0, "text": "Starting...", "done": False, "error": None, "cancel": False}

        def work():
            try:
                fn(lambda f, t: state.update(frac=f, text=t), lambda: state["cancel"])
            except Exception:
                state["error"] = traceback.format_exc()
            state["done"] = True

        th = threading.Thread(target=work, daemon=True)
        timer = QtCore.QTimer(self)
        self._job = (th, timer, state)
        self.progress.setRange(0, 1000)
        self.progress.show()
        self.b_cancel.show()

        def poll():
            self.progress.setValue(int(state["frac"] * 1000))
            self.statusBar().showMessage(f"{label}: {state['text']}")
            if not state["done"]:
                return
            timer.stop()
            self.progress.hide()
            self.b_cancel.hide()
            self._job = None
            if state["error"] and job_cancelled(state):
                self.statusBar().showMessage("Job cancelled", 8000)
                return
            if state["error"]:
                QtWidgets.QMessageBox.critical(self, f"{label} failed", state["error"][-2000:])
                return
            if done:
                done()
            self.statusBar().showMessage(f"{label}: finished", 8000)

        timer.timeout.connect(poll)
        timer.start(250)
        th.start()
        return True

    def add_xline_store(self):
        """Older stores read crosslines from the inline-major copy (slow on big surveys); add the crossline copy."""
        src = self.section.source
        if not src or src[0] != "3D":
            QtWidgets.QMessageBox.information(self, "Fast crosslines", "Show a 3D volume first.")
            return False
        vol = self.project.volume(src[1])
        if vol.xldata is not None:
            QtWidgets.QMessageBox.information(self, "Fast crosslines", f"{src[1]} already has the crossline copy.")
            return False
        gb = vol.nil * vol.nxl * vol.ns * 2 / 1e9
        if QtWidgets.QMessageBox.question(self, "Fast crosslines", f"Write a crossline copy of {src[1]} "
                                          f"(~{gb:.1f} GB next to the volume store)?") != QtWidgets.QMessageBox.Yes:
            return False
        from ..volume import build_xline_store

        def fn(progress, cancel):
            build_xline_store(vol.store, progress, cancel)

        def done():
            self.project._vol_cache.pop(src[1], None)  # reopen with the new copy
            self.section.redraw()

        return self.run_job(f"Crossline copy {src[1]}", fn, done)

    def compute_dip(self):
        src = self.section.source
        if not src or src[0] != "3D":
            QtWidgets.QMessageBox.information(self, "Dip field", "Show a 3D volume first. (2D lines: choose "
                                              "'Inline dip' or 'Planarity' in the display box - computed on the fly.)")
            return
        from ..rgt.dip import compute_dip_volume
        self.section.reset_rgt_editors()  # editors keep the dip files open (Windows cannot replace them)
        vol = self.project.volume(src[1])
        gb = vol.nil * vol.nxl * vol.ns * 2 * 3 / 1e9
        if QtWidgets.QMessageBox.question(self, "Dip field", f"Compute inline/crossline dip and planarity for "
                                          f"{src[1]}?\nWrites ~{gb:.1f} GB next to the volume store.") \
                != QtWidgets.QMessageBox.Yes:
            return

        def fn(progress, cancel):
            compute_dip_volume(vol, progress=progress, cancel=cancel)

        def done():
            self.section.attr_combo.setCurrentText("Inline dip")
            self.section.set_attr("Inline dip")

        self.run_job(f"Dip field {src[1]}", fn, done)

    def compute_attribute_volume(self, name=None):
        src = self.section.source
        if not src or src[0] != "3D":
            QtWidgets.QMessageBox.information(self, "Attribute volume", "Show a 3D volume first (2D lines: use the "
                                              "display box - attributes are computed on the fly).")
            return
        from ..attributes import ATTRIBUTES, compute_volume
        if name is None:
            name, ok = QtWidgets.QInputDialog.getItem(self, "Attribute volume", "Attribute:", list(ATTRIBUTES), 0, False)
            if not ok:
                return
        result = {}

        def fn(progress, cancel):
            result["name"] = compute_volume(self.project, src[1], str(name), None, 16, progress, cancel)

        def done():
            if self.project.path:
                self.project.save()
            self._refresh()
            self.section.show("3D", result["name"], self.section.mode, self.section.number)

        self.run_job(f"Attribute {name}", fn, done)

    def _rgt_inputs(self, dlg):
        hz = [h for h in self.project.horizons.items if h.visible] if dlg.use_hz.isChecked() else []
        fl = self.project.faults.visible_faults() if dlg.use_flt.isChecked() else []
        return hz, fl

    def well_seeds(self, source="tops", radius=6, wavelet=None, vol_name=None):
        """PLAN 24d: tied-well tops (or synthetic events) -> snapped horizon patches 'seed: <name>' in the project;
        Compute RGT uses visible horizons as control, T grows a seed into a full horizon."""
        from ..wellseeds import well_seeds
        src = ("3D", vol_name) if vol_name else self.section.source
        if not src or src[0] != "3D" or not self.project.well_project:
            self.statusBar().showMessage("Well seeds: show a 3D volume and link a well project with tops and ties")
            return []
        hs = well_seeds(self.project, src[1], source=source, radius=radius, wavelet=wavelet, add_to_project=True)
        self.horizon_dock.refresh(hs[0].name if hs else None)
        self._horizons_changed()
        self.statusBar().showMessage(f"{len(hs)} seed horizon(s) from the wells: " + ", ".join(h.name for h in hs[:6]))
        return hs

    def paleo_animation(self, path, duration_ms=800):
        """PLAN 26b: the current section flattened on each visible horizon, oldest -> youngest, as an animated GIF."""
        from .. import structure as S
        hz_ = self.section.section_horizons()
        if not hz_:
            raise ValueError("show an inline / crossline / 2D line with interpreted horizons")
        frames = S.paleo_frames(np.asarray(self.section._data, float), np.asarray(self.section._yaxis, float), hz_)
        S.export_gif(S.frames_to_rgba(frames), path, duration_ms)
        self.statusBar().showMessage(f"Paleo animation ({len(frames)} frames: {', '.join(n for n, _ in frames)}) -> {path}")
        return [n for n, _ in frames]

    def _paleo_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Paleo-structure animation", "paleo.gif", "GIF (*.gif)")
        if path:
            try:
                self.paleo_animation(path)
            except Exception as e:
                self.statusBar().showMessage(f"Paleo animation: {e}")

    def compute_rgt_3d(self):
        src = self.section.source
        if not src or src[0] != "3D":
            QtWidgets.QMessageBox.information(self, "RGT", "Show a 3D volume first.")
            return
        dlg = RGTDialog(self, True, sum(h.visible for h in self.project.horizons.items),
                        len(self.project.faults.visible_faults()))
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        from ..rgt.solve import compute_rgt_3d
        self.section.reset_rgt_editors()
        vol = self.project.volume(src[1])
        params = dlg.params()
        hz, fl = self._rgt_inputs(dlg)

        def fn(progress, cancel):
            _, sysm = compute_rgt_3d(vol, params, fl, hz, progress, cancel, vol_name=src[1])
            self._rgt_stats = sysm.stats

        def done():
            self.section.reset_rgt_editors()
            self.section.attr_combo.setCurrentText("RGT")
            self.section.set_attr("RGT")
            self.statusBar().showMessage(f"RGT done: {getattr(self, '_rgt_stats', {})}", 15000)

        self.run_job(f"RGT {src[1]}", fn, done)

    def compute_rgt_2d(self):
        names = [l["name"] for l in self.project.lines]
        if not names:
            QtWidgets.QMessageBox.information(self, "RGT", "The project has no 2D lines.")
            return
        if not self.project.path:
            QtWidgets.QMessageBox.information(self, "RGT", "Save the project first (results are stored next to it).")
            return
        dlg = RGTDialog(self, False, sum(h.visible for h in self.project.horizons.items),
                        len(self.project.faults.visible_faults()))
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        from ..rgt.solve import compute_rgt_2d
        params = dlg.params()
        hz, fl = self._rgt_inputs(dlg)

        def fn(progress, cancel):
            _, sysm = compute_rgt_2d(self.project, names, params, fl, hz, progress, cancel)
            self._rgt_stats = sysm.stats

        def done():
            self.section.reset_rgt_editors()
            if self.section.source and self.section.source[0] == "2D":
                self.section.attr_combo.setCurrentText("RGT")
                self.section.set_attr("RGT")
            self.statusBar().showMessage(f"RGT done ({len(names)} lines): {getattr(self, '_rgt_stats', {})}", 15000)

        self.run_job("RGT 2D lines", fn, done)

    def well_tie(self, well=None):
        if not self.project.well_project:
            QtWidgets.QMessageBox.information(self, "Well tie", "Link a well project first (File > Link well project).")
            return None
        if not self.section.source:
            return None
        from .welltie_dialog import WellTieDialog
        try:
            dlg = WellTieDialog(self.project, self.section.source, self, well)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Well tie", str(e))
            return None
        dlg._changed()
        dlg.exec()
        if getattr(dlg, "batch_rows", None):  # "Tie all wells" was run: bulk-shift map on the basemap
            from ..tie import shift_map
            g = shift_map(dlg.batch_rows)
            if g is not None:
                self.basemap.show_grid(g)
        self.section.redraw()
        return dlg

    def analytics(self):
        from .analytics_window import AnalyticsWindow
        dlg = AnalyticsWindow(self.project, self.basemap, self)
        dlg.show()
        self._analytics = dlg
        return dlg

    def dashboard(self):
        from .dashboard_window import DashboardWindow
        win = DashboardWindow(self.project, self.basemap, self)
        win.analytics_window = getattr(self, "_analytics", None)
        if self.project.well_project:
            win.load_project_wells()
        win.show()
        self._dashboard = win
        self.cursor_bus.subscribe("dashboard", lambda m: win.show_cursor(m))
        return win

    def velocity_model(self):
        if not self.project.horizons.items or not self.project.well_project:
            QtWidgets.QMessageBox.information(self, "Velocity model", "Needs horizons and a linked well project "
                                              "(with tops or time-depth).")
            return None
        from .velocity_dialog import VelocityDialog
        def changed():
            self.section.adopt_depth_model(dlg.ddm)
            self.section.redraw()
        dlg = VelocityDialog(self.project, self.basemap, self, changed)
        dlg.show()
        self._velocity_dialog = dlg
        return dlg

    def inversion(self):
        from .inversion_dialog import InversionDialog
        src = self.section.source
        vol = src[1] if src and src[0] == "3D" else None

        def new_volumes(names):
            if self.project.path:
                self.project.save()
            self._refresh()
        dlg = InversionDialog(self.project, self, self.run_job, new_volumes, vol)
        dlg.show()
        self._inversion_dialog = dlg
        return dlg

    def basin_model(self):
        from .basin_dialog import BasinDialog
        dlg = BasinDialog(self.project, self.basemap, self, self.section, self.run_job)
        dlg.on_wheeler = lambda prop: self.maplink.publish_current_section("Basin / Wheeler sections",
                                                                           wheeler=True, prop=prop)
        dlg.show()
        self._basin_dialog = dlg
        return dlg

    def prognosis(self):
        if not self.project.display.get("velocity_model"):
            QtWidgets.QMessageBox.information(self, "Well prognosis", "Define a depth model first: Interpret > Velocity "
                                              "model, Build, 'Use for depth display'.")
            return None
        from .prognosis_dialog import PrognosisDialog

        def changed():
            self.section.redraw()
            dlg = self._velocity_dialog
            if dlg is not None and dlg.ddm is self.section.depth_model():
                dlg.model_refreshed(dlg.ddm.last)
        dlg = PrognosisDialog(self.project, self.section.depth_model, self, changed)
        dlg.show()
        self._prognosis_dialog = dlg
        return dlg

    def mistie_analysis(self):
        if not self.project.lines:
            QtWidgets.QMessageBox.information(self, "Mistie", "The project has no 2D lines.")
            return None
        from .mistie_dialog import MistieDialog

        def refreshed():
            self.section.reset_rgt_editors()
            self.section.redraw()
            self._horizons_changed()
            self.maplink.misties_changed(sorted((self.project.display.get("mistie") or {}).keys()))

        dlg = MistieDialog(self.project, self, refreshed, self.basemap)
        dlg.on_residuals = lambda ties, sol: self.maplink.misties_changed(
            [k for k in sol if not str(k).startswith("3D:")], sol, ties)
        dlg.show()
        self._mistie_dialog = dlg
        return dlg

    def horizon_maps(self):
        if not self.project.horizons.items:
            QtWidgets.QMessageBox.information(self, "Horizon maps", "Pick or create a horizon first.")
            return None
        from .map_dialog import MapDialog
        dlg = MapDialog(self.project, self.basemap, self, self.section.source)
        if self.section.active_horizon:
            dlg.h1.setCurrentText(self.section.active_horizon)
        dlg.show()
        self._map_dialog = dlg
        return dlg

    def toggle_rgt_mode(self):
        on = not self.section.rgt_mode
        self.section.set_mode("rgt" if on else None)
        self.statusBar().showMessage("RGT edit mode: click an iso-RGT level, then click where it belongs (Ctrl+Z "
                                     "undoes)" if on else "RGT edit mode off", 8000)

    def rgt_horizon_constraint(self):
        sp = self.section
        h = self.project.horizons.get(sp.active_horizon) if sp.active_horizon else None
        if h is None or not sp.source or h.get(sp.section_key()) is None:
            QtWidgets.QMessageBox.information(self, "RGT constraint", "Select a horizon with picks on the survey shown.")
            return
        ed = sp.rgt_editor()
        if ed is None:
            return
        fl = self.project.faults.visible_faults() or None
        if sp.source[0] == "3D":
            ed.add_horizon(h.name, h.get(sp.section_key()), fl)
        else:
            g = h.get(sp.section_key())
            j = np.nonzero(np.isfinite(g))[0]
            v = float(np.nanmean([ed.value_at(k, g[k]) for k in j]))
            for k in j[::max(1, len(j) // 50)]:
                ed.add_point(int(k), float(g[k]), v, fl)
        sp.redraw()
        self.statusBar().showMessage(f"RGT re-solved with {h.name} as a constraint", 8000)

    def rgt_relink_fault(self):
        sp = self.section
        f = self.project.faults.get(sp.active_fault) if sp.active_fault else None
        if f is None or not len(f.triangles()) or not sp.source or sp.source[0] != "3D":
            QtWidgets.QMessageBox.information(self, "RGT re-link", "Select a fault with at least two sticks and show "
                                              "the 3D volume.")
            return
        ed = sp.rgt_editor()
        if ed is None:
            return
        ed.relink_fault(f, self.project.faults.visible_faults())
        sp.redraw()
        self.statusBar().showMessage(f"RGT re-solved around {f.name}", 8000)

    def build_stack_3d(self, n_levels=None):
        src = self.section.source
        if not src or src[0] != "3D":
            QtWidgets.QMessageBox.information(self, "Horizon stack", "Show a 3D volume with an RGT model first.")
            return
        from ..rgt.solve import RGTVolume
        from ..rgt.stack import build_stack_3d
        vol = self.project.volume(src[1])
        folder = os.path.join(vol.store, "rgt")
        if not RGTVolume.exists(folder):
            QtWidgets.QMessageBox.information(self, "Horizon stack", "Compute the RGT model first.")
            return
        if n_levels is None:
            n_levels, ok = QtWidgets.QInputDialog.getInt(self, "Horizon stack", "Number of iso-RGT surfaces:", 100, 5, 2000)
            if not ok:
                return

        def fn(progress, cancel):
            build_stack_3d(vol, RGTVolume(folder), int(n_levels), progress=progress, cancel=cancel)

        def done():
            self.section.mode_combo.setCurrentText("Stratal slice")
            self.section._mode_changed("Stratal slice")

        self.run_job(f"Horizon stack {src[1]}", fn, done)

    def build_stack_2d(self, n_levels=None):
        from ..rgt.stack import build_stack_2d
        if n_levels is None:
            n_levels, ok = QtWidgets.QInputDialog.getInt(self, "Horizon stack (2D)", "Number of iso-RGT surfaces:", 100, 5, 2000)
            if not ok:
                return
        try:
            st = build_stack_2d(self.project, [l["name"] for l in self.project.lines], int(n_levels))
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Horizon stack (2D)", str(e))
            return
        self.statusBar().showMessage(f"2D horizon stack: {len(st['levels'])} levels on {len(st['lines'])} lines", 8000)
        self.section.redraw()

    def stack_level_to_horizon(self, level=None, name=None):
        """New horizon from a stack level (3D: the current stratal slice level by default; 2D: all lines)."""
        from ..rgt.stack import HorizonStack3D, load_stack_2d, stack_to_horizon
        src = self.section.source
        st3 = vol_name = None
        if src and src[0] == "3D":
            folder = os.path.join(self.project.volume(src[1]).store, "rgt")
            st3 = HorizonStack3D(folder) if HorizonStack3D.exists(folder) else None
            vol_name = src[1]
        st2 = load_stack_2d(os.path.join(self.project.seismic_dir(), "rgt")) if self.project.path else None
        if st3 is None and st2 is None:
            QtWidgets.QMessageBox.information(self, "Stack level", "Build a horizon stack first.")
            return None
        n = len(st3) if st3 is not None else len(st2["levels"])
        if level is None:
            cur = int(self.section.number or 0) if self.section.mode == "Stratal slice" else 0
            level, ok = QtWidgets.QInputDialog.getInt(self, "Stack level to horizon", f"Level (0-{n - 1}):", cur, 0, n - 1)
            if not ok:
                return None
        h = stack_to_horizon(self.project.horizons, name or f"RGT level {int(level)}", int(level), st3, vol_name, st2)
        self.horizon_dock.refresh(h.name)
        self.section.active_horizon = h.name
        self._horizons_changed()
        self.statusBar().showMessage(f"Horizon {h.name} created from stack level {level} - edit it like any horizon", 8000)
        return h

    def closeEvent(self, ev):
        if self._job:
            if QtWidgets.QMessageBox.question(self, "Import running",
                                              "A 3D import is still running. Quit anyway (the import is lost)?") \
                    != QtWidgets.QMessageBox.Yes:
                ev.ignore()
                return
        if self.project.path:
            try:
                self.project.save()
            except Exception:
                pass
        self.maplink.close()
        ev.accept()


class _LinkApp:
    """What a QGIS message can do in GeoHarmonic (qgislink.handle calls these)."""

    def __init__(self, win):
        self.w = win

    def select_wells(self, names):
        self.w.basemap.set_selected_wells(names, emit=False)

    def zoom(self, bbox):
        if bbox and len(bbox) == 4:
            try:
                self.w.basemap.plot.getPlotItem().getViewBox().setRange(xRange=(bbox[0], bbox[2]), yRange=(bbox[1], bbox[3]))
            except Exception:
                pass

    def open_well(self, name):
        self.w.well_clicked(name)

    def section_line(self, name, xs, ys):
        self.w.open_arbitrary(name, xs, ys)

    def polygon(self, name, xs, ys):
        st = self.w.basemap.stack
        lay = next((l for l in st.layers if l["kind"] == "polygons" and l["name"] == "Prospects (QGIS)"), None)
        if lay is None:
            lid = st.add("polygons", "Prospects (QGIS)", src={"polygons": []})
            lay = st.get(lid)
        polys = [p for p in lay["src"]["polygons"] if p.get("name") != name]
        polys.append({"name": name, "x": [float(v) for v in xs], "y": [float(v) for v in ys], "from": "message"})
        lay["src"]["polygons"] = polys
        st.save()
        self.w.basemap.draw_layers()
        self.w.layers_panel.refresh()

    def refresh(self):
        self.w.send_map_to_qgis()

