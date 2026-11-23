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
        self.resizeDocks([self.dock_data, self.dock_basemap], [200, 500], QtCore.Qt.Vertical)
        self.resizeDocks([self.dock_basemap], [520], QtCore.Qt.Horizontal)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self._job = None

        self.section.cursorInfo.connect(lambda s: self.statusBar().showMessage(s))
        self.section.positionChanged.connect(lambda: self.basemap.show_current(self.section.current_trace_line()))
        self.basemap.openLine.connect(lambda n: self.section.show("2D", n))
        self.basemap.openInline.connect(lambda v, il, xl: self.section.show("3D", v, "Inline", il))
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
        self._act(m, "Compute &dip field (current 3D volume)...", self.compute_dip)
        self._act(m, "Compute attri&bute volume (current 3D volume)...", self.compute_attribute_volume)
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
        m.addSeparator()
        self._act(m, "&Undo (horizon or fault edit)", self.horizon_dock.undo, "Ctrl+Z")
        m = mb.addMenu("&View")
        self._act(m, "Next line / slice (PgDown)", lambda: self.section.step(1))
        self._act(m, "Previous line / slice (PgUp)", lambda: self.section.step(-1))
        m.addSeparator()
        m.addAction(self.dock_data.toggleViewAction())
        m.addAction(self.dock_basemap.toggleViewAction())
        m.addAction(self.dock_horizons.toggleViewAction())
        m.addAction(self.dock_faults.toggleViewAction())
        m = mb.addMenu("&Help")
        self._act(m, "About", lambda: QtWidgets.QMessageBox.information(
            self, "Seismic Interpretation Tool",
            f"Seismic Interpretation Tool {__version__}\nGeo Tool Kit\n\n"
            "PgUp/PgDown step through lines (Shift = 10x). Mouse wheel zooms, right-drag stretches axes, "
            "right-click > View all resets.\n\nHorizons: P = pick mode (click = seed, drag = manual line, "
            "Shift = erase), T = track from seeds, Ctrl+Z = undo.\nFaults: F = fault mode (click points, Enter ends "
            "the stick, Esc cancels), N = new fault, Shift+click deletes a stick."))

    def _act(self, menu, text, fn, key=None):
        a = QtGui.QAction(text, self)
        if key:
            a.setShortcut(QtGui.QKeySequence(key))
        a.triggered.connect(fn)
        menu.addAction(a)
        return a

    # ------------------------------------------------------------ project
    def _refresh(self):
        p = self.project
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
        self.basemap.set_project(p)
        self.section.set_project(p)
        self.horizon_dock.set_project(p)
        self.fault_dock.set_project(p)
        self._horizons_changed()
        self._faults_changed()

    def _faults_changed(self):
        self.section.draw_faults()
        self.basemap.draw_faults()

    def _horizons_changed(self):
        self.section.draw_horizons()
        self.basemap.show_horizon(self.horizon_dock.active() if self.horizon_dock.map_chk.isChecked() else None)
        self.basemap.show_current(self.section.current_trace_line())

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

    def new_project(self):
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
        args = dict(il_byte=dlg.il.value(), xl_byte=dlg.xl.value(), x_byte=dlg.xb.value(), y_byte=dlg.yb.value())
        state = {"frac": 0.0, "text": "Starting...", "done": False, "error": None}

        def work():
            try:
                import_3d(path, store, progress=lambda f, t: state.update(frac=f, text=t), **args)
            except Exception:
                state["error"] = traceback.format_exc()
            state["done"] = True

        th = threading.Thread(target=work, daemon=True)
        timer = QtCore.QTimer(self)
        self._job = (th, timer, state)
        self.progress.setRange(0, 1000)
        self.progress.show()

        def poll():
            self.progress.setValue(int(state["frac"] * 1000))
            self.statusBar().showMessage(f"Importing {name}: {state['text']}")
            if not state["done"]:
                return
            timer.stop()
            self.progress.hide()
            self._job = None
            if state["error"]:
                QtWidgets.QMessageBox.critical(self, "Import failed", state["error"][-2000:])
                return
            self.project.volumes = [v for v in self.project.volumes if v["name"] != name]
            self.project._vol_cache.pop(name, None)
            self.project.volumes.append({"name": name, "store": store, "segy": path, **args})
            self.project.save()
            self._refresh()
            self.section.show("3D", name, "Inline")
            self.statusBar().showMessage(f"Imported {name}", 8000)

        timer.timeout.connect(poll)
        timer.start(250)
        th.start()

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

        def poll():
            self.progress.setValue(int(state["frac"] * 1000))
            self.statusBar().showMessage(f"{label}: {state['text']}")
            if not state["done"]:
                return
            timer.stop()
            self.progress.hide()
            self._job = None
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
        self.section.redraw()
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

        dlg = MistieDialog(self.project, self, refreshed, self.basemap)
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
        ev.accept()
