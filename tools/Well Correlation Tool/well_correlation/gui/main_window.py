"""Main window: base map + cross section + well lists. Step 1 of the Well Correlation Tool."""
import glob
import os
import time
import traceback

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .. import __version__
from .. import tops as T
from ..export import export_section
from ..project import Project, PROJECT_EXT
from ..section import TRACKS
from .map_view import MapView
from .section_view import SectionView

# antialias off: much faster with long log curves; images use row-major arrays
pg.setConfigOptions(antialias=False, background="w", foreground="k", imageAxisOrder="row-major")
ORG, APP = "GeoToolKit", "WellCorrelation"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, path=None):
        super().__init__()
        self.project = None
        self.dirty = False
        self.settings = QtCore.QSettings(ORG, APP)
        self.resize(1500, 900)

        # central: map | section
        self.map = MapView()
        self.section = SectionView()
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.map)
        split.addWidget(self.section)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        split.setSizes([450, 1050])
        self.setCentralWidget(split)

        # docks
        self.wells_list = QtWidgets.QListWidget()
        self.wells_list.setToolTip("Double-click to add a well to the section")
        self.sec_list = QtWidgets.QListWidget()
        self.sec_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.sec_list.setToolTip("Drag to reorder. Delete key removes a well.")
        self.sec_combo = QtWidgets.QComboBox()
        self.sec_combo.setToolTip("Saved sections")
        sec_box = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(sec_box)
        sl.setContentsMargins(2, 2, 2, 2)
        sl.addWidget(self.sec_combo)
        sl.addWidget(self.sec_list)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("Reverse", self.reverse_section), ("Clear", self.clear_section), ("New", self.new_section)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        sl.addLayout(row)
        self.info = QtWidgets.QTextBrowser()
        self._dock("Wells", self.wells_list, QtCore.Qt.LeftDockWidgetArea)
        self._dock("Section wells", sec_box, QtCore.Qt.LeftDockWidgetArea)
        self.info_dock = self._dock("Well info", self.info, QtCore.Qt.LeftDockWidgetArea)
        self.tops_dock = self._dock("Tops", self._build_tops_panel(), QtCore.Qt.RightDockWidgetArea)
        self.undo_stack, self.redo_stack = [], []
        self._petro_dlg = None

        self._build_actions()
        self._build_toolbar()
        self.status = self.statusBar()
        self.cursor_label = QtWidgets.QLabel()
        self.status.addPermanentWidget(self.cursor_label)

        # wiring
        self.map.wellClicked.connect(self.toggle_well)
        self.map.wellSelected.connect(self.select_well)
        self.map.lineSelected.connect(self.set_section_wells)
        self.map.mapMessage.connect(lambda m: self.status.showMessage(m, 8000))
        self.section.wellSelected.connect(self.select_well)
        self.section.cursorInfo.connect(self.cursor_label.setText)
        self.section.topEdited.connect(self.on_top_edited)
        self.section.formationSelected.connect(self.select_formation)
        for key, delta in (("Up", -0.5), ("Down", 0.5), ("Shift+Up", -5.0), ("Shift+Down", 5.0)):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self.section)
            sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda d=delta: self.nudge_top(d))
        sc = QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Delete), self.section)
        sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        sc.activated.connect(self.delete_active_top)
        for key, fn in (("Return", self.accept_proposals), ("Enter", self.accept_proposals),
                        ("Escape", self.clear_proposals), ("G", self.start_ghost),
                        ("[", lambda: self.section.ghost_stretch(1 / 1.05)), ("]", lambda: self.section.ghost_stretch(1.05))):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self.section)
            sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        self.wells_list.itemDoubleClicked.connect(lambda it: self.toggle_well(it.text()))
        self.wells_list.currentTextChanged.connect(lambda t: t and self.select_well(t))
        self.sec_list.model().rowsMoved.connect(self._section_reordered)
        self.sec_list.currentTextChanged.connect(lambda t: t and self.select_well(t))
        self.sec_combo.currentTextChanged.connect(self._switch_section)
        sc = QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Delete), self.sec_list)
        sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        sc.activated.connect(self.remove_selected_from_section)

        self._update_title()
        if not path:
            path = self._default_project()
        if path:
            self.open_project(path)
        else:
            self._show_start_hint()

    # ------------------------------------------------------------------ ui
    def _dock(self, title, widget, area):
        d = QtWidgets.QDockWidget(title, self)
        d.setObjectName(title)
        d.setWidget(widget)
        self.addDockWidget(area, d)
        return d

    def _build_actions(self):
        m = self.menuBar()
        f = m.addMenu("&File")
        self._act(f, "&New project from data folder...", self.new_project, "Ctrl+N")
        self._act(f, "&Open project...", self.open_dialog, "Ctrl+O")
        self.recent_menu = f.addMenu("Open &recent")
        self._fill_recent()
        f.addSeparator()
        self._act(f, "&Save", self.save, "Ctrl+S")
        self._act(f, "Save &as...", self.save_as, "Ctrl+Shift+S")
        f.addSeparator()
        self._act(f, "&Export section image (PNG/PDF)...", self.export_image, "Ctrl+E")
        f.addSeparator()
        self._act(f, "Import tops (CSV / TSV)...", self.import_tops)
        ex = f.addMenu("Export tops")
        self._act(ex, "CSV (all columns)...", lambda: self.export_tops("csv"))
        self._act(ex, "OpendTect well markers (tab: well, marker, MD)...", lambda: self.export_tops("odt"))
        self._act(ex, "Kingdom tops (tab: UWI, well, formation, MD, TVDSS)...", lambda: self.export_tops("kingdom"))
        self._act(f, "Reload logs (rebuild cache)", self.reload_logs)
        f.addSeparator()
        self._act(f, "E&xit", self.close, "Ctrl+Q")
        e = m.addMenu("&Edit")
        self.act_undo = self._act(e, "&Undo", self.undo, "Ctrl+Z")
        self.act_redo = self._act(e, "&Redo", self.redo, "Ctrl+Y")
        e.addSeparator()
        self.act_pick = self._act(e, "&Pick tops", self.toggle_pick, "P")
        self.act_pick.setCheckable(True)
        self._act(e, "Delete active top in selected well", self.delete_active_top)
        c = m.addMenu("&Correlate")
        self._act(c, "Suggest active top in selected well", self.suggest_selected, "S")
        self._act(c, "Suggest active top in all section wells missing it", self.suggest_section, "Shift+S")
        self._act(c, "Suggest all formations in selected well", self.suggest_all_formations, "Alt+S")
        c.addSeparator()
        self._act(c, "Accept suggestions (Enter)", self.accept_proposals)
        self._act(c, "Clear suggestions (Esc)", self.clear_proposals)
        c.addSeparator()
        self._act(c, "Ghost curve at cursor (G)", self.start_ghost)
        c.addSeparator()
        self._act(c, "Zone statistics...", self.show_zone_stats, "Z")
        self._act(c, "Compare tops with reference (e.g. PR3099)...", self.show_reference, "R")
        pm = m.addMenu("&Petrophysics")
        self._act(pm, "Petrophysics window...", self.show_petro, "Ctrl+Shift+P")
        self._act(pm, "Curve QC + tool report (selected well)", lambda *_: self.show_petro().run_qc())
        v = m.addMenu("&View")
        self._act(v, "Fit section", lambda: self.section.redraw(), "F")
        self._act(v, "Fit map", self.map.fit)
        v.addSeparator()
        for d in self.findChildren(QtWidgets.QDockWidget):
            v.addAction(d.toggleViewAction())
        h = m.addMenu("&Help")
        self._act(h, "Data folder layout", self.help_layout)
        self._act(h, "Mouse and keys", self.help_mouse)
        self._act(h, "About", lambda: QtWidgets.QMessageBox.about(
            self, "Well Correlation Tool", f"Well Correlation Tool {__version__}\nGeo Tool Kit\n\nPySide6 + pyqtgraph, runs offline."))

    def _act(self, menu, text, fn, key=None):
        a = QtGui.QAction(text, self)
        if key:
            a.setShortcut(QtGui.QKeySequence(key))
        a.triggered.connect(fn)
        menu.addAction(a)
        return a

    def _build_toolbar(self):
        tb = self.addToolBar("Display")
        tb.setObjectName("Display")
        tb.addAction(self.act_pick)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Hang: "))
        self.hang = QtWidgets.QComboBox()
        self.hang.currentIndexChanged.connect(self._display_changed)
        tb.addWidget(self.hang)
        tb.addWidget(QtWidgets.QLabel("  Spacing: "))
        self.spacing = QtWidgets.QComboBox()
        self.spacing.addItems(["Equal", "Proportional to distance"])
        self.spacing.currentIndexChanged.connect(self._display_changed)
        tb.addWidget(self.spacing)
        tb.addWidget(QtWidgets.QLabel("  GR cutoff: "))
        self.gr_cut = QtWidgets.QDoubleSpinBox()
        self.gr_cut.setRange(0, 300)
        self.gr_cut.setDecimals(0)
        self.gr_cut.setSuffix(" API")
        self.gr_cut.setKeyboardTracking(False)
        self.gr_cut.valueChanged.connect(self._display_changed)
        tb.addWidget(self.gr_cut)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Tracks: "))
        self.track_actions = {}
        for tid in TRACKS:
            a = QtGui.QAction(tid, self)
            a.setCheckable(True)
            a.setToolTip(TRACKS[tid][0])
            a.toggled.connect(self._display_changed)
            tb.addAction(a)
            self.track_actions[tid] = a
        tb.addWidget(QtWidgets.QLabel("  Fill between wells: "))
        self.between = QtWidgets.QComboBox()
        self.between.addItem("Lithology", "lithology")
        self.between.addItem("Formation zones", "zones")
        self.between.addItem("None", "none")
        self.between.setToolTip("Lithology: Vsh from normalised GR (plus coal / carbonate from density) carried\n"
                                "between wells along the correlated tops.\nFormation zones: colour between tops.")
        self.between.currentIndexChanged.connect(self._display_changed)
        tb.addWidget(self.between)
        from ..lithology import CODES
        legend = QtWidgets.QLabel("  " + "  ".join(f'<span style="color:{c}; font-size:14pt">&#9632;</span>{n}'
                                                    for n, c in CODES.values()))
        legend.setTextFormat(QtCore.Qt.RichText)
        tb.addWidget(legend)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Detail: "))
        self.detail = QtWidgets.QComboBox()
        self.detail.addItems(["Auto", "Always full"])
        self.detail.setToolTip("Auto: curves are thinned (peaks kept) when more than 600 m is in view,\n"
                               "full resolution when zoomed in. Always full: never thin.")
        self.detail.currentIndexChanged.connect(self._display_changed)
        tb.addWidget(self.detail)

    def _sync_controls(self):
        """Project display settings -> toolbar widgets (without triggering redraws)."""
        d = self.project.display
        widgets = [self.hang, self.spacing, self.gr_cut, self.detail, self.between] + list(self.track_actions.values())
        for w in widgets:
            w.blockSignals(True)
        self.hang.clear()
        self.hang.addItem("TVDSS (structural)", "TVDSS")
        self.hang.addItem("MD", "MD")
        for f in self.project.formations:
            self.hang.addItem(f"Flatten on {f['name']}", "FLAT:" + f["name"])
        i = self.hang.findData(d.get("hang", "TVDSS"))
        self.hang.setCurrentIndex(max(0, i))
        self.spacing.setCurrentIndex(1 if d.get("spacing") == "proportional" else 0)
        self.gr_cut.setValue(float(d.get("gr_cutoff", 75)))
        for tid, a in self.track_actions.items():
            a.setChecked(tid in d.get("tracks", []))
        self.detail.setCurrentIndex(1 if d.get("detail") == "full" else 0)
        self.between.setCurrentIndex(max(0, self.between.findData(d.get("between", "lithology"))))
        for w in widgets:
            w.blockSignals(False)

    def _display_changed(self, *_):
        if not self.project:
            return
        d = self.project.display
        d["hang"] = self.hang.currentData() or "TVDSS"
        d["spacing"] = "proportional" if self.spacing.currentIndex() == 1 else "equal"
        d["gr_cutoff"] = self.gr_cut.value()
        d["tracks"] = [t for t in TRACKS if self.track_actions[t].isChecked()] or ["GR"]
        d["detail"] = "full" if self.detail.currentIndex() == 1 else "auto"
        d.pop("max_points", None)
        d["between"] = self.between.currentData() or "none"
        self.section.lod_full_below = 1e12 if d["detail"] == "full" else 600.0
        self.dirty = True
        keep = sender_is(self.sender(), self.gr_cut)
        self.section.redraw(keep_view=keep)
        self._update_title()

    # ---------------------------------------------------------------- project
    def new_project(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose the data folder (contains wells\\<Well>\\las and wells_header.csv)",
            self.settings.value("last_dir", os.path.expanduser("~")))
        if not folder or not self._maybe_save():
            return
        self.settings.setValue("last_dir", folder)
        prog = _progress(self, "Scanning wells...", 0)
        prog.show()

        def cb(i, n, name):
            prog.setMaximum(n)
            prog.setValue(i)
            prog.setLabelText(f"Scanning {name}")
            QtWidgets.QApplication.processEvents()

        try:
            p = Project.from_folder(folder, progress=cb)
        except Exception as e:
            prog.close()
            self._error("Could not read the folder", e)
            return
        prog.close()
        if not p.wells:
            QtWidgets.QMessageBox.warning(self, "No wells", "No well folders were found. See Help > Data folder layout.")
            return
        default = os.path.join(folder, p.name + PROJECT_EXT)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project as", default, f"Well correlation project (*{PROJECT_EXT})")
        if not path:
            return
        p.save(path)
        self._set_project(p)

    def open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open project", self.settings.value("last_dir", ""),
                                                        f"Well correlation project (*{PROJECT_EXT});;All files (*)")
        if path:
            self.open_project(path)

    def open_project(self, path):
        if not self._maybe_save():
            return
        try:
            p = Project.open(path)
        except Exception as e:
            self._error("Could not open the project", e)
            return
        self.settings.setValue("last_dir", os.path.dirname(path))
        self._set_project(p)

    def _set_project(self, p):
        self.project = p
        self.dirty = False
        self._add_recent(p.path)
        self.wells_list.clear()
        for w in p.wells.values():
            it = QtWidgets.QListWidgetItem(w.name)
            if not w.las_paths:
                it.setForeground(QtGui.QBrush(QtGui.QColor(150, 150, 150)))
                it.setToolTip("No LAS files found")
            self.wells_list.addItem(it)
        self.section.lod_full_below = 1e12 if p.display.get("detail") == "full" else 600.0
        self._sync_controls()
        self.map.set_project(p)
        self.section.set_project(p)
        self._refresh_section_combo()
        self.undo_stack, self.redo_stack = [], []
        self._refresh_formations()
        self._apply_section()
        self._update_title()

    def save(self):
        if not self.project:
            return
        if not self.project.path:
            return self.save_as()
        try:
            self.project.save()
            self.dirty = False
            self._update_title()
            self.status.showMessage("Saved", 3000)
        except Exception as e:
            self._error("Save failed", e)

    def save_as(self):
        if not self.project:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project as", self.project.path or "",
                                                        f"Well correlation project (*{PROJECT_EXT})")
        if path:
            self.project.save(path)
            self.dirty = False
            self._add_recent(self.project.path)
            self._update_title()

    def reload_logs(self):
        if not self.project:
            return
        for n in self.project.section_wells():
            self.project.wells[n].loaded = False
        self._load_with_progress(self.project.section_wells(), force=True)
        self.section.redraw()

    def export_image(self):
        if not self.project or not self.project.section_wells():
            return
        base = os.path.splitext(self.project.path or "section")[0].replace(".wcproj", "")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export section", f"{base}_{self.project.current_section}.png",
                                                        "PNG image (*.png);;PDF (*.pdf)")
        if not path:
            return
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            export_section(self.project, self.project.section_wells(), path, ylim=self.section.view_depth_range())
            self.status.showMessage(f"Exported {path}", 5000)
        except Exception as e:
            self._error("Export failed", e)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # ---------------------------------------------------------------- sections
    def _refresh_section_combo(self):
        self.sec_combo.blockSignals(True)
        self.sec_combo.clear()
        self.sec_combo.addItems(list(self.project.sections))
        self.sec_combo.setCurrentText(self.project.current_section)
        self.sec_combo.blockSignals(False)

    def _switch_section(self, name):
        if self.project and name in self.project.sections:
            self.project.current_section = name
            self._apply_section()

    def new_section(self):
        if not self.project:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "New section", "Name:", text=f"Section {len(self.project.sections) + 1}")
        if ok and name:
            self.project.sections[name] = []
            self.project.current_section = name
            self._refresh_section_combo()
            self._apply_section()
            self.dirty = True

    def set_section_wells(self, names):
        if not self.project:
            return
        self.project.sections[self.project.current_section] = list(names)
        self.dirty = True
        self._apply_section()

    def toggle_well(self, name):
        if not self.project:
            return
        cur = self.project.sections.setdefault(self.project.current_section, [])
        if name in cur:
            cur.remove(name)
        else:
            cur.append(name)
        self.dirty = True
        self._apply_section()

    def remove_selected_from_section(self):
        it = self.sec_list.currentItem()
        if it:
            self.toggle_well(it.text())

    def reverse_section(self):
        if self.project:
            self.project.sections[self.project.current_section].reverse()
            self.dirty = True
            self._apply_section()

    def clear_section(self):
        if self.project:
            self.project.sections[self.project.current_section] = []
            self.dirty = True
            self._apply_section()

    def _section_reordered(self, *args):
        names = [self.sec_list.item(i).text() for i in range(self.sec_list.count())]
        self.project.sections[self.project.current_section] = names
        self.dirty = True
        QtCore.QTimer.singleShot(0, self._apply_section)

    def _apply_section(self):
        names = self.project.section_wells()
        self.sec_list.blockSignals(True)
        self.sec_list.clear()
        self.sec_list.addItems(names)
        self.sec_list.blockSignals(False)
        self._load_with_progress(names)
        self.map.set_section(names)
        self.section.set_wells(names)
        self._update_title()

    def _load_with_progress(self, names, force=False):
        todo = [n for n in names if force or not self.project.wells[n].loaded]
        if not todo:
            return
        prog = None
        if len(todo) > 1:
            prog = _progress(self, "Loading logs...", len(todo))
            prog.setMinimumDuration(300)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for i, n in enumerate(todo):
                if prog:
                    prog.setLabelText(f"Loading {n} ({i + 1}/{len(todo)})")
                    prog.setValue(i)
                    QtWidgets.QApplication.processEvents()
                try:
                    self.project.load_well(n, force=force)
                except Exception as e:
                    self.status.showMessage(f"{n}: {e}", 8000)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            if prog:
                prog.close()

    # ---------------------------------------------------------------- info
    def select_well(self, name):
        if not self.project or name not in self.project.wells:
            return
        self.map.set_selected(name)
        self.section.set_selected(name)
        w = self.project.wells[name]
        try:
            self.project.load_well(name)
        except Exception:
            pass
        rows = [("UWI", w.uwi), ("Lat / Lon", f"{w.lat}, {w.lon}" if w.lat is not None else "-"),
                ("NZTM E / N", f"{w.x:.0f}, {w.y:.0f}" if w.x is not None else "-"),
                ("Elevation", f"{w.elev_ref} {w.elev} m above MSL"), ("TD", f"{w.td} m" if w.td else "-"),
                ("Deviation survey", os.path.basename(w.survey_path) if w.survey_path else "none (vertical assumed)")]
        for k in ("Operator", "Purpose", "Status", "Spud_Date"):
            if w.info.get(k):
                rows.append((k.replace("_", " "), w.info[k]))
        html = [f"<h3>{name}</h3><table>"]
        html += [f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in rows]
        html.append("</table><h4>Curves used (best first)</h4>")
        if not w.logs:
            html.append("<p><i>No usable log curves. Check the LAS folder (DLIS files need converting).</i></p>")
        for fam, src in w.sources.items():
            html.append(f"<b>{fam}</b><br>" + "<br>".join(f"&nbsp;&nbsp;{s}" for s in src) + "<br>")
        html.append(f"<h4>LAS files ({len(w.las_paths)})</h4>" + "<br>".join(os.path.basename(p) for p in w.las_paths))
        self.info.setHtml("".join(html))
        self._refresh_tops_table()

    # ---------------------------------------------------------------- tops
    def _build_tops_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        self.btn_pick = QtWidgets.QPushButton("Pick tops (P)")
        self.btn_pick.setCheckable(True)
        self.btn_pick.toggled.connect(lambda on: self.act_pick.setChecked(on) or self._pick_changed(on))
        lay.addWidget(self.btn_pick)
        lay.addWidget(QtWidgets.QLabel("Formations (top to bottom) - selected = active:"))
        self.fm_list = QtWidgets.QListWidget()
        self.fm_list.currentRowChanged.connect(self._fm_row_changed)
        self.fm_list.itemDoubleClicked.connect(lambda it: self.rename_formation())
        lay.addWidget(self.fm_list, 2)
        row = QtWidgets.QHBoxLayout()
        for text, tip, fn in (("Add", "New formation below the selected one", self.add_formation),
                              ("Rename", "Rename (or double-click)", self.rename_formation),
                              ("Colour", "Change colour", self.recolor_formation),
                              ("\u25b2", "Move up", lambda: self.move_formation(-1)),
                              ("\u25bc", "Move down", lambda: self.move_formation(1)),
                              ("Delete", "Delete formation and all its tops", self.delete_formation)):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            b.setMaximumWidth(60)
            row.addWidget(b)
        lay.addLayout(row)
        self.tops_label = QtWidgets.QLabel("Tops in selected well:")
        lay.addWidget(self.tops_label)
        self.tops_table = QtWidgets.QTableWidget(0, 4)
        self.tops_table.setHorizontalHeaderLabels(["Formation", "MD (m)", "TVDSS (m)", "Source"])
        self.tops_table.horizontalHeader().setStretchLastSection(True)
        self.tops_table.verticalHeader().setVisible(False)
        self.tops_table.setToolTip("Edit MD to move a top exactly. Clear the cell to delete it.")
        self.tops_table.itemChanged.connect(self._tops_cell_changed)
        lay.addWidget(self.tops_table, 3)
        hint = QtWidgets.QLabel("Pick mode: click = set active top, drag a top = move it, "
                                "Ctrl+click = choose it, \u2191/\u2193 nudge, Del removes.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#555")
        lay.addWidget(hint)
        return w

    def toggle_pick(self):
        on = self.act_pick.isChecked()
        self.btn_pick.blockSignals(True)
        self.btn_pick.setChecked(on)
        self.btn_pick.blockSignals(False)
        self._pick_changed(on)

    def _pick_changed(self, on):
        self.section.set_pick_mode(on)
        if on and not self.section.active_formation and self.fm_list.count():
            self.fm_list.setCurrentRow(0)
        if on:
            self.status.showMessage(f"Pick mode - active top: {self.section.active_formation or '(none)'}", 6000)
            self.section.setFocus()

    def _refresh_formations(self, keep=None):
        keep = keep or self.section.active_formation
        self.fm_list.blockSignals(True)
        self.fm_list.clear()
        for f in self.project.formations if self.project else []:
            pm = QtGui.QPixmap(14, 14)
            pm.fill(QtGui.QColor(f.get("color", "#999999")))
            n = sum(1 for t in self.project.tops.values() if f["name"] in t)
            it = QtWidgets.QListWidgetItem(QtGui.QIcon(pm), f"{f['name']}  ({n})")
            it.setData(QtCore.Qt.UserRole, f["name"])
            self.fm_list.addItem(it)
        names = [f["name"] for f in self.project.formations] if self.project else []
        if keep in names:
            self.fm_list.setCurrentRow(names.index(keep))
        self.fm_list.blockSignals(False)
        self.section.active_formation = keep if keep in names else None

    def _fm_row_changed(self, row):
        it = self.fm_list.item(row) if row >= 0 else None
        name = it.data(QtCore.Qt.UserRole) if it else None
        self.section.set_active_formation(name)
        if name:
            self.status.showMessage(f"Active top: {name}", 4000)

    def select_formation(self, name):
        for i in range(self.fm_list.count()):
            if self.fm_list.item(i).data(QtCore.Qt.UserRole) == name:
                self.fm_list.setCurrentRow(i)
                return

    def _current_fm(self):
        it = self.fm_list.currentItem()
        return it.data(QtCore.Qt.UserRole) if it else None

    def _refresh_tops_table(self):
        name = self.section.selected
        self.tops_table.blockSignals(True)
        self.tops_table.setRowCount(0)
        if self.project and name in self.project.wells:
            self.tops_label.setText(f"Tops in {name}:")
            w = self.project.wells[name]
            tops = self.project.tops.get(name, {})
            for f in self.project.formations:
                fm = f["name"]
                r = self.tops_table.rowCount()
                self.tops_table.insertRow(r)
                a = QtWidgets.QTableWidgetItem(fm)
                a.setFlags(a.flags() & ~QtCore.Qt.ItemIsEditable)
                a.setForeground(QtGui.QBrush(QtGui.QColor(f.get("color", "#000000"))))
                md = tops.get(fm)
                b = QtWidgets.QTableWidgetItem("" if md is None else f"{md:.1f}")
                b.setData(QtCore.Qt.UserRole, fm)
                try:
                    z = "" if md is None else f"{float(w.md_to_tvdss([md])[0]):.1f}"
                except Exception:
                    z = ""
                c = QtWidgets.QTableWidgetItem(z)
                c.setFlags(c.flags() & ~QtCore.Qt.ItemIsEditable)
                meta = self.project.tops_meta.get(name, {}).get(fm, {})
                src = meta.get("source", "") + (f" ({meta['confidence']})" if meta.get("confidence") else "")
                d = QtWidgets.QTableWidgetItem(src if md is not None else "")
                d.setFlags(d.flags() & ~QtCore.Qt.ItemIsEditable)
                for col, item in enumerate((a, b, c, d)):
                    self.tops_table.setItem(r, col, item)
        else:
            self.tops_label.setText("Tops in selected well: (click a well)")
        self.tops_table.resizeColumnsToContents()
        self.tops_table.blockSignals(False)

    def _tops_cell_changed(self, item):
        if item.column() != 1 or not self.section.selected:
            return
        fm = item.data(QtCore.Qt.UserRole)
        txt = item.text().strip()
        if not txt:
            self._edit(lambda: T.set_top(self.project, self.section.selected, fm, None))
            return
        try:
            md = float(txt)
        except ValueError:
            self._refresh_tops_table()
            return
        self._edit(lambda: T.set_top(self.project, self.section.selected, fm, md))

    def _edit(self, fn, formations_changed=False):
        """Run an edit function that returns its inverse; record for undo; refresh."""
        if not self.project:
            return
        try:
            inv = fn()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Tops", str(e))
            return
        self.undo_stack.append(inv)
        self.redo_stack.clear()
        self._after_edit(formations_changed or inv.get("op") == "formations", inv.get("fm"))

    def _after_edit(self, formations_changed, fm=None):
        self.dirty = True
        self._refresh_formations()
        if formations_changed:
            self.map.set_formations()
        else:
            self.map.refresh_map()
        hang = self.project.display.get("hang", "")
        if formations_changed:
            self._sync_controls()
        if formations_changed or (hang.startswith("FLAT:") and (fm is None or hang[5:] == fm)):
            self.section.redraw(keep_view=True)  # depths of the logs move: full redraw
        else:
            self.section.refresh_tops()  # only tops + between-well fills
        self._refresh_tops_table()
        self._update_title()

    def undo(self):
        if self.undo_stack:
            inv = T.apply_edit(self.project, self.undo_stack.pop())
            self.redo_stack.append(inv)
            self._after_edit(inv.get("op") == "formations", inv.get("fm"))
            self.status.showMessage("Undone", 2000)

    def redo(self):
        if self.redo_stack:
            inv = T.apply_edit(self.project, self.redo_stack.pop())
            self.undo_stack.append(inv)
            self._after_edit(inv.get("op") == "formations", inv.get("fm"))
            self.status.showMessage("Redone", 2000)

    def on_top_edited(self, well, fm, md):
        self._edit(lambda: T.set_top(self.project, well, fm, md))
        self.status.showMessage(f"{fm} in {well} set to MD {md:.1f} m", 4000)

    def nudge_top(self, delta):
        well, fm = self.section.selected, self.section.active_formation
        if not (self.project and well and fm):
            return
        md = self.project.tops.get(well, {}).get(fm)
        if md is not None:
            self._edit(lambda: T.set_top(self.project, well, fm, md + delta))

    def delete_active_top(self):
        well, fm = self.section.selected, self.section.active_formation
        if self.project and well and fm and fm in self.project.tops.get(well, {}):
            self._edit(lambda: T.set_top(self.project, well, fm, None))
            self.status.showMessage(f"Deleted {fm} in {well}", 4000)

    def add_formation(self):
        if not self.project:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Add formation", "Name:")
        if ok and name.strip():
            row = self.fm_list.currentRow()
            self._edit(lambda: T.add_formation(self.project, name.strip(), index=row + 1 if row >= 0 else None), True)
            self.select_formation(name.strip())

    def rename_formation(self):
        old = self._current_fm()
        if not old:
            return
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename formation", "New name:", text=old)
        if ok and new.strip() and new.strip() != old:
            self._edit(lambda: T.rename_formation(self.project, old, new.strip()), True)
            self.select_formation(new.strip())

    def recolor_formation(self):
        fm = self._current_fm()
        if not fm:
            return
        cur = next((f.get("color", "#999999") for f in self.project.formations if f["name"] == fm), "#999999")
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(cur), self, f"Colour for {fm}")
        if c.isValid():
            self._edit(lambda: T.recolor_formation(self.project, fm, c.name()), True)

    def move_formation(self, delta):
        fm = self._current_fm()
        if fm:
            self._edit(lambda: T.move_formation(self.project, fm, delta), True)
            self.select_formation(fm)

    def delete_formation(self):
        fm = self._current_fm()
        if not fm:
            return
        n = sum(1 for t in self.project.tops.values() if fm in t)
        r = QtWidgets.QMessageBox.question(self, "Delete formation", f"Delete '{fm}' and its {n} tops?\n(Ctrl+Z undoes this)")
        if r == QtWidgets.QMessageBox.Yes:
            self._edit(lambda: T.delete_formation(self.project, fm), True)

    def import_tops(self):
        if not self.project:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import tops", self.settings.value("last_dir", ""),
                                                        "Text (*.csv *.txt *.tsv);;All files (*)")
        if not path:
            return
        result = {}

        def run():
            n, unknown, inv = T.import_csv(self.project, path)
            result.update(n=n, unknown=unknown)
            return inv

        self._edit(run, True)
        if result:
            msg = f"Imported {result['n']} tops."
            if result["unknown"]:
                msg += "\n\nWells not in this project (kept, not shown):\n" + ", ".join(result["unknown"][:20])
            QtWidgets.QMessageBox.information(self, "Import tops", msg)

    def export_tops(self, kind):
        if not self.project:
            return
        ext = {"csv": ("tops.csv", "CSV (*.csv)"), "odt": ("tops_opendtect.txt", "Text (*.txt)"),
               "kingdom": ("tops_kingdom.txt", "Text (*.txt)")}[kind]
        base = os.path.dirname(self.project.path or "")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export tops", os.path.join(base, ext[0]), ext[1])
        if not path:
            return
        for n in self.project.tops:
            if n in self.project.wells and not self.project.wells[n].loaded:
                try:
                    self.project.load_well(n)
                except Exception:
                    pass
        {"csv": T.export_csv, "odt": T.export_opendtect, "kingdom": T.export_kingdom}[kind](self.project, path)
        self.status.showMessage(f"Exported {path}", 5000)

    # ---------------------------------------------------------------- step 3: correlate
    def _run_suggest(self, wells, formations):
        from ..autocorr import suggest_tops, confidence
        props = dict(self.section.proposals)
        n = 0
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for wn in wells:
                r = suggest_tops(self.project, wn, formations=formations)
                for fm, (md, score, pred) in r.items():
                    props.setdefault(wn, {})[fm] = (md, score, "suggest")
                    n += 1
                QtWidgets.QApplication.processEvents()
        except Exception as e:
            self._error("Suggest pick failed", e)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.section.set_proposals(props)
        if n:
            lines = [f"{w}: {fm} MD {md:.1f} (score {sc:.2f}, {confidence(sc)})"
                     for w, d in props.items() for fm, (md, sc, src) in d.items() if sc is not None and w in wells]
            self.status.showMessage(f"{n} suggestion(s) shown in magenta - Enter accepts, Esc clears.  " + "; ".join(lines[:4]), 15000)
        else:
            self.status.showMessage("No suggestion: the formation needs to be picked in at least one other well "
                                    "with logs over that depth", 8000)
        self.section.setFocus()

    def suggest_selected(self):
        wn, fm = self.section.selected, self.section.active_formation
        if not (self.project and wn and fm):
            self.status.showMessage("Select a well (click it) and an active formation first", 6000)
            return
        self._run_suggest([wn], [fm])

    def suggest_section(self):
        fm = self.section.active_formation
        if not (self.project and fm):
            self.status.showMessage("Choose an active formation in the Tops panel first", 6000)
            return
        wells = [n for n in self.project.section_wells() if fm not in self.project.tops.get(n, {})]
        if not wells:
            self.status.showMessage(f"Every section well already has {fm}", 6000)
            return
        self._run_suggest(wells, [fm])

    def suggest_all_formations(self):
        wn = self.section.selected
        if not (self.project and wn):
            self.status.showMessage("Select a well first", 6000)
            return
        self._run_suggest([wn], None)

    def accept_proposals(self):
        props = self.section.proposals
        if not props or not self.project:
            return
        from ..autocorr import confidence
        items = []
        for wn, d in props.items():
            for fm, (md, score, src) in d.items():
                extra = {"how": src, "source": "auto" if src == "suggest" else "user"}
                if score is not None:
                    extra.update(score=round(score, 2), confidence=confidence(score))
                items.append((wn, fm, md, extra))
        self.section.proposals = {}
        self._edit(lambda: T.set_tops_batch(self.project, items))
        self.status.showMessage(f"Accepted {len(items)} top(s) (Ctrl+Z undoes)", 5000)

    def clear_proposals(self):
        if self.section._ghost is not None:
            self.section.cancel_ghost()
            self.status.showMessage("Ghost cancelled", 3000)
            return
        if self.section.proposals:
            self.section.set_proposals({})
            self.status.showMessage("Suggestions cleared", 3000)

    def start_ghost(self):
        if not self.section.start_ghost():
            self.status.showMessage("Hover over a well at the depth to copy, then press G", 6000)
        self.section.setFocus()

    def show_reference(self):
        if not self.project:
            return
        if getattr(self, "_ref_dlg", None) is None:
            self._ref_dlg = ReferenceDialog(self)
        self._ref_dlg.refresh()
        self._ref_dlg.show()
        self._ref_dlg.raise_()

    def show_petro(self):
        """Petrophysics window (PLAN 20e) on the selected (or first section) well."""
        from .petro_dialog import PetroDialog
        if not self.project:
            return None
        if getattr(self, "_petro_dlg", None) is None or self._petro_dlg.project is not self.project:
            self._petro_dlg = PetroDialog(self)
        sel = getattr(self.section, "selected", None)
        if sel and sel in self.project.wells and sel != self._petro_dlg.well:
            self._petro_dlg.set_well(sel)
        self._petro_dlg.show()
        self._petro_dlg.raise_()
        return self._petro_dlg

    def show_zone_stats(self):
        if not self.project:
            return
        if getattr(self, "_zone_dlg", None) is None:
            self._zone_dlg = ZoneStatsDialog(self)
        self._zone_dlg.refresh()
        self._zone_dlg.show()
        self._zone_dlg.raise_()

    # ---------------------------------------------------------------- misc
    def _update_title(self):
        t = "Well Correlation Tool"
        if self.project:
            t += f" - {self.project.name}" + (" *" if self.dirty else "")
            t += f"   [{self.project.current_section}: {len(self.project.section_wells())} wells]"
        self.setWindowTitle(t)

    def _maybe_save(self):
        if not self.project or not self.dirty:
            return True
        r = QtWidgets.QMessageBox.question(self, "Unsaved changes", "Save changes to the current project?",
                                           QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
        if r == QtWidgets.QMessageBox.Save:
            self.save()
            return not self.dirty
        return r == QtWidgets.QMessageBox.Discard

    def closeEvent(self, ev):
        if self._maybe_save():
            ev.accept()
        else:
            ev.ignore()

    def _fill_recent(self):
        self.recent_menu.clear()
        for p in self._recent():
            a = self.recent_menu.addAction(p)
            a.triggered.connect(lambda checked=False, path=p: self.open_project(path))

    def _recent(self):
        v = self.settings.value("recent", [])
        if isinstance(v, str):
            v = [v]
        return [p for p in (v or []) if os.path.exists(p)]

    def _add_recent(self, path):
        if not path:
            return
        r = [path] + [p for p in self._recent() if os.path.normcase(p) != os.path.normcase(path)]
        self.settings.setValue("recent", r[:8])
        self._fill_recent()

    def _error(self, title, e):
        QtWidgets.QMessageBox.critical(self, title, f"{e}\n\n{traceback.format_exc()[-1500:]}")

    def _default_project(self):
        """Last project used, or the single project file next to the tool folder."""
        r = self._recent()
        if r:
            return r[0]
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        found = glob.glob(os.path.join(os.path.dirname(here), "*" + PROJECT_EXT))
        return found[0] if len(found) == 1 else None

    def _show_start_hint(self):
        self.status.showMessage("File > Open project, or File > New project from data folder", 15000)

    def help_layout(self):
        QtWidgets.QMessageBox.information(self, "Data folder layout", (
            "data folder\\\n"
            "    wells_header.csv   (Well, UWI, Latitude, Longitude, Elevation_m, Elev_Ref, TD_m, ...)\n"
            "    wells\\<Well>\\las\\*.las      any number of LAS files per well\n"
            "    wells\\<Well>\\survey\\*.csv   optional MD, INC, AZI\n\n"
            "Curves are matched by mnemonic (GR, ILD, LLD, RHOB, NPHI, DT, ...). Runs and passes are\n"
            "spliced automatically; the Well info panel shows which curves were used."))

    def help_mouse(self):
        QtWidgets.QMessageBox.information(self, "Mouse and keys", (
            "Cross section:\n"
            "  Wheel = vertical zoom, Ctrl+wheel = horizontal zoom, drag = pan\n"
            "  Click a well = select it (info panel, map highlight)\n"
            "  Right-click = View all, F = fit section\n\n"
            "Map:\n"
            "  Click a well = add/remove it from the section\n"
            "  'Draw section line' = click points, double-click to finish; wells within the\n"
            "  buffer are added in order along the line\n\n"
            "Section wells list: drag to reorder, Delete to remove\n\n"
            "Tops (press P or the Pick tops button):\n"
            "  Choose the active formation in the Tops panel\n"
            "  Click in a well = set that top at the cursor\n"
            "  Drag a top line = move it (any formation; it becomes active)\n"
            "  Ctrl+click a top = make its formation active\n"
            "  Up/Down = nudge 0.5 m, Shift+Up/Down = 5 m, Delete = remove (selected well)\n"
            "  Ctrl+Z / Ctrl+Y = undo / redo\n"
            "  Line style: solid = published or yours, dashed = auto pick, dotted + ? = low-confidence auto\n\n"
            "Correlate:\n"
            "  S = suggest the active top in the selected well, Shift+S = in every section well missing it\n"
            "  Alt+S = suggest all formations in the selected well\n"
            "  Suggestions are magenta with a match score (0-1). Enter accepts all, Esc clears\n"
            "  G = ghost: hover a well at the depth to copy, press G, move over another well,\n"
            "      wheel or [ ] to stretch, click to drop (its tops become suggestions)\n"
            "  Z = zone statistics"))


class ReferenceDialog(QtWidgets.QDialog):
    """Compare current tops with a reference set (published or a saved snapshot)."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Compare tops with reference")
        self.resize(900, 560)
        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.combo = QtWidgets.QComboBox()
        self.combo.currentIndexChanged.connect(self.refresh_table)
        self.tol = QtWidgets.QDoubleSpinBox()
        self.tol.setRange(0.5, 200)
        self.tol.setValue(10)
        self.tol.setPrefix("Flag if moved > ")
        self.tol.setSuffix(" m")
        self.tol.valueChanged.connect(self.refresh_table)
        self.show_ticks = QtWidgets.QCheckBox("Show as grey ticks in section")
        self.show_ticks.toggled.connect(self._ticks)
        b_snap = QtWidgets.QPushButton("Save current tops as reference...")
        b_snap.clicked.connect(self._snapshot)
        for wdg in (QtWidgets.QLabel("Reference:"), self.combo, self.tol, self.show_ticks):
            row.addWidget(wdg)
        row.addStretch(1)
        row.addWidget(b_snap)
        lay.addLayout(row)
        self.notes = QtWidgets.QLabel()
        self.notes.setWordWrap(True)
        self.notes.setStyleSheet("background:#fff8e0; padding:6px")
        lay.addWidget(self.notes)
        self.table = QtWidgets.QTableWidget()
        self.table.setSortingEnabled(True)
        lay.addWidget(self.table, 1)

    def refresh(self):
        p = self.win.project
        cur = self.combo.currentText()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(list(p.reference_tops))
        if cur in p.reference_tops:
            self.combo.setCurrentText(cur)
        self.combo.blockSignals(False)
        self.show_ticks.blockSignals(True)
        self.show_ticks.setChecked(bool(p.display.get("show_reference")))
        self.show_ticks.blockSignals(False)
        self.refresh_table()

    def refresh_table(self, *_):
        from ..tops import compare_with_reference
        p = self.win.project
        name = self.combo.currentText()
        self.notes.setText(p.reference_tops.get(name, {}).get("notes", "") or "(no notes)")
        rows = compare_with_reference(p, name, self.tol.value()) if name else []
        cols = ["Well", "Formation", "Current_MD", "Reference_MD", "Diff_m", "Source", "Flag"]
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels([c.replace("_", " ") for c in cols])
        self.table.setRowCount(len(rows))
        colors = {"moved": "#ffd6d6", "missing": "#ffe9c7", "not in reference": "#e8f0ff"}
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                v = r[c]
                it = QtWidgets.QTableWidgetItem()
                if isinstance(v, (int, float)):
                    it.setData(QtCore.Qt.DisplayRole, float(v))
                else:
                    it.setText("" if v is None else str(v))
                if r["Flag"] in colors:
                    it.setBackground(QtGui.QBrush(QtGui.QColor(colors[r["Flag"]])))
                self.table.setItem(i, j, it)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        if self.show_ticks.isChecked():
            self._ticks(True)

    def _ticks(self, on):
        p = self.win.project
        p.display["show_reference"] = self.combo.currentText() if on else None
        self.win.section.refresh_tops(new_layout=False)

    def _snapshot(self):
        from ..tops import add_reference_from_current
        name, ok = QtWidgets.QInputDialog.getText(self, "Save reference", "Name for this set of tops:",
                                                  text=time.strftime("My tops %Y-%m-%d %H%M"))
        if ok and name:
            add_reference_from_current(self.win.project, name, "Snapshot of the project tops.")
            self.win.dirty = True
            self.refresh()
            self.combo.setCurrentText(name)


class ZoneStatsDialog(QtWidgets.QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Zone statistics")
        self.resize(1100, 500)
        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.only_section = QtWidgets.QCheckBox("Section wells only")
        self.only_section.setChecked(True)
        self.only_section.toggled.connect(self.refresh)
        self.cut = QtWidgets.QDoubleSpinBox()
        self.cut.setRange(0.05, 0.95)
        self.cut.setSingleStep(0.05)
        self.cut.setValue(0.35)
        self.cut.setPrefix("Net sand: Vsh < ")
        self.cut.valueChanged.connect(self.refresh)
        b_ref = QtWidgets.QPushButton("Refresh")
        b_ref.clicked.connect(self.refresh)
        b_exp = QtWidgets.QPushButton("Export CSV...")
        b_exp.clicked.connect(self.export)
        for wdg in (self.only_section, self.cut):
            row.addWidget(wdg)
        row.addStretch(1)
        row.addWidget(b_ref)
        row.addWidget(b_exp)
        lay.addLayout(row)
        self.table = QtWidgets.QTableWidget()
        self.table.setSortingEnabled(True)
        lay.addWidget(self.table)
        note = QtWidgets.QLabel("Zones run from each top to the next top present in that well. Vsh from GR normalised "
                                "in a moving 400 m window; PHID from density (matrix 2.65, fluid 1.0) in net sand.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#555")
        lay.addWidget(note)
        self.rows = []

    def refresh(self, *_):
        from ..zones import zone_table, COLUMNS
        p = self.win.project
        wells = p.section_wells() if self.only_section.isChecked() else None
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.rows = zone_table(p, wells, vsh_cutoff=self.cut.value())
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(COLUMNS))
        self.table.setHorizontalHeaderLabels([c.replace("_", " ") for c in COLUMNS])
        self.table.setRowCount(len(self.rows))
        for i, r in enumerate(self.rows):
            for j, c in enumerate(COLUMNS):
                v = r.get(c)
                it = QtWidgets.QTableWidgetItem()
                if isinstance(v, (int, float)) and v is not None:
                    it.setData(QtCore.Qt.DisplayRole, float(v))
                else:
                    it.setText("" if v is None else str(v))
                self.table.setItem(i, j, it)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()

    def export(self):
        from ..zones import export_zone_csv
        base = os.path.dirname(self.win.project.path or "")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export zone statistics",
                                                        os.path.join(base, "zone_statistics.csv"), "CSV (*.csv)")
        if path:
            export_zone_csv(self.rows, path)


def _progress(parent, text, maximum):
    d = QtWidgets.QProgressDialog(parent)
    d.setLabelText(text)
    d.setCancelButton(None)
    d.setRange(0, maximum)
    d.setWindowModality(QtCore.Qt.WindowModal)
    d.setWindowTitle("Well Correlation Tool")
    return d


def sender_is(sender, widget):
    return sender is widget
