"""Map layers dock (PLAN 21a): the map hub's layer list - visibility, order, opacity, colour ramp, contour interval,
add grids (current basemap grid, CPS-3 file, horizon time map), petro values (21b), polygons and text; saved with the
project (display["map_layers"]).  Logic in plain methods (testable under the Qt mock)."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtWidgets  # noqa: E402

from .. import maplayers as ML  # noqa: E402
from . import colormaps  # noqa: E402


class LayersPanel(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, basemap, parent=None):
        super().__init__(parent)
        self.basemap = basemap
        self.stack = None
        self._ids = []
        self._sel = None
        self.last_report = ""
        self.analytics_hook = None  # set by the main window: callable(AttributeTable) -> Analytics window
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self.list = QtWidgets.QListWidget()
        self.list.itemChanged.connect(self._item_changed)
        self.list.currentRowChanged.connect(lambda *_: self._sync_controls())
        lay.addWidget(self.list)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("Up", lambda *_: self.move(+1)), ("Down", lambda *_: self.move(-1)),
                         ("Remove", lambda *_: self.remove())):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        self.add_menu = QtWidgets.QToolButton()
        self.add_menu.setText("Add...")
        self.add_menu.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        m = QtWidgets.QMenu(self.add_menu)
        for text, fn in (("Current basemap grid", lambda *_: self.add_current_grid()),
                         ("CPS-3 grid file...", lambda *_: self._add_cps3_dialog()),
                         ("Horizon time map...", lambda *_: self._add_horizon_dialog()),
                         ("Petrophysics value per well...", lambda *_: self._add_petro_dialog()),
                         ("Mineral pies per well (mineral solver)...", lambda *_: self._add_pies_dialog()),
                         ("Text...", lambda *_: self._add_text_dialog()),
                         ("Co-blend 2-3 grids (RGB / CMY / HSV)...", lambda *_: self._add_blend_dialog()),
                         ("Shade selected grid by structure (amplitude on structure / bump map)...",
                          lambda *_: self._shade_dialog()),
                         ("Grid a zone property (IDW / kriging / KED / cokriging)...", lambda *_: self.prop_grid()),
                         ("Grid calculator...", lambda *_: self.grid_calc()),
                         ("Cross-check petrophysics vs maps (+ predicted maps, Analytics)...", lambda *_: self.cross_check()),
                         ("Vector layer (shapefile / GeoPackage / KML / CSV)...", lambda *_: self._add_vector_dialog()),
                         ("Project coordinate system (EPSG)...", lambda *_: self._crs_dialog()),
                         ("Export for QGIS / ArcGIS (GeoTIFF, shapefile, GeoPackage, KML)...", lambda *_: self._export_gis_dialog()),
                         ("Go to legal description (PLSS)...", lambda *_: self._legal_dialog()),
                         ("Attribute table of the selected layer...", lambda *_: self._attr_dialog()),
                         ("New wells: import + petro + re-grid + tie...", lambda *_: self._add_wells_dialog())):
            a = m.addAction(text)
            a.triggered.connect(fn)
        self.add_menu.setMenu(m)
        row.addWidget(self.add_menu)
        lay.addLayout(row)
        row2 = QtWidgets.QHBoxLayout()
        for text, fn, tip in (("Draw polygon", lambda *_: self.start_polygon(),
                               "Click vertices on the basemap, double-click to close: the polygon is kept as a layer and "
                               "the wells inside it are selected"),
                              ("Petro cross-plot...", lambda *_: self.crossplot(),
                               "Zone values per well (or a map grid at the wells) - selected wells highlighted")):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row2.addWidget(b)
        lay.addLayout(row2)
        row3 = QtWidgets.QHBoxLayout()
        self.preset = QtWidgets.QComboBox()
        from ..welltie import wc
        wc()
        from well_correlation.carto import STYLE_PRESETS
        self.preset.addItems(list(STYLE_PRESETS))
        self.preset.setToolTip("Map style preset: hillshaded perceptual colour fill, labelled index contours, well symbols")
        b_style = QtWidgets.QPushButton("Apply style")
        b_style.clicked.connect(lambda *_: self.apply_preset(self.preset.currentText()))
        b_exp = QtWidgets.QPushButton("Export map...")
        b_exp.setToolTip("PNG / PDF at 300 dpi with title block, legend, scale bar and north arrow")
        b_exp.clicked.connect(lambda *_: self._export_dialog())
        for w in (self.preset, b_style, b_exp):
            row3.addWidget(w)
        lay.addLayout(row3)
        self._xplot = None
        if basemap is not None:
            basemap.polygonDrawn.connect(lambda xy: self.polygon_drawn(*xy))
        form = QtWidgets.QFormLayout()
        self.opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.valueChanged.connect(lambda v: self.set_opacity(v / 100.0))
        self.lut = QtWidgets.QComboBox()
        self.lut.addItems(colormaps.NAMES)
        self.lut.currentTextChanged.connect(lambda t: t and self.set_style(lut=t))
        self.interval = QtWidgets.QDoubleSpinBox()
        self.interval.setRange(0, 1e6)
        self.interval.setSpecialValueText("auto")
        self.interval.valueChanged.connect(lambda v: self.set_style(interval=(v or None)))
        form.addRow("Opacity", self.opacity)
        form.addRow("Colour ramp", self.lut)
        form.addRow("Contour interval", self.interval)
        lay.addLayout(form)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

    # ------------------------------------------------------------------ model
    def set_stack(self, stack):
        self.stack = stack
        self.refresh()

    def refresh(self):
        st = self.stack
        try:
            self.list.blockSignals(True)
            self.list.clear()
            self._ids = []
            if st is None:
                return
            for l in reversed(st.ordered()):  # top of the list = drawn on top
                it = QtWidgets.QListWidgetItem(f"{l['name']}  [{l['kind']}]")
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.Checked if l["visible"] else QtCore.Qt.Unchecked)
                self.list.addItem(it)
                self._ids.append(l["id"])
        finally:
            self.list.blockSignals(False)

    def selected_id(self):
        return self._sel if self._sel in self._ids else None

    def select(self, lid):
        self._sel = lid
        if lid in self._ids:
            try:
                self.list.setCurrentRow(self._ids.index(lid))
            except Exception:
                pass

    def _apply(self, msg=""):
        if self.stack is None:
            return
        self.stack.save()
        if self.basemap is not None:
            self.basemap.draw_layers()
        self.refresh()
        errs = getattr(self.basemap, "layer_errors", []) or []
        self.status.setText(msg + ("; " if msg and errs else "") + "; ".join(f"{e[1]}: {e[2]}" for e in errs))
        self.changed.emit()

    def _item_changed(self, item):
        r = self.list.row(item)
        if 0 <= r < len(self._ids):
            self.set_visible(self._ids[r], item.checkState() == QtCore.Qt.Checked)

    def _sync_controls(self):
        try:
            r = int(self.list.currentRow())
        except Exception:
            r = -1
        if 0 <= r < len(self._ids):
            self._sel = self._ids[r]
        lid = self.selected_id()
        if not lid or self.stack is None:
            return
        l = self.stack.get(lid)
        for w, fn in ((self.opacity, lambda: self.opacity.setValue(int(round(100 * l["opacity"])))),
                      (self.lut, lambda: self.lut.setCurrentText(l["style"].get("lut", colormaps.NAMES[0]))),
                      (self.interval, lambda: self.interval.setValue(float(l["style"].get("interval") or 0)))):
            w.blockSignals(True)
            try:
                fn()
            finally:
                w.blockSignals(False)

    # ------------------------------------------------------------------ actions (testable)
    def set_visible(self, lid, on):
        self.stack.set_visible(lid, on)
        self._apply()

    def move(self, delta, lid=None):
        lid = lid or self.selected_id()
        if lid:
            self.stack.move(lid, delta)
            self._apply()
            self.select(lid)

    def remove(self, lid=None):
        lid = lid or self.selected_id()
        if lid:
            self.stack.remove(lid)
            self._apply("layer removed")

    def set_opacity(self, a, lid=None):
        lid = lid or self.selected_id()
        if lid:
            self.stack.set_opacity(lid, a)
            self._apply()

    def set_style(self, lid=None, **kw):
        lid = lid or self.selected_id()
        if not lid:
            return
        l = self.stack.get(lid)
        if "interval" in kw and l["kind"] == "grid":  # interval applies to the grid's contour layer
            for c in self.stack.layers:
                if c["kind"] == "contours" and c["src"].get("grid") == lid:
                    self.stack.set_style(c["id"], interval=kw.pop("interval"))
                    break
        if kw:
            self.stack.set_style(lid, **kw)
        self._apply()

    def add_current_grid(self, name=None):
        g = getattr(self.basemap, "_grid", (None, None))[0] if self.basemap is not None else None
        if g is None:
            self.status.setText("No grid on the basemap - compute one in Interpret > Horizon maps first")
            return None
        lid, _ = self.stack.add_grid(g, name or g.label)
        self._apply(f"added grid {name or g.label}")
        self.select(lid)
        return lid

    def add_cps3(self, path, name=None):
        from ..maps import read_cps3
        g = read_cps3(path)
        g.label = g.label or os.path.splitext(os.path.basename(path))[0]
        lid, _ = self.stack.add_grid(g, name or os.path.splitext(os.path.basename(path))[0])
        self._apply(f"added {os.path.basename(path)}")
        return lid

    def add_horizon(self, hname):
        lid = self.stack.add("horizon", f"{hname} (time map)", src={"horizon": hname})
        cid = self.stack.add("contours", f"{hname} contours", src={"grid": lid})
        self._apply(f"added horizon {hname}")
        return lid, cid

    def add_petro(self, prop="SoPhiH", zone=None):
        lid = self.stack.add("petro", f"{prop}" + (f" {zone}" if zone else " (all zones)"),
                             src={"property": prop, "zone": zone})
        vals = ML.petro_values(self.stack.project, prop, zone)
        self._apply(f"{len(vals)} wells with {prop}" if vals else
                    "no petrophysics summary yet - run the Petrophysics window in the Well Correlation Tool")
        return lid

    def add_blend(self, grid_ids, mode="rgb", clip=(2.0, 98.0), gamma=1.0, name=None):
        """Co-blend layer of 2-3 grid / horizon layers (PLAN 23a); drawn on the first grid's geometry."""
        ids = [i for i in grid_ids if i][:3]
        if len(ids) < 2:
            raise ValueError("pick 2 or 3 grid layers")
        names = [self.stack.get(i)["name"] for i in ids]
        lid = self.stack.add("blend", name or f"{mode.upper()} blend: " + " / ".join(names), src={"grids": ids},
                             style={"mode": mode, "clip": tuple(clip), "gamma": gamma})
        self._apply(f"{mode.upper()} co-blend of {', '.join(names)}")
        return lid

    def shade_by(self, lid, structure_lid=None, azimuth=315.0, elevation=35.0, ve=1.0, strength=0.6):
        """Amplitude on structure / bump map (PLAN 23b): colour of grid lid shaded by the illuminated structure grid
        (itself when structure_lid is None)."""
        self.stack.set_style(lid, shade_from=structure_lid, hillshade=True, azimuth=float(azimuth),
                             elevation=float(elevation), ve=float(ve), shade_strength=float(strength))
        self._apply(f"{self.stack.get(lid)['name']} lit from {azimuth:.0f} deg / {elevation:.0f} deg")
        return lid

    def blend_legend(self, lid=None):
        """(rgba triangle, [(x, y, label)]) legend of a blend layer - corners are its grids."""
        from .. import blend as B
        lay = self.stack.get(lid or self.selected_id())
        names = [self.stack.get(i)["name"] for i in lay["src"].get("grids", [])]
        return B.legend_triangle(lay["style"].get("mode", "rgb"), 96, names + [""] * (3 - len(names)))

    def _grid_choices(self):
        return [(l["id"], l["name"]) for l in self.stack.ordered() if l["kind"] in ("grid", "horizon")]

    def _add_blend_dialog(self):
        ch = self._grid_choices()
        if len(ch) < 2:
            self.status.setText("co-blend needs at least 2 grid / horizon layers")
            return
        items = [f"{i}: {n}" for i, n in ch]
        picked = []
        for k, col in enumerate(("red / cyan / hue", "green / magenta / saturation", "blue / yellow / value")):
            opts = (["(none)"] if k == 2 else []) + items
            t, ok = QtWidgets.QInputDialog.getItem(self, "Co-blend", f"Channel {k + 1} ({col})", opts, 0, False)
            if not ok:
                return
            if t != "(none)":
                picked.append(t.split(":")[0])
        mode, ok = QtWidgets.QInputDialog.getItem(self, "Co-blend", "Mode", ["rgb", "cmy", "hsv"], 0, False)
        if ok:
            try:
                self.add_blend(picked, mode)
            except Exception as e:
                self.status.setText(str(e))

    def _shade_dialog(self):
        lid = self.selected_id()
        if not lid or self.stack.get(lid)["kind"] not in ("grid", "horizon"):
            self.status.setText("select the grid to colour (e.g. an amplitude map) first")
            return
        items = ["(itself)"] + [f"{i}: {n}" for i, n in self._grid_choices() if i != lid]
        t, ok = QtWidgets.QInputDialog.getItem(self, "Shade by structure", "Structure grid", items, 0, False)
        if not ok:
            return
        az, ok = QtWidgets.QInputDialog.getDouble(self, "Shade by structure", "Light azimuth (deg from N)", 315, 0, 360, 0)
        if not ok:
            return
        el, ok = QtWidgets.QInputDialog.getDouble(self, "Shade by structure", "Light elevation (deg)", 35, 1, 90, 0)
        if not ok:
            return
        ve, ok = QtWidgets.QInputDialog.getDouble(self, "Shade by structure", "Vertical exaggeration", 2, 0.01, 1000, 2)
        if ok:
            self.shade_by(lid, None if t == "(itself)" else t.split(":")[0], az, el, ve)

    def add_polygon(self, name, xs, ys, layer_name="Polygons"):
        lay = next((l for l in self.stack.layers if l["kind"] == "polygons" and l["name"] == layer_name), None)
        lid = lay["id"] if lay else self.stack.add("polygons", layer_name, src={"polygons": []})
        self.stack.get(lid)["src"]["polygons"].append({"name": name, "x": [float(v) for v in xs], "y": [float(v) for v in ys]})
        self._apply(f"polygon {name}")
        return lid

    def add_text(self, x, y, text, layer_name="Text"):
        lay = next((l for l in self.stack.layers if l["kind"] == "text" and l["name"] == layer_name), None)
        lid = lay["id"] if lay else self.stack.add("text", layer_name, src={"items": []})
        self.stack.get(lid)["src"]["items"].append({"x": float(x), "y": float(y), "text": str(text)})
        self._apply()
        return lid

    def add_pies(self, zone=None):
        lid = self.stack.add("petro", "Mineral pies" + (f" {zone}" if zone else " (all zones)"),
                             src={"property": "minerals", "zone": zone}, style={"mode": "pies"})
        from ..maphub import petro_pies
        n = len(petro_pies(self.stack.project, zone))
        self._apply(f"{n} wells with mineral volumes" if n else
                    "no mineral-solver results yet - run Mineral solver in the Petrophysics window")
        return lid

    def start_polygon(self):
        if self.basemap is not None:
            self.basemap.start_polygon()
            self.status.setText("Click vertices on the basemap; double-click to close the polygon")

    def polygon_drawn(self, xs, ys, name=None):
        """A finished polygon: keep it as a layer, select the wells inside (map rings + cross-plot highlight)."""
        from ..maphub import wells_in_polygon
        n = sum(len(l["src"].get("polygons", [])) for l in self.stack.layers if l["kind"] == "polygons")
        lid = self.add_polygon(name or f"Polygon {n + 1}", xs, ys)
        inside = wells_in_polygon(self.stack.project, xs, ys)
        if self.basemap is not None:
            self.basemap.set_selected_wells(inside)
        if self._xplot is not None:
            self._xplot.redraw()
        self.status.setText(f"{len(inside)} wells inside: {', '.join(inside[:12])}{' ...' if len(inside) > 12 else ''}")
        return lid, inside

    def prop_grid(self):
        from .propgrid_dialog import PropGridDialog
        dlg = PropGridDialog(self.stack, self, self)
        dlg.show()
        return dlg

    def grid_calc(self):
        from .propgrid_dialog import GridCalcDialog
        dlg = GridCalcDialog(self.stack, self, self)
        dlg.show()
        return dlg

    def add_new_wells(self, folder, tie=True, change_layers=True):
        """PLAN 21d: import new wells, petro batch, blind test of the old maps, re-grid, tie. Returns the report."""
        from .. import addwells as AW
        rep = AW.add_wells(self.stack.project, self.stack, folder, tie=tie, change_layers=change_layers)
        self.last_report = AW.report_text(rep)
        if self.basemap is not None:
            self.basemap.redraw()
        self._apply(f"{len(rep['added'])} new wells")
        return rep

    def _add_wells_dialog(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder with the new wells (wells/<Well>/las ...)")
        if not folder:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.add_new_wells(folder)
        except Exception as e:
            self.last_report = f"Add wells failed: {type(e).__name__}: {e}"
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("New wells")
        box.setText(self.last_report.split("\n")[0] if self.last_report else "done")
        box.setDetailedText(self.last_report or "")
        box.exec()

    def cross_check(self):
        from .propgrid_dialog import CrossCheckDialog
        dlg = CrossCheckDialog(self.stack, self, self)
        dlg.show()
        return dlg

    def apply_preset(self, name):
        from ..mapexport import apply_preset
        apply_preset(self.stack, name)
        self._apply(f"style {name}")
        return name

    def export_map(self, path, **kw):
        from ..mapexport import export_map
        out = export_map(self.stack, path, **kw)
        self.status.setText(f"wrote {path}")
        return out

    def _export_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export map", "map.pdf", "PDF (*.pdf);;PNG (*.png)")
        if not path:
            return
        title, ok = QtWidgets.QInputDialog.getText(self, "Export map", "Title")
        if not ok:
            return
        page, ok = QtWidgets.QInputDialog.getItem(self, "Export map", "Page", ["A4 landscape", "A4 portrait", "A3 landscape",
                                                                             "A3 portrait", "A1 landscape"], 0, False)
        if not ok:
            return
        try:
            self.export_map(path, title=title, page=page, crs=(self.stack.project.display.get("crs") or "")
                            if self.stack.project is not None else "")
        except Exception as e:
            self.status.setText(f"export failed: {e}")

    # ------------------------------------------------------------------ GIS (21g)
    def add_vector(self, path, name=None, style=None, table=None):
        lid = self.stack.add("vector", name or os.path.splitext(os.path.basename(path))[0],
                             src={"file": os.path.abspath(path), "table": table}, style=style)
        from ..gis.vector import layer_of
        vl = layer_of(self.stack, self.stack.get(lid))
        msg = f"{len(vl)} features" if vl is not None else "could not read the file"
        if vl is not None and vl.crs is None:
            msg += " (no .prj: assumed in the project CRS)"
        if self.stack.get(lid)["src"].get("approx_datum"):
            msg += " - NAD27 -> NAD83 shift is approximate (no NADCON grids)"
        self._apply(msg)
        return lid

    def set_crs(self, epsg=None, wkt=None):
        from ..gis.export import set_project_crs
        c = set_project_crs(self.stack.project, epsg, wkt)
        from ..gis import vector as V
        V._CACHE.clear()
        self._apply(f"project CRS: {c['name']}")
        return c

    def export_gis(self, folder, epsg=None):
        from ..gis.export import export_stack
        out = export_stack(self.stack, folder, epsg=epsg)
        self.status.setText(f"{len(out['files'])} files + GeoPackage + KML in {folder}" +
                            ("; " + "; ".join(out["warnings"]) if out["warnings"] else ""))
        return out

    def goto_legal(self, text, lid=None):
        """Spot a PLSS legal description in a vector layer (sections) and mark it with a text layer item."""
        from ..gis.plss import spot
        from ..gis.vector import layer_of
        cands = [l for l in self.stack.layers if l["kind"] == "vector"] if lid is None else [self.stack.get(lid)]
        for lay in cands:
            vl = layer_of(self.stack, lay)
            if vl is None:
                continue
            try:
                r = spot(vl, text)
            except ValueError:
                continue
            if r:
                x, y, i = r
                self.add_text(x, y, text.strip(), layer_name="Legal descriptions")
                if self.basemap is not None:
                    try:
                        self.basemap.plot.getPlotItem().getViewBox().setRange(xRange=(x - 3000, x + 3000),
                                                                             yRange=(y - 3000, y + 3000))
                    except Exception:
                        pass
                return x, y
        self.status.setText(f"not found: {text}")
        return None

    def identify(self, x, y, tol):
        """Attribute row of the top-most visible vector feature at (x, y)."""
        from ..gis.vector import layer_of
        for lay in reversed(self.stack.ordered()):
            if lay["kind"] != "vector" or not lay["visible"]:
                continue
            vl = layer_of(self.stack, lay)
            if vl is None:
                continue
            i, row = vl.identify(x, y, tol)
            if i is not None:
                self.status.setText(f"{lay['name']} #{i}: " + ", ".join(f"{k}={v}" for k, v in list(row.items())[:8]))
                return lay["id"], i, row
        return None

    def attribute_rows(self, lid, filter_expr=""):
        from ..gis.vector import layer_of
        vl = layer_of(self.stack, self.stack.get(lid))
        if vl is None:
            return [], []
        idx = __import__("numpy").nonzero(vl.select(filter_expr))[0]
        return vl.fields(), [vl.row(i) for i in idx]

    def _add_vector_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Vector layer", "", "Vector (*.shp *.gpkg *.kml *.csv);;All (*)")
        if path:
            try:
                self.add_vector(path)
            except Exception as e:
                self.status.setText(f"could not add {path}: {e}")

    def _crs_dialog(self):
        cur = (self.stack.project.display.get("crs_epsg") if self.stack.project is not None else None) or 2193
        code, ok = QtWidgets.QInputDialog.getInt(self, "Project coordinate system", "EPSG code (2193 NZTM, 326xx UTM, "
                                                 "32139 / 2278 Texas S Central, 5070 CONUS Albers ...)", int(cur), 1, 999999)
        if ok:
            try:
                self.set_crs(code)
            except Exception as e:
                self.status.setText(str(e))

    def _export_gis_dialog(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the GIS files")
        if folder:
            try:
                self.export_gis(folder)
            except Exception as e:
                self.status.setText(f"export failed: {e}")

    def _legal_dialog(self):
        t, ok = QtWidgets.QInputDialog.getText(self, "Legal description", "e.g. 660 FNL 1980 FWL Sec 14 T12N R3W")
        if ok and t:
            self.goto_legal(t)

    def _attr_dialog(self):
        lid = self.selected_id()
        if not lid or self.stack.get(lid)["kind"] != "vector":
            self.status.setText("select a vector layer first")
            return
        flt, ok = QtWidgets.QInputDialog.getText(self, "Attribute table", "Filter (optional), e.g. STATE == 'TX'")
        if not ok:
            return
        try:
            fields, rows = self.attribute_rows(lid, flt)
        except Exception as e:
            self.status.setText(str(e))
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"{self.stack.get(lid)['name']} - {len(rows)} features")
        lay = QtWidgets.QVBoxLayout(dlg)
        t = QtWidgets.QTableWidget(min(len(rows), 5000), len(fields))
        t.setHorizontalHeaderLabels(fields)
        for i, r in enumerate(rows[:5000]):
            for j, f in enumerate(fields):
                t.setItem(i, j, QtWidgets.QTableWidgetItem(str(r[f])))
        t.setSortingEnabled(True)
        lay.addWidget(t)
        dlg.resize(900, 500)
        dlg.show()

    def crossplot(self):
        from .petro_strip import PetroCrossplotDialog
        if self._xplot is None:
            self._xplot = PetroCrossplotDialog(self.stack.project, self.basemap, self)
        self._xplot.redraw()
        self._xplot.show()
        return self._xplot

    # ------------------------------------------------------------------ dialogs
    def _add_pies_dialog(self):
        zone, ok = QtWidgets.QInputDialog.getText(self, "Mineral pies", "Zone (empty = all zones)")
        if ok:
            self.add_pies(zone.strip() or None)

    def _add_cps3_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "CPS-3 grid", "", "CPS-3 (*.cps *.dat *.txt *.grd);;All (*)")
        if path:
            try:
                self.add_cps3(path)
            except Exception as e:
                self.status.setText(f"could not read {path}: {e}")

    def _add_horizon_dialog(self):
        p = self.stack.project if self.stack else None
        names = [h.name for h in p.horizons.items] if p is not None else []
        if not names:
            self.status.setText("no horizons")
            return
        n, ok = QtWidgets.QInputDialog.getItem(self, "Horizon time map", "Horizon", names, 0, False)
        if ok and n:
            self.add_horizon(n)

    def _add_petro_dialog(self):
        props = ["SoPhiH", "NetPay_TVD", "PHIH_pay", "PHIE_res", "Sw_pay", "Vsh_avg", "NtG_res", "kh_pay", "TOC_avg"]
        prop, ok = QtWidgets.QInputDialog.getItem(self, "Petrophysics on the map", "Value", props, 0, False)
        if not ok:
            return
        zone, ok = QtWidgets.QInputDialog.getText(self, "Petrophysics on the map", "Zone (empty = all zones)")
        if ok:
            self.add_petro(prop, zone.strip() or None)

    def _add_text_dialog(self):
        t, ok = QtWidgets.QInputDialog.getText(self, "Map text", "Text (placed at the map centre)")
        if ok and t:
            v = self.basemap.label_view() if self.basemap is not None else None
            x, y = ((v[0][0] + v[0][1]) / 2, (v[1][0] + v[1][1]) / 2) if v else (0.0, 0.0)
            self.add_text(x, y, t)
