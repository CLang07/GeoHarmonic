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
                         ("Grid a zone property (IDW / kriging / KED / cokriging)...", lambda *_: self.prop_grid()),
                         ("Grid calculator...", lambda *_: self.grid_calc())):
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
