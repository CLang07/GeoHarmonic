"""Cross-section view (pyqtgraph). One canvas for all wells so correlation lines and zoom are shared.

Mouse:  wheel = vertical zoom,  Ctrl+wheel = horizontal zoom,  drag = pan,
        right-click = view menu (View All etc.),  click a well = select it.
"""
import numpy as np
import os
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ..section import SectionLayout

TRACK_LINE = (170, 170, 170)


class SectionViewBox(pg.ViewBox):
    owner = None

    def wheelEvent(self, ev, axis=None):
        o = self.owner
        if o is not None and o._ghost is not None:  # wheel stretches / squeezes the ghost
            o.ghost_stretch(1.03 if ev.delta() > 0 else 1 / 1.03)
            ev.accept()
            return
        mods = ev.modifiers()
        super().wheelEvent(ev, axis=0 if mods & QtCore.Qt.ControlModifier else 1)

    def mouseDragEvent(self, ev, axis=None):
        o = self.owner
        if o is not None and o.pick_mode and ev.button() == QtCore.Qt.LeftButton:
            if ev.isStart():
                if o._start_drag(self.mapToView(ev.buttonDownPos())):
                    ev.accept()
                    return
            elif o._drag is not None:
                ev.accept()
                pt = self.mapToView(ev.pos())
                if ev.isFinish():
                    o._finish_drag(pt)
                else:
                    o._update_drag(pt)
                return
        super().mouseDragEvent(ev, axis)


class SectionView(pg.GraphicsLayoutWidget):
    wellSelected = QtCore.Signal(str)
    cursorInfo = QtCore.Signal(str)
    topEdited = QtCore.Signal(str, str, float)  # well, formation, new MD
    formationSelected = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground("w")
        self.header = self.addPlot(row=0, col=0)
        self.header.hideAxis("bottom")
        self.header.hideAxis("left")
        self.header.setMouseEnabled(False, False)
        self.header.hideButtons()
        self.header.setMenuEnabled(False)
        self.header.setYRange(0, 1, padding=0)
        self.header.setMinimumHeight(95)
        self.header.setMaximumHeight(95)
        self.vb = SectionViewBox()
        self.vb.owner = self
        self.main = self.addPlot(row=1, col=0, viewBox=self.vb)
        self.main.invertY(True)
        self.main.hideAxis("bottom")
        self.main.showGrid(x=False, y=True, alpha=0.25)
        self.main.hideButtons()
        self.header.setXLink(self.main)
        self.main.getAxis("left").setWidth(60)
        self.header.getAxis("left").setWidth(60)
        self.header.showAxis("left")
        self.header.getAxis("left").setStyle(showValues=False)
        self.header.getAxis("left").setPen(pg.mkPen(None))
        self.ci.layout.setRowStretchFactor(1, 10)

        self.project = None
        self.layout_ = None
        self.names = []
        self.max_points = 0
        self.selected = None
        self._highlight = None
        self.pick_mode = False
        self.active_formation = None
        self._marks = []  # (well, formation, y, x0, x1) for hit testing
        self._drag = None
        self._drag_item = None
        self._dyn_items = []
        self._curve_items = []
        self._gap_cache = {}
        self.proposals = {}  # well -> {formation: (md, score or None, source)}
        self._ghost = None
        self._last_pt = None
        self.ghost_half = 30.0  # metres either side of the cursor
        self._ylim = (0.0, 1.0)
        self.lod_points = 3000  # points per curve when zoomed out (min/max kept)
        self.lod_full_below = 600.0  # metres of section in view below which curves are full resolution
        self._lod_full = False
        self._lod_timer = QtCore.QTimer(self)
        self._lod_timer.setSingleShot(True)
        self._lod_timer.timeout.connect(self._apply_lod)
        self.vb.sigRangeChanged.connect(self._range_changed)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.scene().sigMouseMoved.connect(self._mouse_moved)
        self.scene().sigMouseClicked.connect(self._mouse_clicked)

    # ---------------------------------------------------------------- data
    def set_project(self, project):
        self.project = project
        self.names = []
        self.redraw()

    def set_wells(self, names, keep_view=False):
        self.names = list(names)
        self.redraw(keep_view=keep_view)

    def set_selected(self, name):
        self.selected = name
        self._draw_highlight()

    # ---------------------------------------------------------------- draw
    def redraw(self, keep_view=False):
        """Full redraw: logs, fills, headers, then the tops layer."""
        view = self.vb.viewRange() if keep_view else None
        self.main.clear()
        self.header.clear()
        self._highlight = None
        self._dyn_items = []
        self._curve_items = []
        self._ghost = None
        if not self.project or not self.names:
            self.layout_ = None
            t = pg.TextItem("Click wells on the map (or draw a line) to build a cross section", color=(90, 90, 90))
            self.main.addItem(t, ignoreBounds=True)
            t.setPos(0, 0)
            return
        lay = SectionLayout(self.project, self.names)
        self.layout_ = lay
        self.main.setLabel("left", lay.y_label())
        lo, hi = lay.depth_range()
        self._ylim = (lo, hi)
        try:
            for img, x0, x1, y0, y1 in lay.track_lith_images():
                self._add_image(img, x0, x1, y0, y1, -20)
            for p in lay.panels:
                for img, x0, x1, y0, y1 in lay.fill_images(p):
                    self._add_image(img, x0, x1, y0, y1, -15)
        except Exception as e:  # never let a fill problem stop the logs drawing
            self.cursorInfo.emit(f"Fill skipped: {e}")
        for p in lay.panels:
            full = list(lay.curves(p))
            dec = list(lay.curves(p, max_points=self.lod_points))
            for c, d in zip(full, dec):
                item = pg.PlotCurveItem(d["x"], d["y"], pen=pg.mkPen(c["color"], width=c["width"]), connect="finite")
                item.setZValue(5)
                self.main.addItem(item)
                self._curve_items.append((item, c["x"], c["y"], d["x"], d["y"]))
            self._draw_header(lay, p)
        self._lod_full = False
        # track boundaries as one item (fast)
        bx, by = [], []
        for p in lay.panels:
            for tid, tx0, tx1 in p.tracks:
                for xx in (tx0, tx1):
                    bx += [xx, xx]
                    by += [lo, hi]
        self.main.addItem(pg.PlotCurveItem(np.array(bx), np.array(by), pen=pg.mkPen(TRACK_LINE, width=1), connect="pairs"))
        for i, dkm in enumerate(lay.distances):
            if dkm is None:
                continue
            a, b = lay.panels[i], lay.panels[i + 1]
            t = pg.TextItem(f"{dkm:.1f} km", color=(90, 90, 90), anchor=(0.5, 0.5))
            t.setPos((a.x1 + b.x0) / 2, 0.15)
            self.header.addItem(t, ignoreBounds=True)
        self.header.setYRange(0, 1, padding=0)
        self.refresh_tops(new_layout=False)
        if view:
            self.vb.setRange(xRange=view[0], yRange=view[1], padding=0)
        else:
            self.vb.setRange(xRange=(-0.3, lay.total_width + 0.3), yRange=(lo, hi), padding=0.01)
        self._apply_lod()

    def refresh_tops(self, new_layout=True):
        """Redraw only what depends on tops: top lines, labels, between-well fills, highlight."""
        if not self.project or not self.names or self.layout_ is None:
            return
        if new_layout:  # tops changed but not the wells or display: reuse geometry, refresh tops
            self.layout_ = SectionLayout(self.project, self.names)
        for it in self._dyn_items:
            self.main.removeItem(it)
        self._dyn_items = []
        lay = self.layout_
        try:
            for img, x0, x1, y0, y1 in lay.gap_images(self._ylim, cache=self._gap_cache):
                self._dyn_items.append(self._add_image(img, x0, x1, y0, y1, -20))
        except Exception as e:
            self.cursorInfo.emit(f"Between-well fill skipped: {e}")
        n0 = len(self.main.items)
        self._draw_tops(lay)
        self._draw_reference(lay)
        self._draw_proposals(lay)
        self._dyn_items += list(self.main.items[n0:])
        if len(self._gap_cache) > 400:
            self._gap_cache.clear()
        self._draw_highlight()

    def _add_image(self, img, x0, x1, y0, y1, z):
        item = pg.ImageItem(img, levels=(0, 255), autoLevels=False)
        item.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        item.setZValue(z)
        self.main.addItem(item, ignoreBounds=True)
        return item

    # ---------------------------------------------------------------- level of detail
    def _range_changed(self, *_):
        self._lod_timer.start(120)

    def _apply_lod(self):
        if not self._curve_items:
            return
        (x0, x1), (y0, y1) = self.vb.viewRange()
        full = abs(y1 - y0) < self.lod_full_below
        if full == self._lod_full:
            return
        self._lod_full = full
        for item, fx, fy, dx, dy in self._curve_items:
            item.setData(fx, fy, connect="finite") if full else item.setData(dx, dy, connect="finite")

    def _draw_header(self, lay, p):
        w = self.project.wells[p.name]
        bold = QtGui.QFont()
        bold.setBold(True)
        bold.setPointSize(10)
        t = pg.TextItem(p.name, color=(0, 0, 0), anchor=(0.5, 0))
        t.setFont(bold)
        t.setPos((p.x0 + p.x1) / 2, 1.0)
        self.header.addItem(t, ignoreBounds=True)
        t = pg.TextItem(f"{w.elev_ref} {w.elev:.1f} m" + ("" if w.logs or not w.loaded else "  (no logs)"),
                        color=(80, 80, 80), anchor=(0.5, 0))
        t.setPos((p.x0 + p.x1) / 2, 0.75)
        self.header.addItem(t, ignoreBounds=True)
        small = QtGui.QFont()
        small.setPointSize(7)
        for tx0, tx1, title, curves in lay.headers(p):
            for k, (lab, l, r, col) in enumerate(curves):
                y = 0.5 - k * 0.17
                items = [(lab.split(" ")[0], (tx0 + tx1) / 2, (0.5, 0.5))]
                if l != "":
                    items += [(f"{l:g}", tx0, (0, 0.5)), (f"{r:g}", tx1, (1, 0.5))]
                for txt, x, anchor in items:
                    ti = pg.TextItem(txt, color=col, anchor=anchor)
                    ti.setFont(small)
                    ti.setPos(x, y)
                    self.header.addItem(ti, ignoreBounds=True)

    def _draw_tops(self, lay):
        self._marks = []
        for t in lay.top_marks():
            active = t["name"] == self.active_formation
            dpen = pg.mkPen(t["color"], width=2 if active else 1, style=QtCore.Qt.DashLine)
            for (p, y, md), meta in zip(t["marks"], t["metas"]):
                auto = meta.get("source") == "auto"
                low = auto and meta.get("confidence") == "low"
                pen = pg.mkPen(t["color"], width=(4 if active else 2.5),
                               style=QtCore.Qt.DotLine if low else (QtCore.Qt.DashLine if auto else QtCore.Qt.SolidLine))
                self.main.addItem(pg.PlotCurveItem([p.x0, p.x1], [y, y], pen=pen))
                self._marks.append((p.name, t["name"], y, p.x0, p.x1))
                if low:
                    q = pg.TextItem("?", color=t["color"], anchor=(0, 1))
                    q.setPos(p.x0, y)
                    self.main.addItem(q, ignoreBounds=True)
            for (x0, y0), (x1, y1) in t["segments"]:
                self.main.addItem(pg.PlotCurveItem([x0, x1], [y0, y1], pen=dpen))
            if t["marks"]:
                p, y, md = t["marks"][-1]
                lbl = pg.TextItem(t["name"], color=t["color"], anchor=(0, 0.5))
                if active:
                    f = QtGui.QFont()
                    f.setBold(True)
                    lbl.setFont(f)
                lbl.setPos(p.x1 + 0.05, y)
                self.main.addItem(lbl, ignoreBounds=True)

    def _draw_reference(self, lay):
        """Grey ticks for a reference tops set (e.g. published PR3099) on the right of each well."""
        name = self.project.display.get("show_reference")
        ref = self.project.reference_tops.get(name, {}).get("tops", {}) if name else {}
        if not ref:
            return
        pen = pg.mkPen((90, 90, 90), width=2, style=QtCore.Qt.DashLine)
        for p in lay.panels:
            w = self.project.wells[p.name]
            for fm, md in ref.get(p.name, {}).items():
                y = float(lay.y_of(w, [md])[0])
                if not np.isfinite(y):
                    continue
                x0 = p.x1 - 0.35 * (p.x1 - p.x0)
                self.main.addItem(pg.PlotCurveItem([x0, p.x1 + 0.1], [y, y], pen=pen))
                t = pg.TextItem("ref", color=(90, 90, 90), anchor=(0, 0.5))
                t.setPos(p.x1 + 0.1, y)
                self.main.addItem(t, ignoreBounds=True)

    def _draw_proposals(self, lay):
        if not self.proposals:
            return
        colors = {f["name"]: f.get("color", "#cc00cc") for f in self.project.formations}
        for p in lay.panels:
            for fm, (md, score, src) in self.proposals.get(p.name, {}).items():
                w = self.project.wells[p.name]
                y = float(lay.y_of(w, [md])[0])
                if not np.isfinite(y):
                    continue
                pen = pg.mkPen((200, 0, 200), width=3, style=QtCore.Qt.DashDotLine)
                self.main.addItem(pg.PlotCurveItem([p.x0 - 0.15, p.x1 + 0.15], [y, y], pen=pen))
                txt = f"{fm}?" + (f" {score:.2f}" if score is not None else f" ({src})")
                t = pg.TextItem(txt, color=(170, 0, 170), anchor=(1, 1))
                t.setPos(p.x1, y)
                self.main.addItem(t, ignoreBounds=True)

    def set_proposals(self, proposals):
        self.proposals = proposals
        self.refresh_tops(new_layout=False)

    # ---------------------------------------------------------------- ghost curve
    def start_ghost(self):
        """Copy GR around the cursor in the well under the cursor; it then follows the mouse."""
        if self._ghost is not None:
            self.cancel_ghost()
            return False
        if not self.layout_ or self._last_pt is None:
            return False
        x, y = self._last_pt
        p = self.layout_.panel_at(x)
        if not p:
            return False
        w = self.project.wells[p.name]
        gr = w.logs.get("GR")
        if gr is None:
            self.cursorInfo.emit(f"{p.name} has no GR for a ghost")
            return False
        yy = self.layout_.y_full(w)
        sel = np.isfinite(yy) & (np.abs(yy - y) <= self.ghost_half)
        if sel.sum() < 10:
            return False
        frac = np.clip(gr[sel] / 150.0, 0, 1)
        self._ghost = {"src": p.name, "y0": y, "dy": yy[sel] - y, "frac": frac, "stretch": 1.0,
                       "item": pg.PlotCurveItem(pen=pg.mkPen((255, 0, 160, 200), width=2.5), connect="finite"),
                       "box": pg.PlotCurveItem(pen=pg.mkPen((255, 0, 160, 120), width=1, style=QtCore.Qt.DashLine))}
        self.main.addItem(self._ghost["item"])
        self.main.addItem(self._ghost["box"])
        self._ghost["item"].setZValue(50)
        self._ghost_move(x, y)
        self.cursorInfo.emit("Ghost: move over another well, wheel or [ ] to stretch, click to drop, Esc to cancel")
        return True

    def _gr_track(self, p):
        for tid, t0, t1 in p.tracks:
            if tid == "GR":
                return t0, t1
        return p.tracks[0][1], p.tracks[0][2]

    def _ghost_move(self, x, y):
        g = self._ghost
        p = self.layout_.panel_at(x) if self.layout_ else None
        if g is None or not p:
            return
        t0, t1 = self._gr_track(p)
        ys = y + g["dy"] * g["stretch"]
        g["item"].setData(t0 + g["frac"] * (t1 - t0), ys)
        a, b = ys.min(), ys.max()
        g["box"].setData([t0, t1, t1, t0, t0], [a, a, b, b, a])
        g["target"], g["ty"] = p.name, y

    def ghost_stretch(self, factor):
        g = self._ghost
        if g is None:
            return
        g["stretch"] = float(np.clip(g["stretch"] * factor, 0.3, 3.0))
        if self._last_pt:
            self._ghost_move(*self._last_pt)
        self.cursorInfo.emit(f"Ghost stretch {g['stretch']:.2f}x (wheel or [ ] to change, click to drop)")

    def cancel_ghost(self):
        g = self._ghost
        if g is not None:
            self.main.removeItem(g["item"])
            self.main.removeItem(g["box"])
        self._ghost = None

    def _drop_ghost(self):
        """Map the source well's tops inside the ghost window onto the target well as proposals."""
        g = self._ghost
        if g is None or g.get("target") in (None, g["src"]):
            self.cursorInfo.emit("Move the ghost over a different well, then click")
            return
        src, tgt = self.project.wells[g["src"]], self.project.wells[g["target"]]
        found = {}
        for fm, md in self.project.tops.get(g["src"], {}).items():
            ys = float(self.layout_.y_of(src, [md])[0])
            if np.isfinite(ys) and abs(ys - g["y0"]) <= self.ghost_half:
                yt = g["ty"] + (ys - g["y0"]) * g["stretch"]
                mdt = self.layout_.md_at(tgt, yt)
                if mdt is not None:
                    found[fm] = (mdt, None, "ghost")
        self.cancel_ghost()
        if not found:
            self.cursorInfo.emit(f"No {g['src']} tops inside the ghost window - pick by clicking instead")
            return
        props = dict(self.proposals)
        props[g["target"]] = {**props.get(g["target"], {}), **found}
        self.set_proposals(props)
        self.cursorInfo.emit(f"Ghost placed {len(found)} top(s) in {g['target']}: Enter accepts, Esc clears")

    # ---------------------------------------------------------------- picking
    def set_pick_mode(self, on):
        self.pick_mode = bool(on)
        self.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)

    def set_active_formation(self, name):
        if name != self.active_formation:
            self.active_formation = name
            if self.layout_:
                self.refresh_tops(new_layout=False)

    def _hit_top(self, pt):
        p = self.layout_.panel_at(pt.x()) if self.layout_ else None
        if not p:
            return None
        tol = abs(self.vb.viewPixelSize()[1]) * 8
        best = None
        for well, fm, y, x0, x1 in self._marks:
            if well == p.name and abs(y - pt.y()) <= tol and (best is None or abs(y - pt.y()) < best[0]):
                best = (abs(y - pt.y()), well, fm, x0, x1)
        return best

    def _start_drag(self, pt):
        hit = self._hit_top(pt)
        if not hit:
            return False
        _, well, fm, x0, x1 = hit
        self._drag = (well, fm, x0, x1)
        if fm != self.active_formation:
            self.formationSelected.emit(fm)
        self._drag_item = pg.PlotCurveItem([x0, x1], [pt.y(), pt.y()], pen=pg.mkPen((220, 0, 0), width=3))
        self.main.addItem(self._drag_item)
        return True

    def _update_drag(self, pt):
        if self._drag_item is not None:
            well, fm, x0, x1 = self._drag
            self._drag_item.setData([x0, x1], [pt.y(), pt.y()])
            w = self.project.wells[well]
            md = self.layout_.md_at(w, pt.y())
            if md is not None:
                self.cursorInfo.emit(f"Moving {fm} in {well}: MD {md:.1f} m")

    def _finish_drag(self, pt):
        well, fm, x0, x1 = self._drag
        self._drag = None
        if self._drag_item is not None:
            self.main.removeItem(self._drag_item)
            self._drag_item = None
        md = self.layout_.md_at(self.project.wells[well], pt.y())
        if md is not None:
            self.topEdited.emit(well, fm, float(md))

    def _draw_highlight(self):
        if self._highlight is not None:
            self.header.removeItem(self._highlight)
            self._highlight = None
        if not self.layout_ or not self.selected:
            return
        for p in self.layout_.panels:
            if p.name == self.selected:
                r = QtCore.QRectF(p.x0, 0.0, p.width, 1.0)
                item = QtWidgets.QGraphicsRectItem(r)
                item.setPen(pg.mkPen((200, 22, 27), width=2))
                item.setBrush(pg.mkBrush(255, 230, 230, 120))
                item.setZValue(-10)
                self.header.addItem(item)
                self._highlight = item

    # ---------------------------------------------------------------- mouse
    def _mouse_moved(self, pos):
        if not self.layout_ or not self.vb.sceneBoundingRect().contains(pos):
            return
        pt = self.vb.mapSceneToView(pos)
        self._last_pt = (pt.x(), pt.y())
        if self._ghost is not None:
            self._ghost_move(pt.x(), pt.y())
        p = self.layout_.panel_at(pt.x())
        if not p:
            self.cursorInfo.emit(f"{self.layout_.y_label()}: {pt.y():.1f}")
            return
        w = self.project.wells[p.name]
        md = self.layout_.md_at(w, pt.y())
        parts = [p.name]
        if md is not None and w.depth is not None and len(w.depth):
            parts.append(f"MD {md:.1f} m")
            parts.append(f"TVDSS {float(w.md_to_tvdss([md])[0]):.1f} m")
            i = int(np.clip(np.searchsorted(w.depth, md), 0, len(w.depth) - 1))
            for fam in ("GR", "RDEEP", "RHOB", "NPHI", "DT"):
                v = w.logs.get(fam)
                if v is not None and np.isfinite(v[i]):
                    parts.append(f"{fam} {v[i]:.3g}")
        self.cursorInfo.emit("   ".join(parts))

    def _mouse_clicked(self, ev):
        if not self.layout_ or ev.button() != QtCore.Qt.LeftButton:
            return
        pos = ev.scenePos()
        if self._ghost is not None and self.vb.sceneBoundingRect().contains(pos):
            self._drop_ghost()
            return
        if self.pick_mode and self.vb.sceneBoundingRect().contains(pos):
            pt = self.vb.mapSceneToView(pos)
            p = self.layout_.panel_at(pt.x())
            if p:
                self.selected = p.name
                self._draw_highlight()
                self.wellSelected.emit(p.name)
                if ev.modifiers() & QtCore.Qt.ControlModifier:
                    hit = self._hit_top(pt)
                    if hit:
                        self.formationSelected.emit(hit[2])
                    return
                if self.active_formation:
                    md = self.layout_.md_at(self.project.wells[p.name], pt.y())
                    if md is not None:
                        self.topEdited.emit(p.name, self.active_formation, float(md))
                else:
                    self.cursorInfo.emit("Choose a formation in the Tops panel first")
            return
        for box in (self.vb, self.header.getViewBox()):
            if box.sceneBoundingRect().contains(pos):
                x = box.mapSceneToView(pos).x()
                p = self.layout_.panel_at(x)
                if p:
                    self.selected = p.name
                    self._draw_highlight()
                    self.wellSelected.emit(p.name)
                return

    def view_depth_range(self):
        (x0, x1), (y0, y1) = self.vb.viewRange()
        return min(y0, y1), max(y0, y1)
