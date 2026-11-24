"""QGIS side of the GeoHarmonic Link (PLAN 21h).  Uses qgis.PyQt so it runs on the Qt5 and Qt6 builds of QGIS."""
import os

from qgis.PyQt import QtCore, QtWidgets
from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer, QgsPointXY, QgsGeometry, QgsWkbTypes, QgsRectangle

from . import core

try:
    from qgis.gui import QgsRubberBand, QgsVertexMarker
except ImportError:  # pragma: no cover
    QgsRubberBand = QgsVertexMarker = None


class GeoHarmonicLinkPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.client = None
        self.gpkg = None
        self.watcher = None
        self.layers = {}
        self.group = None
        self.actions = []
        self.section_band = None
        self.cursor_marker = None
        self._suppress = False
        self.timer = None
        self.log = []

    # ------------------------------------------------------------------ QGIS plugin API
    def initGui(self):  # noqa: N802
        tb = self.iface.addToolBar("GeoHarmonic Link")
        for text, fn in (("Open GeoHarmonic project", self.open_dialog), ("Connect to GeoHarmonic", self.connect),
                         ("Send selected wells", self.send_selected_wells),
                         ("Send selected line(s) as seismic section", self.send_selected_lines),
                         ("Send selected polygons as prospects", self.send_selected_polygons),
                         ("Open selected well in GeoHarmonic", self.open_selected_well)):
            a = QtWidgets.QAction(text, self.iface.mainWindow())
            a.triggered.connect(lambda *_, f=fn: f())
            tb.addAction(a)
            self.iface.addPluginToMenu("GeoHarmonic Link", a)
            self.actions.append(a)
        self.toolbar = tb
        self.timer = QtCore.QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    def unload(self):
        if self.timer is not None:
            self.timer.stop()
        for a in self.actions:
            self.iface.removePluginMenu("GeoHarmonic Link", a)
        if self.client:
            self.client.close()

    # ------------------------------------------------------------------ project
    def open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.iface.mainWindow(), "GeoHarmonic shared GeoPackage", "",
                                                        "GeoPackage (*_gis.gpkg *.gpkg)")
        if path:
            self.open_project(path)

    def open_project(self, gpkg):
        """Add every layer of the shared GeoPackage (styled from its layer_styles table) + the grid GeoTIFFs, grouped."""
        info = core.read_project(gpkg)
        self.gpkg = gpkg
        self.watcher = core.FileWatcher(gpkg)
        root = QgsProject.instance().layerTreeRoot()
        self.group = root.findGroup("GeoHarmonic") or root.insertGroup(0, "GeoHarmonic")
        self.layers = {}
        for t in info["editable"] + list(reversed(info["vector"])):
            lyr = QgsVectorLayer(f"{gpkg}|layername={t}", t, "ogr")
            QgsProject.instance().addMapLayer(lyr, False)
            self.group.addLayer(lyr)
            self.layers[t] = lyr
            if t == "wells":
                try:
                    lyr.selectionChanged.connect(lambda *_: self.send_selected_wells(auto=True))
                except Exception:
                    pass
        for name, path, ramp, vmin, vmax in info["rasters"]:
            if os.path.exists(path):
                r = QgsRasterLayer(path, name)
                QgsProject.instance().addMapLayer(r, False)
                self.group.addLayer(r)
                self.layers["raster:" + name] = r
        self.log.append(f"opened {gpkg}: {len(info['vector'])} layers, {len(info['rasters'])} grids")
        return info

    def reload(self):
        for lyr in self.layers.values():
            try:
                lyr.dataProvider().reloadData()
                lyr.triggerRepaint()
            except Exception:
                pass
        self.log.append("reloaded")

    # ------------------------------------------------------------------ link
    def connect(self, port=core.DEFAULT_PORT):
        self.client = core.LinkClient(port)
        ok = self.client.connect()
        msg = "connected to GeoHarmonic" if ok else self.client.error
        self.log.append(msg)
        try:
            self.iface.messageBar().pushMessage("GeoHarmonic Link", msg)
        except Exception:
            pass
        return ok

    def tick(self):
        if self.watcher is not None and self.watcher.changed():
            self.reload()
        if self.client is None:
            return []
        done = []
        for m in self.client.poll():
            done.append(self.handle(m))
        return done

    def handle(self, m):
        t = m.get("type")
        if t == "select_wells":
            lyr = self.layers.get("wells")
            if lyr is not None:
                self._suppress = True
                try:
                    names = set(m.get("wells") or [])
                    ids = [f.id() for f in lyr.getFeatures() if f["name"] in names]
                    lyr.selectByIds(ids)
                finally:
                    self._suppress = False
            return f"selected {len(m.get('wells') or [])} wells"
        if t == "refresh":
            self.reload()
            return "refresh"
        if t == "current_section":
            pts = [QgsPointXY(x, y) for x, y in m.get("xy") or []]
            if QgsRubberBand is not None:
                if self.section_band is None:
                    self.section_band = QgsRubberBand(self.iface.mapCanvas(), QgsWkbTypes.LineGeometry)
                    self.section_band.setWidth(3)
                self.section_band.setToGeometry(QgsGeometry.fromPolylineXY(pts), None)
            return f"section line {len(pts)} points"
        if t == "cursor":
            if QgsVertexMarker is not None:
                if self.cursor_marker is None:
                    self.cursor_marker = QgsVertexMarker(self.iface.mapCanvas())
                self.cursor_marker.setCenter(QgsPointXY(m.get("x", 0.0), m.get("y", 0.0)))
            return "cursor"
        if t == "zoom":
            b = m.get("bbox")
            if b:
                self.iface.mapCanvas().setExtent(QgsRectangle(b[0], b[1], b[2], b[3]))
                self.iface.mapCanvas().refresh()
            return "zoom"
        return t

    # ------------------------------------------------------------------ QGIS -> GeoHarmonic
    def _send(self, msg):
        if self.client is None or not self.client.send(msg):
            self.log.append("not connected")
            return False
        return True

    def send_selected_wells(self, auto=False):
        if auto and self._suppress:
            return False
        lyr = self.layers.get("wells")
        if lyr is None:
            return False
        names = [f["name"] for f in lyr.selectedFeatures()]
        return self._send({"type": "select_wells", "wells": names})

    def _selected(self, kind):
        lyr = self.iface.activeLayer()
        if lyr is None:
            return []
        return [f for f in lyr.selectedFeatures() if f.geometry().type() == kind]

    def send_selected_lines(self):
        sent = 0
        for f in self._selected(QgsWkbTypes.LineGeometry):
            g = f.geometry()
            pts = g.asMultiPolyline()[0] if g.isMultipart() else g.asPolyline()
            sent += self._send(core.line_message(str(f.attribute("name") if "name" in f.fields().names() else f.id()),
                                                 [(p.x(), p.y()) for p in pts]))
        return sent

    def send_selected_polygons(self):
        sent = 0
        for f in self._selected(QgsWkbTypes.PolygonGeometry):
            g = f.geometry()
            ring = g.asMultiPolygon()[0][0] if g.isMultipart() else g.asPolygon()[0]
            sent += self._send(core.polygon_message(str(f.attribute("name") if "name" in f.fields().names() else f.id()),
                                                    [(p.x(), p.y()) for p in ring]))
        return sent

    def open_selected_well(self):
        lyr = self.layers.get("wells")
        sel = lyr.selectedFeatures() if lyr is not None else []
        return self._send({"type": "open_well", "well": sel[0]["name"]}) if sel else False
