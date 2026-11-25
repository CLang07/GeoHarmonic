"""3D Viewer window (PLAN 33): Petrel-style 3D view of the seismic project.

The scene (meshes, camera, picking, state) is `scene3d.py` (pure numpy, tested).  This file is the thin Qt layer:
a QOpenGLWidget with one GLSL 1.20 shader (position + colour per vertex, mvp + opacity uniforms) drawing triangle
and line meshes from vertex buffers.  When no OpenGL context can be created (remote desktop, old driver) the window
says so and falls back to the software renderer (scene.render_png) - same scene, slower, still interactive enough
to look around.  PySide6's own QtOpenGL is used: no new dependency (CLAUDE.md rule 4).

Links (map bus, PLAN 31): the active inline / crossline / time slice of the seismic window moves the 3D planes and
vice versa (open_section callback + bus), wells selected on the Live Map are highlighted, new picks / faults / tops
rebuild their objects (DataChanged), the plane positions are published as the active 'section'.
"""
import os
import tempfile

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import mapbus as MB
from .. import scene3d as S3

try:  # PySide6 >= 6.0
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram
    HAVE_GL = True
except Exception:  # pragma: no cover - depends on the installation
    QOpenGLWidget = QtWidgets.QWidget
    HAVE_GL = False

GL_TRIANGLES, GL_LINES, GL_FLOAT = 0x0004, 0x0001, 0x1406
GL_DEPTH_TEST, GL_BLEND = 0x0B71, 0x0BE2
GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA = 0x0302, 0x0303
GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT = 0x4000, 0x0100


def _enum(cls, scoped, name):
    """PySide6 scoped enum with a fallback to the old flat name (6.5 accepts both)."""
    holder = getattr(cls, scoped, None)
    return getattr(holder, name) if holder is not None and hasattr(holder, name) else getattr(cls, name)

VS = """#version 120
attribute vec3 pos;
attribute vec4 col;
uniform mat4 mvp;
uniform float opacity;
varying vec4 vcol;
void main() {
    gl_Position = mvp * vec4(pos, 1.0);
    vcol = vec4(col.rgb, col.a * opacity);
}
"""
FS = """#version 120
varying vec4 vcol;
void main() {
    if (vcol.a < 0.01) discard;
    gl_FragColor = vcol;
}
"""


class GLCanvas(QOpenGLWidget):
    """OpenGL view of a scene3d.Scene.  Mouse: left drag orbit, middle / Shift+left pan, wheel zoom."""

    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.program = None
        try:
            fmt = QtGui.QSurfaceFormat()
            fmt.setDepthBufferSize(24)
            fmt.setSamples(4)
            self.setFormat(fmt)
        except Exception:
            pass
        self.buffers = {}  # id(mesh) -> (buffer, version, count)
        self.gl_error = None
        self._last = None
        self._press = None
        self._buttons = None
        self.on_release = None  # callable(): the window re-reads planes at full resolution after a drag
        self.on_click = None  # callable(px, py, w, h): a click without dragging (3D picking, item 34)
        self.glow_failed = ""  # PLAN 39d: set when the driver cannot draw the glow pass (notice, no crash)
        self.max_line_width = 16.0

    # ---------------------------------------------------------------- GL
    def _funcs(self):
        """The GL function table of the CURRENT context, fetched fresh on every call (PLAN 38a).

        Never cache it: `context().functions()` returns a wrapper Python does not own - Qt can reuse its address
        for the next object (on Chris's PC the shader program), so a cached `self.f` turned into a
        QOpenGLShaderProgram and paintGL died with "no attribute 'glClearColor'".  Returns None without a context."""
        ctx = self.context()
        if ctx is None:
            return None
        f = ctx.functions()
        if f is None or not hasattr(f, "glClearColor"):  # defensive: build a table Python owns for this call
            f = QtGui.QOpenGLFunctions(ctx)
            f.initializeOpenGLFunctions()
        return f

    def initializeGL(self):
        try:
            f = self._funcs()
            if f is None:
                raise RuntimeError("no OpenGL context")
            f.glEnable(GL_DEPTH_TEST)
            f.glEnable(GL_BLEND)
            f.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            self.buffers = {}  # a (re)created context owns no buffers yet
            p = QOpenGLShaderProgram(self)
            p.addShaderFromSourceCode(_enum(QOpenGLShader, "ShaderTypeBit", "Vertex"), VS)
            p.addShaderFromSourceCode(_enum(QOpenGLShader, "ShaderTypeBit", "Fragment"), FS)
            p.bindAttributeLocation("pos", 0)
            p.bindAttributeLocation("col", 1)
            if not p.link():
                raise RuntimeError(p.log())
            self.program = p
        except Exception as e:
            self.gl_error = str(e) or "OpenGL initialisation failed"

    def _buffer(self, mesh):
        key = id(mesh)
        hit = self.buffers.get(key)
        if hit is not None and hit[2] == len(mesh):
            return hit[0]
        data = mesh.interleaved().tobytes()
        b = QOpenGLBuffer(_enum(QOpenGLBuffer, "Type", "VertexBuffer"))
        b.create()
        b.bind()
        b.allocate(data, len(data))
        b.release()
        self.buffers[key] = (b, 0, len(mesh))
        return b

    def resizeGL(self, w, h):
        f = self._funcs()  # QOpenGLWidget sets the viewport itself; this only keeps the table fresh + checks it
        if f is not None and self.program is not None:
            try:
                r = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
                f.glViewport(0, 0, int(max(1, w) * r), int(max(1, h) * r))
            except Exception:
                pass

    def paintGL(self):
        if self.program is None:
            return
        f = self._funcs()
        if f is None:
            return
        bg = self.scene.background
        f.glClearColor(bg[0], bg[1], bg[2], 1.0)
        f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        w, h = max(1, self.width()), max(1, self.height())
        mvp = self.scene.camera.mvp(w / h)
        p = self.program
        p.bind()
        p.setUniformValue("mvp", QtGui.QMatrix4x4(*[float(v) for v in mvp.ravel()]))
        live = set()
        # pass 1 (PLAN 39d): glow / halo underlays - wide lines + transparent shells, depth test on, depth write off
        glow = []
        if not self.glow_failed:
            try:
                glow = self.scene.glow_list()
            except Exception as e:
                self.glow_failed, glow = str(e) or "glow failed", []
        if glow:
            try:
                f.glDepthMask(False)
                for mesh, op in glow:
                    b = self._buffer(mesh)
                    live.add(id(mesh))
                    b.bind()
                    p.enableAttributeArray(0)
                    p.setAttributeBuffer(0, GL_FLOAT, 0, 3, 28)
                    p.enableAttributeArray(1)
                    p.setAttributeBuffer(1, GL_FLOAT, 12, 4, 28)
                    p.setUniformValue1f("opacity", float(op))
                    if mesh.kind == "lines":
                        try:  # wide lines are optional in core profiles / some drivers: 1 px glow then
                            f.glLineWidth(float(max(1.0, min(mesh.meta.get("width_px", 1.0), self.max_line_width))))
                        except Exception:
                            self.max_line_width = 1.0
                    f.glDrawArrays(GL_TRIANGLES if mesh.kind == "triangles" else GL_LINES, 0, len(mesh))
                    b.release()
                f.glLineWidth(1.0)
            except Exception as e:  # unsupported GPU / driver: no glow + one-line notice, never crash
                self.glow_failed = str(e) or "glow not supported by this OpenGL driver"
            finally:
                try:
                    f.glDepthMask(True)
                except Exception:
                    pass
        # pass 2: the objects themselves
        for mesh, op in self.scene.draw_list():
            b = self._buffer(mesh)
            live.add(id(mesh))
            b.bind()
            p.enableAttributeArray(0)
            p.setAttributeBuffer(0, GL_FLOAT, 0, 3, 28)
            p.enableAttributeArray(1)
            p.setAttributeBuffer(1, GL_FLOAT, 12, 4, 28)
            p.setUniformValue1f("opacity", float(op))
            f.glDrawArrays(GL_TRIANGLES if mesh.kind == "triangles" else GL_LINES, 0, len(mesh))
            b.release()
        p.release()
        for k in [k for k in self.buffers if k not in live]:  # meshes that were replaced
            try:
                self.buffers.pop(k)[0].destroy()
            except Exception:
                pass

    # ---------------------------------------------------------------- mouse
    def mousePressEvent(self, ev):
        self._last = (ev.position().x(), ev.position().y())
        self._press = self._last
        self._buttons = ev.buttons()

    def mouseMoveEvent(self, ev):
        if self._last is None:
            return
        x, y = ev.position().x(), ev.position().y()
        dx, dy = x - self._last[0], y - self._last[1]
        self._last = (x, y)
        pan = bool(ev.buttons() & QtCore.Qt.MiddleButton) or bool(ev.modifiers() & QtCore.Qt.ShiftModifier)
        if pan:
            self.scene.camera.pan(dx, dy, self.height())
        else:
            self.scene.camera.orbit(dx, dy)
        self.update()

    def mouseReleaseEvent(self, ev):
        self._last = None
        p0 = self._press
        self._press = None
        x, y = ev.position().x(), ev.position().y()
        if p0 is not None and abs(x - p0[0]) + abs(y - p0[1]) < 3 and self.on_click is not None:
            self.on_click(x, y, self.width(), self.height())
            return
        if self.on_release is not None:
            self.on_release()

    def wheelEvent(self, ev):
        self.scene.camera.zoom(1.15 if ev.angleDelta().y() > 0 else 1 / 1.15)
        self.update()


class SoftwareCanvas(QtWidgets.QLabel):
    """Fallback without OpenGL: the scene rendered by matplotlib into a pixmap (same mouse controls)."""

    def __init__(self, scene, parent=None, reason=""):
        super().__init__(parent)
        self.scene = scene
        self.reason = reason
        self._last = None
        self.on_release = None
        self.on_click = None
        self.setMinimumSize(400, 300)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self._png = os.path.join(tempfile.gettempdir(), f"geoharmonic_3d_{os.getpid()}.png")

    def update(self):
        try:
            w, h = max(320, self.width()), max(240, self.height())
        except Exception:
            w, h = 640, 480
        if not isinstance(w, int):
            w, h = 640, 480
        self.scene.render_png(self._png, (w, h), max_tris=30000,
                              title=f"3D (software rendering{': ' + self.reason if self.reason else ''})")
        try:
            self.setPixmap(QtGui.QPixmap(self._png))
        except Exception:
            pass
        return self._png

    def mousePressEvent(self, ev):
        self._last = (ev.position().x(), ev.position().y())

    def mouseReleaseEvent(self, ev):
        if self._last is None:
            return
        x, y = ev.position().x(), ev.position().y()
        if abs(x - self._last[0]) + abs(y - self._last[1]) < 3 and self.on_click is not None:
            self._last = None
            self.on_click(x, y, self.width(), self.height())
            return
        self.scene.camera.orbit(x - self._last[0], y - self._last[1])
        self._last = None
        self.update()
        if self.on_release is not None:
            self.on_release()

    def wheelEvent(self, ev):
        self.scene.camera.zoom(1.15 if ev.angleDelta().y() > 0 else 1 / 1.15)
        self.update()


class Viewer3D(QtWidgets.QMainWindow):
    """project = SeisProject; bus = the seismic window's MapBus; open_section(kind, name, mode, number) moves the
    seismic window; stack = the map hub LayerStack (culture layers)."""

    def __init__(self, project, vol_name=None, bus=None, parent=None, open_section=None, stack=None,
                 force_software=False):
        super().__init__(parent)
        self.setWindowTitle("3D Viewer - GeoHarmonic")
        self.setObjectName("Viewer3D")
        self.project = project
        self.vol_name = vol_name or (project.volumes[0]["name"] if project.volumes else None)
        self.vol = project.volume(self.vol_name) if self.vol_name else None
        self.bus = bus or MB.MapBus()
        self.open_section = open_section
        self.stack = stack
        self.scene, self._saved_objects = S3.Scene.load_state(project)
        self.lut_name, self.clip = "Seismic (red-white-blue)", None
        self.planes = {}  # mode -> number
        self.highlight = []
        self.messages = []
        self.seeds3d = []  # (i, j) of horizon picks made in 3D (34b tracking seeds)
        self.stick = []  # fault-stick points being picked in 3D
        self.bodies, self.body_labels = [], None
        self.labels = []
        self._fitted = False
        self.software = force_software or not HAVE_GL
        reason = "" if not self.software else ("forced" if force_software else "PySide6 QtOpenGL not available")
        self.canvas = SoftwareCanvas(self.scene, self, reason) if self.software else GLCanvas(self.scene, self)
        self.canvas.on_release = lambda: self.full_resolution()
        self.canvas.on_click = lambda x, y, w, h: self.click(x, y, w, h)
        self.pick_mode = None  # None | 'horizon' | 'fault'
        self.pick_target = ""  # horizon / fault name
        self.setCentralWidget(self.canvas)
        self._build_tree()
        self._build_toolbar()
        self.bus.subscribe("view3d", self._on_event)
        # Glow / Halo (PLAN 39d-f): same settings as the seismic window, emphasis from the map bus selection
        from .. import glow as GL
        self.glow_state = GL.GlowState(self.bus, GL.GlowSettings.load(project), name="view3d-glow")
        self.glow_state.listeners.append(lambda: self.redraw())
        self.scene.glow = self.glow_state.settings
        self.scene.glow_emphasis = self._glow_emphasis
        self.resize(1300, 850)
        if not self.software:  # an OpenGL context that fails to initialise -> software fallback with a message
            QtCore.QTimer.singleShot(800, lambda: self._check_gl())
        if self.vol is not None:
            self.build_default()

    def _check_gl(self):
        err = getattr(self.canvas, "gl_error", None)
        if not isinstance(err, str) or not err:
            return False
        self.software = True
        self.canvas = SoftwareCanvas(self.scene, self, "OpenGL failed")
        self.canvas.on_release = lambda: self.full_resolution()
        self.canvas.on_click = lambda x, y, w, h: self.click(x, y, w, h)
        self.setCentralWidget(self.canvas)
        self._say(f"OpenGL could not start ({err}) - showing the software rendering instead")
        self.redraw()
        return True

    # ---------------------------------------------------------------- UI
    def _build_tree(self):
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Object", "Opacity"])
        self.tree.itemChanged.connect(lambda it, col: self._tree_changed(it))
        d = QtWidgets.QDockWidget("Objects", self)
        d.setObjectName("Objects")
        d.setWidget(self.tree)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, d)

    def _build_toolbar(self):
        tb = self.addToolBar("3D")
        tb.setObjectName("3D tools")
        self.spins = {}
        for mode in ("Inline", "Crossline", "Time slice"):
            tb.addWidget(QtWidgets.QLabel(f" {mode} "))
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setMaximumWidth(160)
            s.valueChanged.connect(lambda v, m=mode: self.set_plane(m, v, dragging=True))
            s.sliderReleased.connect(lambda *_: self.full_resolution())
            tb.addWidget(s)
            self.spins[mode] = s
        tb.addWidget(QtWidgets.QLabel(" VE "))
        self.ve_spin = QtWidgets.QDoubleSpinBox()
        self.ve_spin.setRange(0.1, 50.0)
        self.ve_spin.setValue(self.scene.ve)
        self.ve_spin.valueChanged.connect(lambda v: self.set_ve(v))
        tb.addWidget(self.ve_spin)
        for text, fn in (("Pick horizon...", lambda *_: self._pick_mode_dialog("horizon")),
                         ("Pick fault...", lambda *_: self._pick_mode_dialog("fault")),
                         ("End stick", lambda *_: self.end_stick(self.pick_target) if self.pick_mode == "fault" else None),
                         ("Stop picking", lambda *_: self.set_pick_mode(None)),
                         ("Track", lambda *_: self.track(self.pick_target, threaded=True,
                                                         run_job=getattr(self.parent(), "run_job", None))
                          if self.pick_mode == "horizon" and self.pick_target else None),
                         ("Volume...", lambda *_: self._volume_dialog()),
                         ("Volume render", lambda *_: self.volume_render()),
                         ("Geobodies...", lambda *_: self._geobody_dialog()),
                         ("Fit", lambda *_: self.fit()), ("Save view", lambda *_: self._save_view_dialog()),
                         ("Screenshot...", lambda *_: self._screenshot_dialog()),
                         ("Turntable GIF...", lambda *_: self._turntable_dialog())):
            a = QtGui.QAction(text, self)
            a.triggered.connect(fn)
            tb.addAction(a)
        self.act_glow = QtGui.QAction("Glow (G)", self)  # PLAN 39c: G toggles glow globally
        self.act_glow.setShortcut(QtGui.QKeySequence("G"))
        self.act_glow.triggered.connect(lambda *_: self.toggle_glow())
        tb.addAction(self.act_glow)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(lambda pos: self._tree_menu(pos))

    def _tree_menu(self, pos):
        it = self.tree.itemAt(pos)
        oid = it.data(0, QtCore.Qt.UserRole) if it is not None else None
        o = self.scene.get(oid) if isinstance(oid, str) else None
        if o is None:
            return
        from .glow import add_glow_menu
        from .. import glow as GL
        m = QtWidgets.QMenu(self)
        add_glow_menu(m, self.glow_state.settings, GL.KIND_3D.get(o.kind, o.kind), o.name,
                      lambda: (self.glow_state.broadcast(), self.redraw()))
        m.exec(self.tree.viewport().mapToGlobal(pos))

    def _refresh_tree(self):
        self.tree.blockSignals(True)
        self.tree.clear()
        for o in self.scene.objects:
            it = QtWidgets.QTreeWidgetItem([o.name, f"{o.opacity:.2f}"])
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEditable)
            it.setCheckState(0, QtCore.Qt.Checked if o.visible else QtCore.Qt.Unchecked)
            it.setData(0, QtCore.Qt.UserRole, o.id)
            self.tree.addTopLevelItem(it)
        self.tree.blockSignals(False)

    def _tree_changed(self, it):
        oid = it.data(0, QtCore.Qt.UserRole)
        if not isinstance(oid, str):
            return
        self.scene.set_visible(oid, it.checkState(0) == QtCore.Qt.Checked)
        try:
            self.scene.set_opacity(oid, float(it.text(1)))
        except (TypeError, ValueError):
            pass
        self.redraw()

    def redraw(self):
        try:
            self.canvas.update()
        except Exception:
            pass

    def _say(self, msg):
        self.messages.append(msg)
        del self.messages[:-50]
        try:
            self.statusBar().showMessage(msg, 6000)
        except Exception:
            pass

    # ---------------------------------------------------------------- content
    def _lut(self, name=None):
        from .colormaps import lut
        return lut(name or self.lut_name)

    def _clip(self):
        if self.clip is None:
            vol = self.vol
            sample = np.asarray(vol.inline(vol.inlines[vol.nil // 2]), float)
            c = float(np.percentile(np.abs(sample), 99)) or 1.0
            self.clip = (-c, c)
        return self.clip

    def build_default(self):
        """Survey box, three planes in the middle of the volume, visible horizons, faults, wells."""
        vol = self.vol
        box, labels = S3.box_mesh(vol, self.scene.zscale, self.scene.ve)
        self.labels = labels
        self.scene.replace("box", "Survey box", [box])
        for mode, rng in (("Inline", vol.inlines), ("Crossline", vol.xlines), ("Time slice", vol.times)):
            s = self.spins[mode]
            s.blockSignals(True)
            s.setRange(int(rng[0]), int(rng[-1]))
            s.blockSignals(False)
        self.set_plane("Inline", int(vol.inlines[vol.nil // 2]))
        self.set_plane("Crossline", int(vol.xlines[vol.nxl // 2]))
        self.set_plane("Time slice", float(vol.times[vol.ns // 2]))
        self.add_horizons()
        self.add_faults()
        self.add_wells()
        self.add_lines()
        self.add_culture()
        self._apply_saved()
        if not self._fitted:
            self.scene.fit()
            self._fitted = True
        self._refresh_tree()
        self.redraw()

    def _apply_saved(self):
        for d in self._saved_objects:
            o = self.scene.find(d.get("kind"), d.get("name"))
            if o is not None:
                o.visible, o.opacity = bool(d.get("visible", True)), float(d.get("opacity", 1.0))

    def set_plane(self, mode, number, dragging=False, publish=True):
        """Move a slice plane (decimated while dragging, full resolution on release)."""
        vol = self.vol
        if mode == "Inline":
            number = int(np.clip(number, min(vol.inlines[0], vol.inlines[-1]), max(vol.inlines[0], vol.inlines[-1])))
        elif mode == "Crossline":
            number = int(np.clip(number, min(vol.xlines[0], vol.xlines[-1]), max(vol.xlines[0], vol.xlines[-1])))
        else:
            k = int(np.argmin(np.abs(np.asarray(vol.times) - float(number))))
            number = float(vol.times[k])
        self.planes[mode] = number
        dec = 4 if dragging else 1
        m = S3.slice_mesh(vol, mode, number, self._lut(), self._clip(), dec, self.scene.zscale, self.scene.ve)
        self.scene.replace("slice", mode, [m], colorbar=(self.lut_name, self._clip()[0], self._clip()[1], ""),
                           src={"mode": mode, "number": number})
        if publish:
            k = {"Inline": "inline", "Crossline": "crossline", "Time slice": "timeslice"}[mode]
            self.bus.active.set(k, number, source="view3d", volume=self.vol_name)
            if not dragging and self.open_section is not None and mode != "Time slice":
                try:
                    self.open_section("3D", self.vol_name, mode, number)
                except Exception:
                    pass
        self.redraw()
        return m

    def full_resolution(self):
        for mode, number in list(self.planes.items()):
            o = self.scene.find("slice", mode)
            if o is not None and o.meshes and o.meshes[0].meta.get("decimate", 1) > 1:
                self.set_plane(mode, number, dragging=False)

    def set_ve(self, ve):
        self.scene.ve = float(ve)
        self.rebuild()
        return self.scene.ve

    def rebuild(self):
        for mode, number in list(self.planes.items()):
            self.set_plane(mode, number, publish=False)
        box, self.labels = S3.box_mesh(self.vol, self.scene.zscale, self.scene.ve)
        self.scene.replace("box", "Survey box", [box])
        self.add_horizons()
        self.add_faults()
        self.add_wells()
        self.add_lines()
        self.add_culture()
        self.redraw()

    def add_horizons(self):
        vol = self.vol
        I, J = np.meshgrid(vol.inlines, vol.xlines, indexing="ij")
        X, Y = vol.xy(I, J)
        dec = max(1, int(np.ceil(np.sqrt(vol.nil * vol.nxl / 250_000.0))))
        n = 0
        for h in self.project.horizons.items:
            g = h.data.get(f"3D:{self.vol_name}")
            if g is None or not np.isfinite(g).any():
                continue
            ok = np.isfinite(g)
            fin = np.asarray(g)[ok]
            m = S3.surface_mesh(X, Y, g, self._lut("Rainbow (attributes)"), (float(fin.min()), float(fin.max())),
                                max_jump=25.0, zscale=self.scene.zscale, ve=self.scene.ve, name=h.name, decimate=dec)
            self.scene.replace("horizon", h.name, [m], colorbar=("Rainbow (attributes)", float(fin.min()),
                                                                 float(fin.max()), "ms"), visible=h.visible)
            n += 1
        return n

    def add_faults(self):
        n = 0
        for f in self.project.faults.items:
            tris = f.triangles()
            if not len(tris):
                continue
            c = f.color if isinstance(f.color, (tuple, list)) else (200, 0, 0)
            self.scene.replace("fault", f.name, [S3.fault_mesh(tris, c, 0.75, self.scene.zscale, self.scene.ve,
                                                               f.name)], visible=f.visible)
            n += 1
        return n

    def add_wells(self, radius=None):
        """Well paths (deviated from surveys) as tubes; tops as discs; selected wells highlighted."""
        try:
            ws = self.project.wellset()
        except Exception:
            return 0
        from ..laterals import trajectories_from_wellset
        try:
            trajs = trajectories_from_wellset(ws)
        except Exception:
            trajs = []
        xs, ys = self.vol.outline()
        r = radius or 0.006 * float(max(np.ptp(xs), np.ptp(ys)))
        sel = {MB.norm_id(s) for s in self.highlight}
        have = {t.name for t in trajs}
        t_end = float(self.vol.times[-1])
        extra = [(n, x, y) for n, x, y in self.project.wells() if n not in have]
        n = 0
        for t in list(trajs) + extra:
            if isinstance(t, tuple):  # no survey and no TD: vertical through the whole volume
                name = t[0]
                zz = S3.world_z(np.array([float(self.vol.times[0]), t_end]), self.scene.zscale, self.scene.ve)
                path = np.array([[t[1], t[2], zz[0]], [t[1], t[2], zz[1]]], float)

                class _T:
                    pass
                t = _T()
                t.name = name
            else:
                try:
                    td = ws.td_for(t.name)
                except Exception:
                    td = None
                path = S3.well_path_world(t, td, self.scene.zscale, self.scene.ve)
            ok = np.isfinite(path).all(1)
            if ok.sum() < 2:
                continue
            col = np.array([1.0, 0.85, 0.0, 1.0] if MB.norm_id(t.name) in sel else [0.85, 0.85, 0.85, 1.0], np.float32)
            meshes = [S3.tube_mesh(path[ok], r * (1.6 if MB.norm_id(t.name) in sel else 1.0), col, 8, t.name)]
            try:
                tops = ws.tops_twt(t.name)
            except Exception:
                tops = {}
            for fm, twt in tops.items():
                k = int(np.argmin(np.abs(path[ok][:, 2] - S3.world_z(twt, self.scene.zscale, self.scene.ve))))
                c = np.asarray(ws.formation_color(fm), float) / 255.0
                meshes.append(S3.disc_mesh(path[ok][k], (0, 0, 1), 2.5 * r, list(c) + [1.0], 16, f"{t.name} {fm}"))
            self.scene.replace("well", t.name, meshes)
            n += 1
        return n

    def add_lines(self, decimate=2):
        n = 0
        for l in self.project.lines:
            try:
                ln = self.project.line(l["name"])
            except Exception:
                continue
            if not ln.has_coords:
                continue
            m = S3.curtain_mesh(ln.x, ln.y, ln.times, ln.data(), self._lut(), self._clip(), decimate,
                                self.scene.zscale, self.scene.ve, l["name"])
            self.scene.replace("line2d", l["name"], [m], visible=False)
            n += 1
        return n

    def add_culture(self, level_ms=None):
        """Visible map-hub vector / polygon layers draped at the top of the survey (or a given level)."""
        if self.stack is None:
            return 0
        lvl = float(self.vol.times[0]) if level_ms is None else float(level_ms)
        n = 0
        for lay in self.stack.layers:
            if not lay["visible"] or lay["kind"] not in ("vector", "polygons"):
                continue
            meshes = []
            if lay["kind"] == "polygons":
                for p in lay["src"].get("polygons", []):
                    x, y = list(map(float, p["x"])), list(map(float, p["y"]))
                    if not p.get("open"):
                        x, y = x + x[:1], y + y[:1]
                    meshes.append(S3.polyline_mesh(x, y, S3.drape(x, y, None, lvl, self.scene.zscale, self.scene.ve),
                                                   (1.0, 0.3, 0.3, 1.0)))
            else:
                try:
                    from ..gis.vector import layer_of
                    vl = layer_of(self.stack, lay)
                except Exception:
                    vl = None
                for g in (vl.geoms if vl is not None else [])[:5000]:
                    for part in (g or {}).get("parts", []):
                        if len(part) >= 2:
                            meshes.append(S3.polyline_mesh(part[:, 0], part[:, 1],
                                                           S3.drape(part[:, 0], part[:, 1], None, lvl,
                                                                    self.scene.zscale, self.scene.ve),
                                                           (0.6, 0.8, 1.0, 1.0)))
            if meshes:
                self.scene.replace("culture", lay["name"], meshes)
                n += 1
        return n

    # ---------------------------------------------------------------- 3D interpretation (PLAN 34)
    def set_volume(self, vol_name):
        """34e: show another volume on the planes (inversion, attribute, RGT ... - any project volume)."""
        self.vol_name = vol_name
        self.vol = self.project.volume(vol_name)
        self.clip = None
        meta = getattr(self.vol, "meta", {}) or {}
        self.lut_name = "Rainbow (attributes)" if (meta.get("attribute") or meta.get("inversion")) else \
            "Seismic (red-white-blue)"
        for mode, number in list(self.planes.items()):
            self.set_plane(mode, number, publish=False)
        return self.vol_name

    def set_pick_mode(self, mode, target=""):
        """3D picking on / off: clicks (no drag) on a plane add horizon points or fault-stick points."""
        self.pick_mode = mode if mode in ("horizon", "fault") else None
        self.pick_target = str(target or "")
        self.stick = []
        self._say(f"3D picking: {self.pick_mode} {self.pick_target}" if self.pick_mode else "3D picking off")
        return self.pick_mode

    def click(self, px, py, w, h):
        if self.pick_mode == "horizon" and self.pick_target:
            return self.pick_horizon(self.pick_target, px, py, w, h)
        if self.pick_mode == "fault":
            return self.pick_fault_point(px, py, w, h)
        hit = self.pick(px, py, w, h)
        if hit is not None:
            self._say(f"{hit[0]}: IL {hit[1]:.0f}  XL {hit[2]:.0f}  {hit[3]:.0f} ms")
        return hit

    def _pick_mode_dialog(self, mode):
        items = self.project.horizons.names() if mode == "horizon" else self.project.faults.names()
        name, ok = QtWidgets.QInputDialog.getItem(self, f"Pick {mode} in 3D", "Name (or type a new one):",
                                                  list(items) or [f"3D {mode}"], 0, True)
        if ok and name:
            self.set_pick_mode(mode, name)

    def _volume_dialog(self):
        names = [v["name"] for v in self.project.volumes]
        name, ok = QtWidgets.QInputDialog.getItem(self, "Volume on the planes", "Volume:", names, 0, False)
        if ok and name:
            self.set_volume(name)

    def _geobody_dialog(self):
        v, ok = QtWidgets.QInputDialog.getDouble(self, "Geobodies", "Keep values above:", 0.5, -1e9, 1e9, 3)
        if ok:
            b = self.geobodies(lo=v)
            self._say(f"{len(b)} geobodies")

    def pick(self, px, py, w=None, h=None):
        """Pixel -> nearest hit on the visible planes: (mode, il, xl, t) or None (34a ray-plane picking)."""
        w = w or max(1, int(self.canvas.width() or 0)) or 800
        h = h or max(1, int(self.canvas.height() or 0)) or 600
        best = None
        o, d = self.scene.camera.ray(px, py, w, h)
        for mode, number in self.planes.items():
            ob = self.scene.find("slice", mode)
            if ob is None or not ob.visible:
                continue
            hit = S3.pick_on_plane(self.scene.camera, px, py, w, h, self.vol, mode, number, self.scene.zscale,
                                   self.scene.ve)
            if hit is None:
                continue
            x, y = self.vol.xy(hit[0], hit[1])
            p = np.array([float(x), float(y), S3.world_z(hit[2], self.scene.zscale, self.scene.ve)])
            dist = float(np.dot(p - o, d))
            if best is None or dist < best[0]:
                best = (dist, (mode,) + tuple(hit))
        return None if best is None else best[1]

    def pick_horizon(self, name, px, py, w=None, h=None, mode="Peak"):
        """34a: a horizon pick on a 3D plane, snapped like a 2D pick, into the same horizon object (undo shared)."""
        hit = self.pick(px, py, w, h)
        if hit is None:
            return None
        from ..horizons import Picker
        hz = self.project.horizons.get(name) or self.project.horizons.new(name)
        key = f"3D:{self.vol_name}"
        i = int(round((hit[1] - self.vol.il0) / self.vol.il_step))
        j = int(round((hit[2] - self.vol.xl0) / self.vol.xl_step))
        t = Picker(self.project, self.project.horizons).pick_points(hz, key, [i], [j], [hit[3]], mode)
        self.seeds3d = self.seeds3d + [(i, j)]
        self.add_horizons()
        self.bus.notify("picks", ids=[name], source="view3d")
        self.redraw()
        return (i, j, float(t[0]))

    def pick_fault_point(self, px, py, w=None, h=None):
        """34a: a fault-stick point on a 3D plane; end_stick() stores the stick in the current fault."""
        hit = self.pick(px, py, w, h)
        if hit is None:
            return None
        x, y = self.vol.xy(hit[1], hit[2])
        self.stick = self.stick + [(float(x), float(y), float(hit[3]))]
        return self.stick[-1]

    def end_stick(self, fault_name):
        pts = list(self.stick)
        self.stick = []
        if len(pts) < 2:
            return None
        fs = self.project.faults
        f = fs.get(fault_name) or fs.new(fault_name)
        fs.snapshot(f)
        f.add_stick(np.array(sorted(pts, key=lambda p: p[2])))
        self.add_faults()
        self.bus.notify("faults", ids=[fault_name], source="view3d")
        self.redraw()
        return f

    def track(self, name, params=None, threaded=False, run_job=None):
        """34b: autotrack the horizon from the seeds picked in 3D (item-1 tracker); the surface updates when done."""
        from ..horizons import Picker
        hz = self.project.horizons.get(name)
        if hz is None:
            raise KeyError(name)
        key = f"3D:{self.vol_name}"
        seeds = list(self.seeds3d) or None
        result = {}

        def fn(progress=None, cancel=None):
            result["n"] = Picker(self.project, self.project.horizons).track(hz, key, seeds, params, progress=progress)
            return result["n"]

        def done():
            self.glow_state.set_tracking(name, False)
            self.add_horizons()
            self.bus.notify("picks", ids=[name], source="view3d")
            self.redraw()
        self.glow_state.set_tracking(name, True)  # PLAN 39e: the horizon being tracked glows while the worker runs
        if threaded and run_job is not None:
            run_job(f"Track {name}", fn, done)
            return None
        fn()
        done()
        return result.get("n")

    def volume_render(self, box=None, curve=None, n_slabs=48, max_voxels=32e6, renderer=""):
        """34c: opacity-curve rendering of a (user-boxed or decimated) sub-volume as stacked slabs."""
        from .. import geobody as GB
        vol = self.vol
        if box is None:
            box = ((0, vol.nil), (0, vol.nxl), (0, vol.ns))
        (a, b), (c, d_), (e, f) = box
        step = 1
        while not GB.fits_budget(((b - a) // step + 1, (d_ - c) // step + 1, (f - e) // step + 1), max_voxels):
            step += 1
        sub = np.asarray(vol.ilxl[a:b:step], np.float32)[:, c:d_:step, e:f:step] * vol.inv
        clip = self._clip()
        curve = curve or [(clip[0], 0.6), (0.3 * clip[0], 0.0), (0.3 * clip[1], 0.0), (clip[1], 0.6)]
        meshes = GB.render_slabs(sub, (a, c, e), vol, self._lut(), clip, curve, n_slabs, 2, self.scene.zscale,
                                 self.scene.ve, 1, step)
        warn = GB.integrated_graphics(renderer)
        self.scene.replace("volume", "Volume rendering", meshes, opacity=1.0)
        if warn:
            self._say("Integrated graphics: volume rendering can be slow - use a smaller box")
        self._refresh_tree()
        self.redraw()
        return {"slabs": len(meshes), "step": step, "warning": warn}

    def geobodies(self, lo=None, hi=None, min_cells=50, max_bodies=20, push_polygons=True):
        """34d: threshold -> connected bodies -> 3D surfaces + body list + footprints on the Live Map."""
        from .. import geobody as GB
        lab, bodies = GB.label_volume(self.vol, lo, hi, min_cells=min_cells)
        self.bodies = bodies
        self.body_labels = lab
        for o in [o for o in self.scene.objects if o.kind == "geobody"]:
            self.scene.remove(o.id)
        items = []
        for b in bodies[:max_bodies]:
            m = GB.body_mesh_world(lab, b["id"], self.vol, self.scene.zscale, self.scene.ve, bbox=b["bbox"])
            self.scene.add("geobody", f"Body {b['id']} ({b['volume_m3'] / 1e6:.2f} Mm3)", [m])
            items.append(GB.footprint(lab, b["id"], self.vol, b["bbox"]))
        if push_polygons and self.stack is not None and items:
            name = f"Geobodies: {self.vol_name}"
            lay = next((l for l in self.stack.layers if l["kind"] == "polygons" and l["name"] == name), None)
            if lay is None:
                self.stack.add("polygons", name, src={"polygons": items}, style={"color": (230, 120, 20)})
            else:
                lay["src"]["polygons"] = items
            self.stack.save()
            self.bus.notify("layers", ids=[name], source="view3d", what="geobody outlines")
        self._refresh_tree()
        self.redraw()
        return bodies

    def colour_horizon_by(self, name, values, label="attribute", lut="Rainbow (attributes)"):
        """34e: a horizon surface coloured by an attribute grid (nil, nxl) - e.g. an amplitude extraction."""
        vol = self.vol
        hz = self.project.horizons.get(name)
        g = hz.data.get(f"3D:{self.vol_name}") if hz is not None else None
        if g is None:
            raise KeyError(name)
        I, J = np.meshgrid(vol.inlines, vol.xlines, indexing="ij")
        X, Y = vol.xy(I, J)
        v = np.asarray(values, float)
        fin = v[np.isfinite(v)]
        m = S3.surface_mesh(X, Y, g, self._lut(lut), (float(fin.min()), float(fin.max())), v, 25.0,
                            self.scene.zscale, self.scene.ve, f"{name}: {label}")
        self.scene.replace("horizon", name, [m], colorbar=(lut, float(fin.min()), float(fin.max()), label))
        self.redraw()
        return m

    def highlight_wells(self, names):
        self.highlight = list(names or [])
        self.add_wells()
        self.redraw()
        return self.highlight

    def fit(self):
        self.scene.fit()
        self.redraw()

    # ---------------------------------------------------------------- output (33e)
    def screenshot(self, path, with_glow=None):
        """PNG of the view.  with_glow False = "Export without glow" (PLAN 39g); None = the export_glow setting."""
        keep = self.scene.glow_suppressed, self.scene.glow_export
        if with_glow is not None:
            self.scene.glow_suppressed, self.scene.glow_export = (not with_glow), bool(with_glow)
            self.redraw()
        try:
            return self._screenshot(path)
        finally:
            self.scene.glow_suppressed, self.scene.glow_export = keep
            self.redraw()

    def _screenshot(self, path):
        if not self.software and hasattr(self.canvas, "grabFramebuffer"):
            try:
                img = self.canvas.grabFramebuffer()
                if img.save(path):
                    return path
            except Exception:
                pass
        legend = [(o.name, (0.9, 0.9, 0.9)) for o in self.scene.objects if o.visible and o.kind in ("horizon", "fault")]
        return self.scene.render_png(path, (1200, 900), title=f"{self.project.name} - {self.vol_name}", legend=legend)

    def turntable(self, path, frames=36, with_glow=None):
        """Turntable GIF; glow as shown unless with_glow=False ("Export without glow", PLAN 39g)."""
        keep = self.scene.glow_export
        if with_glow is not None:
            self.scene.glow_export = bool(with_glow)
        try:
            return self.scene.turntable(path, frames)
        finally:
            self.scene.glow_export = keep

    def save_state(self):
        self.scene.save(self.project)
        if self.project.path:
            try:
                self.project.save()
            except Exception:
                pass
        return self.project.display.get(S3.Scene.KEY)

    def _save_view_dialog(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save view", "Name:")
        if ok and name:
            self.scene.save_view(name)
            self.save_state()

    def _screenshot_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Screenshot", "", "PNG (*.png)")
        if path:
            choice, ok = QtWidgets.QInputDialog.getItem(self, "Screenshot", "Glow:",
                                                        ["With glow (as shown)", "Export without glow"], 0, False)
            if ok:
                self._say(f"Saved {self.screenshot(path, with_glow=not str(choice).startswith('Export without'))}")

    def _turntable_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Turntable animation", "", "GIF (*.gif)")
        if path:
            self._say(f"Saved {self.turntable(path)}")

    # ---------------------------------------------------------------- map bus (33d)
    def _on_event(self, ev):
        t, k = ev["topic"], ev.get("kind")
        if t == "active" and k in ("inline", "crossline", "timeslice") and ev.get("volume") == self.vol_name:
            mode = {"inline": "Inline", "crossline": "Crossline", "timeslice": "Time slice"}[k]
            if self.planes.get(mode) != ev.get("value"):
                self.set_plane(mode, ev.get("value"), publish=False)
        elif t == "selection" and k == "wells":
            self.highlight_wells(ev.get("ids") or [])
        elif t == "data" and k in ("picks", "horizons"):
            self.add_horizons()
            self.redraw()
        elif t == "data" and k == "faults":
            self.add_faults()
            self.redraw()
        elif t == "data" and k in ("tops", "wells"):
            self.add_wells()
            self.redraw()
        elif t == "data" and k in ("layers",):
            self.add_culture()
            self.redraw()

    # ---------------------------------------------------------------- glow (PLAN 39d)
    def _glow_emphasis(self, obj):
        from .. import glow as GL
        kind = {"horizon": "horizon", "fault": "fault", "well": "well", "line2d": "line"}.get(obj.kind)
        if obj.kind == "well" and obj.name in (self.highlight or []):
            return "selected"
        if obj.kind == "geobody" and self.bodies:
            return None
        if kind is None:
            return None
        return self.glow_state.emphasis(kind, obj.name)

    def glow_notice(self):
        """One-line status: auto-limit or 'glow not supported by this driver'."""
        f = getattr(self.canvas, "glow_failed", "")
        if isinstance(f, str) and f:
            return f"Glow off: {f}"
        return self.scene.glow_notice

    def toggle_glow(self):
        self.glow_state.settings.toggle()
        self.glow_state.broadcast()
        self.redraw()
        return self.glow_state.settings["enabled"]

    def closeEvent(self, ev):
        try:
            self.save_state()
        except Exception:
            pass
        self.bus.unsubscribe("view3d")
        try:
            self.glow_state.close()
        except Exception:
            pass
        ev.accept()
