"""3D viewer tests (PLAN 33): slices vs the store, surfaces (holes, faults), well tubes, camera math, picking,
software rendering + turntable, scene state."""
import math
import os

import numpy as np

from . import scene3d as S3


def _project(tmp, name="v3d"):
    from .test_maplayers import _setup
    return _setup(tmp, name)


def run(check, tmp):
    from .gui.colormaps import lut
    proj, truth, d = _project(tmp)
    vol = proj.volume("Synth")
    L = lut("Grey")
    il = int(vol.inlines[7])
    sec = np.asarray(vol.inline(il), float)
    m = S3.slice_mesh(vol, "Inline", il, L, (-1.0, 1.0))
    ntri = (sec.shape[0] - 1) * (sec.shape[1] - 1) * 2
    check(m.kind == "triangles" and len(m) == 3 * ntri, f"33b: inline plane = one quad pair per cell ({ntri} triangles)")
    # vertex colours = the store values through the LUT at the same positions
    x, y = vol.xy(float(il), float(vol.xlines[3]))
    k = 10
    zt = S3.world_z(vol.times[k])
    hit = np.flatnonzero((np.abs(m.pos[:, 0] - x) < 1e-3) & (np.abs(m.pos[:, 1] - y) < 1e-3) &
                         (np.abs(m.pos[:, 2] - zt) < 1e-3))
    exp = S3.lut_rgba([sec[3, k]], L, -1.0, 1.0)[0]
    check(len(hit) > 0 and np.allclose(m.rgba[hit[0]], exp), "33b: slice colours come from the stored samples")
    md = S3.slice_mesh(vol, "Inline", il, L, (-1, 1), decimate=4)
    check(len(md) < len(m) / 10, "33b: decimated plane while dragging (x16 fewer triangles)")
    ts = S3.slice_mesh(vol, "Time slice", float(vol.times[40]), L, (-1, 1))
    xl = S3.slice_mesh(vol, "Crossline", int(vol.xlines[5]), L, (-1, 1))
    check(np.allclose(ts.pos[:, 2], S3.world_z(vol.times[40])) and len(xl) > 0, "33b: time slice flat at its time; "
          "crossline plane")
    c = S3.plane_corners(vol, "Inline", il)
    check(c.shape == (4, 3) and abs(c[0, 2] - S3.world_z(vol.times[0])) < 1e-9, "33b: plane corners for the outline")
    # 2D line curtain
    ln = proj.line(proj.lines[0]["name"])
    cm = S3.curtain_mesh(ln.x, ln.y, ln.times, ln.data(), L, (-1, 1), decimate=2)
    check(len(cm) > 0 and np.nanmax(cm.pos[:, 0]) <= np.nanmax(ln.x) + 1e-6, "33b: 2D line as a vertical curtain")
    # surfaces: NaN holes stay open, fault jumps not bridged
    ny, nx = 20, 30
    Y, X = np.meshgrid(np.arange(ny) * 25.0, np.arange(nx) * 25.0, indexing="ij")
    T = 1000 + 0.2 * X
    T[5:8, 5:8] = np.nan
    T[:, 15:] += 60.0  # a fault with 60 ms throw between columns 14 and 15
    sm = S3.surface_mesh(X, Y, T, L, max_jump=20.0)
    full = (ny - 1) * (nx - 1) * 2
    tz = -sm.pos[:, 2].reshape(-1, 3) / S3.DEFAULT_ZSCALE
    span = tz.max(1) - tz.min(1)
    Tf = 1000 + 0.2 * X
    Tf[:, 15:] += 60.0
    nohole = S3.surface_mesh(X, Y, Tf, L, max_jump=20.0).meta["triangles"]
    check(np.isfinite(sm.pos).all() and span.max() <= 20.0 + 1e-6 and sm.meta["triangles"] == nohole - 30 and
          nohole == full - 2 * (ny - 1), f"33c: horizon surface - NaN holes open, fault not bridged "
          f"({sm.meta['triangles']} of {full} triangles)")
    att = S3.surface_mesh(X, Y, T, L, values=np.where(np.isfinite(T), X, np.nan), vrange=(0, 725))
    check(att.meta["vrange"] == (0, 725), "33c: surface coloured by an attribute")
    # deviated well tube
    from .laterals import synthetic_laterals
    tr = synthetic_laterals(1, x0=0.0, y0=0.0, lateral=800.0)[0]
    path = np.column_stack([tr.x, tr.y, -tr.z])[::5]
    tb = S3.tube_mesh(path, 10.0, np.array([0.9, 0.2, 0.1, 1.0]), sides=8)
    pts = tb.pos

    def dist_to_path(p):
        a, b = path[:-1], path[1:]
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
        return np.min(np.linalg.norm(a + t[:, None] * ab - p, axis=1))
    ds = [dist_to_path(p) for p in pts[::97]]
    check(len(tb) == (len(path) - 1) * 8 * 2 * 3 and max(abs(np.array(ds) - 10.0)) < 0.6,
          "33c: well tube follows a deviated path at the tube radius")
    logc = S3.lut_rgba(np.linspace(0, 150, len(path)), L, 0, 150)
    tl = S3.tube_mesh(path, 8.0, logc)
    check(len(tl) == len(tb), "33c: well tube coloured by a log")
    dm = S3.disc_mesh((0, 0, -1000), (0, 0, 1), 30.0, (1, 1, 0, 1), sides=12)
    check(len(dm) == 36 and np.allclose(dm.pos[:, 2], -1000), "33c: top disc perpendicular to the path")
    pl = S3.polyline_mesh([0, 100, np.nan, 200, 300], [0, 0, 0, 0, 0], [-10] * 5, (1, 1, 1, 1))
    check(pl.kind == "lines" and len(pl) == 4, "33c: culture polyline (NaN breaks the line)")

    # camera math
    cam = S3.Camera(target=(500.0, 500.0, -1000.0), distance=2000.0, azimuth=225.0, elevation=30.0)
    p = cam.project([cam.target], 800, 600)[0]
    check(abs(p[0] - 400) < 1e-6 and abs(p[1] - 300) < 1e-6, "33a: camera target projects to the centre")
    o, dvec = cam.ray(400, 300, 800, 600)
    on = o + dvec * np.dot(cam.target - o, dvec)
    check(np.linalg.norm(on - cam.target) < 1e-6 * cam.distance, "33a: picking ray through the centre hits the target")
    e0 = cam.eye()
    cam.orbit(100, 0)
    check(abs(cam.azimuth - 265.0) < 1e-9 and abs(np.linalg.norm(cam.eye() - cam.target) - 2000) < 1e-6 and
          not np.allclose(cam.eye(), e0), "33a: orbit keeps the distance")
    cam.orbit(0, 1000)
    check(cam.elevation == 89.0, "33a: elevation clamped")
    cam.elevation = 30.0
    t0 = cam.target.copy()
    pa = cam.project([t0], 800, 600)[0]
    cam.pan(50, 0, 600)
    pb = cam.project([t0], 800, 600)[0]
    check(abs((pb[0] - pa[0]) - 50) < 1.0 and abs(pb[1] - pa[1]) < 1.0, "33a: pan - the scene follows the mouse")
    cam.zoom(2.0)
    check(abs(cam.distance - 1000.0) < 1e-9, "33a: zoom")
    cam.fit((0, 0, -2000), (1000, 1000, 0))
    corners = np.array([[x_, y_, z_] for x_ in (0, 1000) for y_ in (0, 1000) for z_ in (-2000, 0)])
    pp = cam.project(corners, 800, 600)
    check(np.all((pp[:, 0] > 0) & (pp[:, 0] < 800) & (pp[:, 1] > 0) & (pp[:, 1] < 600)), "33a: fit shows the box")
    # pick a known point on an inline plane
    sc = S3.Scene()
    sc.camera = S3.Camera(distance=3000.0, azimuth=200.0, elevation=20.0)
    c = S3.plane_corners(vol, "Inline", il)
    sc.camera.target = c.mean(axis=0)
    xl_t, t_t = float(vol.xlines[12]), float(vol.times[60])
    xw, yw = vol.xy(float(il), xl_t)
    px = sc.camera.project([[float(xw), float(yw), S3.world_z(t_t)]], 800, 600)[0]
    hitp = S3.pick_on_plane(sc.camera, px[0], px[1], 800, 600, vol, "Inline", il)
    check(hitp is not None and abs(hitp[0] - il) < 1e-9 and abs(hitp[1] - xl_t) < 0.05 and abs(hitp[2] - t_t) < 0.5,
          f"33a/34a: ray-plane pick maps to the right il / xl / time ({hitp})")
    check(S3.pick_on_plane(sc.camera, 2, 2, 800, 600, vol, "Inline", il) is None, "33a: picks outside the plane rejected")
    # vertical exaggeration
    check(abs(S3.world_z(100, 1.25, 3.0) + 375.0) < 1e-9, "33a: vertical exaggeration scales z")
    # scene, state, rendering
    box, labels = S3.box_mesh(vol)
    check(len(box) == 24 and len(labels) == 15 and labels[0][1].startswith("IL"), "33a: axis box + IL / XL / time labels")
    sc.add("box", "Survey box", [box])
    sc.add("slice", "Inline", [m], colorbar=("Grey", -1, 1, ""))
    so = sc.add("surface", "Top", [S3.surface_mesh(*vol.xy(*np.meshgrid(vol.inlines, vol.xlines, indexing="ij")),
                                                  truth["3D:Synth"], lut("Rainbow (attributes)"))], opacity=0.6)
    sc.fit()
    dl = sc.draw_list()
    check(dl[-1][1] < 1.0 and all(op == 1.0 for _, op in dl[:-1]), "33a: transparent objects drawn last")
    sc.set_visible(so.id, False)
    check(all(mm is not so.meshes[0] for mm, _ in sc.draw_list()), "33a: hidden objects not drawn")
    sc.set_visible(so.id, True)
    names = sc.save_view("NE look")
    az = sc.camera.azimuth
    sc.camera.orbit(90, 0)
    sc.restore_view("NE look")
    check(names == ["NE look"] and abs(sc.camera.azimuth - az) < 1e-9, "33a: saved camera views")

    class P:
        display = {}
    sc.save(P)
    s2, objs = S3.Scene.load_state(P)
    check(s2.views == sc.views and len(objs) == 3 and objs[2]["opacity"] == 0.6, "33a: window state saved in the project")
    png = os.path.join(d, "v3d.png")
    sc.render_png(png, (640, 480), title="3D test", legend=[("Top", (1, 0, 0))])
    from PIL import Image
    im = np.asarray(Image.open(png).convert("RGB"), float)
    check(im.shape[:2] == (480, 640) and im.std() > 10, "33e: screenshot PNG (software renderer)")
    gif = sc.turntable(os.path.join(d, "turn.gif"), frames=4, size=(240, 180))
    g = Image.open(gif)
    check(getattr(g, "n_frames", 1) == 4 and abs(sc.camera.azimuth - az) < 1e-9, "33e: turntable GIF (camera restored)")
    return sc


def real_qt():
    """True when the real PySide6 is imported (Chris's PC), False under dev/fakeqt."""
    import PySide6
    return "fakeqt" not in (getattr(PySide6, "__file__", "") or "").replace("\\", "/")


def run_gl_real(check, mw, vol):
    """PLAN 38a regression under real Qt (offscreen): create the canvas, initializeGL, paintGL twice, resize,
    paintGL again, grab the framebuffer.  Without a usable GL context: skip and check the software fallback."""
    from PySide6 import QtWidgets
    from .gui import view3d as V3
    app = QtWidgets.QApplication.instance()
    g = V3.Viewer3D(mw.project, "Synth", mw.mapbus)
    cv = g.canvas
    check("f" not in vars(cv), "38a: the GL canvas keeps no cached QOpenGLFunctions (self.f)")
    g.show()
    for _ in range(5):
        app.processEvents()
    ctx = cv.context() if hasattr(cv, "context") else None
    if ctx is None or not ctx.isValid() or cv.program is None:
        print(f"SKIP 38a: no OpenGL context here ({getattr(cv, 'gl_error', None) or 'context not created'})"
              " - checking the software fallback instead")
        if not getattr(cv, "gl_error", None):
            cv.gl_error = "no OpenGL context (test)"
        check(g._check_gl() and g.software, "38a: no GL context -> software fallback window")
        g.close()
        g.bus.unsubscribe("view3d")
        return
    errors = []
    try:
        cv.makeCurrent()
        cv.initializeGL()
        cv.paintGL()
        cv.paintGL()
        n1 = len(cv.buffers)
        cv.resize(640, 480)
        cv.resizeGL(640, 480)
        cv.paintGL()
        g.set_plane("Inline", int(vol.inlines[3]))
        cv.paintGL()
        cv.doneCurrent()
    except Exception as e:  # the PC failure was an AttributeError here
        errors.append(f"{type(e).__name__}: {e}")
    check(not errors and cv.program is not None,
          "38a: real OpenGL - initializeGL, paintGL x2, resize, paintGL" + (f" ({errors[0]})" if errors else ""))
    check(len(cv.buffers) == len(g.scene.draw_list()) and n1 > 0 if not errors else False,
          "38a: real OpenGL - one GL buffer per mesh after a plane change")
    try:
        img = cv.grabFramebuffer()
        ok = not img.isNull() and img.width() > 0
    except Exception as e:
        ok = False
        print(f"(grabFramebuffer: {e})")
    check(ok, "38a: real OpenGL - grabFramebuffer returns the rendered image")
    g.close()
    g.bus.unsubscribe("view3d")


def run_gui(check, tmp):
    """33 GUI: 3D Viewer from the main window - planes, objects, map bus links both ways, screenshot, state."""
    from .gui.main_window import MainWindow
    from .faults import Fault  # noqa: F401
    proj, truth, d = _project(tmp, "v3d_gui")
    f = proj.faults.new("F1")
    vol = proj.volume("Synth")
    for il in (vol.inlines[5], vol.inlines[15]):
        a = vol.xy(float(il), float(vol.xlines[10]))
        b = vol.xy(float(il), float(vol.xlines[14]))
        f.add_stick(np.array([[a[0], a[1], 200.0], [b[0], b[1], 700.0]]))
    proj.save()
    mw = MainWindow(proj.path)
    v = mw.viewer3d("Synth", force_software=True)
    kinds = {o.kind for o in v.scene.objects}
    check({"box", "slice", "horizon", "fault"} <= kinds and len([o for o in v.scene.objects if o.kind == "slice"]) == 3,
          f"33a GUI: 3D window with box, three planes, horizons, fault ({sorted(kinds)})")
    check("well" in kinds, "33c GUI: wells (project well paths) as tubes")
    opened = []
    v.open_section = lambda *a: opened.append(a)
    il = int(vol.inlines[9])
    v.set_plane("Inline", il, dragging=True)
    o = v.scene.find("slice", "Inline")
    check(o.meshes[0].meta["decimate"] == 4 and not opened, "33b GUI: decimated plane while dragging")
    v.full_resolution()
    check(v.scene.find("slice", "Inline").meshes[0].meta["decimate"] == 1 and opened and opened[-1][2:] == ("Inline", il)
          and mw.mapbus.active.get("inline") == il, "33b/33d GUI: full resolution on release, seismic window follows")
    # seismic window -> 3D plane
    mw.mapbus.active.set("crossline", int(vol.xlines[7]), source="seismic", volume="Synth")
    check(v.planes["Crossline"] == int(vol.xlines[7]), "33d GUI: crossline from the seismic window moves the 3D plane")
    # Live Map selection highlights wells
    mw.mapbus.selection.select("wells", ["W-2"], source="livemap")
    w2 = v.scene.find("well", "W-2")
    check(v.highlight == ["W-2"] and w2 is not None and np.allclose(w2.meshes[0].rgba[0], [1.0, 0.85, 0.0, 1.0]),
          "33d GUI: wells selected on the Live Map highlighted in 3D")
    # new picks rebuild the horizon surface
    h = mw.project.horizons.get("Top")
    old = v.scene.find("horizon", "Top").version
    h.data["3D:Synth"] = h.data["3D:Synth"] + 10.0
    mw._horizons_changed()
    check(v.scene.find("horizon", "Top").version > old, "33d GUI: edited picks update the 3D surface")
    ve0 = len(v.scene.find("slice", "Inline").meshes[0])
    v.set_ve(3.0)
    zmin = float(v.scene.find("slice", "Inline").meshes[0].pos[:, 2].min())
    check(abs(zmin - S3.world_z(vol.times[-1], v.scene.zscale, 3.0)) < 1e-3 and ve0 > 0, "33a GUI: vertical exaggeration")
    png = os.path.join(d, "shot.png")
    v.screenshot(png)
    check(os.path.getsize(png) > 5000, "33e GUI: screenshot")
    v.scene.save_view("view A")
    st = v.save_state()
    from .project import SeisProject
    p2 = SeisProject.open(proj.path)
    check("view A" in p2.display["view3d"]["views"] and st["ve"] == 3.0, "33a GUI: 3D window state saved in the project")
    v2 = mw.viewer3d("Synth")
    check(v2 is v, "33 GUI: one 3D window per volume")
    from .gui import view3d as V3
    check(isinstance(V3.HAVE_GL, bool), f"33a GUI: OpenGL available here: {V3.HAVE_GL} (software fallback tested)")
    # the OpenGL path: initialise + paint with whatever Qt provides (the mock here, a real context on the PC)
    if V3.HAVE_GL:
        if real_qt():
            run_gl_real(check, mw, vol)
        else:
            g = V3.Viewer3D(mw.project, "Synth", mw.mapbus)
            cv = g.canvas
            check("f" not in vars(cv), "38a: the GL canvas keeps no cached QOpenGLFunctions (self.f)")
            cv.initializeGL()
            if cv.program is None:  # Qt mock: link() is falsy - drive the draw code with mock objects anyway
                cv.program = V3.QOpenGLShaderProgram()
            cv.paintGL()
            n_buf = len(cv.buffers)
            check(n_buf == len(g.scene.draw_list()), f"33a GUI: OpenGL draw path uploads one buffer per mesh ({n_buf})")
            g.set_plane("Inline", int(vol.inlines[3]))
            cv.paintGL()
            check(len(cv.buffers) == len(g.scene.draw_list()), "33a GUI: replaced meshes release their GL buffers")
            cv.resizeGL(640, 480)
            cv.paintGL()
            check(len(cv.buffers) == len(g.scene.draw_list()), "38a: resizeGL + paintGL again (mock)")
            g.bus.unsubscribe("view3d")
    mw.maplink.close()
