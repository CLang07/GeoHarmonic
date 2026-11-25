"""PLAN 39 tests: Glow / Halo display effects.

run      - stroke specs, Glow / Halo / Off, per-object overrides, active-only + auto-limit, halo auto-contrast, project
           round-trip, map-bus emphasis + display broadcast, section figure export with / without glow (pixels),
           3D glow meshes (outline, shell, plane outline, tubes), software render with / without glow, glow cache.
run_gui  - (Qt mock) GlowManager underlays: created / synced on data, colour, visibility, removed on detach,
           never interactive, below the parent; Off; override; auto-limit + status note; decimated while dragging;
           main window: Display menu actions, G toggle, horizons + faults + wells glow on the section, selection in
           one window glows in the others (Live Map, 3D) via the bus, section export with / without glow, 3D
           screenshot with / without glow.
"""
import os

import numpy as np

from . import glow as G


def _px(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), float)


def run(check, tmp):
    from .gui.colormaps import lut
    os.makedirs(tmp, exist_ok=True)
    st = G.GlowSettings()
    # ---- 39a/b stroke specs
    s = G.strokes(st, "glow", (255, 200, 0), 2.0)
    w = [x[0] for x in s]
    a = [x[1][3] for x in s]
    check(len(s) == st["layers"] == 3 and w == sorted(w, reverse=True) and a == sorted(a) and w[-1] > 2.0
          and abs(w[0] - (2.0 + 2 * st["width_px"])) < 1e-9,
          f"39a: glow = {len(s)} underlay strokes, outermost widest + faintest (widths {np.round(w, 1).tolist()}, "
          f"alpha {a})")
    check(all(x[1][:3] == (255, 200, 0) for x in s), "39b: glow colour matches the object by default")
    st.update(colour="white")
    check(G.strokes(st, "glow", (255, 200, 0))[0][1][:3] == (255, 255, 255), "39c: glow colour white / black / custom")
    st.update(colour="match")
    h = G.strokes(st, "halo", (255, 200, 0), 2.0, lut=lut("Seismic (red-white-blue)"))
    check(len(h) == 1 and h[0][1] == (0, 0, 0, 255) and abs(h[0][0] - (2.0 + 2 * st["halo_px"])) < 1e-9,
          "39b: halo = one solid outline, black against the red-white-blue seismic (auto contrast)")
    check(G.contrast_colour(against=(21, 24, 29)) == (255, 255, 255) and G.contrast_colour(lut("Grey")) == (0, 0, 0)
          and G.contrast_colour(np.tile([[10, 10, 40, 255]], (256, 1))) == (255, 255, 255),
          "39b: halo auto colour white on dark backgrounds / dark colour tables, black on light ones")
    check(G.strokes(st, None, (1, 2, 3)) == [] and G.strokes(st, "off", (1, 2, 3)) == [], "39b: Off = no underlays")
    for k in (1, 4):
        st.update(layers=k)
        check(len(G.strokes(st, "glow", (9, 9, 9))) == k, f"39c: falloff layers {k}")
    st.update(layers=3)
    # ---- 39b/c modes + overrides
    check(st.effective("horizon", "Top") == "glow", "39e: horizons glow by default")
    check(st.effective("fault", "F1") == "glow" and st.effective("stick", "F1") == "glow", "39e: faults glow by default")
    check(st.effective("culture", "Coast") is None and st.effective("culture", "Coast", "selected") == "glow",
          "39e: other objects glow only when selected / active")
    st.update(mode="halo")
    check(st.effective("horizon", "Top") == "halo", "39b: Glow -> Halo switch")
    st.update(mode="off")
    check(st.effective("horizon", "Top") is None, "39b: Off switch")
    st.update(mode="glow", enabled=False)
    st.set_override("horizon", "Top", "on")
    check(st.effective("horizon", "Top") == "glow" and st.effective("horizon", "Base") is None,
          "39c: per-object 'On' beats the global switch (global off)")
    st.update(enabled=True)
    st.set_override("horizon", "Top", "off")
    check(st.effective("horizon", "Top", "active") is None and st.effective("horizon", "Base") == "glow",
          "39c: per-object 'Off' beats global on (even when active)")
    st.set_override("horizon", "Top", "global", colour=(0, 255, 255))
    check(G.strokes(st, "glow", (255, 0, 0), override=st.override("horizon", "Top"))[0][1][:3] == (0, 255, 255),
          "39c: per-object glow colour")
    st.set_override("horizon", "Top", "global")
    check("horizon:Top" not in st.overrides, "39c: 'Use global' clears the override")
    st.toggle()
    check(not st["enabled"], "39c: G toggles glow globally")
    st.toggle()
    # ---- 39e active-only + emphasis
    st.update(active_only=True)
    check(st.effective("horizon", "Top") is None and st.effective("horizon", "Top", "active") == "glow"
          and st.effective("horizon", "Top", "qc") == "glow",
          "39e: 'glow only the active object' - plain objects off, active / QC still glow")
    st.update(active_only=False)
    plain, act = G.strokes(st, "glow", (1, 1, 1), 2), G.strokes(st, "glow", (1, 1, 1), 2, "active")
    check(act[0][0] > plain[0][0] and act[-1][1][3] >= plain[-1][1][3], "39e: the active object glows stronger")
    check(G.strokes(st, "glow", (1, 200, 1), 2, "qc")[0][1][:3] == G.QC_RGB, "39e: QC flags glow red")
    st.update(hover=False)
    check(st.effective("horizon", "X", "hover") is None, "39e: hover highlight can be switched off")
    st.update(hover=True)
    # ---- 39h auto-limit + decimation
    st.update(threshold=2000)
    check(not G.auto_limit(2000, st) and G.auto_limit(2001, st) and "active / selected" in G.limit_note(2500, st)
          and st.effective("stick", "F9", limited=True) is None and st.effective("stick", "F9", "selected", True),
          "39h: above 2,000 objects glow is limited to the active / selected ones (status note)")
    x = np.arange(20000.0)
    y = np.sin(x / 50.0)
    y[777] = np.nan
    dx, dy = G.decimate_xy(x, y, 3000)
    check(len(dx) <= 3002 and dx[0] == 0 and dx[-1] == 19999 and np.isnan(dy).any(),
          "39h: underlays decimated while dragging (gaps and ends kept)")
    # ---- 39c project round trip
    class P:
        display = {}
    st2 = G.GlowSettings(mode="halo", width_px=9, layers=2, colour="custom", custom=(10, 20, 30))
    st2.set_override("fault", "F1", "off")
    st2.save(P)
    st3 = G.GlowSettings.load(P)
    check(st3["mode"] == "halo" and st3["layers"] == 2 and st3["custom"] == (10, 20, 30)
          and st3.override("fault", "F1")["mode"] == "off" and "glow" in P.display,
          "39c: glow settings + per-object overrides saved in the project")
    check(G.GlowSettings.load(None)["mode"] == "glow", "39c: no project -> defaults")
    # ---- 39f map bus
    from .mapbus import MapBus
    bus = MapBus()
    a_state = G.GlowState(bus, G.GlowSettings(), name="seismic-glow")
    b_state = G.GlowState(bus, G.GlowSettings(), name="map-glow")
    bus.selection.select("horizons", ["Top"], source="livemap")
    check(a_state.emphasis("horizon", "Top") == "selected" and b_state.emphasis("horizon", "Top") == "selected"
          and a_state.emphasis("horizon", "Base") is None, "39f: a horizon selected in one window glows in all")
    bus.active.set("fault", "F1", source="seismic")
    check(b_state.emphasis("fault", "F1") == "active", "39f: the active fault (ActiveObject) glows stronger everywhere")
    bus.selection.select("wells", ["Maui 5"], source="livemap")
    check(a_state.emphasis("well", "maui-5") == "selected", "39f: well ids matched like the bus (Maui 5 == maui-5)")
    a_state.settings.update(mode="halo", width_px=11)
    n0 = b_state.events
    a_state.broadcast()
    check(b_state.settings["mode"] == "halo" and b_state.settings["width_px"] == 11 and b_state.events > n0
          and bus.last("display", "glow") is not None, "39f: settings changes broadcast as a display event")
    a_state.set_hover("horizon", "Base")
    check(b_state.hover == ("horizon", "Base") and b_state.emphasis("horizon", "Base") == "hover"
          and bus.last("display", "hover").get("local"), "39e/f: hover highlight shared (in-process only)")
    a_state.set_qc("well", "W-9")
    a_state.set_tracking("Deep")
    check(a_state.emphasis("well", "W-9") == "qc" and a_state.emphasis("horizon", "Deep") == "tracking",
          "39e: QC flags and the autotracker front")
    a_state.close()
    b_state.close()
    # ---- 39g export with / without glow
    nx, nt = 120, 100
    data = np.random.default_rng(1).normal(0, 0.05, (nx, nt))
    xs = np.arange(nx, dtype=float)
    ts = np.arange(nt, dtype=float) * 4.0
    hzn = {"x": xs, "y": np.full(nx, 200.0), "colour": (255, 220, 0), "width": 2, "kind": "horizon", "name": "Top"}
    flt = {"x": np.array([60.0, 70.0]), "y": np.array([40.0, 380.0]), "colour": (230, 30, 30), "width": 2,
           "kind": "fault", "name": "F1"}
    st = G.GlowSettings()
    p1, p0 = os.path.join(tmp, "glow_on.png"), os.path.join(tmp, "glow_off.png")
    n1 = G.export_section_png(p1, data, xs, ts, [hzn, flt], lut=lut("Grey"), settings=st, with_glow=True)
    n0 = G.export_section_png(p0, data, xs, ts, [hzn, flt], lut=lut("Grey"), settings=st, with_glow=False)
    A, B = _px(p1), _px(p0)
    # the horizon row in pixels: where the yellow line is, look a few pixels above it
    yel = np.argwhere((B[:, :, 0] > 200) & (B[:, :, 1] > 180) & (B[:, :, 2] < 90))
    r = int(np.median(yel[:, 0])) if len(yel) else A.shape[0] // 2
    cols = slice(int(A.shape[1] * 0.15), int(A.shape[1] * 0.4))
    near = np.abs(A[r - 6, cols] - B[r - 6, cols]).mean()
    far = np.abs(A[r - 60, cols] - B[r - 60, cols]).mean()
    check(n1 == 2 and n0 == 0 and len(yel) and near > 10 and far < 2,
          f"39g: section export with glow shows the glow beside the horizon (mean |diff| {near:.0f} at 6 px, "
          f"{far:.1f} at 60 px); 'Export without glow' has none")
    eff = G.mpl_path_effects(st, "glow", (255, 0, 0), 1.5)
    check(len(eff) == st["layers"] + 1, "39g: matplotlib path effects = glow strokes + the line")
    st_h = G.GlowSettings(mode="halo")
    p2 = os.path.join(tmp, "halo.png")
    G.export_section_png(p2, data, xs, ts, [hzn], lut=lut("Grey"), settings=st_h)
    H = _px(p2)
    check(H[r - 2, cols].mean() < B[r - 2, cols].mean() - 20, "39b/g: halo export = dark outline next to the line")
    # ---- 39d 3D
    from . import scene3d as S3
    X, Y = np.meshgrid(np.arange(3.0) * 100, np.arange(3.0) * 100)
    T = np.full(X.shape, 500.0)
    m = S3.surface_mesh(X, Y, T, lut("Rainbow (attributes)"), vrange=(0, 1000))
    e = G.boundary_edges(m.pos)
    check(len(e) == 8, f"39d: surface outline = boundary edges only ({len(e)} of 16 for a 2x2-cell grid)")
    scn = S3.Scene()
    o = scn.add("horizon", "Top", [m])
    tube = S3.tube_mesh(np.array([[0, 0, 0], [0, 0, -500.0]]), 10.0, np.array([[1, 1, 1, 1], [1, 1, 1, 1.0]]))
    ow = scn.add("well", "W-1", [tube])
    sl = scn.add("slice", "Inline", [S3.surface_mesh(X, np.zeros_like(X), np.array([[0, 0, 0], [500, 500, 500],
                                                                                    [1000, 1000, 1000.0]]),
                                                     lut("Grey"), vrange=(0, 1))])
    check(scn.glow_list() == [], "39d: no glow without settings (3D viewer default before the window sets them)")
    scn.glow = G.GlowSettings()
    gl = scn.glow_list()
    kinds = {mm.name.split(":")[0] for mm, _ in gl}
    check(len(gl) > 0 and all(mm.meta.get("glow") for mm, _ in gl) and any(mm.kind == "lines" for mm, _ in gl),
          f"39d: 3D glow underlays for horizons / wells / plane outlines ({len(gl)} meshes)")
    pl = G.plane_outline(sl.meshes[0].pos)
    check(pl.shape == (4, 2, 3), "39e: 3D plane outline = 4 edges")
    plain = len(gl)
    scn.glow_emphasis = lambda ob: "active" if ob is o else None
    gl2 = scn.glow_list()
    check(any(mm.meta.get("shell") for mm, _ in gl2) and len(gl2) > plain,
          "39d: the active surface gets an enlarged transparent shell (fallback for vtkOutlineGlowPass)")
    check(scn.glow_list()[0][0] is gl2[0][0], "39h: 3D glow meshes cached until the object or settings change")
    tw = [mm for mm, _ in gl2 if mm.name.startswith("glow:") and mm.kind == "triangles"]
    check(tw and np.abs(tw[0].pos[:, :2]).max() > 10.0, "39d: well path glow = fatter transparent tube")
    scn.fit()
    q1, q0 = os.path.join(tmp, "g3d_on.png"), os.path.join(tmp, "g3d_off.png")
    scn.render_png(q1, (300, 240))
    scn.glow_export = False
    scn.render_png(q0, (300, 240))
    scn.glow_export = None
    d3 = np.abs(_px(q1) - _px(q0)).mean()
    check(d3 > 0.05, f"39g: 3D screenshot / turntable with glow differs from 'without glow' (mean |diff| {d3:.2f})")
    scn.glow_suppressed = True
    check(scn.glow_list() == [], "39g: glow suppressed while exporting without glow")
    scn.glow_suppressed = False
    scn.glow.update(threshold=1)
    scn.glow_emphasis = None
    check(scn.glow_list() == [] and "limited" in scn.glow_notice, "39h: 3D auto-limit above the threshold + notice")

    class Bad:
        id, kind, name, version, visible, opacity = "x", "horizon", "Bad", 0, True, 1.0
        meshes = [S3.Mesh("triangles", np.full((3, 3), np.nan), np.ones((3, 4)))]
    scn.glow.update(threshold=2000)
    scn.objects.append(Bad())
    try:
        scn.glow_list()
        ok = True
    except Exception:
        ok = False
    check(ok, "39d: a mesh the glow cannot handle never breaks the 3D view")


# ====================================================================== GUI (Qt mock or real Qt offscreen)
class _FakeItem:
    def __init__(self, x=None, y=None, connect="finite", pen=None, size=None, brush=None, **k):
        self.x = None if x is None else np.asarray(x, float)
        self.y = None if y is None else np.asarray(y, float)
        self.z = 0.0
        self.visible = True
        self.size = size
        self.pen = pen
        self.brush = brush

    def getData(self):
        return self.x, self.y

    def setData(self, x, y):
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)

    def zValue(self):
        return self.z

    def setZValue(self, z):
        self.z = float(z)

    def isVisible(self):
        return self.visible

    def setVisible(self, v):
        self.visible = bool(v)

    def setAcceptedMouseButtons(self, b):
        self.buttons = b

    def setAcceptHoverEvents(self, b):
        self.hover = b


class _FakePlot:
    def __init__(self):
        self.items = []

    def addItem(self, it, **k):
        self.items.append(it)

    def removeItem(self, it):
        if it in self.items:
            self.items.remove(it)


class _FakePG:
    PlotCurveItem = _FakeItem
    ScatterPlotItem = _FakeItem

    @staticmethod
    def mkPen(*a, **k):
        return ("pen", a, k)

    @staticmethod
    def mkBrush(*a, **k):
        return ("brush", a)


def run_gui(check, tmp):
    from .gui import glow as GG
    real_pg = GG.pg
    GG.pg = _FakePG
    try:
        _manager_tests(check, GG)
    finally:
        GG.pg = real_pg
    _window_tests(check, tmp)


def _manager_tests(check, GG):
    plot = _FakePlot()
    notes = []
    mgr = GG.GlowManager(plot, G.GlowSettings(), status=notes.append)
    hz = _FakeItem(np.arange(10.0), np.full(10, 100.0))
    hz.z = 5.0
    plot.addItem(hz)
    e = mgr.attach(hz, "horizon", "Top", (255, 220, 0), 2)
    und = mgr.underlays(hz)
    check(len(und) == 3 and all(u in plot.items for u in und) and all(u.z < hz.z for u in und)
          and und[0].z <= und[-1].z, "39a GUI: horizon gets 3 underlays, all below the line (outermost lowest)")
    check(all(getattr(u, "_glow_underlay", False) and hasattr(u, "buttons") and u.hover is False for u in und), "39a/i GUI: underlays ignore mouse buttons / hover (never picked or hit-tested)")

    class Ev:
        ignored = False

        def ignore(self):
            self.ignored = True
    ev = Ev()
    und[0].mouseClickEvent(ev)
    check(ev.ignored, "39i GUI: a click on an underlay passes through")
    hz.setData(np.arange(20.0), np.full(20, 140.0))
    und2 = mgr.underlays(hz)
    check(len(und2) == 3 and all(u not in plot.items for u in und) and len(und2[0].x) == 20
          and np.allclose(und2[0].y, 140.0), "39a GUI: underlays follow the parent's data (setData)")
    hz.setVisible(False)
    mgr.sync_visibility(hz)
    check(all(not u.visible for u in mgr.underlays(hz)), "39a GUI: underlays follow the parent's visibility")
    hz.setVisible(True)
    mgr.sync_visibility(hz)
    old = mgr.underlays(hz)
    mgr.set_colour(hz, (0, 255, 0))
    new = mgr.underlays(hz)
    check(e.colour == (0, 255, 0) and len(new) == 3 and not set(map(id, old)) & set(map(id, new)),
          "39a GUI: underlays follow the parent's colour (rebuilt)")
    mgr.settings.update(mode="halo")
    mgr.refresh()
    check(len(mgr.underlays(hz)) == 1, "39b GUI: Halo = one outline underlay")
    mgr.settings.update(mode="off")
    mgr.refresh()
    check(mgr.underlays(hz) == [], "39b GUI: Off removes the underlays")
    mgr.settings.update(mode="glow")
    mgr.settings.set_override("horizon", "Top", "off")
    mgr.refresh()
    check(mgr.underlays(hz) == [], "39c GUI: per-object Off override")
    mgr.settings.set_override("horizon", "Top", "global")
    mgr.refresh()
    sc = _FakeItem(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    mgr.attach(sc, "well", "W-1", (0, 0, 0), symbol_size=10)
    su = mgr.underlays(sc)
    check(len(su) == 3 and su[0].size > su[-1].size > 10, "39a GUI: scatter markers get soft larger symbols underneath")
    n_before = len(plot.items)
    check(mgr.detach(hz) and len(plot.items) == n_before - 3 and not mgr.underlays(hz),
          "39i GUI: underlays removed with the parent (delete)")
    # auto-limit
    mgr.settings.update(threshold=3)
    sticks = [_FakeItem(np.array([0.0, 1.0]), np.array([0.0, 1.0])) for _ in range(5)]
    for k, it in enumerate(sticks):
        mgr.attach(it, "stick", f"F{k}", (200, 0, 0), 2)
    mgr.state.qc.add(("stick", "F2"))
    mgr.refresh()
    check(not mgr.underlays(sticks[0]) and mgr.underlays(sticks[2]) and any("limited" in n for n in notes),
          "39h/i GUI: above the threshold only active / selected / QC objects glow + status-bar note")
    mgr.settings.update(threshold=2000)
    big = _FakeItem(np.arange(10000.0), np.zeros(10000))
    mgr.attach(big, "horizon", "Big", (1, 1, 1), 2)
    mgr.set_dragging(True)
    d1 = len(mgr.underlays(big)[0].x)
    mgr.set_dragging(False)
    d0 = len(mgr.underlays(big)[0].x)
    check(d1 <= mgr.max_drag_points + 2 and d0 == 10000, "39h GUI: decimated underlays while dragging, full on release")
    mgr.clear()
    check(not mgr.entries and not any(getattr(i, "_glow_underlay", False) for i in plot.items),
          "39a GUI: clear() removes every underlay")


def _window_tests(check, tmp):
    os.makedirs(tmp, exist_ok=True)
    from .gui.main_window import MainWindow
    from .test_scene3d import _project
    proj, truth, d = _project(tmp, "glow_gui")
    vol = proj.volume("Synth")
    f = proj.faults.new("F1")
    for il in (vol.inlines[5], vol.inlines[15]):
        a = vol.xy(float(il), float(vol.xlines[10]))
        b = vol.xy(float(il), float(vol.xlines[14]))
        f.add_stick(np.array([[a[0], a[1], 200.0], [b[0], b[1], 700.0]]))
    proj.save()
    mw = MainWindow(proj.path)
    sec = mw.section
    sec.source, sec.mode, sec.number = ("3D", "Synth"), "Inline", int(vol.inlines[10])
    sec.redraw()
    kinds = {e.kind for e in sec.glow.entries.values()}
    check({"horizon", "fault"} <= kinds, f"39e GUI: horizons and faults on the section are glow-managed ({sorted(kinds)})")
    check(sec.glow.state is mw.glow_state and mw.glow_state.settings is sec.glow.settings,
          "39f GUI: the section shares the main window's glow state (bus)")
    en = mw.glow_state.settings["enabled"]
    mw.toggle_glow()
    check(mw.glow_state.settings["enabled"] != en and mw.project.display["glow"]["settings"]["enabled"] != en,
          "39c GUI: Display > Glow on/off (G) toggles and saves in the project")
    mw.toggle_glow()
    mw.set_glow(mode="halo")
    v = mw.viewer3d("Synth", force_software=True)
    check(v.scene.glow is not None and v.scene.glow["mode"] == "halo", "39d GUI: the 3D Viewer uses the project glow")
    mw.set_glow(mode="glow", width_px=8.0)
    check(v.scene.glow["mode"] == "glow" and v.scene.glow["width_px"] == 8.0,
          "39f GUI: glow settings changed in the seismic window reach the 3D Viewer (display event)")
    lm = mw.live_map()
    names = [h.name for h in mw.project.horizons.items]
    other = next((n for n in names if n != mw.mapbus.active.get("horizon")), names[0])
    mw.mapbus.selection.select("horizons", [other], source="livemap")
    o = v.scene.find("horizon", other)
    check(mw.glow_state.emphasis("horizon", other) == "selected" and v._glow_emphasis(o) == "selected"
          and lm.glow_state.emphasis("horizon", other) == "selected",
          "39f GUI: a horizon selected in one window glows (stronger) in the section, Live Map and 3D")
    o = v.scene.find("horizon", "Top")
    sec.active_horizon = "Top"
    mw._horizons_changed()
    check(mw.mapbus.active.get("horizon") == "Top" and v._glow_emphasis(o) == "active",
          "39f GUI: the active horizon is published on the bus (ActiveObject) and glows as active in 3D")
    hit = sec.hover_at(float(vol.xlines[3]), 10000.0, now=1e9)
    check(hit is None, "39e GUI: hover far from any curve highlights nothing")
    p1, p0 = os.path.join(tmp, "sec_glow.png"), os.path.join(tmp, "sec_noglow.png")
    sec.export_figure(p1, with_glow=True)
    sec.export_figure(p0, with_glow=False)
    check(os.path.getsize(p1) > 5000 and os.path.getsize(p0) > 5000,
          "39g GUI: File > Export section image with glow and without glow")
    q1, q0 = os.path.join(tmp, "v3d_glow.png"), os.path.join(tmp, "v3d_noglow.png")
    v.screenshot(q1, with_glow=True)
    v.screenshot(q0, with_glow=False)
    diff = np.abs(_px(q1) - _px(q0)).mean()
    check(diff > 0.01 and not v.scene.glow_suppressed, f"39g GUI: 3D screenshot with / without glow (|diff| {diff:.2f})")
    on = v.toggle_glow()
    check(on == mw.glow_state.settings["enabled"], "39c GUI: G in the 3D Viewer toggles glow everywhere")
    v.toggle_glow()
    check(isinstance(v.glow_notice(), str), "39d GUI: 3D glow notice line (auto-limit / unsupported driver)")
    v.track("Top")
    check(not mw.glow_state.tracking and not v.glow_state.tracking, "39e GUI: tracking glow cleared when the run ends")
    for w in (lm, v):
        try:
            w.close()
        except Exception:
            pass
