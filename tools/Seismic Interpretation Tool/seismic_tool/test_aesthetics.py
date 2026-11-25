"""PLAN 40 tests: aesthetics package (hillshade + sun, colour tables, co-rendering, DPI, 3D depth cues, themes,
smooth motion, figure templates, automatic colours).  Lithology / track styles (40h) are tested in the Well
Correlation Tool (test_trackstyle)."""
import json
import math
import os

import numpy as np

from . import aesthetics as AE
from . import colortables as CT


def _px(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), float)


def run(check, tmp):
    os.makedirs(tmp, exist_ok=True)
    # ---------------------------------------------------------------- 40a hillshade vs an analytic surface
    x = np.arange(80) * 25.0
    X, Y = np.meshgrid(x, x)
    a = 0.3
    z = a * X  # plane rising to the east -> faces west (aspect 270)
    sl, asp = AE.slope_aspect(z, 25.0, 25.0)
    check(abs(sl[40, 40] - math.atan(a)) < 1e-9 and abs(math.degrees(asp[40, 40]) - 270.0) < 1e-6,
          "40a: slope / aspect of a plane = analytic (atan 0.3, faces west)")
    for az in (270.0, 315.0, 90.0):
        sun = AE.Sun(az, 35.0)
        hs = AE.hillshade(z, 25.0, 25.0, sun)
        e, s = math.radians(35.0), math.atan(a)
        want = max(0.0, math.sin(e) * math.cos(s) + math.cos(e) * math.sin(s) * math.cos(math.radians(az) - 1.5 * math.pi))
        check(abs(hs[40, 40] - want) < 1e-9, f"40a: hillshade = analytic Lambert value (sun az {az:g}: {hs[40, 40]:.4f})")
    dome = 200 * np.exp(-((X - 1000) ** 2 + (Y - 1000) ** 2) / 400.0 ** 2)
    h1 = AE.hillshade(dome, 25, 25, AE.Sun(315, 35))
    hm = AE.hillshade(dome, 25, 25, AE.Sun(315, 35, multi=True))
    check(h1[60, 20] > h1[20, 60] and np.std(hm) < np.std(h1),
          "40a: dome lit from the NW is bright on its NW flank; multi-directional is softer")
    hd = AE.hillshade(-dome, 25, 25, AE.Sun(315, 35), positive_down=True)
    check(np.allclose(hd, h1), "40a: time / depth maps (positive down): structural highs lit like hills")
    hv = AE.hillshade(dome, 25, 25, AE.Sun(315, 35, ve=5))
    check(np.std(hv) > np.std(h1), "40a: vertical exaggeration strengthens the relief")
    from .gui.colormaps import lut
    rgba = AE.shaded_grid(dome, lut("Viridis-style"), 25, 25, AE.Sun(315, 35, strength=0.6))
    flat = AE.shaded_grid(np.zeros_like(dome), lut("Viridis-style"), 25, 25, AE.Sun(315, 35),
                          values=dome)
    plain = AE.shaded_grid(dome, lut("Viridis-style"), 25, 25, AE.Sun(on=False))
    check(rgba.shape == dome.shape + (4,) and np.abs(flat.astype(int) - plain.astype(int)).max() <= 2
          and np.abs(rgba.astype(int) - plain.astype(int)).max() > 30,
          "40a: shading blends with the colour fill (flat ground keeps its colour; off = plain fill)")
    for az, el in ((0, 10), (90, 45), (225, 80)):
        x0, y0 = AE.sun_dial_xy(AE.Sun(az, el))
        b = AE.sun_from_dial(x0, y0)
        check(abs(((b.azimuth - az + 180) % 360) - 180) < 1e-6 and abs(b.elevation - el) < 1e-6,
              f"40a: sun dial round trip (az {az}, el {el})")

    class P:
        display = {}
    AE.Sun(120, 25, 3, 0.4, True).save(P, "livemap")
    s2 = AE.Sun.load(P, "livemap")
    check(s2.azimuth == 120 and s2.multi and AE.Sun.load(P, "maphub").azimuth == 315,
          "40a: sun settings per view saved in the project")
    lv = AE.light_vector(AE.Sun(90, 0.0001))
    check(np.allclose(lv, [1, 0, 0], atol=0.02), "40a: 3D light vector from the same sun (east, north, up)")
    from .maplayers import LayerStack
    st = LayerStack()
    st.layers = [{"id": "L1", "kind": "grid", "style": {}, "src": {}}, {"id": "L2", "kind": "wells", "style": {},
                                                                      "src": {}}]
    st.set_style = lambda lid, **kw: next(l for l in st.layers if l["id"] == lid)["style"].update(kw)
    n = AE.apply_sun(st, AE.Sun(45, 30, 2, 0.5))
    check(n == 1 and st.layers[0]["style"]["azimuth"] == 45 and st.layers[0]["style"]["hillshade"]
          and not st.layers[1]["style"], "40a: the sun lights every grid layer of a view (not wells)")
    # ---------------------------------------------------------------- 40b colour tables
    names = CT.names()
    check({"Seismic (red-white-blue)", "Grey", "Petrel-like", "Kingdom-style", "High contrast", "Viridis-style",
           "Cividis-style", "Magma-style"} <= set(names), f"40b: colour library ({len(names)} tables)")
    for nm in sorted(CT.MONOTONIC):
        check(CT.is_monotonic(CT.get(nm).lut()), f"40b: {nm} is monotonic in lightness (OKLab)")
    Lv = CT.lightness(CT.get("Viridis-style").lut())
    check(np.std(np.diff(Lv)) < 0.002, "40b: perceptual tables have even lightness steps")
    check(CT.label("Cividis-style").endswith("[CVD safe]") and not CT.label("Petrel-like").endswith("]"),
          "40b: colourblind-safe tables are marked")
    cv = CT.simulate_cvd(CT.get("Cividis-style").lut(), "deutan")
    check(CT.is_monotonic(cv), "40b: cividis-style stays monotonic for a deuteranope")
    from .gui import colormaps
    old = {"Grey": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))]}
    check(np.array_equal(colormaps.lut("Grey")[:, :3], np.repeat(np.linspace(0, 255, 256).round()[:, None], 3, 1))
          and old["Grey"] == CT.LIBRARY["Grey"], "40b: existing tables unchanged (seismic viewer, maps, 3D)")
    t = CT.get("Seismic (red-white-blue)")
    t.name = "My bar"
    k = t.add_point(0.6, (0, 200, 0))
    p = t.move_point(k, 0.99)
    check(p < 0.75 and t.points[k][0] == p, "40b: editor - dragging a control point stops at its neighbour")
    check(t.move_point(0, 0.3) == 0.0, "40b: end points stay at 0 and 1")
    t.flip, t.gamma = True, 2.0
    l2 = t.lut()
    check(np.array_equal(l2[0, :3], [140, 0, 0]) and not np.array_equal(l2[64], t.lut(apply_display=False)[::-1][64]),
          "40b: flip + gamma change the bar")
    data = np.random.default_rng(3).normal(0, 1, 5000)
    t.clip = 98.0
    lo, hi = t.levels(data)
    check(abs(lo + hi) < 1e-12 and abs(hi - np.percentile(np.abs(data), 98)) < 1e-9, "40b: symmetric clip")
    t.symmetric = False
    lo, hi = t.levels(data + 5)
    check(abs(lo - np.percentile(data + 5, 1)) < 1e-9, "40b: asymmetric percentile clip")
    proj = P()
    proj.display = {}
    CT.save_custom(proj, t)
    t2 = CT.get("My bar", proj)
    check(t2 is not None and t2.to_dict() == t.to_dict() and "My bar" in CT.names(proj),
          "40b: custom tables saved in the project round-trip")
    jp, cp = os.path.join(tmp, "bar.json"), os.path.join(tmp, "bar.csv")
    CT.export_json(t, jp)
    CT.export_csv(t, cp)
    tj, tc = CT.import_file(jp), CT.import_file(cp, "csv bar")
    check(tj.to_dict() == t.to_dict() and np.abs(tc.lut(apply_display=False).astype(int)
                                                  - t.lut().astype(int)).max() <= 3,
          "40b: export / import JSON exact; CSV = the bar as displayed (256 rows), back within 3/255")
    # ---------------------------------------------------------------- 40c co-render maths
    base = np.full((2, 3, 4), 200, np.uint8)
    base[..., 3] = 255
    attr = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    g = lut("Grey")
    out_a = AE.corender(base, attr, g, "alpha", 0.5, vmin=0, vmax=1)
    out_m = AE.corender(base, attr, g, "multiply", 1.0, vmin=0, vmax=1)
    out_t = AE.corender(base, attr, g, "transparency", 1.0, vmin=0, vmax=1)
    check(out_a[0, 0, 0] == 100 and out_a[0, 2, 0] == round(200 * 0.5 + 255 * 0.5),
          "40c: alpha blend = base (1 - a) + layer a")
    check(out_m[0, 0, 0] == 0 and out_m[0, 2, 0] == 200 and abs(int(out_m[0, 1, 0]) - round(200 * 128 / 255)) <= 1,
          "40c: multiply = base x layer (dark coherence darkens the seismic)")
    check(out_t[0, 0, 0] == 200 and out_t[0, 2, 0] == 255, "40c: transparency by value - low values transparent")
    out_ti = AE.corender(base, attr, g, "transparency", 1.0, vmin=0, vmax=1, invert_alpha=True)
    check(out_ti[0, 0, 0] == 0 and out_ti[0, 2, 0] == 200, "40c: inverted transparency (low coherence opaque)")
    # ---------------------------------------------------------------- 40d DPI
    for pct, dpi in ((100, 96), (150, 144), (200, 192)):
        sc = AE.dpi_scale(dpi)
        check(abs(sc - pct / 100) < 1e-9 and AE.scaled(2.0, sc) == 2.0 * pct / 100 and AE.font_pt(9, sc) == 9,
              f"40d: {pct} % scaling -> line widths x{sc:g}, fonts stay in points (Qt scales them)")
    env = AE.qt_env({})
    check(env["QT_SCALE_FACTOR_ROUNDING_POLICY"] == "PassThrough", "40d: per-monitor DPI env (PassThrough)")
    # ---------------------------------------------------------------- 40e depth cues
    dep = np.full((40, 40), np.inf)
    dep[10:30, 10:30] = 5.0
    dep[15:25, 15:25] = 3.0  # nearer box on top
    e = AE.eye_dome(dep)
    check(e[0, 0] == 1.0 and e[20, 20] == 1.0 and e[12, 20] == 1.0 and e[14, 20] < 0.9 and e[9, 20] < 0.9,
          "40e: eye-dome lighting darkens silhouettes and depth steps, leaves flat areas")
    rgb = np.full((1, 3, 3), 255.0)
    f = AE.fog(rgb, np.array([[0.0, 0.5, 1.0]]), (0, 0, 0), density=1.0)
    check(f[0, 0, 0] == 255 and f[0, 2, 0] < f[0, 1, 0] < 255, "40e: fog grows with distance")
    from . import scene3d as S3
    sc3 = S3.Scene()
    Xg, Yg = np.meshgrid(np.arange(6.0) * 100, np.arange(6.0) * 100)
    Tg = 500 + 0.4 * Xg
    m = S3.surface_mesh(Xg, Yg, Tg, lut("Grey"), vrange=(0, 1))
    lit = S3.shade_mesh(m, AE.Sun(270, 30))
    dark = S3.shade_mesh(m, AE.Sun(90, 30))
    check(lit.meta.get("lit") and lit.rgba[:, :3].mean() != dark.rgba[:, :3].mean()
          and S3.shade_mesh(m, None) is m, "40a/e: the sun lights 3D surfaces (none = unchanged)")
    fm = S3.fault_mesh(np.array([[[0, 0, 100], [100, 0, 100], [0, 0, 400]], [[100, 0, 100], [100, 0, 400],
                                                                            [0, 0, 400]]], float), (200, 0, 0), 0.55)
    ed = S3.edge_mesh(fm)
    check(ed.kind == "lines" and len(ed) == 8 and fm.rgba[0, 3] < 1, "40e: semi-transparent fault surface with "
          "crisp edges (4 boundary edges)")
    sc3.add("horizon", "H", [lit])
    sc3.add("fault", "F", [fm, ed])
    sc3.fit()
    p0, p1, p2 = (os.path.join(tmp, f"cue{k}.png") for k in range(3))
    sc3.render_png(p0, (240, 180))
    sc3.render_png(p1, (240, 180), cues={"edl": True})
    sc3.render_png(p2, (240, 180), cues={"fog": 0.8})
    A0, A1, A2 = _px(p0), _px(p1), _px(p2)
    check(A1.shape == A0.shape and A1.mean() < A0.mean() and np.abs(A2 - A0).mean() > 0.1,
          "40e: software renderer draws eye-dome lighting and fog (fallback path, no GPU)")
    sc3.cues["shadows"] = True
    sc3.render_png(p0, (240, 180))
    check(os.path.getsize(p0) > 0, "40e: shadows requested -> plain lighting fallback, no crash")
    st3 = sc3.state()
    check(st3["cues"]["shadows"] and "sun" in st3, "40e: depth-cue toggles saved with the 3D view")
    # ---------------------------------------------------------------- 40f themes
    for nm in ("dark", "light"):
        qss = AE.stylesheet(nm)
        pc = AE.plot_colours(nm)
        bgl = CT.lightness(np.array([AE.hex_rgb(pc["background"])]))[0]
        fgl = CT.lightness(np.array([AE.hex_rgb(pc["foreground"])]))[0]
        check("QWidget" in qss and abs(bgl - fgl) > 0.5 and (bgl < 0.35) == (nm == "dark"),
              f"40f: {nm} theme - stylesheet + plot colours with strong contrast")
    sp = os.path.join(tmp, "settings.json")
    AE.save_user_settings({"theme": "light"}, sp)
    AE.save_user_settings({"antialias": False}, sp)
    check(AE.load_user_settings(sp) == {"theme": "light", "antialias": False}, "40f: theme saved in user settings")
    from .glow import contrast_colour
    check(contrast_colour(against=AE.hex_rgb(AE.plot_colours("dark")["background"])) == (255, 255, 255)
          and contrast_colour(against=AE.hex_rgb(AE.plot_colours("light")["background"])) == (0, 0, 0),
          "40f: glow halo auto-contrast follows the theme background")
    # ---------------------------------------------------------------- 40g smooth motion
    clock = [0.0]
    frames = []
    tw = AE.Tween([0.0, 0.0], [10.0, 20.0], 0.5, clock=lambda: clock[0], on_frame=frames.append)
    clock[0] = 0.25
    v = tw.step()
    check(abs(v[0] - 5.0) < 1e-9 and not tw.done, "40g: eased tween half way at half time")
    clock[0] = 0.6
    v = tw.step()
    check(tw.done and np.allclose(v, [10, 20]) and tw.step() is None, "40g: tween finishes on time")
    off = AE.Tween([0.0], [3.0], AE.SPEEDS["Off"], on_frame=frames.append)
    check(off.done and np.allclose(frames[-1], [3.0]), "40g: speed Off = jump, no animation")
    t3 = AE.Tween([0.0], [1.0], 5.0, clock=lambda: clock[0])
    t3.interrupt()
    check(t3.done and t3.step() is None, "40g: any user input stops an animation")
    import time as _t
    t0 = _t.time()
    AE.Tween([0.0], [1.0], 30.0).step()
    check(_t.time() - t0 < 0.05, "40g: animations never block (one frame per timer tick)")
    cam = S3.Camera((0, 0, 0), 1000, 350, 30)
    tw = AE.fly_to_camera(cam, np.array([5000, 5000, -100.0]), np.array([6000, 6000, 0.0]), 0.4,
                          clock=lambda: clock[0])
    clock[0] += 1.0
    tw.step()
    check(np.allclose(cam.target, [5500, 5500, -50]) and abs(((cam.azimuth - 350 + 180) % 360) - 180) < 1e-6,
          "40g: 3D fly-to frames the object (same view direction)")
    zr = AE.zoom_range((0, 100, 0, 50), (10, 20, 10, 20))
    check(abs((zr[1] - zr[0]) / (zr[3] - zr[2]) - 2.0) < 1e-9 and zr[0] < 10 and zr[1] > 20,
          "40g: eased map zoom keeps the view aspect")
    # ---------------------------------------------------------------- 40i figure templates
    vol_x = np.arange(200.0)
    tt = np.arange(100) * 4.0
    data = np.random.default_rng(2).normal(0, 1, (200, 100))

    def draw(ax):
        ax.imshow(data.T, aspect="auto", cmap="gray", extent=[0, 200, 400, 0])
    fp = os.path.join(tmp, "fig.png")
    tpl = AE.default_template()
    tpl.update(title="Inline 425", author="Chris", project="F3", page="16:9 slide", dpi=60)
    r = AE.compose_figure(fp, draw=draw, template=tpl, legend=[("Top", (255, 220, 0)), ("Fault A", (230, 30, 30))],
                          colourbar=(lut("Grey"), -1, 1, "amp"), x_extent=5000.0)
    img = _px(fp)
    check(img.shape[1] == int(13.333 * 60) and {"title_block", "scale_bar", "legend", "colourbar"} <= set(r["boxes"])
          and r["boxes"]["scale_bar"][1] == 1000.0, "40i: figure has title block, scale bar (1 km of 5 km), "
          "colour bar and legend on a 16:9 page")
    fp2 = os.path.join(tmp, "fig.pdf")
    r2 = AE.compose_figure(fp2, image=img.astype(np.uint8), template=dict(tpl, page="A4 landscape", north_arrow=True),
                           north_deg=10.0, kind="map")
    check(os.path.getsize(fp2) > 1000 and "north_arrow" in r2["boxes"] and r2["page"] == AE.PAGES["A4 landscape"],
          "40i: any view image on an A4 page with a north arrow (PDF)")
    AE.save_template(P, "Slides", dict(tpl, page="Poster 36x24"))
    check(AE.load_template(P, "Slides")["page"] == "Poster 36x24" and AE.load_template(P, "x")["page"] == "16:9 slide",
          "40i: figure templates saved in the project")
    check(AE.nice_length(4300) == 500 and AE.nice_length(12000) == 2000, "40i: nice scale-bar lengths")
    # ---------------------------------------------------------------- 40j automatic colours
    seis = lut("Seismic (red-white-blue)")
    cols = AE.distinct_colours(12, avoid_lut=seis)
    md = AE.min_difference(cols)
    d_lut = min(float(CT.delta_e(np.asarray(c, float)[None], seis[::8, :3].astype(float)).min()) for c in cols)
    check(len(set(cols)) == 12 and md > 12 and d_lut > 5,
          f"40j: 12 automatic colours stay distinct (min dE {md:.1f}) and away from the seismic bar (dE {d_lut:.1f})")
    more = AE.distinct_colours(3, existing=cols, avoid_lut=seis)
    check(AE.min_difference(cols + more) > 8, "40j: new colours avoid the existing ones")
    check(AE.ics_colour("Miocene") == (255, 255, 0) and AE.ics_colour("Late Cretaceous") == (166, 216, 74)
          and AE.ics_colour(age_ma=10.0) == (255, 255, 0) and AE.ics_colour(age_ma=110.0) == (140, 205, 87)
          and AE.ics_colour("Unknown") is None, "40j: ICS chronostratigraphic colours by unit or age")
    proj.display = {}
    ca = AE.ColourAssigner(proj)
    c1 = ca.colour_for("horizon", "H1", lut=seis)
    ca.lock("horizon", "H1", (10, 20, 30))
    check(ca.colour_for("horizon", "H1", lut=seis) == (10, 20, 30) and ca.is_locked("horizon", "H1"),
          "40j: locked colours are kept")
    re = ca.reassign([("H1", (10, 20, 30)), ("H2", (255, 0, 0)), ("H3", (255, 0, 0))], "horizon", seis)
    check(re["H1"] == (10, 20, 30) and re["H2"] != re["H3"] and AE.min_difference(list(re.values())) > 10
          and ca.colour_for("horizon", "H4", unit="Pliocene") == (255, 255, 153),
          "40j: reassign keeps locked colours, recolours the rest distinctly; ICS when a unit is known")
    json.dumps(proj.display)  # everything stored is JSON-safe
    check(True, "40: display settings JSON-safe")


def run_gui(check, tmp):
    """Qt mock / real Qt offscreen: View > Theme, co-render, 3D sun / cues / fly-to / overlay, Live Map sun + eased
    zoom, figure exports, automatic colours, Well-Correlation style toggles are in its own guitest."""
    os.makedirs(tmp, exist_ok=True)
    from .gui import theme as TH
    from .gui.main_window import MainWindow
    from .test_scene3d import _project
    proj, truth, d = _project(tmp, "aes_gui")
    mw = MainWindow(proj.path)
    vol = mw.project.volume("Synth")
    sec = mw.section
    sec.source, sec.mode, sec.number = ("3D", "Synth"), "Inline", int(vol.inlines[10])
    sec.redraw()
    seen = []

    class W:
        def __init__(self):
            self.bg = None

        def setBackground(self, c):
            self.bg = c
            seen.append(c)

        def getPlotItem(self):
            raise AttributeError
    ws = [W(), W()]
    old = AE.settings_path
    AE.settings_path = lambda: os.path.join(tmp, "user_settings.json")
    try:
        n = TH.apply_theme(None, "light", bus=mw.mapbus, widgets=ws)
        check(n == 2 and all(w.bg == AE.plot_colours("light")["background"] for w in ws)
              and mw.mapbus.last("display", "theme")["theme"] == "light"
              and AE.load_user_settings()["theme"] == "light",
              "40f GUI: theme switch restyles every open view, tells the bus, saved in user settings")
        v = mw.viewer3d("Synth", force_software=True)
        mw.set_theme("light")
        check(v.scene.background[0] > 0.9 and mw.theme == "light", "40f GUI: View > Theme > Light - 3D view follows")
        mw.set_theme("dark")
        check(v.scene.background[0] < 0.2, "40f GUI: back to dark without a restart")
    finally:
        AE.settings_path = old
    # co-render on the section
    sec.set_blend({"overlay": "Envelope", "mode": "multiply", "opacity": 0.7, "lut": "Grey"})
    rg = sec._rgba
    check(rg is not None and rg.shape[:2] == np.shape(sec._data) and rg.dtype == np.uint8,
          "40c GUI: co-render (multiply) of an attribute over the amplitude on the inline")
    sec.set_blend(None)
    # 3D: sun, cues, overlay, fly-to
    v.set_sun(AE.Sun(270, 30))
    o = v.scene.find("horizon", "Top")
    check(o is not None and o.meshes[0].meta.get("lit"), "40a GUI: the sun lights the 3D horizons")
    lm = mw.live_map()
    lm.set_sun(AE.Sun(100, 20))
    check(abs(v.scene.sun.azimuth - 100) < 1e-9 and AE.Sun.load(mw.project, "livemap").azimuth == 100,
          "40a GUI: the Live Map sun (saved per view) drives the 3D surface lighting via the bus")
    cues = v.set_cues(edl=True, fog=0.3)
    check(cues["edl"] and cues["fog"] == 0.3, "40e GUI: 3D display panel toggles (EDL, fog)")
    v.set_cues(edl=False, fog=0.0)
    ov = v.set_overlay("Synth", "multiply", 0.5, "Grey")
    check(ov is not None and v.scene.find("slice", "Inline") is not None, "40c GUI: co-render on the 3D planes")
    v.set_overlay(None)
    lo, hi = v.scene.bounds()
    a = v.fly_to(lo, hi, speed="Off")
    check(a.done, "40g GUI: fly-to with speed Off jumps")
    a = v.fly_to(lo * 0.5, hi * 0.5, speed="Normal")
    v.animate_step(a.t0 + 0.2)
    mid = not a.done
    v.stop_animation()
    check(mid and a.done, "40g GUI: fly-to animates frame by frame and stops on user input")
    z = lm.zoom_to((0, 10, 0, 10), speed="Off")
    check(len(z) == 4, "40g GUI: Live Map eased zoom (speed Off jumps)")
    # figure exports
    fp = os.path.join(tmp, "page.png")
    r = sec.export_page(fp, {"dpi": 50, "title": "Inline test"})
    check(os.path.exists(fp) and {"title_block", "legend"} <= set(r["boxes"]),
          "40i GUI: section figure with title block + legend")
    fp3 = os.path.join(tmp, "page3d.png")
    r3 = v.export_page(fp3, {"dpi": 50})
    check(os.path.exists(fp3) and r3["kind"] == "3d", "40i GUI: 3D view figure")
    # colour-bar editor (40b)
    ed = mw.colorbar_editor()
    bar = ed.bar
    bar.width = lambda: 340
    bar.height = lambda: 60
    ed.load("Seismic (red-white-blue)")
    k = bar.press(10 + 0.25 * 320)
    p = bar.move(10 + 0.40 * 320)
    bar.release()
    n0 = len(ed.table.points)
    bar.press(10 + 0.9 * 320, double=True)
    n1 = len(ed.table.points)
    bar.press(10 + 0.9 * 320, button="right")
    nm = ed.save("Chris bar")
    from . import colortables as CT
    from .gui import colormaps as CM
    check(k == 1 and abs(p - 0.40) < 1e-6 and n1 == n0 + 1 and len(ed.table.points) == n0
          and CT.get("Chris bar", mw.project) is not None and "Chris bar" in CM.NAMES,
          "40b GUI: colour-bar editor - drag a control point, add (double-click), remove (right-click), save to the "
          "project -> listed in every colour picker")
    ep = os.path.join(tmp, "chris.json")
    ed.export(ep)
    t2 = ed.import_(ep)
    check(t2.to_dict()["points"] == CT.get("Chris bar", mw.project).to_dict()["points"],
          "40b GUI: editor export / import")
    # automatic colours
    h = mw.horizon_dock.new_horizon("Auto A")
    h2 = mw.horizon_dock.new_horizon("Auto B")
    cols = [tuple(x.color) for x in mw.project.horizons.items]
    check(h is not None and AE.min_difference(cols) > 8, f"40j GUI: new horizons get distinct automatic colours "
          f"(min dE {AE.min_difference(cols):.1f})")
    n = mw.reassign_colours()
    check(n >= 0 and AE.min_difference([tuple(x.color) for x in mw.project.horizons.items]) > 8,
          "40j GUI: Display > Reassign colours keeps them distinct")
    for w in (lm, v):
        try:
            w.close()
        except Exception:
            pass
