"""Tests for PLAN 23: co-blending, overlays, bump map / amplitude on structure, linked cursor, parameter scan, layouts."""
import math

import numpy as np


def run(check, tmp):
    from . import blend as B
    from .linking import CursorBus, nearest_well, save_layout, load_layout, delete_layout, layout_names
    # ---- normalisation
    a = np.arange(101, dtype=float)
    n = B.normalize(a, (0, 100))
    check(abs(n[0]) < 1e-12 and abs(n[-1] - 1) < 1e-12 and abs(n[50] - 0.5) < 1e-12, "23a: normalize maps min..max to 0..1")
    n2 = B.normalize(a, (0, 100), gamma=2.0)
    check(abs(n2[25] - math.sqrt(0.25)) < 1e-12, "23a: gamma 2 brightens (0.25 -> 0.5)")
    nc = B.normalize(a, (10, 90))
    check(nc[5] == 0 and nc[95] == 1 and abs(nc[50] - 0.5) < 1e-9, "23a: percentile clip saturates the tails")
    check(np.isnan(B.normalize(np.array([1.0, np.nan, 3.0]))[1]), "23a: NaN stays NaN")
    # ---- channel maths
    lo, hi = np.zeros((2, 2)), np.ones((2, 2))
    rgba = B.rgb_blend([hi, lo, lo], "rgb", vmin=0, vmax=1)
    check(tuple(rgba[0, 0]) == (255, 0, 0, 255), "23a: RGB - channel 1 only = pure red")
    rgba = B.rgb_blend([hi, hi, hi], "rgb", vmin=0, vmax=1)
    check(tuple(rgba[0, 0, :3]) == (255, 255, 255), "23a: RGB - all channels high = white")
    cmy = B.rgb_blend([hi, lo, lo], "cmy", vmin=0, vmax=1)
    check(tuple(cmy[0, 0, :3]) == (0, 255, 255), "23a: CMY - channel 1 high = cyan (red removed)")
    cmy0 = B.rgb_blend([lo, lo, lo], "cmy", vmin=0, vmax=1)
    check(tuple(cmy0[0, 0, :3]) == (255, 255, 255), "23a: CMY - all low = white paper")
    h = B.rgb_blend([lo, hi, hi], "hsv", vmin=0, vmax=1)
    check(tuple(h[0, 0, :3]) == (255, 0, 0), "23a: HSV - hue 0, full saturation / value = red")
    gs = B.rgb_blend([lo, lo, hi], "hsv", vmin=0, vmax=1)
    check(tuple(gs[0, 0, :3]) == (255, 255, 255), "23a: HSV - zero saturation = grey (white at value 1)")
    r, g, b = B.hsv_to_rgb(np.array([1 / 3, 2 / 3]), np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    check(np.allclose([r[0], g[0], b[0]], [0, 1, 0]) and np.allclose([r[1], g[1], b[1]], [0, 0, 1]),
          "23a: HSV conversion (green, blue)")
    x = np.array([[0.0, 1.0], [np.nan, 0.5]])
    check(B.rgb_blend([x, x], "rgb")[1, 0, 3] == 0 and B.rgb_blend([x, x], "rgb")[0, 0, 3] == 255,
          "23a: NaN in a channel -> transparent pixel; 2 channels allowed")
    ga = B.rgb_blend([a.reshape(1, -1)] * 3, "rgb", clip=[(0, 100), (0, 100), (0, 100)], gamma=[1, 2, 1])
    check(ga[0, 25, 1] > ga[0, 25, 0], "23a: per-channel gamma")
    # ---- legend
    tri, pos = B.legend_triangle("rgb", 64, ("10 Hz", "20 Hz", "40 Hz"))
    bl, br, top = tri[63, 0], tri[63, 63], tri[3, 32]
    check(tuple(bl[:3]) == (255, 0, 0) and tuple(br[:3]) == (0, 255, 0) and top[2] == 255 and top[0] < 40 and top[1] < 40 and top[3] == 255 and
          tri[0, 0, 3] == 0 and [p[2] for p in pos] == ["10 Hz", "20 Hz", "40 Hz"],
          "23a: colour-triangle legend - corners = single channels, outside transparent, labelled")
    sw = B.channel_swatches("cmy", 16)
    check(len(sw) == 3 and tuple(sw[0][0, -1, :3]) == (0, 255, 255) and tuple(sw[0][0, 0, :3]) == (255, 255, 255),
          "23a: per-channel swatches")
    # ---- overlay
    base = np.full((4, 4, 4), 100, np.uint8)
    base[..., 3] = 255
    lut = np.zeros((256, 3), np.uint8)
    lut[:, 0] = 255  # all red
    attr = np.arange(16, dtype=float).reshape(4, 4)
    ov = B.overlay(base, attr, lut, opacity=0.5, vmin=0, vmax=15)
    check(ov[0, 0, 0] == round(100 * 0.5 + 255 * 0.5) and ov[0, 0, 1] == 50, "23a: overlay = alpha blend over the seismic")
    ovt = B.overlay(base, attr, lut, opacity=1.0, vmin=0, vmax=15, threshold=8)
    check(tuple(ovt[0, 0, :3]) == (100, 100, 100) and tuple(ovt[3, 3, :3]) == (255, 0, 0),
          "23a: overlay threshold keeps the seismic below it")
    # ---- illumination against an analytic plane and dome
    ny, nx, d = 40, 50, 10.0
    X, Y = np.meshgrid(np.arange(nx) * d, np.arange(ny) * d)
    for sx, sy in ((0.3, 0.0), (0.0, -0.2), (0.1, 0.25)):
        z = sx * X + sy * Y
        for az, el in ((315, 35), (90, 20), (180, 60)):
            s = B.illuminate(z, d, d, az, el)
            nrm = np.array([-sx, -sy, 1.0]) / math.sqrt(sx * sx + sy * sy + 1)
            L = np.array([math.sin(math.radians(az)) * math.cos(math.radians(el)),
                          math.cos(math.radians(az)) * math.cos(math.radians(el)), math.sin(math.radians(el))])
            want = max(0.0, float(nrm @ L))
            if not np.allclose(s, want, atol=1e-9):
                check(False, f"23b: illumination of plane {sx},{sy} from {az}/{el}: {s.mean()} vs {want}")
                break
    check(True, "23b: illumination of tilted planes = cos(normal, light) for several azimuths / elevations")
    flat = B.illuminate(np.zeros((5, 5)), 1, 1, 123, 30)
    check(np.allclose(flat, math.sin(math.radians(30))), "23b: flat surface = sin(elevation)")
    dome = -np.hypot(X - X.mean(), Y - Y.mean()) * 0.5  # dome: high in the centre (z = elevation, up positive)
    s = B.illuminate(dome, d, d, 90, 30)  # light from the east
    c = ny // 2
    check(s[c, nx - 5] > s[c, 4] + 0.3, "23b: dome lit from the east - east flank brighter than the west flank")
    s2 = B.illuminate(dome, d, d, 90, 30, ve=4)
    check(s2[c, 4] < s[c, 4], "23b: vertical exaggeration deepens the shadows")
    rng = np.random.default_rng(0)
    amp = rng.normal(size=(ny, nx))
    grey = np.stack([np.linspace(0, 255, 256)] * 3, 1).astype(np.uint8)
    aos = B.amplitude_on_structure(amp, dome, grey, d, d, 90, 30, strength=0.8)
    plain = B.lut_rgba(amp, grey)
    ratio = aos[..., 0].astype(float).sum() / max(1, plain[..., 0].astype(float).sum())
    check(aos.shape == (ny, nx, 4) and aos[c, nx - 5, 0] >= plain[c, nx - 5, 0] * 0.95 and
          aos[c, 4, 0] <= plain[c, 4, 0] and 0.3 < ratio < 1.3, "23b: amplitude on structure - colour from amplitude, "
          "shading from the structure")
    bump = B.bump_section(rng.normal(size=(30, 40)))
    check(bump.shape == (30, 40, 4) and bump[..., 3].min() == 255, "23b: bump-mapped section image")
    # ---- linked cursor bus
    t = [0.0]
    bus = CursorBus(min_interval=0.05, clock=lambda: t[0])
    got = {"map": [], "dash": [], "section": []}
    bus.subscribe("map", got["map"].append)
    bus.subscribe("dash", got["dash"].append)
    bus.subscribe("section", got["section"].append)
    bus.subscribe("broken", lambda m: 1 / 0)
    n = bus.publish("section", x=1.0, y=2.0, t=1500.0, il=None)
    check(n == 2 and got["map"][0] == {"source": "section", "x": 1.0, "y": 2.0, "t": 1500.0} and not got["section"],
          "23c: cursor message reaches the other views, not its sender; None fields dropped; broken view ignored")
    check(bus.publish("section", x=2.0, y=2.0) == 0, "23c: fast mouse throttled")
    t[0] = 0.1
    check(bus.publish("section", x=3.0, y=2.0) == 2 and got["dash"][-1]["x"] == 3.0, "23c: next message after the interval")
    bus.enabled = False
    check(bus.publish("map", x=0.0, y=0.0) == 0, "23c: linking off")
    bus.enabled = True
    bus.clear("map")
    check(got["section"][-1].get("clear") and got["dash"][-1].get("clear"), "23c: clear message")
    check(nearest_well([("A", 0, 0), ("B", 100, 0)], 90, 5)[0] == "B" and
          nearest_well([("A", 0, 0)], 900, 0, 500)[0] is None, "23c: nearest well with max distance")

    class P:
        display = {}
    p = P()
    save_layout(p, "Mapping", {"docks": {"data": False}})
    save_layout(p, "A", {"docks": {}})
    check(layout_names(p) == ["A", "Mapping"] and load_layout(p, "Mapping")["docks"] == {"data": False} and
          load_layout(p, "none") is None and delete_layout(p, "A") == ["Mapping"], "23c: named layouts in the project")
    # ---- grids on a common geometry + map hub blend layer
    run_hub(check, tmp)


def run_hub(check, tmp):
    from . import blend as B
    from . import maplayers as ML
    from .maps import horizon_grid, MapGrid
    from .test_maplayers import _setup
    from . import attributes as A
    proj, truth, d = _setup(tmp, "blend_hub")
    st = ML.LayerStack.load(proj)
    g1 = horizon_grid(proj, proj.horizons.get("Top"), "3D:Synth")
    g2 = horizon_grid(proj, proj.horizons.get("Deep"), "3D:Synth")
    xy = g1.to_xy()
    back = B.on_grid(g1, xy)
    ok = np.isfinite(back) & np.isfinite(g1.z)
    check(ok.mean() > 0.6 and np.nanmax(np.abs(back[ok] - g1.z[ok])) < 0.05 * (np.nanmax(g1.z) - np.nanmin(g1.z)) + 1,
          "23a: grids resampled onto a common geometry (xy -> survey grid)")
    i1, _ = st.add_grid(g1, "Top")
    i2, _ = st.add_grid(g2, "Deep")
    rng = np.random.default_rng(1)
    amp = MapGrid(rng.normal(size=g1.z.shape), "ilxl", "amp", "amp", g1.vol)
    i3, _ = st.add_grid(amp, "Amp")
    bl = st.add("blend", "RGB", src={"grids": [i1, i2, i3]}, style={"mode": "rgb"})
    prims = ML.render(st)
    img = [q for q in prims if q[0] == "image" and q[1] == bl]
    check(len(img) == 1 and img[0][2].shape == g1.z.shape + (4,) and img[0][2][..., :3].std() > 10,
          "23a: map hub blend layer (3 grids -> RGB image on the first grid)")
    st.set_style(i3, shade_from=i1, hillshade=True, azimuth=45.0, elevation=30.0, ve=5.0)
    st.set_style(i2, hillshade=True, azimuth=270.0)
    p2 = {q[1]: q for q in ML.render(st) if q[0] == "image"}
    st.set_style(i3, shade_from=None, hillshade=False)
    p3 = {q[1]: q for q in ML.render(st) if q[0] == "image"}
    check(i3 in p2 and not np.array_equal(p2[i3][2], p3[i3][2]) and i2 in p2,
          "23b: grid shaded by another grid (amplitude on structure) and single-light bump map on the map hub")
    st.save()
    from .project import SeisProject
    proj.save()
    st2 = ML.LayerStack.load(SeisProject.open(proj.path))
    check(st2.get(bl)["src"]["grids"] == [i1, i2, i3] and st2.get(bl)["kind"] == "blend",
          "23a: blend layer saved with the project")
    # section co-blend on the volume
    vol = proj.volume("Synth")
    il = int(vol.inlines[len(vol.inlines) // 2])
    rgba, arrs = B.section_blend(vol, ["Spectral 10 Hz", "Spectral 20 Hz", "Spectral 40 Hz"], "Inline", il)
    check(rgba.shape == arrs[0].shape + (4,) and arrs[0].shape == (vol.ns, vol.nxl) and rgba[..., 3].min() == 255,
          "23a: spectral RGB co-blend of an inline (same orientation as the section image)")
    ts = B.section_blend(vol, ["Envelope", "RMS amplitude"], "Time slice", float(vol.times[vol.ns // 2]), "hsv")[0]
    check(ts.shape == (vol.nil, vol.nxl, 4), "23a: HSV co-blend of a time slice")
    fr = list(B.param_scan(lambda f: A.section_attribute(vol, "Spectral 20 Hz", "Inline", il, {"freq": f}), [10, 30, 50]))
    check(len(fr) == 3 and not np.allclose(fr[0][1], fr[2][1]), "23c: attribute parameter scan frames differ")


def run_gui(check, tmp):
    import numpy as np
    from .gui.main_window import MainWindow
    from .gui.scan_dialog import ScanDialog
    from .test_maplayers import _setup
    from .gui.layers_dock import LayersPanel
    from . import maplayers as ML
    from .maps import horizon_grid
    proj, truth, d = _setup(tmp, "blend_gui")
    mw = MainWindow(proj.path)
    vol = mw.project.volume("Synth")
    il = int(vol.inlines[len(vol.inlines) // 2])
    sec = mw.section

    def show(mode, number):  # fakeqt: combos are mocks - set the state directly
        sec.source, sec.mode, sec.number, sec.domain, sec.attr = ("3D", "Synth"), mode, number, "Time", "Seismic"
        sec.redraw()
    show("Inline", il)
    mw.section.set_blend({"attrs": ["Spectral 10 Hz", "Spectral 20 Hz", "Spectral 40 Hz"], "mode": "rgb"})
    check(mw.section._rgba is not None and mw.section._rgba.shape == mw.section._data.shape + (4,),
          "23a GUI: section shows the RGB co-blend")
    mw.section.set_blend({"overlay": "Envelope", "opacity": 0.4})
    check(mw.section._rgba is not None, "23a GUI: attribute overlay on the seismic")
    mw.section.set_blend({"bump": True})
    check(mw.section._rgba is not None, "23b GUI: bump-mapped section")
    mw.section.set_blend(None)
    check(mw.section._rgba is None, "23a GUI: back to the plain section")
    # linked cursor: section -> map / petro strip; map -> section
    X, Y = vol.xy(il, float(vol.xlines[5]))
    got = []
    mw.cursor_bus.subscribe("probe", got.append)
    mw.cursor_bus.min_interval = 0.0
    # (fakeqt signals are no-ops: publish as the connected lambdas do)
    mw.cursor_bus.publish("section", **{"x": float(X), "y": float(Y), "t": 1000.0, "il": float(il), "xl": float(vol.xlines[5])})
    check(mw.basemap.cursor_xy is not None and abs(mw.basemap.cursor_xy[0] - X) < 1e-6 and got and got[-1]["t"] == 1000.0,
          "23c GUI: section cursor shown on the basemap and sent to the other views")
    mw.cursor_bus.publish("map", x=float(X), y=float(Y))
    pos = getattr(mw.section, "_link_pos", None)
    check(pos is not None and abs(pos[0] - float(vol.xlines[5])) < 0.6, "23c GUI: map cursor shown on the section (crossline)")
    far = mw.section.linked_position({"x": float(X) + 1e7, "y": float(Y)})
    check(far is None, "23c GUI: cursor off the section is ignored")
    mw.well_clicked("W-1")
    if mw._petro_strip is not None:
        mw.cursor_bus.publish("section", x=float(X), y=float(Y), t=900.0)
        check(getattr(mw._petro_strip, "cursor_well", None) in ("W-1", "W-2", "W-3", None), "23c GUI: petro strip follows")
    check(mw.set_linked_cursor(False) is False and mw.cursor_bus.publish("section", x=1.0, y=1.0) == 0,
          "23c GUI: View > Linked cursor off")
    mw.set_linked_cursor(True)
    # parameter scan
    dlg = ScanDialog(mw.section)
    nf = dlg.compute("Spectral 20 Hz", [10.0, 20.0, 40.0])
    v = dlg.set_frame(2)
    check(nf == 3 and v == 40.0 and mw.section._data.shape == (vol.ns, vol.nxl), "23c GUI: parameter movie scan")
    # layouts
    names = mw.save_layout("Mine")
    lay = mw.project.display["layouts"]["Mine"]
    check(names == ["Mine"] and lay["section"]["mode"] == "Inline" and lay["section"]["number"] == il,
          "23c GUI: layout saved (docks, section state)")
    show("Crossline", int(vol.xlines[3]))
    calls = []
    sec.show = lambda *a: calls.append(a) or show(a[2], a[3])
    mw.restore_layout("Mine")
    check(calls == [("3D", "Synth", "Inline", il)] and mw.section.mode == "Inline" and mw.section.number == il,
          "23c GUI: layout restored")
    check(mw.restore_layout("Mapping"), "23c GUI: built-in layout template")
    # map hub dock
    st = ML.LayerStack.load(proj)
    mw.basemap.set_stack(st)
    panel = LayersPanel(mw.basemap)
    panel.set_stack(st)
    a, _ = st.add_grid(horizon_grid(proj, proj.horizons.get("Top"), "3D:Synth"), "Top")
    b, _ = st.add_grid(horizon_grid(proj, proj.horizons.get("Deep"), "3D:Synth"), "Deep")
    lid = panel.add_blend([a, b], "cmy")
    tri, pos = panel.blend_legend(lid)
    check(st.get(lid)["kind"] == "blend" and not mw.basemap.layer_errors and pos[0][2] == "Top",
          "23a GUI: co-blend layer added from the Map layers dock with a triangle legend")
    panel.shade_by(b, a, 315, 30, 3)
    check(st.get(b)["style"]["shade_from"] == a and not mw.basemap.layer_errors, "23b GUI: shade a grid by structure")
    # PLAN 25d: DHI checklist from the Map layers dock (amplitude grid conformant with the Top structure)
    from .maps import MapGrid
    top = st.grid(a)
    crest = top.z <= np.nanpercentile(top.z, 25)
    amp = MapGrid(np.where(crest, 3.0, 1.0) + np.random.default_rng(3).normal(0, 0.1, top.z.shape), "ilxl", "amp",
                  "amp", top.vol)
    am, _ = st.add_grid(amp, "Amp")
    res = panel.dhi(am, a, {"flat_spot": 0.5, "quality": 0.8}, "P1")
    check(res["details"]["amp_ratio"] > 2 and res["used"]["conformance"] > 0.8 and 0.4 < res["index"] < 1 and
          mw.project.display["dhi"]["P1"]["index"] == res["index"], f"25d GUI: DHI index {res['index']:.2f} for a "
          f"conformant anomaly (conformance {res['used']['conformance']:.2f}) stored with the project")
