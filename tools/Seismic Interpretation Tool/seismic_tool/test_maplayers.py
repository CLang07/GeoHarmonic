"""Tests for the map hub layers (PLAN 21a / 21b petro posting)."""
import json
import os

import numpy as np


def _petro_well_project(folder, wells):
    """Minimal .wcproj.json with well locations + <wcproj>_petro/summary.csv (what the Well Correlation Tool writes)."""
    wp = os.path.join(folder, "wells.wcproj.json")
    with open(wp, "w") as f:
        json.dump({"app": "well_correlation", "wells": [{"name": n, "x": x, "y": y, "las": []} for n, x, y, _ in wells]}, f)
    d = os.path.join(folder, "wells_petro")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "summary.csv"), "w") as f:
        f.write("Well,Zone,SoPhiH,PHIE_res\n")
        for n, _, _, vals in wells:
            for zone, v in vals.items():
                f.write(f"{n},{zone},{v},{0.1 + v / 100}\n")
    return wp


def _setup(tmp, name):
    from .synth import make_project
    d = os.path.join(tmp, name)
    os.makedirs(d)
    proj, truth = make_project(d, nil=20, nxl=25, ns=200, noise=0.0)
    for nm, k in (("Top", "3D:Synth"), ("Deep", "deep")):
        h = proj.horizons.new(nm)
        h.data["3D:Synth"] = truth[k].astype(np.float32)
    vol = proj.volume("Synth")
    xs, ys = vol.outline()
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    wells = [("W-1", cx, cy, {"A": 3.0, "B": 1.0}), ("W-2", cx + 200, cy, {"A": 5.0}),
             ("W-3", cx, cy + 200, {"A": 1.0, "B": 0.5})]
    proj.well_project = _petro_well_project(d, wells)
    proj.save()
    return proj, truth, d


def run(check, tmp):
    from . import maplayers as ML
    from .maps import horizon_grid
    from .project import SeisProject
    proj, truth, d = _setup(tmp, "layers")
    st = ML.LayerStack.load(proj)
    kinds = [l["kind"] for l in st.ordered()]
    check(kinds == ["outline3d", "lines2d", "wells"], f"21a: default stack = legacy basemap content ({kinds})")
    g = horizon_grid(proj, proj.horizons.get("Top"), "3D:Synth")
    gid, cid = st.add_grid(g, "Top time", interval=10.0)
    check(os.path.exists(os.path.join(ML.maps_dir(proj), st.get(gid)["src"]["file"])), "21a: grid stored as npz in "
          "<project>_seismic/maps (not in the JSON)")
    pid = st.add("polygons", "Prospects", src={"polygons": [{"name": "P1", "x": [0, 500, 500], "y": [0, 0, 500]}]})
    tid = st.add("text", "Notes", src={"items": [{"x": 10.0, "y": 20.0, "text": "hello"}]})
    st.set_opacity(gid, 0.5)
    st.set_style(gid, lut="Grey")
    st.set_visible(tid, False)
    order = st.move(pid, -1)
    st.save()
    proj.save()
    p2 = SeisProject.open(proj.path)
    st2 = ML.LayerStack.load(p2)
    same = [(l["id"], l["kind"], l["visible"], l["opacity"], l["z"], l["style"].get("lut")) for l in st2.ordered()] == \
        [(l["id"], l["kind"], l["visible"], l["opacity"], l["z"], l["style"].get("lut")) for l in st.ordered()]
    g2 = st2.grid(gid)
    check(same and np.allclose(np.nan_to_num(g2.z), np.nan_to_num(g.z), atol=1e-3) and g2.kind == "ilxl",
          "21a: layer state (order, visibility, opacity, style) + grid restored after save / open")
    check([l["id"] for l in st2.ordered()] == order, "21a: move changes the drawing order")
    prims = ML.render(st2)
    kinds_drawn = {q[1] for q in prims}
    check(tid not in kinds_drawn and gid in kinds_drawn and pid in kinds_drawn, "21a: hidden layers are not drawn")
    img = next(q for q in prims if q[0] == "image" and q[1] == gid)
    check(img[2][..., 3].max() == 127 and img[2].shape[:2] == g.z.shape, "21a: grid opacity in the image alpha")
    curves = [q for q in prims if q[0] == "curve" and q[1] == cid]
    widths = {q[5] for q in curves}
    check(len(curves) >= 3 and widths <= {1.0, 2.0} and (len(curves) < 5 or 2.0 in widths),
          f"21a: contours at the interval with bold index contours ({len(curves)} levels)")
    lv = ML.contour_levels(g.z, 10.0)
    check(all(abs(v / 10 - round(v / 10)) < 1e-9 for v in lv), "21a: contour levels on the interval")
    leg = st2.legend()
    check([e["id"] for e in leg] == [l["id"] for l in reversed(st2.ordered()) if l["visible"]],
          "21a: legend lists exactly the visible layers, top first")
    cb = st2.colorbar(gid)
    check(cb["reverse"] and cb["unit"] == "ms" and cb["lut"] == "Grey", "21a: colour bar (time grids reversed: shallow red)")
    st2.remove(gid)
    check(all(l["src"].get("grid") != gid for l in st2.layers), "21a: removing a grid removes its contours")
    hid = st2.add("horizon", "Deep map", src={"horizon": "Deep"})
    hg = st2.grid(hid)
    check(hg is not None and np.allclose(hg.z, truth["deep"], equal_nan=True), "21a: horizon time-map layer")
    # 21b-style petro posting
    vals = ML.petro_values(proj, "SoPhiH", "A")
    check(sorted((w, v) for w, _, _, v in vals) == [("W-1", 3.0), ("W-2", 5.0), ("W-3", 1.0)], "21b: zone value per well "
          "from the petro summary.csv")
    allz = {w: v for w, _, _, v in ML.petro_values(proj, "SoPhiH")}
    avg = {w: v for w, _, _, v in ML.petro_values(proj, "PHIE_res")}
    check(allz["W-1"] == 4.0 and abs(avg["W-1"] - (0.13 + 0.11) / 2) < 1e-9, "21b: all zones = sum of thickness-type "
          "values, mean of averages")
    pl = st2.add("petro", "SoPhiH A", src={"property": "SoPhiH", "zone": "A"})
    pts = [q for q in ML.render(st2) if q[0] == "points" and q[1] == pl][0]
    sizes = np.asarray(pts[5])
    names = [w for w, _, _, _ in ML.petro_values(proj, "SoPhiH", "A")]
    check(len(pts[2]) == 3 and names[int(np.argmax(sizes))] == "W-2", "21b: petro bubbles drawn, biggest = largest value")
    check(st2.colorbar(pl)["vmin"] == 1.0 and st2.colorbar(pl)["vmax"] == 5.0, "21b: petro colour bar range")


def run_gui(check, tmp):
    from .gui.basemap import Basemap
    from .gui.layers_dock import LayersPanel
    from .gui.main_window import MainWindow
    from . import maplayers as ML
    from .maps import horizon_grid
    proj, truth, d = _setup(tmp, "layers_gui")
    bm = Basemap()
    bm.project = proj
    st = ML.LayerStack.load(proj)
    bm.set_stack(st)
    panel = LayersPanel(bm)
    panel.set_stack(st)
    check(len(panel._ids) == 3 and len(bm._layer_items) > 0, "21a GUI: basemap drawn from the layer stack")
    check(set(bm._lines) == {"L-A", "L-B"}, "21a GUI: 2D lines still clickable in layer mode")
    bm.show_grid(horizon_grid(proj, proj.horizons.get("Top"), "3D:Synth"))
    lid = panel.add_current_grid("Top time")
    check(lid is not None and proj.display["map_layers"]["layers"][-2]["id"] == lid, "21a GUI: current basemap grid added as a "
          "layer and saved in the project display")
    panel.select(lid)
    panel.set_opacity(0.3)
    panel.set_style(lut="Grey", interval=20.0)
    con = next(l for l in st.layers if l["kind"] == "contours" and l["src"]["grid"] == lid)
    check(st.get(lid)["opacity"] == 0.3 and st.get(lid)["style"]["lut"] == "Grey" and con["style"]["interval"] == 20.0,
          "21a GUI: opacity, colour ramp, contour interval")
    before = [l["id"] for l in st.ordered()]
    panel.move(+1)
    check([l["id"] for l in st.ordered()] != before, "21a GUI: Up moves the layer")
    pid = panel.add_polygon("P1", [0, 100, 100], [0, 0, 100])
    panel.add_text(5, 5, "note")
    pet = panel.add_petro("SoPhiH", "A")
    check(st.get(pid)["src"]["polygons"][0]["name"] == "P1" and st.get(pet)["kind"] == "petro", "21a/b GUI: polygon, text, petro layers")
    panel.set_visible(pid, False)
    n_vis = len(bm._layer_items)
    panel.set_visible(pid, True)
    check(len(bm._layer_items) > n_vis, "21a GUI: visibility toggles redraw")
    panel.apply_preset("Print")
    out = panel.export_map(os.path.join(d, "panel_map.png"), title="t", dpi=60)
    check(proj.display.get("map_style") == "Print" and os.path.exists(out["path"]), "21f GUI: style preset + map export from "
          "the Map layers dock")
    panel.remove(pid)
    check(all(l["id"] != pid for l in st.layers) and not bm.layer_errors, "21a GUI: remove; no layer errors")
    mw = MainWindow(proj.path)
    check(mw.basemap.stack is not None and mw.layers_panel.stack is mw.basemap.stack and len(mw.layers_panel._ids) >= 3,
          "21a GUI: main window opens the project with the saved map layers")


def _mm_summary(folder):
    d = os.path.join(folder, "wells_petro")
    with open(os.path.join(d, "mm_summary.csv"), "w") as f:
        f.write("Well,Zone,Gross_TVD,V_quartz,V_illite,V_water,V_gas\n")
        f.write("W-1,A,10,0.6,0.2,0.1,0.1\nW-1,B,30,0.4,0.4,0.2,0.0\nW-2,A,20,0.7,0.1,0.05,0.15\n")


def run_hub(check, tmp):
    """21b: mineral pies, hub request, lasso selection, cross-plot of zone values, petro strip data."""
    from . import maphub as MH
    from . import maplayers as ML
    proj, truth, d = _setup(tmp, "hub")
    _mm_summary(d)
    pies = {w: fr for w, _, _, fr in MH.petro_pies(proj, "A")}
    check(abs(sum(pies["W-1"].values()) - 1) < 1e-12 and abs(pies["W-1"]["quartz"] - 0.6) < 1e-12, "21b: mineral pie "
          "fractions from mm_summary.csv")
    allz = {w: fr for w, _, _, fr in MH.petro_pies(proj)}
    check(abs(allz["W-1"]["quartz"] - (10 * 0.6 + 30 * 0.4) / 40) < 1e-12, "21b: all-zones pie weighted by gross thickness")
    wedges = MH.pie_polygons(0.0, 0.0, {"a": 0.25, "b": 0.75}, 10.0, n=400)

    def area(x, y):
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    a = area(*wedges[0][1:])
    check(abs(a - 0.25 * np.pi * 100) < 0.5, f"21b: pie wedge area proportional to the fraction ({a:.1f} vs {25 * np.pi:.1f})")
    st = ML.LayerStack.load(proj)
    lid = st.add("petro", "pies", src={"zone": "A"}, style={"mode": "pies"})
    polys = [q for q in ML.render(st) if q[0] == "polygon" and q[1] == lid]
    check(len(polys) == 8 and st.legend()[0]["symbol"] == "pie" and len(st.legend()[0]["components"]) == 4,
          "21b: pies rendered (4 components x 2 wells) with a component legend")
    req = {"property": "PHIE_res", "zone": "A", "wells": ["W-1", "W-2"]}
    l1 = MH.apply_hub_request(st, req)
    l2 = MH.apply_hub_request(st, dict(req, property="SoPhiH"))
    check(l1 == l2 and st.get(l1)["src"]["property"] == "SoPhiH" and proj.display["map_layers"], "21b: map-hub request "
          "adds one petro layer and updates it")
    sq = ([0, 10, 10, 0], [0, 0, 10, 10])
    inside = MH.point_in_polygon([5, 15, 5, 10.0001], [5, 5, -1, 5], *sq)
    check(list(inside) == [True, False, False, False], "21b: point in polygon")
    vol = proj.volume("Synth")
    xs, ys = vol.outline()
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    sel = MH.wells_in_polygon(proj, [cx - 50, cx + 300, cx + 300, cx - 50], [cy - 50, cy - 50, cy + 50, cy + 50])
    check(sorted(sel) == ["W-1", "W-2"], f"21b: wells inside a lasso polygon {sel}")
    xp = MH.petro_crossplot_data(proj, "SoPhiH", "PHIE_res", "A", selected=sel)
    check(xp["n"] == 3 and abs(xp["r"] - 1) < 1e-9 and xp["selected"].sum() == 2, "21b: linked cross-plot (zone values, "
          "selection carried over)")
    from .maps import horizon_grid
    g = horizon_grid(proj, proj.horizons.get("Top"), "3D:Synth")
    xg = MH.petro_crossplot_data(proj, "grid", "SoPhiH", "A", grid=g)
    check(xg["n"] == 3 and np.all(np.isfinite(xg["x"])), "21b: map grid sampled at the wells vs a zone value")
    here = os.path.dirname(os.path.abspath(__file__))
    nz = os.path.normpath(os.path.join(here, "..", "..", "..", "data", "nz", "NZ Well Correlation.wcproj.json"))
    if os.path.exists(nz):
        proj.well_project = nz
        sd = MH.petro_strip_data(proj, "Maui-7")
        check(sd is not None and np.isfinite(sd["tracks"]["GR"][0][1]).sum() > 1000 and sd["tops"], "21b: petro strip data "
              "for a real well (logs + tops)")


def run_gui_hub(check, tmp):
    from .gui.basemap import Basemap
    from .gui.layers_dock import LayersPanel
    from .gui.main_window import MainWindow
    from . import maplayers as ML
    proj, truth, d = _setup(tmp, "hub_gui")
    _mm_summary(d)
    bm = Basemap()
    bm.project = proj
    st = ML.LayerStack.load(proj)
    bm.set_stack(st)
    panel = LayersPanel(bm)
    panel.set_stack(st)
    pl = panel.add_pies("A")
    check(st.get(pl)["style"]["mode"] == "pies" and not bm.layer_errors, "21b GUI: mineral pies layer drawn")
    vol = proj.volume("Synth")
    xs, ys = vol.outline()
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    bm.start_polygon()
    for x, y in ((cx - 50, cy - 50), (cx + 300, cy - 50), (cx + 300, cy + 50), (cx - 50, cy + 50)):
        bm.add_polygon_point(x, y)
    xy = bm.finish_polygon()
    lid, inside = panel.polygon_drawn(*xy)
    check(sorted(inside) == ["W-1", "W-2"] and sorted(bm.selected_wells) == ["W-1", "W-2"] and st.get(lid)["kind"] == "polygons",
          "21b GUI: drawn polygon kept as a layer, wells inside selected on the map")
    dlg = panel.crossplot()
    dd = dlg.compute("SoPhiH", "PHIE_res", "A")
    check(dd["n"] == 3 and dd["selected"].sum() == 2, "21b GUI: petro cross-plot highlights the lasso selection")
    check(bm.well_at(cx, cy, 5.0) == "W-1" and bm.well_at(cx + 1000, cy, 5.0) is None, "21b GUI: well under the cursor")
    mw = MainWindow(proj.path)
    mw.hub_tick()  # no request yet
    import json as _json
    with open(os.path.join(d, "wells_petro", "hub_request.json"), "w") as fh:
        _json.dump({"property": "NetPay_TVD", "zone": "A", "wells": ["W-1"]}, fh)
    lid = mw.hub_tick(force=True)
    check(lid is not None and mw.basemap.stack.get(lid)["src"]["property"] == "NetPay_TVD" and mw.basemap.selected_wells == ["W-1"],
          "21b GUI: Show on map hub request applied by the Seismic tool (layer + selection)")
    sd = mw.well_clicked("W-1")
    check(sd is not None and mw._petro_strip is not None, "21b GUI: clicking a well with petro results opens its petro strip")


def run_carto(check, tmp):
    """21f: style presets, hillshade, contour labels, fault-polygon blanking + ticks, well symbols, print export."""
    from . import maplayers as ML
    from . import mapexport as MX
    from .maps import horizon_grid
    from .welltie import wc
    wc()
    from well_correlation import carto as C
    proj, truth, d = _setup(tmp, "carto")
    with open(proj.well_project) as fh:
        wj = json.load(fh)
    for w, stt in zip(wj["wells"], ["Gas condensate discovery", "Dry hole", "Oil producer"]):
        w["info"] = {"Status": stt}
    with open(proj.well_project, "w") as fh:
        json.dump(wj, fh)
    st = ML.LayerStack.load(proj)
    gid, cid = st.add_grid(horizon_grid(proj, proj.horizons.get("Top"), "3D:Synth"), "Top time", interval=5.0)
    vol = proj.volume("Synth")
    xs, ys = vol.outline()
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    poly = {"name": "F1", "x": [cx - 20, cx + 20, cx + 60, cx + 20], "y": [cy - 200, cy - 200, cy + 200, cy + 200], "down": "W"}
    fid = st.add("polygons", "Fault polygons", src={"polygons": [poly]}, style={"fault": True})
    st.set_style(gid, blank=True)
    MX.apply_preset(st, "Classic")
    check(st.get(gid)["style"]["hillshade"] and st.get(gid)["style"]["ramp"] == "depth" and st.get(cid)["style"]["labels"]
          and proj.display["map_style"] == "Classic", "21f: style preset applied (hillshade, depth ramp, labelled contours)")
    prims = ML.render(st)
    check(not [q for q in prims if q[0] == "error"], "21f: styled map renders without layer errors")
    labs = [q for q in prims if q[0] == "text" and q[1] == cid and len(q) > 7]
    check(len(labs) >= 2, f"21f: {len(labs)} contour labels along the lines")
    boxes = [C.label_box(q[2], q[3], q[7], q[4], 1.0, 1.4) for q in labs]
    check(not C.boxes_overlap(boxes) or len(labs) < 2, "21f: contour label boxes do not overlap")
    hatched = [q for q in prims if q[0] == "polygon" and q[1] == fid and len(q) > 6 and q[6].get("hatch")]
    ticks = [q for q in prims if q[0] == "curve" and q[1] == fid and q[6] == "pairs"]
    tx = np.concatenate([q[2] for q in ticks]) if ticks else np.array([])
    check(hatched and ticks and np.all(tx[1::2] <= tx[0::2] + 1e-9), "21f: fault heave polygon hatched, ticks only on the "
          "downthrown (west) side")
    crosses = 0
    px_, py_ = np.append(poly["x"], poly["x"][0]), np.append(poly["y"], poly["y"][0])
    for q in prims:
        if q[0] == "curve" and q[1] == cid:
            x, y = q[2], q[3]
            n = len(x) // 2 * 2
            for k in range(len(px_) - 1):
                crosses += int(C._seg_cross(x[0:n:2], y[0:n:2], x[1:n:2], y[1:n:2], px_[k], py_[k], px_[k + 1], py_[k + 1]).sum())
    check(crosses == 0, "21f: contours never cross the fault polygon (grid blanked + dilated)")
    img = next(q for q in prims if q[0] == "image" and q[1] == gid)
    g = st.grid(gid)
    X, Y = g.node_xy()
    inside = ML.blank_polygons_grid(g, [(poly["x"], poly["y"])])
    check(np.isnan(inside.z).sum() > 0 and img[2][..., 3][np.isnan(inside.z)].max() == 0, "21f: grid blanked inside the fault polygon")
    syms = {q[6] for q in prims if q[0] == "points" and st.get(q[1])["kind"] == "wells"}
    check(syms == {"star", "x", "o"}, f"21f: well status symbols on the map {syms}")
    for page in ("A4 landscape", "A3 portrait", "A1 landscape"):
        bb = MX.layout_boxes(page)
        check(not MX.boxes_overlap(bb.values()), f"21f/g: {page} layout panels do not overlap")
    png = os.path.join(d, "map.png")
    out = MX.export_map(st, png, title="Top time", author="test", page="A4 landscape", dpi=80, crs="NZTM2000 (EPSG:2193)",
                        contour_interval=5.0)
    vis = [l["id"] for l in reversed(st.ordered()) if l["visible"]]
    check(os.path.getsize(png) > 20000 and [e["id"] for e in out["legend"]] == vis, "21f: print export (title block, legend = the "
          "visible layers, scale bar, north arrow)")
    pdf = os.path.join(d, "map.pdf")
    MX.export_map(st, pdf, title="Top time", dpi=72)
    check(os.path.getsize(pdf) > 5000, "21f: PDF export")
