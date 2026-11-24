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
    panel.remove(pid)
    check(all(l["id"] != pid for l in st.layers) and not bm.layer_errors, "21a GUI: remove; no layer errors")
    mw = MainWindow(proj.path)
    check(mw.basemap.stack is not None and mw.layers_panel.stack is mw.basemap.stack and len(mw.layers_panel._ids) >= 3,
          "21a GUI: main window opens the project with the saved map layers")
