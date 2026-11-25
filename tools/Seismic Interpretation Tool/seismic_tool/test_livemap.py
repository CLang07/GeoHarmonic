"""Live Map logic tests (PLAN 31b map core, 31d well styling + linked views)."""
import math

import numpy as np

from . import livemap as LM
from . import mapbus as MB


def _table():
    t = MB.WellTable()
    rng = np.random.default_rng(3)
    names = [f"W-{k}" for k in range(1, 13)]
    x = np.arange(12) * 100.0
    y = (np.arange(12) % 4) * 100.0
    MB.table_from_wells(list(zip(names, x, y)), t)
    t.set_column("Net pay", dict(zip(names, np.round(rng.uniform(0, 30, 12), 1))), owner="petro", unit="m")
    t.set_column("Status", dict(zip(names, ["Oil", "Gas", "Dry", "Oil"] * 3)), owner="wellcorr")
    mist = {n: float(v) for n, v in zip(names, [2, 15, 4, 30, 1, 0, 12, 3, 8, 22, 5, 9])}
    mist["W-12"] = None
    t.set_column("Mistie_ms", mist, owner="mistie", unit="ms")
    t.set_column("V_sand", dict(zip(names, rng.uniform(0, 1, 12))), owner="petro")
    t.set_column("V_shale", dict(zip(names, rng.uniform(0, 1, 12))), owner="petro")
    t.set_column("NoXY", {"Z-99": 1.0}, owner="x")  # a well without coordinates
    return t


def run(check, tmp):
    # ---- 31b: scale bar, north arrow, cursor readout, state
    L, n, lab = LM.nice_scale(12345.0, 0.2)
    check((L, n, lab) == (2500.0, 4, "2.5 km"), f"31b: nice scale bar ({L}, {n}, {lab})")
    check(LM.nice_scale(80.0)[2] == "20 m" and LM.nice_scale(4000.0)[0] == 1000.0, "31b: scale bar m / km labels")
    g = LM.scale_bar_geometry((0, 10000, 0, 5000))
    check(abs(g["segments"][-1][1] - g["segments"][0][0] - g["length"]) < 1e-6 and g["segments"][0][2]
          and not g["segments"][1][2], "31b: alternating scale-bar segments span the bar length")
    from .gis import crs as CRS
    utm = CRS.from_epsg(32631)  # F3 is in UTM 31N (ED50 in reality; WGS84 zone geometry is the same)
    x_cm, y = CRS.forward(utm, np.array([53.0]), np.array([3.0]))  # on the central meridian
    a0 = LM.north_angle(utm, float(x_cm[0]), float(y[0]))
    x_e, y_e = CRS.forward(utm, np.array([53.0]), np.array([5.5]))  # 2.5 deg east of the CM
    a1 = LM.north_angle(utm, float(x_e[0]), float(y_e[0]))
    gamma = math.degrees(math.atan(math.tan(math.radians(2.5)) * math.sin(math.radians(53.0))))
    check(abs(a0) < 0.01 and abs(abs(a1) - gamma) < 0.05, f"31b: north arrow = grid convergence ({a1:.3f} vs "
          f"{gamma:.3f} deg)")
    check(LM.north_angle(None, 0, 0) == 0.0, "31b: no CRS -> grid north")
    rx, ry, (lx, ly) = LM.north_arrow_geometry((0, 1000, 0, 1000), 0.0)
    check(abs(rx[0] - rx[-1]) < 1e-9 and ry.max() == ry[0] and ly > ry.max(), "31b: north arrow points up, N above")
    rx2, ry2, (lx2, ly2) = LM.north_arrow_geometry((0, 1000, 0, 1000), 90.0)
    check(rx2[0] > rx2[2] + 10 and lx2 > rx2.mean(), "31b: rotated north arrow points east at +90 deg")
    txt = LM.cursor_text(float(x_e[0]), float(y_e[0]), utm)
    check("53.00000 N" in txt and "5.50000 E" in txt and txt.startswith("X "), f"31b: cursor readout ({txt})")

    class Vol:
        inlines = np.arange(100, 200)
        xlines = np.arange(300, 400)
        il_step = xl_step = 1

        def ilxl_at(self, x, y):
            return 100 + x / 25.0, 300 + y / 25.0
    check("IL 140  XL 320" in LM.cursor_text(1000.0, 500.0, None, Vol()) and
          "IL" not in LM.cursor_text(1e6, 0.0, None, Vol()), "31b: cursor readout adds IL / XL inside the survey")

    class P:
        display = {}
    st = LM.MapState()
    st.view = [0, 10, 0, 5]
    st.well_style["color"]["mode"] = "categorical"
    st.well_style["color"]["column"] = "Status"
    st.save(P)
    st2 = LM.MapState.load(P)
    check(st2.view == [0, 10, 0, 5] and st2.well_style["color"]["column"] == "Status"
          and st2.well_style["size"]["fixed"] == 9.0, "31b: map state saved in the project (defaults merged)")

    # ---- 31d: well styling
    t = _table()
    s = LM.style_wells(t, {})
    check(s["rgba"].shape == (13, 4) and len(set(map(tuple, s["rgba"]))) == 1 and s["visible"].sum() == 12,
          "31d: default style = one colour; a well without X / Y is hidden")
    s = LM.style_wells(t, {"color": {"mode": "continuous", "column": "Net pay", "ramp": "viridis"}})
    v = t.numeric("Net pay")
    lo, hi = np.nanargmin(v[:12]), np.nanargmax(v[:12])
    check(tuple(s["rgba"][lo][:3]) == (68, 1, 84) and tuple(s["rgba"][hi][:3]) == (253, 231, 37)
          and tuple(s["rgba"][12]) == LM.NAN_COLOR, "31d: continuous ramp min / max, missing value grey")
    s = LM.style_wells(t, {"color": {"mode": "continuous", "column": "Net pay", "min": 0, "max": 60}})
    check(tuple(s["rgba"][hi][:3]) != (253, 231, 37), "31d: fixed ramp range")
    s = LM.style_wells(t, {"color": {"mode": "categorical", "column": "Status", "categories": {"Gas": "#ff0000"}}})
    st_ = t.text("Status")
    gas = [k for k, v in enumerate(st_) if v == "Gas"]
    oil = [k for k, v in enumerate(st_) if v == "Oil"]
    check(all(tuple(s["rgba"][k]) == (255, 0, 0, 255) for k in gas) and len({tuple(s["rgba"][k]) for k in oil}) == 1
          and [e[0] for e in s["legend"]["color"]] == ["Oil", "Gas", "Dry"], "31d: categorical colours + legend")
    rules = [{"rule": "Mistie_ms > 10", "color": "#ff0000", "label": "mistie > 10 ms"},
             {"rule": "Mistie_ms > 5", "color": "#ffa500"}]
    s = LM.style_wells(t, {"color": {"mode": "rules", "rules": rules, "default": "#00ff00"}})
    m = t.numeric("Mistie_ms")
    red = [k for k in range(13) if tuple(s["rgba"][k][:3]) == (255, 0, 0)]
    orange = [k for k in range(13) if tuple(s["rgba"][k][:3]) == (255, 165, 0)]
    check(red == [k for k in range(13) if m[k] > 10] and orange == [k for k in range(13) if 5 < m[k] <= 10]
          and s["legend"]["color"][0][0] == "mistie > 10 ms", "31d: rule-based colour, first rule wins")
    s = LM.style_wells(t, {"size": {"column": "Net pay", "min_px": 4, "max_px": 20}})
    check(abs(s["size"][lo] - 4) < 1e-9 and abs(s["size"][hi] - 20) < 1e-9 and s["legend"]["size"][0] == "Net pay",
          "31d: bubble size by column (area ~ value)")
    s = LM.style_wells(t, {"shape": {"column": "Status", "map": {"Dry": "x"}}, "label": {"column": "Mistie_ms"}})
    check(s["symbol"][2] == "x" and s["symbol"][0] == "o" and s["symbol"][1] == "s" and s["label"][0] == "2"
          and s["label"][11] == "", "31d: shape and label by column")
    s = LM.style_wells(t, {"pie": {"columns": ["V_sand", "V_shale"]}})
    f = s["pies"]["W-1"]
    check(abs(sum(f.values()) - 1) < 1e-9 and "Z-99" not in s["pies"], "31d: pie fractions per well")
    bus = MB.MapBus()
    bus.filters.set_values("Status", ["Oil"], source="map")
    s = LM.style_wells(t, {}, bus.filters, ["w-3", "W-4"])
    check(s["visible"].sum() == 6 and s["selected"].tolist().count(True) == 2 and not s["visible"][1],
          "31d: Filters panel hides wells instantly; selection flags")
    check(LM.labels_in_view(s, (0, 450, 0, 400)) == [0, 3, 4] and LM.labels_in_view(s, (0, 2000, 0, 500), 2) == [],
          "31d: labels only for visible wells in view, none above the limit")
    bus.filters.clear(source="map")
    try:
        LM.style_wells(t, {"color": {"mode": "continuous", "column": "Nope"}})
        check(False, "31d: unknown style column raises")
    except KeyError:
        check(True, "31d: unknown style column raises")

    # ---- linked views
    poly = [(-50, -50), (350, -50), (350, 150), (-50, 150)]
    check(LM.lasso_select(t, poly) == ["W-1", "W-2"] and LM.box_select(t, -1, 301, -1, 301) == ["W-1", "W-2", "W-3",
                                                                                                    "W-4"],
          "31d: lasso and box selection on the map")
    vis = np.ones(13, bool)
    vis[0] = False
    check(LM.lasso_select(t, poly, vis) == ["W-2"], "31d: lasso ignores filtered-out wells")
    bus.filters.set_range("Mistie_ms", None, 10, source="map")
    bus.filters.set_values("Status", ["Oil", "Gas"], source="map")
    h = LM.histogram_data(t, "Mistie_ms", bus.filters, ["W-1"], bins=5)
    expect = sum(1 for k in range(12) if t.text("Status")[k] in ("Oil", "Gas") and np.isfinite(m[k]))
    check(h["counts"].sum() == expect and h["selected"].sum() == 1,
          "31d: histogram ignores its own filter, honours the others, stacks the selection")
    h2 = LM.histogram_data(t, "Status", None, ["W-1"])
    check(h2["categories"][:3] == ["Oil", "Gas", "Dry"] and h2["counts"][:3] == [6, 3, 3] and h2["selected"][0] == 1,
          "31d: categorical histogram (with the coordinate-less well's blank category)")
    s = LM.style_wells(t, {}, bus.filters, ["W-1"])
    cp = LM.crossplot_data(t, "Net pay", "Mistie_ms", s)
    check(all(s["visible"][t.index(i)] for i in cp["ids"]) and cp["selected"].sum() == 1 and np.isfinite(cp["r"]),
          "31d: crossplot of visible wells with selection + r")
    cols, rows = LM.table_rows(t, ["Net pay", "Status"], s["visible"], ["W-5"], sort="Net pay", descending=True)
    nv = [float(r[1][0]) for r in rows[1:]]
    check(rows[0][0] == "W-5" and nv == sorted(nv, reverse=True) and len(rows) == int(s["visible"].sum()),
          "31d: well table: visible wells, selected first, sorted")
    check(LM.filter_widget_spec(t, "Net pay")[0] == "range" and LM.filter_widget_spec(t, "Status") ==
          ("values", ["Oil", "Gas", "Dry", ""]), "31d: filter widget per column type")


def run_gui(check, tmp):
    """Live Map window wiring (31b / 31d) under the Qt mock or real Qt."""
    from .gui.main_window import MainWindow
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "livemap_gui")
    mw = MainWindow(proj.path)
    lm = mw.live_map()
    check(lm is mw.live_map() and lm.bus is mw.mapbus, "31 GUI: Live Map opens on the shared map bus (one window)")
    t = lm.bus.table
    check(sorted(t.ids) == ["W-1", "W-2", "W-3"] and lm.styled is not None and int(lm.styled["visible"].sum()) == 3,
          "31 GUI: wells drawn from the bus WellTable")
    check(lm.canvas.skip_kinds == ("wells",), "31 GUI: hub well layer replaced by the styled wells")
    lm.canvas.view_override = (t.numeric("X").min() - 500, t.numeric("X").max() + 500,
                               t.numeric("Y").min() - 500, t.numeric("Y").max() + 500)
    deco = lm.canvas.draw_decorations()
    check(deco is not None and deco["scale"]["length"] > 0 and deco["north"] == 0.0,
          "31b GUI: scale bar + north arrow drawn (no CRS -> grid north)")
    x0, y0 = float(t.numeric("X")[0]), float(t.numeric("Y")[0])
    txt = lm.cursor_moved(x0, y0)
    check(txt.startswith("X ") and "IL " in txt, f"31b GUI: cursor readout with IL / XL ({txt})")
    # Spotfire-style styling from a column another tool pushed
    t.set_column("Mistie_ms", {"W-1": 2.0, "W-2": 14.0, "W-3": 6.0}, owner="mistie", unit="ms")
    lm.bus.notify("welltable", source="test")
    rules = lm.parse_rules("Mistie_ms > 10 = red; Mistie_ms > 5 = #ffa500")
    s = lm.set_color("rules", rules=rules)
    k2 = t.index("W-2")
    check(len(rules) == 2 and tuple(s["rgba"][k2][:3]) == LM.hex_rgba("#e15759")[:3], "31d GUI: rule colouring")
    s = lm.set_size("Mistie_ms", 5, 25)
    check(abs(s["size"][k2] - 25) < 1e-9, "31d GUI: bubble size by column")
    lm.set_label("Mistie_ms")
    lm.set_pies(["Mistie_ms"])
    check(lm.styled["label"][k2] == "14" and "W-2" in lm.styled["pies"], "31d GUI: labels / pies by column")
    # filters -> wells hidden instantly, shared through the bus
    spec = lm.add_filter("Mistie_ms")
    lm.set_range_filter("Mistie_ms", None, 10)
    check(spec[0] == "range" and int(lm.styled["visible"].sum()) == 2 and not lm.styled["visible"][k2]
          and mw.mapbus.filters.filters["Mistie_ms"]["max"] == 10, "31d GUI: range filter hides a well (bus filter)")
    check(len(lm._table_ids) == 2, "31d GUI: well table shows only the visible wells")
    lm.clear_filters()
    check(int(lm.styled["visible"].sum()) == 3 and not lm._filter_widgets, "31d GUI: clear filters")
    # selection: lasso on the map -> bus -> seismic basemap rings; seismic selection -> Live Map
    xs = [x0 - 50, x0 + 50, x0 + 50, x0 - 50]
    ys = [y0 - 50, y0 - 50, y0 + 50, y0 + 50]
    ids = lm.lasso(xs, ys)
    check(ids == [t.ids[0]] and mw.basemap.selected_wells == ids and lm.styled["selected"][0],
          "31d GUI: lasso selection shared with the seismic basemap")
    mw.maplink.wells_selected(["W-3"])
    check(lm.styled["selected"].tolist() == [False, False, True] and lm._table_ids[0] == "W-3",
          "31d GUI: seismic selection highlighted in the Live Map, selected first in the table")
    d = lm.update_crossplot("X", "Mistie_ms")
    check(d is not None and len(d["ids"]) == 3 and d["selected"].sum() == 1, "31d GUI: linked crossplot")
    h = lm.update_histogram("Mistie_ms")
    check(h["counts"].sum() == 3 and h["selected"].sum() == 1, "31d GUI: linked histogram")
    # active section from the seismic window drawn on the Live Map
    vol = mw.project.volume("Synth")
    sec = mw.section
    sec.source, sec.mode, sec.number, sec.domain, sec.attr = ("3D", "Synth"), "Inline", int(vol.inlines[4]), "Time", \
        "Seismic"
    sec.redraw()
    mw.maplink.position_changed()
    check(lm.canvas._section_item is not None, "31b GUI: current inline drawn on the Live Map")
    # state saved in the project
    st = lm.save_state()
    from .project import SeisProject
    p2 = SeisProject.open(proj.path)
    check(st["well_style"]["size"]["column"] == "Mistie_ms" and
          p2.display["live_map"]["well_style"]["color"]["mode"] == "rules", "31b GUI: Live Map state saved in the project")
    # a bad style column (column removed by its owner) falls back to the default style, no crash
    lm.set_color("continuous", column="Gone")
    check(lm.styled is not None and lm.state.well_style["color"]["mode"] == "fixed" and lm.messages,
          "31d GUI: unknown style column -> default style + message")
    mw.maplink.close()


def run_culture(check, tmp):
    """31c: shapefile / GeoJSON culture layers, presets, style by attribute, CRS warning."""
    import os
    from .gis import geojson as GJ
    from .gis import shapefile as SHP
    from .gis.vector import VectorLayer, culture_style, layer_of, vector_primitives, category_colors
    from . import maplayers as ML
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "culture")
    xs, ys = proj.volume("Synth").outline()
    x0, y0 = float(np.min(xs)), float(np.min(ys))
    sq = lambda a, b, s: np.array([[a, b], [a, b + s], [a + s, b + s], [a + s, b], [a, b]], float)  # clockwise
    # leases: own / competitor / open blocks
    geoms = [{"type": "Polygon", "parts": [sq(x0 + k * 300, y0, 250)]} for k in range(3)]
    attrs = {"LEASE": ["A-1", "B-7", "C-2"], "STATUS": ["Own", "Competitor", "Open"]}
    shp = os.path.join(d, "leases.shp")
    SHP.write(shp, geoms, attrs)
    vl = VectorLayer.load(shp)
    st = culture_style("Leases / permits (own / competitor / open)", vl)
    check(st["by"] == "STATUS" and st["labels"] == "LEASE" and st["fill"], "31c: lease preset finds STATUS / LEASE fields")
    cols, cmap = category_colors(vl, "STATUS", st["categories"])
    check(cmap["Own"] == (46, 160, 67) and cmap["Competitor"] == (214, 39, 40) and cmap["Open"] == (190, 190, 190),
          "31c: own / competitor / open colours (case-insensitive)")
    stack = ML.LayerStack.load(proj)
    lid = stack.add("vector", "Leases", src={"file": shp}, style=st)
    prims = vector_primitives(stack, stack.get(lid))
    fills = [q for q in prims if q[0] == "polygon"]
    labels = [q[4] for q in prims if q[0] == "text"]
    check(len(fills) == 3 and fills[0][4][:3] == (46, 160, 67) and labels == ["A-1", "B-7", "C-2"],
          "31c: lease polygons filled by status, labelled")
    # GeoJSON round trip: polygon with a hole, multipolygon, multilinestring, points
    hole = sq(x0 + 50, y0 + 50, 50)[::-1]  # counter-clockwise = hole
    g2 = [{"type": "Polygon", "parts": [sq(x0, y0, 200), hole]},
          {"type": "Polygon", "parts": [sq(x0 + 500, y0, 100), sq(x0 + 700, y0, 100)]},
          {"type": "LineString", "parts": [np.array([[x0, y0], [x0 + 900, y0 + 900]]),
                                            np.array([[x0, y0 + 50], [x0 + 400, y0 + 50]])]},
          {"type": "Point", "parts": [np.array([[x0 + 10.5, y0 + 20.25]])]}]
    a2 = {"name": ["no-go", "blocks", "Gas pipeline", "Platform A"], "value": [1, 2.5, np.nan, np.int64(4)]}
    gj = os.path.join(d, "culture.geojson")
    GJ.write_geojson(gj, g2, a2, epsg=2193)
    import json
    doc = json.load(open(gj))
    ext = doc["features"][0]["geometry"]["coordinates"][0]
    outer_ccw = GJ._area2(np.array(ext)) > 0
    check(doc["crs"]["properties"]["name"].endswith("2193") and outer_ccw and
          doc["features"][1]["geometry"]["type"] == "MultiPolygon" and
          doc["features"][2]["geometry"]["type"] == "MultiLineString" and doc["features"][0]["properties"]["value"] == 1
          and doc["features"][2]["properties"]["value"] is None, "31c/31i: GeoJSON writer (RFC 7946 rings, crs member)")
    rg, ra, epsg = GJ.read_geojson(gj)
    from .gis.shapefile import ring_roles
    check(epsg == 2193 and ring_roles(rg[0]["parts"]) == ["outer", "hole"] and len(rg[1]["parts"]) == 2 and
          len(rg[2]["parts"]) == 2 and np.allclose(rg[3]["parts"][0], [[x0 + 10.5, y0 + 20.25]]) and
          ra["name"][3] == "Platform A", "31c: GeoJSON reader (holes, multi-parts, attributes, EPSG)")
    vg = VectorLayer.load(gj)
    check(len(vg) == 4 and vg.crs is not None, "31c: GeoJSON as a vector layer")
    # pipelines dashed, no-go hatched
    pl = os.path.join(d, "pipes.geojson")
    GJ.write_geojson(pl, [g2[2]], {"COMMODITY": ["gas"]}, epsg=2193)
    lp = stack.add("vector", "Pipelines", src={"file": pl}, style=culture_style("Pipelines", VectorLayer.load(pl)))
    curves = [q for q in vector_primitives(stack, stack.get(lp)) if q[0] == "curve"]
    ng = os.path.join(d, "nogo.geojson")
    GJ.write_geojson(ng, [g2[0]], {"NAME": ["reserve"]}, epsg=2193)
    ln = stack.add("vector", "No-go", src={"file": ng}, style=culture_style("No-go / conservation areas"))
    polys = [q for q in vector_primitives(stack, stack.get(ln)) if q[0] == "polygon"]
    check(len(curves) == 2 and all(q[7]["dash"] for q in curves) and polys and polys[0][6]["hatch"],
          "31c: pipeline preset dashed, no-go preset hatched")
    # CRS warning: lat/lon layer while the project CRS is unset and the wells are in metres
    ll = os.path.join(d, "coast.geojson")
    GJ.write_geojson(ll, [{"type": "LineString", "parts": [np.array([[173.0, -39.5], [173.5, -39.4]])]}], {})
    lc = stack.add("vector", "Coast", src={"file": ll}, style=culture_style("Coastline"))
    layer_of(stack, stack.get(lc))
    check("project CRS is not set" in stack.get(lc)["src"]["crs_warning"], "31c: CRS mismatch warning")
    stack.set_visible(lc, False)  # the mis-registered lat/lon layer would stretch the view
    for l in stack.layers:
        if l["kind"] == "lines2d":  # synthetic 2D lines are in local coordinates
            stack.set_visible(l["id"], False)
    stack.save()
    # render through the hub renderer (matplotlib) = what the PNG / PDF export draws
    png = os.path.join(d, "culture.png")
    ML.render_png(stack, png)
    check(os.path.getsize(png) > 2000, "31c: culture layers render (PNG)")


def run_gui_culture(check, tmp):
    """'Add shapefile...' with a preset through the Map layers panel of the Live Map."""
    import os
    from .gui.main_window import MainWindow
    from .gis import geojson as GJ
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "culture_gui")
    mw = MainWindow(proj.path)
    lm = mw.live_map()
    xs, ys = proj.volume("Synth").outline()
    x0, y0 = float(np.min(xs)), float(np.min(ys))
    p = os.path.join(d, "leases.geojson")
    sq = np.array([[x0, y0], [x0, y0 + 100], [x0 + 100, y0 + 100], [x0 + 100, y0], [x0, y0]])
    GJ.write_geojson(p, [{"type": "Polygon", "parts": [sq]}], {"Status": ["own"], "Lease": ["L-1"]})
    lid = lm.layers.add_vector(p, preset="Leases / permits (own / competitor / open)")
    lay = lm.canvas.stack.get(lid)
    check(lay["style"]["by"] == "Status" and lay["src"]["preset"].startswith("Leases") and lay["style"]["fill"],
          "31c GUI: Add shapefile with the lease preset")
    mw.maplink.close()


def run_seismic(check, tmp):
    """31e: mistie markers, shotpoint annotation on 2D lines."""
    from . import maplayers as ML
    from .test_maplayers import _setup
    ev = {"ties": [{"a": "L1", "b": "L2", "x": 1.0, "y": 2.0, "shift_ms": 9.0, "res_shift_ms": 0.5},
                   {"a": "L1", "b": "L3", "x": 3.0, "y": 4.0, "shift_ms": -6.0, "res_shift_ms": -3.0},
                   {"a": "L2", "b": "L3", "x": 5.0, "y": 6.0, "shift_ms": 14.0, "res_shift_ms": 7.5}], "role": "mistie"}
    m = LM.mistie_markers(ev)
    check([tuple(c[:3]) for c in m["rgba"]] == [(0, 170, 0), (240, 150, 0), (220, 0, 0)] and
          m["label"][1] == "L1 x L3: -3.0 ms" and LM.mistie_markers({}) is None,
          "31e: mistie residual markers green / orange / red")
    m2 = LM.mistie_markers(ev, use_residual=False)
    check(m2["value"].tolist() == [9.0, 6.0, 14.0], "31e: markers by raw mistie")
    idx, lab = LM.shotpoint_marks(np.arange(10), np.arange(10), np.arange(101, 351), 100)
    check(idx == [0, 99, 199, 249] and lab == ["101", "200", "300", "350"], "31e: shotpoint annotation every 100 + ends")
    proj, truth, d = _setup(tmp, "seislayers")
    st = ML.LayerStack.load(proj)
    lid = next(l["id"] for l in st.layers if l["kind"] == "lines2d")
    n0 = len([q for q in ML.render(st) if q[1] == lid and q[0] == "text"])
    st.set_style(lid, shotpoints=10)
    prims = [q for q in ML.render(st) if q[1] == lid]
    n1 = len([q for q in prims if q[0] == "text"])
    check(proj.lines and n1 > n0 and any(q[0] == "points" for q in prims), f"31e: 2D lines annotated with shotpoints "
          f"({n1 - n0} labels)")


def run_gui_seismic(check, tmp):
    """31e GUI: misties drawn on the Live Map, arbitrary line opens a section, click a line opens it."""
    from .gui.main_window import MainWindow
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "seis_gui")
    mw = MainWindow(proj.path)
    lm = mw.live_map()
    ties = [{"a": "L1", "b": "3D:Synth", "xy": (500100.0, 6000100.0), "shift_ms": 4.0, "res_shift_ms": 6.0}]
    mw.maplink.misties_changed(["L1"], {"L1": {"shift_ms": 4.0}}, ties)
    check(lm.canvas.misties is not None and len(lm.canvas.misties["x"]) == 1 and
          tuple(lm.canvas.misties["rgba"][0][:3]) == (220, 0, 0), "31e GUI: seismic mistie residuals on the Live Map")
    opened = []
    lm.open_arbitrary = lambda name, xs, ys: opened.append((name, xs, ys)) or name
    vol = mw.project.volume("Synth")
    xs, ys = vol.outline()
    x0, y0, x1, y1 = float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))
    lm.start_arbitrary()
    check(lm.canvas.min_poly_points == 2, "31e GUI: arbitrary line accepts two points")
    r = lm.drawn([x0 + 10, x1 - 10], [y0 + 10, y1 - 10])
    check(r == "Arbitrary 1" and opened and mw.mapbus.active.info("section")["kind2"] == "arbitrary" and
          lm.canvas.min_poly_points == 3, "31e GUI: arbitrary line -> section window + active section on the bus")
    real = MainWindow.open_arbitrary(mw, "Arb", [x0 + 10, x1 - 10], [y0 + 10, y1 - 10])
    check(real is not None, "31e GUI: main window opens the arbitrary section from the Live Map line")
    n = lm.set_shotpoints(10)
    lay = next(l for l in lm.canvas.stack.layers if l["kind"] == "lines2d")
    check(n == 1 and lay["style"]["shotpoints"] == 10, "31e GUI: shotpoint annotation toggled on the 2D lines")
    lm.set_shotpoints(0)
    shown = []
    lm.open_section = lambda *a: shown.append(a)
    line = mw.project.lines[0]["name"]
    lm._open("2D", line)
    check(shown and shown[0][:2] == ("2D", line) and mw.mapbus.active.get("line") == line,
          "31e GUI: clicking a 2D line on the Live Map opens it (and makes it active)")
    mw.maplink.close()


def run_wellsection(check, tmp):
    """31f: wells along a drawn line, section trace."""
    t = _table()
    hits = LM.wells_along_line(t, [-10, 1200], [0, 0], buffer=30)
    check([h[0] for h in hits] == ["W-1", "W-5", "W-9"] and hits[1][1] == 410.0, "31f: wells along a line, in order")
    hits = LM.wells_along_line(t, [0, 300, 300], [0, 300, 0], buffer=1)
    check([h[0] for h in hits] == ["W-1", "W-2", "W-3", "W-4"] and LM.wells_along_line(t, [0], [0]) == [],
          "31f: polyline with a bend; one point = nothing")
    check(LM.section_xy(t, ["W-2", "Z-99", "W-1"]) == [[100.0, 100.0], [0.0, 0.0]], "31f: section trace skips wells "
          "without coordinates")


def run_gui_wellsection(check, tmp):
    """31f GUI: Live Map draws a well section; the Well Correlation Tool (other process) would follow via the bus."""
    from .gui.main_window import MainWindow
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "ws_gui")
    mw = MainWindow(proj.path)
    lm = mw.live_map()
    other = MB.MapBus()
    br = MB.FileBridge(other, mw.maplink.bridge.folder, "wellcorr")
    x, y = lm.bus.table.xy()
    order = np.argsort(x + 1e-3 * y)
    wells = lm.well_section_along([float(x.min()) - 10, float(x.max()) + 10], [float(y.min()), float(y.min())], 5.0)
    br.poll()
    ws = other.active.info("well_section")
    check(wells and ws.get("wells") == wells and lm.canvas._well_section_item is not None and
          other.selection.get("wells") == wells, "31f GUI: well section along a drawn line -> bus (Well Correlation)")
    lm.select_wells(["W-3", "W-1"])
    w2 = lm.well_section_from_selection()
    br.poll()
    check(w2 == ["W-3", "W-1"] and other.active.info("well_section")["wells"] == ["W-3", "W-1"],
          "31f GUI: section from the selected wells (selection order)")
    other.active.set("well_section", "Section 1", source="wellcorr", wells=["W-2", "W-3"], xy=[[0, 0], [1, 1]])
    mw.bus_tick()
    check(lm.canvas._well_section_item is not None and lm.bus.active.info("well_section")["wells"] == ["W-2", "W-3"],
          "31f GUI: the correlation window's section line drawn on the Live Map")
    other.table.set_column("Top F0", {"W-1": 2500.0, "W-2": 2510.0, "W-3": 2490.0}, owner="wellcorr", unit="m")
    br.save_table("wellcorr")
    mw.bus_tick()
    s = lm.set_color("continuous", column="Top F0")
    check("Top F0" in lm.bus.table.cols and tuple(s["rgba"][lm.bus.table.index("W-3")][:3]) == (68, 1, 84),
          "31f GUI: picked tops arrive in the WellTable and colour the map instantly")
    br.close()
    mw.maplink.close()


def run_livegrid(check, tmp):
    """31g: live gridding of tops / picks, coarse-then-full scheduling, fault barriers, depth layers, isochore."""
    from . import maplayers as ML
    from .livegrid import LiveGridder, depth_grids, isochore
    from .maps import MapGrid
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "livegrid")
    st = ML.LayerStack.load(proj)
    t = MB.WellTable()
    rng = np.random.default_rng(5)
    px, py = rng.uniform(0, 4000, 25), rng.uniform(0, 3000, 25)
    names = [f"T-{k}" for k in range(25)]
    MB.table_from_wells(list(zip(names, px, py)), t)
    plane = lambda x, y: 2000.0 + 0.05 * x - 0.02 * y  # noqa: E731
    t.set_column("Top F0", dict(zip(names, plane(px, py))), owner="wellcorr", unit="m")

    class Clock:
        now = 0.0

        def __call__(self):
            return self.now
    clk = Clock()
    lg = LiveGridder(st, t, proj, pause=1.5, clock=clk, threaded=False)
    lg.add_source("F0", "column", column="Top F0", cell=100.0, method="tps", extent=(0, 4000, 0, 3000))
    lid = lg.sources["F0"]["lid"]
    g = st.grid(lid)
    X, Y = g.node_xy()
    check(g.level == "full" and g.dx == 100.0 and np.nanmax(np.abs(g.z - plane(X, Y))) < 0.5 and
          st.get(lg.sources["F0"]["cid"]) is not None, "31g: tops gridded (TPS reproduces a plane) + contour layer")
    # a top moves: coarse grid immediately, full grid after the pause, same layer replaced in place
    t.set_column("Top F0", dict(zip(names, plane(px, py) + 10.0)), owner="wellcorr", unit="m")
    hit = lg.changed_column(["Top F0"])
    g = st.grid(lid)
    check(hit == ["F0"] and g.level == "coarse" and g.dx == 400.0 and lg.sources["F0"]["lid"] == lid,
          "31g: change -> coarse grid at once (cell x 4) in the same layer")
    clk.now = 1.0
    lg.tick()
    check(st.grid(lid).level == "coarse", "31g: no full grid while still picking (before the pause)")
    clk.now = 2.0
    lg.tick()
    g = st.grid(lid)
    X, Y = g.node_xy()
    check(g.level == "full" and np.nanmax(np.abs(g.z - plane(X, Y) - 10.0)) < 0.5, "31g: full grid after the pause")
    check(lg.changed_column(["Other"]) == [], "31g: unrelated column changes are ignored")
    # threaded worker: two quick changes -> only the newest result is applied
    lg.threaded = True
    t.set_column("Top F0", dict(zip(names, plane(px, py) + 20.0)), owner="wellcorr", unit="m")
    lg.changed("F0")
    t.set_column("Top F0", dict(zip(names, plane(px, py) + 30.0)), owner="wellcorr", unit="m")
    lg.changed("F0")
    lg.wait()
    g = st.grid(lid)
    X, Y = g.node_xy()
    check(g.level == "coarse" and abs(np.nanmedian(g.z - plane(X, Y)) - 30.0) < 1.0 and not lg.busy(),
          "31g: background worker, stale result dropped")
    lg.threaded = False
    # fault barrier: down-thrown block east of x = 2000 by 100 m
    throw = np.where(px > 2000, 100.0, 0.0)
    t.set_column("Top F1", dict(zip(names, plane(px, py) + throw)), owner="wellcorr", unit="m")
    st.add("polygons", "Fault A", src={"polygons": [{"name": "A", "x": [2000, 2000, 2010, 2010],
                                                     "y": [-100, 3100, 3100, -100]}]}, style={"fault": True})
    lg.add_source("F1 faulted", "column", column="Top F1", cell=100.0, method="idw", barriers=True,
                  extent=(0, 4000, 0, 3000))
    lg.add_source("F1 smooth", "column", column="Top F1", cell=100.0, method="idw", barriers=False,
                  extent=(0, 4000, 0, 3000))
    gb, gs = st.grid(lg.sources["F1 faulted"]["lid"]), st.grid(lg.sources["F1 smooth"]["lid"])
    j0, j1 = int(1900 / 100), int(2100 / 100)
    jump_b = float(np.nanmedian(gb.z[:, j1] - gb.z[:, j0]))
    jump_s = float(np.nanmedian(gs.z[:, j1] - gs.z[:, j0]))
    check(len(lg.barrier_lines()) == 1 and jump_b > 80 and jump_s < jump_b - 30,
          f"31g: fault polygon is a barrier (jump {jump_b:.0f} m vs {jump_s:.0f} m without)")
    # picks of a horizon
    lg.add_source("Top picks", "horizon", horizon="Top", key="3D:Synth", cell=50.0)
    gh = st.grid(lg.sources["Top picks"]["lid"])
    check(np.isfinite(gh.z).mean() > 0.9 and gh.unit == "ms", "31g: horizon picks gridded")
    check(lg.changed_horizon(["Top"]) == ["Top picks"] and lg.changed_horizon(["Deep"]) == [],
          "31g: horizon edits re-grid only their sources")

    # depth layers from a depth model (stand-in with the VelocityModel interface)
    class VM:
        layers = [("Top", None), ("Deep", None)]

        def __init__(self, add):
            self.add = add

        def depth_grid(self, k):
            return MapGrid(np.full((5, 6), 1000.0 * (k + 1) + self.add), "xy", "m", "d", None, 0, 0, 100, 100)
    ids = depth_grids(st, VM(0.0))
    ids2 = depth_grids(st, VM(25.0))
    check(ids == ids2 and len(ids) == 2 and float(st.grid(ids[1]).z[0, 0]) == 2025.0,
          "31g: depth maps follow the depth model (layers refreshed in place)")
    iso = isochore(st.grid(ids[0]), st.grid(ids[1]))
    check(np.allclose(iso.z, 1000.0) and iso.dx == 100, "31g: isochore = base - top")
    lg.remove_source("F1 smooth")
    check("F1 smooth" not in lg.sources and not any(l["name"] == "Live: F1 smooth" for l in st.layers),
          "31g: removing a live source removes its layer")


def run_gui_livegrid(check, tmp):
    """31g GUI: tops pushed by the Well Correlation Tool re-grid live on the Live Map; picks re-grid horizons."""
    from .gui.main_window import MainWindow
    from .maps import MapGrid
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "lg_gui")
    mw = MainWindow(proj.path)
    lm = mw.live_map()
    lm.gridder.threaded = False
    t = lm.bus.table
    x, y = t.xy()
    t.set_column("Top F0", {i: 2000.0 + 0.1 * xx for i, xx in zip(t.ids, x - x.min())}, owner="wellcorr", unit="m")
    lid = lm.add_live_grid("Top F0", "column", column="Top F0", method="idw", barriers=False)
    check(lid is not None and lm.canvas.stack.grid(lid).level == "full", "31g GUI: live grid layer added")
    other = MB.MapBus()
    br = MB.FileBridge(other, mw.maplink.bridge.folder, "wellcorr")
    other.table.set_column("Top F0", {i: 2100.0 for i in t.ids}, owner="wellcorr", unit="m")
    br.save_table("wellcorr")
    mw.bus_tick()
    g = lm.canvas.stack.grid(lid)
    check(g.level == "coarse" and abs(float(np.nanmedian(g.z)) - 2100.0) < 1e-6, "31g GUI: pushed tops -> coarse re-grid")
    lm.gridder.pause = 0.0
    lm.grid_tick()
    check(lm.canvas.stack.grid(lid).level == "full", "31g GUI: full grid on the next pause")
    hid = lm.add_live_grid("Top", "horizon", horizon="Top", key="3D:Synth", cell=50.0)
    mw._horizons_changed()  # a pick edit in the seismic window
    check(lm.canvas.stack.grid(hid).level == "coarse", "31g GUI: horizon edit -> coarse re-grid of its live grid")

    class VM:
        layers = [("Top", None)]

        def depth_grid(self, k):
            return MapGrid(np.full((4, 4), 1500.0), "xy", "m", "d", None, float(x.min()), float(y.min()), 50, 50)
    lm.depth_model_fn = lambda: VM()
    ids = lm.set_live_depth(True)
    mw.maplink.depth_model_changed({"message": "rebuilt", "layers": [0]})
    check(len(ids) == 1 and any(l["name"] == "Live depth: Top" for l in lm.canvas.stack.layers),
          "31g GUI: live depth maps follow the dynamic depth model")
    br.close()
    mw.maplink.close()


def run_livedepth(check, tmp):
    """31g with the real item-17 DynamicDepthModel: a moved top changes the live depth map in place."""
    from . import maplayers as ML
    from .ddc import DynamicDepthModel
    from .livegrid import depth_grids
    from .velocity import VelocityModel
    from .test_ddc import ddc_case, kz_earth, LAYERS, _edit_tops
    sp, T = ddc_case(tmp, "livedepth", kz_earth)
    dm = DynamicDepthModel(sp, VelocityModel(sp, "3D:V", LAYERS, functions=[{"func": "v0kz"}, {"func": "v0"}])
                           .to_dict()).build()
    st = ML.LayerStack.load(sp)
    ids = depth_grids(st, dm)
    z0 = st.grid(ids[1]).z.copy()
    _edit_tops(sp.well_project, lambda t: t["W-3"].__setitem__("Top B", t["W-3"]["Top B"] + 4.0))
    rep = dm.refresh()
    ids2 = depth_grids(st, dm)
    z1 = st.grid(ids2[1]).z
    check(rep is not None and ids2 == ids and np.nanmax(np.abs(z1 - z0)) > 1.0 and
          np.array_equal(st.grid(ids[0]).z, dm.vm.depth_grid(0).z, equal_nan=True),
          "31g: live depth map updated in place after a moved top (real dynamic depth model)")


def run_output(check, tmp):
    """31h layer requests + 31i outputs (ZMAP+, GeoJSON, shapefile, PNG / PDF with the styled wells)."""
    import json
    import os
    from . import maplayers as ML
    from .gis.zmap import read_zmap, write_zmap
    from .maps import MapGrid
    from .test_maplayers import _setup
    # ZMAP+ round trip incl. nulls and a non-square grid
    z = np.arange(35, dtype=float).reshape(5, 7) * 1.5 + 1000.0
    z[1, 2] = np.nan
    g = MapGrid(z, "xy", "m", "Top F0 depth", None, 500000.0, 6000000.0, 25.0, 50.0)
    p = os.path.join(tmp, "f0.zmap")
    write_zmap(p, g)
    r = read_zmap(p)
    check(r.z.shape == (5, 7) and np.allclose(r.z, z, equal_nan=True) and abs(r.x0 - 500000) < 1e-6 and
          abs(r.dy - 50.0) < 1e-6 and r.label.startswith("Top F0"), "31i: ZMAP+ grid round trip (nulls, rows / columns)")
    txt = open(p).read().splitlines()
    hdr = [ln for ln in txt if ln.startswith("@")][0]
    check(hdr.endswith("HEADER, GRID, 5") and txt[[k for k, ln in enumerate(txt) if ln.startswith("@")][0] + 2]
          .startswith("5, 7, 500000"), "31i: ZMAP+ header (rows, columns, extent)")
    # 31h: a registered section line + a grid file from another tool
    proj, truth, d = _setup(tmp, "output")
    st = ML.LayerStack.load(proj)
    xs, ys = proj.volume("Synth").outline()
    x0, y0, x1, y1 = float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))
    req = LM.layer_request("Basin / Wheeler sections", "lines",
                           [{"name": "IL 5", "x": [x0, x1], "y": [y0, y0 + 10], "action": {"kind": "3D", "name": "Synth",
                                                                                          "mode": "Inline", "number": 5}}],
                           owner="seismic")
    lid = LM.apply_layer_request(st, req)
    req["items"].append({"name": "IL 9", "x": [x0, x1], "y": [y1, y1 - 10], "action": {}})
    lid2 = LM.apply_layer_request(st, req)
    check(lid == lid2 and len(st.get(lid)["src"]["polygons"]) == 2 and st.get(lid)["src"]["polygons"][0]["open"],
          "31h: registered section lines replaced in place by name")
    hit = LM.section_hit(st, 0.5 * (x0 + x1), y0 + 6, 20.0)
    check(hit is not None and hit[1]["name"] == "IL 5" and LM.section_hit(st, x0, 0.5 * (y0 + y1), 5.0) is None,
          "31h: click near a registered section line finds it")
    prims = [q for q in ML.render(st) if q[1] == lid]
    check(any(q[0] == "curve" for q in prims) and not any(q[0] == "polygon" for q in prims),
          "31h: section lines drawn as open polylines")
    gp = os.path.join(d, "inv_ai.zmap")
    write_zmap(gp, MapGrid(np.full((4, 5), 7000.0), "xy", "", "AI avg", None, x0, y0, 100, 100))
    gl = LM.apply_layer_request(st, LM.layer_request("Inversion: AI average", "grid", file=gp, owner="inversion"))
    check(st.grid(gl) is not None and float(st.grid(gl).z[0, 0]) == 7000.0, "31h: grid registered from a file "
          "(inversion / attribute extraction)")
    # 31i: GIS export with the WellTable + sections + picks
    t = MB.WellTable()
    MB.table_from_wells(proj.wells(), t)
    t.set_column("Net pay", {w[0]: 5.0 * (k + 1) for k, w in enumerate(proj.wells())}, owner="petro", unit="m")
    t.set_column("Status", {w[0]: "Oil" for w in proj.wells()}, owner="wc")
    styled = LM.style_wells(t, {"color": {"mode": "continuous", "column": "Net pay"}, "label": {"column": "Well"}})
    folder = os.path.join(d, "gis_out")
    out = LM.export_gis(st, t, folder, styled, {"well_section: S1": [[x0, y0], [x1, y1]]},
                        {"Top": (np.array([x0, x1]), np.array([y0, y1]), np.array([1000.0, np.nan]))})
    names = {os.path.basename(f) for f in out["files"]}
    wt = json.load(open(os.path.join(folder, "wells_table.geojson")))
    check({"wells_table.geojson", "wells_table.shp", "section_lines.geojson", "picks.geojson"} <= names and
          any(n.endswith(".zmap") for n in names) and any(n.endswith(".geojson") and n.startswith("Basin") for n in names)
          and wt["features"][0]["properties"]["Net pay"] == 5.0 and
          len(json.load(open(os.path.join(folder, "picks.geojson")))["features"]) == 1,
          f"31i: GIS export - wells with all columns, sections, picks, layers as GeoJSON, grids as ZMAP+ ({len(names)} files)")
    sec = json.load(open(os.path.join(folder, "Basin___Wheeler_sections.geojson"))) if os.path.exists(
        os.path.join(folder, "Basin___Wheeler_sections.geojson")) else None
    check(sec is None or sec["features"][0]["geometry"]["type"] == "LineString", "31i: section lines export as lines")
    for l in st.layers:
        if l["kind"] == "lines2d":  # synthetic 2D lines are in local coordinates
            st.set_visible(l["id"], False)
    png = os.path.join(d, "livemap.png")
    o = LM.export_map(st, styled, png, title="Live Map test", dpi=80)
    wl = [l for l in st.layers if l["kind"] == "wells"]
    check(os.path.getsize(png) > 5000 and all(l["visible"] for l in wl), "31i: PNG with title block, legend and the "
          "styled wells (well layer restored)")
    pdf = os.path.join(d, "livemap.pdf")
    LM.export_map(st, styled, pdf, title="Live Map test", dpi=80)
    check(open(pdf, "rb").read(4) == b"%PDF", "31i: PDF export")
    return png


def run_gui_output(check, tmp):
    """31h / 31i GUI: Wheeler section registered by the basin dialog, clickable; exports from the Live Map."""
    import os
    from .gui.main_window import MainWindow
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "out_gui")
    mw = MainWindow(proj.path)
    lm = mw.live_map()
    vol = mw.project.volume("Synth")
    sec = mw.section
    sec.source, sec.mode, sec.number, sec.domain, sec.attr = ("3D", "Synth"), "Inline", int(vol.inlines[6]), "Time", \
        "Seismic"
    sec.redraw()
    ev = mw.maplink.publish_current_section("Basin / Wheeler sections", wheeler=True, prop="ro")
    lay = next((l for l in lm.canvas.stack.layers if l["name"] == "Basin / Wheeler sections"), None)
    check(ev is not None and lay is not None and lay["src"]["polygons"][0]["action"]["wheeler"],
          "31h GUI: basin / Wheeler section location registered on the Live Map")
    shown = []
    lm.open_section = lambda *a: shown.append(a)
    it = lay["src"]["polygons"][0]
    mid = len(it["x"]) // 2
    hit = lm.map_clicked(it["x"][mid], it["y"][mid], 5.0)
    check(hit is not None and shown and shown[0][:3] == ("3D", "Synth", "Inline"),
          "31h GUI: clicking the registered section opens it in the seismic window")
    png = os.path.join(d, "lm.png")
    o = lm.export_map(png, dpi=60)
    out = lm.export_gis(os.path.join(d, "gis"))
    check(os.path.exists(png) and any(f.endswith("wells_table.geojson") for f in out["files"]),
          "31i GUI: Live Map export (PNG + GIS folder)")
    x, y = lm.bus.table.xy()
    n0 = len(lm.canvas._overlay)
    n = lm.show_paths(True, {"W-1": (np.array([x[0], x[0] + 300.0]), np.array([y[0], y[0] + 50.0]))})
    check(n == 1 and len(lm.canvas._overlay) == n0 + 1 and lm.show_paths(False) == 0,
          "31d GUI: deviated well paths drawn in the well colour, toggled off")
    mw.maplink.close()


def run_paths(check, tmp):
    """31d: deviated well paths as map polylines; vertical wells left out."""
    from .laterals import synthetic_laterals, Trajectory
    tr = synthetic_laterals(3, x0=1000.0, y0=2000.0, lateral=1500.0)
    vert = Trajectory("V-1", [0, 1000, 2000], [5.0, 5.0, 5.0], [7.0, 7.0, 7.0], [0, 1000, 2000])
    p = LM.path_polylines(tr + [vert], max_points=50)
    lens = [float(np.hypot(np.ptp(v[0]), np.ptp(v[1]))) for v in p.values()]
    check(len(p) == 3 and "V-1" not in p and all(len(v[0]) <= 51 for v in p.values()) and min(lens) > 1400,
          "31d: deviated well paths (laterals) as map polylines, vertical wells skipped")
