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
