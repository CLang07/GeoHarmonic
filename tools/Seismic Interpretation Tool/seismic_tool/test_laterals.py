"""Item 14 core: laterals (trajectories, attributes along MD, per-lateral stats, batched colour drawing)."""
import os
import time

import numpy as np


def run(check, tmp):
    from . import laterals as L
    from .maps import MapGrid
    # minimum curvature: straight 30 deg hole of 1000 m
    dz, dn, de = L.min_curvature([0, 1000], [30, 30], [90, 90])
    check(abs(dz[-1] - 866.025) < 0.01 and abs(de[-1] - 500) < 0.01 and abs(dn[-1]) < 1e-9, "minimum curvature")
    # synthetic horizontal well: lands at the target TVD, lateral of the requested length
    md, inc, azi, kop, md_land = L.synthetic_survey(None, 2530.0, 2500.0, 90.0, 3.0)
    t = L.Trajectory.from_survey("H", md, inc, azi, 1000.0, 5000.0, kb=30.0)
    R = 30.0 / np.radians(3.0)
    check(abs(t.z[-1] - 2500.0) < 0.5 and abs(t.x[-1] - 1000.0 - (R + 2500.0)) < 1.0 and abs(t.y[-1] - 5000.0) < 1e-6,
          f"synthetic lateral lands at 2500 m TVDSS (got {t.z[-1]:.2f}) with toe {t.x[-1] - 1000:.0f} m east")
    ll = t.lateral_length()
    check(2500.0 <= ll <= 2500.0 + R * np.radians(10) + 31, f"lateral length from inclination ({ll:.0f} m)")
    tr = t.resample(10.0)
    check(np.allclose(np.diff(tr.md)[:-1], 10.0) and abs(tr.z[-1] - t.z[-1]) < 1e-6, "resample along MD")
    # survey CSV
    p = os.path.join(tmp, "survey.csv")
    with open(p, "w") as fh:
        fh.write("MD (m),Inclination,Azimuth\n" + "\n".join(f"{a},{b},{c}" for a, b, c in zip(md, inc, azi)))
    m2, i2, a2 = L.read_survey_csv(p)
    check(np.allclose(m2, md) and np.allclose(i2, inc), "survey CSV read")
    # map grid along the well
    xs = np.arange(0, 6001, 50.0)
    X, Y = np.meshgrid(xs, np.arange(4000, 6001, 50.0))
    g = MapGrid(0.01 * X + 0.002 * Y, "xy", "ms", "plane", None, 0.0, 4000.0, 50.0, 50.0)
    v = L.sample_grid(t, g)
    check(np.nanmax(np.abs(v - (0.01 * t.x + 0.002 * t.y))) < 1e-6, "map attribute along the wellbore")
    # zones: wavy lateral through a 10 m zone (2495-2505) -> known % in zone
    wav = L.Trajectory("W", tr.md, tr.x, tr.y, tr.z + np.where(tr.lateral_mask(), 8.0 * np.sin(tr.md / 50.0), 0.0),
                       tr.inc)
    top = MapGrid(np.full(X.shape, 2495.0), "xy", "m", "top", None, 0.0, 4000.0, 50.0, 50.0)
    base = MapGrid(np.full(X.shape, 2505.0), "xy", "m", "base", None, 0.0, 4000.0, 50.0, 50.0)
    zin = L.zone_along(wav, top, base)
    m = wav.lateral_mask()
    expect = 100 * np.mean(np.abs(wav.z[m] - 2500.0) <= 5.0)
    v = L.sample_grid(wav, g)
    st = L.lateral_stats(wav, v, zin)
    check(abs(st["pct_in_zone"] - expect) < 1e-9 and 30 < st["pct_in_zone"] < 90,
          f"% in zone along a wavy lateral ({st['pct_in_zone']:.1f}%)")
    d = L.distance_to_surface(wav, top)
    check(np.allclose(d, wav.z - 2495.0), "distance to a surface along the well")
    check(st["p10"] < st["p50"] < st["p90"] and abs(st["mean"] - np.mean(v[m])) < 1e-9, "lateral P10/P50/P90/mean")
    # stages
    stages = [(md_land + 100 * k, md_land + 100 * k + 60, 1000.0 + 10 * k) for k in range(20)]
    sv = L.stage_values(tr, stages)
    k5 = np.argmin(np.abs(tr.md - (md_land + 530)))
    check(sv[k5] == 1050.0 and np.isnan(sv[np.argmin(np.abs(tr.md - (md_land + 580)))]), "stage values along MD")
    sp = os.path.join(tmp, "stages.csv")
    with open(sp, "w") as fh:
        fh.write("API,Stage,Top MD,Base MD,Proppant per ft\n")
        for k, (a, b, c) in enumerate(stages):
            fh.write(f"42-000-00001,{k + 1},{a},{b},{c}\n")
    rs = L.read_stages_csv(sp)
    w1 = rs["42-000-00001"]
    check(len(w1) == 20 and w1[3][2]["Proppant per ft"] == 1030.0 and w1[3][0] == stages[3][0], "stage CSV read")
    # seismic along the wellbore (synthetic 3D; twt = z ms, i.e. 2000 m/s)
    from .synth import make_project
    dd = os.path.join(tmp, "lat3d")
    os.makedirs(dd)
    proj, truth = make_project(dd, nil=20, nxl=30, ns=200, lines=False)
    vol = proj.volume("Synth")
    il, xl = vol.inlines[7], vol.xlines[3:25]
    x, y = vol.xy(np.full(len(xl), il), xl)
    z = np.linspace(300, 500, len(xl))
    wt = L.Trajectory("S", np.arange(len(xl)) * 25.0, x, y, z)
    sv = L.sample_volume(wt, vol, lambda zz: zz)
    direct = np.array([np.interp(zz / 1000.0 / vol.dt, np.arange(vol.ns), vol.trace(il, c)) for zz, c in zip(z, xl)])
    check(np.allclose(sv, direct, atol=1e-5), "seismic amplitude along the wellbore (nearest trace, twt from depth)")
    out = L.Trajectory("O", [0, 10], [0, 0], [0, 0], [300, 300])
    check(np.all(np.isnan(L.sample_volume(out, vol, lambda zz: zz))), "outside the survey -> NaN")
    # per-lateral statistics table
    wells = L.synthetic_laterals(8, 1000.0, 4200.0, 200.0, 2000.0, 1500.0)
    attrs = {"amp": {w.name: L.sample_grid(w, g) for w in wells}}
    zt = MapGrid(np.full(X.shape, 1990.0), "xy", "m", "", None, 0.0, 4000.0, 50.0, 50.0)
    zb = MapGrid(np.full(X.shape, 2010.0), "xy", "m", "", None, 0.0, 4000.0, 50.0, 50.0)
    tab = L.stats_table(wells, attrs, {w.name: L.zone_along(w, zt, zb) for w in wells})
    check(len(tab) == 8 and tab.ids[0] == "42-000-00001" and np.all(tab.cols["pct_in_zone"] == 100.0)
          and np.all(np.abs(tab.cols["landing_tvdss"] - 2000) < 25) and "amp p90" in tab.cols
          and np.all(tab.cols["amp p10"] <= tab.cols["amp p90"]), "per-lateral statistics table (analytics inputs)")
    # drawing: colour bins cover every segment exactly once; batched over 1,000 laterals
    bins = L.color_bins(tr.x, tr.y, np.arange(len(tr.md), dtype=float), n_bins=8)
    nseg = sum(np.isfinite(xs[0::3]).sum() for _, xs, _ in bins)
    check(nseg == len(tr.md) - 1 and len(bins) == 8, "colour bins: one segment per station interval")
    big = L.synthetic_laterals(1000, 0.0, 0.0, 50.0, 2000.0, 2500.0, seed=2)
    vals = {w.name: w.z for w in big}
    t0 = time.time()
    b = L.batch_color_bins(big, vals, 1990.0, 2010.0, 16, pixel=40.0)
    dtb = time.time() - t0
    nfull = sum(len(w.md) for w in big)
    npts = sum(np.isfinite(xs).sum() for xs, _ in b.values()) // 2
    check(len(b) <= 16 and npts < nfull and dtb < 10, f"1,000 laterals -> {len(b)} batched curves ({npts} of {nfull} "
                                                     f"segments after LOD) in {dtb:.2f} s")
    li = L.lod_indices(tr.x, tr.y, 100.0)
    check(li[0] == 0 and li[-1] == len(tr.md) - 1 and len(li) < len(tr.md) / 3, "level-of-detail decimation")
    check(L.colormap_rgb(16).shape == (16, 3), "bin colours")
    dashboard_checks(check, tmp, tab)
    return True


def dashboard_checks(check, tmp, tab):
    """Linked selection / filters / panel data of the dashboard model on the per-lateral table."""
    from .dashboard_model import DashboardModel, points_in_polygon
    events = []
    dm = DashboardModel(tab, {"operator": ["A", "B"] * 4})
    dm.subscribe(events.append)
    dm.add_column("eur", np.arange(8) * 10.0)
    dm.select_box("eur", "lateral_m", 15, 45, -1e9, 1e9)
    check(dm.selected == {2, 3, 4} and events[-1] == "selection", "dashboard: box selection on a crossplot")
    dm.set_filter("operator", values=["A"])
    check(list(np.nonzero(dm.visible())[0]) == [0, 2, 4, 6] and dm.selected == {2, 4},
          "dashboard: text filter hides rows and drops them from the selection")
    dm.set_filter("eur", lo=10.0)
    e, c, cs = dm.histogram("eur", 3)
    check(c.sum() == 3 and cs.sum() == 2, "dashboard: histogram of visible + selected rows")
    dm.select_hist_bin("eur", 55, 100, "add")
    check(dm.selected == {2, 4, 6}, "dashboard: histogram bin selection (add)")
    xp = dm.crossplot("eur", "lateral_m", "operator")
    check(list(xp["rows"]) == [2, 4, 6] and xp["selected"].all() and list(xp["c"]) == ["A"] * 3,
          "dashboard: crossplot data follows filters + selection")
    tr = dm.table_rows(["eur"], sort="eur", descending=True)
    check([r for r, _ in tr] == [6, 4, 2], "dashboard: sorted table rows")
    xy = tab.xy
    dm.clear_filters()
    dm.select_lasso(None, None, [(xy[:, 0].min() - 1, xy[1, 1] - 1), (xy[:, 0].max() + 1, xy[1, 1] - 1),
                                 (xy[:, 0].max() + 1, xy[3, 1] + 1), (xy[:, 0].min() - 1, xy[3, 1] + 1)])
    check(dm.selected == {1, 2, 3}, "dashboard: lasso on the map (surface locations)")
    pin = points_in_polygon([0.5, 2, 0.5], [0.5, 0.5, np.nan], [(0, 0), (1, 0), (1, 1), (0, 1)])
    check(list(pin) == [True, False, False], "point in polygon")
    dm.set_filter("eur", 0, 60)
    lp = os.path.join(tmp, "layout.json")
    dm.layout["x"] = "eur"
    dm.save_layout(lp)
    dm2 = DashboardModel(tab, {"operator": ["A", "B"] * 4})
    dm2.load_layout(lp)
    check(dm2.filters == {"eur": ("range", 0, 60)} and dm2.selected == {1, 2, 3} and dm2.layout["x"] == "eur",
          "dashboard: layout saved / restored (filters, selection by API, panels)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [0, 1])
    out = dm.export_html(os.path.join(tmp, "dash.html"), "Laterals", [("test", fig)])
    plt.close(fig)
    txt = open(out, encoding="utf-8").read()
    check("data:image/png" in txt and txt.count("class='sel'") == 3 and "42-000-00002" in txt,
          "dashboard: standalone HTML export")


def run_gui(check, tmp):
    """Dashboard window logic: wells from the linked project (survey + vertical), colouring by TVDSS / grid /
    seismic / stages / zone, lateral stats, linked selection, analytics hand-off, HTML export."""
    from . import laterals as L
    from .gui.dashboard_window import DashboardWindow
    from .maps import MapGrid, horizon_grid
    from .test_velocity import velocity_case
    d = os.path.join(tmp, "dashgui")
    os.makedirs(d)
    sp, T = velocity_case(d)
    ws = sp.wellset()
    b = ws.well("B-1")
    md, inc, azi, kop, md_land = L.synthetic_survey(None, 1000.0 + float(b.elev or 0), 400.0, 90.0, 6.0)
    sv = os.path.join(d, "B-1_survey.csv")
    with open(sv, "w") as fh:
        fh.write("MD,INC,AZI\n" + "\n".join(f"{a},{c},{e}" for a, c, e in zip(md, inc, azi)))
    b.survey_path = sv
    trajs = L.trajectories_from_wellset(ws)
    tb = next(t for t in trajs if t.name == "B-1")
    ta = next(t for t in trajs if t.name == "A-1")
    check(len(trajs) == 3 and abs(tb.z[-1] - 1000.0) < 1.0 and tb.x[-1] > b.x + 600 and ta.lateral_length() == 0.0
          and abs(ta.z[0] + float(ws.well("A-1").elev or 0)) < 1e-6,
          "wells from the project: deviated (survey, min. curvature) + vertical")
    win = DashboardWindow(sp)
    win._ws = ws
    from .analytics.table import AttributeTable
    extra = AttributeTable(["A-1", "B-1", "C-1"]).add("eur", [1.0, 2.0, 3.0])
    win.set_trajectories(trajs, extra)
    check(win.model.table.names[:2] == ["lateral_m", "lateral_ft"] and list(win.model.table.cols["eur"]) == [1, 2, 3],
          "dashboard: wells + joined engineering table")
    g1 = horizon_grid(sp, sp.horizons.get("H1"), "3D:V")
    win.add_grid("H1 time", g1)
    srcs = win.color_sources()
    check(srcs[:2] == ["TVDSS", "grid: H1 time"] and "seismic: V" in srcs, f"dashboard: colour sources {srcs}")
    vz = win.color_by("TVDSS")
    check(np.allclose(vz["B-1"], tb.z) and win.vrange[0] < win.vrange[1], "dashboard: colour by TVDSS")
    vg = win.color_by("grid: H1 time")
    check(np.allclose(vg["B-1"], g1.sample(tb.x, tb.y), equal_nan=True), "dashboard: colour by a map grid")
    vs = win.color_by("seismic: V")
    td = ws.td_for("B-1")
    k = len(tb.md) - 1
    direct = L.sample_volume(L.Trajectory("p", [0], [tb.x[k]], [tb.y[k]], [tb.z[k]]), sp.volume("V"), td.twt)[0]
    check(np.isfinite(vs["B-1"]).mean() > 0.5 and abs(vs["B-1"][k] - direct) < 1e-6,
          "dashboard: colour by seismic amplitude along the wellbore (well time-depth)")
    win.color_by("TVDSS")
    bat = win.map_batches()
    check(0 < len(bat) <= 16, "dashboard: batched colour curves")
    st = win.add_lateral_stats()
    check("TVDSS mean" in win.model.table.cols and np.isfinite(win.model.table.cols["TVDSS mean"][1])
          and np.isnan(win.model.table.cols["TVDSS mean"][0]), "dashboard: lateral statistics (only for laterals)")
    zt = MapGrid(np.full(g1.z.shape, 990.0), g1.kind, "m", "top", g1.vol, g1.x0, g1.y0, g1.dx, g1.dy)
    zb = MapGrid(np.full(g1.z.shape, 1010.0), g1.kind, "m", "base", g1.vol, g1.x0, g1.y0, g1.dx, g1.dy)
    win.add_grid("top", zt)
    win.add_grid("base", zb)
    win.set_zone("top", "base")
    strip_checks(check, sp, ws, tb, zt, zb)
    vzn = win.color_by("in zone")
    win.add_lateral_stats()
    check(np.nanmax(vzn["B-1"]) == 1.0 and win.model.table.cols["pct_in_zone"][1] == 100.0,
          "dashboard: in-zone colouring + % in zone")
    stp = os.path.join(d, "stages.csv")
    with open(stp, "w") as fh:
        fh.write("Well,Top MD,Base MD,Proppant per ft\n")
        for s in range(5):
            fh.write(f"B-1,{md_land + 80 * s},{md_land + 80 * s + 60},{1000 + 100 * s}\n")
    win.load_stages(stp)
    vst = win.color_by("stage: Proppant per ft")
    check(np.nanmax(vst["B-1"]) == 1400.0 and np.all(np.isnan(vst["A-1"])), "dashboard: colour by stage CSV values")
    md_log = np.arange(0, tb.md[-1] + 1, 0.5)
    win.well_logs = lambda n: {"GR": (md_log, 50 + 0.01 * md_log)} if n == "B-1" else {}
    win._log_names = None
    check("log: GR" in win.color_sources(), "dashboard: log curves offered as colour sources")
    vl = win.color_by("log: GR")
    check(np.allclose(vl["B-1"], 50 + 0.01 * tb.md) and np.all(np.isnan(vl["A-1"])), "dashboard: colour by a log along MD")
    sd = win.open_strip(trajs.index(tb))
    check(sd is not None and "section" in sd and "GR" in sd["logs"] and "top" in sd and np.isfinite(sd["twt"]).all(),
          "dashboard: well strip (log, zone, seismic along the wellbore)")
    check(win.nearest_well(tb.x[-1] + 5, tb.y[-1]) == trajs.index(tb), "dashboard: map click picks the nearest wellbore")
    win.set_background("H1 time")
    check(win.background == "H1 time", "dashboard: background grid")
    win.model.add_text("operator", ["X", "Y", "X"])
    win.text_filter("operator", "X, Z")
    check(list(np.nonzero(win.model.visible())[0]) == [0, 2], "dashboard: text filter")
    win.text_filter("operator", "")
    check(win.model.visible().all(), "dashboard: text filter removed")
    win.model.select([1])
    check(win.model.selected == {1}, "dashboard: linked selection")
    win.filter_from_region("eur", 1.5, 10.0, (1.0, 3.0))
    check(win.model.filters["eur"] == ("range", 1.5, None) and list(np.nonzero(win.model.visible())[0]) == [1, 2],
          "dashboard: histogram range handles filter the wells")
    win.filter_from_region("eur", 0.0, 9.0, (1.0, 3.0))
    check("eur" not in win.model.filters, "dashboard: range dragged to the data limits removes the filter")

    class _A:
        got = None

        def add_table(self, t):
            self.got = t
            return t
    a = _A()
    win.send_to_analytics(a)
    check(a.got is not None and "pct_in_zone" in a.got.cols, "dashboard: lateral stats sent to Analytics")
    out = win.export_html(os.path.join(d, "dash.html"))
    check(os.path.getsize(out) > 5000 and "data:image/png" in open(out, encoding="utf-8").read(),
          "dashboard: HTML export with the coloured lateral map")
    return True


def strip_checks(check, sp, ws, tb, zt, zb):
    """Well strip data: section along the wellbore, well time path, zone top/base, logs."""
    from . import laterals as L
    vol = sp.volume("V")
    td = ws.td_for(tb.name)
    md_log = np.arange(0, tb.md[-1] + 1, 0.5)
    gr = 50 + 0.01 * md_log
    sd = L.strip_data(tb, {"GR": (md_log, gr)}, zt, zb, vol, td.twt, step=10.0)
    k = len(sd["md"]) // 2
    xk, yk, _ = tb.at_md(sd["md"][k])
    il, xl = vol.ilxl_at(float(xk), float(yk))
    tr = vol.trace(il, xl)
    check(sd["section"].shape == (len(sd["md"]), vol.ns) and np.allclose(sd["section"][k], tr),
          "well strip: seismic section along the wellbore (nearest traces)")
    check(np.allclose(sd["logs"]["GR"], 50 + 0.01 * sd["md"]) and np.allclose(sd["below_top"], sd["z"] - 990.0)
          and np.nanmax(sd["twt"]) > 0 and np.all(np.diff(sd["twt"][sd["z"] < 900]) >= -1e-9),
          "well strip: log along MD, distance below the zone top, two-way time path")
