"""Horizon picking / autotracking checks (called from selftest.run)."""
import os

import numpy as np


def run(check, tmp):
    from . import horizons as hz
    from .project import SeisProject
    from .synth import make_project, ricker

    dt_s = 0.004
    # ---- snapping on an analytic trace with a sub-sample event
    t = np.arange(200) * dt_s
    tr = ricker(t - 0.40123, 30.0) - 0.8 * ricker(t - 0.5, 30.0)
    new, ok = hz.snap_times(tr[None], [100.0 + 3.0], "Peak", 6)
    err = abs(new[0] * 4.0 - 401.23)
    check(ok[0] and err < 0.4, f"peak snap sub-sample ({err:.2f} ms error)")
    new, ok = hz.snap_times(tr[None], [125.0 - 2.0], "Trough", 6)
    check(ok[0] and abs(new[0] * 4.0 - 500.0) < 0.4, "trough snap")
    # zero crossing - to + just above the trough of a ricker (symmetric) -> trough at 500, its side lobes...
    # use a pure sine: zero crossings at k*pi
    s = np.sin(np.arange(200) / 10.0)
    new, ok = hz.snap_times(s[None], [np.pi * 10 + 1.3], "Zero + to -", 4)
    check(ok[0] and abs(new[0] - np.pi * 10) < 0.02, "zero crossing + to - (linear interp)")
    new, ok = hz.snap_times(s[None], [np.pi * 20 - 1.0], "Zero - to +", 4)
    check(ok[0] and abs(new[0] - np.pi * 20) < 0.02, "zero crossing - to +")
    new, ok = hz.snap_times(np.zeros((1, 50)), [20.0], "Peak", 3)
    check(not ok[0] and new[0] == 20.0, "dead trace: no snap, time unchanged")

    # ---- synthetic project: folded 3D + two crossing 2D lines
    d = os.path.join(tmp, "hz")
    os.makedirs(d)
    proj, truth = make_project(d, nil=50, nxl=70, ns=250, noise=0.05)
    key = "3D:Synth"
    top = truth[key]
    hs = proj.horizons
    pk = hz.Picker(proj, hs)
    h = hs.new("Top")
    t0 = pk.pick_points(h, key, 25, 30, top[25, 30] + 7.0)  # click 7 ms off -> snapped to the peak
    check(abs(t0[0] - top[25, 30]) < 1.0, f"seed click snapped to peak ({t0[0] - top[25, 30]:+.2f} ms)")
    n = pk.track(h, key, [(25, 30)])
    g = h.get(key)
    err = np.abs(g - top)
    check(np.isfinite(g).all() and n == g.size - 1, f"3D autotrack covers the survey ({n} picks)")
    check(np.nanmax(err) < 4.0 and np.nanmean(err) < 0.5,
          f"3D autotrack recovers folded reflector (max {np.nanmax(err):.2f} ms, mean {np.nanmean(err):.2f} ms, dt 4 ms)")
    # second horizon on the deeper (trough) reflector, along one inline only
    h2 = hs.new("Deep")
    deep = truth["deep"]
    vol = proj.volume("Synth")
    il = int(vol.inlines[10])
    pk.pick_points(h2, key, 10, 5, deep[10, 5] - 5, "Trough")
    p = hz.TrackParams(mode="Trough")
    pk.track(h2, key, [(10, 5)], p, scope=("Inline", il))
    g2 = h2.get(key)
    check(np.isfinite(g2[10]).all() and np.isfinite(g2).sum() == vol.nxl, "track along this inline only")
    check(np.nanmax(np.abs(g2[10] - deep[10])) < 4.0, "trough horizon follows the deeper reflector")
    # undo restores the state before tracking
    before = np.isfinite(h2.get(key)).sum()
    hs.undo()
    check(np.isfinite(h2.get(key)).sum() == 1 and before == vol.nxl, "undo track")
    # manual line + erase
    pk.manual_line(h2, key, [20, 20], [0, 69], [deep[20, 0], deep[20, 69]], "Trough", 30.0)
    check(np.isfinite(h2.get(key)[20]).all(), "manual line fills every trace between the drawn points")
    pk.erase(h2, key, np.full(10, 20), np.arange(10))
    check(np.isfinite(h2.get(key)[20]).sum() == 60, "erase picks")
    # min correlation stops tracking into noise
    h3 = hs.new("Noise test")
    pk.pick_points(h3, key, 25, 30, 900.0, "None")  # pick in pure noise below the reflectors
    n3 = pk.track(h3, key, [(25, 30)], hz.TrackParams(mode="Peak", min_corr=0.95, schedule=()))
    check(n3 < 0.2 * vol.nil * vol.nxl, f"no tracking through incoherent noise with min corr 0.95 ({n3} picks)")
    hs.delete("Noise test")
    check(hs.get("Noise test") is None, "delete horizon")
    hs.undo()
    check(hs.get("Noise test") is not None, "undo delete")
    hs.delete("Noise test")

    # ---- 2D: ties
    lines = {n: (proj.line(n).x, proj.line(n).y) for n in ("L-A", "L-B")}
    ties = hz.line_intersections(lines)
    check(len(ties) == 1 and ties[0][1] == 60 and ties[0][3] == 40, f"2D line intersection {ties}")
    h4 = hs.new("2D top")
    pk.pick_points(h4, "2D:L-A", 0, 10, truth["2D:L-A"][10] + 4)
    res = pk.track_2d_network(h4, ["L-A"])
    ga, gb = h4.get("2D:L-A"), h4.get("2D:L-B")
    ea = np.nanmax(np.abs(ga - truth["2D:L-A"]))
    eb = np.nanmax(np.abs(gb - truth["2D:L-B"])) if gb is not None else 99
    check(set(res) == {"L-A", "L-B"} and np.isfinite(ga).all() and gb is not None and np.isfinite(gb).all(),
          "2D tracking follows the tie onto the crossing line")
    check(ea < 4 and eb < 4, f"2D picks accurate (max {ea:.2f} / {eb:.2f} ms)")

    # ---- bridging dead traces and lateral mixing (direct track_grid on an in-memory line)
    from .synth import traces_from_surfaces
    ntr = 200
    tt = 300 + 40 * np.sin(np.arange(ntr) / 25.0)
    line = traces_from_surfaces([tt], [1.0], 250, 0.004, noise=0.02)
    line[100:102] = 0.0  # two dead traces
    get = lambda ii, jj, a, b: line[jj, a:b]  # noqa: E731
    for gap, expect in ((1, False), (3, True)):
        gi = np.full((1, ntr), np.nan)
        gi[0, 20] = tt[20] / 4.0
        hz.track_grid(get, gi, np.isfinite(gi), hz.TrackParams(max_gap=gap), 4.0, 250)
        full = bool(np.isfinite(gi).all())
        check(full == expect, f"bridge dead traces with max gap {gap}: {'tracked across' if full else 'stops'}")
    check(np.nanmax(np.abs(gi[0, 103:] * 4.0 - tt[103:])) < 4.0, "picks after the bridged gap are on the event")
    noisy = traces_from_surfaces([tt], [1.0], 250, 0.004, noise=0.45, seed=3)

    class _L:  # minimal line object for smoothed_line_data
        def data(self):
            return noisy
    errs = {}
    for sm in (1, 7):
        gi = np.full((1, ntr), np.nan)
        gi[0, 100] = tt[100] / 4.0
        hz.track_grid(hz.line_traces(_L(), sm), gi, np.isfinite(gi), hz.TrackParams(smooth=sm), 4.0, 250)
        e = np.abs(gi[0] * 4.0 - tt)
        errs[sm] = (np.isfinite(e) & (e < 4.0)).sum()
    check(errs[7] > errs[1] and errs[7] > 0.9 * ntr,
          f"lateral mix 7 tracks noisy data better (correct picks {errs[1]} -> {errs[7]} of {ntr})")

    # ---- display helpers
    c = hz.section_curve(h, key, vol, "Crossline", int(vol.xlines[3]))
    check(c is not None and len(c[0]) == vol.nil and np.allclose(c[1], g[:, 3]), "section curve on crossline")
    xs, ys = hz.slice_points(h, key, vol, 320.0, tol_ms=2.0)
    check(len(xs) > 0 and np.all(np.abs(g[vol.il_index(ys[0]), vol.xl_index(xs[0])] - 320) <= 2),
          "horizon/time-slice intersection points")
    m11, m12, m21, m22, dx, dy = hz.grid_transform(vol)
    i, j = 7, 11
    X = m11 * (j + 0.5) + m21 * (i + 0.5) + dx
    Y = m12 * (j + 0.5) + m22 * (i + 0.5) + dy
    xx, yy = vol.xy(vol.inlines[i], vol.xlines[j])
    check(abs(X - xx) < 1e-6 and abs(Y - yy) < 1e-6, "map image transform puts pixel centres on traces")
    from .gui.colormaps import lut
    rgba, lo, hi = hz.structure_rgba(h2.get(key), lut("Grey"))
    check(rgba[0, 0, 3] == 0 and rgba[20, 30, 3] == 255, "structure map: unpicked cells transparent")

    # ---- save / load round trip
    proj.save()
    p2 = SeisProject.open(proj.path)
    check(p2.horizons.names() == hs.names(), "horizons listed after reopen")
    a, b = p2.horizons.get("Top").get(key), g
    check(a is not None and np.allclose(a, b, equal_nan=True), "horizon grid saved and reloaded")
    check(np.allclose(p2.horizons.get("2D top").get("2D:L-B"), gb, equal_nan=True), "2D horizon saved and reloaded")
    p2.horizons.rename("Top", "Top renamed")
    p2.horizons.delete("Deep")
    p2.save()
    p3 = SeisProject.open(proj.path)
    hdir = os.path.join(p3.seismic_dir(), "horizons")
    check(p3.horizons.names() == ["Top renamed", "2D top"] and not os.path.exists(os.path.join(hdir, "Deep"))
          and not os.path.exists(os.path.join(hdir, "Top")), "rename/delete persisted, stale folders removed")
    return proj


def run_gui(check, tmp):
    """Exercise the GUI logic under the Qt mock (dev/fakeqt on PYTHONPATH)."""
    from .synth import make_project
    from .gui.section_panel import SectionPanel
    from .gui.horizon_dock import HorizonDock
    from .gui.basemap import Basemap
    from .gui.main_window import MainWindow
    d = os.path.join(tmp, "gui")
    os.makedirs(d)
    proj, truth = make_project(d, nil=30, nxl=40, ns=200, noise=0.02)
    panel = SectionPanel()
    panel.project = proj
    dock = HorizonDock(panel)
    dock.project = proj
    h = dock.new_horizon("Top")
    check(panel.active_horizon == "Top" and panel.pick_mode, "dock: new horizon becomes active, pick mode on")
    panel.source = ("3D", "Synth")
    panel.mode = "Inline"
    vol = proj.volume("Synth")
    panel.number = int(vol.inlines[12])
    panel.redraw()
    top = truth["3D:Synth"]
    t = panel.pick_at(float(vol.xlines[20]), float(top[12, 20] + 6))
    check(t is not None and abs(t - top[12, 20]) < 1.0 and len(panel.seeds) == 1, "panel: click = snapped seed")
    n = panel.track()
    g = h.get("3D:Synth")
    check(n > 0 and np.isfinite(g).all() and panel.seeds == [], "panel: T tracks the whole survey from the seed")
    panel.pick_at(float(vol.xlines[5]), 0, erase=True)
    check(not np.isfinite(g[12, 5]) if g is h.get("3D:Synth") else not np.isfinite(h.get("3D:Synth")[12, 5]),
          "panel: Shift+click erases the pick")
    panel.undo()
    check(np.isfinite(h.get("3D:Synth")[12, 5]), "panel: undo")
    panel.pick_drag([(float(vol.xlines[2]), 300.0), (float(vol.xlines[8]), 320.0)], erase=True)
    check(np.isfinite(h.get("3D:Synth")[12]).sum() == vol.nxl - 7, "panel: Shift+drag erases a span")
    panel.mode = "Time slice"
    panel.number = 320
    panel.redraw()
    check(panel.pick_at(100.0, 320.0) is None, "panel: no picking on time slices")
    panel.mode = "Crossline"
    panel.number = int(vol.xlines[4])
    panel.redraw()
    check(len(panel._hz_items) >= 1, "panel: horizon drawn on crossline")
    dock.clear_line()
    check(not np.isfinite(h.get("3D:Synth")[:, 4]).any() and np.isfinite(h.get("3D:Synth")[:, 10]).all(),
          "dock: clear line removes picks on the crossline only")
    # 2D with ties via the panel
    panel.source = ("2D", "L-A")
    panel.mode = "2D"
    panel.number = None
    panel.redraw()
    line = proj.line("L-A")
    panel.pick_at(float(line.cdp[30]), float(truth["2D:L-A"][30] - 5))
    panel.track_scope = "Whole survey"
    panel.track()
    check(np.isfinite(h.get("2D:L-B")).all(), "panel: 2D track follows ties to the crossing line")
    bm = Basemap()
    bm.project = proj
    layers, rng = bm.horizon_map_layers("Top")
    kinds = sorted(l[0] for l in layers)
    check(kinds == ["image", "points", "points"] and rng[0] < rng[1], "basemap: time map layers for 3D + 2D")
    bm.show_horizon("Top")
    # main window wiring
    proj.save()
    w = MainWindow(proj.path)
    check(w.project.horizons.names() == ["Top"], "main window: horizons reloaded with the project")
    w.horizon_dock.toggle_pick_mode()
    w._horizons_changed()
    return True
