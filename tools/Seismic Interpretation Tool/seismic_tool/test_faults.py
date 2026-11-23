"""Fault sticks / surfaces / tracker barrier checks (called from selftest.run and run_gui)."""
import os

import numpy as np


def _faulted_project(tmp, sub):
    from .project import SeisProject
    from .synth import faulted_cube, write_3d
    from .volume import import_3d
    d = os.path.join(tmp, sub)
    os.makedirs(d)
    cube, fxl, base = faulted_cube()
    write_3d(os.path.join(d, "f.sgy"), cube)
    store = os.path.join(d, "f_seismic", "F")
    import_3d(os.path.join(d, "f.sgy"), store)
    p = SeisProject()
    p.path = os.path.join(d, "f.seisproj.json")
    p.name = "f"
    p.volumes.append({"name": "F", "store": store, "segy": os.path.join(d, "f.sgy")})
    # a 2D line crossing the fault at right angles (along inline index 20), taken from the cube
    from .selftest import write_segy
    vol = p.volume("F")
    tr = cube[20]
    x, y = vol.xy(np.full(vol.nxl, vol.inlines[20]), vol.xlines)
    write_segy(os.path.join(d, "L.sgy"), tr, {21: np.arange(1, vol.nxl + 1), 181: np.round(x).astype(np.int64),
                                              185: np.round(y).astype(np.int64)})
    p.lines.append({"name": "L", "segy": os.path.join(d, "L.sgy"), "cdp_byte": 21})
    p.save()
    return p, fxl, base


def run(check, tmp):
    from . import faults as flt
    from . import horizons as hz
    from .project import SeisProject
    from .synth import fault_sticks_from

    # ---- triangulation
    A = np.array([[0, 0, 100], [0, 10, 200], [0, 20, 300]], float)
    B = np.array([[50, 0, 100], [50, 10, 200]], float)
    tris = flt.zip_sticks(A, B)
    check(len(tris) == 3, "zip two sticks (3 + 2 points) into 3 triangles")
    f = flt.Fault("t")
    f.add_stick(B[::-1])
    f.add_stick(A)
    f.add_stick(np.array([[25, 0, 100], [25, 10, 220]]))
    st = f.ordered_sticks()
    check([s[0, 0] for s in st] in ([0, 25, 50], [50, 25, 0]) and np.all(np.diff(st[0][:, 2]) > 0),
          "sticks ordered along strike, points by time")
    check(f.triangles().shape == (2 + 3, 3, 3), "fault surface triangles")

    p, fxl, base = _faulted_project(tmp, "flt")
    vol = p.volume("F")
    fs = p.faults
    fault = fs.new("F1")
    for s in fault_sticks_from(fxl, vol, [0, 13, 26, 39]):
        fault.add_stick(s)
    # ---- section cuts
    seg = flt.section_segments_3d(fault, vol, "Inline", int(vol.inlines[20]))
    xs, ts = np.r_[seg[0], seg[2]], np.r_[seg[1], seg[3]]
    want = vol.xl0 + fxl(20, ts)
    check(len(seg[0]) >= 1 and np.max(np.abs(xs - want)) < 0.05, "fault cut on an inline at the true position")
    seg = flt.section_segments_3d(fault, vol, "Crossline", int(vol.xlines[30]))
    check(seg is not None and len(seg[0]) >= 1, "fault cut on a crossline")
    seg = flt.section_segments_3d(fault, vol, "Time slice", 400.0)
    xs, ils = np.r_[seg[0], seg[2]], np.r_[seg[1], seg[3]]
    check(np.max(np.abs(xs - (vol.xl0 + fxl(ils - vol.il0, 400.0)))) < 0.05, "fault trace on a time slice")
    seg = flt.section_segments_3d(fault, vol, "Inline", int(vol.inlines[-1]))
    check(seg is not None and len(seg[0]) >= 1, "fault cut on the last inline (end stick plane)")
    line = p.line("L")
    seg = flt.section_segments_line(fault, line.x, line.y)
    k = np.r_[seg[0], seg[2]]
    t = np.r_[seg[1], seg[3]]
    check(len(seg[0]) >= 1 and np.max(np.abs(k - fxl(20, t))) < 0.6, "fault cut on a 2D line")
    on = flt.sticks_on_section_3d(fault, vol, "Inline", int(vol.inlines[13]))
    check(len(on) == 1 and len(flt.sticks_on_section_3d(fault, vol, "Inline", int(vol.inlines[14]))) == 0,
          "stick shown only on the inline it was drawn on")

    # ---- tracker stops at the fault (throw = reflector spacing: ignoring the fault mis-ties)
    key = "3D:F"
    jj = np.arange(vol.nxl)[None]
    ii = np.arange(vol.nil)[:, None]
    hang = jj > fxl(ii, base)
    res = {}
    for use in (False, True):
        hs = hz.HorizonSet()
        pk = hz.Picker(p, hs)
        h = hs.new("T")
        pk.pick_points(h, key, 20, 5, base[20, 5])
        pk.track(h, key, [(20, 5)], faults=[fault] if use else None)
        g = h.get(key)
        res[use] = (np.isfinite(g[~hang]).mean(), np.isfinite(g[hang]).mean())
    check(res[False][1] > 0.9, f"without faults the tracker crosses the fault ({res[False][1]:.0%} of hanging wall)")
    check(res[True][1] == 0 and res[True][0] > 0.95,
          f"with the fault the tracker stays in the footwall ({res[True][0]:.0%} footwall, {res[True][1]:.0%} hanging wall)")
    hs = hz.HorizonSet()
    pk = hz.Picker(p, hs)
    h = hs.new("T2")
    pk.pick_points(h, "2D:L", 0, 5, base[20, 5])
    pk.track(h, "2D:L", None, hz.TrackParams(smooth=1), faults=[fault])
    g = h.get("2D:L")
    cut = int(np.floor(fxl(20, base[20, 30])))
    check(np.isfinite(g[:cut - 1]).all() and not np.isfinite(g[cut + 2:]).any(), "2D tracker stops at the fault")
    # gap bridging must not jump the fault either
    h3 = hs.new("T3")
    pk.pick_points(h3, "2D:L", 0, 5, base[20, 5])
    pk.track(h3, "2D:L", None, hz.TrackParams(smooth=1, max_gap=6), faults=[fault])
    check(not np.isfinite(h3.get("2D:L")[cut + 2:]).any(), "gap bridging does not jump over a fault")

    # ---- undo + persistence
    n0 = len(fault.sticks)
    fs.snapshot(fault)
    del fault.sticks[0]
    fault.invalidate()
    fs.undo()
    check(len(fault.sticks) == n0 and len(fault.triangles()) > 0, "undo stick delete")
    f2 = fs.new("F2")
    f2.add_stick([(0, 0, 100), (10, 0, 300)])
    fs.delete("F2")
    fs.undo()
    check(fs.get("F2") is not None, "undo fault delete")
    p.save()
    q = SeisProject.open(p.path)
    check(q.faults.names() == ["F1", "F2"] and np.allclose(q.faults.get("F1").sticks[1], fault.sticks[1], atol=1e-3),
          "faults saved and reloaded with the project")
    return p


def run_gui(check, tmp):
    from .gui.section_panel import SectionPanel
    from .gui.horizon_dock import HorizonDock
    from .gui.fault_dock import FaultDock
    from .gui.basemap import Basemap
    from .synth import fault_sticks_from  # noqa: F401
    p, fxl, base = _faulted_project(tmp, "fgui")
    vol = p.volume("F")
    panel = SectionPanel()
    panel.project = p
    hd = HorizonDock(panel)
    hd.project = p
    fd = FaultDock(panel)
    fd.project = p
    f = fd.new_fault()
    check(panel.fault_mode and not panel.pick_mode and panel.active_fault == f.name, "fault dock: N = new fault, fault mode on")
    hd.set_pick_mode(True)
    check(panel.pick_mode and not panel.fault_mode, "horizon pick mode switches fault mode off")
    fd.set_fault_mode(True)
    check(panel.fault_mode and not panel.pick_mode, "fault mode switches horizon pick mode off")
    panel.source = ("3D", "F")
    panel.mode = "Inline"
    for i in (0, 20, 39):
        panel.number = int(vol.inlines[i])
        panel.redraw()
        for tm in (150.0, 350.0, 600.0):
            panel.fault_click(float(vol.xl0 + fxl(i, tm)), tm)
        check(len(panel.stick) == 3, f"stick points on inline {panel.number}") if i == 0 else None
        panel.end_stick()
    check(len(f.sticks) == 3 and len(f.triangles()) > 0, "three sticks -> fault surface")
    panel.number = int(vol.inlines[10])
    panel.redraw()
    kinds = [k for _, k, _, _ in panel.fault_lines()]
    check(kinds == ["cut"], "fault surface drawn where it cuts an inline without sticks")
    panel.number = int(vol.inlines[20])
    panel.redraw()
    kinds = sorted(k for _, k, _, _ in panel.fault_lines())
    check(kinds == ["cut", "stick"], "stick + cut drawn on the inline with a stick")
    panel.fault_click(float(vol.xl0 + 5), 300.0)
    panel.undo()
    check(panel.stick == [], "Ctrl+Z removes the last point of the stick being drawn")
    # stick on another section ends the previous one
    panel.fault_click(float(vol.xl0 + fxl(20, 200)), 200.0)
    panel.number = int(vol.inlines[25])
    panel.redraw()
    panel.fault_click(float(vol.xl0 + fxl(25, 200)), 200.0)
    check(len(panel.stick) == 1, "moving to another section starts a new stick")
    panel.cancel_stick()
    # horizon tracking from the panel respects the fault
    h = hd.new_horizon("Top")
    # the Qt mock returns 0/False for widget values: set the dock defaults explicitly
    panel.track_faults = True
    panel.track_params.min_corr = 0.6
    panel.track_params.max_gap = 3
    hd.apply_settings = lambda: None  # keep these (refresh() would read the mock widgets again)
    panel.number = int(vol.inlines[20])
    panel.redraw()
    panel.pick_at(float(vol.xlines[5]), float(base[20, 5]))
    panel.track()
    g = h.get("3D:F")
    jj = np.arange(vol.nxl)[None]
    ii = np.arange(vol.nil)[:, None]
    hang = jj > fxl(ii, base)
    check(np.isfinite(g[~hang]).mean() > 0.9 and not np.isfinite(g[hang]).any(), "panel T: tracker stops at the fault")
    # undo order: last edit was the horizon track -> undo restores the horizon, not the fault
    nst = len(f.sticks)
    panel.undo()
    check(np.isfinite(h.get("3D:F")).sum() == 1 and len(f.sticks) == nst, "Ctrl+Z undoes the most recent edit (horizon)")
    panel.undo()  # horizon seed pick
    fd.set_fault_mode(True)
    panel.delete_stick_near(float(vol.xl0 + fxl(20, 350)), 350.0)
    check(len(f.sticks) == nst - 1, "Shift+click deletes the stick near the cursor")
    panel.undo()
    check(len(f.sticks) == nst, "Ctrl+Z restores the deleted stick")
    # 2D line display
    panel.source = ("2D", "L")
    panel.mode = "2D"
    panel.number = None
    panel.redraw()
    check(any(k == "cut" for _, k, _, _ in panel.fault_lines()), "fault cut shown on the 2D line")
    bm = Basemap()
    bm.project = p
    check(len(bm.fault_map_lines()) == len(f.sticks) + 1, "basemap: sticks + fault trace")
    bm.draw_faults()
    fd.refresh()
    return True
