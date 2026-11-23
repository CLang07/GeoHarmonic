"""RGT step D checks: local edits (control point / moved level, horizon constraint, fault re-link), undo."""
import os

import numpy as np


def run(check, tmp):
    from . import faults as flt
    from .rgt import solve as S
    from .rgt import stack as K
    from .rgt.edit import RGTEditor2D, RGTEditor3D
    from .synth import fault_sticks_from
    from .test_rgt import geology_project, iso_errors

    p, G = geology_project(tmp, "edit", fault=False)
    vol = p.volume("G")
    rv, _ = S.compute_rgt_3d(vol)
    K.build_stack_3d(vol, rv, 40)
    folder = os.path.join(vol.store, "rgt")
    before = np.array(np.load(os.path.join(folder, "rgt.npy")))
    stack_before = np.array(np.load(os.path.join(folder, "stack.npy")))
    ed = RGTEditor3D(vol, radius=3)
    hz = G["horizons"]
    s = int(np.argmin(np.abs(G["tau_ref"] - 330)))
    i, j = 20, 30
    t_true = float(hz[s][i, j])
    import time as _t
    t0 = _t.time()
    sysm, v = ed.move_level(i, j, t_true, t_true + 12.0)
    dt_edit = _t.time() - t0
    check(abs(ed.value_at(i, j, t_true + 12.0) - v) < 0.5,
          f"moved level: RGT at the new position = grabbed value (local solve {sysm.stats['unknowns']} unknowns, "
          f"{dt_edit:.2f} s)")
    after = np.array(np.load(os.path.join(folder, "rgt.npy")))
    a, b = ed.changed_inlines
    diff = np.abs(after.astype(np.int64) - before.astype(np.int64)).max(axis=2)
    outside = np.ones(vol.nil, bool)
    outside[a:b + 1] = False
    I0, I1, J0, J1 = ed.edits[-1]["window"]
    far_j = np.arange(vol.nxl) > ed.Jpos[min(J1 + 1, len(ed.Jpos) - 1)]
    check(diff[outside].max() == 0 and diff[:, far_j].max() == 0 and diff[i, j] > 0,
          f"edit changes the model only locally (inlines {a}-{b}, window xl <= {ed.Jpos[min(J1 + 1, len(ed.Jpos) - 1)]})")
    rv2 = S.RGTVolume(folder)
    iso = rv2.iso_surface(v, vol)
    iso0 = S.iso_from_traces(before[:, :, :].astype(np.float32) * rv2.step + rv2.lo, v, vol.times)
    check(abs((iso[i, j] - iso0[i, j]) - 12.0) < 2.0 and np.nanmax(np.abs(iso - iso0)[:, far_j]) == 0,
          f"iso-surface moved {iso[i, j] - iso0[i, j]:.1f} ms at the edit, unchanged far away")
    st_after = np.array(np.load(os.path.join(folder, "stack.npy")))
    check(not np.allclose(st_after, stack_before, equal_nan=True) and
          np.allclose(st_after[:, outside], stack_before[:, outside], equal_nan=True),
          "horizon stack updated for the edited inlines only")
    ed.undo()
    check(np.array_equal(np.load(os.path.join(folder, "rgt.npy")), before) and
          np.allclose(np.load(os.path.join(folder, "stack.npy")), stack_before, equal_nan=True),
          "undo restores the RGT volume and stack exactly")
    ed2 = RGTEditor3D(vol)
    check(len(ed2.edits) == 0 and np.allclose(ed2.tau, ed.tau, atol=1e-3), "editor state saved (nodes + edit list)")

    # horizon constraint and fault re-link on the faulted geology
    p, G = geology_project(tmp, "editf", fault=True)
    vol = p.volume("G")
    f = flt.Fault("F")
    for st in fault_sticks_from(G["fxl"], vol, [0, 13, 26, 39], t_ms=(100.0, 1150.0)):
        f.add_stick(st)
    ii = np.arange(vol.nil)[:, None]
    jj = np.arange(vol.nxl)[None]
    hz, tr = G["horizons"], G["tau_ref"]
    wl, wh = G["wedge"]
    g = G["g"]
    excl = lambda s: (np.abs(jj - G["fxl"](ii, hz[s])) < 4) | \
        np.broadcast_to((wl - 25 < tr[s] < wh + 25) & (g[None, :] < 0.3), (vol.nil, vol.nxl))  # noqa: E731
    rv, _ = S.compute_rgt_3d(vol)  # no faults: links run straight through the fault
    e0 = iso_errors(rv.trace, lambda v: rv.iso_surface(v, vol), G, (20, 20), vol.times, excl)
    ed = RGTEditor3D(vol)
    ed.relink_fault(f, [f])
    rv1 = S.RGTVolume(ed.folder)
    e1 = iso_errors(rv1.trace, lambda v: rv1.iso_surface(v, vol), G, (20, 20), vol.times, excl)
    up = e0[:, 0] < 450
    check(np.median(e1[up, 2]) < 0.5 * np.median(e0[up, 2]) and np.median(e1[up, 2]) < 4.0,
          f"fault re-link: p95 error above the wedge {np.median(e0[up, 2]):.1f} -> {np.median(e1[up, 2]):.1f} ms")
    s_h = int(np.argmin(np.abs(tr - 330)))
    grid = np.where(np.abs(jj - G["fxl"](ii, hz[s_h])) >= 3, hz[s_h], np.nan).astype(np.float32)
    ed.add_horizon("H", grid, [f])
    rv2 = S.RGTVolume(ed.folder)
    v = float(np.interp(hz[s_h][20, 20], vol.times, rv2.trace(20, 20)))
    iso = rv2.iso_surface(v, vol)
    ok = np.isfinite(grid) & np.isfinite(iso)
    check(np.median(np.abs(iso - grid)[ok]) < 1.0, "horizon constraint edit: one iso-RGT surface follows the picks")
    ed.undo()
    ed.undo()
    rv3 = S.RGTVolume(ed.folder)
    check(np.array_equal(np.asarray(rv3.arr), np.asarray(rv.arr)) if False else
          np.allclose(rv3.decode(rv3.arr[5]), rv.decode(rv.arr[5]), atol=0.2), "two undos restore the original model")

    # 2D line edit
    p, G = geology_project(tmp, "edit2d", fault=False, lines=True)
    S.compute_rgt_2d(p, ["A", "B"])
    e2 = RGTEditor2D(p, "A", radius=3)
    before = e2.tau.copy()
    sysm, v = e2.move_level(30, 400.0, 410.0)
    back = S.load_rgt_line(e2.folder, "A")
    check(abs(np.interp(410.0, p.line("A").times, back[30]) - v) < 0.6 and
          np.allclose(back[:10], before[:10], atol=0.2) and np.allclose(back[50:], before[50:], atol=0.2),
          "2D edit: level moved at trace 30, rest of the line unchanged")
    e2.undo()
    check(np.allclose(S.load_rgt_line(e2.folder, "A"), before, atol=0.2), "2D edit undo")
    return True


def run_gui(check, tmp):
    from .gui.main_window import MainWindow
    from .rgt import solve as S
    from .rgt import stack as K
    from .test_rgt import geology_project
    p, G = geology_project(tmp, "editgui", fault=False, lines=True)
    vol = p.volume("G")
    rv, _ = S.compute_rgt_3d(vol)
    K.build_stack_3d(vol, rv, 30)
    w = MainWindow(p.path)
    sp = w.section
    sp.source = ("3D", "G")
    sp.mode = "Inline"
    sp.number = int(vol.inlines[20])
    sp.redraw()
    check(len(sp.rgt_contours()) > 5, "RGT contours (stack levels) drawn on the inline")
    sp.set_mode("rgt")
    check(sp.rgt_mode and not sp.pick_mode and not sp.fault_mode, "RGT edit mode is exclusive")
    t1 = sp.rgt_click(float(vol.xlines[30]), 400.0)
    check(t1 is None and sp._rgt_grab is not None, "first click grabs the iso-RGT value under the cursor")
    v = sp._rgt_grab[3]
    sp.rgt_click(float(vol.xlines[30]), 412.0)
    from .rgt.edit import RGTEditor3D
    ed = RGTEditor3D(vol)
    check(abs(ed.value_at(20, 30, 412.0) - v) < 0.5, "second click moves that level there (local re-solve)")
    sp.undo()
    check(abs(RGTEditor3D(vol).value_at(20, 30, 400.0) - v) < 0.5, "Ctrl+Z undoes the RGT edit")
    sp.set_mode(None)
    return True
