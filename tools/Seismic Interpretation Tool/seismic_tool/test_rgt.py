"""RGT step B checks: node mesh, links, CGLS solve, iso-RGT surfaces vs the true horizons (3D + 2D)."""
import os

import numpy as np


def geology_project(tmp, sub, fault=True, lines=False):
    """Project with the synthetic geology volume 'G' (and optionally 2 crossing 2D lines cut from it)."""
    from .project import SeisProject
    from .selftest import write_segy
    from .synth import rgt_geology, write_3d
    from .volume import import_3d
    d = os.path.join(tmp, sub)
    os.makedirs(d)
    G = rgt_geology(fault=fault)
    write_3d(os.path.join(d, "g.sgy"), G["cube"])
    store = os.path.join(d, "g_seismic", "G")
    import_3d(os.path.join(d, "g.sgy"), store)
    p = SeisProject()
    p.path = os.path.join(d, "g.seisproj.json")
    p.name = "g"
    p.volumes.append({"name": "G", "store": store, "segy": ""})
    if lines:
        vol = p.volume("G")
        for name, sl, ils, xls in (("A", (20, slice(None)), np.full(vol.nxl, 20), np.arange(vol.nxl)),
                                   ("B", (slice(None), 45), np.arange(vol.nil), np.full(vol.nil, 45))):
            tr = G["cube"][sl]
            x, y = vol.xy(vol.inlines[ils], vol.xlines[xls])
            pth = os.path.join(d, name + ".sgy")
            write_segy(pth, tr, {21: np.arange(1, len(tr) + 1), 181: np.round(x).astype(np.int64),
                                 185: np.round(y).astype(np.int64)})
            p.lines.append({"name": name, "segy": pth, "cdp_byte": 21})
    p.save()
    return p, G


def iso_errors(rv_trace_fn, iso_fn, G, ref, times, exclude):
    """For every true horizon: RGT value at the reference trace -> iso surface -> |error| stats (ms)."""
    hz, tr = G["horizons"], G["tau_ref"]
    tau0 = rv_trace_fn(*ref)
    out = []
    for s in range(len(tr)):
        th = hz[s][ref]
        if th < 60 or th > 1140:
            continue
        iso = iso_fn(float(np.interp(th, times, tau0)))
        ok = np.isfinite(iso) & ~exclude(s) & (hz[s] > 60) & (hz[s] < 1140)
        if ok.sum() < 10:
            continue
        e = np.abs(iso - hz[s])[ok]
        out.append((tr[s], np.median(e), np.percentile(e, 95)))
    return np.array(out)


def run(check, tmp):
    from . import faults as flt
    from .horizons import Horizon
    from .rgt import solve as S
    from .rgt.dip import compute_dip_volume
    from .synth import fault_sticks_from

    # ---- CGLS on a tiny known system
    A = np.array([[2.0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 3, 1]])
    xt = np.array([1.0, -2.0, 0.5])
    r, c = np.nonzero(A)
    x, it = S.cgls(r, c, A[r, c], A @ xt, np.zeros(3), 3, 200, 1e-12, dx_tol=0)
    check(np.allclose(x, xt, atol=1e-8), f"CGLS solves a small least-squares system ({it} iterations)")

    # ---- 3D without fault: folded layers with a pinch-out
    p, G = geology_project(tmp, "rgt", fault=False)
    vol = p.volume("G")
    compute_dip_volume(vol)
    rv, sysm = S.compute_rgt_3d(vol)
    check(sysm.stats["iterations"] > 0 and rv.arr.shape == (vol.nil, vol.nxl, vol.ns), f"RGT solved {sysm.stats}")
    tau = rv.decode(rv.arr[10, 30])
    check(np.all(np.diff(tau) >= 0), "RGT increases monotonically down each trace")
    wl, wh = G["wedge"]
    g = G["g"]
    tr = G["tau_ref"]
    excl = lambda s: np.broadcast_to((wl - 25 < tr[s] < wh + 25) & (g[None, :] < 0.3), (vol.nil, vol.nxl))  # noqa
    e = iso_errors(rv.trace, lambda v: rv.iso_surface(v, vol), G, (20, 50), vol.times, excl)
    check(len(e) > 40 and np.median(e[:, 1]) < 1.0 and np.mean(e[:, 2] < 4.0) >= 0.9,
          f"iso-RGT surfaces match {len(e)} true horizons: median {np.median(e[:, 1]):.2f} ms, "
          f"{np.mean(e[:, 2] < 4.0):.0%} with p95 < 1 sample (pinch-out zone excluded)")

    # coarse-to-fine start (used automatically on big surveys) gives the same answer
    from .rgt.dip import DipField
    sm = S.solve_3d_multires(vol, DipField(os.path.join(vol.store, "rgt")), S.RGTParams(), min_unknowns=5000)
    rvm = S.write_rgt_3d(vol, sm, os.path.join(vol.store, "rgt_multires"))
    em = iso_errors(rvm.trace, lambda v: rvm.iso_surface(v, vol), G, (20, 50), vol.times, excl)
    check(sm.stats["coarse_start"] and np.median(em[:, 1]) < 1.0 and np.mean(em[:, 2] < 4.0) >= 0.9,
          f"coarse-to-fine solve: median {np.median(em[:, 1]):.2f} ms in {sm.stats['iterations']} fine iterations")

    # ---- 3D with a fault: blocks tied across it
    p, G = geology_project(tmp, "rgtf", fault=True)
    vol = p.volume("G")
    compute_dip_volume(vol)
    f = flt.Fault("F")
    for st in fault_sticks_from(G["fxl"], vol, [0, 13, 26, 39], t_ms=(100.0, 1150.0)):
        f.add_stick(st)
    ii = np.arange(vol.nil)[:, None]
    jj = np.arange(vol.nxl)[None]
    hz = G["horizons"]
    tr = G["tau_ref"]
    excl_f = lambda s: (np.abs(jj - G["fxl"](ii, hz[s])) < 4) | \
        np.broadcast_to((wl - 25 < tr[s] < wh + 25) & (g[None, :] < 0.3), (vol.nil, vol.nxl))  # noqa: E731
    rv, sysm = S.compute_rgt_3d(vol, faults=[f])
    e = iso_errors(rv.trace, lambda v: rv.iso_surface(v, vol), G, (20, 20), vol.times, excl_f)
    upper = e[:, 0] < 450  # where the hanging wall is wide
    check(np.median(e[:, 1]) < 1.0 and np.all(e[upper, 2] < 4.0),
          f"with a fault: iso-RGT surfaces match across the fault above the wedge (p95 max "
          f"{e[upper, 2].max():.1f} ms), median {np.median(e[:, 1]):.2f} ms overall")
    rv0, _ = S.compute_rgt_3d(vol, S.RGTParams(fault_links="none"), faults=[f],
                              out_dir=os.path.join(vol.store, "rgt_nolinks"))
    e0 = iso_errors(rv0.trace, lambda v: rv0.iso_surface(v, vol), G, (20, 20), vol.times, excl_f)
    check(np.median(e0[:, 2]) > 10, f"(without cross-fault links the hanging wall is off: median p95 "
                                    f"{np.median(e0[:, 2]):.0f} ms)")
    # horizon constraints tie the blocks when cross-fault links are off
    s_h = int(np.argmin(np.abs(tr - 330)))
    h = Horizon("Constraint")
    h.data["3D:G"] = np.where(np.abs(jj - G["fxl"](ii, hz[s_h])) >= 3, hz[s_h], np.nan).astype(np.float32)
    rv1, sysm1 = S.compute_rgt_3d(vol, S.RGTParams(fault_links="none"), faults=[f], horizons=[h], vol_name="G",
                                  out_dir=os.path.join(vol.store, "rgt_constrained"))
    iso = rv1.iso_surface(float(np.interp(hz[s_h][20, 20], vol.times, rv1.trace(20, 20))), vol)
    okc = np.isfinite(h.data["3D:G"]) & np.isfinite(iso)
    check(np.median(np.abs(iso - hz[s_h])[okc]) < 1.0, "horizon constraint: its iso-RGT surface follows the picks "
                                                       "on both sides of the fault")
    e1 = iso_errors(rv1.trace, lambda v: rv1.iso_surface(v, vol), G, (20, 20), vol.times, excl_f)
    near = np.abs(e1[:, 0] - tr[s_h]) < 60
    check(np.median(e1[near, 2]) < np.median(e0[near, 2]) / 2,
          f"constraint pulls neighbouring horizons into place (p95 {np.median(e0[near, 2]):.0f} -> "
          f"{np.median(e1[near, 2]):.1f} ms)")
    rv2 = S.RGTVolume(os.path.join(vol.store, "rgt"))
    check(rv2.section(vol, "Inline", int(vol.inlines[3])).shape == (vol.ns, vol.nxl) and
          os.path.exists(os.path.join(vol.store, "rgt", "rgt_nodes.npz")), "RGT volume reopens; node solution saved")

    # ---- 2D network: two crossing lines tied
    p, G = geology_project(tmp, "rgt2d", fault=False, lines=True)
    out, sysm = S.compute_rgt_2d(p, ["A", "B"])
    check(len(sysm.ties) == 1, f"2D tie found {sysm.ties}")
    a, ia, b, ib = sysm.ties[0]
    la, lb = p.line(a), p.line(b)
    ta, tb = out[a][ia], out[b][ib]
    check(np.median(np.abs(ta - tb)[20:-20]) < 1.0, "RGT equal on both lines at the tie")
    times = la.times
    hz = G["horizons"]
    errs = []
    for s in range(len(tr)):
        th = hz[s][20, 50]
        if th < 60 or th > 1140 or (wl - 25 < tr[s] < wh + 25):
            continue
        v = float(np.interp(th, times, out["A"][50]))
        for name, truth in (("A", hz[s][20, :]), ("B", hz[s][:, 45])):
            iso = S.iso_from_traces(out[name], v, times)
            ok = np.isfinite(iso)
            errs.append(np.median(np.abs(iso - truth)[ok]))
    check(np.median(errs) < 1.0 and np.percentile(errs, 90) < 4.0,
          f"2D iso-RGT horizons match on both lines (median {np.median(errs):.2f} ms, p90 {np.percentile(errs, 90):.2f})")
    folder = os.path.join(p.seismic_dir(), "rgt")
    back = S.load_rgt_line(folder, "B")
    check(back is not None and np.max(np.abs(back - out["B"])) < 0.1, "2D RGT saved per line (uint16) and reloaded")
    return p


def run_gui(check, tmp):
    from .gui.section_panel import SectionPanel
    from .rgt import solve as S
    p, G = geology_project(tmp, "rgtgui", fault=False, lines=True)
    vol = p.volume("G")
    panel = SectionPanel()
    panel.project = p
    panel.source = ("3D", "G")
    panel.mode = "Inline"
    panel.number = int(vol.inlines[5])
    panel.set_attr("RGT")
    check(panel._data is None, "RGT display before computing: message, nothing drawn")
    S.compute_rgt_3d(vol)
    panel.set_attr("RGT")
    check(panel._data is not None and np.all(np.diff(panel._data[:, 10]) >= 0), "panel shows the RGT section")
    S.compute_rgt_2d(p, ["A", "B"])
    panel.source = ("2D", "A")
    panel.mode = "2D"
    panel.set_attr("RGT")
    check(panel._data is not None and panel._data.shape == (vol.ns, vol.nxl), "panel shows 2D RGT")
    panel.set_attr("Seismic")
    return True
