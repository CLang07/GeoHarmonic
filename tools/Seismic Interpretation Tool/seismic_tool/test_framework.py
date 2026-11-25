"""Structural framework + 3D grid tests (PLAN 35): synthetic faulted layer-cake."""
import os

import numpy as np

from . import framework as FW

THROW = 60.0
FX = 1010.0  # fault x (between grid nodes)


def truth(x, y):
    """Depths of H1 / H2 / H3: east of x = 1000 down-thrown by THROW; zones 100 m and 80 m thick."""
    h1 = 1000.0 + 0.05 * x + 0.02 * y + np.where(x > FX, THROW, 0.0)
    return h1, h1 + 100.0, h1 + 180.0


def case(seed=1, bad=True):
    rng = np.random.default_rng(seed)
    px, py = rng.uniform(0, 2000, 300), rng.uniform(0, 1500, 300)
    keep = np.abs(px - FX) > 30.0  # no data right on the fault
    px, py = px[keep], py[keep]
    h1, h2, h3 = truth(px, py)
    h2b = h2.copy()
    if bad:
        h2b[:5] = h1[:5] - 20.0  # bad picks: H2 above H1 at five points -> conformable rule fixes the crossing
    geom = FW.Geometry(0.0, 0.0, 50.0, 50.0, 40, 30)
    fault = ("F1", (np.array([FX, FX]), np.array([-100.0, 1600.0])))
    hz = [("H1", (px, py, h1), "conformable"), ("H2", (px, py, h2b), "conformable"), ("H3", (px, py, h3), "conformable")]
    return FW.FrameworkModel(geom, hz, [fault], method="tps", domain="depth").build()


def run(check, tmp):
    from .faults import Fault
    from .maps import MapGrid
    m = case()
    X, Y = m.X, m.Y
    t1, t2, t3 = truth(X, Y)
    check(m.nblocks == 2 and len(np.unique(m.blocks[:, :20])) == 1 and m.blocks[0, 0] != m.blocks[0, 39],
          "35b: two fault blocks either side of the fault")
    err = [float(np.nanmedian(np.abs(g - t))) for g, t in zip(m.grids, (t1, t2, t3))]
    check(max(err) < 1.0, f"35b: horizons gridded per fault block (median errors {np.round(err, 2).tolist()} m)")
    jw, je = 19, 21  # nodes x = 950 / 1050
    jump = float(np.median(m.grids[0][:, je] - m.grids[0][:, jw])) - 0.05 * 100.0  # minus the regional dip
    check(abs(jump - THROW) < 3.0, f"35b: fault-block throw honoured at the fault ({jump:.1f} m, true {THROW})")
    check(m.report["crossings_after"] == [0, 0], f"35b: no crossing horizons after the rules (before "
          f"{m.report['crossings_before']})")
    # erosional rule truncates older horizons
    g = [a.copy() for a in m.grids]
    m2 = FW.FrameworkModel(m.geom, [("U", (np.array([0, 2000, 0, 2000.0]), np.array([0, 0, 1500, 1500.0]),
                                           np.full(4, 1110.0)), "erosional"),
                                    ("H2", m.horizons[1][1], "conformable")], [], "idw", "depth").build()
    check(np.all(m2.grids[1] >= m2.grids[0] - 1e-9) and np.any(m2.grids[1] == 1110.0),
          "35b: erosional surface truncates the horizon below it")
    del g
    # 35a: fault trace on a horizon + stick editing
    f = Fault("F1")
    for y in (0.0, 750.0, 1500.0):
        f.add_stick(np.array([[1000.0, y, 900.0], [1000.0, y, 1300.0]]))
    gx = np.arange(0, 2001, 50.0)
    gy = np.arange(0, 1501, 50.0)
    GX, GY = np.meshgrid(gx, gy)
    hg = MapGrid(1000.0 + 0.05 * GX + 0.02 * GY, "xy", "m", "H1 west", None, 0.0, 0.0, 50.0, 50.0)
    tr = FW.fault_trace(f, hg)
    check(len(tr) >= 3 and np.allclose(tr[:, 0], 1000.0, atol=1e-6) and np.ptp(tr[:, 1]) > 1000,
          "35a: fault trace where the fault surface cuts the horizon")
    f.add_stick(np.array([[1000.0, 300.0, 900.0], [1010.0, 300.0, 1100.0], [1000.0, 300.0, 1300.0]]))
    FW.smooth_sticks(f)
    n0 = len(f.sticks)
    FW.remove_stick(f, 1)
    check(len(f.sticks) == n0 - 1 and len(f.triangles()) > 0, "35a: fault editing - smooth, remove a stick")
    # 35c: depth through a model
    class VM:
        def depth_of_time(self, x, y, times):
            return np.asarray(times, float)[None, :] * 1.25 + 0.0 * np.asarray(x, float)[:, None]
    mt = FW.FrameworkModel(m.geom, [(n, (d[0], d[1], d[2] / 1.25), r) for n, d, r in m.horizons], m.faults, "tps",
                           "time").build()
    md = mt.to_depth(FW.depth_fn_from_model(VM()))
    check(md.domain == "depth" and float(np.nanmedian(np.abs(md.grids[0] - t1))) < 1.0,
          "35c: framework converted to depth through a velocity model")
    # 35d: 3D grid
    G = FW.Grid3D(case(bad=False), [("proportional", 5), ("top", 20.0)])
    q = G.quality()
    check((G.nz, G.ny, G.nx) == (9, 30, 40) and q["layers_per_zone"] == [5, 4] and q["cells"] == 9 * 30 * 40,
          f"35d: grid {G.nx} x {G.ny} x {G.nz}, layers per zone {q['layers_per_zone']}")
    z1 = G.zone_thickness(1)
    z2 = G.zone_thickness(2)
    check(abs(float(np.median(z1)) - 100.0) < 1.0 and abs(float(np.median(z2)) - 80.0) < 1.0,
          f"35d: zone thicknesses {np.median(z1):.1f} / {np.median(z2):.1f} m (true 100 / 80)")
    top = G.zcorn[0].reshape(G.ny, 2, G.nx, 2)
    west, east = top[:, :, 19, 1], top[:, :, 20, 0]  # pillar x = 1000 seen from the cells either side of it
    check(abs(float(np.median(east - west)) - THROW) < 3.0 and len(np.unique(G.block)) == 2,
          f"35d: stair-stepped fault - the shared pillar has two depths ({np.median(east - west):.1f} m apart)")
    check(q["negative_thickness"] == 0 and q["active"] == int(G.actnum.sum()) and q["volume_m3"] > 0,
          "35d: quality report (no negative cells, active cells, volume)")
    # 35e: GRDECL round trip + surfaces
    p = os.path.join(tmp, "fw.grdecl")
    FW.write_grdecl(p, G, props={"PORO": np.full((G.nz, G.ny, G.nx), 0.2)})
    r = FW.read_grdecl(p)
    check(r["SPECGRID"][:3].tolist() == [40, 30, 9] and np.allclose(r["ZCORN"], G.zcorn.ravel(), atol=1e-3) and
          np.allclose(r["COORD"], G.coord().ravel(), atol=1e-3) and np.array_equal(r["ACTNUM"], G.actnum.ravel()) and
          np.array_equal(r["ZONE"], G.zone_index().ravel()) and len(r["PORO"]) == G.nz * G.ny * G.nx,
          "35e: GRDECL written and read back (COORD, ZCORN, ACTNUM, ZONE, FAULTBLK, props)")
    paths = FW.write_surfaces(m, os.path.join(tmp, "fw_surf"))
    from .gis.zmap import read_zmap
    back = read_zmap(paths[0])
    check(len(paths) == 3 and np.allclose(back.z, m.grids[0], atol=1e-3, equal_nan=True), "35e: surfaces to ZMAP+")
    xyz = FW.write_surfaces(m, os.path.join(tmp, "fw_xyz"), fmt="xyz")
    check(os.path.getsize(xyz[0]) > 1000, "35e: surfaces to XYZ")
    return G


def run_gui(check, tmp):
    """35 GUI: framework dialog on a synthetic survey with two horizons and a fault -> depth -> grid -> GRDECL."""
    from .gui.main_window import MainWindow
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "fw_gui")
    vol = proj.volume("Synth")
    f = proj.faults.new("F1")
    for il in (vol.inlines[0], vol.inlines[-1]):
        a = vol.xy(float(il), float(vol.xlines[12]))
        f.add_stick(np.array([[a[0], a[1], 100.0], [a[0] + 1.0, a[1], 900.0]]))
    proj.save()
    mw = MainWindow(proj.path)
    dlg = mw.framework("Synth")
    m = dlg.build()
    check(m.nblocks == 2 and len(m.grids) == 2, f"35 GUI: framework from the picks with the fault as a barrier "
          f"({m.nblocks} blocks)")

    class VM:
        def depth_of_time(self, x, y, times):
            return np.asarray(times, float)[None, :] * 1.1 + 0.0 * np.asarray(x, float)[:, None]
    dlg.depth_model_fn = lambda: VM()
    md = dlg.depth()
    check(md is not None and md.domain == "depth" and np.nanmedian(md.grids[0] / m.grids[0]) > 1.09,
          "35c GUI: framework converted with the current depth model")
    q = dlg.make_grid([("proportional", 4)])
    check(q["cells"] == dlg.grid.nx * dlg.grid.ny * 4 and q["negative_thickness"] == 0, "35d GUI: 3D grid built")
    out = dlg.export(os.path.join(d, "synth.grdecl"), os.path.join(d, "surf"))
    check(os.path.exists(out[0]) and len(out) == 3, "35e GUI: GRDECL + surfaces exported")
    dlg.depth_model_fn = lambda: type("VM2", (), {"depth_of_time": lambda self, x, y, t: np.asarray(t, float)[None, :]
                                                  * 1.3 + 0.0 * np.asarray(x, float)[:, None]})()
    mw.mapbus.notify("velocities", source="seismic")
    check(np.nanmedian(dlg.model.grids[0] / m.grids[0]) > 1.29 and dlg.grid is not None,
          "35c GUI: depth framework + grid rebuilt when the velocity model changes")
    mw.maplink.close()
