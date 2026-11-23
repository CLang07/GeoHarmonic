"""Horizon maps: structure, isochron, amplitude extraction, depth conversion, contours, XYZ / CPS-3 / PNG."""
import os

import numpy as np


def run(check, tmp):
    from . import maps as M
    from .horizons import Horizon
    from .synth import make_project
    d = os.path.join(tmp, "maps")
    os.makedirs(d)
    proj, truth = make_project(d, nil=30, nxl=40, ns=250, noise=0.0)
    vol = proj.volume("Synth")
    top, deep = truth["3D:Synth"], truth["deep"]
    h1, h2 = Horizon("Top"), Horizon("Deep")
    h1.data["3D:Synth"] = top.astype(np.float32)
    h2.data["3D:Synth"] = deep.astype(np.float32)
    g1 = M.horizon_grid(proj, h1, "3D:Synth")
    g2 = M.horizon_grid(proj, h2, "3D:Synth")
    iso = M.isochron(g1, g2)
    check(np.allclose(iso.z, deep - top), "isochron = base - top")
    X, Y = g1.node_xy()
    check(np.allclose(g1.sample(X[7:9, 11], Y[7:9, 11]), top[7:9, 11], atol=1e-6), "map sampling at node positions")
    # amplitude extraction vs direct computation on the cube
    cube = np.asarray(vol.ilxl, np.float32) * vol.inv
    t = vol.times
    rms = M.extract_amplitude(vol, top, 8.0, 12.0, "rms", block=7)
    i, j = 13, 17
    m = (t >= top[i, j] - 8.0) & (t <= top[i, j] + 12.0)
    check(abs(rms[i, j] - np.sqrt(np.mean(cube[i, j, m] ** 2))) < 1e-4, "RMS amplitude in a window")
    mx = M.extract_amplitude(vol, top, 4.0, 4.0, "max")
    at = M.extract_amplitude(vol, top, kind="at")
    check(np.allclose(mx[i, j], cube[i, j, m & (t <= top[i, j] + 4) & (t >= top[i, j] - 4)].max()) and
          abs(at[i, j] - np.interp(top[i, j], t, cube[i, j])) < 1e-4, "max and at-horizon extraction")
    check(np.nanmedian(at) > 0.6, f"at-horizon amplitude on the peak ({np.nanmedian(at):.2f})")
    # contours of a plane: straight lines at the right place
    plane = M.MapGrid(np.add.outer(np.zeros(30), np.arange(40) * 2.0), "ilxl", "ms", "plane", vol)
    segs = plane.contours([31.0])
    xs, ys = segs[31.0]
    il, xl = zip(*[vol.ilxl_at(x, y) for x, y in zip(xs, ys)])
    check(np.allclose(xl, vol.xl0 + 15.5, atol=1e-6), "contour of a plane lies on the expected crossline")
    # export round trips
    pxyz = M.write_xyz(g1, os.path.join(d, "top.xyz"))
    a = np.loadtxt(pxyz)
    check(a.shape == (30 * 40, 3) and np.allclose(a[:, 2], top.ravel(), atol=1e-3), "XYZ export")
    pc = M.write_cps3(g1, os.path.join(d, "top.cps"))
    back = M.read_cps3(pc)
    gx = g1.to_xy()
    check(np.allclose(back.z, gx.z, equal_nan=True, atol=1e-4) and abs(back.x0 - gx.x0) < 1e-3 and abs(back.dx - gx.dx) < 1e-4,
          "CPS-3 export reads back (axis-aligned resample)")
    inside = np.isfinite(gx.z)
    Xg, Yg = gx.node_xy()
    check(inside.mean() > 0.99 and np.nanmax(np.abs(gx.z[inside] - g1.sample(Xg[inside], Yg[inside]))) < 1e-6,
          "XY resample of the inline/crossline grid")
    pp = M.export_png(g1, os.path.join(d, "top.png"))
    check(os.path.getsize(pp) > 5000, "PNG map export")
    # 2D gridding
    h3 = Horizon("Lines")
    for name in ("L-A", "L-B"):
        h3.data[f"2D:{name}"] = truth[f"2D:{name}"]
    g2d = M.horizon_grid(proj, h3, "2D", cell=50.0, reach=200.0)
    la = proj.line("L-A")
    v = g2d.sample(la.x[10:100:10], la.y[10:100:10])
    check(np.nanmax(np.abs(v - truth["2D:L-A"][10:100:10])) < 3.0 and np.isnan(g2d.sample([la.x[10]], [la.y[10] + 2000])[0]),
          "2D picks gridded (honours the lines, blank far away)")

    # depth conversion with well velocities (synthetic wells from test_welltie)
    from .test_welltie import well_project, seismic_project, true_twt
    wpath, xy = well_project(d)
    sp = seismic_project(d, wpath, xy)
    v2 = sp.volume("S")
    ws = sp.wellset()
    z_true = 1500.0 - 25.0  # Sand A TVDSS at W-1
    tz = true_twt(z_true)
    hz = Horizon("Sand A t")
    hz.data["3D:S"] = np.full((v2.nil, v2.nxl), tz, np.float32)
    sp.horizons.items.append(hz)
    tg = M.horizon_grid(sp, hz, "3D:S")
    pts = M.well_velocity_points(ws, tg)
    dg = M.depth_convert(tg, pts)
    zz = dg.sample([xy[0]], [xy[1]])[0]
    check(len(pts) == 1 and pts[0][0] == "W-1" and abs(zz - z_true) < 1.0,
          f"depth conversion with the well's average velocity ({pts[0][3]:.0f} m/s): {zz:.1f} m vs {z_true:.1f} m")
    pts2 = M.well_velocity_points(ws, tg, formation="Sand A")
    check(abs(pts2[0][5] - z_true) < 1e-6, "depth calibrated to the formation top at the well")
    return True


def run_gui(check, tmp):
    from .gui.map_dialog import MapDialog
    from .gui.basemap import Basemap
    from .horizons import Horizon
    from .synth import make_project
    d = os.path.join(tmp, "maps_gui")
    os.makedirs(d)
    proj, truth = make_project(d, nil=20, nxl=25, ns=200, noise=0.0)
    for nm, k in (("Top", "3D:Synth"), ("Deep", "deep")):
        h = proj.horizons.new(nm)
        h.data["3D:Synth"] = truth[k].astype(np.float32)
    bm = Basemap()
    bm.project = proj
    dlg = MapDialog(proj, bm)
    g = dlg.compute(key="3D:Synth", product="Isochron", h1="Top", h2="Deep")
    check(np.allclose(g.z, truth["deep"] - truth["3D:Synth"]), "map dialog: isochron")
    g = dlg.compute(key="3D:Synth", product="Amplitude", h1="Top", amp="rms", above=4.0, below=4.0)
    check(g.unit == "amp" and np.isfinite(g.z).all(), "map dialog: RMS amplitude map")
    bm.show_grid(g)
    p = dlg.export("cps3", os.path.join(d, "a.cps"))
    check(os.path.exists(p), "map dialog: CPS-3 export")
    return True
