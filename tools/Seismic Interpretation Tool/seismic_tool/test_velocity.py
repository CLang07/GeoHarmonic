"""Velocity model / depth conversion checks (synthetic wells with known layer velocities)."""
import os

import numpy as np

LAT, LON = -39.50, 173.30


def velocity_case(tmp):
    """3 wells inside a 2 x 2 km 3D survey; layer 1 velocity varies west-east, layer 2 constant."""
    from .welltie import wc
    wc()
    from well_correlation.geo import tm_forward
    from .project import SeisProject
    from .synth import write_3d, traces_from_surfaces
    from .volume import import_3d
    d = os.path.join(tmp, "vel")
    os.makedirs(os.path.join(d, "wells_data"))
    x0, y0 = tm_forward(LAT, LON)
    x0, y0 = round(x0), round(y0)
    nil = nxl = 41
    v1 = lambda x, y: 2000.0 + 0.5 * (x - x0)  # noqa: E731  (2000 -> 2500 m/s across 1 km)
    V2 = 3000.0
    i, j = np.meshgrid(np.arange(nil), np.arange(nxl), indexing="ij")
    X, Y = x0 + 25.0 * j, y0 + 25.0 * i
    t1 = 600 + 0.05 * (X - x0) + 20 * np.sin((Y - y0) / 300.0)
    t2 = t1 + 300 + 0.03 * (Y - y0)
    z1 = v1(X, Y) * t1 / 2000.0
    z2 = z1 + V2 * (t2 - t1) / 2000.0
    wells = {"A-1": (5, 5), "B-1": (35, 8), "C-1": (20, 34)}
    lines = ["Well,UWI,Latitude,Longitude,Elevation_m,Elev_Ref,TD_m"]
    m_per_deg_lat = 111320.0
    for n, (ii, jj) in wells.items():
        os.makedirs(os.path.join(d, "wells_data", "wells", n))
        x, y = X[ii, jj], Y[ii, jj]
        # invert NZTM locally (small offsets)
        lat = LAT + (y - y0) / m_per_deg_lat
        lon = LON + (x - x0) / (m_per_deg_lat * np.cos(np.radians(LAT)))
        lines.append(f"{n},1,{lat},{lon},25.0,KB,3000")
    with open(os.path.join(d, "wells_data", "wells_header.csv"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    P = wc().project.Project
    wp = P.from_folder(os.path.join(d, "wells_data"))
    tops = {}
    pos = {}
    for n in wells:
        w = wp.wells[n]
        # exact grid position of the projected well, then true depths there
        fj, fi = (w.x - x0) / 25.0, (w.y - y0) / 25.0
        pos[n] = (fi, fj)
        tt1 = 600 + 0.05 * (w.x - x0) + 20 * np.sin((w.y - y0) / 300.0)
        tt2 = tt1 + 300 + 0.03 * (w.y - y0)
        zz1 = v1(w.x, w.y) * tt1 / 2000.0
        zz2 = zz1 + V2 * (tt2 - tt1) / 2000.0
        tops[n] = {"Top A": zz1 + 25.0, "Top B": zz2 + 25.0}
    wp.tops = tops
    wp.formations = [{"name": "Top A", "color": "#ff0000"}, {"name": "Top B", "color": "#0000ff"}]
    wpath = os.path.join(d, "wells_data", "vel.wcproj.json")
    wp.save(wpath)
    ns = 200
    cube = traces_from_surfaces([t1, t2], [1.0, -1.0], ns, 0.008, noise=0.0)
    write_3d(os.path.join(d, "v.sgy"), cube, dt_us=8000, x0=x0, y0=y0)
    store = os.path.join(d, "v_seismic", "V")
    import_3d(os.path.join(d, "v.sgy"), store)
    sp = SeisProject()
    sp.path = os.path.join(d, "v.seisproj.json")
    sp.name = "v"
    sp.well_project = wpath
    sp.volumes.append({"name": "V", "store": store, "segy": ""})
    for name, g in (("H1", t1), ("H2", t2)):
        h = sp.horizons.new(name)
        h.data["3D:V"] = g.astype(np.float32)
    sp.save()
    return sp, {"z1": z1, "z2": z2, "t1": t1, "t2": t2, "v1": v1, "X": X, "Y": Y}


def run(check, tmp):
    from .velocity import VelocityModel
    sp, T = velocity_case(tmp)
    vm = VelocityModel(sp, "3D:V", [("H1", "Top A"), ("H2", "Top B")]).build()
    wt = vm.grids()["wells"]
    check(len(wt) == 3, f"wells inside the model: {[r['well'] for r in wt]}")
    v_err = max(abs(r["vint"][1] - 3000.0) for r in wt)
    check(v_err < 5.0, f"interval velocity of layer 2 from tops + horizon times ({v_err:.1f} m/s error)")
    mt = vm.misfit_table()
    res = np.array([m[5] for m in mt])
    check(np.max(np.abs(res)) < 3.0, f"well misfit small without correction (max {np.max(np.abs(res)):.2f} m)")
    d2 = vm.depth_grid(1)
    err = np.abs(d2.z - T["z2"])
    check(np.median(err) / np.median(T["z2"]) < 0.02, f"depth of H2 everywhere: median error {np.median(err):.1f} m "
                                                     f"({np.median(err) / np.median(T['z2']):.1%}) with 3 wells")
    vmr = VelocityModel(sp, "3D:V", [("H1", "Top A"), ("H2", "Top B")], residual_correction=True).build()
    res_r = np.array([m[5] for m in vmr.misfit_table()])
    check(np.max(np.abs(res_r)) < 0.5, f"residual correction ties the wells (max {np.max(np.abs(res_r)):.2f} m)")
    # depth conversion of a section: the H2 reflector lands at the model depth of H2
    vol = sp.volume("V")
    il = int(vol.inlines[20])
    sec = vol.inline(il).T
    xs, ys = vol.xy(np.full(vol.nxl, il), vol.xlines)
    dsec, z = vm.section_to_depth(sec, xs, ys, vol.times, dz=2.0)
    jj = 12
    k = np.nanargmin(dsec[:, jj])  # H2 is the trough
    zm = float(d2.sample([xs[jj]], [ys[jj]])[0])
    check(abs(z[k] - zm) < 6.0, f"depth section: trough at {z[k]:.0f} m, model H2 {zm:.0f} m")
    vm2 = VelocityModel.from_dict(sp, vm.to_dict())
    check(vm2.layers == vm.layers and vm2.key == "3D:V", "model settings round trip")
    return sp


def run_gui(check, tmp):
    from .gui.velocity_dialog import VelocityDialog
    from .gui.section_panel import SectionPanel
    sp, T = velocity_case(tmp)
    dlg = VelocityDialog(sp)
    vm = dlg.build("3D:V", [("H1", "Top A"), ("H2", "Top B")], False)
    check(vm is not None and len(vm.misfit_table()) == 6, "velocity dialog: model + misfit table")
    g = dlg.show_depth_map(1)
    check(g.unit == "m", "velocity dialog: depth map")
    dlg.use()
    check(sp.display.get("velocity_model", {}).get("survey") == "3D:V", "velocity model stored for depth display")
    p = dlg.write_misfit(os.path.join(tmp, "misfit.csv"))
    check(sum(1 for _ in open(p)) == 7, "misfit CSV")
    panel = SectionPanel()
    panel.project = sp
    panel.source = ("3D", "V")
    vol = sp.volume("V")
    panel.mode = "Inline"
    panel.number = int(vol.inlines[20])
    panel.domain = "Depth"
    panel.redraw()
    check(panel._data is not None and panel._yaxis[-1] > 1000 and len(panel._hz_items) >= 2,
          "section in depth with horizons converted")
    check(panel.trace_at(float(vol.xlines[3])) is None, "no picking in the depth domain")
    ov = panel.well_overlay()
    panel.domain = "Time"
    return True
