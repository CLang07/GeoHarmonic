"""Dynamic depth conversion (item 17) checks: velocity functions per layer on synthetic V0 + kZ / V0 + kT earths."""
import os

import numpy as np

LAT, LON = -39.50, 173.30
NIL = NXL = 41
DX = 25.0
WELLS6 = {"W-1": (4, 5), "W-2": (36, 6), "W-3": (20, 20), "W-4": (6, 35), "W-5": (34, 33), "W-6": (12, 22)}

# layer laws of the V0 + kZ earth (different per layer)
KZ = [(1700.0, 0.55), (2300.0, 0.35)]
# V0 + kT earth
KT = [(1650.0, 900.0), (2100.0, 700.0)]


def t_surfaces(x, y, x0, y0):
    t1 = 500 + 0.3 * (x - x0) + 25 * np.sin((y - y0) / 260.0)
    t2 = t1 + 280 + 0.12 * (y - y0) + 0.05 * (x - x0)
    return t1, t2


def kz_earth(t1, t2):
    (a1, k1), (a2, k2) = KZ
    z1 = (0.0 + a1 / k1) * np.exp(k1 * t1 / 2000.0) - a1 / k1
    z2 = (z1 + a2 / k2) * np.exp(k2 * (t2 - t1) / 2000.0) - a2 / k2
    return z1, z2


def kz_layer1_depth(t):
    a1, k1 = KZ[0]
    return a1 / k1 * np.expm1(k1 * np.asarray(t, float) / 2000.0)


def kt_earth(t1, t2):
    (a1, k1), (a2, k2) = KT
    b1, b2 = t1 / 2000.0, t2 / 2000.0
    z1 = a1 * b1 + 0.5 * k1 * b1 ** 2
    z2 = z1 + a2 * (b2 - b1) + 0.5 * k2 * (b2 ** 2 - b1 ** 2)
    return z1, z2


def ddc_case(tmp, name, earth, wells=None):
    """3D survey (41 x 41, 25 m) with two horizons and wells whose tops follow `earth(t1, t2) -> (z1, z2)`."""
    from .welltie import wc
    wc()
    from well_correlation.geo import tm_forward
    from .project import SeisProject
    from .synth import write_3d, traces_from_surfaces
    from .volume import import_3d
    wells = wells or WELLS6
    d = os.path.join(tmp, name)
    os.makedirs(os.path.join(d, "wells_data"))
    x0, y0 = tm_forward(LAT, LON)
    x0, y0 = round(x0), round(y0)
    i, j = np.meshgrid(np.arange(NIL), np.arange(NXL), indexing="ij")
    X, Y = x0 + DX * j, y0 + DX * i
    t1, t2 = t_surfaces(X, Y, x0, y0)
    z1, z2 = earth(t1, t2)
    lines = ["Well,UWI,Latitude,Longitude,Elevation_m,Elev_Ref,TD_m"]
    m_per_deg_lat = 111320.0
    for n, (ii, jj) in wells.items():
        os.makedirs(os.path.join(d, "wells_data", "wells", n))
        x, y = X[ii, jj], Y[ii, jj]
        lat = LAT + (y - y0) / m_per_deg_lat
        lon = LON + (x - x0) / (m_per_deg_lat * np.cos(np.radians(LAT)))
        lines.append(f"{n},1,{lat},{lon},25.0,KB,4000")
    with open(os.path.join(d, "wells_data", "wells_header.csv"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    P = wc().project.Project
    wp = P.from_folder(os.path.join(d, "wells_data"))
    tops = {}
    for n in wells:
        w = wp.wells[n]
        tt1, tt2 = t_surfaces(np.float64(w.x), np.float64(w.y), x0, y0)
        zz1, zz2 = earth(tt1, tt2)
        tops[n] = {"Top A": float(zz1) + 25.0, "Top B": float(zz2) + 25.0}
    wp.tops = tops
    wp.formations = [{"name": "Top A", "color": "#ff0000"}, {"name": "Top B", "color": "#0000ff"}]
    wpath = os.path.join(d, "wells_data", f"{name}.wcproj.json")
    wp.save(wpath)
    ns = 250
    cube = traces_from_surfaces([t1, t2], [1.0, -1.0], ns, 0.008, noise=0.0)
    write_3d(os.path.join(d, "v.sgy"), cube, dt_us=8000, x0=x0, y0=y0)
    store = os.path.join(d, "v_seismic", "V")
    import_3d(os.path.join(d, "v.sgy"), store)
    sp = SeisProject()
    sp.path = os.path.join(d, f"{name}.seisproj.json")
    sp.name = name
    sp.well_project = wpath
    sp.volumes.append({"name": "V", "store": store, "segy": ""})
    for hn, g in (("H1", t1), ("H2", t2)):
        h = sp.horizons.new(hn)
        h.data["3D:V"] = g.astype(np.float32)
    sp.save()
    return sp, {"t1": t1, "t2": t2, "z1": z1, "z2": z2, "X": X, "Y": Y}


class _TD:
    """Stand-in well time-depth function (twt ms -> TVDSS m)."""

    def __init__(self, fn):
        self.fn = fn

    def depth(self, twt):
        return self.fn(np.asarray(twt, float))


LAYERS = [("H1", "Top A"), ("H2", "Top B")]


def run(check, tmp):
    from .velocity import VelocityModel, fit_line, kz_base, _ek
    # ---- helpers
    check(abs(float(_ek(1e-12, 0.4)) - 0.4) < 1e-9 and abs(float(_ek(0.5, 0.4)) - np.expm1(0.2) / 0.5) < 1e-12,
          "17a: (exp(k tau)-1)/k with the k -> 0 limit")
    check(abs(float(kz_base(100.0, 2000.0, 0.0, 0.5)) - 1100.0) < 1e-9, "17a: V0 + kZ with k = 0 is constant velocity")
    xs = np.arange(10.0)
    ys = 3 + 2 * xs
    ys[6] += 8.0
    b, a, r2, res, out = fit_line(xs, ys)
    check(out.tolist() == [i == 6 for i in range(10)] and 0.5 < r2 < 1.0, "17a: cross-plot line flags a 2-sigma outlier")

    # ---- V0 + kZ earth
    sp, T = ddc_case(tmp, "kz", kz_earth)
    vm = VelocityModel(sp, "3D:V", LAYERS, functions=[{"func": "v0kz"}, {"func": "v0kz"}]).build()
    wt = vm.grids()["wells"]
    check(len(wt) == 6, f"17a: 6 wells in the V0 + kZ model ({len(wt)})")
    info = vm.layer_info()
    kerr = [abs(info[i]["k"] - KZ[i][1]) / KZ[i][1] for i in range(2)]
    check(max(kerr) < 0.02, f"17a: k fitted per layer {info[0]['k']:.4f} / {info[1]['k']:.4f} "
                            f"(true {KZ[0][1]} / {KZ[1][1]}, max error {max(kerr):.2%})")
    v0 = [vm.v0_grid(i).z for i in range(2)]
    check(all(abs(np.median(v0[i]) - KZ[i][0]) < 0.02 * KZ[i][0] for i in range(2)),
          f"17a: V0 grids {np.median(v0[0]):.0f} / {np.median(v0[1]):.0f} m/s (true {KZ[0][0]:.0f} / {KZ[1][0]:.0f})")
    e1 = np.nanmax(np.abs(vm.depth_grid(0).z - T["z1"]))
    e2 = np.nanmax(np.abs(vm.depth_grid(1).z - T["z2"]))
    check(max(e1, e2) < 1.0, f"17a: V0 + kZ depth maps within 1 m everywhere (max {e1:.2f} / {e2:.2f} m)")
    vc = VelocityModel(sp, "3D:V", LAYERS).build()
    ec = np.nanmax(np.abs(vc.depth_grid(1).z - T["z2"]))
    check(ec > 3 * max(e2, 0.1), f"17a: constant-V0 model is worse on this earth ({ec:.1f} m vs {e2:.2f} m)")
    # fixed k, and k -> 0 equals the item-12 (V0) model
    vf = VelocityModel(sp, "3D:V", LAYERS, functions=[{"func": "v0kz", "k": 0.4}, None]).build()
    check(vf.layer_info()[0]["k"] == 0.4 and vf.layer_info()[0]["k_source"] == "fixed", "17a: user-fixed k is used")
    v00 = VelocityModel(sp, "3D:V", LAYERS, functions=[{"func": "v0kz", "k": 0.0}, {"func": "v0kz", "k": 0.0}]).build()
    d0 = max(np.nanmax(np.abs(v00.depth_grid(i).z - vc.depth_grid(i).z)) for i in range(2))
    check(d0 < 1e-6, f"17a: k = 0 reproduces the constant-velocity model (max diff {d0:.1e} m)")
    # cross-plot data
    cp = vm.crossplot(0)
    check(len(cp["x"]) == 6 and cp["slope"] > 0 and cp["r2"] > 0.9 and not cp["outlier"].any()
          and np.allclose(cp["v0_wells"], KZ[0][0], rtol=0.02),
          f"17a: cross-plot V_int vs mid depth (slope {cp['slope']:.3f}/s, R2 {cp['r2']:.3f}, V0 per well ok)")
    # depth conversion of times inside a layer follows the function (not linear)
    vol = sp.volume("V")
    il = int(vol.inlines[20])
    xs_, ys_ = vol.xy(np.full(vol.nxl, il), vol.xlines)
    times = np.asarray(vol.times, float)
    zt = vm.depth_of_time(xs_, ys_, times)
    jj = 10
    tH1 = float(T["t1"][20, jj])
    inside = times < tH1 - 1
    err_in = np.max(np.abs(zt[jj, inside] - kz_layer1_depth(times[inside])))
    zlin = vc.depth_of_time(xs_, ys_, times)
    err_lin = np.max(np.abs(zlin[jj, inside] - kz_layer1_depth(times[inside])))
    check(err_in < 1.0 and err_lin > 5 * err_in, f"17a: depth of time inside layer 1 follows V0 + kZ (max {err_in:.2f} m; "
                                                 f"linear-in-time {err_lin:.1f} m)")
    zh = vm.depth_of_time(xs_[jj:jj + 1], ys_[jj:jj + 1], np.array([tH1, float(T['t2'][20, jj])]))[0]
    zmap = [float(vm.depth_grid(i).sample([xs_[jj]], [ys_[jj]])[0]) for i in range(2)]
    check(np.max(np.abs(zh - zmap)) < 0.05, f"17a: horizon times land on the depth maps ({np.max(np.abs(zh - zmap)):.3f} m)")
    # constant-velocity model: new vectorised depth_of_time = item-12 formula at grid nodes
    Tn, Vn = vc.trace_layers(xs_, ys_)
    i = jj
    tk = np.maximum.accumulate(np.r_[0.0, Tn[i]])
    vk = np.r_[Vn[i], Vn[i][-1]]
    zk = np.r_[0.0, np.cumsum(vk[:-1] * np.diff(tk) / 2000.0)]
    lay = np.clip(np.searchsorted(tk, times, "right") - 1, 0, len(vk) - 1)
    old = zk[np.minimum(lay, len(zk) - 1)] + vk[lay] * (times - tk[np.minimum(lay, len(tk) - 1)]) / 2000.0
    check(np.max(np.abs(zlin[i] - old)) < 0.05, f"17a: V0 default unchanged vs item-12 depth_of_time at a node "
                                               f"({np.max(np.abs(zlin[i] - old)):.4f} m)")
    # round trip of the functions
    vr = VelocityModel.from_dict(sp, vm.to_dict())
    check(vr.functions == vm.functions and "functions" not in vc.to_dict(), "17a: functions saved with the model "
                                                                             "(old models unchanged)")
    # ---- well T-D function layers (stand-in checkshots following the layer-1 law)
    ws = sp.wellset()
    ws.td = {n: _TD(kz_layer1_depth) for n in ws.names}
    for blend in ("idw", "nearest"):
        vt = VelocityModel(sp, "3D:V", LAYERS, functions=[{"func": "td", "blend": blend}, None]).build()
        et = np.nanmax(np.abs(vt.depth_grid(0).z - T["z1"]))
        zt_td = vt.depth_of_time(xs_, ys_, times)
        ei = np.max(np.abs(zt_td[jj, inside] - kz_layer1_depth(times[inside])))
        check(et < 0.5 and ei < 0.5, f"17a: T-D function layer ({blend}): map {et:.3f} m, inside layer {ei:.3f} m")
    ws.td = {}

    # ---- V0 + kT earth
    sp2, T2 = ddc_case(tmp, "kt", kt_earth)
    vt = VelocityModel(sp2, "3D:V", LAYERS, functions=[{"func": "v0kt"}, {"func": "v0kt"}]).build()
    inf = vt.layer_info()
    kerr = [abs(inf[i]["k"] - KT[i][1]) / KT[i][1] for i in range(2)]
    e2 = max(np.nanmax(np.abs(vt.depth_grid(i).z - T2[f"z{i + 1}"])) for i in range(2))
    check(max(kerr) < 0.02 and e2 < 1.0, f"17a: V0 + kT: k {inf[0]['k']:.0f} / {inf[1]['k']:.0f} (true {KT[0][1]:.0f} / "
                                         f"{KT[1][1]:.0f}), depth max error {e2:.2f} m")
    cp = vt.crossplot(1)
    check(cp["xlabel"].startswith("mid-layer one-way time"), "17a: V0 + kT cross-plot against time")
    return sp


def run_gui(check, tmp):
    from .gui.velocity_dialog import VelocityDialog
    sp, T = ddc_case(tmp, "kzgui", kz_earth)
    dlg = VelocityDialog(sp)
    vm = dlg.build("3D:V", LAYERS, False, functions=[{"func": "v0kz"}, {"func": "v0kz", "k": 0.35}])
    check(vm is not None and vm.layer_info()[1]["k"] == 0.35, "17a dialog: build with layer functions")
    cp = dlg.crossplot(0)
    check(cp is not None and len(cp["x"]) == 6 and "k" in dlg.info_text(), "17a dialog: cross-plot data + layer info")
    dlg.set_layer_function("H1", "v0kt", None)
    fs = dlg.functions_from_table(["H1", "H2"])
    check(fs[0]["func"] == "v0kt" and fs[0].get("k") is None, "17a dialog: layer function from the table")
    dlg.set_layer_function("H1", "v0kz", "0.5")
    check(dlg.functions_from_table(["H1", "H2"])[0] == {"func": "v0kz", "k": 0.5}, "17a dialog: fixed k from the table")
    dlg.use()
    d = sp.display.get("velocity_model", {})
    check(d.get("functions", [{}])[1].get("k") == 0.35, "17a dialog: functions stored for the depth display")
    return True
