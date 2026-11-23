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


def ddc_case(tmp, name, earth, wells=None, n=None, ns=250):
    """3D survey (41 x 41, 25 m) with two horizons and wells whose tops follow `earth(t1, t2) -> (z1, z2)`.
    n = survey size (n x n) for speed tests (ns small: the traces do not matter to the velocity model)."""
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
    nil = nxl = n or NIL
    i, j = np.meshgrid(np.arange(nil), np.arange(nxl), indexing="ij")
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


def _bump_mtime(path):
    st = os.stat(path)
    os.utime(path, (st.st_atime, st.st_mtime + 5))


def _edit_tops(path, fn):
    import json
    with open(path) as fh:
        d = json.load(fh)
    fn(d["tops"])
    with open(path, "w") as fh:
        json.dump(d, fh)
    _bump_mtime(path)


def run_dynamic(check, tmp):
    """17c: dynamic update - only layers at/below a change are rebuilt; incremental == full; what-if wells."""
    import time
    from .ddc import DynamicDepthModel
    from .velocity import VelocityModel
    sp, T = ddc_case(tmp, "dyn", kz_earth)
    md = VelocityModel(sp, "3D:V", LAYERS, functions=[{"func": "v0kz"}, {"func": "v0"}]).to_dict()
    dm = DynamicDepthModel(sp, md).build()
    check(dm.refresh() is None, "17c: no change -> no rebuild")
    z0 = [a.copy() for a in dm.vm.grids()["z"]]
    wpath = sp.well_project
    _edit_tops(wpath, lambda t: t["W-3"].__setitem__("Top B", t["W-3"]["Top B"] + 4.0))
    rep = dm.refresh()
    z1 = dm.vm.grids()["z"]
    check(rep is not None and rep["from_layer"] == 1 and "W-3 Top B top moved +4.0 m" in rep["message"]
          and "rebuilt layer 2" in rep["message"], f"17c: top change detected: {rep and rep['message']}")
    check(np.array_equal(z1[0], z0[0]), "17c: map above the changed top is bit-identical")
    d = np.abs(z1[1] - z0[1])
    check(d.max() > 1.0 and d.max() < 6.0, f"17c: map of the changed layer moves (max {d.max():.2f} m)")
    full = DynamicDepthModel(sp, md).build()
    dd = max(np.nanmax(np.abs(full.vm.grids()["z"][k] - z1[k])) for k in range(2))
    check(dd < 1e-9, f"17c: incremental rebuild == full rebuild ({dd:.1e} m)")
    mt = {r[0] + r[1]: r[5] for r in dm.vm.misfit_table()}
    check(abs(mt["W-3H2"]) < 4.0, "17c: misfit table follows the moved top")
    _edit_tops(wpath, lambda t: t["W-1"].__setitem__("Top A", t["W-1"]["Top A"] - 3.0))
    rep = dm.refresh()
    check(rep["from_layer"] == 0 and not np.array_equal(dm.vm.grids()["z"][0], z0[0]),
          f"17c: shallower top -> all layers rebuilt ({rep['message']})")
    # horizon edit (in memory, as the picker does)
    z2 = [a.copy() for a in dm.vm.grids()["z"]]
    h = sp.horizons.get("H2")
    h.data["3D:V"][24:30, 8:14] += 10.0  # away from the wells (a well top pins its depth)
    rep = dm.refresh()
    z3 = dm.vm.grids()["z"]
    check(rep is not None and rep["from_layer"] == 1 and "horizon H2 changed" in rep["message"]
          and np.array_equal(z3[0], z2[0]) and np.nanmax(z3[1] - z2[1]) > 3.0,
          f"17c: horizon edit -> layer 2 rebuilt, layer 1 identical ({rep and rep['message']})")
    h.data["3D:V"][24:30, 8:14] -= 10.0
    dm.refresh()
    # what-if well
    zb = [a.copy() for a in dm.vm.grids()["z"]]
    x, y = float(T["X"][20, 30]), float(T["Y"][20, 30])
    zm = float(dm.vm.depth_grid(1).sample([x], [y])[0])
    rep = dm.add_what_if("P-1", x, y, {"Top B": zm + 30.0})
    za = dm.vm.grids()["z"]
    zn = float(dm.vm.depth_grid(1).sample([x], [y])[0])
    check(np.array_equal(za[0], zb[0]) and abs(zn - (zm + 30.0)) < 3.0,
          f"17c: what-if top pulls layer 2 ({zm:.1f} -> {zn:.1f} m, target {zm + 30:.1f}); layer 1 identical")
    check(sp.display.get("ddc_what_if", {}).get("wells", {}).get("P-1") is not None and
          any(r[0] == "what-if: P-1" for r in dm.vm.misfit_table()), "17c: what-if well saved in the seismic project "
                                                                      "+ in the misfit table")
    import json
    with open(wpath) as fh:
        check("P-1" not in json.load(fh)["tops"], "17c: well project untouched by the what-if well")
    dm2 = DynamicDepthModel(sp, md).build()
    check(np.allclose(dm2.vm.grids()["z"][1], za[1]), "17c: what-if well reloaded with the project")
    dm.remove_what_if("P-1")
    check(np.allclose(dm.vm.grids()["z"][1], zb[1]) and "ddc_what_if" not in sp.display,
          "17c: removing the what-if well restores the model")
    # speed: 50 wells, 300 x 300 map
    rng = np.random.default_rng(3)
    wells = {f"S-{i}": (int(a), int(b)) for i, (a, b) in enumerate(rng.integers(5, 295, (50, 2)))}
    spb, _ = ddc_case(tmp, "big", kz_earth, wells=wells, n=300, ns=8)
    mdb = VelocityModel(spb, "3D:V", LAYERS, functions=[{"func": "v0kz"}, {"func": "v0kz"}]).to_dict()
    t0 = time.perf_counter()
    dmb = DynamicDepthModel(spb, mdb).build()
    tb = time.perf_counter() - t0
    _edit_tops(spb.well_project, lambda t: t["S-7"].__setitem__("Top A", t["S-7"]["Top A"] + 5.0))
    t0 = time.perf_counter()
    rep = dmb.refresh()
    tr = time.perf_counter() - t0
    check(rep is not None and rep["from_layer"] == 0 and tr < 2.0,
          f"17c: 50 wells, 300 x 300 map: full build {tb:.2f} s, rebuild after a top change {tr:.2f} s (< 2 s)")
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


def run_gui_dynamic(check, tmp):
    """17c in the main window: View > Auto-update depth model, watcher tick, shared model with the dialog, what-if."""
    from .gui.main_window import MainWindow
    sp, T = ddc_case(tmp, "dyngui", kz_earth)
    w = MainWindow(sp.path)
    dlg = w.velocity_model()
    dlg.build("3D:V", LAYERS, False, functions=[{"func": "v0kz"}, {"func": "v0"}])
    dlg.use()
    check(w.section.depth_model() is dlg.ddm, "17c GUI: 'Use' shares the dialog's dynamic model with the section")
    check(w.ddc_tick() is None, "17c GUI: auto-update off -> tick does nothing")
    w.set_ddc(True)
    check(w.ddc_auto and w._ddc_watched == [w.project.well_project], "17c GUI: auto-update on watches the well project")
    check(w.ddc_tick() is None, "17c GUI: nothing changed -> no rebuild")
    z0 = dlg.model.grids()["z"][0].copy()
    _edit_tops(w.project.well_project, lambda t: t["W-2"].__setitem__("Top B", t["W-2"]["Top B"] - 6.0))
    rep = w.ddc_tick()
    check(rep is not None and "W-2 Top B top moved -6.0 m" in rep["message"] and
          np.array_equal(dlg.model.grids()["z"][0], z0), f"17c GUI: tick rebuilds after a top edit ({rep and rep['message']})")
    vol = w.project.volume("V")
    p = w.section
    p.source = ("3D", "V")
    p.mode = "Inline"
    p.number = int(vol.inlines[20])
    p.domain = "Depth"
    p.redraw()
    check(p._data is not None and p._yaxis[-1] > 1000, "17c GUI: depth section drawn from the dynamic model")
    x, y = float(T["X"][20, 30]), float(T["Y"][20, 30])
    zm = float(dlg.model.depth_grid(1).sample([x], [y])[0])
    rep = dlg.add_what_if("P-1", x, y, "H2", zm + 20.0)
    check(rep is not None and "what-if P-1 added" in rep["message"] and
          abs(float(w.section.velocity_model().depth_grid(1).sample([x], [y])[0]) - zm - 20.0) < 2.0,
          "17c GUI: what-if pseudo-top from the dialog reaches the depth display")
    check(w.ddc_tick() is None, "17c GUI: the what-if change is not rebuilt twice")
    rep = dlg.clear_what_if()
    check(rep is not None and "ddc_what_if" not in w.project.display, "17c GUI: clear what-if wells")
    h = w.project.horizons.get("H1")
    h.data["3D:V"][0:5, 0:5] += 4.0
    w._horizons_changed()
    rep = w.ddc_tick()
    check(rep is not None and rep["from_layer"] == 0, "17c GUI: horizon edit picked up by the tick")
    w.set_ddc(False)
    check(not w.ddc_auto and w._ddc_watched == [], "17c GUI: auto-update off")
    return True
