"""Velocity-volume depth conversion tests (PLAN 37): synthetic earth with lateral + vertical velocity change."""
import os

import numpy as np

from .test_ddc import ddc_case, t_surfaces, LAYERS, WELLS6

WELLS7 = dict(WELLS6, **{"W-7": (28, 14)})
WELLS_LEFT = {"L-1": (5, 3), "L-2": (20, 6), "L-3": (35, 4), "L-4": (12, 12), "L-5": (28, 11)}  # x <= 300 m only
GX, KT, JUMP = 0.25, 0.4, 300.0  # lateral m/s per m, vertical m/s per ms, velocity jump below H2


def v_true(dx, t, t2):
    return 1800.0 + GX * dx + KT * t + np.where(t >= t2, JUMP, 0.0)


def z_true(dx, t, t2):
    a = 1800.0 + GX * dx
    z = (a * t + 0.5 * KT * t * t) / 2000.0
    return z + np.where(t >= t2, JUMP * (t - t2) / 2000.0, 0.0)


def earth(t1, t2, dx):
    return z_true(dx, t1, t2), z_true(dx, t2, t2)


class _TD:
    def __init__(self, dx, t2):
        self.dx, self.t2 = dx, t2

    def depth(self, twt):
        return z_true(self.dx, np.asarray(twt, float), self.t2)


def make(tmp, name, wells):
    sp, T = ddc_case(tmp, name, earth, wells=wells, earth_xy=True)
    x0 = float(T["X"][0, 0])
    y0 = float(T["Y"][0, 0])
    pos = {n: (x, y) for n, x, y in sp.wells()}

    def td_fn(n):
        x, y = pos[n]
        _, t2 = t_surfaces(np.float64(x), np.float64(y), x0, y0)
        return _TD(x - x0, float(t2))

    def tops_fn():
        out = {}
        for n, (x, y) in pos.items():
            t1, t2 = t_surfaces(np.float64(x), np.float64(y), x0, y0)
            z1, z2 = earth(t1, t2, x - x0)
            out[n] = {"Top A": float(z1), "Top B": float(z2)}
        return out
    return sp, T, td_fn, tops_fn, x0


def run(check, tmp):
    from .velvolume import VelocityVolumeModel, dix, difference_grids, save_scenario
    from .velocity import VelocityModel
    from .scenarios import blind_test
    sp, T, td_fn, tops_fn, x0 = make(tmp, "vv", WELLS7)
    m = VelocityVolumeModel(sp, "V", LAYERS, td_fn=td_fn, tops_fn=tops_fn, method="idw").build()
    check(len(m.curves) == 7 and all(abs(float(np.median(c.v)) - 2200) < 900 for c in m.curves.values()),
          "37a: interval velocities from the well time-depth functions")
    vt = v_true(T["X"][:, :, None] - x0, np.asarray(m.vol.times, float)[None, None, :], T["t2"][:, :, None])
    i, j = WELLS7["W-3"]
    vw = m.traces([i], [j])[0]
    t = np.asarray(m.vol.times, float)
    inside = (t > 50) & (t < T["t2"][i, j] + 300)
    check(np.median(np.abs(vw[inside] - vt[i, j][inside])) < 15, "37c: volume honours the well velocity at a well "
          f"({np.median(np.abs(vw[inside] - vt[i, j][inside])):.1f} m/s)")
    # 37e: the final maps tie every top
    tops = tops_fn()
    errs = []
    for n, x, y in sp.wells():
        for k, (h, fm) in enumerate(LAYERS):
            errs.append(abs(m._sample_grid(m.z_corr[k], x, y) - tops[n][fm]))
    check(max(errs) < 0.1, f"37e: corrected depth maps tie every top (max {max(errs):.4f} m)")
    zc = m._depth_grids(calibrated=True)
    dmax = max(float(np.nanmax(np.abs(zc[k] - m.z_corr[k]))) for k in range(2))
    check(dmax < 1.0, f"37e: calibrated velocity volume reproduces the corrected maps (max {dmax:.2f} m)")
    zt2 = T["z2"]
    err_vol = float(np.median(np.abs(m.z_corr[1] - zt2)))
    check(err_vol < 15.0, f"37d: depth maps through the volume close to the true earth (median {err_vol:.2f} m)")
    # depth sections / depth_of_time consistent with the traces
    zt = m.depth_of_time(T["X"][i, j], T["Y"][i, j], t)
    check(np.allclose(zt[0], m.depth_traces([i], [j], True)[0]), "37d: depth of time (sections, SEG-Y) = volume")
    # 37f: a moved top rebuilds only its layer and below
    z0 = [a.copy() for a in m.z_corr]
    xw, yw = next((x, y) for n, x, y in sp.wells() if n == "W-3")
    rep = m.set_top("W-3", "Top B", tops["W-3"]["Top B"] + 4.0)
    check(rep["from_layer"] == 1 and np.array_equal(m.z_corr[0], z0[0]) and
          abs(m._sample_grid(m.z_corr[1], xw, yw) - (tops["W-3"]["Top B"] + 4.0)) < 0.1,
          f"37f: moved top -> only {rep['layers']} rebuilt, map ties the new top ({rep['message']})")
    m.set_top("W-3", "Top B", tops["W-3"]["Top B"])
    # an edited tie of one well (velocities 3 % faster)
    base = td_fn("W-3")

    class Faster:
        def depth(self, twt):
            return 1.03 * base.depth(twt)
    zr = m.z_raw.copy()
    rep = m.update_well("W-3", Faster())
    far = np.unravel_index(np.argmin(m.W[:, :, m.names.index("W-3")]), m.W.shape[:2])
    check(rep["traces"] > 0 and float(np.nanmax(np.abs(m.z_raw[1] - zr[1]))) > 1.0 and
          abs(m._sample_grid(m.z_corr[1], xw, yw) - tops["W-3"]["Top B"]) < 0.1,
          f"37f: edited tie re-integrates {rep['traces']} traces; maps still tie the tops")
    del far
    m.update_well("W-3", base)
    # 37g: blind test - volume mode vs layer-cake (both on the same tops)
    bt_v = m.blind_test()
    vm = VelocityModel(sp, "3D:V", LAYERS)
    vm.build()
    bt_l = blind_test(vm)
    rv = bt_v["summary"]["H2"]["rms"]
    rl = bt_l["summary"]["H2"]["rms"]
    check(len(bt_v["rows"]) >= 12 and np.isfinite(rv) and np.isfinite(rl),
          f"37g: leave-one-well-out blind test, H2 rms: volume {rv:.1f} m vs layer-cake {rl:.1f} m")
    pm = m.probability_maps(1, bt_v)
    check(np.nanmin(pm["P90"] - pm["P10"]) >= -1e-9 and np.allclose(pm["P50"], m.z_corr[1]),
          "37g: P10 / P50 / P90 depth from the blind-test error")
    dg = difference_grids(m, vm)
    check(len(dg) == 2 and np.isfinite(dg[1].z).all(), "37g: difference maps volume - layer-cake")
    names = save_scenario(sp, "volume idw", m)
    check("volume idw" in names and sp.display["velocity_scenarios"]["volume idw"]["mode"] == "volume",
          "37g: registered as a named scenario")
    # stored volume + layer velocity map
    vname = m.write("Velocity volume")
    v = sp.volume(vname)
    got = np.asarray(v.ilxl[i], np.float32)[j] * v.inv
    check(np.median(np.abs(got - m.traces([i], [j], True)[0])) < 5.0, "37c: calibrated velocity volume stored "
          "(float16 store, viewable / SEG-Y export like any volume)")
    lg = m.layer_velocity_grid(1)
    check(np.nanmedian(lg.z) > 1800 and lg.unit == "m/s", "37h: layer-average velocity map for the map hub")
    vs, hz = m.velocity_section("Inline", int(m.vol.inlines[i]))
    check(vs.shape == (m.vol.nxl, m.vol.ns) and hz.shape == (2, m.vol.nxl), "37g: velocity section + horizons")

    # stacking-velocity trend recovers a lateral gradient the wells miss (all wells on the west side)
    sp2, T2, td2, tops2, x02 = make(tmp, "vvleft", WELLS_LEFT)
    plain = VelocityVolumeModel(sp2, "V", LAYERS, td_fn=td2, tops_fn=tops2).build()
    t = np.asarray(plain.vol.times, float)
    dxg = T2["X"] - x02
    vint = v_true(dxg[:, :, None], t[None, None, :], T2["t2"][:, :, None])
    # RMS velocity cube consistent with the interval velocities (what a stacking-velocity volume gives)
    dt = np.diff(np.r_[t[0], t])  # sample k = interval (t[k-1], t[k]]
    with np.errstate(invalid="ignore", divide="ignore"):
        vrms = np.sqrt(np.cumsum(vint ** 2 * dt, axis=2) / t)
    vrms[..., 0] = vint[..., 0]
    back = dix(t, vrms)
    check(np.median(np.abs(back[..., 5:] - vint[..., 5:])) < 1.0, "37c: Dix inversion of RMS velocities")
    guided = VelocityVolumeModel(sp2, "V", LAYERS, td_fn=td2, tops_fn=tops2,
                                 trend={"array": vrms.astype(np.float32), "kind": "rms"}).build()
    east = dxg > 700
    e_plain = float(np.median(np.abs(plain.z_raw[1] - T2["z2"])[east]))
    e_guided = float(np.median(np.abs(guided.z_raw[1] - T2["z2"])[east]))
    check(e_guided < 0.3 * e_plain and e_guided < 10.0, f"37c: stacking-velocity trend recovers the lateral gradient "
          f"(east side error {e_guided:.1f} m vs {e_plain:.1f} m with the wells alone)")
    bt_g = guided.blind_test()
    vm2 = VelocityModel(sp2, "3D:V", LAYERS)
    vm2.build()
    bt_l2 = blind_test(vm2)
    check(bt_g["summary"]["H2"]["rms"] < bt_l2["summary"]["H2"]["rms"],
          f"37i: volume mode beats layer-cake in the blind test (H2 rms {bt_g['summary']['H2']['rms']:.1f} vs "
          f"{bt_l2['summary']['H2']['rms']:.1f} m)")
    return m


def run_gui(check, tmp):
    """37h GUI: Interpret > Velocity volume dialog: steps, maps to the hub, blind test, dynamic re-tie."""
    from .gui.main_window import MainWindow
    from .velocity import VelocityModel
    from .test_ddc import _edit_tops
    sp, T, td_fn, tops_fn, x0 = make(tmp, "vvgui", WELLS7)
    sp.display["velocity_model"] = VelocityModel(sp, "3D:V", LAYERS).to_dict()
    sp.save()
    mw = MainWindow(sp.path)
    dlg = mw.velocity_volume()
    check(dlg.vol_name == "V" and dlg.layers == [tuple(x) for x in LAYERS], "37h GUI: dialog picks up the volume + "
          "layers of the velocity model")
    rows = dlg.well_velocities(td_fn=td_fn)
    check(len(rows) == 7, "37h GUI: a. well velocities table")
    seen = []
    mw.mapbus.subscribe("probe", seen.append)
    m = dlg.build(td_fn=td_fn)
    check(m._built and any(e.get("kind") == "velocities" for e in seen), "37h GUI: b-e. build + tie, map bus notified")
    ids = dlg.add_maps()
    check(len(ids) == 6 and all(mw.basemap.stack.grid(i) is not None for i in ids), "37h GUI: depth / velocity / "
          "residual maps in the map hub")
    bt = dlg.blind_test()
    check("volume" in bt and "layer-cake" in bt, "37g GUI: blind test volume vs layer-cake")
    nm = dlg.store()
    check(any(v["name"] == nm for v in mw.project.volumes), "37c GUI: calibrated volume stored in the project")
    # dynamic update: a top moved in the well project -> re-tie on the next tick
    _edit_tops(sp.well_project, lambda t: t["W-3"].__setitem__("Top B", t["W-3"]["Top B"] + 6.0))
    mw.ddc_auto = True
    mw.ddc_tick()
    check(dlg.model.history and dlg.model.history[-1]["from_layer"] == 1, "37f GUI: moved top re-ties the volume maps "
          "on the dynamic-update tick")
    mw.ddc_auto = False
    mw.maplink.close()
