"""RGT step C checks: horizon stack, Wheeler transform, stratal slices, stack level -> horizon."""
import os

import numpy as np


def run(check, tmp):
    from .rgt import solve as S
    from .rgt import stack as K
    from .test_rgt import geology_project

    # iso_times against np.interp per trace
    rng = np.random.default_rng(1)
    tau = np.cumsum(rng.uniform(0, 2, (7, 50)), axis=1)
    tau[:, 10:13] = tau[:, 10:11]  # flat part (hiatus)
    times = np.arange(50) * 4.0
    lev = np.linspace(-5, tau.max() + 5, 33)
    got = K.iso_times(tau, lev, times)
    want = np.array([[np.interp(v, tr, times) if tr[0] <= v <= tr[-1] else np.nan for v in lev] for tr in tau])
    ok = np.isfinite(want) & ~np.isin(lev, tau)
    check(np.allclose(got[ok], want[ok], atol=1e-3) and np.array_equal(np.isfinite(got), np.isfinite(want)),
          "iso_times (one global searchsorted) == per-trace interpolation, NaN outside")

    p, G = geology_project(tmp, "stack", fault=False)
    vol = p.volume("G")
    rv, _ = S.compute_rgt_3d(vol)
    st = K.build_stack_3d(vol, rv, 60, block=7)
    check(len(st) == 60 and st.times.shape == (60, vol.nil, vol.nxl), "horizon stack: 60 iso-RGT surfaces")
    k = 25
    direct = rv.iso_surface(st.levels[k], vol)
    check(np.allclose(st.surface(k), direct, equal_nan=True, atol=1e-3), "stack surface == direct iso-surface")
    d = np.diff(np.asarray(st.times), axis=0)
    check(np.all(d[np.isfinite(d)] >= -1e-3), "stack surfaces never cross")
    # stratal slice: on a true horizon surface the amplitude is the reflector's amplitude (+/- noise)
    hz = G["horizons"]
    cube = G["cube"]
    surf = (300.0 + 2.5 * np.arange(vol.nil)[:, None] + 1.3 * np.arange(vol.nxl)[None, :]).astype(np.float32)
    surf[3, 4] = np.nan
    amp = K.stratal_slice(vol, surf)
    u = surf / 4.0
    k0 = np.floor(np.nan_to_num(u)).astype(int)
    w = np.nan_to_num(u) - k0
    ref = np.take_along_axis(cube, k0[..., None], 2)[..., 0] * (1 - w) + \
        np.take_along_axis(cube, (k0 + 1)[..., None], 2)[..., 0] * w
    err = np.nanmax(np.abs(amp - ref)) / np.abs(cube).max()
    check(err < 0.01 and np.isnan(amp[3, 4]), f"stratal slice = volume interpolated on the surface (err {err:.1e})")
    # Wheeler: pinch-out shows as a hiatus gap, flat reflectors elsewhere
    seis = np.asarray(vol.ilxl[20], np.float32)
    tau20 = rv.decode(rv.arr[20])
    W, gap = K.wheeler(seis, tau20, st.levels, vol.times)
    wl, wh = G["wedge"]
    to_rgt = lambda geo: np.interp(np.interp(geo, G["tau_ref"], hz[:, 20, 50]), vol.times, tau20[50])  # noqa
    lv_in = (st.levels > to_rgt(wl + 10)) & (st.levels < to_rgt(wh - 10))
    tt = K.iso_times(tau20, st.levels, vol.times)
    thick = np.nansum(np.diff(tt, axis=1)[:, lv_in[:-1]], axis=1)
    check(thick[:5].mean() < 0.5 * thick[45:].mean(),
          f"RGT levels converge towards the pinch-out ({thick[:5].mean():.0f} vs {thick[45:].mean():.0f} ms; "
          "below tuning the RGT cannot reach zero thickness)")
    # hiatus detection: on the first 10 traces geologic time jumps by 40 at 400 ms (missing section)
    tau_h = np.tile(np.arange(300, dtype=float) * 4.0, (30, 1))
    tau_h[:10, 100:] += 40.0
    Wh, gh = K.wheeler(np.ones((30, 300)), tau_h, np.arange(0, 1100, 8.0), np.arange(300) * 4.0)
    lev_gap = (np.arange(0, 1100, 8.0) > 400) & (np.arange(0, 1100, 8.0) < 432)
    check(gh[lev_gap][:, :10].all() and not gh[lev_gap][:, 10:].any() and np.isnan(Wh[lev_gap][:, :10]).all(),
          "Wheeler: a hiatus (no deposition) shows as a gap on the traces where it occurs")
    rows = W[~lv_in & np.isfinite(W).all(axis=1)]
    flat = 1 - np.mean(np.var(rows, axis=1)) / np.var(rows)
    check(flat > 0.8, f"Wheeler: amplitude is constant along each RGT level ({flat:.0%} of the variance is "
                      "between levels)")
    # stack level -> horizon
    from .horizons import HorizonSet
    hs = HorizonSet()
    h = K.stack_to_horizon(hs, "L25", k, st, "G")
    check(np.allclose(h.get("3D:G"), st.surface(k), equal_nan=True), "stack level converted to an editable horizon")
    # 2D stack
    p2, G2 = geology_project(tmp, "stack2d", fault=False, lines=True)
    S.compute_rgt_2d(p2, ["A", "B"])
    s2 = K.build_stack_2d(p2, ["A", "B"], 40)
    check(set(s2["lines"]) == {"A", "B"} and s2["lines"]["A"].shape == (40, 60), "2D horizon stack per line")
    h2 = K.stack_to_horizon(hs, "L2d", 10, stack2d=s2)
    check(set(h2.data) == {"2D:A", "2D:B"}, "2D stack level -> horizon on every line")
    return p


def run_gui(check, tmp):
    from .gui.main_window import MainWindow
    from .rgt import solve as S
    from .rgt import stack as K
    from .test_rgt import geology_project
    p, G = geology_project(tmp, "stackgui", fault=False, lines=True)
    vol = p.volume("G")
    rv, _ = S.compute_rgt_3d(vol)
    K.build_stack_3d(vol, rv, 50)
    S.compute_rgt_2d(p, ["A", "B"])
    K.build_stack_2d(p, ["A", "B"], 50)
    w = MainWindow(p.path)
    sp = w.section
    sp.source = ("3D", "G")
    sp._mode_changed("Stratal slice", number=20)
    check(sp.mode == "Stratal slice" and sp._data is not None and sp._data.shape == (vol.nil, vol.nxl),
          "stratal slice mode shows an amplitude map")
    check(sp.trace_at(float(vol.xlines[3])) is None, "no picking on stratal slices")
    sp.mode = "Inline"
    sp.number = int(vol.inlines[20])
    sp.domain = "Wheeler"
    sp.redraw()
    check(sp._data is not None and sp._data.shape == (50, vol.nxl) and np.isnan(sp._data).any(),
          "Wheeler view of an inline (stack levels on the vertical axis, hiatus gaps)")
    h = w.stack_level_to_horizon(level=20, name="From stack")
    check(h is not None and np.isfinite(h.get("3D:G")).mean() > 0.9 and "2D:A" in h.data,
          "Interpret > Convert stack level to horizon (3D + 2D)")
    sp.redraw()
    check(len(sp._hz_items) >= 1, "horizon drawn in the Wheeler domain")
    sp.domain = "Time"
    sp.source = ("2D", "A")
    sp.mode = "2D"
    sp.domain = "Wheeler"
    sp.redraw()
    check(sp._data is not None and sp._data.shape[0] == 50, "Wheeler view of a 2D line")
    sp.domain = "Time"
    return True
