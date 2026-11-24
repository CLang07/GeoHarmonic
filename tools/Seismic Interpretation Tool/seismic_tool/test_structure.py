"""Tests for PLAN 26: tectonic phases + fault growth, flattening / decompaction / animation, 2D restoration."""
import os

import numpy as np


def _faulted_section(ns=300, n=80, dt=2.0, throw=30.0, j_f=40, seed=0):
    """Layered section (Ricker reflectors at a few times, gentle dip) and the same with the hanging wall (traces >=
    j_f) dropped by `throw` ms.  Returns times, unfaulted, faulted, horizons (unfaulted), horizons (faulted)."""
    from .synth import ricker
    times = np.arange(ns) * dt
    x = np.arange(n)
    tops = {"A": 150 + 0.3 * x, "B": 260 + 0.3 * x, "C": 370 + 0.3 * x}
    amps = {"A": 1.0, "B": -0.8, "C": 0.6}

    def make(hz):
        sec = np.zeros((ns, n))
        for k, t in hz.items():
            sec += amps[k] * ricker((times[:, None] - t[None, :]) / 1000.0, 30.0)
        return sec
    hf = {k: np.where(x >= j_f, v + throw, v) for k, v in tops.items()}
    return times, make(tops), make(hf), tops, hf


def run(check, tmp):
    from . import structure as S
    from .basin.chrono import AgeTable
    from .basin.burial import solid_thickness
    # ---- 26a phases
    tb = AgeTable([{"name": "Seabed", "age": 0.0}, {"name": "H1", "age": 5.0}, {"name": "H2", "age": 20.0},
                   {"name": "H3", "age": 40.0}, {"name": "Base", "age": 80.0}])
    S.set_phase(tb, "Seabed", "post-rift")
    S.set_phase(tb, "H1", "post-rift")
    S.set_phase(tb, "H2", "syn-rift")
    S.set_phase(tb, "H3", "pre-rift")
    iv = S.phase_intervals(tb)
    cols = S.phase_colors_for_axis(tb, [2.0, 30.0, 60.0])
    check([p for _, _, p, _ in iv] == ["post-rift", "post-rift", "syn-rift", "pre-rift"] and
          S.phase_of_age(tb, 30.0) == "syn-rift" and tuple(cols[1]) == S.PHASE_COLORS["syn-rift"],
          "26a: tectonic phase per age interval + Wheeler colours")
    try:
        S.set_phase(tb, "H1", "wobbly")
        bad = False
    except ValueError:
        bad = True
    check(bad, "26a: unknown phase rejected")
    # growth fault: footwall intervals 50 ms; hanging wall 50 / 50 / 80 / 80 / 50 (syn-rift = intervals 2-3)
    nil, nxl = 30, 60
    fx = lambda i: 25 + 0.2 * i  # noqa: E731
    th_fw = [50.0] * 5
    th_hw = [50.0, 50.0, 80.0, 80.0, 50.0]
    names = ["H0", "H1", "H2", "H3", "H4", "H5"]
    J = np.arange(nxl)[None, :]
    I = np.arange(nil)[:, None]
    hw_side = J >= np.round(fx(I))
    grids = {}
    t_fw = 400.0
    t_hw = 400.0
    for k, n in enumerate(names):
        grids[n] = np.where(hw_side, t_hw, t_fw) + 0.5 * J + 0.2 * I
        if k < 5:
            t_fw += th_fw[k]
            t_hw += th_hw[k]
    fg = S.fault_growth(grids, fx, offset=3)
    check(np.allclose(fg["ei"], [1.0, 1.0, 1.6, 1.6, 1.0], atol=0.02) and np.all(np.diff(fg["throw"]) >= -1e-9) and
          fg["throw"][-1] > fg["throw"][0] + 50,
          f"26a: expansion index per interval {np.round(fg['ei'], 2).tolist()}, throw grows with depth")
    check(fg["phases"] == {"H0": "post-rift", "H1": "post-rift", "H2": "syn-rift", "H3": "syn-rift", "H4": "pre-rift"},
          f"26a: proposed phases from the growth {fg['phases']}")
    # ---- 26b flattening / decompaction / animation
    times, unf, flt, tops, hf = _faulted_section()
    fl = S.flatten(unf, times, tops["B"], 260.0)
    k = int(260 / 2)
    peak_rows = np.argmax(np.abs(np.nan_to_num(fl[100:160])), axis=0) + 100
    check(np.all(np.abs(peak_rows - k) <= 1), "26b: flattening puts the horizon on the datum on every trace")
    depths = [np.full(5, 1000.0), np.full(5, 1500.0), np.full(5, 2300.0)]
    pd = S.paleo_depths(depths, 1, phi0=0.55, c=0.0005)
    s_now = solid_thickness(1500.0, 2300.0, 0.55, 0.0005)
    s_then = solid_thickness(pd[0], pd[1], 0.55, 0.0005)
    check(np.allclose(s_then, s_now, rtol=1e-6) and np.all(pd[1] - pd[0] > 800.0) and np.all(pd[0] == 0),
          f"26b: decompacted paleo-depth (unit {800:.0f} m now -> {float(pd[1][0] - pd[0][0]):.0f} m at H1 time, grains "
          "conserved)")
    frames = S.paleo_frames(unf, times, tops)
    rg = S.frames_to_rgba(frames)
    path = S.export_gif(rg, os.path.join(tmp, "paleo.gif"))
    from PIL import Image
    im = Image.open(path)
    check([f[0] for f in frames] == ["C", "B", "A"] and getattr(im, "n_frames", 1) == 3 and
          np.isnan(frames[0][1][: int(np.nanmean(tops["C"]) / 2) - 2]).all(),
          "26b: paleo animation oldest -> youngest (younger section blanked), GIF written")
    # ---- 26c restoration
    steps = S.restore_fault(flt, times, hf, 40)
    name, throw, rest, rh = steps[0]
    inner = (times > 60) & (times < 520)
    err = np.sqrt(np.nanmean((rest[inner][:, 5:-5] - unf[inner][:, 5:-5]) ** 2)) / np.std(unf[inner])
    check(name == "A" and abs(throw - 30.0) < 0.5 and err < 0.05 and np.allclose(rh["C"], tops["C"]),
          f"26c: block restoration of the fault returns the unfaulted layers (throw {throw:.1f} ms, rel. error "
          f"{err:.3f})")
    x = np.arange(80) * 25.0
    check(abs(S.line_length(x, rh["A"] * 1.0) - S.line_length(x, tops["A"])) < 1e-6 and
          abs(S.area_between(x, rh["A"], rh["B"]) - S.area_between(x, tops["A"], tops["B"])) < 1e-6,
          "26c: line-length and area checks after restoration")
    uf = S.unfold_vertical_shear(unf, times, tops["A"], 150.0)
    check(np.all(np.abs(np.argmax(np.abs(np.nan_to_num(uf[50:100])), axis=0) + 50 - 75) <= 1),
          "26c: vertical-shear unfolding to a regional datum")


def run_gui(check, tmp):
    from .gui.main_window import MainWindow
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "paleo_gui")
    mw = MainWindow(proj.path)
    sec = mw.section
    vol = mw.project.volume("Synth")
    sec.source, sec.mode, sec.number, sec.domain, sec.attr = ("3D", "Synth"), "Inline", int(vol.inlines[10]), "Time", "Seismic"
    sec.redraw()
    hz = sec.section_horizons()
    names = mw.paleo_animation(os.path.join(tmp, "p.gif"))
    check(list(hz) == ["Top", "Deep"] and names == ["Deep", "Top"] and os.path.exists(os.path.join(tmp, "p.gif")),
          "26b GUI: paleo animation of the current inline (Deep then Top) written as GIF")
