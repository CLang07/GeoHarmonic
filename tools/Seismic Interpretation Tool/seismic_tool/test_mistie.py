"""2D-3D / 2D-2D mistie measurement, network solve, applying corrections."""
import os

import numpy as np


def run(check, tmp):
    from . import mistie as MI
    from .horizons import Horizon
    from .selftest import write_segy
    from .test_rgt import geology_project

    # analytic signal
    t = np.arange(512)
    z = MI.analytic(np.cos(2 * np.pi * t / 32.0))
    check(np.allclose(z.imag[32:-32], np.sin(2 * np.pi * t / 32.0)[32:-32], atol=1e-6), "numpy analytic signal (Hilbert)")
    # single trace measurement: b = corrupted a; measure returns the inverse
    rng = np.random.default_rng(3)
    refl = np.zeros(600)
    refl[rng.integers(20, 580, 60)] = rng.uniform(-1, 1, 60)
    from .welltie import ricker
    a = np.convolve(refl, ricker(30.0, 4.0), "same")
    b = MI.apply_correction(a, 4.0, -9.0, -35.0, 0.5)[0]  # b is early by 9 ms, rotated, weaker
    r = MI.measure(a, b, 4.0, (200, 2200), 40.0)
    check(abs(r["shift_ms"] - 9.0) < 0.5 and abs(MI.wrap(r["phase_deg"] - 35.0)) < 4 and abs(r["gain"] - 2.0) < 0.1
          and r["corr"] > 0.95, f"mistie measure: shift {r['shift_ms']:.2f} ms, phase {r['phase_deg']:.1f} deg, "
                                f"gain {r['gain']:.2f} (true 9, 35, 2)")
    # 2D lines cut from a 3D survey along oblique paths, with known corruptions
    d = os.path.join(tmp, "mistie")
    os.makedirs(d)
    p, G = geology_project(d, "m", fault=False)
    vol = p.volume("G")
    truth = {"P": (8.0, 30.0, 2.0), "Q": (-6.0, -50.0, 0.7), "R": (3.0, 170.0, 1.0)}
    paths = {"P": ((3, 3), (36, 55)), "Q": ((36, 4), (4, 56)), "R": ((20, 0), (21, 59))}
    for name, ((i0, j0), (i1, j1)) in paths.items():
        n = 90
        ii, jj = np.linspace(i0, i1, n), np.linspace(j0, j1, n)
        x, y = vol.xy(vol.il0 + ii * vol.il_step, vol.xl0 + jj * vol.xl_step)
        tr, inside = MI.extract_3d_along(vol, x, y)
        sh, ph, gn = truth[name]
        tr2 = MI.apply_correction(tr, 4.0, -sh, -ph, 1.0 / gn).astype(np.float32)  # inverse of the truth
        tr2 += 0.01 * rng.standard_normal(tr2.shape).astype(np.float32)
        pth = os.path.join(d, name + ".sgy")
        write_segy(pth, tr2, {21: np.arange(1, n + 1), 181: np.round(x).astype(np.int64),
                              185: np.round(y).astype(np.int64)})
        p.lines.append({"name": name, "segy": pth, "cdp_byte": 21})
    p.save()
    r = MI.tie_line_to_volume(p, "G", "P", (100, 1100), 40.0, profile_traces=30)
    sh, ph, gn = truth["P"]
    check(abs(r["shift_ms"] - sh) < 1.0 and abs(MI.wrap(r["phase_deg"] - ph)) < 5 and abs(r["gain"] / gn - 1) < 0.05,
          f"2D-3D tie: {r['shift_ms']:.2f} ms / {r['phase_deg']:.1f} deg / x{r['gain']:.2f} (true {sh}, {ph}, {gn}), "
          f"corr {r['corr']:.2f}")
    prof = np.array([q[1] for q in r["profile"]])
    check(len(prof) > 3 and np.max(np.abs(prof - sh)) < 2.0, "along-line mistie profile is flat for a constant mistie")
    ties = MI.all_ties(p, (100, 1100), 40.0)
    kinds = sorted(t["kind"] for t in ties)
    check(kinds.count("2D-3D") == 3 and kinds.count("2D-2D") >= 2, f"ties found: {kinds}")
    # add a deliberately wrong 2D-2D tie (cycle skip) - robust solve must down-weight it
    bad = dict(next(t for t in ties if t["kind"] == "2D-2D"))
    bad["shift_ms"] += 30.0
    bad["corr"] = 0.9
    sol, tt = MI.solve_network(["P", "Q", "R"], ties + [bad])
    errs = {n: (sol[n]["shift_ms"] - truth[n][0], MI.wrap(sol[n]["phase_deg"] - truth[n][1]),
                sol[n]["gain"] / truth[n][2] - 1) for n in truth}
    check(all(abs(e[0]) < 1.0 and abs(e[1]) < 6 and abs(e[2]) < 0.06 for e in errs.values()),
          "network solve recovers every line's shift/phase/gain (3D fixed): " +
          ", ".join(f"{n} {e[0]:+.2f} ms {e[1]:+.1f} deg" for n, e in errs.items()))
    check(abs(bad["res_shift_ms"]) > 20 and bad["weight"] < 0.2 * bad["corr"] ** 2,
          f"robust (Huber) solve isolates the bad tie (residual {bad['res_shift_ms']:+.1f} ms)")
    sol_nr, _ = MI.solve_network(["P", "Q", "R"], ties + [dict(bad)], robust=False)
    worst = max(abs(sol_nr[n]["shift_ms"] - truth[n][0]) for n in truth)
    check(worst > 1.0, f"(without robust weights the bad tie pulls lines off by up to {worst:.1f} ms)")
    # phase averaging across +/-180: unit vectors
    cl = MI.loop_closures(ties)
    check(len(cl) >= 1 and max(abs(c) for _, c in cl) < 2.0, f"loop closures small ({len(cl)} triangles)")
    # ---- flex mode: a line whose mistie ramps 0 -> 12 ms along the line
    n = 120
    ii, jj = np.linspace(5, 34, n), np.linspace(2, 57, n)
    x, y = vol.xy(vol.il0 + ii * vol.il_step, vol.xl0 + jj * vol.xl_step)
    tr, inside = MI.extract_3d_along(vol, x, y)
    ramp = np.linspace(0.0, 12.0, n)
    trf = MI.apply_correction(tr, 4.0, -ramp, 0.0, 1.0).astype(np.float32)
    pth = os.path.join(d, "F.sgy")
    write_segy(pth, trf, {21: np.arange(1, n + 1), 181: np.round(x).astype(np.int64), 185: np.round(y).astype(np.int64)})
    p.lines.append({"name": "F", "segy": pth, "cdp_byte": 21})
    p._line_cache.clear()
    const = {"F": MI.tie_line_to_volume(p, "G", "F", (100, 1100), 40.0)}
    fx = MI.solve_flex(p, ["F"], [], const, lam=0.3, knot_traces=15, window_ms=(100, 1100), profile_traces=20)["F"]
    est = MI.flex_shift_per_trace(p.line("F"), fx)
    err = np.abs(est - ramp)[10:-10]
    check(abs(const["F"]["shift_ms"] - 6.0) < 1.5 and err.max() < 1.5,
          f"flex mode follows an along-line ramp (constant {const['F']['shift_ms']:.1f} ms; flex max err {err.max():.2f} ms)")
    fx_stiff = MI.solve_flex(p, ["F"], [], const, lam=50.0, knot_traces=15, window_ms=(100, 1100), profile_traces=20)["F"]
    check(np.ptp(fx_stiff["shift"]) < 1.0, f"high smoothness -> nearly constant per line (range {np.ptp(fx_stiff['shift']):.2f} ms)")
    p.lines = [l for l in p.lines if l["name"] != "F"]
    p._line_cache.clear()
    hf = Horizon("HF")
    hf.data["2D:F"] = np.full(n, 600.0, np.float32)
    p.lines.append({"name": "F", "segy": pth, "cdp_byte": 21})
    p.horizons.items.append(hf)
    MI.apply_solution(p, {"F": dict(const["F"], flex=fx)})
    lnF = p.line("F")
    trc, _ = MI.extract_3d_along(vol, lnF.x, lnF.y)
    m = (vol.times > 100) & (vol.times < 1100)
    ccf = np.corrcoef(lnF.data()[10:-10][:, m].ravel(), trc[10:-10][:, m].ravel())[0, 1]
    check(ccf > 0.97 and abs(hf.data["2D:F"][-1] - (600.0 + est[-1])) < 1e-3,
          f"flex correction applied per trace (corr {ccf:.2f}); picks moved by the local shift")
    MI.apply_solution(p, {})
    p.horizons.items.remove(hf)
    p.lines = [l for l in p.lines if l["name"] != "F"]
    p._line_cache.clear()

    # ---- horizon-based misties
    hz = G["horizons"]
    s_h = 25
    hh = Horizon("Hz")
    hh.data["3D:G"] = hz[s_h].astype(np.float32)
    from .maps import MapGrid
    g3 = MapGrid(hz[s_h], "ilxl", "ms", "", vol)
    for name in ("P", "Q", "R"):
        ln = p.line(name)
        hh.data[f"2D:{name}"] = (g3.sample(ln.x, ln.y) - truth[name][0]).astype(np.float32)
    ht = MI.horizon_ties(p, hh)
    h3 = {t["b"]: t["shift_ms"] for t in ht if t["kind"] == "horizon 2D-3D"}
    check(set(h3) == {"P", "Q", "R"} and all(abs(h3[n] - truth[n][0]) < 0.5 for n in h3) and
          any(t["kind"] == "horizon 2D-2D" for t in ht), "horizon-based misties (2D-3D and 2D-2D)")
    # ---- survey groups and key lines
    solg, _ = MI.solve_grouped(["P", "Q", "R"], [dict(t) for t in ties], groups={"P": "S1", "Q": "S1"})
    check(abs(solg["P"]["shift_ms"] - solg["Q"]["shift_ms"]) < 1e-9 and solg["P"]["group"] == "S1" and
          min(truth["P"][0], truth["Q"][0]) - 1 < solg["P"]["shift_ms"] < max(truth["P"][0], truth["Q"][0]) + 1,
          f"survey group: P and Q share one correction ({solg['P']['shift_ms']:.1f} ms)")
    solk, _ = MI.solve_grouped(["P", "Q", "R"], [dict(t) for t in ties if t["kind"] == "2D-2D"], fixed_lines=["R"])
    check(solk["R"]["shift_ms"] == 0.0 and abs((solk["P"]["shift_ms"] - solk["Q"]["shift_ms"]) -
                                               (truth["P"][0] - truth["Q"][0])) < 1.5,
          "key line held fixed; others solved relative to it (2D-2D ties only)")
    # ---- time-variant mistie
    tv_true = {"t": [200.0, 1100.0], "shift": [4.0, 12.0]}
    ln = p.line("P")
    tr3, ins = MI.extract_3d_along(vol, ln.x, ln.y)
    trt = MI.apply_time_variant(tr3, vol.times, {"t": tv_true["t"], "shift": [-4.0, -12.0]}).astype(np.float32)
    ptv = os.path.join(d, "T.sgy")
    write_segy(ptv, trt, {21: np.arange(1, len(trt) + 1), 181: np.round(ln.x).astype(np.int64),
                          185: np.round(ln.y).astype(np.int64)})
    p.lines.append({"name": "T", "segy": ptv, "cdp_byte": 21})
    p._line_cache.clear()
    tv = MI.time_variant_line(p, "G", "T", 100.0, 1150.0, 300.0, 150.0)
    want = np.interp(tv["t"], tv_true["t"], tv_true["shift"])
    check(tv is not None and len(tv["t"]) >= 4 and np.max(np.abs(np.array(tv["shift"]) - want)) < 1.0,
          f"time-variant mistie: {', '.join(f'{a:.0f} ms: {b:.1f}' for a, b in zip(tv['t'], tv['shift']))}")
    MI.apply_solution(p, {"T": {"shift_ms": 0.0, "phase_deg": 0.0, "gain": 1.0, "tv": tv}})
    lt = p.line("T")
    m = (vol.times > 300) & (vol.times < 1000)
    cct = np.corrcoef(lt.data()[:, m].ravel(), tr3[:, m].ravel())[0, 1]
    check(cct > 0.97, f"time-variant correction applied (corr with 3D {cct:.2f})")
    MI.apply_solution(p, {})
    p.lines = [l for l in p.lines if l["name"] != "T"]
    # ---- overlapping duplicate line (reprocessed vintage of P, 5 ms later)
    lp = p.line("P")
    po = os.path.join(d, "P_reproc.sgy")
    write_segy(po, MI.apply_correction(lp.raw_data(), 4.0, 5.0, 0.0, 1.0).astype(np.float32),
               {21: np.arange(1, len(lp.cdp) + 1), 181: np.round(lp.x).astype(np.int64),
                185: np.round(lp.y).astype(np.int64)})
    p.lines.append({"name": "P-reproc", "segy": po, "cdp_byte": 21})
    p._line_cache.clear()
    ov = MI.overlap_ties(p)
    o = [t for t in ov if {t["a"], t["b"]} == {"P", "P-reproc"}]
    check(len(o) == 1 and abs(o[0]["shift_ms"] - (-5.0 if o[0]["b"] == "P-reproc" else 5.0)) < 0.5 and
          len(ov) == 1, f"overlapping vintage detected and tied along the overlap ({o[0]['shift_ms'] if o else '-'} ms)")
    MI.set_preferred(p, "P", ["P-reproc"])
    check("P-reproc" in p.display["hidden_lines"], "duplicate hidden (preferred version kept)")
    p.lines = [l for l in p.lines if l["name"] != "P-reproc"]
    p.display.pop("hidden_lines", None)
    p.display.pop("preferred_lines", None)
    p._line_cache.clear()
    # ---- exports: corrected SEG-Y, tie table, residual positions
    MI.apply_solution(p, sol)
    from .volume import Line2D
    out_sgy = MI.export_corrected_segy(p.line("P"), os.path.join(d, "P_corrected.sgy"))
    back = Line2D("Pc", out_sgy)
    check(np.allclose(back.raw_data(), p.line("P").data(), atol=1e-4) and np.allclose(back.x, p.line("P").x),
          "corrected SEG-Y export (IEEE, headers copied)")
    MI.apply_solution(p, {})
    MI.tie_positions(p, ties)
    tab = MI.write_tie_table(ties, sol, os.path.join(d, "ties.csv"))
    check(all(t.get("xy") for t in ties) and sum(1 for _ in open(tab)) == len(ties) + len(sol) + 5,
          "tie positions for the residual map; tie table CSV")

    # apply: corrected lines match the 3D; picks follow
    h = Horizon("H")
    h.data["2D:P"] = np.full(90, 500.0, np.float32)
    p.horizons.items.append(h)
    MI.apply_solution(p, sol)
    ln = p.line("P")
    tr3, inside = MI.extract_3d_along(vol, ln.x, ln.y)
    m = (vol.times > 100) & (vol.times < 1100)
    cc = np.corrcoef(ln.data()[inside][:, m].ravel(), tr3[inside][:, m].ravel())[0, 1]
    cc0 = np.corrcoef(ln.raw_data()[inside][:, m].ravel(), tr3[inside][:, m].ravel())[0, 1]
    check(cc > 0.97 and cc0 < 0.7, f"corrected 2D line matches the 3D (corr {cc0:.2f} -> {cc:.2f})")
    check(abs(h.data["2D:P"][0] - (500.0 + sol["P"]["shift_ms"])) < 1e-3, "2D horizon picks moved with the correction")
    p.save()
    from .project import SeisProject
    q = SeisProject.open(p.path)
    check(q.line("P").correction is not None and abs(q.line("P").correction[0] - sol["P"]["shift_ms"]) < 1e-6,
          "corrections saved with the project and applied on reopening")
    MI.apply_solution(q, {})
    check(abs(q.horizons.get("H").data["2D:P"][0] - 500.0) < 1e-3 and q.line("P").correction is None,
          "reset corrections restores lines and picks")
    return True


def run_gui(check, tmp):
    from .gui.mistie_dialog import MistieDialog
    import tempfile
    d = tempfile.mkdtemp(dir=tmp)
    run(lambda c, m: None, d)  # builds the project in d/mistie
    from .project import SeisProject
    p = SeisProject.open(os.path.join(d, "mistie", "m", "g.seisproj.json"))
    dlg = MistieDialog(p)
    dlg.time_window = (100.0, 1100.0)
    dlg.max_lag = 40.0
    ties, sol = dlg.compute()
    check(len(ties) >= 5 and set(sol) >= {"P", "Q", "R"}, "mistie dialog: ties + solution")
    dlg.apply()
    check(set(p.display.get("mistie", {})) == {"P", "Q", "R"}, "mistie dialog: apply stores corrections")
    dlg.reset()
    check(not p.display.get("mistie"), "mistie dialog: reset")
    dlg.flex, dlg.lam, dlg.knot_traces = True, 1.0, 20
    ties, sol = dlg.compute()
    check(all("flex" in sol[k] for k in ("P", "Q", "R")), "mistie dialog: flex mode adds along-line shifts")
    dlg.apply()
    check("flex" in p.display["mistie"]["P"] and np.ndim(p.line("P").correction[0]) == 1,
          "flex corrections applied per trace on the line")
    n = dlg.export_segy(d)
    check(n == 3 and os.path.exists(os.path.join(d, "P_mistie_corrected.sgy")), "dialog: corrected SEG-Y export")
    dlg.reset()
    dlg.flex = False
    dlg.groups, dlg.key_lines, dlg.time_variant = {"P": "S1", "Q": "S1"}, [], True
    ties, sol = dlg.compute()
    check(sol["P"]["group"] == "S1" and "tv" in sol["R"], "dialog: groups + time-variant")
    dlg.horizon = None
    pos = dlg.show_residuals()
    check(len(pos) == len(ties), "dialog: residual map positions")
    return True
