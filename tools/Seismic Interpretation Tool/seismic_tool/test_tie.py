"""Item 16: well tie upgrade - auto-correlate, stretch/squeeze (event pairs + DTW), drift QC, calibrated sonic,
wavelets, multi-well batch."""
import os

import numpy as np


def case(seed=1, dt=2.0, f=30.0, t0=1000.0, t1=2400.0):
    """Random sparse reflectivity, Ricker synthetic on a regular grid."""
    from . import welltie as WT
    rng = np.random.default_rng(seed)
    t = np.arange(t0, t1, dt)
    r = rng.standard_normal(len(t)) * (rng.random(len(t)) < 0.3)
    syn = np.convolve(r, WT.ricker(f, dt), "same")
    return t, r, syn


def seismic_from(t, syn, ts, shift=0.0, stretch=1.0, phase=0.0, anchor=None, warp=None, noise=0.0, seed=2):
    """Seismic trace = the synthetic moved by a known tie (and phase rotation) + noise."""
    from .tie import rotate_phase
    anchor = t[0] if anchor is None else anchor
    tm = t + shift + (stretch - 1.0) * (t - anchor)
    if warp is not None:
        tm = tm + warp(tm)
    s = np.interp(ts, tm, rotate_phase(syn, phase), left=0.0, right=0.0)
    return s + noise * np.std(s) * np.random.default_rng(seed).standard_normal(len(s))


def run(check, tmp):
    from . import tie as TK
    t, r, syn = case()
    ts = np.arange(0.0, 4000.0, 2.0)
    # ---- 16a: correlate (lag, polarity, phase, cycle skip)
    s = seismic_from(t, syn, ts, shift=12.3, phase=35.0, noise=0.1)
    c = TK.correlate(t, syn, ts, s, 60.0)
    check(abs(c["env_lag_ms"] - 12.3) < 1.0 and abs(c["best_phase_deg"] - 35.0) < 5.0 and c["polarity"] == "normal"
          and not c["cycle_skip"], f"correlate: lag {c['env_lag_ms']:.2f} ms (12.3), phase {c['best_phase_deg']:.1f} deg "
                                   f"(35), corr {c['best_phase_corr']:.2f}")
    rv = TK.correlate(t, syn, ts, -seismic_from(t, syn, ts, shift=4.0, noise=0.1), 60.0)
    check(rv["polarity"] == "reverse" and abs(abs(rv["best_phase_deg"]) - 180.0) < 5.0 and abs(rv["best_lag_ms"] - 4.0) < 1.0,
          f"correlate: reversed polarity detected (phase {rv['best_phase_deg']:.0f} deg)")
    # ringing (narrow-band) case: the next cycle correlates almost as well -> cycle-skip warning
    tt = np.arange(1000.0, 1600.0, 2.0)
    ring = np.sin(2 * np.pi * 25.0 * tt / 1000.0) * np.hanning(len(tt)) ** 0.2
    cs = TK.correlate(tt, ring, ts, seismic_from(tt, ring, ts, shift=6.0), 60.0)
    check(cs["cycle_skip"] and cs["second_peak"] is not None and abs(abs(cs["second_peak"][0] - cs["best_lag_ms"]) - 40.0) < 4,
          f"cycle-skip warning on a ringing synthetic (second peak at {cs['second_peak'][0]:+.0f} ms, "
          f"corr {cs['second_peak'][1]:.2f})")
    # ---- 16a: fast auto tie (shift + stretch about the window top), held-out score, NRMS
    s = seismic_from(t, syn, ts, shift=12.3, stretch=1.02, noise=0.1)
    a = TK.auto_tie_fft(t, syn, ts, s, 60.0, np.linspace(0.96, 1.04, 9))
    check(abs(a["shift_ms"] - 12.3) < 1.0 and abs(a["stretch"] - 1.02) < 0.005 and a["corr"] > 0.95
          and a["corr_holdout"] > 0.9 and a["nrms"] < 40,
          f"auto tie (FFT): shift {a['shift_ms']:.2f} ms, stretch {a['stretch']:.4f}, corr {a['corr']:.3f}, "
          f"held-out corr {a['corr_holdout']:.3f}, NRMS {a['nrms']:.0f} %")
    w = TK.auto_tie_fft(t, syn, ts, s, 60.0, np.linspace(0.96, 1.04, 9), window=(1400, 2000))
    check(abs(w["stretch"] - 1.02) < 0.01 and abs(w["t_centre"] - 1400) < 2.1
          and abs(w["shift_ms"] - (12.3 + 0.02 * (w["t_centre"] - 1000.0))) < 1.5,
          f"auto tie in a window: anchored at the window top ({w['t_centre']:.0f} ms), shift {w['shift_ms']:.1f} ms")
    # best trace among neighbours
    trs = np.array([seismic_from(t, syn, ts, shift=10.0, noise=n_, seed=k) for k, n_ in enumerate((2.0, 1.5, 0.05, 1.0, 2.0))])
    b = TK.best_trace(t, syn, ts, trs, 30.0, offsets=[-2, -1, 0, 1, 2])
    check(b["index"] == 2 and b["offset"] == 0 and abs(b["lag_ms"] - 10.0) < 1.0, "best trace within 2 traces found")
    # ---- 16b: DTW with a 3-segment velocity error
    def warp(tm):
        return np.interp(tm, [1000, 1400, 1800, 2400], [0.0, 8.0, 2.0, 10.0])
    s = seismic_from(t, syn, ts, warp=warp, noise=0.05)
    wi, wo, lag = TK.dtw_warp(t, syn, ts, s, 20.0, 0.10)
    err = np.abs(lag - warp(t))[20:-20]
    st = TK.TieState(0.0, 1.0, t[0])
    c0 = TK._pearson(np.interp(st.apply(t), ts, s), syn)
    st.set_warp(wi, wo)
    c1 = TK._pearson(np.interp(st.apply(t), ts, s), syn)
    check(err.max() < 1.0 and TK.max_strain_of(wi, wo) <= 0.1 + 1e-9 and c1 > 0.95,
          f"DTW stretch/squeeze recovers a 3-segment velocity error: max err {err.max():.2f} ms, "
          f"corr {c0:.2f} -> {c1:.2f}")
    s2 = seismic_from(t, syn, ts, warp=lambda tm: np.interp(tm, [1000, 1100, 2400], [0.0, 15.0, 15.0]), noise=0.05)
    wi2, wo2, lag2 = TK.dtw_warp(t, syn, ts, s2, 20.0, 0.05)
    check(TK.max_strain_of(wi2, wo2) <= 0.05 + 1e-9, f"DTW respects the 5 % velocity-change cap where the truth "
                                                      f"needs 15 % (got {100 * TK.max_strain_of(wi2, wo2):.1f} %)")
    # event pairs + undo
    st = TK.TieState(5.0, 1.0, 1000.0)
    ok1 = st.add_pair(1200.0, 1206.0) and st.add_pair(1800.0, 1795.0)
    bad = st.add_pair(1500.0, 1810.0)  # would fold time
    a_ = st.apply(np.array([1195.0, 1495.0, 1795.0, 2295.0]))  # raw -> bulk +5 -> pairs
    ok2 = np.allclose(a_, [1206.0, 1500.5, 1795.0, 2295.0])
    st.undo()
    check(ok1 and not bad and ok2 and len(st.pairs) == 1 and np.allclose(st.apply(1795.0), 1806.0),
          "event pairs: piecewise warp, folding pair rejected, undo")
    rt = TK.TieState.from_dict(st.to_dict())
    check(np.allclose(rt.apply(t), st.apply(t)), "tie state saved / restored (bulk + warp)")
    # ---- 16b: drift QC + checkshot-calibrated sonic + TD export
    z = np.arange(500.0, 3000.0, 1.0)
    true_t = 2000.0 * np.cumsum(np.r_[0, np.diff(z)] / (1800.0 + 0.6 * z)) + 600.0
    son = true_t - 0.004 * (z - 500.0)  # sonic drifts 10 ms over 2.5 km (dispersion)
    zc = np.arange(600.0, 3000.0, 300.0)
    tc = np.interp(zc, z, true_t)
    tcal, (dz_, dd) = TK.calibrate_sonic(z, son, zc, tc)
    inside = (z >= zc[0]) & (z <= zc[-1])
    check(np.abs(np.interp(zc, z, tcal) - tc).max() < 0.01 and np.abs(tcal - true_t)[inside].max() < 0.1
          and abs(dd[-1] - 0.004 * (zc[-1] - 500)) < 0.01,
          f"calibrated sonic honours the checkshot (drift {dd[0]:.1f} -> {dd[-1]:.1f} ms removed)")
    rows = TK.drift_table(lambda zz: np.interp(zz, z, son), zc, tc)
    check(len(rows) == len(zc) and abs(rows[-1]["drift_ms"] + 0.004 * (zc[-1] - 500)) < 0.05
          and np.isnan(rows[0]["v_cs"]) and abs(rows[1]["v_cs"] - rows[1]["v_fn"]) / rows[1]["v_cs"] < 0.01,
          "drift table: drift + interval velocities per checkshot level")
    from .welltie import read_checkshot_csv
    p = TK.write_td_csv(os.path.join(tmp, "td.csv"), zc, tc, "X-1")
    back = read_checkshot_csv(p)
    check(back is not None and np.allclose(back[0], zc) and np.allclose(back[1], tc, atol=1e-3),
          "tied time-depth exported as checkshot CSV and read back")
    wavelet_checks(check, tmp)
    return True


def wavelet_checks(check, tmp):
    """Item 16c - filled in by the wavelet / batch part."""
    from . import tie as TK
    if not hasattr(TK, "deterministic_wavelet"):
        return


def run_gui(check, tmp):
    """Well tie dialog: correlate, best trace, DTW, event pairs + undo, save with warp, drift QC, TD export."""
    from .gui.welltie_dialog import WellTieDialog
    from .test_welltie import well_project, seismic_project, SHIFT
    from .welltie import read_checkshot_csv
    d = os.path.join(tmp, "tie16_gui")
    os.makedirs(d)
    wpath, xy = well_project(d)
    sp = seismic_project(d, wpath, xy)
    dlg = WellTieDialog(sp, ("3D", "S"))
    dlg.well_name = "W-1"
    dlg.wavelet_kind = "Ricker"
    dlg.compute()
    res = dlg.auto()
    c = dlg.correlate()
    check(abs(res["shift_ms"] - SHIFT) < 1.5 and abs(c["best_lag_ms"]) < 1.0 and c["polarity"] == "normal",
          f"dialog: auto tie then Correlate shows ~0 residual lag ({c['best_lag_ms']:+.2f} ms)")
    b = dlg.find_best_trace()
    check(b is not None and len(b["all"]) >= 3, f"dialog: best trace among {len(b['all'])} traces")
    corr_bulk = dlg.tie_corr()
    lag = dlg.dtw()
    check(lag is not None and dlg.state.warp_kind == "dtw" and dlg.tie_corr() >= corr_bulk - 0.02
          and np.abs(lag).max() < 3.0, f"dialog: DTW after a good bulk tie keeps corr ({corr_bulk:.3f} -> {dlg.tie_corr():.3f})")
    dlg.undo()
    check(dlg.state.warp_in is None, "dialog: undo removes the DTW warp")
    t0 = float(dlg.tied_times()[200])
    dlg.pick_event(2.0, t0)
    ok = dlg.pick_event(8.0, t0 + 3.0)
    check(ok and len(dlg.state.pairs) == 1 and abs(float(dlg.tied_times()[200]) - (t0 + 3.0)) < 1e-6,
          "dialog: event pair picked on synthetic then seismic")
    dlg.save()
    tie = sp.display["well_ties"]["W-1"]
    td = sp.wellset().td_for("W-1")
    check(tie.get("warp_kind") == "pairs" and td.warp_in is not None
          and abs(float(td.twt(1500.0)) - float(dlg._tied_td().twt(1500.0))) < 1e-6,
          "dialog: tie saved with its warp and applied to the well's time-depth")
    rows = dlg.drift_qc()
    check(rows is not None and len(rows) > 5 and all(abs(r["drift_ms"] - SHIFT) < 6 for r in rows if 900 < r["z"] < 2400),
          "dialog: drift QC against the checkshot (drift ~ the tie shift)")
    p = dlg.export_td(os.path.join(d, "W-1_tied.csv"))
    back = read_checkshot_csv(p)
    check(back is not None and abs(np.interp(1500.0, back[0], back[1]) - float(td.twt(1500.0))) < 0.1,
          "dialog: tied time-depth exported")
    sp.display["td_mode"] = "calibrated sonic"
    ws = sp.wellset(reload=True)
    tdc = ws.td_for("W-1")
    check(tdc.source.startswith("sonic calibrated") and tdc.warp_in is not None, f"calibrated-sonic TD mode ({tdc.source})")
    sp.display.pop("td_mode")
    return True
