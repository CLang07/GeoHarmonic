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
    """Item 16c: deterministic / multitaper wavelets, phase, library, log conditioning, angle synthetics,
    multi-well batch."""
    from . import tie as TK
    from . import welltie as WT
    rng = np.random.default_rng(7)
    dt = 4.0
    n = 400
    r = rng.standard_normal(n) * (rng.random(n) < 0.35)
    true = TK.rotate_phase(WT.ricker(25.0, dt, 128.0), 30.0)
    s = np.convolve(r, true, "same")
    s = s + 0.1 * s.std() * rng.standard_normal(n)
    nh = len(true) // 2
    dw = TK.deterministic_wavelet(r, s, nh)
    c = np.corrcoef(dw["wavelet"], true)[0, 1]
    check(c > 0.95 and abs(dw["phase_deg"] - 30.0) < 10 and dw["pep"] > 0.8,
          f"deterministic wavelet = true wavelet at 10 % noise (corr {c:.3f}, phase {dw['phase_deg']:.0f} deg, "
          f"PEP {dw['pep']:.2f})")
    mt = TK.multitaper_wavelet(r, s, nh, 5, dt)
    cm = np.corrcoef(mt["wavelet"], true)[0, 1]
    fr = mt["freqs_hz"]
    band = (fr > 10) & (fr < 45)
    check(cm > 0.9 and mt["coherence"][band].mean() > 0.8, f"multitaper wavelet (corr {cm:.3f}, coherence in band "
                                                            f"{mt['coherence'][band].mean():.2f})")
    ph = [TK.wavelet_phase(TK.rotate_phase(WT.ricker(30.0, 2.0), a)) for a in (-90, 0, 45, 170)]
    check(all(abs((p_ - a + 180) % 360 - 180) < 3 for p_, a in zip(ph, (-90, 0, 45, 170))),
          "constant-phase estimate of rotated wavelets (" + ", ".join(f"{p_:.0f}" for p_ in ph) + " deg)")
    f_, amp, _ = TK.wavelet_spectrum(WT.ricker(30.0, 2.0), 2.0)
    check(abs(f_[np.argmax(amp)] - 30.0) < 3.0, f"wavelet spectrum peaks at the Ricker frequency ({f_[np.argmax(amp)]:.1f} Hz)")
    lib = TK.WaveletLibrary(os.path.join(tmp, "wavelets.json"))
    lib.add("det W-1", dw["wavelet"], dt, source="deterministic", well="W-1")
    lib.add("ricker 30", WT.ricker(30.0, 2.0), 2.0)
    lib.save()
    lib2 = TK.WaveletLibrary(os.path.join(tmp, "wavelets.json"))
    w4 = lib2.get("ricker 30", 4.0)
    check(lib2.names() == ["det W-1", "ricker 30"] and np.allclose(lib2.get("det W-1"), dw["wavelet"])
          and np.allclose(w4, WT.ricker(30.0, 4.0), atol=1e-6), "wavelet library saved, reloaded, resampled 2 -> 4 ms")
    sh = [TK._shift0(true, k) for k in (0, 2, -3)]
    av = TK.average_wavelet([x * g for x, g in zip(sh, (1.0, 2.0, 0.5))])
    check(np.corrcoef(av, true)[0, 1] > 0.98, "average wavelet of shifted / scaled copies")
    # log conditioning
    x = np.sin(np.linspace(0, 20, 500)) * 10 + 50
    xs = x.copy()
    xs[[50, 200, 201, 420]] += [80, -60, -55, 90]
    cl, bad = TK.despike(xs)
    check(set(np.nonzero(bad)[0]) >= {50, 200, 201, 420} and bad.sum() <= 6 and np.abs(cl - x).max() < 2.0,
          f"despike: {bad.sum()} spikes replaced")
    cal = np.full(100, 8.6)
    cal[40:50] = 11.0
    check(TK.washout_mask(cal, 8.5).sum() == 10, "washout mask from caliper vs bit size")
    z = np.arange(0, 100, 0.5)
    lay = (z // 1.0) % 2 == 0
    vp, vs, rho = np.where(lay, 2500.0, 3500.0), np.where(lay, 1100.0, 1900.0), np.where(lay, 2.2, 2.5)
    vpb, vsb, rhob = TK.backus(z, vp, vs, rho, 20.0)
    m = 1.0 / (0.5 / (2.2 * 2500.0 ** 2) + 0.5 / (2.5 * 3500.0 ** 2))
    check(abs(vpb[100] - np.sqrt(m / 2.35)) < 20 and vpb[100] < 3000.0 and abs(rhob[100] - 2.35) < 0.01,
          f"Backus average of thin layers ({vpb[100]:.0f} m/s vs arithmetic 3000)")
    # angle synthetics: shale over gas sand (class 3) - more negative with angle; small angles = AI reflectivity
    vp, vs, rho = np.array([2700.0, 2500.0]), np.array([1200.0, 1600.0]), np.array([2.4, 2.0])
    ar = TK.rc_aki_richards(vp, vs, rho, [0, 10, 20, 30])[0]
    shu = TK.rc_shuey(vp, vs, rho, [0, 10, 20, 30])[0]
    ai = (2500 * 2.0 - 2700 * 2.4) / (2500 * 2.0 + 2700 * 2.4)
    check(abs(ar[0] - ai) < 0.005 and ar[3] < ar[0] < 0 and np.abs(ar[:3] - shu[:3]).max() < 0.01,
          f"angle reflectivity: AR {ar[0]:.3f} -> {ar[3]:.3f} at 30 deg (class 3), Shuey agrees to 20 deg")
    g, rr = TK.angle_synthetics(np.r_[np.full(50, 2700.0), np.full(50, 2500.0)], np.r_[np.full(50, 1200.0),
                                np.full(50, 1600.0)], np.r_[np.full(50, 2.4), np.full(50, 2.0)], [0, 30],
                                WT.ricker(30.0, 2.0))
    check(g.shape == (100, 2) and abs(rr[50, 0] - ar[0]) < 1e-9 and abs(g[:, 1]).max() > abs(g[:, 0]).max(),
          "angle gather: reflector at the right sample, far angle brighter")
    batch_checks(check, tmp)


def batch_project(tmp, nwell=10, trend=6.0):
    """nwell wells (same layered logs, each with a checkshot) inside a 3D whose seismic is the synthetic delayed by
    8 ms + trend ms per km eastwards."""
    from .test_welltie import V0, K, true_twt, LAT, LON
    from .welltie import wc
    from . import welltie as WT
    wc()
    from well_correlation.selftest import _write_las
    from well_correlation.geo import tm_forward
    from .project import SeisProject
    from .synth import write_3d
    from .volume import import_3d
    d = os.path.join(tmp, "batch")
    rng = np.random.default_rng(9)
    md = np.arange(900.0, 2600.0, 0.1524)
    z = md - 25.0
    edges = np.sort(rng.uniform(900, 2600, 70))
    pert = rng.uniform(-0.12, 0.12, 71)[np.searchsorted(edges, md)]
    v = (V0 + K * z) * (1 + pert)
    logs = {"DT": 304800.0 / v, "RHOB": 0.31 * v ** 0.25 * (1 + 0.5 * pert), "GR": 60 + 60 * (pert < 0)}
    hdr = ["Well,UWI,Latitude,Longitude,Elevation_m,Elev_Ref,TD_m"]
    lon_m = 1.0 / (111320.0 * np.cos(np.radians(LAT)))
    offs = rng.uniform(100, 800, (nwell, 2))
    for k in range(nwell):
        name = f"W-{k}"
        wd = os.path.join(d, "wells", name, "las")
        os.makedirs(wd)
        _write_las(os.path.join(wd, f"{name}.las"), name, md, logs, {"DT": "US/F", "RHOB": "G/C3", "GR": "GAPI"})
        hdr.append(f"{name},{k},{LAT + offs[k, 1] / 110950.0},{LON + offs[k, 0] * lon_m},25.0,KB,2600")
    with open(os.path.join(d, "wells_header.csv"), "w") as fh:
        fh.write("\n".join(hdr) + "\n")
    for k in range(nwell):  # every well has a checkshot, so the batch test measures the tie, not sonic drift
        cs = os.path.join(d, "wells", f"W-{k}", "checkshot")
        os.makedirs(cs)
        with open(os.path.join(cs, "cs.csv"), "w") as fh:
            fh.write("TVD_SRD_m,TWT_ms\n" + "\n".join(f"{zz:.1f},{true_twt(zz):.3f}"
                                                       for zz in np.arange(100.0, 2700.0, 100.0)))
    P = wc().project.Project
    p = P.from_folder(d)
    wpath = os.path.join(d, "b.wcproj.json")
    p.save(wpath)
    x0, y0 = tm_forward(LAT, LON)
    sp = SeisProject()
    sp.path = os.path.join(d, "b.seisproj.json")
    sp.well_project = wpath
    ws = sp.wellset()
    w = ws.well("W-0")
    ws.wcp.load_well("W-0")
    tdt = WT.WellTD("W-0", np.arange(0, 3000, 10.0), true_twt(np.arange(0, 3000, 10.0)), "truth")
    t, ai = WT.logs_in_time(w, tdt, 4.0)
    _, syn = WT.synthetic(t, ai, WT.ricker(30.0, 4.0))
    nil = nxl = 40
    ns = 700
    tt = np.arange(ns) * 4.0
    cube = np.zeros((nil, nxl, ns), np.float32)
    for j in range(nxl):  # crossline = east (x)
        cube[:, j] = np.interp(tt, t + 8.0 + trend * (25.0 * j) / 1000.0, syn, left=0, right=0)
    write_3d(os.path.join(d, "b.sgy"), cube, x0=round(x0), y0=round(y0))
    store = os.path.join(d, "b_seismic", "S")
    import_3d(os.path.join(d, "b.sgy"), store)
    sp.volumes.append({"name": "S", "store": store, "segy": ""})
    sp.save()
    return sp, round(x0)


def batch_checks(check, tmp):
    from . import tie as TK
    sp, x0 = batch_project(tmp)
    ws = sp.wellset(reload=True)
    vol = sp.volume("S")
    rows = TK.batch_tie(ws, "3D", vol, stretches=np.linspace(0.98, 1.02, 9))
    good = [r for r in rows if "error" not in r]
    exp = [8.0 + 6.0 * (r["x"] - x0) / 1000.0 for r in good]
    err = max(abs(r["shift_ms"] - e) for r, e in zip(good, exp)) if good else 99
    summ = TK.batch_summary(rows)
    check(len(good) == 10 and err < 1.5 and all(r["corr"] > 0.9 for r in good) and summ["n"] == 10,
          f"multi-well batch: 10 wells tied, bulk shift trend recovered (max err {err:.2f} ms)")
    from .welltie import ricker
    aw = summ["average_wavelet"]
    rk = ricker(30.0, 4.0, 128.0)
    check(len(aw) == len(rk) and np.corrcoef(aw, rk)[0, 1] > 0.9 and abs(TK.wavelet_phase(aw)) < 15,
          f"batch: average deterministic wavelet ~ the 30 Hz Ricker (corr {np.corrcoef(aw, rk)[0, 1]:.2f})")
    g = TK.shift_map(rows)
    xs = np.array([r["x"] for r in good])
    ys = np.array([r["y"] for r in good])
    gv = g.sample(xs, ys)
    check(g is not None and np.nanmax(np.abs(gv - np.array([r["shift_ms"] for r in good]))) < 0.5,
          "batch: bulk-shift map honours the wells")
    p = TK.write_batch_csv(os.path.join(tmp, "batch.csv"), rows)
    with open(p) as fh:
        check(len(fh.read().strip().splitlines()) == 11, "batch table CSV written")
    # deviated well: seismic sampled along the wellbore follows the path, not the surface location
    from .laterals import Trajectory
    from .welltie import WellTD
    zz = np.arange(0.0, 2800.0, 10.0)
    xx = float(vol.xy(vol.inlines[5], vol.xlines[2])[0]) + 0.5 * zz  # drifts 0.5 m east per m of depth
    yy = np.full(len(zz), float(vol.xy(vol.inlines[5], vol.xlines[2])[1]))
    tr = Trajectory("D-1", zz, xx, yy, zz)
    from .test_welltie import true_twt
    td = WellTD("D-1", zz, true_twt(zz), "truth")
    tt = np.arange(1200.0, 1600.0, 4.0)
    seis, sx, sy = TK.seismic_along_wellbore(tr, td, vol, tt)
    il, xl = vol.ilxl_at(sx, sy)
    jj = np.rint((xl - vol.xl0) / vol.xl_step).astype(int)
    want = np.array([np.asarray(vol.ilxl[5, j], np.float32)[int(round(t_ / 4.0))] * vol.inv for j, t_ in zip(jj, tt)])
    check(np.allclose(seis, want, atol=1e-4) and jj[-1] - jj[0] >= 3,
          f"deviated well: seismic along the wellbore ({jj[0]} -> {jj[-1]} crosslines over the window)")


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
    shown = float(dlg.tied_times()[280])  # displayed (warped) time of a deeper synthetic event
    dlg.pick_event(2.0, shown)
    ok2 = dlg.pick_event(8.0, shown + 5.0)
    check(ok2 and len(dlg.state.pairs) == 2 and abs(float(dlg.tied_times()[280]) - (shown + 5.0)) < 0.05,
          "dialog: second event pair lands where clicked (click mapped back through the existing warp)")
    dlg.undo()
    tt = dlg.tied_times()
    dlg.window = (float(tt[100]), float(tt[300]))
    m = dlg._win_mask()
    check(m.sum() == 201 and m[100] and m[300] and not m[99], "dialog: window read in tied (displayed) time")
    dlg.window = (99990.0, 99999.0)
    check(dlg._win_mask().all() and dlg.correlate() is not None, "dialog: empty window falls back to the whole log")
    dlg.window = None
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
    sp.wellset(reload=True)
    # 16c in the dialog: deterministic wavelet, library, despike, along wellbore, angle gather, batch
    dlg = WellTieDialog(sp, ("3D", "S"))
    dlg.well_name = "W-1"
    dlg.wavelet_kind = "Ricker"
    dlg.compute()
    dlg.auto()
    c_r = dlg.tie_corr()
    res = dlg.estimate_wavelet()
    check(res is not None and dlg.wavelet_kind == "Deterministic" and abs(res["phase_deg"]) < 20 and res["pep"] > 0.8
          and dlg.tie_corr() > c_r - 0.05, f"dialog: deterministic wavelet (phase {res['phase_deg']:+.0f} deg, PEP "
                                           f"{res['pep']:.2f}), synthetic corr {c_r:.3f} -> {dlg.tie_corr():.3f}")
    mt = dlg.estimate_wavelet(method="multitaper")
    check(mt is not None and mt["pep"] > 0.7 and abs(mt["phase_deg"]) < 25, f"dialog: multitaper wavelet (PEP {mt['pep']:.2f})")
    dlg.estimate_wavelet(method="least squares")
    dlg.save_wavelet("W-1 det")
    from .tie import WaveletLibrary
    lib = WaveletLibrary(dlg.library_path())
    w = dlg.use_library_wavelet("W-1 det")
    check("W-1 det" in lib.names() and w is not None and len(w) == len(dlg.wavelet), "dialog: wavelet library save + use")
    dlg.despike_logs = True
    dc = dlg.compute()
    dlg.despike_logs = False
    check(dc is not None and len(dc["syn"]) > 100, "dialog: synthetic from despiked logs")
    dlg.along_wellbore = True
    da = dlg.compute()
    dlg.along_wellbore = False
    check(da is not None and np.allclose(da["seis"], da["traces"][len(da["traces"]) // 2], atol=1e-5),
          "dialog: seismic along the (vertical) wellbore = trace at the well")
    ag = dlg.angle_gather()
    check(ag is not None and ag[1].shape[1] == 5 and np.isfinite(ag[1]).all(), "dialog: angle gather 0-40 deg")
    rows, summ = dlg.batch()
    check(len(rows) == 2 and summ["n"] >= 1 and os.path.exists(os.path.join(sp.seismic_dir(), "well_tie_batch.csv"))
          and "average (batch)" in WaveletLibrary(dlg.library_path()).names(),
          f"dialog: tie all wells ({summ['n']} tied), table + average wavelet saved")
    return True
