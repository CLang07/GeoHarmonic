"""Tests for PLAN 24a / 24b: resolution (spectrum, tuning, wedge, beds, recommendation) and single-well inversion."""
import math
import os

import numpy as np


def run(check, tmp):
    from . import resolution as R
    from .welltie import ricker
    from .inversion.poststack import synthetic_ln
    dt = 2.0
    rng = np.random.default_rng(0)
    w = ricker(30.0, dt)
    refl = rng.normal(size=(60, 500)) * 0.05
    tr = np.array([np.convolve(r, w, "same") for r in refl])
    st_hi = R.spectrum_stats(tr + rng.normal(size=tr.shape) * 0.03, dt)
    st_lo = R.spectrum_stats(tr + rng.normal(size=tr.shape) * 0.003, dt)
    check(abs(st_hi["f_dom"] - 30) < 4 and abs(st_lo["f_dom"] - 30) < 4, f"24a: dominant frequency of Ricker-30 data "
          f"({st_hi['f_dom']:.1f} Hz)")
    check(st_lo["f_hi"] > st_hi["f_hi"] + 5 and st_lo["snr_db"] > st_hi["snr_db"] + 10,
          f"24a: usable band widens as the noise floor drops ({st_hi['f_hi']:.0f} -> {st_lo['f_hi']:.0f} Hz)")
    noise = rng.normal(size=tr.shape) * 0.03
    st_n = R.spectrum_stats(tr + noise, dt, noise_traces=noise)
    check(abs(st_n["f_hi"] - st_hi["f_hi"]) < 15, "24a: noise floor from a noise window agrees with the tail estimate")
    # tuning: wedge vs lambda / 4 and the Ricker analytic value (Kallweit & Wood 1982: TWT 1 / (2.31 fp))
    wd = R.wedge_model(np.arange(0, 60, 0.25), w, dt)
    check(abs(wd["tuning_ms"] - 1000 / (2.31 * 30)) < 1.0, f"24a: wedge tuning {wd['tuning_ms']:.2f} ms = Kallweit-Wood "
          f"{1000 / (2.31 * 30):.2f} ms for a 30 Hz Ricker")
    q4, q8 = R.tuning_thickness(3000.0, 30.0)
    tun_m = wd["tuning_ms"] / 2000 * 3000
    check(abs(q4 - 25.0) < 1e-9 and abs(q8 - 12.5) < 1e-9 and abs(tun_m / q4 - 1) < 0.2,
          f"24a: lambda/4 = 25 m, wedge tuning {tun_m:.1f} m within 20 % of it")
    thick = wd["thick_ms"] > 30
    check(np.allclose(wd["apparent_ms"][thick], wd["thick_ms"][thick], atol=0.6) and 1.3 < wd["amp_ratio"] < 1.7,
          f"24a: thick beds: apparent = true thickness; tuning amplitude x{wd['amp_ratio']:.2f}")
    thin = (wd["thick_ms"] > 0) & (wd["thick_ms"] < 8)
    check(np.all(wd["apparent_ms"][thin] > wd["thick_ms"][thin]), "24a: below tuning the apparent thickness stays "
          "at the limit (thin beds look thicker)")
    # beds from a log flag
    depth = np.arange(1000, 1100, 0.5)
    flag = np.zeros(len(depth), bool)
    flag[10:20] = True   # 5 m
    flag[40:44] = True   # 2 m
    flag[100:160] = True  # 30 m
    b = R.bed_thicknesses(depth, flag)
    check(sorted(b.tolist()) == [2.0, 5.0, 30.0], "24a: bed thicknesses from a reservoir flag")
    # spectral balancing broadens the spectrum
    bal = R.spectral_balance(tr, dt, st_hi["f_lo"], st_hi["f_hi"])
    sb = R.spectrum_stats(bal, dt)
    def rel(st_, fq):
        return np.interp(fq, st_["freqs"], st_["amp"]) / st_["amp"].max()
    check(sb["f_centroid"] > st_hi["f_centroid"] + 2 and rel(sb, 55) > 2 * rel(st_hi, 55),
          f"24a: spectral balancing flattens the band and raises the centroid frequency "
          f"({st_hi['f_centroid']:.0f} -> {sb['f_centroid']:.0f} Hz)")
    rec = R.recommend(st_hi, 3000.0, b, dt)
    check(rec["tuning_balanced_m"] < rec["tuning_m"] and rec["gain_pct"] > 10 and abs(rec["f_needed"] - 150) < 1e-6 and
          rec["resolved_balanced"] >= rec["resolved_now"] and "needs" in rec["verdict"],
          f"24a: recommendation - {rec['verdict']}")
    bw = R.band_wavelet(10, 60, dt)
    f, A = R.amplitude_spectrum(bw[None], dt, 1024)
    flat = (f > 15) & (f < 55)
    check(A[flat].min() > 0.6 * A[flat].max(), "24a: balanced wavelet is flat over the band")
    # ---- 24b helpers
    r90 = R.rotate_phase(R.rotate_phase(w, 90), 90)
    check(np.corrcoef(r90, -w)[0, 1] > 0.98, "24b: two 90 deg phase rotations = polarity flip")
    st2 = R.spectrum_stats(R.stretch(w, 1.5)[None], dt)
    st1 = R.spectrum_stats(w[None], dt)
    check(abs(st2["f_dom"] / st1["f_dom"] - 1.5) < 0.15, "24b: wavelet frequency stretch")
    # ---- 24b single-well inversion
    n = 400
    z = np.log(6000) + np.cumsum(np.where(rng.random(n) < 0.06, rng.normal(0, 0.08, n), 0)) + np.linspace(0, 0.3, n)
    phi = 0.2 - 0.6 * (z - z.mean()) + rng.normal(0, 0.01, n)
    s_clean = synthetic_ln(z[None], w)[0] * 1.7
    ai = np.exp(z)
    ai[:30] = np.nan
    ai[-30:] = np.nan
    good = R.single_well_inversion(s_clean + rng.normal(0, 0.02 * np.std(s_clean), n), ai, w, dt, n_mc=40, prop=phi)
    check(good["coverage"] > 0.75 and np.all(good["p10"][good["mask"]] <= good["p90"][good["mask"]]),
          f"24b: Monte-Carlo P10-P90 band contains the band-limited log ({100 * good['coverage']:.0f} % of samples)")
    check(good["corr_inv"] > 0.8 and good["verdict"].startswith("YES") and good["corr_full"] > good["corr_lfm"],
          f"24b: clean data - {good['verdict']}")
    bad = R.single_well_inversion(s_clean + rng.normal(0, 3.0 * np.std(s_clean), n), ai, w, dt, n_mc=40, prop=phi)
    check(bad["corr_inv"] < good["corr_inv"] - 0.2 and bad["rel_width"] > good["rel_width"] and
          bad["verdict"].startswith("NO"), f"24b: noisy data - {bad['verdict']}")
    check(abs(good["prop_coef"][0] + 0.6) < 0.05 and good["prop_width"] < good["prop_contrast"],
          "24b: property band from the well AI-property regression")
    # ---- frequency map on a synthetic survey
    from .synth import make_project
    d = os.path.join(tmp, "res3d")
    os.makedirs(d)
    proj, truth = make_project(d, nil=12, nxl=14, ns=200, noise=0.05)
    g = R.frequency_map(proj.volume("Synth"), None, None, block=4)
    check(g.z.shape == (12, 14) and np.all(np.isfinite(g.z)) and 15 < np.nanmedian(g.z) < 45,
          f"24a: dominant-frequency map (median {np.nanmedian(g.z):.0f} Hz)")


def run_seeds(check, tmp):
    """24d: synthetic events -> snapped seed patches -> RGT control across a fault."""
    from . import wellseeds as WS
    from .rgt import solve as S
    from .test_rgt import geology_project
    from .welltie import synthetic, ricker
    # events of a blocky log's synthetic
    t = np.arange(0, 800, 2.0)
    ai = np.full(len(t), 5000.0)
    ai[t >= 200] = 6500.0   # strong positive (peak)
    ai[t >= 420] = 5200.0   # strong negative (trough)
    ai[t >= 600] = 5300.0   # weak
    _, syn = synthetic(t, ai, ricker(30.0, 2.0))
    ev = WS.synthetic_events(t, syn, 0.5)  # Ricker side lobes are 0.45
    check([(round(e[0]), e[1]) for e in ev] == [(200, 1), (420, -1)], f"24d: strong synthetic events (peak at 200 ms, "
          f"trough at 420 ms; weak 600 ms ignored): {[(round(e[0]), e[1]) for e in ev]}")
    p, G = geology_project(tmp, "seeds", fault=True)
    vol = p.volume("G")
    hz, tr = G["horizons"], G["tau_ref"]
    s_h = int(np.argmin(np.abs(tr - 500)))
    wells = [(5, 10), (30, 15), (8, 50), (32, 52), (20, 22)]  # 2 in the hanging wall, 3 in the footwall
    picks = []
    for i, j in wells:
        x, y = vol.xy(vol.inlines[i], vol.xlines[j])
        picks.append((float(x), float(y), float(hz[s_h][i, j]) + 2.0))
    hs = WS.seed_horizons(vol, {"H": picks}, "G", radius=5)
    g = hs[0].get("3D:G")
    ok = np.isfinite(g)
    d = (g - hz[s_h])[ok]
    check(len(hs) == 1 and hs[0].name == "seed: H" and ok.sum() > 5 * 80 and
          np.median(np.abs(d - np.median(d))) < 3.0, f"24d: seed patches snapped to the event at 5 wells ({ok.sum()} "
          f"traces; follow the reflector to {np.median(np.abs(d - np.median(d))):.2f} ms median)")

    def spread(rv):
        return float(np.ptp([np.interp(g[i, j], vol.times, rv.trace(i, j)) for i, j in wells]))
    rv0, _ = S.compute_rgt_3d(vol, S.RGTParams(fault_links="none"), out_dir=os.path.join(vol.store, "rgt_noseed"))
    rv1, _ = S.compute_rgt_3d(vol, S.RGTParams(fault_links="none"), horizons=hs, vol_name="G",
                              out_dir=os.path.join(vol.store, "rgt_seeded"))
    s0, s1 = spread(rv0), spread(rv1)
    check(s1 < 2.0 and s0 > 10, f"24d: RGT honours the well seeds across the fault (RGT spread at the 5 wells "
          f"{s0:.1f} -> {s1:.2f})")
