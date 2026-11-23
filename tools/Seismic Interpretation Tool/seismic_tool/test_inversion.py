"""Seismic inversion checks (PLAN 19): low-frequency model (19a), post-stack inversion (19b), pre-stack
simultaneous inversion (19c), GUI logic (19d)."""
import os

import numpy as np

DT = 4.0
WELL_IJ = [(4, 5), (25, 6), (6, 24), (24, 25)]
BLIND_IJ = (15, 15)


def earth(nil=30, nxl=30, ns=400):
    """Known elastic earth on a time grid: horizons h1 / h2 (h2 converging east), ln AI / Vp / rho conformable to
    the zones (proportional between the horizons, parallel above / below) + a depth trend."""
    t = np.arange(ns) * DT
    I, J = np.meshgrid(np.arange(nil), np.arange(nxl), indexing="ij")
    h1 = 600.0 + 2.0 * I + 1.0 * J
    h2 = h1 + 300.0 - 3.0 * J
    T = t[None, None, :]
    a1, a2 = h1[..., None], h2[..., None]
    zone = (T >= a1).astype(int) + (T >= a2).astype(int)
    u0 = (T - a1) / 40.0
    u1 = (T - a1) / (a2 - a1)
    u2 = (T - a2) / 50.0
    pat = np.where(zone == 0, 0.05 * np.sin(2 * np.pi * u0), np.where(zone == 1, 0.08 * np.sin(2 * np.pi * 5 * u1),
                                                                      0.06 * np.sin(2 * np.pi * u2)))
    off = np.where(zone == 1, 0.15, np.where(zone == 2, -0.05, 0.0))
    ln_vp = np.log(2000.0) + 0.0004 * T + off + pat
    vp = np.exp(ln_vp)
    rho = 0.31 * vp ** 0.25 * np.exp(0.3 * pat)
    vs = vp / np.where(zone == 1, 1.75, 2.1) * np.exp(-0.5 * pat)
    return {"t": t, "h1": h1, "h2": h2, "vp": vp, "vs": vs, "rho": rho, "ai": vp * rho, "zone": zone}


def seismic_of(ai, wavelet):
    r = np.zeros_like(ai)
    r[..., 1:] = (ai[..., 1:] - ai[..., :-1]) / (ai[..., 1:] + ai[..., :-1])
    flat = r.reshape(-1, r.shape[-1])
    out = np.array([np.convolve(tr, wavelet, "same") for tr in flat])
    return out.reshape(r.shape).astype(np.float32)


def inversion_project(tmp, name="inv", x0=500000.0, y0=6000000.0):
    """Seismic project with volume 'S' = convolutional synthetic of the earth (Ricker 25 Hz), horizons H1 / H2."""
    from .project import SeisProject
    from .synth import write_3d
    from .volume import import_3d
    from .welltie import ricker
    E = earth()
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    wav = ricker(25.0, DT)
    cube = seismic_of(E["ai"], wav)
    write_3d(os.path.join(d, "s.sgy"), cube, x0=x0, y0=y0)
    store = os.path.join(d, "s_seismic", "S")
    import_3d(os.path.join(d, "s.sgy"), store)
    sp = SeisProject()
    sp.path = os.path.join(d, f"{name}.seisproj.json")
    sp.volumes.append({"name": "S", "store": store, "segy": ""})
    for hn, g in (("H1", E["h1"]), ("H2", E["h2"])):
        h = sp.horizons.new(hn)
        h.data["3D:S"] = g.astype(np.float32)
    sp.save()
    E["wavelet"] = wav
    E["cube"] = cube
    return sp, E


def curves_from_earth(vol, E, ijs, t0=300.0, t1=1400.0):
    from .inversion.lfm import WellCurve
    out = []
    for k, (i, j) in enumerate(ijs):
        x, y = vol.xy(vol.inlines[i], vol.xlines[j])
        m = (E["t"] >= t0) & (E["t"] <= t1)
        vals = {p: np.where(m, E[p][i, j], np.nan) for p in ("ai", "vp", "vs", "rho")}
        out.append(WellCurve(f"W-{k}", float(x), float(y), E["t"], vals))
    return out


def run_lfm(check, tmp):
    from .inversion import common as CM
    from .inversion import lfm as L
    # filter response: exactly -3 dB at the cut, zero phase
    check(abs(CM.butterworth_response(8.0, 8.0) - 2 ** -0.5) < 1e-12 and CM.butterworth_response(0.0, 8.0) == 1.0,
          "Butterworth low-pass: 1 at 0 Hz, -3 dB at the cut frequency")
    t = np.arange(500) * DT
    lo = np.sin(2 * np.pi * 2.0 * t / 1000.0)
    hi = np.sin(2 * np.pi * 30.0 * t / 1000.0)
    y = CM.butter_filter(lo + hi, DT, 8.0, 4)
    mid = slice(100, 400)
    check(np.max(np.abs(y[mid] - lo[mid])) < 0.02, f"zero-phase low-pass keeps 2 Hz, removes 30 Hz "
          f"(max error {np.max(np.abs(y[mid] - lo[mid])):.3f}, no time shift)")
    sp, E = inversion_project(tmp)
    vol = sp.volume("S")
    f, A = CM.amplitude_spectrum(np.asarray(vol.ilxl[10, ::3], np.float32), DT)
    fc = CM.pick_lowcut(f, A)
    check(3.0 <= fc <= 20.0, f"low cut picked from the seismic spectrum: {fc:.1f} Hz (Ricker 25 Hz)")
    # lateral weights
    px, py = np.array([0.0, 1000.0, 0.0]), np.array([0.0, 0.0, 1000.0])
    for meth in ("idw", "kriging"):
        W = L.lateral_weights([0.0, 300.0], [0.0, 400.0], px, py, meth)
        check(np.allclose(W.sum(1), 1.0) and abs(W[0, 0] - 1.0) < 1e-6, f"{meth} weights sum to 1, a well takes its "
              "own location")
    curves = curves_from_earth(vol, E, WELL_IJ)
    vl = L.VolumeLFM(sp, "S", curves, "ai", ["H1", "H2"], ["proportional"] * 3, fc=None)
    i, j = WELL_IJ[1]
    got = vl.traces([i], [j])[0]
    m = np.isfinite(curves[1].values["ai"])
    check(np.allclose(got[m], E["ai"][i, j][m], rtol=1e-9), "LFM (unfiltered) honours the well log at the well trace")
    bi, bj = BLIND_IJ
    blind = vl.traces([bi], [bj])[0]
    true = E["ai"][bi, bj]
    zone = E["zone"][bi, bj]
    inside = (E["t"] > 350) & (E["t"] < 1350)
    err = np.abs(np.log(blind[inside] / true[inside]))
    flat = L.VolumeLFM(sp, "S", curves, "ai", [], fc=None).traces([bi], [bj])[0]
    errf = np.abs(np.log(flat[inside] / true[inside]))
    check(np.max(err) < 0.04 and np.max(errf) > 3 * np.max(err),
          f"blind trace follows the framework: max ln error {np.max(err):.3f} vs {np.max(errf):.3f} without horizons")
    hi_zone = np.log(blind[(zone == 1) & inside]).mean() - np.log(blind[(zone == 0) & inside]).mean()
    check(hi_zone > 0.1, f"no layer crossing: the high-AI zone stays between H1 and H2 at the blind trace "
          f"(+{hi_zone:.2f} in ln AI)")
    vf = L.VolumeLFM(sp, "S", curves, "ai", ["H1", "H2"], ["proportional"] * 3, fc=fc)
    lf = vf.traces([i], [j])[0]
    ref = np.exp(CM.butter_filter(np.log(got), DT, fc))
    check(np.allclose(lf, ref, rtol=1e-9) and np.std(np.diff(np.log(lf))) < 0.5 * np.std(np.diff(np.log(got))),
          f"LFM low-passed at {fc:.1f} Hz (smooth, = filtered well log at the well)")
    # kriging + trend-guided
    vk = L.VolumeLFM(sp, "S", curves, "ai", ["H1", "H2"], method="kriging", fc=None)
    check(np.allclose(vk.traces([i], [j])[0][m], E["ai"][i, j][m], rtol=1e-6), "kriging LFM honours the wells")
    lf_obj = L.LFM(curves, "ai")
    lf_obj.set_well_coords([(np.zeros(len(E["t"]), int), E["t"]) for _ in curves])
    trend_w = [np.full(len(E["t"]), 2.0) * np.nan_to_num(c.values["ai"], nan=1.0) / 2.0 for c in curves]
    trend_b = np.full((1, len(E["t"])), 5000.0)
    zz = np.zeros((1, len(E["t"])), int)
    vt = lf_obj.block([0.0], [0.0], E["t"], zz, E["t"][None, :], trend_b, trend_w)
    check(np.allclose(vt[0], 5000.0), "trend-guided LFM: wells equal to the trend -> the trend itself")
    # written volume
    name = vf.write()
    v2 = sp.volume(name)
    back = np.asarray(v2.ilxl[i, j], np.float32) * v2.inv
    check(name == "S [LFM AI]" and np.max(np.abs(back / lf - 1)) < 0.003,
          "LFM stored as a volume 'S [LFM AI]' (float16 store within 0.3 %)")
    try:
        vf.write(name="LFM X", cancel=lambda: True)
        ok = False
    except RuntimeError:
        ok = "S [LFM X]" not in [v["name"] for v in sp.volumes]
    check(ok, "LFM volume write cancellable (no partial volume)")
    return sp, E


def run_poststack(check, tmp, sp=None, E=None):
    from .inversion import lfm as L
    from .inversion import poststack as PS
    rng = np.random.default_rng(3)
    for m in (31, 32):
        w = rng.normal(size=m)
        x = rng.normal(size=(2, 200))
        op = PS.ConvOp(w, 200)
        y = rng.normal(size=(2, 200))
        same = np.array([np.convolve(r, w, "same") for r in x])
        check(np.allclose(op.forward(x), same) and abs(np.sum(op.forward(x) * y) - np.sum(x * op.adjoint(y))) < 1e-9,
              f"convolution operator == np.convolve 'same' and its adjoint passes the dot test (wavelet {m})")
    z = rng.normal(size=(2, 100))
    r = rng.normal(size=(2, 100))
    check(abs(np.sum(PS.D(z) * r) - np.sum(z * PS.DT(r))) < 1e-12, "reflectivity operator adjoint (dot test)")
    if sp is None:
        sp, E = inversion_project(tmp, "inv_ps")
    vol = sp.volume("S")
    curves = curves_from_earth(vol, E, WELL_IJ)
    fc = 8.0
    wav = E["wavelet"]
    i, j = WELL_IJ[0]
    s_w = np.asarray(vol.ilxl[i, j], np.float32).astype(float) * vol.inv
    w_scaled, k = PS.scale_wavelet(wav * 3.0, s_w, E["ai"][i, j])
    check(abs(k * 3.0 - 1.0) < 0.02, f"wavelet scaled to the seismic at a well (k = {k * 3:.3f}, true 1)")
    vl = L.VolumeLFM(sp, "S", curves, "ai", ["H1", "H2"], fc=fc)
    bi, bj = BLIND_IJ
    prior = vl.traces([bi], [bj])
    s = np.asarray(vol.ilxl[bi, bj], np.float32).astype(float)[None, :] * vol.inv
    zt = PS.invert_traces(s, np.log(prior), wav, lam=0.02)
    true = E["ai"][bi, bj]
    e_inv = PS.band_error(zt[0], true, DT, 60.0)
    e_lfm = PS.band_error(np.log(prior[0]), true, DT, 60.0)
    check(e_inv < 3.0 and e_inv < 0.5 * e_lfm, f"model-based inversion at a blind trace: {e_inv:.2f} % AI error in "
          f"the seismic band (LFM alone {e_lfm:.2f} %)")
    nr = PS.residual_nrms(s, zt, wav)[0]
    check(nr < 25.0, f"synthetic of the result matches the seismic (NRMS {nr:.1f} %)")
    noisy = s + rng.normal(0, 0.15 * s.std(), s.shape)
    zn = PS.invert_traces(noisy, np.log(prior), wav, lam=0.1)
    en = PS.band_error(zn[0], true, DT, 60.0)
    check(en < 5.0, f"with 15 % noise: {en:.2f} % AI error")
    # blocky earth: L1 (IRLS) beats Tikhonov
    n = 300
    blocky = np.exp(np.log(6000.0) + np.cumsum(np.where(rng.random(n) < 0.04, rng.normal(0, 0.12, n), 0.0)))
    sb = PS.synthetic_ln(np.log(blocky)[None, :], wav)
    pr = np.exp(np.convolve(np.log(blocky), np.ones(41) / 41, "same"))[None, :]
    pr[:, :20], pr[:, -20:] = blocky[:20].mean(), blocky[-20:].mean()
    z2 = PS.invert_traces(sb, np.log(pr), wav, lam=0.002)
    z1 = PS.invert_traces(sb, np.log(pr), wav, lam=0.002, mu=0.005, eps=1e-3)
    e2 = np.mean(np.abs(np.exp(z2[0]) / blocky - 1))
    e1 = np.mean(np.abs(np.exp(z1[0]) / blocky - 1))
    check(e1 < e2, f"blocky (L1 / TV by IRLS) inversion on a blocky earth: {100 * e1:.2f} % vs Tikhonov {100 * e2:.2f} %")
    # coloured inversion
    ci = PS.Coloured(sp, "S", curves)
    rel = ci.traces([bi], [bj])[0]
    bp = PS.butter_filter(PS.butter_filter(np.log(true), DT, 60.0), DT, 5.0, 4, "high")
    c = PS.corr(rel[50:-50], bp[50:-50])
    check(c > 0.8 and ci.alpha < 0.5, f"coloured inversion: alpha {ci.alpha:.2f}, correlation with the band-limited "
          f"true ln AI {c:.2f}")
    # volume + blind well test
    lfm_name = vl.write("LFM AI 8Hz")
    inv = PS.PostStack(sp, "S", lfm_name, wav, lam=0.02)
    name = inv.write()
    v2 = sp.volume(name)
    back = np.asarray(v2.ilxl[bi, bj], np.float32) * v2.inv
    check(name == "S [AI model-based]" and PS.band_error(np.log(back), true, DT, 60.0) < 3.5,
          f"model-based AI volume written ({PS.band_error(np.log(back), true, DT, 60.0):.2f} % at the blind trace)")
    lat = PS.PostStack(sp, "S", lfm_name, wav, lam=0.02, lateral=0.5)
    zl = np.log(lat.compute(10, 14))
    z0 = np.log(inv.compute(10, 14))
    check(np.std(np.diff(zl, axis=1)) <= np.std(np.diff(z0, axis=1)) + 1e-12, "lateral smoothing of the update")
    rows = PS.blind_well_test(sp, "S", curves, wav, ["H1", "H2"], fc=fc, lam=0.02)
    check(len(rows) == 4 and all(r["corr"] > 0.9 and r["error_pct"] < 4.0 for r in rows),
          "blind-well test: " + ", ".join(f"{r['well']} r={r['corr']:.2f} {r['error_pct']:.1f} %" for r in rows))
    return sp, E


ANGLES = [("Near", 8.0), ("Mid", 18.0), ("Far", 28.0)]


def add_angle_stacks(sp, E, tmp, x0=500000.0, y0=6000000.0):
    """Angle stacks (Aki-Richards, Ricker 25 Hz) of the earth as volumes Near / Mid / Far."""
    from .synth import write_3d
    from .tie import angle_synthetics
    from .volume import import_3d
    d = os.path.dirname(sp.path)
    nil, nxl, ns = E["ai"].shape
    cubes = np.zeros((len(ANGLES), nil, nxl, ns), np.float32)
    angs = [a for _, a in ANGLES]
    for i in range(nil):
        for j in range(nxl):
            g, _ = angle_synthetics(E["vp"][i, j], E["vs"][i, j], E["rho"][i, j], angs, E["wavelet"])
            cubes[:, i, j] = g.T
    for k, (name, _) in enumerate(ANGLES):
        write_3d(os.path.join(d, f"{name}.sgy"), cubes[k], x0=x0, y0=y0)
        store = os.path.join(d, "s_seismic", name)
        import_3d(os.path.join(d, f"{name}.sgy"), store)
        sp.volumes.append({"name": name, "store": store, "segy": ""})
    return cubes


def run_prestack(check, tmp, sp=None, E=None):
    from .inversion import lfm as L
    from .inversion import poststack as PS
    from .inversion import prestack as PR
    from .tie import rc_aki_richards
    if sp is None:
        sp, E = inversion_project(tmp, "inv_pre")
    # Fatti linearisation vs Aki-Richards on small contrasts
    vp, vs, rho = np.array([2500.0, 2650.0]), np.array([1200.0, 1330.0]), np.array([2.2, 2.27])
    ar = rc_aki_richards(vp, vs, rho, [0.0, 15.0, 30.0])[0]
    K = 0.5 * (vs[0] / vp[0] + vs[1] / vp[1])
    zp, zs = vp * rho, vs * rho
    fa = np.array([sum(c * 0.5 * np.diff(np.log(x))[0] for c, x in zip(PR.fatti_coeffs(a, K), (zp, zs, rho)))
                   for a in (0.0, 15.0, 30.0)])
    check(np.max(np.abs(fa - ar)) < 0.03 * np.max(np.abs(ar)) + 2e-4, f"Fatti coefficients == Aki-Richards for small "
          f"contrasts ({np.round(fa, 4)} vs {np.round(ar, 4)})")
    add_angle_stacks(sp, E, tmp)
    vol = sp.volume("S")
    curves = curves_from_earth(vol, E, WELL_IJ)
    bg = PR.fit_background(curves)
    check(0.5 < bg["k"] < 1.5 and 0.0 < bg["m"] < 0.5 and 0.4 < bg["K"] < 0.65,
          f"background trends from the wells: ln Zs = {bg['k']:.2f} ln Zp + c, ln rho = {bg['m']:.2f} ln Zp + c, "
          f"Vs/Vp {bg['K']:.2f}")
    pri = {}
    for prop in ("ai", "vs", "rho"):
        pri[prop] = L.VolumeLFM(sp, "S", curves, prop, ["H1", "H2"], fc=8.0).write(f"LFM {prop.upper()}")
    names = [n for n, _ in ANGLES]
    wavs = [E["wavelet"]] * 3
    ps = PR.PreStack(sp, ANGLES, wavs, pri, bg)
    bi, bj = BLIND_IJ
    lzp, lzs, lrho = ps.traces([bi], [bj])
    ezp = PS.band_error(lzp[0], E["ai"][bi, bj], DT, 60.0)
    ezs = PS.band_error(lzs[0], E["vs"][bi, bj] * E["rho"][bi, bj], DT, 60.0)
    erh = PS.band_error(lrho[0], E["rho"][bi, bj], DT, 60.0)
    vpvs = np.exp(lzp[0] - lzs[0])
    evv = PS.band_error(np.log(vpvs), E["vp"][bi, bj] / E["vs"][bi, bj], DT, 60.0)
    lf = {k: np.asarray(sp.volume(v).ilxl[bi, bj], np.float32) * sp.volume(v).inv for k, v in pri.items()}
    erh0 = PS.band_error(np.log(lf["rho"]), E["rho"][bi, bj], DT, 60.0)
    ezs0 = PS.band_error(np.log(lf["vs"] * lf["rho"]), E["vs"][bi, bj] * E["rho"][bi, bj], DT, 60.0)
    check(ezp < 3.0 and ezs < 5.0 and evv < 5.0 and ezs < ezs0, f"simultaneous inversion at a blind trace: Zp "
          f"{ezp:.2f} %, Zs {ezs:.2f} % (LFM {ezs0:.2f} %), Vp/Vs {evv:.2f} %")
    check(erh < max(3.0, erh0), f"density {erh:.2f} % (LFM alone {erh0:.2f} % - least constrained)")
    outs = ps.write()
    check(len(outs) == 6 and outs[0] == "Near [SI Zp]" and all(o in [v["name"] for v in sp.volumes] for o in outs),
          "Zp, Zs, density, Vp/Vs, lambda-rho, mu-rho volumes written in one pass")
    v = sp.volume("Near [SI VpVs]")
    back = np.asarray(v.ilxl[bi, bj], np.float32) * v.inv
    check(np.allclose(back, vpvs, rtol=3e-3), "Vp/Vs volume == Zp / Zs of the inversion")
    lr = sp.volume("Near [SI LambdaRho]")
    lrb = np.asarray(lr.ilxl[bi, bj], np.float32) * lr.inv
    zpb, zsb = np.exp(lzp[0]), np.exp(lzs[0])
    check(np.allclose(lrb, zpb ** 2 - 2 * zsb ** 2, rtol=5e-3, atol=1e-3 * np.abs(lrb).max()),
          "lambda-rho = Zp^2 - 2 Zs^2")
    return sp, E


def well_project_for(sp, E, tmp, x0, y0):
    """Well Correlation project with 4 wells at WELL_IJ: DT / DTS / RHOB / GR logs in depth from the earth at the
    well's trace, checkshot = the true time-depth; linked to the seismic project."""
    from .test_welltie import LAT, LON
    from .welltie import wc
    wc()
    from well_correlation.geo import tm_forward
    from well_correlation.selftest import _write_las
    d = os.path.join(os.path.dirname(sp.path), "wells_data")
    hdr = ["Well,UWI,Latitude,Longitude,Elevation_m,Elev_Ref,TD_m"]
    lon_m = 1.0 / (111320.0 * np.cos(np.radians(LAT)))
    t = E["t"]
    for k, (i, j) in enumerate(WELL_IJ):
        lat, lon = LAT + 25.0 * i / 110950.0, LON + 25.0 * j * lon_m
        x, y = tm_forward(lat, lon)
        ii, jj = int(round((y - y0) / 25.0)), int(round((x - x0) / 25.0))
        vp, vs, rho = E["vp"][ii, jj], E["vs"][ii, jj], E["rho"][ii, jj]
        z = np.r_[0.0, np.cumsum(0.5 * (vp[1:] + vp[:-1]) * DT / 2000.0)]
        md = np.arange(z[np.searchsorted(t, 250.0)], z[np.searchsorted(t, 1450.0)], 0.1524) + 25.0
        zz = md - 25.0
        tt = np.interp(zz, z, t)
        logs = {"DT": 304800.0 / np.interp(tt, t, vp), "DTS": 304800.0 / np.interp(tt, t, vs),
                "RHOB": np.interp(tt, t, rho), "GR": 60 + 40 * (np.interp(tt, t, E["zone"][ii, jj]) == 1) +
                np.random.default_rng(k).normal(0, 3, len(tt))}
        name = f"W-{k}"
        wd = os.path.join(d, "wells", name, "las")
        os.makedirs(wd)
        _write_las(os.path.join(wd, f"{name}.las"), name, md, logs,
                   {"DT": "US/F", "DTS": "US/F", "RHOB": "G/C3", "GR": "GAPI"})
        cs = os.path.join(d, "wells", name, "checkshot")
        os.makedirs(cs)
        with open(os.path.join(cs, "cs.csv"), "w") as fh:
            fh.write("TVD_SRD_m,TWT_ms\n" + "\n".join(f"{a:.2f},{b:.3f}" for a, b in zip(z[::10], t[::10])))
        hdr.append(f"{name},{k},{lat},{lon},25.0,KB,3000")
    with open(os.path.join(d, "wells_header.csv"), "w") as fh:
        fh.write("\n".join(hdr) + "\n")
    P = wc().project.Project
    p = P.from_folder(d)
    wpath = os.path.join(d, "inv.wcproj.json")
    p.save(wpath)
    sp.well_project = wpath
    sp.save()
    return wpath


def run_gui(check, tmp):
    """19d: Interpret > Seismic inversion window on a synthetic earth with LAS wells + checkshots."""
    from .gui.main_window import MainWindow
    from .segy import SegyFile
    from .test_welltie import LAT, LON
    from .welltie import wc
    wc()
    from well_correlation.geo import tm_forward
    x0, y0 = (round(v) for v in tm_forward(LAT, LON))
    sp, E = inversion_project(tmp, "invgui", x0, y0)
    well_project_for(sp, E, tmp, x0, y0)
    add_angle_stacks(sp, E, tmp, x0, y0)
    sp.save()
    w = MainWindow(sp.path)
    w.section.source = ("3D", "S")
    dlg = w.inversion()
    dlg.run_job_fn = None  # the Qt mock has no event loop: run jobs inline
    check(dlg.vol_name == "S", "19d: Seismic inversion window opens on the shown volume")
    curves = dlg.load_curves()
    vol = w.project.volume("S")
    errs = []
    for c in curves:
        il, xl = vol.ilxl_at(c.x, c.y)
        i, j = vol.il_index(il), vol.xl_index(xl)
        m = np.isfinite(c.values["ai"]) & (vol.times > 320) & (vol.times < 1400)
        errs.append(np.percentile(np.abs(c.values["ai"][m] / E["ai"][i, j][m] - 1), 95))
    check(len(curves) == 4 and max(errs) < 0.03, f"19d: 4 wells' AI in time from LAS + checkshot (P95 "
          f"{100 * max(errs):.1f} % from the earth; blocking at the zone steps)")
    fc = dlg.auto_fc()
    names = dlg.build_lfm(("ai", "vs", "rho"), ["H1", "H2"], fc=8.0)
    check(names == ["S [LFM AI]", "S [LFM VS]", "S [LFM RHO]"] and set(dlg.lfm_names) == {"ai", "vs", "rho"},
          f"19d: LFM volumes for AI, Vs, density (auto cut {fc:.1f} Hz offered)")
    dlg.set_wavelet("Ricker", 25.0, scale=True)
    check(abs(dlg.wavelet_k - 1.0) < 0.1, f"19d: Ricker wavelet scaled at the wells (k = {dlg.wavelet_k:.3f})")
    out = dlg.run_poststack("model-based", None, lam=0.02)
    check(out == ["S [AI model-based]"], "19d: model-based inversion volume")
    qc = dlg.qc_at_well("W-1")
    check(qc["error_pct"] < 4.0 and qc["nrms"] < 30.0, f"19d: QC at W-1: {qc['error_pct']:.1f} % AI error, residual "
          f"NRMS {qc['nrms']:.0f} %")
    rows = dlg.blind_test(0.02, ["H1", "H2"], 8.0)
    check(len(rows) == 4 and np.median([r["error_pct"] for r in rows]) < 4.0,
          "19d: blind-well test table (" + ", ".join(f"{r['error_pct']:.1f} %" for r in rows) + ")")
    out_b = dlg.run_poststack("blocky (L1)", None, lam=0.002, mu=0.005)
    out_c = dlg.run_poststack("coloured")
    check(out_b == ["S [AI blocky]"] and out_c == ["S [Relative AI (coloured)]"], "19d: blocky and coloured runs")
    try:
        dlg.run_prestack()
        ok = False
    except RuntimeError as e:
        ok = "angle stacks" in str(e)
    check(ok, "19d: pre-stack needs two or more stacks")
    for n, a in ANGLES:
        dlg.add_stack(n, a)
    out_p = dlg.run_prestack()
    check(len(out_p) == 6 and "Near [SI VpVs]" in out_p, "19d: simultaneous inversion -> 6 volumes")
    cp = dlg.crossplot("Near [SI Zp]", "Near [SI VpVs]", "log: GR")
    hi = cp["c"] > 80
    check(len(cp["x"]) > 100 and cp["c"] is not None and np.median(cp["y"][hi]) < np.median(cp["y"][~hi]),
          "19d: Zp vs Vp/Vs cross-plot coloured by GR (high-GR zone has the lower Vp/Vs here)")
    cp2 = dlg.crossplot("S [AI model-based]", "S [LFM AI]", "Near [SI Density]")
    check(len(cp2["x"]) > 1000 and len(cp2["c"]) == len(cp2["x"]), "19d: cross-plot coloured by a volume")
    path = dlg.export_segy("S [AI model-based]", os.path.join(tmp, "ai.sgy"))
    sf = SegyFile(path)
    tr = sf.traces(0, 1)[0]
    v = w.project.volume("S [AI model-based]")
    check(np.allclose(tr, np.asarray(v.ilxl[0, 0], np.float32) * v.inv, rtol=1e-6),
          "19d: SEG-Y export of the AI volume reads back")
    sec = w.section
    sec.source, sec.mode, sec.number, sec.domain = ("3D", "S [AI model-based]"), "Inline", int(vol.inlines[15]), "Time"
    sec.redraw()
    lo, hi_ = w.section.color_levels()
    check(lo > 1000 and hi_ > lo, f"19d: AI volume displayed with a positive colour range ({lo:.0f}-{hi_:.0f})")
    return True


def run(check, tmp):
    sp, E = run_lfm(check, tmp)
    run_poststack(check, tmp, sp, E)
    run_prestack(check, tmp, sp, E)
