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


def inversion_project(tmp, name="inv"):
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
    write_3d(os.path.join(d, "s.sgy"), cube)
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


def run(check, tmp):
    run_lfm(check, tmp)
