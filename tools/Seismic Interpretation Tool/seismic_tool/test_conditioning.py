"""PLAN 41 tests: seismic conditioning on synthetic volumes with known dipping reflectors + random noise + a fault +
an acquisition footprint + spikes."""
import os

import numpy as np

from . import conditioning as C

DT = 4.0


def ricker(f, dt_ms, n=41):
    t = (np.arange(n) - n // 2) * dt_ms / 1000.0
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def synth(ni=20, nj=36, ns=160, dip_i=0.25, dip_j=0.15, throw=6, fault_j=18, seed=0):
    """Clean reflectivity convolved with a 30 Hz Ricker: 6 dipping reflectors; the block j >= fault_j is shifted
    down by `throw` samples (a vertical fault)."""
    rng = np.random.default_rng(seed)
    w = ricker(30.0, DT)
    clean = np.zeros((ni, nj, ns))
    t0s = [25, 45, 62, 85, 105, 128]
    amps = rng.uniform(0.6, 1.0, len(t0s)) * rng.choice([-1, 1], len(t0s))
    for i in range(ni):
        for j in range(nj):
            r = np.zeros(ns)
            sh = throw if (fault_j is not None and j >= fault_j) else 0
            for t0, am in zip(t0s, amps):
                tt = t0 + dip_i * i + dip_j * j + sh
                k = int(np.floor(tt))
                f = tt - k
                if 0 <= k < ns - 1:
                    r[k] += am * (1 - f)
                    r[k + 1] += am * f
            clean[i, j] = np.convolve(r, w, mode="same")
    return clean


def run(check, tmp):
    os.makedirs(tmp, exist_ok=True)
    rng = np.random.default_rng(7)
    clean = synth()
    sig = float(np.sqrt(np.mean(clean ** 2)))
    noisy = clean + rng.normal(0, 0.5 * sig, clean.shape)
    # ---------------------------------------------------------------- 41b bandpass + spectra
    p = C.params({"type": "bandpass"})
    p["f"] = [5.0, 10.0, 40.0, 60.0]
    imp = np.zeros(256)
    imp[128] = 1.0
    out = C.bandpass(imp, DT, p)
    f, A = np.fft.rfftfreq(1024, DT / 1000.0), np.abs(np.fft.rfft(out, 1024))
    want = C.ormsby_response(f, *p["f"])
    check(np.max(np.abs(A - want)) < 0.02 and abs(np.argmax(out) - 128) == 0,
          f"41b: Ormsby bandpass matches the designed response (max error {np.max(np.abs(A - want)):.3f}), zero phase")
    pb = dict(p, kind="butterworth", f=[8.0, 50.0], order=4)
    ob = C.bandpass(imp, DT, pb)
    Ab = np.abs(np.fft.rfft(ob, 1024))
    wb = C.butterworth_band_response(f, 8.0, 50.0, 4)
    check(np.max(np.abs(Ab - wb)) < 0.02 and abs(wb[np.argmin(np.abs(f - 8))] - 2 ** -0.5) < 0.03,
          "41b: Butterworth band = designed response (-3 dB at the corners)")
    fq, sp = C.amplitude_spectrum(clean, DT)
    check(abs(fq[np.argmax(sp)] - 30.0) < 8.0, f"41b: spectrum of the synthetic peaks near 30 Hz ({fq[np.argmax(sp)]:.0f} Hz)")
    pw = dict(p, whiten=True)
    wh = C.bandpass(noisy[:2, :4], DT, pw)
    check(np.isfinite(wh).all(), "41b: spectral balancing (off by default) runs")
    # ---------------------------------------------------------------- 41c spikes
    spiky = noisy.copy()
    locs = [(3, 5, 40), (10, 20, 90), (15, 30, 130), (7, 1, 20), (18, 34, 150)]
    for i, j, k in locs:
        spiky[i, j, k] += 25 * sig * (1 if (i + j) % 2 else -1)
    rep = []
    fixed = C.remove_spikes(spiky, DT, C.params({"type": "spikes"}), rep)
    found = set(map(tuple, rep))
    check(set(locs) <= found and len(found) == len(locs),
          f"41c: all {len(locs)} spikes found, no false positives ({len(found)} replaced)")
    check(np.abs(fixed[3, 5, 40] - noisy[3, 5, 40]) < 3 * sig, "41c: spike replaced by the neighbouring traces' median")
    rep2 = []
    C.remove_spikes(clean + rng.normal(0, 0.3 * sig, clean.shape), DT, C.params({"type": "spikes"}), rep2)
    check(len(rep2) == 0, "41c: no false positives on clean data")
    # ---------------------------------------------------------------- 41d FX decon
    pf = C.params({"type": "fxdecon"})
    pf.update(window_traces=12, window_ms=240.0, filter_len=3, fmax=70.0)
    flat = synth(dip_i=0, dip_j=0, fault_j=None)
    nflat = flat + rng.normal(0, 0.5 * sig, flat.shape)
    sec_f = C.fx_section(nflat[5], DT, pf)
    sec_d = C.fx_section(noisy[5, :17], DT, pf)
    g1 = C.snr_db(flat[5], sec_f) - C.snr_db(flat[5], nflat[5])
    g2 = C.snr_db(clean[5, :17], sec_d) - C.snr_db(clean[5, :17], noisy[5, :17])
    check(g1 > 3 and g2 > 2, f"41d: FX decon improves SNR on flat (+{g1:.1f} dB) and dipping (+{g2:.1f} dB) events")
    m = dict(pf, mix=1.0)
    check(np.allclose(C.fx_section(nflat[5], DT, m), nflat[5]), "41d: mix-back 100 % returns the input")
    # ---------------------------------------------------------------- 41e dips + planarity
    pi, pj, pl = C.dips(clean, 1.0, 2.5)
    inner = (slice(4, -4), slice(4, 14), slice(20, 140))
    e_i = float(np.median(np.abs(pi[inner] - 0.25)))
    e_j = float(np.median(np.abs(pj[inner] - 0.15)))
    pl_f = float(np.median(pl[:, 16:20, 20:140]))
    pl_c = float(np.median(pl[inner]))
    check(e_i < 0.05 and e_j < 0.05, f"41e: structure-tensor dips recovered (IL {e_i:.3f}, XL {e_j:.3f} samples/trace)")
    check(pl_c > 0.8 and pl_f < pl_c, f"41e: planarity high on the reflectors ({pl_c:.2f}), lower at the fault ({pl_f:.2f})")
    az, mag = C.dip_attributes(np.array([1.0]), np.array([0.0]), DT)
    check(abs(az[0]) < 1e-9 and abs(mag[0] - DT / 25.0) < 1e-9, "41e: dip azimuth / magnitude")
    # ---------------------------------------------------------------- 41f SOF
    ps = C.params({"type": "sof"})
    f1 = C.sof(noisy, DT, ps)
    gain = C.snr_db(clean, f1) - C.snr_db(clean, noisy)
    q = C.qc_metrics(noisy, f1, DT)
    check(gain > 3, f"41f: structure-oriented filter improves SNR by {gain:.1f} dB (1 pass, protection on)")
    check(q["leakage"] < 0.35, f"41i/m: difference has no coherent reflectors (leakage {q['leakage']:.2f} < 0.35)")
    check(np.array_equal(C.sof(noisy, DT, dict(ps, strength=0.0)), noisy.astype(np.float32)),
          "41f: strength 0 % returns the input exactly")
    half = C.sof(noisy, DT, dict(ps, strength=0.5))
    check(np.allclose(half, noisy + 0.5 * (f1 - noisy), atol=1e-4), "41f: strength blend = original + s (filtered - original)")

    def throw_at(vol, j1, j2):
        """Throw (samples) between traces j1 | j2 from the cross-correlation lag (the fault is between them)."""
        a = vol[:, j1, 15:150].mean(axis=0)
        b = vol[:, j2, 15:150].mean(axis=0)
        lags = np.arange(-12, 13)
        cc = [np.dot(a[12:-12], np.roll(b, -L)[12:-12]) for L in lags]
        k = int(np.argmax(cc))
        if 0 < k < len(cc) - 1:
            den = cc[k - 1] - 2 * cc[k] + cc[k + 1]
            return lags[k] + (0.5 * (cc[k - 1] - cc[k + 1]) / den if den else 0.0)
        return float(lags[k])
    true_t = throw_at(clean, 17, 18)
    t_prot = throw_at(f1, 17, 18)
    unprot = C.sof(noisy, DT, dict(ps, kuwahara=False, guard=0.0, filter="mean"))
    err_p = np.sqrt(np.mean((f1[:, 15:21] - clean[:, 15:21]) ** 2))
    err_u = np.sqrt(np.mean((unprot[:, 15:21] - clean[:, 15:21]) ** 2))
    check(abs(t_prot - true_t) < 1.0 and err_p < err_u,
          f"41f: fault throw preserved with edge protection ({t_prot:.1f} vs true {true_t:.1f} samples); fault zone "
          f"error {err_p:.3f} < {err_u:.3f} without protection")
    skip = C.sof(noisy, DT, dict(ps, guard=0.99, guard_mode="skip", kuwahara=False))
    check(np.mean(skip == noisy.astype(np.float32)) > 0.3, "41f: planarity guard 'skip' leaves low-confidence samples")
    old = C.SOF_MAX_MB
    C.SOF_MAX_MB = 0.5  # force the chunked gather (a few inlines at a time) - same answer
    chunked = C.sof(noisy, DT, ps)
    C.SOF_MAX_MB = old
    check(np.allclose(chunked, f1, atol=1e-6), "41k: SOF gather in inline chunks (bounded memory) = one-shot result")
    two = C.sof(noisy, DT, dict(ps, passes=2))
    check(np.isfinite(two).all(), "41f: 2 passes (dips re-estimated between passes)")
    alpha = C.sof(noisy, DT, dict(ps, filter="alpha"))
    check(C.snr_db(clean, alpha) > C.snr_db(clean, noisy), "41f: alpha-trimmed mean also improves SNR")
    # aggressive run flags leakage / RMS warning: undirected mean over a big window
    agg = C.sof_pass(noisy.astype(np.float32), np.zeros_like(pi), np.zeros_like(pj), np.ones_like(pl),
                     dict(ps, half_il=4, half_xl=6, kuwahara=False, guard=0.0, filter="mean"))
    qa = C.qc_metrics(noisy, agg, DT)
    check(qa["leakage"] > q["leakage"] and qa["warnings"],
          f"41i: an aggressive (not dip-steered) filter is flagged ({qa['warnings'][0][:40] if qa['warnings'] else ''}...)")
    # ---------------------------------------------------------------- 41g footprint
    fp = np.zeros_like(clean)
    jj = np.arange(clean.shape[1])
    fp[:, :, :70] = (0.6 * sig * np.cos(2 * np.pi * jj / 4.0))[None, :, None]
    unf = synth(fault_j=None)  # footprint test on unfaulted reflectors (a fault is broadband on the same axis)
    with_fp = unf + fp
    pfp = C.params({"type": "footprint"})
    pfp.update(t1_ms=280.0)
    cleaned, peaks = C.remove_footprint(with_fp, DT, pfp)
    resid = cleaned[:, :, :70] - unf[:, :, :70]
    removed = 1 - abs(float(np.sum(resid * fp[:, :, :70]))) / float(np.sum(fp[:, :, :70] ** 2))
    check(peaks and removed > 0.95, f"41g: footprint peaks found automatically ({len(peaks)}), {100 * removed:.0f} % of "
          "the injected pattern removed")
    F0 = np.abs(np.fft.fft2(np.moveaxis(unf[:, :, :70], 2, 0))).mean(axis=0)
    F1 = np.abs(np.fft.fft2(np.moveaxis(cleaned[:, :, :70], 2, 0))).mean(axis=0)
    Fw = np.abs(np.fft.fft2(np.moveaxis(with_fp[:, :, :70], 2, 0))).mean(axis=0)
    mask, _ = C.footprint_mask(np.fft.fftshift(Fw), pfp, clean.shape[:2])
    far = np.fft.ifftshift(mask) > 0.99
    rel = float(np.abs(F1[far] - F0[far]).sum() / F0[far].sum())
    check(rel < 0.05, f"41g: rest of the kx-ky spectrum unchanged ({100 * rel:.1f} % change outside the notches)")
    check(np.allclose(cleaned[:, :, 75:], with_fp[:, :, 75:]), "41g: restricted to the shallow time gate")
    cl2, pk2 = C.remove_footprint(with_fp, DT, dict(pfp, directions="inline", notches=[(0.0, 0.25)]))
    check(pk2 == [(0.0, 0.25)] and np.sqrt(np.mean((cl2[:, :, :70] - unf[:, :, :70]) ** 2)) < 0.3 * np.sqrt(
        np.mean(fp[:, :, :70] ** 2)), "41g: user-drawn notch (ky, kx) works without auto detection")
    # ---------------------------------------------------------------- chain, bricks, presets, protect
    chain = C.preset("Interpretation")
    chain[2]["params"] = {"window_traces": 8, "window_ms": 200.0, "filter_len": 3}
    chain[3]["params"] = {"sigma_s": 1.5, "half_il": 1, "half_xl": 2}
    big = synth(ni=44, nj=24, ns=120, seed=3)
    vol = big + np.random.default_rng(5).normal(0, 0.5 * sig, big.shape)
    whole = C.apply_chain(vol, [s for s in chain if s["type"] != "footprint"], DT, include_global=False)
    h = C.chain_halo(chain)
    bricks = np.zeros_like(whole)
    for a in range(0, vol.shape[0], 6):
        b = min(vol.shape[0], a + 6)
        a0, b0 = max(0, a - h), min(vol.shape[0], b + h)
        r = C.apply_chain(vol[a0:b0], [s for s in chain if s["type"] != "footprint"], DT, include_global=False,
                          offset=(a0, 0), il_total=vol.shape[0])
        bricks[a:b] = r[a - a0:a - a0 + b - a]
    err = float(np.max(np.abs(bricks - whole)))
    check(err < 1e-4 * float(np.max(np.abs(whole))) and h < 44 // 2,
          f"41k/m: brick processing (6-inline bricks, halo {h}) = whole-volume processing (no seams, max diff {err:.2e})")

    class P:
        display = {}
    C.save_preset(P, "My chain", chain)
    check(C.preset("My chain", P) == C.validate_chain(chain) and "My chain" in C.preset_names(P)
          and C.preset_names()[:3] == ["Interpretation", "Light (for inversion)", "Fault-attribute input"],
          "41a/m: presets round-trip (built-ins + saved)")
    try:
        C.validate_chain([{"type": "magic"}])
        ok = False
    except ValueError:
        ok = True
    check(ok, "41a: unknown steps rejected")
    w = C.protect_weights(vol.shape, zones=[{"polygon": [(0, 0), (0, 10), (9, 10), (9, 0)], "t0": 200, "t1": 400,
                                             "weight": 0.0}], t_ms=np.arange(vol.shape[2]) * DT)
    w = C.protect_weights(noisy.shape, zones=[{"polygon": [(0, 0), (0, 10), (9, 10), (9, 0)], "t0": 200, "t1": 400,
                                               "weight": 0.0}], t_ms=np.arange(noisy.shape[2]) * DT)
    pr = C.sof(noisy, DT, ps, w)
    inside = w == 0
    check(inside.any() and np.array_equal(pr[inside], noisy.astype(np.float32)[inside])
          and not np.allclose(pr[~inside], noisy[~inside]),
          "41j: protect polygon + time gate - filtering skipped inside, applied outside")


def run_volume_test(check, tmp):
    """41k on a real (small) project volume: new volume, history, difference volume, cancel, SOF warning hook."""
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "cond")
    vol = proj.volume("Synth")
    chain = C.preset("Light (for inversion)")
    prog = []
    r = C.run_volume(proj, "Synth", chain, "Synth [light]", brick=5, progress=lambda f, m: prog.append(f),
                     diff=True)
    v2 = proj.volume(r["name"])
    a0 = np.asarray(vol.inline(int(vol.inlines[3])), float)
    a1 = np.asarray(v2.inline(int(vol.inlines[3])), float)
    hist = C.history(proj, r["name"])
    check(v2.nil == vol.nil and v2.ns == vol.ns and np.corrcoef(a0.ravel(), a1.ravel())[0, 1] > 0.8 and hist
          and hist[-1]["chain"][0]["type"] == "spikes" and prog and prog[-1] == 1.0,
          "41k: full-volume run -> new project volume with the chain in its history (original unchanged)")
    check(r["diff"] in [v["name"] for v in proj.volumes] and C.history_text(proj, r["name"]) == "spikes > bandpass",
          "41k/l: optional difference volume; volume pickers show the history ('spikes > bandpass')")
    check(not C.has_sof(proj, r["name"]), "41a: Light preset volume = no SOF (safe for inversion)")
    r2 = C.run_volume(proj, "Synth", C.preset("Fault-attribute input"), "Synth [faults]", brick=4)
    check(C.has_sof(proj, r2["name"]), "41a: SOF volumes are recognised (inversion warns)")
    try:
        C.run_volume(proj, "Synth", chain, "Synth [x]", cancel=lambda: True)
        cancelled = False
    except RuntimeError:
        cancelled = True
    check(cancelled and "Synth [x]" not in [v["name"] for v in proj.volumes], "41k: Cancel leaves nothing behind")


def run_gui(check, tmp):
    """Qt mock / real Qt offscreen: Seismic > Conditioning dialog - chain editing, presets, preview metrics, protect
    zone on the bus, full run (unthreaded) -> new volume in the seismic window; inversion SOF warning."""
    os.makedirs(tmp, exist_ok=True)
    from .gui.main_window import MainWindow
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "condgui")
    mw = MainWindow(proj.path)
    vol = mw.project.volume("Synth")
    sec = mw.section
    sec.source, sec.mode, sec.number = ("3D", "Synth"), "Inline", int(vol.inlines[8])
    sec.redraw()
    dlg = mw.conditioning()
    check([s["type"] for s in dlg.chain] == ["spikes", "bandpass", "fxdecon", "sof", "footprint"],
          "41a GUI: Seismic > Conditioning opens with the Interpretation preset")
    dlg.load_preset("Light (for inversion)")
    check([s["type"] for s in dlg.chain] == ["spikes", "bandpass"], "41a GUI: Light preset")
    dlg.add_step("sof")
    dlg.move_step(-1, row=2)
    check([s["type"] for s in dlg.chain] == ["spikes", "sof", "bandpass"], "41a GUI: add / reorder steps")
    pp = dlg.apply_params(row=1, text='{"half_il": 1, "half_xl": 2, "strength": 0.7, "bogus": 1}')
    check(pp == {"half_il": 1, "half_xl": 2, "strength": 0.7}, "41a GUI: step parameters edited (unknown keys dropped)")
    dlg.remove_step(row=2)
    dlg.save_preset_dialog("Test chain")
    check(C.preset("Test chain", mw.project)[1]["params"]["strength"] == 0.7, "41a GUI: chain saved as a named preset")
    q = dlg.preview("Inline", int(vol.inlines[8]))
    check(q is not None and 0 <= q["rms_removed_pct"] < 100 and "leakage" in q and dlg.last["original"].shape ==
          dlg.last["conditioned"].shape, f"41i GUI: preview original | conditioned | difference + metrics "
          f"(RMS removed {q['rms_removed_pct']:.1f} %, leakage {q['leakage']:.2f})")
    q2 = dlg.preview("Time slice", float(vol.times[60]))
    check(q2 is not None, "41i GUI: preview on a time slice")
    q3 = dlg.preview("Crossline", int(vol.xlines[5]))
    check(q3 is not None, "41i GUI: preview on a crossline")
    z = dlg.add_zone(vol.inlines[0], vol.inlines[5], vol.xlines[0], vol.xlines[5], 100, 400, 0.0)
    ev = mw.mapbus.last("data", "polygons")
    check(ev is not None and ev.get("style") == "protect" and len(ev["polygons"]) == 1
          and mw.project.display["conditioning_protect"], "41j GUI: protect zone stored + sent to the Live Map / 3D")
    r = dlg.run_full("Synth [test]", diff=True, threaded=False)
    names = [v["name"] for v in mw.project.volumes]
    ev = mw.mapbus.last("data", "volumes")
    check(r and "Synth [test]" in names and ev and "Synth [test]" in ev["ids"],
          "41k/l GUI: full run -> new project volume announced on the bus (DataChanged volumes)")
    check(mw.project.volume("Synth [test]").nil == vol.nil,
          "41l GUI: the conditioned volume opens like any project volume (seismic window, 3D, attributes, tracking)")
    from .gui.inversion_dialog import InversionDialog
    r2 = dlg.run_full("Synth [sof]", threaded=False)
    inv = InversionDialog(mw.project, mw)
    msg = inv.sof_warning("Synth [sof]")
    check("structure-oriented" in msg and inv.sof_warning("Synth") == "",
          "41a GUI: the inversion dialog warns when given an SOF volume")
