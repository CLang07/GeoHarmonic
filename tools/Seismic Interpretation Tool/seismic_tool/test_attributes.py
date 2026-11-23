"""Seismic attribute checks."""
import os

import numpy as np


def run(check, tmp):
    from . import attributes as A
    dt = 4.0
    t = np.arange(400) * dt / 1000.0
    # envelope / phase / frequency of a modulated 25 Hz cosine
    env_true = 1 + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    x = env_true * np.cos(2 * np.pi * 25 * t)
    e = A.envelope(x)
    check(np.max(np.abs(e[40:-40] - env_true[40:-40])) < 0.05, "envelope of a modulated cosine")
    f = A.inst_freq(x, dt)
    check(np.max(np.abs(f[40:-40] - 25.0)) < 1.0, f"instantaneous frequency {np.median(f[40:-40]):.2f} Hz (25 Hz)")
    ph = A.inst_phase(np.cos(2 * np.pi * 25 * t + 0.7))
    check(abs(((ph[100] - np.degrees(2 * np.pi * 25 * t[100] + 0.7)) + 180) % 360 - 180) < 2, "instantaneous phase")
    r = A.rms(np.ones((2, 100)) * 3.0, dt, 40.0)
    check(np.allclose(r, 3.0), "RMS of a constant")
    # spectral decomposition: 20 Hz in the first half, 50 Hz in the second
    y = np.where(t < 0.8, np.cos(2 * np.pi * 20 * t), np.cos(2 * np.pi * 50 * t))
    s20, s50 = A.spectral(y, dt, 20.0), A.spectral(y, dt, 50.0)
    check(s20[100] > 3 * s50[100] and s50[300] > 3 * s20[300], "spectral decomposition separates 20 and 50 Hz")
    # coherence: planar dipping layers vs random noise; dip steering matters on steep dips
    from .synth import traces_from_surfaces
    rng = np.random.default_rng(1)
    i, j = np.meshgrid(np.arange(12), np.arange(12), indexing="ij")
    surfs = [t0 + 6.0 * j for t0 in np.sort(rng.uniform(100, 1300, 50))]  # 6 ms per trace dip
    blk = traces_from_surfaces(surfs, list(rng.uniform(-1, 1, 50)), 400, 0.004, f_hz=35.0)
    flat = A.semblance(blk, dt, 24.0)
    steer = A.semblance(blk, dt, 24.0, dip_i=np.zeros(blk.shape), dip_j=np.full(blk.shape, 6.0 / dt))
    noise = A.semblance(rng.standard_normal(blk.shape), dt, 24.0)
    c = (slice(None), slice(None), slice(60, 340))
    check(np.median(steer[c]) > 0.95 and np.median(steer[c]) > np.median(flat[c]) + 0.1 and np.median(noise[c]) < 0.3,
          f"coherence: steered {np.median(steer[c]):.2f} > unsteered {np.median(flat[c]):.2f} on 6 ms/trace dip; "
          f"noise {np.median(noise[c]):.2f}")
    # coherence low along a fault (random reflectivity, 28 ms throw)
    from .test_rgt import geology_project
    p, G = geology_project(tmp, "attr", fault=True, lines=True)
    p.volumes[0]["name"] = "F"
    p._vol_cache.clear()
    fxl = G["fxl"]
    vol = p.volume("F")
    co = A.section_attribute(vol, "Coherence", "Inline", int(vol.inlines[20]))
    tt = vol.times
    k = int(np.argmin(np.abs(tt - 400)))
    jf = int(round(fxl(20, 400.0)))
    check(co[k, jf - 1:jf + 2].min() < 0.6 and np.median(co[k, :jf - 4]) > 0.9,
          f"coherence drops at the fault ({co[k, jf - 1:jf + 2].min():.2f} vs {np.median(co[k, :jf - 4]):.2f})")
    # section on the fly == the same slice of a computed volume
    name = A.compute_volume(p, "F", "Envelope", block=7)
    ev = p.volume(name)
    sec = A.section_attribute(vol, "Envelope", "Inline", int(vol.inlines[9]))
    check(name == "F [Envelope]" and np.allclose(ev.inline(int(vol.inlines[9])).T, sec, rtol=0.01, atol=0.01 * sec.max()),
          "attribute volume (blocks) equals the on-the-fly section")
    name_c = A.compute_volume(p, "F", "Coherence", block=7)
    cv = p.volume(name_c)
    sc = A.section_attribute(vol, "Coherence", "Crossline", int(vol.xlines[30]))
    vv = cv.xline(int(vol.xlines[30])).T
    check(np.corrcoef(vv.ravel(), sc.ravel())[0, 1] > 0.99, "coherence volume (with halo between blocks) equals the section")
    ts = A.section_attribute(vol, "Envelope", "Time slice", 400.0)
    check(ts.shape == (vol.nil, vol.nxl) and np.allclose(ts, ev.time_slice(400.0), rtol=0.02, atol=0.02 * ts.max()),
          "attribute on a time slice")
    p.save()
    from .project import SeisProject
    check(any(v["name"] == "F [Envelope]" for v in SeisProject.open(p.path).volumes), "attribute volume saved in the project")
    # 2D line attributes
    ln = p.line("A")
    lc = A.line_attribute(ln, "Coherence")
    check(lc.shape == ln.data().shape and lc[:, k].min() < 0.6, "2D line coherence (fault visible)")
    # horizon curvature: paraboloid z = (x^2 + y^2) / (2R) -> mean curvature 1/R at the apex
    from .maps import MapGrid
    R = 2000.0
    xs = np.arange(-1000, 1001, 25.0)
    X, Y = np.meshgrid(xs, xs)
    tms = (X ** 2 + Y ** 2) / (2 * R) * 2000.0 / 2000.0  # v = 2000 m/s -> z (m) = t (ms)
    g = MapGrid(tms, "xy", "ms", "bowl", None, xs[0], xs[0], 25.0, 25.0)
    km = A.horizon_curvature(g, "mean", 2000.0)
    kg = A.horizon_curvature(g, "gaussian", 2000.0)
    c0 = len(xs) // 2
    check(abs(km.z[c0, c0] - 1000.0 / R) < 0.01 and abs(kg.z[c0, c0] - 1000.0 / R ** 2) < 1e-3,
          f"horizon curvature of a paraboloid (mean {km.z[c0, c0]:.3f} 1/km, expected {1000 / R:.3f})")
    dipg = A.horizon_curvature(MapGrid(np.add.outer(np.zeros(20), np.arange(20) * 25.0 * 0.1), "xy", "ms", "",
                                       None, 0, 0, 25.0, 25.0), "dip", 2000.0)
    check(abs(np.median(dipg.z) - np.degrees(np.arctan(0.1))) < 0.01, "horizon dip")
    return True


def run_gui(check, tmp):
    from .gui.section_panel import SectionPanel
    from .gui.map_dialog import MapDialog
    from .synth import make_project
    d = os.path.join(tmp, "attr_gui")
    os.makedirs(d)
    proj, truth = make_project(d, nil=12, nxl=15, ns=150, noise=0.01)
    panel = SectionPanel()
    panel.project = proj
    panel.source = ("3D", "Synth")
    vol = proj.volume("Synth")
    panel.mode = "Inline"
    panel.number = int(vol.inlines[4])
    for a in ("Envelope", "Coherence", "Spectral 30 Hz", "Inst. frequency"):
        panel.set_attr(a)
        check(panel._data is not None and panel._data.shape == (vol.ns, vol.nxl), f"panel shows {a} on the fly")
    panel.mode = "Time slice"
    panel.number = 300
    panel.set_attr("Coherence (dip-steered)")
    check(panel._data is not None and panel._data.shape == (vol.nil, vol.nxl), "dip-steered coherence on a time slice")
    panel.source = ("2D", "L-A")
    panel.mode = "2D"
    panel.set_attr("Envelope")
    check(panel._data is not None and panel.color_levels()[0] >= 0, "2D line attribute")
    panel.set_attr("Seismic")
    h = proj.horizons.new("Top")
    h.data["3D:Synth"] = truth["3D:Synth"].astype(np.float32)
    dlg = MapDialog(proj)
    g = dlg.compute(key="3D:Synth", product="Curvature / dip", h1="Top", curv="mean", velocity=2500.0)
    check(g.unit == "1/km" and np.isfinite(g.z).mean() > 0.9, "map dialog: horizon curvature")
    return True
