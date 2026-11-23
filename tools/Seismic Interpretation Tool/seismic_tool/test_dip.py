"""RGT step A checks: dip field from the structure tensor (called from selftest.run / run_gui)."""
import os

import numpy as np


def _planar_project(tmp, sub, a_ms=2.4, b_ms=-3.6, nil=36, nxl=44, ns=200):
    from .project import SeisProject
    from .synth import traces_from_surfaces, write_3d
    from .volume import import_3d
    d = os.path.join(tmp, sub)
    os.makedirs(d)
    rng = np.random.default_rng(0)
    i, j = np.meshgrid(np.arange(nil), np.arange(nxl), indexing="ij")
    t0s = np.sort(rng.uniform(80, 720, 45))
    surfs = [t0 + a_ms * (i - nil / 2) + b_ms * (j - nxl / 2) for t0 in t0s]
    cube = traces_from_surfaces(surfs, list(rng.uniform(-1, 1, len(t0s))), ns, 0.004, noise=0.02)
    write_3d(os.path.join(d, "p.sgy"), cube)
    store = os.path.join(d, "p_seismic", "P")
    import_3d(os.path.join(d, "p.sgy"), store)
    p = SeisProject()
    p.path = os.path.join(d, "p.seisproj.json")
    p.name = "p"
    p.volumes.append({"name": "P", "store": store, "segy": ""})
    p.save()
    return p, cube


def run(check, tmp):
    from .rgt import dip as D
    from .synth import traces_from_surfaces

    # kernels
    x = np.arange(50, dtype=np.float32)
    g = D.correlate1d(x[None] * 3.0, D.gaussian_kernel(1.5, deriv=True), 1)[0]
    check(np.allclose(g[10:-10], 3.0, atol=1e-4), "derivative-of-Gaussian kernel exact on a ramp")
    s = D.gsmooth(np.ones((5, 30), np.float32), (1.0, 2.0))
    check(np.allclose(s, 1.0, atol=1e-6), "Gaussian smoothing preserves a constant (reflect edges)")

    # 3D planar dips
    a_ms, b_ms = 2.4, -3.6
    p, cube = _planar_project(tmp, "dip", a_ms, b_ms)
    vol = p.volume("P")
    df = D.compute_dip_volume(vol, block=7)
    c = (slice(6, -6), slice(6, -6), slice(30, -30))
    il = np.asarray(df.arr["dip_il"], np.float32)[c]
    xl = np.asarray(df.arr["dip_xl"], np.float32)[c]
    e1 = np.percentile(np.abs(il - a_ms) / abs(a_ms), 90)
    e2 = np.percentile(np.abs(xl - b_ms) / abs(b_ms), 90)
    check(e1 < 0.05 and e2 < 0.05, f"3D dips recovered: inline {np.median(il):.3f} (true {a_ms}), crossline "
                                   f"{np.median(xl):.3f} (true {b_ms}) ms/trace, 90% within {max(e1, e2):.1%}")
    pl = np.asarray(df.arr["planarity"], np.float32)[c]
    check(np.median(pl) > 0.95, f"planarity of clean layering {np.median(pl):.3f}")
    whole = D.dip_3d_block(np.asarray(vol.ilxl, np.float32))[0] * 4.0
    check(np.max(np.abs(np.asarray(df.arr["dip_il"], np.float32) - whole)) < 0.02,
          "block-wise computation (with halo) equals whole-volume computation")
    sec = df.section("dip_xl", vol, "Inline", int(vol.inlines[5]))
    check(sec.shape == (vol.ns, vol.nxl) and df.section("planarity", vol, "Time slice", 400.0).shape == (vol.nil, vol.nxl),
          "dip field sections have the display layout")
    rng = np.random.default_rng(5)
    _, _, pn = D.dip_3d_block(rng.standard_normal((20, 20, 120)).astype(np.float32))
    check(np.median(pn) < 0.6, f"planarity of random noise is low ({np.median(pn):.2f})")

    # 2D folded reflectors: dip follows the local slope
    ntr = 160
    k = np.arange(ntr)
    surfs = [t0 + 30 * np.sin(k / 20.0) for t0 in np.arange(100, 760, 37.0)]
    line = traces_from_surfaces(surfs, list(np.random.default_rng(2).uniform(-1, 1, len(surfs))), 200, 0.004,
                                noise=0.01)
    dip, plan = D.dip_2d(line)
    true = 30 * np.cos(k / 20.0) / 20.0  # ms per trace
    got = np.median(dip[:, 40:-40] * 4.0, axis=1)
    err = np.abs(got - true)[8:-8]
    check(np.max(err) < 0.05 * 1.5 + 0.03, f"2D dip follows a fold (max err {np.max(err):.3f} ms/trace, max dip 1.5)")

    class _L:
        name = "fold"
        cdp = k
        ns = 200
        dt = 0.004

        def data(self):
            return line
    folder = os.path.join(tmp, "dipcache")
    d1, _ = D.line_dip(_L(), folder)
    d2, _ = D.line_dip(_L(), folder)
    check(os.path.exists(os.path.join(folder, "fold_dip.npz")) and np.allclose(d1, d2, atol=0.01),
          "2D dip cached next to the project")
    return p


def run_gui(check, tmp):
    from .gui.section_panel import SectionPanel
    from .rgt.dip import compute_dip_volume
    from .synth import make_project
    p, _ = _planar_project(tmp, "dipgui")
    vol = p.volume("P")
    panel = SectionPanel()
    panel.project = p
    panel.source = ("3D", "P")
    panel.mode = "Inline"
    panel.number = int(vol.inlines[10])
    panel.set_attr("Inline dip")
    check(panel._data is None, "dip display before computing: nothing drawn (message instead)")
    compute_dip_volume(vol)
    panel.set_attr("Crossline dip")
    check(panel._data is not None and abs(np.median(panel._data[30:-30, 6:-6]) + 3.6) < 0.2,
          "panel shows the crossline dip section")
    panel.set_attr("Planarity")
    check(panel.color_levels() == (0.0, 1.0), "planarity display levels 0..1")
    panel.mode = "Time slice"
    panel.number = 400
    panel.redraw()
    check(panel._data.shape == (vol.nil, vol.nxl), "planarity on a time slice")
    q, _ = make_project(os.path.join(tmp, "dipgui"), nil=10, nxl=12, ns=120, noise=0.0)
    panel.project = q
    panel.source = ("2D", "L-A")
    panel.mode = "2D"
    panel.set_attr("Inline dip")
    check(panel._data is not None and panel._data.shape == (120, 120), "2D line dip display")
    panel.set_attr("Seismic")
    return True
