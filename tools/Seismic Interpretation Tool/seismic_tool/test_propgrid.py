"""Tests for zone-property gridding, cokriging / KED, cross-validation and the grid calculator (PLAN 21c)."""
import os

import numpy as np


def _secondary(n=80, L=10000.0):
    from .maps import MapGrid
    x = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, x)
    s = np.sin(X / 2500.0) + 0.6 * np.cos(Y / 1800.0) + X / L
    return MapGrid(s, "xy", "amp", "attr", None, 0.0, 0.0, x[1] - x[0], x[1] - x[0])


def run(check, tmp):
    from . import propgrid as PG
    from . import vgrid
    from .maps import MapGrid
    rng = np.random.default_rng(5)
    sec = _secondary()
    n = 25
    px, py = rng.uniform(500, 9500, n), rng.uniform(500, 9500, n)
    sw = PG.sample_grid(sec, px, py)
    check(np.allclose(sw, [float(sec.sample([a], [b])[0]) for a, b in zip(px, py)], equal_nan=True),
          "21c: vectorised grid sampling = MapGrid.sample")
    X, Y = sec.node_xy()
    # perfectly correlated primary -> cokriging = the rescaled secondary everywhere
    v = 2.0 + 3.0 * sw
    est = PG.estimate("cck", px, py, v, X, Y, sec)
    check(np.nanmax(np.abs(est - (2 + 3 * sec.z))) < 1e-6, "21c: collocated cokriging with rho = 1 reproduces the secondary trend")
    # uncorrelated (rho forced 0) -> simple kriging
    v2 = 10 + np.sin(px / 3000.0) * 2 + rng.normal(0, 0.2, n)
    e0 = PG.estimate("cck", px, py, v2, X, Y, sec, {"rho": 0.0})
    ek = vgrid.grid_values(X, Y, px, py, v2, "kriging")
    check(np.nanmax(np.abs(e0 - ek)) < 1e-6 * np.ptp(v2), "21c: collocated cokriging with rho = 0 = simple kriging")
    r, nn = PG.correlation(px, py, v, sec)
    check(abs(r - 1) < 1e-9 and nn == n, "21c: correlation with the secondary at the wells")
    # KED: trend from the secondary + a smooth residual; exact at wells; better LOO than kriging alone
    v3 = 5 + 2 * sw + 0.3 * np.sin(px / 1500.0 + py / 2000.0)
    ek3 = PG.estimate("ked", px, py, v3, px, py, sec)
    check(np.max(np.abs(ek3 - v3)) < 1e-6, "21c: kriging with external drift honours the wells")
    cv = PG.cross_validate(px, py, v3, ("idw", "tps", "kriging", "ked", "cck"), sec)
    check(cv["ked"]["rmse"] < cv["kriging"]["rmse"] and cv["cck"]["rmse"] < cv["kriging"]["rmse"] and all(c["n"] == n for c in cv.values()),
          "21c: leave-one-out RMSE: KED {:.3f}, CCK {:.3f} < kriging {:.3f}, IDW {:.3f}, TPS {:.3f}".format(
              cv["ked"]["rmse"], cv["cck"]["rmse"], cv["kriging"]["rmse"], cv["idw"]["rmse"], cv["tps"]["rmse"]))
    i = 3
    k = np.arange(n) != i
    manual = float(np.ravel(vgrid.grid_values(np.array([px[i]]), np.array([py[i]]), px[k], py[k], v3[k], "idw"))[0])
    check(abs(cv["idw"]["pred"][i] - manual) < 1e-9, "21c: LOO prediction = the method without that well")
    g = PG.grid_property(px, py, v3, "ked", sec, label="NetPay")
    check(g.z.shape == sec.z.shape and g.label == "NetPay" and np.isfinite(g.z).all(), "21c: property grid on the secondary geometry")
    rm = PG.residual_map(px, py, cv["kriging"]["resid"], g)
    check(np.isfinite(rm.z).all() and abs(float(PG.sample_grid(rm, px[:1], py[:1])[0]) - cv["kriging"]["resid"][0]) <
          0.2 * np.nanmax(np.abs(cv["kriging"]["resid"])), "21c: LOO residual map")
    gi = PG.grid_property(px, py, v3, "idw", cell=250.0)
    check(gi.kind == "xy" and abs(gi.dx - 250.0) < 1e-9 and np.nanmin(gi.z) >= v3.min() - 1e-9, "21c: IDW property grid on an XY box")
    # grid calculator
    A = sec
    B = MapGrid(sec.z * 2, "xy", "amp", "B", None, 0.0, 0.0, sec.dx, sec.dy)
    d = PG.calc("B - A", {"A": A, "B": B})
    check(np.allclose(d.z, sec.z), "21c: grid calculator B - A")
    C = MapGrid(np.ones((11, 11)) * 7.0, "xy", "", "C", None, -500.0, -500.0, 1200.0, 1200.0)
    e = PG.calc("clip(A * 10, 0, 5) + C", {"A": A, "C": C})
    check(np.allclose(e.z, np.clip(sec.z * 10, 0, 5) + 7.0), "21c: functions + a grid on another geometry (resampled)")
    try:
        PG.calc("__import__('os').getcwd()", {"A": A})
        safe = False
    except ValueError:
        safe = True
    check(safe, "21c: grid calculator rejects anything but arithmetic on grids")
    m = PG.mask_polygon(sec, [0, 5000, 5000, 0], [0, 0, 5000, 5000])
    X, Y = sec.node_xy()
    inside = (X < 5000) & (Y < 5000) & (X > 0) & (Y > 0)
    check(np.isfinite(m.z[inside]).all() and not np.isfinite(m.z[(X > 5100) | (Y > 5100)]).any(), "21c: mask by polygon")


def run_gui(check, tmp):
    from .gui.basemap import Basemap
    from .gui.layers_dock import LayersPanel
    from .gui.propgrid_dialog import PropGridDialog, GridCalcDialog
    from . import maplayers as ML
    from .test_maplayers import _setup
    from .maps import horizon_grid
    proj, truth, d = _setup(tmp, "propgrid_gui")
    bm = Basemap()
    bm.project = proj
    st = ML.LayerStack.load(proj)
    bm.set_stack(st)
    panel = LayersPanel(bm)
    panel.set_stack(st)
    gid, _ = st.add_grid(horizon_grid(proj, proj.horizons.get("Top"), "3D:Synth"), "Top time")
    dlg = PropGridDialog(st, panel)
    g = dlg.compute(prop="SoPhiH", zone="A", method="idw", secondary=None, cell=20.0)
    check(g is not None and np.isfinite(g.z).any() and dlg.points[0] == ["W-1", "W-2", "W-3"], "21c GUI: zone property gridded")
    g2 = dlg.compute(prop="SoPhiH", zone="A", method="ked", secondary=gid)
    check(g2.kind == "ilxl" and dlg.corr is not None, f"21c GUI: KED on the Top time layer (r = {dlg.corr:.2f})")
    rows = dlg.cross_validate(("idw", "kriging"))
    check(len(rows) == 2 and rows[0]["method"] == "idw" and rows[0]["n"] == 3, "21c GUI: cross-validation table")
    lid = dlg.add_layer()
    rid = dlg.add_residual_layer("idw")
    check(st.get(lid)["kind"] == "grid" and st.get(rid)["name"].startswith("LOO residual"), "21c GUI: property + residual layers")
    calc = GridCalcDialog(st, panel)
    letters = calc.letters()
    a = next(k for k, v in letters.items() if v == gid)
    out = calc.evaluate(f"{a} * 2", name="double")
    check(out is not None and np.allclose(st.grid(out).z, st.grid(gid).z * 2, equal_nan=True), "21c GUI: grid calculator layer")
    vol = proj.volume("Synth")
    xs, ys = vol.outline()
    mid = calc.mask(gid, list(np.asarray(xs[:4]) * 0 + [xs[0], xs[0] + 200, xs[0] + 200, xs[0]]),
                    [ys[0], ys[0], ys[0] + 200, ys[0] + 200])
    check(np.isfinite(st.grid(mid).z).sum() < np.isfinite(st.grid(gid).z).sum(), "21c GUI: mask by polygon layer")
