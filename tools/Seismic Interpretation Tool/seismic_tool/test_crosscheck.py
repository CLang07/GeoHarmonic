"""Tests for the petro vs maps cross-check (PLAN 21e)."""
import json
import os

import numpy as np


def _project(tmp, n=30, seed=3):
    from .project import SeisProject
    from .maps import MapGrid
    from . import maplayers as ML
    rng = np.random.default_rng(seed)
    d = os.path.join(tmp, "cc")
    os.makedirs(d)
    x = np.linspace(0, 10000, 60)
    X, Y = np.meshgrid(x, x)
    s1 = np.sin(X / 2500.0) + Y / 10000.0
    s2 = np.cos(Y / 2000.0)
    g1 = MapGrid(s1, "xy", "amp", "RMS amp", None, 0.0, 0.0, x[1], x[1])
    g2 = MapGrid(s2, "xy", "", "AI", None, 0.0, 0.0, x[1], x[1])
    px, py = rng.uniform(300, 9700, n), rng.uniform(300, 9700, n)
    from .propgrid import sample_grid
    y = 0.10 + 0.05 * sample_grid(g1, px, py) - 0.02 * sample_grid(g2, px, py) + rng.normal(0, 0.005, n)
    wells = [(f"W{i}", float(a), float(b)) for i, (a, b) in enumerate(zip(px, py))]
    wp = os.path.join(d, "w.wcproj.json")
    with open(wp, "w") as f:
        json.dump({"wells": [{"name": nm, "x": a, "y": b, "las": []} for nm, a, b in wells]}, f)
    os.makedirs(os.path.join(d, "w_petro"))
    with open(os.path.join(d, "w_petro", "summary.csv"), "w") as f:
        f.write("Well,Zone,PHIE_res\n" + "".join(f"W{i},R,{v}\n" for i, v in enumerate(y)))
    sp = SeisProject()
    sp.path = os.path.join(d, "cc.seisproj.json")
    sp.well_project = wp
    sp.save()
    st = ML.LayerStack.load(sp)
    l1, _ = st.add_grid(g1, "RMS amp", contours=False)
    l2, _ = st.add_grid(g2, "AI", contours=False)
    return sp, st, l1, l2, y


def run(check, tmp):
    from . import crosscheck as CC
    from .propgrid import sample_grid
    sp, st, l1, l2, y = _project(tmp)
    cc = CC.crosscheck(st, "PHIE_res", "R", [l1, l2])
    check(cc["n"] == 30 and np.allclose(cc["coef"], [0.10, 0.05, -0.02], atol=0.006) and cc["r"] > 0.95,
          f"21e: regression of the zone value on two maps recovered (coef {np.round(cc['coef'], 3)}, r {cc['r']:.3f})")
    check(abs(cc["sigma"] - 0.005) < 0.003 and cc["loo_rmse"] >= cc["sigma"] * 0.8, f"21e: residual sigma {cc['sigma']:.4f}, "
          f"LOO RMSE {cc['loo_rmse']:.4f}")
    m = CC.predicted_maps(st, cc)
    check(np.all(m["P90"].z < m["P50"].z) and np.all(m["P50"].z < m["P10"].z), "21e: P90 < P50 < P10 everywhere")
    lo = sample_grid(m["P90"], cc["px"], cc["py"])
    hi = sample_grid(m["P10"], cc["px"], cc["py"])
    cov = float(np.mean((y >= lo) & (y <= hi)))
    check(0.6 <= cov <= 0.97, f"21e: 80 % interval covers {100 * cov:.0f} % of the wells")
    corr = sample_grid(m["corrected"], cc["px"], cc["py"])
    p50 = sample_grid(m["P50"], cc["px"], cc["py"])
    check(np.mean(np.abs(corr - y)) < np.mean(np.abs(p50 - y)), "21e: residual-corrected map closer to the wells than P50")
    t = CC.to_attribute_table(st, "PHIE_res", "R", [l1, l2])
    check(len(t) == 30 and t.names == ["PHIE_res R", "RMS amp", "AI"] and t.xy.shape == (30, 2), "21e: attribute table for Analytics")
    c1 = CC.crosscheck(st, "PHIE_res", "R", [l1])
    check(len(c1["coef"]) == 2 and c1["r"] < cc["r"], "21e: one map explains less than two")


def run_gui(check, tmp):
    from .gui.layers_dock import LayersPanel
    from .gui.propgrid_dialog import CrossCheckDialog
    sp, st, l1, l2, y = _project(tmp, seed=4)
    panel = LayersPanel(None)
    panel.set_stack(st)
    got = []
    panel.analytics_hook = got.append
    dlg = CrossCheckDialog(st, panel)
    cc = dlg.compute("PHIE_res", "R", [l1, l2])
    ids = dlg.add_maps()
    t = dlg.send_to_analytics()
    check(cc["n"] == 30 and set(ids) == {"P50", "P90", "P10", "residual", "corrected"} and got and got[0] is t,
          "21e GUI: cross-check, predicted / P90 / P10 / residual / corrected layers, sent to Analytics")
