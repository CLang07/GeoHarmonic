"""Tests for PLAN 30: depletion factor, RF / remaining resource / locations, RTA, frac complexity, economics."""
import math
import os

import numpy as np


def run(check, tmp):
    from . import depletion as D
    d = 10.0
    ax = np.arange(0, 2000 + d / 2, d)
    X, Y = np.meshgrid(ax, ax)
    so = np.ones(X.shape)  # SoPhiH 1 m everywhere
    par = {"name": "P", "cx": 600.0, "cy": 1000.0, "a": 200.0, "b": 600.0, "az": 90.0}
    same = dict(par, name="C")
    inpl = D.inplace_in(same, so, X, Y, d * d, 1.2)
    check(abs(inpl / (math.pi * 200 * 600 / 1.2) - 1) < 0.01, "30a: in-place inside the ellipse = SoPhiH x area / B")
    r_same = D.depletion_factor(same, [par], {"P": 50000.0}, so, X, Y, d * d, 1.2)
    check(abs(r_same["df"] - 50000.0 / inpl) < 1e-9 and r_same["pressure_proxy"] < 1,
          f"30a: child on top of its parent - depletion factor = parent cum / in-place ({r_same['df']:.3f})")
    dfs = [D.depletion_factor(dict(par, name="C", cx=600.0 + s), [par], {"P": 50000.0}, so, X, Y, d * d)["df"]
           for s in (0.0, 100.0, 200.0, 300.0, 450.0)]
    check(np.all(np.diff(dfs) < 0) and dfs[-1] == 0.0, f"30a: depletion factor falls as the child moves away "
          f"({', '.join(f'{v:.3f}' for v in dfs)})")
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 0.5, 40)
    pv = D.performance_vs_depletion(x, 100 - 50 * x + rng.normal(0, 1, 40))
    check(abs(pv["slope"] + 50) < 3 and pv["r"] < -0.9, "30a: child performance vs depletion factor (slope -50)")
    # ---- 30b
    rf = D.recovery_factors([par], {"P": 0.1 * D.inplace_in(par, so, X, Y, d * d)}, so, X, Y, d * d)
    check(abs(rf["P"] - 0.1) < 1e-12, "30b: RF = EUR / in-place in the drainage ellipse (hand case 0.10)")
    rem = D.remaining_map(so, X, Y, d * d, [par], rf)
    m = D.ellipse_mask(par, X, Y)
    check(np.allclose(rem[m], 0.9 * d * d / 1.2) and np.allclose(rem[~m], d * d / 1.2), "30b: remaining-resource map")
    ex = {"name": "E", "cx": 300.0, "cy": 1000.0, "a": 1000.0, "b": 300.0, "az": 0.0}  # N-S drainage at x 0-600
    n, slots = D.undrilled_locations(np.ones(X.shape, bool), X, Y, [ex], 200.0, azimuth=0.0, lateral=1800.0)
    check(n == 7 and abs(slots[0] - 700) < 1e-6, f"30b: undrilled locations at 200 m spacing next to existing drainage "
          f"({n}: {slots[:3]} ...)")
    # ---- 30c RTA on an analytic linear-flow well
    t = np.arange(15.0, 1500.0, 30.0)
    C = 2000.0
    t_elf = 300.0
    q = np.where(t < t_elf, C / np.sqrt(t), C / math.sqrt(t_elf) * np.exp(-(t - t_elf) / 200.0))
    r = D.rta_linear(t, q)
    check(abs(r["slope"] * C - 1) < 0.02 and abs(r["intercept"]) < 0.1 / C * 10 and 300 <= r["t_elf"] <= 450 and r["proxy"],
          f"30c: sqrt-time linear-flow slope 1/C recovered ({r['slope'] * C:.3f}), end of linear flow at "
          f"{r['t_elf']:.0f} d (true 300), rate-only proxy flagged")
    dp = 10.0 * np.ones(len(t))
    r2 = D.rta_linear(t, q, dp)
    check(abs(r2["slope"] * C / 10 - 1) < 0.02 and not r2["proxy"], "30c: pressure-normalised version")
    tmb = D.material_balance_time(np.full(5, 100.0), 30.0)
    check(np.allclose(tmb, [30, 60, 90, 120, 150]), "30c: material-balance time")
    check(abs(D.investigation_distance(100.0, 25.0) - 50.0) < 1e-9, "30c: investigation distance sqrt(eta t)")
    cx = D.frac_complexity_index("slickwater", 60, 2500, 15, 0.05, 0.7)
    pl = D.frac_complexity_index("gel", 15, 600, 60, 0.4, 0.2)
    check(cx["fci"] > 0.6 and pl["fci"] < 0.35 and cx["aspect"] > pl["aspect"] and cx["shape"].startswith("complex"),
          f"30c: frac complexity index slickwater / tight clusters {cx['fci']:.2f} vs gel / wide {pl['fci']:.2f} -> "
          "drainage aspect")
    # ---- 30d economics
    cf = D.cash_flow(np.full(12, 1000.0), price=50.0, royalty=0.2, opex_fixed=5000.0, opex_var=5.0, capex=200000.0)
    hand = sum((30000.0 - (200000.0 if i == 0 else 0.0)) / 1.1 ** ((i + 1) / 12) for i in range(12))
    check(np.allclose(cf[1:], 30000.0) and abs(D.npv(cf) - hand) < 1e-6, f"30d: NPV10 = hand-computed cash flow "
          f"({hand:,.0f})")
    tail = D.cash_flow(np.r_[np.full(6, 1000.0), np.full(6, 100.0)], 50.0, 0.2, 5000.0, 5.0)
    check(abs(D.npv(tail) - D.npv(tail[:6], economic_limit=False)) < 1e-6, "30d: economic limit cuts negative tail months")
    tc = D.type_curves({"a": [100.0] * 12, "b": [200.0] * 12, "c": [300.0] * 12})
    check(np.allclose(tc["all"]["p50"], 200) and np.allclose(tc["all"]["p90"], 120) and np.allclose(tc["all"]["p10"], 280),
          "30d: type curve P90 / P50 / P10 per month")
    inv = D.inventory({"a": np.full(24, 500.0)}, 5, np.full(36, 1500.0),
                      dict(price=50.0, royalty=0.2, opex_fixed=5000.0, opex_var=5.0, capex=300000.0))
    path = D.area_summary(os.path.join(tmp, "area.html"), "Test area", tc, inv, ["synthetic"])
    check(inv["pud_npv"] == 5 * inv["pv_per_location"] and inv["total"] > inv["pdp_npv"] and os.path.getsize(path) > 200,
          "30d: PDP + PUD inventory and one-page area summary")
