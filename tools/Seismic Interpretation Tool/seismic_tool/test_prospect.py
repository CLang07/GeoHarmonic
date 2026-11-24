"""Tests for PLAN 28: lognormal maths, GRV, volumetrics, Pg + DHI revision, portfolio, CSV."""
import csv
import math
import os

import numpy as np


def run(check, tmp):
    from . import prospect as P
    mu, s = P.lognormal_from_p90_p10(10.0, 100.0)
    st = P.lognormal_stats(10.0, 100.0)
    check(abs(math.exp(mu) - math.sqrt(1000)) < 1e-9 and abs(st["mean"] - math.exp(mu + s * s / 2)) < 1e-9,
          "28a: lognormal from P90 / P10 (P50 = geometric mean)")
    rng = np.random.default_rng(0)
    x = P.sample(("lognormal", 10.0, 100.0), 200000, rng)
    pv = P.pvalues(x)
    check(abs(pv["p90"] / 10 - 1) < 0.02 and abs(pv["p10"] / 100 - 1) < 0.02 and abs(pv["mean"] / st["mean"] - 1) < 0.02
          and abs(pv["swanson"] - (0.3 * pv["p90"] + 0.4 * pv["p50"] + 0.3 * pv["p10"])) < 1e-9,
          f"28a: sampled P90 / P10 / mean match the lognormal ({pv['p90']:.2f}, {pv['p10']:.1f}, {pv['mean']:.1f}); "
          "Swanson's mean")
    # GRV of an analytic cone: top = 1000 + 0.2 r, contact 1100 -> R = 500 m, V = pi R^2 H / 3
    d = 5.0
    ax = np.arange(-600, 601, d)
    X, Y = np.meshgrid(ax, ax)
    top = 1000 + 0.2 * np.hypot(X, Y)
    v = P.grv(top, 1100.0, d * d)
    exact = math.pi * 500 ** 2 * 100 / 3
    check(abs(v / exact - 1) < 0.01, f"28a: GRV of a cone {v:.4g} m3 vs analytic {exact:.4g}")
    lv, ar = P.area_depth(top, d * d, np.linspace(1000, 1150, 301))
    v_ad = P.grv_from_area_depth(lv, ar, 1100.0)
    v_t = P.grv(top, 1100.0, d * d, thickness=30.0)
    v_adt = P.grv_from_area_depth(lv, ar, 1100.0, thickness=30.0)
    check(abs(v_ad / v - 1) < 0.01 and abs(v_adt / v_t - 1) < 0.02 and v_t < v,
          "28a: area-depth integration agrees (thick and 30 m gross thickness cases)")
    vol = P.volumetrics(20000, v_t, 0.8, 0.2, 0.3, 1.25, 0.35, "oil")
    det = v_t * 0.8 * 0.2 * 0.7 / 1.25
    check(abs(vol["inplace_sm3"]["p50"] / det - 1) < 1e-9 and abs(vol["recoverable"]["p50"] / (det * 0.35 * P.M3_TO_BBL) - 1) < 1e-9,
          "28a: deterministic volumetrics (STOIIP = GRV NTG phi (1 - Sw) / Bo, bbl)")
    vg = P.volumetrics(20000, ("lognormal", 0.8 * v_t, 1.25 * v_t), ("uniform", 0.6, 0.9), ("normal", 0.2, 0.03, 0.05, 0.35),
                       ("triangular", 0.2, 0.3, 0.5), 0.005, ("uniform", 0.5, 0.8), "gas")
    r = vg["recoverable"]
    check(r["p90"] < r["p50"] < r["p10"] and r["p90"] < r["mean"] < r["p10"] and vg["unit"] == "scf",
          "28a: Monte-Carlo gas volumetrics P90 < P50 < P10")
    con, why = P.contact_depth(1000.0, 1100.0, fault_column=60.0)
    check(con == 1060.0 and why == "fault seal", "28a: contact limited by the fault-seal column")
    # ---- 28b
    f = {"source": 0.8, "reservoir": 0.7, "trap": 0.6, "seal": 0.7, "timing": 0.9}
    pg = P.chance_of_success(f)
    check(abs(pg - 0.8 * 0.7 * 0.6 * 0.7 * 0.9) < 1e-12, "28b: Pg = product of the factors")
    qs = np.linspace(0, 1, 21)
    rev = [P.dhi_revision(pg, q) for q in qs]
    check(all(0 < v < 1 for v in rev) and np.all(np.diff(rev) > 0) and abs(P.dhi_revision(pg, 0.3) - pg) < 1e-12 and
          abs(P.dhi_revision(pg, 0.9, quality=0.0) - pg) < 1e-12 and rev[-1] > pg > rev[0],
          f"28b: DHI revision bounded, monotonic, neutral at 0.3 (Pg {pg:.2f} -> {rev[0]:.2f} .. {rev[-1]:.2f})")
    pre = P.prefill_factors({"after_trap": 0.6, "verdict": "timing OK"}, (60.0, "fault seal"), {"index": 0.7, "reasons": ["x"]},
                            0.7, 0.8)
    check(set(pre) == {"source", "timing", "seal", "reservoir", "dhi_index"} and 0 < pre["timing"][0] < 1 and
          "fault seal" in pre["seal"][1], "28b: factors prefilled with their evidence")
    a = {"name": "A", "factors": {"source": 0.5}, "volume": 10.0}
    b = {"name": "B", "factors": {"source": 0.5}, "volume": 20.0}
    ind = P.portfolio([a, b], n=200000)
    dep = P.portfolio([a, b], shared={"source": ["A", "B"]}, n=200000)
    check(abs(ind["p_any"] - 0.75) < 0.01 and abs(dep["p_any"] - 0.5) < 0.01 and abs(dep["p_all"] - 0.5) < 0.01 and
          abs(ind["expected_volume"] - 15.0) < 0.3 and abs(dep["expected_volume"] - 15.0) < 0.3,
          "28b: dependent prospects (shared source) - same risked volume, P(at least one) 0.75 -> 0.50")
    row = P.prospect_row("A", vol, f, "oil", 0.7)
    path = P.export_csv(os.path.join(tmp, "portfolio.csv"), [row])
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    check(rows[0]["Prospect"] == "A" and float(rows[0]["Pg_DHI"]) > float(rows[0]["Pg"]) and
          abs(float(rows[0]["Risked_mean"]) - row["Pg_DHI"] * row["Mean"]) < 1e-6 * row["Mean"],
          "28b: portfolio CSV (Rose-style columns)")
