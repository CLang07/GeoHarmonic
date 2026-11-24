"""Tests for PLAN 25: AVO maths and classes, fizz vs commercial gas, angle-stack risk, DHI checklist."""
import numpy as np


def run(check, tmp):
    from . import avo as V
    sh = (2900.0, 1300.0, 2.45)
    # ---- 25a Zoeppritz / Aki-Richards / classes
    z = V.zoeppritz_pp(*sh, 3300, 1700, 2.35, [0.0, 10.0, 20.0, 30.0])
    z1, z2 = 2900 * 2.45, 3300 * 2.35
    ak = V.aki_richards(*sh, 3300, 1700, 2.35, np.array([0.0, 10.0, 20.0, 30.0]))
    check(abs(z[0] - (z2 - z1) / (z2 + z1)) < 1e-9 and np.max(np.abs(z - ak)) < 0.003,
          "25a: Zoeppritz exact at normal incidence, Aki-Richards within 0.003 to 30 deg")
    zc = V.zoeppritz_pp(2000, 1000, 2.2, 4000, 2300, 2.5, [10.0, 40.0, 60.0])
    check(np.all(np.isfinite(zc)) and abs(zc[-1]) <= 1.0 + 1e-9, "25a: Zoeppritz beyond the critical angle stays finite")
    cases = {"I": (3800, 1900, 2.55), "IIp": (3200, 1800, 2.35), "II": (3050, 1700, 2.33), "III": (2500, 1500, 2.2)}
    got = {k: V.interface_avo(sh, low)["class"] for k, low in cases.items()}
    got["IV"] = V.interface_avo((3000, 1800, 2.4), (2600, 1550, 2.25))["class"]
    check(all(k == v for k, v in got.items()), f"25a: AVO classes from known elastic contrasts {got}")
    rng = np.random.default_rng(0)
    ang = np.arange(0, 36, 5.0)
    A0, B0 = 0.05, -0.2
    g = A0 + B0 * np.sin(np.radians(ang))[:, None] ** 2 + rng.normal(0, 1e-4, (len(ang), 3))
    A, B = V.ig_from_gather(g, ang)
    check(np.allclose(A, A0, atol=1e-3) and np.allclose(B, B0, atol=3e-3), "25a: intercept / gradient from a gather")
    Aw = rng.normal(0, 0.05, 2000)
    Bw = -0.8 * Aw + rng.normal(0, 0.01, 2000)  # wet background trend
    Ah = np.full(20, -0.1)
    Bh = -0.8 * Ah - 0.12  # hydrocarbon: below the trend
    tr = V.background_trend(np.r_[Aw, Ah], np.r_[Bw, Bh])
    zan = V.ig_anomaly(Ah, Bh, tr)
    check(abs(tr[0] + 0.8) < 0.05 and np.all(zan < -3) and np.mean(np.abs(V.ig_anomaly(Aw, Bw, tr)) > 3) < 0.03,
          f"25a: robust background trend (slope {tr[0]:.2f}) and anomalies below it (z {np.median(zan):.1f})")
    sc = V.scenario_avo(sh, 0.25, 0.1)
    ff = {k: float(V.fluid_factor(r["A"], r["B"])) for k, r in sc.items()}
    check(sc["gas"]["A"] < sc["brine"]["A"] and sc["gas"]["B"] < sc["brine"]["B"] and ff["gas"] < ff["brine"],
          "25a: Gassmann scenarios - gas brighter and more negative gradient / fluid factor than brine")
    # ---- 25b fizz
    check(abs(sc["fizz"]["A"] - sc["gas"]["A"]) < abs(sc["fizz"]["A"] - sc["brine"]["A"]) * 1.6,
          "25b: fizz AVO looks like commercial gas")
    fz_g = V.fizz_assessment(-0.075, 0.25, rho_err=0.02)
    fz_f = V.fizz_assessment(-0.005, 0.25, rho_err=0.02)
    fz_u = V.fizz_assessment(-0.04, 0.25, rho_err=0.08)
    check(fz_g["p_fizz"] < 0.05 and fz_f["p_fizz"] > 0.9 and fz_g["separation_sd"] > 3,
          f"25b: density separates fizz from commercial gas (P(fizz) {fz_g['p_fizz']:.2f} vs {fz_f['p_fizz']:.2f})")
    check(abs(fz_u["score"] - 0.3) < 0.15 and "too uncertain" in " ".join(fz_u["reasons"]),
          "25b: with a poor density estimate the fizz risk stays at the prior and says why")
    # low-frequency shadow under a 'gas' zone
    ns, n, dt = 300, 40, 2.0
    sec = rng.normal(size=(ns, n))
    t_res = np.full(n, 250.0)
    from .inversion.common import butter_filter
    lp = butter_filter(sec.T, dt, 15.0).T
    gas = np.arange(n) >= 20
    sec[135:, gas] = lp[135:, gas] * 3  # below the reservoir (270 ms+): high frequencies attenuated
    ratio, drop = V.lf_shadow(sec, dt, t_res, below_ms=(30.0, 90.0), above_ms=(120.0, 20.0))
    check(np.median(ratio[gas]) > 2 * np.median(ratio[~gas]) and np.median(drop[gas]) > 10 > abs(np.median(drop[~gas])),
          "25b: low-frequency shadow and centroid-frequency drop below the gas zone")
    # ---- 25c angle-stack vs gather risk
    r35 = V.stack_vs_gather_risk(35.0, n_mc=3000)
    r50 = V.stack_vs_gather_risk(50.0)
    order = [r35[k]["rho"] for k in ("gather", "5 stacks", "3 stacks", "2 stacks")]
    check(all(a <= b + 1e-12 for a, b in zip(order, order[1:])), f"25c: density error gather <= 5 <= 3 <= 2 stacks "
          f"({', '.join(f'{v:.3f}' for v in order)})")
    check(r35["gather"]["rho"] > r35["gather"]["zp"] * 3 and r50["2 stacks"]["rho"] < r35["2 stacks"]["rho"] and
          r50["gather"]["rho"] < 0.7 * r35["gather"]["rho"], "25c: density is the casualty and needs far angles")
    mc_ok = all(abs(r35[k]["mc"][j] / r35[k][q] - 1) < 0.1 for k in ("gather", "2 stacks") for j, q in
                enumerate(("zp", "zs", "rho")))
    check(mc_ok, "25c: analytic errors confirmed by Monte-Carlo (10 %)")
    # ---- 25d DHI
    ny, nx = 60, 80
    Y, X = np.mgrid[0:ny, 0:nx]
    depth = 2000 + 0.04 * ((X - 40) ** 2 + (Y - 30) ** 2)  # dome, crest 2000
    spill = 2030.0
    fits = depth <= spill
    amp = np.where(fits, 3.0, 1.0) + rng.normal(0, 0.2, depth.shape)
    s_fit, lev = V.conformance_score(np.abs(amp) > 2, depth)
    off = ((X - 60) ** 2 + (Y - 15) ** 2) < 150  # anomaly crossing the contours
    amp_off = np.where(off, 3.0, 1.0) + rng.normal(0, 0.2, depth.shape)
    s_off, _ = V.conformance_score(np.abs(amp_off) > 2, depth)
    check(s_fit > 0.85 and abs(lev - spill) < 5 and s_off < 0.5, f"25d: conformance - anomaly that fits the closure "
          f"{s_fit:.2f} (contour {lev:.0f} m) vs one that cuts the contours {s_off:.2f}")
    bg = depth > 2060
    a_fit = V.assess_prospect(amp, depth, fits, bg, manual={"flat_spot": 0.8, "quality": 0.7})
    a_off = V.assess_prospect(amp_off, depth, fits, bg, manual={"flat_spot": 0.0, "quality": 0.7})
    check(a_fit["index"] > a_off["index"] + 0.2 and a_fit["reasons"] and abs(sum(a_fit["weights"].values()) - 0.65) < 1e-9,
          f"25d: DHI index {a_fit['index']:.2f} (conformant, flat spot) vs {a_off['index']:.2f}; weights renormalised over "
          "the scored checks")
    d2 = V.dhi_index({"amplitude": 1.0, "conformance": 0.0}, weights={"amplitude": 0.9, "conformance": 0.1})
    check(abs(d2["index"] - 0.9) < 1e-9, "25d: user-editable weights")
    sc_st = V.stack_consistency_score([amp, amp * 1.2 + 0.1, amp + rng.normal(0, 0.1, amp.shape)])
    check(sc_st > 0.9, "25d: consistency across stacks")
