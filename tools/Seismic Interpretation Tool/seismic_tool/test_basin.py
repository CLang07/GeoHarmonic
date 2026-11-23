"""Basin model checks (PLAN 18): chronostratigraphy + age Wheeler (18a), burial (18b), thermal / maturity (18c),
maps + event chart (18d), GUI logic (18e)."""
import os
import types

import numpy as np


def wedge_section(ntr=60, ns=300, dt=2.0, r_top=200.0, r_base=280.0):
    """RGT section (ntr, ns) with a wedge between RGT r_top and r_base thinning to zero at 2/3 of the line."""
    times = np.arange(ns) * dt
    r = np.linspace(0, 2 * ns * dt, 4000)
    tau = np.zeros((ntr, ns))
    f = np.clip(1 - np.arange(ntr) / (ntr * 2 / 3), 0, 1)
    for j in range(ntr):
        fj = max(f[j], 1e-4)
        t = np.where(r < r_top, r, np.where(r <= r_base, r_top + (r - r_top) * fj, r_top + (r_base - r_top) * fj +
                                            (r - r_base)))
        tau[j] = np.interp(times, t, r)
    return tau, times, f


def run_chrono(check, tmp):
    from .basin import chrono as C
    from .rgt.stack import wheeler

    T = C.AgeTable.taranaki()
    check(not T.problems() and T.entries[0]["name"] == "Seabed" and len(T.zones()) == len(T.entries) - 1,
          f"Taranaki age template: {len(T.entries)} dated surfaces, no problems")
    bad = C.AgeTable([{"name": "A", "age": 10, "gap": 5}, {"name": "B", "age": 12}])
    check(any("younger" in p for p in bad.problems()), "age table: a surface inside the hiatus above is reported")
    p = types.SimpleNamespace(display={})
    T.set("Tikorangi Limestone", gap=3.0, horizon="Top Tik")
    T.save(p)
    T2 = C.AgeTable.load(p)
    check(T2.age_below("Top Tik") == 25.0 and T2.get("tikorangi limestone")["horizon"] == "Top Tik",
          "age table saved in project.display['basin'] and found by horizon name")

    # ages for RGT levels: monotonic, honour the anchors, jump at an unconformity
    anchors = [(100.0, 5.0, 5.0, "a"), (200.0, 22.0, 30.0, "b"), (300.0, 40.0, 40.0, "c")]
    M = C.AgeModel(anchors)
    lev = np.linspace(0, 400, 401)
    la = C.level_ages(lev, M)
    check(np.all(np.diff(la) >= 0) and la[0] >= 0 and abs(M(150.0) - 13.5) < 1e-9 and abs(M(250.0) - 35.0) < 1e-9
          and abs(M(199.999) - 22.0) < 1e-3 and abs(M(200.001) - 30.0) < 1e-3,
          "level ages: monotonic, linear between dated horizons, jump 22 -> 30 Ma at the unconformity")
    check(abs(M(350.0) - 45.0) < 1e-9 and M(0.0) >= 0, "level ages extrapolate with the nearest interval rate")
    try:
        C.anchors_from(T2, {"Seabed": 50.0, "Tikorangi Limestone": 40.0})
        ok = False
    except ValueError:
        ok = True
    check(ok, "horizons out of age order in RGT are refused")

    # synthetic pinch-out: wedge from RGT 200 (22 Ma) to 280 (30 Ma) pinches out at 2/3 of the line
    tau, times, f = wedge_section()
    ntr = tau.shape[0]
    seis = np.sin(tau / 6.0).astype(np.float32)
    levels = np.arange(0, 600, 2.0)
    amp, hiatus = wheeler(seis, tau, levels, times)
    tab = C.AgeTable([{"name": "Top wedge", "age": 22.0}, {"name": "Base wedge", "age": 30.0},
                      {"name": "Top", "age": 2.0}])
    ht = {"Top wedge": np.array([np.interp(200.0, tau[j], times) for j in range(ntr)]),
          "Base wedge": np.array([np.interp(280.0, tau[j], times) for j in range(ntr)]),
          "Top": np.array([np.interp(20.0, tau[j], times) for j in range(ntr)])}
    model, rgts = C.section_age_model(tab, tau, times, ht)
    check(abs(rgts["Top wedge"] - 200) < 1.0 and abs(rgts["Base wedge"] - 280) < 2.0,
          f"horizon RGT from picks on the section ({rgts['Top wedge']:.1f}, {rgts['Base wedge']:.1f})")
    la = C.level_ages(levels, model)
    ax = C.age_axis(la, step=0.1)
    W, blank = C.wheeler_age(amp, hiatus, la, ax, model.gaps())
    pinched = ntr - 3
    inside = (ax > 22.5) & (ax < 29.5)
    full = 2
    check(blank[inside, pinched].all() and not blank[inside, full].any(),
          "age Wheeler: pinched-out traces blank across the wedge age, full-thickness traces filled")
    labels = C.gap_labels(blank, ax, np.arange(ntr))
    big = labels[0] if labels else None
    check(big is not None and abs(big["top"] - 22.0) < 0.8 and abs(big["base"] - 30.0) < 0.8 and
          big["traces"][1] == ntr - 1 and "Ma" in big["text"],
          "hiatus label spans ~22-30 Ma at the pinch-out: " + (big["text"] if big else "none"))
    # an explicit unconformity gap blanks every trace
    tab2 = C.AgeTable([{"name": "Top", "age": 2.0}, {"name": "Top wedge", "age": 18.0, "gap": 4.0},
                       {"name": "Base wedge", "age": 30.0}])
    m2, _ = C.section_age_model(tab2, tau, times, ht)
    la2 = C.level_ages(levels, m2)
    ax2 = C.age_axis(la2, step=0.1)
    W2, b2 = C.wheeler_age(amp, hiatus, la2, ax2, m2.gaps())
    g = (ax2 > 18.2) & (ax2 < 21.8)
    check(b2[g].all() and abs(sum(d["span"] for d in C.gap_labels(b2[:, :5], ax2)) - 4.0) < 0.3,
          "unconformity (18 -> 22 Ma) blank on all traces with a 4 My label")

    # zone lithology from (fake) wells
    class W_:
        def __init__(self, gr):
            self.depth = np.arange(len(gr)) * 0.5 + 1000.0
            self.logs = {"GR": gr}
    gr = np.concatenate([np.full(400, 25.0), np.full(400, 130.0), np.full(400, 25.0), np.full(400, 130.0)])
    gr += np.random.default_rng(0).normal(0, 1, gr.size)
    proj = types.SimpleNamespace(wells={"W1": None}, tops={"W1": {"A": 1000.0, "B": 1200.0, "C": 1300.0}},
                                 load_well=lambda n: W_(gr))
    tab3 = C.AgeTable([{"name": "A", "age": 1}, {"name": "B", "age": 2}, {"name": "C", "age": 3}])
    zl = C.zone_lithology(proj, tab3)
    check(zl["A"]["lith"]["sand"] > 0.9 and zl["B"]["lith"]["shale"] > 0.9 and
          0.3 < zl["C"]["lith"]["sand"] < 0.6, "zone lithology from well Vsh codes (sand / shale / mixed zones)")
    C.apply_lithology(tab3, zl)
    check(tab3.get("A")["lith"]["sand"] > 0.9 and "logs" in tab3.get("A")["lith_source"],
          "zone lithology copied into the age table")


def burial_table():
    from .basin import chrono as C
    return C.AgeTable([{"name": "Seabed", "age": 0.0, "lith": {"shale": 1}},
                       {"name": "A", "age": 10.0, "lith": {"sand": 1}},
                       {"name": "B", "age": 30.0, "lith": {"shale": 0.5, "carbonate": 0.5}},
                       {"name": "C", "age": 50.0, "lith": {"sand": 0.6, "shale": 0.4}},
                       {"name": "Base", "age": 80.0}])


def run_burial(check, tmp):
    from .basin import burial as B
    from .basin import chrono as C
    # analytic Athy: 1000 m from the surface with phi0 0.5, c 0.5/km holds 1000 - 1000 (1 - e^-0.5) grains
    s = float(B.solid_thickness(0.0, 1000.0, 0.5, 0.5e-3))
    check(abs(s - (1000 - 1000 * (1 - np.exp(-0.5)))) < 1e-9 and abs(float(B.base_for(0.0, s, 0.5, 0.5e-3)) - 1000) < 1e-6,
          f"Athy grain thickness {s:.2f} m and its inverse (base depth 1000 m)")
    # decompaction of a 100 m unit at 3000 m to the surface == analytic root (bisection)
    s2 = float(B.solid_thickness(3000.0, 3100.0, 0.63, 0.51e-3))
    h = float(B.base_for(0.0, s2, 0.63, 0.51e-3))
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        m = 0.5 * (lo + hi)
        f = m - 0.63 / 0.51e-3 * (1 - np.exp(-0.51e-3 * m)) - s2
        lo, hi = (m, hi) if f < 0 else (lo, m)
    check(abs(h - lo) < 1e-6 and h > 150, f"decompacted thickness at the surface {h:.2f} m == analytic {lo:.2f} m")
    m = B.mix({"sand": 0.5, "shale": 0.5})
    check(abs(m["phi0"] - 0.56) < 1e-9 and abs(m["k"] - np.sqrt(3.2 * 1.9)) < 1e-9, "lithology mix (mean phi0, "
          "geometric-mean conductivity)")

    T = burial_table()
    depths = {"Seabed": 100.0, "A": 900.0, "B": 2000.0, "C": 3200.0, "Base": 4000.0}
    units = B.column_units(T, depths, seabed=100.0)
    bh = B.BurialHistory(units, step=1.0, water_depth=100.0)
    tops, bases = bh.present()
    want = np.array([0, 800, 1900, 3100.0]), np.array([800, 1900, 3100, 3900.0])
    check(np.allclose(tops, want[0], atol=1e-4) and np.allclose(bases, want[1], atol=1e-4),
          "burial history reproduces the present-day column at 0 Ma")
    S = np.array([[B.solid_thickness(bh.top[i, k], bh.base[i, k], bh.phi0[k], bh.c[k]) if np.isfinite(bh.top[i, k])
                   else 0.0 for k in range(4)] for i in range(len(bh.ages))])
    full = bh.ages[:, None] <= np.array([u["age_top"] for u in units])[None, :]
    check(np.allclose(S[full], np.broadcast_to(bh.solid_full, S.shape)[full], rtol=1e-7, atol=1e-6),
          "grain thickness conserved through time (all units, all steps)")
    i50 = int(np.argmin(np.abs(bh.ages - 50.0)))
    i30 = int(np.argmin(np.abs(bh.ages - 30.0)))
    thick_c = bh.base[i50, 3] - bh.top[i50, 3]
    check(thick_c > 1000.0 and bh.top[i50, 3] == 0.0 and abs(bh.top[i30, 3] - (bh.base[i30, 2] - bh.top[i30, 2])) < 1e-9,
          f"unit C at 50 Ma: at the surface, decompacted to {thick_c:.0f} m (800 m today); under B at 30 Ma")
    # zero-porosity lithology: no compaction
    LZ = {k: dict(v) for k, v in B.LITHOLOGY.items()}
    LZ["sand"]["phi0"] = 0.0
    TZ = C.AgeTable([{"name": "Seabed", "age": 0.0, "lith": {"shale": 1}}, {"name": "A", "age": 10.0, "lith": {"sand": 1}},
                     {"name": "Base", "age": 20.0}])
    bz = B.BurialHistory(B.column_units(TZ, {"Seabed": 0.0, "A": 1500.0, "Base": 2500.0}, 0.0, props=LZ))
    hz = (bz.base - bz.top)[:, 1]
    check(np.allclose(hz[np.isfinite(hz) & (bz.ages <= 10.0)], 1000.0, atol=1e-6),
          "zero-porosity lithology keeps its thickness at every burial depth")
    # burial curves: monotonic burial without erosion, one curve per surface
    curves = bh.surfaces()
    names = [n for n, _, _ in curves]
    dC = dict((n, d) for n, _, d in curves)["C"]
    ok = np.isfinite(dC)
    check(names == ["Seabed", "A", "B", "C", "Base"] and np.all(np.diff(dC[ok]) >= -1e-6),
          "burial curves per surface; C deepens monotonically without erosion")
    # erosion: 600 m eroded at A during a 10 My hiatus (A = 10 Ma, deposits below end at 20 Ma)
    TE = burial_table()
    TE.set("A", gap=10.0, eroded=600.0)
    TE.set("B", age=30.0)
    ue = B.column_units(TE, {"Seabed": 100.0, "A": 400.0, "B": 2000.0, "C": 3200.0, "Base": 4000.0}, 100.0)
    be = B.BurialHistory(ue, step=0.5, water_depth=100.0)
    ghost = [i for i, u in enumerate(ue) if u["ghost"]][0]
    th = be.base[:, ghost] - be.top[:, ghost]
    i15 = int(np.argmin(np.abs(be.ages - 15.0)))
    dB = dict((n, d) for n, _, d in be.surfaces())["B"]
    t0 = be.present()
    check(abs(th[i15] - 600.0) < 1.0 and (np.isnan(th[-1]) or th[-1] == 0) and dB[i15] > dB[-1] and
          abs(t0[0][ghost + 1] - 300.0) < 1e-3 and abs(t0[1][ghost + 1] - 1900.0) < 1e-3,
          f"erosion: 600 m section on top at 15 Ma, removed by 10 Ma; B buried to {dB[i15]:.0f} m then uplifted "
          f"(today {dB[-1]:.0f} m); present column honoured")
    # porosity under the erosion surface stays at its maximum-burial value
    ph = be.porosity()[:, ghost + 1]
    check(ph[-1] <= np.nanmin(ph[be.ages >= 10.0]) + 1e-9, "no decompaction on uplift (porosity of maximum burial)")
    bh.to_csv(os.path.join(tmp, "burial.csv"))
    check(open(os.path.join(tmp, "burial.csv")).readline().startswith("age_Ma,"), "burial curves CSV")


def run_thermal(check, tmp):
    from .basin import burial as B
    from .basin import chrono as C
    from .basin import maturity as M
    from .basin import thermal as TH
    # steady state = analytic layered geotherm (no porosity, deposited long ago)
    L0 = {k: dict(v, phi0=0.0, A=0.0) for k, v in B.LITHOLOGY.items()}
    T0 = C.AgeTable([{"name": "Sea floor", "age": 0.0, "gap": 149.0, "lith": {"sand": 1}},
                     {"name": "X", "age": 150.0, "lith": {"shale": 1}},
                     {"name": "Base", "age": 151.0}])
    u = B.column_units(T0, {"Sea floor": 0.0, "X": 1000.0, "Base": 3000.0}, 0.0, props=L0)
    bh = B.BurialHistory(u, step=5.0)
    th = TH.ThermalHistory(bh, heat_flow=70.0, surface_t=10.0, basement={"A": 0.0})
    z, T = th.present_profile()
    q = 0.070
    ana = np.where(z <= 1000, 10 + q * z / 3.2, 10 + q * 1000 / 3.2 + q * (z - 1000) / 1.9)
    sed = z <= 3000
    err = np.max(np.abs(T[sed] - ana[sed]))
    check(err < 0.01, f"steady state == analytic two-layer geotherm (max error {err:.4f} degC)")
    LA = {k: dict(v, phi0=0.0) for k, v in B.LITHOLOGY.items()}
    LA["shale"]["A"] = 2.0
    u = B.column_units(C.AgeTable([{"name": "S", "age": 0.0, "gap": 149.0, "lith": {"shale": 1}},
                                    {"name": "Base", "age": 150.0}]),
                       {"S": 0.0, "Base": 4000.0}, 0.0, props=LA)
    th = TH.ThermalHistory(B.BurialHistory(u, step=5.0), heat_flow=60.0, surface_t=5.0, basement={"A": 0.0})
    z, T = th.present_profile()
    A = 2.0e-6
    qs = 0.060 + A * 4000.0
    ana = 5 + (qs * z - A * z * z / 2) / 1.9
    err = np.max(np.abs(T[z <= 4000] - ana[z <= 4000]))
    check(err < 0.01, f"steady state with radiogenic heat == analytic T = Ts + (qs z - A z^2/2)/k ({err:.4f} degC)")
    # transient: fast burial is cooler than its steady state (sedimentation blanketing)
    Tf = C.AgeTable([{"name": "S", "age": 0.0, "lith": {"shale": 1}}, {"name": "Base", "age": 2.0}])
    bf = B.BurialHistory(B.column_units(Tf, {"S": 0.0, "Base": 4000.0}, 0.0), step=0.1)
    tf = TH.ThermalHistory(bf, heat_flow=60.0, surface_t=10.0)
    Ts = C.AgeTable([{"name": "S", "age": 0.0, "lith": {"shale": 1}}, {"name": "Base", "age": 200.0}])
    bs = B.BurialHistory(B.column_units(Ts, {"S": 0.0, "Base": 4000.0}, 0.0), step=2.0)
    ts = TH.ThermalHistory(bs, heat_flow=60.0, surface_t=10.0)
    check(tf.temperature_at(3900.0) < ts.temperature_at(3900.0) - 10,
          f"transient: 4 km deposited in 2 My is cooler at 3.9 km ({float(tf.temperature_at(3900.0)):.0f} vs "
          f"{float(ts.temperature_at(3900.0)):.0f} degC steady)")
    # heat-flow calibration recovers a synthetic value
    T1 = burial_table()
    bh = B.BurialHistory(B.column_units(T1, {"Seabed": 100.0, "A": 900.0, "B": 2000.0, "C": 3200.0, "Base": 4000.0},
                                        100.0), water_depth=100.0)
    truth = TH.ThermalHistory(bh, heat_flow=65.0, surface_t=12.0)
    zs = np.array([1200.0, 2100.0, 2900.0, 3700.0])
    obs = np.c_[zs, truth.temperature_at(zs)]
    fit = TH.fit_heat_flow(bh, obs, heat_flow=50.0, surface_t=12.0)
    check(abs(fit["present_q"] - 65.0) < 0.05 and fit["rms"] < 0.05,
          f"heat-flow fit recovers 65 mW/m2 ({fit['present_q']:.2f}, rms {fit['rms']:.3f} degC)")
    noisy = obs.copy()
    noisy[:, 1] += np.array([1.5, -1.0, 0.8, -1.2])
    f2 = TH.fit_heat_flow(bh, noisy, heat_flow={"kind": "mckenzie", "rift_age": 80.0, "beta": 1.8, "present": 50.0},
                          surface_t=12.0)
    check(abs(f2["present_q"] - 65.0) < 3.0 and 0.5 < f2["rms"] < 2.0,
          f"fit with a McKenzie history shape and noisy data ({f2['present_q']:.1f} mW/m2, rms {f2['rms']:.2f})")
    ages = np.linspace(120, 0, 121)
    qm = TH.mckenzie_heat_flow(ages, 80.0, 2.0)
    q1 = TH.mckenzie_heat_flow(ages, 80.0, 1.0)
    after = ages <= 80
    check(np.all(np.diff(qm[after]) <= 1e-9) and qm[after][0] > 1.5 * qm[-1] and np.allclose(q1, q1[0]) and
          abs(TH.mckenzie_heat_flow([0.0], 80.0, 2.0, present=60.0)[0] - 60.0) < 1e-9,
          f"McKenzie rift heat flow: {qm[after][0]:.0f} -> {qm[-1]:.0f} mW/m2 (beta 2), constant for beta 1")
    # maturity
    tA = np.linspace(100.0, 0.0, 2001)
    tT = 10 + 2.0 * (tA[0] - tA)
    ro = M.easy_ro(tA, tT)
    r = lambda temp: float(np.interp(temp, tT, ro))  # noqa
    check(abs(ro[0] - np.exp(-1.6)) < 1e-12 and np.all(np.diff(ro) >= 0) and 0.5 < r(100) < 0.7 and
          0.9 < r(150) < 1.3 and 1.8 < r(200) < 2.6,
          f"EASY%Ro at 2 degC/My: {r(100):.2f} @100, {r(150):.2f} @150, {r(200):.2f} @200 degC (published curve)")
    # independent check: constant heating -> integral in temperature with fine quadrature
    H = 2.0 / M.SEC_MY
    Tk = np.linspace(283.15, 473.15, 200001)
    I = np.array([np.trapezoid(M.EASY_A * np.exp(-E / M.R_KCAL / Tk), Tk) / H for E in M.EASY_E]) \
        if hasattr(np, "trapezoid") else np.array([np.trapz(M.EASY_A * np.exp(-E / M.R_KCAL / Tk), Tk) / H
                                                    for E in M.EASY_E])
    ref = np.exp(-1.6 + 3.7 * np.sum(M.EASY_F * (1 - np.exp(-I))))
    check(abs(r(200) - ref) < 0.01, f"EASY%Ro == independent quadrature at 200 degC ({r(200):.3f} vs {ref:.3f})")
    check(abs(M.tti(np.linspace(15, 0, 151), np.full(151, 105.0))[-1] - 15.0) < 1e-9 and
          abs(M.tti(np.linspace(10, 0, 101), np.full(101, 125.0))[-1] - 40.0) < 1e-9,
          "Lopatin TTI: 15 My at 105 degC = 15, 10 My at 125 degC = 40")
    tr = {k: M.transformation_ratio(tA, tT, k) for k in M.KINETICS}
    t50 = {k: float(np.interp(0.5, v, tT)) for k, v in tr.items()}
    ks = list(M.KINETICS)
    check(all(np.all(np.diff(v) >= -1e-12) and v[0] == 0 and v[-1] > 0.99 for v in tr.values()) and
          t50[ks[2]] < t50[ks[1]] < t50[ks[3]] < t50[ks[4]] and 115 < t50[ks[1]] < 140,
          "transformation ratio 0 -> 1; TR50 order II-S < II < III < III-gas; type II at " + f"{t50[ks[1]]:.0f} degC")
    w = M.window_timing(tA, ro)
    check(w["oil_start"] is not None and w["oil_start"] > w["oil_end"] > 0 and M.window_name(0.8) == "oil window",
          f"oil window entry {w['oil_start']:.1f} Ma, exit {w['oil_end']:.1f} Ma")
    # maturity of every surface of a burial + thermal history: deeper = more mature today
    st = truth.surface_temperatures()
    ros = [float(M.easy_ro(truth.ages, Tt)[-1]) for _, _, Tt in st]
    check(np.all(np.diff(ros) > 0) and ros[0] < 0.3, "present %Ro increases with depth over the dated surfaces "
          + ", ".join(f"{n} {v:.2f}" for (n, _, _), v in zip(st, ros)))
    # BHT: Horner + LAS header parsing
    tc = 4.0
    tt = np.array([6.0, 12.0, 18.0])
    bht = 120 - 15 * np.log10((tc + tt) / tt)
    check(abs(TH.horner(tt, bht, tc) - 120.0) < 1e-9, "Horner extrapolation recovers the static temperature")
    las = os.path.join(tmp, "bht.las")
    with open(las, "w") as f:
        f.write("~Version\nVERS. 2.0 :\n~Well\nSTRT.M 100 :\n~Parameter\nBLI .F 9842.5 :Bottom log interval\n"
                "TCS . 0830 15JUN :Time circ. stopped\nTLAB. 0200 16JUN :Time logger at btm\n"
                "BHT .DEGF 212.0 :Bottom hole temp\nMRT .DEGF 0.0 :Max recorded temp\n~Curve\nDEPT.M :\n~A\n100\n")
    recs = TH.las_temperatures(las)
    check(len(recs) == 1 and abs(recs[0]["T"] - 100.0) < 1e-9 and abs(recs[0]["md"] - 3000.0) < 0.1 and
          abs(recs[0]["tsc_h"] - 17.5) < 1e-9, "LAS header BHT: degF -> degC, ft -> m, 17.5 h since circulation")
    wt = TH.well_temperatures([las], surface_t=10.0)
    check(len(wt) == 1 and abs(wt[0]["T"] - (100 + 0.05 * 90)) < 1e-9, "single-BHT correction (> 12 h: +5 %)")
    sc = TH.screen_temperatures([{"md": 500.0, "T": 95.0}, {"md": 3000.0, "T": 100.0}], lambda md: md - 25.0, 10.0, 100.0)
    check([r_["ok"] for r_ in sc] == [False, True], "implausible BHT (190 degC/km) flagged")


def run(check, tmp):
    run_chrono(check, tmp)
    run_burial(check, tmp)
    run_thermal(check, tmp)


def run_gui_chrono(check, tmp):
    from .basin import chrono as C
    from .gui.main_window import MainWindow
    from .rgt import solve as S
    from .rgt import stack as K
    from .test_rgt import geology_project
    p, G = geology_project(tmp, "basingui", fault=False)
    vol = p.volume("G")
    rv, _ = S.compute_rgt_3d(vol)
    st = K.build_stack_3d(vol, rv, 50)
    w = MainWindow(p.path)
    sp = w.section
    hs = sp.hset()
    K.stack_to_horizon(hs, "Shallow", 8, stack3d=st, vol_name="G")
    K.stack_to_horizon(hs, "Deep", 40, stack3d=st, vol_name="G")
    C.AgeTable([{"name": "Shallow", "age": 10.0}, {"name": "Deep", "age": 40.0, "gap": 0.0}]).save(sp.project)
    sp.source = ("3D", "G")
    sp.mode = "Inline"
    sp.number = int(vol.inlines[20])
    sp.domain = "Wheeler (age)"
    sp.redraw()
    ax = sp._age["axis"] if sp._age else None
    check(sp._data is not None and ax is not None and sp._data.shape == (len(ax), vol.nxl) and
          ax[0] >= 0 and 9.0 < np.interp(st.levels[8], st.levels, C.level_ages(st.levels, sp._age["model"])) < 11.0,
          f"Wheeler (age) domain: inline on an age axis {ax[0]:.1f}-{ax[-1]:.1f} Ma" if ax is not None else
          "Wheeler (age) domain")
    labs = sp.age_labels()
    check(isinstance(labs, list) and all("Ma" in t for _, _, t in labs),
          f"age Wheeler hiatus labels ({len(labs)}; e.g. {labs[0][2] if labs else '-'})")
    check(len(sp._hz_items) >= 2, "horizons drawn on the age axis")
    C.AgeTable([{"name": "Shallow", "age": 10.0}]).save(sp.project)
    sp.redraw()
    check(sp._age is None, "age axis with one dated horizon -> message, no crash")
    sp.domain = "Time"
    sp.redraw()
    return True


def run_gui(check, tmp):
    run_gui_chrono(check, tmp)
