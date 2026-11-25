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
    I = np.array([getattr(np, "trapezoid", getattr(np, "trapz", None))(M.EASY_A * np.exp(-E / M.R_KCAL / Tk), Tk) / H for E in M.EASY_E]) \
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


def synthetic_depth_grids(ny=31, nx=41, dx=250.0):
    """Depth maps (MapGrid 'xy') of the burial_table surfaces: a dome deepening to the east."""
    from .maps import MapGrid
    Y, X = np.mgrid[0:ny, 0:nx].astype(float)
    dome = 300.0 * np.exp(-((X - nx / 2) ** 2 + (Y - ny / 2) ** 2) / (2 * (nx / 5) ** 2))
    east = 25.0 * X
    z = {"Seabed": 100.0 + 0 * X, "A": 900.0 + 0.3 * east - 0.2 * dome, "B": 2000.0 + 0.6 * east - 0.6 * dome,
         "C": 3200.0 + east - dome, "Base": 4000.0 + east - dome}
    return {k: MapGrid(v.astype(np.float32), "xy", "m", k, None, 0.0, 0.0, dx, dx) for k, v in z.items()}


def run_maps(check, tmp):
    import time

    from .basin import burial as B
    from .basin import maturity as MT
    from .basin import pseudowells as PW
    from .basin import thermal as TH
    T = burial_table()
    names = ["Seabed", "A", "B", "C", "Base"]
    D = np.array([[100, 900, 2000, 3200, 4000], [100, 1000, 2300, 3600, 4500], [100, 700, 1500, 2500, 3000.0]])
    res = PW.run_columns(T, names, D, 100.0, heat_flow=62.0, surface_t=11.0, surfaces=["B", "C"], n_sub=4, full=True)
    single = []
    for row in D:
        bh = B.BurialHistory(B.column_units(T, dict(zip(names, row)), 100.0), water_depth=100.0)
        th = TH.ThermalHistory(bh, heat_flow=62.0, surface_t=11.0, n_sub=4)
        Tc = dict((n, t) for n, _, t in th.surface_temperatures())["C"]
        single.append((Tc[-1], MT.easy_ro(th.ages, Tc)[-1]))
    single = np.array(single)
    check(np.allclose(res["surfaces"]["C"]["temperature"], single[:, 0], atol=1e-6) and
          np.allclose(res["surfaces"]["C"]["ro"], single[:, 1], atol=1e-3),
          "batched pseudo-wells == one-column model (temperature, %Ro at C)")
    c = res["surfaces"]["C"]
    check(c["ro"][1] > c["ro"][0] > c["ro"][2] and np.all(np.isfinite(c["oil_entry"][:2])),
          "deeper column is more mature; oil-window entry ages " + ", ".join(f"{a:.1f}" for a in c["oil_entry"]))
    # Wheeler colouring: property of the sediment of each age, today and at 30 Ma
    ax = np.arange(0.0, 80.0, 0.5)
    W = PW.wheeler_property(res, ax, "ro", 0.0)
    k30 = int(np.argmin(np.abs(ax - 30.0)))
    check(np.allclose(W[k30], res["surfaces"]["B"]["ro"], atol=1e-3) and np.all(np.diff(W[:, 0][np.isfinite(W[:, 0])]) >= -1e-6),
          "Wheeler %Ro: the 30 Ma row == %Ro at surface B; %Ro increases with age")
    Wp = PW.wheeler_property(res, ax, "temperature", 30.0)
    check(np.isnan(Wp[ax < 29.9]).all() and np.isfinite(Wp[ax > 30.1]).all(),
          "Wheeler at 30 Ma: younger sediment not yet deposited (blank)")
    Wphi = PW.wheeler_property(res, ax, "porosity", 0.0)
    check(np.nanmax(Wphi) < 0.7 and np.nanmin(Wphi) > 0.0, "Wheeler porosity today")
    tr_x = np.linspace(0, 2, 11)
    tw = PW.to_traces(W, [0.0, 1.0, 2.0], tr_x)
    check(tw.shape == (len(ax), 11) and np.allclose(tw[:, 5], W[:, 1], equal_nan=True), "columns -> section traces")
    sec0 = PW.burial_section(res, 0.0)
    sec30 = PW.burial_section(res, 30.0)
    check(np.allclose(sec0["C"], D[:, 3], atol=1e-3) and np.all(sec30["C"] < sec0["C"]) and "A" not in sec30,
          f"restored section at 30 Ma: C at {sec30['C'][0]:.0f} m (today {sec0['C'][0]:.0f} m), A not yet deposited")
    # maps
    G = synthetic_depth_grids()
    t0 = time.time()
    M = PW.map_run(G, T, "C", stride=5, heat_flow=62.0, surface_t=11.0)
    dt = time.time() - t0
    ro = M["ro"].z
    check(ro.shape == G["C"].z.shape and np.isfinite(ro).all() and ro[:, -1].mean() > ro[:, 0].mean() and
          ro[15, 20] < ro[15, 35], f"%Ro map at C: {M['_info']['columns']} pseudo-wells in {dt:.2f} s, "
          f"{ro.min():.2f}-{ro.max():.2f} %Ro, crest less mature")
    j, i = 10, 15
    one = PW.run_columns(T, names, np.array([[float(G[n].z[i, j]) for n in names]]), 100.0, 62.0, None, 11.0,
                         surfaces=["C"])
    check(abs(float(one["surfaces"]["C"]["ro"][0]) - float(ro[i, j])) < 1e-4,
          "map node on the stride == direct column run")
    check(set(M) >= {"temperature", "ro", "tr", "oil_entry"} and M["temperature"].unit == "degC",
          "map outputs: temperature, %Ro, TR, oil-window entry")
    G2 = dict(G)
    G2["C"] = type(G["C"])(G["C"].z + 1, "xy", "m", "C", None, 0, 0, 250.0, 250.0)
    check(PW.fingerprint(G) != PW.fingerprint(G2) and PW.fingerprint(G) == PW.fingerprint(dict(G)),
          "map-run fingerprint changes with the depth maps (live update)")
    try:
        PW.map_run(G, T, "C", stride=5, cancel=lambda: True)
        ok = False
    except RuntimeError as e:
        ok = "cancel" in str(e)
    check(ok, "map run cancellable")
    # event chart
    T.set("C", role="source")
    T.set("B", role="reservoir")
    T.set("A", role="seal")
    T.set("Seabed", role="overburden")
    trs = PW.source_tr(res, "C", 0.5, column=1)
    ch = PW.event_chart(T, res["ages"], trs, trap=(12.0, 5.0))
    els = [d["element"] for d in ch["events"]]
    gen = [d for d in ch["events"] if d["element"] == "generation"]
    check(els[:4] == ["source", "reservoir", "seal", "overburden"] and "trap" in els and gen and
          gen[0]["start"] > gen[0]["end"] and ch["critical_moment"] is not None,
          f"event chart: elements from the age table, generation {gen[0]['start']:.1f}-{gen[0]['end']:.1f} Ma, "
          f"critical moment {ch['critical_moment']:.1f} Ma" if gen and ch["critical_moment"] else "event chart")
    png = PW.plot_event_chart(ch, os.path.join(tmp, "events.png"))
    check(os.path.getsize(png) > 5000, "event chart PNG")


def run(check, tmp):
    run_chrono(check, tmp)
    run_burial(check, tmp)
    run_thermal(check, tmp)
    run_maps(check, tmp)


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


def run_gui_basin(check, tmp):
    """18e: Interpret > Basin model window on the item-17 synthetic survey (2 horizons, 6 wells, depth model)."""
    from .basin import chrono as C
    from .gui.main_window import MainWindow
    from .rgt import solve as S
    from .rgt import stack as K
    from .test_ddc import LAYERS, _edit_tops, ddc_case, kz_earth
    sp, G = ddc_case(tmp, "basin18e", kz_earth)
    w = MainWindow(sp.path)
    dlg = w.basin_model()
    check(dlg.table.entries[0]["name"] == "Seabed" and len(dlg.table.entries) == 10 and dlg.grid.rowCount() == 10
          if isinstance(dlg.grid.rowCount(), int) else dlg.table.entries[0]["name"] == "Seabed",
          "18e: Basin model window opens with the Taranaki age template")
    dlg.table = C.AgeTable([{"name": "Seabed", "age": 0.0, "lith": {"shale": 0.8, "sand": 0.2}},
                            {"name": "Top A", "horizon": "H1", "age": 12.0, "lith": {"sand": 0.6, "shale": 0.4}},
                            {"name": "Top B", "horizon": "H2", "age": 35.0, "lith": {"shale": 1.0}}])
    dlg.refresh_table()
    dlg.set_entry("Top A", gap=2.0)
    dlg.set_settings(heat_flow={"kind": "constant", "q": 70.0}, surface_t=12.0, water_depth=100.0, step=1.0)
    dlg.save()
    check(w.project.display["basin"]["ages"][1]["gap"] == 2.0 and
          w.project.display["basin"]["settings"]["heat_flow"]["q"] == 70.0, "18e: age table + settings saved in the project")
    r = dlg.run_well("W-3")
    names = [row[0] for row in r["rows"]]
    check(names == ["Seabed", "Top A", "Top B"] and len(dlg.plot_data["burial"]) == 3 and "W-3" in dlg._last_text,
          "18e: Run well -> burial / temperature / %Ro curves per surface")
    dlg._safe(lambda: dlg.calibrate("W-3"))
    check("no usable temperatures" in dlg._last_text, "18e: calibration without BHTs explains why")
    z = r["burial"].present()[1][-1] + 100.0
    fit = dlg.calibrate("W-3", obs=[(0.5 * z, 12.0 + 0.5 * z * 0.04), (z, 12.0 + z * 0.04)])
    q_cal = fit["present_q"]
    r2 = dlg.run_well("W-3")
    check(abs(r2["thermal"].q[-1] - q_cal) < 1e-6 and "calibrated" in dlg._last_text and
          dlg.plot_data["calibration"]["obs"].shape == (2, 2), f"18e: heat flow calibrated to {q_cal:.1f} mW/m2 and "
          "used for the well")
    p = dlg.write_burial_csv(os.path.join(tmp, "b.csv"))
    check(os.path.exists(p), "18e: burial CSV export")
    # maps need the depth model
    try:
        dlg.run_map("Top B", 8, threaded=False)
        ok = False
    except RuntimeError as e:
        ok = "depth model" in str(e)
    check(ok, "18e: map run without a depth model explains why")
    vd = w.velocity_model()
    vd.build("3D:V", LAYERS, False)
    vd.use()
    m = dlg.run_map("Top B", 8, threaded=False)
    g = m["ro"]
    check(g.z.shape == G["z2"].shape and np.isfinite(g.z).mean() > 0.95 and m["_info"]["valid"] > 0,
          f"18e: %Ro map at Top B from {m['_info']['columns']} pseudo-wells")
    deep = np.unravel_index(np.nanargmax(G["z2"]), G["z2"].shape)
    shallow = np.unravel_index(np.nanargmin(G["z2"]), G["z2"].shape)
    check(m["temperature"].z[deep] > m["temperature"].z[shallow], "18e: map temperature follows the depth model")
    check(os.path.exists(dlg.export_cps3(os.path.join(tmp, "ro.cps3"), "ro")), "18e: map CPS-3 export")
    # live update: a top moves -> depth model rebuilds -> map re-runs
    dlg.auto_update = True
    fp0 = dlg.map_result["_info"]["fingerprint"]
    w.set_ddc(True)
    _edit_tops(w.project.well_project, lambda t: t["W-3"].__setitem__("Top B", t["W-3"]["Top B"] + 60.0))
    rep = w.ddc_tick()
    check(rep is not None and dlg.map_result["_info"]["fingerprint"] != fp0,
          "18e: basin map re-runs when the dynamic depth model changes")
    w.set_ddc(False)
    # Wheeler coloured by %Ro (needs RGT on the volume)
    vol = w.project.volume("V")
    rv, _ = S.compute_rgt_3d(vol)
    K.build_stack_3d(vol, rv, 40)
    sec = w.section
    sec.source = ("3D", "V")
    sec.mode = "Inline"
    sec.number = int(vol.inlines[20])
    data = dlg.colour_wheeler("ro", 0.0)
    fin = data[np.isfinite(data)]
    check(data is not None and fin.size > 0 and 0.15 < fin.min() and fin.max() < 3.0 and sec._age["prop"] == "ro" and
          sec.color_levels()[0] >= 0.15, f"18e: Wheeler (age) coloured by %Ro ({fin.min():.2f}-{fin.max():.2f})"
          if fin.size else "18e: Wheeler coloured by %Ro")
    dt = dlg.colour_wheeler("temperature", 20.0)
    ax = sec._age["axis"]
    check(np.isnan(dt[ax < 19.0]).all() and np.isfinite(dt[ax > 21.0]).any(),
          "18e: Wheeler temperature at 20 Ma - younger sediment blank")
    dlg.colour_wheeler("(seismic)")
    check(sec.basin_colour is None and sec._age is not None and sec._age["prop"] is None,
          "18e: back to seismic amplitude on the age axis")
    ch = dlg.event_chart("Top A", (8.0, 4.0), os.path.join(tmp, "ev.png"), well="W-3")
    check(os.path.exists(os.path.join(tmp, "ev.png")) and any(d["element"] == "trap" for d in ch["events"]) and
          dlg.table.get("Top A")["role"] == "source", "18e: event chart PNG (source role, trap)")
    sec.domain = "Time"
    return True


def run_gui(check, tmp):
    run_gui_chrono(check, tmp)
    run_gui_basin(check, tmp)
