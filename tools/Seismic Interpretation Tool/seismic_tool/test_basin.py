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


def run(check, tmp):
    run_chrono(check, tmp)
    run_burial(check, tmp)


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
