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


def run(check, tmp):
    run_chrono(check, tmp)


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
