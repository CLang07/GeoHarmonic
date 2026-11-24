"""Tests for PLAN 27: flow paths, traps, fill-spill, critical moment, fault seal (juxtaposition, SGR, stress), top seal."""
import math

import numpy as np


def _two_domes(ny=60, nx=100):
    Y, X = np.mgrid[0:ny, 0:nx].astype(float)
    # regional dip up to the west (smaller depth = shallower), two domes on it, a saddle between them
    return 2000 + 1.0 * X - 60 * np.exp(-((X - 25) ** 2 + (Y - 30) ** 2) / 80) - \
        60 * np.exp(-((X - 70) ** 2 + (Y - 30) ** 2) / 80), nx


def run(check, tmp):
    from . import migration as M
    # ---- 27a analytic cone: one trap, spill at the nearest edge
    ny, nx = 41, 41
    Y, X = np.mgrid[0:ny, 0:nx].astype(float)
    cone = 1000 + 2.0 * np.hypot(X - 20, Y - 15)
    an = M.spill_analysis(cone)
    t = an["traps"]
    c = 15 * nx + 20
    check(list(t) == [c] and abs(t[c]["spill"] - (1000 + 2 * 15)) < 1e-9 and t[c]["to"] == -1,
          "27a: cone - one trap at the crest, spill depth at the nearest edge (y = 0, 30 m closure)")
    rec = an["rec"]
    path = M.flow_path(rec, 40 * nx + 2)
    zs = cone.ravel()[path]
    check(path[-1] == c and np.all(np.diff(zs) < 0), "27a: flow path climbs monotonically to the crest")
    # two domes on a regional dip: the deeper one spills over the saddle into the up-dip one, which leaks off the map
    z, nx2 = _two_domes()
    an2 = M.spill_analysis(z, cell_area=625.0)
    tr = an2["traps"]
    up, down = 30 * nx2 + 24, 30 * nx2 + 69
    sad_i, sad_j = divmod(tr[down]["spill_node"], nx2)
    check(set(tr) == {up, down} and tr[down]["to"] == up and tr[up]["to"] == -1 and 40 < sad_j < 55 and
          abs(tr[up]["closure"] - tr[down]["closure"]) < 1.0,
          f"27a: dome + saddle - deeper trap spills at the saddle (column {sad_j}) into the up-dip trap, which leaks")
    small, _ = M.fill_spill(an2, M.drainage_charge(an2, 1000.0))
    big, lost = M.fill_spill(an2, M.drainage_charge(an2, 20000.0))
    check(small[up]["fill_frac"] < 1 and small[down]["spilled"] == 0 and big[down]["fill_frac"] == 1.0 and
          big[up]["received"] > 0 and big[up]["fill_frac"] == 1.0 and lost > 0,
          "27a: fill-spill order - deep trap fills first, its excess fills the up-dip trap, the rest is lost")
    cm = M.critical_moment(30.0, [50, 40, 30, 20, 10, 0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    late = M.critical_moment(5.0, [50, 40, 30, 20, 10, 0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    check(abs(cm["after_trap"] - 0.6) < 1e-9 and cm["verdict"] == "timing OK" and late["after_trap"] < 0.11 and
          late["verdict"] != "timing OK", "27a: critical moment - share of the charge expelled after trap formation")
    # ---- 27b fault seal
    jx, pairs = M.juxtaposition(np.arange(1000, 1200, 1.0), {"S1": (1000, 1040), "Sh": (1040, 1100), "S2": (1100, 1150)},
                                {"S1": (1060, 1100), "Sh": (1100, 1160), "S2": (1160, 1200)}, {"S1", "S2"})
    check(pairs == [] and jx.sum() == 0, "27b: juxtaposition: sands of both sides do not overlap - sealed by juxtaposition")
    jx2, pairs2 = M.juxtaposition(np.arange(1000, 1200, 1.0), {"S1": (1000, 1040), "Sh": (1040, 1100),
                                                               "S2": (1100, 1150)},
                                  {"S1": (1030, 1070), "Sh": (1070, 1130), "S2": (1130, 1180)}, {"S1", "S2"})
    check(jx2.sum() == 10 + 20 and ("S1", "S1") in pairs2 and ("S2", "S2") in pairs2,
          "27b: sand-on-sand windows (S1/S1 10 m, S2/S2 20 m) where the throw is smaller than the sands")
    d = np.arange(0, 200, 0.5)
    vsh = np.where((d // 10) % 2 == 0, 0.1, 0.9)  # 10 m beds alternating
    check(abs(M.sgr(d, vsh, 50.0, 20.0) - 0.5) < 0.03 and abs(M.sgr(d, vsh, 50.0, 0.0) - 0.9) < 1e-9,
          "27b: shale gouge ratio of alternating 10 m beds over a 20 m throw = 0.5")
    col, afpd = M.sgr_column(0.5, 2000.0)
    exp_afpd = 10 ** (50 / 27 - 0.5)
    check(abs(afpd - exp_afpd) < 1e-9 and abs(col - exp_afpd * 1e5 / (0.78 * 1000 * 9.81)) < 1e-6,
          f"27b: SGR 50 % at 2 km -> {afpd:.1f} bar -> {col:.0f} m gas column")
    perp = M.stress_tendency(90.0, 90.0, 50.0, 40.0, 30.0, 0.0)  # vertical E-W plane: normal = N = SHmax
    hz_ = M.stress_tendency(0.0, 0.0, 50.0, 40.0, 30.0, 0.0)
    check(abs(perp["sigma_n"] - 40.0) < 1e-9 and perp["tau"] < 1e-9 and abs(hz_["sigma_n"] - 50.0) < 1e-9,
          "27b: stress tensor - normal stress on planes normal to SHmax / horizontal")
    dips = np.arange(10, 90, 5.0)
    sl = [M.stress_tendency(0.0, dp, 60.0, 40.0, 30.0, 0.0, pp=20.0)["slip"] for dp in dips]  # N-S plane, Shmin E-W
    best = dips[int(np.argmax(sl))]
    check(50 <= best <= 70, f"27b: normal-faulting regime - slip tendency peaks on ~60 deg planes ({best:.0f} deg)")
    # ---- 27c top seal
    pce = M.micp_to_reservoir(100.0)
    check(abs(pce - 100.0 * 30.0 / (485.0 * math.cos(math.radians(40.0)))) < 1e-9, "27c: MICP -> reservoir entry "
          "pressure")
    cols = M.top_seal_column(np.array([0.6, 0.6]), np.array([50.0, 5.0]))
    check(abs(cols[0] - 10e5 / (0.78 * 1000 * 9.81)) < 1e-6 and cols[1] == 0,
          f"27c: top seal Vsh 0.6 -> 10 bar -> {cols[0]:.0f} m; thin seal holds nothing")
    check(M.limiting_column(80.0, 120.0, 60.0) == (60.0, "top seal") and M.limiting_column(50.0, 120.0) == (50.0, "spill"),
          "27c: limiting column = min(spill, fault seal, top seal)")


def run_gui(check, tmp):
    from .gui.basemap import Basemap
    from .gui.layers_dock import LayersPanel
    from . import maplayers as ML
    from .maps import MapGrid
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "mig_gui")
    bm = Basemap()
    bm.project = proj
    st = ML.LayerStack.load(proj)
    bm.set_stack(st)
    panel = LayersPanel(bm)
    panel.set_stack(st)
    z, nx = _two_domes()
    g = MapGrid(z, "xy", "m", "depth", None, 0.0, 0.0, 25.0, 25.0)
    lid, _ = st.add_grid(g, "Depth")
    an, fill = panel.migration(lid)
    names = [l["name"] for l in st.layers]
    txt = next(l for l in st.layers if l["name"] == "Traps (fill-spill)")
    check(len(an["traps"]) == 2 and "Drainage areas (Depth)" in names and len(txt["src"]["items"]) == 2 and
          any("-> T1" in it["text"] for it in txt["src"]["items"]) and not bm.layer_errors,
          "27a GUI: migration on the map hub - drainage areas + trap labels (T2 spills into T1)")
