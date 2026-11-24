"""Tests for the 'Add wells' workflow (PLAN 21d)."""
import os

import numpy as np


def _well(root, name, h, rng):
    from .welltie import wc
    wc()
    from well_correlation.petro.test_petro import write_las
    d = os.path.join(root, "wells", name, "las")
    os.makedirs(d)
    z = np.arange(1000.0, 1200.0, 0.1524)
    pay = (z >= 1060) & (z < 1060 + h)
    V = np.where(pay, 0.1, 0.9)
    P = np.where(pay, 0.22, 0.0)
    S = np.where(pay, 0.3, 1.0)
    e = lambda s_: s_ * rng.standard_normal(z.size)  # noqa: E731
    rhob = (1 - P - V) * 2.65 + V * 2.55 + P + e(0.0003)
    nphi = P + V * 0.35 + e(0.0003)
    gr = 20 + 100 * V + e(0.05)
    rt = 1.0 / (P ** 2 * S ** 2 / 0.05 + V * S / 2.0) * np.exp(e(0.002))
    write_las(os.path.join(d, f"{name}.las"), name, z, {"GR": gr, "RHOB": rhob, "NPHI": nphi, "LLD": rt}, {"NPHI": "V/V"})


def _header(root, rows):
    with open(os.path.join(root, "wells_header.csv"), "w") as f:
        f.write("Well,UWI,Latitude,Longitude,Elevation_m,Elev_Ref,TD_m\n")
        for n, lat, lon in rows:
            f.write(f"{n},{n},{lat},{lon},0,KB,1200\n")


def _build(tmp):
    from .welltie import wc
    wc()
    from well_correlation.project import Project
    from well_correlation.petro import interp as PI
    from .project import SeisProject
    from . import maplayers as ML
    from . import propgrid as PG
    from . import addwells as AW
    rng = np.random.default_rng(9)
    root = os.path.join(tmp, "aw_wells")
    hs = {"A-1": 20.0, "B-1": 40.0, "C-1": 30.0, "D-1": 10.0, "E-1": 50.0}
    loc = {"A-1": (-39.60, 173.30), "B-1": (-39.62, 173.34), "C-1": (-39.64, 173.30), "D-1": (-39.61, 173.32),
           "E-1": (-39.63, 173.33)}
    for n in ("A-1", "B-1", "C-1"):
        _well(root, n, hs[n], rng)
    _header(root, [(n,) + loc[n] for n in ("A-1", "B-1", "C-1")])
    wp = Project.from_folder(root)
    wp.formations = [{"name": "R", "color": "#ffff00"}, {"name": "Base", "color": "#000000"}]
    wpath = os.path.join(tmp, "aw.wcproj.json")
    wp.save(wpath)
    new = os.path.join(tmp, "aw_new")
    for n in ("D-1", "E-1"):
        _well(new, n, hs[n], rng)
    _header(new, [(n,) + loc[n] for n in ("D-1", "E-1")])
    # tops for all five (the new wells' tops arrive with them in practice; here written into the project up front)
    wp = Project.open(wpath)
    wp.tops = {n: {"R": 1050.0, "Base": 1150.0} for n in hs}
    wp.save()
    pset = {"default": {"gr_clean": 20.0, "gr_shale": 120.0, "rho_sh": 2.55, "nphi_sh": 0.35, "rw": 0.05, "rsh": 2.0,
                        "sw_method": "simandoux"}}
    PI.save_params(wp, pset)
    PI.interpret_project(wp)
    sp = SeisProject()
    sp.path = os.path.join(tmp, "aw.seisproj.json")
    sp.name = "aw"
    sp.well_project = wpath
    sp.save()
    st = ML.LayerStack.load(sp)
    vals = ML.petro_values(sp, "SoPhiH", "R")
    px, py, v = (np.array([q[i] for q in vals], float) for i in (1, 2, 3))
    g = PG.grid_property(px, py, v, "idw", cell=100.0, label="SoPhiH R")
    lid, _ = st.add_grid(g, "SoPhiH R (idw)")
    st.get(lid)["src"]["recipe"] = {"prop": "SoPhiH", "zone": "R", "method": "idw", "secondary": None, "cell": 100.0}
    st.save()
    return sp, st, new, lid, wpath, hs, len(vals)


def run(check, tmp):
    from .welltie import wc
    wc()
    from well_correlation.project import Project
    from . import maplayers as ML
    from . import propgrid as PG
    from . import addwells as AW
    sp, st, new, lid, wpath, hs, n0 = _build(tmp)
    check(n0 == 3, "21d setup: 3 wells with SoPhiH")
    rep = AW.add_wells(sp, st, new, tie=False, change_layers=True)
    check(rep["added"] == ["D-1", "E-1"] and len(Project.open(wpath).wells) == 5, "21d: new wells imported into the well project")
    vals2 = {w: v for w, _, _, v in ML.petro_values(sp, "SoPhiH", "R")}
    check(set(vals2) == set(hs) and abs(vals2["E-1"] - 50 * 0.22 * 0.7) < 0.3, f"21d: petrophysics run on the new wells "
          f"(E-1 SoPhiH {vals2.get('E-1', 0):.2f}, expect {50 * 0.154:.2f})")
    res = {r["well"]: r for r in rep["residuals"]}
    check(set(res) == {"D-1", "E-1"} and all(np.isfinite(r["predicted"]) for r in res.values()) and
          abs(res["E-1"]["residual"] - (vals2["E-1"] - res["E-1"]["predicted"])) < 1e-9,
          "21d: old map tested at the new wells (blind residuals)")
    rg = rep["regridded"][0]
    newg = st.grid(lid)
    xy = {n: (x, y) for n, x, y in sp.wells()}
    at_e = float(PG.sample_grid(newg, [xy["E-1"][0]], [xy["E-1"][1]])[0])
    check(rg["rms_change"] > 0 and abs(at_e - vals2["E-1"]) < 0.05 * vals2["E-1"], f"21d: map re-gridded with the new wells "
          f"(value at E-1 {at_e:.2f} vs {vals2['E-1']:.2f}); change layer added")
    check(rg.get("change_layer") and st.get(rg["change_layer"])["kind"] == "grid", "21d: before / after change layer")
    txt = AW.report_text(rep)
    check("RMS residual" in txt and "re-gridded" in txt, "21d: report text")
    rep2 = AW.add_wells(sp, st, new, tie=False)
    check(rep2["added"] == [], "21d: adding the same folder again adds nothing")


def run_gui(check, tmp):
    from .gui.basemap import Basemap
    from .gui.layers_dock import LayersPanel
    sp, st, new, lid, wpath, hs, n0 = _build(tmp)
    bm = Basemap()
    bm.project = sp
    bm.set_stack(st)
    panel = LayersPanel(bm)
    panel.set_stack(st)
    rep = panel.add_new_wells(new, tie=True)
    check(rep["added"] == ["D-1", "E-1"] and "RMS residual" in panel.last_report and sp.display["map_layers"],
          "21d GUI: Map layers > New wells runs the workflow (no survey: tie skipped) and reports")
