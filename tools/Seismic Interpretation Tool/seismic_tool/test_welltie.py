"""Wells on seismic: time-depth, tops in TWT, synthetic seismogram, auto tie (synthetic data + real NZ data)."""
import os

import numpy as np

LAT, LON = -39.50, 173.30
V0 = 1800.0  # checkshot velocity function v(z) = V0 + K z  (m/s, z below MSL)
K = 0.45
SHIFT = 12.0  # ms: seismic is later than the uncorrected synthetic


def true_twt(z):
    z = np.asarray(z, float)
    return 2000.0 * np.log1p(K * z / V0) / K


def well_project(tmp):
    """Well Correlation project with W-1 (logs + checkshot CSV) and W-2 (logs only, 1.2 km east)."""
    from .welltie import wc
    wcpkg = wc()
    from well_correlation.selftest import _write_las
    from well_correlation.geo import tm_forward
    d = os.path.join(tmp, "wells_data")
    rng = np.random.default_rng(4)
    x1, y1 = tm_forward(LAT, LON)
    rows = []
    for name, dx in (("W-1", 0.0), ("W-2", 1200.0)):
        wd = os.path.join(d, "wells", name, "las")
        os.makedirs(wd)
        md = np.arange(900.0, 2600.0, 0.1524)
        z = md - 25.0
        # blocky layers; velocity follows the checkshot trend so sonic and checkshot agree
        edges = np.sort(rng.uniform(900, 2600, 60))
        lay = np.searchsorted(edges, md)
        pert = rng.uniform(-0.12, 0.12, 61)[lay]
        v = (V0 + K * z) * (1 + pert)
        dt = 304800.0 / v
        rho = 0.31 * v ** 0.25 * (1 + 0.5 * pert)
        gr = 60 + 60 * (pert < 0)
        _write_las(os.path.join(wd, f"{name}.las"), name, md, {"DT": dt, "RHOB": rho, "GR": gr},
                   {"DT": "US/F", "RHOB": "G/C3", "GR": "GAPI"})
        rows.append((name, dx))
    # header: W-2 placed 1200 m east (convert back to lat/lon by a local approximation)
    lon2 = LON + 1200.0 / (111320.0 * np.cos(np.radians(LAT)))
    with open(os.path.join(d, "wells_header.csv"), "w") as fh:
        fh.write("Well,UWI,Latitude,Longitude,Elevation_m,Elev_Ref,TD_m\n")
        fh.write(f"W-1,1,{LAT},{LON},25.0,KB,2600\nW-2,2,{LAT},{lon2},25.0,KB,2600\n")
    cs = os.path.join(d, "wells", "W-1", "checkshot")
    os.makedirs(cs)
    zc = np.arange(100.0, 2600.0, 100.0)
    with open(os.path.join(cs, "w1_checkshot.csv"), "w") as fh:
        fh.write("MD_m,TVD_SRD_m,transit_ms,OWT_ms,TWT_ms,n_shots,std_ms\n")
        for z in zc:
            fh.write(f"{z + 25:.2f},{z:.2f},0,{true_twt(z) / 2:.3f},{true_twt(z):.3f},3,0.1\n")
    P = wcpkg.project.Project
    p = P.from_folder(d)
    p.tops = {"W-1": {"Sand A": 1500.0, "Shale B": 2100.0}, "W-2": {"Sand A": 1520.0}}
    p.formations = [{"name": "Sand A", "color": "#ffcc00"}, {"name": "Shale B", "color": "#666666"}]
    path = os.path.join(d, "test.wcproj.json")
    p.save(path)
    return path, (x1, y1)


def seismic_project(tmp, wpath, xy):
    """3D survey centred on W-1 whose traces are W-1's synthetic, delayed by SHIFT ms."""
    from .project import SeisProject
    from .synth import write_3d
    from .volume import import_3d
    from . import welltie as WT
    d = os.path.join(tmp, "seis")
    os.makedirs(d)
    sp = SeisProject()
    sp.path = os.path.join(d, "ws.seisproj.json")
    sp.name = "ws"
    sp.well_project = wpath
    ws = WellSet = WT.WellSet(sp)  # noqa: F841
    w = ws.well("W-1")
    ws.wcp.load_well("W-1")
    td = WT.WellTD("W-1", np.r_[0, np.arange(0, 3000, 10.0)], np.r_[0, true_twt(np.arange(0, 3000, 10.0))], "truth")
    t, ai = WT.logs_in_time(w, td, 4.0)
    _, syn = WT.synthetic(t, ai, WT.ricker(30.0, 4.0))
    ns = 700
    tt = np.arange(ns) * 4.0
    tr = np.interp(tt, t + SHIFT, syn, left=0, right=0).astype(np.float32)
    nil, nxl = 15, 17
    cube = np.tile(tr, (nil, nxl, 1)) + 0.002 * np.random.default_rng(1).standard_normal((nil, nxl, ns)).astype(np.float32)
    x0, y0 = xy[0] - 25.0 * 8, xy[1] - 25.0 * 7
    write_3d(os.path.join(d, "ws.sgy"), cube, x0=round(x0), y0=round(y0))
    store = os.path.join(d, "ws_seismic", "S")
    import_3d(os.path.join(d, "ws.sgy"), store)
    sp.volumes.append({"name": "S", "store": store, "segy": ""})
    sp.save()
    return sp


def run(check, tmp):
    from . import welltie as WT
    d = os.path.join(tmp, "welltie")
    os.makedirs(d)
    wpath, xy = well_project(d)
    sp = seismic_project(d, wpath, xy)
    ws = sp.wellset(reload=True)
    check(set(ws.names) == {"W-1", "W-2"}, "well project linked: wells listed")
    td1 = ws.td_for("W-1")
    z = np.array([500.0, 1500.0, 2400.0])
    check(td1.source.startswith("checkshot") and np.allclose(td1.twt(z), true_twt(z), atol=0.5),
          f"W-1 time-depth from its checkshot ({td1.source})")
    td2 = ws.td_for("W-2")
    zz = np.array([1200.0, 2000.0, 2500.0])
    err = np.abs(td2.twt(zz) - true_twt(zz)).max()
    check("sonic anchored to checkshot of W-1" in td2.source and err < 12.0,
          f"W-2 time-depth from sonic anchored to the nearest checkshot (max err {err:.1f} ms)")
    tops = ws.tops_twt("W-1")
    check(abs(tops["Sand A"] - true_twt(1500.0 - 25.0)) < 0.5 and tops["Shale B"] > tops["Sand A"],
          "tops converted MD -> TVDSS -> TWT")
    vol = sp.volume("S")
    il, xl = vol.ilxl_at(*xy)
    near = ws.near_section("3D", vol, "Inline", int(round(il)))
    check([n for n, _, _ in near] == ["W-1"] and abs(near[0][1] - xl) < 1e-6, "well found on the inline through it")
    check(ws.near_section("3D", vol, "Inline", int(vol.inlines[0])) == [], "well not shown on a far inline")
    # synthetic + auto tie
    w = ws.well("W-1")
    t, ai = WT.logs_in_time(w, td1, 4.0)
    r, syn = WT.synthetic(t, ai, WT.ricker(30.0, 4.0))
    ts, trs = ws.trace_at_well("3D", vol, "W-1")
    c0 = WT.tie_correlation(t, syn, ts, trs[0], 0.0, 1.0, float(np.median(t)))
    res = WT.auto_tie(t, syn, ts, trs[0], 60.0, np.linspace(0.98, 1.02, 5))
    check(abs(res["shift_ms"] - SHIFT) < 1.0 and abs(res["stretch"] - 1.0) < 0.011 and res["corr"] > 0.9,
          f"auto tie recovers the {SHIFT:.0f} ms shift: {res['shift_ms']:.2f} ms, stretch {res['stretch']:.3f}, "
          f"corr {c0:.2f} -> {res['corr']:.2f}")
    _, trs9 = ws.trace_at_well("3D", vol, "W-1", n_around=1)
    wav = WT.statistical_wavelet(trs9, 4.0, 128.0)
    _, syn2 = WT.synthetic(t, ai, wav)
    res2 = WT.auto_tie(t, syn2, ts, trs[0], 60.0, [1.0])
    check(abs(res2["shift_ms"] - SHIFT) < 2.0 and res2["corr"] > 0.8 and abs(np.argmax(wav) - len(wav) // 2) <= 1,
          f"statistical (zero-phase) wavelet tie: shift {res2['shift_ms']:.1f} ms, corr {res2['corr']:.2f}")
    ws.save_tie("W-1", res["shift_ms"], res["stretch"], res["t_centre"], res["corr"])
    sp.save()
    from .project import SeisProject
    sp2 = SeisProject.open(sp.path)
    td1b = sp2.wellset().td_for("W-1")
    check(abs(td1b.twt(1500.0) - (true_twt(1500.0) + res["shift_ms"] + (res["stretch"] - 1) * (true_twt(1500.0) - res["t_centre"]))) < 0.6,
          "saved tie applied to the well's time-depth after reopening")

    # ---- real NZ data: Tui-1 checkshot from DLIS, Maui wells from sonic
    real = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "nz",
                                         "NZ Well Correlation.wcproj.json"))
    if not os.path.exists(real):
        print("SKIP real well checks")
        return
    from .project import SeisProject as SP
    rp = SP()
    rp.path = os.path.join(d, "real.seisproj.json")
    rp.well_project = real
    wsr = rp.wellset()
    tui = wsr.td_for("Tui-1")
    check(tui.source.startswith("checkshot") and 2700 < tui.twt(3875.0) < 2900, f"Tui-1 TD from DLIS checkshot "
                                                                                  f"(3875 m TVDSS -> {tui.twt(3875.0):.0f} ms)")
    m5 = wsr.td_for("Maui-5")
    tt = wsr.tops_twt("Maui-5")
    check(("sonic" in m5.source or "checkshot of" in m5.source) and all(1000 < v < 4000 for v in tt.values()) if tt
          else ("sonic" in m5.source or "checkshot of" in m5.source),
          f"Maui-5 TD: {m5.source}; tops {', '.join(f'{k} {v:.0f}' for k, v in list(tt.items())[:3])} ms")


def run_gui(check, tmp):
    from .gui.section_panel import SectionPanel
    from .gui.welltie_dialog import WellTieDialog
    d = os.path.join(tmp, "welltie_gui")
    os.makedirs(d)
    wpath, xy = well_project(d)
    sp = seismic_project(d, wpath, xy)
    panel = SectionPanel()
    panel.project = sp
    panel.source = ("3D", "S")
    vol = sp.volume("S")
    il, xl = vol.ilxl_at(*xy)
    panel.mode = "Inline"
    panel.number = int(round(il))
    panel.redraw()
    ov = panel.well_overlay()
    check(len(ov) == 1 and ov[0]["name"] == "W-1" and len(ov[0]["tops"]) == 2, "wells + tops drawn on the inline")
    panel.mode = "Time slice"
    panel.number = 1200
    panel.redraw()
    check(len(panel.well_overlay()) == 1, "well marker on the time slice")
    dlg = WellTieDialog(sp, ("3D", "S"))
    dlg.well_name = "W-1"
    dlg.freq = 30.0
    dlg.wavelet_kind = "Ricker"
    data = dlg.compute()
    check(data is not None and len(data["syn"]) > 100, "tie dialog: synthetic computed")
    res = dlg.auto()
    check(abs(res["shift_ms"] - SHIFT) < 1.5 and res["corr"] > 0.9, f"tie dialog: auto tie {res['shift_ms']:.1f} ms")
    dlg.save()
    check("W-1" in sp.display.get("well_ties", {}), "tie dialog: tie saved in the seismic project")
    return True
