"""Property modelling + volumetrics tests (PLAN 36)."""
import os

import numpy as np

from . import framework as FW
from . import propmodel as PM


def box_grid(nx=20, ny=16, nz=10, top=1000.0, thick=100.0, cell=50.0):
    """Flat layer-cake grid: one zone, top..top+thick, nz proportional layers."""
    geom = FW.Geometry(0.0, 0.0, cell, cell, nx, ny)
    pts = (np.array([0, nx * cell, 0, nx * cell]), np.array([0, 0, ny * cell, ny * cell]))
    hz = [("Top", (pts[0], pts[1], np.full(4, top)), "conformable"),
          ("Base", (pts[0], pts[1], np.full(4, top + thick)), "conformable")]
    m = FW.FrameworkModel(geom, hz, [], "idw", "depth").build()
    return FW.Grid3D(m, [("proportional", nz)])


def run(check, tmp):
    G = box_grid()
    # 36a upscaling: vertical well at x = 525, y = 425 -> column (8, 10); samples every 1 m
    z = np.arange(990.0, 1110.0, 1.0)
    v = 0.1 + 0.001 * (z - 1000.0)
    cells, qc = PM.upscale_well(G, np.full(len(z), 525.0), np.full(len(z), 425.0), z, v)
    k3 = cells.get((3, 8, 10))
    exp = float(np.mean(v[(z >= 1030.0) & (z < 1040.0)]))
    check(len(cells) == 10 and abs(k3 - exp) < 1e-9 and qc[0]["samples"] == 10,
          f"36a: upscaling - one value per cell the well crosses (arithmetic {k3:.4f} = {exp:.4f})")
    ch, _ = PM.upscale_well(G, np.full(len(z), 525.0), np.full(len(z), 425.0), z, v, "harmonic")
    cg, _ = PM.upscale_well(G, np.full(len(z), 525.0), np.full(len(z), 425.0), z, v, "geometric")
    check(ch[(3, 8, 10)] <= cg[(3, 8, 10)] <= cells[(3, 8, 10)], "36a: harmonic <= geometric <= arithmetic")
    fac = np.where(z < 1050, 1, 2)
    cf, _ = PM.upscale_well(G, np.full(len(z), 525.0), np.full(len(z), 425.0), z, fac, discrete=True)
    check(cf[(4, 8, 10)] == 1 and cf[(5, 8, 10)] == 2, "36a: discrete facies upscaled with 'most of'")
    # 36d kriging exact at data, smooth between
    rng = np.random.default_rng(3)
    data = np.full((G.nz, G.ny, G.nx), np.nan)
    wells = [(2, 3), (12, 15), (5, 17), (13, 4), (8, 10)]
    for i, j in wells:
        data[:, i, j] = 0.15 + 0.1 * rng.random(G.nz)
    vario = PM.Vario(sill=0.001, r_major=8, r_minor=8, r_vert=4)
    K = PM.krige(G, data, vario)
    ok = np.isfinite(data)
    check(np.allclose(K[ok], data[ok]) and np.isfinite(K).all() and K.min() >= 0.1 and K.max() <= 0.3,
          "36d: kriging exact at the data cells, bounded between them")
    # SGS reproduces the histogram and the variogram
    big = box_grid(40, 40, 6)
    dat = np.full((big.nz, big.ny, big.nx), np.nan)
    ref = rng.normal(0.2, 0.03, 4000)
    for _ in range(25):
        i, j = rng.integers(0, 40, 2)
        dat[:, i, j] = rng.choice(ref, big.nz)
    vg = PM.Vario(sill=0.03 ** 2, r_major=8, r_minor=8, r_vert=3)
    reals = [PM.sgs(big, dat, vg, seed=s, ref=ref) for s in (1, 2)]
    r0 = reals[0]
    check(abs(np.mean(r0) - 0.2) < 0.01 and abs(np.std(r0) - 0.03) < 0.008 and not np.array_equal(r0, reals[1]),
          f"36d: SGS reproduces the input histogram (mean {np.mean(r0):.3f}, std {np.std(r0):.3f}); seeds differ")
    check(np.array_equal(PM.sgs(big, dat, vg, seed=1, ref=ref), r0), "36d: fixed seed -> same realization")
    Xc, Yc, Zc, _, _ = PM.cell_centres(big)
    lay = 2
    h, gam, cnt = PM.variogram_3d(Xc, Yc, np.full(Xc.shape, Zc[lay].mean()), r0[lay], azimuth=0.0, lags=8, lag=50.0)
    model = PM._model(h / 50.0, vg.sill, 8.0)
    near = h <= 300
    check(np.nanmax(np.abs(gam[near] - model[near])) < 0.6 * vg.sill,
          "36b/36d: SGS realization honours the variogram model (experimental vs model)")
    s, r, nug = PM.fit_variogram(h, gam, cnt)
    check(0.4 * vg.sill < s < 1.8 * vg.sill and 100 < r < 900, f"36b: variogram fit (sill {s:.5f}, range {r:.0f} m)")
    e, c = PM.histogram(r0, 15)
    check(c.sum() == r0.size and len(e) == 16, "36b: histogram")
    zs = PM.zone_stats(big, r0)
    check(len(zs) == 1 and zs[0]["n"] == r0.size, "36b: zone statistics")
    # 36c facies: SIS reproduces proportions; a soft trend pulls them
    fd = np.full((big.nz, big.ny, big.nx), np.nan)
    for _ in range(30):
        i, j = rng.integers(0, 40, 2)
        fd[:, i, j] = (rng.random(big.nz) < 0.3).astype(float)  # 30 % sand (1)
    vf = PM.Vario(sill=0.21, r_major=6, r_minor=6, r_vert=2)
    f1 = PM.sis(big, fd, [0, 1], vf, seed=4, proportions={0: 0.7, 1: 0.3})
    p_sand = float(np.mean(f1 == 1))
    soft = np.zeros((2,) + fd.shape)
    soft[1] = 0.6
    soft[0] = 0.4
    f2 = PM.sis(big, fd, [0, 1], vf, seed=4, soft=soft, proportions={0: 0.7, 1: 0.3})
    known = np.isfinite(fd)
    check(abs(p_sand - 0.3) < 0.08 and np.array_equal(f1[known], fd[known]) and np.mean(f2 == 1) > p_sand + 0.05,
          f"36c: SIS honours data + proportions ({p_sand:.2f} sand); soft trend raises it to {np.mean(f2 == 1):.2f}")
    P = PM.indicator_kriging(big, fd, [0, 1], vf, proportions={0: 0.7, 1: 0.3})
    check(np.allclose(P.sum(axis=0), 1.0) and np.allclose(P[1][known], fd[known]), "36c: indicator kriging "
          "probabilities sum to 1 and honour the data")
    phi_f = PM.sgs(big, dat, vg, seed=7, facies=f1, facies_code=1, ref=ref)
    check(np.isfinite(phi_f[f1 == 1]).all() and np.isnan(phi_f[f1 == 0]).all(), "36d: porosity simulated in the "
          "sand facies only")
    # 36e volumetrics on a box reservoir: 1000 x 800 m, 1000-1100 m, OWC 1050 m, GOC 1020 m
    B = box_grid(20, 16, 10)
    phi = np.full((B.nz, B.ny, B.nx), 0.2)
    sw = np.full(phi.shape, 0.3)
    rows = PM.volumetrics(B, phi, sw, owc=1050.0, goc=1020.0, bo=1.25, bg=0.004)
    area = 1000.0 * 800.0
    oil_grv = area * 30.0
    gas_grv = area * 20.0
    stoiip = oil_grv * 0.2 * 0.7 / 1.25
    giip = gas_grv * 0.2 * 0.7 / 0.004
    check(abs(PM.totals(rows) - stoiip) < 1e-6 * stoiip and abs(PM.totals(rows, "giip_m3") - giip) < 1e-6 * giip and
          abs(rows[0]["grv_m3"] - area * 50.0) < 1e-6 * area,
          f"36e: volumetrics = hand-computed box (STOIIP {PM.totals(rows) / 1e6:.3f} Mm3, GIIP {giip / 1e9:.3f} Gm3)")
    st = PM.realization_stats([100.0, 120.0, 90.0, 110.0, 130.0])
    check(st["P90"] < st["P50"] < st["P10"] and st["n"] == 5, "36e: P90 / P50 / P10 over realizations")
    bs = PM.block_summary(B, rows)
    check(len(bs) == 1 and abs(bs[0]["x"] - 500.0) < 1e-6, "36e: per-block totals + positions for the Live Map")
    p = PM.write_csv(os.path.join(tmp, "vols.csv"), rows)
    gp = FW.write_grdecl(os.path.join(tmp, "props.grdecl"), B, props={"PORO": phi, "SWAT": sw})
    r = FW.read_grdecl(gp)
    check(os.path.getsize(p) > 50 and np.allclose(r["PORO"], 0.2) and np.allclose(r["SWAT"], 0.3),
          "36f: property grids to GRDECL, volumetrics to CSV")


def run_gui(check, tmp):
    """36 GUI: property dialog from the framework dialog: upscale, variogram, SGS, volumetrics -> Live Map, export."""
    from .gui.main_window import MainWindow
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "pm_gui")
    mw = MainWindow(proj.path)
    fd = mw.framework("Synth")
    fd.build()

    class VM:
        def depth_of_time(self, x, y, times):
            return np.asarray(times, float)[None, :] * 1.25 + 0.0 * np.asarray(x, float)[:, None]
    fd.depth_model_fn = lambda: VM()
    fd.depth()
    fd.make_grid([("proportional", 4)])
    pd = fd.properties()
    G = fd.grid
    Xc, Yc, Zc, top, bot = PM.cell_centres(G)
    rng = np.random.default_rng(2)
    wells = []
    for i, j in ((2, 2), (5, 8), (8, 3), (3, 9)):
        if i >= G.ny or j >= G.nx:
            continue
        z = np.linspace(np.nanmin(top[:, i, j]), np.nanmax(bot[:, i, j]) - 0.01, 60)
        wells.append((np.full(60, Xc[i, j]), np.full(60, Yc[i, j]), z, 0.2 + 0.02 * rng.standard_normal(60)))
    a = pd.upscale("PHIE", wells)
    check(np.isfinite(a).sum() >= 4 * len(wells) - 2, "36a GUI: logs upscaled into the grid")
    v = pd.variogram("PHIE")
    check(v["sill"] > 0 and pd.vario.sill > 0, "36b GUI: variogram fitted")
    reals = pd.simulate("PHIE", 3)
    check(len(reals) == 3 and all(np.isfinite(r[G.actnum > 0]).all() for r in reals), "36d GUI: 3 SGS realizations")
    seen = []
    mw.mapbus.subscribe("probe", seen.append)
    owc = float(np.nanmedian(Zc))
    st = pd.volumetrics(owc=owc)
    check(st["P90"] <= st["P50"] <= st["P10"] and st["P50"] > 0, "36e GUI: STOIIP P90 / P50 / P10 over realizations")
    lm = mw.live_map()
    ev = [e for e in seen if e.get("kind") == "layers"]
    check(ev and ev[-1]["layer"]["kind"] == "points", "36e GUI: per-block volumes sent to the Live Map")
    lm.add_layer_request(ev[-1]["layer"])
    check(any(l["name"] == "Volumetrics per block" for l in lm.canvas.stack.layers), "36e GUI: shown on the Live Map")
    out = pd.export(os.path.join(d, "props.grdecl"), os.path.join(d, "vol.csv"))
    check(len(out) == 2 and all(os.path.exists(p) for p in out), "36f GUI: GRDECL with PORO + volumetrics CSV")
    mw.maplink.close()
