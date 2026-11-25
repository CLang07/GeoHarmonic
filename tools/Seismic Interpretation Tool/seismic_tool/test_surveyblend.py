"""PLAN 42 tests: survey horizon blending on a synthetic two-survey overlap with a known spatially varying residual
(tilt + local bump), a fault across the seam and wells inside and outside the blend zone."""
import os

import numpy as np

from . import surveyblend as SB

CELL = 50.0


def scene(spike=False, fault=True):
    g = SB.MasterGrid(0.0, 0.0, CELL, CELL, 100, 240)  # 5 km E-W x 12 km N-S
    X, Y = g.xy()
    T = 2000.0 + 0.004 * X + 0.003 * Y + 8 * np.sin(X / 900.0)
    fx = 3525.0
    if fault:
        T = T + np.where(X > fx, 40.0, 0.0)  # N-S fault, 40 m throw, crossing the seam
    rng = np.random.default_rng(1)
    za = np.where(Y <= 3000.0, T + rng.normal(0, 0.05, T.shape), np.nan)  # survey A (priority): south
    rtrue = -(4.0 + 0.003 * X) - 5.0 * np.exp(-((X - 1500) ** 2) / (2 * 500.0 ** 2))  # B - truth: tilt + bump
    zb = np.where(Y >= 2000.0, T + rtrue, np.nan)  # survey B: north, 1 km overlap
    if spike:
        zb = zb.copy()
        zb[50, 30] += 400.0  # a bad pick in the overlap
    faults = [[(fx, -100.0), (fx, 12100.0)]] if fault else []
    return g, X, Y, T, za, zb, rtrue, faults, fx


def run(check, tmp):
    os.makedirs(tmp, exist_ok=True)
    g, X, Y, T, za, zb, rtrue, faults, fx = scene()
    s = 2.0  # m/km
    A, B = SB.Survey("A", za, 1), SB.Survey("B", zb, 2)
    r = SB.blend(g, [A, B], slope=s, unit="m/km", method="hybrid", w_min=500.0, w_max=12000.0, faults=faults)
    near_f = SB.rasterize_polylines(g, faults, 3)
    near_f |= SB._shift(near_f, 0, 1, False) | SB._shift(near_f, 0, -1, False)
    ok = np.isfinite(r.z) & ~near_f
    ok[:2] = ok[-2:] = False
    ok[:, :2] = ok[:, -2:] = False
    smax = float(np.nanmax(r.qc["added_slope"][ok])) * 1000.0
    check(smax <= s * 1.05, f"42d/n: added slope across the seam never exceeds the tolerance ({smax:.2f} <= {s} m/km)")
    tot = r.qc["added_slope_total"][ok]
    check(np.isfinite(tot).mean() > 0.9 and np.nanmax(tot) >= np.nanmax(r.qc["added_slope"][ok]), "42l: total added slope (incl. the residual's own variation "
          "along the seam) reported as a separate map")
    W = r.width
    check(np.all(W >= 500.0) and abs(np.median(W) - np.median(np.abs(r.residual)) / (s / 1000.0)) < 1.0
          and W.max() > 4 * W.min(), f"42d: width = |residual| / slope, varying along the seam "
          f"({W.min() / 1000:.1f}-{W.max() / 1000:.1f} km)")
    seam = r.qc["seam"]
    rows = np.flatnonzero(seam.any(axis=1))
    i0 = int(rows.max())
    cols = [j for j in range(5, 95) if abs(j * CELL - fx) > 3 * CELL]
    jump = np.abs(r.z[i0 + 1, cols] - r.z[i0, cols] - (T[i0 + 1, cols] - T[i0, cols]))
    check(float(np.max(jump)) < 0.6, f"42n: no step at the seam - merged surface continuous (max {np.max(jump):.2f} m "
          "beyond the true dip)")
    ov = r.qc["overlap"]
    res_after = np.abs(r.z - za)[ov & ~near_f & (Y < 2900)]
    check(float(np.nanmedian(res_after)) < 0.05, f"42n: overlap residual ~0 against the priority survey "
          f"(median {np.nanmedian(res_after):.3f} m)")
    for yy in (3200.0, 4500.0):
        i = int(round(yy / CELL))
        jl, jr = int(fx // CELL) - 3, int(fx // CELL) + 4
        thr = (r.z[i, jr] - r.z[i, jl]) - (T[i, jr] - T[i, jl]) + 40.0
        check(abs(thr - 40.0) < 1.5, f"42h/n: fault throw preserved across the seam zone at y={yy:g} m "
              f"({thr:.1f} m vs 40 m)")
    blocks = r.qc["blocks"]
    check(len(np.unique(blocks[blocks >= 0])) == 2, "42h: the fault splits the map into two blocks (barrier)")
    # 42e units
    check(abs(SB.slope_per_m(2.0, "m/km") - 0.002) < 1e-12 and abs(SB.slope_per_m(10.56, "ft/mile") - 0.002) < 1e-6,
          "42e: allowed slope in m/km or ft/mile")
    # 42n spike robustness
    g2, X2, Y2, T2, za2, zb2, *_ = scene(spike=True)
    r2 = SB.blend(g2, [SB.Survey("A", za2, 1), SB.Survey("B", zb2, 2)], slope=s, method="hybrid", w_min=500.0,
                  w_max=12000.0, faults=faults)
    check(np.max(np.abs(r2.width - r.width)) < 200.0, f"42c/n: a picking spike does not widen the blend "
          f"(max width change {np.max(np.abs(r2.width - r.width)):.0f} m)")
    # 42g width > overlap -> residual surface in the feather method
    rf = SB.blend(g, [A, B], slope=s, method="feather", w_min=500.0, w_max=12000.0, faults=faults)
    tw = rf.qc["too_wide"]
    check(tw.any() and np.abs(rf.qc["corr_b"]).max() > 1.0,
          f"42g: width larger than the 1 km overlap on {100 * tw.mean():.0f} % of the seam -> residual surface into "
          "the lower-priority survey")
    small = SB.blend(g, [A, B], slope=50.0, method="feather", w_min=200.0, w_max=900.0, faults=faults)
    check(not small.qc["too_wide"].any() and np.nanmin(small.qc["weight"]) >= 0 and np.nanmax(small.qc["weight"]) <= 1,
          "42f: distance-weighted feather inside the overlap when it is wide enough (weights 0..1)")
    rr = SB.blend(g, [A, B], slope=s, method="residual", w_min=500.0, w_max=12000.0, faults=faults)
    check(np.isfinite(rr.z).sum() == np.isfinite(np.where(np.isfinite(za), za, zb)).sum(),
          "42f: residual-surface method covers both surveys")
    # 42n priority cut reproduced exactly with blending off
    rc = SB.blend(g, [A, B], method="cut")
    cut = np.where(np.isfinite(za), za, zb)
    check(np.array_equal(np.nan_to_num(rc.z, nan=-9), np.nan_to_num(cut, nan=-9)) and rc.readout.startswith("Priority"),
          "42f/n: priority cut reproduced exactly when blending is off")
    # 42i wells inside and outside the blend zone tie within 0.1 m
    wells = [("W-A", 1200.0, 1500.0, float(SB.bilinear(T, 1500 / CELL, 1200 / CELL))),
             ("W-blend", 2600.0, 3600.0, float(SB.bilinear(T, 3600 / CELL, 2600 / CELL))),
             ("W-far", 1000.0, 10500.0, float(SB.bilinear(T, 10500 / CELL, 1000 / CELL)))]
    rw = SB.blend(g, [A, B], slope=s, method="hybrid", w_min=500.0, w_max=12000.0, faults=faults, wells=wells)
    ties = rw.qc["well_ties"]
    check(len(ties) == 3 and max(abs(t[5]) for t in ties) <= 0.1 and abs(ties[1][4]) > 0.1,
          f"42i/n: every top ties within 0.1 m after blending (before: "
          f"{', '.join(f'{t[0]} {t[4]:+.2f}' for t in ties)} m)")
    # 42l QC outputs
    q = rw.qc
    need = {"residual_before", "residual_after", "weight", "width", "added_slope", "seam_gradient", "histogram",
            "well_ties"}
    check(need <= set(q) and np.nanmax(np.abs(q["residual_before"])) > 5 and "Blend width" in rw.readout
          and "% of seam" in rw.readout, f"42l/m: QC maps + readout ('{rw.readout}')")
    # 42j blend the picks then grid once ~ grid-then-blend
    jj = np.arange(5, 95, 7)
    ii = np.arange(int(3100 / CELL), int(5000 / CELL), 9)
    I, J = np.meshgrid(ii, jj, indexing="ij")
    px, py = J.ravel() * CELL, I.ravel() * CELL
    pz = zb[I.ravel(), J.ravel()]
    pc = SB.correct_points(g, r, px, py, pz, "B")
    zz = r.z[I.ravel(), J.ravel()]
    check(np.nanmax(np.abs(pc - zz)) < 0.6, f"42j: corrected picks = blended grid at the picks (max "
          f"{np.nanmax(np.abs(pc - zz)):.2f} m) - blend the picks, then grid once")
    # 42k dynamic update: a moved top re-blends only that horizon
    bl = SB.Blender(g, slope=s, method="hybrid", w_min=500.0, w_max=12000.0, faults=faults)
    r_top = bl.add("Top", za, zb)
    r_base = bl.add("Base", za - 300.0, zb - 300.0)
    zb_m = zb.copy()
    zb_m[(X < 1500) & (Y > 2000)] += 3.0  # an edit in the west of B
    r_top2, changed, msg = bl.update("Top", zb=zb_m)
    check(bl.results["Base"] is r_base and r_top2 is not r_top and 0 < changed < len(r_top.residual)
          and "Top" in msg, f"42k: a moved pick re-blends only that horizon and only part of the seam ({msg})")
    # gap flag: B does not continue beyond A in the east
    zb_g = np.where(X < 3000, zb, np.where(Y <= 3000, zb, np.nan))
    rg = SB.blend(g, [A, SB.Survey("B", zb_g, 2)], slope=s, method="hybrid", w_min=500.0, w_max=12000.0)
    check(rg.qc["gap"].any() and "gap" in rg.readout, "42g: seam stretches with no overlap flagged as a gap")
    # time domain: ms/km
    rt = SB.blend(g, [A, B], slope=3.0, unit="ms/km", method="hybrid", w_min=500.0, w_max=12000.0)
    check(rt.width.max() < r.width.max(), "42e: time-domain blending with ms/km (a larger tolerance = narrower blend)")
    # resampling a survey grid onto the master grid
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "blend")
    vol = proj.volume("Synth")
    mg = SB.MasterGrid.covering([vol])
    hz = np.asarray(proj.horizons.get("Top").data["3D:Synth"], float)
    zr = mg.resample(vol, hz)
    x, y = vol.xy(float(vol.inlines[5]), float(vol.xlines[7]))
    fi, fj = mg.index(x, y)
    check(abs(float(SB.bilinear(zr, fi, fj)) - hz[5, 7]) < 0.5 and np.isnan(zr).any() is not None,
          "42a: survey grid resampled onto the common master bin grid")


def run_gui(check, tmp):
    """Qt mock / real Qt offscreen: Interpret > Survey horizon blending on two overlapping project surveys."""
    os.makedirs(tmp, exist_ok=True)
    from .gui.main_window import MainWindow
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "blendgui")
    v = proj.volumes[0]
    proj.volumes.append(dict(v, name="Synth B"))
    h = proj.horizons.get("Top")
    g = np.asarray(h.data["3D:Synth"], float)
    n = g.shape[0]
    ga = g.copy()
    ga[int(0.65 * n):] = np.nan
    gb = g + 6.0
    gb[:int(0.35 * n)] = np.nan
    h.data["3D:Synth"] = ga
    h.data["3D:Synth B"] = gb
    mw = MainWindow(None)
    mw.project = proj
    dlg = mw.survey_blend()
    grid, za, zb = dlg.load_inputs("Top", "Synth", "Synth B", "Time")
    check(np.isfinite(za).any() and np.isfinite(zb).any() and (np.isfinite(za) & np.isfinite(zb)).any(),
          "42a GUI: horizon of two surveys resampled to the master grid (overlap found)")
    r = dlg.preview(method="hybrid", slope=4.0, unit="ms/km", w_min=50.0, w_max=5000.0, trim=0)
    check(r is not None and "Blend width" in r.readout and np.isfinite(r.z).sum() >= np.isfinite(r.cut).sum() - 5,
          f"42m GUI: preview + readout ({r.readout if r else ''})")
    ev = mw.mapbus.last("data", "grids")
    check(ev is not None and "Top (merged)" in ev["ids"], "42m GUI: merged grid announced on the Live Map bus")
    rc = dlg.preview(method="cut")
    check(np.array_equal(np.nan_to_num(rc.z, nan=-1), np.nan_to_num(rc.cut, nan=-1)), "42m GUI: priority cut option")
    dlg.preview(method="hybrid", slope=4.0, unit="ms/km", w_min=50.0, w_max=5000.0, trim=0)
    files = dlg.export(os.path.join(tmp, "blend_out"))
    check(len(files) >= 4 and all(os.path.getsize(f) > 100 for f in files),
          f"42m GUI: export merged grid + QC maps as ZMAP+ ({len(files)} files) and map-hub layers")
