"""Tests for carto.py (PLAN 21f)."""
import math

import numpy as np


def run(check, tmp):
    from . import carto as C
    # hillshade: plane rising to the east lit from the west (azimuth 270) = analytic Lambert value
    x = np.arange(60) * 10.0
    X, Y = np.meshgrid(x, x)
    z = 0.5 * X
    hs = C.hillshade(z, 10.0, 10.0, azimuths=(270.0,), altitude=45.0)
    s = math.atan(0.5)
    want = math.sin(math.radians(45)) * math.cos(s) + math.cos(math.radians(45)) * math.sin(s)
    check(abs(hs[30, 30] - want) < 1e-9 and np.ptp(hs[5:-5, 5:-5]) < 1e-9, f"21f: hillshade of a plane = analytic ({hs[30, 30]:.4f})")
    hs_e = C.hillshade(z, 10.0, 10.0, azimuths=(90.0,), altitude=45.0)
    check(hs_e[30, 30] < hs[30, 30], "21f: slope facing away from the light is darker")
    dome = 100 * np.exp(-((X - 300) ** 2 + (Y - 300) ** 2) / 150.0 ** 2)
    hd = C.hillshade(dome, 10.0, 10.0, azimuths=(315.0,))
    check(hd[40, 20] > hd[20, 40], "21f: dome lit from the NW: NW flank brighter than SE flank")
    hv = C.hillshade(dome, 10.0, 10.0, azimuths=(315.0,), ve=3.0)
    check(np.ptp(hv[np.isfinite(hv)]) > np.ptp(hd[np.isfinite(hd)]), "21f: vertical exaggeration increases contrast")
    lut = C.ramp_lut("depth")
    check(lut.shape == (256, 4) and np.all(C.ramp_lut("depth", True)[0] == lut[-1]), "21f: colour ramps (256 x RGBA, reversible)")
    check(C.default_ramp("m", "Top depth")[0] == "depth" and C.default_ramp("", "LOO residual")[0] == "diverging"
          and C.default_ramp("m", "net pay isochore")[0] == "thickness", "21f: default ramp by grid type")
    rgba = C.shaded_rgba(dome, lut, shade=hd, strength=0.6)
    check(rgba.shape == dome.shape + (4,) and rgba[..., 3].min() == 255, "21f: shaded colour fill")
    # contours chained + labels without overlap
    g = {"z": np.hypot(X - 300, Y - 300), "x0": 0.0, "y0": 0.0, "dx": 10.0, "dy": 10.0}
    pl = C.contour_polylines(g, [100.0, 150.0, 200.0])
    one = pl[100.0]
    check(len(one) == 1 and math.hypot(one[0][0][0] - one[0][0][-1], one[0][1][0] - one[0][1][-1]) < 1e-6,
          "21f: marching-squares segments chained into one closed polyline per ring")
    r = np.hypot(one[0][0] - 300, one[0][1] - 300)
    check(np.max(np.abs(r - 100)) < 1.0, "21f: contour polyline on the level")
    cands = []
    for lev, lines in pl.items():
        cands += C.label_candidates(lines, f"{lev:g}", spacing=40.0)
    kept = C.place_labels(cands, char_w=12.0, char_h=14.0)
    boxes = [C.label_box(c[0], c[1], c[2], c[3], 12.0, 14.0) for c in kept]
    check(len(cands) > len(kept) > 5 and not C.boxes_overlap(boxes), f"21f: {len(kept)} of {len(cands)} contour labels "
          "kept, no overlapping label boxes")
    check(all(-90 <= c[2] <= 90 for c in kept), "21f: labels rotated with the contour, kept readable")
    # contours never cross a fault polygon
    plane = {"z": 0.3 * X + 0.1 * Y, "x0": 0.0, "y0": 0.0, "dx": 10.0, "dy": 10.0}
    poly = (np.array([200.0, 260.0, 300.0, 240.0]), np.array([100.0, 100.0, 450.0, 450.0]))
    gb = C.blank_polygons(plane, [poly])
    lines = C.contour_polylines(gb, list(np.arange(10, 200, 10.0)))
    crosses, inside = 0, 0
    px, py = np.append(poly[0], poly[0][0]), np.append(poly[1], poly[1][0])
    for lev, ls in lines.items():
        for lx, ly in ls:
            inside += int(C.point_in_polygon(lx, ly, *poly).sum())
            for k in range(len(px) - 1):
                crosses += int(C._seg_cross(lx[:-1], ly[:-1], lx[1:], ly[1:], px[k], py[k], px[k + 1], py[k + 1]).sum())
    check(crosses == 0 and inside == 0 and sum(len(v) for v in lines.values()) > 10, "21f: contours stop at the fault polygon "
          "(no vertex inside, no segment crossing)")
    # fault blocks: wells only influence their own side
    fault = (np.array([310.0, 290.0]), np.array([-10.0, 600.0]))
    lab = C.fault_blocks(60, 60, 0.0, 0.0, 10.0, 10.0, [fault])
    check(len(np.unique(lab)) == 2, "21f: a fault across the map makes two blocks")

    def idw(Xn, Yn, px_, py_, v_):
        w = 1.0 / np.maximum((Xn[..., None] - px_) ** 2 + (Yn[..., None] - py_) ** 2, 1.0)
        return (w * v_).sum(-1) / w.sum(-1)

    gg = C.grid_by_blocks([100.0, 500.0], [300.0, 300.0], [1000.0, 1200.0], {"z": np.zeros((60, 60)), "x0": 0.0, "y0": 0.0,
                                                                                 "dx": 10.0, "dy": 10.0}, [fault], idw)
    check(np.allclose(gg["z"][:, :25], 1000.0) and np.allclose(gg["z"][:, 35:], 1200.0), "21f: gridding per fault block "
          "(no smearing across the fault)")
    tx, ty = C.fault_ticks([0.0, 1000.0], [0.0, 0.0], side="right", spacing=200.0, length=50.0)
    check(len(tx) == 10 and np.all(ty[1::2] < 0) and np.allclose(ty[0::2], 0.0), "21f: downthrown ticks on the right-hand side")
    # spill point: cone (deepening outwards) with a shallow saddle channel to the east edge
    n = 101
    I, J = np.mgrid[0:n, 0:n]
    rr = np.hypot(I - 50, J - 50)
    depth = 1000.0 + rr
    strip = (np.abs(I - 50) <= 1) & (J > 70)
    depth[strip] = 1020.0 + 0.1 * (J[strip] - 70)
    sp = C.spill_point(depth)
    check(abs(sp["spill_depth"] - (1020 + 0.1 * 30)) < 0.05 and sp["spill_node"][1] >= 99 and abs(sp["relief"] - 23.0) < 0.05,
          f"21f: spill point at the saddle channel ({sp['spill_depth']:.2f} m, node {sp['spill_node']})")
    check(sp["mask"][50, 50] and not sp["mask"][0, 0] and sp["mask"].sum() > 1000, "21f: closed area at spill")
    check(C.classify_status("Gas condensate discovery, P&A") == "oil+gas" or C.classify_status("Gas condensate discovery, P&A")
          in ("oil", "gas", "oil+gas"), "21f: status -> symbol (hydrocarbon P&A keeps its fluid)")
    check(C.classify_status("Dry hole") == "dry" and C.classify_status("Plugged and abandoned") == "p&a"
          and C.classify_status("Oil producer") == "oil" and C.classify_status("Planned location") == "planned",
          "21f: well status classification")
    check(len(C.symbol_paths("gas")["lines"]) == 9 and C.symbol_paths("oil")["fill"] == "full", "21f: well symbols")
    sc = C.nice_scale(37000.0)
    check(sc["length"] == 10000 and sc["label"] == "10 km", f"21f: scale bar {sc}")
    check(abs(C.north_arrow_angle(-40, 173, 173)) < 1e-12 and C.north_arrow_angle(-40, 175, 173) < 0,
          "21f: grid convergence (0 on the central meridian, sign by side)")
    sm = C.styled_map({"z": 1000 + np.hypot(X - 300, Y - 300), "x0": 0.0, "y0": 0.0, "dx": 10.0, "dy": 10.0}, ci=50.0)
    boxes = [C.label_box(x_, y_, a_, t_, 600 / 120.0, 1.4 * 600 / 120.0) for x_, y_, a_, t_ in sm["labels"]]
    check(sm["rgba"].shape == (60, 60, 4) and 1050.0 in sm["levels"] and sm["labels"] and not C.boxes_overlap(boxes),
          "21f: styled map for the Well Correlation map (shaded fill, contours, non-overlapping labels)")
