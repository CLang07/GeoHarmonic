"""Tests for trackstyle.py (Seismic tool PLAN 40h: lithology patterns, gradient fills, pay flags, crossover,
track templates) - pure numpy + the section layout on the bundled demo when available."""
import numpy as np


def run(check, tmp):
    from . import trackstyle as TS
    lib = TS.library()
    check({"Sand", "Shale", "Limestone", "Dolomite", "Coal"} <= set(lib), "40h: lithology pattern library "
          "(sand, shale, limestone, dolomite, coal ...)")
    tiles = {n: TS.pattern_tile(lib[n], 16) for n in ("Sand", "Shale", "Limestone", "Dolomite", "Coal")}
    inked = {n: int(np.any(t[:, :, :3] != np.array(TS.hex_rgba(lib[n]["colour"]))[:3], axis=2).sum())
             for n, t in tiles.items()}
    check(all(t.shape == (16, 16, 4) for t in tiles.values()) and all(inked[n] > 0 for n in ("Sand", "Shale",
          "Limestone", "Dolomite")) and inked["Coal"] == 0 and len({tiles[n].tobytes() for n in tiles}) == 5,
          f"40h: patterns render, each different (ink pixels {inked})")
    codes = np.array([0] * 20 + [2] * 20 + [3] * 10 + [4] * 10)
    col = TS.pattern_column(codes, 24, lib)
    check(col.shape == (60, 24, 4) and (col[45, :, :3] == TS.hex_rgba(lib["Coal"]["colour"])[:3]).all()
          and col[:20, :, 3].min() == 255, "40h: lithology column drawn with the pattern of each code")

    class P:
        display = {}
    TS.set_pattern(P, "Sand", colour="#ffee00", kind="dashes", spacing=5)
    check(TS.library(P)["Sand"]["kind"] == "dashes" and TS.library(P)["Sand"]["colour"] == "#ffee00"
          and TS.library()["Sand"]["kind"] == "dots", "40h: pattern library is editable per project")
    try:
        TS.set_pattern(P, "Sand", kind="zigzag")
        bad = False
    except ValueError:
        bad = True
    check(bad, "40h: unknown pattern kinds rejected")
    gr = np.linspace(0, 150, 101)
    lut = TS.gr_lut()
    img = TS.curve_fill_image(gr, lut, 0, 150, 40, x=gr / 150.0, side="left")
    check(img.shape == (101, 40, 4) and img[0, :, 3].max() == 0 and img[100, :, 3].min() == 255
          and (img[50, :20, 3] == 255).all() and (img[50, 21:, 3] == 0).all()
          and not np.array_equal(img[30, 0, :3], img[90, 0, :3]),
          "40h: GR gradient fill - coloured by value, filled from the track edge to the curve")
    d = np.arange(1000.0, 1100.0, 0.5)
    vsh = np.where((d >= 1020) & (d < 1040), 0.2, 0.7)
    phi = np.where((d >= 1020) & (d < 1040), 0.2, 0.05)
    sw = np.where((d >= 1020) & (d < 1030), 0.3, 0.9)
    r = TS.net_pay(d, vsh, phi, sw, 0.4, 0.08, 0.6)
    check(abs(r["net_m"] - 20.0) < 1.0 and abs(r["pay_m"] - 10.0) < 1.0 and r["intervals_pay"][0][0] == 1020.0
          and len(r["intervals_net"]) == 1, f"40h: net / pay flags from cutoffs (net {r['net_m']:.1f} m, pay "
          f"{r['pay_m']:.1f} m)")
    rhob = np.array([2.65, 2.20, 2.45])
    nphi = np.array([0.0, 0.10, 0.35])
    x = TS.crossover(rhob, nphi)
    check(list(x["gas"]) == [False, True, False] and list(x["shale"]) == [False, False, True],
          "40h: neutron-density crossover: gas where density porosity > neutron, shale separation the other way")
    P.display = {}
    TS.save_template(P, "Reservoir", dict(TS.default_template(), tracks=["LITH", "GR", "DN"], crossover=True))
    t = TS.apply_template(P, "Reservoir")
    check(P.display["tracks"] == ["LITH", "GR", "DN"] and P.display["crossover"] and t["gr_fill"] == "gradient",
          "40h: track styles saved as templates and applied")
