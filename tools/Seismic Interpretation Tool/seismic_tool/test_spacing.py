"""Tests for PLAN 29: public-data readers, spacing, parent / child, drainage ellipses, gun barrel."""
import math
import os

import numpy as np


def run(check, tmp):
    from . import spacing as SP
    # ---- 29a readers
    p1 = os.path.join(tmp, "rrc.csv")
    with open(p1, "w") as fh:
        fh.write("API_NO,LEASE_NAME,OPERATOR_NAME,FIRST_PROD_DATE,SURFACE_LAT,SURFACE_LONG,LATERAL_LENGTH,FORMATION\n"
                 "4238912345,SMITH 1H,ACME,03/15/2019,31.5,-103.2,\"10,250\",WOLFCAMP A\n")
    p2 = os.path.join(tmp, "fracfocus.csv")
    with open(p2, "w") as fh:
        fh.write("APINumber,JobStartDate,TotalBaseWaterVolume,Latitude,Longitude,OperatorName\n"
                 "33053012340000,1/2/2020 12:00:00 AM,12500000,47.8,-103.4,XYZ OIL\n")
    r1 = SP.read_public_csv(p1, "RRC")[0]
    r2 = SP.read_public_csv(p2)[0]
    check(r1["api"] == "4238912345" and r1["name"] == "SMITH 1H" and r1["first_prod"] == "2019-03-15" and
          r1["lateral"] == 10250.0 and r1["zone"] == "WOLFCAMP A" and r1["lat"] == 31.5,
          "29a: Texas RRC style CSV mapped to the common schema")
    check(r2["api"] == "33053012340000" and r2["first_prod"] == "2020-01-02" and r2["fluid"] == 12500000.0 and
          r2["operator"] == "XYZ OIL", "29a: FracFocus CSV mapped (job start -> first_prod, base water -> fluid)")
    wsm = [{"x": 0, "y": 0, "azi": 175.0, "quality": "A"}, {"x": 1000, "y": 0, "azi": 5.0, "quality": "B"},
           {"x": 500, "y": 0, "azi": 90.0, "quality": "E"}]
    az = SP.shmax_from_wsm(wsm, 500, 0)
    check(min(az, 180 - az) < 1.0, f"29a: World Stress Map axial mean of 175 and 5 deg = 0 deg ({az:.1f}); E quality ignored")
    # ---- 29b spacing
    trajs, info = SP.synthetic_field(n_pads=3, wells_per_pad=4, spacing=200.0, azimuth=0.0)
    zones = {n: v["zone"] for n, v in info.items()}
    rows = SP.well_distances(trajs, zones)
    r12 = next(r for r in rows if {r["a"], r["b"]} == {"P1-1", "P1-2"})
    r13 = next(r for r in rows if {r["a"], r["b"]} == {"P1-1", "P1-3"})
    check(abs(r12["horizontal"] - 200) < 1e-6 and abs(abs(r12["vertical"]) - 60) < 15 and not r12["same_zone"] and
          abs(r13["horizontal"] - 400) < 1e-6 and r13["same_zone"] and abs(r12["overlap"] - 2500) < 60,
          "29b: inter-well distances (200 m wine-rack neighbours 60 m apart vertically, 400 m in the same zone)")
    nb = SP.nearest_neighbours(rows, "P2-2")
    check([q[0] for q in nb[:2]] in (["P2-1", "P2-3"], ["P2-3", "P2-1"]), "29b: nearest neighbours")
    pc = SP.classify_parent_child({n: v["first_prod"] for n, v in info.items()}, rows, max_dist=450.0)
    check(pc["P1-4"] == "parent" and pc["P2-1"] == "child" and pc["P1-1"] == "co-completed" and pc["P3-4"] in
          ("co-completed", "child"), f"29b: parent / child / co-completed ({pc['P1-4']}, {pc['P2-1']}, {pc['P1-1']})")
    sec = [{"name": "S1", "x": [-100, 700, 700, -100], "y": [0, 0, 3000, 3000]},
           {"name": "S2", "x": [700, 1700, 1700, 700], "y": [0, 0, 3000, 3000]}]
    wp = SP.wells_in_polygons(trajs, sec)
    check(len(wp["S1"]) == 4 and len(wp["S2"]) == 5, "29b: wells per section / DSU")
    # ---- 29c ellipses
    e = SP.drainage_ellipse(trajs[0], 45.0, half_length=150.0, width=50.0)
    xs, ys = SP.ellipse_polygon(e)
    k = int(np.argmax(np.hypot(xs - e["cx"], ys - e["cy"])))
    ang = math.degrees(math.atan2(xs[k] - e["cx"], ys[k] - e["cy"])) % 180.0
    check(abs(ang - 45.0) < 3.0 and e["az"] == 45.0, f"29c: ellipse major axis follows SHmax ({ang:.1f} deg)")
    e0 = SP.drainage_ellipse(trajs[0], 90.0, half_length=150.0, width=50.0)  # SHmax across the N-S lateral
    check(abs(e0["a"] - 150.0) < 1e-6 and abs(e0["b"] - (1250 + 50)) < 1e-6, "29c: lateral across SHmax: ellipse = "
          "frac half-length along SHmax, lateral length + width across")
    ea = SP.drainage_ellipse(trajs[0], 90.0, 150.0, 50.0)
    eb = SP.drainage_ellipse(trajs[1], 90.0, 150.0, 50.0)
    ov = SP.ellipse_overlap(ea, eb)
    far = SP.ellipse_overlap(ea, SP.drainage_ellipse(trajs[3], 90.0, 150.0, 50.0))
    check(ov > 0 and far == 0, f"29c: overlapping drainage between 200 m neighbours ({ov / 1e4:.1f} ha), none at 600 m")
    # ---- 29d gun barrel
    gb = SP.gun_barrel(trajs, ellipses=[dict(ea, name="P1-1")], zones=zones)
    offs = [w["offset"] for w in gb["wells"]]
    check(np.allclose(np.diff(offs), 200.0) and abs(gb["az_section"] - 90.0) < 1e-6 and
          {w["tvd"] > 2530 for w in gb["wells"] if w["zone"] == "B"} == {True} and
          abs(gb["ellipses"][0]["half_width"] - 150.0) < 1e-6,
          "29d: gun-barrel offsets match the 200 m pattern, zones by TVD, ellipse half-width across the section")


def run_gui(check, tmp):
    from . import spacing as SP
    from .gui.dashboard_window import DashboardWindow
    trajs, info = SP.synthetic_field(n_pads=2, wells_per_pad=3, spacing=250.0)
    win = DashboardWindow(None, None)
    win.set_trajectories(trajs)
    res = win.spacing(45.0, first_prod={n: v["first_prod"] for n, v in info.items()},
                      zones={n: v["zone"] for n, v in info.items()})
    gb = win.gun_barrel_view()
    check(len(res["ellipses"]) == 6 and len(win._ellipse_items) == 6 and abs(res["per_well"]["P1-2"]["Nearest_m"] - 250) < 1e-6
          and res["per_well"]["P2-1"]["Class"] == "child" and len(gb["wells"]) == 6 and len(gb["ellipses"]) == 6,
          "29 GUI: dashboard spacing (ellipses on the map, nearest distances, parent / child) + gun-barrel window")
