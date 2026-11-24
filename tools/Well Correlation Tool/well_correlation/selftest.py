"""Self-test of the data layer with synthetic LAS files (no GUI, no real data needed)."""
import os
import shutil
import tempfile

import numpy as np


def _write_las(path, well, depth, curves, units, null=-999.25, depth_unit="M"):
    with open(path, "w") as f:
        f.write("~Version\n VERS. 2.0 : test\n WRAP. NO :\n~Well\n")
        f.write(f" STRT.{depth_unit} {depth[0]} :\n STOP.{depth_unit} {depth[-1]} :\n NULL. {null} :\n WELL. {well} : WELL\n")
        f.write("~Curve\n DEPT.%s : depth\n" % depth_unit)
        for n in curves:
            f.write(f" {n}.{units.get(n, '')} : {n}\n")
        f.write("~A\n")
        arr = np.column_stack([depth] + [np.where(np.isfinite(v), v, null) for v in curves.values()])
        for row in arr:
            f.write(" ".join(f"{x:.4f}" for x in row) + "\n")


def run():
    from .project import Project
    from .geo import tm_forward, min_curvature
    from .section import SectionLayout
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok &= bool(cond)

    # NZTM: point on the central meridian at 41S
    e, n = tm_forward(-41.0, 173.0)
    check(abs(e - 1600000) < 0.01 and abs(n - 5461242.9) < 1.0, f"NZTM projection ({e:.1f}, {n:.1f})")
    # min curvature: straight 30 deg hole
    tvd, dn, de = min_curvature([0, 1000], [30, 30], [90, 90])
    check(abs(tvd[-1] - 1000 * np.cos(np.radians(30))) < 0.01 and abs(de[-1] - 500) < 0.01, "minimum curvature")

    from .curves import remove_flatlines
    zz = np.arange(0, 20, 0.1524)
    vv = np.sin(zz)
    vv[30:80] = 2.65  # 7.6 m stuck value
    r = remove_flatlines(zz, vv)
    check(np.isnan(r[30:80]).all() and np.isfinite(r[:29]).all(), "flat-lined (cased-hole) stretch removed")

    tmp = tempfile.mkdtemp(prefix="wc_selftest_")
    try:
        rng = np.random.default_rng(1)
        for name, off, elev in (("Alpha-1", 0.0, 25.0), ("Beta-1", 40.0, 10.0)):
            d = os.path.join(tmp, "wells", name, "las")
            os.makedirs(d)
            z = np.arange(1000, 2000, 0.1524)
            gr = 90 + 10 * rng.standard_normal(z.size)
            sand = (z > 1500 + off) & (z < 1530 + off)
            gr[sand] = 30 + 3 * rng.standard_normal(sand.sum())  # a sand
            res = np.where(gr < 60, 50.0, 2.0) * (1 + 0.05 * rng.standard_normal(z.size))
            # run 1 (upper, feet!) and run 2 (lower) + a short repeat pass with the same mnemonic
            up = z < 1500
            _write_las(os.path.join(d, "run1.las"), name, z[up] / 0.3048, {"GR": gr[up], "ILD": res[up]}, {"GR": "API"}, depth_unit="F")
            lo = z >= 1490
            rep = (z > 1600) & (z < 1650)
            _write_las(os.path.join(d, "run2.las"), name, z[lo], {"GR": gr[lo], "LLD": res[lo], "NPHI": 25.0 + rng.standard_normal(lo.sum())},
                       {"NPHI": "%"})
            _write_las(os.path.join(d, "repeat.las"), name, z[rep], {"GR": gr[rep] + 500}, {})
        with open(os.path.join(tmp, "wells_header.csv"), "w") as f:
            f.write("Well,UWI,Latitude,Longitude,Elevation_m,Elev_Ref,TD_m\n")
            f.write("Alpha-1,1,-39.60,173.30,25,KB,2000\nBeta-1,2,-39.62,173.35,10,DF,2000\n")
        p = Project.from_folder(tmp)
        check(set(p.wells) == {"Alpha-1", "Beta-1"}, "folder import finds wells")
        path = os.path.join(tmp, "test.wcproj.json")
        p.sections["Section 1"] = ["Alpha-1", "Beta-1"]
        p.save(path)
        p = Project.open(path)
        w = p.load_well("Alpha-1")
        check(w.depth[0] < 1001 and w.depth[-1] > 1998, f"feet converted, splice covers {w.depth[0]:.0f}-{w.depth[-1]:.0f} m")
        check("RDEEP" in w.logs and "GR" in w.logs, "aliases: ILD/LLD -> RDEEP")
        check(abs(np.nanmedian(w.logs["NPHI"]) - 0.25) < 0.01, "NPHI % -> v/v")
        i = np.searchsorted(w.depth, 1620)
        check(w.logs["GR"][i] < 300, "repeat pass did not overwrite the main pass")
        check(abs(float(w.md_to_tvdss([1500])[0]) - 1475) < 1e-6, "TVDSS = MD - elevation (vertical well)")
        p2 = Project.open(path)
        w2 = p2.load_well("Alpha-1")
        check(np.allclose(np.nan_to_num(w2.logs["GR"]), np.nan_to_num(w.logs["GR"])), "cache round trip")
        p.formations = [{"name": "SandTop", "color": "#ff0000"}]
        p.tops = {"Alpha-1": {"SandTop": 1500.0}, "Beta-1": {"SandTop": 1540.0}}
        lay = SectionLayout(p, ["Alpha-1", "Beta-1"], {"hang": "FLAT:SandTop", "spacing": "proportional"})
        tm = lay.top_marks()[0]
        check(all(abs(y) < 1e-6 for _, y, _ in tm["marks"]), "flatten puts the datum at 0")
        check(abs(lay.md_at(p.wells["Beta-1"], 0.0) - 1540.0) < 0.2, "cursor depth inverse")
        from . import tops as T
        inv = T.set_top(p, "Alpha-1", "SandTop", 1510.0)
        check(p.tops["Alpha-1"]["SandTop"] == 1510.0 and p.tops_meta["Alpha-1"]["SandTop"]["source"] == "user", "set top")
        T.apply_edit(p, inv)
        check(p.tops["Alpha-1"]["SandTop"] == 1500.0, "undo top edit")
        T.rename_formation(p, "SandTop", "Sand A")
        check("Sand A" in p.tops["Beta-1"] and p.formations[0]["name"] == "Sand A", "rename formation carries tops")
        csvp = os.path.join(tmp, "tops.csv")
        T.export_csv(p, csvp)
        p.tops = {}
        n, unknown, _ = T.import_csv(p, csvp)
        check(n == 2 and p.tops["Beta-1"]["Sand A"] == 1540.0, "tops CSV export / import round trip")
        from .autocorr import suggest_tops
        saved = p.tops["Beta-1"].pop("Sand A")
        r = suggest_tops(p, "Beta-1", ["Alpha-1"], formations=["Sand A"])
        check("Sand A" in r and abs(r["Sand A"][0] - saved) < 3.0, f"suggest pick finds the sand top ({r.get('Sand A', (0,))[0]:.1f} vs {saved})")
        p.tops["Beta-1"]["Sand A"] = saved
        from .zones import zone_table
        p.formations.append({"name": "Base", "color": "#000000"})
        p.tops["Alpha-1"]["Base"] = 1600.0
        zt = zone_table(p, ["Alpha-1"])
        check(len(zt) == 1 and 25 <= zt[0]["Net_sand_m"] <= 35, f"zone stats net sand ({zt[0]['Net_sand_m'] if zt else '-'} m, expect ~30)")
        from . import mapping as M
        pts = [("A", 0.0, 0.0, 100.0), ("B", 3000.0, 0.0, 130.0), ("C", 0.0, 3000.0, 110.0), ("D", 3000.0, 3000.0, 160.0)]
        g = M.grid_points(pts, "tps")
        errs = [abs(g["z"][int(round((y - g["y0"]) / g["dy"])), int(round((x - g["x0"]) / g["dx"]))] - v) for _, x, y, v in pts]
        check(max(errs) < 2.0, f"spline map honours wells (max error {max(errs):.2f})")
        segs = M.contour_segments(g, [120.0])
        check(120.0 in segs and len(segs[120.0][0]) > 10, "contours traced")
        gi = M.grid_points(pts, "idw")
        check(np.nanmin(gi["z"]) >= 100 - 1e-6 and np.nanmax(gi["z"]) <= 160 + 1e-6, "IDW map stays within the data range")
        from .tops import compare_with_reference, add_reference_from_current
        add_reference_from_current(p, "snap")
        p.tops["Beta-1"]["Sand A"] += 15
        flags = {(r["Well"], r["Formation"]): r["Flag"] for r in compare_with_reference(p, "snap")}
        check(flags.get(("Beta-1", "Sand A")) == "moved" and flags.get(("Alpha-1", "Sand A")) == "ok", "reference comparison flags moved tops")
        from .lithology import well_lithology
        v, code = well_lithology(w)
        i = np.searchsorted(w.depth, 1515)
        check(code[i] == 0, "sand classified as sand")
        try:
            from .export import export_section
            export_section(p, ["Alpha-1", "Beta-1"], os.path.join(tmp, "s.png"))
            check(os.path.getsize(os.path.join(tmp, "s.png")) > 1000, "PNG export")
        except ImportError:
            print("SKIP PNG export (matplotlib not installed)")
        # ---- DLIS reader (synthetic file + real Tui-1 files when present)
        from . import test_dlis
        test_dlis.run(check, tmp)
        # ---- petrophysics (PLAN 20)
        from .petro import test_petro
        test_petro.run_qc(check, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    return ok
