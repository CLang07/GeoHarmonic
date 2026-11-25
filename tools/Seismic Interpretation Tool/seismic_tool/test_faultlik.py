"""Automated fault mapping tests (PLAN 32): synthetic planar normal faults of known strike / dip / throw, a crossing
conjugate pair and noise."""
import math
import os

import numpy as np

from . import faultlik as FL
from . import synth

FAULTS = [dict(i0=24, j0=24, k0=80, az=30, slope=8.0, throw=6),     # conjugate pair crossing at the centre
          dict(i0=24, j0=24, k0=80, az=210, slope=8.0, throw=5),
          dict(i0=12, j0=36, k0=70, az=120, slope=10.0, throw=7)]  # a third fault, other strike
EI, EJ = np.array([25.0, 0.0]), np.array([0.0, 25.0])
DT = 4.0
_CACHE = {}


def case():
    if "c" not in _CACHE:
        d = synth.planar_faults_cube(nil=56, nxl=56, ns=180, faults=FAULTS)
        r = FL.fault_likelihood_block(d["cube"], DT)
        _CACHE["c"] = (d, r)
    return _CACHE["c"]


def plane_points(f, margin=4, n_max=56, k_range=(25, 160), step=4, half=20):
    a = math.radians(f["az"])
    out = []
    for k in range(k_range[0], k_range[1], step):
        for s_ in np.linspace(-half, half, 21):
            h = (k - f["k0"]) / f["slope"]
            ii = f["i0"] + h * math.cos(a) - s_ * math.sin(a)
            jj = f["j0"] + h * math.sin(a) + s_ * math.cos(a)
            if margin <= ii < n_max - margin and margin <= jj < n_max - margin:
                out.append((ii, jj, k))
    return out


def run(check, tmp):
    d, r = case()
    fl, tfl, n = r["fl"], r["tfl"], r["n"]
    check(fl.shape == d["cube"].shape and 0 <= fl.min() and fl.max() <= 1 and np.median(fl) < 0.15,
          f"32a: fault likelihood in [0, 1], low away from faults (median {np.median(fl):.3f})")
    for q, f in enumerate(FAULTS):
        a = math.radians(f["az"])
        dist, hit, pts = [], 0, plane_points(f)
        nrm = []
        for ii, jj, k in pts:
            best = None
            for dh in np.linspace(-3, 3, 13):
                v = FL.sample3(tfl, np.array([ii + dh * math.cos(a)]), np.array([jj + dh * math.sin(a)]),
                               np.array([float(k)]))[0]
                if best is None or v > best[0]:
                    best = (v, dh)
            if best[0] > 0.3:
                hit += 1
                dist.append(abs(best[1]))
                nrm.append(n[int(round(ii)), int(round(jj)), k])
        st, dp = FL.geo_orientation(np.array(nrm), EI, EJ, DT)
        ts, td = FL.geo_orientation(d["normals"][q], EI, EJ, DT)
        ds = float(np.median((st - ts + 180) % 360 - 180))
        dd = float(np.median(dp) - td)
        check(hit > 0.7 * len(pts) and np.median(dist) <= 1.0,
              f"32a: TFL peaks within 1 trace of fault {q + 1} ({hit}/{len(pts)} plane points, median "
              f"{np.median(dist):.2f} traces)")
        check(abs(ds) < 5 and abs(dd) < 5, f"32a: fault {q + 1} strike {ts:.0f} / dip {td:.0f} deg recovered within 5 deg "
              f"(errors {ds:+.1f} / {dd:+.1f})")
    # geo <-> index normal round trip
    nn = FL.plane_normal_index(37.0, 62.0, EI, EJ, DT)
    s2, d2 = FL.geo_orientation(nn, EI, EJ, DT)
    check(abs(float(s2) - 37.0) < 1e-6 and abs(float(d2) - 62.0) < 1e-6, "32a: strike / dip <-> index normal round trip")
    rot = (np.array([0.0, 25.0]), np.array([-25.0, 0.0]))  # survey rotated 90 deg (inline axis pointing north)
    s3, _ = FL.geo_orientation(FL.plane_normal_index(37.0, 62.0, *rot, DT), *rot, DT)
    check(abs(float(s3) - 37.0) < 1e-6, "32a: strike in map azimuth on a rotated survey")
    # TFL is thin: across a fault at most ~1-2 non-zero voxels
    f = FAULTS[2]
    a = math.radians(f["az"])
    widths = []
    for ii, jj, k in plane_points(f)[::7]:
        prof = [FL.sample3(tfl, np.array([ii + dh * math.cos(a)]), np.array([jj + dh * math.sin(a)]),
                           np.array([float(k)]))[0] for dh in np.arange(-4, 4.01, 1.0)]
        widths.append(sum(1 for v in prof if v > 0.3))
    check(np.median(widths) <= 2, f"32a: thinned likelihood is 1-2 voxels wide (median {np.median(widths)})")
    # ant-track-style enhancement of a noisy edge attribute
    rng = np.random.default_rng(0)
    edge = (1.0 - r["semblance"]) + 0.3 * rng.random(fl.shape).astype(np.float32)
    enh, _ = FL.enhance(edge, iterations=2)
    on = [enh[int(round(p[0])), int(round(p[1])), p[2]] for p in plane_points(FAULTS[2])]
    near = [float(np.max([FL.sample3(enh, np.array([p[0] + dh * math.cos(a)]), np.array([p[1] + dh * math.sin(a)]),
                                     np.array([float(p[2])]))[0] for dh in (-1, 0, 1)])) for p in plane_points(FAULTS[2])]
    bg = enh[d["T"] > -1e8][::97]
    check(np.median(near) > 4 * max(np.median(bg), 1e-3) and np.mean(enh > 0) < 0.2,
          f"32a: ant-track-style enhancement keeps the fault, suppresses noise (fault {np.median(near):.2f} vs "
          f"background {np.median(bg):.3f})")
    del on
    # 2D line through the volume (an inline)
    il = 30
    fl2, tfl2, n2 = FL.fault_likelihood_2d(d["cube"][il], DT)
    f0 = FAULTS[0]
    a0 = math.radians(f0["az"])
    hit2 = 0
    tot = 0
    for k in range(30, 150, 6):
        # where fault 1 crosses inline il: (il - i0) cos a + (j - j0) sin a = h
        h = (k - f0["k0"]) / f0["slope"]
        jj = f0["j0"] + (h - (il - f0["i0"]) * math.cos(a0)) / math.sin(a0)
        if 4 <= jj < 52:
            tot += 1
            hit2 += float(np.max(tfl2[max(0, int(jj) - 2):int(jj) + 3, k])) > 0.3
    check(tot > 5 and hit2 >= 0.7 * tot, f"32g: 2D fault likelihood finds the fault on a line ({hit2}/{tot})")


def run_volume(check, tmp):
    """Fault volumes written block-wise with a halo = the in-memory result; registered in the project."""
    from .project import SeisProject
    from .synth import write_3d
    from .volume import import_3d
    d, r = case()
    folder = os.path.join(tmp, "faultvol")
    os.makedirs(folder, exist_ok=True)
    cube = d["cube"][:, :40, :120]
    sgy = os.path.join(folder, "f.sgy")
    write_3d(sgy, cube, dt_us=4000)
    sp = SeisProject()
    sp.path = os.path.join(folder, "f.seisproj.json")
    store = os.path.join(folder, "f_seismic", "F")
    import_3d(sgy, store)
    sp.volumes.append({"name": "F", "store": store, "segy": sgy})
    names = FL.compute_fault_volumes(sp, "F", block=10)
    check(names == ["F [Fault likelihood]", "F [TFL]", "F [Fault strike]", "F [Fault dip]"] and
          all(any(v["name"] == nm for v in sp.volumes) for nm in names), "32a: fault volumes registered in the project")
    v = sp.volume("F [Fault likelihood]")
    full = FL.fault_likelihood_block(np.asarray(sp.volume("F").ilxl, np.float32), DT)["fl"]
    got = np.asarray(v.ilxl[:], np.float32) * v.inv
    mid = slice(12, 44)  # away from the volume edges the block halo reproduces the one-block result
    err = float(np.percentile(np.abs(got[mid] - full[mid]), 99))
    check(err < 0.03, f"32a: block-wise volume = whole-volume computation (p99 diff {err:.4f})")
    dipv = sp.volume("F [Fault dip]")
    dv = np.asarray(dipv.ilxl[:], np.float32) * dipv.inv
    check(np.nanmax(dv) <= 90.5 and np.percentile(dv[got > 0.8], 50) > 30, "32a: fault dip volume in degrees")
    sec = FL.section_likelihood(sp.volume("F"), "Inline", int(sp.volume("F").inlines[20]))
    check(sec.shape == (40, 120) and float(np.max(sec)) > 0.8, "32a: quick fault likelihood on one inline")


def _skins():
    if "s" not in _CACHE:
        from . import faultskin as FS
        d, r = case()
        _CACHE["s"] = FS.extract_skins(r["tfl"], r["n"], r["fl"], threshold=0.5, min_size=150,
                                       scale=FS.default_scale(EI, EJ, DT))
    return _CACHE["s"]


def _match(skin, d):
    angs = [math.degrees(math.acos(min(1.0, abs(float(skin.normal @ n))))) for n in d["normals"]]
    return int(np.argmin(angs)), float(min(angs))


def run_skins(check, tmp):
    """32c extraction, 32d QC / filtering / merging, 32b / 32e throw, 32f fault polygons."""
    from . import faultskin as FS
    from .faults import FaultSet
    d, r = case()
    skins = _skins()
    big = [s for s in skins if len(s) > 1000]
    matches = [_match(s, d) for s in big]
    check(len(big) >= 3 and all(a < 5 for _, a in matches) and {q for q, _ in matches} == {0, 1, 2},
          f"32c: skins for all three faults, each a single orientation ({[len(s) for s in big]})")
    pair = [q for q, _ in matches]
    check(all(len({q for q, _ in [_match(s, d)]}) == 1 for s in big), "32c: crossing conjugate faults in separate skins")
    # merge assistant: the halves of a fault cut by a younger fault are offered for merging, different faults not
    cand = FS.merge_candidates(big, max_gap=6.0, max_angle=10.0)
    by = {s.id: _match(s, d)[0] for s in big}
    good = [c for c in cand if by[c[0]] == by[c[1]]]
    bad = [c for c in cand if by[c[0]] != by[c[1]]]
    check(len(good) >= 1 and not bad, f"32d: merge assistant pairs halves of one fault, never two faults ({cand})")
    merged = list(big)
    for a, b, _, _ in good:
        if any(s.id == a for s in merged) and any(s.id == b for s in merged):
            merged = FS.merge_skins(merged, a, b)
    per = {}
    for s in merged:
        per.setdefault(_match(s, d)[0], []).append(s)
    check(len(merged) <= 4 and all(len(v) <= 2 for v in per.values()) and len(per) == 3,
          f"32d: after merging ~one skin per fault ({ {k: [len(s) for s in v] for k, v in per.items()} })")
    main = {k: max(v, key=len) for k, v in per.items()}
    for q, s in sorted(main.items()):
        st, dp = s.strike_dip(EI, EJ, DT)
        ts, td = FL.geo_orientation(d["normals"][q], EI, EJ, DT)
        thr = FS.throw_at_points(s, d["T"])
        check(abs((st - float(ts) + 180) % 360 - 180) < 5 and abs(dp - float(td)) < 5 and
              abs(float(np.nanmedian(thr)) - FAULTS[q]["throw"]) <= 1.0,
              f"32c/32b: fault {q + 1} skin strike {st:.0f} dip {dp:.0f}, RGT throw {np.nanmedian(thr):.1f} samples "
              f"(true {FAULTS[q]['throw']})")
    # split + table + D-L flag
    s0 = main[2]
    sp = FS.split_skin(merged, s0.id, axis=2)
    check(len(sp) == len(merged) + 1 and sum(len(s) for s in sp) == sum(len(s) for s in merged), "32d: split a skin")
    rows = FS.skin_table(merged, EI, EJ, DT)
    check(len(rows) == len(merged) and rows[0]["points"] > 0 and rows[0]["throw_max"] is not None,
          "32d: fault table (points, strike, dip, likelihood, max throw)")
    check(FS.displacement_length_flag(1000.0, 500.0) == "too large for its length" and
          FS.displacement_length_flag(1000.0, 20.0) == "", "32e: displacement-length QC flag")
    # stereonet + rose + brushing
    sts = np.array([s.strike_dip(EI, EJ, DT)[0] for s in merged])
    dps = np.array([s.strike_dip(EI, EJ, DT)[1] for s in merged])
    net = FS.stereonet(sts, dps, weights=[len(s) for s in merged])
    rr = np.hypot(net["x"], net["y"])
    check(np.all(rr <= 1.0 + 1e-9) and np.nanmax(net["density"]) > 2.0, "32d: stereonet poles + Kamb density")
    tr, pl = FS.poles(np.array([90.0]), np.array([90.0]))  # vertical E-W plane: horizontal pole to the north
    x, y = FS.stereo_xy(tr, pl)
    xc, yc = FS.stereo_xy(*FS.poles(np.array([0.0]), np.array([0.0])))  # horizontal plane: vertical pole
    check(abs(float(x[0])) < 1e-9 and abs(float(y[0]) - 1.0) < 1e-9 and abs(float(xc[0])) < 1e-9 and
          abs(float(yc[0])) < 1e-9, "32d: equal-area net: horizontal pole on the rim, vertical pole at the centre")
    edges, cnt = FS.rose(sts, bins=18, weights=[len(s) for s in merged])
    check(cnt.sum() == sum(len(s) for s in merged) and len(edges) == 19, "32d: rose diagram of strikes (axial)")
    f3 = FS.filter_skins(merged, EI, EJ, DT, strike=(220, 260))
    check(f3 and all(_match(s, d)[0] == 2 for s in f3), "32d: filter by strike keeps fault 3 only")
    px, py = FS.stereo_xy(*FS.poles(np.array([float(main[2].strike_dip(EI, EJ, DT)[0])]),
                                    np.array([float(main[2].strike_dip(EI, EJ, DT)[1])])))
    brush = [(px[0] - 0.1, py[0] - 0.1), (px[0] + 0.1, py[0] - 0.1), (px[0] + 0.1, py[0] + 0.1), (px[0] - 0.1, py[0] + 0.1)]
    fb = FS.filter_skins(merged, EI, EJ, DT, poly=brush)
    check(fb and all(_match(s, d)[0] == 2 for s in fb), "32d: brushing the stereonet selects that fault's skins")
    check(not FS.filter_skins(merged, EI, EJ, DT, dip=(80, 90)), "32d: dip filter")

    # skins -> item-2 faults with editable sticks
    class Vol:  # index -> map like synth.write_3d (25 m bins)
        inlines = 1001 + np.arange(56)
        xlines = 2001 + np.arange(56)
        il_step = xl_step = 1
        dt, t0, nil, nxl = 0.004, 0.0, 56, 56

        def xy(self, il, xl):
            return 500000.0 + 25.0 * (np.asarray(xl) - 2001), 6000000.0 + 25.0 * (np.asarray(il) - 1001)
    fs = FaultSet()
    f = FS.skin_to_fault(main[2], Vol(), fs, every=5)
    ok = len(f.sticks) >= 3 and all(len(s) >= 2 for s in f.sticks)
    tris = f.triangles() if ok else []
    check(ok and len(tris) > 0 and f.name in fs.names(), f"32c: skin -> fault with {len(f.sticks)} editable sticks + "
          "triangulated surface")
    st0 = np.asarray(f.sticks[0])
    check(np.all(np.diff(st0[:, 2]) > 0) and st0[0, 2] < st0[-1, 2], "32c: sticks ordered top to bottom (ms)")
    # 32f: fault polygons on a horizon (iso geologic-time surface)
    tau0 = 95.0
    k = np.argmax(d["T"] >= tau0, axis=2).astype(float)
    surf = k * DT
    polys = {}
    for q, s in main.items():
        fp = FS.fault_polygon(s, surf, DT)
        polys[q] = fp
    fp = polys[2]
    check(fp is not None and len(fp["poly"]) >= 4 and abs(fp["throw_ms"] - FAULTS[2]["throw"] * DT) <= DT * 1.5,
          f"32f: heave polygon of fault 3 with throw {fp['throw_ms'] if fp else None:.0f} ms (true "
          f"{FAULTS[2]['throw'] * DT:.0f})")
    from .maphub import point_in_polygon
    others = [p for q, p in polys.items() if q != 2 and p is not None]
    inside, tot = 0, 0
    for o in others:
        c = o["poly"]
        inside += int(point_in_polygon(c[:, 0], c[:, 1], fp["poly"][:, 0], fp["poly"][:, 1]).sum())
        tot += len(c)
    closed = np.allclose(fp["poly"][0], fp["fw"][0]) and np.allclose(fp["poly"][-1], fp["hw"][0])
    check(closed and inside <= 0.1 * tot, f"32f: fault polygons close and do not overlap other faults' (only where "
          f"crossing faults meet: {inside} of {tot} vertices)")
    items = FS.polygons_layer_items([main[2]], surf, Vol())
    check(items and items[0]["down"] in ("N", "NE", "E", "SE", "S", "SW", "W", "NW") and len(items[0]["x"]) >= 4,
          "32f: polygons as hub layer items with the downthrown side")
    # 32b: throw cube from RGT + throw-weighted probability
    cube, w = FS.rgt_throw_volume(d["T"], r["fl"], r["n"], threshold=0.8)
    on = r["tfl"] > 0.8
    check(np.nanmedian(cube[on]) > 3 and float(w.max()) <= 1.0 and float(np.mean(w[on] > 0.5)) > 0.5,
          f"32b: RGT throw cube (median {np.nanmedian(cube[on]):.1f} samples on faults) + throw-weighted probability")


def run_gui(check, tmp):
    """32 GUI: fault mapping dialog from the main window: volumes, extraction, filters, faults, polygons -> map."""
    import json
    from .gui.main_window import MainWindow
    from .project import SeisProject
    from .synth import write_3d
    from .volume import import_3d
    d, r = case()
    folder = os.path.join(tmp, "faultgui")
    os.makedirs(folder, exist_ok=True)
    sgy = os.path.join(folder, "f.sgy")
    write_3d(sgy, d["cube"], dt_us=4000)
    sp = SeisProject()
    sp.path = os.path.join(folder, "f.seisproj.json")
    store = os.path.join(folder, "f_seismic", "F")
    import_3d(sgy, store)
    sp.volumes.append({"name": "F", "store": store, "segy": sgy})
    from .selftest import write_segy
    for nm, img, xs, ys in (("IL16", d["cube"][16], 500000.0 + 25.0 * np.arange(56), np.full(56, 6000000.0 + 400.0)),
                            ("IL40", d["cube"][40], 500000.0 + 25.0 * np.arange(56), np.full(56, 6000000.0 + 1000.0)),
                            ("XL14", d["cube"][:, 14], np.full(56, 500000.0 + 350.0), 6000000.0 + 25.0 * np.arange(56)),
                            ("XL42", d["cube"][:, 42], np.full(56, 500000.0 + 1050.0), 6000000.0 + 25.0 * np.arange(56))):
        pth = os.path.join(folder, nm + ".sgy")
        write_segy(pth, img, {21: np.arange(1, 57), 181: xs.astype(np.int64), 185: ys.astype(np.int64)}, dt_us=4000)
        sp.lines.append({"name": nm, "segy": pth, "cdp_byte": 21})
    h = sp.horizons.new("H95")
    k = np.argmax(d["T"] >= 95.0, axis=2).astype(np.float32)
    h.data["3D:F"] = k * 4.0
    sp.save()
    mw = MainWindow(sp.path)
    dlg = mw.fault_mapping("F")
    names = dlg.compute(threaded=False)
    check(len(names) == 4 and any(v["name"] == "F [TFL]" for v in mw.project.volumes),
          "32 GUI: fault likelihood volumes computed from the dialog")
    dlg.threshold, dlg.min_size = 0.5, 150
    sk = dlg.extract()
    check(len(sk) >= 3 and len(dlg.rows()) == len(sk), f"32 GUI: {len(sk)} fault skins extracted + table")
    st3 = float(FL.geo_orientation(d["normals"][2], *FL.index_frame(mw.project.volume("F")), 4.0)[0])
    f3 = dlg.apply_filters(strike=(st3 - 15, st3 + 15))
    check(f3 and len(f3) < len(sk), f"32 GUI: table filtered by strike ({st3:.0f} +- 15 deg)")
    dlg.apply_filters(strike=None)
    dlg.filters = {}
    dlg.apply_filters()
    png = dlg.stereonet_png(os.path.join(folder, "stereo.png"))
    check(os.path.getsize(png) > 3000, "32 GUI: stereonet + rose plot")
    n_sugg = len(dlg.suggestions())
    names = dlg.create_faults(every=5)
    check(names and all(n in mw.project.faults.names() for n in names) and
          all(len(mw.project.faults.get(n).sticks) >= 2 for n in names), "32 GUI: faults with sticks in the fault list")
    seen = []
    mw.mapbus.subscribe("probe", seen.append)
    lid, items = dlg.polygons("H95", mw.basemap.stack)
    lay = mw.basemap.stack.get(lid)
    check(items and lay["style"]["fault"] and any(e.get("kind") == "layers" for e in seen),
          f"32f GUI: fault polygons on the horizon as a fault layer (barrier), map bus notified ({n_sugg} merge hints)")
    lm = mw.live_map()
    check(lm.canvas.stack is mw.basemap.stack and lm.gridder.barrier_lines(), "32f GUI: Live Map shares the layers; "
          "fault polygons are gridding barriers")
    cuts, groups, n2d = dlg.lines_2d()
    check(len(cuts) >= 6 and len(n2d) >= 3 and all(len(mw.project.faults.get(n).sticks) >= 2 for n in n2d),
          f"32g GUI: 2D fault cuts linked across the lines -> {len(n2d)} faults with one stick per cut")
    p2 = SeisProject.open(sp.path)
    check(set(names) <= set(p2.faults.names()) and json.load(open(sp.path)), "32 GUI: auto faults saved in the project")
    mw.maplink.close()


def run_throw_2d(check, tmp):
    """32e correlation throw (no RGT); 32g 2D cuts + cut correlation across intersecting lines; 32h ML fallback."""
    from . import faultskin as FS
    d, r = case()
    skins = _skins()
    for s in skins[:5]:
        q, ang = _match(s, d)
        if len(s) < 3000:
            continue
        th = FS.throw_by_correlation(s, d["cube"])
        check(np.isfinite(th).mean() > 0.5 and abs(float(np.nanmedian(th)) - FAULTS[q]["throw"]) <= 1.0,
              f"32e: correlation throw of fault {q + 1}: {np.nanmedian(th):.1f} samples (true {FAULTS[q]['throw']}, "
              f"{np.isfinite(th).mean():.0%} of points)")
        cs, med, mx = FS.throw_profile(s, "strike", 10)
        se, te, tm = FS.throw_map(s, (8, 6))
        check(len(cs) == 10 and np.nanmax(mx) >= np.nanmax(np.abs(med)) and tm.shape == (8, 6),
              f"32e: throw-distance profile + throw map on the fault plane (fault {q + 1})")
        break
    # 2D lines: two inlines + two crosslines of the cube with map coordinates (25 m bins)
    lines = {}
    for il in (16, 40):
        lines[f"IL{il}"] = (d["cube"][il], 25.0 * np.arange(56), np.full(56, 25.0 * il))
    for xl in (14, 42):
        lines[f"XL{xl}"] = (d["cube"][:, xl], np.full(56, 25.0 * xl), 25.0 * np.arange(56))
    cuts = []
    for name, (img, x, y) in lines.items():
        fl2, tfl2, n2 = FL.fault_likelihood_2d(img, DT)
        cuts += FS.cuts_on_line(name, x, y, tfl2, n2, DT, image=img, threshold=0.5, min_len=15)
    groups = FS.link_cuts(cuts)
    # which true fault does each cut belong to: nearest plane in index space (x = 25 j, y = 25 i)

    def fault_of(c):
        best = None
        for q, f in enumerate(FAULTS):
            a = math.radians(f["az"])
            i_, j_ = c.y / 25.0, c.x / 25.0
            h = (i_ - f["i0"]) * math.cos(a) + (j_ - f["j0"]) * math.sin(a)
            dist = np.median(np.abs(h - (c.times / DT - f["k0"]) / f["slope"]))
            if best is None or dist < best[0]:
                best = (dist, q)
        return best[1] if best[0] < 2.0 else None
    truth = [fault_of(c) for c in cuts]
    multi = [g for g in groups if len({cuts[m].line for m in g}) >= 2]
    maj, fr = [], []
    for g in multi:
        lab = [truth[m] for m in g if truth[m] is not None]
        if lab:
            top = max(set(lab), key=lab.count)
            maj.append(top)
            fr.append(lab.count(top) / len(lab))
    check(len(cuts) >= 6 and set(maj) == {0, 1, 2} and len(maj) == 3 and min(fr) >= 0.8,
          f"32g: {len(cuts)} fault cuts on 4 crossing lines linked into {len(multi)} faults, one per true fault "
          f"(purity {min(fr) if fr else 0:.0%}; a cut where two faults cross is ambiguous)")
    be = FS.ml_backend()
    try:
        FS.cnn_fault_probability(np.zeros((4, 4, 4)))
        ok = False
    except RuntimeError as e:
        ok = ("Install 3D extras" in str(e)) if be is None else ("model" in str(e))
    check(ok, f"32h: optional CNN fault probability gives a clear message (backend: {be})")
