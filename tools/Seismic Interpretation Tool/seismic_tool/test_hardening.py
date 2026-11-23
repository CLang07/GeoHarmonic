"""Item 15 (scale + hardening): cancel, well index, overview volume."""
import os

import numpy as np


def run(check, tmp):
    from .synth import write_3d, traces_from_surfaces, folded_surfaces
    from .volume import import_3d
    d = os.path.join(tmp, "hard")
    os.makedirs(d)
    top, deep = folded_surfaces(30, 40)
    cube = traces_from_surfaces([top, deep], [1.0, -0.7], 120, 0.004)
    sgy = os.path.join(d, "c.sgy")
    write_3d(sgy, cube)
    store = os.path.join(d, "store")
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 1
    try:
        import_3d(sgy, store, chunk=300, cancel=cancel)
        ok = False
    except RuntimeError as e:
        ok = str(e) == "cancelled"
    check(ok and not os.path.exists(os.path.join(store, "data_ilxl.npy")) and not os.path.exists(
        os.path.join(store, "meta.json")), "import cancelled: partial store removed")
    v = import_3d(sgy, store, chunk=300, cancel=lambda: False)
    check(v.nil == 30 and v.nxl == 40, "import after a cancelled one works")
    # irregular geometry: missing traces, a NaN trace, a dead trace, IBM floats
    from .selftest import write_segy
    nil, nxl, ns = 12, 15, 80
    tr = traces_from_surfaces([np.full((nil, nxl), 150.0)], [1.0], ns, 0.004).reshape(nil * nxl, ns)
    il = np.repeat(np.arange(100, 100 + nil), nxl)
    xl = np.tile(np.arange(10, 10 + 2 * nxl, 2), nil)
    keep = np.ones(nil * nxl, bool)
    keep[[3, 4, 50, 51, 52]] = False  # holes
    tr[20] = np.nan
    tr[21] = 0.0
    p2 = os.path.join(d, "irr.sgy")
    write_segy(p2, tr[keep], {189: il[keep], 193: xl[keep], 181: 1000 + 25 * (xl[keep] - 10) // 2,
                              185: 5000 + 25 * (il[keep] - 100)}, fmt=5)
    v2 = import_3d(p2, os.path.join(d, "irr"))
    a = np.asarray(v2.ilxl, np.float32)
    check(v2.nxl == nxl and v2.xl_step == 2 and abs(v2.meta["fill_fraction"] - (1 - 5 / (nil * nxl))) < 1e-9
          and np.isfinite(a).all() and np.abs(a[1, 5]).max() == 0 and np.abs(a[0, 3]).max() == 0
          and np.abs(a[2, 0]).max() > 0, "irregular 3D: holes, NaN + dead traces, crossline step 2")
    wellindex_checks(check, d)
    welllayer_checks(check)
    parallel_import_checks(check, d)
    xline_store_checks(check, d)
    review_checks(check, d)
    review2_checks(check, d)
    return v


def parallel_import_checks(check, d):
    """Threaded import writes exactly the same stores as the serial one; cancel still cleans up."""
    from .synth import write_3d, traces_from_surfaces, folded_surfaces
    from .volume import import_3d
    top, deep = folded_surfaces(23, 31)
    cube = traces_from_surfaces([top, deep], [1.0, -0.7], 150, 0.004)
    sgy = os.path.join(d, "par.sgy")
    write_3d(sgy, cube)
    a = import_3d(sgy, os.path.join(d, "ser"), chunk=97, workers=1)
    b = import_3d(sgy, os.path.join(d, "par"), chunk=97, workers=3)
    same = (np.array_equal(np.asarray(a.ilxl), np.asarray(b.ilxl)) and np.array_equal(np.asarray(a.tdata),
            np.asarray(b.tdata)) and a.meta["fill_fraction"] == b.meta["fill_fraction"] == 1.0)
    check(same, "parallel import (3 threads, 8 chunks) == serial import, both stores")
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2  # header pass + one finished chunk, then cancel
    store = os.path.join(d, "parc")
    try:
        import_3d(sgy, store, chunk=50, workers=3, cancel=cancel)
        ok = False
    except RuntimeError as e:
        ok = str(e) == "cancelled"
    check(ok and not os.path.exists(os.path.join(store, "data_ilxl.npy")), "parallel import cancelled cleanly")


def xline_store_checks(check, d):
    """Crossline-major copy: written at import, equal to the inline-major store, added later to old stores;
    sorted surveys take the contiguous block write and match a shuffled (scattered-write) import exactly."""
    from .synth import write_3d, traces_from_surfaces, folded_surfaces
    from .selftest import write_segy
    from .volume import import_3d, build_xline_store, Volume3D
    top, deep = folded_surfaces(19, 27)
    cube = traces_from_surfaces([top, deep], [1.0, -0.7], 120, 0.004, noise=0.05)
    sgy = os.path.join(d, "xl.sgy")
    write_3d(sgy, cube)
    v = import_3d(sgy, os.path.join(d, "xl_new"), chunk=64, workers=2)
    ref = np.asarray(v.ilxl)
    check(v.xldata is not None and v.meta.get("xline_store") and np.array_equal(np.asarray(v.xldata),
          np.transpose(ref, (1, 0, 2))), "import writes the crossline copy (= inline store transposed)")
    ok = all(np.array_equal(v.xline(int(x)), np.asarray(ref[:, k, :], np.float32) * v.inv)
             for k, x in enumerate(v.xlines))
    check(ok, "crossline read from the crossline copy == read from the inline store")
    old = import_3d(sgy, os.path.join(d, "xl_old"), chunk=64, xline_store=False)
    check(old.xldata is None and np.array_equal(old.xline(int(old.xlines[5])), v.xline(int(v.xlines[5]))),
          "store without the crossline copy still reads crosslines")
    build_xline_store(old.store)
    re = Volume3D(old.store)
    check(re.xldata is not None and np.array_equal(np.asarray(re.xldata), np.asarray(v.xldata))
          and re.meta.get("xline_store"), "--xlstore adds the crossline copy to an old store")
    nil, nxl, ns = cube.shape
    il = np.repeat(np.arange(1001, 1001 + nil), nxl)
    xl = np.tile(np.arange(2001, 2001 + nxl), nil)
    perm = np.random.default_rng(3).permutation(nil * nxl)
    shp = os.path.join(d, "xl_shuf.sgy")
    write_segy(shp, cube.reshape(-1, ns)[perm], {189: il[perm], 193: xl[perm],
                                                 181: (500000 + 25 * (xl[perm] - 2001)).astype(np.int64),
                                                 185: (6000000 + 25 * (il[perm] - 1001)).astype(np.int64)})
    sh = import_3d(shp, os.path.join(d, "xl_shuf"), chunk=64, workers=2)
    check(np.array_equal(np.asarray(sh.ilxl), ref) and np.array_equal(np.asarray(sh.xldata), np.asarray(v.xldata)),
          "unsorted SEG-Y (scattered writes) == sorted SEG-Y (block writes)")


def welllayer_checks(check):
    """Basemap well layer: labels only in view and only up to max_labels; index path == numpy path."""
    from .wellindex import WellLayer
    rng = np.random.default_rng(11)
    xy = rng.uniform(0, 100000, (5000, 2))
    wells = [(f"W{i}", x, y) for i, (x, y) in enumerate(xy)] + [("NOXY", None, None), ("NAN", float("nan"), 1.0)]
    big = WellLayer(wells)  # indexed (>= 2000)
    small = WellLayer(wells, index_from=10 ** 9)  # numpy box test
    ok = big.index is not None and small.index is None and len(big) == 5000
    for x0, y0, w in ((1000, 2000, 3000), (50000, 50000, 8000), (90000, 10, 15000)):
        ok &= np.array_equal(big.in_view(x0, x0 + w, y0, y0 + w), small.in_view(x0 + w, x0, y0, y0 + w))
    ok &= big.labels(None) == [] and big.labels(((0, 100000), (0, 100000))) == []
    lab = big.labels(((20000, 23000), (40000, 43000)))
    want = {f"W{i}" for i, (x, y) in enumerate(xy) if 20000 <= x <= 23000 and 40000 <= y <= 43000}
    ok &= 0 < len(lab) <= 150 and {n_ for n_, _, _ in lab} == want
    for qx, qy in rng.uniform(0, 100000, (10, 2)):
        ok &= big.nearest(qx, qy) == small.nearest(qx, qy) == int(np.argmin(((xy - [qx, qy]) ** 2).sum(1)))
    few = WellLayer([("A", 0.0, 0.0), ("B", 10.0, 0.0)])
    ok &= [n_ for n_, _, _ in few.labels(None)] == ["A", "B"] and WellLayer([]).labels(None) == [] \
        and WellLayer([]).nearest(0, 0) is None
    check(ok, f"basemap well layer: 5,000 wells, labels only in view ({len(lab)}), index == numpy, nearest")


def wellindex_checks(check, d):
    import time
    from .wellindex import WellIndex
    rng = np.random.default_rng(5)
    n = 10000
    xy = rng.uniform(0, 200000, (n, 2))
    path = os.path.join(d, "wells.sqlite")
    ix = WellIndex(path, bucket=5000.0)
    t0 = time.time()
    ix.add_wells((f"W{i:05d}", f"42-{i:03d}-{i:05d}", x, y, 30.0, 3000.0) for i, (x, y) in enumerate(xy))
    ix.add_wells([("NOXY", None, None, None, None, None)])
    ix.add_tops({f"W{i:05d}": {"Top A": 1000.0 + i % 7} for i in range(0, n, 2)})
    ix.add_logs([(f"W{i:05d}", "GR", 100.0, 3000.0, f"W{i}.las") for i in range(0, n, 3)])
    t_build = time.time() - t0
    t0 = time.time()
    got = ix.in_box(50000, 60000, 120000, 135000)
    t_q = time.time() - t0
    want = {f"W{i:05d}" for i, (x, y) in enumerate(xy) if 50000 <= x <= 60000 and 120000 <= y <= 135000}
    check({g[0] for g in got} == want and t_q < 0.2, f"well index: box query = brute force ({len(want)} of {n} wells, "
                                                    f"{1000 * t_q:.1f} ms; build {t_build:.2f} s)")
    ok = True
    for qx, qy in rng.uniform(0, 200000, (20, 2)):
        nn = ix.nearest(qx, qy, 3)
        d2 = ((xy - [qx, qy]) ** 2).sum(1)
        ok &= [r[0] for r in nn] == [f"W{i:05d}" for i in np.argsort(d2)[:3]]
    check(ok, "well index: k-nearest = brute force")
    ix.close()
    ix = WellIndex(path)
    check(len(ix) == n + 1 and ix.bucket == 5000.0 and len(ix.tops_of("Top A")) == n // 2
          and len(ix.wells_with_curve("GR")) == len(range(0, n, 3)) and ix.by_api("4200700007") == ["W00007"]
          and len(ix.headers(["W00001", "W00002", "NOXY"])) == 3 and ix.xy()[1].shape == (n, 2),
          "well index: reopened from disk (tops, log catalogue, API lookup, headers)")
    ix.close()


def run_gui(check, tmp):
    from .gui.main_window import MainWindow, job_cancelled
    check(job_cancelled({"cancel": True, "error": "Traceback...\nRuntimeError: cancelled"}) and
          not job_cancelled({"cancel": False, "error": "RuntimeError: cancelled"}), "cancel vs real error")

    class _S:
        pass
    s = _S()
    s._job = None
    s.statusBar = lambda: _S()
    check(MainWindow.cancel_job(s) is False, "cancel with no job")
    st = {"cancel": False}
    s._job = (None, None, st)
    s.statusBar = lambda: type("SB", (), {"showMessage": lambda self, *a: None})()
    check(MainWindow.cancel_job(s) and st["cancel"], "Cancel button flags the running job")
    # basemap: thousands of wells -> one scatter item + labels only for the wells in view
    from .gui.basemap import Basemap
    from .project import SeisProject
    import json
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 50000, (3000, 2))
    wp = os.path.join(tmp, "many.wcproj.json")
    with open(wp, "w") as fh:
        json.dump({"wells": [{"name": f"W{i}", "x": float(x), "y": float(y)} for i, (x, y) in enumerate(xy)]}, fh)
    sp = SeisProject()
    sp.well_project = wp
    bm = Basemap()
    bm.set_project(sp)
    all_view = bm.update_labels(((0, 50000), (0, 50000)))
    zoom = bm.update_labels(((1000, 3000), (1000, 3000)))
    want = {f"W{i}" for i, (x, y) in enumerate(xy) if 1000 <= x <= 3000 and 1000 <= y <= 3000}
    check(len(bm.wells) == 3000 and all_view == [] and {z[0] for z in zoom} == want and
          len(bm._label_items) == len(want), f"basemap: 3,000 wells, {len(want)} labels when zoomed in, none zoomed out")
    # dashboard nearest well (vectorised over every station of every wellbore)
    from .gui.dashboard_window import DashboardWindow
    from . import laterals as L
    trajs = L.synthetic_laterals(40)
    s2 = _S()
    s2.trajs = trajs
    ok = True
    for t in trajs[::7]:
        px, py = t.x[len(t.x) // 2] + 1.0, t.y[len(t.y) // 2] - 1.0
        brute = int(np.argmin([np.min(np.hypot(u.x - px, u.y - py)) for u in trajs]))
        ok &= DashboardWindow.nearest_well(s2, px, py) == brute
    check(ok, "dashboard: vectorised nearest wellbore == per-well search")
    # section colour levels from a strided sample of a big section
    from .gui.section_panel import SectionPanel
    s3 = _S()
    s3._data = rng.standard_normal((3000, 2000)).astype(np.float16)
    s3._level_sample = lambda limit=1_000_000: SectionPanel._level_sample(s3, limit)
    s3.attr = "Seismic"
    s3.clip_spin = type("Sp", (), {"value": lambda self: 99.0})()
    lo, hi = SectionPanel.color_levels(s3)
    full = float(np.percentile(np.abs(s3._data.astype(np.float32)), 99.0))
    check(abs(hi - full) / full < 0.01 and lo == -hi, f"section clip from a 1M sample of 6M values ({hi:.3f} vs {full:.3f})")
    review_gui_checks(check, tmp)
    return True


def review_gui_checks(check, tmp):
    """Regressions for the run-2 code review (GUI logic + laterals)."""
    from .dashboard_model import DashboardModel
    from .analytics.table import AttributeTable
    tab = AttributeTable([f"W{i}" for i in range(10)])
    tab.add("a", np.arange(10.0))
    m = DashboardModel(tab)
    m.set_filter("a", 3.0, 6.0)
    e, c, _ = m.histogram("a")
    check(e[0] == 0.0 and e[-1] == 9.0 and c.sum() == 4, "dashboard: histogram of a filtered column spans its full range")
    from . import laterals as L
    t = L.synthetic_laterals(1)[0]
    v = L.sample_log(t, np.arange(0.0, 3000.0, 10.0), np.full(300, np.nan))
    check(np.isnan(v).all() and len(v) == len(t.md), "all-null log curve samples to NaN (no crash)")
    from .gui.section_panel import SectionPanel
    from .project import SeisProject
    sp = SectionPanel()

    class Combo:  # PySide6 hands the stored tuple back as a list
        def __init__(self):
            self.items = [["3D", "A"], ["2D", "L1"]]
            self.cur = None

        def count(self):
            return len(self.items)

        def itemData(self, i):
            return self.items[i]

        def blockSignals(self, b):
            pass

        def setCurrentIndex(self, i):
            self.cur = i
    sp.src_combo = Combo()
    got = []
    sp._source_changed = lambda i, number=None, mode=None: got.append(i)
    sp.show("2D", "L1")
    check(got == [1], "section panel: show() matches a source stored as a list")
    sp._rgt_editors = {"3D:A": object()}
    sp.source = ("3D", "A")
    sp.refresh_sources = lambda select=None: None
    sp.set_project(SeisProject())
    check(sp.source is None and sp._rgt_editors == {}, "section panel: new project drops the old source and RGT editors")
    from .gui.welltie_dialog import WellTieDialog
    from . import tie as TK

    class D:
        pass
    dlg = D()
    dlg.well_name = "A"
    dlg.state = TK.TieState(5.0, 1.01, 900.0)
    dlg.shift, dlg.stretch, dlg.t_centre, dlg.corr, dlg.trace_offset, dlg._pending, dlg._data = 5.0, 1.01, 900.0, 0.8, 2, 1.0, {}
    WellTieDialog.set_well(dlg, "B")
    check(dlg.well_name == "B" and dlg.shift == 0.0 and dlg.state.shift == 0.0 and dlg.corr is None
          and dlg.trace_offset == 0, "well tie: switching wells resets the tie (no carry-over)")


def review_checks(check, tmp):
    """Regressions for the run-2 code review (core modules)."""
    from .horizons import correlate_pairs, HorizonSet
    # 1. fractional parent time kept by the trace-to-trace correlation (no snapping)
    t = np.arange(200.0)
    tr = np.exp(-0.5 * ((t - 100.4) / 3.0) ** 2)
    lag, c = correlate_pairs(tr[None], tr[None], np.array([100.4]), 8, 4)
    check(abs(lag[0]) < 0.05 and c[0] > 0.99, f"tracker: same trace at a fractional time -> lag {lag[0]:+.3f} (0)")
    # 2 + 3. horizon folders: undo of a delete after saving, name collisions (also case-only on Windows)
    d = os.path.join(tmp, "hzfold")
    hs = HorizonSet()
    for nm, v in (("Top C1.5", 1.0), ("Top C1_5", 2.0), ("top", 3.0), ("Top", 4.0)):
        hs.new(nm).data["3D:S"] = np.full((3, 3), v, np.float32)
    hs.save(d)
    js = hs.to_json(d)
    back = HorizonSet.load(js, d)
    ok = [float(back.get(n).data["3D:S"][0, 0]) for n in ("Top C1.5", "Top C1_5", "top", "Top")] == [1, 2, 3, 4]
    check(ok and len({os.path.normcase(e["dir"]).lower() for e in js}) == 4, "horizon folders unique (C1.5 / C1_5, top / Top)")
    hs.delete("Top C1.5")
    hs.save(d)
    hs.dirty = False
    hs.undo()
    check(hs.dirty, "undo of a deleted horizon marks the set dirty (its files are written again)")
    # 4. 2D line with descending CDP numbers keeps its navigation
    from .selftest import write_segy
    from .volume import Line2D
    n = 30
    cdp = np.arange(500, 500 - n, -1)
    p = os.path.join(tmp, "desc.sgy")
    write_segy(p, np.random.default_rng(1).standard_normal((n, 50)).astype(np.float32), {21: cdp})
    nav = np.column_stack([np.arange(400.0, 520.0), 1000.0 + 25 * np.arange(120.0), np.zeros(120)])
    ln = Line2D("D", p, 21, nav)
    check(np.allclose(ln.x, 1000.0 + 25 * (cdp - 400)), "2D line with descending CDPs keeps its navigation")
    # 6. junk extended-header count in a rev-0 file is ignored; ns = 0 raises
    from .segy import SegyFile
    raw = bytearray(open(p, "rb").read())
    raw[3504:3506] = (2).to_bytes(2, "big")
    raw[3500:3502] = (0).to_bytes(2, "big")
    p2 = os.path.join(tmp, "junkext.sgy")
    open(p2, "wb").write(bytes(raw))
    check(SegyFile(p2).ntraces == n, "rev-0 file with junk extended-header count: all traces read")
    # 8. amplitude extraction with the surface below the record -> NaN, no crash
    from .synth import write_3d
    from .volume import import_3d
    from .maps import extract_amplitude
    cube = np.random.default_rng(2).standard_normal((4, 5, 60)).astype(np.float32)
    write_3d(os.path.join(tmp, "short.sgy"), cube)
    v = import_3d(os.path.join(tmp, "short.sgy"), os.path.join(tmp, "short"))
    surf = np.full((4, 5), 5000.0)
    surf[0, 0] = 100.0
    ok = True
    for kind in ("rms", "max", "max_abs", "at"):
        r = extract_amplitude(v, surf, 8, 8, kind)
        ok &= np.isnan(r[1:, :]).all() and np.isfinite(r[0, 0])
    check(ok, "amplitude extraction: surface outside the record -> NaN, not a crash")
    # 10 + 11. single-inline survey map transform; nominal grid spacing with crossline step 2
    from .volume import _fix_degenerate, _affine
    xl = np.arange(10, 40, 2.0)
    il = np.full(len(xl), 7.0)
    x, y = 1000 + 12.5 * (xl - 10), 2000 + 0 * xl
    cx, cy = _affine(il, xl, x, y)
    cx, cy = _fix_degenerate(cx, cy, 1, len(xl), 1, 2, 7, 10)
    px = cx[0] + cx[1] * il + cx[2] * xl
    py = cy[0] + cy[1] * il + cy[2] * xl
    step_il = np.hypot(cx[1], cy[1])
    check(np.allclose(px, x) and np.allclose(py, y) and abs(step_il - 25.0) < 1e-6 and abs(cx[1]) < 1e-9,
          f"single-inline survey: inline axis perpendicular, {step_il:.1f} m (was arbitrary)")
    # 9. depth section outside the model / folding depths
    from .velocity import _monotonic_depths
    z = _monotonic_depths(np.array([[0.0, 10.0, 8.0, 20.0, np.nan]]))
    check(np.allclose(z[0, :4], [0, 10, 10, 20]) and np.isnan(z[0, 4]), "depth conversion never folds (running max)")
    # 5. re-import goes to a new store folder
    from .volume import unused_store
    check(unused_store(os.path.join(tmp, "short")) == os.path.join(tmp, "short") + "_2"
          and unused_store(os.path.join(tmp, "new")) == os.path.join(tmp, "new"), "re-import never overwrites a store")


def review2_checks(check, tmp):
    """Regressions for the second run-2 review (analytics, well readers, RGT / mistie / tie helpers)."""
    from .welltie import wc, ricker, WellTD
    wc()
    from well_correlation.geo import read_survey
    p = os.path.join(tmp, "surv.csv")
    with open(p, "w") as fh:
        fh.write("MD,INC,AZI,TVD,NS,EW\n1500,30,90,1477,1,2\n1600,32,92,1560,3,4\n1700,33,95,1650,5,6\n")
    md, inc, azi = read_survey(p)
    check(list(md[1:]) == [1500, 1600, 1700] and list(inc[1:]) == [30, 32, 33] and list(azi[1:]) == [90, 92, 95],
          "survey CSV with MD,INC,AZI,TVD,... header read by column name (was read as seq,MD,INC,AZI)")
    from .analytics import engineering as E
    ok = (E.detect_units("Proppant_Total_LBS", None) == "lb" and E.detect_units("TotalBaseWaterVolume", None) == "gal"
          and E.detect_units("Proppant (tons)", None) == "ton" and E.detect_units("Oil (m3)", None) == "m3")
    check(ok, "unit detection by whole tokens (LBS is lb, FracFocus water is gallons, US tons)")
    check(abs(E._date("2020-01-15T00:00:00") - E._date("01/15/2020")) < 1e-9 and E._date("2020/01/15") == E._date("2020-01-15"),
          "dates: ISO with time, YYYY/MM/DD")
    rows = [{"API": "1", "Month": "2020-01", "Oil (m3)": "10"}, {"API": "1", "Month": "2020-04", "Oil (m3)": "5"}]
    s = E.monthly_series(rows, {"api": "API", "month": "Month", "oil": "Oil (m3)"})["1"]
    check(len(s["month"]) == 4 and abs(s["oil"][0] - 62.8981) < 1e-3 and s["oil"][1] == 0,
          "production: m3 -> bbl, missing months filled with 0")
    q = E.arps_rate(1000.0, 0.1, 1.0, np.arange(0, 400))
    ser = np.r_[200.0, 500.0, 800.0, q[:9]]  # ramp-up, peak at month 3 (index 3)
    ft = E.arps_fit(ser[3:])
    true = ser.sum() + q[9:360 - 3].sum()
    check(abs(E.eur(ser, ft, peak=3) - true) / true < 0.01, "EUR forecast starts after the peak month")
    from .analytics.table import normalise_id
    check(normalise_id("512345678") == normalise_id("05-123-45678") and normalise_id("4225501234.0") == "4225501234",
          "API ids mangled by Excel (lost leading zero, trailing .0) still join")
    from .analytics.models import KMeans
    X = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    try:
        KMeans(3).fit(X)
        ok = True
    except Exception:
        ok = False
    check(ok, "k-means with more clusters than distinct rows does not crash")
    w3 = ricker(30.0, 3.0)
    check(len(w3) % 2 == 1 and np.argmax(w3) == len(w3) // 2, "Ricker wavelet centred for any dt (3 ms)")
    z = np.arange(1000.0, 2000.0, 1.0)
    t = 1000.0 + 2000.0 * (z - 1000.0) / 2500.0
    t[-1] = t[-2] + 2000.0 * 1.0 / 1500.0 * 3  # one washout-like interval at the end
    td = WellTD("x", z, t, "sonic")
    t_below = float(td.twt_raw(2499.0)) - t[-1]
    check(abs(t_below - 2000.0 * 500.0 / 2500.0) < 50.0, f"time-depth extrapolation not driven by one end sample "
                                                         f"({t_below:.0f} ms for 500 m at 2500 m/s = 400)")
    from .mistie import measure
    rng = np.random.default_rng(3)
    a = np.convolve(rng.standard_normal((1, 300))[0], ricker(40.0, 1.0, 40.0), "same")
    b = np.interp(np.arange(300) + 6.0, np.arange(300), a)
    r = measure(np.tile(a, (3, 1)), np.tile(b, (3, 1)), 1.0, (100, 160), 80.0)
    check(abs(abs(r["shift_ms"]) - 6.0) < 1.0, f"mistie measure with max lag > window ({r['shift_ms']:+.2f} ms, |6|)")
    from .selftest import write_segy
    from .volume import Line2D
    tr = rng.standard_normal((20, 40)).astype(np.float32)
    tr[3, 5] = np.nan
    pth = os.path.join(tmp, "nan2d.sgy")
    write_segy(pth, tr, {21: np.arange(1, 21)})
    check(np.isfinite(Line2D("N", pth).data()).all(), "2D line NaN samples replaced (dips / RGT would crash)")
