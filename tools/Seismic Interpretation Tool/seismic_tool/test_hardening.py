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
    return v


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
    return True
