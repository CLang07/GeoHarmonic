"""Self-test: writes small synthetic 2D and 3D SEG-Y files (IBM and IEEE), imports and checks them."""
import os
import shutil
import tempfile

import numpy as np



def _guarded(check, name, fn):
    """Run one test module; a crash is recorded as a FAIL with its traceback and the next module still runs
    (PLAN 38a: on Chris's PC one exception in the 3D Viewer test hid every GUI check after it)."""
    import traceback
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - report, keep going
        print(traceback.format_exc().rstrip())
        check(False, f"{name.split('(')[0]} crashed: {type(e).__name__}: {e}")

def _ieee2ibm(x):
    """float32 -> IBM float bits (uint32) for test files."""
    x = np.asarray(x, np.float64)
    out = np.zeros(x.shape, np.uint32)
    nz = x != 0
    sign = (x[nz] < 0).astype(np.uint32) << 31
    a = np.abs(x[nz])
    e = np.ceil(np.log(a) / np.log(16)).astype(int)
    m = a / 16.0 ** e
    over = m >= 1
    e[over] += 1
    m[over] /= 16
    mant = np.round(m * 16777216).astype(np.uint64)
    mant[mant >= 16777216] = 16777215
    out[nz] = sign | ((e + 64).astype(np.uint32) << 24) | mant.astype(np.uint32)
    return out


def write_segy(path, traces, headers, dt_us=4000, fmt=5):
    """traces (n, ns) float32; headers: {byte: int array} (4-byte words)."""
    traces = np.ascontiguousarray(traces)
    n, ns = traces.shape
    with open(path, "wb") as f:
        f.write(("C01 synthetic test" + " " * 62).encode("cp500") + b"\x40" * (3200 - 80))
        b = bytearray(400)
        b[16:18] = dt_us.to_bytes(2, "big")
        b[20:22] = ns.to_bytes(2, "big")
        b[24:26] = fmt.to_bytes(2, "big")
        f.write(bytes(b))
        th = np.zeros((n, 240), np.uint8)
        for byte, vals in headers.items():
            th[:, byte - 1: byte + 3] = np.asarray(vals, ">i4").view(np.uint8).reshape(n, 4)
        th[:, 114:116] = np.full(n, ns, ">i2").view(np.uint8).reshape(n, 2)
        body = (_ieee2ibm(traces).astype(">u4") if fmt == 1 else traces.astype(">f4")).view(np.uint8).reshape(n, -1)
        f.write(np.concatenate([th, body], axis=1).tobytes())


def run():
    from .segy import SegyFile, ibm2ieee
    from .volume import import_3d, Line2D
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok &= bool(cond)

    vals = np.array([1.0, -2.5, 0.15625, 1234.5, -1e-3], np.float32)
    check(np.allclose(ibm2ieee(_ieee2ibm(vals)), vals, rtol=1e-6), "IBM float conversion")
    tmp = tempfile.mkdtemp(prefix="seis_selftest_")
    try:
        # ---- 3D: 30 inlines x 40 crosslines x 200 samples, rotated grid, inline-sorted, IBM
        nil, nxl, ns = 30, 40, 200
        il = np.repeat(np.arange(1001, 1001 + 2 * nil, 2), nxl)
        xl = np.tile(np.arange(2001, 2001 + nxl), nil)
        t = np.arange(ns)
        cube = (np.sin(t[None, None, :] / 7.0 + il.reshape(nil, nxl)[:, :, None] / 20.0)
                * (1 + xl.reshape(nil, nxl)[:, :, None] / 1000.0) * 500).astype(np.float32)
        ang = np.radians(30)
        x = (600000 + 12.5 * (xl - 2001) * np.cos(ang) - 25 * (il - 1001) / 2 * np.sin(ang)) * 100
        y = (5600000 + 12.5 * (xl - 2001) * np.sin(ang) + 25 * (il - 1001) / 2 * np.cos(ang)) * 100
        hdr = {189: il, 193: xl, 181: x.astype(np.int64), 185: y.astype(np.int64)}
        p3 = os.path.join(tmp, "cube.sgy")
        write_segy(p3, cube.reshape(-1, ns), hdr, fmt=1)
        # scalar -100 at byte 71 (2 bytes)
        mm = np.memmap(p3, np.uint8, "r+")
        trl = 240 + ns * 4
        for k in range(nil * nxl):
            mm[3600 + k * trl + 70: 3600 + k * trl + 72] = np.frombuffer((-100).to_bytes(2, "big", signed=True), np.uint8)
        mm.flush()
        del mm
        f = SegyFile(p3)
        check(f.ntraces == nil * nxl and f.ns == ns and abs(f.dt - 0.004) < 1e-9, "SEG-Y header read")
        check(f.guess_il_xl() == (189, 193), "inline/crossline bytes auto-detected")
        v = import_3d(p3, os.path.join(tmp, "store"))
        check((v.nil, v.nxl, v.il_step) == (nil, nxl, 2), f"3D geometry {v.nil}x{v.nxl}, inline step {v.il_step}")
        err = np.abs(v.inline(1011) - cube[5]).max() / 500
        check(err < 2e-3, f"inline matches input (rel err {err:.1e})")
        err = np.abs(v.xline(2010) - cube[:, 9]).max() / 500
        check(err < 2e-3, f"crossline matches input (rel err {err:.1e})")
        err = np.abs(v.time_slice(400.0) - cube[:, :, 100]).max() / 500
        check(err < 2e-3, f"time slice matches input (rel err {err:.1e})")
        xx, yy = v.xy(1021, 2030)
        k = (1021 - 1001) // 2 * nxl + (2030 - 2001)
        check(abs(xx - x[k] / 100) < 0.5 and abs(yy - y[k] / 100) < 0.5, "trace coordinates (rotated grid, scalar -100)")
        i2, x2 = v.ilxl_at(xx, yy)
        check(abs(i2 - 1021) < 0.01 and abs(x2 - 2030) < 0.01, "map position -> inline/crossline")
        sec, ils, xls = v.arbitrary([xx, v.xy(1041, 2010)[0]], [yy, v.xy(1041, 2010)[1]])
        check(sec.shape[1] == ns and ils[0] == 1021 and xls[-1] == 2010, "arbitrary line")
        # ---- 2D line, IEEE, cdp at byte 21
        tr2 = np.random.default_rng(0).standard_normal((120, 300)).astype(np.float32)
        p2 = os.path.join(tmp, "line.sgy")
        write_segy(p2, tr2, {21: np.arange(1, 121)}, fmt=5)
        nav = np.array([[1, 0.0, 0.0], [120, 1190.0, 0.0]])
        l = Line2D("L", p2, nav=nav)
        check(np.allclose(l.data(), tr2) and l.has_coords and abs(l.x[60] - 600) < 1e-6, "2D line + nav")
        # ---- horizons: snapping, 3D/2D autotracking, ties, undo, save/load
        from . import test_horizons
        _guarded(check, 'test_horizons.run(check, tmp)', lambda: test_horizons.run(check, tmp))
        # ---- faults: sticks, surfaces, section cuts, tracker barrier
        from . import test_faults
        _guarded(check, 'test_faults.run(check, tmp)', lambda: test_faults.run(check, tmp))
        # ---- RGT step A: dip field
        from . import test_dip
        _guarded(check, 'test_dip.run(check, tmp)', lambda: test_dip.run(check, tmp))
        # ---- RGT step B: mesh + global solve
        from . import test_rgt
        _guarded(check, 'test_rgt.run(check, tmp)', lambda: test_rgt.run(check, tmp))
        # ---- RGT step C: horizon stack, Wheeler, stratal slices
        from . import test_stack
        _guarded(check, 'test_stack.run(check, tmp)', lambda: test_stack.run(check, tmp))
        # ---- RGT step D: local editing
        from . import test_edit
        _guarded(check, 'test_edit.run(check, tmp)', lambda: test_edit.run(check, tmp))
        # ---- wells on seismic: time-depth, tops, synthetics, ties
        from . import test_welltie
        _guarded(check, 'test_welltie.run(check, tmp)', lambda: test_welltie.run(check, tmp))
        # ---- horizon maps
        from . import test_maps
        _guarded(check, 'test_maps.run(check, tmp)', lambda: test_maps.run(check, tmp))
        # ---- 2D-3D / 2D-2D mistie
        from . import test_mistie
        _guarded(check, 'test_mistie.run(check, tmp)', lambda: test_mistie.run(check, tmp))
        # ---- seismic attributes
        from . import test_attributes
        _guarded(check, 'test_attributes.run(check, tmp)', lambda: test_attributes.run(check, tmp))
        # ---- velocity model + depth conversion
        from . import test_velocity
        _guarded(check, 'test_velocity.run(check, tmp)', lambda: test_velocity.run(check, tmp))
        # ---- dynamic depth conversion (item 17)
        from . import test_ddc
        os.makedirs(os.path.join(tmp, "ddc"))
        _guarded(check, 'test_ddc.run(check, os.path.join(tmp, "ddc"))', lambda: test_ddc.run(check, os.path.join(tmp, "ddc")))
        _guarded(check, 'test_ddc.run_dynamic(check, os.path.join(tmp, "ddc"))', lambda: test_ddc.run_dynamic(check, os.path.join(tmp, "ddc")))
        _guarded(check, 'test_ddc.run_gridding(check, os.path.join(tmp, "ddc"))', lambda: test_ddc.run_gridding(check, os.path.join(tmp, "ddc")))
        _guarded(check, 'test_ddc.run_prognosis(check, os.path.join(tmp, "ddc"))', lambda: test_ddc.run_prognosis(check, os.path.join(tmp, "ddc")))
        _guarded(check, 'test_ddc.run_scenarios(check, os.path.join(tmp, "ddc"))', lambda: test_ddc.run_scenarios(check, os.path.join(tmp, "ddc")))
        # ---- analytics
        from . import test_analytics
        _guarded(check, 'test_analytics.run(check, tmp)', lambda: test_analytics.run(check, tmp))
        from . import test_engineering
        _guarded(check, 'test_engineering.run(check, tmp)', lambda: test_engineering.run(check, tmp))
        from . import test_models
        _guarded(check, 'test_models.run(check, tmp)', lambda: test_models.run(check, tmp))
        from . import test_explain
        _guarded(check, 'test_explain.run(check, tmp)', lambda: test_explain.run(check, tmp))
        from . import test_qc
        _guarded(check, 'test_qc.run(check, tmp)', lambda: test_qc.run(check, tmp))
        from . import test_laterals
        _guarded(check, 'test_laterals.run(check, tmp)', lambda: test_laterals.run(check, tmp))
        from . import test_hardening
        _guarded(check, 'test_hardening.run(check, tmp)', lambda: test_hardening.run(check, tmp))
        from . import test_tie
        os.makedirs(os.path.join(tmp, "tie16"))
        _guarded(check, 'test_tie.run(check, os.path.join(tmp, "tie16"))', lambda: test_tie.run(check, os.path.join(tmp, "tie16")))
        from . import test_basin
        os.makedirs(os.path.join(tmp, "basin"))
        _guarded(check, 'test_basin.run(check, os.path.join(tmp, "basin"))', lambda: test_basin.run(check, os.path.join(tmp, "basin")))
        from . import test_inversion
        os.makedirs(os.path.join(tmp, "inversion"))
        _guarded(check, 'test_inversion.run(check, os.path.join(tmp, "inversion"))', lambda: test_inversion.run(check, os.path.join(tmp, "inversion")))
        from . import test_maplayers
        _guarded(check, 'test_maplayers.run(check, tmp)', lambda: test_maplayers.run(check, tmp))
        _guarded(check, 'test_maplayers.run_hub(check, tmp)', lambda: test_maplayers.run_hub(check, tmp))
        _guarded(check, 'test_maplayers.run_carto(check, tmp)', lambda: test_maplayers.run_carto(check, tmp))
        from . import test_propgrid
        _guarded(check, 'test_propgrid.run(check, tmp)', lambda: test_propgrid.run(check, tmp))
        from . import test_addwells
        os.makedirs(os.path.join(tmp, "addwells"))
        _guarded(check, 'test_addwells.run(check, os.path.join(tmp, "addwells"))', lambda: test_addwells.run(check, os.path.join(tmp, "addwells")))
        from . import test_crosscheck
        _guarded(check, 'test_crosscheck.run(check, tmp)', lambda: test_crosscheck.run(check, tmp))
        from . import test_gis
        _guarded(check, 'test_gis.run(check, tmp)', lambda: test_gis.run(check, tmp))
        _guarded(check, 'test_gis.run_export(check, tmp)', lambda: test_gis.run_export(check, tmp))
        from . import test_qgislink
        _guarded(check, 'test_qgislink.run(check, tmp)', lambda: test_qgislink.run(check, tmp))
        from . import test_blend
        _guarded(check, 'test_blend.run(check, tmp)', lambda: test_blend.run(check, tmp))
        from . import test_resolution
        _guarded(check, 'test_resolution.run(check, tmp)', lambda: test_resolution.run(check, tmp))
        _guarded(check, 'test_resolution.run_seeds(check, tmp)', lambda: test_resolution.run_seeds(check, tmp))
        _guarded(check, 'test_resolution.run_predictive(check, tmp)', lambda: test_resolution.run_predictive(check, tmp))
        from . import test_avo
        _guarded(check, 'test_avo.run(check, tmp)', lambda: test_avo.run(check, tmp))
        from . import test_structure
        _guarded(check, 'test_structure.run(check, tmp)', lambda: test_structure.run(check, tmp))
        from . import test_migration
        _guarded(check, 'test_migration.run(check, tmp)', lambda: test_migration.run(check, tmp))
        from . import test_prospect
        _guarded(check, 'test_prospect.run(check, tmp)', lambda: test_prospect.run(check, tmp))
        from . import test_spacing
        _guarded(check, 'test_spacing.run(check, tmp)', lambda: test_spacing.run(check, tmp))
        from . import test_depletion
        _guarded(check, 'test_depletion.run(check, tmp)', lambda: test_depletion.run(check, tmp))
        from . import test_mapbus
        _guarded(check, 'test_mapbus.run(check, tmp)', lambda: test_mapbus.run(check, tmp))
        from . import test_livemap
        _guarded(check, 'test_livemap.run(check, tmp)', lambda: test_livemap.run(check, tmp))
        _guarded(check, 'test_livemap.run_culture(check, tmp)', lambda: test_livemap.run_culture(check, tmp))
        _guarded(check, 'test_livemap.run_seismic(check, tmp)', lambda: test_livemap.run_seismic(check, tmp))
        _guarded(check, 'test_livemap.run_wellsection(check, tmp)', lambda: test_livemap.run_wellsection(check, tmp))
        _guarded(check, 'test_livemap.run_livegrid(check, tmp)', lambda: test_livemap.run_livegrid(check, tmp))
        _guarded(check, 'test_livemap.run_livedepth(check, os.path.join(tmp, "livedepth"))', lambda: test_livemap.run_livedepth(check, os.path.join(tmp, "livedepth")))
        _guarded(check, 'test_livemap.run_output(check, tmp)', lambda: test_livemap.run_output(check, tmp))
        _guarded(check, 'test_livemap.run_paths(check, tmp)', lambda: test_livemap.run_paths(check, tmp))
        from . import test_faultlik
        _guarded(check, 'test_faultlik.run(check, tmp)', lambda: test_faultlik.run(check, tmp))
        _guarded(check, 'test_faultlik.run_volume(check, tmp)', lambda: test_faultlik.run_volume(check, tmp))
        _guarded(check, 'test_faultlik.run_skins(check, tmp)', lambda: test_faultlik.run_skins(check, tmp))
        _guarded(check, 'test_faultlik.run_throw_2d(check, tmp)', lambda: test_faultlik.run_throw_2d(check, tmp))
        from . import test_velvolume
        _guarded(check, 'test_velvolume.run(check, os.path.join(tmp, "vv"))', lambda: test_velvolume.run(check, os.path.join(tmp, "vv")))
        from . import test_scene3d
        _guarded(check, 'test_scene3d.run(check, tmp)', lambda: test_scene3d.run(check, tmp))
        from . import test_geobody
        _guarded(check, 'test_geobody.run(check, tmp)', lambda: test_geobody.run(check, tmp))
        _guarded(check, 'test_geobody.run_volume(check, tmp)', lambda: test_geobody.run_volume(check, tmp))
        from . import test_framework
        _guarded(check, 'test_framework.run(check, tmp)', lambda: test_framework.run(check, tmp))
        from . import test_propmodel
        _guarded(check, 'test_propmodel.run(check, tmp)', lambda: test_propmodel.run(check, tmp))
        from . import test_pcfixes
        _guarded(check, 'test_pcfixes.run(check, tmp)', lambda: test_pcfixes.run(check, tmp))
        _guarded(check, 'test_pcfixes.run_tutorial(check, tmp)', lambda: test_pcfixes.run_tutorial(check, tmp))
        from . import test_glow
        _guarded(check, 'test_glow.run(check, tmp)', lambda: test_glow.run(check, os.path.join(tmp, "glow")))
        from . import test_aesthetics
        _guarded(check, 'test_aesthetics.run(check, tmp)', lambda: test_aesthetics.run(check, os.path.join(tmp, "aes")))
        from . import test_conditioning
        _guarded(check, 'test_conditioning.run(check, tmp)', lambda: test_conditioning.run(check, os.path.join(tmp, "cond")))
        _guarded(check, 'test_conditioning.run_volume_test(check, tmp)', lambda: test_conditioning.run_volume_test(check, os.path.join(tmp, "condvol")))
        from . import test_surveyblend
        _guarded(check, 'test_surveyblend.run(check, tmp)', lambda: test_surveyblend.run(check, os.path.join(tmp, "sblend")))
        from . import test_retest
        _guarded(check, 'test_retest.run(check, tmp)', lambda: test_retest.run(check, os.path.join(tmp, "retest")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    return ok


def run_gui():
    """GUI logic tests. Here: PYTHONPATH=<repo>/dev/fakeqt:. python3 -m seismic_tool --guitest.
    On a PC with Qt it runs against real widgets (offscreen)."""
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok &= bool(cond)

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    from . import test_horizons
    tmp = tempfile.mkdtemp(prefix="seis_guitest_")
    try:
        _guarded(check, 'test_horizons.run_gui(check, tmp)', lambda: test_horizons.run_gui(check, tmp))
        from . import test_faults
        _guarded(check, 'test_faults.run_gui(check, tmp)', lambda: test_faults.run_gui(check, tmp))
        from . import test_dip
        _guarded(check, 'test_dip.run_gui(check, tmp)', lambda: test_dip.run_gui(check, tmp))
        from . import test_rgt
        _guarded(check, 'test_rgt.run_gui(check, tmp)', lambda: test_rgt.run_gui(check, tmp))
        from . import test_stack
        _guarded(check, 'test_stack.run_gui(check, tmp)', lambda: test_stack.run_gui(check, tmp))
        from . import test_edit
        _guarded(check, 'test_edit.run_gui(check, tmp)', lambda: test_edit.run_gui(check, tmp))
        from . import test_welltie
        _guarded(check, 'test_welltie.run_gui(check, tmp)', lambda: test_welltie.run_gui(check, tmp))
        from . import test_maps
        _guarded(check, 'test_maps.run_gui(check, tmp)', lambda: test_maps.run_gui(check, tmp))
        from . import test_mistie
        _guarded(check, 'test_mistie.run_gui(check, tmp)', lambda: test_mistie.run_gui(check, tmp))
        from . import test_attributes
        _guarded(check, 'test_attributes.run_gui(check, tmp)', lambda: test_attributes.run_gui(check, tmp))
        from . import test_velocity
        os.makedirs(os.path.join(tmp, "velgui"))
        _guarded(check, 'test_velocity.run_gui(check, os.path.join(tmp, "velgui"))', lambda: test_velocity.run_gui(check, os.path.join(tmp, "velgui")))
        from . import test_ddc
        os.makedirs(os.path.join(tmp, "ddcgui"))
        _guarded(check, 'test_ddc.run_gui(check, os.path.join(tmp, "ddcgui"))', lambda: test_ddc.run_gui(check, os.path.join(tmp, "ddcgui")))
        _guarded(check, 'test_ddc.run_gui_dynamic(check, os.path.join(tmp, "ddcgui"))', lambda: test_ddc.run_gui_dynamic(check, os.path.join(tmp, "ddcgui")))
        _guarded(check, 'test_ddc.run_gui_gridding(check, os.path.join(tmp, "ddcgui"))', lambda: test_ddc.run_gui_gridding(check, os.path.join(tmp, "ddcgui")))
        _guarded(check, 'test_ddc.run_gui_prognosis(check, os.path.join(tmp, "ddcgui"))', lambda: test_ddc.run_gui_prognosis(check, os.path.join(tmp, "ddcgui")))
        _guarded(check, 'test_ddc.run_gui_scenarios(check, os.path.join(tmp, "ddcgui"))', lambda: test_ddc.run_gui_scenarios(check, os.path.join(tmp, "ddcgui")))
        from . import test_explain
        os.makedirs(os.path.join(tmp, "angui"))
        _guarded(check, 'test_explain.run_gui(check, os.path.join(tmp, "angui"))', lambda: test_explain.run_gui(check, os.path.join(tmp, "angui")))
        from . import test_qc
        _guarded(check, 'test_qc.run_gui(check, tmp)', lambda: test_qc.run_gui(check, tmp))
        from . import test_engineering
        _guarded(check, 'test_engineering.run_gui(check, tmp)', lambda: test_engineering.run_gui(check, tmp))
        from . import test_laterals
        _guarded(check, 'test_laterals.run_gui(check, tmp)', lambda: test_laterals.run_gui(check, tmp))
        from . import test_hardening
        _guarded(check, 'test_hardening.run_gui(check, tmp)', lambda: test_hardening.run_gui(check, tmp))
        from . import test_tie
        _guarded(check, 'test_tie.run_gui(check, tmp)', lambda: test_tie.run_gui(check, tmp))
        from . import test_basin
        os.makedirs(os.path.join(tmp, "basingui"))
        _guarded(check, 'test_basin.run_gui(check, os.path.join(tmp, "basingui"))', lambda: test_basin.run_gui(check, os.path.join(tmp, "basingui")))
        from . import test_inversion
        os.makedirs(os.path.join(tmp, "invgui"))
        _guarded(check, 'test_inversion.run_gui(check, os.path.join(tmp, "invgui"))', lambda: test_inversion.run_gui(check, os.path.join(tmp, "invgui")))
        from . import test_maplayers
        _guarded(check, 'test_maplayers.run_gui(check, tmp)', lambda: test_maplayers.run_gui(check, tmp))
        _guarded(check, 'test_maplayers.run_gui_hub(check, tmp)', lambda: test_maplayers.run_gui_hub(check, tmp))
        from . import test_propgrid
        _guarded(check, 'test_propgrid.run_gui(check, tmp)', lambda: test_propgrid.run_gui(check, tmp))
        from . import test_addwells
        os.makedirs(os.path.join(tmp, "addwellsgui"))
        _guarded(check, 'test_addwells.run_gui(check, os.path.join(tmp, "addwellsgui"))', lambda: test_addwells.run_gui(check, os.path.join(tmp, "addwellsgui")))
        from . import test_crosscheck
        os.makedirs(os.path.join(tmp, "ccgui"))
        _guarded(check, 'test_crosscheck.run_gui(check, os.path.join(tmp, "ccgui"))', lambda: test_crosscheck.run_gui(check, os.path.join(tmp, "ccgui")))
        from . import test_gis
        _guarded(check, 'test_gis.run_gui(check, tmp)', lambda: test_gis.run_gui(check, tmp))
        from . import test_qgislink
        _guarded(check, 'test_qgislink.run_gui(check, tmp)', lambda: test_qgislink.run_gui(check, tmp))
        from . import test_blend
        _guarded(check, 'test_blend.run_gui(check, tmp)', lambda: test_blend.run_gui(check, tmp))
        from . import test_structure
        _guarded(check, 'test_structure.run_gui(check, tmp)', lambda: test_structure.run_gui(check, tmp))
        from . import test_migration
        _guarded(check, 'test_migration.run_gui(check, tmp)', lambda: test_migration.run_gui(check, tmp))
        from . import test_spacing
        _guarded(check, 'test_spacing.run_gui(check, tmp)', lambda: test_spacing.run_gui(check, tmp))
        from . import test_mapbus
        _guarded(check, 'test_mapbus.run_gui(check, tmp)', lambda: test_mapbus.run_gui(check, tmp))
        from . import test_livemap
        _guarded(check, 'test_livemap.run_gui(check, tmp)', lambda: test_livemap.run_gui(check, tmp))
        _guarded(check, 'test_livemap.run_gui_culture(check, tmp)', lambda: test_livemap.run_gui_culture(check, tmp))
        _guarded(check, 'test_livemap.run_gui_seismic(check, tmp)', lambda: test_livemap.run_gui_seismic(check, tmp))
        _guarded(check, 'test_livemap.run_gui_wellsection(check, tmp)', lambda: test_livemap.run_gui_wellsection(check, tmp))
        _guarded(check, 'test_livemap.run_gui_livegrid(check, tmp)', lambda: test_livemap.run_gui_livegrid(check, tmp))
        _guarded(check, 'test_livemap.run_gui_output(check, tmp)', lambda: test_livemap.run_gui_output(check, tmp))
        from . import test_faultlik
        _guarded(check, 'test_faultlik.run_gui(check, tmp)', lambda: test_faultlik.run_gui(check, tmp))
        from . import test_velvolume
        _guarded(check, 'test_velvolume.run_gui(check, os.path.join(tmp, "vvgui"))', lambda: test_velvolume.run_gui(check, os.path.join(tmp, "vvgui")))
        from . import test_scene3d
        _guarded(check, 'test_scene3d.run_gui(check, tmp)', lambda: test_scene3d.run_gui(check, tmp))
        from . import test_geobody
        _guarded(check, 'test_geobody.run_gui(check, tmp)', lambda: test_geobody.run_gui(check, tmp))
        from . import test_framework
        _guarded(check, 'test_framework.run_gui(check, tmp)', lambda: test_framework.run_gui(check, tmp))
        from . import test_propmodel
        _guarded(check, 'test_propmodel.run_gui(check, tmp)', lambda: test_propmodel.run_gui(check, tmp))
        from . import test_glow
        _guarded(check, 'test_glow.run_gui(check, tmp)', lambda: test_glow.run_gui(check, os.path.join(tmp, "glowgui")))
        from . import test_aesthetics
        _guarded(check, 'test_aesthetics.run_gui(check, tmp)', lambda: test_aesthetics.run_gui(check, os.path.join(tmp, "aesgui")))
        from . import test_conditioning
        _guarded(check, 'test_conditioning.run_gui(check, tmp)', lambda: test_conditioning.run_gui(check, os.path.join(tmp, "condgui")))
        from . import test_surveyblend
        _guarded(check, 'test_surveyblend.run_gui(check, tmp)', lambda: test_surveyblend.run_gui(check, os.path.join(tmp, "sblendgui")))
        from . import test_retest
        _guarded(check, 'test_retest.run_gui(check, tmp)', lambda: test_retest.run_gui(check, os.path.join(tmp, "retestgui")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("GUI TEST PASSED" if ok else "GUI TEST FAILED")
    return ok
