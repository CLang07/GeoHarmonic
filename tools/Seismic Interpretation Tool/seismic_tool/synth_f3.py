"""Synthetic F3-like raw data (PLAN 43i): the same folder layout, names and coordinate frame as dGB's F3 Demo 2023
download, small enough for a test run.  `make_project(folder)` writes the raw folder and runs tools/setup_f3.py on it,
so the result is built exactly like Chris's real 'F3 Netherlands' project:

    volume 'F3 original' (inlines 100-750, crosslines 300-1250, every `step`-th line, real F3 bin grid: rotated
    ~1.6 deg, ED50 / UTM 31N), horizons Shallow, Truncation, FS8, FS7, FS6, FS4, MFS4, Fault A (sticks, 20 ms throw),
    wells F02-1, F03-2, F03-4, F06-1 (GR / DT / RHOB LAS, checkshots, markers = the horizons).

Everything is synthetic (dome + regional dip + fault); only the names and the survey frame follow F3."""
import os
import sys

import numpy as np

# real F3 bin grid (OpendTect .survey): x = a + b*il + c*xl, y = d + e*il + f*xl
AX = [598402.0, -0.698446, 24.991117]
AY = [6070840.0, 24.991117, 0.698446]
IL, XL = (100, 750), (300, 1250)
HORIZONS = [("Shallow", 480.0), ("Truncation", 620.0), ("FS8", 780.0), ("FS7", 900.0), ("FS6", 1000.0),
            ("FS4", 1120.0), ("MFS4", 1240.0)]
WELLS = {"F02-1": (606554.0, 6080126.0, 30.0), "F03-2": (619101.0, 6089491.0, 30.0),
         "F03-4": (623255.0, 6082586.0, 30.0), "F06-1": (607903.0, 6077213.0, 30.0)}
FAULT_XL = 800.0  # Fault A: dips east, downthrown to the east below Truncation
THROW_MS = 20.0
VEL = 2000.0  # m/s: z = v * t / 2 (simple and invertible for the checkshots)


def xy(il, xl):
    il, xl = np.asarray(il, float), np.asarray(xl, float)
    return AX[0] + AX[1] * il + AX[2] * xl, AY[0] + AY[1] * il + AY[2] * xl


def ilxl(x, y):
    M = np.array([[AX[1], AX[2]], [AY[1], AY[2]]])
    il, xl = np.linalg.solve(M, np.array([np.asarray(x, float) - AX[0], np.asarray(y, float) - AY[0]]))
    return il, xl


def fault_xl(t_ms):
    """Crossline of Fault A at time t (dips east: 0.15 xl per ms)."""
    return FAULT_XL + 0.15 * (np.asarray(t_ms, float) - 600.0)


def horizon_time(name, il, xl):
    """Synthetic time (ms) of a horizon at (il, xl): dome around (450, 700) + regional dip + Fault A throw."""
    t0 = dict(HORIZONS)[name]
    il, xl = np.asarray(il, float), np.asarray(xl, float)
    k = [n for n, _ in HORIZONS].index(name)
    dome = (25.0 + 8.0 * k) * np.exp(-(((il - 450.0) / 180.0) ** 2 + ((xl - 700.0) / 260.0) ** 2))
    t = t0 + 0.04 * (xl - 300.0) + 0.03 * (il - 100.0) - dome
    if k >= 2:  # below Truncation: faulted
        t = t + np.where(xl > fault_xl(t), THROW_MS, 0.0)
    return t


def _write_las(path, well, md, curves):
    with open(path, "w") as f:
        f.write("~Version\n VERS. 2.0 :\n WRAP. NO :\n~Well\n")
        f.write(f" STRT.M {md[0]:.2f} :\n STOP.M {md[-1]:.2f} :\n STEP.M {md[1] - md[0]:.4f} :\n NULL. -999.25 :\n")
        f.write(f" WELL. {well} :\n COMP. GeoHarmonic synthetic F3-like :\n")
        f.write("~Curve\n DEPT.M :\n" + "".join(f" {n}.{u} :\n" for n, u, _ in curves))
        f.write("~A\n")
        data = np.column_stack([md] + [v for _, _, v in curves])
        np.savetxt(f, data, fmt="%.4f")


def make_raw(folder, step=10, ns=463, dt_ms=4.0, seed=3):
    """Raw F3-like folder (what setup_f3.py reads). Returns the folder."""
    from .selftest import write_segy
    from .synth import ricker
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(folder, "Rawdata"), exist_ok=True)
    with open(os.path.join(folder, ".survey"), "w") as f:
        f.write("dTect V6.6\nSurvey Info\n!\nName: F3 Demo synthetic\n")
        f.write(f"In-line range: {IL[0]}`{IL[1]}`1\nCross-line range: {XL[0]}`{XL[1]}`1\n")
        f.write(f"Z range: 0`{(ns - 1) * dt_ms / 1000.0:g}`{dt_ms / 1000.0:g}\nZ Domain: Time\n")
        f.write(f"Coord-X-BinID: {AX[0]}`{AX[1]}`{AX[2]}\nCoord-Y-BinID: {AY[0]}`{AY[1]}`{AY[2]}\n")
        f.write("Coordinate System.Projection.ID: EPSG`23031\n")
    ils = np.arange(IL[0], IL[1] + 1, step)
    xls = np.arange(XL[0], XL[1] + 1, step)
    I, J = np.meshgrid(ils, xls, indexing="ij")
    t = np.arange(ns) * dt_ms
    cube = np.zeros(I.shape + (ns,), np.float32)
    amps = [0.8, -1.0, 0.7, -0.6, 1.0, -0.8, 0.9]
    for (name, _), a in zip(HORIZONS, amps):
        ht = horizon_time(name, I, J)
        cube += a * ricker((t[None, None, :] - ht[..., None]) / 1000.0, 28.0).astype(np.float32)
    cube += 0.05 * rng.standard_normal(cube.shape).astype(np.float32)
    X, Y = xy(I, J)
    write_segy(os.path.join(folder, "Rawdata", "Seismic_data.sgy"), cube.reshape(-1, ns),
               {189: I.ravel(), 193: J.ravel(), 181: np.rint(X.ravel()).astype(np.int64),
                185: np.rint(Y.ravel()).astype(np.int64)}, dt_us=int(dt_ms * 1000))
    sd = os.path.join(folder, "Rawdata", "Surface_data")
    os.makedirs(sd, exist_ok=True)
    hi, hj = np.meshgrid(np.arange(IL[0], IL[1] + 1, max(1, step // 2)), np.arange(XL[0], XL[1] + 1, max(1, step // 2)),
                         indexing="ij")
    hx, hy = xy(hi, hj)
    for name, _ in HORIZONS:
        np.savetxt(os.path.join(sd, f"F3-Horizon-{name}.txt"),
                   np.column_stack([hx.ravel(), hy.ravel(), horizon_time(name, hi, hj).ravel()]), fmt="%.2f")
    fd = os.path.join(folder, "Rawdata", "Faults")
    os.makedirs(fd, exist_ok=True)
    rows = []
    for k, il in enumerate(range(IL[0] + 20, IL[1] - 19, 60)):
        for tt in (600.0, 900.0, 1200.0, 1500.0):
            fx, fy = xy(il, fault_xl(tt))
            rows.append((float(fx), float(fy), tt, k))
    np.savetxt(os.path.join(fd, "FaultA.txt"), np.array(rows), fmt="%.2f")
    raw = os.path.join(folder, "Rawdata", "Well_data", "All_wells_RawData", "All_wells_RawData")
    for sub in ("Track", "Lasfiles", "Checkshot", "Tops"):
        os.makedirs(os.path.join(raw, sub), exist_ok=True)
    for w, (x, y, kb) in WELLS.items():
        td = 1900.0
        with open(os.path.join(raw, "Track", w + ".track"), "w") as f:
            for md in (0.0, td):
                f.write(f"{x:.2f} {y:.2f} {md - kb:.2f} {md:.2f}\n")
        il, xl = ilxl(x, y)
        tops = [(name, VEL * float(horizon_time(name, il, xl)) / 2000.0 + kb) for name, _ in HORIZONS]
        with open(os.path.join(raw, "Tops", w + "_markers.txt"), "w") as f:
            f.write(f"{kb:.2f}\tSeaSurface\n")
            for name, md in tops:
                f.write(f"{md:.2f}\t{name}\n")
        md = np.arange(30.0, td, 0.5)
        gr = 80 + 15 * np.sin(md / 23.0) + 2 * rng.standard_normal(len(md))
        vint = 1700.0 + 0.45 * md + 8.0 * np.sin(md / 37.0)  # m/s, compaction trend (logs vary: no flat lines)
        dt = 1e6 / vint / 3.2808 + 1.5 * rng.standard_normal(len(md))  # us/ft
        rho = 1.95 + 0.00025 * md + 0.004 * rng.standard_normal(len(md))
        for k, (name, mdt) in enumerate(tops):
            below = md >= mdt
            gr = np.where(below & (md < mdt + 12.0), 35.0 + k, gr)  # a clean sand under each marker
            rho = np.where(below, rho + 0.12 * amps[k], rho)  # AI step with the reflector's polarity
        _write_las(os.path.join(raw, "Lasfiles", w + "_logs.las"), w, md,
                   [("GR", "GAPI", gr), ("DT", "US/F", dt), ("RHOB", "G/C3", rho)])
        with open(os.path.join(raw, "Checkshot", w + "_TD.txt"), "w") as f:
            for mdc in np.arange(100.0, td, 150.0):
                f.write(f"{mdc:.1f} {(mdc - kb) * 2.0 / VEL:.5f}\n")
    return folder


def make_project(folder, step=10, quiet=True):
    """Raw folder in <folder>/F3_Demo_2023 + project in <folder>/F3 Netherlands via tools/setup_f3.py. Returns the
    .seisproj.json path."""
    raw = make_raw(os.path.join(folder, "F3_Demo_2023"), step=step)
    out = os.path.join(folder, "F3 Netherlands")
    tools = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import importlib
    setup_f3 = importlib.import_module("setup_f3")
    if quiet:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            setup_f3.main([raw, out])
    else:
        setup_f3.main([raw, out])
    return os.path.join(out, "F3 Netherlands.seisproj.json")
