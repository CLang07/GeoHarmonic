"""Benchmarks (PLAN item 15): synthetic SEG-Y of a chosen size -> import, slice latency, arbitrary line, peak RAM;
10,000-well index; 1,000 coloured laterals.  Appends a table to docs/BENCHMARKS.md.

    python dev/bench.py --gb 2 --dir D:\\bench          (keep --dir on a disk with 3x the size free)
"""
import argparse
import os
import platform
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools", "Seismic Interpretation Tool"))


def peak_ram_mb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024.0 if sys.platform != "darwin" else r / 1e6
    except Exception:  # Windows: no resource module
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            fn = getattr(k32, "K32GetProcessMemoryInfo", None) or ctypes.WinDLL("psapi").GetProcessMemoryInfo
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            fn.restype = wintypes.BOOL
            if not fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return float("nan")
            return c.PeakWorkingSetSize / 2 ** 20
        except Exception:
            return float("nan")


def write_big_segy(path, nil, nxl, ns, dt_us=4000, chunk_il=8):
    """Streaming SEG-Y writer (IEEE float): never holds more than chunk_il inlines in memory."""
    from seismic_tool.synth import traces_from_surfaces
    with open(path, "wb") as f:
        f.write(("C01 GeoHarmonic benchmark" + " " * 55).encode("cp500") + b"\x40" * (3200 - 80))
        b = bytearray(400)
        b[16:18] = dt_us.to_bytes(2, "big")
        b[20:22] = ns.to_bytes(2, "big")
        b[24:26] = (5).to_bytes(2, "big")
        f.write(bytes(b))
        for a in range(0, nil, chunk_il):
            e = min(nil, a + chunk_il)
            i, j = np.meshgrid(np.arange(a, e), np.arange(nxl), indexing="ij")
            surf = [400 + 0.5 * i + 0.3 * j + 20 * np.sin(j / 40.0), 900 + 0.2 * i - 0.1 * j, 1500 + 0.1 * (i + j)]
            surf = [s for s in surf if s.max() < ns * dt_us / 1000 - 50] or [np.full(i.shape, ns * dt_us / 2000.0)]
            tr = traces_from_surfaces(surf, [1.0, -0.8, 0.6][:len(surf)], ns, dt_us / 1e6, noise=0.05, seed=a)
            n = tr.shape[0] * tr.shape[1]
            tr = tr.reshape(n, ns).astype(">f4")
            th = np.zeros((n, 240), np.uint8)
            il = (1000 + i).ravel()
            xl = (2000 + j).ravel()
            for byte, vals in ((189, il), (193, xl), (181, (500000 + 25 * (xl - 2000))), (185, (6000000 + 25 * (il - 1000)))):
                th[:, byte - 1: byte + 3] = np.asarray(vals, ">i4").view(np.uint8).reshape(n, 4)
            th[:, 114:116] = np.full(n, ns, ">i2").view(np.uint8).reshape(n, 2)
            f.write(np.concatenate([th, tr.view(np.uint8).reshape(n, -1)], axis=1).tobytes())


def timed(fn, *a, repeat=1, **k):
    t0 = time.perf_counter()
    for _ in range(repeat):
        r = fn(*a, **k)
    return (time.perf_counter() - t0) / repeat, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=0.5, help="SEG-Y size in GB")
    ap.add_argument("--ns", type=int, default=1000)
    ap.add_argument("--wells", type=int, default=10000)
    ap.add_argument("--laterals", type=int, default=1000)
    ap.add_argument("--dir", default=None, help="work folder (deleted afterwards unless --keep)")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs", "BENCHMARKS.md"))
    a = ap.parse_args()
    work = a.dir or tempfile.mkdtemp(prefix="gh_bench_")
    os.makedirs(work, exist_ok=True)
    ns = a.ns
    ntr = int(a.gb * 1e9 / (240 + 4 * ns))
    nxl = int(np.sqrt(ntr * 1.25))
    nil = max(1, ntr // nxl)
    res = []
    try:
        sgy = os.path.join(work, "bench.sgy")
        t, _ = timed(write_big_segy, sgy, nil, nxl, ns)
        size = os.path.getsize(sgy) / 1e9
        res.append(("write synthetic SEG-Y", f"{size:.2f} GB, {nil} x {nxl} x {ns}", t))
        from seismic_tool.volume import import_3d
        t, vol = timed(import_3d, sgy, os.path.join(work, "store"))
        res.append(("import 3D (two float16 stores)", f"{size / t * 1000:.0f} MB/s", t))
        rng = np.random.default_rng(0)
        t, _ = timed(lambda: [vol.inline(int(vol.inlines[k])) for k in rng.integers(0, vol.nil, 10)])
        res.append(("inline section (random, x10)", f"{t / 10 * 1000:.1f} ms each", t))
        t, _ = timed(lambda: [vol.xline(int(vol.xlines[k])) for k in rng.integers(0, vol.nxl, 10)])
        res.append(("crossline section (random, x10)", f"{t / 10 * 1000:.1f} ms each", t))
        t, _ = timed(lambda: [vol.time_slice(float(vol.times[k])) for k in rng.integers(0, vol.ns, 10)])
        res.append(("time slice (random, x10)", f"{t / 10 * 1000:.1f} ms each", t))
        x0, y0 = vol.xy(vol.inlines[0], vol.xlines[0])
        x1, y1 = vol.xy(vol.inlines[-1], vol.xlines[-1])
        t, _ = timed(vol.arbitrary, [float(x0), float(x1)], [float(y0), float(y1)])
        res.append(("arbitrary line corner to corner", "", t))
        from seismic_tool.wellindex import WellIndex
        xy = rng.uniform(0, 200000, (a.wells, 2))
        ix = WellIndex(":memory:", 5000.0)
        t, _ = timed(ix.add_wells, ((f"W{i}", None, x, y, 0.0, 3000.0) for i, (x, y) in enumerate(xy)))
        res.append((f"well index build ({a.wells:,} wells)", "", t))
        t, hits = timed(ix.in_box, 50000, 70000, 50000, 70000, repeat=20)
        res.append(("well index box query (view)", f"{len(hits)} wells", t))
        from seismic_tool import laterals as L
        t, lats = timed(L.synthetic_laterals, a.laterals, 0.0, 0.0, 60.0, 2000.0, 2500.0)
        res.append((f"build {a.laterals:,} synthetic laterals", "", t))
        vals = {w.name: w.z for w in lats}
        t, b = timed(L.batch_color_bins, lats, vals, 1990.0, 2010.0, 16, 25.0)
        res.append(("colour-batch laterals for the map (LOD 25 m)", f"{len(b)} curves", t))
        res.append(("peak RAM of the benchmark process", f"{peak_ram_mb():.0f} MB", float("nan")))
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)
    lines = [f"\n## {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} - {platform.system()} {platform.machine()}, "
             f"Python {platform.python_version()}, numpy {np.__version__}, --gb {a.gb}\n",
             "| step | detail | time (s) |", "|---|---|---|"]
    lines += [f"| {s} | {d} | {'' if t != t else f'{t:.3f}'} |" for s, d, t in res]
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        new = not os.path.exists(a.out)
        with open(a.out, "a", encoding="utf-8") as fh:
            if new:
                fh.write("# Benchmarks\n\nAppended by `python dev/bench.py` (see its --help). Compare runs on the "
                         "same machine; the cloud build box is not Chris's PC.\n")
            fh.write(text)


if __name__ == "__main__":
    main()
