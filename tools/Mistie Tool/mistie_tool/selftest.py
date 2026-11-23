"""Synthetic end-to-end test: build a grid of SEG-Y lines with known misties and check recovery."""
import json, os, tempfile
import numpy as np
from .measure import fourier_shift, phase_rotate


def _write_segy(path, data, dt, x, y, cdp):
    ntr, ns = data.shape
    text = (b"C 1 SYNTHETIC LINE FOR MISTIE_TOOL SELFTEST".ljust(80)) * 40
    binh = bytearray(400)
    binh[16:18] = int(dt * 1e6).to_bytes(2, "big"); binh[20:22] = ns.to_bytes(2, "big")
    binh[24:26] = (5).to_bytes(2, "big")
    hdr = np.zeros((ntr, 240), np.uint8)
    put = lambda col, vals, n: hdr.__setitem__((slice(None), slice(col - 1, col - 1 + n)),
                                               np.asarray(vals, ">i4" if n == 4 else ">i2").view(np.uint8).reshape(ntr, n))
    put(1, np.arange(1, ntr + 1), 4); put(21, cdp, 4); put(71, np.full(ntr, 1), 2)
    put(181, np.round(x).astype(int), 4); put(185, np.round(y).astype(int), 4)
    body = np.asarray(data, ">f4").view(np.uint8).reshape(ntr, -1)
    with open(path, "wb") as f:
        f.write(text); f.write(bytes(binh)); f.write(np.concatenate([hdr, body], axis=1).tobytes())


def run(verbose=True):
    rng = np.random.default_rng(7)
    dt, ns, ntr, spacing = 0.004, 750, 400, 25.0
    refl = rng.normal(0, 1, ns) * (rng.random(ns) < 0.06)
    tw = np.arange(-25, 26) * dt
    ricker = (1 - 2 * (np.pi * 25 * tw) ** 2) * np.exp(-(np.pi * 25 * tw) ** 2)
    earth = np.convolve(refl, ricker, "same")
    # true errors (shift ms, phase deg, gain) applied to each line; L1 is the reference
    truth = {"L1": (0, 0, 1.0), "L2": (9, 30, 0.8), "L3": (-6, -45, 1.25), "L4": (14, 170, 1.0), "L5": (-3, 10, 0.9)}
    lines = {  # straight lines in metres: L1,L2 north-south; L3,L4,L5 east-west
        "L1": ((2000, 0), (2000, 10000)), "L2": ((7000, 0), (7000, 10000)),
        "L3": ((0, 2500), (10000, 2500)), "L4": ((0, 5000), (10000, 5000)), "L5": ((0, 7500), (10000, 7500))}
    d = tempfile.mkdtemp(prefix="mistie_selftest_")
    for name, (p0, p1) in lines.items():
        n = int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / spacing)
        f = np.linspace(0, 1, n)
        x, y = p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1])
        s, p, g = truth[name]
        tr = phase_rotate(fourier_shift(earth, -s / 1000 / dt), -p) / g
        data = np.tile(tr, (n, 1)) + rng.normal(0, 0.08 * earth.std(), (n, ns))
        _write_segy(os.path.join(d, name + ".sgy"), data.astype(np.float32), dt, x, y, np.arange(1, n + 1))
    cfg = {"data_dir": ".", "output_dir": "out", "reference": "L1", "default_window": [0.3, 2.7],
           "lines": [{"name": n, "segy": n + ".sgy"} for n in lines]}
    cp = os.path.join(d, "cfg.json"); json.dump(cfg, open(cp, "w"))
    from .project import Project
    from .segy import SegyLine
    P = Project(cp); P.solve(); P.save()
    ok = len(P.ties) == 6
    msgs = [f"found {len(P.ties)} ties (expected 6)"]
    for n, (s, p, g) in truth.items():
        c = P.sol[n]
        ds, dp, dg = c["shift"] - s, (c["phase"] - p + 180) % 360 - 180, c["gain"] / g - 1
        good = abs(ds) < 1.0 and abs(dp) < 10 and abs(dg) < 0.05
        ok &= good
        msgs.append(f"{n}: recovered {c['shift']:+6.2f} ms {c['phase']:+6.1f} deg gain {c['gain']:.3f} | "
                    f"truth {s:+3d} ms {p:+4d} deg {g:.2f}  {'ok' if good else 'FAIL'}")
    out = P.write_corrected_segy()
    back = SegyLine(out[1])
    rt = back.data.shape == P.lines["L2"].data.shape and back.fmt == 5
    ok &= rt
    msgs.append(f"corrected SEG-Y write/read round trip {'ok' if rt else 'FAIL'}")
    if verbose:
        print("\n".join(msgs)); print("SELFTEST", "PASSED" if ok else "FAILED", "(files in", d + ")")
    return ok
