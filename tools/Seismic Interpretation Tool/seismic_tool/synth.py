"""Synthetic seismic for tests: Ricker reflectors at exact (sub-sample) times, written as SEG-Y and imported."""
import os

import numpy as np


def ricker(t_s, f_hz):
    a = (np.pi * f_hz * t_s) ** 2
    return (1 - 2 * a) * np.exp(-a)


def traces_from_surfaces(surfaces_ms, amps, ns, dt_s, f_hz=30.0, noise=0.0, seed=0):
    """surfaces_ms: list of arrays (any common shape S) of reflector times; amps: list of scalars or arrays.
    Returns float32 array S + (ns,)."""
    t = np.arange(ns) * dt_s
    shape = np.shape(surfaces_ms[0])
    out = np.zeros(shape + (ns,), np.float32)
    for s, a in zip(surfaces_ms, amps):
        s = np.asarray(s, float)[..., None] / 1000.0
        out += (np.asarray(a, float)[..., None] if np.ndim(a) else a) * ricker(t - s, f_hz).astype(np.float32)
    if noise:
        out += noise * np.random.default_rng(seed).standard_normal(out.shape).astype(np.float32)
    return out


def folded_surfaces(nil, nxl):
    """Two folded/dipping reflectors (ms) on an (nil, nxl) grid: a dome + regional dip, and a deeper one."""
    i, j = np.meshgrid(np.arange(nil), np.arange(nxl), indexing="ij")
    top = 300 + 0.9 * j + 0.4 * i - 40 * np.exp(-(((i - nil * 0.5) / (nil * 0.3)) ** 2 + ((j - nxl * 0.45) / (nxl * 0.3)) ** 2))
    deep = top + 120 + 10 * np.sin(j / 9.0)
    return top, deep


def write_3d(path, cube, il0=1001, xl0=2001, dt_us=4000, fmt=5):
    from .selftest import write_segy
    nil, nxl, ns = cube.shape
    il = np.repeat(np.arange(il0, il0 + nil), nxl)
    xl = np.tile(np.arange(xl0, xl0 + nxl), nil)
    x = 500000 + 25.0 * (xl - xl0)
    y = 6000000 + 25.0 * (il - il0)
    write_segy(path, cube.reshape(-1, ns), {189: il, 193: xl, 181: x.astype(np.int64), 185: y.astype(np.int64)},
               dt_us=dt_us, fmt=fmt)


def make_project(folder, nil=50, nxl=70, ns=250, dt_us=4000, noise=0.05, lines=True):
    """Synthetic project in folder: 3D 'Synth' with two folded reflectors, and two crossing 2D lines carrying
    the same top reflector (time = 1-D profile).  Returns (project, truth dict)."""
    from .project import SeisProject
    from .selftest import write_segy
    from .volume import import_3d
    dt = dt_us / 1e6
    top, deep = folded_surfaces(nil, nxl)
    cube = traces_from_surfaces([top, deep], [1.0, -0.7], ns, dt, noise=noise)
    p3 = os.path.join(folder, "synth3d.sgy")
    write_3d(p3, cube, dt_us=dt_us)
    store = os.path.join(folder, "synth_seismic", "Synth")
    import_3d(p3, store)
    proj = SeisProject()
    proj.path = os.path.join(folder, "synth.seisproj.json")
    proj.name = "synth"
    proj.volumes.append({"name": "Synth", "store": store, "segy": p3})
    truth = {"3D:Synth": top.astype(np.float32), "deep": deep}
    if lines:
        n = 120
        # EW line along y = 1000, NS line along x = 1500 (crossing at cdp 60 / 40)
        tA = 400 + 0.6 * np.arange(n) + 15 * np.sin(np.arange(n) / 15.0)
        tB = 400 + 0.6 * 60 + 15 * np.sin(60 / 15.0) + 0.8 * (np.arange(n) - 40)
        for name, tt, xy in (("L-A", tA, (np.arange(n) * 25.0, np.full(n, 1000.0))),
                             ("L-B", tB, (np.full(n, 1500.0), 1000.0 + (np.arange(n) - 40) * 25.0))):
            tr = traces_from_surfaces([tt, tt + 150], [1.0, 0.8], ns, dt, noise=noise, seed=len(name))
            pth = os.path.join(folder, name + ".sgy")
            write_segy(pth, tr, {21: np.arange(1, n + 1), 181: xy[0].astype(np.int64),
                                 185: xy[1].astype(np.int64)}, dt_us=dt_us)
            proj.lines.append({"name": name, "segy": pth, "cdp_byte": 21})
            truth["2D:" + name] = tt.astype(np.float32)
    proj.save()
    return proj, truth


def faulted_cube(nil=40, nxl=60, ns=250, dt_s=0.004, throw_ms=40.0, spacing_ms=40.0, noise=0.03, seed=1):
    """Layer-cake of identical reflectors every spacing_ms, cut by a planar normal fault that dips towards
    +crossline.  Fault position: xl index xf(t) = 25 + 0.04 * (t_ms - 200) + 0.1 * il.  Hanging wall (xl > xf)
    is shifted down by throw_ms - with throw = spacing a tracker that ignores the fault jumps to the wrong
    reflector.  Returns (cube, fault_xl(il, t) function, footwall top surface ms (nil, nxl) for the reflector
    at 300 ms, hanging-wall truth)."""
    i, j = np.meshgrid(np.arange(nil), np.arange(nxl), indexing="ij")
    base = 300 + 0.3 * j + 0.2 * i
    t = np.arange(ns) * dt_s * 1000.0
    fxl = lambda il, tm: 25 + 0.04 * (tm - 200) + 0.1 * il  # noqa: E731
    surf = [base + k * spacing_ms for k in range(-4, 8)]
    foot = traces_from_surfaces(surf, [1.0] * len(surf), ns, dt_s, noise=noise, seed=seed)
    hang = traces_from_surfaces([s + throw_ms for s in surf], [1.0] * len(surf), ns, dt_s, noise=noise,
                                seed=seed + 1)
    is_hang = j[..., None] > fxl(i[..., None], t[None, None, :])
    cube = np.where(is_hang, hang, foot).astype(np.float32)
    return cube, fxl, base


def fault_sticks_from(fxl, vol, ils_idx, t_ms=(150.0, 600.0)):
    """Sticks (map x, y, t) along the synthetic fault on the given inline indices."""
    sticks = []
    for i in ils_idx:
        pts = []
        for tm in t_ms:
            xl = vol.xl0 + vol.xl_step * fxl(i, tm)
            x, y = vol.xy(vol.il0 + vol.il_step * i, xl)
            pts.append((float(x), float(y), tm))
        sticks.append(pts)
    return sticks
