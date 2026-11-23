"""Seismic attributes (numpy): complex-trace attributes, RMS, coherence (semblance, optionally dip-steered),
spectral decomposition, horizon curvature.

Two ways to use them:
  * on the fly for the section shown (inline / crossline / time slice / 2D line) - the section panel's display
    box; only the traces needed are read (neighbouring inlines for coherence, a time slab for time slices);
  * as a full volume: compute_volume() writes a normal volume store (both float16 copies) next to the input
    and adds it to the project as "<volume> [<attribute>]", so it can be viewed, mapped and extracted like
    any volume.
"""
import json
import os
import shutil

import numpy as np

from .mistie import analytic

# name -> (needs lateral neighbours, default parameters)
ATTRIBUTES = {
    "Envelope": (False, {}),
    "Inst. phase": (False, {}),
    "Inst. frequency": (False, {}),
    "Cosine of phase": (False, {}),
    "RMS amplitude": (False, {"window_ms": 40.0}),
    "Coherence": (True, {"window_ms": 24.0, "steered": False}),
    "Coherence (dip-steered)": (True, {"window_ms": 24.0, "steered": True}),
    "Spectral 10 Hz": (False, {"freq": 10.0}),
    "Spectral 20 Hz": (False, {"freq": 20.0}),
    "Spectral 30 Hz": (False, {"freq": 30.0}),
    "Spectral 40 Hz": (False, {"freq": 40.0}),
    "Spectral 60 Hz": (False, {"freq": 60.0}),
}


# ====================================================================== trace attributes (last axis = time)
def envelope(x):
    return np.abs(analytic(x, -1)).astype(np.float32)


def inst_phase(x):
    return np.degrees(np.angle(analytic(x, -1))).astype(np.float32)


def cos_phase(x):
    return np.cos(np.angle(analytic(x, -1))).astype(np.float32)


def inst_freq(x, dt_ms):
    """Hz from the phase advance between neighbouring samples, angle(z[n+1] conj(z[n-1])) / (2 dt 2 pi), with
    the complex products averaged over 5 samples (amplitude weighted, unbiased for a pure tone)."""
    z = analytic(x, -1)
    pr = np.zeros(z.shape, complex)
    pr[..., 1:-1] = z[..., 2:] * np.conj(z[..., :-2])
    pr[..., 0], pr[..., -1] = pr[..., 1], pr[..., -2]
    pr = running_mean(pr.real, 5) + 1j * running_mean(pr.imag, 5)
    return (np.angle(pr) / (2 * np.pi) / (2 * dt_ms / 1000.0)).astype(np.float32)


def running_mean(a, n, axis=-1):
    """Centred running mean over n samples along an axis (edges use the available samples)."""
    a = np.moveaxis(np.asarray(a, np.float64), axis, -1)
    h = n // 2
    cs = np.cumsum(np.pad(a, [(0, 0)] * (a.ndim - 1) + [(h + 1, h)], mode="constant"), axis=-1)
    cnt = np.cumsum(np.pad(np.ones(a.shape[-1]), (h + 1, h)))
    s = cs[..., n:] - cs[..., :-n]
    c = cnt[n:] - cnt[:-n]
    return np.moveaxis(s / c, -1, axis)


def rms(x, dt_ms, window_ms=40.0):
    n = max(1, int(round(window_ms / dt_ms)) | 1)
    return np.sqrt(running_mean(np.asarray(x, np.float64) ** 2, n)).astype(np.float32)


def spectral(x, dt_ms, freq, cycles=3.0):
    """Spectral decomposition: amplitude of the Morlet-like (Gaussian-windowed complex exponential) filter
    response at freq Hz - a constant-Q short-window FFT band."""
    sigma_ms = cycles / (2 * np.pi * freq) * 1000.0
    half = int(np.ceil(3 * sigma_ms / dt_ms))
    t = np.arange(-half, half + 1) * dt_ms
    w = np.exp(-0.5 * (t / sigma_ms) ** 2) * np.exp(2j * np.pi * freq * t / 1000.0)
    w /= np.abs(w).sum() / 2.0
    x = np.asarray(x, np.float64)
    flat = x.reshape(-1, x.shape[-1])
    n = flat.shape[1] + len(w) - 1
    nf = int(2 ** np.ceil(np.log2(n)))
    y = np.fft.ifft(np.fft.fft(flat, nf, axis=1) * np.fft.fft(w, nf)[None], axis=1)[:, half:half + flat.shape[1]]
    return np.abs(y).reshape(x.shape).astype(np.float32)


# ====================================================================== coherence
def semblance(block, dt_ms, window_ms=24.0, dip_i=None, dip_j=None):
    """Semblance coherence of the centre traces of a (ni, nj, ns) block over its 3x3 neighbourhood
    (ni or nj may be 1 for 2D lines).  dip_i / dip_j (samples per trace, same shape) steer the neighbours
    along the local dip.  Returns (ni-2 or 1, nj-2 or 1, ns)."""
    b = np.asarray(block, np.float64)
    ni, nj, ns = b.shape
    ii = range(1, ni - 1) if ni >= 3 else [0]
    jj = range(1, nj - 1) if nj >= 3 else [0]
    offs = [(di, dj) for di in ((-1, 0, 1) if ni >= 3 else (0,)) for dj in ((-1, 0, 1) if nj >= 3 else (0,))]
    i0, i1 = ii[0], ii[-1] + 1
    j0, j1 = jj[0], jj[-1] + 1
    t = np.arange(ns, dtype=float)
    ssum = np.zeros((i1 - i0, j1 - j0, ns))
    esum = np.zeros_like(ssum)
    for di, dj in offs:
        nb = b[i0 + di:i1 + di, j0 + dj:j1 + dj]
        if (dip_i is not None or dip_j is not None) and (di or dj):
            sh = np.zeros_like(nb)
            if dip_i is not None:
                sh += di * np.asarray(dip_i)[i0:i1, j0:j1]
            if dip_j is not None:
                sh += dj * np.asarray(dip_j)[i0:i1, j0:j1]
            src = np.clip(t[None, None, :] + sh, 0, ns - 1.000001)
            k = np.floor(src).astype(int)
            w = src - k
            nb = np.take_along_axis(nb, k, 2) * (1 - w) + np.take_along_axis(nb, k + 1, 2) * w
        ssum += nb
        esum += nb * nb
    n = max(1, int(round(window_ms / dt_ms)) | 1)
    num = running_mean(ssum * ssum, n)
    den = len(offs) * running_mean(esum, n)
    return np.where(den > 1e-20, num / np.maximum(den, 1e-20), 0.0).astype(np.float32)


# ====================================================================== dispatch
def trace_attribute(name, x, dt_ms, params=None):
    p = dict(ATTRIBUTES.get(name, (False, {}))[1])
    p.update(params or {})
    if name == "Envelope":
        return envelope(x)
    if name == "Inst. phase":
        return inst_phase(x)
    if name == "Cosine of phase":
        return cos_phase(x)
    if name == "Inst. frequency":
        return inst_freq(x, dt_ms)
    if name == "RMS amplitude":
        return rms(x, dt_ms, p["window_ms"])
    if name.startswith("Spectral"):
        return spectral(x, dt_ms, p["freq"])
    raise ValueError(name)


def block_attribute(name, block, dt_ms, params=None, dip_i=None, dip_j=None):
    """Attribute of the centre of a (ni, nj, ns) block (1-trace halo for coherence)."""
    lateral, p0 = ATTRIBUTES[name]
    p = dict(p0)
    p.update(params or {})
    if not lateral:
        return trace_attribute(name, block, dt_ms, p)
    steer = p.get("steered")
    return semblance(block, dt_ms, p["window_ms"], dip_i if steer else None, dip_j if steer else None)


def _dips(vol, i_sl, j_sl, t_sl=slice(None)):
    """Dip (samples/trace) for a sub-block (optionally a time slab) from the stored dip field, or None."""
    from .rgt.dip import DipField
    folder = os.path.join(vol.store, "rgt")
    if not DipField.exists(folder):
        return None, None
    df = DipField(folder)
    dt_ms = vol.dt * 1000.0
    return (np.asarray(df.arr["dip_il"][i_sl, j_sl, t_sl], np.float32) / dt_ms,
            np.asarray(df.arr["dip_xl"][i_sl, j_sl, t_sl], np.float32) / dt_ms)


def section_attribute(vol, name, mode, number, params=None):
    """Attribute on the displayed 3D section: inline/crossline -> (ns, n), time slice -> (nil, nxl)."""
    lateral = ATTRIBUTES[name][0]
    dt_ms = vol.dt * 1000.0
    if mode in ("Inline", "Crossline"):
        if mode == "Inline":
            i = vol.il_index(number)
            a, b = max(0, i - 1), min(vol.nil, i + 2)
            blk = np.asarray(vol.ilxl[a:b], np.float32) * vol.inv  # (k, nxl, ns)
            c = i - a
            if not lateral:
                return trace_attribute(name, blk[c], dt_ms, params).T
            di, dj = _dips(vol, slice(a, b), slice(None))
            pad = np.pad(blk, ((0, 0), (1, 1), (0, 0)), mode="edge")
            dpi = np.pad(di, ((0, 0), (1, 1), (0, 0)), mode="edge") if di is not None else None
            dpj = np.pad(dj, ((0, 0), (1, 1), (0, 0)), mode="edge") if dj is not None else None
            if blk.shape[0] == 3:
                out = block_attribute(name, pad, dt_ms, params, dpi, dpj)[0]
            else:  # first / last inline: neighbours only on one side -> treat as a 2D line
                out = block_attribute(name, pad[c:c + 1], dt_ms, params, None, dpj[c:c + 1] if dpj is not None else None)[0]
            return out.T
        j = vol.xl_index(number)
        a, b = max(0, j - 1), min(vol.nxl, j + 2)
        blk = np.transpose(np.asarray(vol.ilxl[:, a:b], np.float32) * vol.inv, (1, 0, 2))  # (k, nil, ns)
        c = j - a
        if not lateral:
            return trace_attribute(name, blk[c], dt_ms, params).T
        di, dj = _dips(vol, slice(None), slice(a, b))
        pad = np.pad(blk, ((0, 0), (1, 1), (0, 0)), mode="edge")
        # in this layout axis 0 = crossline, axis 1 = inline
        dpi = np.pad(np.transpose(dj, (1, 0, 2)), ((0, 0), (1, 1), (0, 0)), mode="edge") if dj is not None else None
        dpj = np.pad(np.transpose(di, (1, 0, 2)), ((0, 0), (1, 1), (0, 0)), mode="edge") if di is not None else None
        if blk.shape[0] == 3:
            out = block_attribute(name, pad, dt_ms, params, dpi, dpj)[0]
        else:
            out = block_attribute(name, pad[c:c + 1], dt_ms, params, None, dpj[c:c + 1] if dpj is not None else None)[0]
        return out.T
    # time slice: read a time slab, compute along time, take the centre sample
    k = vol.t_index(number)
    h = 48
    a, b = max(0, k - h), min(vol.ns, k + h + 1)
    slab = np.transpose(np.asarray(vol.tdata[a:b], np.float32) * vol.inv, (1, 2, 0))  # (nil, nxl, nk)
    if lateral:
        di, dj = _dips(vol, slice(None), slice(None), slice(a, b))  # only the slab, not two whole volumes
        pad = np.pad(slab, ((1, 1), (1, 1), (0, 0)), mode="edge")
        pdi = np.pad(di, ((1, 1), (1, 1), (0, 0)), mode="edge") if di is not None else None
        pdj = np.pad(dj, ((1, 1), (1, 1), (0, 0)), mode="edge") if dj is not None else None
        out = block_attribute(name, pad, dt_ms, params, pdi, pdj)
    else:
        out = trace_attribute(name, slab, dt_ms, params)
    return out[:, :, k - a]


def line_attribute(line, name, params=None, folder=None):
    """Attribute of a 2D line (ntr, ns); coherence along the line (neighbours +/- 1 trace)."""
    x = line.data()
    dt_ms = line.dt * 1000.0
    if not ATTRIBUTES[name][0]:
        return trace_attribute(name, x, dt_ms, params)
    dj = None
    if ATTRIBUTES[name][1].get("steered") or (params or {}).get("steered"):
        from .rgt.dip import line_dip
        dj = line_dip(line, folder)[0][None] / dt_ms
        dj = np.pad(dj, ((0, 0), (1, 1), (0, 0)), mode="edge")
    pad = np.pad(x[None], ((0, 0), (1, 1), (0, 0)), mode="edge")
    return block_attribute(name, pad, dt_ms, params, None, dj)[0]


# ====================================================================== full attribute volumes
def compute_volume(project, vol_name, name, params=None, block=16, progress=None, cancel=None):
    """Write an attribute volume store (same layout as an imported volume) and register it in the project as
    '<vol_name> [<name>]'.  Returns the new volume name."""
    from .volume import Volume3D, STORE_VERSION
    vol = project.volume(vol_name)
    lateral = ATTRIBUTES[name][0]
    dt_ms = vol.dt * 1000.0
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    from .volume import unused_store
    # a new folder when the attribute exists already: the old store may be open (memory-mapped) for display
    store = unused_store(os.path.join(os.path.dirname(vol.store), f"{os.path.basename(vol.store)}__{safe}"))
    os.makedirs(store, exist_ok=True)
    tmp = os.path.join(store, "data_ilxl.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                    shape=(vol.nil, vol.nxl, vol.ns))
    # pass 1: compute in blocks of inlines (+1 halo for coherence), track a robust scale
    samples = []
    raw = np.lib.format.open_memmap(os.path.join(store, "raw.tmp.npy"), mode="w+", dtype=np.float32,
                                    shape=(vol.nil, vol.nxl, vol.ns))
    for a in range(0, vol.nil, block):
        if cancel and cancel():
            del out, raw  # release the memmaps (Windows) and remove the unfinished, unregistered store
            shutil.rmtree(store, ignore_errors=True)
            raise RuntimeError("cancelled")
        b = min(vol.nil, a + block)
        if lateral:
            a0, b0 = max(0, a - 1), min(vol.nil, b + 1)
            blk = np.asarray(vol.ilxl[a0:b0], np.float32) * vol.inv
            di, dj = _dips(vol, slice(a0, b0), slice(None))
            pad = np.pad(blk, ((1 if a0 == a else 0, 1 if b0 == b else 0), (1, 1), (0, 0)), mode="edge")
            if di is not None:
                pw = ((1 if a0 == a else 0, 1 if b0 == b else 0), (1, 1), (0, 0))
                di, dj = np.pad(di, pw, mode="edge"), np.pad(dj, pw, mode="edge")
            res = block_attribute(name, pad, dt_ms, params, di, dj)
        else:
            res = block_attribute(name, np.asarray(vol.ilxl[a:b], np.float32) * vol.inv, dt_ms, params)
        raw[a:b] = res
        samples.append(np.abs(res[::max(1, len(res) // 4), ::7, ::5]).ravel())
        if progress:
            progress(0.7 * b / vol.nil, f"{name}: inline {b}/{vol.nil}")
    p999 = float(np.percentile(np.concatenate(samples), 99.9)) or 1.0
    scale = 1000.0 / p999
    for a in range(0, vol.nil, block):
        b = min(vol.nil, a + block)
        out[a:b] = np.clip(raw[a:b] * scale, -60000, 60000).astype(np.float16)
    out.flush()
    del raw
    os.remove(os.path.join(store, "raw.tmp.npy"))
    tm = np.lib.format.open_memmap(os.path.join(store, "data_t.npy"), mode="w+", dtype=np.float16,
                                   shape=(vol.ns, vol.nil, vol.nxl))
    for a in range(0, vol.nil, block):
        b = min(vol.nil, a + block)
        tm[:, a:b, :] = np.transpose(np.asarray(out[a:b]), (2, 0, 1))
        if progress:
            progress(0.7 + 0.3 * b / vol.nil, f"{name}: time slices {b}/{vol.nil}")
    tm.flush()
    del tm, out
    filled = os.path.join(vol.store, "filled.npy")
    if os.path.exists(filled):
        shutil.copy(filled, os.path.join(store, "filled.npy"))
    meta = dict(vol.meta)
    meta.update({"version": STORE_VERSION, "scale": scale, "attribute": name, "source_volume": vol_name,
                 "params": params or {}})
    with open(os.path.join(store, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    new_name = f"{vol_name} [{name}]"
    project.volumes = [v for v in project.volumes if v["name"] != new_name]
    project._vol_cache.pop(new_name, None)
    project.volumes.append({"name": new_name, "store": store, "segy": "", "attribute": name, "source": vol_name})
    Volume3D(store)  # validates
    return new_name


# ====================================================================== horizon curvature
def horizon_curvature(grid, kind="mean", velocity=2000.0):
    """Curvature of a time-structure MapGrid converted to depth with a constant velocity (z = v t / 2000, m).
    kind: 'mean', 'gaussian', 'max' (most positive), 'min' (most negative), 'dip' (deg), 'azimuth' (deg).
    Units 1/km for curvatures.  Returns a MapGrid."""
    from .maps import MapGrid
    z = grid.z * velocity / 2000.0
    if grid.kind == "ilxl":
        v = grid.vol
        hy = float(np.hypot(v.ax[1], v.ay[1]) * v.il_step)  # metres between inlines (rows)
        hx = float(np.hypot(v.ax[2], v.ay[2]) * v.xl_step)  # metres between crosslines (cols)
    else:
        hx, hy = grid.dx, grid.dy
    zy, zx = np.gradient(z, hy, hx)
    zyy, zyx = np.gradient(zy, hy, hx)
    zxy, zxx = np.gradient(zx, hy, hx)
    zxy = 0.5 * (zxy + zyx)
    g = 1 + zx * zx + zy * zy
    if kind == "mean":
        k = ((1 + zx * zx) * zyy - 2 * zx * zy * zxy + (1 + zy * zy) * zxx) / (2 * g ** 1.5)
    elif kind == "gaussian":
        k = (zxx * zyy - zxy * zxy) / g ** 2
    elif kind in ("max", "min"):
        km = ((1 + zx * zx) * zyy - 2 * zx * zy * zxy + (1 + zy * zy) * zxx) / (2 * g ** 1.5)
        kg = (zxx * zyy - zxy * zxy) / g ** 2
        r = np.sqrt(np.maximum(km * km - kg, 0))
        k = km + r if kind == "max" else km - r
    elif kind == "dip":
        return MapGrid(np.degrees(np.arctan(np.hypot(zx, zy))), grid.kind, "deg", f"Dip {grid.label}", grid.vol,
                       grid.x0, grid.y0, grid.dx, grid.dy)
    elif kind == "azimuth":
        return MapGrid(np.degrees(np.arctan2(zx, zy)) % 360, grid.kind, "deg", f"Dip azimuth {grid.label}",
                       grid.vol, grid.x0, grid.y0, grid.dx, grid.dy)
    else:
        raise ValueError(kind)
    return MapGrid(k * 1000.0, grid.kind, "1/km", f"{kind} curvature {grid.label}", grid.vol, grid.x0, grid.y0,
                   grid.dx, grid.dy)
