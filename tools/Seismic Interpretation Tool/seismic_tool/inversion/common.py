"""Shared inversion helpers: zero-phase Butterworth filters, spectra, the low-cut pick, volume store writer,
elastic well logs on the (tied) time axis."""
import json
import os
import shutil

import numpy as np


# ====================================================================== filters + spectra
def butterworth_response(f, fc, order=4, kind="low"):
    """Zero-phase Butterworth amplitude: 1 / sqrt(1 + (f/fc)^(2n)) (low) - exactly -3 dB at fc."""
    f = np.abs(np.asarray(f, float))
    r = (f / max(fc, 1e-9)) ** (2 * order)
    return 1.0 / np.sqrt(1.0 + r) if kind == "low" else np.sqrt(r / (1.0 + r))


def butter_filter(x, dt_ms, fc, order=4, kind="low", axis=-1):
    """Zero-phase Butterworth (frequency domain) along `axis`; the ends are mirror-padded against wrap-around."""
    x = np.asarray(x, float)
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    pad = min(n - 1, max(8, n // 2))
    xp = np.concatenate([x[..., pad:0:-1], x, x[..., -2:-pad - 2:-1]], axis=-1) if pad > 0 else x
    nfft = int(2 ** np.ceil(np.log2(xp.shape[-1])))
    f = np.fft.rfftfreq(nfft, dt_ms / 1000.0)
    H = butterworth_response(f, fc, order, kind)
    if kind == "low":
        mean = xp.mean(axis=-1, keepdims=True)
        y = np.fft.irfft(np.fft.rfft(xp - mean, nfft) * H, nfft)[..., :xp.shape[-1]] + mean
    else:
        y = np.fft.irfft(np.fft.rfft(xp, nfft) * H, nfft)[..., :xp.shape[-1]]
    y = y[..., pad:pad + n]
    return np.moveaxis(y, -1, axis)


def amplitude_spectrum(traces, dt_ms, nfft=None):
    """Average amplitude spectrum (freqs Hz, amplitude) of traces (m, ns) (Hann taper)."""
    tr = np.atleast_2d(np.asarray(traces, float))
    tr = tr[np.all(np.isfinite(tr), axis=1)]
    ns = tr.shape[1]
    nfft = nfft or int(2 ** np.ceil(np.log2(ns)))
    w = np.hanning(ns)
    A = np.abs(np.fft.rfft((tr - tr.mean(axis=1, keepdims=True)) * w, nfft)).mean(axis=0)
    return np.fft.rfftfreq(nfft, dt_ms / 1000.0), A


def pick_lowcut(freqs, amp, frac=0.2):
    """Low end of the seismic band: the lowest frequency where the (smoothed) spectrum reaches frac x its peak -
    the LFM cut-off (the inversion takes over above it)."""
    a = np.convolve(np.asarray(amp, float), np.ones(3) / 3, "same")
    k = int(np.argmax(a))
    above = np.nonzero(a[:k + 1] >= frac * a[k])[0]
    return float(freqs[above[0]]) if len(above) else float(freqs[k])


# ====================================================================== volume store writer
def write_volume(project, vol_name, name, compute, block=16, progress=None, cancel=None, meta_extra=None):
    """New volume store with the layout of `vol_name` filled by compute(a, b) -> float32 (b - a, nxl, ns) for
    inline indices [a, b); registered as '<vol_name> [<name>]'. Cancel removes the partial store."""
    return write_volumes(project, vol_name, [name], lambda a, b: {name: compute(a, b)}, block, progress, cancel,
                         meta_extra)[0]


def write_volumes(project, vol_name, names, compute, block=16, progress=None, cancel=None, meta_extra=None):
    """Several volumes from one pass: compute(a, b) -> {name: float32 (b - a, nxl, ns)}. Returns the new names."""
    from ..volume import STORE_VERSION, Volume3D, unused_store
    vol = project.volume(vol_name)
    stores, raws, samples = {}, {}, {}
    for name in names:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        st = unused_store(os.path.join(os.path.dirname(vol.store), f"{os.path.basename(vol.store)}__{safe}"))
        os.makedirs(st, exist_ok=True)
        stores[name] = st
        raws[name] = np.lib.format.open_memmap(os.path.join(st, "raw.tmp.npy"), mode="w+", dtype=np.float32,
                                               shape=(vol.nil, vol.nxl, vol.ns))
        samples[name] = []
    try:
        for a in range(0, vol.nil, block):
            if cancel and cancel():
                raise RuntimeError("cancelled")
            b = min(vol.nil, a + block)
            res = compute(a, b)
            for name in names:
                r = np.asarray(res[name], np.float32)
                raws[name][a:b] = r
                fin = np.abs(r[::max(1, len(r) // 4), ::5, ::3])
                samples[name].append(fin[np.isfinite(fin)].ravel())
            if progress:
                progress(0.8 * b / vol.nil, f"inline {b}/{vol.nil}")
    except BaseException:
        raws.clear()
        for st in stores.values():
            shutil.rmtree(st, ignore_errors=True)
        raise
    out_names = []
    for name in names:
        st, raw = stores[name], raws[name]
        allv = np.concatenate(samples[name]) if samples[name] else np.ones(1)
        p999 = float(np.percentile(allv, 99.9)) if allv.size else 1.0
        scale = 1000.0 / (p999 or 1.0)
        out = np.lib.format.open_memmap(os.path.join(st, "data_ilxl.npy"), mode="w+", dtype=np.float16,
                                        shape=(vol.nil, vol.nxl, vol.ns))
        for a in range(0, vol.nil, block):
            b = min(vol.nil, a + block)
            out[a:b] = np.clip(np.nan_to_num(raw[a:b]) * scale, -60000, 60000).astype(np.float16)
        out.flush()
        raws[name] = None
        del raw
        os.remove(os.path.join(st, "raw.tmp.npy"))
        tm = np.lib.format.open_memmap(os.path.join(st, "data_t.npy"), mode="w+", dtype=np.float16,
                                       shape=(vol.ns, vol.nil, vol.nxl))
        for a in range(0, vol.nil, block):
            b = min(vol.nil, a + block)
            tm[:, a:b, :] = np.transpose(np.asarray(out[a:b]), (2, 0, 1))
        tm.flush()
        del tm, out
        filled = os.path.join(vol.store, "filled.npy")
        if os.path.exists(filled):
            shutil.copy(filled, os.path.join(st, "filled.npy"))
        meta = dict(vol.meta)
        meta.update({"version": STORE_VERSION, "scale": scale, "attribute": name, "source_volume": vol_name})
        meta.update(meta_extra or {})
        with open(os.path.join(st, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        new_name = f"{vol_name} [{name}]"
        project.volumes = [v for v in project.volumes if v["name"] != new_name]
        project._vol_cache.pop(new_name, None)
        project.volumes.append({"name": new_name, "store": st, "segy": "", "attribute": name, "source": vol_name})
        Volume3D(st)
        out_names.append(new_name)
        if progress:
            progress(0.8 + 0.2 * len(out_names) / len(names), f"{name}: stored")
    return out_names


# ====================================================================== well logs in time
def well_elastic(well, td, dt_ms, t_axis=None, despike=True, gardner=(0.31, 0.25)):
    """Vp, Vs, rho, AI of a well blocked onto the seismic time axis (tied TWT, td.twt). Conditioning: spikes
    removed (running median, 4 MAD); rho from Gardner and Vs from Castagna where missing / outside physical
    limits. Returns dict t (ms), vp, vs, rho, ai (NaN outside the logged interval) or None."""
    from ..tie import despike as _despike
    from ..tie import vs_castagna
    if not well.logs or well.depth is None or well.logs.get("DT") is None:
        return None
    md = np.asarray(well.depth, float)
    dtl = np.asarray(well.logs["DT"], float)
    vp = 304800.0 / np.where((dtl > 30) & (dtl < 250), dtl, np.nan)
    rho = well.logs.get("RHOB")
    rho = np.where((np.asarray(rho, float) > 1.2) & (np.asarray(rho, float) < 3.2), rho, np.nan) \
        if rho is not None else np.full(md.shape, np.nan)
    dts = well.logs.get("DTS")
    vs = 304800.0 / np.where((np.asarray(dts, float) > 50) & (np.asarray(dts, float) < 600), dts, np.nan) \
        if dts is not None else np.full(md.shape, np.nan)
    if despike:
        vp = _despike(vp)[0]
        rho = _despike(rho)[0] if np.isfinite(rho).sum() > 20 else rho
    rho = np.where(np.isfinite(rho), rho, gardner[0] * np.power(vp, gardner[1]))
    vs = np.where(np.isfinite(vs), vs, vs_castagna(vp))
    ok = np.isfinite(vp) & np.isfinite(rho) & np.isfinite(vs)
    if ok.sum() < 20:
        return None
    t = td.twt(well.md_to_tvdss(md))
    i0, i1 = np.nonzero(ok)[0][[0, -1]]
    sl = slice(i0, i1 + 1)
    t, ok = t[sl], ok[sl]
    o = np.argsort(t, kind="stable")
    t = t[o]
    tt = np.asarray(t_axis, float) if t_axis is not None else np.arange(np.ceil(t[0] / dt_ms) * dt_ms, t[-1], dt_ms)
    edges = np.r_[tt - dt_ms / 2, tt[-1] + dt_ms / 2]
    out = {"t": tt}
    inside = (tt >= t[0]) & (tt <= t[-1])
    for key, a in (("vp", vp[sl][o]), ("vs", vs[sl][o]), ("rho", rho[sl][o])):
        okk = ok[o]
        a = np.interp(t, t[okk], a[okk])
        cum = np.r_[0.0, np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(t))]
        v = np.diff(np.interp(edges, t, cum)) / dt_ms
        out[key] = np.where(inside, v, np.nan)
    out["ai"] = out["vp"] * out["rho"]
    return out


def export_segy(vol, path, progress=None, cancel=None, note=""):
    """Write a 3D volume (physical values, IEEE float) as SEG-Y: inline 189 / 9, crossline 193 / 21, X / Y 181 / 185
    (scalar -100), sample interval from the volume. Written to <path>.part first (Cancel leaves nothing)."""
    from ..scenarios import _text_header
    ns = vol.ns
    dt_us = int(round(vol.dt * 1e6))
    tmp = path + ".part"
    try:
        with open(tmp, "wb") as fh:
            fh.write(_text_header([f"GEO TOOL KIT SEISMIC INTERPRETATION TOOL - {note}",
                                   f"TIME DOMAIN: {ns} SAMPLES, {dt_us} US, START {vol.t0 * 1000:g} MS",
                                   "INLINE BYTES 189 + 9, CROSSLINE 193 + 21, CDP X/Y 181/185 SCALAR -100 (BYTE 71)",
                                   "FORMAT 5 IEEE FLOAT, BIG ENDIAN"]))
            b = bytearray(400)
            b[16:18] = dt_us.to_bytes(2, "big")
            b[20:22] = ns.to_bytes(2, "big")
            b[24:26] = (5).to_bytes(2, "big")
            b[28:30] = (1).to_bytes(2, "big")
            b[54:56] = (1).to_bytes(2, "big")
            fh.write(bytes(b))
            for n, il in enumerate(vol.inlines):
                if cancel is not None and cancel():
                    raise RuntimeError("cancelled")
                data = np.nan_to_num(vol.inline(il)).astype(">f4")
                xls = vol.xlines
                x, y = vol.xy(np.full(vol.nxl, il), xls)
                th = np.zeros((vol.nxl, 240), np.uint8)

                def put(byte, vals, size=4):
                    th[:, byte - 1:byte - 1 + size] = np.asarray(vals, f">i{size}").view(np.uint8).reshape(vol.nxl,
                                                                                                         size)
                put(1, n * vol.nxl + np.arange(1, vol.nxl + 1))
                put(9, np.full(vol.nxl, int(il)))
                put(21, np.asarray(xls, np.int64))
                put(189, np.full(vol.nxl, int(il)))
                put(193, np.asarray(xls, np.int64))
                put(181, np.rint(np.asarray(x) * 100).astype(np.int64))
                put(185, np.rint(np.asarray(y) * 100).astype(np.int64))
                put(71, np.full(vol.nxl, -100), 2)
                put(115, np.full(vol.nxl, ns), 2)
                put(117, np.full(vol.nxl, dt_us), 2)
                fh.write(np.concatenate([th, data.view(np.uint8).reshape(vol.nxl, -1)], axis=1).tobytes())
                if progress:
                    progress((n + 1) / vol.nil, f"SEG-Y inline {il}")
        if os.path.exists(path):
            os.remove(path)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path
