"""RGT step C: horizon stack (iso-RGT surfaces), Wheeler transform, stratal slices.

Horizon stack = N surfaces at evenly spaced RGT values.  3D: <store>/rgt/stack.npy (N, nil, nxl) TWT ms
(NaN where the surface is outside the trace) + stack.json (levels); 2D: <project>_seismic/rgt/stack_2d.npz
(levels + one (N, ntr) array per line).  A Wheeler section puts RGT on the vertical axis: every trace is
resampled at the time of each stack level; where consecutive levels collapse to (almost) the same time
there was no deposition or erosion (a hiatus) and the Wheeler view shows a gap.
"""
import json
import os

import numpy as np


def iso_times(tau, levels, times):
    """tau (T, ns) monotonic non-decreasing per row; levels (N,) -> times (T, N) where tau == level
    (linear interpolation), NaN outside a trace's range.  One global searchsorted for all traces."""
    tau = np.asarray(tau, np.float64)
    T, ns = tau.shape
    levels = np.asarray(levels, np.float64)
    lo, hi = min(tau.min(), levels.min()), max(tau.max(), levels.max())
    span = hi - lo + 1.0
    key = (tau - lo) + span * np.arange(T)[:, None]  # globally sorted
    q = (levels[None, :] - lo) + span * np.arange(T)[:, None]
    flat = key.ravel()
    pos = np.searchsorted(flat, q.ravel(), side="left").reshape(T, len(levels))
    k = pos - ns * np.arange(T)[:, None]  # index within the row of the first tau >= level
    ok = (k > 0) & (k < ns)
    k1 = np.clip(k, 1, ns - 1)
    r = np.arange(T)[:, None]
    t0, t1 = tau[r, k1 - 1], tau[r, k1]
    f = np.where(t1 > t0, (levels[None, :] - t0) / np.where(t1 > t0, t1 - t0, 1), 0.0)
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    out = times[k1 - 1] + np.clip(f, 0, 1) * dt
    # a level equal to the very first sample's tau is inside the trace too
    exact0 = (k == 0) & (np.abs(tau[:, :1] - levels[None, :]) < 1e-9)
    out = np.where(exact0, times[0], out)
    return np.where(ok | exact0, out, np.nan).astype(np.float32)


def default_levels(tau_sample, n=100):
    a = np.asarray(tau_sample, float)
    a = a[np.isfinite(a)]
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    return np.linspace(lo, hi, n)


# ====================================================================== 3D stack
def build_stack_3d(vol, rgtvol, n_levels=100, levels=None, out_dir=None, block=16, progress=None, cancel=None):
    """Iso-RGT stack for a volume; writes stack.npy/stack.json in the RGT folder. Returns HorizonStack3D."""
    out_dir = out_dir or rgtvol.folder
    if levels is None:
        samp = rgtvol.decode(rgtvol.arr[::max(1, vol.nil // 20), ::max(1, vol.nxl // 20)])
        levels = default_levels(samp, n_levels)
    levels = np.asarray(levels, float)
    tmp = os.path.join(out_dir, "stack.tmp.npy")
    st = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(levels), vol.nil, vol.nxl))
    times = vol.times
    for a in range(0, vol.nil, block):
        if cancel and cancel():
            del st
            raise RuntimeError("cancelled")
        b = min(vol.nil, a + block)
        tau = rgtvol.decode(rgtvol.arr[a:b]).reshape(-1, vol.ns)
        st[:, a:b, :] = iso_times(tau, levels, times).T.reshape(len(levels), b - a, vol.nxl)
        if progress:
            progress(b / vol.nil, f"Horizon stack: inline {b}/{vol.nil}")
    st.flush()
    del st
    os.replace(tmp, os.path.join(out_dir, "stack.npy"))
    with open(os.path.join(out_dir, "stack.json"), "w") as fh:
        json.dump({"levels": levels.tolist()}, fh)
    return HorizonStack3D(out_dir)


class HorizonStack3D:
    def __init__(self, folder):
        self.folder = folder
        self.times = np.load(os.path.join(folder, "stack.npy"), mmap_mode="r")
        with open(os.path.join(folder, "stack.json")) as fh:
            self.levels = np.asarray(json.load(fh)["levels"], float)

    @staticmethod
    def exists(folder):
        return os.path.exists(os.path.join(folder, "stack.npy")) and os.path.exists(os.path.join(folder, "stack.json"))

    def __len__(self):
        return len(self.levels)

    def surface(self, k):
        return np.asarray(self.times[int(np.clip(k, 0, len(self.levels) - 1))], np.float32)

    def section(self, vol, mode, number):
        """(N, n traces) times of all levels along an inline / crossline."""
        if mode == "Inline":
            return np.asarray(self.times[:, vol.il_index(number), :], np.float32)
        return np.asarray(self.times[:, :, vol.xl_index(number)], np.float32)


def stratal_slice(vol, surface_ms):
    """Amplitude on a surface (nil, nxl) of TWT ms, linear in time; reads only the time slab it spans."""
    fin = np.isfinite(surface_ms)
    out = np.full(surface_ms.shape, np.nan, np.float32)
    if not fin.any():
        return out
    idx = (surface_ms / 1000.0 - vol.t0) / vol.dt
    k0 = int(max(0, np.floor(np.nanmin(idx))))
    k1 = int(min(vol.ns - 1, np.ceil(np.nanmax(idx)) + 1))
    slab = np.asarray(vol.tdata[k0:k1 + 1], np.float32) * vol.inv  # (nk, nil, nxl)
    rel = np.clip(np.where(fin, idx, k0) - k0, 0, slab.shape[0] - 1.000001)
    k = np.floor(rel).astype(np.int64)
    w = (rel - k).astype(np.float32)
    k2 = np.minimum(k + 1, slab.shape[0] - 1)
    a = np.take_along_axis(slab, k[None], 0)[0]
    b = np.take_along_axis(slab, k2[None], 0)[0]
    out[fin] = (a * (1 - w) + b * w)[fin]
    return out


# ====================================================================== Wheeler
def wheeler(seis, tau, levels, times, gap_frac=0.25):
    """seis, tau (ntr, ns); levels (N,) -> (amplitude (N, ntr), hiatus mask (N, ntr)).
    The interval between a level and the next is a hiatus / condensed section on a trace where it is
    thinner than gap_frac x its median thickness along the section (or < a quarter sample) -> NaN, mask True."""
    t = iso_times(tau, levels, times)  # (ntr, N)
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    thick = np.diff(t, axis=1)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(np.where(np.isfinite(thick), thick, np.nan), axis=0) if len(thick) else thick
    med = np.where(np.isfinite(med), med, 0.0)
    hiatus = np.zeros(t.shape, bool)
    hiatus[:, :-1] = np.isfinite(thick) & ((thick < gap_frac * med[None, :]) | (thick < 0.25 * dt))
    idx = (t - times[0]) / dt
    fin = np.isfinite(idx)
    ns = seis.shape[1]
    ic = np.clip(np.where(fin, idx, 0), 0, ns - 1.000001)
    k = np.floor(ic).astype(np.int64)
    w = ic - k
    r = np.arange(seis.shape[0])[:, None]
    amp = seis[r, k] * (1 - w) + seis[r, k + 1] * w
    amp = np.where(fin & ~hiatus, amp, np.nan)
    return amp.T.astype(np.float32), hiatus.T


def to_rgt(tau_trace, t_ms, times):
    """RGT value at time(s) t_ms on one trace (linear)."""
    return np.interp(t_ms, times, tau_trace, left=np.nan, right=np.nan)


# ====================================================================== 2D stack
def build_stack_2d(project, names, n_levels=100, levels=None, folder=None):
    """Stack for 2D lines with RGT (shared RGT scale - the lines were solved together)."""
    from .solve import load_rgt_line
    folder = folder or os.path.join(project.seismic_dir(), "rgt")
    taus = {n: load_rgt_line(folder, n) for n in names}
    taus = {n: t for n, t in taus.items() if t is not None}
    if not taus:
        raise RuntimeError("no 2D RGT found - compute RGT for the 2D lines first")
    if levels is None:
        levels = default_levels(np.concatenate([t[::7].ravel() for t in taus.values()]), n_levels)
    levels = np.asarray(levels, float)
    data = {"levels": levels}
    for n, t in taus.items():
        data["L_" + n] = iso_times(t, levels, project.line(n).times).T  # (N, ntr)
    np.savez(os.path.join(folder, "stack_2d.npz"), **data)
    return load_stack_2d(folder)


def load_stack_2d(folder):
    p = os.path.join(folder, "stack_2d.npz")
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        return {"levels": z["levels"], "lines": {k[2:]: z[k] for k in z.files if k.startswith("L_")}}


def stack_to_horizon(hset, name, level_index, stack3d=None, vol_name=None, stack2d=None):
    """New editable horizon from stack level k (3D surface and/or every 2D line of a 2D stack)."""
    h = hset.new(name)
    if stack3d is not None and vol_name:
        h.data[f"3D:{vol_name}"] = stack3d.surface(level_index).astype(np.float32)
    if stack2d is not None:
        for n, arr in stack2d["lines"].items():
            h.data[f"2D:{n}"] = np.asarray(arr[int(level_index)], np.float32)
    hset.dirty = True
    return h
