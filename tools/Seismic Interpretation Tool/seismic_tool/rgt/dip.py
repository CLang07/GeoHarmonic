"""Local reflector dip from the gradient structure tensor (RGT step A).

For a locally plane wave u(t - p_i * i - p_x * x) the gradients satisfy g_i = -p_i g_t and g_x = -p_x g_t, so
the least-squares slopes over a Gaussian window are
    p_i = -<g_t g_i> / <g_t g_t>,   p_x = -<g_t g_x> / <g_t g_t>      (samples per trace)
(<.> = Gaussian smoothing of the tensor component).  Planarity = 1 - residual / gradient energy, in [0, 1]:
1 for a perfect plane wave, ~0 for noise or chaotic zones.  Cheap (no per-voxel eigen solves) and exact for
any dip that is not aliased.

Volumes are processed in blocks of inlines with a halo and written as float16 memmaps:
    <store>/rgt/dip_il.npy, dip_xl.npy  (n_inline, n_xline, n_samples)  ms per trace (+ = deeper with
                                        increasing inline / crossline number)
    <store>/rgt/planarity.npy            same shape, 0..1
2D lines: dip (ms per trace, + = deeper with increasing CDP) and planarity, (n_traces, n_samples).
"""
import json
import os

import numpy as np


def gaussian_kernel(sigma, deriv=False):
    if sigma <= 0:
        return np.array([1.0]) if not deriv else np.array([-0.5, 0.0, 0.5])
    r = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=float)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    g /= g.sum()
    if deriv:  # derivative of Gaussian, correlation form: out[n] = sum_k w[k] a[n+k] ~ da/dx
        d = x / sigma ** 2 * g
        return d / np.sum(d * x)  # exact on a linear ramp
    return g


def correlate1d(a, w, axis):
    """out[..., n, ...] = sum_k w[k] * a[..., n + k - r, ...] with reflect padding (float32 in, float32 out)."""
    r = (len(w) - 1) // 2
    if r == 0:
        return a * np.float32(w[0])
    a = np.moveaxis(a, axis, -1)
    n = a.shape[-1]
    pad = [(0, 0)] * (a.ndim - 1) + [(r, r)]
    mode = "reflect" if n > r else "edge"
    ap = np.pad(a, pad, mode=mode)
    out = np.zeros(a.shape, np.float32)
    for k, wk in enumerate(w):
        if wk != 0:
            out += np.float32(wk) * ap[..., k:k + n]
    return np.moveaxis(out, -1, axis)


def gsmooth(a, sigmas):
    out = np.asarray(a, np.float32)
    for ax, s in enumerate(sigmas):
        if s and s > 0:
            out = correlate1d(out, gaussian_kernel(s), ax)
    return out


def gradients(a, sigma_grad=1.0):
    """Derivative-of-Gaussian gradients along every axis (in samples)."""
    out = []
    for ax in range(a.ndim):
        g = np.asarray(a, np.float32)
        for ax2 in range(a.ndim):
            g = correlate1d(g, gaussian_kernel(sigma_grad, deriv=(ax2 == ax)), ax2)
        out.append(g)
    return out


def dip_2d(sec, sigma_grad=1.0, sigma_x=3.0, sigma_t=4.0, eps=1e-12):
    """sec (n_traces, n_samples) -> (dip samples/trace, planarity) float32."""
    gx, gt = gradients(sec, sigma_grad)
    Ttt = gsmooth(gt * gt, (sigma_x, sigma_t))
    Ttx = gsmooth(gt * gx, (sigma_x, sigma_t))
    Txx = gsmooth(gx * gx, (sigma_x, sigma_t))
    p = -Ttx / (Ttt + eps)
    resid = Txx + 2 * p * Ttx + p * p * Ttt
    plan = np.clip(1.0 - resid / (Ttt + Txx + eps), 0, 1)
    return p.astype(np.float32), plan.astype(np.float32)


def dip_3d_block(block, sigma_grad=1.0, sigma_s=2.0, sigma_t=4.0, eps=1e-12):
    """block (ni, nj, ns) -> (p_il, p_xl, planarity) in samples per trace."""
    gi, gj, gt = gradients(block, sigma_grad)
    sig = (sigma_s, sigma_s, sigma_t)
    Ttt = gsmooth(gt * gt, sig)
    Tti = gsmooth(gt * gi, sig)
    Ttj = gsmooth(gt * gj, sig)
    pi_ = -Tti / (Ttt + eps)
    pj_ = -Ttj / (Ttt + eps)
    del Tti, Ttj
    Tii = gsmooth(gi * gi, sig)
    Tjj = gsmooth(gj * gj, sig)
    # residual of g_i + p_i g_t and g_j + p_j g_t
    resid = Tii + Tjj - pi_ * pi_ * Ttt - pj_ * pj_ * Ttt
    plan = np.clip(1.0 - resid / (Ttt + Tii + Tjj + eps), 0, 1)
    return pi_.astype(np.float32), pj_.astype(np.float32), plan.astype(np.float32)


def compute_dip_volume(vol, out_dir=None, sigma_grad=1.0, sigma_s=2.0, sigma_t=4.0, block=16, progress=None,
                       cancel=None):
    """Dip field for a Volume3D, block by block over inlines (bounded memory). Returns a DipField."""
    out_dir = out_dir or os.path.join(vol.store, "rgt")
    os.makedirs(out_dir, exist_ok=True)
    shape = (vol.nil, vol.nxl, vol.ns)
    names = ("dip_il", "dip_xl", "planarity")
    tmp = {n: np.lib.format.open_memmap(os.path.join(out_dir, n + ".tmp.npy"), mode="w+", dtype=np.float16,
                                        shape=shape) for n in names}
    halo = int(np.ceil(3 * max(sigma_s, sigma_grad))) + 2
    dt_ms = vol.dt * 1000.0
    for a in range(0, vol.nil, block):
        if cancel and cancel():
            for m in tmp.values():
                del m
            raise RuntimeError("cancelled")
        b = min(vol.nil, a + block)
        a0, b0 = max(0, a - halo), min(vol.nil, b + halo)
        blk = np.asarray(vol.ilxl[a0:b0], np.float32)
        pi_, pj_, pl = dip_3d_block(blk, sigma_grad, sigma_s, sigma_t)
        s = slice(a - a0, a - a0 + (b - a))
        tmp["dip_il"][a:b] = (pi_[s] * dt_ms).astype(np.float16)
        tmp["dip_xl"][a:b] = (pj_[s] * dt_ms).astype(np.float16)
        tmp["planarity"][a:b] = pl[s].astype(np.float16)
        if progress:
            progress(b / vol.nil, f"Dip field: inline {b}/{vol.nil}")
    for n, m in tmp.items():
        m.flush()
    del m
    tmp.clear()
    for n in names:
        os.replace(os.path.join(out_dir, n + ".tmp.npy"), os.path.join(out_dir, n + ".npy"))
    with open(os.path.join(out_dir, "dip.json"), "w") as fh:
        json.dump({"sigma_grad": sigma_grad, "sigma_s": sigma_s, "sigma_t": sigma_t, "units": "ms per trace",
                   "shape": list(shape)}, fh)
    return DipField(out_dir)


class DipField:
    """Read access to a stored 3D dip field (same geometry as its volume)."""
    NAMES = ("dip_il", "dip_xl", "planarity")

    def __init__(self, folder):
        self.folder = folder
        self.arr = {n: np.load(os.path.join(folder, n + ".npy"), mmap_mode="r") for n in self.NAMES}

    @staticmethod
    def exists(folder):
        return all(os.path.exists(os.path.join(folder, n + ".npy")) for n in DipField.NAMES)

    def section(self, name, vol, mode, number):
        """Same layout as the seismic display arrays: inline/crossline -> (ns, n), time slice -> (nil, nxl)."""
        a = self.arr[name]
        if mode == "Inline":
            return np.asarray(a[vol.il_index(number)], np.float32).T
        if mode == "Crossline":
            return np.asarray(a[:, vol.xl_index(number)], np.float32).T
        return np.asarray(a[:, :, vol.t_index(number)], np.float32)


def line_dip(line, folder=None, sigma_grad=1.0, sigma_x=3.0, sigma_t=4.0):
    """(dip ms/trace, planarity) for a 2D line, cached as <folder>/<line>_dip.npz when folder is given."""
    path = None
    if folder:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in line.name)
        path = os.path.join(folder, safe + "_dip.npz")
        if os.path.exists(path):
            z = np.load(path)
            if z["dip"].shape == (len(line.cdp), line.ns):
                return z["dip"].astype(np.float32), z["planarity"].astype(np.float32)
    p, pl = dip_2d(line.data(), sigma_grad, sigma_x, sigma_t)
    p *= line.dt * 1000.0
    if path:
        os.makedirs(folder, exist_ok=True)
        np.savez(path, dip=p.astype(np.float16), planarity=pl.astype(np.float16))
    return p, pl
