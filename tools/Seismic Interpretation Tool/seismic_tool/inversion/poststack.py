"""Post-stack inversion (PLAN 19b): coloured inversion (relative AI) and model-based inversion in log impedance.

Model-based: z = ln AI per trace, reflectivity r = D z (r_k = (z_k - z_(k-1)) / 2 ~ (AI_k - AI_(k-1)) /
(AI_k + AI_(k-1))), seismic = w * r ('same' convolution, like welltie.synthetic). Solves
    min ||s - G z||^2 + lam ||z - z_lfm||^2 (+ mu sum |D z|  via IRLS -> blocky / TV)
with conjugate gradients, matrix-free (FFT convolutions), vectorised over a block of traces (each trace its
own CG scalars). Optional lateral smoothing of the update between neighbouring traces.
Coloured inversion (Lancaster & Whitcombe): operator amplitude = well AI spectrum trend (f^alpha) / seismic
spectrum, phase -90 deg, band-limited; output scaled to the band-limited well ln AI.
"""
import numpy as np

from .common import amplitude_spectrum, butter_filter


# ====================================================================== operators
def _nfft(n):
    return int(2 ** np.ceil(np.log2(max(2, n))))


class ConvOp:
    """'same' convolution with wavelet w (odd or even length, centred like np.convolve(mode='same'))."""

    def __init__(self, w, n):
        self.w = np.asarray(w, float)
        self.n = n
        m = len(self.w)
        self.m = m
        self.o = (m - 1) // 2
        self.N = _nfft(n + m - 1)
        self.W = np.fft.rfft(self.w, self.N)
        self.Wr = np.fft.rfft(self.w[::-1], self.N)

    def forward(self, x):
        full = np.fft.irfft(np.fft.rfft(x, self.N, axis=-1) * self.W, self.N, axis=-1)
        return full[..., self.o:self.o + self.n]

    def adjoint(self, y):
        z = np.zeros(y.shape[:-1] + (self.n + self.m - 1,))
        z[..., self.o:self.o + self.n] = y
        full = np.fft.irfft(np.fft.rfft(z, self.N, axis=-1) * self.Wr, self.N, axis=-1)
        return full[..., self.m - 1:self.m - 1 + self.n]


def D(z):
    r = np.zeros_like(z)
    r[..., 1:] = 0.5 * (z[..., 1:] - z[..., :-1])
    return r


def DT(r):
    out = np.zeros_like(r)
    out[..., 1:] += 0.5 * r[..., 1:]
    out[..., :-1] -= 0.5 * r[..., 1:]
    return out


def synthetic_ln(z, w):
    """Seismic of ln-impedance traces z (b, n) with wavelet w."""
    return ConvOp(w, z.shape[-1]).forward(D(z))


def scale_wavelet(w, seismic, ai):
    """Scale factor k with seismic ~ k * (w * r(ai)) at a well (least squares on finite samples) -> k w."""
    ok = np.isfinite(ai) & (ai > 0) & np.isfinite(seismic)
    z = np.log(np.where(ok, ai, 1.0))
    syn = synthetic_ln(z[None, :], w)[0]
    m = ok.copy()
    m[1:] &= ok[:-1]
    k = float(np.sum(syn[m] * seismic[m]) / max(1e-12, np.sum(syn[m] ** 2)))
    return np.asarray(w, float) * k, k


# ====================================================================== model-based inversion
def invert_traces(seis, z_prior, w, lam=0.05, mu=0.0, irls_iter=6, cg_iter=80, eps=None, mask=None, lateral=None,
                  shape=None):
    """Model-based inversion of traces seis (b, n) around the prior z_prior (b, n) = ln LFM.
    lam: prior weight relative to the data term's diagonal; mu: L1 (TV) weight on the reflectivity relative to it
    (0 = Tikhonov only); mask (b, n) bool: samples where the seismic is valid (others carry no data);
    lateral: weight of the update smoothing between neighbouring traces (needs shape=(n_il, n_xl) of the block).
    Returns ln AI (b, n)."""
    s = np.asarray(seis, float)
    z0 = np.asarray(z_prior, float)
    b, n = s.shape
    G = ConvOp(w, n)
    M = np.ones((b, n)) if mask is None else np.asarray(mask, float)
    s = np.where(M > 0, s, 0.0)
    diag = 0.5 * float(np.sum(np.asarray(w, float) ** 2))  # ~ diagonal of (G D)^T (G D)
    lam_ = lam * diag
    mu_ = mu * diag
    eps = eps if eps is not None else 1e-3
    Q = np.zeros((b, n))

    def A(x):
        y = DT(G.adjoint(M * G.forward(D(x)))) + lam_ * x
        if mu_ > 0:
            y = y + mu_ * DT(Q * D(x))
        return y

    rhs_base = DT(G.adjoint(M * s)) + lam_ * z0
    z = z0.copy()
    for it in range(irls_iter if mu_ > 0 else 1):
        if mu_ > 0:
            Q = 1.0 / np.sqrt(D(z) ** 2 + eps ** 2)
        x = z.copy()
        r = rhs_base - A(x)
        p = r.copy()
        rr = np.sum(r * r, axis=1, keepdims=True)
        rr0 = rr.copy()
        for _ in range(cg_iter):
            Ap = A(p)
            alpha = rr / np.maximum(np.sum(p * Ap, axis=1, keepdims=True), 1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            rn = np.sum(r * r, axis=1, keepdims=True)
            if np.all(rn <= 1e-12 * rr0):
                break
            p = r + (rn / np.maximum(rr, 1e-30)) * p
            rr = rn
        z = x
    if lateral and shape is not None and lateral > 0:
        dz = (z - z0).reshape(shape + (n,))
        p_ = np.pad(dz, ((1, 1), (1, 1), (0, 0)), mode="edge")
        nb = (p_[:-2, 1:-1] + p_[2:, 1:-1] + p_[1:-1, :-2] + p_[1:-1, 2:]) / 4.0
        z = z0 + ((1 - lateral) * dz + lateral * nb).reshape(b, n)
    return z


def residual_nrms(seis, z, w, mask=None):
    """NRMS (%) between the seismic and the synthetic of the inverted ln AI, per trace."""
    syn = synthetic_ln(z, w)
    s = np.asarray(seis, float)
    m = np.ones(s.shape, bool) if mask is None else mask
    num = np.sqrt(np.sum(np.where(m, syn - s, 0) ** 2, axis=1))
    den = np.sqrt(np.sum(np.where(m, s, 0) ** 2, axis=1)) + 1e-12
    return 100.0 * num / den


# ====================================================================== coloured inversion
def well_spectrum_slope(ai_logs, dt_ms, fmin=5.0, fmax=60.0):
    """alpha of the AI amplitude spectrum trend A(f) ~ f^alpha over [fmin, fmax] (log-log fit) from well AI
    logs (list of arrays, NaN-free pieces)."""
    xs, ys = [], []
    for a in ai_logs:
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        if len(a) < 32:
            continue
        f, A = amplitude_spectrum(np.log(a)[None, :], dt_ms)
        m = (f >= fmin) & (f <= fmax) & (A > 0)
        xs.append(np.log(f[m]))
        ys.append(np.log(A[m]))
    if not xs:
        return -1.0
    x, y = np.concatenate(xs), np.concatenate(ys)
    return float(np.polyfit(x, y, 1)[0])


def coloured_operator(seis_traces, dt_ms, alpha, fband=(5.0, 60.0), smooth_hz=10.0):
    """Frequency-domain coloured-inversion operator (complex, rfft of length nfft) for traces of the volume.
    The seismic spectrum is smoothed over smooth_hz (the operator must follow the wavelet, not the reflectivity)."""
    seis_traces = np.atleast_2d(np.asarray(seis_traces, float))
    n = seis_traces.shape[1]
    N = _nfft(2 * n)
    f, S = amplitude_spectrum(seis_traces, dt_ms, N)
    k = max(1, int(round(smooth_hz / max(1e-9, f[1] - f[0]))))
    S = np.convolve(np.pad(S, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "same")[k:-k]
    fl, fh = fband
    target = np.where(f > 0, np.maximum(f, 1e-3) ** alpha, 0.0)
    taper = np.clip((f - 0.5 * fl) / max(1e-9, 0.5 * fl), 0, 1) * np.clip((1.25 * fh - f) / max(1e-9, 0.25 * fh), 0, 1)
    amp = np.where(S > 1e-9 * S.max(), target / np.maximum(S, 1e-9 * S.max()), 0.0) * taper
    return f, N, amp * (-1j)  # -90 degrees: integration (reflectivity -> impedance)


def coloured_inversion(traces, dt_ms, op, scale=1.0):
    """Relative ln AI of traces (b, n) with an operator from coloured_operator."""
    f, N, H = op
    x = np.asarray(traces, float)
    n = x.shape[-1]
    y = np.fft.irfft(np.fft.rfft(x, N, axis=-1) * H, N, axis=-1)[..., :n]
    return y * scale


def ci_scale(ci_at_wells, ai_logs, dt_ms, fband):
    """Scale of the coloured inversion: RMS of the band-passed well ln AI / RMS of the operator output."""
    num, den = [], []
    for c, a in zip(ci_at_wells, ai_logs):
        ok = np.isfinite(a) & np.isfinite(c)
        if ok.sum() < 32:
            continue
        la = np.log(a[ok])
        bp = butter_filter(butter_filter(la, dt_ms, fband[1], 4, "low"), dt_ms, fband[0], 4, "high")
        num.append(np.std(bp))
        den.append(np.std(c[ok]))
    return float(np.mean(num) / max(1e-12, np.mean(den))) if num else 1.0


# ====================================================================== QC
def band_error(z_inv, ai_true, dt_ms, fmax):
    """Mean |AI_inv / AI_true - 1| (%) after low-passing both ln AI to fmax (the seismic band)."""
    ok = np.isfinite(ai_true) & (ai_true > 0)
    a = butter_filter(np.log(np.where(ok, ai_true, np.nanmedian(ai_true))), dt_ms, fmax)
    b = butter_filter(np.asarray(z_inv, float), dt_ms, fmax)
    return 100.0 * float(np.mean(np.abs(np.exp(b[ok] - a[ok]) - 1)))


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


# ====================================================================== volume wrappers
class PostStack:
    """Model-based inversion of a 3D volume around an LFM volume (ln AI prior)."""

    def __init__(self, project, vol_name, lfm_name, wavelet, lam=0.05, mu=0.0, lateral=0.0, cg_iter=80,
                 irls_iter=6):
        self.project, self.vol_name, self.lfm_name = project, vol_name, lfm_name
        self.vol = project.volume(vol_name)
        self.lfm = project.volume(lfm_name)
        if (self.lfm.nil, self.lfm.nxl, self.lfm.ns) != (self.vol.nil, self.vol.nxl, self.vol.ns):
            raise ValueError("the LFM must be on the seismic volume's grid")
        self.w = np.asarray(wavelet, float)
        self.lam, self.mu, self.lateral = lam, mu, lateral
        self.cg_iter, self.irls_iter = cg_iter, irls_iter

    def traces(self, ii, jj):
        ii, jj = np.asarray(ii), np.asarray(jj)
        s = np.asarray(self.vol.ilxl[ii, jj], np.float32).astype(float) * self.vol.inv
        z0 = np.log(np.maximum(np.asarray(self.lfm.ilxl[ii, jj], np.float32).astype(float) * self.lfm.inv, 1e-6))
        return invert_traces(s, z0, self.w, self.lam, self.mu, self.irls_iter, self.cg_iter)

    def compute(self, a, b):
        v = self.vol
        s = np.asarray(v.ilxl[a:b], np.float32).astype(float).reshape(-1, v.ns) * v.inv
        z0 = np.log(np.maximum(np.asarray(self.lfm.ilxl[a:b], np.float32).astype(float).reshape(-1, v.ns) *
                               self.lfm.inv, 1e-6))
        z = invert_traces(s, z0, self.w, self.lam, self.mu, self.irls_iter, self.cg_iter, lateral=self.lateral,
                          shape=(b - a, v.nxl))
        return np.exp(z).reshape(b - a, v.nxl, v.ns)

    def write(self, name=None, progress=None, cancel=None):
        from .common import write_volume
        name = name or ("AI blocky" if self.mu > 0 else "AI model-based")
        meta = {"inversion": {"kind": "model-based", "lfm": self.lfm_name, "lam": self.lam, "mu": self.mu,
                              "lateral": self.lateral, "wavelet": self.w.tolist()}}
        return write_volume(self.project, self.vol_name, name, self.compute, block=8, progress=progress,
                            cancel=cancel, meta_extra=meta)


class Coloured:
    """Coloured inversion of a 3D volume (relative ln AI)."""

    def __init__(self, project, vol_name, curves=(), alpha=None, fband=(5.0, 60.0), n_sample=400):
        self.project, self.vol_name = project, vol_name
        v = self.vol = project.volume(vol_name)
        dt = v.dt * 1000.0
        logs = [c.values["ai"] for c in curves]
        self.alpha = alpha if alpha is not None else well_spectrum_slope(logs, dt, *fband)
        rng = np.random.default_rng(0)
        k = min(n_sample, v.nil * v.nxl)
        idx = rng.choice(v.nil * v.nxl, k, replace=False)
        tr = np.asarray(v.ilxl[idx // v.nxl, idx % v.nxl], np.float32).astype(float) * v.inv
        self.op = coloured_operator(tr, dt, self.alpha, fband)
        self.fband = fband
        self.scale = 1.0
        if curves:
            ci = []
            for c in curves:
                il, xl = v.ilxl_at(c.x, c.y)
                t = np.asarray(v.ilxl[v.il_index(il), v.xl_index(xl)], np.float32).astype(float) * v.inv
                ci.append(coloured_inversion(t[None, :], dt, self.op)[0])
            self.scale = ci_scale(ci, logs, dt, fband)

    def traces(self, ii, jj):
        v = self.vol
        t = np.asarray(v.ilxl[np.asarray(ii), np.asarray(jj)], np.float32).astype(float) * v.inv
        return coloured_inversion(t, v.dt * 1000.0, self.op, self.scale)

    def compute(self, a, b):
        v = self.vol
        t = np.asarray(v.ilxl[a:b], np.float32).astype(float).reshape(-1, v.ns) * v.inv
        return coloured_inversion(t, v.dt * 1000.0, self.op, self.scale).reshape(b - a, v.nxl, v.ns)

    def write(self, name="Relative AI (coloured)", progress=None, cancel=None):
        from .common import write_volume
        meta = {"inversion": {"kind": "coloured", "alpha": self.alpha, "band": list(self.fband), "scale": self.scale}}
        return write_volume(self.project, self.vol_name, name, self.compute, progress=progress, cancel=cancel,
                            meta_extra=meta)


def blind_well_test(project, vol_name, curves, wavelet, horizons=(), modes=None, fc=8.0, lam=0.05, mu=0.0,
                    method="idw", fmax=60.0):
    """Leave each well out: LFM from the others, invert its trace, compare with its log in the seismic band.
    Returns rows {well, corr, error_pct, nrms, lfm_error_pct}."""
    from .lfm import VolumeLFM
    vol = project.volume(vol_name)
    dt = vol.dt * 1000.0
    rows = []
    for k, c in enumerate(curves):
        others = [o for j, o in enumerate(curves) if j != k]
        if not others:
            continue
        vl = VolumeLFM(project, vol_name, others, "ai", horizons, modes, method, fc=fc)
        il, xl = vol.ilxl_at(c.x, c.y)
        i, j = vol.il_index(il), vol.xl_index(xl)
        prior = vl.traces([i], [j])
        s = np.asarray(vol.ilxl[i, j], np.float32).astype(float)[None, :] * vol.inv
        z = invert_traces(s, np.log(prior), wavelet, lam, mu)
        ai = c.values["ai"]
        ok = np.isfinite(ai)
        bl = butter_filter(np.log(np.where(ok, ai, np.nanmedian(ai))), dt, fmax)
        rows.append({"well": c.name, "corr": corr(butter_filter(z[0], dt, fmax)[ok], bl[ok]),
                     "error_pct": band_error(z[0], ai, dt, fmax), "nrms": float(residual_nrms(s, z, wavelet)[0]),
                     "lfm_error_pct": band_error(np.log(prior[0]), ai, dt, fmax)})
    return rows
