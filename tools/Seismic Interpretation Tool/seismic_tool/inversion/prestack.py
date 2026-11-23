"""Pre-stack simultaneous inversion (PLAN 19c): partial angle stacks -> Zp, Zs, rho (Hampson, Russell &
Bankhead 2005 style).

Fatti et al. (1994) linearisation with r = D(ln x) = (ln x_k - ln x_(k-1)) / 2 (as in post-stack):
    R(theta) = A D(ln Zp) + B D(ln Zs) + C D(ln rho),   A = 1 + tan^2, B = -8 K^2 sin^2, C = -(tan^2 - 4 K^2 sin^2)
K = Vs / Vp of the background. Background trends from the wells: ln Zs = k ln Zp + kc + dLs, ln rho = m ln Zp +
mc + dLd; unknowns per trace are ln Zp, dLs, dLd (decorrelated), each pulled towards its LFM prior with its own
weight (density the strongest - least constrained by the angle range). CG, matrix-free, vectorised over traces,
one wavelet per angle stack.
"""
import numpy as np

from .poststack import D, DT, ConvOp


def fatti_coeffs(theta_deg, K):
    th = np.radians(np.asarray(theta_deg, float))
    t2, s2 = np.tan(th) ** 2, np.sin(th) ** 2
    return 1.0 + t2, -8.0 * K * K * s2, -(t2 - 4.0 * K * K * s2)


def fit_background(curves):
    """{k, kc, m, mc, K}: ln Zs = k ln Zp + kc, ln rho = m ln Zp + mc from the wells' logs; K = median Vs / Vp."""
    lp, ls, ld, ratio = [], [], [], []
    for c in curves:
        v = c.values
        ok = np.isfinite(v["ai"]) & np.isfinite(v["vs"]) & np.isfinite(v["rho"]) & np.isfinite(v["vp"])
        if ok.sum() < 5:
            continue
        lp.append(np.log(v["ai"][ok]))
        ls.append(np.log(v["vs"][ok] * v["rho"][ok]))
        ld.append(np.log(v["rho"][ok]))
        ratio.append(v["vs"][ok] / v["vp"][ok])
    if not lp:
        raise ValueError("no well with Vp, Vs and density in time")
    lp, ls, ld = np.concatenate(lp), np.concatenate(ls), np.concatenate(ld)
    k, kc = np.polyfit(lp, ls, 1)
    m, mc = np.polyfit(lp, ld, 1)
    return {"k": float(k), "kc": float(kc), "m": float(m), "mc": float(mc),
            "K": float(np.median(np.concatenate(ratio)))}


def invert_prestack(stacks, angles, wavelets, zp0, zs0, rho0, bg, lam=(0.02, 0.1, 1.0), cg_iter=120):
    """stacks (n_st, b, n) seismic; angles (n_st,) mean angle of each stack (deg); wavelets [n_st arrays];
    zp0 / zs0 / rho0 (b, n) priors (LFM). Returns ln Zp, ln Zs, ln rho (b, n)."""
    stacks = np.asarray(stacks, float)
    nst, b, n = stacks.shape
    ops = [ConvOp(w, n) for w in wavelets]
    co = [fatti_coeffs(a, bg["K"]) for a in angles]
    A_ = [A + B * bg["k"] + C * bg["m"] for A, B, C in co]
    Lp0 = np.log(zp0)
    dLs0 = np.log(zs0) - bg["k"] * Lp0 - bg["kc"]
    dLd0 = np.log(rho0) - bg["m"] * Lp0 - bg["mc"]
    diag = 0.5 * float(np.mean([np.sum(np.asarray(w, float) ** 2) for w in wavelets])) * nst
    lp_, ls_, ld_ = (float(l) * diag for l in lam)

    def fwd(Lp, dLs, dLd, i):
        B, C = co[i][1], co[i][2]
        return ops[i].forward(A_[i] * D(Lp) + B * D(dLs) + C * D(dLd))

    def adj(y, i):
        g = DT(ops[i].adjoint(y))
        return A_[i] * g, co[i][1] * g, co[i][2] * g

    def normal(X):
        Lp, dLs, dLd = X
        o = [lp_ * Lp, ls_ * dLs, ld_ * dLd]
        for i in range(nst):
            gp, gs, gd = adj(fwd(Lp, dLs, dLd, i), i)
            o[0] = o[0] + gp
            o[1] = o[1] + gs
            o[2] = o[2] + gd
        return o

    rhs = [lp_ * Lp0, ls_ * dLs0, ld_ * dLd0]
    for i in range(nst):
        gp, gs, gd = adj(stacks[i], i)
        rhs = [rhs[0] + gp, rhs[1] + gs, rhs[2] + gd]
    X = [Lp0.copy(), dLs0.copy(), dLd0.copy()]
    AX = normal(X)
    R = [r - a for r, a in zip(rhs, AX)]
    P = [r.copy() for r in R]

    def dot(U, V):
        return sum(np.sum(u * v, axis=1, keepdims=True) for u, v in zip(U, V))
    rr = dot(R, R)
    rr0 = rr.copy()
    for _ in range(cg_iter):
        AP = normal(P)
        alpha = rr / np.maximum(dot(P, AP), 1e-30)
        X = [x + alpha * p for x, p in zip(X, P)]
        R = [r - alpha * a for r, a in zip(R, AP)]
        rn = dot(R, R)
        if np.all(rn <= 1e-14 * rr0):
            break
        P = [r + (rn / np.maximum(rr, 1e-30)) * p for r, p in zip(R, P)]
        rr = rn
    Lp, dLs, dLd = X
    return Lp, bg["k"] * Lp + bg["kc"] + dLs, bg["m"] * Lp + bg["mc"] + dLd


def derived(lzp, lzs, lrho):
    """Zp, Zs, rho, Vp/Vs, lambda-rho, mu-rho from the ln results (units as the inputs: m/s x g/cc)."""
    zp, zs, rho = np.exp(lzp), np.exp(lzs), np.exp(lrho)
    return {"Zp": zp, "Zs": zs, "Density": rho, "VpVs": zp / zs, "LambdaRho": zp ** 2 - 2 * zs ** 2,
            "MuRho": zs ** 2}


class PreStack:
    """Simultaneous inversion of partial angle stacks of a 3D survey.
    stacks: [(volume name, mean angle deg)]; wavelets: one per stack; priors: {"ai", "vs", "rho"} LFM volume names
    (Zs prior = Vs x rho)."""

    def __init__(self, project, stacks, wavelets, priors, bg, lam=(0.02, 0.1, 1.0), cg_iter=120):
        self.project = project
        self.names = [s[0] for s in stacks]
        self.angles = [float(s[1]) for s in stacks]
        self.vols = [project.volume(n) for n in self.names]
        v0 = self.vols[0]
        for v in self.vols[1:]:
            if (v.nil, v.nxl, v.ns) != (v0.nil, v0.nxl, v0.ns):
                raise ValueError("angle stacks must share the same grid")
        self.wavelets = [np.asarray(w, float) for w in wavelets]
        self.pv = {k: project.volume(priors[k]) for k in ("ai", "vs", "rho")}
        self.bg, self.lam, self.cg_iter = bg, lam, cg_iter

    def _read(self, vol, sl):
        return np.asarray(vol.ilxl[sl], np.float32).astype(float).reshape(-1, vol.ns) * vol.inv

    def _run(self, sl):
        st = np.stack([self._read(v, sl) for v in self.vols])
        zp0 = np.maximum(self._read(self.pv["ai"], sl), 1e-3)
        rho0 = np.maximum(self._read(self.pv["rho"], sl), 1e-3)
        zs0 = np.maximum(self._read(self.pv["vs"], sl), 1e-3) * rho0
        return invert_prestack(st, self.angles, self.wavelets, zp0, zs0, rho0, self.bg, self.lam, self.cg_iter)

    def traces(self, ii, jj):
        return self._run((np.asarray(ii), np.asarray(jj)))

    def write(self, outputs=("Zp", "Zs", "Density", "VpVs", "LambdaRho", "MuRho"), progress=None, cancel=None):
        """One volume per output ('<first stack> [SI <output>]'), all from one inversion pass."""
        from .common import write_volumes
        v = self.vols[0]

        def compute(a, b):
            res = derived(*self._run(slice(a, b)))
            return {f"SI {k}": res[k].reshape(b - a, v.nxl, v.ns) for k in outputs}
        return write_volumes(self.project, self.names[0], [f"SI {k}" for k in outputs], compute, block=8,
                             progress=progress, cancel=cancel, meta_extra={"inversion": {
                                 "kind": "simultaneous", "stacks": self.names, "angles": self.angles,
                                 "background": self.bg, "lam": list(self.lam)}})
