"""RGT step B: node mesh, dip-guided correlation links, global weighted least-squares solve (CGLS, numpy).

Unknowns: relative geologic time tau at nodes = (node column, node time).  Node columns sit every
`node_traces` traces (3D: on a regular il/xl grid incl. the last inline/crossline; 2D: along each line plus
the tie traces), node times every `node_ms`.

Equations (each weighted, solved in the least-squares sense):
  lateral link   tau(a, t_k) - tau(b, t_k + shift) = 0     shift = dip integrated trace by trace from a to b
                                                            (Heun steps on the dip field), refined by a
                                                            small cross-correlation search; weight = corr^2
  fault link     same, but the dip path crosses a fault: wide correlation search (+/- max throw) with a long
                 window, accepted only if corr >= fault_min_corr, weight halved   (params.fault_links)
  vertical       tau(a, t_k+1) - tau(a, t_k) = node_ms      weak (sedimentation-rate prior)
  datum          mean(tau) = mean(t) per survey            (fixes the free constant)
  horizon        tau(a, t_h) - c_h = 0                       strong; one extra unknown c_h per horizon
  2D ties        tau(line A at the tie, t) - tau(line B at the tie, t + mistie) = 0

The solution is made monotonic per column, interpolated to every sample (linear in time, bilinear between
node columns) and stored as uint16 (RGT range scaled to 0..65535, ~0.1 ms resolution for a 6 s range):
3D <store>/rgt/rgt.npy (nil, nxl, ns) + rgt.json + rgt_nodes.npz;  2D <project>_seismic/rgt/<line>_rgt.npy.
"""
import json
import os
import time

import numpy as np


class RGTParams:
    def __init__(self, node_traces=4, node_ms=8.0, window_ms=24.0, search_ms=8.0, min_corr=0.5,
                 w_vertical=0.05, w_horizon=10.0, fault_links="correlate", max_throw_ms=80.0,
                 fault_min_corr=0.8, max_iter=3000, tol=1e-9, refine=True, max_iter_fine=400):
        self.node_traces, self.node_ms = int(node_traces), float(node_ms)
        self.window_ms, self.search_ms, self.min_corr = window_ms, search_ms, min_corr
        self.w_vertical, self.w_horizon = w_vertical, w_horizon
        self.fault_links, self.max_throw_ms, self.fault_min_corr = fault_links, max_throw_ms, fault_min_corr
        self.max_iter, self.tol = max_iter, tol
        self.refine = refine  # refine the dip-predicted shift by cross-correlation
        self.max_iter_fine = max_iter_fine  # iteration cap when starting from a coarser solution
        self.dx_tol = 0.02  # CGLS stops when no node moved more than this (ms) for 10 iterations

    def to_dict(self):
        return dict(self.__dict__)


# ====================================================================== helpers
def interp_rows(D, rows, t):
    """D (n, ns); rows int array, t fractional sample index (same shape) -> linearly interpolated values."""
    ns = D.shape[1]
    tc = np.clip(t, 0, ns - 1.000001)
    k = np.floor(tc).astype(np.int64)
    w = tc - k
    return D[rows, k] * (1 - w) + D[rows, k + 1] * w


def refine_corr(S, ra, ta, rb, tb, half, lag):
    """Cross-correlate trace ra around ta with trace rb around tb (+/- lag samples).
    Returns (tb_refined, corr).  S (n, ns) float."""
    ns = S.shape[1]
    ca = np.round(ta).astype(np.int64)
    cb = np.round(tb).astype(np.int64)
    off = np.arange(-half, half + 1)
    lags = np.arange(-lag - 1, lag + 2)
    ia = np.clip(ca[:, None] + off[None], 0, ns - 1)
    A = S[ra[:, None], ia]
    ib = np.clip(cb[:, None, None] + lags[None, :, None] + off[None, None, :], 0, ns - 1)
    B = S[rb[:, None, None], ib]
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=2, keepdims=True)
    num = np.einsum("nk,nlk->nl", A, B)
    den = np.sqrt((A * A).sum(axis=1))[:, None] * np.sqrt((B * B).sum(axis=2))
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.where(den > 1e-12, num / den, -1.0)
    n = len(ta)
    k = np.argmax(cc, axis=1)
    best = cc[np.arange(n), k]
    km, kp = np.clip(k - 1, 0, len(lags) - 1), np.clip(k + 1, 0, len(lags) - 1)
    y0, y2 = cc[np.arange(n), km], cc[np.arange(n), kp]
    den2 = y0 - 2 * best + y2
    inner = (k > 0) & (k < len(lags) - 1) & (np.abs(den2) > 1e-12)
    delta = np.where(inner, 0.5 * (y0 - y2) / np.where(den2 == 0, 1, den2), 0.0)
    best = np.where(np.abs(lags[k]) > lag, -1.0, best)  # optimum outside the search: reject
    tb2 = ta + (cb + lags[k] + np.clip(delta, -0.5, 0.5) - ca)
    return tb2, best


def dip_path(D, pos_a, pos_b, t0, hit=None):
    """Integrate dip (samples per trace, D (n_traces, ns)) from trace pos_a to pos_b (arrays (P,), any order)
    for start times t0 (P, K) (sample index). Heun steps. hit(k_edge (P,K), t (P,K)) -> bool marks fault
    crossings. Returns (t_end (P, K), crossed (P, K))."""
    P, K = t0.shape
    t = t0.astype(np.float64).copy()
    crossed = np.zeros((P, K), bool)
    sgn = np.sign(pos_b - pos_a).astype(np.int64)
    n = np.abs(pos_b - pos_a)
    for m in range(int(n.max()) if len(n) else 0):
        act = m < n
        if not act.any():
            break
        j = np.clip(pos_a + sgn * m, 0, D.shape[0] - 1)[:, None].repeat(K, 1)
        j2 = j + sgn[:, None]
        k1 = interp_rows(D, j, t) * sgn[:, None]
        k2 = interp_rows(D, np.clip(j2, 0, D.shape[0] - 1), t + k1) * sgn[:, None]
        if hit is not None:
            edge = np.minimum(j, j2)
            crossed |= act[:, None] & hit(edge, t + 0.5 * k1)
        t = np.where(act[:, None], t + 0.5 * (k1 + k2), t)
    return t, crossed


class _Eq:
    """Accumulates sparse rows: each equation = up to 4 (col, coef) pairs + rhs, with a weight."""

    def __init__(self):
        self.rows, self.cols, self.vals, self.rhs = [], [], [], []
        self.n = 0

    def add(self, cols_list, coefs_list, rhs, weight):
        """cols_list/coefs_list: lists of equal-length arrays (one entry per equation each)."""
        m = len(rhs)
        if m == 0:
            return
        sw = np.sqrt(np.asarray(weight, float)) * np.ones(m)
        r = self.n + np.arange(m)
        for c, v in zip(cols_list, coefs_list):
            self.rows.append(r)
            self.cols.append(np.asarray(c, np.int64))
            self.vals.append(np.asarray(v, float) * sw)
        self.rhs.append(np.asarray(rhs, float) * sw)
        self.n += m

    def finish(self):
        rows = np.concatenate(self.rows) if self.rows else np.zeros(0, np.int64)
        cols = np.concatenate(self.cols) if self.cols else np.zeros(0, np.int64)
        vals = np.concatenate(self.vals) if self.vals else np.zeros(0)
        b = np.concatenate(self.rhs) if self.rhs else np.zeros(0)
        keep = vals != 0
        return rows[keep], cols[keep], vals[keep], b


def cgls(rows, cols, vals, b, x0, n, max_iter=600, tol=1e-7, progress=None, cancel=None, dx_tol=0.02):
    """min ||A x - b|| with A given as COO (rows, cols, vals); column-scaled (Jacobi) CGLS. Returns x, iters.
    Stops when the gradient norm fell by tol, or when no unknown changed by more than dx_tol (ms) for 10
    iterations in a row."""
    m = len(b)
    colnorm = np.sqrt(np.bincount(cols, vals * vals, minlength=n))
    s = np.where(colnorm > 0, 1.0 / np.where(colnorm > 0, colnorm, 1), 1.0)
    v = vals * s[cols]

    def A(y):
        return np.bincount(rows, v * y[cols], minlength=m)

    def At(r):
        return np.bincount(cols, v * r[rows], minlength=n)

    y = x0 / s
    r = b - A(y)
    g = At(r)
    p = g.copy()
    gamma = gamma0 = float(g @ g)
    it = 0
    quiet = 0
    for it in range(1, max_iter + 1):
        if gamma <= tol * tol * max(gamma0, 1e-300) or gamma == 0:
            break
        q = A(p)
        qq = float(q @ q)
        if qq == 0:
            break
        alpha = gamma / qq
        y += alpha * p
        quiet = quiet + 1 if float(np.max(np.abs(alpha * p * s))) < dx_tol else 0
        if quiet >= 10:
            break
        r -= alpha * q
        g = At(r)
        gnew = float(g @ g)
        p = g + (gnew / gamma) * p
        gamma = gnew
        if progress and it % 20 == 0:
            progress(it / max_iter, f"Solving RGT: iteration {it}")
        if cancel and cancel():
            raise RuntimeError("cancelled")
    return y * s, it


def fault_links(S, D, ra, ta, rb, hit, params, dt_ms, half):
    """Tie node (ra, ta) to trace rb across a fault.  Correlating right at the fault mixes the two blocks, so
    step away from the fault on both sides (a' = a - s, b' = b + s traces, following the dip), correlate a'
    with b' over +/- max throw where neither window touches the fault, then follow the dip back to b.
    Tries s = 2, 4, 8 traces; returns (tb, corr) with corr = -1 where no clean tie was found."""
    n_tr = S.shape[0]
    big = max(2, int(round(params.max_throw_ms / dt_ms)))
    whalf = 2 * half
    sgn = np.sign(rb - ra)
    tb_out = np.full(len(ra), np.nan)
    cc_out = np.full(len(ra), -1.0)

    def clean(tr, t):
        bad = np.zeros(len(tr), bool)
        if hit is None:
            return ~bad
        for e in (tr - 1, tr):
            e_ok = (e >= 0) & (e < n_tr - 1)
            bad |= e_ok & hit(np.clip(e, 0, n_tr - 2), t, float(whalf))
        return ~bad

    todo = np.arange(len(ra))
    for s in (2, 4, 8):
        if not len(todo):
            break
        a, b, t = ra[todo], rb[todo], ta[todo]
        a2 = np.clip(a - sgn[todo] * s, 0, n_tr - 1)
        b2 = np.clip(b + sgn[todo] * s, 0, n_tr - 1)
        ta2, cr1 = dip_path(D, a, a2, t[:, None], hit)
        ta2, cr1 = ta2[:, 0], cr1[:, 0]
        tb2, cc = refine_corr(S, a2, ta2, b2, ta2, whalf, big)
        tb, cr2 = dip_path(D, b2, b, tb2[:, None], hit)
        tb, cr2 = tb[:, 0], cr2[:, 0]
        ok = ~cr1 & ~cr2 & clean(a2, ta2) & clean(b2, tb2) & (cc >= params.fault_min_corr)
        tb_out[todo[ok]] = tb[ok]
        cc_out[todo[ok]] = cc[ok]
        todo = todo[~ok]
    return np.where(np.isfinite(tb_out), tb_out, ta), cc_out


# ====================================================================== link builder (shared 2D / 3D)
def links_along(S, D, pos, node_t, params, dt_ms, hit=None):
    """Links between consecutive node columns lying on one line of traces.

    S (n_traces, ns) seismic, D (n_traces, ns) dip in samples/trace, pos: sorted trace indices of the node
    columns on this line, node_t: node times in sample units (K,).  hit(k_edge, t) -> fault crossing.
    Returns list of (ia, k, ib, t_b_samples, weight) arrays: node (column ia, time k) == column ib at t_b."""
    pos = np.asarray(pos, np.int64)
    if len(pos) < 2:
        return []
    half = max(2, int(round(params.window_ms / dt_ms)))
    lag = max(1, int(round(params.search_ms / dt_ms)))
    out = []
    for fwd in (True, False):  # links both ways make the system symmetric in a and b
        a, b = (pos[:-1], pos[1:]) if fwd else (pos[1:], pos[:-1])
        ia = np.arange(len(pos) - 1) + (0 if fwd else 1)
        ib = ia + (1 if fwd else -1)
        K = len(node_t)
        t0 = np.broadcast_to(node_t[None, :], (len(a), K))
        tend, crossed = dip_path(D, a, b, t0, hit)
        ra = np.repeat(a, K)
        rb = np.repeat(b, K)
        ta, tb = t0.ravel(), tend.ravel()
        if params.refine:
            tb2, cc = refine_corr(S, ra, ta, rb, tb, half, lag)
        else:  # trust the dip path; correlation (best lag within +/-1 sample) only sets the confidence
            _, cc = refine_corr(S, ra, ta, rb, tb, half, 1)
            tb2 = tb.copy()
        w = np.where(cc >= params.min_corr, cc * cc, 0.0)
        cr = crossed.ravel()
        if cr.any():
            if params.fault_links == "correlate":
                sel = np.nonzero(cr)[0]
                tbf, ccf = fault_links(S, D, ra[sel], ta[sel], rb[sel], hit, params, dt_ms, half)
                tb2[sel] = tbf
                w[sel] = np.where(ccf >= params.fault_min_corr, 0.5 * ccf * ccf, 0.0)
            else:
                w[cr] = 0.0
        kk = np.tile(np.arange(K), len(a))
        out.append((np.repeat(ia, K), kk, np.repeat(ib, K), tb2, w))
    return out


# ====================================================================== system
class NodeSystem:
    """Node columns of one or more surveys + the equations. Column c of survey s has unknowns
    offset[c] + k for k in range(K_s)."""

    def __init__(self, params):
        self.p = params
        self.eq = _Eq()
        self.surveys = {}  # key -> dict(cols=(nc,) info, K, node_t_ms (K,), offset (nc,))
        self.n = 0
        self.x0 = []
        self.hz_cols = {}  # horizon name -> unknown index

    def add_survey(self, key, ncols, node_t_ms):
        K = len(node_t_ms)
        off = self.n + K * np.arange(ncols)
        self.surveys[key] = {"K": K, "node_t": np.asarray(node_t_ms, float), "offset": off, "ncols": ncols}
        self.n += K * ncols
        self.x0.append(np.tile(np.asarray(node_t_ms, float), ncols))
        return off

    def horizon_unknown(self, name, guess):
        if name not in self.hz_cols:
            self.hz_cols[name] = self.n
            self.n += 1
            self.x0.append(np.array([guess], float))
        return self.hz_cols[name]

    def _interp_cols(self, key, col, t_ms):
        """Unknown indices and weights for tau(col, t_ms) (linear between nodes); valid mask."""
        sv = self.surveys[key]
        u = (np.asarray(t_ms, float) - sv["node_t"][0]) / self.p.node_ms
        valid = np.isfinite(u) & (u >= 0) & (u <= sv["K"] - 1)
        u = np.where(valid, u, 0)
        m = np.minimum(np.floor(u).astype(np.int64), sv["K"] - 2)
        w = u - m
        base = sv["offset"][col]
        return base + m, base + m + 1, 1 - w, w, valid

    def add_links(self, key_a, col_a, k_a, key_b, col_b, t_b_ms, weight):
        sa = self.surveys[key_a]
        ua = sa["offset"][col_a] + k_a
        b0, b1, w0, w1, ok = self._interp_cols(key_b, col_b, t_b_ms)
        ok &= weight > 0
        self.eq.add([ua[ok], b0[ok], b1[ok]], [np.ones(ok.sum()), -w0[ok], -w1[ok]], np.zeros(ok.sum()), weight[ok])
        return int(ok.sum())

    def add_regularisation(self):
        p = self.p
        for key, sv in self.surveys.items():
            K, nc = sv["K"], sv["ncols"]
            base = (sv["offset"][:, None] + np.arange(K - 1)[None, :]).ravel()
            m = len(base)
            self.eq.add([base + 1, base], [np.ones(m), -np.ones(m)], np.full(m, p.node_ms), p.w_vertical)
            if not getattr(self, "datum", True):
                continue
            # datum: the mean of tau equals the mean node time (one dense row; a per-node 'tau = t' damping
            # would flatten the structure more the finer the node grid is)
            allk = (sv["offset"][:, None] + np.arange(K)[None, :]).ravel()
            m = len(allk)
            self.eq.rows.append(np.full(m, self.eq.n))
            self.eq.cols.append(allk)
            self.eq.vals.append(np.full(m, 1.0 / m))
            self.eq.rhs.append(np.array([float(np.mean(np.tile(sv["node_t"], nc)))]))
            self.eq.n += 1

    def add_horizon(self, name, key, cols, t_ms):
        """Horizon picks at node columns -> tau(col, t) == c_name."""
        t_ms = np.asarray(t_ms, float)
        ok = np.isfinite(t_ms)
        if not ok.any():
            return 0
        hc = self.horizon_unknown(name, float(np.mean(t_ms[ok])))
        b0, b1, w0, w1, v = self._interp_cols(key, np.asarray(cols)[ok], t_ms[ok])
        m = int(v.sum())
        self.eq.add([b0[v], b1[v], np.full(m, hc)], [w0[v], w1[v], -np.ones(m)], np.zeros(m), self.p.w_horizon)
        return m

    def solve(self, progress=None, cancel=None, x0=None):
        """x0: optional initial guess for the node unknowns (e.g. from a coarser solve)."""
        self.add_regularisation()
        rows, cols, vals, b = self.eq.finish()
        x_init = np.concatenate(self.x0)
        self.coarse_start = x0 is not None
        if x0 is not None:
            x_init[:len(x0)] = x0
        x0 = x_init
        t = time.time()
        max_iter = self.p.max_iter if not self.coarse_start else min(self.p.max_iter, self.p.max_iter_fine)
        x, it = cgls(rows, cols, vals, b, x0, self.n, max_iter, self.p.tol, progress, cancel,
                     dx_tol=getattr(self.p, "dx_tol", 0.02))
        self.stats = {"unknowns": int(self.n), "equations": int(len(b)), "iterations": int(it),
                      "seconds": round(time.time() - t, 2)}
        self.x = x
        return x

    def node_values(self, key):
        """(ncols, K) solved tau for a survey, made monotonic in time."""
        sv = self.surveys[key]
        v = self.x[sv["offset"][:, None] + np.arange(sv["K"])[None, :]]
        return np.maximum.accumulate(v, axis=1)


def node_times(ns, dt_ms, t0_ms, node_ms):
    tmax = t0_ms + (ns - 1) * dt_ms
    nt = np.arange(t0_ms, tmax + 1e-6, node_ms)
    if tmax - nt[-1] > 1e-6:
        nt = np.r_[nt, nt[-1] + node_ms]  # last node just below the data (links clip to the trace end)
    return nt


def node_positions(n, step, extra=()):
    pos = set(range(0, n, max(1, step))) | {n - 1} | {int(e) for e in extra if 0 <= e < n}
    return np.array(sorted(pos), np.int64)


# ====================================================================== 3D
def build_3d(vol, dipfield, params, faults=None, horizons=None, progress=None, vol_name=None, Ipos=None,
             Jpos=None):
    """NodeSystem for a 3D volume (dipfield: rgt.dip.DipField of the volume).  Ipos/Jpos: optional node
    inline/crossline indices (a sub-window for local re-solves); default = the full regular node grid."""
    from ..faults import barrier_3d
    dt_ms, t0_ms = vol.dt * 1000.0, vol.t0 * 1000.0
    ntm = node_times(vol.ns, dt_ms, t0_ms, params.node_ms)
    nts = (ntm - t0_ms) / dt_ms
    Ipos = node_positions(vol.nil, params.node_traces) if Ipos is None else np.asarray(Ipos, np.int64)
    Jpos = node_positions(vol.nxl, params.node_traces) if Jpos is None else np.asarray(Jpos, np.int64)
    nI, nJ = len(Ipos), len(Jpos)
    sysm = NodeSystem(params)
    key = "3D"
    sysm.add_survey(key, nI * nJ, ntm)
    sysm.grid = (Ipos, Jpos)
    eb = barrier_3d(faults, vol) if faults else None
    nlinks = 0
    # links along crosslines direction on every node inline, then along inlines on every node crossline
    for n_line, (axis, lines) in enumerate(((1, Ipos), (0, Jpos))):
        for q, line in enumerate(lines):
            if axis == 1:
                S = np.asarray(vol.ilxl[line], np.float32)
                D = np.asarray(dipfield.arr["dip_xl"][line], np.float32) / dt_ms
                pos = Jpos
                colid = lambda idx: q * nJ + idx  # noqa: E731
            else:
                S = np.asarray(vol.ilxl[:, line], np.float32)
                D = np.asarray(dipfield.arr["dip_il"][:, line], np.float32) / dt_ms
                pos = Ipos
                colid = lambda idx: idx * nJ + q  # noqa: E731
            hit = (lambda k, t, x=0.0, a=axis, ln=line: eb.hit(a, ln, k, t, x)) if eb else None
            for ia, kk, ib, tb, w in links_along(S, D, pos, nts, params, dt_ms, hit):
                nlinks += sysm.add_links(key, colid(ia), kk, key, colid(ib), tb * dt_ms + t0_ms, w)
            if progress:
                progress(0.5 * (n_line + (q + 1) / len(lines)) * 0.5, f"Linking nodes ({nlinks:,} links)")
    for h in horizons or []:
        g = h.get(f"3D:{vol_name}") if vol_name else None
        if g is None:
            continue
        sub = g[np.ix_(Ipos, Jpos)].ravel()
        sysm.add_horizon(h.name, key, np.arange(nI * nJ), sub)
    sysm.nlinks = nlinks
    return sysm


def interp_node_grid(tau_c, Ipos_c, Jpos_c, nt_c, Ipos, Jpos, nt):
    """Trilinear interpolation of a coarse node solution (nIc, nJc, Kc) onto finer node positions."""
    def axis_w(pc, p):
        p = np.asarray(p, float)
        i1 = np.clip(np.searchsorted(pc, p, "right"), 1, len(pc) - 1)
        i0 = i1 - 1
        w = np.clip((p - pc[i0]) / np.maximum(1e-9, pc[i1] - pc[i0]), 0, 1)
        return i0, i1, w
    a0, a1, wa = axis_w(np.asarray(Ipos_c, float), Ipos)
    b0, b1, wb = axis_w(np.asarray(Jpos_c, float), Jpos)
    c0, c1, wc = axis_w(np.asarray(nt_c, float), nt)
    t = tau_c[:, :, c0] * (1 - wc) + tau_c[:, :, c1] * wc  # (nIc, nJc, K)
    t = t[:, b0] * (1 - wb)[None, :, None] + t[:, b1] * wb[None, :, None]
    t = t[a0] * (1 - wa)[:, None, None] + t[a1] * wa[:, None, None]
    return t


def solve_3d_multires(vol, dipfield, params, faults=None, horizons=None, progress=None, cancel=None,
                      vol_name=None, min_unknowns=30000):
    """Coarse-to-fine: solve on a node grid twice as coarse (recursively) and use it as the start for the
    fine solve - CGLS then only has to fix short-wavelength detail (far fewer iterations on big surveys)."""
    import copy
    n_est = len(node_positions(vol.nil, params.node_traces)) * len(node_positions(vol.nxl, params.node_traces)) * \
        len(node_times(vol.ns, vol.dt * 1000.0, vol.t0 * 1000.0, params.node_ms))
    x0 = None
    if n_est > min_unknowns:
        pc = copy.copy(params)
        pc.node_traces = params.node_traces * 2
        pc.node_ms = params.node_ms * 2
        coarse = solve_3d_multires(vol, dipfield, pc, faults, horizons, None, cancel, vol_name, min_unknowns)
        Ic, Jc = coarse.grid
        tau_c = coarse.node_values("3D").reshape(len(Ic), len(Jc), -1)
        nt_c = coarse.surveys["3D"]["node_t"]
    sysm = build_3d(vol, dipfield, params, faults, horizons, progress, vol_name)
    if n_est > min_unknowns:
        I, J = sysm.grid
        x0 = interp_node_grid(tau_c, Ic, Jc, nt_c, I, J, sysm.surveys["3D"]["node_t"]).ravel()
        hz0 = [coarse.x[coarse.hz_cols[n]] for n in sorted(sysm.hz_cols, key=sysm.hz_cols.get)
               if n in coarse.hz_cols]
        if len(hz0) == len(sysm.hz_cols):
            x0 = np.r_[x0, hz0]
    sysm.solve((lambda f, t: progress(0.25 + 0.5 * f, t)) if progress else None, cancel, x0)
    sysm.stats["coarse_start"] = x0 is not None
    return sysm


class NodeInterp:
    """Full-resolution RGT from a node solution tau (nI, nJ, K): linear in time, bilinear between columns."""

    def __init__(self, tau, Ipos, Jpos, node_t, node_ms, times, nil, nxl):
        self.tau, self.Ipos, self.Jpos = tau, np.asarray(Ipos), np.asarray(Jpos)
        K = tau.shape[2]
        u = np.clip((times - node_t[0]) / node_ms, 0, K - 1)
        self.m = np.minimum(np.floor(u).astype(int), K - 2)
        self.w = (u - self.m).astype(np.float32)
        jj = np.arange(nxl)
        nJ = len(self.Jpos)
        self.J1 = np.clip(np.searchsorted(self.Jpos, jj, "right"), 1, nJ - 1)
        self.J0 = self.J1 - 1
        self.wj = ((jj - self.Jpos[self.J0]) / np.maximum(1, self.Jpos[self.J1] - self.Jpos[self.J0]))[:, None]
        self.wj = self.wj.astype(np.float32)
        self.cache = {}

    def full_row(self, I):  # (nJ, ns) tau of node row I at every sample
        if I not in self.cache:
            if len(self.cache) > 4:
                self.cache.clear()
            r = self.tau[I]
            self.cache[I] = r[:, self.m] * (1 - self.w) + r[:, self.m + 1] * self.w
        return self.cache[I]

    def inline(self, i):
        """(nxl, ns) RGT of inline index i."""
        nI = len(self.Ipos)
        I1 = int(np.clip(np.searchsorted(self.Ipos, i, "right"), 1, nI - 1))
        I0 = I1 - 1
        wi = (i - self.Ipos[I0]) / max(1, self.Ipos[I1] - self.Ipos[I0])
        a, b = self.full_row(I0), self.full_row(I1)
        r0 = a[self.J0] * (1 - self.wj) + a[self.J1] * self.wj
        r1 = b[self.J0] * (1 - self.wj) + b[self.J1] * self.wj
        return r0 * (1 - wi) + r1 * wi


def write_rgt_3d(vol, sysm, out_dir=None, progress=None):
    """Interpolate the node solution to every sample, write uint16 rgt.npy + rgt.json + rgt_nodes.npz."""
    out_dir = out_dir or os.path.join(vol.store, "rgt")
    os.makedirs(out_dir, exist_ok=True)
    Ipos, Jpos = sysm.grid
    nI, nJ = len(Ipos), len(Jpos)
    sv = sysm.surveys["3D"]
    # round to float32 first: the editor works from the float32 copy in rgt_nodes.npz and must reproduce the
    # same uint16 values for untouched traces
    tau = sysm.node_values("3D").reshape(nI, nJ, sv["K"]).astype(np.float32).astype(np.float64)
    lo, hi = float(tau.min()), float(tau.max())
    pad = 0.05 * (hi - lo + 1)  # head room for later local edits
    lo, hi = lo - pad, hi + pad
    scale = 65535.0 / (hi - lo)
    ni = NodeInterp(tau, Ipos, Jpos, sv["node_t"], sysm.p.node_ms, vol.times, vol.nil, vol.nxl)
    tmp = os.path.join(out_dir, "rgt.tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint16, shape=(vol.nil, vol.nxl, vol.ns))
    for i in range(vol.nil):
        out[i] = np.clip(np.round((ni.inline(i) - lo) * scale), 0, 65535).astype(np.uint16)
        if progress and i % 16 == 0:
            progress(0.75 + 0.25 * i / vol.nil, f"Writing RGT volume {i}/{vol.nil}")
    out.flush()
    del out
    os.replace(tmp, os.path.join(out_dir, "rgt.npy"))
    with open(os.path.join(out_dir, "rgt.json"), "w") as fh:
        json.dump({"lo": lo, "hi": hi, "params": sysm.p.to_dict(), "stats": getattr(sysm, "stats", {}),
                   "links": int(getattr(sysm, "nlinks", 0)), "horizons": sorted(sysm.hz_cols)}, fh, indent=1)
    np.savez(os.path.join(out_dir, "rgt_nodes.npz"), tau=tau.astype(np.float32), Ipos=Ipos, Jpos=Jpos,
             node_t=sv["node_t"])
    return RGTVolume(out_dir)


def compute_rgt_3d(vol, params=None, faults=None, horizons=None, progress=None, cancel=None, out_dir=None,
                   vol_name=None):
    """Full pipeline for a 3D volume: dip field (computed if missing, in <store>/rgt), node solve, RGT volume
    written to out_dir (default <store>/rgt)."""
    from .dip import DipField
    params = params or RGTParams()
    dip_folder = os.path.join(vol.store, "rgt")
    folder = out_dir or dip_folder
    if not DipField.exists(dip_folder):
        from .dip import compute_dip_volume
        compute_dip_volume(vol, dip_folder, progress=(lambda f, t: progress(0.2 * f, t)) if progress else None,
                           cancel=cancel)
    df = DipField(dip_folder)
    sysm = solve_3d_multires(vol, df, params, faults, horizons, progress, cancel, vol_name)
    return write_rgt_3d(vol, sysm, folder, progress), sysm


class RGTVolume:
    """Stored 3D RGT (uint16 scaled)."""

    def __init__(self, folder):
        self.folder = folder
        self.arr = np.load(os.path.join(folder, "rgt.npy"), mmap_mode="r")
        with open(os.path.join(folder, "rgt.json")) as fh:
            self.meta = json.load(fh)
        self.lo, self.hi = self.meta["lo"], self.meta["hi"]
        self.step = (self.hi - self.lo) / 65535.0

    @staticmethod
    def exists(folder):
        return os.path.exists(os.path.join(folder, "rgt.npy")) and os.path.exists(os.path.join(folder, "rgt.json"))

    def decode(self, a):
        return self.lo + np.asarray(a, np.float32) * np.float32(self.step)

    def trace(self, i, j):
        return self.decode(self.arr[i, j])

    def section(self, vol, mode, number):
        if mode == "Inline":
            return self.decode(self.arr[vol.il_index(number)]).T
        if mode == "Crossline":
            return self.decode(self.arr[:, vol.xl_index(number)]).T
        return self.decode(self.arr[:, :, vol.t_index(number)])

    def iso_surface(self, value, vol, block=32):
        """TWT (ms) grid (nil, nxl) where RGT == value; NaN where the value is outside the trace."""
        out = np.full((vol.nil, vol.nxl), np.nan, np.float32)
        times = vol.times
        for a in range(0, vol.nil, block):
            b = min(vol.nil, a + block)
            out[a:b] = iso_from_traces(self.decode(self.arr[a:b]), value, times)
        return out


def iso_from_traces(tau, value, times):
    """tau (..., ns) monotonic along the last axis -> time (...) where tau == value (linear), NaN outside."""
    ns = tau.shape[-1]
    k = (tau < value).sum(axis=-1)
    ok = (k > 0) & (k < ns)
    k1 = np.clip(k, 1, ns - 1)
    t0 = np.take_along_axis(tau, (k1 - 1)[..., None], -1)[..., 0]
    t1 = np.take_along_axis(tau, k1[..., None], -1)[..., 0]
    f = np.where(t1 > t0, (value - t0) / np.where(t1 > t0, t1 - t0, 1), 0.0)
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    t = times[k1 - 1] + np.clip(f, 0, 1) * dt
    return np.where(ok, t, np.nan).astype(np.float32)


# ====================================================================== 2D network
def build_2d(project, names, params, faults=None, horizons=None, dip_folder=None, progress=None):
    """One NodeSystem for several 2D lines, tied at their intersections."""
    from ..faults import barrier_line
    from ..horizons import line_intersections
    from .dip import line_dip
    lines = {n: project.line(n) for n in names}
    geo = {n: (l.x, l.y) for n, l in lines.items() if l.has_coords}
    ties = line_intersections(geo) if len(geo) > 1 else []
    tie_tr = {n: [] for n in names}
    for a, ia, b, ib in ties:
        tie_tr[a].append(ia)
        tie_tr[b].append(ib)
    sysm = NodeSystem(params)
    sysm.lines = {}
    nlinks = 0
    for q, n in enumerate(names):
        l = lines[n]
        dt_ms, t0_ms = l.dt * 1000.0, l.t0 * 1000.0
        ntm = node_times(l.ns, dt_ms, t0_ms, params.node_ms)
        pos = node_positions(len(l.cdp), params.node_traces, tie_tr[n])
        key = f"2D:{n}"
        sysm.add_survey(key, len(pos), ntm)
        sysm.lines[n] = pos
        S = l.data()
        dip, _ = line_dip(l, dip_folder)
        D = dip / dt_ms
        eb = barrier_line(faults, l) if faults else None
        hit = (lambda k, t, x=0.0, e=eb: e.hit(1, 0, k, t, x)) if eb else None
        for ia, kk, ib, tb, w in links_along(S, D, pos, (ntm - t0_ms) / dt_ms, params, dt_ms, hit):
            nlinks += sysm.add_links(key, ia, kk, key, ib, tb * dt_ms + t0_ms, w)
        for h in horizons or []:
            g = h.get(key)
            if g is not None:
                sysm.add_horizon(h.name, key, np.arange(len(pos)), g[pos])
        if progress:
            progress(0.25 * (q + 1) / len(names), f"Linking line {n}")
    # ties: tau(A at tie, t_k) == tau(B at tie, t_k + mistie), mistie by correlation
    for a, ia, b, ib in ties:
        la, lb = lines[a], lines[b]
        if abs(la.dt - lb.dt) > 1e-9:
            continue  # different sampling: tie skipped (resample in a later version)
        ka, kb = f"2D:{a}", f"2D:{b}"
        ca = int(np.searchsorted(sysm.lines[a], ia))
        cb = int(np.searchsorted(sysm.lines[b], ib))
        sva = sysm.surveys[ka]
        dt_ms, t0a = la.dt * 1000.0, la.t0 * 1000.0
        ta = (sva["node_t"] - t0a) / dt_ms
        S2 = np.stack([la.data()[ia], lb.data()[ib]])
        half = max(2, int(round(params.window_ms / dt_ms)))
        lag = max(1, int(round(params.search_ms / dt_ms)))
        tb, cc = refine_corr(S2, np.zeros(len(ta), np.int64), ta, np.ones(len(ta), np.int64),
                             ta + (lb.t0 - la.t0) * 1000.0 / dt_ms, half, lag)
        w = np.where(cc >= params.min_corr, 2.0 * cc * cc, 0.0)
        tb_ms = tb * dt_ms + t0a
        nlinks += sysm.add_links(ka, np.full(len(ta), ca), np.arange(len(ta)), kb, np.full(len(ta), cb), tb_ms, w)
    sysm.ties = ties
    sysm.nlinks = nlinks
    return sysm


def write_rgt_2d(project, sysm, folder=None):
    """Interpolate each line's node solution to every trace/sample; returns {line: tau (ntr, ns) float32}
    and writes <folder>/<line>_rgt.npz (uint16 + lo/hi) when a folder is given."""
    if folder:
        os.makedirs(folder, exist_ok=True)
    out = {}
    for n, pos in sysm.lines.items():
        l = project.line(n)
        key = f"2D:{n}"
        sv = sysm.surveys[key]
        tau = sysm.node_values(key)  # (ncols, K)
        u = np.clip((l.times - sv["node_t"][0]) / sysm.p.node_ms, 0, sv["K"] - 1)
        m = np.minimum(np.floor(u).astype(int), sv["K"] - 2)
        w = u - m
        cols = tau[:, m] * (1 - w) + tau[:, m + 1] * w  # (ncols, ns)
        k = np.arange(len(l.cdp))
        c1 = np.clip(np.searchsorted(pos, k, "right"), 1, len(pos) - 1)
        c0 = c1 - 1
        wk = ((k - pos[c0]) / np.maximum(1, pos[c1] - pos[c0]))[:, None]
        full = (cols[c0] * (1 - wk) + cols[c1] * wk).astype(np.float32)
        out[n] = full
        if not folder:
            continue
        lo, hi = float(full.min()) - 1, float(full.max()) + 1
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in n)
        np.savez(os.path.join(folder, safe + "_rgt.npz"),
                 q=np.round((full - lo) / (hi - lo) * 65535).astype(np.uint16), lo=lo, hi=hi)
    return out


def load_rgt_line(folder, name):
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    p = os.path.join(folder, safe + "_rgt.npz")
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        lo, hi = float(z["lo"]), float(z["hi"])
        return lo + z["q"].astype(np.float32) * np.float32((hi - lo) / 65535.0)


def solve_2d_multires(project, names, params, faults=None, horizons=None, folder=None, progress=None,
                      cancel=None, min_unknowns=30000):
    """Coarse-to-fine like solve_3d_multires, for a 2D line network."""
    import copy
    n_est = 0
    for n in names:
        l = project.line(n)
        n_est += len(node_positions(len(l.cdp), params.node_traces)) * \
            len(node_times(l.ns, l.dt * 1000.0, l.t0 * 1000.0, params.node_ms))
    coarse = None
    if n_est > min_unknowns:
        pc = copy.copy(params)
        pc.node_traces, pc.node_ms = params.node_traces * 2, params.node_ms * 2
        coarse = solve_2d_multires(project, names, pc, faults, horizons, folder, None, cancel, min_unknowns)
    sysm = build_2d(project, names, params, faults, horizons, folder, progress)
    x0 = None
    if coarse is not None:
        x0 = np.zeros(sum(sv["K"] * sv["ncols"] for sv in sysm.surveys.values()))
        for n in names:
            key = f"2D:{n}"
            sv, svc = sysm.surveys[key], coarse.surveys[key]
            tc = coarse.node_values(key)[None]  # (1, nc, Kc) -> reuse the 3D interpolator with one row
            t = interp_node_grid(tc, np.array([0.0]), coarse.lines[n].astype(float), svc["node_t"],
                                 np.array([0.0]), sysm.lines[n].astype(float), sv["node_t"])[0]
            x0[sv["offset"][0]: sv["offset"][0] + sv["K"] * sv["ncols"]] = t.ravel()
        hz0 = [coarse.x[coarse.hz_cols[h]] for h in sorted(sysm.hz_cols, key=sysm.hz_cols.get) if h in coarse.hz_cols]
        if len(hz0) == len(sysm.hz_cols):
            x0 = np.r_[x0, hz0]
    sysm.solve((lambda f, t: progress(0.25 + 0.7 * f, t)) if progress else None, cancel, x0)
    sysm.stats["coarse_start"] = x0 is not None
    return sysm


def compute_rgt_2d(project, names, params=None, faults=None, horizons=None, progress=None, cancel=None):
    params = params or RGTParams()
    folder = os.path.join(project.seismic_dir(), "rgt") if project.path else None
    sysm = solve_2d_multires(project, names, params, faults, horizons, folder, progress, cancel)
    return write_rgt_2d(project, sysm, folder), sysm
