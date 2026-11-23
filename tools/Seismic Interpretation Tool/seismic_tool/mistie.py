"""2D-3D and 2D-2D mistie analysis and network correction (numpy only).

Convention (same as the Mistie Tool): measure(a, b) returns the correction (shift_ms, phase_deg, gain) that
makes b match a:  b_corrected = gain * rotate(delay(b, shift_ms), phase_deg); delay moves the trace down
(positive = later).  Each 2D line L gets one correction c_L; a tie between a and b then measures c_b - c_a.
3D volumes are references (correction zero).

Measurement (Bishop & Nunns style, simultaneous): stacked cross-correlation over the overlapping traces,
time lag from the envelope peak (phase independent), constant phase from the analytic-signal inner product
at that lag, gain = rms ratio.  Network: weighted least squares (weights corr^2) with references fixed,
IRLS/Huber down-weighting of bad ties, phase solved as unit vectors (350 and 10 deg average to 0, not 180),
gain in log space.
"""
import numpy as np


def analytic(x, axis=-1):
    """Analytic signal via FFT (numpy replacement for scipy.signal.hilbert)."""
    x = np.asarray(x, float)
    n = x.shape[axis]
    X = np.fft.fft(x, axis=axis)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    shape = [1] * x.ndim
    shape[axis] = n
    return np.fft.ifft(X * h.reshape(shape), axis=axis)


def apply_correction(x, dt_ms, shift_ms, phase_deg, gain):
    """Correction of one trace or every row of a panel (fractional shift by FFT, zero padded).
    shift_ms may be a scalar or one value per row (flex corrections)."""
    x = np.atleast_2d(np.asarray(x, float))
    n = x.shape[1]
    pad = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfftfreq(pad)
    sh = np.asarray(shift_ms, float).reshape(-1, 1) if np.ndim(shift_ms) else shift_ms
    X = np.fft.rfft(x, pad, axis=1) * np.exp(-2j * np.pi * f[None, :] * sh / dt_ms)
    y = np.fft.irfft(X, pad, axis=1)[:, :n]
    if phase_deg:
        y = np.real(analytic(y, 1) * np.exp(1j * np.radians(phase_deg)))
    y = y * gain
    return y


def measure(A, B, dt_ms, window_ms=None, max_lag_ms=40.0, t0_ms=0.0):
    """Correction of B relative to A from one trace pair or stacks of traces (rows).  Returns dict
    (shift_ms, phase_deg, gain, corr, n)."""
    A = np.atleast_2d(np.asarray(A, float))
    B = np.atleast_2d(np.asarray(B, float))
    ns = A.shape[1]
    i0, i1 = 0, ns
    if window_ms is not None:
        i0 = max(0, int(round((window_ms[0] - t0_ms) / dt_ms)))
        i1 = min(ns, int(round((window_ms[1] - t0_ms) / dt_ms)))
    if i1 - i0 < 10:
        raise ValueError("correlation window too short")
    a = A[:, i0:i1] - A[:, i0:i1].mean(axis=1, keepdims=True)
    b = B[:, i0:i1] - B[:, i0:i1].mean(axis=1, keepdims=True)
    ok = (a.std(axis=1) > 0) & (b.std(axis=1) > 0)
    a, b = a[ok], b[ok]
    if not len(a):
        return {"shift_ms": 0.0, "phase_deg": 0.0, "gain": 1.0, "corr": 0.0, "n": 0}
    m = a.shape[1]
    L = max(1, int(max_lag_ms / dt_ms))
    nf = int(2 ** np.ceil(np.log2(2 * m)))
    an = a / np.linalg.norm(a, axis=1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    # stacked cross-correlation xc[k] = sum_t a[t + k] b[t]  (k > 0: a later than b -> delay b by k)
    xc = np.fft.irfft((np.fft.rfft(an, nf, axis=1) * np.conj(np.fft.rfft(bn, nf, axis=1))).sum(0), nf)
    xc = np.r_[xc[-L:], xc[:L + 1]]
    env = np.abs(analytic(xc))
    k = int(np.argmax(env))
    frac = 0.0
    if 0 < k < len(env) - 1:
        y0, y1, y2 = env[k - 1], env[k], env[k + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    lag = (k - L) + frac
    bs = apply_correction(B, dt_ms, lag * dt_ms, 0.0, 1.0)[ok][:, i0:i1]
    bs = bs - bs.mean(axis=1, keepdims=True)
    hb = analytic(bs, 1)
    # rotation phi maximising sum a * Re(hb e^{i phi}) -> phi = -angle(sum a * hb)... solve explicitly
    s = np.sum(a * np.real(hb)) - 1j * np.sum(a * np.imag(hb))
    phase = float(np.degrees(np.angle(s)))
    br = np.real(hb * np.exp(1j * np.radians(phase)))
    den = np.sum(br ** 2)
    gain = float(np.sqrt(np.sum(a ** 2) / den)) if den > 0 else 1.0
    corr = float(np.sum(a * br) / np.sqrt(np.sum(a * a) * den)) if den > 0 else 0.0
    return {"shift_ms": float(lag * dt_ms), "phase_deg": phase, "gain": gain, "corr": corr, "n": int(len(a))}


def wrap(p):
    return (np.asarray(p, float) + 180.0) % 360.0 - 180.0


# ====================================================================== geometry / extraction
def extract_3d_along(vol, x, y, times_out=None):
    """3D traces at 2D trace positions (bilinear between the 4 surrounding traces).
    Returns (traces (n, ns_out), inside mask (n,)).  times_out: resample to these times (ms)."""
    a, b = vol.ax, vol.ay
    Minv = np.linalg.inv(np.array([[a[1], a[2]], [b[1], b[2]]]))
    il = Minv[0, 0] * (x - a[0]) + Minv[0, 1] * (y - b[0])
    xl = Minv[1, 0] * (x - a[0]) + Minv[1, 1] * (y - b[0])
    fi, fj = (il - vol.il0) / vol.il_step, (xl - vol.xl0) / vol.xl_step
    inside = (fi >= 0) & (fi <= vol.nil - 1) & (fj >= 0) & (fj <= vol.nxl - 1)
    out = np.zeros((len(x), vol.ns), np.float32)
    if inside.any():
        i0 = np.clip(np.floor(fi[inside]).astype(int), 0, max(0, vol.nil - 2))
        j0 = np.clip(np.floor(fj[inside]).astype(int), 0, max(0, vol.nxl - 2))
        wi = np.clip(fi[inside] - i0, 0, 1)[:, None]
        wj = np.clip(fj[inside] - j0, 0, 1)[:, None]
        i1 = np.minimum(i0 + 1, vol.nil - 1)
        j1 = np.minimum(j0 + 1, vol.nxl - 1)
        g = lambda ii, jj: np.asarray(vol.ilxl[ii, jj], np.float32) * vol.inv  # noqa: E731
        out[inside] = (g(i0, j0) * (1 - wi) * (1 - wj) + g(i0, j1) * (1 - wi) * wj + g(i1, j0) * wi * (1 - wj) +
                       g(i1, j1) * wi * wj)
    if times_out is not None and not np.array_equal(times_out, vol.times):
        out = np.array([np.interp(times_out, vol.times, tr, left=0, right=0) for tr in out], np.float32)
    return out, inside


def resample(traces, t_in, t_out):
    if len(t_in) == len(t_out) and np.allclose(t_in, t_out):
        return np.asarray(traces, float)
    return np.array([np.interp(t_out, t_in, tr, left=0, right=0) for tr in traces])


# ====================================================================== ties
def tie_line_to_volume(project, vol_name, line_name, window_ms=None, max_lag_ms=40.0, profile_traces=0):
    """Correction of a 2D line relative to a 3D volume over their overlap. Uses the line's raw data
    (no current correction).  profile_traces > 0 adds an along-line profile of local measurements."""
    vol = project.volume(vol_name)
    line = project.line(line_name)
    if not line.has_coords:
        return None
    raw = line.raw_data() if hasattr(line, "raw_data") else line.data()
    tr3, inside = extract_3d_along(vol, line.x, line.y)
    if inside.sum() < 5:
        return None
    t = vol.times
    tr2 = resample(raw[inside], line.times, t)
    res = measure(tr3[inside], tr2, vol.dt * 1000.0, window_ms, max_lag_ms, t[0])
    res.update({"a": f"3D:{vol_name}", "b": line_name, "kind": "2D-3D", "traces": int(inside.sum())})
    if profile_traces:
        idx = np.nonzero(inside)[0]
        prof = []
        for c in range(0, len(idx), max(1, profile_traces // 2)):
            sel = idx[max(0, c - profile_traces // 2): c + profile_traces // 2 + 1]
            if len(sel) < 3:
                continue
            r = measure(tr3[sel], resample(raw[sel], line.times, t), vol.dt * 1000.0, window_ms, max_lag_ms, t[0])
            prof.append((float(line.cdp[sel[len(sel) // 2]]), r["shift_ms"], r["phase_deg"], r["gain"], r["corr"]))
        res["profile"] = prof
    return res


def tie_lines(project, a, b, ia, ib, half=5, window_ms=None, max_lag_ms=40.0):
    """Correction of line b relative to line a at their intersection (traces ia, ib +/- half)."""
    la, lb = project.line(a), project.line(b)
    ra = la.raw_data() if hasattr(la, "raw_data") else la.data()
    rb = lb.raw_data() if hasattr(lb, "raw_data") else lb.data()
    A = ra[max(0, ia - half): ia + half + 1]
    B = resample(rb[max(0, ib - half): ib + half + 1], lb.times, la.times)
    n = min(len(A), len(B))
    res = measure(A[:n].mean(0, keepdims=True), B[:n].mean(0, keepdims=True), la.dt * 1000.0, window_ms,
                  max_lag_ms, la.times[0])
    res.update({"a": a, "b": b, "kind": "2D-2D", "traces": int(n), "at": (int(ia), int(ib))})
    return res


def all_ties(project, window_ms=None, max_lag_ms=40.0, lines=None, volumes=None):
    from .horizons import line_intersections
    names = lines or [l["name"] for l in project.lines]
    vols = volumes if volumes is not None else [v["name"] for v in project.volumes]
    ties = []
    for v in vols:
        for n in names:
            try:
                r = tie_line_to_volume(project, v, n, window_ms, max_lag_ms)
            except Exception:
                r = None
            if r is not None:
                ties.append(r)
    geo = {}
    for n in names:
        ln = project.line(n)
        if ln.has_coords:
            geo[n] = (ln.x, ln.y)
    for a, ia, b, ib in line_intersections(geo):
        try:
            ties.append(tie_lines(project, a, b, ia, ib, 5, window_ms, max_lag_ms))
        except Exception:
            continue
    return ties


# ====================================================================== network
def solve_network(lines, ties, references=(), robust=True, min_corr=0.3, huber_ms=4.0, iters=10, weights=None):
    """Per-line corrections from ties.  references: node names fixed at zero (3D volumes are always fixed).
    Returns (solution {line: dict(shift_ms, phase_deg, gain, n_ties)}, ties with residuals and final weights)."""
    nodes = list(lines)
    fixed = set(references) | {t["a"] for t in ties if t["a"].startswith("3D:")} | \
        {t["b"] for t in ties if t["b"].startswith("3D:")}
    unk = [n for n in nodes if n not in fixed]
    idx = {n: i for i, n in enumerate(unk)}
    use = [t for t in ties if t["corr"] >= min_corr and (t["a"] in idx or t["b"] in idx)]
    n = len(unk)
    if n == 0:
        return {}, ties
    w0 = np.array([max(t["corr"], 0.0) ** 2 for t in use]) if weights is None else np.asarray(weights, float)

    def design():
        A = np.zeros((len(use), n))
        for r, t in enumerate(use):
            if t["b"] in idx:
                A[r, idx[t["b"]]] += 1
            if t["a"] in idx:
                A[r, idx[t["a"]]] -= 1
        return A

    A = design()
    damp = 1e-6
    w = w0.copy()

    def lsq(d, w):
        M = np.vstack([A * np.sqrt(w)[:, None], damp * np.eye(n)])
        rhs = np.r_[d * np.sqrt(w), np.zeros(n)]
        return np.linalg.lstsq(M, rhs, rcond=None)[0]

    d_shift = np.array([t["shift_ms"] for t in use])
    for _ in range(iters if robust else 1):
        s = lsq(d_shift, w)
        r = d_shift - A @ s
        if not robust:
            break
        a = np.abs(r)
        w = w0 * np.where(a <= huber_ms, 1.0, huber_ms / np.maximum(a, 1e-9))
    g = np.exp(lsq(np.log(np.maximum([t["gain"] for t in use], 1e-6)), w))
    # phase as unit vectors: z_b - e^{i phi_ab} z_a = 0, fixed nodes z = 1
    rows, rhs, ww = [], [], []
    for t, wt in zip(use, w):
        e = np.exp(1j * np.radians(t["phase_deg"]))
        row = np.zeros(n, complex)
        rr = 0j
        if t["b"] in idx:
            row[idx[t["b"]]] += 1
        else:
            rr -= 1
        if t["a"] in idx:
            row[idx[t["a"]]] -= e
        else:
            rr += e
        rows.append(row)
        rhs.append(rr)
        ww.append(wt)
    M = np.array(rows) * np.sqrt(ww)[:, None]
    z = np.linalg.lstsq(np.vstack([M, damp * np.eye(n)]), np.r_[np.array(rhs) * np.sqrt(ww), np.zeros(n)],
                        rcond=None)[0]
    ph = np.degrees(np.angle(np.where(np.abs(z) > 0, z, 1)))
    sol = {}
    for k, nme in enumerate(unk):
        nt = sum(1 for t in use if nme in (t["a"], t["b"]))
        sol[nme] = {"shift_ms": float(s[k]), "phase_deg": float(wrap(ph[k])), "gain": float(g[k]), "n_ties": nt}
    for nme in fixed:
        sol.setdefault(nme, {"shift_ms": 0.0, "phase_deg": 0.0, "gain": 1.0, "n_ties": 0, "fixed": True})
    for t in ties:
        cb, ca = sol.get(t["b"]), sol.get(t["a"])
        if cb is None or ca is None:
            continue
        t["res_shift_ms"] = t["shift_ms"] - (cb["shift_ms"] - ca["shift_ms"])
        t["res_phase_deg"] = float(wrap(t["phase_deg"] - (cb["phase_deg"] - ca["phase_deg"])))
        t["used"] = t in use
    for t, wt in zip(use, w):
        t["weight"] = float(wt)
    return sol, ties


def loop_closures(ties, sol=None):
    """Raw shift closure around every triangle of ties (lines and volumes); small = consistent network."""
    import itertools
    by = {}
    for t in ties:
        by[(t["a"], t["b"])] = t["shift_ms"]
        by[(t["b"], t["a"])] = -t["shift_ms"]
    nodes = sorted({t["a"] for t in ties} | {t["b"] for t in ties})
    out = []
    for p, q, r in itertools.combinations(nodes, 3):
        if (p, q) in by and (q, r) in by and (r, p) in by:
            out.append(((p, q, r), by[(p, q)] + by[(q, r)] + by[(r, p)]))
    return out


# ====================================================================== applying corrections
def apply_solution(project, sol):
    """Store per-line corrections in the project; shift existing 2D horizon picks by the change in shift so
    interpretation stays on the same reflectors; reload lines (data() then returns corrected traces)."""
    old = project.display.get("mistie", {}) or {}
    new = {}
    for name, c in sol.items():
        if name.startswith("3D:"):
            continue
        new[name] = {"shift_ms": c["shift_ms"], "phase_deg": c["phase_deg"], "gain": c["gain"]}
        if c.get("flex"):
            new[name]["flex"] = c["flex"]
        if c.get("tv"):
            new[name]["tv"] = c["tv"]

    def shifts(entry, line, t=None):
        if not entry:
            return 0.0
        base = flex_shift_per_trace(line, entry["flex"]) if entry.get("flex") else entry.get("shift_ms", 0.0)
        if entry.get("tv") and t is not None:  # time-variant: the shift at each pick's time
            base = base + np.interp(t, entry["tv"]["t"], entry["tv"]["shift"])
        return base

    for name in set(old) | set(new):
        try:
            line = project.line(name)
        except Exception:
            continue
        for h in project.horizons.items:
            a = h.data.get(f"2D:{name}")
            if a is None:
                continue
            # picks are in the old corrected time; undo the old shift at the uncorrected time, apply the new
            t_raw = a - np.asarray(shifts(old.get(name), line, a), float)
            if old.get(name, {}).get("tv"):  # iterate once more for the time-variant part
                t_raw = a - np.asarray(shifts(old.get(name), line, t_raw), float)
            d = np.asarray(shifts(new.get(name), line, t_raw), float) + t_raw - a
            if np.any(np.abs(d[np.isfinite(d)]) > 1e-9):
                h.data[f"2D:{name}"] = (a + d).astype(np.float32)
                project.horizons.dirty = True
    project.display["mistie"] = new
    project._line_cache.clear()
    return new


# ====================================================================== flex mode (along-line shifts)
def solve_flex(project, lines, ties, constant, lam=1.0, knot_traces=40, window_ms=None, max_lag_ms=40.0,
               profile_traces=30, volumes=None, prior=0.05):
    """Along-line flex: shift(line, cdp) = piecewise-linear function with knots every knot_traces, solved as one
    regularised least-squares system:
        2D-3D local measurements (along-line profiles)   s_L(k) = m          weight corr^2
        2D-2D ties                                         s_b(kb) - s_a(ka) = m_ab
        smoothness                                         lam * (s_L(i+1) - s_L(i)) = 0   (high lam -> constant)
        prior                                              prior * (s_L(i) - constant_L) = 0
    Returns {line: {"cdp": knot cdps, "shift": knot shifts, "max_flex": max |shift - constant|}}."""
    vols = volumes if volumes is not None else [v["name"] for v in project.volumes]
    knots, off, n = {}, {}, 0
    for L in lines:
        ln = project.line(L)
        k = np.unique(np.r_[np.arange(0, len(ln.cdp), knot_traces), len(ln.cdp) - 1])
        knots[L] = k
        off[L] = n
        n += len(k)
    rows, rhs, w = [], [], []

    def interp_row(L, trace):
        k = knots[L]
        j = int(np.clip(np.searchsorted(k, trace, "right"), 1, len(k) - 1))
        f = (trace - k[j - 1]) / max(1, k[j] - k[j - 1])
        return [(off[L] + j - 1, 1 - f), (off[L] + j, f)]

    for v in vols:
        for L in lines:
            try:
                r = tie_line_to_volume(project, v, L, window_ms, max_lag_ms, profile_traces)
            except Exception:
                r = None
            if not r or not r.get("profile"):
                continue
            ln = project.line(L)
            for cdp, sh, ph, gn, cc in r["profile"]:
                if cc < 0.3:
                    continue
                trace = int(np.argmin(np.abs(ln.cdp - cdp)))
                rows.append(interp_row(L, trace))
                rhs.append(sh)
                w.append(cc * cc)
    for t in ties:
        if t.get("kind") != "2D-2D" or t["corr"] < 0.3 or t["a"] not in knots or t["b"] not in knots:
            continue
        ia, ib = t["at"]
        row = [(c, -f) for c, f in interp_row(t["a"], ia)] + interp_row(t["b"], ib)
        rows.append(row)
        rhs.append(t["shift_ms"])
        w.append(t.get("weight", t["corr"] ** 2))
    for L in lines:
        k = knots[L]
        for i in range(len(k) - 1):
            rows.append([(off[L] + i + 1, 1.0), (off[L] + i, -1.0)])
            rhs.append(0.0)
            w.append(lam * lam * (knot_traces / max(1, k[i + 1] - k[i])))
        c = constant.get(L, {}).get("shift_ms", 0.0)
        for i in range(len(k)):
            rows.append([(off[L] + i, 1.0)])
            rhs.append(c)
            w.append(prior * prior)
    A = np.zeros((len(rows), n))
    for r, row in enumerate(rows):
        for c, f in row:
            A[r, c] += f
    sw = np.sqrt(np.asarray(w))
    x = np.linalg.lstsq(A * sw[:, None], np.asarray(rhs) * sw, rcond=None)[0]
    out = {}
    for L in lines:
        ln = project.line(L)
        sh = x[off[L]:off[L] + len(knots[L])]
        c = constant.get(L, {}).get("shift_ms", 0.0)
        out[L] = {"cdp": ln.cdp[knots[L]].tolist(), "shift": sh.tolist(), "max_flex": float(np.max(np.abs(sh - c)))}
    return out


def flex_shift_per_trace(line, flex):
    return np.interp(line.cdp, flex["cdp"], flex["shift"])


# ====================================================================== horizon-based misties
def horizon_ties(project, h, volumes=None):
    """Ties from an interpreted horizon instead of waveforms: 2D-3D = median(t_3D - t_2D) along the overlap,
    2D-2D = t_a - t_b at intersections (correction of b relative to a).  corr = consistency (1 - spread/10 ms)."""
    from .horizons import line_intersections
    from .maps import MapGrid
    ties = []
    vols = volumes if volumes is not None else [v["name"] for v in project.volumes]
    lines = [k[3:] for k in h.data if k.startswith("2D:")]
    for v in vols:
        g3 = h.get(f"3D:{v}")
        if g3 is None:
            continue
        vol = project.volume(v)
        grid = MapGrid(g3, "ilxl", "ms", "", vol)
        for L in lines:
            ln = project.line(L)
            if not ln.has_coords:
                continue
            t2 = h.get(f"2D:{L}")
            t3 = grid.sample(ln.x, ln.y)
            ok = np.isfinite(t2) & np.isfinite(t3)
            if ok.sum() < 3:
                continue
            d = t3[ok] - t2[ok]
            med = float(np.median(d))
            spread = float(np.median(np.abs(d - med)))
            k = np.nonzero(ok)[0]
            mid = k[len(k) // 2]
            ties.append({"a": f"3D:{v}", "b": L, "kind": "horizon 2D-3D", "shift_ms": med, "phase_deg": 0.0,
                         "gain": 1.0, "corr": float(np.clip(1 - spread / 10.0, 0.3, 1.0)), "n": int(ok.sum()),
                         "xy": (float(ln.x[mid]), float(ln.y[mid]))})
    geo = {L: (project.line(L).x, project.line(L).y) for L in lines if project.line(L).has_coords}
    for a, ia, b, ib in line_intersections(geo):
        ta, tb = h.get(f"2D:{a}")[ia], h.get(f"2D:{b}")[ib]
        if np.isfinite(ta) and np.isfinite(tb):
            la = project.line(a)
            ties.append({"a": a, "b": b, "kind": "horizon 2D-2D", "shift_ms": float(ta - tb), "phase_deg": 0.0,
                         "gain": 1.0, "corr": 1.0, "n": 1, "at": (int(ia), int(ib)),
                         "xy": (float(la.x[ia]), float(la.y[ia]))})
    return ties


# ====================================================================== groups / fixed lines
def solve_grouped(lines, ties, groups=None, fixed_lines=(), **kw):
    """Network solve where lines of one group (survey / vintage) share one correction and fixed_lines (key /
    reference lines) are held at zero.  Implemented by renaming the tie ends to their group."""
    groups = groups or {}
    gname = lambda n: n if n.startswith("3D:") else ("FIXED:" + n if n in fixed_lines else groups.get(n, n))  # noqa
    renamed = []
    for t in ties:
        a, b = gname(t["a"]), gname(t["b"])
        if a == b:
            continue
        r = dict(t)
        r["a"], r["b"] = a, b
        r["_orig"] = t
        renamed.append(r)
    nodes = sorted({gname(L) for L in lines})
    refs = [n for n in nodes if n.startswith("FIXED:")]
    sol_g, rt = solve_network(nodes, renamed, references=refs, **kw)
    for r in rt:
        o = r["_orig"]
        for k in ("res_shift_ms", "res_phase_deg", "weight", "used"):
            if k in r:
                o[k] = r[k]
    zero = {"shift_ms": 0.0, "phase_deg": 0.0, "gain": 1.0, "n_ties": 0, "fixed": True}
    sol = {L: dict(sol_g.get(gname(L), zero), group=gname(L)) for L in lines}
    return sol, ties


# ====================================================================== time-variant corrections
def measure_windows(A, B, dt_ms, t0_ms, windows, max_lag_ms=40.0):
    """Shift (and corr) of B relative to A in several time windows: [(t_centre, shift_ms, corr)].
    t_centre is the reflection-energy centroid of A in the window (where the measurement really applies)."""
    out = []
    A = np.atleast_2d(np.asarray(A, float))
    tt = t0_ms + dt_ms * np.arange(A.shape[1])
    env2 = (np.abs(analytic(A, 1)) ** 2).sum(0)
    for w0, w1 in windows:
        try:
            r = measure(A, B, dt_ms, (w0, w1), max_lag_ms, t0_ms)
        except ValueError:
            continue
        m = (tt >= w0) & (tt <= w1)
        tc = float((tt[m] * env2[m]).sum() / env2[m].sum()) if env2[m].sum() > 0 else 0.5 * (w0 + w1)
        out.append((tc, r["shift_ms"], r["corr"]))
    return out


def time_variant_line(project, vol_name, line_name, t0, t1, win_ms=300.0, step_ms=150.0, max_lag_ms=40.0,
                      min_corr=0.5):
    """Time-variant shift of a 2D line vs a 3D: smooth (quadratic in TWT) shift function from windowed
    stacked correlations, refined iteratively; returned as knots (t, shift) at the windows' energy centroids."""
    vol = project.volume(vol_name)
    line = project.line(line_name)
    raw = line.raw_data()
    tr3, inside = extract_3d_along(vol, line.x, line.y)
    if inside.sum() < 5:
        return None
    t = vol.times
    tr2 = resample(raw[inside], line.times, t)
    wins = [(a, a + win_ms) for a in np.arange(t0, t1 - win_ms + 1e-6, step_ms)]
    A = tr3[inside]
    tc = np.array([0.5 * (a + b) for a, b in wins])
    shift = np.zeros(len(wins))
    corr = np.zeros(len(wins))
    for _ in range(4):  # iterate: correct, re-measure the residual (removes the window-edge bias)
        Bc = apply_time_variant(tr2, t, {"t": tc, "shift": shift}) if np.any(shift) else tr2
        res = measure_windows(A, Bc, vol.dt * 1000.0, t[0], wins, max_lag_ms)
        if len(res) != len(wins):
            break
        d = np.array([r[1] for r in res])
        corr = np.array([r[2] for r in res])
        if not np.any(shift):
            tc = np.array([r[0] for r in res])  # energy centroids (first pass)
        shift = shift + d
        # windowed measurements are averages over overlapping windows: keep the model smooth (low-order
        # polynomial in time, weights corr^2) or the iteration can settle on an oscillating solution
        deg = 2 if len(tc) >= 5 else 1 if len(tc) >= 2 else 0
        if deg:
            cw = np.clip(corr, 0, 1) ** 2 + 1e-6
            coef = np.polyfit(tc, shift, deg, w=np.sqrt(cw))
            shift = np.polyval(coef, tc)
        if np.max(np.abs(d)) < 0.05:
            break
    keep = corr >= min_corr
    if not keep.any():
        return None
    return {"t": tc[keep].tolist(), "shift": shift[keep].tolist(), "corr": corr[keep].tolist()}


def apply_time_variant(x, times, tv):
    """Delay every trace by s(t) (piecewise linear in time, constant beyond the knots): y(t) = x(t - s(t))."""
    x = np.atleast_2d(np.asarray(x, float))
    s = np.interp(times, tv["t"], tv["shift"])
    src = (times - s - times[0]) / (times[1] - times[0])
    k = np.clip(np.floor(src).astype(int), 0, len(times) - 2)
    w = np.clip(src - k, 0, 1)
    y = x[:, k] * (1 - w) + x[:, k + 1] * w
    y[:, (src < 0) | (src > len(times) - 1)] = 0.0
    return np.ascontiguousarray(y)


# ====================================================================== overlapping / duplicate lines
def overlap_ties(project, tol_m=15.0, min_traces=20, window_ms=None, max_lag_ms=40.0):
    """Collinear overlaps (reprocessed vintages, overlapping segments): line pairs with >= min_traces traces
    within tol_m of each other are tied along the whole overlap (stacked measurement)."""
    names = [l["name"] for l in project.lines]
    lines = {n: project.line(n) for n in names}
    out = []
    for i, a in enumerate(names):
        la = lines[a]
        if not la.has_coords:
            continue
        for b in names[i + 1:]:
            lb = lines[b]
            if not lb.has_coords:
                continue
            if la.x.max() < lb.x.min() - tol_m or lb.x.max() < la.x.min() - tol_m or \
                    la.y.max() < lb.y.min() - tol_m or lb.y.max() < la.y.min() - tol_m:
                continue
            pa, pb = [], []
            step = max(1, len(la.x) // 2000)
            for k in range(0, len(la.x), step):
                d = np.hypot(lb.x - la.x[k], lb.y - la.y[k])
                j = int(np.argmin(d))
                if d[j] <= tol_m:
                    pa.append(k)
                    pb.append(j)
            if len(pa) * step < min_traces:
                continue
            A = la.raw_data()[pa]
            B = resample(lb.raw_data()[pb], lb.times, la.times)
            r = measure(A, B, la.dt * 1000.0, window_ms, max_lag_ms, la.times[0])
            mid = pa[len(pa) // 2]
            r.update({"a": a, "b": b, "kind": "overlap", "traces": len(pa) * step,
                      "xy": (float(la.x[mid]), float(la.y[mid]))})
            out.append(r)
    return out


def set_preferred(project, keep, hide):
    """Mark `hide` lines as duplicates of `keep` (hidden on the basemap; still tied)."""
    d = project.display.setdefault("hidden_lines", [])
    for h in hide:
        if h not in d:
            d.append(h)
    if keep in d:
        d.remove(keep)
    project.display.setdefault("preferred_lines", {})[keep] = list(hide)


# ====================================================================== exports
def tie_positions(project, ties):
    """Add 'xy' (map position) to ties that lack it (for the residual map)."""
    for t in ties:
        if "xy" in t:
            continue
        try:
            if "at" in t:
                la = project.line(t["a"])
                t["xy"] = (float(la.x[t["at"][0]]), float(la.y[t["at"][0]])) if la.has_coords else None
            else:
                ln = project.line(t["b"])
                vol = project.volume(t["a"][3:])
                _, inside = extract_3d_along(vol, ln.x[::5], ln.y[::5])
                k = np.nonzero(inside)[0]
                t["xy"] = (float(ln.x[::5][k[len(k) // 2]]), float(ln.y[::5][k[len(k) // 2]])) if len(k) else None
        except Exception:
            t["xy"] = None
    return ties


def write_tie_table(ties, sol, path):
    """CSV of ties (measured, residual, weight) and per-line solution - for QC or the Mistie Tool."""
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# ties: correction of B relative to A"])
        w.writerow(["A", "B", "kind", "shift_ms", "phase_deg", "gain", "corr", "residual_ms", "residual_deg", "weight"])
        for t in ties:
            w.writerow([t["a"], t["b"], t["kind"], f"{t['shift_ms']:.2f}", f"{t['phase_deg']:.1f}", f"{t['gain']:.3f}",
                        f"{t['corr']:.3f}", f"{t.get('res_shift_ms', float('nan')):.2f}",
                        f"{t.get('res_phase_deg', float('nan')):.1f}", f"{t.get('weight', 0):.3f}"])
        w.writerow([])
        w.writerow(["# corrections per line"])
        w.writerow(["line", "shift_ms", "phase_deg", "gain", "n_ties", "group"])
        for L, c in sol.items():
            w.writerow([L, f"{c['shift_ms']:.2f}", f"{c['phase_deg']:.1f}", f"{c['gain']:.3f}", c.get("n_ties", 0),
                        c.get("group", L)])
    return path


def export_corrected_segy(line, out_path):
    """Write the line's corrected traces (data()) as a new IEEE-float SEG-Y, headers copied from the input."""
    f = line.f
    data = np.asarray(line.data(), np.float32)
    mm = f.mm
    with open(out_path, "wb") as fh:
        hdr = bytearray(mm[:3600].tobytes())
        bo = "big" if f.endian == ">" else "little"
        hdr[3224:3226] = (5).to_bytes(2, bo)  # format code 5 = IEEE float
        fh.write(bytes(hdr))
        if f.data_offset > 3600:
            fh.write(mm[3600:f.data_offset].tobytes())
        dt = ">f4" if f.endian == ">" else "<f4"
        for k in range(f.ntraces):
            fh.write(f._tr[k, :240].tobytes())
            fh.write(data[k].astype(dt).tobytes())
    return out_path
