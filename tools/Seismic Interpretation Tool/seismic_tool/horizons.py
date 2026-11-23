"""Horizons: storage, snapping, autotracking (3D grids and 2D lines, with 2D tie propagation).

A Horizon holds picks for any number of surveys, keyed "3D:<volume>" -> float32 (n_inline, n_xline) TWT ms
or "2D:<line>" -> float32 (n_traces,) TWT ms.  NaN = unpicked.

Files: <project>_seismic/horizons/<safe name>/<safe key>.npy, list + colours in the project JSON.

Tracking (`track_grid`) is a vectorised wavefront flood fill: every pass looks at all unpicked neighbours of
the current front, cross-correlates a window of the parent trace with the child trace over lags
+/- max jump, accepts children whose best correlation passes the threshold, then snaps them to the event
(peak / trough / zero crossing) with sub-sample interpolation.  The threshold is relaxed in steps
(0.95 -> min corr) so the most reliable picks grow first, like a priority-queue tracker, at numpy speed.
"""
import json
import os

import numpy as np

def _next_seq():
    from .faults import next_seq
    return next_seq()


SNAP_MODES = ["Peak", "Trough", "Zero + to -", "Zero - to +", "None"]
DEFAULT_COLORS = [(255, 200, 0), (0, 200, 255), (60, 220, 60), (255, 80, 200), (255, 120, 0), (150, 100, 255),
                  (0, 140, 70), (200, 0, 0)]


def safe_name(s):
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in str(s)).strip() or "_"


def survey_key(kind, name):
    return f"{kind}:{name}"


# ====================================================================== snapping
def snap_times(traces, t_idx, mode="Peak", half=3):
    """Move each predicted (fractional) sample index to the nearest event of the given type.

    traces (n, ns) float; t_idx (n,) float sample indices; half = search half-window in samples.
    Returns (new_idx float64 (n,), ok bool (n,)).  Peak/trough use parabolic interpolation, zero crossings
    linear interpolation.  ok=False where no such event is inside the window (idx returned unchanged)."""
    traces = np.asarray(traces, np.float64)
    if traces.ndim == 1:
        traces = traces[None]
    t_idx = np.atleast_1d(np.asarray(t_idx, np.float64))
    n, ns = traces.shape
    if mode == "None" or n == 0:
        return t_idx.copy(), np.isfinite(t_idx)
    c = np.round(t_idx).astype(np.int64)
    off = np.arange(-half - 1, half + 2)  # one extra each side for the local-extremum test
    idx = np.clip(c[:, None] + off[None], 0, ns - 1)
    w = traces[np.arange(n)[:, None], idx]
    if mode == "Trough":
        w = -w
    if mode in ("Peak", "Trough"):
        mid = w[:, 1:-1]
        is_ext = (mid >= w[:, :-2]) & (mid > w[:, 2:])
        pos = off[1:-1].astype(float)
        dist = np.abs(c[:, None] + pos[None] - t_idx[:, None])
        dist = np.where(is_ext, dist, np.inf)
        k = np.argmin(dist, axis=1)
        ok = np.isfinite(dist[np.arange(n), k])
        y0, y1, y2 = w[np.arange(n), k], w[np.arange(n), k + 1], w[np.arange(n), k + 2]
        den = y0 - 2 * y1 + y2
        delta = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
        delta = np.clip(delta, -0.5, 0.5)
        new = c + off[1:-1][k] + delta
    else:
        a, b = w[:, :-1], w[:, 1:]
        if mode == "Zero + to -":
            is_zc = (a > 0) & (b <= 0)
        else:
            is_zc = (a < 0) & (b >= 0)
        frac = np.where(is_zc, a / np.where(a - b == 0, 1, a - b), 0.0)
        pos = c[:, None] + off[:-1][None] + frac
        dist = np.where(is_zc, np.abs(pos - t_idx[:, None]), np.inf)
        k = np.argmin(dist, axis=1)
        ok = np.isfinite(dist[np.arange(n), k])
        new = pos[np.arange(n), k]
    ok &= (new >= 0) & (new <= ns - 1)
    return np.where(ok, new, t_idx), ok


# ====================================================================== tracker
def _windows(traces, centre, half, lags):
    """traces (n, ns); centre (n,) int; returns (n, len(lags), 2*half+1) windows (clipped at the ends)."""
    n, ns = traces.shape
    off = np.arange(-half, half + 1)
    idx = centre[:, None, None] + np.asarray(lags)[None, :, None] + off[None, None, :]
    idx = np.clip(idx, 0, ns - 1)
    return traces[np.arange(n)[:, None, None], idx]


def correlate_pairs(parent_tr, child_tr, t_parent, half, maxlag):
    """Normalised cross-correlation of a parent window (around t_parent, fractional index) with the child
    trace at integer lags -maxlag..maxlag.  Returns (best_lag_frac (n,), best_corr (n,)).
    best_lag_frac is the parabolic-refined lag (samples) to add to t_parent."""
    n = len(t_parent)
    c = np.round(t_parent).astype(np.int64)
    search = maxlag + 1  # one extra lag each side so an optimum beyond max jump is recognised and rejected
    lags = np.arange(-search, search + 1)
    p = _windows(parent_tr, c, half, [0])[:, 0, :]
    q = _windows(child_tr, c, half, lags)
    p = p - p.mean(axis=1, keepdims=True)
    q = q - q.mean(axis=2, keepdims=True)
    num = np.einsum("nk,nlk->nl", p, q)
    den = np.sqrt((p * p).sum(axis=1))[:, None] * np.sqrt((q * q).sum(axis=2))
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.where(den > 1e-12, num / den, -1.0)
    k = np.argmax(cc, axis=1)
    best = cc[np.arange(n), k]
    # parabolic refinement of the correlation peak
    km, kp = np.clip(k - 1, 0, len(lags) - 1), np.clip(k + 1, 0, len(lags) - 1)
    y0, y1, y2 = cc[np.arange(n), km], best, cc[np.arange(n), kp]
    den2 = y0 - 2 * y1 + y2
    inner = (k > 0) & (k < len(lags) - 1) & (np.abs(den2) > 1e-12)
    delta = np.where(inner, 0.5 * (y0 - y2) / np.where(den2 == 0, 1, den2), 0.0)
    lag = lags[k] + np.clip(delta, -0.5, 0.5)  # waveform shift = child time - parent time (fraction kept)
    # optimum beyond max jump (incl. the extra edge lag): the event moved too far - reject
    best = np.where(np.abs(lags[k]) > maxlag, -1.0, best)
    return lag, best


class TrackParams:
    def __init__(self, mode="Peak", window_ms=20.0, max_jump_ms=10.0, min_corr=0.6, snap_ms=None,
                 schedule=(0.95, 0.9, 0.8, 0.7), max_gap=3, smooth=1):
        self.mode, self.window_ms, self.max_jump_ms, self.min_corr = mode, window_ms, max_jump_ms, min_corr
        self.max_gap = max_gap  # bridge up to this many traces when growth stalls (1 = off)
        self.smooth = smooth  # lateral mix (traces) applied to the data the tracker sees (1 = off)
        self.snap_ms = snap_ms  # search half-window for the snap after correlation (default ~2 samples)
        self.schedule = schedule

    def samples(self, dt_ms):
        half = max(1, int(round(self.window_ms / dt_ms)))
        maxlag = max(1, int(np.ceil(self.max_jump_ms / dt_ms - 1e-9)))
        snap = max(2, int(round((self.snap_ms or 2 * dt_ms) / dt_ms)))
        return half, maxlag, snap

    def thresholds(self):
        th = [t for t in self.schedule if t > self.min_corr] + [self.min_corr]
        return sorted(set(th), reverse=True)


def track_grid(get_traces, grid_idx, seeds, params, dt_ms, ns, allowed=None, blocked=None, progress=None,
               max_nodes=None):
    """Autotrack on a 2D grid of traces (3D: (nil, nxl); 2D line: (1, ntr)).

    get_traces(ii, jj, a, b) -> float array (n, b - a) of samples a..b-1 for traces (ii[k], jj[k]).
    grid_idx: float (ni, nj) fractional sample index of picks, NaN unpicked - updated IN PLACE.
    seeds: bool (ni, nj) - nodes to grow from (must be picked).
    allowed: optional bool (ni, nj) mask of nodes that may be picked (e.g. one inline only).
    blocked: optional fn(pi, pj, ci, cj) -> bool array; True = do not link (faults, item 2).
    When growth stalls at the lowest threshold, the tracker tries to bridge up to params.max_gap traces
    (noisy / dead traces); the skipped traces are filled by interpolation + snap.
    Returns number of nodes added."""
    ni, nj = grid_idx.shape
    half, maxlag, snap = params.samples(dt_ms)
    picked = np.isfinite(grid_idx)
    if allowed is None:
        allowed = np.ones((ni, nj), bool)
    state = {"added": 0}
    unit = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def grow(front, steps, th):
        """One wavefront step from `front` using offsets `steps`; returns new front (bool) or None."""
        fi, fj = np.nonzero(front)
        P, C, S = [], [], []
        for di, dj in steps:
            ci, cj = fi + di, fj + dj
            ok = (ci >= 0) & (ci < ni) & (cj >= 0) & (cj < nj)
            pi_, pj_, ci, cj = fi[ok], fj[ok], ci[ok], cj[ok]
            ok = ~picked[ci, cj] & allowed[ci, cj]
            pi_, pj_, ci, cj = pi_[ok], pj_[ok], ci[ok], cj[ok]
            if blocked is not None and len(ci):
                ok = ~np.asarray(blocked(pi_, pj_, ci, cj), bool)
                pi_, pj_, ci, cj = pi_[ok], pj_[ok], ci[ok], cj[ok]
            P.append(np.column_stack([pi_, pj_]))
            C.append(np.column_stack([ci, cj]))
            S.append(np.full(len(ci), max(abs(di), abs(dj))))
        P, C, S = np.concatenate(P), np.concatenate(C), np.concatenate(S)
        if not len(C):
            return None
        dist = int(S.max())
        tp = grid_idx[P[:, 0], P[:, 1]]
        ml = maxlag * dist
        pad = half + ml + snap + 4
        a = int(max(0, np.floor(tp.min()) - pad))
        b = int(min(ns, np.ceil(tp.max()) + pad + 1))
        allk = np.concatenate([P, C])  # read each distinct trace once
        uk, inv = np.unique(allk[:, 0] * nj + allk[:, 1], return_inverse=True)
        tr = np.asarray(get_traces(uk // nj, uk % nj, a, b), np.float64)
        ptr, ctr = tr[inv[:len(P)]], tr[inv[len(P):]]
        lag, cc = correlate_pairs(ptr, ctr, tp - a, half, ml)
        tc = tp - a + lag
        if params.mode != "None":
            # snap to the event; where the event fades (no extremum nearby) keep the correlation estimate
            tc, _ok = snap_times(ctr, tc, params.mode, snap)
        good = cc >= th
        if not good.any():
            return None
        P, C, tc, cc = P[good], C[good], tc[good] + a, cc[good]
        order = np.argsort(-cc)  # best correlation wins when a child has several parents
        P, C, tc = P[order], C[order], tc[order]
        _, first = np.unique(C[:, 0] * nj + C[:, 1], return_index=True)
        P, C, tc = P[first], C[first], tc[first]
        grid_idx[C[:, 0], C[:, 1]] = tc
        picked[C[:, 0], C[:, 1]] = True
        state["added"] += len(C)
        if dist > 1:  # fill the bridged traces: interpolate parent -> child, then snap
            fill_i, fill_j, fill_t = [], [], []
            for k in range(1, dist):
                w = k / dist
                ii = np.round(P[:, 0] + (C[:, 0] - P[:, 0]) * w).astype(int)
                jj = np.round(P[:, 1] + (C[:, 1] - P[:, 1]) * w).astype(int)
                ok = ~picked[ii, jj] & allowed[ii, jj]
                fill_i.append(ii[ok])
                fill_j.append(jj[ok])
                fill_t.append((grid_idx[P[:, 0], P[:, 1]] * (1 - w) + tc * w)[ok])
            fi_, fj_, ft = np.concatenate(fill_i), np.concatenate(fill_j), np.concatenate(fill_t)
            if len(fi_):
                a2 = int(max(0, np.floor(ft.min()) - snap - 3))
                b2 = int(min(ns, np.ceil(ft.max()) + snap + 4))
                if params.mode != "None":
                    ft2, _ = snap_times(np.asarray(get_traces(fi_, fj_, a2, b2), np.float64), ft - a2,
                                        params.mode, max(1, snap // 2))
                    ft = ft2 + a2
                grid_idx[fi_, fj_] = ft
                picked[fi_, fj_] = True
                state["added"] += len(fi_)
        new = np.zeros_like(picked)
        new[C[:, 0], C[:, 1]] = True
        if progress:
            progress(state["added"])
        return new

    def boundary():
        open_n = np.zeros_like(picked)
        open_n[:-1] |= ~picked[1:] & allowed[1:]
        open_n[1:] |= ~picked[:-1] & allowed[:-1]
        open_n[:, :-1] |= ~picked[:, 1:] & allowed[:, 1:]
        open_n[:, 1:] |= ~picked[:, :-1] & allowed[:, :-1]
        return picked & open_n

    ths = params.thresholds()
    front = seeds & picked
    for n_th, th in enumerate(ths):
        if n_th:  # relaxed pass: restart from every picked node with an open neighbour
            front = boundary()
        while True:
            while front is not None and front.any():
                front = grow(front, unit, th)
                if max_nodes and state["added"] >= max_nodes:
                    return state["added"]
            if n_th < len(ths) - 1 or params.max_gap < 2:
                break
            # lowest threshold and stalled: try to bridge bad traces
            front = None
            bnd = boundary()
            for g in range(2, params.max_gap + 1):
                front = grow(bnd, [(g, 0), (-g, 0), (0, g), (0, -g)], th)
                if front is not None:
                    break
            if front is None:
                break
    return state["added"]


# ====================================================================== survey adapters
def volume_traces(vol, smooth=1):
    """get_traces for a Volume3D (reads only the needed time window off the memmap).
    smooth > 1: each trace is the mean of a smooth x smooth block of neighbours (lateral mix for noisy data)."""
    r = max(0, (int(smooth) - 1) // 2)

    def get(ii, jj, a, b):
        ii, jj = np.asarray(ii), np.asarray(jj)
        if r == 0:
            return np.asarray(vol.ilxl[ii, jj, a:b], np.float32) * vol.inv
        acc = np.zeros((len(ii), b - a), np.float32)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                acc += vol.ilxl[np.clip(ii + di, 0, vol.nil - 1), np.clip(jj + dj, 0, vol.nxl - 1), a:b]
        return acc * (vol.inv / (2 * r + 1) ** 2)
    return get


def smoothed_line_data(line, smooth=1):
    """(ntr, ns) float32 line data with a running mean over `smooth` traces (cached on the line)."""
    r = max(0, (int(smooth) - 1) // 2)
    if r == 0:
        return line.data()
    cache = getattr(line, "_smooth_cache", None)
    if cache and cache[0] == r:
        return cache[1]
    d = line.data()
    cs = np.cumsum(np.pad(d.astype(np.float64), ((r + 1, r), (0, 0)), mode="edge"), axis=0)
    out = ((cs[2 * r + 1:] - cs[:-2 * r - 1]) / (2 * r + 1)).astype(np.float32)
    line._smooth_cache = (r, out)
    return out


def line_traces(line, smooth=1):
    data = smoothed_line_data(line, smooth)

    def get(ii, jj, a, b):
        return data[jj, a:b]
    return get


def ms_to_idx(t_ms, t0, dt):
    return (np.asarray(t_ms, np.float64) / 1000.0 - t0) / dt


def idx_to_ms(idx, t0, dt):
    return 1000.0 * (t0 + dt * np.asarray(idx, np.float64))


# ====================================================================== 2D line ties
def line_intersections(lines, max_vertices=2000):
    """lines: {name: (x array, y array)} (trace positions).  Returns list of (a, ia, b, ib) for every
    crossing of two lines, ia/ib = nearest trace index on each line."""
    names = [n for n in lines if len(lines[n][0]) > 1]
    dec = {}
    for n in names:
        x, y = np.asarray(lines[n][0], float), np.asarray(lines[n][1], float)
        step = max(1, len(x) // max_vertices)
        k = np.unique(np.r_[np.arange(0, len(x), step), len(x) - 1])
        dec[n] = (x[k], y[k], k)
    out = []
    for ia_, a in enumerate(names):
        xa, ya, ka = dec[a]
        for b in names[ia_ + 1:]:
            xb, yb, kb = dec[b]
            if xa.max() < xb.min() or xb.max() < xa.min() or ya.max() < yb.min() or yb.max() < ya.min():
                continue
            p, r = np.column_stack([xa[:-1], ya[:-1]]), np.column_stack([np.diff(xa), np.diff(ya)])
            q, s = np.column_stack([xb[:-1], yb[:-1]]), np.column_stack([np.diff(xb), np.diff(yb)])
            rxs = r[:, None, 0] * s[None, :, 1] - r[:, None, 1] * s[None, :, 0]
            qp = q[None, :, :] - p[:, None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (qp[..., 0] * s[None, :, 1] - qp[..., 1] * s[None, :, 0]) / rxs
                u = (qp[..., 0] * r[:, None, 1] - qp[..., 1] * r[:, None, 0]) / rxs
            hit = (np.abs(rxs) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
            seen = set()
            for i, j in zip(*np.nonzero(hit)):
                X, Y = p[i, 0] + t[i, j] * r[i, 0], p[i, 1] + t[i, j] * r[i, 1]
                xA, yA = np.asarray(lines[a][0], float), np.asarray(lines[a][1], float)
                xB, yB = np.asarray(lines[b][0], float), np.asarray(lines[b][1], float)
                lo, hi = ka[i], ka[i + 1] + 1
                ja = lo + int(np.argmin(np.hypot(xA[lo:hi] - X, yA[lo:hi] - Y)))
                lo, hi = kb[j], kb[j + 1] + 1
                jb = lo + int(np.argmin(np.hypot(xB[lo:hi] - X, yB[lo:hi] - Y)))
                if (ja, jb) not in seen:
                    seen.add((ja, jb))
                    out.append((a, int(ja), b, int(jb)))
    return out


# ====================================================================== horizon objects
class Horizon:
    def __init__(self, name, color=None):
        self.name = name
        self.color = tuple(color) if color else DEFAULT_COLORS[0]
        self.data = {}  # key -> float32 array (ms)
        self.visible = True

    def get(self, key, shape=None, create=False):
        a = self.data.get(key)
        if a is None and create:
            a = np.full(shape, np.nan, np.float32)
            self.data[key] = a
        return a

    def count(self):
        return int(sum(np.isfinite(a).sum() for a in self.data.values()))


class HorizonSet:
    """All horizons of a project + undo stack."""

    def __init__(self):
        self.items = []
        self.undo_stack = []
        self.max_undo = 30
        self.dirty = False

    def names(self):
        return [h.name for h in self.items]

    def get(self, name):
        return next((h for h in self.items if h.name == name), None)

    def new(self, name=None, color=None):
        base = name or f"Horizon {len(self.items) + 1}"
        name, k = base, 2
        while self.get(name):
            name, k = f"{base} ({k})", k + 1
        h = Horizon(name, color or DEFAULT_COLORS[len(self.items) % len(DEFAULT_COLORS)])
        self.items.append(h)
        self.dirty = True
        return h

    def delete(self, name):
        h = self.get(name)
        if h:
            pos = self.items.index(h)
            self.items.remove(h)
            self.undo_stack.append((_next_seq(), "delete", h, pos))
            self.dirty = True

    def rename(self, old, new):
        h = self.get(old)
        if h and not self.get(new):
            h.name = new
            self.dirty = True
            return True
        return False

    # -- undo: snapshot an array before an edit
    def snapshot(self, h, key):
        a = h.data.get(key)
        self.undo_stack.append((_next_seq(), "data", h, key, None if a is None else a.copy()))
        del self.undo_stack[:-self.max_undo]
        self.dirty = True

    def last_seq(self):
        return self.undo_stack[-1][0] if self.undo_stack else 0

    def undo(self):
        if not self.undo_stack:
            return None
        e = self.undo_stack.pop()
        if e[1] == "delete":
            h = e[2]
            if not self.get(h.name):
                self.items.insert(min(e[3], len(self.items)), h)
            self.dirty = True  # its files were removed by a save after the delete
            return h.name
        _, _, h, key, old = e
        if old is None:
            h.data.pop(key, None)
        else:
            h.data[key] = old
        self.dirty = True
        return h.name

    # -- persistence
    def folders(self):
        """{horizon name: folder name}: safe_name, made unique case-insensitively ('Top C1.5' / 'Top C1_5', or
        'top' / 'Top' on Windows, must not share a folder)."""
        out, used = {}, set()
        for h in self.items:
            base = safe_name(h.name)
            d, k = base, 2
            while os.path.normcase(d).lower() in used:
                d = f"{base}_{k}"
                k += 1
            used.add(os.path.normcase(d).lower())
            out[h.name] = d
        return out

    def to_json(self, folder_rel):
        fd = self.folders()
        return [{"name": h.name, "color": list(h.color), "visible": h.visible,
                 "dir": os.path.join(folder_rel, fd[h.name]),
                 "keys": {safe_name(k): k for k in h.data}} for h in self.items]

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        keep = set()
        fd = self.folders()
        for h in self.items:
            d = os.path.join(folder, fd[h.name])
            os.makedirs(d, exist_ok=True)
            keep.add(fd[h.name].lower())
            present = set()
            for k, a in h.data.items():
                fn = safe_name(k) + ".npy"
                np.save(os.path.join(d, fn), np.asarray(a, np.float32))
                present.add(fn)
            for fn in os.listdir(d):  # surveys whose picks were removed
                if fn.endswith(".npy") and fn not in present:
                    os.remove(os.path.join(d, fn))
        for dn in os.listdir(folder):  # folders of deleted / renamed horizons (only if they hold just .npy)
            d = os.path.join(folder, dn)
            if os.path.isdir(d) and dn.lower() not in keep and all(f.endswith(".npy") for f in os.listdir(d)):
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))
                os.rmdir(d)
        with open(os.path.join(folder, "horizons.json"), "w") as fh:
            json.dump(self.to_json("."), fh, indent=1)
        self.dirty = False

    @classmethod
    def load(cls, entries, base):
        s = cls()
        for e in entries or []:
            h = Horizon(e["name"], e.get("color"))
            h.visible = e.get("visible", True)
            d = e["dir"] if os.path.isabs(e["dir"]) else os.path.join(base, e["dir"])
            for fn_key, key in e.get("keys", {}).items():
                p = os.path.join(d, fn_key + ".npy")
                if os.path.exists(p):
                    h.data[key] = np.load(p).astype(np.float32)
            s.items.append(h)
        return s


# ====================================================================== high-level picking ops
class Picker:
    """Survey-aware editing/tracking operations on a HorizonSet (used by the GUI and tests)."""

    def __init__(self, project, hset):
        self.project = project
        self.hset = hset

    # -- survey helpers
    def survey(self, key):
        kind, name = key.split(":", 1)
        return self.project.volume(name) if kind == "3D" else self.project.line(name)

    def shape(self, key):
        s = self.survey(key)
        return (s.nil, s.nxl) if key.startswith("3D:") else (len(s.cdp),)

    def grid(self, h, key, create=True):
        return h.get(key, self.shape(key), create=create)

    def _traces(self, key, smooth=1):
        s = self.survey(key)
        return volume_traces(s, smooth) if key.startswith("3D:") else line_traces(s, smooth)

    # -- manual picks
    def pick_points(self, h, key, ii, jj, t_ms, mode="Peak", snap_ms=12.0, snapshot=True):
        """Snap and store picks at trace (ii, jj) (2D: ii ignored). Returns the stored times (ms)."""
        s = self.survey(key)
        ii, jj = np.atleast_1d(ii).astype(int), np.atleast_1d(jj).astype(int)
        t_ms = np.atleast_1d(np.asarray(t_ms, float))
        dt_ms = s.dt * 1000.0
        idx = ms_to_idx(t_ms, s.t0, s.dt)
        a = int(max(0, np.floor(np.nanmin(idx)) - snap_ms / dt_ms - 4))
        b = int(min(s.ns, np.ceil(np.nanmax(idx)) + snap_ms / dt_ms + 5))
        tr = self._traces(key)(ii, jj, a, b)
        new, ok = snap_times(tr, idx - a, mode, max(1, int(round(snap_ms / dt_ms))))
        out = idx_to_ms(np.where(ok, new + a, idx), s.t0, s.dt).astype(np.float32)
        if snapshot:
            self.hset.snapshot(h, key)
        g = self.grid(h, key)
        if key.startswith("3D:"):
            g[ii, jj] = out
        else:
            g[jj] = out
        return out

    def manual_line(self, h, key, ii, jj, t_ms, mode="Peak", snap_ms=4.0):
        """Drag-drawn line: interpolate between the drawn vertices along the section traces, snap lightly."""
        ii, jj, t_ms = np.asarray(ii, float), np.asarray(jj, float), np.asarray(t_ms, float)
        if len(jj) == 0:
            return
        by_j = key.startswith("2D:") or np.ptp(ii) <= np.ptp(jj)
        along = jj if by_j else ii
        order = np.argsort(along)
        along, ii, jj, t_ms = along[order], ii[order], jj[order], t_ms[order]
        k = np.arange(int(round(along[0])), int(round(along[-1])) + 1)
        t = np.interp(k, along, t_ms)
        if by_j:
            JJ, II = k, np.full(len(k), int(round(ii[0])) if len(ii) else 0)
        else:
            II, JJ = k, np.full(len(k), int(round(jj[0])))
        return self.pick_points(h, key, II, JJ, t, mode, snap_ms)

    def erase(self, h, key, ii=None, jj=None, t_ms=None, tol_ms=None):
        """Erase picks at traces (ii, jj); with t_ms/tol_ms only picks near that time."""
        g = h.get(key)
        if g is None:
            return 0
        self.hset.snapshot(h, key)
        ii = None if ii is None else np.atleast_1d(ii).astype(int)
        jj = np.atleast_1d(jj).astype(int)
        sel = (ii, jj) if key.startswith("3D:") else (jj,)
        cur = g[sel]
        hit = np.isfinite(cur)
        if t_ms is not None and tol_ms:
            hit &= np.abs(cur - np.atleast_1d(t_ms)) <= tol_ms
        vals = cur.copy()
        vals[hit] = np.nan
        g[sel] = vals
        return int(hit.sum())

    def clear_section(self, h, key, mode=None, number=None):
        g = h.get(key)
        if g is None:
            return
        self.hset.snapshot(h, key)
        if key.startswith("2D:"):
            g[:] = np.nan
            return
        v = self.survey(key)
        if mode == "Inline":
            g[v.il_index(number)] = np.nan
        elif mode == "Crossline":
            g[:, v.xl_index(number)] = np.nan
        elif mode in (None, "All"):
            g[:] = np.nan

    # -- tracking
    def barrier(self, key, faults, tol_samples=2.0):
        """EdgeBarrier for a survey from a list of Fault objects (None if no faults)."""
        from . import faults as flt
        faults = [f for f in (faults or []) if len(f.triangles())]
        if not faults:
            return None
        s = self.survey(key)
        return flt.barrier_3d(faults, s, tol_samples) if key.startswith("3D:") else \
            flt.barrier_line(faults, s, tol_samples)

    def track(self, h, key, seeds_ij=None, params=None, scope=None, snapshot=True, progress=None, faults=None):
        """Autotrack on one survey from seed traces (list of (i, j); None = all current picks).
        scope: None (whole survey) or ("Inline", il) / ("Crossline", xl) for 3D. Returns nodes added."""
        params = params or TrackParams()
        s = self.survey(key)
        g = self.grid(h, key)
        if snapshot:
            self.hset.snapshot(h, key)
        g2 = g if g.ndim == 2 else g[None, :]
        idx = ms_to_idx(g2, s.t0, s.dt)
        seeds = np.zeros(idx.shape, bool)
        if seeds_ij is None:
            seeds = np.isfinite(idx)
        else:
            for i, j in seeds_ij:
                seeds[0 if g.ndim == 1 else int(i), int(j)] = True
        allowed = None
        if scope and g.ndim == 2:
            allowed = np.zeros(idx.shape, bool)
            if scope[0] == "Inline":
                allowed[s.il_index(scope[1])] = True
            elif scope[0] == "Crossline":
                allowed[:, s.xl_index(scope[1])] = True
            else:
                allowed[:] = True
        get = self._traces(key, params.smooth)
        eb = self.barrier(key, faults)
        n = track_grid(get, idx, seeds, params, s.dt * 1000.0, s.ns, allowed=allowed, progress=progress,
                       blocked=eb.blocked(idx) if eb else None)
        new = idx_to_ms(idx, s.t0, s.dt).astype(np.float32)
        g[...] = new if g.ndim == 2 else new[0]
        return n

    def track_2d_network(self, h, start_lines, params=None, snap_ms=None, progress=None, faults=None):
        """Track the start lines, then follow every line tie: where a tracked line crosses a line without a
        pick at the tie, seed that line with the (snapped) time and track it too.  Returns {line: added}."""
        params = params or TrackParams()
        snap_ms = params.max_jump_ms if snap_ms is None else snap_ms
        names = [l["name"] for l in self.project.lines]
        geo = {}
        for n in names:
            try:
                ln = self.project.line(n)
            except Exception:
                continue
            if ln.has_coords:
                geo[n] = (ln.x, ln.y)
        ties = line_intersections(geo)
        queue = [n for n in start_lines]
        done = {}
        while queue:
            a = queue.pop(0)
            key = f"2D:{a}"
            if h.get(key) is None or not np.isfinite(h.get(key)).any():
                continue
            done[a] = done.get(a, 0) + self.track(h, key, None, params, progress=progress, faults=faults)
            ga = h.get(key)
            for (p, ip, q, iq) in ties:
                if a not in (p, q):
                    continue
                ia, b, ib = (ip, q, iq) if a == p else (iq, p, ip)
                if not np.isfinite(ga[ia]):
                    continue
                gb = self.grid(h, f"2D:{b}")
                if np.isfinite(gb[ib]):
                    continue
                t = self.pick_points(h, f"2D:{b}", 0, ib, ga[ia], params.mode, snap_ms)
                if np.isfinite(t).all():
                    queue.append(b)
        return done


# ====================================================================== display helpers
def section_curve(h, key, survey, mode=None, number=None):
    """(x axis values, times ms) of a horizon on the displayed section, NaN gaps kept; None if no picks."""
    g = h.get(key)
    if g is None:
        return None
    if key.startswith("2D:"):
        return survey.cdp, g
    if mode == "Inline":
        return survey.xlines, g[survey.il_index(number)]
    if mode == "Crossline":
        return survey.inlines, g[:, survey.xl_index(number)]
    return None


def slice_points(h, key, survey, t_ms, tol_ms=None):
    """(xl values, il values) of horizon picks within tol of a time slice."""
    g = h.get(key)
    if g is None:
        return None
    tol = tol_ms if tol_ms is not None else survey.dt * 500.0
    i, j = np.nonzero(np.abs(g - t_ms) <= tol)
    return survey.xlines[j], survey.inlines[i]


def grid_transform(vol):
    """(m11, m12, m21, m22, dx, dy) for a QTransform mapping image pixel (col = xl index, row = il index)
    corners to map X/Y (pixel centres land on the trace positions)."""
    ax, ay = vol.ax, vol.ay
    m11, m12 = ax[2] * vol.xl_step, ay[2] * vol.xl_step
    m21, m22 = ax[1] * vol.il_step, ay[1] * vol.il_step
    x0, y0 = vol.xy(vol.il0, vol.xl0)
    dx = float(x0) - 0.5 * (m11 + m21)
    dy = float(y0) - 0.5 * (m12 + m22)
    return m11, m12, m21, m22, dx, dy


def structure_rgba(grid, lut, vmin=None, vmax=None):
    """Colour a time grid (NaN transparent). Shallow = first LUT colour. Returns (rgba uint8, vmin, vmax)."""
    g = np.asarray(grid, float)
    fin = np.isfinite(g)
    if not fin.any():
        return np.zeros(g.shape + (4,), np.uint8), 0.0, 1.0
    vmin = float(np.nanmin(g)) if vmin is None else vmin
    vmax = float(np.nanmax(g)) if vmax is None else vmax
    k = np.clip((g - vmin) / max(1e-9, vmax - vmin) * 255, 0, 255)
    k = np.where(fin, k, 0).astype(np.int64)
    out = np.asarray(lut)[k].astype(np.uint8)
    out[..., 3] = np.where(fin, 255, 0)
    return out, vmin, vmax
