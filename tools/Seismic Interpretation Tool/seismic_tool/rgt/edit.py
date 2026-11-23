"""RGT step D: interactive editing with fast local re-solves.

Edits (all undoable):
  - control point: "this sample (trace i, j at time t) has geologic age v" - typically: grab an iso-RGT
    contour (its value v) and put it on the reflector where it belongs;
  - horizon constraint: all picks of a horizon get one (free) age;
  - fault re-link: re-solve around a fault after it was drawn or edited.
Each edit rebuilds only the node equations inside a window of node columns around the edit, keeps the
window border fixed to the current solution (so the model outside does not change), solves the small
system, writes the changed node values back to rgt_nodes.npz and re-interpolates only the affected
inlines of rgt.npy (and of the horizon stack if there is one).  2D lines: same idea on a window of one line.
"""
import json
import os

import numpy as np

from ..faults import next_seq
from . import solve as S


def _params_from(meta):
    p = S.RGTParams()
    for k, v in (meta.get("params") or {}).items():
        if hasattr(p, k):
            setattr(p, k, v)
    return p


class RGTEditor3D:
    def __init__(self, vol, folder=None, radius=6):
        from .dip import DipField
        self.vol = vol
        self.folder = folder or os.path.join(vol.store, "rgt")
        self.radius = radius  # node columns around an edit
        with np.load(os.path.join(self.folder, "rgt_nodes.npz")) as z:  # closed at once (Windows rewrites it)
            self.tau = z["tau"].astype(np.float64)
            self.Ipos, self.Jpos, self.node_t = z["Ipos"].copy(), z["Jpos"].copy(), z["node_t"].copy()
        with open(os.path.join(self.folder, "rgt.json")) as fh:
            self.meta = json.load(fh)
        self.p = _params_from(self.meta)
        self.p.dx_tol = 1e-5  # local systems are small: converge fully
        self.dip = DipField(os.path.join(vol.store, "rgt"))
        self.undo_stack = []
        self.edits = self._load_edits()

    # ------------------------------------------------------------ bookkeeping
    def _load_edits(self):
        p = os.path.join(self.folder, "edits.json")
        if os.path.exists(p):
            with open(p) as fh:
                return json.load(fh)
        return []

    def _save(self):
        np.savez(os.path.join(self.folder, "rgt_nodes.npz"), tau=self.tau.astype(np.float32), Ipos=self.Ipos,
                 Jpos=self.Jpos, node_t=self.node_t)
        with open(os.path.join(self.folder, "edits.json"), "w") as fh:
            json.dump(self.edits, fh, indent=1)

    def last_seq(self):
        return self.undo_stack[-1][0] if self.undo_stack else 0

    def value_at(self, i, j, t_ms):
        """Current RGT at trace (i, j) and time t (from the node solution)."""
        ni = self._interp()
        row = ni.inline(int(i))[int(j)]
        return float(np.interp(t_ms, self.vol.times, row))

    def _interp(self):
        return S.NodeInterp(self.tau, self.Ipos, self.Jpos, self.node_t, self.p.node_ms, self.vol.times,
                            self.vol.nil, self.vol.nxl)

    def window_around(self, i_list, j_list, radius=None):
        r = self.radius if radius is None else radius
        I = np.searchsorted(self.Ipos, np.asarray(i_list))
        J = np.searchsorted(self.Jpos, np.asarray(j_list))
        nI, nJ = len(self.Ipos), len(self.Jpos)
        return (max(0, int(I.min()) - r), min(nI - 1, int(I.max()) + r),
                max(0, int(J.min()) - r), min(nJ - 1, int(J.max()) + r))

    # ------------------------------------------------------------ local solve
    def _resolve(self, win, points=(), horizons=(), faults=None, label="edit"):
        I0, I1, J0, J1 = win
        nI, nJ = len(self.Ipos), len(self.Jpos)
        Iw, Jw = self.Ipos[I0:I1 + 1], self.Jpos[J0:J1 + 1]
        sysm = S.build_3d(self.vol, self.dip, self.p, faults, None, None, None, Iw, Jw)
        sysm.datum = False
        sv = sysm.surveys["3D"]
        K = sv["K"]
        nIw, nJw = len(Iw), len(Jw)
        cur = self.tau[I0:I1 + 1, J0:J1 + 1]
        # border columns fixed to the current model (unless the border is the survey edge)
        II, JJ = np.meshgrid(np.arange(nIw), np.arange(nJw), indexing="ij")
        border = ((II == 0) & (I0 > 0)) | ((II == nIw - 1) & (I1 < nI - 1)) | \
            ((JJ == 0) & (J0 > 0)) | ((JJ == nJw - 1) & (J1 < nJ - 1))
        if not border.any() and not points and not horizons:
            border[:] = True
        cols = np.nonzero(border.ravel())[0]
        if len(cols):
            u = (sv["offset"][cols][:, None] + np.arange(K)[None, :]).ravel()
            vals = cur.reshape(-1, K)[cols].ravel()
            sysm.eq.add([u], [np.ones(len(u))], vals, 100.0)
        if not len(cols):  # nothing fixed: keep the datum of the current model
            sysm.datum = True
        # control points: bilinear over the 4 node columns around the trace, linear in time
        for (i, j, t_ms, v) in points:
            ci = np.clip(np.searchsorted(Iw, i, "right"), 1, nIw - 1)
            cj = np.clip(np.searchsorted(Jw, j, "right"), 1, nJw - 1)
            wi = np.clip((i - Iw[ci - 1]) / max(1, Iw[ci] - Iw[ci - 1]), 0, 1)
            wj = np.clip((j - Jw[cj - 1]) / max(1, Jw[cj] - Jw[cj - 1]), 0, 1)
            cl, cf = [], []
            for a, wa in ((ci - 1, 1 - wi), (ci, wi)):
                for b, wb in ((cj - 1, 1 - wj), (cj, wj)):
                    c = a * nJw + b
                    u0, u1, w0, w1, ok = sysm._interp_cols("3D", np.array([c]), np.array([t_ms]))
                    if not ok[0]:
                        continue
                    cl += [u0, u1]
                    cf += [w0 * wa * wb, w1 * wa * wb]
            if cl:
                sysm.eq.add(cl, cf, np.array([v]), 1e4)  # near-hard constraint
        for name, grid in horizons:
            sub = grid[np.ix_(Iw, Jw)].ravel()
            sysm.add_horizon(name, "3D", np.arange(nIw * nJw), sub)
        sysm.p = self.p
        old = self.tau[I0:I1 + 1, J0:J1 + 1].copy()
        x0 = cur.ravel()
        sysm.solve(None, None, x0)
        new = sysm.node_values("3D").reshape(nIw, nJw, K)
        self.undo_stack.append((next_seq(), win, old, label))
        self.tau[I0:I1 + 1, J0:J1 + 1] = new
        self.edits.append({"type": label, "window": list(map(int, win)),
                           "points": [list(map(float, pt)) for pt in points],
                           "horizons": [n for n, _ in horizons]})
        self._write_back(win)
        self._save()
        return sysm

    def _write_back(self, win):
        """Re-interpolate the inlines whose interpolation uses the changed node rows; update the stack."""
        I0, I1, J0, J1 = win
        nI = len(self.Ipos)
        a = int(self.Ipos[max(0, I0 - 1)])
        b = int(self.Ipos[min(nI - 1, I1 + 1)])
        lo, hi = self.meta["lo"], self.meta["hi"]
        scale = 65535.0 / (hi - lo)
        arr = np.load(os.path.join(self.folder, "rgt.npy"), mmap_mode="r+")
        ni = self._interp()
        rows = {}
        step = np.float32((hi - lo) / 65535.0)
        for i in range(a, b + 1):
            q = np.clip(np.round((ni.inline(i) - lo) * scale), 0, 65535).astype(np.uint16)
            arr[i] = q
            rows[i] = lo + q.astype(np.float32) * step  # what RGTVolume.decode returns (stack consistency)
        arr.flush()
        del arr
        self.changed_inlines = (a, b)
        from .stack import HorizonStack3D, iso_times
        if HorizonStack3D.exists(self.folder):
            st = HorizonStack3D(self.folder)
            levels = st.levels
            del st
            smm = np.load(os.path.join(self.folder, "stack.npy"), mmap_mode="r+")
            for i, r in rows.items():
                smm[:, i, :] = iso_times(r, levels, self.vol.times).T
            smm.flush()
            del smm

    # ------------------------------------------------------------ edits
    def add_point(self, i, j, t_ms, value, radius=None, faults=None):
        """Constrain RGT(i, j, t) = value and re-solve locally."""
        win = self.window_around([i], [j], radius)
        return self._resolve(win, [(int(i), int(j), float(t_ms), float(value))], (), faults, "point")

    def move_level(self, i, j, t_from_ms, t_to_ms, radius=None, faults=None, half_ms=24.0, i_to=None, j_to=None):
        """Grab the iso-RGT surface passing through (i, j, t_from) and move it to t_to on trace (i_to, j_to)
        (default: the same trace).  The model is shifted as a segment (t_from +/- half_ms -> t_to +/- half_ms)
        so the levels just above and below move with it (a single-point constraint would fold the model)."""
        i_to = i if i_to is None else i_to
        j_to = j if j_to is None else j_to
        v = self.value_at(i, j, t_from_ms)
        ni = self._interp()
        row = ni.inline(int(i))[int(j)]
        d = np.arange(-half_ms, half_ms + 1e-6, self.p.node_ms / 2)
        vals = np.interp(t_from_ms + d, self.vol.times, row)
        pts = [(int(i_to), int(j_to), float(t_to_ms + dd), float(vv)) for dd, vv in zip(d, vals)]
        win = self.window_around([i, i_to], [j, j_to], radius)
        return self._resolve(win, pts, (), faults, "move"), v

    def add_horizon(self, name, grid, faults=None, margin=2):
        """All picks of a horizon (3D grid, ms) get one age; window = bounding box of the picks + margin."""
        fin = np.isfinite(grid)
        if not fin.any():
            return None
        ii, jj = np.nonzero(fin)
        win = self.window_around([ii.min(), ii.max()], [jj.min(), jj.max()], margin)
        return self._resolve(win, (), [(name, grid)], faults, "horizon")

    def relink_fault(self, fault, faults, margin=3):
        """Re-solve around a (new or edited) fault with all current faults as barriers."""
        from ..faults import tri_index_coords
        tris = fault.triangles()
        if not len(tris):
            return None
        ii, jj, _ = tri_index_coords(self.vol, tris)
        ii = np.clip(ii, 0, self.vol.nil - 1)
        jj = np.clip(jj, 0, self.vol.nxl - 1)
        win = self.window_around([ii.min(), ii.max()], [jj.min(), jj.max()], margin)
        return self._resolve(win, (), (), faults, "fault")

    def undo(self):
        if not self.undo_stack:
            return None
        _, win, old, label = self.undo_stack.pop()
        I0, I1, J0, J1 = win
        self.tau[I0:I1 + 1, J0:J1 + 1] = old
        if self.edits:
            self.edits.pop()
        self._write_back(win)
        self._save()
        return label


class RGTEditor2D:
    """Local edits on one 2D line of a solved network (tie columns stay fixed so ties remain consistent)."""

    def __init__(self, project, name, radius=12):
        from .dip import line_dip
        self.project, self.name, self.radius = project, name, radius
        self.folder = os.path.join(project.seismic_dir(), "rgt")
        self.line = project.line(name)
        self.tau = S.load_rgt_line(self.folder, name)
        if self.tau is None:
            raise RuntimeError("no RGT for this line")
        self.dip, _ = line_dip(self.line, self.folder)
        self.p = S.RGTParams(node_traces=4)
        self.p.dx_tol = 1e-5
        self.undo_stack = []

    def last_seq(self):
        return self.undo_stack[-1][0] if self.undo_stack else 0

    def value_at(self, j, t_ms):
        return float(np.interp(t_ms, self.line.times, self.tau[int(j)]))

    def add_point(self, j, t_ms, value, faults=None):
        l = self.line
        n = len(l.cdp)
        step = self.p.node_traces
        a = max(0, int(j) - self.radius * step)
        b = min(n - 1, int(j) + self.radius * step)
        pos = np.unique(np.r_[np.arange(a, b + 1, step), b, int(j)])
        dt_ms, t0_ms = l.dt * 1000.0, l.t0 * 1000.0
        ntm = S.node_times(l.ns, dt_ms, t0_ms, self.p.node_ms)
        sysm = S.NodeSystem(self.p)
        sysm.datum = False
        key = "L"
        sv_off = sysm.add_survey(key, len(pos), ntm)  # noqa: F841
        from ..faults import barrier_line
        eb = barrier_line(faults, l) if faults else None
        hit = (lambda k, t, x=0.0, e=eb: e.hit(1, 0, k, t, x)) if eb else None
        for ia, kk, ib, tb, w in S.links_along(l.data(), self.dip / dt_ms, pos, (ntm - t0_ms) / dt_ms, self.p,
                                               dt_ms, hit):
            sysm.add_links(key, ia, kk, key, ib, tb * dt_ms + t0_ms, w)
        sv = sysm.surveys[key]
        K = sv["K"]
        cur = np.array([np.interp(ntm, l.times, self.tau[c]) for c in pos])  # (ncols, K)
        fixed = [0, len(pos) - 1] if (a > 0 or b < n - 1) else []
        fixed = [c for c in fixed if (c == 0 and a > 0) or (c == len(pos) - 1 and b < n - 1)]
        for c in fixed:
            u = sv["offset"][c] + np.arange(K)
            sysm.eq.add([u], [np.ones(K)], cur[c], 100.0)
        if not fixed:
            sysm.datum = True
        c = int(np.searchsorted(pos, int(j)))
        t_arr = np.atleast_1d(np.asarray(t_ms, float))
        v_arr = np.broadcast_to(np.asarray(value, float), t_arr.shape)
        u0, u1, w0, w1, ok = sysm._interp_cols(key, np.full(len(t_arr), c), t_arr)
        if ok.any():
            sysm.eq.add([u0[ok], u1[ok]], [w0[ok], w1[ok]], v_arr[ok], 1e4)
        sysm.solve(None, None, cur.ravel())
        new = sysm.node_values(key)
        old = self.tau[a:b + 1].copy()
        # interpolate the window back to every trace
        u = np.clip((l.times - ntm[0]) / self.p.node_ms, 0, K - 1)
        m = np.minimum(np.floor(u).astype(int), K - 2)
        w = u - m
        colfull = new[:, m] * (1 - w) + new[:, m + 1] * w
        k = np.arange(a, b + 1)
        c1 = np.clip(np.searchsorted(pos, k, "right"), 1, len(pos) - 1)
        c0 = c1 - 1
        wk = ((k - pos[c0]) / np.maximum(1, pos[c1] - pos[c0]))[:, None]
        self.tau[a:b + 1] = colfull[c0] * (1 - wk) + colfull[c1] * wk
        self.undo_stack.append((next_seq(), (a, b), old))
        self._save()
        return sysm

    def move_level(self, j, t_from_ms, t_to_ms, faults=None, half_ms=24.0, j_to=None):
        j_to = j if j_to is None else j_to
        v = self.value_at(j, t_from_ms)
        d = np.arange(-half_ms, half_ms + 1e-6, self.p.node_ms / 2)
        vals = np.interp(t_from_ms + d, self.line.times, self.tau[int(j)])
        return self.add_point(j_to, t_to_ms + d, vals, faults), v

    def _save(self):
        lo, hi = float(self.tau.min()) - 1, float(self.tau.max()) + 1
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.name)
        np.savez(os.path.join(self.folder, safe + "_rgt.npz"),
                 q=np.round((self.tau - lo) / (hi - lo) * 65535).astype(np.uint16), lo=lo, hi=hi)

    def undo(self):
        if not self.undo_stack:
            return None
        _, (a, b), old = self.undo_stack.pop()
        self.tau[a:b + 1] = old
        self._save()
        return "point"
