"""Wells on seismic: time-depth functions, tops in TWT, synthetic seismograms, well ties.

Well data come from the linked Well Correlation project (the sibling tool's package is imported from
tools/Well Correlation Tool).  Depths there are MD below the well reference (KB/DF); TVDSS = TVD - elevation
(m below MSL); seismic time is TWT ms below the seismic reference datum (SRD = MSL).

Time-depth per well, in order of preference:
  1. checkshot: wells/<Well>/checkshot/*.dlis (CSAT/VSI, see well_correlation.dlis.checkshot_from_dlis) or a
     CSV with TVD_SRD_m + TWT_ms columns (as written by --checkshot);
  2. sonic: TWT = t(z_top) + 2 * integral(DT dz); t(z_top) taken from the nearest well with a checkshot (at the
     same TVDSS), else a default velocity function (water 1500 m/s to the seafloor, then 2000 m/s);
  3. no sonic: the nearest checkshot well's function, or the default.
A saved well tie (bulk shift + stretch about the log centre) is applied on top.
"""
import csv
import glob
import os
import sys

import numpy as np

_WC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Well Correlation Tool"))


def wc():
    """The Well Correlation Tool package (imported from the sibling tool folder)."""
    if _WC not in sys.path:
        sys.path.insert(0, _WC)
    import well_correlation  # noqa: F401
    import well_correlation.project  # noqa: F401
    return well_correlation


# ====================================================================== time-depth
def default_td(water_depth=120.0, v_water=1500.0, v_sed=2000.0, zmax=8000.0):
    z = np.array([0.0, water_depth, zmax])
    t = np.array([0.0, 2000.0 * water_depth / v_water, 2000.0 * (water_depth / v_water + (zmax - water_depth) / v_sed)])
    return z, t


def read_checkshot_csv(path):
    z, t = [], []
    with open(path, newline="") as fh:
        rows = [r for r in fh if not r.startswith("#")]
    for r in csv.DictReader(rows):
        try:
            z.append(float(r.get("TVD_SRD_m") or r.get("TVDSS_m") or r["TVD"]))
            t.append(float(r.get("TWT_ms") or r["TWT"]))
        except (KeyError, ValueError, TypeError):
            continue
    if len(z) < 2:
        return None
    return np.array(z), np.array(t)


def checkshot_for(well, base):
    """(z_ss m, twt ms, source) from the well's checkshot folder, or None."""
    dirs = {os.path.dirname(os.path.dirname(p)) for p in well.las_paths}
    if base:
        dirs.add(os.path.join(base, "wells", well.name))
    files = []
    for d in dirs:
        files += glob.glob(os.path.join(d, "checkshot", "*"))
    zs, ts, srcs = [], [], []
    for f in sorted(set(files)):
        ext = f.lower().rsplit(".", 1)[-1]
        try:
            if ext == "dlis":
                from well_correlation.dlis import checkshot_from_dlis
                t = checkshot_from_dlis(f)
                if t is None:
                    continue
                z, tw = t["tvd"], t["twt_ms"]
            elif ext == "csv":
                r = read_checkshot_csv(f)
                if r is None:
                    continue
                z, tw = r
            else:
                continue  # the NZ checkshot LAS files hold no time-depth pairs
        except Exception:
            continue
        zs.append(np.asarray(z, float))
        ts.append(np.asarray(tw, float))
        srcs.append(os.path.basename(f))
    if not zs:
        return None
    # merge all surveys of the well (intermediate + main runs): average levels within 0.5 m
    z, t = np.concatenate(zs), np.concatenate(ts)
    key = np.round(z * 2) / 2
    uk, inv = np.unique(key, return_inverse=True)
    t = np.bincount(inv, t) / np.bincount(inv)
    z = uk
    keep = t > np.maximum.accumulate(np.r_[-np.inf, t[:-1]])  # strictly later than every shallower level
    z, t = np.r_[0.0, z[keep]], np.r_[0.0, t[keep]]
    return z, t, " + ".join(srcs)


def sonic_td(depth_md, dt_usft, tvdss_fn, anchor):
    """Integrate sonic: returns (z_ss, twt) over the sonic interval, anchored at its top with anchor(z)->twt."""
    dt = np.asarray(dt_usft, float)
    ok = np.isfinite(dt) & (dt > 30) & (dt < 250)
    if ok.sum() < 50:
        return None
    md = np.asarray(depth_md, float)
    i0, i1 = np.nonzero(ok)[0][[0, -1]]
    md, dt = md[i0:i1 + 1], dt[i0:i1 + 1]
    dt = np.interp(md, md[np.isfinite(dt) & (dt > 30) & (dt < 250)], dt[np.isfinite(dt) & (dt > 30) & (dt < 250)])
    z = tvdss_fn(md)
    slow = dt * 1e-6 / 0.3048  # s/m
    dz = np.diff(z)
    twt = np.r_[0.0, np.cumsum(2000.0 * 0.5 * (slow[1:] + slow[:-1]) * dz)]
    return z, twt + anchor(z[0])


class WellTD:
    """Time-depth for one well (z_ss -> twt), with an optional tie correction."""

    def __init__(self, name, z, t, source):
        self.name, self.z, self.t, self.source = name, np.asarray(z, float), np.asarray(t, float), source
        self.shift = 0.0
        self.stretch = 1.0
        self.t_centre = float(np.median(self.t)) if len(self.t) else 0.0
        self.warp_in = None  # optional piecewise warp after shift/stretch (event pairs / DTW, see tie.py)
        self.warp_out = None

    def twt_raw(self, z_ss):
        z = np.asarray(z_ss, float)
        t = np.interp(z, self.z, self.t)
        # linear extrapolation with the end interval velocities
        if len(self.z) > 1:  # end slowness averaged over >= 50 m (one washout sample must not set it)
            za = min(self.z[-1], self.z[0] + 50.0)
            zb = max(self.z[0], self.z[-1] - 50.0)
            s0 = (np.interp(za, self.z, self.t) - self.t[0]) / max(1e-9, za - self.z[0])
            s1 = (self.t[-1] - np.interp(zb, self.z, self.t)) / max(1e-9, self.z[-1] - zb)
            t = np.where(z < self.z[0], self.t[0] + (z - self.z[0]) * s0, t)
            t = np.where(z > self.z[-1], self.t[-1] + (z - self.z[-1]) * s1, t)
        return t

    def twt(self, z_ss):
        t = self.twt_raw(z_ss)
        t = t + self.shift + (self.stretch - 1.0) * (t - self.t_centre)
        if self.warp_in is not None and len(self.warp_in):
            t = t + np.interp(t, self.warp_in, self.warp_out - self.warp_in)
        return t

    def depth(self, twt):
        """Inverse: TVDSS at a (corrected) TWT."""
        zz = np.linspace(self.z.min() - 500, self.z.max() + 2000, 4000)
        return np.interp(twt, self.twt(zz), zz)


class WellSet:
    """Wells of the linked Well Correlation project with time-depth functions and ties."""

    def __init__(self, seisproject):
        self.sp = seisproject
        self.wcp = None
        self.td = {}
        path = seisproject.well_project
        if path and os.path.exists(path):
            P = wc().project.Project
            self.wcp = P.open(path)
        self._checkshots = None

    @property
    def names(self):
        return list(self.wcp.wells) if self.wcp else []

    def well(self, name):
        return self.wcp.wells[name]

    def base(self):
        return os.path.dirname(os.path.abspath(self.wcp.path)) if self.wcp and self.wcp.path else None

    def checkshots(self):
        if self._checkshots is None:
            self._checkshots = {}
            for n, w in (self.wcp.wells.items() if self.wcp else []):
                r = checkshot_for(w, self.base())
                if r is not None:
                    self._checkshots[n] = r
        return self._checkshots

    def nearest_checkshot(self, name):
        cs = self.checkshots()
        w = self.well(name)
        best = None
        for n, r in cs.items():
            o = self.well(n)
            if w.x is None or o.x is None:
                continue
            d = np.hypot(w.x - o.x, w.y - o.y)
            if best is None or d < best[0]:
                best = (d, n, r)
        return best

    def td_for(self, name):
        """WellTD (cached). Applies the saved tie from the seismic project."""
        if name in self.td:
            return self.td[name]
        w = self.well(name)
        cs = self.checkshots().get(name)
        if cs is not None and self.sp.display.get("td_mode") == "calibrated sonic":
            td = self.calibrated_td(name)
            if td is not None:
                self.td[name] = td
                return td
        if cs is not None:
            td = WellTD(name, cs[0], cs[1], f"checkshot {cs[2]}")
        else:
            near = self.nearest_checkshot(name)
            if near is not None:
                zf, tf = near[2][0], near[2][1]
                anchor_src = f"checkshot of {near[1]} ({near[0] / 1000:.1f} km)"
            else:
                zf, tf = default_td()
                anchor_src = "default velocity"
            anchor = lambda z, zf=zf, tf=tf: float(np.interp(z, zf, tf))  # noqa: E731
            try:
                self.wcp.load_well(name)
            except Exception:
                pass
            dt = w.logs.get("DT") if w.logs else None
            st = sonic_td(w.depth, dt, w.md_to_tvdss, anchor) if dt is not None else None
            if st is not None:
                z, t = st
                # above the sonic: the anchor function
                above = zf < z[0]
                td = WellTD(name, np.r_[zf[above], z], np.r_[tf[above], t], f"sonic anchored to {anchor_src}")
            else:
                td = WellTD(name, zf, tf, anchor_src)
        tie = (self.sp.display.get("well_ties") or {}).get(name)
        if tie:
            td.shift, td.stretch = float(tie.get("shift_ms", 0)), float(tie.get("stretch", 1))
            td.t_centre = float(tie.get("t_centre", td.t_centre))
            if tie.get("warp_in"):
                td.warp_in = np.asarray(tie["warp_in"], float)
                td.warp_out = np.asarray(tie["warp_out"], float)
        self.td[name] = td
        return td

    def calibrated_td(self, name):
        """WellTD from the sonic calibrated to the well's own checkshot (drift interpolated between levels), or
        None when the well lacks a checkshot or a usable sonic. The saved tie is applied as in td_for."""
        from .tie import calibrate_sonic
        cs = self.checkshots().get(name)
        if cs is None:
            return None
        w = self.well(name)
        try:
            self.wcp.load_well(name)
        except Exception:
            pass
        dt = w.logs.get("DT") if w.logs else None
        anchor = lambda z: float(np.interp(z, cs[0], cs[1]))  # noqa: E731
        st = sonic_td(w.depth, dt, w.md_to_tvdss, anchor) if dt is not None else None
        if st is None:
            return None
        z, t = st
        tcal, _ = calibrate_sonic(z, t, cs[0], cs[1])
        above, below = cs[0] < z[0], cs[0] > z[-1]
        td = WellTD(name, np.r_[cs[0][above], z, cs[0][below]], np.r_[cs[1][above], tcal, cs[1][below]],
                    f"sonic calibrated to checkshot {cs[2]}")
        keep = (td.t > np.maximum.accumulate(np.r_[-np.inf, td.t[:-1]])) & \
            (td.z > np.maximum.accumulate(np.r_[-np.inf, td.z[:-1]]))  # strictly increasing in both
        td.z, td.t = td.z[keep], td.t[keep]
        tie = (self.sp.display.get("well_ties") or {}).get(name)
        if tie:
            td.shift, td.stretch = float(tie.get("shift_ms", 0)), float(tie.get("stretch", 1))
            td.t_centre = float(tie.get("t_centre", td.t_centre))
            if tie.get("warp_in"):
                td.warp_in, td.warp_out = np.asarray(tie["warp_in"], float), np.asarray(tie["warp_out"], float)
        return td

    def tops_twt(self, name):
        """{formation: twt ms} for the well's tops (MD -> TVDSS -> TWT)."""
        tops = (self.wcp.tops or {}).get(name, {}) if self.wcp else {}
        if not tops:
            return {}
        w = self.well(name)
        try:
            self.wcp.load_well(name)
        except Exception:
            pass
        td = self.td_for(name)
        return {fm: float(td.twt(w.md_to_tvdss(md))) for fm, md in tops.items() if md is not None}

    def formation_color(self, fm):
        for f in (self.wcp.formations if self.wcp else []):
            if f.get("name") == fm:
                c = f.get("color", "#000000").lstrip("#")
                return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
        return (0, 0, 0)

    def save_tie(self, name, shift, stretch, t_centre, corr, extra=None):
        ties = self.sp.display.setdefault("well_ties", {})
        ties[name] = {"shift_ms": float(shift), "stretch": float(stretch), "t_centre": float(t_centre),
                      "corr": float(corr), **(extra or {})}
        self.td.pop(name, None)

    # -------------------------------------------------------------- geometry
    def near_section(self, kind, survey, mode=None, number=None, tol_traces=3.0, tol_m=500.0):
        """[(well name, x value on the section axis, distance m)] for wells close to the section shown."""
        out = []
        for n in self.names:
            w = self.well(n)
            if w.x is None:
                continue
            if kind == "3D":
                il, xl = survey.ilxl_at(w.x, w.y)
                if not (survey.inlines[0] - survey.il_step <= il <= survey.inlines[-1] + survey.il_step and
                        survey.xlines[0] - survey.xl_step <= xl <= survey.xlines[-1] + survey.xl_step):
                    continue
                if mode == "Inline" and abs(il - number) <= tol_traces * survey.il_step:
                    out.append((n, float(xl), abs(il - number)))
                elif mode == "Crossline" and abs(xl - number) <= tol_traces * survey.xl_step:
                    out.append((n, float(il), abs(xl - number)))
                elif mode in ("Time slice", "Stratal slice"):
                    out.append((n, (float(xl), float(il)), 0.0))
            else:
                if not survey.has_coords:
                    continue
                d = np.hypot(survey.x - w.x, survey.y - w.y)
                k = int(np.argmin(d))
                if d[k] <= tol_m:
                    out.append((n, float(survey.cdp[k]), float(d[k])))
        return out

    def trace_at_well(self, kind, survey, name, n_around=0, strict=False, max_dist_m=500.0):
        """Seismic trace(s) at the well: (times ms, traces (m, ns)) - nearest trace (+ neighbours).
        strict: raise ValueError when the well is outside the 3D survey / farther than max_dist_m from the 2D line
        (otherwise the nearest edge trace is used)."""
        w = self.well(name)
        if kind == "3D":
            il, xl = survey.ilxl_at(w.x, w.y)
            if strict and not (survey.inlines[0] - 0.5 * survey.il_step <= il <= survey.inlines[-1] + 0.5 * survey.il_step
                               and survey.xlines[0] - 0.5 * survey.xl_step <= xl <= survey.xlines[-1] + 0.5 * survey.xl_step):
                raise ValueError("outside the survey")
            i, j = survey.il_index(il), survey.xl_index(xl)
            ii = np.clip(np.arange(i - n_around, i + n_around + 1), 0, survey.nil - 1)
            jj = np.clip(np.arange(j - n_around, j + n_around + 1), 0, survey.nxl - 1)
            I, J = np.meshgrid(ii, jj, indexing="ij")
            tr = np.asarray(survey.ilxl[I.ravel(), J.ravel()], np.float32) * survey.inv
            return survey.times, tr
        d = np.hypot(survey.x - w.x, survey.y - w.y)
        k = int(np.argmin(d))
        if strict and (not survey.has_coords or d[k] > max_dist_m):
            raise ValueError(f"{d[k]:.0f} m from the line")
        kk = np.clip(np.arange(k - n_around, k + n_around + 1), 0, len(survey.cdp) - 1)
        return survey.times, survey.data()[kk]


# ====================================================================== synthetics + tie
def ricker(f_hz, dt_ms, length_ms=128.0):
    n = int(length_ms / 2 // dt_ms)
    t = np.arange(-n, n + 1) * dt_ms / 1000.0  # odd length, a sample at t = 0 for any dt
    a = (np.pi * f_hz * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def statistical_wavelet(traces, dt_ms, length_ms=128.0, t_window=None, times=None):
    """Zero-phase wavelet with the average amplitude spectrum of the traces (optionally in a time window)."""
    tr = np.atleast_2d(np.asarray(traces, float))
    if t_window is not None and times is not None:
        m = (times >= t_window[0]) & (times <= t_window[1])
        tr = tr[:, m]
    tr = tr - tr.mean(axis=1, keepdims=True)
    tr = tr * np.hanning(tr.shape[1])[None]
    spec = np.abs(np.fft.rfft(tr, axis=1)).mean(axis=0)
    w = np.fft.irfft(spec, n=tr.shape[1])
    w = np.roll(w, tr.shape[1] // 2)
    n = int(round(length_ms / dt_ms)) // 2
    c = tr.shape[1] // 2
    w = w[c - n:c + n + 1] * np.hanning(2 * n + 1)
    return w / (np.abs(w).max() or 1.0)


def logs_in_time(well, td, dt_ms):
    """Resample AI (RHOB x velocity) onto a regular TWT axis.  Returns (t ms, ai) or None."""
    if not well.logs or well.depth is None:
        return None
    dt_log, rho = well.logs.get("DT"), well.logs.get("RHOB")
    if dt_log is None:
        return None
    md = np.asarray(well.depth, float)
    v = 304800.0 / np.where((dt_log > 30) & (dt_log < 250), dt_log, np.nan)  # m/s
    r = rho if rho is not None else 0.31 * np.power(np.where(np.isfinite(v), v, np.nan), 0.25)  # Gardner fallback
    ai = v * np.where((r > 1.0) & (r < 3.5), r, np.nan)
    ok = np.isfinite(ai)
    if ok.sum() < 50:
        return None
    t = td.twt_raw(well.md_to_tvdss(md))
    i0, i1 = np.nonzero(ok)[0][[0, -1]]
    t, ai, ok = t[i0:i1 + 1], ai[i0:i1 + 1], ok[i0:i1 + 1]
    ai = np.interp(t, t[ok], ai[ok])  # fill gaps
    tt = np.arange(np.ceil(t[0] / dt_ms) * dt_ms, t[-1], dt_ms)
    # average AI inside each output sample (anti-alias) via cumulative trapezoid
    cum = np.r_[0.0, np.cumsum(0.5 * (ai[1:] + ai[:-1]) * np.diff(t))]
    edges = np.r_[tt - dt_ms / 2, tt[-1] + dt_ms / 2]
    ce = np.interp(edges, t, cum)
    ai_t = np.diff(ce) / dt_ms
    return tt, ai_t


def synthetic(t, ai, wavelet):
    """Reflectivity (at sample boundaries, assigned to the lower sample) convolved with the wavelet."""
    r = np.zeros_like(ai)
    r[1:] = (ai[1:] - ai[:-1]) / (ai[1:] + ai[:-1])
    s = np.convolve(r, wavelet, mode="same")
    return r, s


def tie_correlation(t_syn, syn, t_seis, seis, shift, stretch, t_centre):
    """Correlation of the synthetic (after shift/stretch) with the seismic trace on the overlap."""
    tc = t_syn + shift + (stretch - 1.0) * (t_syn - t_centre)
    s = np.interp(tc, t_seis, seis, left=np.nan, right=np.nan)
    ok = np.isfinite(s) & np.isfinite(syn)
    if ok.sum() < 10:
        return -1.0
    a, b = syn[ok] - syn[ok].mean(), s[ok] - s[ok].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else -1.0


def auto_tie(t_syn, syn, t_seis, seis, max_shift_ms=100.0, stretches=None, step_ms=None):
    """Grid search bulk shift (and stretch about the synthetic's centre) maximising correlation.
    Returns dict(shift_ms, stretch, t_centre, corr)."""
    dt = (t_seis[1] - t_seis[0]) if len(t_seis) > 1 else 4.0
    step = step_ms or dt / 4.0
    t_centre = float(np.median(t_syn))
    stretches = np.linspace(0.95, 1.05, 21) if stretches is None else np.asarray(stretches)
    best = (-2.0, 0.0, 1.0)
    for st in stretches:
        for sh in np.arange(-max_shift_ms, max_shift_ms + step / 2, step):
            c = tie_correlation(t_syn, syn, t_seis, seis, sh, st, t_centre)
            if c > best[0]:
                best = (c, sh, st)
    # refine the shift around the best
    c0, sh0, st0 = best
    for sh in np.arange(sh0 - step, sh0 + step, step / 10):
        c = tie_correlation(t_syn, syn, t_seis, seis, sh, st0, t_centre)
        if c > best[0]:
            best = (c, sh, st0)
    return {"shift_ms": float(best[1]), "stretch": float(best[2]), "t_centre": t_centre, "corr": float(best[0])}
