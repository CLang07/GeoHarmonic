"""Horizontal wells (laterals) for the dashboard (PLAN item 14, core): trajectories from deviation surveys
(minimum curvature), synthetic horizontal wells, attributes sampled along MD (map grids, 3D volume via a
time-depth function, logs, stage / completion intervals, zone membership), per-lateral statistics as an
AttributeTable, and drawing helpers (colour bins -> NaN-separated polylines for one batched curve per colour,
level-of-detail decimation).  Pure numpy; GUI-free."""
import csv
import math

import numpy as np


# ====================================================================== trajectory
def min_curvature(md, inc, azi):
    """Minimum curvature: (dz down, dN, dE) relative to the first station (same length as md)."""
    md, inc, azi = (np.asarray(a, float) for a in (md, inc, azi))
    i1, i2 = np.radians(inc[:-1]), np.radians(inc[1:])
    a1, a2 = np.radians(azi[:-1]), np.radians(azi[1:])
    dmd = np.diff(md)
    cosdl = np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1 - np.cos(a2 - a1))
    dl = np.arccos(np.clip(cosdl, -1, 1))
    rf = np.where(dl > 1e-9, 2 / np.where(dl > 1e-9, dl, 1) * np.tan(dl / 2), 1.0)
    dn = dmd / 2 * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * rf
    de = dmd / 2 * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf
    dv = dmd / 2 * (np.cos(i1) + np.cos(i2)) * rf
    c = lambda a: np.concatenate([[0.0], np.cumsum(a)])  # noqa: E731
    return c(dv), c(dn), c(de)


class Trajectory:
    """Well path: md (m), x, y (map), z (TVDSS, m, positive down), inc (deg) at each station."""

    def __init__(self, name, md, x, y, z, inc=None, api=None):
        self.name, self.api = name, api
        self.md = np.asarray(md, float)
        self.x, self.y, self.z = (np.asarray(a, float) for a in (x, y, z))
        self.inc = None if inc is None else np.asarray(inc, float)

    @classmethod
    def from_survey(cls, name, md, inc, azi, x0, y0, kb=0.0, api=None):
        """kb: elevation of the MD reference above sea level (m); first station assumed at md[0] vertically
        below the reference."""
        dz, dn, de = min_curvature(md, inc, azi)
        z = np.asarray(md, float)[0] + dz - kb
        return cls(name, md, x0 + de, y0 + dn, z, inc, api)

    def resample(self, step=10.0):
        md = np.arange(self.md[0], self.md[-1] + step / 2, step)
        md[-1] = min(md[-1], self.md[-1])
        f = lambda a: None if a is None else np.interp(md, self.md, a)  # noqa: E731
        return Trajectory(self.name, md, f(self.x), f(self.y), f(self.z), f(self.inc), self.api)

    def at_md(self, md):
        md = np.asarray(md, float)
        return np.interp(md, self.md, self.x), np.interp(md, self.md, self.y), np.interp(md, self.md, self.z)

    def lateral_mask(self, min_inc=80.0):
        """Stations in the horizontal part (inclination >= min_inc; from geometry when inc is unknown)."""
        inc = self.inc
        if inc is None:
            dh = np.hypot(np.gradient(self.x), np.gradient(self.y))
            inc = np.degrees(np.arctan2(dh, np.abs(np.gradient(self.z))))
        return inc >= min_inc

    def lateral_length(self, min_inc=80.0):
        m = self.lateral_mask(min_inc)
        if m.sum() < 2:
            return 0.0
        return float(self.md[m][-1] - self.md[m][0])

    def heel_toe(self, min_inc=80.0):
        m = np.nonzero(self.lateral_mask(min_inc))[0]
        if not len(m):
            return None, None
        return (self.x[m[0]], self.y[m[0]]), (self.x[m[-1]], self.y[m[-1]])


def synthetic_survey(kop=1500.0, landing_tvd=2500.0, lateral=2500.0, azimuth=90.0, build_deg_30m=3.0, step=30.0):
    """MD / INC / AZI of a vertical - build (constant dog-leg) - horizontal well landing at landing_tvd (TVD
    below the MD reference) with a lateral of the given length."""
    R = 30.0 / math.radians(build_deg_30m)  # build radius (m)
    kop = max(0.0, landing_tvd - R) if kop is None else min(kop, landing_tvd - R)
    md_land = kop + R * math.pi / 2
    md = np.arange(0.0, md_land + lateral + step / 2, step)
    md = np.unique(np.r_[md, kop, md_land, md_land + lateral])
    md = md[md <= md_land + lateral + 1e-6]
    inc = np.clip((md - kop) / R, 0, math.pi / 2) * 180 / math.pi
    return md, inc, np.full(len(md), azimuth), kop, md_land


def read_survey_csv(path):
    """MD, INC, AZI columns (header names containing md / inc / azi, or the first three numeric columns)."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    head = [h.strip().lower() for h in rows[0]]
    pick = lambda keys: next((i for i, h in enumerate(head) if any(k in h for k in keys)), None)  # noqa: E731
    cols = [pick(["md", "measured"]), pick(["inc"]), pick(["azi", "az"])]
    start = 1
    if None in cols:
        cols, start = [0, 1, 2], (0 if _isnum(rows[0][0]) else 1)
    a = np.array([[float(r[c]) for c in cols] for r in rows[start:] if r and all(_isnum(r[c]) for c in cols)])
    return a[:, 0], a[:, 1], a[:, 2]


def _isnum(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


# ====================================================================== attributes along MD
def sample_grid(traj, grid):
    """Map grid value (bilinear) under every station."""
    return grid.sample(traj.x, traj.y)


def sample_volume(traj, vol, twt_of_z, halfwin=0):
    """Seismic value along the wellbore: nearest trace under each station, sampled at the station's two-way
    time twt_of_z(z TVDSS) (ms) by linear interpolation; halfwin > 0 averages |amplitude| (RMS) over +/- halfwin
    samples.  NaN outside the survey."""
    a, b = vol.ax, vol.ay
    M = np.linalg.inv(np.array([[a[1], a[2]], [b[1], b[2]]]))
    il = M[0, 0] * (traj.x - a[0]) + M[0, 1] * (traj.y - b[0])
    xl = M[1, 0] * (traj.x - a[0]) + M[1, 1] * (traj.y - b[0])
    fi = (il - vol.il0) / vol.il_step
    fj = (xl - vol.xl0) / vol.xl_step
    ii, jj = np.rint(fi).astype(int), np.rint(fj).astype(int)
    inside = (ii >= 0) & (ii < vol.nil) & (jj >= 0) & (jj < vol.nxl)
    t = np.asarray(twt_of_z(traj.z), float)
    k = (t / 1000.0 - vol.t0) / vol.dt
    out = np.full(len(traj.md), np.nan)
    if not inside.any():
        return out
    key = ii * vol.nxl + jj
    for u in np.unique(key[inside]):
        sel = inside & (key == u)
        tr = np.asarray(vol.ilxl[u // vol.nxl, u % vol.nxl], np.float32) * vol.inv
        if halfwin > 0:
            w = np.ones(2 * halfwin + 1) / (2 * halfwin + 1)
            tr = np.sqrt(np.convolve(tr.astype(float) ** 2, w, "same"))
        ks = k[sel]
        v = np.interp(ks, np.arange(len(tr)), tr)
        v[(ks < 0) | (ks > len(tr) - 1)] = np.nan
        out[sel] = v
    return out


def sample_log(traj, log_md, log_values):
    """Log curve (on its own MD axis) at the trajectory stations (NaN outside the logged interval)."""
    log_md, v = np.asarray(log_md, float), np.asarray(log_values, float)
    ok = np.isfinite(v)
    out = np.interp(traj.md, log_md[ok], v[ok], left=np.nan, right=np.nan)
    return out


def stage_values(traj, stages):
    """stages: [(md_top, md_base, value)] (e.g. proppant per ft per stage) -> value at each station (NaN
    between stages)."""
    out = np.full(len(traj.md), np.nan)
    for top, base, val in stages:
        lo, hi = min(top, base), max(top, base)
        out[(traj.md >= lo) & (traj.md <= hi)] = val
    return out


def read_stages_csv(path):
    """{well id: [(md_top, md_base, {column: value})]} from a stage CSV with well/api, top/from, base/to columns."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    keys = {k.lower().replace(" ", "").replace("_", ""): k for k in rows[0]}
    find = lambda names: next((keys[n] for n in names if n in keys), None)  # noqa: E731
    kid = find(["api", "uwi", "well", "wellname", "api14"])
    ktop = find(["top", "mdtop", "from", "topmd", "stagetop", "topmdft", "topmdm"])
    kbase = find(["base", "bottom", "mdbase", "to", "basemd", "stagebottom", "btmmd", "bottommd"])
    if not (kid and ktop and kbase):
        raise ValueError("stage CSV needs well/API, top and base MD columns")
    out = {}
    for r in rows:
        vals = {}
        for k, v in r.items():
            if k in (kid, ktop, kbase):
                continue
            try:
                vals[k] = float(v)
            except (TypeError, ValueError):
                pass
        out.setdefault(r[kid], []).append((float(r[ktop]), float(r[kbase]), vals))
    return out


def zone_along(traj, top_depth_grid, base_depth_grid=None, thickness=None):
    """1 where the station lies between the zone top and base depth surfaces (TVDSS grids, m), else 0
    (NaN where the top is undefined).  Base = top + thickness when no base grid is given."""
    zt = top_depth_grid.sample(traj.x, traj.y)
    zb = base_depth_grid.sample(traj.x, traj.y) if base_depth_grid is not None else zt + (thickness or 0.0)
    out = ((traj.z >= zt) & (traj.z <= zb)).astype(float)
    out[~np.isfinite(zt) | ~np.isfinite(zb)] = np.nan
    return out


def distance_to_surface(traj, depth_grid):
    """Signed distance (m) of each station below (+) / above (-) a depth surface (TVDSS grid)."""
    return traj.z - depth_grid.sample(traj.x, traj.y)


# ====================================================================== statistics
def lateral_stats(traj, values, in_zone=None, min_inc=80.0):
    """Statistics of an attribute over the horizontal part: mean, P10, P50, P90 (P10 = low value), std, fraction
    of finite samples, % in zone (when given), lateral length, landing TVDSS, toe-up (m, + = toe shallower)."""
    m = traj.lateral_mask(min_inc)
    v = np.asarray(values, float)[m]
    f = v[np.isfinite(v)]
    out = {"lateral_m": traj.lateral_length(min_inc),
           "landing_tvdss": float(traj.z[m][0]) if m.any() else np.nan,
           "toe_up_m": float(traj.z[m][0] - traj.z[m][-1]) if m.any() else np.nan,
           "coverage": float(len(f) / max(1, m.sum()))}
    if len(f):
        out.update(mean=float(f.mean()), p10=float(np.percentile(f, 10)), p50=float(np.percentile(f, 50)),
                   p90=float(np.percentile(f, 90)), std=float(f.std()))
    else:
        out.update(mean=np.nan, p10=np.nan, p50=np.nan, p90=np.nan, std=np.nan)
    if in_zone is not None:
        z = np.asarray(in_zone, float)[m]
        z = z[np.isfinite(z)]
        out["pct_in_zone"] = float(100 * z.mean()) if len(z) else np.nan
    return out


def stats_table(trajs, attributes, zones=None, min_inc=80.0):
    """AttributeTable (rows = wells, ids = API when every well has a unique one, else well names) with '<attr> mean/p10/p50/p90' columns per attribute,
    lateral length (m, ft), landing depth, toe-up and '% in zone'.
    attributes: {attr name: {well name: values along that well's stations}}; zones: {well name: in-zone array}."""
    from .analytics.table import AttributeTable
    apis = [str(t.api or "").strip() for t in trajs]
    use_api = all(apis) and len(set(apis)) == len(apis)  # placeholder / duplicate UWIs -> well names
    ids = apis if use_api else [t.name for t in trajs]
    tab = AttributeTable(ids)
    cols = {}

    def put(k, i, v):
        cols.setdefault(k, np.full(len(trajs), np.nan))[i] = v

    for i, t in enumerate(trajs):
        z = zones.get(t.name) if zones else None
        base = lateral_stats(t, np.zeros(len(t.md)), z, min_inc)
        put("lateral_m", i, base["lateral_m"])
        put("lateral_ft", i, base["lateral_m"] * 3.28084)
        put("landing_tvdss", i, base["landing_tvdss"])
        put("toe_up_m", i, base["toe_up_m"])
        if z is not None:
            put("pct_in_zone", i, base["pct_in_zone"])
        for a, by in attributes.items():
            if t.name not in by:
                continue
            s = lateral_stats(t, by[t.name], None, min_inc)
            for k in ("mean", "p10", "p50", "p90"):
                put(f"{a} {k}", i, s[k])
    for k, v in cols.items():
        tab.add(k, v)
    tab.xy = np.array([[t.x[0], t.y[0]] for t in trajs]) if trajs else None
    tab.meta["wells"] = [t.name for t in trajs]
    return tab


# ====================================================================== drawing helpers
def color_bins(x, y, values, vmin=None, vmax=None, n_bins=16):
    """Segment-by-segment colouring for fast drawing: each segment (station i -> i+1) gets the bin of the mean of
    its two end values.  Returns [(bin index, xs, ys)] where xs/ys hold the bin's segments separated by NaN (one
    batched curve per colour; pyqtgraph connect='finite').  Segments with NaN values are dropped."""
    x, y, v = (np.asarray(a, float) for a in (x, y, values))
    sv = 0.5 * (v[:-1] + v[1:])
    ok = np.isfinite(sv)
    if not ok.any():
        return []
    vmin = np.nanmin(sv) if vmin is None else vmin
    vmax = np.nanmax(sv) if vmax is None else vmax
    span = (vmax - vmin) or 1.0
    b = np.clip(((np.where(ok, sv, vmin) - vmin) / span * n_bins).astype(int), 0, n_bins - 1)
    out = []
    for k in np.unique(b[ok]):
        idx = np.nonzero(ok & (b == k))[0]
        xs = np.full(3 * len(idx), np.nan)
        ys = np.full(3 * len(idx), np.nan)
        xs[0::3], xs[1::3] = x[idx], x[idx + 1]
        ys[0::3], ys[1::3] = y[idx], y[idx + 1]
        out.append((int(k), xs, ys))
    return out


def batch_color_bins(trajs, values_by_well, vmin, vmax, n_bins=16, pixel=None):
    """color_bins over many wells merged into one (xs, ys) per bin - n_bins draw calls for any number of wells.
    pixel: map units per screen pixel for level-of-detail decimation (None = full resolution)."""
    acc = {}
    for t in trajs:
        v = values_by_well.get(t.name)
        if v is None:
            continue
        idx = lod_indices(t.x, t.y, pixel) if pixel else np.arange(len(t.md))
        for k, xs, ys in color_bins(t.x[idx], t.y[idx], np.asarray(v)[idx], vmin, vmax, n_bins):
            acc.setdefault(k, ([], []))
            acc[k][0].append(xs)
            acc[k][1].append(ys)
    return {k: (np.concatenate(a), np.concatenate(b)) for k, (a, b) in acc.items()}


def lod_indices(x, y, pixel):
    """Keep a station only when it is at least `pixel` map units (along the path) from the last kept one; the
    end points are always kept."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or not pixel:
        return np.arange(len(x))
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    cell = np.floor(s / pixel).astype(np.int64)
    keep = np.r_[True, cell[1:] != cell[:-1]]
    keep[-1] = True
    return np.nonzero(keep)[0]


def colormap_rgb(n_bins=16, name="viridis"):
    """(n_bins, 3) uint8 colours for the bins (matplotlib colormap if available, else a blue-red ramp)."""
    try:
        from matplotlib import colormaps
        cm = colormaps[name]
        return (np.array([cm((k + 0.5) / n_bins)[:3] for k in range(n_bins)]) * 255).astype(np.uint8)
    except Exception:
        t = (np.arange(n_bins) + 0.5) / n_bins
        return (np.column_stack([t, 0.3 + 0.0 * t, 1 - t]) * 255).astype(np.uint8)


# ====================================================================== synthetic test data
def synthetic_laterals(n=6, x0=0.0, y0=0.0, spacing=300.0, landing_tvdss=2000.0, lateral=2000.0, azimuth=90.0,
                       kb=30.0, seed=0, dip_per_m=0.0):
    """n parallel horizontal wells (surface locations along y every `spacing`), landing near landing_tvdss with
    small random landing errors.  dip_per_m tilts the lateral (inc = 90 - atan(dip)).  Returns [Trajectory]."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        land = landing_tvdss + kb + rng.normal(0, 5.0)
        md, inc, azi, kop, md_land = synthetic_survey(None, land, lateral, azimuth)
        if dip_per_m:
            inc = np.where(md > md_land, 90.0 - np.degrees(np.arctan(dip_per_m)), inc)
        out.append(Trajectory.from_survey(f"H-{i + 1}", md, inc, azi, x0, y0 + i * spacing, kb,
                                          api=f"42-000-{i + 1:05d}"))
    return out


# ====================================================================== wells of the linked well project
def trajectories_from_wellset(ws, step=10.0):
    """[Trajectory] for every well of a WellSet with a location: deviation survey (Well Correlation project
    survey file, minimum curvature) when there is one, otherwise vertical to the deepest log / TD.  MD reference =
    the well's elevation (KB)."""
    from .welltie import wc
    geo = wc().geo
    out = []
    for n in ws.names:
        w = ws.well(n)
        if w.x is None or w.y is None:
            continue
        kb = float(w.elev or 0.0)
        s = None
        if getattr(w, "survey_path", None):
            try:
                s = geo.read_survey(w.survey_path)
            except Exception:
                s = None
        if s is not None:
            md, inc, azi = s
            t = Trajectory.from_survey(n, md, inc, azi, float(w.x), float(w.y), kb, api=getattr(w, "uwi", None) or None)
        else:
            td = float(w.td or 0.0) or (float(w.depth[-1]) if getattr(w, "depth", None) is not None and len(w.depth)
                                        else 0.0)
            if td <= 0:
                continue
            t = Trajectory(n, [0.0, td], [w.x, w.x], [w.y, w.y], [-kb, td - kb], [0.0, 0.0],
                           api=getattr(w, "uwi", None) or None)
        out.append(t.resample(step) if len(t.md) > 1 else t)
    return out


# ====================================================================== well strip / geosteering
def wellbore_section(traj, vol, step=None):
    """Seismic along the wellbore: nearest trace under each station (optionally resampled every `step` m of MD).
    Returns (data (n, ns) with NaN rows outside the survey, md (n,), times (ns,) ms)."""
    t = traj.resample(step) if step else traj
    a, b = vol.ax, vol.ay
    M = np.linalg.inv(np.array([[a[1], a[2]], [b[1], b[2]]]))
    il = M[0, 0] * (t.x - a[0]) + M[0, 1] * (t.y - b[0])
    xl = M[1, 0] * (t.x - a[0]) + M[1, 1] * (t.y - b[0])
    ii = np.rint((il - vol.il0) / vol.il_step).astype(int)
    jj = np.rint((xl - vol.xl0) / vol.xl_step).astype(int)
    inside = (ii >= 0) & (ii < vol.nil) & (jj >= 0) & (jj < vol.nxl)
    out = np.full((len(t.md), vol.ns), np.nan, np.float32)
    if inside.any():
        out[inside] = np.asarray(vol.ilxl[ii[inside], jj[inside]], np.float32) * vol.inv
    return out, t.md, vol.times


def strip_data(traj, logs=None, top=None, base=None, vol=None, twt_of_z=None, step=None):
    """Everything a well strip / geosteering view draws, along MD:
    md, z (TVDSS), logs {name: values at md}, top / base depth of the zone under each station (from depth grids),
    distance above top (+ = well below top), in-zone flag, and with a volume + time-depth: the seismic section along
    the wellbore (data, times) and the well's two-way time path.  logs: {name: (log md, values)}."""
    t = traj.resample(step) if step else traj
    out = {"md": t.md, "z": t.z, "logs": {}}
    for n, (lmd, v) in (logs or {}).items():
        out["logs"][n] = sample_log(t, lmd, v)
    if top is not None:
        out["top"] = top.sample(t.x, t.y)
        out["below_top"] = t.z - out["top"]
    if base is not None:
        out["base"] = base.sample(t.x, t.y)
    if top is not None and base is not None:
        out["in_zone"] = zone_along(t, top, base)
    if vol is not None:
        data, _, times = wellbore_section(t, vol)
        out["section"], out["times"] = data, times
        if twt_of_z is not None:
            out["twt"] = np.asarray(twt_of_z(t.z), float)
    return out
