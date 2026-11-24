"""Synthetics as PaleoScan-style seeds (PLAN 24d): tied-well events (tops in TWT or strong events of the tied
synthetic) become small snapped horizon patches around each well.  The patches are ordinary horizons
(project.horizons, key "3D:<volume>") - so the horizon tracker can grow them (T) - and RGT control points: pass them
as `horizons=` to rgt.solve.compute_rgt_3d (NodeSystem.add_horizon ties every node inside a patch to one geologic
time), so the RGT is guided at the wells, also across faults.

    ev = synthetic_events(t_ms, syn, min_rel=0.5)                    # [(t, sign)] strong peaks / troughs
    hs = seed_horizons(vol, {"Top C": [(x, y, t_ms), ...]}, radius=6) # [SeedHorizon] snapped patches
    hs = well_seeds(project, "Vol", source="tops")                   # from the linked well project's tops
"""
import numpy as np


class SeedHorizon:
    """Horizon-like object (name + data {key: grid}) accepted by the RGT solver and the horizon store."""

    def __init__(self, name, key, grid):
        self.name = name
        self.data = {key: grid}

    def get(self, key):
        return self.data.get(key)


def synthetic_events(t_ms, syn, min_rel=0.5, min_sep_ms=12.0, max_n=None):
    """Strong extremes of a synthetic: local maxima of |syn| at least min_rel x max, min_sep_ms apart, with the
    sign of the event (+1 peak, -1 trough).  Returns [(t_ms, sign, rel amplitude)] sorted by time."""
    t = np.asarray(t_ms, float)
    s = np.nan_to_num(np.asarray(syn, float))
    a = np.abs(s)
    if not a.any():
        return []
    mx = a.max()
    k = np.nonzero((a[1:-1] >= a[:-2]) & (a[1:-1] > a[2:]) & (a[1:-1] >= min_rel * mx))[0] + 1
    out = []
    for i in sorted(k, key=lambda i: -a[i]):
        if all(abs(t[i] - o[0]) >= min_sep_ms for o in out):
            out.append((float(t[i]), int(np.sign(s[i])), float(a[i] / mx)))
            if max_n and len(out) >= max_n:
                break
    return sorted(out)


def snap(trace, times, t_ms, sign=0, snap_ms=8.0):
    """Time of the nearest extreme of the given sign (+1 peak, -1 trough, 0 either) within +-snap_ms, parabolic
    sub-sample; NaN if none."""
    tr = np.asarray(trace, float)
    times = np.asarray(times, float)
    dt = times[1] - times[0]
    i0 = int(round((t_ms - times[0]) / dt))
    h = max(1, int(round(snap_ms / dt)))
    a, b = max(1, i0 - h), min(len(tr) - 1, i0 + h + 1)
    if b - a < 1:
        return np.nan
    seg = tr[a - 1:b + 1]
    v = seg if sign > 0 else (-seg if sign < 0 else np.abs(seg))
    loc = np.nonzero((v[1:-1] >= v[:-2]) & (v[1:-1] > v[2:]))[0]
    if not len(loc):
        return np.nan
    best = loc[np.argmin(np.abs(loc + a - i0))]
    y0, y1, y2 = v[best], v[best + 1], v[best + 2]
    den = y0 - 2 * y1 + y2
    frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return float(times[0] + (a + best + float(np.clip(frac, -0.5, 0.5))) * dt)


def seed_patch(vol, il, xl, t_ms, radius=6, sign=0, snap_ms=8.0, follow=True):
    """Grid (nil, nxl) NaN except a snapped patch of half-width `radius` traces around (il, xl).  follow: each ring
    snaps around the mean of the already-snapped inner neighbours (follows dip) instead of the well time."""
    g = np.full((vol.nil, vol.nxl), np.nan, np.float32)
    i0, j0 = vol.il_index(il), vol.xl_index(xl)
    times = np.asarray(vol.times, float)
    c = snap(np.asarray(vol.ilxl[i0, j0], np.float32) * vol.inv, times, t_ms, sign, snap_ms)
    if not np.isfinite(c):
        return g
    if sign == 0:  # keep the polarity found at the well for the whole patch
        k = int(round((c - times[0]) / (times[1] - times[0])))
        sign = int(np.sign(np.asarray(vol.ilxl[i0, j0, k], np.float32))) or 1
    g[i0, j0] = c
    for r in range(1, radius + 1):
        for i in range(i0 - r, i0 + r + 1):
            for j in range(j0 - r, j0 + r + 1):
                if max(abs(i - i0), abs(j - j0)) != r or not (0 <= i < vol.nil and 0 <= j < vol.nxl):
                    continue
                if follow:
                    nb = g[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
                    ref = float(np.nanmean(nb)) if np.isfinite(nb).any() else np.nan
                else:
                    ref = c
                if np.isfinite(ref):
                    g[i, j] = snap(np.asarray(vol.ilxl[i, j], np.float32) * vol.inv, times, ref, sign, snap_ms / 2 if follow else snap_ms)
    return g


def seed_horizons(vol, events, vol_name, radius=6, snap_ms=8.0, prefix="seed: "):
    """events {name: [(x, y, t_ms[, sign])]} (one entry per well) -> [SeedHorizon] with all wells' patches merged."""
    out = []
    key = f"3D:{vol_name}"
    for name, picks in events.items():
        g = np.full((vol.nil, vol.nxl), np.nan, np.float32)
        for p in picks:
            x, y, t = p[:3]
            sign = p[3] if len(p) > 3 else 0
            il, xl = vol.ilxl_at(x, y)
            if not (vol.inlines.min() <= il <= vol.inlines.max() and vol.xlines.min() <= xl <= vol.xlines.max()):
                continue
            il = vol.inlines[np.argmin(np.abs(vol.inlines - il))]
            xl = vol.xlines[np.argmin(np.abs(vol.xlines - xl))]
            patch = seed_patch(vol, il, xl, t, radius, sign, snap_ms)
            g = np.where(np.isfinite(patch), patch, g)
        if np.isfinite(g).any():
            out.append(SeedHorizon(prefix + name, key, g))
    return out


def well_seeds(project, vol_name, wells=None, source="tops", radius=6, snap_ms=8.0, wavelet=None, min_rel=0.5,
               add_to_project=False, max_events=5):
    """Seeds from the linked well project: source "tops" (formation tops in tied TWT, one horizon per formation) or
    "synthetic" (the max_events strongest events of each well's synthetic, wavelet required; horizons named by well +
    time).
    add_to_project: also store them as project horizons (so they show and can be tracked)."""
    ws = project.wellset()
    vol = project.volume(vol_name)
    events = {}
    if source == "tops":
        for n in wells or ws.names:
            try:
                w = ws.well(n)
                for fm, t in ws.tops_twt(n).items():
                    events.setdefault(fm, []).append((float(w.x), float(w.y), float(t)))
            except Exception:
                continue
    else:
        from .inversion.lfm import well_curves
        from .welltie import synthetic
        if wavelet is None:
            raise ValueError("synthetic seeds need a wavelet")
        for c in well_curves(project, vol_name, wells):
            ai = c.values["ai"]
            ok = np.isfinite(ai)
            if ok.sum() < 20:
                continue
            _, syn = synthetic(c.t[ok], ai[ok], wavelet)
            h = len(wavelet) // 2 + 1  # log ends: convolution edge effects, not geology
            if len(syn) <= 2 * h + 3:
                continue
            for t, sg, _ in synthetic_events(c.t[ok][h:-h], syn[h:-h], min_rel, max_n=max_events):
                events[f"{c.name} {t:.0f} ms"] = [(c.x, c.y, t, sg)]
    hs = seed_horizons(vol, events, vol_name, radius, snap_ms)
    if add_to_project:
        for h in hs:
            hz = project.horizons.get(h.name) or project.horizons.new(h.name)
            hz.data[f"3D:{vol_name}"] = h.data[f"3D:{vol_name}"]
    return hs
