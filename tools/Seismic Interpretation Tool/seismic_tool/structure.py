"""Structural timing, paleo-structure and restoration (PLAN 26).  Pure numpy (+ PIL for GIF export).

26a  Tectonic phases in the age table (entry key "phase": pre-rift / syn-rift / post-rift / inversion / other, for the
     unit BELOW the surface, down to the next older one); fault growth: throw per horizon across a fault and the
     expansion index per interval (hanging-wall / footwall thickness) -> proposed syn-rift interval; phase colours for
     the Wheeler diagram.
26b  Flattening (section or volume traces shifted so a horizon sits at a datum), decompacted paleo-depth of every
     older horizon at the time of a younger one (Athy, basin/burial), animation frames oldest -> youngest and GIF.
26c  2D restoration: block restoration of a fault (hanging-wall traces shifted to match the footwall cut-off of the
     restored horizon, horizon by horizon), vertical-shear unfolding (= flattening to a regional datum), line-length
     and area checks.
"""
import numpy as np

PHASES = ("pre-rift", "syn-rift", "post-rift", "inversion", "other")
PHASE_COLORS = {"pre-rift": (120, 120, 200), "syn-rift": (220, 80, 60), "post-rift": (90, 170, 90),
                "inversion": (230, 170, 40), "other": (160, 160, 160), None: (200, 200, 200)}


# ====================================================================== 26a
def set_phase(table, surface, phase):
    """Tag the unit below `surface` (down to the next older surface) with a tectonic phase."""
    if phase not in PHASES and phase is not None:
        raise ValueError(f"phase must be one of {PHASES}")
    return table.set(surface, phase=phase)


def phase_intervals(table):
    """[(young age, old age, phase, surface name)] for the units of the age table (youngest first)."""
    es = table.entries
    out = []
    for k in range(len(es) - 1):
        out.append((es[k]["age"] + es[k].get("gap", 0.0), es[k + 1]["age"], es[k].get("phase"), es[k]["name"]))
    return out


def phase_of_age(table, age):
    for y, o, ph, _ in phase_intervals(table):
        if y <= age <= o:
            return ph
    return None


def phase_colors_for_axis(table, axis):
    """RGB (n, 3) per age of a Wheeler age axis - for colouring the Wheeler background by phase."""
    return np.array([PHASE_COLORS.get(phase_of_age(table, a), PHASE_COLORS[None]) for a in np.asarray(axis, float)],
                    np.uint8)


def throw_across(grid_hw, grid_fw):
    """Throw (hanging-wall minus footwall time / depth) of each horizon at matched points across a fault: inputs are
    arrays (n_horizons, n_points) sampled just inside each block.  Returns (n_horizons,) median throw."""
    return np.nanmedian(np.asarray(grid_hw, float) - np.asarray(grid_fw, float), axis=1)


def expansion_index(hw, fw):
    """Expansion index per interval between consecutive horizons (shallow -> deep rows, n_horizons x n_points):
    hanging-wall thickness / footwall thickness (> 1 = growth during that interval).  Returns (n_horizons - 1,)."""
    hw, fw = np.asarray(hw, float), np.asarray(fw, float)
    th_h = np.diff(hw, axis=0)
    th_f = np.diff(fw, axis=0)
    return np.nanmedian(th_h / np.where(np.abs(th_f) > 1e-9, th_f, np.nan), axis=1)


def propose_phases(names, ei, syn_min=1.1):
    """From expansion indices per interval (interval k between names[k] above and names[k+1] below) propose phases:
    intervals with EI >= syn_min are syn-rift, everything older pre-rift, younger post-rift.
    Returns {surface name (top of the interval): phase}."""
    ei = np.asarray(ei, float)
    syn = np.nonzero(ei >= syn_min)[0]
    out = {}
    for k in range(len(ei)):
        if not len(syn):
            out[names[k]] = "post-rift"
        elif k < syn.min():
            out[names[k]] = "post-rift"
        elif k > syn.max():
            out[names[k]] = "pre-rift"
        else:
            out[names[k]] = "syn-rift"
    return out


def fault_growth(horizon_grids, fault_mask_fn, offset=3):
    """Throw and expansion index from horizon grids (dict name -> (nil, nxl) time / depth, shallow -> deep order) and a
    fault: fault_mask_fn(i) -> crossline index of the fault on inline i (hanging wall = higher crosslines).  Samples
    `offset` traces either side.  Returns {names, throw (n), ei (n-1), phases}."""
    names = list(horizon_grids)
    g0 = np.asarray(horizon_grids[names[0]])
    hw, fw = [], []
    for n in names:
        g = np.asarray(horizon_grids[n], float)
        h_, f_ = [], []
        for i in range(g0.shape[0]):
            j = int(round(fault_mask_fn(i)))
            if offset <= j < g.shape[1] - offset:
                h_.append(g[i, j + offset])
                f_.append(g[i, j - offset])
        hw.append(h_)
        fw.append(f_)
    th = throw_across(hw, fw)
    ei = expansion_index(hw, fw)
    return {"names": names, "throw": th, "ei": ei, "phases": propose_phases(names[:-1], ei)}


# ====================================================================== 26b
def flatten(section, times, horizon_t, datum_t=None):
    """Shift every trace (section (ns, n)) so the horizon time horizon_t (n,) moves to datum_t (default its mean);
    linear interpolation, NaN outside.  Returns the flattened section."""
    s = np.asarray(section, float)
    times = np.asarray(times, float)
    h = np.asarray(horizon_t, float)
    datum = float(np.nanmean(h)) if datum_t is None else float(datum_t)
    out = np.full(s.shape, np.nan)
    for j in range(s.shape[1]):
        if not np.isfinite(h[j]):
            continue
        src = times + (h[j] - datum)
        out[:, j] = np.interp(src, times, s[:, j], left=np.nan, right=np.nan)
    return out


def paleo_depths(depths, k, phi0=0.55, c=0.0005, water_depth=0.0):
    """Decompacted depths of the horizons at the time the k-th horizon was the surface.  depths: list of present-day
    depth grids (or arrays) shallow -> deep (m); horizons above k are removed and every unit below decompacts
    (Athy phi0 / c per unit or scalar).  Returns a list for horizons k.. (k first = water_depth)."""
    from .basin.burial import solid_thickness, base_for
    d = [np.asarray(x, float) for x in depths]
    ph = phi0 if np.ndim(phi0) else [phi0] * len(d)
    cc = c if np.ndim(c) else [c] * len(d)
    out = [np.full(d[k].shape, float(water_depth))]
    top = out[0]
    for u in range(k, len(d) - 1):
        s = solid_thickness(d[u], d[u + 1], ph[u], cc[u])
        base = base_for(top, s, ph[u], cc[u])
        out.append(base)
        top = base
    return out


def paleo_frames(section, times, horizons, order=None, datum_t=None):
    """Animation frames oldest -> youngest: the section flattened on each horizon in turn (data above the horizon
    blanked - it was not deposited yet).  horizons: {name: t (n,)} shallow -> deep.  Returns [(name, frame)]."""
    names = list(order or horizons)[::-1]  # deepest (oldest) first
    out = []
    for n in names:
        h = np.asarray(horizons[n], float)
        f = flatten(section, times, h, datum_t)
        dat = float(np.nanmean(h)) if datum_t is None else float(datum_t)
        f[np.asarray(times) < dat - 1e-9, :] = np.nan
        out.append((n, f))
    return out


def frames_to_rgba(frames, clip=99.0, lut=None):
    """Grey (or LUT) rgba uint8 frames on a common symmetric clip; NaN = white."""
    allv = np.concatenate([np.abs(f[np.isfinite(f)]) for _, f in frames]) if frames else np.ones(1)
    m = float(np.percentile(allv, clip)) if allv.size else 1.0
    out = []
    for name, f in frames:
        n = np.clip((np.nan_to_num(f) / (m or 1) + 1) / 2, 0, 1)
        if lut is None:
            g = (n * 255).astype(np.uint8)
            rgba = np.dstack([g, g, g, np.full(g.shape, 255, np.uint8)])
        else:
            lut = np.asarray(lut)
            rgba = np.zeros(f.shape + (4,), np.uint8)
            rgba[..., :3] = lut[(n * (len(lut) - 1)).astype(int)][..., :3]
            rgba[..., 3] = 255
        rgba[~np.isfinite(f)] = (255, 255, 255, 255)
        out.append((name, rgba))
    return out


def export_gif(rgba_frames, path, duration_ms=700, scale=2, labels=True):
    """Animated GIF (PIL) of rgba frames [(label, (h, w, 4))]."""
    from PIL import Image, ImageDraw
    ims = []
    for lab, rgba in rgba_frames:
        im = Image.fromarray(np.ascontiguousarray(rgba[..., :3])).convert("RGB")
        if scale != 1:
            im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
        if labels and lab:
            ImageDraw.Draw(im).text((6, 4), str(lab), fill=(200, 0, 0))
        ims.append(im.convert("P", palette=Image.ADAPTIVE))
    if not ims:
        raise ValueError("no frames")
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=duration_ms, loop=0)
    return path


# ====================================================================== 26c
def restore_fault(section, times, horizons, fault_j, restore="all"):
    """Block restoration of a vertical fault at trace index fault_j (scalar or per-horizon dict) on a section
    (ns, n): hanging wall = traces >= fault_j.  For the restored horizon (default: each horizon in turn, shallow ->
    deep, returning one step per horizon) the throw = HW cut-off time - FW cut-off time; the hanging-wall traces are
    shifted up by it.  horizons: {name: t (n,)} shallow -> deep.  Returns [(name, throw, restored section,
    restored horizons)]."""
    s = np.asarray(section, float)
    times = np.asarray(times, float)
    names = list(horizons)
    steps = []
    cur_s = s.copy()
    cur_h = {n: np.asarray(horizons[n], float).copy() for n in names}
    for n in (names if restore == "all" else [restore]):
        j = int(fault_j[n] if isinstance(fault_j, dict) else fault_j)
        h = cur_h[n]
        throw = cutoff_throw(h, j)
        new = cur_s.copy()
        for jj in range(j, s.shape[1]):
            new[:, jj] = np.interp(times + throw, times, cur_s[:, jj], left=np.nan, right=np.nan)
        cur_s = new
        for m in names:
            cur_h[m] = cur_h[m].copy()
            cur_h[m][j:] = cur_h[m][j:] - throw
        steps.append((n, throw, cur_s.copy(), {m: v.copy() for m, v in cur_h.items()}))
    return steps


def cutoff_throw(h, j, n=6):
    """Throw at a fault between traces j-1 and j: hanging-wall and footwall cut-offs extrapolated linearly (n traces
    each side, skipping the trace next to the fault) to the fault position, HW minus FW."""
    h = np.asarray(h, float)
    x = np.arange(len(h))

    def at_fault(sel):
        sel = sel[np.isfinite(h[sel])]
        if len(sel) == 0:
            return np.nan
        if len(sel) == 1:
            return float(h[sel[0]])
        a, b = np.polyfit(x[sel], h[sel], 1)
        return float(a * (j - 0.5) + b)
    fw = np.arange(max(0, j - 1 - n), max(0, j - 1))
    hw = np.arange(min(len(h), j + 1), min(len(h), j + 1 + n))
    return at_fault(hw) - at_fault(fw)


def unfold_vertical_shear(section, times, horizon_t, datum_t=None):
    """Vertical-shear unfolding of a horizon to a regional datum (identical to flatten: vertical columns shift)."""
    return flatten(section, times, horizon_t, datum_t)


def line_length(x, z):
    """Length of a horizon polyline (x, z in the same units)."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    ok = np.isfinite(x) & np.isfinite(z)
    return float(np.sum(np.hypot(np.diff(x[ok]), np.diff(z[ok]))))


def area_between(x, z_top, z_base):
    """Area between two horizons along x (trapezoid)."""
    x = np.asarray(x, float)
    t = np.asarray(z_base, float) - np.asarray(z_top, float)
    ok = np.isfinite(t)
    return float(np.trapezoid(t[ok], x[ok])) if hasattr(np, "trapezoid") else float(np.trapz(t[ok], x[ok]))
