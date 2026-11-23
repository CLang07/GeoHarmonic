"""Simple, transparent log lithology + between-well colour fills.

Per sample:
  Vsh   = (GR - GR_clean) / (GR_shale - GR_clean), with GR_clean / GR_shale = 5th / 95th percentile
          of that well's GR in a moving 400 m window. This removes old-vs-new GR tool differences,
          GR drift with depth and level shifts between logging runs.
  coal      RHOB < 1.80  (or, without density, DT > 115 us/ft)  and Vsh < 0.7
  carbonate RHOB > 2.60  and Vsh < 0.5  and (PE > 4 if PE exists, else DT < 65 if DT exists)
  sand Vsh < 0.35, silt 0.35-0.60, shale > 0.60

Between wells, depths are matched along the correlated tops (piecewise-linear between
tops both wells share), so colours follow the correlation like Kingdom / GeoGraphix fills.
"""
import numpy as np

CODES = {0: ("Sand", "#f2d64b"), 1: ("Silt", "#c9b97a"), 2: ("Shale", "#8e9a7c"),
         3: ("Coal", "#1e1e1e"), 4: ("Carbonate", "#5b9bd5")}
NO_DATA = (255, 255, 255, 0)


def _rgb(hexcol, a=255):
    h = hexcol.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


PALETTE = np.array([_rgb(CODES[i][1]) for i in range(5)] + [NO_DATA], dtype=np.uint8)  # index 5 = none


def gr_baselines(depth, gr, window=400.0, step=50.0, pmin=5, pmax=95, min_samples=200):
    """Moving clean-sand / shale GR baselines (5th / 95th percentile in a 400 m window).
    Follows GR drift with depth and level shifts between logging runs."""
    ok = np.isfinite(gr)
    d, g = depth[ok], gr[ok]
    if g.size < min_samples:
        lo, hi = np.nanpercentile(gr, pmin), np.nanpercentile(gr, pmax)
        return np.full(depth.shape, lo), np.full(depth.shape, hi)
    centres, los, his = [], [], []
    for c in np.arange(d.min(), d.max() + step, step):
        sel = (d >= c - window / 2) & (d <= c + window / 2)
        if sel.sum() >= min_samples:
            centres.append(c)
            los.append(np.percentile(g[sel], pmin))
            his.append(np.percentile(g[sel], pmax))
    if not centres:
        lo, hi = np.percentile(g, pmin), np.percentile(g, pmax)
        return np.full(depth.shape, lo), np.full(depth.shape, hi)
    return np.interp(depth, centres, los), np.interp(depth, centres, his)


def well_lithology(well):
    """Return (vsh, code) arrays on the well's MD grid (code 5 = no data). Cached on the well."""
    cached = getattr(well, "_lith", None)
    if cached is not None and len(cached[0]) == len(well.depth):
        return cached
    n = len(well.depth) if well.depth is not None else 0
    vsh = np.full(n, np.nan)
    code = np.full(n, 5, dtype=np.int8)
    gr = well.logs.get("GR")
    if gr is not None and np.isfinite(gr).sum() > 50:
        lo, hi = gr_baselines(well.depth, gr)
        span = hi - lo
        vsh = np.where(span > 5, np.clip((gr - lo) / np.where(span > 5, span, 1), 0, 1), np.nan)
        ok = np.isfinite(vsh)
        code[ok & (vsh < 0.35)] = 0
        code[ok & (vsh >= 0.35) & (vsh < 0.60)] = 1
        code[ok & (vsh >= 0.60)] = 2
        rhob, dt, pe = well.logs.get("RHOB"), well.logs.get("DT"), well.logs.get("PE")
        coal = np.zeros(n, bool)
        if rhob is not None:
            coal |= np.isfinite(rhob) & (rhob < 1.80)
        if dt is not None:
            nod = np.ones(n, bool) if rhob is None else ~np.isfinite(rhob)
            coal |= nod & np.isfinite(dt) & (dt > 115)
        code[ok & coal & (vsh < 0.7)] = 3
        if rhob is not None:
            carb = np.isfinite(rhob) & (rhob > 2.60) & (np.nan_to_num(vsh, nan=1) < 0.5)
            if pe is not None and np.isfinite(pe).sum() > 50:
                carb &= np.nan_to_num(pe) > 4.0
            elif dt is not None and np.isfinite(dt).sum() > 50:
                carb &= np.nan_to_num(dt, nan=99) < 65
            code[ok & carb] = 4
    well._lith = (vsh, code)
    return well._lith


def smoothed_vsh(well, window_m=2.0):
    """Vsh smoothed with a running mean (for the between-well fill)."""
    cached = getattr(well, "_vsh_s", None)
    if cached is not None and len(cached) == len(well.depth):
        return cached
    vsh, _ = well_lithology(well)
    k = max(1, int(round(window_m / 0.1524)))
    ok = np.isfinite(vsh)
    num = np.convolve(np.where(ok, vsh, 0.0), np.ones(k), "same")
    den = np.convolve(ok.astype(float), np.ones(k), "same")
    out = np.where(ok & (den > 0), num / np.maximum(den, 1e-9), np.nan)
    well._vsh_s = out
    return out


def _common_tops(layout, wa, wb):
    """Display-y of tops shared by both wells, in order, as (yA[], yB[], formation[])."""
    ta, tb = layout.project.tops.get(wa.name, {}), layout.project.tops.get(wb.name, {})
    ya, yb, names = [], [], []
    for f in layout.project.formations:
        n = f["name"]
        if n in ta and n in tb:
            a = float(layout.y_of(wa, [ta[n]])[0])
            b = float(layout.y_of(wb, [tb[n]])[0])
            if np.isfinite(a) and np.isfinite(b) and (not ya or (a > ya[-1] and b > yb[-1])):
                ya.append(a)
                yb.append(b)
                names.append(n)
    return np.array(ya), np.array(yb), names


def _map_to_wells(y, t, ya, yb):
    """For display y in a gap column at fraction t, return the matching y in well A and well B."""
    if len(ya) == 0:
        return y, y
    ys = (1 - t) * ya + t * yb  # surfaces in this column
    if len(ys) == 1:
        d = y - ys[0]
        return ya[0] + d, yb[0] + d
    k = np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2)
    y0, y1 = ys[k], ys[k + 1]
    s = (y - y0) / np.where(y1 > y0, y1 - y0, 1)
    above, below = y < ys[0], y > ys[-1]
    yA = ya[k] + s * (ya[k + 1] - ya[k])
    yB = yb[k] + s * (yb[k + 1] - yb[k])
    yA = np.where(above, ya[0] + (y - ys[0]), np.where(below, ya[-1] + (y - ys[-1]), yA))
    yB = np.where(above, yb[0] + (y - ys[0]), np.where(below, yb[-1] + (y - ys[-1]), yB))
    return yA, yB


def _sample(well, layout, ydisp, arr, kind="nearest"):
    """Sample a per-MD array at display depths."""
    y = layout.y_full(well)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.full(ydisp.shape, np.nan if kind != "code" else 5)
    y, a = y[ok], arr[ok]
    i = np.clip(np.searchsorted(y, ydisp), 0, len(y) - 1)
    out = a[i].astype(float)
    outside = (ydisp < y[0]) | (ydisp > y[-1])
    out[outside] = np.nan if kind != "code" else 5
    return out


def track_image(layout, panel, dy=0.5):
    """RGBA image (rows = depth) for a well's lithology column, plus (y0, y1)."""
    w = layout.project.wells[panel.name]
    if w.depth is None or not len(w.depth):
        return None
    vsh, code = well_lithology(w)
    y = layout.y_full(w)
    ok = np.isfinite(y) & (code < 5)
    if ok.sum() < 2:
        return None
    y0, y1 = float(np.nanmin(y[ok])), float(np.nanmax(y[ok]))
    grid = np.arange(y0, y1 + dy, dy)
    c = _sample(w, layout, grid, code, "code").astype(int)
    img = PALETTE[np.clip(c, 0, 5)][:, None, :]
    return img, y0, y1 + dy


def gap_key(layout, pa, pb, mode, ylim):
    """Everything a gap image depends on (for caching)."""
    wa, wb = layout.project.wells[pa.name], layout.project.wells[pb.name]
    ya, yb, names = _common_tops(layout, wa, wb)
    cols = tuple(f.get("color") for f in layout.project.formations) if mode == "zones" else ()
    return (pa.name, pb.name, round(pa.x1, 4), round(pb.x0, 4), mode, layout.hang, tuple(np.round(ya, 2)),
            tuple(np.round(yb, 2)), tuple(names), cols, tuple(np.round(ylim, 1)))


def gap_image(layout, pa, pb, mode, ylim, ncols=40, dy=0.5, alpha=170):
    """RGBA image for the gap between panels pa and pb. mode: 'zones' or 'lithology'."""
    wa, wb = layout.project.wells[pa.name], layout.project.wells[pb.name]
    if wa.depth is None or wb.depth is None or not len(wa.depth) or not len(wb.depth):
        return None
    ya, yb, names = _common_tops(layout, wa, wb)
    y0, y1 = ylim
    grid = np.arange(y0, y1 + dy, dy)
    img = np.zeros((len(grid), ncols, 4), np.uint8)
    if mode == "zones":
        if len(names) == 0:
            return None
        colors = {f["name"]: _rgb(f.get("color", "#999999"), alpha) for f in layout.project.formations}
        for j in range(ncols):
            t = (j + 0.5) / ncols
            ys = (1 - t) * ya + t * yb
            k = np.searchsorted(ys, grid, side="right") - 1  # zone = top above
            for z in range(len(names)):
                sel = k == z
                if z == len(names) - 1:  # last zone: only a short distance below the deepest shared top
                    sel &= grid < ys[-1] + 60
                img[sel, j] = colors[names[z]]
        return img, pa.x1, pb.x0, y0, y1 + dy
    # lithology: interpolate Vsh along the correlation, specials from the nearer well
    _, ca = well_lithology(wa)
    _, cb = well_lithology(wb)
    vsa, vsb = smoothed_vsh(wa), smoothed_vsh(wb)
    for j in range(ncols):
        t = (j + 0.5) / ncols
        yA, yB = _map_to_wells(grid, t, ya, yb)
        va, vb = _sample(wa, layout, yA, vsa), _sample(wb, layout, yB, vsb)
        sa, sb = _sample(wa, layout, yA, ca, "code"), _sample(wb, layout, yB, cb, "code")
        v = np.where(np.isfinite(va) & np.isfinite(vb), (1 - t) * va + t * vb, np.where(np.isfinite(va), va, vb))
        # only fill where both wells have data (no extrapolation past a log's end)
        both = np.isfinite(va) & np.isfinite(vb)
        code = np.full(len(grid), 5)
        code[both & (v < 0.35)] = 0
        code[both & (v >= 0.35) & (v < 0.60)] = 1
        code[both & (v >= 0.60)] = 2
        near = sa if t < 0.5 else sb
        far = sb if t < 0.5 else sa
        for special in (3, 4):  # coal / carbonate from the nearer well, or both
            code[both & ((near == special) | ((far == special) & (np.abs(t - 0.5) < 0.25)))] = special
        col = PALETTE[code].copy()
        col[code < 5, 3] = alpha
        img[:, j] = col
    return img, pa.x1, pb.x0, y0, y1 + dy
