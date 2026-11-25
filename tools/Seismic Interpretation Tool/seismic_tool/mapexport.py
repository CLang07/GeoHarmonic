"""Print-quality map export (PLAN 21f; page layouts are extended in 21g): the layer stack drawn with matplotlib on a page
with a side panel - title block (project, map, contour interval, date, author), legend built from the VISIBLE layers
(industry well symbols, colour bars, pies), scale bar, north arrow (grid convergence when the CRS is transverse Mercator
and a reference lat/lon is known), projected coordinate ticks with the CRS name.  PNG / PDF by extension, 300 dpi.

    export_map(stack, "map.pdf", title="Top C-sands depth", author="Chris", page="A3 landscape")
    apply_preset(stack, "Classic")      # style presets (saved in project.display["map_style"])
"""
import datetime
import math

import numpy as np

PAGES_MM = {"A4": (297, 210), "A3": (420, 297), "A1": (841, 594)}


def page_size_in(page):
    """'A3 landscape' -> (width, height) inches."""
    name, *orient = (page or "A4 landscape").split()
    w, h = PAGES_MM.get(name.upper(), PAGES_MM["A4"])
    if orient and orient[0].lower().startswith("port"):
        w, h = h, w
    return w / 25.4, h / 25.4


def layout_boxes(page="A4 landscape", panel_frac=0.24, margin_in=0.4):
    """Figure-fraction boxes (x0, y0, w, h) of the map frame and side panel parts - no overlaps (tested)."""
    W, H = page_size_in(page)
    mx, my = margin_in / W, margin_in / H
    pw = panel_frac
    gap = 0.012
    mapb = (mx, my, 1 - 2 * mx - pw - gap, 1 - 2 * my)
    px = 1 - mx - pw
    title = (px, 1 - my - 0.2, pw, 0.2)
    locator = (px, 1 - my - 0.2 - gap - 0.2, pw, 0.2)
    legend = (px, my + 0.2, pw, locator[1] - gap - (my + 0.2))
    scale = (px, my + 0.1, pw, 0.09)
    north = (px, my, pw, 0.09)
    return {"map": mapb, "title": title, "locator": locator, "legend": legend, "scale": scale, "north": north}


def save_template(project, name, settings):
    """Named export settings (page, title, author, crs, graticule, locator, preset ...) in display["map_templates"]."""
    project.display.setdefault("map_templates", {})[name] = dict(settings)
    return name


def load_template(project, name):
    return dict((project.display.get("map_templates") or {})[name])


def graticule_lines(c, x0, x1, y0, y1, n_target=5):
    """Lat / long lines (map coordinates) covering the extent: {"lat": [(value, xs, ys)], "lon": [...]}, step in
    degrees from 1/60 to 10 chosen for ~n_target lines."""
    from .gis import crs as CRS
    gx = np.linspace(x0, x1, 9)
    gy = np.linspace(y0, y1, 9)
    GX, GY = np.meshgrid(gx, gy)
    la, lo = CRS.inverse(c, GX, GY)
    lat_lo, lat_hi, lon_lo, lon_hi = float(la.min()), float(la.max()), float(lo.min()), float(lo.max())
    span = max(lat_hi - lat_lo, lon_hi - lon_lo)
    steps = [1 / 60, 2 / 60, 5 / 60, 10 / 60, 15 / 60, 0.5, 1.0, 2.0, 5.0, 10.0]
    step = next((s_ for s_ in steps if span / s_ <= n_target), 10.0)
    out = {"lat": [], "lon": [], "step": step}
    for v in np.arange(np.ceil(lat_lo / step) * step, lat_hi, step):
        lon = np.linspace(lon_lo, lon_hi, 60)
        x, y = CRS.forward(c, np.full(lon.shape, v), lon)
        out["lat"].append((float(v), x, y))
    for v in np.arange(np.ceil(lon_lo / step) * step, lon_hi, step):
        lat = np.linspace(lat_lo, lat_hi, 60)
        x, y = CRS.forward(c, lat, np.full(lat.shape, v))
        out["lon"].append((float(v), x, y))
    return out


def _dms(v, pos, neg):
    h = pos if v >= 0 else neg
    v = abs(v)
    d = int(v)
    m = int(round((v - d) * 60))
    if m == 60:
        d, m = d + 1, 0
    return f"{d}\u00b0{m:02d}'{h}"


def boxes_overlap(boxes):
    b = list(boxes)
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            x0, y0, w0, h0 = b[i]
            x1, y1, w1, h1 = b[j]
            if x0 < x1 + w1 - 1e-9 and x1 < x0 + w0 - 1e-9 and y0 < y1 + h1 - 1e-9 and y1 < y0 + h0 - 1e-9:
                return True
    return False


def apply_preset(stack, name):
    """Apply a carto style preset to the grid / contour / well layers and remember it in the project display."""
    from .welltie import wc
    wc()
    from well_correlation.carto import STYLE_PRESETS, default_ramp
    pr = STYLE_PRESETS[name]
    for lay in stack.layers:
        st = lay["style"]
        if lay["kind"] in ("grid", "horizon"):
            g = stack.grid(lay["id"])
            st["ramp"] = pr["ramp"] or (default_ramp(g.unit if g is not None else "", lay["name"])[0])
            st["hillshade"] = pr["hillshade"]
            st["shade_strength"] = pr["shade_strength"]
            st["ve"] = pr["ve"]
            if pr.get("fill") is False:
                lay["visible"] = False
        elif lay["kind"] == "contours":
            st.update(color=pr["contour_color"], index_every=pr["index_every"], labels=pr["labels"],
                      label_size=pr["label_size"])
        elif lay["kind"] == "wells":
            st.update(symbols=True, size=pr["well_size"])
    if stack.project is not None:
        stack.project.display["map_style"] = name
    stack.save()
    return name


def _draw_symbol(ax, x, y, kind, r, transform=None, color="k"):
    from well_correlation.carto import symbol_paths
    sp = symbol_paths(kind)
    for k, (lx, ly) in enumerate(sp["lines"]):
        ax.plot(x + r * lx, y + r * ly, color=color, lw=0.8, ls="--" if (sp["dashed"] and k == 0) else "-", zorder=7)
    cx, cy = sp["lines"][0]
    if sp["fill"] == "full":
        ax.fill(x + r * cx, y + r * cy, color=color, zorder=7)
    elif sp["fill"] == "half":
        h = cy >= 0
        ax.fill(x + r * cx[h], y + r * cy[h], color=color, zorder=7)
    elif sp["fill"] == "dot":
        ax.plot([x], [y], "o", color=color, ms=1.5, zorder=7)


def export_map(stack, path, title="", subtitle="", author="", page="A4 landscape", dpi=300, crs="",
               ref_latlon=None, lon0=None, contour_interval=None, symbols=True, graticule=True, locator=True,
               source="", neatline=True, extra_prims=None, extra_legend=None):
    """Draw the visible layers + furniture to PNG / PDF.  Returns {"path", "legend", "boxes"}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.transforms import Affine2D
    from .welltie import wc
    wc()
    from well_correlation.carto import nice_scale, north_arrow_angle
    from .maplayers import render, well_status
    W, H = page_size_in(page)
    fig = plt.figure(figsize=(W, H), dpi=dpi)
    B = layout_boxes(page)
    ax = fig.add_axes(B["map"])
    bx, by = [], []
    prims = list(render(stack)) + list(extra_prims or [])  # Live Map: its styled wells (31i)
    wells_lids = {l["id"] for l in stack.layers if l["kind"] == "wells" and l["visible"]}
    kinds = well_status(stack.project) if symbols else {}
    for q in prims:
        kind = q[0]
        if kind == "image":
            m11, m12, m21, m22, dx, dy = q[3]
            h, w = q[2].shape[:2]
            for cxp, cyp in ((0, 0), (w, 0), (0, h), (w, h)):
                bx.append(m11 * cxp + m21 * cyp + dx)
                by.append(m12 * cxp + m22 * cyp + dy)
            tr = Affine2D(np.array([[m11, m21, dx], [m12, m22, dy], [0, 0, 1]], float)) + ax.transData
            ax.imshow(q[2], origin="lower", extent=(0, w, 0, h), transform=tr, interpolation="nearest", zorder=1)
        elif kind == "curve":
            xs, ys = np.asarray(q[2]), np.asarray(q[3])
            bx.extend(xs[np.isfinite(xs)].tolist())
            by.extend(ys[np.isfinite(ys)].tolist())
            c = np.array(q[4], float) / 255.0
            if q[6] == "pairs":
                from matplotlib.collections import LineCollection
                n = len(xs) // 2 * 2
                segs = np.stack([np.column_stack([xs[0:n:2], ys[0:n:2]]), np.column_stack([xs[1:n:2], ys[1:n:2]])], 1)
                ax.add_collection(LineCollection(segs, colors=[c], linewidths=q[5] * 0.45, zorder=3))
            else:
                ax.plot(xs, ys, color=c, lw=q[5] * 0.45, zorder=3,
                        ls="--" if len(q) > 7 and (q[7] or {}).get("dash") else "-")
        elif kind == "polygon":
            hatch = len(q) > 6 and (q[6] or {}).get("hatch")
            fc = np.array(q[4], float) / 255.0
            ax.fill(q[2], q[3], facecolor="none" if hatch else fc, hatch="////" if hatch else None,
                    edgecolor=fc if hatch else "none", lw=0.6, zorder=4)
        elif kind == "points":
            x, y = np.asarray(q[2], float), np.asarray(q[3], float)
            bx.extend(x.tolist())
            by.extend(y.tolist())
            if q[1] in wells_lids and symbols:
                continue  # drawn below with the exact industry symbols
            cols = np.array(q[4], float) / 255.0
            ax.scatter(x, y, s=np.asarray(q[5], float) ** 2 * 0.35, c=cols if cols.ndim == 2 else [cols],
                       edgecolors="w", linewidths=0.4, zorder=6)
        elif kind == "text":
            ax.text(q[2], q[3], q[4], color=np.array(q[5], float) / 255.0, fontsize=q[6] * 0.7, zorder=8,
                    rotation=q[7] if len(q) > 7 else 0, ha="center" if len(q) > 7 else "left",
                    va="center" if len(q) > 7 else "bottom", rotation_mode="anchor",
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    if not bx and stack.project is not None:
        for _, x, y in stack.project.wells():
            bx.append(x)
            by.append(y)
    x0, x1 = (min(bx), max(bx)) if bx else (0, 1)
    y0, y1 = (min(by), max(by)) if by else (0, 1)
    pad = 0.04 * max(x1 - x0, y1 - y0, 1.0)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=6)
    for lab in ax.get_yticklabels():
        lab.set_rotation(90)
        lab.set_va("center")
    pcrs = None
    try:
        from .gis.export import project_crs
        pcrs = project_crs(stack.project) if stack.project is not None else None
    except Exception:
        pcrs = None
    crs = crs or (pcrs["name"] if pcrs else "")
    if crs:
        ax.set_xlabel(f"Easting (m) - {crs}", fontsize=6)
        ax.set_ylabel("Northing (m)", fontsize=6)
    if graticule and pcrs is not None and pcrs["kind"] != "geographic":
        xl, yl = ax.get_xlim(), ax.get_ylim()
        gr = graticule_lines(pcrs, xl[0], xl[1], yl[0], yl[1])
        for v, gx_, gy_ in gr["lat"]:
            ax.plot(gx_, gy_, color=(0.35, 0.35, 0.35), lw=0.4, ls=":", zorder=2)
            ax.text(xl[0], np.interp(xl[0], gx_, gy_), _dms(v, "N", "S"), fontsize=5, ha="right", va="center",
                    color=(0.3, 0.3, 0.3), clip_on=False)
        for v, gx_, gy_ in gr["lon"]:
            ax.plot(gx_, gy_, color=(0.35, 0.35, 0.35), lw=0.4, ls=":", zorder=2)
            o = np.argsort(gy_)
            ax.text(np.interp(yl[1], gy_[o], gx_[o]), yl[1], _dms(v, "E", "W"), fontsize=5, ha="center", va="bottom",
                    color=(0.3, 0.3, 0.3), clip_on=False)
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        if ref_latlon is None and pcrs["kind"] == "tm":
            from .gis import crs as CRS
            la_, lo_ = CRS.inverse(pcrs, 0.5 * (xl[0] + xl[1]), 0.5 * (yl[0] + yl[1]))
            ref_latlon, lon0 = (float(la_), float(lo_)), pcrs["lon0"]
    # wells with industry symbols + names
    if symbols and wells_lids and stack.project is not None:
        xl = ax.get_xlim()
        r = 0.006 * (xl[1] - xl[0])
        for name, x, y in stack.project.wells():
            if not (x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad):
                continue
            _draw_symbol(ax, x, y, kinds.get(name, "unknown"), r)
            ax.text(x + 1.6 * r, y + 1.6 * r, name, fontsize=5.5, zorder=8,
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    # ---- side panel
    tb = fig.add_axes(B["title"])
    tb.axis("off")
    tb.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1.0, transform=tb.transAxes))
    proj = stack.project.name if stack.project is not None else ""
    lines = [(title or "Map", 11, "bold"), (subtitle, 8, "normal"), (f"Project: {proj}", 7, "normal")]
    if contour_interval:
        lines.append((f"Contour interval: {contour_interval:g}", 7, "normal"))
    lines += [(f"Date: {datetime.date.today().isoformat()}", 7, "normal"), (f"Author: {author}" if author else "", 7, "normal"),
              (crs, 6, "normal")]
    yy = 0.9
    for text, size, weight in lines:
        if text:
            tb.text(0.05, yy, text, fontsize=size, weight=weight, va="top", transform=tb.transAxes, wrap=True)
            yy -= 0.12 if size >= 8 else 0.1
    if source:
        tb.text(0.05, 0.04, f"Data: {source}", fontsize=5, va="bottom", transform=tb.transAxes)
    if locator:
        la = fig.add_axes(B["locator"])
        la.set_xticks([])
        la.set_yticks([])
        ex, ey = [], []
        if stack.project is not None:
            for _, x, y in stack.project.wells():
                ex.append(x)
                ey.append(y)
            for v in stack.project.volumes:
                try:
                    vx, vy = stack.project.volume(v["name"]).outline()
                    la.plot(vx, vy, color=(0.4, 0.4, 0.4), lw=0.6)
                    ex += list(vx)
                    ey += list(vy)
                except Exception:
                    pass
        if ex:
            la.plot(ex, ey, ".", color=(0.3, 0.3, 0.3), ms=1.5)
        xl, yl = ax.get_xlim(), ax.get_ylim()
        la.add_patch(plt.Rectangle((xl[0], yl[0]), xl[1] - xl[0], yl[1] - yl[0], fill=False, ec="red", lw=1.2))
        allx = ex + [xl[0], xl[1]]
        ally = ey + [yl[0], yl[1]]
        cxl, cyl = 0.5 * (min(allx) + max(allx)), 0.5 * (min(ally) + max(ally))
        half = 0.6 * max(max(allx) - min(allx), max(ally) - min(ally), 1.0) * 1.5
        la.set_xlim(cxl - half, cxl + half)
        la.set_ylim(cyl - half, cyl + half)
        la.set_aspect("equal", adjustable="box")
        la.set_title("Location", fontsize=6)
    if neatline:
        fig.add_artist(plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False, lw=1.2, transform=fig.transFigure))
    lg = fig.add_axes(B["legend"])
    lg.axis("off")
    lg.text(0.02, 0.98, "Legend", fontsize=8, weight="bold", va="top")
    legend = stack.legend()
    yy = 0.9
    used_kinds = sorted(set(kinds.values())) if (symbols and wells_lids) else []
    for e in legend:
        if yy < 0.05:
            break
        if e["symbol"] == "ramp" and e.get("colorbar"):
            from .maplayers import _ramp_or_lut
            if yy < 0.89:
                yy -= 0.03  # the ramp title sits above yy: keep it clear of the entry above
            lay = stack.get(e["id"])
            lut = _ramp_or_lut(lay["style"], e["colorbar"])
            cb = e["colorbar"]
            lg.imshow(lut[None, :, :3], extent=(0.05, 0.75, yy - 0.05, yy - 0.015), aspect="auto", transform=lg.transAxes)
            lg.text(0.05, yy, e["name"], fontsize=6, va="bottom", transform=lg.transAxes)
            lg.text(0.05, yy - 0.065, f"{cb['vmin']:.4g}", fontsize=5, va="top", transform=lg.transAxes)
            lg.text(0.75, yy - 0.065, f"{cb['vmax']:.4g} {cb['unit']}", fontsize=5, va="top", ha="right", transform=lg.transAxes)
            yy -= 0.12
        elif e["symbol"] == "pie":
            lg.text(0.05, yy, e["name"], fontsize=6, va="top", transform=lg.transAxes)
            yy -= 0.045
            for comp, c in e["components"]:
                lg.add_patch(plt.Rectangle((0.07, yy - 0.025), 0.05, 0.025, color=np.array(c) / 255.0, transform=lg.transAxes))
                lg.text(0.15, yy - 0.012, comp, fontsize=5.5, va="center", transform=lg.transAxes)
                yy -= 0.035
        elif e["kind"] == "wells" and used_kinds:
            from well_correlation.carto import WELL_SYMBOLS
            for kd in used_kinds:
                _draw_symbol(lg, 0.1, yy - 0.012, kd, 0.012)
                lg.text(0.18, yy - 0.012, WELL_SYMBOLS[kd]["label"], fontsize=5.5, va="center", transform=lg.transAxes)
                yy -= 0.04
            lg.set_xlim(0, 1)
            lg.set_ylim(0, 1)
        else:
            c = np.array(e.get("color") or (0, 0, 0), float)[:3] / 255.0
            lg.plot([0.05, 0.14], [yy - 0.012, yy - 0.012], color=c, lw=1.5, transform=lg.transAxes)
            lg.text(0.18, yy - 0.012, e["name"], fontsize=5.5, va="center", transform=lg.transAxes)
            yy -= 0.04
    for label, c in (extra_legend or []):  # Live Map well styling (colour legend)
        if yy < 0.05:
            break
        lg.scatter([0.095], [yy - 0.012], s=18, color=np.array(c, float)[:3] / 255.0, transform=lg.transAxes)
        lg.text(0.18, yy - 0.012, str(label), fontsize=5.5, va="center", transform=lg.transAxes)
        yy -= 0.04
    # scale bar (in map units, drawn in the panel at the map's scale)
    sa = fig.add_axes(B["scale"])
    sa.axis("off")
    xl = ax.get_xlim()
    map_w_in = B["map"][2] * W
    sc = nice_scale(xl[1] - xl[0], 0.25)
    frac_of_panel = (sc["length"] / (xl[1] - xl[0])) * map_w_in / (B["scale"][2] * W)
    frac_of_panel = min(frac_of_panel, 0.9)
    n = sc["segments"]
    for k in range(n):
        sa.add_patch(plt.Rectangle((0.05 + k * frac_of_panel / n, 0.45), frac_of_panel / n, 0.15,
                                   facecolor="k" if k % 2 == 0 else "w", edgecolor="k", lw=0.6, transform=sa.transAxes))
    sa.text(0.05, 0.7, "0", fontsize=6, ha="center", transform=sa.transAxes)
    sa.text(0.05 + frac_of_panel, 0.7, sc["label"], fontsize=6, ha="center", transform=sa.transAxes)
    # north arrow
    na = fig.add_axes(B["north"])
    na.axis("off")
    ang = 0.0
    if ref_latlon and lon0 is not None:
        ang = north_arrow_angle(ref_latlon[0], ref_latlon[1], lon0)
    a = math.radians(ang)
    cx, cy, L = 0.5, 0.3, 0.45
    na.annotate("", xy=(cx + L * math.sin(-a) * 0.3, cy + L), xytext=(cx, cy), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="k"))
    na.text(cx, cy + L + 0.05, "N", fontsize=9, weight="bold", ha="center", transform=na.transAxes)
    if ang:
        na.text(cx, 0.05, f"grid convergence {ang:+.2f} deg", fontsize=5, ha="center", transform=na.transAxes)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return {"path": path, "legend": legend, "boxes": B}
