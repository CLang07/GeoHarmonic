"""Live Map logic (PLAN 31b + 31d) - Qt-free, so every rule is unit-tested headless.

The Live Map is the item-21 map hub (maplayers.LayerStack + render) in its own Kingdom-style window
(gui/live_map.py), driven by the map bus (mapbus.py): wells come from the bus WellTable, the Filters panel edits the
bus FilterService, selections go through the SelectionService, the current inline / line comes from ActiveObject.

31b map core: `nice_scale` (scale-bar length), `north_angle` (true north vs grid north from the project CRS),
`cursor_text` (X / Y / lat-lon readout), `MapState` (view, well style, panels; saved in display["live_map"]).
31d well styling (Spotfire-style): `WellStyle` spec dict, `style_wells` -> per-well colour / size / symbol / label /
visibility, `legend_entries`, `pie_fractions`; linked views `histogram_data`, `crossplot_data`, `table_rows`,
`lasso_select`.
"""
import math

import numpy as np

from . import mapbus as MB

# Spotfire-like categorical palette (distinct on dark and light backgrounds)
PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f",
           "#bab0ac", "#1f9e89", "#d4a6c8"]
RAMPS = {
    "viridis": [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)), (0.75, (94, 201, 98)),
                (1.0, (253, 231, 37))],
    "red-yellow-green": [(0.0, (215, 48, 39)), (0.5, (255, 255, 191)), (1.0, (26, 152, 80))],
    "blue-red": [(0.0, (49, 54, 149)), (0.5, (240, 240, 240)), (1.0, (165, 0, 38))],
    "blues": [(0.0, (222, 235, 247)), (1.0, (8, 48, 107))],
    "heat": [(0.0, (255, 255, 178)), (0.5, (253, 141, 60)), (1.0, (189, 0, 38))],
    "spectral": [(0.0, (158, 1, 66)), (0.25, (244, 109, 67)), (0.5, (255, 255, 191)), (0.75, (102, 194, 165)),
                 (1.0, (94, 79, 162))],
}
SYMBOLS = ["o", "s", "t", "d", "star", "+", "x", "p", "h", "t1"]  # pyqtgraph symbol names
NAN_COLOR = (140, 140, 140, 255)


def hex_rgba(h, alpha=255):
    if isinstance(h, (tuple, list)):
        c = tuple(int(v) for v in h)
        return c if len(c) == 4 else c + (alpha,)
    h = str(h).lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), (int(h[6:8], 16) if len(h) >= 8 else alpha)


def rgba_hex(c):
    return "#%02x%02x%02x" % tuple(int(v) for v in c[:3])


def ramp_colors(t, ramp="viridis", reverse=False):
    """t in [0, 1] (NaN -> grey) -> (n, 4) uint8."""
    stops = RAMPS.get(ramp, RAMPS["viridis"])
    pos = np.array([s[0] for s in stops])
    col = np.array([s[1] for s in stops], float)
    t = np.asarray(t, float)
    tt = np.clip(1.0 - t if reverse else t, 0, 1)
    out = np.empty((t.size, 4), np.uint8)
    fin = np.isfinite(tt)
    for k in range(3):
        out[:, k] = np.round(np.interp(np.where(fin, tt, 0.0), pos, col[:, k])).astype(np.uint8)
    out[:, 3] = 255
    out[~np.isfinite(t)] = NAN_COLOR
    return out


# ====================================================================== 31b map core
def nice_scale(width, fraction=0.2):
    """(length in map units, n segments, label) for a scale bar ~fraction of the view width."""
    target = max(float(width) * fraction, 1e-9)
    mag = 10 ** math.floor(math.log10(target))
    best = min((m * mag for m in (1, 2, 2.5, 5, 10)), key=lambda L: abs(L - target))
    segs = 5 if round(best / mag, 6) in (1, 5, 10) else 4
    label = f"{best / 1000:g} km" if best >= 1000 else f"{best:g} m"
    return best, segs, label


def scale_bar_geometry(view, fraction=0.2, margin=0.04):
    """Scale bar in map coordinates for view (x0, x1, y0, y1): {'segments': [(xa, xb, filled)], 'y', 'height',
    'label', 'label_xy'} placed bottom-left."""
    x0, x1, y0, y1 = [float(v) for v in view]
    w, h = x1 - x0, y1 - y0
    L, n, label = nice_scale(w, fraction)
    xa = x0 + margin * w
    yb = y0 + margin * h
    seg = L / n
    segs = [(xa + k * seg, xa + (k + 1) * seg, k % 2 == 0) for k in range(n)]
    return {"segments": segs, "y": yb, "height": 0.012 * h, "label": label, "label_xy": (xa + L, yb), "length": L}


def north_angle(crs, x, y):
    """Degrees from grid north to true north (clockwise positive) at map point x, y; 0 without a CRS."""
    if not crs:
        return 0.0
    from .gis import crs as CRS
    try:
        lat, lon = CRS.inverse(crs, np.array([x], float), np.array([y], float))
        lat, lon = float(np.ravel(lat)[0]), float(np.ravel(lon)[0])
        if not (np.isfinite(lat) and np.isfinite(lon)) or abs(lat) > 89.0:
            return 0.0
        x2, y2 = CRS.forward(crs, np.array([lat + 0.01]), np.array([lon]))
        dx, dy = float(np.ravel(x2)[0]) - x, float(np.ravel(y2)[0]) - y
        return math.degrees(math.atan2(dx, dy))
    except Exception:
        return 0.0


def north_arrow_geometry(view, angle_deg=0.0, size=0.06, margin=0.06):
    """Arrow polygon (xs, ys) + 'N' label position, top-right of the view, rotated by angle_deg (clockwise)."""
    x0, x1, y0, y1 = [float(v) for v in view]
    w, h = x1 - x0, y1 - y0
    s = size * min(w, h)
    cx, cy = x1 - margin * w, y1 - margin * h - s
    a = math.radians(angle_deg)
    pts = np.array([[0, 1.0], [0.35, -0.6], [0, -0.25], [-0.35, -0.6], [0, 1.0]]) * s
    rx = cx + pts[:, 0] * math.cos(a) + pts[:, 1] * math.sin(a)
    ry = cy - pts[:, 0] * math.sin(a) + pts[:, 1] * math.cos(a)
    lx, ly = cx + 1.35 * s * math.sin(a), cy + 1.35 * s * math.cos(a)
    return rx, ry, (lx, ly)


def cursor_text(x, y, crs=None, vol=None, extra=None):
    """'X 612,345  Y 5,987,654 | 39.60123 S 173.30210 E | IL 120  XL 330' (parts left out when unknown)."""
    parts = [f"X {x:,.1f}  Y {y:,.1f}"]
    if crs:
        try:
            from .gis import crs as CRS
            lat, lon = CRS.inverse(crs, np.array([x], float), np.array([y], float))
            lat, lon = float(np.ravel(lat)[0]), float(np.ravel(lon)[0])
            if np.isfinite(lat) and np.isfinite(lon):
                parts.append(f"{abs(lat):.5f} {'S' if lat < 0 else 'N'}  {abs(lon):.5f} {'W' if lon < 0 else 'E'}")
        except Exception:
            pass
    if vol is not None:
        try:
            il, xl = vol.ilxl_at(x, y)
            il, xl = float(np.ravel(il)[0]), float(np.ravel(xl)[0])
            if vol.inlines[0] - vol.il_step <= il <= vol.inlines[-1] + vol.il_step and \
                    vol.xlines[0] - vol.xl_step <= xl <= vol.xlines[-1] + vol.xl_step:
                parts.append(f"IL {il:.0f}  XL {xl:.0f}")
        except Exception:
            pass
    if extra:
        parts.append(str(extra))
    return "  |  ".join(parts)


DEFAULT_STYLE = {
    "color": {"mode": "fixed", "column": None, "fixed": "#4e79a7", "ramp": "viridis", "reverse": False,
              "min": None, "max": None, "categories": {}, "rules": [], "default": "#bab0ac"},
    "size": {"column": None, "fixed": 9.0, "min_px": 6.0, "max_px": 26.0, "min": None, "max": None},
    "shape": {"column": None, "fixed": "o", "map": {}},
    "label": {"column": "Well", "max_labels": 300},
    "pie": {"columns": [], "radius_px": 14.0, "colors": {}},
}


class MapState:
    """Live Map window state in project.display['live_map']: view, well style, filters panel columns, panels."""

    KEY = "live_map"

    def __init__(self, d=None):
        d = dict(d or {})
        self.view = d.get("view")
        self.well_style = merge_style(d.get("well_style"))
        self.filter_columns = list(d.get("filter_columns") or [])
        self.table_columns = list(d.get("table_columns") or [])
        self.crossplot = dict(d.get("crossplot") or {"x": None, "y": None})
        self.histogram = d.get("histogram")
        self.panels = dict(d.get("panels") or {"layers": True, "style": True, "filters": True, "table": True,
                                                "crossplot": True})
        self.theme = d.get("theme", "dark")

    def to_dict(self):
        return {"view": self.view, "well_style": self.well_style, "filter_columns": self.filter_columns,
                "table_columns": self.table_columns, "crossplot": self.crossplot, "histogram": self.histogram,
                "panels": self.panels, "theme": self.theme}

    @classmethod
    def load(cls, project):
        return cls((getattr(project, "display", None) or {}).get(cls.KEY))

    def save(self, project):
        project.display[self.KEY] = self.to_dict()
        return project.display[self.KEY]


def merge_style(style):
    out = {k: dict(v) for k, v in DEFAULT_STYLE.items()}
    for k, v in (style or {}).items():
        if k in out and isinstance(v, dict):
            out[k].update(v)
    return out


# ====================================================================== 31d well styling
def _column(table, name):
    if name in (None, "", "Well"):
        return None
    if name not in table.cols:
        raise KeyError(f"well style: no column {name!r}")
    return name


def color_values(table, cs):
    """(n, 4) uint8 colours + legend entries [(label, rgba)] for the colour spec."""
    n = len(table)
    mode = cs.get("mode", "fixed")
    if mode == "fixed" or not cs.get("column") and mode != "rules":
        c = hex_rgba(cs.get("fixed", "#4e79a7"))
        return np.tile(np.array(c, np.uint8), (n, 1)), [("Wells", c)]
    if mode == "rules":
        out = np.tile(np.array(hex_rgba(cs.get("default", "#bab0ac")), np.uint8), (n, 1))
        taken = np.zeros(n, bool)
        leg = []
        for r in cs.get("rules") or []:  # first matching rule wins (like Spotfire's rule order)
            m = MB.eval_rule(r["rule"], table) & ~taken
            c = hex_rgba(r["color"])
            out[m] = c
            taken |= m
            leg.append((r.get("label") or r["rule"], c))
        leg.append(("other", hex_rgba(cs.get("default", "#bab0ac"))))
        return out, leg
    col = _column(table, cs.get("column"))
    if mode == "continuous":
        v = table.numeric(col)
        fin = v[np.isfinite(v)]
        lo = cs.get("min") if cs.get("min") is not None else (float(fin.min()) if fin.size else 0.0)
        hi = cs.get("max") if cs.get("max") is not None else (float(fin.max()) if fin.size else 1.0)
        t = (v - lo) / (hi - lo) if hi > lo else np.where(np.isfinite(v), 0.5, np.nan)
        out = ramp_colors(t, cs.get("ramp", "viridis"), cs.get("reverse", False))
        leg = [(f"{lo:g}", tuple(ramp_colors([0.0], cs.get("ramp"), cs.get("reverse"))[0])),
               (f"{hi:g}", tuple(ramp_colors([1.0], cs.get("ramp"), cs.get("reverse"))[0]))]
        return out, leg
    # categorical
    vals = table.text(col)
    cats = table.categories(col) if table.meta[col]["kind"] == "text" else sorted(set(vals) - {""})
    given = cs.get("categories") or {}
    cmap, k = {}, 0
    for c in cats:
        if c == "":
            continue
        if c in given:
            cmap[c] = hex_rgba(given[c])
        else:
            cmap[c] = hex_rgba(PALETTE[k % len(PALETTE)])
            k += 1
    out = np.array([cmap.get(v, NAN_COLOR) for v in vals], np.uint8).reshape(n, 4)
    return out, [(c, cmap[c]) for c in cmap]


def size_values(table, ss):
    n = len(table)
    col = ss.get("column")
    if not col:
        return np.full(n, float(ss.get("fixed", 9.0)))
    v = table.numeric(_column(table, col))
    fin = v[np.isfinite(v)]
    lo = ss.get("min") if ss.get("min") is not None else (float(fin.min()) if fin.size else 0.0)
    hi = ss.get("max") if ss.get("max") is not None else (float(fin.max()) if fin.size else 1.0)
    t = np.clip((v - lo) / (hi - lo), 0, 1) if hi > lo else np.full(n, 0.5)
    a, b = float(ss.get("min_px", 6.0)), float(ss.get("max_px", 26.0))
    # bubble area proportional to the value (sqrt), Spotfire-like
    out = a + (b - a) * np.sqrt(np.where(np.isfinite(t), t, 0.0))
    out[~np.isfinite(v)] = a * 0.8
    return out


def shape_values(table, sh):
    n = len(table)
    col = sh.get("column")
    if not col:
        return [sh.get("fixed", "o")] * n, []
    vals = table.text(_column(table, col))
    given = sh.get("map") or {}
    smap, k = {}, 0
    for v in vals:
        if v not in smap:
            if v in given:
                smap[v] = given[v]
            else:
                smap[v] = SYMBOLS[k % len(SYMBOLS)]
                k += 1
    return [smap[v] for v in vals], [(v or "(blank)", s) for v, s in smap.items()]


def label_values(table, ls):
    col = ls.get("column")
    if col in (None, ""):
        return [""] * len(table)
    if col == "Well":
        return list(table.ids)
    vals = table.text(_column(table, col))
    return vals


def pie_fractions(table, columns):
    """{well: {column: fraction}} from numeric columns (negative / NaN -> 0); wells with a zero total are left out."""
    out = {}
    if not columns:
        return out
    m = np.column_stack([np.nan_to_num(table.numeric(c), nan=0.0) for c in columns]).clip(min=0)
    tot = m.sum(axis=1)
    for k, i in enumerate(table.ids):
        if tot[k] > 0:
            out[i] = {c: float(m[k, j] / tot[k]) for j, c in enumerate(columns)}
    return out


def style_wells(table, style, filters=None, selection=None):
    """Everything the canvas needs to draw the well layer.

    Returns {ids, x, y, rgba (n,4), size (n,), symbol [n], label [n], visible (n,) bool, selected (n,) bool,
    legend {'color': [...], 'shape': [...], 'size': (column, lo, hi) | None}, pies {well: {col: f}}}.
    Wells without coordinates are never visible."""
    st = merge_style(style)
    n = len(table)
    x, y = table.xy()
    rgba, cleg = color_values(table, st["color"])
    size = size_values(table, st["size"])
    sym, sleg = shape_values(table, st["shape"])
    lab = label_values(table, st["label"])
    vis = np.isfinite(x) & np.isfinite(y)
    if filters is not None:
        vis &= filters.mask(table)
    sel = np.zeros(n, bool)
    if selection:
        keys = {MB.norm_id(s) for s in selection}
        sel = np.array([MB.norm_id(i) in keys for i in table.ids], bool)
    zleg = None
    if st["size"].get("column"):
        v = table.numeric(st["size"]["column"])
        fin = v[np.isfinite(v)]
        if fin.size:
            zleg = (st["size"]["column"], float(fin.min()), float(fin.max()))
    return {"ids": list(table.ids), "x": x, "y": y, "rgba": rgba, "size": size, "symbol": sym, "label": lab,
            "visible": vis, "selected": sel, "legend": {"color": cleg, "shape": sleg, "size": zleg},
            "pies": pie_fractions(table, st["pie"].get("columns") or [])}


def labels_in_view(styled, view, max_labels=300):
    """Indices of visible wells inside view (x0, x1, y0, y1) to label; [] above max_labels (zoom in)."""
    x0, x1, y0, y1 = view
    x, y = styled["x"], styled["y"]
    m = styled["visible"] & (x >= min(x0, x1)) & (x <= max(x0, x1)) & (y >= min(y0, y1)) & (y <= max(y0, y1))
    idx = np.flatnonzero(m)
    return [] if len(idx) > max_labels else idx.tolist()


# ====================================================================== linked views
def lasso_select(table, poly, visible=None):
    """Well ids inside polygon [(x, y), ...] (only visible ones when a mask is given)."""
    from .maphub import point_in_polygon
    x, y = table.xy()
    ok = np.isfinite(x) & np.isfinite(y)
    if visible is not None:
        ok &= np.asarray(visible, bool)
    if len(poly) < 3 or not ok.any():
        return []
    inside = np.zeros(len(table), bool)
    inside[ok] = point_in_polygon(x[ok], y[ok], [p[0] for p in poly], [p[1] for p in poly])
    return [i for i, k in zip(table.ids, inside) if k]


def box_select(table, x0, x1, y0, y1, visible=None):
    x, y = table.xy()
    m = (x >= min(x0, x1)) & (x <= max(x0, x1)) & (y >= min(y0, y1)) & (y <= max(y0, y1))
    if visible is not None:
        m &= visible
    return [i for i, k in zip(table.ids, m) if k]


def histogram_data(table, column, filters=None, selection=None, bins=20):
    """Histogram of a numeric column over the wells passing the OTHER filters (Spotfire style) with the selected
    wells' counts stacked: {edges, counts, selected, lo, hi}.  Text columns: one bar per category."""
    mask = filters.mask(table, exclude=column) if filters is not None else np.ones(len(table), bool)
    sel = np.zeros(len(table), bool)
    if selection:
        keys = {MB.norm_id(s) for s in selection}
        sel = np.array([MB.norm_id(i) in keys for i in table.ids], bool)
    if table.meta[column]["kind"] == "text":
        vals = table.text(column)
        cats = [c for c in table.categories(column)]
        counts = [sum(1 for v, m in zip(vals, mask) if m and v == c) for c in cats]
        scount = [sum(1 for v, m, s in zip(vals, mask, sel) if m and s and v == c) for c in cats]
        return {"categories": cats, "counts": counts, "selected": scount}
    v = table.numeric(column)
    ok = mask & np.isfinite(v)
    if not ok.any():
        return {"edges": np.array([0.0, 1.0]), "counts": np.zeros(1, int), "selected": np.zeros(1, int),
                "lo": np.nan, "hi": np.nan}
    lo, hi = float(v[ok].min()), float(v[ok].max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    counts = np.histogram(v[ok], edges)[0]
    scounts = np.histogram(v[ok & sel], edges)[0]
    return {"edges": edges, "counts": counts, "selected": scounts, "lo": lo, "hi": hi}


def crossplot_data(table, xcol, ycol, styled=None):
    """Points of the visible wells: {ids, x, y, rgba, selected, r (Pearson, NaN if < 3)}."""
    xv, yv = table.numeric(xcol), table.numeric(ycol)
    vis = np.isfinite(xv) & np.isfinite(yv)
    if styled is not None:
        vis &= styled["visible"]
    idx = np.flatnonzero(vis)
    r = float(np.corrcoef(xv[idx], yv[idx])[0, 1]) if len(idx) >= 3 and np.std(xv[idx]) > 0 and np.std(yv[idx]) > 0 \
        else float("nan")
    return {"ids": [table.ids[k] for k in idx], "x": xv[idx], "y": yv[idx],
            "rgba": styled["rgba"][idx] if styled is not None else None,
            "selected": styled["selected"][idx] if styled is not None else np.zeros(len(idx), bool), "r": r}


def table_rows(table, columns=None, visible=None, selection=None, sort=None, descending=False, selected_first=True):
    """[(id, [values as text])] for the well table view."""
    cols = list(columns) if columns else list(table.cols)
    keys = {MB.norm_id(s) for s in (selection or [])}
    idx = [k for k in range(len(table)) if visible is None or visible[k]]
    if sort and sort in table.cols:
        if table.meta[sort]["kind"] == "numeric":
            v = table.numeric(sort)
            idx.sort(key=lambda k: (not np.isfinite(v[k]), v[k] if np.isfinite(v[k]) else 0.0), reverse=False)
            if descending:
                fin = [k for k in idx if np.isfinite(v[k])][::-1]
                idx = fin + [k for k in idx if not np.isfinite(v[k])]
        else:
            t = table.text(sort)
            idx.sort(key=lambda k: t[k].lower(), reverse=descending)
    elif sort == "Well":
        idx.sort(key=lambda k: table.ids[k].lower(), reverse=descending)
    if selected_first and keys:
        idx = [k for k in idx if MB.norm_id(table.ids[k]) in keys] + [k for k in idx if MB.norm_id(table.ids[k]) not in keys]
    out = []
    for k in idx:
        row = []
        for c in cols:
            if table.meta[c]["kind"] == "numeric":
                v = table.cols[c][k]
                row.append("" if not np.isfinite(v) else f"{v:.6g}")
            else:
                row.append(table.cols[c][k])
        out.append((table.ids[k], row))
    return cols, out


def filter_widget_spec(table, column, max_categories=40):
    """What the Filters panel shows for a column: ('range', lo, hi) or ('values', [categories]) or None."""
    if column not in table.cols:
        return None
    if table.meta[column]["kind"] == "numeric":
        v = table.numeric(column)
        fin = v[np.isfinite(v)]
        if not fin.size:
            return None
        cats = np.unique(fin)
        if len(cats) <= 2 and set(cats.tolist()) <= {0.0, 1.0}:  # flag columns (0 / 1): checkboxes
            return ("values", [f"{c:g}" for c in cats])
        return ("range", float(fin.min()), float(fin.max()))
    cats = table.categories(column)
    return ("values", cats[:max_categories])


# ====================================================================== 31e seismic layers
MISTIE_COLORS = ((2.0, (0, 170, 0)), (5.0, (240, 150, 0)), (float("inf"), (220, 0, 0)))


def mistie_markers(event, use_residual=True, limits=(2.0, 5.0)):
    """Tie markers of a 'misties' DataChanged event: {x, y, rgba (n,4), label [n], value} coloured by |residual|
    (green < 2 ms, orange < 5 ms, red above - the item-10 basemap convention).  None when the event has no ties."""
    ties = (event or {}).get("ties") or []
    if not ties:
        return None
    key = "res_shift_ms" if use_residual else "shift_ms"
    v = np.array([abs(float(t.get(key) or 0.0)) for t in ties])
    cols = np.array([(0, 170, 0, 255) if a < limits[0] else (240, 150, 0, 255) if a < limits[1] else (220, 0, 0, 255)
                     for a in v], np.uint8).reshape(-1, 4)
    return {"x": np.array([t["x"] for t in ties], float), "y": np.array([t["y"] for t in ties], float), "rgba": cols,
            "label": [f"{t['a']} x {t['b']}: {float(t.get(key) or 0):+.1f} ms" for t in ties], "value": v,
            "source": event.get("role") or event.get("source")}


def shotpoint_marks(x, y, cdp, every=100):
    """Annotation every `every` CDPs / shotpoints: (indices, labels) - first and last trace always included."""
    cdp = np.asarray(cdp)
    n = len(cdp)
    if n == 0:
        return [], []
    every = max(1, int(every))
    idx = np.flatnonzero(np.asarray(cdp, float) % every == 0).tolist()
    for k in (0, n - 1):
        if k not in idx:
            idx.append(k)
    idx = sorted(set(idx))
    return idx, [f"{float(cdp[k]):g}" for k in idx]


# ====================================================================== 31f well sections
def wells_along_line(table, xs, ys, buffer=None, visible=None):
    """Wells within `buffer` map units of the polyline, ordered along it: [(id, distance along, offset)].
    buffer None = 5 % of the line length."""
    px, py = np.asarray(xs, float), np.asarray(ys, float)
    if len(px) < 2:
        return []
    seg = np.hypot(np.diff(px), np.diff(py))
    cum = np.r_[0.0, np.cumsum(seg)]
    if buffer is None:
        buffer = 0.05 * cum[-1]
    x, y = table.xy()
    out = []
    for k, i in enumerate(table.ids):
        if not (np.isfinite(x[k]) and np.isfinite(y[k])) or (visible is not None and not visible[k]):
            continue
        best = None
        for j in range(len(seg)):
            if seg[j] <= 0:
                continue
            ux, uy = (px[j + 1] - px[j]) / seg[j], (py[j + 1] - py[j]) / seg[j]
            t = np.clip((x[k] - px[j]) * ux + (y[k] - py[j]) * uy, 0.0, seg[j])
            d = math.hypot(x[k] - (px[j] + t * ux), y[k] - (py[j] + t * uy))
            if best is None or d < best[1]:
                best = (cum[j] + t, d)
        if best is not None and best[1] <= buffer:
            out.append((i, float(best[0]), float(best[1])))
    out.sort(key=lambda r: r[1])
    return out


def section_xy(table, wells):
    """Map trace [[x, y], ...] through the wells (those with coordinates) in the given order."""
    x, y = table.xy()
    out = []
    for w in wells:
        k = table.index(w)
        if k is not None and np.isfinite(x[k]) and np.isfinite(y[k]):
            out.append([round(float(x[k]), 2), round(float(y[k]), 2)])
    return out


# ====================================================================== 31h layers from other modules
def layer_request(name, kind="lines", items=None, owner="", style=None, file=None):
    """What a tool publishes (bus.notify('layers', layer=...)) to put something on the Live Map:
    kind 'lines' (items [{name, x, y, action}] - e.g. basin / Wheeler section locations; action = what a click opens:
    {kind 2D | 3D, name, mode, number}), 'polygons'
    (items [{name, x, y}]), 'points' (items [{x, y, text}]) or 'grid' (file = CPS-3 / ZMAP+ / .npz MapGrid)."""
    return {"name": str(name), "kind": kind, "items": list(items or []), "owner": str(owner), "style": dict(style or {}),
            "file": file}


def apply_layer_request(stack, req):
    """Add or replace (by name) the requested layer in the hub stack.  Returns the layer id."""
    name = req.get("name") or "layer"
    kind = req.get("kind", "lines")
    style = dict(req.get("style") or {})
    if kind == "grid":
        g = read_grid_file(req["file"])
        lay = next((l for l in stack.layers if l["kind"] == "grid" and l["name"] == name), None)
        if lay is None:
            lid, _ = stack.add_grid(g, name, style=style or None, contours=True)
        else:
            lid = lay["id"]
            stack._grids[lid] = g
            if stack.project is not None and getattr(stack.project, "path", None):
                from .maplayers import save_grid
                lay["src"]["file"] = save_grid(stack.project, g, lid)
        stack.get(lid)["src"]["registered"] = {"owner": req.get("owner"), "file": req["file"]}
        return lid
    lk = {"lines": "polygons", "polygons": "polygons", "points": "text"}[kind]
    if kind == "points":
        src = {"items": [{"x": float(i["x"]), "y": float(i["y"]), "text": str(i.get("text", ""))}
                         for i in req.get("items") or []]}
    else:
        src = {"polygons": [dict(i, x=[float(v) for v in i["x"]], y=[float(v) for v in i["y"]],
                                 open=(kind == "lines")) for i in req.get("items") or []]}
    src["registered"] = {"owner": req.get("owner"), "kind": kind}
    lay = next((l for l in stack.layers if l["kind"] == lk and l["name"] == name), None)
    if lay is None:
        if kind == "lines":
            style = dict({"color": (0, 200, 230), "width": 2.5, "fill": None}, **style)
        lid = stack.add(lk, name, src=src, style=style or None)
    else:
        lid = lay["id"]
        lay["src"] = src
        if style:
            lay["style"].update(style)
    return lid


def read_grid_file(path):
    """MapGrid from a CPS-3 / ZMAP+ grid file or an npz written by maplayers.save_grid."""
    import os
    from .maps import MapGrid, read_cps3
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as z:
            x0, y0, dx, dy = (float(v) for v in z["geo"])
            return MapGrid(z["z"].astype(np.float64), "xy", str(z["unit"]), str(z["label"]), None, x0, y0, dx, dy)
    if ext in (".zmap", ".zmp", ".dat"):
        from .gis.zmap import read_zmap
        return read_zmap(path)
    return read_cps3(path)


def section_hit(stack, x, y, tol):
    """The registered section line (kind 'lines' layers) nearest to (x, y) within tol: (layer id, item) or None."""
    best = None
    for l in stack.layers:
        if l["kind"] != "polygons" or not l.get("visible", True):
            continue
        for it in l["src"].get("polygons", []):
            if not it.get("open"):
                continue
            px, py = np.asarray(it["x"], float), np.asarray(it["y"], float)
            for k in range(len(px) - 1):
                ex, ey = px[k + 1] - px[k], py[k + 1] - py[k]
                L2 = ex * ex + ey * ey
                t = 0.0 if L2 == 0 else min(1.0, max(0.0, ((x - px[k]) * ex + (y - py[k]) * ey) / L2))
                d = math.hypot(x - (px[k] + t * ex), y - (py[k] + t * ey))
                if d <= tol and (best is None or d < best[0]):
                    best = (d, l["id"], it)
    return None if best is None else (best[1], best[2])


# ====================================================================== 31i output
def styled_primitives(styled, max_labels=300):
    """Hub render primitives of the Live Map's styled wells (for the PNG / PDF export)."""
    if styled is None:
        return []
    vis = np.flatnonzero(styled["visible"])
    if not len(vis):
        return []
    prim = [("points", "livemap-wells", styled["x"][vis], styled["y"][vis],
             [tuple(int(v) for v in styled["rgba"][k]) for k in vis], styled["size"][vis], "o")]
    if len(vis) <= max_labels:
        for k in vis:
            if styled["label"][k]:
                prim.append(("text", "livemap-wells", float(styled["x"][k]), float(styled["y"][k]),
                             str(styled["label"][k]), (30, 30, 30, 255), 7))
    return prim


def export_map(stack, styled, path, title="Live Map", **kw):
    """PNG / PDF with title block, legend, scale, north arrow (item-21f layout) + the styled wells."""
    from .mapexport import export_map as _export
    hidden = [l for l in stack.layers if l["kind"] == "wells" and l["visible"]]
    for l in hidden:  # the Live Map draws its own wells
        l["visible"] = False
    try:
        leg = list((styled or {}).get("legend", {}).get("color") or []) if styled else []
        return _export(stack, path, title=title, extra_prims=styled_primitives(styled), extra_legend=leg,
                       symbols=False, **kw)
    finally:
        for l in hidden:
            l["visible"] = True


def export_gis(stack, table, folder, styled=None, sections=None, picks=None, epsg=None):
    """Everything on the Live Map for QGIS / ArcGIS / Petrel: the hub layers (shapefile + GeoJSON + GeoPackage + KML,
    grids as GeoTIFF + ZMAP+), the wells with every WellTable column (wells_table.geojson / .shp), section lines
    ({name: [[x, y], ...]}) and 2D picks ({horizon: (x, y, t)}) as GeoJSON."""
    import os
    from .gis.export import export_stack, project_crs
    from .gis.geojson import write_geojson
    from .gis import shapefile as SHP
    out = export_stack(stack, folder, epsg=epsg, geojson=True, zmap=True)
    c = project_crs(stack.project) if stack.project is not None else None
    code = epsg or (c or {}).get("epsg")
    x, y = table.xy()
    ok = np.isfinite(x) & np.isfinite(y)
    if styled is not None:
        ok &= styled["visible"]
    idx = np.flatnonzero(ok)
    if len(idx):
        geoms = [{"type": "Point", "parts": [np.array([[x[k], y[k]]])]} for k in idx]
        attrs = {"well": [table.ids[k] for k in idx]}
        for col in table.cols:
            if col in ("X", "Y"):
                continue
            v = table.cols[col]
            attrs[col] = [(float(v[k]) if np.isfinite(v[k]) else None) if table.meta[col]["kind"] == "numeric" else v[k]
                          for k in idx]
        out["files"].append(write_geojson(os.path.join(folder, "wells_table.geojson"), geoms, attrs, epsg=code))
        try:  # dBASE field names: 10 characters
            short = {}
            for k_ in attrs:
                s = "".join(ch for ch in k_ if ch.isalnum() or ch == "_")[:10] or "f"
                while s in short.values():
                    s = s[:8] + f"{len(short) % 100:02d}"
                short[k_] = s
            SHP.write(os.path.join(folder, "wells_table.shp"), geoms,
                      {short[k_]: [("" if v is None else v) for v in vals] for k_, vals in attrs.items()}, None, "Point")
            out["files"].append(os.path.join(folder, "wells_table.shp"))
        except Exception as e:
            out["warnings"].append(f"wells_table.shp: {e}")
    if sections:
        names = [n for n, xy in sections.items() if xy and len(xy) >= 2]
        if names:
            out["files"].append(write_geojson(os.path.join(folder, "section_lines.geojson"),
                                              [{"type": "LineString", "parts": [np.asarray(sections[n], float)]}
                                               for n in names], {"name": names}, epsg=code))
    if picks:
        geoms, hn, tv = [], [], []
        for h, (px, py, pt) in picks.items():
            for a, b, t in zip(px, py, pt):
                if np.isfinite(t):
                    geoms.append({"type": "Point", "parts": [np.array([[a, b]], float)]})
                    hn.append(h)
                    tv.append(float(t))
        if geoms:
            out["files"].append(write_geojson(os.path.join(folder, "picks.geojson"), geoms,
                                              {"horizon": hn, "twt_ms": tv}, epsg=code))
    return out


def picks_2d(project, max_per_line=2000):
    """{horizon: (x, y, t)} of the 2D-line picks (decimated per line) for export_gis."""
    out = {}
    for h in getattr(project.horizons, "items", []) if project is not None else []:
        xs, ys, ts = [], [], []
        for k, a in h.data.items():
            if not k.startswith("2D:"):
                continue
            try:
                line = project.line(k[3:])
            except Exception:
                continue
            if not line.has_coords:
                continue
            a = np.asarray(a, float)
            ok = np.flatnonzero(np.isfinite(a))
            ok = ok[:: max(1, len(ok) // max_per_line)]
            xs.append(np.asarray(line.x, float)[ok])
            ys.append(np.asarray(line.y, float)[ok])
            ts.append(a[ok])
        if xs:
            out[h.name] = (np.concatenate(xs), np.concatenate(ys), np.concatenate(ts))
    return out


# ====================================================================== 31d deviated well paths
def path_polylines(trajectories, min_offset=5.0, max_points=200):
    """{well: (x, y)} map traces of deviated wells (horizontal offset > min_offset m); vertical wells are left out
    (their symbol is the path).  Each trace decimated to <= max_points vertices."""
    out = {}
    for t in trajectories or []:
        x, y = np.asarray(t.x, float), np.asarray(t.y, float)
        if len(x) < 2:
            continue
        off = np.hypot(x - x[0], y - y[0])
        if not np.isfinite(off).any() or np.nanmax(off) <= min_offset:
            continue
        k = max(1, int(np.ceil(len(x) / max_points)))
        idx = np.unique(np.r_[np.arange(0, len(x), k), len(x) - 1])
        out[t.name] = (x[idx], y[idx])
    return out


def well_paths(project):
    """Deviated well paths of the project's wells (survey files of the linked well project)."""
    try:
        from .laterals import trajectories_from_wellset
        return path_polylines(trajectories_from_wellset(project.wellset()))
    except Exception:
        return {}
