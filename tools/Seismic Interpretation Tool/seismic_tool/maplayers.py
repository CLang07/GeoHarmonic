"""Map hub layers (PLAN 21a): an ordered, saved stack of map layers with visibility, opacity, order and style, and a
Qt-free renderer that turns the stack into drawing primitives (the basemap draws them; PNG export and the page
layouts of 21g can reuse them).

Layer kinds
    outline3d  3D survey outline(s)                src {"volume": name | None = all}
    lines2d    2D lines                            src {}
    wells      wells from the linked well project  src {}; style labels on/off
    faults     fault sticks / traces               src {}
    horizon    a horizon's time map                src {"horizon": name}
    grid       a stored grid (any MapGrid)         src {"file": maps/<id>.npz}; style lut, reverse, vmin, vmax
    contours   contours of a grid layer            src {"grid": layer id}; style interval, index_every, colour
    polygons   named polygons (prospects, leases)  src {"polygons": [{"name", "x": [...], "y": [...]}]}
    text       free text                           src {"items": [{"x", "y", "text"}]}
    petro      a petrophysics zone value per well  src {"property", "zone"} (21b)

Saved in project.display["map_layers"]; grid data in <project>_seismic/maps/<id>.npz (never inside the JSON).
"""
import copy
import math
import os

import numpy as np

KINDS = ("outline3d", "lines2d", "wells", "faults", "horizon", "grid", "contours", "polygons", "text", "petro", "vector",
         "blend")
DEFAULT_Z = {"grid": 10, "blend": 11, "horizon": 12, "vector": 25, "contours": 20, "polygons": 30, "faults": 40, "outline3d": 50, "lines2d": 60,
             "wells": 70, "petro": 75, "text": 80}
DEFAULT_STYLE = {
    "outline3d": {"color": (31, 119, 180), "width": 2.0},
    "lines2d": {"color": (90, 90, 90), "width": 1.5, "labels": True},
    "wells": {"color": (0, 0, 0), "size": 8, "labels": True, "symbols": False},  # symbols: status symbols (21f)
    "faults": {"color": None, "width": 2.0},
    "horizon": {"lut": "Rainbow (attributes)", "reverse": True},
    "grid": {"lut": "Rainbow (attributes)", "reverse": None, "vmin": None, "vmax": None,
             "ramp": None, "hillshade": False, "ve": 1.0, "shade_strength": 0.55, "blank": None,
             "shade_from": None, "azimuth": None, "elevation": 35.0},
    # shade_from: layer id of a structure grid whose illumination shades this one (amplitude on structure, 23b);
    # azimuth: single light direction (bump map) instead of the multi-directional hillshade; elevation in degrees
    "blend": {"mode": "rgb", "clip": (2.0, 98.0), "gamma": 1.0},  # src {"grids": [lid, lid, lid]} co-blend (23a)
    # ramp: carto ramp (depth / thickness / property / diverging / terrain / bathymetry) instead of lut; blank: list of
    # polygon-layer ids (or True = every polygons layer with style fault) blanked on the grid and its contours
    "contours": {"interval": None, "index_every": 5, "color": (40, 40, 40), "width": 1.0, "index_width": 2.0,
                 "labels": False, "label_size": 8},
    "polygons": {"color": (200, 0, 0), "width": 2.0, "fill": (200, 0, 0, 40), "labels": True, "fault": False,
                 "down": None},  # fault: heave polygons, hatched; ticks outward on the edges facing 'down' (N, E, SW
                                 # or an azimuth; per polygon "down" overrides; None = all edges)
    "text": {"color": (0, 0, 0), "size": 10},
    "vector": {"color": (80, 80, 80), "width": 1.0, "fill": None, "by": None, "mode": "categorized", "classes": 5,
               "labels": None, "filter": "", "size": 6, "symbol": "o", "simplify": True},  # gis/vector.py (21g)
    "petro": {"lut": "Rainbow (attributes)", "reverse": False, "size": 14, "labels": True, "mode": "bubbles",
              "pie_radius": None},  # mode bubbles | values | pies (mineral solver volumes, 21b)
}
LUT_NAMES = None  # filled lazily from gui.colormaps (pure numpy module)


def _lut(name, reverse=False):
    from .gui.colormaps import lut
    return lut(name, reverse)


# ------------------------------------------------------------------------------------------- grid persistence
def maps_dir(project, create=False):
    d = os.path.join(project.seismic_dir(), "maps")
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def save_grid(project, grid, layer_id):
    """Write a MapGrid to <project>_seismic/maps/<id>.npz; returns the path relative to the maps folder."""
    d = maps_dir(project, create=True)
    fn = f"{layer_id}.npz"
    path = os.path.join(d, fn)
    vol = getattr(grid, "vol", None)
    vname = ""
    if grid.kind == "ilxl" and vol is not None:
        for v in project.volumes:  # project.volume() caches the Volume3D objects
            try:
                if project.volume(v["name"]) is vol:
                    vname = v["name"]
                    break
            except Exception:
                continue
        if not vname and project.volumes:
            vname = project.volumes[0]["name"]
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, z=np.asarray(grid.z, np.float32), kind=np.array(grid.kind), unit=np.array(grid.unit or ""),
                        label=np.array(grid.label or ""), vol=np.array(vname),
                        geo=np.array([grid.x0, grid.y0, grid.dx, grid.dy], float))
    os.replace(tmp, path)
    return fn


def load_grid(project, fn):
    from .maps import MapGrid
    path = os.path.join(maps_dir(project), fn)
    with np.load(path, allow_pickle=False) as z:
        kind = str(z["kind"])
        vol = project.volume(str(z["vol"])) if kind == "ilxl" and str(z["vol"]) else None
        x0, y0, dx, dy = (float(v) for v in z["geo"])
        return MapGrid(z["z"].astype(np.float64), kind, str(z["unit"]), str(z["label"]), vol, x0, y0, dx, dy)


# ------------------------------------------------------------------------------------------- the stack
class LayerStack:
    def __init__(self, project=None, layers=None):
        self.project = project
        self.layers = layers or []
        self._grids = {}
        self._next = 1 + max([int(str(l["id"]).lstrip("L") or 0) for l in self.layers] + [0])

    # ---- persistence
    def to_dict(self):
        return {"version": 1, "layers": copy.deepcopy(self.layers)}

    def save(self):
        if self.project is not None:
            self.project.display["map_layers"] = self.to_dict()
        return self.to_dict()

    @classmethod
    def load(cls, project, default=True):
        d = (project.display or {}).get("map_layers") if project is not None else None
        if d and d.get("layers") is not None:
            ls = []
            for l in d["layers"]:
                l = dict(l)
                l["style"] = dict(DEFAULT_STYLE.get(l["kind"], {}), **(l.get("style") or {}))
                ls.append(l)
            return cls(project, ls)
        return default_stack(project) if default else cls(project)

    # ---- editing
    def add(self, kind, name=None, src=None, style=None, visible=True, opacity=1.0, z=None):
        if kind not in KINDS:
            raise ValueError(f"unknown layer kind {kind}")
        lid = f"L{self._next}"
        self._next += 1
        lay = {"id": lid, "kind": kind, "name": name or kind, "visible": bool(visible), "opacity": float(opacity),
               "z": float(DEFAULT_Z[kind] if z is None else z), "src": dict(src or {}),
               "style": dict(DEFAULT_STYLE.get(kind, {}), **(style or {}))}
        # keep z unique-ish: new layers of the same kind go on top of their group
        same = [l["z"] for l in self.layers if abs(l["z"] - lay["z"]) < 1]
        if same and z is None:
            lay["z"] = max(same) + 0.01
        self.layers.append(lay)
        return lid

    def add_grid(self, grid, name=None, style=None, contours=True, interval=None):
        """Store a MapGrid as a layer (file in <project>_seismic/maps) and optionally a contour layer on it."""
        lid = self.add("grid", name or grid.label or "grid", style=style)
        if self.project is not None and self.project.path:
            self.get(lid)["src"]["file"] = save_grid(self.project, grid, lid)
        self._grids[lid] = grid
        cid = None
        if contours:
            cid = self.add("contours", f"{name or grid.label or 'grid'} contours", src={"grid": lid},
                           style={"interval": interval})
        return lid, cid

    def get(self, lid):
        for l in self.layers:
            if l["id"] == lid:
                return l
        raise KeyError(lid)

    def remove(self, lid):
        lay = self.get(lid)
        self.layers.remove(lay)
        for l in [l for l in self.layers if l["kind"] == "contours" and l["src"].get("grid") == lid]:
            self.layers.remove(l)
        self._grids.pop(lid, None)

    def ordered(self):
        """Layers bottom -> top."""
        return sorted(self.layers, key=lambda l: l["z"])

    def move(self, lid, delta):
        """Move a layer up (+1) or down (-1) in the drawing order (swaps z with its neighbour)."""
        o = self.ordered()
        i = next(k for k, l in enumerate(o) if l["id"] == lid)
        j = i + (1 if delta > 0 else -1)
        if 0 <= j < len(o):
            o[i]["z"], o[j]["z"] = o[j]["z"], o[i]["z"]
            if o[i]["z"] == o[j]["z"]:
                o[i]["z"] += 0.01 if delta > 0 else -0.01
        return [l["id"] for l in self.ordered()]

    def set_visible(self, lid, on):
        self.get(lid)["visible"] = bool(on)

    def set_opacity(self, lid, a):
        self.get(lid)["opacity"] = float(np.clip(a, 0.0, 1.0))

    def set_style(self, lid, **kw):
        self.get(lid)["style"].update(kw)

    def rename(self, lid, name):
        self.get(lid)["name"] = str(name)

    def grid(self, lid):
        """MapGrid of a grid layer (loaded from its npz once)."""
        if lid in self._grids:
            return self._grids[lid]
        lay = self.get(lid)
        if lay["kind"] == "horizon":
            g = horizon_grid_of(self.project, lay["src"].get("horizon"))
        else:
            fn = lay["src"].get("file")
            g = load_grid(self.project, fn) if fn and self.project is not None else None
        self._grids[lid] = g
        return g

    # ---- legend / colour bars
    def colorbar(self, lid):
        lay = self.get(lid)
        if lay["kind"] not in ("grid", "horizon", "petro"):
            return None
        if lay["kind"] == "petro":
            if lay["style"].get("mode") == "pies":
                return None
            vals = petro_values(self.project, lay["src"].get("property"), lay["src"].get("zone"))
            v = np.array([q[3] for q in vals], float)
            unit = ""
        else:
            g = self.grid(lid)
            if g is None:
                return None
            v = g.z[np.isfinite(g.z)]
            unit = g.unit
        if not v.size:
            return None
        st = lay["style"]
        lo = st.get("vmin") if st.get("vmin") is not None else float(np.nanmin(v))
        hi = st.get("vmax") if st.get("vmax") is not None else float(np.nanmax(v))
        rev = st.get("reverse")
        if rev is None:
            rev = unit in ("ms", "m")
        return {"lut": st.get("lut", "Rainbow (attributes)"), "reverse": bool(rev), "vmin": lo, "vmax": hi, "unit": unit,
                "label": lay["name"]}

    def legend(self):
        """Legend entries for the VISIBLE layers, top first: [{"id", "name", "kind", "symbol", "color", "colorbar"}]."""
        out = []
        for lay in reversed(self.ordered()):
            if not lay["visible"]:
                continue
            st = lay["style"]
            sym = {"outline3d": "line", "lines2d": "line", "faults": "line", "contours": "line", "polygons": "polygon",
                   "wells": "point", "petro": "point", "text": "text", "grid": "ramp", "horizon": "ramp",
                   "vector": "line"}[lay["kind"]]
            if lay["kind"] == "petro" and st.get("mode") == "pies":
                from .maphub import petro_pies, COMPONENT_COLORS
                comps = sorted({c for q in petro_pies(self.project, lay["src"].get("zone")) for c in q[3]})
                out.append({"id": lay["id"], "name": lay["name"], "kind": "petro", "symbol": "pie", "color": None,
                            "colorbar": None, "components": [(c, COMPONENT_COLORS.get(c, (128, 128, 128))) for c in comps]})
                continue
            out.append({"id": lay["id"], "name": lay["name"], "kind": lay["kind"], "symbol": sym,
                        "color": st.get("color"), "colorbar": self.colorbar(lay["id"]) if sym == "ramp" or lay["kind"] == "petro"
                        else None})
        return out


def default_stack(project):
    """The legacy basemap content as layers: grids underneath, outlines, 2D lines, faults, wells on top."""
    st = LayerStack(project)
    if project is None:
        return st
    for v in project.volumes:
        st.add("outline3d", f"{v['name']} (3D outline)", src={"volume": v["name"]})
    if project.lines:
        st.add("lines2d", "2D lines")
    if getattr(project, "faults", None) is not None and project.faults.items:
        st.add("faults", "Faults")
    st.add("wells", "Wells")
    return st


def horizon_grid_of(project, name):
    """MapGrid of a horizon on the first survey that has it (3D), else gridded 2D picks."""
    if project is None or not name:
        return None
    h = project.horizons.get(name)
    if h is None:
        return None
    from .maps import horizon_grid
    keys = sorted(h.data, key=lambda k: (not k.startswith("3D:"), k))
    for k in keys:
        try:
            return horizon_grid(project, h, k)
        except Exception:
            continue
    return None


def petro_values(project, prop, zone=None):
    """[(well, x, y, value)] of a zone property from the linked well project's <wcproj>_petro/summary.csv (21b)."""
    import csv
    if project is None or not project.well_project or not prop:
        return []
    base = project.well_project[: -len(".wcproj.json")] if project.well_project.endswith(".wcproj.json") else project.well_project
    path = base + "_petro/summary.csv"
    if not os.path.exists(path):
        return []
    xy = {n: (x, y) for n, x, y in project.wells()}
    vals = {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and prop not in rows[0]:  # field-report counts / contacts (well_correlation petro/reports.py, item 22b)
        rp = base + "_petro/reports_summary.csv"
        if os.path.exists(rp):
            with open(rp, newline="") as f:
                rows = [dict(r, Zone=zone) for r in csv.DictReader(f)]
    for r in rows:
        if zone and r.get("Zone") != zone:
            continue
        try:
            v = float(r.get(prop, ""))
        except (TypeError, ValueError):
            continue
        w = r.get("Well")
        if w in xy:
            vals.setdefault(w, []).append(v)
    out = []
    for w, vs in vals.items():
        v = float(np.sum(vs)) if (zone is None and prop in ("NetPay_TVD", "NetRes_TVD", "PHIH_pay", "PHIH_res", "SoPhiH",
                                                           "kh_pay", "Gross_TVD")) else float(np.mean(vs))
        out.append((w, xy[w][0], xy[w][1], v))
    return out


# ------------------------------------------------------------------------------------------- rendering (Qt-free)
def _rgba_image(z, lut, vmin, vmax, opacity):
    from .horizons import structure_rgba
    rgba, lo, hi = structure_rgba(z, lut, vmin, vmax)
    rgba = rgba.copy()
    rgba[..., 3] = (rgba[..., 3].astype(float) * opacity).astype(np.uint8)
    return rgba, lo, hi


def _compass(d):
    """'N', 'NE', 'E' ... or an azimuth in degrees -> unit (east, north) vector; None = no preference."""
    if d is None or d in ("", "right", "left"):
        return None
    az = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}.get(str(d).upper())
    if az is None:
        try:
            az = float(d)
        except ValueError:
            return None
    a = math.radians(az)
    return (math.sin(a), math.cos(a))


def _ramp_or_lut(st, cb):
    if st.get("ramp"):
        from .welltie import wc
        wc()
        from well_correlation.carto import ramp_lut
        return ramp_lut(st["ramp"], bool(st.get("ramp_reverse")))
    return _lut(cb["lut"], cb["reverse"])


def nice_interval(z, target=12):
    from .maps import nice_levels
    lv = nice_levels(z, target)
    return (lv[1] - lv[0]) if len(lv) > 1 else None


def contour_levels(z, interval):
    f = z[np.isfinite(z)]
    if not f.size or not interval:
        return []
    lo, hi = float(f.min()), float(f.max())
    return list(np.arange(np.ceil(lo / interval) * interval, hi + 1e-9, interval))


QT_WELL_SYMBOLS = {  # basemap approximations of the industry symbols (exact ones in the matplotlib export)
    "oil": ("o", (0, 0, 0)), "gas": ("star", (220, 30, 30)), "oil+gas": ("star", (0, 0, 0)), "show": ("o", (150, 150, 150)),
    "dry": ("x", (0, 0, 0)), "p&a": ("+", (0, 0, 0)), "planned": ("o", (255, 255, 255)), "injector": ("t", (30, 90, 220)),
    "unknown": ("o", (0, 0, 0)),
}


def _cell_sizes(g):
    m11, m12, m21, m22, _, _ = g.transform()
    return math.hypot(m11, m12) or 1.0, math.hypot(m21, m22) or 1.0


def blank_polygons_grid(g, polys):
    """MapGrid copy with NaN inside the polygons and within 0.75 cell of their edges, so that no contour cell can
    straddle a polygon edge (contours stop at fault polygons)."""
    from .maps import MapGrid
    from .maphub import point_in_polygon
    if not polys:
        return g
    X, Y = g.node_xy()
    x, y = np.ravel(X), np.ravel(Y)
    cell = max(_cell_sizes(g))
    m = np.zeros(x.shape, bool)
    for px, py in polys:
        px, py = np.asarray(px, float), np.asarray(py, float)
        m |= point_in_polygon(x, y, px, py)
        ex, ey = np.append(px, px[0]), np.append(py, py[0])
        for k in range(len(ex) - 1):
            ax, ay, bx, by = ex[k], ey[k], ex[k + 1], ey[k + 1]
            L2 = (bx - ax) ** 2 + (by - ay) ** 2
            t = np.clip(((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / max(L2, 1e-12), 0, 1)
            m |= np.hypot(x - (ax + t * (bx - ax)), y - (ay + t * (by - ay))) < 0.75 * cell
    z = np.where(m.reshape(g.z.shape), np.nan, g.z)
    return MapGrid(z, g.kind, g.unit, g.label, g.vol, g.x0, g.y0, g.dx, g.dy)


def _blank_polys(stack, spec):
    if not spec:
        return []
    ids = [l["id"] for l in stack.layers if l["kind"] == "polygons" and l["style"].get("fault")] if spec is True else list(spec)
    out = []
    for lid in ids:
        try:
            lay = stack.get(lid)
        except KeyError:
            continue
        out += [(p_["x"], p_["y"]) for p_ in lay["src"].get("polygons", []) if len(p_["x"]) >= 3]
    return out


def display_grid(stack, lid):
    """The grid of a grid / horizon layer as drawn (blanked by its fault polygons when styled so)."""
    g = stack.grid(lid)
    if g is None:
        return None
    return blank_polygons_grid(g, _blank_polys(stack, stack.get(lid)["style"].get("blank")))


def well_status(project):
    """{well: symbol kind} from the well project's header info (Status / Purpose / Result columns)."""
    import json
    from .welltie import wc
    if project is None or not project.well_project or not os.path.exists(project.well_project):
        return {}
    wc()
    from well_correlation.carto import classify_status
    with open(project.well_project) as f:
        d = json.load(f)
    out = {}
    for w in d.get("wells", []):
        info = w.get("info") or {}
        text = " ".join(str(info.get(k, "")) for k in ("Status", "Result", "Purpose", "Type", "Well_Status") if info.get(k))
        out[w["name"]] = classify_status(text)
    return out


def render(stack, wells=None):
    """Drawing primitives for the visible layers, bottom -> top:
       ("image", layer_id, rgba, transform6)
       ("curve", layer_id, xs, ys, rgba, width, connect)       connect 'all' | 'pairs' | 'finite'
       ("points", layer_id, x, y, rgba list/array, size, symbol)
       ("text", layer_id, x, y, text, rgba, size)
       ("polygon", layer_id, xs, ys, fill rgba, edge rgba | None[, {"hatch": True}])   filled (pies, fills, faults)
       text may carry an 8th element: angle in degrees (contour labels)
    wells: optional [(name, x, y)] (default project.wells())."""
    prim = []
    p = stack.project
    for lay in stack.ordered():
        if not lay["visible"]:
            continue
        lid, st, a = lay["id"], lay["style"], lay["opacity"]
        kind = lay["kind"]

        def col(c, alpha=1.0):
            c = tuple(c or (0, 0, 0))
            base_a = c[3] if len(c) > 3 else 255
            return (int(c[0]), int(c[1]), int(c[2]), int(base_a * a * alpha))

        try:
            if kind == "outline3d" and p is not None:
                names = [lay["src"]["volume"]] if lay["src"].get("volume") else [v["name"] for v in p.volumes]
                for vn in names:
                    vol = p.volume(vn)
                    xs, ys = vol.outline()
                    prim.append(("curve", lid, np.asarray(xs, float), np.asarray(ys, float), col(st["color"]), st["width"], "all"))
                    prim.append(("text", lid, float(xs[0]), float(ys[0]), f"{vn} (3D)", col(st["color"]), 9))
            elif kind == "lines2d" and p is not None:
                hidden = set(p.display.get("hidden_lines", []) or [])
                for l in p.lines:
                    if l["name"] in hidden:
                        continue
                    line = p.line(l["name"])
                    if not line.has_coords:
                        continue
                    step = max(1, len(line.x) // 400)
                    x, y = np.asarray(line.x[::step], float), np.asarray(line.y[::step], float)
                    prim.append(("curve", lid, x, y, col(st["color"]), st["width"], "all"))
                    if st.get("labels", True):
                        prim.append(("text", lid, float(x[0]), float(y[0]), l["name"], col((60, 60, 60)), 8))
            elif kind == "wells":
                ws = wells if wells is not None else (p.wells() if p is not None else [])
                if ws and st.get("symbols"):
                    kinds = well_status(p)
                    groups = {}
                    for w in ws:
                        groups.setdefault(kinds.get(w[0], "unknown"), []).append(w)
                    for kd, lst in groups.items():
                        sym, c = QT_WELL_SYMBOLS.get(kd, QT_WELL_SYMBOLS["unknown"])
                        prim.append(("points", lid, np.array([w[1] for w in lst], float), np.array([w[2] for w in lst], float),
                                     col(c), st["size"] * (1.4 if sym == "star" else 1.0), sym))
                elif ws:
                    x = np.array([w[1] for w in ws], float)
                    y = np.array([w[2] for w in ws], float)
                    prim.append(("points", lid, x, y, col(st["color"]), st["size"], "o"))
            elif kind == "faults" and p is not None and getattr(p, "faults", None) is not None:
                for f in p.faults.items:
                    if not f.visible:
                        continue
                    c = st.get("color") or _hex(f.color)
                    for s in f.sticks:
                        prim.append(("curve", lid, s[:, 0], s[:, 1], col(c), st["width"], "all"))
                    o = f.ordered_sticks()
                    if len(o) > 1:
                        prim.append(("curve", lid, np.array([s[0, 0] for s in o]), np.array([s[0, 1] for s in o]),
                                     col(c), st["width"], "all"))
            elif kind in ("grid", "horizon"):
                g = display_grid(stack, lid)
                cb = stack.colorbar(lid)
                if g is not None and cb is not None:
                    lut = _ramp_or_lut(st, cb)
                    if st.get("shade_from") or (st.get("hillshade") and st.get("azimuth") is not None):
                        from . import blend as B
                        sg = display_grid(stack, st["shade_from"]) if st.get("shade_from") else g
                        if sg is None:
                            raise ValueError(f"shade_from layer {st.get('shade_from')} has no grid")
                        sz = B.on_grid(g, sg)
                        cx_, cy_ = _cell_sizes(g)
                        sh = B.illuminate(sz, cx_, cy_, 315.0 if st.get("azimuth") is None else float(st["azimuth"]),
                                          float(st.get("elevation") or 35.0), float(st.get("ve") or 1.0))
                        rgba, _, _ = _rgba_image(g.z, lut, cb["vmin"], cb["vmax"], a)
                        rgba = B.shade_rgba(rgba, sh, float(st.get("shade_strength", 0.55)))
                    elif st.get("hillshade"):
                        from .welltie import wc
                        wc()
                        from well_correlation.carto import hillshade, shaded_rgba
                        cx_, cy_ = _cell_sizes(g)
                        sh = hillshade(g.z, cx_, cy_, ve=float(st.get("ve") or 1.0))
                        rgba = shaded_rgba(g.z, lut, cb["vmin"], cb["vmax"], sh, float(st.get("shade_strength", 0.55)), a)
                    else:
                        rgba, _, _ = _rgba_image(g.z, lut, cb["vmin"], cb["vmax"], a)
                    prim.append(("image", lid, rgba, tuple(g.transform())))
            elif kind == "blend":
                from . import blend as B
                ids = [i for i in (lay["src"].get("grids") or []) if i]
                gs = [display_grid(stack, i) for i in ids]
                if len(gs) < 2 or any(g is None for g in gs):
                    raise ValueError("a blend layer needs 2-3 grid layers")
                arrs = [B.on_grid(gs[0], g) for g in gs[:3]]
                rgba = B.rgb_blend(arrs, st.get("mode", "rgb"), st.get("clip", (2.0, 98.0)), st.get("gamma", 1.0),
                                   opacity=a)
                prim.append(("image", lid, rgba, tuple(gs[0].transform())))
            elif kind == "contours":
                g = display_grid(stack, lay["src"].get("grid")) if lay["src"].get("grid") else None
                if g is not None:
                    iv = st.get("interval") or nice_interval(g.z)
                    levels = contour_levels(g.z, iv)
                    ie = int(st.get("index_every") or 0)
                    segs = g.contours(levels)
                    cands = []
                    for lev, (xs, ys) in segs.items():
                        k = int(round(lev / iv)) if iv else 0
                        idx = ie > 0 and k % ie == 0
                        prim.append(("curve", lid, np.asarray(xs, float), np.asarray(ys, float), col(st["color"]),
                                     st["index_width"] if idx else st["width"], "pairs"))
                        if st.get("labels") and (idx or ie <= 0):
                            from .welltie import wc
                            wc()
                            from well_correlation.carto import chain_segments, label_candidates
                            X_, Y_ = g.node_xy()
                            ext = max(np.nanmax(X_) - np.nanmin(X_), np.nanmax(Y_) - np.nanmin(Y_), 1.0)
                            cands += [c + (1,) for c in label_candidates(chain_segments(xs, ys), f"{lev:g}", ext / 5.0)]
                    if cands:
                        from well_correlation.carto import place_labels
                        X_, Y_ = g.node_xy()
                        ext = max(np.nanmax(X_) - np.nanmin(X_), np.nanmax(Y_) - np.nanmin(Y_), 1.0)
                        cw = ext / 140.0 * st.get("label_size", 8) / 8.0
                        for c in place_labels(cands, cw, 1.4 * cw):
                            prim.append(("text", lid, c[0], c[1], c[3], col(st["color"]), st.get("label_size", 8), c[2]))
            elif kind == "polygons":
                for poly in lay["src"].get("polygons", []):
                    x = np.asarray(poly["x"], float)
                    y = np.asarray(poly["y"], float)
                    if len(x) < 2:
                        continue
                    xc, yc = np.append(x, x[0]), np.append(y, y[0])
                    if st.get("fault") and len(x) >= 3:  # fault heave polygon: hatched + downthrown ticks
                        from .welltie import wc
                        wc()
                        from well_correlation.carto import fault_ticks
                        prim.append(("polygon", lid, xc, yc, col(st["color"], 0.6), None, {"hatch": True}))
                        L = float(np.sum(np.hypot(np.diff(xc), np.diff(yc))))
                        dv = _compass(poly.get("down", st.get("down")))
                        ccw = 0.5 * float(np.sum(xc[:-1] * yc[1:] - xc[1:] * yc[:-1])) > 0
                        txs, tys = [], []
                        for k in range(len(xc) - 1):  # ticks only on the edges facing the downthrown side
                            ex, ey = xc[k + 1] - xc[k], yc[k + 1] - yc[k]
                            el = math.hypot(ex, ey) or 1.0
                            nx_, ny_ = (ey / el, -ex / el) if ccw else (-ey / el, ex / el)  # outward normal
                            if dv is not None and nx_ * dv[0] + ny_ * dv[1] < 0.3:
                                continue
                            side = "right" if ccw else "left"
                            tx, ty = fault_ticks([xc[k], xc[k + 1]], [yc[k], yc[k + 1]], side, L / 25.0, L / 60.0)
                            txs.append(tx)
                            tys.append(ty)
                        if txs and sum(len(t) for t in txs):
                            prim.append(("curve", lid, np.concatenate(txs), np.concatenate(tys), col(st["color"]), 1.5, "pairs"))
                    elif st.get("fill") and len(x) >= 3:
                        prim.append(("polygon", lid, xc, yc, col(st["fill"]), None))
                    prim.append(("curve", lid, xc, yc, col(st["color"]), st["width"], "all"))
                    if st.get("labels", True):
                        prim.append(("text", lid, float(np.mean(x)), float(np.mean(y)), poly.get("name", ""), col(st["color"]), 9))
            elif kind == "vector":
                from .gis.vector import vector_primitives, layer_of
                vl = layer_of(stack, lay)
                pix = None
                if vl is not None and st.get("simplify", True) and vl.n_vertices() > 20000:
                    b = vl.bounds()
                    pix = max(b[1] - b[0], b[3] - b[2]) / 3000.0 if b else None
                prim += vector_primitives(stack, lay, None, pix)
            elif kind == "text":
                for t in lay["src"].get("items", []):
                    prim.append(("text", lid, float(t["x"]), float(t["y"]), str(t["text"]), col(st["color"]), st["size"]))
            elif kind == "petro" and st.get("mode") == "pies":
                from .maphub import petro_pies, pie_polygons, COMPONENT_COLORS
                pies = petro_pies(p, lay["src"].get("zone"))
                if pies:
                    xs_ = [q[1] for q in pies]
                    ys_ = [q[2] for q in pies]
                    ext = max(np.ptp(xs_) if len(xs_) > 1 else 0, np.ptp(ys_) if len(ys_) > 1 else 0)
                    rad = st.get("pie_radius") or (0.03 * ext if ext > 0 else 200.0)
                    for (w, x, y, fr) in pies:
                        for comp, px, py in pie_polygons(x, y, fr, rad):
                            prim.append(("polygon", lid, px, py, col(COMPONENT_COLORS.get(comp, (128, 128, 128))),
                                         col((255, 255, 255))))
                        if st.get("labels", True):
                            prim.append(("text", lid, x + rad, y + rad, w, col((0, 0, 0)), 8))
            elif kind == "petro":
                vals = petro_values(p, lay["src"].get("property"), lay["src"].get("zone"))
                cb = stack.colorbar(lid)
                if vals and cb:
                    lut = _lut(cb["lut"], cb["reverse"])
                    v = np.array([q[3] for q in vals], float)
                    k = np.clip((v - cb["vmin"]) / max(1e-12, cb["vmax"] - cb["vmin"]) * 255, 0, 255).astype(int)
                    rgba = [col(tuple(lut[i][:3])) for i in k]
                    size = st["size"]
                    sizes = size * (0.5 + np.clip((v - cb["vmin"]) / max(1e-12, cb["vmax"] - cb["vmin"]), 0, 1)) \
                        if st.get("mode") == "bubbles" else np.full(len(v), size)
                    prim.append(("points", lid, np.array([q[1] for q in vals]), np.array([q[2] for q in vals]), rgba,
                                 sizes, "o"))
                    if st.get("labels", True):
                        for (w, x, y, val) in vals:
                            prim.append(("text", lid, x, y, f"{val:.3g}", col((0, 0, 0)), 8))
        except Exception as e:  # one broken layer must not stop the map
            prim.append(("error", lid, f"{type(e).__name__}: {e}"))
    return prim


def _hex(c):
    if isinstance(c, (tuple, list)):
        return tuple(c)
    c = str(c).lstrip("#")
    if len(c) >= 6:
        return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
    return (0, 0, 0)


def render_png(stack, path, title="", size=(8, 7), dpi=110):
    """Draw the visible layers with matplotlib (PNG / PDF by extension) - quick export and a Qt-free visual check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Affine2D
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    bx, by = [], []
    for q in render(stack):
        kind = q[0]
        if kind == "image":
            m11, m12, m21, m22, dx, dy = q[3]
            rgba = q[2]
            h, w = rgba.shape[:2]
            for cxp, cyp in ((0, 0), (w, 0), (0, h), (w, h)):
                bx.append(m11 * cxp + m21 * cyp + dx)
                by.append(m12 * cxp + m22 * cyp + dy)
            tr = Affine2D(np.array([[m11, m21, dx], [m12, m22, dy], [0, 0, 1]], float)) + ax.transData
            ax.imshow(rgba, origin="lower", extent=(0, rgba.shape[1], 0, rgba.shape[0]), transform=tr,
                      interpolation="nearest")
        elif kind == "curve":
            xs, ys = np.asarray(q[2]), np.asarray(q[3])
            bx.extend(xs[np.isfinite(xs)].tolist())
            by.extend(ys[np.isfinite(ys)].tolist())
            c = np.array(q[4], float) / 255.0
            if q[6] == "pairs":
                n = len(xs) // 2 * 2
                segs = np.stack([np.column_stack([xs[0:n:2], ys[0:n:2]]), np.column_stack([xs[1:n:2], ys[1:n:2]])], 1)
                from matplotlib.collections import LineCollection
                ax.add_collection(LineCollection(segs, colors=[c], linewidths=q[5] * 0.6))
            else:
                ax.plot(xs, ys, color=c, lw=q[5] * 0.6)
        elif kind == "points":
            bx.extend(np.asarray(q[2], float).tolist())
            by.extend(np.asarray(q[3], float).tolist())
            cols = np.array(q[4], float) / 255.0
            ax.scatter(q[2], q[3], s=np.asarray(q[5], float) ** 2 * 0.6, c=cols if cols.ndim == 2 else [cols],
                       edgecolors="w", linewidths=0.5, zorder=5)
        elif kind == "polygon":
            bx.extend(np.asarray(q[2], float).tolist())
            by.extend(np.asarray(q[3], float).tolist())
            hatch = len(q) > 6 and (q[6] or {}).get("hatch")
            fc = np.array(q[4], float) / 255.0
            ax.fill(q[2], q[3], facecolor="none" if hatch else fc, hatch="////" if hatch else None,
                    edgecolor=fc if hatch else ((np.array(q[5], float) / 255.0) if q[5] else "none"), lw=0.8)
        elif kind == "text":
            import matplotlib.patheffects as pe
            ax.text(q[2], q[3], q[4], color=np.array(q[5], float) / 255.0, fontsize=q[6] * 0.8, zorder=6,
                    rotation=q[7] if len(q) > 7 else 0, ha="center" if len(q) > 7 else "left", va="center" if len(q) > 7
                    else "bottom", rotation_mode="anchor", path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    ax.set_aspect("equal")
    if bx:
        x0, x1, y0, y1 = min(bx), max(bx), min(by), max(by)
        pad = 0.05 * max(x1 - x0, y1 - y0, 1.0)
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
