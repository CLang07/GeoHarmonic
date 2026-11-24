"""Generic vector layers for the map hub (PLAN 21g "any shapefile"): PLSS, counties, pipelines, leases, permits ...

    vl = VectorLayer.load("plss_sections.shp")      # .shp / .gpkg (first or named table) / .kml / .csv with WKT or X,Y
    vl.reproject(dst_crs)                             # from the .prj / GeoPackage SRS (or a given CRS)
    vl.select("STATE == 'TX' and ACRES > 600")       # attribute filter (safe expression)
    vl.categorized("OPERATOR") / vl.graduated("DIAMETER", 5)
    vl.simplified(tol) (Douglas-Peucker, cached per tolerance), vl.in_view(x0, x1, y0, y1) (grid-bucket index)
    vl.identify(x, y, tol) -> attribute row

Map-hub layer kind "vector": src {"file", "table", "crs_wkt" (source), "cache"}; style {"color", "width", "fill",
"by" (field), "mode" categorized | graduated, "classes", "labels" (field), "filter" (expression), "min_scale"}.
Large layers are cached as npz next to the source (<file>.ghcache.npz) after the first load.
"""
import ast
import csv
import hashlib
import os
import re

import numpy as np

from . import crs as CRS

PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75),
           (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]


class VectorLayer:
    def __init__(self, geoms, attrs=None, crs=None, name=""):
        self.geoms = list(geoms)
        self.attrs = {k: list(v) for k, v in (attrs or {}).items()}
        self.crs = crs
        self.name = name
        self._simpl = {}
        self._index = None

    def __len__(self):
        return len(self.geoms)

    def geom_type(self):
        return next((g["type"] for g in self.geoms if g), "Point")

    # ------------------------------------------------------------------ IO
    @classmethod
    def load(cls, path, table=None, x_field=None, y_field=None):
        ext = os.path.splitext(path)[1].lower()
        name = os.path.splitext(os.path.basename(path))[0]
        crs = None
        if ext == ".shp":
            from .shapefile import read
            geoms, attrs, prj = read(path)
            crs = CRS.from_wkt(prj) if prj else None
        elif ext == ".gpkg":
            from .gpkg import list_layers, read_layer
            table = table or list_layers(path)[0]
            geoms, attrs, wkt = read_layer(path, table)
            crs = CRS.from_wkt(wkt) if wkt else None
            name = table
        elif ext == ".kml":
            from .gpkg import read_kml
            geoms, attrs = read_kml(path)
            crs = CRS.from_epsg(4326)
        elif ext in (".csv", ".txt"):
            geoms, attrs = read_csv(path, x_field, y_field)
        else:
            raise ValueError(f"unsupported vector format {ext} (.shp, .gpkg, .kml, .csv)")
        vl = cls(geoms, attrs, crs, name)
        vl.source = path
        return vl

    def reproject(self, dst):
        """In place to dst CRS; returns the 'approximate datum shift' flag (NAD27)."""
        if self.crs is None or dst is None or CRS.same(self.crs, dst):
            self.crs = dst if self.crs is None else self.crs
            return False
        approx = False
        for g in self.geoms:
            if not g:
                continue
            new = []
            for p in g["parts"]:
                x, y, a = CRS.transform(p[:, 0], p[:, 1], self.crs, dst)
                approx |= a
                new.append(np.column_stack([x, y]))
            g["parts"] = new
        self.crs = dst
        self._simpl, self._index = {}, None
        return approx

    # ------------------------------------------------------------------ attributes
    def fields(self):
        return list(self.attrs)

    def row(self, i):
        return {k: v[i] for k, v in self.attrs.items()}

    def select(self, expr):
        """Boolean mask of the features matching a filter expression (field names, numbers, strings, == != < <= > >=,
        and / or / not, in [...], + - * /).  Empty expression = all."""
        if not expr or not expr.strip():
            return np.ones(len(self), bool)
        tree = ast.parse(expr, mode="eval")
        n = len(self)

        def ev(node):
            if isinstance(node, ast.Expression):
                return ev(node.body)
            if isinstance(node, ast.BoolOp):
                vals = [ev(v) for v in node.values]
                out = vals[0]
                for v in vals[1:]:
                    out = (out & v) if isinstance(node.op, ast.And) else (out | v)
                return out
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                return ~ev(node.operand)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -ev(node.operand)
            if isinstance(node, ast.Compare):
                left = ev(node.left)
                out = np.ones(n, bool)
                for op, comp in zip(node.ops, node.comparators):
                    right = ev(comp)
                    with np.errstate(invalid="ignore"):
                        if isinstance(op, ast.In):
                            r = np.isin(left, right)
                        elif isinstance(op, ast.NotIn):
                            r = ~np.isin(left, right)
                        else:
                            f = {ast.Eq: np.equal, ast.NotEq: np.not_equal, ast.Lt: np.less, ast.LtE: np.less_equal,
                                 ast.Gt: np.greater, ast.GtE: np.greater_equal}[type(op)]
                            try:
                                r = f(left, right)
                            except TypeError:
                                r = np.zeros(n, bool)
                    out &= np.asarray(r, bool)
                    left = right
                return out
            if isinstance(node, ast.BinOp):
                a, b = ev(node.left), ev(node.right)
                f = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide}[type(node.op)]
                return f(a, b)
            if isinstance(node, ast.Name):
                if node.id not in self.attrs:
                    raise ValueError(f"unknown field {node.id}; fields: {', '.join(self.fields())}")
                col = self.attrs[node.id]
                try:
                    return np.array([np.nan if v is None else float(v) for v in col])
                except (TypeError, ValueError):
                    return np.array(["" if v is None else str(v) for v in col], dtype=object)
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, (ast.List, ast.Tuple)):
                return [ev(e) for e in node.elts]
            raise ValueError("not allowed in a filter")

        return np.asarray(ev(tree), bool)

    def subset(self, mask):
        idx = np.nonzero(np.asarray(mask, bool))[0]
        return VectorLayer([self.geoms[i] for i in idx], {k: [v[i] for i in idx] for k, v in self.attrs.items()}, self.crs,
                           self.name)

    # ------------------------------------------------------------------ styling
    def categorized(self, field, palette=PALETTE, max_classes=40):
        vals = [str(v) for v in self.attrs[field]]
        cats = list(dict.fromkeys(vals))[:max_classes]
        cmap = {c: palette[k % len(palette)] for k, c in enumerate(cats)}
        return [cmap.get(v, (150, 150, 150)) for v in vals], cmap

    def graduated(self, field, classes=5, ramp="property"):
        from ..welltie import wc
        wc()
        from well_correlation.carto import ramp_lut
        v = np.array([np.nan if x is None else float(x) for x in self.attrs[field]])
        f = v[np.isfinite(v)]
        edges = np.quantile(f, np.linspace(0, 1, classes + 1)) if f.size else np.linspace(0, 1, classes + 1)
        k = np.clip(np.searchsorted(edges, v, "right") - 1, 0, classes - 1)
        lut = ramp_lut(ramp)
        cols = [tuple(int(c) for c in lut[int(255 * kk / max(1, classes - 1))][:3]) for kk in range(classes)]
        return [cols[kk] if np.isfinite(vv) else (150, 150, 150) for kk, vv in zip(k, v)], (edges, cols)

    # ------------------------------------------------------------------ geometry helpers
    def bounds(self):
        allp = [p for g in self.geoms if g for p in g["parts"] if len(p)]
        if not allp:
            return None
        a = np.concatenate(allp)
        return float(a[:, 0].min()), float(a[:, 0].max()), float(a[:, 1].min()), float(a[:, 1].max())

    def n_vertices(self):
        return int(sum(len(p) for g in self.geoms if g for p in g["parts"]))

    def simplified(self, tol):
        """Douglas-Peucker simplified copy of the geometries (cached per tolerance)."""
        key = float(tol)
        if key in self._simpl:
            return self._simpl[key]
        out = []
        for g in self.geoms:
            if not g or g["type"] in ("Point", "MultiPoint") or tol <= 0:
                out.append(g)
                continue
            out.append({"type": g["type"], "parts": [douglas_peucker(p, tol, closed=g["type"] == "Polygon") for p in g["parts"]]})
        self._simpl[key] = out
        return out

    def _build_index(self):
        b = self.bounds()
        boxes = []
        for g in self.geoms:
            if not g:
                boxes.append((np.nan,) * 4)
                continue
            a = np.concatenate(g["parts"])
            boxes.append((a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 1].max()))
        boxes = np.array(boxes, float).reshape(-1, 4)
        span = max(b[1] - b[0], b[3] - b[2], 1.0) if b else 1.0
        cell = span / 64.0
        buckets = {}
        for i, (x0, x1, y0, y1) in enumerate(boxes):
            if not np.isfinite(x0):
                continue
            for bx in range(int(np.floor(x0 / cell)), int(np.floor(x1 / cell)) + 1):
                for by in range(int(np.floor(y0 / cell)), int(np.floor(y1 / cell)) + 1):
                    buckets.setdefault((bx, by), []).append(i)
        self._index = (cell, buckets, boxes)

    def in_view(self, x0, x1, y0, y1):
        """Indices of the features whose bounding box intersects the view (grid-bucket spatial index)."""
        if self._index is None:
            self._build_index()
        cell, buckets, boxes = self._index
        cand = set()
        for bx in range(int(np.floor(x0 / cell)), int(np.floor(x1 / cell)) + 1):
            for by in range(int(np.floor(y0 / cell)), int(np.floor(y1 / cell)) + 1):
                cand.update(buckets.get((bx, by), ()))
                if len(cand) > 5_000_000:
                    break
        idx = np.array(sorted(cand), int)
        if not len(idx):
            return idx
        b = boxes[idx]
        ok = (b[:, 0] <= x1) & (b[:, 1] >= x0) & (b[:, 2] <= y1) & (b[:, 3] >= y0)
        return idx[ok]

    def identify(self, x, y, tol):
        """Feature under (x, y): polygon containing it, else the nearest line / point within tol. -> (index, row)."""
        from ..maphub import point_in_polygon
        idx = self.in_view(x - tol, x + tol, y - tol, y + tol)
        best = None
        for i in idx:
            g = self.geoms[i]
            if g["type"] == "Polygon":
                from .shapefile import group_rings
                for poly in group_rings(g["parts"]):
                    if point_in_polygon([x], [y], poly[0][:, 0], poly[0][:, 1])[0] and \
                            not any(point_in_polygon([x], [y], h[:, 0], h[:, 1])[0] for h in poly[1:]):
                        return int(i), self.row(i)
            for p in g["parts"]:
                if len(p) == 1:
                    d = float(np.hypot(p[0, 0] - x, p[0, 1] - y))
                else:
                    d = _dist_polyline(p, x, y)
                if d <= tol and (best is None or d < best[0]):
                    best = (d, int(i))
        return (best[1], self.row(best[1])) if best else (None, None)

    def to_cache(self, path):
        parts, kinds, counts = [], [], []
        for g in self.geoms:
            kinds.append(g["type"] if g else "")
            counts.append(len(g["parts"]) if g else 0)
            parts += [np.asarray(p, float) for p in (g["parts"] if g else [])]
        lens = np.array([len(p) for p in parts], int)
        xy = np.concatenate(parts) if parts else np.zeros((0, 2))
        np.savez_compressed(path, xy=xy, lens=lens, kinds=np.array(kinds), counts=np.array(counts),
                            crs=np.array(CRS.to_wkt(self.crs) if self.crs else ""))
        return path

    @classmethod
    def from_cache(cls, path, attrs=None):
        with np.load(path, allow_pickle=False) as z:
            xy, lens, kinds, counts, wkt = z["xy"], z["lens"], z["kinds"], z["counts"], str(z["crs"])
        splits = np.split(xy, np.cumsum(lens)[:-1]) if len(lens) else []
        geoms, k = [], 0
        for kind, c in zip(kinds, counts):
            if not kind:
                geoms.append(None)
                continue
            geoms.append({"type": str(kind), "parts": list(splits[k:k + c])})
            k += c
        return cls(geoms, attrs or {}, CRS.from_wkt(wkt) if wkt else None)


def _dist_polyline(p, x, y):
    a, b = p[:-1], p[1:]
    d = b - a
    L2 = (d ** 2).sum(1)
    t = np.clip(((x - a[:, 0]) * d[:, 0] + (y - a[:, 1]) * d[:, 1]) / np.where(L2 > 0, L2, 1), 0, 1)
    return float(np.min(np.hypot(a[:, 0] + t * d[:, 0] - x, a[:, 1] + t * d[:, 1] - y)))


def douglas_peucker(p, tol, closed=False):
    """Iterative Douglas-Peucker (keeps the end points; closed rings keep >= 4 points)."""
    p = np.asarray(p, float)
    n = len(p)
    if n < 3:
        return p
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = p[i], p[j]
        seg = p[i + 1:j]
        d = b - a
        L = np.hypot(*d)
        if L == 0:
            dist = np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1])
        else:
            dist = np.abs(d[0] * (a[1] - seg[:, 1]) - d[1] * (a[0] - seg[:, 0])) / L
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    out = p[keep]
    if closed and len(out) < 4:
        idx = np.linspace(0, n - 1, min(n, 4)).astype(int)
        out = p[idx]
    return out


# ------------------------------------------------------------------------------------------- CSV / WKT
_WKT_RX = re.compile(r"^\s*(MULTI)?(POINT|LINESTRING|POLYGON)\s*(Z|M|ZM)?\s*\(", re.I)


def parse_wkt_geom(s):
    s = s.strip()
    m = _WKT_RX.match(s)
    if not m:
        return None
    kind = m.group(2).upper()
    body = s[s.index("("):]
    nums = lambda t: np.array([[float(v) for v in pt.split()[:2]] for pt in t.split(",") if pt.strip()])  # noqa: E731
    if kind == "POINT":
        pts = nums(body.replace("(", " ").replace(")", " "))
        return {"type": "Point" if len(pts) == 1 else "MultiPoint", "parts": [pts]}
    rings = re.findall(r"\(([^()]*)\)", body)
    parts = [nums(r) for r in rings]
    return {"type": "LineString" if kind == "LINESTRING" else "Polygon", "parts": parts}


def read_csv(path, x_field=None, y_field=None):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], {}
    cols = list(rows[0])
    wkt_col = next((c for c in cols if c.lower() in ("wkt", "geometry", "geom", "the_geom", "shape")), None)
    if x_field is None:
        x_field = next((c for c in cols if c.lower() in ("x", "easting", "longitude", "lon", "long")), None)
    if y_field is None:
        y_field = next((c for c in cols if c.lower() in ("y", "northing", "latitude", "lat")), None)
    geoms = []
    for r in rows:
        if wkt_col:
            geoms.append(parse_wkt_geom(r[wkt_col] or ""))
        elif x_field and y_field:
            try:
                geoms.append({"type": "Point", "parts": [np.array([[float(r[x_field]), float(r[y_field])]])]})
            except (TypeError, ValueError):
                geoms.append(None)
        else:
            raise ValueError("CSV needs a WKT column or X / Y columns")
    attrs = {c: [] for c in cols if c != wkt_col}
    for r in rows:
        for c in attrs:
            v = r[c]
            try:
                v = float(v) if v not in ("", None) and re.match(r"^-?[\d.]+(e-?\d+)?$", v.strip(), re.I) else v
            except ValueError:
                pass
            attrs[c].append(v)
    return geoms, attrs


# ------------------------------------------------------------------------------------------- map-hub glue
_CACHE = {}


def layer_of(stack, lay):
    """VectorLayer of a map-hub 'vector' layer, reprojected to the project CRS, cached in memory and as npz."""
    from .export import project_crs
    src = lay["src"]
    path = src.get("file")
    if not path or not os.path.exists(path):
        return None
    st = os.stat(path)
    key = (path, st.st_mtime, src.get("table"))
    if key in _CACHE:
        return _CACHE[key]
    vl = VectorLayer.load(path, src.get("table"))
    if vl.crs is None and src.get("crs_wkt"):
        vl.crs = CRS.from_wkt(src["crs_wkt"])
    dst = project_crs(stack.project) if stack.project is not None else None
    if dst is not None and vl.crs is not None:
        lay["src"]["approx_datum"] = bool(vl.reproject(dst))
    _CACHE[key] = vl
    return vl


def cache_path(path):
    h = hashlib.sha1(path.encode()).hexdigest()[:8]
    return f"{path}.{h}.ghcache.npz"


def vector_primitives(stack, lay, view=None, pixel=None):
    """Drawing primitives for a vector layer: filter, colour by field, simplified to ~pixel size, clipped to view."""
    vl = layer_of(stack, lay)
    if vl is None:
        return [("error", lay["id"], f"cannot read {lay['src'].get('file')}")]
    st = lay["style"]
    mask = vl.select(st.get("filter") or "")
    idx = np.nonzero(mask)[0]
    if view is not None:
        idx = np.intersect1d(idx, vl.in_view(*view))
    if st.get("by") and st["by"] in vl.attrs:
        cols = vl.graduated(st["by"], int(st.get("classes") or 5))[0] if st.get("mode") == "graduated" else \
            vl.categorized(st["by"])[0]
    else:
        cols = [tuple(st.get("color") or (80, 80, 80))] * len(vl)
    geoms = vl.simplified(pixel) if pixel else vl.geoms
    a = lay["opacity"]
    prim = []
    lid = lay["id"]
    pts_x, pts_y, pts_c = [], [], []
    for i in idx:
        g = geoms[i]
        if not g:
            continue
        c = cols[i]
        if g["type"] in ("Point", "MultiPoint"):
            p = np.concatenate(g["parts"])
            pts_x += list(p[:, 0])
            pts_y += list(p[:, 1])
            pts_c += [(c[0], c[1], c[2], int(255 * a))] * len(p)
        elif g["type"] == "Polygon":
            fill = st.get("fill")
            falpha = int((60 if fill is True else int(fill or 0)) * a)
            for p in g["parts"]:
                if fill and len(p) >= 3:
                    prim.append(("polygon", lid, p[:, 0], p[:, 1], (c[0], c[1], c[2], falpha), None))
                prim.append(("curve", lid, p[:, 0], p[:, 1], (c[0], c[1], c[2], int(255 * a)), st.get("width", 1.0), "all"))
        else:
            for p in g["parts"]:
                prim.append(("curve", lid, p[:, 0], p[:, 1], (c[0], c[1], c[2], int(255 * a)), st.get("width", 1.0), "all"))
        if st.get("labels") and st["labels"] in vl.attrs:
            p = np.concatenate(g["parts"])
            prim.append(("text", lid, float(p[:, 0].mean()), float(p[:, 1].mean()), str(vl.attrs[st["labels"]][i]),
                         (40, 40, 40, int(255 * a)), 8))
    if pts_x:
        prim.append(("points", lid, np.array(pts_x), np.array(pts_y), pts_c, st.get("size", 6), st.get("symbol", "o")))
    return prim
