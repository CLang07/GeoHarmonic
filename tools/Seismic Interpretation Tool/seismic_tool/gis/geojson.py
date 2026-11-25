"""GeoJSON read / write (PLAN 31c / 31i) in the gis geometry format {"type": Point | MultiPoint | LineString |
Polygon, "parts": [(n, 2) arrays]} used by shapefile.py / gpkg.py / vector.py.  stdlib json + numpy only.

RFC 7946 GeoJSON is WGS84 lon / lat; older files may carry a "crs" member ("EPSG:2193" or
"urn:ogc:def:crs:EPSG::2193") - read_geojson returns that EPSG code (else 4326) so the map hub can reproject.
Polygons: outer rings are written counter-clockwise and holes clockwise (RFC 7946); on reading they are re-oriented
to the shapefile convention (outer clockwise) so ring_roles / fills work as for shapefiles.
"""
import json
import re

import numpy as np


def _area2(p):
    x, y = p[:, 0], p[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _ring(coords, outer):
    p = np.asarray(coords, float)[:, :2]
    if len(p) and not np.allclose(p[0], p[-1]):
        p = np.vstack([p, p[:1]])
    a = _area2(p)
    if (outer and a > 0) or (not outer and a < 0):  # shapefile: outer clockwise (negative area)
        p = p[::-1].copy()
    return p


def _geom(g):
    if g is None:
        return None
    t, c = g.get("type"), g.get("coordinates")
    if t == "Point":
        return {"type": "Point", "parts": [np.asarray([c[:2]], float)]}
    if t == "MultiPoint":
        return {"type": "MultiPoint", "parts": [np.asarray([q[:2] for q in c], float)]}
    if t == "LineString":
        return {"type": "LineString", "parts": [np.asarray(c, float)[:, :2]]}
    if t == "MultiLineString":
        return {"type": "LineString", "parts": [np.asarray(l, float)[:, :2] for l in c if len(l) >= 2]}
    if t == "Polygon":
        return {"type": "Polygon", "parts": [_ring(r, k == 0) for k, r in enumerate(c) if len(r) >= 3]}
    if t == "MultiPolygon":
        return {"type": "Polygon", "parts": [_ring(r, k == 0) for poly in c for k, r in enumerate(poly) if len(r) >= 3]}
    if t == "GeometryCollection":
        gs = [_geom(q) for q in g.get("geometries") or []]
        gs = [q for q in gs if q]
        return gs[0] if gs else None
    raise ValueError(f"GeoJSON: unsupported geometry type {t!r}")


def crs_epsg(doc):
    """EPSG code from a legacy 'crs' member (None when absent = RFC 7946 WGS84)."""
    c = (doc or {}).get("crs") or {}
    name = str(((c.get("properties") or {}).get("name")) or "")
    m = re.search(r"EPSG:+(\d+)", name)
    if m:
        return int(m.group(1))
    if "CRS84" in name:
        return 4326
    return None


def read_geojson(path):
    """-> (geoms, attrs {field: [values]}, epsg)."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("type") == "FeatureCollection":
        feats = doc.get("features") or []
    elif doc.get("type") == "Feature":
        feats = [doc]
    else:  # a bare geometry
        feats = [{"type": "Feature", "geometry": doc, "properties": {}}]
    geoms, props = [], []
    for ft in feats:
        geoms.append(_geom(ft.get("geometry")))
        props.append(ft.get("properties") or {})
    fields = []
    for p in props:
        for k in p:
            if k not in fields:
                fields.append(k)
    attrs = {k: [p.get(k) for p in props] for k in fields}
    return geoms, attrs, crs_epsg(doc) or 4326


def _coords(a):
    return [[round(float(x), 9), round(float(y), 9)] for x, y in np.asarray(a, float)[:, :2]]


def _rfc_ring(p, outer):
    p = np.asarray(p, float)
    a = _area2(p)
    if (outer and a < 0) or (not outer and a > 0):  # RFC 7946: outer counter-clockwise
        p = p[::-1]
    return _coords(p)


def to_geometry(g):
    if not g:
        return None
    t, parts = g["type"], g["parts"]
    if t == "Point":
        return {"type": "Point", "coordinates": _coords(parts[0])[0]}
    if t == "MultiPoint":
        return {"type": "MultiPoint", "coordinates": _coords(np.concatenate(parts))}
    if t == "LineString":
        if len(parts) == 1:
            return {"type": "LineString", "coordinates": _coords(parts[0])}
        return {"type": "MultiLineString", "coordinates": [_coords(p) for p in parts]}
    if t == "Polygon":
        from .shapefile import group_rings
        polys = group_rings(parts)
        out = [[_rfc_ring(r, k == 0) for k, r in enumerate(poly)] for poly in polys]
        return {"type": "Polygon", "coordinates": out[0]} if len(out) == 1 else \
            {"type": "MultiPolygon", "coordinates": out}
    raise ValueError(f"GeoJSON: cannot write {t}")


def _jsonable(v):
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def write_geojson(path, geoms, attrs=None, epsg=None):
    """Write a FeatureCollection.  epsg other than 4326 / None adds a legacy 'crs' member (QGIS / ArcGIS read it)."""
    attrs = attrs or {}
    feats = []
    for i, g in enumerate(geoms):
        feats.append({"type": "Feature", "geometry": to_geometry(g),
                      "properties": {k: _jsonable(v[i]) for k, v in attrs.items()}})
    doc = {"type": "FeatureCollection", "features": feats}
    if epsg and int(epsg) != 4326:
        doc["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{int(epsg)}"}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return path
