"""GIS export of the map hub (PLAN 21g): grids as GeoTIFF (with overviews), contours / faults / polygons / wells / petro
values as shapefiles + one GeoPackage, and a KML (WGS84) - so final layouts can also be made in QGIS / ArcGIS Pro.

    export_stack(stack, folder, epsg=2193)            -> {"files": [...], "gpkg": path, "kml": path, "warnings": [...]}
    project_crs(project) / set_project_crs(project, epsg=None, wkt=None)
"""
import os
import re

import numpy as np

from . import crs as CRS
from . import geotiff, gpkg, shapefile


def project_crs(project):
    d = project.display if project is not None else {}
    if d.get("crs_epsg"):
        c = CRS.from_epsg(d["crs_epsg"])
        c["epsg"] = int(d["crs_epsg"])
        return c
    if d.get("crs_wkt"):
        return CRS.from_wkt(d["crs_wkt"])
    return None


def set_project_crs(project, epsg=None, wkt=None):
    if epsg:
        CRS.from_epsg(epsg)  # validates
        project.display["crs_epsg"] = int(epsg)
        project.display.pop("crs_wkt", None)
    elif wkt:
        CRS.from_wkt(wkt)
        project.display["crs_wkt"] = wkt
        project.display.pop("crs_epsg", None)
    return project_crs(project)


def _safe(s):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(s)).strip("_") or "layer"


def layer_features(stack, lay):
    """(geometry type, geoms, attrs) of a vector-like layer in map coordinates, or None."""
    from ..maplayers import petro_values
    p = stack.project
    k = lay["kind"]
    if k == "wells" and p is not None:
        ws = p.wells()
        return "Point", [{"type": "Point", "parts": [np.array([[x, y]])]} for _, x, y in ws], {"name": [w[0] for w in ws]}
    if k == "petro" and p is not None:
        vals = petro_values(p, lay["src"].get("property"), lay["src"].get("zone"))
        prop = str(lay["src"].get("property") or "value")[:10]
        return "Point", [{"type": "Point", "parts": [np.array([[x, y]])]} for _, x, y, _ in vals], \
            {"well": [q[0] for q in vals], prop: [q[3] for q in vals]}
    if k == "polygons":
        polys = lay["src"].get("polygons", [])
        lines = [q for q in polys if q.get("open")]
        if lines and len(lines) == len(polys):  # registered section lines (Live Map 31h): polylines
            return "LineString", [{"type": "LineString", "parts": [np.column_stack([q["x"], q["y"]])]} for q in lines], \
                {"name": [q.get("name", "") for q in lines]}
        polys = [q for q in polys if not q.get("open")]
        return "Polygon", [{"type": "Polygon", "parts": [np.column_stack([q["x"], q["y"]])]} for q in polys], \
            {"name": [q.get("name", "") for q in polys]}
    if k == "contours" and lay["src"].get("grid"):
        from ..maplayers import contour_levels, nice_interval, display_grid
        from ..welltie import wc
        wc()
        from well_correlation.carto import chain_segments
        g = display_grid(stack, lay["src"]["grid"])
        if g is None:
            return None
        iv = lay["style"].get("interval") or nice_interval(g.z)
        geoms, levs = [], []
        for lev, (xs, ys) in g.contours(contour_levels(g.z, iv)).items():
            for x, y in chain_segments(xs, ys):
                geoms.append({"type": "LineString", "parts": [np.column_stack([x, y])]})
                levs.append(float(lev))
        return "LineString", geoms, {"level": levs}
    if k == "faults" and p is not None and getattr(p, "faults", None) is not None:
        geoms, names = [], []
        for f in p.faults.items:
            o = f.ordered_sticks()
            if len(o) > 1:
                geoms.append({"type": "LineString", "parts": [np.array([[s[0, 0], s[0, 1]] for s in o])]})
                names.append(f.name)
        return "LineString", geoms, {"name": names}
    if k == "lines2d" and p is not None:
        geoms, names = [], []
        for l in p.lines:
            try:
                line = p.line(l["name"])
            except Exception:
                continue
            if line.has_coords:
                geoms.append({"type": "LineString", "parts": [np.column_stack([line.x, line.y])]})
                names.append(l["name"])
        return "LineString", geoms, {"name": names}
    if k == "outline3d" and p is not None:
        geoms, names = [], []
        for v in p.volumes:
            xs, ys = p.volume(v["name"]).outline()
            geoms.append({"type": "Polygon", "parts": [np.column_stack([xs, ys])]})
            names.append(v["name"])
        return "Polygon", geoms, {"name": names}
    if k == "vector":
        from .vector import layer_of
        vl = layer_of(stack, lay)
        if vl is None:
            return None
        return vl.geom_type(), vl.geoms, vl.attrs
    return None


def export_stack(stack, folder, epsg=None, wkt=None, visible_only=True, kml=True, overviews=(2, 4, 8), geojson=False,
                 zmap=False):
    os.makedirs(folder, exist_ok=True)
    c = CRS.from_epsg(epsg) if epsg else (CRS.from_wkt(wkt) if wkt else project_crs(stack.project))
    if c is not None and epsg:
        c["epsg"] = int(epsg)
    prj = CRS.to_wkt(c) if c is not None else None
    srs_id = int(c.get("epsg") or 99999) if c is not None else -1
    out = {"files": [], "gpkg": None, "kml": None, "warnings": []}
    if c is None:
        out["warnings"].append("no CRS set for the project: files written without a projection (set one first)")
    gp = os.path.join(folder, _safe(getattr(stack.project, "name", "map")) + "_map.gpkg")
    gpkg.remove(gp)
    kml_layers = []
    for lay in stack.ordered():
        if visible_only and not lay["visible"]:
            continue
        name = _safe(lay["name"])
        if lay["kind"] in ("grid", "horizon"):
            from ..maplayers import display_grid
            g = display_grid(stack, lay["id"])
            if g is None:
                continue
            gx = g.to_xy() if g.kind == "ilxl" else g
            path = os.path.join(folder, name + ".tif")
            geotiff.write(path, gx.z, gx.x0, gx.y0, gx.dx, gx.dy, epsg=c.get("epsg") if c else None, wkt=prj,
                          overviews=overviews)
            out["files"].append(path)
            if zmap:  # Live Map 31i: ZMAP+ for Petrel / Kingdom / GeoGraphix
                from .zmap import write_zmap
                out["files"].append(write_zmap(os.path.join(folder, name + ".zmap"), gx, lay["name"]))
            continue
        feats = layer_features(stack, lay)
        if not feats or not feats[1]:
            continue
        gtype, geoms, attrs = feats
        path = shapefile.write(os.path.join(folder, name + ".shp"), geoms, attrs, prj, gtype)
        out["files"].append(path)
        if geojson:  # Live Map 31i
            from .geojson import write_geojson
            out["files"].append(write_geojson(os.path.join(folder, name + ".geojson"), geoms, attrs,
                                              epsg=(c or {}).get("epsg")))
        gpkg.write_layer(gp, name, geoms, attrs, srs_id, prj or "undefined", c["name"] if c else "")
        out["gpkg"] = gp
        if kml and c is not None:
            ll = []
            for g in geoms:
                parts = []
                for p in g["parts"]:
                    lat, lon = CRS.inverse(c, p[:, 0], p[:, 1])
                    lat, lon, approx = CRS.datum_shift(lat, lon, c, CRS.from_epsg(4326))
                    parts.append(np.column_stack([lon, lat]))
                ll.append({"type": g["type"], "parts": parts})
            label = "name" if "name" in attrs else ("well" if "well" in attrs else None)
            kml_layers.append((lay["name"], ll, attrs, label))
    if kml_layers:
        out["kml"] = gpkg.write_kml(os.path.join(folder, _safe(getattr(stack.project, "name", "map")) + ".kml"), kml_layers)
    return out
