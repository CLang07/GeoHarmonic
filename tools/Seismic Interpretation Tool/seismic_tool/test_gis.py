"""Tests for the GIS package (PLAN 21g): projections vs published values, shapefile / GeoTIFF / GeoPackage / KML round
trips, vector filter + styles + simplification + spatial index, PLSS lookup, map-hub export."""
import os
import time

import numpy as np


def run(check, tmp):
    from .gis import crs as C, shapefile as SH, geotiff as GT, gpkg as GP, plss as PL
    from .gis.vector import VectorLayer, douglas_peucker
    d = os.path.join(tmp, "gis")
    os.makedirs(d, exist_ok=True)
    # ---- projections against published examples
    lcc = C.crs("lcc", "Clarke1866", lat0=23, lon0=-96, lat1=33, lat2=45)
    x, y = C.forward(lcc, 35, -75)
    check(abs(x - 1894410.9) < 0.1 and abs(y - 1564649.5) < 0.1, f"21g: LCC = Snyder example ({x:.1f}, {y:.1f})")
    aea = C.crs("albers", "Clarke1866", lat0=23, lon0=-96, lat1=29.5, lat2=45.5)
    x, y = C.forward(aea, 35, -75)
    check(abs(x - 1885472.7) < 0.1 and abs(y - 1535925.0) < 0.1, f"21g: Albers = Snyder example ({x:.1f}, {y:.1f})")
    tm = C.crs("tm", "Airy1830", lat0=49, lon0=-2, k0=0.9996012717, fe=400000, fn=-100000)
    x, y = C.forward(tm, 50.5, 0.5)
    check(abs(x - 577274.99) < 0.05 and abs(y - 69740.50) < 0.05, f"21g: TM = EPSG OSGB example ({x:.2f}, {y:.2f})")
    sp = C.from_epsg(32040)
    x, y = C.forward(sp, 28.5, -96)
    check(abs(x - 2963503.91) < 0.05 and abs(y - 254759.80) < 0.05, f"21g: NAD27 Texas South Central (ftUS) = EPSG example "
          f"({x:.2f}, {y:.2f})")
    from .welltie import wc
    wc()
    from well_correlation.geo import tm_forward
    nz = C.from_epsg(2193)
    e1, n1 = C.forward(nz, -39.6, 175.1)
    e2, n2 = tm_forward(-39.6, 175.1)
    check(abs(e1 - e2) < 1e-6 and abs(n1 - n2) < 1e-6, "21g: NZTM2000 = the Well Correlation Tool's projection")
    lats = np.linspace(-46, -34, 25)
    lons = np.linspace(166.5, 178.5, 25)
    LA, LO = np.meshgrid(lats, lons)
    X, Y = C.forward(nz, LA, LO)
    la2, lo2 = C.inverse(nz, X, Y)
    X2, Y2 = C.forward(nz, la2, lo2)
    err = float(np.max(np.hypot(X2 - X, Y2 - Y)))
    check(err < 0.001, f"21g: inverse TM against forward TM over NZ, max {err * 1000:.3f} mm")
    for c in (lcc, aea, C.from_epsg(5070), C.from_epsg(3857), C.from_epsg(32614)):
        xx, yy = C.forward(c, 31.0, -100.0)
        la, lo = C.inverse(c, xx, yy)
        check(abs(la - 31.0) < 1e-8 and abs(lo + 100.0) < 1e-8, f"21g: {c['name'] or c['kind']} forward / inverse round trip")
    for code in (2193, 32040, 2278, 5070, 4326, 32760):
        c = C.from_epsg(code)
        check(C.same(C.from_wkt(C.to_wkt(c)), c), f"21g: EPSG:{code} -> .prj WKT -> CRS round trip")
    esri = ('PROJCS["NZGD_2000_New_Zealand_Transverse_Mercator",GEOGCS["GCS_NZGD_2000",DATUM["D_NZGD_2000",SPHEROID['
            '"GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION['
            '"Transverse_Mercator"],PARAMETER["False_Easting",1600000.0],PARAMETER["False_Northing",10000000.0],PARAMETER['
            '"Central_Meridian",173.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')
    check(C.same(C.from_wkt(esri), nz), "21g: ESRI .prj of NZTM parsed")
    x27, y27 = C.forward(C.from_epsg(32040), 30.0, -97.0)
    xx, yy, approx = C.transform(x27, y27, C.from_epsg(32040), C.from_epsg(2278))
    xx2, yy2, _ = C.transform(xx, yy, C.from_epsg(2278), C.from_epsg(32040))
    check(approx and abs(xx2 - x27) < 0.01 and abs(yy2 - y27) < 0.01 and abs(xx - x27) > 1000,
          "21g: NAD27 -> NAD83 state plane flagged approximate, reversible")
    # ---- shapefile round trip: points, polylines (multi-part), polygons with holes + attributes
    pts = [{"type": "Point", "parts": [np.array([[1.5, 2.5]])]}, {"type": "Point", "parts": [np.array([[3.0, -4.0]])]}]
    SH.write(os.path.join(d, "pts.shp"), pts, {"name": ["A", "Bé"], "md": [1200, 3400], "phi": [0.21, None],
                                               "gas": [True, False]}, C.to_wkt(nz))
    g, a, prj = SH.read(os.path.join(d, "pts.shp"))
    check(np.allclose(g[1]["parts"][0], [[3.0, -4.0]]) and a["name"] == ["A", "Bé"] and a["md"] == [1200, 3400]
          and abs(a["phi"][0] - 0.21) < 1e-9 and a["phi"][1] is None and a["gas"] == [True, False] and C.same(C.from_wkt(prj), nz),
          "21g: shapefile points + dbf (C / N / F / L) + .prj round trip")
    outer = np.array([[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]], float)
    hole = np.array([[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]], float)
    second = np.array([[20, 0], [20, 5], [25, 5], [20, 0]], float)
    polys = [{"type": "Polygon", "parts": [outer, hole, second]}]
    SH.write(os.path.join(d, "poly.shp"), polys, {"name": ["P"]})
    g, a, _ = SH.read(os.path.join(d, "poly.shp"))
    roles = SH.ring_roles(g[0]["parts"])
    check(len(g[0]["parts"]) == 3 and roles == ["outer", "hole", "outer"] and np.allclose(g[0]["parts"][0], outer),
          "21g: multi-part polygon with a hole (rings oriented outer CW / hole CCW)")
    lines = [{"type": "LineString", "parts": [np.array([[0, 0], [1, 1], [2, 0]]), np.array([[5, 5], [6, 6]])]}]
    SH.write(os.path.join(d, "lines.shp"), lines, {"id": [7]})
    g, a, _ = SH.read(os.path.join(d, "lines.shp"))
    check(len(g[0]["parts"]) == 2 and np.allclose(g[0]["parts"][1], [[5, 5], [6, 6]]) and a["id"] == [7], "21g: polylines")
    # ---- GeoTIFF round trip
    z = np.random.default_rng(1).normal(size=(37, 53)).astype(np.float32).astype(float)
    z[3, 4] = np.nan
    GT.write(os.path.join(d, "g.tif"), z, 1600000.0, 5600000.0, 25.0, 25.0, epsg=2193)
    r = GT.read(os.path.join(d, "g.tif"))
    check(np.array_equal(np.isnan(r["z"]), np.isnan(z)) and np.allclose(np.nan_to_num(r["z"]), np.nan_to_num(z)) and
          abs(r["x0"] - 1600000.0) < 1e-9 and abs(r["y0"] - 5600000.0) < 1e-9 and r["dx"] == 25.0 and r["epsg"] == 2193,
          "21g: GeoTIFF write -> read identical (values, NaN, geometry, EPSG)")
    check(r["overviews"] >= 2, f"21g/h: GeoTIFF internal overviews ({r['overviews']})")
    GT.write(os.path.join(d, "g2.tif"), z, 0.0, 0.0, 10.0, 10.0, wkt=C.to_wkt(lcc), compress=False, overviews=())
    r2 = GT.read(os.path.join(d, "g2.tif"))
    check(r2["epsg"] is None and r2["wkt"] and C.same(C.from_wkt(r2["wkt"]), lcc), "21g: GeoTIFF without EPSG keeps the CRS in .aux.xml")
    # ---- GeoPackage + KML
    gp = os.path.join(d, "t.gpkg")
    GP.write_layer(gp, "wells", pts, {"name": ["A", "B"], "md": [1200, 3400]}, 2193, C.to_wkt(nz), "NZTM")
    GP.write_layer(gp, "polys", polys, {"name": ["P"]}, 2193, C.to_wkt(nz), "NZTM")
    g, a, wkt = GP.read_layer(gp, "polys")
    check(set(GP.list_layers(gp)) == {"wells", "polys"} and len(g[0]["parts"]) == 3 and C.same(C.from_wkt(wkt), nz),
          "21g: GeoPackage layers written and read back (multipolygon with hole, SRS)")
    import sqlite3
    try:
        sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE t USING rtree(id, a, b, c, e)")
        rt_ok = GP.has_rtree(gp, "wells")
    except sqlite3.OperationalError:
        rt_ok = True
        print("SKIP rtree (SQLite built without it)")
    check(rt_ok, "21g/h: GeoPackage R-tree spatial index")
    k = GP.write_kml(os.path.join(d, "t.kml"), [("Wells", [{"type": "Point", "parts": [np.array([[173.3, -39.6]])]}],
                                                  {"name": ["A"]}, "name"), ("Polys", polys, {"name": ["P"]}, "name")])
    kg, ka = GP.read_kml(k)
    check(ka["name"] == ["A", "P"] and np.allclose(kg[0]["parts"][0], [[173.3, -39.6]]) and len(kg[1]["parts"]) == 3,
          "21g: KML written and read back")
    # ---- generic vector layer: filter, categorised / graduated style, CSV WKT, simplification, index
    vl = VectorLayer.load(os.path.join(d, "pts.shp"))
    check(list(vl.select("md > 2000 and name != 'A'")) == [False, True] and list(vl.select("name in ['A']")) == [True, False],
          "21g: attribute filter expressions")
    try:
        vl.select("__import__('os')")
        bad = False
    except Exception:
        bad = True
    check(bad, "21g: filter rejects code")
    cols, cmap = vl.categorized("name")
    check(len(cmap) == 2 and cols[0] != cols[1], "21g: categorised colours by attribute")
    gcols, (edges, _) = vl.graduated("md", 2)
    check(gcols[0] != gcols[1] and len(edges) == 3, "21g: graduated colours")
    with open(os.path.join(d, "leases.csv"), "w") as f:
        f.write("lease,operator,wkt\nL1,Acme,\"POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))\"\nL2,Beta,\"POINT (5 5)\"\n")
    cl = VectorLayer.load(os.path.join(d, "leases.csv"))
    check(cl.geoms[0]["type"] == "Polygon" and cl.geoms[1]["type"] == "Point" and cl.attrs["operator"] == ["Acme", "Beta"],
          "21g: CSV with WKT geometry")
    i, row = cl.identify(0.5, 0.5, 0.1)
    check(i == 0 and row["lease"] == "L1", "21g: identify (click inside a polygon -> its attribute row)")
    # 1M-vertex layer: simplification + view query fast
    n = 1_000_000
    t = np.linspace(0, 400 * np.pi, n)
    big = VectorLayer([{"type": "LineString", "parts": [np.column_stack([t * 100, 1000 * np.sin(t) + 0.1 * np.cos(37 * t)])]}])
    t0 = time.perf_counter()
    s = big.simplified(5.0)
    t_s = time.perf_counter() - t0
    nv = len(s[0]["parts"][0])
    check(nv < n / 10 and t_s < 20.0, f"21g: 1M-vertex line simplified to {nv} vertices in {t_s:.1f} s")
    dp = douglas_peucker(np.array([[0, 0], [1, 0.001], [2, 0], [3, 5], [4, 0]], float), 0.01)
    check(len(dp) == 4, "21g: Douglas-Peucker keeps the significant vertices")
    sq = VectorLayer([{"type": "Point", "parts": [np.array([[float(x), float(y)]])]} for x in range(100) for y in range(100)])
    t0 = time.perf_counter()
    iv = sq.in_view(10.5, 20.5, 30.5, 40.5)
    check(len(iv) == 100 and time.perf_counter() - t0 < 2.0, "21g: grid-bucket view query")
    # ---- PLSS
    geoms, attrs = PL.synthetic_township()
    pl = VectorLayer(geoms, attrs)
    lg = PL.parse_legal("660' FNL & 1980' FWL, Sec. 14, T12N, R3W, 6th P.M.")
    check(lg["section"] == 14 and lg["township"] == 12 and lg["tdir"] == "N" and lg["range"] == 3 and lg["rdir"] == "W"
          and lg["calls"] == [(660.0, "N"), (1980.0, "W")] and lg["meridian"] == "6", "21g: PLSS legal description parsed")
    check(PL.parse_legal("14-12N-3W")["section"] == 14 and PL.parse_legal("S14 T12N R3W")["range"] == 3, "21g: short forms")
    x, y, i = PL.spot(pl, "660 FNL 1980 FWL Sec 14 T12N R3W")
    p = geoms[i]["parts"][0]
    check(attrs["FRSTDIVNO"][i] == 14 and abs(x - (p[:, 0].min() + 1980 * 0.3048)) < 1e-6 and abs(y - (p[:, 1].max() - 660 * 0.3048)) < 1e-6,
          "21g: well spot from section + footage calls")
    mile = 1609.344
    check(abs(geoms[attrs["FRSTDIVNO"].index(1)]["parts"][0][:, 0].min() - 5 * mile) < 1e-6 and
          abs(geoms[attrs["FRSTDIVNO"].index(7)]["parts"][0][:, 0].min()) < 1e-6, "21g: synthetic township numbering (1 at NE, 7 at W)")


def run_export(check, tmp):
    """21g: map-hub export (GeoTIFF + shapefiles + GeoPackage + KML) and vector layers on the map."""
    from . import maplayers as ML
    from .gis import export as EX, gpkg as GP, geotiff as GT, shapefile as SH, crs as C
    from .maps import horizon_grid
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "gisexport")
    EX.set_project_crs(proj, 2193)
    st = ML.LayerStack.load(proj)
    gid, cid = st.add_grid(horizon_grid(proj, proj.horizons.get("Top"), "3D:Synth"), "Top time", interval=10.0)
    st.add("polygons", "Prospects", src={"polygons": [{"name": "P1", "x": [500100, 500300, 500300], "y": [6000100, 6000100, 6000300]}]})
    out = EX.export_stack(st, os.path.join(d, "gis_out"))
    names = sorted(os.path.basename(f) for f in out["files"])
    check("Top_time.tif" in names and "Top_time_contours.shp" in names and "Prospects.shp" in names and "Wells.shp" in names,
          f"21g: export writes GeoTIFF + shapefiles ({len(names)} files)")
    r = GT.read(os.path.join(d, "gis_out", "Top_time.tif"))
    g = st.grid(gid).to_xy()
    check(r["epsg"] == 2193 and np.allclose(np.nan_to_num(r["z"]), np.nan_to_num(g.z.astype(np.float32)), atol=1e-3),
          "21g: grid exported as GeoTIFF with its CRS")
    lays = GP.list_layers(out["gpkg"])
    gg, aa, _ = SH.read(os.path.join(d, "gis_out", "Top_time_contours.shp"))
    check({"Wells", "Prospects", "Top_time_contours"} <= set(lays) and gg and all(v % 10 == 0 for v in aa["level"]),
          "21g: GeoPackage with all vector layers; contours carry their level")
    check(out["kml"] and os.path.getsize(out["kml"]) > 500, "21g: KML (WGS84) written")
    # a vector layer from a shapefile in another CRS (NZGD2000 geographic) shown on the NZTM map
    wpts = [{"type": "Point", "parts": [np.array([[173.3, -39.6]])]}, {"type": "Point", "parts": [np.array([[173.4, -39.7]])]}]
    shp = SH.write(os.path.join(d, "ll.shp"), wpts, {"name": ["a", "b"], "op": ["X", "Y"]}, C.to_wkt(C.from_epsg(4167)))
    vid = st.add("vector", "Lease points", src={"file": shp}, style={"by": "op", "labels": "name"})
    prims = [q for q in ML.render(st) if q[1] == vid]
    pts = next(q for q in prims if q[0] == "points")
    e, n = C.forward(C.from_epsg(2193), -39.6, 173.3)
    check(abs(pts[2][0] - e) < 1e-6 and abs(pts[3][0] - n) < 1e-6 and len({tuple(c) for c in pts[4]}) == 2
          and any(q[0] == "text" for q in prims), "21g: vector layer reprojected to the project CRS, coloured by field, labelled")


def run_gui(check, tmp):
    from .gui.basemap import Basemap
    from .gui.layers_dock import LayersPanel
    from . import maplayers as ML
    from . import mapexport as MX
    from .gis import shapefile as SH, crs as C, plss as PL
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "gis_gui")
    bm = Basemap()
    bm.project = proj
    st = ML.LayerStack.load(proj)
    bm.set_stack(st)
    panel = LayersPanel(bm)
    panel.set_stack(st)
    c = panel.set_crs(2193)
    check(c["name"].startswith("NZGD2000") and proj.display["crs_epsg"] == 2193, "21g GUI: project CRS set")
    geoms, attrs = PL.synthetic_township(500000.0, 6000000.0)
    shp = SH.write(os.path.join(d, "plss.shp"), geoms, attrs, C.to_wkt(C.from_epsg(2193)))
    lid = panel.add_vector(shp, style={"by": "FRSTDIVNO", "fill": 40})
    check(not bm.layer_errors and any(isinstance(it, object) for it in bm._layer_items), "21g GUI: vector layer drawn")
    xy = panel.goto_legal("660 FNL 1980 FWL Sec 14 T12N R3W")
    check(xy is not None and any(l["name"] == "Legal descriptions" for l in st.layers), "21g GUI: go to legal description")
    hit = panel.identify(xy[0], xy[1], 10.0)
    check(hit is not None and hit[2]["FRSTDIVNO"] == 14, "21g GUI: identify on click -> attribute row (section 14)")
    fields, rows = panel.attribute_rows(lid, "FRSTDIVNO <= 6")
    check("RANGENO" in fields and len(rows) == 6, "21g GUI: attribute table with a filter")
    out = panel.export_gis(os.path.join(d, "gis"))
    check(out["gpkg"] and any(f.endswith("plss.shp") for f in out["files"]), "21g GUI: export for QGIS / ArcGIS")
    MX.save_template(proj, "A3 print", {"page": "A3 landscape", "author": "Chris", "graticule": True})
    tpl = MX.load_template(proj, "A3 print")
    png = os.path.join(d, "layout.png")
    res = panel.export_map(png, title="PLSS test", dpi=50, **{k: v for k, v in tpl.items() if k in ("page", "author", "graticule")})
    check(os.path.exists(png) and "locator" in res["boxes"], "21g GUI: page layout from a saved template (locator, graticule)")
    gl = MX.graticule_lines(c, 1700000, 1800000, 5550000, 5650000)
    check(gl["lat"] and gl["lon"] and gl["step"] <= 0.5, f"21g: lat / long graticule ({len(gl['lat'])} x {len(gl['lon'])} "
          f"lines, {gl['step'] * 60:.0f}')")
