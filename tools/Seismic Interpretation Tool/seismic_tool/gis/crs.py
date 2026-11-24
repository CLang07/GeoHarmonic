"""Coordinate reference systems in numpy (PLAN 21g): geographic, transverse Mercator (UTM, NZTM, state plane TM),
Lambert conformal conic (1SP / 2SP, most US state plane zones), Albers equal area, spherical (web) Mercator.  Formulas:
Snyder, Map Projections - A Working Manual (USGS PP 1395).  WKT (.prj, OGC / ESRI) in and out, a table of common EPSG
codes, and datum shifts by a 3-parameter geocentric translation: WGS84 = NAD83 = NZGD2000 (GRS80, < 2 m apart - not
shifted); NAD27 -> NAD83 uses the NADCON-free CONUS mean (-8, 160, 176) m and is FLAGGED APPROXIMATE (errors of a few
metres; use NADCON grids in a GIS for survey work).

    c = from_epsg(2193); x, y = forward(c, lat, lon); lat, lon = inverse(c, x, y)
    x2, y2 = transform(x, y, src, dst)
"""
import math
import re

import numpy as np

ELLIPSOIDS = {"GRS80": (6378137.0, 298.257222101), "WGS84": (6378137.0, 298.257223563),
              "Clarke1866": (6378206.4, 294.9786982), "International1924": (6378388.0, 297.0),
              "Airy1830": (6377563.396, 299.3249646), "Bessel1841": (6377397.155, 299.1528128)}
FT_US = 0.3048006096012192
FT = 0.3048
NAD27_TO_WGS84 = (-8.0, 160.0, 176.0)


def crs(kind, ellps="GRS80", unit=1.0, name="", datum="WGS84", towgs84=None, **p):
    a, rf = ELLIPSOIDS[ellps] if isinstance(ellps, str) else ellps
    c = {"kind": kind, "a": float(a), "rf": float(rf), "unit": float(unit), "name": name or kind, "datum": datum,
         "towgs84": towgs84, "lat0": 0.0, "lon0": 0.0, "k0": 1.0, "fe": 0.0, "fn": 0.0, "lat1": None, "lat2": None}
    c.update({k: float(v) if v is not None else None for k, v in p.items()})
    return c


def _e(c):
    f = 1.0 / c["rf"] if c["rf"] else 0.0
    e2 = f * (2 - f)
    return e2, math.sqrt(e2)


# ------------------------------------------------------------------------------------------------- TM
def _M(c, phi):
    a = c["a"]
    e2, _ = _e(c)
    e4, e6 = e2 * e2, e2 ** 3
    return a * ((1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * phi)
                + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * phi) - (35 * e6 / 3072) * np.sin(6 * phi))


def _tm_fwd(c, lat, lon):
    a, k0 = c["a"], c["k0"]
    e2, _ = _e(c)
    ep2 = e2 / (1 - e2)
    phi = np.radians(lat)
    lam = np.radians(lon - c["lon0"])
    lam = (lam + np.pi) % (2 * np.pi) - np.pi
    N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    T = np.tan(phi) ** 2
    C = ep2 * np.cos(phi) ** 2
    A = lam * np.cos(phi)
    M, M0 = _M(c, phi), _M(c, math.radians(c["lat0"]))
    x = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T * T + 72 * C - 58 * ep2) * A ** 5 / 120)
    y = k0 * (M - M0 + N * np.tan(phi) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
                                          + (61 - 58 * T + T * T + 600 * C - 330 * ep2) * A ** 6 / 720))
    return x, y


def _tm_inv(c, x, y):
    a, k0 = c["a"], c["k0"]
    e2, _ = _e(c)
    ep2 = e2 / (1 - e2)
    M = _M(c, math.radians(c["lat0"])) + y / k0
    mu = M / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    p1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu) + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
          + (151 * e1 ** 3 / 96) * np.sin(6 * mu) + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    C1 = ep2 * np.cos(p1) ** 2
    T1 = np.tan(p1) ** 2
    N1 = a / np.sqrt(1 - e2 * np.sin(p1) ** 2)
    R1 = a * (1 - e2) / (1 - e2 * np.sin(p1) ** 2) ** 1.5
    D = x / (N1 * k0)
    phi = p1 - (N1 * np.tan(p1) / R1) * (D * D / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * ep2) * D ** 4 / 24
                                         + (61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252 * ep2 - 3 * C1 * C1) * D ** 6 / 720)
    lam = (D - (1 + 2 * T1 + C1) * D ** 3 / 6 + (5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * ep2 + 24 * T1 * T1) * D ** 5 / 120) \
        / np.cos(p1)
    return np.degrees(phi), c["lon0"] + np.degrees(lam)


# ------------------------------------------------------------------------------------------------- conics
def _m(e2, phi):
    return np.cos(phi) / np.sqrt(1 - e2 * np.sin(phi) ** 2)


def _t(e, phi):
    return np.tan(np.pi / 4 - phi / 2) / ((1 - e * np.sin(phi)) / (1 + e * np.sin(phi))) ** (e / 2)


def _lcc_consts(c):
    e2, e = _e(c)
    p0 = math.radians(c["lat0"])
    if c.get("lat1") is not None and c.get("lat2") is not None and abs(c["lat1"] - c["lat2"]) > 1e-12:
        p1, p2 = math.radians(c["lat1"]), math.radians(c["lat2"])
        m1, m2 = _m(e2, p1), _m(e2, p2)
        t1, t2 = _t(e, p1), _t(e, p2)
        n = (math.log(m1) - math.log(m2)) / (math.log(t1) - math.log(t2))
        F = m1 / (n * t1 ** n)
        k = 1.0
    else:  # 1SP: lat1 = lat0 with scale k0
        p1 = math.radians(c["lat1"] if c.get("lat1") is not None else c["lat0"])
        n = math.sin(p1)
        F = _m(e2, p1) / (n * _t(e, p1) ** n)
        k = c["k0"]
    rho0 = c["a"] * F * k * _t(e, p0) ** n
    return n, F * k, rho0, e2, e


def _lcc_fwd(c, lat, lon):
    n, F, rho0, e2, e = _lcc_consts(c)
    phi = np.radians(lat)
    rho = c["a"] * F * _t(e, phi) ** n
    th = n * np.radians(lon - c["lon0"])
    return rho * np.sin(th), rho0 - rho * np.cos(th)


def _lcc_inv(c, x, y):
    n, F, rho0, e2, e = _lcc_consts(c)
    s = 1.0 if n > 0 else -1.0
    rho = s * np.sqrt(x ** 2 + (rho0 - y) ** 2)
    t = (rho / (c["a"] * F)) ** (1 / n)
    th = np.arctan2(s * x, s * (rho0 - y))
    lam = th / n
    phi = np.pi / 2 - 2 * np.arctan(t)
    for _ in range(15):
        phi = np.pi / 2 - 2 * np.arctan(t * ((1 - e * np.sin(phi)) / (1 + e * np.sin(phi))) ** (e / 2))
    return np.degrees(phi), c["lon0"] + np.degrees(lam)


def _q(e2, e, phi):
    s = np.sin(phi)
    return (1 - e2) * (s / (1 - e2 * s * s) - (1 / (2 * e)) * np.log((1 - e * s) / (1 + e * s)))


def _aea_consts(c):
    e2, e = _e(c)
    p0, p1, p2 = (math.radians(c[k]) for k in ("lat0", "lat1", "lat2"))
    m1, m2 = _m(e2, p1), _m(e2, p2)
    q0, q1, q2 = _q(e2, e, p0), _q(e2, e, p1), _q(e2, e, p2)
    n = (m1 * m1 - m2 * m2) / (q2 - q1)
    C = m1 * m1 + n * q1
    rho0 = c["a"] * math.sqrt(C - n * q0) / n
    return n, C, rho0, e2, e


def _aea_fwd(c, lat, lon):
    n, C, rho0, e2, e = _aea_consts(c)
    rho = c["a"] * np.sqrt(C - n * _q(e2, e, np.radians(lat))) / n
    th = n * np.radians(lon - c["lon0"])
    return rho * np.sin(th), rho0 - rho * np.cos(th)


def _aea_inv(c, x, y):
    n, C, rho0, e2, e = _aea_consts(c)
    rho = np.sqrt(x ** 2 + (rho0 - y) ** 2)
    q = (C - (rho * n / c["a"]) ** 2) / n
    th = np.arctan2(x, rho0 - y)
    phi = np.arcsin(np.clip(q / 2, -1, 1))
    for _ in range(20):
        s = np.sin(phi)
        phi = phi + (1 - e2 * s * s) ** 2 / (2 * np.cos(phi)) * (
            q / (1 - e2) - s / (1 - e2 * s * s) + (1 / (2 * e)) * np.log((1 - e * s) / (1 + e * s)))
    return np.degrees(phi), c["lon0"] + np.degrees(th / n)


# ------------------------------------------------------------------------------------------------- dispatch
def forward(c, lat, lon):
    """Geographic (degrees, on the CRS datum) -> projected coordinates in the CRS linear unit."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    k = c["kind"]
    if k == "geographic":
        return lon.copy(), lat.copy()
    if k == "tm":
        x, y = _tm_fwd(c, lat, lon)
    elif k == "lcc":
        x, y = _lcc_fwd(c, lat, lon)
    elif k == "albers":
        x, y = _aea_fwd(c, lat, lon)
    elif k == "merc":
        R = c["a"]
        x = R * np.radians(lon - c["lon0"])
        y = R * np.log(np.tan(np.pi / 4 + np.radians(np.clip(lat, -85.06, 85.06)) / 2))
    else:
        raise ValueError(f"unsupported projection {k}")
    u = c["unit"]
    return (x + c["fe"] * u) / u, (y + c["fn"] * u) / u


def inverse(c, x, y):
    """Projected (CRS unit) -> geographic (lat, lon degrees on the CRS datum)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = c["kind"]
    if k == "geographic":
        return y.copy(), x.copy()
    u = c["unit"]
    xm, ym = x * u - c["fe"] * u, y * u - c["fn"] * u
    if k == "tm":  # series inverse + Newton steps on the forward series: exact round trip, also far from the meridian
        lat, lon = _tm_inv(c, xm, ym)
        for _ in range(3):
            fx, fy = _tm_fwd(c, lat, lon)
            h = 1e-6
            ax_, ay_ = _tm_fwd(c, lat + h, lon)
            bx_, by_ = _tm_fwd(c, lat, lon + h)
            j11, j12, j21, j22 = (ax_ - fx) / h, (bx_ - fx) / h, (ay_ - fy) / h, (by_ - fy) / h
            det = j11 * j22 - j12 * j21
            rx, ry = xm - fx, ym - fy
            with np.errstate(invalid="ignore", divide="ignore"):
                dlat = (j22 * rx - j12 * ry) / det
                dlon = (-j21 * rx + j11 * ry) / det
            ok = np.isfinite(dlat) & np.isfinite(dlon) & (np.abs(dlat) < 1) & (np.abs(dlon) < 1)
            lat = np.where(ok, lat + np.where(ok, dlat, 0), lat)
            lon = np.where(ok, lon + np.where(ok, dlon, 0), lon)
        return lat, lon
    if k == "lcc":
        return _lcc_inv(c, xm, ym)
    if k == "albers":
        return _aea_inv(c, xm, ym)
    if k == "merc":
        R = c["a"]
        return np.degrees(2 * np.arctan(np.exp(ym / R)) - np.pi / 2), c["lon0"] + np.degrees(xm / R)
    raise ValueError(f"unsupported projection {k}")


def _ecef(c, lat, lon, h=0.0):
    e2, _ = _e(c)
    phi, lam = np.radians(lat), np.radians(lon)
    N = c["a"] / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    return ((N + h) * np.cos(phi) * np.cos(lam), (N + h) * np.cos(phi) * np.sin(lam), (N * (1 - e2) + h) * np.sin(phi))


def _geodetic(c, X, Y, Z):
    e2, _ = _e(c)
    a = c["a"]
    lon = np.arctan2(Y, X)
    p = np.hypot(X, Y)
    phi = np.arctan2(Z, p * (1 - e2))
    for _ in range(8):
        N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
        h = p / np.cos(phi) - N
        phi = np.arctan2(Z, p * (1 - e2 * N / (N + h)))
    return np.degrees(phi), np.degrees(lon)


def datum_shift(lat, lon, src, dst):
    """Geographic coordinates from src datum to dst datum (3-parameter geocentric translation via WGS84).
    Returns (lat, lon, approximate flag)."""
    s, d = src.get("towgs84"), dst.get("towgs84")
    if (s is None or s == (0, 0, 0)) and (d is None or d == (0, 0, 0)) and src["a"] == dst["a"]:
        return lat, lon, False
    if (s or (0, 0, 0)) == (d or (0, 0, 0)) and abs(src["a"] - dst["a"]) < 1e-6 and abs(src["rf"] - dst["rf"]) < 1e-6:
        return lat, lon, False
    X, Y, Z = _ecef(src, lat, lon)
    s = s or (0.0, 0.0, 0.0)
    d = d or (0.0, 0.0, 0.0)
    X, Y, Z = X + s[0] - d[0], Y + s[1] - d[1], Z + s[2] - d[2]
    la, lo = _geodetic(dst, X, Y, Z)
    return la, lo, bool(tuple(s) != (0, 0, 0) or tuple(d) != (0, 0, 0))


def transform(x, y, src, dst):
    """Coordinates from one CRS to another (via geographic + datum shift).  Returns (x, y, approximate flag)."""
    lat, lon = inverse(src, x, y)
    lat, lon, approx = datum_shift(lat, lon, src, dst)
    X, Y = forward(dst, lat, lon)
    return X, Y, approx


# ------------------------------------------------------------------------------------------------- EPSG + WKT
def utm(zone, south=False, datum="WGS84"):
    e = "GRS80" if datum in ("NAD83", "GRS80") else ("Clarke1866" if datum == "NAD27" else "WGS84")
    tw = NAD27_TO_WGS84 if datum == "NAD27" else None
    return crs("tm", e, name=f"{datum} / UTM zone {zone}{'S' if south else 'N'}", datum=datum, towgs84=tw,
               lon0=-183 + 6 * zone, k0=0.9996, fe=500000.0, fn=10000000.0 if south else 0.0)


def from_epsg(code):
    code = int(code)
    if code == 4326:
        return crs("geographic", "WGS84", name="WGS 84", datum="WGS84")
    if code == 4269:
        return crs("geographic", "GRS80", name="NAD83", datum="NAD83")
    if code == 4267:
        return crs("geographic", "Clarke1866", name="NAD27", datum="NAD27", towgs84=NAD27_TO_WGS84)
    if code == 4167:
        return crs("geographic", "GRS80", name="NZGD2000", datum="NZGD2000")
    if code == 2193:
        return crs("tm", "GRS80", name="NZGD2000 / New Zealand Transverse Mercator 2000", datum="NZGD2000", lon0=173.0,
                   k0=0.9996, fe=1600000.0, fn=10000000.0)
    if code == 3857:
        return crs("merc", "WGS84", name="WGS 84 / Pseudo-Mercator", datum="WGS84")
    if code == 5070:
        return crs("albers", "GRS80", name="NAD83 / Conus Albers", datum="NAD83", lat0=23.0, lon0=-96.0, lat1=29.5, lat2=45.5)
    if code == 32139:
        return crs("lcc", "GRS80", name="NAD83 / Texas South Central", datum="NAD83", lat0=27.8333333333333,
                   lon0=-99.0, lat1=30.2833333333333, lat2=28.3833333333333, fe=600000.0, fn=4000000.0)
    if code == 2278:
        return crs("lcc", "GRS80", unit=FT_US, name="NAD83 / Texas South Central (ftUS)", datum="NAD83",
                   lat0=27.8333333333333, lon0=-99.0, lat1=30.2833333333333, lat2=28.3833333333333, fe=1968500.0,
                   fn=13123333.333)
    if code == 32040:
        return crs("lcc", "Clarke1866", unit=FT_US, name="NAD27 / Texas South Central", datum="NAD27",
                   towgs84=NAD27_TO_WGS84, lat0=27.8333333333333, lon0=-99.0, lat1=28.3833333333333,
                   lat2=30.2833333333333, fe=2000000.0, fn=0.0)
    if 32601 <= code <= 32660:
        return utm(code - 32600)
    if 32701 <= code <= 32760:
        return utm(code - 32700, south=True)
    if 26901 <= code <= 26923:
        return utm(code - 26900, datum="NAD83")
    if 26701 <= code <= 26722:
        return utm(code - 26700, datum="NAD27")
    raise KeyError(f"EPSG:{code} not in the built-in table - load the layer's .prj instead")


EPSG_TABLE = [4326, 4269, 4267, 4167, 2193, 3857, 5070, 32139, 2278, 32040] + list(range(32601, 32661)) + \
    list(range(32701, 32761)) + list(range(26901, 26924)) + list(range(26701, 26723))

_PROJ_NAMES = {"transverse_mercator": "tm", "gauss_kruger": "tm", "lambert_conformal_conic": "lcc",
               "lambert_conformal_conic_2sp": "lcc", "lambert_conformal_conic_1sp": "lcc",
               "albers": "albers", "albers_conic_equal_area": "albers", "mercator_auxiliary_sphere": "merc",
               "popular_visualisation_pseudo_mercator": "merc", "mercator_1sp": "merc", "mercator": "merc"}


def from_wkt(text):
    """CRS from OGC / ESRI WKT1 (.prj)."""
    t = text.strip()
    sph = re.search(r'SPHEROID\s*\[\s*"([^"]*)"\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)', t, re.I)
    a, rf = (float(sph.group(2)), float(sph.group(3))) if sph else ELLIPSOIDS["WGS84"]
    dat = re.search(r'DATUM\s*\[\s*"([^"]*)"', t, re.I)
    datum = (dat.group(1) if dat else "WGS84").upper()
    tw = re.search(r"TOWGS84\s*\[([^\]]*)\]", t, re.I)
    towgs84 = None
    if tw:
        vals = [float(v) for v in tw.group(1).split(",")[:3]]
        towgs84 = tuple(vals) if any(vals) else None
    elif "1927" in datum or "NAD27" in datum.replace("_", "") or "NORTH_AMERICAN_1927" in datum:
        towgs84 = NAD27_TO_WGS84
    dname = "NAD27" if towgs84 == NAD27_TO_WGS84 else ("NAD83" if "1983" in datum else datum.title())
    name_m = re.search(r'^(?:PROJCS|GEOGCS|PROJCRS|GEOGCRS)\s*\[\s*"([^"]*)"', t, re.I)
    name = name_m.group(1) if name_m else ""
    if not re.match(r"^\s*PROJ", t, re.I):
        return crs("geographic", (a, rf), name=name, datum=dname, towgs84=towgs84)
    pm = re.search(r'PROJECTION\s*\[\s*"([^"]*)"', t, re.I)
    pname = (pm.group(1) if pm else "").lower()
    kind = _PROJ_NAMES.get(pname)
    if kind is None:
        raise ValueError(f"projection {pname!r} not supported (TM, LCC, Albers, Mercator)")
    params = {m.group(1).lower(): float(m.group(2)) for m in
              re.finditer(r'PARAMETER\s*\[\s*"([^"]*)"\s*,\s*([-\d.eE+]+)', t, re.I)}
    units = re.findall(r'UNIT\s*\[\s*"([^"]*)"\s*,\s*([-\d.eE+]+)', t, re.I)
    unit = float(units[-1][1]) if units else 1.0
    if unit < 1e-3:  # angular unit picked up by mistake
        unit = 1.0
    g = lambda *names, d=0.0: next((params[n] for n in names if n in params), d)  # noqa: E731
    lat1 = g("standard_parallel_1", d=None)
    lat2 = g("standard_parallel_2", d=None)
    if kind == "lcc" and lat2 is None and lat1 is not None:
        lat2 = None
    return crs(kind, (a, rf), unit=unit, name=name, datum=dname, towgs84=towgs84,
               lat0=g("latitude_of_origin", "latitude_of_center", "standard_parallel_1" if kind == "merc" else "_none"),
               lon0=g("central_meridian", "longitude_of_center", "longitude_of_origin"),
               k0=g("scale_factor", d=1.0), fe=g("false_easting"), fn=g("false_northing"), lat1=lat1, lat2=lat2)


def to_wkt(c):
    """ESRI-style WKT1 for a .prj file."""
    a, rf = c["a"], c["rf"]
    ell = next((k for k, v in ELLIPSOIDS.items() if abs(v[0] - a) < 1e-6 and abs(v[1] - rf) < 1e-6), "Unknown")
    datum = {"NAD27": "D_North_American_1927", "NAD83": "D_North_American_1983", "NZGD2000": "D_NZGD_2000"}.get(
        c["datum"], "D_WGS_1984")
    geog = (f'GEOGCS["GCS_{c["datum"]}",DATUM["{datum}",SPHEROID["{ell}",{a},{rf}]'
            + (f',TOWGS84[{c["towgs84"][0]},{c["towgs84"][1]},{c["towgs84"][2]},0,0,0,0]' if c.get("towgs84") else "")
            + '],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')
    if c["kind"] == "geographic":
        return geog
    pname = {"tm": "Transverse_Mercator", "lcc": "Lambert_Conformal_Conic", "albers": "Albers",
             "merc": "Mercator_Auxiliary_Sphere"}[c["kind"]]
    ps = [("False_Easting", c["fe"]), ("False_Northing", c["fn"]), ("Central_Meridian", c["lon0"])]
    if c["kind"] == "tm":
        ps += [("Scale_Factor", c["k0"]), ("Latitude_Of_Origin", c["lat0"])]
    elif c["kind"] in ("lcc", "albers"):
        ps += [("Standard_Parallel_1", c["lat1"] if c["lat1"] is not None else c["lat0"]),
               ("Standard_Parallel_2", c["lat2"] if c["lat2"] is not None else (c["lat1"] if c["lat1"] is not None else c["lat0"])),
               ("Scale_Factor", c["k0"]), ("Latitude_Of_Origin", c["lat0"])]
    else:
        ps += [("Standard_Parallel_1", 0.0), ("Auxiliary_Sphere_Type", 0.0)]
    uname = "Foot_US" if abs(c["unit"] - FT_US) < 1e-12 else ("Foot" if abs(c["unit"] - FT) < 1e-12 else "Meter")
    name = re.sub(r"[^A-Za-z0-9_]+", "_", c.get("name") or pname)
    return (f'PROJCS["{name}",{geog},PROJECTION["{pname}"],' + ",".join(f'PARAMETER["{k}",{v!r}]' for k, v in ps)
            + f',UNIT["{uname}",{c["unit"]}]]')


def same(c1, c2, tol=1e-9):
    keys = ("kind", "a", "rf", "unit", "lat0", "lon0", "k0", "fe", "fn", "lat1", "lat2")
    for k in keys:
        a, b = c1.get(k), c2.get(k)
        if isinstance(a, float) or isinstance(b, float):
            if (a is None) != (b is None) or (a is not None and abs(a - b) > tol * max(1.0, abs(a))):
                return False
        elif a != b:
            return False
    return (c1.get("towgs84") or (0, 0, 0)) == (c2.get("towgs84") or (0, 0, 0))
