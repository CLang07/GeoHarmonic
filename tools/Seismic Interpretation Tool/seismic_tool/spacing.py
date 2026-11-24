"""Reservoir-engineering foundations: data readers, well spacing, drainage ellipses, gun-barrel view (PLAN 29).

29a  read_public_csv: Texas RRC, NDIC and FracFocus CSV exports (column aliases -> one schema: api, name, operator,
     first_prod, x / y or lat / lon, lateral length, proppant, fluid, TVD, zone); synthetic_field (unconventional pad
     development with parents and children, for tests); stress: SHmax azimuth constant, from a World Stress Map CSV
     (nearest / circular-mean within a radius) or a grid.
29b  lateral_segment (heel-toe of the horizontal part), well_distances (horizontal distance between parallel laterals
     over their overlapping length + vertical offset, same / adjacent zone), neighbours, parent / child / co-completed
     classification by first-production date, wells per section / DSU polygon.
29c  drainage_ellipse (half-length along SHmax, width across; from completion size or user), ellipse polygons for the
     map hub, overlap areas between neighbours.
29d  gun_barrel: section perpendicular to the laterals (lateral offset vs TVD) with the wells, their ellipse cross-
     sections and the zones; the section line for the map strip.
"""
import csv
import math
import os

import numpy as np

ALIASES = {
    "api": ("api", "api_no", "api number", "apinumber", "api14", "api_14", "uwi", "api_wellno"),
    "name": ("well name", "well_name", "wellname", "lease_name", "well"),
    "operator": ("operator", "operator name", "operator_name", "current operator", "company"),
    "first_prod": ("first prod", "first_prod", "first production date", "first_prod_date", "firstproddate",
                   "spud date", "jobstartdate", "job start date", "completion date", "completion_date"),
    "lat": ("lat", "latitude", "surface latitude", "surface_lat", "latitude83"),
    "lon": ("lon", "long", "longitude", "surface longitude", "surface_long", "longitude83"),
    "x": ("x", "easting", "surface_x"),
    "y": ("y", "northing", "surface_y"),
    "lateral": ("lateral length", "lateral_length", "perf interval", "perforated interval", "lateral_ft",
                "lat_len"),
    "proppant": ("proppant", "total proppant", "proppant_lbs", "totalbaseproppant", "proppant (lbs)"),
    "fluid": ("fluid", "total fluid", "totalbasewatervolume", "total base water volume", "fluid_gal",
              "water (gal)"),
    "tvd": ("tvd", "true vertical depth", "truevertical depth", "tvd_ft"),
    "zone": ("zone", "formation", "target formation", "reservoir", "producing formation", "pool"),
}


def _norm(s):
    return "".join(str(s).strip().lower().replace("_", " ").split())  # case, spaces and underscores ignored


def read_public_csv(path, source=None):
    """Rows of an RRC / NDIC / FracFocus (or similar) CSV mapped to the common schema; unknown columns are kept
    under their original names.  Numbers parsed where possible; dates kept as text (YYYY-MM-DD sortable)."""
    amap = {}
    for k, al in ALIASES.items():
        for a in al:
            amap[_norm(a)] = k
    out = []
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            row = {"source": source or os.path.basename(path)}
            for col, v in r.items():
                key = amap.get(_norm(col or ""), col)
                if key in row and row[key] not in ("", None):
                    continue
                v = (v or "").strip()
                if key in ("lat", "lon", "x", "y", "lateral", "proppant", "fluid", "tvd"):
                    try:
                        v = float(v.replace(",", "")) if v else None
                    except ValueError:
                        v = None
                elif key == "first_prod" and v:
                    v = _iso_date(v)
                row[key] = v
            out.append(row)
    return out


def _iso_date(s):
    s = s.strip().split(" ")[0]
    if "/" in s:
        p = s.split("/")
        if len(p) == 3:
            m, d, y = p
            if len(y) == 2:
                y = "20" + y
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return s


def synthetic_field(n_pads=3, wells_per_pad=4, spacing=200.0, lateral=2500.0, azimuth=0.0, zones=("A", "B"),
                    zone_tvd=(2500.0, 2560.0), seed=0, child_delay_days=900):
    """Unconventional development for tests: pads along x, wells alternating zones (wine-rack), first pad producing
    first (parents), later pads (children) years later.  Returns ([Trajectory], {name: info})."""
    from .laterals import Trajectory
    rng = np.random.default_rng(seed)
    trajs, info = [], {}
    az = math.radians(azimuth)
    ux, uy = math.sin(az), math.cos(az)
    vx, vy = math.cos(az), -math.sin(az)  # across the laterals
    k = 0
    for p in range(n_pads):
        for w in range(wells_per_pad):
            off = (p * wells_per_pad + w) * spacing
            zone = zones[w % len(zones)]
            tvd = zone_tvd[w % len(zones)] + rng.normal(0, 3)
            s = np.arange(0, lateral + 1, 50.0)
            x0, y0 = off * vx, off * vy
            md = np.r_[0.0, tvd, tvd + s + 1]
            x = np.r_[x0, x0, x0 + s * ux]
            y = np.r_[y0, y0, y0 + s * uy]
            z = np.r_[0.0, tvd, np.full(len(s), tvd)]
            inc = np.r_[0.0, 0.0, np.full(len(s), 90.0)]
            k += 1
            name = f"P{p + 1}-{w + 1}"
            trajs.append(Trajectory(name, md[:len(x)], x, y, z, inc, api=f"33-000-{k:05d}"))
            info[name] = {"zone": zone, "first_prod": f"{2015 + p * child_delay_days // 365:04d}-0{1 + w % 9}-01",
                          "proppant": float(rng.uniform(1500, 2500) * lateral / 0.3048), "lateral": lateral}
    return trajs, info


# ---------------------------------------------------------------- stress
def shmax_from_wsm(rows, x, y, radius=50000.0, min_quality="C"):
    """SHmax azimuth at (x, y) from World Stress Map rows [{x, y, azi, quality}]: circular mean (axial data, angles
    doubled) of the points within radius with quality A..min_quality; nearest point if none.  Returns degrees 0-180."""
    q_ok = "ABCDE"[: "ABCDE".index(min_quality) + 1]
    pts = [r for r in rows if str(r.get("quality", "A")).upper()[:1] in q_ok]
    if not pts:
        return None
    d = np.array([math.hypot(r["x"] - x, r["y"] - y) for r in pts])
    sel = d <= radius
    if not sel.any():
        sel = d == d.min()
    a = np.radians(2 * np.array([r["azi"] for r in pts])[sel])
    w = 1.0 / np.maximum(d[sel], 1.0)
    m = math.degrees(math.atan2(np.sum(w * np.sin(a)), np.sum(w * np.cos(a)))) / 2
    return m % 180.0


# ====================================================================== 29b
def lateral_segment(traj, min_inc=80.0):
    """(heel xyz, toe xyz) of the horizontal part (inc >= min_inc, or z within 30 m of the deepest point)."""
    inc = traj.inc
    m = inc >= min_inc if inc is not None else traj.z >= traj.z.max() - 30
    if not m.any():
        m = traj.z >= traj.z.max() - 30
    i = np.nonzero(m)[0]
    h = np.array([traj.x[i[0]], traj.y[i[0]], traj.z[i[0]]])
    t = np.array([traj.x[i[-1]], traj.y[i[-1]], traj.z[i[-1]]])
    return h, t


def pair_distance(seg_a, seg_b):
    """Horizontal perpendicular distance, vertical offset and along-lateral overlap (m) between two lateral
    segments (heel, toe)."""
    ha, ta = seg_a
    hb, tb = seg_b
    u = (ta - ha)[:2]
    L = float(np.linalg.norm(u))
    if L == 0:
        return float(np.linalg.norm((hb - ha)[:2])), float(hb[2] - ha[2]), 0.0
    u = u / L
    nrm = np.array([-u[1], u[0]])
    mid_b = 0.5 * (hb + tb)
    horiz = abs(float(nrm @ (mid_b[:2] - ha[:2])))
    sb = sorted([float(u @ (hb[:2] - ha[:2])), float(u @ (tb[:2] - ha[:2]))])
    overlap = max(0.0, min(L, sb[1]) - max(0.0, sb[0]))
    vert = float(mid_b[2] - 0.5 * (ha[2] + ta[2]))
    return horiz, vert, overlap


def well_distances(trajs, zones=None, min_overlap=0.3):
    """All pairs of laterals overlapping by >= min_overlap of the shorter one: rows {a, b, horizontal, vertical,
    distance_3d, overlap, same_zone}."""
    segs = {t.name: lateral_segment(t) for t in trajs}
    rows = []
    names = list(segs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            h, v, ov = pair_distance(segs[a], segs[b])
            la = float(np.linalg.norm((segs[a][1] - segs[a][0])[:2]))
            lb = float(np.linalg.norm((segs[b][1] - segs[b][0])[:2]))
            if ov < min_overlap * max(1.0, min(la, lb)):
                continue
            rows.append({"a": a, "b": b, "horizontal": h, "vertical": v, "distance_3d": math.hypot(h, v), "overlap": ov,
                         "same_zone": (zones or {}).get(a) == (zones or {}).get(b) if zones else None})
    return rows


def nearest_neighbours(rows, name, same_zone_only=False):
    """Closest neighbour each side ('left' / 'right' are not distinguished here): sorted list of (other, horizontal,
    vertical) for a well."""
    out = []
    for r in rows:
        if name in (r["a"], r["b"]) and (not same_zone_only or r["same_zone"]):
            o = r["b"] if r["a"] == name else r["a"]
            out.append((o, r["horizontal"], r["vertical"] if r["a"] == name else -r["vertical"]))
    return sorted(out, key=lambda q: math.hypot(q[1], q[2]))


def classify_parent_child(first_prod, rows, co_days=90, max_dist=600.0):
    """Parent / child / co-completed / standalone per well: a well is a child when a neighbour within max_dist started
    producing more than co_days earlier; parent when such a later neighbour exists; co-completed when neighbours
    started within co_days.  first_prod {name: 'YYYY-MM-DD'}."""
    import datetime as dt

    def day(s):
        return dt.date.fromisoformat(str(s)[:10]).toordinal()
    out = {n: set() for n in first_prod}
    for r in rows:
        if r["distance_3d"] > max_dist or r["a"] not in first_prod or r["b"] not in first_prod:
            continue
        da, db = day(first_prod[r["a"]]), day(first_prod[r["b"]])
        if abs(da - db) <= co_days:
            out[r["a"]].add("co-completed")
            out[r["b"]].add("co-completed")
        elif da < db:
            out[r["a"]].add("parent")
            out[r["b"]].add("child")
        else:
            out[r["b"]].add("parent")
            out[r["a"]].add("child")
    res = {}
    for n, s in out.items():
        res[n] = "child" if "child" in s else ("parent" if "parent" in s else ("co-completed" if s else "standalone"))
    return res


def wells_in_polygons(trajs, polygons):
    """Wells whose lateral midpoint is inside each polygon (sections / DSUs): {poly name: [wells]}."""
    from .maphub import point_in_polygon
    out = {}
    for p in polygons:
        xs, ys = np.asarray(p["x"], float), np.asarray(p["y"], float)
        inside = []
        for t in trajs:
            h, to = lateral_segment(t)
            m = 0.5 * (h + to)
            if point_in_polygon(float(m[0]), float(m[1]), xs, ys):
                inside.append(t.name)
        out[p["name"]] = inside
    return out


# ====================================================================== 29c
def drainage_ellipse(traj, shmax_az, half_length=None, width=None, proppant_lbs=None, k_len=0.08, aspect=0.35):
    """Drainage ellipse of a lateral: centre at the lateral midpoint, axis 'a' along SHmax (fractures grow along
    SHmax), axis 'b' across it.  Each axis = the lateral's half-extent projected on it + the frac half-length (along
    SHmax) or the drainage width (across).  half_length default from proppant per ft: k_len * sqrt(lbs/ft) * 10 m;
    width default aspect x half-length.  Returns {cx, cy, z, a, b, az, half_length, width}."""
    h, t = lateral_segment(traj)
    c = 0.5 * (h + t)
    L = float(np.linalg.norm((t - h)[:2]))
    lat_az = math.degrees(math.atan2(t[0] - h[0], t[1] - h[1])) % 180.0
    if half_length is None:
        ppf = (proppant_lbs or 2000.0 * L / 0.3048) / max(1.0, L / 0.3048)
        half_length = k_len * math.sqrt(ppf) * 10.0
    width = width if width is not None else aspect * half_length
    ang = math.radians((lat_az - shmax_az) % 180.0)
    # project the lateral onto the SHmax axes: extent along SHmax and across it
    along = abs(L / 2 * math.cos(ang)) + half_length
    across = abs(L / 2 * math.sin(ang)) + width
    return {"cx": float(c[0]), "cy": float(c[1]), "a": along, "b": across, "az": float(shmax_az % 180.0),
            "z": float(c[2]), "half_length": half_length, "width": width}


def ellipse_polygon(e, n=72):
    """(xs, ys) of an ellipse dict (major axis 'a' along azimuth 'az' clockwise from north)."""
    t = np.linspace(0, 2 * np.pi, n)
    az = math.radians(e["az"])
    ua = np.array([math.sin(az), math.cos(az)])
    ub = np.array([math.cos(az), -math.sin(az)])
    pts = e["a"] * np.cos(t)[:, None] * ua + e["b"] * np.sin(t)[:, None] * ub
    return e["cx"] + pts[:, 0], e["cy"] + pts[:, 1]


def ellipse_overlap(e1, e2, n=200):
    """Overlap area (m2) of two ellipses by sampling a grid over their bounding box."""
    xs = [ellipse_polygon(e, 36)[0] for e in (e1, e2)]
    ys = [ellipse_polygon(e, 36)[1] for e in (e1, e2)]
    x0, x1 = min(np.min(a) for a in xs), max(np.max(a) for a in xs)
    y0, y1 = min(np.min(a) for a in ys), max(np.max(a) for a in ys)
    X, Y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))

    def inside(e):
        az = math.radians(e["az"])
        dx, dy = X - e["cx"], Y - e["cy"]
        pa = dx * math.sin(az) + dy * math.cos(az)
        pb = dx * math.cos(az) - dy * math.sin(az)
        return (pa / e["a"]) ** 2 + (pb / e["b"]) ** 2 <= 1
    cell = (x1 - x0) / (n - 1) * (y1 - y0) / (n - 1)
    return float(np.sum(inside(e1) & inside(e2)) * cell)


# ====================================================================== 29d
def gun_barrel(trajs, origin=None, azimuth=None, ellipses=None, zones=None):
    """Gun-barrel view: wells projected on a section perpendicular to the laterals.  azimuth: lateral direction
    (default: mean of the laterals); offsets measured along the section direction (azimuth + 90) from origin
    (default: first well's midpoint).  Returns {wells: [{name, offset, tvd, zone}], line: (xs, ys) for the map strip,
    ellipses: [{name, offset, half_width, tvd}], az_section}."""
    segs = {t.name: lateral_segment(t) for t in trajs}
    if azimuth is None:
        az = [math.degrees(math.atan2(*(s[1] - s[0])[:2])) % 180.0 for s in segs.values()]
        a2 = np.radians(2 * np.array(az))
        azimuth = (math.degrees(math.atan2(np.mean(np.sin(a2)), np.mean(np.cos(a2)))) / 2) % 180.0
    sec_az = math.radians(azimuth + 90.0)
    u = np.array([math.sin(sec_az), math.cos(sec_az)])
    mids = {n: 0.5 * (s[0] + s[1]) for n, s in segs.items()}
    o = np.asarray(origin, float) if origin is not None else mids[trajs[0].name][:2]
    wells = []
    for n, m in mids.items():
        wells.append({"name": n, "offset": float(u @ (m[:2] - o)), "tvd": float(m[2]), "zone": (zones or {}).get(n)})
    wells.sort(key=lambda w: w["offset"])
    offs = [w["offset"] for w in wells]
    lo, hi = min(offs) - 200.0, max(offs) + 200.0
    line = (np.array([o[0] + lo * u[0], o[0] + hi * u[0]]), np.array([o[1] + lo * u[1], o[1] + hi * u[1]]))
    ell = []
    for e in ellipses or []:
        # half-width of the ellipse along the section direction
        az = math.radians(e["az"])
        ua = np.array([math.sin(az), math.cos(az)])
        ub = np.array([math.cos(az), -math.sin(az)])
        hw = math.sqrt((e["a"] * (ua @ u)) ** 2 + (e["b"] * (ub @ u)) ** 2)
        ell.append({"name": e.get("name"), "offset": float(u @ (np.array([e["cx"], e["cy"]]) - o)), "half_width": hw,
                    "tvd": e.get("z")})
    return {"wells": wells, "line": line, "ellipses": ell, "az_section": (azimuth + 90.0) % 360.0}
