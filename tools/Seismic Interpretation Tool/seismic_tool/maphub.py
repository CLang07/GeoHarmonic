"""Map hub logic (PLAN 21b): petrophysics on the map - mineral pies, the Well Correlation Tool's "Show on map hub"
request, the petro strip of a clicked well, wells inside a drawn polygon (lasso) and the linked cross-plot of zone
values.  Qt-free; the basemap / dialogs call these."""
import csv
import json
import os

import numpy as np

# mineral / fluid colours for pies (fixed so maps are comparable)
COMPONENT_COLORS = {
    "quartz": (242, 214, 75), "calcite": (91, 155, 213), "dolomite": (150, 120, 220), "k-feldspar": (240, 150, 120),
    "illite": (120, 150, 90), "kaolinite": (170, 190, 120), "smectite": (90, 120, 60), "chlorite": (60, 140, 110),
    "anhydrite": (200, 120, 200), "halite": (230, 230, 230), "coal": (30, 30, 30), "pyrite": (200, 180, 0),
    "water": (40, 110, 230), "oil": (40, 170, 40), "gas": (230, 40, 40),
}


def petro_folder(project):
    wp = getattr(project, "well_project", None)
    if not wp:
        return None
    base = wp[: -len(".wcproj.json")] if wp.endswith(".wcproj.json") else wp
    return base + "_petro"


def _read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def petro_pies(project, zone=None):
    """[(well, x, y, {component: fraction})] from mm_summary.csv (mean volumes of the zone; all zones: thickness-
    weighted by Gross_TVD).  Fractions sum to 1."""
    rows = _read_csv(os.path.join(petro_folder(project) or "", "mm_summary.csv"))
    xy = {n: (x, y) for n, x, y in project.wells()}
    acc = {}
    for r in rows:
        if zone and r.get("Zone") != zone:
            continue
        w = r.get("Well")
        if w not in xy:
            continue
        try:
            wt = float(r.get("Gross_TVD") or 1.0)
        except ValueError:
            wt = 1.0
        d = acc.setdefault(w, {})
        for k, v in r.items():
            if k.startswith("V_") and v not in ("", None):
                d[k[2:]] = d.get(k[2:], 0.0) + wt * float(v)
    out = []
    for w, d in acc.items():
        tot = sum(d.values())
        if tot > 0:
            out.append((w, xy[w][0], xy[w][1], {k: v / tot for k, v in d.items()}))
    return out


def pie_polygons(x, y, fractions, radius, n=48):
    """[(component, xs, ys)] wedges of a pie centred at (x, y) (map units), clockwise from north."""
    out = []
    a0 = 0.0
    for comp, f in fractions.items():
        if f <= 0:
            continue
        a1 = a0 + 2 * np.pi * f
        k = max(2, int(np.ceil(n * f)))
        ang = np.linspace(a0, a1, k + 1)
        xs = np.concatenate([[x], x + radius * np.sin(ang), [x]])
        ys = np.concatenate([[y], y + radius * np.cos(ang), [y]])
        out.append((comp, xs, ys))
        a0 = a1
    return out


def read_hub_request(project):
    p = os.path.join(petro_folder(project) or "", "hub_request.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def apply_hub_request(stack, req):
    """Add (or update) the petro layer the Well Correlation Tool asked for; returns the layer id."""
    if not req:
        return None
    prop, zone = req.get("property", "SoPhiH"), req.get("zone")
    for l in stack.layers:
        if l["kind"] == "petro" and l["src"].get("from_request"):
            l["src"].update({"property": prop, "zone": zone})
            l["name"] = f"{prop}" + (f" {zone}" if zone else " (all zones)")
            l["visible"] = True
            stack.save()
            return l["id"]
    lid = stack.add("petro", f"{prop}" + (f" {zone}" if zone else " (all zones)"),
                    src={"property": prop, "zone": zone, "from_request": True})
    stack.save()
    return lid


def point_in_polygon(x, y, px, py):
    """Vectorised even-odd test of points (x, y) against one polygon (px, py)."""
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    px, py = np.asarray(px, float), np.asarray(py, float)
    inside = np.zeros(x.shape, bool)
    n = len(px)
    j = n - 1
    for i in range(n):
        cond = ((py[i] > y) != (py[j] > y))
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i]
        inside ^= cond & (x < xc)
        j = i
    return inside


def wells_in_polygon(project, px, py):
    ws = project.wells()
    if not ws:
        return []
    x = np.array([w[1] for w in ws], float)
    y = np.array([w[2] for w in ws], float)
    m = point_in_polygon(x, y, px, py)
    return [w[0] for w, k in zip(ws, m) if k]


def zone_values(project, zone=None):
    """{well: {column: value}} of the petro summary for one zone (all zones: sums / means as in petro_values)."""
    from .maplayers import petro_values
    rows = _read_csv(os.path.join(petro_folder(project) or "", "summary.csv"))
    cols = [c for c in (rows[0].keys() if rows else []) if c not in ("Well", "Zone")]
    out = {}
    for c in cols:
        for w, _, _, v in petro_values(project, c, zone):
            out.setdefault(w, {})[c] = v
    return out


def petro_crossplot_data(project, prop_x, prop_y, zone=None, selected=None, grid=None):
    """Cross-plot of two zone values per well (prop_x may be 'grid' = value of a MapGrid at the well).
    Returns {"wells", "x", "y", "selected" bool array, "r" (Pearson), "n"}."""
    vals = zone_values(project, zone)
    xy = {n: (x, y) for n, x, y in project.wells()}
    wells, X, Y = [], [], []
    for w, d in vals.items():
        if prop_y not in d:
            continue
        if prop_x == "grid":
            if grid is None or w not in xy:
                continue
            xv = float(grid.sample([xy[w][0]], [xy[w][1]])[0])
        else:
            if prop_x not in d:
                continue
            xv = d[prop_x]
        if not np.isfinite(xv):
            continue
        wells.append(w)
        X.append(xv)
        Y.append(d[prop_y])
    X, Y = np.array(X, float), np.array(Y, float)
    sel = np.array([w in set(selected or []) for w in wells], bool)
    r = float(np.corrcoef(X, Y)[0, 1]) if len(X) > 2 and np.std(X) > 0 and np.std(Y) > 0 else float("nan")
    return {"wells": wells, "x": X, "y": Y, "selected": sel, "r": r, "n": len(wells), "xlabel": prop_x, "ylabel": prop_y}


def petro_strip_data(project, well, families=("GR", "RDEEP", "RHOB", "NPHI")):
    """Logs + petrophysics results of one well vs MD for the strip view: {"md", "tracks": {name: [(label, values,
    colour, log)]}, "tops": [(name, md)], "zones": [summary rows]} or None when the well is not in the project."""
    from .welltie import wc
    wc()
    from well_correlation.project import Project
    from well_correlation.petro.store import well_curves
    if not project.well_project or not os.path.exists(project.well_project):
        return None
    wcp = getattr(project, "_hub_wcp", None)
    if wcp is None or wcp.path != project.well_project:
        wcp = Project.open(project.well_project)
        project._hub_wcp = wcp
    if well not in wcp.wells:
        return None
    w = wcp.load_well(well)
    pc = well_curves(wcp, w)
    L = w.logs
    nan = np.full(len(w.depth), np.nan)
    tracks = {
        "GR": [("GR", L.get("GR", nan), (27, 127, 42), False)],
        "Resistivity": [("RDEEP", L.get("RDEEP", nan), (192, 22, 27), True), ("RSHAL", L.get("RSHAL", nan), (31, 79, 191), True)],
        "Density-neutron": [("RHOB", L.get("RHOB", nan), (192, 22, 27), False), ("NPHI", L.get("NPHI", nan), (31, 79, 191), False)],
        "Vsh": [("VSH", pc.get("VSH", nan), (139, 90, 43), False)],
        "Porosity": [("PHIE", pc.get("PHIE", nan), (31, 79, 191), False), ("BVW", pc.get("BVW", nan), (0, 160, 160), False)],
        "Sw": [("SW", pc.get("SW", nan), (192, 22, 27), False)],
    }
    tops = sorted(((f, md) for f, md in wcp.tops.get(well, {}).items()), key=lambda t: t[1])
    zones = [r for r in _read_csv(os.path.join(petro_folder(project) or "", "summary.csv")) if r.get("Well") == well]
    return {"well": well, "md": w.depth, "tracks": tracks, "tops": tops, "zones": zones, "has_petro": bool(pc)}
