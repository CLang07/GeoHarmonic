"""Migration and seal (PLAN 27).  Pure numpy.

27a  Map-based buoyant migration on a (paleo-)depth grid (smaller = shallower): steepest-ascent flow paths (D8),
     drainage areas per trap (local crest), spill points by priority flooding (union of basins in depth order),
     fill-spill of charge through the trap chain, critical-moment numbers (charge after trap formation).
27b  Fault seal: juxtaposition along a fault trace (sand-on-sand windows), shale gouge ratio from a Vsh log and the
     throw, SGR -> across-fault pressure / supported column (Yielding-type calibration, editable), slip and dilation
     tendency from a stress tensor.
27c  Top seal: capillary entry pressure (MICP conversion or Vsh lookup), supported column, limiting column =
     min(spill, fault seal, top seal).
"""
import math

import numpy as np

NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ====================================================================== 27a
def receivers(depth):
    """D8 up-dip receiver of every node: the shallowest neighbour if shallower than the node (flat index), else -1
    (a crest / local trap).  NaN nodes -> -2."""
    z = np.asarray(depth, float)
    ny, nx = z.shape
    zp = np.pad(z, 1, constant_values=np.inf)
    zp = np.where(np.isnan(zp), np.inf, zp)
    best = np.full((ny, nx), np.inf)
    rec = np.full((ny, nx), -1, np.int64)
    I, J = np.mgrid[0:ny, 0:nx]
    for di, dj in NB8:
        nb = zp[1 + di:1 + di + ny, 1 + dj:1 + dj + nx]
        d = math.hypot(di, dj)
        slope = (z - nb) / d  # positive = neighbour shallower
        better = (nb < z) & (slope > np.where(np.isfinite(best), best, -np.inf))
        better &= (I + di >= 0) & (I + di < ny) & (J + dj >= 0) & (J + dj < nx)
        best = np.where(better, slope, best)
        rec = np.where(better, (I + di) * nx + (J + dj), rec)
    rec[np.isnan(z)] = -2
    return rec


def crest_of(rec):
    """Crest (local trap node, flat index) every node drains to by following receivers."""
    r = rec.ravel()
    n = len(r)
    out = np.arange(n)
    out[r == -2] = -1
    cur = np.where(r >= 0, r, np.arange(n))
    for _ in range(n):
        nxt = np.where(r[cur] >= 0, r[cur], cur)
        if np.array_equal(nxt, cur):
            break
        cur = nxt
    out = np.where(r == -2, -1, cur)
    return out.reshape(rec.shape)


def flow_path(rec, start):
    """Nodes (flat indices) from start up-dip to its crest."""
    r = rec.ravel()
    path = [int(start)]
    while r[path[-1]] >= 0 and len(path) < len(r):
        path.append(int(r[path[-1]]))
    return path


def is_edge(shape):
    e = np.zeros(shape, bool)
    e[0, :] = e[-1, :] = e[:, 0] = e[:, -1] = True
    return e


def spill_analysis(depth, cell_area=1.0):
    """Traps (crests that are not on the grid edge) with their spill depth, spill node, where they spill to (another
    trap's crest or -1 = off the map / leak) and closure, by priority flooding: nodes in increasing depth, basins
    merge at saddles.  Returns {crest: {depth, spill, spill_node, to, area_nodes, capacity}} + "drain" grid."""
    z = np.asarray(depth, float)
    ny, nx = z.shape
    rec = receivers(z)
    drain = crest_of(rec)
    flat = z.ravel()
    order = np.argsort(np.where(np.isfinite(flat), flat, np.inf), kind="stable")
    parent = np.full(len(flat), -1, np.int64)
    rep = {}  # root -> representative crest (shallowest)
    edge = is_edge(z.shape).ravel()
    traps = {}
    for c in np.unique(drain[drain >= 0]):
        if not edge[c]:
            traps[int(c)] = {"depth": float(flat[c]), "spill": None, "spill_node": None, "to": None}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    open_edge = set()  # roots whose region touches the edge
    for idx in order:
        if not np.isfinite(flat[idx]):
            break
        parent[idx] = idx
        rep[idx] = int(drain.ravel()[idx]) if drain.ravel()[idx] >= 0 else int(idx)
        if edge[idx]:
            open_edge.add(idx)
        i, j = divmod(int(idx), nx)
        for di, dj in NB8:
            a, b = i + di, j + dj
            if not (0 <= a < ny and 0 <= b < nx):
                continue
            k = a * nx + b
            if parent[k] < 0:
                continue
            ra, rb = find(idx), find(k)
            if ra == rb:
                continue
            ca, cb = rep[ra], rep[rb]
            ea, eb = ra in open_edge, rb in open_edge
            if ca != cb:  # two basins meet at a saddle: the closed one(s) spill here into the other
                for r_, c_, other_c in ((ra, ca, cb), (rb, cb, ca)):
                    if c_ in traps and traps[c_]["spill"] is None and r_ not in open_edge:
                        traps[c_].update(spill=float(flat[idx]), spill_node=int(idx),
                                         to=(other_c if other_c in traps else -1))
            new_rep = ca if flat[ca] <= flat[cb] else cb
            parent[rb] = ra
            rep[ra] = new_rep
            if ea or eb:
                open_edge.add(ra)
        r0 = find(idx)
        if edge[idx] and rep[r0] in traps and traps[rep[r0]]["spill"] is None:
            traps[rep[r0]].update(spill=float(flat[idx]), spill_node=int(idx), to=-1)
        if edge[idx]:
            open_edge.add(r0)
    # closure areas and capacities (bulk volume above the spill depth, connected to the crest)
    for c, t in traps.items():
        if t["spill"] is None:
            t["spill"] = float(np.nanmax(flat))
            t["to"] = -1
        m = _closure_mask(z, c, t["spill"])
        t["area_nodes"] = int(m.sum())
        t["capacity"] = float(np.sum((t["spill"] - z[m])) * cell_area)
        t["closure"] = t["spill"] - t["depth"]
    return {"traps": traps, "drain": drain, "rec": rec}


def _closure_mask(z, crest, level):
    """Nodes connected to the crest with depth < level (flood fill)."""
    ny, nx = z.shape
    m = np.zeros(z.shape, bool)
    stack = [divmod(int(crest), nx)]
    while stack:
        i, j = stack.pop()
        if m[i, j] or not (z[i, j] < level):
            continue
        m[i, j] = True
        for di, dj in NB8:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and not m[a, b]:
                stack.append((a, b))
    return m


def fill_spill(analysis, charge):
    """Distribute charge volumes {crest: volume} (e.g. drainage area x expelled volume per area) through the trap
    chain: each trap keeps up to its capacity, the excess spills to its 'to' trap (deepest crests first), -1 = lost.
    Returns {crest: {"charge", "received", "fill", "spilled", "fill_frac"}} and the lost volume."""
    traps = analysis["traps"]
    rec = {c: 0.0 for c in traps}
    out = {}
    lost = 0.0
    for c in sorted(traps, key=lambda c: -traps[c]["depth"]):  # deepest crest first: spill goes up-dip
        t = traps[c]
        total = float(charge.get(c, 0.0)) + rec[c]
        fill = min(t["capacity"], total)
        ex = total - fill
        if ex > 0:
            if t["to"] in traps:
                rec[t["to"]] += ex
            else:
                lost += ex
        out[c] = {"charge": float(charge.get(c, 0.0)), "received": rec[c], "fill": fill, "spilled": ex,
                  "fill_frac": fill / t["capacity"] if t["capacity"] > 0 else 0.0}
    return out, lost


def drainage_charge(analysis, per_node):
    """Charge per trap = sum of per_node (grid of expelled volume per node, or a scalar) over its drainage area."""
    drain = analysis["drain"]
    pn = np.broadcast_to(np.asarray(per_node, float), drain.shape)
    return {c: float(np.sum(pn[drain == c])) for c in analysis["traps"]}


def critical_moment(trap_age, ages, tr, migration_lag=0.0):
    """Fraction of the source rock's transformation (TR curve tr(ages), ages Ma old -> young, TR 0..1) that happens
    after the trap formed (trap_age Ma), plus timing numbers for the event chart."""
    ages = np.asarray(ages, float)
    tr = np.asarray(tr, float)
    o = np.argsort(-ages)
    ages, tr = ages[o], tr[o]
    tr_now = float(tr[-1])
    tr_trap = float(np.interp(-(trap_age - migration_lag), -ages, tr))
    f = (tr_now - tr_trap) / tr_now if tr_now > 0 else 0.0
    onset = ages[np.argmax(tr >= 0.1)] if np.any(tr >= 0.1) else None
    peak = ages[np.argmax(tr >= 0.5)] if np.any(tr >= 0.5) else None
    return {"after_trap": float(np.clip(f, 0, 1)), "tr_at_trap": tr_trap, "tr_now": tr_now, "onset_ma": onset,
            "peak_ma": peak, "trap_ma": float(trap_age),
            "verdict": "timing OK" if f > 0.5 else ("partial" if f > 0.1 else "trap too late")}


# ====================================================================== 27b
def juxtaposition(z, fw_tops, hw_tops, sand_units):
    """Sand-on-sand windows along the fault at depths z: fw_tops / hw_tops {unit: (top, base)} on each side;
    sand_units: names of reservoir units.  Returns (bool array: sand against sand at z, list of (unit_fw, unit_hw))."""
    z = np.asarray(z, float)
    fw_s = np.zeros(len(z), bool)
    hw_s = np.zeros(len(z), bool)
    for u, (a, b) in fw_tops.items():
        if u in sand_units:
            fw_s |= (z >= a) & (z < b)
    for u, (a, b) in hw_tops.items():
        if u in sand_units:
            hw_s |= (z >= a) & (z < b)
    pairs = []
    for u, (a, b) in fw_tops.items():
        for v, (c, d) in hw_tops.items():
            if u in sand_units and v in sand_units and max(a, c) < min(b, d):
                pairs.append((u, v))
    return fw_s & hw_s, pairs


def sgr(depth, vsh, z, throw):
    """Shale gouge ratio at fault depth z: mean Vsh of the beds that slid past the point, i.e. over [z, z + throw]
    of the footwall column (Yielding et al. 1997).  depth / vsh: the log (regular or not)."""
    d = np.asarray(depth, float)
    v = np.asarray(vsh, float)
    if throw <= 0:
        return float(np.interp(z, d, v))
    zz = np.linspace(z, z + throw, max(3, int(throw / max(1e-6, np.median(np.diff(d)))) + 1))
    return float(np.mean(np.interp(zz, d, v)))


def sgr_column(sgr_value, depth_m, rho_w=1.03, rho_hc=0.25, c=None):
    """Supported hydrocarbon column (m) from SGR (fraction or %): AFPD (bar) = 10 ** (SGR% / 27 - C) with C = 0.5
    (< 3 km), 0.25 (3-3.5 km), 0 (> 3.5 km) unless given (Yielding 2002 type calibration, editable)."""
    s = float(sgr_value) * (100.0 if sgr_value <= 1.0 else 1.0)
    if c is None:
        c = 0.5 if depth_m < 3000 else (0.25 if depth_m < 3500 else 0.0)
    afpd_bar = 10 ** (s / 27.0 - c)
    return afpd_bar * 1e5 / ((rho_w - rho_hc) * 1000.0 * 9.81), afpd_bar


def stress_tendency(strike_deg, dip_deg, sv, shmax, shmin, shmax_az, pp=0.0):
    """Slip tendency (tau / sigma_n') and dilation tendency ((s1 - sn) / (s1 - s3)) of a plane (strike, dip right-hand
    rule) in a stress field: Sv vertical, SHmax at azimuth shmax_az, Shmin perpendicular (MPa); pp pore pressure."""
    az = math.radians(shmax_az)
    eH = np.array([math.sin(az), math.cos(az), 0.0])  # east, north, up
    eh = np.array([math.cos(az), -math.sin(az), 0.0])
    ev = np.array([0.0, 0.0, 1.0])
    S = sv * np.outer(ev, ev) + shmax * np.outer(eH, eH) + shmin * np.outer(eh, eh)
    st, dp = math.radians(strike_deg), math.radians(dip_deg)
    dd = st + math.pi / 2  # dip direction
    n = np.array([math.sin(dd) * math.sin(dp), math.cos(dd) * math.sin(dp), math.cos(dp)])
    t = S @ n
    sn = float(n @ t)
    tau = float(np.sqrt(max(0.0, t @ t - sn * sn)))
    s1, s3 = max(sv, shmax, shmin), min(sv, shmax, shmin)
    return {"slip": tau / max(1e-9, sn - pp), "dilation": (s1 - sn) / max(1e-9, s1 - s3), "sigma_n": sn, "tau": tau}


# ====================================================================== 27c
def micp_to_reservoir(pc_lab, ift_lab=485.0, theta_lab=140.0, ift_res=30.0, theta_res=0.0):
    """Convert an air-mercury capillary pressure (any unit) to reservoir conditions (brine-hydrocarbon)."""
    return pc_lab * (ift_res * math.cos(math.radians(theta_res))) / abs(ift_lab * math.cos(math.radians(theta_lab)))


def column_height(pce_bar, rho_w=1.03, rho_hc=0.25):
    """Hydrocarbon column (m) a seal with entry pressure pce_bar holds."""
    return pce_bar * 1e5 / ((rho_w - rho_hc) * 1000.0 * 9.81)


VSH_PCE = ((0.2, 0.5), (0.4, 3.0), (0.6, 10.0), (0.8, 25.0))  # Vsh -> reservoir Pce (bar), editable defaults


def top_seal_column(seal_vsh, seal_thickness, table=VSH_PCE, min_thickness=10.0, rho_w=1.03, rho_hc=0.25):
    """Column (m) held by the top seal from its Vsh (log-linear lookup of Pce) - zero where the seal is thinner than
    min_thickness (breached / sand-prone)."""
    v = np.asarray(seal_vsh, float)
    xs = np.array([a for a, _ in table])
    ys = np.log(np.array([b for _, b in table]))
    pce = np.exp(np.interp(v, xs, ys))
    col = column_height(pce, rho_w, rho_hc)
    return np.where(np.asarray(seal_thickness, float) >= min_thickness, col, 0.0)


def limiting_column(spill_column, fault_column=None, top_column=None):
    """Column actually held = min of the spill (closure), fault-seal and top-seal limits, and which one limits."""
    cands = {"spill": spill_column}
    if fault_column is not None:
        cands["fault seal"] = fault_column
    if top_column is not None:
        cands["top seal"] = top_column
    k = min(cands, key=lambda q: cands[q])
    return float(cands[k]), k
