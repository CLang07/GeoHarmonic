"""Pseudo-well runs (PLAN 18d): the 1D burial / thermal / maturity model on many columns at once - maps of
present temperature, %Ro, transformation ratio and oil-window entry at a horizon, a Wheeler (age axis)
coloured by a model property, a burial (restored) section at any age, and the petroleum-system event chart.

Columns come from depth maps (item 17 velocity model) or any {entry: TVDSS} source; every column uses the
same age table, so all share the units and time axis and run as one batch (vectorised, chunked, cancellable).
"""
import hashlib

import numpy as np

from . import maturity as MT
from . import thermal as TH
from .burial import BurialBatch

PROPERTIES = ("temperature", "ro", "tr", "porosity")
UNITS = {"temperature": "degC", "ro": "%Ro", "tr": "fraction", "porosity": "fraction", "oil_entry": "Ma",
         "depth": "m"}


def _q_series(ages, heat_flow, present=None):
    """Heat-flow history (nt,) and per-column scaling to present-day values (B,) -> (B, nt) mW/m2."""
    q = TH.heat_flow_history(ages, heat_flow)
    if present is None:
        return q[None, :]
    present = np.asarray(present, float)
    return q[None, :] * (present / max(1e-9, q[-1]))[:, None]


def surface_node(units, name, m):
    """Index of the top node of the first unit whose top is the named surface (or the base of the column)."""
    for i, u in enumerate(units):
        if not u["ghost"] and u["top_surface"] == name:
            return i * m
    last = [i for i, u in enumerate(units) if not u["ghost"]][-1]
    if units[last]["base_surface"] == name:
        return (last + 1) * m
    raise KeyError(f"surface {name!r} is not in the column")


def run_columns(table, entry_names, depths, seabed, heat_flow=60.0, present_q=None, surface_t=10.0, step=1.0,
                surfaces=None, kinetics="Type III (terrigenous / coaly, organofacies D/E)", chunk=400, n_sub=4,
                full=False, progress=None, cancel=None):
    """Run B columns. depths (B, K) TVDSS of entry_names (NaN = missing -> the column is skipped);
    seabed (B,) or scalar; present_q (B,) mW/m2 optional map of present heat flow (scales the history).
    Returns {"surfaces": {name: {"depth", "temperature", "ro", "tr", "oil_entry", "porosity"} (B,)},
    "valid" (B,), "ages", and with full=True per-node arrays "T", "z", "ro", "tr", "phi" (B, nt, nodes)
    plus "units", "m"}."""
    depths = np.asarray(depths, float)
    Bn = depths.shape[0]
    seabed = np.broadcast_to(np.asarray(seabed, float), (Bn,)).astype(float)
    valid = np.isfinite(depths).all(axis=1) & np.isfinite(seabed)
    surfaces = list(surfaces or entry_names)
    out = {"surfaces": {s: {k: np.full(Bn, np.nan) for k in ("depth", "temperature", "ro", "tr", "oil_entry",
                                                              "porosity")} for s in surfaces},
           "valid": valid}
    idx = np.nonzero(valid)[0]
    if not len(idx):
        return out
    full_parts = []
    for c0 in range(0, len(idx), chunk):
        if cancel is not None and cancel():
            raise RuntimeError("cancelled")
        sel = idx[c0:c0 + chunk]
        bb = BurialBatch(table, entry_names, depths[sel], seabed[sel], step)
        q = _q_series(bb.ages, heat_flow, None if present_q is None else np.asarray(present_q, float)[sel])
        r = TH.solve(bb, np.broadcast_to(q, (len(sel), len(bb.ages))).copy(), surface_t, n_sub=n_sub, cancel=cancel)
        ages = r["ages"]
        out["ages"] = ages
        for s in surfaces:
            try:
                node = surface_node(bb.units, s, r["m"])
            except KeyError:
                continue
            u_age = [u["age_top"] for u in bb.units if u["top_surface"] == s and not u["ghost"]]
            age = u_age[0] if u_age else max(u["age_base"] for u in bb.units)
            Tn = np.where(ages[:, None] <= age + 1e-9, r["T"][:, :, node].T, np.nan)  # (nt, b)
            ro = MT.easy_ro(ages, Tn)
            tr = MT.transformation_ratio(ages, Tn, kinetics)
            d = out["surfaces"][s]
            d["depth"][sel] = r["z"][:, -1, node] + bb.water[:, -1]
            d["temperature"][sel] = Tn[-1]
            d["ro"][sel] = ro[-1]
            d["tr"][sel] = tr[-1]
            d["oil_entry"][sel] = MT.crossing_age(ages, ro, MT.WINDOWS["oil"][0])
            phi = r["phi"][:, -1]
            d["porosity"][sel] = phi[:, min(node, phi.shape[1] - 1)]
        if full:
            full_parts.append((sel, r, bb))
        if progress is not None:
            progress(min(1.0, (c0 + len(sel)) / len(idx)))
    if full:
        sel, r, bb = full_parts[0] if len(full_parts) == 1 else _merge(full_parts)
        nt, nn = r["T"].shape[1], r["T"].shape[2]
        phin = np.concatenate([r["phi"][:, :, :1], 0.5 * (r["phi"][:, :, 1:] + r["phi"][:, :, :-1]),
                               r["phi"][:, :, -1:]], axis=2)
        out.update({"T": np.full((Bn, nt, nn), np.nan), "z": np.full((Bn, nt, nn), np.nan),
                    "phi": np.full((Bn, nt, nn), np.nan), "water": np.full((Bn, nt), np.nan),
                    "units": bb.units, "m": r["m"], "kinetics": kinetics})
        out["T"][sel], out["z"][sel], out["phi"][sel], out["water"][sel] = r["T"], r["z"], phin, bb.water
    return out


def _merge(parts):
    sel = np.concatenate([p[0] for p in parts])
    r = dict(parts[0][1])
    for k in ("T", "z", "phi"):
        r[k] = np.concatenate([p[1][k] for p in parts], axis=0)

    class _BB:
        pass
    bb = _BB()
    bb.units = parts[0][2].units
    bb.water = np.concatenate([p[2].water for p in parts], axis=0)
    return sel, r, bb


# ====================================================================== maps
def depth_grids_from_vm(vm, table):
    """{entry name: MapGrid} depth maps of a VelocityModel's layers matched to the age table."""
    out = {}
    for k, (hname, fm) in enumerate(vm.layers):
        for nm in (fm, hname):
            e = table.get(nm) if nm else None
            if e is not None and e["name"] not in out:
                out[e["name"]] = vm.depth_grid(k)
                break
    return out


def fingerprint(grids, *extra):
    """Hash of the depth maps + settings - a map run is stale when it changes (live update with item 17)."""
    h = hashlib.blake2b(digest_size=12)
    for n in sorted(grids):
        h.update(n.encode())
        h.update(np.ascontiguousarray(np.asarray(grids[n].z, np.float32)).tobytes())
    h.update(repr(extra).encode())
    return h.hexdigest()


def _upsample(coarse, ii, jj, shape):
    """Bilinear (separable) from the strided node rows ii / columns jj to the full grid shape, NaN-aware."""
    def interp_axis(a, src, n, axis):
        a = np.moveaxis(a, axis, -1)
        out = np.full(a.shape[:-1] + (n,), np.nan)
        dst = np.arange(n)
        for idx in np.ndindex(a.shape[:-1]):
            row = a[idx]
            ok = np.isfinite(row)
            if ok.sum() >= 2:
                out[idx] = np.interp(dst, src[ok], row[ok], left=np.nan, right=np.nan)
            elif ok.sum() == 1:
                out[idx][np.abs(dst - src[ok][0]) <= 0] = row[ok][0]
        return np.moveaxis(out, -1, axis)
    a = interp_axis(coarse, jj, shape[1], 1)
    return interp_axis(a, ii, shape[0], 0)


def map_run(depth_grids, table, surface, outputs=("temperature", "ro", "tr", "oil_entry"), stride=10,
            water_depth=100.0, heat_flow=60.0, present_q=None, surface_t=10.0, step=1.0,
            kinetics="Type III (terrigenous / coaly, organofacies D/E)", progress=None, cancel=None):
    """Maps of model results at one dated surface from a grid of pseudo-wells (every `stride`-th node, then
    bilinear back to the full grid). depth_grids {entry: MapGrid} (same geometry); the sea floor is the
    'Seabed' grid when present, else water_depth. present_q: MapGrid of present heat flow (optional).
    Returns {output: MapGrid} plus "_info" {columns, valid, seconds, fingerprint}."""
    import time

    from ..maps import MapGrid
    t0 = time.time()
    names = [e["name"] for e in table.entries if e["name"] in depth_grids]
    if surface not in names:
        raise ValueError(f"no depth map for {surface!r}")
    g0 = depth_grids[names[0]]
    ny, nx = g0.z.shape
    ii = np.arange(0, ny, max(1, stride))
    jj = np.arange(0, nx, max(1, stride))
    if ii[-1] != ny - 1:
        ii = np.append(ii, ny - 1)
    if jj[-1] != nx - 1:
        jj = np.append(jj, nx - 1)
    I, J = np.meshgrid(ii, jj, indexing="ij")
    D = np.stack([np.asarray(depth_grids[n].z, float)[I, J].ravel() for n in names], axis=1)
    if "Seabed" in depth_grids:
        sb = D[:, names.index("Seabed")]
    else:
        sb = np.full(D.shape[0], float(water_depth))
        if table.get("Seabed") is not None:
            names = names + ["Seabed"]
            D = np.c_[D, sb]
    pq = None
    if present_q is not None:
        pq = np.asarray(present_q.z, float)[I, J].ravel() if hasattr(present_q, "z") else \
            np.full(D.shape[0], float(present_q))
    res = run_columns(table, names, D, sb, heat_flow, pq, surface_t, step, [surface], kinetics, progress=progress,
                      cancel=cancel)
    d = res["surfaces"][surface]
    out = {}
    for o in outputs:
        coarse = d[o].reshape(len(ii), len(jj))
        z = _upsample(coarse, ii, jj, (ny, nx)).astype(np.float32)
        out[o] = MapGrid(z, g0.kind, UNITS.get(o, ""), f"{surface} {o}", g0.vol, g0.x0, g0.y0, g0.dx, g0.dy)
    out["_info"] = {"columns": int(D.shape[0]), "valid": int(res["valid"].sum()), "seconds": time.time() - t0,
                    "fingerprint": fingerprint(depth_grids, surface, stride, heat_flow, surface_t)}
    return out


# ====================================================================== section: Wheeler colouring, restoration
def material_history(res, arr, i, f):
    """History (B, nt, n) of node array arr (B, nt, nodes) following the sediment at fractions f (n,) of unit i
    (0 = top / youngest, 1 = base): while the unit is still being deposited a particle deposited at age a_f sits
    at fraction (a_f - t) / (a_base - t) of the partial unit; before a_f it does not exist (NaN)."""
    u, m = res["units"][i], res["m"]
    ages = res["ages"][:, None]
    f = np.asarray(f, float)[None, :]
    a1, a0 = u["age_top"], u["age_base"]
    af = a1 + f * (a0 - a1)
    g = np.where(ages <= a1, f, (af - ages) / np.maximum(1e-12, a0 - ages))
    exists = ages <= af + 1e-9
    node = i * m + np.clip(g, 0, 1) * m
    lo = np.clip(np.floor(node).astype(int), 0, arr.shape[2] - 1)
    hi = np.minimum(lo + 1, arr.shape[2] - 1)
    w = node - lo
    vlo = np.take_along_axis(arr, np.broadcast_to(lo[None], (arr.shape[0],) + lo.shape), 2)
    vhi = np.take_along_axis(arr, np.broadcast_to(hi[None], (arr.shape[0],) + hi.shape), 2)
    v = vlo * (1 - w)[None] + vhi * w[None]
    return np.where(exists[None], v, np.nan)


def unit_index(units, surface):
    for i, u in enumerate(units):
        if not u["ghost"] and u["top_surface"] == surface:
            return i
    raise KeyError(f"no unit below {surface!r}")


def source_tr(res, surface, frac=0.5, column=0, kinetics=None):
    """Transformation-ratio history (nt,) of the source rock in the unit below `surface` (at a fraction of
    its thickness) in one column of a full run."""
    i = unit_index(res["units"], surface)
    Th = material_history(res, res["T"][column:column + 1], i, [frac])[0, :, 0]
    return MT.transformation_ratio(res["ages"], Th, kinetics or res["kinetics"])


def wheeler_property(res, age_axis, prop="ro", t_eval=0.0):
    """Model property of the sediment deposited at each age of `age_axis`, evaluated at age t_eval (0 = today),
    for every column of a run_columns(full=True) result -> (len(axis), B). NaN where nothing of that age exists
    (hiatus / erosion) or where it was not yet deposited at t_eval. Maturity (ro, tr) integrates each
    sediment particle's own temperature history."""
    units = res["units"]
    ages = res["ages"]
    it = int(np.argmin(np.abs(ages - t_eval)))
    axis = np.asarray(age_axis, float)
    Bn = res["T"].shape[0]
    out = np.full((len(axis), Bn), np.nan)
    for i, u in enumerate(units):
        if u["ghost"]:
            continue
        a1, a0 = u["age_top"], u["age_base"]
        sel = np.nonzero((axis >= a1 - 1e-9) & (axis <= a0 + 1e-9) & (axis >= ages[it] - 1e-9))[0]
        if not len(sel):
            continue
        f = (axis[sel] - a1) / max(1e-12, a0 - a1)
        if prop in ("temperature", "porosity"):
            v = material_history(res, res["T"] if prop == "temperature" else res["phi"], i, f)[:, it]  # (B, n)
        else:
            Th = material_history(res, res["T"], i, f)  # (B, nt, n)
            flat = Th.transpose(1, 0, 2).reshape(len(ages), -1)
            mat = MT.easy_ro(ages, flat) if prop == "ro" else MT.transformation_ratio(ages, flat, res["kinetics"])
            v = mat.reshape(len(ages), Bn, len(sel))[it]
        out[sel] = np.asarray(v).T
    return out


def to_traces(values, col_x, trace_x):
    """Columns (n, B) at positions col_x -> traces (n, ntr) by linear interpolation along the section."""
    values = np.asarray(values, float)
    col_x = np.asarray(col_x, float)
    o = np.argsort(col_x)
    out = np.full((values.shape[0], len(trace_x)), np.nan)
    for k in range(values.shape[0]):
        row = values[k, o]
        ok = np.isfinite(row)
        if ok.sum() >= 2:
            out[k] = np.interp(trace_x, col_x[o][ok], row[ok], left=np.nan, right=np.nan)
        elif ok.sum() == 1:
            out[k] = row[ok][0]
    return out


def burial_section(res, age):
    """Restored section at an age: {surface name: depth below sea level (B,)} from a full run."""
    units, m = res["units"], res["m"]
    ages = res["ages"]
    it = int(np.argmin(np.abs(ages - age)))
    out = {}
    for i, u in enumerate(units):
        if u["ghost"] or u["age_top"] < age - 1e-9:
            continue
        out.setdefault(u["top_surface"], res["z"][:, it, i * m] + res["water"][:, it])
    last = [i for i, u in enumerate(units) if not u["ghost"]][-1]
    if units[last]["age_base"] >= age - 1e-9:
        out.setdefault(units[last]["base_surface"], res["z"][:, it, (last + 1) * m] + res["water"][:, it])
    return out


# ====================================================================== petroleum-system event chart
ROLES = ("source", "reservoir", "seal", "overburden")


def event_chart(table, ages, tr_source=None, trap=None, preservation=True):
    """Petroleum-system events: [{"element", "label", "start", "end"}] (Ma, start older) + critical moment.
    Element ages come from the age table (entry "role": source / reservoir / seal / overburden - the zone
    below that surface); generation from the source's transformation ratio history tr_source (nt,) (TR 10 %
    -> 90 %), expulsion from TR 50 %; trap (start, end) Ma typed in."""
    ev = []
    ents = table.entries
    for k, e in enumerate(ents):
        role = e.get("role")
        if role not in ROLES:
            continue
        top = e["age"] + e["gap"]
        base = ents[k + 1]["age"] if k + 1 < len(ents) else top
        ev.append({"element": role, "label": e["name"], "start": float(base), "end": float(top)})
    if trap is not None:
        ev.append({"element": "trap", "label": "trap formation", "start": float(trap[0]), "end": float(trap[1])})
    crit = None
    if tr_source is not None:
        g0 = MT.crossing_age(ages, tr_source, 0.10)
        g1 = MT.crossing_age(ages, tr_source, 0.90)
        x0 = MT.crossing_age(ages, tr_source, 0.50)
        if g0 is not None:
            ev.append({"element": "generation", "label": "TR 10-90 %", "start": g0, "end": g1 if g1 is not None else 0.0})
        if x0 is not None:
            ev.append({"element": "expulsion", "label": "TR > 50 %", "start": x0, "end": 0.0 if g1 is None else g1})
            crit = x0
    if preservation:
        ev.append({"element": "preservation", "label": "", "start": crit if crit is not None else 0.0, "end": 0.0})
    order = {k: i for i, k in enumerate(ROLES + ("trap", "generation", "expulsion", "preservation"))}
    ev.sort(key=lambda d: order.get(d["element"], 99))
    return {"events": ev, "critical_moment": crit}


def plot_event_chart(chart, path, max_age=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = chart["events"]
    rows = []
    for d in ev:
        if d["element"] not in rows:
            rows.append(d["element"])
    top = max_age or max([d["start"] for d in ev] + [1.0])
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1.5))
    for d in ev:
        y = rows.index(d["element"])
        ax.barh(y, d["start"] - d["end"], left=d["end"], height=0.6, color="#4a7ab7" if d["element"] in ROLES else
                "#c0504d", alpha=0.85)
        if d["label"]:
            ax.text(0.5 * (d["start"] + d["end"]), y, d["label"], ha="center", va="center", fontsize=7, color="w")
    if chart.get("critical_moment") is not None:
        ax.axvline(chart["critical_moment"], color="k", ls="--", lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlim(top, 0)
    ax.invert_yaxis()
    ax.set_xlabel("Age (Ma)")
    ax.set_title("Petroleum system events")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
