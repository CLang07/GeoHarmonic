"""Uncertainty + scenarios for the depth model (PLAN item 17e).

- Named velocity-model scenarios in the seismic project (display["velocity_scenarios"]), difference maps.
- Leave-one-well-out blind test: each well's tops predicted by the model built without it - an honest error estimate
  per horizon (mean, std, RMS, P10/P90 of the error) and its growth with distance to the nearest other well. The
  held-out model is built at points only (the test well + the training wells), so this is fast on any map size.
- P10/P50/P90 depth maps: P50 = model depth; sigma(x, y) = a + b * distance to the nearest well (from the blind test);
  P10 = shallow case (P50 - 1.2816 sigma), P90 = deep case (P50 + 1.2816 sigma).
- Depth conversion of a whole 3D volume to a depth SEG-Y (inline by inline, cancellable) and depth -> time for grids.
"""
import os

import numpy as np

from .maps import MapGrid
from .velocity import PointGrid, VelocityModel, _monotonic_depths

Z90 = 1.2815515655446004


# --------------------------------------------------------------------------- scenarios
def save_scenario(project, name, vm):
    project.display.setdefault("velocity_scenarios", {})[name] = vm.to_dict()
    if getattr(project, "path", None):
        project.save()


def scenario_names(project):
    return list((project.display or {}).get("velocity_scenarios", {}))


def load_scenario(project, name):
    return VelocityModel.from_dict(project, project.display["velocity_scenarios"][name])


def delete_scenario(project, name):
    project.display.get("velocity_scenarios", {}).pop(name, None)
    if getattr(project, "path", None):
        project.save()


def difference_grid(a, b, k, label=None):
    """Depth of layer k: model a - model b (m), on a's grid."""
    ga = a.depth_grid(k)
    gb = b.depth_grid(k)
    if gb.z.shape == ga.z.shape and gb.kind == ga.kind:
        zb = gb.z
    else:
        X, Y = ga.node_xy()
        zb = gb.sample(X.ravel(), Y.ravel()).reshape(ga.z.shape)
    return MapGrid(ga.z - zb, ga.kind, "m", label or f"Depth difference {a.layers[k][0]}", ga.vol, ga.x0, ga.y0,
                   ga.dx, ga.dy)


# --------------------------------------------------------------------------- blind test
def _clone(vm, exclude=None):
    return VelocityModel(vm.project, vm.key, vm.layers, vm.residual_correction, vm.cell, vm.below_velocity,
                         vm.functions, vm.gridding, exclude if exclude is not None else vm.exclude)


def blind_test(vm, progress=None, cancel=None, extra_wells=None):
    """Leave-one-well-out test. Returns {"rows": [{well, horizon, formation, actual, predicted, error, distance}],
    "summary": {horizon: {n, mean, std, rms, p10, p90, a, b}}} with error = actual - predicted (m, + = deeper than
    predicted) and sigma(d) = a + b * d (d = distance to the nearest other well, m)."""
    tg = vm.time_grids()
    rows = vm.well_table(tg)
    K = len(vm.layers)
    out = []
    for i, r in enumerate(rows):
        if cancel is not None and cancel():
            raise RuntimeError("cancelled")
        if progress:
            progress(i / max(len(rows), 1), f"blind test {r['well']} ({i + 1}/{len(rows)})")
        train = rows[:i] + rows[i + 1:]
        if not train:
            continue
        xs = np.r_[r["x"], [t["x"] for t in train]].astype(float)
        ys = np.r_[r["y"], [t["y"] for t in train]].astype(float)
        ptg = [PointGrid(xs, ys, np.r_[r["t"][k], [t["t"][k] for t in train]]) for k in range(K)]
        m = _clone(vm)
        try:
            m.build(time_grids=ptg, wells=train)
        except (ValueError, np.linalg.LinAlgError):
            continue
        dmin = float(np.min(np.hypot(xs[1:] - xs[0], ys[1:] - ys[0])))
        for k, (h, fm) in enumerate(vm.layers):
            if not str(r["source"][k]).startswith("top"):
                continue
            pred = float(m._grids["z"][k][0])
            if not np.isfinite(pred):
                continue
            out.append({"well": r["well"], "horizon": h, "formation": fm or "", "actual": r["z"][k],
                        "predicted": pred, "error": r["z"][k] - pred, "distance": dmin})
    if progress:
        progress(1.0, "blind test done")
    return {"rows": out, "summary": summarise(out, [h for h, _ in vm.layers])}


def summarise(rows, horizons):
    s = {}
    for h in horizons:
        e = np.array([r["error"] for r in rows if r["horizon"] == h], float)
        d = np.array([r["distance"] for r in rows if r["horizon"] == h], float)
        if not len(e):
            continue
        rms = float(np.sqrt(np.mean(e ** 2)))
        a, b = rms, 0.0
        if len(e) >= 4 and np.ptp(d) > 0:
            # |e| ~ (a + b d) * sqrt(2/pi) for normal errors; least squares, a, b >= 0
            ae = np.abs(e) * np.sqrt(np.pi / 2)
            bb, aa = np.polyfit(d, ae, 1)
            if bb < 0:
                aa, bb = float(np.mean(ae)), 0.0
            if aa < 0:
                aa, bb = 0.0, float(np.sum(d * ae) / np.sum(d * d))
            a, b = float(aa), float(bb)
        s[h] = {"n": int(len(e)), "mean": float(np.mean(e)), "std": float(np.std(e)), "rms": rms,
                "p10": float(np.percentile(e, 10)), "p90": float(np.percentile(e, 90)), "a": a, "b": b}
    return s


def sigma_by_horizon(bt):
    """{horizon: RMS blind-test error} - the 1-sigma column of the well prognosis."""
    return {h: v["rms"] for h, v in bt["summary"].items()}


# --------------------------------------------------------------------------- P10 / P50 / P90
def distance_to_wells(X, Y, px, py):
    d = np.full(np.shape(X), np.inf)
    t = np.empty(np.shape(X))
    for x, y in zip(px, py):
        np.hypot(np.asarray(X, float) - x, np.asarray(Y, float) - y, out=t)
        np.minimum(d, t, out=d)
    return d


def probability_maps(vm, k, bt):
    """{p10, p50, p90, sigma} MapGrids of layer k from the blind-test error model of its horizon."""
    h = vm.layers[k][0]
    st = bt["summary"].get(h)
    if st is None:
        raise ValueError(f"no blind-test errors for {h} (needs formation tops at 2+ wells)")
    g = vm.depth_grid(k)
    X, Y = g.node_xy()
    wells = vm.grids()["wells"]
    px = np.array([r["x"] for r in wells if not r.get("pseudo")], float)
    py = np.array([r["y"] for r in wells if not r.get("pseudo")], float)
    d = distance_to_wells(X, Y, px, py)
    sig = st["a"] + st["b"] * d
    if vm.residual_correction:  # the model honours the wells: the error tapers to 0 within 200 m of a well
        sig = sig * np.clip(d / 200.0, 0.0, 1.0)
    mk = lambda z, u, lab: MapGrid(z, g.kind, u, lab, g.vol, g.x0, g.y0, g.dx, g.dy)  # noqa: E731
    return {"p50": mk(g.z.copy(), "m", f"{h} depth P50"), "p10": mk(g.z - Z90 * sig, "m", f"{h} depth P10 (shallow)"),
            "p90": mk(g.z + Z90 * sig, "m", f"{h} depth P90 (deep)"), "sigma": mk(sig, "m", f"{h} depth 1 sigma")}


# --------------------------------------------------------------------------- conversions
def interp_rows(xq, XP, FP):
    """Row-wise linear interpolation: XP, FP (n, p) with XP non-decreasing per row; xq (m,) common queries or (n, m)
    per-row queries -> (n, m), NaN outside a row's range or where the row has NaN. One global searchsorted (rows
    offset into separate bands)."""
    XP = np.asarray(XP, float)
    FP = np.asarray(FP, float)
    xq = np.asarray(xq, float)
    n, p = XP.shape
    Q = np.broadcast_to(xq, (n, xq.shape[-1])) if xq.ndim == 1 else xq
    out = np.full(Q.shape, np.nan)
    good = np.all(np.isfinite(XP), axis=1) & np.all(np.isfinite(Q), axis=1)
    if not good.any() or p < 2:
        return out
    XPg, Qg = XP[good], Q[good]
    lo = min(float(XPg.min()), float(Qg.min()))
    span = max(float(XPg.max()), float(Qg.max())) - lo + 1.0
    off = span * np.arange(len(XPg))[:, None]
    flat = (XPg - lo + off).ravel()
    q = (Qg - lo) + off
    idx = np.searchsorted(flat, q.ravel(), "right").reshape(q.shape) - 1 - (p * np.arange(len(XPg)))[:, None]
    r = np.arange(len(XPg))[:, None]
    exact_end = Qg == XPg[:, -1:]
    idx = np.where(exact_end, p - 2, idx)
    valid = (idx >= 0) & (idx <= p - 2)
    i0 = np.clip(idx, 0, p - 2)
    x0, x1 = XPg[r, i0], XPg[r, i0 + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(x1 > x0, (Qg - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0)
    FPg = FP[good]
    val = FPg[r, i0] * (1 - f) + FPg[r, i0 + 1] * f
    out[good] = np.where(valid, val, np.nan)
    return out


def depth_axis(vm, vol, dz=None, zmax=None):
    """Depth sampling for a converted volume: dz = median depth step of the time samples, zmax = deepest sample of
    the four corner + centre traces."""
    il = vol.inlines[[0, 0, -1, -1, vol.nil // 2]]
    xl = vol.xlines[[0, -1, 0, -1, vol.nxl // 2]]
    x, y = vol.xy(il, xl)
    zt = _monotonic_depths(vm.depth_of_time(x, y, vol.times))
    if dz is None:
        dz = float(np.nanmedian(np.diff(zt, axis=1)))
        dz = float(max(0.5, round(dz * 2) / 2))
    if zmax is None:
        zmax = float(np.nanmax(zt))
    return np.arange(0.0, zmax + dz / 2, dz)


def _text_header(lines):
    out = b""
    for i in range(40):
        t = lines[i] if i < len(lines) else ""
        out += (f"C{i + 1:2d} " + t)[:80].ljust(80).encode("cp500", errors="replace")
    return out


def depth_convert_volume(vm, vol, out_path, dz=None, zmax=None, progress=None, cancel=None, name=""):
    """Write the depth-converted 3D volume as an IEEE-float SEG-Y. The sample interval in the headers is dz in
    millimetres (dz x 1000 in the 'microseconds' field - the usual depth SEG-Y convention, so other software and this
    tool read 1 'ms' = 1 m). Inline / crossline in bytes 189 / 193 (and 9 / 21), X / Y in 181 / 185 (scalar -100).
    Returns the depth axis (m)."""
    z = depth_axis(vm, vol, dz, zmax)
    dz = float(z[1] - z[0]) if len(z) > 1 else 1.0
    nz = len(z)
    if nz > 32767 or dz * 1000 > 65535:
        raise ValueError(f"depth axis too long for SEG-Y ({nz} samples of {dz} m)")
    tmp = out_path + ".part"
    try:
        with open(tmp, "wb") as fh:
            fh.write(_text_header([
                f"DEPTH-CONVERTED SEISMIC - GEO TOOL KIT SEISMIC INTERPRETATION TOOL {name}",
                f"DEPTH DOMAIN: SAMPLE INTERVAL {dz:g} M (BINARY/TRACE HEADER VALUE IN MM), {nz} SAMPLES FROM 0 M TVDSS",
                f"VELOCITY MODEL: {', '.join(h for h, _ in vm.layers)}",
                "INLINE BYTES 189 + 9, CROSSLINE 193 + 21, CDP X/Y 181/185 SCALAR -100 (BYTE 71)",
                "FORMAT 5 IEEE FLOAT, BIG ENDIAN"]))
            b = bytearray(400)
            b[16:18] = int(round(dz * 1000)).to_bytes(2, "big")
            b[20:22] = nz.to_bytes(2, "big")
            b[24:26] = (5).to_bytes(2, "big")
            b[28:30] = (1).to_bytes(2, "big")  # trace sorting: as recorded
            b[54:56] = (1).to_bytes(2, "big")  # measurement system: metres
            fh.write(bytes(b))
            times = np.asarray(vol.times, float)
            for n, il in enumerate(vol.inlines):
                if cancel is not None and cancel():
                    raise RuntimeError("cancelled")
                sec = vol.inline(il)  # (nxl, ns)
                xls = vol.xlines
                x, y = vol.xy(np.full(vol.nxl, il), xls)
                zt = _monotonic_depths(vm.depth_of_time(x, y, times))
                data = interp_rows(z, zt, sec)
                data = np.where(np.isfinite(data), data, 0.0).astype(">f4")
                th = np.zeros((vol.nxl, 240), np.uint8)

                def put(byte, vals, size=4):
                    th[:, byte - 1:byte - 1 + size] = np.asarray(vals, f">i{size}").view(np.uint8).reshape(vol.nxl, size)
                put(1, n * vol.nxl + np.arange(1, vol.nxl + 1))
                put(9, np.full(vol.nxl, int(il)))
                put(21, np.asarray(xls, np.int64))
                put(189, np.full(vol.nxl, int(il)))
                put(193, np.asarray(xls, np.int64))
                put(181, np.rint(np.asarray(x) * 100).astype(np.int64))
                put(185, np.rint(np.asarray(y) * 100).astype(np.int64))
                put(71, np.full(vol.nxl, -100), 2)
                put(115, np.full(vol.nxl, nz), 2)
                put(117, np.full(vol.nxl, int(round(dz * 1000))), 2)
                fh.write(np.concatenate([th, data.view(np.uint8).reshape(vol.nxl, -1)], axis=1).tobytes())
                if progress:
                    progress((n + 1) / vol.nil, f"inline {il} ({n + 1}/{vol.nil})")
        if os.path.exists(out_path):
            os.remove(out_path)
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return z


def depth_to_time_grid(vm, g, dt=1.0, tmax=None, chunk=20000):
    """Depth grid (m TVDSS, any MapGrid) -> TWT grid (ms) through the model's depth_of_time."""
    X, Y = g.node_xy()
    x, y, z = np.ravel(X), np.ravel(Y), np.ravel(g.z).astype(float)
    if tmax is None:
        tg = vm.grids()["t"]
        tmax = float(np.nanmax(tg[-1].z)) * 2.0 + 500.0
    times = np.arange(0.0, tmax + dt / 2, dt)
    out = np.full(z.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(z))
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        zt = _monotonic_depths(vm.depth_of_time(x[ii], y[ii], times))
        out[ii] = interp_rows(z[ii][:, None], zt, np.broadcast_to(times, zt.shape))[:, 0]
    return MapGrid(out.reshape(np.shape(g.z)), g.kind, "ms", f"{g.label} in time", g.vol, g.x0, g.y0, g.dx, g.dy)
