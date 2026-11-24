"""'Add wells' workflow for the map hub (PLAN 21d): one action when new wells arrive.

    rep = add_wells(project, stack, folder)        # folder in the Well Correlation layout (wells/<Well>/las|dlis ...)

Steps (each reported; a failing step does not stop the others):
  1. import the new wells into the linked Well Correlation project (existing wells untouched) and save it
  2. petrophysics batch on the new wells with the project's parameter sets (Well Correlation Tool item 20)
  3. residuals at the new wells: every property-grid layer (made by "Grid a zone property") sampled at the new wells
     BEFORE re-gridding vs the wells' new values - the honest blind test of the old maps
  4. re-grid those layers from their stored recipe (before / after kept: optional change layer)
  5. batch well tie (item 16) of the new wells on the first 3D survey (when there is a survey and the wells have sonic)
  6. dynamic depth model (item 17c): refresh when the project has a velocity model; LFM (item 19a): flagged out of date
"""
import os
import time

import numpy as np


def property_layers(stack):
    """Grid layers that carry a gridding recipe (prop, zone, method, secondary, cell)."""
    return [l for l in stack.layers if l["kind"] == "grid" and l["src"].get("recipe")]


def regrid_layer(stack, lid):
    """Recompute a recipe grid layer from the current petro summary; returns (old MapGrid, new MapGrid)."""
    from . import propgrid as PG
    from .maplayers import petro_values, save_grid
    lay = stack.get(lid)
    r = lay["src"]["recipe"]
    vals = petro_values(stack.project, r["prop"], r.get("zone"))
    if len(vals) < 2:
        raise ValueError(f"{len(vals)} wells with {r['prop']}")
    px, py, v = (np.array([q[i] for q in vals], float) for i in (1, 2, 3))
    sg = stack.grid(r["secondary"]) if r.get("secondary") else None
    old = stack.grid(lid)
    new = PG.grid_property(px, py, v, r.get("method", "idw"), sg, r.get("cell"), label=lay["name"])
    stack._grids[lid] = new
    if stack.project is not None and stack.project.path:
        lay["src"]["file"] = save_grid(stack.project, new, lid)
    return old, new


def add_wells(project, stack, folder, tie=True, change_layers=False, progress=None):
    from .welltie import wc
    from .maplayers import petro_values
    wc()
    from well_correlation.project import Project
    from well_correlation.petro import interp as PI
    t0 = time.perf_counter()
    rep = {"added": [], "steps": [], "residuals": [], "regridded": [], "ties": [], "errors": []}

    def step(name, msg):
        rep["steps"].append((name, msg))
        if progress:
            progress(name, msg)

    if not project.well_project or not os.path.exists(project.well_project):
        raise ValueError("link a Well Correlation project first (File > Link well project)")
    wcp = Project.open(project.well_project)
    added = wcp.add_from_folder(folder)
    rep["added"] = added
    if not added:
        step("import", f"no new wells in {folder}")
        return rep
    wcp.save()
    project.__dict__.pop("_wellset", None)  # wellset / hub caches read the well project again
    project.__dict__.pop("_hub_wcp", None)
    step("import", f"{len(added)} new wells: {', '.join(added)}")
    # 2 petro batch
    try:
        res = PI.interpret_project(wcp, added)
        ok = [n for n, r in res.items() if r.get("zones")]
        step("petrophysics", f"{len(ok)} of {len(added)} new wells interpreted (zones need tops; wells without tops get "
                             "curves only)")
    except Exception as e:
        rep["errors"].append(f"petrophysics: {e}")
    # 3 residuals of the old maps at the new wells
    xy = {n: (x, y) for n, x, y in project.wells()}
    layers = property_layers(stack)
    for lay in layers:
        r = lay["src"]["recipe"]
        try:
            g = stack.grid(lay["id"])
            vals = {w: v for w, _, _, v in petro_values(project, r["prop"], r.get("zone"))}
            for n in added:
                if n not in vals or n not in xy:
                    continue
                from .propgrid import sample_grid
                pred = float(sample_grid(g, [xy[n][0]], [xy[n][1]])[0])
                rep["residuals"].append({"well": n, "layer": lay["name"], "predicted": pred, "actual": vals[n],
                                         "residual": vals[n] - pred if np.isfinite(pred) else np.nan})
        except Exception as e:
            rep["errors"].append(f"residuals {lay['name']}: {e}")
    if layers:
        step("blind test", f"{len(rep['residuals'])} old-map predictions checked at the new wells")
    # 4 re-grid
    for lay in layers:
        try:
            old, new = regrid_layer(stack, lay["id"])
            d = new.z - old.z if old is not None and old.z.shape == new.z.shape else None
            row = {"layer": lay["name"], "id": lay["id"]}
            if d is not None and np.isfinite(d).any():
                row.update(max_change=float(np.nanmax(np.abs(d))), rms_change=float(np.sqrt(np.nanmean(d ** 2))))
                if change_layers:
                    from .maps import MapGrid
                    cg = MapGrid(d, new.kind, new.unit, f"{lay['name']} change", new.vol, new.x0, new.y0, new.dx, new.dy)
                    row["change_layer"] = stack.add_grid(cg, f"{lay['name']} change (new wells)",
                                                         style={"lut": "Red-white-black"}, contours=False)[0]
            rep["regridded"].append(row)
        except Exception as e:
            rep["errors"].append(f"re-grid {lay['name']}: {e}")
    if layers:
        step("re-grid", f"{len(rep['regridded'])} property maps re-gridded")
    stack.save()
    # 5 batch tie of the new wells
    if tie and project.volumes:
        try:
            from .tie import batch_tie
            ws = project.wellset(reload=True)
            vol = project.volume(project.volumes[0]["name"])
            rows = batch_tie(ws, "3D", vol, wells=[n for n in added if n in ws.names])
            rep["ties"] = [{k: v for k, v in r.items() if k != "wavelet"} for r in rows]
            good = [r for r in rows if "error" not in r]
            step("well tie", f"{len(good)} of {len(rows)} new wells tied on {project.volumes[0]['name']}"
                             + (f" (mean shift {np.mean([r['shift_ms'] for r in good]):.1f} ms)" if good else ""))
        except Exception as e:
            rep["errors"].append(f"well tie: {e}")
    # 6 depth model + LFM
    if project.display.get("velocity_model"):
        try:
            from .ddc import DynamicDepthModel
            dm = DynamicDepthModel(project, project.display["velocity_model"]).build()
            step("depth model", "velocity model rebuilt with the new wells' tops" + (
                f": {dm.last.get('message', '')}" if isinstance(getattr(dm, "last", None), dict) else ""))
        except Exception as e:
            rep["errors"].append(f"depth model: {e}")
    if any("inversion" in (v.get("meta") or {}) or "[LFM" in v["name"] for v in project.volumes):
        step("LFM", "low-frequency model volumes exist: rebuild them in Interpret > Seismic inversion (not automatic)")
    rep["seconds"] = time.perf_counter() - t0
    return rep


def report_text(rep):
    L = []
    for name, msg in rep["steps"]:
        L.append(f"{name}: {msg}")
    if rep["residuals"]:
        L.append("")
        L.append("Old maps at the new wells (actual - predicted):")
        for r in rep["residuals"]:
            L.append(f"  {r['well']:12s} {r['layer'][:36]:36s} predicted {r['predicted']:.3g}  actual {r['actual']:.3g}  "
                     f"residual {r['residual']:+.3g}")
        res = np.array([r["residual"] for r in rep["residuals"]], float)
        res = res[np.isfinite(res)]
        if res.size:
            L.append(f"  RMS residual {np.sqrt(np.mean(res ** 2)):.3g}, mean {res.mean():+.3g} (n {res.size})")
    for r in rep["regridded"]:
        if "max_change" in r:
            L.append(f"re-gridded {r['layer']}: max change {r['max_change']:.3g}, RMS {r['rms_change']:.3g}")
    for t in rep["ties"]:
        L.append(f"tie {t['well']}: " + (t["error"] if "error" in t else f"shift {t['shift_ms']:+.1f} ms, corr {t['corr']:.2f}"))
    for e in rep["errors"]:
        L.append(f"! {e}")
    return "\n".join(L)
