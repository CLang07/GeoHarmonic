"""Cross-check view (PLAN 21e): petrophysics zone values vs seismic attribute / inversion maps at the wells.

    cc = crosscheck(stack, "PHIE_res", "C-sands", [grid layer ids])
    cc["r"], cc["coef"], cc["resid"]          linear (multiple) regression of the zone value on the grids at the wells
    predicted_maps(stack, cc)                  -> P50 (regression), P90 / P10 (80 % prediction interval), residual map,
                                                  corrected map (= P50 + IDW of the residuals; regression-kriging style)
    to_attribute_table(stack, ...)             -> analytics.table.AttributeTable for the item-13 models (GBM, SHAP ...)
"""
import numpy as np

from .maps import MapGrid
from .maplayers import petro_values
from .propgrid import residual_map, sample_grid

Z80 = 1.2815515655446004  # P10 / P90 of a normal distribution


def crosscheck(stack, prop, zone, grid_lids):
    vals = petro_values(stack.project, prop, zone)
    names = [q[0] for q in vals]
    px, py, y = (np.array([q[i] for q in vals], float) for i in (1, 2, 3))
    grids = [stack.grid(l) for l in grid_lids]
    labels = [stack.get(l)["name"] for l in grid_lids]
    X = np.column_stack([sample_grid(g, px, py) for g in grids]) if grids else np.zeros((len(y), 0))
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    n, k = int(ok.sum()), X.shape[1]
    out = {"prop": prop, "zone": zone, "wells": names, "px": px, "py": py, "y": y, "X": X, "ok": ok, "labels": labels,
           "grid_lids": list(grid_lids), "n": n}
    if n < k + 2:
        out.update(coef=None, r=np.nan, resid=np.full(len(y), np.nan), sigma=np.nan, loo_rmse=np.nan)
        return out
    A = np.column_stack([np.ones(n), X[ok]])
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    pred = A @ coef
    resid = np.full(len(y), np.nan)
    resid[ok] = y[ok] - pred
    dof = max(n - k - 1, 1)
    sigma = float(np.sqrt(np.sum(resid[ok] ** 2) / dof))
    AtA_inv = np.linalg.pinv(A.T @ A)
    H = np.einsum("ij,jk,ik->i", A, AtA_inv, A)  # leverages
    loo = resid[ok] / np.maximum(1 - H, 1e-9)  # exact LOO residuals of least squares
    r = float(np.corrcoef(pred, y[ok])[0, 1]) if np.std(pred) > 0 else np.nan
    out.update(coef=coef, r=r, r2=r * r if np.isfinite(r) else np.nan, resid=resid, sigma=sigma, AtA_inv=AtA_inv,
               loo_rmse=float(np.sqrt(np.mean(loo ** 2))), dof=dof)
    return out


def predicted_maps(stack, cc):
    """{"P50", "P90", "P10", "residual", "corrected"} MapGrids on the first grid's geometry (None without a fit)."""
    if cc.get("coef") is None:
        return None
    grids = [stack.grid(l) for l in cc["grid_lids"]]
    base = grids[0]
    Xn, Yn = base.node_xy()
    cols = [base.z] + [sample_grid(g, Xn, Yn) for g in grids[1:]]
    A = np.stack([np.ones(base.z.shape)] + cols, -1)
    p50 = A @ cc["coef"]
    se = cc["sigma"] * np.sqrt(1.0 + np.einsum("...i,ij,...j->...", A, cc["AtA_inv"], A))
    lab = f"{cc['prop']} {cc['zone'] or 'all zones'}"

    def mk(z, what):
        return MapGrid(z, base.kind, "", f"{lab} {what}", base.vol, base.x0, base.y0, base.dx, base.dy)

    ok = cc["ok"]
    rm = residual_map(cc["px"][ok], cc["py"][ok], cc["resid"][ok], base)
    return {"P50": mk(p50, "predicted (P50)"), "P90": mk(p50 - Z80 * se, "P90 (low)"),
            "P10": mk(p50 + Z80 * se, "P10 (high)"), "residual": mk(rm.z, "residual at wells (IDW)"),
            "corrected": mk(p50 + rm.z, "predicted + residual")}


def to_attribute_table(stack, prop, zone, grid_lids):
    """AttributeTable (ids = wells, target column = the zone value, one column per grid layer, xy) for Analytics."""
    from .analytics.table import AttributeTable
    cc = crosscheck(stack, prop, zone, grid_lids)
    t = AttributeTable(cc["wells"])
    t.add(f"{prop}{' ' + zone if zone else ''}", cc["y"])
    for j, lab in enumerate(cc["labels"]):
        t.add(lab, cc["X"][:, j])
    t.xy = np.column_stack([cc["px"], cc["py"]])
    t.meta["source"] = "map hub cross-check"
    return t
