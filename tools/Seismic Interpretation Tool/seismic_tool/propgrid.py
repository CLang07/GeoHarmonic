"""Gridding of zone properties for the map hub (PLAN 21c) + grid calculator.

Methods (well values px, py, v; optional secondary MapGrid s, e.g. a seismic attribute or inversion map):
  idw      inverse distance (power 2)
  tps      thin-plate spline (exact)
  kriging  simple kriging, variogram fitted to the wells (vgrid)
  ked      kriging with external drift: universal kriging with drift [1, s(x)], residual variogram; exact at wells
  cck      simple collocated cokriging, Markov model 1 (C12 = rho C11) on standardised variables:
             mu = rho (1 - q) / (1 - rho^2 q), q = r0' R^-1 r0;  Z1* = (1 - mu rho) r0' R^-1 z1 + mu Z2(x0)
           rho = 0 -> simple kriging; rho = 1 -> the secondary (rescaled) exactly.
Cross-validation: leave-one-well-out per method (RMSE, MAE, bias, residuals), residual map (IDW of the LOO residuals).
Grid calculator: calc("A - B", {"A": g1, "B": g2}) on A's geometry (others resampled bilinearly), + - * / ** unary -,
numbers, min / max / abs / clip(x, lo, hi), log10 / exp / sqrt; mask_polygon(grid, px, py, keep="inside").
"""
import ast
import math

import numpy as np

from . import vgrid
from .maps import MapGrid
from .velocity import _idw_nodes

METHODS = {"idw": "Inverse distance", "tps": "Thin-plate spline", "kriging": "Kriging (simple)",
           "ked": "Kriging with external drift", "cck": "Collocated cokriging"}


# ------------------------------------------------------------------------------------------- geometry helpers
def target_grid(px, py, cell=None, like=None, pad_frac=0.15, max_nodes=250000):
    """A MapGrid to fill: the secondary grid's geometry (like) or an XY box around the wells."""
    if like is not None:
        return MapGrid(np.full(like.z.shape, np.nan), like.kind, "", "", like.vol, like.x0, like.y0, like.dx, like.dy)
    px, py = np.asarray(px, float), np.asarray(py, float)
    span = max(np.ptp(px), np.ptp(py), 1000.0)
    pad = pad_frac * span
    x0, x1, y0, y1 = px.min() - pad, px.max() + pad, py.min() - pad, py.max() + pad
    cell = float(cell or span / 100.0)
    nx = int(np.ceil((x1 - x0) / cell)) + 1
    ny = int(np.ceil((y1 - y0) / cell)) + 1
    while nx * ny > max_nodes:
        cell *= 1.25
        nx = int(np.ceil((x1 - x0) / cell)) + 1
        ny = int(np.ceil((y1 - y0) / cell)) + 1
    return MapGrid(np.full((ny, nx), np.nan), "xy", "", "", None, x0, y0, cell, cell)


def sample_grid(g, x, y):
    """Vectorised bilinear sample of a MapGrid at points (NaN outside)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    shape = x.shape
    x, y = x.ravel(), y.ravel()
    if g.kind == "ilxl":
        v = g.vol
        il, xl = v.ilxl_at(x, y)
        fi, fj = (np.asarray(il) - v.il0) / v.il_step, (np.asarray(xl) - v.xl0) / v.xl_step
    else:
        fj, fi = (x - g.x0) / g.dx, (y - g.y0) / g.dy
    ny, nx = g.z.shape
    ok = (fi >= -1e-6) & (fi <= ny - 1 + 1e-6) & (fj >= -1e-6) & (fj <= nx - 1 + 1e-6)
    i0 = np.clip(np.floor(fi).astype(int), 0, max(0, ny - 2))
    j0 = np.clip(np.floor(fj).astype(int), 0, max(0, nx - 2))
    wi, wj = np.clip(fi - i0, 0, 1), np.clip(fj - j0, 0, 1)
    i1, j1 = np.minimum(i0 + 1, ny - 1), np.minimum(j0 + 1, nx - 1)
    z = g.z
    val = z[i0, j0] * (1 - wi) * (1 - wj) + z[i0, j1] * (1 - wi) * wj + z[i1, j0] * wi * (1 - wj) + z[i1, j1] * wi * wj
    return np.where(ok, val, np.nan).reshape(shape)


# ------------------------------------------------------------------------------------------- estimators
def _cov(vario, d):
    return vario["sill"] * (1.0 - vgrid._shape(d, vario["range"], vario["model"]))


def ked_fit(px, py, v, s, vario=None):
    px, py, v, s = (np.asarray(a, float) for a in (px, py, v, s))
    n = len(v)
    if vario is None:
        A = np.c_[np.ones(n), s]
        coef = np.linalg.lstsq(A, v, rcond=None)[0]
        vario = vgrid.fit_variogram(px, py, v - A @ coef)
    d = np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :])
    C = _cov(vario, d) + np.eye(n) * 1e-9 * vario["sill"]
    F = np.c_[np.ones(n), s]
    K = np.zeros((n + 2, n + 2))
    K[:n, :n], K[:n, n:], K[n:, :n] = C, F, F.T
    rhs = np.concatenate([v, [0.0, 0.0]])
    sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
    return {"a": sol[:n], "b": sol[n:], "px": px, "py": py, "vario": vario}


def ked_eval(fit, X, Y, S):
    out = fit["b"][0] + fit["b"][1] * np.asarray(S, float)
    vg = fit["vario"]
    for x, y, a in zip(fit["px"], fit["py"], fit["a"]):
        out = out + a * _cov(vg, np.hypot(X - x, Y - y))
    return out


def cck_estimate(px, py, v, s_wells, X, Y, S, rho=None, vario=None, s_mean=None, s_sd=None):
    """Simple collocated cokriging (MM1).  Returns (estimate, rho used)."""
    px, py, v, s_wells = (np.asarray(a, float) for a in (px, py, v, s_wells))
    m1, sd1 = float(np.mean(v)), float(np.std(v)) or 1.0
    S = np.asarray(S, float)
    # secondary standardised with its statistics AT THE WELLS (consistent with rho measured there): with rho = 1 the
    # estimate is then exactly the linear rescaling of the secondary that the wells imply
    m2 = float(np.mean(s_wells)) if s_mean is None else s_mean
    sd2 = (float(np.std(s_wells)) or 1.0) if s_sd is None else s_sd
    z1 = (v - m1) / sd1
    if rho is None:
        rho = float(np.corrcoef(v, s_wells)[0, 1]) if len(v) > 2 and np.std(s_wells) > 0 else 0.0
    rho = float(np.clip(rho, -1.0, 1.0))
    if vario is None:
        vario = vgrid.fit_variogram(px, py, z1)
    vn = dict(vario, sill=1.0)
    d = np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :])
    R = _cov(vn, d) + np.eye(len(v)) * 1e-9
    Ri = np.linalg.inv(R)
    Xf, Yf = np.ravel(X), np.ravel(Y)
    r0 = _cov(vn, np.hypot(Xf[:, None] - px[None, :], Yf[:, None] - py[None, :]))  # (nodes, n)
    q = np.einsum("ij,jk,ik->i", r0, Ri, r0)
    sk = r0 @ (Ri @ z1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = rho * (1 - q) / np.where(np.abs(1 - rho * rho * q) > 1e-12, 1 - rho * rho * q, np.nan)
    mu = np.where(np.isfinite(mu), mu, rho)
    z2 = (np.ravel(S) - m2) / sd2
    est = (1 - mu * rho) * sk + mu * np.where(np.isfinite(z2), z2, 0.0)
    est = np.where(np.isfinite(z2), est, sk)  # no secondary there: simple kriging
    return (m1 + sd1 * est).reshape(np.shape(X)), rho


def estimate(method, px, py, v, X, Y, secondary=None, opts=None):
    """Values at map points X, Y (arrays) with one method; secondary = MapGrid (needed by ked / cck)."""
    opts = opts or {}
    px, py, v = (np.asarray(a, float) for a in (px, py, v))
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if method in ("idw", "tps", "kriging"):
        return vgrid.grid_values(X, Y, px, py, v, method, opts)
    if secondary is None:
        raise ValueError(f"{method} needs a secondary grid")
    sw = sample_grid(secondary, px, py)
    ok = np.isfinite(sw)
    if ok.sum() < 3:
        raise ValueError("fewer than 3 wells inside the secondary grid")
    S = sample_grid(secondary, X, Y) if secondary is not None else None
    if method == "ked":
        fit = ked_fit(px[ok], py[ok], v[ok], sw[ok])
        out = ked_eval(fit, X, Y, np.where(np.isfinite(S), S, np.nanmean(sw[ok])))
        return out
    if method == "cck":
        est, _ = cck_estimate(px[ok], py[ok], v[ok], sw[ok], X, Y, S, opts.get("rho"))
        return est
    raise ValueError(f"unknown method {method}")


def grid_property(px, py, v, method="idw", secondary=None, cell=None, opts=None, label="", unit=""):
    """MapGrid of well values (on the secondary grid's geometry when there is one)."""
    g = target_grid(px, py, cell, like=secondary)
    X, Y = g.node_xy()
    g.z = np.asarray(estimate(method, px, py, v, X, Y, secondary, opts), float).reshape(X.shape)
    if secondary is not None:
        g.z = np.where(np.isfinite(secondary.z), g.z, np.nan)
    g.label, g.unit = label or METHODS.get(method, method), unit
    return g


def correlation(px, py, v, secondary):
    """Pearson r between the well values and the secondary grid at the wells (+ n)."""
    sw = sample_grid(secondary, px, py)
    ok = np.isfinite(sw) & np.isfinite(v)
    if ok.sum() < 3 or np.std(sw[ok]) == 0:
        return float("nan"), int(ok.sum())
    return float(np.corrcoef(np.asarray(v)[ok], sw[ok])[0, 1]), int(ok.sum())


def cross_validate(px, py, v, methods=("idw", "tps", "kriging"), secondary=None, opts=None):
    """Leave-one-well-out per method: {method: {"pred", "resid", "rmse", "mae", "bias"}} (resid = pred - true)."""
    px, py, v = (np.asarray(a, float) for a in (px, py, v))
    n = len(v)
    out = {}
    for m in methods:
        pred = np.full(n, np.nan)
        for i in range(n):
            k = np.arange(n) != i
            try:
                pred[i] = float(np.ravel(estimate(m, px[k], py[k], v[k], np.array([px[i]]), np.array([py[i]]),
                                                   secondary, opts))[0])
            except Exception:
                pred[i] = np.nan
        r = pred - v
        f = np.isfinite(r)
        out[m] = {"pred": pred, "resid": r, "rmse": float(np.sqrt(np.mean(r[f] ** 2))) if f.any() else float("nan"),
                  "mae": float(np.mean(np.abs(r[f]))) if f.any() else float("nan"),
                  "bias": float(np.mean(r[f])) if f.any() else float("nan"), "n": int(f.sum())}
    return out


def residual_map(px, py, resid, like):
    """IDW map of LOO residuals on a grid's geometry."""
    g = MapGrid(np.full(like.z.shape, np.nan), like.kind, like.unit, "LOO residual", like.vol, like.x0, like.y0,
                like.dx, like.dy)
    f = np.isfinite(resid)
    X, Y = g.node_xy()
    g.z = _idw_nodes(X, Y, np.asarray(px)[f], np.asarray(py)[f], np.asarray(resid)[f])
    return g


# ------------------------------------------------------------------------------------------- grid calculator
_FUNCS = {"abs": np.abs, "sqrt": np.sqrt, "log10": np.log10, "exp": np.exp, "min": np.fmin, "max": np.fmax,
          "clip": lambda a, lo, hi: np.clip(a, lo, hi)}


def calc(expr, grids):
    """Evaluate a grid expression; all grids are resampled to the first referenced grid's geometry.
    Returns a MapGrid.  Only arithmetic, numbers, grid names and the functions in _FUNCS are allowed."""
    tree = ast.parse(expr, mode="eval")
    nodes = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id not in _FUNCS),
                   key=lambda n: (n.lineno, n.col_offset))  # source order: the first grid named sets the geometry
    names = list(dict.fromkeys(n.id for n in nodes))
    unknown = [n for n in names if n not in grids]
    if unknown:
        raise ValueError(f"unknown grid(s): {', '.join(unknown)}")
    if not names:
        raise ValueError("the expression uses no grid")
    base = grids[names[0]]
    X, Y = base.node_xy()
    arrays = {}
    for n in set(names):
        g = grids[n]
        same = (g is base) or (g.kind == base.kind and g.z.shape == base.z.shape and g.vol is base.vol and
                               (g.kind == "ilxl" or (math.isclose(g.x0, base.x0) and math.isclose(g.y0, base.y0)
                                                    and math.isclose(g.dx, base.dx) and math.isclose(g.dy, base.dy))))
        arrays[n] = np.asarray(g.z, float) if same else sample_grid(g, X, Y)

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            return arrays[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            ops = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
            f = ops.get(type(node.op))
            if f is None:
                raise ValueError("operator not allowed")
            with np.errstate(divide="ignore", invalid="ignore"):
                return f(a, b)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            args = [ev(a) for a in node.args]
            with np.errstate(divide="ignore", invalid="ignore"):
                return _FUNCS[node.func.id](*args)
        raise ValueError(f"not allowed in a grid expression: {ast.dump(node)[:40]}")

    z = np.broadcast_to(ev(tree), base.z.shape).astype(float)
    z = np.where(np.isfinite(z), z, np.nan)
    return MapGrid(z, base.kind, base.unit, expr, base.vol, base.x0, base.y0, base.dx, base.dy)


def mask_polygon(grid, px, py, keep="inside"):
    """Blank a grid outside (keep='inside') or inside (keep='outside') a polygon."""
    from .maphub import point_in_polygon
    X, Y = grid.node_xy()
    inside = point_in_polygon(np.ravel(X), np.ravel(Y), px, py).reshape(X.shape)
    m = inside if keep == "inside" else ~inside
    z = np.where(m, grid.z, np.nan)
    return MapGrid(z, grid.kind, grid.unit, grid.label + " (masked)", grid.vol, grid.x0, grid.y0, grid.dx, grid.dy)
