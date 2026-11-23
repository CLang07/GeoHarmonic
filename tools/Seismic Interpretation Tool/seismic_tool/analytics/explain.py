"""Explainability: permutation importance, exact TreeSHAP (Lundberg et al. 2018, path-dependent, polynomial
time) for the numpy tree models, KernelSHAP for any model, partial dependence + ICE, model save/load, plots
and an HTML run report."""
import base64
import io
import itertools
import json
import math

import numpy as np


# ====================================================================== permutation importance
def permutation_importance(model, X, y, metric, n_repeats=5, seed=0):
    """Drop in score when each column is shuffled (mean, std over repeats).  Higher = more important."""
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    base = metric(y, model.predict(X))
    out = np.zeros((X.shape[1], n_repeats))
    for j in range(X.shape[1]):
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            out[j, r] = base - metric(y, model.predict(Xp))
    return out.mean(1), out.std(1)


def grouped_permutation_importance(model, X, y, metric, groups, n_repeats=5, seed=0):
    """Permutation importance of groups of correlated columns shuffled together: {group: (mean, std)}."""
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    base = metric(y, model.predict(X))
    res = {}
    for g, cols in groups.items():
        vals = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X))
            Xp = X.copy()
            Xp[:, cols] = X[perm][:, cols]
            vals.append(base - metric(y, model.predict(Xp)))
        res[g] = (float(np.mean(vals)), float(np.std(vals)))
    return res


# ====================================================================== TreeSHAP
class _P:
    __slots__ = ("d", "z", "o", "w")

    def __init__(self, d, z, o, w):
        self.d, self.z, self.o, self.w = d, z, o, w


def _copy(m):
    return [_P(e.d, e.z, e.o, e.w) for e in m]


def _extend(m, pz, po, pi):
    l = len(m)
    m = _copy(m) + [_P(pi, pz, po, 1.0 if l == 0 else 0.0)]
    for i in range(l - 1, -1, -1):
        m[i + 1].w += po * m[i].w * (i + 1) / (l + 1)
        m[i].w = pz * m[i].w * (l - i) / (l + 1)
    return m


def _unwind(m, i):
    l = len(m) - 1
    m = _copy(m)
    n = m[l].w
    for j in range(l - 1, -1, -1):
        if m[i].o != 0:
            t = m[j].w
            m[j].w = n * (l + 1) / ((j + 1) * m[i].o)
            n = t - m[j].w * m[i].z * (l - j) / (l + 1)
        else:
            m[j].w = m[j].w * (l + 1) / (m[i].z * (l - j))
    for j in range(i, l):
        m[j].d, m[j].z, m[j].o = m[j + 1].d, m[j + 1].z, m[j + 1].o
    return m[:l]


def _unwound_sum(m, i):
    l = len(m) - 1
    total = 0.0
    n = m[l].w
    if m[i].o != 0:
        for j in range(l - 1, -1, -1):
            t = n * (l + 1) / ((j + 1) * m[i].o)
            total += t
            n = m[j].w - t * m[i].z * (l - j) / (l + 1)
    else:
        for j in range(l - 1, -1, -1):
            total += m[j].w * (l + 1) / (m[i].z * (l - j))
    return total


def _goes_left(tree, j, x):
    v = x[tree.feature[j]]
    return bool(tree.nan_left[j]) if np.isnan(v) else bool(v < tree.threshold[j])


def tree_shap_single(tree, x, M):
    """Exact SHAP values of one tree for one sample x (length M)."""
    phi = np.zeros(M)

    def recurse(j, m, pz, po, pi):
        m = _extend(m, pz, po, pi)
        if tree.feature[j] < 0:
            for i in range(1, len(m)):
                w = _unwound_sum(m, i)
                phi[m[i].d] += w * (m[i].o - m[i].z) * tree.value[j]
            return
        a, b = tree.left[j], tree.right[j]
        h, c = (a, b) if _goes_left(tree, j, x) else (b, a)
        iz = io = 1.0
        d = tree.feature[j]
        k = next((i for i in range(1, len(m)) if m[i].d == d), None)
        if k is not None:
            iz, io = m[k].z, m[k].o
            m = _unwind(m, k)
        rj = tree.cover[j]
        recurse(h, m, iz * tree.cover[h] / rj, io, d)
        recurse(c, m, iz * tree.cover[c] / rj, 0.0, d)

    recurse(0, [], 1.0, 1.0, -1)
    return phi


def tree_expectation(tree, x=None, S=()):
    """E[f(x) | x_S] by the tree's path-dependent rule (cover-weighted for features outside S).
    S empty -> the tree's expected value."""
    S = set(S)

    def rec(j):
        if tree.feature[j] < 0:
            return tree.value[j]
        if tree.feature[j] in S:
            return rec(tree.left[j] if _goes_left(tree, j, x) else tree.right[j])
        a, b = tree.left[j], tree.right[j]
        return (tree.cover[a] * rec(a) + tree.cover[b] * rec(b)) / tree.cover[j]

    return rec(0)


def _trees_of(model):
    """(trees, weight, offset) such that f(x) = offset + weight * sum(tree(x))."""
    from .models import DecisionTree, GradientBoosting, RandomForest
    if isinstance(model, GradientBoosting):
        return model.trees_, 1.0, model.base_
    if isinstance(model, RandomForest):
        return model.trees_, 1.0 / len(model.trees_), 0.0
    if isinstance(model, DecisionTree):
        return [model.tree_], 1.0, 0.0
    raise TypeError("TreeSHAP needs a numpy tree model (DecisionTree, RandomForest, GradientBoosting)")


def tree_shap(model, X):
    """SHAP values (n, M) and expected value for a tree model (raw output; log-odds for logistic GBM).
    Local accuracy: expected + phi.sum(1) == model raw prediction."""
    X = np.atleast_2d(np.asarray(X, float))
    trees, w, off = _trees_of(model)
    M = X.shape[1]
    phi = np.zeros(X.shape)
    for t in trees:
        for i in range(len(X)):
            phi[i] += tree_shap_single(t, X[i], M)
    expected = off + w * sum(tree_expectation(t) for t in trees)
    return phi * w, float(expected)


def brute_force_shap(model, x):
    """Exact Shapley values by enumerating all coalitions with the trees' path-dependent value function
    (for testing TreeSHAP on small models)."""
    trees, w, off = _trees_of(model)
    M = len(x)

    def v(S):
        return off + w * sum(tree_expectation(t, x, S) for t in trees)

    phi = np.zeros(M)
    for i in range(M):
        others = [j for j in range(M) if j != i]
        for r in range(M):
            for S in itertools.combinations(others, r):
                wt = math.factorial(r) * math.factorial(M - r - 1) / math.factorial(M)
                phi[i] += wt * (v(set(S) | {i}) - v(S))
    return phi


# ====================================================================== KernelSHAP
def kernel_shap(f, x, background, n_samples=2048, seed=0):
    """Model-agnostic SHAP (Lundberg & Lee 2017): weighted linear regression over feature coalitions, missing
    features filled from background rows.  Exact enumeration when 2^M is small."""
    x = np.asarray(x, float)
    B = np.atleast_2d(np.asarray(background, float))
    M = len(x)
    fx = float(f(x[None])[0])
    ef = float(np.mean(f(B)))
    rng = np.random.default_rng(seed)
    if 2 ** M - 2 <= n_samples:
        Z = np.array([[int(c) for c in np.binary_repr(k, M)] for k in range(1, 2 ** M - 1)], float)
    else:
        sizes = np.arange(1, M)
        pw = np.array([(M - 1) / (s * (M - s)) for s in sizes])
        pw /= pw.sum()
        Z = np.zeros((n_samples, M))
        for r in range(n_samples):
            s = rng.choice(sizes, p=pw)
            Z[r, rng.choice(M, s, replace=False)] = 1
    s = Z.sum(1)
    wts = (M - 1) / (np.array([math.comb(M, int(k)) for k in s]) * s * (M - s))
    ey = np.empty(len(Z))
    for r, z in enumerate(Z):
        Xr = B.copy()
        Xr[:, z > 0] = x[z > 0]
        ey[r] = np.mean(f(Xr))
    # constraint sum(phi) = fx - ef: eliminate the last feature
    yv = ey - ef - Z[:, -1] * (fx - ef)
    A = Z[:, :-1] - Z[:, -1:]
    W = np.sqrt(wts)
    sol = np.linalg.lstsq(A * W[:, None], yv * W, rcond=None)[0]
    phi = np.r_[sol, (fx - ef) - sol.sum()]
    return phi, ef


# ====================================================================== partial dependence
def partial_dependence(model, X, j, grid=None, ice=False, n_grid=25):
    X = np.asarray(X, float)
    grid = np.linspace(np.nanpercentile(X[:, j], 2), np.nanpercentile(X[:, j], 98), n_grid) if grid is None else grid
    curves = np.zeros((len(X), len(grid)))
    for k, g in enumerate(grid):
        Xg = X.copy()
        Xg[:, j] = g
        curves[:, k] = model.predict(Xg)
    return (grid, curves.mean(0), curves) if ice else (grid, curves.mean(0))


# ====================================================================== maps + persistence
def ensemble_map(models, Xnodes):
    """Mean and spread (std) of predictions of several models (e.g. one per CV fold) at map nodes."""
    P = np.array([m.predict(Xnodes) for m in models])
    return P.mean(0), P.std(0)


def _tree_dict(t):
    return {k: getattr(t, k).tolist() for k in ("feature", "threshold", "nan_left", "left", "right", "value",
                                                "cover", "gain")}


def save_model(model, path, names=None):
    from .models import GradientBoosting, LinearModel, Lasso, RandomForest
    d = {"type": type(model).__name__, "params": model.get_params(), "names": names}
    if isinstance(model, (GradientBoosting, RandomForest)):
        d["trees"] = [_tree_dict(t) for t in model.trees_]
        d["edges"] = [e.tolist() for e in model.binner.edges]
        if isinstance(model, GradientBoosting):
            d["base"] = model.base_
    elif isinstance(model, (LinearModel, Lasso)):
        d["coef"], d["intercept"] = model.coef_.tolist(), float(model.intercept_)
    else:
        raise TypeError(f"saving {type(model).__name__} not supported")
    with open(path, "w") as fh:
        json.dump(d, fh)
    return path


def load_model(path):
    from . import models as M
    with open(path) as fh:
        d = json.load(fh)
    cls = getattr(M, d["type"])
    params = d["params"]
    if "monotone" in params and params["monotone"]:
        params["monotone"] = {int(k): v for k, v in params["monotone"].items()}
    m = cls(**params)
    if "trees" in d:
        m.trees_ = []
        for td in d["trees"]:
            t = M.Tree()
            for k, v in td.items():
                setattr(t, k, v)
            m.trees_.append(t.finish())
        m.binner = M.Binner(params.get("n_bins", 64))
        m.binner.edges = [np.array(e) for e in d["edges"]]
        if "base" in d:
            m.base_ = d["base"]
    else:
        m.coef_, m.intercept_ = np.array(d["coef"]), d["intercept"]
    m.names_ = d.get("names")
    return m


# ====================================================================== plots + report
def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_shap_summary(phi, X, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(np.abs(phi).mean(0))
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(names) + 1.5))
    rng = np.random.default_rng(0)
    for r, j in enumerate(order):
        v = X[:, j]
        c = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-12)
        ax.scatter(phi[:, j], r + 0.25 * rng.uniform(-1, 1, len(v)), c=c, cmap="coolwarm", s=10)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[j] for j in order])
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("SHAP value (impact on prediction); colour = input value (blue low, red high)")
    fig.tight_layout()
    return fig


def plot_bar(values, names, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(names) + 1.2))
    ax.barh([names[j] for j in order], np.asarray(values)[order], color="#4a7bb7")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_dependence(phi, X, names, j, color_by=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    c = X[:, color_by] if color_by is not None else None
    ax.scatter(X[:, j], phi[:, j], c=c, cmap="coolwarm", s=12)
    ax.set_xlabel(names[j])
    ax.set_ylabel(f"SHAP {names[j]}")
    fig.tight_layout()
    return fig


def plot_waterfall(phi_row, expected, names, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(np.abs(phi_row))[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(names) + 1.2))
    cur = expected
    for r, j in enumerate(order):
        ax.barh(r, phi_row[j], left=cur, color="#d6604d" if phi_row[j] > 0 else "#4393c3")
        cur += phi_row[j]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[j] for j in order])
    ax.invert_yaxis()
    ax.set_title(f"{title}  E[f] {expected:.3g} -> f(x) {cur:.3g}")
    fig.tight_layout()
    return fig


def plot_pdp(model, X, names, j):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid, pd, ice = partial_dependence(model, X, j, ice=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    for c in ice[:: max(1, len(ice) // 50)]:
        ax.plot(grid, c, color="0.8", lw=0.6)
    ax.plot(grid, pd, color="k", lw=2)
    ax.set_xlabel(names[j])
    ax.set_ylabel("prediction (PDP black, ICE grey)")
    fig.tight_layout()
    return fig


def html_report(path, title, sections):
    """sections: [(heading, html text or ('png', fig) )]."""
    parts = [f"<html><head><meta charset='utf-8'><title>{title}</title>"
             "<style>body{font-family:Segoe UI,Arial;margin:24px;max-width:1000px}table{border-collapse:collapse}"
             "td,th{border:1px solid #ccc;padding:3px 8px}</style></head><body>", f"<h1>{title}</h1>"]
    for head, content in sections:
        parts.append(f"<h2>{head}</h2>")
        if isinstance(content, tuple) and content[0] == "png":
            parts.append(f"<img src='data:image/png;base64,{_png(content[1])}'/>")
        else:
            parts.append(str(content))
    parts.append("</body></html>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    return path


def table_html(rows, header):
    h = "".join(f"<th>{c}</th>" for c in header)
    body = "".join("<tr>" + "".join(f"<td>{v:.4g}</td>" if isinstance(v, float) else f"<td>{v}</td>" for v in r) +
                   "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{body}</table>"
