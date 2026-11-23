"""Model QC toolkit (13d): validation designs, baselines, leakage tests, residual diagnostics (spatial),
uncertainty (quantile GBM, split conformal, calibration), applicability domain, stability, sanity checks,
model comparison and a one-page HTML QC report."""
import math

import numpy as np

from . import validation as V


# ====================================================================== validation designs
def blind_split(ids, n_blind=None, frac=0.2, seed=0):
    """Wells never used in training or tuning: (train mask, blind mask)."""
    u = np.unique(ids)
    rng = np.random.default_rng(seed)
    k = n_blind or max(1, int(round(frac * len(u))))
    blind = rng.choice(u, k, replace=False)
    b = np.isin(ids, blind)
    return ~b, b


def temporal_split(dates, frac_train=0.7):
    """Train on the older wells, test on the newer (dates: numeric, e.g. months)."""
    d = np.asarray(dates, float)
    cut = np.nanquantile(d, frac_train)
    return d <= cut, d > cut


def repeated_cv_scores(model, X, y, groups, k=5, repeats=5, metric=V.r2):
    """Spread of the group-CV score over random fold assignments."""
    return np.array([V.r2(y, V.cross_val_predict(model, X, y, V.group_kfold(groups, k, seed=r))) if metric is V.r2
                     else metric(y, V.cross_val_predict(model, X, y, V.group_kfold(groups, k, seed=r)))
                     for r in range(repeats)])


# ====================================================================== baselines + leakage
def idw_loo(xy, y, power=2.0):
    """Leave-one-out inverse-distance prediction of the target from the other wells (spatial baseline)."""
    xy = np.asarray(xy, float)
    d2 = ((xy[:, None] - xy[None]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    w = 1.0 / np.maximum(d2, 1e-9) ** (power / 2)
    return (w * y[None]).sum(1) / w.sum(1)


def baselines(X, y, xy, groups, names):
    """CV R2 of: mean, IDW of the target alone, best single-input linear regression."""
    from .models import LinearModel
    out = {"mean": V.r2(y, V.cross_val_predict(LinearModel(), np.zeros((len(y), 1)), y, V.group_kfold(groups, 5)))}
    if xy is not None:
        out["IDW of target"] = V.r2(y, idw_loo(xy, y))
    best = (-np.inf, None)
    for j, nm in enumerate(names):
        s = V.r2(y, V.cross_val_predict(LinearModel(), X[:, [j]], y, V.group_kfold(groups, 5)))
        if s > best[0]:
            best = (s, nm)
    out[f"best single input ({best[1]})"] = best[0]
    return out


def y_randomization(model, X, y, groups, n=5, seed=0):
    """CV R2 with the target shuffled - should collapse to <= 0; a high value means leakage / overfitting."""
    rng = np.random.default_rng(seed)
    return np.array([V.r2(ys, V.cross_val_predict(model, X, ys, V.group_kfold(groups, 5)))
                     for ys in (rng.permutation(y) for _ in range(n))])


def noise_feature_test(model, X, y, seed=0):
    """Add a random column; returns (permutation importances, index of the noise column). Real inputs should
    out-rank it."""
    from .explain import permutation_importance
    rng = np.random.default_rng(seed)
    Xn = np.column_stack([X, rng.normal(size=len(X))])
    m = model.clone().fit(Xn, y)
    imp, _ = permutation_importance(m, Xn, y, V.r2, 5, seed)
    return imp, Xn.shape[1] - 1


# ====================================================================== residual diagnostics
def residual_diagnostics(y, pred, X=None, names=None):
    r = np.asarray(y, float) - np.asarray(pred, float)
    out = {"resid_vs_pred_corr": float(np.corrcoef(pred, r)[0, 1]) if np.std(r) > 0 and np.std(pred) > 0 else 0.0}
    if X is not None:
        out["resid_vs_input_corr"] = {nm: float(np.corrcoef(X[:, j], r)[0, 1]) if np.std(X[:, j]) > 0 else 0.0
                                      for j, nm in enumerate(names)}
    q = np.sort((r - r.mean()) / (r.std() or 1))
    p = (np.arange(1, len(q) + 1) - 0.5) / len(q)
    theo = np.array([_norm_ppf(v) for v in p])
    out["qq"] = (theo, q)
    out["qq_corr"] = float(np.corrcoef(theo, q)[0, 1]) if len(q) > 2 else 1.0
    return out


def _norm_ppf(p):
    # Acklam's rational approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - pl:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def variogram(xy, v, n_bins=10, max_dist=None):
    """Experimental semivariogram: (bin centres, semivariance, pair counts)."""
    xy = np.asarray(xy, float)
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    g = 0.5 * (v[:, None] - v[None]) ** 2
    iu = np.triu_indices(len(v), 1)
    d, g = d[iu], g[iu]
    max_dist = max_dist or np.percentile(d, 60)
    edges = np.linspace(0, max_dist, n_bins + 1)
    k = np.digitize(d, edges) - 1
    ok = (k >= 0) & (k < n_bins)
    cnt = np.bincount(k[ok], minlength=n_bins)
    sv = np.bincount(k[ok], g[ok], minlength=n_bins) / np.maximum(cnt, 1)
    return 0.5 * (edges[1:] + edges[:-1]), sv, cnt


def morans_i(xy, v, power=1.0):
    """Moran's I with inverse-distance weights, and its z-score under randomisation (normal approximation).
    Significant positive I in the residuals = spatial structure left = a missing input."""
    xy = np.asarray(xy, float)
    v = np.asarray(v, float) - np.mean(v)
    n = len(v)
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    W = np.where(d > 0, 1.0 / np.maximum(d, 1e-9) ** power, 0.0)
    S0 = W.sum()
    I = n / S0 * (v @ W @ v) / (v @ v)
    EI = -1.0 / (n - 1)
    S1 = 0.5 * ((W + W.T) ** 2).sum()
    S2 = ((W.sum(0) + W.sum(1)) ** 2).sum()
    VI = (n * n * S1 - n * S2 + 3 * S0 * S0) / ((n * n - 1) * S0 * S0) - EI * EI
    return float(I), float((I - EI) / math.sqrt(max(VI, 1e-18)))


# ====================================================================== uncertainty
def conformal_intervals(model, X, y, Xnew, alpha=0.2, groups=None, seed=0):
    """Split-conformal prediction intervals ((1 - alpha) coverage): calibrate absolute residuals on held-out
    wells.  Returns (lo, hi, q)."""
    rng = np.random.default_rng(seed)
    g = np.asarray(groups) if groups is not None else np.arange(len(y))
    u = rng.permutation(np.unique(g))
    if len(u) < 3:
        raise ValueError("conformal intervals need at least 3 wells (groups): one third is held out for calibration")
    cal = np.isin(g, u[: len(u) // 3])
    m = model.clone().fit(X[~cal], y[~cal])
    res = np.abs(y[cal] - m.predict(X[cal]))
    n = len(res)
    q = np.quantile(res, min(1.0, math.ceil((n + 1) * (1 - alpha)) / n))
    p = m.predict(Xnew)
    return p - q, p + q, float(q)


def quantile_intervals(X, y, Xnew, lo=0.1, hi=0.9, **kw):
    from .models import GradientBoosting
    kw = dict({"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3}, **kw)
    ml = GradientBoosting(objective="quantile", quantile=lo, **kw).fit(X, y)
    mh = GradientBoosting(objective="quantile", quantile=hi, **kw).fit(X, y)
    return ml.predict(Xnew), mh.predict(Xnew)


def coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def calibration(y, prob, n_bins=10):
    """Reliability diagram data (bin mean predicted, observed frequency, count) and Brier score."""
    y, prob = np.asarray(y, float), np.asarray(prob, float)
    k = np.clip((prob * n_bins).astype(int), 0, n_bins - 1)
    cnt = np.bincount(k, minlength=n_bins)
    mp = np.bincount(k, prob, minlength=n_bins) / np.maximum(cnt, 1)
    ob = np.bincount(k, y, minlength=n_bins) / np.maximum(cnt, 1)
    return (mp, ob, cnt), float(np.mean((prob - y) ** 2))


# ====================================================================== applicability domain
def applicability(Xtrain, Xnew, quantile=0.975, k=5):
    """Mahalanobis distance of new rows to the training cloud and mean distance to the k nearest training rows
    (standardised).  Returns (inside mask, mahalanobis, knn distance); outside = beyond the training rows'
    own `quantile` of either measure."""
    Xt = np.asarray(Xtrain, float)
    mu = Xt.mean(0)
    C = np.cov(Xt.T) + 1e-9 * np.eye(Xt.shape[1])
    Ci = np.linalg.inv(np.atleast_2d(C))
    md = lambda A: np.sqrt(np.einsum("ij,jk,ik->i", A - mu, Ci, A - mu))  # noqa: E731
    sd = Xt.std(0)
    sd = np.where(sd > 0, sd, 1)
    Zt, Zn = (Xt - mu) / sd, (np.asarray(Xnew, float) - mu) / sd

    def knn(A, self_ex=False):
        d = np.sqrt(((A[:, None] - Zt[None]) ** 2).sum(-1))
        if self_ex:
            np.fill_diagonal(d, np.inf)
        return np.sort(d, 1)[:, :k].mean(1)

    mt, kt = md(Xt), knn(Zt, True)
    mn, kn = md(np.asarray(Xnew, float)), knn(Zn)
    inside = (mn <= np.quantile(mt, quantile)) & (kn <= np.quantile(kt, quantile))
    return inside, mn, kn


# ====================================================================== stability + sanity + comparison
def importance_stability(model, X, y, groups, n_boot=5, seed=0):
    """Rank correlation (Spearman) of gain/permutation importances between bootstrap refits: (mean rho, ranks)."""
    from .explain import permutation_importance
    rng = np.random.default_rng(seed)
    u = np.unique(groups)
    ranks = []
    for _ in range(n_boot):
        pick = np.isin(groups, rng.choice(u, len(u), replace=True))
        m = model.clone().fit(X[pick], y[pick])
        imp, _ = permutation_importance(m, X, y, V.r2, 2, seed)
        ranks.append(np.argsort(np.argsort(-imp)))
    R = np.array(ranks, float)
    rhos = [np.corrcoef(R[a], R[b])[0, 1] for a in range(len(R)) for b in range(a + 1, len(R))]
    return float(np.mean(rhos)), R


def hyperparameter_sensitivity(model, grid_name, values, X, y, groups):
    return [(v, float(V.r2(y, V.cross_val_predict(model.clone().set_params(**{grid_name: v}), X, y,
                                                      V.group_kfold(groups, 5))))) for v in values]


def pdp_direction_checks(model, X, expected, names):
    """expected: {input name: +1 / -1}.  Returns {name: (slope sign ok, slope)} from the partial dependence."""
    from .explain import partial_dependence
    out = {}
    for nm, sgn in expected.items():
        j = names.index(nm)
        grid, pd = partial_dependence(model, X, j)
        slope = float(np.polyfit(grid, pd, 1)[0])
        out[nm] = (np.sign(slope) == np.sign(sgn), slope)
    return out


def paired_comparison(y, pred_a, pred_b, splits):
    """Per-fold squared-error difference of two models (same folds): mean difference, t statistic, two-sided
    p-value (normal approximation) and sign-test count.  Negative mean = model A better."""
    diffs = []
    for _, te in splits:
        diffs.append(np.mean((y[te] - pred_a[te]) ** 2) - np.mean((y[te] - pred_b[te]) ** 2))
    d = np.array(diffs)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    t = d.mean() / (sd / math.sqrt(len(d))) if sd > 0 else (-np.inf if d.mean() < 0 else np.inf)
    p = math.erfc(abs(t) / math.sqrt(2)) if np.isfinite(t) else 0.0
    return {"mean_diff": float(d.mean()), "t": float(t), "p": float(p), "a_better_folds": int((d < 0).sum()),
            "folds": len(d)}


# ====================================================================== report
def qc_report(path, title, items):
    """items: [(heading, html | ('png', fig))] -> one-page HTML."""
    from .explain import html_report
    return html_report(path, title, items)
