"""Data preparation and exploration: missing values, imputation, scaling, outliers, correlation, VIF, PCA,
crossplot regression lines."""
import numpy as np


def missing_report(X, names):
    X = np.asarray(X, float)
    n = len(X)
    return [(nm, int(np.isnan(X[:, k]).sum()), float(np.isnan(X[:, k]).mean() * 100) if n else 0.0)
            for k, nm in enumerate(names)]


def impute(X, method="median", k=5):
    """Fill NaN: 'median' per column, or 'knn' (mean of the k nearest rows in the standardised observed
    columns).  Returns a new array."""
    X = np.asarray(X, float).copy()
    nan = np.isnan(X)
    med = np.nanmedian(np.where(np.all(nan, 0), 0.0, X), axis=0) if len(X) else np.zeros(X.shape[1])
    if method == "median" or not nan.any():
        X[nan] = np.take(med, np.nonzero(nan)[1])
        return X
    mu = np.nanmean(X, 0)
    sd = np.nanstd(X, 0)
    sd = np.where(sd > 0, sd, 1)
    Z = (X - mu) / sd
    for i in np.nonzero(nan.any(1))[0]:
        obs = ~nan[i]
        if not obs.any():
            X[i] = med
            continue
        cand = np.nonzero(~nan[:, ~obs].any(1))[0]
        cand = cand[cand != i]
        if not len(cand):
            X[i, ~obs] = med[~obs]
            continue
        d = np.sqrt(np.nanmean((Z[cand][:, obs] - Z[i, obs]) ** 2, axis=1))
        nn = cand[np.argsort(d)[:k]]
        X[i, ~obs] = X[nn][:, ~obs].mean(0)
    return X


class Scaler:
    def fit(self, X):
        X = np.asarray(X, float)
        self.mu = np.nanmean(X, 0)
        sd = np.nanstd(X, 0)
        self.sd = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mu) / self.sd

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse(self, Z):
        return np.asarray(Z) * self.sd + self.mu


def outlier_flags(X, z=3.5):
    """Robust z-score (median / MAD) > z per cell."""
    X = np.asarray(X, float)
    med = np.nanmedian(X, 0)
    mad = np.nanmedian(np.abs(X - med), 0) * 1.4826
    mad = np.where(mad > 0, mad, np.nan)
    with np.errstate(invalid="ignore"):
        return np.abs(X - med) / mad > z


def correlation(X, method="pearson"):
    X = np.asarray(X, float)
    if method == "spearman":
        X = np.column_stack([_rank(X[:, k]) for k in range(X.shape[1])])
    ok = np.all(np.isfinite(X), 1)
    return np.corrcoef(X[ok].T) if ok.sum() > 2 else np.full((X.shape[1],) * 2, np.nan)


def _rank(v):
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    r[order] = np.arange(len(v))
    # average ties
    vs = v[order]
    i = 0
    while i < len(vs):
        j = i
        while j + 1 < len(vs) and vs[j + 1] == vs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def vif(X):
    """Variance inflation factor per column: 1 / (1 - R^2) of each column regressed on the others."""
    X = np.asarray(X, float)
    X = X[np.all(np.isfinite(X), 1)]
    out = []
    for k in range(X.shape[1]):
        y = X[:, k]
        A = np.column_stack([np.ones(len(X)), np.delete(X, k, 1)])
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        res = y - A @ coef
        r2 = 1 - res.var() / y.var() if y.var() > 0 else 0.0
        out.append(np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2))
    return np.array(out)


def pca(X, n=None):
    """PCA on standardised data: dict(explained (fraction), loadings (k, p), scores (n, k), scaler)."""
    sc = Scaler().fit(X)
    Z = sc.transform(X)
    Z = np.where(np.isfinite(Z), Z, 0.0)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    ev = S ** 2 / max(1, len(Z) - 1)
    frac = ev / ev.sum() if ev.sum() > 0 else ev
    n = n or len(S)
    return {"explained": frac[:n], "loadings": Vt[:n], "scores": U[:, :n] * S[:n], "scaler": sc}


def regression_line(x, y):
    """Least-squares y = a + b x and Pearson r for a crossplot."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan
    b, a = np.polyfit(x[ok], y[ok], 1)
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    return float(a), float(b), float(r)
