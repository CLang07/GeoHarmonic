"""Models (numpy, sklearn-like API: fit / predict / get_params / set_params).

Regression: LinearModel (OLS / ridge), Lasso (coordinate descent), KNN, DecisionTree, RandomForest,
GradientBoosting (XGBoost style: second-order gradients, histogram splits, L1/L2 leaf regularisation, gamma,
min_child_weight, shrinkage, row/column subsampling, early stopping, optional monotone constraints), MLP.
Classification: GradientBoosting(objective="logistic"), KNN(classify=True).
Clustering: KMeans (k-means++), GaussianMixture (diagonal EM), SOM.  Metrics in validation.py.
Optional backends: make_model("xgboost", ...) / ("sklearn-rf", ...) use the libraries if they import.
"""
import numpy as np


class Model:
    params = ()

    def get_params(self):
        return {k: getattr(self, k) for k in self.params}

    def set_params(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self

    def clone(self):
        return type(self)(**self.get_params())


def _std(X):
    mu = np.nanmean(X, 0)
    sd = np.nanstd(X, 0)
    return mu, np.where(sd > 0, sd, 1.0)


# ====================================================================== linear
class LinearModel(Model):
    params = ("alpha",)

    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu, self.sd = _std(X)
        Z = (X - self.mu) / self.sd
        self.y0 = y.mean()
        p = Z.shape[1]
        A = Z.T @ Z + self.alpha * len(Z) * np.eye(p)
        self.w = np.linalg.solve(A + 1e-12 * np.eye(p), Z.T @ (y - self.y0))
        self.coef_ = self.w / self.sd
        self.intercept_ = self.y0 - self.coef_ @ self.mu
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


class Lasso(Model):
    params = ("alpha", "max_iter", "tol")

    def __init__(self, alpha=0.01, max_iter=2000, tol=1e-7):
        self.alpha, self.max_iter, self.tol = alpha, max_iter, tol

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu, self.sd = _std(X)
        Z = (X - self.mu) / self.sd
        self.y0 = y.mean()
        r = y - self.y0
        n, p = Z.shape
        w = np.zeros(p)
        zz = (Z * Z).sum(0) / n
        for _ in range(self.max_iter):
            wmax = 0.0
            for j in range(p):
                old = w[j]
                rho = Z[:, j] @ r / n + zz[j] * old
                new = np.sign(rho) * max(abs(rho) - self.alpha, 0.0) / zz[j] if zz[j] > 0 else 0.0
                if new != old:
                    r -= Z[:, j] * (new - old)
                    w[j] = new
                    wmax = max(wmax, abs(new - old))
            if wmax < self.tol:
                break
        self.w = w
        self.coef_ = w / self.sd
        self.intercept_ = self.y0 - self.coef_ @ self.mu
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


# ====================================================================== kNN
class KNN(Model):
    params = ("k", "weights", "classify")

    def __init__(self, k=5, weights="distance", classify=False):
        self.k, self.weights, self.classify = k, weights, classify

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.mu, self.sd = _std(X)
        self.Z = (X - self.mu) / self.sd
        self.y = np.asarray(y)
        return self

    def predict(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        d = np.sqrt(((Z[:, None, :] - self.Z[None]) ** 2).sum(-1))
        k = min(self.k, len(self.Z))
        nn = np.argsort(d, 1)[:, :k]
        dn = np.take_along_axis(d, nn, 1)
        w = 1.0 / np.maximum(dn, 1e-9) if self.weights == "distance" else np.ones_like(dn)
        if self.classify:
            labels = np.unique(self.y)
            votes = np.stack([(w * (self.y[nn] == c)).sum(1) for c in labels], 1)
            return labels[np.argmax(votes, 1)]
        return (w * self.y[nn]).sum(1) / w.sum(1)


# ====================================================================== trees (shared histogram builder)
class Binner:
    def __init__(self, n_bins=64):
        self.n_bins = n_bins

    def fit(self, X):
        X = np.asarray(X, float)
        self.edges = []
        for j in range(X.shape[1]):
            v = X[:, j][np.isfinite(X[:, j])]
            if not len(v):
                self.edges.append(np.array([]))
                continue
            q = np.unique(np.quantile(v, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
            self.edges.append(q)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        B = np.empty(X.shape, np.int32)
        for j in range(X.shape[1]):
            b = np.searchsorted(self.edges[j], X[:, j], side="right")
            B[:, j] = np.where(np.isfinite(X[:, j]), b, self.n_bins)  # NaN -> missing bin
        return B

    def threshold(self, j, b):
        """Split 'bin <= b' as a value threshold 'x < t'."""
        e = self.edges[j]
        return e[b] if b < len(e) else np.inf


class Tree:
    """Arrays: feature, threshold (go left if x < threshold), nan_left, left, right, value, cover, gain."""

    def __init__(self):
        self.feature, self.threshold, self.nan_left = [], [], []
        self.left, self.right, self.value, self.cover, self.gain = [], [], [], [], []

    def add(self):
        for a in (self.feature, self.threshold, self.nan_left, self.left, self.right, self.value, self.cover, self.gain):
            a.append(0)
        self.feature[-1] = -1
        return len(self.feature) - 1

    def finish(self):
        self.feature = np.array(self.feature, int)
        self.threshold = np.array(self.threshold, float)
        self.nan_left = np.array(self.nan_left, bool)
        self.left, self.right = np.array(self.left, int), np.array(self.right, int)
        self.value, self.cover, self.gain = (np.array(a, float) for a in (self.value, self.cover, self.gain))
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        node = np.zeros(len(X), int)
        active = self.feature[node] >= 0
        while active.any():
            idx = np.nonzero(active)[0]
            f = self.feature[node[idx]]
            x = X[idx, f]
            go_left = np.where(np.isnan(x), self.nan_left[node[idx]], x < self.threshold[node[idx]])
            node[idx] = np.where(go_left, self.left[node[idx]], self.right[node[idx]])
            active = self.feature[node] >= 0
        return self.value[node]


def _soft(g, alpha):
    return np.sign(g) * np.maximum(np.abs(g) - alpha, 0.0)


def build_tree(B, binner, g, h, rows, cols, max_depth=6, min_child_weight=1.0, lam=1.0, alpha=0.0, gamma=0.0,
               min_samples_leaf=1, monotone=None, rng=None, max_features=None):
    """Histogram tree on binned data B (int, NaN = n_bins) for gradients g / hessians h (XGBoost gain).
    rows: sample indices; cols: candidate features.  monotone: {feature: +1/-1}."""
    nb = binner.n_bins
    tree = Tree()
    stack = [(tree.add(), rows, 0, -np.inf, np.inf)]
    while stack:
        nid, idx, depth, lo, hi = stack.pop()
        G, H = g[idx].sum(), h[idx].sum()
        w = -_soft(G, alpha) / (H + lam)
        w = float(np.clip(w, lo, hi))
        tree.value[nid], tree.cover[nid] = w, H
        if depth >= max_depth or len(idx) < 2 * min_samples_leaf or H < 2 * min_child_weight:
            continue
        parent = _soft(G, alpha) ** 2 / (H + lam)
        best = (0.0, None)
        fcols = cols
        if max_features and max_features < len(cols):
            fcols = rng.choice(cols, max_features, replace=False)
        for j in fcols:
            bj = B[idx, j]
            Gb = np.bincount(bj, g[idx], minlength=nb + 1)
            Hb = np.bincount(bj, h[idx], minlength=nb + 1)
            Nb = np.bincount(bj, minlength=nb + 1)
            Gm, Hm, Nm = Gb[nb], Hb[nb], Nb[nb]
            GL, HL, NL = np.cumsum(Gb[:nb])[:-1], np.cumsum(Hb[:nb])[:-1], np.cumsum(Nb[:nb])[:-1]
            for nan_left in (True, False):
                gl = GL + (Gm if nan_left else 0)
                hl = HL + (Hm if nan_left else 0)
                nl = NL + (Nm if nan_left else 0)
                gr, hr, nr = G - gl, H - hl, len(idx) - nl
                ok = (hl >= min_child_weight) & (hr >= min_child_weight) & (nl >= min_samples_leaf) & \
                    (nr >= min_samples_leaf)
                if not ok.any():
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    gain = 0.5 * (_soft(gl, alpha) ** 2 / (hl + lam) + _soft(gr, alpha) ** 2 / (hr + lam) - parent) - gamma
                if monotone and j in monotone:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        wl = -_soft(gl, alpha) / (hl + lam)
                        wr = -_soft(gr, alpha) / (hr + lam)
                    ok &= (wr - wl) * monotone[j] >= 0
                gain = np.where(ok, gain, -np.inf)
                b = int(np.argmax(gain))
                if gain[b] > best[0]:
                    best = (float(gain[b]), (j, b, nan_left))
        if best[1] is None:
            continue
        j, b, nan_left = best[1]
        bj = B[idx, j]
        go_left = np.where(bj == nb, nan_left, bj <= b)
        li, ri = idx[go_left], idx[~go_left]
        tree.feature[nid] = j
        tree.threshold[nid] = binner.threshold(j, b)
        tree.nan_left[nid] = nan_left
        tree.gain[nid] = best[0]
        lid, rid = tree.add(), tree.add()
        tree.left[nid], tree.right[nid] = lid, rid
        llo, lhi, rlo, rhi = lo, hi, lo, hi
        if monotone and j in monotone:  # children bounded by the midpoint (XGBoost-style monotone constraints)
            wl = float(np.clip(-_soft(g[li].sum(), alpha) / (h[li].sum() + lam), lo, hi))
            wr = float(np.clip(-_soft(g[ri].sum(), alpha) / (h[ri].sum() + lam), lo, hi))
            mid = 0.5 * (wl + wr)
            if monotone[j] > 0:
                lhi, rlo = min(hi, mid), max(lo, mid)
            else:
                llo, rhi = max(lo, mid), min(hi, mid)
        stack.append((lid, li, depth + 1, llo, lhi))
        stack.append((rid, ri, depth + 1, rlo, rhi))
    return tree.finish()


class DecisionTree(Model):
    params = ("max_depth", "min_samples_leaf", "n_bins")

    def __init__(self, max_depth=6, min_samples_leaf=2, n_bins=128):
        self.max_depth, self.min_samples_leaf, self.n_bins = max_depth, min_samples_leaf, n_bins

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.binner = Binner(self.n_bins).fit(X)
        B = self.binner.transform(X)
        self.tree_ = build_tree(B, self.binner, -y, np.ones(len(y)), np.arange(len(y)), np.arange(X.shape[1]),
                                self.max_depth, 0.0, 0.0, 0.0, 0.0, self.min_samples_leaf)
        return self

    def predict(self, X):
        return self.tree_.predict(X)


class RandomForest(Model):
    params = ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "seed")

    def __init__(self, n_estimators=100, max_depth=8, min_samples_leaf=2, max_features="sqrt", seed=0):
        self.n_estimators, self.max_depth, self.min_samples_leaf = n_estimators, max_depth, min_samples_leaf
        self.max_features, self.seed = max_features, seed

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        self.binner = Binner(64).fit(X)
        B = self.binner.transform(X)
        p = X.shape[1]
        mf = {"sqrt": max(1, int(np.sqrt(p))), "third": max(1, p // 3), None: p}.get(self.max_features, self.max_features)
        self.trees_ = []
        n = len(y)
        for _ in range(self.n_estimators):
            rows = rng.integers(0, n, n)
            self.trees_.append(build_tree(B, self.binner, -y, np.ones(n), rows, np.arange(p), self.max_depth, 0.0,
                                          0.0, 0.0, 0.0, self.min_samples_leaf, rng=rng, max_features=mf))
        self.expected_value = float(np.mean([t.value[0] for t in self.trees_]))
        return self

    def predict(self, X):
        return np.mean([t.predict(X) for t in self.trees_], 0)


class GradientBoosting(Model):
    params = ("n_estimators", "learning_rate", "max_depth", "min_child_weight", "reg_lambda", "reg_alpha", "gamma",
              "subsample", "colsample", "n_bins", "objective", "early_stopping", "monotone", "seed", "quantile")

    def __init__(self, n_estimators=300, learning_rate=0.1, max_depth=4, min_child_weight=1.0, reg_lambda=1.0,
                 reg_alpha=0.0, gamma=0.0, subsample=0.8, colsample=1.0, n_bins=64, objective="squared",
                 early_stopping=None, monotone=None, seed=0, quantile=0.5):
        self.n_estimators, self.learning_rate, self.max_depth = n_estimators, learning_rate, max_depth
        self.min_child_weight, self.reg_lambda, self.reg_alpha, self.gamma = min_child_weight, reg_lambda, reg_alpha, gamma
        self.subsample, self.colsample, self.n_bins, self.objective = subsample, colsample, n_bins, objective
        self.early_stopping, self.monotone, self.seed = early_stopping, monotone, seed
        self.quantile = quantile

    def _grad(self, y, f):
        if self.objective == "logistic":
            p = 1.0 / (1.0 + np.exp(-f))
            return p - y, np.maximum(p * (1 - p), 1e-6)
        if self.objective == "quantile":  # pinball loss; unit hessian (gradient-descent leaf steps)
            return np.where(y > f, -self.quantile, 1.0 - self.quantile), np.ones_like(y)
        return f - y, np.ones_like(y)

    def fit(self, X, y, X_val=None, y_val=None):
        X, y = np.asarray(X, float), np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        self.binner = Binner(self.n_bins).fit(X)
        B = self.binner.transform(X)
        n, p = X.shape
        if self.objective == "logistic":
            m = np.clip(y.mean(), 1e-6, 1 - 1e-6)
            self.base_ = float(np.log(m / (1 - m)))
        elif self.objective == "quantile":
            self.base_ = float(np.quantile(y, self.quantile))
        else:
            self.base_ = float(y.mean())
        f = np.full(n, self.base_)
        fv = None if X_val is None else np.full(len(X_val), self.base_)
        self.trees_, self.val_curve_ = [], []
        best, best_k = np.inf, 0
        for k in range(self.n_estimators):
            g, h = self._grad(y, f)
            rows = np.nonzero(rng.random(n) < self.subsample)[0] if self.subsample < 1 else np.arange(n)
            cols = np.sort(rng.choice(p, max(1, int(round(self.colsample * p))), replace=False)) \
                if self.colsample < 1 else np.arange(p)
            t = build_tree(B, self.binner, g, h, rows, cols, self.max_depth, self.min_child_weight, self.reg_lambda,
                           self.reg_alpha, self.gamma, 1, self.monotone)
            t.value *= self.learning_rate
            self.trees_.append(t)
            f += t.predict(X)
            if fv is not None:
                fv += t.predict(X_val)
                loss = self._loss(np.asarray(y_val, float), fv)
                self.val_curve_.append(loss)
                if loss < best - 1e-12:
                    best, best_k = loss, k + 1
                elif self.early_stopping and k + 1 - best_k >= self.early_stopping:
                    break
        if fv is not None and self.early_stopping:
            self.trees_ = self.trees_[:best_k]
        return self

    def _loss(self, y, f):
        if self.objective == "logistic":
            p = np.clip(1.0 / (1.0 + np.exp(-f)), 1e-9, 1 - 1e-9)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if self.objective == "quantile":
            d = y - f
            return float(np.mean(np.maximum(self.quantile * d, (self.quantile - 1) * d)))
        return float(np.mean((y - f) ** 2))

    def raw(self, X):
        X = np.asarray(X, float)
        return self.base_ + (np.sum([t.predict(X) for t in self.trees_], 0) if self.trees_ else 0.0)

    def predict(self, X):
        f = self.raw(X)
        return (f > 0).astype(float) if self.objective == "logistic" else f

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-self.raw(X)))

    def importance(self, kind="gain"):
        p = len(self.binner.edges)
        imp = np.zeros(p)
        for t in self.trees_:
            for f, gval in zip(t.feature, t.gain):
                if f >= 0:
                    imp[f] += gval if kind == "gain" else 1
        return imp / imp.sum() if imp.sum() > 0 else imp


# ====================================================================== MLP
class MLP(Model):
    params = ("hidden", "lr", "epochs", "alpha", "seed", "patience")

    def __init__(self, hidden=16, lr=0.01, epochs=2000, alpha=1e-4, seed=0, patience=100):
        self.hidden, self.lr, self.epochs, self.alpha, self.seed, self.patience = hidden, lr, epochs, alpha, seed, patience

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        self.mu, self.sd = _std(X)
        Z = (X - self.mu) / self.sd
        self.ym, self.ys = y.mean(), y.std() or 1.0
        t = (y - self.ym) / self.ys
        p, hdn = Z.shape[1], self.hidden
        P = [rng.normal(0, 1 / np.sqrt(p), (p, hdn)), np.zeros(hdn), rng.normal(0, 1 / np.sqrt(hdn), hdn), 0.0]
        m = [np.zeros_like(a) if isinstance(a, np.ndarray) else 0.0 for a in P]
        v = [np.zeros_like(a) if isinstance(a, np.ndarray) else 0.0 for a in P]
        best, best_P, wait = np.inf, None, 0
        for ep in range(1, self.epochs + 1):
            H = np.tanh(Z @ P[0] + P[1])
            out = H @ P[2] + P[3]
            e = out - t
            loss = np.mean(e * e)
            dout = 2 * e / len(t)
            gW2 = H.T @ dout + self.alpha * P[2]
            gb2 = dout.sum()
            dH = np.outer(dout, P[2]) * (1 - H * H)
            gW1 = Z.T @ dH + self.alpha * P[0]
            gb1 = dH.sum(0)
            grads = [gW1, gb1, gW2, gb2]
            for i in range(4):
                m[i] = 0.9 * m[i] + 0.1 * grads[i]
                v[i] = 0.999 * v[i] + 0.001 * grads[i] ** 2
                P[i] = P[i] - self.lr * (m[i] / (1 - 0.9 ** ep)) / (np.sqrt(v[i] / (1 - 0.999 ** ep)) + 1e-8)
            if loss < best - 1e-9:
                best, best_P, wait = loss, [a.copy() if isinstance(a, np.ndarray) else a for a in P], 0
            else:
                wait += 1
                if wait > self.patience:
                    break
        self.P = best_P
        return self

    def predict(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        H = np.tanh(Z @ self.P[0] + self.P[1])
        return (H @ self.P[2] + self.P[3]) * self.ys + self.ym


# ====================================================================== clustering
class KMeans(Model):
    params = ("k", "n_init", "max_iter", "seed")

    def __init__(self, k=3, n_init=5, max_iter=200, seed=0):
        self.k, self.n_init, self.max_iter, self.seed = k, n_init, max_iter, seed

    def fit(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            C = [X[rng.integers(len(X))]]
            for _ in range(1, self.k):  # k-means++
                d = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
                C.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
            C = np.array(C)
            for _ in range(self.max_iter):
                lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
                Cn = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(self.k)])
                if np.allclose(Cn, C):
                    break
                C = Cn
            inertia = float(((X - C[lab]) ** 2).sum())
            if best is None or inertia < best[0]:
                best = (inertia, C, lab)
        self.inertia_, self.centers_, self.labels_ = best
        return self

    def predict(self, X):
        return np.argmin(((np.asarray(X, float)[:, None] - self.centers_[None]) ** 2).sum(-1), 1)


class GaussianMixture(Model):
    params = ("k", "max_iter", "seed")

    def __init__(self, k=3, max_iter=200, seed=0):
        self.k, self.max_iter, self.seed = k, max_iter, seed

    def fit(self, X):
        X = np.asarray(X, float)
        km = KMeans(self.k, 3, 100, self.seed).fit(X)
        mu = km.centers_.copy()
        var = np.array([X[km.labels_ == j].var(0) + 1e-6 if np.sum(km.labels_ == j) > 1 else X.var(0) + 1e-6
                        for j in range(self.k)])
        pi = np.array([np.mean(km.labels_ == j) for j in range(self.k)]) + 1e-9
        ll_old = -np.inf
        for _ in range(self.max_iter):
            lp = np.stack([np.log(pi[j]) - 0.5 * np.sum(np.log(2 * np.pi * var[j]) + (X - mu[j]) ** 2 / var[j], 1)
                           for j in range(self.k)], 1)
            m = lp.max(1, keepdims=True)
            ll = float(np.sum(m[:, 0] + np.log(np.exp(lp - m).sum(1))))
            R = np.exp(lp - m)
            R /= R.sum(1, keepdims=True)
            nk = R.sum(0) + 1e-12
            pi = nk / len(X)
            mu = (R.T @ X) / nk[:, None]
            var = np.stack([(R[:, j:j + 1] * (X - mu[j]) ** 2).sum(0) / nk[j] + 1e-6 for j in range(self.k)])
            if abs(ll - ll_old) < 1e-8 * abs(ll):
                break
            ll_old = ll
        self.weights_, self.means_, self.vars_, self.loglik_ = pi, mu, var, ll
        self.labels_ = np.argmax(R, 1)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        lp = np.stack([np.log(self.weights_[j]) - 0.5 * np.sum(np.log(2 * np.pi * self.vars_[j]) +
                                                              (X - self.means_[j]) ** 2 / self.vars_[j], 1)
                       for j in range(len(self.weights_))], 1)
        return np.argmax(lp, 1)


class SOM(Model):
    params = ("rows", "cols", "iters", "seed")

    def __init__(self, rows=6, cols=6, iters=2000, seed=0):
        self.rows, self.cols, self.iters, self.seed = rows, cols, iters, seed

    def fit(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed)
        self.mu, self.sd = _std(X)
        Z = (X - self.mu) / self.sd
        gy, gx = np.meshgrid(np.arange(self.rows), np.arange(self.cols), indexing="ij")
        self.grid = np.column_stack([gy.ravel(), gx.ravel()]).astype(float)
        W = Z[rng.integers(0, len(Z), len(self.grid))].copy()
        s0 = max(self.rows, self.cols) / 2.0
        for t in range(self.iters):
            frac = t / self.iters
            sig = s0 * (0.05 / s0) ** frac + 1e-3
            lr = 0.5 * (0.01 / 0.5) ** frac
            z = Z[rng.integers(len(Z))]
            bmu = np.argmin(((W - z) ** 2).sum(1))
            hgt = np.exp(-((self.grid - self.grid[bmu]) ** 2).sum(1) / (2 * sig * sig))
            W += lr * hgt[:, None] * (z - W)
        self.W = W
        self.labels_ = self.predict(X)
        return self

    def predict(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        return np.argmin(((Z[:, None] - self.W[None]) ** 2).sum(-1), 1)


def silhouette(X, labels):
    X = np.asarray(X, float)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    s = np.zeros(len(X))
    labs = np.unique(labels)
    for i in range(len(X)):
        same = labels == labels[i]
        a = D[i, same & (np.arange(len(X)) != i)].mean() if same.sum() > 1 else 0.0
        b = min(D[i, labels == c].mean() for c in labs if c != labels[i]) if len(labs) > 1 else 0.0
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def elbow(X, ks=range(1, 9), seed=0):
    return [(k, KMeans(k, 3, 100, seed).fit(X).inertia_) for k in ks]


# ====================================================================== optional backends
def make_model(name, **kw):
    """'linear', 'ridge', 'lasso', 'knn', 'tree', 'rf', 'gbm', 'mlp' (numpy) or 'xgboost' / 'sklearn-rf' when
    those libraries are installed (falls back to the numpy model otherwise)."""
    if name == "xgboost":
        try:
            import xgboost  # noqa: F401
            from xgboost import XGBRegressor
            return _Wrap(XGBRegressor(**kw))
        except Exception:
            return GradientBoosting(**{k: v for k, v in kw.items() if k in GradientBoosting.params})
    if name == "sklearn-rf":
        try:
            from sklearn.ensemble import RandomForestRegressor
            return _Wrap(RandomForestRegressor(**kw))
        except Exception:
            return RandomForest(**{k: v for k, v in kw.items() if k in RandomForest.params})
    return {"linear": lambda: LinearModel(0.0), "ridge": lambda: LinearModel(kw.get("alpha", 0.1)),
            "lasso": lambda: Lasso(**kw), "knn": lambda: KNN(**kw), "tree": lambda: DecisionTree(**kw),
            "rf": lambda: RandomForest(**kw), "gbm": lambda: GradientBoosting(**kw), "mlp": lambda: MLP(**kw)}[name]()


class _Wrap(Model):
    def __init__(self, m):
        self.m = m

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)

    def get_params(self):
        return self.m.get_params()

    def clone(self):  # xgboost / sklearn estimator: a fresh unfitted copy with the same parameters
        return _Wrap(type(self.m)(**self.m.get_params()))
