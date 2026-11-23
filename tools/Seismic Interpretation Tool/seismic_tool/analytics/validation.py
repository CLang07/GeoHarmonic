"""Honest validation for well data: metrics, k-fold / group k-fold (by well) / spatial block CV /
leave-one-group-out, cross-validated predictions, nested CV grid search, learning curves."""
import itertools

import numpy as np


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else 0.0


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def accuracy(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def confusion(y, p, labels=None):
    labels = np.unique(np.r_[y, p]) if labels is None else np.asarray(labels)
    M = np.zeros((len(labels), len(labels)), int)
    idx = {l: i for i, l in enumerate(labels)}
    for a, b in zip(y, p):
        M[idx[a], idx[b]] += 1
    return M, labels


def f1(y, p, positive=1):
    y, p = np.asarray(y), np.asarray(p)
    tp = np.sum((y == positive) & (p == positive))
    fp = np.sum((y != positive) & (p == positive))
    fn = np.sum((y == positive) & (p != positive))
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


# ---------------------------------------------------------------------- splitters (yield train, test idx)
def kfold(n, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    for f in range(k):
        te = idx[f::k]
        yield np.setdiff1d(idx, te), np.sort(te)


def group_kfold(groups, k=5, seed=0):
    g = np.asarray(groups)
    ug = np.random.default_rng(seed).permutation(np.unique(g))
    for f in range(k):
        te_g = ug[f::k]
        te = np.nonzero(np.isin(g, te_g))[0]
        if len(te):
            yield np.nonzero(~np.isin(g, te_g))[0], te


def leave_one_group_out(groups):
    g = np.asarray(groups)
    for u in np.unique(g):
        yield np.nonzero(g != u)[0], np.nonzero(g == u)[0]


def spatial_blocks(xy, block, k=5, seed=0, buffer=0.0):
    """Blocked spatial CV: square blocks of size `block` assigned to k folds; training rows within `buffer`
    of any test row are dropped (spatial-buffer CV)."""
    xy = np.asarray(xy, float)
    bid = np.floor(xy[:, 0] / block).astype(int) * 100003 + np.floor(xy[:, 1] / block).astype(int)
    for tr, te in group_kfold(bid, k, seed):
        if buffer > 0 and len(te):
            d = np.min(np.hypot(xy[tr, None, 0] - xy[None, te, 0], xy[tr, None, 1] - xy[None, te, 1]), 1)
            tr = tr[d > buffer]
        yield tr, te


def cross_val_predict(model, X, y, splits):
    X, y = np.asarray(X, float), np.asarray(y, float)
    pred = np.full(len(y), np.nan)
    for tr, te in splits:
        m = model.clone().fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def cross_val_score(model, X, y, splits, metric=r2):
    X, y = np.asarray(X, float), np.asarray(y, float)
    out = []
    for tr, te in splits:
        m = model.clone().fit(X[tr], y[tr])
        out.append(metric(y[te], m.predict(X[te])))
    return np.array(out)


def grid_search(model, grid, X, y, split_fn, metric=r2):
    """Best params by mean CV score.  split_fn() -> fresh iterator of (train, test)."""
    keys = list(grid)
    best = (-np.inf, None)
    for vals in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, vals))
        s = cross_val_score(model.clone().set_params(**p), X, y, split_fn(), metric).mean()
        if s > best[0]:
            best = (s, p)
    return best[1], best[0]


def nested_cv(model, grid, X, y, outer, inner_fn, metric=r2):
    """Nested CV: hyperparameters chosen inside each outer training fold.  inner_fn(train_idx) -> splits.
    Returns (outer scores, chosen params per fold)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    scores, chosen = [], []
    for tr, te in outer:
        Xi, yi = X[tr], y[tr]
        p, _ = grid_search(model, grid, Xi, yi, lambda tr=tr: inner_fn(tr), metric)
        m = model.clone().set_params(**p).fit(Xi, yi)
        scores.append(metric(y[te], m.predict(X[te])))
        chosen.append(p)
    return np.array(scores), chosen


def learning_curve(model, X, y, split_fn, fractions=(0.2, 0.4, 0.6, 0.8, 1.0), metric=r2, seed=0):
    """Mean CV score vs fraction of the training rows used (would more wells help?)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    out = []
    for f in fractions:
        s = []
        for tr, te in split_fn():
            k = max(3, int(round(f * len(tr))))
            sub = rng.choice(tr, k, replace=False)
            m = model.clone().fit(X[sub], y[sub])
            s.append(metric(y[te], m.predict(X[te])))
        out.append((f, float(np.mean(s))))
    return out
