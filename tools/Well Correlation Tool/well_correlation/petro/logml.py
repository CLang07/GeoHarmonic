"""Machine learning on logs (PLAN 22a): predict missing curves (DT, RHOB, Vs ...) with depth-blocked, well-grouped
cross-validation and P10 / P90 uncertainty; electrofacies by k-means or a Gaussian mixture.

Models come from the Seismic tool's numpy analytics package (GBM / random forest / k-means / GMM, item 13), loaded from
the sibling tool folder.  Depth-blocked CV: samples are grouped by well + depth block (default 30 m) and whole groups are
held out, with a buffer that drops training samples within `buffer_m` of a test block - random K-fold on logs leaks
(neighbouring samples are nearly identical) and looks much better than it is.

    r = predict_curve(project, "DT", ["GR", "RHOB", "NPHI", "RDEEP"])   # CV report + predictions for wells missing DT
    f = electrofacies(project, ["GR", "RHOB", "NPHI", "RDEEP"], k=4)      # FACIES curve per well + facies table
"""
import os
import sys

import numpy as np

from .store import petro_dir, save_curves

LOG_INPUTS = ("RDEEP", "RSHAL", "PERM")


def analytics():
    """seismic_tool.analytics (models, validation) from the sibling 'Seismic Interpretation Tool' folder."""
    here = os.path.dirname(os.path.abspath(__file__))
    st = os.path.normpath(os.path.join(here, "..", "..", "..", "Seismic Interpretation Tool"))
    if st not in sys.path:
        sys.path.insert(0, st)
    from seismic_tool.analytics import models, validation
    return models, validation


def _features(w, names):
    cols = []
    for n in names:
        v = np.asarray(w.logs.get(n, np.full(len(w.depth), np.nan)), float)
        if n in LOG_INPUTS:
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.log10(np.where(v > 0, v, np.nan))
        cols.append(v)
    return np.column_stack(cols) if cols else np.zeros((len(w.depth), 0))


def blocked_splits(depth, wells, k=5, block_m=30.0, buffer_m=10.0, seed=0):
    """Folds of whole (well, depth-block) groups; training samples of the same well within buffer_m of any test
    sample are dropped.  Yields (train idx, test idx)."""
    _, validation = analytics()
    depth = np.asarray(depth, float)
    wells = np.asarray(wells)
    _, wid = np.unique(wells, return_inverse=True)
    groups = wid * 1000003 + np.floor(depth / block_m).astype(int)
    for tr, te in validation.group_kfold(groups, k, seed):
        if buffer_m > 0:
            keep = np.ones(len(tr), bool)
            d = depth[tr]
            for w in np.unique(wid[te]):
                tw = te[wid[te] == w]
                trw = wid[tr] == w
                if not trw.any():
                    continue
                for b in np.unique(np.floor(depth[tw] / block_m)):  # drop same-well training samples near a test block
                    keep &= ~(trw & (d >= b * block_m - buffer_m) & (d < (b + 1) * block_m + buffer_m))
            tr = tr[keep]
        yield tr, te


def gather(project, target, inputs, wells=None, depth_range=None):
    """Training table: rows with the target and every input finite.  Returns (X, y, depth, well names)."""
    Xs, ys, ds, ws = [], [], [], []
    for n in wells or list(project.wells):
        w = project.load_well(n)
        if w.depth is None or not len(w.depth) or target not in w.logs:
            continue
        X = _features(w, inputs)
        y = np.asarray(w.logs[target], float)
        if target in LOG_INPUTS:
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.log10(np.where(y > 0, y, np.nan))
        ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
        if depth_range:
            ok &= (w.depth >= depth_range[0]) & (w.depth <= depth_range[1])
        if ok.sum() < 20:
            continue
        Xs.append(X[ok])
        ys.append(y[ok])
        ds.append(w.depth[ok])
        ws += [n] * int(ok.sum())
    if not Xs:
        return np.zeros((0, len(inputs))), np.zeros(0), np.zeros(0), []
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ds), ws


def make(method="gbm", seed=0, quantile=None):
    models, _ = analytics()
    if method == "rf":
        return models.RandomForest(n_estimators=60, max_depth=10, seed=seed)
    if quantile is not None:
        return models.GradientBoosting(n_estimators=200, learning_rate=0.08, max_depth=4, objective="quantile",
                                       quantile=quantile, seed=seed)
    return models.GradientBoosting(n_estimators=250, learning_rate=0.08, max_depth=4, seed=seed)


def predict_curve(project, target, inputs, train_wells=None, apply_wells=None, method="gbm", block_m=30.0,
                  buffer_m=10.0, k=5, max_rows=40000, save=True, seed=0):
    """Train on the wells that have `target`, cross-validate (depth-blocked) and predict it (P50 + P10 / P90 from
    quantile GBMs) in the wells / intervals where it is missing.  Returns a report dict."""
    _, validation = analytics()
    X, y, d, ws = gather(project, target, inputs, train_wells)
    if len(y) < 100:
        raise ValueError(f"only {len(y)} training samples with {target} and {', '.join(inputs)}")
    rng = np.random.default_rng(seed)
    if len(y) > max_rows:  # thin uniformly for speed (keeps every well)
        sel = np.sort(rng.choice(len(y), max_rows, replace=False))
        X, y, d, ws = X[sel], y[sel], d[sel], [ws[i] for i in sel]
    model = make(method, seed)
    pred_cv = validation.cross_val_predict(model, X, y, blocked_splits(d, ws, k, block_m, buffer_m, seed))
    ok = np.isfinite(pred_cv)
    rep = {"target": target, "inputs": list(inputs), "n_train": int(len(y)), "wells": sorted(set(ws)), "method": method,
           "cv_r2": float(validation.r2(y[ok], pred_cv[ok])), "cv_rmse": float(validation.rmse(y[ok], pred_cv[ok])),
           "block_m": block_m, "buffer_m": buffer_m}
    final = model.clone().fit(X, y)
    q10 = make(method, seed, 0.1).fit(X, y) if method == "gbm" else None
    q90 = make(method, seed, 0.9).fit(X, y) if method == "gbm" else None
    resid_sd = float(np.std(y[ok] - pred_cv[ok]))
    rep["predicted"] = {}
    for n in apply_wells or list(project.wells):
        w = project.load_well(n)
        if w.depth is None or not len(w.depth):
            continue
        Xa = _features(w, inputs)
        have = np.all(np.isfinite(Xa), 1)
        cur = np.asarray(w.logs.get(target, np.full(len(w.depth), np.nan)), float)
        need = have & ~np.isfinite(cur)
        if not need.any():
            continue
        p = np.full(len(w.depth), np.nan)
        lo = np.full(len(w.depth), np.nan)
        hi = np.full(len(w.depth), np.nan)
        p[need] = final.predict(Xa[need])
        if q10 is not None:
            lo[need] = np.minimum(q10.predict(Xa[need]), p[need])
            hi[need] = np.maximum(q90.predict(Xa[need]), p[need])
        else:
            lo[need], hi[need] = p[need] - 1.2816 * resid_sd, p[need] + 1.2816 * resid_sd
        if target in LOG_INPUTS:
            p, lo, hi = 10 ** p, 10 ** lo, 10 ** hi
        rep["predicted"][n] = int(need.sum())
        if save and petro_dir(project):
            save_curves(project, n, w.depth, {f"{target}_PRED": p.astype(np.float32), f"{target}_P10": lo.astype(np.float32),
                                              f"{target}_P90": hi.astype(np.float32)})
    return rep


def electrofacies(project, curves, k=4, method="kmeans", wells=None, save=True, seed=0):
    """Cluster samples of all wells on standardised curves (log10 for resistivity).  Saves FACIES (0..k-1, NaN where a
    curve is missing) per well and returns {"centers" (in curve units), "counts", "labels" {well: array}, "curves"}."""
    models, _ = analytics()
    rows, per = [], []
    names = wells or list(project.wells)
    for n in names:
        w = project.load_well(n)
        if w.depth is None or not len(w.depth):
            continue
        X = _features(w, curves)
        ok = np.all(np.isfinite(X), 1)
        if ok.sum():
            rows.append(X[ok])
            per.append((n, w, X, ok))
    if not rows:
        raise ValueError("no samples with all the curves")
    A = np.concatenate(rows)
    mu, sd = A.mean(0), A.std(0) + 1e-12
    Z = (A - mu) / sd
    sub = Z if len(Z) <= 30000 else Z[np.random.default_rng(seed).choice(len(Z), 30000, replace=False)]
    m = (models.GaussianMixture(k, seed=seed) if method == "gmm" else models.KMeans(k, seed=seed)).fit(sub)
    centers = (m.means_ if method == "gmm" else m.centers_) * sd + mu
    order = np.argsort(centers[:, 0])  # facies 0 = lowest first curve (e.g. cleanest GR)
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    out = {"centers": centers[order], "curves": list(curves), "labels": {}, "counts": np.zeros(k, int), "method": method}
    for n, w, X, ok in per:
        lab = np.full(len(w.depth), np.nan)
        lab[ok] = remap[m.predict((X[ok] - mu) / sd)]
        out["labels"][n] = lab
        out["counts"] += np.bincount(lab[ok].astype(int), minlength=k)
        if save and petro_dir(project):
            save_curves(project, n, w.depth, {"FACIES": lab.astype(np.float32)})
    return out


def facies_proportions(project, fac, zone_rows):
    """Fraction of each facies per well and zone (zone rows from interp.interpret_well) -> [{Well, Zone, F0..}]."""
    out = []
    for r in zone_rows:
        lab = fac["labels"].get(r["Well"])
        if lab is None:
            continue
        w = project.load_well(r["Well"])
        m = (w.depth >= r["Top_MD"]) & (w.depth < r["Base_MD"]) & np.isfinite(lab)
        if not m.any():
            continue
        row = {"Well": r["Well"], "Zone": r["Zone"]}
        k = len(fac["centers"])
        cnt = np.bincount(lab[m].astype(int), minlength=k)
        row.update({f"Facies{j}_frac": round(float(cnt[j] / cnt.sum()), 3) for j in range(k)})
        out.append(row)
    return out
