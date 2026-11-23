"""Analytics 13b: models + validation checks."""
import numpy as np


def data(n=500, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 6))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 4 * np.sin(3 * np.pi * X[:, 2]) + 6 * X[:, 3] * X[:, 4] + noise * rng.normal(size=n)
    return X, y


def run(check, tmp):
    from .analytics import models as M
    from .analytics import validation as V
    rng = np.random.default_rng(1)
    # linear / ridge / lasso
    X = rng.normal(size=(300, 5))
    y = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.5 + 0.01 * rng.normal(size=300)
    lm = M.LinearModel().fit(X, y)
    check(np.allclose(lm.coef_, [1.5, -2, 0, 0, 0], atol=0.01) and abs(lm.intercept_ - 0.5) < 0.01, "OLS coefficients")
    rd = M.LinearModel(alpha=1.0).fit(X, y)
    check(np.all(np.abs(rd.coef_[:2]) < np.abs(lm.coef_[:2])), "ridge shrinks")
    ls = M.Lasso(alpha=0.05).fit(X, y)
    check(np.all(ls.coef_[2:] == 0) and abs(ls.coef_[0]) > 1.3, f"lasso zeroes irrelevant inputs {np.round(ls.coef_, 2)}")
    # nonlinear benchmark
    Xa, ya = data(600, 0)
    Xt, yt = data(400, 1)
    scores = {}
    for name, m in (("linear", M.LinearModel()), ("knn", M.KNN(7)), ("tree", M.DecisionTree(6)),
                    ("rf", M.RandomForest(40, 10, 2, "third", 0)), ("gbm", M.GradientBoosting(250, 0.08, 4)),
                    ("mlp", M.MLP(24, 0.01, 1500))):
        scores[name] = V.r2(yt, m.fit(Xa, ya).predict(Xt))
    check(scores["gbm"] > scores["linear"] + 0.2 and scores["gbm"] > 0.9,
          "GBM beats linear on nonlinear + interaction data: " + ", ".join(f"{k} {v:.2f}" for k, v in scores.items()))
    check(scores["rf"] > 0.75 and scores["mlp"] > 0.8 and scores["knn"] > 0.5 and scores["tree"] > 0.6,
          "RF, MLP, kNN, tree all learn the function")
    # early stopping, missing values, monotone constraints, classification
    g = M.GradientBoosting(800, 0.1, 3, early_stopping=20).fit(Xa[:400], ya[:400], Xa[400:], ya[400:])
    check(len(g.trees_) < 800 and len(g.val_curve_) > len(g.trees_), f"early stopping at {len(g.trees_)} trees")
    Xn = Xa.copy()
    Xn[::7, 0] = np.nan
    gn = M.GradientBoosting(200, 0.08, 4).fit(Xn, ya)
    check(V.r2(yt, gn.predict(Xt)) > 0.85, "GBM with missing values (learned default direction)")
    gm = M.GradientBoosting(150, 0.1, 4, monotone={0: 1}).fit(Xa, ya)
    grid = np.tile(Xt[:50].mean(0), (40, 1))
    grid[:, 0] = np.linspace(0, 1, 40)
    pm = gm.predict(grid)
    check(np.all(np.diff(pm) >= -1e-9), "monotone constraint: prediction never decreases with input 0")
    yc = (np.sin(3 * Xa[:, 0]) + Xa[:, 1] ** 2 > 0.9).astype(float)
    yct = (np.sin(3 * Xt[:, 0]) + Xt[:, 1] ** 2 > 0.9).astype(float)
    gc = M.GradientBoosting(120, 0.15, 3, objective="logistic").fit(Xa, yc)
    acc = V.accuracy(yct, gc.predict(Xt))
    check(acc > 0.9 and 0 < V.f1(yct, gc.predict(Xt)) <= 1, f"GBM classification accuracy {acc:.2f}")
    imp = g.importance()
    check(np.argmin(imp) == 5 and abs(imp.sum() - 1) < 1e-9, "gain importance ranks the noise input last")
    # honest validation on spatially correlated wells
    n = 300
    xy = np.column_stack([rng.normal(0, 1, n) + np.repeat(rng.uniform(0, 30, 30), 10),
                          rng.normal(0, 1, n) + np.repeat(rng.uniform(0, 30, 30), 10)])
    field = np.sin(xy[:, 0] / 3) + np.cos(xy[:, 1] / 4)
    Xs = np.column_stack([xy, rng.normal(size=(n, 2))])
    ys = field + np.repeat(rng.normal(0, 0.8, 30), 10) + 0.05 * rng.normal(size=n)  # cluster effect
    groups = np.repeat(np.arange(30), 10)
    model = M.RandomForest(25, 10, 1, None, 0)
    r_rand = V.r2(ys, V.cross_val_predict(model, Xs, ys, V.kfold(n, 5)))
    r_group = V.r2(ys, V.cross_val_predict(model, Xs, ys, V.group_kfold(groups, 5)))
    r_block = V.r2(ys, V.cross_val_predict(model, Xs, ys, V.spatial_blocks(xy, 8.0, 5)))
    check(r_rand > r_group + 0.2 and r_rand > r_block + 0.2,
          f"group / spatial CV more honest than random CV (R2 random {r_rand:.2f}, group {r_group:.2f}, block {r_block:.2f})")
    lo = list(V.leave_one_group_out(groups[:30]))
    check(len(lo) == 3 and all(len(te) == 10 for _, te in lo), "leave-one-well(group)-out splits")
    sc, chosen = V.nested_cv(M.KNN(), {"k": [1, 5, 15]}, Xa[:200], ya[:200], V.kfold(200, 3),
                             lambda tr: V.kfold(len(tr), 3, 1))
    check(len(sc) == 3 and all(c["k"] in (1, 5, 15) for c in chosen), f"nested CV (inner choices {[c['k'] for c in chosen]})")
    lc = V.learning_curve(M.GradientBoosting(60, 0.15, 3), Xa, ya, lambda: V.kfold(len(ya), 3), (0.1, 0.5, 1.0))
    check(lc[0][1] < lc[-1][1], f"learning curve rises ({lc[0][1]:.2f} -> {lc[-1][1]:.2f})")
    M_, lab = V.confusion([0, 1, 1, 0], [0, 1, 0, 0])
    check(M_.tolist() == [[2, 0], [1, 1]] and abs(V.rmse([1, 2], [1, 4]) - np.sqrt(2)) < 1e-12 and V.mae([1, 2], [2, 2]) == 0.5,
          "metrics (confusion, RMSE, MAE)")
    # clustering
    cen = np.array([[0, 0], [6, 0], [0, 6]])
    Xb = np.vstack([c + rng.normal(0, 0.7, (80, 2)) for c in cen])
    truth = np.repeat([0, 1, 2], 80)
    km = M.KMeans(3).fit(Xb)
    purity = sum(np.max(np.bincount(truth[km.labels_ == j], minlength=3)) for j in range(3)) / len(truth)
    gmm = M.GaussianMixture(3).fit(Xb)
    pur_g = sum(np.max(np.bincount(truth[gmm.labels_ == j], minlength=3)) for j in range(3)) / len(truth)
    check(purity > 0.97 and pur_g > 0.97, f"k-means / GMM recover 3 clusters (purity {purity:.2f} / {pur_g:.2f})")
    check(M.silhouette(Xb, km.labels_) > M.silhouette(Xb, M.KMeans(6).fit(Xb).labels_), "silhouette prefers k = 3")
    el = M.elbow(Xb, range(1, 6))
    check(all(a[1] >= b[1] for a, b in zip(el, el[1:])), "elbow curve decreases")
    som = M.SOM(4, 4, 3000).fit(Xb)
    check(len(np.unique(som.labels_)) >= 3, f"SOM maps the data onto {len(np.unique(som.labels_))} nodes")
    mm = M.make_model("xgboost", n_estimators=50)
    check(hasattr(mm.fit(Xa, ya), "predict"), "optional backend falls back to numpy when the library is absent")
    return True
