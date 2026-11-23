"""Analytics 13d: model QC toolkit checks on synthetic wells with known structure."""
import os

import numpy as np


def run(check, tmp):
    from .analytics import models as M
    from .analytics import qc as Q
    from .analytics import validation as V
    rng = np.random.default_rng(5)
    n = 240
    xy = rng.uniform(0, 20000, (n, 2))
    att1 = rng.normal(size=n)
    att2 = rng.normal(size=n)
    spatial = np.sin(xy[:, 0] / 3000.0) + np.cos(xy[:, 1] / 4000.0)
    y = 2 * att1 + np.where(att2 > 0, 1.5, 0) + spatial + 0.3 * rng.normal(size=n)
    X = np.column_stack([att1, att2])
    groups = np.arange(n)
    names = ["att1", "att2"]
    gbm = M.GradientBoosting(120, 0.1, 3)
    # baselines
    b = Q.baselines(X, y, xy, groups, names)
    cvs = V.r2(y, V.cross_val_predict(gbm, X, y, V.group_kfold(groups, 5)))
    check(b["mean"] < 0.05 and cvs > max(b.values()),
          f"model beats the baselines (CV R2 {cvs:.2f}; " + ", ".join(f"{k} {v:.2f}" for k, v in b.items()) + ")")
    yr = Q.y_randomization(gbm, X, y, groups, 3)
    check(yr.max() < 0.1, f"y-randomisation collapses the score (max {yr.max():.2f})")
    imp, k_noise = Q.noise_feature_test(gbm, X, y)
    check(np.all(imp[:k_noise] > imp[k_noise]), f"real inputs out-rank a random noise column ({np.round(imp, 2)})")
    # residual diagnostics: spatial structure left in residuals without x/y, gone with x/y
    pred = V.cross_val_predict(gbm, X, y, V.group_kfold(groups, 5))
    I1, z1 = Q.morans_i(xy, y - pred)
    Xs = np.column_stack([X, xy])
    pred2 = V.cross_val_predict(M.GradientBoosting(200, 0.1, 4), Xs, y, V.group_kfold(groups, 5))
    I2, z2 = Q.morans_i(xy, y - pred2)
    check(z1 > 3 and z2 < z1 / 2, f"Moran's I finds spatial structure left in residuals (z {z1:.1f} -> {z2:.1f} with x/y)")
    h, sv, cnt = Q.variogram(xy, y - pred)
    check(sv[0] < sv[-1] and cnt.sum() > 0, "residual variogram rises with distance (spatial correlation)")
    rd = Q.residual_diagnostics(y, pred2, Xs, names + ["x", "y"])
    check(rd["qq_corr"] > 0.97 and abs(rd["resid_vs_pred_corr"]) < 0.4, "residual QQ / residual vs prediction")
    # uncertainty: conformal + quantile GBM coverage on new wells
    Xt = np.column_stack([rng.normal(size=400), rng.normal(size=400)])
    xyt = rng.uniform(0, 20000, (400, 2))
    yt = 2 * Xt[:, 0] + np.where(Xt[:, 1] > 0, 1.5, 0) + np.sin(xyt[:, 0] / 3000.0) + np.cos(xyt[:, 1] / 4000.0) + \
        0.3 * rng.normal(size=400)
    lo, hi, q = Q.conformal_intervals(gbm, X, y, Xt, 0.2, groups)
    cov = Q.coverage(yt, lo, hi)
    check(0.72 <= cov <= 0.9, f"split-conformal 80% interval covers {cov:.0%} of new wells")
    ql, qh = Q.quantile_intervals(X, y, Xt, 0.1, 0.9)
    covq = Q.coverage(yt, ql, qh)
    check(0.65 <= covq <= 0.92, f"quantile GBM 10-90% interval covers {covq:.0%}")
    yc = (y > np.median(y)).astype(float)
    gc = M.GradientBoosting(100, 0.1, 3, objective="logistic").fit(X, yc)
    (mp, ob, cnt), brier = Q.calibration(yc, gc.predict_proba(X))
    check(brier < 0.2 and np.all(np.abs(mp - ob)[cnt > 30] < 0.35), f"classifier calibration (Brier {brier:.3f})")
    # applicability domain
    Xnew = np.vstack([X[:20], X[:20] * 6.0])
    inside, md, kd = Q.applicability(X, Xnew)
    check(inside[:20].mean() > 0.9 and inside[20:].mean() < 0.3, "applicability domain flags extrapolated inputs")
    # stability, sensitivity, sanity, comparison, temporal / blind designs
    rho, R = Q.importance_stability(gbm, X, y, groups, 4)
    check(rho > 0.5, f"importance ranking stable across bootstraps (rho {rho:.2f})")
    hs = Q.hyperparameter_sensitivity(gbm, "max_depth", [1, 3], X, y, groups)
    check(len(hs) == 2, "hyperparameter sensitivity")
    pdc = Q.pdp_direction_checks(gbm.clone().fit(X, y), X, {"att1": +1}, names)
    check(pdc["att1"][0], f"partial dependence direction matches physics (slope {pdc['att1'][1]:.2f})")
    splits = list(V.group_kfold(groups, 8))
    pa = V.cross_val_predict(gbm, X, y, splits)
    pb = V.cross_val_predict(M.LinearModel(), X[:, :1], y, splits)  # simpler model: att1 only
    cmpd = Q.paired_comparison(y, pa, pb, splits)
    check(cmpd["mean_diff"] < 0 and cmpd["a_better_folds"] >= 6, f"paired fold comparison GBM vs linear(att1) "
                                                                 f"(p {cmpd['p']:.3f}, {cmpd['a_better_folds']}/8 folds)")
    tr, te = Q.temporal_split(np.arange(n), 0.7)
    trb, blind = Q.blind_split(groups, frac=0.1)
    rep = Q.repeated_cv_scores(gbm, X, y, groups, 5, 3)
    check(tr.sum() == 168 and blind.sum() == 24 and rep.std() < 0.1, f"temporal / blind designs; repeated CV spread {rep.std():.3f}")
    out = Q.qc_report(os.path.join(tmp, "qc.html"), "QC", [("Baselines", str(b)), ("Moran", f"I={I1:.2f}")])
    check(os.path.exists(out), "QC report written")
    return True


def run_gui(check, tmp):
    """Analytics window: Model QC on a trained model (60 synthetic wells), and the few-wells fallback."""
    from .gui.analytics_window import AnalyticsWindow
    from .analytics.table import AttributeTable

    class _P:
        well_project = None

        def base(self):
            return tmp

    rng = np.random.default_rng(3)
    n = 60
    xy = rng.uniform(0, 20000, (n, 2))
    a1, a2 = rng.normal(size=n), rng.normal(size=n)
    t = AttributeTable([f"W-{i}" for i in range(n)])
    t.add("amp", a1).add("isochron", a2).add("net pay", 2 * a1 + np.where(a2 > 0, 1.0, 0) + 0.2 * rng.normal(size=n))
    t.xy = xy
    win = AnalyticsWindow(_P())
    win.table = t
    r = win.train(["amp", "isochron"], "net pay", "gbm", "group 5-fold")
    lines, path = win.qc(os.path.join(tmp, "qc_gui.html"))
    txt = "\n".join(lines)
    check(r["r2"] > 0.7 and "beats all" in txt and "(OK)" in txt and "n/a" not in txt and os.path.exists(path),
          "analytics window: Model QC report (baselines, y-randomisation, noise, Moran, conformal, stability)")
    win.table = AttributeTable(["A", "B"]).add("amp", [1.0, 2.0]).add("net pay", [3.0, 4.0])
    win.train(["amp"], "net pay", "linear", "leave one well out")
    lines, path = win.qc()
    check(path is None and "Too few wells" in lines[-1], "Model QC with too few wells explains instead of failing")
    return True
