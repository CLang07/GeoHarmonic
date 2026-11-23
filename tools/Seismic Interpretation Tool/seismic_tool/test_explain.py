"""Analytics 13c: explainability checks (TreeSHAP exactness, KernelSHAP, importance, PDP, save/load, report)."""
import os

import numpy as np


def run(check, tmp):
    from .analytics import explain as XP
    from .analytics import models as M
    from .analytics.validation import r2
    from .test_models import data
    X, y = data(300, 2)
    X = X[:, :4]
    y = 3 * X[:, 0] + 4 * np.sin(3 * X[:, 1]) + 5 * X[:, 2] * X[:, 3]
    g = M.GradientBoosting(6, 0.3, 3, subsample=0.8, seed=1).fit(X, y)
    phi, ev = XP.tree_shap(g, X[:20])
    raw = g.raw(X[:20])
    check(np.max(np.abs(ev + phi.sum(1) - raw)) < 1e-9, "TreeSHAP local accuracy (sum of SHAP + E[f] = prediction)")
    bf = np.array([XP.brute_force_shap(g, X[i]) for i in range(5)])
    check(np.max(np.abs(bf - phi[:5])) < 1e-9, f"TreeSHAP == brute-force Shapley (max diff {np.max(np.abs(bf - phi[:5])):.1e})")
    rf = M.RandomForest(4, 4, 3, None, 0).fit(X, y)
    pr, er = XP.tree_shap(rf, X[:5])
    bfr = np.array([XP.brute_force_shap(rf, X[i]) for i in range(3)])
    check(np.max(np.abs(er + pr.sum(1) - rf.predict(X[:5]))) < 1e-9 and np.max(np.abs(bfr - pr[:3])) < 1e-9,
          "TreeSHAP exact for the random forest too")
    Xn = X.copy()
    Xn[0, 1] = np.nan
    gn = M.GradientBoosting(5, 0.3, 3).fit(np.where(np.arange(len(X))[:, None] % 9 == 0, np.nan, X), y)
    pn, en = XP.tree_shap(gn, Xn[:3])
    check(np.max(np.abs(en + pn.sum(1) - gn.raw(Xn[:3]))) < 1e-9, "TreeSHAP with missing values")
    # KernelSHAP: exact for a linear model with independent background
    lm = M.LinearModel().fit(X, y)
    ks, ek = XP.kernel_shap(lm.predict, X[3], X)
    exact = lm.coef_ * (X[3] - X.mean(0))
    check(np.max(np.abs(ks - exact)) < 1e-6, "KernelSHAP = exact Shapley for a linear model")
    gb = M.GradientBoosting(120, 0.1, 3).fit(X, y)
    ks2, _ = XP.kernel_shap(gb.predict, X[7], X[::6])
    ts2, _ = XP.tree_shap(gb, X[7:8])
    check(np.corrcoef(ks2, ts2[0])[0, 1] > 0.9, f"KernelSHAP agrees with TreeSHAP on a GBM (corr {np.corrcoef(ks2, ts2[0])[0, 1]:.2f})")
    # permutation importance, PDP, ensemble map
    pm, ps = XP.permutation_importance(gb, X, y, r2, 3)
    check(np.argmax(pm) in (1, 2, 3) and pm.min() >= -0.05, f"permutation importance {np.round(pm, 2)}")
    gpi = XP.grouped_permutation_importance(gb, X, y, r2, {"x2x3": [2, 3], "x0": [0]}, 3)
    check(gpi["x2x3"][0] > gpi["x0"][0], "grouped permutation importance")
    grid, pdv = XP.partial_dependence(lm, X, 0)
    check(abs(np.polyfit(grid, pdv, 1)[0] - lm.coef_[0]) < 1e-6, "partial dependence slope = linear coefficient")
    mean, sd = XP.ensemble_map([gb, M.GradientBoosting(120, 0.1, 3, seed=3).fit(X, y)], X[:10])
    check(mean.shape == (10,) and np.all(sd >= 0), "ensemble prediction + uncertainty")
    # save / load
    p = XP.save_model(gb, os.path.join(tmp, "gbm.json"), ["a", "b", "c", "d"])
    g2 = XP.load_model(p)
    pl = XP.save_model(lm, os.path.join(tmp, "lin.json"))
    check(np.allclose(g2.predict(X), gb.predict(X)) and np.allclose(XP.load_model(pl).predict(X), lm.predict(X)),
          "models saved to JSON and reloaded give identical predictions")
    # report with plots
    names = ["a", "b", "c", "d"]
    phi_all, ev_all = XP.tree_shap(gb, X[:60])
    rep = XP.html_report(os.path.join(tmp, "report.html"), "Test run", [
        ("Scores", XP.table_html([("GBM", 0.9)], ["model", "R2"])),
        ("SHAP summary", ("png", XP.plot_shap_summary(phi_all, X[:60], names))),
        ("Importance", ("png", XP.plot_bar(np.abs(phi_all).mean(0), names, "mean |SHAP|"))),
        ("Dependence", ("png", XP.plot_dependence(phi_all, X[:60], names, 2, 3))),
        ("Waterfall", ("png", XP.plot_waterfall(phi_all[0], ev_all, names, "well 0"))),
        ("PDP", ("png", XP.plot_pdp(gb, X, names, 1)))])
    check(os.path.getsize(rep) > 20000 and open(rep).read().count("data:image/png") == 5, "HTML run report with plots")
    return True


def run_gui(check, tmp):
    from .gui.analytics_window import AnalyticsWindow
    from .maps import horizon_grid
    from .test_velocity import velocity_case
    sp, T = velocity_case(tmp)
    win = AnalyticsWindow(sp)
    g1 = horizon_grid(sp, sp.horizons.get("H1"), "3D:V")
    g2 = horizon_grid(sp, sp.horizons.get("H2"), "3D:V")
    win.add_grid("H1 time", g1)
    win.add_grid("H2 time", g2)
    csv = os.path.join(tmp, "targets.csv")
    with open(csv, "w") as fh:
        fh.write("Well,Porosity\nA-1,0.12\nB-1,0.18\nC-1,0.15\n")
    win.add_csv(csv)
    t = win.table
    check(t.names == ["H1 time", "H2 time", "Porosity"] and np.isfinite(t.cols["Porosity"]).sum() == 3,
          "analytics: table of map values + CSV target at the wells")
    txt = win.explore()
    check("Missing values" in txt and "VIF" in txt, "analytics: explore (missing, correlation, VIF)")
    r = win.train(["H1 time"], "Porosity", "linear", "leave one well out")
    check(r["warning"].startswith("WARNING") and np.isfinite(r["r2"]), "analytics: LOO validation + few-wells warning")
    g = win.predict_map()
    check(g is not None and g.z.shape == g1.z.shape and np.isfinite(g.uncertainty.z).any(),
          "analytics: predicted map + uncertainty map")
    r2 = win.train(["H1 time", "H2 time"], "Porosity", "gbm", "group 5-fold")
    rep = win.report(os.path.join(tmp, "an.html"))
    check("shap" in r2 and os.path.getsize(rep) > 1000, "analytics: GBM + SHAP + HTML report")
    return True
