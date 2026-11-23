"""--check-ml: compare the numpy models with xgboost / scikit-learn / shap when they are installed."""
import numpy as np


def run():
    from . import models as M
    from . import validation as V
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (800, 5))
    y = 3 * X[:, 0] + 4 * np.sin(6 * X[:, 1]) + 5 * X[:, 2] * X[:, 3] + 0.2 * rng.normal(size=800)
    tr, te = slice(0, 600), slice(600, None)
    g = M.GradientBoosting(300, 0.05, 4).fit(X[tr], y[tr])
    print(f"numpy GBM            R2 {V.r2(y[te], g.predict(X[te])):.3f}")
    rf = M.RandomForest(100, 10).fit(X[tr], y[tr])
    print(f"numpy random forest  R2 {V.r2(y[te], rf.predict(X[te])):.3f}")
    try:
        from xgboost import XGBRegressor
        xg = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4).fit(X[tr], y[tr])
        print(f"xgboost              R2 {V.r2(y[te], xg.predict(X[te])):.3f}")
    except Exception as e:
        print("xgboost not available:", e)
    try:
        from sklearn.ensemble import RandomForestRegressor
        sk = RandomForestRegressor(100, max_depth=10, random_state=0).fit(X[tr], y[tr])
        print(f"sklearn random forest R2 {V.r2(y[te], sk.predict(X[te])):.3f}")
    except Exception as e:
        print("scikit-learn not available:", e)
    try:
        import shap
        from .explain import tree_shap
        phi, ev = tree_shap(g, X[te][:20])
        print(f"numpy TreeSHAP local accuracy error {np.max(np.abs(ev + phi.sum(1) - g.raw(X[te][:20]))):.1e}")
        print("shap", shap.__version__, "installed (numpy TreeSHAP is used by the tool; values are directly comparable "
              "for xgboost models via shap.TreeExplainer)")
    except Exception as e:
        print("shap not available:", e)
