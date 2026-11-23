"""Analytics checks (13a: table + preparation)."""
import os

import numpy as np


def run(check, tmp):
    from .analytics.table import AttributeTable, normalise_id
    from .analytics import prep
    rng = np.random.default_rng(0)
    n = 200
    a = rng.normal(size=n)
    b = 0.9 * a + 0.1 * rng.normal(size=n)  # collinear with a
    c = rng.normal(size=n)
    t = AttributeTable([f"W-{i}" for i in range(n)])
    t.add("a", a).add("b", b).add("c", c)
    X = t.matrix()
    v = prep.vif(X)
    check(v[0] > 10 and v[1] > 10 and v[2] < 1.2, f"VIF flags collinear inputs ({v.round(1).tolist()})")
    cm = prep.correlation(X)
    check(abs(cm[0, 1] - np.corrcoef(a, b)[0, 1]) < 1e-12 and abs(cm[0, 2]) < 0.2, "correlation matrix")
    sp = prep.correlation(np.column_stack([a, np.exp(a)]), "spearman")
    check(abs(sp[0, 1] - 1.0) < 1e-12, "Spearman rank correlation of a monotonic transform = 1")
    p = prep.pca(X)
    check(p["explained"][0] > 0.6 and abs(p["explained"].sum() - 1) < 1e-9, f"PCA (PC1 explains {p['explained'][0]:.0%})")
    Xm = X.copy()
    Xm[::10, 2] = np.nan
    Xm[5, 0] = np.nan
    rep = prep.missing_report(Xm, ["a", "b", "c"])
    check(rep[2][1] == 20 and rep[0][1] == 1, "missing-value report")
    Xk = prep.impute(Xm, "knn", 5)
    check(abs(Xk[5, 0] - X[5, 0]) < 0.3 and not np.isnan(Xk).any(), "kNN imputation uses the correlated column")
    Xmed = prep.impute(Xm, "median")
    check(not np.isnan(Xmed).any() and abs(Xmed[0, 2] - np.nanmedian(Xm[:, 2])) < 1e-12, "median imputation")
    Xo = X.copy()
    Xo[7, 2] = 25.0
    fl = prep.outlier_flags(Xo)
    check(fl[7, 2] and fl.sum() <= 3, "robust outlier flag")
    aa, bb, r = prep.regression_line(a, 2 * a + 1)
    check(abs(aa - 1) < 1e-9 and abs(bb - 2) < 1e-9 and abs(r - 1) < 1e-9, "crossplot regression line")
    # joins by API / name and CSV round trip
    check(normalise_id("42-123-45678-00-00") == normalise_id("4212345678") and normalise_id("Maui-5") == "maui5",
          "well id normalisation (API-14 -> API-10, names)")
    u = AttributeTable(["4212345678", "Maui-5"]).add("eur", [100.0, 50.0])
    w = AttributeTable(["42-123-45678-0000", "MAUI 5", "Kiwi-1"]).add("amp", [1.0, 2.0, 3.0])
    j = w.join(u)
    check(np.allclose(j.cols["eur"][:2], [100, 50]) and np.isnan(j.cols["eur"][2]) and len(w.join(u, "inner")) == 2,
          "join by API / well name (left and inner)")
    path = t.to_csv(os.path.join(tmp, "t.csv"))
    t2 = AttributeTable.from_csv(path)
    check(t2.names == ["a", "b", "c"] and np.allclose(t2.cols["c"], c, atol=1e-5), "attribute table CSV round trip")
    # from maps at wells / map nodes
    from .maps import MapGrid
    g = MapGrid(np.add.outer(np.arange(10.0), np.arange(12.0)), "xy", "ms", "g", None, 0.0, 0.0, 10.0, 10.0)
    tw = AttributeTable.from_maps_at_wells([("A", 20.0, 30.0), ("B", 55.0, 45.0)], {"g": g})
    check(np.allclose(tw.cols["g"], [3 + 2, 4.5 + 5.5]), "map values sampled at wells")
    tn = AttributeTable.from_map_nodes({"g": g})
    check(len(tn) == 120 and tn.meta["grid_shape"] == (10, 12), "table of map nodes for prediction")
    # zone statistics from the Well Correlation Tool joined as well properties
    from .test_welltie import well_project
    from .welltie import wc
    d = os.path.join(tmp, "an_wells")
    os.makedirs(d)
    wpath, xy = well_project(d)
    wp = wc().project.Project.open(wpath)
    zt = AttributeTable.from_zone_stats(wp, "Sand A")
    check(zt.ids == ["W-1"] and "NtG" in zt.cols and abs(zt.cols["Gross_MD_m"][0] - 600.0) < 1e-6 and
          zt.xy.shape == (1, 2), f"zone stats as well properties ({len(zt.names)} columns; W-2 has no base top)")
    return True
