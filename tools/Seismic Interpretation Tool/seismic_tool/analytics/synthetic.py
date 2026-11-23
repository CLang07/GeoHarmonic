"""Synthetic engineering + production data set with known geology and completion effects (for tests and for
trying the analytics workflow).  Written as CSV files like a vendor export."""
import csv
import os

import numpy as np

from .engineering import arps_rate


def geology_fields(x, y):
    """Smooth 'true' geology at map points (m): porosity (frac) and net thickness (m)."""
    phi = 0.08 + 0.05 * np.sin(x / 7000.0) * np.cos(y / 9000.0) + 0.02 * (x / 20000.0)
    h = 25 + 10 * np.cos(x / 11000.0 + y / 13000.0)
    return phi, h


def make(folder, n=120, seed=7, end_month=(2024 - 1900) * 12):
    """Writes completions.csv (Enverus-like headers, fluid in m3 for a few wells to test units) and
    production.csv (monthly).  Returns truth dict."""
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    x = rng.uniform(0, 20000, n)
    y = rng.uniform(0, 20000, n)
    phi, h = geology_fields(x, y)
    fp = rng.integers((2014 - 1900) * 12, (2022 - 1900) * 12, n)  # first production month
    trend = (fp - fp.min()) / max(1, np.ptp(fp))  # completion design intensifies with time (confound)
    lat = rng.uniform(5000, 12000, n)
    ppf = 800 + 1500 * trend + rng.normal(0, 150, n)
    fpf = 20 + 25 * trend + rng.normal(0, 3, n)
    stages = np.round(lat / rng.uniform(180, 260, n))
    prop = ppf * lat
    fluid = fpf * lat
    # child wells: neighbour within 1500 ft (457 m) producing earlier
    child = np.zeros(n)
    for i in range(n):
        d = np.hypot(x - x[i], y - y[i])
        d[i] = np.inf
        child[i] = float(np.any((d < 457.2) & (fp < fp[i])))
    eur_true = 40.0 * (phi / 0.08) * (h / 25.0) * (lat / 1000.0) ** 0.9 * (ppf / 1000.0) ** 0.3 * \
        np.where(child > 0, 0.85, 1.0) * np.exp(rng.normal(0, 0.08, n))  # kBOE
    b, di = 1.0, 0.15
    # qi so that the 360-month Arps cum matches the EUR
    tt = np.arange(360)
    unit_cum = arps_rate(1.0, di, b, tt).sum()
    qi = eur_true * 1000.0 / unit_cum  # BOE / month
    gor = 1500.0 + 1000.0 * (x / 20000.0)  # scf/bbl
    rows = []
    api = [f"42-{100 + i // 1000:03d}-{i:05d}-00-00" for i in range(n)]
    for i in range(n):
        nm = int(min(120, end_month - fp[i]))
        q = arps_rate(qi[i], di, b, np.arange(nm))
        oil = q / (1 + gor[i] / 6000.0)
        gas = oil * gor[i] / 1000.0  # mcf
        wat = oil * (0.3 + 0.4 * (1 - phi[i] / 0.15))
        for m in range(nm):
            mon = fp[i] + m
            rows.append([api[i], f"{1900 + mon // 12:04d}-{mon % 12 + 1:02d}-01", f"{oil[m]:.1f}", f"{gas[m]:.1f}",
                         f"{wat[m]:.1f}", 30 if m else 16])
    with open(os.path.join(folder, "production.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["API14", "ProdDate", "Oil_BBL", "Gas_MCF", "Water_BBL", "Days"])
        w.writerows(rows)
    metric = np.zeros(n, bool)
    metric[::17] = True  # some wells reported in m3 -> separate file to test unit handling
    for fname, sel, unit in (("completions.csv", ~metric, "bbl"), ("completions_metric.csv", metric, "m3")):
        with open(os.path.join(folder, fname), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["API14", "X", "Y", "LateralLength_FT", "FracStages", "Proppant_LBS",
                        "TotalFluidPumped_BBL" if unit == "bbl" else "TotalFluid_M3", "FirstProdDate"])
            for i in np.nonzero(sel)[0]:
                fl = fluid[i] if unit == "bbl" else fluid[i] / 6.28981
                w.writerow([api[i], f"{x[i]:.1f}", f"{y[i]:.1f}", f"{lat[i]:.0f}", int(stages[i]), f"{prop[i]:.0f}",
                            f"{fl:.1f}", f"{1900 + fp[i] // 12:04d}-{fp[i] % 12 + 1:02d}-15"])
    return {"api": api, "x": x, "y": y, "phi": phi, "h": h, "lat": lat, "ppf": ppf, "fpf": fpf, "child": child,
            "eur_kboe": eur_true, "fp": fp, "di": di, "b": b, "metric": metric}
