"""Line geometry and intersections."""
import csv
import numpy as np


def read_nav_csv(path):
    """CSV with columns line,cdp,x,y (header row required). Returns {line: array[n,3]}."""
    nav = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            nav.setdefault(row["line"].strip(), []).append((float(row["cdp"]), float(row["x"]), float(row["y"])))
    return {k: np.array(sorted(v)) for k, v in nav.items()}


def polyline_from_traces(cdp, x, y, step=25):
    idx = np.unique(np.r_[np.arange(0, len(cdp), step), len(cdp) - 1])
    return np.column_stack([cdp[idx], x[idx], y[idx]])


def intersections(pa, pb):
    """All crossings of two polylines [cdp, x, y]. Returns list of (cdp_a, cdp_b, x, y)."""
    out = []
    for i in range(len(pa) - 1):
        p, r = pa[i, 1:3], pa[i + 1, 1:3] - pa[i, 1:3]
        for j in range(len(pb) - 1):
            q, s = pb[j, 1:3], pb[j + 1, 1:3] - pb[j, 1:3]
            den = r[0] * s[1] - r[1] * s[0]
            if den == 0:
                continue
            qp = q - p
            t = (qp[0] * s[1] - qp[1] * s[0]) / den
            u = (qp[0] * r[1] - qp[1] * r[0]) / den
            if 0 <= t <= 1 and 0 <= u <= 1:
                ca = pa[i, 0] + t * (pa[i + 1, 0] - pa[i, 0])
                cb = pb[j, 0] + u * (pb[j + 1, 0] - pb[j, 0])
                xy = p + t * r
                out.append((float(ca), float(cb), float(xy[0]), float(xy[1])))
    # merge near-duplicates (a crossing exactly on a vertex is found twice)
    merged = []
    for c in out:
        if not any(abs(c[0] - m[0]) < 2 and abs(c[1] - m[1]) < 2 for m in merged):
            merged.append(c)
    return merged
