"""Cross-plot data for the Petrophysics window (PLAN 20e) - pure numpy, so it is testable without Qt and also renders
to PNG with matplotlib.

    nd(...)        neutron-density with sandstone / limestone / dolomite matrix lines (limestone-unit neutron)
    mn(...)        M-N lithology plot with mineral points
    pickett(...)   log Rt vs log PHIE with Sw = 1 / 0.5 / 0.25 lines for the zone's a, m, n, Rw
    buckles(...)   PHIE vs Sw with constant-BVW hyperbolae
    histogram(...) any curve, with discriminators {curve: (lo, hi)}
"""
import numpy as np

MAX_POINTS = 20000

# (NPHI limestone units, RHOB) of the matrices and fresh water; matrix lines join them (approximate, chart-free)
ND_LINES = {"sandstone": (-0.035, 2.65), "limestone": (0.0, 2.71), "dolomite": (0.02, 2.87)}
WATER = (1.0, 1.0)


def _zone_mask(res, zone):
    z = res["depth"]
    if zone is None:
        return np.ones(len(z), bool)
    row = next((r for r in res["zones"] if r["Zone"] == zone), None)
    if row is None:
        return np.zeros(len(z), bool)
    return (z >= row["Top_MD"]) & (z < row["Base_MD"])


def _thin(*arrays, n=MAX_POINTS):
    ok = np.ones(len(arrays[0]), bool)
    for a in arrays:
        ok &= np.isfinite(a)
    idx = np.where(ok)[0]
    if len(idx) > n:
        idx = idx[np.linspace(0, len(idx) - 1, n).astype(int)]
    return [np.asarray(a)[idx] for a in arrays], idx


def nd(res, logs, zone=None, color="VSH"):
    m = _zone_mask(res, zone)
    x, y = logs.get("NPHI"), logs.get("RHOB")
    if x is None or y is None:
        return None
    c = res["curves"].get(color, logs.get(color))
    c = np.full(len(x), np.nan) if c is None else c
    (xx, yy), idx = _thin(np.where(m, x, np.nan), y)
    lines = {}
    for name, (n0, r0) in ND_LINES.items():
        t = np.linspace(0, 0.4, 9)  # porosity along the line
        lines[name] = (n0 + t * (WATER[0] - n0), r0 + t * (WATER[1] - r0), t)
    return {"kind": "N-D", "x": xx, "y": yy, "c": np.asarray(c)[idx], "xlabel": "NPHI (v/v, limestone units)",
            "ylabel": "RHOB (g/cc)", "xlim": (-0.05, 0.6), "ylim": (2.95, 1.9), "lines": lines, "color": color}


def mn(res, logs, zone=None, dt_fl=189.0, rho_fl=1.0, nphi_fl=1.0, minerals=None):
    from .multimin import MINERALS
    minerals = minerals or ["quartz", "calcite", "dolomite", "illite", "anhydrite"]
    m = _zone_mask(res, zone)
    dt, rho, nph = logs.get("DT"), logs.get("RHOB"), logs.get("NPHI")
    if dt is None or rho is None or nph is None:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        M = 0.01 * (dt_fl - dt) / (rho - rho_fl)
        N = (nphi_fl - nph) / (rho - rho_fl)
    (xx, yy), idx = _thin(np.where(m, N, np.nan), M)
    pts = {}
    for k in minerals:
        e = MINERALS[k]
        pts[k] = ((nphi_fl - e["NPHI"]) / (e["RHOB"] - rho_fl), 0.01 * (dt_fl - e["DT"]) / (e["RHOB"] - rho_fl))
    return {"kind": "M-N", "x": xx, "y": yy, "c": res["curves"]["VSH"][idx], "points": pts, "xlabel": "N",
            "ylabel": "M", "xlim": (0.3, 0.8), "ylim": (0.5, 1.1), "color": "VSH"}


def pickett(res, zone=None, vsh_max=None):
    m = _zone_mask(res, zone)
    c = res["curves"]
    if vsh_max is not None:
        m &= np.nan_to_num(c["VSH"], nan=1) <= vsh_max
    (rt, phi), idx = _thin(np.where(m & (res["rt"] > 0) & (c["PHIE"] > 0.01), res["rt"], np.nan), c["PHIE"])
    u = res["used"].get(zone, {}) if zone else {}
    a, mm, n, rw = u.get("a", 1.0), u.get("m", 2.0), u.get("n", 2.0), u.get("rw", 0.05)
    ph = np.logspace(np.log10(0.02), np.log10(0.5), 20)
    lines = {f"Sw {s:g}": (a * rw / (ph ** mm * s ** n), ph) for s in (1.0, 0.5, 0.25)}
    return {"kind": "Pickett", "x": rt, "y": phi, "c": c["VSH"][idx], "lines": lines, "xlabel": "Rt (ohm.m)",
            "ylabel": "PHIE", "logx": True, "logy": True, "xlim": (0.1, 1000), "ylim": (0.01, 0.5),
            "params": {"a": a, "m": mm, "n": n, "rw": rw}, "color": "VSH"}


def buckles(res, zone=None, net_only=True):
    m = _zone_mask(res, zone)
    c = res["curves"]
    if net_only:
        m &= c["NET_RES"] > 0.5
    (phi, sw), idx = _thin(np.where(m, c["PHIE"], np.nan), c["SW"])
    ph = np.linspace(0.02, 0.4, 40)
    lines = {f"BVW {b:g}": (ph, np.clip(b / ph, 0, 1)) for b in (0.02, 0.04, 0.06, 0.08, 0.1)}
    return {"kind": "Buckles", "x": phi, "y": sw, "c": c["VSH"][idx], "lines": lines, "xlabel": "PHIE",
            "ylabel": "Sw", "xlim": (0, 0.4), "ylim": (0, 1), "color": "VSH"}


def histogram(values, bins=50, discriminators=None, curves=None, mask=None, rng=None, log=False):
    """Histogram of `values` where every discriminator {name: (lo, hi)} on `curves` holds (and mask)."""
    v = np.asarray(values, float)
    sel = np.isfinite(v) if mask is None else (np.isfinite(v) & np.asarray(mask, bool))
    for name, (lo, hi) in (discriminators or {}).items():
        cv = np.asarray((curves or {})[name], float)
        with np.errstate(invalid="ignore"):
            sel &= (cv >= lo) & (cv <= hi)
    x = v[sel]
    if log:
        x = np.log10(x[x > 0])
    if not x.size:
        return {"kind": "Histogram", "counts": np.zeros(bins), "edges": np.linspace(0, 1, bins + 1), "n": 0}
    counts, edges = np.histogram(x, bins=bins, range=rng)
    return {"kind": "Histogram", "counts": counts, "edges": edges, "n": int(x.size), "mean": float(x.mean()),
            "p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)),
            "log": log}


def render_png(data, path, title=""):
    """Draw one cross-plot / histogram dict with matplotlib (Agg) - for reports and for checking without Qt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
    if data["kind"] == "Histogram":
        e = data["edges"]
        ax.bar(0.5 * (e[:-1] + e[1:]), data["counts"], width=np.diff(e), color="#4a78b5", edgecolor="white")
        ax.set_ylabel("count")
    else:
        sc = ax.scatter(data["x"], data["y"], c=data["c"], s=4, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(sc, ax=ax, label=data.get("color", ""))
        for name, ln in data.get("lines", {}).items():
            ax.plot(ln[0], ln[1], lw=1, label=name)
        for name, (px, py) in data.get("points", {}).items():
            ax.plot(px, py, "k^")
            ax.annotate(name, (px, py), fontsize=7)
        if data.get("logx"):
            ax.set_xscale("log")
        if data.get("logy"):
            ax.set_yscale("log")
        ax.set_xlim(*data["xlim"])
        ax.set_ylim(*data["ylim"])
        ax.set_xlabel(data["xlabel"])
        ax.set_ylabel(data["ylabel"])
        if data.get("lines"):
            ax.legend(fontsize=7)
    ax.set_title(title or data["kind"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
