"""Basin model settings + project glue (PLAN 18e): settings in project.display["basin"]["settings"], 1D models
of wells (Well Correlation tops + BHTs), pseudo-wells along a section / on maps from the item-17 depth model."""
import copy

import numpy as np

from . import burial as B
from . import maturity as MT
from . import pseudowells as PW
from . import thermal as TH
from .chrono import AgeTable

DEFAULTS = {"heat_flow": {"kind": "constant", "q": 60.0}, "surface_t": 12.0, "water_depth": 110.0, "step": 1.0,
            "kinetics": "Type III (terrigenous / coaly, organofacies D/E)", "stride": 10, "trap": None,
            "well_water": {}, "calibrated": {}, "section_columns": 24}


def settings(project):
    b = (project.display.get("basin") or {}) if project is not None else {}
    s = copy.deepcopy(DEFAULTS)
    s.update(copy.deepcopy(b.get("settings") or {}))
    return s


def save_settings(project, s):
    b = dict(project.display.get("basin") or {})
    b["settings"] = copy.deepcopy(s)
    project.display["basin"] = b


def heat_flow_spec(s, well=None):
    """Heat-flow spec of the settings; a calibrated well uses its fitted present value (scales the shape)."""
    spec = copy.deepcopy(s["heat_flow"])
    cal = (s.get("calibrated") or {}).get(well) if well else None
    if cal:
        spec = {"kind": "scaled", "base": spec, "present": float(cal["present_q"])}
    return spec


def q_history(ages, spec):
    if isinstance(spec, dict) and spec.get("kind") == "scaled":
        q = TH.heat_flow_history(ages, spec["base"])
        return q * spec["present"] / max(1e-9, q[-1])
    return TH.heat_flow_history(ages, spec)


# ====================================================================== wells
def well_column(wcp, well, table, s):
    """(depths {entry: TVDSS}, water depth) of a well; the Seabed entry gets the well's water depth."""
    d = B.well_depths(wcp, well, table)
    wd = float((s.get("well_water") or {}).get(well, s["water_depth"]))
    if table.get("Seabed") is not None:
        d["Seabed"] = wd
    return d, wd


def well_model(wcp, well, table, s, heat_flow=None):
    """Burial + thermal history of a well -> (BurialHistory, ThermalHistory)."""
    d, wd = well_column(wcp, well, table, s)
    if len(d) < 2:
        raise ValueError(f"{well}: fewer than two dated tops")
    bh = B.BurialHistory(B.column_units(table, d, wd), step=s["step"], water_depth=wd)
    q = q_history(bh.ages, heat_flow if heat_flow is not None else heat_flow_spec(s, well))
    return bh, TH.ThermalHistory(bh, q, s["surface_t"])


def well_observations(wcp, well, s):
    """Corrected, screened temperatures of a well [{md, tvdss, T, raw, method, ok}]."""
    w = wcp.load_well(well)
    recs = TH.well_temperatures(w.las_paths, s["surface_t"])
    wd = float((s.get("well_water") or {}).get(well, s["water_depth"]))
    return TH.screen_temperatures(recs, lambda md: float(w.md_to_tvdss(md)), s["surface_t"], wd)


def calibrate_well(wcp, well, table, s, obs=None):
    """Fit the heat-flow history to the well's temperatures; stores the present value in s["calibrated"]."""
    d, wd = well_column(wcp, well, table, s)
    bh = B.BurialHistory(B.column_units(table, d, wd), step=s["step"], water_depth=wd)
    if obs is None:
        obs = [(r["tvdss"], r["T"]) for r in well_observations(wcp, well, s) if r["ok"]]
    if not len(obs):
        raise ValueError(f"{well}: no usable temperatures (BHT) in the LAS / DLIS headers")
    fit = TH.fit_heat_flow(bh, obs, q_history(bh.ages, s["heat_flow"]), s["surface_t"])
    s.setdefault("calibrated", {})[well] = {"present_q": fit["present_q"], "rms": fit["rms"], "n": len(obs)}
    fit["obs"] = np.asarray(obs, float)
    return fit


def well_maturity(th, kinetics):
    """[(surface, age, T today, %Ro today, TR today, window, oil entry Ma)] for each dated surface."""
    rows = []
    for name, age, T in th.surface_temperatures():
        ro = MT.easy_ro(th.ages, T)
        tr = MT.transformation_ratio(th.ages, T, kinetics)
        rows.append((name, age, float(T[-1]), float(ro[-1]), float(tr[-1]), MT.window_name(ro[-1]),
                     MT.crossing_age(th.ages, ro, MT.WINDOWS["oil"][0])))
    return rows, ro


# ====================================================================== depth model columns
def vm_columns(vm, table, x, y, s):
    """(entry names, depths (B, K), seabed (B,)) at map points from a VelocityModel's depth maps."""
    pts = B.velocity_model_depths(vm, table, x, y)
    names = [e["name"] for e in table.entries if any(e["name"] in p for p in pts)]
    D = np.array([[p.get(n, np.nan) for n in names] for p in pts], float).reshape(len(pts), len(names))
    if "Seabed" in names:
        sb = D[:, names.index("Seabed")]
    else:
        sb = np.full(len(pts), float(s["water_depth"]))
        if table.get("Seabed") is not None:
            names.append("Seabed")
            D = np.c_[D, sb]
    return names, D, sb


def section_property(vm, table, x, y, axis, prop, t_eval, s):
    """Model property on an age axis along a section (len(axis), ntr): pseudo-wells every few traces through
    the depth model, interpolated to every trace."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    nc = int(min(n, max(2, s.get("section_columns", 24))))
    ci = np.unique(np.linspace(0, n - 1, nc).round().astype(int))
    names, D, sb = vm_columns(vm, table, x[ci], y[ci], s)
    if len(names) < 2:
        raise ValueError("the depth model has fewer than two layers matched to the age table")
    res = PW.run_columns(table, names, D, sb, s["heat_flow"], None, s["surface_t"], s["step"],
                         kinetics=s["kinetics"], full=True)
    if "T" not in res:
        raise ValueError("no pseudo-well on this section has all its depths")
    W = PW.wheeler_property(res, axis, prop, t_eval)
    return PW.to_traces(W, ci.astype(float), np.arange(n, dtype=float))


def load_table(project):
    return AgeTable.load(project)
