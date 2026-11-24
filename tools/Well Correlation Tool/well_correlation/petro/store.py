"""Where petrophysics results live: ``<project>_petro/`` next to the .wcproj.json (PLAN item 20 design).

    <project>_petro/<well>.npz        computed curves on the well's composite MD grid (QC flags, VSH, PHIE, SW, ...)
    <project>_petro/qc/<well>.json    20a QC + tool report
    <project>_petro/qc_summary.csv    one row per well x family
    <project>_petro/summary.csv       per-zone results (20b)
    <project>_petro/tool_table.json   optional user edits of the tool table
"""
import json
import os

import numpy as np

from ..project import PROJECT_EXT, _safe


def petro_dir(project, create=False, override=None):
    d = override or getattr(project, "petro_out", None)
    if d is None:
        if not project.path:
            return None
        base = project.path[: -len(PROJECT_EXT)] if project.path.endswith(PROJECT_EXT) else project.path
        d = base + "_petro"
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def curves_path(project, well):
    d = petro_dir(project)
    return os.path.join(d, f"{_safe(well)}.npz") if d else None


def load_curves(project, well):
    """(depth, {name: array}) of saved computed curves, or (None, {})."""
    p = curves_path(project, well)
    if not p or not os.path.exists(p):
        return None, {}
    with np.load(p, allow_pickle=False) as z:
        depth = z["depth"]
        return depth, {k[2:]: z[k] for k in z.files if k.startswith("c_")}


def save_curves(project, well, depth, curves, replace_prefix=None):
    """Merge `curves` into the well's npz (same MD grid; a different grid replaces the file).
    replace_prefix: drop existing curves starting with it first (e.g. 'QC_' when QC is re-run)."""
    petro_dir(project, create=True)
    p = curves_path(project, well)
    old_depth, old = load_curves(project, well)
    if old_depth is None or len(old_depth) != len(depth) or not np.allclose(old_depth, depth):
        old = {}
    if replace_prefix:
        old = {k: v for k, v in old.items() if not k.startswith(replace_prefix)}
    old.update({k: np.asarray(v) for k, v in curves.items()})
    tmp = p + ".tmp.npz"
    np.savez_compressed(tmp, depth=np.asarray(depth, float), **{"c_" + k: v for k, v in old.items()})
    os.replace(tmp, p)
    return p


def save_json(project, sub, name, obj):
    d = os.path.join(petro_dir(project, create=True), sub)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{_safe(name)}.json")
    with open(p + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(p + ".tmp", p)
    return p


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
