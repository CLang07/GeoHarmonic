"""Well-log track styling (Seismic tool PLAN 40h) - pure numpy, drawn by section.py / export.py.

- Lithology pattern library (editable, saved in project.display['lith_patterns']): sand (dots), shale (dashes),
  silt, limestone (bricks), dolomite (bricks with rhombs), coal (solid black), anhydrite, salt, marl ... each = fill
  colour + pattern kind + spacing; `pattern_tile` rasterises one tile, `pattern_column` a whole column by code.
- Gradient / colour fill of any curve through a colour table (`curve_fill_image`, e.g. GR yellow -> green / brown).
- Shaded pay / net flags from cutoffs (`net_pay`: Vsh <=, PHI >=, Sw <=) -> flags + intervals + net / pay thickness.
- Neutron-density crossover shading (`crossover`): gas (density porosity > neutron porosity) and shale separation.
- Track templates (`save_template` / `load_template`) - the styled track set saved with the project.
"""
import copy

import numpy as np

PATTERN_KINDS = ("solid", "dots", "dashes", "lines", "bricks", "rhombs", "crosses", "waves")
DEFAULT_PATTERNS = {
    "Sand": {"colour": "#f2d64b", "kind": "dots", "spacing": 6, "ink": "#6b5a10"},
    "Silt": {"colour": "#c9b97a", "kind": "dots", "spacing": 4, "ink": "#5c5230"},
    "Shale": {"colour": "#8e9a7c", "kind": "dashes", "spacing": 6, "ink": "#3a4232"},
    "Limestone": {"colour": "#5b9bd5", "kind": "bricks", "spacing": 8, "ink": "#1f3f60"},
    "Dolomite": {"colour": "#9b7fc6", "kind": "rhombs", "spacing": 8, "ink": "#3a2a5c"},
    "Coal": {"colour": "#1e1e1e", "kind": "solid", "spacing": 6, "ink": "#1e1e1e"},
    "Anhydrite": {"colour": "#d9a0c8", "kind": "crosses", "spacing": 8, "ink": "#6b2f5b"},
    "Salt": {"colour": "#e6f2ff", "kind": "crosses", "spacing": 6, "ink": "#7a8aa0"},
    "Marl": {"colour": "#a3c49a", "kind": "waves", "spacing": 8, "ink": "#36503a"},
    "Carbonate": {"colour": "#5b9bd5", "kind": "bricks", "spacing": 8, "ink": "#1f3f60"},
}
# lithology codes of lithology.py -> library names
CODE_NAMES = {0: "Sand", 1: "Silt", 2: "Shale", 3: "Coal", 4: "Carbonate"}
KEY_PATTERNS = "lith_patterns"
KEY_TEMPLATES = "track_templates"


def hex_rgba(h, a=255):
    h = str(h).lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


def library(project=None):
    """Default patterns overlaid with the project's edits (name -> spec)."""
    lib = copy.deepcopy(DEFAULT_PATTERNS)
    d = (getattr(project, "display", None) or {}).get(KEY_PATTERNS) if project is not None else None
    for k, v in (d or {}).items():
        lib[k] = dict(lib.get(k, {}), **v)
    return lib


def set_pattern(project, name, colour=None, kind=None, spacing=None, ink=None):
    """Edit / add a library entry (saved in the project)."""
    if kind is not None and kind not in PATTERN_KINDS:
        raise ValueError(f"pattern kind must be one of {PATTERN_KINDS}")
    d = project.display.setdefault(KEY_PATTERNS, {})
    e = d.setdefault(name, {})
    for k, v in (("colour", colour), ("kind", kind), ("spacing", spacing), ("ink", ink)):
        if v is not None:
            e[k] = int(v) if k == "spacing" else v
    return library(project)[name]


def pattern_tile(spec, size=16):
    """(size, size, 4) uint8 tile: fill colour + ink pattern (tiles seamlessly)."""
    s = int(size)
    bg = np.array(hex_rgba(spec.get("colour", "#ffffff")), np.uint8)
    ink = np.array(hex_rgba(spec.get("ink", "#000000")), np.uint8)
    t = np.tile(bg, (s, s, 1))
    k = spec.get("kind", "solid")
    sp = max(2, int(spec.get("spacing", 6)))
    yy, xx = np.mgrid[0:s, 0:s]
    if k == "dots":
        m = ((yy % sp) == sp // 2) & ((xx + (yy // sp % 2) * (sp // 2)) % sp == sp // 2)
    elif k == "dashes":
        m = ((yy % sp) == 0) & ((xx + (yy // sp % 2) * (sp // 2)) % sp < sp // 2 + 1)
    elif k == "lines":
        m = (yy % sp) == 0
    elif k == "bricks":
        m = ((yy % sp) == 0) | (((xx + (yy // sp % 2) * sp) % (2 * sp)) == 0)
    elif k == "rhombs":
        m = ((yy % sp) == 0) | (((xx + (yy // sp % 2) * sp) % (2 * sp)) == 0) | \
            (np.abs((xx % sp) - sp / 2) + np.abs((yy % sp) - sp / 2) == sp // 2)
    elif k == "crosses":
        m = ((xx % sp) == (yy % sp)) | ((xx % sp) == (sp - 1 - yy % sp))
    elif k == "waves":
        m = (((yy + np.round(1.5 * np.sin(2 * np.pi * xx / sp)).astype(int)) % sp) == 0)
    else:
        m = np.zeros((s, s), bool)
    t[m] = ink
    return t


def pattern_column(codes, width=24, lib=None, names=None):
    """Lithology column image (rows = samples): each code drawn with its library pattern (tile repeated)."""
    lib = lib or DEFAULT_PATTERNS
    names = names or CODE_NAMES
    codes = np.asarray(codes)
    n = len(codes)
    img = np.zeros((n, int(width), 4), np.uint8)
    for c in np.unique(codes[np.isfinite(codes)] if codes.dtype.kind == "f" else codes):
        nm = names.get(int(c))
        if nm is None or nm not in lib:
            continue
        tile = pattern_tile(lib[nm], 16)
        rows = np.flatnonzero(codes == c)
        img[rows] = tile[rows[:, None] % 16, np.arange(int(width))[None, :] % 16]
    return img


def curve_fill_image(values, lut, vmin, vmax, ncols=40, x=None, baseline=None, side="left"):
    """Gradient / colour fill of a curve (rows = samples): every row coloured by the curve value through the colour
    table.  With x (curve position 0..1 across the track) and a baseline the fill covers only baseline..curve
    ('left' fills from the left edge to the curve, like a GR shading)."""
    v = np.asarray(values, float)
    lut = np.asarray(lut)
    t = np.clip((v - vmin) / (vmax - vmin if vmax != vmin else 1.0), 0, 1)
    k = np.where(np.isfinite(t), np.round(t * (len(lut) - 1)), 0).astype(int)
    col = lut[k].astype(np.uint8)
    if col.shape[1] == 3:
        col = np.c_[col, np.full(len(col), 255, np.uint8)]
    img = np.repeat(col[:, None, :], int(ncols), axis=1)
    img[~np.isfinite(v)] = 0
    if x is not None:
        cx = (np.arange(ncols) + 0.5) / ncols
        xx = np.asarray(x, float)
        b = 0.0 if baseline is None else float(baseline)
        if side == "left":
            inside = cx[None, :] <= np.nan_to_num(xx, nan=-1)[:, None]
        elif side == "right":
            inside = cx[None, :] >= np.nan_to_num(xx, nan=2)[:, None]
        else:  # between the baseline and the curve
            lo, hi = np.minimum(xx, b), np.maximum(xx, b)
            inside = (cx[None, :] >= np.nan_to_num(lo, nan=2)[:, None]) & (cx[None, :] <= np.nan_to_num(hi, nan=-1)[:, None])
        img[~inside] = 0
    return img


def net_pay(depth, vsh=None, phi=None, sw=None, vsh_max=0.4, phi_min=0.08, sw_max=0.6):
    """Net (reservoir) = Vsh <= vsh_max and PHI >= phi_min; pay = net and Sw <= sw_max.
    Returns {net, pay (bool arrays), net_m, pay_m (thickness), intervals_net, intervals_pay [(top, base)]}."""
    d = np.asarray(depth, float)
    n = len(d)
    ok = np.ones(n, bool)
    net = ok.copy()
    if vsh is not None:
        v = np.asarray(vsh, float)
        net &= np.isfinite(v) & (v <= vsh_max)
    if phi is not None:
        p = np.asarray(phi, float)
        net &= np.isfinite(p) & (p >= phi_min)
    if vsh is None and phi is None:
        net[:] = False
    pay = net.copy()
    if sw is not None:
        s = np.asarray(sw, float)
        pay &= np.isfinite(s) & (s <= sw_max)
    else:
        pay[:] = False
    step = np.gradient(d) if n > 1 else np.ones(n)
    return {"net": net, "pay": pay, "net_m": float(np.abs(step[net]).sum()), "pay_m": float(np.abs(step[pay]).sum()),
            "intervals_net": intervals(d, net), "intervals_pay": intervals(d, pay)}


def intervals(depth, flag):
    """[(top, base)] runs of True."""
    d = np.asarray(depth, float)
    f = np.asarray(flag, bool)
    if not f.any():
        return []
    e = np.diff(np.r_[0, f.astype(int), 0])
    starts, ends = np.flatnonzero(e == 1), np.flatnonzero(e == -1) - 1
    return [(float(d[a]), float(d[b])) for a, b in zip(starts, ends)]


def flag_image(flag, rgba=(220, 40, 40, 200), ncols=6):
    f = np.asarray(flag, bool)
    img = np.zeros((len(f), int(ncols), 4), np.uint8)
    img[f] = rgba
    return img


def density_porosity(rhob, rho_ma=2.65, rho_fl=1.0):
    return (rho_ma - np.asarray(rhob, float)) / (rho_ma - rho_fl)


def crossover(rhob, nphi, rho_ma=2.65, rho_fl=1.0, min_sep=0.0):
    """Neutron-density crossover on a limestone / sandstone compatible scale: gas where density porosity exceeds
    neutron porosity (curves cross), shale where neutron reads higher.  Returns {gas, shale, separation}."""
    dphi = density_porosity(rhob, rho_ma, rho_fl)
    n = np.asarray(nphi, float)
    sep = dphi - n
    ok = np.isfinite(sep)
    return {"gas": ok & (sep > min_sep), "shale": ok & (sep < -min_sep), "separation": sep}


# ====================================================================== templates
def default_template():
    return {"tracks": ["LITH", "GR", "RES", "DN", "DT"], "gr_fill": "gradient", "gr_lut": "Sand-shale",
            "pay": {"vsh_max": 0.4, "phi_min": 0.08, "sw_max": 0.6, "show": True},
            "crossover": True, "patterns": True}


def save_template(project, name, tpl):
    project.display.setdefault(KEY_TEMPLATES, {})[str(name)] = copy.deepcopy(tpl)
    return project.display[KEY_TEMPLATES][str(name)]


def load_template(project, name):
    t = default_template()
    t.update(copy.deepcopy(((project.display or {}).get(KEY_TEMPLATES) or {}).get(str(name)) or {}))
    return t


def apply_template(project, name):
    """Copy a saved template into the live display settings (the section redraws from project.display)."""
    t = load_template(project, name)
    project.display["tracks"] = list(t["tracks"])
    project.display["gr_fill"] = t["gr_fill"]
    project.display["gr_lut"] = t["gr_lut"]
    project.display["pay"] = dict(t["pay"])
    project.display["crossover"] = bool(t["crossover"])
    project.display["patterns"] = bool(t["patterns"])
    return t


def gr_lut(name="Sand-shale"):
    """Colour table for GR fills - from the shared Seismic-tool library when it is importable (one library,
    PLAN 40b), else a built-in sand-yellow -> shale-green / brown ramp."""
    try:
        from .maplink import import_mapbus
        import_mapbus()  # puts the Seismic tool folder on sys.path when it is a sibling
        from seismic_tool import colortables as CT
        t = CT.get(name)
        if t is not None:
            return t.lut(apply_display=False)
    except Exception:
        pass
    stops = [(0.0, (242, 214, 75)), (0.45, (200, 185, 110)), (0.7, (142, 154, 124)), (1.0, (110, 80, 50))]
    x = np.linspace(0, 1, 256)
    pos = np.array([s[0] for s in stops])
    col = np.array([s[1] for s in stops], float)
    out = np.empty((256, 4), np.uint8)
    for k in range(3):
        out[:, k] = np.interp(x, pos, col[:, k]).astype(np.uint8)
    out[:, 3] = 255
    return out
