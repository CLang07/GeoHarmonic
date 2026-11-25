"""Glow / Halo display effects (PLAN 39) - everything except the Qt drawing, pure Python + numpy, unit-tested.

A PowerPoint-style glow is drawn as 2-4 wider, more transparent "underlay" strokes under the crisp line; a halo is one
solid contrasting outline.  The Qt side (`gui/glow.py`) turns the stroke specs below into non-interactive pyqtgraph
underlay items; matplotlib exports use `mpl_path_effects`; the 3D viewer uses `glow_meshes` (scene3d meshes).

    st = GlowSettings.load(project)                  # project.display['glow'] (global settings + per-object overrides)
    mode = st.effective("horizon", "Top", emphasis="active")   # 'glow' | 'halo' | None
    for width_px, rgba in strokes(st, mode, base_rgb=(255, 220, 0), line_width=2, emphasis="active", lut=lut): ...
    st.set_override("horizon", "Top", "off")         # per-object: 'on' | 'off' | 'global' (+ colour)
    st.save(project)

Emphasis: None (plain object), "selected", "active", "hover", "qc" (QC flags glow red), "tracking" (autotracker front).
The active / selected object glows stronger; with `active_only` (default on for dense projects, see `auto_limit`)
plain objects do not glow at all.

Map bus (PLAN 39f): `GlowState(bus)` follows SelectionService / ActiveObject so selecting a horizon, fault or well in
any window makes it glow in every window, and settings changes are broadcast as a "display" event (kind "glow").
"""
import copy

import numpy as np

MODES = ("glow", "halo", "off")
COLOUR_MODES = ("match", "white", "black", "custom")
OVERRIDES = ("global", "on", "off")
EMPHASES = (None, "selected", "active", "hover", "qc", "tracking")
# object kinds that glow by default (PLAN 39e); everything else only when selected / active / hovered / QC-flagged
DEFAULT_KINDS = ("horizon", "fault", "stick", "fault_polygon", "fault_surface", "well", "well_top", "geobody",
                 "section_line", "current_line", "plane_outline", "tracking")
QC_RGB = (255, 40, 30)
AUTO_LIMIT = 2000  # more glowing items than this -> active / selected only (PLAN 39h)

DEFAULTS = {
    "enabled": True,
    "mode": "glow",          # glow | halo | off
    "width_px": 6.0,         # how far the glow reaches beyond the line, screen pixels (each side)
    "intensity": 0.75,       # 0..1 opacity of the innermost glow stroke
    "layers": 3,             # falloff strokes 1..4
    "colour": "match",       # match (object colour) | white | black | custom
    "custom": (255, 255, 160),
    "halo_colour": "auto",   # auto (black / white against the seismic colour table) | white | black | custom
    "halo_custom": (0, 0, 0),
    "halo_px": 2.0,          # halo outline width each side
    "active_only": False,    # only selected / active / hovered / QC objects glow
    "active_boost": 1.6,     # width and intensity factor for the active / selected object
    "hover": True,           # hover highlight (throttled)
    "hover_ms": 60,
    "kinds": list(DEFAULT_KINDS),
    "threshold": AUTO_LIMIT,
    "export_glow": True,     # PNG / figure / turntable exports include glow ("Export without glow" unticks it)
}


def _rgb(c, default=(255, 255, 255)):
    if c is None:
        return tuple(default)
    if isinstance(c, str):
        s = c.strip().lstrip("#")
        named = {"white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0), "yellow": (255, 255, 0)}
        if c.lower() in named:
            return named[c.lower()]
        if len(s) == 6:
            return tuple(int(s[k:k + 2], 16) for k in (0, 2, 4))
        return tuple(default)
    v = [float(x) for x in list(c)[:3]]
    if max(v) <= 1.0 and any(0 < x < 1 for x in v):
        v = [x * 255.0 for x in v]
    return tuple(int(round(min(255, max(0, x)))) for x in v)


def luminance(rgb):
    """Relative luminance 0..1 (sRGB, Rec. 709 weights on linearised channels)."""
    c = np.asarray(rgb, float)[..., :3] / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722])


def contrast_colour(lut=None, against=None):
    """Black or white halo, whichever stands out more against the displayed seismic.

    `lut` (256 x 3/4 uint8) is weighted toward its middle (most seismic samples sit near zero amplitude, so the
    centre of the bar is what a line is mostly drawn over); `against` = one RGB background instead."""
    if against is not None:
        lum = float(luminance(_rgb(against)))
    elif lut is not None:
        L = np.asarray(lut, float)[:, :3]
        x = np.linspace(-1, 1, len(L))
        w = np.exp(-0.5 * (x / 0.35) ** 2)
        lum = float((luminance(L) * w).sum() / w.sum())
    else:
        lum = 1.0  # white plot background
    # contrast ratio against black vs white (WCAG): pick the larger
    return (0, 0, 0) if (lum + 0.05) / 0.05 >= 1.05 / (lum + 0.05) else (255, 255, 255)


class GlowSettings:
    """Global glow settings + per-object overrides, saved in project.display['glow']."""
    KEY = "glow"

    def __init__(self, **kw):
        self.d = copy.deepcopy(DEFAULTS)
        self.overrides = {}  # "kind:name" -> {"mode": on|off|global, "colour": rgb|None}
        self.update(**kw)
        self._listeners = []

    # ------------------------------------------------------------ values
    def __getitem__(self, k):
        return self.d[k]

    def get(self, k, default=None):
        return self.d.get(k, default)

    def update(self, **kw):
        for k, v in kw.items():
            if k not in DEFAULTS:
                raise KeyError(f"unknown glow setting {k!r}")
            self.d[k] = v
        self.validate()
        for fn in list(getattr(self, "_listeners", [])):
            fn(self)
        return self

    def validate(self):
        d = self.d
        d["enabled"] = bool(d["enabled"])
        if d["mode"] not in MODES:
            d["mode"] = "glow"
        d["width_px"] = float(np.clip(float(d["width_px"]), 0.5, 40.0))
        d["intensity"] = float(np.clip(float(d["intensity"]), 0.0, 1.0))
        d["layers"] = int(np.clip(int(d["layers"]), 1, 4))
        if d["colour"] not in COLOUR_MODES:
            d["colour"] = "match"
        d["custom"] = _rgb(d["custom"])
        if d["halo_colour"] not in ("auto", "white", "black", "custom"):
            d["halo_colour"] = "auto"
        d["halo_custom"] = _rgb(d["halo_custom"], (0, 0, 0))
        d["halo_px"] = float(np.clip(float(d["halo_px"]), 0.5, 12.0))
        d["active_boost"] = float(np.clip(float(d["active_boost"]), 1.0, 4.0))
        d["threshold"] = int(max(1, int(d["threshold"])))
        d["hover_ms"] = int(max(0, int(d["hover_ms"])))
        d["kinds"] = [str(k) for k in (d["kinds"] or [])]
        return self

    def on_change(self, fn):
        self._listeners.append(fn)

    def toggle(self):
        """Shortcut G: glow on / off globally."""
        return self.update(enabled=not self.d["enabled"])

    # ------------------------------------------------------------ per-object overrides
    @staticmethod
    def key(kind, name):
        return f"{kind}:{name}"

    def set_override(self, kind, name, mode="global", colour=None):
        if mode not in OVERRIDES:
            raise ValueError(f"glow override must be one of {OVERRIDES}")
        k = self.key(kind, name)
        if mode == "global" and colour is None:
            self.overrides.pop(k, None)
        else:
            self.overrides[k] = {"mode": mode, "colour": None if colour is None else _rgb(colour)}
        for fn in list(self._listeners):
            fn(self)

    def override(self, kind, name):
        return self.overrides.get(self.key(kind, name), {"mode": "global", "colour": None})

    def rename(self, kind, old, new):
        o = self.overrides.pop(self.key(kind, old), None)
        if o is not None:
            self.overrides[self.key(kind, new)] = o

    # ------------------------------------------------------------ decision
    def effective(self, kind, name=None, emphasis=None, limited=False):
        """'glow' | 'halo' | None for one object.

        Per-object override 'off' always wins, 'on' glows even when the global switch is off; QC flags and the
        tracking front glow whenever glow is on; with `active_only` (or `limited` = auto-limit hit) a plain object
        does not glow."""
        o = self.override(kind, name) if name is not None else {"mode": "global"}
        mode = self.d["mode"] if self.d["mode"] != "off" else "glow"
        if o["mode"] == "off":
            return None
        if o["mode"] == "on":
            return mode
        if not self.d["enabled"] or self.d["mode"] == "off":
            return None
        if emphasis in ("selected", "active", "qc", "tracking"):
            return mode
        if emphasis == "hover":
            return mode if self.d["hover"] else None
        if self.d["active_only"] or limited:
            return None
        return mode if kind in self.d["kinds"] else None

    # ------------------------------------------------------------ persistence
    def to_dict(self):
        d = copy.deepcopy(self.d)
        d["custom"], d["halo_custom"] = list(d["custom"]), list(d["halo_custom"])
        return {"settings": d, "overrides": {k: {"mode": v["mode"], "colour": None if v["colour"] is None
                                                 else list(v["colour"])} for k, v in self.overrides.items()}}

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        s = cls()
        known = {k: v for k, v in (d.get("settings") or {}).items() if k in DEFAULTS}
        s.d.update(known)
        s.validate()
        for k, v in (d.get("overrides") or {}).items():
            if isinstance(v, dict) and v.get("mode") in OVERRIDES:
                s.overrides[str(k)] = {"mode": v["mode"], "colour": None if v.get("colour") is None
                                       else _rgb(v["colour"])}
        return s

    def save(self, project):
        disp = getattr(project, "display", None)
        if disp is None:
            project.display = disp = {}
        disp[self.KEY] = self.to_dict()
        return disp[self.KEY]

    @classmethod
    def load(cls, project):
        return cls.from_dict(((getattr(project, "display", None) or {}).get(cls.KEY)) if project is not None else None)


# ====================================================================== stroke specs
def glow_colour(st, base_rgb, override=None, emphasis=None):
    if emphasis == "qc":
        return QC_RGB
    if override is not None and override.get("colour") is not None:
        return _rgb(override["colour"])
    c = st["colour"]
    if c == "white":
        return (255, 255, 255)
    if c == "black":
        return (0, 0, 0)
    if c == "custom":
        return _rgb(st["custom"])
    return _rgb(base_rgb)


def halo_colour(st, lut=None, background=None, override=None):
    if override is not None and override.get("colour") is not None:
        return _rgb(override["colour"])
    h = st["halo_colour"]
    if h == "white":
        return (255, 255, 255)
    if h == "black":
        return (0, 0, 0)
    if h == "custom":
        return _rgb(st["halo_custom"])
    return contrast_colour(lut, background)


def strokes(st, mode, base_rgb=(255, 255, 0), line_width=2.0, emphasis=None, lut=None, background=None,
            override=None):
    """Underlay strokes, OUTERMOST first: [(width_px, (r, g, b, a 0-255))].  Empty for mode None / 'off'.

    Glow: `layers` strokes, the k-th (k = 1 outermost) has total width line + 2 * reach * (n - k + 1) / n and
    alpha = intensity * ((k) / n) ** 1.5 * 255 - wide and faint outside, narrow and strong inside (additive look).
    Halo: one solid stroke line + 2 * halo_px.  Active / selected / tracking: reach and intensity x active_boost."""
    if mode not in ("glow", "halo"):
        return []
    boost = st["active_boost"] if emphasis in ("active", "selected", "tracking", "qc") else 1.0
    lw = float(line_width)
    if mode == "halo":
        c = halo_colour(st, lut, background, override) if emphasis != "qc" else QC_RGB
        return [(lw + 2.0 * st["halo_px"] * (1.0 + 0.5 * (boost - 1.0)), (c[0], c[1], c[2], 255))]
    c = glow_colour(st, base_rgb, override, emphasis)
    n = int(st["layers"])
    reach = st["width_px"] * boost
    inten = min(1.0, st["intensity"] * (1.0 + 0.5 * (boost - 1.0)))
    out = []
    for k in range(1, n + 1):
        w = lw + 2.0 * reach * (n - k + 1) / n
        a = inten * (k / n) ** 1.5
        out.append((w, (c[0], c[1], c[2], int(round(255 * a)))))
    return out


def marker_halo(st, mode, size_px, base_rgb, emphasis=None, lut=None, override=None):
    """Scatter markers: soft larger symbols underneath [(size_px, rgba)], outermost first."""
    return [(size_px + w - 2.0, c) for w, c in strokes(st, mode, base_rgb, 2.0, emphasis, lut, None, override)]


def auto_limit(n_items, st):
    """True when there are too many glowing objects: glow only active / selected (status-bar note)."""
    return int(n_items) > int(st["threshold"])


def limit_note(n_items, st):
    return (f"Glow limited to the active / selected objects ({n_items:,} objects > {st['threshold']:,})"
            if auto_limit(n_items, st) else "")


def decimate_xy(x, y, max_points=4000):
    """Underlay geometry while dragging: every k-th vertex (NaN breaks kept), full resolution on release."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n <= max_points:
        return x, y
    k = int(np.ceil(n / max_points))
    idx = np.arange(0, n, k)
    gaps = np.flatnonzero(~np.isfinite(x) | ~np.isfinite(y))
    if len(gaps):
        idx = np.union1d(idx, gaps)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return x[idx], y[idx]


# ====================================================================== map bus
class GlowState:
    """Selection / active / hover emphasis shared through the map bus (PLAN 39f).

    Watches selection kinds horizons / faults / wells / lines and the active horizon / fault / well / inline /
    crossline / line.  `emphasis(kind, name)` answers what a window should draw; `listeners` are called on any
    change (windows redraw their underlays).  Settings changes go out as a display event (kind 'glow')."""
    SEL = {"horizon": "horizons", "fault": "faults", "well": "wells", "line": "lines", "polygon": "polygons"}
    ACT = {"horizon": "horizon", "fault": "fault", "well": "well", "line": "line"}

    def __init__(self, bus=None, settings=None, name="glow"):
        self.bus = bus
        self.settings = settings or GlowSettings()
        self.name = name
        self.hover = None  # (kind, name)
        self.qc = set()  # {(kind, name)} QC flags (mistie hotspots, wells that do not tie, blend seam, grid problems)
        self.tracking = set()  # horizons whose autotracker is running
        self.listeners = []
        self.events = 0
        if bus is not None:
            bus.subscribe(name, self._on_event)

    def _changed(self):
        self.events += 1
        for fn in list(self.listeners):
            fn()

    def _on_event(self, ev):
        t = ev.get("topic")
        if t in ("selection", "active"):
            self._changed()
        elif t == "display" and ev.get("kind") == "glow" and ev.get("settings"):
            s = GlowSettings.from_dict(ev["settings"])
            self.settings.d.update(s.d)
            self.settings.overrides = s.overrides
            self._changed()
        elif t == "display" and ev.get("kind") == "hover":
            h = ev.get("object")
            self.hover = tuple(h) if h else None
            self._changed()

    def emphasis(self, kind, name):
        nm = str(name)
        if (kind, nm) in self.qc:
            return "qc"
        if kind == "horizon" and nm in self.tracking:
            return "tracking"
        if self.bus is not None:
            ak = self.ACT.get(kind)
            if ak is not None and self.bus.active.get(ak) is not None and str(self.bus.active.get(ak)) == nm:
                return "active"
            sk = self.SEL.get(kind)
            if sk is not None:
                from .mapbus import norm_id
                if norm_id(nm) in {norm_id(i) for i in self.bus.selection.get(sk)}:
                    return "selected"
        if self.hover == (kind, nm):
            return "hover"
        return None

    def set_hover(self, kind=None, name=None, source=None):
        h = (kind, str(name)) if kind else None
        if h == self.hover:
            return False
        self.hover = h
        if self.bus is not None:
            self.bus.publish("display", source or self.name, kind="hover", object=list(h) if h else None, local=True)
        self._changed()
        return True

    def set_qc(self, kind, name, on=True):
        (self.qc.add if on else self.qc.discard)((kind, str(name)))
        self._changed()

    def set_tracking(self, name, on=True):
        (self.tracking.add if on else self.tracking.discard)(str(name))
        self._changed()

    def broadcast(self, source=None):
        """Settings changed in this window: tell the others (and the other tools through the FileBridge)."""
        source = source or self.name
        if self.bus is not None:
            self.bus.publish("display", source, kind="glow", settings=self.settings.to_dict())
        self._changed()

    def close(self):
        if self.bus is not None:
            self.bus.unsubscribe(self.name)


# ====================================================================== exports (matplotlib)
def mpl_path_effects(st, mode, base_rgb, line_width_pt=1.5, emphasis=None, lut=None, dpi=100.0, override=None):
    """matplotlib path effects reproducing the on-screen glow / halo (widths in screen pixels -> points)."""
    from matplotlib import patheffects as pe
    px2pt = 72.0 / float(dpi)
    lw_px = line_width_pt / px2pt
    eff = []
    for w, c in strokes(st, mode, base_rgb, lw_px, emphasis, lut, None, override):
        eff.append(pe.Stroke(linewidth=w * px2pt, foreground=tuple(np.asarray(c[:3]) / 255.0), alpha=c[3] / 255.0,
                             capstyle="round", joinstyle="round"))
    if eff:
        eff.append(pe.Normal())
    return eff


def export_section_png(path, data, xaxis, times, curves, lut=None, clip=99.0, settings=None, with_glow=True,
                       title="", xlabel="Trace", ylabel="TWT (ms)", dpi=100, size=(12.0, 6.5), emphasis=None):
    """Section figure export (PNG / PDF / SVG by extension) with the interpretation drawn as on screen.

    data (n_x, n_t); curves = [{"x", "y", "colour", "width", "kind", "name", "points"?}] (horizons, fault cuts /
    sticks, well paths).  `with_glow=False` = the "Export without glow" checkbox (PLAN 39g).  Returns the number of
    glowing curves drawn (tests)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    st = settings or GlowSettings()
    data = np.asarray(data, float)
    xaxis = np.asarray(xaxis, float)
    times = np.asarray(times, float)
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    fin = np.isfinite(data)
    c = float(np.percentile(np.abs(data[fin]), clip)) if fin.any() else 1.0
    cmap = ListedColormap(np.asarray(lut, float)[:, :3] / 255.0) if lut is not None else "seismic"
    ax.imshow(data.T, aspect="auto", cmap=cmap, vmin=-c, vmax=c, interpolation="bilinear",
              extent=[xaxis[0], xaxis[-1], times[-1], times[0]])
    n_glow = 0
    emph = emphasis or {}
    for cv in curves:
        kind, name = cv.get("kind", "horizon"), cv.get("name")
        e = emph.get((kind, name))
        mode = st.effective(kind, name, e) if with_glow else None
        col = np.asarray(_rgb(cv.get("colour", (255, 255, 0))), float) / 255.0
        lw = float(cv.get("width", 2.0))
        eff = mpl_path_effects(st, mode, _rgb(cv.get("colour")), lw * 0.75, e, lut, dpi,
                               st.override(kind, name)) if mode else []
        n_glow += bool(eff)
        ls = "--" if cv.get("dashed") else "-"
        (ln,) = ax.plot(cv["x"], cv["y"], color=col, lw=lw * 0.75, ls=ls, solid_capstyle="round",
                        marker="o" if cv.get("points") else None, ms=3, label=cv.get("label"))
        if eff:
            ln.set_path_effects(eff)
    ax.set_xlim(xaxis[0], xaxis[-1])
    ax.set_ylim(times[-1], times[0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if any(cv.get("label") for cv in curves):
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return n_glow


# ====================================================================== 3D
def boundary_edges(tri_pos):
    """Outline of a triangle soup (n*3, 3): edges used by exactly one triangle -> (m, 2, 3) segments."""
    P = np.asarray(tri_pos, np.float64).reshape(-1, 3, 3)
    if not len(P):
        return np.zeros((0, 2, 3))
    q = np.round(P.reshape(-1, 3), 6)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1, 3)
    e = np.concatenate([inv[:, [0, 1]], inv[:, [1, 2]], inv[:, [2, 0]]])
    seg = np.concatenate([P[:, [0, 1]], P[:, [1, 2]], P[:, [2, 0]]])
    es = np.sort(e, axis=1)
    _, first, cnt = np.unique(es, axis=0, return_index=True, return_counts=True)
    return seg[first[cnt == 1]]


def vertex_normals(tri_pos):
    P = np.asarray(tri_pos, np.float64).reshape(-1, 3, 3)
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    L = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(L > 0, n / np.where(L > 0, L, 1), 0)
    return np.repeat(n, 3, axis=0)


def shell(tri_pos, offset):
    """Enlarged transparent shell: the surface pushed out by `offset` along the normals on both sides (a horizon
    is open, so both sides glow).  Returns (m*3, 3) triangles."""
    P = np.asarray(tri_pos, np.float64).reshape(-1, 3)
    n = vertex_normals(P)
    return np.concatenate([P + n * offset, P - n * offset])


KIND_3D = {"horizons": "horizon", "faults": "fault_surface", "fault": "fault_surface", "wells": "well",
           "geobodies": "geobody", "lines": "section_line", "line2d": "section_line", "slice": "plane_outline"}


def plane_outline(pos):
    """Rectangle outline (4 segments) of a planar slice mesh from its two principal in-plane axes."""
    P = np.asarray(pos, np.float64).reshape(-1, 3)
    if len(P) < 3:
        return np.zeros((0, 2, 3))
    sub = P[:: max(1, len(P) // 4000)]
    c = sub.mean(axis=0)
    _, _, vt = np.linalg.svd(sub - c, full_matrices=False)
    u, v = vt[0], vt[1]
    pu, pv = (P - c) @ u, (P - c) @ v
    a0, a1, b0, b1 = pu.min(), pu.max(), pv.min(), pv.max()
    cs = [c + a * u + b * v for a, b in ((a0, b0), (a1, b0), (a1, b1), (a0, b1))]
    return np.array([[cs[k], cs[(k + 1) % 4]] for k in range(4)])


def glow_meshes(obj, st, emphasis=None, scale=1.0, lut=None, background=(0.09, 0.10, 0.12)):
    """Underlay meshes for one scene3d.SceneObject (drawn first, depth-write off, by the GL canvas and the software
    renderer).  Lines -> a wide line underlay (meta['width_px']) per stroke; triangles (horizons, fault surfaces,
    geobodies) -> outline glow + a faint offset shell for the active / selected object; tubes (wells) -> a fatter
    transparent tube (same mesh pushed out).  `scale` = world units per screen pixel at the object (shell offset)."""
    from .scene3d import Mesh
    kind = KIND_3D.get(obj.kind, obj.kind)
    mode = st.effective(kind, obj.name, emphasis)
    if mode is None:
        return []
    out = []
    bg255 = tuple(int(255 * v) for v in background[:3])
    for m in obj.meshes:
        if not len(m):
            continue
        base = tuple(int(255 * v) for v in np.nanmean(m.rgba[:, :3], axis=0))
        sts = strokes(st, mode, base, 1.0, emphasis, lut, bg255, st.override(kind, obj.name))
        if not sts:
            continue
        if obj.kind == "slice":  # plane outline only (PLAN 39e: 3D plane outlines)
            segs = plane_outline(m.pos).reshape(-1, 3)
        elif m.kind == "lines":
            segs = m.pos.reshape(-1, 3)
        elif obj.kind in ("well", "wells") or m.meta.get("tube"):
            n = vertex_normals(m.pos)
            P3 = np.asarray(m.pos, np.float64).reshape(-1, 3, 3).mean(axis=1)
            if np.mean(np.sign(np.einsum("ij,ij->i", n[::3], P3 - P3.mean(axis=0)))) < 0:
                n = -n  # tube winding points inward: grow outward
            for w, c in sts:
                pos = m.pos + n * (0.5 * w * scale)
                out.append(Mesh("triangles", pos, np.array(c, np.float32) / 255.0, name=f"glow:{m.name}",
                                meta={"glow": True, "width_px": w}))
            continue
        else:
            edges = boundary_edges(m.pos)
            segs = edges.reshape(-1, 3)
            if emphasis in ("active", "selected", "tracking", "qc") and mode == "glow":
                w, c = sts[-1]
                cc = np.array(c, np.float32) / 255.0
                cc[3] *= 0.25
                out.append(Mesh("triangles", shell(m.pos, 0.5 * w * scale), cc, name=f"glowshell:{m.name}",
                                meta={"glow": True, "width_px": w, "shell": True}))
        if len(segs):
            for w, c in sts:
                out.append(Mesh("lines", segs, np.array(c, np.float32) / 255.0, name=f"glow:{m.name}",
                                meta={"glow": True, "width_px": w}))
    return out
