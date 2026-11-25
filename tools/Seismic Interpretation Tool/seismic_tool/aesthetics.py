"""Aesthetics package (PLAN 40) - display logic shared by every window, pure numpy (the Qt glue is gui/theme.py and
the windows).  Colour tables are `colortables.py` (40b); glow is `glow.py` (item 39).

40a  Sun / hillshade: `Sun` (azimuth, elevation, vertical exaggeration, strength, multi-directional), `slope_aspect`,
     `hillshade`, `shade` (colour fill x shade), `drape` (an attribute map coloured, lit by a structure grid),
     `sun_dial_xy` / `sun_from_dial` (the small dial widget), per-view settings in project.display['sun'],
     `light_vector` (the same sun lights 3D surfaces, scene3d.shade_mesh).
40c  Co-rendering: `corender(base_rgba, attr, lut, mode, opacity)` - alpha, multiply, transparency-by-value.
40d  High DPI: `dpi_scale`, `scaled`, `font_pt` - widths / fonts follow the screen scale (100 / 150 / 200 %).
40e  3D depth cues: `eye_dome` (image-space EDL from a depth raster), `fog` (blend toward the background by depth).
40f  Themes: THEMES dark / light (Qt palette roles, plot colours, stylesheet), user settings file.
40g  Smooth motion: easing functions, `Tween` (non-blocking, stops on user input), camera / range fly-to frames.
40i  Figure templates: page / DPI presets, title block, scale bar, north arrow, legend - `compose_figure`.
40j  Automatic object colours: `distinct_colours` (max-min OKLab distance, avoiding the seismic colour table),
     ICS chronostratigraphic colours by unit name or age, lock / reassign.
"""
import json
import math
import os
import time

import numpy as np

from . import colortables as CT


# ====================================================================== 40a sun + hillshade
class Sun:
    KEY = "sun"

    def __init__(self, azimuth=315.0, elevation=40.0, ve=1.0, strength=0.6, multi=False, on=True):
        self.azimuth, self.elevation, self.ve = float(azimuth) % 360.0, float(np.clip(elevation, 1, 90)), float(ve)
        self.strength, self.multi, self.on = float(np.clip(strength, 0, 1)), bool(multi), bool(on)

    def to_dict(self):
        return {"azimuth": self.azimuth, "elevation": self.elevation, "ve": self.ve, "strength": self.strength,
                "multi": self.multi, "on": self.on}

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(**{k: d[k] for k in ("azimuth", "elevation", "ve", "strength", "multi", "on") if k in d})

    def save(self, project, view):
        disp = getattr(project, "display", None)
        if disp is None:
            project.display = disp = {}
        disp.setdefault(self.KEY, {})[str(view)] = self.to_dict()
        return disp[self.KEY][str(view)]

    @classmethod
    def load(cls, project, view):
        d = ((getattr(project, "display", None) or {}).get(cls.KEY) or {}).get(str(view)) if project else None
        return cls.from_dict(d)


def slope_aspect(z, dx=1.0, dy=1.0, ve=1.0):
    """Slope (rad) and aspect (rad, direction the surface faces = downhill, clockwise from north / +rows)."""
    z = np.asarray(z, float) * float(ve)
    fin = np.isfinite(z)
    zf = np.where(fin, z, np.nanmean(z) if fin.any() else 0.0)
    if zf.shape[0] < 2 or zf.shape[1] < 2:
        return np.zeros(z.shape), np.zeros(z.shape)
    gy, gx = np.gradient(zf, abs(dy), abs(dx))
    if dy < 0:
        gy = -gy
    if dx < 0:
        gx = -gx
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.mod(np.arctan2(-gx, -gy), 2 * np.pi)
    return np.where(fin, slope, np.nan), np.where(fin, aspect, np.nan)


def hillshade(z, dx=1.0, dy=1.0, sun=None, positive_down=False):
    """0..1 shade: cos(zenith) cos(slope) + sin(zenith) sin(slope) cos(azimuth - aspect) (the ESRI formula);
    multi-directional = mean of 4 suns 45 deg apart around the chosen azimuth (softer, no blind direction)."""
    sun = sun or Sun()
    slope, aspect = slope_aspect(-np.asarray(z, float) if positive_down else z, dx, dy, sun.ve)
    alt = math.radians(sun.elevation)
    azs = [sun.azimuth + d for d in ((-67.5, -22.5, 22.5, 67.5) if sun.multi else (0.0,))]
    acc = np.zeros(np.shape(slope))
    for az in azs:
        a = math.radians(az)
        acc += math.sin(alt) * np.cos(slope) + math.cos(alt) * np.sin(slope) * np.cos(a - aspect)
    return np.clip(acc / len(azs), 0, 1)


def shade(rgba, sh, strength=0.6):
    """rgb x ((1 - s) + s * shade / flat), where flat = the shade of a flat surface (so flat areas keep their colour)."""
    out = np.asarray(rgba, float).copy()
    f = (1 - strength) + strength * np.nan_to_num(np.asarray(sh, float), nan=1.0) / 0.75
    out[..., :3] *= f[..., None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def shaded_grid(z, lut, dx=1.0, dy=1.0, sun=None, vmin=None, vmax=None, values=None, positive_down=False):
    """Colour fill of `values` (default z) through the lut, lit by the hillshade of z (40a draped attribute map).
    positive_down=True for time / depth maps (structural highs = small values are lit like hills)."""
    from .blend import lut_rgba
    sun = sun or Sun()
    v = z if values is None else values
    rgba = lut_rgba(v, lut, vmin, vmax, clip=(0.0, 100.0) if vmin is None else (1.0, 99.0))
    if not sun.on:
        return rgba
    flat = math.sin(math.radians(sun.elevation))
    sh = hillshade(z, dx, dy, sun, positive_down) / max(1e-6, flat) * 0.75
    out = shade(rgba, sh, sun.strength)
    out[..., 3] = rgba[..., 3]
    return out


drape = shaded_grid


def sun_dial_xy(sun, radius=1.0):
    """Dial widget: the sun as a point; azimuth = angle clockwise from north (up), elevation 90 = centre."""
    r = radius * (1.0 - sun.elevation / 90.0)
    a = math.radians(sun.azimuth)
    return r * math.sin(a), r * math.cos(a)


def sun_from_dial(x, y, radius=1.0, sun=None):
    """Inverse of sun_dial_xy (a click / drag on the dial)."""
    s = Sun(**(sun.to_dict() if sun else {}))
    r = min(1.0, math.hypot(x, y) / radius)
    s.azimuth = math.degrees(math.atan2(x, y)) % 360.0
    s.elevation = float(np.clip(90.0 * (1.0 - r), 1.0, 90.0))
    return s


def light_vector(sun):
    """Unit vector toward the sun in (east, north, up) - drives 3D surface lighting."""
    a, e = math.radians(sun.azimuth), math.radians(sun.elevation)
    return np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)])


# ====================================================================== 40c co-rendering
CORENDER_MODES = ("alpha", "multiply", "transparency")


def corender(base_rgba, attr, lut, mode="alpha", opacity=0.5, vmin=None, vmax=None, clip=(1.0, 99.0),
             invert_alpha=False):
    """Second volume / attribute over the amplitude image.  Each layer keeps its own colour table and clipping.

    alpha        out = base (1 - a) + top a,            a = opacity
    multiply     out = base * lerp(1, top / 255, a)     (dark attribute values darken: coherence / edges)
    transparency out = base (1 - a v) + top a v,        v = normalised attribute 0..1 (1 - v with invert_alpha):
                 low values transparent, high values opaque (transparency-by-value)."""
    from .blend import lut_rgba, normalize
    if mode not in CORENDER_MODES:
        raise ValueError(f"co-render mode must be one of {CORENDER_MODES}")
    base = np.asarray(base_rgba, float)
    top = lut_rgba(attr, lut, vmin, vmax, clip).astype(float)
    valid = top[..., 3:4] / 255.0
    a = float(np.clip(opacity, 0, 1))
    out = base.copy()
    if mode == "alpha":
        w = a * valid
        out[..., :3] = base[..., :3] * (1 - w) + top[..., :3] * w
    elif mode == "multiply":
        w = a * valid
        out[..., :3] = base[..., :3] * (1 - w + w * top[..., :3] / 255.0)
    else:
        v = np.nan_to_num(normalize(attr, clip, 1.0, vmin, vmax), nan=0.0)[..., None]
        if invert_alpha:
            v = 1.0 - v
        w = a * v * valid
        out[..., :3] = base[..., :3] * (1 - w) + top[..., :3] * w
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# ====================================================================== 40d DPI
def dpi_scale(logical_dpi=96.0, device_pixel_ratio=1.0):
    """Screen scale factor: Windows 'Scale and layout' 100 / 150 / 200 % -> 1.0 / 1.5 / 2.0."""
    return max(1.0, float(logical_dpi) / 96.0) * max(1.0, float(device_pixel_ratio))


def scaled(px, scale):
    """Line widths / marker sizes in device pixels (at least 1)."""
    return max(1.0, float(px) * float(scale))


def font_pt(base_pt, scale, logical=True):
    """Qt scales fonts itself on a high-DPI aware app (logical=True): keep points; pixel fonts scale."""
    return float(base_pt) if logical else float(base_pt) * float(scale)


def qt_env(env=None):
    """Environment for per-monitor DPI on Windows before QApplication exists (PySide6 is DPI aware by default;
    PassThrough keeps 150 % sharp instead of rounding to 200 %)."""
    env = os.environ if env is None else env
    env.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    env.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")
    return env


# ====================================================================== 40e 3D depth cues
def eye_dome(depth, strength=1.0, radius=1, background=np.inf):
    """Eye-dome lighting (Boucheny 2009) on a depth raster: shade = exp(-strength * sum(max(0, log z - log z_n)))
    over 4 neighbours at `radius` px.  Background pixels (inf / NaN) -> 1.  Returns 0..1 (multiply the image)."""
    d = np.asarray(depth, float)
    bad = ~np.isfinite(d) | (d == background)
    lz = np.log(np.where(bad, np.nan, np.maximum(d, 1e-9)))
    ref = np.nanmax(lz) if np.isfinite(lz).any() else 0.0
    lz = np.where(bad, ref + 10.0, lz)  # background counts as far away -> silhouettes get dark edges
    acc = np.zeros(d.shape)
    for dy, dx in ((radius, 0), (-radius, 0), (0, radius), (0, -radius)):
        n = np.roll(np.roll(lz, dy, 0), dx, 1)
        acc += np.maximum(0.0, lz - n)
    return np.exp(-strength * 300.0 * acc / 4.0)  # background beside an object darkens too: the silhouette line


def fog(rgb, depth, background, density=0.5, near=None, far=None):
    """Distance fog: rgb -> background with weight 1 - exp(-density * (z - near) / (far - near))."""
    d = np.asarray(depth, float)
    fin = np.isfinite(d)
    if not fin.any():
        return np.asarray(rgb)
    near = float(np.nanmin(d[fin])) if near is None else near
    far = float(np.nanmax(d[fin])) if far is None else far
    t = np.clip((np.where(fin, d, far) - near) / max(1e-9, far - near), 0, 1)
    w = (1.0 - np.exp(-float(density) * 3.0 * t))[..., None]
    bg = np.asarray(background, float)[:3]
    if bg.max() <= 1.0 and np.asarray(rgb).max() > 1.0:
        bg = bg * 255.0
    out = np.asarray(rgb, float)[..., :3] * (1 - w) + bg * w
    return out


# ====================================================================== 40f themes
THEMES = {
    "dark": {"window": "#1e2127", "base": "#15181d", "alt": "#23272e", "text": "#e6e6e6", "button": "#2b3038",
             "highlight": "#2f7dd1", "highlight_text": "#ffffff", "mid": "#3a404a", "disabled": "#7d838c",
             "plot_bg": "#15181d", "plot_fg": "#cfd3da", "grid": (200, 205, 215, 50), "legend_bg": (30, 33, 39, 220),
             "section_bg": "#15181d"},
    "light": {"window": "#f3f3f3", "base": "#ffffff", "alt": "#f7f7f7", "text": "#1b1b1b", "button": "#e8e8e8",
              "highlight": "#2f6fc4", "highlight_text": "#ffffff", "mid": "#c8c8c8", "disabled": "#8a8a8a",
              "plot_bg": "#ffffff", "plot_fg": "#202020", "grid": (40, 40, 40, 50), "legend_bg": (255, 255, 255, 230),
              "section_bg": "#ffffff"},
}
DEFAULT_THEME = "dark"


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))


def stylesheet(name):
    t = THEMES.get(name, THEMES[DEFAULT_THEME])
    return (f"QWidget {{ background-color: {t['window']}; color: {t['text']}; }}\n"
            f"QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QTreeWidget, QTableWidget, QTextEdit, "
            f"QPlainTextEdit {{ background-color: {t['base']}; color: {t['text']}; border: 1px solid {t['mid']}; }}\n"
            f"QPushButton, QToolButton {{ background-color: {t['button']}; color: {t['text']}; "
            f"border: 1px solid {t['mid']}; padding: 3px 8px; }}\n"
            f"QPushButton:checked, QToolButton:checked {{ background-color: {t['highlight']}; "
            f"color: {t['highlight_text']}; }}\n"
            f"QMenuBar, QMenu, QToolBar, QStatusBar, QDockWidget {{ background-color: {t['window']}; "
            f"color: {t['text']}; }}\n"
            f"QMenu::item:selected, QListWidget::item:selected, QTreeWidget::item:selected "
            f"{{ background-color: {t['highlight']}; color: {t['highlight_text']}; }}\n"
            f"QHeaderView::section {{ background-color: {t['alt']}; color: {t['text']}; }}\n"
            f"QWidget:disabled {{ color: {t['disabled']}; }}\n")


def plot_colours(name):
    t = THEMES.get(name, THEMES[DEFAULT_THEME])
    return {"background": t["plot_bg"], "foreground": t["plot_fg"], "grid": t["grid"], "legend": t["legend_bg"],
            "section_background": t["section_bg"]}


def settings_path():
    return os.path.join(os.path.expanduser("~"), ".geoharmonic", "settings.json")


def load_user_settings(path=None):
    path = path or settings_path()
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_user_settings(values, path=None):
    path = path or settings_path()
    d = load_user_settings(path)
    d.update(values)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, path)
    return d


# ====================================================================== 40g smooth motion
def ease_in_out(t):
    t = float(np.clip(t, 0, 1))
    return 3 * t * t - 2 * t * t * t


def ease_out(t):
    t = float(np.clip(t, 0, 1))
    return 1 - (1 - t) ** 3


SPEEDS = {"Off": 0.0, "Fast": 0.25, "Normal": 0.5, "Slow": 0.9}  # seconds


class Tween:
    """Non-blocking animation: `step(now)` returns the eased values for this frame (call it from a QTimer) and
    finishes by itself; `interrupt()` (any mouse / key input) stops it where it is.  Duration 0 (speed Off) jumps."""

    def __init__(self, start, end, duration=0.5, ease=ease_in_out, clock=time.time, on_frame=None):
        self.a, self.b = np.asarray(start, float), np.asarray(end, float)
        self.duration = float(duration)
        self.ease, self.clock, self.on_frame = ease, clock, on_frame
        self.t0 = clock()
        self.done = self.duration <= 0
        self.frames = 0
        if self.done and on_frame:
            on_frame(self.b.copy())

    def value(self, now=None):
        if self.done:
            return self.b.copy()
        t = (self.clock() if now is None else now) - self.t0
        return self.a + (self.b - self.a) * self.ease(t / self.duration)

    def step(self, now=None):
        if self.done:
            return None
        now = self.clock() if now is None else now
        v = self.value(now)
        self.frames += 1
        if now - self.t0 >= self.duration:
            self.done = True
            v = self.b.copy()
        if self.on_frame:
            self.on_frame(v)
        return v

    def interrupt(self):
        self.done = True


def camera_vector(cam):
    return [cam.target[0], cam.target[1], cam.target[2], cam.distance, cam.azimuth, cam.elevation]


def apply_camera_vector(cam, v):
    cam.target = np.asarray(v[:3], float)
    cam.distance, cam.azimuth, cam.elevation = float(v[3]), float(v[4]) % 360.0, float(v[5])
    return cam


def fly_to_camera(cam, lo, hi, duration=0.5, clock=time.time):
    """Tween from the current camera to one framing the box lo..hi (same view direction, shortest azimuth turn)."""
    from .scene3d import Camera
    tgt = Camera(cam.target, cam.distance, cam.azimuth, cam.elevation, cam.fov)
    tgt.fit(np.asarray(lo, float), np.asarray(hi, float))
    a = camera_vector(cam)
    b = camera_vector(tgt)
    b[4] = a[4] + ((b[4] - a[4] + 180.0) % 360.0 - 180.0)
    return Tween(a, b, duration, clock=clock, on_frame=lambda v: apply_camera_vector(cam, v))


def zoom_range(view, box, pad=0.1):
    """(x0, x1, y0, y1) target range around box with padding, same aspect as the current view."""
    x0, x1, y0, y1 = [float(v) for v in box]
    w, h = max(1e-9, x1 - x0) * (1 + 2 * pad), max(1e-9, y1 - y0) * (1 + 2 * pad)
    vw, vh = view[1] - view[0], view[3] - view[2]
    asp = vw / vh if vh else 1.0
    if w / h < asp:
        w = h * asp
    else:
        h = w / asp
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2


# ====================================================================== 40i figure templates
PAGES = {"16:9 slide": (13.333, 7.5), "Letter landscape": (11.0, 8.5), "Letter portrait": (8.5, 11.0),
         "A4 landscape": (11.69, 8.27), "A4 portrait": (8.27, 11.69), "Poster 36x24": (36.0, 24.0)}
DPI_PRESETS = {"Screen (100)": 100, "Slide (150)": 150, "Print (300)": 300}
FIG_KEY = "figure_templates"


def default_template():
    return {"page": "16:9 slide", "dpi": 150, "title": "", "subtitle": "", "author": "", "project": "",
            "date": time.strftime("%Y-%m-%d"), "scale_bar": True, "north_arrow": False, "legend": True,
            "grid": False, "theme": "light", "glow": True}


def save_template(project, name, tpl):
    disp = getattr(project, "display", None)
    if disp is None:
        project.display = disp = {}
    disp.setdefault(FIG_KEY, {})[str(name)] = dict(tpl)
    return disp[FIG_KEY][str(name)]


def load_template(project, name):
    d = ((getattr(project, "display", None) or {}).get(FIG_KEY) or {}).get(str(name))
    t = default_template()
    t.update(d or {})
    return t


def nice_length(width, fraction=0.2):
    """Scale-bar length: 1 / 2 / 5 x 10^n closest below fraction x width."""
    target = max(1e-12, float(width) * fraction)
    p = 10 ** math.floor(math.log10(target))
    for m in (5, 2, 1):
        if m * p <= target:
            return m * p
    return p


def compose_figure(path, image=None, draw=None, template=None, legend=None, colourbar=None, units="m",
                   x_extent=None, north_deg=None, kind="section"):
    """One-click figure: any view as PNG / PDF / SVG on a page preset with a title block (project, line / well,
    date, author), scale bar (x_extent = map / section width in `units`), north arrow (maps), colour bar / legend.

    image: an RGB(A) array or a PNG path (e.g. a 3D screenshot / correlation export); or draw(ax) draws into the
    main axes (sections / maps).  colourbar = (lut, vmin, vmax, label).  legend = [(label, rgb)].
    Returns {path, boxes} (axes positions in figure fractions, used by tests)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import FancyArrow, Rectangle
    t = default_template()
    t.update(template or {})
    w, h = PAGES.get(t["page"], PAGES["16:9 slide"])
    th = THEMES["light" if t.get("theme") != "dark" else "dark"]
    bg, fg = th["plot_bg"], th["plot_fg"]
    fig = plt.figure(figsize=(w, h), dpi=int(t["dpi"]))
    fig.patch.set_facecolor(bg)
    tb_h = 0.11
    side = 0.16 if (legend or colourbar) else 0.0
    main = fig.add_axes([0.06, tb_h + 0.08, 0.92 - side - 0.04, 0.88 - tb_h - 0.08])
    main.set_facecolor(bg)
    for sp in main.spines.values():
        sp.set_color(fg)
    main.tick_params(colors=fg)
    if image is not None:
        img = plt.imread(image) if isinstance(image, str) else np.asarray(image)
        main.imshow(img)
        main.set_axis_off()
    elif draw is not None:
        draw(main)
        main.xaxis.label.set_color(fg)
        main.yaxis.label.set_color(fg)
    if t.get("grid"):
        main.grid(True, alpha=0.3, color=fg)
    boxes = {"main": main.get_position().bounds}
    # title block
    tb = fig.add_axes([0.02, 0.015, 0.96, tb_h - 0.02])
    tb.set_axis_off()
    tb.add_patch(Rectangle((0, 0), 1, 1, transform=tb.transAxes, fill=False, ec=fg, lw=1.0))
    tb.text(0.01, 0.62, t.get("title") or "", fontsize=14, weight="bold", color=fg, transform=tb.transAxes)
    tb.text(0.01, 0.2, t.get("subtitle") or "", fontsize=10, color=fg, transform=tb.transAxes)
    info = "   |   ".join(s for s in (t.get("project"), t.get("author"), t.get("date"), "GeoHarmonic") if s)
    tb.text(0.99, 0.2, info, fontsize=9, color=fg, ha="right", transform=tb.transAxes)
    boxes["title_block"] = tb.get_position().bounds
    # scale bar
    if t.get("scale_bar") and x_extent:
        L = nice_length(x_extent)
        frac = L / float(x_extent) * boxes["main"][2]
        sb = fig.add_axes([boxes["main"][0], tb_h + 0.015, max(frac, 0.02), 0.03])
        sb.set_axis_off()
        sb.add_patch(Rectangle((0, 0.55), 0.5, 0.35, color=fg, transform=sb.transAxes))
        sb.add_patch(Rectangle((0.5, 0.55), 0.5, 0.35, fill=False, ec=fg, transform=sb.transAxes))
        lab = f"{L / 1000:g} km" if units == "m" and L >= 1000 else f"{L:g} {units}"
        sb.text(1.02, 0.5, lab, color=fg, fontsize=9, va="center", transform=sb.transAxes)
        boxes["scale_bar"] = (sb.get_position().bounds, L)
    if t.get("north_arrow") and north_deg is not None:
        na = fig.add_axes([0.9 - side, 0.78, 0.05, 0.1])
        na.set_axis_off()
        na.set_xlim(-1, 1)
        na.set_ylim(-1, 1)
        a = math.radians(north_deg)
        na.add_patch(FancyArrow(0, -0.6, 1.1 * math.sin(a), 1.1 * math.cos(a), width=0.12, color=fg))
        na.text(0, 0.85, "N", ha="center", color=fg, fontsize=11, weight="bold")
        boxes["north_arrow"] = na.get_position().bounds
    y = 0.85
    if colourbar is not None and t.get("legend", True):
        lut, vmin, vmax, lab = colourbar
        cb = fig.add_axes([1.0 - side + 0.01, tb_h + 0.12, 0.022, 0.6])
        grad = np.linspace(0, 1, 256)[:, None]
        cb.imshow(grad, aspect="auto", origin="lower", cmap=ListedColormap(np.asarray(lut)[:, :3] / 255.0),
                  extent=[0, 1, vmin, vmax])
        cb.set_xticks([])
        cb.yaxis.tick_right()
        cb.tick_params(colors=fg, labelsize=8)
        cb.set_title(lab, color=fg, fontsize=8)
        boxes["colourbar"] = cb.get_position().bounds
    if legend and t.get("legend", True):
        lg = fig.add_axes([1.0 - side + 0.06, tb_h + 0.12, side - 0.07, 0.7])
        lg.set_axis_off()
        for k, (lab, rgb) in enumerate(legend):
            yy = 0.97 - 0.06 * k
            lg.add_patch(Rectangle((0.0, yy - 0.015), 0.18, 0.03, color=np.asarray(rgb[:3], float) / 255.0,
                                   transform=lg.transAxes))
            lg.text(0.24, yy, str(lab), va="center", color=fg, fontsize=9, transform=lg.transAxes)
        boxes["legend"] = lg.get_position().bounds
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"path": path, "boxes": boxes, "page": (w, h), "dpi": int(t["dpi"]), "kind": kind, "y": y}


# ====================================================================== 40j automatic colours
def _candidates(n=12):
    """A dense grid of displayable colours (sRGB cube), the pool distinct colours are drawn from."""
    v = np.linspace(0, 255, n)
    r, g, b = np.meshgrid(v, v, v, indexing="ij")
    return np.stack([r.ravel(), g.ravel(), b.ravel()], 1)


def distinct_colours(k, existing=(), avoid_lut=None, min_l=0.35, max_l=0.92, seed_colour=None):
    """k colours maximally distinct from each other, from `existing` object colours and from the colours of the
    current seismic colour table (sampled) - greedy max-min distance in OKLab.  Very dark / very light colours are
    excluded (they disappear on seismic and on white).  Returns [(r, g, b)] ints."""
    pool = _candidates()
    lab = CT.rgb_to_oklab(pool)
    keep = (lab[:, 0] >= min_l) & (lab[:, 0] <= max_l) & (np.hypot(lab[:, 1], lab[:, 2]) > 0.08)
    pool, lab = pool[keep], lab[keep]
    ref = [CT.rgb_to_oklab(np.asarray(c, float)[None, :3])[0] for c in existing]
    if avoid_lut is not None:
        L = np.asarray(avoid_lut, float)[::16, :3]
        ref += list(CT.rgb_to_oklab(L))
    ref = np.array(ref) if ref else np.zeros((0, 3))
    if len(ref):
        dmin = np.min(np.linalg.norm(lab[:, None, :] - ref[None, :, :], axis=2), axis=1)
    else:
        dmin = np.full(len(lab), np.inf)
    out = []
    first = None if seed_colour is None else CT.rgb_to_oklab(np.asarray(seed_colour, float)[None])[0]
    for _ in range(int(k)):
        if not np.isfinite(dmin).any():
            # nothing to stay away from yet: start from the seed or the most saturated candidate
            j = int(np.argmin(np.linalg.norm(lab - first, axis=1))) if first is not None else \
                int(np.argmax(np.hypot(lab[:, 1], lab[:, 2])))
        else:
            j = int(np.argmax(dmin))
        out.append(tuple(int(v) for v in pool[j]))
        dmin = np.minimum(dmin, np.linalg.norm(lab - lab[j], axis=1))
    return out


def min_difference(colours):
    c = np.asarray(colours, float)
    if len(c) < 2:
        return np.inf
    d = CT.delta_e(c[:, None, :], c[None, :, :])
    d[np.eye(len(c), dtype=bool)] = np.inf
    return float(d.min())


# ICS International Chronostratigraphic Chart colours (RGB) with base ages (Ma) for the units commonly mapped
ICS = [  # name, base age Ma, top age Ma, rgb
    ("Holocene", 0.0117, 0.0, (254, 242, 236)), ("Pleistocene", 2.58, 0.0117, (255, 242, 174)),
    ("Pliocene", 5.333, 2.58, (255, 255, 153)), ("Miocene", 23.03, 5.333, (255, 255, 0)),
    ("Oligocene", 33.9, 23.03, (254, 192, 122)), ("Eocene", 56.0, 33.9, (253, 180, 108)),
    ("Paleocene", 66.0, 56.0, (253, 167, 95)), ("Upper Cretaceous", 100.5, 66.0, (166, 216, 74)),
    ("Lower Cretaceous", 145.0, 100.5, (140, 205, 87)), ("Upper Jurassic", 161.5, 145.0, (179, 227, 238)),
    ("Middle Jurassic", 174.7, 161.5, (128, 207, 216)), ("Lower Jurassic", 201.4, 174.7, (66, 174, 208)),
    ("Triassic", 251.902, 201.4, (129, 43, 146)), ("Permian", 298.9, 251.902, (240, 64, 40)),
    ("Carboniferous", 358.9, 298.9, (103, 165, 153)), ("Devonian", 419.2, 358.9, (203, 140, 55)),
    ("Silurian", 443.8, 419.2, (179, 225, 182)), ("Ordovician", 485.4, 443.8, (0, 146, 112)),
    ("Cambrian", 538.8, 485.4, (127, 160, 86)),
    # periods / sub-eras (names only - matched by name)
    ("Quaternary", 2.58, 0.0, (249, 249, 127)), ("Neogene", 23.03, 2.58, (255, 230, 25)),
    ("Paleogene", 66.0, 23.03, (253, 154, 82)), ("Cretaceous", 145.0, 66.0, (127, 198, 78)),
    ("Jurassic", 201.4, 145.0, (52, 178, 201)),
]


def ics_colour(unit=None, age_ma=None):
    """ICS colour for a unit name (case-insensitive, 'Late'/'Early' accepted for Upper / Lower) or an age in Ma
    (finest unit containing it).  None when unknown."""
    if unit:
        u = str(unit).lower().replace("late ", "upper ").replace("early ", "lower ").strip()
        for name, _, _, rgb in ICS:
            if name.lower() == u:
                return rgb
        for name, _, _, rgb in ICS:
            if name.lower() in u:
                return rgb
    if age_ma is not None:
        a = float(age_ma)
        hits = [(base - top, rgb) for name, base, top, rgb in ICS if top <= a < base]
        if hits:
            return min(hits)[1]
    return None


class ColourAssigner:
    """Automatic colours for new horizons / faults (40j): distinct, avoiding the seismic colour table, ICS colours
    when an age / unit is known; locked colours are never changed; `reassign` recolours the unlocked ones.
    State in project.display['object_colours'] = {"kind:name": {"rgb", "locked"}}."""
    KEY = "object_colours"

    def __init__(self, project=None):
        self.project = project

    def _d(self):
        disp = getattr(self.project, "display", None) if self.project is not None else None
        if disp is None:
            return {}
        return disp.setdefault(self.KEY, {})

    def colour_for(self, kind, name, existing=(), lut=None, unit=None, age_ma=None, use_ics=True):
        rec = self._d().get(f"{kind}:{name}")
        if rec and rec.get("locked"):
            return tuple(rec["rgb"])
        rgb = ics_colour(unit, age_ma) if use_ics and (unit or age_ma is not None) else None
        if rgb is None:
            rgb = distinct_colours(1, existing, lut)[0]
        self._d()[f"{kind}:{name}"] = {"rgb": list(rgb), "locked": bool(rec and rec.get("locked"))}
        return tuple(rgb)

    def lock(self, kind, name, rgb=None, locked=True):
        rec = self._d().setdefault(f"{kind}:{name}", {"rgb": list(rgb or (255, 255, 0)), "locked": False})
        if rgb is not None:
            rec["rgb"] = list(rgb)
        rec["locked"] = bool(locked)
        return rec

    def is_locked(self, kind, name):
        return bool((self._d().get(f"{kind}:{name}") or {}).get("locked"))

    def reassign(self, items, kind, lut=None):
        """items = [(name, current rgb)]: recolour every unlocked one, keeping locked ones fixed (and avoided)."""
        locked = [tuple(c) for n, c in items if self.is_locked(kind, n)]
        free = [n for n, c in items if not self.is_locked(kind, n)]
        cols = distinct_colours(len(free), locked, lut)
        out = {n: tuple(c) for n, c in items if self.is_locked(kind, n)}
        for n, c in zip(free, cols):
            out[n] = c
            self._d()[f"{kind}:{n}"] = {"rgb": list(c), "locked": False}
        return out


def apply_sun(stack, sun, lids=None):
    """40a: light every grid / horizon layer of a map LayerStack with one sun (or switch shading off).  Single
    direction -> the bump-map path (azimuth set); multi-directional -> the soft multi-sun hillshade."""
    n = 0
    for lay in list(getattr(stack, "layers", [])):
        if lay.get("kind") not in ("grid", "horizon") or (lids is not None and lay["id"] not in lids):
            continue
        stack.set_style(lay["id"], hillshade=bool(sun.on), azimuth=None if sun.multi else float(sun.azimuth),
                        elevation=float(sun.elevation), ve=float(sun.ve), shade_strength=float(sun.strength))
        n += 1
    return n
