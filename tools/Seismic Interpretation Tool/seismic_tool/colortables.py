"""Colour-table library (PLAN 40b) - one library for the seismic viewer, maps, 3D and the correlation tool.

Pure numpy.  Tables are control points (position 0..1, RGB 0-255); perceptual tables are generated here in OKLab with
strictly monotonic lightness (viridis / cividis / magma-style - generated, not copied), colourblind-safe tables are
flagged.  A `ColourTable` carries display options (flip, gamma, clip %, symmetric clip) and can be edited (drag control
points), saved to the project (project.display['colour_tables']), exported / imported as JSON or CSV.

    t = get("Kingdom-style")                       # library or project table
    lut = t.lut()                                  # (256, 4) uint8
    lo, hi = t.levels(data)                        # clip / symmetric clip
    t.move_point(2, 0.42); t.add_point(0.6, (255, 0, 0)); t.remove_point(3)
    save_custom(project, t); export_json(t, path); t2 = import_file(path)
    L = lightness(lut)                             # OKLab L per entry (monotonic check)
"""
import csv
import json
import os

import numpy as np

# ====================================================================== colour science (OKLab, Ottosson 2020)
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def rgb_to_oklab(rgb):
    """RGB 0-255 (..., 3) -> OKLab (..., 3) with L in 0..1."""
    lin = srgb_to_linear(np.asarray(rgb, float)[..., :3] / 255.0)
    lms = np.cbrt(lin @ _M1.T)
    return lms @ _M2.T


def oklab_to_rgb(lab):
    lab = np.asarray(lab, float)
    lms = lab @ np.linalg.inv(_M2).T
    lin = (lms ** 3) @ np.linalg.inv(_M1).T
    return np.clip(np.round(linear_to_srgb(lin) * 255.0), 0, 255)


def lightness(lut):
    return rgb_to_oklab(np.asarray(lut)[:, :3])[:, 0]


def delta_e(a, b):
    """Colour difference in OKLab (x100 ~ CIEDE-like scale, 2-3 = just noticeable)."""
    return 100.0 * np.linalg.norm(rgb_to_oklab(a) - rgb_to_oklab(b), axis=-1)


def simulate_cvd(rgb, kind="deutan"):
    """Rough dichromat simulation (Vienot 1999 style, linear RGB) for colourblind checks."""
    m = {"protan": np.array([[0.11238, 0.88762, 0.0], [0.11238, 0.88762, 0.0], [0.00401, -0.00401, 1.0]]),
         "deutan": np.array([[0.29275, 0.70725, 0.0], [0.29275, 0.70725, 0.0], [-0.02234, 0.02234, 1.0]])}[kind]
    lin = srgb_to_linear(np.asarray(rgb, float)[..., :3] / 255.0)
    return np.clip(np.round(linear_to_srgb(lin @ m.T) * 255), 0, 255)


# ====================================================================== table definitions
def _oklab_ramp(stops_rgb, n=9):
    """Control points evenly spaced through OKLab between anchor colours (perceptually even steps)."""
    lab = rgb_to_oklab(np.asarray(stops_rgb, float))
    t = np.linspace(0, 1, len(lab))
    x = np.linspace(0, 1, n)
    pts = np.stack([np.interp(x, t, lab[:, k]) for k in range(3)], axis=1)
    return [(float(p), tuple(int(v) for v in oklab_to_rgb(c))) for p, c in zip(x, pts)]


def _perceptual(anchors, n=17):
    """Perceptually uniform ramp generated in OKLab: hue / chroma path through the anchor colours, lightness
    replaced by a straight line from the first to the last anchor (strictly monotonic - the viridis construction)."""
    lab = rgb_to_oklab(np.asarray(anchors, float))
    t = np.linspace(0, 1, len(lab))
    x = np.linspace(0, 1, n)
    pts = np.stack([np.interp(x, t, lab[:, k]) for k in range(3)], axis=1)
    pts[:, 0] = np.linspace(lab[0, 0], lab[-1, 0], n)
    for _ in range(20):  # pull chroma in until every step is inside sRGB (clipping would bend lightness)
        rgb = oklab_to_rgb(pts)
        back = rgb_to_oklab(rgb)
        bad = np.abs(back[:, 0] - pts[:, 0]) > 0.004
        if not bad.any():
            break
        pts[bad, 1:] *= 0.9
    return [(float(p), tuple(int(v) for v in col)) for p, col in zip(x, oklab_to_rgb(pts))]


SEISMIC = {
    "Seismic (red-white-blue)": [(0.0, (0, 0, 140)), (0.25, (40, 90, 255)), (0.5, (255, 255, 255)),
                                 (0.75, (255, 60, 40)), (1.0, (140, 0, 0))],
    "Grey": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
    "Red-white-black": [(0.0, (190, 0, 0)), (0.5, (255, 255, 255)), (1.0, (0, 0, 0))],
    "Petrel-like": [(0.0, (160, 255, 255)), (0.15, (0, 70, 220)), (0.42, (60, 60, 60)), (0.5, (200, 200, 200)),
                    (0.58, (90, 60, 50)), (0.85, (230, 40, 0)), (1.0, (255, 255, 120))],
    # Kingdom's default seismic bar look: dark blue troughs through white to dark red peaks with a yellow tip
    "Kingdom-style": [(0.0, (0, 0, 90)), (0.2, (0, 60, 200)), (0.42, (170, 200, 255)), (0.5, (255, 255, 255)),
                      (0.58, (255, 200, 160)), (0.8, (210, 30, 0)), (0.95, (120, 0, 0)), (1.0, (255, 230, 0))],
    "High contrast": [(0.0, (0, 0, 0)), (0.3, (0, 0, 255)), (0.48, (255, 255, 255)), (0.52, (255, 255, 255)),
                      (0.7, (255, 0, 0)), (1.0, (0, 0, 0))],
    "Blue-white-orange (CVD safe)": [(0.0, (8, 48, 107)), (0.25, (66, 146, 198)), (0.5, (247, 247, 247)),
                                     (0.75, (253, 141, 60)), (1.0, (127, 39, 4))],
}
ATTRIBUTE = {
    "Rainbow (attributes)": [(0.0, (48, 18, 59)), (0.25, (40, 160, 230)), (0.5, (120, 250, 80)),
                             (0.75, (250, 180, 40)), (1.0, (122, 4, 3))],
    "Viridis-style": _perceptual([(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]),
    "Cividis-style": _perceptual([(0, 32, 77), (66, 78, 108), (124, 123, 120), (188, 175, 111), (253, 234, 69)]),
    "Magma-style": _perceptual([(0, 0, 4), (81, 18, 124), (183, 55, 121), (252, 137, 97), (252, 253, 191)]),
    "Grey (perceptual)": _oklab_ramp([(0, 0, 0), (255, 255, 255)]),
    "Sand-shale": [(0.0, (242, 214, 75)), (0.45, (200, 185, 110)), (0.7, (142, 154, 124)), (1.0, (110, 80, 50))],
}
CVD_SAFE = {"Grey", "Grey (perceptual)", "Viridis-style", "Cividis-style", "Magma-style",
            "Blue-white-orange (CVD safe)"}
MONOTONIC = {"Grey", "Grey (perceptual)", "Viridis-style", "Cividis-style", "Magma-style"}
LIBRARY = dict(SEISMIC, **ATTRIBUTE)


class ColourTable:
    def __init__(self, name, points, flip=False, gamma=1.0, clip=99.0, symmetric=True, custom=False):
        self.name = str(name)
        self.points = [(float(p), tuple(int(c) for c in rgb[:3])) for p, rgb in points]
        self.flip, self.gamma, self.clip, self.symmetric = bool(flip), float(gamma), float(clip), bool(symmetric)
        self.custom = bool(custom)
        self._sort()

    def _sort(self):
        self.points.sort(key=lambda q: q[0])
        if not self.points:
            self.points = [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))]

    # ------------------------------------------------------------ editing (colour bar editor)
    def move_point(self, k, pos):
        """Drag control point k; the ends stay at 0 / 1, inner points cannot pass their neighbours."""
        n = len(self.points)
        if k <= 0 or k >= n - 1:
            return self.points[k][0]
        lo, hi = self.points[k - 1][0] + 1e-4, self.points[k + 1][0] - 1e-4
        p = float(np.clip(pos, lo, hi))
        self.points[k] = (p, self.points[k][1])
        return p

    def add_point(self, pos, rgb=None):
        pos = float(np.clip(pos, 0, 1))
        if rgb is None:
            rgb = tuple(int(v) for v in self.lut(apply_display=False)[int(round(pos * 255)), :3])
        self.points.append((pos, tuple(int(c) for c in rgb[:3])))
        self._sort()
        return [p for p, _ in self.points].index(pos)

    def remove_point(self, k):
        if 0 < k < len(self.points) - 1:
            del self.points[k]
            return True
        return False

    def set_colour(self, k, rgb):
        self.points[k] = (self.points[k][0], tuple(int(c) for c in rgb[:3]))

    # ------------------------------------------------------------ output
    def lut(self, n=256, apply_display=True):
        pos = np.array([p for p, _ in self.points])
        col = np.array([c for _, c in self.points], float)
        x = np.linspace(0, 1, n)
        if apply_display and self.gamma != 1.0:
            # gamma about the centre for diverging (symmetric) tables, from 0 otherwise
            if self.symmetric:
                d = 2 * x - 1
                x = 0.5 + 0.5 * np.sign(d) * np.abs(d) ** (1.0 / self.gamma)
            else:
                x = x ** (1.0 / self.gamma)
        out = np.empty((n, 4), np.uint8)
        for k in range(3):
            out[:, k] = np.round(np.interp(x, pos, col[:, k])).astype(np.uint8)
        out[:, 3] = 255
        return out[::-1].copy() if (apply_display and self.flip) else out

    def levels(self, data):
        """(lo, hi) display levels: symmetric clip = +/- the clip percentile of |a|; else percentile range."""
        a = np.asarray(data, float)
        fin = a[np.isfinite(a)]
        if not fin.size:
            return 0.0, 1.0
        if self.symmetric:
            m = float(np.percentile(np.abs(fin), self.clip)) or 1.0
            return -m, m
        q = (100.0 - self.clip) / 2.0
        lo, hi = float(np.percentile(fin, q)), float(np.percentile(fin, 100 - q))
        return (lo, hi) if hi > lo else (lo, lo + 1.0)

    def to_dict(self):
        return {"name": self.name, "points": [[p, list(c)] for p, c in self.points], "flip": self.flip,
                "gamma": self.gamma, "clip": self.clip, "symmetric": self.symmetric}

    @classmethod
    def from_dict(cls, d, custom=True):
        return cls(d["name"], [(p, c) for p, c in d["points"]], d.get("flip", False), d.get("gamma", 1.0),
                   d.get("clip", 99.0), d.get("symmetric", True), custom)

    def copy(self, name=None):
        return ColourTable.from_dict(dict(self.to_dict(), name=name or self.name), self.custom)

    @property
    def cvd_safe(self):
        return self.name in CVD_SAFE

    @property
    def monotonic(self):
        return self.name in MONOTONIC


def is_monotonic(lut, tol=1e-3):
    L = lightness(lut)
    d = np.diff(L)
    return bool(np.all(d >= -tol) or np.all(d <= tol))


# ====================================================================== library + project storage
KEY = "colour_tables"


def names(project=None, kind=None):
    """Library names (seismic first) + the project's custom tables."""
    base = list(SEISMIC) + list(ATTRIBUTE) if kind is None else list(SEISMIC if kind == "seismic" else ATTRIBUTE)
    return base + [n for n in custom_tables(project) if n not in base]


def label(name):
    """Name as shown in pickers: colourblind-safe tables are marked."""
    return f"{name}  [CVD safe]" if name in CVD_SAFE else name


def custom_tables(project):
    d = (getattr(project, "display", None) or {}).get(KEY) or {} if project is not None else {}
    return {k: ColourTable.from_dict(v) for k, v in d.items()}


_REGISTRY = {}  # in-process custom tables (registered by the open project) so lut(name) finds them anywhere


def register(table):
    _REGISTRY[table.name] = table
    return table


def get(name, project=None):
    if project is not None:
        ct = custom_tables(project)
        if name in ct:
            return ct[name]
    if name in _REGISTRY:
        return _REGISTRY[name].copy()
    pts = LIBRARY.get(name)
    if pts is None:
        return None
    return ColourTable(name, pts, symmetric=name in SEISMIC)


def save_custom(project, table):
    disp = getattr(project, "display", None)
    if disp is None:
        project.display = disp = {}
    table.custom = True
    disp.setdefault(KEY, {})[table.name] = table.to_dict()
    register(table)
    return disp[KEY][table.name]


def delete_custom(project, name):
    d = (getattr(project, "display", None) or {}).get(KEY) or {}
    _REGISTRY.pop(name, None)
    return d.pop(name, None) is not None


def load_project(project):
    """Register the project's custom tables (called when a project opens)."""
    for t in custom_tables(project).values():
        register(t)
    return list(custom_tables(project))


def export_json(table, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table.to_dict(), f, indent=1)
    return path


def export_csv(table, path, n=256):
    """256-row position,R,G,B table (Kingdom / Petrel / OpendTect colour-bar imports read this shape)."""
    lut = table.lut(n)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["position", "R", "G", "B"])
        for k in range(n):
            w.writerow([f"{k / (n - 1):.6f}", int(lut[k, 0]), int(lut[k, 1]), int(lut[k, 2])])
    return path


def import_file(path, name=None):
    """JSON (our format) or CSV / text with 3 (R G B) or 4 (pos R G B) numbers per row, 0-255 or 0-1."""
    nm = name or os.path.splitext(os.path.basename(path))[0]
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        d["name"] = name or d.get("name") or nm
        return ColourTable.from_dict(d)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = [p for p in line.replace(",", " ").replace(";", " ").split() if p]
            try:
                v = [float(p) for p in parts]
            except ValueError:
                continue
            if len(v) in (3, 4):
                rows.append(v)
    if len(rows) < 2:
        raise ValueError(f"{path}: no colour rows found")
    a = np.array(rows, float)
    if a.shape[1] == 3:
        a = np.c_[np.linspace(0, 1, len(a)), a]
    if a[:, 1:].max() <= 1.0:
        a[:, 1:] *= 255.0
    a[:, 0] = (a[:, 0] - a[0, 0]) / max(1e-12, a[-1, 0] - a[0, 0])
    if len(a) > 32:  # thin a 256-row bar to the control points needed to rebuild it within 2/255
        keep = [0, len(a) - 1]
        for _ in range(96):
            kk = sorted(keep)
            rec = np.stack([np.interp(a[:, 0], a[kk, 0], a[kk, c]) for c in (1, 2, 3)], axis=1)
            err = np.abs(rec - a[:, 1:]).max(axis=1)
            j = int(np.argmax(err))
            if err[j] <= 2.0:
                break
            keep.append(j)
        a = a[sorted(set(keep))]
    return ColourTable(nm, [(r[0], tuple(int(round(v)) for v in r[1:4])) for r in a], custom=True)
