"""Cross-section geometry, independent of any GUI toolkit.

Everything is laid out on one canvas: x in "track units" (one track = width 1.0 by default),
y in metres (MD, TVDSS or flattened).  The Qt view and the PNG/PDF exporter both draw from this.
"""
import math
import numpy as np

from .curves import FAMILY_INFO

# Track definitions: id -> (title, [(family, left, right, log, colour, width)])
TRACKS = {
    "LITH": ("Lithology", []),
    "GR": ("Gamma ray", [("GR", 0, 150, False, "#1b7f2a", 1.4)]),
    "RES": ("Resistivity", [("RDEEP", 0.2, 2000, True, "#c0161b", 1.4), ("RSHAL", 0.2, 2000, True, "#1f4fbf", 1.0)]),
    "DN": ("Density-neutron", [("RHOB", 1.95, 2.95, False, "#c0161b", 1.3), ("NPHI", 0.45, -0.15, False, "#1f4fbf", 1.1)]),
    "DT": ("Sonic", [("DT", 140, 40, False, "#7a2fb8", 1.2)]),
    "PE": ("PE", [("PE", 0, 10, False, "#8a5a00", 1.0)]),
    "SP": ("SP", [("SP", -100, 50, False, "#444444", 1.0)]),
    "CALI": ("Caliper", [("CALI", 6, 16, False, "#666666", 1.0)]),
    # petrophysics results (PLAN 20e; curves from <project>_petro/<well>.npz)
    "VSH": ("Petro Vsh", [("VSH", 0, 1, False, "#8b5a2b", 1.2)]),
    "PHI": ("Petro porosity", [("PHIE", 0.5, 0, False, "#1f4fbf", 1.2), ("BVW", 0.5, 0, False, "#00a0a0", 1.0)]),
    "SW": ("Petro Sw", [("SW", 1, 0, False, "#c0161b", 1.2)]),
    "PERM": ("Petro perm", [("PERM", 0.01, 10000, True, "#7a2fb8", 1.1)]),
    "FAC": ("Electrofacies", [("FACIES", -0.5, 7.5, False, "#333333", 1.4)]),
    "DTP": ("Predicted DT (P10-P90)", [("DT_P10", 140, 40, False, "#c9a0e8", 0.8), ("DT_P90", 140, 40, False, "#c9a0e8", 0.8),
                                       ("DT_PRED", 140, 40, False, "#7a2fb8", 1.3)]),
}
PETRO_FAMILIES = {"VSH", "PHIE", "BVW", "SW", "PERM", "FACIES", "DT_PRED", "DT_P10", "DT_P90"}
HC_COLOR = "#3cb043"
SAND_COLOR = "#f2d64b"
SHALE_COLOR = "#9aa48a"
GAS_COLOR = "#ffcc00"


def TRACK_SPAN(panel, t0):
    for _, a, b in panel.tracks:
        if abs(a - t0) < 1e-9:
            return b - a
    return 1.0


def peak_decimate(x, y, n):
    """Keep min and max x of each depth bin (y must be sorted). Returns about n points."""
    m = max(1, n // 2)
    k = int(np.ceil(len(x) / m))
    pad = m * k - len(x)
    xp = np.concatenate([x, np.full(pad, np.nan)]).reshape(m, k)
    yp = np.concatenate([y, np.full(pad, y[-1])]).reshape(m, k)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanmin(xp, axis=1), np.nanmax(xp, axis=1)
            ym = np.nanmean(yp, axis=1)
    xx = np.empty(2 * m)
    yy = np.empty(2 * m)
    xx[0::2], xx[1::2] = lo, hi
    yy[0::2], yy[1::2] = ym, ym
    return xx, yy


def to_x(values, left, right, log, x0, width):
    v = np.asarray(values, float)
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            lv = np.log10(np.where(v > 0, v, np.nan))
        f = (lv - math.log10(left)) / (math.log10(right) - math.log10(left))
    else:
        f = (v - left) / (right - left)
    f = np.clip(f, 0.0, 1.0)  # clip, do not wrap (wrapping comes later)
    return x0 + f * width


class WellPanel:
    def __init__(self, name, x0, width, tracks):
        self.name = name
        self.x0 = x0
        self.width = width
        self.tracks = tracks  # [(track_id, tx0, tx1)]

    @property
    def x1(self):
        return self.x0 + self.width


class SectionLayout:
    """Compute panel positions and display coordinates for a list of wells."""

    def __init__(self, project, well_names, display=None):
        self.project = project
        self.names = [n for n in well_names if n in project.wells]
        self.d = dict(project.display)
        if display:
            self.d.update(display)
        self.track_ids = [t for t in self.d.get("tracks", ["GR", "RES", "DN", "DT"]) if t in TRACKS]
        self.tw = float(self.d.get("track_width", 1.0))
        self.gap = float(self.d.get("well_gap", 0.6))
        self.hang = self.d.get("hang", "TVDSS")  # "MD", "TVDSS", or "FLAT:<formation>"
        self.panels = []
        self.distances = []  # km between consecutive wells
        self._ycache = {}
        self._layout()

    # ------------------------------------------------------------- layout
    def _layout(self):
        wells = [self.project.wells[n] for n in self.names]
        dist = []
        for a, b in zip(wells[:-1], wells[1:]):
            if None not in (a.x, a.y, b.x, b.y):
                dist.append(math.hypot(b.x - a.x, b.y - a.y) / 1000.0)
            else:
                dist.append(None)
        self.distances = dist
        width = self.tw * len(self.track_ids)
        gaps = [self.gap * width] * len(dist)
        if self.d.get("spacing") == "proportional" and dist and all(x is not None for x in dist):
            mean = sum(dist) / len(dist) or 1.0
            # keep the average gap the same as "equal", minimum 15 % of a panel
            gaps = [max(0.15 * width, self.gap * width * x / mean) for x in dist]
        x = 0.0
        self.panels = []
        for i, n in enumerate(self.names):
            tracks = []
            tx = x
            for t in self.track_ids:
                tracks.append((t, tx, tx + self.tw))
                tx += self.tw
            self.panels.append(WellPanel(n, x, width, tracks))
            x += width + (gaps[i] if i < len(gaps) else 0)
        self.total_width = x

    # ------------------------------------------------------------- depth
    def y_of(self, well, md):
        """Display y for MD values. Positive down."""
        md = np.asarray(md, float)
        if self.hang == "MD":
            return md
        z = well.md_to_tvdss(md)
        if self.hang.startswith("FLAT:"):
            f = self.hang[5:]
            top = self.project.tops.get(well.name, {}).get(f)
            if top is None:
                return np.full_like(z, np.nan)
            return z - float(well.md_to_tvdss(top))
        return z

    def y_full(self, well):
        """Display y for the well's whole MD grid (cached per layout)."""
        y = self._ycache.get(well.name)
        if y is None or len(y) != len(well.depth):
            y = self.y_of(well, well.depth)
            self._ycache[well.name] = y
        return y

    def md_at(self, well, y):
        """Inverse of y_of for a single display y. Returns MD or None."""
        if self.hang == "MD":
            return float(y)
        if self.hang.startswith("FLAT:"):
            top = self.project.tops.get(well.name, {}).get(self.hang[5:])
            if top is None:
                return None
            y = y + float(well.md_to_tvdss([top])[0])
        return float(well.tvdss_to_md([y])[0])

    def panel_at(self, x):
        for p in self.panels:
            if p.x0 <= x <= p.x1:
                return p
        return None

    def y_label(self):
        if self.hang == "MD":
            return "Measured depth (m)"
        if self.hang.startswith("FLAT:"):
            return f"Depth relative to {self.hang[5:]} (m)"
        return "TVDSS (m)"

    # ------------------------------------------------------------- curves
    def curves(self, panel, max_points=None):
        """Yield dicts for every log curve of a panel: x, y, colour, width, family, track.
        With max_points, curves are peak-decimated (min and max kept per bin, spikes survive)."""
        w = self.project.wells[panel.name]
        if not w.loaded:
            self.project.load_well(w.name)
        if w.depth is None or not len(w.depth):
            return
        y = self.y_full(w)
        for tid, tx0, tx1 in panel.tracks:
            for fam, left, right, log, col, lw in TRACKS[tid][1]:
                v = self.log_values(w, fam)
                if v is None:
                    continue
                x = to_x(v, left, right, log, tx0, tx1 - tx0)
                x = np.where(np.isfinite(v), x, np.nan)
                xx, yy = (peak_decimate(x, y, max_points) if max_points and len(x) > max_points else (x, y))
                yield {"x": xx, "y": yy, "color": col, "width": lw, "family": fam, "track": tid,
                       "left": left, "right": right, "log": log}

    def log_values(self, w, fam):
        """A composited log, or a saved petrophysics result curve (VSH, PHIE, BVW, SW, PERM)."""
        v = w.logs.get(fam)
        if v is None and fam in PETRO_FAMILIES:
            try:
                from .petro.store import well_curves
                v = well_curves(self.project, w).get(fam)
            except Exception:
                v = None
        return v

    def fills(self, panel):
        """Yield fill polygons as (x_a, x_b, y, colour): fill between two x curves along y."""
        w = self.project.wells[panel.name]
        if not w.loaded:
            self.project.load_well(w.name)
        if w.depth is None or not len(w.depth):
            return
        y = self.y_full(w)
        for tid, tx0, tx1 in panel.tracks:
            if tid == "GR" and "GR" in w.logs and self.d.get("gr_fill", "cutoff") != "gradient":
                gr = w.logs["GR"]
                cut = float(self.d.get("gr_cutoff", 75.0))
                xc = to_x([cut], 0, 150, False, tx0, tx1 - tx0)[0]
                x = to_x(gr, 0, 150, False, tx0, tx1 - tx0)
                ok = np.isfinite(gr)
                xs = np.where(ok, np.minimum(x, xc), xc)
                xh = np.where(ok, np.maximum(x, xc), xc)
                yield (xs, np.full_like(xs, xc), y, SAND_COLOR)
                yield (np.full_like(xh, xc), xh, y, SHALE_COLOR)
            if tid == "PHI":
                phie, bvw = self.log_values(w, "PHIE"), self.log_values(w, "BVW")
                if phie is not None and bvw is not None:
                    ok = np.isfinite(phie) & np.isfinite(bvw)
                    xp = to_x(np.where(ok, phie, 0), 0.5, 0, False, tx0, tx1 - tx0)
                    xb = to_x(np.where(ok, bvw, 0), 0.5, 0, False, tx0, tx1 - tx0)
                    yield (xp, xb, y, HC_COLOR)  # hydrocarbon = PHIE - BVW
            if tid == "DN" and "RHOB" in w.logs and "NPHI" in w.logs:
                xr = to_x(w.logs["RHOB"], 1.95, 2.95, False, tx0, tx1 - tx0)
                xn = to_x(w.logs["NPHI"], 0.45, -0.15, False, tx0, tx1 - tx0)
                ok = np.isfinite(w.logs["RHOB"]) & np.isfinite(w.logs["NPHI"])
                a = np.where(ok, xr, tx0)
                b = np.where(ok, np.maximum(xn, xr), tx0)
                yield (a, b, y, GAS_COLOR)

    # ------------------------------------------------------------- tops
    def top_marks(self):
        """For each formation: list of (panel, y) and connecting segments between neighbours."""
        out = []
        for f in self.project.formations:
            name, col = f["name"], f.get("color", "#000000")
            marks = []
            metas = []
            for p in self.panels:
                md = self.project.tops.get(p.name, {}).get(name)
                if md is None:
                    marks.append(None)
                    continue
                w = self.project.wells[p.name]
                marks.append((p, float(self.y_of(w, [md])[0]), md))
                metas.append(self.project.tops_meta.get(p.name, {}).get(name, {}))
            segs = []
            for a, b in zip(marks[:-1], marks[1:]):
                if a and b and np.isfinite(a[1]) and np.isfinite(b[1]):
                    segs.append(((a[0].x1, a[1]), (b[0].x0, b[1])))
            out.append({"name": name, "color": col, "marks": [m for m in marks if m], "segments": segs, "metas": metas})
        return out

    def fill_images(self, panel, dy=0.25, ncols=40):
        """Rasterised GR and density-neutron fills: [(rgba, x0, x1, y0, y1)] (fast to draw)."""
        out = {}
        for xa, xb, y, col in self.fills(panel):
            ok = np.isfinite(y)
            if ok.sum() < 2:
                continue
            yy, a, b = y[ok], xa[ok], xb[ok]
            lo_x, hi_x = np.minimum(a, b), np.maximum(a, b)
            # which track is this fill in?
            tx = next(((t0, t1) for _, t0, t1 in panel.tracks if t0 - 1e-6 <= np.nanmin(lo_x) <= t1 + 1e-6), None)
            if tx is None:
                continue
            t0, t1 = tx
            y0, y1 = float(yy.min()), float(yy.max())
            grid = np.arange(y0, y1 + dy, dy)
            idx = np.clip(np.searchsorted(yy, grid), 0, len(yy) - 1)
            gl, gh = lo_x[idx], hi_x[idx]
            cx = t0 + (np.arange(ncols) + 0.5) / ncols * (t1 - t0)
            mask = (cx[None, :] >= gl[:, None]) & (cx[None, :] <= gh[:, None])
            key = (t0, y0, y1)
            img = out.get(key)
            if img is None:
                img = np.zeros((len(grid), ncols, 4), np.uint8)
                out[key] = img
            h = col.lstrip("#")
            img[mask] = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 215)
        return [(img, t0, t0 + (TRACK_SPAN(panel, t0)), y0, y1 + dy) for (t0, y0, y1), img in out.items()]

    def track_lith_images(self):
        """Lithology columns for LITH tracks: [(rgba, x0, x1, y0, y1)]."""
        from .lithology import track_codes, track_image
        res = []
        lib = None
        if self.d.get("patterns"):  # 40h: pattern library (dots / dashes / bricks ...) instead of flat colours
            from .trackstyle import library
            lib = library(self.project)
        for p in self.panels:
            if not self.project.wells[p.name].loaded:
                self.project.load_well(p.name)
            for tid, tx0, tx1 in p.tracks:
                if tid == "LITH" and lib is not None:
                    from .trackstyle import pattern_column
                    r = track_codes(self, p, dy=0.5)
                    if r:
                        codes, y0, y1 = r
                        res.append((pattern_column(codes, 24, lib), tx0, tx1, y0, y1))
                    continue
                if tid == "LITH":
                    r = track_image(self, p)
                    if r:
                        img, y0, y1 = r
                        res.append((img, tx0, tx1, y0, y1))
        return res

    def style_images(self, panel, dy=0.5, ncols=40):
        """40h track styling: GR gradient fill (display gr_fill = 'gradient'), shaded net / pay flags from cutoffs
        (display pay.show), neutron-density crossover (display crossover: gas yellow, shale grey).
        [(rgba, x0, x1, y0, y1)]."""
        from . import trackstyle as TS
        w = self.project.wells[panel.name]
        if not w.loaded:
            self.project.load_well(w.name)
        if w.depth is None or not len(w.depth):
            return []
        y = self.y_full(w)
        ok = np.isfinite(y)
        if ok.sum() < 2:
            return []
        y0, y1 = float(np.nanmin(y[ok])), float(np.nanmax(y[ok]))
        grid = np.arange(y0, y1 + dy, dy)
        order = np.argsort(y[ok])
        ys = y[ok][order]

        def on_grid(v):
            v = np.asarray(v, float)[ok][order]
            return np.interp(grid, ys, v, left=np.nan, right=np.nan)
        out = []
        for tid, tx0, tx1 in panel.tracks:
            if tid == "GR" and "GR" in w.logs and self.d.get("gr_fill") == "gradient":
                g = on_grid(w.logs["GR"])
                img = TS.curve_fill_image(g, TS.gr_lut(self.d.get("gr_lut", "Sand-shale")), 0.0, 150.0, ncols,
                                          x=np.clip(g / 150.0, 0, 1), side="left")
                out.append((img, tx0, tx1, y0, y1 + dy))
            if tid == "GR" and (self.d.get("pay") or {}).get("show"):
                pp = self.d["pay"]
                res = TS.net_pay(grid, on_grid(self.log_values(w, "VSH")) if self.log_values(w, "VSH") is not None
                                 else None,
                                 on_grid(self.log_values(w, "PHIE")) if self.log_values(w, "PHIE") is not None else None,
                                 on_grid(self.log_values(w, "SW")) if self.log_values(w, "SW") is not None else None,
                                 pp.get("vsh_max", 0.4), pp.get("phi_min", 0.08), pp.get("sw_max", 0.6))
                strip = np.zeros((len(grid), 4, 4), np.uint8)
                strip[res["net"]] = (60, 170, 60, 220)
                strip[res["pay"]] = (220, 40, 40, 230)
                out.append((strip, tx1 - 0.08 * (tx1 - tx0), tx1, y0, y1 + dy))
            if tid == "DN" and self.d.get("crossover") and "RHOB" in w.logs and "NPHI" in w.logs:
                rh, np_ = on_grid(w.logs["RHOB"]), on_grid(w.logs["NPHI"])
                xr = (rh - 1.95) / (2.95 - 1.95)
                xn = (np_ - 0.45) / (-0.15 - 0.45)
                cx = (np.arange(ncols) + 0.5) / ncols
                lo, hi = np.minimum(xr, xn), np.maximum(xr, xn)
                between = (cx[None, :] >= np.nan_to_num(lo, nan=2)[:, None]) & \
                          (cx[None, :] <= np.nan_to_num(hi, nan=-1)[:, None])
                gas = (np.nan_to_num(xn, nan=-1) > np.nan_to_num(xr, nan=2))[:, None]
                img = np.zeros((len(grid), ncols, 4), np.uint8)
                img[between & gas] = (255, 204, 0, 200)
                img[between & ~gas] = (150, 150, 150, 140)
                out.append((img, tx0, tx1, y0, y1 + dy))
        return out

    def gap_images(self, ylim, mode=None, cache=None):
        """Between-well fills: [(rgba, x0, x1, y0, y1)]. Pass a dict as cache to reuse unchanged gaps."""
        from .lithology import gap_image, gap_key
        mode = mode or self.d.get("between", "lithology")
        res = []
        if mode not in ("zones", "lithology"):
            return res
        for p in self.panels:
            if not self.project.wells[p.name].loaded:
                self.project.load_well(p.name)
        for a, b in zip(self.panels[:-1], self.panels[1:]):
            key = gap_key(self, a, b, mode, ylim)
            if cache is not None and key in cache:
                r = cache[key]
            else:
                r = gap_image(self, a, b, mode, ylim)
                if cache is not None:
                    cache[key] = r
            if r:
                res.append(r)
        return res

    def lith_images(self, ylim, mode=None):
        """All images (track lithology + between-well) as (rgba, x0, x1, y0, y1, kind)."""
        for img, x0, x1, y0, y1 in self.track_lith_images():
            yield img, x0, x1, y0, y1, "track"
        for p in self.panels:
            for img, x0, x1, y0, y1 in self.style_images(p):
                yield img, x0, x1, y0, y1, "style"
        for img, x0, x1, y0, y1 in self.gap_images(ylim, mode):
            yield img, x0, x1, y0, y1, "gap"

    def depth_range(self):
        lo, hi = np.inf, -np.inf
        for p in self.panels:
            w = self.project.wells[p.name]
            if not w.loaded:
                self.project.load_well(w.name)
            if w.depth is None or not len(w.depth):
                continue
            have = np.zeros(len(w.depth), bool)
            for v in w.logs.values():
                have |= np.isfinite(v)
            if have.any():
                y = self.y_of(w, w.depth[have])
                y = y[np.isfinite(y)]
                if y.size:
                    lo, hi = min(lo, y.min()), max(hi, y.max())
        if not np.isfinite(lo):
            return 0.0, 1000.0
        return float(lo), float(hi)

    def headers(self, panel):
        """Header text per track: list of (tx0, tx1, title, [(label, left, right, colour)])."""
        out = []
        for tid, tx0, tx1 in panel.tracks:
            title, curves = TRACKS[tid]
            if not curves:  # lithology column: legend-style label
                out.append((tx0, tx1, title, [("Lith", "", "", "#444444")]))
                continue
            out.append((tx0, tx1, title, [(FAMILY_INFO[f][0], l, r, c) for f, l, r, lg, c, lw in curves]))
        return out
