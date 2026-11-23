"""Horizon maps: time structure, isochron, amplitude extraction, depth conversion with well velocities,
contours and grid export (XYZ, CPS-3 ASCII, PNG).

A MapGrid is either on a 3D survey's inline/crossline grid ("ilxl": rows = inlines, cols = crosslines,
rotated in map space by the survey affine) or an axis-aligned XY grid ("xy": rows = y, cols = x).  2D-line
horizons are gridded to XY with inverse-distance weighting; cells far from any pick stay blank.
"""
import math

import numpy as np


class MapGrid:
    def __init__(self, z, kind, unit="ms", label="", vol=None, x0=0.0, y0=0.0, dx=1.0, dy=1.0):
        self.z = np.asarray(z, np.float64)
        self.kind = kind
        self.unit, self.label = unit, label
        self.vol = vol
        self.x0, self.y0, self.dx, self.dy = x0, y0, dx, dy

    # -------------------------------------------------------------- geometry
    def node_xy(self):
        """(X, Y) arrays of node coordinates, same shape as z."""
        ny, nx = self.z.shape
        if self.kind == "ilxl":
            v = self.vol
            I, J = np.meshgrid(v.inlines, v.xlines, indexing="ij")
            return v.xy(I, J)
        X, Y = np.meshgrid(self.x0 + self.dx * np.arange(nx), self.y0 + self.dy * np.arange(ny))
        return X, Y

    def sample(self, x, y):
        """Bilinear value at map points (NaN outside)."""
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        if self.kind == "ilxl":
            v = self.vol
            out = []
            for xx, yy in zip(x, y):
                il, xl = v.ilxl_at(xx, yy)
                out.append(((il - v.il0) / v.il_step, (xl - v.xl0) / v.xl_step))
            fi, fj = np.array(out).T if out else (np.array([]), np.array([]))
        else:
            fj = (x - self.x0) / self.dx
            fi = (y - self.y0) / self.dy
        ny, nx = self.z.shape
        ok = (fi >= 0) & (fi <= ny - 1) & (fj >= 0) & (fj <= nx - 1)
        i0 = np.clip(np.floor(fi).astype(int), 0, max(0, ny - 2))
        j0 = np.clip(np.floor(fj).astype(int), 0, max(0, nx - 2))
        wi, wj = np.clip(fi - i0, 0, 1), np.clip(fj - j0, 0, 1)
        i1, j1 = np.minimum(i0 + 1, ny - 1), np.minimum(j0 + 1, nx - 1)
        z = self.z
        val = (z[i0, j0] * (1 - wi) * (1 - wj) + z[i0, j1] * (1 - wi) * wj + z[i1, j0] * wi * (1 - wj) +
               z[i1, j1] * wi * wj)
        return np.where(ok, val, np.nan)

    def to_xy(self, cell=None):
        """Axis-aligned XY grid covering this grid (needed for CPS-3); bilinear resampling."""
        if self.kind == "xy":
            return self
        X, Y = self.node_xy()
        v = self.vol
        cell = cell or min(math.hypot(v.ax[2], v.ay[2]) * v.xl_step, math.hypot(v.ax[1], v.ay[1]) * v.il_step)
        cell = float(cell) or 25.0
        def axis(lo, hi):  # tolerant of round-off in the survey affine
            start = np.floor(lo / cell + 1e-6) * cell
            return start + cell * np.arange(int(np.floor((hi - start) / cell + 1e-6)) + 1)
        xs, ys = axis(X.min(), X.max()), axis(Y.min(), Y.max())
        GX, GY = np.meshgrid(xs, ys)
        a, b = v.ax, v.ay
        M = np.linalg.inv(np.array([[a[1], a[2]], [b[1], b[2]]]))
        il = M[0, 0] * (GX - a[0]) + M[0, 1] * (GY - b[0])
        xl = M[1, 0] * (GX - a[0]) + M[1, 1] * (GY - b[0])
        fi, fj = (il - v.il0) / v.il_step, (xl - v.xl0) / v.xl_step
        ny, nx = self.z.shape
        eps = 1e-6
        ok = (fi >= -eps) & (fi <= ny - 1 + eps) & (fj >= -eps) & (fj <= nx - 1 + eps)
        i0 = np.clip(np.floor(fi).astype(int), 0, ny - 2)
        j0 = np.clip(np.floor(fj).astype(int), 0, nx - 2)
        wi, wj = np.clip(fi - i0, 0, 1), np.clip(fj - j0, 0, 1)
        z = self.z
        val = (z[i0, j0] * (1 - wi) * (1 - wj) + z[i0, j0 + 1] * (1 - wi) * wj + z[i0 + 1, j0] * wi * (1 - wj) +
               z[i0 + 1, j0 + 1] * wi * wj)
        return MapGrid(np.where(ok, val, np.nan), "xy", self.unit, self.label, None, xs[0], ys[0], cell, cell)

    def transform(self):
        """(m11, m12, m21, m22, dx, dy) placing image pixels (col, row) on the map (QTransform order)."""
        if self.kind == "ilxl":
            from .horizons import grid_transform
            return grid_transform(self.vol)
        return self.dx, 0.0, 0.0, self.dy, self.x0 - self.dx / 2, self.y0 - self.dy / 2

    def contours(self, levels):
        """{level: (xs, ys)} segment pairs (NaN-free, connect='pairs') in map coordinates."""
        from .welltie import wc
        wc()
        from well_correlation.mapping import contour_segments
        ny, nx = self.z.shape
        g = {"z": self.z, "x0": 0.0, "y0": 0.0, "dx": 1.0, "dy": 1.0, "nx": nx, "ny": ny}
        segs = contour_segments(g, levels)
        out = {}
        for lev, (cx, cy) in segs.items():
            if self.kind == "ilxl":
                v = self.vol
                X, Y = v.xy(v.il0 + v.il_step * cy, v.xl0 + v.xl_step * cx)
            else:
                X, Y = self.x0 + self.dx * cx, self.y0 + self.dy * cy
            out[lev] = (np.asarray(X), np.asarray(Y))
        return out

    def stats(self):
        f = self.z[np.isfinite(self.z)]
        return (float(f.min()), float(f.max()), float(f.mean())) if f.size else (np.nan, np.nan, np.nan)


def nice_levels(z, target=12):
    f = z[np.isfinite(z)]
    if not f.size:
        return []
    lo, hi = float(f.min()), float(f.max())
    span = max(hi - lo, 1e-6)
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag), 10 * mag)
    return list(np.arange(math.ceil(lo / step) * step, hi, step))


# ====================================================================== grids from horizons
def idw_grid(x, y, v, cell, reach=None, power=2.0, max_pts=4000, pad=None):
    """Inverse-distance grid of scattered values; cells farther than reach from every point stay NaN."""
    x, y, v = (np.asarray(a, float) for a in (x, y, v))
    ok = np.isfinite(v)
    x, y, v = x[ok], y[ok], v[ok]
    if len(v) > max_pts:  # decimate evenly (2D lines are densely sampled)
        k = np.linspace(0, len(v) - 1, max_pts).astype(int)
        x, y, v = x[k], y[k], v[k]
    reach = reach or 10 * cell
    pad = reach if pad is None else pad
    xs = np.arange(np.floor((x.min() - pad) / cell) * cell, x.max() + pad + cell, cell)
    ys = np.arange(np.floor((y.min() - pad) / cell) * cell, y.max() + pad + cell, cell)
    GX, GY = np.meshgrid(xs, ys)
    num = np.zeros(GX.shape)
    den = np.zeros(GX.shape)
    dmin = np.full(GX.shape, np.inf)
    for k0 in range(0, len(v), 256):
        px, py, pv = x[k0:k0 + 256], y[k0:k0 + 256], v[k0:k0 + 256]
        d2 = (GX[..., None] - px) ** 2 + (GY[..., None] - py) ** 2
        w = 1.0 / np.maximum(d2, 1e-6) ** (power / 2)
        num += (w * pv).sum(-1)
        den += w.sum(-1)
        dmin = np.minimum(dmin, np.sqrt(d2.min(-1)))
    z = np.where(dmin <= reach, num / np.maximum(den, 1e-300), np.nan)
    return xs, ys, z


def horizon_grid(project, h, key, cell=100.0, reach=None):
    """MapGrid of a horizon on one survey key ("3D:<vol>") or on all its 2D lines ("2D")."""
    if key.startswith("3D:"):
        vol = project.volume(key[3:])
        return MapGrid(h.get(key), "ilxl", "ms", f"{h.name} time", vol)
    xs, ys, vs = [], [], []
    for k, a in h.data.items():
        if not k.startswith("2D:"):
            continue
        line = project.line(k[3:])
        if not line.has_coords:
            continue
        ok = np.isfinite(a)
        xs.append(line.x[ok])
        ys.append(line.y[ok])
        vs.append(a[ok])
    if not vs:
        raise ValueError(f"{h.name} has no picks on 2D lines with coordinates")
    gx, gy, z = idw_grid(np.concatenate(xs), np.concatenate(ys), np.concatenate(vs), cell, reach)
    return MapGrid(z, "xy", "ms", f"{h.name} time (2D, gridded)", None, gx[0], gy[0], cell, cell)


def isochron(g_top, g_base):
    z = g_base.z - g_top.z
    return MapGrid(z, g_top.kind, "ms", f"Isochron {g_top.label} - {g_base.label}", g_top.vol, g_top.x0, g_top.y0,
                   g_top.dx, g_top.dy)


def extract_amplitude(vol, surface_ms, above_ms=0.0, below_ms=0.0, kind="rms", block=32):
    """Amplitude attribute of a 3D volume in a window around a surface (nil, nxl):
    kind = 'rms' | 'max_abs' | 'max' | 'min' | 'mean' | 'at' (value interpolated at the surface)."""
    s = np.asarray(surface_ms, float)
    out = np.full(s.shape, np.nan)
    dt_ms, t0_ms = vol.dt * 1000.0, vol.t0 * 1000.0
    for a in range(0, vol.nil, block):
        b = min(vol.nil, a + block)
        sb = s[a:b]
        fin = np.isfinite(sb)
        if not fin.any():
            continue
        k_lo = int(max(0, np.floor((np.nanmin(sb) - above_ms - t0_ms) / dt_ms) - 1))
        k_hi = int(min(vol.ns - 1, np.ceil((np.nanmax(sb) + below_ms - t0_ms) / dt_ms) + 1))
        tr = np.asarray(vol.ilxl[a:b, :, k_lo:k_hi + 1], np.float32) * vol.inv  # (bi, nxl, nk)
        tk = t0_ms + dt_ms * np.arange(k_lo, k_hi + 1)
        if kind == "at":
            idx = (sb - tk[0]) / dt_ms
            idc = np.clip(np.nan_to_num(idx), 0, tr.shape[2] - 1.000001)
            k0 = np.floor(idc).astype(int)
            w = idc - k0
            v0 = np.take_along_axis(tr, k0[..., None], 2)[..., 0]
            v1 = np.take_along_axis(tr, (k0 + 1)[..., None], 2)[..., 0]
            out[a:b] = np.where(fin, v0 * (1 - w) + v1 * w, np.nan)
            continue
        m = (tk[None, None, :] >= sb[..., None] - above_ms - 1e-6) & (tk[None, None, :] <= sb[..., None] + below_ms + 1e-6)
        cnt = m.sum(-1)
        x = np.where(m, tr, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            if kind == "rms":
                v = np.sqrt((x * x).sum(-1) / cnt)
            elif kind == "mean":
                v = x.sum(-1) / cnt
            elif kind == "max_abs":
                v = np.where(m, np.abs(tr), -np.inf).max(-1)
            elif kind == "max":
                v = np.where(m, tr, -np.inf).max(-1)
            elif kind == "min":
                v = np.where(m, tr, np.inf).min(-1)
            else:
                raise ValueError(kind)
        out[a:b] = np.where(fin & (cnt > 0), v, np.nan)
    return out


# ====================================================================== depth conversion
def well_velocity_points(wellset, tgrid, formation=None):
    """[(well, x, y, v_avg m/s, t ms, z m)] at the wells inside the time grid.
    z = the well's formation top (TVDSS) when `formation` is given and picked, else depth of the horizon time
    from the well's time-depth function."""
    pts = []
    for n in wellset.names:
        w = wellset.well(n)
        if w.x is None:
            continue
        t = float(tgrid.sample([w.x], [w.y])[0])
        if not np.isfinite(t) or t <= 0:
            continue
        z = None
        if formation:
            md = (wellset.wcp.tops or {}).get(n, {}).get(formation)
            if md is not None:
                try:
                    wellset.wcp.load_well(n)
                except Exception:
                    pass
                z = float(w.md_to_tvdss(md))
        if z is None:
            z = float(wellset.td_for(n).depth(t))
        pts.append((n, w.x, w.y, 2000.0 * z / t, t, z))
    return pts


def depth_convert(tgrid, vel_pts, method="idw"):
    """Depth (m TVDSS) = V_avg(x, y) * t / 2000, V_avg gridded from the wells (constant if one well)."""
    if not vel_pts:
        raise ValueError("no well inside the map with a time-depth function")
    X, Y = tgrid.node_xy()
    vx = np.array([p[1] for p in vel_pts])
    vy = np.array([p[2] for p in vel_pts])
    vv = np.array([p[3] for p in vel_pts])
    if len(vv) == 1:
        V = np.full(X.shape, vv[0])
    else:
        num = np.zeros(X.shape)
        den = np.zeros(X.shape)
        for x, y, v in zip(vx, vy, vv):
            w = 1.0 / np.maximum((X - x) ** 2 + (Y - y) ** 2, 1.0)
            num += w * v
            den += w
        V = num / den
    z = V * tgrid.z / 2000.0
    g = MapGrid(z, tgrid.kind, "m", tgrid.label.replace("time", "depth") + " (TVDSS)", tgrid.vol, tgrid.x0, tgrid.y0,
                tgrid.dx, tgrid.dy)
    g.velocity = V
    return g


# ====================================================================== export
def write_xyz(g, path, header=True):
    X, Y = g.node_xy()
    ok = np.isfinite(g.z)
    with open(path, "w") as fh:
        if header:
            fh.write(f"# {g.label} [{g.unit}]\n# X Y Z\n")
        np.savetxt(fh, np.column_stack([X[ok], Y[ok], g.z[ok]]), fmt="%.2f %.2f %.4f")
    return path


NULL_CPS = 1e30


def write_cps3(g, path, name=None):
    """CPS-3 ASCII grid (axis aligned; 3D grids are resampled). Rows from the top (max Y) down, values along X."""
    gx = g.to_xy()
    ny, nx = gx.z.shape
    xmax = gx.x0 + gx.dx * (nx - 1)
    ymax = gx.y0 + gx.dy * (ny - 1)
    f = gx.z[np.isfinite(gx.z)]
    zmin, zmax = (float(f.min()), float(f.max())) if f.size else (0.0, 0.0)
    vals = np.where(np.isfinite(gx.z), gx.z, NULL_CPS)[::-1].ravel()  # top row first
    with open(path, "w") as fh:
        fh.write(f"FSASCI 0 1 COMPUTED 0 {NULL_CPS:.6E}\nFSATTR 0 0\n")
        fh.write(f"FSLIMI {gx.x0:.4f} {xmax:.4f} {gx.y0:.4f} {ymax:.4f} {zmin:.6f} {zmax:.6f}\n")
        fh.write(f"FSNROW {ny} {nx}\nFSXINC {gx.dx:.6f} {gx.dy:.6f}\n->{name or g.label}\n")
        for k in range(0, len(vals), 5):
            fh.write(" ".join(f"{v:.7E}" for v in vals[k:k + 5]) + "\n")
    return path


def read_cps3(path):
    with open(path) as fh:
        lines = fh.read().splitlines()
    hdr = {}
    data = []
    for ln in lines:
        t = ln.split()
        if not t:
            continue
        if t[0].startswith("FS"):
            hdr[t[0]] = t[1:]
        elif ln.startswith("->"):
            continue
        else:
            data += [float(v) for v in t]
    ny, nx = int(hdr["FSNROW"][0]), int(hdr["FSNROW"][1])
    x0, _, y0, _ = (float(v) for v in hdr["FSLIMI"][:4])
    dx, dy = float(hdr["FSXINC"][0]), float(hdr["FSXINC"][1])
    z = np.array(data[:nx * ny]).reshape(ny, nx)[::-1]
    z[z >= 1e29] = np.nan
    return MapGrid(z, "xy", "", "", None, x0, y0, dx, dy)


def export_png(g, path, title=None, wells=None, levels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gx = g.to_xy()
    ny, nx = gx.z.shape
    X = gx.x0 + gx.dx * np.arange(nx)
    Y = gx.y0 + gx.dy * np.arange(ny)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=120)
    levels = levels or nice_levels(gx.z)
    if len(levels) >= 2:
        cf = ax.contourf(X, Y, gx.z, levels=levels, cmap="RdYlBu_r" if g.unit in ("m", "ms") else "viridis", extend="both")
        cs = ax.contour(X, Y, gx.z, levels=levels, colors="k", linewidths=0.4)
        ax.clabel(cs, fmt="%g", fontsize=7)
        fig.colorbar(cf, ax=ax, label=g.unit, shrink=0.8)
    else:
        im = ax.imshow(gx.z, origin="lower", extent=[X[0], X[-1], Y[0], Y[-1]])
        fig.colorbar(im, ax=ax, label=g.unit, shrink=0.8)
    for n, x, y in wells or []:
        ax.plot(x, y, "ko", ms=4)
        ax.annotate(n, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_aspect("equal")
    ax.set_title(title or g.label)
    ax.ticklabel_format(style="plain", useOffset=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
