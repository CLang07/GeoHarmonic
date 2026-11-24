"""Co-blending, overlays and directional illumination (PLAN 23a / 23b).  Pure numpy - used by the section panel, the
map hub (layer kind "blend", grid style "shade_from") and tests.

    rgba = rgb_blend([a10, a20, a40], mode="rgb", clip=(2, 98), gamma=(1, 1, 1))     # spectral decomposition blend
    tri = legend_triangle("rgb")                                                      # colour-triangle legend
    rgba = overlay(seis_rgba, attr, lut, opacity=0.5)                                  # attribute over the seismic
    s = illuminate(z, dx, dy, azimuth=315, elevation=35, ve=2)                         # bump map, Lambert 0..1
    rgba = amplitude_on_structure(amp, struct, lut, dx, dy)                            # colour = amp, shade = structure

Modes: "rgb" (additive: channel k drives R / G / B), "cmy" (subtractive on white: channel k removes R / G / B, so high
values = cyan / magenta / yellow), "hsv" (channel 1 = hue, 2 = saturation, 3 = value; with 2 channels value = 1).
NaN in any channel -> transparent.
"""
import math

import numpy as np

MODES = ("rgb", "cmy", "hsv")


def normalize(a, clip=(2.0, 98.0), gamma=1.0, vmin=None, vmax=None):
    """0..1 with percentile clip (or explicit vmin / vmax) and gamma (out = n ** (1 / gamma)); NaN stays NaN."""
    a = np.asarray(a, float)
    fin = np.isfinite(a)
    if not fin.any():
        return np.full(a.shape, np.nan)
    lo = float(np.percentile(a[fin], clip[0])) if vmin is None else float(vmin)
    hi = float(np.percentile(a[fin], clip[1])) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1e-12
    n = np.clip((a - lo) / (hi - lo), 0, 1)
    g = float(gamma or 1.0)
    if g != 1.0:
        n = n ** (1.0 / g)
    return np.where(fin, n, np.nan)


def _per(v, k, default):
    if v is None:
        return default
    if np.ndim(v) == 0:
        return v
    v = list(v)
    return v[k] if k < len(v) and v[k] is not None else default


def hsv_to_rgb(h, s, v):
    """Vectorised HSV (0..1 each) -> RGB 0..1."""
    h = np.mod(h, 1.0) * 6.0
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return r, g, b


def channels_to_rgb(ch, mode="rgb", hue_span=0.8):
    """Normalised channels (list of 2-3 arrays 0..1) -> r, g, b 0..1 (NaN propagates)."""
    mode = mode.lower()
    if mode not in MODES:
        raise ValueError(f"blend mode {mode} not in {MODES}")
    ch = [np.asarray(c, float) for c in ch]
    if not 1 <= len(ch) <= 3:
        raise ValueError("blend needs 1-3 channels")
    zero = np.zeros_like(ch[0])
    if mode == "hsv":
        h = ch[0] * hue_span  # 0..0.8: red -> magenta, avoids hue wrap-around (low and high both red)
        s = ch[1] if len(ch) > 1 else zero + 1
        v = ch[2] if len(ch) > 2 else zero + 1
        bad = ~(np.isfinite(h) & np.isfinite(s) & np.isfinite(v))
        r, g, b = hsv_to_rgb(np.nan_to_num(h), np.nan_to_num(s), np.nan_to_num(v))
        return [np.where(bad, np.nan, c) for c in (r, g, b)]
    full = ch + [zero] * (3 - len(ch))
    if mode == "rgb":
        return full
    return [1.0 - c for c in full]  # cmy: subtractive


def rgb_blend(arrays, mode="rgb", clip=(2.0, 98.0), gamma=1.0, vmin=None, vmax=None, opacity=1.0, hue_span=0.8):
    """2-3 attribute arrays (same shape) -> rgba uint8 (..., 4).  clip / gamma / vmin / vmax: one value or one per
    channel."""
    ch = []
    for k, a in enumerate(arrays):
        c = clip if (np.ndim(clip) == 1 and len(clip) == 2 and np.ndim(clip[0]) == 0) else _per(clip, k, (2.0, 98.0))
        ch.append(normalize(a, c, _per(gamma, k, 1.0), _per(vmin, k, None), _per(vmax, k, None)))
    r, g, b = channels_to_rgb(ch, mode, hue_span)
    return to_rgba(r, g, b, opacity)


def to_rgba(r, g, b, opacity=1.0):
    bad = ~(np.isfinite(r) & np.isfinite(g) & np.isfinite(b))
    out = np.zeros(np.shape(r) + (4,), np.uint8)
    for k, c in enumerate((r, g, b)):
        out[..., k] = np.clip(np.round(np.nan_to_num(c) * 255), 0, 255).astype(np.uint8)
    out[..., 3] = np.where(bad, 0, int(round(255 * float(opacity)))).astype(np.uint8)
    return out


def legend_triangle(mode="rgb", n=96, labels=("A", "B", "C"), hue_span=0.8):
    """Colour-triangle legend (rgba n x n, outside the triangle transparent) + corner label positions.  Barycentric
    weights of the three corners are the channel values, so each corner is one channel at full strength and the
    centre is all three at 1/3; for hsv the triangle is hue (along the base) x value (height) at full saturation."""
    yy, xx = np.mgrid[0:n, 0:n] / max(1, n - 1)
    # corners: channel 1 bottom-left, 2 bottom-right, 3 top (image rows down, so top = row 0)
    P = np.array([[0.0, 1.0], [1.0, 1.0], [0.5, 0.0]])
    T = np.array([[P[0, 0] - P[2, 0], P[1, 0] - P[2, 0]], [P[0, 1] - P[2, 1], P[1, 1] - P[2, 1]]])
    Ti = np.linalg.inv(T)
    l1 = Ti[0, 0] * (xx - P[2, 0]) + Ti[0, 1] * (yy - P[2, 1])
    l2 = Ti[1, 0] * (xx - P[2, 0]) + Ti[1, 1] * (yy - P[2, 1])
    l3 = 1 - l1 - l2
    inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
    if mode == "hsv":
        r, g, b = channels_to_rgb([np.clip(xx, 0, 1), np.ones_like(xx), np.clip(1 - yy, 0, 1)], "hsv", hue_span)
    else:
        m = np.maximum(np.maximum(l1, l2), l3)
        m = np.where(m > 0, m, 1)
        r, g, b = channels_to_rgb([np.clip(l1 / m, 0, 1), np.clip(l2 / m, 0, 1), np.clip(l3 / m, 0, 1)], mode)
    rgba = to_rgba(r, g, b)
    rgba[..., 3] = np.where(inside, 255, 0)
    pos = [(float(P[k, 0] * (n - 1)), float(P[k, 1] * (n - 1)), labels[k] if k < len(labels) else "") for k in range(3)]
    return rgba, pos


def channel_swatches(mode="rgb", n=64):
    """Per-channel ramps (list of 3 rgba (1, n, 4)): what each channel alone looks like from 0 to its max."""
    out = []
    ramp = np.linspace(0, 1, n)[None, :]
    z = np.zeros_like(ramp)
    for k in range(3):
        ch = [z, z, z]
        if mode == "hsv":
            ch = [ramp if k == 0 else z + 0.5, ramp if k == 1 else z + 1, ramp if k == 2 else z + 1]
        else:
            ch[k] = ramp
        out.append(to_rgba(*channels_to_rgb(ch, mode)))
    return out


def lut_rgba(a, lut, vmin=None, vmax=None, clip=(1.0, 99.0)):
    """Values -> rgba uint8 through a 256-entry LUT (NaN transparent)."""
    n = normalize(a, clip, 1.0, vmin, vmax)
    lut = np.asarray(lut)
    k = np.where(np.isfinite(n), np.round(np.nan_to_num(n) * (len(lut) - 1)), 0).astype(int)
    rgba = np.zeros(np.shape(n) + (4,), np.uint8)
    rgba[..., :lut.shape[1]] = lut[k][..., :lut.shape[1]]
    if lut.shape[1] == 3:
        rgba[..., 3] = 255
    rgba[..., 3] = np.where(np.isfinite(n), rgba[..., 3], 0)
    return rgba


def overlay(base_rgba, attr, lut, opacity=0.5, vmin=None, vmax=None, clip=(1.0, 99.0), threshold=None):
    """Attribute (through lut) alpha-blended over a base rgba image.  threshold: only values >= threshold (in
    attribute units) are overlaid - e.g. show only high sweetness / low coherence over the seismic."""
    base = np.asarray(base_rgba, float)
    top = lut_rgba(attr, lut, vmin, vmax, clip).astype(float)
    a = top[..., 3:4] / 255.0 * float(opacity)
    if threshold is not None:
        a = a * (np.nan_to_num(np.asarray(attr, float), nan=-np.inf) >= threshold)[..., None]
    out = base.copy()
    out[..., :3] = base[..., :3] * (1 - a) + top[..., :3] * a
    out[..., 3] = np.maximum(base[..., 3], a[..., 0] * 255)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def gray_rgba(a, clip=(1.0, 99.0), symmetric=True):
    """Seismic as grey rgba (symmetric clip about 0 by default)."""
    a = np.asarray(a, float)
    if symmetric:
        fin = np.isfinite(a)
        m = float(np.percentile(np.abs(a[fin]), clip[1])) if fin.any() else 1.0
        n = normalize(a, clip, 1.0, -m, m)
    else:
        n = normalize(a, clip)
    return to_rgba(n, n, n)


def illuminate(z, dx=1.0, dy=1.0, azimuth=315.0, elevation=35.0, ve=1.0):
    """Single light source Lambert shading 0..1 (NaN where z is NaN): cos of the angle between the surface normal
    and the light (azimuth clockwise from north = +rows, elevation above horizon), negative -> 0.  Rows are +y,
    columns +x.  Use ve to exaggerate (structure in ms over cells in m usually wants ve 2-10)."""
    z = np.asarray(z, float) * float(ve)
    fin = np.isfinite(z)
    zf = np.where(fin, z, np.nanmean(z) if fin.any() else 0.0)
    if zf.shape[0] < 2 or zf.shape[1] < 2:
        return np.where(fin, math.sin(math.radians(elevation)), np.nan)
    gy, gx = np.gradient(zf, abs(dy), abs(dx))
    if dy < 0:
        gy = -gy
    if dx < 0:
        gx = -gx
    az, el = math.radians(azimuth), math.radians(elevation)
    L = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])  # (east, north, up)
    norm = np.sqrt(gx ** 2 + gy ** 2 + 1.0)
    s = (-gx * L[0] - gy * L[1] + L[2]) / norm
    return np.where(fin, np.clip(s, 0, 1), np.nan)


def shade_rgba(rgba, shade, strength=0.6):
    """Darken / brighten rgb by a 0..1 shade: rgb * ((1 - s) + s * 1.25 * shade)."""
    out = np.asarray(rgba, float).copy()
    f = (1 - strength) + strength * 1.25 * np.nan_to_num(np.asarray(shade, float), nan=0.8)
    out[..., :3] *= f[..., None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def amplitude_on_structure(amp, struct, lut, dx=1.0, dy=1.0, azimuth=315.0, elevation=35.0, ve=1.0, strength=0.6,
                           vmin=None, vmax=None, clip=(1.0, 99.0), opacity=1.0):
    """Colour = amplitude (lut), shading = illuminated structure grid (same shape)."""
    rgba = lut_rgba(amp, lut, vmin, vmax, clip)
    sh = illuminate(struct, dx, dy, azimuth, elevation, ve)
    out = shade_rgba(rgba, sh, strength)
    out[..., 3] = (out[..., 3].astype(float) * opacity * np.isfinite(np.asarray(struct, float))).astype(np.uint8)
    return out


# ---------------------------------------------------------------- grids on a common geometry (map hub)
def on_grid(target, other):
    """Values of MapGrid `other` at the nodes of MapGrid `target` (bilinear, NaN outside); fast paths for the same
    survey grid and for xy grids."""
    if other is target:
        return np.asarray(other.z, float)
    if (other.kind == target.kind == "ilxl" and other.vol is target.vol and other.z.shape == target.z.shape):
        return np.asarray(other.z, float)
    if (other.kind == target.kind == "xy" and other.z.shape == target.z.shape and
            np.allclose([other.x0, other.y0, other.dx, other.dy], [target.x0, target.y0, target.dx, target.dy])):
        return np.asarray(other.z, float)
    X, Y = target.node_xy()
    X, Y = np.asarray(X, float).ravel(), np.asarray(Y, float).ravel()
    if other.kind == "ilxl":
        v = other.vol
        a, b = v.ax, v.ay
        M = np.linalg.inv(np.array([[a[1], a[2]], [b[1], b[2]]]))
        il = M[0, 0] * (X - a[0]) + M[0, 1] * (Y - b[0])
        xl = M[1, 0] * (X - a[0]) + M[1, 1] * (Y - b[0])
        fi, fj = (il - v.il0) / v.il_step, (xl - v.xl0) / v.xl_step
    else:
        fj, fi = (X - other.x0) / other.dx, (Y - other.y0) / other.dy
    ny, nx = other.z.shape
    eps = 1e-6
    ok = (fi >= -eps) & (fi <= ny - 1 + eps) & (fj >= -eps) & (fj <= nx - 1 + eps)
    i0 = np.clip(np.floor(fi).astype(int), 0, max(0, ny - 2))
    j0 = np.clip(np.floor(fj).astype(int), 0, max(0, nx - 2))
    wi, wj = np.clip(fi - i0, 0, 1), np.clip(fj - j0, 0, 1)
    i1, j1 = np.minimum(i0 + 1, ny - 1), np.minimum(j0 + 1, nx - 1)
    z = np.asarray(other.z, float)
    val = (z[i0, j0] * (1 - wi) * (1 - wj) + z[i0, j1] * (1 - wi) * wj + z[i1, j0] * wi * (1 - wj) + z[i1, j1] * wi * wj)
    return np.where(ok, val, np.nan).reshape(target.z.shape)


# ---------------------------------------------------------------- sections / slices
def section_blend(vol, names, mode, number, blend_mode="rgb", clip=(2.0, 98.0), gamma=1.0, params=None):
    """Co-blend of 2-3 attributes (attributes.ATTRIBUTES names, e.g. spectral 10 / 20 / 40 Hz) on an inline,
    crossline or time slice of a 3D volume -> (rgba, [arrays])."""
    from .attributes import section_attribute
    arrs = [np.asarray(section_attribute(vol, n, mode, number, (params or {}).get(n)), float) for n in names]
    return rgb_blend(arrs, blend_mode, clip, gamma), arrs


def spectral_blend(x, dt_ms, freqs=(10.0, 20.0, 40.0), mode="rgb", clip=(2.0, 98.0), gamma=1.0, cycles=3.0):
    """Spectral-decomposition RGB of a 2D section / slice array x (n, ns) -> rgba (ns, n, 4) via
    attributes.spectral (Morlet magnitude)."""
    from .attributes import spectral
    arrs = [np.asarray(spectral(np.asarray(x, np.float32), dt_ms, f, cycles), float).T for f in freqs]
    return rgb_blend(arrs, mode, clip, gamma)


def bump_section(arr, azimuth=315.0, elevation=45.0, ve=1.0, strength=0.6, clip=(1.0, 99.0)):
    """Bump-mapped seismic section / time slice: grey amplitude shaded by its own illuminated relief."""
    a = np.asarray(arr, float)
    sd = np.nanstd(a) or 1.0
    sh = illuminate(a / sd, 1.0, 1.0, azimuth, elevation, ve)
    return shade_rgba(gray_rgba(a, clip), sh, strength)


def param_scan(fn, values):
    """Attribute-parameter 'movie': yields (value, fn(value)) for each value - the GUI slider steps through the
    frames (cached), tests just iterate."""
    for v in values:
        yield v, fn(v)
