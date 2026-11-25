"""3D viewer scene (PLAN 33) - everything except the OpenGL draw calls, pure numpy and unit-tested.

World coordinates: x = easting, y = northing (m), z = -TWT (ms) * zscale * vertical exaggeration (z up), so map
units and time share one frame; `zscale` (m per ms) defaults to v / 2000 with v = 2500 m/s.

Meshes are flat vertex arrays ready for one simple shader (position + RGBA per vertex):
    Mesh(kind 'triangles' | 'lines', pos (n, 3) float32, rgba (n, 4) float32 in 0..1)
built by: `slice_mesh` (inline / crossline / time-slice planes, read on demand and decimated while dragging),
`curtain_mesh` (2D lines as vertical curtains), `surface_mesh` (horizon grids; NaN holes and fault jumps left open),
`fault_mesh` (item-2 fault triangles), `tube_mesh` (well paths, deviated or vertical, coloured by a log),
`disc_mesh` (tops), `polyline_mesh` (culture / grids draped at a level or on a surface), `box_mesh` (axis box +
tick positions).  `Camera` = orbit / pan / zoom / vertical exaggeration, view + projection matrices, picking rays
and ray-plane intersection (for 3D picking, item 34).  `Scene` = object list (visibility, opacity, colour bar) with
its state saved in project.display['view3d'], and a matplotlib renderer (`render_png`) for screenshots without a GPU,
tests and turntable GIFs.
"""
import math

import numpy as np

DEFAULT_ZSCALE = 1.25  # m per ms (2500 m/s two-way)


# ====================================================================== meshes
class Mesh:
    def __init__(self, kind, pos, rgba, name="", meta=None):
        self.kind = kind
        self.pos = np.ascontiguousarray(pos, np.float32).reshape(-1, 3)
        self.rgba = np.ascontiguousarray(rgba, np.float32).reshape(-1, 4)
        if len(self.rgba) == 1 and len(self.pos) > 1:
            self.rgba = np.repeat(self.rgba, len(self.pos), 0)
        self.name = name
        self.meta = meta or {}

    def __len__(self):
        return len(self.pos)

    def bounds(self):
        if not len(self.pos):
            return None
        return self.pos.min(axis=0), self.pos.max(axis=0)

    def interleaved(self):
        """(n, 7) float32 x y z r g b a - one VBO for the shader."""
        return np.ascontiguousarray(np.hstack([self.pos, self.rgba]), np.float32)


def lut_rgba(values, lut, vmin, vmax):
    """Values -> (n, 4) float RGBA through a 256-entry uint8 LUT; NaN -> transparent."""
    v = np.asarray(values, float)
    t = np.clip((v - vmin) / (vmax - vmin if vmax > vmin else 1.0), 0, 1)
    idx = np.where(np.isfinite(t), np.round(t * 255), 0).astype(int)
    c = np.asarray(lut, np.float32)[idx] / 255.0
    c[~np.isfinite(v), 3] = 0.0
    return c


def grid_triangles(ny, nx, valid=None):
    """Two triangles per grid cell whose 4 corners are valid: (m, 3) vertex indices into the flattened grid."""
    I, J = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    a = (I * nx + J).ravel()
    b, c, d = a + 1, a + nx, a + nx + 1
    tri = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])
    if valid is not None:
        v = np.asarray(valid).ravel()
        tri = tri[v[tri].all(axis=1)]
    return tri


def _expand(tri, pos, rgba):
    return pos[tri.ravel()], rgba[tri.ravel()]


def world_z(t_ms, zscale=DEFAULT_ZSCALE, ve=1.0):
    return -np.asarray(t_ms, float) * zscale * ve


def slice_mesh(vol, mode, number, lut, clip, decimate=1, zscale=DEFAULT_ZSCALE, ve=1.0, data=None):
    """Plane through the volume: mode 'Inline' | 'Crossline' | 'Time slice', number = il / xl / ms.  Read on demand
    from the store (one line or one slice), decimated by `decimate` (drag: 4, release: 1).  clip = (vmin, vmax)."""
    st = max(1, int(decimate))
    times = np.asarray(vol.times, float)
    if mode == "Time slice":
        sl = np.asarray(data if data is not None else vol.time_slice(number), float) * (1.0 if data is not None else 1.0)
        sl = sl[::st, ::st]
        ils = np.asarray(vol.inlines, float)[::st]
        xls = np.asarray(vol.xlines, float)[::st]
        I, J = np.meshgrid(ils, xls, indexing="ij")
        X, Y = vol.xy(I, J)
        Z = np.full(X.shape, world_z(number, zscale, ve))
        vals = sl
    else:
        if mode == "Inline":
            sec = np.asarray(data if data is not None else vol.inline(number), float)  # (nxl, ns)
            xls = np.asarray(vol.xlines, float)[::st]
            x, y = vol.xy(np.full(len(xls), float(number)), xls)
        else:
            sec = np.asarray(data if data is not None else vol.xline(number), float)  # (nil, ns)
            ils = np.asarray(vol.inlines, float)[::st]
            x, y = vol.xy(ils, np.full(len(ils), float(number)))
        sec = sec[::st, ::st]
        tt = times[::st]
        X = np.repeat(np.asarray(x, float)[:, None], len(tt), 1)
        Y = np.repeat(np.asarray(y, float)[:, None], len(tt), 1)
        Z = np.repeat(world_z(tt, zscale, ve)[None, :], len(x), 0)
        vals = sec
    ny, nx = vals.shape
    pos = np.stack([np.ravel(X), np.ravel(Y), np.ravel(Z)], 1)
    rgba = lut_rgba(np.ravel(vals), lut, clip[0], clip[1])
    tri = grid_triangles(ny, nx, np.isfinite(np.ravel(vals)))
    p, c = _expand(tri, pos, rgba)
    return Mesh("triangles", p, c, f"{mode} {number:g}", {"mode": mode, "number": float(number), "decimate": st})


def plane_corners(vol, mode, number, zscale=DEFAULT_ZSCALE, ve=1.0):
    """Four world corners of a slice plane (outline drawn while dragging, picking)."""
    t0, t1 = float(vol.times[0]), float(vol.times[-1])
    if mode == "Inline":
        a = vol.xy(float(number), float(vol.xlines[0]))
        b = vol.xy(float(number), float(vol.xlines[-1]))
        return np.array([[a[0], a[1], world_z(t0, zscale, ve)], [b[0], b[1], world_z(t0, zscale, ve)],
                         [b[0], b[1], world_z(t1, zscale, ve)], [a[0], a[1], world_z(t1, zscale, ve)]], float)
    if mode == "Crossline":
        a = vol.xy(float(vol.inlines[0]), float(number))
        b = vol.xy(float(vol.inlines[-1]), float(number))
        return np.array([[a[0], a[1], world_z(t0, zscale, ve)], [b[0], b[1], world_z(t0, zscale, ve)],
                         [b[0], b[1], world_z(t1, zscale, ve)], [a[0], a[1], world_z(t1, zscale, ve)]], float)
    xs, ys = vol.outline()
    z = world_z(number, zscale, ve)
    return np.array([[xs[k], ys[k], z] for k in range(4)], float)


def curtain_mesh(x, y, times, data, lut, clip, decimate=1, zscale=DEFAULT_ZSCALE, ve=1.0, name=""):
    """2D line as a vertical curtain along its navigation (data (ntr, ns))."""
    st = max(1, int(decimate))
    d = np.asarray(data, float)[::st, ::st]
    xx, yy = np.asarray(x, float)[::st], np.asarray(y, float)[::st]
    tt = np.asarray(times, float)[::st]
    X = np.repeat(xx[:, None], len(tt), 1)
    Y = np.repeat(yy[:, None], len(tt), 1)
    Z = np.repeat(world_z(tt, zscale, ve)[None, :], len(xx), 0)
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    tri = grid_triangles(*d.shape, np.isfinite(d.ravel()))
    p, c = _expand(tri, pos, lut_rgba(d.ravel(), lut, clip[0], clip[1]))
    return Mesh("triangles", p, c, name)


def surface_mesh(X, Y, T, lut, vrange=None, values=None, max_jump=None, zscale=DEFAULT_ZSCALE, ve=1.0, name="",
                 decimate=1):
    """Horizon (or any grid) as a triangulated surface: X, Y map (ny, nx), T ms; coloured by `values` (default T).
    Cells with a NaN corner stay open (holes), and so do cells whose corner times differ by more than max_jump ms
    (fault throws) - a surface never bridges a fault."""
    st = max(1, int(decimate))
    X, Y, T = (np.asarray(a, float)[::st, ::st] for a in (X, Y, T))
    V = T if values is None else np.asarray(values, float)[::st, ::st]
    ny, nx = T.shape
    valid = np.isfinite(T) & np.isfinite(X) & np.isfinite(Y)
    tri = grid_triangles(ny, nx, valid.ravel())
    if max_jump is not None and len(tri):
        tz = T.ravel()[tri]
        tri = tri[(np.nanmax(tz, 1) - np.nanmin(tz, 1)) <= max_jump]
    pos = np.stack([X.ravel(), Y.ravel(), world_z(T.ravel(), zscale, ve)], 1)
    fin = V[np.isfinite(V)]
    lo, hi = vrange or ((float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0))
    p, c = _expand(tri, pos, lut_rgba(V.ravel(), lut, lo, hi))
    return Mesh("triangles", p, c, name, {"vrange": (lo, hi), "triangles": len(tri)})


def fault_mesh(tris, color=(200, 0, 0), alpha=0.8, zscale=DEFAULT_ZSCALE, ve=1.0, name=""):
    """Item-2 fault triangles [(3, 3) map x, y, t ms] -> mesh."""
    if not len(tris):
        return Mesh("triangles", np.zeros((0, 3)), np.zeros((0, 4)), name)
    a = np.asarray(tris, float).reshape(-1, 3)
    pos = np.column_stack([a[:, 0], a[:, 1], world_z(a[:, 2], zscale, ve)])
    c = np.array(list(np.asarray(color[:3], float) / 255.0) + [alpha], np.float32)
    return Mesh("triangles", pos, np.tile(c, (len(pos), 1)), name)


def _frames(path):
    """Unit tangent + two normals along a polyline (rotation-minimising: parallel transport)."""
    p = np.asarray(path, float)
    d = np.gradient(p, axis=0)
    t = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(t[0, 2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    n = np.cross(t[0], ref)
    n /= np.linalg.norm(n) + 1e-12
    N = np.zeros_like(p)
    for k in range(len(p)):
        n = n - np.dot(n, t[k]) * t[k]
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            n = np.cross(t[k], ref)
            ln = np.linalg.norm(n) + 1e-12
        n = n / ln
        N[k] = n
    B = np.cross(t, N)
    return t, N, B


def tube_mesh(path, radius, rgba_per_point, sides=8, name=""):
    """Tube along a 3D world path (well bore); rgba_per_point (m, 4) colours it (a log, or one colour)."""
    p = np.asarray(path, float)
    m = len(p)
    if m < 2:
        return Mesh("triangles", np.zeros((0, 3)), np.zeros((0, 4)), name)
    _, N, B = _frames(p)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = p[:, None, :] + radius * (np.cos(ang)[None, :, None] * N[:, None, :] + np.sin(ang)[None, :, None] * B[:, None, :])
    pos = ring.reshape(-1, 3)
    col = np.repeat(np.asarray(rgba_per_point, np.float32).reshape(-1, 4) if len(np.shape(rgba_per_point)) > 1
                    else np.tile(np.asarray(rgba_per_point, np.float32), (m, 1)), sides, 0)
    tris = []
    for k in range(m - 1):
        for s in range(sides):
            a, b = k * sides + s, k * sides + (s + 1) % sides
            c, d = a + sides, b + sides
            tris += [(a, c, b), (b, c, d)]
    tri = np.array(tris, int)
    pp, cc = _expand(tri, pos, col)
    return Mesh("triangles", pp, cc, name, {"radius": radius, "points": m})


def disc_mesh(center, normal, radius, rgba, sides=16, name=""):
    """Filled disc (a well top) centred on `center`, perpendicular to `normal`."""
    c = np.asarray(center, float)
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    ang = np.linspace(0, 2 * np.pi, sides + 1)
    rim = c + radius * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v)
    pos = np.concatenate([[c, rim[k], rim[k + 1]] for k in range(sides)])
    return Mesh("triangles", pos, np.tile(np.asarray(rgba, np.float32), (len(pos), 1)), name)


def polyline_mesh(xs, ys, zs, rgba, name=""):
    """Line segments (GL_LINES pairs) of a polyline; NaN breaks the line."""
    p = np.column_stack([np.asarray(xs, float), np.asarray(ys, float), np.asarray(zs, float)])
    a, b = p[:-1], p[1:]
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    seg = np.stack([a[ok], b[ok]], 1).reshape(-1, 3)
    return Mesh("lines", seg, np.tile(np.asarray(rgba, np.float32), (len(seg), 1)), name)


def drape(xs, ys, grid=None, level_ms=None, zscale=DEFAULT_ZSCALE, ve=1.0, lift_ms=2.0):
    """z for culture lines / grids draped on a surface (MapGrid of times, sampled) or at a fixed level (ms)."""
    if grid is not None:
        t = np.asarray(grid.sample(np.asarray(xs, float), np.asarray(ys, float)), float) - lift_ms
    else:
        t = np.full(len(xs), float(level_ms or 0.0))
    return world_z(t, zscale, ve)


def well_path_world(traj, td=None, zscale=DEFAULT_ZSCALE, ve=1.0, domain="time"):
    """World path of a laterals.Trajectory (x, y, z TVDSS): in time through a well T-D (td.twt(z)), else depth."""
    z = np.asarray(traj.z, float)
    if domain == "time" and td is not None:
        zz = world_z(td.twt(z), zscale, ve)
    else:
        zz = -z * ve
    return np.column_stack([np.asarray(traj.x, float), np.asarray(traj.y, float), zz])


def box_mesh(vol, zscale=DEFAULT_ZSCALE, ve=1.0, rgba=(0.8, 0.8, 0.8, 1.0), ticks=5):
    """Survey bounding box (12 edges) + tick label positions [(xyz, text)] for IL / XL / time."""
    xs, ys = vol.outline()
    t0, t1 = float(vol.times[0]), float(vol.times[-1])
    z0, z1 = world_z(t0, zscale, ve), world_z(t1, zscale, ve)
    c = [(xs[k], ys[k]) for k in range(4)]
    seg = []
    for k in range(4):
        a, b = c[k], c[(k + 1) % 4]
        seg += [(a[0], a[1], z0), (b[0], b[1], z0), (a[0], a[1], z1), (b[0], b[1], z1), (a[0], a[1], z0),
                (a[0], a[1], z1)]
    m = Mesh("lines", np.array(seg), np.tile(np.asarray(rgba, np.float32), (len(seg), 1)), "box")
    labels = []
    for il in np.linspace(vol.inlines[0], vol.inlines[-1], ticks):
        x, y = vol.xy(float(il), float(vol.xlines[0]))
        labels.append(((float(x), float(y), z0), f"IL {il:.0f}"))
    for xl in np.linspace(vol.xlines[0], vol.xlines[-1], ticks):
        x, y = vol.xy(float(vol.inlines[0]), float(xl))
        labels.append(((float(x), float(y), z0), f"XL {xl:.0f}"))
    for t in np.linspace(t0, t1, ticks):
        labels.append(((float(c[0][0]), float(c[0][1]), world_z(t, zscale, ve)), f"{t:.0f} ms"))
    return m, labels


# ====================================================================== camera
def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-9:
        s = np.cross(f, np.array([0.0, 1.0, 0.0]))
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    M = np.zeros((4, 4))
    M[0, 0] = f / max(aspect, 1e-9)
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = 2 * far * near / (near - far)
    M[3, 2] = -1.0
    return M


class Camera:
    """Orbit camera around a target: azimuth (deg from north, clockwise), elevation (deg), distance."""

    def __init__(self, target=(0.0, 0.0, 0.0), distance=1000.0, azimuth=225.0, elevation=30.0, fov=35.0):
        self.target = np.asarray(target, float)
        self.distance, self.azimuth, self.elevation, self.fov = float(distance), float(azimuth), float(elevation), fov

    def eye(self):
        a, e = math.radians(self.azimuth), math.radians(self.elevation)
        d = np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)])
        return self.target + self.distance * d

    def view(self):
        return look_at(self.eye(), self.target)

    def projection(self, aspect):
        return perspective(self.fov, aspect, max(self.distance * 0.01, 1e-3), self.distance * 20.0)

    def mvp(self, aspect):
        return self.projection(aspect) @ self.view()

    def orbit(self, dx_px, dy_px, deg_per_px=0.4):
        self.azimuth = (self.azimuth + dx_px * deg_per_px) % 360.0
        self.elevation = float(np.clip(self.elevation + dy_px * deg_per_px, -89.0, 89.0))

    def pan(self, dx_px, dy_px, height_px):
        """Move the target in the view plane (drag = the scene follows the mouse)."""
        V = self.view()
        right, up = V[0, :3], V[1, :3]
        k = 2.0 * self.distance * math.tan(math.radians(self.fov) / 2.0) / max(height_px, 1)
        self.target = self.target - right * dx_px * k + up * dy_px * k

    def zoom(self, factor):
        self.distance = float(np.clip(self.distance / max(factor, 1e-6), 1e-2, 1e9))

    def fit(self, lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        self.target = 0.5 * (lo + hi)
        r = 0.5 * float(np.linalg.norm(hi - lo))
        self.distance = max(r / math.sin(math.radians(self.fov) / 2.0) * 1.1, 1.0)

    def project(self, pts, w, h):
        """World points -> pixel coordinates (x right, y down) + depth; for labels and tests."""
        p = np.c_[np.asarray(pts, float).reshape(-1, 3), np.ones(len(np.asarray(pts).reshape(-1, 3)))]
        c = (self.mvp(w / max(h, 1)) @ p.T).T
        ndc = c[:, :3] / c[:, 3:4]
        return np.column_stack([(ndc[:, 0] + 1) * 0.5 * w, (1 - ndc[:, 1]) * 0.5 * h, ndc[:, 2]])

    def ray(self, px, py, w, h):
        """World ray (origin, unit direction) through pixel (px, py)."""
        inv = np.linalg.inv(self.mvp(w / max(h, 1)))
        x, y = 2.0 * px / w - 1.0, 1.0 - 2.0 * py / h
        a = inv @ np.array([x, y, -1.0, 1.0])
        b = inv @ np.array([x, y, 1.0, 1.0])
        a, b = a[:3] / a[3], b[:3] / b[3]
        d = b - a
        return a, d / np.linalg.norm(d)

    def to_dict(self):
        return {"target": self.target.tolist(), "distance": self.distance, "azimuth": self.azimuth,
                "elevation": self.elevation, "fov": self.fov}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("target", (0, 0, 0)), d.get("distance", 1000.0), d.get("azimuth", 225.0),
                   d.get("elevation", 30.0), d.get("fov", 35.0))


def ray_plane(origin, direction, p0, normal):
    """Intersection point of a ray with a plane (None when parallel or behind)."""
    n = np.asarray(normal, float)
    den = float(np.dot(direction, n))
    if abs(den) < 1e-12:
        return None
    s = float(np.dot(np.asarray(p0, float) - origin, n)) / den
    return None if s < 0 else origin + s * np.asarray(direction, float)


def pick_on_plane(cam, px, py, w, h, vol, mode, number, zscale=DEFAULT_ZSCALE, ve=1.0):
    """Pixel -> (il, xl, t ms) on a slice plane, or None outside it (34a uses this)."""
    c = plane_corners(vol, mode, number, zscale, ve)
    n = np.cross(c[1] - c[0], c[3] - c[0])
    o, d = cam.ray(px, py, w, h)
    p = ray_plane(o, d, c[0], n)
    if p is None:
        return None
    il, xl = vol.ilxl_at(p[0], p[1])
    t = -p[2] / (zscale * ve)
    il, xl = float(np.ravel(il)[0]), float(np.ravel(xl)[0])
    if mode == "Inline":
        il = float(number)
    elif mode == "Crossline":
        xl = float(number)
    else:
        t = float(number)
    inside = (min(vol.inlines[0], vol.inlines[-1]) - 0.5 <= il <= max(vol.inlines[0], vol.inlines[-1]) + 0.5 and
              min(vol.xlines[0], vol.xlines[-1]) - 0.5 <= xl <= max(vol.xlines[0], vol.xlines[-1]) + 0.5 and
              vol.times[0] - 1e-6 <= t <= vol.times[-1] + 1e-6)
    return (il, xl, t) if inside else None


# ====================================================================== scene
class SceneObject:
    def __init__(self, oid, kind, name, meshes, visible=True, opacity=1.0, colorbar=None, src=None):
        self.id, self.kind, self.name = oid, kind, name
        self.meshes = list(meshes)
        self.visible, self.opacity = bool(visible), float(opacity)
        self.colorbar = colorbar  # (lut name, vmin, vmax, unit) or None
        self.src = dict(src or {})
        self.version = 0  # bumped when the meshes change (the GL layer re-uploads)

    def set_meshes(self, meshes):
        self.meshes = list(meshes)
        self.version += 1


class Scene:
    KEY = "view3d"

    def __init__(self, zscale=DEFAULT_ZSCALE, ve=1.0):
        self.objects = []
        self.zscale, self.ve = float(zscale), float(ve)
        self.camera = Camera()
        self.views = {}
        self._n = 0
        self.background = (0.09, 0.10, 0.12)

    def add(self, kind, name, meshes, **kw):
        self._n += 1
        o = SceneObject(f"o{self._n}", kind, name, meshes, **kw)
        self.objects.append(o)
        return o

    def replace(self, kind, name, meshes, **kw):
        o = self.find(kind, name)
        if o is None:
            return self.add(kind, name, meshes, **kw)
        o.set_meshes(meshes)
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    def find(self, kind, name=None):
        return next((o for o in self.objects if o.kind == kind and (name is None or o.name == name)), None)

    def get(self, oid):
        return next((o for o in self.objects if o.id == oid), None)

    def remove(self, oid):
        self.objects = [o for o in self.objects if o.id != oid]

    def set_visible(self, oid, v):
        self.get(oid).visible = bool(v)

    def set_opacity(self, oid, a):
        self.get(oid).opacity = float(np.clip(a, 0, 1))

    def bounds(self):
        lo, hi = None, None
        for o in self.objects:
            if not o.visible:  # Fit frames what is shown (a hidden far-away 2D line must not shrink the view)
                continue
            for m in o.meshes:
                b = m.bounds()
                if b is None:
                    continue
                lo = b[0] if lo is None else np.minimum(lo, b[0])
                hi = b[1] if hi is None else np.maximum(hi, b[1])
        return lo, hi

    def fit(self):
        lo, hi = self.bounds()
        if lo is not None:
            self.camera.fit(lo, hi)

    def draw_list(self):
        """[(mesh, opacity)] of visible objects, opaque first then transparent (painter order for blending)."""
        out = [(m, o.opacity) for o in self.objects if o.visible for m in o.meshes if len(m)]
        return sorted(out, key=lambda q: q[1] < 0.999)

    def save_view(self, name):
        self.views[str(name)] = self.camera.to_dict()
        return sorted(self.views)

    def restore_view(self, name):
        self.camera = Camera.from_dict(self.views[name])
        return self.camera

    def state(self):
        return {"camera": self.camera.to_dict(), "views": self.views, "ve": self.ve, "zscale": self.zscale,
                "objects": [{"kind": o.kind, "name": o.name, "visible": o.visible, "opacity": o.opacity,
                             "src": o.src} for o in self.objects]}

    def save(self, project):
        project.display[self.KEY] = self.state()
        return project.display[self.KEY]

    @classmethod
    def load_state(cls, project):
        d = (getattr(project, "display", {}) or {}).get(cls.KEY) or {}
        s = cls(d.get("zscale", DEFAULT_ZSCALE), d.get("ve", 1.0))
        if d.get("camera"):
            s.camera = Camera.from_dict(d["camera"])
        s.views = dict(d.get("views") or {})
        return s, list(d.get("objects") or [])

    # ------------------------------------------------------------------ software rendering (no GPU)
    def render_png(self, path, size=(900, 700), max_tris=60000, dpi=100, title=None, legend=None):
        """Screenshot without OpenGL (matplotlib, painter's algorithm on projected triangles).  Used for the
        fallback view, tests and turntable GIF frames."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection, PolyCollection
        w, h = size
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        fig.patch.set_facecolor(self.background)
        polys, cols, depth = [], [], []
        for m, op in self.draw_list():
            if m.kind == "triangles":
                n = len(m.pos) // 3
                step = max(1, n // max_tris)
                P = self.camera.project(m.pos, w, h).reshape(-1, 3, 3)[::step]
                C = m.rgba.reshape(-1, 3, 4)[::step].mean(axis=1)
                C[:, 3] *= op
                ok = np.all(np.abs(P[:, :, 2]) <= 1.0, axis=1) & (C[:, 3] > 0)
                polys.append(P[ok][:, :, :2])
                cols.append(C[ok])
                depth.append(P[ok][:, :, 2].mean(axis=1))
            else:
                P = self.camera.project(m.pos, w, h).reshape(-1, 2, 3)
                ok = np.all(np.abs(P[:, :, 2]) <= 1.0, axis=1)
                ax.add_collection(LineCollection(P[ok][:, :, :2], colors=m.rgba[::2][ok], linewidths=1.0, zorder=3))
        if polys:
            P = np.concatenate(polys)
            C = np.clip(np.concatenate(cols), 0, 1)
            D = np.concatenate(depth)
            o = np.argsort(-D)  # far first
            ax.add_collection(PolyCollection(P[o], facecolors=C[o], edgecolors="none", zorder=2))
        if title:
            ax.text(10, 20, title, color="w", fontsize=11)
        for k, (lab, c) in enumerate(legend or []):
            ax.text(10, 45 + 16 * k, "■ " + str(lab), color=tuple(np.asarray(c, float)[:3]), fontsize=8)
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    def turntable(self, path, frames=24, size=(480, 360), duration_ms=120):
        """Animated GIF orbiting the scene (PIL).  Restores the camera afterwards."""
        import os
        import tempfile
        from PIL import Image
        az0 = self.camera.azimuth
        tmp = tempfile.mkdtemp(prefix="turntable_")
        imgs = []
        try:
            for k in range(frames):
                self.camera.azimuth = (az0 + 360.0 * k / frames) % 360.0
                p = self.render_png(os.path.join(tmp, f"f{k:03d}.png"), size, max_tris=20000)
                imgs.append(Image.open(p).convert("P", palette=Image.ADAPTIVE))
            imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0)
        finally:
            self.camera.azimuth = az0
            for f in os.listdir(tmp):
                os.remove(os.path.join(tmp, f))
            os.rmdir(tmp)
        return path
