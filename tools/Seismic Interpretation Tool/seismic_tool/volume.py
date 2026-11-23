"""3D volumes and 2D lines.

3D SEG-Y is imported once into a 'store' folder holding two float16 copies of the data:
    data_ilxl.npy  shape (n_inline, n_xline, n_samples)   -> fast inlines and crosslines
    data_t.npy     shape (n_samples, n_inline, n_xline)   -> fast time slices / horizon extraction
    data_xl.npy    shape (n_xline, n_inline, n_samples)   -> fast crosslines (optional; +50% disk; older
                   stores without it read crosslines from data_ilxl, slower; add it with --xlstore STORE)
float16 halves the size; amplitudes are scaled so the 99.9th percentile is ~1000 (the scale is
stored and undone on read).  Together the two copies are about the size of a float32 SEG-Y.
Both are memory-mapped, so a 10 GB survey opens instantly and slices come straight off disk.
"""
import json
import math
import os
import time

import numpy as np

from .segy import SegyFile

STORE_VERSION = 1


def _affine(il, xl, x, y):
    """Least-squares x = a + b*il + c*xl, y = d + e*il + f*xl."""
    A = np.column_stack([np.ones_like(il, float), il, xl])
    cx = np.linalg.lstsq(A, x, rcond=None)[0]
    cy = np.linalg.lstsq(A, y, rcond=None)[0]
    return cx.tolist(), cy.tolist()


def _fix_degenerate(cx, cy, nil, nxl, il_step, xl_step, il0, xl0):
    """A single-inline (or single-crossline) survey leaves the other axis' coefficients undetermined by the
    least-squares fit: make that axis perpendicular to the varying one, same spacing per trace, keeping the fitted
    positions of the existing traces."""
    cx, cy = list(map(float, cx)), list(map(float, cy))
    if nil == 1 and nxl > 1:
        Cx, Cy = cx[0] + cx[1] * il0, cy[0] + cy[1] * il0  # constant part at the one inline
        ex, ey = cx[2] * xl_step, cy[2] * xl_step  # metres per crossline trace
        cx[1], cy[1] = -ey / il_step, ex / il_step
        cx[0], cy[0] = Cx - cx[1] * il0, Cy - cy[1] * il0
    elif nxl == 1 and nil > 1:
        Cx, Cy = cx[0] + cx[2] * xl0, cy[0] + cy[2] * xl0
        ex, ey = cx[1] * il_step, cy[1] * il_step
        cx[2], cy[2] = ey / xl_step, -ex / xl_step
        cx[0], cy[0] = Cx - cx[2] * xl0, Cy - cy[2] * xl0
    return cx, cy


def import_3d(segy_path, store_dir, il_byte=None, xl_byte=None, x_byte=181, y_byte=185, scalar_byte=71,
              chunk=20000, progress=None, cancel=None, workers=None, xline_store=True):
    """Scan headers, then write the two float16 stores. progress(fraction, text) and cancel() -> bool are
    optional; a cancelled import removes its partial store and raises RuntimeError('cancelled').
    workers: threads for the trace conversion and the time-slice copy (None = up to 4, 1 = serial)."""
    try:
        return _import_3d(segy_path, store_dir, il_byte, xl_byte, x_byte, y_byte, scalar_byte, chunk, progress, cancel,
                          workers, xline_store)
    except RuntimeError as e:
        if str(e) == "cancelled":
            for fn in ("data_ilxl.npy", "data_t.npy", "data_xl.npy", "filled.npy", "meta.json"):
                try:
                    os.remove(os.path.join(store_dir, fn))
                except OSError:
                    pass
        raise


def _check_cancel(cancel):
    if cancel and cancel():
        raise RuntimeError("cancelled")


def _workers(w):
    if w is None:
        w = min(4, os.cpu_count() or 1)
    return max(1, int(w))


def _import_3d(segy_path, store_dir, il_byte, xl_byte, x_byte, y_byte, scalar_byte, chunk, progress, cancel,
               workers=None, xline_store=True):
    f = SegyFile(segy_path)
    if il_byte is None or xl_byte is None:
        g = f.guess_il_xl()
        if g is None:
            raise ValueError("Could not find inline/crossline header bytes - set them in the import dialog")
        il_byte, xl_byte = g
    n = f.ntraces
    il = np.empty(n, np.int64)
    xl = np.empty(n, np.int64)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        il[s:e] = f.header(il_byte, 4, s, e)
        xl[s:e] = f.header(xl_byte, 4, s, e)
        _check_cancel(cancel)
        if progress:
            progress(0.1 * e / n, f"Reading headers {e:,}/{n:,}")
    ui, ux = np.unique(il), np.unique(xl)
    il_step = int(np.min(np.diff(ui))) if len(ui) > 1 else 1
    xl_step = int(np.min(np.diff(ux))) if len(ux) > 1 else 1
    il0, xl0 = int(ui[0]), int(ux[0])
    nil = (int(ui[-1]) - il0) // il_step + 1
    nxl = (int(ux[-1]) - xl0) // xl_step + 1
    ii = (il - il0) // il_step
    jj = (xl - xl0) // xl_step
    ns = f.ns
    # amplitude scale from a sample of traces
    pick = np.linspace(0, n - 1, min(n, 3000)).astype(int)
    samp = np.abs(f.traces(idx=pick))
    p999 = float(np.percentile(samp[np.isfinite(samp)], 99.9)) or 1.0
    scale = 1000.0 / p999
    os.makedirs(store_dir, exist_ok=True)
    ilxl = np.lib.format.open_memmap(os.path.join(store_dir, "data_ilxl.npy"), mode="w+", dtype=np.float16,
                                     shape=(nil, nxl, ns))
    filled = np.zeros((nil, nxl), bool)
    workers = _workers(workers)
    flat = ilxl.reshape(nil * nxl, ns)  # view on the same memmap
    lin = ii * nxl + jj  # trace position in the store

    def write_chunk(s, e):
        tr = f.traces(s, e)
        np.multiply(tr, scale, out=tr)
        bad = ~np.isfinite(tr)  # corrupt / dead samples (one pass; nan_to_num made three)
        if bad.any():
            tr[bad] = 0.0
        np.clip(tr, -60000, 60000, out=tr)
        k = lin[s:e]
        if e - s > 1 and k[-1] - k[0] == e - s - 1 and np.all(np.diff(k) == 1):
            flat[k[0]:k[-1] + 1] = tr  # sorted survey: one contiguous block write (casts to float16 in place)
        else:
            ilxl[ii[s:e], jj[s:e], :] = tr.astype(np.float16)  # unsorted / irregular: scattered write
        filled[ii[s:e], jj[s:e]] = True
        return e

    def run_parallel(jobs, fn, p0, p1, text):
        """Run fn(a, b) over jobs with `workers` threads (numpy releases the GIL in the conversions and
        copies); cancel is checked between chunks, progress in order of completion."""
        done = 0
        if workers <= 1:
            for a, b in jobs:
                fn(a, b)
                done += 1
                _check_cancel(cancel)
                if progress:
                    progress(p0 + (p1 - p0) * done / len(jobs), text(done, len(jobs)))
            return
        from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
        it = iter(jobs)
        with ThreadPoolExecutor(workers) as ex:
            pending = set()
            for _ in range(workers + 1):  # bounded queue: workers + 1 chunks in memory (<= ~64 MB each)
                job = next(it, None)
                if job is not None:
                    pending.add(ex.submit(fn, *job))
            while pending:
                fin, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fu in fin:
                    fu.result()  # re-raise worker errors
                    done += 1
                if cancel and cancel():
                    for fu in pending:
                        fu.cancel()
                    wait(pending)
                    raise RuntimeError("cancelled")
                if progress:
                    progress(p0 + (p1 - p0) * done / len(jobs), text(done, len(jobs)))
                for _ in fin:
                    job = next(it, None)
                    if job is not None:
                        pending.add(ex.submit(fn, *job))

    chunk = max(1, min(int(chunk), int(64e6 // max(1, ns * 4))))  # <= ~64 MB of float32 per chunk
    jobs = [(s, min(n, s + chunk)) for s in range(0, n, chunk)]
    try:
        run_parallel(jobs, write_chunk, 0.1, 0.7, lambda d, t: f"Writing traces {min(n, d * chunk):,}/{n:,}")
    except RuntimeError:
        del ilxl
        raise
    ilxl.flush()
    # time-major copy, built from blocks of inlines (contiguous reads, contiguous writes)
    tm = np.lib.format.open_memmap(os.path.join(store_dir, "data_t.npy"), mode="w+", dtype=np.float16,
                                   shape=(ns, nil, nxl))
    xm = np.lib.format.open_memmap(os.path.join(store_dir, "data_xl.npy"), mode="w+", dtype=np.float16,
                                   shape=(nxl, nil, ns)) if xline_store else None
    blk = max(1, int(64e6 // max(1, nxl * ns * 2)))

    def transpose_block(a, b):
        block = np.asarray(ilxl[a:b])  # one contiguous read serves both copies
        tm[:, a:b, :] = np.transpose(block, (2, 0, 1))
        if xm is not None:
            xm[:, a:b, :] = np.transpose(block, (1, 0, 2))

    try:
        run_parallel([(a, min(nil, a + blk)) for a in range(0, nil, blk)], transpose_block, 0.7, 1.0,
                     lambda d, t: f"Building time slices {d}/{t} blocks")
    except RuntimeError:
        del tm, ilxl, xm
        raise
    tm.flush()
    if xm is not None:
        xm.flush()
        del xm
    # coordinates
    x, y = f.coordinates(x_byte, y_byte, scalar_byte, 0, n)
    sel = np.linspace(0, n - 1, min(n, 20000)).astype(int)
    if np.ptp(x[sel]) > 0 or np.ptp(y[sel]) > 0:
        cx, cy = _affine(il[sel].astype(float), xl[sel].astype(float), x[sel], y[sel])
        cx, cy = _fix_degenerate(cx, cy, nil, nxl, il_step, xl_step, il0, xl0)
    else:  # no coordinates: use il/xl numbers as a nominal 25 m grid (coefficients are per header number)
        cx, cy = [0.0, 0.0, 25.0 / xl_step], [0.0, 25.0 / il_step, 0.0]
    np.save(os.path.join(store_dir, "filled.npy"), filled)
    meta = {"version": STORE_VERSION, "segy": os.path.abspath(segy_path), "il_byte": il_byte, "xl_byte": xl_byte,
            "il0": il0, "il_step": il_step, "nil": nil, "xl0": xl0, "xl_step": xl_step, "nxl": nxl,
            "ns": ns, "dt": f.dt, "t0": f.t0, "scale": scale, "affine_x": cx, "affine_y": cy,
            "ntraces": int(n), "fill_fraction": float(filled.mean()), "xline_store": bool(xline_store),
            "imported": time.strftime("%Y-%m-%d %H:%M"), "text_header": f.text_header[:3200]}
    with open(os.path.join(store_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    return Volume3D(store_dir)


def build_xline_store(store_dir, progress=None, cancel=None):
    """Add the crossline-major copy (data_xl.npy) to an existing store, so crosslines read as fast as inlines.
    Returns the path.  A cancelled build removes the partial file."""
    with open(os.path.join(store_dir, "meta.json")) as fh:
        m = json.load(fh)
    nil, nxl, ns = m["nil"], m["nxl"], m["ns"]
    ilxl = np.load(os.path.join(store_dir, "data_ilxl.npy"), mmap_mode="r")
    path = os.path.join(store_dir, "data_xl.npy")
    tmp = path + ".part.npy"
    xm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(nxl, nil, ns))
    blk = max(1, int(64e6 // max(1, nxl * ns * 2)))
    try:
        for a in range(0, nil, blk):
            b = min(nil, a + blk)
            xm[:, a:b, :] = np.transpose(np.asarray(ilxl[a:b]), (1, 0, 2))
            if cancel and cancel():
                raise RuntimeError("cancelled")
            if progress:
                progress(b / nil, f"Crossline copy {b}/{nil} inlines")
        xm.flush()
    except BaseException:
        del xm
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    del xm, ilxl
    os.replace(tmp, path)
    m["xline_store"] = True
    with open(os.path.join(store_dir, "meta.json"), "w") as fh:
        json.dump(m, fh, indent=1)
    return path


class Volume3D:
    kind = "3D"

    def __init__(self, store_dir):
        self.store = store_dir
        with open(os.path.join(store_dir, "meta.json")) as fh:
            self.meta = m = json.load(fh)
        self.ilxl = np.load(os.path.join(store_dir, "data_ilxl.npy"), mmap_mode="r")
        self.tdata = np.load(os.path.join(store_dir, "data_t.npy"), mmap_mode="r")
        xp = os.path.join(store_dir, "data_xl.npy")
        self.xldata = np.load(xp, mmap_mode="r") if os.path.exists(xp) else None
        self.il0, self.il_step, self.nil = m["il0"], m["il_step"], m["nil"]
        self.xl0, self.xl_step, self.nxl = m["xl0"], m["xl_step"], m["nxl"]
        self.ns, self.dt, self.t0 = m["ns"], m["dt"], m.get("t0", 0.0)
        self.inv = 1.0 / m["scale"]
        self.ax, self.ay = m["affine_x"], m["affine_y"]

    # ------------------------------------------------------------ geometry
    @property
    def inlines(self):
        return self.il0 + self.il_step * np.arange(self.nil)

    @property
    def xlines(self):
        return self.xl0 + self.xl_step * np.arange(self.nxl)

    @property
    def times(self):
        """Sample times in milliseconds."""
        return 1000.0 * (self.t0 + self.dt * np.arange(self.ns))

    def il_index(self, il):
        return int(np.clip(round((il - self.il0) / self.il_step), 0, self.nil - 1))

    def xl_index(self, xl):
        return int(np.clip(round((xl - self.xl0) / self.xl_step), 0, self.nxl - 1))

    def t_index(self, t_ms):
        return int(np.clip(round((t_ms / 1000.0 - self.t0) / self.dt), 0, self.ns - 1))

    def xy(self, il, xl):
        il, xl = np.asarray(il, float), np.asarray(xl, float)
        a, b = self.ax, self.ay
        return a[0] + a[1] * il + a[2] * xl, b[0] + b[1] * il + b[2] * xl

    def ilxl_at(self, x, y):
        """Inverse of xy (nearest inline / crossline numbers, unclipped)."""
        a, b = self.ax, self.ay
        M = np.array([[a[1], a[2]], [b[1], b[2]]])
        il, xl = np.linalg.solve(M, [x - a[0], y - b[0]])
        return il, xl

    def outline(self):
        ils = [self.inlines[0], self.inlines[-1], self.inlines[-1], self.inlines[0], self.inlines[0]]
        xls = [self.xlines[0], self.xlines[0], self.xlines[-1], self.xlines[-1], self.xlines[0]]
        return self.xy(ils, xls)

    # ------------------------------------------------------------ data
    def inline(self, il):
        """(n_xline, n_samples) float32."""
        return np.asarray(self.ilxl[self.il_index(il)], np.float32) * self.inv

    def xline(self, xl):
        if self.xldata is not None:  # crossline-major copy: one contiguous read
            return np.asarray(self.xldata[self.xl_index(xl)], np.float32) * self.inv
        return np.asarray(self.ilxl[:, self.xl_index(xl), :], np.float32) * self.inv

    def time_slice(self, t_ms):
        """(n_inline, n_xline) float32."""
        return np.asarray(self.tdata[self.t_index(t_ms)], np.float32) * self.inv

    def trace(self, il, xl):
        return np.asarray(self.ilxl[self.il_index(il), self.xl_index(xl)], np.float32) * self.inv

    def arbitrary(self, xs, ys, spacing=None):
        """Section along a polyline in map coordinates; returns (data (n, ns), il, xl) by nearest trace."""
        pts = []
        for (x0, y0), (x1, y1) in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
            d = math.hypot(x1 - x0, y1 - y0)
            n = max(2, int(d / (spacing or 25.0)))
            for t in np.linspace(0, 1, n, endpoint=False):
                pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
        pts.append((xs[-1], ys[-1]))
        ilxl = [self.ilxl_at(x, y) for x, y in pts]
        ii = np.array([self.il_index(i) for i, _ in ilxl])
        jj = np.array([self.xl_index(j) for _, j in ilxl])
        return np.asarray(self.ilxl[ii, jj], np.float32) * self.inv, self.inlines[ii], self.xlines[jj]


class Line2D:
    kind = "2D"

    def __init__(self, name, segy_path, cdp_byte=21, nav=None, x_byte=181, y_byte=185, scalar_byte=71):
        self.name = name
        self.path = segy_path
        self.f = SegyFile(segy_path)
        self.ns, self.dt, self.t0 = self.f.ns, self.f.dt, self.f.t0
        cdp = self.f.header(cdp_byte, 4).astype(float)
        d = np.diff(cdp)
        if not (np.all(d > 0) or np.all(d < 0)):  # descending CDP numbers are common and fine
            cdp = np.arange(1, self.f.ntraces + 1, dtype=float)
        self.cdp = cdp
        x, y = self.f.coordinates(x_byte, y_byte, scalar_byte)
        if nav is not None and len(nav):  # nav array [cdp, x, y] sorted by cdp
            x = np.interp(cdp, nav[:, 0], nav[:, 1])
            y = np.interp(cdp, nav[:, 0], nav[:, 2])
        self.x, self.y = x, y
        self.has_coords = bool(np.hypot(np.ptp(x), np.ptp(y)) > 1.0)
        self._data = None
        self._raw = None
        self.correction = None  # (shift_ms, phase_deg, gain) from the mistie solution, applied in data()
        self.time_variant = None  # {"t": [...], "shift": [...]} extra time-variant shift

    @property
    def times(self):
        return 1000.0 * (self.t0 + self.dt * np.arange(self.ns))

    def raw_data(self):
        """(n_traces, ns) float32 as stored in the SEG-Y (no mistie correction), cached."""
        if self._raw is None:
            self._raw = self.f.traces()
            if not np.isfinite(self._raw).all():  # corrupt samples would poison dips / RGT / FFTs
                np.nan_to_num(self._raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return self._raw

    def data(self):
        """(n_traces, ns) float32, cached; with the line's mistie correction applied when it has one."""
        if self._data is None:
            raw = self.raw_data()
            c = self.correction
            if c and (np.any(np.abs(c[0]) > 1e-9) or abs(c[1]) > 1e-9 or abs(c[2] - 1.0) > 1e-9):
                from .mistie import apply_correction
                self._data = apply_correction(raw, self.dt * 1000.0, c[0], c[1], c[2]).astype(np.float32)
            else:
                self._data = raw
            tv = getattr(self, "time_variant", None)
            if tv:
                from .mistie import apply_time_variant
                self._data = apply_time_variant(self._data, self.times, tv).astype(np.float32)
        return self._data

    def set_correction(self, c):
        """(shift_ms (scalar or one per trace), phase_deg, gain) or None."""
        self.correction = tuple(c) if c else None
        self._data = None
        self._smooth_cache = None

    def release(self):
        self._data = None
        self._raw = None
        self._smooth_cache = None


def read_nav_csv(path):
    """CSV with columns line,cdp,x,y. Returns {line: array[n, 3]} sorted by cdp."""
    import csv
    nav = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            keys = {k.lower().strip(): v for k, v in row.items()}
            nav.setdefault(keys["line"].strip(), []).append((float(keys["cdp"]), float(keys["x"]), float(keys["y"])))
    return {k: np.array(sorted(v)) for k, v in nav.items()}


def match_nav(name, nav):
    """Nav entry for a line name, tolerant of R_2 / R-2 / r2 spelling."""
    if name in nav:
        return nav[name]
    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
    for k, v in nav.items():
        if norm(k) == norm(name):
            return v
    return None


def unused_store(store):
    """store, or store_2, store_3 ... when a volume store already exists there."""
    cand, k = store, 2
    while os.path.exists(os.path.join(cand, "meta.json")) or os.path.exists(os.path.join(cand, "data_ilxl.npy")):
        cand = f"{store}_{k}"
        k += 1
    return cand
