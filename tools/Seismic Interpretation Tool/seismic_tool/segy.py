"""SEG-Y reading with numpy memory-mapping (rev 0/1/2, big- or little-endian, IBM/IEEE/int).

Nothing is loaded into memory up front: headers and traces are read from the file on demand,
so a multi-GB volume opens instantly and memory use stays small.
"""
import codecs
import numpy as np

# sample format code -> (bytes, numpy dtype for big-endian)
_FMT = {1: (4, ">u4"), 2: (4, ">i4"), 3: (2, ">i2"), 5: (4, ">f4"), 6: (8, ">f8"), 8: (1, "i1"),
        10: (4, ">u4"), 11: (2, ">u2"), 16: (1, "u1")}

# common header byte positions (1-based) for inline / crossline in 3D surveys
IL_XL_CANDIDATES = [(189, 193), (9, 21), (5, 21), (17, 21), (13, 17), (221, 225), (9, 13)]


def ibm2ieee(u32):
    """IBM System/360 float (as uint32) -> float32, vectorised."""
    u = u32.astype(np.uint32)
    sign = np.where(u >> 31, -1.0, 1.0)
    expo = ((u >> 24) & 0x7F).astype(np.int32) - 64
    mant = (u & 0x00FFFFFF).astype(np.float64) / 16777216.0
    return (sign * mant * np.power(16.0, expo)).astype(np.float32)


class SegyFile:
    def __init__(self, path):
        self.path = str(path)
        self.mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        if self.mm.size < 3600 + 240:
            raise ValueError(f"{path}: too small to be SEG-Y")
        self.text_header = self._text(self.mm[:3200].tobytes())
        b = self.mm[3200:3600].tobytes()
        self.endian = ">"
        fmt = int.from_bytes(b[24:26], "big")
        if fmt not in _FMT and int.from_bytes(b[24:26], "little") in _FMT:
            self.endian = "<"
            fmt = int.from_bytes(b[24:26], "little")
        if fmt not in _FMT:
            raise ValueError(f"{path}: unsupported sample format code {fmt}")
        bo = "big" if self.endian == ">" else "little"
        self.fmt = fmt
        self.dt = int.from_bytes(b[16:18], bo) / 1e6  # seconds
        self.ns = int.from_bytes(b[20:22], bo)
        ext = int.from_bytes(b[304:306], bo, signed=True)  # rev 1 extended text headers
        self.data_offset = 3600 + max(0, ext) * 3200
        bps, dt = _FMT[fmt]
        self.bps = bps
        self.sample_dtype = dt if self.endian == ">" else dt.replace(">", "<")
        # trust the first trace header's sample count if the binary header disagrees
        ns_tr = int.from_bytes(self.mm[self.data_offset + 114: self.data_offset + 116].tobytes(), bo)
        if self.ns == 0 or (ns_tr and ns_tr != self.ns and (self.mm.size - self.data_offset) % (240 + ns_tr * bps) == 0):
            self.ns = ns_tr
        if self.dt == 0:
            self.dt = int.from_bytes(self.mm[self.data_offset + 116: self.data_offset + 118].tobytes(), bo) / 1e6 or 0.004
        self.trace_len = 240 + self.ns * bps
        self.ntraces = (self.mm.size - self.data_offset) // self.trace_len
        if self.ntraces <= 0:
            raise ValueError(f"{path}: no traces")
        self._tr = np.ndarray((self.ntraces, self.trace_len), np.uint8, self.mm, self.data_offset)
        self.t0 = 0.0
        delay = self.header(109, 2, 0, 1)
        if delay.size and delay[0]:
            self.t0 = float(delay[0]) / 1000.0

    @staticmethod
    def _text(raw):
        t = None
        try:
            t = codecs.decode(raw, "cp500")
            if not (sum(c.isprintable() for c in t) > 0.8 * len(t) and "C" in t[:5]):
                t = None
        except Exception:
            t = None
        if t is None:
            t = raw.decode("ascii", "replace")
        if "\n" not in t:
            t = "\n".join(t[i:i + 80].rstrip() for i in range(0, len(t), 80))
        return t.rstrip()

    def header(self, byte, nbytes=4, start=0, stop=None):
        """Header word (1-based byte) for traces [start:stop] as int64."""
        stop = self.ntraces if stop is None else stop
        col = np.ascontiguousarray(self._tr[start:stop, byte - 1: byte - 1 + nbytes])
        dt = {4: "i4", 2: "i2"}[nbytes]
        return col.view(self.endian + dt).ravel().astype(np.int64)

    def traces(self, start=0, stop=None, idx=None):
        """Trace samples as float32 (n, ns). Either a range or an index array."""
        if idx is not None:
            body = self._tr[np.asarray(idx), 240:]
        else:
            body = self._tr[start:self.ntraces if stop is None else stop, 240:]
        body = np.ascontiguousarray(body)
        raw = body.view(self.sample_dtype)
        if self.fmt == 1:
            return ibm2ieee(raw)
        return raw.astype(np.float32)

    def coordinates(self, x_byte=181, y_byte=185, scalar_byte=71, start=0, stop=None):
        sc = self.header(scalar_byte, 2, start, stop).astype(float)
        with np.errstate(divide="ignore"):
            sc = np.where(sc < 0, 1.0 / np.where(sc == 0, 1, np.abs(sc)), np.where(sc > 0, sc, 1.0))
        return self.header(x_byte, 4, start, stop) * sc, self.header(y_byte, 4, start, stop) * sc

    def guess_il_xl(self, sample=20000):
        """Find header bytes that look like a regular inline / crossline grid."""
        n = min(self.ntraces, sample)
        best = None
        for ib, xb in IL_XL_CANDIDATES:
            il, xl = self.header(ib, 4, 0, n), self.header(xb, 4, 0, n)
            if il.min() == il.max() and xl.min() == xl.max():
                continue
            ui, ux = np.unique(il), np.unique(xl)
            if len(ui) < 2 or len(ux) < 2 or len(ui) * len(ux) < 0.5 * n or len(ui) * len(ux) > 4 * n:
                continue
            if (il <= 0).any() or (xl <= 0).any():
                continue
            pairs = len(np.unique(il * 10_000_000 + xl))
            score = pairs / n  # 1.0 = every trace a unique (il, xl)
            if best is None or score > best[0]:
                best = (score, ib, xb)
        return (best[1], best[2]) if best and best[0] > 0.95 else None
