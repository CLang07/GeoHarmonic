"""Minimal SEG-Y rev 0/1 reader/writer for 2D post-stack lines (big-endian)."""
import numpy as np

_FMT = {1: 4, 2: 4, 3: 2, 5: 4, 8: 1}


def _ibm_to_float(u):
    u = u.astype(np.uint32)
    sign = np.where(u >> 31, -1.0, 1.0)
    expo = ((u >> 24) & 0x7F).astype(np.int32) - 64
    mant = (u & 0x00FFFFFF) / 16777216.0
    return (sign * mant * np.power(16.0, expo)).astype(np.float32)


def _hdr_int(hdr, byte, nbytes):
    """Read a big-endian signed int from 1-based byte position in every trace header."""
    col = hdr[:, byte - 1: byte - 1 + nbytes].copy()
    return col.view(">i4" if nbytes == 4 else ">i2").ravel().astype(np.int64)


class SegyLine:
    """One 2D line: traces [ntr, ns] (float32), dt (s), cdp numbers and (optional) coordinates."""

    def __init__(self, path, name=None, cdp_byte=21, x_byte=181, y_byte=185, scalar_byte=71):
        self.path, self.name = path, name or path
        raw = open(path, "rb").read()
        self.text_header = raw[:3200]
        self.bin_header = bytearray(raw[3200:3600])
        self.dt = int.from_bytes(raw[3216:3218], "big") / 1e6
        self.ns = int.from_bytes(raw[3220:3222], "big")
        self.fmt = int.from_bytes(raw[3224:3226], "big")
        if self.fmt not in _FMT:
            raise ValueError(f"{path}: unsupported SEG-Y sample format code {self.fmt}")
        bps = _FMT[self.fmt]
        trl = 240 + self.ns * bps
        ntr = (len(raw) - 3600) // trl
        if ntr <= 0:
            raise ValueError(f"{path}: no traces found (check that this is a SEG-Y file)")
        buf = np.frombuffer(raw, np.uint8, offset=3600, count=ntr * trl).reshape(ntr, trl)
        self.trace_headers = buf[:, :240].copy()
        body = buf[:, 240:].copy()
        if self.fmt == 1:
            self.data = _ibm_to_float(body.view(">u4"))
        elif self.fmt == 2:
            self.data = body.view(">i4").astype(np.float32)
        elif self.fmt == 3:
            self.data = body.view(">i2").astype(np.float32)
        elif self.fmt == 5:
            self.data = body.view(">f4").astype(np.float32)
        else:
            self.data = body.view("i1").astype(np.float32)
        h = self.trace_headers
        self.cdp = _hdr_int(h, cdp_byte, 4).astype(float)
        if not np.all(np.diff(self.cdp) > 0):           # fall back to trace sequence numbers
            self.cdp = np.arange(1, ntr + 1, dtype=float)
        sc = _hdr_int(h, scalar_byte, 2).astype(float)
        sc = np.where(sc < 0, 1.0 / np.abs(sc), np.where(sc > 0, sc, 1.0))
        self.x = _hdr_int(h, x_byte, 4) * sc
        self.y = _hdr_int(h, y_byte, 4) * sc

    @property
    def has_coords(self):
        return np.hypot(np.ptp(self.x), np.ptp(self.y)) > 1.0   # line must span more than one unit

    def write(self, path, data=None):
        """Write a copy with (optionally) new trace data, as IEEE float (format 5)."""
        data = self.data if data is None else data
        binh = bytearray(self.bin_header)
        binh[24:26] = (5).to_bytes(2, "big")
        with open(path, "wb") as f:
            f.write(self.text_header); f.write(bytes(binh))
            body = np.asarray(data, dtype=">f4")
            out = np.concatenate([self.trace_headers, body.view(np.uint8).reshape(len(body), -1)], axis=1)
            f.write(out.tobytes())
