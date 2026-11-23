"""DLIS (RP66 v1) reader in numpy, with LAS 2.0 export.

Structure: Storage Unit Label (80 bytes) -> Visible Records (4-byte header: length, 0xFF, 0x01) -> Logical
Record Segments (4-byte header: length, attribute bits, type; optional padding / checksum / trailing length)
-> Logical Records: EFLRs (explicitly formatted: sets of objects with attributes - FILE-HEADER, ORIGIN,
CHANNEL, FRAME, PARAMETER, TOOL, ...) and IFLRs (frame data: frame OBNAME, frame number, channel samples).
A FILE-HEADER starts a new logical file.

Frame data are decoded vectorised: all IFLRs of a frame with the same layout are gathered and read with one
numpy structured dtype (big-endian), so a 100 MB file takes seconds.

    f = DlisFile(path)
    for lf in f.logical_files:
        lf.origin, lf.parameters, lf.channels, lf.frames, lf.tools
        for fr in lf.frames.values():
            data = lf.frame_data(fr.name)     # {channel name: array (n,) or (n, *dimension)}
    dlis_to_las(path, out_dir)               # one LAS 2.0 per frame
"""
import os
import struct

import numpy as np

# ---------------------------------------------------------------------- representation codes
# fixed-size codes: code -> (numpy dtype string, count of values per element)
FIXED = {
    2: (">f4", 1),   # FSINGL
    3: (">f4", 2),   # FSING1 value + bound
    4: (">f4", 3),   # FSING2 value + 2 bounds
    7: (">f8", 1),   # FDOUBL
    8: (">f8", 2),   # FDOUB1
    9: (">f8", 3),   # FDOUB2
    10: (">f4", 2),  # CSINGL complex
    11: (">f8", 2),  # CDOUBL
    12: (">i1", 1),  # SSHORT
    13: (">i2", 1),  # SNORM
    14: (">i4", 1),  # SLONG
    15: (">u1", 1),  # USHORT
    16: (">u2", 1),  # UNORM
    17: (">u4", 1),  # ULONG
    26: (">u1", 1),  # STATUS
}
SIZES = {1: 2, 2: 4, 3: 8, 4: 12, 5: 4, 6: 4, 7: 8, 8: 16, 9: 24, 10: 8, 11: 16, 12: 1, 13: 2, 14: 4, 15: 1,
         16: 2, 17: 4, 21: 8, 26: 1}
NAMES = {1: "FSHORT", 2: "FSINGL", 3: "FSING1", 4: "FSING2", 5: "ISINGL", 6: "VSINGL", 7: "FDOUBL", 8: "FDOUB1",
         9: "FDOUB2", 10: "CSINGL", 11: "CDOUBL", 12: "SSHORT", 13: "SNORM", 14: "SLONG", 15: "USHORT",
         16: "UNORM", 17: "ULONG", 18: "UVARI", 19: "IDENT", 20: "ASCII", 21: "DTIME", 22: "ORIGIN",
         23: "OBNAME", 24: "OBJREF", 25: "ATTREF", 26: "STATUS", 27: "UNITS"}


class DlisError(Exception):
    pass


class ObName(tuple):
    """(origin, copy, identifier)"""

    @property
    def id(self):
        return self[2]

    def __str__(self):
        return self[2]


def ibm32(u):
    u = np.asarray(u, np.uint32)
    sign = np.where(u >> 31, -1.0, 1.0)
    exp = ((u >> 24) & 0x7F).astype(np.int64) - 64
    frac = (u & 0x00FFFFFF).astype(np.float64) / 16777216.0
    return sign * frac * 16.0 ** exp


def vax32(raw):
    """VAX F float from 4 bytes in file order."""
    b = np.frombuffer(raw, np.uint8).reshape(-1, 4)
    u = (b[:, 1].astype(np.uint32) << 24) | (b[:, 0].astype(np.uint32) << 16) | \
        (b[:, 3].astype(np.uint32) << 8) | b[:, 2].astype(np.uint32)
    sign = np.where(u >> 31, -1.0, 1.0)
    e = ((u >> 23) & 0xFF).astype(np.int64)
    f = (u & 0x7FFFFF).astype(np.float64)
    return np.where(e == 0, 0.0, sign * (0.5 + f / 16777216.0) * 2.0 ** (e - 128))


def fshort(raw):
    v = np.frombuffer(raw, ">u2").astype(np.int64)
    mant = v >> 4
    mant = np.where(mant >= 2048, mant - 4096, mant)
    return mant / 2048.0 * 2.0 ** (v & 0xF)


class Reader:
    """Sequential decoder of RP66 values from a bytes object."""

    def __init__(self, buf, pos=0):
        self.b = buf
        self.p = pos

    def left(self):
        return len(self.b) - self.p

    def take(self, n):
        if self.p + n > len(self.b):
            raise DlisError("read past end of record")
        r = self.b[self.p:self.p + n]
        self.p += n
        return r

    def u1(self):
        self.p += 1
        return self.b[self.p - 1]

    def uvari(self):
        c = self.b[self.p]
        if c < 0x80:
            self.p += 1
            return c
        if c < 0xC0:
            v = struct.unpack(">H", self.take(2))[0]
            return v & 0x3FFF
        v = struct.unpack(">I", self.take(4))[0]
        return v & 0x3FFFFFFF

    def ident(self):
        n = self.u1()
        return bytes(self.take(n)).decode("latin-1")

    def ascii(self):
        n = self.uvari()
        return bytes(self.take(n)).decode("latin-1")

    def obname(self):
        o = self.uvari()
        c = self.u1()
        return ObName((o, c, self.ident()))

    def value(self, code):
        if code in (2, 7, 12, 13, 14, 15, 16, 17, 26):
            dt, _ = FIXED[code]
            n = SIZES[code]
            return np.frombuffer(self.take(n), dt)[0].item()
        if code in (3, 4, 8, 9, 10, 11):
            dt, k = FIXED[code]
            return tuple(np.frombuffer(self.take(SIZES[code]), dt).tolist())
        if code == 1:
            return float(fshort(self.take(2))[0])
        if code == 5:
            return float(ibm32(np.frombuffer(self.take(4), ">u4"))[0])
        if code == 6:
            return float(vax32(self.take(4))[0])
        if code in (18, 22):
            return self.uvari()
        if code in (19, 27):
            return self.ident()
        if code == 20:
            return self.ascii()
        if code == 21:
            y, tzm, d, h, mn, s = (self.u1() for _ in range(6))
            ms = struct.unpack(">H", self.take(2))[0]
            return f"{1900 + y:04d}-{tzm & 0x0F:02d}-{d:02d} {h:02d}:{mn:02d}:{s:02d}.{ms:03d}"
        if code == 23:
            return self.obname()
        if code == 24:
            t = self.ident()
            return (t, self.obname())
        if code == 25:
            t = self.ident()
            o = self.obname()
            return (t, o, self.ident())
        raise DlisError(f"unknown representation code {code}")


# ---------------------------------------------------------------------- physical layer
def read_logical_records(buf):
    """Yields (is_eflr, type, encrypted, body bytes) for every logical record of a file buffer."""
    n = len(buf)
    if n < 80:
        raise DlisError("file too short for a storage unit label")
    sul = bytes(buf[:80]).decode("latin-1")
    if "V1" not in sul[4:9] and "RECORD" not in sul[9:15]:
        raise DlisError(f"not a DLIS v1 file (storage unit label {sul[:20]!r})")
    pos = 80
    parts, cur = [], None
    while pos + 4 <= n:
        vlen = (buf[pos] << 8) | buf[pos + 1]
        if vlen < 4:
            # some writers pad the end of the file with zeros
            break
        if buf[pos + 2] != 0xFF:
            raise DlisError(f"bad visible record header at byte {pos}")
        vend = min(n, pos + vlen)
        p = pos + 4
        while p + 4 <= vend:
            slen = (buf[p] << 8) | buf[p + 1]
            attr = buf[p + 2]
            typ = buf[p + 3]
            if slen < 4:
                raise DlisError(f"bad segment length at byte {p}")
            body_end = p + slen
            e = body_end
            if attr & 0x02:  # trailing length
                e -= 2
            if attr & 0x04:  # checksum
                e -= 2
            if attr & 0x01:  # padding: last byte = pad count
                e -= buf[e - 1]
            body = buf[p + 4:e]
            if attr & 0x10 and attr & 0x08 and not (attr & 0x40):
                pass  # encryption packet stays in the body (record skipped below)
            if not (attr & 0x40):  # no predecessor: new logical record
                cur = {"eflr": bool(attr & 0x80), "type": typ, "enc": bool(attr & 0x10), "parts": []}
            if cur is not None:
                cur["parts"].append(body)
            if not (attr & 0x20):  # no successor: record complete
                if cur is not None:
                    parts.append(cur)
                cur = None
            p = body_end
        pos += vlen
    for r in parts:
        yield r["eflr"], r["type"], r["enc"], b"".join(bytes(x) for x in r["parts"])


# ---------------------------------------------------------------------- EFLR parsing
class Obj:
    def __init__(self, set_type, name):
        self.type = set_type
        self.name = name
        self.attrs = {}  # label -> dict(value=list, units, code)

    def get(self, label, default=None):
        a = self.attrs.get(label)
        if a is None or a.get("value") is None:
            return default
        return a["value"]

    def first(self, label, default=None):
        v = self.get(label)
        if v is None:
            return default
        return v[0] if isinstance(v, list) and v else v

    def units(self, label):
        a = self.attrs.get(label)
        return a.get("units", "") if a else ""

    def __repr__(self):
        return f"<{self.type} {self.name.id}>"


def parse_eflr(body):
    """Returns (set_type, set_name, [Obj])."""
    r = Reader(body)
    desc = r.u1()
    role = desc >> 5
    if role not in (5, 6, 7):
        raise DlisError("EFLR does not start with a SET component")
    set_type = r.ident() if desc & 0x10 else ""
    set_name = r.ident() if desc & 0x08 else ""
    template = []
    objs = []
    # template
    while r.left():
        desc = r.b[r.p]
        role = desc >> 5
        if role == 3:  # OBJECT
            break
        r.p += 1
        a = {"label": "", "count": 1, "code": 19, "units": "", "value": None}
        if desc & 0x10:
            a["label"] = r.ident()
        if desc & 0x08:
            a["count"] = r.uvari()
        if desc & 0x04:
            a["code"] = r.u1()
        if desc & 0x02:
            a["units"] = r.ident()
        if desc & 0x01:
            a["value"] = [r.value(a["code"]) for _ in range(a["count"])]
        template.append(a)
    # objects
    while r.left():
        desc = r.u1()
        role = desc >> 5
        if role != 3:
            raise DlisError("expected OBJECT component")
        name = r.obname() if desc & 0x10 else ObName((0, 0, ""))
        o = Obj(set_type, name)
        for t in template:
            if not r.left():
                o.attrs[t["label"]] = dict(t)
                continue
            d = r.b[r.p]
            rl = d >> 5
            if rl == 3:  # next object begins: remaining attributes take template defaults
                o.attrs[t["label"]] = dict(t)
                continue
            r.p += 1
            if rl == 0:  # ABSATR
                o.attrs[t["label"]] = {"label": t["label"], "count": 0, "code": t["code"], "units": t["units"],
                                       "value": None}
                continue
            a = dict(t)
            if d & 0x10:
                a["label"] = r.ident() or t["label"]
            if d & 0x08:
                a["count"] = r.uvari()
            if d & 0x04:
                a["code"] = r.u1()
            if d & 0x02:
                a["units"] = r.ident()
            if d & 0x01:
                a["value"] = [r.value(a["code"]) for _ in range(a["count"])]
            o.attrs[t["label"]] = a
        objs.append(o)
    return set_type, set_name, objs


# ---------------------------------------------------------------------- logical file
class Channel:
    def __init__(self, o):
        self.obj = o
        self.name = o.name.id
        self.long_name = o.first("LONG-NAME", "")
        if isinstance(self.long_name, tuple):
            self.long_name = str(self.long_name[-1])
        self.units = o.first("UNITS", "") or ""
        self.code = int(o.first("REPRESENTATION-CODE", 2) or 2)
        dim = o.get("DIMENSION") or [1]
        self.dimension = [int(d) for d in dim]
        self.count = int(np.prod(self.dimension))

    def __repr__(self):
        return f"<Channel {self.name} [{self.units}] {NAMES.get(self.code, self.code)} {self.dimension}>"


class Frame:
    def __init__(self, o, channels):
        self.obj = o
        self.name = o.name.id
        self.key = tuple(o.name)
        refs = o.get("CHANNELS") or []
        self.channels = []
        for ref in refs:
            nm = ref.id if isinstance(ref, ObName) else (ref[1].id if isinstance(ref, tuple) and len(ref) == 2 else str(ref))
            ch = channels.get((tuple(ref) if isinstance(ref, ObName) else None)) or channels.get(nm)
            if ch is None:
                raise DlisError(f"frame {self.name}: channel {nm} not defined")
            self.channels.append(ch)
        self.index_type = o.first("INDEX-TYPE", "") or ""
        self.direction = o.first("DIRECTION", "") or ""
        self.spacing = o.first("SPACING")
        self.spacing_units = o.units("SPACING")
        self.records = []  # IFLR bodies (bytes)

    @property
    def index(self):
        return self.channels[0] if self.channels else None

    def __repr__(self):
        return f"<Frame {self.name} {len(self.channels)} channels, {len(self.records)} frames, {self.index_type}>"


class LogicalFile:
    def __init__(self):
        self.header = None
        self.origin = None
        self.sets = []  # (set_type, [Obj])
        self.channels = {}  # obname tuple / name -> Channel
        self.frames = {}  # name -> Frame
        self.parameters = []
        self.tools = []
        self.encrypted = 0
        self._frame_by_key = {}

    def objects(self, set_type):
        return [o for t, objs in self.sets if t == set_type for o in objs]

    def _finish_eflr(self, set_type, objs):
        self.sets.append((set_type, objs))
        if set_type == "FILE-HEADER" and objs:
            self.header = objs[0]
        elif set_type == "ORIGIN" and objs and self.origin is None:
            self.origin = objs[0]
        elif set_type == "CHANNEL":
            for o in objs:
                ch = Channel(o)
                self.channels[tuple(o.name)] = ch
                self.channels.setdefault(o.name.id, ch)
        elif set_type == "FRAME":
            for o in objs:
                fr = Frame(o, self.channels)
                self.frames[fr.name] = fr
                self._frame_by_key[fr.key] = fr
        elif set_type == "PARAMETER":
            self.parameters.extend(objs)
        elif set_type == "TOOL":
            self.tools.extend(objs)

    def channel_list(self):
        return [c for k, c in self.channels.items() if isinstance(k, tuple)]

    def origin_info(self):
        o = self.origin
        if o is None:
            return {}
        keys = ("WELL-NAME", "WELL-ID", "FIELD-NAME", "COMPANY", "PRODUCER-NAME", "CREATION-TIME", "FILE-ID",
                "RUN-NUMBER", "DESCENT-NUMBER")
        out = {}
        for k in keys:
            v = o.first(k)
            if v is not None:
                out[k] = v.strip() if isinstance(v, str) else v
        return out

    def frame_data(self, frame_name):
        """{channel: array} for one frame; array shape (n,) or (n, *dimension). Index first."""
        fr = self.frames[frame_name]
        recs = fr.records
        if not recs:
            return {c.name: np.zeros((0,) + tuple(c.dimension if c.count > 1 else ())) for c in fr.channels}
        fixed = all(c.code in SIZES for c in fr.channels)
        if not fixed:
            return self._frame_data_slow(fr)
        dtype_fields = []
        for k, c in enumerate(fr.channels):
            raw = c.code in (1, 5, 6, 21)
            dt = (">u2" if c.code == 1 else ">u4" if c.code in (5, 6) else "V8") if raw else FIXED[c.code][0]
            per = FIXED[c.code][1] if c.code in FIXED else 1
            shape = (c.count * per,) if c.count * per > 1 else ()
            dtype_fields.append((f"c{k}", dt, shape) if shape else (f"c{k}", dt))
        rec_dt = np.dtype(dtype_fields)
        # strip OBNAME + frame number from each record
        datas, fnums = [], []
        for body in recs:
            r = Reader(body)
            r.obname()
            fnums.append(r.uvari())
            datas.append(body[r.p:])
        sz = rec_dt.itemsize
        good = [d for d in datas if len(d) >= sz]
        if len(good) != len(datas):
            raise DlisError(f"frame {fr.name}: {len(datas) - len(good)} short frame records")
        buf = b"".join(d[:sz] for d in datas)
        arr = np.frombuffer(buf, rec_dt)
        out = {}
        for k, c in enumerate(fr.channels):
            v = arr[f"c{k}"]
            if c.code == 1:
                v = fshort(v.astype(">u2").tobytes()).reshape(v.shape)
            elif c.code == 5:
                v = ibm32(v)
            elif c.code == 6:
                v = vax32(v.astype(">u4").tobytes()).reshape(v.shape)
            elif c.code in (3, 4, 8, 9):  # value + bounds: keep the value
                per = FIXED[c.code][1]
                v = v.reshape(len(arr), c.count, per)[..., 0]
            v = np.asarray(v)
            if c.count > 1:
                v = v.reshape((len(arr),) + tuple(c.dimension))
            else:
                v = v.reshape(len(arr))
            out[c.name] = v.astype(np.float64) if v.dtype.kind in "iuf" else v
        out["__frame_number__"] = np.asarray(fnums)
        return out

    def _frame_data_slow(self, fr):
        cols = {c.name: [] for c in fr.channels}
        for body in fr.records:
            r = Reader(body)
            r.obname()
            r.uvari()
            for c in fr.channels:
                vals = [r.value(c.code) for _ in range(c.count)]
                cols[c.name].append(vals[0] if c.count == 1 else vals)
        return {k: np.asarray(v) for k, v in cols.items()}


class DlisFile:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as fh:
            buf = fh.read()
        self.logical_files = []
        lf = None
        pending = []  # IFLRs seen before their FRAME object (rare)
        for is_eflr, typ, enc, body in read_logical_records(memoryview(buf)):
            if is_eflr:
                if enc:
                    if lf is not None:
                        lf.encrypted += 1
                    continue
                try:
                    set_type, _, objs = parse_eflr(body)
                except DlisError:
                    continue
                if set_type == "FILE-HEADER" or lf is None:
                    lf = LogicalFile()
                    self.logical_files.append(lf)
                lf._finish_eflr(set_type, objs)
            else:
                if lf is None or enc or typ != 0:
                    continue  # NOFORMAT / EOD / encrypted
                r = Reader(body)
                try:
                    key = tuple(r.obname())
                except Exception:
                    continue
                fr = lf._frame_by_key.get(key)
                if fr is None:
                    fr = next((f for f in lf.frames.values() if f.name == key[2]), None)
                if fr is not None:
                    fr.records.append(body)
                else:
                    pending.append((lf, key, body))
        for lf_, key, body in pending:
            fr = lf_._frame_by_key.get(key)
            if fr is not None:
                fr.records.append(body)

    def summary(self):
        lines = [f"{os.path.basename(self.path)}: {len(self.logical_files)} logical file(s)"]
        for k, lf in enumerate(self.logical_files):
            info = lf.origin_info()
            lines.append(f"  [{k}] {info.get('WELL-NAME', '?')} / {info.get('COMPANY', '')} / "
                         f"{info.get('FILE-ID', '')}  frames: {len(lf.frames)}  channels: {len(lf.channel_list())}"
                         f"  parameters: {len(lf.parameters)}  tools: {len(lf.tools)}")
            for fr in lf.frames.values():
                lines.append(f"      frame {fr.name}: {len(fr.records)} rows, index {fr.index.name if fr.index else '-'}"
                             f" [{fr.index.units if fr.index else ''}] {fr.index_type} {fr.direction}; "
                             + ", ".join(c.name + ("" if c.count == 1 else str(c.dimension)) for c in fr.channels[1:]))
        return "\n".join(lines)


# ---------------------------------------------------------------------- units + LAS export
DEPTH_UNITS_TO_M = {"m": 1.0, "M": 1.0, "meters": 1.0, "metres": 1.0, "ft": 0.3048, "F": 0.3048, "FT": 0.3048,
                    "feet": 0.3048, "0.1 in": 0.00254, "0.1in": 0.00254, ".1 in": 0.00254, "in": 0.0254,
                    "cm": 0.01, "mm": 0.001, "0.5 mm": 0.0005}


def depth_to_m(values, units):
    f = DEPTH_UNITS_TO_M.get((units or "").strip())
    if f is None:
        f = DEPTH_UNITS_TO_M.get((units or "").strip().lower())
    return (np.asarray(values, float) * f, "m") if f else (np.asarray(values, float), units or "")


def _las_value(v):
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) == 1 else " ".join(str(x) for x in v)
    return str(v).replace(":", " ")


def frame_to_las(lf, frame_name, path, arrays=False, null=-999.25):
    """Write one frame as LAS 2.0 (index converted to metres when it is a depth)."""
    fr = lf.frames[frame_name]
    data = lf.frame_data(frame_name)
    idx_ch = fr.index
    idx, idx_units = data[idx_ch.name], idx_ch.units
    is_depth = "DEPTH" in fr.index_type.upper() or idx_ch.name.upper() in ("DEPT", "TDEP", "DEPTH", "MD") or \
        (idx_units or "").strip() in DEPTH_UNITS_TO_M
    if is_depth:
        idx, idx_units = depth_to_m(idx, idx_units)
    cols = [("DEPT" if is_depth else idx_ch.name, idx_units, idx_ch.long_name or fr.index_type, idx)]
    skipped = []
    for c in fr.channels[1:]:
        v = data[c.name]
        if v.ndim > 1:
            if not arrays:
                skipped.append(c.name)
                continue
            for k in range(int(np.prod(v.shape[1:]))):
                cols.append((f"{c.name}[{k}]", c.units, c.long_name, v.reshape(len(v), -1)[:, k]))
            continue
        cols.append((c.name, c.units, c.long_name, v))
    info = lf.origin_info()
    n = len(idx)
    step = float(np.median(np.diff(idx))) if n > 1 else 0.0
    if n > 1 and np.max(np.abs(np.diff(idx) - step)) > 1e-3 * max(abs(step), 1e-9):
        step = 0.0
    with open(path, "w", newline="\r\n") as fh:
        fh.write("~Version Information\n VERS.                 2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0\n"
                 " WRAP.                  NO : One line per depth step\n")
        fh.write("~Well Information Block\n")
        u = idx_units or ""
        fh.write(f" STRT.{u:<6} {idx[0] if n else 0:>14.4f} : START\n STOP.{u:<6} {idx[-1] if n else 0:>14.4f} : STOP\n")
        fh.write(f" STEP.{u:<6} {step:>14.4f} : STEP\n NULL.       {null:>14.2f} : NULL VALUE\n")
        fh.write(f" WELL.        {_las_value(info.get('WELL-NAME', ''))} : WELL\n")
        fh.write(f" COMP.        {_las_value(info.get('COMPANY', ''))} : COMPANY\n")
        fh.write(f" FLD .        {_las_value(info.get('FIELD-NAME', ''))} : FIELD\n")
        fh.write(f" SRVC.        {_las_value(info.get('PRODUCER-NAME', ''))} : SERVICE COMPANY\n")
        fh.write(f" DATE.        {_las_value(info.get('CREATION-TIME', ''))} : DATE\n")
        fh.write(f" UWI .        {_las_value(info.get('WELL-ID', ''))} : UNIQUE WELL ID\n")
        fh.write(f"~Curve Information Block\n")
        for name, unit, desc, _ in cols:
            fh.write(f" {name:<12}.{(unit or ''):<10} : {_las_value(desc or '')}\n")
        if lf.parameters:
            fh.write("~Parameter Information Block\n")
            for p in lf.parameters:
                v = p.get("VALUES")
                if v is None or len(v) == 0:
                    continue
                fh.write(f" {p.name.id[:20]:<20}.{p.units('VALUES'):<8} {_las_value(v)[:60]} : "
                         f"{_las_value(p.first('LONG-NAME', '') or '')[:60]}\n")
        fh.write(f"~Other\n Converted from {os.path.basename(getattr(lf, 'source', ''))} frame {frame_name} "
                 f"by GeoHarmonic dlis.py. Skipped array channels: {', '.join(skipped) or 'none'}\n")
        fh.write("~A " + " ".join(c[0] for c in cols) + "\n")
        M = np.column_stack([np.asarray(c[3], float) for c in cols]) if cols else np.zeros((0, 0))
        M = np.where(np.isfinite(M), M, null)
        M = np.where(np.abs(M) > 1e29, null, M)  # absent-value markers
        np.savetxt(fh, M, fmt="%.6g", delimiter=" ")
    return {"path": path, "rows": n, "curves": [c[0] for c in cols], "skipped": skipped}


def dlis_to_las(path, out_dir, arrays=False):
    """Convert every frame of every logical file to LAS 2.0. Returns a list of result dicts."""
    os.makedirs(out_dir, exist_ok=True)
    f = DlisFile(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    out = []
    for k, lf in enumerate(f.logical_files):
        lf.source = path
        for fname, fr in lf.frames.items():
            if not fr.records:
                continue
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in fname)
            name = f"{stem}_{k}_{safe}.las" if len(f.logical_files) > 1 else f"{stem}_{safe}.las"
            out.append(frame_to_las(lf, fname, os.path.join(out_dir, name), arrays))
    return out


# ---------------------------------------------------------------------- log loading (Well Correlation Tool)
ABSENT = (-999.25, -9999.0, -999.0)


def clean(v):
    v = np.asarray(v, float).copy()
    bad = ~np.isfinite(v) | (np.abs(v) > 1e29)
    for a in ABSENT:
        bad |= np.isclose(v, a)
    v[bad] = np.nan
    return v


class DlisLogs:
    """A DLIS file presented like las.LasFile (curves = LasCurve per scalar channel of every depth frame,
    depth in metres, increasing).  Used by project.load_well for wells/<Well>/dlis/*.dlis."""

    def __init__(self, path):
        from .las import LasCurve
        self.path = str(path)
        self.curves = []
        self.well = {}
        self.depth_unit = "M"
        f = DlisFile(path)
        col = 0
        for k, lf in enumerate(f.logical_files):
            info = lf.origin_info()
            if info.get("WELL-NAME") and "WELL" not in self.well:
                self.well["WELL"] = (str(info["WELL-NAME"]), "", "")
            for fr in lf.frames.values():
                if not fr.records or fr.index is None:
                    continue
                idx_units = (fr.index.units or "").strip()
                is_depth = "DEPTH" in fr.index_type.upper() or idx_units in DEPTH_UNITS_TO_M
                if not is_depth:
                    continue
                data = lf.frame_data(fr.name)
                depth, u = depth_to_m(data[fr.index.name], idx_units)
                if u != "m":
                    continue
                order = np.argsort(depth, kind="stable")
                depth = depth[order]
                for c in fr.channels[1:]:
                    v = data[c.name]
                    if v.ndim != 1 or v.dtype.kind not in "iuf":
                        continue
                    col += 1
                    desc = f"{c.long_name} [DLIS frame {fr.name}]".strip()
                    self.curves.append(LasCurve(c.name.upper(), c.units, desc, depth, clean(v[order]), self.path, col))

    @property
    def well_name(self):
        return self.well.get("WELL", ("",))[0]


# ---------------------------------------------------------------------- checkshots (CSAT / VSI seismic frames)
def checkshot_from_dlis(path, source_depth_m=0.0):
    """Time-depth table from a Schlumberger checkshot / VSP DLIS (frame with DEP1 or TDEP, CTT1/BTI1).

    Per shot: measured depth (m), TVD below SRD (if U-SEIS_TVDSRD1 is present), transit time (ms, one way,
    surface-sensor referenced = CTT1, else BTI1 - BTRS).  Shots with zero / absent times are dropped, repeated
    shots at one station are combined by the median.  Vertical one-way time uses the cosine (straight-ray)
    correction with the source offset (ODFW) and source_depth_m below SRD.
    Returns dict of arrays (md, tvd, transit_ms, owt_ms, twt_ms, n_shots, std_ms) sorted by depth."""
    f = DlisFile(path)
    for lf in f.logical_files:
        for fr in lf.frames.values():
            names = {c.name for c in fr.channels}
            if not ({"CTT1", "BTI1"} & names) or not ({"DEP1", "TDEP", "SDEP"} & names):
                continue
            d = lf.frame_data(fr.name)
            units = {c.name: c.units for c in fr.channels}

            def in_m(name):  # every depth channel in metres (a file in feet would otherwise give feet)
                return clean(depth_to_m(d[name], units.get(name))[0])
            md = in_m("DEP1") if "DEP1" in d else in_m("SDEP") if "SDEP" in d else in_m("TDEP")
            tvd = in_m("U-SEIS_TVDSRD1") if "U-SEIS_TVDSRD1" in d else md.copy()
            if "CTT1" in d:
                tt = clean(d["CTT1"])
            else:
                tt = clean(d["BTI1"]) - (clean(d["U-SEIS_BTRS"]) if "U-SEIS_BTRS" in d else 0.0)
            off = clean(d["ODFW"]) if "ODFW" in d else np.zeros_like(md)
            off = np.where(np.isfinite(off), off, 0.0)
            ok = np.isfinite(md) & np.isfinite(tt) & (tt > 1.0) & np.isfinite(tvd)
            if not ok.any():
                continue
            md, tvd, tt, off = md[ok], tvd[ok], tt[ok], off[ok]
            key = np.round(md, 1)
            stations = np.unique(key)
            rows = []
            for s in stations:
                m = key == s
                t = np.median(tt[m])
                good = np.abs(tt[m] - t) < 5.0  # drop mis-picked shots
                t = float(np.median(tt[m][good])) if good.any() else float(t)
                z = float(np.median(tvd[m]))
                o = float(np.median(off[m]))
                dz = max(z - source_depth_m, 1e-6)
                owt = t * dz / np.hypot(dz, o)
                rows.append((float(np.median(md[m])), z, t, owt, 2 * owt, int(m.sum()), float(np.std(tt[m][good]))
                             if good.sum() > 1 else 0.0))
            a = np.array(sorted(rows))
            info = lf.origin_info()
            return {"md": a[:, 0], "tvd": a[:, 1], "transit_ms": a[:, 2], "owt_ms": a[:, 3], "twt_ms": a[:, 4],
                    "n_shots": a[:, 5].astype(int), "std_ms": a[:, 6], "well": info.get("WELL-NAME", ""),
                    "source": os.path.basename(path), "frame": fr.name}
    return None


def write_checkshot_csv(table, path):
    with open(path, "w", newline="") as fh:
        fh.write(f"# checkshot from {table['source']} frame {table['frame']} well {table['well']}\n")
        fh.write("# owt/twt: vertical times (cosine correction for source offset); tvd below SRD\n")
        fh.write("MD_m,TVD_SRD_m,transit_ms,OWT_ms,TWT_ms,n_shots,std_ms\n")
        for r in zip(table["md"], table["tvd"], table["transit_ms"], table["owt_ms"], table["twt_ms"],
                     table["n_shots"], table["std_ms"]):
            fh.write("{:.2f},{:.2f},{:.3f},{:.3f},{:.3f},{:d},{:.3f}\n".format(*r))
    return path
