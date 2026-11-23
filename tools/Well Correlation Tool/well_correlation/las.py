"""Minimal, tolerant LAS 1.2 / 2.0 reader (numpy only).

Handles the things the NZ open-file data throws at us:
  * wrapped and unwrapped data
  * duplicate mnemonics (main pass + repeat pass in one file)
  * '*******' overflow fields and other junk -> NaN
  * depth in feet (converted to metres)
"""
import re
import numpy as np

_FT = {"F", "FT", "FEET", "FOOT", ".F", "'"}


class LasCurve:
    __slots__ = ("mnemonic", "unit", "description", "depth", "values", "file", "index")

    def __init__(self, mnemonic, unit, description, depth, values, file, index):
        self.mnemonic = mnemonic
        self.unit = unit
        self.description = description
        self.depth = depth
        self.values = values
        self.file = file
        self.index = index  # column number in the file (to tell repeat passes apart)

    @property
    def n_valid(self):
        return int(np.isfinite(self.values).sum())

    def valid_range(self):
        ok = np.isfinite(self.values)
        if not ok.any():
            return (np.nan, np.nan)
        d = self.depth[ok]
        return float(d.min()), float(d.max())

    def __repr__(self):
        t, b = self.valid_range()
        return f"<LasCurve {self.mnemonic} [{self.unit}] {t:.1f}-{b:.1f} m>"


class LasFile:
    def __init__(self, path):
        self.path = str(path)
        self.version = {}
        self.well = {}
        self.params = {}
        self.curve_defs = []  # (mnemonic, unit, description)
        self.depth = np.array([])
        self.curves = []  # LasCurve (excluding depth)
        self.depth_unit = "M"
        self._read()

    # ------------------------------------------------------------------ parse
    @staticmethod
    def _header_line(line):
        line = line.strip()
        if not line or line.startswith("#"):
            return None
        # MNEM .UNIT  VALUE : DESCRIPTION   (unit glued to the dot, value may contain spaces/colons)
        m = re.match(r"^([^.]*?)\s*\.(\S*)\s*(.*)$", line)
        if not m:
            return None
        mnem, unit, rest = m.groups()
        if ":" in rest:
            i = rest.rfind(":")
            value, desc = rest[:i], rest[i + 1:]
        else:
            value, desc = rest, ""
        return mnem.strip(), unit.strip(), value.strip(), desc.strip()

    def _read(self):
        with open(self.path, "r", errors="replace") as f:
            text = f.read()
        sections = re.split(r"\n(?=\s*~)", "\n" + text)
        data_text = None
        for sec in sections:
            s = sec.lstrip()
            if not s.startswith("~"):
                continue
            key = s[1:2].upper()
            body = s.split("\n", 1)[1] if "\n" in s else ""
            if key == "A":
                data_text = body
                continue
            target = {"V": self.version, "W": self.well, "P": self.params}.get(key)
            for line in body.split("\n"):
                h = self._header_line(line)
                if not h:
                    continue
                mnem, unit, value, desc = h
                if key == "C":
                    self.curve_defs.append((mnem, unit, desc))
                elif target is not None:
                    target[mnem.upper()] = (value, unit, desc)
        if not self.curve_defs or data_text is None:
            return
        null = self._float(self.well.get("NULL", ("-999.25",))[0], -999.25)
        wrap = self.version.get("WRAP", ("NO",))[0].upper().startswith("Y")
        dlm = self.version.get("DLM", ("SPACE",))[0].upper()
        tokens = re.split(r"[\s,]+", data_text.strip()) if ("COMMA" in dlm or "," in data_text[:2000]) \
            else data_text.split()
        n = len(self.curve_defs)
        try:
            vals = np.array(tokens, dtype=float)  # fast path
        except ValueError:
            vals = self._fast(tokens)  # junk like '*********' present
        m = len(vals) // n
        if m == 0:
            return
        arr = vals[: m * n].reshape(m, n)
        arr[np.isclose(arr, null)] = np.nan
        arr[arr <= -999.0] = np.nan  # common alternative nulls (-999, -9999)
        depth = arr[:, 0]
        du = self.curve_defs[0][1].upper() or self.well.get("STRT", ("", ""))[1].upper()  # blank unit: STRT's
        if du in _FT:
            depth = depth * 0.3048
            self.depth_unit = "FT->M"
        self.depth = depth
        for i, (mn, un, de) in enumerate(self.curve_defs[1:], 1):
            self.curves.append(LasCurve(mn.upper(), un, de, depth, arr[:, i], self.path, i))

    @staticmethod
    def _fast(tokens):
        out = np.empty(len(tokens))
        for i, t in enumerate(tokens):
            try:
                out[i] = float(t)
            except ValueError:
                out[i] = np.nan
        return out

    @staticmethod
    def _float(t, default=np.nan):
        try:
            return float(t)
        except (TypeError, ValueError):
            return default

    # ---------------------------------------------------------------- helpers
    def header_value(self, key, default=""):
        v = self.well.get(key.upper())
        if not v:
            return default
        value, unit, desc = v
        # old LAS 1.2 files put the value in the description
        return value or desc or default

    @property
    def well_name(self):
        return self.header_value("WELL")
