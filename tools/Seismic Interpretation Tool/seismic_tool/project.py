"""Seismic project: 3D volumes (imported stores), 2D lines (SEG-Y + nav), linked wells. One JSON file.

Paths are stored relative to the project file so the whole folder can be moved.
"""
import csv
import json
import os

from . import __version__
from .horizons import HorizonSet
from .volume import Line2D, Volume3D, match_nav, read_nav_csv

EXT = ".seisproj.json"


class SeisProject:
    def __init__(self):
        self.path = None
        self.name = "Untitled"
        self.volumes = []  # [{"name", "store", "segy"}]
        self.lines = []  # [{"name", "segy", "cdp_byte", "nav"}]
        self.nav_files = []  # CSV paths (line,cdp,x,y)
        self.well_project = None  # path to a Well Correlation project (.wcproj.json)
        self.display = {"cmap": "seismic", "clip": 99.0, "reverse": False}
        self.horizons = HorizonSet()
        self._hz_saved_to = None
        self._vol_cache = {}
        self._line_cache = {}
        self._nav = None

    # ------------------------------------------------------------ io
    def base(self):
        return os.path.dirname(os.path.abspath(self.path)) if self.path else os.getcwd()

    def _rel(self, p):
        if not p:
            return p
        try:
            return os.path.relpath(p, self.base())
        except ValueError:
            return p

    def _abs(self, p):
        if not p:
            return p
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(self.base(), p))

    def seismic_dir(self):
        """<project>_seismic folder next to the project file (volume stores, interpretation)."""
        stem = os.path.basename(self.path)[:-len(EXT)] if self.path else self.name
        return os.path.join(self.base(), stem + "_seismic")

    def save(self, path=None):
        if path:
            self.path = path if path.endswith(EXT) else path + EXT
        hz_dir = os.path.join(self.seismic_dir(), "horizons")
        if self.horizons.items and (self.horizons.dirty or self._hz_saved_to != hz_dir):
            self.horizons.save(hz_dir)
            self._hz_saved_to = hz_dir
        d = {"app": "seismic_tool", "version": __version__, "name": self.name,
             "volumes": [dict(v, store=self._rel(v["store"]), segy=self._rel(v.get("segy"))) for v in self.volumes],
             "lines": [dict(l, segy=self._rel(l["segy"])) for l in self.lines],
             "nav_files": [self._rel(n) for n in self.nav_files],
             "well_project": self._rel(self.well_project), "display": self.display,
             "horizons": self.horizons.to_json(self._rel(hz_dir))}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(d, fh, indent=1)
        os.replace(tmp, self.path)

    @classmethod
    def open(cls, path):
        p = cls()
        p.path = path
        with open(path) as fh:
            d = json.load(fh)
        p.name = d.get("name", p.name)
        p.volumes = [dict(v, store=p._abs(v["store"]), segy=p._abs(v.get("segy"))) for v in d.get("volumes", [])]
        p.lines = [dict(l, segy=p._abs(l["segy"])) for l in d.get("lines", [])]
        p.nav_files = [p._abs(n) for n in d.get("nav_files", [])]
        p.well_project = p._abs(d.get("well_project"))
        p.display.update(d.get("display", {}))
        try:
            p.horizons = HorizonSet.load(d.get("horizons"), p.base())
            p._hz_saved_to = os.path.join(p.seismic_dir(), "horizons")
        except Exception as e:  # never block opening the project because of interpretation files
            print("Could not load horizons:", e)
        return p

    # ------------------------------------------------------------ data access
    def nav(self):
        if self._nav is None:
            self._nav = {}
            for n in self.nav_files:
                if os.path.exists(n):
                    self._nav.update(read_nav_csv(n))
        return self._nav

    def volume(self, name):
        if name not in self._vol_cache:
            v = next(v for v in self.volumes if v["name"] == name)
            self._vol_cache[name] = Volume3D(v["store"])
        return self._vol_cache[name]

    def line(self, name):
        if name not in self._line_cache:
            l = next(l for l in self.lines if l["name"] == name)
            self._line_cache[name] = Line2D(name, l["segy"], l.get("cdp_byte", 21), match_nav(name, self.nav()))
        return self._line_cache[name]

    def add_lines(self, paths, cdp_byte=21):
        added = []
        names = {l["name"] for l in self.lines}
        for pth in paths:
            name = os.path.splitext(os.path.basename(pth))[0].replace("_", "-")
            base, k = name, 2
            while name in names:
                name, k = f"{base} ({k})", k + 1
            self.lines.append({"name": name, "segy": os.path.abspath(pth), "cdp_byte": cdp_byte})
            names.add(name)
            added.append(name)
        return added

    def add_nav(self, path):
        if path not in self.nav_files:
            self.nav_files.append(os.path.abspath(path))
        self._nav = None
        self._line_cache.clear()

    def wells(self):
        """[(name, x, y)] from the linked Well Correlation project (headers only)."""
        if not self.well_project or not os.path.exists(self.well_project):
            return []
        try:
            with open(self.well_project) as fh:
                d = json.load(fh)
            return [(w["name"], w["x"], w["y"]) for w in d.get("wells", []) if w.get("x") is not None]
        except Exception:
            return []
