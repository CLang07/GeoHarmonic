"""Project model: wells, composited logs, tops, saved sections.  One JSON file per project.

Folder layout the importer understands (what we built for the NZ data):

    <data folder>/
        wells_header.csv            Well, UWI, Latitude, Longitude, Elevation_m, Elev_Ref, TD_m, ...
        wells/<Well>/las/*.las      log data (any number of files / runs)
        wells/<Well>/dlis/*.dlis    DLIS log data (read directly, depth frames -> curves like LAS)
        wells/<Well>/survey/*.csv   optional MD,INC,AZI (also picks up *Survey*.txt in mudlog/)

Composited logs are cached as .npz next to the project file (``<project>_cache/``) so a
project with hundreds of wells opens fast after the first load.
"""
import csv
import glob
import hashlib
import json
import os
import time

import numpy as np

from . import __version__
from .curves import DEFAULT_ALIASES, composite, STEP
from .geo import tm_forward, read_survey, min_curvature
from .las import LasFile

PROJECT_EXT = ".wcproj.json"


class Well:
    def __init__(self, name):
        self.name = name
        self.uwi = ""
        self.lat = self.lon = None
        self.x = self.y = None  # NZTM / projected metres
        self.elev = 0.0  # reference elevation above MSL (KB/DF/RT)
        self.elev_ref = "KB"
        self.td = None
        self.info = {}
        self.las_paths = []
        self.survey_path = None
        # loaded on demand
        self.depth = None  # MD grid (m)
        self.logs = {}
        self.sources = {}
        self.candidates = {}
        self.tvd = None  # TVD below reference at each MD grid point
        self.loaded = False

    # --------------------------------------------------------------- depth
    def md_to_tvdss(self, md):
        md = np.asarray(md, float)
        if self.tvd is not None and self.depth is not None and len(self.depth):
            tvd = np.interp(md, self.depth, self.tvd)
        else:
            tvd = md
        return tvd - (self.elev or 0.0)

    def tvdss_to_md(self, z):
        z = np.asarray(z, float)
        if self.tvd is not None and self.depth is not None and len(self.depth):
            return np.interp(z + (self.elev or 0.0), self.tvd, self.depth)
        return z + (self.elev or 0.0)

    def to_dict(self):
        return {
            "name": self.name, "uwi": self.uwi, "lat": self.lat, "lon": self.lon, "x": self.x, "y": self.y,
            "elev": self.elev, "elev_ref": self.elev_ref, "td": self.td, "info": self.info,
            "las": self.las_paths, "survey": self.survey_path,
        }

    @classmethod
    def from_dict(cls, d):
        w = cls(d["name"])
        for k in ("uwi", "lat", "lon", "x", "y", "elev", "elev_ref", "td", "info"):
            if k in d:
                setattr(w, k, d[k])
        w.las_paths = d.get("las", [])
        w.survey_path = d.get("survey")
        return w


class Project:
    def __init__(self):
        self.path = None
        self.name = "Untitled"
        self.wells = {}  # name -> Well (insertion ordered)
        self.extra_aliases = {}  # user additions, e.g. {"GR": ["MYGR"]}; defaults stay in code
        self.curve_overrides = {}  # well -> family -> {"mnemonic":..,"file":..}
        self.formations = []  # [{"name":..,"color":"#rrggbb"}] in stratigraphic order (step 2)
        self.tops = {}  # well -> {formation: md}
        self.tops_meta = {}  # well -> {formation: {"source": "published|auto|user", "score": .., "note": ..}}
        self.reference_tops = {}  # name -> {"tops": {well: {formation: md}}, "notes": str}
        self.sections = {"Section 1": []}
        self.current_section = "Section 1"
        self.display = {"gr_cutoff": 75.0, "tracks": ["LITH", "GR", "RES", "DN", "DT"], "spacing": "equal",
                        "between": "lithology",
                        "hang": "TVDSS", "track_width": 1.0, "well_gap": 0.6}
        self.created = time.strftime("%Y-%m-%d %H:%M")

    @property
    def aliases(self):
        a = {k: list(v) for k, v in DEFAULT_ALIASES.items()}
        for fam, names in self.extra_aliases.items():
            a.setdefault(fam, [])
            a[fam] = [n.upper() for n in names] + [n for n in a[fam] if n not in [m.upper() for m in names]]
        return a

    # ------------------------------------------------------------ import
    @classmethod
    def from_folder(cls, folder, progress=None):
        p = cls()
        p.name = os.path.basename(os.path.normpath(folder))
        hdr = {}
        hfile = os.path.join(folder, "wells_header.csv")
        if os.path.exists(hfile):
            with open(hfile, newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    hdr[r.get("Well", "").strip().lower()] = r
        wells_dir = os.path.join(folder, "wells")
        if not os.path.isdir(wells_dir):
            wells_dir = folder
        names = sorted(d for d in os.listdir(wells_dir) if os.path.isdir(os.path.join(wells_dir, d)))
        for i, name in enumerate(names):
            wd = os.path.join(wells_dir, name)
            las = sorted(glob.glob(os.path.join(wd, "las", "*.las")) + glob.glob(os.path.join(wd, "las", "*.LAS")))
            if not las:
                las = sorted(glob.glob(os.path.join(wd, "*.las")) + glob.glob(os.path.join(wd, "*.LAS")))
            las = sorted(set(las), key=str.lower)
            las += sorted(set(glob.glob(os.path.join(wd, "dlis", "*.dlis")) + glob.glob(os.path.join(wd, "dlis", "*.DLIS"))),
                          key=str.lower)
            w = Well(name)
            w.las_paths = las
            sv = glob.glob(os.path.join(wd, "survey", "*")) + [f for f in glob.glob(os.path.join(wd, "mudlog", "*")) if "survey" in os.path.basename(f).lower()]
            w.survey_path = sv[0] if sv else None
            h = hdr.get(name.lower())
            if h:
                w.uwi = h.get("UWI", "")
                w.lat, w.lon = _f(h.get("Latitude")), _f(h.get("Longitude"))
                w.elev = _f(h.get("Elevation_m"), 0.0)
                w.elev_ref = h.get("Elev_Ref", "KB")
                w.td = _f(h.get("TD_m"))
                w.info = {k: v for k, v in h.items() if k not in ("Well",)}
            if w.lat is not None and w.lon is not None:
                w.x, w.y = tm_forward(w.lat, w.lon)
            p.wells[name] = w
            if progress:
                progress(i + 1, len(names), name)
        return p

    def add_from_folder(self, folder, progress=None):
        """Add the wells of a data folder (same layout as from_folder) that are not in the project yet.
        Returns the names added (existing wells are left untouched)."""
        other = Project.from_folder(folder, progress)
        added = []
        for name, w in other.wells.items():
            if name in self.wells:
                continue
            self.wells[name] = w
            added.append(name)
        return added

    # ------------------------------------------------------------ save / open
    def to_dict(self):
        return {
            "app": "well_correlation", "version": __version__, "name": self.name, "created": self.created,
            "wells": [w.to_dict() for w in self.wells.values()], "extra_aliases": self.extra_aliases,
            "curve_overrides": self.curve_overrides, "formations": self.formations, "tops": self.tops, "tops_meta": self.tops_meta,
            "reference_tops": self.reference_tops,
            "sections": self.sections, "current_section": self.current_section, "display": self.display,
        }

    def save(self, path=None):
        path = path or self.path
        if not path.endswith(PROJECT_EXT):
            path += PROJECT_EXT
        base = os.path.dirname(os.path.abspath(path))
        d = self.to_dict()
        for wd in d["wells"]:  # store paths relative to the project file so the folder can move
            wd["las"] = [_rel(p, base) for p in wd["las"]]
            wd["survey"] = _rel(wd["survey"], base) if wd["survey"] else None
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, path)
        self.path = path

    @classmethod
    def open(cls, path):
        with open(path) as f:
            d = json.load(f)
        p = cls()
        p.path = path
        base = os.path.dirname(os.path.abspath(path))
        p.name = d.get("name", p.name)
        p.created = d.get("created", p.created)
        for wd in d.get("wells", []):
            wd["las"] = [_abs(x, base) for x in wd.get("las", [])]
            wd["survey"] = _abs(wd["survey"], base) if wd.get("survey") else None
            w = Well.from_dict(wd)
            p.wells[w.name] = w
        p.extra_aliases = d.get("extra_aliases", {})
        p.curve_overrides = d.get("curve_overrides", {})
        p.formations = d.get("formations", [])
        p.tops = d.get("tops", {})
        p.tops_meta = d.get("tops_meta", {})
        p.reference_tops = d.get("reference_tops", {})
        p.sections = d.get("sections", p.sections) or {"Section 1": []}
        p.current_section = d.get("current_section", next(iter(p.sections)))
        p.display.update(d.get("display", {}))
        return p

    # ------------------------------------------------------------ log loading
    def cache_dir(self):
        if self.path:
            return self.path[: -len(PROJECT_EXT)] + "_cache" if self.path.endswith(PROJECT_EXT) else self.path + "_cache"
        return None

    def _cache_key(self, w):
        h = hashlib.sha1()
        for p in w.las_paths + ([w.survey_path] if w.survey_path else []):
            try:
                st = os.stat(p)
                h.update(f"{p}|{st.st_size}|{st.st_mtime}".encode())
            except OSError:
                h.update(p.encode())
        h.update(json.dumps(self.aliases, sort_keys=True).encode())
        h.update(json.dumps(self.curve_overrides.get(w.name, {}), sort_keys=True).encode())
        h.update(b"v4")  # v4: survey column parsing, TVD below the last station, LAS fixes (run 2)
        return h.hexdigest()[:16]

    def load_well(self, name, force=False):
        w = self.wells[name]
        if w.loaded and not force:
            return w
        key = self._cache_key(w)
        cdir = self.cache_dir()
        cfile = os.path.join(cdir, f"{_safe(name)}_{key}.npz") if cdir else None
        if cfile and os.path.exists(cfile) and not force:
            z = np.load(cfile, allow_pickle=False)
            w.depth = z["depth"]
            w.tvd = z["tvd"] if "tvd" in z.files and z["tvd"].size else None
            w.logs = {k[4:]: z[k] for k in z.files if k.startswith("log_")}
            meta = json.loads(str(z["meta"]))
            w.sources, w.candidates = meta["sources"], meta["candidates"]
            w.loaded = True
            return w
        files = []
        for pth in w.las_paths:
            try:
                if pth.lower().endswith(".dlis"):
                    from .dlis import DlisLogs
                    files.append(DlisLogs(pth))
                else:
                    files.append(LasFile(pth))
            except Exception as e:  # keep going, report in sources
                w.sources.setdefault("_errors", []).append(f"{os.path.basename(pth)}: {e}")
        grid, logs, src, cands = composite(files, self.aliases, self.curve_overrides.get(name))
        w.depth, w.logs, w.sources, w.candidates = grid, logs, src, cands
        w.tvd = None
        if w.survey_path and len(grid):
            s = read_survey(w.survey_path)
            if s is not None:
                md, inc, azi = s
                tvd, dn, de = min_curvature(md, inc, azi)
                w.tvd = np.interp(grid, md, tvd)
                below = grid > md[-1]  # logs below the last station: straight on at the last inclination
                w.tvd[below] = tvd[-1] + (grid[below] - md[-1]) * np.cos(np.radians(inc[-1]))
        w.loaded = True
        if cfile:
            os.makedirs(cdir, exist_ok=True)
            for old in glob.glob(os.path.join(cdir, f"{_safe(name)}_*.npz")):
                try:
                    os.remove(old)
                except OSError:
                    pass
            meta = json.dumps({"sources": src, "candidates": cands})
            np.savez_compressed(cfile, depth=grid, tvd=w.tvd if w.tvd is not None else np.array([]),
                                meta=np.array(meta), **{"log_" + k: v for k, v in logs.items()})
        return w

    # ------------------------------------------------------------ sections
    def section_wells(self, name=None):
        return [n for n in self.sections.get(name or self.current_section, []) if n in self.wells]


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _rel(p, base):
    try:
        return os.path.relpath(p, base)
    except ValueError:  # different drive on Windows
        return p


def _abs(p, base):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def _safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)
