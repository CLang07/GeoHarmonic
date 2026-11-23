"""Chronostratigraphy (PLAN 18a): an age table for horizons / formations, ages for RGT stack levels,
the Wheeler diagram on an age axis (Ma) with labelled hiatus gaps, and zone lithology from the wells.

Age table entry (one per dated surface, youngest first):
    {"name": "Tikorangi Limestone",   # label; matched to well tops (formation) and seismic horizons
     "horizon": "Top Tikorangi",      # optional seismic horizon name (default = name)
     "age": 22.0,                     # Ma of the surface = onset of the deposits ABOVE it
     "gap": 0.0,                      # My of hiatus at the surface (unconformity); deposits BELOW end at age + gap
     "eroded": 0.0,                   # m eroded at this unconformity (used by the burial history, 18b)
     "lith": {"sand": .., "silt": .., "shale": .., "carbonate": .., "coal": ..}}   # zone BELOW the surface
Ages increase downwards. RGT increases downwards too (tau ~ TWT), so age(tau) is increasing.
"""
import copy

import numpy as np

LITHS = ("sand", "silt", "shale", "carbonate", "coal")
_CODE_LITH = {0: "sand", 1: "silt", 2: "shale", 3: "coal", 4: "carbonate"}  # well_correlation.lithology codes

# Maui field, offshore Taranaki. Ages are approximate (NZ timescale, King & Thrasher 1996 / PR3099 stratigraphy)
# and meant as a starting point - edit them in the Basin model window. The Seabed row dates the top of the column.
TARANAKI_TEMPLATE = [
    {"name": "Seabed", "age": 0.0, "lith": {"shale": 0.7, "silt": 0.2, "sand": 0.1},
     "note": "Giant Foresets / Rotokare mudstones above Tikorangi (Pliocene-Miocene)"},
    {"name": "Tikorangi Limestone", "age": 22.0, "lith": {"carbonate": 0.35, "shale": 0.65},
     "note": "Tikorangi limestone + Otaraoa marl + Turi shale (Oligocene - late Eocene)"},
    {"name": "C-sands (Mangahewa)", "age": 36.0, "lith": {"sand": 0.55, "shale": 0.35, "silt": 0.05, "coal": 0.05},
     "note": "Mangahewa Fm (mid - late Eocene), coal measures"},
    {"name": "D-sands (Kaimiro)", "age": 46.0, "lith": {"sand": 0.5, "shale": 0.4, "silt": 0.05, "coal": 0.05},
     "note": "Kaimiro Fm (early - mid Eocene)"},
    {"name": "E-shale", "age": 53.0, "lith": {"shale": 0.9, "sand": 0.1}, "note": "early Eocene shale"},
    {"name": "Tane Member", "age": 55.5, "lith": {"sand": 0.4, "shale": 0.6}, "note": "latest Paleocene"},
    {"name": "F0 (Farewell F-sands)", "age": 56.0, "lith": {"sand": 0.7, "shale": 0.25, "coal": 0.05},
     "note": "Farewell Fm (Paleocene)"},
    {"name": "F1 (Farewell F-sands)", "age": 58.0, "lith": {"sand": 0.7, "shale": 0.25, "coal": 0.05},
     "note": "Farewell Fm (Paleocene)"},
    {"name": "Base Kapuni F-sands", "age": 66.0, "lith": {"shale": 0.6, "sand": 0.3, "coal": 0.1},
     "note": "Pakawau Group (Late Cretaceous rift fill, Rakopi / North Cape source rocks)"},
    {"name": "Basement", "age": 84.0, "lith": {"shale": 0.0, "sand": 1.0},
     "note": "onset of Late Cretaceous rifting (base of the sediment column)"},
]


def _norm_lith(d):
    out = {k: max(0.0, float((d or {}).get(k, 0.0) or 0.0)) for k in LITHS}
    s = sum(out.values())
    if s <= 0:
        return {"sand": 0.0, "silt": 0.0, "shale": 1.0, "carbonate": 0.0, "coal": 0.0}
    return {k: v / s for k, v in out.items()}


def _entry(e):
    e = dict(e)
    e["name"] = str(e.get("name", "")).strip()
    e["age"] = float(e.get("age", 0.0))
    e["gap"] = max(0.0, float(e.get("gap", 0.0) or 0.0))
    e["eroded"] = max(0.0, float(e.get("eroded", 0.0) or 0.0))
    e["lith"] = _norm_lith(e.get("lith"))
    if e.get("horizon") in ("", None):
        e.pop("horizon", None)
    return e


class AgeTable:
    """Dated surfaces, youngest first. Saved in project.display["basin"]["ages"]."""

    def __init__(self, entries=None):
        self.entries = [_entry(e) for e in (entries or [])]
        self.sort()

    @classmethod
    def taranaki(cls):
        return cls(copy.deepcopy(TARANAKI_TEMPLATE))

    def sort(self):
        self.entries.sort(key=lambda e: e["age"])

    def names(self):
        return [e["name"] for e in self.entries]

    def get(self, name):
        key = str(name).strip().lower()
        for e in self.entries:
            if e["name"].lower() == key or str(e.get("horizon", "")).lower() == key:
                return e
        return None

    def set(self, name, **kw):
        e = self.get(name)
        if e is None:
            e = {"name": name, "age": 0.0}
            self.entries.append(e)
        e.update(kw)
        i = self.entries.index(e)
        self.entries[i] = _entry(e)
        self.sort()
        return self.entries[self.entries.index(self.get(name))]

    def remove(self, name):
        e = self.get(name)
        if e is not None:
            self.entries.remove(e)

    def age_above(self, name):
        e = self.get(name)
        return None if e is None else e["age"]

    def age_below(self, name):
        e = self.get(name)
        return None if e is None else e["age"] + e["gap"]

    def problems(self):
        """Readable list of problems (empty = OK)."""
        out = []
        seen = set()
        last = None
        for e in self.entries:
            if not e["name"]:
                out.append("an entry has no name")
            if e["name"].lower() in seen:
                out.append(f"duplicate name '{e['name']}'")
            seen.add(e["name"].lower())
            if e["age"] < 0:
                out.append(f"{e['name']}: negative age")
            if last is not None and e["age"] < last["age"] + last["gap"] - 1e-9:
                out.append(f"{e['name']} ({e['age']:g} Ma) is younger than the end of the hiatus under "
                           f"{last['name']} ({last['age'] + last['gap']:g} Ma)")
            last = e
        return out

    def zones(self):
        """[(top entry, base entry)] consecutive pairs = the depositional units, youngest first."""
        return list(zip(self.entries[:-1], self.entries[1:]))

    def match_horizons(self, names):
        """{entry name: horizon name} for the given seismic horizon names (case-insensitive)."""
        low = {str(n).lower(): n for n in names}
        out = {}
        for e in self.entries:
            for cand in (e.get("horizon"), e["name"]):
                if cand and str(cand).lower() in low:
                    out[e["name"]] = low[str(cand).lower()]
                    break
        return out

    def to_dict(self):
        return [dict(e) for e in self.entries]

    @classmethod
    def from_dict(cls, d):
        return cls(d or [])

    # ---- project storage
    @classmethod
    def load(cls, project, default_template=True):
        b = (project.display.get("basin") or {}) if project is not None else {}
        if b.get("ages"):
            return cls.from_dict(b["ages"])
        return cls.taranaki() if default_template else cls()

    def save(self, project):
        b = dict(project.display.get("basin") or {})
        b["ages"] = self.to_dict()
        project.display["basin"] = b


# ====================================================================== RGT -> age
def horizon_rgt(tau, times, horizon_t):
    """RGT of a horizon on a section: tau (ntr, ns), times (ns,) ms, horizon_t (ntr,) ms (NaN = unpicked).
    Returns (median RGT, per-trace RGT)."""
    tau = np.asarray(tau, float)
    times = np.asarray(times, float)
    ht = np.asarray(horizon_t, float)
    k = (ht - times[0]) / (times[1] - times[0])
    ok = np.isfinite(k) & (k >= 0) & (k <= len(times) - 1)
    kc = np.clip(np.where(ok, k, 0), 0, len(times) - 1.000001)
    k0 = np.floor(kc).astype(int)
    w = kc - k0
    r = np.arange(len(ht))
    per = np.where(ok, tau[r, k0] * (1 - w) + tau[r, k0 + 1] * w, np.nan)
    med = float(np.nanmedian(per)) if np.isfinite(per).any() else np.nan
    return med, per


def anchors_from(table, horizon_rgts):
    """Anchors [(rgt, age_above, age_below, name)] sorted by rgt, from {entry name: rgt}. Raises when the
    horizon order in RGT disagrees with the age order."""
    out = []
    for e in table.entries:
        r = horizon_rgts.get(e["name"])
        if r is None or not np.isfinite(r):
            continue
        out.append((float(r), e["age"], e["age"] + e["gap"], e["name"]))
    out.sort(key=lambda a: a[0])
    for a, b in zip(out[:-1], out[1:]):
        if b[1] < a[2] - 1e-9 or b[0] <= a[0]:
            raise ValueError(f"horizon order in RGT ({a[3]} above {b[3]}) disagrees with the age table")
    return out


class AgeModel:
    """Piecewise-linear age(tau) through dated horizons; at an unconformity the age jumps from age_above
    (just above the surface) to age_below (just below). Outside the dated range: the rate of the nearest
    interval (never younger than 0 Ma)."""

    def __init__(self, anchors):
        anchors = sorted(anchors, key=lambda a: a[0])
        if len(anchors) < 2:
            raise ValueError("need at least two dated horizons on the section")
        self.anchors = anchors
        xs, ys = [], []
        for r, above, below, _ in anchors:
            xs += [r, r]
            ys += [above, below]
        self.x = np.array(xs, float)
        self.y = np.array(ys, float)
        r = np.array([a[0] for a in anchors])
        self.rate_top = (anchors[1][1] - anchors[0][2]) / max(1e-9, r[1] - r[0])
        self.rate_bot = (anchors[-1][1] - anchors[-2][2]) / max(1e-9, r[-1] - r[-2])

    def gaps(self):
        """[(age_above, age_below, name)] of the unconformities (gap > 0)."""
        return [(a[1], a[2], a[3]) for a in self.anchors if a[2] > a[1] + 1e-9]

    def __call__(self, tau):
        tau = np.asarray(tau, float)
        x, y = self.x, self.y
        # np.interp with a repeated abscissa: just above r -> towards age_above, just below r -> age_below
        out = np.interp(tau, x, y)
        lo = tau < x[0]
        hi = tau > x[-1]
        out = np.where(lo, y[0] - (x[0] - tau) * self.rate_top, out)
        out = np.where(hi, y[-1] + (tau - x[-1]) * self.rate_bot, out)
        out = np.maximum(out, 0.0)
        return np.where(np.isfinite(tau), out, np.nan)


def level_ages(levels, model):
    """Ages (Ma) of the RGT stack levels; strictly non-decreasing."""
    a = model(np.asarray(levels, float))
    return np.maximum.accumulate(np.where(np.isfinite(a), a, -np.inf))


# ====================================================================== Wheeler on an age axis
def age_axis(level_age, n=None, step=None):
    a0, a1 = float(np.nanmin(level_age)), float(np.nanmax(level_age))
    if step is None:
        n = n or max(50, min(2000, 2 * len(level_age)))
        step = max(1e-3, (a1 - a0) / max(1, n - 1))
    return np.arange(a0, a1 + 0.5 * step, step)


def wheeler_age(amp, hiatus, lev_age, axis, gaps=()):
    """Wheeler (levels, ntr) -> age axis (len(axis), ntr). amp / hiatus from rgt.stack.wheeler; lev_age (N,)
    ages of the levels; gaps [(age_above, age_below, ...)] = unconformities (always blank).
    An age between levels k and k+1 is blank where that interval is a hiatus on the trace."""
    amp = np.asarray(amp, float)
    hiatus = np.asarray(hiatus, bool)
    lev_age = np.asarray(lev_age, float)
    axis = np.asarray(axis, float)
    N = len(lev_age)
    k = np.clip(np.searchsorted(lev_age, axis, side="right") - 1, 0, max(0, N - 2))
    span = lev_age[np.minimum(k + 1, N - 1)] - lev_age[k]
    w = np.where(span > 0, np.clip((axis - lev_age[k]) / np.where(span > 0, span, 1), 0, 1), 0.0)
    a = amp[k] * (1 - w[:, None]) + amp[np.minimum(k + 1, N - 1)] * w[:, None]
    blank = hiatus[k] | ~np.isfinite(a)
    outside = (axis < lev_age[0] - 1e-9) | (axis > lev_age[-1] + 1e-9)
    blank |= outside[:, None]
    for g in gaps:
        inside = (axis > g[0] + 1e-9) & (axis < g[1] - 1e-9)
        blank[inside] = True
    return np.where(blank, np.nan, a).astype(np.float32), blank


def gap_labels(blank, axis, x=None, min_my=0.0):
    """Group blank cells of an age-axis Wheeler (len(axis), ntr) into hiatus bodies (4-connected) and label
    each with its age span: [{"x": centre x, "age": mid age, "top": youngest, "base": oldest, "span": My,
    "traces": (j0, j1), "text": "hiatus 22.0-30.0 Ma (8.0 My)"}], largest first."""
    blank = np.asarray(blank, bool)
    na, nt = blank.shape
    axis = np.asarray(axis, float)
    x = np.arange(nt) if x is None else np.asarray(x, float)
    da = (axis[1] - axis[0]) if na > 1 else 0.0
    lab = np.zeros(blank.shape, np.int32)
    out = []
    cur = 0
    for j in range(nt):  # vertical runs per trace, linked to overlapping runs of the previous trace
        col = blank[:, j]
        if not col.any():
            continue
        d = np.diff(np.concatenate([[0], col.astype(np.int8), [0]]))
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        for s, e in zip(starts, ends):
            prev = set(lab[s:e, j - 1][lab[s:e, j - 1] > 0].tolist()) if j > 0 else set()
            if prev:
                tgt = min(prev)
                for p in prev - {tgt}:  # merge bodies joined by this run
                    lab[lab == p] = tgt
            else:
                cur += 1
                tgt = cur
            lab[s:e, j] = tgt
    for b in np.unique(lab[lab > 0]):
        ii, jj = np.nonzero(lab == b)
        top = float(axis[ii.min()]) - 0.5 * da
        base = float(axis[ii.max()]) + 0.5 * da
        span = base - top
        if span < min_my:
            continue
        jc = int(np.median(jj))
        out.append({"x": float(x[jc]), "age": 0.5 * (top + base), "top": top, "base": base, "span": span,
                    "traces": (int(jj.min()), int(jj.max())),
                    "text": f"hiatus {max(0.0, top):.1f}-{base:.1f} Ma ({span:.1f} My)"})
    out.sort(key=lambda d: -d["span"] * (d["traces"][1] - d["traces"][0] + 1))
    return out


def section_age_model(table, tau, times, horizon_times):
    """AgeModel for one section from {horizon name: times (ntr,) ms}; returns (model, {entry: rgt})."""
    match = table.match_horizons(list(horizon_times))
    rgts = {}
    for entry, hname in match.items():
        med, _ = horizon_rgt(tau, times, horizon_times[hname])
        if np.isfinite(med):
            rgts[entry] = med
    return AgeModel(anchors_from(table, rgts)), rgts


# ====================================================================== lithology from wells
def zone_lithology(project, table, wells=None):
    """Lithology fractions per zone (below each dated surface) from Well Correlation logs (Vsh lithology codes
    sand / silt / shale / coal / carbonate). project = well_correlation Project (formations, tops, load_well).
    Returns {entry name: {"lith": fractions, "wells": n, "metres": logged thickness}}."""
    from .. import welltie
    welltie.wc()
    from well_correlation.lithology import well_lithology
    names = [e["name"] for e in table.entries]
    acc = {n: (np.zeros(5), 0, set()) for n in names}
    for wn in wells or list(project.wells):
        tops = project.tops.get(wn, {})
        present = sorted([(tops[n], n) for n in names if n in tops])
        if len(present) < 1:
            continue
        try:
            w = project.load_well(wn)
        except Exception:
            continue
        if w.depth is None or not len(w.depth):
            continue
        _, code = well_lithology(w)
        bounds = present + [(float(w.depth[-1]) + 1e-6, None)]
        for (top, n), (base, _) in zip(bounds[:-1], bounds[1:]):
            m = (w.depth >= top) & (w.depth < base) & (code < 5)
            if not m.any():
                continue
            cnt = np.bincount(code[m].astype(int), minlength=5)[:5].astype(float)
            c, k, ws = acc[n]
            ws.add(wn)
            acc[n] = (c + cnt, k + int(m.sum()), ws)
    out = {}
    for n, (c, k, ws) in acc.items():
        if k == 0:
            continue
        frac = {_CODE_LITH[i]: float(c[i] / k) for i in range(5)}
        out[n] = {"lith": _norm_lith(frac), "wells": len(ws), "samples": k}
    return out


def apply_lithology(table, zl):
    """Copy zone_lithology results into the table (only zones that had log data)."""
    for n, d in zl.items():
        e = table.get(n)
        if e is not None:
            e["lith"] = _norm_lith(d["lith"])
            e["lith_source"] = f"logs ({d['wells']} wells)"
    return table
