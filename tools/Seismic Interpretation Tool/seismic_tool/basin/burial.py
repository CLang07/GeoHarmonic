"""Burial history (PLAN 18b): backstripping / decompaction with Athy porosity, erosion events, paleo water depth.

Porosity phi(z) = phi0 exp(-c z), z = depth below the sediment surface (m). A unit with top zt and base zb holds
grain (solid) thickness  s = (zb - zt) - phi0 / c (exp(-c zt) - exp(-c zb)); moved to a new top depth its base
solves the same equation (Newton). Compaction is irreversible: a unit never decompacts above its deepest top
(uplift / erosion keep the porosity of maximum burial).

Lithology mixes: phi0, c, grain density, heat capacity, radiogenic heat = fraction-weighted means; matrix
conductivity = geometric mean. Defaults: Sclater & Christie (1980) compaction; conductivity / radiogenic heat
typical values (Hantschel & Kauerauf 2009). All editable via `props`.
"""
import numpy as np

from .chrono import LITHS, _norm_lith

LITHOLOGY = {  # phi0 (frac), c (1/m), rho grain (kg/m3), k matrix (W/m/K), A (uW/m3), cp grain (J/kg/K)
    "sand": {"phi0": 0.49, "c": 0.27e-3, "rho": 2650.0, "k": 3.2, "A": 0.7, "cp": 850.0},
    "silt": {"phi0": 0.56, "c": 0.39e-3, "rho": 2680.0, "k": 2.2, "A": 1.3, "cp": 850.0},
    "shale": {"phi0": 0.63, "c": 0.51e-3, "rho": 2720.0, "k": 1.9, "A": 2.0, "cp": 860.0},
    "carbonate": {"phi0": 0.70, "c": 0.71e-3, "rho": 2710.0, "k": 2.8, "A": 0.6, "cp": 840.0},
    "coal": {"phi0": 0.75, "c": 0.60e-3, "rho": 1350.0, "k": 0.3, "A": 0.4, "cp": 1300.0},
}
WATER = {"rho": 1030.0, "k": 0.6, "cp": 4180.0}


def mix(fracs, table=None):
    """Bulk matrix properties of a lithology mix {lith: fraction}."""
    table = table or LITHOLOGY
    f = _norm_lith(fracs)
    out = {}
    for key in ("phi0", "c", "rho", "A", "cp"):
        out[key] = float(sum(f[l] * table[l][key] for l in LITHS))
    out["k"] = float(np.exp(sum(f[l] * np.log(table[l]["k"]) for l in LITHS)))
    return out


# ====================================================================== Athy compaction
def solid_thickness(zt, zb, phi0, c):
    zt, zb = np.asarray(zt, float), np.asarray(zb, float)
    if np.ndim(c) == 0 and c <= 0:
        return (zb - zt) * (1.0 - phi0)
    c = np.asarray(c, float)
    cc = np.where(c > 0, c, 1.0)
    return np.where(c > 0, (zb - zt) - phi0 / cc * (np.exp(-cc * zt) - np.exp(-cc * zb)), (zb - zt) * (1 - phi0))


def base_for(zt, solid, phi0, c, iters=40):
    """Base depth of a unit of grain thickness `solid` with its top at zt (vectorised Newton)."""
    zt = np.asarray(zt, float)
    s = np.asarray(solid, float)
    phi0 = np.asarray(phi0, float)
    c = np.asarray(c, float)
    lin = (c <= 0) | (phi0 <= 0)
    cc = np.where(lin, 1.0, c)
    zb = zt + s / np.maximum(1e-9, 1 - phi0 * np.exp(-cc * zt))
    for _ in range(iters):
        f = (zb - zt) - phi0 / cc * (np.exp(-cc * zt) - np.exp(-cc * zb)) - s
        d = 1 - phi0 * np.exp(-cc * zb)
        step = f / np.maximum(d, 1e-9)
        zb = np.maximum(zt, zb - step)
        if np.all(np.abs(step) < 1e-7):
            break
    lin_b = zt + s / np.maximum(1e-9, 1 - np.where(c <= 0, phi0, 0.0))
    return np.where(lin, lin_b, zb)


def mean_porosity(zt, zb, phi0, c):
    h = np.asarray(zb, float) - np.asarray(zt, float)
    s = solid_thickness(zt, zb, phi0, c)
    return np.where(h > 0, 1 - s / np.where(h > 0, h, 1), phi0 * np.exp(-np.asarray(c) * np.asarray(zt, float)))


# ====================================================================== column -> units
def column_units(table, depths, seabed=0.0, props=None):
    """Depositional units from present-day surface depths.
    table: chrono.AgeTable; depths: {entry name: TVDSS m (positive down)}; seabed: sea-floor TVDSS (m).
    Units are youngest first: {name, top, base (m below sea floor), age_top, age_base (Ma), lith, props,
    eroded_above (m eroded at its top surface), ghost (eroded unit, no present thickness)}.
    The deposits above the shallowest dated surface (when it is older than 0 Ma) form an 'Overburden' unit."""
    props = props or LITHOLOGY
    ents = [e for e in table.entries if e["name"] in depths and np.isfinite(depths[e["name"]])]
    if not ents:
        raise ValueError("no dated surface has a depth here")
    units = []
    first = ents[0]
    zf = float(depths[first["name"]]) - seabed
    if first["age"] > 1e-9 and zf > 0:
        above = [e for e in table.entries if e["age"] < first["age"]]
        lith = above[-1]["lith"] if above else {"shale": 1.0}
        units.append({"name": f"Overburden (above {first['name']})", "top_surface": "Sea floor",
                      "base_surface": first["name"], "top": 0.0, "base": zf, "age_top": 0.0,
                      "age_base": first["age"], "lith": _norm_lith(lith)})
    for a, b in zip(ents[:-1], ents[1:]):
        zt = max(0.0, float(depths[a["name"]]) - seabed)
        zb = max(zt, float(depths[b["name"]]) - seabed)
        if a.get("eroded", 0) > 0 and a["gap"] > 0:  # eroded unit deposited in the first half of the hiatus
            units.append({"name": f"Eroded at {a['name']}", "top_surface": a["name"], "base_surface": a["name"],
                          "top": zt, "base": zt, "age_top": a["age"] + 0.5 * a["gap"],
                          "age_base": a["age"] + a["gap"], "erosion_age": a["age"], "lith": a["lith"],
                          "ghost": True, "eroded": float(a["eroded"])})
        units.append({"name": a["name"], "top_surface": a["name"], "base_surface": b["name"], "top": zt,
                      "base": zb, "age_top": a["age"] + a["gap"], "age_base": b["age"], "lith": a["lith"]})
    for u in units:
        u["props"] = mix(u["lith"], props)
        u.setdefault("ghost", False)
    return units


def _time_axis(units, step):
    oldest = max(u["age_base"] for u in units)
    ev = {0.0, oldest}
    for u in units:
        ev.update([u["age_top"], u["age_base"]] + ([u["erosion_age"]] if u.get("erosion_age") is not None else []))
    reg = np.arange(0.0, oldest, step)
    a = np.unique(np.round(np.concatenate([np.array(sorted(ev)), reg]), 9))
    return a[::-1]  # oldest first


class BurialHistory:
    """Backstripped burial history of one column. ages (nt,) Ma oldest -> 0; top / base (nt, nu) depth below
    the sediment surface (NaN = not yet deposited / eroded); solid (nt, nu) grain thickness; water (nt,)."""

    def __init__(self, units, step=1.0, water_depth=0.0, sea_level=None):
        self.units = units
        nu = len(units)
        p = [u["props"] for u in units]
        self.phi0 = np.array([q["phi0"] for q in p])
        self.c = np.array([q["c"] for q in p])
        # present grain thickness; below an erosion surface the porosity is that of maximum burial
        off = np.zeros(nu)
        for i, u in enumerate(units):
            if u["ghost"]:
                over = max(0.0, u["eroded"] - u["top"])  # net uplift: eroded column not re-buried
                off[i + 1:] = np.maximum(off[i + 1:], over)
        self.offset = off
        s_now = np.array([0.0 if u["ghost"] else
                          float(solid_thickness(u["top"] + off[i], u["base"] + off[i], self.phi0[i], self.c[i]))
                          for i, u in enumerate(units)])
        for i, u in enumerate(units):
            if u["ghost"]:
                s_now[i] = float(solid_thickness(0.0, u["eroded"], self.phi0[i], self.c[i]))
        self.solid_full = s_now
        self.ages = _time_axis(units, step)
        nt = len(self.ages)
        self.top = np.full((nt, nu), np.nan)
        self.base = np.full((nt, nu), np.nan)
        self.solid = np.zeros((nt, nu))
        self.top_eff = np.full((nt, nu), np.nan)  # compaction depth of each unit top (>= top: max burial)
        zmax = np.zeros(nu)
        for it, t in enumerate(self.ages):
            z = 0.0
            for i in range(nu):  # youngest unit is on top
                s = self._solid_at(i, t)
                if s <= 0:
                    if t <= units[i]["age_base"] + 1e-12 and not units[i]["ghost"]:
                        self.top[it, i] = self.base[it, i] = z  # deposition starting / zero-thickness unit
                    continue
                zmax[i] = max(zmax[i], z)
                ze = max(z, zmax[i])
                zb = float(base_for(ze, s, self.phi0[i], self.c[i]))
                h = zb - ze
                self.top[it, i], self.base[it, i] = z, z + h
                self.top_eff[it, i] = ze
                self.solid[it, i] = s
                z += h
        wd = water_depth
        if callable(wd):
            self.water = np.array([float(wd(t)) for t in self.ages])
        elif np.ndim(wd) == 0:
            self.water = np.full(nt, float(wd))
        else:
            arr = np.asarray(wd, float)  # [(age, depth)]
            o = np.argsort(arr[:, 0])
            self.water = np.interp(self.ages, arr[o, 0], arr[o, 1])
        if sea_level is None:
            self.sea = np.zeros(nt)
        else:
            arr = np.asarray(sea_level, float)
            o = np.argsort(arr[:, 0])
            self.sea = np.interp(self.ages, arr[o, 0], arr[o, 1])

    def _solid_at(self, i, t):
        u = self.units[i]
        s = self.solid_full[i]
        a0, a1 = u["age_base"], u["age_top"]  # deposition from a0 (older) to a1
        if t >= a0:
            return 0.0
        if t > a1:
            return s * (a0 - t) / max(1e-12, a0 - a1)
        if u["ghost"]:
            e = u["erosion_age"]
            if t <= e:
                return 0.0
            return s * (t - e) / max(1e-12, a1 - e)
        return s

    # ---- results
    def surfaces(self):
        """[(name, age Ma, depth below sea level (nt,))] - burial curve of each dated surface (top of each unit
        and the base of the oldest); NaN before the surface existed."""
        out = []
        seen = set()
        for i, u in enumerate(self.units):
            if u["ghost"]:
                continue
            nm = u["top_surface"]
            if nm not in seen:
                seen.add(nm)
                d = self.top[:, i] + self.water
                d = np.where(self.ages <= u["age_top"] + 1e-9, d, np.nan)
                out.append((nm, u["age_top"], d))
        last = [i for i, u in enumerate(self.units) if not u["ghost"]][-1]
        u = self.units[last]
        d = np.where(self.ages <= u["age_base"] + 1e-9, self.base[:, last] + self.water, np.nan)
        out.append((u["base_surface"], u["age_base"], d))
        return out

    def present(self):
        """(tops, bases) at 0 Ma (m below the sediment surface)."""
        return self.top[-1], self.base[-1]

    def thickness(self):
        return self.base - self.top

    def porosity(self):
        """Mean porosity (nt, nu) of each unit (NaN where absent)."""
        h = self.base - self.top
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(h > 1e-9, 1 - self.solid / h, np.nan)

    def to_csv(self, path):
        names = [n for n, _, _ in self.surfaces()]
        curves = np.array([d for _, _, d in self.surfaces()]).T
        with open(path, "w") as f:
            f.write("age_Ma," + ",".join(f'"{n} depth_m"' for n in names) + "\n")
            for a, row in zip(self.ages, curves):
                f.write(f"{a:.4f}," + ",".join("" if not np.isfinite(v) else f"{v:.2f}" for v in row) + "\n")


def well_depths(wc_project, well, table):
    """{entry name: TVDSS} of a well's tops for the dated surfaces (Well Correlation project, tops in MD)."""
    tops = wc_project.tops.get(well, {})
    names = [e["name"] for e in table.entries if e["name"] in tops]
    if not names:
        return {}
    w = wc_project.load_well(well)
    tv = w.md_to_tvdss([tops[n] for n in names])
    return {n: float(v) for n, v in zip(names, np.atleast_1d(tv))}


def velocity_model_depths(vm, table, x, y):
    """[{entry name: TVDSS}] per map point from the depth maps of a VelocityModel (item 17); a layer matches
    an age-table entry by its horizon name or its formation name."""
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    Z = vm._sample_layers("z", x, y)  # (n, K)
    match = {}
    for k, (hname, fm) in enumerate(vm.layers):
        for nm in (fm, hname):
            e = table.get(nm) if nm else None
            if e is not None:
                match[e["name"]] = k
                break
    return [{n: float(Z[i, k]) for n, k in match.items() if np.isfinite(Z[i, k])} for i in range(len(x))]
