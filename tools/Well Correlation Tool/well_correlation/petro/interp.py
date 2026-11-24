"""Deterministic petrophysical interpretation per zone with parameter sets (PLAN 20b).

Equations (all vectorised, fractions not %, resistivities ohm.m, DT us/ft, density g/cc, temperature degC):
  Vsh   IGR = (GR - GRcl) / (GRsh - GRcl); linear, Larionov Tertiary 0.083 (2^(3.7 I) - 1), Larionov older rocks
        0.33 (2^(2 I) - 1), Clavier 1.7 - sqrt(3.38 - (I + 0.7)^2), Steiber I / (3 - 2 I); SP (PSP-SSP style linear);
        neutron-density (NPHI - PHID) / (NPHI_sh - PHID_sh); 'min' = minimum of the available methods.
  PHI   density (rho_ma - RHOB) / (rho_ma - rho_fl); sonic Wyllie (DT - DTma) / (DTfl - DTma) and Raymer-Hunt-Gardner
        0.625 (DT - DTma) / DT; neutron-density PHIT = (PHID + NPHI) / 2, or sqrt((PHID^2 + NPHI^2) / 2) where gas
        crossover (PHID - NPHI > gas_xover); PHIE = shale-corrected ND (each log minus Vsh x its shale value) or
        PHIT - Vsh x PHIT_sh.
  Rw    Rwa = PHIE^m Rt / a (minimum = P5 in clean rock), Pickett water-line fit (Rw and optionally m), SP (SSP = -K
        log10(Rmfe / Rwe), K = 61 + 0.133 T degF, Rmfe = 0.85 Rmf; flagged approximate), Arps temperature conversion.
  Sw    Archie; Simandoux (1963, n = 2 closed form); modified Simandoux (Bardon-Pied, any n, solved); Indonesia
        (Poupon-Leveaux); Waxman-Smits (B from Juhasz 1981, Qv per zone); dual water (Clavier 1984, Swb from Vsh).
  Perm  Timur 8581 PHI^4.4 / Swi^2; Coates ((100 PHI / C)^2 (1 - Swi) / Swi)^2, C = 10; Wyllie-Rose (C PHI^3 / Swi)^2.
  TOC   Passey dlogR = log10(Rt / Rt_base) + 0.02 (DT - DT_base); TOC = dlogR 10^(2.297 - 0.1688 LOM).
  Cutoffs Vsh <= vsh_cut and PHIE >= phi_cut -> net reservoir; plus Sw <= sw_cut -> net pay.  Per zone: gross, net
        res, net pay (MD and TVD), N/G, averages, PHIH, SoPhiH (= HCPV per unit area), kh; cutoff sensitivity curves.

Parameter sets: ``DEFAULTS`` < project default < zone < well+zone, stored as ``<project>_petro/params.json``
({"default": {...}, "zones": {zone: {...}}, "wells": {well: {zone: {...}}}}).  None = auto (GR baselines from the
well, Rw from the Rwa minimum, shale points from the Vsh > 0.8 samples, ...).  The values actually used are reported.
"""
import copy
import csv
import json
import os

import numpy as np

from ..lithology import gr_baselines
from .store import petro_dir, save_curves

DEFAULTS = {
    "vsh_method": "gr_linear",  # gr_linear | larionov_tertiary | larionov_old | clavier | steiber | sp | nd | min
    "gr_clean": None, "gr_shale": None, "sp_clean": None, "sp_shale": None,
    "phi_method": "nd",  # nd | density | sonic_wyllie | sonic_rhg
    "rho_ma": 2.65, "rho_fl": 1.0, "dt_ma": 55.5, "dt_fl": 189.0, "nphi_shift": 0.0,
    "rho_sh": None, "nphi_sh": None, "phit_sh": None, "gas_xover": 0.03, "phie_method": "shale_corrected",
    "sw_method": "archie",  # archie | simandoux | modified_simandoux | indonesia | waxman_smits | dual_water
    "a": 1.0, "m": 2.0, "n": 2.0, "rw": None, "rw_temp": None, "rw_method": "rwa_min",  # rwa_min | pickett | sp | value
    "rmf": None, "rmf_temp": None, "rsh": None, "qv": None, "rwb": None,
    "surface_temp": 15.0, "gradient": 30.0,  # degC, degC/km (TVDSS) - or a temperature curve
    "perm_method": "timur", "perm_c": None, "swi_min": 0.05, "fit_m": False,
    "toc": False, "rt_base": None, "dt_base": None, "lom": 10.0,
    "vsh_cut": 0.4, "phi_cut": 0.08, "sw_cut": 0.6,
}

METHOD_CHOICES = {
    "vsh_method": ["gr_linear", "larionov_tertiary", "larionov_old", "clavier", "steiber", "sp", "nd", "min"],
    "phi_method": ["nd", "density", "sonic_wyllie", "sonic_rhg"],
    "sw_method": ["archie", "simandoux", "modified_simandoux", "indonesia", "waxman_smits", "dual_water"],
    "rw_method": ["rwa_min", "pickett", "sp", "value"],
    "perm_method": ["timur", "coates", "wyllie_rose"],
}


# ------------------------------------------------------------------------------------------------- Vsh
def igr(gr, gr_clean, gr_shale):
    span = np.asarray(gr_shale, float) - np.asarray(gr_clean, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip((np.asarray(gr, float) - gr_clean) / np.where(np.abs(span) > 1e-9, span, np.nan), 0.0, 1.0)


def vsh_from_igr(i, method="gr_linear"):
    i = np.asarray(i, float)
    if method in ("gr_linear", "linear", "sp", "nd"):
        v = i
    elif method == "larionov_tertiary":
        v = 0.083 * (2.0 ** (3.7 * i) - 1.0)
    elif method == "larionov_old":
        v = 0.33 * (2.0 ** (2.0 * i) - 1.0)
    elif method == "clavier":
        v = 1.7 - np.sqrt(np.maximum(3.38 - (i + 0.7) ** 2, 0.0))
    elif method == "steiber":
        v = i / (3.0 - 2.0 * i)
    else:
        raise ValueError(f"unknown Vsh method {method}")
    return np.clip(v, 0.0, 1.0)


def vsh_sp(sp, sp_clean, sp_shale):
    return igr(sp, sp_clean, sp_shale)


def phi_density(rhob, rho_ma=2.65, rho_fl=1.0):
    return (rho_ma - np.asarray(rhob, float)) / (rho_ma - rho_fl)


def vsh_nd(nphi, rhob, nphi_sh, rho_sh, rho_ma=2.65, rho_fl=1.0):
    """(NPHI - PHID) / (NPHI_sh - PHID_sh): the shale's neutron-density separation scales the sample's."""
    phid = phi_density(rhob, rho_ma, rho_fl)
    phid_sh = phi_density(rho_sh, rho_ma, rho_fl)
    sep_sh = nphi_sh - phid_sh
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip((np.asarray(nphi, float) - phid) / (sep_sh if abs(sep_sh) > 1e-6 else np.nan), 0.0, 1.0)


# ------------------------------------------------------------------------------------------------- porosity
def phi_sonic_wyllie(dt, dt_ma=55.5, dt_fl=189.0):
    return (np.asarray(dt, float) - dt_ma) / (dt_fl - dt_ma)


def phi_sonic_rhg(dt, dt_ma=55.5, c=0.625):
    dt = np.asarray(dt, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return c * (dt - dt_ma) / dt


def phi_nd(phid, nphi, gas_xover=0.03):
    """PHIT from density + neutron porosity (same matrix units).  Where PHID exceeds NPHI by more than gas_xover
    (gas crossover) the root-mean-square form is used.  Returns (phit, gas flag)."""
    phid, nphi = np.asarray(phid, float), np.asarray(nphi, float)
    gas = np.isfinite(phid) & np.isfinite(nphi) & (phid - nphi > gas_xover)
    phit = np.where(gas, np.sqrt(np.maximum((phid ** 2 + nphi ** 2) / 2.0, 0.0)), 0.5 * (phid + nphi))
    one = np.isfinite(phid) ^ np.isfinite(nphi)  # only one log present
    phit = np.where(one, np.where(np.isfinite(phid), phid, nphi), phit)
    return phit, gas


# ------------------------------------------------------------------------------------------------- Rw, temperature
def temperature(tvdss_m, surface_temp=15.0, gradient=30.0):
    return surface_temp + gradient * np.maximum(np.asarray(tvdss_m, float), 0.0) / 1000.0


def arps(r1, t1, t2):
    """Resistivity at temperature t2 (degC) from r1 at t1 (Arps)."""
    return np.asarray(r1, float) * (np.asarray(t1, float) + 21.5) / (np.asarray(t2, float) + 21.5)


def rwa(rt, phi, a=1.0, m=2.0):
    with np.errstate(invalid="ignore"):
        return np.asarray(phi, float) ** m * np.asarray(rt, float) / a


def rw_from_rwa(rt, phi, vsh=None, vsh_max=0.15, phi_min=0.08, a=1.0, m=2.0, pct=5.0):
    """Rw = low percentile of Rwa in clean, porous rock (assumes some water-bearing rock in the interval)."""
    r = rwa(rt, phi, a, m)
    sel = np.isfinite(r) & (np.asarray(phi) >= phi_min) & (r > 0)
    if vsh is not None:
        sel &= np.nan_to_num(np.asarray(vsh, float), nan=1.0) <= vsh_max
    return float(np.percentile(r[sel], pct)) if sel.sum() >= 10 else None


def pickett_fit(rt, phi, sel=None, m=None, a=1.0, iterations=6):
    """Water line on the Pickett plot (log Rt vs log phi): log10 Rt = log10(a Rw) - m log10 phi.
    Iteratively keeps the points at or below the line (lowest Rt per phi = water-bearing) and refits.
    m=None fits m too.  Returns {"rw", "m", "n_points", "x", "y"} (x, y = log10 phi, log10 Rt of the points)."""
    rt, phi = np.asarray(rt, float), np.asarray(phi, float)
    ok = np.isfinite(rt) & np.isfinite(phi) & (rt > 0) & (phi > 0.02)
    if sel is not None:
        ok &= np.asarray(sel, bool)
    if ok.sum() < 10:
        return None
    x, y = np.log10(phi[ok]), np.log10(rt[ok])
    keep = np.ones(x.size, bool)
    mm, c = (m if m is not None else 2.0), None
    for _ in range(iterations):
        if keep.sum() < 5:
            break
        if m is None:
            A = np.column_stack([np.ones(keep.sum()), -x[keep]])
            c, mm = np.linalg.lstsq(A, y[keep], rcond=None)[0]
            mm = float(np.clip(mm, 1.0, 3.5))
        c = float(np.median(y[keep] + mm * x[keep]))
        res = y - (c - mm * x)
        keep = res <= np.percentile(res, 30)  # lower envelope = water line
    c = _lower_mode(y + mm * x)
    return {"rw": 10 ** c / a, "m": float(mm), "n_points": int(ok.sum()), "x": x, "y": y}


def _lower_mode(r, width=0.02):
    """Mode of the lower half of the per-point intercepts (water-bearing points sit on the lowest dense line;
    hydrocarbon-bearing points scatter above it)."""
    r = np.sort(np.asarray(r, float))
    lo = r[: max(5, len(r) // 2)]
    if np.ptp(lo) < width:
        return float(np.median(lo))
    h, e = np.histogram(lo, bins=max(5, int(np.ptp(lo) / width)))
    k = int(np.argmax(h))
    c = 0.5 * (e[k] + e[k + 1])
    near = lo[np.abs(lo - c) <= 1.5 * width]
    return float(near.mean()) if near.size else c


def rw_from_sp(ssp_mv, rmf, temp_c, rmf_temp=None):
    """Rw from the static SP: SSP = -K log10(Rmfe / Rwe), K = 61 + 0.133 T(degF), Rmfe = 0.85 Rmf (Rmf at
    formation temperature).  Rw ~ Rwe (no Rwe -> Rw chart correction; flagged approximate)."""
    if rmf_temp is not None:
        rmf = float(arps(rmf, rmf_temp, temp_c))
    tf = temp_c * 9.0 / 5.0 + 32.0
    K = 61.0 + 0.133 * tf
    rmfe = 0.85 * rmf
    return float(rmfe / 10 ** (-ssp_mv / K))


def juhasz_b(temp_c, rw):
    """Waxman-Smits equivalent counter-ion conductance B (S m2 / meq), Juhasz (1981)."""
    t = np.asarray(temp_c, float)
    return (-1.28 + 0.225 * t - 0.0004059 * t * t) / (1.0 + np.asarray(rw, float) ** 1.23 * (0.045 * t - 0.27))


# ------------------------------------------------------------------------------------------------- Sw
def sw_archie(rt, rw, phi, a=1.0, m=2.0, n=2.0):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip((a * rw / (np.asarray(phi, float) ** m * np.asarray(rt, float))) ** (1.0 / n), 0.0, 1.0)


def sw_simandoux(rt, rw, phi, vsh, rsh, a=1.0, m=2.0):
    """Simandoux (1963), n = 2: Sw = (a Rw / (2 phi^m)) [sqrt((Vsh/Rsh)^2 + 4 phi^m / (a Rw Rt)) - Vsh/Rsh]."""
    phi, rt, vsh = (np.asarray(x, float) for x in (phi, rt, vsh))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = vsh / rsh
        pm = phi ** m
        sw = (a * rw / (2.0 * pm)) * (np.sqrt(c * c + 4.0 * pm / (a * rw * rt)) - c)
    return np.clip(sw, 0.0, 1.0)


def solve_sw(ct_of_sw, ct, lo=1e-4, hi=1.0, iters=50):
    """Vectorised bisection for Sw in [lo, hi] with ct_of_sw(sw) increasing in sw (conductivity model)."""
    ct = np.asarray(ct, float)
    a = np.full(ct.shape, lo)
    b = np.full(ct.shape, hi)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        f_hi = ct_of_sw(b) - ct
        for _ in range(iters):
            mid = 0.5 * (a + b)
            f = ct_of_sw(mid) - ct
            up = f < 0
            a = np.where(up, mid, a)
            b = np.where(up, b, mid)
    sw = 0.5 * (a + b)
    sw = np.where(f_hi < 0, 1.0, sw)  # model cannot reach Ct even at Sw = 1 -> wet
    sw[~np.isfinite(ct)] = np.nan
    return sw


def sw_modified_simandoux(rt, rw, phi, vsh, rsh, a=1.0, m=2.0, n=2.0):
    """Bardon & Pied (1969): 1/Rt = phi^m Sw^n / (a Rw (1 - Vsh)) + Vsh Sw / Rsh."""
    phi, rt, vsh = (np.asarray(x, float) for x in (phi, rt, vsh))
    pm = phi ** m

    def ct(sw):
        return pm * sw ** n / (a * rw * np.maximum(1.0 - vsh, 1e-3)) + vsh * sw / rsh

    out = solve_sw(ct, 1.0 / rt)
    out[~(np.isfinite(phi) & np.isfinite(vsh))] = np.nan
    return out


def sw_indonesia(rt, rw, phi, vsh, rsh, a=1.0, m=2.0, n=2.0):
    """Poupon-Leveaux: 1/sqrt(Rt) = [Vsh^(1 - Vsh/2) / sqrt(Rsh) + phi^(m/2) / sqrt(a Rw)] Sw^(n/2)."""
    phi, rt, vsh = (np.asarray(x, float) for x in (phi, rt, vsh))
    with np.errstate(invalid="ignore", divide="ignore"):
        term = vsh ** (1.0 - vsh / 2.0) / np.sqrt(rsh) + phi ** (m / 2.0) / np.sqrt(a * rw)
        sw = (1.0 / (np.sqrt(rt) * term)) ** (2.0 / n)
    return np.clip(sw, 0.0, 1.0)


def sw_waxman_smits(rt, rw, phi, qv, temp_c, a=1.0, m=2.0, n=2.0, b=None):
    """Waxman-Smits: Ct = phi^m Sw^n / a (Cw + B Qv / Sw).  qv in meq/cc (scalar or array)."""
    phi, rt = np.asarray(phi, float), np.asarray(rt, float)
    B = juhasz_b(temp_c, rw) if b is None else b
    cw = 1.0 / rw
    pm = phi ** m

    def ct(sw):
        return pm * sw ** n / a * (cw + B * qv / sw)

    out = solve_sw(ct, 1.0 / rt)
    out[~np.isfinite(phi)] = np.nan
    return out


def sw_dual_water(rt, rw, phit, swb, rwb, a=1.0, m=2.0, n=2.0):
    """Clavier dual water: Ct = phit^m Swt^n / a [Cw + (Swb / Swt)(Cwb - Cw)].  Returns total Sw (Swt >= Swb)."""
    phit, rt, swb = (np.asarray(x, float) for x in (phit, rt, swb))
    cw, cwb = 1.0 / rw, 1.0 / rwb
    pm = phit ** m

    def ct(sw):
        swt = np.maximum(sw, swb)
        return pm * swt ** n / a * (cw + (swb / swt) * (cwb - cw))

    out = solve_sw(ct, 1.0 / rt)
    out = np.maximum(out, swb)
    out[~np.isfinite(phit)] = np.nan
    return np.clip(out, 0.0, 1.0)


# ------------------------------------------------------------------------------------------------- perm, TOC
def perm_timur(phi, swi):
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.136 * 10 ** 4.8 * np.maximum(np.asarray(phi, float), 0) ** 4.4 / np.asarray(swi, float) ** 2


def perm_coates(phi, swi, c=10.0):
    swi = np.asarray(swi, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((100.0 * np.maximum(np.asarray(phi, float), 0) / c) ** 2 * (1.0 - swi) / swi) ** 2


def perm_wyllie_rose(phi, swi, c=250.0):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (c * np.maximum(np.asarray(phi, float), 0) ** 3 / np.asarray(swi, float)) ** 2


def toc_passey(rt, dt, rt_base, dt_base, lom=10.0):
    with np.errstate(invalid="ignore", divide="ignore"):
        dlr = np.log10(np.asarray(rt, float) / rt_base) + 0.02 * (np.asarray(dt, float) - dt_base)
    return np.maximum(dlr * 10 ** (2.297 - 0.1688 * lom), 0.0), dlr


# ------------------------------------------------------------------------------------------------- parameter sets
def params_path(project):
    d = petro_dir(project)
    return os.path.join(d, "params.json") if d else None


def load_params(project):
    p = params_path(project)
    if p and os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
    else:
        d = {}
    d.setdefault("default", {})
    d.setdefault("zones", {})
    d.setdefault("wells", {})
    return d


def save_params(project, pset):
    petro_dir(project, create=True)
    p = params_path(project)
    with open(p + ".tmp", "w") as f:
        json.dump(pset, f, indent=1)
    os.replace(p + ".tmp", p)
    return p


def resolve(pset, well, zone):
    """Parameters for one well + zone: DEFAULTS < default < zones[zone] < wells[well][zone] (and wells[well]['*'])."""
    out = dict(DEFAULTS)
    pset = pset or {}
    out.update(pset.get("default", {}))
    out.update(pset.get("zones", {}).get(zone, {}) if zone else {})
    wz = pset.get("wells", {}).get(well, {})
    out.update(wz.get("*", {}))
    if zone:
        out.update(wz.get(zone, {}))
    return out


def zones_of(project, name, depth):
    """[(zone name, top MD, base MD)] from the tops (formation order; a zone runs to the next top present).
    Samples above the first / below the last top get the zone None (default parameters)."""
    tops = project.tops.get(name, {})
    order = [f["name"] for f in project.formations]
    present = sorted([(tops[f], f) for f in order if f in tops] + [(v, k) for k, v in tops.items() if k not in order])
    z = []
    for (t, f), nxt in zip(present, present[1:] + [(float(depth[-1]) + 1e-6 if len(depth) else t, None)]):
        b = nxt[0]
        if b > t:
            z.append((f, float(t), float(b)))
    return z


def _shale_point(x, vsh, thr=0.8):
    sel = np.isfinite(x) & (np.nan_to_num(vsh, nan=0) >= thr)
    if sel.sum() < 10:
        sel = np.isfinite(x) & (np.nan_to_num(vsh, nan=0) >= np.nanpercentile(vsh, 90)) if np.isfinite(vsh).any() else sel
    return float(np.median(x[sel])) if sel.sum() >= 3 else None


# ------------------------------------------------------------------------------------------------- the well run
CURVES_OUT = ["VSH", "PHIT", "PHIE", "SW", "SWT", "BVW", "PERM", "RWA", "TEMP", "GAS_FLAG", "NET_RES", "NET_PAY", "TOC",
              "DLOGR", "ZONE_ID"]


def interpret_well(project, name, pset=None, save=True, temp_curve=None):
    """Run the interpretation on a well.  Returns a result dict:
    {"well", "depth", "tvdss", "curves" {name: array}, "zones" [summary row], "used" {zone: params}, "notes" []}.
    temp_curve: optional formation temperature (degC) on the well's MD grid (e.g. from the basin model)."""
    w = project.load_well(name)
    pset = load_params(project) if pset is None else pset
    z = w.depth
    n = len(z)
    res = {"well": name, "depth": z, "curves": {}, "zones": [], "used": {}, "notes": []}
    if z is None or not n:
        res["notes"].append("no logs")
        return res
    tvdss = w.md_to_tvdss(z)
    res["tvdss"] = tvdss
    res["rt"] = np.asarray(w.logs.get("RDEEP", np.full(n, np.nan)), float)
    L = {k: np.asarray(v, float) for k, v in w.logs.items()}
    nan = np.full(n, np.nan)
    gr, rt, rhob, nphi, dt, sp = (L.get(k, nan) for k in ("GR", "RDEEP", "RHOB", "NPHI", "DT", "SP"))
    out = {k: np.full(n, np.nan) for k in CURVES_OUT}
    out["ZONE_ID"] = np.full(n, -1.0)
    zones = zones_of(project, name, z)
    segs = [(None, -np.inf, zones[0][1] if zones else np.inf)] + zones
    if zones:
        segs.append((None, zones[-1][2], np.inf))
    temp_all = temp_curve if temp_curve is not None else None
    gr_lo_auto, gr_hi_auto = gr_baselines(z, gr) if np.isfinite(gr).sum() > 50 else (nan, nan)
    for zi, (zone, top, base) in enumerate(segs):
        m = (z >= top) & (z < base)
        if not m.any():
            continue
        P = resolve(pset, name, zone)
        used = {}
        idx = np.where(m)[0]
        # temperature
        T = (temp_all[idx] if temp_all is not None else temperature(tvdss[idx], P["surface_temp"], P["gradient"]))
        out["TEMP"][idx] = T
        # ---- Vsh
        gcl = P["gr_clean"] if P["gr_clean"] is not None else gr_lo_auto[idx]
        gsh = P["gr_shale"] if P["gr_shale"] is not None else gr_hi_auto[idx]
        used["gr_clean"] = _summ(gcl)
        used["gr_shale"] = _summ(gsh)
        meth = P["vsh_method"]
        cand = {}
        if np.isfinite(gr[idx]).any():
            cand["gr"] = vsh_from_igr(igr(gr[idx], gcl, gsh), meth if meth not in ("sp", "nd", "min") else "gr_linear")
        vsh_gr = cand.get("gr", np.full(len(idx), np.nan))
        if (meth in ("sp", "min")) and np.isfinite(sp[idx]).sum() > 10:
            scl = P["sp_clean"] if P["sp_clean"] is not None else float(np.nanpercentile(sp[idx], 5))
            ssh = P["sp_shale"] if P["sp_shale"] is not None else float(np.nanpercentile(sp[idx], 95))
            cand["sp"] = vsh_sp(sp[idx], scl, ssh)
            used["sp_clean"], used["sp_shale"] = scl, ssh
        nph = nphi[idx] + P["nphi_shift"]
        rho_sh = P["rho_sh"] if P["rho_sh"] is not None else _shale_point(rhob[idx], vsh_gr)
        nphi_sh = P["nphi_sh"] if P["nphi_sh"] is not None else _shale_point(nph, vsh_gr)
        used["rho_sh"], used["nphi_sh"] = rho_sh, nphi_sh
        if meth in ("nd", "min") and rho_sh is not None and nphi_sh is not None and np.isfinite(rhob[idx]).any():
            cand["nd"] = vsh_nd(nph, rhob[idx], nphi_sh, rho_sh, P["rho_ma"], P["rho_fl"])
        if meth == "min" and cand:
            vsh = np.nanmin(np.stack(list(cand.values())), axis=0)
        elif meth in ("sp", "nd"):
            vsh = cand.get(meth, vsh_gr)
            if meth not in cand:
                res["notes"].append(f"{zone}: Vsh method {meth} not possible (logs / shale point missing) - used GR")
        else:
            vsh = vsh_gr
        out["VSH"][idx] = vsh
        # ---- porosity
        pm = P["phi_method"]
        phid = phi_density(rhob[idx], P["rho_ma"], P["rho_fl"])
        phid_sh = phi_density(rho_sh, P["rho_ma"], P["rho_fl"]) if rho_sh is not None else 0.0
        gas = np.zeros(len(idx), bool)
        if pm == "nd":
            phit, gas = phi_nd(phid, nph, P["gas_xover"])
            phit_sh_auto = 0.5 * (phid_sh + (nphi_sh or 0.0)) if rho_sh is not None else 0.0
            if P["phie_method"] == "shale_corrected" and rho_sh is not None and nphi_sh is not None:
                pdc = phid - vsh * phid_sh
                pnc = nph - vsh * nphi_sh
                phie, _ = phi_nd(pdc, pnc, P["gas_xover"])
            else:
                phie = phit - vsh * (P["phit_sh"] if P["phit_sh"] is not None else phit_sh_auto)
            if not np.isfinite(phit).any() and np.isfinite(dt[idx]).any():
                pm = "sonic_wyllie"
                res["notes"].append(f"{zone}: no density / neutron - sonic porosity used")
        if pm != "nd":
            if pm == "density":
                phit = phid
                phi_sh = phid_sh
            elif pm == "sonic_wyllie":
                phit = phi_sonic_wyllie(dt[idx], P["dt_ma"], P["dt_fl"])
                phi_sh = _shale_point(phit, vsh) or 0.0
            elif pm == "sonic_rhg":
                phit = phi_sonic_rhg(dt[idx], P["dt_ma"])
                phi_sh = _shale_point(phit, vsh) or 0.0
            else:
                raise ValueError(f"unknown porosity method {pm}")
            phie = phit - vsh * (P["phit_sh"] if P["phit_sh"] is not None else phi_sh)
        phit = np.clip(phit, 0.0, 0.6)
        phie = np.clip(np.minimum(phie, phit), 0.0, 0.6)
        out["PHIT"][idx], out["PHIE"][idx], out["GAS_FLAG"][idx] = phit, phie, gas.astype(float)
        phit_sh = P["phit_sh"] if P["phit_sh"] is not None else (_shale_point(phit, vsh) or 0.0)
        used["phit_sh"] = phit_sh
        # ---- Rw (at formation temperature)
        rtz = rt[idx]
        rw = P["rw"]
        how = "value"
        if rw is not None and P["rw_temp"] is not None:
            rw = arps(rw, P["rw_temp"], T)
            how = "value at temperature (Arps)"
        elif rw is None:
            if P["rw_method"] == "pickett":
                fit = pickett_fit(rtz, phie, sel=np.nan_to_num(vsh, nan=1) <= 0.2, m=None if P.get("fit_m") else P["m"], a=P["a"])
                rw, how = (fit["rw"], "Pickett water line") if fit else (None, "")
                if fit and P.get("fit_m"):
                    P["m"] = fit["m"]
            elif P["rw_method"] == "sp" and P["rmf"] is not None and np.isfinite(sp[idx]).sum() > 10:
                ssp = float(np.nanpercentile(sp[idx], 5) - np.nanpercentile(sp[idx], 95))
                rw = rw_from_sp(ssp, P["rmf"], float(np.nanmedian(T)), P["rmf_temp"])
                how = f"SP (SSP {ssp:.0f} mV, approximate)"
            if rw is None:
                rw = rw_from_rwa(rtz, phie, vsh, a=P["a"], m=P["m"])
                how = "Rwa minimum (P5, clean porous rock)"
            if rw is None:
                rw = 0.05
                how = "fallback 0.05 (no clean porous rock to estimate it)"
                res["notes"].append(f"{zone}: Rw could not be estimated - 0.05 ohm.m used, set it in the zone parameters")
        used["rw"] = _summ(rw)
        used["rw_how"] = how
        out["RWA"][idx] = rwa(rtz, phie, P["a"], P["m"])
        # ---- Sw
        sm = P["sw_method"]
        rsh = P["rsh"] if P["rsh"] is not None else _shale_point(rtz, vsh)
        if rsh is None or not np.isfinite(rsh) or rsh <= 0:
            rsh = 2.0
        used["rsh"] = rsh
        a_, m_, n_ = P["a"], P["m"], P["n"]
        if sm == "archie":
            sw = sw_archie(rtz, rw, phie, a_, m_, n_)
            swt = sw
        elif sm == "simandoux":
            sw = sw_simandoux(rtz, rw, phie, vsh, rsh, a_, m_)
            swt = sw
        elif sm == "modified_simandoux":
            sw = sw_modified_simandoux(rtz, rw, phie, vsh, rsh, a_, m_, n_)
            swt = sw
        elif sm == "indonesia":
            sw = sw_indonesia(rtz, rw, phie, vsh, rsh, a_, m_, n_)
            swt = sw
        elif sm == "waxman_smits":
            qv = P["qv"] if P["qv"] is not None else 0.3 * vsh  # meq/cc, crude Vsh proxy unless given
            used["qv"] = _summ(qv)
            sw = sw_waxman_smits(rtz, rw, np.maximum(phit, 1e-4), qv, T, a_, m_, n_)
            swt = sw
        elif sm == "dual_water":
            with np.errstate(invalid="ignore", divide="ignore"):
                swb = np.clip(vsh * phit_sh / np.maximum(phit, 1e-4), 0.0, 1.0)
            rwb = P["rwb"] if P["rwb"] is not None else (rsh * phit_sh ** m_ / a_ if phit_sh > 0 else rw)
            used["rwb"] = _summ(rwb)
            swt = sw_dual_water(rtz, rw, np.maximum(phit, 1e-4), swb, rwb, a_, m_, n_)
            with np.errstate(invalid="ignore", divide="ignore"):
                sw = np.clip(1.0 - phit * (1.0 - swt) / np.maximum(phie, 1e-4), 0.0, 1.0)  # effective Sw
        else:
            raise ValueError(f"unknown Sw method {sm}")
        sw = np.where(phie > 0.005, sw, 1.0)
        sw[~np.isfinite(rtz)] = np.nan
        out["SW"][idx], out["SWT"][idx] = sw, swt
        out["BVW"][idx] = phie * sw
        # ---- perm
        swi = np.clip(np.nan_to_num(sw, nan=1.0), P["swi_min"], 1.0)
        pmeth = P["perm_method"]
        c = P["perm_c"]
        if pmeth == "timur":
            k = perm_timur(phie, swi)
        elif pmeth == "coates":
            k = perm_coates(phie, swi, c or 10.0)
        elif pmeth == "wyllie_rose":
            k = perm_wyllie_rose(phie, swi, c or 250.0)
        else:
            raise ValueError(f"unknown perm method {pmeth}")
        out["PERM"][idx] = np.where(phie > 0, k, 0.0)
        # ---- TOC
        if P["toc"]:
            shale = np.nan_to_num(vsh, nan=0) >= 0.6
            rb = P["rt_base"] if P["rt_base"] is not None else (float(np.nanpercentile(rtz[shale], 20)) if shale.sum() > 10 else None)
            db = P["dt_base"] if P["dt_base"] is not None else (float(np.nanpercentile(dt[idx][shale], 20)) if shale.sum() > 10 else None)
            if rb and db:
                toc, dlr = toc_passey(rtz, dt[idx], rb, db, P["lom"])
                out["TOC"][idx], out["DLOGR"][idx] = toc, dlr
                used["rt_base"], used["dt_base"] = rb, db
        # ---- flags
        netres = (np.nan_to_num(vsh, nan=1) <= P["vsh_cut"]) & (np.nan_to_num(phie, nan=0) >= P["phi_cut"])
        pay = netres & (np.nan_to_num(sw, nan=1) <= P["sw_cut"])
        out["NET_RES"][idx], out["NET_PAY"][idx] = netres, pay
        out["ZONE_ID"][idx] = zi
        if zone is not None:
            used.update({k: P[k] for k in ("vsh_method", "phi_method", "sw_method", "a", "m", "n", "rho_ma", "rho_fl",
                                           "vsh_cut", "phi_cut", "sw_cut", "perm_method")})
            res["used"][zone] = used
            res["zones"].append(zone_summary(name, zone, top, base, z[idx], tvdss[idx], out, idx))
    res["curves"] = out
    if save and petro_dir(project):
        save_curves(project, name, z, {k: v.astype(np.float32) for k, v in out.items()})
        write_summary(project, name, res["zones"])
        res["saved"] = True
    return res


def _summ(v):
    a = np.asarray(v, float)
    if a.ndim == 0:
        return float(a)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else None


SUMMARY_COLUMNS = ["Well", "Zone", "Top_MD", "Base_MD", "Gross_MD", "Gross_TVD", "NetRes_MD", "NetRes_TVD", "NetPay_MD",
                   "NetPay_TVD", "NtG_res", "NtG_pay", "Vsh_avg", "PHIE_res", "PHIE_pay", "Sw_pay", "PHIH_res",
                   "PHIH_pay", "SoPhiH", "kh_pay", "Perm_pay_geo", "TOC_avg", "Coverage_pct"]


def zone_summary(well, zone, top, base, md, tvdss, out, idx, cuts=None):
    """Summary row for one zone from the computed curves (idx = sample indices of the zone)."""
    step = np.diff(md, append=md[-1] + (md[-1] - md[-2] if len(md) > 1 else 0.1524)) if len(md) else md
    dh = np.abs(np.diff(tvdss, append=tvdss[-1] + (tvdss[-1] - tvdss[-2] if len(tvdss) > 1 else 0.1524))) if len(md) else md
    vsh, phie, sw, k = out["VSH"][idx], out["PHIE"][idx], out["SW"][idx], out["PERM"][idx]
    if cuts is None:
        res_f, pay_f = out["NET_RES"][idx] > 0.5, out["NET_PAY"][idx] > 0.5
    else:
        vc, pc, sc = cuts
        res_f = (np.nan_to_num(vsh, nan=1) <= vc) & (np.nan_to_num(phie, nan=0) >= pc)
        pay_f = res_f & (np.nan_to_num(sw, nan=1) <= sc)
    have = np.isfinite(vsh) & np.isfinite(phie)
    gross_md, gross_tvd = float(step.sum()), float(dh.sum())

    def wavg(x, sel, wts=None):
        wts = dh if wts is None else wts
        s = sel & np.isfinite(x)
        return float((x[s] * wts[s]).sum() / wts[s].sum()) if s.any() and wts[s].sum() > 0 else None

    pv = phie * dh
    sw_pay = float((sw[pay_f] * pv[pay_f]).sum() / pv[pay_f].sum()) if pay_f.any() and pv[pay_f].sum() > 0 else None
    kp = k[pay_f & (k > 0)]
    toc = out["TOC"][idx]
    return {
        "Well": well, "Zone": zone, "Top_MD": round(top, 2), "Base_MD": round(base, 2), "Gross_MD": round(gross_md, 2),
        "Gross_TVD": round(gross_tvd, 2), "NetRes_MD": round(float(step[res_f].sum()), 2),
        "NetRes_TVD": round(float(dh[res_f].sum()), 2), "NetPay_MD": round(float(step[pay_f].sum()), 2),
        "NetPay_TVD": round(float(dh[pay_f].sum()), 2),
        "NtG_res": round(float(dh[res_f].sum()) / gross_tvd, 3) if gross_tvd > 0 else None,
        "NtG_pay": round(float(dh[pay_f].sum()) / gross_tvd, 3) if gross_tvd > 0 else None,
        "Vsh_avg": _rnd(wavg(vsh, have)), "PHIE_res": _rnd(wavg(phie, res_f)), "PHIE_pay": _rnd(wavg(phie, pay_f)),
        "Sw_pay": _rnd(sw_pay), "PHIH_res": round(float(np.nansum(pv[res_f])), 3), "PHIH_pay": round(float(np.nansum(pv[pay_f])), 3),
        "SoPhiH": round(float(np.nansum((phie * (1 - sw) * dh)[pay_f])), 3),
        "kh_pay": round(float(np.nansum((k * dh)[pay_f])), 1),
        "Perm_pay_geo": _rnd(float(np.exp(np.log(kp).mean())) if kp.size else None, 2),
        "TOC_avg": _rnd(float(np.nanmean(toc)) if np.isfinite(toc).any() else None),
        "Coverage_pct": round(100.0 * have.sum() / max(1, len(idx)), 1),
    }


def _rnd(v, k=3):
    return None if v is None else round(v, k)


def summary_path(project):
    d = petro_dir(project)
    return os.path.join(d, "summary.csv") if d else None


def read_summary(project):
    p = summary_path(project)
    if not p or not os.path.exists(p):
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def write_summary(project, well, rows):
    """Replace the well's rows in <project>_petro/summary.csv (other wells kept)."""
    petro_dir(project, create=True)
    old = [r for r in read_summary(project) if r.get("Well") != well]
    p = summary_path(project)
    with open(p + ".tmp", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        wr.writeheader()
        for r in old + rows:
            wr.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in SUMMARY_COLUMNS})
    os.replace(p + ".tmp", p)
    return p


def interpret_project(project, wells=None, pset=None, progress=None, save=True):
    out = {}
    names = list(wells or project.wells)
    for i, n in enumerate(names):
        try:
            out[n] = interpret_well(project, n, pset, save)
        except Exception as e:
            out[n] = {"well": n, "zones": [], "notes": [f"{type(e).__name__}: {e}"]}
        if progress:
            progress(i + 1, len(names), n)
    return out


# ------------------------------------------------------------------------------------------------- sensitivity
def cutoff_sensitivity(result, zone, ranges=None):
    """IP-style cutoff sensitivity: vary one cutoff at a time (others at the zone's values).
    Returns {cutoff: {"values", "net_pay_tvd", "phih_pay", "sophih"}}."""
    z = result["depth"]
    out = result["curves"]
    used = result["used"].get(zone)
    if used is None:
        raise KeyError(f"zone {zone} not interpreted")
    zrow = next(r for r in result["zones"] if r["Zone"] == zone)
    idx = np.where((z >= zrow["Top_MD"] - 1e-6) & (z < zrow["Base_MD"]))[0]
    base = (used["vsh_cut"], used["phi_cut"], used["sw_cut"])
    ranges = ranges or {"vsh_cut": np.linspace(0.1, 0.8, 15), "phi_cut": np.linspace(0.0, 0.3, 31),
                        "sw_cut": np.linspace(0.2, 1.0, 17)}
    res = {}
    for key, vals in ranges.items():
        j = ["vsh_cut", "phi_cut", "sw_cut"].index(key)
        npay, phih, soph = [], [], []
        for v in vals:
            c = list(base)
            c[j] = float(v)
            row = zone_summary(result["well"], zone, zrow["Top_MD"], zrow["Base_MD"], z[idx], result["tvdss"][idx], out, idx, cuts=c)
            npay.append(row["NetPay_TVD"])
            phih.append(row["PHIH_pay"])
            soph.append(row["SoPhiH"])
        res[key] = {"values": np.asarray(vals, float), "net_pay_tvd": np.array(npay), "phih_pay": np.array(phih),
                    "sophih": np.array(soph), "base": base[j]}
    return res


def pickett_data(result, zone=None, vsh_max=0.2):
    """Points for a Pickett plot (log phi vs log Rt) + the water line of the Rw / m used."""
    w = result["curves"]
    z = result["depth"]
    sel = np.ones(len(z), bool)
    if zone is not None:
        row = next(r for r in result["zones"] if r["Zone"] == zone)
        sel = (z >= row["Top_MD"]) & (z < row["Base_MD"])
    sel &= np.nan_to_num(w["VSH"], nan=1) <= vsh_max
    u = result["used"].get(zone, {})
    return {"phi": w["PHIE"][sel], "rt": result["rt"][sel], "sw": w["SW"][sel], "rw": u.get("rw"), "m": u.get("m"),
            "a": u.get("a"), "n": u.get("n")}


def copy_params(pset):
    return copy.deepcopy(pset)
