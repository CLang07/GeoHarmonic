"""Rock physics at wells (PLAN 20d) - used later by AVO / DHI / fizz-gas / inversion items (23-25) and brittleness (30c).

Units: velocity m/s, density g/cc, moduli GPa, temperature degC, pressure MPa, salinity ppm (NaCl), GOR L/L.

  Fluids      Batzle & Wang (1992): brine (density + velocity polynomial), gas (Z factor, adiabatic K), live / dead oil.
  Mixing      Voigt-Reuss-Hill mineral moduli; fluid bulk modulus by Wood (uniform) or Brie (patchy, exponent e).
  Gassmann    K_dry from K_sat and back; shear modulus unchanged; brine / oil / gas / fizz (low gas saturation) cases.
  Vs          Greenberg-Castagna (1992) brine-rock lithology mix; for hydrocarbon zones iterated with Gassmann.
  Elastic     AI, SI, Vp/Vs, lambda-rho, mu-rho, Poisson's ratio, Young's modulus, Rickman (2008) brittleness index.
"""
import numpy as np

from . import interp as I
from .store import petro_dir, save_curves

# mineral moduli (GPa) and densities (g/cc), Mavko et al. Rock Physics Handbook (typical values)
MINERAL_MODULI = {
    "quartz": (36.6, 45.0, 2.65), "calcite": (76.8, 32.0, 2.71), "dolomite": (94.9, 45.0, 2.87),
    "clay": (21.0, 7.0, 2.58), "illite": (39.4, 11.7, 2.75), "kaolinite": (1.5, 1.4, 1.58), "k-feldspar": (37.5, 15.0, 2.62),
    "anhydrite": (56.1, 29.1, 2.98), "halite": (24.8, 14.9, 2.16), "pyrite": (147.4, 132.5, 4.93), "coal": (2.5, 1.5, 1.3),
}

# Greenberg-Castagna brine-rock Vs(Vp) in km/s: coefficients a2, a1, a0 of Vs = a2 Vp^2 + a1 Vp + a0
GC = {"sandstone": (0.0, 0.80416, -0.85588), "shale": (0.0, 0.76969, -0.86735),
      "limestone": (-0.05508, 1.01677, -1.03049), "dolomite": (0.0, 0.58321, -0.07775)}

RP_DEFAULTS = {"salinity": 30000.0, "api": 35.0, "gor": 100.0, "gas_gravity": 0.65, "rp_hc": "gas",
               "pressure_gradient": 10.1,  # MPa/km on TVDSS (hydrostatic pore pressure), or a pressure curve
               "rp_sw_hc": 0.2, "rp_fizz_sg": 0.07, "brie_e": None, "phi_min_gassmann": 0.03,
               "rp_lith": "sandstone",  # matrix of the clean fraction: sandstone | limestone | dolomite
               "e_min": 1.0, "e_max": 8.0, "pr_min": 0.15, "pr_max": 0.40}  # Rickman ranges (Mpsi, -)


# --------------------------------------------------------------------------------------------- Batzle-Wang fluids
_W = np.array([[1402.85, 1.524, 3.437e-3, -1.197e-5],
               [4.871, -0.0111, 1.739e-4, -1.628e-6],
               [-0.04783, 2.747e-4, -2.135e-6, 1.237e-8],
               [1.487e-4, -6.503e-7, -1.455e-8, 1.327e-10],
               [-2.197e-7, 7.987e-10, 5.230e-11, -4.614e-13]])


def brine(T, P, salinity_ppm=30000.0):
    """Batzle-Wang brine: (density g/cc, velocity m/s, K GPa)."""
    T, P = np.asarray(T, float), np.asarray(P, float)
    S = salinity_ppm * 1e-6
    rw = 1 + 1e-6 * (-80 * T - 3.3 * T ** 2 + 0.00175 * T ** 3 + 489 * P - 2 * T * P + 0.016 * T ** 2 * P
                     - 1.3e-5 * T ** 3 * P - 0.333 * P ** 2 - 0.002 * T * P ** 2)
    rb = rw + S * (0.668 + 0.44 * S + 1e-6 * (300 * P - 2400 * P * S + T * (80 + 3 * T - 3300 * S - 13 * P + 47 * P * S)))
    vw = np.zeros(np.broadcast(T, P).shape)
    for i in range(5):
        for j in range(4):
            vw = vw + _W[i, j] * T ** i * P ** j
    vb = vw + S * (1170 - 9.6 * T + 0.055 * T ** 2 - 8.5e-5 * T ** 3 + 2.6 * P - 0.0029 * T * P - 0.0476 * P ** 2) \
        + S ** 1.5 * (780 - 10 * P + 0.16 * P ** 2) - 1820 * S ** 2
    return rb, vb, rb * vb ** 2 * 1e-6


def _gas_z(ppr, tpr):
    e = 0.109 * (3.85 - tpr) ** 2 * np.exp(-(0.45 + 8 * (0.56 - 1 / tpr) ** 2) * ppr ** 1.2 / tpr)
    return (0.03 + 0.00527 * (3.5 - tpr) ** 3) * ppr + (0.642 * tpr - 0.007 * tpr ** 4 - 0.52) + e


def gas(T, P, G=0.65):
    """Batzle-Wang hydrocarbon gas: (density g/cc, K GPa)."""
    T, P = np.asarray(T, float), np.asarray(P, float)
    ta = T + 273.15
    ppr = P / (4.892 - 0.4048 * G)
    tpr = ta / (94.72 + 170.75 * G)
    z = _gas_z(ppr, tpr)
    rho = 28.8 * G * P / (z * 8.314 * ta)
    h = 1e-4
    dz = (_gas_z(ppr + h, tpr) - _gas_z(ppr - h, tpr)) / (2 * h)
    g0 = 0.85 + 5.6 / (ppr + 2) + 27.1 / (ppr + 3.5) ** 2 - 8.7 * np.exp(-0.65 * (ppr + 1))
    k = P * g0 / (1 - ppr / z * dz)  # MPa
    return rho, k * 1e-3


def oil(T, P, api=35.0, gor=0.0, G=0.65):
    """Batzle-Wang oil (dead if gor = 0, else live, gor in L/L): (density g/cc, velocity m/s, K GPa)."""
    T, P = np.asarray(T, float), np.asarray(P, float)
    r0 = 141.5 / (api + 131.5)
    if gor and gor > 0:
        b0 = 0.972 + 0.00038 * (2.4 * gor * np.sqrt(G / r0) + T + 17.8) ** 1.175
        rps = r0 / (b0 * (1 + 0.001 * gor))  # pseudo density for velocity
        rg = (r0 + 0.0012 * G * gor) / b0  # saturated density
    else:
        rps, rg = r0, r0
    rp = rg + (0.00277 * P - 1.71e-7 * P ** 3) * (rg - 1.15) ** 2 + 3.49e-4 * P
    rho = rp / (0.972 + 3.81e-4 * (T + 17.78) ** 1.175)
    v = 2096 * np.sqrt(rps / (2.6 - rps)) - 3.7 * T + 4.64 * P + 0.0115 * (4.12 * np.sqrt(1.08 / rps - 1) - 1) * T * P
    return rho, v, rho * v ** 2 * 1e-6


# --------------------------------------------------------------------------------------------- mixing, Gassmann
def vrh(fractions, moduli):
    """Voigt-Reuss-Hill average. fractions (..., m) summing to 1; moduli (m,). Returns the Hill average."""
    f = np.asarray(fractions, float)
    mo = np.asarray(moduli, float)
    voigt = (f * mo).sum(-1)
    with np.errstate(divide="ignore"):
        reuss = 1.0 / (f / np.where(mo > 0, mo, np.nan)).sum(-1)
    return 0.5 * (voigt + reuss)


def fluid_mix(sw, k_w, k_h, rho_w, rho_h, brie_e=None):
    """Fluid bulk modulus + density for water saturation sw: Wood (uniform) or Brie (patchy, exponent e)."""
    sw = np.asarray(sw, float)
    rho = sw * rho_w + (1 - sw) * rho_h
    if brie_e:
        k = (k_w - k_h) * sw ** brie_e + k_h
    else:
        k = 1.0 / (sw / k_w + (1 - sw) / k_h)
    return k, rho


def gassmann_dry(k_sat, k_min, k_fl, phi):
    a = phi * k_min / k_fl + 1 - phi
    return (k_sat * a - k_min) / (phi * k_min / k_fl + k_sat / k_min - 1 - phi)


def gassmann_sat(k_dry, k_min, k_fl, phi):
    return k_dry + (1 - k_dry / k_min) ** 2 / (phi / k_fl + (1 - phi) / k_min - k_dry / k_min ** 2)


def moduli(vp, vs, rho):
    """(K, mu) in GPa from m/s and g/cc."""
    vp, vs, rho = (np.asarray(x, float) for x in (vp, vs, rho))
    mu = rho * vs ** 2 * 1e-6
    return rho * vp ** 2 * 1e-6 - 4.0 / 3.0 * mu, mu


def velocities(k, mu, rho):
    k, mu, rho = (np.asarray(x, float) for x in (k, mu, rho))
    with np.errstate(invalid="ignore"):
        return np.sqrt((k + 4.0 / 3.0 * mu) / rho * 1e6), np.sqrt(mu / rho * 1e6)


def substitute(vp, vs, rho, phi, k_min, k_fl1, rho_fl1, k_fl2, rho_fl2, phi_min=0.03):
    """Gassmann fluid substitution fluid 1 -> fluid 2.  Samples with phi < phi_min are returned unchanged."""
    vp, vs, rho, phi = (np.asarray(x, float) for x in (vp, vs, rho, phi))
    k_sat, mu = moduli(vp, vs, rho)
    ok = np.isfinite(phi) & (phi >= phi_min)
    ph = np.where(ok, phi, 0.1)
    with np.errstate(invalid="ignore", divide="ignore"):
        k_dry = gassmann_dry(k_sat, k_min, k_fl1, ph)
        k_dry = np.clip(k_dry, 0.01, k_min)
        k2 = gassmann_sat(k_dry, k_min, k_fl2, ph)
    rho2 = rho + ph * (rho_fl2 - rho_fl1)
    vp2, vs2 = velocities(k2, mu, rho2)
    return (np.where(ok, vp2, vp), np.where(ok, vs2, vs), np.where(ok, rho2, rho), np.where(ok, k_dry, np.nan))


# --------------------------------------------------------------------------------------------- Vs prediction
def vs_greenberg_castagna(vp, fractions):
    """Brine-rock Vs (m/s) from Vp (m/s) and lithology fractions {sandstone|shale|limestone|dolomite: array}.
    Vs = 1/2 [sum X_i Vs_i + (sum X_i / Vs_i)^-1] (fractions normalised to the given ones)."""
    vk = np.asarray(vp, float) / 1000.0
    tot = sum(np.asarray(f, float) for f in fractions.values())
    arith = 0.0
    harm = 0.0
    for lith, f in fractions.items():
        a2, a1, a0 = GC[lith]
        x = np.asarray(f, float) / np.where(tot > 0, tot, np.nan)
        vsi = np.maximum(a2 * vk ** 2 + a1 * vk + a0, 0.05)
        arith = arith + x * vsi
        harm = harm + x / vsi
    return 500.0 * (arith + 1.0 / harm)


def vs_with_fluid(vp, rho, phi, sw, fractions, k_min, k_w, rho_w, k_h, rho_h, iters=6, brie_e=None, phi_min=0.03):
    """Greenberg-Castagna for hydrocarbon-bearing rock (iterative): Vs guess -> Gassmann to brine -> GC Vs (brine)
    -> back to in-situ (mu unchanged, density changes)."""
    k_fl, rho_fl = fluid_mix(sw, k_w, k_h, rho_w, rho_h, brie_e)
    vs = vs_greenberg_castagna(vp, fractions)
    for _ in range(iters):
        vpb, vsb, rhob_, _ = substitute(vp, vs, rho, phi, k_min, k_fl, rho_fl, k_w, rho_w, phi_min)
        vs_b = vs_greenberg_castagna(vpb, fractions)
        mu = rhob_ * vs_b ** 2
        vs = np.sqrt(mu / rho)
    return vs


# --------------------------------------------------------------------------------------------- elastic logs
def elastic(vp, vs, rho, e_range=(1.0, 8.0), pr_range=(0.15, 0.40)):
    """AI, SI (m/s.g/cc), Vp/Vs, lambda-rho, mu-rho (GPa.g/cc), Poisson's ratio, Young's modulus (GPa) and the Rickman
    brittleness index (0-100; E in Mpsi scaled between e_range, PR between pr_range)."""
    vp, vs, rho = (np.asarray(x, float) for x in (vp, vs, rho))
    ai, si = rho * vp, rho * vs
    with np.errstate(invalid="ignore", divide="ignore"):
        vpvs = vp / vs
        pr = (vp ** 2 - 2 * vs ** 2) / (2 * (vp ** 2 - vs ** 2))
        ym = rho * vs ** 2 * (3 * vp ** 2 - 4 * vs ** 2) / (vp ** 2 - vs ** 2) * 1e-6
    lr = (ai ** 2 - 2 * si ** 2) * 1e-6
    mr = si ** 2 * 1e-6
    e_mpsi = ym * 0.1450377
    bi = 50.0 * (np.clip((e_mpsi - e_range[0]) / (e_range[1] - e_range[0]), 0, 1)
                 + np.clip((pr_range[1] - pr) / (pr_range[1] - pr_range[0]), 0, 1))
    return {"AI": ai, "SI": si, "VPVS": vpvs, "LR": lr, "MR": mr, "PR": pr, "YM": ym, "BI": bi}


# --------------------------------------------------------------------------------------------- well workflow
SHEAR_MNEMONICS = ["DTSM", "DTS", "DTSH", "DT2", "DTST", "DTS_XMAC", "DTSX", "DTS_WS"]


def rock_physics_well(project, name, pset=None, det=None, scenarios=("brine", "oil", "gas", "fizz"), save=True):
    """Elastic logs + fluid-substitution scenarios on a well, using the 20b interpretation (VSH, PHIE / PHIT, SW, TEMP).
    Shear: measured DTSM-type curve when present, else Greenberg-Castagna (iterated with Gassmann in hydrocarbons).
    Returns {"well", "depth", "curves" {name: array}, "fluids" {zone: {...}}, "vs_source"}."""
    from .qc import _curve_on_grid
    from .tools import open_sources
    w = project.load_well(name)
    pset = I.load_params(project) if pset is None else pset
    det = det or I.interpret_well(project, name, pset, save=False)
    z = w.depth
    nn = len(z)
    nan = np.full(nn, np.nan)
    dt = np.asarray(w.logs.get("DT", nan), float)
    rho = np.asarray(w.logs.get("RHOB", nan), float)
    with np.errstate(invalid="ignore", divide="ignore"):
        vp = np.where(dt > 20, 304800.0 / dt, np.nan)
    srcs, _ = open_sources(w)
    dts, dts_c = _curve_on_grid(srcs, SHEAR_MNEMONICS, z)
    c = det["curves"]
    vsh = np.clip(np.nan_to_num(c["VSH"], nan=0.5), 0, 1)
    phi = c["PHIE"]
    sw = np.clip(np.nan_to_num(c["SW"], nan=1.0), 0, 1)
    T = np.nan_to_num(c["TEMP"], nan=60.0)
    tvd = np.maximum(np.nan_to_num(det["tvdss"], nan=0.0), 1.0)
    out = {k: np.full(nn, np.nan) for k in ("VP", "VS", "RHO", "K_DRY", "MU", "K_FL", "K_MIN")}
    zinfo = {}
    segs = [(None, -np.inf, np.inf)] + I.zones_of(project, name, z)
    zone_of = np.full(nn, -1)
    for k, (zone, top, base) in enumerate(segs):
        zone_of[(z >= top) & (z < base)] = k
    for k, (zone, top, base) in enumerate(segs):
        idx = np.where(zone_of == k)[0]
        if not idx.size:
            continue
        P = dict(RP_DEFAULTS)
        P.update({kk: v for kk, v in I.resolve(pset, name, zone).items() if kk in RP_DEFAULTS or kk.startswith("rp_")})
        pr_ = P["pressure_gradient"] * tvd[idx] / 1000.0
        Tz = T[idx]
        rw_, vw_, kw = brine(Tz, pr_, P["salinity"])
        if P["rp_hc"] == "oil":
            rh, _, kh = oil(Tz, pr_, P["api"], P["gor"], P["gas_gravity"])
        else:
            rh, kh = gas(Tz, pr_, P["gas_gravity"])
        clean = {"sandstone": "quartz", "limestone": "calcite", "dolomite": "dolomite"}[P["rp_lith"]]
        km_c, mu_c, _ = MINERAL_MODULI[clean]
        km_s, mu_s, _ = MINERAL_MODULI["clay"]
        fr = np.column_stack([1 - vsh[idx], vsh[idx]])
        kmin = vrh(fr, [km_c, km_s])
        kfl, rfl = fluid_mix(sw[idx], kw, kh, rw_, rh, P["brie_e"])
        vpz, rhoz = vp[idx], rho[idx]
        fractions = {P["rp_lith"]: 1 - vsh[idx], "shale": vsh[idx]}
        if dts is not None and np.isfinite(dts[idx]).any():
            with np.errstate(invalid="ignore", divide="ignore"):
                vsz = np.where(dts[idx] > 30, 304800.0 / dts[idx], np.nan)
            src = f"measured {dts_c.mnemonic}"
            miss = ~np.isfinite(vsz) & np.isfinite(vpz)
            if miss.any():
                vsz[miss] = vs_with_fluid(vpz[miss], rhoz[miss], phi[idx][miss], sw[idx][miss],
                                          {kk: v[miss] for kk, v in fractions.items()}, kmin[miss], kw[miss], rw_[miss],
                                          kh[miss], rh[miss], brie_e=P["brie_e"], phi_min=P["phi_min_gassmann"])
                src += " + Greenberg-Castagna where missing"
        else:
            vsz = vs_with_fluid(vpz, rhoz, phi[idx], sw[idx], fractions, kmin, kw, rw_, kh, rh, brie_e=P["brie_e"],
                                phi_min=P["phi_min_gassmann"])
            src = "Greenberg-Castagna (Gassmann-iterated in hydrocarbons)"
        ksat, mu = moduli(vpz, vsz, rhoz)
        with np.errstate(invalid="ignore", divide="ignore"):
            kdry = gassmann_dry(ksat, kmin, kfl, np.where(phi[idx] >= P["phi_min_gassmann"], phi[idx], np.nan))
        out["VP"][idx], out["VS"][idx], out["RHO"][idx] = vpz, vsz, rhoz
        out["K_DRY"][idx], out["MU"][idx], out["K_FL"][idx], out["K_MIN"][idx] = kdry, mu, kfl, kmin
        # fluid-substitution scenarios
        targets = {"brine": 1.0, "oil": P["rp_sw_hc"], "gas": P["rp_sw_hc"], "fizz": 1.0 - P["rp_fizz_sg"]}
        ro, _, ko = oil(Tz, pr_, P["api"], P["gor"], P["gas_gravity"])
        rg, kg = gas(Tz, pr_, P["gas_gravity"])
        for sc in scenarios:
            hk, hr = (ko, ro) if sc == "oil" else (kg, rg)
            k2, r2 = fluid_mix(targets[sc], kw, hk, rw_, hr, P["brie_e"] if sc != "fizz" else None)
            vp2, vs2, rho2, _ = substitute(vpz, vsz, rhoz, phi[idx], kmin, kfl, rfl, k2, r2, P["phi_min_gassmann"])
            # only reservoir-quality rock is substituted (net reservoir flag of 20b)
            res_ = c["NET_RES"][idx] > 0.5
            for key, val, orig in (("VP", vp2, vpz), ("VS", vs2, vsz), ("RHO", rho2, rhoz)):
                arr = out.setdefault(f"{key}_{sc.upper()}", np.full(nn, np.nan))
                arr[idx] = np.where(res_, val, orig)
        zinfo[zone] = {"brine_rho": float(np.nanmedian(rw_)), "brine_k": float(np.nanmedian(kw)),
                       "hc": P["rp_hc"], "hc_rho": float(np.nanmedian(rh)), "hc_k": float(np.nanmedian(kh)),
                       "gas_rho": float(np.nanmedian(rg)), "gas_k": float(np.nanmedian(kg)),
                       "oil_rho": float(np.nanmedian(ro)), "oil_k": float(np.nanmedian(ko)),
                       "T": float(np.nanmedian(Tz)), "P": float(np.nanmedian(pr_)), "vs_source": src}
    P0 = dict(RP_DEFAULTS)
    P0.update({kk: v for kk, v in I.resolve(pset, name, None).items() if kk in RP_DEFAULTS})
    el = elastic(out["VP"], out["VS"], out["RHO"], (P0["e_min"], P0["e_max"]), (P0["pr_min"], P0["pr_max"]))
    out.update(el)
    for sc in scenarios:
        e2 = elastic(out[f"VP_{sc.upper()}"], out[f"VS_{sc.upper()}"], out[f"RHO_{sc.upper()}"])
        for key in ("AI", "SI", "VPVS", "LR", "MR", "PR"):
            out[f"{key}_{sc.upper()}"] = e2[key]
    res = {"well": name, "depth": z, "curves": out, "fluids": zinfo,
           "vs_source": sorted({v["vs_source"] for v in zinfo.values()})}
    if save and petro_dir(project):
        save_curves(project, name, z, {"RP_" + k: np.asarray(v, np.float32) for k, v in out.items()}, replace_prefix="RP_")
    return res


def scenario_table(res, zone_rows, net=None):
    """Mean AI / Vp/Vs / density per zone for each fluid scenario - the AVO-modelling starting point.
    net: optional flag curve (e.g. the 20b NET_RES) to average over net reservoir only."""
    z = res["depth"]
    c = res["curves"]
    rows = []
    for zr in zone_rows:
        m = (z >= zr["Top_MD"]) & (z < zr["Base_MD"])
        if net is not None:
            m &= np.asarray(net) > 0.5
        r = {"Zone": zr["Zone"]}
        for sc in ("", "_BRINE", "_OIL", "_GAS", "_FIZZ"):
            for key in ("AI", "VPVS", "RHO"):
                k = f"{key}{sc}"
                if k in c:
                    v = c[k][m]
                    r[(k if sc else key + "_INSITU")] = round(float(np.nanmean(v)), 3) if np.isfinite(v).any() else None
        rows.append(r)
    return rows
