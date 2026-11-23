"""Maturity (PLAN 18c): EASY%Ro (Sweeney & Burnham 1990), Lopatin TTI, kerogen transformation ratio with
Gaussian activation-energy kinetics, oil / gas window timing.

All functions take ages (nt,) Ma from old to young and temperatures T (nt,) or (nt, m) deg C, linear in time
between samples; integration uses sub-steps of at most 1 deg C (exact enough for the Arrhenius integral).
"""
import numpy as np

from .thermal import SEC_MY

R_KCAL = 1.98720e-3  # kcal/mol/K
R_KJ = 8.314462e-3  # kJ/mol/K

# Sweeney & Burnham (1990) EASY%Ro: A = 1e13 /s, E = 34..72 kcal/mol, stoichiometric factors (sum 0.85)
EASY_E = np.arange(34.0, 73.0, 2.0)
EASY_F = np.array([0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.04, 0.04, 0.07, 0.06, 0.06, 0.06, 0.05, 0.05, 0.04,
                   0.03, 0.02, 0.02, 0.01])
EASY_A = 1e13

# Bulk kerogen -> petroleum kinetics, Gaussian activation energies (Pepper & Corvi 1995 organofacies; approximate
# published values - A (1/s), mean E and sigma (kJ/mol)).
KINETICS = {
    "Type I (lacustrine, organofacies C)": {"A": 2.44e14, "E": 221.4, "sigma": 3.9},
    "Type II (marine, organofacies B)": {"A": 8.14e13, "E": 215.2, "sigma": 8.3},
    "Type II-S (marine carbonate, organofacies A)": {"A": 2.13e14, "E": 206.4, "sigma": 8.2},
    "Type III (terrigenous / coaly, organofacies D/E)": {"A": 4.97e14, "E": 228.2, "sigma": 7.9},
    "Type III gas-prone (organofacies F)": {"A": 1.23e17, "E": 259.1, "sigma": 6.6},
}
WINDOWS = {"oil": (0.55, 1.3), "gas": (1.3, 2.0)}


def _arrhenius_integral(ages, T, A, E_over_R, max_dT=1.0):
    """Cumulative integral of A exp(-E/RT) dt for each energy at every age: (nt, [m,] nE). T linear in time
    between samples, sub-steps of <= max_dT; NaN temperatures (not yet deposited) add nothing."""
    ages = np.asarray(ages, float)
    T = np.asarray(T, float)
    one = T.ndim == 1
    T2 = T[:, None] if one else T
    nt, m = T2.shape
    E_over_R = np.asarray(E_over_R, float)
    A = np.broadcast_to(np.asarray(A, float), E_over_R.shape)
    I = np.zeros((nt, m, len(E_over_R)))
    cur = np.zeros((m, len(E_over_R)))
    for i in range(1, nt):
        dt = (ages[i - 1] - ages[i]) * SEC_MY
        t0, t1 = T2[i - 1], T2[i]
        ok = np.isfinite(t0) & np.isfinite(t1)
        if dt > 0 and ok.any():
            dT = float(np.max(np.abs(np.where(ok, t1 - t0, 0.0))))
            n = max(1, int(np.ceil(dT / max_dT)))
            for s in range(n):
                w = (s + 0.5) / n
                K = np.where(ok, t0 * (1 - w) + t1 * w, 0.0) + 273.15
                rate = A[None, :] * np.exp(-E_over_R[None, :] / K[:, None])
                cur = cur + np.where(ok[:, None], rate * (dt / n), 0.0)
        I[i] = cur
    return I[:, 0] if one else I


def easy_ro(ages, T):
    """EASY%Ro at every age (nt,) or (nt, m). Samples before deposition (NaN T) count as unheated."""
    I = _arrhenius_integral(ages, T, EASY_A, EASY_E / R_KCAL)
    F = np.sum(EASY_F * (1 - np.exp(-I)), axis=-1)
    return np.exp(-1.6 + 3.7 * F)


def tti(ages, T):
    """Lopatin time-temperature index: sum of dt (My) x 2^n, n = floor((T - 100) / 10) (100-110 degC -> n = 0)."""
    ages = np.asarray(ages, float)
    T = np.asarray(T, float)
    out = np.zeros(T.shape)
    cur = np.zeros(T.shape[1:])
    for i in range(1, len(ages)):
        dt = ages[i - 1] - ages[i]
        Tm = 0.5 * (T[i - 1] + T[i])
        n = np.floor((Tm - 100.0) / 10.0)
        cur = cur + np.where(np.isfinite(Tm), dt * 2.0 ** n, 0.0)
        out[i] = cur
    return out


def kinetics_energies(kind, n=21):
    k = KINETICS[kind] if isinstance(kind, str) else kind
    z = np.linspace(-3, 3, n)
    w = np.exp(-0.5 * z * z)
    w /= w.sum()
    return k["A"], (k["E"] + z * k["sigma"]) / R_KJ, w


def transformation_ratio(ages, T, kind="Type II (marine, organofacies B)"):
    """Fraction of reactive kerogen converted (0-1) at every age."""
    A, EoR, w = kinetics_energies(kind)
    I = _arrhenius_integral(ages, T, A, EoR)
    return np.sum(w * (1 - np.exp(-I)), axis=-1)


def crossing_age(ages, values, level):
    """Age (Ma) when an increasing series first reaches `level` (linear between samples); None if never.
    values (nt,) or (nt, m) -> float or array (NaN = never)."""
    ages = np.asarray(ages, float)
    v = np.asarray(values, float)
    one = v.ndim == 1
    v2 = v[:, None] if one else v
    out = np.full(v2.shape[1], np.nan)
    above = np.nan_to_num(v2, nan=-np.inf) >= level
    first = np.argmax(above, axis=0)
    has = above.any(axis=0)
    for j in np.nonzero(has)[0]:
        i = first[j]
        if i == 0:
            out[j] = ages[0]
            continue
        a, b = v2[i - 1, j], v2[i, j]
        w = (level - a) / (b - a) if np.isfinite(a) and b != a else 1.0
        out[j] = ages[i - 1] + (ages[i] - ages[i - 1]) * np.clip(w, 0, 1)
    if one:
        return None if np.isnan(out[0]) else float(out[0])
    return out


def window_timing(ages, ro):
    """{"oil_start", "oil_end", "gas_start", "gas_end"} ages (Ma) of the %Ro window crossings."""
    return {"oil_start": crossing_age(ages, ro, WINDOWS["oil"][0]), "oil_end": crossing_age(ages, ro, WINDOWS["oil"][1]),
            "gas_start": crossing_age(ages, ro, WINDOWS["gas"][0]), "gas_end": crossing_age(ages, ro, WINDOWS["gas"][1])}


def window_name(ro):
    ro = float(ro)
    if not np.isfinite(ro):
        return ""
    if ro < WINDOWS["oil"][0]:
        return "immature"
    if ro < WINDOWS["oil"][1]:
        return "oil window"
    if ro < WINDOWS["gas"][1]:
        return "wet gas"
    return "dry gas"
