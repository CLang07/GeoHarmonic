"""Curve QC per well (PLAN 20a): gaps, flat lines, spikes, bad hole, depth shift between runs, splices, bed
boundaries + thin-bed flags against each curve's effective vertical resolution, and the tool report.

    rep = well_qc(project, "Tui-1")        # dict (also saved to <project>_petro/qc/Tui-1.json + QC curves npz)
    print(format_report(rep))
    qc_project(project)                    # every well + qc_summary.csv
"""
import csv
import os
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..curves import STEP, family_of, normalise_units, remove_flatlines, _resample
from . import tools as T
from .store import petro_dir, save_curves, save_json

BIT_SIZES = np.array([3.875, 4.75, 5.875, 6.0, 6.125, 6.5, 7.875, 8.375, 8.5, 8.75, 9.5, 9.875, 10.625, 12.25,
                      14.75, 16.0, 17.5, 22.0, 24.0, 26.0, 36.0])
DRHO_MNEMONICS = ("DRHO", "HDRA", "ZCOR", "DRHB", "DCOR", "QRHO")
BS_MNEMONICS = ("BS", "BIT", "BITSIZE", "BSZ")
LOG_FAMILIES = ("RDEEP", "RSHAL")


# ------------------------------------------------------------------------------------------------ primitives
def intervals(mask, depth):
    """[(top, base)] of runs of True (base = depth of the last True sample)."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return [(float(depth[a]), float(depth[b])) for a, b in zip(starts, ends)]


def _run_lengths(mask, depth, step):
    return [(t, b, b - t + step) for t, b in intervals(mask, depth)]


def gaps(depth, values, min_gap_m=0.5):
    """NaN stretches inside the logged range (first..last valid sample) at least min_gap_m long."""
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    if not ok.any():
        return []
    i0, i1 = np.argmax(ok), len(ok) - 1 - np.argmax(ok[::-1])
    inside = np.zeros(len(v), bool)
    inside[i0:i1 + 1] = True
    step = T.sample_interval(depth)
    return [(t, b) for t, b, L in _run_lengths(inside & ~ok, depth, step) if L >= min_gap_m]


def flatline_mask(depth, values, min_len_m=3.0):
    v = np.asarray(values, float)
    return np.isfinite(v) & np.isnan(remove_flatlines(depth, v, min_len_m))


def rolling_median(v, half):
    """Centred running nanmedian and nanMAD (window 2*half+1), vectorised."""
    v = np.asarray(v, float)
    pad = np.concatenate([np.full(half, np.nan), v, np.full(half, np.nan)])
    W = sliding_window_view(pad, 2 * half + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(W, axis=1)
        mad = np.nanmedian(np.abs(W - med[:, None]), axis=1)
    return med, mad


def spike_mask(values, half=5, k=6.0, log=False):
    """Samples deviating from the running median by more than k robust sigmas.  The sigma has a floor of half
    the curve's global robust noise so blocky (piecewise constant) logs do not flag every small wiggle."""
    v = np.asarray(values, float)
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log10(np.where(v > 0, v, np.nan))
    ok = np.isfinite(v)
    if ok.sum() < 2 * half + 1:
        return np.zeros(len(v), bool)
    med, mad = rolling_median(v, half)
    dev = np.abs(v - med)
    ad = np.abs(np.diff(v[ok]))
    noise = 1.4826 * float(np.median(ad)) / np.sqrt(2)  # global robust sample-to-sample noise
    if noise <= 0:  # quantised curves (calipers): most differences are exactly 0
        nz = ad[ad > 0]
        noise = float(np.percentile(nz, 25)) if nz.size else 1e-12
    floor = 0.5 * noise + 1e-12
    sig = np.maximum(1.4826 * np.nan_to_num(mad), floor)
    out = ok & (dev > k * sig)
    # isolated excursions 1-2 samples wide (a spike next to a bed boundary inflates the local MAD; a step differs
    # from one side only, a spike from both sides, with the same sign).  Resampling to the composite grid can
    # spread a one-sample spike over two samples.
    n = len(v)
    d1 = np.full(n, np.nan)
    d1[1:] = v[1:] - v[:-1]
    loc, _ = rolling_median(np.abs(d1), 2 * half)  # local roughness (rugose hole, noisy intervals)
    thr = 1.5 * k * np.maximum(1.4826 * np.nan_to_num(loc) / np.sqrt(2), noise)
    iso = np.zeros(n, bool)
    with np.errstate(invalid="ignore"):
        for wdt in (1, 2):
            i = np.arange(1, n - wdt)
            blk = np.stack([v[i + j] for j in range(wdt)])
            left = blk - v[i - 1]
            right = blk - v[i + wdt]
            lo = np.minimum(np.abs(left).min(0), np.abs(right).min(0))
            same = (np.sign(left) == np.sign(left[0])).all(0) & (np.sign(right) == np.sign(left[0])).all(0)
            hit = same & (lo > thr[i])
            for j in range(wdt):
                iso[i[hit] + j] = True
    return out | (ok & iso)


def estimate_bit_size(depth, cali, window_m=60.0):
    """Bit size per sample: running 40th percentile of the caliper (in gauge hole most samples are near bit
    size) snapped to the nearest standard bit size within 0.5 in."""
    c = np.asarray(cali, float)
    out = np.full(len(c), np.nan)
    ok = np.isfinite(c)
    if ok.sum() < 20:
        return out
    step = T.sample_interval(depth)
    n = max(5, int(window_m / step))
    idx = np.where(ok)[0]
    blocks = np.arange(idx[0], idx[-1] + 1, n)
    for b in blocks:
        seg = c[b:b + n]
        seg = seg[np.isfinite(seg)]
        if seg.size < 5:
            continue
        p = float(np.percentile(seg, 40))
        j = int(np.argmin(np.abs(BIT_SIZES - p)))
        out[b:b + n] = BIT_SIZES[j] if abs(BIT_SIZES[j] - p) <= 0.5 else p
    out[~ok] = np.nan
    return out


def bad_hole(depth, cali, bit_size=None, washout_in=1.0, drho=None, drho_lim=0.15):
    """Bad-hole mask: caliper > bit size + washout_in, or |density correction| > drho_lim.
    Returns (mask, bit size array used, source text)."""
    n = len(depth)
    mask = np.zeros(n, bool)
    src = []
    bs = None
    if cali is not None:
        c = np.asarray(cali, float)
        if bit_size is not None and np.isfinite(bit_size).any():
            bs = np.asarray(bit_size, float)
            src.append("BS curve")
        else:
            bs = estimate_bit_size(depth, c)
            src.append("bit size estimated from caliper")
        with np.errstate(invalid="ignore"):
            mask |= np.isfinite(c) & np.isfinite(bs) & (c - bs > washout_in)
    if drho is not None:
        d = np.asarray(drho, float)
        with np.errstate(invalid="ignore"):
            mask |= np.isfinite(d) & (np.abs(d) > drho_lim)
        src.append(f"|DRHO| > {drho_lim}")
    return mask, bs, ", ".join(src)


def _detrend(v, half):
    med, _ = rolling_median(v, half) if half > 0 else (np.zeros_like(v), None)
    return v - med


def depth_shift(ref_z, ref_v, z, v, max_shift_m=5.0, step=STEP, detrend_m=15.0, log=False):
    """Bulk depth shift that best matches curve (z, v) to the reference (ref_z, ref_v).

    Returns (shift_m, corr, overlap_m): add shift_m to z to align (v(z) ~ ref(z + shift)).  The curves are
    z-scored after removing a running median (detrend_m) so tool calibration / trend differences do not matter."""
    ref_z, ref_v, z, v = (np.asarray(a, float) for a in (ref_z, ref_v, z, v))
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            ref_v, v = np.log10(np.where(ref_v > 0, ref_v, np.nan)), np.log10(np.where(v > 0, v, np.nan))
    okr, okc = np.isfinite(ref_v) & np.isfinite(ref_z), np.isfinite(v) & np.isfinite(z)
    if okr.sum() < 20 or okc.sum() < 20:
        return 0.0, float("nan"), 0.0
    lo = max(ref_z[okr].min(), z[okc].min()) - max_shift_m
    hi = min(ref_z[okr].max(), z[okc].max()) + max_shift_m
    if hi - lo < 4 * max_shift_m:
        return 0.0, float("nan"), max(0.0, hi - lo - 2 * max_shift_m)
    g = np.arange(lo, hi, step)
    a = _resample(ref_z, ref_v, g)
    b = _resample(z, v, g)
    half = int(detrend_m / step / 2)
    if half > 2:
        a, b = _detrend(a, half), _detrend(b, half)
    K = int(round(max_shift_m / step))
    n = len(g)
    lags = np.arange(-K, K + 1)
    cc = np.full(len(lags), np.nan)
    cnt = np.zeros(len(lags))
    for j, k in enumerate(lags):  # b shifted down by k samples: b[i - k] vs a[i]
        if k >= 0:
            x, y = a[k:], b[:n - k]
        else:
            x, y = a[:n + k], b[-k:]
        m = np.isfinite(x) & np.isfinite(y)
        cnt[j] = m.sum()
        if cnt[j] < 30:
            continue
        xx, yy = x[m] - x[m].mean(), y[m] - y[m].mean()
        den = np.sqrt((xx * xx).sum() * (yy * yy).sum())
        cc[j] = (xx * yy).sum() / den if den > 0 else np.nan
    if not np.isfinite(cc).any():
        return 0.0, float("nan"), 0.0
    j = int(np.nanargmax(cc))
    frac = 0.0
    if 0 < j < len(lags) - 1 and np.isfinite(cc[j - 1]) and np.isfinite(cc[j + 1]):
        den = cc[j - 1] - 2 * cc[j] + cc[j + 1]
        if den < 0:
            frac = float(np.clip(0.5 * (cc[j - 1] - cc[j + 1]) / den, -0.5, 0.5))
    return float((lags[j] + frac) * step), float(cc[j]), float(cnt[j] * step)


def gsmooth(v, sigma):
    """Gaussian smoothing that ignores NaN (normalised convolution); sigma in samples."""
    v = np.asarray(v, float)
    if sigma <= 0:
        return v.copy()
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    ok = np.isfinite(v)
    num = np.convolve(np.where(ok, v, 0.0), k, "same")
    den = np.convolve(ok.astype(float), k, "same")
    out = np.where(den > 1e-6, num / np.where(den > 1e-6, den, 1), np.nan)
    out[~ok] = np.nan
    return out


def bed_boundaries(depth, values, smooth_m=0.15, k=4.0, log=False, min_sep_m=None):
    """Bed boundaries = local maxima of |dv/dz| of the lightly smoothed curve above k robust sigmas of the
    derivative.  Returns boundary depths (sub-sample, at the derivative peak)."""
    z = np.asarray(depth, float)
    v = np.asarray(values, float)
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log10(np.where(v > 0, v, np.nan))
    step = T.sample_interval(z)
    if not np.isfinite(step) or np.isfinite(v).sum() < 10:
        return np.array([])
    s = gsmooth(v, smooth_m / step)
    g = np.abs(np.gradient(s))
    ok = np.isfinite(g)
    if ok.sum() < 5:
        return np.array([])
    gm = np.nanmedian(g)
    sig = 1.4826 * np.nanmedian(np.abs(g[ok] - gm)) + 1e-12
    thr = gm + k * sig
    gg = np.where(ok, g, 0.0)
    peak = (gg[1:-1] > thr) & (gg[1:-1] >= gg[:-2]) & (gg[1:-1] > gg[2:])
    idx = np.where(peak)[0] + 1
    # parabolic refinement
    a, b, c = gg[idx - 1], gg[idx], gg[idx + 1]
    den = a - 2 * b + c
    frac = np.where(den < 0, 0.5 * (a - c) / np.where(den < 0, den, -1), 0.0)
    zb = np.interp(idx + np.clip(frac, -0.5, 0.5), np.arange(len(z)), z)
    if min_sep_m and len(zb) > 1:
        keep = [0]
        for i in range(1, len(zb)):
            if zb[i] - zb[keep[-1]] >= min_sep_m:
                keep.append(i)
            elif gg[idx[i]] > gg[idx[keep[-1]]]:
                keep[-1] = i
        zb = zb[keep]
    return zb


def bed_thickness_curve(depth, boundaries, valid=None):
    """Per-sample thickness (m) of the bed the sample is in (beds between consecutive boundaries; the open
    ends above the first / below the last boundary get NaN)."""
    z = np.asarray(depth, float)
    out = np.full(len(z), np.nan)
    b = np.asarray(boundaries, float)
    if len(b) < 2:
        return out
    j = np.searchsorted(b, z) - 1
    inside = (j >= 0) & (j < len(b) - 1)
    out[inside] = b[j[inside] + 1] - b[j[inside]]
    if valid is not None:
        out[~np.asarray(valid, bool)] = np.nan
    return out


# ------------------------------------------------------------------------------------------------ well report
def _curve_on_grid(sources, mnemonics, grid, aliases=None):
    """First raw curve (by mnemonic list order, then coverage) resampled to grid, or None."""
    best = None
    for src in sources:
        for c in src.curves:
            if c.mnemonic.upper() in mnemonics and c.n_valid > 20:
                t, b = c.valid_range()
                key = (mnemonics.index(c.mnemonic.upper()), -(b - t))
                if best is None or key < best[0]:
                    best = (key, c)
    if best is None:
        return None, None
    c = best[1]
    return _resample(c.depth, np.asarray(c.values, float), grid), c


def splice_report(sources, family, aliases, max_shift_m=3.0):
    """Overlaps between different runs / files of the same curve family: length, depth shift, level difference."""
    cands = []
    for src in sources:
        for c in src.curves:
            fam, _ = family_of(c.mnemonic, aliases)
            if fam == family and c.n_valid >= 50:
                vals, _ = normalise_units(fam, c.unit, np.asarray(c.values, float))
                cands.append((c, vals))
    out = []
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            (ci, vi), (cj, vj) = cands[i], cands[j]
            if ci.file == cj.file and ci.mnemonic == cj.mnemonic:
                continue  # same file + mnemonic = repeat pass in one file: handled by the composite ranking
            ti, bi = ci.valid_range()
            tj, bj = cj.valid_range()
            ov = min(bi, bj) - max(ti, tj)
            if ov < 15.0:
                continue
            g = np.arange(max(ti, tj), min(bi, bj), STEP)
            a, b = _resample(ci.depth, vi, g), _resample(cj.depth, vj, g)
            log = family in LOG_FAMILIES
            if log:
                with np.errstate(divide="ignore", invalid="ignore"):
                    a, b = np.log10(np.where(a > 0, a, np.nan)), np.log10(np.where(b > 0, b, np.nan))
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 50:
                continue
            diff = float(np.median(b[m] - a[m]))
            scale = float(np.median(np.abs(a[m]))) or 1.0
            sh, cc, _ = depth_shift(ci.depth, vi, cj.depth, vj, max_shift_m, log=log, detrend_m=10.0)
            out.append({"_ka": (ci.mnemonic.upper(), os.path.basename(ci.file).lower()),
                        "_kb": (cj.mnemonic.upper(), os.path.basename(cj.file).lower()), "family": family, "a": f"{ci.mnemonic} [{os.path.basename(ci.file)}]",
                        "b": f"{cj.mnemonic} [{os.path.basename(cj.file)}]", "overlap_m": round(float(m.sum() * STEP), 1),
                        "level_diff": round(diff, 4), "level_diff_rel": round(diff / scale, 3) if not log else None,
                        "level_diff_log10": round(diff, 3) if log else None,
                        "shift_m": round(sh, 2), "corr": None if not np.isfinite(cc) else round(cc, 2)})
    return out


def finest_family(info_by_family, logs, order=("RSHAL", "PE", "RHOB", "GR", "NPHI", "DT", "RDEEP")):
    """Curve family with the best effective resolution and enough data - used for bed boundaries."""
    best = None
    for fam in order:
        if fam not in logs or np.isfinite(logs[fam]).sum() < 50:
            continue
        res = info_by_family.get(fam, {}).get("eff_res_m", T.KIND_RES.get(T.FAMILY_KIND.get(fam, ""), 0.6))
        if best is None or res < best[1] - 1e-9:
            best = (fam, res)
    return best


def well_qc(project, name, ref_family="GR", min_gap_m=0.5, washout_in=1.0, drho_lim=0.15, shift_max_m=5.0,
            save=True, table=None):
    """QC + tool report for one well.  Saves QC curves (QC_BADHOLE, QC_SPIKE_<fam>, BED_THICK, BS_USED) and the
    JSON report into <project>_petro/ when save=True and the project has a path."""
    w = project.load_well(name)
    rep = {"well": name, "files": [], "families": {}, "curves": [], "shifts": [], "splices": [], "beds": {},
           "warnings": [], "errors": []}
    if w.depth is None or not len(w.depth):
        rep["warnings"].append("no logs")
        return rep
    sources, errors = T.open_sources(w)
    rep["errors"] = errors
    ident = T.identify_well(project, name, table=table, sources=sources)
    rep["files"] = ident["files"]
    used = T.used_curves(w, ident)
    z = w.depth
    step = T.sample_interval(z)
    # ---- per raw curve: tool + gaps + flat lines
    for c_info, (src, c) in zip(ident["curves"], [(s, c) for s in sources for c in s.curves if c.n_valid >= 2]):
        vals = np.asarray(c.values, float)
        fl = flatline_mask(c.depth, vals)
        c_info["flat_m"] = round(float(fl.sum() * (c_info["sample_m"] if np.isfinite(c_info["sample_m"]) else STEP)), 1)
        c_info["gaps"] = len(gaps(c.depth, vals, min_gap_m))
        rep["curves"].append(c_info)
    # ---- per composited family
    qc_curves = {}
    cali = w.logs.get("CALI")
    bs, _ = _curve_on_grid(sources, list(BS_MNEMONICS), z)
    drho, drho_c = _curve_on_grid(sources, list(DRHO_MNEMONICS), z)
    bh, bs_used, bh_src = bad_hole(z, cali, bs, washout_in, drho, drho_lim)
    if cali is not None or drho is not None:
        qc_curves["QC_BADHOLE"] = bh.astype(np.float32)
        if bs_used is not None:
            qc_curves["BS_USED"] = np.asarray(bs_used, np.float32)
    info_by_family = {}
    gap_groups = []
    for fam, v in w.logs.items():
        u = used.get(fam, [])
        first = u[0] if u else {}
        info_by_family[fam] = first
        sp = spike_mask(v, log=fam in LOG_FAMILIES)
        qc_curves[f"QC_SPIKE_{fam}"] = sp.astype(np.float32)
        ok = np.isfinite(v)
        g = gaps(z, v, min_gap_m)
        logged = (z[ok].max() - z[ok].min()) if ok.any() else 0.0
        tools_used = sorted({c["tool"] for c in u})
        convs = sorted({c["conveyance"] for c in u if c["conveyance"] != "unknown"})
        res = max([c["eff_res_m"] for c in u], default=T.KIND_RES.get(T.FAMILY_KIND.get(fam, ""), 0.6))
        rep["families"][fam] = {
            "top": float(z[ok].min()) if ok.any() else None, "base": float(z[ok].max()) if ok.any() else None,
            "coverage_m": round(float(ok.sum() * step), 1), "gaps": len(g), "gap_m": round(sum(b - t + step for t, b in g), 1),
            "spikes": int(sp.sum()), "bad_hole_m": round(float((bh & ok).sum() * step), 1),
            "bad_hole_pct": round(100.0 * (bh & ok).sum() / max(1, ok.sum()), 1),
            "tools": tools_used, "vendors": sorted({c["vendor"] for c in u}), "kind": first.get("kind", ""),
            "conveyance": "/".join(convs) or "unknown", "eff_res_m": round(float(res), 3),
            "sources": len(u), "logged_m": round(float(logged), 1),
        }
        if len(convs) > 1:
            rep["warnings"].append(f"{fam}: composite mixes {' and '.join(convs)} curves - check the splice")
        if "resistivity (type unknown)" in {c["kind"] for c in u}:
            rep["warnings"].append(f"{fam}: laterolog vs induction not identified - add the mnemonic to tool_table.json")
        if g and sum(b - t for t, b in g) > 0.05 * max(logged, 1):
            t, b = max(g, key=lambda tb: tb[1] - tb[0])
            for grp in gap_groups:
                gt, gb = min(x[1] for x in grp), max(x[2] for x in grp)
                if min(b, gb) - max(t, gt) > 0.5 * min(b - t, gb - gt):
                    grp.append((fam, t, b))
                    break
            else:
                gap_groups.append([(fam, t, b)])
        if rep["families"][fam]["bad_hole_pct"] > 20 and fam in ("RHOB", "NPHI", "PE", "RSHAL"):
            rep["warnings"].append(f"{fam}: {rep['families'][fam]['bad_hole_pct']} % of samples in bad hole ({bh_src})")
    for lst in gap_groups:
        t, b = min(x[1] for x in lst), max(x[2] for x in lst)
        rep["warnings"].append(f"unlogged gap {t:.0f}-{b:.0f} m MD ({b - t:.0f} m) in " + ", ".join(x[0] for x in lst)
                               + " (separate runs / suites?)")
    under = {}
    for c in rep["curves"]:
        if c["undersampled"] and c["family"]:
            under.setdefault((c["file"], round(c["sample_m"], 3)), []).append(f"{c['mnemonic']} ({c['tool_res_m']} m)")
    for (f, smp), lst in under.items():
        rep["warnings"].append(f"{f}: sample interval {smp} m is coarser than half the tool resolution of "
                               + ", ".join(lst) + " - effective resolution = 2 x sample")
    # ---- depth shift of every other run's reference curve vs the longest one
    refs = []
    for src in sources:
        for c in src.curves:
            fam, _ = family_of(c.mnemonic, project.aliases)
            if fam == ref_family and c.n_valid >= 100:
                t, b = c.valid_range()
                refs.append((b - t, c))
    refs.sort(key=lambda x: -x[0])
    if len(refs) > 1:
        rc = refs[0][1]
        seen = {rc.file}
        for _, c in refs[1:]:
            if c.file in seen:
                continue
            seen.add(c.file)
            sh, cc, ov = depth_shift(rc.depth, rc.values, c.depth, c.values, shift_max_m, log=ref_family in LOG_FAMILIES)
            if ov < 30.0:
                continue  # runs do not overlap enough to compare
            row = {"file": os.path.basename(c.file), "curve": c.mnemonic, "reference": f"{rc.mnemonic} [{os.path.basename(rc.file)}]",
                   "shift_m": round(sh, 2), "corr": None if not np.isfinite(cc) else round(cc, 2), "overlap_m": round(ov, 1)}
            rep["shifts"].append(row)
            if np.isfinite(cc) and cc >= 0.5 and abs(sh) >= 0.3:
                rep["warnings"].append(f"depth shift: {row['file']} {c.mnemonic} is {sh:+.2f} m off {row['reference']} "
                                       f"(corr {cc:.2f}, {ov:.0f} m overlap)")
    # ---- splices (same mnemonic in different files = runs; plus consecutive curves the composite used)
    for fam in w.logs:
        used_keys = [(c["mnemonic"], c["file"].lower()) for c in used.get(fam, [])]
        consecutive = set(zip(used_keys[:-1], used_keys[1:]))
        best = {}
        for s in splice_report(sources, fam, project.aliases):  # one row per pair of curves (repeat columns)
            k = (s["_ka"], s["_kb"])
            if k not in best or s["overlap_m"] > best[k]["overlap_m"]:
                best[k] = s
        for s in best.values():
            ka, kb = s.pop("_ka"), s.pop("_kb")
            in_comp = (ka, kb) in consecutive or (kb, ka) in consecutive
            if ka[0] != kb[0] and not in_comp:
                continue
            s["in_composite"] = in_comp
            rep["splices"].append(s)
            big = (s["level_diff_rel"] is not None and abs(s["level_diff_rel"]) > 0.1) or \
                  (s["level_diff_log10"] is not None and abs(s["level_diff_log10"]) > 0.1)
            if big and in_comp:
                rep["warnings"].append(f"splice {fam}: {s['a']} vs {s['b']} differ by "
                                       f"{s['level_diff_rel'] if s['level_diff_rel'] is not None else s['level_diff_log10']}"
                                       f"{'' if s['level_diff_rel'] is not None else ' log10'} over {s['overlap_m']} m "
                                       "(the composite switches between them)")
            if in_comp and s["corr"] is not None and s["corr"] >= 0.5 and abs(s["shift_m"]) >= 0.3:
                rep["warnings"].append(f"splice {fam}: {s['b']} is {s['shift_m']:+.2f} m off {s['a']} (corr {s['corr']})")
    # ---- barite mud: PEF of clean low-density rock far above quartz (1.8) / even calcite-free sand
    pe, rho = w.logs.get("PE"), w.logs.get("RHOB")
    if pe is not None and rho is not None:
        from ..lithology import well_lithology
        vsh, _ = well_lithology(w)
        cs = np.isfinite(pe) & np.isfinite(rho) & (np.nan_to_num(vsh, nan=1) < 0.25) & (rho < 2.45)
        med = float(np.median(pe[cs])) if cs.sum() > 100 else 0.0
        if med > 4.0:
            rep["warnings"].append(f"PE: median {med:.1f} b/e in clean low-density rock - barite in the mud likely. "
                                   "Do not use PE / U in lithology or the mineral solver (mm_logs without 'U')")
            rep["barite_suspected"] = "likely"
        elif med > 2.9:
            rep["warnings"].append(f"PE: median {med:.1f} b/e in clean low-density rock (quartz sand ~1.8-2.7) - barite "
                                   "or heavy minerals? Check before using PE / U in the mineral solver")
            rep["barite_suspected"] = "possible"
    # ---- beds + thin beds
    ff = finest_family(info_by_family, w.logs)
    if ff:
        fam, res = ff
        bnd = bed_boundaries(z, w.logs[fam], smooth_m=max(step, 0.5 * res) if np.isfinite(step) else 0.15,
                             log=fam in LOG_FAMILIES, min_sep_m=max(step, 0.5 * res) if np.isfinite(step) else None)
        thick = bed_thickness_curve(z, bnd, np.isfinite(w.logs[fam]))
        qc_curves["BED_THICK"] = thick.astype(np.float32)
        beds = np.diff(bnd)
        thin = {}
        for f2, info in rep["families"].items():
            r2 = info["eff_res_m"]
            sel = beds < r2
            thin[f2] = {"eff_res_m": r2, "thin_beds": int(sel.sum()), "thin_m": round(float(beds[sel].sum()), 1),
                        "thin_pct_of_beds": round(100.0 * sel.sum() / max(1, len(beds)), 1)}
            qc_curves[f"QC_THIN_{f2}"] = (np.isfinite(thick) & (thick < r2)).astype(np.float32)
        rep["beds"] = {"from": fam, "resolution_m": round(float(res), 3), "boundaries": int(len(bnd)), "beds": int(len(beds)),
                       "median_thickness_m": round(float(np.median(beds)), 2) if len(beds) else None, "thin": thin}
        worst = [f2 for f2, t in thin.items() if t["thin_pct_of_beds"] > 50]
        if worst:
            rep["warnings"].append("thin beds: more than half of the beds (from " + fam + ") are below the resolution of "
                                   + ", ".join(sorted(worst)))
    rep["qc_curves"] = sorted(qc_curves)
    if save and petro_dir(project):
        save_curves(project, name, z, qc_curves, replace_prefix="QC_")
        rep["saved"] = save_json(project, "qc", name, rep)
    rep["_qc_arrays"] = qc_curves  # not in the JSON (saved as npz)
    return rep


SUMMARY_COLUMNS = ["Well", "Family", "Top", "Base", "Coverage_m", "Gaps", "Gap_m", "Spikes", "BadHole_pct", "Tools",
                   "Vendors", "Kind", "Conveyance", "EffRes_m", "ThinBeds_pct"]


def qc_project(project, wells=None, progress=None, save=True, **kw):
    """well_qc for every well (or `wells`); writes <project>_petro/qc_summary.csv.  Returns {well: report}."""
    out = {}
    names = list(wells or project.wells)
    for i, n in enumerate(names):
        try:
            out[n] = well_qc(project, n, save=save, **kw)
        except Exception as e:  # one bad well must not stop a batch
            out[n] = {"well": n, "errors": [f"{type(e).__name__}: {e}"], "families": {}, "warnings": [], "files": [],
                      "curves": [], "shifts": [], "splices": [], "beds": {}}
        if progress:
            progress(i + 1, len(names), n)
    if save and petro_dir(project):
        path = os.path.join(petro_dir(project, create=True), "qc_summary.csv")
        write_summary(out, path)
    return out


def write_summary(reports, path):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(SUMMARY_COLUMNS)
        for n, rep in reports.items():
            for fam, i in rep.get("families", {}).items():
                thin = rep.get("beds", {}).get("thin", {}).get(fam, {})
                wr.writerow([n, fam, _r(i["top"]), _r(i["base"]), i["coverage_m"], i["gaps"], i["gap_m"], i["spikes"],
                             i["bad_hole_pct"], " / ".join(i["tools"]), " / ".join(i["vendors"]), i["kind"], i["conveyance"],
                             i["eff_res_m"], thin.get("thin_pct_of_beds", "")])
    return path


def _r(v, k=1):
    return "" if v is None else round(v, k)


def format_report(rep):
    """Plain-text report for the console / a dialog."""
    L = [f"=== {rep['well']}"]
    for e in rep.get("errors", []):
        L.append(f"  ERROR {e}")
    for f in rep.get("files", []):
        L.append(f"  file {f['file']}: vendor {f['vendor'] or '?'}"
                 + (f" ({f['vendor_source']})" if f.get("vendor_source") else "")
                 + f", {f['conveyance']}" + (f" ({f['conveyance_reason']})" if f.get("conveyance_reason") else "")
                 + (f", run {f['run']}" if f.get("run") else "")
                 + (f", DLIS tools {', '.join(f['dlis_tools'])}" if f.get("dlis_tools") else ""))
    if rep.get("families"):
        L.append(f"  {'family':6s} {'tool(s)':26s} {'kind':18s} {'conv':9s} {'res m':>5s} {'cover m':>8s} {'gaps':>4s} "
                 f"{'spikes':>6s} {'badhole%':>8s}")
        for fam, i in rep["families"].items():
            L.append(f"  {fam:6s} {' / '.join(i['tools'])[:26]:26s} {i['kind'][:18]:18s} {i['conveyance'][:9]:9s} "
                     f"{i['eff_res_m']:5.2f} {i['coverage_m']:8.1f} {i['gaps']:4d} {i['spikes']:6d} {i['bad_hole_pct']:8.1f}")
    b = rep.get("beds") or {}
    if b:
        L.append(f"  beds from {b['from']} (res {b['resolution_m']} m): {b['beds']} beds, median "
                 f"{b['median_thickness_m']} m; below resolution: "
                 + ", ".join(f"{f} {t['thin_pct_of_beds']}%" for f, t in b["thin"].items()))
    for s in rep.get("shifts", []):
        L.append(f"  depth shift {s['file']} {s['curve']} vs {s['reference']}: {s['shift_m']:+.2f} m (corr {s['corr']}, "
                 f"{s['overlap_m']} m)")
    for w in rep.get("warnings", []):
        L.append(f"  ! {w}")
    return "\n".join(L)
