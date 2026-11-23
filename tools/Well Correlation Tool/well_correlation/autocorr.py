"""Suggest tops by log-pattern matching ('Suggest pick').

For each formation top known in one or more control wells, take a window of GR and
log10(deep resistivity) around the top in each control well, z-score normalise it (so tool
calibration differences between old and new GR do not matter), slide it through the target
well within a search range, and score the match with normalised cross-correlation (-1..1).
Scores from several controls are combined, weighted by 1/distance.

The predicted depth comes from the interval thickness (isochore) in the controls below the
nearest top that the target well already has - so picks you have made guide the next ones.
Without such an anchor, the distance-weighted TVDSS of the controls is used with a wider search.
"""
import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

STEP = 0.1524


def _tvdss_series(well, fam):
    v = well.logs.get(fam)
    if v is None or well.depth is None or not len(well.depth):
        return None, None
    z = well.md_to_tvdss(well.depth)
    if fam in ("RDEEP", "RSHAL"):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log10(np.where(v > 0, v, np.nan))
    return z, v


def _resample(z, v, grid):
    ok = np.isfinite(z) & np.isfinite(v)
    if ok.sum() < 10:
        return np.full(grid.shape, np.nan)
    return np.interp(grid, z[ok], v[ok], left=np.nan, right=np.nan)


def _ncc_sliding(series, window):
    """Normalised cross-correlation of `window` against every position of `series`.
    Positions with less than 60 % valid samples give NaN."""
    L = len(window)
    if len(series) < L:
        return np.array([])
    W = sliding_window_view(series, L)  # (n, L)
    okw = np.isfinite(window)
    valid = np.isfinite(W) & okw[None, :]
    cnt = valid.sum(axis=1)
    Wf = np.where(valid, W, 0.0)
    wf = np.where(valid, window[None, :], 0.0)
    c = np.maximum(cnt, 1)
    mw, mt = wf.sum(1) / c, Wf.sum(1) / c
    a = np.where(valid, wf - mw[:, None], 0.0)
    b = np.where(valid, Wf - mt[:, None], 0.0)
    num = (a * b).sum(1)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    r = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    r[cnt < 0.6 * L] = np.nan
    return r


def suggest_tops(project, target, controls=None, formations=None, half_window=40.0, search=120.0,
                 families=("GR", "RDEEP"), isochore_search=40.0, use_existing=True):
    """Return {formation: (md, score, predicted_tvdss)} for the target well.

    controls: wells to learn from (default: every other well that has the formation).
    use_existing: anchor on tops the target well already has (yours or earlier suggestions).
    """
    tw = project.load_well(target)
    if tw.x is None or tw.depth is None or not len(tw.depth):
        return {}
    order = [f["name"] for f in project.formations]
    formations = formations or order
    if controls is None:
        controls = [n for n in project.wells if n != target and project.tops.get(n)]
    ctrl = []
    for c in controls:
        if c == target or c not in project.wells:
            continue
        cw = project.load_well(c)
        if cw.x is None or cw.depth is None or not len(cw.depth):
            continue
        d = max(0.2, math.hypot(cw.x - tw.x, cw.y - tw.y) / 1000.0)
        ctrl.append((cw, 1.0 / d))
    out = {}
    nwin = int(round(half_window / STEP))
    series_cache = {}

    def anchor_for(f):
        """Deepest formation above f that the target already has (or was just suggested)."""
        if f not in order:
            return None, None
        best = (None, None)
        for g in order[: order.index(f)]:
            md = out.get(g, (None,))[0] if g in out else (project.tops.get(target, {}).get(g) if use_existing else None)
            if md is not None:
                best = (g, float(tw.md_to_tvdss([md])[0]))
        return best

    for f in formations:
        known = [(cw, wt, float(cw.md_to_tvdss([project.tops[cw.name][f]])[0])) for cw, wt in ctrl
                 if f in project.tops.get(cw.name, {})
                 and project.tops_meta.get(cw.name, {}).get(f, {}).get("confidence") != "low"]
        if not known:
            continue
        pred = sum(wt * z for _, wt, z in known) / sum(wt for _, wt, _ in known)
        srch = search
        prev_f, last_z = anchor_for(f)
        if prev_f is not None:
            th = [(wt, z - float(cw.md_to_tvdss([project.tops[cw.name][prev_f]])[0])) for cw, wt, z in known
                  if prev_f in project.tops.get(cw.name, {})]
            if th:
                pred = last_z + sum(w * t for w, t in th) / sum(w for w, _ in th)
                srch = isochore_search
        lo = pred - srch if prev_f is None else max(pred - srch, last_z + 2.0)
        hi = pred + srch
        if hi <= lo:
            continue
        cand = np.arange(lo, hi, STEP)
        score = np.zeros(cand.shape)
        wsum = np.zeros(cand.shape)
        for fam in families:
            tz, tv = _tvdss_series(tw, fam)
            if tz is None:
                continue
            tgrid = cand[0] - nwin * STEP + np.arange(len(cand) + 2 * nwin) * STEP
            tser = _resample(tz, tv, tgrid)
            for cw, wt, zc in known:
                key = (cw.name, fam)
                if key not in series_cache:
                    series_cache[key] = _tvdss_series(cw, fam)
                cz, cv = series_cache[key]
                if cz is None:
                    continue
                cwin = _resample(cz, cv, zc + np.arange(-nwin, nwin + 1) * STEP)
                if np.isfinite(cwin).sum() < nwin:
                    continue
                r = _ncc_sliding(tser, cwin)[: len(cand)]
                ok = np.isfinite(r)
                r = r - 0.15 * np.abs(cand[: len(r)] - pred) / srch  # mild pull toward the prediction
                score[: len(r)][ok] += wt * r[ok]
                wsum[: len(r)][ok] += wt
        okc = wsum > 0
        if not okc.any():
            continue
        s = np.where(okc, score / np.where(okc, wsum, 1), -np.inf)
        i = int(np.argmax(s))
        zbest = float(cand[i])
        md = float(tw.tvdss_to_md([zbest])[0])
        out[f] = (md, float(s[i]), float(pred))
    return out


def confidence(score):
    return "high" if score >= 0.45 else "medium" if score >= 0.25 else "low"
