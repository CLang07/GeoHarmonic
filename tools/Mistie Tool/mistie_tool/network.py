"""Tie network: weighted least-squares solution, residuals, loop closure, robust rejection."""
import numpy as np


def wrap(p):
    return (np.asarray(p, dtype=float) + 180.0) % 360.0 - 180.0


def is_used(t, cfg):
    if t["use"] == "on":
        return True
    if t["use"] == "off":
        return False
    return t["corr"] >= cfg["min_corr"] and t["half_diff"] <= cfg["max_half_diff_ms"]


def weight(t, cfg):
    w = t["corr"] ** 2 / (1.0 + (t["half_diff"] / 5.0) ** 2)
    return w * (cfg["near_end_weight"] if t["near_end"] else 1.0)


def connected(lines, ties, use):
    adj = {L: set() for L in lines}
    for t, u in zip(ties, use):
        if u:
            adj[t["a"]].add(t["b"]); adj[t["b"]].add(t["a"])
    seen, stack = {lines[0]}, [lines[0]]
    while stack:
        for v in adj[stack.pop()]:
            if v not in seen:
                seen.add(v); stack.append(v)
    return len(seen) == len(lines)


def solve(lines, ties, use, ref, cfg):
    """Per-line (shift_ms, phase_deg, gain) so that c_b - c_a matches each used tie."""
    idx = {L: i for i, L in enumerate(lines)}
    T = [t for t, u in zip(ties, use) if u]
    n = len(lines)
    rows, w = [], []
    for t in T:
        r = np.zeros(n); r[idx[t["b"]]] += 1; r[idx[t["a"]]] -= 1
        rows.append(r); w.append(weight(t, cfg))
    r = np.zeros(n); r[idx[ref]] = 1; rows.append(r); w.append(1e6)
    for i in range(n):                      # tiny damping keeps isolated lines at zero
        r = np.zeros(n); r[i] = 1; rows.append(r); w.append(1e-6)
    A = np.array(rows); W = np.sqrt(np.array(w))

    def lsq(d):
        dd = np.concatenate([d, np.zeros(1 + n)])
        return np.linalg.lstsq(A * W[:, None], dd * W, rcond=None)[0]

    s = lsq(np.array([t["shift"] for t in T]))
    g = np.exp(lsq(np.log([max(t["gain"], 1e-6) for t in T]))) if T else np.ones(n)
    p = np.zeros(n)
    meas = np.array([t["phase"] for t in T])
    for _ in range(25):                     # wrap-aware iteration (handles polarity flips)
        pred = np.array([p[idx[t["b"]]] - p[idx[t["a"]]] for t in T])
        p = p + lsq(wrap(meas - pred))
    p = wrap(p)
    nties = {L: sum(1 for t in T if L in (t["a"], t["b"])) for L in lines}
    return {L: dict(shift=float(s[idx[L]]), phase=float(p[idx[L]]), gain=float(g[idx[L]]),
                    n_ties=nties[L]) for L in lines}


def residuals(sol, ties):
    for t in ties:
        a, b = sol[t["a"]], sol[t["b"]]
        t["res_shift"] = t["shift"] - (b["shift"] - a["shift"])
        t["res_phase"] = float(wrap(t["phase"] - (b["phase"] - a["phase"])))
    return ties


def loops(lines, ties, use):
    """Fundamental cycles of the tie graph and their raw closure (sum of misties around the loop)."""
    best = {}
    for k, (t, u) in enumerate(zip(ties, use)):
        key = tuple(sorted((t["a"], t["b"])))
        if u and (key not in best or t["corr"] > ties[best[key]]["corr"]):
            best[key] = k
    adj = {L: [] for L in lines}
    for (a, b), k in best.items():
        adj[a].append((b, k)); adj[b].append((a, k))
    parent, order, tree = {}, [], set()
    for root in lines:
        if root in parent:
            continue
        parent[root] = (None, None); order.append(root)
        q = [root]
        while q:
            v = q.pop(0)
            for u, k in adj[v]:
                if u not in parent:
                    parent[u] = (v, k); tree.add(k); q.append(u)
    out = []
    for (a, b), k in best.items():
        if k in tree:
            continue

        def path(x):
            o = [x]
            while parent[x][0] is not None:
                x = parent[x][0]; o.append(x)
            return o
        pa, pb = path(a), path(b)
        common = [x for x in pa if x in pb]
        if not common:
            continue
        c = common[0]
        cyc = pa[:pa.index(c) + 1] + pb[:pb.index(c)][::-1]
        cs = cp = 0.0
        for x, y in zip(cyc, cyc[1:] + cyc[:1]):
            t = ties[best[tuple(sorted((x, y)))]]
            sg = 1 if (t["a"], t["b"]) == (x, y) else -1
            cs += sg * t["shift"]; cp += sg * t["phase"]
        out.append(dict(lines=cyc, shift=cs, phase=float(wrap(cp))))
    return out


def choose_reference(lines, ties, use):
    """Line with the best average correlation, preferring lines with at least 3 ties."""
    score = {}
    for L in lines:
        c = [t["corr"] for t, u in zip(ties, use) if u and L in (t["a"], t["b"])]
        score[L] = (np.mean(c) if c else 0) * min(1.0, len(c) / 3.0)
    return max(score, key=score.get)


def auto_reject(lines, ties, ref, cfg, log=None):
    """Turn off the worst-fitting auto tie one at a time while the network stays connected."""
    log = [] if log is None else log
    while True:
        use = [is_used(t, cfg) for t in ties]
        sol = solve(lines, ties, use, ref, cfg); residuals(sol, ties)
        score = [abs(t["res_shift"]) / cfg["reject_tol_ms"] + abs(t["res_phase"]) / cfg["reject_tol_deg"]
                 if u and t["use"] == "auto" else -1 for t, u in zip(ties, use)]
        k = int(np.argmax(score))
        if score[k] <= 2.0:
            return sol, log
        trial = list(use); trial[k] = False
        if not connected(lines, ties, trial):
            return sol, log
        ties[k]["use"] = "off"
        log.append(f"rejected {ties[k]['a']} x {ties[k]['b']}: residual "
                   f"{ties[k]['res_shift']:+.1f} ms / {ties[k]['res_phase']:+.0f} deg")


def solve_groups(lines, ties, use, ref, cfg, group_of=None, fixed=None):
    """General solver.
    group_of: {line: group name} - lines in a group share one correction (e.g. survey/vintage).
    fixed:    {group: (shift, phase, gain)} - groups held at given values (locked / manual).
    ref:      group held at zero (ignored if it is in fixed)."""
    group_of = group_of or {L: L for L in lines}
    fixed = dict(fixed or {})
    groups = list(dict.fromkeys(group_of[L] for L in lines))
    gi = {g: i for i, g in enumerate(groups)}
    if ref is not None and group_of.get(ref, ref) not in fixed:
        fixed[group_of.get(ref, ref)] = (0.0, 0.0, 1.0)
    n = len(groups)
    T = [t for t, u in zip(ties, use) if u and group_of[t["a"]] != group_of[t["b"]]]
    rows, w = [], []
    for t in T:
        r = np.zeros(n); r[gi[group_of[t["b"]]]] += 1; r[gi[group_of[t["a"]]]] -= 1
        rows.append(r); w.append(weight(t, cfg))
    fx = [(gi[g], v) for g, v in fixed.items() if g in gi]
    for i, _ in fx:
        r = np.zeros(n); r[i] = 1; rows.append(r); w.append(1e6)
    for i in range(n):
        r = np.zeros(n); r[i] = 1; rows.append(r); w.append(1e-6)
    A = np.array(rows); W = np.sqrt(np.array(w))

    def lsq(tie_vals, fix_vals):
        d = np.concatenate([np.asarray(tie_vals, float), np.asarray(fix_vals, float), np.zeros(n)])
        return np.linalg.lstsq(A * W[:, None], d * W, rcond=None)[0]

    s = lsq([t["shift"] for t in T], [v[0] for _, v in fx])
    g = np.exp(lsq(np.log([max(t["gain"], 1e-6) for t in T]), np.log([max(v[2], 1e-6) for _, v in fx])))
    p = np.zeros(n)
    for _ in range(25):
        rt = [wrap(t["phase"] - (p[gi[group_of[t["b"]]]] - p[gi[group_of[t["a"]]]])) for t in T]
        rf = [wrap(v[1] - p[i]) for i, v in fx]
        p = p + lsq(rt, rf)
    p = wrap(p)
    nties = {L: sum(1 for t in T if L in (t["a"], t["b"])) for L in lines}
    out = {}
    for L in lines:
        k = gi[group_of[L]]
        out[L] = dict(shift=float(s[k]), phase=float(p[k]), gain=float(g[k]), n_ties=nties[L],
                      fixed=group_of[L] in fixed, group=group_of[L])
    return out
