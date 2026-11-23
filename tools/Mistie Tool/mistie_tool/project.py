"""Project: config, lines, ties, picks, outputs."""
import csv, json, os, itertools
import numpy as np
from .segy import SegyLine
from .geometry import read_nav_csv, polyline_from_traces, intersections
from .measure import measure_mistie, half_window_difference, apply_correction
from . import network as net

DEFAULTS = dict(data_dir=".", nav_csv=None, output_dir="mistie_output",
                header_bytes=dict(cdp=21, x=181, y=185, scalar=71),
                default_window=[0.4, 2.6], max_lag_ms=40.0, tie_avg_traces=5,
                panel_half_width=60, display_tmax=None, min_corr=0.5, max_half_diff_ms=10.0,
                near_end_traces=200, near_end_weight=0.3, reject_tol_ms=4.0, reject_tol_deg=35.0,
                reference="auto")


class Project:
    def __init__(self, config_path=None, cfg_dict=None, base_dir=None, progress=None, use_picks_file=True):
        if cfg_dict is None:
            config_path = os.path.abspath(config_path)
            cfg_dict = json.load(open(config_path))
            base_dir = os.path.dirname(config_path)
        self.config_path = config_path or os.path.join(base_dir, "_unsaved_")
        self.base_dir = base = os.path.abspath(base_dir)
        self.progress = progress or (lambda msg, frac=None: None)
        cfg = dict(DEFAULTS); cfg.update(cfg_dict)
        cfg["header_bytes"] = {**DEFAULTS["header_bytes"], **cfg.get("header_bytes", {})}
        self.cfg = cfg
        self.data_dir = os.path.join(base, cfg["data_dir"])
        self.out_dir = os.path.join(base, cfg["output_dir"])
        os.makedirs(self.out_dir, exist_ok=True)
        self.picks_path = os.path.join(self.out_dir, "picks.json")
        hb = cfg["header_bytes"]
        self.lines, self.names = {}, []
        for k, item in enumerate(cfg["lines"]):
            self.progress(f"Reading {item['segy']}", k / max(1, len(cfg["lines"])))
            L = SegyLine(os.path.join(self.data_dir, item["segy"]), item["name"],
                         hb["cdp"], hb["x"], hb["y"], hb["scalar"])
            self.lines[item["name"]] = L; self.names.append(item["name"])
        dts = {round(L.dt, 6) for L in self.lines.values()}
        if len(dts) > 1:
            raise ValueError(f"lines have different sample rates {dts}; resample first")
        self.dt = dts.pop()
        self.geom = self._geometry()
        self.ties = self._find_ties()
        if use_picks_file:
            self.load_picks()
        for k, t in enumerate(self.ties):
            self.progress("Measuring ties", k / len(self.ties))
            self.measure(t)
        self.ref = cfg["reference"] if cfg["reference"] != "auto" else \
            net.choose_reference(self.names, self.ties, [net.is_used(t, cfg) for t in self.ties])

    # ---------- geometry
    def _geometry(self):
        nav = read_nav_csv(os.path.join(self.base_dir, self.cfg["nav_csv"])) \
            if self.cfg["nav_csv"] else {}
        norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
        nav_norm = {norm(k): v for k, v in nav.items()}
        geom = {}
        for n, L in self.lines.items():
            stem = os.path.splitext(os.path.basename(L.path))[0]
            hit = nav.get(n) if n in nav else nav_norm.get(norm(n), nav_norm.get(norm(stem)))
            if hit is not None:
                geom[n] = hit
            elif L.has_coords:
                geom[n] = polyline_from_traces(L.cdp, L.x, L.y)
            else:
                raise ValueError(f"{n}: no coordinates in trace headers (bytes "
                                 f"{self.cfg['header_bytes']['x']}/{self.cfg['header_bytes']['y']}) "
                                 f"and not in nav_csv. Supply a nav CSV (line,cdp,x,y).")
        return geom

    def _find_ties(self):
        ties = []
        for A, B in itertools.combinations(self.names, 2):
            for ca, cb, x, y in intersections(self.geom[A], self.geom[B]):
                LA, LB = self.lines[A], self.lines[B]
                ne = self.cfg["near_end_traces"]
                near = min(ca - LA.cdp[0], LA.cdp[-1] - ca) < ne or min(cb - LB.cdp[0], LB.cdp[-1] - cb) < ne
                ties.append(dict(a=A, b=B, ca=ca, cb=cb, x=x, y=y, near_end=bool(near),
                                 window=list(self.cfg["default_window"]), picked=False, use="auto"))
        if not ties:
            raise ValueError("no intersections found between the lines - check coordinates/units")
        return ties

    @staticmethod
    def key(t):
        return f"{t['a']}|{t['b']}|{t['ca']:.0f}|{t['cb']:.0f}"

    # ---------- traces
    def tie_trace(self, name, c):
        L = self.lines[name]
        i = int(np.argmin(np.abs(L.cdp - c)))
        h = self.cfg["tie_avg_traces"] // 2
        return L.data[max(0, i - h): i + h + 1].mean(axis=0), i

    def tie_traces(self, t):
        ns = min(self.lines[t["a"]].ns, self.lines[t["b"]].ns)
        a, ia = self.tie_trace(t["a"], t["ca"]); b, ib = self.tie_trace(t["b"], t["cb"])
        return a[:ns], b[:ns], ia, ib

    def panels(self, t):
        """Butt display: line A approaching the tie (left), line B leaving it (right)."""
        a, b, ia, ib = self.tie_traces(t)
        ns, n = len(a), self.cfg["panel_half_width"]
        dA, dB = self.lines[t["a"]].data[:, :ns], self.lines[t["b"]].data[:, :ns]
        pA = dA[max(0, ia - n): ia + 1] if ia - n >= 0 else dA[ia: ia + n + 1][::-1]
        pB = dB[ib: ib + n + 1] if ib + n < len(dB) else dB[max(0, ib - n): ib + 1][::-1]
        return pA, pB

    def measure(self, t):
        ck = (self.key(t), round(t["window"][0], 4), round(t["window"][1], 4))
        cache = self.__dict__.setdefault("_mcache", {})
        if ck in cache:
            t.update(cache[ck]); return
        a, b, _, _ = self.tie_traces(t)
        w, ml = t["window"], self.cfg["max_lag_ms"]
        t["shift"], t["phase"], t["gain"], t["corr"] = measure_mistie(a, b, self.dt, w, ml)
        t["half_diff"] = half_window_difference(a, b, self.dt, w, ml)
        cache[ck] = {k: t[k] for k in ("shift", "phase", "gain", "corr", "half_diff")}

    # ---------- network
    def use_flags(self):
        return [net.is_used(t, self.cfg) for t in self.ties]

    def solve(self):
        self.sol = net.solve(self.names, self.ties, self.use_flags(), self.ref, self.cfg)
        net.residuals(self.sol, self.ties)
        self.loops = net.loops(self.names, self.ties, self.use_flags())
        return self.sol

    def auto_reject(self):
        log = []
        self.sol, log = net.auto_reject(self.names, self.ties, self.ref, self.cfg, log)
        self.loops = net.loops(self.names, self.ties, self.use_flags())
        return log

    def summary(self):
        use = self.use_flags()
        U = [t for t, u in zip(self.ties, use) if u]
        rms = lambda v: float(np.sqrt(np.mean(np.square(v)))) if len(v) else float("nan")
        return dict(n_used=len(U), n_ties=len(self.ties),
                    before_ms=rms([t["shift"] for t in U]), after_ms=rms([t["res_shift"] for t in U]),
                    before_deg=rms(net.wrap([t["phase"] for t in U])), after_deg=rms([t["res_phase"] for t in U]))

    # ---------- persistence / outputs
    def load_picks(self):
        try:
            saved = json.load(open(self.picks_path))
        except (OSError, ValueError):
            return
        for t in self.ties:
            s = saved.get(self.key(t))
            if s:
                t["window"], t["picked"], t["use"] = s["window"], s.get("picked", True), s.get("use", "auto")

    def save(self):
        json.dump({self.key(t): dict(window=t["window"], picked=t["picked"], use=t["use"]) for t in self.ties},
                  open(self.picks_path, "w"), indent=1)
        use = self.use_flags()
        with open(os.path.join(self.out_dir, "ties.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["line_a", "cdp_a", "line_b", "cdp_b", "x", "y", "top_s", "base_s", "picked", "use_setting",
                        "used", "shift_ms", "phase_deg", "gain", "corr", "half_diff_ms", "near_end",
                        "residual_ms", "residual_deg"])
            for t, u in zip(self.ties, use):
                w.writerow([t["a"], f"{t['ca']:.1f}", t["b"], f"{t['cb']:.1f}", f"{t['x']:.1f}", f"{t['y']:.1f}",
                            f"{t['window'][0]:.3f}", f"{t['window'][1]:.3f}", t["picked"], t["use"], u,
                            f"{t['shift']:.2f}", f"{t['phase']:.1f}", f"{t['gain']:.3f}", f"{t['corr']:.3f}",
                            f"{t['half_diff']:.1f}", t["near_end"], f"{t['res_shift']:.2f}", f"{t['res_phase']:.1f}"])
        with open(os.path.join(self.out_dir, "corrections.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["line", "shift_ms", "phase_deg", "gain", "ties_used", "reference"])
            for L in self.names:
                c = self.sol[L]
                w.writerow([L, f"{c['shift']:.2f}", f"{c['phase']:.1f}", f"{c['gain']:.4f}", c["n_ties"], L == self.ref])
        with open(os.path.join(self.out_dir, "loops.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["loop", "closure_ms", "closure_deg"])
            for lp in self.loops:
                w.writerow([" > ".join(lp["lines"] + [lp["lines"][0]]), f"{lp['shift']:.2f}", f"{lp['phase']:.1f}"])
        return self.out_dir

    def write_corrected_segy(self):
        out = []
        for n, L in self.lines.items():
            c = self.sol[n]
            data = apply_correction(L.data, self.dt, c["shift"], c["phase"], c["gain"]).astype(np.float32)
            p = os.path.join(self.out_dir, os.path.splitext(os.path.basename(L.path))[0] + "_mistie.sgy")
            L.write(p, data); out.append(p)
        return out
