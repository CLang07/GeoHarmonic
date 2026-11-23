"""Editable session on top of a Project: project files, undo/redo, bulk edits, locks,
survey grouping, scenarios, ranking. No GUI code here, so it can be tested anywhere."""
import copy, csv, glob, json, os
import numpy as np
from .project import Project, DEFAULTS
from . import network as net

FORMAT = "mistie_tool project"
SEGY_PATTERNS = ("*.sgy", "*.SGY", "*.segy", "*.SEGY")


class Session:
    MAX_UNDO = 200

    def __init__(self, path=None, cfg=None, base_dir=None, progress=None):
        """Open a project file (.mistie.json), a plain config JSON, or a config dict."""
        state, scenarios = None, {}
        if path is not None:
            path = os.path.abspath(path)
            d = json.load(open(path))
            base_dir = os.path.dirname(path)
            if d.get("format") == FORMAT:
                cfg, state, scenarios = d["config"], d.get("state"), d.get("scenarios", {})
                self.path = path
            else:
                cfg, self.path = d, None          # plain config: becomes a project on first save
                self.default_save = os.path.join(base_dir, os.path.splitext(os.path.basename(path))[0] + ".mistie.json")
        else:
            self.path = None
        if not hasattr(self, "default_save"):
            self.default_save = os.path.join(base_dir, "project.mistie.json")
        self.cfg_raw = copy.deepcopy(cfg)
        self.base_dir = base_dir
        self.P = Project(cfg_dict=cfg, base_dir=base_dir, progress=progress, use_picks_file=False)
        self.names = self.P.names
        self.survey_of = {item["name"]: item.get("survey", item["name"]) for item in cfg["lines"]}
        self.lines_state = {L: dict(lock=False, shift=0.0, phase=0.0, gain=1.0) for L in self.names}
        self.solve_by = "line"
        self.scenarios = scenarios
        self.undo_stack, self.redo_stack = [], []
        self.dirty = False
        if state:
            self._apply_state(state)
        self.solve()

    # ------------------------------------------------------------------ state
    def state(self):
        return dict(
            ties={self.P.key(t): dict(window=list(t["window"]), picked=t["picked"], use=t["use"]) for t in self.P.ties},
            lines=copy.deepcopy(self.lines_state), reference=self.P.ref, solve_by=self.solve_by)

    def _apply_state(self, st):
        for t in self.P.ties:
            s = st.get("ties", {}).get(self.P.key(t))
            if s:
                changed = [round(v, 4) for v in s["window"]] != [round(v, 4) for v in t["window"]]
                t["window"], t["picked"], t["use"] = list(s["window"]), s["picked"], s["use"]
                if changed or "shift" not in t:
                    self.P.measure(t)
        for L, v in st.get("lines", {}).items():
            if L in self.lines_state:
                self.lines_state[L].update(v)
        if st.get("reference") in self.names:
            self.P.ref = st["reference"]
        self.solve_by = st.get("solve_by", self.solve_by)

    def _change(self, fn):
        before = self.state()
        try:
            fn()
        except Exception:
            self._apply_state(before)  # a failure half-way must not leave some ties changed without an undo step
            raise
        self.undo_stack.append(before)
        del self.undo_stack[:-self.MAX_UNDO]
        self.redo_stack.clear()
        self.dirty = True
        self.solve()

    def undo(self):
        if not self.undo_stack:
            return False
        self.redo_stack.append(self.state()); self._apply_state(self.undo_stack.pop())
        self.dirty = True; self.solve(); return True

    def redo(self):
        if not self.redo_stack:
            return False
        self.undo_stack.append(self.state()); self._apply_state(self.redo_stack.pop())
        self.dirty = True; self.solve(); return True

    # ------------------------------------------------------------------ edits (all undoable)
    def set_window(self, idxs, window, picked=True):
        w = [float(window[0]), float(window[1])]
        if w[1] - w[0] < 0.1:
            raise ValueError("window must be at least 0.1 s long")

        def f():
            for i in idxs:
                t = self.P.ties[i]; t["window"] = list(w); t["picked"] = picked; self.P.measure(t)
        self._change(f)

    def reset_windows(self, idxs):
        self.set_window(idxs, self.P.cfg["default_window"], picked=False)

    def set_use(self, idxs, mode):
        assert mode in ("auto", "on", "off")
        self._change(lambda: [self.P.ties[i].__setitem__("use", mode) for i in idxs])

    def set_reference(self, line):
        self._change(lambda: setattr(self.P, "ref", line))

    def set_lock(self, line, lock, shift=None, phase=None, gain=None):
        def f():
            s = self.lines_state[line]
            s["lock"] = bool(lock)
            if shift is not None: s["shift"] = float(shift)
            if phase is not None: s["phase"] = float(phase)
            if gain is not None: s["gain"] = float(gain)
        self._change(f)

    def lock_at_current(self, line):
        c = self.P.sol[line]
        self.set_lock(line, True, c["shift"], c["phase"], c["gain"])

    def set_solve_by(self, mode):
        assert mode in ("line", "survey")
        self._change(lambda: setattr(self, "solve_by", mode))

    def all_auto(self):
        self.set_use(range(len(self.P.ties)), "auto")

    def reject_worst(self):
        """Exclude the worst-fitting used tie unless that would disconnect a line. Returns the tie index or None."""
        use = self.P.use_flags()
        order = [i for i in self.ranking() if use[i]]
        for i in order[:1]:
            trial = list(use); trial[i] = False
            if not net.connected(self.names, self.P.ties, trial):
                return None
            self.set_use([i], "off")
            return i
        return None

    def auto_reject(self):
        out = []
        while True:
            i = self.reject_worst_if_bad()
            if i is None:
                return out
            out.append(i)

    def reject_worst_if_bad(self):
        r = self.ranking()
        use = self.P.use_flags()
        for i in r:
            if use[i] and self.P.ties[i]["use"] == "auto":
                if self.score(self.P.ties[i]) <= 2.0:
                    return None
                trial = list(use); trial[i] = False
                if not net.connected(self.names, self.P.ties, trial):
                    return None
                self.set_use([i], "off")
                return i
        return None

    # ------------------------------------------------------------------ scenarios
    def save_scenario(self, name):
        self.scenarios[name] = self.state(); self.dirty = True

    def load_scenario(self, name):
        st = self.scenarios[name]
        self._change(lambda: self._apply_state(st))

    def delete_scenario(self, name):
        self.scenarios.pop(name, None); self.dirty = True

    # ------------------------------------------------------------------ solving / QC
    def solve(self):
        use = self.P.use_flags()
        group_of = {L: (self.survey_of[L] if self.solve_by == "survey" else L) for L in self.names}
        fixed = {}
        for L, s in self.lines_state.items():
            if s["lock"]:
                fixed[group_of[L]] = (s["shift"], s["phase"], s["gain"])
        self.P.sol = net.solve_groups(self.names, self.P.ties, use, self.P.ref, self.P.cfg, group_of, fixed)
        net.residuals(self.P.sol, self.P.ties)
        self.P.loops = net.loops(self.names, self.P.ties, use)
        return self.P.sol

    def score(self, t):
        c = self.P.cfg
        return abs(t["res_shift"]) / c["reject_tol_ms"] + abs(t["res_phase"]) / c["reject_tol_deg"]

    def is_misfit(self, t):
        c = self.P.cfg
        return abs(t["res_shift"]) > c["reject_tol_ms"] or abs(t["res_phase"]) > c["reject_tol_deg"]

    def ranking(self):
        """Tie indices worst first: used ties by misfit score, then unused ties by correlation."""
        use = self.P.use_flags()
        used = sorted([i for i, u in enumerate(use) if u], key=lambda i: -self.score(self.P.ties[i]))
        unused = sorted([i for i, u in enumerate(use) if not u], key=lambda i: self.P.ties[i]["corr"])
        return used + unused

    def next_worst(self, current):
        r = self.ranking()
        if current not in r:
            return r[0]
        return r[(r.index(current) + 1) % len(r)]

    def summary(self):
        return self.P.summary()

    def warnings(self):
        out = []
        for L in self.names:
            c = self.P.sol[L]
            if c["fixed"] or L == self.P.ref:
                continue
            if c["n_ties"] == 0:
                out.append(f"{L} has no usable ties - its correction is undetermined (shown as 0)")
            elif c["n_ties"] == 1:
                out.append(f"{L} rests on a single tie - no independent check")
        for lp in self.P.loops:
            if abs(lp["shift"]) > 5 or abs(lp["phase"]) > 40:
                out.append("loop " + " > ".join(lp["lines"]) + f" does not close ({lp['shift']:+.1f} ms, {lp['phase']:+.0f} deg)")
        return out

    # ------------------------------------------------------------------ files
    def save(self, path=None):
        path = os.path.abspath(path or self.path or self.default_save)
        new_base = os.path.dirname(path)
        cfg = copy.deepcopy(self.cfg_raw)
        for key in ("data_dir", "nav_csv", "output_dir"):
            v = cfg.get(key, DEFAULTS.get(key))
            if v:
                absv = os.path.normpath(os.path.join(self.base_dir, v))
                try:
                    cfg[key] = os.path.relpath(absv, new_base)
                except ValueError:                  # different drive on Windows
                    cfg[key] = absv
        d = dict(format=FORMAT, version=2, config=cfg, state=self.state(), scenarios=self.scenarios)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, path)
        self.path, self.cfg_raw, self.base_dir, self.dirty = path, cfg, new_base, False
        return path

    def export_csv(self, out_dir=None):
        if out_dir:
            self.P.out_dir = out_dir; os.makedirs(out_dir, exist_ok=True)
        d = self.P.save()
        with open(os.path.join(self.P.out_dir, "corrections.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["line", "survey", "shift_ms", "phase_deg", "gain", "ties_used", "reference", "locked"])
            for L in self.names:
                c = self.P.sol[L]
                w.writerow([L, self.survey_of[L], f"{c['shift']:.2f}", f"{c['phase']:.1f}", f"{c['gain']:.4f}",
                            c["n_ties"], L == self.P.ref, self.lines_state[L]["lock"]])
        try:
            os.remove(self.P.picks_path)            # picks live in the project file now
        except OSError:
            pass
        return d

    def write_corrected_segy(self, out_dir=None):
        if out_dir:
            self.P.out_dir = out_dir; os.makedirs(out_dir, exist_ok=True)
        return self.P.write_corrected_segy()

    # ------------------------------------------------------------------ new project from a folder
    @staticmethod
    def scan_folder(folder):
        files = sorted({os.path.basename(p) for pat in SEGY_PATTERNS for p in glob.glob(os.path.join(folder, pat))})
        return files

    @classmethod
    def from_folder(cls, folder, files=None, nav_csv=None, progress=None, **settings):
        files = files or cls.scan_folder(folder)
        if not files:
            raise ValueError(f"no SEG-Y files (*.sgy / *.segy) in {folder}")
        cfg = dict(data_dir=".", nav_csv=os.path.relpath(nav_csv, folder) if nav_csv else None,
                   output_dir="mistie_output",
                   lines=[dict(name=os.path.splitext(f)[0], segy=f) for f in files])
        cfg.update(settings)
        s = cls(cfg=cfg, base_dir=os.path.abspath(folder), progress=progress)
        s.default_save = os.path.join(os.path.abspath(folder), "project.mistie.json")
        s.dirty = True
        return s
