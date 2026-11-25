"""Mistie Tool <-> Live Map bus (PLAN 31a).  The bus module is imported from the sibling Seismic Interpretation
Tool folder (numpy + stdlib only); without it the link is off.  No tkinter here (testable headless).

Which bus folder: the project config key "mapbus_folder" (relative to the project file), else "well_project" (a
.wcproj.json -> its `<stem>_mapbus` folder, shared with the Well Correlation and Seismic tools).  Neither -> off.

Publishes: the current tie's lines (selection 'lines' + active 'line' = line A), the network solution after every
change (DataChanged 'misties' with per-line shift / phase / gain and the RMS before / after).
Follows: an active 2D line chosen in another tool selects the worst tie on that line.
"""
import importlib
import os
import sys

ROLE = "mistie"


def import_mapbus():
    try:
        return importlib.import_module("seismic_tool.mapbus")
    except Exception:
        pass
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sib = os.path.join(here, "Seismic Interpretation Tool")
    if os.path.isdir(sib) and sib not in sys.path:
        sys.path.append(sib)
    try:
        return importlib.import_module("seismic_tool.mapbus")
    except Exception:
        return None


def folder_for(session):
    """Bus folder of a Mistie session (None = no link configured)."""
    MB = import_mapbus()
    if MB is None:
        return None
    cfg = getattr(session, "cfg_raw", None) or {}
    base = getattr(session, "base_dir", None) or "."
    if cfg.get("mapbus_folder"):
        f = os.path.join(base, cfg["mapbus_folder"])
        os.makedirs(f, exist_ok=True)
        return f
    if cfg.get("well_project"):
        return MB.bus_folder(None, os.path.join(base, cfg["well_project"]))
    return None


class MistieMapLink:
    def __init__(self, app=None, mb=None):
        self.app = app
        self.MB = mb or import_mapbus()
        self.bus = self.MB.MapBus() if self.MB else None
        self.bridge = None
        self.messages = []
        self._last_sig = None
        self._following = False
        if self.bus is not None:
            self.bus.subscribe(ROLE, self._on_event)

    @property
    def enabled(self):
        return self.bus is not None

    def attach(self, session, folder=None):
        if not self.enabled:
            return None
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
        folder = folder or folder_for(session)
        if not folder:
            return None
        self.bridge = self.MB.FileBridge(self.bus, folder, ROLE)
        self._last_sig = None
        return folder

    def poll(self):
        return self.bridge.poll() if self.bridge is not None else []

    def close(self):
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None

    # ------------------------------------------------------------------ out
    def tie_selected(self, tie):
        if self.bridge is None or self._following or not tie:
            return None
        self.bus.selection.select("lines", [tie["a"], tie["b"]], source=ROLE)
        return self.bus.active.set("line", tie["a"], source=ROLE, cdp=float(tie.get("ca", 0)), other=tie["b"],
                                   x=float(tie.get("x", 0)), y=float(tie.get("y", 0)))

    def solution_changed(self, session):
        """Publish the solution when it differs from the last one sent.  Returns the event or None."""
        if self.bridge is None or session is None:
            return None
        sol = getattr(session.P, "sol", None) or {}
        lines = {L: {k: round(float(v[k]), 3) for k in ("shift", "phase", "gain") if k in v} for L, v in sol.items()}
        try:
            s = session.summary()
        except Exception:
            s = {}
        summ = {k: round(float(v), 3) for k, v in s.items() if isinstance(v, (int, float))}
        sig = repr((sorted(lines.items()), sorted(summ.items())))
        if sig == self._last_sig:
            return None
        self._last_sig = sig
        ties = []
        for t in getattr(session.P, "ties", []) or []:
            try:
                ties.append({"a": t["a"], "b": t["b"], "x": float(t["x"]), "y": float(t["y"]),
                             "shift_ms": round(float(t.get("shift", 0.0)), 3),
                             "res_shift_ms": round(float(t.get("res_shift", 0.0)), 3)})
            except (KeyError, TypeError, ValueError):
                continue
        return self.bus.notify("misties", ids=sorted(lines), source=ROLE, solution=lines, summary=summ,
                               reference=getattr(session.P, "ref", None), ties=ties)

    # ------------------------------------------------------------------ in
    def _on_event(self, ev):
        if ev.get("origin") == self.bus.origin or self.app is None:
            return
        if ev["topic"] == "active" and ev.get("kind") == "line":
            S = getattr(self.app, "S", None)
            if S is None:
                return
            k = worst_tie_on(S, ev.get("value"))
            if k is not None and k != getattr(self.app, "cur", None):
                self._following = True
                try:
                    self.app.select(k)
                finally:
                    self._following = False
                self.messages.append(f"Line {ev.get('value')} from {ev.get('role') or ev.get('source')}")


def worst_tie_on(session, line):
    """Index of the worst-scoring tie involving `line` (None if the line has no tie)."""
    best, bk = None, -1.0
    for k, t in enumerate(session.P.ties):
        if line not in (t["a"], t["b"]):
            continue
        try:
            sc = session.score(t)
        except Exception:
            sc = 0.0
        if sc > bk:
            best, bk = k, sc
    return best
