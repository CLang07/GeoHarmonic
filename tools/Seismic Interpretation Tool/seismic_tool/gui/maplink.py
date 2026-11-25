"""Seismic Interpretation Tool <-> map bus (PLAN 31a retrofit).  No Qt imports: the window is duck-typed, so the
logic runs under dev/fakeqt and in plain unit tests.

Publishes: selected wells (basemap lasso / QGIS), the current inline / crossline / time slice / 2D line + its map
trace ("section" with xy), horizon / fault edits (DataChanged picks / faults), depth-model updates (velocities +
grids), mistie solutions (misties).
Follows: well selections and active line / inline / well from the other tools (Well Correlation, Mistie, Live Map),
tops changes (starts the dynamic depth update when Auto-update is on), WellTable reloads.
"""
import numpy as np

from .. import mapbus as MB

ROLE = "seismic"


class SeismicMapLink:
    def __init__(self, win, bus=None):
        self.win = win
        self.bus = bus or MB.MapBus()
        self.bridge = None
        self.messages = []  # what the link did for the user (status bar), newest last
        self.bus.subscribe(ROLE, self._on_event)

    # ------------------------------------------------------------------ project
    def attach(self, project):
        """(Re)connect to the project's shared bus folder; seeds the WellTable with the project wells."""
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
        folder = MB.bus_folder(getattr(project, "path", None), getattr(project, "well_project", None) or None)
        if not folder:
            return None
        self.bridge = MB.FileBridge(self.bus, folder, ROLE)
        try:
            wells = project.wells()
        except Exception:
            wells = []
        if wells:
            have = {MB.norm_id(i) for i in self.bus.table.ids}
            if any(MB.norm_id(w[0]) not in have for w in wells) or "X" not in self.bus.table.cols:
                MB.table_from_wells(wells, self.bus.table)
                try:
                    self.bridge.save_table("project")
                except OSError as e:
                    self._say(f"Map bus: could not write the well table ({e})")
        return folder

    def poll(self):
        return self.bridge.poll() if self.bridge is not None else []

    def close(self):
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None

    # ------------------------------------------------------------------ out
    def wells_selected(self, names, source="map"):
        return self.bus.selection.select("wells", list(names or []), source=source)

    def position_changed(self):
        """Current section of the panel -> ActiveObject (inline / crossline / timeslice / line) + 'section' trace."""
        sec = self.win.section
        src = getattr(sec, "source", None)
        if not src:
            return None
        kind, name = src[0], src[1]
        mode, num = getattr(sec, "mode", None), getattr(sec, "number", None)
        ev = None
        if kind == "2D":
            ev = self.bus.active.set("line", name, source=ROLE)
        elif mode in ("Inline", "Crossline", "Time slice") and num is not None:
            k = {"Inline": "inline", "Crossline": "crossline", "Time slice": "timeslice"}[mode]
            ev = self.bus.active.set(k, _num(num), source=ROLE, volume=name)
            self.bus.active.set("volume", name, source=ROLE)
        xy = None
        try:
            xy = sec.current_trace_line()
        except Exception:
            pass
        if xy is not None and len(xy[0]):
            x, y = np.asarray(xy[0], float), np.asarray(xy[1], float)
            k = max(1, len(x) // 100)
            idx = np.unique(np.r_[np.arange(0, len(x), k), len(x) - 1])
            label = name if kind == "2D" else f"{name} {mode} {num}"
            self.bus.active.set("section", label, source=ROLE, xy=np.column_stack([x[idx], y[idx]]).round(2).tolist(),
                                kind2=kind, name=name, mode=mode, number=_num(num))
        return ev

    def horizons_changed(self, names=None):
        hs = names
        if hs is None:
            try:
                hs = [h.name for h in self.win.project.horizons.items]
            except Exception:
                hs = []
        return self.bus.notify("picks", ids=hs, source=ROLE)

    def faults_changed(self):
        try:
            names = [f.name for f in self.win.project.faults.items]
        except Exception:
            names = []
        return self.bus.notify("faults", ids=names, source=ROLE)

    def depth_model_changed(self, rep):
        rep = rep or {}
        return self.bus.notify("velocities", ids=list(rep.get("changes") or [])[:50], source=ROLE,
                               message=str(rep.get("message", "")), layers=list(rep.get("layers") or []))

    def misties_changed(self, lines=None, solution=None):
        info = {}
        if solution:
            info["solution"] = {str(k): {kk: _num(vv) for kk, vv in (v or {}).items() if _scalar(vv)}
                                for k, v in solution.items()}
        return self.bus.notify("misties", ids=list(lines or []), source=ROLE, **info)

    # ------------------------------------------------------------------ in
    def _on_event(self, ev):
        w = self.win
        t = ev["topic"]
        if t == "selection" and ev.get("kind") == "wells":
            try:
                w.basemap.set_selected_wells(ev.get("ids") or [], emit=False)
            except Exception:
                pass
            if ev.get("ids"):
                self._say(f"{len(ev['ids'])} well(s) selected in {ev.get('role') or ev.get('source')}")
        elif t == "active":
            self._follow_active(ev)
        elif t == "data":
            k = ev.get("kind")
            if k == "tops":
                if getattr(w, "ddc_auto", False):
                    try:
                        w._ddc_debounce.start()
                    except Exception:
                        pass
                self._say(f"Tops changed in {ev.get('role') or ev.get('source')}: {', '.join(ev.get('ids') or [])}")
            elif k in ("misties", "grids", "welltable", "velocities") and ev.get("origin") != self.bus.origin:
                self._say(f"Map bus: {k} updated by {ev.get('role') or ev.get('source')}")

    def _follow_active(self, ev):
        """Open what another tool made active - only when it is ours to show and not already shown."""
        if ev.get("origin") == self.bus.origin:
            return
        w = self.win
        sec = w.section
        proj = w.project
        k, val = ev.get("kind"), ev.get("value")
        try:
            if k == "line" and any(l["name"] == val for l in proj.lines):
                if tuple(sec.source or ()) != ("2D", val):
                    sec.show("2D", val)
                    self._say(f"Opened 2D line {val} (from {ev.get('role') or ev.get('source')})")
            elif k in ("inline", "crossline", "timeslice"):
                vol = ev.get("volume")
                if vol and any(v["name"] == vol for v in proj.volumes):
                    mode = {"inline": "Inline", "crossline": "Crossline", "timeslice": "Time slice"}[k]
                    if not (tuple(sec.source or ()) == ("3D", vol) and sec.mode == mode and _num(sec.number) == _num(val)):
                        sec.show("3D", vol, mode, val)
            elif k == "well":
                if val and any(MB.norm_id(n) == MB.norm_id(val) for n, _x, _y in proj.wells()):
                    w.basemap.set_selected_wells([val], emit=False)
        except Exception as e:
            self._say(f"Map bus: could not follow {k} {val}: {e}")

    def _say(self, msg):
        self.messages.append(msg)
        del self.messages[:-50]
        try:
            self.win.statusBar().showMessage(msg, 6000)
        except Exception:
            pass


def _num(v):
    try:
        f = float(v)
        return int(f) if f == int(f) else f
    except (TypeError, ValueError):
        return v


def _scalar(v):
    return isinstance(v, (int, float, bool, str, np.integer, np.floating))
