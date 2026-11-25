"""Well Correlation Tool <-> Live Map bus (PLAN 31a).  The bus module lives in the Seismic Interpretation Tool
(`seismic_tool/mapbus.py`, numpy + stdlib only) and is imported from the sibling folder; without it the link is
simply off.  No Qt here: the main window is duck-typed so the logic is testable headless.

Publishes: selected well (selection + active well), the wells of the current cross section (active 'section' with
the map trace), tops edits (DataChanged 'tops' + WellTable columns 'Top <formation>' in TVDSS m, owner 'wellcorr').
Follows: a well selected in another tool (Seismic basemap, Live Map, QGIS via the seismic tool) is selected here.
"""
import importlib
import os
import sys

ROLE = "wellcorr"


def import_mapbus():
    try:
        return importlib.import_module("seismic_tool.mapbus")
    except Exception:
        pass
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sib = os.path.join(here, "Seismic Interpretation Tool")
    if os.path.isdir(sib) and sib not in sys.path:
        sys.path.append(sib)  # appended: never shadows this tool's own modules
    try:
        return importlib.import_module("seismic_tool.mapbus")
    except Exception:
        return None


def tops_tvdss(project):
    """{well: {formation: TVDSS m}} from the MD tops (vertical wells: TVDSS = MD - elevation)."""
    out = {}
    for wname, d in (project.tops or {}).items():
        w = project.wells.get(wname)
        if w is None:
            continue
        row = {}
        for fm, md in (d or {}).items():
            if md is None:
                continue
            try:
                row[fm] = float(w.md_to_tvdss(float(md)))
            except Exception:
                continue
        out[wname] = row
    return out


class WellCorrMapLink:
    def __init__(self, win, mb=None):
        self.win = win
        self.MB = mb or import_mapbus()
        self.bus = self.MB.MapBus() if self.MB else None
        self.bridge = None
        self.messages = []
        self._following = False
        if self.bus is not None:
            self.bus.subscribe(ROLE, self._on_event)

    @property
    def enabled(self):
        return self.bus is not None

    def attach(self, project):
        if not self.enabled:
            return None
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
        path = getattr(project, "path", None)
        if not path:
            return None
        folder = self.MB.bus_folder(None, path)
        self.bridge = self.MB.FileBridge(self.bus, folder, ROLE)
        self.push_tops(notify=False)
        return folder

    def poll(self):
        return self.bridge.poll() if self.bridge is not None else []

    def close(self):
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None

    # ------------------------------------------------------------------ out
    def well_selected(self, name):
        if not self.enabled or self._following or not name:
            return None
        self.bus.selection.select("wells", [name], source=ROLE)
        return self.bus.active.set("well", name, source=ROLE)

    def section_changed(self, name, wells):
        if not self.enabled:
            return None
        p = self.win.project
        xy = []
        for n in wells or []:
            w = p.wells.get(n)
            if w is not None and w.x is not None and w.y is not None:
                xy.append([round(float(w.x), 2), round(float(w.y), 2)])
        return self.bus.active.set("section", name or "", source=ROLE, wells=list(wells or []), xy=xy,
                                   kind2="well section")

    def push_tops(self, wells=None, formation=None, notify=True):
        """Write the 'Top <fm>' columns (TVDSS) for every well into the WellTable and tell the other tools."""
        if not self.enabled or self.win.project is None:
            return None
        p = self.win.project
        t = self.bus.table
        wl = [(n, w.x, w.y) for n, w in p.wells.items() if w.x is not None and w.y is not None]
        if wl:
            self.MB.table_from_wells(wl, t)
            for c in ("X", "Y"):
                t.meta[c]["owner"] = ROLE
        for c in t.columns(owner=ROLE):
            if c.startswith("Top "):
                t.remove_column(c)
        self.MB.tops_columns(t, tops_tvdss(p), owner=ROLE)
        if self.bridge is not None:
            try:
                self.bridge.save_table(ROLE)
            except OSError as e:
                self.messages.append(f"Map bus: could not write the well table ({e})")
        if notify:
            return self.bus.notify("tops", ids=list(wells or []), source=ROLE, formation=formation or "")
        return None

    # ------------------------------------------------------------------ in
    def _on_event(self, ev):
        if ev.get("origin") == self.bus.origin:
            return
        w = self.win
        who = ev.get("role") or ev.get("source")
        if ev["topic"] == "selection" and ev.get("kind") == "wells":
            ids = ev.get("ids") or []
            p = w.project
            names = [n for n in ids if p is not None and n in p.wells]
            if names:
                self._follow(names[0])
                self.messages.append(f"{len(ids)} well(s) selected in {who}")
        elif ev["topic"] == "active" and ev.get("kind") == "well":
            if w.project is not None and ev.get("value") in w.project.wells:
                self._follow(ev["value"])

    def _follow(self, name):
        self._following = True
        try:
            self.win.select_well(name)
        finally:
            self._following = False
