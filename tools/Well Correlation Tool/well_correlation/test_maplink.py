"""Live Map bus link of the Well Correlation Tool (PLAN 31a)."""
import os
import types

from .maplink import WellCorrMapLink, import_mapbus, tops_tvdss
from .project import Project, Well


def _project(tmp):
    p = Project()
    for name, x, y, elev in (("Alpha-1", 1000.0, 2000.0, 25.0), ("Beta-1", 1500.0, 2600.0, 10.0),
                             ("Gamma-1", 2000.0, 2100.0, 0.0)):
        w = Well(name)
        w.x, w.y, w.elev = x, y, elev
        p.wells[name] = w
    p.formations = [{"name": "Sand A", "color": "#ffcc00"}, {"name": "Shale B", "color": "#888888"}]
    p.tops = {"Alpha-1": {"Sand A": 1500.0, "Shale B": 1600.0}, "Beta-1": {"Sand A": 1540.0}}
    p.sections = {"Section 1": ["Alpha-1", "Beta-1"]}
    p.save(os.path.join(tmp, "WC test.wcproj.json"))
    return p


class _Win:
    def __init__(self, p):
        self.project = p
        self.selected = []

    def select_well(self, name):  # what MainWindow.select_well does with the link
        self.selected.append(name)
        self.maplink.well_selected(name)


def run(check, tmp):
    MB = import_mapbus()
    check(MB is not None, "map bus: imported from the Seismic Interpretation Tool folder")
    if MB is None:
        return
    d = os.path.join(tmp, "wc_maplink")
    os.makedirs(d, exist_ok=True)
    p = _project(d)
    tv = tops_tvdss(p)
    check(tv["Alpha-1"]["Sand A"] == 1475.0 and tv["Beta-1"]["Sand A"] == 1530.0 and "Gamma-1" not in tv,
          "map bus: tops MD -> TVDSS for the WellTable")
    win = _Win(p)
    win.maplink = link = WellCorrMapLink(win, MB)
    folder = link.attach(p)
    check(folder and folder.endswith("WC test_mapbus") and os.path.exists(os.path.join(folder, "welltable.csv")),
          "map bus: folder next to the .wcproj.json, WellTable written on attach")
    seis = MB.MapBus()
    br = MB.FileBridge(seis, folder, "seismic")
    check(seis.table.value("Alpha-1", "Top Sand A") == 1475.0 and seis.table.value("Gamma-1", "X") == 2000.0,
          "map bus: another tool sees X / Y and Top columns")
    seen = []
    seis.subscribe("seis", seen.append)
    win.select_well("Beta-1")
    link.section_changed("Section 1", ["Alpha-1", "Beta-1"])
    br.poll()
    sec = seis.active.info("well_section")
    check(seis.selection.get("wells") == ["Beta-1"] and seis.active.get("well") == "Beta-1"
          and sec.get("wells") == ["Alpha-1", "Beta-1"] and sec.get("xy") == [[1000.0, 2000.0], [1500.0, 2600.0]],
          "map bus: selected well + section wells / trace published")
    # a top edit: column updated, DataChanged sent
    p.tops["Gamma-1"] = {"Shale B": 1700.0}
    link.push_tops(wells=["Gamma-1"], formation="Shale B")
    br.poll()
    ev = seis.last("data", "tops")
    check(ev is not None and ev["ids"] == ["Gamma-1"] and ev["formation"] == "Shale B"
          and seis.table.value("Gamma-1", "Top Shale B") == 1700.0, "map bus: top edit -> DataChanged + WellTable")
    # formation renamed: the old column goes away
    p.tops = {w: {("Sand X" if k == "Sand A" else k): v for k, v in d_.items()} for w, d_ in p.tops.items()}
    link.push_tops(formation="Sand X")
    br.poll()
    check("Top Sand A" not in seis.table.cols and seis.table.value("Alpha-1", "Top Sand X") == 1475.0,
          "map bus: renamed formation replaces its column")
    # a selection made in the seismic tool is followed, without echo
    n = len(seis.history)
    seis.selection.select("wells", ["NotHere", "Gamma-1"], source="map")
    link.poll()
    br.poll()
    check(win.selected[-1] == "Gamma-1" and len(seis.history) == n + 1, "map bus: foreign selection followed, no echo")
    seis.active.set("well", "Alpha-1", source="map")
    link.poll()
    check(win.selected[-1] == "Alpha-1", "map bus: active well from another tool selected here")
    # a section built on the Live Map opens here as its own section (user sections untouched)
    seis.active.set("well_section", "Live Map", source="livemap", wells=["Gamma-1", "Nope", "Alpha-1"], xy=[])
    link.poll()
    check(p.sections.get("Live Map") == ["Gamma-1", "Alpha-1"] and p.current_section == "Live Map"
          and p.sections["Section 1"] == ["Alpha-1", "Beta-1"], "map bus: Live Map cross section followed")
    link.close()
    br.close()
    # no project path -> off, no crash
    link2 = WellCorrMapLink(types.SimpleNamespace(project=Project()), MB)
    check(link2.attach(Project()) is None and link2.well_selected("x") is not None,
          "map bus: unsaved project = no folder (local bus only)")


def run_gui(check, tmp):
    """MainWindow wiring under the Qt mock / real Qt."""
    from .gui.main_window import MainWindow
    d = os.path.join(tmp, "wc_maplink_gui")
    os.makedirs(d, exist_ok=True)
    p = _project(d)
    win = MainWindow(p.path)
    check(getattr(win, "maplink", None) is not None and win.maplink.enabled and win.maplink.bridge is not None,
          "map bus GUI: link attached when the project opens")
    MB = win.maplink.MB
    seis = MB.MapBus()
    br = MB.FileBridge(seis, win.maplink.bridge.folder, "seismic")
    win.select_well("Alpha-1")
    br.poll()
    check(seis.selection.get("wells") == ["Alpha-1"], "map bus GUI: select_well publishes")
    seis.selection.select("wells", ["Beta-1"], source="map")
    win.bus_tick()
    check(win.maplink.bus.selection.get("wells") == ["Beta-1"], "map bus GUI: bus_tick delivers other tools' events")
    win.on_top_edited("Beta-1", "Shale B", 1650.0)
    br.poll()
    check(seis.table.value("Beta-1", "Top Shale B") == 1640.0 and seis.last("data", "tops") is not None,
          "map bus GUI: top edit pushed to the WellTable (TVDSS)")
    seis.active.set("well_section", "Live Map", source="livemap", wells=["Beta-1", "Alpha-1"], xy=[])
    win.bus_tick()
    check(win.project.current_section == "Live Map" and win.project.section_wells() == ["Beta-1", "Alpha-1"],
          "map bus GUI: cross section drawn on the Live Map opens in the correlation window")
    win.dirty = False
    win.maplink.close()
    br.close()
