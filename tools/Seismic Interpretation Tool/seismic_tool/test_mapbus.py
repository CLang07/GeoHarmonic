"""Tests for the Live Map bus (PLAN 31a): services, rules, WellTable files, cross-process FileBridge."""
import json
import os
import subprocess
import sys

import numpy as np

from . import mapbus as MB


def _table():
    t = MB.WellTable()
    MB.table_from_wells([("A-1", 0.0, 0.0), ("B-2", 100.0, 50.0), ("C-3", 200.0, 80.0), ("D 4", 300.0, 10.0)], t)
    t.set_column("Mistie_ms", {"A-1": 3.0, "B-2": 12.5, "C-3": 25.0}, owner="mistie", unit="ms")
    t.set_column("Status", {"A-1": "Oil", "B-2": "Dry", "C-3": "Gas", "D 4": "Oil"}, owner="wellcorr")
    t.set_column("Net pay", {"A-1": 10, "B-2": 0, "C-3": 22.5, "D 4": 4}, owner="petro", unit="m")
    return t


def run(check, tmp):
    folder = os.path.join(tmp, "mapbus")
    os.makedirs(folder, exist_ok=True)

    # ---- pub/sub: no echo, topic filter, broken subscriber isolated
    bus = MB.MapBus()
    got = {"map": [], "section": [], "data": []}
    bus.subscribe("map", lambda e: got["map"].append(e))
    bus.subscribe("section", lambda e: got["section"].append(e))
    bus.subscribe("dataonly", lambda e: got["data"].append(e), topics="data")
    bus.subscribe("broken", lambda e: 1 / 0)
    bus.selection.select("wells", ["A-1", "B-2"], source="map")
    check(not got["map"] and len(got["section"]) == 1 and got["section"][0]["ids"] == ["A-1", "B-2"]
          and not got["data"], "map bus: selection delivered to others only, never echoed to the sender")
    check(bus.errors and bus.errors[-1][0] == "broken", "map bus: a broken subscriber is recorded, others still served")
    n = len(got["section"])
    r = bus.selection.select("wells", ["b-2", "a 1"], source="map")  # same set, other spelling
    check(len(got["section"]) == n and r == ["A-1", "B-2"], "map bus: unchanged selection publishes nothing")
    bus.selection.select("wells", ["C-3"], mode="add", source="map")
    bus.selection.select("wells", ["A-1", "D 4"], mode="toggle", source="map")
    check(bus.selection.get("wells") == ["B-2", "C-3", "D 4"], "map bus: add / toggle selection modes")
    bus.selection.select("wells", ["c-3"], mode="remove", source="map")
    check(bus.selection.get("wells") == ["B-2", "D 4"] and bus.selection.contains("wells", "d-4"),
          "map bus: remove + case-insensitive membership")
    bus.selection.select("lines", ["R-2"], source="section")
    check(bus.selection.get("lines") == ["R-2"] and bus.selection.get("wells") == ["B-2", "D 4"],
          "map bus: selection kinds are independent")

    # ---- active object
    e = bus.active.set("inline", 120, source="section", volume="F3")
    check(e is not None and bus.active.get("inline") == 120 and bus.active.info("inline")["volume"] == "F3",
          "map bus: active inline with extras")
    check(bus.active.set("inline", 120, source="map", volume="F3") is None, "map bus: same active value not re-sent")
    check(bus.active.set("inline", 120, source="map", volume="F3", force=True) is not None,
          "map bus: force re-sends the active object")
    bus.active.set("section", "IL 120", source="section", xy=[[0, 0], [10, 5]])
    check(bus.active.info("section")["xy"] == [[0, 0], [10, 5]], "map bus: section line geometry kept")

    # ---- DataChanged
    bus.notify("tops", ids=["A-1"], source="wellcorr", formation="F0")
    check(got["data"] and got["data"][-1]["kind"] == "tops" and got["data"][-1]["formation"] == "F0"
          and bus.last("data", "tops")["ids"] == ["A-1"], "map bus: DataChanged event with info and last()")

    # ---- WellTable
    t = _table()
    check(len(t) == 4 and t.index("d-4") == 3 and np.isnan(t.value("D 4", "Mistie_ms")),
          "well table: ids, normalised lookup, missing = NaN")
    check(t.meta["Status"]["kind"] == "text" and t.meta["Net pay"]["kind"] == "numeric", "well table: column kinds")
    t.set_column("New", {"E-5": 1.0}, owner="x")
    check(len(t) == 5 and np.isnan(t.value("E-5", "Net pay")) and t.text("Status")[4] == "",
          "well table: a column with a new well extends every column")
    t.remove_column("New")
    check(t.categories("Status") == ["Oil", "Dry", "Gas", ""], "well table: categories in first-seen order")
    p = os.path.join(folder, "welltable.csv")
    t.save(p)
    t2 = MB.WellTable.load(p)
    check(t2.ids == t.ids and np.allclose(t2.numeric("Net pay"), t.numeric("Net pay"), equal_nan=True)
          and t2.meta["Mistie_ms"]["owner"] == "mistie" and t2.meta["Mistie_ms"]["unit"] == "ms",
          "well table: CSV + meta round trip")
    # two tools write their own columns: no overwrite
    a = MB.WellTable.load(p)
    a.set_column("Mistie_ms", {"A-1": 99.0}, owner="mistie")
    b = MB.WellTable.load(p)
    b.set_column("Net pay", {"A-1": 7.0, "B-2": 1.0, "C-3": 2.0, "D 4": 3.0}, owner="petro")
    b.set_column("Phi", {"A-1": 0.2}, owner="petro")
    a.save(p, owner="mistie")
    b.save(p, owner="petro")
    m = MB.WellTable.load(p)
    check(m.value("A-1", "Mistie_ms") == 99.0 and m.value("A-1", "Net pay") == 7.0 and m.value("A-1", "Phi") == 0.2,
          "well table: read-merge-write keeps both tools' columns")
    b.remove_column("Phi")
    b.save(p, owner="petro")
    check("Phi" not in MB.WellTable.load(p).cols and "Mistie_ms" in b.cols,
          "well table: an owner's removed column disappears; merged view adopted")
    tops = {"A-1": {"F0": 2500.0, "C-sands": 2100.0}, "B-2": {"F0": 2550.0}}
    cols = MB.tops_columns(t, tops)
    check(cols == ["Top F0", "Top C-sands"] and np.isnan(t.value("B-2", "Top C-sands"))
          and t.value("A-1", "Top F0") == 2500.0, "well table: tops -> one column per formation")

    # ---- rules + filters
    t = _table()
    r = MB.eval_rule("Mistie_ms > 10", t)
    check(r.tolist() == [False, True, True, False], "rule: numeric comparison, NaN is False")
    r = MB.eval_rule("[Net pay] >= 4 and Status in ('Oil', 'Gas')", t)
    check(r.tolist() == [True, False, True, True], "rule: bracketed name, and, in (...)")
    r = MB.eval_rule("not (Status == 'Dry') and (Mistie_ms < 20 or isnan(Mistie_ms))", t)
    check(r.tolist() == [True, False, False, True], "rule: not / or / isnan")
    r = MB.eval_rule("0 < X < 250 and contains(Status, 'a')", t)
    check(r.tolist() == [False, False, True, False], "rule: chained comparison + contains()")
    r = MB.eval_rule("[Net pay] / 2 > Mistie_ms - 10", t)
    check(r.tolist() == [True, False, False, False], "rule: arithmetic")
    bad = 0
    for ex in ("__import__('os')", "X.real > 1", "open('x')", "[Nope] > 1"):
        try:
            MB.eval_rule(ex, t)
        except (ValueError, KeyError, SyntaxError):
            bad += 1
    check(bad == 4, "rule: calls / attributes / unknown columns rejected")
    bus2 = MB.MapBus()
    seen = []
    bus2.subscribe("table", lambda e: seen.append(e), topics="filter")
    bus2.filters.set_range("Net pay", 4, None, source="map")
    check(bus2.filters.visible_ids(t) == ["A-1", "C-3", "D 4"] and seen and seen[-1]["filters"]["Net pay"]["min"] == 4,
          "filters: range filter + event with the full filter dict")
    bus2.filters.set_values("Status", ["Oil"], source="map")
    check(bus2.filters.visible_ids(t) == ["A-1", "D 4"], "filters: category filter combines with range")
    check(bus2.filters.mask(t, exclude="Status").tolist() == [True, False, True, True],
          "filters: exclude one column (histogram of that column)")
    bus2.filters.set_rule("big mistie", "Mistie_ms > 10", source="map")
    check(bus2.filters.visible_ids(t) == [], "filters: rule filter")
    try:
        bus2.filters.set_rule("bad", "import os", source="map")
        check(False, "filters: bad rule rejected at set time")
    except (ValueError, SyntaxError):
        check("bad" not in bus2.filters.filters, "filters: bad rule rejected at set time")
    bus2.filters.remove("big mistie", source="map")
    bus2.filters.clear(source="map")
    check(bus2.filters.filters == {} and len(bus2.filters.visible_ids(t)) == 4, "filters: remove + clear")

    # ---- bus folder
    f1 = MB.bus_folder(None, os.path.join(tmp, "Study.wcproj.json"))
    f2 = MB.bus_folder(os.path.join(tmp, "Seis.seisproj.json"), None)
    check(f1.endswith("Study_mapbus") and f2.endswith(os.path.join("Seis_seismic", "mapbus"))
          and os.path.isdir(f1) and MB.bus_folder() is None, "bus folder: next to the well project, else seismic")

    # ---- FileBridge (same process, two buses = two tools)
    bf = os.path.join(tmp, "bridge")
    s1, s2 = MB.MapBus(), MB.MapBus()
    b1, b2 = MB.FileBridge(s1, bf, "seismic"), MB.FileBridge(s2, bf, "wellcorr")
    rec = []
    s2.subscribe("wc_view", lambda e: rec.append(e))
    s1.selection.select("wells", ["A-1"], source="map")
    s1.active.set("inline", 77, source="section", volume="F3")
    check(b2.poll() and s2.selection.get("wells") == ["A-1"] and s2.active.get("inline") == 77
          and [e["topic"] for e in rec] == ["selection", "active"], "bridge: events reach the other tool")
    check(b1.poll() == [], "bridge: a tool never reads its own events back")
    s2.selection.select("wells", ["B-2"], source="wc_view")  # reply from tool 2
    n_written = b2.written
    b1.poll()
    check(s1.selection.get("wells") == ["B-2"] and b2.written == n_written and b1.written == 2,
          "bridge: incoming events are not re-written (no ping-pong)")
    st = json.load(open(os.path.join(bf, "state.json")))
    check(st["selection"]["wells"] == ["B-2"] and st["active"]["inline"]["value"] == 77, "bridge: state.json snapshot")
    s3 = MB.MapBus()
    late = []
    s3.subscribe("late", lambda e: late.append(e))
    b3 = MB.FileBridge(s3, bf, "mistie")
    check(s3.selection.get("wells") == ["B-2"] and s3.active.info("inline")["volume"] == "F3" and late
          and b3.poll() == [], "bridge: a late tool catches up from state.json, no journal replay")
    # partial line (being written) is not consumed
    with open(b1.events_path, "a") as f:
        f.write('{"topic": "data", "kind": "tops"')
    check(b2.poll() == [], "bridge: a half-written line waits")
    with open(b1.events_path, "a") as f:
        f.write(', "source": "x", "origin": "zzz", "seq": 1, "ids": ["C-3"]}\n')
    ev = b2.poll()
    check(len(ev) == 1 and ev[0]["kind"] == "tops", "bridge: line completed -> delivered")
    # journal restart
    b1.max_bytes = 10
    s1.notify("grids", source="map", name="F0 depth")
    s1.notify("grids", source="map", name="F1 depth")
    ev = b2.poll()
    check(ev and ev[-1].get("name") == "F1 depth" and os.path.getsize(b1.events_path) < 400,
          "bridge: journal restarts above max_bytes, readers follow")
    # well table through the bridge
    s1.table.set_column("Mistie_ms", {"A-1": 4.0}, owner="mistie")
    b1.save_table("mistie")
    b2.poll()
    check(s2.table.value("A-1", "Mistie_ms") == 4.0, "bridge: WellTable change reloaded by the other tool")
    s2.table.set_column("Top F0", {"A-1": 2500.0}, owner="wellcorr")
    b2.save_table("wellcorr")
    b1.poll()
    check(s1.table.value("A-1", "Top F0") == 2500.0 and s1.table.value("A-1", "Mistie_ms") == 4.0,
          "bridge: both tools' columns survive")
    # numpy payloads serialise
    s1.notify("picks", source="section", ids=["H1"], xy=np.arange(4.0).reshape(2, 2), n=np.int64(3))
    ev = b2.poll()
    check(ev and ev[-1]["xy"] == [[0.0, 1.0], [2.0, 3.0]] and ev[-1]["n"] == 3, "bridge: numpy payloads as JSON")
    for b in (b1, b2, b3):
        b.close()

    # ---- real second process
    child = os.path.join(tmp, "child.py")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(child, "w") as f:
        f.write(f"""import sys, time
sys.path.insert(0, {root!r})
from seismic_tool import mapbus as MB
bus = MB.MapBus(); br = MB.FileBridge(bus, {bf!r}, "child")
bus.subscribe("echo", lambda e: bus.active.set("well", e["ids"][0], source="echo") if e["topic"] == "selection" else None)
t0 = time.time()
while time.time() - t0 < 20:
    br.poll()
    if bus.active.get("well"):
        break
    time.sleep(0.02)
""")
    s4 = MB.MapBus()
    b4 = MB.FileBridge(s4, bf, "parent")
    proc = subprocess.Popen([sys.executable, child])
    import time as _t
    _t.sleep(1.0)
    s4.selection.select("wells", ["Z-9"], source="map")
    t0 = _t.time()
    while _t.time() - t0 < 20 and s4.active.get("well") != "Z-9":
        b4.poll()
        _t.sleep(0.02)
    proc.wait(timeout=30)
    check(s4.active.get("well") == "Z-9", "bridge: round trip with a second process (selection -> active well)")
    b4.close()


def run_gui(check, tmp):
    """Seismic MainWindow <-> map bus wiring (PLAN 31a) under the Qt mock or real Qt."""
    from .gui.main_window import MainWindow
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, "mapbus_gui")
    mw = MainWindow(proj.path)
    link = mw.maplink
    check(link.bridge is not None and link.bridge.folder == MB.bus_folder(None, proj.well_project),
          "31a GUI: map bus attached next to the linked well project")
    check(sorted(link.bus.table.ids) == ["W-1", "W-2", "W-3"] and "X" in link.bus.table.cols,
          "31a GUI: WellTable seeded with the project wells")
    other = MB.MapBus()
    br = MB.FileBridge(other, link.bridge.folder, "wellcorr")
    vol = mw.project.volume("Synth")
    il = int(vol.inlines[3])
    sec = mw.section
    sec.source, sec.mode, sec.number, sec.domain, sec.attr = ("3D", "Synth"), "Inline", il, "Time", "Seismic"
    sec.redraw()
    link.position_changed()  # positionChanged is a mock signal here
    br.poll()
    check(other.active.get("inline") == il and other.active.info("inline")["volume"] == "Synth"
          and len(other.active.info("section").get("xy") or []) >= 2, "31a GUI: current inline + its map trace published")
    mw.basemap.set_selected_wells(["W-2"], emit=False)
    link.wells_selected(["W-2"])
    br.poll()
    check(other.selection.get("wells") == ["W-2"], "31a GUI: basemap well selection published")
    # follow another tool: active 2D line opens in the section, selected wells ringed on the basemap
    shown = []
    real_show = sec.show

    def show(kind, name, mode=None, number=None):  # the Qt mock's source combo has no items: set the state here
        shown.append((kind, name, mode, number))
        real_show(kind, name, mode, number)
        if tuple(sec.source or ()) != (kind, name) or (kind == "3D" and (sec.mode != mode or sec.number != number)):
            sec.source, sec.mode, sec.number = (kind, name), (mode or "2D"), number
    sec.show = show
    line = mw.project.lines[0]["name"] if mw.project.lines else None
    other.selection.select("wells", ["W-3"], source="wc")
    if line:
        other.active.set("line", line, source="wc")
    mw.bus_tick()
    check(mw.basemap.selected_wells == ["W-3"], "31a GUI: selection from another tool shown on the basemap")
    check(line is None or tuple(sec.source) == ("2D", line), "31a GUI: active 2D line from another tool opened")
    # other.active inline -> section follows
    other.active.set("inline", int(vol.inlines[5]), source="wc", volume="Synth")
    mw.bus_tick()
    check(tuple(sec.source) == ("3D", "Synth") and sec.mode == "Inline" and int(sec.number) == int(vol.inlines[5]),
          "31a GUI: active inline from another tool shown")
    n = len(other.history)
    mw._horizons_changed()
    mw._faults_changed()
    br.poll()
    kinds = [e.get("kind") for e in other.history[n:] if e["topic"] == "data"]
    check("picks" in kinds and "faults" in kinds, "31a GUI: horizon / fault edits -> DataChanged")
    other.notify("tops", ids=["W-1"], source="wc")
    mw.ddc_auto = True
    mw.bus_tick()
    check(any("Tops changed" in m for m in link.messages), "31a GUI: tops change from the Well Correlation Tool noticed")
    mw.ddc_auto = False
    link.depth_model_changed({"message": "rebuilt layers 2-3", "layers": [2, 3], "changes": ["W-1 Top"]})
    link.misties_changed(["L1"], {"L1": {"shift": 2.0, "phase": 5.0, "gain": 1.0, "n_ties": 2}})
    br.poll()
    ev = other.last("data", "misties")
    check(other.last("data", "velocities")["layers"] == [2, 3] and ev["solution"]["L1"]["shift"] == 2.0,
          "31a GUI: depth-model and mistie updates published")
    mw.maplink.close()
    br.close()
