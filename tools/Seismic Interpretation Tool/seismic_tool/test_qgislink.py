"""Tests for the QGIS link (PLAN 21h): shared GeoPackage, user edits, message protocol between two processes, plugin
logic under a fake qgis module."""
import os
import socket
import subprocess
import sys
import textwrap

import numpy as np

PLUGIN_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "QGIS Plugin"))
FAKE_QGIS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "dev", "fakeqgis"))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _project(tmp, name):
    from . import maplayers as ML
    from .gis.export import set_project_crs
    from .maps import horizon_grid
    from .test_maplayers import _setup
    proj, truth, d = _setup(tmp, name)
    import json
    with open(proj.well_project) as fh:
        wj = json.load(fh)
    wj["tops"] = {"W-1": {"A": 2000.0, "B": 2100.0}, "W-2": {"A": 2010.0}}
    with open(proj.well_project, "w") as fh:
        json.dump(wj, fh)
    set_project_crs(proj, 2193)
    st = ML.LayerStack.load(proj)
    st.add_grid(horizon_grid(proj, proj.horizons.get("Top"), "3D:Synth"), "Top time", interval=10.0)
    st.add("polygons", "Prospects", src={"polygons": [{"name": "P1", "x": [500100, 500300, 500300], "y": [6000100, 6000100, 6000300]}]})
    st.add("petro", "SoPhiH A", src={"property": "SoPhiH", "zone": "A"})
    proj.save()
    return proj, st, d


class _App:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a: self.calls.append((name,) + a)


def run(check, tmp):
    from . import qgislink as QL
    from .gis import gpkg as GP, geotiff as GT
    proj, st, d = _project(tmp, "qgis")
    out = QL.write_shared(proj, st)
    gp = out["gpkg"]
    lays = set(GP.list_layers(gp))
    check({"wells", "tops", "survey_3d", "prospects", "prospects_edit", "leases", "section_lines"} <= lays,
          f"21h: shared GeoPackage layers {sorted(lays)}")
    g, a, wkt = GP.read_layer(gp, "wells")
    check(len(g) == 3 and "SoPhiH" in a and a["status"][0] and "Transverse_Mercator" in (wkt or ""),
          "21h: wells with status + petro summary columns, CRS stored")
    import sqlite3
    con = sqlite3.connect(gp)
    nstyle = con.execute("SELECT count(*) FROM layer_styles WHERE useAsDefault = 1").fetchone()[0]
    gtype = con.execute("SELECT geometry_type_name FROM gpkg_geometry_columns WHERE table_name = 'prospects_edit'").fetchone()[0]
    rasters = list(con.execute("SELECT path FROM gh_rasters"))
    con.close()
    check(nstyle >= len(lays) - 1 and gtype == "POLYGON", "21h: default QGIS styles in layer_styles; editable tables typed")
    r = GT.read(rasters[0][0])
    check(r["overviews"] >= 1 and r["epsg"] == 2193, "21h: grids as GeoTIFF with overviews next to the GeoPackage")
    # QGIS user edits: a prospect digitised into prospects_edit, a section line
    QL.read_user_edits(proj)  # baseline
    changed, h = QL.read_user_edits(proj)
    GP.write_layer(gp, "prospects_edit", [{"type": "Polygon", "parts": [np.array([[500200, 6000200], [500250, 6000200],
                                                                                 [500250, 6000250], [500200, 6000200]])]}],
                   {"name": ["QP1"]}, 2193, "x", geometry_type="Polygon")
    GP.write_layer(gp, "section_lines", [{"type": "LineString", "parts": [np.array([[500050, 6000100], [500550, 6000400]])]}],
                   {"name": ["L1"]}, 2193, "x", geometry_type="LineString")
    changed, h2 = QL.read_user_edits(proj, h)
    res = QL.apply_user_edits(st, changed)
    lay = next(l for l in st.layers if l["name"] == "Prospects (QGIS)")
    check(set(changed) == {"prospects_edit", "section_lines"} and lay["src"]["polygons"][0]["name"] == "QP1" and
          res["section_lines"][0][0] == "L1", "21h: QGIS edits detected and applied (prospect layer, section line)")
    changed3, _ = QL.read_user_edits(proj, h2)
    check(changed3 == {}, "21h: unchanged tables are not re-applied")
    QL.write_shared(proj, st)
    g, a, _ = GP.read_layer(gp, "prospects_edit")
    check(a["name"] == ["QP1"], "21h: re-writing the shared GeoPackage keeps the user's edits")
    sys.path.insert(0, PLUGIN_DIR)
    try:
        from geoharmonic_link import core
        info = core.read_project(gp)
        check("wells" in info["vector"] and info["editable"] and info["rasters"] and info["meta"]["crs_epsg"] == 2193,
              "21h: plugin reads the shared GeoPackage (layers, editable tables, grids, CRS)")
        bad = core.LinkClient(free_port())
        check(not bad.connect(timeout=0.5) and "not listening" in bad.error, "21h: clean failure when GeoHarmonic is not running")
    finally:
        sys.path.remove(PLUGIN_DIR)
    # message protocol between two processes
    port = free_port()
    ep = QL.LinkEndpoint("server", port)
    check(ep.start(), "21h: GeoHarmonic link server listening on localhost")
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {PLUGIN_DIR!r})
        from geoharmonic_link import core
        c = core.LinkClient({port})
        assert c.connect(5.0), c.error
        c.wait("hello", 5)
        c.send({{"type": "select_wells", "wells": ["W-1", "W-2"]}})
        c.send(core.polygon_message("QGIS prospect", [(0, 0), (10, 0), (10, 10)]))
        c.send(core.line_message("XL", [(1, 2), (3, 4)]))
        m = c.wait("current_section", 10)
        c.send({{"type": "cursor", "x": m["xy"][0][0], "y": m["xy"][0][1]}})
        c.wait("bye", 10)
        c.close()
    """)
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        app = _App()
        got = []
        import time
        t0 = time.time()
        while len(got) < 4 and time.time() - t0 < 15:
            for m in ep.poll():
                got.append(m["type"])
                QL.handle(m, app)
            time.sleep(0.05)
        names = [c[0] for c in app.calls]
        check(names[:3] == ["select_wells", "polygon", "section_line"] and app.calls[0][1] == ["W-1", "W-2"] and
              np.allclose(app.calls[2][2], [1, 3]), f"21h: QGIS -> GeoHarmonic messages handled ({names})")
        ep.send({"type": "current_section", "xy": [[123.0, 456.0], [130.0, 460.0]]})
        cur = ep.wait("cursor", 10)
        check(cur is not None and cur["x"] == 123.0, "21h: GeoHarmonic -> QGIS -> GeoHarmonic round trip (section cursor)")
        ep.send({"type": "bye"})
        proc.wait(timeout=15)
        check(proc.returncode == 0, "21h: QGIS-side process finished cleanly" + ("" if proc.returncode == 0 else
                                                                                f": {proc.stderr.read().decode()[-300:]}"))
    finally:
        if proc.poll() is None:
            proc.kill()
        ep.stop()


def run_gui(check, tmp):
    from .gui.main_window import MainWindow
    from . import qgislink as QL
    proj, st, d = _project(tmp, "qgis_gui")
    for p in (FAKE_QGIS, PLUGIN_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    fake_qt = os.path.join(os.path.dirname(FAKE_QGIS), "fakeqt")  # the QGIS mock needs its 'anything' helper;
    if fake_qt not in sys.path:  # appended last so it can never shadow the real PySide6 / pyqtgraph
        sys.path.append(fake_qt)
    try:
        from geoharmonic_link.plugin import GeoHarmonicLinkPlugin
        from anything import Anything
        mw = MainWindow(proj.path)
        mw.basemap.stack = st
        out = mw.send_map_to_qgis()
        pl = GeoHarmonicLinkPlugin(Anything())
        pl.initGui()
        info = pl.open_project(out["gpkg"])
        check("wells" in pl.layers and len(pl.actions) == 6 and info["rasters"], "21h GUI: QGIS plugin (fake qgis) opens the "
              "shared GeoPackage: layers grouped, grids added, 6 actions")
        check(pl.handle({"type": "select_wells", "wells": ["W-1"]}).startswith("selected") and
              pl.handle({"type": "current_section", "xy": [[1, 2], [3, 4]]}).startswith("section"),
              "21h GUI: plugin handles selection + section-cursor messages")
        port = free_port()
        check(mw.toggle_qgis(True, port), "21h GUI: View > Live link with QGIS starts the server")
        from geoharmonic_link import core
        c = core.LinkClient(port)
        check(c.connect(), "21h GUI: plugin connects")
        import time
        c.send(core.polygon_message("From QGIS", [(500100, 6000100), (500200, 6000100), (500200, 6000200)]))
        c.send({"type": "select_wells", "wells": ["W-2"]})
        t0 = time.time()
        done = []
        while time.time() - t0 < 5 and len(done) < 3:
            done += mw.qgis_tick()
            time.sleep(0.05)
        lay = next((l for l in mw.basemap.stack.layers if l["name"] == "Prospects (QGIS)"), None)
        check(lay is not None and lay["src"]["polygons"][0]["name"] == "From QGIS" and mw.basemap.selected_wells == ["W-2"],
              f"21h GUI: polygon from QGIS -> prospect layer; QGIS selection shown on the basemap ({done})")
        mw.send_current_section((np.array([500000.0, 500600.0]), np.array([6000200.0, 6000200.0])))
        m = c.wait("current_section", 5)
        check(m is not None and len(m["xy"]) == 2, "21h GUI: current section line sent to QGIS")
        dlg = mw.open_arbitrary("L", [500050, 500550], [6000100, 6000400])
        check(dlg is not None and dlg.data["section"].shape[1] == len(proj.volume("Synth").times), "21h GUI: line -> arbitrary "
              "seismic section")
        c.close()
        mw.toggle_qgis(False)
        check(mw.qgis is None, "21h GUI: link stopped")
    finally:
        for p in (FAKE_QGIS, PLUGIN_DIR):
            if p in sys.path:
                sys.path.remove(p)
