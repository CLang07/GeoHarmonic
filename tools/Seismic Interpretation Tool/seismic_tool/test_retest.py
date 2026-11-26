"""PLAN 43a-g regression tests: fixes from the second install test on Chris's PC (26 Nov 2026).

43a  Survey blending window: an attribute `show` (QComboBox) shadowed QWidget.show -> TypeError on open.
43b  Wheeler: a level at the last sample read one past the end (float32 rounding of n - 1.000001, numpy 1.26).
43c  Framework from the window: 1 fault block instead of 2 (a fault vertex exactly on a grid edge cut nothing).
43d  Live grid of FS6 off the F3 outline: now built on the survey's own rotated bin grid.
43e  Live Map arbitrary line: open path, ended by double-click / Enter / right-click, opens the section.
43f  Rule colours: any CSS / SVG name ('lime'), clear message for a bad one, legend rows escaped.
43g  Filters: hidden wells lose their name labels too.
run() = Qt-free (self-test), run_gui() = Qt mock or real Qt offscreen (GUI test)."""
import ast
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", ".."))

# QWidget / QDialog / QMainWindow methods an instance attribute must never hide
SHADOW = {"show", "close", "update", "resize", "move", "hide", "raise_", "setVisible", "size", "pos", "width",
          "height", "layout", "font", "palette", "style", "windowTitle", "accept", "reject", "exec", "exec_",
          "repaint", "window", "parent", "rect", "geometry", "isVisible", "setLayout", "setWindowTitle", "done",
          "open", "grab", "x", "y", "cursor", "setEnabled", "isEnabled", "sizeHint", "adjustSize", "deleteLater",
          "objectName", "setStyleSheet", "lower", "setParent", "setFocus", "activateWindow", "children", "event",
          "render", "scroll", "winId", "minimumSize", "maximumSize", "frameGeometry"}


def shadowing_attributes(roots):
    """[(file, class, attribute, line)] of `self.<Qt method name> = ...` in classes derived from Qt widgets (also
    through our own widget classes, e.g. LiveMapCanvas(Basemap))."""
    classes = []
    for root in roots:
        for d, _, files in os.walk(root):
            if ".venv" in d or "__pycache__" in d:
                continue
            for fn in files:
                if fn.endswith(".py"):
                    p = os.path.join(d, fn)
                    try:
                        tree = ast.parse(open(p, encoding="utf-8").read())
                    except (SyntaxError, OSError, UnicodeDecodeError):
                        continue
                    for c in ast.walk(tree):
                        if isinstance(c, ast.ClassDef):
                            bases = [b.attr if isinstance(b, ast.Attribute) else getattr(b, "id", "") for b in c.bases]
                            classes.append((p, c, bases))
    qt = {c.name for _, c, b in classes if any(str(x).startswith("Q") for x in b)}
    grew = True
    while grew:
        n = len(qt)
        qt |= {c.name for _, c, b in classes if any(x in qt for x in b)}
        grew = len(qt) > n
    hits = []
    for p, c, _ in classes:
        if c.name not in qt:
            continue
        for n in ast.walk(c):
            tg = n.targets if isinstance(n, ast.Assign) else ([n.target] if isinstance(n, (ast.AnnAssign, ast.AugAssign)) else [])
            for g in tg:
                for e in ast.walk(g):
                    if (isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id == "self"
                            and e.attr in SHADOW):
                        hits.append((os.path.relpath(p, TOOLS), c.name, e.attr, n.lineno))
    return hits


def _real_qt():
    try:
        import PySide6
    except ImportError:
        return False
    return "fakeqt" not in (getattr(PySide6, "__file__", "") or "").replace("\\", "/")


# ====================================================================== Qt-free part (self-test)
def run(check, tmp):
    os.makedirs(tmp, exist_ok=True)
    # ---- 43a: no widget attribute hides a Qt method (the survey-blend `show` combo)
    hits = shadowing_attributes([os.path.join(TOOLS, "Seismic Interpretation Tool", "seismic_tool"),
                                 os.path.join(TOOLS, "Well Correlation Tool", "well_correlation")])
    check(not hits, "43a: no dialog / window attribute shadows a QWidget / QDialog method"
          + (f" ({'; '.join(f'{f} {c}.{a} line {ln}' for f, c, a, ln in hits[:5])})" if hits else ""))
    src = open(os.path.join(HERE, "gui", "surveyblend_dialog.py"), encoding="utf-8").read()
    check("self.show_combo" in src and "self.show =" not in src, "43a: survey blending 'Show' combo renamed show_combo")

    # ---- 43b: linear-interpolation indices at / beyond the last sample
    from .lininterp import lin_index
    from .rgt.stack import wheeler
    from .rgt.solve import interp_rows
    for dt in (np.float32, np.float64):
        k, k1, w = lin_index(np.array([0.0, 298.9999999, 299.0, 299.5, -3.0, np.nan], dt), 300)
        check(k.max() <= 298 and k1.max() <= 299 and np.all((w >= 0) & (w <= 1)) and k[2] == 298 and w[2] == 1.0,
              f"43b: lin_index keeps k <= n - 2 and w in [0, 1] ({np.dtype(dt).name})")
    ns = 300
    for tdt in (np.float32, np.float64):
        times = (np.arange(ns) * 4.0).astype(tdt)
        tau = np.tile(times, (6, 1)).astype(tdt)
        seis = np.tile(np.arange(ns, dtype=tdt), (6, 1))
        try:
            amp, hiatus = wheeler(seis, tau, np.array([times[0], 598.0, times[-1]], tdt), times)
            ok = abs(float(amp[2, 0]) - (ns - 1)) < 1e-3 and abs(float(amp[1, 0]) - 149.5) < 1e-3
        except IndexError as e:
            ok, amp = False, str(e)
        check(ok, f"43b: Wheeler level exactly at the last sample interpolates (no IndexError; {np.dtype(tdt).name})")
        amp2, _ = wheeler(seis, tau, np.array([float(times[-1]) + 4.0], tdt), times)
        check(not np.isfinite(amp2).any(), "43b: a level beyond the last sample gives no amplitude (NaN)")
    D = np.tile(np.arange(ns, dtype=np.float32), (3, 1))
    v = interp_rows(D, np.array([0, 1, 2]), np.array([ns - 1, ns - 1 - 1e-7, ns + 5], np.float32))
    check(np.allclose(v, ns - 1), "43b: RGT interp_rows at the last sample / beyond is clamped, not out of bounds")
    from .basin.chrono import horizon_rgt
    med, per = horizon_rgt(tau, times, np.full(6, float(times[-1])))
    check(np.isfinite(per).all() and abs(med - float(times[-1])) < 1e-3, "43b: horizon RGT at the last sample")

    # ---- 43c: fault traces with vertices exactly on node rows / columns still split the framework grid
    from . import framework as FW
    geom = FW.Geometry(500000.0, 6000000.0, 50.0, 50.0, 12, 10)
    X, Y = geom.nodes()
    bad = []
    for fx in (500300.0, 500300.26, 500325.0, 500350.0):
        for y0, y1 in ((6000000.0, 6000500.0), (6000000.0 + 1e-9, 6000500.0 - 1e-9), (5999990.0, 6000510.0)):
            tr = np.array([[fx, y0], [fx + 0.01, 0.5 * (y0 + y1)], [fx + 0.02, y1]])
            tr = FW.extend_trace(tr, 150.0)
            nb = len(np.unique(FW.fault_blocks(X, Y, [(tr[:, 0], tr[:, 1])])))
            if nb != 2:
                bad.append((fx, y0, nb))
    check(not bad, f"43c: fault through nodes / along grid lines -> 2 blocks every time ({bad[:3]})")
    tr = np.array([[500300.0, 6000000.0], [500300.0 + 1e-7, 6000000.0 + 1e-7], [500300.0, 6000500.0]])
    e = FW.extend_trace(tr, 150.0)
    check(abs(e[0, 0] - 500300.0) < 1.0 and e[0, 1] < 5999860.0, "43c: trace extension follows the strike even "
          "with near-duplicate end points")

    # ---- 43d: Live grid on a rotated F3-like survey
    _run_livegrid(check, tmp)

    # ---- 43f: colours
    from . import livemap as LM
    ok = (LM.colour_hex("lime") == "#00ff00" and LM.colour_hex("Dark Orange") == "#ff8c00"
          and LM.colour_hex("#abc") == "#aabbcc" and LM.colour_hex("#00FF00") == "#00ff00"
          and LM.colour_hex("red") == "#e15759" and LM.hex_rgba("steelblue")[:3] == (70, 130, 180))
    check(ok, "43f: CSS / SVG colour names (lime, dark orange, steelblue) and #rgb / #rrggbb accepted")
    try:
        LM.colour_hex("li")
        msg = ""
    except ValueError as e:
        msg = str(e)
    check("'li'" in msg, f"43f: an unknown colour is named in the message ({msg})")


def _run_livegrid(check, tmp):
    from . import synth_f3 as S
    from . import maplayers as ML
    from . import livegrid as LG
    from .project import SeisProject
    path = S.make_project(os.path.join(tmp, "f3like"))
    p = SeisProject.open(path)
    vol = p.volume("F3 original")
    st = ML.LayerStack.load(p)
    lg = LG.LiveGridder(st, None, p, threaded=False)
    lg.add_source("FS6", "horizon", horizon="FS6", key="3D:F3 original")
    if lg.errors:
        check(False, f"43d: Live grid failed: {lg.errors}")
        return
    lid = lg.sources["FS6"]["lid"]
    g = st.grid(lid)
    check(g.kind == "ilxl" and g.vol.parent is vol, "43d: Live grid of a 3D horizon is built on the survey bin grid")
    X, Y = g.node_xy()
    h = np.asarray(p.horizons.get("FS6").data["3D:F3 original"], float)
    s = g.vol.sub[1]
    i, j = 3, 5
    il, xl = vol.inlines[i * s], vol.xlines[j * s]
    ex, ey = vol.xy(il, xl)
    check(abs(X[i, j] - ex) < 1e-6 and abs(Y[i, j] - ey) < 1e-6 and abs(g.z[i, j] - h[i * s, j * s]) < 1e-3,
          f"43d: grid node (IL {il}, XL {xl}) lands on its exact X/Y ({X[i, j]:.1f}, {Y[i, j]:.1f}) with the picked "
          f"time {g.z[i, j]:.1f} ms")
    ox, oy = vol.outline()
    from .maphub import point_in_polygon
    inside = point_in_polygon(X.ravel(), Y.ravel(), list(ox), list(oy))
    near_edge = np.zeros(X.shape, bool)
    near_edge[[0, -1], :] = True
    near_edge[:, [0, -1]] = True
    check(bool(np.all(inside | near_edge.ravel())) and float(X.min()) >= float(min(ox)) - 1.0
          and float(X.max()) <= float(max(ox)) + 1.0,
          f"43d: Live grid inside the survey outline (X {X.min():.0f}-{X.max():.0f}, outline "
          f"{min(ox):.0f}-{max(ox):.0f}; the PC grid started 6 km east and ran past the edge)")
    # the drawn image: pixel centre (j + 0.5, i + 0.5) under the render transform = node X/Y
    prim = [q for q in ML.render(st) if q[0] == "image" and q[1] == lid]
    m11, m12, m21, m22, dx, dy = prim[0][3] if prim else (0, 0, 0, 0, 0, 0)
    px = m11 * (j + 0.5) + m21 * (i + 0.5) + dx
    py = m12 * (j + 0.5) + m22 * (i + 0.5) + dy
    check(bool(prim) and abs(px - ex) < 0.01 and abs(py - ey) < 0.01, "43d: drawn grid pixel centred on its node "
          "(same transform as the seismic basemap's horizon map)")
    # saved + reloaded (what the Live Map draws after a restart / Export map uses)
    st.save()
    p.save()
    st2 = ML.LayerStack.load(SeisProject.open(path))
    g2 = st2.grid(lid)
    X2, Y2 = g2.node_xy()
    check(g2.kind == "ilxl" and g2.z.shape == g.z.shape and np.allclose(X2, X) and np.allclose(Y2, Y),
          "43d: the Live grid keeps its survey geometry when saved and reloaded")
    cont = g.contours([1000.0])
    cx = np.concatenate([c[0] for c in cont.values()]) if cont else np.zeros(0)
    check(len(cx) > 0 and cx.min() >= min(ox) - 30 and cx.max() <= max(ox) + 30, "43d: contours inside the survey")
    out = os.path.join(tmp, "f3like_map.png")
    from .livemap import export_map
    export_map(st, None, out, title="FS6 Live grid")
    check(os.path.getsize(out) > 5000, "43d: Export map with the Live grid")
    # fill: picked nodes kept, holes filled per fault block
    z = np.full((20, 30), np.nan)
    z[::4, ::5] = 100.0
    z[::4, 20::5] = 200.0
    blocks = np.zeros(z.shape, int)
    blocks[:, 18:] = 1
    f = LG.fill_holes(z, blocks)
    check(np.isfinite(f).all() and np.allclose(f[:, :18], 100.0) and np.allclose(f[:, 18:], 200.0),
          "43d: holes filled from the picks of their own fault block")
    lg.add_source("coarse", "horizon", horizon="FS6", key="3D:F3 original")
    gc = lg.compute("coarse", "coarse")
    check(gc.vol.sub[1] == lg.coarse_factor * s and gc.z.size < g.z.size, "43d: coarse Live grid = every 4th node")


# ====================================================================== GUI part
def run_gui(check, tmp):
    os.makedirs(tmp, exist_ok=True)
    from PySide6 import QtWidgets
    from .gui.main_window import MainWindow
    from .test_maplayers import _setup
    real = _real_qt()
    app = QtWidgets.QApplication.instance()
    proj, truth, d = _setup(tmp, "retest_gui")
    v = proj.volumes[0]
    proj.volumes.append(dict(v, name="Synth B"))
    h = proj.horizons.get("Top")
    g = np.asarray(h.data["3D:Synth"], float)
    h.data["3D:Synth B"] = (g + 5.0).astype(np.float32)
    vol = proj.volume("Synth")
    f = proj.faults.new("F1")
    for il in (vol.inlines[0], vol.inlines[-1]):
        a = vol.xy(float(il), float(vol.xlines[12]))
        f.add_stick(np.array([[a[0], a[1], 100.0], [a[0] + 1.0, a[1], 900.0]]))
    proj.save()
    mw = MainWindow(proj.path)

    # ---- 43a: every non-modal dialog / window opens, shows and closes (real Qt: the survey-blend TypeError)
    openers = ["survey_blend", "conditioning", "inversion", "velocity_model", "velocity_volume", "framework",
               "basin_model", "prognosis", "mistie_analysis", "fault_mapping", "horizon_maps", "analytics",
               "dashboard", "colorbar_editor", "scan_dialog", "live_map"]
    failed, shown = [], 0
    for name in openers:
        try:
            dlg = getattr(mw, name)()
            if dlg is None:
                continue
            bad = sorted(a for a in SHADOW if a in vars(dlg))
            if bad:
                failed.append(f"{name}: shadows {bad}")
            dlg.show()
            if app is not None and real:
                app.processEvents()
            dlg.close()
            shown += 1
        except Exception as e:  # noqa: BLE001
            failed.append(f"{name}: {type(e).__name__}: {e}")
    check(not failed and shown >= 10, f"43a GUI: {shown} dialogs / windows opened, shown and closed"
          + (f" - failed: {'; '.join(failed)}" if failed else "") + (" (real Qt)" if real else " (Qt mock)"))
    from .gui.glow import GlowSettingsDialog
    from .glow import GlowSettings
    try:
        gd = GlowSettingsDialog(GlowSettings(), mw)
        gd.show()
        gd.close()
        ok = "width" not in vars(gd)
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"glow dialog: {e}")
    check(ok, "43a GUI: Glow settings dialog opens (its width spin no longer hides QWidget.width)")

    # ---- 43c: framework from the window with the fault at the survey edge -> 2 blocks
    dlg = mw.framework("Synth")
    m = dlg.build()
    check(m.nblocks == 2, f"43c GUI: framework from the window with the fault as a barrier ({m.nblocks} blocks)")
    f2 = mw.project.faults.new("F2 on nodes")
    for il in (vol.inlines[0], vol.inlines[-1]):  # stick tops exactly on bin centres = framework node rows
        a = vol.xy(float(il), float(vol.xlines[8]))
        f2.add_stick(np.array([[round(a[0]), round(a[1]), 0.0], [round(a[0]), round(a[1]), 1000.0]]))
    dlg2 = mw.framework("Synth")
    dlg2.faults = ["F2 on nodes"]
    m2 = dlg2.build()
    check(m2.nblocks == 2, f"43c GUI: a vertical fault exactly on grid nodes also gives 2 blocks ({m2.nblocks})")
    mw.project.faults.items.remove(f2)

    # ---- 43e: arbitrary line on the Live Map
    lm = mw.live_map()
    opened = []
    lm.open_arbitrary = lambda name, xs, ys: opened.append((name, list(xs), list(ys))) or name
    xs, ys = vol.outline()
    pts = [(float(xs[0]) + 20, float(ys[0]) + 20), (float(np.mean(xs)), float(np.mean(ys))),
           (float(xs[2]) - 20, float(ys[2]) - 20)]
    lm.start_arbitrary()
    check(lm.canvas.drawing() and lm.canvas._poly_closed is False, "43e GUI: arbitrary line is drawn as an open path")
    for k, (x, y) in enumerate(pts):
        lm.canvas.poly_click(x, y, "left", double=(k == len(pts) - 1))
    check(len(opened) == 1 and len(opened[0][1]) == 3 and (opened[0][1][0], opened[0][2][0]) != (opened[0][1][-1], opened[0][2][-1]),
          f"43e GUI: double-click ends the line (3 points, not closed) and opens the section ({len(opened)} opened)")
    sec = lm.canvas._section_item is not None or mw.mapbus.active.info("section").get("kind2") == "arbitrary"
    check(sec and not lm.canvas.drawing(), "43e GUI: the path stays on the map as the section line")
    lm.start_arbitrary()
    for x, y in pts[:2]:
        lm.canvas.poly_click(x, y)
    lm.finish_drawing()  # Enter
    check(len(opened) == 2 and len(opened[1][1]) == 2, "43e GUI: Enter ends the line and opens the section")
    lm.start_arbitrary()
    for x, y in pts[:2]:
        lm.canvas.poly_click(x, y)
    lm.canvas.poly_click(0, 0, "right")
    check(len(opened) == 3, "43e GUI: right-click ends the line too")
    n_poly = sum(1 for l in lm.canvas.stack.layers if l["kind"] == "polygons")
    check(n_poly == 0, "43e GUI: arbitrary lines are not stored as closed polygon layers")
    lm.start_arbitrary()
    lm.canvas.poly_click(*pts[0])
    lm.cancel_drawing()
    check(not lm.canvas.drawing() and len(opened) == 3, "43e GUI: Esc cancels the line")
    lm.canvas.start_polygon()
    lm.canvas.poly_click(*pts[0])
    lm.canvas.poly_click(*pts[0], double=True)  # double-click delivers the same point twice: dropped
    check(lm.canvas._poly is None, "43e GUI: repeated double-click point ignored, short shape discarded")
    real_dlg = MainWindow.open_arbitrary(mw, "Arb", [p_[0] for p_ in pts], [p_[1] for p_ in pts])
    check(real_dlg is not None, "43e GUI: the seismic window opens the arbitrary section from the 3-point line")

    # ---- 43f: rule colours + legend
    t = lm.bus.table
    xw = t.numeric("X")
    cut = float(np.round(np.median(xw))) + 1.0
    try:
        rules = lm.parse_rules(f"X <= {cut:.0f} = lime; X > {cut:.0f} = steelblue")
        st = lm.set_color("rules", rules=rules)
        k0 = int(np.argmin(xw))
        ok = tuple(st["rgba"][k0][:3]) == (0, 255, 0)
    except Exception as e:  # noqa: BLE001
        ok, rules = False, str(e)
    check(ok, f"43f GUI: rule 'X <= ... = lime' colours the wells ({rules})")
    html = lm.legend_html()
    check(f"X &lt;= {cut:.0f}" in html and f"X &gt; {cut:.0f}" in html and "other" in html,
          "43f GUI: legend lists each rule by its own text (escaped) and then 'other'")
    before = dict(lm.state.well_style["color"])
    if real:
        lm.rules_edit.setText("X <= 1 = lmie")
        lm.c_mode.setCurrentText("rules")
    else:  # Qt mock: widget values are not stored
        lm.rules_edit.text = lambda: "X <= 1 = lmie"
        lm.c_mode.currentText = lambda: "rules"
    msgs = []
    old_say = lm._say
    lm._say = lambda m: msgs.append(m)
    lm.apply_style_widgets()
    lm._say = old_say
    check(any("'lmie'" in m for m in msgs) and lm.state.well_style["color"].get("rules") == before.get("rules"),
          f"43f GUI: a bad colour is named and the style is kept ({msgs[-1] if msgs else ''})")

    # ---- 43g: filtered wells lose their labels
    lm.set_color("fixed")
    lm.canvas.view_override = (float(xw.min()) - 1000, float(xw.max()) + 1000, float(t.numeric("Y").min()) - 1000,
                               float(t.numeric("Y").max()) + 1000)
    lm.restyle()
    all_lab = lm.canvas.label_ids()
    lm.add_filter("X")
    lm.set_range_filter("X", None, cut)
    lab = lm.canvas.label_ids()
    hidden = [i for i, vis in zip(lm.styled["ids"], lm.styled["visible"]) if not vis]
    check(len(all_lab) == 3 and hidden and not set(hidden) & set(lab) and len(lab) == 3 - len(hidden),
          f"43g GUI: wells hidden by a filter lose their labels ({len(lab)} of {len(all_lab)} labelled)")
    from .livemap import styled_primitives
    texts = [q[4] for q in styled_primitives(lm.styled) if q[0] == "text"]
    check(not set(hidden) & set(texts), "43g GUI: no labels of filtered wells in Export map")
    check(len(lm.canvas.update_labels()) == len(lab), "43g GUI: labels refreshed on zoom / pan follow the filter")
    lm.clear_filters()
    try:
        mw.maplink.close()
    except Exception:
        pass
