#!/usr/bin/env python3
"""Build the illustrated GeoHarmonic tutorial (PLAN 38d/e) in one command.

    python3 dev/build_tutorial.py                 # -> docs/TUTORIAL.md, docs/TUTORIAL.html, docs/tutorial/*.png
    python  dev\\build_tutorial.py --pc           # Chris's PC (run by Install GeoHarmonic.bat): real Qt screenshots,
                                                  #    written to tutorial_local\\ (not tracked by git)
Options: --out DIR, --mock (force the Qt mock), --results FILE (test output to quote; repeatable),
         --run-tests (run the self-tests when no results file is found).

What it does
  1. builds a small synthetic demo project (3D cube, horizons Top / Deep, wells W-1..W-3) in a temp folder, and uses
     the bundled NZ well project (data/nz) for the Well Correlation chapter;
  2. captures the pictures from the program itself:
       real Qt (PC): every window is opened off screen and grabbed (QWidget.grab); numbered callouts are placed on
                     the window's own panels (menu bar, toolbars, docks, tabs, group boxes) and listed in a legend;
       no Qt (build server): the windows cannot be drawn - the caption says so - and the chapter shows the tool's
                     own image exports instead (software 3D renderer, Live Map PNG export, map export, section export);
  3. quotes the numbers the tests verify (PASS lines from test_results.txt / the given results files);
  4. writes TUTORIAL.md (images in tutorial/) and a self-contained TUTORIAL.html (images embedded).
Never draws mock-ups.  CLAUDE.md: re-run after any change to a window; add a chapter for every new module
(dev/tutorial_content.py).
"""
import argparse
import base64
import datetime
import html
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

DEV = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(DEV)
SEIS_DIR = os.path.join(REPO, "tools", "Seismic Interpretation Tool")
WC_DIR = os.path.join(REPO, "tools", "Well Correlation Tool")
NZ_PROJECT = os.path.join(REPO, "data", "nz", "NZ Well Correlation.wcproj.json")
MAX_W = 1400
for p in (DEV, SEIS_DIR, WC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


# ============================================================================================ Qt
def detect_qt(force_mock=False):
    """'real' when the real PySide6 imports (off screen), else put dev/fakeqt first and return 'mock'."""
    if not force_mock:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
        if sys.platform.startswith("win"):
            fd = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
            if os.path.isdir(fd):
                os.environ.setdefault("QT_QPA_FONTDIR", fd)
        try:
            import PySide6
            from PySide6 import QtWidgets  # noqa: F401
            if "fakeqt" not in (getattr(PySide6, "__file__", "") or "").replace("\\", "/"):
                return "real"
        except Exception:
            pass
    for k in [k for k in sys.modules if k == "PySide6" or k.startswith("PySide6.") or k == "pyqtgraph"
              or k.startswith("pyqtgraph.")]:
        del sys.modules[k]
    sys.path.insert(0, os.path.join(DEV, "fakeqt"))
    return "mock"


def pump(seconds=0.4):
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance()
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def patch_modals():
    """Modal dialogs open non-modally (so they can be grabbed); message boxes / file dialogs are auto-answered."""
    from PySide6 import QtGui, QtWidgets
    MB = QtWidgets.QMessageBox
    for kind in ("information", "warning", "critical", "about"):
        setattr(MB, kind, staticmethod(lambda *a, **k: MB.StandardButton.Ok))
    MB.question = staticmethod(lambda *a, **k: MB.StandardButton.No)

    def dexec(self, *a, **k):
        self.show()
        return 0
    for cls in (QtWidgets.QDialog,):
        cls.exec = dexec
        cls.exec_ = dexec
    for cls in (QtWidgets.QMessageBox, QtWidgets.QFileDialog, QtWidgets.QInputDialog, QtWidgets.QColorDialog,
                QtWidgets.QProgressDialog):
        cls.exec = lambda self, *a, **k: 0
        cls.exec_ = cls.exec
    QtWidgets.QMenu.exec = lambda self, *a, **k: None
    FD = QtWidgets.QFileDialog
    FD.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))
    FD.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))
    FD.getOpenFileNames = staticmethod(lambda *a, **k: ([], ""))
    FD.getExistingDirectory = staticmethod(lambda *a, **k: "")
    ID = QtWidgets.QInputDialog
    ID.getText = staticmethod(lambda *a, **k: ("", False))
    ID.getItem = staticmethod(lambda *a, **k: ("", False))
    ID.getInt = staticmethod(lambda *a, **k: (0, False))
    ID.getDouble = staticmethod(lambda *a, **k: (0.0, False))
    QtWidgets.QColorDialog.getColor = staticmethod(lambda *a, **k: QtGui.QColor())


def auto_marks(win, limit=9):
    """Numbered callouts on a real window's own panels: [(x, y, label)] in window pixels."""
    from PySide6 import QtCore, QtWidgets
    marks = []

    def add(w, label):
        if w is None or not w.isVisible() or w.width() < 20 or w.height() < 12:
            return
        p = w.mapTo(win, QtCore.QPoint(0, 0))
        if any(abs(p.x() - x) < 30 and abs(p.y() - y) < 20 for x, y, _ in marks):
            return
        marks.append((p.x() + 14, p.y() + 12, label))

    if isinstance(win, QtWidgets.QMainWindow):
        mb = win.menuBar()
        if mb is not None and mb.actions():
            add(mb, "Menu bar: " + ", ".join(a.text().replace("&", "") for a in mb.actions()[:6]))
        for tb in win.findChildren(QtWidgets.QToolBar):
            add(tb, "Toolbar" + (f" '{tb.windowTitle()}'" if tb.windowTitle() else ""))
        for d in win.findChildren(QtWidgets.QDockWidget):
            add(d, f"Panel '{d.windowTitle() or d.objectName()}'")
        cw = win.centralWidget()
        if cw is not None:
            add(cw, "Main view (" + type(cw).__name__ + ")")
    for t in win.findChildren(QtWidgets.QTabWidget):
        add(t, "Tabs: " + ", ".join(t.tabText(i).replace("&", "") for i in range(min(t.count(), 6))))
    for g in win.findChildren(QtWidgets.QGroupBox):
        if g.title():
            add(g, f"Box '{g.title()}'")
    if len(marks) < 4:
        for b in win.findChildren(QtWidgets.QPushButton):
            if b.text():
                add(b, f"Button '{b.text().replace('&', '')}'")
            if len(marks) >= limit:
                break
    return marks[:limit]


def grab_window(win, path, size=(1360, 860), wait=0.8):
    """Show a real window off screen, grab it to PNG, return its callouts."""
    from PySide6 import QtWidgets  # noqa: F401
    try:
        win.resize(*size)
    except Exception:
        pass
    win.show()
    pump(wait)
    pm = win.grab()
    if pm.isNull() or pm.width() < 50:
        raise RuntimeError("window grab returned an empty image")
    pm.save(path, "PNG")
    marks = auto_marks(win)
    r = pm.devicePixelRatio() if hasattr(pm, "devicePixelRatio") else 1.0
    return [(x * r, y * r, lab) for x, y, lab in marks]


# ============================================================================================ images
def finish_png(path, marks=()):
    """Shrink to <= MAX_W px, draw numbered callouts (circle + number), save optimised.  Returns legend labels."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(path).convert("RGB")
    s = 1.0
    if im.width > MAX_W:
        s = MAX_W / im.width
        im = im.resize((MAX_W, int(round(im.height * s))), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except Exception:
        try:
            font = ImageFont.truetype("arialbd.ttf", 15)
        except Exception:
            font = ImageFont.load_default()
    labels = []
    for i, (x, y, lab) in enumerate(marks, 1):
        if 0 < x <= 1 and 0 < y <= 1 and isinstance(x, float):  # fractional position
            cx, cy = x * im.width, y * im.height
        else:
            cx, cy = x * s, y * s
        r = 13
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(230, 60, 30), outline=(255, 255, 255), width=2)
        t = str(i)
        try:
            bb = dr.textbbox((0, 0), t, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            dr.text((cx - tw / 2 - bb[0], cy - th / 2 - bb[1]), t, fill="white", font=font)
        except Exception:
            dr.text((cx - 4, cy - 6), t, fill="white", font=font)
        labels.append(lab)
    im.save(path, "PNG", optimize=True)
    return labels


# ============================================================================================ demo data
class Ctx:
    """Demo project + windows shared by the shots."""

    def __init__(self, mode, tmp):
        self.mode, self.tmp = mode, tmp
        self._mw = None
        self.proj = None

    def project(self):
        if self.proj is None:
            from seismic_tool.test_maplayers import _setup
            self.proj, self.truth, self.dir = _setup(self.tmp, "demo")
        return self.proj

    def mw(self):
        if self._mw is None:
            from seismic_tool.gui.main_window import MainWindow
            self._mw = MainWindow(self.project().path)
            vol = self._mw.project.volume("Synth")
            sec = self._mw.section
            try:
                sec.show("3D", "Synth", "Inline", int(vol.inlines[len(vol.inlines) // 2]))
            except Exception:
                sec.source, sec.mode, sec.number = ("3D", "Synth"), "Inline", int(vol.inlines[len(vol.inlines) // 2])
                sec.redraw()
        return self._mw

    def nz_project(self):
        """Copy of the bundled NZ well project with absolute LAS paths (the repo copy is never written)."""
        d = os.path.join(self.tmp, "nzwells")
        dst = os.path.join(d, "NZ Well Correlation.wcproj.json")
        if not os.path.exists(dst):
            os.makedirs(d, exist_ok=True)
            js = json.load(open(NZ_PROJECT, encoding="utf-8"))
            base = os.path.dirname(NZ_PROJECT)
            for w in js.get("wells", []):
                w["las"] = [p if os.path.isabs(p) else os.path.join(base, p) for p in w.get("las", [])]
                for k in ("dlis", "files"):
                    if isinstance(w.get(k), list):
                        w[k] = [p if os.path.isabs(p) else os.path.join(base, p) for p in w[k]]
            json.dump(js, open(dst, "w", encoding="utf-8"), indent=1)
            cache = os.path.join(base, "NZ Well Correlation_cache")
            if os.path.isdir(cache):
                shutil.copytree(cache, os.path.join(d, "NZ Well Correlation_cache"))
        return dst


# ---------------------------------------------------------------------------- exports (work with or without Qt)
def exp_view3d(ctx, path):
    v = ctx.mw().viewer3d("Synth", force_software=True)
    for o in v.scene.objects:  # the demo's synthetic 2D lines sit in local coordinates, far from the 3D cube
        if o.kind == "line2d":
            o.visible = False
    v.scene.fit()
    v.scene.camera.zoom(1.25)
    v.scene.render_png(path, size=(1200, 780), title="3D Viewer - software renderer (same scene as the OpenGL view)")
    try:
        v.bus.unsubscribe("view3d")
    except Exception:
        pass
    return [(0.25, 0.09, "Survey box"), (0.52, 0.25, "Inline / crossline planes (seismic, red-white-blue)"),
            (0.29, 0.43, "Horizon 'Top' coloured by time"), (0.31, 0.56, "Time-slice plane and horizon 'Deep'"),
            (0.52, 0.86, "Well W-1 (tube)")], \
        "the 3D Viewer's own software renderer (the picture the window shows when OpenGL is not available)"


def styled_live_map(ctx):
    """Live Map with a demo WellTable column styled Spotfire-style (rule colours, bubble size, labels)."""
    lm = ctx.mw().live_map()
    if not getattr(ctx, "_lm_styled", False):
        try:
            t = lm.bus.table
            t.set_column("Mistie_ms", {"W-1": 2.0, "W-2": 14.0, "W-3": 6.0}, owner="tutorial", unit="ms")
            lm.bus.notify("welltable", source="tutorial")
            lm.set_color("rules", rules=lm.parse_rules("Mistie_ms > 10 = red; Mistie_ms > 5 = #ffa500"))
            lm.set_size("Mistie_ms", 8, 26)
            lm.set_label("Mistie_ms")
        except Exception:  # noqa: BLE001 - plain wells are still a true picture
            traceback.print_exc()
        ctx._lm_styled = True
    return lm


def exp_livemap(ctx, path):
    lm = styled_live_map(ctx)
    for l in lm.canvas.stack.layers:
        if l.get("kind") == "lines2d":  # synthetic 2D lines are in local coordinates
            lm.canvas.stack.set_visible(l["id"], False)
    lm.export_map(path, title="Live Map - demo project", dpi=110)
    return [(0.36, 0.08, "Map: survey outline and wells styled by a demo column Mistie_ms (rule colours "
                         "> 10 ms red, > 5 ms orange; bubble size + label = value)"),
            (0.71, 0.06, "Title block (title, project, date)"), (0.75, 0.30, "Location inset"),
            (0.71, 0.53, "Legend (layers + well style)"), (0.71, 0.79, "Scale bar"), (0.80, 0.90, "North arrow")], "the Live Map's own PNG export (Export map (PNG / PDF)...)"


def exp_horizon_map(ctx, path):
    from seismic_tool import maps as M
    proj = ctx.project()
    g = M.horizon_grid(proj, proj.horizons.get("Top"), "3D:Synth")
    M.export_png(g, path, title="Top - time structure (ms)")
    return [(0.45, 0.5, "Gridded horizon time with contours"), (0.92, 0.5, "Colour bar (ms)")], \
        "the horizon-map PNG export of the map hub"


def exp_seis_section(ctx, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    proj = ctx.project()
    vol = proj.volume("Synth")
    i = len(vol.inlines) // 2
    data = np.asarray(vol.inline(int(vol.inlines[i])), dtype=float) if hasattr(vol, "inline") else None
    if data is None:
        raise RuntimeError("volume has no inline reader")
    if data.shape[0] != len(vol.xlines):
        data = data.T
    t = np.asarray(vol.times, dtype=float)
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=100)
    c = np.nanpercentile(np.abs(data), 98) or 1.0
    ax.imshow(data.T, aspect="auto", cmap="seismic", vmin=-c, vmax=c,
              extent=[vol.xlines[0], vol.xlines[-1], t[-1], t[0]])
    for name, col in (("Top", "yellow"), ("Deep", "lime")):
        h = proj.horizons.get(name)
        z = h.data.get("3D:Synth") if h is not None else None
        if z is not None:
            ax.plot(vol.xlines, np.asarray(z)[i], color=col, lw=2, label=name)
    ax.set_xlabel("Crossline")
    ax.set_ylabel("TWT (ms)")
    ax.set_title(f"Inline {int(vol.inlines[i])} of the demo cube with the picked horizons")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return [(0.5, 0.55, "Seismic (red-white-blue), TWT down"), (0.87, 0.80, "Horizons Top / Deep (picked)")], \
        "plotted by the tutorial builder from the demo project's volume and horizons (not a window)"


def exp_wc_section(ctx, path):
    from well_correlation.export import export_section
    from well_correlation.project import Project
    p = Project.open(ctx.nz_project())
    wells = p.section_wells()[:5] if hasattr(p, "section_wells") else list(p.wells)[:5]
    export_section(p, wells, path)
    return [(0.5, 0.04, "Well name, datum and track headers"), (0.08, 0.45, "Lithology column"),
            (0.5, 0.5, "Log tracks (GR, resistivity, density / neutron, sonic)"),
            (0.42, 0.73, "Formation tops correlated between the wells")], \
        "the Well Correlation Tool's own section export (File > Export section image) on the bundled NZ wells"


def _glow_section(ctx, path, mode, with_glow=True):
    """Glow / Halo (PLAN 39): the Seismic tool's own section export (File > Export section image, the function
    behind it: seismic_tool.glow.export_section_png) on the demo inline with both horizons and a fault."""
    import numpy as np
    from seismic_tool import faults as flt
    from seismic_tool import glow as G
    from seismic_tool.gui.colormaps import lut
    proj = ctx.project()
    vol = proj.volume("Synth")
    i = len(vol.inlines) // 2
    il = int(vol.inlines[i])
    if proj.faults.get("Fault A") is None:  # one fault crossing the demo inline, picked with the tool's own API
        f = proj.faults.new("Fault A", color=(230, 30, 30))
        for k in (i - 5, i + 5):
            a = vol.xy(float(vol.inlines[k]), float(vol.xlines[len(vol.xlines) // 2 - 3]))
            b = vol.xy(float(vol.inlines[k]), float(vol.xlines[len(vol.xlines) // 2 + 3]))
            f.add_stick(np.array([[a[0], a[1], float(vol.times[5])], [b[0], b[1], float(vol.times[-5])]]))
    data = np.asarray(vol.inline(il), float)
    curves = []
    for name, col in (("Top", (255, 220, 0)), ("Deep", (0, 230, 90))):
        h = proj.horizons.get(name)
        z = h.data.get("3D:Synth") if h is not None else None
        if z is not None:
            curves.append({"x": np.asarray(vol.xlines, float), "y": np.asarray(z, float)[i], "colour": col,
                           "width": 2, "kind": "horizon", "name": name, "label": name})
    seg = flt.section_segments_3d(proj.faults.get("Fault A"), vol, "Inline", il)
    if seg is not None and len(seg[0]):
        xs = np.insert(np.c_[seg[0], seg[2]], 2, np.nan, axis=1).ravel()
        ts = np.insert(np.c_[seg[1], seg[3]], 2, np.nan, axis=1).ravel()
        curves.append({"x": xs, "y": ts, "colour": (230, 30, 30), "width": 2, "kind": "fault", "name": "Fault A",
                       "label": "Fault A"})
    st = G.GlowSettings(mode=mode if mode != "off" else "glow")
    G.export_section_png(path, data, vol.xlines, vol.times, curves, lut=lut("Seismic (red-white-blue)"), settings=st,
                         with_glow=with_glow and mode != "off", title=f"Inline {il} - "
                         + {"glow": "Glow", "halo": "Halo", "off": "no glow (before)"}[mode],
                         xlabel="Crossline", dpi=100, size=(11.0, 5.2))
    return [(0.5, 0.5, "Horizons Top / Deep with their glow" if mode == "glow" else
             "Horizons Top / Deep with a black halo" if mode == "halo" else "Horizons Top / Deep, plain lines"),
            (0.43, 0.16, "Fault A")], \
        "the Seismic tool's own section export (File > Export section image) on the demo inline - glow as on screen"


def exp_glow_off(ctx, path):
    return _glow_section(ctx, path, "off")


def exp_glow_on(ctx, path):
    return _glow_section(ctx, path, "glow")


def exp_glow_halo(ctx, path):
    return _glow_section(ctx, path, "halo")


def _aes_section(ctx):
    import numpy as np
    proj = ctx.project()
    vol = proj.volume("Synth")
    i = len(vol.inlines) // 2
    return proj, vol, i, int(vol.inlines[i]), np.asarray(vol.inline(int(vol.inlines[i])), float)


def exp_aes_hillshade(ctx, path):
    """40a: the demo horizon as a time-structure map, colour fill lit by the sun (aesthetics.shaded_grid) on the
    figure export (aesthetics.compose_figure)."""
    import numpy as np
    from seismic_tool import aesthetics as AE
    from seismic_tool.gui.colormaps import lut
    proj, vol, i, il, data = _aes_section(ctx)
    z = np.asarray(proj.horizons.get("Top").data["3D:Synth"], float)
    L = lut("Viridis-style")
    plain = AE.shaded_grid(z, L, 25.0, 25.0, AE.Sun(on=False))
    lit = AE.shaded_grid(z, L, 25.0, 25.0, AE.Sun(315, 35, ve=4, strength=0.7), positive_down=True)

    def draw(ax):
        ax.imshow(np.concatenate([plain, np.zeros((z.shape[0], 3, 4), np.uint8), lit], axis=1), origin="lower",
                  interpolation="nearest")
        ax.set_axis_off()
    AE.compose_figure(path, draw=draw, template={"title": "Top - time structure: plain (left) / hillshaded (right)",
                                                  "subtitle": "Sun az 315, el 35, VE 4, blend 70 %",
                                                  "project": "demo", "dpi": 80, "scale_bar": False})
    return [(0.27, 0.5, "Plain colour fill"), (0.72, 0.5, "Same map lit by the sun (hillshade)"),
            (0.5, 0.94, "Title block")], "the figure export (Export section figure / compose_figure) with the 40a " \
        "hillshade on the demo horizon"


def exp_aes_corender(ctx, path):
    import numpy as np
    from seismic_tool import aesthetics as AE
    from seismic_tool import attributes as A
    from seismic_tool import blend as B
    from seismic_tool import colortables as CT
    from seismic_tool.gui.colormaps import lut
    proj, vol, i, il, data = _aes_section(ctx)
    t = CT.get("Seismic (red-white-blue)")
    lo, hi = t.levels(data)
    base = B.lut_rgba(data.T, t.lut(), lo, hi)
    env = np.asarray(A.section_attribute(vol, "Envelope", "Inline", il), float)
    if env.shape != data.T.shape:
        env = env.T
    co = AE.corender(base, env, lut("Grey"), "transparency", 0.8, invert_alpha=True)

    def draw(ax):
        ax.imshow(np.concatenate([base, np.zeros((base.shape[0], 4, 4), np.uint8), co], axis=1), aspect="auto",
                  interpolation="nearest")
        ax.set_axis_off()
    AE.compose_figure(path, draw=draw, template={"title": f"Inline {il}: amplitude (left) / envelope co-rendered "
                                                 "(right, transparency by value)", "project": "demo", "dpi": 80,
                                                 "scale_bar": False})
    return [(0.27, 0.5, "Amplitude, red-white-blue"), (0.72, 0.5, "Envelope over it, low values transparent")], \
        "View > Co-render ... (aesthetics.corender) on the demo inline, shown with the figure export"


def _aes_theme(ctx, path, theme):
    import numpy as np
    from seismic_tool import aesthetics as AE
    from seismic_tool import glow as G
    from seismic_tool.gui.colormaps import lut
    proj, vol, i, il, data = _aes_section(ctx)
    curves = []
    for name, col in (("Top", (255, 220, 0)), ("Deep", (0, 230, 90))):
        h = proj.horizons.get(name)
        if h is not None and h.data.get("3D:Synth") is not None:
            curves.append({"x": np.asarray(vol.xlines, float), "y": np.asarray(h.data["3D:Synth"], float)[i],
                           "colour": col, "width": 2, "kind": "horizon", "name": name})
    AE.compose_figure(path, draw=lambda ax: G.draw_section(ax, data, vol.xlines, vol.times, curves,
                                                           lut("Seismic (red-white-blue)"), 99.0, G.GlowSettings(),
                                                           True, 80),
                      template={"title": f"Inline {il} - {theme} theme", "subtitle": "File > Export section figure",
                                "project": "demo", "author": "Chris", "dpi": 80, "theme": theme},
                      legend=[(c["name"], c["colour"]) for c in curves],
                      colourbar=(lut("Seismic (red-white-blue)"), -1, 1, "amplitude (relative)"), x_extent=2500.0)
    return [(0.45, 0.5, "Section with glowing horizons"), (0.5, 0.94, "Title block (title, project, author, date)"),
            (0.12, 0.86, "Scale bar"), (0.88, 0.45, "Colour bar + legend")], \
        "File > Export section figure (the code behind it) on the demo inline, " + theme + " theme"


def exp_aes_dark(ctx, path):
    return _aes_theme(ctx, path, "dark")


def exp_aes_light(ctx, path):
    return _aes_theme(ctx, path, "light")


def exp_aes_wc(ctx, path):
    from well_correlation.export import export_section
    from well_correlation.project import Project
    p = Project.open(ctx.nz_project())
    p.display.update(gr_fill="gradient", patterns=True, crossover=True)
    wells = p.section_wells()[:4] if hasattr(p, "section_wells") else list(p.wells)[:4]
    export_section(p, wells, path)
    return [(0.08, 0.45, "Lithology column with patterns"), (0.3, 0.5, "GR gradient fill (sand yellow -> shale)"),
            (0.5, 0.3, "Neutron-density crossover shading")], \
        "the Well Correlation Tool's section export with View > Track style turned on (bundled NZ wells)"


def exp_conditioning(ctx, path):
    """41: Seismic > Conditioning preview logic (conditioning.apply_chain, Interpretation preset) on the demo inline:
    original | conditioned | difference on one colour scale, with the QC numbers in the subtitle."""
    import numpy as np
    from seismic_tool import aesthetics as AE
    from seismic_tool import conditioning as C
    from seismic_tool.gui.colormaps import lut
    proj = ctx.project()
    vol = proj.volume("Synth")
    i = vol.nil // 2
    h = 6
    blk = np.asarray(vol.ilxl[max(0, i - h):i + h + 1], np.float32) * vol.inv
    chain = [s for s in C.preset("Interpretation") if s["type"] != "footprint"]
    out = C.apply_chain(blk, chain, vol.dt * 1000.0, offset=(max(0, i - h), 0), include_global=False,
                        il_total=vol.nil)
    k = i - max(0, i - h)
    o, c = blk[k], out[k]
    q = C.qc_metrics(o, c, vol.dt * 1000.0)
    lim = float(np.percentile(np.abs(o), 99)) or 1.0
    from matplotlib.colors import ListedColormap
    cm = ListedColormap(lut("Seismic (red-white-blue)")[:, :3] / 255.0)

    def draw(ax):
        gap = np.full((o.shape[1], 3), np.nan)
        img = np.concatenate([o.T, gap, c.T, gap, (c - o).T], axis=1)
        ax.imshow(img, aspect="auto", cmap=cm, vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_axis_off()
    AE.compose_figure(path, draw=draw, template={
        "title": f"Inline {int(vol.inlines[i])}: original | conditioned (Interpretation preset) | difference",
        "subtitle": f"RMS removed {q['rms_removed_pct']:.1f} %, signal leakage {q['leakage']:.2f} - same colour scale",
        "project": "demo", "dpi": 80, "scale_bar": False})
    return [(0.2, 0.45, "Original"), (0.5, 0.45, "Conditioned"), (0.8, 0.45, "Difference (should look like noise)"),
            (0.5, 0.94, "QC numbers: RMS removed, signal leakage")], \
        "the Conditioning preview computation (conditioning.apply_chain) on the demo cube, shown with the figure export"


def exp_surveyblend(ctx, path):
    """42: the blend of the synthetic two-survey test case (surveyblend.blend) - priority cut vs hybrid blend,
    width and added-slope QC maps - on the figure export."""
    import numpy as np
    from seismic_tool import aesthetics as AE
    from seismic_tool import surveyblend as SB
    from seismic_tool.test_surveyblend import scene
    g, X, Y, T, za, zb, rtrue, faults, fx = scene()
    r = SB.blend(g, [SB.Survey("A", za, 1), SB.Survey("B", zb, 2)], slope=2.0, method="hybrid", w_min=500.0,
                 w_max=12000.0, faults=faults)
    import matplotlib.pyplot as plt

    def draw(ax):
        ax.set_axis_off()
        fig = ax.figure
        bx = ax.get_position()
        panels = [("Priority cut (step at the seam)", r.cut - T, "RdBu_r", (-20, 20)),
                  ("Hybrid blend (2 m/km)", r.z - T, "RdBu_r", (-20, 20)),
                  ("Blend width (m)", r.qc["width"], "viridis", None),
                  ("Added slope across the seam (m/km)", r.qc["added_slope"] * 1000, "magma", (0, 2.2))]
        w = bx.width / 4
        for k, (title, arr, cm, lim) in enumerate(panels):
            a = fig.add_axes([bx.x0 + k * w + 0.01, bx.y0, w - 0.02, bx.height])
            im = a.imshow(arr, origin="lower", cmap=cm, vmin=None if lim is None else lim[0],
                          vmax=None if lim is None else lim[1], extent=[0, 5, 0, 12])
            a.set_title(title, fontsize=9)
            a.set_xlabel("km", fontsize=8)
            a.tick_params(labelsize=7)
            a.axvline(fx / 1000, color="k", lw=0.8, ls="--")
            fig.colorbar(im, ax=a, fraction=0.05)
    AE.compose_figure(path, draw=draw, template={
        "title": "Survey blending: surveys A (south, priority) and B (north), 1 km overlap, fault at 3.5 km",
        "subtitle": "Maps show (merged - truth): the cut leaves a 4-18 m step; the blend spreads it at <= 2 m/km. "
                    + r.readout, "project": "test case", "dpi": 80, "scale_bar": False})
    return [(0.15, 0.45, "Priority cut: step at the seam"), (0.38, 0.45, "Blended: misfit spread gently"),
            (0.62, 0.45, "Width: wide where the surveys disagree"), (0.85, 0.45, "Added slope stays <= 2 m/km")], \
        "Interpret > Survey horizon blending (surveyblend.blend) on the synthetic test case of the build (two surveys, " \
        "residual tilt + bump, a fault across the seam)"


EXPORTS = {"surveyblend": exp_surveyblend, "conditioning": exp_conditioning, "aes_hillshade": exp_aes_hillshade, "aes_corender": exp_aes_corender, "aes_dark": exp_aes_dark,
           "aes_light": exp_aes_light, "aes_wc": exp_aes_wc, "glow_off": exp_glow_off, "glow_on": exp_glow_on, "glow_halo": exp_glow_halo, "view3d": exp_view3d, "fix_after": exp_view3d, "livemap": exp_livemap, "horizon_map": exp_horizon_map,
           "seis_section": exp_seis_section, "wc_section": exp_wc_section}


# ---------------------------------------------------------------------------- real windows (PC only)
def _w_seis_main(ctx):
    return ctx.mw()


def _w_wc_main(ctx):
    from well_correlation.gui.main_window import MainWindow
    ctx._wc = MainWindow(ctx.nz_project())
    return ctx._wc


def _w_wc_petro(ctx):
    wc = getattr(ctx, "_wc", None) or _w_wc_main(ctx)
    r = wc.show_petro()
    return r if r is not None else getattr(wc, "petro_win", None) or getattr(wc, "_petro", None)


def _w_framework(ctx):
    fw = ctx.mw().framework("Synth")
    try:
        fw.build()
    except Exception:
        pass
    ctx._fw = fw
    return fw


def _w_propmodel(ctx):
    fw = getattr(ctx, "_fw", None) or _w_framework(ctx)
    return fw.properties()


def _w_view3d(ctx):
    return ctx.mw().viewer3d("Synth")


WINDOWS = {
    "seis_main": _w_seis_main, "wc_main": _w_wc_main, "wc_petro": _w_wc_petro,
    "livemap": styled_live_map, "view3d": _w_view3d, "fix_after": _w_view3d,
    "welltie": lambda c: c.mw().well_tie(), "velocity": lambda c: c.mw().velocity_model(),
    "velvolume": lambda c: c.mw().velocity_volume(), "faultmap": lambda c: c.mw().fault_mapping("Synth"),
    "framework": _w_framework, "propmodel": _w_propmodel, "mistie": lambda c: c.mw().mistie_analysis(),
    "basin": lambda c: c.mw().basin_model(), "inversion": lambda c: c.mw().inversion(),
    "analytics": lambda c: c.mw().analytics(), "dashboard": lambda c: c.mw().dashboard(),
    "horizon_map": lambda c: c.mw().horizon_maps(),
}

TITLES = {
    "seis_main": "Seismic Interpretation main window", "seis_section": "Inline with picked horizons",
    "wc_main": "Well Correlation main window (bundled NZ wells)", "wc_section": "Correlation section export",
    "wc_petro": "Petrophysics window", "livemap": "Live Map", "horizon_map": "Horizon map",
    "view3d": "3D Viewer", "fix_after": "3D Viewer after the fix", "welltie": "Well tie dialog",
    "velocity": "Velocity model + depth conversion dialog", "velvolume": "Velocity volume (advanced) dialog",
    "faultmap": "Automated fault mapping dialog", "framework": "Structural framework + 3D grid dialog",
    "propmodel": "Property modelling + volumetrics dialog", "mistie": "Mistie analysis dialog",
    "basin": "Basin model dialog", "inversion": "Seismic inversion dialog", "analytics": "Analytics window",
    "dashboard": "Dashboard window", "glow_off": "Before: horizons and a fault without glow",
    "glow_on": "After: Glow mode", "glow_halo": "After: Halo mode",
    "aes_hillshade": "Hillshaded horizon map", "aes_corender": "Co-rendered envelope over amplitude",
    "aes_dark": "Exported figure, dark theme", "aes_light": "Exported figure, light theme",
    "aes_wc": "Styled correlation panel",
    "conditioning": "Conditioning preview: original | conditioned | difference",
    "surveyblend": "Survey blending: cut vs blend, width and added-slope maps",
}


def take_shot(ctx, sid, img_dir):
    """-> dict(id, title, file or None, caption, legend)."""
    path = os.path.join(img_dir, sid + ".png")
    title = TITLES.get(sid, sid)
    errors = []
    if ctx.mode == "real" and sid in WINDOWS:
        try:
            w = WINDOWS[sid](ctx)
            if w is None:
                raise RuntimeError("the window did not open on the demo project")
            marks = grab_window(w, path, wait=1.5 if sid in ("view3d", "fix_after") else 0.8)
            legend = finish_png(path, marks)
            return dict(id=sid, title=title, file=path, legend=legend,
                        caption=f"{title} - screenshot of the real window on the demo project "
                                f"(captured {datetime.date.today():%d %b %Y}).")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")
            traceback.print_exc()
    if sid in EXPORTS:
        try:
            marks, what = EXPORTS[sid](ctx, path)
            legend = finish_png(path, marks)
            note = "" if ctx.mode == "real" else " The window itself could not be drawn where this tutorial was " \
                "built (no Qt on the build server); Install GeoHarmonic.bat rebuilds this page with the real window."
            if errors:
                note = f" The window could not be captured ({errors[0]}); this is the export instead."
            return dict(id=sid, title=title, file=path, legend=legend, caption=f"{title} - {what}.{note}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")
            traceback.print_exc()
    why = errors[0] if errors else ("no Qt on the build server - the window cannot be drawn here" if ctx.mode != "real"
                                    else "no capture defined")
    return dict(id=sid, title=title, file=None, legend=[],
                caption=f"{title} - not captured ({why}). Install GeoHarmonic.bat rebuilds the tutorial on your PC "
                        f"with a real screenshot.")


# ============================================================================================ test results
def load_results(files):
    lines = []
    for f in files:
        if f and os.path.isfile(f):
            lines += open(f, encoding="utf-8", errors="replace").read().splitlines()
    return lines


def default_results(run_tests, tmp):
    out = []
    for tool, pkg in (("Seismic Interpretation Tool", "seismic_tool"), ("Well Correlation Tool", "well_correlation")):
        f = os.path.join(REPO, "tools", tool, "test_results.txt")
        if os.path.isfile(f):
            out.append(f)
        elif run_tests:
            o = os.path.join(tmp, pkg + "_results.txt")
            print(f"running {pkg} --selftest for the results chapter ...", flush=True)
            with open(o, "w") as fh:
                subprocess.run([sys.executable, "-m", pkg, "--selftest"], cwd=os.path.join(REPO, "tools", tool),
                               stdout=fh, stderr=subprocess.STDOUT, timeout=1800)
            out.append(o)
    return out


def pick_results(lines, patterns, limit=8):
    hits = []
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for l in lines:
            if rx.search(l) and l not in hits:
                hits.append(l)
    numeric = [h for h in hits if re.search(r"\d", h[5:])]
    return (numeric or hits)[:limit], len(hits)


def summary_counts(files):
    rows = []
    for f in files:
        if not os.path.isfile(f):
            continue
        txt = open(f, encoding="utf-8", errors="replace").read()
        p = len(re.findall(r"^PASS ", txt, flags=re.M))
        fl = len(re.findall(r"^FAIL ", txt, flags=re.M))
        verdicts = re.findall(r"^(SELFTEST PASSED|SELFTEST FAILED|GUI TEST PASSED|GUI TEST FAILED)", txt, flags=re.M)
        rows.append((os.path.basename(os.path.dirname(f)) if f.endswith("test_results.txt") else os.path.basename(f),
                     p, fl, ", ".join(verdicts) or "-"))
    rows = [(re.sub(r"\.txt$", "", a), b, c, d) for a, b, c, d in rows]
    return rows


# ============================================================================================ writers
def md_escape(s):
    return s


def build_markdown(chapters, shots, results, counts, mode, img_rel):
    from tutorial_content import INTRO
    out = ["# GeoHarmonic - illustrated tutorial", "",
           f"*Generated by `dev/build_tutorial.py` on {datetime.datetime.now():%d %b %Y %H:%M} "
           f"({'real Qt screenshots' if mode == 'real' else 'build server: no Qt, tool exports only'}).*", "",
           INTRO, "", "## Contents", ""]
    for c in chapters:
        out.append(f"- [{c['title']}](#{c['id']})")
    out.append("")
    for c in chapters:
        lead = "" if c.get("plain") else "**What it is for.** "
        out += [f'<a id="{c["id"]}"></a>', "", f"## {c['title']}", "", lead + c["purpose"], ""]
        if c["shots"]:
            out += ["### How it looks", ""]
            for sid in c["shots"]:
                s = shots[sid]
                if s["file"]:
                    out.append(f"![{s['title']}]({img_rel}/{os.path.basename(s['file'])})")
                    out.append("")
                out.append(f"*{s['caption']}*")
                out.append("")
                for i, lab in enumerate(s["legend"], 1):
                    out.append(f"{i}. {lab}")
                if s["legend"]:
                    out.append("")
        if c["steps"]:
            out += ["### How it works", ""]
            out += [f"{i}. {st}" for i, st in enumerate(c["steps"], 1)]
            out.append("")
        res, n = results.get(c["id"], ([], 0))
        if c.get("results") or res:
            out += ["### Results", ""]
            if c.get("results"):
                out += [c["results"], ""]
            if c["id"] == "fixes" and counts:
                out += ["| Test output | PASS | FAIL | Verdict |", "|---|---|---|---|"]
                out += [f"| {a} | {b} | {cc} | {d} |" for a, b, cc, d in counts]
                out.append("")
            if res:
                out += ["```"] + res + ["```", f"*{len(res)} of {n} matching test checks shown.*", ""]
        if c.get("limits"):
            out += ["### Known limits", ""] + [f"- {x}" for x in c["limits"]] + [""]
        if c.get("trouble"):
            out += ["### Troubleshooting", ""] + [f"- {x}" for x in c["trouble"]] + [""]
    return "\n".join(out) + "\n"


_CSS = """
:root{--bg:#fbfbf9;--fg:#1d2329;--mut:#5b6670;--acc:#c2401f;--card:#fff;--line:#d9dde1;--code:#f1f3f5}
@media (prefers-color-scheme:dark){:root{--bg:#15181b;--fg:#e4e7ea;--mut:#9aa5af;--acc:#ff7a52;--card:#1d2125;
--line:#30363c;--code:#22272c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 "Segoe UI",system-ui,
-apple-system,sans-serif}main{max-width:1100px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:2rem;margin:.2em 0}h2{margin-top:2.4em;padding-top:.6em;border-top:2px solid var(--line)}
h3{color:var(--acc);font-size:1.05rem;text-transform:uppercase;letter-spacing:.04em;margin-top:1.6em}
nav{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 18px}
nav ol{columns:2;margin:0;padding-left:1.2em}@media(max-width:700px){nav ol{columns:1}}
a{color:var(--acc)}figure{margin:16px 0;background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:10px}figure img{max-width:100%;height:auto;display:block;margin:0 auto}figcaption{color:var(--mut);
font-size:.92rem;margin-top:8px}.legend{margin:6px 0 0;padding-left:1.4em;font-size:.92rem}
.legend li::marker{color:var(--acc);font-weight:700}pre{background:var(--code);border-radius:6px;padding:10px;
overflow-x:auto;font-size:.82rem}table{border-collapse:collapse;margin:10px 0}td,th{border:1px solid var(--line);
padding:4px 10px}code{background:var(--code);padding:1px 4px;border-radius:3px}.small{color:var(--mut);font-size:.9rem}
"""


def md_inline(s):
    """Tiny markdown subset -> HTML (bold, italic, code, links)."""
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", s)
    return s


def md_block(text):
    """Paragraphs + pipe tables + plain lines."""
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()
        if lines and lines[0].startswith("|"):
            rows = [l.strip().strip("|").split("|") for l in lines if not re.match(r"^\|[-| ]+\|$", l.strip())]
            t = ["<table>"]
            for i, r in enumerate(rows):
                tag = "th" if i == 0 else "td"
                t.append("<tr>" + "".join(f"<{tag}>{md_inline(c.strip())}</{tag}>" for c in r) + "</tr>")
            out.append("\n".join(t + ["</table>"]))
        else:
            out.append("<p>" + md_inline(" ".join(l.strip() for l in lines)) + "</p>")
    return "\n".join(out)


def build_html(chapters, shots, results, counts, mode):
    from tutorial_content import INTRO
    h = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         "<title>GeoHarmonic Tutorial</title><style>" + _CSS + "</style></head><body><main>",
         "<h1>GeoHarmonic - illustrated tutorial</h1>",
         f"<p class='small'>Generated by <code>dev/build_tutorial.py</code> on {datetime.datetime.now():%d %b %Y %H:%M} "
         f"({'real Qt screenshots' if mode == 'real' else 'build server: no Qt, tool exports only'}).</p>",
         md_block(INTRO), "<nav><b>Contents</b><ol>"]
    h += [f"<li><a href='#{c['id']}'>{html.escape(c['title'])}</a></li>" for c in chapters]
    h.append("</ol></nav>")
    for c in chapters:
        h.append(f"<h2 id='{c['id']}'>{html.escape(c['title'])}</h2>")
        h.append(md_block(("" if c.get("plain") else "**What it is for.** ") + c["purpose"]))
        if c["shots"]:
            h.append("<h3>How it looks</h3>")
            for sid in c["shots"]:
                s = shots[sid]
                h.append("<figure>")
                if s["file"]:
                    b64 = base64.b64encode(open(s["file"], "rb").read()).decode()
                    h.append(f"<img alt='{html.escape(s['title'])}' src='data:image/png;base64,{b64}'>")
                h.append(f"<figcaption>{md_inline(s['caption'])}</figcaption>")
                if s["legend"]:
                    h.append("<ol class='legend'>" + "".join(f"<li>{md_inline(x)}</li>" for x in s["legend"]) + "</ol>")
                h.append("</figure>")
        if c["steps"]:
            h.append("<h3>How it works</h3><ol>" + "".join(f"<li>{md_inline(x)}</li>" for x in c["steps"]) + "</ol>")
        res, n = results.get(c["id"], ([], 0))
        if c.get("results") or res:
            h.append("<h3>Results</h3>")
            if c.get("results"):
                h.append(md_block(c["results"]))
            if c["id"] == "fixes" and counts:
                h.append("<table><tr><th>Test output</th><th>PASS</th><th>FAIL</th><th>Verdict</th></tr>" +
                         "".join(f"<tr><td>{html.escape(a)}</td><td>{b}</td><td>{cc}</td><td>{html.escape(d)}</td></tr>"
                                 for a, b, cc, d in counts) + "</table>")
            if res:
                h.append("<pre>" + html.escape("\n".join(res)) + "</pre>")
                h.append(f"<p class='small'>{len(res)} of {n} matching test checks shown.</p>")
        if c.get("limits"):
            h.append("<h3>Known limits</h3><ul>" + "".join(f"<li>{md_inline(x)}</li>" for x in c["limits"]) + "</ul>")
        if c.get("trouble"):
            h.append("<h3>Troubleshooting</h3><ul>" + "".join(f"<li>{md_inline(x)}</li>" for x in c["trouble"]) + "</ul>")
    h.append("</main></body></html>")
    return "\n".join(h)


# ============================================================================================ main
def build(out_dir, mode, result_files, log=print):
    import importlib
    import tutorial_content
    importlib.reload(tutorial_content)
    chapters = tutorial_content.CHAPTERS
    img_dir = os.path.join(out_dir, "tutorial")
    os.makedirs(img_dir, exist_ok=True)
    for f in os.listdir(img_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(img_dir, f))
    tmp = tempfile.mkdtemp(prefix="gh_tutorial_")
    try:
        if mode == "real":
            from PySide6 import QtWidgets
            QtWidgets.QApplication.instance() or QtWidgets.QApplication([sys.argv[0]])
            patch_modals()
        ctx = Ctx(mode, tmp)
        shots = {}
        for c in chapters:
            for sid in c["shots"]:
                if sid not in shots:
                    log(f"  shot {sid} ...")
                    shots[sid] = take_shot(ctx, sid, img_dir)
                    log(f"    -> {'ok' if shots[sid]['file'] else 'not captured'}")
        lines = load_results(result_files)
        results = {c["id"]: pick_results(lines, c.get("results_re", [])) for c in chapters}
        counts = summary_counts(result_files)
        md = build_markdown(chapters, shots, results, counts, mode, "tutorial")
        open(os.path.join(out_dir, "TUTORIAL.md"), "w", encoding="utf-8").write(md)
        page = build_html(chapters, shots, results, counts, mode)
        hp = os.path.join(out_dir, "TUTORIAL.html")
        open(hp, "w", encoding="utf-8").write(page)
        n_img = sum(1 for s in shots.values() if s["file"])
        log(f"wrote {hp} ({os.path.getsize(hp) / 1e6:.1f} MB, {n_img}/{len(shots)} pictures) and TUTORIAL.md")
        return dict(shots=shots, results=results, html=hp, md=os.path.join(out_dir, "TUTORIAL.md"))
    finally:
        try:
            if mode == "real":
                from PySide6 import QtWidgets
                for w in QtWidgets.QApplication.topLevelWidgets():
                    w.close()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--pc", action="store_true", help="write to tutorial_local/ (untracked) - used by the installer")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--results", action="append", default=[])
    ap.add_argument("--run-tests", action="store_true")
    a = ap.parse_args(argv)
    import faulthandler
    faulthandler.dump_traceback_later(15 * 60, exit=True)  # never hang the installer
    out = a.out or os.path.join(REPO, "tutorial_local" if a.pc else "docs")
    os.makedirs(out, exist_ok=True)
    mode = detect_qt(force_mock=a.mock)
    print(f"GeoHarmonic tutorial builder - Qt: {mode} - output: {out}", flush=True)
    tmp = tempfile.mkdtemp(prefix="gh_tut_res_")
    files = a.results or default_results(a.run_tests, tmp)
    r = build(out, mode, files)
    shutil.rmtree(tmp, ignore_errors=True)
    print("TUTORIAL BUILT", flush=True)
    return r


if __name__ == "__main__":
    main()
