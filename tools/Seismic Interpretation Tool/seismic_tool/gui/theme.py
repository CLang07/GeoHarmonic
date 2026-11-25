"""Dark / light theme + anti-aliasing switch (PLAN 40d / 40f) - the Qt side of aesthetics.THEMES.

    apply_theme(app, "light", bus)      # palette + stylesheet + every open plot (background, axes, grid) + 3D views
    set_antialias(app, False)           # speed switch (View > Anti-aliasing)

Switches without a restart: every open pyqtgraph view is restyled, the map bus gets a display event (kind 'theme')
so the 3D Viewer / Live Map / glow halo auto-contrast follow; the choice is saved in ~/.geoharmonic/settings.json.
"""
from PySide6 import QtGui, QtWidgets
import pyqtgraph as pg

from .. import aesthetics as AE

CURRENT = {"theme": AE.DEFAULT_THEME, "antialias": False}


def palette(name):
    t = AE.THEMES.get(name, AE.THEMES[AE.DEFAULT_THEME])
    pal = QtGui.QPalette()
    roles = (("Window", "window"), ("WindowText", "text"), ("Base", "base"), ("AlternateBase", "alt"),
             ("Text", "text"), ("Button", "button"), ("ButtonText", "text"), ("Highlight", "highlight"),
             ("HighlightedText", "highlight_text"), ("ToolTipBase", "base"), ("ToolTipText", "text"),
             ("Mid", "mid"))
    for role, key in roles:
        r = getattr(QtGui.QPalette, role, None)
        if r is None:
            r = getattr(getattr(QtGui.QPalette, "ColorRole", None), role, None)
        if r is not None:
            try:
                pal.setColor(r, QtGui.QColor(t[key]))
            except Exception:
                pass
    return pal


def style_plot(widget, name):
    """Background, axes, labels and grid of one pyqtgraph PlotWidget / GraphicsLayoutWidget."""
    c = AE.plot_colours(name)
    n = 0
    try:
        widget.setBackground(c["background"])
        n += 1
    except Exception:
        return 0
    items = []
    try:
        items.append(widget.getPlotItem())
    except Exception:
        pass
    try:
        ci = widget.ci  # GraphicsLayoutWidget
        items += [it for it in ci.items if hasattr(it, "getAxis")]
    except Exception:
        pass
    fg = c["foreground"]
    for pi in items:
        for ax in ("left", "bottom", "right", "top"):
            try:
                a = pi.getAxis(ax)
                a.setPen(pg.mkPen(fg))
                a.setTextPen(pg.mkPen(fg))
            except Exception:
                pass
    return n


def plot_widgets(app=None):
    app = app or QtWidgets.QApplication.instance()
    if app is None:
        return []
    out = []
    for w in app.allWidgets():
        if isinstance(w, (pg.PlotWidget, pg.GraphicsLayoutWidget)):
            out.append(w)
    return out


def apply_theme(app=None, name=None, bus=None, save=True, widgets=None):
    """Switch the whole app; returns the number of plot views restyled."""
    name = name if name in AE.THEMES else AE.DEFAULT_THEME
    app = app or QtWidgets.QApplication.instance()
    CURRENT["theme"] = name
    c = AE.plot_colours(name)
    try:
        pg.setConfigOptions(background=c["background"], foreground=c["foreground"])
    except Exception:
        pass
    if app is not None:
        try:
            app.setPalette(palette(name))
            app.setStyleSheet(AE.stylesheet(name))
        except Exception:
            pass
    n = 0
    for w in (widgets if widgets is not None else plot_widgets(app)):
        n += style_plot(w, name)
    if bus is not None:
        bus.publish("display", "theme", kind="theme", theme=name, plot=c)
    if save:
        try:
            AE.save_user_settings({"theme": name})
        except OSError:
            pass
    return n


def set_antialias(on, app=None, save=True):
    """Anti-aliased lines / text in pyqtgraph (off = faster); 3D MSAA is requested by the GL view itself."""
    CURRENT["antialias"] = bool(on)
    pg.setConfigOptions(antialias=bool(on))
    app = app or QtWidgets.QApplication.instance()
    n = 0
    if app is not None:
        for w in plot_widgets(app):
            try:
                w.setAntialiasing(bool(on))
                n += 1
            except Exception:
                pass
    if save:
        try:
            AE.save_user_settings({"antialias": bool(on)})
        except OSError:
            pass
    return n


def startup(app):
    """At program start: saved theme (dark by default) + anti-aliasing setting."""
    s = AE.load_user_settings()
    apply_theme(app, s.get("theme", AE.DEFAULT_THEME), save=False)
    set_antialias(bool(s.get("antialias", True)), app, save=False)
    return s
