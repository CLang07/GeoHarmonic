"""Live-linked cursor and saved window layouts (PLAN 23c).  Qt-free so it can be tested headless.

CursorBus: publish / subscribe of the cursor position in map coordinates (+ time / depth and an optional well).
The section panel publishes as the mouse moves; the basemap shows a cross, the dashboard and the well strip follow.
A subscriber never receives its own messages (no echo loops); `min_interval` throttles a fast mouse.

    bus = CursorBus()
    bus.subscribe("map", lambda m: basemap.show_cursor(m["x"], m["y"]))
    bus.publish("section", x=..., y=..., t=..., il=..., xl=...)

Layouts: named window states stored in project.display["layouts"] = {name: {...}} - which docks / tool windows are
open, the section source / mode / number / attribute / blend, and an opaque Qt state string (base64 of
QMainWindow.saveState / saveGeometry) restored when present.
"""
import time

FIELDS = ("x", "y", "t", "z", "il", "xl", "cdp", "line", "well", "md")


class CursorBus:
    def __init__(self, min_interval=0.0, clock=time.monotonic):
        self._subs = {}
        self.min_interval = float(min_interval)
        self._clock = clock
        self._last = {}
        self.last = None
        self.enabled = True
        self.sent = 0

    def subscribe(self, name, fn):
        self._subs[name] = fn

    def unsubscribe(self, name):
        self._subs.pop(name, None)

    def names(self):
        return list(self._subs)

    def publish(self, source, **kw):
        """Send a cursor message {source, x, y, t, ...} to every other subscriber.  Returns the number delivered
        (0 when disabled or throttled)."""
        if not self.enabled:
            return 0
        now = self._clock()
        if self.min_interval > 0 and now - self._last.get(source, -1e18) < self.min_interval:
            return 0
        self._last[source] = now
        msg = {"source": source}
        msg.update({k: v for k, v in kw.items() if v is not None})
        self.last = msg
        n = 0
        for name, fn in list(self._subs.items()):
            if name == source:
                continue
            try:
                fn(msg)
                n += 1
            except Exception:
                pass  # a broken window must not stop the others
        self.sent += n
        return n

    def clear(self, source="*"):
        """Cursor left a view: subscribers get {"source", "clear": True}."""
        msg = {"source": source, "clear": True}
        for name, fn in list(self._subs.items()):
            if name != source:
                try:
                    fn(msg)
                except Exception:
                    pass


def nearest_well(wells, x, y, max_dist=None):
    """(name, distance) of the well [(name, x, y)] nearest to (x, y) or (None, inf)."""
    best, bd = None, float("inf")
    for w in wells or []:
        d = ((w[1] - x) ** 2 + (w[2] - y) ** 2) ** 0.5
        if d < bd:
            best, bd = w[0], d
    if max_dist is not None and bd > max_dist:
        return None, bd
    return best, bd


# ---------------------------------------------------------------- layouts
def save_layout(project, name, state):
    """Store a layout dict under project.display['layouts'][name]; returns the sorted layout names."""
    if not name:
        raise ValueError("layout needs a name")
    lay = project.display.setdefault("layouts", {})
    lay[str(name)] = dict(state)
    return sorted(lay)


def load_layout(project, name):
    return dict((project.display.get("layouts") or {}).get(name) or {}) or None


def delete_layout(project, name):
    (project.display.get("layouts") or {}).pop(name, None)
    return sorted(project.display.get("layouts") or {})


def layout_names(project):
    return sorted((project.display.get("layouts") or {})) if project is not None else []


# built-in templates (window sets) offered even before anything is saved
TEMPLATES = {
    "Interpretation": {"docks": {"data": True, "basemap": True, "layers": False, "horizons": True, "faults": True},
                       "windows": []},
    "Mapping": {"docks": {"data": True, "basemap": True, "layers": True, "horizons": True, "faults": False},
                "windows": []},
    "Well review": {"docks": {"data": True, "basemap": True, "layers": True, "horizons": False, "faults": False},
                    "windows": ["petro_strip", "dashboard"]},
}
