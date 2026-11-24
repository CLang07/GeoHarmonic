"""QGIS-independent part of the GeoHarmonic Link plugin: the localhost message client and the shared-GeoPackage reader.
Same JSON-lines protocol as seismic_tool/qgislink.py (port 47561)."""
import json
import os
import queue
import socket
import sqlite3
import threading
import time

DEFAULT_PORT = 47561


class LinkClient:
    def __init__(self, port=DEFAULT_PORT, host="127.0.0.1"):
        self.port, self.host = int(port), host
        self.inbox = queue.Queue()
        self.sock = None
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    def connect(self, timeout=2.0):
        try:
            s = socket.create_connection((self.host, self.port), timeout=timeout)
        except OSError as e:
            self.error = f"GeoHarmonic is not listening on {self.host}:{self.port} ({e}) - View > Live link with QGIS"
            return False
        s.settimeout(0.2)
        self.sock = s
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.send({"type": "hello", "app": "QGIS"})
        return True

    def _loop(self):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    try:
                        self.inbox.put(json.loads(line.decode("utf-8")))
                    except ValueError:
                        pass
        self.sock = None
        self.inbox.put({"type": "disconnected"})

    def send(self, msg):
        if self.sock is None:
            return False
        try:
            self.sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))
            return True
        except OSError as e:
            self.error = str(e)
            return False

    def poll(self, n=100):
        out = []
        while len(out) < n:
            try:
                out.append(self.inbox.get_nowait())
            except queue.Empty:
                break
        return out

    def wait(self, kind, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                m = self.inbox.get(timeout=0.05)
            except queue.Empty:
                continue
            if m.get("type") == kind:
                return m
        return None

    def close(self):
        self._stop.set()
        try:
            if self.sock:
                self.sock.close()
        except OSError:
            pass


def read_project(gpkg):
    """{"vector": [table], "editable": [table], "rasters": [(name, path, ramp, vmin, vmax)], "meta": {...}}."""
    con = sqlite3.connect(gpkg)
    try:
        tables = [r[0] for r in con.execute("SELECT table_name FROM gpkg_contents WHERE data_type = 'features' ORDER BY table_name")]
        try:
            rasters = list(con.execute("SELECT name, path, ramp, vmin, vmax FROM gh_rasters"))
        except sqlite3.OperationalError:
            rasters = []
        try:
            meta = {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM gh_meta")}
        except sqlite3.OperationalError:
            meta = {}
    finally:
        con.close()
    editable = [t for t in ("prospects_edit", "leases", "section_lines") if t in tables]
    order = ["survey_3d", "lines_2d", "fault_polygons", "faults", "prospects", "tops", "wells"]
    vec = sorted([t for t in tables if t not in editable], key=lambda t: (order.index(t) if t in order else -1, t))
    return {"vector": vec, "editable": editable, "rasters": rasters, "meta": meta}


class FileWatcher:
    """mtime poll (QFileSystemWatcher misses replace-by-rename on some platforms)."""

    def __init__(self, path):
        self.path = path
        self.mtime = os.path.getmtime(path) if os.path.exists(path) else None

    def changed(self):
        m = os.path.getmtime(self.path) if os.path.exists(self.path) else None
        if m != self.mtime:
            self.mtime = m
            return True
        return False


def polygon_message(name, ring):
    return {"type": "polygon", "name": name, "xy": [[float(x), float(y)] for x, y in ring]}


def line_message(name, pts):
    return {"type": "section_line", "name": name, "xy": [[float(x), float(y)] for x, y in pts]}
