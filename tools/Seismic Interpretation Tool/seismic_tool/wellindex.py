"""Well index for large well counts (PLAN item 15): SQLite (Python standard library) table of well headers, tops
and a log catalogue, with a grid-bucket spatial index so maps / the dashboard query only the wells in view.
Logs themselves stay in their LAS / npz caches and are loaded lazily."""
import math
import os
import sqlite3

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS wells (name TEXT PRIMARY KEY, api TEXT, x REAL, y REAL, elev REAL, td REAL,
                                  bx INTEGER, by INTEGER, survey TEXT);
CREATE INDEX IF NOT EXISTS wells_bucket ON wells (bx, by);
CREATE INDEX IF NOT EXISTS wells_api ON wells (api);
CREATE TABLE IF NOT EXISTS tops (well TEXT, formation TEXT, md REAL, PRIMARY KEY (well, formation));
CREATE INDEX IF NOT EXISTS tops_fm ON tops (formation);
CREATE TABLE IF NOT EXISTS logs (well TEXT, curve TEXT, top REAL, base REAL, path TEXT, PRIMARY KEY (well, curve));
CREATE INDEX IF NOT EXISTS logs_curve ON logs (curve);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


class WellIndex:
    def __init__(self, path=":memory:", bucket=2000.0):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        row = self.db.execute("SELECT v FROM meta WHERE k='bucket'").fetchone()
        if row:
            self.bucket = float(row[0])
        else:
            self.bucket = float(bucket)
            self.db.execute("INSERT INTO meta VALUES ('bucket', ?)", (str(self.bucket),))
            self.db.commit()

    def close(self):
        self.db.close()

    def __len__(self):
        return self.db.execute("SELECT COUNT(*) FROM wells").fetchone()[0]

    # ------------------------------------------------------------ writing
    def _b(self, v):
        return None if v is None or not math.isfinite(v) else int(math.floor(v / self.bucket))

    def add_wells(self, rows):
        """rows: iterable of dicts / tuples (name, api, x, y, elev, td[, survey path])."""
        data = []
        for r in rows:
            if isinstance(r, dict):
                r = (r["name"], r.get("api"), r.get("x"), r.get("y"), r.get("elev"), r.get("td"), r.get("survey"))
            name, api, x, y, elev, td = r[:6]
            survey = r[6] if len(r) > 6 else None
            x = None if x is None else float(x)
            y = None if y is None else float(y)
            data.append((name, api, x, y, elev, td, self._b(x), self._b(y), survey))
        self.db.executemany("INSERT OR REPLACE INTO wells VALUES (?,?,?,?,?,?,?,?,?)", data)
        self.db.commit()
        return len(data)

    def add_tops(self, tops):
        """tops: {well: {formation: md}}."""
        self.db.executemany("INSERT OR REPLACE INTO tops VALUES (?,?,?)",
                            [(w, f, float(md)) for w, d in tops.items() for f, md in d.items() if md is not None])
        self.db.commit()

    def add_logs(self, rows):
        """rows: (well, curve, top md, base md, file path)."""
        self.db.executemany("INSERT OR REPLACE INTO logs VALUES (?,?,?,?,?)", list(rows))
        self.db.commit()

    @classmethod
    def from_wellset(cls, ws, path=":memory:", bucket=2000.0):
        """Index of a Well Correlation project (headers + tops + LAS file list; logs are not read)."""
        ix = cls(path, bucket)
        wp = ws.wcp
        ix.add_wells((n, w.uwi or None, w.x, w.y, w.elev, w.td, w.survey_path) for n, w in wp.wells.items())
        ix.add_tops(wp.tops or {})
        ix.add_logs((n, os.path.basename(p), None, None, p) for n, w in wp.wells.items() for p in w.las_paths)
        return ix

    # ------------------------------------------------------------ queries
    def in_box(self, x0, x1, y0, y1, limit=None):
        """[(name, x, y)] inside the box: bucket range lookup on the index, then an exact test."""
        x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        q = ("SELECT name, x, y FROM wells WHERE bx BETWEEN ? AND ? AND by BETWEEN ? AND ? "
             "AND x BETWEEN ? AND ? AND y BETWEEN ? AND ?" + (f" LIMIT {int(limit)}" if limit else ""))
        return self.db.execute(q, (self._b(x0), self._b(x1), self._b(y0), self._b(y1), x0, x1, y0, y1)).fetchall()

    def nearest(self, x, y, k=1, max_rings=50):
        """k nearest wells to (x, y): grow the bucket ring until k wells are certainly found."""
        bx, by = self._b(x), self._b(y)
        found = []
        for r in range(max_rings):
            rows = self.db.execute("SELECT name, x, y FROM wells WHERE bx BETWEEN ? AND ? AND by BETWEEN ? AND ?",
                                   (bx - r, bx + r, by - r, by + r)).fetchall()
            if len(rows) >= k:
                found = sorted(rows, key=lambda t: (t[1] - x) ** 2 + (t[2] - y) ** 2)
                # wells in the next ring could still be closer than the k-th found unless it is within r buckets
                if math.hypot(found[k - 1][1] - x, found[k - 1][2] - y) <= r * self.bucket:
                    return found[:k]
        return found[:k]

    def by_api(self, api):
        from .analytics.table import normalise_id
        key = normalise_id(api)
        return [r[0] for r in self.db.execute("SELECT name, api FROM wells").fetchall()
                if r[1] and normalise_id(r[1]) == key]

    def tops_of(self, formation):
        """{well: md} for one formation."""
        return dict(self.db.execute("SELECT well, md FROM tops WHERE formation=?", (formation,)).fetchall())

    def wells_with_curve(self, curve_like):
        return [r[0] for r in self.db.execute("SELECT DISTINCT well FROM logs WHERE curve LIKE ?",
                                              (f"%{curve_like}%",)).fetchall()]

    def headers(self, names=None):
        if names is None:
            rows = self.db.execute("SELECT name, api, x, y, elev, td FROM wells").fetchall()
        else:
            names = list(names)
            rows = []
            for i in range(0, len(names), 900):  # SQLite variable limit
                part = names[i:i + 900]
                rows += self.db.execute(f"SELECT name, api, x, y, elev, td FROM wells WHERE name IN "
                                        f"({','.join('?' * len(part))})", part).fetchall()
        return rows

    def xy(self):
        r = self.db.execute("SELECT name, x, y FROM wells WHERE x IS NOT NULL").fetchall()
        return [a for a, _, _ in r], np.array([[b, c] for _, b, c in r], float).reshape(-1, 2)


class WellLayer:
    """What the basemap draws for the wells, so 10,000 wells stay responsive (pure data, no Qt).

    Every well is one point of a single scatter item (cheap even for 100k points); text labels are the slow part
    in pyqtgraph, so they are made only for wells inside the current view and only when at most `max_labels` wells
    are in view. Below `index_from` wells a numpy box test is used, above it an in-memory WellIndex (grid buckets).
    """

    def __init__(self, wells, max_labels=150, index_from=2000, bucket=None, label_all_below=500):
        wells = [w for w in (wells or []) if w[1] is not None and w[2] is not None
                 and math.isfinite(float(w[1])) and math.isfinite(float(w[2]))]
        self.names = [w[0] for w in wells]
        self.xy = np.array([[float(w[1]), float(w[2])] for w in wells], float).reshape(-1, 2)
        self.max_labels = int(max_labels)
        self.label_all_below = int(label_all_below)  # small projects: every well in view keeps its name
        self.index = None
        if len(wells) >= index_from:
            if bucket is None:  # ~ 30 wells per bucket on average
                span = max(float(np.ptp(self.xy[:, 0])), float(np.ptp(self.xy[:, 1])), 1.0)
                bucket = max(1.0, span * math.sqrt(30.0 / len(wells)))
            self.index = WellIndex(":memory:", bucket=bucket)
            self.index.add_wells((n, None, x, y, None, None) for n, (x, y) in zip(self.names, self.xy))
            self._row = {n: k for k, n in enumerate(self.names)}

    def __len__(self):
        return len(self.names)

    def points(self):
        """(x, y) arrays of every well - one scatter item."""
        return self.xy[:, 0], self.xy[:, 1]

    def in_view(self, x0, x1, y0, y1, limit=False):
        """Row numbers of the wells inside the box (limit: stop early - enough to know there are too many)."""
        if not len(self.names):
            return np.zeros(0, int)
        if self.index is not None:
            hit = self.index.in_box(x0, x1, y0, y1, limit=max(self.max_labels, 1) * 20 if limit else None)
            return np.array(sorted(self._row[r[0]] for r in hit), int)
        x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        m = (self.xy[:, 0] >= x0) & (self.xy[:, 0] <= x1) & (self.xy[:, 1] >= y0) & (self.xy[:, 1] <= y1)
        return np.nonzero(m)[0]

    def labels(self, view=None):
        """[(name, x, y)] to label for view ((x0, x1), (y0, y1)) or None (= everything); [] when too many."""
        if view is None:
            rows = np.arange(len(self.names))
        else:
            (x0, x1), (y0, y1) = view
            rows = self.in_view(x0, x1, y0, y1, limit=True)
        if len(rows) > self.max_labels and len(self.names) > self.label_all_below:
            return []
        return [(self.names[k], float(self.xy[k, 0]), float(self.xy[k, 1])) for k in rows]

    def nearest(self, x, y):
        """Row of the nearest well (None when there are none)."""
        if not len(self.names):
            return None
        if self.index is not None:
            r = self.index.nearest(x, y, 1)
            if r:
                return self._row[r[0][0]]
        return int(np.argmin((self.xy[:, 0] - x) ** 2 + (self.xy[:, 1] - y) ** 2))
