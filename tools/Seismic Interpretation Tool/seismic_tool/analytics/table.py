"""Attribute table: rows (wells or map nodes) x named numeric columns, with joins and CSV IO."""
import csv

import numpy as np


class AttributeTable:
    def __init__(self, ids=None):
        self.ids = list(ids or [])
        self.cols = {}  # name -> float array (len(ids))
        self.xy = None  # optional (n, 2) map positions (spatial CV, maps)
        self.meta = {}

    def __len__(self):
        return len(self.ids)

    @property
    def names(self):
        return list(self.cols)

    def add(self, name, values):
        v = np.asarray(values, float)
        if len(v) != len(self.ids):
            raise ValueError(f"column {name}: {len(v)} values for {len(self.ids)} rows")
        self.cols[name] = v
        return self

    def matrix(self, names=None):
        names = names or self.names
        return np.column_stack([self.cols[n] for n in names]) if names else np.zeros((len(self), 0))

    def subset(self, mask):
        mask = np.asarray(mask)
        idx = np.nonzero(mask)[0] if mask.dtype == bool else mask
        t = AttributeTable([self.ids[i] for i in idx])
        t.cols = {k: v[idx] for k, v in self.cols.items()}
        t.xy = None if self.xy is None else self.xy[idx]
        t.meta = dict(self.meta)
        return t

    def join(self, other, how="left"):
        """Join another table's columns by id (case-insensitive, API/UWI normalised)."""
        key = lambda s: normalise_id(s)  # noqa: E731
        pos = {key(i): k for k, i in enumerate(other.ids)}
        rows = [pos.get(key(i)) for i in self.ids]
        out = AttributeTable(self.ids)
        out.cols = dict(self.cols)
        out.xy = self.xy
        for n, v in other.cols.items():
            col = np.full(len(self.ids), np.nan)
            for r, k in enumerate(rows):
                if k is not None:
                    col[r] = v[k]
            out.cols[n if n not in out.cols else f"{n}_2"] = col
        if how == "inner":
            return out.subset(np.array([k is not None for k in rows]))
        return out

    # -------------------------------------------------------------- builders
    @classmethod
    def from_csv(cls, path, id_col=None):
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = [r for r in csv.DictReader(fh)]
        if not rows:
            return cls()
        id_col = id_col or next(iter(rows[0]))
        t = cls([r[id_col] for r in rows])
        for c in rows[0]:
            if c == id_col:
                continue
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[c]))
                except (TypeError, ValueError):
                    vals.append(np.nan)
            if np.isfinite(vals).any():
                t.add(c, vals)
        return t

    def to_csv(self, path):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id"] + self.names + (["x", "y"] if self.xy is not None else []))
            for k, i in enumerate(self.ids):
                row = [i] + [f"{self.cols[n][k]:.6g}" for n in self.names]
                if self.xy is not None:
                    row += [f"{self.xy[k, 0]:.2f}", f"{self.xy[k, 1]:.2f}"]
                w.writerow(row)
        return path

    @classmethod
    def from_maps_at_wells(cls, wells, maps):
        """wells: [(name, x, y)]; maps: {column: maps.MapGrid} -> one row per well inside any map."""
        t = cls([w[0] for w in wells])
        t.xy = np.array([[w[1], w[2]] for w in wells], float) if wells else np.zeros((0, 2))
        for name, g in maps.items():
            t.add(name, g.sample(t.xy[:, 0], t.xy[:, 1]) if len(wells) else [])
        return t

    @classmethod
    def from_zone_stats(cls, wc_project, zone, wells=None, **kw):
        """Well properties of one zone from the Well Correlation Tool (net sand, NtG, average PHID, GR, ...)."""
        from well_correlation.zones import zone_table
        rows = [r for r in zone_table(wc_project, wells, **kw) if r.get("Zone") == zone]
        t = cls([r["Well"] for r in rows])
        if rows:
            for k, v in rows[0].items():
                if k in ("Well", "Zone"):
                    continue
                try:
                    t.add(k, [float(r[k]) if r[k] not in (None, "") else np.nan for r in rows])
                except (TypeError, ValueError):
                    continue
        xy = []
        for r in rows:
            w = wc_project.wells.get(r["Well"])
            xy.append((w.x if w and w.x is not None else np.nan, w.y if w and w.y is not None else np.nan))
        t.xy = np.array(xy, float) if xy else np.zeros((0, 2))
        return t

    @classmethod
    def from_map_nodes(cls, maps, ref=None, step=1):
        """One row per node of the reference grid (for predicting maps); other maps sampled there."""
        ref = ref or next(iter(maps.values()))
        X, Y = ref.node_xy()
        X, Y = X[::step, ::step], Y[::step, ::step]
        t = cls([f"node{i}" for i in range(X.size)])
        t.xy = np.column_stack([X.ravel(), Y.ravel()])
        t.meta["grid_shape"] = X.shape
        t.meta["ref"] = ref
        t.meta["step"] = step
        for name, g in maps.items():
            t.add(name, g.z[::step, ::step].ravel() if g is ref else g.sample(X.ravel(), Y.ravel()))
        return t


def normalise_id(s):
    """Well id key: API/UWI digits without dashes (API-14 -> API-10 when the tail is zeros), else lower-case
    name without spaces/dashes."""
    s = str(s).strip()
    digits = "".join(c for c in s if c.isdigit())
    if len(digits) >= 10 and len(digits) >= 0.8 * len(s.replace("-", "").replace(" ", "")):
        if len(digits) in (12, 14) and set(digits[10:]) <= {"0"}:
            digits = digits[:10]
        return digits
    return "".join(c for c in s.lower() if c.isalnum())
