"""Dashboard data model (PLAN item 14): one table of wells (AttributeTable columns + text columns such as
formation / operator), filters, a shared selection and listeners, so linked panels (map, histogram, crossplot,
table, well strip) stay in sync.  Also histogram / crossplot data, layouts saved as JSON, and a standalone
HTML export.  GUI-free; panels call `select`, `set_filter` and read `visible()` / `selected`."""
import html
import json

import numpy as np


class DashboardModel:
    def __init__(self, table=None, text=None):
        """table: AttributeTable (rows = wells); text: {column: [str per row]} (formation, operator, ...)."""
        self.table = table
        self.text = {k: list(v) for k, v in (text or {}).items()}
        self.filters = {}  # column -> ("range", lo, hi) | ("in", set of values)
        self.selected = set()  # row indices
        self.listeners = []
        self.layout = {"panels": ["map", "histogram", "crossplot", "table"], "hist": None, "x": None, "y": None,
                       "color": None}

    # ------------------------------------------------------------ data
    def __len__(self):
        return 0 if self.table is None else len(self.table)

    @property
    def ids(self):
        return [] if self.table is None else list(self.table.ids)

    def columns(self):
        return ([] if self.table is None else self.table.names) + list(self.text)

    def values(self, col):
        if col in self.text:
            return np.array(self.text[col], dtype=object)
        return self.table.cols[col]

    def add_text(self, col, values):
        if len(values) != len(self):
            raise ValueError(f"{col}: {len(values)} values for {len(self)} wells")
        self.text[col] = list(values)
        self._emit("data")

    def add_column(self, col, values):
        self.table.add(col, values)
        self._emit("data")

    # ------------------------------------------------------------ filters + selection
    def set_filter(self, col, lo=None, hi=None, values=None):
        """Numeric range (lo/hi, None = open) or a set of allowed text values; values=None and lo=hi=None
        removes the filter."""
        if values is not None:
            self.filters[col] = ("in", set(values))
        elif lo is None and hi is None:
            self.filters.pop(col, None)
        else:
            self.filters[col] = ("range", lo, hi)
        self._emit("filter")

    def clear_filters(self):
        self.filters.clear()
        self._emit("filter")

    def visible(self, exclude=None):
        """Boolean mask of rows passing all filters (NaN fails a numeric range); exclude: ignore that column's
        own filter (the histogram of a filtered column shows its full range so the range can be widened)."""
        m = np.ones(len(self), bool)
        for col, f in self.filters.items():
            if col == exclude:
                continue
            v = self.values(col)
            if f[0] == "in":
                m &= np.array([x in f[1] for x in v])
            else:
                v = np.asarray(v, float)
                ok = np.isfinite(v)
                if f[1] is not None:
                    ok &= v >= f[1]
                if f[2] is not None:
                    ok &= v <= f[2]
                m &= ok
        return m

    def select(self, rows, mode="replace"):
        """mode: replace | add | toggle | remove.  Only visible rows can be selected."""
        rows = {int(r) for r in np.atleast_1d(rows)} & set(np.nonzero(self.visible())[0].tolist())
        if mode == "replace":
            self.selected = rows
        elif mode == "add":
            self.selected |= rows
        elif mode == "remove":
            self.selected -= rows
        else:
            self.selected ^= rows
        self._emit("selection")

    def select_ids(self, ids, mode="replace"):
        from .analytics.table import normalise_id
        want = {normalise_id(i) for i in ids}
        self.select([k for k, i in enumerate(self.ids) if normalise_id(i) in want], mode)

    def select_box(self, xcol, ycol, x0, x1, y0, y1, mode="replace"):
        """Rectangle selection on a crossplot (or the map with xcol/ycol = None for the table's x/y)."""
        x, y = self._xy(xcol, ycol)
        lo_x, hi_x, lo_y, hi_y = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        with np.errstate(invalid="ignore"):
            hit = (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)
        self.select(np.nonzero(hit)[0], mode)

    def select_lasso(self, xcol, ycol, poly, mode="replace"):
        x, y = self._xy(xcol, ycol)
        self.select(np.nonzero(points_in_polygon(x, y, poly))[0], mode)

    def select_hist_bin(self, col, lo, hi, mode="replace"):
        v = np.asarray(self.values(col), float)
        with np.errstate(invalid="ignore"):
            self.select(np.nonzero((v >= lo) & (v < hi))[0], mode)

    def _xy(self, xcol, ycol):
        if xcol is None:
            xy = self.table.xy
            return xy[:, 0], xy[:, 1]
        return np.asarray(self.values(xcol), float), np.asarray(self.values(ycol), float)

    def subscribe(self, fn):
        """fn(kind) with kind in data | filter | selection."""
        self.listeners.append(fn)

    def _emit(self, kind):
        if kind == "filter":
            vis = self.visible()
            self.selected = {r for r in self.selected if vis[r]}
        for fn in list(self.listeners):
            fn(kind)

    # ------------------------------------------------------------ panel data
    def histogram(self, col, bins=20):
        """(edges, counts of visible rows, counts of selected rows). The edges span the column's values under the
        other columns' filters (not its own), so its range filter can be widened again."""
        v = np.asarray(self.values(col), float)
        vis = self.visible() & np.isfinite(v)
        base = self.visible(exclude=col) & np.isfinite(v)
        if not base.any():
            return np.array([0.0, 1.0]), np.zeros(1, int), np.zeros(1, int)
        edges = np.histogram_bin_edges(v[base], bins)
        c, _ = np.histogram(v[vis], edges)
        sel = np.zeros(len(v), bool)
        sel[list(self.selected)] = True
        cs, _ = np.histogram(v[vis & sel], edges)
        return edges, c, cs

    def crossplot(self, xcol, ycol, color=None):
        """Visible rows: dict(rows, x, y, c (or None), selected mask, fit (slope, intercept, r))."""
        x, y = np.asarray(self.values(xcol), float), np.asarray(self.values(ycol), float)
        rows = np.nonzero(self.visible() & np.isfinite(x) & np.isfinite(y))[0]
        sel = np.isin(rows, list(self.selected))
        fit = None
        if len(rows) > 2 and np.std(x[rows]) > 0 and np.std(y[rows]) > 0:
            a, b = np.polyfit(x[rows], y[rows], 1)
            fit = (float(a), float(b), float(np.corrcoef(x[rows], y[rows])[0, 1]))
        c = None if color is None else self.values(color)[rows]
        return {"rows": rows, "x": x[rows], "y": y[rows], "c": c, "selected": sel, "fit": fit}

    def table_rows(self, cols=None, sort=None, descending=False, selected_first=False):
        """[(row index, [values])] of visible rows for the table panel."""
        cols = cols or self.columns()
        rows = np.nonzero(self.visible())[0]
        if sort:
            v = self.values(sort)[rows]
            key = np.array([str(a) for a in v]) if sort in self.text else np.where(np.isfinite(v.astype(float)), v, np.inf)
            rows = rows[np.argsort(key, kind="stable")]
            if descending:
                rows = rows[::-1]
        if selected_first:
            rows = np.r_[[r for r in rows if r in self.selected], [r for r in rows if r not in self.selected]].astype(int)
        return [(int(r), [self.values(c)[r] for c in cols]) for r in rows]

    def summary(self, col):
        v = np.asarray(self.values(col), float)[self.visible()]
        v = v[np.isfinite(v)]
        if not len(v):
            return {"n": 0}
        return {"n": int(len(v)), "mean": float(v.mean()), "p10": float(np.percentile(v, 10)),
                "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)), "min": float(v.min()),
                "max": float(v.max())}

    # ------------------------------------------------------------ layouts + export
    def state(self):
        f = {k: (["in", sorted(v[1])] if v[0] == "in" else ["range", v[1], v[2]]) for k, v in self.filters.items()}
        return {"layout": self.layout, "filters": f, "selected_ids": [self.ids[r] for r in sorted(self.selected)]}

    def save_layout(self, path):
        with open(path, "w") as fh:
            json.dump(self.state(), fh, indent=1)

    def load_layout(self, path):
        with open(path) as fh:
            d = json.load(fh)
        self.layout.update(d.get("layout", {}))
        self.filters = {}
        for k, f in d.get("filters", {}).items():
            if k in self.columns():
                self.filters[k] = ("in", set(f[1])) if f[0] == "in" else ("range", f[1], f[2])
        self._emit("filter")
        self.select_ids(d.get("selected_ids", []))

    def export_html(self, path, title="Dashboard", figures=()):
        """Standalone HTML: filters, summary statistics of the visible wells, embedded PNG figures
        [(caption, matplotlib figure)] and the table of visible wells (selected rows highlighted)."""
        import base64
        import io
        parts = [f"<h1>{html.escape(title)}</h1>",
                 f"<p>{int(self.visible().sum())} of {len(self)} wells shown; {len(self.selected)} selected.</p>"]
        if self.filters:
            parts.append("<p>Filters: " + "; ".join(html.escape(f"{k} {v[1:] if v[0] == 'range' else sorted(v[1])}")
                                                       for k, v in self.filters.items()) + "</p>")
        for cap, fig in figures:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            parts.append(f"<h3>{html.escape(cap)}</h3><img src='data:image/png;base64,"
                         f"{base64.b64encode(buf.getvalue()).decode()}'>")
        cols = self.columns()
        num = [c for c in cols if c not in self.text]
        parts.append("<h3>Summary</h3><table><tr><th>column</th><th>n</th><th>mean</th><th>P10</th><th>P50</th>"
                     "<th>P90</th></tr>" + "".join(
                         "<tr><td>{}</td><td>{}</td>{}</tr>".format(html.escape(c), s["n"], "".join(
                             f"<td>{s.get(k, float('nan')):.4g}</td>" for k in ("mean", "p10", "p50", "p90")))
                         for c in num for s in [self.summary(c)]) + "</table>")
        rows = self.table_rows(cols)
        fmt = lambda v: html.escape(str(v)) if isinstance(v, str) else ("" if not np.isfinite(v) else f"{v:.4g}")  # noqa
        parts.append("<h3>Wells</h3><table><tr><th>id</th>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols)
                     + "</tr>" + "".join(
                         "<tr{}><td>{}</td>{}</tr>".format(" class='sel'" if r in self.selected else "",
                                                           html.escape(str(self.ids[r])),
                                                           "".join(f"<td>{fmt(v)}</td>" for v in vals))
                         for r, vals in rows) + "</table>")
        css = ("body{font-family:sans-serif;margin:16px}table{border-collapse:collapse;font-size:12px}"
               "td,th{border:1px solid #ccc;padding:2px 6px}tr.sel{background:#ffe9a8}img{max-width:100%}")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
                     f"<style>{css}</style></head><body>{''.join(parts)}</body></html>")
        return path


def points_in_polygon(x, y, poly):
    """Even-odd rule, vectorised over points.  poly: [(x, y), ...]."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    P = np.asarray(poly, float)
    inside = np.zeros(len(x), bool)
    for (x1, y1), (x2, y2) in zip(P, np.roll(P, -1, 0)):
        with np.errstate(invalid="ignore", divide="ignore"):
            cross = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / np.where(y2 != y1, y2 - y1, 1e-300) + x1)
        inside ^= cross
    return inside & np.isfinite(x) & np.isfinite(y)
