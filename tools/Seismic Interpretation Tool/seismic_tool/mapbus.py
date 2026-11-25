"""Live Map bus - shared project services every GeoHarmonic tool talks through (PLAN 31a).

Pure Python + numpy, no Qt, so the Well Correlation Tool and the Mistie Tool can import it from the sibling folder
(`import_mapbus()` pattern) and everything is unit-testable headless.

    bus = MapBus()
    bus.subscribe("map", on_event)                       # fn(event dict); never called for its own publishes
    bus.selection.select("wells", ["A-1", "B-2"], source="map")
    bus.filters.set_range("Net pay", 5, None, source="map")
    bus.active.set("inline", 120, source="section", volume="F3")
    bus.notify("tops", ids=["A-1"], source="wellcorr", formation="F0")
    bus.table.set_column("Mistie_ms", {"A-1": 4.0}, owner="mistie")

Events are dicts {seq, topic, source, origin, time, ...}:
    selection  {kind, ids}                 full id list of that kind after the change (idempotent)
    filter     {filters}                   full filter dict {column: spec}
    active     {kind, value, ...extras}    current inline / crossline / timeslice / line / well / horizon / section
    data       {kind, ids, ...info}        DataChanged: picks, tops, grids, misties, velocities, wells, faults,
                                           horizons, layers, welltable

Cross-tool (separate processes): `FileBridge(bus, folder, role)` appends local events to `<folder>/events.jsonl` and
`poll()` re-publishes other processes' events into the local bus; `state.json` keeps the last selection / filters /
active objects so a tool started later catches up.  The WellTable is saved as `<folder>/welltable.csv` + `.json`
(column owners, units); every tool rewrites only its own columns (read-merge-write).  Folder: `bus_folder(...)` -
next to the linked well project (`<wcproj>_mapbus`) so all tools of one study share it.
"""
import ast
import csv
import json
import os
import re
import time
import uuid

import numpy as np

TOPICS = ("selection", "filter", "active", "data")
SELECTION_KINDS = ("wells", "lines", "points", "polygons", "faults", "horizons")
ACTIVE_KINDS = ("volume", "inline", "crossline", "timeslice", "line", "well", "horizon", "section", "fault")
DATA_KINDS = ("picks", "tops", "grids", "misties", "velocities", "wells", "faults", "horizons", "layers",
              "welltable", "volumes", "polygons")
STATE_TOPICS = ("selection", "filter", "active")


def norm_id(s):
    """Well / line id key: case-insensitive, spaces / underscores / dashes collapsed ('Maui 5' == 'maui-5')."""
    return re.sub(r"[\s_\-]+", "-", str(s).strip()).lower()


# ====================================================================== bus
class MapBus:
    def __init__(self, history=200, clock=time.time):
        self._subs = {}
        self._clock = clock
        self.origin = uuid.uuid4().hex[:12]
        self.seq = 0
        self.history = []
        self._hmax = int(history)
        self.enabled = True
        self.errors = []
        self.selection = SelectionService(self)
        self.filters = FilterService(self)
        self.active = ActiveObject(self)
        self.table = WellTable()

    # ---------------------------------------------------------------- subscribe
    def subscribe(self, name, fn, topics=None):
        """fn(event) for the topics given (None = all).  Re-subscribing a name replaces it."""
        tops = None if topics is None else tuple([topics] if isinstance(topics, str) else topics)
        self._subs[name] = (fn, tops)

    def unsubscribe(self, name):
        self._subs.pop(name, None)

    def names(self):
        return list(self._subs)

    # ---------------------------------------------------------------- publish
    def publish(self, topic, source, **payload):
        """Apply to the services' state and deliver to every subscriber except `source`.  Returns the event."""
        if topic not in TOPICS:
            raise ValueError(f"unknown map bus topic {topic!r}")
        self.seq += 1
        ev = {"seq": self.seq, "topic": topic, "source": str(source), "origin": self.origin, "time": self._clock()}
        ev.update(payload)
        return self.deliver(ev)

    def deliver(self, ev):
        """Deliver an event (local or from a FileBridge).  State services update first, then subscribers."""
        self._apply(ev)
        self.history.append(ev)
        if len(self.history) > self._hmax:
            del self.history[: len(self.history) - self._hmax]
        if not self.enabled:
            return ev
        for name, (fn, tops) in list(self._subs.items()):
            if name == ev.get("source") or (tops is not None and ev["topic"] not in tops):
                continue
            if ev.get("_bridge") and name == ev.get("_bridge"):
                continue  # a bridge never gets its own incoming events back
            try:
                fn(ev)
            except Exception as e:  # a broken window must not stop the others
                self.errors.append((name, repr(e)))
                del self.errors[:-20]
        return ev

    def _apply(self, ev):
        t = ev["topic"]
        if t == "selection":
            self.selection._set(ev.get("kind", "wells"), ev.get("ids") or [])
        elif t == "filter":
            self.filters._set(ev.get("filters") or {})
        elif t == "active":
            self.active._set(ev)
        elif t == "data" and ev.get("kind") == "welltable" and ev.get("origin") != self.origin:
            path = ev.get("path")
            if path and os.path.exists(path):
                try:
                    self.table.reload(path)
                except Exception as e:
                    self.errors.append(("welltable", repr(e)))

    def notify(self, kind, ids=None, source="", **info):
        """DataChanged event: kind in DATA_KINDS (others allowed), ids = affected wells / lines / horizons."""
        return self.publish("data", source, kind=str(kind), ids=list(ids or []), **info)

    def last(self, topic=None, kind=None):
        for ev in reversed(self.history):
            if (topic is None or ev["topic"] == topic) and (kind is None or ev.get("kind") == kind):
                return ev
        return None

    def state(self):
        """Snapshot of the state topics (what state.json holds)."""
        return {"selection": {k: sorted(v) for k, v in self.selection.sets.items()},
                "filters": self.filters.filters, "active": self.active.state}


# ====================================================================== services
class SelectionService:
    """Selected wells / lines / points ... shared by every view.  ids keep their display spelling; membership is
    case-insensitive (norm_id)."""

    def __init__(self, bus):
        self.bus = bus
        self.sets = {}

    def _set(self, kind, ids):
        out, seen = [], set()
        for i in ids:
            k = norm_id(i)
            if k not in seen:
                seen.add(k)
                out.append(str(i))
        self.sets[kind] = out

    def get(self, kind="wells"):
        return list(self.sets.get(kind, []))

    def contains(self, kind, i):
        return norm_id(i) in {norm_id(v) for v in self.sets.get(kind, [])}

    def select(self, kind, ids, mode="replace", source=""):
        """mode replace / add / remove / toggle.  Publishes only when the set changed; returns the new list."""
        ids = [str(i) for i in (ids or [])]
        cur = self.get(kind)
        ck = [norm_id(c) for c in cur]
        if mode == "replace":
            new = ids
        elif mode == "add":
            new = cur + [i for i in ids if norm_id(i) not in ck]
        elif mode == "remove":
            drop = {norm_id(i) for i in ids}
            new = [c for c in cur if norm_id(c) not in drop]
        elif mode == "toggle":
            tk = {norm_id(i) for i in ids}
            new = [c for c in cur if norm_id(c) not in tk] + [i for i in ids if norm_id(i) not in ck]
        else:
            raise ValueError(f"selection mode {mode!r}")
        nk = [norm_id(v) for v in new]
        if kind in self.sets and len(set(nk)) == len(set(ck)) and set(nk) == set(ck):
            return cur
        self.bus.publish("selection", source, kind=kind, ids=new)
        return self.get(kind)

    def clear(self, kind=None, source=""):
        for k in ([kind] if kind else list(self.sets)):
            if self.sets.get(k):
                self.bus.publish("selection", source, kind=k, ids=[])


class FilterService:
    """Filters on WellTable columns.  spec: {"min": a, "max": b} (either may be None), {"values": [...]} (categories)
    or {"rule": "expr"} (see eval_rule).  Hidden = fails any active filter; NaN fails a range filter."""

    def __init__(self, bus):
        self.bus = bus
        self.filters = {}

    def _set(self, filters):
        self.filters = {str(k): dict(v) for k, v in (filters or {}).items()}

    def set(self, column, spec, source=""):
        f = dict(self.filters)
        if spec is None:
            f.pop(column, None)
        else:
            f[column] = dict(spec)
        self.bus.publish("filter", source, filters=f)
        return self.filters

    def set_range(self, column, lo=None, hi=None, source=""):
        return self.set(column, {"min": None if lo is None else float(lo), "max": None if hi is None else float(hi)},
                        source)

    def set_values(self, column, values, source=""):
        return self.set(column, {"values": [str(v) for v in values]}, source)

    def set_rule(self, name, expr, source=""):
        parse_rule(expr)  # raise now on a bad expression, not in every view
        return self.set(name, {"rule": str(expr)}, source)

    def remove(self, column, source=""):
        return self.set(column, None, source) if column in self.filters else self.filters

    def clear(self, source=""):
        if self.filters:
            self.bus.publish("filter", source, filters={})

    def mask(self, table, exclude=None):
        """Boolean visibility per table row.  `exclude` = a column whose own filter is ignored (Spotfire: a
        histogram shows its full range while the others filter)."""
        n = len(table)
        m = np.ones(n, bool)
        for col, spec in self.filters.items():
            if col == exclude:
                continue
            if "rule" in spec:
                m &= eval_rule(spec["rule"], table)
                continue
            if col not in table.cols:
                continue
            if "values" in spec:
                allowed = {str(v) for v in spec["values"]}
                m &= np.array([str(v) in allowed for v in table.text(col)], bool)
            else:
                v = table.numeric(col)
                ok = np.isfinite(v)
                if spec.get("min") is not None:
                    ok &= v >= spec["min"]
                if spec.get("max") is not None:
                    ok &= v <= spec["max"]
                m &= ok
        return m

    def visible_ids(self, table):
        m = self.mask(table)
        return [i for i, k in zip(table.ids, m) if k]


class ActiveObject:
    """Current inline / crossline / time slice / 2D line / well / horizon / section line (+ volume)."""

    def __init__(self, bus):
        self.bus = bus
        self.state = {}

    def _set(self, ev):
        k = ev.get("kind")
        if not k:
            return
        d = {key: v for key, v in ev.items() if key not in ("seq", "topic", "origin", "time", "kind", "_bridge")}
        self.state[k] = d

    def get(self, kind, default=None):
        d = self.state.get(kind)
        return default if d is None else d.get("value", default)

    def info(self, kind):
        return dict(self.state.get(kind) or {})

    def set(self, kind, value, source="", force=False, **extra):
        """Publishes unless the value and extras are unchanged (force=True always publishes)."""
        cur = self.state.get(kind)
        new = {"source": str(source), "value": value}
        new.update(extra)
        if not force and cur is not None and _same(cur, new):
            return None
        return self.bus.publish("active", source, kind=kind, value=value, **extra)


def _same(a, b):
    ka = {k: v for k, v in a.items() if k != "source"}
    kb = {k: v for k, v in b.items() if k != "source"}
    try:
        return json.dumps(ka, sort_keys=True, default=str) == json.dumps(kb, sort_keys=True, default=str)
    except Exception:
        return False


# ====================================================================== well table
class WellTable:
    """Project well table: one row per well, columns any tool can add.  Numeric columns are float arrays (NaN =
    missing), text columns lists of str.  meta[col] = {owner, unit, kind numeric|text}."""

    BASE = ("X", "Y")

    def __init__(self, ids=None):
        self.ids = []
        self._key = {}
        self.cols = {}
        self.meta = {}
        if ids:
            self.ensure_ids(ids)

    def __len__(self):
        return len(self.ids)

    def index(self, i):
        return self._key.get(norm_id(i))

    def ensure_ids(self, ids):
        new = []
        for i in ids:
            k = norm_id(i)
            if k not in self._key:
                self._key[k] = len(self.ids)
                self.ids.append(str(i))
                new.append(str(i))
        if new:
            for c, v in self.cols.items():
                if self.meta[c]["kind"] == "numeric":
                    self.cols[c] = np.concatenate([v, np.full(len(new), np.nan)])
                else:
                    self.cols[c] = list(v) + [""] * len(new)
        return new

    def columns(self, kind=None, owner=None):
        return [c for c in self.cols if (kind is None or self.meta[c]["kind"] == kind)
                and (owner is None or self.meta[c].get("owner") == owner)]

    def set_column(self, name, values, owner="", unit="", kind=None):
        """values: {id: value} (wells added as needed, others keep NaN / '') or a sequence aligned with self.ids."""
        if isinstance(values, dict):
            self.ensure_ids(list(values))
            items = values
            if kind is None:
                kind = "numeric" if all(_is_num(v) for v in items.values()) else "text"
            if kind == "numeric":
                arr = np.full(len(self.ids), np.nan)
                for i, v in items.items():
                    arr[self.index(i)] = _num(v)
            else:
                arr = [""] * len(self.ids)
                for i, v in items.items():
                    arr[self.index(i)] = "" if v is None else str(v)
        else:
            seq = list(values)
            if len(seq) != len(self.ids):
                raise ValueError(f"column {name!r}: {len(seq)} values for {len(self.ids)} wells")
            if kind is None:
                kind = "numeric" if all(_is_num(v) for v in seq) else "text"
            arr = np.array([_num(v) for v in seq], float) if kind == "numeric" else ["" if v is None else str(v)
                                                                                       for v in seq]
        self.cols[name] = arr
        self.meta[name] = {"owner": str(owner), "unit": str(unit), "kind": kind}
        return name

    def remove_column(self, name):
        self.cols.pop(name, None)
        self.meta.pop(name, None)

    def numeric(self, name):
        v = self.cols[name]
        if self.meta[name]["kind"] == "numeric":
            return v
        return np.array([_num(s) for s in v], float)

    def text(self, name):
        v = self.cols[name]
        if self.meta[name]["kind"] == "text":
            return list(v)
        return ["" if not np.isfinite(x) else (f"{x:g}") for x in v]

    def value(self, i, name):
        k = self.index(i)
        if k is None or name not in self.cols:
            return None
        return self.cols[name][k]

    def row(self, i):
        k = self.index(i)
        if k is None:
            return None
        return {c: (float(v[k]) if self.meta[c]["kind"] == "numeric" else v[k]) for c, v in self.cols.items()}

    def categories(self, name):
        """Distinct values of a column in first-seen order (text) / sorted (numeric)."""
        if self.meta[name]["kind"] == "numeric":
            v = self.cols[name]
            return sorted(set(float(x) for x in v[np.isfinite(v)]))
        out = []
        for s in self.cols[name]:
            if s not in out:
                out.append(s)
        return out

    def xy(self):
        if "X" not in self.cols or "Y" not in self.cols:
            return np.full(len(self), np.nan), np.full(len(self), np.nan)
        return self.numeric("X"), self.numeric("Y")

    # ---------------------------------------------------------------- files
    def save(self, path, owner=None):
        """Write CSV + .json meta.  owner given: read the file first and replace only that owner's columns (and the
        rows it knows), so tools never overwrite each other.  Atomic replace with retries (Windows)."""
        out = self
        if owner is not None and os.path.exists(path):
            disk = WellTable()
            disk.reload(path)
            disk.ensure_ids(self.ids)
            for c in self.columns(owner=owner):
                disk.cols.pop(c, None)
                vals = {i: self.cols[c][k] for i, k in zip(self.ids, range(len(self.ids)))}
                disk.set_column(c, vals, owner=owner, unit=self.meta[c]["unit"], kind=self.meta[c]["kind"])
            for c in list(disk.cols):  # a column this owner removed
                if disk.meta[c].get("owner") == owner and c not in self.cols:
                    disk.remove_column(c)
            out = disk
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp = path + f".{os.getpid()}.tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Well"] + list(out.cols))
            for k, i in enumerate(out.ids):
                row = [i]
                for c, v in out.cols.items():
                    if out.meta[c]["kind"] == "numeric":
                        row.append("" if not np.isfinite(v[k]) else repr(float(v[k])))
                    else:
                        row.append(v[k])
                w.writerow(row)
        _replace(tmp, path)
        mtmp = _meta_path(path) + f".{os.getpid()}.tmp"
        with open(mtmp, "w", encoding="utf-8") as f:
            json.dump({"meta": out.meta, "saved": time.time()}, f, indent=1)
        _replace(mtmp, _meta_path(path))
        if out is not self:  # adopt the merged view (other owners' columns)
            self.ids, self._key, self.cols, self.meta = out.ids, out._key, out.cols, out.meta
        return path

    def reload(self, path):
        meta = {}
        mp = _meta_path(path)
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                meta = json.load(f).get("meta", {})
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.ids, self._key, self.cols, self.meta = [], {}, {}, {}
        if not rows:
            return self
        head = rows[0][1:]
        ids = [r[0] for r in rows[1:] if r]
        self.ensure_ids(ids)
        for j, c in enumerate(head):
            vals = [r[j + 1] if j + 1 < len(r) else "" for r in rows[1:] if r]
            m = meta.get(c, {})
            kind = m.get("kind") or ("numeric" if all(v == "" or _is_num(v) for v in vals) else "text")
            self.set_column(c, vals, owner=m.get("owner", ""), unit=m.get("unit", ""), kind=kind)
        return self

    @classmethod
    def load(cls, path):
        return cls().reload(path)


def _meta_path(path):
    return os.path.splitext(path)[0] + ".json"


def _replace(tmp, path, tries=20):
    for k in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: a reader has it open
            time.sleep(0.05 * (k + 1))
    os.replace(tmp, path)


def _is_num(v):
    if v is None or v == "":
        return True
    if isinstance(v, (bool, np.bool_)):
        return False
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _num(v):
    if v is None or v == "":
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


# ====================================================================== rules
_ALLOWED = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd, ast.Compare,
            ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq, ast.In, ast.NotIn, ast.Name, ast.Load, ast.Constant,
            ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Tuple, ast.List, ast.Call)
_FUNCS = {"abs": np.abs, "isnan": lambda v: ~np.isfinite(np.asarray(v, float)) if np.asarray(v).dtype != object
          else np.array([s == "" for s in v]), "contains": None}


def _rule_names(expr):
    """'[Net pay] > 5 and Status == "Oil"' -> (python expr, {placeholder: column})."""
    cols = {}

    def sub(m):
        k = f"__c{len(cols)}"
        cols[k] = m.group(1)
        return k
    return re.sub(r"\[([^\]]+)\]", sub, str(expr)), cols


def parse_rule(expr):
    py, cols = _rule_names(expr)
    tree = ast.parse(py.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"rule: {type(node).__name__} not allowed in {expr!r}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
            raise ValueError(f"rule: only {sorted(_FUNCS)} may be called")
    return tree, cols


def eval_rule(expr, table):
    """Evaluate a rule over a WellTable -> bool array.  Column names as identifiers or [Bracketed Name]; numbers,
    'strings', and/or/not, comparisons (chained), in (...), + - * /, abs(), isnan(), contains(col, 'text').
    Missing values compare False (NaN) / '' (text)."""
    tree, cols = parse_rule(expr)
    n = len(table)

    def col(name):
        if name not in table.cols:
            raise KeyError(f"rule: no column {name!r}")
        return table.numeric(name) if table.meta[name]["kind"] == "numeric" else np.array(table.text(name), object)

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in cols:
                return col(cols[node.id])
            if node.id in table.cols:
                return col(node.id)
            if node.id in ("True", "False"):
                return node.id == "True"
            raise KeyError(f"rule: no column {node.id!r}")
        if isinstance(node, (ast.Tuple, ast.List)):
            return [ev(e) for e in node.elts]
        if isinstance(node, ast.BoolOp):
            vals = [np.asarray(ev(v), bool) for v in node.values]
            out = vals[0]
            for v in vals[1:]:
                out = (out & v) if isinstance(node.op, ast.And) else (out | v)
            return out
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand)
            if isinstance(node.op, ast.Not):
                return ~np.asarray(v, bool)
            return -np.asarray(v, float) if isinstance(node.op, ast.USub) else np.asarray(v, float)
        if isinstance(node, ast.BinOp):
            a, b = np.asarray(ev(node.left), float), np.asarray(ev(node.right), float)
            with np.errstate(divide="ignore", invalid="ignore"):
                return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b, ast.Div: a / b}[type(node.op)]
        if isinstance(node, ast.Call):
            name = node.func.id
            args = [ev(a) for a in node.args]
            if name == "contains":
                return np.array([str(args[1]).lower() in str(s).lower() for s in args[0]], bool)
            return _FUNCS[name](*args)
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            out = np.ones(n, bool)
            for op, rn in zip(node.ops, node.comparators):
                right = ev(rn)
                out &= _cmp(op, left, right, n)
                left = right
            return out
        raise ValueError(f"rule: cannot evaluate {type(node).__name__}")

    r = ev(tree)
    r = np.asarray(r)
    if r.shape == ():
        r = np.full(n, bool(r))
    return r.astype(bool)


def _cmp(op, a, b, n):
    if isinstance(op, (ast.In, ast.NotIn)):
        vals = b if isinstance(b, list) else [b]
        sv = {str(v) for v in vals}
        fv = {float(v) for v in vals if _is_num(v) and v != ""}
        aa = a if isinstance(a, np.ndarray) else np.full(n, a, object)
        res = np.array([(str(x) in sv) or (_is_num(x) and x != "" and np.isfinite(_num(x)) and _num(x) in fv)
                        for x in aa], bool)
        return ~res if isinstance(op, ast.NotIn) else res
    text = any(isinstance(x, str) or (isinstance(x, np.ndarray) and x.dtype == object) for x in (a, b))
    if text:
        aa = a if isinstance(a, np.ndarray) else np.full(n, a, object)
        bb = b if isinstance(b, np.ndarray) else np.full(n, b, object)
        if isinstance(op, (ast.Eq, ast.NotEq)):
            res = np.array([str(x) == str(y) for x, y in zip(aa, bb)], bool)
            return res if isinstance(op, ast.Eq) else ~res
        fa = np.array([_num(x) for x in aa], float)
        fb = np.array([_num(y) for y in bb], float)
        a, b = fa, fb
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    with np.errstate(invalid="ignore"):
        res = {ast.Gt: a > b, ast.GtE: a >= b, ast.Lt: a < b, ast.LtE: a <= b, ast.Eq: a == b,
               ast.NotEq: (a != b) & np.isfinite(a) & np.isfinite(b)}[type(op)]
    return np.broadcast_to(res, (n,)).copy()


# ====================================================================== cross-process
def bus_folder(seis_project_path=None, wcproj_path=None, create=True):
    """Shared folder of a study: `<wcproj stem>_mapbus` next to the well project when one is linked (all tools of a
    study share the well project), else `<seisproj stem>_seismic/mapbus`."""
    if wcproj_path:
        p = str(wcproj_path)
        stem = p[: -len(".wcproj.json")] if p.endswith(".wcproj.json") else os.path.splitext(p)[0]
        folder = stem + "_mapbus"
    elif seis_project_path:
        p = str(seis_project_path)
        stem = p[: -len(".seisproj.json")] if p.endswith(".seisproj.json") else os.path.splitext(p)[0]
        folder = os.path.join(stem + "_seismic", "mapbus")
    else:
        return None
    if create:
        os.makedirs(folder, exist_ok=True)
    return folder


class FileBridge:
    """Connects a MapBus to the shared folder.  Local events -> events.jsonl (one JSON per line); poll() delivers
    events of other processes.  A journal over max_bytes is restarted (readers notice the shrink)."""

    def __init__(self, bus, folder, role="tool", max_bytes=2_000_000, catch_up=True):
        self.bus = bus
        self.folder = folder
        self.role = str(role)
        self.name = f"bridge:{self.role}"
        self.max_bytes = int(max_bytes)
        os.makedirs(folder, exist_ok=True)
        self.events_path = os.path.join(folder, "events.jsonl")
        self.state_path = os.path.join(folder, "state.json")
        self.table_path = os.path.join(folder, "welltable.csv")
        if not os.path.exists(self.events_path):
            self._new_journal()
        self._journal = self._journal_id()
        self._offset = os.path.getsize(self.events_path)
        self._delivering = False
        self.received = 0
        self.written = 0
        self.errors = []
        bus.subscribe(self.name, self._on_local)
        if os.path.exists(self.table_path):
            try:
                bus.table.reload(self.table_path)
            except Exception as e:
                self.errors.append(repr(e))
        if catch_up:
            self.catch_up()

    def close(self):
        self.bus.unsubscribe(self.name)

    # ---------------------------------------------------------------- out
    def _on_local(self, ev):
        if ev.get("origin") != self.bus.origin:
            return  # only this process's own events go out (incl. ones its windows publish in reaction to others)
        rec = {k: v for k, v in ev.items() if not k.startswith("_")}
        rec["role"] = self.role
        try:
            line = json.dumps(rec, default=_json_default)
        except Exception as e:
            self.errors.append(repr(e))
            return
        try:
            if not os.path.exists(self.events_path) or os.path.getsize(self.events_path) > self.max_bytes:
                self._new_journal()
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self.written += 1
            if ev["topic"] in STATE_TOPICS:
                self.write_state()
        except OSError as e:
            self.errors.append(repr(e))

    def _new_journal(self):
        """(Re)start the journal; its first line names it so readers notice a restart even when the new file has
        grown past their old offset."""
        jid = uuid.uuid4().hex
        with open(self.events_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"journal": jid}) + "\n")
        self._journal = jid
        self._offset = os.path.getsize(self.events_path)

    def _journal_id(self):
        try:
            with open(self.events_path, "rb") as f:
                head = f.readline()
            return json.loads(head.decode("utf-8")).get("journal") if head.endswith(b"\n") else None
        except Exception:
            return None

    def write_state(self):
        tmp = self.state_path + f".{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.bus.state(), f, default=_json_default)
        _replace(tmp, self.state_path)

    def save_table(self, owner):
        """Write this tool's WellTable columns (merged with the others') and tell the other tools."""
        self.bus.table.save(self.table_path, owner=owner)
        return self.bus.notify("welltable", source=self.role, path=self.table_path, owner=owner)

    # ---------------------------------------------------------------- in
    def poll(self):
        """Deliver new events written by other processes.  Returns the list delivered."""
        if not os.path.exists(self.events_path):
            return []
        size = os.path.getsize(self.events_path)
        jid = self._journal_id()
        if jid != self._journal or size < self._offset:  # journal restarted by another tool: read it from the top
            self._journal = jid
            self._offset = 0
        if size == self._offset:
            return []
        with open(self.events_path, "rb") as f:
            f.seek(self._offset)
            data = f.read()
        end = data.rfind(b"\n")
        if end < 0:
            return []  # a line still being written
        self._offset += end + 1
        out = []
        for raw in data[: end + 1].splitlines():
            try:
                ev = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            if ev.get("origin") == self.bus.origin or ev.get("topic") not in TOPICS:
                continue
            out.append(self._deliver(ev))
        return out

    def _deliver(self, ev):
        ev = dict(ev)
        ev["_bridge"] = self.name
        self._delivering = True
        try:
            self.bus.deliver(ev)
        finally:
            self._delivering = False
        self.received += 1
        return ev

    def catch_up(self):
        """Apply state.json (selection / filters / active of the other tools) to the local bus and its subscribers."""
        if not os.path.exists(self.state_path):
            return 0
        try:
            with open(self.state_path, encoding="utf-8") as f:
                st = json.load(f)
        except Exception:
            return 0
        n = 0
        base = {"origin": "state", "source": "state", "time": time.time(), "seq": 0, "replay": True}
        for kind, ids in (st.get("selection") or {}).items():
            self._deliver(dict(base, topic="selection", kind=kind, ids=ids))
            n += 1
        if st.get("filters"):
            self._deliver(dict(base, topic="filter", filters=st["filters"]))
            n += 1
        for kind, d in (st.get("active") or {}).items():
            ev = dict(base, topic="active", kind=kind)
            ev.update({k: v for k, v in d.items() if k != "source"})
            ev["source"] = d.get("source", "state")
            self._deliver(ev)
            n += 1
        return n


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


# ====================================================================== helpers for the tools
def table_from_wells(wells, table=None):
    """[(name, x, y)] -> WellTable with X, Y (owner 'project')."""
    t = table if table is not None else WellTable()
    wells = list(wells or [])
    if wells:
        t.set_column("X", {w[0]: w[1] for w in wells}, owner="project", unit="m", kind="numeric")
        t.set_column("Y", {w[0]: w[2] for w in wells}, owner="project", unit="m", kind="numeric")
    return t


def tops_columns(table, tops, owner="wellcorr", prefix="Top "):
    """tops: {well: {formation: depth}} -> one numeric column per formation ('Top F0')."""
    fms = []
    for d in tops.values():
        for fm in d:
            if fm not in fms:
                fms.append(fm)
    for fm in fms:
        table.set_column(prefix + fm, {w: d.get(fm) for w, d in tops.items()}, owner=owner, unit="m", kind="numeric")
    return [prefix + fm for fm in fms]

