"""Formation tops: editing operations (undoable), import and export.

Every edit goes through apply_edit(), which returns the inverse edit, so the GUI can keep a
simple undo/redo stack.  Picks keep their provenance in project.tops_meta:
source = "published" | "auto" | "user", plus score/confidence for auto picks.
"""
import csv
import time

PALETTE = ["#3a8dde", "#e8a33d", "#d4572a", "#6b6b6b", "#2ca02c", "#b8860b", "#8b4513", "#7b2d8e",
           "#17becf", "#e377c2", "#bcbd22", "#1f77b4"]


# ------------------------------------------------------------------ edits
def snapshot_top(project, well, fm):
    md = project.tops.get(well, {}).get(fm)
    meta = project.tops_meta.get(well, {}).get(fm)
    return {"op": "set_top", "well": well, "fm": fm, "md": md, "meta": dict(meta) if meta else None}


def apply_edit(project, edit):
    """Apply an edit dict; return the inverse edit."""
    op = edit["op"]
    if op == "set_top":
        inv = snapshot_top(project, edit["well"], edit["fm"])
        w, f, md = edit["well"], edit["fm"], edit["md"]
        if md is None:
            project.tops.get(w, {}).pop(f, None)
            project.tops_meta.get(w, {}).pop(f, None)
        else:
            project.tops.setdefault(w, {})[f] = round(float(md), 2)
            meta = edit.get("meta") or {"source": "user", "time": time.strftime("%Y-%m-%d %H:%M")}
            project.tops_meta.setdefault(w, {})[f] = meta
        return inv
    if op == "batch":  # several edits as one undo step
        invs = [apply_edit(project, e) for e in edit["edits"]]
        return {"op": "batch", "edits": invs[::-1], "fm": edit.get("fm")}
    if op == "formations":  # whole formation list + tops (rename / delete / reorder / colour)
        inv = {"op": "formations", "formations": [dict(f) for f in project.formations],
               "tops": {w: dict(t) for w, t in project.tops.items()},
               "tops_meta": {w: {f: dict(m) for f, m in t.items()} for w, t in project.tops_meta.items()}}
        project.formations = [dict(f) for f in edit["formations"]]
        project.tops = {w: dict(t) for w, t in edit["tops"].items()}
        project.tops_meta = {w: {f: dict(m) for f, m in t.items()} for w, t in edit["tops_meta"].items()}
        return inv
    raise ValueError(op)


def set_top(project, well, fm, md, source="user"):
    meta = {"source": source, "time": time.strftime("%Y-%m-%d %H:%M")}
    return apply_edit(project, {"op": "set_top", "well": well, "fm": fm, "md": md, "meta": meta})


def set_tops_batch(project, items, source="auto"):
    """items: [(well, fm, md, extra_meta_dict)] applied as one undoable edit."""
    stamp = time.strftime("%Y-%m-%d %H:%M")
    edits = []
    for well, fm, md, extra in items:
        meta = {"source": source, "time": stamp}
        meta.update(extra or {})
        edits.append({"op": "set_top", "well": well, "fm": fm, "md": md, "meta": meta})
    fms = {e["fm"] for e in edits}
    return apply_edit(project, {"op": "batch", "edits": edits, "fm": fms.pop() if len(fms) == 1 else None})


def _formations_state(project):
    return {"op": "formations", "formations": [dict(f) for f in project.formations],
            "tops": {w: dict(t) for w, t in project.tops.items()},
            "tops_meta": {w: {f: dict(m) for f, m in t.items()} for w, t in project.tops_meta.items()}}


def add_formation(project, name, color=None, index=None):
    st = _formations_state(project)
    if any(f["name"] == name for f in project.formations):
        raise ValueError(f"'{name}' already exists")
    color = color or PALETTE[len(project.formations) % len(PALETTE)]
    fl = st["formations"]
    fl.insert(len(fl) if index is None else index, {"name": name, "color": color})
    return apply_edit(project, st)


def rename_formation(project, old, new):
    if any(f["name"] == new for f in project.formations):
        raise ValueError(f"'{new}' already exists")
    st = _formations_state(project)
    for f in st["formations"]:
        if f["name"] == old:
            f["name"] = new
    for d in (st["tops"], st["tops_meta"]):
        for w in d:
            if old in d[w]:
                d[w][new] = d[w].pop(old)
    if project.display.get("hang") == "FLAT:" + old:
        project.display["hang"] = "FLAT:" + new
    return apply_edit(project, st)


def delete_formation(project, name):
    st = _formations_state(project)
    st["formations"] = [f for f in st["formations"] if f["name"] != name]
    for d in (st["tops"], st["tops_meta"]):
        for w in d:
            d[w].pop(name, None)
    if project.display.get("hang") == "FLAT:" + name:
        project.display["hang"] = "TVDSS"
    return apply_edit(project, st)


def move_formation(project, name, delta):
    st = _formations_state(project)
    fl = st["formations"]
    i = next(k for k, f in enumerate(fl) if f["name"] == name)
    j = max(0, min(len(fl) - 1, i + delta))
    fl.insert(j, fl.pop(i))
    return apply_edit(project, st)


def recolor_formation(project, name, color):
    st = _formations_state(project)
    for f in st["formations"]:
        if f["name"] == name:
            f["color"] = color
    return apply_edit(project, st)


# ------------------------------------------------------------------ export
def _rows(project):
    for wn, tops in project.tops.items():
        w = project.wells.get(wn)
        for fm, md in sorted(tops.items(), key=lambda kv: kv[1]):
            meta = project.tops_meta.get(wn, {}).get(fm, {})
            try:
                z = float(w.md_to_tvdss([md])[0]) if w is not None else None
            except Exception:
                z = None
            yield wn, w, fm, md, z, meta


def export_csv(project, path):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Well", "UWI", "Formation", "MD_m", "TVDSS_m", "X_NZTM", "Y_NZTM", "Source", "Confidence", "Score"])
        for wn, w, fm, md, z, meta in _rows(project):
            wr.writerow([wn, getattr(w, "uwi", ""), fm, f"{md:.2f}", "" if z is None else f"{z:.2f}",
                         "" if not w or w.x is None else f"{w.x:.1f}", "" if not w or w.y is None else f"{w.y:.1f}",
                         meta.get("source", ""), meta.get("confidence", ""), meta.get("score", "")])


def export_opendtect(project, path):
    """Tab-separated: well name, marker name, MD (m). Use OpendTect's multi-well marker import
    and map the three columns (depth = MD, metres)."""
    with open(path, "w", newline="") as f:
        f.write("Well\tMarker\tMD_m\n")
        for wn, w, fm, md, z, meta in _rows(project):
            f.write(f"{wn}\t{fm}\t{md:.2f}\n")


def export_kingdom(project, path):
    """Tab-separated: UWI, well name, formation, MD (m), TVDSS (m). Map the columns in Kingdom's
    formation-tops ASCII import (UWI or well name as the key)."""
    with open(path, "w", newline="") as f:
        f.write("UWI\tWell\tFormation\tMD_m\tTVDSS_m\n")
        for wn, w, fm, md, z, meta in _rows(project):
            f.write(f"{getattr(w, 'uwi', '')}\t{wn}\t{fm}\t{md:.2f}\t{'' if z is None else f'{z:.2f}'}\n")


def import_csv(project, path, source="imported"):
    """Read a CSV/TSV with columns Well, Formation, MD (header names are matched loosely).
    Unknown formations are added at the end. Returns (n_tops, unknown_wells)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        rows = list(csv.reader(f, dialect))
    head = [h.strip().lower() for h in rows[0]]

    def col(*keys):
        for k in keys:
            for i, h in enumerate(head):
                if h == k or h.startswith(k):
                    return i
        return None

    iw, ifm, imd = col("well", "wellname", "name"), col("formation", "marker", "top", "surface"), col("md", "depth")
    if None in (iw, ifm, imd):
        raise ValueError("Need columns Well, Formation (or Marker) and MD (or Depth)")
    names = {w.lower(): w for w in project.wells}
    st = _formations_state(project)
    known = {f["name"] for f in st["formations"]}
    n, unknown = 0, set()
    stamp = time.strftime("%Y-%m-%d %H:%M")
    for r in rows[1:]:
        if len(r) <= max(iw, ifm, imd) or not r[imd].strip():
            continue
        wn = names.get(r[iw].strip().lower(), r[iw].strip())
        if wn not in project.wells:
            unknown.add(wn)
        fm = r[ifm].strip()
        if fm not in known:
            st["formations"].append({"name": fm, "color": PALETTE[len(st["formations"]) % len(PALETTE)]})
            known.add(fm)
        st["tops"].setdefault(wn, {})[fm] = float(r[imd])
        st["tops_meta"].setdefault(wn, {})[fm] = {"source": source, "time": stamp, "file": path.split("\\")[-1].split("/")[-1]}
        n += 1
    inv = apply_edit(project, st)
    return n, sorted(unknown), inv


# ------------------------------------------------------------------ reference sets
def compare_with_reference(project, ref_name, tol=10.0):
    """Rows comparing current tops with a reference set: well, formation, current, reference, diff, flag."""
    ref = project.reference_tops.get(ref_name, {}).get("tops", {})
    rows = []
    for wn in sorted(set(ref) | set(project.tops)):
        if wn not in ref:
            continue
        for f in [f["name"] for f in project.formations] + sorted(set(ref.get(wn, {})) - {f["name"] for f in project.formations}):
            r = ref.get(wn, {}).get(f)
            c = project.tops.get(wn, {}).get(f)
            if r is None and c is None:
                continue
            diff = None if (r is None or c is None) else c - r
            meta = project.tops_meta.get(wn, {}).get(f, {})
            flag = ("missing" if c is None else "not in reference" if r is None
                    else "moved" if abs(diff) > tol else "ok")
            rows.append({"Well": wn, "Formation": f, "Current_MD": c, "Reference_MD": r,
                         "Diff_m": None if diff is None else round(diff, 1), "Source": meta.get("source", ""),
                         "Flag": flag})
    return rows


def add_reference_from_current(project, name, notes=""):
    project.reference_tops[name] = {"tops": {w: dict(t) for w, t in project.tops.items()}, "notes": notes}
