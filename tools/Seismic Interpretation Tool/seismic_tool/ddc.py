"""Dynamic depth conversion (PLAN item 17c, Kingdom DDC-style).

`DynamicDepthModel` wraps a `velocity.VelocityModel` and keeps it current: `refresh()` looks for changed inputs
(formation tops in the linked .wcproj.json, horizon picks of the model's layers, what-if wells) and rebuilds only
the layers at and below the shallowest change - maps above stay bit-identical (they are copied, not recomputed).
Changed wells are re-read, the other wells' table rows are cached. What-if wells are pseudo-tops at a map position,
kept in the seismic project (display["ddc_what_if"]), never written to the well project; a what-if well is used
from its shallowest pseudo-top down, and for deeper layers without a pseudo-top it carries the model's own
interval velocity (so it does not pull those layers).
"""
import hashlib
import os
import time

import numpy as np

from .velocity import VelocityModel, _vint


def horizon_fingerprint(h, key):
    """Hash of a horizon's picks on a survey key ("3D:<vol>" or "2D" = all 2D lines); None if missing."""
    if h is None:
        return None
    hb = hashlib.blake2b(digest_size=16)
    keys = [key] if key.startswith("3D:") else sorted(k for k in h.data if k.startswith("2D:"))
    for k in keys:
        a = h.data.get(k)
        if a is None:
            continue
        a = np.ascontiguousarray(a, np.float32)
        hb.update(k.encode())
        hb.update(str(a.shape).encode())
        hb.update(a.tobytes())
    return hb.hexdigest()


class DynamicDepthModel:
    def __init__(self, project, model):
        self.project = project
        self.vm = model if isinstance(model, VelocityModel) else VelocityModel.from_dict(project, model)
        wi = (project.display or {}).get("ddc_what_if") or {}
        self.pseudo = {n: dict(p) for n, p in (wi.get("wells") or {}).items()} if wi.get("survey") == self.vm.key \
            else {}
        self._tg = None
        self._rows = {}  # real well -> table row (or None when the well is outside the map)
        self._order = []  # well order of the well project
        self._tops = {}  # well -> {formation: md} of the layer formations (snapshot)
        self._hfp = []  # horizon fingerprints per layer
        self._mtime = None
        self.last = None  # last refresh report
        self.history = []

    # ------------------------------------------------------------------ helpers
    def _formations(self):
        return [fm for _, fm in self.vm.layers if fm]

    def _snapshot_tops(self, tops):
        fms = self._formations()
        return {w: {fm: t[fm] for fm in fms if t.get(fm) is not None} for w, t in (tops or {}).items()}

    def _wcproj_mtime(self):
        p = self.project.well_project
        try:
            return os.path.getmtime(p) if p else None
        except OSError:
            return None

    def _time_grids(self):
        tg = self.vm.time_grids()
        for g in tg:  # own copies: picking edits the horizon arrays in place
            g.z = np.array(g.z, copy=True)
        return tg

    def _fingerprints(self):
        hs = self.project.horizons
        return [horizon_fingerprint(hs.get(n), self.vm.key) for n, _ in self.vm.layers]

    def _layer_of(self, key):
        for k, (h, fm) in enumerate(self.vm.layers):
            if key == fm or key == h:
                return k
        return None

    def pseudo_row(self, name):
        """Well-table row of a what-if well: its pseudo-tops (TVDSS) where given; other layers below its shallowest
        pseudo-top follow the current model's interval velocity at that point."""
        p = self.pseudo[name]
        x, y = float(p["x"]), float(p["y"])
        ts = [float(g.sample([x], [y])[0]) for g in self._tg]
        if not all(np.isfinite(ts)):
            return None
        idx = {self._layer_of(k): float(z) for k, z in p["tops"].items() if self._layer_of(k) is not None}
        if not idx:
            return None
        first = min(idx)
        g = self.vm._grids
        zs, src = [], []
        z_prev, t_prev = 0.0, 0.0
        for k, t in enumerate(ts):
            if k in idx:
                z, s = idx[k], "what-if top"
            elif k < first:  # not used for these layers: model depth
                z = float(self.vm.depth_grid(k).sample([x], [y])[0]) if g is not None else np.nan
                s = "model"
            else:
                v = float(self.vm.velocity_grid(k).sample([x], [y])[0]) if g is not None else np.nan
                z, s = z_prev + v * max(t - t_prev, 0.0) / 2000.0, "model velocity"
            zs.append(z)
            src.append(s)
            z_prev, t_prev = z, t
        if not np.all(np.isfinite(zs[first:])):
            return None
        return {"well": f"what-if: {name}", "x": x, "y": y, "t": ts, "z": zs, "vint": _vint(ts, zs), "source": src,
                "pseudo": True, "from_layer": first}

    def _wells(self):
        rows = [self._rows[n] for n in self._order if self._rows.get(n) is not None]
        for n in self.pseudo:
            r = self.pseudo_row(n)
            if r is not None:
                rows.append(r)
        return rows

    # ------------------------------------------------------------------ full build
    def build(self):
        t0 = time.perf_counter()
        ws = self.project.wellset()
        self._tg = self._time_grids()
        self._order = list(ws.names)
        self._rows = {n: self.vm.well_row(n, self._tg) for n in self._order}
        self._tops = self._snapshot_tops(ws.wcp.tops if ws.wcp else {})
        self._hfp = self._fingerprints()
        self._mtime = self._wcproj_mtime()
        real = [self._rows[n] for n in self._order if self._rows.get(n) is not None]
        self.vm.build(time_grids=self._tg, wells=real)
        if self.pseudo:
            first = min((self.pseudo_row(n) or {"from_layer": len(self._tg)})["from_layer"] for n in self.pseudo)
            if first < len(self._tg):
                self.vm.build(from_layer=first, reuse=self.vm._grids, time_grids=self._tg, wells=self._wells())
        self.last = {"changes": ["full build"], "from_layer": 0, "seconds": time.perf_counter() - t0,
                     "message": f"Depth model built ({len(self._tg)} layers, {len(real)} wells) in "
                                f"{time.perf_counter() - t0:.2f} s"}
        return self

    # ------------------------------------------------------------------ dynamic update
    def _top_changes(self, new_tops):
        """[(well, formation, old md, new md)] for the layer formations."""
        old = self._tops
        new = self._snapshot_tops(new_tops)
        out = []
        for w in sorted(set(old) | set(new)):
            a, b = old.get(w, {}), new.get(w, {})
            for fm in set(a) | set(b):
                if a.get(fm) != b.get(fm):
                    out.append((w, fm, a.get(fm), b.get(fm)))
        return out

    def refresh(self, force=False):
        """Rebuild what changed since the last build/refresh. Returns a report {changes, from_layer, layers,
        seconds, message} or None when nothing changed."""
        if self.vm._grids is None or self._tg is None:
            self.build()
            return self.last
        t0 = time.perf_counter()
        K = len(self.vm.layers)
        k0 = K
        changes = []
        reread = set()
        ws = self.project.wellset()
        # horizons
        fps = self._fingerprints()
        hchanged = [k for k in range(K) if fps[k] != self._hfp[k]]
        if hchanged:
            k0 = min(k0, hchanged[0])
            changes += [f"horizon {self.vm.layers[k][0]} changed" for k in hchanged]
        # well tops (+ wells added / removed) in the linked well project
        mt = self._wcproj_mtime()
        new_wcp = None
        if mt is not None and (force or mt != self._mtime):
            from .welltie import wc
            try:
                new_wcp = wc().project.Project.open(self.project.well_project)
            except Exception as e:  # half-written file: try again next time
                self.last = {"changes": [], "from_layer": None, "seconds": 0.0,
                             "message": f"Depth model: cannot read the well project yet ({e})"}
                return None
            tc = self._top_changes(new_wcp.tops)
            layer_of_fm = {fm: k for k, (_, fm) in enumerate(self.vm.layers) if fm}
            for w, fm, a, b in tc:
                k0 = min(k0, layer_of_fm[fm])
                reread.add(w)
                if a is None:
                    changes.append(f"{w} {fm} top added")
                elif b is None:
                    changes.append(f"{w} {fm} top removed")
                else:
                    changes.append(f"{w} {fm} top moved {b - a:+.1f} m")
            added = [n for n in new_wcp.wells if n not in ws.wcp.wells]
            removed = [n for n in ws.wcp.wells if n not in new_wcp.wells]
            for n in added:
                ws.wcp.wells[n] = new_wcp.wells[n]
                reread.add(n)
                changes.append(f"well {n} added")
            for n in removed:
                ws.wcp.wells.pop(n, None)
                changes.append(f"well {n} removed")
            if added or removed:
                k0 = 0
            ws.wcp.tops = new_wcp.tops
            self._tops = self._snapshot_tops(new_wcp.tops)
            self._mtime = mt
        # what-if wells edited elsewhere (another window saved the seismic project display)
        wi = (self.project.display or {}).get("ddc_what_if") or {}
        disk = {n: dict(p) for n, p in (wi.get("wells") or {}).items()} if wi.get("survey") == self.vm.key else {}
        if disk != self.pseudo:
            changed = {n for n in set(disk) | set(self.pseudo) if disk.get(n) != self.pseudo.get(n)}
            ks = [self._layer_of(t) for n in changed for src in (disk, self.pseudo)
                  for t in (src.get(n) or {}).get("tops", {})]
            ks = [k for k in ks if k is not None]
            self.pseudo = disk
            if ks:
                k0 = min(k0, min(ks))
                changes.append("what-if wells changed")
        if k0 >= K and not force:
            return None
        k0 = 0 if force else k0
        if hchanged or force:
            self._tg = self._time_grids()
            reread = set(ws.names)  # well times change
        self._order = list(ws.names)
        for n in list(self._rows):
            if n not in self._order:
                del self._rows[n]
        for n in self._order:
            if n in reread or n not in self._rows:
                self._rows[n] = self.vm.well_row(n, self._tg)
        old = self.vm._grids
        try:
            self.vm.build(from_layer=k0, reuse=old, time_grids=self._tg, wells=self._wells())
        except Exception as e:
            self.vm._grids = old
            self.last = {"changes": changes, "from_layer": k0, "seconds": time.perf_counter() - t0,
                         "message": f"Depth model not updated: {e}"}
            return self.last
        self._hfp = fps
        dt = time.perf_counter() - t0
        what = "; ".join(changes[:3]) + (f" (+{len(changes) - 3} more)" if len(changes) > 3 else "") or "update"
        rng = f"layer {k0 + 1}" if k0 == K - 1 else f"layers {k0 + 1}-{K}"
        self.last = {"changes": changes, "from_layer": k0, "layers": list(range(k0, K)), "seconds": dt,
                     "message": f"{what} -> rebuilt {rng} in {dt:.2f} s"}
        self.history.append(self.last)
        return self.last

    # ------------------------------------------------------------------ what-if wells
    def add_what_if(self, name, x, y, tops, save=True):
        """Pseudo-well at map (x, y) with {formation or horizon name: TVDSS m}; rebuilds from its shallowest top."""
        unknown = [k for k in tops if self._layer_of(k) is None]
        if unknown:
            raise ValueError(f"not a layer of the model: {', '.join(unknown)}")
        if self.vm._grids is None:
            self.build()
        self.pseudo[name] = {"x": float(x), "y": float(y), "tops": {k: float(v) for k, v in tops.items()}}
        r = self.pseudo_row(name)
        if r is None:
            del self.pseudo[name]
            raise ValueError("the what-if position is outside the horizons")
        return self._pseudo_rebuild(min(self._layer_of(k) for k in tops), f"what-if {name} added", save)

    def remove_what_if(self, name=None, save=True):
        """Remove one what-if well (or all with name=None)."""
        names = list(self.pseudo) if name is None else [name]
        layers = [self._layer_of(k) for n in names for k in self.pseudo.get(n, {}).get("tops", {})]
        for n in names:
            self.pseudo.pop(n, None)
        if not layers:
            return None
        return self._pseudo_rebuild(min(layers), f"what-if {'wells' if name is None else name} removed", save)

    def _pseudo_rebuild(self, k0, what, save):
        t0 = time.perf_counter()
        self.vm.build(from_layer=k0, reuse=self.vm._grids, time_grids=self._tg, wells=self._wells())
        dt = time.perf_counter() - t0
        K = len(self.vm.layers)
        self.last = {"changes": [what], "from_layer": k0, "layers": list(range(k0, K)), "seconds": dt,
                     "message": f"{what} -> rebuilt layers {k0 + 1}-{K} in {dt:.2f} s"}
        self.history.append(self.last)
        if save:
            self.save_what_if()
        return self.last

    def save_what_if(self):
        if self.pseudo:
            self.project.display["ddc_what_if"] = {"survey": self.vm.key, "wells": self.pseudo}
        else:
            self.project.display.pop("ddc_what_if", None)
        if getattr(self.project, "path", None):
            self.project.save()
