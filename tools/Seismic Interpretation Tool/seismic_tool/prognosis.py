"""Well prognosis / geosteering view (PLAN item 17d).

`Prognosis(dm, trajectory)` lists where a planned or drilling well meets every horizon of the dynamic depth model
(ddc.DynamicDepthModel): MD, TVDSS, map position, status (planned / drilled / ahead of the bit / below TD projected),
an uncertainty (1 sigma, m; from the blind test of item 17e when given) and the change since the previous update.
`update()` recomputes after the model changed (the GUI calls it after each dynamic rebuild). While drilling,
`add_top(horizon, md)` turns an encountered top into a what-if pseudo-top at that point of the wellbore, the model
rebuilds from that horizon down, and the horizons ahead of the bit move with it.
"""
import csv

import numpy as np


class Prognosis:
    def __init__(self, dm, trajectory, uncertainty=None, step=2.0):
        self.dm = dm
        self.traj = trajectory.resample(step) if step and len(trajectory.md) > 1 else trajectory
        self.name = trajectory.name
        self.uncertainty = dict(uncertainty or {})  # horizon name -> 1 sigma (m)
        self.bit_md = None
        self.rows = []
        self.history = []

    # ------------------------------------------------------------------ geometry
    def _crossings(self, k):
        """[(md, x, y, z, direction)] where the wellbore passes the model depth surface of horizon k."""
        t = self.traj
        vm = self.dm.vm
        Zk = np.asarray(vm.depth_grid(k).sample(t.x, t.y), float)
        d = t.z - Zk  # > 0: below the horizon
        out = []
        fin = np.isfinite(d)
        for i in range(len(d) - 1):
            if not (fin[i] and fin[i + 1]):
                continue
            a, b = d[i], d[i + 1]
            if (a <= 0 < b) or (a > 0 >= b):
                f = a / (a - b) if a != b else 0.0
                md = t.md[i] + f * (t.md[i + 1] - t.md[i])
                x, y, z = t.at_md(md)
                out.append((float(md), float(x), float(y), float(z), "down" if b > a else "up"))
        if not out and fin.any():
            j = int(np.flatnonzero(fin)[-1])
            if d[j] < 0 and j == len(d) - 1:  # TD above the horizon: project along the last segment
                if len(t.md) > 1:
                    dzdmd = (t.z[-1] - t.z[-2]) / max(t.md[-1] - t.md[-2], 1e-9)
                else:
                    dzdmd = 1.0
                dzdmd = max(dzdmd, 0.05)
                md = t.md[-1] + (Zk[-1] - t.z[-1]) / dzdmd
                out.append((float(md), float(t.x[-1]), float(t.y[-1]), float(Zk[-1]), "below TD"))
        return out

    # ------------------------------------------------------------------ prognosis table
    def compute(self):
        old = {(r["horizon"], r["n"]): r for r in self.rows}
        rows = []
        for k, (h, fm) in enumerate(self.dm.vm.layers):
            for n, (md, x, y, z, dirn) in enumerate(self._crossings(k)):
                if dirn == "below TD":
                    status = "below TD (projected)"
                elif self.bit_md is None:
                    status = "planned"
                else:
                    status = "drilled" if md <= self.bit_md else "ahead"
                prev = old.get((h, n))
                rows.append({"well": self.name, "horizon": h, "formation": fm or "", "n": n, "md": md, "tvdss": z,
                             "x": x, "y": y, "direction": dirn, "status": status,
                             "sigma": self.uncertainty.get(h), "change": (z - prev["tvdss"]) if prev else None})
        self.rows = rows
        return rows

    def update(self):
        """Recompute after a model change; returns rows whose depth changed (|change| > 1 cm)."""
        self.compute()
        moved = [r for r in self.rows if r["change"] is not None and abs(r["change"]) > 0.01]
        if moved:
            self.history.append([(r["horizon"], r["change"]) for r in moved])
        return moved

    def ahead(self):
        """Rows ahead of the bit (or all not yet drilled when no bit depth is set)."""
        return [r for r in self.rows if r["status"] in ("ahead", "planned", "below TD (projected)")]

    # ------------------------------------------------------------------ drilling
    def set_bit(self, md):
        self.bit_md = None if md is None else float(md)
        return self.compute()

    def what_if_name(self, key):
        return f"drilling {self.name}: {key}"

    def add_top(self, key, md):
        """A top met while drilling (horizon or formation name, MD along this wellbore): pseudo-top at that point;
        the model rebuilds from that layer down. Returns the refresh report."""
        x, y, z = (float(a) for a in self.traj.at_md(md))
        rep = self.dm.add_what_if(self.what_if_name(key), x, y, {key: z})
        if self.bit_md is None or md > self.bit_md:
            self.bit_md = float(md)
        self.update()
        return rep

    def clear_tops(self):
        names = [n for n in self.dm.pseudo if n.startswith(f"drilling {self.name}: ")]
        rep = None
        for n in names:
            rep = self.dm.remove_what_if(n)
        self.update()
        return rep

    # ------------------------------------------------------------------ export
    def write_csv(self, path):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["well", "horizon", "formation", "occurrence", "md_m", "tvdss_m", "x", "y", "direction",
                        "status", "sigma_m", "change_m"])
            for r in self.rows:
                w.writerow([r["well"], r["horizon"], r["formation"], r["n"] + 1, f"{r['md']:.2f}", f"{r['tvdss']:.2f}",
                            f"{r['x']:.2f}", f"{r['y']:.2f}", r["direction"], r["status"],
                            "" if r["sigma"] is None else f"{r['sigma']:.2f}",
                            "" if r["change"] is None else f"{r['change']:+.2f}"])
        return path


def vertical_trajectory(name, x, y, td_tvdss, kb=0.0, step=5.0):
    """Planned vertical well at map (x, y) from the MD reference (kb m above sea level) down to td_tvdss."""
    from .laterals import Trajectory
    md_td = td_tvdss + kb
    md = np.arange(0.0, md_td + step / 2, step)
    md[-1] = md_td
    return Trajectory(name, md, np.full(len(md), float(x)), np.full(len(md), float(y)), md - kb, np.zeros(len(md)))
