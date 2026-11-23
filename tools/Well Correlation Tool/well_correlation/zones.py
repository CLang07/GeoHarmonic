"""Zone statistics between tops (GeoGraphix ZoneManager style).

A zone runs from a formation top down to the next formation top present in the same well
(formation order from the Tops panel). For each well and zone:
  gross MD and TVD thickness, net sand (Vsh < cutoff, Vsh from GR normalised per well),
  net-to-gross, sand count, average GR / RHOB / NPHI / DT, geometric-mean deep resistivity,
  and average density porosity in the net sand (PHID = (rho_ma - RHOB) / (rho_ma - rho_fl)).
"""
import csv
import numpy as np

from .lithology import well_lithology

COLUMNS = ["Well", "Zone", "Top_MD", "Base_MD", "Gross_MD_m", "Gross_TVD_m", "Net_sand_m", "NtG",
           "Sand_beds", "GR_avg", "RHOB_avg", "NPHI_avg", "DT_avg", "RDEEP_geomean", "PHID_net_avg", "Coverage_%"]


def zone_table(project, wells=None, vsh_cutoff=0.35, rho_ma=2.65, rho_fl=1.0):
    rows = []
    order = [f["name"] for f in project.formations]
    for wn in wells or list(project.wells):
        tops = project.tops.get(wn, {})
        if not tops:
            continue
        w = project.load_well(wn)
        if w.depth is None or not len(w.depth):
            continue
        present = [(f, tops[f]) for f in order if f in tops]
        present.sort(key=lambda kv: kv[1])
        vsh, code = well_lithology(w)
        for (f, top), (g, base) in zip(present[:-1], present[1:]):
            if base <= top:
                continue
            m = (w.depth >= top) & (w.depth < base)
            n = int(m.sum())
            if n == 0:
                continue
            step = float(np.median(np.diff(w.depth))) if len(w.depth) > 1 else 0.1524
            v = vsh[m]
            have = np.isfinite(v)
            sand = have & (v < vsh_cutoff)
            beds = int(np.sum(np.diff(np.concatenate([[0], sand.astype(int)])) == 1))
            tvd = float(np.diff(w.md_to_tvdss([top, base]))[0])

            def avg(fam, sel=None):
                a = w.logs.get(fam)
                if a is None:
                    return None
                a = a[m] if sel is None else a[m][sel]
                a = a[np.isfinite(a)]
                return float(a.mean()) if a.size else None

            rd = w.logs.get("RDEEP")
            rgm = None
            if rd is not None:
                r = rd[m]
                r = r[np.isfinite(r) & (r > 0)]
                rgm = float(np.exp(np.log(r).mean())) if r.size else None
            phid = None
            rho = w.logs.get("RHOB")
            if rho is not None and sand.any():
                rr = rho[m][sand]
                rr = rr[np.isfinite(rr)]
                if rr.size:
                    phid = float(np.clip((rho_ma - rr) / (rho_ma - rho_fl), 0, 0.5).mean())
            gross = base - top
            net = float(sand.sum() * step)
            rows.append({
                "Well": wn, "Zone": f, "Top_MD": round(top, 1), "Base_MD": round(base, 1),
                "Gross_MD_m": round(gross, 1), "Gross_TVD_m": round(tvd, 1), "Net_sand_m": round(net, 1),
                "NtG": round(net / gross, 2) if gross > 0 and have.any() else None, "Sand_beds": beds,
                "GR_avg": _r(avg("GR"), 1), "RHOB_avg": _r(avg("RHOB"), 3), "NPHI_avg": _r(avg("NPHI"), 3),
                "DT_avg": _r(avg("DT"), 1), "RDEEP_geomean": _r(rgm, 2), "PHID_net_avg": _r(phid, 3),
                "Coverage_%": round(100.0 * have.sum() / n, 0),
            })
    return rows


def _r(v, k):
    return None if v is None else round(v, k)


def export_zone_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLUMNS})
