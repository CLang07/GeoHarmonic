"""Build a GeoHarmonic project from the F3 Netherlands demo data (dGB Earth Sciences / TerraNubis, CC BY-SA 3.0).

    python setup_f3.py [F3 folder] [output folder]

Defaults: F3 folder = %USERPROFILE%\\Downloads\\F3_Demo_2023\\F3_Demo_2023 (the one holding .survey and Rawdata),
output = %USERPROFILE%\\Documents\\F3 Netherlands.  Uses only the raw exchange files in F3's Rawdata folder
(SEG-Y, LAS, markers, checkshots, horizon and fault text files) plus the .survey file for the exact bin grid.

What it builds in the output folder:
    wells_header.csv, wells/<Well>/las, wells/<Well>/checkshot      Well Correlation data folder (X/Y in ED50 UTM 31N)
    F3 Netherlands.wcproj.json                                       wells + F3 markers as tops + one section
    seismic/F3_seismic.sgy  ->  F3 Netherlands_seismic/F3 original   3D store (inline / crossline / time-slice)
    F3 Netherlands.seisproj.json                                     3D volume + 7 F3 horizons + Fault A + linked wells
    raw/                                                             markers, velocity functions, survey file
Safe to run again: existing wells / projects are rebuilt; the 3D import is skipped if the store is already there.
"""
import csv
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "Seismic Interpretation Tool"))
sys.path.insert(0, os.path.join(HERE, "Well Correlation Tool"))

VOL = "F3 original"
NAME = "F3 Netherlands"
CRS = "ED50 / UTM zone 31N (EPSG:23031)"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
          "#17becf", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#e6550d", "#31a354",
          "#756bb1", "#636363"]


_LOG = []


def say(msg):
    print(msg, flush=True)
    _LOG.append(str(msg))
    if len(_LOG) > 1 and _LOG[0].startswith("@"):
        with open(_LOG[0][1:], "a", encoding="utf-8") as fh:
            fh.write(str(msg) + "\n")


# ------------------------------------------------------------------------------------------------ readers
def read_survey(path):
    """OpendTect .survey: ranges and the exact bin grid (x = a + b*il + c*xl), CRS."""
    d = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if ":" in line:
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
    f = lambda s: [float(t) for t in s.split("`")[:3]]
    il = [int(float(t)) for t in d["In-line range"].split("`")[:3]]
    xl = [int(float(t)) for t in d["Cross-line range"].split("`")[:3]]
    z = d["Z range"].split("`")
    return {"il": il, "xl": xl, "z": [float(z[0]), float(z[1]), float(z[2])], "ax": f(d["Coord-X-BinID"]),
            "ay": f(d["Coord-Y-BinID"]), "epsg": d.get("Coordinate System.Projection.ID", "").replace("EPSG`", "")}


def read_track(path):
    """X Y TVDSS MD rows (TVDSS negative above sea level)."""
    rows = [list(map(float, l.split())) for l in open(path) if l.strip()]
    x, y = rows[0][0], rows[0][1]
    kb = -rows[0][2] if rows[0][3] == 0 else 0.0
    return x, y, kb, max(r[3] for r in rows)


def read_markers(path):
    out = []
    for l in open(path, encoding="utf-8", errors="replace"):
        parts = l.strip().split(None, 1) if "\t" not in l else l.strip().split("\t", 1)
        if len(parts) == 2:
            try:
                out.append((float(parts[0]), parts[1].strip()))
            except ValueError:
                pass
    return out


def read_td(path):
    """MD (m below KB) / TWT (s) pairs."""
    rows = []
    for l in open(path):
        p = l.split()
        if len(p) >= 2:
            try:
                rows.append((float(p[0]), float(p[1])))
            except ValueError:
                pass
    return rows


# ------------------------------------------------------------------------------------------------ steps
def build_wells(src, dst):
    raw = os.path.join(src, "Rawdata", "Well_data", "All_wells_RawData", "All_wells_RawData")
    if not os.path.isdir(raw):  # the zip next to it may not have been unpacked
        z = os.path.join(src, "Rawdata", "Well_data", "All_wells_RawData.zip")
        if os.path.exists(z):
            import zipfile
            zipfile.ZipFile(z).extractall(os.path.join(src, "Rawdata", "Well_data", "All_wells_RawData"))
    names = sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(raw, "Track", "*.track")))
    if not names:
        raise SystemExit(f"No well tracks found under {raw}")
    hdr, tops = [], {}
    for w in names:
        x, y, kb, td = read_track(os.path.join(raw, "Track", w + ".track"))
        wd = os.path.join(dst, "wells", w)
        os.makedirs(os.path.join(wd, "las"), exist_ok=True)
        os.makedirs(os.path.join(wd, "checkshot"), exist_ok=True)
        las = glob.glob(os.path.join(raw, "Lasfiles", w + "*.las"))
        for f in las:
            shutil.copy2(f, os.path.join(wd, "las", os.path.basename(f)))
        cs = os.path.join(raw, "Checkshot", w + "_TD.txt")
        if os.path.exists(cs):
            with open(os.path.join(wd, "checkshot", w + "_checkshot.csv"), "w", newline="") as fh:
                fh.write(f"# {w} checkshot from F3 Rawdata (MD below KB {kb:g} m, vertical well), TWT from MSL\n")
                cw = csv.writer(fh)
                cw.writerow(["MD_m", "TVDSS_m", "TWT_ms"])
                for md, t in read_td(cs):
                    cw.writerow([f"{md:g}", f"{md - kb:.2f}", f"{t * 1000:.1f}"])
        mk = os.path.join(raw, "Tops", w + "_markers.txt")
        tops[w] = [(md, n) for md, n in (read_markers(mk) if os.path.exists(mk) else []) if n.lower() != "seasurface"]
        hdr.append({"Well": w, "UWI": "", "X": f"{x:.1f}", "Y": f"{y:.1f}", "Latitude": "", "Longitude": "",
                    "Elevation_m": f"{kb:g}", "Elev_Ref": "KB", "TD_m": f"{td:g}", "CRS": CRS,
                    "Source": "F3 Demo 2023 (dGB / TerraNubis, CC BY-SA 3.0) Rawdata/Well_data"})
        say(f"  {w}: X {x:.0f} Y {y:.0f}, KB {kb:g} m, {len(las)} LAS, {len(tops[w])} markers"
            f"{', checkshot' if os.path.exists(cs) else ''}")
    with open(os.path.join(dst, "wells_header.csv"), "w", newline="") as fh:
        cw = csv.DictWriter(fh, fieldnames=list(hdr[0]))
        cw.writeheader()
        cw.writerows(hdr)
    rawd = os.path.join(dst, "raw")
    os.makedirs(rawd, exist_ok=True)
    for f in glob.glob(os.path.join(raw, "Tops", "*")):
        shutil.copy2(f, rawd)
    with open(os.path.join(dst, "tops_f3_markers.csv"), "w", newline="") as fh:
        cw = csv.writer(fh)
        cw.writerow(["Well", "Formation", "MD_m", "Source"])
        for w, lst in tops.items():
            for md, n in lst:
                cw.writerow([w, n, md, "F3 markers"])
    return names, tops


def build_well_project(dst, names, tops):
    from well_correlation.project import Project
    p = Project.from_folder(dst)
    p.name = NAME
    # formations in stratigraphic order: median subsea depth over the wells that have them
    order = {}
    for w, lst in tops.items():
        kb = p.wells[w].elev or 0.0
        for md, n in lst:
            order.setdefault(n, []).append(md - kb)
    fms = sorted(order, key=lambda n: (np.median(order[n]), n))
    p.formations = [{"name": n, "color": COLORS[i % len(COLORS)]} for i, n in enumerate(fms)]
    p.tops = {w: {n: md for md, n in lst} for w, lst in tops.items()}
    p.tops_meta = {w: {n: {"source": "published", "note": "F3 Demo well markers (MD below KB)"} for _, n in lst}
                   for w, lst in tops.items()}
    # west -> east section through all wells
    xs = sorted(names, key=lambda w: p.wells[w].x if p.wells[w].x is not None else 0)
    p.sections = {"F3 wells W-E": xs}
    p.current_section = "F3 wells W-E"
    try:
        p.display.update({"tracks": ["LITH", "GR", "DN", "DT"], "hang": "TVDSS"})
        p.display.setdefault("map", {}).update({"kind": "structure", "top": "FS6", "method": "tps"})
    except Exception:
        pass
    path = os.path.join(dst, NAME + ".wcproj.json")
    p.save(path)
    say(f"  {os.path.basename(path)}: {len(p.wells)} wells, {len(fms)} formations")
    return path


def build_volume(src, dst, survey):
    from seismic_tool.volume import import_3d
    segs = sorted(glob.glob(os.path.join(src, "Rawdata", "*.sgy")) + glob.glob(os.path.join(src, "Rawdata", "*.segy")),
                  key=os.path.getsize, reverse=True)
    if not segs:
        raise SystemExit("No SEG-Y in the F3 Rawdata folder")
    seis_dir = os.path.join(dst, "seismic")
    os.makedirs(seis_dir, exist_ok=True)
    segy = os.path.join(seis_dir, "F3_seismic.sgy")
    if not (os.path.exists(segy) and os.path.getsize(segy) == os.path.getsize(segs[0])):
        say(f"  copying {os.path.basename(segs[0])} ({os.path.getsize(segs[0]) / 1e6:.0f} MB)...")
        shutil.copy2(segs[0], segy)
    store = os.path.join(dst, NAME + "_seismic", VOL)
    meta_p = os.path.join(store, "meta.json")
    if os.path.exists(meta_p):
        say("  3D store already there - import skipped")
    else:
        t0 = time.time()
        last = [0.0]

        def prog(f, text):
            if f - last[0] >= 0.05 or f >= 1.0:
                last[0] = f
                say(f"    {f * 100:5.1f}%  {text}")
        import_3d(segy, store, progress=prog)
        say(f"  3D import done in {(time.time() - t0) / 60:.1f} min")
    with open(meta_p) as fh:
        m = json.load(fh)
    # Use the survey's exact bin grid when the SEG-Y numbering matches it
    il, xl = survey["il"], survey["xl"]
    same = (m["il0"] >= il[0] and m["il0"] + (m["nil"] - 1) * m["il_step"] <= il[1]
            and m["xl0"] >= xl[0] and m["xl0"] + (m["nxl"] - 1) * m["xl_step"] <= xl[1])
    ax0, ay0 = m["affine_x"], m["affine_y"]
    cil, cxl = np.array([m["il0"], m["il0"] + (m["nil"] - 1) * m["il_step"]]), np.array([m["xl0"], m["xl0"] + (m["nxl"] - 1) * m["xl_step"]])
    G = np.array(np.meshgrid(cil, cxl)).reshape(2, -1)
    fx = ax0[0] + ax0[1] * G[0] + ax0[2] * G[1]
    fy = ay0[0] + ay0[1] * G[0] + ay0[2] * G[1]
    sx = survey["ax"][0] + survey["ax"][1] * G[0] + survey["ax"][2] * G[1]
    sy = survey["ay"][0] + survey["ay"][1] * G[0] + survey["ay"][2] * G[1]
    err = float(np.max(np.hypot(fx - sx, fy - sy)))
    if same:
        m["affine_x"], m["affine_y"] = survey["ax"], survey["ay"]
        m["affine_source"] = f"OpendTect .survey bin grid (SEG-Y header fit differed by up to {err:.1f} m)"
        with open(meta_p, "w") as fh:
            json.dump(m, fh, indent=1)
        say(f"  geometry: survey bin grid applied (header coordinates were {err:.1f} m off at the corners)")
    else:
        say(f"  WARNING: SEG-Y inline/crossline numbers do not match the survey; kept header coordinates")
    say(f"  inlines {m['il0']}-{m['il0'] + (m['nil'] - 1) * m['il_step']}, crosslines "
        f"{m['xl0']}-{m['xl0'] + (m['nxl'] - 1) * m['xl_step']}, {m['ns']} samples at {m['dt'] * 1000:g} ms")
    return segy, store, m


def _grid_index(m, x, y):
    a, b = m["affine_x"], m["affine_y"]
    M = np.array([[a[1], a[2]], [b[1], b[2]]])
    Minv = np.linalg.inv(M)
    il = Minv[0, 0] * (x - a[0]) + Minv[0, 1] * (y - b[0])
    xl = Minv[1, 0] * (x - a[0]) + Minv[1, 1] * (y - b[0])
    return np.rint((il - m["il0"]) / m["il_step"]).astype(int), np.rint((xl - m["xl0"]) / m["xl_step"]).astype(int)


def build_horizons(src, m):
    from seismic_tool.horizons import HorizonSet
    hs = HorizonSet()
    key = f"3D:{VOL}"
    files = sorted(glob.glob(os.path.join(src, "Rawdata", "Surface_data", "*.txt")))
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0].replace("F3-Horizon-", "").replace("-", " ")
        try:
            a = np.loadtxt(f, usecols=(0, 1, 2), ndmin=2)
        except Exception as e:
            say(f"  {name}: could not read ({e})")
            continue
        i, j = _grid_index(m, a[:, 0], a[:, 1])
        ok = (i >= 0) & (i < m["nil"]) & (j >= 0) & (j < m["nxl"]) & np.isfinite(a[:, 2])
        h = hs.new(name)
        g = h.get(key, (m["nil"], m["nxl"]), create=True)
        g[i[ok], j[ok]] = a[ok, 2].astype(np.float32)
        nb = int(np.isfinite(g).sum())
        rng = f", {np.nanmin(g):.0f}-{np.nanmax(g):.0f} ms" if nb else " (outside this volume)"
        say(f"  horizon {name}: {nb:,} of {g.size:,} bins{rng}")
    hs.dirty = True
    return hs


def build_faults(src):
    from seismic_tool.faults import FaultSet
    fs = FaultSet()
    for f in sorted(glob.glob(os.path.join(src, "Rawdata", "Faults", "*.txt"))):
        a = np.loadtxt(f, ndmin=2)
        flt = fs.new(os.path.splitext(os.path.basename(f))[0].replace("FaultA", "Fault A"))
        sticks = a[:, 3] if a.shape[1] > 3 else np.zeros(len(a))
        for s in np.unique(sticks):
            pts = a[sticks == s][:, :3]
            if len(pts) >= 2:
                flt.add_stick(pts)
        say(f"  fault {flt.name}: {len(flt.sticks)} sticks")
    fs.dirty = True
    return fs


def build_seis_project(dst, segy, store, wcproj, hs, fs):
    from seismic_tool.project import SeisProject
    p = SeisProject()
    p.path = os.path.join(dst, NAME + ".seisproj.json")
    p.name = NAME
    p.volumes = [{"name": VOL, "store": store, "segy": segy}]
    p.well_project = wcproj
    p.horizons = hs
    p.faults = fs
    p.save()
    say(f"  {os.path.basename(p.path)}: volume '{VOL}', {len(hs.items)} horizons, {len(fs.items)} fault(s), wells linked")
    return p.path


def main(argv):
    up = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    src = argv[0] if argv else os.path.join(up, "Downloads", "F3_Demo_2023", "F3_Demo_2023")
    if not os.path.exists(os.path.join(src, ".survey")) and os.path.exists(os.path.join(src, "F3_Demo_2023", ".survey")):
        src = os.path.join(src, "F3_Demo_2023")
    dst = argv[1] if len(argv) > 1 else os.path.join(up, "Documents", NAME)
    if not os.path.exists(os.path.join(src, ".survey")):
        raise SystemExit(f"F3 folder not found (no .survey file): {src}")
    os.makedirs(dst, exist_ok=True)
    log = os.path.join(dst, "setup_log.txt")
    open(log, "w").close()
    _LOG[:] = ["@" + log]
    say(f"F3 setup {time.strftime('%Y-%m-%d %H:%M')}\nF3 data:  {src}\nProject:  {dst}\n")
    survey = read_survey(os.path.join(src, ".survey"))
    os.makedirs(os.path.join(dst, "raw"), exist_ok=True)
    shutil.copy2(os.path.join(src, ".survey"), os.path.join(dst, "raw", "F3.survey.txt"))
    vf = os.path.join(src, "Rawdata", "Velocity_functions.txt")
    if os.path.exists(vf):
        shutil.copy2(vf, os.path.join(dst, "raw"))
    say("[1/5] Wells")
    names, tops = build_wells(src, dst)
    say("[2/5] Well Correlation project")
    wcproj = build_well_project(dst, names, tops)
    say("[3/5] 3D seismic (copy + import, 5-20 minutes)")
    segy, store, m = build_volume(src, dst, survey)
    say("[4/5] Horizons and faults")
    hs = build_horizons(src, m)
    fs = build_faults(src)
    say("[5/5] Seismic project")
    sp = build_seis_project(dst, segy, store, wcproj, hs, fs)
    with open(os.path.join(dst, "README.txt"), "w") as fh:
        fh.write(f"F3 Netherlands - GeoHarmonic project, built {time.strftime('%Y-%m-%d %H:%M')} by setup_f3.py\n\n"
                 f"Data: F3 Demo 2023, dGB Earth Sciences via TerraNubis, Creative Commons BY-SA 3.0.\n"
                 f"Coordinates: {CRS}. Depths: metres MD below KB (vertical wells). Seismic: two-way time, ms.\n\n"
                 f"Open in GeoHarmonic:\n  Seismic Interpretation: {os.path.basename(sp)}\n"
                 f"  Well Correlation:      {os.path.basename(wcproj)}\n")
    say("\nF3 SETUP DONE")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except BaseException:
        import traceback
        say("\nF3 SETUP FAILED\n" + traceback.format_exc())
        sys.exit(1)
