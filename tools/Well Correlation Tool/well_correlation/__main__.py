"""Entry point.

    python -m well_correlation                      open the app
    python -m well_correlation MyProject.wcproj.json
    python -m well_correlation --selftest           test the data layer (no GUI needed)
    python -m well_correlation --check FOLDER       print a curve inventory for a data folder
    python -m well_correlation --png PROJECT OUT.png   export the current section without the GUI
    python -m well_correlation --dlisinfo FILE.dlis      list logical files, frames, channels
    python -m well_correlation --dlis2las IN.dlis OUT_DIR [--arrays]   one LAS 2.0 per DLIS frame
    python -m well_correlation --checkshot IN.dlis OUT.csv            time-depth table from a CSAT/VSI DLIS
    python -m well_correlation --petroqc PROJECT [WELL ...] [--out DIR]  tool / vendor / resolution + curve QC report
    python -m well_correlation --tooltable OUT.json                     write the editable logging-tool table
    python -m well_correlation --petro PROJECT [WELL ...] [--out DIR]    deterministic interpretation per zone
                               (parameters: <project>_petro/params.json, written with defaults on the first run)
    python -m well_correlation --multimin PROJECT WELL [--out DIR]        mineral solver (mm_minerals / mm_fluid / mm_logs
                               in params.json; default quartz + illite + oil)
    python -m well_correlation --petromc PROJECT WELL [--n 200] [--method deterministic|multimin] [--out DIR]
                               Monte-Carlo P90 / P50 / P10 of net pay, PHIH, SoPhiH per zone
    python -m well_correlation --rockphysics PROJECT WELL [--out DIR]     elastic logs, Vs, Gassmann brine/oil/gas/fizz
"""
import argparse
import os
import sys

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")  # Anaconda also ships PyQt5


def main(argv=None):
    ap = argparse.ArgumentParser(prog="well_correlation", description="Well log correlation (Geo Tool Kit)")
    ap.add_argument("project", nargs="?", help="project file (.wcproj.json)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", metavar="FOLDER")
    ap.add_argument("--png", nargs=2, metavar=("PROJECT", "OUT"))
    ap.add_argument("--dlisinfo", metavar="FILE")
    ap.add_argument("--dlis2las", nargs=2, metavar=("IN", "OUT_DIR"))
    ap.add_argument("--arrays", action="store_true", help="--dlis2las: also write array channels (one column each)")
    ap.add_argument("--checkshot", nargs=2, metavar=("IN", "OUT_CSV"))
    ap.add_argument("--petroqc", nargs="+", metavar=("PROJECT", "WELL"))
    ap.add_argument("--out", metavar="DIR", help="--petroqc: output folder (default <project>_petro)")
    ap.add_argument("--tooltable", metavar="OUT_JSON")
    ap.add_argument("--petro", nargs="+", metavar=("PROJECT", "WELL"))
    ap.add_argument("--multimin", nargs=2, metavar=("PROJECT", "WELL"))
    ap.add_argument("--petromc", nargs=2, metavar=("PROJECT", "WELL"))
    ap.add_argument("--rockphysics", nargs=2, metavar=("PROJECT", "WELL"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--method", default="deterministic")
    a = ap.parse_args(argv)

    if a.tooltable:
        from .petro.tools import write_tool_table
        print("wrote", write_tool_table(a.tooltable))
        return
    if a.petro:
        import os as _os
        from .project import Project
        from .petro import interp
        from .petro.store import petro_dir
        p = Project.open(a.petro[0])
        if a.out:
            p.petro_out = a.out
        if not _os.path.exists(interp.params_path(p)):
            interp.save_params(p, {"default": {}, "zones": {f["name"]: {} for f in p.formations}, "wells": {},
                                   "_defaults_for_reference": interp.DEFAULTS, "_choices": interp.METHOD_CHOICES})
        res = interp.interpret_project(p, a.petro[1:] or None)
        for n, r in res.items():
            for q in r["zones"]:
                print(f"{n:12s} {q['Zone'][:24]:24s} gross {q['Gross_TVD']:7.1f}  net res {q['NetRes_TVD']:6.1f}  "
                      f"net pay {q['NetPay_TVD']:6.1f} m  PHIE {q['PHIE_res']}  Sw(pay) {q['Sw_pay']}  SoPhiH {q['SoPhiH']}")
            for note in r.get("notes", []):
                print(f"{n:12s} ! {note}")
        print("\nsummary.csv, params.json and curves in", petro_dir(p))
        return
    if a.rockphysics:
        from .project import Project
        from .petro import rockphysics as RP, interp
        from .petro.store import petro_dir
        p = Project.open(a.rockphysics[0])
        if a.out:
            p.petro_out = a.out
        det = interp.interpret_well(p, a.rockphysics[1], save=False)
        r = RP.rock_physics_well(p, a.rockphysics[1], det=det)
        print("Vs:", "; ".join(r["vs_source"]))
        print("Zone means over net reservoir (20b cut-offs):")
        for row in RP.scenario_table(r, det["zones"], det["curves"]["NET_RES"]):
            print("  " + "  ".join(f"{k} {v}" for k, v in row.items()))
        for zn, f in r["fluids"].items():
            if zn:
                print(f"  {zn[:24]:24s} T {f['T']:.0f} degC  P {f['P']:.1f} MPa  brine {f['brine_rho']:.3f} g/cc K {f['brine_k']:.2f}"
                      f"  gas {f['gas_rho']:.3f} K {f['gas_k']:.3f}  oil {f['oil_rho']:.3f} K {f['oil_k']:.2f} GPa")
        print("\nRP_ curves saved in", petro_dir(p))
        return
    if a.multimin or a.petromc:
        from .project import Project
        from .petro import multimin as MM
        from .petro.store import petro_dir
        pp, wn = a.multimin or a.petromc
        p = Project.open(pp)
        if a.out:
            p.petro_out = a.out
        if a.multimin:
            r = MM.multimin_well(p, wn)
            for q in r["zones"]:
                vols = ", ".join(f"{k[2:]} {v}" for k, v in q.items() if k.startswith("V_"))
                print(f"{wn:12s} {q['Zone'][:24]:24s} PHIE {q['PHIE_res']}  Sw(pay) {q['Sw_pay']}  net pay {q['NetPay_TVD']} m"
                      f"  misfit {q['Misfit_med']}  [{vols}]")
            print("\nMM_ curves saved in", petro_dir(p))
        else:
            mc = MM.monte_carlo(p, wn, n=a.n, method=a.method)
            for r in MM.mc_table(mc):
                print(f"{wn:12s} {r['Zone'][:24]:24s} net pay {r['NetPay_TVD_P90']} / {r['NetPay_TVD_P50']} / "
                      f"{r['NetPay_TVD_P10']} m   PHIH {r['PHIH_pay_P90']} / {r['PHIH_pay_P50']} / {r['PHIH_pay_P10']}   "
                      f"SoPhiH {r['SoPhiH_P90']} / {r['SoPhiH_P50']} / {r['SoPhiH_P10']}   (P90 / P50 / P10)")
        return
    if a.petroqc:
        from .project import Project
        from .petro import qc
        p = Project.open(a.petroqc[0])
        if a.out:
            p.petro_out = a.out
        reps = qc.qc_project(p, a.petroqc[1:] or None)
        for r in reps.values():
            print(qc.format_report(r))
        from .petro.store import petro_dir
        print("\nreports + QC curves in", petro_dir(p))
        return

    if a.dlisinfo:
        from .dlis import DlisFile
        print(DlisFile(a.dlisinfo).summary())
        return
    if a.dlis2las:
        from .dlis import dlis_to_las
        for r in dlis_to_las(a.dlis2las[0], a.dlis2las[1], a.arrays):
            print(f"wrote {r['path']}: {r['rows']} rows, {len(r['curves'])} curves"
                  + (f", skipped {len(r['skipped'])} array channels" if r["skipped"] else ""))
        return
    if a.checkshot:
        from .dlis import checkshot_from_dlis, write_checkshot_csv
        t = checkshot_from_dlis(a.checkshot[0])
        if t is None:
            print("no checkshot frame (CTT1/BTI1 + depth) found")
            sys.exit(1)
        write_checkshot_csv(t, a.checkshot[1])
        print(f"wrote {a.checkshot[1]}: {len(t['md'])} levels {t['md'].min():.0f}-{t['md'].max():.0f} m MD")
        return

    if a.selftest:
        from .selftest import run
        sys.exit(0 if run() else 1)
    if a.check:
        from .project import Project
        p = Project.from_folder(a.check)
        for n in p.wells:
            w = p.load_well(n)
            fams = ", ".join(sorted(w.logs)) or "NO LOGS"
            print(f"{n:15s} {len(w.las_paths):3d} LAS  {fams}")
        return
    if a.png:
        from .project import Project
        from .export import export_section
        p = Project.open(a.png[0])
        export_section(p, p.section_wells(), a.png[1])
        print("wrote", a.png[1])
        return
    try:
        from PySide6 import QtWidgets  # noqa: F401
        import pyqtgraph  # noqa: F401
    except ImportError as ex:
        print(f"\nQt could not be loaded: {ex}\nPython: {sys.executable}\n"
              "Run 'Install (run once).bat' (it sets up a private .venv for this tool).\n")
        sys.exit(2)
    from PySide6 import QtWidgets
    from .gui.main_window import MainWindow
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("Well Correlation Tool")
    app.setStyle("Fusion")
    win = MainWindow(a.project)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
