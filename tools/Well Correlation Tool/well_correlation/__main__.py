"""Entry point.

    python -m well_correlation                      open the app
    python -m well_correlation MyProject.wcproj.json
    python -m well_correlation --selftest           test the data layer (no GUI needed)
    python -m well_correlation --check FOLDER       print a curve inventory for a data folder
    python -m well_correlation --png PROJECT OUT.png   export the current section without the GUI
    python -m well_correlation --dlisinfo FILE.dlis      list logical files, frames, channels
    python -m well_correlation --dlis2las IN.dlis OUT_DIR [--arrays]   one LAS 2.0 per DLIS frame
    python -m well_correlation --checkshot IN.dlis OUT.csv            time-depth table from a CSAT/VSI DLIS
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
    a = ap.parse_args(argv)

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
