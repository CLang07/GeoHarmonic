"""python -m seismic_tool [project.seisproj.json] | --selftest | --guitest | --check-ml | --info FILE.sgy | --png PROJECT OUT.png"""
import os
import sys

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")


def _info(path):
    from .segy import SegyFile
    f = SegyFile(path)
    print(f"{path}\n  traces {f.ntraces:,}  samples {f.ns}  dt {f.dt * 1000:g} ms  format {f.fmt}")
    g = f.guess_il_xl()
    print(f"  inline/crossline bytes: {g if g else 'not found (2D line?)'}")
    print(f.text_header[:800])


def _png(project, out):
    """Render the project's current section to a PNG without Qt (for checking)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from .project import SeisProject
    p = SeisProject.open(project)
    if not (p.lines or p.volumes):
        raise SystemExit(f"{project}: no 3D volumes or 2D lines to render")
    fig, axes = plt.subplots(1, len(p.lines) + len(p.volumes), figsize=(7 * (len(p.lines) + len(p.volumes)), 6),
                             squeeze=False)
    k = 0
    for v in p.volumes:
        vol = p.volume(v["name"])
        a = vol.inline(vol.inlines[len(vol.inlines) // 2]).T
        c = np.percentile(np.abs(a), 99)
        axes[0, k].imshow(a, cmap="seismic_r", vmin=-c, vmax=c, aspect="auto",
                          extent=[vol.xlines[0], vol.xlines[-1], vol.times[-1], vol.times[0]])
        axes[0, k].set_title(v["name"])
        k += 1
    for l in p.lines:
        line = p.line(l["name"])
        a = line.data().T
        c = np.percentile(np.abs(a), 99)
        axes[0, k].imshow(a, cmap="seismic_r", vmin=-c, vmax=c, aspect="auto",
                          extent=[line.cdp[0], line.cdp[-1], line.times[-1], line.times[0]])
        axes[0, k].set_title(l["name"])
        k += 1
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print("wrote", out)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--selftest":
        from .selftest import run
        sys.exit(0 if run() else 1)
    if argv and argv[0] == "--guitest":
        from .selftest import run_gui
        sys.exit(0 if run_gui() else 1)
    if argv and argv[0] == "--check-ml":
        from .analytics.check_ml import run as check_ml
        check_ml()
        return
    if argv and argv[0] == "--info":
        for a in argv[1:]:
            _info(a)
        return
    if argv and argv[0] == "--png":
        if len(argv) < 3:
            raise SystemExit("usage: --png PROJECT OUT.png")
        _png(argv[1], argv[2])
        return
    try:
        from PySide6 import QtWidgets
        import pyqtgraph as pg
    except Exception as e:
        print("Missing package:", e)
        print("Run 'Install (run once).bat' first.")
        sys.exit(1)
    pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")
    from .gui.main_window import MainWindow
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("Seismic Interpretation Tool")
    w = MainWindow(argv[0] if argv else None)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
