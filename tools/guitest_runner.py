"""GeoHarmonic GUI-test runner (used by Install GeoHarmonic.bat and Run all tests.bat).

    python guitest_runner.py <package> [minutes]

Runs `python -m <package> --guitest` against real Qt, off screen, with three safety nets:
  * every pop-up (message boxes, dialogs' exec(), menus, file/input dialogs) is answered
    automatically and logged, so a hidden modal window can never block the test;
  * output is written line by line, so the results file shows progress as it runs;
  * a watchdog stops the run after [minutes] (default 15) and prints where every thread
    was stuck, so a hang turns into a report instead of an endless wait.
Run it from the tool's folder (the one holding the package).
"""
import faulthandler
import os
import runpy
import sys


def font_dir(environ=None, platform=None):
    """Font folder for offscreen Qt (PLAN 38c): without it Qt prints 'QFontDatabase: Cannot find font directory'.
    Windows: %WINDIR%\\Fonts.  Elsewhere: leave Qt's default (returns None)."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if not platform.startswith("win"):
        return None
    d = os.path.join(environ.get("WINDIR", r"C:\Windows"), "Fonts")
    return d


def _note(kind, args):
    text = " ".join(str(a) for a in args[1:3] if isinstance(a, str))
    print(f"[auto-answered {kind}] {text}".replace("\n", " ")[:220])


def _patch_modals():
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([sys.argv[0]])
    MB = QtWidgets.QMessageBox

    def box(kind, ret):
        def f(*a, **k):
            _note("message " + kind, a)
            return ret
        return f
    for kind in ("information", "warning", "critical", "about"):
        setattr(MB, kind, box(kind, MB.StandardButton.Ok))
    MB.question = box("question", MB.StandardButton.Yes)

    def dexec(self, *a, **k):
        _note("dialog " + type(self).__name__, ("", self.windowTitle()))
        return 0  # same as Cancel: matches the headless (mock-Qt) test runs
    for cls in (QtWidgets.QDialog, QtWidgets.QMessageBox, QtWidgets.QFileDialog,
                QtWidgets.QInputDialog, QtWidgets.QColorDialog, QtWidgets.QProgressDialog):
        cls.exec = dexec
        cls.exec_ = dexec
    QtWidgets.QMenu.exec = lambda self, *a, **k: None
    FD = QtWidgets.QFileDialog
    for name in ("getOpenFileName", "getSaveFileName"):
        setattr(FD, name, staticmethod(lambda *a, _n=name, **k: (_note(_n, a), ("", ""))[1]))
    FD.getOpenFileNames = staticmethod(lambda *a, **k: (_note("getOpenFileNames", a), ([], ""))[1])
    FD.getExistingDirectory = staticmethod(lambda *a, **k: (_note("getExistingDirectory", a), "")[1])
    ID = QtWidgets.QInputDialog
    ID.getText = staticmethod(lambda *a, **k: (_note("getText", a), ("", False))[1])
    ID.getItem = staticmethod(lambda *a, **k: (_note("getItem", a), ("", False))[1])
    ID.getInt = staticmethod(lambda *a, **k: (_note("getInt", a), (0, False))[1])
    ID.getDouble = staticmethod(lambda *a, **k: (_note("getDouble", a), (0.0, False))[1])
    QtWidgets.QColorDialog.getColor = staticmethod(lambda *a, **k: (_note("getColor", a), __import__(
        "PySide6.QtGui", fromlist=["QColor"]).QColor())[1])


def main():
    pkg = sys.argv[1]
    minutes = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    sys.path.insert(0, os.getcwd())
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

    _fd = font_dir()
    if _fd and os.path.isdir(_fd) and "QT_QPA_FONTDIR" not in os.environ:
        os.environ["QT_QPA_FONTDIR"] = _fd
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    print(f"(GUI test watchdog: stops after {minutes} min and reports where it was stuck)")
    faulthandler.dump_traceback_later(minutes * 60, exit=True)

    try:
        _patch_modals()
    except Exception as e:  # Qt missing or broken: let the test itself report it
        print(f"(could not prepare Qt for the GUI test: {e})")
    sys.argv = [pkg, "--guitest"]
    try:
        runpy.run_module(pkg, run_name="__main__", alter_sys=True)
    except SystemExit:
        raise
    except BaseException:  # PLAN 38a: a crash outside the per-module guards still ends as a readable FAIL
        import traceback
        print(traceback.format_exc().rstrip())
        print("FAIL GUI test stopped by the exception above")
        print("GUI TEST FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
