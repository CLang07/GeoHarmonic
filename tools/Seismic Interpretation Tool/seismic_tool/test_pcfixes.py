"""PLAN 38a-c regression tests: fixes from the first install on Chris's Windows PC (25 Nov 2026).

38a  the 3D Viewer cached `context().functions()` (a wrapper Python does not own) -> AttributeError in paintGL;
     one crashing GUI test module stopped every module after it.
38b  PySide6 6.5.3 (shiboken built for NumPy 1.x) under NumPy 2 -> pin "numpy<2" in every installer line.
38c  "QFontDatabase: Cannot find font directory" in offscreen tests -> QT_QPA_FONTDIR on Windows.
The real-Qt OpenGL regression (initializeGL, paintGL x2, resize, paintGL) is test_scene3d.run_gl_real (GUI test).
"""
import importlib.util
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", ".."))
REPO = os.path.dirname(TOOLS)

# numpy names that only exist in NumPy >= 2 (Chris's Qt tools now run on 1.26)
NUMPY2_ONLY = ["np.trapezoid", "np.concat(", "np.unique_values", "np.unique_counts", "np.unique_inverse",
               "np.unique_all", "np.isdtype", "np.bitwise_count", "np.cumulative_sum", "np.cumulative_prod",
               "np.permute_dims", "np.matrix_transpose", "np.vecdot", "np.astype(", "np.acos(", "np.asin(",
               "np.atan(", "np.atan2(", "np.pow(", "np.linalg.matrix_norm", "np.linalg.vector_norm",
               "np.linalg.svdvals", "np.exceptions", "np.dtypes", "StringDType", ".mT"]


def _load_runner():
    path = os.path.join(TOOLS, "guitest_runner.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("gh_guitest_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # no side effects at import (everything runs in main())
    return mod


def numpy2_uses(root):
    """[(file, line no, text)] of NumPy-2-only calls that are not guarded by hasattr / getattr on the same line."""
    hits = []
    for d, _, files in os.walk(root):
        if ".venv" in d or "__pycache__" in d:
            continue
        for fn in files:
            if not fn.endswith(".py") or fn == "test_pcfixes.py":
                continue
            p = os.path.join(d, fn)
            try:
                lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines, 1):
                code = line.split("#", 1)[0]
                for name in NUMPY2_ONLY:
                    if name in code and "hasattr" not in code and "getattr" not in code:
                        if name == ".mT" and not re.search(r"\.mT\b", code):
                            continue
                        hits.append((os.path.relpath(p, root), i, line.strip()))
    return hits


def run(check, tmp):
    # ---- 38a: no cached GL function table in the 3D Viewer
    src = open(os.path.join(HERE, "gui", "view3d.py"), encoding="utf-8").read()
    cached = re.findall(r"self\.f\s*=|self\.f\.", src)
    check(not cached and "def _funcs" in src and src.count("self._funcs()") >= 3,
          f"38a: view3d fetches context().functions() fresh in initializeGL / paintGL / resizeGL (cached uses: {len(cached)})")
    check("def resizeGL" in src, "38a: view3d has resizeGL")

    # ---- 38a: one crashing test module is a FAIL with traceback, the next still runs
    from . import selftest as ST
    seen = []

    def chk(c, m):
        seen.append((bool(c), m))

    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ST._guarded(chk, "test_boom.run_gui(check, tmp)", lambda: 1 / 0)
        ST._guarded(chk, "test_ok.run_gui(check, tmp)", lambda: chk(True, "after"))
    check(seen[0][0] is False and "test_boom.run_gui crashed: ZeroDivisionError" in seen[0][1]
          and seen[1] == (True, "after") and "Traceback" in buf.getvalue(),
          "38a: selftest._guarded records a crash as FAIL (+ traceback) and keeps going")
    sel = open(os.path.join(HERE, "selftest.py"), encoding="utf-8").read()
    gui = sel[sel.index("def run_gui"):]
    bare = re.findall(r"^\s+test_\w+\.run\w*\(check", gui, flags=re.M)
    check(not bare and gui.count("_guarded(check") >= 40,
          f"38a: every GUI test module in run_gui is guarded ({gui.count('_guarded(check')} guarded, {len(bare)} bare)")
    wc = os.path.join(TOOLS, "Well Correlation Tool", "well_correlation", "__main__.py")
    if os.path.isfile(wc):
        s = open(wc, encoding="utf-8").read()
        check("guarded(\"test_petro.run_gui\"" in s and "guarded(\"test_maplink.run_gui\"" in s,
              "38a: Well Correlation --guitest guards its modules the same way")

    # ---- 38a/38c: guitest_runner
    R = _load_runner()
    check(R is not None, "38c: tools/guitest_runner.py found")
    if R is not None:
        fd = R.font_dir({"WINDIR": r"C:\Windows"}, "win32")
        check(fd is not None and fd.replace("\\", "/").endswith("Windows/Fonts"),
              f"38c: QT_QPA_FONTDIR on Windows -> {fd}")
        check(R.font_dir({}, "linux") is None, "38c: QT_QPA_FONTDIR left alone off Windows")
        # a package whose --guitest raises: the runner must print the traceback + GUI TEST FAILED, exit 1
        d = tempfile.mkdtemp(dir=tmp, prefix="runner_")
        os.makedirs(os.path.join(d, "boompkg"))
        open(os.path.join(d, "boompkg", "__init__.py"), "w").close()
        with open(os.path.join(d, "boompkg", "__main__.py"), "w") as f:
            f.write("print('PASS first check')\nraise RuntimeError('boom in a test module')\n")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)  # no Qt mock: the runner must cope without PySide6 too
        p = subprocess.run([sys.executable, os.path.join(TOOLS, "guitest_runner.py"), "boompkg", "1"], cwd=d,
                           env=env, capture_output=True, text=True, timeout=120)
        out = p.stdout + p.stderr
        check(p.returncode == 1 and "GUI TEST FAILED" in out and "boom in a test module" in out
              and "PASS first check" in out, f"38a: guitest_runner turns a crash into a FAIL report (rc={p.returncode})")

    # ---- 38b: numpy<2 pinned wherever PySide6 or the optional extras are installed
    bats = []
    for d, _, files in os.walk(REPO):
        if ".venv" in d or ".git" in d:
            continue
        bats += [os.path.join(d, f) for f in files if f.lower().endswith(".bat")]
    lines = []
    for b in bats:
        for line in open(b, encoding="utf-8", errors="replace").read().splitlines():
            if "pip install" in line and "upgrade pip" not in line and ("PySide6" in line or "xgboost" in line):
                lines.append((os.path.basename(b), line))
    check(len(lines) >= 3 and all('"numpy<2"' in l for _, l in lines),
          f"38b: every Qt / extras pip line pins \"numpy<2\" ({len(lines)} lines in {len(bats)} .bat files)")
    hits = numpy2_uses(TOOLS)
    check(not hits, "38b: no NumPy-2-only API in the tools" + (f": {hits[:3]}" if hits else ""))
    import numpy as np
    check(True, f"38b: numpy here {np.__version__} (Chris's Qt tools: 1.26 after the pin)")
