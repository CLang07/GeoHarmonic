"""Command line entry:  python -m mistie_tool CONFIG.json [--batch] [--auto-reject] [--apply]
                       python -m mistie_tool --selftest
                       python -m mistie_tool --make-config FOLDER"""
import argparse, glob, json, os, sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mistie_tool", description="2D seismic mistie analysis")
    ap.add_argument("config", nargs="?", help="project file (.mistie.json) or config JSON; omit to open the app empty")
    ap.add_argument("--batch", action="store_true", help="no GUI: measure, solve, write CSVs")
    ap.add_argument("--auto-reject", action="store_true", help="drop worst-fitting ties automatically (batch)")
    ap.add_argument("--apply", action="store_true", help="also write corrected SEG-Y copies to the output folder")
    ap.add_argument("--selftest", action="store_true", help="run the synthetic test and exit")
    ap.add_argument("--make-config", metavar="FOLDER", help="write a starter config for the SEG-Y files in FOLDER")
    a = ap.parse_args(argv)

    if a.selftest:
        from .selftest import run
        sys.exit(0 if run() else 1)
    if a.make_config:
        files = sorted(glob.glob(os.path.join(a.make_config, "*.sgy")) + glob.glob(os.path.join(a.make_config, "*.SGY"))
                       + glob.glob(os.path.join(a.make_config, "*.segy")))
        files = sorted(set(files))
        cfg = {"data_dir": ".", "nav_csv": None, "output_dir": "mistie_output",
               "lines": [{"name": os.path.splitext(os.path.basename(f))[0], "segy": os.path.basename(f)} for f in files],
               "header_bytes": {"cdp": 21, "x": 181, "y": 185, "scalar": 71},
               "default_window": [0.4, 2.6], "reference": "auto"}
        p = os.path.join(a.make_config, "mistie_config.json")
        json.dump(cfg, open(p, "w"), indent=2); print(f"wrote {p} with {len(files)} lines - edit names/settings, then run it")
        return
    if not a.config and not a.batch:
        try:
            from .app import main as app_main
        except ImportError as ex:
            ap.error(f"the desktop app needs tkinter ({ex}); give a config file to use the basic viewer")
        app_main(None); return
    if not a.config:
        ap.error("--batch needs a config or project file")

    if not a.batch:
        try:
            from .app import main as app_main
            app_main(a.config); return
        except ImportError:
            print("tkinter not available - using the basic matplotlib viewer")
    from .project import Project
    if json.load(open(a.config)).get("format") == "mistie_tool project":
        from .session import Session
        S = Session(a.config); P = S.P
    else:
        P = Project(a.config)
    print(f"{len(P.names)} lines, {len(P.ties)} ties, dt {P.dt * 1000:.1f} ms, reference {P.ref}")
    if a.batch:
        P.solve()
        if a.auto_reject:
            for line in P.auto_reject():
                print("  " + line)
        P.solve(); out = P.save()
        s = P.summary()
        print(f"RMS mistie {s['before_ms']:.1f} -> {s['after_ms']:.1f} ms, {s['before_deg']:.0f} -> {s['after_deg']:.0f} deg "
              f"({s['n_used']}/{s['n_ties']} ties used)")
        for n in P.names:
            c = P.sol[n]
            print(f"  {n:8s} shift {c['shift']:+7.2f} ms  phase {c['phase']:+6.1f} deg  gain {c['gain']:.3f}  ties {c['n_ties']}")
        for lp in P.loops:
            print(f"  loop {' > '.join(lp['lines'])}: {lp['shift']:+.1f} ms {lp['phase']:+.0f} deg")
        if a.apply:
            for p in P.write_corrected_segy():
                print("  wrote", p)
        print("outputs in", out)
    else:
        from .gui import run
        run(P)


if __name__ == "__main__":
    main()
