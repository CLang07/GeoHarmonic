# GeoHarmonic - instructions for every Claude session

Owner: Chris (oil & gas geoscientist, seismic interpreter, Houston). This repo is his "Geo Tool Kit":
Python desktop tools built on public data. Read `docs/HANDOFF.md` (architecture, conventions, lessons)
and `docs/PLAN.md` (ordered work) before changing anything.

## Rules for scheduled build runs
1. Read `docs/RUNLOG.md` (latest entry at the bottom) and `docs/PLAN.md`. Do the FIRST plan item whose
   status is `todo` (or finish one marked `partial`). One item per run - depth and tests over breadth.
2. Work on branch `dev` (create from `main` if missing; `git pull` first). Never push to `main`;
   Chris merges `dev` after testing on his PC.
3. Before committing: every tool's self-test must pass
   (`cd "tools/<Tool>" && python3 -m <package> --selftest`), plus new tests for the new feature.
   GUI code must import and exercise its non-drawing logic under the Qt mock:
   `PYTHONPATH=<repo>/dev/fakeqt:. python3 ...` (see HANDOFF "Testing GUI code").
4. No network installs (pip is off in these sessions). Only numpy, matplotlib, pandas, PIL are
   available here. On Chris's PC the venv has numpy, matplotlib, PySide6==6.5.3, pyqtgraph==0.13.7.
   Do not add other dependencies (no scipy, segyio, lasio). Write it in numpy.
5. Keep existing files and behaviour working; don't rewrite modules wholesale.
6. At the end: update `docs/PLAN.md` status, append a RUNLOG entry (date, item, what was built,
   files touched, tests run + results, known limitations, "what Chris should test first" as a short
   numbered checklist, next item), commit with a clear message, push `dev`.
7. Final message of the run = the RUNLOG entry text (it is emailed to Chris). Plain, concise.
8. If blocked (tests failing you can't fix, design question only Chris can answer), commit what works
   behind a flag or in a separate module, mark the item `blocked: <reason>` and move to the next item
   only if time allows. Never leave `dev` with failing self-tests.
