# GeoHarmonic - instructions for every Claude session

Owner: Chris (oil & gas geoscientist, seismic interpreter, Houston). This repo is his "Geo Tool Kit":
Python desktop tools built on public data. Read `docs/HANDOFF.md` (architecture, conventions, lessons)
and `docs/PLAN.md` (ordered work) before changing anything.

## Rules for scheduled build runs
1. Read `docs/RUNLOG.md` (latest entry at the bottom) and `docs/PLAN.md`. Do the FIRST plan item whose
   status is `todo` (or finish one marked `partial`). When an item is finished, tested, committed and
   pushed, continue with the next item in the same run. Keep going until the plan is done, you have
   worked about 4 hours of wall-clock time (then stop so runs never overlap), or usage runs out.
   Quality rule still applies: each item fully tested before moving on.
1a. Checkpoint often - usage can run out without warning: commit + push `dev` after every finished item,
   and at least every ~45 minutes during a long item (mark it `partial` in PLAN.md with a note of what is
   left). Never push a commit with failing self-tests; if mid-item code is not passing yet, keep it
   behind a flag / in an unused module so the tools still pass.
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
6. After each item and at the end: update `docs/PLAN.md` status, append a RUNLOG entry (date, item, what was built,
   files touched, tests run + results, known limitations, "what Chris should test first" as a short
   numbered checklist, next item), commit with a clear message, push `dev`.
7. Final message of the run = the RUNLOG entry text (it is emailed to Chris). Plain, concise.
8. If blocked (tests failing you can't fix, design question only Chris can answer), commit what works
   behind a flag or in a separate module, mark the item `blocked: <reason>` and move to the next item
   only if time allows. Never leave `dev` with failing self-tests.
