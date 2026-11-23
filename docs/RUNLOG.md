# Run log

## 2026-11-23 - setup (interactive session with Chris)
- Repo created with the three tools (Mistie, Well Correlation v0.4.0, Seismic v0.1.0 step 1: 2D+3D
  loading, section/time-slice viewer, basemap), NPRA 2D test lines, Maui well data, Qt mock.
- All self-tests pass. Plan in docs/PLAN.md; nightly scheduled runs work on branch `dev`.
- Next: PLAN item 1 (horizon picking + autotracking).

## 2026-11-23 run 1 (scheduled) - IN PROGRESS, started 07:20 UTC
- Item 1 DONE - horizon picking + autotracking (v0.2.0): new horizons.py, synth.py, test_horizons.py,
  gui/horizon_dock.py; section_panel (pick mode, overlays), basemap (time map), main_window (dock,
  Interpret menu P/T/Ctrl+Z), project (horizons saved in <project>_seismic/horizons). Tests: --selftest 47
  checks PASS (incl. 3D folded-reflector autotrack max err 1.45 ms / mean 0.22 ms at 4 ms sampling, 2D tie
  propagation, gap bridging, lateral mix, save/load), --guitest under fakeqt PASS; Mistie + Well
  Correlation self-tests PASS. Real-data check: NPRA R-2 tracks 1,000-2,600 traces with lateral mix 5-9.
- Item 2 DONE - fault sticks + fault surfaces (v0.3.0): new faults.py, gui/fault_dock.py, test_faults.py;
  section_panel fault mode/overlays, basemap fault traces, project saves faults, tracker barrier.
  Tests: --selftest 59 checks PASS (fault cut positions < 0.05 trace, tracker 0% across fault vs 100%
  without), --guitest PASS, other tools PASS.
- Item 3 DONE - dip field (v0.4.0): rgt/dip.py, test_dip.py, section display box (dip/planarity),
  Interpret > Compute dip field (threaded job). Tests: synthetic 3D dips within 0.6%, block == whole volume,
  2D fold dip max err 0.034 ms/trace; --selftest/--guitest PASS; NPRA R-2 dip visually sensible (4.7 s).
- Working on PLAN item 4 next.
