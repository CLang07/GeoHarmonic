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
- Working on PLAN item 2 next.
