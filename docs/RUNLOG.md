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
- Item 4 DONE - RGT mesh + global solve (v0.5.0): rgt/solve.py, test_rgt.py, synth.rgt_geology, RGT
  display, Interpret > Compute RGT 3D/2D with settings dialog. Tests: iso-RGT vs 60 true horizons median
  0.15 ms; fault case p95 < 3.5 ms above the wedge; horizon constraint ties blocks (p95 23 -> 4.3 ms);
  2D network median 0.10 ms; all self-tests + guitest PASS. NPRA R-2+R-3: 100 s, contours follow reflectors.
- Item 5 DONE - horizon stack / Wheeler / stratal slices (v0.6.0): rgt/stack.py, test_stack.py, panel mode
  "Stratal slice" + Time/Wheeler domain, Interpret menu (build stack 3D/2D, stack level -> horizon).
  Tests PASS (selftest + guitest); NPRA R-2 Wheeler flat.
- Item 6 DONE - RGT editing (v0.7.0): rgt/edit.py, test_edit.py, panel RGT edit mode + contours, Interpret
  menu (edit mode E, horizon constraint, fault re-link). Tests PASS (locality bit-exact, undo exact).
- Item 6b DONE - DLIS reader (Well Correlation v0.5.0): well_correlation/dlis.py, test_dlis.py, project loads
  wells/<Well>/dlis/*.dlis, CLI --dlisinfo/--dlis2las/--checkshot, Tui-1 added to the NZ project (logs from
  DLIS: GR, RHOB, NPHI, DT, PE, RDEEP, RSHAL, SP, CALI). Tests: synthetic DLIS round trip + real Tui-1 sanity
  ranges + checkshot PASS; all tool self-tests PASS.
- Item 7 DONE - wells on seismic + well ties (v0.8.0): welltie.py, gui/welltie_dialog.py, test_welltie.py,
  panel well overlay, Interpret > Well tie. Tests PASS (auto tie 12.1 vs 12 ms; real Tui-1/Maui-5 TD).
- Item 8 DONE - horizon maps (v0.9.0): maps.py, gui/map_dialog.py, basemap grids + contours, test_maps.py.
  All tests PASS.
- Item 9 DONE - docs/WHATS_NEW.md: risk-ordered test checklist + known limits + change log.
- Item 10 PARTIAL - mistie analysis core (v0.10.0): mistie.py, gui/mistie_dialog.py, Line2D corrections,
  test_mistie.py. Tests PASS. v0.10.1 adds along-line flex mode (ramp recovered within 0.14 ms).
  v0.11.0 completes item 10: horizon misties, groups/key lines, time-variant shift, overlapping vintages,
  residual map, tie table CSV, corrected SEG-Y export. Item 10 DONE. Tests PASS.
- Item 11 DONE - seismic attributes (v0.12.0): attributes.py, panel on-the-fly attributes, attribute volumes,
  curvature maps, test_attributes.py. Tests PASS; NPRA R-2 coherence/envelope/frequency look sensible (2.6 s).
- Item 12 DONE - velocity model + depth conversion (v0.13.0): velocity.py, gui/velocity_dialog.py, panel Depth
  domain, test_velocity.py. Tests PASS.
- Item 13 PARTIAL - 13a done: analytics package (attribute table, joins by API/name, zone stats, missing
  values, kNN/median imputation, scaling, outliers, Pearson/Spearman, VIF, PCA, regression lines). Tests PASS.
- 13a+ done: engineering/production import (presets, unit detection), completion + spacing + parent/child,
  production features (cums, IP, GOR, WC, Arps + EUR within 0.0% on synthetic), FracFocus, lease allocation,
  synthetic data set generator (dev/make_synthetic_engineering.py). Tests PASS.
- 13b done: models (OLS/ridge/lasso, kNN, tree, RF, XGBoost-style GBM with NaN handling + monotone constraints +
  early stopping + logistic, MLP, k-means, GMM, SOM) and honest validation (group/spatial-block CV, nested CV,
  learning curves). On the benchmark GBM R2 0.97 vs linear 0.29; random CV 0.86 vs group 0.36 / block 0.59.
