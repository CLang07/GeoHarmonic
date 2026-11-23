# Run log

## 2026-11-23 - setup (interactive session with Chris)
- Repo created with the three tools (Mistie, Well Correlation v0.4.0, Seismic v0.1.0 step 1: 2D+3D
  loading, section/time-slice viewer, basemap), NPRA 2D test lines, Maui well data, Qt mock.
- All self-tests pass. Plan in docs/PLAN.md; nightly scheduled runs work on branch `dev`.
- Next: PLAN item 1 (horizon picking + autotracking).

## 2026-11-23 run 1 (scheduled) - 07:20-10:00 UTC, branch dev, Seismic tool v0.2.0 -> v0.17.1
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
- 13c done: explain.py (exact TreeSHAP == brute force, KernelSHAP, permutation importance, PDP/ICE, save/load,
  HTML report), Analytics window, --check-ml + optional extras installer. Tests PASS (v0.14.0).
- 13d done (v0.15.0): analytics/qc.py model QC toolkit - baselines (mean, IDW of target, best single input),
  y-randomisation, noise-feature test, residual QQ / variogram / Moran's I, split-conformal + quantile-GBM
  intervals with coverage, calibration (Brier), applicability domain, importance stability, hyperparameter
  sensitivity, PDP direction checks, paired fold comparison, blind/temporal/repeated designs; Analytics window
  "Model QC..." button (HTML) and applicability mask on predicted maps. Tests: Moran z 24.9 -> 4.1 with x/y,
  conformal 80% covers 84%, y-randomisation collapses. Item 13 DONE with v0.15.1: engineering / production
  CSV import dialog (presets, mapping + units, saved mappings; works without a well project).
- Item 14 DONE (v0.16.0-0.16.2): laterals.py (min-curvature trajectories, attributes along MD from grids /
  3D volume / logs / stages / zones, per-lateral stats, batched colour drawing + LOD), dashboard_model.py
  (filters, linked selection, histogram/crossplot/table data, layouts, HTML export), gui/dashboard_window.py
  (Interpret > Dashboard), gui/well_strip.py (geosteering strip: log, TVDSS + zone, seismic along wellbore).
  1,000 laterals -> 16 batched curves in 0.1 s. Tests PASS.
- Item 15 PARTIAL (v0.17.0-0.17.1): Cancel button for all threaded jobs incl. 3D import (partial store
  removed); wellindex.py SQLite well index (10k wells, box query ~1 ms, exact k-nearest); dev/bench.py +
  docs/BENCHMARKS.md (0.6 GB: import 166 MB/s, slices 0.4-1.3 ms); import replaces NaN/inf samples;
  irregular-geometry test; dashboard draws all wellbores as one curve and caps the table at 3,000 rows.
  Left for item 15: run the benchmark at 5-20 GB on Chris's PC, then fix what it shows (wire the well index
  into basemap/dashboard, parallel import / overview volume only if needed).

Files: see the commits on dev (git log main..dev). New this run (seismic_tool): horizons, faults, rgt/*,
welltie, maps, mistie, attributes, velocity, laterals, dashboard_model, wellindex, analytics/*, gui/*
(horizon_dock, fault_dock, welltie_dialog, map_dialog, mistie_dialog, velocity_dialog, analytics_window,
engineering_dialog, dashboard_window, well_strip) and their test_*.py; Well Correlation: dlis.py; dev/bench.py.

Tests at the end: seismic_tool --selftest PASS, --guitest (Qt mock) PASS, well_correlation --selftest PASS,
mistie_tool --selftest PASS.

Decisions (Chris away): GUI built against the Qt mock only - every new window is real-Qt code that has never
been opened; analytics models are numpy re-implementations (xgboost/sklearn/shap optional); well index built
but not yet used by the GUI until the PC benchmark shows the need; stopped at ~10:00 UTC because the rest of
item 15 needs measurements on Chris's PC.

Known limitations: no real Qt window opened yet; 3D autotracking runs in the GUI thread; RGT between node
columns is interpolated and thin layers do not pinch to zero; 2D RGT edits do not propagate through ties;
wells vertical in the seismic tool except in the dashboard (surveys used there); tie = one shift + one
stretch; Maui checkshot LAS files hold no time-depth pairs; velocity model uses inverse-distance gridding;
dashboard colouring needs grids added from the basemap; benchmark "peak RAM" includes memory-mapped pages.

What Chris should test first:
1. Pull dev; run each tool's self-test, then `python -m seismic_tool --guitest` with real PySide6.
2. Seismic tool on NPRA 2D: horizon pick + autotrack (T), fault sticks (N), RGT (node 8) + Wheeler, and the
   new Cancel button next to the progress bar.
3. Well Correlation NZ project: Tui-1 (from DLIS) tracks; --dlis2las on the big Tui-1 DLIS.
4. Maui 3D + NZ wells: well tie, horizon maps, velocity model + Depth domain.
5. Interpret > Analytics: grids + CSV target, Train, Model QC...; Import engineering / production CSV on the
   synthetic set from dev/make_synthetic_engineering.py.
6. Interpret > Dashboard: deviated wells from surveys, colour by TVDSS / grid / seismic, histogram range
   filter, linked selection, Well strip.
7. `python dev/bench.py --gb 5 --dir D:\bench` and send me docs/BENCHMARKS.md.

Next: item 15 leftovers after the PC benchmark; then "Later (Chris back)" items (Maui subset cut, mistie with
the 2D lines, Tui-1/Amokura-1 wells).

## 2026-11-23 run 2 (scheduled) - 12:19-13:40 UTC, branch dev, Seismic v0.17.1 -> v0.21.1 (+ Well Correlation, Mistie fixes)
- Item 15 DONE (v0.18.0): basemap draws all wells as one scatter item and names only the wells in view (SQLite
  well index above 2,000 wells; all names for projects <= 500 wells); 3D import converts chunks with up to 4 threads
  (bit-identical to serial, float32 IBM conversion); section images auto-downsample when zoomed out, colour clip from
  a 1M-value sample; dashboard map pick vectorised. (Chris's interactive session meanwhile added v0.21.0 fast
  crosslines + contiguous slab writes; merged.)
- Item 16 DONE - well tie upgrade (v0.19.0-0.20.0, new tie.py + dialog rows 2/3): HRS-style Correlate (correlation vs
  lag + envelope, polarity, phase scan, cycle-skip warning), best trace within N, FFT auto tie anchored at the window
  top with NRMS + held-out score, event pairs (<= 10, undo) and constrained DTW stretch/squeeze (max lag, max 10 %
  interval-velocity change), warp saved with the tie, drift QC, checkshot-calibrated sonic (td_mode), tied TD CSV
  export, deterministic + multitaper wavelets with phase / PEP and a JSON wavelet library, despike / washout / Backus,
  Aki-Richards / Shuey angle gathers, seismic along deviated wellbores, "Tie all wells" batch (table, average
  wavelet, bulk-shift map on the basemap).
- Hardening (all plan items done): four independent code reviews of every module -> ~60 fixes with regression tests,
  v0.20.1-0.21.1. Highlights: autotracker lost fractional pick times in mode None; horizon folders could overwrite
  each other / undo-delete lost files; descending-CDP 2D lines lost navigation; survey CSVs with MD,INC,AZI,TVD...
  were read with shifted columns (wrong TVD); engineering units ("..._LBS" read as tonnes, FracFocus gallons as bbl,
  m3/e3m3 production unconverted), gap months, EUR start; RGT edits erased earlier edits, 2D ties with different t0
  were off, 2D edits now keep tie traces + update the stack; horizon misties mixed corrected / raw time; Mistie Tool
  measure returned -max-lag for short windows, and closing after a failed save lost work; real-Qt traps (Qt
  'checked' arg into slots, itemData list vs tuple, transposed well-strip image, Enter shortcut, click-through).
  Full list in docs/WHATS_NEW.md.
- Files: seismic_tool/tie.py, test_tie.py (new); welltie.py, gui/welltie_dialog.py, wellindex.py, gui/basemap.py,
  volume.py, segy.py, horizons.py, maps.py, velocity.py, mistie.py, attributes.py, rgt/{dip,solve,stack,edit}.py,
  laterals.py, dashboard_model.py, analytics/{engineering,table,models,qc,validation}.py, gui/* (most dialogs),
  test_hardening.py, test_edit.py, test_explain.py; Well Correlation geo.py, las.py, dlis.py, project.py (cache key
  bumped -> logs re-cached once); Mistie Tool app.py, gui.py, measure.py, session.py, __main__.py, selftest.py;
  launcher .bat files; docs.
- Tests at the end: seismic_tool --selftest PASS, --guitest (Qt mock) PASS, well_correlation --selftest PASS,
  mistie_tool --selftest PASS. Real-data check: Tui-1 sonic drifts to -21 ms vs its checkshot; calibrated sonic
  removes it at all 30 levels; Kiwi-1 survey still reads 106 stations.
- Known limitations: still no real Qt window opened by me (Chris's guitest with PySide6 is the first check); no 3D
  seismic in the repo to try the well tie on real data (Maui 3D subset pending); horizon constraints of earlier RGT
  edits are not re-applied by later edits (point edits are); 2D RGT editor uses default solve parameters; batch tie
  of sonic-only wells may need DTW (linear stretch cannot follow sonic drift); "Tie all wells" and 2D stack build
  run on the GUI thread; analytics median imputation happens before CV (small leak); duplicate il/xl traces in a
  SEG-Y: which copy wins is thread-timing dependent.
- What Chris should test first:
  1. Pull dev; run each tool's self-test and `python -m seismic_tool --guitest` with real PySide6 (first real-Qt
     run of the new well-tie rows, basemap labels, review fixes).
  2. Well Correlation NZ project: Kiwi-1 trajectory / TVD unchanged, other wells unchanged (logs re-cache once).
  3. Seismic tool menus: Interpret > attribute volume / build horizon stack / stack level -> horizon now ask for
     their settings (they used to get Qt's "checked" flag); double-click in the data tree opens the section.
  4. Interpret > Well tie on Maui 3D + NZ wells: Auto tie, Correlate (phase / cycle-skip text), Auto
     stretch/squeeze, Pick event pair, Undo, Estimate wavelet, Angle gather, Tie all wells (basemap shift map).
  5. Mistie Tool on NPRA: open, measure with a short window; Save then close.
  6. Basemap with your biggest well project: names appear when zoomed in, panning stays smooth.
- Next: no todo items left in PLAN; "Later (Chris back)" items (Maui subset cut, mistie with the 2D lines,
  Tui-1/Amokura-1 wells) and the PC-benchmark-driven import work Chris's session started (v0.21.0).

## 2026-11-23 - plan update (interactive session with Chris)
- Added PLAN item 17: Dynamic depth conversion (Kingdom DDC-style), building on item 12 velocity.py.
  Sub-items 17a velocity functions (V0+kZ, V0+kT, T-D function, cross-plots), 17b gridding (TPS, kriging,
  seismic-velocity guide), 17c live update when tops/horizons change, 17d well prognosis / geosteering view,
  17e scenarios, blind-well test, P10/P50/P90, depth SEG-Y. Run order 17a -> 17c -> 17b -> 17d -> 17e.
- No code changed.
- Next: PLAN item 17a.

## 2026-11-23 - run 3 (scheduled, 17:19-17:55 UTC) - PLAN item 17 finished (last open item)
- Item 17 Dynamic depth conversion (Kingdom DDC-style), all five parts, in run order 17a -> 17c -> 17b -> 17d -> 17e:
  - 17a (v0.22.0) velocity functions per layer: V0 (old default), V0 + kZ (exp form), V0 + kT, well T-D function
    (IDW / nearest); k fitted over the wells or fixed; cross-plot (V_int vs mid depth / time, line, R2, 2-sigma
    outliers); depth sections follow the function inside layers and land on the depth maps (also fixes a small
    section/map disagreement with residual correction).
  - 17c (v0.23.0) dynamic update: `ddc.DynamicDepthModel` rebuilds only layers at/below a changed top or horizon,
    maps above bit-identical, unchanged deeper layers reuse their gridded velocities; View > Auto-update depth model
    (file watcher + 1 s debounce + 2 s poll), status line "W-3 Top B top moved +4.0 m -> rebuilt layer 2 in 0.00 s";
    what-if pseudo-tops kept in the seismic project, never in the well project.
  - 17b (v0.24.0) gridding: IDW / thin-plate spline / simple kriging (fitted spherical or exponential variogram) for
    V, V0 and residuals; seismic-velocity guide per layer (RMS volume -> Dix, or interval volume; ratio or
    difference scaling to the wells).
  - 17d (v0.25.0) Interpret > Well prognosis / geosteering: horizon crossings along project wells or a planned
    vertical well (MD, TVDSS, X, Y, status, sigma, change), bit depth, drilled tops as what-if pseudo-tops with the
    horizons ahead of the bit re-predicted, live update with auto-update, CSV.
  - 17e (v0.26.0) named scenarios + difference maps, leave-one-well-out blind test (computed at points only),
    P10/P50/P90/sigma maps, prognosis sigma from the blind test, depth SEG-Y of a whole volume (threaded,
    cancellable), depth CPS-3 grid -> time.
- Files: seismic_tool/velocity.py, ddc.py (new), vgrid.py (new), prognosis.py (new), scenarios.py (new),
  test_ddc.py (new), selftest.py, gui/velocity_dialog.py, gui/main_window.py, gui/section_panel.py,
  gui/prognosis_dialog.py (new), __init__.py; docs PLAN, HANDOFF, WHATS_NEW, RUNLOG.
- Tests: seismic_tool --selftest PASS (413 checks, ~70 new), --guitest under the Qt mock PASS, well_correlation
  --selftest PASS, mistie_tool --selftest PASS. Key results: k recovered within 0.02% (V0 + kZ) and exactly (V0 + kT);
  V0 + kZ depth maps within 0.02 m vs 21 m for constant V; incremental rebuild == full rebuild (0 m); 50 wells on a
  300 x 300 map rebuild in 0.07 s; seismic guide cuts the east-side depth error from 292 m to 1 m with all wells in
  the west; blind test V0 + kZ RMS 0.01 m vs 15 m constant V; depth SEG-Y reflector within one 4 m sample of the map.
- Known limitations: no real Qt window opened (guitest with PySide6 on the PC is the first check); no real-data test
  (the Maui 3D is not imported in the repo and the Maui checkshot LAS files hold no T-D) - all checks are
  synthetic; on a full 1000 x 1500-node map each layer whose well values change costs ~0.5 s (IDW), so a top
  change in the shallowest of 6 layers may take ~3 s; what-if / drilled-top wells use the model's interval
  velocity below their top and are ignored by T-D-function layers; P10/P90 from the blind-test spread only (no
  Monte-Carlo on V0/k); depth->time of grids on a 1 ms TWT sampling; the depth SEG-Y header carries dz in mm (the
  common convention - check Kingdom/Petrel read it as metres).
- What Chris should test first:
  1. Pull dev; `python -m seismic_tool --selftest` and `--guitest` with real PySide6.
  2. Interpret > Velocity model on Maui (or NPRA) horizons + NZ wells: new columns (function, k, seismic guide),
     gridding row, Cross-plot button, Build, Use for depth display, inline in Depth.
  3. View > Auto-update depth model, then move a Maui top in the Well Correlation Tool and save: status-bar line
     and depth section/misfit update within ~2 s.
  4. Interpret > Well prognosis: a Maui well and a planned vertical well; Set bit, Add drilled top, Export CSV.
  5. Blind test -> P90 map; Save scenario + Difference map.
  6. Depth-convert a (small) volume to SEG-Y, Cancel once, then load the file in Kingdom/Petrel as depth.
- Next: PLAN has no todo/partial items left. "Later (Chris back)" items remain (PC benchmark, Maui 3D subset,
  mistie with 2D lines, Tui-1/Amokura-1 wells) plus the optional item-17 follow-ups noted in PLAN.
