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

## 2026-11-23 - plan update (interactive session with Chris)
- Added PLAN item 18 Basin model integrated with the Wheeler diagram (18a chronostratigraphy + age-axis Wheeler,
  18b burial history / decompaction, 18c thermal + maturity with BHT calibration and EASY%Ro, 18d Wheeler / map
  integration + event chart, 18e GUI) and item 19 Seismic inversion (19a low-frequency model, 19b post-stack
  coloured + model-based, 19c pre-stack simultaneous, 19d GUI + QC). Chris: do 18 first.
- No code changed.
- Next: PLAN item 18a.

## 2026-11-23 - run 4 (scheduled, 22:18-23:12 UTC) - PLAN items 18 and 19 finished (last open items)
- Item 18 Basin model integrated with the Wheeler diagram (v0.27.0-v0.31.0), new package seismic_tool/basin:
  - 18a age table (Taranaki template for the Maui formations - approximate ages, editable), RGT stack levels dated by
    interpolation between dated horizons, section domain "Wheeler (age)" with labelled hiatus gaps, zone lithology from
    the well logs.
  - 18b burial history: Athy decompaction per lithology mix, irreversible compaction, erosion events, paleo water depth.
  - 18c thermal: transient conduction on the moving burial grid (constant / piecewise / McKenzie heat flow), BHTs from
    LAS and DLIS headers (Horner / single-BHT correction, implausible gradients flagged), heat-flow calibration;
    maturity: EASY%Ro, Lopatin TTI, kerogen TR (Pepper & Corvi kinetics), oil / gas window timing.
  - 18d batched pseudo-wells (~5,000 columns in ~3 s): maps of T / %Ro / TR / oil-window entry at a surface, Wheeler
    coloured by T / %Ro / TR / porosity today or at a past age, restored sections, petroleum-system event chart.
  - 18e Interpret > Basin model window; the map re-runs when the dynamic depth model (item 17) changes.
- Item 19 Seismic inversion (v0.32.0-v0.35.0), new package seismic_tool/inversion:
  - 19a LFM from the tied wells through the horizon / RGT framework (IDW, kriging, trend-guided), Butterworth low-pass
    with the cut picked from the seismic spectrum, stored as a volume.
  - 19b coloured inversion; model-based ln-AI inversion (CG, Tikhonov or blocky IRLS, lateral smoothing), wavelet scaled
    at the wells, residual NRMS, blind-well test.
  - 19c pre-stack simultaneous inversion of partial angle stacks (Fatti + well background trends) -> Zp, Zs, density,
    Vp/Vs, lambda-rho, mu-rho.
  - 19d Interpret > Seismic inversion window (LFM, wavelet, post / pre-stack, QC plots, blind-well table, cross-plots
    coloured by a volume or a well log, SEG-Y export); inversion volumes get a positive rainbow colour range.
- Files: seismic_tool/basin/{__init__,chrono,burial,thermal,maturity,pseudowells,model}.py, seismic_tool/inversion/
  {__init__,common,lfm,poststack,prestack}.py, gui/basin_dialog.py, gui/inversion_dialog.py (new); gui/section_panel.py
  (Wheeler (age) domain, basin colouring, inversion colours), gui/main_window.py (2 menu items, ddc_tick hook),
  selftest.py, test_basin.py, test_inversion.py (new), __init__.py; docs PLAN, HANDOFF, WHATS_NEW, RUNLOG.
- Tests: seismic_tool --selftest PASS (498 checks, ~90 new), --guitest under the Qt mock PASS (216, ~30 new),
  well_correlation --selftest PASS, mistie_tool --selftest PASS. Key results: steady-state geotherm = analytic
  (< 0.001 degC), heat-flow fit recovers 65.00 mW/m2, EASY%Ro = independent quadrature, grain thickness conserved,
  batched pseudo-wells = single-column model; inversion blind-trace AI error 2.2 % in band (LFM alone 6.5 %), 2.7 %
  with 15 % noise, simultaneous Zp 2.3 % / Zs 3.9 % / Vp/Vs 2.2 %. Real data: BHTs read from 8 NZ wells (Tui-1 DLIS
  116 degC at 3911 m); Maui-5 / 7 and Tui-1 burial + %Ro runs (present %Ro ~0.5-0.65 at Kapuni level).
- Known limitations: no real Qt window opened; Taranaki ages approximate; calibrated Maui heat flow ~35-38 mW/m2 with
  the default (Sclater-Christie) shale conductivity - lower than published ~55-65, so the default shale matrix k is
  probably too low for these mudstones (edit LITHOLOGY or the age-table lithology); no temperature-dependent
  conductivity, isostasy or heat-flow maps; inversion tested on synthetics only, 3D only (no 2D lines), ~5-10 ms per
  trace (pre-stack ~3x), one stationary wavelet, no partial stacks in the public data.
- What Chris should test first:
  1. Pull dev; `python -m seismic_tool --selftest` and `--guitest` with real PySide6.
  2. Interpret > Basin model on the NZ project: check / edit the age table, set horizon names, Save; Wells tab: Maui-5
     Run well, Calibrate heat flow to BHT.
  3. An inline in "Wheeler (age)" (needs RGT + two dated horizons), then Maps tab: Colour Wheeler by %Ro; Run map at a
     deep surface (needs the item-17 depth model) and Export CPS-3.
  4. Interpret > Seismic inversion on a small Maui subset with tied wells: Build LFM (AI), Set wavelet (library),
     Run model-based, QC at well, Blind-well test.
  5. Export the AI volume as SEG-Y and load it in Kingdom/Petrel.
- Next: PLAN has no todo/partial items left. "Later (Chris back)" items remain (PC benchmark, Maui 3D subset, mistie
  with the 2D lines, Tui-1/Amokura-1 wells).

## 2026-11-24 - interactive (Chris): Phase 2 plan added
- Chris's brainstorm (exploration, petrophysics incl. Geolog / IP ideas, M&A / reservoir engineering, general viewer
  ideas) turned into PLAN items 20-30, all `todo`. Order: 20 petrophysics core -> 21 integrated map hub -> 22 petro
  analytics + field reports -> 23 co-blend / bump map / linking -> 24 resolution, single-well inversion, predictive
  2D modelling, synthetics as RGT seeds -> 25 AVO / DHI / fizz gas / angle-stack risk -> 26 structural timing,
  paleo-structure animations, restoration -> 27 migration + fault / top seal -> 28 Rose-style risking + volumetrics
  -> 29 spacing, stress, drainage ellipses, gun barrel -> 30 depletion factor, resource left, RF, RTA, frac
  complexity, M&A screen. Design decisions are listed at the top of the Phase 2 section in PLAN.md.
- No code changed. Next: item 20a.

## 2026-11-24 - interactive (Chris): map cartography
- Chris asked whether the mapping could look better. Made a before/after mock-up (synthetic dome + fault; matplotlib
  hillshade, index contours, fault heave, well symbols, spill contour, scale bar, north arrow) and added PLAN 21f
  (shared cartography / map-style module). No code changed.

## 2026-11-24 - interactive (Chris): ArcGIS-style maps
- Chris asked for "pretty maps like ArcGIS". Made an A3 layout mock-up (shaded relief land + bathymetry, coastline,
  permit, 2D lines, 3D outline, hillshaded structure inset, title block, locator inset, legend, scale bar, north
  arrow, lat/long graticule) and added PLAN 21g (layout composer, offline cartographic context, GIS export to
  ArcGIS Pro / QGIS). No code changed.

## 2026-11-24 - interactive (Chris): any shapefile layers
- Chris: keep the plan order as is; include any other shapefile data (township / range, county, infrastructure).
  Added to PLAN 21g: generic vector layer loader + styling by attribute, labels, attribute table, PLSS legal-
  description search, reprojection (TM / UTM, LCC State Plane, Albers, geographic, NZTM) in numpy, large-layer
  spatial index + simplification. No code changed.

## 2026-11-24 - interactive (Chris): QGIS link
- Chris has QGIS (no ArcGIS Pro) and wants it interactive. Added PLAN 21h: live shared GeoPackage + GeoTIFF with
  overviews, a QGIS plugin (layers + styles, auto-refresh), and a localhost two-way message link (selection, click
  to open, draw line -> seismic section, draw polygon -> prospect, cursor marker). No code changed.

## 2026-11-24 - run 5 (scheduled, 07:18-10:10 UTC) - PLAN items 20-30 (all PLAN items now done)
- 20a done (well_correlation v0.6.0): new package `well_correlation/petro/` - tool / vendor / service identification
  (DLIS TOOL objects, mnemonics, file-name services, descriptions; editable tool table), wireline vs LWD, effective
  vertical resolution, curve QC (gaps, flat lines, spikes, bad hole, GR depth shift between runs, splices), bed
  boundaries + thin-bed flags; CLI `--petroqc`, `--tooltable`. Real data: Tui-1 HLLD -> HALS from the DLIS TOOL object
  (SLB wireline), Kiwi-1 = SLB D&M LWD (ARC), Maui-5 DLL / DIL / LDL / CNL / BHC from LAS file names and mnemonics.
- 20b done (v0.7.0): `petro/interp.py` - Vsh (GR linear / Larionov / Clavier / Steiber, SP, N-D, min), porosity (density,
  sonic Wyllie / RHG, N-D with gas correction, shale-corrected PHIE), Rw (Rwa min, Pickett fit, SP, Arps), Sw (Archie,
  Simandoux, modified Simandoux, Indonesia, Waxman-Smits, dual water), perm (Timur, Coates, Wyllie-Rose), Passey TOC,
  cutoffs -> net res / net pay / PHIH / SoPhiH / kh per zone, cutoff sensitivity, parameter sets per zone and well;
  CLI `--petro`. Real data (default parameters): Maui-7 C-sands net pay 35 m TVD, PHIE 0.18, Sw 0.14; Maui-5 C-sands
  5.8 m (median Rt 2.3 ohm.m - mostly wet there), Tui-1 F1 9.6 m PHIE 0.16 Sw 0.30.
- 20c done (v0.8.0): `petro/multimin.py` - constrained multi-mineral solver (exact WLS with sum-to-one + non-negativity,
  batched over samples: 20,000 samples x 6 components in ~0.1 s), editable mineral / fluid end-points, per-zone mineral
  sets, reconstructed logs + misfit; Monte Carlo over Rw / m / n / matrix / cut-offs / end-points -> P90 / P50 / P10 per
  zone; barite-mud PE warning in QC (Rahi-1 flagged). Real data: Maui-7 C-sands deterministic PHIE 0.176 vs multimin 0.169
  (without PE); with PE the solver invents ~40 % calcite (PE ~3 in clean sand) - hence the warning. Maui-7 C-sands
  Monte-Carlo SoPhiH P90 / P50 / P10 = 4.9 / 5.4 / 5.8 m.
- 20d done (v0.9.0): `petro/rockphysics.py` - Batzle-Wang brine / gas / oil, VRH, Wood / Brie fluid mixing, Gassmann,
  brine / oil / gas / fizz substitution, Greenberg-Castagna Vs (iterated with Gassmann in hydrocarbons), elastic logs +
  Rickman brittleness; CLI `--rockphysics`. Checks: pure water 1482 m/s / 0.997 g/cc, Gassmann + substitution round
  trips exact, fizz (7 % gas) gives 84 % of the commercial-gas Vp drop but 9 % of its density drop. Real data: Tui-1
  uses measured DTSM where logged, median Vp/Vs 1.76.
- 20e done (v0.10.0): Petrophysics window (`gui/petro_dialog.py`) - parameter table with sources and 4 edit scopes, run
  buttons (QC, interpret well / section / all, mineral solver, Monte Carlo, rock physics), cross-plots (N-D, M-N,
  Pickett, Buckles, histogram with discriminators) drawn with pyqtgraph and exportable as PNG (`petro/plotdata.py`),
  zone results table, "Show on map hub" (summary.csv + hub_request.json), section result tracks VSH / PHI / SW / PERM.
  New `python -m well_correlation --guitest`. Item 20 done. Maui-7 C-sands N-D shows the gas crossover cluster and
  the Pickett plot a clean water line through the wet sands plus the gas leg at 100-1000 ohm.m.
- Item 21 started. 21a done (Seismic v0.36.0): `maplayers.py` layer stack (grid, contours with index lines, horizon,
  polygons, text, petro + the legacy outlines / 2D lines / wells / faults) with order / visibility / opacity / style,
  saved in the project (grids as npz), Qt-free renderer + matplotlib PNG; basemap draws from the stack; Map layers dock.
  21b part: petro zone values per well as labelled bubbles from `<wcproj>_petro/summary.csv`.
- 21b done (Seismic v0.37.0): `maphub.py` + `gui/petro_strip.py` - mineral pies (well_correlation multimin now writes
  mm_summary.csv), petro bubbles per zone, "Show on map hub" watcher, click a well -> petro strip, draw a polygon ->
  wells inside selected -> linked cross-plot of zone values (or a grid at the wells).
- 21c done (Seismic v0.38.0): `propgrid.py` + `gui/propgrid_dialog.py` - zone property gridding (IDW, TPS, kriging, KED,
  collocated cokriging), correlation, leave-one-out table + residual map, grid calculator, mask by polygon. Checks:
  cokriging with rho = 1 = the secondary trend (< 1e-6), rho = 0 = simple kriging; on a synthetic with a secondary-
  driven trend LOO RMSE KED 0.085 / CCK 0.187 vs kriging 0.491 / IDW 0.750.
- 21d done (Seismic v0.39.0, well_correlation Project.add_from_folder): `addwells.py` "New wells" workflow - import, petro
  batch, blind residuals of the old maps at the new wells, re-grid + change maps, batch tie, depth-model rebuild; LFM
  only flagged. Synthetic check: E-1 SoPhiH 7.69 (true 7.70); re-gridded map at E-1 7.68.
- 21e done (Seismic v0.40.0): `crosscheck.py` + CrossCheckDialog - zone value vs grid layers at the wells (multiple
  regression, LOO), predicted P50 / P90 / P10 / residual / corrected maps, Send to Analytics. Synthetic: coefficients
  recovered, the 80 % interval covers 83 % of the wells.
- 21f done (Seismic v0.41.0, well_correlation v0.11.0): `well_correlation/carto.py` (hillshade, ramps, chained +
  labelled contours with collision avoidance, fault-polygon blanking, per-fault-block gridding, downthrown ticks, spill
  point, well status symbols, scale bar, grid convergence, presets) used by the map hub renderer, `mapexport.py`
  (300 dpi PDF / PNG with title block, legend, scale bar, north arrow) and the Well Correlation map. Checks: hillshade of
  a plane = analytic, contours never cross a fault polygon, label boxes never overlap, spill depth of a cone with a
  saddle channel exact, fault-block gridding does not smear across the fault.
- 21g done (Seismic v0.42.0): `seismic_tool/gis/` - projections (TM / UTM / NZTM, LCC, Albers, Mercator; WKT in / out;
  EPSG table; NAD27 shift flagged approximate) verified against Snyder and EPSG examples to < 0.1 m (TM inverse round
  trip 0.000 mm over NZ), shapefile / GeoTIFF / GeoPackage / KML read + write (round trips exact), generic vector layers
  (filter, categorised / graduated styles, identify, Douglas-Peucker: 1M-vertex line simplified in ~1 s), PLSS legal
  description spotting, GIS export of the map, page layout with locator, graticule, neatline, templates.
- 21h done (Seismic v0.43.0): `qgislink.py` (shared GeoPackage with QGIS default styles + grid GeoTIFFs with overviews,
  user-edit tables read back with change detection, localhost JSON-lines link), QGIS plugin `tools/QGIS Plugin/
  geoharmonic_link` (+ dev/build_qgis_plugin.py, dev/fakeqgis), arbitrary seismic section from a QGIS line, View menu
  entries. Protocol tested between two processes (selection, polygon, section line, section cursor round trip). Item 21
  done.
- Item 22 done (well_correlation v0.12.0, Seismic v0.44.0): `petro/logml.py` - curve prediction (GBM + quantile P10 / P90)
  with depth-blocked CV (synthetic: random K-fold R2 0.22 vs blocked 0.17 - the leakage the blocking removes; blind well
  RMSE 4.4 us/ft against an irreducible 4.1, band covers 80 %), GMM / k-means electrofacies (100 % on synthetic facies);
  `petro/reports.py` - report tables + text keyword extraction, pressure gradients + contact (2050.3 m vs 2050 true),
  Haworth gas ratios (Tui-1 3550-3750 m: median wetness 14 %, gas / condensate), core-log calibration, Archie m / n fits.
- Item 23 done (Seismic v0.45.0): `blend.py` - RGB / CMY / HSV co-blending with per-channel clip + gamma, colour-triangle
  legend, attribute overlay on the seismic, single-light illumination (tilted planes match cos(normal, light) exactly),
  amplitude on structure, bump-mapped sections; map hub blend layer + shade-by-structure; section co-blend / overlay /
  bump; `linking.py` linked cursor (section <-> basemap, dashboard, petro strip), attribute parameter scan (movie),
  saved window layouts + templates.
- Item 24 done (Seismic v0.46.0-v0.47.0): `resolution.py` (spectrum + usable band vs noise floor, frequency map, tuning,
  wedge (30 Hz Ricker tuning 14.0 ms = Kallweit-Wood 14.4), bed thicknesses, spectral balancing, recommendation;
  single-well Monte-Carlo inversion: clean synthetic r 0.93 with 91 % of the band-limited log inside P10-P90, noisy
  case flagged NO), `wellseeds.py` (tops / synthetic events -> snapped seed horizons; RGT spread at 5 wells across a
  fault 23 -> 0.7 with seeds), `predictive.py` (pseudo-wells with Gassmann, Aki-Richards / Shuey, Bayesian classifier:
  86 % on independent pseudo-wells, 95 % of traces of a synthetic brine / gas section, P(fluid) volumes).
- Item 25 done (Seismic v0.48.0): `avo.py` - exact Zoeppritz (= AI reflectivity at 0 deg, Aki-Richards within 0.003 to
  30 deg), AVO classes I-IV from known contrasts, background trend + anomaly z-scores, fluid factor, Gassmann scenarios;
  fizz vs commercial gas from density (gas -7.5 % vs fizz -0.7 % density at 25 % porosity; Vp -17 % vs -15 %); angle-
  stack risk (density error 2 stacks >= 3 >= 5 >= gather, halves when the far angle goes 35 -> 50 deg; MC confirms);
  DHI checklist with conformance (fitting anomaly 0.9+ vs cutting anomaly < 0.5) and weighted index; GUI: AVO volumes,
  angle-stack risk, DHI from the Map layers dock.
- Item 26 done (Seismic v0.49.0): `structure.py` - phases in the age table + Wheeler colours, fault growth (synthetic
  growth fault: EI 1.0 / 1.0 / 1.6 / 1.6 / 1.0 recovered, syn-rift proposed for the right two intervals), flattening
  (horizon on the datum to 1 sample), decompacted paleo-depths (grains conserved to 1e-6), paleo animation GIF, block
  restoration (30 ms throw recovered, restored section matches the unfaulted one to 5 % RMS, line length / area
  unchanged); GUI: paleo animation of the current section.
- Item 27 done (Seismic v0.50.0): `migration.py` - flow paths / traps / spill analysis (cone: spill exactly at the
  nearest edge; two domes on a regional dip: deep trap spills at the saddle into the up-dip trap, which leaks off the
  map), fill-spill order, critical moment, juxtaposition, SGR (hand case 0.5), SGR column (22.5 bar -> 294 m gas), slip
  tendency peaking on ~60 deg normal faults, MICP conversion, top-seal and limiting column; map-hub migration action.
- Item 28 done (Seismic v0.51.0): `prospect.py` - lognormal P90 / P10 maths (sampled within 2 %), GRV of a cone within
  1 % of pi R^2 H / 3, area-depth integration check, deterministic + Monte-Carlo volumetrics, contact from the limiting
  column, Pg product, DHI revision (bounded, monotonic, neutral at 0.3), portfolio with shared source (P(any) 0.75 ->
  0.50, same risked volume), Rose-style CSV; LayersPanel.prospect on a trap of the map hub.
- Item 29 done (Seismic v0.52.0): `spacing.py` - RRC / FracFocus CSV mapping, WSM SHmax (axial mean of 175 and 5 deg
  = 0), distances on a known wine-rack pattern (200 m / 60 m vertical, 400 m same zone), parent / child / co-completed,
  wells per section, ellipses along SHmax (major axis at 45.0 +- 3 deg), overlap areas, gun-barrel offsets exactly
  200 m; dashboard Spacing + Gun barrel.
- Item 30 done (Seismic v0.53.0): `depletion.py` - depletion factor on a synthetic parent-child pair (= cum / in-place
  when stacked, falls to 0 as the child moves away), child performance vs DF slope, RF hand case 0.10, remaining map,
  7 undrilled 200 m slots beside existing drainage, RTA on an analytic linear-flow well (slope 1/C within 2 %, end of
  linear flow 300-450 d for true 300 d), frac complexity index, NPV10 equal to a hand cash flow, type curves, inventory,
  HTML summary; dashboard recovery / depletion.
- Tests (end of run, all passing): `python -m seismic_tool --selftest` and `--guitest` (fakeqt), `python -m
  well_correlation --selftest` (+ `--guitest` earlier in the run), `python -m mistie_tool --selftest`. New test files:
  well_correlation petro/test_petro.py, test_carto.py; seismic_tool test_maplayers, test_propgrid, test_addwells,
  test_crosscheck, test_gis, test_qgislink, test_blend, test_resolution, test_avo, test_structure, test_migration,
  test_prospect, test_spacing, test_depletion (+ additions to test_inversion GUI).
- Versions: Well Correlation Tool v0.12.0, Seismic Interpretation Tool v0.53.0, QGIS plugin geoharmonic_link 0.1.
- Known limitations: everything is tested on synthetic data under the Qt mock - the new windows / buttons have not
  been seen in real Qt yet. Items 26-30 are mostly API + one action each (paleo GIF, migration + prospect on the map hub,
  spacing / gun barrel / recovery on the dashboard); their full windows are listed as follow-ups in PLAN. Petro on real
  wells used default parameters. The AVO / predictive rock physics uses a generic soft-sand frame (frame_exp 2) - set
  Scenario values for your basin. RRC / NDIC column names are mapped from common export headers, not yet checked on
  your files. Large grids: migration spill analysis is pure Python in the union loop (fine to ~500k nodes).
- What Chris should test first:
  1. Pull `dev`, run both tools' self-tests and `--guitest` with real PySide6 (first real-Qt run of items 20-30).
  2. Well Correlation > Petrophysics on Maui-7: QC, interpret, mineral solver, Show on map hub; check the C-sands numbers
     (net pay ~35 m, PHIE ~0.18, Sw ~0.14).
  3. Seismic map hub: Map layers dock - add a horizon map, contours, petro bubbles, Apply style "Print", Export
     map PDF; then Send map to QGIS and open it with the plugin (dev/build_qgis_plugin.py builds the zip).
  4. On a 3D inline: View > Co-blend (10/20/40 Hz), move the mouse and watch the basemap cross (linked cursor).
  5. Seismic inversion window: set wavelet, Resolution..., Single-well inversion at a tied well; Pre-stack tab:
     AVO volumes and Angle-stack risk with your near / far stacks.
  6. Interpret > Paleo-structure animation on an inline with 3+ horizons; Map layers > Add > Migration on a depth map.
- Next: no PLAN items left - add new ones (the follow-ups noted on items 20-30 in PLAN.md are good candidates), plus the
  "Later (Chris back)" list (bench on the PC, Maui 3D subset, mistie integration).


## 2026-11-25 (interactive session with Chris) - first real-Qt install on Chris's PC, fixes, installer, F3 setup
First time the tools ran under real PySide6 on Chris's laptop (Anaconda Python 3.11.3, PySide6 6.5.3). Final result
on the PC: all self-tests and GUI tests pass (Well Correlation, Seismic 1,057 checks, Mistie), installer "Failures: 0".
- Bugs found by the real-Qt run and fixed:
  1. Well Correlation `gui/petro_dialog.py`: `QDialog.__init__` rejected a non-widget parent (GUI test stand-in) -
     falls back to a top-level dialog.
  2. Seismic `gui/welltie_dialog.py`: "best trace within N" spin box started at 0 in real Qt, so Find best trace only
     compared one trace - default 1 (3 x 3 traces).
  3. Seismic `gui/basin_dialog.py`: `set_settings()` did not update the widgets, so Save read the old heat flow back -
     now syncs the widgets (`_sync=False` when called from `settings_from_widgets`).
  4. Seismic `dashboard_model.py`: `histogram()` crashed (numpy "Too many bins for data range") on constant or
     near-constant large values (e.g. coordinates) - one-bin fallback.
  5. Seismic `test_qgislink.py`: the QGIS mock needs `dev/fakeqt/anything.py` - appended (last) to sys.path so it can
     never shadow the real PySide6; the crash had cut off the last four GUI test groups.
- New: Well Correlation `project.py` - `wells_header.csv` may carry projected `X`, `Y` (or `Easting`/`Northing`), which
  win over Latitude/Longitude (the importer otherwise projects to NZTM). Needed for F3 (ED50 / UTM 31N).
- New: one-click Windows setup at the repo root (all written without `( )` blocks - a ")" in the folder path breaks them):
  `Install GeoHarmonic.bat` (finds Anaconda, per-tool .venv incl. Mistie, runs every test, desktop shortcuts,
  install_log.txt), `Run all tests.bat` (test_summary.txt), `Uninstall GeoHarmonic.bat`, `START HERE.txt`,
  `Set up F3 project.bat`.
- New: `tools/guitest_runner.py` - runs `--guitest` under real Qt offscreen with every modal auto-answered (a hidden
  dialog hung the first run), line-buffered output and a faulthandler watchdog (10 / 20 min) that dumps the stack.
- New: Mistie Tool gets its own .venv (numpy, scipy, matplotlib): Chris's Anaconda base has a numpy/scipy mismatch
  ("_ARRAY_API not found") that broke scipy. `Mistie Tool.bat` / `Run self-test.bat` use .venv (TCL/TK_LIBRARY set from
  Anaconda), falling back to Anaconda.
- New: `tools/setup_f3.py` - builds Documents\F3 Netherlands from the F3 Demo 2023 download (dGB / TerraNubis,
  CC BY-SA 3.0) using its Rawdata exchange files: 4 wells (LAS, checkshot CSV, 39 marker tops), Well Correlation
  project, 3D import (600,515 traces, IL 100-750, XL 300-1250, 4-1848 ms) with the exact bin grid from the OpendTect
  .survey (header fit was within 0.1 m), 7 horizons, Fault A, linked seismic project. Ran on the PC in ~2 min.
- Known on the PC: harmless "NumPy 1.x" warning from shiboken 6.5.3 under numpy 2.4; Qt "no font directory" note.
- Tests here: all three self-tests and both --guitest (fakeqt) pass; setup_f3.py tested on the real F3 well, horizon,
  fault and survey files with a synthetic SEG-Y on the F3 bin grid.
- Next: Lesson 4 (F3) of the tutorial; PLAN Phase 3 (items 31-36).


## 2026-11-25 (scheduled run 6, 07:18-08:45 UTC) - items 31, 32, 37, 33, 34, 35, 36: ALL PLAN ITEMS DONE
All open PLAN items are finished; nothing is left in todo / partial. No real Qt / OpenGL here - everything GUI-side
ran under the Qt mock, so the first real-Qt look on the PC matters most (checklist below).
- PLAN: added item 37 "Velocity-volume depth conversion" (placed after 32, before the 3D items: 35c uses it). The Live
  Map (31) and 3D Viewer (33) items already existed, so those one-time additions were skipped.
- 31 Live Map (Seismic v0.54, Well Correlation v0.13, Mistie 2.1): shared map bus (`mapbus.py`: selection, filters,
  active line / inline / well / section, DataChanged, WellTable column store; cross-process through
  `<well project>_mapbus/` events.jsonl + state.json + welltable.csv) and retrofit of all three tools; View > Live Map
  (dark, Spotfire-style well styling incl. rule colours, filters, lasso, linked table / crossplot / histogram, scale
  bar, north arrow, readout, culture presets + GeoJSON, current-section line, arbitrary line, shotpoints, mistie
  markers, well sections into the correlation tool, live gridding with fault barriers, live depth maps, layers from
  other modules, PNG / PDF and GIS / ZMAP+ export).
- 32 Automated fault mapping (v0.55): fault likelihood / TFL / strike / dip volumes, fault skins (crossing faults kept
  apart), table + filters + stereonet / rose + merge assistant, faults with sticks, throws (RGT or correlation),
  fault polygons on horizons (gridding barriers), 2D-line cuts correlated across lines. Interpret menu.
- 37 Velocity-volume depth conversion (v0.56): well-tie velocities along the horizon framework (+ optional Dix trend
  from stacking velocities), maps tie every top exactly, calibrated volume, dynamic re-tie, blind test vs layer-cake.
- 33 3D Viewer (v0.57): OpenGL window (PySide6 only) with software fallback; planes (coarse while dragging), horizons,
  faults, wells + tops, culture, VE, saved views, screenshot, turntable GIF; linked both ways through the bus.
- 34 3D interpretation (v0.58): pick horizons / fault sticks on planes, track from 3D seeds, volume rendering,
  geobodies (+ outlines on the Live Map), any volume on the planes.
- 35 Framework + 3D grid (v0.59): fault traces, per-block horizons + rules, depth, stair-stepped grid, GRDECL / ZMAP+.
- 36 Properties + volumetrics (v0.60): upscaling, variograms, SIS / SGS, volumetrics P90 / P50 / P10, GRDECL / CSV.
- Files (new): seismic_tool/mapbus.py, livemap.py, livegrid.py, faultlik.py, faultskin.py, velvolume.py, scene3d.py,
  geobody.py, framework.py, propmodel.py, gis/geojson.py, gis/zmap.py, gui/maplink.py, live_map.py, faultmap_dialog.py,
  velvolume_dialog.py, view3d.py, framework_dialog.py, propmodel_dialog.py + test_* for each; well_correlation/
  maplink.py + test_maplink.py; mistie_tool/maplink.py; dev/fakeqt PySide6.QtOpenGL / QtOpenGLWidgets mocks.
  Changed: main windows of all three tools, basemap / layers_dock / maplayers / mapexport / gis vector + export,
  mistie / basin dialogs, synth.py, selftests. Docs: PLAN, HANDOFF (incl. 3D step-by-step), WHATS_NEW, CLAUDE.md unchanged.
- Tests: Seismic selftest PASSED (1017 checks), Seismic --guitest under the Qt mock PASSED (383), Well Correlation
  selftest PASSED (188) + guitest PASSED (30), Mistie selftest PASSED (incl. the map-bus link).
- Known limitations: no real Qt / GPU run yet (3D Viewer OpenGL path only exercised with mocks); SGS / SIS are per-cell
  Python loops (fine to ~100k cells); fault orientation uses a structure-tensor estimate instead of a brute-force scan;
  polygon fills ignore holes; the Dashboard keeps its own selection model; follow-ups are listed on each PLAN item.
- Test this first (Windows, after pulling dev):
  1. `Run all tests.bat` - all self-tests and GUI tests (real Qt offscreen) must pass.
  2. Seismic tool on F3: View > 3D Viewer - OpenGL or "software rendering"? Drag the Inline slider, orbit, zoom.
  3. View > Live Map: style wells by a column + a rule, add a filter, lasso; open the Well Correlation Tool on the same
     well project and select a well there - it must light up in both seismic windows.
  4. Interpret > Automated fault mapping on F3: 1. volumes (note the time), 2. extract, 3. create faults.
  5. Interpret > Velocity volume (advanced): build, blind test vs layer-cake.
  6. Interpret > Structural framework + 3D grid > 1-3, Export GRDECL, open in ResInsight; 4. Properties + volumetrics.
- Next: no PLAN items left - add new ones (candidates: the follow-ups noted on items 31-37, a guided F3 walk-through).


## 2026-11-25 (scheduled run 7, 17:18-17:50 UTC) - item 38 PC test fixes + illustrated tutorial: ALL PLAN ITEMS DONE
Item 38 was added to PLAN from your 25 Nov PC install (Live Map, velocity volume and 3D items already existed, so
those additions were skipped). It is finished; no todo / partial items are left.
- 38a 3D Viewer crash: `gui/view3d.py` no longer caches `context().functions()` (Qt re-used it for the shader
  program -> "'QOpenGLShaderProgram' object has no attribute 'glClearColor'"); it is fetched in every initializeGL /
  paintGL / resizeGL (`GLCanvas._funcs`), GL buffers reset on a new context, new resizeGL. Every test module in the
  Seismic self-test / GUI test runs through `selftest._guarded` (a crash = one FAIL + traceback, the rest still runs);
  same in the Well Correlation GUI test; `tools/guitest_runner.py` turns any remaining crash into "GUI TEST FAILED".
  New real-Qt regression `test_scene3d.run_gl_real` (create, initializeGL, paintGL x2, resize, paintGL, plane change,
  grabFramebuffer; SKIP + software-fallback check when no GL context).
- 38b `"numpy<2"` in every pip line that installs PySide6 or the analytics extras (Install GeoHarmonic.bat, both
  tools' Install (run once).bat, analytics extras) - existing .venvs are downgraded on the next install; install_log
  now prints the numpy version. One numpy-2-only call (np.trapezoid in test_basin) made 1.26-safe; a scan test guards it.
- 38c guitest_runner sets QT_QPA_FONTDIR=C:\Windows\Fonts on Windows.
- 38d/e Tutorial: `dev/build_tutorial.py` (one command) + `dev/tutorial_content.py` -> docs/TUTORIAL.html (2.1 MB,
  self-contained) + docs/TUTORIAL.md + docs/tutorial/*.png. 19 chapters: install / update, fixes in this release, then
  every tool (well correlation + petro, mistie, seismic + tracking, RGT / Wheeler, well tie, velocity + DDC + velocity
  volume, map hub + Live Map + bus, basin, inversion + resolution / AVO, analytics + dashboard, fault mapping, 3D
  Viewer, 3D interpretation, framework + grid, properties + volumetrics, prospects / migration / spacing). Each has
  purpose (vs Kingdom / Petrel / HRS / PaleoScan), pictures with numbered callouts + legend, click-by-click steps, the
  test numbers (quoted from the actual test output), limits, troubleshooting. Install GeoHarmonic.bat step 6 rebuilds
  it on your PC with real window screenshots (tutorial_local\, gitignored) and makes the "GeoHarmonic - Tutorial"
  desktop shortcut; START HERE.txt points to it. CLAUDE.md rule: rebuild after any window change, new chapter per module.
- Also: 3D Viewer Fit frames only visible objects (a far-away hidden 2D line made the scene a dot).
- Files: gui/view3d.py, scene3d.py, selftest.py, test_scene3d.py, test_basin.py, test_pcfixes.py (new),
  well_correlation/__main__.py, tools/guitest_runner.py, the 4 .bat installers, START HERE.txt, .gitignore,
  dev/build_tutorial.py + dev/tutorial_content.py (new), docs/TUTORIAL.*, docs/tutorial/, PLAN, HANDOFF, WHATS_NEW,
  CLAUDE.md. Versions: Seismic 0.61.0, Well Correlation 0.13.1.
- Tests (here, no real Qt): Seismic selftest PASSED (1037 checks incl. 20 new 38a-e), Seismic --guitest under the Qt
  mock PASSED (385), Well Correlation selftest PASSED (188) + guitest PASSED (30), Mistie selftest PASSED.
- Known limitations: no Qt / GPU here, so the real-OpenGL regression and the real window screenshots only run on
  your PC - on the build server 6 of 20 pictures exist (the tools' own exports: 3D software render, Live Map export,
  horizon map, seismic inline, NZ well section); the other 14 captions say "not captured" until your installer
  rebuilds the tutorial. Window pictures show each window as opened on the demo project (not after running each
  workflow); the demo wells have no logs, so the well-tie window will not open in the tutorial build.
- Test this first (Windows):
  1. Double-click Update_GeoHarmonic.bat, then Install GeoHarmonic.bat - must end with ALL GOOD (install_log.txt:
     "Qt OK - PySide6 6.5.3 - numpy 1.26.x" for both tools).
  2. tools\Seismic Interpretation Tool\test_results.txt: the "38a: real OpenGL ..." lines PASS (or SKIP + software
     fallback) and the GUI test runs to the end - GUI TEST PASSED.
  3. Start the Seismic tool: no NumPy 1.x warning; View > 3D Viewer (Ctrl+Shift+3) opens without "software rendering".
  4. Desktop shortcut "GeoHarmonic - Tutorial": check the window screenshots; tell Claude which still say "not captured".
- Next: no PLAN items left - add new ones (candidates: follow-ups on items 31-38, result screenshots per workflow,
  a guided F3 walk-through chapter).

## 2026-11-25 (scheduled run 8, 20:21-21:30 UTC) - items 39, 40, 41, 42: ALL PLAN ITEMS DONE
All open PLAN items are done (39 Glow / Halo, 40 Aesthetics, 41 Seismic Conditioning, 42 Survey blending - added to
PLAN.md at the start of this run from your 25 Nov requests). Seismic v0.65.0, Well Correlation v0.14.0.
- 39 Glow / Halo: G (Display > Glow on/off) toggles a glow under horizons, faults, wells on the section; Display > Glow
  settings (Glow / Halo / Off, width, intensity, 1-4 falloff layers, colour, halo auto black / white against the
  colour table, active-only, hover); right-click a horizon / fault > Glow > On / Off / Use global / Colour; the
  active / selected object glows stronger in every window via the map bus (new 'display' topic); Live Map glow on
  selected wells, section lines, red glow on mistie hotspots; 3D Viewer two-pass glow (outlines, shell on the active
  surface, fatter well tubes, plane outlines; driver without wide lines -> notice, no crash); File > Export section
  image (with / without glow), 3D screenshot asks too; > 2,000 objects -> active-only + status note.
- 40 Aesthetics: colour-table library (Kingdom-style, high contrast, perceptual viridis / cividis / magma-style built
  in OKLab, CVD-safe marked) + View > Colour bar editor (drag points, save to project, JSON / CSV); Live Map Sun...
  (sun dial hillshade, same sun lights 3D horizons); View > Co-render (alpha / multiply / transparency) also on 3D
  planes and map grids; View > Theme Dark / Light + Anti-aliasing (dark is the default at start-up); 3D Display...
  (EDL + fog in screenshots / software view, fog in OpenGL, semi-transparent faults with edges, fly-to speed);
  Live Map Zoom to selection (eased); Well tool View > Track style (GR gradient fill, lithology patterns, pay flags,
  N-D crossover, templates); File > Export section figure (title block, scale bar, colour bar, legend; 16:9 / Letter
  / A4 / poster); automatic distinct horizon / fault colours (+ ICS colours, lock, reassign).
- 41 Seismic Conditioning: Seismic > Conditioning - spike removal, Ormsby / Butterworth bandpass, FX decon, GST dips +
  planarity, structure-oriented filter (dip-steered median / alpha-trimmed / mean, planarity guard, Kuwahara edge
  protection, 1-3 passes, strength), footprint notches, presets (Interpretation / Light for inversion / Fault-
  attribute input / yours), preview original | conditioned | difference + RMS removed % + signal-leakage warning,
  protect zones, full-volume run in bricks (new volume + history, difference volume, SEG-Y, Cancel); the inversion
  window prefers Light volumes and warns on SOF volumes. Defaults set from your F3 inline-425 test (1 pass,
  protection on).
- 42 Survey blending: Interpret > Survey horizon blending - adaptive seam blend of one horizon from two overlapping
  surveys (width = |residual| / allowed slope, hybrid / residual / feather / cut, faults as barriers, well tops
  re-tied, gap flags, QC maps + readout, ZMAP+ / map-hub export).
- Files: new seismic_tool glow.py, colortables.py, aesthetics.py, conditioning.py, surveyblend.py, gui/glow.py,
  gui/theme.py, gui/sundial.py, gui/colorbar_editor.py, gui/conditioning_dialog.py, gui/surveyblend_dialog.py, tests
  test_glow / test_aesthetics / test_conditioning / test_surveyblend; well_correlation trackstyle.py +
  test_trackstyle.py; changed main_window, section_panel, live_map, view3d, scene3d, mapbus, maplayers,
  inversion_dialog, horizon_dock, fault_dock, gui/colormaps, __main__; WC section.py, lithology.py,
  gui/main_window.py, gui/section_view.py; docs PLAN / HANDOFF / WHATS_NEW / TUTORIAL (chapters 14b-14e, 9 new
  pictures), dev/build_tutorial.py + tutorial_content.py. Installer unchanged.
- Tests (here, Qt mock - no real Qt / GPU): Seismic selftest PASSED (1206 checks), Seismic guitest PASSED (447), Well
  Correlation selftest PASSED + guitest PASSED, Mistie selftest PASSED.
- Known limitations: nothing ran on real Qt / OpenGL here - the glow pass, themes, DPI and the new dialogs need your
  PC; EDL / shadows only in screenshots / the software view; conditioning SEG-Y uses standard GeoHarmonic headers,
  no anisotropic-diffusion engine, protect zones are IL / XL boxes; blending uses one velocity model for both
  surveys and is not yet a step of the depth-conversion dialog; tutorial pictures for these items use the demo cube /
  synthetic test cases (no F3 here). Follow-ups are listed per item in PLAN.md.
- Test this first (Windows):
  1. Double-click Update_GeoHarmonic.bat (no need to rerun Install GeoHarmonic.bat - the installer did not change;
     run it only if you want the tutorial rebuilt with real screenshots). Then Run all tests.bat -> ALL GOOD.
  2. Start the Seismic tool: it now opens in the dark theme; View > Theme > Light and back.
  3. F3 inline with horizons + Fault A: press G (glow off / on), Display > Glow settings > Halo; picking with P / F
     still works on glowing lines.
  4. View > 3D Viewer: glowing outlines; status bar must not say "Glow off: ..." (if it does, send Claude the text).
  5. Seismic > Conditioning on F3 inline 425 > Preview: compare the difference panel with your prototype.
  6. File > Export section figure > 16:9 slide > PDF.
- Next: no PLAN items left - add new ones to resume (candidates: the follow-ups listed in items 39-42).


## Run 9 - 2026-11-26 (scheduled, 20:34-21:15 UTC) - item 43 done; all PLAN items are done
- Item 43 PC retest fixes + F3 worked-example tutorial (added to PLAN this run from Chris's second PC install).
  Seismic v0.66.0, Well Correlation v0.14.1.
- Fixes (each with a regression test, `43a`-`43g` in test_results.txt):
  43a Survey blending window opens (attribute `show` renamed show_combo; also Glow settings `width`, Mistie / Well tie
  `window`); a scan of every Qt-derived class for attributes that hide Qt methods + a GUI test that opens, shows and
  closes 15 windows. 43b Wheeler last-sample IndexError: new `lininterp.lin_index` (float64, k <= n-2) used in Wheeler,
  RGT interp_rows, horizon RGT, section display, amplitude extraction, dip-steered semblance; tested float32 + float64.
  43c Framework 1 block -> 2: half-open segment crossing (`carto.seg_cut`) so a fault point exactly on a grid line cuts,
  strike-following trace extension, and horizons sorted top -> bottom (F3's came alphabetically - that also broke the
  stratigraphic rules: active cells 2,912 -> 8,736 on the F3-like test). 43d Live grid of a 3D horizon now built on the
  survey's own rotated bin grid (`livegrid.SurveyGrid`, holes filled per fault block); test: node (IL 130, XL 350) on
  its exact X/Y, grid inside the outline, saved / reloaded, contours + Export map. 43e Arbitrary line: open path,
  double-click / Enter / right-click ends it and opens the section, Esc cancels (the Map-layers panel no longer turns
  every line into a closed polygon layer). 43f all CSS colour names, bad colour named + style kept, legend escaped.
  43g filtered wells lose their labels (map, zoom / pan, export).
- Tutorial: Part 1 = F3 worked example W1-W14 (`dev/tutorial_f3.py`) that RUNS each step on F3 (working copy - Chris's
  project is not written) and quotes the numbers; Part 2 = tool chapters, RGT / 3D interpretation / prospects now have
  pictures; new "Fixes in this release" with before / after (Live grid picture). `seismic_tool/synth_f3.py` makes a
  synthetic F3-like raw folder and runs tools/setup_f3.py on it (CI). build_tutorial: --f3 / --f3-synthetic / --no-pdf,
  progress line per chapter, tutorial_build_log.txt, TUTORIAL.pdf (letter); installer shows the progress on screen.
- Files: seismic_tool lininterp.py, synth_f3.py, test_retest.py (new); livegrid, livemap, maplayers, framework,
  horizons, maps, attributes, rgt/stack, rgt/solve, basin/chrono, selftest, test_pcfixes, test_mistie, test_tie,
  gui/basemap, live_map, framework_dialog, surveyblend_dialog, glow, mistie_dialog, welltie_dialog, section_panel;
  well_correlation carto.py; dev/tutorial_f3.py (new), build_tutorial.py, tutorial_content.py; Install
  GeoHarmonic.bat (tutorial step); docs PLAN / HANDOFF / WHATS_NEW / TUTORIAL.md / .html; .gitignore (PDF, build log).
- Tests (here: numpy 2.5, Qt mock - no real Qt / GPU): Seismic selftest PASSED (1,235 checks), Seismic guitest PASSED
  (466), Well Correlation selftest PASSED (198) + guitest PASSED (30), Mistie selftest PASSED. Tutorial builder F3 mode
  on the synthetic F3-like project: 14/14 chapters ran, every one with >= 1 picture (45 s).
- Known limitations: no real Qt or real F3 here - the real-Qt halves of 43a / 43e and the F3 run of the tutorial happen
  on your PC; the 6 km Live-grid shift was not reproduced on the F3-like survey (old code placed it correctly here) -
  the grid is now built on the survey bins so it cannot leave the outline, but tell Claude if it ever does; the
  tutorial's RGT step uses horizon-guided RGT on inline 425 (full-volume RGT is 10-20 min) and geobodies on a box;
  volumes in W13 / W14 use illustrative contacts and parameters, not F3 facts.
- Test this first (Windows):
  1. Update_GeoHarmonic.bat, then Install GeoHarmonic.bat -> ALL GOOD; the tutorial step prints one line per F3 chapter.
  2. tools\Seismic Interpretation Tool\test_results.txt: every `43a`-`43g` line PASS.
  3. Interpret > Survey horizon blending opens.
  4. F3: View > Live Map > Live grid... > Horizon: FS6 sits inside the outline; Arbitrary line (3 clicks + double-click)
     opens the section; rule `X <= 615000 = lime` works; an X filter hides the labels too.
  5. Interpret > Structural framework + 3D grid > 1. Framework (time): 2 fault blocks with Fault A.
  6. Desktop shortcut GeoHarmonic - Tutorial: the header says "your F3 Netherlands project" and Part 1 shows F3 numbers
     (tutorial_local\tutorial_build_log.txt lists any chapter that failed).
- Next: no PLAN items left - add new ones to resume.
