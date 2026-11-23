# What's new (for Chris) - Geo Tool Kit

Built unattended 23 Nov 2026 on branch `dev`. Everything below passed the numpy self-tests and the GUI-logic
tests under the Qt mock, but **no real Qt window has been opened yet** - the GUI is the riskiest part.

## Start here: test checklist (highest risk first)
1. Pull `dev`, run `Install (run once).bat` in each tool folder only if the .venv is missing, then
   `Run self-test` / `python -m seismic_tool --selftest` and `python -m seismic_tool --guitest`
   (the guitest now runs against real PySide6 offscreen - first real-Qt check of all new code).
2. Seismic tool, open `NPRA 2D.seisproj.json`: Horizons panel > New, click a strong reflector on R-2
   (left part, ~1200 ms), press T. Check the pick follows the event, Shift+click erases, Ctrl+Z undoes,
   left-drag draws a line. Mouse wheel zoom must still work in pick mode; press P to leave pick mode.
3. Faults: N, click 3-4 points down a fault on R-2, Enter. Draw a second stick on R-3. Check the sticks and
   the thin cut line display, Esc / Ctrl+Z behaviour, Shift+click delete.
4. RGT on NPRA: Interpret > Compute RGT (all 2D lines) with node spacing 8 (about 2 min). Display box "RGT",
   then Interpret > Build horizon stack (2D), tick "RGT contours": contours should follow reflectors.
   Domain box "Wheeler": reflectors flat. Interpret > RGT edit mode (E): click a contour, click a nearby
   reflector, Ctrl+Z.
5. Well Correlation Tool: open the NZ project - Tui-1 is new (from DLIS). Check its GR/RHOB/NPHI/DT tracks
   (2379-2561 m and 3593-3729 m) and that the other wells look unchanged. Try
   `python -m well_correlation --dlis2las <the 171 MB Tui-1 main DLIS> <folder>`.
6. Maui 3D (when imported): link the NZ well project; wells + tops on inlines; Interpret > Well tie for
   Maui-5/6 (time-depth = sonic anchored to Tui-1's checkshot), Auto tie, Save; Interpret > Horizon maps.
7. Analytics: Interpret > Analytics, add 2-3 basemap grids + a CSV target, Train (GBM, leave one well out),
   then **Model QC...** - read the summary (baselines / leakage / Moran) and open the HTML. Try
   **Import engineering / production CSV...** on `python dev/make_synthetic_engineering.py C:\temp\eng`
   (completions.csv first, then production.csv) - the mapping form (combo per column) is new real-Qt code.
8. Dashboard (new real-Qt window): on the NZ project linked to the Maui 3D, Interpret > Dashboard - deviated
   Maui wells should be drawn from their surveys; Colour by TVDSS, then a basemap grid; drag the histogram range;
   click a well on the map / a table row and check the highlight follows in all panels. Select a deviated
   well, Well strip: the seismic panel's yellow time path should sit on the reflector the well follows.
9. Horizon maps on the NPRA horizon (survey "2D"), export CPS-3 and load it in Kingdom/Petrel to confirm
   orientation (top row first).

## Known limits (all items)
- Long jobs (tracking a big 3D, RGT, stacks) run with a progress bar but cancel only by closing the app;
  autotracking of a whole 3D runs in the GUI thread (wait cursor).
- RGT between node columns is interpolated (fine detail at the node spacing); thin layers below tuning do not
  pinch to zero; 2D RGT edits do not propagate through ties.
- Wells drawn vertical at the surface location; tie = one shift + one stretch.
- The Maui checkshot LAS files hold no time-depth pairs (mislabelled export).

## Change log (newest first)

## Seismic v0.16.1 - well strip / geosteering view
- Dashboard > Well strip: log vs MD, TVDSS vs MD with zone top/base, seismic along the wellbore + time path.
- Log curves as lateral colour sources.

## Seismic v0.16.0 - Dashboard with laterals (PLAN item 14, first parts)
- Interpret > Dashboard: map / histogram / crossplot / table docks with linked selection and filters; wells from
  the project with deviation surveys; laterals coloured along MD by TVDSS, grids, seismic amplitude, stage CSV
  values or in-zone; per-lateral stats (mean/P10/P50/P90, % in zone) -> Analytics; layouts; HTML export.
- Drawing is batched: one curve per colour bin for all wells plus level-of-detail decimation (1,000 synthetic
  laterals -> 16 curves in 0.1 s).

## Seismic v0.15.1 - engineering / production import dialog
- Analytics window: **Import engineering / production CSV...** (presets Enverus, IHS, Texas RRC, FracFocus,
  NDIC, ECMC, NM OCD, OCC; editable column mapping + units; save/load mapping). Works without a well project
  (wells come from the CSV, joined by API).

## Seismic v0.15.0 - Analytics 13d: model QC toolkit
- `analytics/qc.py`: baselines (mean, IDW of the target, best single input), y-randomisation, random-noise
  feature, residual QQ / variogram / Moran's I, split-conformal and quantile-GBM intervals with coverage,
  classifier calibration (Brier), applicability domain (Mahalanobis + kNN), importance stability,
  hyperparameter sensitivity, PDP direction checks, paired fold comparison, blind / temporal / repeated designs.
- Analytics window: **Model QC...** button (one-page HTML); Predict map now carries an applicability mask.
- Tests: Moran's I z 24.9 -> 4.1 once x/y is added; conformal 80% interval covers 84% of new wells;
  y-randomisation collapses R2; noise column ranks last.

## Seismic v0.14.0 - Analytics 13a-13c (PLAN item 13, first parts)
- analytics package: attribute table + joins (API/name), data prep (imputation, outliers, correlation, VIF,
  PCA), engineering/production features (presets, units, cums, IP, GOR, Arps EUR, parent/child, FracFocus, lease
  allocation), models (linear/ridge/lasso, kNN, tree, RF, XGBoost-style GBM with NaN handling + monotone
  constraints + early stopping, MLP, k-means, GMM, SOM), honest validation (LOO well, group, spatial blocks,
  nested CV, learning curves), explainability (exact TreeSHAP, KernelSHAP, permutation importance, PDP/ICE,
  HTML report), Analytics window (Interpret > Analytics), optional xgboost/sklearn/shap installer + --check-ml.
- Tests: GBM R2 0.97 vs linear 0.29 on a nonlinear benchmark; random CV 0.86 vs group 0.36 on clustered wells;
  TreeSHAP equals brute-force Shapley (2.8e-16) and sums to the prediction; KernelSHAP exact for linear; Arps
  EUR within 0.0% on the synthetic production set.
- Still to do (next run): 13d QC report (baselines, y-randomisation, residual variogram, conformal intervals,
  applicability domain, stability), engineering CSV import dialog, dashboard (item 14).

## Seismic v0.13.0 - Velocity model + depth conversion (PLAN item 12)
- Interpret > Velocity model: layer cake from horizons + well tops (or time-depth), gridded interval
  velocities, depth maps, well misfit table + CSV, optional residual correction; Depth domain on sections.
- Tests (3 synthetic wells, laterally varying layer-1 velocity): layer-2 velocity exact, well misfit 0.05 m,
  H2 depth everywhere within 1.3% (median), residual correction 0.00 m at wells, depth section reflector within
  4 m of the model depth.
- Known limits: inverse-distance velocity gridding (no kriging / trends); depth display converts the shown
  section only (no depth volume yet); below the deepest horizon the last layer velocity continues.

## Seismic v0.12.0 - Seismic attributes (PLAN item 11)
- On-the-fly attributes in the display box (complex trace, RMS, coherence, dip-steered coherence, spectral
  bands) for any section; Interpret > Compute attribute volume (new project volume); horizon curvature/dip maps.
- Tests: envelope/phase exact, inst. frequency 25.00 Hz on a 25 Hz tone, spectral decomposition separates 20/50 Hz,
  dip-steered coherence 0.99 vs 0.30 unsteered on 6 ms/trace dips, coherence 0.37 at a fault vs 0.98 away,
  block-computed volumes equal on-the-fly sections, paraboloid curvature exact (0.500 1/km).
- Known limits: attribute volumes double the storage (two float16 copies); time-slice coherence reads a whole
  time slab of the dip field (slow on very large surveys).

## Seismic v0.10.0 - 2D-3D / 2D-2D mistie analysis (PLAN item 10, first part)
- Interpret > Mistie analysis: simultaneous shift + phase + gain per tie, network solve with 3D fixed,
  robust weights, loop closures, apply/reset corrections on the fly (display, tracking, RGT, picks).
- Tests: single tie 8.9 ms / 33.7 deg / x2.00 (true 9 / 35 / 2); 2D cut from 3D with known corruption
  8.01 ms / 30.2 deg / x2.00; network of 3 lines + 3D within 0.7 ms and 3 deg with a deliberately bad tie
  (robust solve isolates it; non-robust is pulled 4 ms off); corrected line vs 3D corr 0.47 -> 0.99.
- v0.10.1: Flex mode (along-line shift with smoothness, knots every N traces): a 0-12 ms ramp is recovered
  within 0.14 ms; very smooth = constant per line; per-trace corrections applied to display and picks;
  warning when a line needs > 10 ms local flex.
- v0.11.0: horizon-based misties, survey groups + key lines, time-variant shift (smooth in TWT; ramp 4-12 ms
  recovered within 0.6 ms), overlapping vintages tied along the overlap (5 ms offset found as 5.05),
  duplicate hiding, residual map, tie table CSV, corrected SEG-Y export (round trip exact).
- Known limits: time-variant only for lines crossing a 3D; the Mistie Tool (tkinter) is linked via the
  CSV tie table, not live; no 2D-2D time-variant ties yet.

## Seismic v0.9.0 - Horizon maps (PLAN item 8)
- Interpret > Horizon maps (Ctrl+M): time structure, isochron, amplitude (RMS/max/min/mean/at horizon),
  depth with well average velocities (optionally calibrated to a formation top); basemap display with
  contours; export XYZ, CPS-3, PNG. 2D horizons are gridded.
- Tests: isochron exact, RMS/max/at values equal direct computation, contour of a plane on the expected line,
  CPS-3 and XYZ round trips, 2D gridding honours the lines, depth conversion 1475.2 m vs 1475.0 m.
- Known limits: velocity between wells by inverse distance (no kriging / velocity cube); CPS-3 row order is
  top row first (Petrel convention) - check orientation on first import into Kingdom.

## Seismic v0.8.0 - Wells on seismic + well ties (PLAN item 7)
- Wells + tops (TWT) on sections and time slices from the linked well project; time-depth from checkshot
  DLIS (Tui-1, Pateke-2), else sonic anchored to the nearest checkshot.
- Interpret > Well tie: synthetic (Ricker / statistical wavelet) vs seismic, shift/stretch, auto tie, save.
- Tests: synthetic survey delayed 12 ms -> auto tie finds 12.1 ms (corr -0.55 -> 0.99); statistical
  wavelet 12.1 ms; Tui-1 3875 m TVDSS -> 2806 ms (checkshot); Maui-5 tops 1864-2110 ms via sonic + Tui-1.
- Known limits: wells treated as vertical at the surface location on sections (deviated paths in item 14);
  stretch is one linear factor (no multi-point squeeze yet); tie uses the nearest trace only.

## Well Correlation Tool v0.5.0 - DLIS reader (PLAN item 6b)
- DLIS files in `wells\<Well>\dlis\` load like LAS; Tui-1 added to the NZ project from its two DLIS
  files (header from wells_header.csv; PR3099 tops were already there).
- `--dlis2las IN OUT_DIR [--arrays]`, `--dlisinfo`, `--checkshot IN.dlis OUT.csv` (Tui-1 + Pateke-2 CSAT).
- Convert the 171 MB Tui-1 main-pass DLIS on your PC: either copy it into `data\nz\wells\Tui-1\dlis\`
  (loads directly, File > Reload logs) or run --dlis2las into `wells\Tui-1\las\`.
- Tests: synthetic DLIS written byte by byte (segments, padding, absent attributes, arrays, IBM floats)
  round-trips exactly; real Tui-1 files parse in 0.5 s with sane GR/RHOZ/TNPH/DTCO ranges.
- Known limits: encrypted records skipped; checkshot vertical times assume a straight ray from a source at
  the SRD (offset 65.7 m at Tui-1, negligible below 1 km).

## v0.7.0 - RGT editing (PLAN item 6)
- RGT contours on sections; RGT edit mode (E): click a level, click where it belongs -> local re-solve
  (~0.15 s on the synthetic), stack updated, Ctrl+Z; active horizon as RGT constraint; re-link RGT around a
  fault. Edits saved at once.
- Tests: a moved level lands exactly where clicked (12.0 ms move), uint16 RGT outside the edited inlines /
  window is bit-identical, undo restores RGT + stack exactly; fault re-link fixes a model computed without
  the fault (p95 error 29 -> 0.6 ms); horizon constraint makes one iso-surface follow the picks; 2D edit
  local to the line window.
- Known limits: 2D edits do not propagate through ties (tie columns stay fixed); edits keep the RGT
  range of the original model (values beyond +/-5% head room are clipped).

## v0.6.0 - Horizon stack, Wheeler view, stratal slices (PLAN item 5)
- Interpret > Build horizon stack (3D / 2D); mode "Stratal slice" with slider through the stack; domain box
  Time / Wheeler (flattened RGT view with hiatus gaps); Interpret > Convert stack level to horizon.
- Tests: stack surfaces equal direct iso-surfaces and never cross; stratal slice = volume interpolated on the
  surface (err 1e-4); Wheeler flattens the synthetic (98% of variance between levels) and shows a true hiatus
  as a gap; NPRA R-2 Wheeler view is flat.
- Known limits: thin layers below seismic tuning do not reach zero thickness in the RGT, so a pinch-out
  shows as converging levels, not as a gap; stratal slices show amplitude only (not dip/attributes yet).

## v0.5.0 - RGT model, step B: mesh + global solve (PLAN item 4)
- Interpret > Compute RGT (current 3D volume) / (all 2D lines): settings dialog (node spacing, interval,
  correlation window, min correlation, links across faults, max throw, horizons/faults as constraints),
  runs in the background; display box "RGT" shows the result on any section.
- Dip-guided node links + correlation, cross-fault ties away from the fault, horizon constraints, 2D line
  ties, global least-squares solve (CGLS in numpy) with a coarse-to-fine start.
- Synthetic tests (fold + pinch-out + fault): iso-RGT surfaces match 60 true horizons, median error
  0.15 ms (4 ms sampling); across the fault p95 < 3.5 ms above the wedge; 2D tied lines median 0.10 ms.
- Real data: NPRA R-2 + R-3 (12,000 traces, 1.1 M unknowns) in ~100 s at 8 traces / 8 ms nodes; iso-RGT
  contours follow the reflectors through the folds (see test checklist).
- Known limits: between node columns the RGT is interpolated bilinearly (fine detail at node spacing);
  NPRA R-2 / R-3 don't cross, so 2D ties were only tested on synthetics; a big 3D (millions of nodes) needs
  minutes per 100 iterations - multi-threading/bricking in item 15. Stored as uint16, not float16 (float16
  has only ~2-4 ms resolution at 2-6 s).

## v0.4.0 - Dip field, RGT step A (PLAN item 3)
- Interpret > Compute dip field: inline/crossline dip (ms/trace) + planarity volumes for the 3D volume
  (background job with progress). Toolbar display box: Seismic / Inline dip / Crossline dip / Planarity,
  also on 2D lines (computed on the fly, cached).
- Synthetic: dips recovered within 0.6% (spec 5%). NPRA R-2: fold flanks clearly shown in the dip.
- Known limits: ~3 us per sample, so a 1-billion-sample 3D takes ~45-60 min (multi-threading in item 15);
  cancel works only via closing the app for now.

## v0.3.0 - Fault sticks + fault surfaces (PLAN item 2)
- Faults panel + Interpret menu: N = new fault, F = fault mode; click points, Enter ends a stick, Esc
  cancels, Shift+click deletes a stick, Ctrl+Z undoes (shared with horizon edits, most recent first).
- Sticks are joined into a triangulated fault surface; its cut is drawn on every inline, crossline, time
  slice and 2D line. Fault traces on the basemap.
- The horizon autotracker stops at visible faults (tick box in the Horizons panel). On a synthetic with
  throw = reflector spacing, tracking without the fault mis-ties 100% of the hanging wall; with the fault 0%.
- Known limits: sticks always go to the active fault (no "re-assign stick" yet); faults with a single
  stick have no surface (stick only shown on its own section, no tracker barrier).

## v0.2.0 - Horizon picking + autotracking (PLAN item 1)
- Horizons panel (right dock) + Interpret menu: New / Rename / Colour / Delete, visibility tick boxes,
  pick count per horizon.
- Pick mode (P): click = seed snapped to Peak / Trough / Zero +to- / Zero -to+ / None; drag = manual line;
  Shift+click / Shift+drag = erase. T = autotrack, Ctrl+Z = undo (30 steps).
- Autotracker: waveform cross-correlation (window +/-20 ms, max jump 10 ms, min corr 0.6 by default), grows
  the most reliable picks first (thresholds 0.95 -> min corr), sub-sample snapping, bridges up to 3
  dead/noisy traces, optional lateral mix for noisy data (2D default 5 traces). 3D tracks across inlines
  and crosslines (~90k traces in 0.5 s on a synthetic), or "This line only". 2D follows line ties onto
  crossing lines automatically.
- Display: horizons on inlines/crosslines/2D lines, dots on time slices, time-structure map of the active
  horizon on the basemap.
- Saved with the project (`<project>_seismic\horizons`).
- Known limits: tracking runs in the GUI thread (wait cursor) - a full 20 GB survey could take a minute;
  2D horizons are not yet tied to 3D horizons; no fault awareness yet (item 2).
