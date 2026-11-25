# Build plan - Seismic Interpretation Tool (Nov 23 - Dec 2, 2026, two runs a day, one item per run)

Status values: `todo`, `partial`, `done`, `blocked: <reason>`. Do the first `todo`/`partial` item.
Chris's priorities: both 2D and 3D everywhere; PaleoScan-style relative geologic time (RGT) workflow;
Kingdom-like interaction; large data.
Phase 2 (items 20-30, added 2026-11-24): petrophysics + integrated map hub first, then exploration risk, then
reservoir engineering / M&A. Several items per run allowed (CLAUDE.md rule 1).

1. **Horizon picking + autotracking** - status: done (run 1, 2026-11-23; v0.2.0)
   - `horizons.py`: Horizon object per survey (3D: float32 grid (nil, nxl) of TWT ms, NaN = unpicked;
     2D: per-line arrays over CDP), save/load as .npy + entry in project JSON; colour, name.
   - Snap to peak / trough / zero crossing (+/-, - to +) with parabolic sub-sample interpolation.
   - Autotrack: seed points -> flood-fill / priority-queue tracker using cross-correlation window
     (default +/-20 ms), max jump, min correlation; 3D tracks across il/xl; 2D along line and across ties
     where lines intersect (use line geometry); also "track along this line only".
   - GUI: Horizons dock (list, new, colour, delete), pick mode on section (click = seed, drag = manual
     line, Shift = erase), keyboard T = track from seeds, Ctrl+Z undo. Horizons drawn on inline/xline/2D
     sections as lines and on time slices as intersection points; time-structure map in the basemap
     (colour image of the grid).
   - Tests: synthetic dipping/folded reflector -> autotrack recovers it < 1 sample error; peak snap.
2. **Fault sticks + fault planes** - status: done (run 1, 2026-11-23; v0.3.0)
   - Draw sticks on sections (click points, Enter ends stick, N new fault), assign to named faults,
     display on intersecting sections; triangulated fault surface per fault; horizon autotracker stops at
     faults. Tests: synthetic faulted volume - tracker does not jump across the fault.
3. **RGT step A - dip field** - status: done (run 1, 2026-11-23; v0.4.0)
   - `rgt/dip.py`: local inline & crossline dip (ms/trace) from gradient structure tensor with Gaussian
     smoothing (separable numpy convolution), plus coherence/planarity; chunked over the memmap; store as
     float16 volumes. Tests: synthetic dips recovered within 5 %.
4. **RGT step B - mesh + global solve** - status: done (run 1, 2026-11-23; v0.5.0; RGT stored as uint16 not float16 for precision)
   - Seed grid of nodes (every N il/xl, every M ms), link neighbours by dip-guided waveform correlation
     (cross-correlation search, confidence weights), fault-aware (no links across fault planes, or links
     allowed with offset); solve relative geologic time for all nodes by sparse weighted least squares
     (write a conjugate-gradient solver in numpy; no scipy) with optional hard constraints from
     interpreted horizons; interpolate RGT to the full volume (store float16 `rgt.npy`).
   - 2D: same on each line, tie lines at intersections. Tests: synthetic with onlap/pinch-out + fault ->
     extracted iso-RGT surfaces match true horizons.
5. **RGT step C - horizon stack, Wheeler, stratal slices** - status: done (run 1, 2026-11-23; v0.6.0)
   - Extract N horizons (iso-RGT) as a horizon stack; Wheeler view (seismic resampled to RGT axis,
     hiatuses shown as gaps); stratal-slice viewer (amplitude on any iso-RGT surface, slider through the
     stack) in the section panel as a new mode; convert any stack surface to an editable horizon.
6. **RGT step D - interactive editing** - status: done (run 1, 2026-11-23; v0.7.0)
   - User adds/moves control points or horizons as constraints, local re-solve in a window around the edit
     (fast), fault re-linking; undo. Tests: constraint changes the model only locally.
6b. **DLIS reader (numpy, RP66 v1)** - status: done (run 1, 2026-11-23; Well Correlation v0.5.0)
   - `tools/Well Correlation Tool/well_correlation/dlis.py`: parse storage unit label, visible records,
     logical record segments (encryption/padding/trailer flags), EFLR sets (FILE-HEADER, ORIGIN, CHANNEL,
     FRAME, PARAMETER, TOOL) with RP66 representation codes, IFLR frame data incl. multi-sample
     (array) channels; return channels per frame with units, index (depth or time), origin metadata.
     Export to LAS 2.0 (`--dlis2las IN.dlis OUT_DIR`, one LAS per frame, skip array channels > 1 sample
     unless asked). Wire into project loading so wells/<Well>/dlis/*.dlis load like LAS.
   - Test data: data/nz/wells/Tui-1/dlis (PEX/DSI/HALS repeat 084PUP, intermediate 045PUP),
     Tui-1 + Pateke-2 checkshot DLIS (CSAT: VSP/checkshot - extract depth vs one-way/two-way time table).
     Compare curves against the printed values where possible; sanity ranges (GR 0-200, RHOB 1.5-3.1).
   - Add Tui-1 to the NZ well project (header from wells_header.csv; it already has PR3099 tops) using the
     converted logs; the large main-pass DLIS (171 MB, not in repo) is converted by Chris on his PC with
     `--dlis2las`. Document that in WHATS_NEW.
   - Checkshot tables (Tui-1, Pateke-2, Maui-5, Maui-6) become the time-depth inputs for item 7.
7. **Wells on seismic** - status: done (run 1, 2026-11-23; v0.8.0)
   - Load well project wells; time-depth from checkshots (Maui-5, Maui-6 LAS in data/nz) or interval
     velocity from DT, nearest-checkshot fallback; show wells + tops (in TWT) on sections and basemap;
     synthetic seismogram (RHOB*DT AI, reflectivity, Ricker wavelet with chosen frequency, or statistical
     wavelet from seismic near the well), bulk shift / stretch-squeeze tie with correlation coefficient.
8. **Horizon maps + attributes** - status: done (run 1, 2026-11-23; v0.9.0)
   - Time-structure, isochron, amplitude extraction (RMS / max in window), depth conversion with a
     velocity from well ties, contours (reuse marching squares from well_correlation/mapping.py), export
     grids (XYZ, CPS-3 ASCII), PNG.
9. **Write-up** - status: done (run 1, 2026-11-23; docs/WHATS_NEW.md checklist ordered by risk; keep it updated)
   - `docs/WHATS_NEW.md` for Chris: features, how to use, test checklist ordered by risk, known limits.

10. **2D-3D merge / tie** - status: done (run 1, 2026-11-23; v0.11.0 - Mistie Tool linked via CSV tie table; time-variant only for lines crossing a 3D)
   - Match 2D lines to a 3D volume: extract the 3D along each 2D line (arbitrary line), estimate per-line
     bulk time shift, phase rotation and amplitude scalar by cross-correlation in a window; apply and show
     before/after; integrate with the Mistie Tool network solve (2D-2D + 2D-3D ties). Tests: synthetic 2D
     made from the 3D with a known shift/phase/gain - recovered within tolerance.
   - Match industry mistie features: simultaneous time + phase + amplitude estimation at each tie
     (Bishop & Nunns 1994 style, envelope cross-correlation for the time part), network least squares with
     reference lines held fixed (0 shift, 0 deg, gain 1), per-survey "key line" option for multi-vintage
     2D, horizon-based misties (from picked horizons at ties) as an alternative to waveform, constant or
     time-variant (windowed) corrections, max-shift and correlation-window limits, outlier tie rejection,
     **Flex mode** (after the constant solve): per-line smooth correction functions instead of one constant -
     along-line (shift as a spline of CDP) and optionally time-variant (shift as a smooth function of TWT,
     from windowed/dynamic-warping correlation at several times per tie) - solved as one regularised least-
     squares system: tie residuals + smoothness penalty (slider: 0 = exact closure, high = constant per
     line) + anchors (3D, well ties, reference lines) + max-gradient cap. QC: flex map / profile per line,
     added structural curvature vs original, before/after loop closure, warning when a tie needs a large
     local flex (likely cycle skip or navigation error rather than a real mistie). Apply to seismic display,
     horizons only, or both; export corrected SEG-Y optional.
     Residual-mistie map/table before vs after, redundancy handling: ties weighted by correlation quality,
     robust re-weighting (IRLS/Huber) to down-weight bad ties, loop-closure report, minimum-norm or
     reference-line datum (the network only fixes relative shifts), phase solved as unit vectors so 350/10 deg
     don't average to 180, collinear/overlapping lines (reprocessed vintages, overlapping segments) tied
     along the whole overlap with a "preferred version" flag and duplicate hiding, and corrections applied on the fly to BOTH seismic display
     and interpretation (horizons) without rewriting SEG-Y (optional export of corrected SEG-Y).
11. **Seismic attributes** - status: done (run 1, 2026-11-23; v0.12.0)
   - Volume attributes computed in chunks to new float16 stores, viewable like any volume: envelope,
     instantaneous phase/frequency (Hilbert via numpy FFT), RMS amplitude, coherence/semblance, dip-steered
     variance, curvature on horizons, spectral decomposition (short-window FFT bands). Attribute picker in
     the section panel.
12. **Velocity model + depth conversion** - status: done (run 1, 2026-11-23; v0.13.0)
   - Interval/average velocity from well time-depth pairs + horizons (layer-cake), gridded; convert
     horizons and sections to depth; well misfit table.
13. **Analytics (Transform-style + modern data-science workflow)** - status: done (run 1, 2026-11-23; v0.15.1 - analytics/table, prep, engineering, synthetic, models, validation, explain, check_ml, qc, gui/analytics_window with Model QC, gui/engineering_dialog. Optional follow-ups: QC plots (QQ, variogram, reliability diagram) in the QC report; basemap hatching of cells outside the applicability domain (`g.inside` is computed))
   Split into 13a-13c; do them in order, checkpoint after each.
   13a. Data + attribute table: `seismic_tool/analytics/` (shared, importable by the Well Correlation Tool
     too). Table built from horizon/zone extractions (item 11 attributes, isochrons, structure, curvature)
     sampled at wells and on the map grid, joined with well properties (zone stats: net sand, NtG, avg
     PHID/Vsh) and any per-well target CSV (production / EUR later). Prep: missing-value report +
     imputation (median / kNN), scaling, outlier flags, correlation matrix, multicollinearity (VIF),
     crossplot matrix with regression lines, PCA.
   13a+. Engineering + production data (`analytics/engineering.py`), joined to wells by API/UWI
     (normalise API-10/12/14, strip dashes) or name:
     - Import CSV/Excel-exported CSV with a column-mapping dialog and saved mapping presets (Enverus/IHS
       exports, Texas RRC, FracFocus, NDIC, Colorado ECMC, NM OCD, OCC layouts). Units detected/converted
       (lb vs t, bbl vs m3, ft vs m).
     - Completion table: completion date, lateral length (or computed from survey), stage count, stage
       spacing, cluster spacing, total proppant + proppant/ft, total fluid + bbl/ft, fluid system,
       landing zone (from item 14 % in zone), perforated interval, well spacing to nearest neighbours
       (computed from laterals, same zone, parent/child flag by first-production date).
     - Monthly production (oil, gas, water, days on): derived features cum 3/6/9/12/24-month and total
       CUM per stream and BOE (6:1 configurable), peak month, IP30/IP90-equivalents, per-1,000-ft lateral
       normalised versions, GOR, water cut, months on production; month 1 = first month with production
       (option: skip partial first month). Optional Arps decline fit (qi, Di, b) + EUR with fit quality.
     - FracFocus helper: total water volume and proppant mass estimated from ingredient mass % (flagged as
       estimate). Texas RRC caveat: oil is reported by lease - allocation by well (equal / by test / by
       days) with a clear "allocated" flag.
     - All of it lands in the analytics table (inputs or targets), is filterable on the dashboard, and can
       colour laterals/bubbles (e.g. 12-month cum per ft).
     - QC specific to engineering data: completion design changes over time confound geology (vintage
       trend plot, time-split validation from 13d, partial-dependence with date as a control), parent/
       child effects, outlier and unit-error flags (e.g. proppant/ft > 5,000 lb/ft), wells with < N months
       excluded from N-month targets.
     - Test data: synthetic engineering + monthly production set (dev/make_synthetic_engineering.py,
       generated at test time) with known geology + completion effects that the models must recover.
   13b. Models + validation (numpy implementations, sklearn-like API: fit / predict / get_params):
     linear, ridge, lasso (coordinate descent); k-nearest neighbours; random forest; **gradient-boosted
     trees in the XGBoost style** (second-order gradients, histogram split finding, L1/L2 regularisation,
     shrinkage, row/column subsampling, max_depth/min_child_weight/gamma, early stopping); small MLP.
     Validation that is honest for wells: group k-fold by well and spatial/blocked CV (no leakage between
     nearby wells), leave-one-well-out, nested CV for hyperparameter search (grid/random), metrics R2,
     RMSE, MAE, classification accuracy/F1/confusion matrix, learning curves, prediction-vs-actual plots.
     Clustering: k-means, Gaussian mixture, SOM; elbow/silhouette.
     Optional backends: if `xgboost`, `sklearn` or `shap` import successfully (Chris's PC, see 13c), the
     same API uses them; numpy versions are the default and what the tests cover. Never require them.
   13c. Explainability: gain/split importance, permutation importance (with CV), **SHAP values - exact
     TreeSHAP for the tree models** (Lundberg et al. polynomial-time algorithm) and KernelSHAP for any
     model; plots: SHAP summary (beeswarm), bar, dependence (with interaction colouring), waterfall per
     well, partial dependence + ICE curves. Predicted-property map and uncertainty map (CV ensemble
     spread), saved models (JSON/npz) and a run report (inputs, CV scores, importances) exported as HTML.
     GUI: Analytics window (pick inputs/target, explore, train/validate, explain, push predictions back as
     maps/volumes) with a clear warning when wells are too few for the number of inputs.
     Add `Install analytics extras (optional).bat` that pip installs xgboost, scikit-learn and shap into
     the tool's .venv, and a `--check-ml` command that compares numpy vs library results on a fixed dataset.
   13d. Model QC toolkit (a "QC report" tab run on any trained model):
     - Validation designs: true blind wells (never seen in any tuning), spatial-buffer CV (drop training
       wells within R of the test well), temporal holdout (train on older wells, test on newer - needs a
       date column), repeated/bootstrapped CV with score spread.
     - Baselines every model must beat: mean, IDW/kriging of the target alone (does seismic add value over
       spatial interpolation?), best single attribute regression. Y-randomization (shuffled-target) test
       and an added random-noise feature (real inputs must out-rank noise) to expose overfitting/leakage.
     - Residual diagnostics: residual vs predicted, vs each input, QQ plot, residual map and residual
       variogram / Moran's I (spatial structure left in residuals = missing input).
     - Uncertainty: quantile GBM and split-conformal prediction intervals with measured coverage
       (e.g. 80% interval should contain ~80% of blind wells); classifier calibration (reliability
       diagram, Brier score).
     - Applicability domain: flag/mask map cells whose inputs are outside the training data
       (Mahalanobis / nearest-neighbour distance in input space) - no confident predictions there.
     - Stability: importance and SHAP rankings across folds/bootstraps, grouped permutation importance for
       correlated inputs, hyperparameter sensitivity, learning curve (would more wells help?).
     - Geologic sanity: optional monotonic constraints in the GBM, partial-dependence direction checks
       against expected physics, predicted vs observed at every well on the map.
     - Model comparison: paired CV-fold comparison with a significance estimate; one-page HTML QC report.
   Tests: synthetic data with known linear + nonlinear + interaction effects -> models recover them;
   TreeSHAP values sum to prediction minus expected value (local accuracy) and match brute-force Shapley
   on small trees exactly; KernelSHAP close to exact; GBM beats linear on the nonlinear case; group CV
   gives a worse (honest) score than random CV on spatially correlated synthetic wells.
14. **Dashboard with lateral (horizontal-well) attribute maps** - status: done (run 1, v0.16.2: 14a core done -
   `laterals.py` (min-curvature trajectories, synthetic laterals, survey/stage CSV, attributes along MD from
   grids / 3D volume / logs / stages / zones, per-lateral stats table, colour bins batched over many wells + LOD)
   and `dashboard_model.py` (filters, linked selection, histogram/crossplot/table data, layouts, HTML export).
   14b done: `gui/dashboard_window.py` (Interpret > Dashboard; docks map / histogram / crossplot / table, linked
   selection, histogram range filter, wells from the linked project with surveys, colour by TVDSS / map grid /
   seismic via well time-depth / stage CSV / in-zone, lateral stats, send to Analytics, layouts, HTML export).
   14c done: well strip / geosteering view (`gui/well_strip.py`: log vs MD, TVDSS vs MD with zone top/base,
   seismic along the wellbore with the well's time path) and log curves as colour sources.
   14d: background grid under the dashboard map, text filters (Filter by text...).
   Possible follow-ups: bubbles/pies per well; predicted-property maps need "Add basemap grid" after Predict map)
   - Dashboard window (dockable, linked panels, Spotfire-style): map, histogram, crossplot, data table,
     well strip view; selecting wells/points in one panel highlights them in all others; filters
     (formation, operator, date, attribute ranges); save/restore layouts; export PNG + standalone HTML.
   - Map: background grid (structure, isochron, attribute or predicted-property map) with contours, fault
     polygons, 2D lines / 3D outline; wells as surface + bottom-hole symbols with **full trajectories**
     (min-curvature from deviation surveys, geo.py) drawn as polylines **coloured segment-by-segment by an
     attribute along MD** with a colour bar and range control. Attribute sources along the wellbore:
     any log curve; seismic attribute/amplitude sampled along the trajectory in the 3D volume (via the
     well's time-depth); predicted property from the analytics models; formation/zone the bore is in
     (landing-zone % in target); stage/completion CSV (stage intervals with values, e.g. proppant per ft).
     Optional bubbles/pies per well (e.g. production or EUR from a CSV).
   - Per-lateral statistics (mean/P10/P90 of the colouring attribute, % in zone, lateral length) written
     to the analytics attribute table (item 13) so they can be model inputs/targets.
   - Well strip / geosteering view: log vs MD along the lateral with the target zone top/base and the
     seismic section along the wellbore underneath.
   - Fast with many wells: decimate trajectories by view (LOD), draw coloured segments as one
     ImageItem-free batched item (pyqtgraph PlotCurveItem per colour bin or a custom QPainterPath item).
   - Test data: synthetic horizontal wells (build section + 2-3 km lateral) through the synthetic 3D
     volume, with synthetic logs and stage CSV; the NZ deviated wells (Maui-5/6/7 surveys) for real
     trajectories. Tests: trajectory colours match the sampled attribute at MD; per-lateral stats correct.
15. **Scale + hardening** - status: done (run 1 + run 2; v0.18.0. Run 1: 15a Cancel for every threaded job, `wellindex.py`; 15b dev/bench.py + docs/BENCHMARKS.md, import hardening. Run 2: basemap well layer (labels only in view, well index above 2,000 wells), threaded import (bit-identical), section autoDownsample + sampled clip, vectorised dashboard pick. Remaining items need Chris's PC: run bench at 5-20 GB, then fix what it shows - see 'Later'.)
   (partial store removed), `wellindex.py` SQLite well index with grid-bucket spatial queries (10k wells: box query
   ~1 ms). 15b: dev/bench.py + docs/BENCHMARKS.md (0.6 GB: import 166 MB/s, slices 0.4-1.3 ms); import
   hardening (NaN/inf samples -> 0, holes + crossline step tested). LEFT: run bench at 5-20 GB on Chris's PC; wire
   the well index into basemap/dashboard for >2,000 wells; parallel import if the PC bench shows import is CPU-bound;
   overview volume only if zoomed-out display is slow there)
   PC bench 2026-11-23 (5 GB, docs/BENCHMARKS.md): inline 13.5 ms, time slice 15 ms, crossline 95 ms (fix: crossline
   access is strided - add an xline-major store or 64x64xns bricks, opt-in at import if disk allows), import 48 MB/s
   (50 GB ~18 min; Chris's C: is a Toshiba XG5 NVMe SSD (KXG50ZNV512G, GB/s class) -> import is CPU-bound:
   fuse the per-chunk conversions, write sorted traces as contiguous slabs instead of fancy-index scatter, thread
   the chunks, faster time-major transpose; target >= 150 MB/s on his PC)
   Done 2026-11-23 (interactive, v0.21.0): crossline copy at import + --xlstore / menu for old stores, contiguous slab
   writes for sorted SEG-Y, one block read feeds both copies, bench reports raw disk + import phases. Re-bench on PC., well index / laterals fine. Windows RAM probe fixed.
   - Scale targets (Chris wants big datasets): 3D volume 20-50 GB SEG-Y, 1,000+ 2D lines, 10,000 wells,
     1,000 laterals on the dashboard, all responsive on a 16-32 GB RAM PC.
   - Seismic: import in parallel (threads over trace chunks; numpy releases the GIL in I/O/copy), bricked
     store option (e.g. 64x64x64 bricks + small decimated overview volume for zoomed-out views), never load
     a whole volume into RAM, progress + cancel for every long job. RGT/attributes: multi-resolution
     (coarse solve on decimated grid, refine in bricks), out-of-core chunking, measured peak RAM.
   - Wells: SQLite well index (headers, tops, survey/log catalogue) with spatial index (grid buckets) so
     the map and dashboard query only visible wells; lazy log loading; npz cache already exists.
   - Benchmarks script `dev/bench.py` (synthetic 2-5 GB volume, 10k synthetic wells) recording import time,
     slice latency, map redraw time, peak RAM into docs/BENCHMARKS.md; fix the worst bottlenecks.
   - Read every module for bugs, edge cases (irregular 3D geometry, missing traces, dead traces, IBM vs
     IEEE, big files > 2 GB on Windows), performance on a 2 GB synthetic, docs/WHATS_NEW.md refresh.

16. **Well tie upgrade (HRS-style auto-correlate + constrained stretch/squeeze)** - status: done (run 2: 16a + 16b v0.19.0, 16c v0.20.0 - tie.py, dialog rows 2 + 3, test_tie.py. Follow-ups: multitaper wavelet button, shift map on the basemap, batch with DTW for sonic-only wells)
   Chris asked for this on 2026-11-23 (he likes the Hampson-Russell auto-correlate). Comparison with HRS, RokDoc,
   Kingdom/GeoSyn, Petrel, GeoGraphix + references: Claude doc "Well tie & synthetics" (link in RUNLOG).
   16a. Auto-correlate panel: window, correlation-vs-lag curve (FFT), polarity + phase scan, sub-sample peak,
        second-peak (cycle-skip) warning, best trace within N traces; fast auto_tie (FFT per stretch, stretch
        about the window top / anchor event, held-out-window score + NRMS).
   16b. Event-pair stretch/squeeze (<= 10 pairs, undo) + constrained DTW auto stretch/squeeze (Sakoe-Chiba band,
        max interval velocity change, default 10%); drift / interval-velocity QC panel with checkshots;
        checkshot-calibrated sonic; write the tied time-depth as a checkshot CSV.
   16c. Wavelets: deterministic least-squares (Tikhonov) + multitaper-coherence variant, constant-phase
        estimate, spectra + NMSE/predictability, JSON wavelet library; multi-well tie batch (table, bulk-shift
        map, average wavelet); log conditioning (despike, washout mask, Backus); angle synthetics (Shuey /
        Aki-Richards); deviated-well synthetic via laterals.wellbore_section.
   Tests: synthetic well with known shift, stretch, phase and a 3-segment velocity error -> recovered within
   1 ms / 0.5% / 5 deg; DTW warp respects the velocity cap; deterministic wavelet = true wavelet (corr > 0.95)
   at 10% noise; polarity flip detected; cycle-skip case flagged; multi-well batch on 10 synthetic wells.

17. **Dynamic depth conversion (Kingdom DDC-style)** - status: done (run 3, 2026-11-23: 17a v0.22.0, 17c v0.23.0, 17b v0.24.0, 17d v0.25.0, 17e v0.26.0. Optional follow-ups: Monte-Carlo on V0/k for P10/P90, basemap click to place a what-if well, real-data check on Maui once the 3D subset is imported)
   Chris asked for this on 2026-11-23: he likes Kingdom's Dynamic Depth Conversion (new well tops update the
   velocity model and depth maps immediately; used for geosteering prognosis). Petrel's equivalent is Domain
   Conversion (+ Velit = Kingdom VelPAK as a Petrel plugin, Cegal Blueback Rocks Velocity); GeoGraphix has
   average/interval models with "dynamic depth conversion using inter-well depth point integration".
   Builds on item 12 (`velocity.py` VelocityModel: constant V_int per layer, IDW, optional residual grid). Keep
   the item-12 behaviour as the default; everything below is additive. numpy only (no scipy / pykrige).
   17a. Velocity functions per layer: V0 (current), V0 + kZ (k from a least-squares fit of V_int vs mid-layer
        depth over the wells, or user-fixed k; V0 gridded, k constant per layer), V0 + kT, and "well T-D
        function" (layer uses the nearest/averaged checkshot curve). Depth in a V0 + kZ layer with interval
        time dt: z_base = (z_top + V0/k)·exp(k·dt/2) − V0/k. Cross-plot data per layer (V_int vs depth, fitted
        line, R², outliers flagged > 2 sigma) returned by a function the dialog can plot.
        Tests: synthetic V0 + kZ earth, 6 wells -> k within 2%, depths within 1 m; k → 0 gives the V0 result.
   17b. Gridding choices for V0 / residuals: IDW (current), thin-plate spline (port or import
        well_correlation/mapping.py TPS), simple kriging with a spherical/exponential variogram fitted to
        the well values (numpy solve, ≤ 500 wells). Optional seismic-velocity guide: a stacking/RMS velocity
        volume or grid -> Dix interval velocity per layer used as a trend, scaled to the wells (ratio or
        difference, then grid the scale factor). Tests: kriging exact at wells, smooth between; guide recovers
        a synthetic lateral trend that 3 wells alone miss.
   17c. **Dynamic update** (the core DDC feature): the seismic tool watches the linked .wcproj.json
        (QFileSystemWatcher + mtime check, debounce ~1 s) and the horizon files. When tops or horizons change,
        rebuild only what depends on the change: re-read the changed well(s), recompute their V_int rows, re-grid
        only the affected layers and the layers below, cache everything else. Depth maps, depth sections and
        the misfit table refresh in place; a status-bar line says what changed ("Maui-6 F0 top moved +4 m ->
        rebuilt layers 4-6 in 0.3 s"). Toggle in View: "Auto-update depth model". Also a "what-if" well: add a
        pseudo-top at map X,Y without touching the well project. Tests: change one top in a copied wcproj ->
        only layers at/below that horizon change; maps above are bit-identical; 50-well rebuild < 2 s.
   17d. Well prognosis / geosteering view: for a planned or drilling well (trajectory from the well project or
        laterals.py), list predicted depth (TVDSS + MD) of every horizon, the uncertainty (see 17e) and the
        change since the last update; prognosis updates live as 17c rebuilds. Export as CSV. Entering a new top
        while "drilling" re-runs the model with that well included and shows predicted depths ahead of the bit.
   17e. Uncertainty + scenarios: save several named velocity models (different functions / gridding / wells
        held out), compare depth maps (difference grids), leave-one-well-out blind test table (predicted vs
        actual top for each well = honest error estimate), and a simple P10/P50/P90 depth-to-target map from
        the blind-test error spread (optionally Monte-Carlo on V0/k). Depth-convert a full volume to SEG-Y in
        depth (reuse segy writer; threaded, cancellable) and depth -> time for depth grids.
   Data: Maui-5 / Maui-6 checkshot LAS are in data/nz; look for checkshots / VSPs in the other Maui and Tui
   well completion reports (NZP&M catalogue) - Chris to download if needed.
   Run order: 17a -> 17c -> 17b -> 17d -> 17e (dynamic update is the priority after the velocity functions).
   Update docs/WHATS_NEW.md with a "Dynamic depth conversion" section and test checklist.

18. **Basin model integrated with the Wheeler diagram** - status: done (run 4, 2026-11-23; v0.27.0-v0.31.0)
   Chris asked for this on 2026-11-23 (do it before item 19). Goal: burial, thermal and maturity history from the
   interpreted framework (horizons, RGT horizon stack, depth model of item 17, wells), shown in chronostratigraphic
   (Wheeler) space as well as in depth / map view - PetroMod / Genesis-style 1D + pseudo-3D, numpy only.
   18a. Chronostratigraphy: an age table (Ma) for horizons / formations (user-editable, saved in the seismic project;
        Taranaki template with published ages for the Maui formations as a starting point); RGT stack levels get ages
        by interpolation between dated horizons. Wheeler diagram with an **age axis** (option next to the RGT axis):
        hiatus / erosion gaps shown in Ma, labelled. Lithology per zone from the wells (Vsh -> sand / shale /
        carbonate fractions via well_correlation lithology) or typed in. Tests: ages interpolated monotonic, a
        synthetic pinch-out shows a gap of the right age span.
   18b. Burial history 1D (wells + pseudo-wells at any map point): present-day thicknesses from the depth model,
        backstripping / decompaction (Athy phi = phi0 exp(-c z) per lithology, mixed by fractions), erosion events
        (missing thickness typed in or estimated from Wheeler gaps), optional paleo-water depth and sea level.
        Burial curves plot (depth vs age, one line per horizon). Tests: grain thickness conserved; decompacted
        thickness matches the analytic Athy solution; zero-porosity lithology = no compaction.
   18c. Thermal + maturity: surface temperature history, basal heat flow (constant, piecewise, or McKenzie rift
        transient), bulk conductivity from lithology + porosity (geometric mean), 1D transient conduction (implicit
        finite differences on the moving burial grid, radiogenic heat optional). Calibration to well temperatures
        (BHT read from LAS / DLIS headers where present - Kiwi-1, Maui-4, Tui-1 - with a Horner-type correction) by
        fitting heat flow. Maturity: EASY%Ro (Sweeney & Burnham 1990) and Lopatin TTI; transformation ratio for
        type I / II / III kerogen with a simple published kinetic set; oil / gas window timing. Tests: steady state =
        analytic geotherm; EASY%Ro against published reference values; heat-flow fit recovers a synthetic value.
   18d. Wheeler + map integration: colour the Wheeler diagram by temperature / %Ro / TR / porosity at the time of
        deposition or today; run the 1D model on a grid of pseudo-wells (threaded, cancellable) for maps of
        present-day temperature, %Ro, TR and "time of oil window entry" at any horizon; burial section along an
        inline / 2D line; petroleum-system event chart (source, reservoir, seal, trap, generation / expulsion) from
        the ages and the model. Live update with item 17's dynamic depth model where cheap.
   18e. GUI: Interpret > Basin model window (age table, lithologies, heat-flow + calibration plot, burial / temperature /
        maturity plots per well, map runs, Wheeler colouring), CSV / CPS-3 exports, docs (WHATS_NEW section and
        test checklist). GUI logic tested under dev/fakeqt.
   Run order: 18a -> 18b -> 18c -> 18d -> 18e.

19. **Seismic inversion package (post-stack + pre-stack)** - status: done (run 4, 2026-11-23; v0.32.0-v0.35.0)
   Chris asked for this on 2026-11-23 (after item 18). Builds on item 16 (well tie, wavelets, Aki-Richards / Shuey,
   logs_in_time) and item 17 (vgrid kriging / guide, horizons framework). numpy only (no PyLops / scipy).
   19a. Low-frequency model: well AI (and Vp, Vs, rho for pre-stack) in time from the ties, conditioned (despike,
        washout, Gardner / Castagna fill), interpolated through the structural framework (proportional / parallel to
        top / parallel to base per zone, or along RGT stack levels) with IDW / kriging / velocity-guided trend
        (vgrid), then low-pass filtered (zero-phase Butterworth in numpy, cut picked from the seismic spectrum).
        Stored as a volume. Tests: wells honoured, no layer-crossing trends, filter response.
   19b. Post-stack: coloured inversion (operator from the well AI spectrum vs the seismic spectrum, relative AI) and
        model-based inversion (trace by trace in log-impedance, wavelet from item 16, LFM prior, Tikhonov or blocky
        L1 / TV via IRLS, optional lateral smoothing between traces); threaded, cancellable, writes a volume. QC:
        synthetic vs seismic residual, blind-well test. Tests on a synthetic earth with known AI (recovered within a
        few % in band), blind well.
   19c. Pre-stack simultaneous inversion: partial angle stacks (several volumes with angle ranges) or angle gathers;
        Fatti / Aki-Richards linearisation in ln Zp, ln Zs, ln rho with background trend constraints; wavelet per
        angle stack. Outputs Zp, Zs, rho, Vp/Vs, lambda-rho, mu-rho volumes. Tests on synthetic angle stacks.
   19d. GUI + QC: Interpret > Inversion window (LFM build, wavelet choice, parameters, run, QC plots, cross-plots of
        the results coloured by a well log, export SEG-Y). Note: the public NPRA lines and Maui 3D are stacks, so
        pre-stack is tested on synthetics unless partial stacks turn up in the NZP&M catalogue.
   Run order: 19a -> 19b -> 19c -> 19d.

## Phase 2 - Chris's brainstorm (added 2026-11-24, interactive session)
Source: Chris's list (Exploration / Petrophysics / M&A / General). Order is by dependency: petrophysics + the
integrated map come first because seal, DHI, fizz gas, inversion, Rose risking and the reservoir-engineering items
all consume their per-zone results. Design decisions (Chris can override):
- Petrophysics lives in the Well Correlation Tool as a new package `well_correlation/petro/` (LAS/DLIS, curves,
  zones, tops are already there). Results are saved per well in `<wcproj>_petro/<well>.npz` (computed curves) plus a
  per-zone summary table `<wcproj>_petro/summary.csv`, so the Seismic tool (basemap, dashboard, analytics, inversion,
  Rose) reads them through the linked .wcproj without new links.
- ONE integrated map: the Seismic tool's basemap/dashboard map becomes the "Map hub" (it already has wells, laterals,
  grids, the 3D outline and 2D lines). The Well Correlation Tool gets a "Show on map hub" action (writes the summary and
  the hub reloads on file change, same pattern as the DDC watcher from 17c), and its own mapping.py stays as is.
- Rose & Associates: their software and DHI Consortium database are proprietary. Build Rose-style methodology
  (chance factors, lognormal volumetrics, DHI-based Pg revision with user-editable weights) and CSV export; no copying
  of their database or file formats.
- M&A / completions test data: synthetic fields generated at test time (extend dev/make_synthetic_engineering.py) +
  readers for the public CSV exports Chris downloads (Texas RRC, NDIC, FracFocus - FracFocus helper exists in item 13).
- Animations are written as GIF (PIL) and an MP4 only if ffmpeg exists on the PC (optional, try/except).
- All numpy. Optional ML backends only behind try/except (CLAUDE.md rule 4).

20. **Petrophysics core (Geolog / Interactive Petrophysics style)** - status: done (run 5, 2026-11-24: 20a v0.6.0 tools / QC, 20b v0.7.0 interp, 20c v0.8.0 multimin + Monte Carlo, 20d v0.9.0 rock physics, 20e v0.10.0 Petrophysics window + section tracks. Follow-ups: core-log calibration (22b), thin-bed (laminated sand) analysis, BHT-based temperature, Petro results on the map hub (21b))
   Reference workflows: IP (Interpretation wizard, Mineral Solver, Monte-Carlo, cutoff sensitivity, Pickett /
   Buckles plots, parameter zones, batch), Geolog (Determin / Multimin, project-wide sets, tool-aware logic).
   20a. Data QC + tool type + resolution: identify the logging tool / vendor / service from DLIS frame + tool objects
        and LAS ~Parameter / ~Curve descriptions and mnemonic tables (e.g. HALS/HRLA/DSI/PEX = SLB, HDIL/XMAC = Baker,
        ACRt/WaveSonic = Halliburton; editable JSON table), so corrections and default parameters depend on the tool
        (array induction vs laterolog, wireline vs LWD). Data frequency: sample interval, effective vertical
        resolution per curve (tool resolution table vs sample rate), thin-bed flag where beds (from 20b bed-boundary
        detection) are below resolution; report per well. Curve QC: gaps, flat lines (exists), spikes, caliper /
        bad-hole flag, depth-shift (cross-correlate to a reference GR, reuse autocorr.py), splice runs.
   20b. Deterministic interpretation per zone with parameter sets: Vsh (GR linear / Larionov old + Tertiary /
        Clavier / Steiber, SP, neutron-density, min-of), porosity (density, sonic Wyllie / Raymer-Hunt-Gardner,
        N-D crossplot with gas correction, effective vs total), Rw (Pickett fit, SP, Rwa minimum, temperature from
        the basin-model geotherm of item 18), Sw (Archie, Simandoux, modified Simandoux, Indonesia, Waxman-Smits /
        dual-water), permeability (Timur, Coates, Wyllie-Rose), TOC (Passey delta-log-R) for source rocks, bed
        boundaries. Cutoffs (Vsh, PHIE, Sw) -> net/gross, net res, net pay, PHIH, SoPhiH, HCPV per zone, cutoff
        sensitivity curves (IP-style), all written to the summary table.
   20c. Multi-mineral solver (Multimin / Mineral Solver): constrained weighted least squares (numpy NNLS + sum-to-1)
        over GR, RHOB, NPHI, DT, PEF, Rt with editable end-points (quartz, calcite, dolomite, illite, ...) and fluid;
        reconstructed-curve misfit as QC. Monte-Carlo uncertainty on parameters (Rw, m, n, a, end-points, cutoffs) ->
        P10/P50/P90 of PHIH / SoPhiH per zone.
   20d. Rock physics at wells (needed by items 23-25): Vs prediction (Greenberg-Castagna), dry-rock moduli,
        Gassmann fluid substitution (brine / oil / gas / fizz at 5-10 % gas sat; Batzle-Wang fluids at pressure and
        temperature), elastic logs (AI, SI, Vp/Vs, lambda-rho, mu-rho), Young's modulus + Poisson's ratio +
        brittleness index (also used by item 30c).
   20e. GUI: Petrophysics window in the Well Correlation Tool (zone parameter table, cross-plots: N-D, M-N, Pickett,
        Buckles, histograms with discriminator; results tracks in the section view; batch over selected wells;
        "Show on map hub"). Tests: synthetic wells with known mineralogy / porosity / Sw generated from the forward
        models (recover within tolerance), Archie / Simandoux against hand-computed values, NNLS against an analytic
        2-mineral case, Gassmann round trip, real-data run on Maui-5/6/7 + Tui-1 (DLIS) recorded in the RUNLOG.
   Run order: 20a -> 20b -> 20c -> 20d -> 20e.

21. **Integrated Map hub (petrophysics + seismic + wells on one map)** - status: done (run 5, 2026-11-24: 21a v0.36.0 layers, 21b v0.37.0 petro on the map, 21c v0.38.0 gridding + cokriging + calculator, 21d v0.39.0 new-wells workflow, 21e v0.40.0 cross-check, 21f v0.41.0 cartography (+ well_correlation carto v0.11.0), 21g v0.42.0 GIS package + layouts, 21h v0.43.0 QGIS link + plugin. Follow-ups: deviated-well trajectories on the hub map, web tiles (optional), LZW GeoTIFF, QGIS map tools for live drawing (now: select a feature and press Send), first real test in Chris's QGIS)
   Chris: "having an integrated map is very important". Extends gui/basemap.py + dashboard_window.py.
   21a. Layers manager: background grid (structure, isochron, attribute, inversion slice / horizon extraction,
        predicted property, basin %Ro), contours, fault polygons, 2D lines, 3D outline, wells (surface + BH,
        trajectories), laterals, prospect polygons, text; per-layer opacity, order, colour bar; saved in the project.
   21b. Petro on the map: post any zone summary value (net pay, PHIH, SoPhiH, Vsh, Sw, TOC) as labelled values,
        bubbles or pies (mineral fractions from 20c); filter by zone; click a well -> petro strip (logs + results)
        and its row in the table; lasso -> linked cross-plot.
   21c. Gridding of zone properties: IDW / TPS / kriging (vgrid), plus collocated cokriging and kriging with
        external drift using a seismic attribute or inversion map as the secondary (correlation shown, cross-
        validation per method, leave-one-out residual map). Map arithmetic (grid calculator: +, -, *, /, clip,
        mask by polygon) - SeisWare-style.
   21d. "Add wells" workflow: one action when new wells arrive -> import LAS/DLIS, batch tie (item 16), petro batch
        (20), re-grid affected maps, update the LFM (19a) and DDC (17c), list of what changed with before/after
        maps and a residual-at-new-wells table (the honest test of the old maps).
   21e. Cross-check view: petro values vs seismic attribute / inversion at the wells (cross-plot + map of residuals),
        hook for analytics (item 13) to predict a property map with uncertainty.
   21f. Cartography (Chris 2026-11-24: "can the mapping look better?"; mock-up in RUNLOG): shared map-style module used by
        the map hub, the Well Correlation Tool map and PNG/PDF export. Hillshaded colour fill (multi-directional
        light, vertical exaggeration slider) under smooth perceptual colour ramps (depth, thickness, property,
        diverging residual; reversed for depth), minor + bold index contours with labels along the line (white halo,
        no overlaps), grids blanked inside fault polygons and gridded per fault block so contours stop at faults,
        fault heave polygons (hatched, downthrown ticks), closure / spill and contact contours, industry well symbols
        (oil, gas, oil+gas, show, dry, P&A, planned; deviated with BH symbol and trajectory), haloed labels with
        collision avoidance, posted values, scale bar, north arrow, graticule / coordinate ticks with the CRS, legend
        and a title block (project, horizon, CI, date, author) for export at 300 dpi; style presets saved per project.
   21g. ArcGIS-style page layouts + GIS exchange (Chris 2026-11-24: "pretty maps like ArcGIS"; mock-up in RUNLOG):
        - Layout composer: A4 / A3 / A1 portrait or landscape, main map frame + side panel with title block,
          locator inset (red extent box), legend built from the visible layers, colour bar, scale bar, north arrow,
          lat/long graticule (inverse transverse Mercator in numpy next to geo.tm_forward) plus projected ticks,
          projection / data source / author / date text, neatline; templates saved per project; export PDF + PNG
          (300 dpi) via matplotlib.
        - Cartographic context offline: coastline + bathymetry / topography as shaded relief (land and sea ramps),
          named seas / land labels, licence / permit polygons, fields, infrastructure from user shapefiles / CSV.
          Source data Chris downloads once (Natural Earth coastlines, GEBCO bathymetry GeoTIFF, LINZ / NZP&M permit
          shapefiles) read by numpy readers (shapefile .shp/.dbf, GeoTIFF uncompressed / deflate via zlib, GeoPackage
          via sqlite3). Optional web tiles (satellite / topo) cached to disk with urllib, off by default.
        - GIS export so final layouts can also be made in ArcGIS Pro / QGIS: grids as GeoTIFF with CRS, contours / fault
          polygons / wells / laterals / prospects as shapefile + GeoPackage + KML, .prj from the project CRS.
        - Any vector layer (Chris 2026-11-24: "any other shapefile data like township / range, county, infrastructure"):
          generic shapefile / GeoPackage / KML / CSV-with-WKT layer loader, not tied to one data set. Examples Chris
          will use: BLM PLSS (CadNSDI township / range / section, with "T12N R3W S14" labels by zoom level), US Census
          TIGER counties / states, pipelines + processing plants + compressor stations (EIA / state), roads, railroads,
          leases / units / DSUs, field outlines, protected areas, NZ equivalents (LINZ parcels, NZP&M permits).
          Per-layer style: symbol / line / fill / hatch, categorised or graduated colour by an attribute field, label
          by field(s) with halo and scale-dependent visibility, filter by attribute expression, identify-on-click
          (attribute table row), layer in legend. Attribute table dock (sort, filter, select -> highlight on map,
          linked to the dashboard selection). "Go to" search by PLSS legal description (Sec / Twp / Rng / meridian)
          or county name; well spots from legal description + footage calls where no X/Y.
          Coordinate systems: read .prj WKT and reproject in numpy - geographic, transverse Mercator / UTM, Lambert
          conformal conic (most State Plane zones), Albers equal area, NZTM / NZGD2000; NAD27 -> NAD83 by a
          clearly flagged approximate shift (no NADCON grids), EPSG lookup for the common codes. Large layers
          (statewide PLSS sections, national pipelines): grid-bucket spatial index (as wellindex.py), clip to view,
          Douglas-Peucker simplification per zoom, cached as npz next to the source.
          Tests: synthetic shapefiles with points / multi-part polygons with holes / polylines + dbf attributes
          (written by our writer, read back), LCC + Albers + TM against published EPSG test points (< 1 m), PLSS
          legal-description lookup on a synthetic township grid, filter + categorised style, 1M-vertex layer draws
          with simplification under the target time.
        - Tests: shapefile + GeoTIFF round trip (write -> read -> identical), inverse TM against forward TM < 1 mm,
          legend lists exactly the visible layers, layout geometry at each page size (no overlapping panels).
   21h. Live two-way link with QGIS (Chris 2026-11-24: has QGIS only, no ArcGIS Pro; wants it interactive).
        - Shared project GeoPackage `<project>_gis.gpkg` (sqlite3, no new deps) written / updated by GeoHarmonic:
          wells (status, petro summary per zone), tops, laterals + stage attributes, drainage ellipses, faults
          (sticks -> map polygons), contours, prospects, 2D lines, 3D outline; grids as GeoTIFF with internal
          overviews (pyramids) next to it. CRS from the project. Read back user-edited layers (prospect polygons,
          leases, section lines) with change detection. ArcGIS users can open the same GeoPackage.
        - QGIS plugin "GeoHarmonic Link" (tools/QGIS Plugin/geoharmonic_link, pure Python, `qgis.PyQt` shim so it works
          on Qt5 and Qt6 builds of QGIS): Open GeoHarmonic project -> adds all layers grouped and styled (QML styles
          generated from the 21f cartography presets), auto-refresh when the gpkg changes, install via "Install from
          ZIP" (build a zip in dev/).
        - Live messages over localhost TCP (127.0.0.1, JSON lines, stdlib socket; off unless both sides enable it;
          clean failure when the other app is not running): select / zoom wells in either app and the other follows;
          click a well in QGIS -> open its strip / petro / tie in GeoHarmonic; draw a line in QGIS -> arbitrary seismic
          section in the Seismic tool; draw / edit a polygon -> prospect created / updated and volumetrics (28)
          recomputed; the section cursor in GeoHarmonic shown as a moving marker in QGIS; "Send map to QGIS" for any
          grid, attribute, inversion slice or co-blend image.
        - Tests here (no QGIS installed): gpkg written and read back with sqlite3 (geometry blobs, srs table, rtree),
          GeoTIFF overviews readable, message protocol both directions between two Python processes, plugin logic
          under a fake `qgis` module (like dev/fakeqt). First real test is Chris's QGIS.
   Tests: layer state save/restore, contours never cross a fault polygon, label boxes do not overlap, cokriging against an analytic case (secondary perfectly correlated -> matches
   the secondary trend; uncorrelated -> matches kriging), LOO numbers, add-wells updates only affected grids.

22. **Petrophysics analytics + field reports** - status: done (run 5, 2026-11-24: well_correlation v0.12.0 `petro/logml.py` (22a), `petro/reports.py` (22b), section tracks FAC + DTP, Petrophysics window buttons, map hub posts report counts / contacts (Seismic v0.44.0). Not done: report posting as markers on the log section, MICP -> top seal (item 27c consumes it))
   22a. ML on logs (item 13 models, numpy GBM / RF): predict missing curves (DT, RHOB, Vs) with depth-blocked CV
        and uncertainty, electrofacies (k-means / Gaussian mixture in numpy) with facies logs to the section and map,
        petro-to-seismic property prediction feeding 21e.
   22b. Field reports: importers for shows, DST / production tests, cores (porosity, perm, grain density), MICP
        (used by top seal in 25), mudlog gas (Tui-1 CSV exists) and pressures (RFT/MDT -> fluid gradients, contacts)
        from CSV/XLSX templates (pandas) and plain-text reports with a keyword extractor (depth + show / test
        phrases, flagged "auto-extracted, check"). PDF text only if pypdf exists (optional). Posted on the logs and
        the map hub; core data used to calibrate 20b (core-log cross-plot, m / n fit).
   Tests: blocked-CV leakage check, GMM on synthetic clusters, report parser on sample text, pressure-gradient
   contact pick on a synthetic two-fluid column.

23. **Viewer quick wins: co-blending, bump map, interactive linking (SeisWare-style)** - status: done (run 5, Seismic v0.45.0; follow-ups: co-blend on stratal slices / 2D lines, blend legend drawn on the basemap, movie export to GIF)
   23a. Co-blending on sections, time slices, horizon maps and the map hub: RGB / CMY / HSV of 2-3 attributes (e.g.
        spectral 10/20/40 Hz from attributes.py), per-channel clip + gamma, legend (colour triangle / channel
        swatches - the legend OpendTect lacks), opacity blend of an attribute over the seismic.
   23b. Bump map / hillshade: directional illumination (azimuth, elevation, exaggeration) of horizon grids and of
        amplitude-on-structure (colour = amplitude, shading = structure), also for time slices.
   23c. Interactive: live-linked cursor across section, map hub, well strip and dashboard; attribute parameter
        "movie" scan (slider redraws the section for a parameter range); saved window layouts / templates.
   Tests: blend channel maths, hillshade against an analytic plane / dome, cursor link messages under fakeqt.

24. **Seismic resolution, single-well inversion, predictive 2D modelling** - status: done (run 5, Seismic v0.46.0-v0.47.0; follow-ups: probability maps in the map hub UI, stats cross-plot window, classes beyond the sand-top features (e.g. AI / Vp/Vs from the item-19 inversion))
   24a. Recommended frequency from existing data: dominant frequency + usable bandwidth (spectrum vs noise floor)
        per window and per area, interval velocities from the wells / velocity model, tuning thickness (lambda/4,
        lambda/8) maps, bed-thickness distribution from the logs (20a) -> recommended target frequency / bandwidth
        for the reservoir and the expected gain from spectral balancing (shown on wedge synthetics).
   24b. Single-well inversion: the item-19 model-based / simultaneous inversion run on the trace(s) at one well,
        with Monte-Carlo over wavelet, LFM and noise -> uncertainty bands vs the logs; answers "can the seismic see
        this reservoir property here?".
   24c. Predictive 2D modelling + amplitude statistics: stochastic pseudo-wells (vary thickness, Vsh, porosity,
        fluid via 20d Gassmann) along a 2D section or wedge, synthetic gathers / stacks; amplitude-variation stats
        (amplitude, intercept-gradient, inverted property) vs property and fluid -> Bayesian classification of the
        real seismic into probability-of-pay / fluid volumes and maps (with the stats plots behind it).
   24d. Synthetics as PaleoScan seeds: tied-well synthetic events (16) become horizon seeds and RGT control points
        (rgt/solve add_horizon), so the RGT and horizon tracking are guided at the wells.
   Tests: tuning thickness vs a wedge model, Monte-Carlo bands contain the true synthetic, classifier recovers the
   fluid of a synthetic section, RGT honours well seeds.

25. **AVO, DHI, fizz gas, angle-stack risk** - status: done (run 5, Seismic v0.48.0; follow-ups: IG cross-plot window with brushing to the section, gather input (pre-stack gathers not stored yet), DHI plots per check, fizz assessment in a dialog)
   25a. AVO at wells and on data: Zoeppritz / Aki-Richards / Shuey, AVO class (I, II, IIp, III, IV), intercept-
        gradient from angle stacks or gathers, fluid factor, IG cross-plot with background trend and anomalies
        highlighted on the section / map; Gassmann scenario synthetics from 20d.
   25b. Fizz gas: low-gas-saturation case modelled explicitly (Vp drops like full gas, density does not), so the
        discriminator is density / mu-rho / Vp/Vs-rho from the item-19c pre-stack inversion; low-frequency shadow and
        attenuation (spectral attributes) as supporting evidence; a fizz-risk score per anomaly with its reasoning.
   25c. Angle-stack vs full-gather risk: for the actual angle ranges, noise and wavelet, simulate inversion from
        2-3-5 angle stacks vs full gathers on the well pseudo-models and report the expected error of Zp / Zs / rho
        (density is the usual casualty with max angle < ~35 deg); result feeds the DHI confidence and Pg.
   25d. DHI checklist (Rose / DHI-Consortium style, user-editable weights): amplitude strength vs background, fit to
        structure / conformance (amplitude vs depth-contour fit over the closure), flat spot, phase / polarity change,
        AVO behaviour, edge / termination sharpness, consistency across stacks, data quality; each scored with the
        plot that justifies it; output a DHI index per prospect for item 28.
   Tests: AVO class from known elastic contrasts, fizz vs commercial gas separated only by density, gather vs stack
   error ordering, conformance score on a synthetic anomaly that fits / does not fit a closure.

26. **Structural timing, paleo-structure animations, structural reconstruction** - status: done (run 5, Seismic v0.49.0; follow-ups: phase column + growth table in the basin / Wheeler window, paleo-depth map animation in the map hub, inclined-shear unfolding, restoration UI picking the fault on the section, MP4)
   26a. Wheeler with tectonic phases: tag each age interval pre-rift / syn-rift / post-rift (inversion, other) in the
        age table (18a); fault growth analysis (throw vs depth / time per fault from horizons + sticks, expansion
        index across faults) proposes the syn-rift interval; colour the Wheeler by phase.
   26b. Paleo-structure: successive flattening on each dated horizon (RGT / horizons), decompaction + backstripping
        from the basin model (18b), paleo-depth maps per age; animation oldest -> youngest of sections and maps
        co-rendered with lithology (petro / electrofacies from 20-22 projected along RGT) - and the same with the
        inversion volumes (AI / Vp/Vs, item 19) as Chris asked. Export GIF (MP4 optional).
   26c. Structural reconstruction on 2D sections: remove fault throw horizon by horizon (block restoration matching
        hanging-wall / footwall cutoffs), vertical / inclined shear unfolding, line-length / area checks; per-step
        restored sections in the animation.
   Tests: flattening honours the datum, decompaction conserves grain thickness (exists), restoration of a synthetic
   faulted section returns the unfaulted layers, growth index on a synthetic growth fault.

27. **Migration and seal (charge, fault seal, top seal)** - status: done (run 5, Seismic v0.50.0; follow-ups: per-time-step migration on 26b paleo-depth grids in the UI, Allan diagram plot along a fault plane from sticks, fault-as-conduit in the flow model, seal risk map layers from petro grids, critical-moment chart window)
   27a. Migration prediction: map-based flow-path ray tracing on paleo-structure surfaces (26b) per time step,
        drainage areas, fill-spill chains, accumulations; charge timing from basin-model expulsion (18c TR) vs trap
        timing (age of paleo-closure from 26b) -> critical-moment chart per prospect.
   27b. Fault migration and fault seal: juxtaposition (Allan diagram along the fault plane from horizons + sticks),
        shale gouge ratio from Vsh (20b) and throw, SGR -> supported column height (editable calibration), fault-as-
        conduit along up-dip fault segments in the flow-path model, slip / dilation tendency from a stress tensor
        (SHmax azimuth + magnitudes; shared with item 29).
   27c. Top seal: seal thickness + Vsh maps from petro (map hub), capillary entry pressure from MICP (22b) or
        editable defaults -> column height supported vs spill / fault-seal limited columns; seal risk maps.
   Tests: flow paths on an analytic dome / saddle, fill-spill order, SGR on a hand case, column height maths.

28. **Prospect risking + volumetrics (Rose & Associates style)** - status: done (run 5, Seismic v0.51.0; follow-ups: prospect window (polygon pick, distributions editor, factor table with evidence, portfolio view), base-grid GRV in the UI, Rose file import)
   28a. Prospect object: polygon + reservoir zone; GRV from depth maps (17) with contacts at spill, fault-seal or
        top-seal limited depth (27), area-depth / area-thickness curves; Monte-Carlo volumetrics (lognormal / user
        distributions for area, net pay, phi, Sw, FVF, RF) -> P90/P50/P10, Swanson's mean.
   28b. Chance of success: Rose-style factors (source/charge, reservoir, trap, seal/containment, timing/migration)
        prefilled from items 18, 20, 25, 27 with the evidence attached; DHI-based Pg revision from the 25d index;
        dependency between prospects; portfolio table, CSV export (Rose-compatible columns where known).
   Tests: lognormal maths and P-values, GRV of an analytic cone, Pg revision bounded and monotonic in the DHI index.

29. **Reservoir engineering foundations: spacing, stress, drainage ellipses, gun barrel** - status: done (run 5, Seismic v0.52.0; follow-ups: ellipses as a map-hub layer, SHmax grid input, DSU polygons from PLSS, gun-barrel with zone tops from the depth maps, real RRC / NDIC column check on Chris's exports)
   29a. Data: readers for Texas RRC, NDIC and FracFocus CSV exports + synthetic unconventional field generator (test
        time); stress: SHmax azimuth (user / World Stress Map CSV / map grid), magnitudes optional.
   29b. Spacing: 3D inter-well distances (horizontal + vertical offset in the same / adjacent zone), wells per
        section / per DSU, parent / child / co-completed classification by first-production date.
   29c. Drainage ellipses: per well, half-length along SHmax and width across it (from completion size, frac
        geometry of 30c or user), drawn on the map hub around each lateral, overlap areas flagged.
   29d. Gun-barrel view: section perpendicular to the laterals (TVD vs lateral offset) with zones, wells and their
        drainage ellipse cross-sections, linked to a map strip showing the section line and ellipses.
   Tests: distances on a known pattern, ellipse orientation follows SHmax, gun-barrel offsets match geometry.

30. **Depletion, resource left, recovery factor, RTA, frac complexity, M&A screen** - status: done (run 5, Seismic v0.53.0; follow-ups: production import (RRC / NDIC monthly) wired to the dashboard, RTA plot window, remaining-resource map layer, economics dialog with a price deck, one-page summary with maps)
   30a. Depletion factor (Chris: "!!"): for each child well, parent cumulative production inside the child's
        drainage volume at the child's first production / in-place volume of that drainage volume (from 21 petro
        maps), plus a pressure-depletion proxy; child performance vs depletion factor cross-plot.
   30b. Resource left + RF automation: in-place per unit / zone from petro maps (SoPhiH x area), drained volume from
        ellipses, EUR (Arps from item 13) -> recovery factor per well automated over all wells, remaining-resource
        map and count of undrilled locations at a chosen spacing.
   30c. RTA (rate-transient analysis): rate-normalised plots vs material-balance time, square-root-time linear-flow
        plot (A*sqrt(k) estimate when pressures are available, rate-only proxy otherwise, clearly flagged), end of
        linear flow -> drainage estimate to check the ellipses. Complex vs planar fractures from the frac job: fluid
        type (slickwater / hybrid / gel), rate, fluid + proppant per ft, cluster spacing, ISIP / net pressure if
        present, brittleness (20d) and horizontal stress anisotropy -> fracture complexity index and implied drainage
        shape (narrow planar vs wide SRV) fed back to 29c.
   30d. M&A screen: area type curves (P10/P50/P90 by zone and vintage), PDP / PUD inventory from 30b, simple economics
        (price deck, royalty, opex, capex, NPV10, PV per location) and a one-page area summary export. Uses the item-13
        analytics for completion-vs-geology attribution.
   Tests: depletion factor on a synthetic parent-child pair, RF = EUR / OOIP on a hand case, RTA linear-flow slope on
   a synthetic analytic linear-flow well, economics against a hand-computed cash flow.

## Phase 3 - Live Map + 3D module (added 2026-11-24, interactive session)
Chris asked for these on 2026-11-24. Order: 31 (Live Map), 32 (automated fault mapping), 37 (velocity-volume depth conversion, added 2026-11-25, placed before 33), then the 3D module 33 -> 34 -> 35 -> 36. Full design
notes: Claude project docs "GeoHarmonic Live Map plan item" and "GeoHarmonic 3D Module plan item" (3D items renumbered 33-36).
- The Live Map EXTENDS the item-21 Map hub (layers, gridding, cartography, GIS package, QGIS link already exist) - do
  not rebuild what item 21 has; add the shared map bus and the missing interaction.
- Standing rule: every tool integrates with the Live Map through the shared map bus (selection, filters, active
  line/well/horizon, data-changed events). The 3D window follows the same rule.
- 3D rendering without new dependencies (CLAUDE.md rule 4): PySide6's own QtOpenGL (QOpenGLWidget + shaders via
  QOpenGLFunctions). pyqtgraph.opengl / PyOpenGL / pyvista only as optional extras behind try/except, installed by an
  optional `Install 3D extras (optional).bat` (same pattern as the analytics extras). Kriging, SGS, SIS, geobodies,
  connected components: numpy (no scipy, gstools, GemPy). Rendering logic is tested without a GPU: mesh/slice/colour
  building is pure numpy; only the draw calls touch OpenGL.

31. **Live Map (Kingdom-style live base map, Spotfire-style styling, map bus)** - status: done (run 6, 2026-11-25, Seismic v0.54.0 / Well Correlation v0.13.0 / Mistie 2.1.0: `mapbus.py`, `livemap.py`, `livegrid.py`, `gui/live_map.py`, `gui/maplink.py`, gis `geojson.py` + `zmap.py`, well_correlation + mistie_tool `maplink.py`. Follow-ups: first real-Qt check on the PC; polygon fills ignore holes; the Dashboard window keeps its own selection model (not yet on the bus); inversion / attribute maps are added through Map layers / Horizon maps, not auto-published; QGIS live link does not carry the Live Map well styling)
   31a. Map bus / shared project services (build first): SelectionService (wells/lines/points), FilterService,
        ActiveObject (current inline/crossline/2D line/well/horizon), DataChanged events (picks, tops, grids, misties,
        velocities), project WellTable (column store any tool can add columns to). Pure Python, unit-tested without
        Qt. Retrofit the mistie tool, well correlation, seismic viewer and depth conversion to publish/subscribe
        (cross-tool via the existing linked-file + watcher pattern from 17c / 21).
   31b. Map window core on the item-21 hub: project CRS, cursor X/Y readout, scale bar, north arrow, layer tree
        (visibility/order/opacity), map state saved in the project.
   31c. Culture / shapefile layers: pure-Python .shp/.shx/.dbf/.prj reader (point, polyline, polygon, multipart) if
        item 21g does not already cover it, plus GeoJSON and CSV X/Y; attribute table per layer; style by attribute
        (fill, outline, dash, hatch, labels). Lease/permit blocks (own / competitor / open), Township & Range (PLSS),
        infrastructure, no-go areas, coastline. CRS mismatch warning (pyproj optional). "Add shapefile..." button.
   31d. Well layer with Spotfire-style styling: colour by any WellTable column (continuous, categorical, rule-based
        e.g. "mistie > 10 ms = red"), size by column (bubbles), pie symbols, shape/label by column; Filters panel
        (range sliders, category checkboxes) hides wells instantly; deviated well paths; linked click/lasso selection
        shared with a well table and a crossplot/histogram panel.
   31e. Seismic layers linked to the viewer: 3D outline, 2D lines with shotpoints, live indicator of the current
        inline/crossline/2D line; click a line to open it; draw an arbitrary line -> arbitrary section; misties drawn.
   31f. Well correlation integration: click wells or draw a line to build a cross section; section line on the map;
        selection syncs both ways; picked tops push to the WellTable and map instantly.
   31g. Live gridding and depth: picks and tops feed a background gridding worker (coarse while picking, full on
        pause) with contours and colour fill; fault polygons as barriers; DDC (item 17) updates the depth map live;
        grid math (isochore, add/subtract, clip by polygon).
   31h. Links to other modules: basin model / Wheeler section locations clickable; inversion and attribute
        extractions as map layers; any future tool registers its layers and selection through the map bus.
   31i. Output: PNG/PDF with title block and legend; picks, polygons, section lines to shapefile/GeoJSON; grids to
        ZMAP+ (GeoTIFF where item 21 already writes it).
   Tests: map bus, shapefile reader (tiny .shp built in tests), styling rules, filters, gridding; GUI wiring under
   dev/fakeqt; HANDOFF + WHATS_NEW note on opening the Live Map on Windows (step-by-step).

32. **Automated fault mapping (PaleoScan AFE-style + fault likelihood)** - status: todo
   Chris asked for this on 2026-11-24: the PaleoScan-style features must include fault mapping. Builds on item 2
   (manual sticks / fault surfaces), item 4 (RGT, fault-aware), item 11 (coherence, dip-steered variance), items
   26-27 (fault growth, fault seal). numpy only, chunked out-of-core, Cancel + progress on every job.
   32a. Fault attributes: fault likelihood (Hale 2013 style - semblance along structure-oriented windows, raised to a
        power and scanned over fault strike and dip, smoothed along the candidate plane), plus fault strike and fault
        dip volumes; thinned fault likelihood (TFL, ridges only); ant-track-style enhancement of any edge attribute
        (coherence / variance / TFL). Stored as float16/uint8 volumes, viewable like any attribute.
   32b. RGT fault throw attribute (PaleoScan's signature method): vertical displacement of iso-RGT surfaces across
        discontinuities -> fault throw cube; throw-weighted fault probability so large faults are kept and noise is
        dropped.
   32c. Automatic fault extraction: TFL / throw maxima -> fault points with strike + dip -> link into fault patches
        ("fault skins": grow from high-likelihood seeds, link neighbours with compatible strike/dip) -> triangulated
        fault surfaces stored as the same fault objects as item 2; auto-generate editable sticks on every Nth
        inline/crossline so manual editing works as before. Handles crossing and conjugate faults (separate by
        orientation), minimum size, likelihood threshold.
   32d. Fault QC + filtering: stereonet (lower-hemisphere poles, Kamb/density contours) and rose diagram of all
        patches; filter by dip, azimuth, size, likelihood and throw by brushing the stereonet; Fault Merging Assistant
        (suggest merges from proximity + orientation + throw continuity; merge / split / rename in 2D and 3D views);
        bulk delete of small patches; fault table (name, area, mean strike/dip, max throw, horizons cut).
   32e. Throw and displacement: throw along each fault from horizon cut-offs and from cross-correlation slip across
        the fault (Hale fault slip), throw-distribution map on the fault plane (feeds the 27b Allan diagram / SGR),
        throw-distance and throw-depth plots (feeds 26a growth analysis), displacement-length QC flag for unrealistic
        faults.
   32f. Fault mapping on horizons and maps: automatic hanging-wall / footwall cut-off lines and fault (heave)
        polygons per horizon, downthrown ticks, fault-block labels; pushed to the Live Map (31) and used as barriers by
        gridding, the horizon autotracker (item 1) and RGT linking (item 4); a fault-polygon editor on the map.
   32g. 2D and 2D-3D: fault likelihood + auto fault segments on 2D lines; fault-cut correlation assistant that links
        fault cuts between intersecting 2D lines and the 3D into one named fault by position, strike, dip and throw
        sense (Chris's 2D/3D merge workflow); fault polygons from 2D-only data.
   32h. Optional ML: CNN fault probability (FaultSeg-style) inference only if an optional backend (torch / onnxruntime)
        imports, installed by `Install 3D extras (optional).bat`; numpy TFL stays the default and the tested path.
   Tests: synthetic volume with planar normal faults of known strike/dip/throw plus a crossing conjugate pair and noise ->
   TFL peaks within 1 trace of the true planes, strike/dip within 5 deg, one skin per fault (crossing pair separated),
   throw within 1 sample; stereonet filter keeps the right patches; merge assistant joins two halves of one fault and
   not two neighbouring faults; fault polygons close and do not overlap; 2D cut correlation links the right cuts.

37. **Velocity-volume depth conversion (advanced mode of the Kingdom-style DDC, item 17)** - status: todo
   Chris asked for this on 2026-11-25 (project doc "GeoHarmonic Velocity Volume plan item"): build a 3D interval-
   velocity volume from the well-tie synthetics, framed by the interpreted horizons, and use it to convert to depth.
   WELL TOPS ARE THE HARD DATA; THE VELOCITY VOLUME IS SECONDARY (trend). Final depth maps must tie every well top
   exactly. Item 17's layer-cake modes stay unchanged (`velocity.VelocityModel`, `ddc.DynamicDepthModel`); this is an
   additional mode in a new module (e.g. `velvolume.py`). numpy only. Numbered 37 (added after 32-36) but placed here
   so it runs after 31/32 and before the 3D module: item 35c's depth framework consumes it.
   37a. Velocities from the synthetics: per tied well take the final T-D from the well tie (`welltie.WellTD.twt()` incl.
        shift / stretch / DTW warp from 16a-b, saved via `tie.write_td_csv`), falling back to checkshot
        (`welltie.checkshot_for`) then sonic-integrated T-D (`welltie.sonic_td`, `tie.calibrate_sonic`); interval and
        instantaneous velocity vs TWT (`tie.interval_velocity`); QC: drift vs checkshot (`tie.drift_table`), despike
        (`tie.despike`), min/max caps, flag poorly tied wells (tie correlation from `tie.correlate` / TieState score);
        velocities blocked per zone for display.
   37b. Structural framework: time horizons (+ fault polygons where available, `faults.py`) define zones; layering per
        zone proportional / parallel to top / parallel to base - reuse `inversion/lfm.strat_coords` and
        `lfm.horizon_grids` (item 19a); RGT stack levels optional as in 19a.
   37c. Velocity volume: interpolate well velocities along framework layers - IDW (`lfm.lateral_weights`), kriging
        (`vgrid.fit_variogram` / `krige_fit`), or trend-guided with Dix-converted stacking/RMS velocities as the
        background away from wells (`vgrid.dix`, `vgrid.guide_trends`, `vgrid.guided_grid`, item 17b); optional
        smoothing (`inversion/common.butter_filter`); store as a float16 volume (`inversion/common.write_volume`,
        same pattern as `lfm.VolumeLFM.write`) and export SEG-Y (`inversion/common.export_segy`); threaded + Cancel.
   37d. Depth conversion through the volume: integrate trace by trace (z = sum v_int * dt / 2) to depth horizons, depth
        sections and a depth-converted seismic volume (reuse `scenarios.depth_axis` / `depth_convert_volume` /
        `interp_rows` with a volume-driven time->depth function instead of the layer-cake one).
   37e. Well tops as hard data: residual per well and horizon (top TVDSS minus volume-predicted depth); grid residuals
        (`vgrid` kriging or TPS; collocated cokriging with the volume-derived depth as secondary via the item-21c
        cokriging in `maplayers`/`maps` as an option) and apply so every depth map ties the tops exactly; then calibrate
        the volume (scale each layer's interval velocity at the wells to honour the tops, grid the scale factors like
        17b's ratio guide) and write a calibrated velocity volume consistent with the corrected depth maps.
   37f. Dynamic update via 17c (`ddc.DynamicDepthModel.refresh`, `_top_changes`, `_layer_of`): a new/moved top
        recomputes residuals and calibration only for affected layers and below; a new/edited well tie updates only that
        well's velocities and the nearby volume; status-bar message says what changed.
   37g. QC and uncertainty: leave-one-well-out blind test vs the item-17 layer-cake modes (`scenarios.blind_test`,
        `summarise`); residual maps per horizon; velocity sections with wells and tops overlaid; register as named
        scenarios in 17e (`scenarios.save_scenario`, `difference_grid`, `probability_maps` for P10/P50/P90 depth).
   37h. UI: "Velocity volume (advanced)" mode in `gui/velocity_dialog.py` as a guided a-e workflow; velocity volume on
        seismic sections (as an attribute volume), as a map-hub layer (slice / layer average, `maplayers` / `maphub`)
        and through the Live Map bus (item 31a: DataChanged "velocities").
   37i. Tests: synthetic earth with lateral + vertical velocity change and 6-8 wells -> volume mode beats layer-cake in
        the blind test; final maps tie every top within 0.1 m; calibrated volume reproduces the corrected depth maps
        within 1 m; a moved top rebuilds only affected layers; stacking-velocity trend recovers a lateral gradient the
        wells miss. GUI wiring under dev/fakeqt. WHATS_NEW section + test checklist.

33. **3D Viewer (Petrel-style 3D window)** - status: todo
   33a. 3D window core: QOpenGLWidget canvas (orbit / pan / zoom, vertical exaggeration slider, axis box with
        IL/XL/time labels, background + lighting, saved camera views), object tree (visibility, opacity, colour bar per
        object), window state saved in the project. Clear message + 2D fallback if OpenGL context creation fails.
   33b. Seismic slices: inline, crossline and time-slice planes draggable through the volume, read on demand from the
        existing store (inline store / crossline copy from item 15), decimated while dragging, full resolution on
        release; colour tables and clip shared with the 2D viewer; 2D lines as vertical curtains along navigation.
        F3 Netherlands (~651 x 951 x 462, ~1 GB float32) is the test volume - never load it whole.
   33c. Interpretation objects: horizons as triangulated surfaces coloured by time/depth or any attribute; fault
        sticks + fault surfaces (item 2); well paths (vertical + deviated, laterals.py) with logs as tubes/ribbons and
        tops as discs; culture layers and grids draped on a surface or at a chosen level.
   33d. Map bus links: active inline/crossline in the seismic viewer moves the 3D plane and vice versa; Live Map
        selection highlights wells in 3D; DataChanged updates 3D objects immediately; plane positions on the Live Map.
   33e. Output: screenshot PNG with legend; turntable GIF (PIL), MP4 only if ffmpeg exists.
   Tests: slice extraction vs store, surface triangulation (NaN holes, faults), tube mesh along a deviated well,
   camera math; GUI wiring under dev/fakeqt with OpenGL calls mocked.

34. **3D interpretation (picking, autotracking, volume rendering, geobodies)** - status: todo
   34a. Pick horizon points and fault sticks directly on 3D planes (ray-plane intersection); picks go to the same
        horizon/fault objects as the 2D viewer (item 1/2), undo shared.
   34b. Seeded autotracking from the 3D window using the item-1 tracker, in a background worker with progress and
        Cancel; tracked area grows live in 3D and on the Live Map; undo per tracking run.
   34c. Volume rendering: opacity-curve rendering of amplitude / attribute / RGT volumes on a decimated or user-boxed
        sub-volume (3D texture, GPU-memory check); warning on integrated graphics.
   34d. Geobody extraction: threshold an amplitude/attribute volume, numpy connected-component labelling (chunked,
        out-of-core), body list with volume + cell count, display as surfaces (marching cubes in numpy), outlines to
        the Live Map as polygons.
   34e. Inversion (item 19), attribute (item 11) and RGT / stratal-slice (item 5) volumes load as 3D volumes/slices;
        horizon attribute extractions shown on 3D surfaces.
   Tests: ray-plane pick maps to the right il/xl/sample; geobody labelling on a synthetic with two known bodies
   (counts + volumes exact); marching-cubes closed surface for a sphere; tracking from 3D seeds = item-1 result.

35. **Structural framework + 3D grid** - status: todo
   35a. Fault modeling: fault surfaces from sticks and the item-32 auto-extracted faults (triangulated surfaces) with editing (add/remove
        sticks, smooth); fault polygons per horizon from fault-horizon intersections, shared with Live Map gridding.
   35b. Horizon modeling: grid horizons per fault block (reuse item-21/31g gridding with faults as barriers),
        truncate against faults, conformable / erosional rules between horizons, consistency checks (no crossings).
   35c. Framework in time or depth; depth version from the item-17 DDC model, rebuilt when the velocity model
        changes (DataChanged).
   35d. 3D grid: stair-stepped fault grid between depth-converted horizons - areal cell size + rotation, zones between
        horizons, layering per zone (proportional, follow top, follow base, constant thickness), fault-block and zone
        index per cell; quality report (cell volumes, zero/negative-thickness cells). Full corner-point grid with
        complex fault juxtaposition is OUT of scope.
   35e. Output: grid + zones to Eclipse GRDECL (runs in the free OPM Flow simulator); surfaces to ZMAP+/XYZ.
   Tests: synthetic faulted layer-cake -> grid cell counts, zone thicknesses and fault-block throw correct; GRDECL
   round-trip; no crossing horizons after truncation.

36. **Property modeling + volumetrics** - status: todo
   36a. Log upscaling to the cells each well passes through (arithmetic / harmonic / geometric / most-of), using the
        item-20 petrophysics curves; QC plot raw vs upscaled.
   36b. Data analysis per zone: experimental variograms (major / minor / vertical), interactive model fit,
        histograms, trends (reuse item-17b variogram code).
   36c. Facies modeling: sequential indicator simulation + indicator kriging per zone; optional seismic attribute /
        inversion volume as a soft trend.
   36d. Petrophysical modeling: kriging and sequential Gaussian simulation (numpy), conditioned to facies; multiple
        realizations with fixed seeds.
   36e. Volumetrics: GRV, NTG, porosity, Sw, Bo/Bg with OWC/GOC per zone and fault block; P90/P50/P10 from
        realizations; link to item-28 prospect volumetrics; per-block values on the Live Map; WellTable/report output.
   36f. Output: property grids to GRDECL; volumetrics to CSV.
   Tests: upscaling on a hand case; kriging exact at data points; SGS reproduces the input histogram and variogram
   within tolerance; volumetrics vs a hand-computed box reservoir.
   Run end: HANDOFF + WHATS_NEW "3D module" section with step-by-step Windows instructions (open the 3D window, load
   F3, add horizons and wells, build the grid).

Later (Chris back): run `dev/bench.py --gb 5..20` on the PC and fix what it shows (overview volume only if zoomed-out display is slow); Maui 3D subset cut (`--subset in.sgy out.sgy --il a b --xl c d --tmax`) from the 4.1 GB
MAUI-3D PR2301 T-RMIG SEG-Y; mistie integration with the 2D lines; Tui-1/Amokura-1 wells.
