# Build plan - Seismic Interpretation Tool (Nov 23 - Dec 2, 2026, two runs a day, one item per run)

Status values: `todo`, `partial`, `done`, `blocked: <reason>`. Do the first `todo`/`partial` item.
Chris's priorities: both 2D and 3D everywhere; PaleoScan-style relative geologic time (RGT) workflow;
Kingdom-like interaction; large data.

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

19. **Seismic inversion package (post-stack + pre-stack)** - status: todo
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

Later (Chris back): run `dev/bench.py --gb 5..20` on the PC and fix what it shows (overview volume only if zoomed-out display is slow); Maui 3D subset cut (`--subset in.sgy out.sgy --il a b --xl c d --tmax`) from the 4.1 GB
MAUI-3D PR2301 T-RMIG SEG-Y; mistie integration with the 2D lines; Tui-1/Amokura-1 wells.
