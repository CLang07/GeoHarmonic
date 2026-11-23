# Build plan - Seismic Interpretation Tool (Nov 23 - Dec 2, 2026, two runs a day, one item per run)

Status values: `todo`, `partial`, `done`, `blocked: <reason>`. Do the first `todo`/`partial` item.
Chris's priorities: both 2D and 3D everywhere; PaleoScan-style relative geologic time (RGT) workflow;
Kingdom-like interaction; large data.

1. **Horizon picking + autotracking** - status: todo
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
2. **Fault sticks + fault planes** - status: todo
   - Draw sticks on sections (click points, Enter ends stick, N new fault), assign to named faults,
     display on intersecting sections; triangulated fault surface per fault; horizon autotracker stops at
     faults. Tests: synthetic faulted volume - tracker does not jump across the fault.
3. **RGT step A - dip field** - status: todo
   - `rgt/dip.py`: local inline & crossline dip (ms/trace) from gradient structure tensor with Gaussian
     smoothing (separable numpy convolution), plus coherence/planarity; chunked over the memmap; store as
     float16 volumes. Tests: synthetic dips recovered within 5 %.
4. **RGT step B - mesh + global solve** - status: todo
   - Seed grid of nodes (every N il/xl, every M ms), link neighbours by dip-guided waveform correlation
     (cross-correlation search, confidence weights), fault-aware (no links across fault planes, or links
     allowed with offset); solve relative geologic time for all nodes by sparse weighted least squares
     (write a conjugate-gradient solver in numpy; no scipy) with optional hard constraints from
     interpreted horizons; interpolate RGT to the full volume (store float16 `rgt.npy`).
   - 2D: same on each line, tie lines at intersections. Tests: synthetic with onlap/pinch-out + fault ->
     extracted iso-RGT surfaces match true horizons.
5. **RGT step C - horizon stack, Wheeler, stratal slices** - status: todo
   - Extract N horizons (iso-RGT) as a horizon stack; Wheeler view (seismic resampled to RGT axis,
     hiatuses shown as gaps); stratal-slice viewer (amplitude on any iso-RGT surface, slider through the
     stack) in the section panel as a new mode; convert any stack surface to an editable horizon.
6. **RGT step D - interactive editing** - status: todo
   - User adds/moves control points or horizons as constraints, local re-solve in a window around the edit
     (fast), fault re-linking; undo. Tests: constraint changes the model only locally.
6b. **DLIS reader (numpy, RP66 v1)** - status: todo
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
7. **Wells on seismic** - status: todo
   - Load well project wells; time-depth from checkshots (Maui-5, Maui-6 LAS in data/nz) or interval
     velocity from DT, nearest-checkshot fallback; show wells + tops (in TWT) on sections and basemap;
     synthetic seismogram (RHOB*DT AI, reflectivity, Ricker wavelet with chosen frequency, or statistical
     wavelet from seismic near the well), bulk shift / stretch-squeeze tie with correlation coefficient.
8. **Horizon maps + attributes** - status: todo
   - Time-structure, isochron, amplitude extraction (RMS / max in window), depth conversion with a
     velocity from well ties, contours (reuse marching squares from well_correlation/mapping.py), export
     grids (XYZ, CPS-3 ASCII), PNG.
9. **Write-up** - status: todo
   - `docs/WHATS_NEW.md` for Chris: features, how to use, test checklist ordered by risk, known limits.

10. **2D-3D merge / tie** - status: todo
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
11. **Seismic attributes** - status: todo
   - Volume attributes computed in chunks to new float16 stores, viewable like any volume: envelope,
     instantaneous phase/frequency (Hilbert via numpy FFT), RMS amplitude, coherence/semblance, dip-steered
     variance, curvature on horizons, spectral decomposition (short-window FFT bands). Attribute picker in
     the section panel.
12. **Velocity model + depth conversion** - status: todo
   - Interval/average velocity from well time-depth pairs + horizons (layer-cake), gridded; convert
     horizons and sections to depth; well misfit table.
13. **Analytics (Transform-style + modern data-science workflow)** - status: todo
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
14. **Dashboard with lateral (horizontal-well) attribute maps** - status: todo
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
15. **Scale + hardening** - status: todo
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

Later (Chris back): Maui 3D subset cut (`--subset in.sgy out.sgy --il a b --xl c d --tmax`) from the 4.1 GB
MAUI-3D PR2301 T-RMIG SEG-Y; mistie integration with the 2D lines; Tui-1/Amokura-1 wells.
