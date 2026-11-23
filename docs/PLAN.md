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
   Tests: synthetic data with known linear + nonlinear + interaction effects -> models recover them;
   TreeSHAP values sum to prediction minus expected value (local accuracy) and match brute-force Shapley
   on small trees exactly; KernelSHAP close to exact; GBM beats linear on the nonlinear case; group CV
   gives a worse (honest) score than random CV on spatially correlated synthetic wells.
14. **Hardening** - status: todo
   - Read every module for bugs, edge cases (irregular 3D geometry, missing traces, dead traces, IBM vs
     IEEE, big files > 2 GB on Windows), performance on a 2 GB synthetic, docs/WHATS_NEW.md refresh.

Later (Chris back): Maui 3D subset cut (`--subset in.sgy out.sgy --il a b --xl c d --tmax`) from the 4.1 GB
MAUI-3D PR2301 T-RMIG SEG-Y; mistie integration with the 2D lines; Tui-1/Amokura-1 wells.
