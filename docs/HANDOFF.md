# Handoff - architecture, conventions, lessons

## Repo layout
```
tools/Mistie Tool/              mistie_tool        tkinter+matplotlib 2D mistie analysis (NPRA Alaska lines)
tools/Well Correlation Tool/    well_correlation   PySide6+pyqtgraph well log cross sections (v0.4.0)
tools/Seismic Interpretation Tool/ seismic_tool    PySide6+pyqtgraph 2D+3D seismic (v0.1.0, step 1 done)
data/npra/      R_2.SGY, R_3.SGY (USGS NPRA 2D, int16, 4 ms, 1501 samples, CDP byte 21, no coords in headers),
                npra_nav.csv (line,cdp,x,y; lines named "R-2"), NPRA 2D.seisproj.json (R-2, R-3)
data/nz/        Maui field wells (offshore Taranaki NZ): wells/<Well>/las, checkshot (Maui-5, Maui-6 LAS),
                wells_header.csv, NZ Well Correlation.wcproj.json (tops incl. published PR3099 Table 1.1),
                Maui seismic.seisproj.json (empty; links the well project)
                Tui-1 (wells/Tui-1): DLIS only - checkshot/ (CSAT surveys, Suite 1 intermediate + Suite 2
                main), dlis/ (PEX-DSI-HALS repeat pass, intermediate PEX-HRLA-DSI), mudlog/ gas chromatograph
                CSV, samples/ cuttings metadata. Location from log header: NZMG E 2,531,255 N 6,196,747,
                DF 25 m above LAT, water depth 123 m. Pateke-2/checkshot: 4 CSI DLIS files.
tools/QGIS Plugin/geoharmonic_link/   QGIS plugin "GeoHarmonic Link" (21h): core.py (LinkClient, read_project, FileWatcher -
                no QGIS imports), plugin.py (qgis.PyQt / qgis.core / qgis.gui). Zip: python dev/build_qgis_plugin.py
dev/fakeqt/     Qt + pyqtgraph mock for testing GUI code without Qt
dev/fakeqgis/   qgis mock (module __getattr__ -> fakeqt Anything; needs dev/fakeqt on the path)
docs/           HANDOFF (this), PLAN (ordered work), RUNLOG (one entry per run)
```
Chris runs each tool from its folder on Windows: `Install (run once).bat` creates `.venv` from
`%USERPROFILE%\anaconda3\python.exe` with **PySide6==6.5.3 + pyqtgraph==0.13.7** (newer PySide6 fails with
"DLL load failed ... procedure could not be found" against Anaconda's MSVC runtime; keep the pins). Launch
`.bat` sets `PYQTGRAPH_QT_LIB=PySide6`. Bats must have CRLF line endings.

## Seismic Interpretation Tool (focus of the build)
- `segy.py` `SegyFile(path)`: np.memmap; text header (EBCDIC/ASCII, 80-col lines), endian detect,
  formats 1 (IBM) 2 3 5 6 8 10 11 16; `header(byte, nbytes=4, start, stop)`, `traces(start, stop, idx)`
  -> float32 (n, ns); `coordinates(x_byte=181, y_byte=185, scalar_byte=71)`; `guess_il_xl()` tries
  IL_XL_CANDIDATES ((189,193),(9,21),(5,21),(17,21),(13,17),(221,225),(9,13)); `ibm2ieee`.
- `volume.py`
  - `import_3d(segy, store_dir, il_byte, xl_byte, x_byte, y_byte, scalar_byte, chunk, progress)` writes
    `data_ilxl.npy` float16 (nil, nxl, ns), `data_t.npy` float16 (ns, nil, nxl), `filled.npy`, `meta.json`
    (grid il0/il_step/nil/xl0/xl_step/nxl, ns, dt (s), t0, scale = 1000/p99.9 amplitude, affine_x/y fitted
    il/xl->x/y, text_header). 250 MB synthetic imports in ~1 s.
  - `Volume3D(store_dir)`: `inlines, xlines, times` (ms), `il_index/xl_index/t_index`, `xy(il, xl)`,
    `ilxl_at(x, y)`, `outline()`, `inline(il)` (nxl, ns), `xline(xl)` (nil, ns), `time_slice(t_ms)`
    (nil, nxl), `trace(il, xl)`, `arbitrary(xs, ys, spacing)` -> (section, ils, xls). Amplitudes are scaled
    (float16), not original units.
  - `Line2D(name, segy, cdp_byte=21, nav=None)`: `cdp`, `x`, `y`, `has_coords`, `times`, `data()` (ntr, ns)
    cached, `release()`. `read_nav_csv(path)` -> {line: (n,3) [cdp,x,y]}; `match_nav(name, nav)` tolerant
    of R_2 / R-2.
- `project.py` `SeisProject`: JSON `*.seisproj.json`, paths stored relative to the project file.
  `volumes [{name, store, segy, il_byte...}]`, `lines [{name, segy, cdp_byte}]`, `nav_files`,
  `well_project` (a .wcproj.json; `wells()` -> [(name, x, y)] NZTM2000 metres), `display` dict.
  `volume(name)`, `line(name)` cached. Interpretation objects (horizons, faults, RGT) should be added as
  new keys + files in `<project>_seismic/` next to the project, never inside the JSON if large.
- `horizons.py` (v0.2): `Horizon` (name, color, visible, `data` {"3D:<vol>": float32 (nil, nxl) ms,
  "2D:<line>": float32 (ntr,) ms}, NaN = unpicked), `HorizonSet` (items, undo stack of array snapshots,
  save/load to `<project>_seismic/horizons/<safe name>/<safe key>.npy`, JSON list in the project under
  "horizons"), `snap_times` (peak/trough parabolic, zero crossings linear), `correlate_pairs`,
  `track_grid(get_traces, grid_idx, seeds, params, dt_ms, ns, allowed, blocked)` (vectorised wavefront
  tracker with relaxing thresholds + gap bridging; `blocked(pi,pj,ci,cj)` is the hook for faults),
  `TrackParams` (mode, window_ms, max_jump_ms, min_corr, max_gap, smooth), `Picker(project, hset)`
  (pick_points, manual_line, erase, clear_section, track(scope), track_2d_network via
  `line_intersections`), display helpers `section_curve`, `slice_points`, `grid_transform`,
  `structure_rgba`. `synth.py` builds synthetic projects (folded 3D + crossing 2D lines) for tests;
  `test_horizons.py` holds the horizon tests (`run` in --selftest, `run_gui` in --guitest).
- `faults.py` (v0.3): `Fault` (name, color, visible, `sticks` [(n,3) x, y, t ms], `ordered_sticks()` along
  strike by PCA, `triangles()` zipped between consecutive sticks), `FaultSet` (undo with a sequence number
  shared with HorizonSet so Ctrl+Z undoes the latest edit of either; JSON in the project under "faults"),
  `plane_cut` (triangles x plane; a plane through an end stick is cut just inside), section cuts
  `section_segments_3d(fault, vol, mode, number)` / `section_segments_line(fault, x, y)`, sticks on a section,
  `EdgeBarrier` + `barrier_3d` / `barrier_line` -> `blocked(grid_idx)` for `track_grid` (edges crossed by
  the fault cut on every inline/crossline, stored as time ranges). `Picker.track(..., faults=[...])`.
  `synth.faulted_cube` (throw = reflector spacing) + `test_faults.py`.
- `rgt/dip.py` (v0.4): gradient structure tensor dips, closed form p = -<gt gx>/<gt gt> (no eigen solves),
  planarity = 1 - residual/energy; separable `correlate1d`/`gsmooth`/`gradients` (derivative of Gaussian);
  `dip_2d`, `dip_3d_block`, `compute_dip_volume(vol, block, progress, cancel)` (inline blocks with halo ->
  `<store>/rgt/dip_il|dip_xl|planarity.npy` float16 ms/trace), `DipField.section(name, vol, mode, number)`,
  `line_dip(line, folder)` (npz cache). Speed ~3 us/voxel 3D, 0.6 us/sample 2D. `test_dip.py`.
  Panel `attr` / `set_attr` + toolbar combo (Seismic / Inline dip / Crossline dip / Planarity);
  `MainWindow.run_job(label, fn(progress, cancel), done)` = generic threaded job with progress bar.
- `rgt/solve.py` (v0.5): `RGTParams`; `NodeSystem` (unknowns = tau at node columns x node times; `_Eq` COO
  rows; `add_links`, `add_horizon` (extra unknown per horizon), `add_regularisation` (weak vertical rate prior
  + ONE dense datum row mean(tau)=mean(t) - per-node 'tau=t' damping flattened structure on fine grids, see
  lessons), `solve` (column-scaled CGLS, `cgls()`), `node_values` (monotonic)); `dip_path` (Heun integration
  of dip with fault-crossing detection via `EdgeBarrier.hit(axis, line, k, t, extra)`), `refine_corr`,
  `links_along` (shared 2D/3D), `fault_links` (step 2/4/8 traces away from the fault, correlate +/- max
  throw with clean windows), `build_3d` / `solve_3d_multires` (coarse-to-fine start) / `write_rgt_3d`
  (uint16 + rgt.json + rgt_nodes.npz) / `compute_rgt_3d`, `RGTVolume` (decode, trace, section,
  `iso_surface`), `iso_from_traces`, `build_2d` (ties as extra node columns) / `solve_2d_multires` /
  `write_rgt_2d` / `load_rgt_line` / `compute_rgt_2d`. `synth.rgt_geology` (random reflectivity, fold,
  pinch-out wedge, low-angle fault) + `test_rgt.py`.
- `rgt/stack.py` (v0.6): `iso_times(tau (T, ns), levels, times)` (one global searchsorted, fast),
  `build_stack_3d` -> `<store>/rgt/stack.npy` (N, nil, nxl) + stack.json, `HorizonStack3D`, `stratal_slice(vol,
  surface)` (reads only the time slab from data_t), `wheeler(seis, tau, levels, times, gap_frac)` (gap = interval
  < gap_frac x its median thickness), `build_stack_2d` / `load_stack_2d` (`stack_2d.npz`), `stack_to_horizon`.
  Panel: mode "Stratal slice" (number = stack level), `domain` Time/Wheeler (`wheeler_on()`, `_to_wheeler`,
  horizons converted to RGT), picking disabled there. `test_stack.py`.
- `rgt/edit.py` (v0.7): `RGTEditor3D(vol)` (loads rgt_nodes.npz; `_resolve(window, points, horizons, faults)`
  = `build_3d` with sub-window Ipos/Jpos, border columns fixed (weight 100), datum off, point constraints
  bilinear x linear (weight 1e4), CGLS dx_tol 1e-5; `_write_back` re-quantizes only affected inlines in
  place (r+ memmap) + stack rows; `move_level` (segment +/-24 ms shifted, avoids folding), `add_point`,
  `add_horizon`, `relink_fault`, `undo` (window snapshots, shared seq)), `RGTEditor2D` (window on one line).
  `NodeInterp` in solve.py = shared full-res interpolation; write_rgt_3d rounds tau to float32 first so
  edits reproduce untouched traces bit-exactly. Panel: `set_mode('rgt')`, `rgt_click`, `rgt_contours()`,
  `draw_contours`, editors cached in `_rgt_editors` (reset before recomputing dip/RGT). `test_edit.py`.
- `welltie.py` (v0.8): imports the sibling `well_correlation` package (`wc()` adds tools/Well Correlation Tool
  to sys.path); `checkshot_for` (merges all DLIS/CSV checkshots of a well), `sonic_td`, `WellTD` (z_ss -> twt,
  extrapolation, tie shift/stretch about t_centre, `depth()` inverse), `WellSet(seisproject)` (td_for: checkshot >
  sonic anchored to nearest checkshot > default; tops_twt; near_section; trace_at_well; save_tie into
  `project.display["well_ties"]`), `logs_in_time` (AI averaged per output sample), `ricker`,
  `statistical_wavelet`, `synthetic`, `tie_correlation`, `auto_tie`. `SeisProject.wellset()` cached.
  Panel `well_overlay()` / `draw_wells()`; `gui/welltie_dialog.WellTieDialog` (compute/auto/save testable
  without drawing). `test_welltie.py` (synthetic wells + survey with a known 12 ms shift; real Tui-1/Maui-5).
- `tie.py` (v0.19, item 16): `correlate` (FFT xcorr A + Hilbert B per lag, envelope lag, phase = atan2(-B, A) at
  the envelope peak, polarity, cycle-skip = other peak >= 0.85 x best at >= half a dominant period), `best_trace`,
  `auto_tie_fft` (stretch about the window top / anchor, parabolic refine, NRMS, held-out lower 30 %), `TieState`
  (bulk + warp (pairs | dtw) with undo, to/from_dict), `dtw_lags` (Hale strain-limited DTW, b = ceil(1/strain))
  + `dtw_warp` (box smooth then `_lipschitz` clamp so the cap holds exactly), `drift_table`, `calibrate_sonic`,
  `write_td_csv`. `WellTD.warp_in/out` applied after shift/stretch; ties store warp_in/out/kind/pairs;
  `WellSet.calibrated_td`, display["td_mode"]. Dialog methods: correlate, find_best_trace, dtw, pick_event /
  add_pair, undo, clear_warp, drift_qc, export_td, tied_times, tie_corr. `test_tie.py`.
  16c (v0.20): `conv_matrix`, `deterministic_wavelet` (Tikhonov, PEP/NMSE), `multitaper_wavelet` (sine tapers,
  coherence), `zero_phase_of`, `wavelet_phase` (atan2 on w0 / H(w0)), `wavelet_spectrum`, `align_wavelets` /
  `average_wavelet`, `WaveletLibrary` (JSON, resample on get), `despike`, `washout_mask`, `backus`, `vs_castagna`,
  `rc_aki_richards`, `rc_shuey`, `angle_synthetics`, `elastic_in_time`, `seismic_along_wellbore` (td.depth ->
  trajectory x/y -> nearest trace), `batch_tie` / `batch_summary` / `shift_map` / `write_batch_csv`.
  Dialog: estimate_wavelet, save_wavelet, use_library_wavelet, angle_gather, batch, despike_logs / along_wellbore
  flags (`_conditioned` = SimpleNamespace copy of the well). Library: `<project>_seismic/wavelets.json`.
- `maps.py` (v0.9): `MapGrid` (kind "ilxl" on a survey grid or "xy"; node_xy, sample (bilinear),
  to_xy (axis-aligned resample), transform (QTransform), contours via well_correlation.mapping.contour_segments),
  `idw_grid`, `horizon_grid(project, h, key)` (3D grid or gridded 2D picks), `isochron`,
  `extract_amplitude(vol, surface, above, below, kind)` (inline blocks, only the needed time window),
  `well_velocity_points`, `depth_convert`, `write_xyz`, `write_cps3` / `read_cps3`, `export_png`.
  `gui/map_dialog.MapDialog` (compute(**overrides) testable), `Basemap.show_grid`. `test_maps.py`.
- `mistie.py` (v0.10): numpy `analytic`, `apply_correction` (FFT shift + phase + gain), `measure(A, B, dt,
  window, max_lag)` (stacked, envelope lag, phase from analytic inner product; correction of B to match A),
  `extract_3d_along(vol, x, y)` (bilinear), `tie_line_to_volume` (+ along-line profile), `tie_lines` (2D-2D at
  intersections), `all_ties`, `solve_network` (3D fixed, corr^2 weights, Huber IRLS, complex unit-vector phase,
  log gain), `loop_closures`, `solve_flex` (along-line knots, smoothness lam, prior to the constant), `flex_shift_per_trace`,
  `apply_solution` (project.display["mistie"][line] incl. optional "flex" {cdp, shift}; 2D picks shifted by the
  per-trace change). `apply_correction` accepts one shift per row. v0.11: `horizon_ties`, `solve_grouped`
  (groups / fixed key lines by renaming tie ends), `measure_windows` (energy-centroid times), `time_variant_line`
  (iterative, quadratic-in-time model - plain knots oscillate), `apply_time_variant`, `overlap_ties`,
  `set_preferred` (display hidden_lines), `tie_positions`, `write_tie_table`, `export_corrected_segy`;
  `Line2D.time_variant`; `Basemap.show_ties`.
  `Line2D.raw_data()` vs `data()` (corrected), `Line2D.set_correction`; `SeisProject.line()` applies stored
  corrections. `gui/mistie_dialog.MistieDialog`. `test_mistie.py`.
- `attributes.py` (v0.12): `ATTRIBUTES` {name: (needs neighbours, params)}, `envelope`, `inst_phase`,
  `cos_phase`, `inst_freq` (phase advance of z[n+1] conj z[n-1], unbiased), `running_mean`, `rms`, `spectral`
  (Gaussian-windowed complex exponential, constant Q), `semblance` (3x3, optional dip steering by linear
  interpolation), `section_attribute(vol, name, mode, number)` (reads +/-1 inline/xline or a +/-48-sample
  time slab), `line_attribute`, `compute_volume` (inline blocks with halo -> new store + project volume
  "<vol> [<attr>]"), `horizon_curvature(MapGrid, kind, velocity)`. Panel ATTRS includes them. `test_attributes.py`.
- `velocity.py` (v0.13): `VelocityModel(project, key, [(horizon, formation|None)], residual_correction)`:
  `time_grids`, `well_table` (vint per layer at wells), `build` (IDW of vint, cumulative depth, optional residual
  grid), `depth_grid(k)`, `velocity_grid(k)`, `misfit_table`, `depth_of_time(x, y, times)`, `section_to_depth`,
  `to_dict`/`from_dict` (stored in project.display["velocity_model"]). Panel domain "Depth" (`depth_on`,
  `_to_depth`, generic `_ymap` used for horizon conversion in Wheeler and Depth). `gui/velocity_dialog.py`,
  `test_velocity.py`.
  v0.22 (17a): per-layer `functions` [{func v0|v0kz|v0kt|td, k (None = fit), blend idw|nearest}] (saved only when not
  all v0); helpers `kz_base/kz_v0` (exp form, `_ek` k->0 limit), `kt_base/kt_v0`, `fit_k_kz` (ek^2-weighted V0
  variance, scan + golden), `fit_k_kt` (linear LS), `fit_line` (2-sigma outliers); `layer_info`, `crossplot(k)`,
  `v0_grid(k)`; grids dict has "v0", "layers", "td", "px", "py". `depth_of_time` is vectorised: horizon times hit the
  (residual-corrected) depth maps, inside a layer the function's shape is scaled to them. `test_ddc.py` (ddc_case:
  6 wells, V0+kZ / V0+kT earths). Dialog: function + k columns, set_layer_function, functions_from_table(names),
  crossplot / show_crossplot, info_text.
  v0.23 (17c): `VelocityModel.build(from_layer, reuse, time_grids, wells)` = incremental (layers above copied; per
  layer `sig` of the gridding inputs -> unchanged V / V0 grids reused); grids keep `z_raw` (before residual
  correction); rows may carry `from_layer` / `pseudo`; `well_row(name, tg)`. `ddc.py` `DynamicDepthModel(project,
  dict|VelocityModel)`: build, refresh(force) (horizon blake2b fingerprints, wcproj mtime -> Project.open ->
  per-well/formation top diff, wells added/removed -> layer 0, display["ddc_what_if"] diff), add_what_if /
  remove_what_if / save_what_if, report {changes, from_layer, layers, seconds, message}. Panel: `depth_model()`
  (cache of DynamicDepthModel), `adopt_depth_model`, `ddc_refresh`. MainWindow: `set_ddc`, `toggle_ddc`, `ddc_tick`,
  QFileSystemWatcher + 1 s debounce + 2 s poll; `_horizons_changed` starts the debounce. Dialog: `ddm`,
  `model_refreshed`, `add_what_if`, `clear_what_if`. Tests: `test_ddc.run_dynamic`, `run_gui_dynamic`.
  v0.24 (17b): `vgrid.py` (`grid_values(X, Y, px, py, v, method idw|tps|kriging, opts)`, `tps_fit/eval` in a centred
  scaled frame, `fit_variogram` (binned experimental, LS sill per trial range), `krige_fit/eval` (dual form, simple
  kriging), `sample_volume` (nearest trace, one inline read at a time), `dix`, `guide_trends`, `guided_grid`).
  `VelocityModel(gridding={method, variogram, range, smooth})`, functions[k]["guide"] = {volume, kind rms|interval,
  mode ratio|difference}; trends cached in grids["trends"] keyed by `trend_keys` (horizon arrays); `trend_grid(k)`.
  Dialog: set_gridding, set_layer_guide, column 5 guide combo. Tests: `run_gridding`, `run_gui_gridding`.
  v0.25 (17d): `prognosis.py` `Prognosis(dm, trajectory, uncertainty)` (trajectory resampled 2 m; `_crossings(k)` =
  sign changes of TVDSS - map depth, projection below TD along the last segment), compute / update (change vs
  previous) / set_bit / add_top (what-if "drilling <well>: <key>" at the wellbore point) / clear_tops / write_csv;
  `vertical_trajectory`. `gui/prognosis_dialog.PrognosisDialog(project, depth_model_fn, parent, on_change)`;
  MainWindow.prognosis, ddc_tick updates it. Tests: `run_prognosis`, `run_gui_prognosis`.
  v0.26 (17e): `VelocityModel(exclude=[wells])`, `velocity.PointGrid` (build at scattered points; `_grid_like`).
  `scenarios.py`: save/load/delete_scenario (display["velocity_scenarios"]), difference_grid, blind_test (LOO at
  points: test well + training wells), summarise (mean/std/rms/p10/p90, sigma = a + b d), sigma_by_horizon,
  probability_maps (P10 shallow / P90 deep, 1.2816 sigma), interp_rows (row-wise interp, one searchsorted, per-row
  queries allowed), depth_axis, depth_convert_volume (streams inlines to <path>.part, cancel -> removed; dz in mm),
  depth_to_time_grid. display["ddc_blind"] {survey, sigma, summary} -> PrognosisDialog sigma column. Dialog:
  save_scenario, difference, run_blind_test, probability_map, depth_convert, depth_grid_to_time.
  Tests: `run_scenarios`, `run_gui_scenarios`.
- `basin/` (item 18, v0.27-0.31): `chrono.py` (AgeTable in display["basin"]["ages"]: entries {name, horizon, age,
  gap, eroded, role, lith{sand,silt,shale,carbonate,coal}} youngest first, TARANAKI_TEMPLATE, `horizon_rgt`,
  `anchors_from`, `AgeModel` (piecewise-linear age(tau), duplicated abscissa = jump at unconformities), `level_ages`,
  `wheeler_age` (levels -> age axis, blank = hiatus / gaps), `gap_labels`, `section_age_model`, `zone_lithology` /
  `apply_lithology`); `burial.py` (LITHOLOGY table, `mix`, Athy `solid_thickness` / `base_for` (Newton),
  `column_units` (Overburden unit always present when the top surface is older than 0 Ma; ghost units for erosion),
  `backstrip` (batched over columns, irreversible compaction via zmax, offset below erosion), `BurialHistory` (one
  column), `BurialBatch` (B columns, same units), `well_depths`, `velocity_model_depths`); `thermal.py` (Lagrangian
  nodes n_sub per unit + basement slab, implicit FV, batched `thomas`, `solve(hists|BurialBatch)` -> T, z, phi;
  `ThermalHistory`, `fit_heat_flow` by superposition, McKenzie, Horner, `las_temperatures` / `dlis_temperatures` /
  `well_temperatures` / `screen_temperatures`); `maturity.py` (EASY%Ro, TTI, TR kinetics, `crossing_age`,
  `window_timing`); `pseudowells.py` (`run_columns` chunked batch, `map_run` (stride + bilinear upsample,
  fingerprint for live update), `material_history` (sediment particle inside a growing unit), `wheeler_property`,
  `to_traces`, `burial_section`, `event_chart` / `plot_event_chart`); `model.py` (settings in
  display["basin"]["settings"], well model / calibration, `section_property` for the panel).
  Panel: domain "Wheeler (age)" (`age_on`, `_to_age`, `section_horizon_times`, `age_labels`, `basin_colour`
  {prop, t_eval}). `gui/basin_dialog.BasinDialog` (set_entry, use_template, lithology_from_wells, save, run_well,
  calibrate(well, obs), run_map(surface, stride, threaded), show_map, export_cps3, depth_changed (MainWindow.ddc_tick),
  colour_wheeler, event_chart). Tests: `test_basin.py` (run_chrono/burial/thermal/maps, run_gui_chrono/basin).
- `inversion/` (item 19, v0.32-0.35): `common.py` (`butterworth_response` / `butter_filter` (FFT, zero phase, mirror
  pad), `amplitude_spectrum`, `pick_lowcut`, `write_volume` / `write_volumes` (several stores from one pass, scaled
  float16 like attribute volumes, registered '<vol> [<name>]', cancel removes them), `well_elastic` (vp/vs/rho/ai on
  the seismic axis with td.twt = tied time), `export_segy`); `lfm.py` (WellCurve, `fill_grid`, `strat_coords` (zone +
  u per sample), `lateral_weights` (IDW / simple kriging, linear in the values), `LFM` (ln-domain, per-zone interp of
  each well along u, global ln-trend fallback, optional trend ratio), `lowpass`, `well_curves`, `horizon_grids`,
  `VolumeLFM` (horizons or RGT coordinates, traces / compute / write)); `poststack.py` (`ConvOp` 'same' FFT
  convolution + adjoint, D / DT (r = half ln difference, lower sample), `synthetic_ln`, `scale_wavelet`,
  `invert_traces` (CG over a trace block, IRLS for L1, lateral smoothing), `residual_nrms`, coloured inversion
  (`well_spectrum_slope`, `coloured_operator` smoothed over 10 Hz, `ci_scale`), `band_error`, `PostStack`,
  `Coloured`, `blind_well_test`); `prestack.py` (`fatti_coeffs`, `fit_background`, `invert_prestack`, `derived`,
  `PreStack` -> 6 volumes). Panel: module `_positive_volume` -> rainbow percentile levels for meta["inversion"]
  kinds lfm / model-based / simultaneous. `gui/inversion_dialog.InversionDialog` (load_curves, auto_fc, build_lfm,
  set_wavelet, run_poststack, blind_test, qc_at_well, add_stack / run_prestack, crossplot (volume or 'log: GR'),
  export_segy; `run_job_fn=None` runs inline). Tests: `test_inversion.py` (earth(): known elastic earth on a time grid,
  run_lfm / run_poststack / run_prestack, run_gui with LAS wells + checkshots + angle stacks).
- `analytics/` (13a): `table.AttributeTable` (ids, cols, xy; add, matrix, subset, join by `normalise_id`
  (API-14 -> API-10, names), from_csv/to_csv, from_maps_at_wells, from_map_nodes, from_zone_stats (Well
  Correlation zones)), `engineering` (13a+: ALIASES/PRESETS column mapping + prefix fallback, unit detection
  lb/t, bbl/m3/gal, ft/m; completion_table, spacing_features (nn spacing, child flag), qc_flags, monthly_series,
  production_features (cums, BOE, IP30/90, GOR, WC, per-1000 ft, Arps fit + EUR), fracfocus_estimates,
  allocate_lease), `synthetic.make` (known geology + completion effects; also dev/make_synthetic_engineering.py),
  `models` (13b: LinearModel/ridge, Lasso CD, KNN, Binner + shared histogram `build_tree` (XGBoost gain, L1/L2,
  gamma, min_child_weight, NaN default direction, monotone constraints) used by DecisionTree / RandomForest /
  GradientBoosting (squared / logistic, subsample, colsample, early stopping, gain importance), MLP (Adam),
  KMeans++, GaussianMixture (diag EM), SOM, silhouette, elbow, make_model with optional xgboost/sklearn),
  `validation` (r2/rmse/mae/accuracy/f1/confusion, kfold, group_kfold, leave_one_group_out, spatial_blocks with
  buffer, cross_val_predict/score, grid_search, nested_cv, learning_curve). `test_models.py`.
  `explain` (13c: permutation / grouped permutation importance, exact path-dependent TreeSHAP (Lundberg Alg. 2;
  tested == brute-force Shapley), tree_expectation, brute_force_shap, kernel_shap, partial_dependence + ICE,
  ensemble_map, save_model/load_model JSON, matplotlib plots, html_report), `check_ml` (--check-ml).
  `gui/analytics_window.AnalyticsWindow` (add_grid, add_csv, add_zone, build_table, explore, train, predict_map,
  report, qc (Model QC button), predict_map adds `g.inside` applicability mask + `g.outside_fraction`).
  `Install analytics extras (optional).bat`. `test_explain.py` (run + run_gui).
  `qc` (13d: blind_split, temporal_split, repeated_cv_scores, idw_loo, baselines, y_randomization,
  noise_feature_test, residual_diagnostics (QQ), variogram, morans_i (z under randomisation),
  conformal_intervals (split-conformal by well groups), quantile_intervals (GBM objective='quantile'),
  coverage, calibration (reliability bins + Brier), applicability (Mahalanobis + kNN distance),
  importance_stability, hyperparameter_sensitivity, pdp_direction_checks, paired_comparison, qc_report).
  `test_qc.py` (run + run_gui). Window QC degrades to 'n/a' per check when wells are too few.
  `gui/engineering_dialog.EngineeringImportDialog` (preset, completions/production auto-detect, per-column
  combos + unit combos, save/load mapping JSON {kind, mapping, units}; logic in plain attributes: load,
  set_preset, set_kind, set_column, set_unit, problems, build, summary, apply_saved). AnalyticsWindow.add_table /
  lateral_lengths; build_table uses the first imported table as rows when there is no well project.
  Tests: `test_engineering.run_gui`.
  `prep` (missing_report, impute median/kNN, Scaler, outlier_flags (MAD), correlation
  Pearson/Spearman, vif, pca, regression_line). `test_analytics.py`.
- `laterals.py` (14a): min_curvature, Trajectory (from_survey, resample, lateral_mask/length, heel_toe),
  synthetic_survey/synthetic_laterals, read_survey_csv, read_stages_csv, sample_grid, sample_volume (nearest
  trace, twt_of_z callable), sample_log, stage_values, zone_along, distance_to_surface, lateral_stats,
  stats_table (-> AttributeTable for analytics), color_bins / batch_color_bins (NaN-separated polylines per
  colour bin: one PlotCurveItem(connect='finite') per bin), lod_indices, colormap_rgb. `dashboard_model.py`:
  DashboardModel (filters range/in, selection replace/add/toggle/remove, select_box/lasso/hist_bin/ids,
  subscribe(kind), histogram, crossplot, table_rows, summary, save/load_layout JSON, export_html).
  `laterals.trajectories_from_wellset` (survey via well_correlation.geo, else vertical; MD ref = elev).
  `gui/dashboard_window.DashboardWindow` (QMainWindow with docks; set_trajectories, add_grid, load_stages,
  color_sources/values_for/color_by, add_lateral_stats, send_to_analytics, map_batches, filter_from_region,
  nearest_well, export_html; model events data|filter|selection; selection does not rebuild the table).
  `laterals.wellbore_section`, `strip_data`; `gui/well_strip.WellStrip.show_data`; dashboard `well_logs`,
  `log_names` (<= 300 wells), `strip_for`, `open_strip`.
  Tests: `test_laterals.py` (run + run_gui).
- `volume.py` stores (v0.21.0): data_ilxl + data_t + optional data_xl (nxl, nil, ns; `Volume3D.xldata`, used by
  `xline()` when present; `build_xline_store(store)`, CLI `--xlstore`, menu 'Fast crosslines'); import writes sorted
  chunks as one contiguous slab (`flat[k0:k1]`), scattered write only for unsorted/irregular traces.
- `maplayers.py` (21a, v0.36): `LayerStack` (layers = dicts {id, kind, name, visible, opacity, z, src, style}; kinds
  outline3d, lines2d, wells, faults, horizon, grid, contours, polygons, text, petro; add / add_grid (npz in
  `<project>_seismic/maps/<id>.npz` via save_grid / load_grid, + contour layer) / remove (drops its contours) / move /
  set_visible / set_opacity / set_style / grid(lid) / colorbar / legend (visible only, top first); saved in
  display["map_layers"], `LayerStack.load(project)` -> saved stack or `default_stack` = legacy basemap content).
  `render(stack)` -> Qt-free primitives ("image", lid, rgba, transform6) / ("curve", lid, xs, ys, rgba, width, connect)
  / ("points", ...) / ("text", ...) / ("error", lid, msg); `render_png` (matplotlib, also the visual check);
  `petro_values(project, prop, zone)` reads `<wcproj>_petro/summary.csv` (all zones: sum of thickness-type values, mean
  of averages). Basemap layer mode: `Basemap.stack` + `draw_layers()` (z -100+order, under the old overlays: horizon
  map, show_grid, ties, current line), `_lines_for_clicks`; MainWindow sets `basemap.stack = LayerStack.load(p)` on
  open; dock "Map layers" = `gui/layers_dock.LayersPanel` (set_visible, move, remove, set_opacity, set_style (contour
  interval goes to the grid's contour layer), add_current_grid, add_cps3, add_horizon, add_petro, add_polygon,
  add_text, add_pies, start_polygon / polygon_drawn (polygon layer + wells inside selected), crossplot; `select(lid)` +
  `_sel` because the Qt mock has no current row). Tests `test_maplayers.py` (run, run_hub, run_gui, run_gui_hub).
  `maphub.py` (21b, v0.37): `petro_folder(project)` (= <wcproj>_petro), `petro_pies` (mm_summary.csv written by
  well_correlation multimin; all zones weighted by Gross_TVD), `pie_polygons`, `COMPONENT_COLORS`, `read_hub_request` /
  `apply_hub_request` (one petro layer with src.from_request, updated in place), `point_in_polygon`,
  `wells_in_polygon`, `zone_values`, `petro_crossplot_data` (prop_x may be "grid" = basemap grid at the wells),
  `petro_strip_data` (opens the wcproj once, cached as project._hub_wcp). Render primitive ("polygon", lid, xs, ys,
  fill, edge) for pies and polygon fills (QGraphicsPolygonItem on the basemap). Basemap: signals wellClicked /
  polygonDrawn, start_polygon / add_polygon_point / finish_polygon (double-click), set_selected_wells (magenta rings),
  well_at. MainWindow: well_clicked -> `gui/petro_strip.PetroStripDialog` (when the well has summary rows or the strip
  is open; the click still opens the inline too), hub_tick (3 s QTimer poll of hub_request.json mtime; the first tick
  after opening only records the mtime). `gui/petro_strip.PetroCrossplotDialog` (compute / redraw).
- `propgrid.py` (21c, v0.38): `target_grid` (secondary geometry or XY box, <= 250k nodes), `sample_grid` (vectorised
  bilinear, ilxl via vol.ilxl_at arrays), `ked_fit` / `ked_eval` (universal kriging, drift [1, s], residual variogram),
  `cck_estimate` (simple collocated cokriging MM1 in closed form per node; secondary standardised with its stats at the
  wells so rho = 1 reproduces the secondary exactly; rho = 0 = vgrid simple kriging), `estimate(method, ...)`,
  `grid_property`, `correlation`, `cross_validate` (LOO per method), `residual_map`, `calc(expr, {name: MapGrid})` (ast
  whitelist; first grid in source order sets the geometry), `mask_polygon`. `gui/propgrid_dialog.PropGridDialog(stack,
  panel)` (compute / cross_validate / add_layer / add_residual_layer) and `GridCalcDialog` (letters A, B ... = grid
  layers; evaluate, mask); LayersPanel Add... menu entries. Tests `test_propgrid.py`.
- `addwells.py` (21d, v0.39): `add_wells(project, stack, folder, tie, change_layers)` -> report {added, steps, residuals
  (old recipe-grid value at each new well vs its new petro value), regridded (max / rms change, optional change layer),
  ties, errors, seconds}; `property_layers` (grid layers with src.recipe {prop, zone, method, secondary, cell}, written
  by PropGridDialog.add_layer), `regrid_layer`, `report_text`. Uses well_correlation `Project.add_from_folder` (new in
  wc v0.10) + petro interpret_project; tie via tie.batch_tie on the first volume; DynamicDepthModel rebuilt when
  display["velocity_model"] exists; LFM volumes only flagged. LayersPanel.add_new_wells / _add_wells_dialog. Tests
  `test_addwells.py` (run, run_gui).
- `crosscheck.py` (21e, v0.40): `crosscheck(stack, prop, zone, grid_lids)` (multiple linear regression of the zone value
  on the grid layers sampled at the wells; coef, r, sigma, exact LOO residuals via leverages, AtA_inv), `predicted_maps`
  (P50, P90 / P10 = P50 -/+ 1.2816 x prediction SE, residual IDW, corrected = P50 + residual; first grid's geometry),
  `to_attribute_table` (AttributeTable for AnalyticsWindow.add_table). `gui/propgrid_dialog.CrossCheckDialog` (compute,
  add_maps, send_to_analytics via `LayersPanel.analytics_hook`, set by MainWindow to analytics().add_table). Tests
  `test_crosscheck.py`.
- 21f in the Seismic tool (v0.41): maplayers styles - grid ramp (carto) + hillshade / ve / shade_strength + blank
  (polygon layer ids or True = fault polygon layers) via `display_grid` / `blank_polygons_grid` (NaN inside + within 0.75
  cell of the edges => no contour crosses); contours labels (text primitives with an 8th element = angle); polygons style
  fault (hatched polygon primitive {"hatch": True} + ticks on edges facing "down" compass / azimuth); wells style symbols
  (`well_status` from the wcproj header info Status / Result / Purpose -> `QT_WELL_SYMBOLS`). `mapexport.py`:
  `page_size_in`, `layout_boxes` (map + title / legend / scale / north panel), `boxes_overlap`, `apply_preset` (sets
  display["map_style"]), `export_map(stack, path, title, subtitle, author, page, dpi, crs, ref_latlon, lon0,
  contour_interval, symbols)` (exact industry well symbols via carto.symbol_paths, legend = stack.legend()).
  LayersPanel: preset combo, Apply style, Export map.
- `gis/` (21g, v0.42; numpy + struct / zlib / sqlite3 / xml only): `crs.py` (crs dicts {kind geographic / tm / lcc /
  albers / merc, a, rf, unit (m per unit), lat0, lon0, k0, fe, fn (in the unit), lat1, lat2, datum, towgs84}; Snyder
  formulas, TM inverse = series + 3 Newton steps (round trip exact far from the CM); forward / inverse / transform
  (3-parameter datum shift via ECEF, NAD27 flagged approximate); from_epsg (4326 4269 4267 4167 2193 3857 5070 32139
  2278 32040 UTM 326xx 327xx 269xx 267xx), from_wkt (OGC / ESRI), to_wkt (ESRI .prj), same). Verified against Snyder
  LCC + Albers examples, EPSG OSGB TM and NAD27 Texas South Central examples. `shapefile.py` (read / write shp shx dbf
  prj; Point / MultiPoint / LineString / Polygon (CW outer), ring_roles / group_rings), `geotiff.py` (write float32
  strips, deflate, ModelPixelScale / Tiepoint, GeoKeys EPSG else .aux.xml WKT, overview IFDs (NewSubfileType 1); read
  strips / tiles, none / deflate, predictor 2), `gpkg.py` (write_layer / read_layer / list_layers / has_rtree,
  GeoPackageBinary + WKB, rtree when SQLite has it; write_kml / read_kml), `vector.py` (VectorLayer: load .shp / .gpkg /
  .kml / .csv (WKT or X,Y), reproject, select(expr) ast-safe filter, categorized / graduated, simplified (iterative
  Douglas-Peucker, cached), in_view (grid buckets), identify, npz cache; `layer_of(stack, lay)` reprojects to the
  project CRS; `vector_primitives`), `plss.py` (parse_legal, find_section (CadNSDI field guesses), spot with footage
  calls, synthetic_township), `export.py` (project_crs / set_project_crs (display crs_epsg | crs_wkt), layer_features,
  export_stack -> GeoTIFFs + shapefiles + one GeoPackage + KML). maplayers kind "vector". mapexport: layout_boxes has a
  locator box; export_map(graticule, locator, source, neatline), graticule_lines, save_template / load_template
  (display["map_templates"]). LayersPanel: add_vector, set_crs, export_gis, goto_legal, identify (Basemap.mapClicked),
  attribute_rows. Tests `test_gis.py` (run, run_export, run_gui).
- `qgislink.py` (21h, v0.43): `shared_paths` (<project>_gis.gpkg + <project>_gis_grids/), `write_shared(project, stack)`
  (wells + status + petro columns, tops (vertical approx.), faults + convex-hull footprints, every visible layer via
  gis.export.layer_features, grids as GeoTIFF with overviews; editable tables prospects_edit / leases / section_lines kept
  across rewrites; default styles in the GeoPackage `layer_styles` table (QML, useAsDefault); gh_meta + gh_rasters),
  `read_user_edits(project, last hashes)` / `apply_user_edits(stack, changed)` (polygons tagged from gpkg | message),
  `LinkEndpoint(role, port)` (JSON lines over 127.0.0.1, inbox queue, poll / send / wait, never raises), `handle(msg,
  app)`. MainWindow: View > Live link with QGIS (`toggle_qgis`, 200 ms `qgis_tick`: messages + edits (first read = baseline
  only)), Send map to QGIS (`send_map_to_qgis`), `send_current_section` on positionChanged, basemap.wellsSelected ->
  select_wells (set_selected_wells(emit=False) for selections that came from QGIS), `open_arbitrary` ->
  `gui/arbitrary_dialog.ArbitraryDialog` (Volume3D.arbitrary). Tests `test_qgislink.py` (run: GeoPackage, edits, plugin
  core, two-process protocol; run_gui: plugin under dev/fakeqgis + main window server).
- `blend.py` (23a/b, v0.45): `normalize` (percentile clip, gamma), `channels_to_rgb` / `rgb_blend` (modes rgb additive, cmy
  subtractive on white, hsv: ch1 hue * 0.8 (no wrap), ch2 saturation, ch3 value), `hsv_to_rgb`, `legend_triangle`
  (barycentric, max-normalised; hsv = hue x value), `channel_swatches`, `lut_rgba`, `overlay` (alpha blend + threshold),
  `gray_rgba`, `illuminate` (single-light Lambert, azimuth from north, rows = +y; plane = analytic), `shade_rgba`,
  `amplitude_on_structure`, `on_grid` (MapGrid resampled onto another's nodes, vectorised), `section_blend`,
  `spectral_blend`, `bump_section`, `param_scan`. maplayers: kind "blend" (src grids [lid..]), grid style shade_from /
  azimuth / elevation. LayersPanel add_blend / shade_by / blend_legend. SectionPanel: `blend` spec + `set_blend`,
  `blend_image` (rgba drawn instead of the LUT image; _apply_colors skips LUT), `scan_frames` / `show_frame`
  (gui/scan_dialog.ScanDialog), `cursorMoved` signal, `linked_position` / `show_linked_cursor`.
- `linking.py` (23c): `CursorBus` (publish / subscribe, no echo to the sender, throttle min_interval, enabled, clear;
  broken subscribers ignored), `nearest_well`, layouts in display["layouts"] (`save_layout`, `load_layout`,
  `delete_layout`, `layout_names`, `TEMPLATES`). MainWindow.cursor_bus: section -> "map" (Basemap.show_cursor),
  "dashboard" (DashboardWindow.show_cursor), "petro_strip" (PetroStripDialog.show_cursor, follow flag); basemap
  cursorMoved -> section. MainWindow layout_state / apply_layout (base64 saveState when available) / save_layout /
  restore_layout. Tests `test_blend.py` (run incl. run_hub, run_gui).
- `resolution.py` (24a/b, v0.46): `spectrum_stats` (avg spectrum, noise floor from a noise window or the top 20 % of
  the band, usable band = contiguous S/N >= 2 around the peak, f_dom / centroid / octaves), `frequency_map` (MapGrid per
  block), `tuning_thickness`, `wavelet_from_spectrum`, `band_wavelet`, `wedge_model` (exact interface times; 30 Hz Ricker
  tuning = Kallweit-Wood 1/(2.31 fp)), `bed_thicknesses`, `spectral_balance`, `recommend` (needed f for lambda/4 = P50 bed,
  tuning now vs balanced from wedges, % beds resolved, verdict). `rotate_phase`, `stretch`, `single_well_inversion`
  (MC over wavelet phase / frequency / scale, LFM cut-off, noise -> P10/P50/P90 AI, coverage of the band-limited log,
  corr_inv (seismic-band detail), corr_full vs corr_lfm, optional property via the well AI regression, verdict YES if
  corr_inv >= 0.6 and coverage >= 0.5). InversionDialog.single_well / resolution (buttons on the QC row).
- `wellseeds.py` (24d): `synthetic_events` (|syn| maxima >= min_rel (0.5: Ricker side lobes are 0.45), max_n),
  `snap` (nearest extreme of a sign, parabolic), `seed_patch` (ring-by-ring dip-following snap), `seed_horizons`
  ({name: [(x, y, t[, sign])]} -> SeedHorizon with .get(key)), `well_seeds(project, vol, source tops | synthetic,
  add_to_project)` (synthetic: log ends trimmed by half a wavelet). MainWindow.well_seeds (Interpret menu): horizons
  "seed: ..." -> Compute RGT uses visible horizons as control. Synthetic check: RGT spread at 5 wells across a fault
  with no fault links 23 -> 0.7.
- `predictive.py` (24c, v0.47): `Scenario` (ranges h / phi / vsh, fluids + priors, shale, T / P / salinity, phic,
  frame_exp (dry = mineral (1 - phi/phic)^frame_exp, default 2 = soft Tertiary)), `sand_elastic` (VRH quartz-clay,
  Batzle-Wang via well_correlation.petro.rockphysics, Wood mix, Gassmann), `aki_richards`, `shuey_ab`, `interface_trace`,
  `pseudo_well`, `pw_traces`, `simulate` (features at the sand-top sample: near, far, A, B; plus AI, Vp/Vs),
  `wedge_section`, `BayesClassifier` (Gaussian full covariance + priors; to_dict), `classify_stacks` (min_amp ->
  NaN), `classify_volumes` (write_volumes '<near> [P(fluid)]'), `probability_map` (max in a window around a horizon),
  `stats_table`. InversionDialog.predictive (Pre-stack tab button; nearest / farthest stack; min_amp default 0.3 x
  median |near| of the simulation). Tests `test_resolution.py` (run, run_seeds, run_predictive) + test_inversion GUI.
- `avo.py` (25, v0.48): `zoeppritz_pp` (exact 4x4, complex past critical), `avo_class` (I / IIp / II / III / IV with
  a_small 0.02 and theta_max), `ig_from_gather`, `background_trend` (LS with 2-sd trimming), `ig_anomaly` (signed
  perpendicular distance in sd), `fluid_factor` (Rp - 1.16 Vs/Vp Rs, Rs from A, B), `interface_avo`, `scenario_avo`
  (Gassmann fluids), `ar3_rows`, `stack_vs_gather_risk` (Bayesian linear AR3 with prior_sd, stacks = mean rows with
  noise / sqrt(fold); MC check), `fizz_assessment` (density likelihoods gas vs fizz with rho_err, separation, reasons),
  `lf_shadow` (12 / 30 Hz spectral ratio below vs above + centroid drop), `DHI_CHECKS` (weights, computed flag),
  `amplitude_score`, `conformance_score` (best Jaccard of anomaly vs shallower-than-level), `avo_score`,
  `stack_consistency_score`, `dhi_index` (renormalised weights), `assess_prospect`. InversionDialog.avo_volumes
  (Intercept / Gradient / AVO anomaly / Fluid factor volumes, trend from the stronger half of the samples), stack_risk;
  LayersPanel.dhi (display["dhi"][name]). Tests `test_avo.py`, test_inversion GUI, test_blend GUI.
- `structure.py` (26, v0.49): PHASES / PHASE_COLORS, `set_phase` (AgeTable entry key "phase" = unit below the
  surface), `phase_intervals`, `phase_of_age`, `phase_colors_for_axis`; `throw_across`, `expansion_index`,
  `propose_phases` (EI >= 1.1 -> syn-rift, older pre-, younger post-rift), `fault_growth(grids, fault_xl(i))`;
  `flatten`, `paleo_depths` (Athy decompaction with basin.burial solid_thickness / base_for; grains conserved),
  `paleo_frames` (flatten on each horizon oldest first, younger data blanked), `frames_to_rgba`, `export_gif` (PIL);
  `cutoff_throw` (HW / FW cut-offs extrapolated linearly to the fault), `restore_fault` (vertical fault at trace j,
  one step per horizon), `unfold_vertical_shear`, `line_length`, `area_between`. SectionPanel.section_horizons;
  MainWindow.paleo_animation (Interpret menu). Tests `test_structure.py` (run, run_gui).
- `migration.py` (27, v0.50): `receivers` (D8 steepest up-dip, -1 crest, -2 NaN), `crest_of`, `flow_path`,
  `spill_analysis(depth, cell_area)` (priority flood: nodes by depth, union-find basins keyed by their crest; a closed
  basin spills at the first saddle into the other basin's crest or leaks at the edge; closure mask flood fill,
  capacity = sum(spill - z) x cell area), `fill_spill` (deepest crest first, excess to 'to'), `drainage_charge`,
  `critical_moment` (share of TR after trap age), `juxtaposition`, `sgr` (mean Vsh over [z, z + throw]),
  `sgr_column` (AFPD = 10^(SGR%/27 - C), C 0.5 / 0.25 / 0 by depth), `stress_tendency` (slip, dilation; ENU,
  strike right-hand), `micp_to_reservoir`, `column_height`, `VSH_PCE` + `top_seal_column`, `limiting_column`.
  LayersPanel.migration (drainage-area grid layer + trap text labels). Tests `test_migration.py` (run, run_gui).
- `prospect.py` (28, v0.51): `lognormal_from_p90_p10` / `lognormal_stats`, `sample` (constant, lognormal (P90,
  P10), normal (clipped), uniform, triangular, sample arrays), `pvalues` (P90 = 10th percentile, Swanson), `grv` (top
  to contact, constant thickness or base grid, mask), `area_depth`, `grv_from_area_depth`, `volumetrics` (sm3 + bbl /
  scf), `contact_depth` (migration.limiting_column), FACTORS, `chance_of_success`, `prefill_factors`, `dhi_revision`
  (Bayes, LR = exp(4 q (index - 0.3))), `portfolio` (shared factors), ROSE_COLUMNS + `export_csv`, `prospect_row`.
  LayersPanel.prospect (trap number from the spill analysis -> closure mask, contact, GRV, MC, display["prospects"]).
  Tests `test_prospect.py` + test_migration GUI.
- `spacing.py` (29, v0.52): ALIASES + `read_public_csv` (RRC / NDIC / FracFocus -> api, name, operator, first_prod
  ISO, lat / lon / x / y, lateral, proppant, fluid, tvd, zone; header match ignores case / spaces / underscores),
  `synthetic_field` (pads, wine-rack zones, parents then children), `shmax_from_wsm` (axial weighted mean, quality
  filter), `lateral_segment`, `pair_distance`, `well_distances`, `nearest_neighbours`, `classify_parent_child`,
  `wells_in_polygons` (maphub.point_in_polygon), `drainage_ellipse`, `ellipse_polygon`, `ellipse_overlap`, `gun_barrel`.
  DashboardWindow.spacing (ellipses on its map, per-well nearest distance / class) and gun_barrel_view (dialog +
  section line on the basemap). Tests `test_spacing.py` (run, run_gui).
- `depletion.py` (30, v0.53): `ellipse_mask` / `ellipse_area`, `inplace_in` (SoPhiH x cell / B), `overlap_fraction`,
  `depletion_factor` (parent cum x overlap fraction / child in-place; pressure proxy), `performance_vs_depletion`,
  `recovery_factors`, `remaining_map`, `undrilled_locations` (slots across the unit at a spacing, < 20 % covered),
  `material_balance_time`, `rta_linear` (1/q or dp/q vs sqrt t; t_elf = 3 points > 10 % off; final fit on points
  within 2.5 %), `investigation_distance`, FLUID_COMPLEXITY + `frac_complexity_index` (-> aspect for
  spacing.drainage_ellipse), `type_curves`, `cash_flow`, `npv` (monthly end-of-month discounting, economic limit),
  `inventory`, `area_summary` (HTML). DashboardWindow.recovery (RF + child DF). Tests `test_depletion.py`,
  test_spacing GUI.
- `wellindex.WellIndex` (15a; sqlite3 stdlib): wells (name, api, x, y, elev, td, bucket bx/by, survey), tops,
  logs catalogue, meta(bucket); add_wells/add_tops/add_logs, from_wellset, in_box, nearest (ring growth with
  exactness bound), by_api, tops_of, wells_with_curve, headers, xy. `WellLayer(wells, max_labels=150,
  index_from=2000)` (v0.18) = what the basemap draws: points(), in_view (index or numpy), labels(view) (only in view,
  [] above max_labels), nearest. `Basemap.wells` / `update_labels(view)` (debounced on sigRangeChanged).
- `import_3d(workers=None)` (v0.18): trace conversion + time-slice copy in a bounded ThreadPoolExecutor
  (workers+1 chunks in flight, chunk <= ~64 MB); bit-identical to workers=1. Section ImageItem autoDownsample;
  `SectionPanel._level_sample` (<= 1M values for percentiles).
- Jobs: `MainWindow.run_job` + 3D import poll show a status-bar Cancel button (`cancel_job`); long functions
  raise RuntimeError('cancelled') between blocks; `job_cancelled(state)` turns that into a status message;
  `import_3d(cancel=)` deletes the partial store. Tests: `test_hardening.py`.
- GUI (`gui/`): `main_window.MainWindow` (File menu: new/open/save, Import 3D (threaded, QTimer poll),
  Add 2D lines, Add nav CSV, Link well project; docks: Data tree, Basemap), `section_panel.SectionPanel`
  (ImageItem row-major (rows = time), toolbar: source combo, Inline/Crossline/Time slice, number spin +
  slider + step, colormap, clip %, reverse polarity; PgUp/PgDn; signals positionChanged, cursorInfo;
  `current_trace_line()` for basemap; `PickViewBox` routes left click/drag to `pick_at`/`pick_drag` in pick
  mode; `track()`, `undo()`, `draw_horizons()`), `horizon_dock.HorizonDock` (list, settings, buttons), `fault_dock.FaultDock`; panel `set_mode('horizon'|'fault'|None)`,
  `fault_click`, `end_stick`, `fault_lines()` (pure data for the overlay), `basemap.Basemap` (outlines, 2D lines, wells, red current line, horizon time map via `show_horizon`;
  signals openLine(name), openInline(vol, il, xl)), `colormaps.lut(name, reverse)`.
- `__main__`: `--selftest`, `--info FILE.sgy`, `--png PROJECT OUT.png` (matplotlib render, no Qt).
- `selftest.write_segy(path, traces, headers{byte: int array}, dt_us, fmt)` makes synthetic SEG-Y (IBM or
  IEEE) - reuse it for new tests.

- Live Map + map bus (item 31, v0.54): `mapbus.py` (stdlib + numpy; imported by the sibling tools from the Seismic
  folder): `MapBus` (subscribe(name, fn, topics), publish(topic, source, **payload) -> event {seq, topic, source, origin,
  time}; never delivered to its source name; broken subscribers recorded in `errors`), services updated BEFORE delivery:
  `SelectionService` (sets per kind, select(kind, ids, mode replace|add|remove|toggle), norm_id case/space-insensitive,
  publishes the full id list only on change), `FilterService` (filters {column: {min,max} | {values} | {rule}}, mask(table,
  exclude=col), visible_ids), `ActiveObject` (state[kind] = {value, ...extras}; set() is a no-op when value + extras are
  unchanged -> no echo loops), `notify(kind, ids, source, **info)` = DataChanged; `WellTable` (ids + columns numeric
  (float NaN) / text, meta {owner, unit, kind}; set_column from {id: v} or aligned list; save(path, owner) =
  read-merge-write of that owner's columns, atomic replace with retries; reload); rules `eval_rule` (safe ast: names or
  [Bracketed Name], and/or/not, chained comparisons, in (...), + - * /, abs, isnan, contains); `bus_folder(seis, wcproj)`
  (`<wcproj stem>_mapbus` else `<seisproj>_seismic/mapbus`); `FileBridge(bus, folder, role)`: local events (origin ==
  bus.origin) appended to events.jsonl (first line {"journal": id}; restarted above max_bytes, readers re-read from 0
  when the id changes), poll() delivers other processes' events (half-written last line waits), state.json snapshot for
  late joiners (catch_up), save_table(owner) + DataChanged 'welltable'. Retrofit: `gui/maplink.SeismicMapLink` (main
  window: wellsSelected, positionChanged -> active inline/crossline/timeslice/line + 'section' xy, horizons/faults
  changed, ddc_tick -> 'velocities', mistie dialog on_residuals -> 'misties' with tie xy; follows selection, active line /
  inline / well from other origins; 300 ms bus_tick), well_correlation `maplink.WellCorrMapLink` (select_well, section
  wells -> active 'well_section', _after_edit -> push_tops = 'Top <fm>' TVDSS columns + 'tops'; follows selection,
  active well, 'well_section' -> section named after it), mistie_tool `maplink.MistieMapLink` (config well_project /
  mapbus_folder; tie select -> selected lines + active line; solution_changed after every refresh with per-line
  shift/phase/gain + tie x/y/residuals; follows active line -> worst tie on it). Tests: `test_mapbus.py` (incl. a real
  second process), well_correlation `test_maplink.py`, Mistie selftest `_maplink_check`.
  `livemap.py` (Qt-free): nice_scale / scale_bar_geometry / north_angle (numerical grid convergence from the project
  CRS) / north_arrow_geometry / cursor_text; `MapState` (display["live_map"]: view, well_style, filter_columns,
  crossplot, histogram, panels); styling `style_wells(table, style, filters, selection)` -> ids, x, y, rgba, size,
  symbol, label, visible, selected, legend, pies (colour modes fixed / continuous ramp / categorical / rules (first
  match wins); bubble size ~ sqrt); `labels_in_view`, `lasso_select` / `box_select`, `histogram_data` (ignores its own
  filter), `crossplot_data`, `table_rows`, `filter_widget_spec`; 31e `mistie_markers`, `shotpoint_marks` (lines2d style
  shotpoints=N); 31f `wells_along_line`, `section_xy`; 31h `layer_request` / `apply_layer_request` (lines = open
  polygons items with an 'action' {kind, name, mode, number} for clicks; polygons; points; grid file CPS-3 / ZMAP+ /
  npz), `section_hit`; 31i `styled_primitives`, `export_map` (mapexport.export_map extra_prims / extra_legend),
  `export_gis` (export_stack geojson + zmap, wells_table.geojson/.shp, section_lines, picks), `picks_2d`; 31d
  `path_polylines` / `well_paths`. `livegrid.py`: `LiveGridder(stack, table, project, pause, coarse_factor, clock,
  threaded)` add_source(column | horizon), changed / changed_column / changed_horizon -> coarse job now, tick() -> full
  after the pause, worker thread with generation counter (stale results dropped), barriers = polygons layers with
  style fault + vector layers named *fault* via carto.grid_by_blocks, results replace the 'Live: <name>' grid layer in
  place; `depth_grids(stack, DynamicDepthModel | VelocityModel)` ('Live depth: <h>' layers), `isochore`. `gis/geojson.py`
  (read / write, RFC 7946 ring order, legacy crs member), `gis/zmap.py` (ZMAP+ read / write), gis/vector.py:
  .geojson load, `crs_warning`, `CULTURE_PRESETS` + `culture_style` + `category_colors` (style categories / dash / hatch).
  Basemap: `skip_kinds`, `min_poly_points`, dashed curve primitives (8th element {"dash"}); maplayers polygons items
  with "open" = polylines. `gui/live_map.LiveMapWindow(project, bus, parent, open_section, open_arbitrary,
  depth_model_fn)` with `LiveMapCanvas(Basemap)` (styled wells, paths, pies, rings, labels in view, scale bar, north
  arrow, section (yellow) / well-section (cyan) lines, mistie markers). MainWindow.live_map() (View menu, Ctrl+Shift+M).
  Tests: `test_livemap.py` (run, run_culture, run_seismic, run_wellsection, run_livegrid, run_livedepth, run_output,
  run_paths + GUI run_gui*).
  **Opening the Live Map on Windows**: start `Seismic Interpretation Tool.bat`, open the project, View > Live Map
  (Ctrl+Shift+M). For live links start the Well Correlation Tool on the project's linked .wcproj.json (and the Mistie
  Tool with "well_project" in its config); nothing else to configure - the shared folder is created next to the well
  project.

- Automated fault mapping (item 32, v0.55): `faultlik.py`: `sample3` (vectorised trilinear), `semblance3` (dip-steered 3x3
  semblance for every trace, edges padded), `plane_normals` (gradient structure tensor, largest eigenvector, sign = up
  -> horizontal part points down-dip), `plane_axes` / `plane_smooth` (separable Gaussian along strike then dip,
  trilinear), `thin` (non-max suppression along the horizontal normal), `fault_likelihood_block` (fl = 1 - s_plane^8;
  pass 1 normals from 1 - s, pass 2 from the smoothed fl: dip error 15 -> 2 deg), `enhance` (ant-track-like),
  `index_frame(vol)` (map vectors of +1 inline / crossline), `geo_orientation` / `plane_normal_index` (strike RHR /
  dip with v/2 for the time axis, default 2500 m/s), `compute_fault_volumes` (write_volumes: '[Fault likelihood]',
  '[TFL]', '[Fault strike]', '[Fault dip]', halo = half_w + 3 sigma + 3), `section_likelihood`, `fault_likelihood_2d`.
  `faultskin.py`: `Skin` (pts i/j/k, normals, fl, throw), `grow_components` (seeded BFS fronts; neighbour within
  max_angle of its parent AND max_dev of the seed, judged on METRIC normals (`metric_normals`, `default_scale`) - in
  index space conjugate faults look parallel), `extract_skins` / `extract_from_volumes` (strike/dip volumes ->
  `index_normals`), `skin_sticks` / `skin_to_fault` (sticks every Nth inline or crossline, whichever crosses the
  fault), throws: `throw_at_points` (RGT: iso-T across the fault vs linear footwall extrapolation), `throw_from_rgt`,
  `rgt_throw_volume` (+ throw-weighted probability), `throw_by_correlation` (Hale-style slip, footwall dip removed),
  `throw_profile`, `throw_map`, `displacement_length_flag`; QC: `skin_table`, `poles` / `stereo_xy` / `stereonet`
  (Kamb) / `rose`, `filter_skins` (ranges or a stereonet brush polygon), `merge_candidates` / `merge_skins` /
  `split_skin`; maps: `horizon_cutoffs`, `fault_polygon` (fw line + reversed hw line, throw, down vector),
  `polygons_layer_items` (hub polygons with compass 'down'); 2D: `Cut`, `segments_2d`, `cuts_on_line`,
  `cut_link_score` (same line: extrapolated trace; different lines: A(t)->B(t) parallel at two times = strike, same
  dip side, same perpendicular migration rate = dip), `link_cuts` (mutual best per line pair), `cuts_to_faults`,
  `project_cuts`; `ml_backend` / `cnn_fault_probability` (optional). `synth.planar_faults_cube` (index-space planar
  faults, throws, geologic time T). Tests `test_faultlik.py` (run, run_volume, run_skins, run_throw_2d, run_gui).

- Velocity-volume depth conversion (item 37, v0.56): `velvolume.py` `VelocityVolumeModel(project, vol, [(horizon,
  formation)], modes, method idw|kriging, residual_method tps|kriging|idw, nu, vmin/vmax, trend None | {volume, kind
  rms|interval} | {array, kind} | ndarray, td_fn, tops_fn, exclude)`: `well_curves` (WellSet.td_for -> WellCurve.v per
  sample, despike, caps, QC flags), `_well_zone_curves` (strat_coords -> per zone u axis: proportional 0..1 nu levels,
  top / base ms offsets; ratio to the Dix trend when given), `build` (lateral weights W (nil, nxl, nw) once,
  `traces(ii, jj, calibrated)` nan-aware weighted sums per zone x u, `depth_traces` (z = cumsum v dt / 2 from t = 0),
  `_depth_grids(mask, base)` -> z_raw (K, nil, nxl)), `residuals(from_layer)` (residual at wells = top - bilinear
  z_raw; gridded, then the read-back misfit re-gridded up to 6 times so wells between nodes tie to < 1 mm; z_corr;
  scale per zone = corrected / raw thickness -> calibrated traces), `depth_grid`, `residual_grid`,
  `layer_velocity_grid`, `depth_of_time` (scenarios.depth_convert_volume works with it), `set_top`, `refresh_tops`
  (re-reads the .wcproj when its mtime changes), `update_well` (re-integrates only traces with weight > 1e-3),
  `blind_test` (LOO incl. the residual correction), `probability_maps`, `velocity_section`, `write` (write_volume),
  `to_dict` / `from_dict`; `dix`, `save_scenario`, `difference_grids`. `gui/velvolume_dialog.VelocityVolumeDialog`
  (well_velocities, build, store, blind_test vs the layer-cake, add_maps, refresh = MainWindow.ddc_tick). Tests
  `test_velvolume.py` (run, run_gui; earth v = 1800 + 0.25 dx + 0.4 t + 300 below H2; west-only wells for the trend).

- 3D Viewer (item 33, v0.57): `scene3d.py` - world x, y map m, z = -t ms * zscale (1.25 m/ms) * VE; `Mesh(kind
  triangles|lines, pos, rgba)` + `interleaved()` (x y z r g b a float32 for one VBO); `lut_rgba`, `grid_triangles`
  (valid-corner filter), `slice_mesh(vol, mode, number, lut, clip, decimate)` (reads one inline / crossline / time
  slice), `plane_corners`, `curtain_mesh` (2D lines), `surface_mesh` (NaN holes + max_jump ms keeps faults open),
  `fault_mesh` (Fault.triangles), `tube_mesh` (parallel-transport frames) / `disc_mesh` / `polyline_mesh` / `drape`,
  `well_path_world` (td.twt), `box_mesh` (+ IL / XL / time label positions), `look_at` / `perspective`, `Camera` (orbit,
  pan in the view plane, zoom, fit, project, ray, to/from_dict), `ray_plane`, `pick_on_plane` (pixel -> il, xl, t: the
  34a picking math), `Scene` (objects with visible / opacity / colour bar / version, draw_list opaque first, saved views,
  state in display['view3d'], `render_png` = matplotlib painter's algorithm, `turntable` GIF). `gui/view3d.py`:
  `GLCanvas(QOpenGLWidget)` (24-bit depth, 4x MSAA, shader attributes pos=0 col=1, uniforms mvp (QMatrix4x4 row-major)
  and opacity (setUniformValue1f), one QOpenGLBuffer per mesh keyed by id(mesh), stale buffers destroyed after a
  paint; left drag orbit, middle / Shift drag pan, wheel zoom), `SoftwareCanvas` (fallback), `Viewer3D` (planes via
  toolbar sliders - decimated while dragging, full resolution on release + seismic window follows, VE, horizons, faults,
  wells (trajectories, else vertical), tops discs, 2D curtains (hidden by default), hub culture draped at the survey
  top, bus: active inline / crossline / timeslice, wells selection highlight, picks / faults / tops / layers rebuild),
  `_check_gl` swaps to software when the context fails. dev/fakeqt gained PySide6.QtOpenGL / QtOpenGLWidgets mocks.
  Tests `test_scene3d.py` (run, run_gui incl. the GL draw path under the mock).

- 3D interpretation (item 34, v0.58): `geobody.py` - `label_components` (6-connected, min-label propagation + pointer
  jumping, labels 1..n by size), `label_volume(vol, lo, hi, block, out_path)` (per inline block, union-find across
  block faces, int32 memmap, bodies [{id, cells, volume_m3, bbox}] via `cell_volume` = bin area x v/2 dt),
  `surface_nets` (padded mask; one vertex per surface cell nudged towards its inside corners, one quad per sign-change
  edge -> watertight, Euler 2 for a sphere), `edge_use_counts`, `body_mesh_world`, `convex_hull`, `footprint`,
  `fits_budget`, `opacity_curve`, `render_slabs(sub, lo, vol, lut, clip, curve, n_slabs, axis, ..., step, index_step)`,
  `integrated_graphics`. Viewer3D: `set_volume` (any project volume on the planes, rainbow LUT for attributes /
  inversion), `pick` (nearest visible plane hit), `pick_horizon` (horizons.Picker.pick_points -> same horizon object,
  HorizonSet undo), `pick_fault_point` / `end_stick` (FaultSet snapshot), `track` (Picker.track from the 3D seeds; run_job
  when threaded), `volume_render` (voxel budget -> read stride), `geobodies` (surfaces + 'Geobodies: <vol>' polygons
  layer + bus 'layers'), `colour_horizon_by`. Mock trap again: attributes read with getattr on a Qt subclass must be
  initialised in __init__. Tests `test_geobody.py` (run, run_volume, run_gui).

- Structural framework + 3D grid (item 35, v0.59): `framework.py` - `smooth_sticks` / `remove_stick` / `add_stick`,
  `fault_trace(fault, MapGrid)` (triangle edges crossing the horizon; vertices at the grid edge sampled towards the
  triangle centre), `extend_trace`, `Geometry(x0, y0, dx, dy, nx, ny, rotation)` (`nodes` = pillars (ny+1, nx+1),
  `centres`, `around`), `fault_blocks(X, Y, traces)` (cut node graph, label propagation), `FrameworkModel(geom,
  [(name, (px, py, v), rule)], faults [(name, (xs, ys))], method, domain)`: build -> blocks, surfaces[k] (nblocks,
  ny+1, nx+1) per block (a block without data uses all points), grids[k] (own block per node), `apply_rules`
  (conformable fmax with the horizon above, erosional fmax on all older, base fmin with the one below), report
  crossings before / after, `to_depth(depth_fn)`, `horizon_grid`; `depth_fn_from_model` (depth_of_time in 512-point
  chunks, diagonal); `crossings`; `cell_blocks` (block of a cell = first corner node reachable from the centre without
  crossing a fault); `Grid3D(model, [(kind, n | dz)] per zone)` -> zcorn (2nz, 2ny, 2nx) Eclipse order, zone,
  block, actnum, thickness / cell_volumes / coord / zone_index / block_index / quality / zone_thickness;
  `write_grdecl` (SPECGRID, COORD, ZCORN, ACTNUM, ZONE, FAULTBLK, FIPNUM, props; integers run-length encoded),
  `read_grdecl`, `write_surfaces` (ZMAP+ / XYZ). `gui/framework_dialog.FrameworkDialog` (build from 3D picks +
  fault traces, depth via the velocity-volume model when built else the layer-cake, make_grid, export, bus
  'velocities' -> rebuild). Tests `test_framework.py` (run, run_gui).

## Well Correlation Tool (v0.4.0, stable - Chris uses it)
las.py (LAS reader), curves.py (alias families GR/RDEEP/RSHAL/RHOB/NPHI/DT/PE/SP/CALI, compositing,
flatline removal), geo.py (NZTM2000 projection, min curvature), project.py (Project.open, wells, tops,
tops_meta, reference_tops, sections, load_well with npz cache), section.py / lithology.py (cross section
layout, Vsh lithology fills), tops.py (edit/undo, import/export), autocorr.py (suggest picks), zones.py,
mapping.py (TPS/IDW grids, contours), export.py, gui/. Hang modes TVDSS/MD/FLAT:<fm>. Tops are MD (m below
KB). Formations: Tikorangi Limestone, C-sands (Mangahewa), D-sands (Kaimiro), E-shale, Tane Member,
F0 / F1 (Farewell F-sands), Base Kapuni F-sands, Basement.

- `dlis.py` (v0.5): RP66 v1 reader - `read_logical_records` (SUL, visible records, segments: padding,
  checksum, trailing length, predecessor/successor flags), `parse_eflr` (SET/template/OBJECT/ABSATR, all rep
  codes incl. IBM/VAX/FSHORT/DTIME/OBNAME/OBJREF), `DlisFile` -> `LogicalFile` (origin, channels, frames,
  parameters, tools; FILE-HEADER starts a logical file), `frame_data` (vectorised structured dtype per frame;
  slow path for variable-size codes), `dlis_to_las`, `DlisLogs` (LAS-like adapter used by
  project.load_well for *.dlis in las_paths), `checkshot_from_dlis` / `write_checkshot_csv`. Tests in
  `test_dlis.py` write a synthetic DLIS byte by byte (multi-segment records, padding, arrays, IBM).
  Tui-1 checkshot (main survey): 11 levels 1897-3900 m MD, TWT 1685-2806 ms. Pateke-2 checkshot:
  SEIS_CSI_037PNP.DLIS (27 levels); the CSI_0xx files are cased-hole GR/CCL logs, not checkshots.

- `petro/` (PLAN 20, v0.6+): results in `<project>_petro/` (`store.py`: `petro_dir(project, create, override)` -
  `project.petro_out` overrides the folder, used by tests and the real-data runs so nothing is written into data/;
  `save_curves` merges curves into `<well>.npz` on the composite MD grid (`replace_prefix` drops old QC_* first),
  `load_curves`, `save_json`). `tools.py` (20a): `DEFAULT_TOOLS` table (vendor, desc, kind laterolog / induction /
  propagation / microresistivity / density / ..., conveyance, families, distinctive mnemonics, aliases = service tokens
  + DLIS generic / trademark names, typical vertical resolution m + per-mnemonic overrides, notes such as neutron
  matrix), merged with `<project>_petro/tool_table.json`; `file_evidence` (DLIS ORIGIN PRODUCER-NAME + TOOL objects
  and their CHANNELS, LAS SRVC / OS lines, file-name tokens, LWD vs wireline from drilling channels / D&M / LWD-only
  mnemonics vs TENS/CS), `identify_curve` (order: DLIS tool > mnemonic > file-name service > description > generic;
  eff_res = max(tool res, 2 x sample)), `identify_well`, `used_curves` (maps well.sources strings back to raw curves).
  `qc.py`: `gaps`, `flatline_mask`, `rolling_median`, `spike_mask` (running median/MAD + isolated 1-2 sample
  excursions vs local roughness), `estimate_bit_size` (p40 snapped to standard sizes), `bad_hole` (CALI - BS > 1 in or
  |DRHO| > 0.15), `depth_shift` (detrended z-scored xcorr, parabolic; add shift to z), `bed_boundaries` (|d/dz| peaks
  of the finest-resolution curve), `bed_thickness_curve`, `splice_report`, `well_qc` (report dict + QC_BADHOLE,
  QC_SPIKE_<fam>, QC_THIN_<fam>, BED_THICK, BS_USED curves), `qc_project` (+ qc_summary.csv), `format_report`.
  CLI `--petroqc PROJECT [WELL..] [--out DIR]`, `--tooltable OUT.json`. Tests `petro/test_petro.py` (`run_qc`).
  `interp.py` (20b, v0.7): equations (vsh_from_igr linear / Larionov x2 / Clavier / Steiber, vsh_sp, vsh_nd, phi_density,
  phi_sonic_wyllie / _rhg, phi_nd (gas: rms form where PHID - NPHI > gas_xover), temperature (surface + gradient on
  TVDSS or a curve), arps, rwa, rw_from_rwa (P5 in clean porous rock), pickett_fit (lower-envelope water line, m free or
  fixed, intercept = mode of the lower half), rw_from_sp, juhasz_b, sw_archie / simandoux (closed form) /
  modified_simandoux / waxman_smits / dual_water (vectorised bisection `solve_sw` on the conductivity model) /
  indonesia, perm_timur / coates / wyllie_rose, toc_passey). Parameter sets: `DEFAULTS` < params.json default < zones
  < wells[well]['*'] < wells[well][zone] (`resolve`); None = auto (GR baselines from lithology.gr_baselines, shale points
  = median at Vsh >= 0.8, Rw from Rwa min). `zones_of` (tops in formation order; outside = zone None, default params,
  not summarised). `interpret_well` -> {depth, tvdss, rt, curves (VSH PHIT PHIE SW SWT BVW PERM RWA TEMP GAS_FLAG
  NET_RES NET_PAY TOC DLOGR ZONE_ID), zones (summary rows, TVD-weighted, SoPhiH = sum PHIE (1 - Sw) dTVD over pay),
  used {zone: params incl. rw_how}, notes}; saves curves + summary.csv (rows of other wells kept). `cutoff_sensitivity`,
  `pickett_data`, `interpret_project`. CLI `--petro PROJECT [WELL..] [--out DIR]` (writes a params.json template on the
  first run). Tests `run_interp` (hand values, round trips, forward-modelled shaly-sand well, precedence, real Maui-5 /
  Tui-1).
  `multimin.py` (20c, v0.8): `MINERALS` / `FLUIDS` end-points (GR, RHOB, NPHI limestone units, DT, PEF; clay flag),
  logs GR RHOB NPHI DT U (= PEF x RHOB) BVWR (= (a Rw / Rt)^(1/n) = water volume for m = n), `endpoint_matrix`,
  `solve_volumes(D, E, sigma)` = exact constrained WLS (sum 1, >= 0) by trying every support set with a KKT inverse per
  set and log-availability pattern (batched over samples; 20k x 6 components ~0.1 s), `log_matrix`, `multimin_well`
  (per zone mm_minerals / mm_fluid / mm_logs / mm_vclay_cut from the parameter set, Rw from the 20b run; MM_* curves,
  zone rows with V_<component>, Misfit_med), `monte_carlo(project, well, dists, n, method deterministic|multimin)`
  (draw specs normal / rel_normal / uniform / triangular / lognormal; `rw_mult` multiplies whatever Rw a zone uses; draws
  written into every zone of a copied parameter set; P90 = 10th percentile), `mc_table`. qc.py also flags barite mud
  (median PE > 4 / > 2.9 b/e in clean low-density rock -> barite_suspected likely / possible). CLI `--multimin`,
  `--petromc`. Tests `run_multimin`.
  `rockphysics.py` (20d, v0.9): Batzle-Wang `brine(T, P, ppm)`, `gas(T, P, G)` (Z factor, adiabatic K via numerical
  dZ/dPpr), `oil(T, P, api, gor, G)`; `vrh`, `fluid_mix` (Wood or Brie e), `gassmann_dry` / `gassmann_sat`, `moduli`,
  `velocities`, `substitute` (fluid 1 -> 2; phi < phi_min unchanged), `vs_greenberg_castagna` (lithology mix),
  `vs_with_fluid` (GC iterated through Gassmann for hydrocarbon zones), `elastic` (AI SI VPVS LR MR PR YM BI Rickman),
  `rock_physics_well` (per zone RP_DEFAULTS < parameter set: salinity, api, gor, gas_gravity, rp_hc, pressure_gradient,
  rp_sw_hc, rp_fizz_sg, brie_e, rp_lith; measured shear from DTSM-type raw curves else GC; scenarios brine / oil / gas
  / fizz substituted only in 20b NET_RES rock; saves RP_* curves), `scenario_table(res, zone_rows, net)`. CLI
  `--rockphysics`. Tests `run_rockphysics`.
  20e (v0.10): `petro/plotdata.py` (nd, mn, pickett, buckles, histogram with discriminators, render_png via matplotlib;
  pure numpy, <= 20k points), `gui/petro_dialog.PetroDialog(win)` (win needs .project, .section (or None), optional
  .selected_well; logic: set_well / set_zone / param_rows / set_param(name, text, scope 0 project | 1 zone | 2 well+zone
  | 3 well all zones) -> params.json / run_qc / run_interp(wells) / run_multimin / run_mc / run_rockphysics /
  summary_rows / crossplot_data / set_discriminators / export_png / show_on_hub(prop) -> `<petro>/hub_request.json`
  {time, wells, zone, property, summary, project} for the map hub of item 21). MainWindow menu Petrophysics
  (`show_petro`, Ctrl+Shift+P). Section tracks VSH / PHI (PHIE + BVW, hydrocarbon fill) / SW / PERM: `TRACKS` +
  `FAMILY_INFO` entries, `SectionLayout.log_values` falls back to `petro.store.well_curves` (npz cached on the Well
  by path + mtime). `python -m well_correlation --guitest` (real Qt offscreen or PYTHONPATH=dev/fakeqt; monkeypatches
  SectionView._apply_lod under the mock).

- `carto.py` (21f, v0.11): shared cartography (dict grids {z, x0, y0, dx, dy}): `hillshade` (multi-directional, ve),
  `ramp_lut` (depth / thickness / property / diverging / terrain / bathymetry), `default_ramp`, `shaded_rgba`,
  `chain_segments` (pairs -> polylines; drops zero-length segments), `contour_polylines`, `label_candidates`,
  `label_box`, `place_labels` (greedy, priority), `boxes_overlap`, `point_in_polygon`, `blank_polygons` (dilated),
  `fault_blocks` (BFS over node graph with fault-crossing edges cut) + `grid_by_blocks`, `fault_ticks`, `spill_point`
  (binary search on fill level until the crest region touches the edge / NaN; spill node), `WELL_SYMBOLS`,
  `classify_status`, `symbol_paths`, `nice_scale`, `north_arrow_angle`, `STYLE_PRESETS`, `styled_map` (used by
  gui/map_view.refresh_map when display["map"]["carto"] is not False). Tests `test_carto.py`.

- `petro/logml.py` (22a, v0.12): `analytics()` imports seismic_tool.analytics (models, validation) from the sibling tool
  folder; `blocked_splits(depth, wells, k, block_m, buffer_m)` (group_kfold over well x depth block, same-well training
  samples within buffer_m of a test block dropped), `gather`, `make` (GBM / RF / quantile GBM), `predict_curve` (CV
  report + <T>_PRED / _P10 / _P90 curves where T is missing), `electrofacies` (k-means / GMM on standardised curves,
  facies ordered by the first curve's centre, FACIES curve), `facies_proportions`. `petro/reports.py` (22b): ALIASES +
  `read_table(path, kind show|test|core|micp|pressure, depth_unit, pressure_unit)` (CSV, XLSX via pandas+openpyxl),
  `extract_from_text` (sentence split, depth / range regex, kinds from TEXT_PATTERNS, flag "auto-extracted, check"),
  `pdf_text` (pypdf optional), `fluid_gradients` (exhaustive 2-leg fit, contact = line crossing; 1 leg unless SSE drops
  30 % and gradients differ > 0.0015 MPa/m), `gas_ratios` (Haworth Wh / Bh / Ch + zone), `read_mudgas`,
  `core_log_calibration` (depth shift scan + linear fit), `fit_archie_m` / `fit_archie_n`, `save_reports` /
  `load_reports` (reports.json + reports_summary.csv; maplayers.petro_values falls back to it for Shows / Tests / Cores /
  Pressures / Contact_TVDSS). Section TRACKS FAC (FACIES) and DTP (DT_PRED + P10 / P90). PetroDialog: run_predict,
  run_facies, import_report. Tests `run_ml_reports` (+ 22 GUI check in run_gui).

## Conventions
- numpy only for algorithms; vectorise; memmap big arrays; float16 stores for volumes, float32 math.
- GUI: PySide6 imported before pyqtgraph; `pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")`;
  heavy work in a thread with progress polled by QTimer; undo for edits; keyboard shortcuts listed in Help.
- Every feature gets a selftest check (PASS/FAIL lines, final "SELFTEST PASSED").
- Time in ms in the UI, seconds in meta. TWT axis inverted (down).
- Units: metres; NZ coords NZTM2000 (EPSG:2193); NPRA nav in its own projected metres.

## Testing GUI code (no Qt here)
`cd "tools/Seismic Interpretation Tool" && PYTHONPATH=../../dev/fakeqt:. python3 -m seismic_tool --guitest`
runs the GUI logic tests (also works with real Qt, offscreen).
`PYTHONPATH=/path/to/repo/dev/fakeqt:. python3 -c "..."` - `dev/fakeqt/anything.py` returns a permissive
`Anything` for every Qt/pyqtgraph attribute; signals are no-ops; combo/spin values are mocks (not real), so
set panel state directly (e.g. `panel.source=("3D","Synth"); panel.mode="Inline"; panel.number=7;
panel.redraw()`) and monkeypatch `vb.mapSceneToView`, `vb.viewRange`, `vb.viewPixelSize` as needed. This
catches import errors, wrong attribute names in our own code, and logic bugs - not layout bugs. Also render
key results with matplotlib to PNG and look at them.

## Lessons learned
- Pin PySide6 6.5.3 / pyqtgraph 0.13.7 (see above).
- FDCC is a density correction, not RHOB. Remove flat-lined cased-hole log sections.
- numpy 2: use `np.ptp(a)`, not `a.ptp()`.
- Trace-to-trace tracking on the NPRA land 2D lines fails within tens of traces on raw data (static
  jitter, noise); a 5-9 trace lateral mix + gap bridging makes it track 1000+ traces. Synthetic tests alone
  would not have shown this - always try new algorithms on the real data in data/.
- RGT solve: a per-node damping 'tau = t' looks harmless but its influence grows with node density (the
  lateral chain stiffness falls as 1/n links) - it flattened structure by several ms; one global datum row
  fixed it (median error 2.7 -> 0.15 ms). Correlation windows that straddle a fault lock onto the wrong
  reflector (errors = throw); always tie across faults away from the fault plane.
- Windows: close npz files (`with np.load(...)`) and drop memmaps before rewriting/replacing a file.
- RGT point edits: a single tau(t)=v constraint folds the model (non-monotonic) when the level moves more
  than a node interval; constrain a +/-24 ms segment instead (`move_level`).
- The Maui-5 / Maui-6 "checkshot" LAS files are mislabelled exports (TIME index with GR/CALI columns) - no
  usable time-depth; Tui-1 (4 CSAT DLIS) and Pateke-2 (SEIS_CSI_037PNP) are the real checkshots.
- Chris wants Kingdom/GeoGraphix-style behaviour and speed with big datasets; avoid per-trace Python loops.
- Real-Qt traps the Qt mock hides (run 2 review): `clicked` / `triggered` pass `checked` into slots with default
  arguments (always connect `lambda *_: fn()`); `QComboBox.itemData` returns a stored tuple as a list (compare with
  `tuple(...)`); the app uses `imageAxisOrder="row-major"` (rows = y, transpose (n_x, n_t) arrays); scatter symbols
  swallow clicks (`_click_through`); `hasattr`/`getattr` on a mocked QWidget subclass always succeeds - initialise
  attributes in `__init__` instead of probing.
- Never rewrite a store / .npy that may be memory-mapped (Windows refuses, Linux corrupts the live view): new
  imports and attribute volumes go to a fresh folder (`volume.unused_store`); edits use r+ memmaps of their own.
- Units / ids from CSV headers: match whole tokens ("..._LBS" contains "_t"); FracFocus water is gallons; API numbers
  lose leading zeros / gain ".0" in Excel; state production data omit zero months.
- Two Claude sessions may push to `dev` the same day (scheduled run + Chris's interactive session): always
  `git pull --rebase` before pushing and merge both sides' version bumps / WHATS_NEW entries.

