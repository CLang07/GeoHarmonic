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
dev/fakeqt/     Qt + pyqtgraph mock for testing GUI code without Qt
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

