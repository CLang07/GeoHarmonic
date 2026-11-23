# Build plan - Seismic Interpretation Tool (Nov 23 - Dec 1, 2026, one item per run)

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

Later (Chris back): Maui 3D subset cut (`--subset in.sgy out.sgy --il a b --xl c d --tmax`) from the 4.1 GB
MAUI-3D PR2301 T-RMIG SEG-Y; mistie integration with the 2D lines; Tui-1/Amokura-1 wells.
