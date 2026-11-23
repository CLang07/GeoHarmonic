# What's new (for Chris) - Seismic Interpretation Tool

Newest first. Each section: what it does, how to use it, what to test.

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
