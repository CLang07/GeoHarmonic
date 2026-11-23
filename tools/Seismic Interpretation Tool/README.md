# Seismic Interpretation Tool (Geo Tool Kit) - v0.7: loading, viewing, horizons, faults, dip field, RGT model (stack, Wheeler, stratal slices, editing)

2D lines and 3D volumes in one project, PySide6 + pyqtgraph (same setup as the Well Correlation Tool).

## Install / run
1. Double-click **Install (run once).bat** (creates `.venv`, ~3 min). Look for `Qt OK` and `SELFTEST PASSED`.
   A shiboken traceback printed before `Qt OK` is harmless.
2. Double-click **Seismic Interpretation Tool.bat**, then File > Open project.

Starter projects:
- `Documents\2D Mistie Test\NPRA 2D.seisproj.json` - the 7 NPRA lines with npra_nav.csv.
- `Documents\NZ Well Correlation\Maui seismic.seisproj.json` - linked to the well project (wells on the
  basemap); import the MAUI-3D SEG-Y into it when it's downloaded.

## Loading data (File menu)
- **Add 2D SEG-Y lines** - read in place (memory-mapped, nothing copied). CDP from byte 21.
  If headers have no X/Y, **Add 2D navigation CSV** (`line,cdp,x,y`); line names match tolerantly (R_2 = R-2).
- **Import 3D SEG-Y** - inline/crossline bytes are detected automatically (189/193, 9/21, 5/21, ...), you
  can override them and see the text header. Import runs in the background and writes two float16
  copies (inline-ordered and time-slice-ordered) to `<project>_seismic\<name>\` so any inline, crossline
  or time slice displays in milliseconds. Disk needed ~ the SEG-Y size (int16/IBM 4-byte -> 2x 2-byte).
  The 4.1 GB MAUI-3D file -> ~4 GB of store, import roughly 5-15 min depending on the disk.
- **Link well project** - shows wells from a `.wcproj.json` on the basemap.

## Viewing
- Data tree (double-click), or click on the basemap: a 2D line opens it; inside a 3D outline opens the inline there.
- Toolbar: source, Inline / Crossline / Time slice, number + slider, step, colour bar, clip percentile, reverse polarity.
- **PgUp / PgDown** step lines (Shift = 10x). Mouse wheel zoom, right-drag stretch, right-click > View all.
- Status bar: IL, XL, time, amplitude, X, Y under the cursor. Red line on the basemap = section shown.

## Horizons (Horizons panel, right side; Interpret menu)
- **New** creates a horizon and turns on **Pick mode (P)**. In pick mode on an inline, crossline or 2D line:
  - click = seed pick, snapped to the nearest Peak / Trough / Zero crossing (+ to - or - to +) within
    "Snap search" ms, with sub-sample interpolation;
  - left-drag = manual line through the dragged points (every trace in between gets a lightly snapped pick);
  - Shift+click = erase the pick on that trace, Shift+drag = erase the span.
  Pan with the mouse wheel / right-drag while pick mode is on, or press P to leave pick mode.
- **T (Track)** grows the horizon from the seeds (no seeds: from all its picks). Settings:
  correlation window (+/- ms), max jump per trace, min correlation, lateral mix (2D: running mean over N
  traces - 5-9 helps a lot on noisy land lines such as NPRA; 3D: n x n block), bridge gaps (jump over up
  to N dead/noisy traces). Scope: **Whole survey** (3D: all inlines/crosslines; 2D: follows line ties onto
  crossing lines) or **This line only**.
- **Ctrl+Z** undoes the last pick/erase/track/delete (30 steps). **Clear line** removes the active horizon's
  picks on the section shown.
- Horizons show as coloured lines on sections, as dots on time slices, and the active horizon's time map
  is drawn on the basemap (3D as an image, 2D as coloured dots; red = shallow).
- Saved with the project: `<project>_seismic\horizons\<name>\*.npy` + list in the project file.

## Faults (Faults panel; Interpret menu)
- **N** = new fault (also turns on **Fault mode, F**). In fault mode click points down the fault on an
  inline, crossline or 2D line; **Enter** ends the stick, **Esc** cancels it, **Ctrl+Z** removes the last
  point. Shift+click deletes the nearest stick of the active fault. Sticks go to the active fault (select
  it in the list first).
- Draw sticks on several lines along strike: they are ordered along strike and joined into a triangulated
  surface. The surface is drawn (thin line) wherever it cuts the section shown - inlines, crosslines, time
  slices and 2D lines, including surveys the sticks were not drawn on. The sticks themselves show
  (thick, with dots) on the section they were drawn on. Basemap: stick traces + a line joining the stick tops.
- **Tracker stops at faults** (Horizons panel, on by default): the autotracker does not link traces across
  a visible fault surface (also not by gap bridging). Hide a fault (untick) to track across it.
- Saved inside the project file (sticks in map X, Y, TWT).

## Dip field (relative geologic time, step A)
- **Interpret > Compute dip field (current 3D volume)** runs in the background (progress bar) and writes
  inline dip, crossline dip (ms per trace, + = deeper towards higher inline/crossline numbers) and
  planarity (0-1, how plane-layered the data is; low in noise, chaos, at faults) as float16 volumes in
  `<volume store>\rgt\` (~3x the size of one store copy).
- The display box in the section toolbar switches between **Seismic**, **Inline dip**, **Crossline dip**,
  **Planarity**. On 2D lines either dip choice shows the dip along the line, computed on the fly (a few
  seconds per line) and cached in `<project>_seismic\rgt\`.
- This dip field drives the RGT model.

## Relative geologic time (RGT) model (PaleoScan-style, step B)
- **Interpret > Compute RGT (current 3D volume)** / **Compute RGT (all 2D lines)**. Settings: node spacing
  (traces) and interval (ms) of the model mesh (4 traces / 8 ms default; long 2D lines or big 3D: 8 / 8-16 is
  much faster), correlation window, min correlation, links across faults (correlate / none), max throw,
  use visible horizons as constraints, use visible faults.
- How it works: nodes are linked to their neighbours by following the dip field trace by trace and
  refining with a small cross-correlation; across faults the tie is correlated a few traces away from the
  fault (so the window does not mix the blocks). A global least-squares solve (all nodes at once) gives the
  geologic age of every node; interpreted horizons are hard constraints (one age per horizon); 2D lines are
  solved together and tied at intersections. Result: an RGT value for every sample (display box **RGT**).
- Iso-RGT surfaces are the model's horizons.

## Horizon stack, Wheeler view, stratal slices (step C)
- **Interpret > Build horizon stack (3D)** (number of surfaces, default 100) or **(2D lines)**: evenly spaced
  iso-RGT surfaces, saved next to the RGT.
- Mode **Stratal slice** (3D mode box): amplitude on one stack surface; the slider / PgUp / PgDn step
  through the stack - the PaleoScan-style way to look at depositional geometry (channels, fans) slice by slice.
- Domain box **Time / Wheeler**: Wheeler puts relative geologic time on the vertical axis (inlines,
  crosslines, 2D lines). Reflectors become flat; where an interval is much thinner than along the rest of the
  section (or missing - hiatus, erosion) the Wheeler view shows a gap. Horizons are drawn in RGT too.
  Picking and fault sticks are off in Wheeler and stratal-slice views.
- **Interpret > Convert stack level to horizon**: makes an ordinary, editable horizon from a stack level (the
  current stratal slice by default; 2D: on every line).
- Files: 3D `<volume store>\rgt\rgt.npy` (uint16, ~0.1 ms resolution) + `rgt.json` + `rgt_nodes.npz`;
  2D `<project>_seismic\rgt\<line>_rgt.npz`.

## Editing the RGT model (step D)
- **RGT contours** (toolbar tick box): the horizon-stack levels drawn as thin grey lines on sections.
- **Interpret > RGT edit mode (E)**: click on a level (grabs its geologic age), then click where it belongs
  (same or another trace of the section). The model is re-solved in a window around the edit (a fraction
  of a second); the rest of the model is untouched. The stack is updated too. Ctrl+Z undoes.
- **Use active horizon as RGT constraint**: one geologic age for all its picks, re-solved over the horizon's
  area. **Re-link RGT around active fault**: after drawing or editing a fault, re-solves around it so the
  model honours it (no need to recompute everything).
- Edits are saved immediately (`rgt_nodes.npz`, `edits.json` in the RGT folder).

## Command line
`python -m seismic_tool --info FILE.sgy` (header summary), `--selftest`, `--guitest` (GUI logic checks), `--png PROJECT OUT.png`.

## Coming next
2. Relative geologic time (RGT) workflow: dip field, global solve, horizon stack, Wheeler view.
3. Wells on seismic: tops in time via checkshots, synthetics from DT/RHOB.
4. Time/structure maps from horizons, amplitude extractions, mistie integration for 2D.
