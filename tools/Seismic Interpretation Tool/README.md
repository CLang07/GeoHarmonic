# Seismic Interpretation Tool (Geo Tool Kit) - v0.2: loading, viewing, horizon picking + autotracking

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

## Command line
`python -m seismic_tool --info FILE.sgy` (header summary), `--selftest`, `--guitest` (GUI logic checks), `--png PROJECT OUT.png`.

## Coming next
2. Fault sticks and planes (tracker stops at faults), relative geologic time (RGT) workflow.
3. Wells on seismic: tops in time via checkshots, synthetics from DT/RHOB.
4. Time/structure maps from horizons, amplitude extractions, mistie integration for 2D.
