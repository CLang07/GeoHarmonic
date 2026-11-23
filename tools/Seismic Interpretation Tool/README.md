# Seismic Interpretation Tool (Geo Tool Kit) - v0.1, step 1: loading and viewing

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

## Command line
`python -m seismic_tool --info FILE.sgy` (header summary), `--selftest`, `--png PROJECT OUT.png`.

## Coming next
2. Horizon picking (seed + autotrack on peak/trough, 2D and 3D), faults sticks.
3. Wells on seismic: tops in time via checkshots, synthetics from DT/RHOB.
4. Time/structure maps from horizons, amplitude extractions, mistie integration for 2D.
