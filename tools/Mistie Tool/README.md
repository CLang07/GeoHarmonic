# Mistie Tool (mistie_tool 2.0)

Offline 2D seismic mistie analysis: find line intersections, measure the time shift, phase
rotation and amplitude difference at each tie inside a window you pick, solve the whole network
for one correction per line, check loop closures, and write the corrections (and optionally
corrected SEG-Y). Everything runs locally with **numpy, scipy and matplotlib**. No network access,
no AI, no data leaves the machine.

## Install

Python 3.9 or later.

    pip install numpy scipy matplotlib

Unzip this folder anywhere. Check it works:

    python -m mistie_tool --selftest

The self-test builds five synthetic SEG-Y lines with known errors, runs the full workflow and
checks that the corrections come back within 1 ms, 10 degrees and 5 %. It should print
`SELFTEST PASSED`.

## Quick start

1. Put the SEG-Y lines in one folder.
2. Write a starter config:

        python -m mistie_tool --make-config "C:\path\to\folder"

   Edit `mistie_config.json` there (line names, header bytes, window, reference line).
3. Open the interactive picker:

        python -m mistie_tool "C:\path\to\folder\mistie_config.json"

4. Or run without the GUI:

        python -m mistie_tool mistie_config.json --batch --auto-reject --apply


## Desktop app (version 2)

Double-click **Mistie Tool.bat** (Windows, Anaconda or any Python with numpy, scipy, matplotlib).
Or run `python -m mistie_tool` (empty) / `python -m mistie_tool my_project.mistie.json`.

**File**: New project from a folder of SEG-Y lines (asks for a nav CSV if the headers have no
coordinates), Open, Open recent, Save / Save as (one `.mistie.json` project file holds the settings,
every pick, tie on/off setting, locks and saved scenarios), Export tables (CSV), Write corrected
SEG-Y, Export report (HTML with maps and tables).

**Edit**: Undo / Redo (Ctrl+Z / Ctrl+Y, up to 200 steps) for every change; set the window for the
selected ties or all ties; copy the current tie's window to the selected ties; reset windows;
auto / force on / exclude for the selected ties (A / O / X). Settings for the QC thresholds.

**View**: tie panel raw / tie-corrected / network-corrected (1 / 2 / 3); map shows tie status,
measured misties or residuals (time or phase) with zoom and pan; map labels all / misfits / none.

**Network**: choose the reference line; solve one correction per line or per survey/vintage
(set a line's survey by right-clicking it in the Lines tab, or `"survey"` in the config); exclude the
worst tie; auto-reject misfits; save, load and delete scenarios (loading is undoable).

**Tables**: the tie table sorts by any column (click the header; Status sorts worst first) and
supports multi-select (Shift/Ctrl-click). The Lines tab shows each line's correction and how well it
is constrained; right-click to set the reference, lock a line at its current or a typed-in
correction, or select all ties on that line. Loops and Warnings tabs flag loops that do not close and
lines resting on one tie or none.

**Moving fast**: W jumps to the next worst tie, N / P step through the table in its current order.

## The basic viewer (version 1, used if tkinter is missing)

| Area | What it shows |
|---|---|
| Map | Lines and ties. Click a tie to open it. Green = fits the network, red = misfit, hollow = not used. |
| Tie panel | Butt display: the first line approaching the intersection (left), the second leaving it (right). The cyan line is the window top, magenta the base. View raw, tie-corrected or network-corrected. |
| Trace overlay | The averaged tie traces inside the window before and after the tie correction, with shift, phase, gain, correlation and the shallow-vs-deep stability check. |
| Network | Correction per line, ties used per line (1 = no independent check), loop closures. |
| Residuals | What is left at each tie after the network correction. |

Mouse and keys: drag the top/base lines or click to move the nearest one; `t` / `b` set top/base
at the cursor; `n` / `p` next/previous tie; `u` cycles a tie between auto / forced on / forced off;
`r` resets the window; `w` rejects the worst-fitting tie; `1` `2` `3` switch views; `s` saves.
Picks and outputs are saved automatically after every change and on close.

## Config reference (`mistie_config.json`)

| Key | Default | Meaning |
|---|---|---|
| `lines` | required | list of `{"name": ..., "segy": ...}` |
| `data_dir` | `"."` | folder with the SEG-Y files, relative to the config |
| `nav_csv` | `null` | optional CSV `line,cdp,x,y`; use when trace headers lack coordinates |
| `header_bytes` | cdp 21, x 181, y 185, scalar 71 | 1-based trace-header byte positions (SEG-Y rev 1 CDP X/Y); use 73/77 for source X/Y |
| `default_window` | `[0.4, 2.6]` | tie window in seconds before you pick one |
| `max_lag_ms` | 40 | largest shift searched; keep under about half the dominant period plus the expected mistie |
| `tie_avg_traces` | 5 | traces averaged at each tie point |
| `panel_half_width` | 60 | traces shown each side of the tie |
| `display_tmax` | full trace | bottom of the tie panel, seconds |
| `min_corr` | 0.5 | auto ties below this correlation are not used |
| `max_half_diff_ms` | 10 | auto ties whose shallow and deep halves disagree by more are not used |
| `near_end_traces` / `near_end_weight` | 200 / 0.3 | ties this close to a line end get this weight |
| `reject_tol_ms` / `reject_tol_deg` | 4 / 35 | residual scale used to rank and reject misfits |
| `reference` | `"auto"` | line held at zero correction; `auto` picks the best-tied line |
| `output_dir` | `"mistie_output"` | where results go |

## Outputs (in `output_dir`)

- `corrections.csv` — per line: shift (ms), phase (deg), gain, ties used, reference flag.
- `ties.csv` — per tie: window, measured mistie, correlation, stability, used, residual.
- `loops.csv` — closure of each independent loop.
- `picks.json` — your windows and use settings (reloaded next time).
- `*_mistie.sgy` (with `--apply`) — corrected copies, IEEE float, original headers kept.

## Sign conventions

A correction `(shift_ms, phase_deg, gain)` is applied as:
delay the trace by `shift_ms` (positive = later), rotate it by `phase_deg`
(Re{analytic signal x e^(i phase)}), multiply by `gain`. A tie measurement of line B against line A
is the correction that makes B match A, and the network solves `corr_B - corr_A = measured`
in a weighted least-squares sense, with phase solved wrap-aware so polarity flips are handled.

## How it decides which ties to trust

1. Correlation after alignment must reach `min_corr`.
2. The shift measured in the shallow and deep halves of the window must agree (`max_half_diff_ms`).
   Large disagreement means a time-variant mistie or a cycle skip.
3. Ties near line ends are down-weighted.
4. Loop closure: the raw misties summed around a closed loop should be near zero. A loop that does
   not close contains a bad tie, even if every tie in it correlates well.
5. `--auto-reject` (or `w` in the GUI) removes the worst-fitting tie one at a time, never one that
   would leave a line disconnected.

## Validate before trusting it

- Run `--selftest` on the machine you will use.
- Run a project whose misties were already solved in Petrel or Kingdom and compare `corrections.csv`.
- After `--apply`, point a second config at the corrected SEG-Y files and run `--batch`: the
  measured misties should be near zero.

## Limits

- Constant (not time-variant) correction per line. If the shallow and deep halves disagree a lot,
  pick a window on the interval you care about, or tie shallow and deep intervals in separate runs.
- 2D post-stack lines, big-endian SEG-Y, formats 1, 2, 3, 5 and 8. All lines must share one sample rate.
- Coordinates must be in one consistent projected system (feet or metres).
- To tie 2D lines to a 3D volume, extract arbitrary lines from the 3D along each 2D line (or along
  a crossing line) in your interpretation package, export as SEG-Y, and add them here as reference lines.

## Files

    mistie_tool/segy.py      SEG-Y read/write
    mistie_tool/geometry.py  coordinates, nav CSV, intersections
    mistie_tool/measure.py   mistie measurement and corrections
    mistie_tool/network.py   least-squares network, loops, rejection
    mistie_tool/project.py   config, ties, picks, outputs
    mistie_tool/gui.py       interactive picker
    mistie_tool/selftest.py  synthetic end-to-end test
    tools/npra_nav_to_csv.py converts the USGS NPRA navigation file to a nav CSV
    examples_npra_config.json config for the NPRA 7-line test (copy into that folder)
