# Well Correlation Tool (Geo Tool Kit) – v0.5 (steps 1-4 + DLIS)

Kingdom-style log cross sections with a GeoGraphix-style base map. PySide6 + pyqtgraph, runs offline.

## Install (once)
Double-click **Install (run once).bat**. It creates a private Python environment (`.venv` in this
folder, built from your Anaconda Python) and installs numpy, matplotlib, PySide6 and pyqtgraph there.
Keeping it separate stops Anaconda's own Qt 5 files clashing with PySide6 ("DLL load failed while
importing QtWidgets"). It should print `Qt OK` and end with `SELFTEST PASSED`.

## Run
Double-click **Well Correlation Tool.bat**. It opens the last project you used, or the
`NZ Well Correlation.wcproj.json` project in the folder above on first run.

## What step 1 does
- **Base map** (NZTM2000 metres): click wells to add/remove them from the section, or
  *Draw section line* (click points, double-click to finish) to pick up wells within a buffer,
  ordered along the line.
- **Cross section**: one canvas for all wells. Tracks: GR (sand/shale fill at the cutoff),
  deep/shallow resistivity (log), density–neutron (gas crossover fill), sonic, plus PE, SP, caliper.
  Hang on **TVDSS** (structural) or **MD**; *Flatten on* appears once tops exist (step 2).
  Spacing **equal** or **proportional to distance**; km between wells shown in the header.
- **Lithology** (toolbar *Fill between wells* + the LITH track): sand / silt / shale from GR
  normalised per well (5th-95th percentile), coal from low density, carbonate from high density
  (+ PE or sonic). Between wells the lithology is carried along the correlated tops, or choose
  *Formation zones* to colour between tops. Flat-lined (cased-hole / stuck tool) log stretches
  are removed on load.
- **Mouse**: wheel = vertical zoom, Ctrl+wheel = horizontal zoom, drag = pan, F = fit.
  The status bar shows well, MD, TVDSS and curve values under the cursor.
- **Section wells** panel: drag to reorder, Delete to remove, Reverse/Clear/New; several saved sections.
- **Well info** panel: header data and exactly which LAS curves were spliced into each log.
- **File > Export section image**: PNG or PDF of the current view.

## Tops (step 2)
- **Tops panel** (right): formation list top-to-bottom; the selected one is the *active* top.
  Add / Rename (or double-click) / Colour / up / down / Delete. A table lists the selected
  well's tops; type an MD to move a top exactly, clear the cell to delete it.
- **Pick tops** (P, toolbar, or the panel button), then in the cross section:
  - click in a well = put the active top there
  - drag any top line = move it (that formation becomes active)
  - Ctrl+click a top = make its formation active
  - Up/Down = nudge 0.5 m, Shift+Up/Down = 5 m, Delete = remove (selected well)
- **Undo / Redo**: Ctrl+Z / Ctrl+Y for every tops and formation change.
- **Line styles**: solid = published or your picks, dashed = auto pick, dotted with "?" =
  low-confidence auto pick. Any top you move becomes source "user".
- **Hang > Flatten on ...** lists every formation; fills and correlation lines follow your edits.
- **File > Import tops**: CSV/TSV with Well, Formation (or Marker), MD (or Depth) columns.
- **File > Export tops**: CSV (MD, TVDSS, NZTM X/Y, source, confidence), OpendTect markers
  (well, marker, MD), Kingdom (UWI, well, formation, MD, TVDSS). Tab-separated text you
  map to columns in the import dialog of each program.

## Correlate (step 3) - Correlate menu
- **Suggest pick**: `S` = active top in the selected well, `Shift+S` = in every section well
  that lacks it, `Alt+S` = all formations in the selected well. Suggestions appear in magenta
  with a match score (0-1; >= 0.45 high, 0.25-0.45 medium, < 0.25 low). **Enter** accepts
  them all, **Esc** clears. Accepted suggestions are stored as auto picks with their score.
  How it works: GR + log(deep resistivity) around the top in the other wells is matched by
  normalised cross-correlation, weighted by 1/distance, searching around the depth predicted
  from the interval thickness below the nearest top the well already has (your picks guide it).
- **Ghost curve**: hover a well at the depth you want to copy and press `G`. A magenta copy of
  60 m of GR follows the mouse; move it over another well, stretch / squeeze with the wheel or
  `[` `]`, click to drop. Tops inside the ghost window become suggestions (Enter accepts).
- **Zone statistics** (`Z`): every zone between tops in every well - gross MD/TVD, net sand,
  net-to-gross, sand beds, average GR/RHOB/NPHI/DT, geometric-mean resistivity, density
  porosity in net sand. Sortable; Export CSV. Net sand cutoff adjustable (default Vsh < 0.35).
- Vsh now uses a moving 400 m GR baseline (handles GR drift and run-to-run shifts).

## Maps and checks (step 4)
- **Map row** above the base map: *Structure (TVDSS)* of a top, or *Isopach (TVD)* between two
  tops; *Spline* (thin-plate, smooth, honours every well - best with few wells) or *IDW*
  (inverse distance, never overshoots); contour interval (auto or set). Values are posted next
  to each well. Areas far from any well are left blank instead of extrapolated.
  The map regrids automatically whenever you move a top. *Export map...* writes PNG/PDF.
- **Compare with reference** (`R`, Correlate menu): compares your current tops with a reference
  set - the project ships with *PR3099 Table 1.1 (NZOP 2005)* and its key findings. Moved
  tops are flagged (tolerance adjustable), and *Show as grey ticks* draws the reference picks
  beside your own in the cross section. *Save current tops as reference* snapshots your work.

## Data handling
- Any number of LAS files per well. Curves are matched by mnemonic aliases
  (e.g. GR/GRXS/GR_ARC/ECGR -> GR; HLLD/LLD/ILD/P40H -> deep resistivity).
- Runs are spliced; repeat passes (short duplicates of a mnemonic) are only used to fill gaps.
- Units normalised: depth ft -> m, NPHI % -> v/v, RHOB kg/m3 -> g/cc, DT us/m -> us/ft.
- Deviation survey (MD, INC, AZI) used for TVD when present (Kiwi-1 has one).
- Composited logs are cached in `<project>_cache\` so large projects reopen fast.
  *File > Reload logs* rebuilds the cache after you change LAS files.
- Extra aliases can be added to the `"extra_aliases"` block of the project file, e.g. `{"GR": ["MYGR"]}`.

## DLIS files (v0.5)
- DLIS (RP66 v1) is read directly, no extra packages: put files in `wells\<Well>\dlis\` and they load like
  LAS (every depth frame's scalar channels become curves; depth "0.1 in" etc. converted to metres; absent
  values -999.25 dropped). Tui-1 is now in the NZ project from its two DLIS files (2379-2561 m intermediate
  run, 3593-3729 m repeat pass).
- The big Tui-1 main-pass DLIS (171 MB, not in the repo) and Pateke-2's combined DLIS: convert or just drop
  them into the well's `dlis` folder. To convert to LAS 2.0 (one LAS per frame):
  `python -m well_correlation --dlis2las IN.dlis OUT_FOLDER` (`--arrays` also writes array channels such as
  waveforms, one column per element). `--dlisinfo FILE.dlis` lists frames and channels.
- Checkshots: `python -m well_correlation --checkshot TUI_1_MAIN_SURVEY_096LNP.DLIS out.csv` extracts the
  time-depth table (median of repeated shots, vertical times from the source offset) from CSAT/VSI files.

## Command line
    python -m well_correlation --selftest
    python -m well_correlation --check "C:\path\to\data folder"     curve inventory
    python -m well_correlation --png project.wcproj.json out.png    export without the GUI
    python -m well_correlation --dlisinfo FILE.dlis | --dlis2las IN.dlis OUT_DIR [--arrays] | --checkshot IN.dlis OUT.csv

## Next ideas
Amokura-1 (Tui area), seismic well ties (synthetics, in the Seismic Interpretation Tool),
depth conversion of maps with a velocity model.
