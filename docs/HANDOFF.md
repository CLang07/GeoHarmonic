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
- GUI (`gui/`): `main_window.MainWindow` (File menu: new/open/save, Import 3D (threaded, QTimer poll),
  Add 2D lines, Add nav CSV, Link well project; docks: Data tree, Basemap), `section_panel.SectionPanel`
  (ImageItem row-major (rows = time), toolbar: source combo, Inline/Crossline/Time slice, number spin +
  slider + step, colormap, clip %, reverse polarity; PgUp/PgDn; signals positionChanged, cursorInfo;
  `current_trace_line()` for basemap), `basemap.Basemap` (outlines, 2D lines, wells, red current line;
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

## Conventions
- numpy only for algorithms; vectorise; memmap big arrays; float16 stores for volumes, float32 math.
- GUI: PySide6 imported before pyqtgraph; `pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")`;
  heavy work in a thread with progress polled by QTimer; undo for edits; keyboard shortcuts listed in Help.
- Every feature gets a selftest check (PASS/FAIL lines, final "SELFTEST PASSED").
- Time in ms in the UI, seconds in meta. TWT axis inverted (down).
- Units: metres; NZ coords NZTM2000 (EPSG:2193); NPRA nav in its own projected metres.

## Testing GUI code (no Qt here)
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
- Chris wants Kingdom/GeoGraphix-style behaviour and speed with big datasets; avoid per-trace Python loops.
