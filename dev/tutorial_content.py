"""Text of the GeoHarmonic illustrated tutorial (PLAN 38d).  dev/build_tutorial.py turns this into
docs/TUTORIAL.md + docs/TUTORIAL.html (and, on Chris's PC, tutorial_local/ with real window screenshots).

Rule (CLAUDE.md): every new module gets a chapter here; re-run `python3 dev/build_tutorial.py` after changing a window.

Chapter fields
  id, title       anchor + heading
  purpose         what it is for (one short paragraph, compared with Kingdom / Petrel / HRS / PaleoScan)
  shots           screenshot ids (see SHOTS in build_tutorial.py) shown under HOW IT LOOKS
  steps           HOW IT WORKS: numbered click-by-click steps
  results         RESULTS: text, then the numbers the tests verify (PASS lines matching these regexes)
  limits, trouble known limits / troubleshooting bullets
"""

INTRO = """\
This tutorial walks through every GeoHarmonic tool in the order you would use them on a project. Each chapter has
four parts: **How it looks** (screenshots of the real windows with numbered callouts), **How it works** (click by
click), **Results** (what you should see, plus the numbers the automatic tests prove on every build) and
**Known limits / troubleshooting**.

Menu paths are written like **Interpret > Velocity model + depth conversion...**. Keys are written like **Ctrl+S**.
"Seismic tool" means the *GeoHarmonic - Seismic Interpretation* window; "Well tool" means *GeoHarmonic - Well
Correlation*.

About the pictures: every screenshot is captured by `dev/build_tutorial.py` from the actual program on a small
synthetic demo project (3D cube, two horizons, three wells). Nothing is drawn by hand. When a window could not be
opened where the tutorial was built (the build server has no Qt), the caption says so and shows the tool's own
image export instead. When you run **Install GeoHarmonic.bat** on your PC it rebuilds the tutorial with real
window screenshots from your machine (see chapter 1) - the desktop shortcut *GeoHarmonic - Tutorial* opens that
copy.
"""

CHAPTERS = [
    # ------------------------------------------------------------------------------------------------ install
    dict(
        id="install", plain=True, title="1. Install, update, test",
        purpose="""\
Everything is started by double-clicking `.bat` files - no command window typing. The installer builds a private
Python environment (`.venv`) inside each tool folder, so nothing in your Anaconda install is changed.""",
        shots=[],
        steps=[
            "**After each build email: update.** Double-click **Update_GeoHarmonic.bat** on your Desktop. It pulls the "
            "latest `dev` branch into your GeoHarmonic folder. Wait for it to finish.",
            "**Install / repair.** In the GeoHarmonic folder double-click **Install GeoHarmonic.bat** (if Windows says "
            "*Windows protected your PC*: *More info* > *Run anyway*). It takes 5-15 minutes the first time. It "
            "installs numpy 1.x, matplotlib, PySide6 6.5.3 and pyqtgraph 0.13.7 for the two Qt tools, numpy / scipy / "
            "matplotlib for the Mistie Tool, runs every self-test and every GUI test, rebuilds this tutorial with "
            "screenshots from your PC and makes the desktop shortcuts.",
            "**Read the last lines.** The window ends with **ALL GOOD** or a list of problems. Problems are explained "
            "in `install_log.txt` (GeoHarmonic folder) and in `test_results.txt` inside each tool folder - send those "
            "files to Claude.",
            "**Desktop shortcuts** made by the installer: *GeoHarmonic - Well Correlation*, *GeoHarmonic - Seismic "
            "Interpretation*, *GeoHarmonic - Mistie*, *GeoHarmonic - Tutorial* (this document).",
            "**Re-run the tests only** (no reinstall): double-click **Run all tests.bat**. Results: `test_summary.txt` "
            "in the GeoHarmonic folder plus `test_results.txt` in each tool folder. Up to ~30 minutes.",
            "**F3 Netherlands demo project:** download *F3 Demo 2023* (dGB / TerraNubis), then double-click **Set up F3 "
            "project.bat**. It builds `Documents\\F3 Netherlands` (3D import, 4 wells with LAS + checkshots, 39 "
            "markers, 7 horizons, Fault A) in about 2 minutes.",
            "**Uninstall:** **Uninstall GeoHarmonic.bat** removes the `.venv` folders and shortcuts; your projects stay.",
        ],
        results="""\
The installer's last screen reads **ALL GOOD. Both tools are installed and every test passed.** The current test
counts on the build server are listed in chapter 2.""",
        results_re=[],
        limits=["The tools need Python 3.9-3.11 (PySide6 6.5.3 has no build for 3.12+). Anaconda with 3.11 is ideal.",
                "The first install needs internet (pip downloads); later runs only check the packages."],
        trouble=["*Package install failed*: check the internet connection / VPN, then run the installer again.",
                 "*Qt would not load*: send `install_log.txt` to Claude.",
                 "A GUI test that hangs is stopped by a watchdog after 10 / 20 minutes and reported as FAILED with "
                 "the place it was stuck - it never blocks the installer forever."],
    ),
    # ------------------------------------------------------------------------------------------------ fixes
    dict(
        id="fixes", plain=True, title="2. Fixes in this release (first PC install, 25 Nov 2026)",
        purpose="""\
Your first install (Anaconda Python 3.11.3, PySide6 6.5.3, pyqtgraph 0.13.7, numpy 2.4.6) gave:

| Tool | Self-test | GUI test (real Qt, off screen) |
|---|---|---|
| Well Correlation | PASSED | PASSED (218 checks, 0 failures) |
| Seismic Interpretation | PASSED (~1,000 checks) | **FAILED** at the 3D Viewer |
| Mistie | PASSED | - |

**Bug 1 - 3D Viewer crash (fixed).** The error was
`AttributeError: 'PySide6.QtOpenGL.QOpenGLShaderProgram' object has no attribute 'glClearColor'`.
In plain words: the 3D window kept a "phone number" for the graphics card's drawing functions. Qt does not let
Python own that number, and handed it to the next object it created (the shader program). The next time the window
painted, it "dialled" the shader program instead of the graphics card. **Change:** the 3D window now asks Qt for
the drawing functions every time it paints, resizes or starts (`gui/view3d.py`), and keeps nothing stale.

**Bug 1b - one crash hid the rest of the GUI test (fixed).** Because the test stopped at the 3D Viewer, the
checks after it (3D interpretation, framework + 3D grid, properties) never ran on your PC. **Change:** every test
module now runs inside a guard: a crash is written as a FAIL with its full error text and the next module still
runs (Seismic and Well Correlation tools, and `tools/guitest_runner.py`).

**Bug 2 - "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.6" at every start (fixed).**
PySide6 6.5.3 was built for NumPy 1. **Change:** the installers now ask for `numpy<2` for both Qt tools on every
run, so your existing `.venv` folders are moved back to numpy 1.26. The code was checked for NumPy-2-only calls.

**Noise - "QFontDatabase: Cannot find font directory" in the off-screen tests (fixed).** The GUI test runner now
points Qt at `C:\\Windows\\Fonts`.""",
        shots=["fix_after"],
        steps=[
            "Double-click **Update_GeoHarmonic.bat** on the Desktop.",
            "Double-click **Install GeoHarmonic.bat** in the GeoHarmonic folder. It downgrades numpy to 1.26 in both Qt "
            "tool environments (you will see *Installing numpy 1.x ...*) and re-runs every test.",
            "Wait for **ALL GOOD**. In `tools\\Seismic Interpretation Tool\\test_results.txt` search for `38a`: the "
            "lines *38a: real OpenGL - initializeGL, paintGL x2, resize, paintGL* and *grabFramebuffer returns the "
            "rendered image* must say PASS (or SKIP + *software fallback* on a PC without OpenGL).",
            "`install_log.txt` now shows *Qt OK - PySide6 6.5.3 - numpy 1.26.x* for both tools, and starting a tool "
            "no longer prints the NumPy 1.x warning.",
            "Open the Seismic tool, **View > 3D Viewer** (**Ctrl+Shift+3**): the window title bar must not mention "
            "*software rendering* on your RTX card; drag the planes, orbit, zoom.",
        ],
        results="""\
Before: the GUI test stopped with the AttributeError above. After (build server, Qt mock + software renderer):
the 3D Viewer draws the scene shown above, and the regression tests pass:""",
        results_re=[r"^PASS 38[abc]"],
        limits=["The build server has no graphics card: the real-OpenGL regression test only runs on your PC (it "
                "is part of Install GeoHarmonic.bat / Run all tests.bat)."],
        trouble=["If the 3D Viewer still fails on your PC, it now switches to *software rendering* with the reason "
                 "in the status bar instead of crashing - send `test_results.txt` to Claude."],
    ),
    # ------------------------------------------------------------------------------------------------ well corr
    dict(
        id="wellcorr", title="3. Well Correlation (+ petrophysics window)",
        purpose="""\
The well-log correlation and tops tool - think Kingdom's cross-section / log correlation window or Petrel's Well
Section, plus a Geolog / IP-style petrophysics window. You pick formation tops on a flattened or structural
section, let the tool suggest tops in the other wells, and compute Vsh, porosity, Sw and net pay.""",
        shots=["wc_main", "wc_section", "wc_petro"],
        steps=[
            "Start **GeoHarmonic - Well Correlation**. **File > New project from data folder...** (**Ctrl+N**) and pick "
            "the folder with your LAS files (one sub-folder per well works too), or **File > Open project...** "
            "(**Ctrl+O**). The F3 set-up makes `Documents\\F3 Netherlands\\F3 wells` for you.",
            "Click wells on the map (left) to put them on the section; drag them to reorder.",
            "**Edit > Pick tops** (**P**), choose the formation, click on a log track to place the top. **Ctrl+Z** / "
            "**Ctrl+Y** undo / redo.",
            "**Correlate > Suggest active top in selected well** (**S**) or *in all section wells missing it* "
            "(**Shift+S**); accept with **Enter**, reject with **Esc**. **G** shows a ghost curve at the cursor.",
            "**Correlate > Zone statistics...** (**Z**) for averages per zone; **Correlate > Compare tops with "
            "reference...** (**R**) to check against a published tops list.",
            "**Petrophysics > Petrophysics window...** (**Ctrl+Shift+P**): curve QC, Vsh, porosity, Sw (Archie / "
            "Simandoux / Indonesia), multimineral solve, Monte-Carlo uncertainty, net-pay cut-offs, rock physics.",
            "**File > Export section image (PNG/PDF)...** (**Ctrl+E**); **File > Export tops > Kingdom tops** or "
            "*OpendTect well markers* for the other packages. **Ctrl+S** saves.",
        ],
        results="""\
A section with your tops, correlation fill between wells and the petrophysics tracks. The self-test proves the
LAS / DLIS reading, top suggestion and petrophysics against hand-computed answers:""",
        results_re=[r"^PASS .*(suggest|autocorr|Archie|Vsh|net pay|porosity|DLIS).*\d"],
        results_tool="wc",
        limits=["Deviated wells are drawn along MD in the section (TVD shading in the petro window)."],
        trouble=["*Logs look flat / missing*: **File > Reload logs (rebuild cache)**.",
                 "A well with no logs is listed by the self-test check `--check <folder>` in the README."],
    ),
    # ------------------------------------------------------------------------------------------------ mistie
    dict(
        id="mistie", title="4. Mistie analysis (2D-2D, 2D-3D)",
        purpose="""\
Measures and solves the time, phase and gain misties between 2D lines and 3D surveys - the equivalent of Kingdom's
Mistie Analysis / Petrel's Seismic Well Tie mistie tools. There are two ways in: the stand-alone **GeoHarmonic -
Mistie** tool (for the NPRA-style 2D line sets) and **Interpret > Mistie analysis (2D-3D, 2D-2D)...** inside the
Seismic tool.""",
        shots=["mistie"],
        steps=[
            "Seismic tool: load your 2D lines (**File > Add 2D SEG-Y lines...**, **Ctrl+2**) and the 3D (**File > "
            "Import 3D SEG-Y...**, **Ctrl+3**).",
            "**Interpret > Mistie analysis (2D-3D, 2D-2D)...** finds every line crossing and measures shift, phase and "
            "gain by cross-correlation in the window you set.",
            "Tick **Robust (down-weight ties that disagree with the network)** so one bad tie cannot pull the network; hold a key line fixed. Options: **Flex mode**, **Tie overlapping / duplicate lines along the overlap**, **Time-variant shift for lines crossing a 3D**.",
            "Apply the solution; the corrected lines are used everywhere (sections, tracking, maps). The misties are "
            "drawn on the Live Map as coloured markers.",
            "Stand-alone tool: double-click **GeoHarmonic - Mistie**, open the config JSON (e.g. "
            "`examples_npra_config.json`), *Run*. Its tie table can be linked into the Seismic tool.",
        ],
        results="The tests build lines with known misties and recover them:",
        results_re=[r"^PASS .*(mistie|2D-3D tie|robust \(Huber\)|time-variant)"],
        limits=["Time-variant mistie only for 2D lines that cross a 3D."],
        trouble=["*No ties found*: the 2D navigation must be in the same coordinate system as the 3D "
                 "(**File > Add 2D navigation CSV...**)."],
    ),
    # ------------------------------------------------------------------------------------------------ seismic
    dict(
        id="seismic", title="5. Seismic viewer, horizon + fault picking, autotracking",
        purpose="""\
The main interpretation window - Kingdom's Vertical Seismic View + Map View in one. Inline / crossline / time slice
/ 2D line display, horizon picking with seeded autotracking, fault sticks, attributes, co-blending and wells on
the section.""",
        shots=["seis_main", "seis_section"],
        steps=[
            "Start **GeoHarmonic - Seismic Interpretation**. **File > New project...** (**Ctrl+N**), then **File > "
            "Import 3D SEG-Y...** (**Ctrl+3**) and/or **File > Add 2D SEG-Y lines...** (**Ctrl+2**). Big files are "
            "read from disk on demand (no full load).",
            "Pick the line in the toolbar (Inline / Crossline / Time slice and its number). **PgDown / PgUp** step to "
            "the next / previous line.",
            "Horizons panel: **New** (or **Interpret > New horizon...**), then **Pick mode (P)** and click seeds on "
            "the reflector.",
            "**Track (T)** (**Interpret > Track horizon from seeds**) autotracks peak / trough / zero crossing from "
            "your seeds across the 3D and onto crossing 2D lines; it stops at faults. **Ctrl+Z** undoes the last "
            "edit or tracking run.",
            "Faults panel: **New (N)**, **Fault mode (F)**, click points of a stick, **End stick (Enter)**.",
            "**File > Link well project...** shows the Well tool's wells, logs and tops on the section.",
            "**View > Co-blend attributes on the section (RGB / CMY / HSV)...**, **View > Attribute overlay...** and "
            "**View > Bump-map the section / slice** for SeisWare-style displays. **Ctrl+S** saves.",
        ],
        results="The tracker is tested on folded, noisy and faulted synthetic data:",
        results_re=[r"^PASS (3D autotrack|no tracking|lateral mix|with the fault|without faults)"],
        limits=["Tracking from the 3D window is described in chapter 15; both write the same horizon."],
        trouble=["*Tracking runs through noise*: raise **Min correlation** in the Horizons panel; set **Lateral mix "
                 "2D** to 5-9 traces for noisy land 2D; keep **Tracker stops at faults** ticked."],
    ),
    # ------------------------------------------------------------------------------------------------ rgt
    dict(
        id="rgt", title="6. RGT, horizon stack, Wheeler, stratal slices",
        purpose="""\
PaleoScan-style relative geologic time: a dip field, a global RGT solve, then hundreds of iso-RGT horizons (the
horizon stack), a Wheeler (chronostratigraphic) display and stratal slices.""",
        shots=[],
        steps=[
            "**Interpret > Compute dip field (current 3D volume)...**.",
            "**Interpret > Compute RGT (current 3D volume)...** (or *all 2D lines*). Optional constraints: **Interpret > "
            "Use active horizon as RGT constraint**, **Well seeds from tied tops**.",
            "**Interpret > Build horizon stack (3D)...**; browse the levels on the section; **Interpret > Convert stack "
            "level to horizon...** keeps one as a normal horizon.",
            "Fix the model where needed: **Interpret > RGT edit mode on/off** (**E**) - click a level, then click where "
            "it belongs; **Re-link RGT around active fault** for cross-fault links.",
            "Wheeler: set the section's domain box to **Wheeler** (vertical axis = relative geologic time, gaps = hiatus) "
            "or **Wheeler (age)** (Ma, from the Basin model age table); tick **RGT contours** to draw the stack levels.",
        ],
        results="RGT accuracy on synthetic geology with a fault and a pinch-out:",
        results_re=[r"^PASS (iso-RGT|with a fault: iso-RGT|RGT solved|Wheeler|fault re-link|moved level)"],
        limits=["The RGT is stored as uint16 per sample (precision about 1/65,000 of the time range)."],
        trouble=["*Levels jump across a fault*: pick the fault first, then **Re-link RGT around active fault**."],
    ),
    # ------------------------------------------------------------------------------------------------ welltie
    dict(
        id="welltie", title="7. Well tie (synthetic seismogram)",
        purpose="""\
Hampson-Russell-style well tie: synthetic from sonic + density, auto-correlate (bulk shift + stretch), constrained
stretch / squeeze (DTW), wavelet estimation, drift QC against the checkshot, and multi-well batch ties.""",
        shots=["welltie"],
        steps=[
            "Link the well project (**File > Link well project...**), then **Interpret > Well tie (synthetic "
            "seismogram)...** and pick the well.",
            "**Find best trace** searches around the well; **Auto tie** gives bulk shift + stretch with the correlation "
            "before / after.",
            "**Auto stretch/squeeze** (with a velocity-change cap) or **Pick event pair** for manual ties; **Undo** / "
            "**Clear warp**.",
            "**Estimate wavelet** / **Save wavelet**; **Drift QC** compares the tied T-D with the checkshot.",
            "**Save tie** writes the time-depth table used by depth conversion and the velocity volume. **Tie all "
            "wells...** runs the batch.",
        ],
        results="Known shifts and velocity errors are recovered:",
        results_re=[r"^PASS (auto tie|statistical|DTW|multi-well batch)"],
        limits=["Sonic-only wells use a Gardner density."],
        trouble=["*Poor correlation*: despike logs (**Despike logs**), check the polarity, use a window tie."],
    ),
    # ------------------------------------------------------------------------------------------------ velocity
    dict(
        id="velocity", title="8. Velocity model, dynamic depth conversion, velocity-volume mode",
        purpose="""\
Kingdom Dynamic Depth Conversion: a layer-cake velocity model from horizons + tops (V0+kZ, interval velocities,
checkshots), residual gridding so the depth maps tie every top, scenarios and blind tests - and an advanced
**velocity-volume** mode built from the well-tie velocities (tops stay the hard data).""",
        shots=["velocity", "velvolume"],
        steps=[
            "**Interpret > Velocity model + depth conversion...**: list the layers (time horizon + matching top), pick "
            "the method per layer, **Build model**.",
            "**Depth map of selected layer**, **Cross-plot of selected layer**, **Export misfit CSV...**; **Use for "
            "depth display** switches sections to depth.",
            "What-if wells: **Add pseudo-top**; scenarios: **Save scenario**, **Difference map vs scenario**.",
            "Dynamic: **View > Auto-update depth model (dynamic depth conversion)** rebuilds the maps whenever you pick "
            "a top or change a horizon.",
            "Advanced: **Interpret > Velocity volume depth conversion (advanced)...** - **a. Well velocities** (from "
            "the saved well ties, QC flags), then **b-e. Build + tie to tops** (layering, interpolation, optional "
            "stacking-velocity trend, residual gridding). **Blind test vs layer-cake**, **Store volume**, **Maps to "
            "map hub**.",
            "**Depth-convert volume to SEG-Y...** writes a depth cube.",
        ],
        results="Depth maps tie the tops exactly; the volume mode is tested against the layer-cake:",
        results_re=[r"^PASS (37[cdefi]|residual correction ties|depth of H2|interval velocity of layer)"],
        limits=["The velocity-volume model is not yet used by *Use for depth display* in the section (follow-up)."],
        trouble=["*Maps do not tie a top*: check the top is in the layer's top list and not flagged as an outlier."],
    ),
    # ------------------------------------------------------------------------------------------------ maps
    dict(
        id="maps", title="9. Map hub, Live Map and the map bus",
        purpose="""\
The Kingdom-style base map. The **Map layers** panel (left of the Seismic tool) holds grids, wells, culture,
shapefiles, petrophysics bubbles / pies, cokriging and the GIS / QGIS link. **View > Live Map** is a dark,
Spotfire-style map with styled wells, filters, lasso selection and a linked table / crossplot / histogram. The
**map bus** keeps every window in sync: select a well in the Well tool and it lights up in the Live Map and the 3D
Viewer; move the inline in the Seismic tool and the map line moves.""",
        shots=["livemap", "horizon_map"],
        steps=[
            "**Interpret > Horizon maps (structure, isochron, amplitude, depth)...** (**Ctrl+M**) grids a horizon onto "
            "the base map.",
            "In **Map layers** click **Add...** for grids, CPS-3 files, petrophysics per well, mineral pies, "
            "shapefiles / culture (leases, PLSS, pipelines, no-go), DHI checklist, migration, grid calculator, "
            "cross-check, EPSG and GIS export.",
            "**View > Live Map (styled wells, filters, linked table + crossplot)...**: in *Well style* choose *Colour "
            "by* / *Size by* / *Shape by* / *Label* columns (rules such as *mistie > 10 ms = red* under *Rules*), "
            "**Apply style**.",
            "*Filters*: **Add filter** (range slider or category ticks) hides wells instantly; **Clear filters**.",
            "**Lasso select** (click around wells, double-click to close) - the table, crossplot and histogram below "
            "follow; **Section from selected wells** opens them in the Well tool.",
            "**Arbitrary line** (click points, double-click the last) opens an arbitrary section in the Seismic tool.",
            "**Live grid...** grids a WellTable column or picks while you work; **Live depth maps** follow the "
            "dynamic depth model.",
            "**Export map (PNG / PDF)...** with title block + legend; **Export GIS (shapefile, GeoJSON, ZMAP+)...**.",
        ],
        results="Map bus, styling, filters and gridding are tested without and with the GUI:",
        results_re=[r"^PASS 31[a-j]: .*\d"],
        limits=["Polygon fills ignore holes; the Dashboard keeps its own selection (not yet on the bus)."],
        trouble=["*Wells do not light up across tools*: both tools must open the same well project (the bus files "
                 "live in `<well project>_mapbus`)."],
    ),
    # ------------------------------------------------------------------------------------------------ basin
    dict(
        id="basin", title="10. Basin model (burial, thermal, maturity) + Wheeler",
        purpose="""\
A 1D / pseudo-3D basin model in the spirit of PetroMod 1D / Genesis: decompaction and burial history, heat flow
and temperature, EASY%Ro maturity, timing of generation, all linked to the Wheeler diagram.""",
        shots=["basin"],
        steps=[
            "**Interpret > Basin model (burial, thermal, maturity, Wheeler)...**.",
            "Tab **Age table**: layers with ages, lithologies, erosion; tab **Wells**: the wells to model, heat flow; run. Tick **Re-run when the depth model changes** to keep it live.",
            "Tab *Maps / Wheeler / events*: maturity maps, **Colour Wheeler (age)**, the events chart (source, "
            "reservoir, seal, trap, generation).",
        ],
        results="Checked against analytic solutions:",
        results_re=[r"^PASS (Athy|decompacted|steady state|EASY|18[a-e])"],
        limits=["1D per well / grid node (no lateral heat or fluid flow)."],
        trouble=["*Ro too high*: check the erosion amounts and heat flow history."],
    ),
    # ------------------------------------------------------------------------------------------------ inversion
    dict(
        id="inversion", title="11. Seismic inversion (+ resolution, AVO / DHI)",
        purpose="""\
Hampson-Russell STRATA-style post-stack and pre-stack inversion: low-frequency model (LFM) from wells and horizons,
model-based / sparse-spike post-stack inversion, simultaneous pre-stack inversion (AI, SI, Vp/Vs), plus a
resolution / tuning check, single-well inversion and AVO / DHI / fizz-gas tools.""",
        shots=["inversion"],
        steps=[
            "**Interpret > Seismic inversion (LFM, post-stack, pre-stack)...**.",
            "Tab **Low-frequency model**: wells + horizons as the framework (or **along RGT stack levels instead**); tab **Post-stack** or **Pre-stack** to run; results are new "
            "volumes you can show on sections, in the 3D Viewer and as map layers.",
            "**Resolution...** asks for the window, interval velocity and bed thicknesses and says which beds are "
            "resolved.",
            "**AVO volumes (intercept, gradient, anomaly, fluid factor)** from a near and a far stack.",
        ],
        results="Inversion and AVO are tested on synthetic earths with known answers:",
        results_re=[r"^PASS (19[a-e]|2[45][a-e]).*\d"],
        limits=["Pre-stack gathers are not stored yet - angle stacks are used."],
        trouble=["*Blocky result*: lower the sparsity weight or add a horizon to the LFM."],
    ),
    # ------------------------------------------------------------------------------------------------ analytics
    dict(
        id="analytics", title="12. Analytics + Dashboard",
        purpose="""\
Transform / Spotfire-style data science: an attribute table (wells, laterals, grid values), feature engineering,
models with honest validation (spatial blocks), SHAP-style explanations and a QC report; and a Dashboard that colours
wells / laterals by any attribute, with spacing ellipses and a gun-barrel view.""",
        shots=["analytics", "dashboard"],
        steps=[
            "**Interpret > Analytics (attribute table, models, SHAP)...**: build the table, pick target + inputs, "
            "train, read the *Model QC* report.",
            "**Interpret > Dashboard (wells, laterals coloured by attribute)...**: **Wells from project**, **Add "
            "basemap grid**, **Stage CSV...**, colour by a column, **Send to Analytics**.",
            "Reservoir engineering from the Dashboard toolbar: **Spacing + ellipses...** (SHmax azimuth asked), "
            "**Gun barrel**.",
        ],
        results="The models are validated on synthetic data with known drivers:",
        results_re=[r"^PASS .*(spatial|SHAP|model|R2|r2).*\d"],
        limits=["Optional xgboost / scikit-learn / shap via *Install analytics extras (optional).bat*; numpy "
                "models otherwise."],
        trouble=["*Too-good model*: use the spatial block validation, not random splits."],
    ),
    # ------------------------------------------------------------------------------------------------ faults
    dict(
        id="faultmap", title="13. Automated fault mapping",
        purpose="""\
PaleoScan AFE-style automatic faults: fault-likelihood, thinned fault likelihood (TFL), strike and dip volumes,
fault skins, a stereonet / rose QC, faults with sticks for the interpretation, throws and fault polygons on
horizons (used as gridding barriers).""",
        shots=["faultmap"],
        steps=[
            "**Interpret > Automated fault mapping (fault likelihood, auto faults, polygons)...**.",
            "**1. Fault likelihood volumes** (note the run time on F3).",
            "**2. Extract faults**: set *TFL threshold* and *Minimum fault size (points)*; check the stereonet / "
            "strike rose; filter and merge in the table.",
            "**3. Create faults (sticks)** - they appear in the Faults panel like hand-picked faults.",
            "**Fault polygons on active horizon** for the map; **2D lines: fault cuts + correlation** for 2D.",
        ],
        results="Faults with known strike / dip / throw are recovered:",
        results_re=[r"^PASS 32[a-h]: .*\d"],
        limits=["Orientation from a structure tensor (not a brute-force scan) - same answer, much cheaper."],
        trouble=["*Too many small faults*: raise the minimum size or the TFL threshold."],
    ),
    # ------------------------------------------------------------------------------------------------ 3D viewer
    dict(
        id="view3d", title="14. 3D Viewer",
        purpose="""\
A Petrel-style 3D window: inline / crossline / time-slice planes through the volume, horizons as surfaces, faults,
wells with tops, culture, vertical exaggeration and saved views. It uses your graphics card (OpenGL) and falls back
to software rendering when OpenGL cannot start (it says so in the title).""",
        shots=["view3d"],
        steps=[
            "Seismic tool: **View > 3D Viewer (slices, horizons, faults, wells)...** (**Ctrl+Shift+3**).",
            "Mouse: left drag = orbit, middle drag or **Shift**+left = pan, wheel = zoom; **Fit** resets.",
            "Move a plane with the *Inline* / *Crossline* / *Time slice* sliders - coarse while dragging, full "
            "resolution when you let go; the Seismic tool follows (and vice versa).",
            "*Objects* tree: tick objects on / off, set opacity; the **VE** box in the toolbar sets vertical exaggeration.",
            "**Save view**, **Screenshot...**, **Turntable GIF...**.",
        ],
        results="Planes, meshes, picking and the map-bus links are tested; on your PC the OpenGL path too:",
        results_re=[r"^PASS 33[a-e]"],
        limits=["Very large planes are drawn with per-vertex colours (decimated while dragging)."],
        trouble=["Title says *software rendering*: OpenGL could not start (remote desktop / driver). Update the "
                 "NVIDIA driver; the reason is in the status bar."],
    ),
    # ------------------------------------------------------------------------------------------------ glow
    dict(
        id="glow", title="14b. Glow / Halo on horizons, faults and wells",
        purpose="""\
A PowerPoint-style glow (or a solid contrasting halo) under interpretation lines so horizons, faults and wells stand
out on busy seismic - in the section, the Live Map and the 3D Viewer. The active / selected object glows stronger;
selecting a horizon, fault or well in any window makes it glow in every window (map bus). Kingdom and Petrel have no
equivalent; it is closest to Petrel's "highlight selected object".""",
        shots=["glow_off", "glow_on", "glow_halo"],
        steps=[
            "Seismic tool: press **G** (or **Display > Glow on/off**) to switch glow on / off everywhere.",
            "**Display > Glow settings (mode, width, intensity, colour)...**: *Mode* Glow / Halo / Off, *Width* "
            "(pixels), *Intensity*, *Falloff layers* 1-4, *Glow colour* (match the object / white / black / custom), "
            "*Halo colour* (Auto picks black or white against the current colour table), *Glow only the active / "
            "selected object*, *Hover highlight*, *Exports include glow*. **Apply** or **OK**; saved with the project "
            "(**Ctrl+S**).",
            "Quick switches: **Display > Glow mode: Glow**, **Display > Glow mode: Halo**, **Display > Glow only the "
            "active / selected object**.",
            "Per object: right-click a horizon in the *Horizons* panel or a fault in the *Faults* panel (or an object "
            "in the 3D Viewer's *Objects* tree) > **Glow** > *On* / *Off* / *Use global* / *Colour...*.",
            "Move the mouse over a horizon or fault on the section: it highlights (hover).",
            "**File > Export section image (PNG / PDF / SVG)...** (**Ctrl+E**): choose *With glow (as shown)* or "
            "*Export without glow*. The 3D Viewer's **Screenshot...** asks the same; **Turntable GIF** follows the "
            "*Exports include glow* setting.",
            "3D Viewer: **Glow (G)** in the toolbar; surfaces get an outline glow, the active / selected one also a "
            "soft shell; wells a wider transparent tube; the three plane outlines glow.",
        ],
        results="The pictures above are the tool's own section export (before / Glow / Halo). The tests check:",
        results_re=[r"^PASS 39[a-i]"],
        limits=["Pictures here are on the small demo cube; on F3 open an inline with your horizons and a fault and "
                "press G to compare.",
                "Wide glow lines in 3D depend on the graphics driver; when it cannot draw them the glow is 1 px wide "
                "or off, with a one-line note in the 3D status bar (never a crash).",
                "More than 2,000 glowing objects (e.g. thousands of fault sticks): glow switches to the active / "
                "selected objects only and the status bar says so."],
        trouble=["*Nothing glows*: press **G** once (glow may be off), check **Display > Glow only the active / "
                 "selected object** is unticked, and that the object has no per-object *Off*.",
                 "*Glow looks too heavy*: Glow settings > Width 3-4 px, Intensity 50 %, or switch to Halo."],
    ),
    # ------------------------------------------------------------------------------------------------ aesthetics
    dict(
        id="aesthetics", title="14c. Aesthetics: hillshade, colour tables, co-rendering, themes, figures",
        purpose="""\
Display polish shared by every window: sun-lit (hillshaded) maps and 3D surfaces, one colour-table library (seismic
bars incl. Kingdom-style, perceptual viridis / cividis / magma-style tables, colourblind-safe ones marked, your own
tables saved in the project), co-rendering of a second volume over the amplitude (like Petrel's co-render /
Kingdom's overlay), dark and light themes, smooth camera moves, styled well-log tracks and one-click figures with a
title block for slides and reports.""",
        shots=["aes_hillshade", "aes_corender", "aes_dark", "aes_light", "aes_wc"],
        steps=[
            "**Sun / hillshade (maps):** Live Map toolbar **Sun...**: drag the sun on the dial (direction = azimuth, "
            "nearer the centre = higher), set *Vertical exaggeration*, *Blend with colour*, *Multi-directional*. Every "
            "grid on the map is lit; the 3D Viewer's horizons are lit by the same sun.",
            "**Colour tables:** the colour box above the section lists the library; colourblind-safe tables are "
            "marked. Custom tables are saved with the project (export / import as JSON or a 256-row CSV).",
            "**Co-render:** **View > Co-render a volume / attribute (alpha, multiply, transparency)...**: pick a "
            "second volume (e.g. coherence) or an attribute, the blend mode, its colour table and opacity. *Multiply* "
            "darkens the seismic where coherence is low; *transparency* hides low values. In 3D the same on the planes.",
            "**Theme:** **View > Theme > Dark** or **Light (print / slides)** - switches immediately, remembered next "
            "time. **View > Anti-aliasing** off = faster drawing on huge sections.",
            "**3D display:** 3D Viewer toolbar **Display...**: sun on/off, eye-dome lighting, fog, shadows (falls back "
            "to plain lighting), fly-to speed (Off / Fast / Normal / Slow). Selecting a well or horizon on the Live Map "
            "flies the 3D camera to it; any mouse click stops the move. Live Map **Zoom to selection** eases in.",
            "**Well tracks (Well tool):** **View > Track style**: GR gradient fill, lithology patterns, net / pay flags "
            "(Vsh / PHIE / Sw cutoffs), neutron-density crossover; **Save track style as template...** / **Apply track "
            "template...**.",
            "**Figures:** Seismic tool **File > Export section figure (title block, scale bar, legend)...**: page (16:9 "
            "slide, Letter, A4, poster), title; PNG / PDF / SVG. The 3D Viewer has the same for its view.",
            "**Automatic colours:** new horizons / faults get colours that differ from each other and from the seismic "
            "colour table; right-click > **Lock colour**; **Display > Reassign horizon + fault colours**.",
        ],
        results="Pictures: the tools' own exports on the demo data. The tests check:",
        results_re=[r"^PASS 40[a-j]"],
        limits=["Eye-dome lighting and soft shadows are drawn by the software renderer (screenshots, fallback view); "
                "the OpenGL view shows plain lighting + fog.",
                "Theme: plot windows restyle immediately; a few dialogs keep their colours until reopened.",
                "High-DPI: tested here as maths (100 / 150 / 200 %); check on your 4K screen.",
                "ICS colours cover epochs / periods commonly mapped (Holocene to Cambrian)."],
        trouble=["*Map looks flat*: Sun... > raise Vertical exaggeration (time maps in ms need 3-10).",
                 "*Theme did not change a window*: close and reopen that window."],
    ),
    # ------------------------------------------------------------------------------------------------ conditioning
    dict(
        id="conditioning", title="14d. Seismic conditioning (spikes, bandpass, FX decon, SOF, footprint)",
        purpose="""\
Post-stack conditioning like Petrel's structural smoothing / PaleoScan's SOF / OpendTect's dip-steered median: a
chain of steps with presets, a live preview with the difference panel, protect zones for geology you must not
smooth, and a full-volume run in the background that writes a NEW volume (the original is never changed). Your F3
test (2 passes, +/-2 IL x +/-3 XL) removed ~47 % of the RMS incl. real small faults at 1200-1350 ms - so the default
is 1 pass, edge protection on, a strength slider, and the difference panel is always shown.""",
        shots=["conditioning"],
        steps=[
            "Seismic tool: open the line you want to judge (e.g. F3 inline 425), then **Seismic > Conditioning (spikes, "
            "bandpass, FX decon, SOF, footprint)...**.",
            "*Input volume*: the volume (its conditioning history is shown in brackets). *Preset*: **Interpretation** "
            "(spike removal > bandpass > FX decon > SOF > footprint), **Light (for inversion)** (spikes > bandpass - "
            "the inversion window picks these volumes by default and warns on SOF volumes) or **Fault-attribute "
            "input** (spikes > SOF with strong edge protection).",
            "Edit the chain: tick steps on / off, **Add** / **Remove** / **Up** / **Down**; select a step to see its "
            "parameters (JSON) - e.g. SOF `half_il`, `half_xl`, `filter` (median / alpha / mean), `passes`, `guard` "
            "(planarity threshold), `guard_mode` (shrink / skip), `kuwahara`, `strength` (0-1); bandpass `kind` "
            "(ormsby / butterworth) and `f` corners; FX decon window, `filter_len`, `fmin` / `fmax`, `mix`; then "
            "**Apply parameters**. **Save preset...** keeps the chain in the project.",
            "**Preview**: original | conditioned | difference on one colour scale (zoom one, all follow), spectra "
            "before / after, *RMS removed %* and *signal leakage*. A warning appears when more than 35 % of the RMS "
            "is removed or the difference shows reflectors - then lower the strength or add a protect zone.",
            "**Add zone...**: inline / crossline range + time gate + strength inside (0 = skip) - e.g. over the "
            "1200-1350 ms small-fault interval. Zones go to the Live Map as polygons.",
            "**Run full volume...**: name of the new volume; tick *Also write the difference volume* / *Export SEG-Y*. "
            "Runs in the background (progress, time left, Cancel in the status bar); the new volume appears in the "
            "seismic window, 3D Viewer, attributes, tracking and fault mapping.",
        ],
        results="The picture is the preview computation on the noise-free demo cube - a case for the warning: there "
                "is no noise to remove, the demo reflectors sit on whole samples (staircase), and the difference "
                "panel shows reflector energy, so the preview says *too aggressive* (RMS removed > 35 %). On real, "
                "noisy data the difference should look like random noise. The tests (synthetic cube with dipping "
                "reflectors, noise, a fault, footprint stripes and spikes) check:",
        results_re=[r"^PASS 41[a-n]"],
        limits=["SEG-Y export writes GeoHarmonic's standard headers (inline / crossline / X / Y), not a copy of the "
                "original trace headers.",
                "Anisotropic diffusion (alternative SOF engine) is not built; AGC is available as a display-only step.",
                "Protect zones are rectangles in inline / crossline + a time gate (drawn polygons: next version).",
                "Full F3 run time: expect tens of minutes for SOF on 2 cores (the test cube runs in seconds)."],
        trouble=["*Difference shows reflectors*: strength 50-70 %, 1 pass, guard 0.6 with `skip`, or a protect zone.",
                 "*Preview is slow*: the preview reads +/-12 inlines around the line; use a smaller SOF window."],
    ),
    # ------------------------------------------------------------------------------------------------ survey blend
    dict(
        id="surveyblend", title="14e. Survey horizon and grid blending (overlapping surveys)",
        purpose="""\
Two unmerged surveys that overlap never tie perfectly, even after the mistie correction (shift / phase / amplitude)
and depth conversion with each survey's velocities. A priority cut leaves a step at the seam; a fixed-width feather
smears geology. This blends only the RESIDUAL misfit with an adaptive width: width = |residual| / the slope you
allow (e.g. 2 m/km: a 4 m misfit gets 2 km, a 20 m misfit 10 km). Faults are barriers, so throw is kept, and well
tops are re-tied exactly afterwards.""",
        shots=["surveyblend"],
        steps=[
            "Bulk-tie the surveys first (**Interpret > Mistie analysis**), and set up the velocity model "
            "(**Interpret > Velocity model + depth conversion**) if you blend in depth.",
            "**Interpret > Survey horizon blending (adaptive seam, overlapping surveys)...**: choose the *Horizon*, "
            "*Survey A (priority)* and *Survey B*, *Domain* (Depth via the velocity model, or Time).",
            "*Method*: **Hybrid** (default: residual surface into survey B, then a narrow feather), *Residual surface*, "
            "*Feather*, or *Priority cut* (no blending). *Allowed added slope* (default 2) in m/km, ft/mile or ms/km; "
            "*Min / Max width* and *Edge trim* (outer traces dropped) under Advanced.",
            "Tick *Faults are barriers* and *Re-tie well tops* (uses the WellTable column `Top <horizon>`).",
            "**Preview**; *Show*: merged grid, residual before, weight, width, added slope, correction; untick "
            "*Show blended* to compare with the priority cut. The readout says e.g. *Blend width 0.8-6.2 km, overlap "
            "fully used on 87 % of seam* - wide = the surveys disagree there (an interpretation flag).",
            "**Export...**: merged grid + QC maps as ZMAP+ files; the merged grid, width and added-slope maps are "
            "added to the map hub (Live Map) layers.",
        ],
        results="The picture is the build's synthetic test case. The tests check:",
        results_re=[r"^PASS 42[a-n]"],
        limits=["Blends two surveys at a time (merge the result with the next survey).",
                "Depth mode uses one velocity model for both surveys (per-survey velocities: next version); it is "
                "not yet a step inside the depth-conversion dialog.",
                "Fault barriers come from the fault sticks' top points (one trace per fault)."],
        trouble=["*Width at the maximum everywhere*: the surveys still differ a lot - check the mistie tie first.",
                 "*'gap' in the readout*: part of the seam has no overlap - that stretch is interpolation, not "
                 "blending."],
    ),
    # ------------------------------------------------------------------------------------------------ 3D interp
    dict(
        id="interp3d", title="15. 3D interpretation (picking, tracking, volume rendering, geobodies)",
        purpose="""\
Interpret directly in 3D: pick horizon points and fault sticks on the planes, track from 3D seeds, render an
amplitude / attribute volume with an opacity curve, and extract geobodies (connected bodies above a threshold) -
like Petrel's 3D interpretation and geobody tools.""",
        shots=[],
        steps=[
            "In the 3D Viewer toolbar: **Pick horizon...** (choose or type a name), click on a plane (a click without "
            "dragging picks). **Stop picking** when done.",
            "**Pick fault...**, click stick points, **End stick**.",
            "**Track** grows the horizon from your 3D seeds (same tracker as the section).",
            "**Volume...** shows another volume on the planes; **Volume render** draws a box of the volume with an "
            "opacity curve.",
            "**Geobodies...**: *Keep values above:*, run; bodies are listed with volume / cell count, drawn in 3D and "
            "sent to the Live Map as outlines.",
        ],
        results="",
        results_re=[r"^PASS 34[a-e]"],
        limits=["Tracking growth is shown when the run ends (not live)."],
        trouble=["*Clicks rotate instead of picking*: click without moving the mouse."],
    ),
    # ------------------------------------------------------------------------------------------------ framework
    dict(
        id="framework", title="16. Structural framework + 3D grid",
        purpose="""\
Petrel's structural framework + 3D grid in a simple stair-stepped form: fault traces, horizons gridded per fault
block with conformable / erosional rules, depth conversion, then a 3D grid (zones, layering, fault-block index)
exported as Eclipse GRDECL for ResInsight / OPM Flow.""",
        shots=["framework"],
        steps=[
            "**Interpret > Structural framework + 3D grid (GRDECL)...**.",
            "**1. Framework (time)** - horizons per fault block, rules, no-crossing checks.",
            "**2. Depth (current depth model)** - uses the layer-cake or the velocity-volume model.",
            "**3. 3D grid** - cell size, layers per zone; the quality report lists zero-thickness cells.",
            "**Export GRDECL...** (ZONE / FAULTBLK / FIPNUM included); open it in ResInsight.",
            "**4. Properties + volumetrics...** opens chapter 17 on this grid.",
        ],
        results="",
        results_re=[r"^PASS 35[a-f]"],
        limits=["Stair-stepped faults with vertical pillars; full corner-point grids are out of scope."],
        trouble=["*Negative thickness cells*: check the horizon rules (erosional / base)."],
    ),
    # ------------------------------------------------------------------------------------------------ properties
    dict(
        id="properties", title="17. Property modelling + volumetrics",
        purpose="""\
Petrel property modelling in brief: upscale logs into the grid, variograms per zone, facies (SIS / indicator
kriging) and petrophysics (kriging / SGS) with fixed-seed realizations, then GRV / NTG / porosity / Sw / Bo
volumetrics per zone and fault block with P90 / P50 / P10.""",
        shots=["propmodel"],
        steps=[
            "Open from the framework dialog: **4. Properties + volumetrics...**.",
            "**Upscale logs** (arithmetic / harmonic / geometric / most-of), check the raw vs upscaled plot.",
            "**Variogram** per zone, then **Simulate (10 realizations)**.",
            "**Volumetrics per block** with contacts and Bo / Bg; **Export GRDECL / CSV...**.",
        ],
        results="",
        results_re=[r"^PASS 36[a-f]"],
        limits=["SGS / SIS are per-cell loops - fine to ~100,000 cells."],
        trouble=["Sw is constant 0.3 when no Sw log is upscaled."],
    ),
    # ------------------------------------------------------------------------------------------------ prospects
    dict(
        id="prospects", title="18. Prospects, migration, seal, spacing, depletion",
        purpose="""\
Exploration and development tools built on the map hub: DHI checklist and Rose-style prospect risking +
volumetrics, migration (flow paths, traps, fill-spill), fault / top seal, structural timing, well spacing with
drainage ellipses and gun-barrel views, depletion / remaining resource and an M&A screen.""",
        shots=[],
        steps=[
            "**Map layers > Add... > DHI checklist (amplitude grid vs structure grid)...** - scores the DHI evidence "
            "and stores a prospect row (risk + volumetrics).",
            "**Map layers > Add... > Migration: flow paths, traps, fill-spill on the selected depth grid**.",
            "**Interpret > Paleo-structure animation of this section (GIF)...** for structural timing.",
            "Dashboard toolbar: **Spacing + ellipses...**, **Gun barrel**.",
        ],
        results="",
        results_re=[r"^PASS (2[6-9]|30)[a-h]?: .*\d"],
        limits=["There is no separate prospect window yet (prospects live in the project and the map hub)."],
        trouble=[],
    ),
    # ------------------------------------------------------------------------------------------------ rebuild
    dict(
        id="rebuild", plain=True, title="19. Rebuilding this tutorial",
        purpose="""\
The tutorial is generated, never edited by hand: `dev/build_tutorial.py` builds the demo project, opens every
window, captures the screenshots, adds the callouts and writes `docs/TUTORIAL.md` + `docs/TUTORIAL.html`. On your
PC, **Install GeoHarmonic.bat** runs it with real Qt and writes `tutorial_local\\TUTORIAL.html` (the desktop
shortcut opens that copy).""",
        shots=[],
        steps=["Nothing to do - the installer rebuilds it. Claude re-runs it after every change to a window."],
        results="", results_re=[], limits=[], trouble=[],
    ),
]
