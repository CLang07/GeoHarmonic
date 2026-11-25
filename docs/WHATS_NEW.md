# What's new (for Chris) - Geo Tool Kit

Built unattended 23 Nov 2026 on branch `dev`. Everything below passed the numpy self-tests and the GUI-logic
tests under the Qt mock, but **no real Qt window has been opened yet** - the GUI is the riskiest part.

## Start here: test checklist (highest risk first)
1. Pull `dev`, run `Install (run once).bat` in each tool folder only if the .venv is missing, then
   `Run self-test` / `python -m seismic_tool --selftest` and `python -m seismic_tool --guitest`
   (the guitest now runs against real PySide6 offscreen - first real-Qt check of all new code).
2. Seismic tool, open `NPRA 2D.seisproj.json`: Horizons panel > New, click a strong reflector on R-2
   (left part, ~1200 ms), press T. Check the pick follows the event, Shift+click erases, Ctrl+Z undoes,
   left-drag draws a line. Mouse wheel zoom must still work in pick mode; press P to leave pick mode.
3. Faults: N, click 3-4 points down a fault on R-2, Enter. Draw a second stick on R-3. Check the sticks and
   the thin cut line display, Esc / Ctrl+Z behaviour, Shift+click delete.
4. RGT on NPRA: Interpret > Compute RGT (all 2D lines) with node spacing 8 (about 2 min). Display box "RGT",
   then Interpret > Build horizon stack (2D), tick "RGT contours": contours should follow reflectors.
   Domain box "Wheeler": reflectors flat. Interpret > RGT edit mode (E): click a contour, click a nearby
   reflector, Ctrl+Z.
5. Well Correlation Tool: open the NZ project - Tui-1 is new (from DLIS). Check its GR/RHOB/NPHI/DT tracks
   (2379-2561 m and 3593-3729 m) and that the other wells look unchanged. Try
   `python -m well_correlation --dlis2las <the 171 MB Tui-1 main DLIS> <folder>`.
6. Maui 3D (when imported): link the NZ well project; wells + tops on inlines; Interpret > Well tie for
   Maui-5/6 (time-depth = sonic anchored to Tui-1's checkshot), Auto tie, Save; Interpret > Horizon maps.
7. Analytics: Interpret > Analytics, add 2-3 basemap grids + a CSV target, Train (GBM, leave one well out),
   then **Model QC...** - read the summary (baselines / leakage / Moran) and open the HTML. Try
   **Import engineering / production CSV...** on `python dev/make_synthetic_engineering.py C:\temp\eng`
   (completions.csv first, then production.csv) - the mapping form (combo per column) is new real-Qt code.
8. Dashboard (new real-Qt window): on the NZ project linked to the Maui 3D, Interpret > Dashboard - deviated
   Maui wells should be drawn from their surveys; Colour by TVDSS, then a basemap grid; drag the histogram range;
   click a well on the map / a table row and check the highlight follows in all panels. Select a deviated
   well, Well strip: the seismic panel's yellow time path should sit on the reflector the well follows.
9. Horizon maps on the NPRA horizon (survey "2D"), export CPS-3 and load it in Kingdom/Petrel to confirm
   orientation (top row first).

10. **Dynamic depth conversion (item 17, v0.22-v0.26)** - NZ well project + Maui horizons (3D) or NPRA:
    a. Interpret > Velocity model: set V0 + kZ on a layer, Cross-plot of selected layer, Build; try Kriging;
       Use for depth display; an inline in Depth.
    b. View > Auto-update depth model; move a top in the Well Correlation Tool and save -> status line
       "... top moved ... -> rebuilt layers ..." within ~2 s, depth section and misfit table update.
    c. What-if pseudo-top (dialog bottom row) and Clear.
    d. Interpret > Well prognosis: a Maui well, a planned vertical well; Set bit, Add drilled top.
    e. Blind test, then P90 map on the basemap; prognosis shows +/- 1 sigma.
    f. Depth-convert the Maui volume to SEG-Y (Cancel works) and load it in Kingdom/Petrel as depth (dz in the
       header is in mm = 1 "ms" per m); a depth CPS-3 grid -> time.

11. **Basin model (item 18, v0.27-v0.31)** - NZ well project + Maui horizons with RGT, depth model from item 17:
    a. Interpret > Basin model: the Taranaki age template appears (approximate ages - check them); type your seismic
       horizon names in the "horizon" column; Lithology from wells; Save to project.
    b. Section domain box "Wheeler (age)" on an inline with two dated horizons: vertical axis in Ma, hiatus labels.
    c. Wells tab: Maui-5 > Run well (burial / temperature / %Ro curves), Calibrate heat flow to BHT (BHTs come from
       the LAS / DLIS headers, Horner where two runs exist), compare with your own maturity data.
    d. Maps tab: Run map at a deep surface (every 10 nodes) -> %Ro / temperature / oil-window entry on the basemap,
       Export CPS-3; Colour Wheeler (age) by %Ro; Event chart (source surface + trap ages).

12. **Seismic inversion (item 19, v0.32-v0.35)** - Maui 3D (or a subset) + NZ wells with ties saved:
    a. Interpret > Seismic inversion: Low-frequency model tab - select the framework horizons, "Pick from seismic
       spectrum", tick ai (and vs, rho for pre-stack), Build LFM; look at "<vol> [LFM AI]" on an inline.
    b. Post-stack tab: Set wavelet (Ricker or your Well-tie library wavelet, scaled at the wells), Run inversion
       (model-based, then blocky), QC at well, Blind-well test; Coloured inversion for a quick relative AI.
       Start with a small subset - the model-based run costs ~5-10 ms per trace (about 2 h per million traces).
    c. Pre-stack tab only if you have partial angle stacks (none in the public Maui data): add each stack with
       its mean angle, Run simultaneous inversion -> Zp, Zs, density, Vp/Vs, lambda-rho, mu-rho.
    d. Cross-plot / export tab: Zp vs Vp/Vs coloured by "log: GR"; Export SEG-Y and load it in Kingdom/Petrel.

## Known limits (all items)
- Long jobs (RGT, stacks, attributes, import) have a Cancel button (new in v0.17 - test it);
  autotracking of a whole 3D runs in the GUI thread (wait cursor).
- RGT between node columns is interpolated (fine detail at the node spacing); thin layers below tuning do not
  pinch to zero; 2D RGT edits do not propagate through ties.
- Wells drawn vertical at the surface location; tie = one shift + one stretch.
- The Maui checkshot LAS files hold no time-depth pairs (mislabelled export).
- Inversion (item 19): tested on synthetics only (no real-data inversion in the repo - the Maui 3D is not imported
  and there are no partial stacks); 3D volumes only (2D lines not yet); model-based ~5-10 ms per trace (CG on the
  CPU), pre-stack ~3x that; one wavelet for the whole volume (no time-variant wavelet); no anisotropy / NMO stretch
  corrections for the angle stacks; the LFM uses the wells' tied time-depth (save your ties first).
- Basin model (item 18): ages in the Taranaki template are approximate; conductivity is not temperature-dependent;
  no isostasy / tectonic subsidence back-calculation; one lithology mix per zone (no vertical variation inside a
  zone); map pseudo-wells use the regional heat-flow history (no heat-flow map yet unless typed in code); kinetics
  parameters are approximate published values; maturity of a growing unit follows each sediment particle (Wheeler)
  but the surface maps use the top of the unit.
- Depth model (item 17): the what-if / drilled-top pseudo wells carry the model's own interval velocity below
  their top; T-D-function layers ignore what-if wells; P10/P90 come from the blind-test spread (no Monte-Carlo on
  V0/k); depth-to-time of a grid samples the model on a 1 ms TWT grid.

## Change log (newest first)

## Seismic v0.32.0-v0.35.0 - Seismic inversion package (PLAN item 19)
- **Low-frequency model** (v0.32): Vp, Vs, density and AI of every tied well blocked onto the seismic time axis
  (despiked; Gardner density / Castagna Vs where missing), interpolated through the structural framework -
  proportional / parallel-to-top / parallel-to-base between horizons, or along RGT stack levels - laterally by IDW or
  simple kriging (or scaled to a velocity-volume trend), in ln(value) so the model stays positive; zero-phase
  Butterworth low-pass with the cut picked from the seismic spectrum; stored as a volume "<vol> [LFM AI]".
- **Post-stack** (v0.33): coloured inversion (operator = well AI spectrum trend / smoothed seismic spectrum, -90
  degrees, scaled at the wells) and model-based inversion in ln AI (conjugate gradients with FFT convolutions,
  Tikhonov to the LFM or blocky L1 / TV by IRLS, optional lateral smoothing), wavelet scaled at the wells,
  residual NRMS, leave-one-well-out blind test.
- **Pre-stack simultaneous** (v0.34): partial angle stacks with their mean angles, Fatti linearisation with the
  wells' ln Zs / ln rho vs ln Zp background trends; Zp, Zs, density, Vp/Vs, lambda-rho, mu-rho volumes in one pass.
- **Interpret > Seismic inversion** window (v0.35): LFM, wavelet, post-stack, pre-stack, QC at a well (log / LFM /
  inversion, seismic / synthetic), blind-well table, cross-plots coloured by a volume or a well log, SEG-Y export.
  Inversion volumes (AI, Zp, ...) are shown with a positive rainbow colour range.
- Synthetic checks: blind-trace AI within 2.2 % in the seismic band (LFM alone 6.5 %), 2.7 % with 15 % noise, blocky
  earth 0.7 % (Tikhonov 1.0 %), coloured inversion r = 0.93 with the band-limited truth, simultaneous inversion Zp
  2.3 %, Zs 3.9 %, Vp/Vs 2.2 %, density 0.9 %.

## Seismic v0.27.0-v0.31.0 - Basin model integrated with the Wheeler diagram (PLAN item 18)
- **Age table** (v0.27): one row per dated surface (age of the surface, hiatus "gap" below it, eroded metres,
  lithology fractions of the zone below, role). Taranaki template for the Maui formations (approximate published
  ages - edit them). Lithology per zone from the well logs (Vsh lithology of the Well Correlation Tool).
- **Wheeler (age)** domain: RGT stack levels get ages by interpolation between dated horizons (jump at an
  unconformity); hiatus / pinch-out gaps are blank and labelled "hiatus 22.0-30.0 Ma (8.0 My)".
- **Burial history** (v0.28): backstripping with Athy porosity (Sclater & Christie per lithology, mixed by fractions),
  compaction never undone on uplift, erosion events (eroded section deposited in the first half of the hiatus, removed
  in the second), paleo water depth; burial curves CSV.
- **Thermal + maturity** (v0.29): transient 1D conduction on the moving burial grid (basement slab, heat flow at its
  base: constant, piecewise or McKenzie rift), bulk conductivity from lithology + porosity, radiogenic heat.
  Heat-flow calibration to BHTs read from LAS (~P BHT/MRT, BLI, TCS/TLAB, temperature curves) and DLIS parameters
  (Tui-1), Horner when two logging runs share a depth, rule-of-thumb correction otherwise, implausible gradients
  flagged. EASY%Ro, Lopatin TTI, kerogen transformation ratio (Pepper & Corvi organofacies kinetics), oil / gas
  window timing.
- **Pseudo-wells** (v0.30): the 1D model on many columns at once (all columns of a map run as one numpy batch:
  ~5,000 pseudo-wells in ~3 s) - maps of present temperature, %Ro, TR and oil-window entry at any surface;
  Wheeler coloured by temperature / %Ro / TR / porosity of the sediment of each age, today or at any past age;
  restored burial section at an age; petroleum-system event chart (PNG).
- **Interpret > Basin model** window (v0.31): age table, heat flow + calibration plot, per-well burial / temperature /
  %Ro plots, map runs (threaded, Cancel) with CPS-3 export, "Re-run when the depth model changes" (with View >
  Auto-update depth model), Wheeler colouring, event chart.
- Real-data check: BHTs read from Maui-5/6/7, Rahi-1, Kiwi-1 (annular temperature curve), Maui-4 (TEMP curve) and
  Tui-1 DLIS (116 degC at 3911 m); calibrated Maui present heat flow ~35-38 mW/m2 with the default (Sclater & Christie)
  shale properties - lower than published Taranaki values (~55-65), i.e. the default shale conductivity is probably
  too low for these silty mudstones; present %Ro 0.5-0.65 at the Kapuni level.

## Seismic v0.26.0 - Dynamic depth conversion, part 5: scenarios + uncertainty (PLAN item 17e; item 17 done)
- Velocity model dialog, new rows: **Save scenario** (name) keeps the current model settings in the project
  (functions, gridding, guide, held-out wells); **Difference map vs scenario** shows current - scenario depth of the
  selected layer on the basemap.
- **Blind test (leave one well out)**: each well's tops are predicted by the model built without it (fast - only at
  the wells). The info line gives per horizon: count, mean error, RMS, P10 / P90 of the error and how the error grows
  with distance to the nearest well. The RMS becomes the +/- 1 sigma column of the well prognosis.
- **P10 / P50 / P90 / 1 sigma** map of the selected layer: P50 = model depth, sigma = blind-test error model vs
  distance to the nearest well (tapering to 0 at the wells when residual correction is on); P10 = shallow case,
  P90 = deep case (+/- 1.28 sigma).
- **Depth-convert volume to SEG-Y...**: whole 3D volume to depth (threaded, Cancel button, no partial file left);
  IEEE float, inline/crossline in bytes 189/193 and 9/21, X/Y in 181/185, dz in the header in mm.
- **Depth grid (CPS-3) -> time...**: converts a depth grid through the model and writes <name>_time.cps3.

## Seismic v0.25.0 - Dynamic depth conversion, part 4: well prognosis / geosteering (PLAN item 17d)
- Interpret > **Well prognosis / geosteering**: pick a well of the linked well project (deviation survey used when
  present) or a planned vertical well (X, Y, TD). The table lists every horizon the wellbore meets: MD, TVDSS, X, Y,
  status (planned / drilled / ahead of the bit / below TD projected), +/- 1 sigma (filled once the blind test of
  17e runs) and the change since the last update. Laterals list every crossing (e.g. leaving a horizon upward).
- While drilling: "Set bit depth", then "Add drilled top" (horizon + MD): the top becomes a what-if pseudo-top at that
  point of the wellbore, the model rebuilds from that horizon down, and the info line lists the horizons ahead of
  the bit with their new depths. "Clear drilled tops" undoes it. Nothing is written to the well project.
- With View > Auto-update depth model on, the prognosis refreshes when tops / horizons change ("Moved: H1 +0.8 m").
- Export CSV (well, horizon, MD, TVDSS, X, Y, status, sigma, change).
- Test: NZ project + Maui depth model; prognosis for a Maui deviated well; a planned vertical well; set bit, add a
  drilled top 10 m deeper than predicted and check the deeper horizons move.

## Seismic v0.24.0 - Dynamic depth conversion, part 3: gridding choices + seismic velocity guide (PLAN item 17b)
- Velocity model dialog, "Gridding of well values": Inverse distance (old default), Thin-plate spline (honours planar
  trends between wells), Kriging (simple kriging, spherical or exponential variogram fitted to the wells, or type a
  range in m). Used for V / V0 and for the residual-correction grids.
- New table column "seismic velocity guide": pick a velocity volume of the project for a layer. Choose whether the
  volume is RMS / stacking velocity (Dix interval velocity between the layer's horizons) or interval velocity, and
  whether the wells scale it by ratio or shift it by difference. The trend fills in where wells are sparse; the
  wells still decide the level. Load a stacking-velocity SEG-Y with File > Import 3D like any volume.
- Test: Maui 3D + NZ wells with Kriging; if you have a Maui stacking-velocity volume, guide the deeper layers with
  it and compare the depth maps away from the wells.

## Seismic v0.23.0 - Dynamic depth conversion, part 2: live update + what-if wells (PLAN item 17c)
- **View > Auto-update depth model**: while on, the tool watches the linked well project (.wcproj.json) and the
  horizons of the model. Save a moved / new top in the Well Correlation Tool (or re-track a horizon) and within ~1-2 s
  the depth model rebuilds only the layers at and below the change; the depth section redraws, the velocity dialog's
  misfit table and the depth map shown on the basemap refresh, and the status bar says what happened, e.g.
  "Maui-6 F0 top moved +4.0 m -> rebuilt layers 4-6 in 0.30 s". Maps above the change stay bit-identical.
- **What-if well** (Velocity model dialog, bottom row): X, Y, horizon, TVDSS depth -> "Add pseudo-top". The model is
  rebuilt with that pseudo-top as if a well had it (from that horizon down); the well project is not touched (the
  what-if wells are kept in the seismic project). "Clear what-if wells" removes them.
- The velocity dialog and the depth display now share one model after "Use for depth display".
- Speed: 50 wells on a 300 x 300 map rebuild in < 0.1 s; on a full 1000 x 1500 map expect ~0.5 s per layer whose
  well values changed (deeper layers with unchanged well values reuse their gridded velocities).
- Test: link the NZ well project, build + Use a model, View > Auto-update depth model, show an inline in Depth; in the
  Well Correlation Tool move a Maui-6 top and save -> status line + section update. Then a what-if pseudo-top.

## Seismic v0.22.0 - Dynamic depth conversion, part 1: velocity functions (PLAN item 17a)
- Interpret > Velocity model: each layer now has a **velocity function** column and a **k** column:
  V0 (constant, the old behaviour and the default), V0 + kZ, V0 + kT, Well T-D function (IDW-blended or nearest
  well's checkshot / tied sonic). k blank = fitted over the wells (one regional k per layer; V0 per well, gridded);
  type a number to fix k (1/s for V0 + kZ, m/s2 for V0 + kT).
- **Cross-plot of selected layer** button: interval velocity at the wells vs mid-layer depth (vs one-way time for
  V0 + kT), least-squares line, R2, outlier wells (> 2 sigma) as red crosses, the model's k in the caption.
- Depth sections now follow the layer function inside each layer and always land on the depth maps (with residual
  correction the section and the maps used to disagree slightly - fixed). Old saved models load unchanged.
- Test: Velocity model on the NZ wells + Maui horizons: V0 + kZ on the deeper layers, Cross-plot, Build, compare
  the misfit table with the V0 model; Use for depth display and look at an inline in Depth.

## Seismic v0.21.1 - review of today's new code
- Well tie: a second / later event pair now lands where you click (the click is mapped back through the existing
  warp); the tie window is read in displayed (tied) time and an empty window falls back to the whole log (crash
  before); "best trace within N" re-computes when changed (0 = the well's trace only); Correlate after Angle
  gather shows correctly again; the warp-lag plot clears after Clear warp / Undo; picking a library wavelet keeps
  the wavelet box in step; batch tie reads every well path once and falls back to the vertical trace.
- Basemap: projects with up to 500 wells keep all names in view (the 150-label cap only applies to big projects).
- 3D import: IBM conversion in float32 (no float64 temporaries - lower peak RAM with threads; bit-identical
  results); a failed or cancelled import removes only the files it wrote itself.
- Calibrated sonic with no checkshot level inside the sonic: drift from the checkshot extended at its end velocity.

## Seismic v0.21.0 - fast crosslines + faster import (from Chris's PC benchmark)
- New imports write a third copy laid out by crossline (`data_xl.npy`, +50% disk): crosslines read as one
  contiguous block instead of ~1,000 small reads (Chris's PC: 95 ms per crossline before).
- Existing volumes: Interpret > "Fast crosslines for current 3D volume" (or `--xlstore STORE_DIR`) adds the copy.
- Sorted SEG-Y traces are written as one block per chunk instead of trace by trace; one inline-block read now
  feeds both the time-slice and the crossline copies.
- `dev/bench.py` now reports raw disk speed and the import broken into phases.

## Seismic v0.20.4 / Mistie Tool - third review round (run 2)
Seismic:
- Well strip: the seismic along the wellbore was drawn transposed (MD on the time axis) - fixed.
- Velocity dialog: "Depth map of selected layer" used the table row, not the layer (wrong map / crash when a row
  above was unticked); horizons without picks no longer scramble the layer order; errors show in the dialog.
- Mistie dialog: "Apply corrections" before "Measure + solve" now uses the dialog's window / lag / groups / flex.
- Analytics window: errors (e.g. spatial CV without coordinates) show in the output box; the target is never used
  as its own input.
- Engineering import: the Data box shows the auto-detected / loaded kind (completions / production).
- `--png` on a project without data gives a clear message. Launch .bat files (Seismic, Well Correlation): a dragged
  project path containing ")" no longer breaks the launcher.
Mistie Tool:
- Tie measurement with a max lag longer than half the window returned -max lag (e.g. -40 ms for a true +8 ms);
  halved-window QC was affected first. Fixed + self-test check.
- Closing after a failed save (file locked) no longer discards the work; display windows past the record end no
  longer crash the overlay; a failing window change is rolled back; Ctrl+Z with Caps Lock undoes (redo =
  Ctrl+Shift+Z / Ctrl+Y); network menu items do nothing without a project; `--batch` on a saved project solves like
  the GUI (locked lines, per-survey grouping).

## Seismic v0.20.3
- 2D RGT edits keep the traces where the line crosses other RGT lines fixed (network stays consistent) and update
  the 2D horizon stack (Wheeler / stratal views were stale after an edit).
- Well tie: wavelet estimation method box (least squares / multitaper); after "Tie all wells..." the bulk-shift
  map is shown on the basemap when the dialog closes.

## Seismic v0.20.2 / Well Correlation - second review round (run 2)
Well Correlation Tool:
- **Survey files**: a CSV like `MD,INC,AZI,TVD,NS,EW` was read as "seq, MD, INC, AZI" (columns shifted -> wrong TVD
  and trajectories). Columns are now taken from the header names; a sequence column is only assumed when it counts
  1, 2, 3... Kiwi-1's Schlumberger listing still reads the same (106 stations).
- TVD below the last survey station continues at the last inclination (was held constant).
- LAS: comma-delimited data (DLM COMMA) loads; a depth curve with a blank unit uses STRT's unit (feet converted).
- DLIS checkshots: all depth channels converted to metres (only TDEP was).
Analytics / engineering:
- Unit detection by whole words: "Proppant_Total_LBS" was read as tonnes (x2204), FracFocus "TotalBaseWaterVolume"
  is gallons (was taken as bbl, 42x too much), "tons" = US short tons.
- Production volumes in m3 / e3m3 are converted to bbl / mcf (BOE was wrong for Canadian data); months missing
  between first and last production count as zero (cums were over too many calendar months); EUR forecast starts
  after the peak month (was 3 % low with a late peak). Dates like 2020-01-15T00:00:00 and 2020/01/15 read.
- API numbers that Excel mangled (lost leading zero, trailing ".0") still join.
- k-means no longer crashes with more clusters than distinct rows; conformal intervals need >= 3 wells (clear
  message); spatial CV block size from the data extent, clear message when there are no coordinates; optional
  xgboost / sklearn models can be cloned in CV; lateral stats use the lateral midpoint as the well location.
Seismic:
- RGT editing: a new edit no longer erases earlier point edits in its window (they are re-applied).
- 2D RGT ties between lines with different start times were off by the t0 difference - fixed.
- Horizon-based misties use raw (uncorrected) pick times, like the waveform ties (re-solving no longer undoes a
  correction). Mistie measure: max lag limited to the window.
- Batch well tie skips wells outside the survey (were tied to an edge trace).
- Stratal slices: surfaces outside the record give NaN (crash / edge values before). NaN samples in 2D lines are
  replaced on read. Dips are limited to +/-10 samples/trace (flat-amplitude zones gave absurd values); dip block
  halo enlarged (no seams). Time-slice coherence reads only the needed dip slab (was two whole volumes).
- Attribute volumes re-computed under the same name go to a new store folder (the old one may be open); a
  cancelled attribute job removes its unfinished store.
- Time-depth extrapolation uses the slowness of the last / first 50 m (one washout sample set it before);
  checkshot levels that go back in time are dropped properly; Ricker wavelet centred for any sample rate.

## Seismic v0.20.1 - fixes from a full code review (run 2)
Core:
- Autotracker kept only the whole-sample part of each pick in mode "None" (dipping horizons came out flat); fixed.
- Horizons: undo of a delete after Save lost the horizon's files; two horizon names mapping to the same folder
  ("Top C1.5" / "Top C1_5", or "top" / "Top" on Windows) overwrote or deleted each other's picks. Folders are now unique.
- 2D lines with descending CDP numbers lost their navigation (all traces at one end of the line).
- SEG-Y: a junk extended-header count in rev-0 files skipped the first traces; ns = 0 now gives a clear error.
- Amplitude extraction no longer crashes when a horizon lies outside the volume's record (NaN there).
- Depth sections: no crash outside the velocity model; depths can no longer fold.
- Single-inline / single-crossline 3D surveys get a sensible map transform; nominal 25 m grid was wrong with steps != 1.
- Gridding 2D picks over large areas used several GB of RAM (batches now scaled to the grid size).
- Re-importing a volume under the same name writes a new store folder (the old one may be memory-mapped; on
  Windows the overwrite failed, and a cancelled re-import deleted the old volume).
GUI (real-Qt behaviour the mock could not show):
- Menu actions / buttons no longer pass Qt's "checked" flag into optional parameters (attribute volume, horizon
  stack levels and "stack level -> horizon" were affected).
- Opening a section from the data tree / basemap / after import, and restoring the last section, compared a list
  with a tuple and silently did nothing - fixed.
- New / open project resets the section panel (old source, RGT editors) and is refused while a job runs; a finished
  import goes to the project it started in.
- Enter in the line-number box jumps to the line again (Return / Esc stick shortcuts now only on the plot).
- Shift+click on a seed or stick point reaches the pick handler (scatter symbols let clicks through).
- Well tie: switching wells resets shift / stretch / warp (no carry-over into the next well's saved tie); a saved
  tie fills the spin boxes; no crash when the tied synthetic is outside the record.
- Dashboard: histogram range can be widened again after filtering; colour box keeps showing the current colouring
  and re-picking an entry works; right-clicks / axis clicks no longer change the selection, a click away from all
  wells clears it; an all-null log curve no longer breaks "Colour by log" / Well strip.

## Seismic v0.20.0 - well tie upgrade, part 2 (PLAN item 16c; item 16 done)
Third button row in Interpret > Well tie:
- **Estimate wavelet**: deterministic least-squares wavelet from the tied reflectivity and the seismic (window
  applies), reports its constant phase, predictability (PEP) and NMSE, and recomputes the synthetic with it
  (wavelet box "Deterministic (from tie)"). **Save wavelet** puts it in the project's wavelet library
  (`<project>_seismic/wavelets.json`); the library box loads any saved wavelet (resampled to the survey rate).
  A multitaper (coherence) estimate is in `tie.multitaper_wavelet` (no button yet).
- **Despike logs** (DT, RHOB; running median + MAD; the project's logs are not changed).
- **Seismic along wellbore**: for deviated wells the seismic is sampled along the well path (survey) at each
  synthetic time instead of the trace at the surface location.
- **Angle gather**: Aki-Richards synthetics at 0-40 deg (Vs from DTS, else the mudrock line) in the top-right plot.
- **Tie all wells...**: FFT auto tie + deterministic wavelet for every well; table
  `<project>_seismic/well_tie_batch.csv` (shift, stretch, corr, NRMS, held-out corr, wavelet phase, PEP), mean /
  spread of the bulk shift, and an "average (batch)" wavelet in the library. `tie.shift_map(rows)` grids the
  bulk shifts (a trend = velocity / datum problem; one odd well = bad tie or time-depth).
- Library functions also cover washout masks (caliper vs bit, DRHO) and Backus averaging (`tie.backus`).
- Real-data check: Tui-1 sonic integrated from the top checkshot level drifts to -21 ms at TD vs the checkshot;
  "calibrated sonic" removes it at every level.

## Seismic v0.19.0 - well tie upgrade, part 1 (PLAN item 16a + 16b)
Interpret > Well tie has a second button row:
- **Window** (top/base ms of the synthetic; "all" = whole log) - Auto tie, Correlate and stretch/squeeze use it.
- **Auto tie** is now FFT-based (all shifts at once per stretch), stretches about the window top, reports NRMS and a
  held-out score (fit on the upper 70 % of the window, correlation on the lower 30 %).
- **Correlate** (Hampson-Russell style): correlation vs lag (blue) with its envelope (grey), polarity, a constant
  phase scan (bottom-right plot; best rotation = wavelet phase error, 180 = reversed polarity) and a
  **cycle-skip warning** when another peak correlates nearly as well.
- **Find best trace** within N traces of the well (spin box) - the tie then uses that trace.
- **Auto stretch/squeeze**: constrained dynamic time warping after the bulk tie; max lag (default 20 ms) and
  max interval-velocity change (default 10 %) are hard limits. **Pick event pair**: click an event on the red
  synthetic, then the matching black seismic event (<= 10 pairs, no folding). **Undo** / **Clear warp**.
- **Drift QC**: tied time-depth minus checkshot at each level (plot + worst level). **Export TD CSV**: tied
  time-depth in the checkshot CSV format (keep it out of the well's checkshot folder - it would be averaged in).
- Saved ties keep the warp; it is applied wherever the well's time-depth is used (tops in TWT, sections, maps).
- `project.display["td_mode"] = "calibrated sonic"` uses the sonic calibrated to the well's own checkshot
  (drift interpolated between levels) instead of the checkshot alone (no menu yet).

## Seismic v0.18.0 - scale: many wells on the basemap, threaded import (PLAN item 15 finished)
- Basemap with thousands of wells: all wells are one scatter item; well names are drawn only for wells inside the
  view and only when at most 150 are in view (zoom in to see names). Above 2,000 wells the view query uses the
  SQLite well index (`wellindex.WellLayer`). Before, every well got its own text label (10,000 labels = very slow).
- 3D import converts traces and builds the time-slice copy with up to 4 threads (`import_3d(workers=)`);
  chunks capped at ~64 MB, NaN/inf clean-up in one pass. Output is bit-identical to the serial import (tested).
- Sections render with automatic downsampling when zoomed out (long 2D lines, big time slices) and the colour
  clip is computed from a 1M-value sample of the section instead of every value.
- Dashboard: clicking the map finds the nearest wellbore in one vectorised search over all stations.

## Seismic v0.17.0 - Cancel button, well index (PLAN item 15, first part)
- Every long job (3D import, dip, RGT, attributes, stacks) now has a **Cancel** button next to the progress bar;
  a cancelled 3D import removes its half-written store.
- `seismic_tool/wellindex.py`: SQLite index of well headers / tops / log list with spatial buckets (10,000 wells:
  box query about 1 ms) - groundwork for very large well projects on the map and dashboard.

## Seismic v0.16.1 - well strip / geosteering view
- Dashboard > Well strip: log vs MD, TVDSS vs MD with zone top/base, seismic along the wellbore + time path.
- Log curves as lateral colour sources.

## Seismic v0.16.0 - Dashboard with laterals (PLAN item 14, first parts)
- Interpret > Dashboard: map / histogram / crossplot / table docks with linked selection and filters; wells from
  the project with deviation surveys; laterals coloured along MD by TVDSS, grids, seismic amplitude, stage CSV
  values or in-zone; per-lateral stats (mean/P10/P50/P90, % in zone) -> Analytics; layouts; HTML export.
- Drawing is batched: one curve per colour bin for all wells plus level-of-detail decimation (1,000 synthetic
  laterals -> 16 curves in 0.1 s).

## Seismic v0.15.1 - engineering / production import dialog
- Analytics window: **Import engineering / production CSV...** (presets Enverus, IHS, Texas RRC, FracFocus,
  NDIC, ECMC, NM OCD, OCC; editable column mapping + units; save/load mapping). Works without a well project
  (wells come from the CSV, joined by API).

## Seismic v0.15.0 - Analytics 13d: model QC toolkit
- `analytics/qc.py`: baselines (mean, IDW of the target, best single input), y-randomisation, random-noise
  feature, residual QQ / variogram / Moran's I, split-conformal and quantile-GBM intervals with coverage,
  classifier calibration (Brier), applicability domain (Mahalanobis + kNN), importance stability,
  hyperparameter sensitivity, PDP direction checks, paired fold comparison, blind / temporal / repeated designs.
- Analytics window: **Model QC...** button (one-page HTML); Predict map now carries an applicability mask.
- Tests: Moran's I z 24.9 -> 4.1 once x/y is added; conformal 80% interval covers 84% of new wells;
  y-randomisation collapses R2; noise column ranks last.

## Seismic v0.14.0 - Analytics 13a-13c (PLAN item 13, first parts)
- analytics package: attribute table + joins (API/name), data prep (imputation, outliers, correlation, VIF,
  PCA), engineering/production features (presets, units, cums, IP, GOR, Arps EUR, parent/child, FracFocus, lease
  allocation), models (linear/ridge/lasso, kNN, tree, RF, XGBoost-style GBM with NaN handling + monotone
  constraints + early stopping, MLP, k-means, GMM, SOM), honest validation (LOO well, group, spatial blocks,
  nested CV, learning curves), explainability (exact TreeSHAP, KernelSHAP, permutation importance, PDP/ICE,
  HTML report), Analytics window (Interpret > Analytics), optional xgboost/sklearn/shap installer + --check-ml.
- Tests: GBM R2 0.97 vs linear 0.29 on a nonlinear benchmark; random CV 0.86 vs group 0.36 on clustered wells;
  TreeSHAP equals brute-force Shapley (2.8e-16) and sums to the prediction; KernelSHAP exact for linear; Arps
  EUR within 0.0% on the synthetic production set.
- Still to do (next run): 13d QC report (baselines, y-randomisation, residual variogram, conformal intervals,
  applicability domain, stability), engineering CSV import dialog, dashboard (item 14).

## Seismic v0.13.0 - Velocity model + depth conversion (PLAN item 12)
- Interpret > Velocity model: layer cake from horizons + well tops (or time-depth), gridded interval
  velocities, depth maps, well misfit table + CSV, optional residual correction; Depth domain on sections.
- Tests (3 synthetic wells, laterally varying layer-1 velocity): layer-2 velocity exact, well misfit 0.05 m,
  H2 depth everywhere within 1.3% (median), residual correction 0.00 m at wells, depth section reflector within
  4 m of the model depth.
- Known limits: inverse-distance velocity gridding (no kriging / trends); depth display converts the shown
  section only (no depth volume yet); below the deepest horizon the last layer velocity continues.

## Seismic v0.12.0 - Seismic attributes (PLAN item 11)
- On-the-fly attributes in the display box (complex trace, RMS, coherence, dip-steered coherence, spectral
  bands) for any section; Interpret > Compute attribute volume (new project volume); horizon curvature/dip maps.
- Tests: envelope/phase exact, inst. frequency 25.00 Hz on a 25 Hz tone, spectral decomposition separates 20/50 Hz,
  dip-steered coherence 0.99 vs 0.30 unsteered on 6 ms/trace dips, coherence 0.37 at a fault vs 0.98 away,
  block-computed volumes equal on-the-fly sections, paraboloid curvature exact (0.500 1/km).
- Known limits: attribute volumes double the storage (two float16 copies); time-slice coherence reads a whole
  time slab of the dip field (slow on very large surveys).

## Seismic v0.10.0 - 2D-3D / 2D-2D mistie analysis (PLAN item 10, first part)
- Interpret > Mistie analysis: simultaneous shift + phase + gain per tie, network solve with 3D fixed,
  robust weights, loop closures, apply/reset corrections on the fly (display, tracking, RGT, picks).
- Tests: single tie 8.9 ms / 33.7 deg / x2.00 (true 9 / 35 / 2); 2D cut from 3D with known corruption
  8.01 ms / 30.2 deg / x2.00; network of 3 lines + 3D within 0.7 ms and 3 deg with a deliberately bad tie
  (robust solve isolates it; non-robust is pulled 4 ms off); corrected line vs 3D corr 0.47 -> 0.99.
- v0.10.1: Flex mode (along-line shift with smoothness, knots every N traces): a 0-12 ms ramp is recovered
  within 0.14 ms; very smooth = constant per line; per-trace corrections applied to display and picks;
  warning when a line needs > 10 ms local flex.
- v0.11.0: horizon-based misties, survey groups + key lines, time-variant shift (smooth in TWT; ramp 4-12 ms
  recovered within 0.6 ms), overlapping vintages tied along the overlap (5 ms offset found as 5.05),
  duplicate hiding, residual map, tie table CSV, corrected SEG-Y export (round trip exact).
- Known limits: time-variant only for lines crossing a 3D; the Mistie Tool (tkinter) is linked via the
  CSV tie table, not live; no 2D-2D time-variant ties yet.

## Seismic v0.9.0 - Horizon maps (PLAN item 8)
- Interpret > Horizon maps (Ctrl+M): time structure, isochron, amplitude (RMS/max/min/mean/at horizon),
  depth with well average velocities (optionally calibrated to a formation top); basemap display with
  contours; export XYZ, CPS-3, PNG. 2D horizons are gridded.
- Tests: isochron exact, RMS/max/at values equal direct computation, contour of a plane on the expected line,
  CPS-3 and XYZ round trips, 2D gridding honours the lines, depth conversion 1475.2 m vs 1475.0 m.
- Known limits: velocity between wells by inverse distance (no kriging / velocity cube); CPS-3 row order is
  top row first (Petrel convention) - check orientation on first import into Kingdom.

## Seismic v0.8.0 - Wells on seismic + well ties (PLAN item 7)
- Wells + tops (TWT) on sections and time slices from the linked well project; time-depth from checkshot
  DLIS (Tui-1, Pateke-2), else sonic anchored to the nearest checkshot.
- Interpret > Well tie: synthetic (Ricker / statistical wavelet) vs seismic, shift/stretch, auto tie, save.
- Tests: synthetic survey delayed 12 ms -> auto tie finds 12.1 ms (corr -0.55 -> 0.99); statistical
  wavelet 12.1 ms; Tui-1 3875 m TVDSS -> 2806 ms (checkshot); Maui-5 tops 1864-2110 ms via sonic + Tui-1.
- Known limits: wells treated as vertical at the surface location on sections (deviated paths in item 14);
  stretch is one linear factor (no multi-point squeeze yet); tie uses the nearest trace only.

## Well Correlation Tool v0.5.0 - DLIS reader (PLAN item 6b)
- DLIS files in `wells\<Well>\dlis\` load like LAS; Tui-1 added to the NZ project from its two DLIS
  files (header from wells_header.csv; PR3099 tops were already there).
- `--dlis2las IN OUT_DIR [--arrays]`, `--dlisinfo`, `--checkshot IN.dlis OUT.csv` (Tui-1 + Pateke-2 CSAT).
- Convert the 171 MB Tui-1 main-pass DLIS on your PC: either copy it into `data\nz\wells\Tui-1\dlis\`
  (loads directly, File > Reload logs) or run --dlis2las into `wells\Tui-1\las\`.
- Tests: synthetic DLIS written byte by byte (segments, padding, absent attributes, arrays, IBM floats)
  round-trips exactly; real Tui-1 files parse in 0.5 s with sane GR/RHOZ/TNPH/DTCO ranges.
- Known limits: encrypted records skipped; checkshot vertical times assume a straight ray from a source at
  the SRD (offset 65.7 m at Tui-1, negligible below 1 km).

## v0.7.0 - RGT editing (PLAN item 6)
- RGT contours on sections; RGT edit mode (E): click a level, click where it belongs -> local re-solve
  (~0.15 s on the synthetic), stack updated, Ctrl+Z; active horizon as RGT constraint; re-link RGT around a
  fault. Edits saved at once.
- Tests: a moved level lands exactly where clicked (12.0 ms move), uint16 RGT outside the edited inlines /
  window is bit-identical, undo restores RGT + stack exactly; fault re-link fixes a model computed without
  the fault (p95 error 29 -> 0.6 ms); horizon constraint makes one iso-surface follow the picks; 2D edit
  local to the line window.
- Known limits: 2D edits do not propagate through ties (tie columns stay fixed); edits keep the RGT
  range of the original model (values beyond +/-5% head room are clipped).

## v0.6.0 - Horizon stack, Wheeler view, stratal slices (PLAN item 5)
- Interpret > Build horizon stack (3D / 2D); mode "Stratal slice" with slider through the stack; domain box
  Time / Wheeler (flattened RGT view with hiatus gaps); Interpret > Convert stack level to horizon.
- Tests: stack surfaces equal direct iso-surfaces and never cross; stratal slice = volume interpolated on the
  surface (err 1e-4); Wheeler flattens the synthetic (98% of variance between levels) and shows a true hiatus
  as a gap; NPRA R-2 Wheeler view is flat.
- Known limits: thin layers below seismic tuning do not reach zero thickness in the RGT, so a pinch-out
  shows as converging levels, not as a gap; stratal slices show amplitude only (not dip/attributes yet).

## v0.5.0 - RGT model, step B: mesh + global solve (PLAN item 4)
- Interpret > Compute RGT (current 3D volume) / (all 2D lines): settings dialog (node spacing, interval,
  correlation window, min correlation, links across faults, max throw, horizons/faults as constraints),
  runs in the background; display box "RGT" shows the result on any section.
- Dip-guided node links + correlation, cross-fault ties away from the fault, horizon constraints, 2D line
  ties, global least-squares solve (CGLS in numpy) with a coarse-to-fine start.
- Synthetic tests (fold + pinch-out + fault): iso-RGT surfaces match 60 true horizons, median error
  0.15 ms (4 ms sampling); across the fault p95 < 3.5 ms above the wedge; 2D tied lines median 0.10 ms.
- Real data: NPRA R-2 + R-3 (12,000 traces, 1.1 M unknowns) in ~100 s at 8 traces / 8 ms nodes; iso-RGT
  contours follow the reflectors through the folds (see test checklist).
- Known limits: between node columns the RGT is interpolated bilinearly (fine detail at node spacing);
  NPRA R-2 / R-3 don't cross, so 2D ties were only tested on synthetics; a big 3D (millions of nodes) needs
  minutes per 100 iterations - multi-threading/bricking in item 15. Stored as uint16, not float16 (float16
  has only ~2-4 ms resolution at 2-6 s).

## v0.4.0 - Dip field, RGT step A (PLAN item 3)
- Interpret > Compute dip field: inline/crossline dip (ms/trace) + planarity volumes for the 3D volume
  (background job with progress). Toolbar display box: Seismic / Inline dip / Crossline dip / Planarity,
  also on 2D lines (computed on the fly, cached).
- Synthetic: dips recovered within 0.6% (spec 5%). NPRA R-2: fold flanks clearly shown in the dip.
- Known limits: ~3 us per sample, so a 1-billion-sample 3D takes ~45-60 min (multi-threading in item 15);
  cancel works only via closing the app for now.

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

## Petrophysics (PLAN item 20, Well Correlation Tool v0.6+, run 5 - 24 Nov 2026)
20a Tool / vendor / resolution + curve QC (no GUI yet - 20e adds the Petrophysics window):
- `python -m well_correlation --petroqc "NZ Well Correlation.wcproj.json"` (or name wells after it) prints a report per
  well and writes `<project>_petro/qc/<well>.json`, `qc_summary.csv` and QC curves (bad hole, spikes, thin-bed flags,
  bed thickness) in `<project>_petro/<well>.npz`.
- Per curve: tool (HALS / HRLA / DLL / AIT / DIL / TLD / LDL / CNL / DSI / BHC / NGS / ARC / ADN / HDIL / ZDL / CN /
  XMAC / MPR / ACRt / EWR ...), vendor, kind (laterolog vs induction vs LWD propagation, ...), wireline vs LWD and why,
  typical vertical resolution and the effective one (never finer than 2 samples).
- Edit the tool table: `python -m well_correlation --tooltable my_tools.json`, edit, save it as
  `<project>_petro/tool_table.json` (entries merge over the defaults by name).
- QC: gaps, flat lines per raw curve, spikes, bad hole (caliper vs bit size or |DRHO|), GR depth shift of each run vs
  the longest one, splices the composite makes (level difference + shift), bed boundaries from the finest curve and
  the share of beds each curve cannot resolve.
20b Deterministic interpretation per zone (v0.7.0):
- `python -m well_correlation --petro "NZ Well Correlation.wcproj.json" Maui-7` -> zone table (gross / net res / net pay
  TVD, PHIE, Sw, SoPhiH) and `<project>_petro/summary.csv`, `params.json`, curves VSH / PHIT / PHIE / SW / PERM / ...
- Edit `params.json`: "default" for all, "zones": {"C-sands (Mangahewa)": {...}}, "wells": {"Maui-7": {"C-sands ...":
  {...}}}. Choices: vsh_method gr_linear / larionov_tertiary / larionov_old / clavier / steiber / sp / nd / min;
  phi_method nd / density / sonic_wyllie / sonic_rhg; sw_method archie / simandoux / modified_simandoux / indonesia /
  waxman_smits / dual_water; rw (value, or rw_method rwa_min / pickett / sp); a, m, n; cutoffs vsh_cut, phi_cut,
  sw_cut; perm_method timur / coates / wyllie_rose; toc true + LOM for Passey TOC. Empty = automatic, the values used
  are reported.
- Defaults are a sandstone model (rho_ma 2.65, Rw from the Rwa minimum, 15 degC + 30 degC/km): set Rw and the matrix
  for the Tikorangi Limestone before trusting it there.
20c Multi-mineral solver + Monte Carlo (v0.8.0):
- `python -m well_correlation --multimin "NZ Well Correlation.wcproj.json" Maui-7` with params.json e.g.
  "default": {"mm_minerals": ["quartz", "illite", "calcite"], "mm_fluid": "gas", "mm_logs": ["GR", "RHOB", "NPHI", "DT",
  "BVWR"]} -> volumes per sample (MM_V_QUARTZ ...), PHIE, Sw, reconstructed logs and misfit. Minerals available: quartz,
  calcite, dolomite, k-feldspar, illite, kaolinite, smectite, chlorite, anhydrite, halite, coal, pyrite.
- PE warning: Rahi-1 PE reads ~6.7 b/e in clean sand (barite mud) and Maui-7 ~3.0 - leave "U" out of mm_logs there.
- `python -m well_correlation --petromc "NZ Well Correlation.wcproj.json" Maui-7 --n 200` -> P90 / P50 / P10 of net pay,
  PHIH and SoPhiH per zone (Rw x lognormal 0.08 decades, m +/- 0.1, n +/- 0.15, rho_ma +/- 0.01, cut-offs); add
  `--method multimin` to perturb the mineral end-points too.
20d Rock physics at wells (v0.9.0):
- `python -m well_correlation --rockphysics "NZ Well Correlation.wcproj.json" Tui-1` -> Vs (measured DTSM where present,
  else Greenberg-Castagna iterated through Gassmann in hydrocarbons), AI / SI / Vp/Vs / lambda-rho / mu-rho / Poisson /
  Young's / brittleness, and Gassmann brine / oil / gas / fizz (7 % gas) versions of Vp, Vs, density, AI, Vp/Vs
  (RP_* curves). Fluids from Batzle-Wang at the zone's temperature and hydrostatic pressure; set salinity, api, gor,
  gas_gravity, rp_hc (gas / oil), rp_lith (sandstone / limestone / dolomite) per zone in params.json.
20e Petrophysics window (v0.10.0) - Well Correlation Tool > Petrophysics > Petrophysics window (Ctrl+Shift+P):
- Left: every parameter with its value and where it comes from (built-in / project / zone / well + zone). Edit the
  Value column; "Edit applies to" picks project default, zone (all wells), this well + zone, or this well (all zones).
  Type "auto" to remove an override. Saved to `<project>_petro/params.json` at once.
- Buttons: QC report, Interpret well / section wells / all wells, Mineral solver, Monte Carlo, Rock physics, Show on
  map hub (writes summary.csv + hub_request.json; the map hub itself is PLAN item 21).
- Cross-plots: N-D (matrix lines), M-N (mineral points), Pickett (Sw 1 / 0.5 / 0.25 lines from the zone's Rw, m, n),
  Buckles (constant BVW), histogram of any curve with discriminators ("VSH 0 0.4; PHIE 0.08 1"); Export PNG.
- Section: new toolbar tracks VSH, PHI (PHIE + BVW with the hydrocarbon fill), SW, PERM show the saved results.
Test first: open the NZ project, select Maui-7, Ctrl+Shift+P, Interpret well, look at the N-D and Pickett plots for
the C-sands (gas crossover and the gas leg should stand out), switch on the PHI and SW tracks in the section.

## Map hub (PLAN item 21, Seismic tool v0.36+, run 5 - 24 Nov 2026)
21a Layers: View > Map layers (tab next to Data). The basemap is now drawn from a saved layer list: tick boxes for
visibility, Up / Down for drawing order, opacity slider, colour ramp, contour interval (index contours every 5th, bold).
Add...: current basemap grid (from Interpret > Horizon maps), CPS-3 file, horizon time map, petrophysics value per well
(bubbles + labels from the Well Correlation Tool's petro summary - run its Petrophysics window first), text. Layers are
saved with the project (grids in `<project>_seismic/maps/`).
Test first: open the Maui / NPRA project - the basemap should look as before; Horizon maps > compute a structure map
> Map layers > Add... > Current basemap grid; change opacity, interval, order; save, reopen.
21b Petrophysics on the map (v0.37.0):
- Map layers > Add... > Petrophysics value per well (bubbles sized + coloured by the value, labelled) or Mineral pies per
  well (from the Well Correlation Tool's Mineral solver); both per zone or all zones.
- Well Correlation Tool > Petrophysics window > Show on map hub: the Seismic tool picks it up within ~3 s (petro layer +
  the wells ringed in magenta).
- Click a well with petrophysics results on the basemap: Petro strip (GR, resistivity, density-neutron, Vsh, PHIE /
  BVW, Sw with tops) and its zone table.
- Map layers > Draw polygon: click vertices, double-click to close; the wells inside are selected; Petro cross-plot...
  shows zone values per well (or the basemap grid at the wells vs a zone value) with the selection highlighted.
21c Gridding zone properties + grid calculator (v0.38.0): Map layers > Add... > Grid a zone property: value + zone,
method (inverse distance, thin-plate spline, kriging, kriging with external drift, collocated cokriging) and a grid
layer as the secondary (e.g. an RMS amplitude or inversion map); shows the correlation at the wells; Cross-validate
gives the leave-one-out RMSE / MAE / bias per method; Add residual map posts the LOO residuals. Grid calculator: grid
layers are A, B, C ...; type e.g. "A - B" or "clip(A * 0.3048, 0, 500)".
21d New wells (v0.39.0): Map layers > Add... > New wells: pick a folder in the usual layout (wells/<Well>/las|dlis +
wells_header.csv). It imports the wells that are not in the well project yet, runs the petrophysics on them, checks the
existing property maps at the new wells BEFORE re-gridding (the honest blind test: predicted vs actual table), re-grids
the maps made with "Grid a zone property" (plus a change map), ties the new wells on the first 3D survey and rebuilds the
depth model. Low-frequency models for inversion are only flagged - rebuild them in the Inversion window.
21e Cross-check (v0.40.0): Map layers > Add... > Cross-check petrophysics vs maps: pick a zone value and one or more grid
layers (attribute, inversion slice, structure ...): cross-plot with the regression line, r, residual sigma and
leave-one-out RMSE; Add predicted maps posts P50, P90 / P10 (80 % prediction interval), the residual map and the
residual-corrected map; Send to Analytics hands the well table to the Analytics window (GBM / SHAP / Model QC).
21f Better-looking maps (Seismic v0.41.0, Well Correlation v0.11.0):
- Map layers dock: pick a style (Classic / Print / Flat colour / Contours only) > Apply style: hillshaded perceptual
  colour fill (light = shallow for depth / time), bold index contours with labels along the lines (no overlaps), wells
  with industry status symbols (from the Status / Result column of wells_header.csv). Export map...: PDF / PNG at 300 dpi
  with title block, legend of the visible layers, scale bar and north arrow.
- Fault polygons: a polygons layer with style fault = true is drawn hatched with downthrown ticks (set "down": "W" etc.);
  a grid with style blank = true is blanked inside them so contours stop at the faults.
- Well Correlation Tool map: structure / isopach maps are now hillshaded with labels along the contours (set
  display "map" > "carto": false in the project file to get the old look).
21g GIS + page layouts (v0.42.0) - Map layers > Add...:
- Project coordinate system (EPSG): 2193 NZTM, UTM (326xx / 327xx / 269xx / 267xx), 32139 / 2278 Texas South Central,
  32040 (NAD27), 5070 CONUS Albers, 4326 ... Needed for GIS export, lat / long graticule and reprojecting layers.
- Vector layer: any shapefile / GeoPackage / KML / CSV (WKT or X,Y) - PLSS, counties, pipelines, leases, permits.
  Reprojected from its .prj (NAD27 -> NAD83 is an approximate shift, flagged). Style in the project file: "by" field
  (categorised, or "mode": "graduated"), "labels" field, "filter" e.g. "STATE == 'TX' and ACRES > 600", "fill".
  Click a feature on the map: its attributes in the dock status line; Attribute table (with a filter).
- Go to legal description: "660 FNL 1980 FWL Sec 14 T12N R3W" in a loaded PLSS sections layer (CadNSDI fields).
- Export for QGIS / ArcGIS: GeoTIFFs (with overviews), shapefiles, one GeoPackage and a KML of the visible layers.
- Export map...: now with a locator inset (red extent box), lat / long graticule, neatline and saved templates.
21h QGIS link (v0.43.0):
- Seismic tool > View > Send map to QGIS: writes `<project>_gis.gpkg` (wells with status + petro values, tops, faults,
  contours, prospects, 2D lines, 3D outline, vector layers - styled) and the grids as GeoTIFFs in `<project>_gis_grids/`.
  Set the project coordinate system first (Map layers > Add... > Project coordinate system).
- QGIS: run `python dev/build_qgis_plugin.py`, then Plugins > Manage and Install Plugins > Install from ZIP >
  dev/geoharmonic_link.zip. Toolbar "GeoHarmonic Link": Open GeoHarmonic project (pick the _gis.gpkg) adds everything in a
  "GeoHarmonic" group and reloads when GeoHarmonic rewrites it.
- Live: Seismic tool > View > Live link with QGIS, then Connect to GeoHarmonic in QGIS. Selecting wells in either app
  selects them in the other; the current seismic section shows as a line in QGIS; in QGIS select a line and press "Send
  selected line(s) as seismic section" (opens an arbitrary section), select polygons and "Send ... as prospects".
  Polygons / lines you digitise in the prospects_edit, leases or section_lines layers are picked up automatically.
Test first: Send map to QGIS, open it in QGIS with the plugin, connect, select two wells in QGIS.

## Petrophysics analytics + field reports (PLAN item 22, Well Correlation v0.12.0, Seismic v0.44.0)
- Petrophysics window > Predict DT (ML): trains a gradient-boosting model on the wells that have DT (inputs GR, RHOB,
  NPHI, RDEEP), reports a depth-blocked, well-grouped cross-validation (random K-fold on logs is optimistic - the test
  shows it), and writes DT_PRED with a P10-P90 band where DT is missing (section track DTP). Other targets: use
  `petro.logml.predict_curve(project, "RHOB", [...])`.
- Electrofacies: Gaussian-mixture facies on GR / RHOB / NPHI / RDEEP for all wells -> FACIES curve (section track FAC).
- Import report...: CSV / XLSX tables of shows, tests, cores, MICP or pressures (column names are matched loosely, feet
  and psi converted), or a text / PDF report (sentences with shows / tests / cores and a depth are extracted and marked
  "auto-extracted, check"). Pressures give fluid gradients and the contact. Results: <project>_petro/reports.json; the
  map hub can post Shows / Tests / Cores / Pressures / Contact_TVDSS per well (Add... > Petrophysics value).
- Library functions for core calibration: core-log depth shift + regression, Archie m from formation factors, n from
  resistivity index; mud-gas Haworth ratios (Tui-1 chromatograph CSV reads directly).


## Viewer quick wins: co-blending, bump map, linked cursor (PLAN item 23, Seismic v0.45.0)
- View > Co-blend attributes on the section: pick 2-3 attributes (default spectral 10 / 20 / 40 Hz) and RGB, CMY or HSV;
  works on 3D inlines, crosslines and time slices (in time). Per-channel 2-98 % clip; NaN = transparent.
- View > Attribute overlay on the seismic: an attribute in colour over the grey seismic with an opacity (optional
  threshold in code: only values above it are drawn). View > Bump-map the section / slice: seismic lit by its own relief.
  View > Plain section switches back.
- Map layers > Add > Co-blend 2-3 grids: RGB / CMY / HSV of grid / horizon layers (resampled onto the first grid),
  colour-triangle legend (corners = the grids). Add > Shade selected grid by structure: amplitude on structure (colour =
  the selected grid, shading = another grid lit from an azimuth / elevation with vertical exaggeration) or a single-light
  bump map of the grid itself.
- Linked cursor (View > Linked cursor, on by default): moving over the section puts a red cross on the basemap and the
  dashboard map and names the nearest well in the petro strip; moving over the basemap draws dashed lines on the section
  when the point is on the current line (a cross on time slices).
- View > Attribute parameter scan: computes an attribute (spectral frequency, RMS window, coherence window) on the current
  section for a range of values; the slider steps through the frames like a movie.
- View > Save window layout / Restore window layout: docks, open dashboard, section line / attribute / blend stored by
  name in the project; built-in templates Interpretation, Mapping, Well review.
Test first: open an inline of a 3D, View > Co-blend (10 / 20 / 40 Hz, rgb); then move the mouse over it and watch the cross
on the basemap.

## Resolution, single-well inversion, predictive modelling, well seeds (PLAN item 24, Seismic v0.46.0-v0.47.0)
- Interpret > Seismic inversion > Resolution...: dominant frequency and usable band (spectrum vs noise floor) in a time
  window, lambda/4 and lambda/8, wedge-model tuning thickness now and after spectral balancing, and - if you type the bed
  thicknesses from the logs - the frequency the reservoir needs and the share of beds resolved now / after balancing.
- Same window, "Single-well inversion (Monte-Carlo)": inverts the trace at the selected well 40 times with perturbed
  wavelet (phase, frequency, scale), low-frequency cut-off and noise; reports P10-P90 band vs the log, how much of the
  seismic-band detail is recovered and a YES / NO verdict ("can the seismic see the reservoir here?").
- Pre-stack tab, "Predictive modelling + fluid probability": hundreds of pseudo-wells (sand thickness, porosity, Vsh,
  brine / gas / oil / fizz via Batzle-Wang + Gassmann), near / far synthetics, intercept-gradient statistics, a Bayesian
  classifier, and probability volumes '<near> [P(gas)]', '<near> [P(brine)]' from your nearest and farthest stacks
  (samples without a reflection are left at 0).
- Interpret > Well seeds from tied tops: small snapped horizon patches ("seed: <formation>") around every tied well;
  Compute RGT then honours them (also across faults) and T grows a seed into a full horizon.
Test first: on a 3D with tied wells, Seismic inversion > set wavelet > Resolution (window around the reservoir), then
Single-well inversion at a well with a known pay zone.

## AVO, fizz gas, angle-stack risk, DHI checklist (PLAN item 25, Seismic v0.48.0)
- Seismic inversion > Pre-stack tab: "AVO volumes" writes Intercept, Gradient, AVO anomaly (distance from the wet
  background trend in standard deviations; strongly negative = hydrocarbon side) and Fluid factor from the nearest and
  farthest stacks - show them on sections / time slices or as overlays (item 23).
- "Angle-stack risk": for your actual angle range and stack count, the expected errors of Zp, Zs and density from the
  stacks vs a full gather; warns when the far angle is below ~35 deg (density poorly constrained).
- Fizz gas (avo.fizz_assessment, used by the DHI work): Vp drops almost as much as commercial gas, density does not -
  the observed density change from the pre-stack inversion is compared with both Gassmann cases and the result says when
  the density error is too large to decide. Low-frequency shadow / attenuation are reported as supporting evidence.
- Map layers > Add > DHI checklist: pick an amplitude grid and the structure grid; amplitude strength and conformance
  (does the anomaly outline follow a structural contour?) are computed, flat spot / phase change / edges / data quality
  are your scores; weighted DHI index with reasons, stored with the project for prospect risking (item 28).
Test first: AVO volumes on a project with near / far stacks, then View the "[AVO anomaly]" volume on an inline over a
known gas sand.

## Structural timing, paleo-structure animation, restoration (PLAN item 26, Seismic v0.49.0)
- Interpret > Paleo-structure animation of this section (GIF): the shown inline / crossline / 2D line flattened on each
  visible horizon from the oldest to the youngest, younger data blanked - works on any volume you are displaying
  (seismic, AI / Vp/Vs from the inversion, P(gas) ...).
- In code for now (UI next): tectonic phase per age-table unit (pre- / syn- / post-rift, inversion) with Wheeler
  colours; fault growth (throw per horizon, expansion index per interval) proposing the syn-rift interval;
  decompacted paleo-depths of every horizon at the time of a younger one; block restoration of a fault on a section
  (cut-offs extrapolated to the fault, one step per horizon) with line-length / area checks.
Test first: show an inline with 3+ interpreted horizons, Interpret > Paleo-structure animation, open the GIF.

## Migration and seal (PLAN item 27, Seismic v0.50.0)
- Map layers > Add > Migration (select a depth / time structure grid first): drainage areas per trap (grid layer) and a
  label per trap with its closure, fill % and where it spills (T2 -> T1, or "leak" off the map), from buoyant flow
  paths, spill-point analysis and fill-spill of a uniform charge over each drainage area.
- In code for now (UI next): critical moment (share of the source-rock transformation after the trap formed), fault
  juxtaposition (sand-on-sand windows), shale gouge ratio + supported column (editable calibration), slip / dilation
  tendency from a stress tensor, top-seal column from Vsh or MICP, and the limiting column (spill vs fault vs top seal).
Test first: a depth map with 2-3 closures, Add > Migration, check that the spill directions make geological sense.

## Prospect volumetrics + risking (PLAN item 28, Seismic v0.51.0)
- Map hub (code + LayersPanel.prospect; a prospect window is the next UI step): pick a trap of a structure grid, the
  contact comes from the spill point (or a fault-seal / top-seal column if given), GRV with a gross thickness inside the
  closure, Monte-Carlo volumetrics (NTG, porosity, Sw, FVF, RF as distributions) -> P90 / P50 / P10, mean, Swanson's
  mean; Pg = source x reservoir x trap x seal x timing (prefilled from the critical moment, seal and DHI evidence),
  DHI revision of Pg, dependent-prospect portfolio (shared source), CSV with Rose-style columns.
Test first: nothing to click yet beyond the API - check the numbers on a known prospect with python -c (see HANDOFF).

## Spacing, drainage ellipses, gun barrel (PLAN item 29, Seismic v0.52.0)
- Dashboard toolbar "Spacing + ellipses...": SHmax azimuth -> inter-well distances (horizontal along the overlapping
  lateral length + vertical offset), nearest neighbour per well, parent / child / co-completed (when first-production
  dates are loaded) and a drainage ellipse per lateral (frac half-length along SHmax) drawn on the dashboard map.
- "Gun barrel": section across the laterals (offset vs TVD) with the wells and their drainage widths; the section line
  is shown on the main basemap.
- Readers for Texas RRC, NDIC and FracFocus CSV exports (column names mapped to one schema) and the World Stress Map
  (SHmax from nearby A-C quality points) are in seismic_tool/spacing.py (UI hook-up next).
Test first: Dashboard > Wells from project (a pad of laterals) > Spacing + ellipses (your SHmax), then Gun barrel.

## Depletion, recovery factor, RTA, frac complexity, M&A screen (PLAN item 30, Seismic v0.53.0)
- Dashboard (after Spacing + ellipses): recovery(...) gives each well's recovery factor (EUR / in-place inside its
  drainage ellipse from a SoPhiH map) and each child well's depletion factor (parent cumulative inside the child's
  ellipse at its first production / in-place) - UI button with the production import is the next step.
- In seismic_tool/depletion.py: remaining-resource map, undrilled locations at a chosen spacing, square-root-time RTA
  with end of linear flow (rate-only proxy flagged), frac complexity index (fluid type, volumes, cluster spacing,
  stress anisotropy, brittleness) -> drainage shape, type curves P10 / P50 / P90, monthly cash flow / NPV10, PDP + PUD
  inventory and a one-page HTML area summary.

## Live Map + map bus (PLAN item 31, Seismic v0.54.0, Well Correlation v0.13.0, Mistie Tool 2.1.0 - run 6, 25 Nov 2026)
- **Map bus** (every tool): selection, filters, the active inline / crossline / 2D line / well / section, and
  "data changed" events (picks, tops, faults, misties, velocities, grids, layers) are shared between the Seismic tool,
  the Well Correlation Tool and the Mistie Tool - also across their separate windows / processes, through a small
  folder next to the well project (`<well project>_mapbus`: events.jsonl, state.json, welltable.csv). A well clicked
  in one tool is selected in the others; a 2D line picked in the Mistie Tool opens in the seismic window; an inline
  chosen anywhere follows. The shared **WellTable** is a column store every tool writes its own columns to (X / Y,
  `Top <formation>` in TVDSS from the correlation window, misties, petro values ...). The Mistie Tool joins when its
  project config has `"well_project": "<path to .wcproj.json>"` (or `"mapbus_folder": "<folder>"`).
- **Live Map window** (Seismic tool, View > Live Map, Ctrl+Shift+M; dark theme): the item-21 map hub layers on the
  left, map in the middle (scale bar, north arrow from the project CRS, X / Y / lat-lon / IL-XL readout), Well style +
  Filters on the right, well table + crossplot + histogram at the bottom - all linked (Spotfire style).
  - Well style: colour fixed / continuous ramp / categorical / **rules** (`Mistie_ms > 10 = red; [Net pay] > 5 =
    green`), size by column (bubbles), shape and label by column, pies, deviated well paths.
  - Filters: range or category per WellTable column, wells hidden instantly in the map, table and crossplot.
  - Lasso select, click / Ctrl+click wells, table-row selection; all go to the bus.
  - Seismic: current inline / line drawn live (yellow dashed), click a line to open it, **Arbitrary line** tool,
    shotpoint annotation, mistie residual markers (green < 2 ms, orange < 5, red) from the seismic Mistie analysis or
    the Mistie Tool.
  - **Well section** tool: draw a line (or use the selected wells) -> a cross section "Live Map" opens in the Well
    Correlation Tool; its section line is drawn in cyan. Picked tops arrive in the WellTable and recolour the map.
  - **Live grid...**: grid a tops column or a horizon - coarse grid at once while you pick, full grid after a pause,
    fault polygons as barriers; **Live depth maps** follow the dynamic depth model.
  - Basin / Wheeler section locations (Basin dialog > colour the Wheeler) appear as clickable cyan lines.
  - Add > **Add shapefile / culture layer** with presets: leases own / competitor / open, PLSS, pipelines (dashed),
    facilities, roads, no-go areas (hatched), coastline; GeoJSON read + write; CRS mismatch warning.
  - Export map (PNG / PDF with title block, legend incl. the well colours) and Export GIS (shapefiles + GeoJSON +
    GeoPackage + KML, grids as GeoTIFF + **ZMAP+**, wells_table with every column, section lines, 2D picks).

### Test this first (Live Map)
1. `Run all tests.bat` - Seismic, Well Correlation and Mistie self-tests + GUI tests (real Qt offscreen).
2. Seismic tool, F3 project: View > Live Map. Pan / zoom: scale bar + north arrow stay in the corners, the readout
   shows X / Y / IL-XL. Set the CRS first if lat/lon is missing (Map layers > Add > Project coordinate system, 23031).
3. Click wells, lasso a few, Ctrl+click: the seismic basemap rings and the Live Map table follow. Clear selection.
4. Well style: Colour by > continuous > a column; Rules with `Top Truncation > 1000 = red`; Size by; Apply.
   Filters: Add filter on a column, narrow the range - wells disappear from map, table and crossplot.
5. Open the Well Correlation Tool on the F3 well project at the same time: select a well there -> selected in both
   seismic windows within ~0.3 s; move a top -> the Live Map recolours if styled by that Top column.
6. Live Map > Well section: draw a line across 3 wells, double-click -> section "Live Map" in the correlation window.
7. Live Map > Arbitrary line across the F3 cube -> arbitrary section window; current inline shows as yellow dashes.
8. Live grid... > Column: Top <horizon> -> grid + contours; move a top in the correlation window: coarse grid at
   once, full grid ~1.5 s later.
9. Add > Add shapefile / culture layer: any lease / pipeline shapefile you have with the matching preset.
10. Export map (PDF) and Export GIS to a folder; open the folder in QGIS / the .zmap in Petrel or Kingdom.

## Automated fault mapping (PLAN item 32, Seismic v0.55.0 - run 6, 25 Nov 2026)
- Interpret > **Automated fault mapping**: 1. fault likelihood volumes (Hale-style: dip-steered semblance, fault
  orientation from the structure tensor, smoothing in the fault plane, 1 - s^8) + thinned likelihood (TFL) + fault
  strike and dip volumes - all viewable like attributes; 2. extract faults: TFL points grown into fault "skins" with
  consistent orientation (crossing / conjugate faults stay apart), table with size, strike, dip, likelihood, max throw
  (from the RGT volume when one exists, else by cross-correlation); filters by strike / dip / size / throw, stereonet
  (poles + Kamb density) and rose plot, merge suggestions for halves of one fault, split, rename, delete small;
  3. create faults: every skin becomes a normal editable fault (sticks every 5th inline or crossline + surface);
  fault polygons on the active horizon (heave polygons, downthrown ticks) as a map layer that the Live Map gridding
  uses as barriers; 2D lines: fault cuts on every line, correlated across intersecting lines into named faults.
- Optional CNN (onnxruntime / torch + a model file) is detected; the numpy method is the default.

### Test this first (fault mapping)
1. F3: Interpret > Automated fault mapping > 1. Fault likelihood volumes (watch the time per inline block - report
   it), then look at '<F3> [TFL]' on a time slice and an inline.
2. 2. Extract faults with TFL threshold 0.5 / minimum size 2000; check the table strikes against the known F3 fault
   trends (NW-SE graben faults), tighten the filters, 3. Create faults, then look at the sticks on inlines.
3. Pick or load a horizon, Fault polygons on active horizon, then Live Map > Live grid on that horizon: contours must
   break at the polygons.
