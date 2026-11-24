# CHANGELOG

## 2026-11-24 (d) — PRE-REGISTRATION: are lag-0 cells relatively more active while the plan is loaded (encoding) than while it is executed?

Written BEFORE running. Script `scripts/lag0_loading_firing.py` → `group/lag0_loading_firing_<date>/`.

**SK's hypothesis.** The plan is loaded during encoding, and "now" has to be known before anything further ahead can be planned. Cells tuned to the current location should therefore fire relatively more during encoding than during correct loops: over the phase, at correct uncovers, and in ripples.

**Context.**
- We have already seen 54 correlations of lag-0 tuning with ripple firing (entry (c)), including mPFC per-repeat correct-loop tuning × encoding-ripple firing at +0.19 (permutation p = .082).
- The F3 test (2026-11-14) was null, but it used the old tuning (mode maps, stale labels) and pooled mPFC with mOFC.
- So x, the measures and the primary tests are fixed here.

**x (tuning).** Per-cell `r_lag000_noctrl` from `per_lag_loopsets_2026-11-23/per_repeat/execute`. This is SK's a-priori "true tuning": correct loops, the most data, the Fig 3c estimator. It never sees an encoding loop. Final labels.

**Cells.** PRIMARY mPFC. Secondary: mOFC, mid HC, anterior HC.

**y, per cell, three families. Each is encoding − correct loops, in two readings.** Encoding / correct loops = before / from a layout's first correct loop (`phase_windows`, `phase_labels`; the same split as Fig 3c).

**A. Phase firing,** inside visits.
- *Own:* log rate ratio of encoding to correct loops, log((N+0.5)/T).
- *Location-matched:* log((N_enc+0.5)/(E_enc+0.5)), where E_enc = Σ over squares of the cell's correct-loop rate at that square × its encoding dwell there. That is, encoding firing relative to what the cell's own correct-loop place map predicts.
- Needs ≥ 30 s per phase.

**B. Correct uncovers** (reward presses). Rate 0.05–1.0 s after the press.
- *Own:* minus the rate −0.5…−0.05 s before it, divided by the cell's mean rate; encoding mean − correct-loop mean.
- *Matched:* encoding uncovers paired 1:1 with correct-loop uncovers on the stillness that follows (`fin.match_session`, as N1/N2); the post-press rate difference divided by the mean rate.
- Needs ≥ 15 events per phase (MIN_EVENTS).

**C. Ripples.** Rate within ±150 ms of the peak, divided by the cell's mean rate, as F3a.
- *Own:* minus the ripple's own flanks (0.5–1.0 s on both sides).
- *Matched:* minus ripple-free still moments of the same phase in the same stratum (time since the last press × time to the next).
- Encoding ripples (first traversal + rest of encoding) minus correct-loop ripples. Needs ≥ 20 ripples per phase.

Each family's encoding value alone (vs its own baseline) is stored as secondary.

**Test.** Spearman of x with y across cells. Null: x permuted within session, 10,000 times (`lag0_cells.spearman_within`, the F3 function). Prediction ρ > 0; reported two-sided as in F3.

**PRIMARY:** mPFC, 3 families × 2 readings = 6 tests. Raw p and Holm-corrected across the 6 are reported.

**Reported controls:** ρ(x, log mean rate) and ρ(y, log mean rate).

## 2026-11-24 (c) — Lag-0 tuning vs ripple firing (claim 2c) with final labels, all cells, every tuning version

New `scripts/lag0_tuning_vs_ripple_firing.py` (+ `_figures.py`) → `group/lag0_tuning_vs_ripple_firing_2026-11-24/`.

**Method:**
- x = per-cell `r_lag000_noctrl` from `per_lag_loopsets_2026-11-23`: mode / per-repeat × correct / all / encoding loops; final labels.
- y = the unchanged per-cell firing measures: claim 2c's discovery ripple − no ripple, and ripple modulation in encoding and in correct-loop ripples.
- Test: Pearson across cells, within-session permutation (imported `perm_corr`). ROIs: mPFC, mid HC, anterior HC.

**Results:**
- **Claim 2c reproduces:** +0.152, p = .059, n = 148 (published +0.15, p = .065).
- **Nothing becomes reliable:** 1 of 54 Pearson and 2 of 54 Spearman tests at p < .05.
- **mPFC encoding tuning × encoding-ripple firing:** +0.09, p = .22 (per-repeat).
- **Largest mPFC value:** per-repeat correct-loop tuning × encoding-ripple firing, +0.19, p = .082 (Spearman p = .042).
- **Reliability:** per-cell lag-0 r, encoding vs correct loops, ρ ≤ 0.11 with either estimator. The per-cell measure is capped, as in the split-half result.

## 2026-11-24 (b) — Duration-matched baseline for ripple co-firing and coupling; same windows in both panels

SK's questions:
1. Why does the coupling use fixed-width windows while the co-firing uses own-length windows?
2. Is own-length ripples vs ordinary moments really not significant?

**Answer to (2).** As stored (11-23), own − ordinary IS significant for co-firing: +0.12, p < .001. I had reported only fixed − ordinary. That contrast is unmatched: ordinary moments had one width, ripples a variable one.

**New, in `scripts/cofiring_window_controls.py`:**
- **`ordinary_matched`:** ordinary moments with widths drawn from the same session's ripples of the same phase.
- **Contrasts:** own − ordinary_matched, fixed − ordinary, own − ordinary (reference).
- **Output:** `group/swr/cofiring_window_controls_2026-11-24/`.
- **Reproduction:** own / fixed / ordinary reproduce the 11-23 run exactly.

**Results** (encoding loops, 22 sessions):
- **Co-firing:**
  - ripples own +0.127 vs ordinary matched +0.114 (both p < .001); ripples − ordinary p = .26 (correct loops .94).
  - Duration correlates with mean z equally in both: HC .37 vs .36, mPFC .28 vs .27.
  - **There is no ripple-specific HC–mPFC co-firing.**
- **HC now × mPFC now, firing level kept (= Fig 4h read-out):**
  - ripples own +0.027 (p = .032) vs ordinary matched +0.011 (p = .073); ripples − ordinary p = .25.
  - Fixed width: p = .040 (gain kept), .17 (gain removed).
  - **The coupling is above 0 in encoding ripples but not shown to be ripple-specific.**

**Figure:** `core_results_two_frames.py` panels e–f now use the same four windows and the gain-kept coupling. The windows: ripples own / ordinary matched (hatched) and ripples fixed / ordinary fixed (solid), with Δ p brackets. Re-rendered into `group/core_results_two_frames_2026-11-24/`; the `.md` is updated.

## 2026-11-24 (a) — Overview figure, version 2: encoding vs correct loops, both estimators, ripple co-firing and coupling

`scripts/core_results_two_frames.py` rewritten (replaces the 11-23 version) → `group/core_results_two_frames_2026-11-24/` (`.png/.pdf/.svg`, `result.json` with every plotted value, `.md` with caption, coherence check and explanation). It plots stored results only.

**Panels:**
- **a–d:** mid HC + mPFC lag profiles (session ± SEM, 0° repeated faded at 360°).
  - Rows: estimator (mode = published Fig 3c; per-repeat = new).
  - Columns: encoding loops / correct loops.
  - Fig 3c window stars.
- **e:** HC–mPFC all-cell co-firing: own-duration ripple windows, fixed-width ripple windows, ordinary moments.
- **f:** HC now × mPFC now coupling (fixed width, gain removed) vs ordinary moments.

**Coherence checks (no new analysis):**
- **Cells:** in the 61 ripple sessions, the ripple pipeline's cells are the same 905 cells, with the same final labels, as the per-lag table.
- **Phase split:** the lag analysis (`correct`/`rep_correct`) and the ripple analysis (`trial_correct`) give identical per-layout counts of encoding loops in all 22 coupling sessions.
- **HC:** pooled (anterior + mid) in e–f; mid HC only in a–d.

**Values:**
- **mPFC window (30+60°) vs other lags:** p = .004 (mode) and .014 (per-repeat) in correct loops; p ≥ .61 in encoding loops.
- **Mid HC window (0+330°) vs other lags:** p = .057 / .004 in encoding loops; .003 / < .001 in correct loops.
- **mPFC 0° in encoding loops:** +0.052 (mode, p = .043), +0.042 (per-repeat, p = .091).
- **mPFC current square in encoding ripples:** z = +0.31, p = .13 (26 sessions). This is SK's question: up in task time but not in ripples. The reasons are in the `.md`.

## 2026-11-23 (g) — Overview figure for SK's supervisor: the two frames

New `scripts/core_results_two_frames.py` → `group/core_results_two_frames_2026-11-23/` (`.png/.pdf/.svg`, `.md` with caption). It plots stored results only.
- **a. Task time:** the new Fig 3c (per-repeat maps, correct loops, session ± SEM, window stars).
- **b. Real time:** reported Fig 4h HC location in ripples, encoding vs correct loops.
- **c. Real time:** HC now × mPFC now coupling (fixed width, gain removed) vs ordinary moments.

SK's convention from now on:
- the per-cell lag measure for the task-time future-angle frame;
- the population read-out whenever events are in real time (ripples).

Anterior HC is drawn in the CLAUDE.md colour `#23677E`. `overlay_double_dissociation.ROI_COLOURS` still has `#a30d6c` (the lOFC colour) for anterior HC; not changed yet.

## 2026-11-23 (f) — CORRECTION: ripple co-firing was a window-length artefact; only HC now × mPFC now in encoding ripples survives

New `scripts/cofiring_window_controls.py` (+ `_figures.py`) → `group/swr/cofiring_window_controls_2026-11-23/`.

**The flaw.** Ripple windows use each ripple's own duration; ordinary moments use one fixed width. Longer windows carry more spikes in every cell of both regions: across ripples, r(duration, mean z) = .37 (HC) and .28 (mPFC).

**All-cell HC–mPFC co-firing** (entry (e), r = 0.13, p < .001) is that artefact:
- duration partialled out: +0.014, p = .30 (encode) and −0.007 (execute);
- fixed-width windows: +0.020, p = .13;
- no different from ordinary moments.

**Content coupling, rerun with fixed-width windows and the per-event population gain removed:**
- HC now × mPFC now, encoding ripples: **+0.034, p = .022** (reported own-length version .027, p = .032). Its excess over ordinary moments is p = .040 with the gain in, .17 with it removed.
- HC now × mPFC next, correct loops: **+0.002, p = .84** (reported .028, p = .028). It is carried by shared gain. The "partner switch" from entry (c) is withdrawn.

Banners added to `tuned_cofiring_hc_mpfc.md` and `now_coupling_hc_mpfc.md`.

**Not affected:**
- tuned co-firing panels b and c (centred tuning weights cancel a uniform gain; the null shuffles tuning);
- the single-region Fig 4h location estimator (its flanks are duration-matched, and the location signal is also present at fixed-width ordinary moments).

## 2026-11-23 (e) — Old tables archived, one cell-count file, manuscript value check, tuned-cell co-firing in ripples

**Archive (SK: "once and for all").** `scripts/archive_stale_tables.py --move` moved 173 items (27 GB) to `derivatives/_archive_old_labels_2026-11-23/` (`manifest.csv`, `README.md`; relative paths kept, so it is reversible).
- What moved:
  - every result folder under `group/` holding a per-cell table with old ROI labels;
  - the four 11-23 outputs built on them;
  - 14 superseded label tables at the top of `derivatives/` (`human_cell_labels*.xlsx`, `ROI_assignment_27thofAug_2026/`, `neurons_MNI.csv`, `.numbers` copies, `label_qc*`).
- Kept: `ROI_assignment/`, `neurons_with_ROI_labels.csv`, `neurons_MNI_latest.csv`, `old_electrode_tables/`.
- Fresh audit: 52 per-cell tables in `group/`, 0 stale.
- Repointed: `per_lag_encoding.py` (`RELOAD_FROM = None`), `per_lag_encoding_replot.py` (Aug 28 per-lag, Sep 24 sustained-state, Aug 31 RSA cell list), `manuscript_cell_results_core.py` (Aug 31 RSA run). Older exploratory scripts naming archived runs now fail loudly.
- `atlas_roi` is the final column; `alt_final_roi` equals it for the six analysis ROIs and is NaN for the 60 other cells.

**Cell counts.** New `scripts/cell_counts_summary.py` → `derivatives/CELL_COUNTS.md` + `cell_counts.json`, computed from the tables the reported analyses read:
- cohort 984 QC / 924 analysis / 63 sessions / 42 patients;
- per-ROI counts for Fig 2c, Fig 2d RSA, Fig 3c, Fig 3d (sites) and Fig 4h sessions.

**Manuscript check** (whole Google Doc read) → `group/paper_value_check_2026-11-23.md`.
- 20 corrections entered as tracked suggestions (Docs API `writeMode: SUGGEST`):
  - patients 41 → 42;
  - Fig 2c counts (old labels);
  - Fig 3c sessions 32/35 → 33/36, mHC 232 → 233;
  - Fig 3d outside 81 → 71;
  - Methods RSA counts (pre-relabel set);
  - QC panel a 58 → 36;
  - the whole alternative-estimator supplement (old-label spatial-peaks run → Aug 28 run).
- Flagged, not changed:
  - supplement mPFC window-vs-other-lags now q = .054;
  - Fig 2d mPFC q .037 (Aug 27 family) vs .033 (Aug 31);
  - per-repeat vs mode values for Fig 3c/3d;
  - Fig 3d is the only per-cell test (per session, ventral 30° is n.s.);
  - LFP 180/181 and 31/32 inconsistencies;
  - demographics unverifiable.

**Tuned-cell co-firing in ripples** (SK: "just the firing of the cells with higher CV r"). New `scripts/tuned_cofiring_hc_mpfc.py` (+ `_figures.py`) → `group/swr/tuned_cofiring_hc_mpfc_2026-11-23/`.
- Per event: T = mean((r − mean r) × z) per region.
- Per session: r(T_HC, T_mPFC) minus a tuning-shuffle null (same function); 22 sessions.
- All-cell HC–mPFC co-firing in ripples: r = +0.13 (p < .001) in encoding and in correct loops; ≈ 0 at ordinary moments.
- Tuned coupling ≈ 0 everywhere:
  - HC now × mPFC now: encoding ripples −0.003 (p = .87), correct-loop ripples +0.015 (p = .26);
  - HC now × mPFC 30–60°: n.s.;
  - one p = .044 (ordinary moments, negative) among about 20 tests.
- Reading: HC and mPFC co-fire in ripples, but not organised by tuning strength.

`now_coupling_hc_mpfc_figures.bar` now prints "p < .001" instead of "p = .000".

## 2026-11-23 (d) — Fig 3d (gradient split) rebuilt with per-repeat maps; the split itself tested

**Rebuild.** Same cells, final labels, coordinates, fMRI angles and PC1 median split (48 ventral / 39 dorsal cells, 12 / 7 sites). Only the rate-map estimator changes.
- `cell_gradient_master_table.py` gains `--cell-table` and `--label`.
  - `mode` → `cell_gradient_master/2026-11-23_14-22-09_mode_execute/` reproduces the published master exactly.
  - `per_repeat` → `…_14-22-23_per_repeat_execute/`. Fig 3d panels are in `final_splits/`, guide `fig3d.md`.
- `cell_fMRI_angle_match.py`: the profile band for the unweighted split is now ±SEM across cells (as Fig 3c); it was a 16–84 % bootstrap band.
- **Per-repeat numbers:**
  - Peaks unchanged: ventral 30°, dorsal 60°.
  - Ventral 30°: r = 0.096, t(47) = 2.59, q = .013 (published 0.079, 2.44, .019).
  - **Dorsal 60°: r = 0.084, t(38) = 1.99, p = .027, q = .054** (published 0.081, 2.21, .033). Session unit q = .032.

**Direct test of the split.** New `scripts/fig3d_gradient_split_check.py` → `group/fig3d_gradient_check_2026-11-23/` (`split_check.png` + `.md`, `result.json`).
- **Shift S** = d(dorsal) − d(ventral), with d = z(r60) − z(r30):
  - cell t-test p = .023 (both estimators);
  - exact site permutation (50,388 assignments) p = .11 (mode) / .092 (per-repeat);
  - session means p = .11 / .14.
- **Peak stability** (site bootstrap):
  - ventral = 30° in 86 % / 93 % of draws;
  - dorsal = 60° in 63 % / 56 %;
  - within the dorsal group, 60° − 30° is n.s. (cell p = .40).
- **MNI-z median split:** both groups peak at 30°, and the shift is n.s.
- **Reading:**
  - the ventral 30° end is solid; the dorsal 60° end and the split itself are not, once sites are the unit;
  - this was already true for the published version, so the per-repeat estimator is not the cause;
  - present the split as descriptive.
- **Side observation:** the 71 outside-mask cells (more dorsal, MNI z ≈ 14) peak at 0° (t(70) = 3.04, p = .003). They, not the ventral tip, carry the new mPFC 0° code.

## 2026-11-23 (c) — Final ROI labels everywhere, per-repeat maps, Fig 3c rebuilt, when mPFC codes now, HC–mPFC coupling

**ROI labels (SK: "absolutely crucial").**
- **Final assignment:** `derivatives/ROI_assignment/cells_step5_final_labels.csv` (`atlas_roi`, Aug 28 run).
- **New module `mc/analyse/roi_labels.py`:**
  - `read_per_cell_table` re-applies the final labels to any saved per-cell table;
  - `assert_final` is now called by `rrsa.cell_roi_table` and `cs.load_roi_table`; both already matched.
- **Stale source:** `per_lag_encoding/2026-08-04_…_relabelled-final` differs on 33 cells (18 mOFC→HC, 6 mOFC→mPFC, 3 mPFC→mOFC, 3 EC→HC, 3 within HC).
  - It was read by `preferred_lag_firing.PER_CELL` (and so by storage_vs_planning, lag0_cells, discovery_cells, plan_code_vs_ripples, per_lag_encoding_by_phase, first_loops, the 11-23 encode-vs-execute / cross-phase scripts) and by the Fig 3d / fMRI-angle scripts (`cell_fmri_lag_prediction`, `cell_fmri_continuous_angle_summary`, `plot_mean_cell_lag_brain_overlay`, ratemap example scripts).
  - All of them now read the Aug 28 table, whose r values are identical, through `read_per_cell_table`.
  - The Fig 3c source (Aug 28) was already correct.
- **Audit:** `scripts/audit_roi_labels.py` → `group/roi_label_audit_2026-11-23/`.
  - 368 saved per-cell tables; 316 carry at least one stale label, almost all May–Aug runs that are superseded.
  - Recent stale tables (not yet rerun): F2 by-phase (11-14), B1 plan-code (11-14), and the 11-23 encode-vs-execute, cross-phase and B1-lagfree outputs. Their `.md` guides carry a banner.

**Per-repeat rate maps (SK's decision) + Fig 3c rebuilt.**
- **Estimator:** `per_lag_encoding.py` now has `MAP_ESTIMATOR = 'per_repeat'`: every repeat with its own locations; dwell in bins per repeat. `'mode'` is the old one.
- **Run:** `scripts/per_lag_encoding_loopsets.py` → `group/per_lag_loopsets_2026-11-23/`.
  - Loop sets: execute (published) / all / encode × both estimators, final labels.
  - Check: mode + execute reproduces the published r for 924 cells (1e-16) and the old Fig 3c tests to the digit.
- **Fig 3c:** `overlay_double_dissociation.py` gains `publication_panel` (session-balanced ±SEM, window-test stars, lag colour bar) and now defaults to the per-repeat execute table → `per_repeat/execute/overlay_double_dissociation_noctrl/fig3c_lag_profile.pdf`.
  - mPFC window 30+60° vs other lags: p = .005 (cell) / .014 (session); was .002 / .004.
  - mid HC 0+330°: .0001 / .0004; was .012 / .003.
  - **mPFC 0° is now > 0** (+0.035 cell / +0.047 session, p < .05; old n.s.). The per_repeat − mode difference at mPFC 0° is p = .028 / .046.
  - Caveat: the estimator was switched after seeing the old result, so report mode as a sensitivity check.
- **All loops vs execute:** mPFC 0° is unchanged; 30° and 90° weaken (30°: p = .004 / .005).
- **Encode only:** mPFC 0° +0.034 (cell p < .05), and no future code (60° and 90° below execute, session p = .018, .032).

**When does mPFC code now? (learning stage)**
`scripts/per_lag_learning_stage.py` → `group/per_lag_learning_stage_2026-11-23/`.
- **Method:** fixed reference (other configurations, all loops); test = the held-out configuration's loops by stage (encode / correct 1–2 / 3–5 / 6+); loop counts matched (20 draws).
- **mPFC 0°** (session unit):
  - encode +0.026 (p = .079), 1–2 +0.035 (.024), 3–5 +0.048 (.001), 6+ −0.009 (n.s.);
  - 6+ − 1–2: p = .08.
- **mPFC 30+60°:** absent in encode, builds up (trend p = .007).
- **mid HC 0°:** strongest in encode (+0.054) and weakening (cell trend p = .004, session .12).

**HC–mPFC current-square coupling.**
`scripts/now_coupling_hc_mpfc.py` → `group/swr/now_coupling_hc_mpfc_2026-11-23/`.
- **Method:** the reported location estimator on the same events; `events()` was split out of `location_content_ripples_by_phase.session`, reproducing it exactly.
  - Coupling = r(HC CURRENT, mPFC CURRENT or NEXT) across events − a same-square shuffle null; 22 sessions.
- **Results:**

  | | now × now | now × next |
  |---|---|---|
  | encode ripples | **+0.027, p = .032** | n.s. |
  | execute ripples | n.s. | **+0.028, p = .028** |

  - Encode − execute, ripple − ordinary and the partner switch are n.s. for ripples (p = .15, .14, .21); the partner switch for ordinary moments is p = .044.
  - mPFC CURRENT level per event: n.s. everywhere.
- **Reading:** exploratory (about 14 tests, 4 at p < .05).

## 2026-11-23 (b) — Estimator vs display, cross-phase tuning, lag tuning vs ripple firing, B1 on all loops

**Q1 — why F2's lag profile looked cleaner.**
`scripts/per_lag_estimator_comparison.py` → `per_lag_estimator_comparison_2026-11-23/`.
- Same 155 mPFC cells and 32 sessions.
- **Mostly the error bar:** F2 drew ±SEM, the new figures the 95% CI
  (≈ 2 × SEM).
- **F2's pooled estimator** (each repeat's own location instead of the per-bin
  mode) lifts 0°, 90° and the far lags by 0.01–0.04. 60° is unchanged
  (p = .029 vs .032); 90° is n.s. in both (p = .11 vs .30).
- **Per-cell split-half reliability** is not better: 0° −0.09 vs −0.02; 60°
  +0.19 vs +0.10.
- **Recommendation:** keep the published estimator, or adopt pooling only on
  a-priori grounds and everywhere.

**Q2 — encode vs execute tuning per cell.**
`scripts/per_lag_cross_phase_consistency.py` → `per_lag_cross_phase_2026-11-23/`.
- The published functions, run on extra loop sets: all, odd / even execute,
  odd / even encode.
- **Encode vs execute, lag 0:** ρ = +0.06 (mid HC), +0.11 (mPFC), n.s.
- **Split-half reliability of the published r:** |ρ| ≤ 0.17 at every lag;
  lag 0 +0.04 (mid HC), −0.02 (mPFC).
- **Reading:** per-cell lag tuning is not a stable cell property, so
  cross-phase consistency is capped at ≈ 0. Firing-rate tertiles don't help.

**Q3 — lag tuning vs ripple firing.**
- **x and y.** x = tuning from execute / encode / all repeats; y = claim 2c's
  measure, and in-vs-out-of-ripple firing over the whole encode / execute
  phase.
- **Overall:** 6 of 72 correlations at p < .05 per region (chance ≈ 3.6), no
  pattern.
- **Claim 2c** (mPFC lag 0):
  - as published +0.15 (p = .065, ripple-pipeline labels, n = 148);
  - per-lag labels +0.12 (within-session permutation p = .107);
  - without one cell +0.08;
  - not carried by firing rate, event counts or any one session.
- **Bug caught during this run.** `pivot_table` sorted the cells while the
  metadata kept first-appearance order, misaligning ROI, session and y. Caught
  because claim 2c came out at +0.02 instead of +0.12. Fixed, with an
  assertion that each execute value equals the published one per cell. The
  first, misaligned results were discarded.

**ROI labels differ between pipelines.** `rrsa.cell_roi_table` (ripple
pipeline) vs the published per-lag `relabelled-final` table:

| ripple pipeline | per-lag table | cells |
|---|---|---|
| mPFC | mOFC | 6 |
| mOFC | mPFC | 3 |
| HC | mOFC | 18 |
| HC | EC | 3 |

Not changed; needs one canonical table.

**Q4 — B1 on all execution loops, and lag-free codes.**
`scripts/plan_code_vs_ripples_lagfree.py` → `swr/plan_code_vs_ripples_lagfree_2026-11-23/`.
- B1's saved block codes and B1's functions. It reproduces the
  learning-ripples figure exactly: future mean, first 3 loops +0.31,
  p = .006; all loops +0.19, p = .125.

| code variant | first 3 loops: β (p) | all loops: β (p) |
|---|---|---|
| each cell's best lag, chosen on its other blocks | +0.20 (.058) | +0.07 (.62) |
| max over lags | +0.07 (.66) | +0.06 (.52) |

- First 3 − all: n.s. for every variant.
- Pooled partial r for the future mean: +0.085 vs +0.082. The effect is tiny
  either way.

## 2026-11-23 — Encode vs known with OTHER-configuration maps: current square, per-lag profiles, path and rewards in ripples

**Request (SK).**
1. HC current square, encode vs known, in ripples and outside them, compared
   with the published per-lag result (current-square coding in execution).
2. The published per-lag profile on the encode loops, mid HC and mPFC, at
   the cell and session unit.
3. In ripples: rewards vs current square; next reward vs other rewards.
4. Are remote / future / reward squares represented DURING ripples, and
   ordinary place firing otherwise?

All location maps come from OTHER configurations.

**Part 2: `scripts/per_lag_encoding_encode_vs_execute.py`**
→ `per_lag_encode_vs_execute_2026-11-23/`
- **Method.** `per_lag_encoding.py` functions imported unchanged; only the
  input rows change.
  - encode = `correct == 0 & rep_correct == 0` (the rows the published run
    excludes);
  - execute = the published `all_minus_explore`;
  - execute_matched = the first n execute rows per configuration, n = its
    encode rows.
- **Cells.** The published per-cell table: mid HC 232 cells / 35 sessions,
  mPFC 155 / 32.
- **Check.** Execute reproduces the published `r_lag*_noctrl` (max |diff|
  1e-16, 387 cells).
- **Deviations** (none changes r): no per-cell permutations; noctrl and the
  12 lags only.
- **Tests.** `lagwise_aggregation` (Fisher z): vs 0 one-sided; differences
  paired within cell, two-sided; BH across 12 lags.
- **Mid HC 0°.**
  - encode +0.047 (cells p = .001; sessions p = .022);
  - execute +0.041 (.004; .015);
  - encode − execute +0.005 (p = .80) / −0.018 (p = .50).

  **Current-square coding does not depend on phase.**
- **mPFC 30° (cells) / 60° (sessions).**
  - execute +0.051 (p = .003) / +0.068 (p = .016);
  - encode −0.004 / +0.000;
  - encode − execute −0.059 (p = .031) / −0.072 (p = .085).
- **But the matched control kills the phase reading.** Execute with the encode
  loop count is also ≈ 0 (−0.007 / +0.011), and encode − matched is n.s. at
  every lag. The future code needs many repeats to be detectable.
- **mPFC 0°.** encode +0.035 (p = .048) / +0.058 (p = .026); matched
  sessions +0.050 (p = .027); full execute n.s. Not a significant
  difference.

**Parts 1, 3, 4: `scripts/location_content_ripples_by_phase.py`**
→ `swr/location_content_by_phase_2026-11-23/`
- **Method.** `swr_phase_contrast` imported: the same ripples, cells,
  sessions, phases, other-configuration maps, ripple-duration window,
  permutations and count-matched draws.
- **Check.** CURRENT in ripples reproduces the reported per-session values
  (max |diff| 1e-14).
- **Added.**
  - Ordinary moments: median-ripple-width windows tiled every 50 ms inside
    visits, no ripple within 0.5 s, the same n per session as the ripples.
  - Roles past 1–3 / next 1–3 (block visit sequence), next reward, other
    rewards: distance-controlled (minus the other squares at the same
    distance from the current square).
  - Rewards − current.
- **Current square, HC pooled** (count-matched z):

  | | ripples | ordinary |
  |---|---|---|
  | encode | +0.57 | +0.28 (<.001) |
  | known | +0.10 | +0.10 (.024) |
  | encode − known | +0.46 (.008) | +0.18 (.012) |
  | ripple − ordinary, encode | +0.28 (.065) | |
  | ripple − ordinary, known | −0.00 (.99) | |

  **The encode > known effect is not ripple-specific.** Known ripples carry
  the current square exactly as much as ordinary moments do.
- **Rewards − current.** Negative everywhere (encode ripples −0.50,
  p = .001). Next − other rewards in ripples: n.s. in both phases.
- **Other (non-next) rewards in known ripples:** −0.31 (p = .006);
  encode − known +0.50 (p = .007).
- **Path ahead in encode.**
  - Ordinary moments: next 1 +0.14 (p = .002), next 2 +0.16 (p = .006), next
    reward +0.14 (p = .031).
  - Ripples: similar (next 3 +0.29, p = .043).
  - Ripple − ordinary: n.s. for every square ahead and every reward.
- **Past 1 in encode ripples:** −0.42 (p = .002); ripple − ordinary −0.38
  (p = .006).

**Reading.**
- No support for "remote / future / reward squares during ripples, place
  firing otherwise".
- The forward lean during encoding is in ordinary moments too.
- Ripples locally suppress the just-left square during encoding, and avoid
  the non-next rewards once known.
- Exploratory, uncorrected.
- The earlier F2 script (`per_lag_encoding_by_phase.py`) pooled repeats
  instead of averaging them; this run does not.

## 2026-11-20 — D3 vs the reported phase contrast: the location TEMPLATE decides the sign (corrects D3)

**Why.** SK: D3 said HC ripples carry the current square in execution
(+0.62, p < .001) but not in learning (+0.14, n.s.). The reported analysis
(`swr_phase_contrast`, Fig 20/21) says the opposite: encode +0.57
(p < .001), once known +0.21 (n.s.), count-matched difference +0.46
(p = .007). I reported D3 without reconciling it with the reported result.

**What differs.** The two estimators differ in four ways:
- template: D3 = the same block's other traversals (T4 / `future_shift`
  references); reported = other configurations (leave-one-`grid_id`-out);
- window: ±125 ms vs the ripple's own duration;
- ripple selection: D3 is stricter;
- count matching: D3 has none.

**Test.** `scripts/exploratory/location_estimator_reconcile.py` →
`swr/location_estimator_reconcile_2026-11-20/`. It uses the reported
ripples, cells (HC ant + mid), sessions, statistic, 1000 permutations and
1000 count-matched draws, and swaps one thing at a time.
- Check: other configs × ripple duration reproduces the reported per-session
  values (max |diff| 4e-16).

Values are HC current square, z: learning / execution (all ripples), then
learning − execution with counts matched.

| template | ripple duration | ±125 ms |
|---|---|---|
| other configs (reported) | **+0.57** / +0.21; **+0.46 (.007)** | **+0.66** / +0.24; **+0.51 (.004)** |
| other configs, same phase | **+0.35** / +0.21; +0.25 (.17) | **+0.56** / +0.21; **+0.42 (.017)** |
| same block (D3) | +0.18 / **+0.63**; −0.27 (.22) | +0.15 / **+0.81**; **−0.43 (.029)** |
| same block, same phase (n = 43) | +0.29 / +0.25; +0.17 (.43) | +0.29 / +0.36; +0.14 (.51) |

**Reading.**
- **The window and the ripple selection do not matter.** The template source
  alone flips the sign, on the very same ripples.
- **Same-block templates for learning ripples are mostly execution maps.** A
  median of 87% (IQR 76–94%) of their dwell time comes from execution
  traversals.
- **In execution, same-block templates are built from near-identical repeats
  of the same loop.** That is the same route, the same position in the
  sequence and the same next move, so they score more than location: the
  routine at that square, and configuration-specific remapping. These cannot
  be separated here.
- **Restricting same-block templates to the ripple's own phase removes the
  flip** (+0.29 vs +0.25).
- **Location in the sense of the manuscript is the other-configuration
  template.** On that measure the reported result stands: present in encoding
  ripples, not detectable once known. D4c (other configurations, within
  phase) agrees: learning ripples +0.58, execution +0.22.

**Consequences.**
- D3 (2026-11-15, 2026-11-19 (b)) measures same-block content and must not be
  read as location content.
  - Its learning values, and every learning-vs-execution comparison, are
    scored against execution maps.
  - Its within-execution categories (route / reward vs other) need a rerun
    with other-configuration templates before they mean location.
- The same template underlies T4 `path_content` (HC first traversal
  PAST − FUT +0.44; execution FUT +0.35) and `future_shift`. The T4
  learning-vs-execution flip carries the same caveat.

## 2026-11-19 — The position index by phase, with the PUBLISHED definition (corrects D4c)

**Why.** SK: D4c's "mOFC position-selective in learning, gone in execution"
contradicts the chapter-5 result (mOFC index in correct repeats). The two are
different measures:
- chapter 5 (group/position_index_2026-10-02): warped 360-bin traces of
  correct loops; goal progress (within-state cosine) and the per-loop linear
  ramp removed; per-cell leave-one-configuration-out encoding model; held-out
  r of the state block; one-sided Wilcoxon of per-cell r > 0, BH across ROIs.
- D4c: real-time state templates minus their linear A–D ramp, scored in
  250 ms windows in still moments and ripples only.
- The cosine goal-progress basis has zero mean within each state, so it does
  not change state means; D4c's "selective" removes the same linear trend at
  the level of state means. The definitions agree; the MEASUREMENT (whole
  loop vs 250 ms still windows) does not.

**What.** `scripts/position_index_by_phase.py` imports the published pipeline
(`encoding_state_sustained_cv`) unchanged, same per-cell seeds, 10,000 circular
shifts; only the averaged loops change: learning = loops before a block's
first correct loop; execution = correct loops (reproduces the published r and
q exactly); execution_matched = first n correct loops per configuration, n =
its number of learning loops. Output:
`group/position_index_by_phase_2026-11-19/`.

**Results** (Hodges-Lehmann r, q):

| ROI | learning | execution | execution, matched n | exec − learn (paired p) |
|---|---|---|---|---|
| EC | +.004 (.21) | **+.022 (.003)** | +.012 (.060) | +.022 (.031) |
| mOFC | **+.012 (.001)** | **+.013 (.001)** | **+.010 (.010)** | +.002 (.68) |
| mPFC | **+.018 (<.001)** | **+.010 (.002)** | +.002 (.30) | −.009 (.053); matched −.015 (<.001) |
| HC ant | **+.005 (.010)** | **+.009 (.002)** | +.003 (.099) | +.004 (.27) |
| HC mid | **+.012 (<.001)** | **+.009 (.005)** | **+.009 (<.001)** | −.003 (.56) |
| PCC | **+.019 (<.001)** | +.003 (.32) | +.006 (.13) | −.018 (.010) |

- mOFC carries the index in execution (as published) AND in learning,
  unchanged. D4c's "gone in execution" holds only for 250 ms still moments and
  must not be stated as a property of the code.
- The D4c timeline ("cortex has it in learning, HC only after") does NOT
  survive the published definition: HC mid has the index in learning; HC vs
  mOFC change, Mann-Whitney p = .90 (matched .96). Retracted.
- ⚠ PCC, the negative control in execution, is positive in learning. The
  learning measure therefore carries a state-locked signal that is not a
  position code. Preferred states are spread over A–D (not an onset
  transient); candidate: search / error presses distributed unevenly over
  states during learning. Learning levels are not interpretable as an index
  until that is controlled.

## 2026-11-19 (b) — context_content extended: square categories (D3) and BEFORE the ripple (D5). Exploratory.

`mc/analyse/swr_explore/context_content.py`, rerun into
`swr/context_content_2026-11-19/`. All 871 previously reported tests reproduce
exactly. Added: D3 CURRENT and ROUTE − OTHER; D5 = every D4 code in the window
before (−500…−50 ms) and after (+50…+500 ms) the ripple, and the HC location ×
frontal position coupling before the ripple; coupling LEVELS and their
learning − execution difference. Figures redrawn as bars (mean, 95% CI) on
SK's request: D3_square_content, D4c_code_levels, D4a_direction,
D5_before_ripples, D4b_binding.

⚠ **Corrected 2026-11-20.** D3 scores against templates from the same block's
other traversals. Those are mostly execution maps, even for learning
ripples. They are not a location measure: see the 2026-11-20 entry. The
numbers below are same-block content, not location content.

**D3, HC, which squares** (z; still level / ripple level / ripple − matched /
ripple − own):
- current square, execution: +1.44 (<.001) / +0.62 (<.001) / +0.05 (.70) /
  +0.12 (.44). Present, not ripple-specific.
- route − other (same distance), execution: +0.36 (.019) / +0.40 (.020) /
  +0.29 (.087) / +0.64 (.16, n = 10). Not ripple-specific.
- reward − other (same distance), execution: +0.65 (.001) / +0.57 (<.001) /
  **+0.32 (.023)** / +0.14 (.54, n = 24). One baseline only.
- reward − route, execution: +0.28 (.075) / +0.26 (.093) / +0.17 (.25) /
  +0.42 (.075). Rewards are not preferred over the rest of the route.
- Learning: nothing positive; route − other in still moments −0.55 (.037,
  n = 18).

**D5, before the ripple:**
- mOFC position code before learning ripples +0.91 (<.001), but as strong
  before matched still moments (+1.04): matched −0.04 (.88), own −0.03 (.92).
  Away-execution own +0.61 (.053), matched +0.26 (.29).
- Frontal (mPFC + mOFC) position code before learning ripples is LOWER than
  before matched still moments: −0.36 (.027); mPFC −0.54 (.005).
- HC location before ripples: not elevated (learning matched −0.20, .12).
- HC location × frontal position coupling before the ripple: every set ≈ 0
  (largest: execution matched +0.016, .070, driven by negative still-moment
  coupling −0.016, .001).
- Coupling in the ±125 ms window, still moments: learning −0.004 (.51),
  execution −0.006 (.17), learning − execution +0.002 (.85). No
  phase-specific binding outside ripples either.
- In the ripple flanks (±0.5–0.75 s), learning: HC location × mOFC position
  +0.043 (.003); in the ripple itself −0.023 (.16). Not predicted, one of
  many.
- HC conjunction before the ripple: all sets ≈ 0 (all, matched +0.12, .44).

**Reading.** No evidence that location and sequence are co-active before
ripples, or bound in them, or bound in one phase rather than the other. The
D4b-i matched result (+0.29, p = .012) remains a negative still-moment
level, not a ripple conjunction.

## 2026-11-15 — PRE-REGISTRATION: who drives whom, and when? (D1–D4)

Written BEFORE running.

**SK's model.**
- mOFC tracks position in the sequence (state) and lOFC tracks reward.
- At uncovers they signal the hippocampus to store the current position, and
  that is why a ripple follows (cortex → HC, "memorise").
- Later ripples run the other way, HC → cortex, to load the plan. Ripples
  outside the uncover period may carry the rewarded locations, more so once
  the sequence is known.
- A symmetric alternative: location and sequence information are played at
  the same time (bound), not one region sending to the other.

**Shared definitions.** Session = unit, seed 42, sign-flip over sessions
(10,000). Both readings everywhere.
- **Phases:** learning = grid onset → start of the first fully correct repeat
  (exclusive boundary); execution = the rest.
- **Ripple contexts:**
  - UNCOVER = in the pause after an uncover press, (press,
    press + min(pause, 2 s)];
  - AWAY = no uncover press in [t − 2 s, t + 0.5 s] (navigation presses
    allowed).
  - These are crossed with phase: U_L, U_E, A_L, A_E.
- **Matched reading:** ripple minus ripple-free moments (no ripple within
  0.5 s, 50 ms grid) of the SAME context and phase, in the same stratum of time
  since the last press × time to the next press.
- Ripples are aligned to their peak; the bundle stores no onset. Weakness:
  peak timing blurs leads and lags by a few tens of ms.

**D1 — does the HFB direction flip with context?**
`scripts/exploratory/swr_hfb_direction_context.py`. Partition ROIs mOFC,
lOFC, mPFC; 20 shifted nulls.
- Direction index = post (0, +250] − pre [−250, 0) ms of the ripple-locked
  epoch.
- Own reading: real − shifted, minus the |t| ≥ 750 ms baseline. Matched
  reading: ripple − matched moments.
- Model prediction: at U_L ripples cortex leads (index < 0); at later ripples
  it follows (index > 0).
- **PRIMARY: direction(U_L) − direction(A_E) < 0, own reading, in mOFC and in
  lOFC** (two tests, Bonferroni α = .025).
- Secondary:
  - mPFC;
  - U_L − A_L (context within learning) and U_L − U_E (phase within uncover
    ripples);
  - the 2 × 2 interaction;
  - time courses with cluster tests.

**D2 — does the cortical response to an uncover predict the ripple that follows?**
Same script.
- Per uncover (correct and error), single-trial ROI HFB in 0–500 ms minus the
  −1.5…−1.0 s baseline.
- Excluded: any ripple in [−1.5, +0.6] s (keeps the predictor and its baseline
  ripple-free), pauses < 1 s.
- Outcome: a ripple in (+0.6 s, press + min(pause, 2.5 s)].
- Per session and ROI, logistic regression on z(HFB) + log(window length) +
  correct/error + first traversal + phase. Stillness reading adds log(pause)
  and log(time since the previous press).
- β for HFB per session (≥ 10 events with and ≥ 10 without a ripple),
  sign-flip.
- **PRIMARY: β > 0 in LEARNING uncovers, mOFC and lOFC (α = .025 each).**
- Secondary: execution; learning − execution; mPFC (predicted ≈ 0).

**D3 — do ripples AWAY from uncovers carry the rewarded squares, more once the sequence is known?**
`mc/analyse/swr_explore/context_content.py`. HC cells, the T4
(`path_content` / `future_shift`) machinery: same-block references, ±125 ms
windows.
- Per ripple, evidence for:
  - REW: the configuration's reward squares;
  - ROUTE: squares on the block's execution route that are not rewarded;
  - OTHER: all non-rewarded squares.
- The current square and its grid neighbours are excluded. REW vs ROUTE is
  compared within equal Manhattan distance from the current square, then
  averaged over distances.
- Readings: stillness-matched controls (T4 strata, restricted to AWAY
  moments); own flanks.
- **PRIMARY: HC, AWAY ripples in EXECUTION, REW − ROUTE > 0, matched
  reading.**
- Secondary: learning; execution − learning (paired); REW − OTHER; UNCOVER
  ripples; mPFC and mOFC (location templates do not transfer there, so
  descriptive).

**D4 — cortex ↔ HC with the code that works in both: position in sequence (state).**
Same module. Sessions with ≥ 2 HC and ≥ 2 frontal cells (mPFC + mOFC
pooled; 32 sessions). Leave-one-configuration-out templates, as in
`state_content` (detrended state).
- **D4a, direction.** Per ripple, frontal state evidence pre [−500, −50] and
  post [+50, +500] ms, and HC state evidence pre and post.
  - asymmetry = z(ρ(frontal pre, HC post)) − z(ρ(HC pre, frontal post)),
    across ripples, minus the same with ripples paired at random (within
    session, context and phase). Matched reading: ripple asymmetry minus
    control-moment asymmetry.
  - Variant: HC LOCATION evidence in place of HC state.
  - Model prediction: asymmetry > 0 at U_L, < 0 at A_E.
  - **PRIMARY: asymmetry(U_L) − asymmetry(A_E) > 0, own reading.**
- **D4b, the symmetric option: is location and sequence information played at
  the same time?**
  - (i) Within HC, the location × state conjunction interaction
    (`state_content`'s b_both − b_loc − b_state). With equal counts per
    window this is a per-window linear contrast. Tested in ripples minus
    matched moments, z vs 200 permutations of the 36 labels.
  - (ii) ACROSS regions, in the same window: partial r(HC location evidence,
    frontal state evidence | total activity), ripple minus matched moments.
  - (iii) Within HC, location × state coupling by context.
  - **PRIMARY D4b: (ii) all ripples, ripple − matched > 0.** (i) and (iii) are
    co-primary-descriptive.
  - Everything is also given per context.
- Also descriptive: frontal state evidence inside ripples by context. This is
  the side step on mOFC's unexplained "anti-location" ripple pattern.

**AMENDMENT to D4, before running (SK, same day).** Son, Azab, … Küchenhoff
et al., "Human hippocampal codes shift under feedback control during rapid
sequence learning" (Google Doc, same dataset):
- During learning, hippocampal cells RAMP across the four ordinal positions (a
  relative code).
- Position-SELECTIVE (absolute) cells appear only after learning.
- Reward-signal cells lead the ramp's progression by ~400 ms on correct
  presses.

Consequences, fixed now:
- (1) D4 state templates are built WITHIN phase: leave-one-configuration-out
  over learning occupancy for learning events, and over execution occupancy for
  execution events. A template dominated by the after-learning code would
  otherwise measure, during learning, a code the hippocampus does not yet have.
  No time-into-traversal detrending (it removes a ramp); a detrended variant is
  secondary.
- (2) Each cell's 4-state template is split into a RAMP part (its projection
  on the linear contrast −1.5, −0.5, +0.5, +1.5) and a SELECTIVE part (the
  rest). Evidence is computed with each part separately.
- (3) New **D4c — when does each region acquire the position code, and do
  ripples get it first?**
  - Region (HC; frontal = mPFC + mOFC; mOFC alone) × phase (learning /
    execution) × code (ramp / selective). Measured in ordinary still moments
    (the stratified control moments) and in ripples.
  - Paper's prediction: HC ramp in learning, selective in execution. SK's
    prediction: the frontal selective code is already present in learning
    (cortex has the position code first).
  - **PRIMARY D4c: HC selective-code evidence in LEARNING ripples minus
    matched learning moments > 0.** This is "ripples carry the absolute code
    before the hippocampus shows it in ordinary activity".
  - Secondary: the region × phase pattern of each code in still moments, the
    same for frontal and mOFC, and the own (flank) reading.
- D4a and D4b otherwise as above, using the within-phase full templates.
  D4a's direction is also given with HC ramp and selective evidence
  separately.

**Weaknesses stated in advance.**
- Timing shows an ordering, not information flow.
- Ripple-by-ripple correlations rest on noisy 0.25–0.5 s windows of ~5 cells.
- Content has never been ripple-specific, so contrasts across contexts and
  against matched moments carry the inference.
- Counts per context × phase will be uneven. Execution has many AWAY ripples;
  learning has fewer UNCOVER ripples. Any cell with < 15 events is dropped.
  - Correction, found when running: AWAY ripples are the RARE set in
    execution. Uncovers come every few seconds there, so the 2 s look-back
    excludes most ripples.
  - A_E had 8 mOFC / 14 lOFC / 19 mPFC sessions in D1 (median 40–68
    ripples); 1 ripple in session 44.
  - The definition was kept as registered; session counts are reported.

### RESULT D1, D2 (same day) — no context flip; lOFC leads at learning-uncover ripples; the uncover response does not predict the ripple

`scripts/exploratory/swr_hfb_direction_context.py` →
`swr/hfb_direction_context_2026-11-15/`:
- `d1_curves.pkl`, `d1_counts.csv`, `d1_session_values.csv`, `d1_result.json`;
- `d2_events.csv`, `d2_betas.csv`, `d2_result.json`;
- `settings.json`, `D1_D2_direction_context.png`.

**D1 direction index** (post − pre ±250 ms; < 0 = cortex leads), own /
matched:

| ROI | U_L | U_E | A_L | A_E | U_L − A_E (primary for mOFC, lOFC) |
|---|---|---|---|---|---|
| mOFC | +0.001 (.96) / +0.001 (.95) | **−0.013 (.010)** / −0.010 (.081) | −0.007 / −0.007 | −0.024 (.16) / −0.044 (.099) | **+0.006, p = .73** (n = 8) |
| lOFC | **−0.011 (.034)** / −0.006 (.27) | −0.002 / −0.002 | −0.010 (.15) / −0.003 | +0.003 / −0.011 | **−0.016, p = .25** (n = 14) |
| mPFC | −0.002 / −0.001 | +0.001 / +0.002 | −0.007 / −0.003 | −0.008 / +0.010 | +0.005, p = .47 |

- No 2 × 2 interaction anywhere (all p ≥ .24).
- Window means: every ROI is ripple-locked in most sets. mPFC matched is
  strongest for A_E (+0.048, p < .001), while its own reading is flat there.
  That is the same "slow elevation" pattern as F4.

**D2** (logistic β of the 0–500 ms uncover HFB on a ripple 0.6–2.5 s later):

| ROI | learning (PRIMARY for mOFC, lOFC) | execution | learning − execution |
|---|---|---|---|
| mOFC | **−0.14, p = .057** (9) / stillness −0.20 (.077) | +0.42 (.25, 4) | −0.44 (.25) |
| lOFC | **−0.05, p = .50** (15) / −0.04 (.72) | +0.03 (.80) | −0.10 (.55) |
| mPFC | +0.06 (.47) / +0.13 (.23) | −0.18 (.028) / −0.26 (.19) | +0.23 (.014) / +0.24 (.31) |

**Reading.**
- Both primaries fail. The direction of cortex–ripple timing does not flip
  between uncover-learning and away-execution ripples in mOFC (p = .73) or
  lOFC (p = .25).
- A larger mOFC or lOFC response to an uncover does not make a following
  ripple more likely. mOFC points the other way (p = .057, 9 sessions).
- Partial support for the model, own reading only:
  - lOFC rises before the ripple at uncover ripples during learning
    (p = .034, not in the matched reading);
  - mOFC rises before the ripple at uncover ripples in EXECUTION (p = .010),
    not learning.
- Few sessions qualify for D2 (enough uncovers with and without a ripple),
  so it is weak.

### RESULT D3, D4 (same day) — the cortex has the position-selective code during learning, the hippocampus only after; the direction primaries fail or reverse

`python scripts/swr_content_explore.py context_content` →
`swr/context_content_2026-11-15/`:
- `d3_per_session.csv`, `d4c_levels.csv`, `d4a_direction.csv`,
  `d4b_binding.csv`, `d4b_conjunction.csv`;
- `result.json`, `D3_rewarded_squares.png`, `D4_state_codes.png`.

51 HC / 39 frontal / 22 mOFC sessions; 31 with HC + frontal.
- The first full run computed every per-session table but crashed in the
  summary step (a single-key groupby in this pandas returns no tuple). Fixed;
  `from_saved(d)` summarises and draws from the saved tables. Nothing was
  recomputed.

**D3 — reward squares in ripples away from uncovers (HC).**
- **PRIMARY away-execution, reward − route, matched: +0.34, p = .23
  (21 sessions).** Own reading −0.10 (p = .72).
- All execution ripples: ripple level +0.26 (p = .093), own +0.42
  (p = .075), matched +0.17 (p = .25).
- Away execution − away learning: +0.47 (p = .27).
- Reward − all non-rewarded, all execution, matched: +0.31 (p = .023), one
  secondary among many.
- Not supported. At most a weak tendency for execution ripples to favour the
  rewarded squares.

**D4c — position in sequence: who has which code, when** (z vs label
permutations, still moments).

| region | ramp, learning / execution | selective, learning / execution |
|---|---|---|
| HC | **+0.91 (p < .001)** / +0.41 (.020) | +0.17 (.25) / **+0.33 (.041)** |
| frontal (mPFC + mOFC) | +0.80 (.001) / +0.52 (.003) | **+0.69 (p < .001)** / +0.33 (.071) |
| mOFC | +0.73 (.012) / +0.51 (.048) | **+0.70 (.005)** / +0.09 (.67) |

- The hippocampus replicates Son, Azab et al. with this estimator: a ramp
  during learning that declines (learning − execution +0.49, p = .030); a
  selective code only in execution.
- The CORTEX already has the selective code during learning. mOFC's
  selective code is strongest in learning and gone in execution.
- Region × phase, selective code, within sessions:
  - (mOFC L−E) − (HC L−E) **+1.34, p = .026** (17 sessions);
  - frontal +0.75, p = .073; mPFC +0.46, p = .36.
- **PRIMARY D4c: HC selective in learning ripples − matched: +0.25,
  p = .073** (own +0.20, p = .22). The ripple level alone is +0.40
  (p = .026), against +0.17 (p = .25) in learning still moments: the selective
  code is visible in learning ripples. The ripple-specific contrast is a
  trend.
- Frontal selective in learning ripples is LOWER than in matched moments
  (−0.42, p = .008).

**D4a — direction** (frontal state pre → HC post vs HC pre → frontal post):
- **PRIMARY U_L − A_E (predicted > 0): −0.131, p = .031, the OPPOSITE sign.**
- Per set:
  - away-execution ripples +0.088 (p = .085; matched +0.116, p = .082),
    i.e. frontal → HC;
  - uncover-learning −0.032 (p = .23);
  - all ripples −0.019 (p = .042).
- HC location in place of HC state: nothing (U_L − A_E −0.074, p = .34).
- The model's direction (cortex → HC at learning uncovers, HC → cortex later)
  is not supported. If anything the frontal → HC lead appears in
  away-execution ripples. Unpredicted and two-sided.

**D4b — the symmetric option.**
- **(ii) PRIMARY HC location × frontal state, same window, all ripples,
  matched: +0.006, p = .51.**
  - away-learning: LESS coupled in ripples than in control moments (matched
    −0.062, p = .008; own −0.086, p < .001);
  - uncover-execution own +0.048 (p = .033).
- (i) HC conjunction: all, matched +0.29 (p = .012). This is only because
  still moments are negative (−0.45, p = .014) while ripples are ≈ 0 (+0.01);
  own reading −0.05 (p = .74). Not a ripple conjunction.
- (iii) HC location × HC state: away-execution own +0.22 (p = .001), matched
  +0.11 (p = .10); elsewhere null.
- No evidence that location and sequence are played together in ripples, in
  either region or across regions.

**Reading.**
- The clearest result is not about ripples: the code timeline. The
  hippocampus ramps during learning and becomes position-selective only after
  (as in the paper). mOFC (and frontal cortex) is already position-selective
  during learning, and the difference is significant for mOFC vs HC.
- This fits "cortex has the sequence code first, the hippocampus acquires it".
  Whether ripples carry it over is not settled:
  - HC's selective code is visible inside learning ripples (+0.40, p = .026),
    but not significantly more than in matched moments (p = .073);
  - the ripple-by-ripple direction does not show cortex → HC at learning
    uncovers.
- Many tests; only the pre-registered primaries carry inference. Of those:
  D3 null, D4a reversed (two-sided p = .031), D4b null, D4c a trend.

## 2026-11-14 (b) — PRE-REGISTRATION: do ripples during learning build the plan? (B1 block dose–response, C1 co-firing reactivation) + F4 summary sets

Written BEFORE running. Both tests avoid the two limits found in F1–F4: the
per-cell lag tuning is unreliable, and there are few discovery events. Neither
classifies single cells, and both use every learning-phase ripple
(620 blocks, median 11 learning ripples each). Session = unit, seed 42.
Learning = grid onset → start of the first fully correct repeat (the agreed
exclusive boundary); execution = the rest.

**B1 — more ripples while learning, stronger plan code once executing?**
`scripts/plan_code_vs_ripples.py`, on the F2 rate-map estimator.
- Per block (grid run) and mPFC cell: the execution maps of all OTHER
  configurations predict this block's FIRST 3 execution loops, pooled. Same
  dwell QC as F2.
- Block code = mean Fisher z over the session's mPFC cells (≥ 2 finite).
  No cell is classified.
- Predictor: log(1 + number of hippocampal ripples in the block's learning
  window).
- Two readings:
  - own: covariates log(learning duration), number of learning loops, block
    order, run index (re-encounter);
  - stillness-matched: log(learning still time), with still time = Σ
    max(0, gap − 0.5 s) over inter-step gaps, in place of the duration, and the
    same other covariates.
- Per session (≥ 8 blocks with a code), OLS on z-scored predictors. The
  ripple β per session is tested across sessions by sign-flip (10,000).
- **PRIMARY: mPFC, 60°, own reading, β > 0.**
- Secondary:
  - lags 0 (specificity: should not, or less), 30, 90 and the full profile;
  - all execution loops instead of the first 3;
  - mOFC; HC (control region);
  - MixedLM on all blocks;
  - negative control: the PREVIOUS block's execution-ripple count as
    predictor.

**C1 — do learning ripples reactivate the co-firing of the block's later execution?**
`mc/analyse/swr_explore/reactivation.py`, the Peyrache et al. (2009)
reactivation strength.
- Each cell's spike counts in 250 ms bins, z-scored with session-wide stats.
- Template C_b = cell × cell correlation of those bins over block b's
  EXECUTION. Bins within 0.5 s of a ripple are dropped. Diagonal zeroed.
- For a moment t in block b's learning window:
  R(t) = z(t)ᵀ C z(t) / (n(n − 1)), with z(t) = counts in [t − 125, t + 125] ms.
- Two templates:
  - own block C_b;
  - other blocks: the mean R over C_b' of blocks with a DIFFERENT
    configuration.
  - specific = own − other. This removes co-firing that is common to all
    configurations.
- Two readings:
  - own: ripple minus its own flanks ([t − 0.75, t − 0.5] and
    [t + 0.5, t + 0.75] s);
  - stillness-matched: ripple minus ripple-free moments (no ripple within
    0.5 s, 50 ms grid) of the same learning window, in the same stratum of
    time since last press × time to next press (`swr_hfb_discovery_lock`
    strata).
- **PRIMARY: mPFC (sessions with ≥ 3 cells), learning ripples, specific
  reactivation (own − other template), stillness-matched reading, > 0.**
- Secondary:
  - general reactivation (own template alone);
  - own reading;
  - mPFC + mOFC pooled; mOFC; HC (expected positive control);
  - post-FD-pause ripples only; execution ripples.

**F4 summary sets (descriptive, for two figures).** `swr_hfb_discovery_lock`
gains:
- R_ALL (every ripple; matched controls drawn from all task time);
- U_COR (every correct uncover), U_ERR (every incorrect uncover);
- U_FT_ERR (incorrect uncovers in the first traversal).
Same two readings and tests as F4.

### RESULT F4 summary sets (same day) — mPFC: every ripple, no uncover; mOFC: ripples and errors

`swr/hfb_discovery_lock_2026-11-14/` (rerun, superset of F4; F4 numbers
unchanged). Figures
`swr/gradient_claims_followup_2026-11-14/F5_mPFC_ripples_not_uncovers.png`
and `F6_mOFC_uncovers_and_ripples.png`
(`scripts/exploratory/plot_ripple_vs_uncover_locking.py`).

| set | reading | mOFC (18) | mPFC (35) |
|---|---|---|---|
| every in-task ripple | own | +0.0174, p < .001 (−260…+90) | +0.0065, p < .001 (−140…+210) |
| every in-task ripple | matched | +0.0221, p < .001 | +0.0202, p < .001 (−120…+180) |
| every correct uncover | own / matched | +0.003 (.18) / +0.008 (.57) | +0.001 (.44) / 0.000 (.998) |
| every incorrect uncover | own / matched | **+0.011 (.038)** / +0.018 (.059) | 0.000 (.98) / +0.008 (.29) |
| incorrect, first traversal | own / matched | +0.010 (.086) / **+0.026 (.009)**, cluster +310…+410 | −0.002 (.33) / +0.012 (.13) |

**Reading.**
- mPFC HFB is locked to hippocampal ripples in both readings and in the GLM.
  It shows no positive response to any uncover set.
- mOFC is locked to ripples just as strongly. It also responds to incorrect
  uncovers, most in the first traversal, and in the GLM to first
  discoveries.
- mOFC − mPFC differs in the uncover response (all errors own p = .018; FT
  errors own p = .020), not in ripple locking (matched p = .94).

### RESULT B1 (same day) — a trend in the predicted direction, not robust

`scripts/plan_code_vs_ripples.py` → `swr/plan_code_vs_ripples_2026-11-14/`
(`blocks.csv`, `cell_block_codes.csv`, `session_betas.csv`, `result.json`,
`B1_plan_code_vs_ripples.png`). 1458 blocks with an execution phase, median
11 learning ripples (58 blocks > 80). mPFC: 25 sessions, 617 blocks.

| mPFC, 60°, first 3 execution loops | mean ripple β (standardised) | p |
|---|---:|---:|
| **PRIMARY, own reading** | **+0.168** | **.070** |
| stillness-matched reading | +0.069 | .40 |
| all execution loops, own / matched | +0.065 / +0.029 | .50 / .73 |
| MixedLM, all blocks, session-centred | +0.013 (raw units) | .72 |
| negative control: previous block's execution ripples | −0.051 | .36 |

- Specificity: mPFC lag 0 +0.034 (p = .73); stack 30–90° +0.109 (.27);
  mOFC 60° +0.072 (.52); HC lag 0 +0.004 (.97).
- Exploratory lag profile (12 lags × 2 readings × 2 loop sets, uncorrected):
  - 180° +0.31 (p = .005 own, .006 matched, first 3 loops; not with all
    loops); 120° matched +0.19 (p = .041).
  - Ripple β is ≥ 0 at 10 of 12 lags: a weak general tendency, not a
    60°-specific one.

**Reading.** Blocks with more learning ripples have, at best, a slightly
stronger mPFC future code at the start of execution. The primary is p = .070.
It does not survive the stillness covariate or the pooled model.
- 180° is the strongest lag but was not predicted. Bonferroni over 12 lags
  gives p ≈ .06.
- Not evidence for ripples building the plan; also not strong evidence
  against. Each block's code rests on ~5 cells × 3 loops.

### RESULT C1 (same day) — null everywhere, and the HC positive control fails too

`python scripts/swr_content_explore.py reactivation` →
`swr/reactivation_2026-11-14/` (`per_session.csv`, `result.json`,
`C1_reactivation.png`). 21 mPFC / 18 mOFC / 33 mPFC + mOFC / 48 HC sessions
with ≥ 3 cells.

| | own flanks | stillness-matched |
|---|---|---|
| **PRIMARY mPFC, learning ripples, specific (own − other)** | −0.0002, p = .74 | **+0.0006, p = .52** |
| mPFC, general (all execution co-firing) | −0.0003, p = .77 | −0.0001, p = .95 |
| mPFC + mOFC, specific | −0.0001, p = .90 | +0.0004, p = .55 |
| mOFC, specific | −0.0005, p = .45 | −0.0000, p = .97 |
| HC, general (expected positive control) | +0.0004, p = .38 | −0.0004, p = .25 |
| mPFC, post-FD ripples, specific | −0.0004, p = .79 (21) | −0.0034, p = .094 (11) |

- The post-FD matched reading is thin: few ripple-free still moments in those
  short pauses (10–18 sessions).

**Deviation.** Execution ripples were dropped from the pre-registered
secondaries. The template is built from the same execution window, so the
comparison would be circular.

**Reading.** No reactivation of execution co-firing in learning ripples,
specific or general, in any region or reading. Because HC shows none either,
this is a sensitivity failure, not evidence against.
- With ~5 cells, 250 ms windows and human single units that fire only ~4%
  more in ripples, pairwise co-firing in single ripple windows carries too
  little signal.
- Do not cite C1 as a negative result.

### EXPLORATORY (written before running, AFTER seeing B1) — B1b: future lags collapsed, and which ripples

SK, after B1 panel d: β looks positive at nearly every future lag except
270–330°. Two questions:
1. A crude test of the future lags on average.
2. Is the B1 relation specific to certain ripples (e.g. after correct
   uncovers), or does it change with ripples across the whole task?

mPFC only; own reading only (no still-time covariate, at SK's request).
`scripts/plan_code_vs_ripples_types.py`, reusing B1's saved per-cell block
codes (first 3 execution loops). No code is recomputed.

**Collapsed future measures**, fixed now:
- **F_all** = per block, the mean Fisher z over lags 30–330°: all 11 future
  lags, the manuscript's existing "beyond-now" set. Data-independent, and the
  primary for this step.
- F_near = 30–90° (the manuscript's informed subset).
- F_30-240 = the range read off panel d. POST HOC and circular; reported only
  to answer the question as asked.
- Also the per-session mean of the 11 per-lag ripple βs.

The per-lag βs share blocks and cells, so they are not independent tests.
Only the collapsed versions (one value per session) are tested.

**Ripple subsets**, counted per block. Predictor log(1 + n); covariates as B1
(log duration of the window the ripples come from, n learning loops, block
order, run).
- L_all: all learning ripples (= B1).
- L_FT: first traversal only. L_rest: the rest of learning.
- L_postCorrect, L_postFD, L_postError: in the pause after a correct / first
  discovery / incorrect uncover, (press, press + min(pause, 2 s)].
- L_other: learning ripples not in any post-uncover pause.
- E_first3: during the first 3 execution loops (the loops the code is measured
  on).
- E_all: the whole execution. B_all: the whole block.

Learning subsets use log(learning duration) as their covariate, E_first3 /
E_all / B_all the duration of their own window.

**Tests.** Sign-flip over sessions (10,000) of per-session ripple βs, for
outcomes 60°, F_all, F_near, F_30-240 and every lag. BH-FDR over the family
10 subsets × {60°, F_all}. Everything here is exploratory.

### RESULT B1b (same day) — the collapsed future code tracks learning ripples; not tied to post-reward ripples

`scripts/plan_code_vs_ripples_types.py` → `swr/plan_code_vs_ripples_types_2026-11-14/`
(`blocks_with_codes.csv`, `session_betas.csv`, `result.json`,
`B1b_future_and_ripple_types.png`). 617 blocks, 26 mPFC sessions (25 with
≥ 8 blocks). Learning-ripple counts are identical to B1's for every block.

**Future lags collapsed, learning ripples:**

| outcome | mean ripple β | p | q (family) |
|---|---:|---:|---:|
| **F_all, code averaged over 30–330°** | **+0.310** | **.006** | .12 |
| mean of the 11 per-lag βs | +0.098 | .009 | — |
| F_near 30–90° | +0.148 | .12 | — |
| F_30-240 (post hoc, circular) | +0.318 | .013 | — |
| 60° (= B1 primary) | +0.168 | .070 | .28 |

**By ripple type** (60° / F_all):

| subset | median per block | 60° | F_all |
|---|---:|---|---|
| all learning (= B1) | 10 | +0.17, p = .070 | **+0.31, p = .006** |
| first traversal | 9 | +0.07, .22 | **+0.17, .015** |
| rest of learning | 0 (19% of blocks > 0) | +0.08, .67 | +0.21, .34 |
| after correct uncovers | 2 | +0.11, .058 | +0.07, .16 |
| after first discoveries | — | +0.08, .12 | +0.04, .35 |
| after incorrect uncovers | — | +0.01, .90 | +0.01, .86 |
| not after an uncover | 5 | +0.04, .67 | **+0.20, .031** |
| execution, first 3 loops | 7 | −0.06, .25 | +0.03, .61 |
| whole execution | 22 | −0.05, .38 | −0.02, .83 |
| whole block | 33 | +0.01, .90 | +0.13, .21 |

Nothing survives BH over the 20-test family (smallest q = .12, F_all,
learning ripples).

**Reading (exploratory).**
- Averaged over the 11 future lags, the mPFC code at the start of execution
  is stronger in blocks with more learning ripples (p = .006).
- It is a learning-phase relation. Execution ripples of the same block do not
  predict (p = .83), and counting ripples over the whole block dilutes it
  (p = .21).
- It is carried by first-traversal ripples and by ripples NOT following an
  uncover. Post-reward ripples are sparse (median 2 per block) and give at
  most a 60° trend (p = .058). The relation is therefore not specific to
  "loading after a reward".
- Averaging lags makes the code more reliable, which is why F_all β > the
  60° β. It also means F_all is "overall lag-shifted spatial coding",
  including near-past lags 270–330°. Future-specificity against lag 0 cannot
  be tested fairly: a single lag is noisier than an 11-lag mean.

**Not yet checked:**
- the still-time covariate (it removed the 60° effect in B1: p = .40);
- the previous-block negative control for F_all.

**ALL execution loops (SK, same day).**
`scripts/plan_code_vs_ripples_learning_figure.py` →
`swr/plan_code_vs_ripples_learning_2026-11-14/` (`session_betas.csv`,
`result.json`, `learning_ripples_plan_code.png`). Learning ripples, own
reading, mPFC, 25 sessions, 617 blocks. Per-lag βs are identical to B1's
"all" rows (same code and data).
- F_all (30–330° averaged): **+0.19 ± 0.12, p = .125**.
- Per lag: 30° +0.23 (p = .059), everything else p ≥ .13.
- The F_all relation is therefore significant only for the FIRST 3 execution
  loops (+0.31, p = .006) and fades over the whole execution.
  - Under the hypothesis this is expected (later loops refine the plan
    independently of the learning ripples).
  - It also fits a chance first-3 result. It cannot be told apart here.
- Same figure, now with both loop sets (`learning_ripples_plan_code.png`;
  `session_betas_{first3,all}.csv`): F_all first 3 +0.31 (p = .006), all
  +0.19 (p = .125). **Paired first 3 − all: +0.12, p = .38.** The two do not
  differ significantly.

**The mPFC lag profile on the first 3 vs last 3 execution loops (F2
estimator).** `scripts/per_lag_encoding_first_loops.py` →
`per_lag_encoding_by_phase/first_loops_2026-11-14_20-10-46/` (`per_cell.csv`,
`per_session.csv`, `result.json`, `lag_profile_first_vs_last_loops.png`).
155 mPFC cells, 32 sessions; all-loop and discovery profiles reproduce F2
exactly.
- With only each block's first 3 or last 3 execution loops, the 60° peak of
  the full execution profile (+0.081, p < .05) is gone in BOTH (+0.018 /
  +0.021). Three loops per block are too few for this estimator; the profile
  is flat and noisy.
- first 3 − last 3 differs only at 330° (+0.087, p = .047) and 300° (+0.089,
  p = .061), uncorrected over 12 lags. At 330° the first 3 loops resemble
  discovery (+0.067* vs +0.057*).
- B1's own outcome (training on ALL other configurations' execution, testing
  on this block), averaged over blocks: first 3 loops ≤ +0.026 at every lag
  (only 240° p < .05); all loops 60° +0.038 (n.s.).
  - The block-level code B1 regresses on is therefore close to the noise
    floor on average. B1 asks whether ripples explain block-to-block variation
    in a very weak signal.

## 2026-11-14 — PRE-REGISTRATION: lag-0 cells at discovery, tuning across phases, mOFC vs mPFC HFB at discovery (F1–F4)

Written BEFORE running. Follow-ups SK asked for on claims 1 and 2
(`swr/gradient_claims_2026-10-30/`). Shared: `bundle_v2`, session = unit, seed
42. Two readings everywhere: **own** (each event against its own baseline) and
**stillness-matched**. lag-0 tuning = `r_lag000_noctrl` (`per_cell_ALL_ROIs`,
2026-08-04 relabelled-final). It is computed on `all_minus_explore` repeats only
(from the first fully correct repeat on), so it never saw a discovery loop.
"lag-0 cells" = top 30% of `r_lag000_noctrl` within their ROI. That is a display
split only; every test uses the continuous value.

**What already existed (checked first).**
- Claim 2 a/b (`consolidation_2026-10-01`) are arrival-locked over all cells in
  the ROI, with no cell filter.
- `lag_group_ripple_2026-10-01` has per-cell ripple modulation (inside vs
  outside ripples, encode vs execute) for every cell, but tags lag-0 by argmax
  (9 mPFC cells).
- `discovery_cells_2026-10-30` has per-cell FD − RC template contributions.
- No per-lag tuning has ever been computed on discovery loops.
- `hfb_phases_50shifts_2026-10-08` has press-locked correct-uncover sets for
  mOFC/mPFC (learning, execution), but no first-traversal-only or FD-only set,
  and no ripple set restricted to the post-FD pause.
- The HFB GLM (`swr_hfb_glm.py`, 2026-09-29) is MedialFrontal pooled, with no
  phase split.

**F1 — claim 2 a/b in lag-0 cells (display).** The `consolidation` estimator
(evidence for the square just uncovered, 0.05–1.0 s, leave-one-configuration-out
templates, z vs 1000 square-label permutations).
- Run (i) exactly as published (arrival-locked) and (ii) reward-locked, as N2.
- Cells: mPFC, mOFC, HC (each top 30%), side by side with all cells.
- Events: first discovery (FD), repeat correct (RC), incorrect. FD is split by
  a ripple in the post-reward pause (0, min(pause, 2 s)].
- Readings for (ii): own = post − pre-press (−0.5…−0.05 s), same events.
  Matched = FD vs RC, and FD with vs FD without a ripple, paired 1:1 on the
  following stillness.
- Sign-flip over sessions. Descriptive; no primary.

**F2 — do cells change their tuning between discovery and execution?**
`scripts/per_lag_encoding_by_phase.py`, with the `per_lag_encoding` rate-map
estimator.
- Loops: discovery = repeats before the first fully correct one (first traversal
  included); execution = the rest (the published set).
- Per configuration and phase, rate maps are POOLED over repeats rather than
  averaged traces. Discovery paths differ between repeats, so the per-bin mode
  location would blur them. Execution uses the same pooling, and its r is
  checked against the published `r_lag000_noctrl`.
- QC as published: dwell-weighted r, MIN_DWELL_BINS 25, ≥ 3 shared locations,
  phase residualised within each phase subset.
- Per cell, per lag (all 12):
  - r_disc: LOCO within discovery.
  - r_exe: LOCO within execution.
  - r_x: execution maps of the training configurations predict the held-out
    configuration's DISCOVERY map.
  - r_x_n: the same training maps predict k EXECUTION repeats of the held-out
    configuration, with k = its number of discovery repeats (10 random draws,
    seed 42).
  - Change index = r_x_n − r_x. Same training map and the same amount of
    held-out data, so it is > 0 only if discovery firing departs from the
    execution tuning.
- PRIMARY (mPFC, lag 0, session = mean Fisher z over cells): (a) r_disc > 0;
  (b) change index ≠ 0.
- Secondary: mOFC and HC; the full lag profile in each phase; change vs lag-0
  strength across cells.

**F3 — are lag-0 cells the ones the plan is loaded into?** mPFC + mOFC cells,
continuous lag-0.
- **F3a, ripple-locked firing.** Per cell, the rate around ripples (50 ms bins,
  −1…+1 s) / the cell's mean rate. Own = minus the ripple's own flanks
  (0.5–1.0 s both sides). Matched = minus still moments with no ripple within
  0.5 s, in the same stratum (time since last press × time to next press, the
  `future_shift` bin edges) and the same period.
  - Sets: first-traversal ripples (PRIMARY); ripples in the post-FD pause;
    execution ripples; first traversal − execution.
  - Test: across cells, Spearman of lag-0 with the response per bin, cluster
    mass over −0.5…+0.5 s; null = lag-0 permuted within session (10,000). This
    is `discovery_cells.cluster_corr`, the same function as N1. Also the ±150 ms
    window as one value.
- **F3b, the discovered square.** Per cell, its term of the evidence for the
  square just discovered (as N2 `_cell_contrib`). Own = post − pre-press. Matched
  = FD − stillness-matched RC. Also FD with − without a post-reward ripple
  (matched). Spearman with lag-0, within-session permutation.
  - ⚠ The level of evidence at FD is expected to track lag-0 (both are place
    tuning); only the two contrasts are interpretable.

**F4 — mOFC vs mPFC HFB at discovery: ripple-locked or uncover-locked?** Partition
ROIs as in claim 1a (2026-09-28 partition), 20 shifted nulls.
`scripts/exploratory/swr_hfb_discovery_lock.py`.
- Sets:
  - Ripple-locked: first-traversal ripples (R_FT, = claim 1a); ripples in the
    post-FD pause (R_FD).
  - Press-locked: all first-traversal uncovers (U_FT), FD (U_FD); FD with /
    without a post-reward ripple.
- Own reading: real − shifted null; ripples vs flank 750–1000 ms, events vs
  −1.5…−1.0 s.
- Matched reading: uncovers − arrow presses with the same following stillness
  (`fin.matched_sets`); ripples − stratum-matched ripple-free still moments from
  the same period (FT still time for R_FT; post-FD pauses for R_FD).
- Per ROI and set: cluster sign-flip over sessions, and window means (ripple
  ±250 ms; event 0–0.5 s). mOFC − mPFC unpaired (session-label permutation),
  and paired in sessions with both.
- **GLM, the disambiguation.** `swr_hfb_glm.simple`'s differenced single-beta
  model (kernel −250…+500 ms minus a −1500…−750 ms arm) with the events split:
  ripples post-FD / other first-traversal / later; FD / FT error / later correct
  / later error; navigation. All compete for the same continuous HFB, per
  derivation, mean per session and ROI.
- PRIMARY: (Q1) R_FD ±250 ms, mOFC − mPFC, own reading. (Q2) GLM β(FD) −
  β(post-FD ripple) per ROI: > 0 = uncover-locked, < 0 = ripple-locked.
- Uncorrected across the two; everything else secondary.

### RESULT F2 (same day) — mPFC's 60° code is not there in discovery loops; a cell's lag-0 r is not a reliable property

`scripts/per_lag_encoding_by_phase.py` → `ephys_humans/derivatives/group/
per_lag_encoding_by_phase/2026-11-14_18-29-12/` (`per_cell.csv`,
`per_session.csv`, `result.json`, `config.json`, `F2_tuning_by_phase.png`).
924 cells. Sanity: the pooled-map execution r vs the published
`r_lag000_noctrl`: ρ = .75 (mPFC), .77 (mOFC, HC).

**⚠ Seed bug, found and fixed before reporting.** The per-cell seed was
`SEED + hash(neuron)`, copied from `per_lag_encoding.py`. Python salts string
hashes per process, so two runs gave different subsample draws: the mPFC
change at 60° was +0.051 in one run and +0.039 (p = .12) in another. Now
`zlib.crc32`, 50 draws, and two runs agree to every digit. Superseded folders
deleted.
- **`per_lag_encoding.py` has the same line**
  (`seed + (hash(nid) & 0x7FFFFFFF)`), so its per-cell permutation p values
  are not bit-reproducible across runs. The CV r values are unaffected (no
  randomness). Not changed.

| mPFC, lag 0 (32 sessions) | mean (Fisher z) | p (sign-flip) |
|---|---:|---:|
| r_disc (PRIMARY a) | +0.055 | .097 (cell level p = .028) |
| r_exe | +0.045 | .058 |
| r_x (execution → discovery) | +0.019 | .28 |
| r_x_n (execution → same n execution repeats) | +0.025 | .029 |
| change = r_x_n − r_x (PRIMARY b) | −0.005 | .78 |

- **Lag profiles.** mPFC execution peaks at 60° (+0.081, p < .05), as in the
  manuscript. In discovery loops it is flat from 30° to 300°, with lag 0
  +0.055 and 330° +0.057 (p < .05).
- **Change index by lag (secondary, uncorrected over 12).** mPFC peaks at 60°:
  +0.047, p = .040. Discovery firing departs from the execution tuning at 60°,
  not at 0°. mOFC and HC show no change anywhere (HC lag 0 −0.013, p = .27).
- **HC = positive control.** Lag-0 tuning in both phases: exe +0.044
  (p = .018), disc +0.053 (p = .002), exe → disc +0.031 (p = .023).
- **Same cells, both phases (lag 0, independent repeats).** ρ = +0.02 (mPFC,
  p = .75), −0.09 (mOFC, .24), +0.007 (HC, .88).
- **⚠ Reliability.** The same cell's execution lag-0 r from odd vs even
  repeats correlates ρ = −0.09 (mPFC, p = .25), +0.08 (mOFC), +0.005 (HC,
  p = .91); only EC +0.37 (p = .035). At 60°, mPFC +0.19 (p = .12). The
  population mean is real, but a single cell's lag-0 r is mostly noise. A
  top-30% "lag-0 cell" set is therefore close to a random draw. This caps
  every analysis that sorts cells by lag-0 tuning (N1, F1, F3, and
  `lag_group_ripple`'s argmax tag).

**Reading.** In mPFC the lag-0 code, weak as it is, looks the same in
discovery and execution (change ≈ 0). The 60° code, mPFC's signature, is
absent in discovery loops and departs from the execution tuning there
(p = .040, uncorrected). That is what a plan built after discovery predicts.
Caveat: in discovery loops the path ahead is exploration, so the absence of a
future code is also expected without any loading. It does not by itself say
that ripples build it.

### RESULT F4 (same day) — mOFC follows the uncovers (and ripples), mPFC only ripples; ripples right after a first discovery drive neither

`scripts/exploratory/swr_hfb_discovery_lock.py` → `swr/hfb_discovery_lock_2026-11-14/`
(`curves.pkl`, `window_values.csv`, `counts.csv`, `result_epochs.json`,
`glm_betas.csv`, `glm_session_betas.csv`, `result_glm.json`, settings,
`F4_epochs.png`, `F4_glm.png`). 18 mOFC / 35 mPFC sessions, 11 with both.

**Epochs, window means** (ripples ±250 ms, presses 0–500 ms):

| set | reading | mOFC | mPFC | mOFC − mPFC (unpaired) |
|---|---|---|---|---|
| ripples, first traversal | own | +0.0175, p = .004 (cluster −220…+70, p = .001) | +0.0034, p = .14 | +0.014, **p = .006** |
| ripples, first traversal | matched | +0.0174, p = .030 | +0.0147, p < .001 (cluster −170…+250, p < .001) | +0.003, p = .79 |
| ripples after a FD (Q1) | own | +0.0197, p = .16 | −0.0001, p = .99 | +0.020, **p = .11** |
| ripples after a FD | matched | +0.0166, p = .11 | +0.0176, p = .31 | −0.001, p = .97 |
| uncovers, first traversal | own | +0.0078, p = .077 | −0.0020, p = .29 | +0.010, p = .016 |
| uncovers, first traversal | matched | +0.0243, p = .040 | +0.0074, p = .087 | +0.017, p = .10 |
| first discoveries | own / matched | +0.001 / −0.005, n.s. | −0.001 / −0.011, n.s. | n.s. |
| FD, no ripple / ripple follows | both | all n.s. | all n.s. | n.s. |

**GLM** (β per event, window-summed z; sign-flip over sessions):

| regressor | mOFC own | mOFC +still | mPFC own | mPFC +still |
|---|---|---|---|---|
| ripple after a FD | +0.29, p = .32 | +0.27, .35 | −0.15, .30 | −0.14, .31 |
| other first-traversal ripple | +0.36, **.029** | +0.38, **.022** | +0.17, **.014** | +0.17, **.012** |
| later ripple | +0.45, **< .001** | +0.45, **< .001** | +0.27, **< .001** | +0.27, **< .001** |
| first discovery | +0.48, .074 | +0.63, **.026** | +0.03, .74 | +0.12, .20 |
| first-traversal error | +0.73, **.006** | +0.91, **.001** | −0.23, **.049** | −0.12, .26 |
| later correct / error / navigation | n.s. | n.s. | n.s. (error +0.34, .09) | n.s. |
| **FD − ripple after it (Q2)** | **+0.19, p = .61** | +0.36, .37 | **+0.18, p = .30** | +0.27, .12 |

- mOFC − mPFC: first-traversal error +0.95 (p < .001; +still +1.03,
  p < .001); FD +0.45 (p = .060; +still +0.51, p = .028); later ripples +0.18
  (p = .040).
- mPFC: ripples after a FD − other first-traversal ripples −0.31 (p = .055;
  +still −0.32, p = .050).

**Reading.**
- Neither primary is significant. Q1 (post-FD ripples, mOFC − mPFC) +0.020,
  p = .11. Q2 (FD − post-FD ripple) +0.19 / +0.18, p = .61 / .30.
- **mOFC is locked to both.** It rises before first-traversal ripples (both
  readings; replicates claim 1a). With every event type competing in one GLM,
  it also responds to first-traversal uncovers, most to errors (p ≤ .006) and
  to first discoveries in the stillness model (p = .026). So its uncover
  response is about uncovering in search, not specifically reward.
- **mPFC is locked to ripples only.** Its ripple betas are positive in both
  models (p ≤ .014), and it has no uncover response (FD β ≈ 0; FT error
  negative).
- In the stillness-matched epochs mPFC is clearly ripple-locked in the first
  traversal (p < .001). In the own reading it is not (p = .14): its ripple
  rise sits on a slower elevation that the own flanks absorb.
- **The ripples right after a first discovery drive neither region**
  (epochs and GLM n.s.). In mPFC they are, if anything, weaker than other
  first-traversal ripples (p = .05). That is not what "the plan is loaded
  into mPFC by the post-discovery ripples" predicts. Caveat: these are
  10–60 ripples per session.
- The mOFC − mPFC difference is in the uncover response, not in the ripple
  locking (ripple betas same sign, overlapping).
- Uncorrected across sets.

### RESULT F1 + F3 (same day) — lag-0 cells are neither ripple-locked nor the carriers of the discovered square

`python scripts/swr_content_explore.py lag0_cells` →
`swr/lag0_cells_2026-11-14/`:
- `F1_arrival_locked_top30/` (the `consolidation` output);
- `F1_reward_locked_per_session.csv`, `F1_reward_locked_summary.csv`;
- `F3a_cells.pkl`, `F3b_per_cell.csv`, `result.json`;
- `F1_claim2ab_lag0_cells.png`, `F3_lag0_cells_ripples.png`.

Top 30% thresholds: mPFC ≥ 0.13, mOFC ≥ 0.13, HC ≥ 0.16; 248 cells. With
≥ 2 such cells per session, 14 mPFC / 9 mOFC / 34 HC sessions enter.
`consolidation.main` gained `keep` (cell filter) and `out_dir`; its default
behaviour is unchanged.

**Already derivable, checked first.**
- `lag_group_ripple_2026-10-01` per-cell ripple modulation against
  CONTINUOUS lag-0 (Spearman, within-session permutation):
  - mPFC + mOFC: encode ρ = +0.02 (p = .73), execute +0.07 (.28).
  - mPFC: encode +0.12 (p = .12), enc − exe +0.12 (.17).
  - The argmax-tag interaction (p = .004, 9 cells / 8 sessions) does not
    carry over to the continuous measure.

**F1, mPFC evidence for the square just uncovered:**

| | all cells (26 sessions) | top-30% lag-0 (14) |
|---|---|---|
| arrival-locked FD | +0.77, p < .001 (published) | +0.56, p = .084 |
| arrival-locked FD, ripple − none | −0.51, p = .076 (published) | **−0.77, p = .007** |
| reward-locked FD, raw | +0.67, p = .001 | +0.29, p = .20 |
| reward-locked FD − RC, matched | +0.15, p = .50 | −0.26, p = .40 |
| reward-locked FD own (post − pre) | +0.19, p = .40 | −0.34, p = .13 |
| reward-locked ripple − none, own / matched | +0.24 (.24) / +0.38 (.13) | +0.28 (.42) / +0.31 (.34) |

- mOFC top-30%: only RC raw +0.74 (p = .021, 9 sessions).
- HC top-30% behaves like all HC cells: FD +0.69 (p = .001); FD − RC matched
  +0.39 (p = .022; all cells +0.39, p = .037).
- The arrival-locked ripple − none of −0.77 is NOT discovery-specific (RC
  −0.79, p = .017). It flips sign once locked to the press and to the
  post-reward pause. It is not a finding.

**F3a, ripple-locked firing**, mPFC + mOFC, 289 cells, 45 sessions:
- Cells are not ripple-modulated on average at ±150 ms: own −0.003 (p = .86),
  matched +0.003 (p = .90), first traversal. Same for post-FD and execution
  ripples.
- lag-0 predicts no response in any set or reading (all ±150 ms |ρ| ≤ 0.08,
  p ≥ .28). No cluster p < .05 (best .075).
- The only secondary hit is mPFC first traversal − execution, own: ρ = +0.19,
  p = .020. It is not confirmed matched (+0.10, p = .31).

**F3b, per-cell evidence for the discovered square vs lag-0:**
- FD own ρ = +0.05 (p = .39); FD − RC matched −0.05 (.47); ripple − none
  matched −0.003 (.97).
- mOFC alone, FD own +0.14 (p = .091).

**Reading.**
- The cells that hold the current square during execution are not the
  ripple-locked cells, and not the cells that carry the newly found square.
  That holds at discovery, after it, or in ripples.
- The top-30% lag-0 subset carries the new square no better than all cells
  (it is weaker, fewer cells).
- Interpret with F2: a cell's lag-0 r does not replicate across odd/even
  execution repeats. "Lag-0 cell" is therefore not a stable cell category in
  this data, and none of these nulls can say that no such cells exist.
- What is real is at the population level. mPFC's raw new-square evidence
  (+0.67, all cells) still vanishes when stillness-matched.

## 2026-10-30 — PRE-REGISTRATION: the medial-frontal gradient — ripples, phases and 'now' cells (G1–G3, N1–N2)

Written BEFORE running. Scripts `scripts/exploratory/swr_hfb_gradient.py`
(G1–G3) and `mc/analyse/swr_explore/discovery_cells.py` (N1–N2). Output
`swr/gradient_hfb_2026-10-30/`, `swr/discovery_cells_2026-10-30/`. All existing
evidence is collected in `swr/gradient_hypothesis_overview_2026-10-30/` (a
PDF, one page per claim; `scripts/exploratory/plot_gradient_hypothesis_overview.py`).

**SK's hypothesis.** Medial PFC is ordered ventral = now / immediate future to
dorsal = distant future. During encoding, HC replays the 'now' in ripples to
the ventral end, perhaps as far as mOFC. During execution the more dorsal part
plays the immediate future, independent of ripples. lOFC codes reward events,
before or independent of ripples.

**Shared.** `bundle_v2`, session = unit, seed 42. HFB = real − mean of 20
shifted nulls (5–120 s, either sign), contacts on a different shaft from HC.
Group tests are sign-flips over sessions, with the same function for observed
and null. Phases: first traversal = grid onset → first t_D; rest of encoding =
first t_D → start of the first fully correct repeat; execution = from that
start (the agreed exclusive boundary). Anatomical axis = MNI z (data-independent,
2026-09-06); |x| and y are secondary.

**G1 — ripple-locked HFB along the medial axis × phase.** Contacts: roi_family
mPFC|mOFC, |x| ≤ 20 mm (179 derivations, 53 sessions, z −18…+26). Per contact
and phase: peri (±250 ms) − flank (250–750 ms); ≥ 20 ripples. PRIMARY: in
sessions with ≥ 2 contacts spanning ≥ 5 mm in z, the OLS slope on z of
[first traversal − execution]; mean slope vs 0. SK predicts < 0 (ventral
contacts more ripple-locked in the first traversal, relative to execution).
Secondary: the slope in each phase; rest of encoding; a mixed model on all
contacts (value ~ z × phase + (1|session)); direction index (post − pre) on z
per phase; |x| and y.

**G2 — where is the reward-event signal?** All frontal contacts (G1's medial
set + partition lOFC, FrontalPole, vlPFC_IFG). Uncover-locked HFB, baseline
−1.5…−1.0 s, window 0–0.5 s; ≥ 10 events. PRIMARY: correct first-traversal
uncovers, response on z, within-session slope as in G1. SK predicts < 0 (more
ventral → larger reward response). Secondary: |x|; valence (correct − error,
first traversal); execution; the mixed model.

**G3 — lOFC: reward event or ripple?** lOFC partition contacts.
- G3a PRIMARY: ripple-locked lOFC HFB in the pre-ripple window [−250, 0) for
  QUIET ripples (no press in the preceding 2 s). If the lOFC lead is the event
  response it vanishes (≈ 0); if it is ripple-bound it stays > 0. Reported
  alongside the same measure for ripples that follow a press. Secondary:
  direction index for both.
- G3b: lOFC uncover response (0–0.5 s, correct first-traversal uncovers) for
  uncovers followed by a ripple 0.5–2.0 s later vs uncovers with no ripple
  within 2 s. 'Independent of ripples' predicts no difference (reported with
  its CI).

**N1 — 'now' cells at the moment of discovery, over time.** Cells: mPFC + mOFC
(`per_cell_ALL_ROIs`, cell_idx − 1 → cell). Events are locked to the REWARD
PRESS (bundle uncover table). This fixes `storage_vs_planning`'s lock to
arrival, and its ripple split of ±0.5 s around arrival, which missed the
post-reward ripples. First discovery (FD) = correct uncovers in the first
traversal. Repeat correct (RC) = correct uncovers on later repeats of the same
block: same square and action, reward already known. Per cell: rate in 100 ms
bins, −1.5…+2.0 s, minus the cell's −1.5…−1.0 s baseline, divided by its
task-wide mean rate; then FD − RC. Moderator lag0 = `r_lag000_noctrl`.
PRIMARY: across cells, Spearman of lag0 with FD − RC per bin, cluster mass
over 0–2 s (cluster-forming |t| at two-sided 0.05). Null = 10,000
within-session permutations of lag0; one function computes observed and null.
Prediction > 0. Secondary: mPFC and mOFC separately; FD alone; FD with vs
without a ripple in the post-reward pause (0, min(pause, 2 s)]; z as moderator
at the site level (cells averaged per recording coordinate; null = permuted
site z).

**N2 — the newly found square, by anatomy.** The `consolidation` estimator,
re-run locked to the reward press and time-resolved (250 ms windows, 125 ms
steps, −0.5…+2.0 s), for FD vs RC in mPFC, mOFC and HC. Plus per-cell
contributions to FD − RC in 0.05–1.0 s, Spearman with z at the site level
(null = permuted site z). Prediction: ventral sites carry more.

**C4 — cell ripple modulation by depth** (added before running; SK asked for
it in the same session). Per cell, the ripple modulation index already computed
in `lag_group_ripple_2026-10-01` ((in − out)/(in + out), encode and execute).
mPFC + mOFC cells; site = cells sharing a session and MNI coordinate; values
averaged per site. PRIMARY: Spearman across sites of [encode − execute] with
MNI z; null = 10,000 permutations of site z. SK predicts < 0 (ventral sites
more ripple-modulated during encoding, relative to execution). Secondary:
encode and execute separately; mPFC sites alone; mOFC vs mPFC sites (label
permutation). Script `scripts/exploratory/swr_cell_ripple_depth.py`.

### RESULT G1–G3, C4 (same day)

`swr/gradient_hfb_2026-10-30/` (`contact_values.csv`, `result.json`,
`gradient_hfb.png`) and `swr/cell_ripple_depth_2026-10-30/`. 91 medial contacts
with ripple values in 53 sessions; 36 sessions give a G1 slope, 41 a G2 slope;
lOFC 32 contacts in 21 sessions. Slopes are in HFB z per mm of MNI z
(+ = dorsal).

**G1 — ripple-locked HFB along the medial axis.**
- **PRIMARY, (first traversal − execution) slope on z: +0.00016 ± 0.00016,
  p = .33 (36).** No phase-specific gradient.
- But a gradient in EVERY phase: ventral contacts are more ripple-locked.
  Execution −0.00037/mm, p = .001; first traversal −0.00021, p = .10; rest of
  encoding −0.00067, p = .054 (20). Mixed model on all medial contacts: z
  −0.00022, p = .0009; z × phase p ≥ .76. Over the 40 mm covered that is ≈
  0.009 z, the size of the ripple-locked effect itself (intercept +0.006).
- Timing (direction index) does not depend on depth in any phase (p ≥ .43);
  |x| and y are null.
- Reading: the ventral end (mOFC-ward) carries most of the medial ripple
  coupling, but in all phases. The encoding-specific part of SK's hypothesis is
  not supported at contact level.

**G2 — the reward response.** PRIMARY: first-traversal correct-uncover
response on z −0.00053 ± 0.00041, p = .21 (41). Null; |x|, valence and
execution are null too. By region: lOFC first traversal +0.025, p = .021,
execution +0.002, p = .28; medial contacts +0.001, p = .65; vlPFC null;
FrontalPole execution +0.007, p = .031 (exploratory). The reward response is
regional (lOFC), not a gradient.

**G3 — lOFC.**
- **G3a PRIMARY: before QUIET ripples (no press in the preceding 2 s) lOFC is
  already up: +0.0100 ± 0.0032, p = .0055 (20).** After-press ripples +0.0090,
  p = .0002; difference −0.0009, p = .75. Direction (post − pre) is negative
  but n.s. for both. So the lOFC pre-ripple rise is NOT the press/uncover
  response. lOFC is ripple-coupled in its own right, leading or co-occurring.
- G3b: uncover response with a ripple following 0.5–2 s later +0.033 (p =
  .085, 15) vs no ripple +0.014 (p = .19, 15); difference −0.006, p = .63, only
  10 sessions with both. Underpowered; no evidence either way.
- Readings: HFB has no stillness-matched control in this pipeline (arrow
  presses are not mapped onto HFB epochs). The readings are real − shifted
  null and own baseline/flank, as for all HFB here.

**C4 — cell ripple modulation by depth.** 88 sites (42 mPFC, 46 mOFC).
**PRIMARY, encode − execute vs z: ρ = −0.04, p = .69.** Encode alone −0.05
(p = .61), execute −0.07 (p = .54), mPFC sites alone +0.06 (p = .72), mOFC vs
mPFC sites all p ≥ .70. Cell firing inside ripples does not change with depth or
phase. Own-baseline only (in vs out of ripples); no stillness-matched version.

### RESULT N1–N2 (same day) — no 'now'-cell response at discovery; the mPFC new-square signal does not survive stillness matching

`mc/analyse/swr_explore/discovery_cells.py`; output
`swr/discovery_cells_2026-10-30/` (`n1_cells.pkl`, `n2_*.csv`, `result.json`,
`N1_discovery_cells.png`, `N2_new_square.png`). Both readings throughout (the
standing rule).

**N1 — lag-0 cells at the reward press** (289 mPFC + mOFC cells).
- **PRIMARY (own baseline), FD − RC vs lag-0 tuning: no positive cluster.** The
  only cluster is NEGATIVE, +0.2…+0.3 s, p = .12. mPFC alone: none. mOFC: the
  same negative bin, p = .16.
- Stillness-matched: FD − RC vs lag-0, NEGATIVE cluster +0.3…+0.5 s, p = .032.
  If anything, cells with stronger lag-0 tuning respond LESS at a first
  discovery than at a repeat.
- **The p = .065 lead does not survive the corrected design.** FD with − without
  a post-reward ripple vs lag-0: no cluster in either reading (own 271 cells,
  matched 175).
- Depth (sites): own reading, no cluster. Matched, FD − RC vs z, POSITIVE
  +0.6…+1.0 s, p = .003 (88 sites), but within mPFC sites none (42). So dorsal
  (mPFC) sites respond more to first discoveries than ventral (mOFC) sites,
  the opposite of 'ventral now cells'. This is an mPFC-vs-mOFC difference, not a
  within-mPFC gradient.

**N2 — evidence for the square just uncovered, reward-locked** (0.05–1.0 s).

| ROI | FD | RC | FD − RC | FD − RC, stillness-matched | FD, own (post − pre-press) |
|---|---|---|---|---|---|
| mPFC (26) | **+0.70, p = .0007** | +0.06 | **+0.66, p = .003** | +0.16, p = .47 | +0.19, p = .38 |
| mOFC (22) | +0.29, p = .25 | −0.07 | +0.28, p = .24 | +0.14, p = .45 | −0.16, p = .51 |
| HC (51) | **+0.60, p = .0005** | +0.32, p = .027 | **+0.47, p = .005** | **+0.42, p = .028** | +0.14, p = .37 |

- Locking to the reward press reproduces the 2026-10-01 mPFC result
  (+0.77 → +0.70; FD − RC +0.66). The arrival-time bug did not create it.
- **But in mPFC it does not survive the stillness-matched reading.** Against RC
  uncovers matched on the following stillness, FD − RC is +0.16, p = .47. The
  own reading (post-press minus the same event's pre-press window) is null as
  well. The subject stands on the square before pressing, so the square is
  already represented then. mPFC's "new square" signal is at least partly the
  longer dwell after first discoveries (1.5 s vs 0.45 s). In HC the FD − RC
  difference survives matching (+0.42, p = .028).
- Per-cell FD − RC contributions vs depth: mPFC + mOFC sites ρ = +0.26,
  p = .022 (75 sites; dorsal more), mPFC sites alone ρ = −0.02, p = .90. Again
  mPFC > mOFC, not a within-mPFC gradient, and not ventral.

**Reading for claim 2.** With the timing fixed and both readings computed:
lag-0 cells do not respond more at discovery; the ripple dependence was not
real; and mPFC's new-square signal is explained by stillness. The surviving
discovery-specific representation of the just-found square is hippocampal.

**Claim figures.** `scripts/exploratory/plot_gradient_claims.py` →
`swr/gradient_claims_2026-10-30/claim{1..5}_*.png/pdf`: one figure per claim,
every panel one piece of evidence from saved per-session or per-cell values
with the statistic that analysis reported (nothing re-tested). New panels
appear as their results exist.

**T4 re-run with the same-event reading (SK, 2026-10-30; NOT pre-registered).**
SK asked whether the stillness-matched contrast is too strict, and set a
standing rule: every test reports BOTH the own (same-event) baseline and the
stillness-matched reading. Same-event reading for T4: the ripple window
against the same ripple's own flanks (0.25–0.5 s either side, same square, no
other ripple within them), z-scored together, through the same label
permutation. Output `swr/plan_vs_store_2026-10-30/T4_path_content/`. The
matched reading reproduces exactly (primary +0.435, p = .043).

| HC, z vs 0 | matched | same-event |
|---|---|---|
| post-correct, first trav., PAST − FUT | +0.44, p = .043 (39) | +0.01, p = .99 (14) |
| post-correct, first trav., FUT | −0.54, p = .011 | −0.47, p = .061 |
| post-correct, first trav., PAST | +0.27, p = .11 | −0.33, p = .21 |
| post-correct, execution, FUT | +0.35, p = .006 | +0.34, p = .19 |
| all ripples, first trav., FUT | −0.10, p = .52 | **−0.52, p = .013** |
| all ripples, first trav., PAST − FUT | −0.04, p = .80 | +0.42, p = .050 |
| all ripples, execution, FUT | **+0.41, p = .001** | **+0.46, p = .030** |

Reading. The own-flank rule costs data, not the matching: post-correct
ripples cluster, so only 14–17 sessions keep ≥ 20 ripples with clear flanks.
Across readings, the robust component is the FUTURE squares. They are
under-represented in first-traversal ripples and over-represented in
execution ripples. The PAST (reverse-path) component does not replicate in the
same-event reading, so "post-correct ripples carry the path just walked"
should not be claimed. "First-traversal ripples are not ahead of the search;
execution ripples are" can be.

**Kept for later:** 'independent of ripples'
by ripple-masked lag maps; the explore → execute morph in warped time; mOFC
ripple content with ripple-derived templates.

## 2026-10-29 — PRE-REGISTRATION: storage vs next-step planning after correct uncovers (tests T1–T5)

Written BEFORE running. Script `scripts/exploratory/swr_plan_vs_store.py`
(T1–T3, T5); module `mc/analyse/swr_explore/path_content.py` (T4). Output
`swr/plan_vs_store_2026-10-29/<test>/`.

**Why.** A co-author reads the post-correct ripple increase as consolidation,
not next-step planning (replay rises under uncertainty, Sagiv et al. 2025). SK:
'loading' (the current state HC → mPFC) is itself a form of storage. It
predicts communication at EVERY new discovery (and again at execution errors),
not at D specifically. So the split these tests can make is STORAGE (loading
or consolidation: tied to new information arriving) vs NEXT-STEP PLANNING
(tied to the upcoming move and goal). No behavioural test separates loading
from consolidation, because both store the same item.

**Shared.** `bundle_v2`, pad 0.25 s, session = unit, seed 42. Ripple rate goes
through `fin.matched_sets` (arrow presses, caliper 0.15 s / 10%) and
`rip.peri_event_rate`. PRIMARY reading = stillness-matched (target − matched
arrow, raw); own baseline −1.5…−1.0 s is secondary. Window 0–0.5 s; cluster =
`rip.sliding_window_test` (0.3 s). Group test = sign-flip over sessions, the
same function for observed and null. First traversal = `stage == 'first
uncovers'`. Each primary is judged at α = 0.05 (they test different accounts);
secondaries are reported uncorrected and labelled as such.

**T1 — does the increase need a KNOWN next goal?** Correct first-traversal
uncovers, D vs A–C (A–C pooled into one cell). After A–C the next reward is
unknown; after D the next goal (A) is known for the first time.
- T1a PRIMARY: ripple rate, D − (A–C), stillness-matched, 0–0.5 s. Next-step
  planning predicts > 0; storage ≈ 0; an uncertainty account < 0. Reported
  with its minimum detectable effect (80% power) and the A–C effect for scale.
  Secondary: A, B, C, D separately; own baseline.
- T1b PRIMARY: mPFC HFB (partition 2026-09-28) locked to the same uncovers,
  real − mean of 20 shifted nulls (5–120 s, either sign), baseline
  −1.5…−1.0 s, window 0–0.5 s: D − (A–C). Secondary: mOFC, lOFC,
  MedialFrontal; ripple-locked HFB (ripples whose nearest preceding press
  within 2 s is that uncover; peri ±250 ms minus flank 250–750 ms; ≥ 10
  ripples per session and set); execution errors vs execution correct uncovers
  (SK: loading predicts communication again at errors). Positive control:
  MedialFrontal ripple-locked HFB for all correct first-traversal uncovers > 0
  (Fig 4i).

**T2 — the reward clock or the departure clock?** Correct first-traversal
uncovers followed by ≥ 1.0 s of stillness, matched 1:1 to arrow presses on the
same stillness. Excess = uncover − matched arrow. PRIMARY per session:
excess in 0–0.5 s after the press − excess in the 0.5 s before the NEXT press
(the departure). Storage predicts > 0, planning < 0. Secondary: each excess vs
0; ≥ 1.5 s stillness with 0.75 s windows; both alignments as profiles with
cluster tests.

**T3 — behaviour.** Event-level, within session: partial Spearman (ranks
residualised on the covariates), Fisher z, sign-flip over sessions; ≥ 20
events per session. Ripple rate in a pause = HC ripples from the uncover to
the next press (capped at 2 s) per clean second, averaged over derivations
(NaN if less than half clean).
- T3a STORAGE, PRIMARY: predictor = rate in the pause after the FIRST
  discovery of reward X; outcome = wrong uncovers while seeking X on the NEXT
  repeat. Covariates: pause length, state, wrong uncovers before discovering X.
  Storage predicts ρ < 0.
- T3b NEXT-STEP PLANNING, PRIMARY: predictor = rate in the pause after a
  correct uncover of X on any repeat after the first traversal; outcome =
  wrong uncovers on the IMMEDIATELY following segment (seeking X+1).
  Covariates: pause length, state, repeat index. Planning predicts ρ < 0.
- Secondary: path excess (arrow presses − Manhattan distance) as the outcome;
  T3b in learning and execution separately; T3a's predictor against the
  immediately following first-traversal segment (exploration; no prediction).

**T4 — what do post-correct HC ripples carry: the path just walked or the path
ahead?** `future_shift` machinery: same-block references, square-label
permutation z (1000), stratum-matched still moments (phase × reward square ×
time since arrival × time to departure). Window = the ripple ±125 ms; controls
use the same geometry with no ripple within 0.25 s. PRIMARY: HC units, ripples
in the stillness after a correct uncover, first traversal, PAST − FUT (mean of
past1–3 minus fut1–3), ripple − control. Storage/reverse predicts > 0,
planning/forward < 0. Gate: NOW in control moments > 0. Secondary: execution;
all ripples; PAST, FUT and NOW separately; mPFC and mOFC units (expected to
fail the gate).

**T5 — novelty without reward.** First traversal; a square is NOVEL if it has
not been uncovered before in this block. Cells: error-novel, error-familiar,
correct-novel, correct-familiar, each vs matched arrows. PRIMARY: error novel −
familiar (0–0.5 s, stillness-matched). Secondary: correct novel − familiar;
correct − error within novel squares; own baseline.

**Not run:** the directionality test (onset latencies), deprioritised.

### RESULT (same day) — no primary supports next-step planning; the increase follows the reward, not the departure or novelty

Output `swr/plan_vs_store_2026-10-29/{T1a_knowledge, T1b_hfb_state,
T2_two_clocks, T3_behaviour, T5_novelty, T4_path_content}/`, each with
`result.json`, csv tables and a figure. Session = unit throughout.

**T1a, ripple rate, D vs A–C (first traversal, correct).**

| cell | stillness-matched, 0–0.5 s | own baseline, 0–0.5 s |
|---|---|---|
| A–C (61) | +0.036 Hz, p = .027; cluster +0.35…+0.65 s, p = .007 | +0.031, p = .012 |
| D (54 / 61) | −0.004, p = .89 | **+0.056, p = .0008**; cluster +0.25…+0.65, p = .003 |
| **D − (A–C), PRIMARY** | **−0.034 [−0.083, +0.015], p = .18** | +0.025, p = .10 |

The two readings disagree about D: D is the strongest state against its own
baseline and null against its matched presses. **Power:** the minimum detectable
D − (A–C) at 80% power is 0.069 Hz on 54 sessions, **1.95× the A–C effect
itself**. So a D-specific increase as large as the whole post-correct effect
could go undetected. T1a cannot decide between the accounts (SK's power concern
confirmed).

**T1b, frontal HFB, same events.**
- **PRIMARY, mPFC uncover-locked D − (A–C): +0.000 ± 0.008 z, p = 1.00**
  (26 sessions; MDE 0.024 z); no cluster. mPFC has no uncover response to
  either (A–C −0.003, D −0.001), and none in execution (+0.002, p = .31).
- Positive control passes: MedialFrontal ripple-locked HFB after correct
  first-traversal uncovers +0.0086 z, p = .012 (Fig 4i; A–C alone +0.0115,
  p = .0001).
- Ripple-locked D − (A–C): mPFC +0.004 (p = .72, 19 sessions), MedialFrontal
  +0.003 (p = .71, 26), lOFC +0.002 (p = .75, 12); mOFC 2 sessions. D alone is
  numerically the largest ripple-locked value (MedialFrontal +0.015, p = .056).
- Execution errors − execution correct (SK: loading predicts communication
  again at errors): null in every ROI, uncover-locked (mPFC +0.002, p = .85;
  MedialFrontal +0.005, p = .32) and ripple-locked (≤ 19 sessions). The ripple
  RATE at execution errors is also n.s. (Fig 4 panels: +0.012 Hz matched,
  p = .50).
- Exploratory: lOFC responds to correct uncovers in the first traversal
  (+0.029 z, p = .006, 21 sessions) but not in execution (+0.002, p = .24).

**T2, reward clock vs departure clock** (≥ 1.0 s stillness, 2838 matched pairs,
stillness 1.60 / 1.60 s).
- **PRIMARY, excess 0–0.5 s after the uncover − excess 0.5 s before departure:
  −0.001 ± 0.017 Hz, p = .96.** Neither window alone differs from the matched
  presses (+0.010, p = .46; +0.011, p = .36).
- Secondary, time-resolved: the excess forms a cluster at **+0.35…+0.65 s
  after the uncover (p = .003)** and **no cluster in departure-locked time**.
  The same holds at ≥ 1.5 s stillness (cluster +0.35…+0.65, p = .005; none
  departure-locked; windowed after − before +0.024, p = .27, after alone
  +0.032, p = .061).
- Reading: the primary window (0–0.5 s) covers only half of the effect, which
  sits at a fixed latency from the reward. Pause lengths vary, so a
  departure-locked increase would be sharp in departure time and smeared in
  uncover time; the data show the opposite. Nothing is tied to departure.
  Stated as the secondary it is; the pre-registered primary is null.

**T3, behaviour** (partial Spearman, Fisher z; negative = more ripples, better).

| test | n | z | p |
|---|---:|---:|---:|
| **T3a storage**: first-discovery pause → errors on that reward next repeat | 57 | −0.018 ± 0.014 | .21 |
| **T3b planning**: later-repeat pause → errors on the next segment | 61 | +0.008 ± 0.005 | .10 |
| T3a, path excess | 61 | −0.018 ± 0.013 | .17 |
| T3b, path excess | 61 | +0.002 ± 0.005 | .64 |
| T3b, learning repeats | 42 | +0.026 ± 0.025 | .32 |
| T3b, execution repeats | 61 | +0.000 ± 0.004 | .92 |
| control: first-discovery A–C → next exploration segment | 61 | −0.004 ± 0.016 | .79 |

Both primaries null. The storage sign is the predicted one, the planning sign
is the wrong one. A ripple occurs in 41% of first-discovery pauses (median
1.47 s) and 22% of later ones (0.45 s). Consistent with 2026-09-11: pause
ripples do not predict behaviour.

**T5, novelty without reward** (first traversal, stillness-matched).
- **PRIMARY, error novel − familiar: +0.017 [−0.010, +0.044] Hz, p = .21.**
  No 0–0.5 s effect. The only cluster is BEFORE the press (−0.25…−0.05 s,
  p = .014). Own baseline +0.001, p = .98.
- A reward on a square already tried in this block still raises the rate:
  correct-familiar own baseline +0.049, p = .0002; matched cluster
  +0.35…+0.75 s, p = .001. Correct novel − familiar +0.017, p = .45.
- So the post-correct increase follows the REWARD, not a new square. This
  closes the novelty confound flagged on 2026-10-01 for ripple rate.

**T4, path content of HC ripples** (`path_content.py`; HC units, 61
sessions, median 386 eligible ripples per session; ripple window ±125 ms vs
stratum-matched still moments in the same post-uncover pauses).
- **Gate passes:** NOW in control moments +0.85 z, p < .001 (post-correct);
  +1.47, p < .001 (all).
- **PRIMARY, post-correct, first traversal, PAST − FUT: +0.44 ± 0.21 z,
  t(38) = 2.09, p = .043.** That is the storage/reverse direction. Its parts:
  FUT −0.54 (p = .011), PAST +0.27 (p = .11), NOW +0.15 (p = .33). So
  post-correct ripples during encoding are LESS like the upcoming squares than
  matched still moments are, and lean towards the path just walked.
- Secondary, execution: the sign flips. Post-correct FUT +0.35 (p = .006),
  PAST − FUT −0.22 (p = .13); all execution ripples FUT +0.41 (p = .001),
  PAST − FUT −0.29 (p = .021).
- **POST HOC, not pre-registered** (`phase_contrast_posthoc.csv`): post-correct
  first traversal − execution, PAST − FUT +0.70 ± 0.21, p = .002; FUT −0.94 ±
  0.23, p = .0003, 41 sessions.
- mPFC units: nothing (all p ≥ .06). mOFC post-correct first traversal
  PAST − FUT +0.99 (p = .029, 15 sessions), but mOFC fails the gate in that
  subset (NOW in controls +0.08, p = .73), so it is not interpretable.
- ⚠ In the first traversal the upcoming squares are exploration, chosen during
  the pause. "Less future" is therefore "not ahead of the search", not a
  rejection of a known goal. Of all the accounts, only next-step planning
  predicted the opposite sign here.

**Reading.** No primary supports next-step planning. The one positive primary
(T4) points to storage. During encoding, post-correct HC ripples carry the
path just walked rather than the squares ahead. Once the route is known, they
carry the path ahead (post hoc, p = .0003). The rate increase sits at a fixed
latency after the reward rather than before departure (T2, secondary). It
is not driven by square novelty (T5) and does not predict the next step
(T3b). mPFC HFB shows no D-specific (plan-completion) response (T1b). The D
test on ripple rate is underpowered: the MDE is ~2× the effect (T1a). Nothing
here separates loading from consolidation, as stated in advance. The ripples
that rise in rate (encoding) look like storage of what just happened. The
ripples that carry the route ahead (execution) do not rise in rate.

## 2026-10-08 (e) — 50 shifted nulls: OFC / mOFC vs mPFC, ripple- and uncover-locked; a bounds bug fixed

`swr_hfb_ripple_coupling.py phases --lefts=OFC,mOFC --n_shifts=50
--stability=True --sets=all,first_traversal,reward_uncover,reward_uncover_learning,reward_uncover_execution`.
Output `swr/hfb_phases_50shifts_2026-10-08/` (`summary.csv` incl. the
half-split stability rows, `OFC_vs_mPFC/` and `mOFC_vs_mPFC/` with
`overview.png` and one folder per set). ~20 min, background.

**Code.** `phases` rewritten: mOFC, lOFC and mPFC extracted in ONE pass
(`roi_sums` returns sums + counts so OFC = mOFC + lOFC is formed per event);
`--lefts`, `--sets`, `--n_shifts`, `--stability`; new uncover sets, phase by
the uncover's repeat (`uncover_phase`: repeat <= the first fully correct one =
learning, so the D completing the first correct loop is learning).
**BUG FIXED in `roi_epochs`:** the left-edge check read `ci - off[0] >= 0`
(always true), so a window running off the START of the recording wrapped to
the end of the array -- in practice shifted-null epochs within ~2 min of the
recording start (shifts reach −120 s). Re-running the 6-shift OFC "all" set
with the fix reproduces p = 0.072 / paired 0.007 to three decimals; OFC peak
moves −50 → −60 ms. Every earlier `swr_hfb_ripple_coupling.py` output
(run, timeline, chain, 6-shift phases) carried it; effect negligible.

**Outlier (SK).** mOFC "all ripples" direction +0.035 is session 62
(BY2-YFP); its mOFC derivations LF3aOFC02-03 (MNI z −14.3) and RF2bOFC02-03
(z −14.7), both FreeSurfer medialorbitofrontal -- exactly the mOFC median z
(−14.7; IQR −16.8…−13.5). Not misallocated. The session also has two lOFC
derivations (z −6.3; z +2.2, BN A12/47l), which dilute it to +0.012 in the
collapsed OFC.

**Results (50 shifts; stability = the same test on each half of them).**

| pair | set | left ROI rise | mPFC rise | mPFC − left direction, p (halves) | paired p (halves) |
|---|---|---|---|---|---|
| OFC | all ripples | −350…+230, p<.001 | −140…+400, p<.001 | +0.0050, **0.045** (0.053 / 0.044) | **0.006** (0.007 / 0.005) |
| OFC | first traversal | −220…+30, p=.003 | none (p=.32) | +0.0068, 0.16 (0.12 / 0.22) | 0.10 |
| OFC | correct uncovers | none (best .066) | −440…−330, p=.047 (.033 / .071) | −0.0033, 0.37 | **0.006** (0.009 / 0.005) |
| OFC | uncovers, learning | none | none | −0.0052, 0.47 | 0.35 |
| OFC | uncovers, execution | none | none | −0.0023, 0.55 | **0.014** (0.014 / 0.017) |
| mOFC | all ripples | −270…+110, p<.001 | as above | +0.0050, 0.11 (0.075 / 0.16) | 0.21 |
| mOFC | first traversal | −220…+50, p=.002 | none | +0.0067, 0.27 | 0.20 |
| mOFC | correct uncovers | none | as above | −0.0022, 0.60 | 0.20 |
| mOFC | uncovers, learning | none | none | +0.0115, 0.13 | 0.38 |
| mOFC | uncovers, execution | none | none | −0.0056, 0.19 | **0.033** (0.019 / 0.052) |

Contrasts: OFC all ripples, z-scaled, OFC bigger −160…−30 ms (p = 0.045;
paired −250…−130, p = 0.030); mOFC all ripples, z-scaled, mOFC bigger
−150…+30 (p = 0.015). No peak-scaled contrast anywhere.

**Reading.**
- With 50 nulls the p values are stable to ~0.01 across halves.
- All ripples: OFC rises around and before the ripple, mPFC later;
  mPFC − OFC p = 0.045 unpaired (borderline, 0.044–0.053 across halves),
  0.006 paired (robust). For mOFC alone the ordering is n.s.
- First traversal: OFC and mOFC rise significantly, before/at the ripple;
  mPFC does not rise significantly; the ordering test is n.s. (no mPFC rise to
  order against).
- Correct uncovers: no ROI rises against its pre-uncover baseline in any
  phase. The paired direction index is negative (OFC rises later than mPFC
  relative to the uncover) for all uncovers and execution uncovers --
  consistent with a small post-uncover OFC response (peak +0.010 z at
  +250 ms) that does not survive the cluster test. The mPFC pre-uncover
  cluster (−440…−330 ms) is unstable across halves (0.033 / 0.071).
- Uncorrected across sets.

## 2026-10-08 (d) — the phase overview again, mOFC alone vs mPFC

`swr_hfb_ripple_coupling.py phases --left=mOFC` (the left ROI is now a
parameter; `--left=OFC` writes to the same folder name as (b)). Output
`swr/hfb_mofc_mpfc_phases_2026-10-08/`. Identical method to (b): same
extraction, 6 shifts, learning INCLUDES the first fully correct repeat. 18 mOFC
/ 35 mPFC sessions in every set, 11 with both.

| set | mOFC rise (cluster) | mPFC rise (cluster) | mPFC − mOFC direction | p unpaired | p paired (11) |
|---|---|---|---:|---:|---:|
| all ripples | −260…+100, p<.001 | −140…+340, p<.001 | +0.0056 | 0.11 | 0.16 |
| first traversal | **−220…−40, p=.009** | none | +0.0063 | 0.30 | 0.17 |
| learning (incl.) | −90…+80, p=.017 | none | +0.0038 | 0.45 | 0.43 |
| execution | −160…+40, p=.003 | −110…+300, p<.001 | +0.0105 | **0.032** | 0.55 |
| correct uncovers | none | none | −0.0028 | 0.53 | 0.14 |

No time-resolved contrast is significant in any set, unpaired or paired.

**Reading.** mOFC rises around ripples in every phase, and in the first
traversal its cluster lies entirely BEFORE the ripple (−220…−40 ms) while
mPFC has no significant rise there. The ordering (mPFC later) has the same
sign in all four ripple sets but is significant only unpaired in execution.
No reward-uncover response in either ROI.
**Caveat:** with this extraction mOFC's all-ripple rise is significant
(p < 0.001, peak 0.026 z); the region run's sliding test gave no significant
mOFC cluster (best p = 0.082, peak 0.019 z). Same extraction-dependence as
(c), now in a per-ROI test. Uncorrected across the five sets.

## 2026-10-08 (c) — extraction audit: same ripples, same HFB, which averaging choice moves OFC vs mPFC?

`scripts/exploratory/swr_hfb_extraction_audit.py`; output
`swr/hfb_extraction_audit_2026-10-08/` (`extraction_audit.png/pdf`,
`variants.csv`, `ripple_counts.csv`, `result.json`). ~1 min. Nothing
re-detected: both extractions read the same `bundle_v2` ripples and HFB.

**Correction to (b).** I described the phases extraction as "each ripple
counted once". It is not: it merges only peaks that round to the same
millisecond (27 of 77,228 rows). Same-event detections on two HC channels a
few ms apart stay in both extractions.

**Duplicates.** Cross-channel ripples within ±50 ms: 3.2% of all ripple
rows; with each channel circularly shifted (chance) 2.3%. So ~1% are true
co-detections (±25 ms: 1.7% vs 1.1% chance). A ±50 ms merge removes 3.5% of
rows (incl. the chance ones).

**Ripple RATE results are not affected**: `ripples.rate_by_unit` computes a
rate per HC derivation (its own ripples over its own clean time) and averages
derivations within a session; a ripple on two channels is never summed into
one rate, and the unit is the session.

**Variants** (one switch at a time from the region method; both ends
reproduce their saved runs exactly):

| variant | p unpaired | p paired | curve r vs region (OFC / mPFC) |
|---|---:|---:|---|
| region method | 0.032 | 0.016 | 1 / 1 |
| + artifact rule: cortical channel only | 0.063 | 0.016 | 0.85 / 0.81 |
| + the conditions run's 6 shifts (not 8) | 0.069 | 0.023 | 0.95 / 0.94 |
| + ripples pooled, each ripple weighted once | 0.042 | 0.006 | 0.94 / 0.95 |
| phases method (all three) | 0.072 | 0.007 | 0.73 / 0.72 |
| phases + ±50 ms dedup | 0.085 | 0.004 | 0.72 / 0.72 |

**Reading.** The artifact rule changes the per-session curves most (it lets
in ripples whose HC channel has an artifact in the ±1 s window). But drawing
a different random set of shifted nulls alone moves the unpaired p from 0.032
to 0.069. The unpaired timing result is therefore within the noise of the null
estimate; the paired result is < 0.05 in every variant (0.004–0.023).
Group-mean curves are near-identical across all variants.

## 2026-10-08 (b) — OFC vs mPFC by phase, and locked to correct uncovers

`swr_hfb_ripple_coupling.py phases` (new). Output
`swr/hfb_ofc_mpfc_phases_2026-10-08/`: `overview.png/pdf`, one subfolder per
set with the three single figures, captions, `result.json`; `summary.json`,
`counts.csv`. ~3.5 min. `swr_hfb_direction.py` refactored first: analyses
(`analyse_pair`, `analyse_contrast`, no I/O), drawing onto a given axis
(`draw_sliding/timing/contrast`), one writer (`report_pair`); `sliding` and
`contrast` are thin wrappers and reproduce their earlier outputs exactly
(checked: timing, clusters and contrast identical; the contrast caption file
is now `caption_contrast.txt`).

**Design.** New single-ripple extraction from `bundle_v2`: OFC = every mOFC
or lOFC derivation (mean per ripple over those clean for the window), mPFC;
per session the mean over ripples, real − mean of the 6 shifted nulls.
Phases: first traversal = grid onset → first t_D; learning = grid onset →
END of the first fully correct repeat, that repeat INCLUDED (SK's wording;
the project default is exclusive); execution = after it. Correct uncovers:
every correct uncover, all phases, epoch ±1.5 s, baseline −1.5…−1.0 s
before the uncover. Ripple sets: baseline |t| ≥ 750 ms. Then the same three
analyses per set. 27 OFC / 35 mPFC sessions in every set (17 with both).
Uncorrected across the five sets.

| set | OFC rise (cluster) | mPFC rise (cluster) | mPFC − OFC direction | p unpaired | p paired (17) | contrast |
|---|---|---|---:|---:|---:|---|
| all ripples | −340…+220, p<.001 | −140…+340, p<.001 | +0.0048 | 0.072 | **0.007** | paired z: OFC bigger −240…−120, p = .017 |
| first traversal | none | none | +0.0057 | 0.22 | 0.074 | none |
| learning (incl.) | −240…+60, p<.001 | none | +0.0054 | 0.16 | **0.011** | none |
| execution | −260…+80 p<.001; +100…+270 p=.024 | −110…+300, p<.001 | +0.0081 | **0.027** | 0.23 | none |
| correct uncovers | none | none | −0.0037 | 0.37 | **0.017** | paired z: OFC bigger +200…+340, p = .045 |

**Reading.**
- mPFC − OFC is positive (mPFC later) in all four ripple sets, but which
  estimator reaches p < 0.05 flips between sets (learning: paired only;
  execution: unpaired only). Direction consistent, significance fragile.
- The all-ripple reference from this extraction gives unpaired p = 0.072
  (the region run gave 0.032) and paired p = 0.007 (region run 0.015): the
  unpaired all-ripple result depends on the extraction.
- First traversal alone has no significant rise in either ROI (fewer
  ripples); the OFC rise is clearest in learning and execution.
- Locked to correct uncovers neither ROI rises against its pre-uncover
  baseline; the paired sessions show OFC above mPFC at +200…+340 ms, in line
  with the lOFC post-uncover response (2026-09-29 (h)). The sign of the
  timing index reverses (OFC later relative to the uncover).
- No unpaired time-resolved contrast is significant in any set.

## 2026-10-08 — panel a) as a contrast, mPFC vs OFC over time: no time point differs

`swr_hfb_direction.py contrast --left=OFC` (new; `sliding_curves` factored out
of `sliding`, whose output is unchanged). Output
`swr/hfb_direction_contrast_OFC_2026-10-08/` (`contrast_panel_a.pdf/png`,
`caption.txt`, `result.json`). Also: `sliding` now draws its timing test
(`sliding_timing.pdf/png`, `caption_timing.txt`); re-run into
`..._sliding_{OFC,lOFC,mOFC}_2026-10-08/`, same numbers as (e).

**Design.** The `sliding` curves (real − null, own baseline ≥ 750 ms, 50 ms
sliding mean) for OFC (mOFC + lOFC, 27 sessions) and mPFC (35). Two scalings:
as measured (z), and each ROI / its group-mean peak (shape and timing only, as
the twin-axis panel draws it). Welch t per bin, cluster mass, ROI labels
permuted over sessions, scaling recomputed in every permutation by the same
function; paired check on the 17 shared sessions (sign-flip,
`cluster_perm_time`). Figure: both components on one axis, the gap filled by
which is bigger.

**Result.** Unpaired: no cluster forms in either scaling. Paired: nothing
significant (best: mPFC bigger −670…−580 ms, p = 0.07–0.09, far from the
ripple). Visually OFC is above just before the ripple and mPFC above after it,
which is what the direction index integrates (p = 0.032, (e)), but no single
time bin carries a reliable difference.

## 2026-10-07 (e) — panel a) with OFC collapsed (mOFC + lOFC): both rise, OFC earlier

`swr_hfb_direction.py sliding --left={OFC,lOFC,mOFC}`; outputs
`swr/hfb_direction_sliding_{OFC,lOFC,mOFC}_2026-10-07/`. (`..._sliding_2026-10-07/`
from (d) is the mOFC run before the naming change and without the timing
test; superseded by `..._sliding_mOFC_...`.) OFC = every mOFC or lOFC
derivation, averaged per session. Same test as (d); added: the timing test the
clusters cannot give -- the direction index (post − pre, ±250 ms, unsmoothed),
mPFC − OFC, ROI-label permutation, plus paired on shared sessions.

| left ROI | n | cluster (increase) | peak | mPFC − ROI direction | p (unpaired) | p (paired, n) |
|---|---:|---|---|---:|---:|---:|
| OFC (m+l) | 27 | −250 to +170 ms, p < 0.001 | −50 ms | +0.0065 | **0.032** | **0.015** (17) |
| lOFC | 21 | −240 to +170 ms, p < 0.001 | −10 ms | +0.0077 | 0.055 | 0.043 (15) |
| mOFC | 18 | none (best −100…−10, p = 0.08) | −50 ms | +0.0076 | 0.084 | 0.23 (11) |
| mPFC | 35 | −150 to +400 ms, p < 0.001 | +20 ms | — | — | — |

**Reading.** Pooled OFC has a significant ripple-locked rise, and mPFC's rise
is later than OFC's (direction index, p = 0.03 / paired 0.015).
**Caveats:** (1) collapsing OFC was chosen AFTER seeing mOFC (p = 0.08) and
lOFC (p = 0.055) separately -- three overlapping variants were looked at, so
p = 0.03 is not a clean single test; (2) all-task ripples include those that
follow a press, where the press response sits before the ripple
(2026-10-07 (b): the mPFC − mOFC ordering weakened from p = 0.03 to 0.30
when unlabelled first-traversal ripples were added); (3) no same-ripple
coupling between OFC and mPFC (2026-10-07 (b), (c)).

## 2026-10-07 (d) — panel a) redrawn with a sliding-window test per ROI

`swr_hfb_direction.py sliding`; output `swr/hfb_direction_sliding_2026-10-07/`
(`sliding_panel_a.pdf/png`, `caption.txt`, `result.json`). All ripples (region
run), real − shifted null, each ROI re-baselined to its OWN mean ≥ 750 ms from
the ripple (the epoch edges; the 250–750 ms flank of the other panels would
cancel the tested window). 50 ms sliding mean, `swr_sakon.cluster_perm_time`
over ±750 ms (sign-flip over sessions, 10,000 perms, two-sided). Twin y-axes,
each scaled to its ROI's peak.

- mPFC (35 sessions): increase −150 to +400 ms, p < 0.001; peak +0.012 z at
  +20 ms.
- mOFC (18 sessions): no significant cluster; best −100 to −10 ms, p = 0.082;
  peak +0.019 z at −50 ms.

Cluster extents and peak times are descriptive -- a cluster test does not
localise onsets, so "mOFC before, mPFC after" is not established by this.

## 2026-10-07 (c) — same chain with lOFC as the sender: lOFC LEADS the ripple, but does not predict mPFC

`swr_hfb_ripple_coupling.py chain --sender=lOFC` (sender is now a parameter;
output `..._chain_lOFC_2026-10-07/`). Same design as (b): every
first-traversal ripple, real − shifted, sign-flip over sessions. lOFC is the
one frontal ROI with an event-locked reward response (2026-09-29 (h)).

| test | n | effect | p |
|---|---:|---|---:|
| **link 1**: lOFC pre-ripple | 21 | +0.0120 ± 0.0029 | **0.0003** |
| lOFC post-ripple | 21 | +0.0051 ± 0.0031 | 0.12 |
| lOFC direction (post − pre) | 21 | −0.0069 ± 0.0024 | **0.009** |
| **link 2**: mPFC post-ripple | 35 | +0.0058 ± 0.0028 | 0.045 |
| ordering: mPFC − lOFC direction | 35/21 | +0.0082 | 0.066 |
| **chain**: asymmetry (lOFC pre → mPFC post − reverse) | 15 | −0.016 ± 0.026 | **0.53** |

**Reading.** Unlike mOFC, lOFC is TIME-specific: up before the ripple, not
after (direction p = 0.009). It is the only frontal region in this branch that
leads the ripple. But on the same ripple, lOFC-before does not predict
mPFC-after (+0.000 vs shifted; reverse +0.016). Caveat for the lead: lOFC
also responds to the uncover itself, and ripples follow presses (median
0.4 s), so the pre-ripple lOFC rise may be the press/uncover response; not
yet separated (a check: first-traversal ripples with no event in the
preceding 2 s).

## 2026-10-07 (b) — the chain, ripple-locked, EVERY first-traversal ripple: links 1 and 2 hold, the chain does not

`swr_hfb_ripple_coupling.py chain` (new command). Output
`swr/hfb_ripple_coupling_chain_2026-10-07/` (`chain_ripple_locked.png`, csv,
`chain_curves.npz`, `result.json`). ~15 s.

**Design.** No event anywhere. First traversal = grid onset → first t_D
(`swr_windows.windows_phase3_whole`); every ripple inside counts, labelled or
not (median 210 per session). HFB real − 6 shifted nulls, flank-normalised,
pre [−250, 0) / post (0, +250] ms. Pre-specified by SK before the run: link 1,
link 2, chain; the chain needs all three, so each at 0.05 uncorrected
(intersection–union). Sign-flip over sessions.

| test | n | effect | p |
|---|---:|---|---:|
| **link 1**: mOFC pre-ripple | 18 | +0.0152 ± 0.0048 | **0.006** |
| mOFC post-ripple | 18 | +0.0101 ± 0.0062 | 0.12 |
| mPFC pre-ripple | 35 | +0.0046 ± 0.0022 | 0.044 |
| **link 2**: mPFC post-ripple | 35 | +0.0058 ± 0.0028 | **0.045** |
| ordering: mPFC − mOFC direction (label perm) | 35/18 | +0.0063 | 0.30 |
| **chain**: asymmetry (mOFC pre → mPFC post − reverse) | 11 | +0.020 ± 0.024 | **0.43** |

**Reading.** Each ROI is up around the ripple, but neither is TIME-specific:
mOFC is up after as well as before (direction −0.005, p = 0.40), mPFC before as
well as after (+0.001, p = 0.68). So links 1 and 2 pass as stated, but they
would also pass for the reverse chain. The same-ripple coupling is absent in
both directions (−0.006 and −0.026 vs shifted). The chain is not supported.

**Against 2026-10-02.** With event-LABELLED first-traversal ripples the
mPFC − mOFC ordering was +0.015, p = 0.030; with every first-traversal ripple it
is +0.006, p = 0.30. The ordering lives in the ripples that follow a press or
uncover within 2 s -- where the press response sits before the ripple
(`swr_hfb_event_latency.py`). Consistent with the press response contributing
to the ordering; not tested directly.

## 2026-10-07 (RT) — PRE-REGISTRATION: is mPFC's 'now' code visible only when its profile is measured IN ripples?

Written BEFORE running. Module `mc/analyse/swr_explore/ripple_templates.py`,
`python scripts/swr_content_explore.py ripple_templates`.

**SK's idea.** mPFC ripple windows during encoding lean towards the current
square (+0.31, P = 0.13, all-time templates). If ripples inform mPFC about
'now' while it otherwise codes future lags, then a spatial profile built from
all task time (mostly lagged) is the wrong template for ripple windows. A
profile built from ripple windows alone should score the current square
better.

**Templates.** Per cell, leave-one-configuration-out (`cv_group` = `grid_id`),
z-scored across squares, unobserved squares = 0 (abstain):
- `rip`: rate per square from ripple windows only (each ripple's own detection
  duration).
- `flank`: rate from each ripple's ONE matched flank (`matched_flanks`,
  `max_per=1`): same occupancy interval, same width. So the template has the
  same squares, the same dwell and the same amount of data, and only 'inside a
  ripple' differs. **This is the fair comparison.**
- `all`: the standard all-time profile, for reference only (~40× the data).

All three are built from the SAME ripples: those that have a flank (~89%).

**Test windows.** The same paired ripples and their flanks, from the held-out
configuration. Counts are z-scored per cell over both window types. Score =
current square minus the mean of the other 8, z against 1000 permutations of
the square labels. Session is the unit. Phases: encode / execute (the reported
split).

**PRIMARY.** mPFC, encode ripple windows: z(`rip`) − z(`flank`), paired across
sessions, two-sided t (sign-flip P alongside).

**Gate.** HC_all, encode ripple windows, z(`rip`) > 0. Budget measured
beforehand: the median mPFC cell has 4 ripple spikes per square in a
leave-one-configuration-out template (HC 6.5). If ripple-only templates cannot
recover HC's known current-location code, they are too noisy and the mPFC
comparison is uninterpretable.

**Secondaries** (reported, uncorrected):
1. z(`rip`) > 0 alone in mPFC encode.
2. Execute phase, same contrast.
3. The 2×2 interaction (template × window): [rip − flank template on ripple
   windows] − [rip − flank template on flank windows]. Positive = a code that
   exists only DURING ripples.
4. Encode-only templates (built from the other configurations' encode windows).
5. mOFC, HC_all.

**Known limits.** Flanks are taken before the ripple where there is room
(`matched_flanks` order), so the flank template leans pre-ripple. The
leave-one-configuration-out rule is kept even though the budget is thin.
Relaxing it would let the template see the held-out layout.

### RESULT (same day) — null in mPFC; the gate passes; an unplanned mOFC lead

`swr/ripple_templates_2026-10-07/` (per_session.csv, summary.csv,
results_overview.json, settings.json); Figure 39
(`ripple_content_figures_2026-10-07/fig39_ripple_templates.{jpeg,pdf}`).
Session = unit; two-sided t (sign-flip P agrees in every row).

| ROI | encode ripple windows | ripple-only T | flank T | all-time T | ripple-only − flank |
|---|---|---|---|---|---|
| **mPFC** (26) | z | +0.11 (P = .64) | +0.18 (.35) | +0.28 (.19) | **−0.08, t(25) = −0.25, P = .81** |
| HC_all (51) | z | **+0.31 (P = .008)** gate passes | +0.07 (.61) | +0.55 (.001) | +0.23 (.18) |
| mOFC (22) | z | +0.10 (.56) | −0.42 (.09) | −0.55 (.024) | +0.52 (.047) |

- **PRIMARY null.** Building mPFC's spatial profile from ripple windows does
  not make encode ripples look more 'now' than a profile built from the same
  amount of non-ripple data (−0.08, P = 0.81). With encode-only templates it
  is −0.21 (P = 0.47). The 2×2 interaction is null too (+0.34, P = 0.48), and
  execute ripples are null (−0.08, P = 0.78).
- **The gate passes.** Ripple-only templates recover HC's current-location code
  in encode ripples (+0.31, P = 0.008) despite ~6.5 spikes per square, so the
  budget is not what makes mPFC null. The all-time template does better in HC
  (+0.55) because it has ~40× the data.
- **mOFC, exploratory.** Here the sign flips with the template source. The
  all-time template gives the known NEGATIVE (−0.55, P = 0.024; cf. −0.49 and
  −0.544 in earlier entries), and the flank template leans the same way
  (−0.42). The ripple-only template is positive (+0.10), so ripple-only − flank
  = +0.52 (P = 0.047). With encode-only templates: ripple-only +0.54
  (P = 0.008), ripple-only − flank +0.82 (P = 0.013). This would explain the
  unexplained mOFC 'anti-location' ripple result: during ripples mOFC's spatial
  profile is not its usual one. ⚠ Not pre-registered as primary. It is 2-3 of
  ~108 cells, does not survive any correction, and the interaction that would
  make it ripple-SPECIFIC is not significant (+0.58, P = 0.19). Worth a
  targeted, pre-registered test: correlate each mOFC cell's ripple-only and
  all-time maps (an inverted map predicts r < 0).

## 2026-10-07 (FS) — PRE-REGISTRATION: does a ripple move frontal activity from NOW to the UPCOMING squares?

Written BEFORE running. Module `mc/analyse/swr_explore/future_shift.py`,
`python scripts/swr_content_explore.py future_shift`.

**Why a new design (SK).** Every ripple-content test so far scored windows
against templates built from OTHER configurations (leave-one-configuration-out).
If mPFC codes 'now' early in a block and a lagged future later, a whole-task
template averages two codes, and 2026-10-02 shows mPFC place maps do not
transfer across configurations at all (r = −0.002). So here every reference
comes from the SAME block.

**References.** For an event in traversal τ of block b:
R(L) = per-feature mean activity while the subject stood on square L, over every
OTHER traversal of block b (primary, `block`), or the 4 nearest (`near4`,
sensitivity). z-scored across the squares visited for >= 0.2 s (sum to zero,
unvisited = 0). R(L) is 'what this population does when standing on L, in this
block, now'. It assumes nothing about which lag is coded: if mPFC is lagged,
R(L) is lagged too. Moving towards R(future square) therefore means the state
jumped ahead along the route.

**Events.** Hippocampal ripples (deduplicated at 50 ms, duration <= 150 ms)
with the subject on ONE square throughout: pre = [t − 0.35, t − 0.10],
post = [t + 0.10, t + 0.35] s, both inside the ripple's own occupancy interval
(so no move and no press in between), and no other ripple within 50 ms of
either window.

**Roles** (from the block's visit sequence, which continues across traversals
within a block): now; fut1-3 = the next three distinct squares; past1-3 = the
previous three; goal0 = the reward being sought. A role equal to `now`, or not
visited in the reference, is undefined. FUT = mean over the defined fut1-3, and
PAST likewise.

**Statistic.** E_w(L) = x_w · R(L), where x_w is the per-feature z-scored
window (spike count, or mean HFB). Δq = q(post) − q(pre).
- **I = Δ(FUT − NOW)**: became more future-like than current-like.
- **J = Δ(FUT − PAST)**: towards the future, not just away from now.

**Control (mandatory).** Every ordinary still moment, tiled every 50 ms with the
same geometry, with no ripple within 0.25 s of either window. Each ripple is
compared with the mean of the control events in its stratum: phase (first
traversal / rest of encoding / execution) × reward square or not ×
time-since-arrival bin × time-to-departure bin, edges 0.35, 0.5, 0.75, 1.0,
1.5, 2.5 s. This removes the drift of a still period towards the next move,
which would otherwise look like a future shift at every moment.

**Inference.** Per session, z against 1000 permutations of the nine square
labels. The same permutation is applied to ripple and control terms, and row 0
is the identity, so observed and null go through one function. Across sessions:
one-sample t and 10,000 sign flips, two-sided. Minimum 2 features and 20
matched ripples. Subject-level averaging is reported as a check.

**PRIMARY.** I, ripple − matched control, all phases pooled, `block`
references. Two primaries: **mPFC units** and **MedialFrontal HFB**
(mPFC + mOFC contacts with |x| <= 20 mm). Bonferroni over the two:
α = 0.025.

**Gate.** The positive control is evidence for `now` in the control windows,
which must be > 0. If an ROI fails it, its references carry no location
information and a null there is uninterpretable.

**Pre-declared secondaries** (reported, not corrected into the primary):
1. J, ripple − control: needed before any positive I can be read as 'future'.
   A ripple that only DEGRADES the current code raises I but not J.
2. The role profile Δ for past3 … fut3 and goal0. Components are shown
   separately: ripple, matched control, difference.
3. Phase split (first traversal / rest of encoding / execution), count-matched
   (1000 draws to the common minimum), because z carries √n.
4. `near4` references.
5. Other ROIs: HC_all and mOFC units; Visual HFB (confound detector); HPC HFB
   (contaminated by the ripple itself, descriptive only).
6. Per-unit contributions to I (cell level, descriptive; cells within a session
   are not independent).
7. Gain diagnostic: the ripple-evoked mean change of the features. A uniform
   rate change projects onto R(fut) − R(now) whenever the current square is
   special, e.g. a reward square. J is protected against this; I is not.

**Known limits, stated in advance.**
- Encode is almost always ONE traversal per block (1049 of 1441 blocks), so
  encode ripples are necessarily scored with references from the block's later,
  mostly execution, traversals.
- Permuted labels that land on an unvisited square contribute 0, which narrows
  the null slightly (~7.4 of 9 squares are covered). This scales each session's
  z by a similar factor and does not change the sign of anything.
- Only ~22% of ripples have room for both windows on one square
  (17,117 of 76,898 at 250 ms).

### RESULT (same day) — null, and both frontal signals fail the gate

`swr/future_shift_2026-10-07/` (per_session.csv, per_unit.csv,
event_counts.csv, summary.csv, results_overview.json, settings.json);
Figure 38 (`ripple_content_figures_2026-10-07/fig38_future_shift.{jpeg,pdf}`).
Validated first on synthetic place cells (scratch, not kept). A planted
post-ripple switch to fut1 gave I ≈ +4.5 z and J ≈ +2 to +3; with nothing
planted, I and J were ~0 (session SD ~1); gate ≈ +5.

Events: 59 sessions; 10,028 eligible ripples (median 96 per session) of
95,456 in task (the other-ripple exclusion costs more than the 22% estimate
implied); 402,628 control moments.

| | n | I (ripple − matched) | gate: current square in still moments | J |
|---|---|---|---|---|
| **mPFC units** | 25 | **−0.04 z, t = −0.18, P = .86** | +0.44, P = .14 ✗ | +0.01, P = .96 |
| **medial frontal HFB** | 42 | **+0.07 z, t = +0.57, P = .57** | −0.01, P = .95 ✗ | +0.10, P = .53 |
| HC units | 50 | +0.05, P = .67 | **+0.64, t = +4.10, P = .00016** ✓ | +0.03, P = .84 |
| mOFC units | 21 | +0.05, P = .79 | +0.37, P = .09 | +0.23, P = .41 |
| visual HFB | 30 | −0.18, P = .27 | +0.30, P = .13 | +0.10, P = .62 |

- **Both primaries null, and both are UNINTERPRETABLE by the pre-registered
  rule.** Neither mPFC units nor medial frontal HFB carry the current square
  against same-block references in ordinary still moments, so 'more like the
  future square than the current one' has nothing to measure.
- **HC passes the gate and still shows no shift** (I +0.05, J +0.03). The
  estimator works on real data, and in hippocampus ripples do not move the
  pattern along the route either.
- 2 of 144 ripple − control cells reach P < .05 (≈7 expected by chance).
  Nothing in the phase split (count-matched; mPFC has only 5 sessions with
  >= 20 ripples in all three phases), `near4` matches `block`, per-unit
  contributions are null (mPFC 136 units, P = .45), and there is no
  ripple-evoked gain change (mPFC −0.026, P = .22).
- **What this says about SK's premise.** The test needed a within-block,
  lag-agnostic reference to exist in mPFC. It does not, at least in 250 ms
  still windows. Together with 2026-10-02 (no cross-configuration transfer)
  and the plan-reinstatement positive control (no layout-specific pattern at a
  matched square), mPFC units here show no DETECTABLE square-indexed pattern
  that survives cross-validation in real time, within or across layouts. The
  per-lag result lives in time-WARPED space, so the next step is that space:
  references indexed by warped phase rather than by occupied square.

## 2026-10-07 — the chain, link by link: uncover → mOFC → ripple → mPFC. Null at every link

`swr_hfb_ripple_coupling.py timeline` (new command; loaders factored into
`load_inputs` / `session_data` / `session_ripples`, shared with `run`).
Output `swr/hfb_ripple_coupling_timeline_2026-10-07/`: `timeline_ft_all.png`,
`timeline_ft_reward.png`, `chain.png`, csv + json. ~6.5 min extraction, cached
in `timeline_rows.pkl` (`--reuse=True` redraws only).

**Timeline figures.** Single trials sorted by event → ripple latency, in two
alignments (ripple-locked with the event marked; event-locked with the first
ripple marked, and no-ripple events below). Raw z-scored HFB, every row
baselined −750…−250 ms before its EVENT, no shifted null. First traversal:
mOFC 18 sessions / 7318 events, mPFC 35 / 19790; 57–59% of events have a
ripple within 2 s; median event → ripple latency 0.39 s (all events), 0.72 s
(correct uncovers). Nothing is visible at single-trial level: the noise
(±0.5 z) is ~50× the known ripple-locked effects (~0.01 z). Neither ROI has an
event-locked response to correct uncovers -- consistent with 2026-09-29 (h)
("medial frontal has no event-locked reward response in any window"), so not a
bug in this extraction.

**Tests** (pre-specified by SK before running: link 1 = mOFC, link 2 = mPFC,
both on correct uncovers, uncorrected; everything else exploratory BH). Only
ripples 0.5–2 s after the event count, so the ripple's own rise never enters
the 0–500 ms response window.

| link | test | n | effect | p |
|---|---|---:|---|---:|
| 1 | mOFC 0–500 ms response predicts a ripple (AUC − 0.5) | 18 | +0.002 ± 0.035 | 0.95 |
| 2 | mPFC 0–250 ms after the ripple, minus no-ripple events at a matched pseudo-latency | 35 | −0.002 ± 0.011 | 0.88 |
| chain | mOFC response × mPFC after the ripple (Δ Fisher z, ripple − no-ripple events) | 8 | −0.14 ± 0.12 | 0.28 |

No exploratory cell (incorrect uncovers, navigation presses, the other ROI)
reaches p < 0.19; all q = 0.87.

**Reading.**
- Link 1 has nothing to work with: mOFC does not respond to the uncover, so its
  response cannot predict anything. lOFC is the one frontal ROI with an
  event-locked reward response (+0.0175, p = 0.047; 2026-09-29 (h)); it is the
  candidate sender, not mOFC. Not yet run.
- Link 2 is underpowered rather than negative: SEM 0.011 against a known
  ripple-locked mPFC effect of ~0.007 z (shifted-null estimate over all
  ripples). Restricting to first-traversal correct uncovers keeps 1750 of the
  ~90k ripples.
- The chain test rests on 8 sessions.

## 2026-10-06 — Fig 3d gradient brain: cells at exact positions, no jitter

`scripts/cell_fMRI_angle_match.py`, brain panel only. The old render placed
cells as 3D spheres **jittered ±3 mm on every axis**, under a perspective
camera, behind an opaque backdrop on a 25%-transparent cortex. So the sites
did not look like Fig 2c, and cells lateral of the medial wall were hidden by
the gradient.

**Now.** The fsaverage medial surface is rendered with an orthographic camera
along x (see-through cortex at alpha 0.25 as in panel b, gradient backdrop at
alpha 0.65). The cells, the split axis (white) and the median split (black)
are drawn on top by matplotlib in MNI mm, so each cell sits at its exact
(y, z). The panel is drawn at print size (2.1 × 3.0 cm per hemisphere):
cells are 4.0 pt across with a 0.25 pt black rim and a 0.4 pt white ring. At
the earlier 4.2 cm size, shrunk into the figure, the cells read as black or
white blobs. That is the same
projection as the sagittal glass-brain views of Fig 2c. Test spheres rendered
at known (y, z) land within **0.07 mm** (sub-pixel) in both hemispheres. A
side-by-side check against a Fig 2c-style glass brain of the same 158 cells
shows the same site constellation. Co-located cells (one bundle) are drawn as
one marker. Brains are now drawn only for `pc1_ventral_dorsal`
(`BRAIN_SCHEMES`); the `_bars` duplicates and per-image colourbars are gone.

Numbers unchanged: all five CSVs are byte-identical to `final_splits/`.
Output: `cell_gradient_master/2026-08-28_15-19-35/final_splits_2026-10-06/`.
Style sweep (backdrop alpha 1.0 / 0.65 / 0.4 × cell style edge / halo / sphere)
plus the glass-brain QC:
`final_splits/brain_medial_variants_2026-10-06/`.

**Fig 2c coverage panel: new `scripts/plot_cell_coverage_glassbrain.py`.**
Glass brain with right sagittal, left sagittal and axial views. One marker per
recording site at its exact canonical coordinate, with no jitter. Uses the
project ROI colours and is drawn at print size (5.0 × 4.5 cm, 3.2 pt markers
with a darker rim, glass-brain outlines thinned ×0.5). It covers 924 cells,
206 sites and 62 sessions: EC 35/10/8, HC_anterior 295/71/52, HC_mid 233/29/36,
mPFC 158/43/33, mOFC 142/43/27, PCC 61/10/10 (cells/sites/sessions). Output:
`group/cell_coverage_glassbrain/2026-10-06_18-04-25/`. The older sandbox
`roi_brain_visualization.py` spreads co-located cells on a 0.7 mm ring and
colours HC_anterior magenta, so it is not the source of the current Fig 2c
(teal HC ant). That source was not found.

## 2026-10-02 (b) — ripple-by-ripple mOFC → mPFC coupling: null

`scripts/exploratory/swr_hfb_ripple_coupling.py`; output
`swr/hfb_ripple_coupling_2026-10-02/` (`ripple_coupling.png`, per-session and
group csv, `coupling_maps.npz`, `result.json`). New single-ripple pass over
`bundle_v2` HFB (local, ~3 min).

**The question.** SK's chain (mOFC "reward" → HC ripple → mPFC "for the plan")
predicts that on the SAME ripple, mOFC-before predicts mPFC-after. Session
averages cannot test that.

**Statistic.** Per session with both ROIs: z(ρ(mOFC pre, mPFC post)) −
z(ρ(mPFC pre, mOFC post)), Spearman across ripples, pre/post = the direction
index windows on the flank-normalised epoch; the reverse direction cancels slow
shared state. Real − mean of the 6 shifted-time nulls of the conditions run
(seed 42), so a general OFC → PFC lead that is not about ripples cancels too.
ROI epoch = mean over that ROI's different-shaft derivations clean for the
whole ±1 s. Sign-flip over sessions. PRE-SPECIFIED primary: mOFC → mPFC, first
traversal, uncorrected; everything else exploratory, BH.

**Check.** The per-session direction index from this extraction matches the
saved one for the same sessions (first traversal r = 0.95 mOFC, 0.84 mPFC).

**Result.** Primary mOFC → mPFC, first traversal: +0.019 ± 0.023 Δz,
p = 0.42 (11 sessions, 7 subjects, median 174 ripples). The components are
both ≤ 0 (mOFC→mPFC −0.014, mPFC→mOFC −0.033), so there is no ripple-locked
co-fluctuation between the two ROIs above the shifted baseline for an asymmetry
to sit on; the cross-temporal maps are noise-like. Exploratory: all ripples
−0.036 (p = 0.08) and after encoding −0.054 (p = 0.06), both in the REVERSE
direction, q = 0.54. lOFC → mPFC (15 sessions): nothing (all p ≥ 0.23).
Sensitivity: session SD ≈ 0.08 Δz, so a group effect of ~0.05 would have
been visible; weaker coupling cannot be excluded.

**Reading.** The session-average "mPFC later than mOFC" lean is not
accompanied by ripple-by-ripple transfer from mOFC to mPFC. It fits two
separate ROI responses better than a coupled chain, with 11 sessions as the
limit.

## 2026-10-02 — does mPFC HFB lock AFTER the ripple and mOFC BEFORE? Direction, not significant

`scripts/exploratory/swr_hfb_direction.py`; output `swr/hfb_direction_2026-10-02/`
(`direction_index.png`, `direction_curves.png`, `result.json`, per-session csv).

**The question.** SK, from the per-ROI trace grid: mPFC HFB looks to peak
slightly after the ripple, mOFC slightly before. Tested directly.

**Index.** Per session, mean HFB (0, +250] − mean [−250, 0) ms, on the
UNSMOOTHED real-minus-null curve (flank baseline cancels). + = later.
Within ROI: sign-flip permutation vs 0. mPFC vs mOFC: ROI-label permutation,
unpaired (35 vs 18 sessions, only 11 shared; paired-on-11 reported as a check),
Holm over the 10 ripple sets. 10,000 perms, seed 42. Partition =
`hfb_roi_partition_2026-09-28` (the exclusive partition has no mPFC/mOFC).
Sets: a) all ripples (region run); b) all LABELLED ripples per phase
(first_traversal / encoding / after_encoding; 76% of ripples carry a label;
encoding contains first traversal); c) first traversal by preceding event,
plus `still` and the − navigation contrasts.

| set | mPFC − mOFC | p_perm | Holm |
|---|---:|---:|---:|
| a) all ripples | +0.0076 | 0.084 | 0.53 |
| b) first traversal | +0.0151 | 0.030 | 0.27 |
| b) encoding | +0.0147 | 0.021 | 0.21 |
| b) after encoding | +0.0080 | 0.075 | 0.53 |
| c) correct | +0.0168 | 0.18 | 0.92 |
| c) incorrect | +0.0040 | 0.70 | 1.00 |
| c) navigation | +0.0252 | 0.057 | 0.46 |
| c) no press ±2 s (10/5 sess.) | +0.0215 | 0.35 | 1.00 |
| c) correct − nav | −0.0029 | 0.87 | 1.00 |
| c) incorrect − nav | −0.0197 | 0.26 | 1.00 |

**Reading.** The sign matches the impression in 8 of 10 sets (mPFC later than
mOFC), largest during first traversal / encoding, but nothing survives Holm and
no single ROI differs from 0 after BH (best: mPFC encoding p = 0.047, q = 0.46;
mPFC navigation p = 0.021, q = 0.30). The mOFC "before" lean is the bigger
pre-ripple bump, which is where a preceding press/feedback response lands;
mPFC vs mOFC cancels a SHARED press response, not an ROI-specific one. The
− navigation contrasts show no ROI difference, so the effect is not specific
to reward/error. Treat as a non-significant trend; do not report as a gradient.

`swr_hfb_roi_conditions.py`: loader factored into `load_partitioned` (reused
here); output unchanged.

**Addendum (d): condition x phase grid, with lOFC.** Same index and tests,
for correct / incorrect / navigation / no-press / the two − navigation
contrasts in first_traversal, encoding and after_encoding, ROIs mPFC, mOFC,
lOFC; pairs mPFC − mOFC, lOFC − mOFC, mPFC − lOFC. Uncorrected p reported,
BH q in `grid_roi_pairs.csv`. Figures `direction_phase_condition.png` and
`..._curves.png`.
- Correct uncovers, encoding: mPFC +0.002, mOFC −0.016, lOFC −0.008;
  mPFC − mOFC +0.018, p = 0.14. Same pattern as first traversal, n.s.
- Incorrect uncovers, after encoding: errors are rare once the route is known
  -- 10 / 5 / 3 sessions. mPFC − mOFC p = 0.62; lOFC untestable.
- lOFC sits with mOFC, not mPFC: lOFC − mOFC never differs (all p > 0.4).
  mPFC − lOFC is the best cell in the grid: navigation presses during
  encoding, +0.031, p = 0.006 (q = 0.26).
- 3 of the 48 tested pairs reach p < 0.05; about 2.4 are expected by chance.
  None survive BH.

## 2026-10-02 — why is the evidence negative? the template barely transfers

`swr/cross_task_2026-10-02/`; figures 36-37.

**The question.** mOFC gives ripple -0.544 (P = 0.028) but non-ripple +0.031
(P = 0.75) for the SAME template. SK's hypothesis: the same square is visited
at a different point in the sequence in different configurations, so a cell
tuned to location x lag fires differently at that square across tasks, and the
leave-one-out template ends up ANTI-correlated with the held-out layout.

**Hypothesis NOT supported, but the puzzle sharpens.** Leave-one-out
transferability of the place map (each configuration's rate map vs the mean of
the others, per cell, then across sessions):

| ROI | r | P |
|---|---|---|
| HC | +0.057 | 1.9e-5 * |
| mPFC | -0.002 | 0.93 |
| mOFC | +0.004 | 0.84 |

The PFC templates are UNINFORMATIVE, not misleading -- so an anti-correlated
template cannot be the mechanism. And since the template is IDENTICAL for
ripple and non-ripple windows, nothing at the template or configuration level
(including lag divergence) can produce a ripple/non-ripple sign flip. Whatever
does it is a property of activity DURING ripples.

**Lag divergence** (circular mean warped theta per square per configuration;
divergence = mean circular distance): mPFC pooled r = -0.089 (P = 0.034) in
the predicted direction, but within session r = -0.068 (P = 0.16). HC -0.064
(P = 0.098), mOFC n.s. Pooling mixes between- and within-session variance, so
the within-session test is the right one and it is null.
⚠ Underpowered by design: ~28 pairs per session gives SE(r) ~ 0.19, so an r of
-0.07 was never detectable this way. A mixed model over all pairs with session
as a random effect would be the better test.

**What HC's transferability IS.** Three task models over the same
configuration pairs -- occupancy (time per square), reward overlap, trajectory
(9x9 transitions) -- correlated with the neural matrices, within session then
across:

| ROI | occupancy | reward overlap | trajectory |
|---|---|---|---|
| HC | +0.113 (P = 8e-4) * | +0.077 (P = 0.025) * | +0.129 (P = 0.001) * |
| mPFC | +0.067 | +0.076 | +0.052 (all n.s.) |
| mOFC | -0.050 | +0.011 | +0.061 (all n.s.) |

HC's cross-task similarity IS task similarity. mPFC and mOFC track NONE of the
three despite having matrices of comparable magnitude -- their structure is not
task structure.
⚠ The three models intercorrelate, so HC's result is ONE effect seen three
ways. ⚠ The trajectory model is nearly uniform (grid adjacency constrains all
layouts to share transitions), so it has little dynamic range; occupancy is the
only model with real variance. A partial regression is needed to say which HC
actually tracks.

**Ripple vs non-ripple transferability.** HC +0.0025 (P = 0.62) in ripple time
vs +0.0173 (P = 0.0020) outside. Directionally a dissociation, but both are an
order of magnitude below the pooled +0.057 because restricting to ripple time
leaves very little data per square per configuration -- the ripple column is
too attenuated to interpret.

### Figure bug fixed
fig36 panel b drew the pooled point with a 95% CI and the within-session point
with a 1-SEM bar, making the latter look far tighter than it is. Both are now
95% CIs, which is why a bar that appeared clear of zero had P = 0.16.

### Still unexplained
The mPFC flip at already-known correct uncovers (+0.471 outside ripples vs
-0.600 inside, difference -1.071, P = 0.0011) is not accounted for by template
transferability in either window (both ~0). ⚠ There is also an unresolved
tension: if mPFC's template is uninformative, it is not obvious how the
non-ripple value reaches +0.471. The population dot product and the per-cell
map correlation are different measures -- a minority of well-transferring cells
could carry the population match while the mean per-cell correlation sits at
zero. Not yet checked; this is the next thing to look at.

## 2026-10-01 — PRE-REGISTRATION: within-session plan reinstatement in cells (loading analogue)

Written BEFORE running. Module `mc/analyse/plan_reinstatement.py`, runner
`scripts/cells_plan_reinstatement.py`.

**Why a new design.** The fMRI loading RSA cannot be ported: there is no
instruction phase in sEEG, and the 8 shared configurations put 8 different
locations at every state, so the position-locked plan RDM has zero variance
(see 2026-09-15 (c)). `ripple_rsa` (cross-session pseudo-population) was
underpowered (mPFC noise ceiling 0.00). This test needs NO shared layouts and NO
cross-session pooling: it runs within each session, so all sessions with >=2
mPFC cells enter.

**Estimator.** Per session and ROI, spike counts in [t+0.05, t+1.0] s after every
correct uncover, z-scored per cell over all of them. Template T(config, L) =
mean z-vector over EXECUTE-phase correct uncovers at location L in that
configuration (`grid_id`), always excluding the test event's own block
(`grid_num`). For a test event at L in configuration c:

    Δ = [x·T(c, L) − mean_{c' ≠ c, L rewarded in c'} x·T(c', L)] / n_cells

Same location, same event type (correct uncover) in both terms; only the layout,
and hence the future, differs. Blocks of different configurations are
interleaved in time, so drift does not favour 'same'.

**Primary test (mPFC).** Test events = the FIRST discovery of each state in each
block (encode). The number of future rewards that are known at discovery is
0/1/2/3 for A/B/C/D, so a plan predicts Δ_D > Δ_A. Confounds that are constant
across states (state coding, reward response, location role) cancel in D − A.
Per-session D − A, one-sample t across sessions, p from 10,000 sign flips of the
per-session values (same function for empirical and flipped), one-sided.

**Pre-declared secondaries** (reported, not corrected into the primary):
1. Positive control: Δ for held-out EXECUTE uncovers > 0. If this fails the
   approach has no power and a null on the primary is uninterpretable.
2. Heading-free trend A < B < C (per-session slope over A, B, C). Needed because
   execution pauses after a correct uncover are short (median 0.28 s), so the window
   includes heading to the next goal, and only at D is the next goal (A) known.
3. First run of each configuration only (later runs may be recalled).
4. Same tests in HC_all, mOFC, PCC (PCC = confound detector).

**Caveat fixed in advance.** At D the known set equals the whole plan, so this
cannot separate plan from memory the way the fMRI forward/backward design does.
A positive result means "layout-specific execution pattern is reinstated as
rewards are discovered", not "plan, not memory".

### RESULT (same day) — null in mPFC; positive control fails in mPFC, works in HC
Results in `group/plan_reinstatement_2026-10-01/` (summary.csv, results.json,
plan_reinstatement.jpeg). n = sessions; one-sided sign-flip P.

| ROI | n | positive control (execute Δ) | PRIMARY D − A | slope A→C | first run D − A |
|---|---|---|---|---|---|
| mPFC | 26 | +0.0008, t = +0.28, P = .39 | −0.0086, t = −0.84, P = .79 | t = +0.93, P = .19 | t = −1.37, P = .91 |
| HC_all | 51 | **+0.0049, t = +3.25, P = .0003** | t = −0.54, P = .71 | t = +0.59, P = .27 | t = −1.19, P = .87 |
| mOFC | 22 | t = +1.50, P = .076 | t = −1.46, P = .92 | t = −1.25, P = .89 | t = −0.72, P = .76 |
| PCC | 10 | t = +1.69, P = .059 | t = −0.27, P = .60 | t = −0.72, P = .75 | t = −0.64, P = .72 |

Subject-level averaging (41 patients) changes nothing (mPFC D − A subj P = .82;
HC control subj P = .0001).

**Reading.** The primary test is null, but in mPFC it is UNINTERPRETABLE: mPFC
shows no layout-specific execution pattern at a matched square even during
execution, so there is nothing that could be reinstated. HC shows that pattern
(so the estimator has power) but it does not build up over discovery. This
readout is narrower than the published mPFC RSA (1 s after reward uncovers at
the 4 reward squares only, vs. the whole loop in 12 phase bins over a
pseudo-population), so it does not contradict that result.

### ⚠ Deviation from the pre-registration: block-distance bias
The pre-registered estimator (one template per layout, own block excluded) gave
a NEGATIVE positive control in every ROI (mPFC t = −3.93). Cause: similarity
falls steeply with block distance (mPFC +0.066 at 1 block → −0.044 at >12),
and repeats of one layout are spaced further apart than other layouts (mean
10.2 vs 8.1 blocks). Fixed by per-block templates and
`dot ~ same + |offset| + |offset|² + offset` with an event fixed effect;
Δ = coefficient on `same`. Tests, windows, events and inference unchanged.
Set-aside versions kept for the record:
- `plan_reinstatement_2026-10-01_v1_unmatched_biased/`: the pre-registered
  estimator, biased negative.
- `..._v2_exact_distance_match/`: same-layout blocks compared only with
  other-layout blocks at identical distance. Unbiased but kept only 24% of
  events, and the first-run test fell to 1–8 sessions. Its conclusions match:
  HC control P = .018, mPFC control P = .13, mPFC D − A P = .74.

### ⚠ TIMING BUG in the table above — events were locked to ARRIVAL, not reward
On an uncover row of the step table (`swr_location`), `t_s` is when the subject
ARRIVED on the square. The press, and the reward (`t_A`..`t_D`, to within 10 ms),
is the NEXT row's `t_s`: a median 0.25 / 0.24 / 0.33 / 0.50 s later in s05 /
s38 / s50 / s12. The results above therefore used windows starting at arrival.
Fixed in `uncover_table`; the spike result is re-run below and the old folder
kept as `plan_reinstatement_2026-10-01_v3_arrival_time`.
⚠ Same convention in other modules that use `st.t_s[is_uncover]` as "the
uncover" (e.g. `swr_explore/consolidation.py`, `intention_ripple.py`). NOT
changed here; they need checking.

### PRE-REGISTRATION: LFP version (written before running)
Same estimator, events, windows, tests and inference. Features = one per bipolar
derivation (mean signal in [t+0.05, t+1.0] s after the reward, z-scored per
derivation; artifact-touching windows → 0), in place of one per cell. Clocks
checked: step-table presses match `swr_behaviour.uncover_events` to ~10 ms.
- **ROI.** `MedialFrontal` = mPFC + mOFC derivations with |MNI x| <= 20 mm, the
  mask of the HFB ripple-locking analysis. Lateral OFC is NOT pooled in (40% of
  'mOFC' contacts sit beyond |x| = 25 mm). HPC = comparison region, Visual =
  confound detector.
- **Signals.** hfb (70-150 Hz), theta (4-8 Hz), beta (13-30 Hz), ripple band
  (80-120 Hz) envelopes, all as stored in `bundle_v2/hfb`. Four signals means
  four primaries: MedialFrontal D − A is Bonferroni-judged at P < 0.0125.
- Positive control must pass before a null on the primary is read.

### RESULT: spikes (timing fixed) + LFP — null throughout
Folders `group/plan_reinstatement_{spikes,hfb,theta,beta,ripple}_2026-10-01/`.
One-sided sign-flip P, n = sessions. Contacts: MedialFrontal 179 derivations in
53 sessions (44 with >= 2), HPC 181 / 61 (53), Visual 168 / 36 (30).

| signal | ROI | positive control (execute) | PRIMARY D − A | slope A→C |
|---|---|---|---|---|
| spikes | mPFC (26) | t = −0.21, P = .58 | t = +0.51, P = .31 | t = −0.33, P = .62 |
| spikes | HC (51) | **t = +3.13, P = .0010** | t = +1.17, P = .12 | t = +0.33, P = .38 |
| hfb | MedialFrontal (44) | t = −0.17, P = .55 | t = −0.23, P = .59 | t = −1.10, P = .86 |
| theta | MedialFrontal | t = +0.37, P = .36 | t = −1.23, P = .89 | t = −0.89, P = .80 |
| beta | MedialFrontal | t = −2.24, P = .99 | t = −1.16, P = .87 | t = −0.03, P = .52 |
| ripple band | MedialFrontal | t = +0.12, P = .45 | t = +1.63, P = .058 | t = +0.17, P = .44 |

- The timing fix does not change the spike conclusion. HC still passes the
  positive control; mPFC has no layout-specific execution pattern to reinstate.
- **No LFP signal passes the positive control in MedialFrontal.** In HPC only
  the ripple band does, weakly (t = +1.88, P = .036); HC spikes do (P = .001).
  So the contact-level LFP readout does not carry the layout-specific pattern
  that HC cells do.
- No primary reaches the Bonferroni threshold of P < .0125. The one nominal
  secondary hit is MedialFrontal ripple band, first run only, D − A:
  t = +2.43, P = .0095, subject P = .017. It is one of ~90 secondary tests,
  its signal's positive control fails, and the 80-120 Hz band overlaps HFB,
  which is flat. ⚠ Do not read it as a finding.

**Band-specificity figure** (`group/plan_reinstatement_figures_2026-10-01/`,
`scripts/plan_reinstatement_figures.py`). The first-run ripple-band effect is
not a separate 80-120 Hz phenomenon: per-session first-run D − A in ripple band
and in HFB correlate r = +0.59 (n = 44, P = 2e-5), and HFB has the same shape
across A-D (rise at D), only smaller (+0.006 vs +0.016 at D). It is also not
region-specific: HPC (t = +1.01) and Visual (t = +1.12) point the same way,
weaker. Positive control flat for every signal.

### PRE-REGISTRATION: is the D effect heading? is it ripple-locked? (before running)
Target = the first-run MedialFrontal D − A (ripple band, also HFB). Estimator,
events and inference unchanged; only the samples entering each window change.
Pause from reward to next move: discoveries median 1.2-1.7 s (D: 1.38 s),
execution median 0.40 s (10th pct 0.24 s). So movement entered mainly the
execution TEMPLATES, not the discovery windows.
1. **premove**: window [t+0.05, min(t+1.0, first move after the reward)];
   events with < 0.15 s left are dropped. Spikes become rates. Run for ripple
   band, HFB, spikes. Heading predicts that D − A shrinks.
2. **ripmask**: drop samples within ±0.2 s of every hippocampal ripple peak
   (`bundle_v2/ripples.csv`, deduplicated at 50 ms) before averaging the
   window. A window needs >= 5 samples (50 ms) left. **ripshift** = control: the
   same mask with each session's ripple train shifted by a random ±2-5 s
   (5 shifts, seed 42, per-session values averaged over shifts). Ripple-locked
   predicts that ripmask loses the effect and ripshift keeps it.

### RESULT: not ripple-locked; partly lost before the first move
Figure + stats `group/plan_reinstatement_controls_2026-10-01/`
(`scripts/plan_reinstatement_control_figures.py`). The refactored code
reproduces the earlier ripple-band numbers exactly (max |Δt| = 0).
MedialFrontal ripple band, first-run D − A (n = 44):

| variant | mean | t | P (one-sided) |
|---|---|---|---|
| full | +0.0207 | +2.43 | .0095 |
| premove | +0.0119 | +1.46 | .078 |
| ripmask | +0.0206 | +1.86 | .033 |
| ripshift (5-shift mean) | +0.0150 | +2.26 | .015 |
| ripmask − ripshift | +0.0055 | +0.60 | P2 = .55 |

- **Not ripple-locked.** Dropping every sample within ±0.2 s of a hippocampal
  ripple leaves the mean unchanged (+0.0206 vs +0.0207); only the noise grows,
  as it does with the shifted mask. The mask removes ~17-21% of each window and
  is lowest at D (16.8%), so D is not ripple-rich either.
  Why 80-120 Hz beats HFB is therefore NOT ripple-locked activity.
- **Premove** roughly halves the effect (+0.0207 → +0.0119, P = .078); HFB
  premove goes to 0. Ambiguous: premove also shortens the execution templates
  to a median ~0.35 s (noisier), and it truncates mainly the discoveries where
  the subject moves off early. Heading is neither excluded nor shown.
- mPFC spikes: null either way (premove t = 0.00).

**Conclusion.** The fMRI loading effect has no detectable analogue in this
readout, in cells or LFP. The binding limit is upstream of loading: apart from
HC cells, none of these signals shows a layout-specific pattern at a matched
square even during execution.

**Do not re-run as-is hoping for mPFC.** Next options if wanted: the same test
on mPFC HFB across contacts (more sessions), or a readout over the whole loop
(all squares, phase bins) instead of reward uncovers only.

## 2026-10-01 — PFC ripple content: tests (a)(b)(c), loading, and lag tagging

Results in `swr/{phase_contrast,goal_lag,warped_lag,loading,discovery_lag,
intention_ripple,consolidation,discovery_locked,method_check,
preferred_lag_firing,lag_group_ripple,storage_vs_planning}_2026-*`.
Figures 20-33 in `swr/ripple_content_figures_2026-10-01/`.

### THE METHOD IS SOUND -- validated
`method_check` (fig31): the SAME estimator on tiled NON-ripple windows
(>=1 s from any ripple), width and count matched to ripples.

| | non-ripple | ripple |
|---|---|---|
| HC encode | +0.263 (P = 5.8e-4) | +0.558 (P = 2.0e-4) |
| HC execute | **+0.145 (P = 0.043)** | +0.202 (P = 0.16) |

**The published HC current-location result is recovered.** Every earlier
statement that "current location is not encoded in execute" was about RIPPLE
WINDOWS ONLY and was repeatedly phrased as if general -- corrected.

### (a) current location in PFC
mPFC null (+0.31, P = 0.13). mOFC significantly NEGATIVE (-0.49, P = 0.041),
and `method_check` localises it: mOFC non-ripple +0.031 (P = 0.75) vs ripple
-0.544 (P = 0.028), difference **-0.575, P = 0.037**. So mOFC's anti-location
pattern is RIPPLE-SPECIFIC and confined to encode. Three analyses now point at
it. Unexplained; the strongest ripple-specific effect in the project.

### (b) goal sequence -- null
3-way split + count matching dissolves it (~6 of 120 tests at P<0.05, chance).
⚠ Framing error found by SK: the cycle has length 4, so '+3 ahead' IS '1 back'
-- `goal3` was never a distant-future goal, it is the reward just collected.
⚠ Knowability: on the first traversal goal0 is known 7.3% of the time, goal1
25.3%, goal2 49.3%, goal3 75.5%; by state, goal3 is 0% knowable at state 1 and
100% at states 2-4. Any goal-lag test must condition on this.
⚠ fig25's encode panel is 82% first-traversal, so ~76% of its ripples score a
square the subject cannot know.

### (c) warped 12-lag -- a-priori lag hits but reverses
mPFC 60 deg ripple-flank, first traversal +0.428 (P = 0.0099), held out of the
per-lag fit (`TRIALS='all_minus_explore'` excludes 99.9% of first traversals
and 100% of encode-rest). BUT the other held-out phase goes the other way:
paired first_traversal - encode_rest = +0.936, t(16) = +3.30, P = 0.0045.
16/216 tests significant vs ~11 expected; nothing survives Bonferroni.

### L1 loading -- null; the sham control earned its keep
Nothing positive. The one nominal hit (mPFC/goal0/execute -0.498, P = 0.030) is
matched by its SHAM (-0.391, P = 0.057): within-subgoal drift, not ripples.

### mPFC HOLDS A NEWLY DISCOVERED REWARD -- the one clean positive
`consolidation` (fig29). Window t+50 ms to t+1 s, target = the square just
uncovered:

| | first discovery | repeat correct | incorrect |
|---|---|---|---|
| mPFC | **+0.770 (P = 5.6e-4)** | +0.000 | -0.027 |
| HC | +0.54 | +0.33 | +0.62 |

The subject stands on that square in ALL THREE, so current-location coding
predicts all three equal -- and HC behaves exactly that way, which is an
internal positive control. mPFC responds only when the reward is NEW.
Paired: first discovery - repeat correct +0.770, P = 0.0035.
⚠ NOT ripple-delivered: weaker WITH a ripple (repeat correct -1.071,
P = 0.0011), and `discovery_locked` (fig30) finds no modulation around ripples
-- though that analysis has ~104 events/session and is underpowered (HC's known
effect also fails there; sqrt(n) predicts +0.375, observed +0.246).
⚠ CONFOUND: a first discovery is usually also the first VISIT to that square.
Needs a novel-place-without-reward control.

### Intention: HC yes, mPFC no
`intention_ripple`. Before an uncover, HC represents the square about to be
uncovered (+0.591, P = 1.3e-4; STRICT, excluding the current square and the
true goal, +0.319, P = 0.020). mPFC is null even unrestricted (-0.069,
P = 0.75) -- which retrospectively guts the rho = -0.013 correlation reported
earlier: it correlated ripple counts with a quantity that is itself zero.
⚠ STRICT does not control spatial PROXIMITY; the next square is usually
adjacent.

### Lag tagging IS real -- but only WHERE, not WHEN
Continuous `r_lag000_noctrl` predicts how selectively a cell fires at its
cross-validated preferred square, during execute, in all three ROIs:
HC r = +0.230 (P = 1.1e-7, n = 520), mPFC +0.223 (P = 0.0063, n = 148),
mOFC +0.182 (P = 0.031, n = 141). Encode and ripple windows null.
`storage_vs_planning`: lag tuning does NOT predict engagement at storage vs
planning anywhere (18 tests, all n.s.). ⚠ the planning events may be wrong --
all four subgoal onsets were used, but a plan is plausibly assembled only at
the FIRST onset of a traversal.

### Bugs found and fixed (all produced plausible-looking output)
1. **Zero-count artefact.** Per-fold selectivity pinned to -1 whenever a
   held-out configuration had no ripple spikes at the preferred square -- 56%
   of folds -- giving a spurious -0.31. Rebuilt as a count model pooling
   spikes and dwell across folds before any ratio.
2. **`mod` shadowed `DataFrame.mod`**; attribute access returned the method.
   Column renamed `ripmod`.
3. **`cond_map` wants 0-based labels**; passing locations 1-9 made index 0 NaN,
   every window non-finite, and the first `goal_lag` run returned zero rows
   silently.
4. **Shared RNG in `summarise`** made each curve's cluster p depend on which
   curves ran before it (0.048 then 0.051 on identical data). Seeded per
   (ROI, curve).
5. **`suff` used but never defined** in the phase panels -- a NameError that
   killed the figure job silently, so "regenerated with matched levels" was
   false for one round.

### Cell matching between pipelines (verified, not assumed)
`per_cell_ALL_ROIs.csv` (subject_id, cell_idx) -> ripple pipeline
(session, cell) via **cell_idx - 1**. ELECTRODE names agree for 97.9% of 955
cells; the 20 exceptions are a parsing artefact of anatomical labels containing
hyphens. 64 ROI disagreements are relabelling (58 flagged `roi_changed`), not
misalignment. mPFC: 148 of 155 cells enter (2 of 63 sessions have no ripples).

⚠ Preferred lag by argmax is near-uniform (64-94 cells per bin, 80 by chance)
and discards ~94% of cells to form the lag-0 group. Use the CONTINUOUS
strength instead; it recovers the effect in mPFC where the split could not.

## 2026-09-30 (b) — goal-lag 3-way split: null on the target, coherent in the confusion

`swr/goal_lag_2026-09-30/` (`per_session.csv`, `summary.csv`, `confusion.csv`).
Phases are now mutually exclusive: first_traversal / encode_rest (= encode
minus first traversal) / execute, all count-matched to the common minimum with
1000 draws, levels and pairwise contrasts from the same subsamples.

**⚠ This substantially weakens the earlier 2-way result.**

| ROI | effect | raw | matched |
|---|---|---|---|
| mPFC | lag 0, first traversal | +0.476, P = 0.026 | +0.216, P = 0.19 |
| HC | lag 2, first traversal | -0.527, P = 0.001 | -0.253, P = 0.047 |
| HC | lag 2, rest of encode | - | -0.155, P = 0.42 |

The lag-2 effect reported yesterday across all of "encode" (-0.52, P = 0.002)
was driven almost entirely by the FIRST TRAVERSAL and halves once counts are
matched. ~6 of 120 tests reach P < 0.05, which is the chance expectation. On
the target-only tests **(b) is null**.

**The confusion matrices are the informative part** (fig23). Role-collapsed:
rows = which lag's template, columns = the role the scored square plays. Rows
sum to ~0 by construction, so this shows the DISTRIBUTION of evidence.

HC, first traversal: evidence lands on **current** (+0.065*, +0.069*) and
**goal 3, the just-collected reward** (+0.093*) -- never on a future goal. That
is the manuscript's HC claim ("current location along with a trace of the
recent past") recovered from an independent analysis, and it explains the lag-2
negative: evidence concentrates on current/goal3, so the most distant future
goal absorbs the negative.

mPFC, first traversal: one starred cell, **goal 0 under the lag-0 template**
(+0.090*) -- the predicted one, and the only positive in mPFC.

mOFC, first traversal: the largest cells are the **current** column at lags 1
and 2, but not significant. If real this would conflict with the significant
NEGATIVE current-location result in test (a) -- unresolved, do not report
either without resolving it.

⚠ 72 cells per panel-row at P < 0.05 uncorrected, so 3-4 stars per row are
expected by chance. The HC pattern is worth something because it is COHERENT
(stars in the roles the hypothesis names, none elsewhere), not because any
single cell is significant.

### Retracted
The claim that the HC lag-2 negative is "the arithmetic shadow of the
current-location code" was tested and FAILED: across sessions, current-location
z does not predict lag-2 z (r = -0.086, P = 0.55 for HC_all; r = -0.245,
P = 0.20 for HC_mid). The confusion matrices now offer a better account (the
evidence goes to current + just-collected), but the shadow claim as stated was
unsupported and has been withdrawn from the figure and text.

### Per-phase levels are now count-matched
`swr_phase_contrast` previously recorded per-phase z from ALL ripples in each
phase while only the contrast used matched draws. Since z carries a sqrt(n)
term and execute has ~2x the ripples, every phase panel drawn before this
understated the gap. Now records `z_<phase>_matched` from the same draws as the
contrast; figures prefer it automatically.

## 2026-09-30 — PFC ripple content: tests (a) and (b)

### (a) current location in PFC — `swr_phase_contrast`, ROI_SETS extended
`swr/phase_contrast_2026-09-30/`

| ROI | encode | P | ripple - flank (enc) | P |
|---|---|---|---|---|
| HC pooled | +0.57 | 2.0e-4 | +0.23 | 0.115 |
| mPFC | +0.31 | 0.134 | +0.04 | 0.850 |
| mOFC | **-0.49** | **0.041** | -0.30 | 0.134 |

mPFC null for current location, as the manuscript predicts -- but numerically
POSITIVE, 95% CI [-0.10, +0.73], so this is failure-to-detect, not evidence of
absence. A dissociation claim needs the HC-vs-mPFC difference tested directly.

mOFC significantly NEGATIVE (and ripple-flank during execute -0.46, P = 0.025).
Replicates the -0.49, P = 0.056 seen earlier in the all-ripples time course.

### (b) goal-sequence content — `mc/analyse/swr_explore/goal_lag.py`
`swr/goal_lag_2026-09-30/`. goal_lag[g] = rew[(state-1+g) mod 4]; g=0 is the
reward being sought now, g=3 the one just collected.

| ROI | lag | test | n | z | P |
|---|---|---|---|---|---|
| mPFC | 0 | encode | 26 | **+0.386** | 0.075 |
| mPFC | 1 | encode | 26 | -0.314 | 0.177 |
| HC_mid | 2 | encode | 29 | **-0.626** | **0.0084** |
| HC_mid | 2 | ripple - flank, enc | 29 | **-0.677** | **0.0022** |
| HC_all | 2 | encode | 51 | -0.522 | 0.0021 |
| mOFC | any | any | 22 | all null | >= 0.05 |

**⚠ The HC lag-2 negative is NOT an independent finding.** Templates are
z-scored to sum to zero across the nine locations, so sum_L E(L) = 0 exactly
and the statistic is (9/8) * E(target). Coding ANY one location therefore
forces every other location's score negative. Measured at ripple times during
encoding, lag 2 is the spatially furthest from the current square (mean grid
distance 1.88, identity 7.3% -- below the 11% chance), so it carries the
largest negative shadow of HC's current-location code. Its ripple-specificity
follows for the same reason: the current-location code is itself stronger in
ripples.

**mPFC lag 0 is NOT current-location leakage.** The reward being sought
coincides with the current square only 9.4% of the time at ripple times (chance
= 11%; mean grid distance 1.81). So the +0.386 trend is about where the subject
is heading, independent of where they are. At P = 0.075 it is a trend, not a
result.

Overlap table (ripple times, encoding): identity / mean grid distance by lag --
lag0 9.4% / 1.81, lag1 10.1% / 1.82, lag2 7.3% / 1.88, lag3 28.8% / 1.34.
Lag 3 (the reward just collected) is often still underfoot, so it is the one
lag that IS contaminated by current location.

### Bug fixed
First run of (b) returned zero rows: locations 1-9 were passed to `cond_map`,
which wants 0-based labels, so index 0 was always NaN, every window was
non-finite and everything was dropped. `cond_map` silently produces a NaN
column rather than erroring -- worth knowing.

### New: `mc/analyse/swr_warp.py`
The real-time <-> warped-theta bridge for test (c). theta(t) from `state` +
`t_A..t_D`, the same boundary set as `save_humanABCDneurons_normalised.m`.
Smoke-tested: 100% of steps get a finite theta.

Two measurements that constrain (c):
- one physical step spans ~18 deg of warped angle (p90 = 47.5 deg), so 30 deg
  is ~1.7 steps;
- P(location at lag g == location at lag 0): 30 deg 32.8%, 60 deg 12.3%,
  90 deg 6.4%, 180 deg 7.7% (chance 11%). So 60 deg and beyond are clean, but
  **30 deg is a third redundant with the current location** -- a positive there
  alone could be current-location bleed. The a-priori mPFC window is 30-60 deg,
  so 60 deg is the interpretable end of it.

## 2026-09-29 (r) — Deferring to the swr_main / swr_archive split

SK had already built the consolidation in parallel: `scripts/swr_main.py` for
the reported result and `scripts/exploratory/swr_archive.py` for everything
archived, each a single entry point, with the HFB scripts already registered in
both. My separate organisation was replaced by theirs.

Corrections made to fit it:

* **`swr_hfb_event_latency.py` returned to `exploratory/`.** I had promoted it
  to `scripts/` as a reviewer check, which broke the `event_latency` entry in
  `swr_archive.py` -- that dispatcher resolves modules as
  `scripts.exploratory.<name>`. It is archived and reachable as
  `python scripts/exploratory/swr_archive.py event_latency`.
* **My `exploratory/README.md` deleted.** `swr_archive.py` already lists every
  archived analysis with the question it asked and its outcome; two indexes
  would drift apart.
* **Archived sibling imports fixed.** Moving the scripts down a directory broke
  six of them: they import both maintained siblings (in `scripts/`) and each
  other (in `exploratory/`), and only the first directory was on the path. Both
  are now inserted. Verified: **all 8 archived modules import, all 8 maintained
  modules import.**

The two reviewer checks stay where they are useful, as subcommands on the
maintained partition script rather than as separate files:
`swr_hfb_roi_partition.py distance` and `... estimators` -- reachable through
`swr_main.py hfb_partition`.

Answering the original question: no, these did not need to be independent
scripts, and they no longer are -- two dispatchers cover 15 analyses, and
`mc/analyse/` already holds the shared implementation (`swr_hfb.py` for the
HFB computation, `swr_sakon.py` for the cluster test, `swr_stats.py`).

## 2026-09-29 (q) — Reviewer checks promoted out of exploratory

The six archived scripts stay archived -- they are frozen provenance and
merging them would save headers and nothing else. But three of the questions
they answer are ones a reviewer will ask, and two of those only existed as
scratch files. Those are now maintained.

### `swr_hfb_roi_partition.py distance`

Coupling against median distance to the nearest hippocampal derivation,
different-shaft only. Volume conduction predicts coupling FALLING with
distance; it rises.

| ROI | n | median mm | coupling | q |
|---|---:|---:|---|---:|
| ITG | 11 | 25.5 | +0.0030 | 0.674 |
| Auditory | 56 | 28.7 | +0.0014 | 0.613 |
| **MTG** | 105 | 31.2 | +0.0042 | **0.002** |
| TemporalPole | 16 | 32.3 | -0.0002 | 0.899 |
| STG | 132 | 33.3 | -0.0006 | 0.749 |
| TempOcc_Fusiform | 58 | 34.7 | +0.0037 | 0.432 |
| **lOFC** | 55 | 47.1 | +0.0093 | **0.002** |
| FrontalPole | 25 | 56.0 | +0.0065 | 0.147 |
| **MedialFrontal** | 179 | 60.7 | +0.0073 | **0.000** |
| Occipital | 32 | 61.6 | +0.0051 | 0.448 |
| Parietal | 11 | 63.8 | +0.0017 | 0.749 |

Spearman(distance, coupling) = **+0.462, p = 0.112** -- positive, i.e. the
opposite of the volume-conduction prediction, and the two most coupled regions
are the two furthest from the hippocampus.

### `swr_hfb_roi_partition.py estimators`

MedialFrontal minus each ROI under three estimators. The concern is that
these electrodes rarely co-occur, so a paired test discards most of the data.
The three agree throughout:

| ROI | paired | unpaired | mixed (per derivation) |
|---|---|---|---|
| STG | +0.0073 p=0.003 (36) | +0.0079 p<0.001 (53/42) | +0.0066 p<0.001 (311d) |
| TemporalPole | +0.0092 p=0.034 (8) | +0.0075 p=0.002 (53/8) | +0.0079 p=0.012 (195d) |
| Auditory | +0.0052 p=0.027 (22) | +0.0059 p=0.002 (53/25) | +0.0051 p=0.003 (235d) |
| vlPFC_IFG | +0.0043 p=0.141 (21) | +0.0056 p=0.044 (53/21) | +0.0053 p=0.032 (203d) |
| MTG | +0.0021 p=0.079 (30) | +0.0030 p=0.054 (53/33) | +0.0028 p=0.075 (284d) |
| lOFC | -0.0013 p=0.634 (21) | -0.0020 p=0.407 (53/21) | -0.0004 p=0.849 (234d) |

So the STG contrast quoted in the paragraph is not an artefact of the paired
design, and lOFC is not different from MedialFrontal under any estimator.

### `swr_hfb_event_latency.py` moved back to `scripts/`

It answers "is the pre-ripple rise just the button press?" and is maintained.
Its `PHASES`/`COND_KEYS` constants were moved from the archived
`swr_hfb_roi_conditions` into `swr_hfb_roi_partition` as `PHASES_EVENT`, so no
maintained script imports from `exploratory/`. Every script, kept and archived,
verified to import.

Five scripts remain archived: `swr_hfb_roi_conditions`, `swr_hfb_roi_compare`,
`swr_hfb_ripple_count_control`, `swr_hfb_event_specificity`,
`swr_hfb_ripple_vs_event`.

## 2026-09-29 (f) — cleanup: reported vs exploratory, and one set of write-ups

**Code.** The reported analysis had ended up inside the exploratory package
while the main runner still produced the old headline. Fixed:

- `mc/analyse/swr_explore/phase_contrast.py` -> `mc/analyse/swr_phase_contrast.py`
  (it is the reported analysis, not an exploration).
- `_labels` -> `mc.analyse.swr_content.phase_labels`. The encode/execute split
  defines the reported contrast, so it belongs in the library; both the
  reported module and `location_by_phase` now import it from there.
- `swr_content_main.py` gained a `phase` step (the reported analysis) and its
  `panel` step now regenerates fig20/fig21 as well. `timecourse` is retained
  but labelled NOT REPORTED, and dropped from the default `all` sequence.
- `timecourse_by_phase` docstring now says it is exploratory and why its
  fixed-window numbers differ (offset-pooled normalisation).
- Verified after the move: `--step=phase` reproduces bit-identically.

**Write-ups.** Five Methods files and three Results files had accumulated.
Consolidated to `METHODS.md` + `RESULTS.md` + `HOW_TO_RUN.md`; the six
superseded drafts moved to `superseded/` with a README saying why each was
replaced and warning that several report the confounded far-field control as
positive.

`SCRIPTS_GUIDE_swr_content.md` rewritten: reported vs exploratory, the library
map, the write-up index, and a "things that will bite you" list (grid_id vs
grid_num, the sqrt(n) term, first_traversal being a subset of encode, the
far-field confound, the decoder gate, what non-finite evidence means, per-curve
RNG seeding).

## 2026-09-29 (p) — Single-beta GLM, the reported paragraph, and a repo cleanup

### The regression, in two forms

`swr_hfb_glm.py` now has three entry points. All fit the CONTINUOUS HFB per
derivation, different-shaft only, with ripples / correct uncovers / incorrect
uncovers / navigation presses competing for the same variance, plus intercept
and 3rd-order Legendre drift.

  `run`     FIR, 30 lags per event type over -500..+1000 ms. Good for WHEN.
            Its partial R2 is useless for magnitude -- every block lands near
            1e-3 in every region because each carries 30 degrees of freedom.
            Its betas carry an unidentifiable level, so each is demeaned over
            its own lags and only the shape is read.
  `simple`  ONE regressor per event type: a boxcar over the response window
            MINUS a -1500..-750 ms baseline arm. 4 parameters, no level
            ambiguity, betas directly comparable. Three kernels fitted.

**A plain boxcar without the baseline arm is confounded and was discarded.** It
made the ripple beta positive in every region including STG (p = 0.025) and
Auditory (p = 0.0005), where the shifted-null analyses find nothing. Not
collinearity -- the regressors are orthogonal (ripple vs reward r ~ 0.003).
The cause is that a boxcar beta is relative to the SESSION mean, so it inherits
the behavioural state ripples occur in. The baseline arm removes it.

### Differenced single-beta result (kernel -250..+500 ms, per session)

| ROI | ripple | correct | incorrect | navigation |
|---|---|---|---|---|
| **MedialFrontal** | **+0.246 p<0.001** | +0.031 p=0.58 | -0.013 p=0.88 | +0.040 p=0.11 |
| lOFC | +0.189 p<0.001 | **+0.150 p=0.040** | +0.025 p=0.84 | -0.064 p=0.23 |
| MTG | +0.181 p<0.001 | -0.016 p=0.79 | -0.100 p=0.46 | +0.034 p=0.21 |
| FrontalPole | +0.197 p=0.006 | +0.136 p=0.12 | +0.112 p=0.34 | -0.006 p=0.93 |
| TempOcc_Fusiform | +0.152 p=0.001 | **+0.245 p=0.002** | +0.115 p=0.53 | -0.001 p=0.99 |
| Auditory | +0.080 p=0.041 | +0.122 p=0.088 | +0.058 p=0.52 | +0.009 p=0.86 |
| STG | +0.032 p=0.21 | -0.048 p=0.36 | -0.153 p=0.16 | +0.003 p=0.93 |

Within-region contrasts, now legitimate:
  MedialFrontal ripple - correct **+0.215, t(52) = +3.47, p = 0.0011**;
                ripple - navigation +0.206, t(52) = +4.77, p < 0.0001
  MTG           ripple - correct +0.197, t(32) = +3.13, p = 0.0037
  lOFC          ripple - correct +0.039, t(20) = +0.61, **p = 0.55**
Stable across all three kernels for MedialFrontal (ripple +0.246 / +0.164 /
+0.230, all p < 0.001; correct +0.031 / +0.015 / +0.012, all p > 0.58).

So the regression separates the three coupled regions: **mPFC and MTG are
ripple-only; lOFC is both; TempOcc_Fusiform is reward-not-ripple; STG is
neither.**

### Numbers fixed in the manuscript paragraph

* medial frontal event-locked response corrected to **-0.0002 +- 0.0013 z,
  t(52) = -0.20, p = 0.85** (0-500 ms post-uncovering, -1.5..-1.0 s baseline,
  all-ROI extraction). The previously circulated +0.0037 / p = 0.44 came from
  the frontal-only run with a -1.5..-0.75 s baseline.
* control-area count corrected from 8 to **10** (13 ROIs, 3 coupled).
* added the one surviving ROI contrast: medial frontal > STG, +0.0073,
  q = 0.030, against a well-powered null (132 derivations, -0.0006 +- 0.0014).

### Decision: the lOFC reward claim is NOT reported

The existing event-locked test puts lOFC correct uncovers at +0.0046 +- 0.0026,
t(20) = +1.75, **p = 0.095** -- a trend. Only the GLM makes it significant.
Since the paragraph is about mPFC and adding the GLM costs a full methods
section, lOFC is described as ripple-locked and the reward question is left
alone. The GLM stands as an unreported robustness result.

### Repo cleanup

`scripts/exploratory/` created with a README stating, per script, what it
answered and why it is not in the pipeline. Archived: `swr_hfb_roi_conditions`,
`swr_hfb_roi_compare`, `swr_hfb_ripple_count_control`, `swr_hfb_event_latency`,
`swr_hfb_event_specificity`, `swr_hfb_ripple_vs_event`.

The derivation-level pooler `_pooled` was promoted out of the archived
`swr_hfb_roi_compare` into `swr_hfb_locking_contrasts.pooled_curves_by`, so no
maintained script depends on an exploratory file; the archived copy re-exports
it. Path inserts in the moved files were adjusted for the extra directory
level, and every script -- kept and archived -- was verified to import.

Maintained for this branch: `swr_ripple_locked_hfb`, `swr_ripple_hfb_conditions`,
`swr_hfb_roi_partition`, `swr_hfb_event_locked`, `swr_hfb_glm`,
`swr_hfb_panels_2cm`, `swr_hfb_locking_contrasts`, `swr_hfb_coverage_figure`.

## 2026-09-29 (o) — FIR GLM replaces the split-and-contrast designs

`scripts/swr_hfb_glm.py`. SK called the epoch-splitting approach ridiculous and
was right twice over.

### Why the split designs could never work

1. Events overlap in time by construction, so no labelling makes the epochs
   independent.
2. **The rip and clean labels overlapped by 41%.** Measured: of 29366 "ripple-
   free" correct-uncover events, 12142 (41.3%) were ALSO in the `rip` group --
   epochs clean out to +500 ms whose first ripple arrives between +0.5 and
   +2 s. `clean` was defined on (-1.5, +0.5] and `rip` on (0, +2]; two windows,
   one comparison. Same for error (43.3%) and move (41.4%). Every rip-vs-clean
   p-value reported before this entry compared sets sharing ~40% of members.

### The model

FIR GLM on the continuous HFB, per derivation, different-shaft only:
one regressor per event type per 50 ms lag over -500..+1000 ms (30 lags x 4
event types = 120 columns), plus intercept and 3rd-order Legendre drift.
3072 beta rows, 61 sessions, 400 pair labels. Event types are near-orthogonal
(each block's R2 from the others ~= 0.003), so the design CAN separate them.

### The level is not identifiable -- only the shape

Diagnostics forced a change of estimand. Across sessions the ripple and reward
LEVELS trade off: r = -0.63 (MedialFrontal), -0.71 (MTG), -0.57 (STG), all
p < 1e-4, and no beta set returns to zero at the edges of the support. At this
event density a constant can move between regressors almost freely. Each
regressor's timecourse is therefore DEMEANED over its own lags before testing,
and **absolute heights must not be compared across regressors**.

### Result, demeaned shapes, cluster-corrected over lags

| ROI | regressor | peak | at lag | clusters p<0.05 |
|---|---|---|---:|---|
| MedialFrontal | **ripple** | +0.0114 | 0 ms | **-200..+100 (0.0005)**; +400..+950 (0.0005) |
| MedialFrontal | correct uncover | +0.0028 | -500 | none |
| MedialFrontal | incorrect | -0.0054 | +400 | none |
| MedialFrontal | navigation | -0.0019 | 0 | none |
| lOFC | **ripple** | +0.0102 | -100 ms | **-250..+100 (0.0005)**; +300..+550 (0.005); +650..+750 (0.015) |
| lOFC | correct uncover | +0.0057 | +800 | none |
| lOFC | incorrect | +0.0145 | +200 | none |
| MTG | **ripple** | +0.0073 | -50 ms | **-100..+50 (0.0005)**; +800..+900 (0.007) |
| MTG | **correct uncover** | -0.0075 | +150 ms | **+0..+150 (0.0025)** |
| STG | ripple | +0.0022 | -50 | **none** |
| STG | navigation | -0.0027 | -100 | -100..+0 (0.034) |

**The ripple regressor is the only one that survives in medial frontal and
lOFC**, with a transient peaking at 0 and -100 ms respectively. Neither
uncover regressor nor navigation clusters at all in those two regions, with
every other event type partialled out at the same moment. STG -- the negative
control -- has NO ripple cluster.

MTG is the exception: it carries both a ripple transient AND a NEGATIVE
correct-uncover transient at +0..+150 ms.

This is the cleanest version of the result so far, and it is the one to build
the paragraph on: the earlier event-locked contrasts were confounded by the
41% label overlap and the overlapping epochs, and both are gone here.

Output `hfb_glm_2026-09-29/` -- `betas.csv`, `glm_cluster_stats.csv`,
`design_diagnostics.csv`, `glm_betas.{pdf,png}`.

## 2026-09-29 (n) — (ripple in epoch) minus (ripple-free), correct and any uncover

The contrast SK asked for, with CIs, Cohen's d and three post-event windows.
Event-locked, baseline -1.5..-1.0 s, paired within session.

### Correct uncovers

| ROI | n | ripple | ripple-free | difference | 95% CI | d | p |
|---|---:|---|---|---|---|---:|---:|
| MedialFrontal | 53 | +0.0016 | -0.0025 | **+0.0041** | [+0.0006, +0.0077] | +0.32 | **0.023** |
| lOFC | 21 | +0.0078 | +0.0041 | +0.0037 | [-0.0034, +0.0108] | +0.24 | 0.287 |
| MTG | 33 | -0.0004 | -0.0041 | +0.0037 | [-0.0034, +0.0108] | +0.19 | 0.294 |

**The three point estimates are the same** (+0.0037 to +0.0041). What differs is
n -- 53 vs 21 vs 33 -- so only medial frontal reaches p < 0.05. This is a power
difference, not a regional one, and it should not be written as "the effect is
present in medial frontal and absent elsewhere".

### Any uncover

| ROI | n | ripple | ripple-free | difference | 95% CI | d | p |
|---|---:|---|---|---|---|---:|---:|
| MedialFrontal | 53 | -0.0004 | -0.0033 | +0.0029 | [-0.0001, +0.0059] | +0.27 | 0.055 |
| lOFC | 21 | +0.0066 | +0.0049 | +0.0017 | [-0.0043, +0.0077] | +0.13 | 0.570 |
| MTG | 33 | +0.0018 | -0.0053 | **+0.0071** | [+0.0010, +0.0132] | +0.41 | **0.024** |

MTG has the largest effect here (d = 0.41) and pooling correct with incorrect
HELPS it (+0.0037 -> +0.0071) while it hurts lOFC (+0.0037 -> +0.0017). So
whatever MTG is doing is about the uncovering, not about the outcome; lOFC's is
the reverse.

### Robustness to the window

MTG / any uncover is the only contrast stable across all three windows
(+0.0071 p = 0.024, +0.0059 p = 0.037, +0.0058 p = 0.034). MedialFrontal /
correct is significant only in 0-500 ms and fades (p = 0.071 at 0-750,
p = 0.112 at 250-750), so it depends on the window choice.

**Nothing survives BH** (min q = 0.072). Every statement above is a pattern.

Output `hfb_event_specificity_2026-09-29/ripple_vs_clean.{pdf,png}` and
`ripple_vs_clean_windows.csv`.

## 2026-09-29 (m) — The ripple/event plane, and a sliding window over the event epoch

`scripts/swr_hfb_ripple_vs_event.py`. Two quantities, drawn as a plane and
never subtracted: x = ripple-locked coupling (peri +-250 ms minus non-peri),
y = event-locked response in RIPPLE-FREE epochs (0-500 ms minus -1.5..-1.0 s).
The windows have different shapes -- symmetric around a ripple, causal around
an event -- so a difference between them would be a window artefact.

### The plane, correct uncovers

| ROI | ripple-locked | q | event, ripple-free | q |
|---|---|---:|---|---:|
| lOFC | +0.0093 | 0.001 | +0.0041 | 0.627 |
| MedialFrontal | +0.0073 | 0.000 | -0.0025 | 0.363 |
| FrontalPole | +0.0065 | 0.102 | +0.0027 | 0.756 |
| MTG | +0.0042 | 0.001 | -0.0041 | 0.425 |
| TempOcc_Fusiform | +0.0037 | 0.299 | **+0.0110** | 0.071 |
| Auditory | +0.0014 | 0.424 | +0.0063 | 0.225 |
| STG | -0.0006 | 0.672 | +0.0008 | 0.756 |

**"Medial frontal is ripple-locked" is supported**: significant on x, flat and
slightly negative on y. **"lOFC is reward-locked" is NOT**: lOFC is the most
ripple-coupled region of all and its ripple-free event response is +0.0041,
q = 0.63. It sits on the same axis as medial frontal, not the other one.

The only regions leaning to the EVENT axis are TempOcc_Fusiform (+0.0110,
q = 0.071) and Auditory (+0.0063, q = 0.225), neither ripple-coupled and
neither significant. So the dissociation SK expected exists in the figure's
geometry but not in these two regions.

### Sliding window, 250 ms wide, 50 ms steps, BH within each sweep

| ROI | condition | epochs | significant centres | peak |
|---|---|---|---|---|
| lOFC | correct uncovers | with ripple | -525..+625 ms | +0.0107 at +375 ms (q=0.030) |
| MedialFrontal | navigation | ripple-free | +425..+625 ms | +0.0059 at +625 ms (q=0.004) |
| lOFC | navigation | ripple-free | +525..+625 ms | +0.0081 at +625 ms (q=0.040) |
| MTG | navigation | with ripple | -875..+625 ms | +0.0059 at **-825 ms** (q=0.038) |

Two things the sweep exposes that the fixed window hid:

  * **Three of the four peak in the LAST window (+625 ms).** The response is
    still rising where the epoch ends, so 0-500 ms is not capturing it and
    HALF_S = 1.5 s is too short on the post side. Any effect size quoted from
    the 0-500 ms window is a lower bound.
  * **MTG's navigation effect peaks at -825 ms, BEFORE the event**, and is
    significant across almost the whole sweep. That is not an evoked response;
    it is a sustained offset relative to the -1.5..-1.0 s baseline, and it
    means the baseline window itself sits inside whatever is going on. MTG's
    navigation number should not be read as event-evoked.

Output `hfb_ripple_vs_event_2026-09-29/`.

## 2026-09-29 (l) — Is MTG the same as medial frontal? Indistinguishable, but underpowered

Direct paired tests rather than eyeballing two columns.

### Ripple-locked coupling

MedialFrontal +0.0059 vs MTG +0.0037 on the 30 sessions with both.
Difference **+0.0021, 95% CI [-0.0003, +0.0045], t(29) = +1.82, p = 0.079**.
MTG is 64% of MedialFrontal numerically.

The CI is the point: it excludes a difference larger than 0.0045, but
MedialFrontal's whole effect is 0.0059, so a difference of 76% of the effect is
still inside it. This is a failure to reject, NOT equivalence.

### Event-locked, MedialFrontal minus MTG, all four conditions x rip/clean

**0 of 8 differ** (min q = 0.61). Largest gap: incorrect uncovers with a ripple,
-0.0089, CI [-0.0205, +0.0027]. Every CI spans zero widely.

### Proximity does NOT explain MTG

MTG is the tissue the hippocampal electrodes pass through, so the obvious worry
is volume conduction. Median distance to the nearest hippocampal derivation
(different-shaft only): MTG 31.2 mm vs MedialFrontal 60.7 mm
(Mann-Whitney p = 9.4e-21). But distance does not predict coupling:

| ROI | median mm | coupling |
|---|---:|---|
| ITG | 25.5 | +0.0030 |
| Auditory | 28.7 | +0.0014 |
| **MTG** | **31.2** | **+0.0042** |
| TemporalPole | 32.3 | -0.0002 |
| STG | 33.3 | -0.0006 |
| lOFC | 47.1 | +0.0093 |
| **MedialFrontal** | **60.7** | **+0.0073** |
| Occipital | 61.6 | +0.0051 |

MTG's nearest neighbours in distance -- Auditory, TemporalPole, STG, all within
3 mm of its median -- show NO coupling. The two most strongly coupled regions
are the two furthest from the hippocampus. Volume conduction predicts the
opposite ordering, so it does not explain MTG.

### Correction to 2026-09-29 (k)

That entry said "MTG's navigation response is the one clear event-driven signal
that does not require a ripple". Not unique: MedialFrontal's ripple-free
navigation response is also nominally positive (+0.0023, p = 0.041) alongside
MTG's (+0.0038, p = 0.036) and lOFC's (+0.0034, p = 0.045). All three behave the
same way, and none survives BH.

### Verdict

On every measure computed, MTG and MedialFrontal are statistically
indistinguishable and qualitatively identical: both ripple-coupled, neither with
a ripple-free uncover response, both with positive ripple-minus-clean for
uncovers and none for navigation. Whether they are genuinely the same needs an
equivalence test with a pre-specified margin, and n = 30 shared sessions will
not support one at the current effect size.

## 2026-09-29 (k) — Exclusive partition, all-ROI event-locking, and the specificity test

Three corrections and one new extraction, all from SK's questions.

### 1. MedialFrontal now excluded from every other ROI

`swr_hfb_roi_partition.py run --exclude_medial=True`. The 179 MedialFrontal
derivations are REMOVED from their atlas ROI, so every row is disjoint and the
comparison panel is an independent test. Consequences: **mPFC and mOFC vanish
entirely** (they lie 100% inside MedialFrontal, which is exactly why they
looked like independent coupled regions and were not); lOFC drops 72 -> 55
derivations.

Session level, only THREE regions are significantly coupled:

| ROI | sess | der | coupling | q |
|---|---:|---:|---|---:|
| lOFC | 21 | 55 | +0.0093 | 0.002 |
| MedialFrontal | 53 | 179 | +0.0073 | 0.000 |
| MTG | 57 | 105 | +0.0042 | 0.002 |

FrontalPole +0.0065, p = 0.045 but q = 0.147. Still only one contrast survives:
MedialFrontal > STG, +0.0073, q = 0.030. Output
`hfb_roi_partition_exclusive_2026-09-29/`.

### 2. MTG had event-locked data all along -- my restriction, not a data gap

`swr_hfb_event_locked.py` inherited `roi_family.isin(["mPFC","mOFC"])` from
`swr_ripple_hfb_conditions.py`, where it was a cost decision. Removed. The
re-extraction covers all five cortical families: 768 derivations, 61 sessions.
Output `hfb_event_locked_all_2026-09-29/`.

Also per SK: baseline is now **-1.5 to -1.0 s** and the window **0 to +500 ms**,
and a third ripple label was added --

  rip     >=1 ripple in (0, +2 s]
  norip   none in that window (says nothing about BEFORE the event)
  clean   **no ripple anywhere in the epoch, -1.5 to +0.5 s** -- SK's reading,
          and the right one: the measured window is guaranteed ripple-free.

### 3. MTG is MIDDLE temporal gyrus, not medial temporal

Audit: "Middle Temporal Gyrus, posterior division" (211 der), "anterior
division" (61), A21c/A21r, ctx-*-middletemporal. Lateral temporal neocortex.
Nothing to do with MTL, parahippocampal or entorhinal cortex.

### The specificity result (`swr_hfb_event_specificity.py`)

Event-locked, 0-500 ms vs -1.5..-1.0 s baseline, real minus shifted null,
per session, phases pooled.

WITH a ripple in the epoch:

| ROI | condition | effect | p | q |
|---|---|---|---:|---:|
| lOFC | correct uncovers | +0.0078 | 0.011 | 0.060 |
| lOFC | any uncover | +0.0066 | 0.016 | 0.060 |
| MTG | navigation presses | +0.0047 | 0.0056 | 0.060 |
| MTG | incorrect uncovers | +0.0079 | 0.054 | 0.131 |
| MedialFrontal | navigation presses | +0.0020 | 0.020 | 0.060 |

RIPPLE-FREE epochs -- nothing survives BH (all q >= 0.135). Navigation presses
are nominally positive in all three (MedialFrontal +0.0023 p = 0.041, lOFC
+0.0034 p = 0.045, MTG +0.0038 p = 0.036); every uncover cell is n.s. and the
uncover effects in MedialFrontal and MTG are NEGATIVE.

Direct (ripple in epoch) MINUS (ripple-free), paired within session:

| ROI | condition | with ripple | ripple-free | difference | p | q |
|---|---|---|---|---|---:|---:|
| MTG | incorrect uncovers | +0.0079 | -0.0083 | **+0.0161** | 0.010 | 0.096 |
| MedialFrontal | correct uncovers | +0.0016 | -0.0025 | +0.0041 | 0.023 | 0.096 |
| MTG | any uncover | +0.0018 | -0.0053 | +0.0071 | 0.024 | 0.096 |
| MedialFrontal | any uncover | -0.0004 | -0.0033 | +0.0029 | 0.055 | 0.165 |
| MTG | navigation | +0.0047 | +0.0038 | +0.0009 | 0.591 | 0.709 |
| lOFC | correct uncovers | +0.0078 | +0.0041 | +0.0037 | 0.288 | 0.503 |

Nothing survives BH (min q = 0.096), but the pattern is consistent and
interpretable: for UNCOVERS the response needs a ripple in all three regions
(all differences positive, three at p < 0.06); for NAVIGATION it does not
(MTG +0.0009 p = 0.59; MedialFrontal -0.0003 p = 0.79). MTG's navigation
response is the one clear event-driven signal that does not require a ripple.

Output `hfb_event_specificity_2026-09-29/`.

## 2026-09-29 (j) — What is actually special about medial frontal: two candidates, one bug

### Bug: relabelling emptied mPFC and mOFC

The event-locked ROI mapping did `ix.loc[in_MedialFrontal, "roi"] = "MedialFrontal"`.
MedialFrontal contains 100% of the mPFC and mOFC derivations, so that RELABEL
deleted both ROIs (they came back "too few"), and it also stripped lOFC of the
24% of its derivations that fall inside |x| <= 20 mm. Fixed by appending a
duplicate set of MedialFrontal rows instead of overwriting.

Consequence: yesterday's lOFC no-ripple figure of **+0.0175, p = 0.047 is
wrong**; on the full lOFC set it is **+0.0144, p = 0.055**, and all uncovers
+0.0247, p = 0.058.

### Candidate 1: ripple-specificity. Suggestive, NOT significant

Event-locked reward response, 0-500 ms post-uncovering, pre-event baseline,
explore phase:

| ROI | all uncovers | uncovers with NO ripple |
|---|---|---|
| **MedialFrontal** | **+0.0005, t(52)=+0.14, p=0.89** | **-0.0018, t(36)=-0.30, p=0.76** |
| lOFC | +0.0247, t(20)=+2.01, p=0.058 | +0.0144, t(15)=+2.08, p=0.055 |
| mOFC | -0.0031, t(17)=-0.28, p=0.78 | +0.0111, t(9)=+0.75, p=0.47 |
| FrontalPole | -0.0060, t(15)=-0.51, p=0.62 | +0.0021, t(7)=+0.08, p=0.92 |
| mPFC | -0.0018, t(34)=-0.39, p=0.70 | +0.0047, t(24)=+0.64, p=0.53 |

Medial frontal is flat; lOFC is ~50x larger numerically but sits at p = 0.055.
A trend, not a dissociation. MTG cannot be tested at all -- the conditions
extraction covered frontal derivations only.

### Candidate 2: response timing. Does NOT separate them

Peak latency of the region curve, bootstrap CI over sessions:
MedialFrontal -10 ms [-40, +10], lOFC -30 ms [-80, +10], mOFC -60 ms
[-180, +220], mPFC +30 ms [-80, +140]. All overlapping; no timing claim.

Cluster extents differ (MedialFrontal -190..+370 ms, lOFC -260..+150, mOFC
-230..+60) but cluster EXTENT scales with power and MedialFrontal has 53
sessions against lOFC's 21, so this is not a usable contrast either.

### What is defensible

Medial frontal is the most RELIABLY coupled region (t(52) = +6.53,
p = 2.7e-8 -- the largest t of any ROI) and the only one with a flat
event-locked reward response. The ripple-specificity contrast against lOFC is
a trend. Making it decisive needs an event-locked extraction that covers the
non-frontal ROIs, and a commensurable ripple-vs-event estimator -- the two
currently use different baselines and windows and cannot be subtracted.

## 2026-09-29 (i) — Cluster test vs window test: they disagree, and it matters

SK was right and my summary was wrong. Two tests were being conflated:

  WINDOW   mean of the curve over +-250 ms, one value per session, t vs 0.
           Aggregates over 500 ms, so a small sustained offset passes.
  CLUSTER  `swr_sakon.cluster_perm_time` on the timecourse, sign-flip null,
           corrected over time. Needs a temporally CONCENTRATED effect.

Ripple-locked, session level, first traversal:

| ROI | condition | window | cluster |
|---|---|---|---|
| MedialFrontal | correct | t(50)=+2.66, p=0.0105 | **-130..+10 ms, p=0.038** |
| MedialFrontal | incorrect | t(52)=+2.92, p=0.0051 | **NONE** |
| MedialFrontal | navigation | t(50)=+1.38, p=0.175 | NONE |
| lOFC | correct | t(18)=+2.77, p=0.0127 | **-190..+40 ms, p=0.018** |
| lOFC | incorrect | t(20)=+0.66, p=0.514 | NONE |
| lOFC | navigation | t(20)=+0.27, p=0.787 | NONE |
| mPFC | correct | t(33)=+0.76, p=0.456 | NONE |

So "a cluster at ripple time for correct uncovers but not for incorrect or
navigation" is TRUE for both MedialFrontal and lOFC. My earlier note that
incorrect uncovers "are significant" was the WINDOW test, which is a different
claim: incorrect uncovers raise the mean over +-250 ms without producing a
temporally localised cluster.

Encoding phase, for completeness: MedialFrontal correct has no cluster there
(window p = 0.054) but navigation does (+40..+280 ms, p = 0.006); lOFC correct
does (-180..+30 ms, p = 0.044).

### Which regions actually beat medial frontal: none

Session level, MedialFrontal minus each partition ROI, paired. NINE ROIs have
ZERO derivation overlap with MedialFrontal and are the genuine controls (MTG,
STG, Auditory, TemporalPole, ITG, TempOcc_Fusiform, Occipital, Parietal,
vlPFC_IFG); mPFC and mOFC overlap it 100% and are not independent; lOFC 24%,
FrontalPole 11%.

Only ONE survives BH: MedialFrontal > STG, +0.0073 +- 0.0022, t(35) = +3.26,
p = 0.0025, q = 0.033. **No region is significantly higher than medial
frontal**, lOFC included: lOFC - MedialFrontal = +0.0005 +- 0.0014,
t(20) = +0.33, p = 0.74.

### The defensible lOFC claim

lOFC is the only 0%-or-low-overlap region that reproduces medial frontal's
selectivity (cluster for correct uncovers only). But its HFB ALSO rises after
correct uncoverings with no ripple within 2 s (+0.0175, p = 0.047, 0-500 ms
post-event), where medial frontal's does not (-0.0018, p = 0.76). That is the
reward-confound reading, and it is supported.

## 2026-09-29 (h) — The event-locked window was wrong; SK caught it

The 2026-09-29 (g) conclusion "without a ripple there is no response at all"
was an artefact of measuring an EVOKED response in a window CENTRED on the
event. A +-250 ms window spans 250 ms of pre-event baseline and 250 ms of a
response that has barely begun, and the two cancel. SK pointed out that the
event-locked trace is negative at +250 ms and rises at +500 ms, which is where
a feedback-evoked frontal response belongs.

### Window sweep, correct uncovers in explore, pre-event baseline

lOFC, no ripple followed:

| window | effect | p |
|---|---|---:|
| **sym +-250 ms** (what (g) used) | **+0.0014** | **0.83** |
| post 0-500 ms | **+0.0175** | **0.047** |
| post 0-750 ms | +0.0140 | 0.045 |
| post 100-500 ms | +0.0167 | 0.083 |

Same data, opposite conclusion. The 250 ms bin sweep shows why: lOFC no-ripple
is -0.0111 in -250..0 ms and +0.0138 (p = 0.025) in 0..+250 ms.

### Corrected result

With a post-event window (0-500 ms), explore phase:

| ROI | condition | ripple followed | no ripple | difference |
|---|---|---|---|---|
| MedialFrontal | correct | +0.0037 (p=0.44) | -0.0018 (p=0.76) | +0.0112 (p=0.24) |
| MedialFrontal | navigation | +0.0001 (p=0.95) | -0.0025 (p=0.34) | +0.0026 (p=0.46) |
| lOFC | correct | +0.0243 (p=0.11) | **+0.0175 (p=0.047)** | -0.0068 (p=0.51) |
| lOFC | navigation | +0.0049 (p=0.35) | -0.0004 (p=0.93) | +0.0053 (p=0.45) |

**RETRACTED from (g):** "every no-ripple cell is at or below zero" and "the
ripple+ minus ripple- contrast is positive for correct uncovers in all three
ROIs". Both were window artefacts. With a post-event window there is NO
ripple/no-ripple difference anywhere (all p >= 0.22), and the MedialFrontal
contrast that was p = 0.017 becomes p = 0.24.

**What now stands, and it is a cleaner dissociation:**

  * MEDIAL FRONTAL has no event-locked reward response in ANY window -- pooled,
    ripple-followed or no-ripple, symmetric or post-event, every p >= 0.22. Its
    HFB is ripple-locked and not reward-locked.
  * lOFC DOES have an event-locked reward response, and it does NOT need a
    ripple (+0.0175, p = 0.047 with no ripple within 2 s). lOFC is both
    ripple-locked and reward-locked.

Cluster test over the whole epoch finds only one surviving cluster in the whole
event-locked set: lOFC, correct uncovers with a ripple, -50..+40 ms, p = 0.042.
Nothing for medial frontal.

### Lesson for the estimator

A window centred on the lock is right for a ripple, where the response is
symmetric, and wrong for an event, where it is causal. The two branches should
not share a window just because they share an estimator.

Output: window sweep and post-event contrasts in the scratch analysis; the
extraction itself (`hfb_event_locked_2026-09-29/`) is unchanged and already
stores both baselines, so no re-run was needed.

## 2026-09-29 (e) — the reported analysis is now ripple-window only

`mc/analyse/swr_explore/phase_contrast.py`
(`scripts/swr_content_explore.py phase_contrast`), results in
`swr/phase_contrast_2026-09-29/`.

**Why.** The reported fixed-window numbers were being produced by the
41-offset pipeline evaluated at offset 0, so each unit's counts were z-scored
across ALL offsets pooled. Defensible, but an implementation detail with no
scientific motivation: it forced the Methods to describe a sliding window that
is no longer reported, and it made the drafted Methods paragraph subtly not
match the code. The new module counts spikes in the ripple windows and their
matched flanks and nowhere else, so the paragraph is literally true. It also
brings all three reported tests into one place at 1000 permutations (the flank
control was previously at 200 in `location_by_phase`).

**Nothing of substance changed.** Every conclusion is identical:

| test | pooled-offset | ripple window only |
|---|---|---|
| HC pooled encode | +0.576, P = 1.8e-4 | +0.566, P = 2.0e-4 |
| HC pooled contrast | +0.455, P = 0.0085 | +0.462, P = 0.0079 |
| aHP contrast | +0.367, P = 0.027 | +0.370, P = 0.026 |
| mHP contrast | +0.439, P = 0.107 | +0.432, P = 0.112 |

Known-phase values are slightly larger (pooled +0.21 vs +0.12) but stay n.s.
Flank specificity at 1000 perms: pooled encode +0.233 (P = 0.115), aHP +0.230
(P = 0.152), mHP -0.001 (P = 0.997) -- still null, as before.

Figures 20/21 now read `phase_contrast_*` via `_phase_contrast()`, falling back
to the old module so nothing breaks. `timecourse_by_phase` is kept for the
descriptive time course only.

## 2026-09-29 (d) — RETRACTED: the far-field specificity control is confounded

The "surround" control (ripple minus the mean of the 14 offsets at
|t| >= 350 ms) was reported yesterday as showing that the encoding-phase
location signal is ripple-specific (pooled HC +0.37, P = 0.030). **That is
withdrawn.** Every far-field window is scored for the square occupied AT THE
RIPPLE, and measured across all sessions the subject is still on that square
only:

| offset | P(same square) |
|---|---|
| 0 ms | 100.0% |
| +-250 ms | 70.1% |
| +-350 ms | 60.6% |
| +-500 ms | 49.1% |
| **far field (14 offsets)** | **54.6%** |

So ~45% of far-field windows are scored for a square the subject has left,
their evidence is depressed for reasons unrelated to ripples, and the
ripple-minus-far-field difference is inflated. This is the SAME failure that a
fixed +-500 ms flank showed (53.7% same-square, CHANGELOG 2026-09-17 h) and
that the occupancy-matched flank control was built to avoid. The far-field
control silently reintroduced it.

**The valid specificity control is the matched flank, and it is null:**
pooled HC encoding +0.226 (t(50) = +1.56, P = 0.125); mHP +0.010 (P = 0.957);
aHP +0.216 (P = 0.171); execution negative in all. Report specificity as
tested-and-not-found.

**Still valid:** the far-field version of the PHASE CONTRAST. Label expiry
applies equally to both phases, so it cancels in a difference; that control
asks whether the phase effect is localised to the ripple and it is, for pooled
HC (+0.05, P = 0.81 in the far field vs +0.46 at the ripple). ⚠ Not for mHP
(+0.41, P = 0.21).

**Also corrected: which ROI to lead with.** Earlier text leant on aHP because
its P values were smaller. That is a power difference (n = 44 vs 29), not a
biological one. mHP has the larger effect (+0.57 vs +0.41) and the only
significant cluster (P = 0.005 vs 0.158). Lead with pooled hippocampus.

Corrected write-up in
`data/final_results/ripple_analysis/RESULTS_ripple_content_FINAL.md`;
Methods in `METHODS_ripple_content_FINAL.md`.

## 2026-09-29 (g) — Reward-locked HFB: the rise is NOT a reward response (PARTLY RETRACTED by 2026-09-29 (h): the +-250 ms window was wrong for an evoked response)

`scripts/swr_hfb_event_locked.py`. SK asked whether frontal HFB simply goes up
after a rewarded uncovering, with the ripple incidental. New extraction over
all 61 sessions, locked to the EVENT instead of the ripple, every condition
split by whether a ripple followed within 2 s. 29525 traces, 53 sessions,
148 cortical pair labels.

Epochs are +-1.5 s and baselined to -1500..-750 ms, PRE-event. The ripple-locked
convention (250-750 ms flanks) sits on top of the very response it would be
baselining when the lock is an event, so it would subtract the effect. Both
estimators are written out (`peri_pre`, `peri_flank`). Shifted null, seed and
shift range unchanged; different-shaft derivations only.

### There is essentially NO event-locked HFB response

All correct uncovers pooled, pre-event baseline, real minus shifted null,
per session:

| ROI | phase | n | effect | p |
|---|---|---:|---|---:|
| MedialFrontal | explore | 53 | **-0.0008 +- 0.0035** | 0.83 |
| MedialFrontal | execute | 53 | +0.0009 +- 0.0009 | 0.30 |
| lOFC | explore | 21 | +0.0112 +- 0.0078 | 0.17 |
| lOFC | execute | 21 | +0.0009 +- 0.0010 | 0.38 |
| mPFC | explore | 35 | -0.0045 +- 0.0043 | 0.31 |

**0 of 18 cells survive BH**, and the only p < 0.05 anywhere in the table is
lOFC navigation in explore, which is NEGATIVE (-0.0065, p = 0.024). So HFB does
not simply rise after a rewarded uncovering. The ripple-locked effect is not a
reward-evoked response wearing a ripple costume.

### Without a ripple there is no response at all

Event-locked, split by whether a ripple followed:

| ROI | kind | phase | ripple followed | no ripple |
|---|---|---|---|---|
| MedialFrontal | correct | explore | +0.0031 (p=0.50, n=49) | **-0.0096 (p=0.073, n=37)** |
| lOFC | correct | explore | **+0.0189 (p=0.042, n=19)** | -0.0028 (p=0.64, n=16) |
| mPFC | correct | explore | -0.0022 (p=0.71, n=33) | -0.0074 (p=0.28, n=25) |
| MedialFrontal | correct | execute | +0.0015 (p=0.36) | -0.0014 (p=0.43) |

Every `no ripple` cell is at or below zero. The ripple+ minus ripple- contrast
is positive for correct uncovers in all three ROIs during explore
(MedialFrontal +0.0187 p = 0.017; mPFC +0.0145 p = 0.095; lOFC +0.0147
p = 0.152).

### But the contrast does not survive, and depends on the baseline

Across 25 ripple+ vs ripple- contrasts, **0 survive BH under either baseline**.
MedialFrontal correct/explore is p = 0.017, q = 0.32 with the pre-event
baseline and p = 0.052, q = 0.49 with the flank baseline. A result that moves
that much with the baseline is not ready to be leaned on.

⚠ And the comparison is quasi-experimental, not a manipulation. Whether a
ripple follows is not randomly assigned: ripple rate rises with stillness
(SWR_SUMMARY F1), so ripple+ events are plausibly in a different behavioural
state from ripple- ones. "The response needs a ripple" and "the response needs
the state that produces ripples" are not separated by this design.

Figure `event_vs_ripple_explore.pdf`: ripple-locked, event-locked pooled, and
event-locked split by ripple, for MedialFrontal and lOFC. Columns 1 and 2 use
different baselines and epoch widths, so heights are not comparable across
columns -- stated on the figure.

Output `hfb_event_locked_2026-09-29/`.

## 2026-09-29 (f) — Event-centred latency: the ripple-centred contrast was task structure

`swr_hfb_event_latency.py first`. SK asked what happens if you take only the
FIRST ripple after each uncover. It corrects yesterday's interpretation.

### The two views, and why they differ

RIPPLE-centred (`run`): each ripple takes its nearest preceding event, so one
uncovering owns EVERY ripple in the next 2 s. EVENT-centred (`first`): each
event contributes one number, the time to the next ripple, per hippocampal
derivation. Events with no ripple in 2 s kept as censored rows.

### Ripple timing does NOT depend on what just happened

| phase | condition | n ev x probe | any ripple <2 s | median first ripple | IQR | median gap to next EVENT |
|---|---|---:|---:|---:|---|---:|
| first traversal | correct | 16742 | 0.28 | 833 ms | [423, 1404] | **1725 ms** |
| first traversal | incorrect | 57101 | 0.28 | 935 ms | [452, 1443] | 575 ms |
| first traversal | navigation | 66193 | 0.27 | 920 ms | [446, 1444] | **400 ms** |
| encoding | correct | 26931 | 0.28 | 857 ms | [422, 1418] | 1325 ms |
| encoding | navigation | 101485 | 0.27 | 919 ms | [441, 1437] | 450 ms |
| after encoding | correct | 182148 | 0.29 | 907 ms | [429, 1424] | 525 ms |
| after encoding | navigation | 264875 | 0.29 | 901 ms | [428, 1425] | 525 ms |

The next ripple arrives ~830-935 ms later whatever the event was, with
near-identical IQRs, and the probability of ANY ripple within 2 s is 0.27-0.29
in every single cell. The distributions overlap almost exactly.

### RETRACTION of 2026-09-29 (e)

That entry reported "correct uncovers are the FURTHEST of the three conditions
from their ripples (730 ms vs 266 ms for navigation)" and read it as a fact
about ripple timing. **It is not.** It is event spacing: the median gap to the
next event is 1725 ms after a correct uncover on the first traversal against
400 ms after a navigation press, a factor of 4.3. A correct uncovering is
followed by a pause, so it OWNS a long window and collects late ripples; a
navigation press is overtaken by the next press within 400 ms and everything
after that is relabelled. The `median (assigned)` column makes the mechanism
explicit -- 665 ms for first-traversal correct against 251 ms for navigation,
tracking the gaps, not the ripple process.

### What survives, and what it now means

The ripple-centred number is still the right one for the question it was asked
for: FOR THE RIPPLES IN THE reward_explore CONDITION, the uncovering was a
median 730 ms earlier. So the lOFC correct-uncover peak at about -50 ms is
still ~680 ms after the median press and still cannot be the press response.
That conclusion holds.

What does NOT hold is any claim that ripples are timed differently after
different event types. They are not. The conditions differ in BEHAVIOURAL
STATE -- a post-reward pause is stiller, and SWR_SUMMARY F1 already records
that ripple rate rises with stillness -- which is precisely why each condition
carries its own shifted null rather than being compared raw.

One small real timing effect remains visible in the figure: on the first
traversal and in encoding, correct uncovers show a modest excess of first
ripples around 400-500 ms and a deficit beyond 800 ms, while the other two
conditions decline smoothly. Not tested, and not what the medians capture.

Output `hfb_event_latency_2026-09-29/first_ripple.{csv,json,pdf,png}`.

## 2026-09-29 (e) — How ripples get their condition, and when the press actually was (PARTLY RETRACTED by 2026-09-29 (f): the latency contrast is event spacing, not ripple timing)

`scripts/swr_hfb_event_latency.py`, plus a press-latency row on every 2 cm
condition panel.

### The allocation rule, exactly

`swr_ripple_hfb_conditions.session_events` lays down events: every UNCOVERING
(correct or not, every phase) -> `reward_<phase>`/`error_<phase>`; every
MOVEMENT PRESS (every phase, inside the grid's A..D window) -> `move_<phase>`;
pseudo-events every 0.5 s with nothing within +-2 s -> `still_<phase>`. Then
`.assign` gives each ripple the label of the single NEAREST PRECEDING event
within POST_S = 2.0 s; ripples with nothing in the prior 2 s drop out. 77013
ripples labelled over 61 sessions.

So the event ALWAYS precedes the ripple, latency bounded to (0, 2] s. A
pre-ripple HFB bump is exactly where a press response would land -- which is
why the latency had to be measured rather than assumed.

### Event-to-ripple latency, and it does NOT support a press-response account

| phase | condition | n | median | IQR | <=250 ms | <=500 ms |
|---|---|---:|---:|---|---:|---:|
| first traversal | **correct** | 4263 | **730 ms** | [387, 1263] | 0.16 | 0.35 |
| first traversal | incorrect | 6578 | 381 ms | [181, 701] | 0.34 | 0.62 |
| first traversal | **navigation** | 5387 | **266 ms** | [127, 523] | 0.47 | 0.74 |
| encoding | **correct** | 5740 | **676 ms** | [350, 1197] | 0.18 | 0.38 |
| encoding | incorrect | 9028 | 406 ms | [189, 743] | 0.32 | 0.59 |
| encoding | navigation | 10081 | 344 ms | [158, 683] | 0.39 | 0.64 |
| after encoding | correct | 20991 | 422 ms | [185, 798] | 0.33 | 0.57 |
| after encoding | navigation | 27670 | 356 ms | [159, 652] | 0.38 | 0.64 |

**Correct uncovers are the FURTHEST of the three conditions from their
ripples, and navigation presses the closest.** On the first traversal the
median correct uncover is 730 ms before its ripple and only 16% are within
250 ms, while the median navigation press is 266 ms before and 47% are within
250 ms. The density figure shows correct uncovers nearly FLAT across the whole
2 s window in the early phases, against a sharp pile-up near zero for the other
two.

That is the opposite of the confound. The lOFC correct-uncover HFB peaks at
about -50 ms, roughly 680 ms AFTER the median press, so it cannot be the press
response. The condition most at risk of being press-evoked is NAVIGATION, whose
presses sit closest to the ripple -- and which is exactly the condition whose
clusters the ripple-count control already flagged as possibly power-driven.

### Panels now carry the press row

`swr_hfb_panels_2cm.py` draws, per condition, the median (tick) and IQR (line)
of the event time on the ripple-locked axis, i.e. at MINUS the latency. A
median off the left edge is a CARET at -500 with the IQR clipped, never
silently omitted -- omitting it would make the press look closer than it is.
Row order under the traces: press times, ripple extent, then cluster bars.

Output `hfb_event_latency_2026-09-29/`, `hfb_panels_2cm_2026-09-29/`.

## 2026-09-29 (c) — the phase contrast becomes a fixed-window test

`timecourse_by_phase` now also computes the contrast in a single **fixed
window** with **1000 subsample draws**, which is what gets reported; the
41-offset time course is kept as the descriptive figure only. Results in
`swr/timecourse_by_phase_2026-09-29/` (`fixed_window_summary.csv`,
`fixed_window_overview.json`), write-up in
`data/final_results/ripple_analysis/RESULTS_phase_split.md` and
`METHODS_ripple_content.md`.

**Why the ripple window and not ±250 ms:** median occupancy dwell is 0.367 s,
so across a ±250 ms span the subject has frequently left the square the window
is labelled with. Both were computed and they agree, so nothing hinges on it
(pooled contrast +0.455, P = 0.0085 at the ripple; +0.446, P = 0.005 over
±250 ms).

Pooled hippocampus, n = 51, ripple window, counts matched:

| test | z | t(50) | P | 95% CI |
|---|---|---|---|---|
| encoding | +0.58 | +4.05 | 1.8e-4 | [+0.29, +0.86] |
| once known | +0.12 | +1.16 | 0.25 | [-0.09, +0.33] |
| encoding - once known | +0.46 | +2.74 | **0.0085** | [+0.12, +0.79] |
| encoding - surround | +0.37 | +2.23 | **0.030** | [+0.04, +0.71] |
| once known - surround | -0.03 | -0.28 | 0.78 | [-0.26, +0.20] |

aHP: encoding +0.41 (P = 0.008), contrast +0.37 (P = 0.027), surround +0.36
(P = 0.028). mHP: encoding +0.57 (P = 0.009), contrast +0.44 (P = 0.11),
surround +0.09 (P = 0.69).

**Draw count settled empirically.** 20 / 100 / 1000 draws give pooled +0.445
(P = 0.012) / +0.462 (P = 0.008) / +0.455 (P = 0.008); aHP +0.406 (P = 0.015)
/ +0.351 (P = 0.034) / +0.367 (P = 0.027). 20 reaches the same conclusion but
the third decimal of P is unstable, because the within-session SD across draws
(~0.67) puts the Monte-Carlo error at 20 draws (~0.15) on a par with the
between-session s.e.m. (~0.17). Draws are applied as a
(n_draws x n_ripples) selection matrix, so 1000 is one matmul -- use 1000.

**Draws are now shared across windows**, so ripple-minus-surround is paired
draw by draw rather than combining two independently subsampled analyses.

**New control: the phase contrast is localised to the ripple.** Recomputed in
the far field it vanishes in pooled HC (+0.05, P = 0.81) and aHP (-0.13,
P = 0.45). ⚠ It does NOT vanish in mHP (+0.41, P = 0.21), so for mHP alone the
contrast is not demonstrably ripple-localised. Say so.

**⚠ The two specificity controls still disagree** (see 2026-09-29 (b)): the
surround control is significant during encoding, the matched-flank control --
which holds the occupied square fixed -- is not. The reported claim is "peaks
at the ripple relative to the surrounding second".

## 2026-09-29 (b) — phase split at 1000 permutations; the encoding signal IS
## ripple-specific by the surround test

Re-ran `timecourse_by_phase` with **N_PERM 200 -> 1000** and the cluster test at
**10 000** sign-flip permutations. Results in
`swr/timecourse_by_phase_2026-09-29/`, write-up in
`data/final_results/ripple_analysis/RESULTS_phase_split.md`.

Going 200 -> 1000 moved nothing (pooled encoding +0.558 -> +0.580; contrast
+0.462 -> +0.449), so the estimates were already permutation-stable.

Pooled hippocampus, n = 51, ripple counts matched within session:

| curve | z at ripple | t(50) | P | cluster | cluster P |
|---|---|---|---|---|---|
| encoding | +0.58 | 4.09 | 1.6e-4 | -75..+200 ms | **0.0003** |
| once known | +0.13 | 1.25 | 0.22 | - | 0.38 |
| encoding - once known | +0.45 | 2.70 | **0.0096** | -25..+50 ms | **0.041** |
| encoding vs own surround | +0.47 | 3.57 | **8.0e-4** | -25..+50 ms | **0.010** |
| once known vs own surround | +0.04 | 0.43 | 0.67 | - | 0.43 |

**New and it corrects yesterday's caveat:** the surround control (ripple minus
the session's own |t| >= 350 ms far field) was null with all ripples pooled
(P = 0.19). Restricted to encoding it is **significant** (P = 0.0008, cluster
P = 0.010) and flat once known. Pooling the phases was diluting it.

**⚠ The two specificity controls disagree.** Surround says specific; the
matched-flank control (duration-matched windows drawn from inside the ripple's
own occupancy interval, so the square is held fixed) is null in encode too.
The far field often falls in a different occupancy interval, so it partly
rewards the signal for tracking where the subject is; the flank is the stricter
test. Report both; claim "peaks at the ripple relative to the surrounding
second", not "ripples carry it preferentially".

**Reproducibility bug found and fixed.** `summarise` used one shared generator
across the curve loop, so each curve's cluster p depended on which curves ran
before it -- the pooled contrast came out 0.048, then 0.051, for identical
data. Now seeded per (ROI, curve) with `default_rng([SEED, ri, ci])`; verified
bit-identical across two consecutive runs. Anything quoted from the
`timecourse_by_phase_2026-09-28` directory is superseded.

Also refactored: offsets are streamed rather than stacked (the largest session
has 7633 ripples, which at 1000 perms x 41 offsets would have been ~2.5 GB),
and the cluster stats moved into `summarise(d)` so tests can be re-run or
extended without repeating the spike-counting loop.

## 2026-09-29 (d) — Do the phase comparisons need correcting for ripple count?

`scripts/swr_hfb_ripple_count_control.py`. SK asked. Short answer: **no bias
correction is needed, but the POWER asymmetry is real, large, and cuts in
opposite directions for the two findings.**

### The counts are very unequal

| condition | first traversal | after first | ratio | encoding | after encoding | ratio |
|---|---:|---:|---:|---:|---:|---:|
| correct | 3750 | 20069 | 0.19 | 5105 | 18714 | 0.27 |
| incorrect | 5963 | 3244 | 1.64 | 8306 | 901 | 6.80 |
| navigation | 4810 | 28838 | 0.17 | 9290 | 24358 | 0.32 |

### 1. No bias. A mean over n ripples is unbiased in n, and that holds here

Spearman(coupling, n_ripples) per ROI x phase x condition: **0 of 44 cells
significant at q < 0.05** (smallest raw p = 0.062). The value does not track
ripple count, so there is nothing to correct in the estimate itself.

### 2. But the noise is almost entirely ripple count

log(flank SD) ~ log(n_ripples), slope per ROI x condition:

| ROI | correct | incorrect | navigation |
|---|---|---|---|
| MedialFrontal | -0.455±0.027 | -0.439±0.033 | -0.471±0.023 |
| lOFC | -0.400±0.041 | -0.522±0.057 | -0.498±0.030 |
| mPFC | -0.451±0.027 | -0.405±0.035 | -0.432±0.029 |

All nine slopes sit on the -0.50 that pure sampling noise predicts, r between
-0.69 and -0.86, every p < 1e-4. So a phase with a fifth of the ripples really
does give a ~2.2x noisier per-session estimate, and every test on it is
correspondingly weaker.

### 3. Which way it cuts — this is the part that matters

Of 17 phase contrasts, 12 have the effect pointing at the phase that ALSO has
more ripples (counts could explain them) and 5 have it pointing the other way
(counts work against them). The five that survive the concern:

  * **lOFC correct, first traversal > after first** (ratio 0.19)
  * **lOFC correct, encoding > after encoding** (ratio 0.27)
  * MedialFrontal correct, first traversal > after first (ratio 0.19)
  * lOFC incorrect, after first > first traversal (ratio 1.70)
  * MedialFrontal incorrect... (ratio 1.64)

**This changes how yesterday's cluster result should be read.** The claimed
lOFC double dissociation was: correct uncovers cluster in both EARLY panels,
navigation in both LATE panels. Those two halves are not equally safe:

  * the lOFC correct clusters (first traversal p = 0.018, encoding p = 0.044)
    occur in phases with 4-5x FEWER ripples than the late panels where they
    are absent. Count differences make that harder to find, not easier;
  * the lOFC navigation clusters (after first p = 0.0010, after encoding
    p = 0.0035) occur in the phases with 6x and 3x MORE ripples. Those could
    be power alone and should not be reported as a phase preference.

Same for MedialFrontal: its navigation clusters are all in high-count phases.

### 4. Controlling for count does not move the contrasts

Count-matched subsets (sessions within 2x, n = 10-24) and the
difference-at-equal-counts intercept from diff ~ log(count ratio) both leave
every phase contrast non-significant, and mostly shift them by less than their
own SEM. Nothing was being hidden by the count imbalance.

### What this cannot do

A true count-matched test would SUBSAMPLE ripples so both phases average over
the same number. The saved traces are already per-condition means and the
individual epochs are not in the bundle, so that needs a re-extraction with
subsampling. Worth doing if any phase contrast becomes load-bearing; not worth
it while none of them is significant.

Output `hfb_ripple_count_control_2026-09-29/`.

## 2026-09-29 — the peri-ripple location time course, split by phase

`mc/analyse/swr_explore/timecourse_by_phase.py`, run via
`scripts/swr_content_explore.py phase_timecourse`; figure 19; results in
`swr/timecourse_by_phase_2026-09-28/`.

The reported panel's estimator (41 offsets, -500..+500 ms, each ripple's own
duration as the window, session as the unit of inference, cluster-corrected by
sign-flipping whole session time courses) run separately inside `encode` and
`execute`, with ripple counts matched within session between phases (20 draws,
averaged) because z carries a sqrt(n) term and execute has ~2x the ripples.
The same ripple set is used at every offset, so a change across the time course
is a change in the signal and not in which ripples contributed.

| ROI | curve | z at peak | P | cluster P |
|---|---|---|---|---|
| HC pooled | encode | +0.558 | 0.0004 | **0.003** |
| HC pooled | execute | +0.096 | 0.37 | 0.36 |
| HC pooled | encode - execute | +0.462 | **0.008** | **0.043** |
| mHP | encode | +0.564 | 0.013 | **0.005** |
| mHP | execute | +0.165 | 0.28 | 0.24 |
| mHP | encode - execute | +0.399 | 0.16 | 0.30 |
| aHP | encode | +0.400 | 0.008 | 0.14 |
| aHP | execute | +0.072 | 0.47 | 0.44 |
| aHP | encode - execute | +0.328 | 0.052 | 1.00 |

So the phase difference is now significant **in the temporal frame too**, in
pooled hippocampus, both at the ripple peak and cluster-corrected. This is the
time-resolved counterpart of the count-matched contrast logged yesterday and
agrees with it (+0.46 here vs +0.41 there).

**Figure fix that affects every PDF this module has ever written:**
`scripts/swr_content_figures.py` never set `pdf.fonttype`, so matplotlib
embedded Type 3 fonts and Illustrator opened the text as outlines/garbage. Now
set to 42 (TrueType) with `svg.fonttype: "none"`, verified: the PDFs carry
`/FontFile2` and zero `/Type3`. Regenerate any panel that is still in use.

**⚠ Figure 4h's legend is swapped.** Gold is `#CCB178` = HC_mid = mHP and is
the curve carrying the significant cluster (P = 0.019); the panel labels it
aHP. The Results text is right, the legend is not.

**Draft-paragraph review** in
`data/final_results/ripple_analysis/RESULTS_ripple_location_draft_review.md`:
the numbers check out, but "these ripples contained the uncovered location" is
a conjunction that was never tested and that the reward/non-reward split
contradicts (mHP encode: reward +0.167 vs non-reward +0.675, difference
-0.507, P = 0.066), and the null ripple-specificity control has to be stated.

## 2026-09-28 (f) — the current-square result split by phase, and the count bias in z

`mc/analyse/swr_explore/location_by_phase.py`, run via
`scripts/swr_content_explore.py location_by_phase`; figure 18; results in
`swr/location_by_phase_2026-09-28/`.

**Phase definition** (SK's): `encode` = every traversal up to but *excluding*
the first error-free ABCD loop; `execute` = from that loop onwards.
`first_traversal` is reported as an extra view and is **99.95% contained in
`encode`** (71% of it) — it is a subset, not a third phase, so the two are
never pooled and the primary contrast is encode vs execute.

**Result.** Location evidence in ripples (C0) is carried by encoding:
HC_all +0.55 (P = 0.0003) in encode vs +0.22 (P = 0.13) in execute; same
direction in HC_anterior and HC_mid. Ripple-specificity (C1, ripple minus its
matched flank) is null in every phase, as everywhere else.

**Count bias in z — this changed the conclusion.** `z` is
(observed − mean(null)) / SD(null) and the null SD scales as 1/√n, so z carries
a √n term and is **not comparable across subsets of different size**. Execute
has ~2× the ripples per session (median 704 vs 331) purely because the phase is
longer, which inflates its z. SK asked whether ripple *rate* biases this; the
premise was backwards — rate is *lower* in encode (0.482 vs 0.510 Hz,
p = 0.0009) — but the *count* bias is real and works against the effect.

Sub-sampling both phases to the same ripple count (20 subsamples, seeded per
session × ROI) made every contrast **larger**, as predicted:

| ROI | contrast | unmatched | count-matched | P | 95% CI |
|---|---|---|---|---|---|
| HC_all | encode − execute | +0.333 | **+0.408** | 0.016 | [+0.08, +0.74] |
| HC_all | first trav − execute | — | **+0.405** | 0.014 | [+0.08, +0.73] |
| HC_anterior | encode − execute | +0.246 | **+0.364** | 0.025 | [+0.05, +0.68] |
| HC_anterior | first trav − execute | — | **+0.358** | 0.012 | [+0.08, +0.63] |
| HC_mid | encode − execute | +0.367 | +0.467 | 0.094 | [−0.09, +1.02] |

All C1 (ripple-specific) contrasts stay null, P = 0.11–0.60.
**Caveat:** the contrast family is 6 (3 ROIs × 2 measures; the two phase pairs
are nested, not independent), so Bonferroni is 0.0083 and **none of these
survive it**. Report as a nominally significant, directionally consistent
phase difference, not as an established one.

**Subtype splits are all null.** Reward vs non-reward and
common-trajectory vs off-route separate nothing in any phase or ROI (the
largest is HC_mid encode reward − nonreward, −0.507, P = 0.066, in the
*opposite* direction to the obvious prediction). ⚠ Off-route ripples nearly
vanish during execution (357 in total across all sessions, 7–10 sessions with
any) — that cell is empty, not null, and must not be read as a negative result.


## 2026-09-28 (e) — 2 cm panels for lOFC too, with clusters and the ripple bar

`swr_hfb_panels_2cm.py` now takes `--rois` (default MedialFrontal + lOFC),
marks cluster-corrected significance, and draws the mean ripple extent.
Output goes to `<mode>_2cm/<ROI>/`, so `axes_2cm/lOFC/first_traversal.pdf`.

  * the saved region index carries only `roi_family` plus the derived
    MedialFrontal label, so lOFC did not exist there; both the region and the
    condition index are now relabelled through
    `hfb_roi_partition/derivation_assignment.csv`;
  * clusters: `swr_sakon.cluster_perm_time`, the same function, sign-flip null,
    n_perm = 2000, seed 42, tested against ZERO because the curve is already
    real-minus-shifted-null;
  * ripple extent: 67.8 ± 0.4 ms (100737 ripples, 61 sessions), drawn as peak
    ± duration/2 with SE end caps;
  * both annotations sit BELOW the traces and only conditions that actually
    have a cluster get a row -- reserving a row per condition regardless cost
    40% of a 2 cm panel to draw nothing.

**Every surviving cluster:**

| ROI | panel | condition | window | p | n |
|---|---|---|---|---|---:|
| MedialFrontal | region | — | −190 to +370 ms | 0.0005 | 53 |
| MedialFrontal | first traversal | correct | −130 to +10 ms | 0.038 | 51 |
| MedialFrontal | after first | correct | −120 to +100 ms | 0.0015 | 53 |
| MedialFrontal | after first | navigation | −260 to +350 ms | 0.0005 | 53 |
| MedialFrontal | encoding | navigation | +40 to +280 ms | 0.006 | 52 |
| MedialFrontal | after encoding | correct | −130 to +110 ms | 0.0005 | 53 |
| MedialFrontal | after encoding | navigation | −260 to +350 ms | 0.0005 | 53 |
| **lOFC** | region | — | **−260 to +150 ms** | 0.0005 | 21 |
| **lOFC** | first traversal | **correct** | **−190 to +40 ms** | **0.0175** | 19 |
| lOFC | after first | navigation | −200 to +50 ms | 0.0010 | 21 |
| **lOFC** | encoding | **correct** | **−180 to +30 ms** | **0.0435** | 19 |
| lOFC | after encoding | navigation | −190 to +50 ms | 0.0035 | 21 |

**The cluster test finds what the peri-window average could not.** lOFC correct
uncovers clusters in BOTH early panels -- first traversal (p = 0.018) and
encoding (p = 0.044) -- and in neither late panel, while lOFC navigation does
the exact opposite: it clusters in both LATE panels and in neither early one.
That is a double dissociation in the sign of the timing, and it is invisible in
the +-250 ms mean because the lOFC reward rise starts ~190 ms BEFORE the ripple
and is over by +40 ms, whereas the navigation cluster straddles it.

Incorrect uncovers cluster nowhere, in either region.

MedialFrontal shows no such split: correct uncovers cluster in three of four
panels and navigation in three of four, with wide overlapping windows. Consistent
with 2026-09-28 (d) -- medial frontal coupling is flat across task phase, lOFC's
is not.

⚠ These cluster p-values are per panel and are NOT corrected across the 12
panels or across ROIs. Two of the lOFC clusters sit at p = 0.018 and p = 0.044;
neither would survive a Bonferroni over 12.

Output `hfb_panels_2cm_2026-09-28/`.

## 2026-09-28 (d) — How the ROI magnitudes are compared: three estimators, noise, phase profile

`scripts/swr_hfb_roi_compare.py`. SK asked why regions are being compared with a
paired test when the electrodes rarely co-occur, whether lOFC's larger effect is
just noise, and whether medial frontal coupling really is carried by the first
traversal. Its `_pooled` reproduces the published `pooled_cond_curves` exactly at
session level (max |diff| 0.000e+00) and adds a derivation-level grouping the
imported one cannot do, because it looks the ripple weight up with `int(unit)`.

### Coverage: the paired test was throwing away most of the data

lOFC x mPFC co-occur in 15 of 41 sessions, 10 of 29 subjects. Pairwise session
overlap: lOFC-mOFC 12, lOFC-vlPFC 13, mPFC-vlPFC 16, FrontalPole-vlPFC 6.

### lOFC vs mPFC, correct uncovers, three estimators

| phase | paired (n) | unpaired (n) | mixed, per-derivation (n) |
|---|---|---|---|
| all | -0.0018 p=0.671 (15) | -0.0026 p=0.460 (35/21) | -0.0009 p=0.751 (151d) |
| first traversal | -0.0132 p=0.121 (13) | **-0.0126 p=0.059** (34/19) | -0.0075 p=0.159 (138d) |
| after first | -0.0011 p=0.800 (15) | -0.0013 p=0.722 (35/21) | -0.0004 p=0.907 (151d) |
| encoding | -0.0151 p=0.073 (13) | **-0.0127 p=0.059** (34/19) | -0.0071 p=0.136 (138d) |
| after encoding | -0.0006 p=0.880 (15) | -0.0013 p=0.721 (35/21) | -0.0005 p=0.865 (151d) |

All three agree in sign and roughly in size; none reaches p < 0.05 and none
survives BH (smallest q by estimator: paired 0.472, unpaired 0.491, mixed
0.100). Using all the data does NOT rescue the effect -- the unpaired estimate
is slightly SMALLER than the paired one, and the mixed model, which is the
honest pseudopopulation version, halves it again. The paired test was not what
was hiding it.

### Is lOFC just noisier? No, and it cannot be a sparseness artefact

HFB is z-scored per sub-band per channel on artifact-free samples
(`swr_extract_hfb`) before any of this, so a denser or larger raw
high-frequency signal cannot inflate the number; the scale is removed per
derivation. Correct uncovers, all phases:

| ROI | sess | der | der/sess | mean | SD across sess | SEM | flank noise |
|---|---:|---:|---:|---|---|---|---|
| MedialFrontal | 53 | 179 | 3.38 | +0.0063 | 0.0093 | 0.0013 | 0.0167 |
| mPFC | 35 | 79 | 2.26 | +0.0051 | 0.0110 | 0.0019 | 0.0181 |
| lOFC | 21 | 72 | 3.43 | +0.0077 | 0.0131 | 0.0029 | 0.0164 |
| mOFC | 18 | 23 | 1.28 | +0.0107 | 0.0174 | 0.0041 | 0.0266 |
| FrontalPole | 16 | 28 | 1.75 | +0.0011 | 0.0127 | 0.0032 | 0.0237 |
| vlPFC_IFG | 19 | 22 | 1.16 | +0.0054 | 0.0160 | 0.0037 | 0.0252 |

lOFC's flank noise (0.0164) is the LOWEST of any ROI, slightly below medial
frontal -- so it is not a noisier signal. Its SD across sessions is 19% higher
than mPFC's (0.0131 vs 0.0110) and its n is 40% lower (21 vs 35); the SEM ratio
follows almost entirely from n. lOFC fails on SAMPLE SIZE, not on noise and not
on normalisation.

### Medial frontal is NOT carried by the first traversal

All six pairwise phase contrasts, correct uncovers, MedialFrontal:

| A | B | A | B | A-B | p | q |
|---|---|---|---|---|---|---|
| first traversal | encoding | +0.0073 | +0.0050 | +0.0023 | 0.049 | 0.813 |
| after first | after encoding | +0.0062 | +0.0066 | -0.0004 | 0.147 | 0.813 |
| first traversal | after first | +0.0073 | +0.0058 | +0.0015 | 0.648 | 0.877 |
| encoding | after encoding | +0.0050 | +0.0063 | -0.0013 | 0.652 | 0.877 |
| first traversal | after encoding | +0.0073 | +0.0063 | +0.0011 | 0.742 | 0.924 |
| after first | encoding | +0.0058 | +0.0050 | +0.0009 | 0.755 | 0.924 |

The first traversal is numerically the largest, by 0.0011-0.0023 on a +0.0063
baseline -- a 20-35% margin with p >= 0.049 and q >= 0.81. **Encoding is the
LOWEST of the four phases**, and after-encoding (+0.0066) beats encoding
(+0.0050), which is the opposite of a learning-driven account. Medial frontal
correct-uncover coupling is flat across the task; it is significant in every
phase because it is present in every phase, not because one phase carries it.

lOFC's own profile is more graded (first traversal +0.0151, encoding +0.0131,
after-first +0.0084, after-encoding +0.0091) but no contrast approaches
significance (min p = 0.196).

Output `hfb_roi_compare_2026-09-28/` — `roi_diagnostics.csv`,
`roi_pair_three_estimators.csv`, `phase_contrasts_all_pairs.csv`,
`three_estimators.pdf/png`.

## 2026-09-28 (c) — 2 cm publication panels, exclusive-boundary re-run, ROI x condition

### Exclusive phase boundary, re-extracted

`swr_ripple_hfb_conditions.py run --boundary=exclusive` over all 61 sessions ->
`ripple_hfb_conditions_exclusive_2026-09-28/`. SK's "encoding" means everything
up to but EXCLUDING the first fully correct ABCD loop; `ripples.stage_of_repeat`
-- the project default and what the 2026-09-21 run used -- puts that repeat on
the LEARNING side (1432 repeats, 39% of that side). The two are not
interchangeable, so the encoding panels needed their own extraction.

### `swr_hfb_panels_2cm.py` — five panels at 2 x 2 cm, -500 to +500 ms

medial_frontal (53 sessions), first_traversal, after_first, encoding,
after_encoding. 7 pt type, 1.1 pt lines. CLAUDE.md's 9 pt minimum does not fit:
at 9 pt the labels are wider than a 2 cm axis. Written in two conventions --
`axes_2cm/` (axes box exactly 2 cm, labels outside) and `total_2cm/` (whole
canvas 2 cm, data area ~1.1 cm) -- plus `_labelled` proofs and a standalone
`legend_conditions`. Vector .pdf + 600 dpi .png.

**A check caught a wrong assumption and it is worth keeping.** The code
originally took first_traversal/after_first from the INCLUSIVE run, on the
argument that the boundary only moves one repeat per grid between `plan` and
`execute` and so cancels when they are pooled. An invariance check says
otherwise: first_traversal is bit-identical (0.000e+00), after_first is NOT
(max 1.77e-2). Two causes, both in `pooled_cond_curves`, neither in the
boundary logic:
  * it weights each derivation's trace by a SESSION-level ripple count, not by
    the ripples that derivation contributed to that condition, so moving
    ripples between plan and execute shifts the pooled mean (mean |diff| 5e-4
    against a ~1.6e-2 signal; up to 1.3e-2 in one session);
  * MIN_RIPPLES is per (derivation, condition), so a few error derivations
    clear it under one boundary and not the other (error_plan 124 vs 126,
    error_execute 83 vs 80).
All four condition panels are therefore drawn from the EXCLUSIVE run. The check
runs on every invocation and its numbers go into result.json.

### `swr_hfb_roi_conditions.py` — is lOFC reward-linked, and does it differ from mPFC?

Maps the conditions traces onto the partition ROIs and splits by condition and
phase. Statistic identical to the coupling table (peri +-250 ms of the
real-minus-null, flank-normalised curve), smoothing OFF.

⚠ **Frontal only.** The conditions extraction covered 310 derivations, all in
the mPFC/mOFC/MedialFrontal families. MTG, STG and Auditory have no condition
traces at all -- a widened re-extraction, not a re-read.

Coupling, all phases pooled, mean +- SEM over sessions (* q < 0.05 vs 0, BH):

| ROI | correct | incorrect | navigation |
|---|---|---|---|
| mPFC (n35) | +0.0051±0.0019* | +0.0047±0.0030 | +0.0057±0.0014* |
| lOFC (n21) | +0.0076±0.0029 | +0.0041±0.0046 | +0.0074±0.0025* |
| mOFC (n18) | +0.0107±0.0041 | +0.0128±0.0044* | +0.0161±0.0053* |
| FrontalPole (n16) | +0.0013±0.0032 | +0.0073±0.0043 | +0.0084±0.0029 |
| vlPFC_IFG (n19) | +0.0054±0.0037 | +0.0096±0.0049 | +0.0021±0.0025 |

**No, lOFC is not more reward-linked.** reward-minus-navigation in lOFC is
+0.0002 (all, p = 0.955), +0.0066 (encoding, p = 0.408), -0.0017 (after,
p = 0.629). The INTERACTION against mPFC -- the only test that licenses the
claim -- is -0.0003 (all, p = 0.957) and +0.0085 (encoding, p = 0.497); every
interaction in the table has q >= 0.84. No phase contrast survives either
(all q >= 0.62).

**Navigation presses couple as strongly as rewards everywhere**, and are the
significant cell in mPFC, lOFC and mOFC. Consistent with 2026-09-23: navigation
is not a null control, and none of this coupling is reward-specific.

The largest single cell is lOFC correct-uncovers during encoding
(+0.0134±0.0058) against mPFC's +0.0003±0.0028, which is what prompted the
question -- but the paired interaction has 13 shared sessions and p = 0.497.
Underpowered, not absent; it is the one worth a pre-registered test.

### Added: ROI-vs-ROI within condition, and the full answer to "what is bigger than what"

The design was missing its fourth cell: "which region couples most to the same
event" is an ROI x ROI contrast within a condition, not the within-ROI
condition contrast and not the interaction. 80 pairs added.

Five families of test, all BH-corrected within family:

| family | n tests | survive q < 0.05 | min q | min raw p |
|---|---:|---:|---:|---:|
| vs ZERO | 45 | **10** | 0.008 | 0.0002 |
| condition vs condition, within ROI | 39 | 0 | 0.917 | 0.082 |
| ROI vs ROI, within condition | 80 | 0 | 0.579 | 0.0072 |
| interaction vs mPFC | 20 | 0 | 0.844 | 0.183 |
| encoding vs after-encoding | 12 | 0 | 0.619 | 0.103 |

**Nothing is significantly bigger than anything else.** Every comparison of one
cell against another fails correction; the only surviving effects are cells
against zero.

The ten that clear zero, and what they are:

| ROI | phase | condition | n | coupling | q |
|---|---|---|---:|---|---:|
| mPFC | all | navigation | 35 | +0.0057±0.0014 | 0.008 |
| mPFC | after | navigation | 35 | +0.0067±0.0017 | 0.008 |
| mOFC | after | navigation | 18 | +0.0153±0.0043 | 0.034 |
| lOFC | all | navigation | 21 | +0.0074±0.0025 | 0.042 |
| lOFC | after | navigation | 21 | +0.0088±0.0029 | 0.042 |
| FrontalPole | after | navigation | 16 | +0.0118±0.0036 | 0.042 |
| mOFC | all | navigation | 18 | +0.0161±0.0053 | 0.042 |
| mOFC | encoding | incorrect | 18 | +0.0145±0.0047 | 0.042 |
| mPFC | all | correct | 35 | +0.0051±0.0019 | 0.044 |
| mOFC | all | incorrect | 18 | +0.0128±0.0044 | 0.044 |

**Seven of the ten are NAVIGATION PRESSES.** Only one correct-uncover cell
survives (mPFC, all phases). The condition that most reliably shows
ripple-locked frontal HFB is the one with no feedback in it at all. That is the
2026-09-23 conclusion again, now per ROI: this coupling is not reward-specific,
and navigation is not a null control.

The largest raw ROI difference is lOFC > vlPFC_IFG for navigation (+0.0093,
p = 0.0072, q = 0.579 over 80 tests) -- the only comparison anywhere under
p = 0.01, and it is about key presses, not reward.

### Corrected: the condition table was missing the first traversal AND MedialFrontal

Two omissions SK caught from the figure, both mine, both changing conclusions:

1. `PHASES` held only all / encoding / after_encoding. `first_traversal` and
   `after_first` -- the very splits the 2 cm panels plot -- were never tested,
   so the first-traversal effect visible in the figure had no number anywhere.
2. `MedialFrontal` was absent. The 2 cm condition panels plot it (roi=MEDIAL in
   `pooled_cond_curves`), and the table listed only partition ROIs, so figure
   and statistics described different regions. It is now included, first,
   separated by a dotted line and excluded from the ROI-vs-ROI contrasts
   because it is a coordinate cut across mPFC+mOFC.

The table is now 90 cells (was 45) and **32 survive BH vs zero (was 10)**.

**Correct uncovers, the cells that now exist and survive:**

| ROI | phase | n | coupling | q |
|---|---|---:|---|---:|
| lOFC | first traversal | 19 | +0.0151±0.0055 | 0.042 |
| MedialFrontal | first traversal | 51 | +0.0073±0.0028 | 0.038 |
| lOFC | all | 21 | +0.0077±0.0029 | 0.044 |
| MedialFrontal | all | 53 | +0.0063±0.0013 | 0.0003 |
| MedialFrontal | after encoding | 53 | +0.0066±0.0015 | 0.0007 |
| MedialFrontal | after first | 53 | +0.0062±0.0015 | 0.002 |
| lOFC | after encoding | 21 | +0.0072±0.0028 | 0.046 |
| mPFC | after encoding | 35 | +0.0059±0.0023 | 0.044 |
| mPFC | all | 35 | +0.0051±0.0019 | 0.036 |

Just under threshold: mOFC first traversal +0.0175±0.0073 (q = 0.073), mOFC
encoding +0.0163±0.0069 (q = 0.075), lOFC encoding +0.0131±0.0058 (q = 0.088).

**The earlier "seven of ten are navigation" headline was an artefact of the
missing phases.** With first_traversal in, correct uncovers carry 9 of the 32
surviving cells, and the two largest correct-uncover effects anywhere are
lOFC and mOFC on the FIRST TRAVERSAL.

**Still nothing survives any comparison.** lOFC vs mPFC for correct uncovers:
encoding -0.0151±0.0077 (p = 0.073, q = 0.731), first traversal -0.0132±0.0079
(p = 0.121). Within lOFC, reward-minus-error at first traversal +0.0129±0.0074
(p = 0.098). Interaction lOFC vs mPFC on (reward-error) at first traversal
+0.0169±0.0081 (p = 0.060, q = 0.957). All on 13-19 shared sessions: the
pattern is consistent in sign and size across three independent framings and
underpowered in each.

Figure redrawn: 5 phases, MedialFrontal separated, * marking q < 0.05 vs zero,
y-limits from the 2nd-98th percentile of mean±SEM so the 3-session error cells
do not flatten everything else.

Output `hfb_roi_conditions_2026-09-28/roi_vs_roi_within_condition.csv`.

Output `hfb_panels_2cm_2026-09-28/`, `hfb_roi_conditions_2026-09-28/`.

## 2026-09-28 (c) - Rate-map PAIR examples: making the per-lag consistency visible

`scripts/plot_ratemap_pair_examples.py` (new). Visualisation only, no new
statistic. Companion to `plot_ratemaps_from_per_lag_encoding.py`, which shows
all 8 configurations at 2 lags; this shows 2 configurations at 3 lags so the
eye can do the correlation itself.

**The display trick.** Each 3x3 panel is the lag-shifted rate map converted to
a dwell-weighted z-score, then affinely rescaled to its own [min, max], drawn
on RdBu_r with the white point at 0.60 of that range. A Pearson r is exactly
invariant to a per-map affine rescaling, so this cannot move any reported
number -- it only removes the between-task differences in mean and scale that
r already ignores and the eye does not. The lifted white point sends everything
below the top of the range to blue, so a well-aligned pair reads as "the same
one or two red squares, the rest blue".

**BUG THAT COST TWO ROUNDS: the pairwise panels were stricter than the
analysis.** The first version greyed out any location with shared dwell
< MIN_DWELL_BINS = 25 in BOTH single tasks, which caps a task PAIR at 6 of 9
locations anywhere in the dataset, and made every example look sparse. But the
CV never correlates two single tasks: each fold correlates one held-out task
against the dwell-weighted POOL of all the others, so the pooled side always
has ample dwell and the 25-bin rule effectively binds on one task only. Checked
directly: the real CV fold keeps mean 5.4 of 9, max 8. Thin locations are
handled in this design by the min(dwell_a, dwell_b) WEIGHTS -- a 10-bin
location contributes almost nothing -- not by exclusion. So pairwise panels now
use a separate `PAIR_MIN_DWELL_BINS = 10`, printed in every figure footnote.
Shared-location counts per task pair, measured over all 1259 pairs:

| threshold | mean | max | pairs >=7 | >=8 | =9 |
|---|---|---|---|---|---|
| 25 bins (old) | 3.2 | 6 | 0 | 0 | 0 |
| 15 bins | 4.5 | 9 | 89 | 15 | 4 |
| 10 bins (now) | 5.0 | 9 | 211 | 54 | 16 |
| 1 bin | 5.3 | 9 | 288 | 87 | 34 |

**Selection is on a VISUAL-match score, not on r.** Pearson r does not care
WHICH square is red and the eye does, so r alone picks pairs that look
unconvincing (15_21: r = 0.80 but 3 red squares against 1, peaks in different
places -- rejected on sight). Candidates are scored on: same peak square in
both tasks, small overlapping red set (Jaccard above the colormap white point),
high pairwise r, the peak MOVING at both the contrast and the control lag, and
>= 7 of 9 locations filled. Searched exhaustively over every (cell x task-pair)
across mPFC, HC_mid and HC_anterior -- 2610 combinations.

**Chosen examples. All three are now individually significant**, which the
6/9-capped search could not achieve:

| panel | cell | tasks | cov | pair r at key lag | contrast | control 270 deg | cell CV r, p |
|---|---|---|---|---|---|---|---|
| a | 43_12-12-mRF2aCa03-ACC (mPFC) | cfg 6, 7 | 6/9 | 0.94 at 30 deg | -0.52 at 0 deg | -0.70 | 0.20, p = 0.108 |
| b | 27_01-01-mLF2aCa03-ACC (mPFC) | cfg 1, 4 | 7/9 | 0.92 at 60 deg | -0.25 at 0 deg | +0.21 | 0.28, p = 0.013 |
| c | 35_06-06-mRT2bHb06-HC (HC_mid) | cfg 2, 5 | 8/9 | 0.91 at 0 deg | -0.07 at 60 deg | -0.38 | 0.37, p = 0.011 |

Panel b is the cell and the exact configuration pair the user had picked by eye
before any of this ran, which is a decent sanity check on the visual score.
**Panel a took three passes, and the third one found the bug in my own search.**
The first pick (42_14 cfg 1/2, r = 0.67) read as the weak one of the three. A
second pass over the >= 7/9 pool found nothing meeting the bar b and c meet
(perfect red-set match, <= 2 red squares, r >= 0.75, both control rows quiet):
zero hits, with or without the significance filter. Only 8 of the 676 pairs at
>= 7/9 have a perfect red-set match at 30 deg at all, and EVERY one of them has
an aligned control row too.

That is structural, not bad luck, and per_lag_encoding's own docstring already
says why: the location 30 deg ahead coincides with the CURRENT location on ~39%
of bins (~3.5x chance), decaying to chance by +-90 deg. Measured here across all
676 mPFC pairs:

| lag | corr( r at that lag , r at 0 deg ) |
|---|---|
| 30 deg | **+0.243** |
| 60 deg | +0.029 |
| 90 deg | +0.142 |
| 270 deg | +0.075 |

Among pairs with a crisp red-set match at the key lag, 42% have r > +0.2 at
0 deg for 30 deg, against 25% at 60 deg and 28% for HC at 0 deg. A crisp 30 deg
key row drags the 0 deg row up with it. This is the same trajectory
autocorrelation that makes the paper's claim rest on the SHAPE of the lag
profile rather than on any single lag.

**The third pass fixed my own error: I had excluded 6/9 pairs.** The >= 7/9
filter was mine, not the data's, and it was hiding the only pairs that satisfy
the red-square rule at 30 deg. Re-running the mPFC search at >= 6/9 gives 1492
pairs instead of 676.

**The fourth pass fixed the metric.** Jaccard on the red set still let through
a pair (43_05 cfg 1/7, jac 0.67) whose secondary hot square sits middle-LEFT in
one task and middle-RIGHT in the other -- it reads as MIRRORED even though
r = 0.93. Two columns now measure that directly:

* `disc` = number of squares that are red in one task but not the other. This
  is what the eye calls mirroring; it must be **0**.
* `rho`  = Spearman rank agreement over ALL contributing squares, so a swapped
  secondary hot spot is penalised even when it sits below the red threshold.

Applied to all 1492 mPFC pairs, exactly **two** have disc = 0, <= 2 red squares,
both control rows quiet AND an individually significant cell:

| cell | tasks | cov | r at 30 deg | reds | rho | r at 0 deg | r at 270 deg | cell p |
|---|---|---|---|---|---|---|---|---|
| 37_02-02-mRF3bC01-ACC | 6, 7 | 6/9 | 0.90 | 2 | 0.83 | -0.23 | +0.08 | 0.037 |
| 08_01-01-mLF2aCa01-ACC | 1, 5 | 6/9 | 0.81 | 2 | 0.77 | -0.73 | -0.21 | 0.017 |

Dropping the significance requirement adds one more, and it is the cleanest
image of all three: **43_12-12-mRF2aCa03-ACC cfg 6/7** -- r = 0.94, disc = 0,
a SINGLE red square in the same corner of both tasks, both controls strongly
negative (-0.52 and -0.70), but cell p = 0.108. All three were rendered under
`candidates/mPFC_30deg_final/` and the user picked this one, so panel a is NOT
individually significant while b and c are (p = 0.013 and 0.011). The panel
prints its own r and p under the title, as all panels do, and the trade-off is
deliberate: at 30 deg, a non-mirrored key row with two dead control rows exists
only here. The two significant alternatives above remain in CANDIDATES.

Rejected at this step and worth not repeating: 43_05 cfg 1/7 (r = 0.93,
p = 0.004) and 39_06 cfg 1/4 (r = 0.86, 7/9, p = 0.014) both have disc > 0 and
read as mirrored; a high pairwise r does not prevent that, which is the whole
reason `disc` exists.

**Two backup panels, requested separately and kept in CANDIDATES.**

*(d1) mPFC aligned at BOTH 0 and 30 deg, more strongly at 30.* The honest
version of the 30/0 entanglement above: rather than hunting for a dead 0 deg
row, show a cell that carries both and is clearly stronger for the next
action. Criteria: r30 >= 0.70, r0 >= 0.30, gap >= 0.20, zero mirrored squares
at 30 deg, cell significant at 30 deg.

| cell | tasks | cov | r at 30 deg | r at 0 deg | gap | rho at 30 deg | r at 270 deg | cell p |
|---|---|---|---|---|---|---|---|---|
| 34_01-01-mLF1aCb02-ACC | 0, 6 | 7/9 | 0.91 | 0.53 | 0.38 | **0.93** | -0.11 | 0.022 |
| 44_05-05-mLF2aCa06-ACC | 0, 3 | 6/9 | 0.91 | 0.48 | 0.43 | 0.83 | +0.14 | 0.013 |
| 46_01-01-mLF2C02-ACC | 0, 5 | 6/9 | 0.86 | 0.44 | 0.42 | 0.60 | -0.58 | 0.032 |
| 61_21-21-mRF2aC01-ACC | 2, 6 | 6/9 | 0.83 | 0.40 | 0.43 | 0.77 | -0.70 | 0.011 |

*(d2) Hippocampus at 330 deg (= 30 deg in the PAST).* 330 is a PREDICTED
hippocampal lag in per_lag_encoding (`ROI_PREDICTED_LAGS_DEG['HC_mid'] =
(0, 330)`), so this panel tests a real prediction rather than an exploratory
lag. Two consequences for its layout: the contrast row is a FUTURE lag (60)
rather than 0 deg, because 0 and 330 are BOTH predicted for hippocampus and a
cell may legitimately carry both; and the far control is 180 deg rather than
270 deg, which sits only 60 deg from the key lag.

| cell | roi | tasks | cov | r at 330 deg | r at 60 deg | r at 180 deg | cell p |
|---|---|---|---|---|---|---|---|
| 44_19-19-mRT2cHb02-HC | HC_mid | 1, 6 | 7/9 | **0.88** | +0.10 | -0.09 | **0.004** |
| 43_20-20-mRT2cHb06-HC | HC_mid | 0, 7 | **8/9** | 0.79 | -0.61 | -0.20 | 0.024 |
| 42_08-08-mLAHIP7-HC | HC_ant | 1, 5 | 8/9 | 0.93 | **+0.49** | +0.16 | 0.030 |
| 63_11-11-mLT2aHaE06-EC | HC_ant | 3, 5 | 7/9 | 0.74 | +0.25 | -0.04 | 0.020 |

44_19 cfg 1/6 is the pick: a single red square in the same place in both tasks,
both control rows dead, the strongest cell of the set, and a lag profile that
is a clean single spike at 330 deg.

**Two traps found while assembling these, worth not falling into again:**
- `disc = 0` is necessary but NOT sufficient. 39_13 cfg 5/7 has disc = 0 at
  330 deg because both tasks have the same two squares above the white point --
  but their INTENSITIES are swapped (dark red bottom-right vs bottom-left), so
  it still reads as mirrored. Spearman `rho` catches this; the red-set
  comparison alone does not.
- A high pairwise r on a cell with a NEGATIVE cross-validated r is a lucky
  pair, not an effect. 63_13 cfg 5/7 shows r = 0.87 at 330 deg while the cell's
  own CV r at 330 deg is -0.28 (p = 0.97). Always read the cell-level r and p
  printed under the panel title before believing a pair.

**Control lag** fixed a priori at 270 deg (= -90 deg, a quarter trial in the
PAST) for every panel, so the third row is not a per-cell search for the worst
lag. Every panel prints its cell's own cross-validated r and permutation p, and
carries the pair's r at all 12 lags as a profile strip, so the lags not chosen
as rows stay visible.

**Rejected** (kept so the trade-offs are not re-litigated):
- Ranking on pairwise r alone. Produces high-r pairs whose red squares sit in
  different places; they read as unaligned however large r is.
- A shared colour scale across all 6 panels of a figure. The most extreme value
  usually sits in a CONTROL row, which washes the key row out to pale pink.
- 180 deg as the control lag (see above).
- Keeping the 25-bin pairwise rule and accepting 6/9. Forced every example onto
  cells that were positive but NOT individually significant (p = 0.15 / 0.35 /
  0.16); the relaxed rule removed the trade-off entirely.
- Under the relaxed rule, the earlier panel-a candidate 39_04 cfg 0/6 drops
  from r = 0.81 to 0.49 -- the 7th location it gains does not match. It only
  looked good because the harsher rule was hiding that location.

Outputs (three standalone panels + a combined sheet, PDF and PNG, plus
`settings.json`, plus `candidates/<roi>_<lag>/` holding the top 5 per ROI):
`data/ephys_humans/derivatives/group/per_lag_encoding/2026-08-04_07-25-15_reload_from_2026-06-30_18-21-57_relabelled-final/ratemap_pair_examples_<stamp>/`

## 2026-09-28 (b) — Ripple-HFB coupling by ROI, one ROI per derivation

`scripts/swr_hfb_roi_partition.py`. Replaces `swr_hfb_roi_inventory.py`, which
is DELETED along with its outputs: it built overlapping groups and reported a
"31 of 61 sessions" caveat that applied only to a file it did not need.

**All 61 sessions, 42 subject labels, 986 cortical derivations.** Read from the
cluster-built bundle. The 31/61 figure came from `macro_contacts_all.csv`, a
LOCAL rebuild that needs the raw recordings; nothing here touches it.

**Partition.** Each derivation gets exactly one ROI from a three-rung ladder on
its anchor -- Harvard-Oxford cortical (693), Brainnetome (76), FreeSurfer native
(90) -- mapped by an ordered substring rule list. Disjointness is asserted in
code and the audit sums to 986. `roi_label_audit.csv` lists every raw label
next to the ROI it landed in; that audit caught two mapping bugs (MOrG "medial
orbital gyrus" falling through to lOFC, and Unassigned rows carrying no reason).

**Coupling, different-shaft pairs, subject-level, pad 0.25 s:**

| ROI | sess | subj | der | der_ds | coupling | q | vs mPFC | q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lOFC | 21 | 14 | 72 | 72 | +0.0111 | 0.001 | +0.0094 | 0.213 |
| mOFC | 18 | 11 | 23 | 23 | +0.0091 | 0.098 | +0.0019 | 0.741 |
| FrontalPole | 16 | 11 | 28 | 28 | +0.0074 | 0.056 | -0.0042 | 0.562 |
| mPFC | 35 | 25 | 79 | 79 | +0.0050 | 0.007 | — | — |
| MTG | 57 | 40 | 289 | 105 | +0.0046 | 0.001 | -0.0013 | 0.741 |
| Occipital | 9 | 6 | 32 | 32 | +0.0034 | 0.798 | | |
| vlPFC_IFG | 21 | 14 | 24 | 24 | +0.0029 | 0.644 | -0.0022 | 0.753 |
| TempOcc_Fusiform | 26 | 17 | 63 | 58 | +0.0024 | 0.746 | -0.0011 | 0.935 |
| Auditory | 25 | 17 | 56 | 56 | +0.0017 | 0.644 | -0.0031 | 0.741 |
| Parietal | 9 | 6 | 11 | 11 | +0.0015 | 0.872 | -0.0004 | 0.964 |
| ITG | 16 | 10 | 19 | 11 | +0.0012 | 0.872 | | |
| TemporalPole | 9 | 6 | 17 | 16 | +0.0004 | 0.872 | -0.0033 | 0.213 |
| STG | 47 | 30 | 142 | 132 | -0.0003 | 0.872 | -0.0038 | 0.562 |

Insula (3 sessions, 2 subjects) passes the session criterion but not the test
criterion. dlPFC has ONE derivation. Parahippocampal and PCC/precuneus have
none at all. Unassigned = 127 derivations whose anchor is white matter (81) or
labelled Unknown (45); reported, never merged into a region.

**Four ROIs couple at q < 0.05: lOFC, mPFC, MTG (and mOFC at q = 0.098).**
Nothing differs from mPFC -- every paired contrast q >= 0.21. The ordering is
orbitofrontal > medial frontal > lateral temporal > auditory/visual, and the
strongest region is lOFC, not mPFC. With the mOFC label finally split by atlas
rather than by the old neighbourhood rescue, medial OFC is 23 derivations and
lateral OFC is 72 -- the old `mOFC` family was mostly lateral, which is what the
|x| <= 20 mm MedialFrontal rule was working around.

For orientation, the published coordinate-defined MedialFrontal group (179
derivations, |x| <= 20 mm across the mPFC+mOFC families) is +0.0072, n = 35
subjects. It is deliberately NOT a row above: it overlaps several of them.

Note on coverage vs dual use: a probe may carry both a hippocampal and cortical
derivations, but NO CONTACT is ever in two derivations -- the hippocampal
montage runs first and consumes its contacts. `der` vs `der_ds` shows the cost:
MTG has 289 derivations but only 105 different-shaft, because the hippocampal
shafts pass through it.

### Added: MedialFrontal comparison group and the explicit contrast

The `*` in the coupling panel is a BH-corrected two-sided one-sample t against
ZERO -- "is there coupling here" -- not a comparison with anything. It is now
labelled that way on the figure, and a second panel carries the comparison.

`MedialFrontal` (the collapsed |x| <= 20 mm rule across the mPFC+mOFC families,
179 derivations, 53 sessions, 35 subjects, +0.0072, p = 4e-5) is drawn as a
DIAMOND below a divider, not as a partition row: it overlaps mPFC (100% of
mPFC's derivations), mOFC (100%), lOFC (24%) and FrontalPole (11%).

**MedialFrontal minus each ROI, paired on the subjects with both, positive =
MedialFrontal couples more. NOTHING survives correction -- every q >= 0.15.**

| ROI | paired | ovl% | MF − ROI | p | q |
|---|---:|---:|---:|---:|---:|
| ITG | 4 | 0 | +0.0124 | 0.343 | 0.558 |
| TempOcc_Fusiform | 10 | 0 | +0.0072 | 0.328 | 0.558 |
| TemporalPole | 5 | 0 | +0.0065 | 0.106 | 0.344 |
| STG | 23 | 0 | +0.0061 | 0.012 | 0.151 |
| Parietal | 5 | 0 | +0.0049 | 0.488 | 0.634 |
| Auditory | 15 | 0 | +0.0047 | 0.075 | 0.325 |
| Occipital | 4 | 0 | +0.0027 | 0.704 | 0.742 |
| vlPFC_IFG | 14 | 0 | +0.0020 | 0.548 | 0.647 |
| MTG | 20 | 0 | +0.0019 | 0.149 | 0.389 |
| FrontalPole | 11 | 11 | +0.0010 | 0.742 | 0.742 |
| mPFC | 25 | 100 | +0.0007 | 0.433 | 0.626 |
| mOFC | 11 | 100 | −0.0019 | 0.312 | 0.558 |
| lOFC | 14 | 24 | −0.0051 | 0.069 | 0.325 |

STG is the closest (uncorrected p = 0.012, q = 0.151) and would be the one to
pre-register. Overlapping contrasts are drawn hollow on the figure; the mPFC
and mOFC rows are MedialFrontal against groups that contain it and are not
tests. Contrast table in `medialfrontal_vs_roi.csv`.

### Added: averaged traces per ROI (`traces` subcommand)

Look-at-it figures, .jpeg, left open on screen. Per-unit curve is
real-minus-shifted-null baselined to the non-peri band -- the same baseline the
peri/non-peri statistic uses, so the bump height at t = 0 IS the coupling
number. 50 ms smoothing, cosmetic only.

  `roi_traces_<unit>_overlay.jpeg`  all ROIs on one axis, thin lines, opacity
                                    scaled to n so a 2-subject curve cannot be
                                    mistaken for a 35-subject one
  `roi_traces_<unit>_grid.jpeg`     one small panel per ROI, own y-scale, thin
                                    single-unit lines behind mean +- SEM

Written for subject- and session-level. `Unassigned` gets no trace (it is a
bookkeeping bucket, not a region). Grid y-limits are the 2nd-98th percentile of
the single-unit curves: one MedialFrontal subject reaches 0.2, ten times the
mean, and otherwise sets the panel scale and flattens the average to a line.
Nothing is excluded from any average.

The overlay shows what the table says: lOFC and MedialFrontal peak together at
~+0.017 right at the ripple, mPFC about half that, and the temporal/auditory
regions sit in the noise.

Output `derivatives/group/swr/hfb_roi_partition_2026-09-28/`.

## 2026-09-28 — ROI inventory redone as a DISJOINT set (SUPERSEDED by 2026-09-28 (b): script deleted, rebuilt as a true partition)

The 2026-09-27 table was wrong in framing, not in arithmetic: it grouped the
same derivations by an atlas label without removing the ones already claimed by
a region in the analysis, so it reproduced MedialFrontal under other names --
"Paracingulate Gyrus" (100% the same derivations), "Frontal Medial Cortex"
(100%), A13 (100%), A10m (100%) -- and listed them as candidates. Those areas
are what MedialFrontal is MADE of. `swr_hfb_roi_inventory.py` now claims first
and builds candidates only from the remainder, so every row is disjoint and the
composition of each existing region is reported instead.

Claiming order: MedialFrontal, TemporalLateral, Auditory, Visual. mPFC/mOFC are
deliberately NOT claimants -- they are the raw families MedialFrontal was carved
from by |x| <= 20 mm, and letting them claim leaves nothing over and hides the
lateral frontal remainder. A24 is excluded as a post-hoc overlapping band.

Of 768 different-shaft cortical derivations (all 61 sessions, pad 0.25 s):
MedialFrontal 179, TemporalLateral 242, Auditory 63, Visual 153, **pool 131**.

**New disjoint candidates (>= 3 sessions), reference MedialFrontal +0.0072
+- 0.0013 (n = 35 subjects):**

| scheme | ROI | sess | subj | der | effect | q | vs ref | q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ho | Frontal Pole | 15 | 10 | 25 | +0.0070 | 0.044 | -0.0014 | 0.686 |
| ho | Frontal Orbital Cortex | 9 | 8 | 17 | +0.0067 | 0.044 | +0.0015 | 0.686 |
| bn | A11l (lateral OFC) | 22 | 16 | 37 | +0.0098 | 0.002 | +0.0035 | 0.619 |
| bn | A45r | 14 | 9 | 16 | +0.0048 | 0.401 | -0.0012 | 0.841 |
| bn | A44op | 9 | 7 | 12 | +0.0007 | 0.878 | -0.0048 | 0.619 |

Plus four under-powered ones (n < 5 subjects): Insular Cortex, IFG pars tri.,
A12/47o, A12/47l.

**Nothing differs from MedialFrontal.** Every candidate contrast has q >= 0.62.
The lateral frontal remainder -- Frontal Pole / Frontal Orbital / A11l, the same
tissue under two atlases -- locks to ripples about as strongly as medial frontal
does, which is the one substantive result here: the effect is not medial-specific
within frontal cortex. A11l is the best-powered version and the one to
pre-register.

**Never extracted (needs one cluster re-run of the HFB stage,
`--cortical_rois=[...]`, not new patients).** Now split by whether the area
already feeds an ROI above. Genuinely new territory, biggest first: Middle
Frontal Gyrus (24 sessions, 56 grey contacts), Superior Frontal Gyrus (15/27),
Frontal Operculum (8/8), Parahippocampal Gyrus ant (8/6) + post (6/8), IFG pars
opercularis (6/10), Precentral (6/18), PCC (5/1), Postcentral (5/20). Insular
Cortex (19/64) and Frontal Pole (27/111) would DEEPEN candidates that already
exist rather than add a region.

⚠ Those session counts, and only those, are a LOWER BOUND: they come from the
local `macro_contacts_all.csv`, which covers 31 of the 61 sessions, so the true
values are roughly double. Every other table in this analysis is all 61 sessions.

Output `derivatives/group/swr/hfb_roi_inventory_2026-09-28/` — `ROI_INVENTORY.md`,
`roi_inventory.csv`, `roi_composition.csv`, `roi_not_extracted.csv`,
`result.json`, `roi_inventory_{ho,bn}.pdf/png`.

## 2026-09-28 — Effect-size versions of the two main timecourse panels

SK asked what the main panels look like as effect sizes with standard errors.
`*_loso_results.json` already stores `mean` and `sem` (the across-subject mean
and SE of the held-out per-subject readout), so nothing is recomputed: the
one-sample t plotted so far **is** `mean / sem`. `load_trace` now asserts that
identity rather than assuming it, so the two versions cannot silently diverge.

`plot_instr_dir_unord_timecourse.py` draws every panel twice — `<name>_3p5cm`
(group t, unchanged) and `<name>_effect_3p5cm` (mean +- SEM). The p_FWE rails
are identical in both: the sign-flip null is on the subject-level readout, not
on the summary statistic. Run: `instr_dir_unord_timecourse_figure_2026-09-28`.

| panel, k=100 | peak epoch | mean | SEM | t | p_FWE |
|---|---|---|---|---|---|
| mPFC plan (`ABCD_REW-..._unord_vs_exe_dir_across`) | `see-D-first` | .0432 | .0134 | 3.21 | .0097 |
| `MTL_left` memory (demeaned) | `see-C-first` | .0147 | .0058 | 2.55 | .035 |
| `MTL_right` memory (demeaned) | `see-B-first` | .0205 | .0078 | 2.62 | .034 |
| mPFC memory (raw, the null control) | `see-D-second` | -.0178 | .0086 | -1.78 | .16 |

Betas are ~1e-2, so y is scaled to 1e-3 units (`EFFECT_SCALE`); the scaling hits
mean and SEM alike, so the ratio still reads as the t. Ticks use `MaxNLocator`
in effect mode — `integer_yticks` steps by ceil(range/3) and on the wider scaled
range produced ticks like -17/-1/15. The SEM is drawn as a shaded band
(alpha .22) rather than as error bars: at 3.5 cm two sets of bars collide
wherever the L and R traces cross, and the band keeps the line the salient mark.

What the effect panels add over the t panels: the mPFC plan effect is ~3x the
size of the MTL memory effect in the same beta units (.043 vs .015), which the t
panels hide because the mPFC SEM is over twice as large (.013 vs .006). The
memory trace in mPFC sits just below zero across the whole period with SEMs that
cover zero for the first four epochs — visibly a null, not a weak negative.

Also in this run: `figure_stats.json` now records `mean`/`sem` per trace and a
`files` map per panel ({t, effect} -> pdf) in place of the single `file` key.


## 2026-09-27 — Every candidate comparison ROI for ripple-locked HFB (SUPERSEDED by 2026-09-28: candidates overlapped the existing regions)

New `scripts/swr_hfb_roi_inventory.py`. It does NOT re-extract anything: it
re-groups the existing `per_pair.csv` rows under a different ROI label and
applies the run's own `_effect` (imported, not reimplemented) to each group, so
candidates and the `MedialFrontal` reference are computed by the same function
on the same rows. Different-shaft pairs, native pad 0.25 s, subject-level.

Reference: MedialFrontal +0.0072 +- 0.0013, n = 35 subjects.

**Three tiers of "possible ROI":**

1. *Estimable now* — 22 Harvard-Oxford cortical labels and 35 Brainnetome
   labels (hemispheres collapsed) reach >= 3 sessions. Full tables in
   `derivatives/group/swr/hfb_roi_inventory_2026-09-27/`.
2. *Traces exist, no usable estimate* — 1 HO label (Intracalcarine Cortex),
   3 BN labels (cpSTS, A38m, A7r): too few clean peri-ripple epochs.
3. *Contacts exist, never extracted* — 26 HO labels with >= 3 sessions have
   macro contacts but no HFB derivation, because the bundle only built
   derivations for `contact_anatomy.CORTICAL_ROIS` (five regions). Biggest:
   Frontal Pole (27 sessions, 111 grey contacts), Middle Frontal Gyrus (24/56),
   Insular Cortex (19/64), Superior Frontal Gyrus (15/27), Parahippocampal
   Gyrus ant+post (8+6). These need ONE cluster re-run of the HFB stage
   (`--cortical_rois=[...]`), not new patients. ⚠ Those session counts are a
   LOWER BOUND: read from the local `macro_contacts_all.csv`, which covers 31
   of 61 sessions, so the true counts are roughly double.

**Nothing beats medial frontal.** Every candidate's paired contrast vs
MedialFrontal has q >= 0.22 (HO) / q >= 0.56 (BN). The only q < 0.05 contrast
anywhere is `family`: TemporalLateral -0.0058 below MedialFrontal, q = 0.018.
Positive effects at q < 0.05 in their own right: Paracingulate Gyrus +0.0073
(q = 0.023, but 100% overlap with MedialFrontal, so it IS the reference),
A11l +0.0113 (q = 0.001, 16% overlap) and A10m +0.0086 (q = 0.037, 100%
overlap). A11l is lateral OFC by Brainnetome and is the one genuinely new
region worth a pre-registered test.

Screen over ~25 regions per scheme; every p is reported with a BH q within its
scheme and the q is the column to read. Contrasts against the reference are
suppressed where the candidate overlaps it by more than 50% of its derivations
-- a paired test against a group that contains you is not a test.

## 2026-09-27 — Fig 1d showed a mask no reported statistic came from

SK asked why the MTL memory panel showed no significance marker when the text
reports p_FWE = .035. Cause: the panel and the sentence were different masks.

`plot_instr_dir_unord_timecourse.py` drew `memory_HC_LR_*` from
`hippocampus_left/right_bin50` (run `..._DEMEANED_2026-09-17`), while the
reported statistic is the **left Garvert MTL half** from the later run
`..._DEMEANED_UNILATERAL_2026-09-21`. The unilateral Garvert masks did not exist
when the figure folder was written on 09-20, so no panel had ever been drawn for
the test being reported. Both numbers were individually correct:

| k=100, `see-C-first` | t | p_FWE | passes? |
|---|---|---|---|
| `MTL_left` (quoted in the text) | 2.55 | **.035** | yes |
| `HC_left` (what the panel plotted) | 2.35 | .057 | no, t_crit = 2.41 |

The panel was honest about its own data — HC_left misses the epoch-corrected
threshold by 0.06 — it was simply not the analysis the caption claimed.

### Fix

`memory_HC_LR_*` panels replaced by `memory_MTL_LR_*` (`Garvert_MTL_2mm_L/R`,
the bilateral mask split at x = 0) in
`instr_dir_unord_timecourse_figure_2026-09-27`. The hippocampus-only panels were
dropped rather than kept alongside: they tested a tighter mask that nothing in
the manuscript reports, and the effect is better described as surviving in the
larger a-priori ROI. `hippocampus_*_bin50` now appears in no figure.

Both hemispheres carry a significance rail in the new panel, at different
epochs, which the old panel did not show:

| trace | peak epoch | t | p_FWE |
|---|---|---|---|
| `MTL_left`  | `see-C-first` | 2.55 | .035 |
| `MTL_right` | `see-B-first` | 2.62 | .034 |

Also fixed: `figure_stats.json` labelled the new unilateral panels `"betas":
"raw"`, because the demeaned check was `panel["svc"] is SVC_DEMEAN` and the
unilateral run is a second folder. Now tests both.

### Consequences for the text

- "peak effect in left hippocampus" must become left MTL — the *test* is the
  MTL half; the 73%-in-hippocampus figure is a post-hoc localisation of the
  selected voxels, not the mask the p-value was computed in.
- The right-hemisphere peak is now visible on the figure and needs stating. It
  is at `see-B-first`, one epoch earlier than the left, so the two halves are
  not one bilateral effect (consistent with the bilateral test, p = .077).
- "remained high until the last reward presentation" is not supported by either
  trace: `MTL_left` decays monotonically after C (1.84, 1.26, 0.94, 0.86, 1.00,
  0.79). It is elevated through `see-D-first` at most.
- k = 100 is the fixed readout for every LOSO panel in the figure, including
  mPFC, so it is not per-mask selection. But the mPFC plan effect holds at all
  three k (p = .028/.0097/.0016) whereas `MTL_left` holds only at k = 100
  (.141/.035/.068). That asymmetry should not be hidden.


## 2026-09-24 (d) — REPRODUCIBILITY BUG + the balance result, properly tested

**Bug: the permutation seed was not reproducible.** The per-cell seed was
`abs(hash((sub, neuron, tag))) & 0xFFFFFFFF`. Python 3 salts str hashing per
process unless PYTHONHASHSEED is set, so EVERY RUN DREW DIFFERENT
PERMUTATIONS. Measured: the same tuple gave 3855842339, 44291736 and
2509367741 in three consecutive interpreters. Material consequence — the
planned EC-vs-rest Fisher test read P = 0.054, then P = 0.023, on the
identical 924 cells. Every permutation p ever quoted from this script before
today carries that noise. Fixed: `cell_seed()` uses blake2b (verified
identical across processes). N_PERMUTATIONS 1,000 -> 10,000, because the
per-cell p is thresholded at .05 and then COUNTED across cells, so MC error
(~.007 at 1,000 draws) flips borderline cells. The RSA script was unaffected
(explicit seed=42).

**Canonical run: `2026-09-24_18-35-23`** (924 cells, 10,000 perms, blake2b
seed, current ROI labels, full recompute — not a relabelled reload).

**BALANCE = r_state - r_interaction per cell** (phase-invariant minus
phase-modulated held-out generalisation), new in the pipeline with its p-value
from the permutation distribution OF THE DIFFERENCE, built from the same
circular shifts already shared by both components.

| ROI | balance | q_FDR | FWE |
|---|---|---|---|
| EC | +0.0196 | 0.026 | 0.079 |
| mOFC | +0.0142 | **0.0098** | **0.0098** |
| mPFC | +0.0105 | 0.025 | **0.049** |
| HC_anterior | -0.0005 | 0.66 | 1.0 |
| HC_mid | -0.0068 | 0.81 | 1.0 |
| PCC | -0.0101 | 0.81 | 1.0 |

**Permutation omnibus** (shuffle ROI labels, recompute Kruskal-Wallis H,
10,000 draws): H = 15.29, **P = 0.0079** (null mean H = 4.99). Null-centred
variant H = 15.37, P = 0.0077 — identical conclusion.

**The block-size bias concern is dead.** Permutation-null means:
r_state -0.000013, r_interaction +0.000002, difference -0.000016. Both sit on
zero, and `mean_balance` matches `mean_balance_null_centred` to 4-5 decimals
in every ROI. The 4-column vs 8-column worry was unfounded.

**Sustained fractions (reproducible):** EC 12/35 = 34.3%, mPFC 38/158 = 24.1%,
mOFC 33/142 = 23.2%, HC_mid 42/233 = 18.0%, HC_anterior 51/295 = 17.3%,
PCC 8/61 = 13.1%. Planned Fisher EC-vs-rest **P = 0.031, OR = 2.17**;
six-region omnibus chi2(5) = 10.78, P = 0.056.

**Phase-modulated component:** EC q_FDR = 0.079 is the ONLY region failing
FDR; all others q <= 0.0052. Omnibus P = 0.021.

**New figure 13_balance**: the two components side by side with each one's
permutation-null mean drawn as a dotted reference (so the bias is visible, not
asserted), then the difference with a bootstrap CI and the permutation
omnibus.

⚠ STILL UNTESTED: reproducible is not the same as stable. A seed-sensitivity
check (re-run with a different global seed offset and confirm the Fisher p and
the cell counts hold) has NOT been done. Given the test previously wandered
0.054 <-> 0.023 at 1,000 draws, this matters before P = 0.031 is quoted.

**Convergence with the RSA is partial.** The sustained-state RSA
(`DSR_RSA_simple_ROI/2026-09-24_18-17-28`, state + state_phase, no phase
residualisation) leaves state significant in all five ROIs, largest in mOFC
(0.0977 -> 0.0785, -20%). mOFC agrees with the balance ranking; mPFC does not
(2nd on balance, last on surviving state beta, -37%). EC is absent from the
RSA entirely — too few cells for the pseudo-population — so the RSA can never
corroborate the EC half of the claim.

## 2026-09-24 (c) — NEGATIVE RESULT: state_phase control does not make the cell RSA state effect selective

**Hypothesis (mine), now falsified.** The fMRI `state` model is a one-hot over
A-D that is IDENTICAL for a state's path and reward bins, so the fMRI GLM can
only see the phase-invariant part of the position code. The cell RSA has 12
conditions (4 states x 3 equal-time thirds) and no such constraint, so I
predicted that phase-locked position selectivity leaks into `state`, and that
adding a `state_phase` RDM regressor would make the cell effect selective —
surviving in EC/mOFC/mPFC and collapsing in HC_mid/PCC.

**It does not.** Run `2026-09-24_17-09-09`, split_halves_z, state sub-model:

| ROI | fmri_matched_state | + ctrl state_phase | delta |
|---|---|---|---|
| mPFC | 0.0658 | 0.0554 | -0.010 |
| mOFC | 0.0997 | 0.0931 | -0.007 |
| HC_anterior | 0.0639 | 0.0745 | **+0.011** |
| HC_mid | 0.1070 | 0.1008 | -0.006 |
| PCC | 0.0515 | 0.0548 | **+0.003** |

All ten still significant (p_perm <= 0.003, q_FDR <= 0.003). No ROI loses its
state effect; two go UP. And `state_phase` itself is ~zero everywhere and
significant nowhere (beta -0.018 to +0.017; p_perm 0.16-0.83).

Likely why: `PHASE_RESIDUALISE='cosine'` already removes phase at the DATA
level before RDMs are built, so there is little state x phase variance left
for the regressor to absorb. Cosine residualisation removes the phase MAIN
effect and not state x phase, which is what motivated the test — but
empirically the residual state x phase structure the RDM can see is
negligible. Testing this properly would need a run with
PHASE_RESIDUALISE=None; not done.

**Also: the control stack barely matters at all.** `fmri_matched_state`
(state + location + l2_norm + dsr_fmri + reward_path, no buttons) reproduces
the previous `ctrl_fMRI-state_dsrFULL` (which had buttons, no reward_path) to
3-4 decimals: mPFC .0658/.0661, mOFC .0997/.1001, HC_anterior .0639/.0641,
HC_mid .1070/.1073, PCC .0515/.0515. The cell state effect is remarkably
robust to which controls are fitted, which is consistent with the user's
recollection that implementing the fMRI controls "didn't really improve
things".

**What this does NOT overturn:** the cell-level encoding result stands — the
per-cell balance r_state - r_interaction differs across ROIs (Kruskal-Wallis
H = 15.29, P = 0.0092; ANOVA F(5,918) = 4.17, P = 0.00095), EC/mOFC/mPFC
phase-invariant-dominant, HC/PCC phase-locked-dominant. That is a different
analysis on different quantities. The RSA-vs-fMRI discrepancy remains
unexplained by the model's control set; remaining candidates are the
condition definition (cells cut each state into 3 equal-TIME thirds; fMRI
splits path vs reward SEMANTICALLY from behaviour) and voxel-level spatial
clustering.

**Also found while running this.** The script had no `matplotlib.use('Agg')`
and the macOS default backend is interactive; the `plt.show()` after the
phase-mask diagnostic blocked a background run for 38 minutes at 0.0% CPU
(55 s CPU total), looking exactly like slow permutations. Added a guard that
falls back to Agg only when stdout is not a tty, so interactive runs are
unchanged.

## 2026-09-24 (b) — Main-result figure, caption-width bug, RSA pointer

New `fig12_main_result` (`12_main_result.pdf/.png`), built to carry the
rewritten headline at both levels of analysis:
- **a** per-ROI fraction of cells classified sustained (filled) vs phasic-only
  (hatched), with **Wilson 95% CIs** and the 5% chance line. The CIs are the
  point: EC's 31.4% is 11/35 cells, CI 19-48%, which is why the planned Fisher
  contrast sits at P = 0.054. The old fig 8b showed bare bars and invited
  over-reading.
- **b** population-shift **effect size** (matched-pairs rank-biserial of
  per-cell perm-p against 0.5, percentile bootstrap CI over cells, 2,000 draws,
  seed 42) for the sustained statistic and the phase-modulated component, same
  ROI order, connector lines per ROI. Replaces plotting p-values as data.

EC is the only region whose phase-driven marker is n.s. in both panels; PCC is
its mirror image. ROIs ordered by sustained fraction (stated in the caption).
era_brewer Showgirl2 ROI colours per project convention; Arial 11/10/9;
16.5 x 6.5 cm, PDF + 300 dpi PNG.

**Caption bug fixed (affected every figure in this script).** `_add_caption`
relied on matplotlib's `wrap=True`, which wraps at draw time while
`bbox_inches='tight'` re-measures the UNWRAPPED extent. Every captioned figure
was saved many times wider than its axes — fig 11 was 20,647 x 1,645 px for a
6 cm heatmap, fig 05 10,067 px. Captions are now hard-wrapped with `textwrap`
to the figure width. All figures now export at sane sizes (fig 11: 1,263 x
2,433). `_add_caption(..., draw=False)` records to captions.md without burning
the caption into the vector file; used for the publication figure.

**RSA companion pointer updated** to `DSR_RSA_simple_ROI/2026-08-31_17-57-30`
(postdates the ROI-label update) with combo `ctrl_fMRI-state_dsrFULL`. The old
`ctrl_dsrFULL` no longer exists; the new combo adds `bttn_next` and `l2_norm`,
which is what the chapter prose already describes. State betas: mPFC 0.066,
mOFC 0.100, HC_anterior 0.064, HC_mid 0.107, PCC 0.052; all p_perm <= 0.002, so
q_FDR = 0.0012 as before. No EC (insufficient coverage for the pseudo-
population), so that cell of the heatmap stays blank.

⚠ `_roi_palette()` still returns `STATE_COLORS` repeated, i.e. state colours
used where ROI is the categorical variable, against the project convention.
`fig12` uses `ROI_COLOURS`; the older figures that call `_roi_palette` do not.

## 2026-09-24 — Sustained position-in-sequence: re-run on the updated ROI labels

`neurons_with_ROI_labels.csv` was updated 2026-08-28 12:20; the canonical
sustained run `2026-08-17_08-41-10` predates it, so every cell-level number in
the chapter draft was computed on stale labels.

**The cell set is identical** (924 cells, 0 added, 0 dropped), so per-cell
statistics are unchanged and the reload+relabel path is exactly equivalent to a
full re-run. 30 of 924 cells changed region: mOFC→HC_anterior 16, mOFC→mPFC 6,
EC→HC_anterior 3, mPFC→mOFC 3, mOFC→HC_mid 2. Counts EC 38→35, mOFC 163→142,
HC_anterior 276→295, mPFC 155→158, HC_mid 231→233, PCC 61→61.

New run: `group/encoding_state_sustained_cv/
2026-08-17_08-41-10_relabelled_2026-09-24_15-22-10/`

**What changed in the inference.** The planned EC-vs-rest Fisher test on the
sustained fraction moved from p = 0.043 to **p = 0.054**. The effect size did
not move (31.6%→31.4% EC, 18.6%→18.7% rest, OR 2.02→2.00); losing 1 sustained
cell of 38 shifted a discrete small-sample test across 0.05. Do not report this
as significant.

Three results moved the other way and now support the "EC is purely sustained"
claim better than the fraction test did:
- EC's phase-modulation component no longer survives FDR (q 0.046 → **0.098**);
  it is the only region where it fails. Others q ≤ 0.0053 (FWE ≤ 0.026).
- EC's phasic-only fraction is no longer above the 5% chance rate
  (5/38, p = 0.040 → **4/35 = 11.4%, p = 0.096**).
- EC's sustained population shift still survives FDR (q 0.035 → 0.040,
  FWE 0.176 → 0.202).

Unchanged: sustained above chance in every region (all binomial p ≤ 0.011);
main position component significant everywhere except PCC (q = 0.094); no
regional difference in that component (χ²(5) = 2.50, p = 0.776); regions do
differ in phase modulation (χ²(5) = 13.20, p = 0.0216); sustained omnibus null
(χ²(5) = 8.89, p = 0.114).

**Script fixes.** (1) `config.json` was describing a model that was never
fitted — it listed a standalone phase main effect (there is no such column; it
lies in the span of the sum-coded interaction block) and described `r_state` as
a nested-model comparison against a reduced model, when it is a direct
correlation. Replaced with the actual design, input-trace provenance and CV
scheme. (2) `fig11` crashed on the undefined `RSA_STATE_RUN_DATE` (commented
out at L156) so Fig 5d was not being regenerated. (3) `roi_inference.json` now
persists the χ² statistic, dof, Fisher odds ratio and per-ROI k/n, which the
manuscript quotes but nothing stored.

⚠ **The RSA companion column of Fig 5d is still stale.** `RSA_STATE_RUN_DIR`
points at `DSR_RSA_simple_ROI/2026-07-30_15-58-51-fixed_cells-fixed_perms`
(pre-relabel). Newer runs exist; `2026-08-31_17-57-30` postdates the table
update. Its state betas barely move (mOFC 0.094→0.100, HC_mid 0.109→0.107,
mPFC 0.066→0.066, PCC 0.052→0.052, HC_anterior 0.066→0.064; all p_perm ≤ 0.002,
so q_FDR = 0.0012 is unchanged). But the combo was renamed and redefined:
`ctrl_dsrFULL` = [state, location, bttn_curr, dsr_fmri] no longer exists; the
new `ctrl_fMRI-state_dsrFULL` adds `bttn_next` and `l2_norm`. The chapter prose
("controlling for current and future location, as well as current and future
actions") describes the NEW control set, not the one the figure is drawing.
Neither RSA config records the ROI table path, so which table the Aug-31 run
used cannot be verified from the config alone — worth adding.


## 2026-09-23 (f) — Coverage figure: temporal control contacts added

`swr_hfb_coverage_figure.py` now draws the two temporal control regions beside
the medial frontal targets, in the project's SWR colours (`rfig.MONTAGE_C`):
lateral temporal `#8C8C8C`, A1/auditory `#4F4F4F`, i.e. the same greys the
locking-contrast panels use. `--rois` selects the sets; `--control_scale` /
`--control_alpha` size them.

**The temporal sets are NOT filtered to different-shaft pairs.** That filter is
right for the frontal sets and wrong here: lateral temporal cortex is what the
hippocampal electrode passes through, so sharing a shaft is its defining
property, and the filter drops 203 of 445 lateral-temporal derivations. The
hippocampal set stays on the different-shaft table.

Drawn (native pad 0.25 s, `medial_max_abs_x=20`): hippocampal 92 (L 45 / R 47),
mPFC 47, mOFC 54, lateral temporal 248, A1 50; 61 sessions, 42 subjects.
Output `derivatives/group/swr/ripple_locked_hfb_2026-09-18/
coverage_medial_temporal.{pdf,png}` + `_counts.csv` + `_settings.json`.

⚠ **The figure is drawing about half the contacts it should.** Anchor
coordinates come from `group/swr/macro_contacts_all.csv`, and the local copy
covers 32 of 61 sessions, so 51% of hippocampal, 53% of mPFC, 59% of mOFC, 56%
of lateral-temporal and 79% of auditory anchors resolve. This was already true
of the earlier `coverage_medial_4x2` figure. Fix by copying the cluster build of
that file; the counts above will then rise.

Legend in `contact_coverage_3d_figure` now wraps at 3 columns instead of forcing
one row, which at six sets was wider than the panel.

## 2026-09-23 (e) — One phase definition everywhere, HFB re-split, and a results summary

### The phase definition, now used in both branches

| phase | ripple-rate branch (`stage`) | HFB branch (`cond` suffix) |
|---|---|---|
| **explore** | `first uncovers` + `while learning` | `_explore` + `_plan` |
| **known** | `once known` | `_execute` |

The HFB branch labels its stages explore/plan/execute, which map one-to-one onto
first uncovers / while learning / once known. `plan` therefore belongs on the
EXPLORE side, exactly as `while learning` does — it was previously left with
`execute` by omission.

### HFB: re-split and redrawn (`swr_hfb_locking_contrasts.py`)

Conditions are pooled by ripple count, which equals the mean over their union
because each saved trace is already a per-ripple mean. Visual dropped (its whole
effect was same-shaft). Mean ripple duration 67.8 +- 0.4 ms (100737 ripples,
61 sessions), drawn on each panel as peak +- duration/2. Output in
`derivatives/group/swr/hfb_locking_contrasts_2026-09-23/`.

Explore phase (medial frontal), each side vs its own shifted null:
correct +0.0050 z, t(50)=+2.02, p=0.048, no cluster; incorrect +0.0054 z,
t(52)=+3.16, p=0.0026, no cluster; navigation +0.0072 z, t(51)=+2.41, p=0.020,
cluster +40..+270 ms p=0.011. Contrasts: all |d| <= 0.09, all p >= 0.52, no
clusters.

Known phase: correct +0.0066 z, t(52)=+4.59, p<0.0001, cluster -130..+110 ms
p=0.0010; navigation +0.0079 z, t(52)=+5.68, p<0.0001, cluster -260..+350 ms
p=0.0005; incorrect -0.0043 z, t(13)=-0.52, n.s. (only 14 sessions). Contrasts
all n.s.

**The condition claim is now dead in both phases.** With `plan` moved to the
explore side the correct-uncover effect there weakens (p=0.048, no cluster) and
navigation presses become the only explore condition with a cluster. Medial
frontal HFB is ripple-locked throughout; it does not care what the ripple
follows. The ANATOMICAL contrast is unaffected and remains the strong result:
MF - lateral temporal +0.0058 z, t(45)=+3.72, p=0.0005, Holm 0.0011, cluster
-130..+480 ms p=0.0005; MF - auditory +0.0065 z, t(23)=+2.65, p=0.0143,
Holm 0.0143.

### Figure conventions changed in `mc/analyse/ripples.py::plot_rows`

Affects `ripple_main_figure` too, deliberately:

1. `FS` 8 -> 9 (CLAUDE.md's floor), `LW_RATE` 1.2 -> 1.8, `LW` 2.0 -> 2.6.
2. The base-vs-window star now sits over the MIDDLE of the pair. It tests
   base against window, so putting it above the window point implied the
   window alone was significant.
3. Surviving clusters are now shaded on the peri-event trace as well as on the
   t curve. The docstring had always claimed this; the code never did it.

`swr_stage_interaction.py` figures: difference traces carry a 5-bin triangular
DISPLAY smooth (`SMOOTH_DISPLAY`) and thicker lines; every window and cluster
test still reads the unsmoothed per-session profiles.

### `scripts/swr_results_summary.py`

Collapses five result files into one flat list of 81 records with a common
shape (id, what, n, estimate, t, df, p, p_corrected, correction, cluster,
source, apa). `apa` is paste-ready; `source` names the file each number came
from, so no sentence can silently mix two estimators. Output in
`derivatives/group/swr/results_summary_2026-09-23/results_summary.json`.

## 2026-09-23 (d) — Two-phase versions of the main ripple figures

SK asked for the main-figure layout and the pairwise-difference layout redrawn
with two phases (explore = first uncovers + while learning, vs once known), so
the figure and the tested contrast describe the same cells.

`scripts/swr_stage_interaction.py` now writes four figures for the
`explore_known` scheme, all into
`derivatives/group/swr/ripple_stage_interaction_2026-09-23/`:

| file | what it shows |
|---|---|
| `two_phase_main.png` | `ripple_main_figure` layout via `rip.plot_rows`: peri-event rate, base-vs-window, sliding test, three rows |
| `phase_pairwise.png` | the two phases per valence plus a forest of the three tested differences |
| `stage_interaction.png` | the 2 x 2 and the valence difference per phase, sign-filled |
| `correct_phase.png` | correct uncovers alone |

Two things the two-phase main figure makes visible that the three-stage one did
not:

1. `error, exploring` now yields TWO sliding-window clusters (p=0.024 and
   p=0.016), one of them BEFORE the press. In the three-stage split the error
   cells had no clusters at all, so the pre-event drift was invisible. This is
   the same drift documented in 2026-09-23 (a); it is now on the figure rather
   than only in the JSON.
2. Row 3 changes meaning. The published figure's row 3 was `correct, first
   uncovers` against its matched control (+0.0344 Hz, p=0.032). Pooled to
   `correct, exploring` the window effect is no longer significant (+0.0128,
   p=0.243) although a cluster survives at +0.35..+0.55 s (p=0.031). Pooling
   buys the interaction and costs the stillness-matched confirmation, as noted
   in 2026-09-23 (b).

Figure-convention fixes applied to both pairwise figures: forest rows carry a
valence prefix (`corr:` / `err:`) because two rows otherwise read identically
and were separated only by colour, and the interaction is drawn in neutral grey
since it belongs to neither valence.

## 2026-09-23 (c) — Ripple-LOCKING contrasts: regions survive, conditions do not

SK asked precisely what the MedialFrontal-minus-control numbers are computed on,
and pointed out that the claim in panels i/j is about LOCKING (the cluster in
the time course), not about the peri-window magnitude. Both branches only ever
clustered each side against its own shifted null, so "cluster here, none there"
was never a test of a difference.

New `scripts/swr_hfb_locking_contrasts.py`. Reads the saved `timecourses.npz`
of both published runs — nothing re-extracted — rebuilds each per-session locked
curve exactly as the two figure functions do (real minus shifted,
flank-normalised to |t| in 250-750 ms, 100 ms Gaussian smooth), then takes the
PAIRED difference on sessions having both sides and runs the same
`swr_sakon.cluster_perm_time` (2000 sign-flips, seed 42). Holm within each
family. Output in `derivatives/group/swr/hfb_locking_contrasts_2026-09-23/`.

Validation: the per-side values reproduce the published ones exactly, including
reward_explore +0.00747, t(50)=+2.73, p=0.0087, cluster -130..+10 ms p=0.038.

### Regions (different-shaft, session unit)

| contrast | n | peri Δ | t | p | Holm | cluster of the difference |
|---|---|---|---|---|---|---|
| MedialFrontal − TemporalLateral | 46 | +0.00582 | t(45)=+3.72 | 0.0005 | **0.0016** | **−130..+480 ms, p=0.0005** |
| MedialFrontal − Auditory | 24 | +0.00647 | t(23)=+2.65 | 0.0143 | **0.029** | none |
| MedialFrontal − Visual | 24 | +0.00476 | t(23)=+1.44 | 0.163 | 0.163 | none |

### Conditions (medial frontal, explore phase, session unit)

| contrast | n | peri Δ | t | p | Holm | cluster |
|---|---|---|---|---|---|---|
| reward_explore − error_explore | 51 | +0.00180 | t(50)=+0.56 | 0.577 | 1.000 | none |
| reward_explore − move_explore | 49 | +0.00268 | t(48)=+0.59 | 0.556 | 1.000 | none |
| error_explore − move_explore | 51 | +0.00069 | t(50)=+0.19 | 0.852 | 1.000 | none |

### Reading

1. The ANATOMICAL claim survives as a direct contrast, and now with a locking
   cluster: medial frontal locks more than lateral temporal from -130 to +480 ms
   (p=0.0005), and more than auditory in the peri window (Holm 0.029, no
   cluster). Visual does not separate (p=0.163) — consistent with the earlier
   finding that the Visual effect was carried by same-shaft contacts.
2. The CONDITION claim does not survive in any form. All three differences are
   ~0 with d<=0.08 and no cluster. `error_explore` having no cluster of its own
   while `reward_explore` does is a power difference, not a difference between
   them: their locking curves differ by +0.0018 z, p=0.58.
3. So the sentence "after incorrect uncoverings ... was not locked to the
   ripple" cannot be supported. Medial frontal HFB is ripple-locked during
   exploration irrespective of what the ripple follows.

### Note on the earlier region numbers

The `_contrasts` values quoted from `ripple_locked_hfb_2026-09-18/result.json`
(MF−TempLat t(45)=3.30, p=0.0019 etc.) are a DIFFERENT estimator: per-derivation
peri-minus-non-peri, averaged per unit per ROI, real minus shifted, paired
t across units, and uncorrected across the eight contrasts that function
computes. The curve-based estimator above averages traces first and normalises
by the flank; it gives the same sign and a slightly larger t. Either is
defensible; they must not be mixed within one sentence.

## 2026-09-23 (b) — "Bigger here than there": every pairwise stage contrast, and the HFB ones

SK is removing interaction-fallacy claims from the main figure — wherever a
panel says "significant here, not there", the DIFFERENCE has to be tested. Two
families.

### Ripple rate: all six pairwise stage contrasts (new `--scheme=three_stage`)

`scripts/swr_stage_interaction.py` now carries two schemes; the three-stage one
tests every pairwise stage comparison inside each valence. Output in
`derivatives/group/swr/ripple_stage_interaction_three_stage_2026-09-23/`.
Holm over the six. NO cluster survives for any of the six, on either reading.

Own baseline (Sakon Eq. 2):

| contrast | n | Δ (Hz) | t | p_perm | Holm |
|---|---|---|---|---|---|
| correct: first − learning | 61 | +0.0189 | +1.06 | 0.295 | 1.000 |
| correct: first − known | 61 | +0.0308 | +2.18 | 0.029 | 0.172 |
| correct: learning − known | 61 | +0.0119 | +0.98 | 0.331 | 1.000 |
| error: first − learning | 43 | +0.0095 | +0.47 | 0.645 | 1.000 |
| error: first − known | 52 | −0.0418 | −1.77 | 0.085 | 0.424 |
| error: learning − known | 39 | −0.0233 | −0.85 | 0.402 | 1.000 |

Stillness-matched:

| contrast | n | Δ (Hz) | t | p_perm | Holm |
|---|---|---|---|---|---|
| correct: first − learning | 61 | +0.0221 | +1.29 | 0.204 | 0.614 |
| correct: first − known | 61 | +0.0336 | +1.82 | 0.075 | 0.450 |
| correct: learning − known | 61 | +0.0115 | +1.03 | 0.305 | 0.614 |
| error: first − learning | 41 | +0.0268 | +1.47 | 0.153 | 0.614 |
| error: first − known | 49 | −0.0085 | −0.41 | 0.692 | 0.692 |
| error: learning − known | 36 | −0.0378 | −1.63 | 0.115 | 0.574 |

Unsubtracted 0-0.5 s window (second family of six, own Holm):

| contrast | Δ (Hz) | t | p_perm | Holm |
|---|---|---|---|---|
| correct: first − known | **+0.0340** | t(60)=+3.12 | **0.0022** | **0.013** |
| correct: first − learning | +0.0246 | t(60)=+2.27 | 0.024 | 0.120 |
| error: first − learning | +0.0283 | t(42)=+1.92 | 0.065 | 0.258 |
| others | | | ≥0.11 | ≥0.33 |

### Reading

1. NOTHING survives Holm on the pre-specified baseline-subtracted test. Panels
   e and f cannot claim the response is bigger at first uncovers than later —
   the best is correct first − known, Holm 0.172.
2. The ONE contrast that does survive correction is correct first − known on
   the UNSUBTRACTED 0-0.5 s window (+0.0340 Hz, p=0.0022, Holm 0.013). It is
   defensible there because the correct cells' baselines are statistically
   identical (+0.0032, p=0.753), so subtraction only adds noise. But it was
   chosen after the subtracted version failed, which is a forking path and must
   be declared if used.
3. The error baseline problem recurs in the three-stage split: error first −
   known differs by +0.0276 Hz in the BASELINE window alone (p=0.059), the
   largest baseline difference of the six, while the correct baselines differ by
   <0.006 Hz (all p>0.65).

### HFB: the contrasts already existed

No new computation. `ripple_locked_hfb_2026-09-18/result.json`, different-shaft,
real-minus-shifted, MedialFrontal minus each control region:

| contrast | session | subject |
|---|---|---|
| − TemporalLateral | +0.0057, t(45)=3.30, p=0.0019 | +0.0055, t(28)=2.89, p=0.0074 |
| − Auditory | +0.0068, t(23)=2.39, p=0.026 | +0.0056, t(15)=1.94, p=0.072 |
| − Visual | +0.0044, t(23)=1.24, p=0.229 | +0.0068, t(14)=1.43, p=0.174 |

`ripple_hfb_conditions_2026-09-21/result.json`, MedialFrontal, count-balanced:

| condition | vs own shifted null | vs reward_explore |
|---|---|---|
| reward_explore | +0.0076, t(50)=2.73, p=0.0088 | — |
| error_explore | +0.0058, t(52)=2.85, p=0.0063 | +0.0059, t(50)=1.49, p=0.142 |
| move_explore | +0.0049, t(50)=1.34, p=0.186 | −0.0029, t(47)=−0.52, p=0.608 |

**Panel j's framing does not survive.** `error_explore` is itself significantly
ripple-locked (p=0.0063), and `move_explore` is numerically MORE locked than the
target (difference −0.0029). Neither difference approaches significance on the
count-balanced or unbalanced estimate. Medial frontal HFB is ripple-locked
during exploration regardless of what the ripple follows; it is not specific to
reward uncovers. Panel i survives against lateral temporal at both units and
against auditory at session level only (subject p=0.072).

### Not yet tested

Panel i's finer claim that lateral temporal locks "only around the ripple peak"
is a region x time-window claim, not a magnitude one, and has no test.

## 2026-09-23 (a) — The ripple valence x phase interaction is partly inherited from its own baseline

SK asked whether treating baseline-vs-window as a factor (a 2 x 2 with epoch)
would give "an interaction" distinct from the difference-of-differences already
reported. It would not: for a fully within-session 2 x 2 the epoch x phase
interaction applies weights (+1,-1,-1,+1) to the four cell means per session,
which is exactly `(window - baseline)_explore - (window - baseline)_known`.
F(1,60) = t^2; same p, minus the sign-flip null. Restating it as an ANOVA buys
nothing.

But the question exposed something untested: baseline subtraction only removes a
confound if the baselines are comparable, and nothing in the pipeline checks
that. Added `baseline_window_block` to `scripts/swr_stage_interaction.py` — the
same contrast family evaluated on the -1.6..-1.1 s window alone, nothing
subtracted. Output in `derivatives/group/swr/ripple_stage_interaction_2026-09-23/`.

### Baseline window alone (Hz, own rate, nothing subtracted)

| cell | n | baseline rate |
|---|---|---|
| correct, exploring | 61 | 0.1756 ± 0.0055 |
| correct, once known | 61 | 0.1780 ± 0.0059 |
| error, exploring | 61 | **0.1918 ± 0.0064** |
| error, once known | 52 | 0.1700 ± 0.0110 |

| contrast | baseline window | test window, unsubtracted |
|---|---|---|
| correct: explore − known | −0.0024, t(60)=−0.28, p=0.784 | +0.0251, t(60)=+2.51, p=0.011 |
| error: explore − known | +0.0243, t(51)=+1.74, p=0.089 | −0.0204, t(51)=−1.30, p=0.200 |
| **INTERACTION** | **−0.0308, t(51)=−2.05, p=0.047** | **+0.0440, t(51)=+2.28, p=0.027** |

### Reading

1. There is a SIGNIFICANT valence x phase interaction in the baseline window
   itself (−0.0308 Hz, p=0.047), of OPPOSITE sign to the post-event one. The
   arithmetic is exactly additive: 0.0440 − (−0.0308) = 0.0748, the
   baseline-subtracted interaction reported earlier. So roughly 40% of that
   0.0748 is removal of a pre-existing baseline difference, not response.
2. The interaction survives on the unsubtracted test window alone (+0.0440,
   p=0.027), so it is not purely a baseline artefact — but 0.0748 overstates it
   and should not be the quoted effect size.
3. The correct cells' baselines are indistinguishable (0.1756 vs 0.1780,
   p=0.784); the whole baseline interaction comes from the error cells, and
   specifically from `error, exploring` having the highest baseline of the four
   (0.1918). That is the same cell whose pre-event window is already depressed
   (−0.0212 Hz, p=0.007, cluster −0.35..−0.05 s): its baseline is measured at
   the top of a decline that continues through the event.
4. Practical consequence: the -1.6..-1.1 s window is not a neutral reference
   for `error, exploring`, which undercuts Sakon Eq. 2 for that cell. The
   stillness-matched reading, which uses a matched control press instead of a
   pre-event baseline, remains the cleaner design — and it gives p=0.152.

### Caveats

This baseline check is an additional test, outside the Holm family of three.
Its p=0.047 is uncorrected and would not survive correction; it is reported as
a diagnostic of the design, not as a result.

## 2026-09-21 (b) — The full phase x event-type grid: no selectivity anywhere

Incorrect uncoverings were also only extracted for the explore phase — the same
assignment bug as the movement presses. Errors in plan and execute were dropped
entirely, so a ripple following one was assigned to whatever preceded the error,
up to 2 s earlier. Now extracted in every phase.

MedialFrontal, session-level, real minus shifted null:

| | reward | incorrect | navigation |
|---|---|---|---|
| **explore** | +0.0076 (0.009) | +0.0058 (0.006) | +0.0049 (0.19) |
| **planning** | +0.0004 (0.95) | +0.0054 (0.39) | +0.0067 (0.16) |
| **once known** | +0.0068 (3.1e-05) | −0.0049 (0.56, n=14) | **+0.0080 (8.2e-07)** |

**No within-phase contrast is significant**, in either direction:

| contrast | n | difference | p |
|---|---|---|---|
| explore: reward − navigation | 49 | +0.0028 | 0.55 |
| explore: reward − incorrect | 51 | +0.0017 | 0.60 |
| once known: reward − navigation | 53 | −0.0012 | 0.55 |
| once known: reward − incorrect | 14 | +0.0133 | 0.13 |

Nor does either event type differ across phases (reward explore vs once-known
p = 0.73; navigation p = 0.24).

**The apparent selectivity in the per-condition column is a power difference,
not an effect difference.** Which conditions clear p < 0.05 against zero tracks
ripple count and session count: navigation in explore has 10,654 ripples and
misses, navigation once known has 52,228 and is the strongest cell in the grid.
Every direct comparison is null.

What survives: medial frontal HFB is locked to hippocampal ripples across the
whole task (+0.0073, t(52) = +6.14, p = 1.1e-07), and that locking is regionally
specific (auditory and visual null, and further-from-hippocampus regions show
*larger* effects). Nothing in this analysis supports selectivity for reward, for
errors, or for a task phase.

Figures: `phase_explore_2`, `phase_explore_3`, `phase_known_2`, `phase_known_3`
in `ripple_hfb_conditions_2026-09-21/`, each 2.5 x 2.5 cm.

## 2026-09-21 — Navigation presses are NOT a null control once sampled in every phase

**New:** movement-press events extracted in all three phases, plus two collapsed
conditions (`reward_all`, `move_all`) in `swr_ripple_hfb_conditions.py`.
**Results:** `.../group/swr/ripple_hfb_conditions_2026-09-21/`.

### What changed and why

Movement presses were only ever extracted for the **explore** phase. That had
two consequences, one cosmetic and one serious:

- a navigation control that could not be collapsed across phases, and
- **a broken event assignment**: with no movement events in plan or execute, a
  ripple in those phases was assigned to a rewarded uncovering up to 2 s earlier
  even when a navigation press had happened in between.

Extracting movement presses in every phase fixes both. It also **moves ripples**:
`reward_execute` falls from 79,084 to 39,514 ripples, because roughly 40,000 were
in fact closer to a navigation press than to the reward.

### The result this overturns

MedialFrontal, session-level, real minus shifted null:

| condition | ripples | effect | p |
|---|---|---|---|
| reward, explore | 8,041 | +0.0073 | 0.0096 |
| reward, execute | 39,514 | +0.0066 | 4.4 × 10⁻⁵ |
| error, explore | 12,814 | +0.0058 | 0.0063 |
| navigation, explore | 10,654 | +0.0049 | 0.19 |
| **navigation, plan** | 10,376 | **+0.0076** | 0.065 |
| **navigation, execute** | 53,258 | **+0.0085** | **3.5 × 10⁻⁷** |
| **reward, all phases** | 50,542 | **+0.0055** | 1.0 × 10⁻⁴ |
| **navigation, all phases** | 74,582 | **+0.0077** | 6.1 × 10⁻⁶ |

**Navigation presses collapsed across phases give a LARGER ripple-locked medial
frontal response than rewarded uncoverings** (+0.0077 vs +0.0055), and
navigation during execution is the largest single condition in the whole set.
Cluster-corrected time courses agree: reward −120 to +80 ms (p = 0.0005),
navigation −240 to +340 ms (p = 0.0005) — navigation's is the longer one.

The previous "absent after navigation presses" (+0.0049, p = 0.19) was true
**only of explore-phase navigation** — the phase with the fewest navigation
ripples and the weakest effect. It was an artefact of where the control was
sampled, not a property of navigation.

### What the data actually support

Within a phase, event type does not separate: explore gives reward +0.0073,
error +0.0058, navigation +0.0049; execution gives reward +0.0066, navigation
+0.0085. The variation is **across phases**, not across event types. Medial
frontal HFB is locked to hippocampal ripples throughout the task, and this
analysis provides no evidence that the locking is selective for reward.

⚠ **The claim "selectively for reward uncovers" cannot be carried forward**, and
the results paragraph and `ripple_locked_HFB.md` both need revising. The overall
ripple-locking result (+0.0073, t(52) = +6.14, p = 1.1 × 10⁻⁷) and the regional
dissociation against auditory and visual cortex are untouched.


## 2026-09-21 (d) — Ripple rate: explore vs execute, and the crossed error cells

SK looked at row 2 of `ripple_main_figure.png` (pad 0.25) and asked whether the
apparent crossing of the two error cells — rate falls from base to test after an
error while exploring, rises after an error once the grid is known — is really
untested.

It was untested. `ripple_statistics.json` holds `between: stage | error
(first - later)` (t(56)=0.51, p_perm=0.61), but that is a different quantity:
each side is the stillness-matched target-minus-control difference, not the
base→test change, and `later` pools `while learning` with `once known`. SK
pointed out that this pooling is the wrong way round — `while learning` is still
exploration, so putting it on the execute side dilutes exactly the contrast of
interest. Correct.

New script `scripts/swr_stage_interaction.py`. Pooling:

    exploring  = first uncovers + while learning
    once known = the grid is known and being executed

Two readings: the stillness-matched one (uncovering minus its own matched arrow
press, as everywhere else in the pipeline) and the own-baseline one (Sakon
Eq. 2, the quantity the figure draws). Three contrasts per reading, Holm over
that family; same `window_stats`, same sign-flip null, same sliding-window
cluster test as any ordinary condition. Results in
`derivatives/group/swr/ripple_stage_interaction_2026-09-21/`.

### Stillness-matched (uncover − matched arrow press, 0–0.5 s)

| cell | n | Δ (Hz) | t | p_perm | stillness tgt/ctl | events |
|---|---|---|---|---|---|---|
| correct, exploring | 61 | +0.0128 | +1.19 | 0.243 | 0.72 / 0.72 s | 6073 |
| correct, once known | 61 | +0.0008 | +0.08 | 0.934 | 0.50 / 0.50 s | 10412 |
| error, exploring | 61 | −0.0101 | −1.31 | 0.195 | 0.53 / 0.55 s | 7492 |
| error, once known | 49 | +0.0158 | +0.85 | 0.402 | 0.75 / 0.76 s | 1744 |

| contrast | n | Δ (Hz) | t | p_perm | Holm | cluster |
|---|---|---|---|---|---|---|
| error: exploring − once known | 49 | −0.0233 | −1.18 | 0.247 | 0.493 | none |
| correct: exploring − once known | 61 | +0.0120 | +1.00 | 0.323 | 0.493 | none |
| INTERACTION (correct−error) × (exploring − once known) | 49 | +0.0345 | +1.46 | 0.152 | 0.455 | none |

### Own baseline (−1.6..−1.1 s → 0..0.5 s)

| cell | n | base | test | change | p_perm |
|---|---|---|---|---|---|
| correct, exploring | 61 | 0.1756 | 0.2062 | +0.0306 | 0.0009 |
| correct, once known | 61 | 0.1780 | 0.1811 | +0.0031 | 0.626 |
| error, exploring | 61 | 0.1918 | 0.1763 | −0.0155 | 0.033 |
| error, once known | 52 | 0.1700 | 0.2002 | +0.0303 | 0.138 |

| contrast | n | Δ (Hz) | t | p_perm | Holm | cluster |
|---|---|---|---|---|---|---|
| error: exploring − once known | 52 | −0.0448 | −1.97 | 0.055 | 0.055 | none |
| correct: exploring − once known | 61 | +0.0275 | +2.37 | 0.019 | 0.038 | none |
| INTERACTION (correct−error) × (exploring − once known) | 52 | +0.0748 | +2.77 | 0.008 | 0.024 | increase 0.25..0.75 s, p=0.010 |

### The 2 x 2 as tested (sessions every cell has) — `stage_interaction.png`

Figure redesigned on SK's read: 3.2 x 3.2 cm panels, Arial 8-9 pt, lw 2.2.
Colours now come from `rip.condition_colour` itself — valence sets the hue,
stage sets the lightness — so a dark marker is an exploring cell here and in
`ripple_main_figure.png` alike. The interaction time course was replaced: SK
could not read a contrast plotted in Hz, so the two columns now show
`correct − error` at each stage, filled teal where the correct uncover carries
more ripples and pink where the error one does. The interaction is then simply
"the teal panel vs the pink panel", and the cluster is shaded on both. The two
per-stage p values are an uncorrected decomposition; the Holm adjustment sits
on the interaction.

The interaction is paired across all four cells, so it is tested only on their
intersection; the figure now plots that, not each cell on its own session set.

Own baseline, n=52 sessions:

| | exploring | once known |
|---|---|---|
| correct uncover | **+0.0315** (p=0.002) | +0.0015 (p=0.818) |
| error uncover | −0.0145 (p=0.068) | +0.0303 (p=0.138) |
| correct − error | **+0.0460**, t(51)=+3.54, p=0.0010 | −0.0288, t(51)=−1.26, p=0.218 |

Stillness-matched, n=49 sessions:

| | exploring | once known |
|---|---|---|
| correct uncover | +0.0137 (p=0.259) | +0.0024 (p=0.844) |
| error uncover | −0.0075 (p=0.384) | +0.0158 (p=0.402) |
| correct − error | +0.0211, t(48)=+1.68, p=0.102 | −0.0134, t(48)=−0.67, p=0.501 |

Note the asymmetry: on the own-baseline reading the valence effect is carried
entirely by the exploring column (p=0.0010); the once-known column is a
non-significant reversal (p=0.218). So "less for late positive, more for late
negative" overstates it — late positive is flat (+0.0015), not suppressed.

### Correct uncovers alone — `correct_phase.png`

The simple effect of phase inside one valence (not an interaction: with two
phases it is a difference). Explore +0.0306 Hz, t(60)=+3.33, p=0.0009, cluster
+0.25..+0.65 s p=0.006; once known +0.0031 Hz, t(60)=+0.49, p=0.626, no cluster.
Difference +0.0275 Hz [+0.0043,+0.0507], t(60)=+2.37, p_perm=0.019, Holm 0.038,
d=0.30 — but NO cluster survives the window-free test, and stillness-matched it
is +0.0120 Hz, t(60)=+1.00, p=0.323.

Note the asymmetry between the two contrasts: the valence x phase interaction
survives both the fixed 0-0.5 s window (p=0.008) and the window-free cluster
test (+0.25..+0.75 s, p=0.010); the correct-only phase difference survives only
the fixed window. Both tests share the same -1.6..-1.1 s baseline subtraction —
they differ only in whether the post-event window is fixed or scanned.

### Reading

1. The 2×2 valence × stage interaction is real on the own-baseline reading:
   +0.075 Hz, t(51)=2.77, p_perm=0.008, Holm 0.024, and it survives its own
   cluster test at 0.25–0.75 s (p=0.010) — the one place in this family where
   window choice is not doing any work.
2. The within-error stage effect alone is only marginal even with the right
   pooling: −0.045 Hz, t(51)=−1.97, p_perm=0.055, no cluster. So the crossing
   SK saw is carried by both arms together, not by the error arm on its own.
   The correct arm is the cleaner of the two (+0.028, Holm 0.038).
3. The correct pooling does move things in the expected direction. Against the
   wrong pooling (`first uncovers` vs `first + while learning` pooled as
   `later`) the same contrasts gave error −0.028 (p=0.105) and the interaction
   +0.075 (p=0.012, Holm 0.074); pooling `while learning` with exploration
   sharpens the error arm (−0.045) and drops the interaction's Holm p from
   0.074 to 0.024.
4. Nothing survives on the stillness-matched reading: the interaction is
   +0.035 Hz, p_perm=0.152, no cluster — same sign, half the size, and `error,
   once known` keeps only 49 sessions and 1744 matched events, the thinnest
   cell in the design. So the effect is NOT established once each side is
   stillness-adjusted, and the own-baseline result cannot be quoted as though
   it were.

### Caveats

Post-hoc: the contrasts were chosen after looking at the figure, and two
poolings were tried before this one, so the effective family is larger than the
three the Holm adjustment covers. `error, once known` is the limiting cell
(52 sessions own-baseline, 49 matched). The discrepancy between the two readings
is the thing to resolve before this goes in the manuscript — stillness is not
balanced across these cells (0.53 s exploring vs 0.75 s once known for errors),
which is precisely the confound the matched reading exists to remove.

## 2026-09-21 (c) — SK's same-task regressor: tested on sub-02, and a retraction

SK objected: if the cue confound were the whole story, adding `direction` should
fix it completely, so something else must drive the residual. Correct. SK then
proposed a same-task RDM (0 on same-layout pairs, 1 elsewhere) as the control.
Tested on sub-02, cached data RDM `glmbase_01-TR4_grey_matter`, 90 within-half
cells, using the pipeline's own RDM builders and `evaluate_model_vec`
(intercept + z-scored regressors). 126111 of 126404 voxels NaN-free.

### Structure (exact, no estimation)

Within-half cells: 90 (2 halves x 45). The instruction model is 0 on 10 of them,
and **all 10 are exactly the same-layout pairs** -- SK's proposed regressor
targets precisely the right cells.

| correlation over the 90 within-half cells | r |
|---|---|
| instr vs direction | **-0.245** |
| instr vs same-task | **+0.774** |
| direction vs same-task | -0.316 |

This is the quantitative answer to the objection: `direction` shares only ~6% of
its variance with the instruction model, so it could never have removed the
confound. The same-task indicator shares ~60%.

### Result: median single-subject t of the instruction beta across voxels

| design | ordered (ABCD_rew_instr) | unordered (set) |
|---|---|---|
| model alone | -0.508 | -0.606 |
| + direction (**the combo that was run**) | **-0.567** (worse) | **-0.665** (worse) |
| + same-task | **-0.251** | **-0.390** |
| + direction + same-task | -0.250 | -- |

SK's regressor roughly halves the shift; `direction` does not help at all and
slightly worsens it.

### RETRACTION

My earlier claim that "`direction` removes the uniform crossing part
(-0.630 -> -0.289)" is withdrawn. Those two numbers were the UNORDERED model
from the epoch GLMs as a GROUP t over 33 subjects; the -0.630 was the
single-model fit and -0.289 the combo, i.e. different data and a different
statistic from this within-subject test. Here, on matched data, direction makes
the shift worse in both models. So the partial improvement I attributed to
direction absorbing the confound is not supported, and the reasoning built on it
was wrong.

### What is still unexplained

A residual negative shift survives the same-task regressor (-0.25 ordered,
-0.39 unordered). Neither the cue confound nor the same-task contrast accounts
for all of it. Three explanations have now failed (temporal proximity; uniform
within-half similarity absorbed by the intercept; direction partialling), so the
residual is recorded as OPEN rather than explained.

### Cost of the fix

corr(instr, same-task) = +0.77, so adding it removes ~60% of the instruction
model's variance. What identifies the model afterwards is only its graded
structure among DIFFERENT-task pairs -- a weaker and less specific claim than
"the two presentations of one layout look alike", and that graded structure is
shared with the set and execution models.

Caveats: one subject; the TR4 per-TR GLM, not the epoch GLMs the reported result
uses; single-subject regression t, not a group statistic. Script:
scratchpad/sametask_test.py.


## 2026-09-21 (b) — Why the within-half memory effect is negatively shifted and hemisphere-split

Follow-up to the unilateral test above. SK asked how to discuss three oddities of
the `*_REW_INSTR-*_instr_vs_dir_within` (memory) result: it is tested within
task-half only, its t-distribution is strongly negatively shifted despite
controlling for forward/backward demand, and the two hippocampi carry opposite
signs. Four checks, all on the existing beta maps (n=33, 9 instruction epochs).

### 1. The negative shift is a property of the WITHIN-HALF scope, not of the model

Voxelwise group t over the whole brain, RAW (before demeaning), median / % t<0 /
% t<-1.96 / % t>+1.96:

| model | median t | % t<0 | t<-1.96 | t>+1.96 |
|---|---|---|---|---|
| memory ordered, within-half           | -0.455 | 69.7 | 5.8 | 0.4 |
| memory unordered, within-half         | -0.289 | 63.0 | 4.2 | 0.5 |
| memory unordered, **alone**, within   | -0.630 | 76.3 | 8.6 | 0.2 |
| memory unordered, **alone**, ACROSS   | **+0.005** | 49.8 | 2.8 | 2.6 |
| direction alone, within-half          | +1.478 |  6.1 | 0.1 | 29.7 |
| direction alone, across-half          | +0.491 | 30.2 | 0.6 | 6.7 |

The same model fitted across halves is centred on zero to three decimals and has
symmetric tails (2.8% vs 2.6%). Fitted within halves it sits at -0.63. So the
shift is NOT cognitive demand, NOT the model, and NOT the direction competitor:
**dropping the direction regressor makes the shift worse** (-0.289 -> -0.630),
which rules out OLS partialling as the cause.

CORRECTION (same day): my first explanation for this -- temporal proximity of
within-half trials -- is WRONG and is retracted. The instruction onsets show
forward and backward trials are interleaved, not blocked (sub-01 pt01 order:
forw, backw, forw, backw, forw, backw, forw, forw, backw, backw), so there is no
temporal clustering to drive it. It is also ruled out on principle: the RSA OLS
(`my_RSA`, l.1316-1404) appends an intercept and z-scores every regressor, so a
UNIFORM shift of the within-half data RDM is absorbed by the intercept and
biases no model at all.

The real mechanism is documented in `same_direction_mask_2d`
(fMRI_run_RSA_instruction.py l.363-384), written for exactly this problem:

  "The instruction screen is NOT identical for the two directions of a task: it
   shows the same four reward locations plus a 'please backwards' text, or
   nothing, implying forwards. Every `_instr` model ignores that cue and so
   forces dissimilarity 0 on exactly the same-task forward/backward pairs --
   which are precisely the cells where the cue differs. Those cells carry
   enormous leverage (they are the only 0 level in a model whose other levels
   are >= 0.25), and no nuisance regressor can absorb them: anything correlated
   enough with the instruction model to move its coefficient IS the instruction
   model's own main contrast."

So one stimulus fact drives both signs. The instruction model predicts MAXIMUM
similarity (its only d=0 cells, 5 of 45 pairs per half) exactly where the screen
physically differed; the data there are dissimilar, so its beta goes negative.
The direction regressor is 1 on exactly those same forward/backward crossings,
i.e. it IS that cue structure, so its beta goes positive. The two models are
anti-correlated by design, and the anti-correlation is concentrated on the
instruction model's highest-leverage cells. This also explains why adding
`direction` only partially helps (-0.630 -> -0.289 median t) and why controlling
for forward/backward cognitive demand did not help: per the docstring, no
nuisance regressor can absorb it.

### WHY `direction` ONLY HALVES THE SHIFT (SK's objection, and it is correct)

If the cue effect were uniform over crossings, the direction regressor would
absorb it completely and the residual -0.289 would not exist. The geometry says
why it does not. Within a half (10 conditions = 5 layouts x 2 directions,
45 pairs):

| cell type | n | instruction model | direction |
|---|---|---|---|
| same-task crossings  | 5  | **d = 0** (its only zero level) | 1 |
| diff-task crossings  | 20 | >= 0.25 | 1 |
| same-direction pairs | 20 | >= 0.25 | 0 |

`direction` is 1 on 25 cells and treats all 25 alike, so partialling it removes
only the MEAN crossing-vs-non-crossing difference. Any part of the crossing
effect specific to the 5 same-task cells survives -- and those 5 are exactly
where the instruction model places its only d=0 prediction.

There is a concrete reason the 5 differ from the other 20: for one layout, the
forward and backward trials require OPPOSITE executed trajectories, so the
planned action differs maximally exactly there; for different-task crossings the
plans differ anyway. A regressor marking "same task AND crossing" would absorb
it, but that regressor IS the instruction model's own main contrast -- which is
what the `same_direction_mask_2d` docstring means. Hence: direction removes the
uniform part (-0.630 -> -0.289), the same-task-specific part is unabsorbable by
construction, and only MASKING (within_same_direction) can remove it.

This is a hypothesis for the residual, not a demonstration. The samedir run
tests it: if the shift persists on same-direction cells only, the cause is
something else about within-half data RDMs and this account is wrong too.

### THE REPORTED RESULT USES THE CONFOUNDED SCOPE

`condition_files/rsa_instruction_direction_and_unordered.json` sets
`data_rdm_scope: "within_only"` and every instruction combo carries
`"scope": "within_only"`. The fix the code already implements --
`within_same_direction`, which drops every forward/backward crossing -- was NOT
run: no `*-within-samedir` map exists in the per_TR folders. NOTE: the
instruction+direction COMBO is a control, but it is the nuisance-regressor
control, which is the one the docstring says cannot work here; masking and
partialling are different operations. The reported
`ABCD_REW_INSTR-ABCD_instr_vs_dir_within` therefore includes the confounded
cells. Re-running the instruction combos with `"scope": "within_same_direction"`
is the decisive test, and per the config comment all scopes are subsets of the
same cached data RDM, so it costs no new searchlight work. Cost: it keeps 20 of
45 cells per half, and each task pair appears twice with an identical model
value, so only 10 independent predictions per half remain. It may simply be
underpowered -- which is a likely reason it was not the default.

### 2. Demeaning under-corrects in MTL, so the left-HC effect is conservative

Median raw t relative to the whole-brain median: MTL_left +0.012, MTL_right
-0.171, HC_left +0.108, HC_right -0.055. MTL sits at or below the brain-wide
level, so subtracting the brain-wide mean leaves MTL slightly under-corrected.
The surviving positive left-HC effect is therefore not an artefact of
over-correction.

### 3. The L/R asymmetry is NOT created by the within-half bias

The same left>right MTL gradient appears in the across-half PLAN model, which
carries no within-half bias at all, and is LARGER there:

| model | MTL_left | MTL_right | L-R |
|---|---|---|---|
| memory (within-half) | -0.443 | -0.626 | +0.18 |
| plan (across-half)   | +0.629 | +0.226 | **+0.40** |

So the hemispheric difference is a general property of MTL RSA fits in this
dataset, not something the within-half scope manufactured. It does not by itself
establish a memory-specific lateralisation.

### 4. The direction regressor does not explain the right-hemisphere dip

LOSO k=100 on `DIRECTION-ABCD_instr_vs_dir_within` is flat in all four masks
(peak t=0.61, p_FWE=0.70 in MTL_right; peak t=0.22, p=0.81 in MTL_left). The
negative excursion in right HC at `see-D-first` is therefore NOT execution-
direction coding. This refutes the tempting "right HC codes the executed
trajectory" story -- it was tested and is not there.

Run: `instr_dir_unord_svc_loso_DEMEANED_DIRECTION_2026-09-21`.

### 5. Where the selected voxels actually are

Across the 33 LOSO folds in the MTL_left mask, 205 distinct voxels ever enter
the top-100 and 56 are picked in every fold. Of those 56: 73% Left Hippocampus,
11% posterior parahippocampal, 5% anterior parahippocampal; centroid MNI
(-31, -20, -18). In the HC_left mask the 57 always-picked voxels are 100%
hippocampal, centroid (-30, -17, -18). The "left hippocampus" description of the
MTL_left readout is therefore justified by the selection, not only by the peak.

### 6. Incidental: the UNORDERED memory model is stronger, and right-lateralised

`ABCD_REW_INSTR_UNORDERED-ABCD_unord_vs_dir_within`, LOSO k=100, MTL_right:
t = 3.24, p_FWE = 0.0087 at `see-A-first` -- stronger than the left-lateralised
ordered effect (t=2.55, p=.035 at `see-C-first`). Ordered and unordered traces
are near-identical in right HC, so the two models are highly collinear there and
this is not a clean order/set dissociation. Flagged, not claimed; it is one more
test in an already uncorrected family.


## 2026-09-21 — Unilateral MTL/HC test of `*_REW_INSTR-*_instr_vs_dir_within` (instruction phase)

SK asked (a) where and in which condition the peak voxel of
`ABCD_REW_INSTR-ABCD_instr_vs_dir_within` (demeaned, resolved) sits, and
(b) whether the hippocampal effect that vanishes bilaterally survives when each
hemisphere is tested on its own — the bilateral null being suspect because left
and right HC carry opposite signs.

Run: `instr_dir_unord_svc_loso_DEMEANED_UNILATERAL_2026-09-21`
(`per_TR_loso.py --mode run --axis condition --demean`, 9 instruction conditions,
n=33, 10000 sign-flip permutations, seed 0, k=50/100/200, max-t corrected over
voxels x conditions within each mask). Masks: `Garvert_MTL_2mm_L/R.nii.gz`
(Mona's MTL mask split at x=0) and `hippocampus_left/right_bin50.nii.gz`.
HC_left/HC_right were re-run as a reproduction check and match
`instr_dir_unord_svc_loso_DEMEANED_2026-09-17` to 3 decimals.

### (a) Peak voxel

The whole-brain positive peak of the demeaned ABCD map is **t = +4.64 at
MNI (-34, -22, -22), condition `see-C-first`** — left hippocampal/collateral
border (HO sub-prob: 56% cortex, 32% WM, 12% L hippocampus). Inside the HO
hippocampus mask the peak is t = +3.63 at (-34, -18, -18), same condition,
62% L hippocampus.

The bilateral null is indeed a cancellation, not an absence: at `see-D-first`
right HC carries t = -3.77 at (28, -20, -14) (97% R hippocampus) while left HC
is positive at the same condition. Left peaks early (`see-C-first`), right peaks
late (`empty-screen`, t = +3.50).

### (b) Unilateral tests — LOSO readout, peak over the 9 conditions

| mask | k=50 | k=100 | k=200 | peak condition |
|---|---|---|---|---|
| MTL_left (Garvert L)  | t=1.92 p=.141 | **t=2.55 p=.035** | t=2.30 p=.068 | `see-C-first` |
| MTL_right (Garvert R) | **t=2.91 p=.018** | **t=2.62 p=.034** | t=2.11 p=.094 | `see-B-first` |
| HC_left               | t=1.80 p=.173 | t=2.35 p=.057 | **t=2.73 p=.029** | `see-C-first` (k200: `see-D-first`) |
| HC_right              | **t=2.54 p=.042** | t=2.12 p=.094 | t=1.85 p=.151 | `empty-screen` |

At the k=100 readout used in the figures: Garvert L p_FWE = .035, Garvert R
p_FWE = .034, HC_left p_FWE = .057, HC_right p_FWE = .094. Bilateral Garvert MTL
at k=100 was p = .077, so splitting by hemisphere does move both halves below
.05 — but see the caveats.

### Caveats, so this is not over-read

1. **Post-hoc split.** The hemispheres were tested separately only after the
   bilateral test came out null. Bonferroni over the two hemispheres alone turns
   .035/.034 into .07/.068.
2. **k is not fixed a priori.** No mask is significant at all three k. MTL_left
   only at k=100; MTL_right at k=50 and k=100; HC_left only at k=200; HC_right
   only at k=50. Reporting the best k per mask would be selection on the result.
3. **Not the same effect on both sides.** Left peaks at `see-C-first`, right at
   `see-B-first`/`empty-screen`. This is not one bilateral effect masked by
   averaging; the hemispheres peak at different points of the instruction period.
4. **The voxel-wise max-t SVC remains null everywhere.** Across all four
   unilateral masks the voxel-wise peak p_FWE is .21-.41 and no voxel survives
   FWE .05. The effect exists only in the LOSO multivariate readout.
5. 4 models x 4 masks were tested; peak_p_FWE corrects over voxels x conditions
   within each mask only, not across that family.


## 2026-09-19 — Manuscript numbers refreshed to bundle_v2 @ 0.25 s; two reporting bugs fixed

SK spotted that the main figure printed `p = 0.018` for `correct, first
uncovers` while the results draft reported different values. Both were correct;
they were different analyses, and the draft was quoting one that appears on no
figure.

### Bug 1 — the figure's statistic was not in the statistics file

`feedback_stage_rows()` produces the figure's rows 1–2 (each cell against its
OWN baseline, all events — Sakon Eq. 2) but its results were never written to
`ripple_statistics.csv/json`. The draft instead quoted
`cells[...]['vs own baseline']`, which is
`(target − baseline) − (matched control − baseline)` on the matched subset only,
and mislabelled it "Sakon & Kahana's Eq. 2". Three distinct numbers existed for
one claim:

| analysis | events | window 0–0.5 s | cluster |
|---|---|---|---|
| own baseline, all events (Eq. 2) | 5,349 | +0.0340 Hz, t(60) = +2.70, p = 0.0068 | +0.35…+0.65 s, **p = 0.018** |
| baselined matched contrast | 3,074 | +0.0482 Hz, t(60) = +2.31, p = 0.025 | +0.25…+0.75 s, p = 0.007 |
| matched press, no baseline | 3,074 | +0.0344 Hz, t(60) = +2.15, p = 0.032 | +0.35…+0.75 s, p = 0.003 |

`feedback_stage` stats now go into both tables. The drafts were restructured on
**own-baseline as headline, matched press as control**, and the hybrid dropped.

### Bug 2 — padded numerator over unpadded denominator

`repad_bundle` rebuilds `intervals` (which `rip.derivations` uses, so every rate
and test was correct) but deliberately leaves `channel_qc.clean_s` at the
bundle's native pad. `swr_final_ripple_analysis.py` read `clean_hours` from
`channel_qc`, so the reported exposure did not match the reported ripple count.

- reported: 94,654 ripples / **142.4 h** ← unpadded denominator
- correct: 94,654 ripples / **131.9 h** → 0.1994 Hz

`clean_hours` now comes from `intervals`; `clean_hours_channel_qc` is kept
alongside for comparison. This also reconciles with `repad_bundle`'s own
docstring, which cites 91.4 h at the 1.0 s pad against `channel_qc`'s 91.2 h.

### `--pad_s` added to the three stillness scripts, and all reruns done

`swr_stillness_control.py`, `swr_feedback_vs_matched_stillness.py` and
`swr_stillness_anatomy.py` now accept `--pad_s` and record it. Rerun on
`bundle_v2` at 0.25 s; every number in that section of the results draft moved:

| quantity | pad 1.0 | **pad 0.25** |
|---|---|---|
| stage effect, unstratified | +0.0360, p = 0.044 | +0.0256, p = 0.072 |
| stillness-standardised | +0.0169, p = 0.488 | +0.0251, p = 0.176 |
| **attributable to stillness** | **62%** | **51%** |
| task-free rest vs in-task stillness | −0.0234, p = 0.046 | **−0.0332, t(60) = −3.10, p = 0.0037** |
| feedback − matched pause (correct, first) | +0.0105, p = 0.56 | +0.0161, p = 0.282 |

`--rebuild=False` added to the final script so a bundle rebuilt for a pad change
does not trigger a needless 15-minute re-read of the button series.

### A result that did not reproduce

At the 1.0 s pad the matched contrast was already positive before the press
(+0.039 Hz, p = 0.074), and the draft read that as resembling Sakon's PRE effect.
At 0.25 s it is **−0.0002 Hz (p = 0.98)**; own-baseline **+0.014 (p = 0.46)**.
The claim is removed. The effect is now cleanly post-event: the matched contrast
runs −0.014 Hz at the baseline window and +0.034 Hz at the test window.

### Documents updated

- `statistics_results_DRAFT.md` — rewritten end to end on bundle_v2 @ 0.25 s.
- `statistics_methods_DRAFT.md` — sample, control spread (0.019 → 0.028 Hz),
  matching-balance claim (exact in 6 of 7, 1.050 vs 1.037 s in the seventh).
- `methods_03` — ±1 s → ±0.25 s throughout; contamination **41% → 16.5%**
  (IQR 12.2–22.0, range 5.8–46.1); 131.9 h; 94,654 ripples; 181 of 212
  derivations; comparison table gained a third column; added that exclusion is
  decided at a **fixed 1.0 s reference pad** so a pad sweep cannot move the
  sample.
- `methods_04` — 181 derivations, 94,654 events, rate 0.195 Hz, amplitude
  1.37 µV; candidate denominator flagged as not regenerated.
- `methods_01`, `methods_02` — marked **STALE** (2026-09-03 bundle, 46 sessions).
  Not retyped: their numbers come from the localisation and artifact stages, so
  refreshing them means rerunning those, not editing text.

Still outstanding: the IED-clustering χ² in `methods_03` (p = 0.18) is from the
~8-session development set and needs `swr_rejection_bias.py` rerun; the
right/left split and the candidate count need `swr_export.py numbers` on
`bundle_v2`.


## 2026-09-17 (i) — Stage 3, definitive: location content is NOT ripple-specific

Supersedes (e), (f), (h). Two changes, both from SK:

1. **Flanks matched by CONSTRUCTION** (`swc.matched_flanks`) — each flank sits
   inside the ripple's OWN occupancy interval, so it shares location, dwell,
   task phase and width, and never overlaps another ripple. This replaces the
   fixed +-500 ms offsets, which were location-matched only ~53.7% of the time.
   It also keeps far more ripples than the post-hoc matched SUBSET used in (h),
   because it needs the interval to hold ripple+flank, not +-500 ms of stillness.
2. **Per-cell z-scoring of spike counts** (`swc.zscore_cells`) — `E = sum_c n_c *
   z_c(L)` weighted each cell by its raw rate, so a few loud cells carried the
   pattern. Matches the project's single-spike RSA practice.

### Result

| ROI | sess | ripples | flanks | C0 z_ripple | p | z_flank | **C1 z_diff** | p |
|---|---|---|---|---|---|---|---|---|
| HC_mid | 35 | 39,832 | 57,476 | +0.304 | 0.078 | **+0.406** | -0.050 | 0.762 |
| **HC_anterior** | 51 | 58,050 | 82,608 | **+0.391** | **0.0035** | +0.240 | +0.119 | 0.389 |
| mOFC | 26 | 27,216 | 39,272 | -0.474 | 0.083 | +0.199 | -0.476 | 0.037 |
| mPFC | 32 | 36,224 | 52,247 | +0.291 | 0.124 | +0.212 | -0.036 | 0.87 |

**Location content IS present inside ripple windows** (HC_anterior p = 0.0035)
**but it is NOT ripple-specific.** Matched windows in the same dwell at the same
location carry as much or more — for HC_mid the flank EXCEEDS the ripple
(+0.406 vs +0.304). Hippocampus represents where you are continuously; a ripple
is not a moment when that representation is enhanced.

Every earlier positive C1 was the unmatched-flank artefact.

### The cell z-scoring also changed WHICH region leads

Stage 2 positive control at 60 ms, raw counts vs per-cell z:

| ROI | raw | cell-z |
|---|---|---|
| HC_mid | +0.513 (p = 0.008) | +0.678 (p = 0.0057) |
| **HC_anterior** | +0.108 (**p = 0.48**) | **+0.517 (p = 0.0011)** |

HC_anterior was never location-blind — loud cells were swamping it. **Earlier
statements that HC_anterior "fails the current-location control" were an
artefact of the estimator, not a fact about the region**, and the claim that
HC_mid is uniquely the location region is weakened accordingly.

⚠ mOFC C1 is significantly NEGATIVE (-0.476, p = 0.037, subject p = 0.048). One
of eight tests, in a region with no location tuning, and a negative
ripple-specific effect has no interpretation. Treated as an unexplained
systematic artefact, NOT a result. Worth finding.

### Error tally for this analysis line — four now

All four inflated the headline in the favourable direction; none was caught by
the analysis's own controls; three of four were caught by SK asking a pointed
question:
1. (e) estimator/inclusion mismatch between Stage 3 and its Stage 2 control;
2. (g) CV leak — configurations repeat across blocks (SK);
3. (h) flank not location-matched;
4. (i) raw spike counts let loud cells dominate the population read-out (SK).

**What stands:** hippocampus (both subregions) carries current-location
information readable from 60 ms windows. **What does not:** any claim that
ripples carry location information beyond ambient coding.

## 2026-09-17 (h) — ⚠ RETRACTION: the Stage 3 ripple-content effect does not survive a location-matched flank

**Found by the audit SK asked for.** The +-500 ms flank was described throughout
entries (e)-(g) as sharing "trial, current location, phase and stillness" with
the ripple. **The location part is false.** Measured: a flank sits at the SAME
location as the ripple only **53.7%** of the time.

When it does not, the flank window is scored for the location the subject
occupied AT THE RIPPLE — the wrong one — so its evidence is systematically
depressed and ripple-minus-flank comes out positive for a reason that has
nothing to do with ripples. That is a mechanism which manufactures precisely the
C1 effect that was reported.

### Restricting to ripples whose flanks ARE at the same location

`FLANK_MODES = ("all", "matched")` in `scripts/swr_ripple_content.py`.

| HC_mid | all ripples (45,303) | **matched (12,770)** |
|---|---|---|
| **C0** z_ripple | +0.386 (p = 0.044) | **-0.013 (p = 0.941)** |
| **C1** z_diff | +0.316 (p = 0.069) | **-0.055 (p = 0.756)** |
| C1 subject-level | p = 0.161 | p = 0.919 |

**Both collapse to zero.** C0 does not involve the flank at all — it changes
because the matched subset is a DIFFERENT population of ripples: those where the
subject held one location for +-500 ms, i.e. long dwells.

### What this means

**In the ripples where a controlled comparison is possible, there is no location
content at all.** The apparent effect came from ripples where the subject was
moving, and there the flank is at a different location by construction.

⚠ The matched subset is a narrower question, not a pure control: it restricts to
still periods. Ripples cluster in still periods (F1), so it is the right
population to ask about, but it does not exclude location content in ripples
during movement. It does exclude the claim that was made.

Other ROIs, matched: HC_anterior +0.276 (p = 0.109) in-ripple, C1 null;
**mOFC -0.562 (p = 0.008) in-ripple**, significantly NEGATIVE, unexplained and
probably a systematic bias worth understanding; mPFC null.

### The proper fix, not yet built

Stop using fixed +-500 ms offsets. Place each flank INSIDE the ripple's own
occupancy interval, at whatever offset that interval allows — location-matched
by construction, without discarding moving ripples. Until that exists, C1 as
specified cannot be run on the full ripple set.

### Running tally of errors in this analysis line

Three now, all inflating the headline in the favourable direction, none caught
by the analysis's own controls:
1. (e) estimator/inclusion mismatch between Stage 3 and its Stage 2 control;
2. (g) cross-validation leak — configurations repeat across blocks (SK);
3. (h) the flank control was not location-matched.

The Stage 2 positive control (HC_mid reads current location from 60 ms non-ripple
windows, +0.562, p = 0.0056) is unaffected by all three and still stands. Nothing
about ripple CONTENT does.

## 2026-09-17 (g) — ⚠ CROSS-VALIDATION LEAK: configurations repeat across blocks. All of (c)-(f) re-run.

**Found by SK.** Every prior result in entries (c), (d), (e) and (f) used
leave-one-GRID-out where the unit was `grid_num` (the block index). But a
configuration is run in several blocks: **median 24 grid_num vs 8 grid_id per
session, ~3 runs per configuration (up to 5), in 63/63 sessions.** Holding out
one block therefore left 2-4 other runs of the SAME configuration in the
training set, so the template already encoded that configuration's reward
locations and trajectories.

**Fix:** `mc/analyse/swr_content.occupancy` now emits `cv_group = grid_id` and
`place_map` filters on it; every call site and the split-half reliability split
were updated. The numbers below SUPERSEDE entries (c)-(f).

### Effect on each result

| quantity | leaky (grid_num) | corrected (grid_id) |
|---|---|---|
| Stage 2, HC_mid 60 ms | +0.780 (p = 0.00026) | **+0.562 (p = 0.0056)** |
| Stage 2, HC_anterior 60 ms | +0.167 (n.s.) | +0.077 (n.s.) |
| **Stage 3 C0**, HC_mid | +0.459 (p = 0.022) | **+0.386 (p = 0.044)** |
| **Stage 3 C1**, HC_mid | **+0.411 (p = 0.012)** | **+0.316 (p = 0.069)** |
| Stage 3 C1, subject-level | p = 0.040 | **p = 0.161** |
| C2 `known_rew` (target) | +0.033 (n.s.) | -0.070 (n.s.) |
| C2 `unknown_rew` (control) | +0.320 (p = 0.066) | +0.218 (p = 0.217) |

**The headline claim is walked back.** "Ripples carry location information AND it
is ripple-specific" becomes: C0 marginal (p = 0.044), **C1 a trend only
(p = 0.069)**, subject-level not significant. The ripple-SPECIFICITY — the half
that distinguishes real content from ambient hippocampal coding — is
**unresolved**, not established.

### The leak explains part of the `unknown_rew` anomaly, not all of it

A template trained on other runs of the same configuration knows that
configuration's reward locations, including ones not yet discovered at ripple
time — exactly the spurious positive seen in (f). Correcting it cut the
coefficient from +0.320 to +0.218 (p 0.066 -> 0.217), so it is no longer a red
flag firing, but it did not go to zero. **Roughly a third explained; the
remainder is unaccounted for.** I11/I12/I13 all rest on reward-location
regressors, so this should be understood before any of them is built.

### Reliability moved in OPPOSITE directions by region — informative, not just a fix

Split-half now splits by configuration, so the two halves contain DIFFERENT
configurations:

| ROI | leaky | corrected |
|---|---|---|
| **HC_mid** | +0.0886 | **+0.1127 (p = 5.7e-6)** — UP |
| HC_anterior | +0.0812 | +0.0626 (p = 0.0032) — down |
| mOFC | +0.0539 | +0.0390 (n.s.) |
| PCC | +0.1114 | +0.0899 (n.s.) |
| mPFC | -0.0272 | +0.0315 (n.s.) |

HC_mid gets STRONGER when the halves are forced to hold different
configurations: its location code generalises across configurations, i.e. it is
genuinely spatial. HC_anterior gets weaker, so part of its apparent tuning was
configuration-specific. A real dissociation, exposed by the fix.

### ⚠ Pattern worth noting

Two independent methodological errors have now been found in this analysis line
— the estimator/inclusion mismatch in (e) and this CV leak — and **both inflated
the headline in the favourable direction before being caught**. Neither was
found by the analysis's own controls. An audit pass for the same class of error
is warranted before any new analysis is layered on top.

## 2026-09-17 (f) — C2: no evidence for REMOTE reward locations in ripples (underpowered null); plus the raw spike-rate check

**New:** `scripts/swr_ripple_content_c2.py`, `scripts/swr_spike_rate_in_ripples.py`.
**Results:** `.../group/swr/ripple_content_c2_2026-09-17/`,
`.../group/swr/ripple_spike_rate_2026-09-17/`.

### Firing IS higher inside ripples — by about 4%

Rate inside the ripple extent vs all other in-task time, per cell:

| ROI | in-ripple Hz | outside Hz | ratio | % cells up | p (session) |
|---|---|---|---|---|---|
| HC_anterior | 2.835 | 2.650 | 1.04 | 61.1% | 0.00012 |
| HC_mid | 3.378 | 3.228 | 1.04 | 63.0% | 0.025 |
| mOFC | 3.312 | 3.238 | 1.01 | 58.2% | 0.127 |
| mPFC | 2.268 | 2.246 | 1.00 | 50.7% | 0.111 |
| PCC | 3.134 | 2.856 | **1.13** | 80.3% | 0.041 |

**The 0.2 spikes-per-ripple figure is the 60 ms WINDOW, not silent cells** —
3 Hz x 0.06 s = 0.18. But the ripple modulation is only ~4% in hippocampus,
reliable yet far below the severalfold increases the rodent literature assumes.
Consistent with the weak peri-ripple modulation in Fig 1f. Plausible cause:
ripples are detected on MACRO bipolar derivations while units come from
microwires, so a detected ripple need not coincide with local ripple activity at
the wire. Worth stating in methods — the content effects are extracted from a
much weaker ripple-locked firing regime than rodent work.

⚠ **PCC shows the LARGEST modulation (1.13, 80% of cells)** — stronger than
hippocampus, which makes no mechanistic sense. 61 cells in 10 sessions. Do not
report PCC without understanding this.

### C2 — the target is null, and the knowledge control misbehaves

All nine locations enter ONE regression per window, because the roles are
correlated by construction (a known reward is necessarily a recently-visited
place): `E(L) ~ is_current + is_known_reward + is_unknown_reward + recently_visited`.

HC_mid, 35 sessions, 41,624 ripples, ripple minus matched +-500 ms flanks:

| term | mean z | 95% CI | p |
|---|---|---|---|
| current | **+0.466** | [+0.111, +0.822] | **0.012** |
| **known_rew** (target) | **+0.033** | **[-0.353, +0.419]** | 0.863 |
| unknown_rew (knowledge control) | +0.320 | [-0.022, +0.662] | 0.066 |
| recent (recency control) | +0.112 | [-0.245, +0.469] | 0.527 |

**`current` reproduces Stage 3** (+0.466 vs +0.411) inside a completely
different estimator, which is a good internal-consistency check: the null is not
a broken pipeline.

⚠ **This is an UNDERPOWERED null, not an absence.** The known-reward CI excludes
only effects above +0.419 — **90% of the current-location effect**. It rules out
"remote rewards are represented about as strongly as where you are standing" and
essentially nothing weaker.

⚠ **The knowledge control is trending positive and is LARGER than the target**
(+0.320 vs +0.033). The subject CANNOT know the undiscovered reward locations,
so that coefficient should be zero. Templates are leave-one-grid-out and
neighbouring grids carry different configurations, so reward locations should not
be systematically distinctive — no explanation yet. It may be noise (p = 0.066,
one of 32 tests in this table), but **until it is understood, no reward-location
claim should rest on this design**, including a positive one.

Other ROIs: all null for the target. mPFC `known_rew` is NEGATIVE in-ripple
(-0.477, p = 0.025) but mPFC fails the Stage 2 control and the direction is not
predicted; treated as noise.

### What this does and does not say

C2 pools ripples across exploration AND execution. It does not test SK's
sharper questions — rewarded vs non-rewarded locations during exploration
(I11), whole-trajectory replay once the route is known (I12), or blocking at
error locations (I13), all now queued in `POTENTIAL_IDEAS.md`. A pooled null is
weak evidence against any of them.

## 2026-09-17 (e) — STAGE 3: hippocampal ripples carry location information (HC_mid), exploratory

**New:** `scripts/swr_ripple_content.py`; `figure5` in `swr_content_figures.py`.
**Results:** `.../group/swr/ripple_content_stage3_2026-09-17/`.
First test of ripple CONTENT in this project. Everything before was foundations.

### Result — session level, the project's ephys convention

| ROI | sess | ripples | **C0** z_ripple | p | z_flank | **C1** z_diff | p | subj p |
|---|---|---|---|---|---|---|---|---|
| **HC_mid** | 35 | 45,303 | **+0.459** | **0.022** | +0.166 | **+0.411** | **0.012** | 0.040 |
| HC_anterior | 51 | 67,414 | +0.131 | 0.341 | +0.097 | +0.005 | 0.971 | 0.81 |
| mOFC | 26 | 30,868 | -0.177 | 0.533 | +0.005 | -0.175 | 0.408 | 0.95 |
| mPFC | 32 | 41,752 | -0.164 | 0.426 | -0.151 | -0.055 | 0.777 | 0.56 |

C0 = location signal inside the ripple (t_peak +- duration/2) against the
label-permutation null. C1 = the same minus equal-width flanks at +-500 ms,
which share trial, current location, phase and stillness. **The effect is
ripple-SPECIFIC and confined to HC_mid**, the only ROI that passed the Stage 2
current-location control.

### ⚠ EXPLORATORY, and the reason must be recorded

The first Stage 3 run gave C0 p = 0.10 and C1 p = 0.12. Two flaws were then
found and fixed, and the fixed analysis is significant. **The analysis was
changed after seeing a null.** Both fixes are defensible independently, but this
is exploratory and must not be written up as confirmatory.

1. **The estimator did not match its own control.** Stage 2 used
   `target_minus_others`; Stage 3 used `target_z`. C0's stated purpose is to be
   comparable to the Stage 2 number, which is impossible if they are computed
   differently. Primary is now `target_minus_others`, identical to Stage 2;
   `target_z` is kept as a secondary in the `*_sf` columns.
2. **The inclusion rule was biased AGAINST the effect.** `target_z` needs
   sd > 0, discarding ripples where few cells fired, and MIN_RIPPLES = 100 then
   dropped whole sessions: HC_mid fell 35 -> 20. The dropped sessions had a
   HIGHER mean Stage-2 z (**+0.920**) than those kept (**+0.675**) — verified
   numerically before the fix was run. `target_minus_others` needs no such
   guard (an empty window scores 0, which is unbiased), so all 35 return.

⚠ A subject-level p = 0.019 reported from the biased-inclusion run was an
artefact of that inclusion and is p = 0.21 under it. Superseded by the p = 0.040
above, which comes from the corrected run.

### An unexplained comparison, stated rather than smoothed over

With estimators now matched: ordinary navigation windows **+0.780** (Stage 2),
ripple flanks **+0.166**, ripple windows **+0.459**. So the NEIGHBOURHOOD of a
ripple carries much weaker location information than navigation generally, and
the ripple partially restores it. Consistent with F1 (ripples occur during
stillness, when ongoing location coding should be weakest), but equally
consistent with a sampling difference between tiled and ripple-centred windows.
Not resolved. C1 is unaffected either way, being internally matched.

### Vectorisation: the null is free

`E = C.T @ T`, so `C.T @ T[:, perm] == E[:, perm]` — a permuted template is a
column reordering of evidence already computed, and `sum_L E(L)` (and the row
mean/SD) are permutation-invariant. Every null draw is therefore one fancy-index
into a precomputed array: no spike counting, no matmul, no per-permutation loop.
**Whole analysis runs in 37 s** for 61 sessions x 4 ROIs x 100 permutations.
N_PERM reduced 200 -> 100 at SK's request; memory stays at a few (n_ripples x 9)
arrays per session.

### Not yet done

C2 (the OTHER reward locations, beyond the current one) — the first genuinely
non-trivial content question. Memory-vs-plan and sequential order remain
exploratory. The pseudo-population decoder is still to be built as the
complementary pattern-level read-out.

## 2026-09-17 (d) — Stage 2 PASSES for HC_mid; the "inverted U" was a sampling artefact

**Supersedes the Stage 2 and width-sweep results in entry (c).**
`scripts/swr_place_width_sweep.py` now tiles every eligible window;
`tiled_windows` added to `mc/analyse/swr_content.py`.
**New:** `scripts/swr_content_descriptives.py`, `scripts/swr_content_figures.py`,
`scripts/swr_location_timecourse.py`.
**Results:** `.../group/swr/ripple_content_descriptives_2026-09-17/`,
`.../ripple_content_figures_2026-09-17/`, `.../ripple_content_timecourse_2026-09-17/`.

### ⚠ The sampled window draw was unstable and must not be used

Entry (c) sampled 1,500 random windows per session. That estimate moves with the
seed: **HC_mid at 60 ms came out +0.30, +0.44 and +0.53** across three runs
differing only in the draw. The apparent rise from 60 -> 125 ms, read in (c) as
a spike-budget effect, was that noise. Every width-profile conclusion in (c) is
withdrawn.

### Stage 2, deterministic: every eligible window, tiled

Per-session z vs own 200-draw label-permutation null, t across sessions.

| ROI | 60 ms | 125 ms | 250 ms |
|---|---|---|---|
| **HC_mid** | **+0.753 (t=3.94, p=0.00038)** | **+0.716 (t=4.00, p=0.00032)** | **+0.687 (t=3.76, p=0.00065)** |
| HC_anterior | +0.183 (0.252) | +0.165 (0.282) | +0.195 (0.206) |
| mOFC | +0.237 (0.34) | +0.224 (0.375) | +0.153 (0.504) |
| mPFC | -0.097 (0.636) | -0.071 (0.741) | -0.099 (0.65) |

624,309 windows at 60 ms for HC_mid, against 1,500 sampled. Removing the
subsample both RAISES the estimate and FLATTENS the width profile.

**HC_mid carries the current location in a 60 ms window — the width of a
ripple.** The Stage 2 gate passes, and passes decisively rather than marginally.
HC_mid being the current-location region agrees with `per_lag_encoding` (HC_mid
peaks at lag 0); mPFC being null under a lag-INVARIANT place template agrees
with it too.

### Reliability-weighting: equivocal, not adopted

Within-run (identical sessions and windows) it helped HC_mid at 125 ms
(+0.432 -> +0.650) and HC_anterior at 60 ms (-0.003 -> +0.194), and slightly hurt
HC_mid at 60 ms (+0.527 -> +0.493). Deterministic windowing mattered far more.
⚠ The reliability used was full-data split-half, i.e. mildly circular; if it is
ever adopted it must be recomputed leave-one-grid-out.

### Descriptives and figures (`swr_content_figures.py`)

Fig 1 data and budget; Fig 2 location tuning; Fig 2b example place maps (best
cells, r up to 0.94); Fig 3 the width profile; Fig 4 location evidence aligned
to arriving at a new location. Every panel's numbers are written as CSV beside
the figure.

⚠ **Peri-ripple firing (Fig 1f) must be per-CELL normalised.** The first version
summed raw spike counts across cells, so the highest-rate cells dominated a
panel whose entire purpose is checking spike-ripple alignment. Now each cell is
expressed as % change against its own -500:-250 ms baseline before averaging.

### Location evidence over time (Fig 4)

Aligned to arriving at a new location, sliding 125 ms windows. Peak evidence for
the location just entered: HC_mid +0.125 s (+0.032), mOFC +0.062 s (+0.093),
mPFC +0.062 s (+0.018); HC_anterior shows no clear peak (max at -0.125 s).

### Full deterministic width profile: plateau then decline, no rise

| ROI | 60 ms | 125 ms | 250 ms | 500 ms |
|---|---|---|---|---|
| **HC_mid** | **+0.780 (0.00026)** | **+0.741 (0.00030)** | **+0.699 (0.00059)** | +0.374 (0.047) |
| HC_anterior | +0.167 | +0.173 | +0.192 | +0.051 |
| mOFC | +0.192 | +0.234 | +0.150 | +0.137 |
| mPFC | -0.093 | -0.050 | -0.048 | -0.068 |

At 1 s only 8-11 sessions retain >=200 eligible windows and 2 s drops out, so
those columns are not interpretable. **The decline beyond 500 ms is real and
survives tiling (windows straddle locations, median dwell ~0.7 s); the RISE at
the short end did not.** Per-window decoding accuracy is 0.10-0.12 against
1/9 = 0.111 at every width -- only the pooled score carries signal, as the
spikes-per-ripple arithmetic predicts.

### ⚠ Figure 4 broke rule 2 and was rebuilt

The first location-timecourse figure averaged RAW scores across sessions. The
tell was constant offsets with no dynamics (HC_anterior ~-0.04 throughout, mOFC
~+0.07) -- per-session scale leaking into the group mean. Rebuilt with
`swc.target_z`, a within-window z across the 9 location scores, which removes
the scale at source rather than per session.

Rebuilt: **HC_mid shows a clean crossover** -- evidence for the location just
left decays through arrival while evidence for the location just entered rises,
peaking at +0.19 s (z = +0.063), crossing at ~t = 0. HC_anterior flat; mOFC weak
(peak +0.31 s); mPFC consistently NEGATIVE (~-0.02), which is what a
future-coding region would do under a current-location template.

### Where this leaves Stage 3

The Stage 2 gate is passed for HC_mid at ripple width, so the ripple-vs-flank
test is worth running -- against the earlier read in (c), which was based on the
unstable sampled estimate. HC_anterior, mOFC and mPFC do not pass the
current-location control and any ripple result in them would be uninterpretable
under a plain place template.

## 2026-09-17 (c) — Stage 1a/2 of the ripple-content analysis: templates work, the location signal is small, and it peaks at 125-250 ms

**New:** `mc/analyse/swr_content.py`, `scripts/swr_place_templates.py`,
`scripts/swr_place_width_sweep.py`.
**Results:** `.../group/swr/ripple_content_templates_2026-09-17/`,
`.../group/swr/ripple_content_widthsweep_2026-09-17/`.

THE ESTIMATOR. Each cell gets a location template `z_c(L)`, its mean rate at
each of the 9 locations, z-scored across locations so `sum_L z_c(L) = 0`. A
window of spiking scores every location as `E(L) = sum_c n_c * z_c(L)`. Because
every template sums to zero, location-INDEPENDENT firing contributes exactly
zero in expectation, so a session with four cells still gives an unbiased score.
Templates are leave-one-grid-out. The statistic is `E(target) - mean(E(others))`.

### Stage 1a — real-time templates are 2.5x more reliable than the warped ones

Split-half (odd vs even grids) on place maps built from the Stage 0 timeline and
raw spikes:

| ROI | n | mean r | p |
|---|---|---|---|
| HC_anterior | 293 | **+0.0812** | 0.00017 |
| HC_mid | 227 | **+0.0886** | 0.0011 |
| PCC | 61 | +0.1114 | 0.029 |
| mOFC | 141 | +0.0539 | 0.104 |
| mPFC | 148 | -0.0272 | 0.406 |

Against +0.034 (HC) from the warped `all_location_snippets.csv`. Building maps in
real time, on the correct clock, roughly doubles the reliability. mPFC is flat,
as expected if its code is lag-tagged rather than a plain place code.

### ⚠ The raw across-session t-test is invalid — use per-session z

`E` scales with firing rate and cell count, so per-session scores are not
commensurable (|score| vs n_cells: **r = 0.49**; per-session score ranges
-0.75 to +0.61). The first Stage 2 run averaged raw scores across sessions and
returned null everywhere (HC_anterior t = +0.86, HC_mid t = +0.07). Re-scoring
each session against its OWN 200-draw label-permutation null and then testing
across sessions:

| ROI | sessions | mean z | t | p | frac z>0 |
|---|---|---|---|---|---|
| **HC_mid** | 35 | **+0.436** | +2.36 | **0.024** | 0.69 |
| HC_anterior | 51 | +0.219 | +1.49 | 0.142 | 0.65 |
| mPFC | 32 | -0.106 | -0.50 | 0.622 | 0.41 |
| mOFC | 26 | +0.048 | +0.23 | 0.816 | 0.46 |
| PCC | 10 | -0.043 | -0.15 | 0.886 | 0.60 |

⚠ HC_mid is one test in an uncorrected family of five and would not survive
correction across ROIs. ⚠ The z-normalisation was chosen AFTER the raw version
failed. Scale-invariance is not negotiable and the reasoning is sound, but it
was not pre-declared and should be reported as such.

### The width sweep: an inverted U, peaking at 125-250 ms

Same non-ripple navigation windows, widened. Per-session z vs own null:

| width | spk/cell (HC_mid) | HC_mid z (p) | HC_ant z | mPFC z (p) |
|---|---|---|---|---|
| 0.060 s | 0.196 | +0.303 (0.063) | -0.056 | -0.100 |
| **0.125 s** | 0.399 | **+0.468 (0.012)** | +0.210 | +0.004 |
| 0.250 s | 0.806 | +0.398 (0.020) | +0.218 | +0.067 |
| 0.500 s | 1.643 | +0.243 | +0.103 | +0.241 |
| 1.000 s | 3.385 | +0.113 | +0.127 | +0.271 |
| 2.000 s | 7.127 | +0.064 | +0.007 | **+0.410 (0.046)** |

**The signal does NOT grow with spike count.** The rise 60 -> 125 ms is the
spike budget; the fall after ~250 ms is windows STRADDLING LOCATIONS -- the
median dwell is ~0.7 s, so a 1-2 s window spans several locations and "the
current location" stops being well defined. (At 2 s only long dwells are even
eligible, so the sample also thins to 25-34 sessions and skews to reward
dwells.) The inverted U is the signature of a working estimator with a genuine
short-end spike limit, not of a broken one.

**mPFC runs the opposite way** — monotonically increasing, significant only at
2 s. Consistent with mPFC coding task-level rather than instantaneous location.

### Full deterministic width profile: plateau then decline, no rise

| ROI | 60 ms | 125 ms | 250 ms | 500 ms |
|---|---|---|---|---|
| **HC_mid** | **+0.780 (0.00026)** | **+0.741 (0.00030)** | **+0.699 (0.00059)** | +0.374 (0.047) |
| HC_anterior | +0.167 | +0.173 | +0.192 | +0.051 |
| mOFC | +0.192 | +0.234 | +0.150 | +0.137 |
| mPFC | -0.093 | -0.050 | -0.048 | -0.068 |

At 1 s only 8-11 sessions retain >=200 eligible windows and 2 s drops out, so
those columns are not interpretable. **The decline beyond 500 ms is real and
survives tiling (windows straddle locations, median dwell ~0.7 s); the RISE at
the short end did not.** Per-window decoding accuracy is 0.10-0.12 against
1/9 = 0.111 at every width -- only the pooled score carries signal, as the
spikes-per-ripple arithmetic predicts.

### ⚠ Figure 4 broke rule 2 and was rebuilt

The first location-timecourse figure averaged RAW scores across sessions. The
tell was constant offsets with no dynamics (HC_anterior ~-0.04 throughout, mOFC
~+0.07) -- per-session scale leaking into the group mean. Rebuilt with
`swc.target_z`, a within-window z across the 9 location scores, which removes
the scale at source rather than per session.

Rebuilt: **HC_mid shows a clean crossover** -- evidence for the location just
left decays through arrival while evidence for the location just entered rises,
peaking at +0.19 s (z = +0.063), crossing at ~t = 0. HC_anterior flat; mOFC weak
(peak +0.31 s); mPFC consistently NEGATIVE (~-0.02), which is what a
future-coding region would do under a current-location template.

### Where this leaves Stage 3

Per-window decoding accuracy is at chance (0.10-0.12 against 1/9 = 0.111) at
EVERY width and in every ROI, as the spikes-per-ripple arithmetic predicts.
Only the pooled score carries signal, and its peak is z ~= 0.47 per session.
Stage 3 asks for a ripple-minus-flank DIFFERENCE, which must be smaller than
that absolute signal. **On these numbers Stage 3 is underpowered as designed.**
Reliability-weighting the cells is being tested before that is accepted.

## 2026-09-17 (b) — Stage 0 of the ripple-CONTENT analysis: real-time location timeline, full spike cache, ROI alignment

Foundations for asking what hippocampal ripples CONTAIN (as opposed to when they
happen, `SWR_SUMMARY.md`, or whether cortex responds, `ripple_locked_HFB.md`).
No hypothesis is tested here. Three gates, all passed.

**New:** `mc/analyse/swr_location.py`, `scripts/swr_build_location_timeline.py`,
`scripts/swr_verify_cell_roi_alignment.py`.
**Results:** `.../group/swr/location_timeline/` (63 step tables + settings +
`cell_roi_alignment.csv`).

### Gate 1 — the location timeline is exact

`abcd_data.trial_vars` carries `start_location` / `end_location` per move with
`grid_onset_timestamp` giving each move's time on the TRIGGER clock, i.e. the
same clock as spikes and ripples. Extracted for all 63 sessions: **303,413
steps**.

Validation against the bundle's 97,643 uncover events, whose location is known
independently: **97,641 reproduced = 99.998%**, every session >= 99%.

⚠ **Two sources that look right and are not — do not retry either:**

1. **`all_location_snippets.csv` cannot be aligned to ripples.** It is built on
   a 360-bin NORMALISED trial averaged across correct repeats, i.e. time-warped.
   Mapping ripples onto it puts only **7.2%** inside a snippet. It remains a
   good source of location TEMPLATES and a useless source of timing.
2. **Integrating arrow presses from `press_categories.csv` gives 62.9%.** The
   grid is column-major (RightArrow +3, LeftArrow -3, UpArrow -1, DownArrow +1;
   learned from uncover pairs separated by one press, not assumed), and even
   re-anchoring at every uncover the error rate is 22.9% after a single move.
   Drift is not the problem; the movement model is.

⚠ `button_pressed_timestamp` is a Matlab clock and the mat file itself renames
it `DONOTUSE_button_pressed_timestamp`. Use the trigger-derived fields.

⚠ Session numbering: session = mat index + 1. The `session_num` FIELD is 1 in
every session and means "which recording file within this session".

### Gate 2 — the spike cache now covers every ripple session

`load_spike_times` defaults to `DSR_SESSIONS` (28); the mat holds all 63.
Re-run over the 61 ripple sessions: **965 cells, 8,740,798 spikes**, up from
564 cells / 4.57 M. This removes the 28-session constraint that shaped the
earlier ripple-RSA work.

### Gate 3 — ROI labels are row-aligned in all 61 sessions

`cell_roi_table` joins positionally, and that was verified for the 28 DSR
sessions only. Re-checked by matching the mat's `electrodeLabel` against the ROI
table's `electrode label` per cell: **61/61 PASS**, 965 cells on both sides.
The positional join is safe for the full set.

### Coverage this buys (inside the ripple extent, ~60 ms, 61 sessions)

| ROI | sessions | cells | ripples | spikes | rip >=2 cells | rip >=3 cells | pair-obs |
|---|---|---|---|---|---|---|---|
| HC_anterior | 51 | 293 | 85,918 | 83,308 | 13,733 | 5,929 | 41,307 |
| HC_mid | 35 | 227 | 59,211 | 78,528 | 13,845 | 5,904 | 38,394 |
| mPFC | 32 | 148 | 55,699 | 48,813 | 9,515 | 4,437 | 28,012 |
| mOFC | 26 | 141 | 36,669 | 28,262 | 4,851 | 1,740 | 14,358 |
| PCC | 10 | 61 | 17,701 | 20,352 | 3,422 | 1,517 | 11,083 |

**92,470 of 96,597 deduped ripples (95.7%) fall inside the task with a known
location.** HC combined: **79,701 pair-observations** inside the ripple extent
and **11,833 ripples with >= 3 co-active HC cells** — enough for a pooled
temporal-bias statistic and a per-ripple rank-order statistic respectively.

EC is 35 cells in 8 sessions under `atlas_roi` and is not analysable as a
primary ROI. The EC 3-vs-51 disagreement between `atlas_roi` and
`neurons_MNI_latest.csv` (CHANGELOG 2026-09-15 d) is still unresolved.

### Two preliminary facts that shaped the design

- **Single-cell location tuning is near zero.** Split-half (odd vs even grids)
  on the 9-location rate map: HC mean r = +0.034 (p = 0.09), reward-phase only
  +0.043 (p = 0.022); ACC +0.017; OFC path-only +0.081 (p = 0.017). ~53% of
  cells positive.
- **The POPULATION nonetheless decodes location.** Correlation decoder, held-out
  grids, chance 11.1%: all cells 44.6%, HC 26.9%, ACC 21.3%, OFC 24.4%.
  Generalising to a configuration never trained on: 18.9% overall, HC 18.0%,
  ACC 21.4%; OFC (12.1%) and EC (9.1%) do not survive. The drop is partly
  interpretive (much within-config signal is "this location is rewarded in this
  config") and partly fewer cells — not separated yet.
- ⚠ **Per-cell lag tags from `per_lag_encoding` are too noisy to use as hard
  labels.** Across the 12 lags, peak-minus-mean is ~0.34 in EVERY ROI against an
  across-lag SD of ~0.21 — the peak sits ~1.6 SD above the mean, which is what
  12 pure noise draws give. mPFC is if anything slightly LESS peaked than HC
  (0.319 vs 0.343, p = 0.017). The group-level signal is real (mPFC best-lag
  concentrates 26% at 30-60 deg vs 16.7% expected) but the per-cell label is
  not. Weight cells continuously by r at every lag; never assign a cell "its" lag.

## 2026-09-17 — Peri-ripple frontal HFB by what the ripple follows

**New:** `scripts/swr_ripple_hfb_conditions.py`. **Results:**
`.../group/swr/ripple_hfb_conditions_2026-09-17/`. Session-level, different-shaft,
runs from `bundle_v2`.

Each ripple is assigned to its nearest preceding labelled event within 2 s.
Target = a rewarded uncovering during the first traversal (`reward_explore`);
five matched controls. Each condition gets its own shifted null, because the
conditions differ in state by construction (F1: rate rises with stillness; F3:
the first traversal is stiller) so the state confound does NOT cancel.

### Per condition (session-level, real minus shifted null)

| condition | mPFC | mOFC | Frontal |
|---|---|---|---|
| **reward_explore** (target) | −0.0001 (0.98) | **+0.0110 (0.0003)** | +0.0053 (0.039) |
| error_explore | +0.0033 (0.30) | +0.0019 (0.49) | +0.0031 (0.12) |
| reward_plan | +0.0026 (0.67) | −0.0057 (0.32) | −0.0007 (0.88) |
| reward_execute | **+0.0051 (0.0004)** | **+0.0089 (<0.0001)** | **+0.0075 (<0.0001)** |
| move_explore | +0.0014 (0.66) | +0.0037 (0.28) | +0.0023 (0.39) |
| still_explore | −0.0051 (0.58) | +0.0105 (0.17) | +0.0066 (0.19) |

### Contrasts against the target, count-balanced pairwise

**mOFC vs error_explore: +0.0098 ± 0.0040, p = 0.017** — same phase, same
action, opposite feedback. vs reward_plan +0.0166, p = 0.053 (n = 23). Every
other contrast null, in every ROI.

Critically, **reward_explore ≈ reward_execute in mOFC** (−0.0009, p = 0.80).
So this is not explore-specific or novelty-specific: both rewarded conditions
show the response. The dissociation that survives is **reward vs error**, not
**new vs known**.

### Three things that qualify this

1. **The error contrast is not robust to the balancing scheme.** Balancing all
   six conditions to the smallest gave −0.0014, p = 0.77; balancing pairwise —
   subsampling only the larger of each pair — gave +0.0098, p = 0.017. Pairwise
   is the better design (global balancing cut the target from 46 ripples to 30
   even when its comparison had 68) but the swing is large and the result
   should be treated as unresolved rather than established.
   ⚠ The first version of this script promised count-balancing in its docstring
   and did not implement it. Found on reading back; both schemes are now
   computed and the unbalanced number is printed alongside.
2. **Multiple comparisons.** 3 ROIs × 5 contrasts = 15 tests; one reaches
   p < 0.05 and one more p = 0.053, against ~0.75 expected by chance. That is
   weak evidence on its own; what supports it is that both are in the ROI with
   the robust main effect and both run in the predicted direction.
3. **`still_explore` has 11–15 sessions** and is badly underpowered — its wide
   intervals are not evidence of absence.

### Collapsing mPFC + mOFC is a bad idea

The `Frontal` column dilutes everything: the target drops from +0.0110 (mOFC)
to +0.0053, and **no contrast survives**. mPFC's target is flatly null
(−0.0001, p = 0.98) while its `reward_execute` is significant (+0.0051,
p = 0.0004) — the two regions are not doing the same thing, so averaging them
mixes signal with noise. Report them separately.


## 2026-09-16 (d) — Sequence figure (fMRI-style), and why mOFC coverage differs between spikes and HFB

**New:** `scripts/swr_ripple_rsa_sequence_figure.py`.
**Results:** `.../group/swr/ripple_rsa_sequence_2026-09-16/`.

### Why mOFC looked underpowered for spikes but well-powered for HFB

Not a bug — the two measure different things.

| ROI | cells | cell **sessions** | HFB derivations | HFB **sessions** |
|---|---|---|---|---|
| mOFC | 74 | **10** | 107 | **19** |
| mPFC | 65 | 15 | 32 | 17 |
| HC_anterior | 171 | 26 | 47 | 26 |
| Visual | 21 | **1** | 81 | 18 |

mOFC units fire fastest of any ROI (0.32 spikes/ripple, ~4.6 Hz), so
spikes-per-ripple is high — but they sit in only **10 sessions**. A
(config x state) RDM cell is populated only if a session that HAS mOFC cells
also had a ripple in that config and state, so 10 sessions leaves 65 % of
mOFC's config pairs unestimable. **RDM coverage is driven by SESSIONS, not by
firing rate.** HFB mOFC spans 19 sessions, hence the difference.

Same reason the Visual spike "control" was meaningless (21 cells, all from ONE
session) while the HFB Visual control is real (81 derivations, 18 sessions).

### The sequence figure

Direct analogue of the fMRI instruction figure: x-axis is the four reward
DISCOVERIES instead of 12 instruction seconds, and the two lines are the two
model families — `known_set` ("memory": locations uncovered so far) and
`full_abcd` ("plan": the whole configuration). They are identical at D by
construction. Ripples from the first 1 s and first 2 s after each uncover,
capped at the next uncover. Pad 0.25 s, 2000-draw config-relabel null.

**mPFC, 0-1 s: both models rise monotonically A -> B -> C -> D**
(full_abcd: -0.31, -0.11, -0.13, +0.26), which is the shape the fMRI figure
shows and the shape the hypothesis predicts. It is the cleanest-looking result
in the whole project so far. But every point sits inside the +-1 SD null band
except D, and D reaches only p = 0.13. In the 0-2 s window the same panel is
flat (+0.01 at D).

**Two cells reach p < 0.05 uncorrected, and neither helps:**
- mOFC `known_set` at B, +0.45, p = 0.031 (0-1 s)
- PCC `full_abcd` at A, +0.39, p = 0.038 (0-2 s) — **state A is where the
  subject CANNOT know the configuration**, so this is the built-in confound
  detector firing, not a result.

## 2026-09-16 (c) — Bootstrap, pooling, and the HFB variant is built

### CLARIFICATION: the time-resolved figure plots rho, not p

`timeresolved_D.pdf` row 1 is the **Spearman correlation** between data and
model RDM. There are no p values in that figure. **Positive = predicted.** The
dotted lines at 0 s and 1 s are the boundaries of the old post-press window.

### Bootstrap over sessions — the most encouraging number so far

Resampling the 27 sessions with replacement, 500x, uncover D, window 0-1 s:

| pad | ROI | observed | bootstrap 95% CI | % resamples positive |
|---|---|---|---|---|
| 1.00 | **mPFC** | **+0.563** | **[+0.154, +0.713]** | **99 %** |
| 1.00 | PCC | +0.020 | [-0.388, +0.255] | 52 % |
| 1.00 | HC_anterior | -0.260 | [-0.398, +0.055] | 6 % |
| 0.25 | **mPFC** | **+0.266** | **[-0.058, +0.633]** | **96 %** |
| 0.25 | PCC | +0.007 | [-0.247, +0.279] | 61 % |

So the mPFC value is **consistent across sessions** — it is not one subject.
That is a genuine point in its favour and was not visible in the permutation.

**But the two tests answer different questions and both are right.** The
bootstrap resamples SESSIONS while holding the 8 configurations and the model
fixed, so it measures stability across subjects; it cannot see the 28-pair
resolution limit. The config-relabel permutation measures exactly that limit.
An effect can be stable across sessions AND be within what random
configuration labelling produces — which is this one at pad 0.25 (p = 0.101).

### Pooling all four uncovers: nothing

Collapsing A/B/C/D into one condition per configuration (8 conditions, model =
full ABCD, 4x the ripples) across 3 pads x 2 schemes: every ROI p > 0.088,
mPFC between -0.107 and +0.090. No pooled configuration signal.

### HFB variant: BUILT, positive control passes, design limit unchanged

`hfb_roi_table`, `load_hfb`, `cache_hfb_rates` in `mc/analyse/ripple_rsa.py`.
Interchangeable with the spike path — `patterns_from_cache`, `rdm_for`,
`fit_rho` and every null work unchanged.
**Results:** `.../group/swr/ripple_rsa_hfb_2026-09-16/hfb_first_pass.csv`.

Coverage in the 28 DSR sessions (597 usable derivations): mOFC 107 / 19
sessions, Visual 81 / 18, HC_anterior 47 / 26, mPFC 32 / 17, HC_mid 28 / 19.
Signal is float16 at 100 Hz on the same session clock; robust-z (median/IQR)
per derivation before windowing, because gains differ by orders of magnitude.

**Positive control passes** (mean robust-z inside vs outside ripples):
HC_anterior +0.204, HC_mid +0.213 — but these are CIRCULAR, ripples are
detected on those derivations and 80-120 Hz sits inside the 70-150 Hz band, so
they are flagged `is_ripple_source`. Cortical: mPFC +0.021, mOFC +0.007,
Visual -0.006. Cortical HFB barely moves at hippocampal ripples here, matching
the weak spike coupling found earlier.

**First pass, 51 cells (2 pads x 2 schemes x 5 ROIs x states B/C/D):**
7 reach p < 0.05, but they are **dominated by Visual** — the negative control —
at +0.357 (p = 0.035), +0.333, +0.420 (p = 0.025) and +0.473 (p = 0.007), plus
HC_anterior at state B. mPFC is estimable only in the interval scheme
(12 derivations) at +0.238, p = 0.184.

A negative control lighting up as strongly as anything else says the apparent
structure is not ROI-specific.

**And HFB does not fix the resolution problem:** median null SD = 0.212 against
the 28-pair floor of 0.192. More features and a continuous signal do not narrow
a correlation over 28 pairs. That remains the binding constraint.

## 2026-09-16 (b) — I1: ripple-locked cortical HFB. Medial frontal yes, visual null; survives the pad sweep

**New:** `scripts/swr_ripple_locked_hfb.py`. **Results:**
`.../group/swr/ripple_locked_hfb_2026-09-16/`. Runs from `bundle_v2`, no cluster.

61 sessions, 42 subjects, 505 cortical derivations, ~740k ripple-alignments.
He et al. windows: epoch ±750 ms, **peri = ±250 ms**, **non-peri = (−750:−250) ∪
(+250:+750)**. Estimate = peri − non-peri (each ripple against its own flanks),
then **minus a shifted null**. Both windows must be artifact-free in the
hippocampal *and* the cortical derivation.

### Result — different-shaft derivations only

| ROI | derivations | subjects | effect (z) | t | p |
|---|---|---|---|---|---|
| **mOFC** | 100 | 32 | **+0.0083** | 4.71 | **<0.0001** |
| **mPFC** | 51 | 30 | **+0.0040** | 4.38 | **0.0001** |
| Lat. temporal | 129 | 35 | +0.0028 | 3.70 | 0.0008 |
| Auditory | 38 | 18 | +0.0016 | 1.44 | 0.17 |
| Visual | 97 | 20 | +0.0011 | 0.51 | 0.62 |

mOFC > mPFC (+0.0050, p = 0.016) and > lateral temporal (+0.0062, p = 0.015).
Sharp transient peaking within ~100 ms of the ripple peak, back to baseline by
±250 ms. **Replicates He et al.'s medial-frontal-yes / visual-no dissociation**,
with medial OFC the stronger of the two medial frontal regions.

### It passes the pad-stability check that the ripple-RSA effect failed

Same standard as the entry below. 0.75 → 3.0 s takes the usable set from
169,697 to 67,353 alignments (−60%):

| ROI | 0.75 | 1.00 | 1.50 | 2.00 | 3.00 |
|---|---|---|---|---|---|
| mOFC | +0.0079 | +0.0078 | +0.0086 | +0.0079 | +0.0060 |
| mPFC | +0.0040 | +0.0039 | +0.0033 | +0.0025 | +0.0046 |
| Lat. temporal | +0.0029 | +0.0027 | +0.0024 | +0.0025 | +0.0032 |
| Visual | +0.0011 | +0.0008 | −0.0001 | −0.0016 | −0.0015 |

Flat-to-declining, not growing — the healthy direction. **mOFC is robust
(p < 0.02 throughout); mPFC is the wobblier one**, dipping to p = 0.080 at pad
2.0 s before recovering. Auditory drifts up at the largest pads (p = 0.052,
0.072) without ever reaching significance, which at that n reads as noise.

⚠ **The first version of this sweep was vacuous and looked reassuring.** It ran
0.10/0.25/0.50/1.00 and gave a perfectly flat line — because a ±750 ms epoch
must be artifact-free, so **every event nearer than 750 ms to a crossing is
already excluded at any detection pad** (26.7% of the bundle). Three of those
four pads were therefore the identical subset. The sweep now starts at the
epoch half-width, where the comparison is real. A flat sweep is only evidence
if the subsets actually differ.

### Three controls, each of which changed the answer

1. **The reference must be a SHIFTED NULL, not zero.** Against zero *every*
   region is positive, controls included (Visual p = 0.044, Auditory p = 0.044)
   — ripples cluster in states (F1) and cortical HFB tracks state too. Shifting
   ripple times 5–120 s leaves ~0 everywhere. This is what makes the controls
   null.
2. **Same-shaft derivations are volume-conduction contaminated**, and the
   2026-09-13 note recommending them as "the tightest control" was wrong about
   which confound they control. The *entire* Visual effect came from 9
   derivations on the hippocampal electrode (+0.0115, p = 0.039) against
   +0.0011 and null on the other 97. Same-shaft is a good control for recording
   quality, a bad one for anatomical specificity. Primary set is different-shaft.
3. **Figure aggregation must match the test** (found by SK). The time-course
   panel averaged over derivations while the bars averaged over subjects; for
   Visual those differ fourfold (+0.0044 vs +0.0011) because its coverage is
   concentrated — up to 14 derivations in one subject. The figure showed Visual
   responding and mPFC not, the opposite of the statistics beside it. Traces are
   now stored flat with an index and aggregated subject-level.

### Status

**Exploratory.** All three controls were chosen after seeing the naive result,
so they are justified on mechanism rather than by prior declaration. No
confirmatory run on held-out sessions. Effect sizes are small in absolute terms
(0.003–0.008 z) and rest on very large alignment counts — the dissociation and
the time-course shape are the evidence, not the p-values.

⚠ The non-peri flanks are the baseline, so a response broader than ±250 ms
would leak into its own baseline and be under-estimated. The measured responses
return to ~0 by ±250 ms, so this does not bite here.


## 2026-09-16 (b) — Time-resolved, descriptive: what the pad is actually changing

**New:** `scripts/swr_ripple_rsa_timeresolved.py`.
**Results:** `.../group/swr/ripple_rsa_timeresolved_2026-09-16/` (no tests, no
p values anywhere in this entry's figures).

### The pad is a UNIFORM ~30 % thinning

Spearman(lag from uncover press, `dist_to_artifact_s`) = **+0.005, p = 0.69**.
At pad 1.0 s, ~30 % of ripples are removed in EVERY lag bin (0-0.5 s: 29.4 %;
0.5-1 s: 31.4 %; 1-2 s: 29.5 %; 2-4 s: 34.7 %; 4-8 s: 33.5 %). The pad is not
preferentially deleting press-locked events, so "the pad removes the
interesting ripples" is ruled out.

### Therefore the pad-dependence is resampling noise — shown directly

At pad 0.1 (window 0-1 s, 1254 ripples), randomly dropping 33 % of ripples
300 times:

| ROI | full-data rho | range under random thinning | SD |
|---|---|---|---|
| **mPFC D** | **+0.104** | **-0.386 to +0.679** | 0.200 |
| PCC D | +0.056 | -0.486 to +0.641 | 0.172 |
| HC_anterior D | -0.097 | -0.464 to +0.426 | 0.134 |

**2.7 %** of random thinnings reach the pad-1.0 value of +0.563 or more.
Dropping a random third of the data moves rho by up to ±0.4, so the gap between
+0.10 and +0.56 carries essentially no information. Full write-up in
`random_thinning_result.md`.

### Time-resolved view (uncover D, `known_set`, pads 0.25 and 1.0)

CUMULATIVE (all ripples from the press to t) and SLIDING (1.5 s window stepped
through), both capped at each event's own interval end so nothing leaks into
the next reward's condition.

- **Cumulative curves start extreme and decay toward 0 as ripples accumulate.**
  At pad 1.0, mPFC begins at +0.59 with ~50 ripples and falls monotonically to
  ~-0.05 by t = 4 s (~570 ripples). At pad 0.25 the same curve begins at +0.34.
  The published-looking value is the left edge of a decaying curve, i.e. the
  sparsest point.
- **Sliding curves oscillate around 0 in every ROI with no sustained window.**
  If a subset of ripples carried the plan, a sliding window should find a
  stretch where the fit is consistently positive. None exists.
- Ripple counts: cumulative reaches ~870 (pad 0.25) / ~640 (pad 1.0) by 8 s;
  sliding peaks at ~420 / ~300 around 1.5 s.
- **Firing rate inside ripples is genuinely event-locked**: mOFC falls from
  ~4.1 Hz at the press to ~1.8 Hz by 2 s; HC_mid shows the opposite hump. So
  the data do contain real uncover-locked dynamics — they just do not carry
  configuration structure.

### Anatomy: mPFC and mOFC do NOT overlap here

SK asked whether the "mPFC" entry zone might sit in dorsal mOFC. It does not:
mPFC cells sit at MNI z ~ -5 to +13 (mode ~0 to +5), mOFC at z ~ -25 to -12,
with a clean gap between. Caveat: the 65 mPFC cells come from only ~8 distinct
microwire locations (cells cluster on bundles), so this is 8 sites, not 65
independent ones — which is itself part of why the estimate is unstable.

## 2026-09-16 — Artifact-pad sweep on the swr_v2 bundle: the mPFC effect runs backwards

**New:** `scripts/swr_ripple_rsa_pad_sweep.py`; `pad_s` filtering and
`default_bundle_dir()` in `mc/analyse/ripple_rsa.py`.
**Results:** `.../group/swr/ripple_rsa_pad_sweep_2026-09-16/`.

⚠ **Path change:** the old `group/swr/bundle/` is now `bundle_08.09.2026/` and
the new one is `bundle_v2/`. Hardcoded `.../swr/bundle` paths are stale;
`rrsa.default_bundle_dir()` now resolves this.

### The new bundle

`swr_v2` detects at a **0.1 s** artifact pad and carries `dist_to_artifact_s`,
so larger pads are nested subsets. 100,737 events vs 64,760 in swr_v1 (1.56x);
**pad 1.0 s reproduces swr_v1 almost exactly** (64,895 vs 64,760), so that is
the pad the old results were computed at.

`meta.json` warns that re-padding also needs exposure rebuilt from
`artifact_intervals`. **That caveat does not apply here**: this analysis
measures firing DURING ripples and never divides by artifact-free seconds, so
filtering events is sufficient and complete.

Still absent from the export: `t_start_s` / `t_end_s`. The `peak ± duration/2`
approximation therefore stands — worth adding on the next export, it costs
nothing (the columns already exist per-session).

### The sweep: 5 pads x 2 schemes x 5 ROIs x 4 states x 2 models = 332 cells

**The pre-declared cell, mPFC at D, window scheme — it grows as ripples are
REMOVED:**

| pad (s) | ripples at D | pairs missing | rho | p |
|---|---|---|---|---|
| **0.10** (most data) | **311** | 0 | **+0.104** | **0.300** |
| 0.25 | 290 | 0 | +0.266 | 0.101 |
| 0.50 | 257 | 2 | +0.440 | 0.025 |
| 1.00 (old bundle) | 209 | 4 | **+0.563** | 0.017 |
| 2.00 | 139 | — | not estimable | — |

Monotonic, and backwards. A real effect gets clearer with more data; this one
is largest where the data are thinnest and the RDM is least complete. At the
native pad, with 50% more ripples and a complete RDM, it is rho = +0.10,
p = 0.30. The interval scheme shows no mPFC D effect at any pad
(-0.084 to -0.180).

### The whole sweep produces fewer hits than chance

**11 of 332 cells reach p < 0.05 uncorrected = 3.3%**, against the ~5% expected
under the global null. The hits do not concentrate in any ROI, state, model or
pad: mOFC B at pads 0.10/0.25, HC_mid D at 0.25, mPFC D at 0.50/1.00, mPFC B at
0.50/2.00, HC_mid A at 2.00. That is the signature of no effect.

**The pad is not a free parameter.** Choosing 1.0 s because the effect is
biggest there is exactly the post-hoc gate this project avoids; the sweep is a
stability check, and this effect fails it.

### Null width does not improve with more ripples

Median null SD across the sweep: 0.200-0.222, against the 28-pair floor of
1/sqrt(27) = 0.192. Going from 2,610 to 5,866 ripples (interval scheme) moves
it from 0.199 to 0.203 — i.e. not at all. **The binding constraint is the 8 x 8
RDM, not the ripple count.** More ripples cannot fix this design; more
conditions could.

### Where this leaves the analysis

The entry (f) mPFC result is now contradicted from two independent directions:
loosening the ripple-to-condition rule removes it (entry g), and adding ripples
at a smaller artifact pad removes it (here). It should not be carried forward.

## 2026-09-15 (g) — Interval scheme (supervisor's suggestion), and the mPFC D effect does not survive it

**New:** `ripples_in_intervals` + `SCHEMES` in `mc/analyse/ripple_rsa.py`;
`scripts/swr_ripple_rsa_diagnostics.py`.
**Results:** `.../group/swr/ripple_rsa_diagnostics_2026-09-15/`.
The `window` scheme is kept and still runnable — entry (f) is not superseded,
it is contextualised.

### The interval scheme: 4.7x the data, complete RDMs

Every ripple in [uncover_k, uncover_{k+1}) is assigned to state k; state D runs
to the NEXT repeat's t_A. Three advantages: the intervals tile the first
traversal exactly once so **no ripple can enter two conditions**; the knowledge
state is **constant** throughout each interval, which is exactly what the
knowledge-gated model describes; and coverage jumps.

| scheme | ripples | per (config x state) | RDM pairs missing |
|---|---|---|---|
| post 0–1 s window | 885 | 28 | 0–23 depending on ROI |
| inter-uncover interval | 4040 | **126** | **0 everywhere** |

Ripples per state (interval): A 1011, B 1179, C 1132, **D 718** — D is the
thinnest because the interval to the next repeat's t_A is short (median 3.1 s).

### ⚠ The mPFC D effect does not survive

| ROI / state | post 0–1 s window | whole interval |
|---|---|---|
| **mPFC D** | **+0.325** (p = 0.059) | **-0.135** (p = 0.726) |
| PCC D | +0.248 (p = 0.129) | +0.200 (p = 0.162) |
| HC_anterior D | -0.328 | -0.220 |

### The diagnostic that settles it: early vs late within the interval

If the effect were genuinely time-locked to the discovery, it should live in the
first second and be ~0 afterwards. Splitting the SAME intervals:

| ROI / state | early 0–1 s (885) | late 1 s–end (3155) | whole (4040) |
|---|---|---|---|
| **mPFC D** | **+0.325** | **-0.445** | -0.135 |
| PCC D | +0.248 | +0.342 | +0.200 |
| HC_anterior D | -0.328 | -0.274 | -0.220 |

**mPFC flips sign**, and the late half is as extreme negative as the early half
is positive — both ~1.6-2.2 SD of a null whose SD is 0.2. That is what a coin
flip looks like across 20 cells, not a time-locked effect. By contrast **PCC is
the only ROI positive in every subset** (+0.248 / +0.342 / +0.200), though it
reaches p = 0.06 at best.

Conclusion: the entry (f) mPFC result was a window-specific fluctuation. It
should not be carried forward as a finding.

### The two nulls are nearly the same distribution

Plotted as histograms (`null_distributions.pdf`) rather than error bars: the
config-relabel null and the surrogate-window null have almost identical width
and centre. So the surrogate is NOT a tighter test — it answers a different
question (does the window have to be at a ripple) and cannot tell you whether
there is any configuration structure at all. **Keep both**: config relabel as
the statistical null, surrogate as the specificity control.

### Model support (`model_support.pdf`)

Configuration pairs sharing >= 1 reward location, by state: A **0/28**,
B 5/28, C 17/28, D 24/28. Rank offset of shared locations: **offset 0 never
occurs at any state** — which is why a strict order model has literally no data
and `position_locked` is flat. r(known_set, known_seq) = 1.00 / 0.97 / 0.92 at
B / C / D, so the two are not separable over 28 pairs.

### Runs per configuration

A session contributes a median of 3 grids per configuration
(`coverage_sweep.pdf` panel d). Currently all their ripples are pooled into one
pattern. Pooling is NOT biased for the off-diagonal — two different
configurations never share ripples, so their noise is independent — but a
cross-run RDM (odd vs even grid occurrences, as the DSR pipeline does with its
`across_only` scope) would additionally protect against slow drift shared by
configurations recorded close in time. Worth adding now that the interval
scheme makes each cell well populated. NOT yet implemented.

## 2026-09-15 (f) — Controls for the ripple RSA, and why the null is 0.19 wide

`scripts/swr_ripple_rsa_controls.py` + a fast cached path in
`mc/analyse/ripple_rsa.py` (`cache_ripple_rates`, `patterns_from_cache`,
`fit_rho`, `fit_pooled`) — verified to reproduce `collect_spike_patterns`
exactly, 82x faster, which is what makes 2000-draw permutations affordable.
**Results:** `data/ephys_humans/derivatives/group/swr/ripple_rsa_controls_2026-09-15/`.

### The null width is arithmetic, not a pipeline quirk

An 8 x 8 RDM has **28 unique pairs**, so a chance Spearman correlation over it
has SD = 1/sqrt(27) = **0.19**. Measured null SDs: config relabel 0.207,
surrogate window 0.185, pooled-over-84-pairs 0.122 (predicted 0.110),
post-minus-pre 0.291 (a difference carries both variances). All four match
their arithmetic prediction, which is the check that the nulls are built
correctly.

Consequence: **with 28 pairs you need |rho| > ~0.37 for p < 0.05.** A rho of
0.3 that would be large in an RDM with hundreds of cells is 1.5 SD here. The
fix is more RDM cells, not more permutations.

### Four controls

| test | draws | what it asks |
|---|---|---|
| C1 config relabel | 2000 | is the fit about configuration identity? |
| C2 surrogate window | 200 | does the window have to be at a ripple? |
| C3 pooled B+C+D | 2000 | same question, 84 pairs instead of 28 |
| C4 post − pre | 1000 | is the fit specific to after the press? |

C2 is the specificity control: each ripple's window is moved to a random time
inside the SAME +-1 s press window of the SAME uncover event, keeping its
duration. Same events, configs, states, cells and window count — the only
thing removed is that the window sat on a ripple.

### Result: mPFC at D, and C2 separates it from PCC

| ROI | state | rho | C1 p | **C2 p (ripple-specific)** | C4 p |
|---|---|---|---|---|---|
| **mPFC** | **D** | **+0.325** | 0.058 | **0.040** | 0.107 |
| mOFC | D | +0.285 | 0.214 | 0.060 | — |
| PCC | D | +0.248 | 0.129 | **0.125** | 0.174 |
| HC_anterior | D | -0.328 | 0.952 | 0.960 | 0.954 |

mPFC at D is the pre-declared primary cell and is the only one to clear
p < 0.05 on any control. **The surrogate-window control is what separates it
from PCC**: mPFC's fit is specific to windows that sat on a ripple (z = +1.95),
PCC's is not (z = +1.11). Neither survives FDR across the 14 cells of the C2
family (mPFC p_fdr = 0.42), and C1 does not reach 0.05 (p = 0.058).

**The effect is D-specific, not an accumulation.** Pooling B+C+D (C3) gives
mPFC rho = +0.111 on 74 pairs, z = +0.93, p = 0.173 — the D effect is diluted
by B (+0.079) and C (-0.094). That argues against "working memory accumulating
with each reward" and for "the plan is assembled once the configuration is
complete", which is the fMRI claim. It also means the pooled test, which would
have had the tighter null, is the wrong test for this effect.

HC_anterior runs NEGATIVE throughout (D: -0.328; pooled -0.198, z = -1.70) —
configurations sharing more locations have LESS similar hippocampal patterns.
No account offered.

 2026-09-15 (e) — Ripple RSA on raw spikes, +-1 s windows, ripple-extent firing

Supersedes (c) and (d). `scripts/swr_ripple_rsa_plan.py` (binned-matrix
version) DELETED; replaced by `scripts/swr_ripple_rsa_spikes.py`. The
binned-data collection path was removed from `mc/analyse/ripple_rsa.py`.

**Results:** `data/ephys_humans/derivatives/group/swr/ripple_rsa_spikes_2026-09-15/`.

### What changed

- **Firing measured INSIDE the ripple**: `t_peak +- duration_s/2`, rate =
  spikes / window width, from raw `abcd_passed.mat` spike times.
- **Press windows widened to +-1 s** (was [-0.35, 0] / [+0.15, +0.70]). The
  narrow pre-window left a median of 8 ripples per (config x state) and no
  estimable RDM at all. Now: **715 pre / 885 post ripples**, median 22 / 28 per
  condition. Cost: the post window no longer sits only on the rate increase.
- **Silent cells dropped**: 145 (pre) / 132 (post) cells fire no spike in ANY
  ripple of that window and are removed — they are an all-zero column.
  **Individual zero counts are KEPT**: "silent in this ripple" is data, and
  dropping those would bias every rate upward.
- ROIs from `neurons_with_ROI_labels.csv` / `atlas_roi`.

### Positive control now PASSES for HC_anterior

Firing inside the ripple vs two equal-width windows offset by 250 ms either
side (equal width, to avoid the count-vs-window-length trap):

| ROI | sessions | units | change | t | p |
|---|---|---|---|---|---|
| **HC_anterior** | 26 | 171 | **+4.4%** | **2.16** | **0.041** |
| HC_mid | 19 | 145 | +2.9% | 1.05 | 0.310 |
| PCC | 6 | 51 | +2.4% | 1.27 | 0.259 |
| mPFC | 15 | 65 | +1.7% | -0.16 | 0.879 |
| mOFC | 10 | 74 | -0.4% | -0.40 | 0.702 |

Measuring the ripple's own extent is more sensitive than the fixed 0-200 ms
post-peak window used in (c), which gave +2.2%, p = 0.104 for pooled HC.

### ⚠ SIGN CORRECTION — this inverts the conclusions of the first version

The first version of this entry reported "negative rho = predicted". **That was
wrong.** Model and data RDMs are both DISSIMILARITIES, so a region that encodes
the model produces a POSITIVE correlation: two configurations sharing few
locations are far apart in the model AND far apart in the data. Verified by
simulation — patterns constructed so each cell fires for its own rewarded
location give **rho = +0.92** against `known_set`. All one-sided tests are
upper-tail. `fit_model` and the figure labels are fixed.

### known_set at D IS full_abcd

Verified `np.allclose` = True: once D is uncovered the subject knows the whole
configuration, so the knowledge-gated model and the fixed full-ABCD model
coincide exactly at state D. That is why their rho is identical there.
`known_seq` at D is NOT the same (rank-weighted, not a plain set).

The ordered "compare all A-rewards, all B-rewards, ..." model is
`position_locked`, and it is **constant at every state** (sd = 0.000) because
within a state all eight configurations have distinct locations. It cannot be
fitted. The testable model is the unordered set, as expected.

### Result: direction is right in mPFC, but nothing is significant

`known_set`, post-press, positive = predicted:

| ROI | state | rho | exact p (1-sided) | pipeline p | z vs pipeline null |
|---|---|---|---|---|---|
| **mPFC** | **D** | **+0.325** | **0.054** | 0.090 | +1.49 |
| PCC | D | +0.248 | 0.124 | 0.090 | +1.48 |
| mOFC | D | +0.285 | 0.204 | 0.155 | +0.84 |
| HC_mid | B | (full_abcd) +0.373 | 0.013 | — | — |
| HC_anterior | D | -0.328 | 0.941 | 0.990 | -1.73 |

mPFC at D is the largest effect in the predicted direction and the pre-declared
primary cell. It does **not** reach significance under either null (0.054
exact, 0.090 pipeline) and does not survive FDR across the three primary ROIs
(0.162). **PCC at D matches it almost exactly (+0.248, p = 0.090)** and PCC is
not a predicted region — which is the main reason not to read the mPFC value as
a result yet.

HC_anterior at D goes the OTHER way (-0.328, z = -1.73): configurations sharing
more locations have LESS similar hippocampal patterns. No account is offered;
at a null SD of 0.2 this is what the tail looks like.

### THE PERMUTATION: how big is chance here?

`pipeline_null` draws a fresh per-session configuration relabelling and re-runs
the WHOLE estimator -- patterns, cell centring, missing-data structure, RDM,
fit -- 100 times. Each session gets its own relabelling, destroying the
cross-session config alignment that pooling cells depends on. That alignment is
the signal, so this is the right null, and it passes through identical code
(CLAUDE.md rule 4).

**The null SD is ~0.20 for every ROI and every state.** That is the number to
carry around: a Spearman rho of |0.2| in this design IS chance, and |0.3| is
1.5 SD. Values of that size appear all over the table, in both directions, in
regions with and without a prediction.

At 100 permutations the p resolution is 0.01, so prefer z to p.

### post - pre contrast: built a proper null for it, and it is not significant

A difference of two Spearman rhos has no standard sampling distribution, so
`pipeline_null_contrast` draws ONE per-session relabelling and applies it to
BOTH windows before differencing -- the same arithmetic as the real data, with
session composition, cell coverage and ripple counts preserved.

| ROI | state | post-pre | null SD | z | p |
|---|---|---|---|---|---|
| mPFC | B | +0.334 | 0.296 | +1.14 | 0.13 |
| mPFC | C | +0.320 | 0.317 | +1.05 | 0.17 |
| mPFC | D | +0.394 | 0.331 | +1.08 | 0.14 |
| PCC | D | +0.309 | 0.358 | +1.12 | 0.13 |
| HC_anterior | D | -0.501 | 0.262 | -1.88 | 0.98 |

mPFC is positive at all three testable states, which is a coherent pattern, but
the contrast null is WIDER than the single-window null (SD 0.26-0.36, because a
difference carries both variances) and nothing reaches significance. Note also
that the mPFC contrast is driven as much by a NEGATIVE pre fit (-0.255, -0.414,
-0.069) as by a positive post fit. Results in `contrast_permutation.csv`.

HC_anterior flips sign between windows in the opposite direction to mPFC
(pre +0.239 / +0.173 at C / D, post -0.180 / -0.328).

### Figures

- `model_RDMs.png/.pdf` — all four models x four states, with sd printed and
  the constant (unfittable) panels outlined in red. Makes the counterbalancing
  visible: `known_set`/`known_seq` are uniform at A, `position_locked` is
  uniform everywhere.
- `fit_timecourse_A_to_D.png/.pdf` — 3 models x {post, pre, post-pre}, all
  ROIs. Panels ~6 cm wide, Arial 9-11 pt, 2.2 pt lines, per CLAUDE.md.
- `overview_all_ROIs.png/.pdf` — (a) plan-model fit A->D for every ROI, post
  solid / pre dashed; (b) observed effect against its re-estimated null;
  (c) how complete each RDM is; (d) split-half reliability, restricted to
  states with >= 10 observed pairs.
- `rdm_reliability.png` — split-half reliability, so a flat timecourse is not
  misread as a null.
- `data_RDMs_<roi>.jpeg` — the measured RDMs.

## 2026-09-15 (d) — Ripple RSA power, done properly on RAW spike times

Redo of (c) after SK pointed out that the analysis must start from
`abcd_passed.mat`, not the 25 ms binned per-grid matrices, and that firing
should be measured DURING the ripple.

**New:** `scripts/swr_ripple_rsa_power.py`; raw-spike loaders in
`mc/analyse/ripple_rsa.py` (`load_spike_times`, `cell_roi_table`,
`spike_counts_in_windows`, `ripples_near_events`).
**Results:** `data/ephys_humans/derivatives/group/swr/ripple_rsa_power_2026-09-15/`.

### Sources now used

- **Spikes:** `abcd_passed.mat` -> `abcd_data.neural_data(c).spikeTimes`, in
  seconds on the same session clock as behaviour and ripples
  (`save_iEEG_as_csv.m` bins these from 0 with no offset). Cached per session
  as .npz under `group/swr/spike_cache/`. 564 cells, 4.57 M spikes, 28 sessions.
- **ROIs:** `neurons_with_ROI_labels.csv`, column `atlas_roi`, joined
  positionally on `cell idx`. **Verified: that order matches the mat file's
  `electrodeLabel` for all 28/28 sessions.** This replaces
  `neurons_MNI_latest.csv`, which is not row-aligned for s27/s40/s50/s57/s60.

### Ripple onset/offset — no cluster work needed

`swr_detect.py` already computes `t_start_s`, `t_peak_s` and `t_end_s`, and all
three are present in the per-session `ripple_events.csv`. **Only the bundle
export drops them.** Re-exporting the bundle with those two columns is
sufficient; re-detection is not required. On s38 (the one session held locally,
461 accepted ripples, matching the bundle exactly) the peak sits essentially at
the centre of the event — median 23 ms before, 24 ms after, duration 59 ms — so
`peak +- duration/2` reconstructs the extent well in the meantime.

### THE POWER VERDICT: yes, still a problem, and it is arithmetic

Spikes per neuron per ripple, post-press window, 28 sessions:

| ROI | n cells | ±10 ms | ±duration/2 | ±100 ms | % zero (±dur/2) |
|---|---|---|---|---|---|
| HC_anterior | 171 | 0.043 | 0.185 | 0.521 | 87% |
| HC_mid | 145 | 0.072 | 0.278 | 0.787 | 83% |
| mPFC | 65 | 0.032 | 0.129 | 0.389 | 89% |
| mOFC | 74 | 0.115 | 0.319 | 0.810 | 78% |
| PCC | 51 | 0.039 | 0.148 | 0.471 | 88% |
| EC | 3 | 0.000 | 0.167 | 0.333 | 92% |

Firing rate inside ripples is 1.5-4.6 Hz depending on ROI. A ripple is ~60 ms.
**3 Hz x 60 ms = 0.18 spikes.** That product, not the analysis design, is the
constraint.

- **±10 ms is unusable**: 90-100% of (neuron, ripple) pairs contain no spike at
  all; EC contains none anywhere.
- Pooled over ALL 28 sessions, one (config x state) RDM cell is built from a
  median of **0.15 (mPFC) to 0.35 (HC_mid) spikes per neuron**. The RDM entry
  is therefore a comparison of near-empty count vectors.
- Nothing about this changes with the raw data — the binned matrices gave the
  same firing rates; what the raw data adds is the ability to ASK the question
  at ripple resolution, and the answer is that the question is not affordable
  in spikes.

Reaching ~5 spikes per neuron per condition would need roughly 20-30x more
ripples per condition. Available multipliers: all correct uncovers instead of
discoveries only (~7x, but the reward is then already known), pooling the four
states (4x, destroys the state axis), widening to ±100 ms (~2.8x, no longer
"during the ripple"). No combination preserves the design and the question.

**Conclusion unchanged from (c), now on the right data:** ripple-triggered
single-unit RSA is not affordable in this dataset. HFB remains the route —
`swr_extract_hfb` is continuous, has no spike-count floor, and covers every
derivation rather than only the sessions with microwires.

### ROI table disagreement — needs resolving

`atlas_roi` and `neurons_MNI_latest.csv` disagree substantially on the same 28
sessions: HC 316 (171 anterior + 145 mid) vs 236; **EC 3 vs 51**; PCC 51 vs 41;
mPFC 65 vs ACC 67; mOFC 74 vs OFC 68. The EC discrepancy is a factor of 17 and
changes whether EC is analysable at all. `atlas_roi` was used here because it
carries the project's canonical names and its row order is verifiable against
the mat file.

## 2026-09-15 (c) — Ripple-triggered RSA on human cells: UNDERPOWERED, not a null

The cell analogue of the instruction-phase fMRI RSA. Subjects discover A/B/C/D
rather than being shown them, so "how much of the plan is assembled" is indexed
by WHICH reward was just uncovered instead of by time within a 12 s instruction
period. Ripple-triggered firing at the discovery presses, 8x8 config RDM per
state, cells pooled across the 28 shared-config sessions.

**New:** `mc/analyse/ripple_rsa.py`, `scripts/swr_ripple_rsa_plan.py`.
**Results:** `data/ephys_humans/derivatives/group/swr/ripple_rsa_plan_2026-09-15/`.

### What the task design forbids (verified, not assumed)

Within any single state all 8 configs have DISTINCT reward locations
(A: 3 8 1 4 6 9 7 2; B: 7 2 9 8 4 1 3 5; C: 9 6 5 1 2 3 4 7; D: 5 7 8 3 9 4 2 6).
Therefore:

- The position-locked Hamming model used in the fMRI (`rewDSR`) has **sd = 0.000
  at every state** and cannot be fitted in a within-state RDM. The
  generalisation that survives is set overlap of the locations known so far.
- At state A every knowledge-gated model is constant (one known location, all
  eight distinct). A is only testable with the fixed full-ABCD model, where it
  is the knowledge null.
- **Current location is constant within every within-state RDM**, so a
  within-state effect cannot be a place code. The counterbalancing supplies the
  confound control for free — stronger than the pre/post window.
- `known_set` vs `known_seq` correlate r = 1.00 / 0.97 / 0.92 at B / C / D.
  Not separable over 28 pairs; they are NOT run as a horse race. The sequence
  question is asked as a rank-offset test instead.

### Why this is underpowered, with numbers

- 2,665 discovery events (correct, `is_discovery`, explore) in the 28 sessions.
- Ripples in the windows: **265** in pre [-0.35, 0], **570** in post
  [+0.15, +0.70]. That is **0.48 / 0.64 ripples per (session x config x state)**
  — 64% / 56% of condition cells empty. Condition-averaging per session is
  impossible; only cross-session cell pooling makes any RDM estimable at all.
- **The whole `pre` window is unusable**: not one ROI/state yields a complete
  8x8 RDM, so the pre-vs-post control could not be run.
- In `post`, **only HC** gives complete RDMs (median 51-64 cells per pair).
  mPFC/ACC, OFC and EC all have missing pairs at every state, so the exact
  permutation is not defined for them and they are reported as NaN.
- **Split-half reliability of the best-covered RDM (HC, post, D) is
  rho = -0.06.** The RDMs are noise.

### Positive control: WEAK, and it caps everything downstream

Peri (0:+200 ms) vs non-peri (-750:-250 and +250:+750) firing, all ripples,
session-level paired test:

| ROI | sessions | units | mean change | t | p | sessions positive | sign test |
|---|---|---|---|---|---|---|---|
| HC | 25 | 236 | +2.2% | 1.69 | 0.104 | 19/25 | **0.015** |
| ACC (mPFC) | 16 | 67 | +1.0% | 1.08 | 0.296 | 8/16 | 1.00 |
| EC | 9 | 51 | +4.6% | 1.61 | 0.147 | 6/9 | 0.51 |
| OFC | 8 | 68 | +0.7% | 1.13 | 0.297 | 5/8 | 0.73 |

**This is not a clock problem.** A +/-10 s lag scan of HC firing against ripple
peaks maxes at exactly **0.0 s** (`clock_lag_scan.csv`), and the firing matrices
are raw unsmoothed 25 ms spike counts (integers 0-7, lag-1 autocorrelation
0.013). The coupling is genuinely ~2-4%.

**Nor is it probe co-location.** Unit electrode labels are `m` + probe name, so
microwire and ripple-derivation probes can be compared directly. Only **66 of
564 units (10 of 28 sessions)** sit on a probe that also produced ripples — but
restricting to the row-aligned sessions, same-probe units show **+2.9% (n=5
sessions)** and different-probe units **+3.7% (n=13)**. Co-location does not
rescue the coupling; an earlier apparent +6.6% for different-probe was driven
by sessions whose neuron table is not row-aligned (see below).

### Results (all secondary except the one marked)

Pre-declared primary: state D, post window, HC and mPFC, `known_set`.

| ROI | state | model | rho | p (1-sided) | p FDR |
|---|---|---|---|---|---|
| HC | D | known_set | +0.112 | 0.722 | 0.722 | **PRIMARY — null, wrong sign** |
| mPFC | D | known_set | — | — | — | RDM incomplete, not estimable |
| HC | B | known_set | -0.339 | 0.012 | 0.054 |
| HC | B | full_abcd | -0.322 | 0.042 | 0.125 |
| HC | C | known_set | -0.266 | 0.073 | 0.132 |
| HC | A | full_abcd | -0.117 | 0.297 | 0.404 |

The knowledge-accumulation prediction (full-ABCD fit grows A -> D) is **not**
borne out: -0.117, -0.322, -0.095, +0.112 across A/B/C/D. The B and C values sit
in the predicted direction but do not survive FDR, sit in a family of 40-odd
tests, and come from RDMs with ~zero split-half reliability.

`rank_offset` at HC post-D is rho = +0.534, p = 0.007 uncorrected — reported in
`controls.csv` and **not** believed, for the same reliability reason.

### Conclusion

**This is an underpowered design, not a negative result**, exactly as
`POTENTIAL_IDEAS.md` I5 warned ("if it is thin, I5 is not a null result, it is
an underpowered one, and those are different claims"). Do not re-run it as-is.
What would change the answer, in order of expected value:

1. More ripples per condition — the binding constraint. Either loosen detection
   (the `ripple` envelope is stored continuously and is threshold-free) or drop
   to a coarser condition space that keeps config variance.
2. HFB instead of spikes: `swr_extract_hfb` covers every derivation, so the
   feature space is not limited to the sessions with microwires.
3. Accept that mPFC cannot support a pooled-cell RDM here (67 cells, 45%
   coverage per condition) and go per-contact + LME, as He et al. do.

### Incidental finding, needs fixing elsewhere

`neurons_MNI_latest.csv` is **not row-aligned** with
`all_cells_region_labels_sub<N>.txt` for **s27, s40, s50, s57, s60** (38%, 95%,
44%, 81%, 63% of rows agree). Anything that joins the neuron table to firing
matrix rows by position is wrong for those five sessions. This analysis uses
`swu.unit_labels` (the row-aligned file) throughout and is unaffected, but the
mismatch should be resolved at source.

## 2026-09-15 (b) — Direction regressor + order-independent instruction models

Replaces the masking approach (see the correction below) with fitting the
forward/backward cue explicitly, and adds a new model family.

**`scripts/fMRI_run_RSA_instruction.py`**
- `DIRECTION_REGRESSOR = 'direction'` — reserved name, 1 = the cell crosses the
  cue. Built from condition labels (`build_direction_RDM_blocks`), has variance
  in both scopes.
- `UNORDERED_CHANNELS` — `AB/ABC/ABCD_rew_instr_unordered`, with
  `d = 1 - |multiset overlap of the first k instructed locations| / k`. Multiset,
  not set: a location revealed twice counts twice, which keeps the
  0/0.25/0.5/0.75/1 scale of the ordered models. Cut from the same rewDSR
  A_reward chunks the ordered models use.
- `model_family()` — three families (`execution`, `instruction`,
  `instruction_unordered`) plus `direction`, because the set models are
  instruction models that CAN be fitted across task halves. `single_model_scopes`
  now keys on it, accepts an exact model name as an override, and accepts `[]`
  to mean "only ever fit inside a combo".
- Degenerate designs are now DROPPED, not fatal. A regressor with no variance in
  a scope is a property of the subject's task assignment, not a bug; failing the
  run would block all other models for that subject. Recorded in
  `degenerate_maps` in the settings summary. Genuine rank deficiency (collinear
  but non-constant) still raises.

**`condition_files/rsa_instruction_direction_and_unordered.json`** — 37 maps.
Singles: `direction` (within+across) and the three set models (within+across).
Within combos: each ordered `*_rew_instr + direction`, each set model +
direction. Across combos: each set model + its ordered execution counterpart,
with and without direction. No singles for the ordered instruction or execution
models — those are already estimated.

**No k=1 set model.** With one location per condition order cannot matter, so
`A_rew_instr_unordered` is arithmetically identical to `A_rew_instr` — verified,
the whole 20x20 RDM matches at r = 1.0000. Omitted rather than run as a
duplicate. It would also have been constant across halves.

**Why direction is in the across combos.** Across halves `direction` is exactly
orthogonal to every set model (r = 0.0000, because each task appears once
forward and once backward per half). But it correlates about -0.55 with the
execution models there, so once execution is in the design the partial
correlation reappears: +0.029 (AB), **+0.372 (ABC)**, **+0.454 (ABCD)**.
Marginally orthogonal is not conditionally orthogonal; leaving it out would let
execution absorb direction-driven variance.

**k=4 caveat.** The full multiset is reversal-invariant, so
`ABCD_rew_instr_unordered` cannot distinguish instructed from executed ORDER —
read it as a "locations this task uses" model. It is NOT equal to `ABCD_rew`
(r ~ 0.18), which is why every across combo carries its execution counterpart.

**Design pre-flight (sub-02, 01-TR4 proxy):** all fitted designs well
conditioned — within-half condition numbers 1.6-2.3, across-half 1.8-2.9.

**`mc/fmri_analysis/check_RSA_ran.py`** — mirrors the three families and the
exact-name override, so `expected_map_names` resolves the new names.

**`mc/fmri_analysis/submit_RSA_instruction_epochs.sh`** — default config changed
to this one. It had been left pointing at `rsa_instruction_samedirection.json`
by commit c32824a, which would have run the execution model under an `_instr`
name; that config is now marked `_DEPRECATED` in its own body.

## 2026-09-15 — CORRECTION: the same-direction mask cannot test the instruction model

SK spotted this. Verified, and it is exact: **on same-direction cells,
`ABCD_rew_instr` and `ABCD_rew` are bit-identical regressors** — max|diff| =
0.000, r = 1.0000, for the F-F and the B-B cells separately.

Reason: the backward execution sequence is the exact reverse of the forward
one, and reversing BOTH members of a pair preserves position-wise Hamming
distance. So for a B-B pair, `Hamming(reverse(a), reverse(b)) ==
Hamming(a, b)`, which is the instruction value. For an F-F pair the two models
are trivially the same. Masking to same-direction cells therefore does not
produce a cleaner instruction model — it produces the **execution** model under
an `_instr` name, and fitting it returned exactly the same beta (+0.00682) for
both.

### The design entanglement this exposes

The within-half RDM contains only two kinds of cell, and each kills one model:

| cells | n | `ABCD_rew_instr` | `ABCD_rew` | tests |
|---|---:|---|---|---|
| same-direction (F-F, B-B) | 40 | varies | **identical to instr** | reward overlap, not instruction |
| direction-crossing (B-F) | 50 | varies | **constant at 1.0** | instruction, but every pair crosses the cue |

`ABCD_rew` is constant on every crossing cell because a forward and a backward
sequence never match at any of the four positions. So the
instruction-vs-execution contrast is identifiable **only** on direction-crossing
cells — which is exactly where the "please backwards" cue differs.

The confound is therefore not removable by any means tried, and this explains
the whole series: no control regressor works because the confound is collinear
with the hypothesis *by construction*, and masking does not work because it
deletes the contrast entirely. It is a design-level entanglement, not an
analysis choice.

### The one clean fit that remains

On **B-F cross-task cells (40)**, `ABCD_rew` is constant AND direction-crossing
is constant (every pair is one forward with one backward, both orderings
present), so neither can confound anything, and the forced-zero cells are
excluded. Only the instruction model varies there:

    ABCD_rew_instr on B-F cross-task cells = -0.05547

That is an unconfounded test of whether instructed reward-sequence overlap
predicts instruction-period pattern similarity across a direction switch, and
it is clearly negative.

### Status of (d) and (e), written yesterday

`condition_files/rsa_instruction_samedirection.json` and the
`within_same_direction` scope are **not withdrawn but must not be run or
reported under the `_instr` name** — the maps it writes are the execution model.
The scope machinery itself is correct and the runner/audit changes in (e)
(including the smoothing-config collision fix) stand on their own. Decision
pending: drop the config, or rename it to what it actually fits (a
reward-sequence-overlap model on the cells where instruction and execution
coincide).

The 2026-09-14 (c) conclusion — the negative is a forward/backward effect, not
a time effect — is unaffected and still holds.

## 2026-09-14 (e) — Cluster runner + audit adjusted for the same-direction config

**`mc/fmri_analysis/submit_RSA_instruction_epochs.sh`**
- base config is now `${RSA_CONFIG:-rsa_instruction_samedirection.json}`, so the
  same runner drives either instruction config.
- audit dir is `rsa_audit_${configTag}_<date>`, so the two configs do not share
  a todo list.
- default wall time 240 -> 120 min. 240 was sized for 28 models (21 single + 7
  combo) on 90 cells; the same-direction config fits 7 single models on 40
  cells and reuses the searchlight cache. Under-requesting is cheap because the
  RSA resumes. `FSLSUB_T=240` for the cumulative-rew config.

**`mc/fmri_analysis/check_RSA_ran.py`**
- `SCOPE_ALIASES` / `SCOPE_TAGS` gained `within_same_direction` ->
  `within-samedir`, mirroring the runner. Without this the audit KeyErrors on
  the new config.
- **Bug found and fixed while adjusting**: the per-epoch *smoothing* config
  snapshots were named `{smooth_config}_{glm}.json`, with no reference to
  `name_of_RSA` — unlike the RSA snapshots, which are keyed on the base config
  name. Auditing two RSA configs over the same epochs therefore overwrote each
  other's smoothing snapshots, and the surviving one carries a single
  `name_of_RSA`, so the smoothing jobs of one config would have been pointed at
  the other config's results folder. Now keyed on `name_of_RSA` too. Safe
  rename: only this script builds these names and they reach the smoothing job
  via `todo_smooth.txt`.

**Verified locally against the real sub-02 run:**
- `expected_map_names` gives the 7 `*_within-samedir` names, matching the files
  the RSA actually wrote; the cumulative-rew config still resolves to its 41.
- `settings_differences` vs the written `sub-02_settings_summary.json` is empty
  and no expected beta map is missing — the run is correctly marked DONE, so
  there is no permanent RERUN_CHANGED loop.

## 2026-09-14 (d) — New RSA scope `within_same_direction` + masked instruction config

Implements the conclusion of (c). Two changes, no new pipeline script.

**1. `scripts/fMRI_run_RSA_instruction.py`** gains a fourth scope,
`within_same_direction` (aliases `within_samedir`, `same_direction`): the
within-half cells that do NOT cross the forward/backward cue. Added
`same_direction_mask` / `same_direction_mask_2d` next to `within_half_mask`,
registered in `SCOPE_ALIASES` / `SCOPE_TAGS` / `SCOPE_CAPTION`, and wired into
`_scope_cells`, `_mask_2d_for_scope` and `all_scopes_used` so the model
regressor, the cached data RDM and the plotted RDM are all subset identically.

Verified on the real sub-02 condition ordering: 190 -> 90 within-half -> 40
same-direction cells, composition 20 F-F + 20 B-B, zero same-task pairs, zero
cross-half pairs, 2-D display mask agrees with the flat mask.

**2. `condition_files/rsa_instruction_samedirection.json`** — new config,
`name_of_RSA = instr_cumrew_samedir`. The 7 `*_rew_instr` single models only;
`run_combo_models: false`; no execution/planning models, which are unaffected
by the direction cue and stay with `rsa_instruction_cumulative_rew.json`.

Because the searchlight cache is keyed on `regression_version + TR +
searchlight_mask` and NOT on `name_of_RSA`, this reuses
`data_RDMs_glmbase_01-TR4_grey_matter` and recomputes **no searchlights**
(cache mtime confirmed unchanged after the run). Results go to
`RSA_instr_cumrew_samedir_glmbase_01-TR4/`, every map suffixed
`_within-samedir`, so nothing collides with earlier runs.

Run: `python scripts/fMRI_run_RSA_instruction.py <subj_no> rsa_instruction_samedirection.json`

**Validation.** Test run on sub-02 exited 0 with `n_cells_per_searchlight: 40`
and wrote beta/t/p for all 7 models. The pipeline's
`ABCD_rew_instr_within-samedir_beta` reproduces the independent audit
implementation bit-for-bit: mean +0.006784 vs +0.006784, r = 1.00000000,
max |diff| = 5.6e-16.

Still to check when this is run on the group: the F-F vs B-B split-half
reliability per subject (section 10) — in sub-02 it was ~0, and if that holds
across subjects the mask is clean but the data are thin.

## 2026-09-14 (c) — Why `ABCD_rew_instr` is negative: NOT time

Follow-up to the event-locked instruction RSA audit, which found a consistent
negative whole-brain shift for the instruction/memory model across subjects.
Hypothesis tested: the negative reflects **temporal** structure — the two
directions of a task being acquired far apart, so drift makes them look
dissimilar where the instruction model forces them identical.

Audit script has since been deleted (one-off diagnostic, not part of the
pipeline). Its outputs are kept at
`data/derivatives/group/instruction_instr_negative_time_audit_2026-09-14/`.
The conclusion is implemented as the `within_same_direction` RSA scope.

**Scope caveat.** The only locally cached full neural RDM is the older
`glmbase_01-TR4` instruction GLM, sub-02 only — not the 2026-09 event-locked
GLMs. The model side is identical to the event-locked pipeline
(`corr(ABCD_rew_instr, ABCD_rew) = 0.2311` reproduces the stored design audit
exactly), so only the neural side is a proxy. Needs rerunning where the
event-locked caches live.

### Reconstruction is exact

Recomputed sub-02 betas vs the saved `ABCD_rew_instr_within_beta.nii.gz`:
r = 1.0000000, max |diff| = 3.9e-16 over 126,321 searchlights. The cell-level
attribution below is therefore the pipeline's own beta, not an approximation.

### 1. The data DO have a strong time structure

Instruction onsets taken from the `ev_*_instruction_onset.txt` files the
first-level GLM actually used (one block per task, 5 contiguous repeats).
Mean beta over searchlights for a pure time RDM:

| time RDM | mean beta |
|---|---:|
| log lag between onsets | **+0.1439** |
| linear lag | +0.1401 |
| block order gap | +0.1381 |

That is ~3x larger than any reward model effect here. Temporally distant
blocks are reliably more dissimilar. So the confound is real and large.

### 2. But it does NOT explain the negative — it is the wrong sign

The instruction model is *positively*, weakly correlated with lag
(r = 0.048 log / 0.090 linear), so partialling time out makes the negative
**slightly worse**, never better:

| fit | ABCD_rew_instr beta |
|---|---:|
| alone | -0.0498 |
| + log lag | -0.0568 |
| + linear lag | -0.0629 |
| + block order gap | -0.0616 |
| + ABCD_rew + log lag | -0.0568 |

Same for the execution model (-0.0071 → -0.0131). **Time hypothesis rejected.**
Mean lag at the 10 model-forced-identical cells (533 s) is indistinguishable
from all other cells (547 s), and within those 10 cells the contribution is
unrelated to lag (the two longest-lag pairs contribute ~0/positive; the
shortest-lag pair E1 contributes strongly negative).

### 3. The actual mechanism: leverage on the low-dissimilarity cells

The beta is additive, `beta = sum_c z(model_c) * z(data_c) / 90`. Attribution
by model level:

| model dissimilarity | n cells | mean z(neural) | contribution to beta |
|---:|---:|---:|---:|
| 0.00 (same task, forced identical) | 10 | **+0.130** | **-0.0316** |
| 0.25 | 8 | **+0.178** | **-0.0224** |
| 0.75 | 40 | -0.091 | -0.0048 |
| 1.00 | 32 | +0.029 | +0.0091 |

64% of the negative comes from the 10 forced-zero cells and a further 45% from
the 8 cells at 0.25 — i.e. **the cells the model calls most similar are the
cells the data call most dissimilar**. Because the model is skewed (18 of 90
cells at the low end), z(model) there is -2.19, so those few cells carry
enormous leverage.

The same inversion holds for the execution model (its 4 cells at d=0.25 have
mean z(neural) = +0.132, contributing -0.0220) — it just has 4 such cells
instead of 18, so its total stays near zero. **This predicts the across-subject
negative shift scales with how many low-dissimilarity cells a model has**,
which is consistent with the audit: memory-within (many) is the most negative
model, plan-within (few) is near zero.

### 4. It IS the forward/backward comparisons — confirmed structurally

The 10 model-forced-identical cells are **exactly** the 10 same-task
forward/backward pairs (verified as an identical set, not merely overlapping).
No two different tasks share an instructed reward sequence, so dissimilarity =
0 is enforced *only* on direction-crossing comparisons. All the model's
low-end leverage therefore sits on the forward/backward contrast.

Premise check (SK): A1_forw and A1_backw really are instructed with the same
reward sequence — but the screen additionally shows a "please backwards" text,
or nothing, implying forwards. So the instruction screens are **not** identical
across directions, and `ABCD_rew_instr` is mis-specified exactly where it has
the most leverage.

### 5. A forward/backward co-regressor does NOT fix it

| fit | ABCD_rew_instr beta |
|---|---:|
| alone | -0.0498 |
| + direction_diff (binary F/B crossing) | **-0.0578** |
| + direction_diff + log lag | -0.0624 |
| + different_task (binary) | -0.0356 |

`direction_diff` is itself negative alone (-0.0188) and correlates -0.245 with
the instruction model, so partialling it out pushes the instruction beta
*further* negative. A single additive direction term cannot work, because the
direction effect is not additive — it reverses sign with task identity:

| cells | n | mean z(neural) |
|---|---:|---:|
| B-F, same task (the forced-zero cells) | 10 | **+0.130** |
| B-F, cross task | 40 | -0.054 |
| F-F, cross task | 20 | +0.120 |
| B-B, cross task | 20 | -0.078 |

### 6. Where the negative actually lives — model-matched subset refits

Refitting the instruction model on cell subsets (all 126,321 searchlights):

| subset | n cells | mean beta |
|---|---:|---:|
| all cells | 90 | -0.0498 |
| drop the 10 forced-identical cells | 80 | -0.0237 |
| **cross-task, same-direction** | 40 | **+0.0068** |
| **cross-task, different-direction** | 40 | **-0.0554** |
| different-direction only (B-F) | 50 | -0.0893 |

The last two rows are the clean test: identical model-value distributions, no
forced-zero cells in either, differing *only* in whether the pair crosses the
forward/backward boundary. **The negative is entirely on the crossing side;
within same-direction cells the model is flat (~0), not negative.**

So the answer is: not time, and not a removable additive direction confound —
the reward-sequence model simply fails to predict neural similarity whenever a
pair crosses the forward/backward boundary, and the instruction model is built
so that its most influential cells are all such crossings.

### 7. No control regressor can soak up the negative — and here is why

Tested ladder (sub-02, 126,321 searchlights), all as `instr + control`:

| control | cols | mean instr beta |
|---|---:|---:|
| none | 0 | -0.0498 |
| n_backward (0/1/2) | 1 | **-0.0498 (exactly unchanged)** |
| direction_diff (binary crossing) | 1 | -0.0578 |
| direction levels (F-F / B-F / B-B) | 2 | -0.0578 |
| different_task | 1 | -0.0356 |
| condition identity (20 per-condition dummies) | 20 | -0.0704 |
| condition identity + different_task | 21 | -0.1076 |
| condition identity + different_task + log lag | 22 | -0.1213 |

**Every control leaves it unchanged or makes it worse.** The structural reason
is a dilemma, visible in the design correlations:

- `corr(instr, n_backward) = +0.000000` — exactly orthogonal by design (the
  same-task cells sit exactly at the mean of n_backward, and among cross-task
  cells direction is independent of reward overlap). A regressor orthogonal to
  the model cannot move its coefficient at all, and this one moves it by
  literally zero.
- `corr(instr, direction_diff) = -0.245`, and direction_diff is itself negative
  (-0.0188 alone), so partialling it out pushes instr *further* negative.
- `corr(instr, different_task) = +0.774` — this is the only control that
  shrinks the negative, and it does so precisely because it **is** the
  instruction model's own main contrast (same task vs not). Absorbing it
  deletes the hypothesis rather than controlling a confound.

So: anything correlated enough with the instruction model to move its beta *is*
the hypothesis; anything that is a genuine nuisance is uncorrelated and does
nothing. Controls that remove the positively-contributing cells (the d = 1.00
level, +0.0091) make the negative worse, which is exactly what the
condition-identity rows show.

### 8. Respecifying the model to match the screen also does not rescue it

Since the instruction screen carries a direction cue, the instructed stimulus
is 4 reward locations *plus* that cue:
`d = (1-alpha) * reward_hamming + alpha * direction_mismatch`.

| alpha (direction cue weight) | d for same-task F/B | mean beta |
|---:|---:|---:|
| 0.00 (current model) | 0.00 | -0.0498 |
| 0.20 (one of five screen features) | 0.20 | -0.0582 |
| 0.50 | 0.50 | -0.0488 |
| 1.00 (pure direction model) | 1.00 | -0.0188 |

Negative at every weight, including a pure direction model. The reward-sequence
content never earns a positive coefficient in these data.

### 9. Conclusion: this is a cell-selection question, not a regressor question

The only unconfounded test of "do overlapping instructed reward sequences
produce similar instruction-period patterns" is the **cross-task,
same-direction** cells (40 cells, no forced-zero cells, no direction crossing):
**beta = +0.0068**, i.e. null, but not dragged negative.

Restricting the fitted cells is implementable in the existing framework (the
RSA runner already selects cell subsets via `data_rdm_scope`). It is legitimate
here *because the direction cue is a known stimulus property, established from
the screen content independently of any fit* — not because it improves the
number. Choosing among the controls above by which one flips the sign would be
post-hoc and is explicitly rejected.

### 10. What the same-direction mask actually leaves (sub-02)

Figure: `sub-02_same_direction_mask.png` in the output dir.

The mask keeps 40 of 90 cells (20 F-F + 20 B-B), all cross-task. But:

- **The 40 cells are only 20 unique comparisons.** Within a half each task has
  exactly one forward and one backward condition, and the instruction model
  gives the backward condition the forward vector, so
  `d(X_F, Y_F) == d(X_B, Y_B)` identically. Each task pair is therefore
  measured twice with the *same* model value. The model has 20 independent
  predictions, not 40. This is structural and holds for every subject.
- **The d = 0 level disappears entirely** (all 10 zero cells were B-F). Model
  levels retained: 0.25 / 0.75 / 1.00; model sd falls from 0.325 to 0.218 (67%).
- **The "similar" end rests on 2 task pairs** — C1-E1 and C2-E2, the only
  comparisons at d = 0.25 (4 cells). This part is subject-specific: it depends
  on the particular reward sequences.

**The duplication is a free split-half reliability check, and in this subject
it fails:**

| comparison | r |
|---|---:|
| task-pair level, F-F estimate vs B-B estimate (20 pairs) | **+0.020** |
| searchlight beta maps, F-F-only vs B-B-only fit | **-0.080** |

with beta = +0.0130 (F-F only) and +0.0011 (B-B only). So under the mask there
is little reliable neural structure for the model to explain in sub-02 — the
two measurements of the same prediction do not agree.

Caveat in the other direction: single-subject whole-brain RSA reliability is
routinely near zero and effects normally emerge only at group level, so this is
a warning rather than proof the mask is useless. It does mean the mask is not
rescuing anything here, and that this reliability check should be run per
subject on the event-locked caches before the mask is adopted.

### 11. Searchlight NIfTIs exported

`.../instruction_instr_negative_time_audit_2026-09-14/sub-02_searchlight_maps/`

| map | cells | mean beta |
|---|---:|---:|
| `ABCD_rew_instr_all90` | 90 | -0.0498 |
| `ABCD_rew_instr_same-direction` | 40 | +0.0068 |
| `ABCD_rew_instr_forward-only` | 20 | +0.0130 |
| `ABCD_rew_instr_backward-only` | 20 | +0.0011 |
| `ABCD_rew_instr_direction-crossing` | 50 | -0.0893 |

Each as `_beta` and `_t_val`, written on the subject's functional grid using the
pipeline's own saved map as geometry reference (shape 108x108x64, affine
verified identical), naming per `mc.analyse.handle_MRI_files`.

Two caveats attached to these files:

- **Subject functional space, not MNI.** They are not comparable to the group
  maps without applying the subject's registration.
- **The `_t_val` maps are first-level parametric t and must not be used for
  inference.** RDM cells are not independent observations, and under the
  same-direction mask they are doubly non-independent because each task pair
  appears twice (F-F and B-B) with an identical model value, so the nominal
  df = 38 is roughly double the truth. They are included for thresholding and
  visual inspection only; group inference stays a random-effects test over
  subject beta maps.

### What this does not yet settle

- Single subject, proxy GLM (see scope caveat). Must be rerun on the
  event-locked caches before any of this is asserted for the group.
- The F-F (+0.120) vs B-B (-0.078) asymmetry among cross-task cells is
  unexplained and may just be noise at n=20 cells in one subject.
- Whether an *interaction*-style regressor (direction crossing x same task),
  or simply excluding the direction-crossing cells, gives a defensible
  instruction model — not yet tested. Excluding them is a scope change to the
  hypothesis (it removes the only cells where instruction and execution models
  diverge), so this needs a decision rather than a fit.

## 2026-09-14 (b) — Pad to 100 ms; subfield moderator; He et al. methods obtained

### 1. `PAD_S` 0.25 → 0.1 s

**0.1 s is He et al. 2026's own figure** — their Methods (now in hand) discard
"any ripple occurring within 100 ms of IED" and likewise within 100 ms of a
pathological HFO. Recovers ~56% more artifact-free seconds than the 1 s pad.

Chosen as a **lower bound**, not as optimal. Because every event carries
`dist_to_artifact_s` and the bundle carries the unpadded crossings, any
stricter pad can be re-imposed downstream for free — so the cheap mistake is
padding too little here, not too much. Supported by the 2026-09-14 measurement:
ripple properties are flat across distance-to-artifact bins (peak frequency
98.1–98.5 Hz, duration 65.8–67.1 ms from 0.25 s out to ∞).

### 2. Correction — the `swr_v1` bundle CANNOT be re-padded

Stated carelessly yesterday. To be exact about what was measured where:

- the **pad sweep** and the **event-property measurement** were computed on the
  laptop from `continuous.npy` (present locally for 21 sessions), NOT from a
  bundle;
- the **re-padding demonstration** used the *new* s29 `swr_v2` proof-run bundle,
  built by the new code.

The downloaded `swr_v1` bundle (2026-09-08, 61 sessions, 212 derivations) has
no `dist_to_artifact_s` and no `artifact_intervals`, and was built at a 1 s pad.
Events outside that pad were never detected and do not exist in it at any price.
Re-padding only ever goes **stricter**, and only for bundles built by the new
code. This is precisely why the new pad is set at the lower bound.

### 3. `scripts/swr_subfield_moderator.py` — runs on the EXISTING bundle

Is the F5 first-D ripple increase stronger on CA-weighted derivations, as
Sakon et al. report for CA1 vs DG? Subfield is a pure MNI lookup and `pairs`
already carries coordinates, so **no cluster run and no LFP are needed**.

Result (212 derivations, 61 sessions, 42 subjects):

| condition | moderator `delta ~ ca_dg_index + (1\|subject)` |
|---|---|
| first_D | β = −0.078, p = 0.285 |
| later_D | β = +0.008, p = 0.711 |

No CA gradient, and the first-D point estimate runs *opposite* to Sakon.

**But the sample cannot test that contrast.** Only 2 of 199 derivations have
P(DG) > P(CA); the lower half of the index still has P(CA) > P(DG) in 95 of 97.
That is `select_hpc_contacts` **working as intended** — it takes the deepest
hippocampal contact per probe, which mostly sees cornu ammonis — so the montage
is already concentrated where Sakon put the effect, which helps detect F5 at
all. The narrow cost is that the CA-vs-DG contrast has no DG arm: a null on the
moderator is uninformative about the gradient and says nothing about F5 itself.
Testing it properly needs the runner-up contacts per probe (56 of 128 clear the
25% threshold, `methods.md` §4.2), which this montage deliberately excludes.

The script prints this as a precondition rather than leaving it to be inferred.
Its Eq. 2 figure is a derivation-level sanity check, not the canonical F5 —
`fit_eq2` expects trial-level rows and drops subjects below its 10-row floor
(13 of 42 survive). The moderator uses all derivations and is the actual test.

### 4. He et al. 2026 Methods — what they actually did

Obtained, resolving what was previously flagged as unverifiable:

- **Hippocampal contacts are referenced against a nearby WHITE MATTER contact
  on the same probe**, not against an adjacent contact. This project rejected
  white-matter referencing for hippocampus deliberately (`methods.md` §4.2:
  it doubles inter-contact distance and enlarges the lead field). A real
  divergence, and a pre-existing decision — not changed here.
- All other contacts: adjacent-bipolar on the same probe. **Confirms the
  cortical montage choice**: every in-ROI contact, not one per probe.
- Ripple band **70–180 Hz**, 4 SD threshold extended to 2 SD, 20–200 ms,
  30 ms merge. This pipeline follows Chen (80–120 Hz, dual 1.5/3.0 SD,
  38–500 ms) — a deliberate, documented difference.
- HFB **80–140 Hz**, *overlapping* 10 Hz sub-bands at 5 Hz steps, each divided
  by its own channel-and-band mean, summed, averaged, rescaled by the grand
  mean, then 10·log10 to dB. Ours: 70–150 Hz, contiguous 10 Hz sub-bands,
  log-then-z-score per band. Both are standard normalisations; ours follows
  `criterion_broadband_power` already in this codebase.
- They recorded at 50 Hz line frequency, so **100 Hz sits inside their
  80–140 Hz HFB band and they simply notch through it.** Our notch-aware
  sub-band dropping has no counterpart in their pipeline.
- Also: contacts >3 mm from the cortical ribbon excluded; a common-average
  control removes ripples coincident across contacts.


## 2026-09-14 (b) — Exploratory mistake sweep (all null), and the final figures rebuilt

### Exploratory: do ripples predict mistakes? No.

`scripts/swr_explore_ripples_and_mistakes.py`, results in
`swr/explore_ripples_mistakes_2026-09-14/`. **EXPLORATORY** — .png only, nothing
confirmatory. Window +0.35 to +0.75 s (where the stillness-matched cluster
peaked), two measures: `delta` (window minus the same trial's baseline) and
`raw` (window alone). Paired t-tests at session level: within each session the
mean over grids WITH the outcome against the mean over grids WITHOUT.

| question | measure | horizon | no mistake | mistake | Δ Hz | p_perm | n |
|---|---|---|---|---|---|---|---|
| first D | delta | all later repeats | +0.1009 | +0.1012 | +0.0003 | 0.996 | 44 |
| first D | delta | next 3 repeats | +0.0449 | +0.0847 | +0.0398 | 0.685 | 29 |
| first D | raw | all later repeats | +0.2840 | +0.2427 | −0.0413 | 0.307 | 46 |
| first D | raw | next 3 repeats | +0.2857 | +0.2379 | −0.0478 | 0.351 | 36 |
| all first uncovers | delta | all later | +0.0634 | +0.0717 | +0.0083 | 0.796 | 53 |
| all first uncovers | delta | next 3 | +0.0693 | +0.0677 | −0.0015 | 0.968 | 51 |
| all first uncovers | raw | all later | +0.2575 | +0.2464 | −0.0112 | 0.607 | 53 |
| all first uncovers | raw | next 3 | +0.2496 | +0.2511 | +0.0015 | 0.958 | 51 |

Nothing, in either measure or either horizon. The `raw` first-D tests lean the
predicted way (more ripples → fewer later D errors, ≈ −0.04 Hz) but at p ≈ 0.3.

### Descriptive: mistakes have strong structure

27,538 errors, 100% with a location.

- **Immediate repetition is AVOIDED.** In the transition matrix of consecutive
  errors while seeking the same reward, the diagonal is **4.5%** against ~12.5%
  chance. Having just tried a tile, subjects do not retry it.
- **The heavy off-diagonal cells are grid-adjacent** (1→4, 4→7, 2→5): the
  search walks to neighbouring tiles.
- **But errors do recur across repeats:** **23.6%** are eventually repeated at
  the same reward *and* the same location — roughly double chance — evenly
  across A–D (21.9, 22.5, 25.3, 25.1%).
- **Ripples at an error do not prevent repeating it**, and the non-significant
  trend runs the wrong way: repeated errors had *higher* rates
  (delta +0.0450 vs +0.0022, p = 0.151; raw +0.2075 vs +0.1871, p = 0.288).

### Final figures rebuilt

The 3.5 cm eight-panel grid and the raster are retired — the raster showed
nothing at 0.2 Hz over 4 s (under one ripple per row). Replaced by two figures
in the house 16 cm × 4 cm format:

- **`ripple_main_figure`** — three rows through `rip.plot_rows`: positive
  feedback and negative feedback against their own baselines (the
  `feedback_stage` panel), then the same headline cell against its
  stillness-matched control press. One figure, both references.
- **`ripple_methods_figure`** — the controls at the same scale (2 × 2, 16 cm).

**Display smoothing added.** `rip.triangle_smooth` implements Sakon & Kahana's
5-bin triangle, and `plot_rows(smooth_bins=…)` applies it to the LEFT panel
only. The middle and right panels, and every statistic, still read the
unsmoothed profiles — this is why their PVTHs look smooth and ours did not.

**Two `plot_rows` parameters added.** `colours=` overrides `condition_colour`
for labels the project's valence/stage naming cannot colour (the control press
is neither, and was falling through to a tab10 orange — the hue reserved for
state A); `legend_cm=` sizes the legend strip, which a three-row figure with
eight entries overflowed into the bottom row.

On the current bundle the `feedback_stage` rows give: correct/first uncovers
increase +0.25…+0.65 s, p = 0.005; error/while learning decrease +1.15…+1.45 s,
p = 0.046; the other four none. Row 3 (matched) gives +0.35…+0.75 s, p = 0.004.


## 2026-09-14 — Hippocampal subfield, a matched temporal control, and post-hoc re-padding

Follow-up to 2026-09-13 (f), on the same branch `swr-mpfc-hfb`.

### 1. The pad now has evidence, not just a rationale

The 0.25 s choice was a reasoned prior, **not a measurement** — stated as such
at the time. The measurement has now been run: 15,036 accepted events on 29
derivations across 11 sessions, comparing events a 1 s pad would have kept
(`dist_to_artifact_s >= 1`) against those the 0.25 s pad newly admits.

| | kept by 1 s pad | newly admitted | diff |
|---|---|---|---|
| peak frequency (Hz) | 98.12 | 98.38 | +0.3% |
| duration (ms) | 66.3 | 67.0 | +1.1% |
| RMS peak (z) | 3.847 | 3.918 | +1.8% |
| amplitude (µV) | 1.553 | 1.680 | **+8.2%** |

29.5% of events are newly admitted. By distance bin the profile is **flat and
monotonic, with no discontinuity near the artifact**: peak frequency
98.1–98.5 Hz and duration 65.8–67.1 ms across every bin from 0.25 s to ∞.

**Caveats, both real.** The +8.2% amplitude difference is a genuine monotonic
gradient (rms z 3.92 nearest → 3.80 farthest); events near crossings are
slightly larger. That is consistent with contamination *and* with epileptogenic
tissue genuinely producing larger ripples, and this measurement cannot separate
the two. Separately, `spectral_passed_strict` reads 1.000 in every bin only
because the comparison is over events that already passed — it carries no
information here and should not be quoted as if it did.

### 2. Any pad can now be re-imposed on the laptop

Previously impossible, and **`dist_to_artifact_s` was being silently dropped by
the bundle's column filter** — so the per-event distance stored on 2026-09-13
would never have reached the laptop at all.

- detection writes `artifact_intervals.csv`, the **unpadded** criterion union
- `swr_artifact.clean_intervals_at_pad` rebuilds the exposure denominator at
  any pad from it. Verified EXACT against the cluster-side mask at 0.1 / 0.25 /
  0.5 / 1.0 s (dilation distributes over union, so this is not an approximation)
- the bundle carries both, plus `dist_to_artifact_s` per event

⚠ Both halves are needed. Filtering events without shrinking the denominator
inflates the rate; the bundle `meta` says so.

### 3. Hippocampal subfield — as probabilities, not a label

Every contact and hippocampal derivation now carries Jülich `p_CA`, `p_DG`,
`p_SUB` and `ca_dg_index` = (P(CA) − P(DG)) / (P(CA) + P(DG)).

**Three limits, all load-bearing:**

1. **This is not CA1 vs CA3.** Jülich's "cornu ammonis" is one volume covering
   all CA fields. Nothing in this project's atlas stack resolves CA1 from CA3.
2. **A hard label would be a volume artefact** — max-prob gives 88 CA / 5 DG /
   7 SUB of 103 derivations. Probabilities overlap heavily on the same contact
   (median P(CA) 78, P(DG) 38) because the subfields interdigitate below this
   resolution. Use `ca_dg_index` as a continuous moderator only.
3. **ASHS was the right tool and is not available** — populated for 3 of 103
   selected contacts. Obtaining ASHS segmentations is the real fix.

Also surfaced: **12 of 103 selected hippocampal contacts are called *amygdala*
by Jülich** while the Harvard–Oxford rule that selected them says hippocampus.
Worth a sensitivity analysis (`bRAMY2-bRAMY3` in s29 is one such derivation).

### 4. A matched temporal control (SK's suggestion, and it is the better one)

A temporal depth electrode is inserted laterally, so its outer contacts sit in
lateral temporal cortex **on the same shaft as the hippocampal contact** — same
amplifier, reference chain, noise environment and trajectory.

Measured: **353 TemporalLateral derivations over 31 sessions, 158 of them on a
probe that also carries the selected hippocampal contact**, plus 54 Auditory
(Heschl / planum). Flagged per derivation as `same_probe_as_hpc`.

This is strictly better than Visual, which differs from mPFC in every one of
those respects at once. Visual is kept anyway for like-for-like comparison
with He et al.

Labels come from the Harvard–Oxford **cortical** atlas via the new
`contact_anatomy.add_swr_roi`, writing a new `roi_swr` column. It only fills in
contacts the shared ROI ladder had already returned `leftover` for, so
`anatomy_atlas.assign_atlas_roi` — which is byte-for-byte shared with the cell
pipeline — is untouched.

`CORTICAL_ROIS` is now `mPFC, mOFC, TemporalLateral, Auditory, Visual`. Insula
(172 derivations) and PCC (27) were dropped from the default: neither target
nor matched control, and every extra derivation is extra channels on the one
I/O-bound stage.

### 5. Two bugs found by the proof-run

- **Ripple detection was running on the cortical derivations.** They share the
  pairs table so extraction reads every channel in one pass, but Chen's band,
  thresholds and duration gate are hippocampal. This would have emitted a table
  that looks exactly like a ripple table and means nothing. Detection now
  filters on `role == 'ripple'` and says how many rows it skipped.
- **Stale `bipolar_pairs_*.csv` files** from before the cortical montage (s13,
  s24 locally) carry no `role` column. They would extract hippocampus-only and
  look complete. `swr_make_conditions.py` now detects and reports them.

### 6. Montage figure

`ripple_figures.montage_figure`, drawn automatically by `swr_build_contacts.py`
to `group/swr/figures/montage_coverage.pdf/.jpg`. Derivation midpoints — where
a bipolar derivation is actually sensitive — coloured by ROI, with same-shaft
controls ringed in black. Targets take the project's Showgirl2 colours; controls
are grey, so target and control are distinguishable without reading the key.

### 7. Proof-run on the laptop (s29, full chain)

`build_contacts → extract_continuous → detect_session → extract_hfb`, all green:

- 15 derivations: 3 hippocampal, 3 mPFC, 5 mOFC, 3 TemporalLateral, 1 Visual
- 30 unique channels in one raw pass; residual line noise 0.00 at all harmonics
- detection: 3 hippocampal rows, 12 cortical correctly skipped; rate 0.178 Hz
  (Chen 0.17–0.24); 22% of events within 1 s of a crossing
- **120 Hz was notched on 8 of 15 derivations**, so 8 dropped to 6 HFB
  sub-bands. Without the notch-aware band, more than half this session's HFB
  would have been measured through a notch
- **positive control: all 3 hippocampal derivations positive** (+0.099, +0.120,
  +0.085 z peri-ripple vs flanks)
- `hfb.npz` 43 MB for 15 derivations × 0.87 h ≈ 3.3 MB/derivation/hour,
  projecting to ~1.9 GB for the cohort


## 2026-09-14 — Figures split, and two framing corrections

Follow-up to (e), after SK's questions.

**Two figures instead of one.** `ripple_results_figure` shows the data: a
Sakon-style raster (one row per trial × derivation, one tick per ripple), the
peri-event rate for uncovering and its matched control, the sliding *t* curve
with the surviving cluster, and the baseline-vs-test window means.
`ripple_methods_figure` shows the controls: the stillness imbalance that
motivates the design, the matching balance, control neutrality across the four
arrow keys, and the matched-vs-unmatched selection check. All panels measured
to 3.51 × 3.51 cm.

**Correction 1 — "raw / no baseline" was the wrong name.** The matched control
*is* a baseline; it is a different EVENT rather than a different TIME. Renamed
throughout to "own-baseline reference" and "matched-event reference", with a
table in the methods stating what each does and does not control for.

**Correction 2 — the primary/secondary ordering was backwards.** Sakon &
Kahana's Eq. 2 (own-baseline) is the field's statistic and is now the headline:
+0.059 Hz [+0.011, +0.106], t(60) = +2.46, p = 0.018, cluster +0.25…+0.75 s,
p = 0.008. The matched-event contrast (+0.035 Hz, cluster +0.35…+0.75 s,
p = 0.004) is now presented as the control analysis showing the effect is not
stillness. Both readings were always reported; only which leads has changed.

**New observation from the window table.** The effect is a CROSSOVER, not an
offset: at the −1.6…−1.1 s baseline the uncovering sits *below* its matched
control (0.180 vs 0.204 Hz) and by 0–0.5 s it sits above it (0.225 vs
0.190 Hz). The matched-event difference crosses zero about 1 s before the press
and is already positive in the pre-event window (+0.039 Hz, p = 0.074;
own-baseline +0.063 Hz, p = 0.027). A rise beginning before the press resembles
Sakon's PRE effect, though the pre-event portion does not survive cluster
correction alone. Worth pursuing.

**On whether the controls are too strict.** Sakon & Kahana never match on a
behavioural covariate: their PRE and BASE windows both sit in the same silent
pre-vocalisation stretch, so motor state is constant across the comparison by
construction. Our window is post-press and therefore downstream of what the
participant does next, which is exactly what differs between conditions — so
the control is warranted here and would be redundant there. Their 1st-vs-≥2nd
recall comparison probably does carry an analogous timing confound that they
address only with the 2 s exclusion.

## 2026-09-13 (f) — Artifact pad 1.0 s → 0.25 s, and a cortical (HFB) branch

Two changes on branch `swr-mpfc-hfb`, both additive. Motivated by He et al.
2026 (*Nat Neurosci* 29:1711), who show hippocampal ripples update mPFC
representations — the "how is the action plan loaded" step this project has not
yet tested. Ideas the new data makes testable are listed in
`data/final_results/ripple_analysis/POTENTIAL_IDEAS.md`; none has been run.

### 1. Artifact pad (`mc/analyse/swr_artifact.py`)

`PAD_S` 1.0 → **0.25 s**. Measured across 27 derivations in 11 sessions, by
re-dilating the *same* per-criterion masks:

| pad | mean clean fraction | derivations >2/3 contaminated | exposure vs 1.0 s |
|---|---|---|---|
| 1.0 s (old) | 0.569 | 3/27 | — |
| 0.5 s | 0.723 | 0 | +27% |
| **0.25 s (new)** | **0.819** | 0 | **+44%** |
| 0.1 s | 0.885 | 0 | +56% |

The five criteria flag ~2.6% of samples between them but
`frac_combined_after_pad` was 47.6% across the cluster bundle — the pad, not
the criteria, was doing essentially all of the rejection.

The binding reason is cross-regional: a peri-ripple window needs ±750 ms clean
in the hippocampal **and** the cortical derivation. Measured on the bundle, only
56.7% of accepted ripples had a clean ±750 ms in their own derivation, which
against an independent cortical mask leaves ~32% usable.

Detector metrics are stable across the change (s38): rate 0.222 → 0.213 Hz
(Chen 0.17–0.24), spectral rejection 30.3% → 30.1%. Only exposure moves.

Three guards, because this is a parameter change made after seeing the data:

- **`EXCLUSION_PAD_S = 1.0`, fixed.** Contamination is judged at the old pad so
  the set of included derivations cannot move with `PAD_S`. Otherwise "more
  exposure" and "dirtier contacts added" are confounded.
- **`dist_to_artifact_s` is stored per event** — distance to the nearest
  *unpadded* criterion crossing. Any larger pad can be re-imposed by filtering
  that column, so a 0.25 s run reduces exactly to the 1 s result with no
  re-detection. The pad is no longer a decision that must precede the analysis.
- **Rejected metric, recorded so it is not retried:** "fraction of accepted
  events within 100 ms of a crossing" is identically **zero** for any pad above
  100 ms, because detection only ever runs on clean samples. It cannot
  discriminate between pads. The distance distribution can.

`swr_detect_session.py` gained `--pad_s` and `--clean_name` (which extraction to
*read*), so a pad sweep costs no raw I/O. Refactor verified: `criteria_masks` +
`combine_criteria` reproduce the stored pad-1.0 numbers to 5 decimal places.

### 2. Cortical branch — HFB (`mc/analyse/swr_hfb.py`, `scripts/swr_extract_hfb.py`)

`bipolar_pairs_{XX}.csv` now holds both montages, keyed by `role`:

- `ripple` — hippocampal, one derivation per probe. **Unchanged**, verified
  identical to the committed `bipolar_pairs_38.csv`.
- `hfb` — cortical (mPFC, mOFC, Visual, Insula, PCC), non-overlapping adjacent
  pairs anchored on each in-ROI contact. Verified invariant across 6 sessions:
  **no contact appears in two derivations.** One-per-probe was not used here —
  He et al. ran on 47 dmPFC contacts, which that rule cannot reach — but the
  independence it protects is kept.

Cortical pairs ride in the same table so stage 2 reads them in the **same
raw-file pass**; extraction is the only I/O-bound stage and running it twice
would cost hours for nothing. Resources bumped 64G/4h → 96G/8h.

`swr_extract_hfb.py` is a **sibling of detection**, not a successor — both read
`continuous.npy`. Per derivation it writes HFB (70–150 Hz, per-sub-band log +
z-score, averaged) plus `ripple`/`theta`/`beta` envelopes and `theta_phase`, all
continuous at 100 Hz, float16.

- **Notch-aware.** 120 Hz sits inside any HFB band and this pipeline's notch is
  adaptive *per derivation*, so a fixed band would mean different things on
  different contacts of the same session. Sub-bands on a notched harmonic are
  dropped and `n_sub_bands` is carried per derivation.
- **z-scored on artifact-free samples only**, else the flagged samples — by
  construction the largest excursions — set the SD.
- **Nothing is epoched.** Windows, baselines and phase splits stay free choices
  on the laptop; that was the point of the stage.
- **Positive control built in:** peri-ripple HFB at hippocampal derivations is
  computed and printed. On s38: **+0.115 and +0.147** z against flanks. If this
  is not positive the clocks disagree and no cortical number means anything.

Coverage (local 32-session subset; the cluster has ~46): mPFC 80 contacts / 23
sessions / 17 subjects, mOFC 225 / 26 / 18, Visual 179 / 19 / 13. Every session
with an mPFC contact also has a hippocampal one. He et al. worked with 47 dmPFC
+ 12 vmPFC. ⚠ s26/27/28 are one subject contributing 10 mPFC contacts each —
subject-level inference stays mandatory.

### 3. Bundle — still the final stage

`export_bundle(with_hfb=True)` copies the HFB stores to `bundle/hfb/s{NN}_hfb.npz`,
one per session, read back with `swr_bundle.load_hfb` (casts to float32; float16
accumulates error over the long sums that averaging epochs is). Kept beside the
pickle, not inside it, so `swr_bundle.pkl` stays a few MB and loadable alone;
the HFB half is ~1–2 GB. `--hfb_arrays` narrows the transfer.

**Bug found and fixed while testing:** `export_bundle` read `pairs.csv` from
`LFP-clean/{analysis_name}`, so for any run using `--clean_name` the pairs table
came back **empty** — and the `FileNotFoundError` then skipped that session's
behaviour and uncover tables too, silently. It now reads `clean_name` from the
detection settings. This would have hit every pad-sweep bundle.


## 2026-09-13 (e) — Final consolidated ripple analysis, figure and manuscript drafts

The analyses SK settled on are now in one runnable script rather than spread
across five exploratory ones.

**`scripts/swr_final_ripple_analysis.py`** — self-contained (imports only
`mc.analyse.ripples`, `swr_io`, `swr_behaviour`), one command, no hard-coded
bundle. It builds `swr/press_categories.csv` from the raw 25 ms button series
if that cache is missing or older than the bundle, so pointing `--bundle` at
new data is the only change needed when more sessions arrive. The cache build
is the only slow step (~15 min, I/O bound); everything after it is minutes.

It runs, in order: control validity (four arrow keys compared), matching
balance, selection check (matched vs unmatched targets), per-cell contrasts
against stillness-matched controls in two readings (raw = primary, own-baseline
= secondary), and the five between-cell contrasts including the valence × stage
interaction.

**Outputs** in `swr/ripple_final_<date>/`:
- `ripple_final_figure.png` / `.pdf` — 8 panels, each measured to
  3.51 × 3.51 cm (the layout iterates until the axes hit the requested 3.5 cm,
  because `tight_layout` otherwise gives panels whatever the text leaves over).
  Figure 20.1 × 10.7 cm overall. Type steps to 8/7.5/7 pt: the house 9 pt fills
  a third of a 3.5 cm panel.
- `ripple_statistics.csv` — 72 rows, one per test: analysis, contrast, reading,
  window, n_units, mean, SEM, 95% CI, t, df, p, p_perm, Cohen's d, n_events.
- `ripple_statistics.json` — the same plus every setting, the descriptives and
  the full time courses, for provenance.

**Manuscript drafts**, in `data/final_results/ripple_analysis/manuscript/`:
- `statistics_methods_DRAFT.md`
- `statistics_results_DRAFT.md`

Both quote only numbers present in the statistics files. The results draft leads
with the raw reading, states the own-baseline reading as secondary and says why
it is inflated, reports the interaction as underpowered rather than null with
its CI, and carries four caveats into the discussion (exploratory status, button
stillness only, the control not being a blank, and the 57% matched subset).

All values reproduce the exploratory runs of entries (b)–(d) exactly.

## 2026-09-13 (d) — CORRECTION: with 1:1 stillness matching the first-traversal effect SURVIVES

This supersedes the conclusion of entries (b) and (c) for one cell. SK proposed
matching each uncover press against a press of a category carrying no
hypothesis — an arrow key — with the same following stillness, rather than
against a pooled or coarsely binned control. Done properly, the effect that
binning had washed out comes back.

New scripts: `scripts/swr_build_press_categories.py` (one slow read of the
25 ms button series, cached as `swr/press_categories.csv`) and
`scripts/swr_matched_control_presses.py`. Results in
`.../swr/matched_control_presses_2026-09-13/`.

**Press categories.** 248,737 presses, 61 sessions: Return 97,661;
RightArrow 40,554; LeftArrow 37,751; DownArrow 37,173; UpArrow 35,598.
97,660 of 97,661 uncover presses carry a valence/stage label.

**Matching.** Per session, 1:1 nearest neighbour on `still_next_s`, WITHOUT
replacement, caliper max(0.15 s, 10%), targets served in seeded random order
(seed 42), both sides de-duplicated at 2 s before matching. Achieved balance is
essentially exact — target and control median stillness agree to 3 decimals in
every category (e.g. 0.500 vs 0.500 s; 1.175 vs 1.175 s).

**The control is neutral.** Rate after each arrow key, gaps 1–2.5 s:
Left 0.1879, Right 0.2069, Up 0.2041, Down 0.1957 Hz — all within 0.02 Hz with
overlapping SEMs, so "an arrow key" is one category and direction need not
enter the matching.

### Results, 0–0.5 s, uncover minus its matched arrow key

| target | matched | vs own baseline | raw (no baseline) | raw cluster |
|---|---|---|---|---|
| **correct, first uncovers** | 3,074 (57%) | **+0.0586, p = 0.018** | **+0.0349, p = 0.055** | **+0.35…+0.75 s, p = 0.004** |
| correct, while learning | 4,061 (90%) | +0.0409, p = 0.024 | +0.0198, p = 0.090 | none |
| correct, once known | 10,412 (80%) | +0.0343, p = 0.118 | +0.0062, p = 0.611 | none |
| error, first uncovers | 5,671 (94%) | +0.0001, p = 0.996 | +0.0054, p = 0.617 | none |
| error, later | 3,891 (91%) | +0.0231, p = 0.287 | +0.0026, p = 0.875 | none |
| first D (F5) | 1,085 (76%) | +0.0302, p = 0.503 | +0.0036, p = 0.918 | none |

n = 61 sessions for all but error/later (57) and first D (54).

### What this changes

1. **The first-traversal correct uncovering is NOT explained by stillness.**
   Against a press matched on stillness to three decimals, with no baseline
   window anywhere in the statistic, it carries +0.035 Hz more and produces a
   cluster surviving correction over all window positions (p = 0.004; still
   p < 0.05 after Bonferroni over the six targets). Entries (b) and (c)
   concluded otherwise from coarse binning, which compared a target at 1.6 s
   with a control at 2.4 s inside the same bin and lost most of the sample
   (n fell to 32); that was a power and matching failure, not an absence.

2. **The stage gradient is real in the matched data.** Raw: first uncovers
   +0.035 > while learning +0.020 > once known +0.006, with errors flat at
   every stage (+0.005, +0.003). This is the pattern SK saw in
   `feedback_stage`, now with stillness removed by construction.

3. **F5's D-specific claim does not survive; the general first-traversal
   claim does.** `first D` gives +0.0036 Hz (p = 0.92) while all first-traversal
   correct uncoverings give +0.035 Hz. The effect is about uncovering a reward
   for the first time, not about D. Since F5 was the one pre-declared claim,
   this needs stating plainly in any write-up.

4. **The own-baseline reading is inflated and should not be the headline.**
   It exceeds the raw reading in every one of the six cells (+0.059 vs +0.035,
   +0.041 vs +0.020, +0.034 vs +0.006, …). That is the pre-event stillness
   imbalance predicted in entry (b), Limitation 2, showing up exactly as
   expected: controls sit in already-quiet stretches, so their baselines are
   ripple-richer and their (window − baseline) is depressed. The raw reading is
   the one to quote.

### The unmatched 43% are not a favourable selection

The obvious worry about a 57% match rate is that matching kept the events with
the smaller effect. It did not — it kept the ones with the SMALLER effect in the
opposite sense, i.e. the drop is conservative. Each subset against its own
baseline, 0-0.5 s:

| target | matched | still | unmatched | still | paired Δ |
|---|---|---|---|---|---|
| **correct, first uncovers** | **+0.0446** (n=61) | 1.05 s | **+0.0647** (n=55) | 2.19 s | −0.019, t = −0.82, p = 0.42 |
| correct, while learning | +0.0209 (n=61) | 0.52 s | −0.0235 (n=13) | 3.32 s | +0.051, p = 0.38 |
| correct, once known | +0.0167 (n=61) | 0.50 s | −0.0124 (n=45) | 0.53 s | +0.043, p = 0.075 |
| correct, later | +0.0162 (n=61) | 0.52 s | — | 0.57 s | — |
| error, first uncovers | −0.0108 (n=61) | 0.52 s | — | 0.64 s | — |
| first D (F5) | +0.0385 (n=54) | 1.17 s | — | 3.03 s | — |

The unmatched first-traversal events are the LONG-stillness ones (median 2.19 s
vs 1.05 s) and they carry a BIGGER own-baseline effect (+0.065 vs +0.045), with
no significant difference between subsets (p = 0.42). So matching did not
cherry-pick the events that show the effect; it discarded the ones that show it
most, precisely because their stillness had no arrow-key partner. How much of
that extra +0.02 Hz is signal and how much is their doubled stillness cannot be
separated — which is exactly why they are excluded rather than included.

### The valence x stage interaction does NOT hold under matching

Each cell is already `uncover − its own stillness-matched arrow key`, so
differencing two cells is a stillness-adjusted comparison. Raw reading, 0-0.5 s:

| contrast | n | Δ Hz | t | p_perm | 95% CI | cluster |
|---|---|---|---|---|---|---|
| stage within correct (first − later) | 61 | +0.0205 | +1.07 | 0.291 | [−0.018, +0.059] | none |
| stage within error (first − later) | 57 | +0.0008 | +0.05 | 0.957 | — | none |
| valence at first (correct − error) | 61 | +0.0295 | +1.45 | 0.152 | [−0.011, +0.070] | none |
| valence later (correct − error) | 57 | +0.0118 | +0.77 | 0.449 | — | none |
| **INTERACTION valence × stage** | 57 | **+0.0211** | +0.80 | **0.439** | **[−0.032, +0.074]** | none |

Every contrast points the predicted way — the stage effect lives in the correct
cells (+0.021) and not in the error cells (+0.001), and valence separates more
at first (+0.030) than later (+0.012) — but not one of them reaches
significance, and no cluster survives.

This is the same dissociation the 2026-09-11 entry found and it is now
confirmed under the cleanest control available: **the simple effect at
correct/first-uncovers holds against its matched control; the between-cell
contrasts, including the interaction, do not.** The interaction CI spans
[−0.032, +0.074] Hz, which is consistent with anything from a small negative to
a substantial positive interaction — this is underpowered, not clearly null,
and should be reported as such rather than as an absence.

Arithmetically the reason is plain: a between-cell contrast is the difference
of two control-adjusted quantities, so its variance roughly doubles while its
expected size shrinks relative to the simple effect.

### Limitations

- **57% match rate for correct/first-uncovers.** Long-stillness targets are the
  hardest to match (the arrow-key pool thins out past 2 s), so the matched
  subset skews shorter than the full set (median 1.050 s vs 1.375 s). The
  contrast is stillness-balanced by construction, but it is established on that
  subset, not on all first-traversal uncoverings. See the subset comparison
  above: the omission is conservative.
- Matching without replacement discards targets that find no partner; counts
  are reported per category, never silently dropped.
- `first D` has the lowest match rate (76%) and the smallest n (54 sessions);
  its null is weakly powered and should not be read as evidence against F5, only
  as a failure to confirm it under this control.


## 2026-09-13 (c) — What stillness is; ripples are flat inside a pause; task-free rest is the highest rate in the dataset

SK's four objections to the stillness control, each answered.
`scripts/swr_stillness_anatomy.py`; results in
`data/ephys_humans/derivatives/group/swr/stillness_anatomy_2026-09-13/`.
61 sessions, unit = session, min 8 events, rates are RAW (no baseline
subtraction anywhere in this entry).

### Q1 — what "stillness" actually is, and what it was conflating

`still_next_s` = time from the event to the NEXT key press of any kind
(movement or uncover), from the 25 ms button series. It CLASSIFIES the event;
it is not a comparison with baseline, and "the window is more still than
baseline" was never the claim.

It bundles two different problems, which the earlier entries did not separate:

1. **Window contamination.** `correct, first uncovers` has a median gap of
   1.375 s, so the 0–0.5 s test window contains nothing else. `correct, later`
   has 0.425 s, so the same window usually CONTAINS THE NEXT BUTTON PRESS. That
   is a mechanical difference in window content, no theory of brain state
   needed.
2. **Brain state.** Longer quiet → more LIA → more ripples.

Q2 below separates them.

**Limitation of the measure itself:** this is BUTTON stillness. No eye
tracking, no motion capture, no accelerometry. A subject sitting motionless and
thinking hard is indistinguishable from one resting.

### Q2 — inside a long pause, ripples are FLAT

Raw rate in 0.5 s slices from the press that opens each gap ≥ 2.5 s:

| source | 0–0.5 | 0.5–1 | 1–1.5 | 1.5–2 | 2–2.5 |
|---|---|---|---|---|---|
| correct uncovering (n=61, 3,734 ev) | 0.230 | 0.219 | 0.188 | 0.186 | 0.199 |
| movement press (n=43, 1,522 ev) | 0.244 | 0.239 | 0.148 | 0.218 | 0.180 |
| grid end, task-free (n=61, 945 ev) | 0.176 | 0.216 | 0.195 | 0.230 | 0.201 |

**No ramp, no decay — roughly 0.18–0.24 Hz throughout.** Aligning to the END of
the gap gives the same flat picture.

This matters for interpretation. The stillness "dose-response" reported on
2026-09-13 (−0.004 Hz for gaps < 0.5 s rising to +0.075 Hz for gaps ≥ 2.5 s)
is therefore **not** ripples accumulating as a pause lengthens. It is a
difference between two behavioural regimes — quiet stretches sit at ~0.20 Hz
throughout, busy stretches lower — plus, for short gaps, the next press landing
inside the measurement window. Problem 1 above, not problem 2.

### Q3 — task-free rest has the HIGHEST rate in the dataset

SK's idea: after the last D of a grid the subject has finished and the next
grid has not started, so that gap is stillness with no task on it. It works —
1,431 grid endings, 945 with a gap of 2.5–60 s, median 5.92 s, all 61 sessions
contribute (~16 usable per session).

Raw rate 1.5–2.5 s into the gap, by what opened it:

| source | n | rate |
|---|---|---|
| correct uncovering | 61 | 0.1923 Hz |
| movement press | 43 | 0.1988 Hz |
| **grid end (task-free)** | 61 | **0.2157 Hz** |

Paired:

| contrast | Δ Hz | t | p_perm |
|---|---|---|---|
| correct uncovering − grid end | **−0.0234** | t(60) = −2.08 | **0.046** |
| movement press − grid end | +0.0034 | t(42) = +0.14 | 0.894 |
| correct uncovering − movement press | −0.0183 | t(42) = −1.06 | 0.298 |

**The task-free period carries MORE ripples than in-task stillness**, and it is
the highest rate of the three. This is the opposite of "task events drive
ripples" and it is what the classical view predicts: ripples are maximal in
quiet rest with no task demand. It also gives the project a clean reference
rate for "still, nothing to do" — 0.216 Hz — which no task-locked condition in
this dataset exceeds.

Caveat: a grid-end gap still BEGINS with a D uncovering, so it is the aftermath
of a task event rather than a truly neutral epoch; reading it at 1.5–2.5 s is
what keeps any D transient out. The next grid's onset is a median 4.5 s away,
so it is outside the read window for most grids.

### Q4 — event vs stillness-matched control, with no baseline window at all

SK's suggestion, and it is the better design: let the matched pause BE the
reference instead of measuring both against their own pre-event baselines. That
also removes the pre-event stillness imbalance flagged as Limitation 2 in the
previous entry, since no pre-event window enters the statistic.

Restricted to gaps ≥ 2.5 s so both sides are stillness-matched, raw Hz,
sliding cluster test over all positions:

| feedback cell | n | 0–0.5 s Δ Hz | t | p | cluster |
|---|---|---|---|---|---|
| correct, first uncovers | 32 | +0.0202 | +0.58 | 0.56 | none |
| correct, later | 43 | −0.0264 | −0.91 | 0.37 | none |
| error, first uncovers | 7 | −0.0482 | −1.17 | 0.29 | none |
| error, later | 14 | −0.0673 | −2.17 | 0.050 | decrease +0.25…+0.55 s, p = 0.011 |

`correct, first uncovers` — the cell the whole question is about — is +0.020 Hz
and not significant, with no surviving cluster, under the cleanest design
available. The `error, later` cluster rests on 14 sessions and is one of four
tests; not interpreted.

Cost of the design: requiring a ≥ 2.5 s gap on BOTH sides drops n hard
(61 → 32 for correct/first, → 7 for error/first). This is the least biased and
least powered of the three controls run; it agrees with the other two.

### Where the three controls now stand

Stratification (62% of the stage effect attributable to stillness), matched
movement presses (feedback − pause ≈ +0.01 Hz, ns), and now the no-baseline
matched contrast (+0.020 Hz, ns) all point the same way. Against that,
task-free rest carries MORE ripples than any feedback moment. F5 (first-D) has
still not been put through any of these and remains the outstanding job.


## 2026-09-13 (b) — Feedback moments are not distinguishable from pauses of the same length

SK's proposed control, and it is the decisive one: compare feedback moments
against **randomly occurring moments of matched stillness**. Implemented as
`scripts/swr_feedback_vs_matched_stillness.py`. Results in
`data/ephys_humans/derivatives/group/swr/feedback_vs_matched_stillness_2026-09-13/`.

**Control events.** Every MOVEMENT press — a button press that uncovers
nothing, so no information arrives — with its own time-to-next-press.
151,076 of them. Chosen over random time points because a random time point
has no motor onset and would differ from an uncovering in two ways at once;
N1 already showed movement presses carry no ripple modulation on their own.

**Design.** Stratify both feedback events and movement presses by time to the
next key press of any kind, then contrast feedback − pause WITHIN each bin,
paired by session. Same session, same stillness, same motor act; the only
remaining difference is whether information arrived. Stratification rather
than 1:1 matching, so nothing is sampled or discarded (the ≥2.5 s bin holds
only 1,608 movement presses, so 1:1 matching would need replacement there).

### 1. How much does stillness alone do? A lot.

Movement presses vs their own baseline, 0–0.5 s, by how long the subject then
sat still — no feedback anywhere in this analysis:

| stillness after the press | Δ rate 0–0.5 s | t | p_perm |
|---|---|---|---|
| 0–0.5 s | −0.0043 | −0.78 | 0.44 |
| 0.5–1 s | +0.0062 | +1.11 | 0.28 |
| 1–1.5 s | +0.0183 | +1.39 | 0.17 |
| 1.5–2.5 s | **+0.0607** | **+3.09** | **0.0023** |
| ≥ 2.5 s | **+0.0751** | **+3.03** | **0.0026** |

**A pause with nothing in it produces +0.06 to +0.075 Hz — larger than the
entire correct/first-uncovers effect (+0.056 Hz).** Any event class enriched
for long pauses inherits a rise of this size for free.

### 2. Is feedback more than that? No.

Feedback − matched pause, stillness-standardised across bins:

| feedback cell | n | Δ Hz | t | p_perm | cluster |
|---|---|---|---|---|---|
| correct, first uncovers | 58 | +0.0105 | +0.58 | 0.56 | none |
| correct, later | 61 | −0.0020 | −0.19 | 0.85 | none |
| error, first uncovers | 58 | −0.0057 | −0.61 | 0.54 | none |
| error, later | 40 | +0.0016 | +0.13 | 0.90 | none |

**None of the four cells differs from a movement press followed by the same
amount of stillness**, at any bin-standardised estimate or in any surviving
cluster. Per-bin, `correct, first uncovers` runs +0.016, +0.021, +0.015,
+0.007, +0.010 Hz — consistently positive but never significant and an order of
magnitude smaller than the stillness effect it sits on.

Two of twenty bin-level tests reach p < 0.05 — `error, first uncovers` at
1.5–2.5 s (+0.114, p = 0.006, n = 17) and `error, later` at ≥2.5 s (−0.116,
p = 0.015, n = 12). One is expected by chance, they have opposite signs, both
have the smallest n in the table, and neither survives standardisation. They
are noise.

### What this means, and what it does not

The feedback-locked ripple rise in this dataset is, as far as this control can
tell, the pause. Three independent lines now agree: stratification removes it,
direct standardisation attributes 62% of the stage difference to stillness, and
a matched non-feedback press reproduces it entirely.

**This does not by itself overturn F5** (ripples after the first uncovering of
D), which was the one claim stated before any analysis. F5 was tested on a
different contrast — first-D against its own baseline with a sliding cluster
test — and has NOT been put through this control. It should be, and that is
the obvious next job: first-D is exactly the kind of event followed by a long
pause, so it is the most exposed claim in the project, not the least.

**Limitation 1 — the control is not neutral either.** A movement press followed
by 2 s of stillness is not a blank moment: the subject may have arrived
somewhere and stopped to think. If deliberation pauses carry ripples, the
control absorbs genuine signal and the test is conservative. Separating pause
TYPES rather than pause LENGTHS is the way past this: VTE-like deliberation
pauses are theta-rich and ripple-poor, quiet-rest pauses the reverse. The LFP
needed for that split exists (`continuous.npy` per session) but no theta
measure is in the bundle yet.

**Limitation 2 — matching on following stillness leaves PRECEDING stillness
unbalanced.** Median time since the previous press, within each
following-stillness bin:

| bin (time to next press) | correct, first | correct, later | movement press |
|---|---|---|---|
| 0–0.5 s | 0.275 | 0.400 | 0.425 |
| 0.5–1 s | 0.312 | 0.500 | 0.550 |
| 1–1.5 s | 0.350 | 0.625 | 0.800 |
| 1.5–2.5 s | 0.350 | 0.738 | **1.025** |
| ≥ 2.5 s | 0.400 | 0.500 | **1.200** |

A movement press with a long pause after it tends to have a long pause before
it too — it is an isolated press in an already-idle stretch — whereas a
feedback event with a long pause after it was reached by active walking. The
baseline window sits at −1.6 to −1.1 s, so the control's baseline should be the
more ripple-rich one, which would DEFLATE the control's (window − baseline) and
therefore INFLATE feedback − pause. The observed feedback − pause is ~+0.01 Hz
and non-significant even with that bias pushing in its favour, so the null
conclusion is the safe direction. This is a reasoned direction, not a measured
one — the rigorous fix is to stratify on preceding AND following stillness
jointly, which is the next thing to build if this control is to be quoted.


## 2026-09-13 — Positive feedback early vs late: 62% of it is stillness

Follow-up to 2026-09-11. SK pointed out that the contrast she is actually
looking at is not the interaction I had tested: it is **positive feedback
early vs positive feedback late** (correct only, across stages), not
`(correct − error) × stage`. Tested directly, with the stillness controls the
project's own F1 demands.

New script: `scripts/swr_stillness_control.py`. Results in
`data/ephys_humans/derivatives/group/swr/stillness_control_correct_2026-09-11/`.
Bundle `swr/bundle` (2026-09-08), 61 sessions, unit = session, min 10 events
per session per cell per bin, 0–0.5 s vs the same trial's baseline.

### Which number is which

Worth stating because the two are easy to conflate, and the `feedback_stage`
figure shows the first while the question is about the second:

| quantity | value | verdict |
|---|---|---|
| `correct, first uncovers` vs **its own baseline** (the `**` in the figure) | +0.056 Hz, t = +2.76, p_perm = 0.006; cluster +0.25…+0.75 s, p = 0.006 | **holds**, survives cluster correction |
| `correct, first` − `correct, once known` | +0.0396 Hz, t = +2.14, p_perm = 0.034 | one of 3 windows, no cluster |
| `correct, first` − `correct, while learning` | +0.0249 Hz, t = +1.16, p_perm = 0.257 | no |
| `correct, first` − `correct, later` (pooled) | +0.0360 Hz, t = +2.03, p_perm = 0.044 | one of 3 windows, no cluster |

A condition can beat its own baseline strongly and still not beat another
condition: the second test is paired and has to clear the other condition's
rise as well.

### Stillness is a five-fold effect on this very window

Measured on the correct uncoverings themselves, pooled over stages so it
carries no stage information, binned by time to the next key press of any kind:

| stillness after the event | Δ rate 0–0.5 s |
|---|---|
| 0–0.5 s | +0.010 Hz |
| 0.5–1 s | +0.015 Hz |
| 1–1.5 s | +0.029 Hz |
| 1.5–2.5 s | +0.052 Hz |
| ≥ 2.5 s | +0.039 Hz |

And the two stages sit in completely different places on that curve:

| stillness bin | % of `first uncovers` | % of `later` |
|---|---|---|
| 0–0.5 s | 16.9 | 58.2 |
| 0.5–1 s | 18.4 | 28.0 |
| 1–1.5 s | 17.8 | 6.4 |
| 1.5–2.5 s | 23.0 | 3.5 |
| ≥ 2.5 s | 23.9 | 3.8 |

### Stratifying removes most of it

Same contrast computed INSIDE each stillness bin, so like is compared with
like:

| stratum | n | Δ Hz | t | p_perm |
|---|---|---|---|---|
| unstratified | 61 | +0.0360 | +2.03 | 0.044 |
| 0–0.5 s | 32 | +0.0281 | +0.74 | 0.489 |
| 0.5–1 s | 35 | +0.0059 | +0.17 | 0.874 |
| 1–1.5 s | 38 | +0.0214 | +0.73 | 0.472 |
| 1.5–2.5 s | 33 | +0.0242 | +0.71 | 0.496 |
| ≥ 2.5 s | 38 | −0.0281 | −0.50 | 0.629 |
| **stillness-standardised** | **58** | **+0.0169** | **+0.72** | **0.488** |

No surviving cluster for the standardised contrast at either width.

**Direct standardisation.** Taking the rate-vs-stillness curve above (measured
with both stages pooled, so it is stage-blind) and applying each stage's own
stillness distribution to it:

- observed: +0.0360 Hz (t = +2.03)
- predicted by the stillness imbalance alone: **+0.0223 Hz (t = +2.70)**
- residual: +0.0136 Hz, t = +0.89, **p = 0.376**
- **stillness accounts for 62% of the observed difference**, and the remaining
  38% is not distinguishable from zero.

Caveat, stated because it cuts the other way: stratifying costs sessions
(61 → 32–38 per bin), so "not significant within bins" is partly power. The
informative part is that the effect SIZE drops with it (+0.036 → +0.017
standardised), which power loss alone would not do.

### Interpretation, and why this is not automatically a confound

Stillness may be a MEDIATOR rather than a confounder: uncovering a reward for
the first time may cause the pause, and the pause carries the ripples. That is
a real possibility and controlling for stillness then removes part of the
causal path. But the claim that survives in that case is "discovering a reward
makes people pause, and pauses carry ripples", not "the hippocampus signals
discovery". Distinguishing them needs a comparison against pauses of the same
length that contain no feedback — see the next entry.


## 2026-09-11 — Valence × stage is a stillness effect; ripples do not predict later errors

Two questions from SK about the `feedback_stage` panel at `--unit=session
--min_events=15`: (1) the six cells look crossed — positive feedback raises the
rate early in a grid, negative feedback late — can that be tested as an
interaction? (2) does a higher ripple rate early predict fewer mistakes later?

Two new scripts, both reusing the existing test functions rather than
re-implementing them:
`scripts/swr_valence_stage_interaction.py`, `scripts/swr_ripples_predict_errors.py`.
Two new helpers in `mc/analyse/ripples.py`: `contrast_profiles` and
`hotelling_signflip`.

Results: `data/ephys_humans/derivatives/group/swr/valence_x_stage_2026-09-11/`
and `.../ripples_predict_errors_2026-09-11/`. Bundle `swr/bundle` (2026-09-08),
61 sessions, 41 subjects, 180 derivations, 64,760 ripples.

### How the interaction is built

Baseline subtraction is linear, so a weighted combination of RAW per-session
rate profiles, baselined afterwards, is identical to combining already-baselined
profiles (checked: max abs difference 2.2e-16). An interaction contrast
therefore passes through `rip.baseline_subtract`, `rip.window_test` and
`rip.sliding_window_test` unchanged — same sign-flip null, same cluster
correction, no second code path.

### 1. The interaction exists in the right direction and does not survive

0–0.5 s vs the same trial's baseline, session as the unit, min 15 events/cell:

| contrast | n | Δ Hz | t | p_perm |
|---|---|---|---|---|
| correct − error, first uncovers | 61 | +0.0555 | +3.03 | **0.0031** |
| correct − error, while learning | 33 | +0.0326 | +1.27 | 0.224 |
| correct − error, once known | 42 | −0.0042 | −0.16 | 0.866 |
| valence × stage, linear (first − once known) | 42 | +0.0503 | +1.41 | 0.161 |
| valence × stage, quadratic | 27 | +0.0284 | +0.36 | 0.731 |
| valence × stage, pooled (first − later) | 56 | +0.0486 | +1.83 | 0.073 |

Omnibus 2 × 3 (linear + quadratic jointly, Hotelling T² with sign-flip null,
sessions supplying all six cells): **F(2,25) = 0.08, p_perm = 0.924, n = 27**.

Sliding cluster test, both widths: the simple effect at *first uncovers*
survives (+0.25 to +0.75 s, p = 0.006 at 0.3 s; +0.15 to +0.75 s, p = 0.017 at
0.5 s). **No interaction contrast produces any surviving cluster.**

The pooled interaction reaches +0.0685 Hz, t = +2.34, p_perm = 0.021 in the
0.5–1.0 s window, but that is one of three windows inspected (Holm-adjusted
p = 0.063) and no cluster survives, so it is not a result.

Why the interaction is so much weaker than the eye suggests: it is paired, and
pairing costs sessions. `error, while learning` is estimable in 33 sessions and
`error, once known` in 42, so the 2 × 3 crossing has n = 27. Pooling the two
later stages at the EVENT level (before the min-events filter) recovers n = 56,
and that is the best-powered version available.

### 2. The interaction is a stillness artefact

F1 says ripple rate rises with stillness. Stillness — time from an uncovering to
the next key press of ANY kind, movement included, taken from the 25 ms button
series via `swr_probes.press_times` — is wildly unbalanced across these cells:

| cell | median time to next press |
|---|---|
| correct, first uncovers | **1.375 s** |
| correct, while learning | 0.525 s |
| correct, once known | 0.425 s |
| error, first uncovers | 0.450 s |
| error, while learning | 0.575 s |
| error, once known | 0.625 s |

The same interaction contrast on the BEHAVIOUR: **+1.310 s, t(60) = +11.34,
p_perm < 1e-4** (2-stage) and +1.575 s, t = +8.11 (3-stage). The behavioural
interaction is an order of magnitude more significant than the neural one and
has the identical shape. After a correct uncovering on the first traversal the
subject stops and looks; in every other cell they press on within half a second.

Sweeping a common stillness criterion applied identically to all cells
(0–0.5 s window, pooled 2-stage crossing):

| min stillness | valence effect, first uncovers | interaction (first − later) |
|---|---|---|
| none | +0.0555 (n=61, p=0.003) | +0.0486 (n=56, p=0.073) |
| ≥ 0.5 s | +0.0513 (n=59, p=0.006) | +0.0410 (n=47, p=0.170) |
| ≥ 1.0 s | +0.0514 (n=39, p=0.079) | **−0.0066 (n=30, p=0.879)** |
| ≥ 1.5 s | +0.0188 (n=16, p=0.634) | −0.0240 (n=12, p=0.564) |
| ≥ 2.0 s | +0.0223 (n=6, p=0.778) | −0.0153 (n=5, p=0.935) |

The dissociation matters. The **simple effect at first uncovers keeps its
effect size** through ≥ 1.0 s (0.0555 → 0.0513 → 0.0514; only n and therefore p
degrade), so it is not obviously stillness. The **interaction does not** — it
falls to zero and changes sign while n is still 30. Reading the interaction as
neural is not supported.

Caveat on the sweep: thresholding changes cell composition as well as
equalising stillness, and n falls steeply because only 11% of `error, first
uncovers` events are followed by ≥ 1 s of stillness (vs 65% of `correct, first
uncovers`). Distribution matching within session × stage would preserve more n
and is the obvious next step if this is to be pushed further.

### 3. Ripple rate does not predict later errors; stillness does

Unit: the grid (1,381–1,429 grids, 61 sessions, median 24 grids/session).
Spearman within session across grids → Fisher z → t across sessions; null =
1,000 shuffles of the outcome across grids WITHIN session, run through the same
`group_statistic`. Predictor = ripple rate 0–0.5 s minus the same trial's
baseline, averaged over the correct uncoverings of that grid in that stage.

Outcome `errors_total` (every wrong uncovering in the grid):

| predictor | ripples ρ | p_perm | stillness ρ | p_perm |
|---|---|---|---|---|
| all correct uncoverings | +0.019 | 0.422 | **+0.245** | **0.001** |
| first uncovers | −0.000 | 0.988 | **+0.126** | **0.001** |
| while learning | +0.021 | 0.562 | **+0.350** | **0.001** |
| once known | −0.001 | 0.946 | **+0.147** | **0.001** |

Outcome `errors_after` (wrong uncoverings strictly after the last predictor
event — the clean predictive version for `first uncovers`): ripples +0.016,
p = 0.598. First traversal vs the mean of the two later stages: Δz = −0.010,
t(60) = −0.25, p_perm = 0.806 (errors_total); Δz = −0.037, p_perm = 0.592
(errors_after). Ripple rate with stillness partialled out within session:
unchanged, all |ρ| ≤ 0.024.

So the answer is no, and it is a well-powered no: 1,381 grids, and the same
design detects a ρ of +0.13 to +0.35 for stillness at p = 0.001. The sign of
the stillness effect is worth noting — **more pausing goes with MORE errors**,
i.e. it indexes how hard the grid was, not how well it was learned.

`errors_after` is near-degenerate for the `all` and `once known` predictors by
construction (almost no errors follow a grid's last correct uncovering), which
is why their n drops to 26 and 23 sessions; those two cells should be ignored.

### Status

Nothing here was predicted in advance; all of it is exploratory. What survives:
the **simple** valence effect at the first traversal, which was already visible
in `feedback_stage`. What does not: the valence × stage interaction, and any
ripple–behaviour prediction.


## 2026-09-10 — RSA jobs were being killed on walltime; raised -T and made them resumable

**Cause, confirmed from the job log:** `JOB 2348457 ... CANCELLED AT
2026-09-09T15:17:18 DUE TO TIME LIMIT`. `submit_RSA_instruction_epochs.sh`
passed `fsl_sub -T 30`, but one RSA job fits 28 whole-brain searchlight OLS
models (21 single + 7 combo) over ~126k searchlights. Jobs died about two
thirds of the way through, leaving `results/` folders that looked plausible:
sub-01 `instr_see-A-first` held 20 of 41 beta maps -- every single model in
config order up to `D_rew`, then nothing, and no combo maps at all. The
`.pdf`/`.png` panels are drawn before any OLS runs, so the folder looked full.
`check_RSA_ran.py` was right to list those runs; the diff against
`expected_map_names` showed 21 missing and 0 unexpected, i.e. what was on disk
was a strict prefix of what should be.

**Fix 1 -- time.** `-T 30` -> `jobTime="${FSLSUB_T:-240}"`, overridable per
submission.

**Fix 2 -- resume.** `fMRI_run_RSA_instruction.py` now skips any map whose
`_beta` / `_t_val` / `_p_val` volumes are all present and non-empty, so a job
that still runs out of time picks up where it stopped instead of starting over.
Without this a resubmission recomputes all 28 fits and dies at the same
wall-clock point, so it never converges.

Skipping is gated on settings, because otherwise a rerun silently mixes two
analyses in one folder:
  * the settings summary is only written at the END of a run, so a killed job
    never leaves one -- the script now writes `{sub}_run_settings.json` at the
    START and resumes only when it matches;
  * a folder left by a run that FINISHED has no such file but does have a
    settings summary, which is compared against instead (same settings under
    their own key names);
  * a folder with maps and neither file cannot be verified and is NOT resumed,
    unless `"resume_unverified": true` is set in the config. That flag is
    deliberately narrow: it only applies when there is no evidence at all, so
    it cannot trust maps from a different completed analysis.
  * `"resume": false` disables it entirely.
Combos are skipped only when ALL of a combo's maps exist (one OLS produces them
all); the regressor-correlation record is computed either way, so the settings
summary keeps its collinearity entry for skipped combos. The summary records
`resume_enabled` and `resumed_maps`, so a folder completed across several jobs
says so.

**Audit location.** Both audits now write to `analysis/logs_mid_sept/` rather
than `derivatives/group/` -- they are logs, not results. On a laptop
`$analysisDir` is the repo itself, so the shell wrapper redirects there to the
data tree instead; nothing generated ever lands in the repo.

Verified by executing the real resume block out of the script against eight
folder states: fresh, matching rerun, partial map (beta only, no t_val),
changed settings, no run-settings file, `resume_unverified`, `resume: false`,
and an old completed run with different settings plus `resume_unverified` (the
one that must still refuse). All behaved as intended.

## 2026-09-09 — RSA audit emits a per-wrapper resubmission plan

`check_RSA_ran.py` now tracks all five pipeline stages and writes
`resubmit_plan.sh`: the exact commands that would fill every gap, per wrapper,
in pipeline order.

    results/                fMRI_run_RSA_instruction.py   per subject x epoch
    smoothed/               smooth_subject_space.py       per subject x epoch
    standard-space-smooth/  applywarp wrapper             per subject x epoch
    group_..._glmbase_{epoch}/          merge_subj_to_group.sh        per epoch
    group_..._glmbase_{epoch}_cropped/  mask_subj_by_missingvoxels.sh per epoch

Two gating rules keep the plan honest rather than merely complete:
  * a stage only lists work whose PREDECESSOR is complete, so nothing is
    submitted that would read half-written inputs;
  * a run queued for an RSA rerun is excluded from the downstream stages
    entirely -- its maps are about to be rewritten, so smoothing them now is
    wasted queue time. (Caught in testing: a subject whose maps were all
    present but whose settings summary was missing was being offered for
    smoothing and RSA resubmission at once.)
Group stages additionally require the epoch to have NO subject-level gap left,
because merging a partial set gives the wrong volume count and
mask_subj_by_missingvoxels.sh then rejects it on `required_n`.

Smoothing is submitted per (subject, epoch) through
wrapper_python_fMRI_RSA_clean_config.sh rather than through
wrapper_smooth_stat_maps_subj.sh: smooth_subject_space.py has no
skip-if-exists, so the wrapper's loop would re-smooth every subject of the
epoch. That needs a per-epoch smoothing config, which the audit now writes
(`smooth5_config_{epoch}.json`) so concurrent jobs cannot rewrite each other's
`regression_version` in the shared file.

The stages are dependent, so the plan is not a script to run top to bottom: run
one stage, wait for the queue, re-run the audit for the next plan.

Verified on a synthetic tree through the whole chain: mixed subject states ->
RSA + smoothing + standard-space plan; all subjects complete -> merge offered
and crop still gated; merged present (as gunzipped .nii, which the checker
accepts) -> crop offered and merge gone.

## 2026-09-09 — RSA audit now checks all three stages per model

`check_RSA_ran.py` only asked whether the RSA had finished, per (subject,
epoch). It now walks the whole per-subject pipeline, per MODEL, at each stage:

    results/                {map}_beta.nii.gz          fMRI_run_RSA_instruction.py
    smoothed/               smooth_fwhm5_{map}_beta.nii.gz      smooth_subject_space.py
    standard-space-smooth/  smooth_fwhm5_{map}_beta_std.nii.gz  applywarp wrapper

Only the beta map is tracked: it is what the group merge collects
(`*beta_std.nii.gz`) and what `loso.py` reads, so a missing beta is what
actually breaks the pipeline; t_val / p_val are written in the same call.

`expected_map_names` derives the full set of maps from the config, mirroring
the output naming of fMRI_run_RSA_instruction.py: `{model}`, or
`{model}_within` / `_across` where `single_model_scopes` entitles a model to
several scopes, plus `{REGRESSOR}-{combo}` per combo regressor (with the same
scope suffix on the combo name, and the `block` nuisance when
`add_block_nuisance` is set). For rsa_instruction_cumulative_rew.json that is
**41 maps** per (subject, epoch) — 21 single + 20 combo.

New statuses distinguish what actually needs rerunning: `NOT_STARTED` /
`RESULTS_INCOMPLETE` / `RERUN_CHANGED` need the RSA resubmitted and go on
todo_rsa.txt, while `SMOOTHED_INCOMPLETE` and `STANDARD_INCOMPLETE` mean the
RSA is fine and only the cheap downstream wrapper has to be rerun — those are
reported separately and deliberately kept OFF the submission list.
`RESULTS_INCOMPLETE` also covers "all maps present but no settings summary",
i.e. the run died before its last action.

Outputs gain a per-stage table (complete / partial / none / maps missing, with
the script that produces each stage) and `missing_maps.txt`, one line per
individual map absent from disk. settings.json records the expected map names
and the stage tally.

Verified on a synthetic tree covering seven states — all stages complete,
smoothed partial, standard partial, results partial, nothing at all, results
without a summary, and settings changed. Each classified correctly, the stage
tallies reconcile exactly (46 = 5 + 41, 125 = 2 + 3x41, 165 = 1 + 4x41), and
the two downstream-only gaps stayed off todo_rsa.txt.

## 2026-09-09 — group inference runs per TR or per condition (`--axis`)

`per_TR_loso.py` assumed its third axis was the ordered seconds of the
instruction period: `--trs` was parsed with `int()` and substituted into
`dir_pattern.format(tr=tr)`, so the instruction-epoch folders
(`group_RSA_instr_cumrew_glmbase_instr_see-A-first_cropped`) could not be read
at all.

**`--axis {tr,condition}`.** `tr` (default) is unchanged: levels from `--trs`,
peaks reported as TR numbers, figure drawn as a timecourse with the reward
schedule. `condition` takes levels from `--conditions` (the epoch GLM names)
and substitutes them as `{condition}` / `{tr}` / `{level}` in `--dir-pattern`.

**The statistics are identical in both modes** — the same max-t sign-flip
permutation over voxels x levels, one family per mask, through the same
`tstat` / `null_max_t` (CLAUDE.md rule 4). Only reporting differs: `peak_TR`
fields hold the condition name, and `settings.json` records `axis`, `levels`
and an `axis_note` stating that a condition axis must not be read as a
timecourse.

**Plotting is deliberately different per axis.** A condition axis is drawn as
points with error bars, rotated condition labels, no joining line and no reward
schedule — a line between unordered conditions would assert a progression the
design does not contain. A non-numeric axis is forced to that style whatever is
requested, so a mislabelled call cannot draw a timecourse through condition
names. Models are dodged horizontally (x-offset only, no value altered) so
overlapping points stay readable. `--mode plot` reads the axis kind back from
the run's `settings.json`, so old per-TR folders still plot as timecourses.

**Two bugs found while doing this:**
- `CHANNEL_COLOURS` / `CHANNEL_LABELS` still used the pre-rename model names
  (`curr_rew`...), while the RSA now writes `A_rew`...`D_rew`. Every map would
  have plotted grey and the default "four reward channels" selection would have
  matched nothing, silently falling back to plotting all 14 models. Rekeyed to
  the current names with the project state colours; `LEGACY_CHANNEL_ALIASES`
  keeps pre-rename result folders colouring correctly. The cumulative channels
  (AB/ABC/ABCD_rew) stay grey rather than inventing a colour convention.
- `summary_row` reported `loso_peak_TR` as the POSITION along the axis, not the
  level: with `--trs 4,5,6` it printed 0/1/2. Now the level.

Verified end to end on synthetic 32-subject volumes with a planted blob: TR mode
recovers the planted TR, condition mode recovers the planted condition, both
figures render, and `--mode plot` picks the right style from settings.json.

## 2026-09-08 — UCLA's own anatomical labels vs our atlas ROIs (`ucla_label_vs_atlas_roi.py`)

UCLA report that their electrode localisation is very precise, so the labels
they ship were used as an independent check on the ROIs
`cell_to_roi_july26.py` assigns. That script reads only the MNI coordinate out
of `sub-{NNN}_localizations.xlsx` and discards every label column; this
compares those discarded labels against `atlas_roi` / `alt_final_roi`. **Read
only — the ROI table is unchanged.**

140 UCLA cells on 24 microwire bundles, joined on
`(Subject Label, source_electrode)` — the same bundle the coordinate came
from, so the two verdicts describe the same contact. Seven UCLA columns were
mapped onto our ROI vocabulary: `ASHS_ABC` (subject's own T2, MTL subfields),
`aparc+aseg` / `aparc.DKTatlas+aseg` (subject's own T1, FreeSurfer native
space), `Anat` and `AnatMacro_1` (SPM Anatomy toolbox), `NMM`
(Neuromorphometrics), and `region` (the implantation-target code). Labels that
make no regional claim (`Unknown`, `*Cerebral-White-Matter`, `(extra-axial)`)
are scored as "no verdict", not as disagreements. `HC_anterior`/`HC_mid` is our
own y = -21 split and no UCLA column encodes it, so agreement is scored on the
collapsed `HC`.

**Headline: 114/140 cells (81.4 %) and 20/24 bundles agree** with at least one
UCLA column. Per column, over the contacts where that column has a verdict:
NMM 84 %, AnatMacro_1 65 %, aparc+aseg 65 %, DKT 61 %, Anat 54 %,
implantation-target code 48 %. ASHS agrees 11/12 where it is defined.

**The disagreements are not spread out — they are exactly the neighbourhood
fallback.** Splitting by `atlas_reason`:

| assignment mode | cells with a UCLA verdict | agree | % |
|---|---|---|---|
| exact voxel | 104 | 103 | **99.0** |
| `neighbor@1/2/3mm` probe | 28 | 11 | **39.3** |

The single exact-voxel disagreement is a `leftover` cell (Heschl's gyrus,
dropped anyway). So wherever the cell's own voxel lands in an atlas region,
UCLA's independent labelling confirms us essentially always; the risk is
concentrated in rule 12, where the voxel is white matter and we probe outwards.

**What this costs the analysis ROIs.** 93 UCLA cells survive into analyses on
13 bundles; 12 bundles are confirmed, one is not:

* **`RPv-micro` (UC3-0582, 17 cells, currently `HC_mid`) is almost certainly
  pulvinar.** Coordinate (12.2, -31.6, 3.9), assigned via
  `hippocampal_subfield_split_y-21.0_neighbor@2mm`. All four UCLA columns say
  thalamus (`Right-Thalamus`, `Right Thalamus Proper`, `Thal: Temporal`,
  `R Thalamus`) and the electrode is named `RP` = right pulvinar. This is
  **half of the UCLA `HC_mid` cells** (17 of 35).
* `RPHG_micro-1` (UC3-0573, 1 cell, `HC_mid`) — ASHS `Right PHC`, both aparc
  columns `parahippocampal`, AnatMacro `R ParaHippocampal Gyrus`. Also a
  `neighbor@1mm` assignment. Probably PHC.

Per analysis ROI, cells confirmed by at least one UCLA column: PCC 22/22,
mPFC 4/4, EC 3/3, HC_anterior 29/29, **HC_mid 18/35**.

**EC survives the strictest test available.** The 3 EC cells (`LEC-micro`,
UC2-0578) are confirmed by ASHS (`Left_ERC`, segmented on that subject's own
T2) and by SPM Anatomy (`Entorhinal Cortex`) — the two most anatomically
specific columns UCLA ship.

**Caveat: `aparc+aseg` is not uniformly trustworthy, so its 65 % is a floor.**
Per-subject concordance between `aparc+aseg` and `NMM` on the same contacts is
100 % (UC3-0559), 100 % (UC3-0577), 86 % (UC3-0582), 50 % (UC2-0576, UC3-0573)
but **20 % for UC2-0578**, whose column labels an entorhinal contact at
z = -38 as `ctx-lh-insula`, an amygdala contact as `Left-Putamen` and a
hippocampal one as `Left-Pallidum` — anatomically impossible, all displaced the
same way. That subject's FreeSurfer columns look mis-registered; its ASHS and
Anat columns behave normally. So "UCLA's method is precise" holds for their
MTL segmentation, less so for the whole-brain FreeSurfer columns in every
subject.

**Mechanism: there are two neighbourhood probes and only one is dangerous.**

* **Path A — the HC probe inside rule 4** (`hippocampal_subfield_..._neighbor@Nmm`).
  It fires whenever the exact voxel is not hippocampus, and returns *before
  rules 5-11 are ever consulted*, so it can override a perfectly good
  exact-voxel verdict from a lower-priority rule. Its design comment scopes it
  against rule 5 only ("otherwise leaves subicular voxels to be captured by the
  coarser parahippocampal-gyrus rule despite being anatomically hippocampal") —
  but it sits above rules 5-11 and therefore beats *all* of them, including
  Thalamus and Amygdala. That is the RPv bug: at (12.2, -31.6, 3.9) the exact
  voxel is HO-subcortical `Right Thalamus`, rule 10 would have caught it, but
  rule 4's probe finds Juelich cornu ammonis 2 mm away first.
* **Path B — rule 12 proper** (`atlas_neighbor@Nmm`) is reached only when *no*
  rule matched at the exact voxel, so it can only promote `leftover`. It cannot
  contradict an exact-voxel label and is structurally safe.

**Path A is confined to HC by construction** — it lives inside rule 4 and can
only return `HC_anterior`/`HC_mid`. So no other ROI is exposed to this failure
mode. Probe cells in analysis ROIs, whole table: HC_mid 56 (all path A),
HC_anterior 11 (all path A), mOFC 23, mPFC 13, PCC 4 (all path B), **EC 0**.

Re-querying the exact voxel of all 67 path-A cells across all three sites:

| overridden exact-voxel label | HC_ant | HC_mid | sites |
|---|---|---|---|
| PHC — **intended** override | 4 | 39 | Baylor 41, UCLA 1, Utah 1 |
| Thalamus — **not** intended | 0 | 17 | UCLA (RPv) |
| Amygdala — **not** intended | 5 | 0 | Utah |
| white matter — probe justified | 2 | 0 | Utah |

The 43 PHC cells are the documented purpose of rule 4c (the subiculum sits in
what HO calls parahippocampal gyrus) and are **not** errors. The 17 Thalamus
and 5 Amygdala overrides are side effects of the probe outranking rules 10 and
9, which its own rationale never claims.

**Correction to the note above on `RPHG_micro-1`:** it is *not* clearly wrong.
Its exact voxel is HO `Parahippocampal Gyrus, posterior division` and the probe
found Juelich subiculum at 1 mm — the intended rule-4c behaviour — and UCLA's
own SPM Anatomy column independently says `Subiculum` there. ASHS says
`Right PHC`; the two disagree with each other, not with us. **`RPv` (17 cells)
is the only UCLA assignment contradicted by every available source.**

**Not done / open:** the 17 pulvinar cells were left in `HC_mid` — changing the
ROI table is a separate decision, not a side effect of an audit. The 5 Utah
amygdala overrides share the failure mode but have no independent labels to
check against. A minimal fix would be to make rule 4c yield to HO-subcortical
Thalamus/Amygdala at the exact voxel while keeping its PHC precedence; that
would move 17 cells (UCLA) out of HC_mid and 5 (Utah) out of HC_anterior, and
touch nothing else. Baylor and Utah ship no label columns, so the UCLA
agreement check itself covers 140 of 984 cells.

Outputs -> `ephys_humans/derivatives/ROI_assignment/ucla_label_agreement_2026-09-08/`
(`per_cell_labels.csv`, `per_bundle_labels.csv`, `agreement_summary.csv`,
`agreement_by_assignment_mode.csv`, `per_analysis_roi_verdict.csv`,
`analysis_bundles_verdict.csv`, `freesurfer_column_sanity.csv`,
`confusion_*.csv`, `settings.json`).

## 2026-09-07 — instruction RSA submitted only where it can run (`check_RSA_ran.py`)

`submit_RSA_instruction_epochs.sh` used to submit 33 subjects x 11 epochs
unconditionally. Most of those jobs either cannot succeed or need not run: the
RSA opens `glm_instr_<epoch>_pt0{1,2}.feat/stats/pe*.nii.gz` directly, so a
subject whose epoch GLM never finished produces a job that dies on a missing
PE, and a subject already analysed just recomputes the same searchlight maps.

**`check_RSA_ran.py`** decides the list first, per (subject, epoch):
`READY` (submit), `RERUN_CHANGED` (a result exists but with different settings —
submitted, overwriting the old maps, unless `--skip-changed`), `DONE` (skip),
`GLM_NOT_READY`, `MISSING_INPUT` (modelled EVs / example_func / searchlight
mask). GLM readiness calls `check_GLMs_ran.check_one`, so "complete" means the
same thing in the FEAT audit and here; `PROMOTE_TWIN` deliberately does **not**
count as ready, because the finished run is then in a `+` twin while the base
directory the RSA reads is broken.

**"Already done" means done with THESE settings.** The marker is
`{sub}_settings_summary.json`, which the RSA writes as its very last action, so
its presence means the whole run finished rather than some of the maps. Its
contents are compared field by field (`COMPARED_KEYS`: EV_string,
regression_version(+_full), TR, RDM_version, smoothing, fwhm, searchlight_mask,
data_rdm_scope, models_evaluated, run_single/combo_models, combo_models). A
field missing from an older summary counts as a difference — the settings
cannot be shown to match, so it is rerun rather than assumed done.

Writes `todo_rsa.txt`, `report.txt` and `settings.json` to
`derivatives/group/rsa_audit_<name>_<date>/`, plus one per-epoch config
snapshot (`<base>_<epoch>.json`, regression_version = the epoch GLM, TR = null)
next to the base config.

**Bug found and fixed:** `wrapper_python_fMRI_RSA_clean_config.sh` hardcoded
`fMRI_run_RSA_without_rsatoolbox_clean.py` while
`submit_RSA_instruction_epochs.sh` was already passing the script name as `$3`.
Every instruction-epoch RSA job submitted through it therefore ran the WRONG
python script. The wrapper now takes `$3`, defaulting to the old name so
existing callers are unaffected, and the submit script calls the repo copy by
path rather than the unversioned copy sitting loose in `$analysisDir`.

Jobs run in conda env `spyder-env` via `fsl_sub -T 30`. `DRYRUN=1` prints
without submitting; a todo list can be passed explicitly.

## 2026-09-07 — instruction-phase GLM audit: '+' twins handled, reruns gated

FEAT never overwrites an output directory: it appends `+` and leaves the old
one in place. Repeated submissions of the instruction-epoch GLMs (partly after
quota failures) left up to three generations of the same run side by side
(`glm_instr_see-A-first_pt01{,+,++}.feat`) in any mix of complete and broken,
while `my_RSA.py:405` only ever reads the plain `..._pt01.feat`. So a finished
GLM could sit in a `++` twin while the RSA read a truncated base, and
`check_GLMs_ran.py` reported the run as missing.

**`check_GLMs_ran.py` now audits the whole set of `+` generations per run**
instead of the base directory alone, and sorts the grid into: `OK`,
`DUPLICATES` (base complete, twins redundant), `PROMOTE_TWIN` (base
missing/broken but a twin is complete — a rename, *not* a rerun), a failure
status (no complete copy anywhere → rerun), or `NO_EV_FOLDER`/`NO_DRAFT_FSF`
(inputs missing, cannot be submitted at all). The per-folder completeness test
is unchanged (design.mat `/NumWaves` → all PEs non-empty → FILM end-markers →
`task-to-EV.txt` agrees with the design). It writes `report.txt`,
`cleanup_feat_dirs.sh`, `todo_submit.txt` and `settings.json` to
`derivatives/group/glm_audit_<version>_<date>/`.

**No data is deleted by the audit.** `cleanup_feat_dirs.sh` is a dry run unless
given `--apply`, and re-validates every path at the moment of deletion against
`*/derivatives/sub-*/func/glm_*.feat` containing a `stats/` or `design.fsf`,
refusing anything else (verified against injected paths for the derivatives
root, a `func/` dir, an `EVs_*` folder, a non-FEAT `glm_*.feat` and a `..`
traversal — all refused under `--apply`). Promotion moves the broken base aside
first and removes it only once the twin is in place, so an interrupted run
cannot leave the good data deleted.

**`subject_GLM_instruction_epochs.sh` no longer loops over the full grid.** It
runs the audit, submits only runs with no complete copy anywhere, and refuses
to start while cleanup is outstanding (a leftover base directory would send the
new run into yet another `+` twin). `ALL=1` restores the old full-grid
behaviour, `DRYRUN=1` prints without submitting, and an explicit
`todo_submit.txt` can be passed instead. A completed GLM with unchanged
settings is never resubmitted.

Verified end to end on a synthetic derivatives tree covering all cases (clean /
duplicates / promote / incomplete-only / no feat / no EVs / no draft fsf /
base-gone-but-`++`-good): audit → cleanup → re-audit converges, and only the
genuinely missing runs reach the todo list.

## 2026-09-07 — FEAT directories pruned to free scratch space (`prune_feat_dirs.py`)

Scratch ran out of space during the instruction-epoch GLMs. `mc/fmri_analysis/prune_feat_dirs.py`
strips first-level `.feat` directories down to what is actually read.

**What is deleted, and only in old GLMs.** Two levels, as intended:

- *Level 1, never touched:* GLM names matching `01`, `01-TR*`, `instr_*`,
  `all-paths-fixed_stickrews_split-buttons`, plus anything written in the last
  7 days (`--protect-days`). The age rule also covers FEAT jobs still running.
- *Level 2, pruned:* everything else under `sub-*/func/glm_*.feat`. The
  directory stays; `stats/pe*.nii.gz`, `stats/{dof,smoothness}`, the `design.*`
  text files, `absbrainthresh.txt` and `custom_timing_files/` survive.
  Removed: `confoundEV*`/`InputconfoundEV*`, `stats/{cope,varcope,tstat,zstat,
  res4d,sigmasquareds,threshac1}*`, `thresh_zstat*`, `rendered_thresh_zstat*`,
  `cluster_*`, `tsplot/`, `logs/`, `mask/mean_func/example_func`.

**No data is affected.** Every `glm_*.feat` reference in the repo resolves to
`.feat/stats/pe*.nii.gz` (grepped before writing the script); nothing reads the
deleted files. Nothing outside `sub-*/func/glm_*.feat` is considered, so
`preproc_clean_*.feat`, `EVs_*` and `motion/` are out of scope entirely.

**Reproducibility.** Scan writes `settings.json`, `report.txt`, `scan.json` and
`to_delete.txt` to `derivatives/group/feat_cleanup_<stamp>/`; deletion works
only from that manifest, re-validates every path independently, and appends
`deleted.log`. Each pruned directory gets a `PRUNED.json` recording what was
kept and when. `check_GLMs_ran.py` reads that marker so pruned GLMs are not
reported as `FILM_UNFINISHED` (two of its FILM completion markers are among the
deleted files).

## 2026-09-06 — why the HC current-location effect disappeared: `l2_norm`

**Symptom.** In the joint RSA GLM the `location` regressor used to be
significant (uncorrected) in both hippocampal ROIs even with DSR in the model;
in the 2026-08-27 and 2026-08-31 runs it was not.

**Cause: `l2_norm` was added to the shared control stack, not anything about
the data.** Combos across runs (all `split_halves_z`, all 1000 perms,
`shift_and_swap`):

All p below are the one-sided permutation p, UNCORRECTED.

| run | combo | l2_norm in stack | HC ant `location` | HC mid `location` |
|-----|-------|------------------|-------------------|-------------------|
| 2026-07-30 | `ctrl_dsrFULL` | no | b=.0403, p=.009 | b=.0330, p=.025 |
| 2026-08-27/31 | `ctrl_fMRI-state_dsrFULL` | yes | b=.0330, p=.052 | b=.0153, p=.215 |
| 2026-08-31 | `ctrl_fMRI-state_dsrFULL-noL2` | no | b=.0371, p=.012 | b=.0341, p=.021 |

`l2_norm` (graded negative distance from the current location to each of the 9
grid nodes) and the categorical `location` RDM correlate r = 0.588 — two
parameterisations of the same variable. Fitting both splits the spatial
variance: in `ctrl_fMRI-state_dsrFULL` the HC mid spatial signal moves onto
`l2_norm` (b=.0347, p=.057) and off `location` (p=.215).

**Not the cause.** (a) The ROI-table rebuild (`alt_final_roi`, 2026-08-27:
mOFC 85->74, HC ant 162->171, HC mid 143->145) — comparing the two no-L2 combos
across the tables gives HC ant .0403 -> .0371 and HC mid .0330 -> .0341, i.e.
nothing. (b) `bttn_next`, also added in August: it is present in the `-noL2`
combo, which restores the uncorrected effect.

**Consequence — and what may and may not be claimed.** Report
`ctrl_fMRI-state_dsrFULL-noL2` for any location claim. Dropping `l2_norm` does
not touch the DSR conclusions: mPFC dsr_fmri b=.0445 -> .0441 (p=.011 either
way), HC mid b=.0828 -> .0836 (p=.001 either way).

But `-noL2` does NOT make location significant in the joint model. Per-combo
BH-FDR across the 5 ROIs puts both HC effects at **q = .052**, which is why the
`location` column of
`pub_figures_v2/heatmap_roi_x_regressor_FDR_ctrl_fMRI-state_dsrFULL-noL2_split_halves_z.pdf`
carries no stars. The correct wording is therefore: current location is
significant in HC when the model is fitted ON ITS OWN (single-model RSA, see
the next entry: q_FDR = .0025 for both HC ROIs), and a TREND when it has to
compete with DSR and the other controls (HC ant b=.037, t=2.26, p=.012; HC mid
b=.034, t=2.09, p=.021; q=.052 for both). Do not call the joint-model effect
significant.

## 2026-09-06 — Fig 2g: single-model current-location RSA (no DSR control)

New `scripts/RSA_DSR_location_only_figure.py`. Pure plotting of the
`location`-alone rows of `results_summary.csv` (run 2026-08-31_17-57-30,
`split_halves_z`); nothing refitted. Only BH-FDR across the 5 ROIs shown is
computed here.

| ROI | n cells | t | beta | p_perm | q_FDR |
|-----|---------|---|------|--------|-------|
| mPFC | 65 | 0.25 | .0037 | .410 | .410 |
| mOFC | 74 | 0.61 | .0091 | .293 | .366 |
| PCC | 51 | 1.28 | .0190 | .127 | .211 |
| HC ant | 171 | 3.19 | .0472 | .001 | .0025 |
| HC mid | 145 | 4.35 | .0643 | .001 | .0025 |

Middle and anterior hippocampus are the only ROIs that fit the current-location
population geometry. Outputs (heatmap, sagittal + lyrz glass brains, combined
panel, values CSV, config) in the run's `location_only_figure_2026-09-06/`.
`mc.plotting.cell_results.plot_roi_beta_glassbrain` gained optional
`figure`/`axes`/`draw_colorbar`/`draw_footer` args so the brain can be drawn
into a shared panel, and `title=''` now means "no title".

## 2026-09-06 — `CELL_SET` now works on the reload path; non-RSA cohort re-run

**The gap.** `spatial_peaks_simple.CELL_SET` ('rsa' | 'not_in_rsa' |
'all_in_roi_table') was only read on the FULL-COMPUTE path — line 1383's
`cs.load_cells(cell_set=CELL_SET, ...)`. Under `RELOAD_FROM` the flag was
silently ignored, so re-testing a cohort from cached results was impossible
without recomputing 1,000 permutations per cell. `per_lag_encoding.py` had no
cohort flag at all: `_load_cells` hardcoded `cell_set='all_in_roi_table'`.

**The fix.** `CELL_SET` added to `per_lag_encoding.py` with the same three
values, and both scripts now apply it on BOTH paths. On reload each subsets its
cached per-cell table via a new `_subset_cell_set`, which resolves the cohort
from `cs.load_rsa_subjects()` — the same session list `cs.load_cells` uses, so
the compute path and the reload path cannot drift apart. The cohort is recorded
in the run tag and in config/settings JSON. Cached CV r and cached per-cell
permutation p are reused; nothing is recomputed. Verified: the reload run
reproduces a standalone filtered re-run bit-for-bit across
`per_roi_stats.csv`, `per_roi_lagwise_by_unit.csv` and
`per_roi_predicted_window_by_unit.csv`.

**Why it was needed.** The manuscript claims the future-tuning effect replicates
in cells that could not enter the population-RSA pseudopopulation. That claim
carried stale numbers (n = 434, mPFC n = 90) from an older ROI labelling.

**Cohort (current labels, `neurons_with_ROI_labels.csv` / `alt_final_roi`).**
419 of 955 cells, 35 of 63 sessions — the sessions absent from
`all_sessions_dsrRSA_grouping_summary.json`, so overlap with the RSA sample is
exactly zero. Per ROI: mPFC 93 (18 sessions), HC_anterior 124 (26), HC_mid 88
(17), mOFC 68 (17), EC 32 (7), PCC 10 (4). **Supersedes "n = 434, mPFC n = 90".**
Runs:
`per_lag_encoding/2026-09-06_20-56-52_reload_from_2026-08-28_10-18-21_..._not_in_rsa`
and
`spatial_peaks_simple/2026-09-06_20-58-05_reload_from_2026-08-28_10-23-56_..._not_in_rsa`.

**mPFC — replicates under BOTH estimators.** Main (pooled-training) estimator,
a-priori 30+60 deg window: across cells vs zero t(92) = 2.62, p = .0051,
q = .015; vs the same cells' other ten lags t(92) = 2.78, p = .0033, q = .0099.
Across the 18 sessions the specificity contrast survives (t(17) = 2.17,
p = .022, q = .034) but the vs-zero test does not (t(17) = 1.42, p = .087,
q = .131) — expected with 18 rather than 33 sessions. Peak: 30 deg across cells
(r = .072, t(92) = 2.77, q over 12 lags = .041; sign-flip max-t FWE within ROI
p = .038), 60 deg across sessions (r = .070, t(17) = 1.54, p = .071).
Permutation null: 25/93 cells (26.9%) at 30 deg (binomial q = 1.9e-11), 17/93
(18.3%) at 60 deg (q = 2.0e-05).
Paired-grid-group estimator, same cohort: window vs zero t(92) = 3.03,
q = .0048; vs other lags t(92) = 2.81, q = .0090; subject level t(17) = 1.85,
p = .041 and t(17) = 1.90, p = .037 (q = .12 after FDR over 3 ROIs). Peak
30 deg across cells (r = .054, q over 12 lags = .030), 60 deg across sessions
(r = .085, t(17) = 2.24, p = .019). 13/93 cells beat their permutation null
(14.0%, binomial q = .0044).

**HC_mid — partly replicates.** Main estimator, subject level: window vs other
lags t(16) = 2.32, p = .017, q = .034; vs zero t(16) = 2.02, p = .030, q = .091.
0 deg holds (subject r = .093, t(16) = 1.97, p = .033; cell r = .067,
t(87) = 2.26, p = .013), but 330 deg does NOT carry over (subject r = -.003).
18/88 cells (20.5%) beat the null at 0 deg. Under the paired-grid estimator the
window is only a trend (cell vs zero t(87) = 1.77, q = .060).

**HC_anterior — does NOT replicate.** Window null at both units and under both
estimators (main, subject: vs zero t(25) = -0.31, vs other lags t(25) = -0.12);
the cohort's peak drifts to 90 deg (cell r = .036, t(123) = 1.65, FWE n.s.).
Per-cell permutation fractions stay above chance (16/124 at 0 deg, 21/124 at
330 deg) — a statement about spatial structure, not about the predicted window.

**Consequence for the manuscript.** "All ROI-level effects were further
reproduced" is too strong and must be narrowed: mPFC reproduces under both
estimators, HC_mid only its present-lag half and mainly on the specificity
contrast, HC_anterior not at all. The mPFC subject-level vs-zero test being at
trend with 18 rather than 33 sessions should be stated, not omitted.

## 2026-09-06 — Cluster-forming threshold in `get_subj_gradients.py` was inactive

**The bug.** `extract_clusters` took `CLUSTER_THRESHOLD = 90` and applied
`np.percentile(subj_data, 90)` over the WHOLE volume. The input maps are
masked: 4,179 non-zero voxels out of 902,629, i.e. 99.5% exact zeros. The
90th percentile of that volume is therefore exactly 0.0 — verified in 33/33
subjects and in every 4-condition dataset — so the branch reduced to
`subj_data > 0` and the nominal "top 10%" threshold never engaged. Anything
describing this analysis as a 90th-percentile threshold is wrong; it is a
threshold at zero.

Two smaller defects in the same function: threshold values in (0, 20] left
`binary` undefined (NameError), and the `"z"` branch z-scored across the full
volume including the 99.5% zeros rather than within the mask.

**What the analysis actually computes.** Per subject and condition: threshold
the beta map at zero within the ROI mask, label 6-connected components
(scipy default connectivity), keep the component with the largest summed beta,
take its beta-weighted centroid, convert to MNI, read the z-coordinate. All
weights are positive because the cluster is defined as > 0.

The connectivity step is not cosmetic. Positive voxels are 54.6% of the mask
on average (range 2-98%); the retained component holds a mean 93% of them but
as little as 16% (20/132 subject x condition cells below 90%, 5/132 below 50%),
and the resulting z can differ from an all-positive-voxel COM by up to 24 mm.
Dropping connectivity gives t(32) = 2.05, p = 0.048 instead of
t(32) = 2.75, p = 0.0098 — so "centre of mass of the positive effects" is NOT
an accurate description of this analysis.

The ROI mask support is byte-identical (4,179 voxels) across all 33 subjects
and all four quarters, so the Q0->Q3 shift cannot be a mask-geometry artifact.

**The fix.** `CLUSTER_THRESHOLD` replaced by `THRESHOLD_MODE`
("zero" | "percentile" | "z"), `THRESHOLD_VALUE`, and a new
`MIN_CLUSTER_VOXELS` extent filter; all thresholds now computed within the
mask via `cluster_forming_threshold()`. Exposed as CLI flags
(`--threshold-mode`, `--threshold-value`, `--min-cluster-voxels`, `--axis`,
`--n-clusters`, `--peak-modes`, `--out-dir`) and recorded in the run JSON/MD.
Defaults are mode="zero", extent=1, which reproduce the published numbers
exactly: mean z = 17.95, 19.50, 21.89, 22.67; t(32) = 2.748, p = 0.0098.
Verified identical to the stored 2026-08-03 run (t = 2.7484 / 2.7371).

**Threshold sweep (6 thresholds x 3 extents x 3 datasets), reported in full,
no cell selected.** Engaging a real threshold WEAKENS the test at every
setting: in-mask p50 t=2.27, p75 t=1.25, p90 t=1.42, p95 t=1.79, z>1 t=1.64,
against t=2.75 at zero. The slopes barely move (1.0-2.2 mm/step across the
whole grid vs 1.65 at zero) — it is the between-subject SD of the slopes that
inflates, from ~3.45 mm at zero to ~6.75 at p90. Thresholding shrinks the
surviving cluster to a handful of voxels and each subject centroid becomes a
noisier estimate; the effect is not disappearing, the precision is.

FAILED DIRECTION — do not repeat: choosing a threshold because it maximises t
is threshold-shopping, and the currently used setting happens to be the best
cell in the grid. The sweep is a robustness check, not a menu. On that reading
it is reassuring: the slope is positive in 18/18 cells for both real datasets,
while the rotated control flips sign across cells and never approaches
significance (18/18 cells p > 0.25, |t| <= 1.16). Direction is
threshold-independent and specific to the true quarter ordering; only power
depends on the threshold. Sweep saved to
`data/derivatives/group/..._cropped_masked/gradient_threshold_sweep_2026-09-06_13-46-28/`.

**Follow-up: does the gradient-mask PC1 axis fit better than MNI z?** No — it
fits worse. Reproduced the axis exactly as `cell_gradient_master_table.gradient_axis`
defines it (PC1 of the 2,765 `gradient_thr_1.5.nii.gz` voxels, x folded to |x|,
oriented +z): PC1 = [-0.038, -0.409, 0.912], matching the [-0.04, -0.41, 0.91]
in the methods text. Projecting the same subject-wise cluster COMs onto it
(quarters_button_state, cluster_com, defaults):

  MNI z   slope=1.652 mm/step  SD=3.45  t(32)=2.748  p=0.0098
  PC1     slope=1.535 mm/step  SD=3.65  t(32)=2.417  p=0.0215
  MNI y   slope=-0.089         SD=2.71  t(32)=-0.188 p=0.852
  MNI |x| slope=0.200          SD=0.74  t(32)=1.558  p=0.129

The loss decomposes exactly: 0.912*1.652 + (-0.409)*(-0.089) + (-0.038)*0.200
= 1.535. PC1's posterior tilt mixes in the y-axis, which carries no trend at
all, so it dilutes the z signal (slope down 7%) while adding between-subject
variance (SD up 6%). Rotated control stays null on both axes (z p=0.431,
PC1 p=0.316).

Also: PC1 is derived from the group t-maps of these same four quarters in these
same 33 subjects, so projecting their COMs onto it re-uses the data the trend
is being tested on. MNI z is data-independent. z is therefore both the better-
performing and the cleaner choice for the fMRI trend. PC1 remains the right
axis for the CELL projection, where the electrophysiology is an independent
dataset.

Number to correct in the methods text: "r = 0.98 with MNI z" is the correlation
across all 158 mPFC cells, most of which lie OUTSIDE the gradient mask. Among
the 87 cells that actually enter the analysis it is r = 0.789; across the mask
voxels themselves it is r = 0.967. Quoting 0.98 overstates how interchangeable
the two axes are for the cells the analysis uses.

**Follow-up 2: the two axes are NOT interchangeable for the cell median split.**
r = 0.967 is the PC1-vs-z correlation across the 2,765 mask VOXELS (mask
geometry, cell-independent). r = 0.789 is across the 87 in-mask CELLS; Spearman
is 0.675. The drop is range restriction plus clumping: the 87 cells occupy only
19 distinct coordinates (cells on one microwire bundle share a location) and
span just SD = 2.19 mm along either axis, against a mask that is tens of mm long.

Consequence for the median split: 5 of the 19 sites change side when the axis is
swapped, carrying 33 of 87 cells (38%). Group sizes barely move (48/39 on PC1 vs
49/38 on z) but the membership does.

  median fMRI preferred angle:  PC1  ventral 36.3 deg | dorsal 83.5 deg (diff 47.2)
                                z    ventral 27.9 deg | dorsal 93.5 deg (diff 65.6)
  pooled cell profile argmax:   PC1  ventral 30 deg   | dorsal 60 deg
                                z    ventral 30 deg   | dorsal 30 deg

The fMRI-angle contrast survives and is LARGER on z. The pooled cell tuning
result does not: the ventral/dorsal peak difference exists on PC1 and vanishes
on z. It is a one-bin shift on a 30-deg grid either way -- the PC1 dorsal peak
(60 deg) leads its runner-up (30 deg) by only 0.032 in mean r, and under z the
two are 0.005 apart. The dorsal peak is a near-tie between adjacent bins and the
axis choice tips it. PC1 was chosen on stated a priori grounds ("more precise"),
independently of this outcome, but the cell-side result is load-bearing on that
choice and should be reported as such.

**Unrelated issue spotted in the 8-condition set.** The "now" condition
(`LOCATION-...-mask_reward-path_beta_std.nii.gz`) has 146,417 non-zero voxels
against 4,179 for the other seven, i.e. a different and much larger support.
Its COM is therefore not comparable to the mPFC-ROI COMs it is plotted beside,
which affects the eighths/circular analysis. The stored 2026-08-03 run had
pruned this file as missing and ran 7 conditions; it is present now, so the
eighths result changed (t=0.99 -> t=1.67). Not yet addressed.

## 2026-09-06 — State on a 0-1 scale, and panel g2/g3 show only the regressed cells

**State scale.** "Different state" rendered lighter than "different location"
because the two models were on different scales. ``my_RSA.compute_crosscorr``
demeans each row before the cosine, so on a 4-element one-hot two different
states come out at 1 - (-1/3) = 1.333 while the Hamming models top out at 1.0;
on a shared colour scale 1.0 then sits at ~75% of the range. The figure now
scores the state regressor by Hamming on its A-D LABEL (`_state_label_EV`,
`STATE_AS_HAMMING`), putting all three models on 0..1. This is the same binary
same/different matrix, only rescaled, and the RSA GLM z-scores every regressor,
so it cannot change a result — but it is a rescaling, not the pipeline's own
numbers, and the flag documents that. All three models now reproduce their
all-tasks block exactly (previously position-in-sequence did not).

**Regressed cells.** ``used_cell_mask`` reproduces the two restrictions the RSA
config applies (``diagonal_included: false``, ``masked_conds: true``): only the
strict upper triangle is regressed, and only path-path / reward-reward cells
(``make_category_masks(..., mask_only_path_rew_combos=True)`` keeps the
``same``-type cells). Panels g2 and g3 now blank everything else, and g2 gains
a leading "cells regressed" panel showing the mask itself — 12 of 28
upper-triangle cells per 8-bin task block.

Consequence worth knowing: within ONE task block every surviving cell is a
different-state comparison, because the four path bins are states A-D and so
are the four reward bins. The position-in-sequence regressor is therefore
CONSTANT within a task block under the mask and only varies across
configurations (the blue diagonal stripes in g3). That is the same reason
`fMRI_run_RSA_without_rsatoolbox_clean.py` drops `path_rew` and `A-state` from
the path-path and reward-reward subsets.

Output folder rolled over with the date: ``method_schematic_06-09-2026/``.
``method_schematic_05-09-2026/`` holds the previous, unmasked run.

## 2026-09-05 (later) — Panel g split into within-half / across-half / all-tasks

Reverted the example subject to **sub-02** (the previous route: 6-5-8-9-8-7-4-1-2-3),
whose model RDMs carry a range of values rather than only 0 and 1. Its two task
halves solved 5-9-4-3 by different routes at B_path — instead of avoiding that,
panel g now uses it as the teaching point:

- **g1  within one run** (`within_half_rdms`) — bin i vs bin j of task half 1.
  This is the arithmetic panel f counts out, and it is only didactic: the RSA
  never compares a run with itself.
- **g2  across the two runs** (sliced from the all-tasks RDMs) — rows = task
  half 1, columns = task half 2, so no cell can be inflated by shared noise.
  Above it, `draw_route_strips` shows both halves' routes with the differing
  bin framed in red, making visible why the B_path diagonal is 1.00 here while
  it is 0.00 in g1, and why new off-diagonals appear (A_path/B_path 0.50,
  B_path/C_path 0.75).
- **g3  all task configurations** — unchanged, and it now fixes the colour
  limits (`_rdm_limits`, 2nd–98th percentile of the all-tasks matrix) that
  every other RDM panel and panel f reuse, so one dissimilarity has one colour
  throughout the figure.

g2 is sliced from the all-tasks matrices rather than recomputed, so it is that
block by construction whatever metric the pipeline scored a model with.
`across_half_rdms` reproduces it for the two Hamming models (verified at build
time) and symmetrises as `(M + M.T)/2` exactly as `compute_hamming_distance`
does; it differs for position-in-sequence only because the RSA scores that one
with `compute_crosscorr`.

**Panel f overlap bars now use the RDM colormap** (`rdm_colour`): an element
that matches contributes 0 to the Hamming distance and gets the RDM's
"similar" colour, one that differs contributes 1 and gets its "dissimilar"
colour, both under the g3 limits. Each bar now ends in a single square — the
one RDM cell that comparison produces — and the numbers read
"6/96 match = 6.2% → d 0.94", tying panel f directly to panel g.

Numbers are back to the sub-02 values: comparison 1 (A_reward vs B_reward)
concurrent 6/96 = 6.2%, location d 1.00; comparison 2 (B_path vs C_path)
concurrent 6/96 = 6.2%, location d 0.50.

## 2026-09-05 — Panel g single-task RDM now IS the block of the all-tasks RDM

The within-task RDM in panel g did not match the corresponding diagonal block
of the across-tasks RDM. Cause: the two matrices were computed differently.
``single_task_rdms`` compared task half 1 against itself, whereas the RSA (and
therefore the all-tasks matrix) compares task half 1 against task half 2. For
sub-02 the subject walked a different route through B_path in the second half,
so the block had a non-zero diagonal at B_path (1.00) and off-diagonals of
0.50/0.75 where the within-task version had 0 and 0.50.

Two changes, both needed:

1. ``single_task_rdms_from_across`` slices the example task's diagonal block
   straight out of the across-task RDMs, and panel g1 now renders that block
   with the same colour limits as g2 (``_rdm_row_figure`` returns the limits it
   used; g2 is drawn first). The two rows of panel g can no longer drift apart
   whatever the data do. ``across_task_rdms_*`` now also carry ``task_keys``
   and ``block_labels`` so the block can be located.
2. ``_pick_fmri_subject`` chooses the example subject rather than hard-coding
   sub-02: it requires the two task halves to have walked the SAME modal route
   through the example configuration (17 of 33 subjects qualify for 5-9-4-3),
   then prefers the route visiting the most distinct grid locations, no
   non-adjacent steps, the most commonly walked route, lowest subject number.
   This lands on **sub-01** (9 distinct locations, 0 non-adjacent steps, route
   shared by 9 subjects). ``FMRI_SUB`` overrides it.

Verified at build time: the sliced block reproduces the within-task matrix
exactly for the concurrent and current-location models. It differs for
position-in-sequence only because the RSA scores that model with
``compute_crosscorr`` while the within-task helper used a 0/1 categorical —
the figure shows the RSA's own values.

Numbers changed with the subject. Route 3→6→5(A)→6→9(B)→8→7→4(C)→1→2→3.
Panel f now reads 12/96 = 12.5% concurrent for both comparisons, with current
location 0% in both — sub-02's 50% at B_path/C_path was specific to its route.
The informative location cell for sub-01 sits at A_path/B_path instead (both
pass through location 6, dissimilarity 0), visible as the off-diagonal blue in
the current-location RDM. Swap ``FMRI_PAIRS`` to ``((1, 3), (0, 2))`` to put
that pair in panel f.

Output folder: ``data/derivatives/group/method_schematic_05-09-2026/``.

## 2026-09-04 — Methods schematic panel f reworked

Panel f of the RSA methods figure now shows TWO within-task comparisons
instead of three mixed ones — one per comparison type the RSA actually uses,
since across-phase (path–reward) cells are excluded from the regression:

- 1  reward–reward: 5-9-4-3 A_reward vs B_reward — concurrent 6/96 = 6.2%,
     current location 0%, position in sequence 0%
- 2  path–path: 5-9-4-3 B_path vs C_path — concurrent 6/96 = 6.2%,
     current location 50% (both bins pass through location 8),
     position in sequence 0%

Both are the same plan rolled by two bins, so the concurrent overlap is
identical; the two rows differ in what the unfolding code says. The two
across-task comparisons (same place now / same future lags) are dropped from
the default but still reachable via ``build_pairs(..., cross_task=True)`` and
``pick_pair``.

The panel now makes the two time axes explicit, which was the point of the
rework: the rainbow future-lag bar with its degree labels sits on top of EVERY
comparison (across a row = how far into the future), while each row is one
encoded time bin. Every strip carries its own A–D annotation underneath —
"at A" / "→B" per block, with the reward blocks framed in that state's colour —
so it is visible that the same lag position holds a different part of the task
depending on which bin the code is read out from. The four cells of the
position-in-sequence block are labelled A–D inside the boxes.

Two helpers added: ``_is_reward_bin`` (a bin where the subject is AT the
reward: ``*_reward`` for fMRI, ``*_early`` for sEEG) and ``_ink``, which
darkens the pale state colours (C = #C7C6E2) until they are legible as text.
Block labels fall back to one letter per state RUN when there is less than
0.70 cm per block (the 12-bin sEEG case); the runs are found from the labels
rather than assumed, so they stay correct whatever bin the strip starts at.

## 2026-09-04 — Methods schematic for the gradient (harmonic angle) analysis

New `mc/plotting/gradient_schematic.py` + `scripts/build_gradient_schematic_figure.py`
explain `scripts/harmonic_angle_maps.py` in seven panels: the concurrent code
cut into four quarters that enter one searchlight GLM (a); the four β's every
voxel therefore has (b); the angle assigned to each quarter — the bin CENTRE,
45/135/225/315° — and the cos/sin projection (c); those two components as one
vector whose angle is the preferred future step and whose length is the effect
size (d); the per-subject vectors, their group mean and the Hotelling T² test
(e); the same subjects on the unit circle with R̄ and Rayleigh (f, the
`USE_UNIT_VECTOR_MAPS` branch); and the resulting preferred-angle map (g).
Same cm-based layout, 9/11 pt fonts and cyclic rainbow as
`method_schematic.py`. Output: `data/derivatives/group/gradient_schematic_<date>/`.

Everything plotted is real. β profiles, per-subject (cos, sin), Hotelling and
Rayleigh statistics and the angle map are read from the analysis outputs; the
sagittal slice is the one carrying the most suprathreshold mPFC voxels
(x = +10 mm, 39 voxels). The two example voxels are picked by a fixed rule —
among the 480 mPFC voxels with Hotelling p < 0.05, the ventral and dorsal
quintiles along MNI z, and within each the largest amplitude:

- ventral MNI [2, 36, −8]: β = [0.0132, 0.0074, −0.0063, −0.0060],
  angle 79.4°, amplitude 0.0236, Hotelling F = 6.90 p = 0.0033,
  Rayleigh R̄ = 0.31 p = 0.75
- dorsal  MNI [12, 46, +14]: β = [0.0080, 0.0124, −0.0020, −0.0054],
  angle 105.7°, amplitude 0.0204, Hotelling F = 4.30 p = 0.0225,
  Rayleigh R̄ = 0.32 p = 0.71

Both voxels are Hotelling-significant but Rayleigh-null, i.e. a reliable
group-mean vector without per-subject angle agreement. Panel f states this
outcome rather than implying the voxel passed both tests.

**Colour-lookup bug caught while building panel g.** Rendering the angle map
with `vmin=-180, vmax=180` on the cyclic colormap puts 0° in the MIDDLE of the
map (blue) instead of at yellow; the mPFC cluster came out green/teal rather
than red/purple. The angle is now wrapped into [0, 360) before lookup. Only
affects this figure — `harmonic_angle_maps.py` writes signed degrees and the
fsleyes recipes in its README use `hsv`, which has the same wrap issue if the
display range is set to −180..180; worth checking there too.

## 2026-09-04 — Methods schematic: fixed physical sizes, rainbow future scale

`mc/plotting/method_schematic.py` now lays every panel out in absolute cm
(`_add_ax_cm`, `_cm_canvas`, and cm-valued data coordinates for the code
strips) and saves WITHOUT `bbox_inches='tight'`, so the plotted boxes come out
at exactly the declared size on an A4 page and the type stays at its nominal
point size. Sizes: RDM 4x4 cm each, panel c block 4x4 cm, and the code row
7 cm (concurrent) + 2 cm (current location) + 3 cm (position in sequence).
Panels d and e are now one row sharing those three columns, matching panel f.
Fonts: 11 pt headings, 9 pt everywhere else — `FONT_TICK` was 8 pt and is now
9 pt, the A4 floor. `_thin` blanks tick labels that would sit closer than
0.32 cm, and block labels collapse to one letter per state when a loop has
more than one bin per state (the 12-bin sEEG case).

Future lag / task angle now uses the cyclic rainbow of the mPFC gradient
figures (`FUTURE_CMAP`, `lag_colors`): 0 deg yellow, 90 red, 180 blue,
270 green, wrapping back to yellow — replacing the yellow-to-brown ramp.

Panel a places each state letter on the wedge where the subject is AT that
reward (derived from the bin labels) instead of at a fixed 90 deg spacing.

Numbers, comparisons and the data pipeline are unchanged.

## 2026-09-04 — Methods schematic: unfolding vs concurrent code

New `mc/plotting/method_schematic.py` + `scripts/build_method_schematic_figures.py`
build the figure that explains the two competing model geometries (panels a-g:
bins as future angles; example configuration and executed trajectory; the
per-bin encodings; the concurrent code read out from two bins; the unfolding
code; similarity by counting overlapping elements; the resulting RDMs for the
single task and across all tasks). Output goes to
`data/derivatives/group/method_schematic_<date>/` — one assembled overview plus
every panel as its own PDF/PNG, with a settings JSON per figure.

Two example tasks, each in the framework it belongs to:
`5-9-4-3` (fMRI, sub-02 E1_forw, 8 bins x 12 resampled steps -> 96-element DSR,
mirroring `create_fMRI_model_RDMs_on_clean_beh.py`) and `3-7-9-5` (sEEG, modal
path over 870 correct trials, 12 bins x 1 location, mirroring
`RSA_DSR_ROIs_simple.py`). Across-task RDMs come from the real pipeline:
`my_RSA.build_across_halves_model_RDM` on the saved EV pickle for fMRI,
`compute_hamming_distance` over the 8 configurations for sEEG.

**Alignment bug found in the sEEG schematics.** The raw 360-bin sEEG loops for
`3-7-9-5` are anchored on reward D, not reward A: the modal 12-bin trajectory
reads 5-6-3-3-1-7-7-8-9-9-6-5, so labelling bin 0 as 'A' put the wrong
locations under A-D — the mismatch visible in the earlier Fig-1e simulation
panel. `align_to_state_A` now picks the rotation that lines the four
state-onset bins up with the four rewarded locations (here: shift 2, 4/4 onsets
matched) and the shift is logged. The fMRI side needs no rotation: its bins are
labelled from behaviour.

Also logged (`non_adjacent_steps`): downsampling the sEEG loop to 12 bins drops
intermediate locations, so `3-7-9-5` shows two non-adjacent grid steps
(3->1, 1->7). Reported rather than smoothed. The fMRI trajectory
6-5-8-9-8-7-4-1-2-3 is fully adjacent.

Comparisons selected automatically for panel f (fMRI example, Hamming
similarity of the 96-element codes):
- same task, A_reward vs B_reward: concurrent 6.2%, location 0%, position 0%
- 5-9-4-3 A_path vs 1-7-5-3 D_path — same place now: concurrent 12.5%,
  location 100%, position 0%
- 5-9-4-3 B_reward vs 3-5-7-1 C_reward — different place now, same locations at
  the same future lags: concurrent 56.2%, location 0%, position 0%
The last two are the dissociation the RSA rests on and are picked by
`pick_pair`, not hand-chosen.

## 2026-09-03 — One consolidated lag-wise results table

`scripts/overlay_double_dissociation.py` now writes ONE lag-wise table,
`overlay_lagwise_tests.csv` (144 rows), replacing `overlay_per_lag_table.csv`
and `overlay_subject_clustered_lagwise_ttests.csv` (both deleted on re-run,
along with the older `WEIGHTING_RESULTS.md`). `_subject_clustered_lagwise_tests`
and the inline per-lag block in `make_overlay` are gone, replaced by
`_lagwise_tests`.

Grid: unit_of_analysis (cell, subject) x metric (raw_r, fisher_z) x 3 stats
ROIs x 12 lags. Columns: `p_one_sided`, `q_one_sided_across_12_lags`,
`q_one_sided_across_rois`, `q_one_sided_across_rois_and_lags`, plus
`mean_tested_value` (what the t ran on), `mean_raw_r`, `mean_fisher_z`,
`r_from_fisher_z`, `t`, `df`, n's. One-sided only, as requested. Every BH
correction runs INSIDE one unit x metric block -- the four blocks are versions
of the same test, not additional tests. Fisher-z numbers reproduce the two old
tables exactly. The peak-lag permutation now also covers the 2 x 2 grid.

**Raw r is uniformly slightly stronger than Fisher z** (cell mPFC 30 deg
t = 3.09 vs 3.02; subject mPFC 60 deg t = 2.22 vs 2.16), as expected since the
z transform inflates the variance of extreme cell r's. The difference never
changes a conclusion except at one knife-edge, below.

**What survives, by correction scope (one-sided):**
- across 12 lags: ONLY cell mPFC 30 deg (q = .014 raw_r / .018 fisher_z).
  Nothing at subject level, nothing in HC.
- across 3 ROIs at one lag: cell mPFC 30 deg (.0035/.0044), cell HC_mid 0 deg
  (.016/.014), cell HC_anterior 330 deg (.034/.037), cell HC_mid 330 deg
  (.039/.049), subject HC_mid 0 deg (.038/.039).
- across ROIs AND lags (36 tests): cell mPFC 30 deg is q = .042 with raw r but
  q = .053 with Fisher z -- the single place the metric flips a threshold.
  Nothing else survives. Flagged so the choice is never made by outcome; the
  pre-registered metric (Fisher z) stands.


## 2026-09-03 (instruction-phase RSA) — cumulative reward-prefix models + per-combo RDM scope

`scripts/fMRI_run_RSA_instruction.py` gained a second family of models sliced
out of `rewDSR` at the `A_reward` anchor, and combo models can now declare the
task-half scope they are fitted in.

**Cumulative "first k rewards" models.** The `rewDSR` A_reward vector is four
equal 12-element chunks, one per reward step (A, B, C, D), each holding the raw
location value repeated. Alongside the existing per-step SPLIT channels
(`curr_rew` = A, `next_rew` = B, `two_next_rew` = C, `three_next_rew` = D) the
script now also builds the CUMULATIVE prefixes:

| model | chunks | rewards | width |
|---|---|---|---|
| `A_rew` | 0 | A | 12 |
| `AB_rew` | 0–1 | A, B | 24 |
| `ABC_rew` | 0–2 | A, B, C | 36 |
| `ABCD_rew` | 0–3 | A, B, C, D | 48 |

**Models renamed** so the name states which rewards the model knows about. The
per-step SPLIT channels became `A_rew` / `B_rew` / `C_rew` / `D_rew` and the
cumulative ones `A_rew` / `AB_rew` / `ABC_rew` / `ABCD_rew`; `ABCD_rew` is the
whole rewDSR vector, so `rewDSR` and `ABCD_rew` are numerically identical.
The pre-rename names are **rejected with an error**, not silently accepted:
`curr_rew`, `next_rew`, `two_next_rew` and `three_next_rew` are *also* real keys
in `model_EVs` holding the 9-dim ONE-HOT version of the same idea, so an
un-updated config would have fallen through to the standard path and quietly
built a different model (mismatch 2/9 instead of 1) under an unchanged name.
See `LEGACY_MODEL_NAMES` / `check_no_legacy_names`.
`condition_files/rsa_instruction_full.json` was updated to the new names so it
keeps building exactly the models it built before.

Because the chunks are equally long, Hamming on a cumulative model is *exactly*
the mean of the per-step Hammings it contains — verified numerically on sub-02
(max abs difference 1.1e-16). So the family is strictly nested and `rewDSR` **is**
the k = 4 member; no separate model had to be defined for it. Each has an
`_instr` counterpart built through `instruction_relabel_dict` as before, and the
2×2 sub-block uniformity check passes for `two_rew_instr` / `three_rew_instr`.

**Per-model scope.** A combo entry may carry `"scope"` (a string or a list),
and single models are governed by a config-level `single_model_scopes`,
e.g. `{"execution": ["within_only", "across_only"], "instruction":
["within_only"]}` — applied by name, anything ending in `_instr` counts as
instruction. Both override `data_rdm_scope`; the resulting output maps are
suffixed `_within` / `_across` / `_full`. The asymmetry is forced by the design:
an execution model has variance in both blocks, an instruction model is constant
across halves. All scopes are now column masks over the strict
lower triangle of the *same* cached (2n × 2n) data RDM — `within_only` keeps the
two within-half blocks (90 cells for n = 10), `across_only` their complement
(100 cells, exactly the across block), `full_no_diag` all 190 — so fitting one
combo in several scopes costs no extra searchlight computation. The legacy
top-level `'across_only'` mode still uses its own (n × n) cache and refuses to be
mixed with per-combo scopes.

**New config** `condition_files/rsa_instruction_cumulative_rew.json`
(`data_rdm_scope: "within_only"`, since every instruction model is degenerate
across halves). Single models: all 14, each execution one fitted within AND
across halves, each `_instr` one within only. Combos: `first_exe_vs_instr`, `two_exe_vs_instr`,
`three_exe_vs_instr`, `four_exe_vs_instr` (each = execution vs instruction
variant of the same k), `instr_split` (the four instruction step channels), and
`exe_split` (the four execution step channels) fitted **both** within and across
task halves.

All designs are full rank (sub-02, TR4). Execution-vs-instruction shared
variance within half: `A_rew` r = +0.15, `AB_rew` r = +0.16,
`ABC_rew` r = +0.26, `ABCD_rew` r = +0.23. Regressors of `exe_split` are far
more collinear across halves (r = +0.46 to +0.73) than within (r = −0.10 to
+0.45), which is worth keeping in mind when comparing the two `exe_split` fits.

**Reporting.** Model RDMs are now plotted once per scope each model is actually
fitted in (cells outside the scope NaN/white), instead of always showing the
across block; the same holds for the example data-RDM panel. Each figure is
accompanied by a printed line giving the number of fitted cells, distinct
values, range and sd, so a degenerate regressor is visible without opening the
png. Regressor correlations — per combo, per scope — and the
execution-vs-instruction correlations are now written into
`{sub}_settings_summary.json` (`combo_regressor_correlations`,
`exec_vs_instr_correlations`) rather than only printed to stdout.

Also removed the dead `_needs_split_exec` / `_needs_split_instr` expressions that
had already been superseded by the explicit `_base_of` check.

## 2026-09-03 (new instruction-period first-level GLMs) — 11 epoch-wise instruction GLMs

`scripts/create_EVs_instruction_period.py` + `condition_files/EV_config_instruction.json`.
Same shape as the existing `01-TR{k}` per-TR instruction GLMs, but the epochs
are defined by what is on screen rather than by TR. 11 GLMs per task half, each
with 10 EVs (one per task) + the button nuisance:

| GLM | window | dur |
|---|---|---|
| `instr_see-{A,B,C,D}-first` | 0–1.5, 1.5–3, 3–4.5, 4.5–6 | 1.5 s |
| `instr_see-{A,B,C,D}-second` | 6–7, 7–8, 8–9, 9–10 | 1.0 s |
| `instr_collapsed-first-instruction` | 0–6 | 6 s |
| `instr_collapsed-second-instruction` | 6–10 | 4 s |
| `instr_empty-screen` | 10–12 | 2 s |

**Timings read off the experiment code**, not assumed —
`mc/latest_experiment/3x3_fMRI_part1.py` lines 697–712 (opacity schedule) and
744–915 (component stop times). The routine is non-slip timed to 12 s. Note the
second pass is **1.0 s per coin starting at 6 s**, not 1.5 s from 4.5 s, and all
coins plus the "backwards" warning stop being drawn at 10 s — only
`sand_pirate` stays to 12 s, which is what makes 10–12 s an empty screen.

Each GLM is named after what it measures rather than by an index, so the output
is `EVs_instr_see-A-first_pt01/` and `sub-02_draft_GLM_01_instr_see-A-first.fsf`.
EVs are named `{task}_{direction}_instruction_onset`, which is what
`pair_correct_tasks` in `scripts/fMRI_run_RSA_instruction.py` expects.
`load_data_EVs_instr_TRwise` now takes `TR=None` to mean "regression_version is
already the full GLM name" (it still builds `{version}-TR{TR}` otherwise, so the
existing `01-TR{k}` configs are unaffected); the epoch configs set
`"regression_version": "instr_see-A-first"` and no TR.
Submit with `mc/fmri_analysis/subject_GLM_instruction_epochs.sh`, which is
`subject_GLM_RDM_conds.sh` plus a loop over the epoch names.

**Why one GLM per epoch and not one GLM with all epochs.** A joint design was
built first and rejected on the numbers, measured on FEAT's own design matrix
(`feat_model`, sub-02 pt1, gamma HRF δ=6/σ=3, 100 s highpass, TR 1.078):

| design | VIF over the epoch regressors |
|---|---|
| joint, 6 epochs/task (61 EVs) | 4 – 132 |
| joint, 9 epochs/task (91 EVs) | 22 – **2555** |
| **one GLM per epoch (11 EVs)** | **1.01 – 1.05**, max abs r 0.005 |

The epochs are 1–1.5 s, back-to-back, in a fixed order, never jittered, so in a
joint design adjacent second-pass coins correlate at r = 0.95 and each beta is a
mixture of its neighbours. Since the hypothesis under test is a gradient across
A→D, a design that mechanically blends the coin epochs could manufacture that
signature. Per-epoch GLMs cannot: each holds one regressor per task, ~150 s
apart. This does not fix precision — there is still one 1.5 s event per
condition per half — but it makes the noise unstructured rather than organised
by the design.

**Instruction onset.** `instruct_start = (first start_ABCD_screen of that task)
− 12 s`. Validated against the on-flip timestamps (`sand_pirate.started`) in the
`*_all.csv`: those sit on a different clock origin than `globalClock`, but the
interval to the first `start_ABCD_screen` is constant to **SD ≈ 8–12 ms (< 1
frame)** across the 10 tasks of a session. Replaces the legacy `01` approach,
which used `t_reward_afterwait + 3.5` for tasks 2–10.

**Button nuisance regressor: kept, but it does nothing either way.** Measured
variance inflation on the instruction betas 1.0007 (max 1.004), max abs r with any
instruction EV 0.12. It does not fix the baseline — the instruction period is
6.7% of the run and there is no rest anywhere in the design, so the betas are
unavoidably "instruction vs. execution". For RSA that is a common offset across
conditions and cancels in correlation distance. Kept for comparability with the
`01`/`01-TR` GLMs; `"regress_buttons": false` switches it off.

**Runners.** `mc/fmri_analysis/subject_GLM_instruction_epochs.sh` submits the
FEAT jobs (subject x half x epoch) and
`mc/fmri_analysis/submit_RSA_instruction_epochs.sh` the RSA jobs, writing one
config snapshot per epoch with `regression_version` set and `TR` null.
`wrapper_python_fMRI_RSA_clean_config.sh` is unchanged — it only forwards
subject, config and script name.

**`mc/fmri_analysis/check_GLMs_ran.py`** verifies the FEAT runs before they are
trusted. Per (subject x half x GLM) it separates: no EV folder, no .feat, no
stats/, incomplete PEs (killed mid-FILM), FILM unfinished (all PEs but no
dof/smoothness/sigmasquareds/threshac1 — those are written last), task-to-EV.txt
pointing past the design, and a `+.feat` twin, which matters because FEAT never
overwrites an existing output dir: it appends '+' and leaves the stale original
in place, and the stale one is what the RSA reads. `--check-data` additionally
loads every PE the RSA uses and flags all-zero or non-finite maps. Prints a
per-GLM resubmit list and exits 1 when anything failed. Expected GLM names come
from the same EV config the EV script uses, so the two cannot drift apart.

**Caveat — sub-10 pt1.** Its last task's instruction (B1_backw) ends at 1878 s.
If that run is the nominal 1670 vols x 1.078 s = 1800 s, `feat_model` does not
warn — it **aborts** with "No valid [onset duration strength] triplets found"
and builds no design at all. Confirmed locally. The script prints this as an
upfront warning. Check the real dim4 for sub-10 pt1 before running; the old `01`
instruction GLMs have the same exposure.

**Figures.** `plot_instruction_RDM` and `plot_model_correlations` in
`mc/analyse/my_RSA.py` now draw 4 x 4 cm panels in Arial with a 9 pt floor, and
save .pdf next to .png. Specifics: correlation-matrix annotations went from 4 pt
(unreadable at print size) to 9 pt, and that panel grows past 4 cm when a matrix
has too many models for a 9 pt number to fit in a cell — the font floor wins
over the target footprint; `annot=False` keeps 4 cm. RDM tick labels fall back
to per-half block labels when the matrix is an assembled (2n x 2n) one, and
otherwise to every kth real label — labelling both axes TH1/TH2 on the (n x n)
across block, where rows are TH1 and columns are TH2, would have been wrong.
Colorbars carry only their two end ticks, since the default '0.0/0.5/1.0' set is
wider than a 4 cm panel leaves and was being clipped.


## 2026-09-02 (per-TR instruction RSA) — within-half-only run: the offset shrank, flipped sign, and still drove the stats

**Inputs.** `group_RSA_within_th_only_intr-vs-exe_glmbase_01-TR{0..11}_cropped`,
24 maps, 32 subjects, brain mask 144 404 voxels. Analysed with
`scripts/per_TR_loso.py` (statistics in `mc/analyse/loso.py`), 10 000 sign-flip
permutations for the SVC/LOSO, 1 000 for whole brain, masks mPFC + MTL.
Results in `data/derivatives/group/within_th_only_intr-vs-exe_allTRs_2026-09-02`
(raw) and `..._allTRs_DEMEANED_2026-09-02` (demeaned; see below).

**Did dropping the across-half block remove the bias? Partly, and it flipped.**
Median whole-brain t, averaged over maps of each class:

| analysis | execution | instruction | exec - instr |
|---|---|---|---|
| full (across+within), 2026-08-28 | -0.584 | +1.236 | **-1.82** |
| within-half only | +0.171 | -0.450 | **+0.62** |
| within-half + per-subject demeaning | -0.003 | +0.006 | **-0.01** |

The gap shrank ~3x but **reversed sign** — an attenuated version of the same
artefact would have kept its sign.

**Measured cause: restricting to within-half cells RAISES the execution/
instruction regressor correlation.** Pearson r between each execution model and
its own instruction counterpart, sub-02, over the cells the OLS actually fits
(printed by `fMRI_run_RSA_instruction.py`, re-run locally with TR4 — the model
RDMs are built from behaviour, so the TR does not enter this number):

| pair | `full_no_diag` (90+90 cells) | `within_only` (90 cells) |
|---|---|---|
| curr_rew vs curr_rew_instr | +0.02 | **+0.146** |
| next_rew vs next_rew_instr | +0.11 | **+0.318** |
| two_next_rew vs two_next_rew_instr | +0.11 | **+0.318** |
| three_next_rew vs three_next_rew_instr | +0.02 | **+0.146** |
| rewDSR vs rewDSR_instr | (n/a) | **+0.231** |

So in the full scope the two regressor families were near-orthogonal (r = .02-.11)
and the old bias came from the block-structure offset, NOT from exec/instr
collinearity. Dropping the across-half block removed that block offset but
*tripled* the direct correlation between execution and instruction regressors.
In a joint OLS, positively correlated regressors give anti-correlated betas: if
execution carries the signal, the instruction beta is pushed negative to
compensate. That is exactly the observed instruction-negative /
execution-positive offset, and it explains the sign flip. Design-induced
collinearity, not a scanner artefact — but still an additive offset rather than
a regional effect.

(Consistent with the structural argument: of the 90 within-half cells, 10 are
same-task-letter pairs where instruction dissimilarity = 0 while execution
dissimilarity = 1.)

**The residual offset was producing the "significant" results.** Across the 24
maps, a map's global median t predicts its peak statistic almost perfectly:

    corr(global median t, mPFC negative peak t) = +0.918, p = 2.6e-10
    corr(global median t, MTL  negative peak t) = +0.698, p = 1.5e-04
    median t of the 7 maps WITH a significant negative peak: -0.832
    median t of the 17 maps WITHOUT:                         +0.073

The sign-flip null is centred on zero, so a map whose whole brain sits at -0.9
yields a "significant" negative peak somewhere regardless of any regional
effect. **The negative mPFC/HC hits in the raw run are therefore not findings**
(largest: `next_rew_instr` mPFC t = -6.84 p_FWE = .0006; MTL 28/-20/-18, 90 %
right hippocampus, t = -5.43 p = .016). The LOSO timecourses inherit it too
(r = +0.66 mPFC, +0.44 MTL), which invalidates `three_next_rew` MTL
(LOSO p = .017) — it has the largest positive offset of all 24 maps (+0.545).

**Fix: `--demean` (new flag).** Subtracts each subject's whole-brain mean,
separately per TR, from the data array before any statistic, so every sign-flip
permutation inherits it and there is no separate permutation path (rule 4). The
test becomes "is this voxel above this subject's own brain-wide level at this
TR" rather than "above zero". Note this was chosen POST HOC, after seeing the
offset; both runs are kept and reported side by side. Median t ~ 0 afterwards is
guaranteed by construction, not a result — the question is which peaks survive.
After demeaning, corr(offset, mPFC negative peak) collapses +0.918 -> -0.402.

**What changes.** Maps with a positive offset lose, maps with a negative offset
gain, and one result moves the other way — the signature of a real regional
effect (demeaning also removes subject-specific global fluctuation, cutting
between-subject variance):

| map | raw t / p_FWE | demeaned t / p_FWE |
|---|---|---|
| `CURR_REW-splitDSR_vs_instr` (execution, reward A) TR2, 22/-90/32 occipital pole | 5.90 / .110 | **6.86 / .032** |
| `curr_rew` (execution) TR2, 22/-92/34, same site | 5.22 / .300 | 6.32 / .097 |
| `CURR_REW_INSTR-instr_split` (instruction) TR3, 16/-70/26 cuneus/precuneus | 7.18 / **.007** | 6.19 / .103 |

**Surviving results after demeaning** (whole brain corrected over 144 404
voxels x 12 TRs; SVC corrected within each mask over voxels x TRs):

- Whole brain: `CURR_REW-splitDSR_vs_instr`, TR2, MNI 22/-90/32, t = 6.86,
  p_FWE = .032 (44 % occipital pole). Replicated by `curr_rew` at 22/-92/34
  (p = .097). Execution, not instruction.
- MTL SVC, POSITIVE and instruction-driven, at one voxel:
  `REWDSR_INSTR-rewDSR_vs_instr` TR5, MNI -28/-22/-32, t = 5.40, p_FWE = .021;
  `three_next_rew_instr` TR6, same voxel, t = 5.03, p_FWE = .049.
  Harvard-Oxford: 29 % left anterior parahippocampal gyrus, 12 % posterior
  temporal fusiform — parahippocampal/entorhinal, NOT hippocampus proper.
  **These two are NOT independent**: `three_next_rew` is literally one quarter
  of the `rewDSR` A_reward vector (`REWDSR_SPLIT_CHANNELS`), and their t-maps
  correlate r = +0.53. Worse, the result is fragile at threshold —
  `rewDSR_instr` and `REWDSR_INSTR-rewDSR_vs_instr` are near-identical maps
  (r = +0.973) peaking at the SAME voxel, yet t = 4.70 / p = .111 versus
  t = 5.40 / p = .021. A 0.7 change in t moves it across the threshold, so this
  voxel should be treated as a lead to test in an independent contrast, not as
  a result.
- MTL, right posterior parahippocampal 34/-28/-18: `NEXT_REW_INSTR-instr_split`
  TR11 SVC p = .092 with LOSO p = .029; `next_rew_instr` LOSO p = .031 at TR10.

**Multiple comparisons, stated plainly.** 24 maps x 2 masks = 48 SVC tests plus
24 whole-brain tests; each p_FWE corrects only over its own voxels x TRs. None
of the above is corrected across that family. At 48 tests, ~2.4 hits below .05
are expected by chance and we have 2 in MTL, which are not independent of each
other — so the MTL parahippocampal result is NOT established by this run.

**Figure.** `offset_diagnostic.pdf` / `.png` in the DEMEANED folder, produced by
the archived `make_offset_diagnostic.py` beside it.

**Two code fixes made while getting this to run** (both in `mc/analyse/loso.py`
+ `scripts/per_TR_loso.py`):
1. `check_inputs` — a pre-flight that compares each .nii.gz gzip trailer against
   the size its NIfTI header implies, confirming only the suspects by real
   decompression. A truncated download has a readable header and dies inside
   `get_fdata`, which killed a run partway; this fails in ~1 s instead. It found
   exactly the 24 files `gzip -t` found (all of TR0, an interrupted transfer,
   since re-downloaded). Bypass with `--skip-input-check`.
2. `peak_TR` was a position along the TR axis, not a TR number, so any run over
   a non-contiguous `--trs` subset mislabelled its peak (and the plot x-axis,
   labelled "instruction period (s)", assumed index = second). Real TR labels
   are now passed into `run_svc` / `run_loso` / `run_wholebrain`, stored in each
   summary JSON as `trs`, and used for both. Contiguous 0-11 runs are unaffected.

**Not repeated / dead ends.** Testing the negative direction against the same
sign-flip null is only valid when the map has no global offset; with one, it is
guaranteed to "find" something. Do not report negative-direction peaks from a
map whose global median t is far from 0 without demeaning first.

## 2026-09-02 (later still) — Shortest-path metric in `behaviour_summary.py`

Added one behavioural metric to `scripts/behaviour_summary.py`: the percentage
of walks between two consecutive rewards that took the minimum possible number
of steps. The 3x3 grid is 4-connected (no diagonals, no wrap-around), so the
minimum is the Manhattan distance between the two rewards. Four walks per
repeat: D->A, A->B, B->C, C->D. New outputs `fmri_shortest_paths.csv` and
`ephys_shortest_paths.csv` (one row per walk: n_steps, min_steps, is_shortest),
new per-subject/per-session column `shortest_path_percent`, group entries
`shortest_path_percent` / `_pooled` / `_by_repeat` (fMRI) resp.
`_by_rep_correct` (cells), plus one overview histogram per modality.

**Numbers (all data, no exclusions beyond the ones already in the script):**
fMRI 95.8 +- 4.1 % (mean +- s.d. over 33 subjects, 12 514 walks; pooled 95.8 %);
cells 97.4 +- 2.2 % (63 sessions, 56 904 walks; pooled 97.4 %). Mild improvement
over repeats (fMRI 94.5 % -> 96.3 % from repeat 1 to 5; cells 97.0 % -> 97.6 %).
Detours are essentially always +2 steps (fMRI 472 of 524 non-shortest walks).

**Two definitional choices, both to keep the metric about paths BETWEEN
rewards.** (i) The first A of an fMRI configuration is skipped: the location the
subject starts from is not stored in the cleaned table. (ii) In the cells data
the D->A walk is only counted when the preceding attempt was itself a retained,
correct repeat that this attempt continues from, so the exploratory search for A
at the start of a grid never enters the metric. A variant that counts every
D->A / grid-initial walk regardless gives 97.1 % pooled instead of 97.4 %, so
this choice barely moves the number.

**One data glitch, left in.** Exactly 1 of 134 381 cell-data location
transitions is between non-adjacent squares (a dropped sample in the 25-ms
trace), which makes 1 of 56 904 walks come out one step SHORTER than the
Manhattan distance. It is counted as not-shortest; no filter was added.

**Unrelated speed fix in the same file.** The 25-ms location traces are stored
as a single very wide row and `pd.read_csv` parses them column by column
(~0.35 s per file, 1489 files ~ 9 min). `_read_location_trace` reads the line
and parses it with numpy (~10 s total). `ephys_trajectory_candidates` now uses
it too.

## 2026-09-02 (later) — Single-lag reporting for the double dissociation

Added two ways to report ONE lag instead of the 30+60 / 0+330 window in
`scripts/overlay_double_dissociation.py`, because the correction scope has to
match how the lag was chosen. Outputs:
`overlay_prespecified_lag_tests.csv`, `overlay_peak_lag_permutation.csv`.

**(a) a-priori lag, BH across the 3 ROIs** (`PRESPECIFIED_LAG_DEG`, default
mPFC 30 deg, HC 0 deg).

**(b) peak lag chosen by looking, sign-flip permutation, max-t across the 12
lags** (10000 perms, seed 42). Motivated by the lag-lag correlation in
`lag_lag_correlation_noctrl.csv`: neighbouring lags correlate at r = .33
(mPFC), .23 (HC_mid), .22 (HC_anterior), so BH over 12 lags assumes an
independence that does not hold. The permutation flips the sign of whole unit
curves (keeps each unit's lag-lag structure) and shares a subject's sign
across ROIs in the across-ROI null. `_check_vectorised_t` asserts the
vectorised permutation t reproduces `_tstat_gt0` on the observed data, so
empirical and null use one estimator (CLAUDE.md rule 4).

**Result: the max-t correction buys almost nothing** — mPFC 30 deg cell-level
BH q = .0177 vs FWE p = .0159; HC_mid 0 deg cell BH q = .057 vs FWE p = .054.
The lag curves are not correlated enough for the search penalty to shrink much.

**Peak lags under FWE (max-t across 12 lags):** cell mPFC 30 deg t(157) = 3.02,
p_FWE = .016 (*) is the ONLY peak that survives. Cell HC_mid 0 deg p_FWE = .054,
cell HC_anterior 330 deg p_FWE = .13, subject mPFC 60 deg p_FWE = .18, subject
HC_mid 0 deg p_FWE = .14, subject HC_anterior 0 deg p_FWE = .36.

**The a-priori mPFC lag choice is decisive, and that is a trap.** With mPFC
fixed at 30 deg: cell q = .0044 (**) but subject q = .108 (n.s.). With mPFC
fixed at 60 deg: subject q = .029 (*) and all three ROIs significant at subject
level (HC_mid .029, HC_anterior .040), while cell mPFC drops to q = .0496.
This is the cell-vs-subject peak reversal already documented in the supplement.
Picking 30 or 60 after seeing these numbers would be circular; recorded here so
that the choice is made explicitly and on a priori grounds, not by outcome.

**Conclusion: the two-lag window remains the only route where all three ROIs
survive under BOTH weightings**, so it stays the inference for Fig 3c; single
lags are fine to quote descriptively.


## 2026-09-02 — Double-dissociation overlay: readable results table + results.json

`scripts/overlay_double_dissociation.py` now writes `overlay_results.json` and
`RESULTS.md` (replacing `WEIGHTING_RESULTS.md`, which only showed 4 of the 12
lags and one q column). No statistic changed -- this is a reporting refactor.

**Why:** the old CSVs made it impossible to see which t belonged to which r/z,
and three differently-scoped FDR corrections were all called `..._fdr_...`
without saying what family they corrected over.

- Every test record now carries `mean_fisher_z` (what the t was computed on),
  `r_from_fisher_z = tanh(mean_fisher_z)` (the same effect in r units) and
  `mean_raw_r` (what the figure plots), so t <-> r <-> z is unambiguous.
- FDR columns renamed to name their family: `q_*_within_roi_12_lags`,
  `q_*_within_lag_across_overlay_rois`, `q_*_across_rois_and_lags`,
  `q_*_across_rois`. `FDR_FAMILIES` in the script documents each one and is
  copied into the json and the md.
- `overlay_results.json` nests `p` and `fdr_q` inside each test record; also
  holds `settings`, `test_definitions` and `fdr_families`.
- Re-ran on
  `per_lag_encoding/2026-08-28_10-18-21_reload_from_2026-06-30_18-21-57_relabelled/`
  (the run with the corrected cell coordinates).

**Fig 3c star, target-window contrast (predicted lags > other 10 lags,
one-sided, BH across the 3 ROIs), subject-balanced:** mPFC 30+60 deg
dz = 0.057, t(32) = 2.86, p = .0037, q = .0055 (**); HC_mid 0+330 deg
dz = 0.047, t(35) = 2.88, p = .0034, q = .0055 (**); HC_anterior 0+330 deg
dz = 0.032, t(51) = 1.81, p = .038, q = .038 (*). Cell-weighted: mPFC
t(157) = 2.87, q = .0057; HC_anterior t(294) = 2.69, q = .0057; HC_mid
t(232) = 2.27, q = .012. So all three survive FDR under both weightings.


## 2026-09-01 (later still) — Phase-residualisation choice is now self-documenting

`PHASE_RESIDUALISE = 'cosine_2h'` was justified from a run on the old 8-recday
set, and the justification lived only in a comment. It is now RECOMPUTED EVERY
RUN: `PHASE_BASES_TO_COMPARE = ['cosine', 'cosine_2h']` sends each basis through
the same `run_all_recdays` -> `methods_results_stats` path as the primary
analysis, prints a read-out, and writes
`key_analysis_stats.json['phase_residualisation_comparison']` with the
criterion stated in the file. Costs one extra full pass per non-default basis.

Criterion (fixed in advance): use the basis that removes the Subgoal Progress
effect, since subgoal progress dominates this dataset and any residue would
inflate the action-plan fit.

Re-tested on the full 25 recdays / 7 mice — the original conclusion holds:

| basis     | analysis      | Subgoal Progress beta | t(23) | q_FDR    | sig |
|-----------|---------------|-----------------------|-------|----------|-----|
| cosine    | full_z        | +0.0495               | +5.40 | 1.2e-05  | YES |
| cosine    | across_halves | +0.0593               | +3.26 | 2.3e-03  | YES |
| cosine_2h | full_z        | -0.0229               | -5.82 | 1.00     | no  |
| cosine_2h | across_halves | -0.0520               | -3.60 | 0.999    | no  |

A single harmonic leaves a significant positive subgoal-progress effect; two
harmonics remove it. Rodent phase tuning (von Mises, kappa = 3.33) is sharp
enough to carry second-harmonic structure that one harmonic leaves behind.

Important for the manuscript: the choice does NOT manufacture the main effect.
Action Plan is essentially unchanged either way — full_z 0.2259 (t = 9.23) with
'cosine' vs 0.2324 (t = 9.57) with 'cosine_2h'; across-halves 0.2820 (t = 6.73)
vs 0.2964 (t = 7.21). All significant at q < 1e-5.

Caveat to state honestly: 'cosine_2h' does not leave Subgoal Progress at zero,
it leaves it reliably NEGATIVE (t(23) = -5.82). The one-sided criterion "not
significantly positive" is met, but describing it as "null" is inaccurate — it
is better described as over-corrected. Worth pre-empting, since a reviewer can
read it as over-residualisation.

## 2026-09-01 (later) — Full OSF release downloaded + uniform self-normalisation

The authors have now confirmed that **the normalisation they settled on is not
the one published** in `Basic_analysis.ipynb`, and it has not been shared. That
closes the earlier puzzle: the published `raw_to_norm` reproduces their released
`Neuron_*` arrays only to r ~ 0.88 because it is a different method, not because
we were running it wrong. Their released normalised files are therefore
unreproducible, and the only self-consistent option is to normalise everything
ourselves with one function.

**Downloaded (OSF 3d9r2):** all 25 combined ABCD recdays, 7 mice
(ab03, ah03, ah04, ah07, me08, me10, me11), 193 sessions, ~2.0 GB of
`Neuron_raw` / `Location_raw` / `trialtimes` / `Task_data`. Was 8 recdays /
5 mice. ab03 and ah07 are entirely new — they do not exist in the Drive share.

**New: `scripts/normalise_rodent_ephys.py`** + `raw_to_norm`, `normalise_segment`,
`state_boundaries` in `mc.analyse.analyse_ephys_clean`. Transcribed from the
authors' published `partition`/`normalise`/`raw_to_norm`. Output:
`derivatives/normalised_loc-max_<timestamp>/`, with a manifest and a settings
JSON recording every parameter and per-session shape. The raw release is never
modified.

    25 recdays / 7 mice / 193 sessions / 1252 neurons, 0 NaNs, all (n, trials, 360)

**Bug found and fixed in applying their code to Location.** Their `normalise`
stretches a state segment shorter than 90 raw bins by `np.repeat(x,10)/10`. The
`/10` is correct for a firing RATE and wrong for a categorical node ID — it
turns node 7 into 0.7. Their released `Location_*` arrays hold clean integers,
so they clearly do not divide there. `normalise_segment` now takes
`rate_scaled`, defaulting to True for 'mean' and False otherwise. Affects 1.12%
of state segments (109/9720) across 38 sessions; exact-bin agreement with the
released Location files rose 92.3% -> 93.9%.

**Undocumented choice, flagged:** the statistic for Location is not stated
anywhere by the authors. `--location-statistic` defaults to `max` (their
`take_max` option, and the only semantically sound one for node IDs). Agreement
with their released files: median 93.3%, max 92.3%, min 93.0%, mean 88.3% — no
statistic reproduces them, consistent with their method differing. Worth
confirming with them.

**CORRECTION to the earlier entry today.** The three "orphan" normalised
sessions (`ah04_05122021_06122021_3`, `ah04_09122021_10122021_3`,
`me10_09122021_10122021_8`) were downloaded and turn out to be **empty arrays**,
shape `(0,)`. So they were a genuine bad-session flag after all, just encoded as
an empty file rather than a missing one — the previous 61-session analysis was
correct and nothing was being wrongly discarded. `cross_view_session_ids` now
drops sessions that are absent OR empty in either view, so both encodings are
caught.

**`analysis_rodents_complete_clean.py`:** new `NORM_FOLDER` setting (now pointed
at the run above); the recday list is taken from whichever source supplies the
normalised view, so the analysis widens to 25 recdays automatically.
`load_ephys_data` gained `norm_folder`. Also added a **within-mouse robustness
test**: per-recday betas averaged within animal, then run through the IDENTICAL
`methods_results_stats` path (same one-sided t-test, same BH-FDR), written to
`key_analysis_stats.json` as `full_z_by_mouse` / `across_halves_by_mouse`.

Not yet re-run: all previously reported rodent numbers are from the authors'
normalised files at n = 8 recdays / 5 mice and are superseded.

Low-yield recdays to keep an eye on when the results land:
`me10_20122021_21122021` has 1 neuron and `me10_17122021_19122021` has 6, so
their per-recday betas will be very noisy.

## 2026-09-01 — Rodent data: the full release is on OSF, not the Drive share

New: `scripts/download_rodent_ephys_data.py` — per-file downloader (never
builds the multi-GB archive that makes the web download crash), verifies each
file with `np.load`, retries with back-off, resumes after a crash.

**Two sources, and they are not the same dataset.**

| | recdays | mice | raw | normalised 360-bin |
|---|---|---|---|---|
| OSF `3d9r2` (public release) | 25 | 7 | yes | **no** |
| private Google Drive share   | 14 | 5 | yes | yes, for 8 recdays |

Our 8 recdays came from the Drive. A recday is `{mouse}_{day1}_{day2}` — a
recording UNIT (two days spike-sorted together, 6 task configs), not an animal;
the 8 are ah03 x1, ah04 x3, me08 x1, me10 x1, me11 x2 = **8 recdays / 5 mice**.
The analysis docstring previously implied 8 animals; corrected, and
`key_analysis_stats.json` now carries a `settings.sample` block with
`n_recdays`, `n_mice`, `recdays_per_mouse`.

**Missing: 17 recdays, ~2.0 GB, including two entire mice.** `ab03` (3 recdays)
and `ah07` (3 recdays) are absent from the Drive share altogether. On disk vs on
OSF, per mouse: ab03 0/3, ah03 1/2, ah04 3/5, ah07 0/3, me08 1/3, me10 1/4,
me11 2/5. Five of the 25 (`combined_ABCDonly_notone_days.npy`) were recorded
without the state tones — a different sensory regime, kept separable via
`--tone-only`.

**Blocker on using them: OSF ships raw only.** The DSR analysis runs on the
normalised view (n_neurons x n_trials x 360, 90 bins/state), which OSF does not
include.

The authors' normalisation code IS public — `raw_to_norm` / `normalise` in
`Basic_analysis.ipynb` cell 21 of github.com/mohamadyelgaby/mFC_schema:

    Trial_times_conc = np.hstack((np.concatenate(tt[:,:-1]), tt[-1,-1])) // 25
    segments  = partition(raw_neuron, Trial_times_conc)      # one per state
    per_state = binned_statistic(arange(L), seg, 'mean', bins=90)[0]
                # with: if len(seg) < 90 -> seg = np.repeat(seg,10)/10 first
    Neuron_norm = per_state.reshape(n_states//4, 360)        # NO smoothing
                # (smoothing_sigma=10 applies only to raw_to_norm(return_mean=True))

Running it verbatim still does NOT reproduce the shipped `Neuron_*` files.
Over 18 sessions from 6 recdays: **mean r = 0.877** (range 0.75-0.97) for
neurons, **0.785** for locations, **zero exact matches**, and the trial count is
off by one in 8/18 sessions.

Cross-checked against OSF, not just the Drive: `Neuron_raw`, `Location_raw` and
`trialtimes` for ah03_18082021_19082021 (sessions 0, 2) and me08_10092021_11092021
(sessions 0, 2) are BIT-IDENTICAL between the two sources, and re-running the
authors' code on the freshly downloaded OSF raw gives exactly the same r
(0.8605 / 0.8050 / 0.9088 / 0.9648). So the mismatch is not a Drive-vs-OSF
artefact — the released raw is the same everywhere, and it still is not the
array that produced the released normalised files.

Why it cannot match: the bin values are exact rationals whose NUMERATORS agree
with ours but whose DENOMINATORS do not. For ah03_18082021_19082021_0, trial 0,
bin 0 the shipped value is 11/13 = `raw[5:18].mean()`, while their code on the
shipped raw gives 11/9 = `raw[0:9].mean()`. Same spikes, wider window, offset
start. I.e. **the published `Neuron_raw` is not the exact array that was fed to
`raw_to_norm`** — there is an alignment/binning difference upstream of the
released files. The off-by-one trial counts point the same way. So the recipe is
recoverable; the authors' exact output is not.

Consequence: do NOT mix the authors' normalised arrays for the old 8 with a
home-made version for the new 17 — the preprocessing difference lines up
exactly with the mouse/recday split and would confound the group test. Rebuild the normalised view from raw
for ALL 25 recdays with the `raw_to_norm` recipe above — that is now the only
self-consistent option, since the authors' own output cannot be reproduced.

**Fixed (Drive, changes results): 3 orphan normalised sessions restored.**
`Location/Neuron_ah04_05122021_06122021_3`, `..._ah04_09122021_10122021_3` and
`..._me10_09122021_10122021_8` existed on the Drive but had never been
downloaded. `cross_view_session_ids` drops sessions absent from the normalised
view assuming absence is the authors' implicit "bad session" flag; for these
three it was a download gap, so they were being discarded for no reason.
Downloaded 2026-09-01. Session counts now:

    ah04_05122021_06122021   7 -> 8
    ah04_09122021_10122021   7 -> 8
    me10_09122021_10122021   8 -> 9

Raw and normalised session lists now agree exactly for all 8 recdays (64
sessions, 504 neurons), i.e. the cross-view gate is currently a no-op. **The
analysis must be re-run — every number from before 2026-09-01 was computed on
61 sessions.**

Note on where raw is used: with `run_continuous: False` (the setting in
`analysis_rodents_complete_clean.py`) the raw branch of `process_one_recday` is
skipped entirely, so no reported result is computed from the raw files. They are
loaded only to build the `cross_view_session_ids` gate.

Indexing notes: gdown cannot enumerate the Drive folder (Google's folder HTML
caps at 50 entries per folder, the folder has 5037 files) — the script scrapes
`embeddedfolderview` instead. OSF is walked via its public API and cached as
`_osf_index.json`.

## 2026-08-29 — fMRI RSA: collinearity between the model RDMs of `DSR-contr_except_prev_but`

New: `scripts/plot_model_RDM_correlations.py` + `mc.plotting.results.plot_model_correlation_matrix_pub`.
Correlates the *model* RDMs of one combo GLM with each other (design
collinearity check for the RSA), and plots them as a 4 x 4 cm publication panel.

The model RDMs are built with the SAME code path as the searchlight RSA
(`pair_correct_tasks` -> per-model metric -> upper triangle, `diagonal_included:
false`), and, because the config sets `masked_conds: true`, restricted to the
same RDM cells the GLM is fit on — `make_category_masks(...,
mask_only_path_rew_combos=True)` keeps only same-type pairs (path-path and
reward-reward): 1,560 of 3,160 cells. No cells are dropped anywhere else.

Combo `DSR-contr_except_prev_but` (`rsa_config_quarters_DSR_controls.json`,
EVs `DSR_loc-fut-rews-state-dur-type`), n = 32 subjects with a local EV pickle,
group value = Fisher-z mean of the per-subject Pearson r:

|              | DSR   | location | A-state | l2_norm | next_buttons |
|--------------|-------|----------|---------|---------|--------------|
| location     |  .531 |          |         |         |              |
| A-state      | -.009 |  -.024   |         |         |              |
| l2_norm      |  .294 |   .570   |  -.018  |         |              |
| next_buttons |  .269 |   .199   |  -.014  |  .265   |              |
| buttons_out  |  .269 |   .163   |   .015  |  .224   |  .162        |

SD across subjects <= .054 everywhere, i.e. the design geometry is essentially
identical in every subject. Highest collinearity is location <-> l2_norm (.57)
and DSR <-> location (.53) — both expected (the DSR is built from the location
vectors; l2_norm is a graded version of location). A-state is orthogonal to
everything (|r| <= .024).

Outputs (mean/SD csv, per-subject matrices npy, settings json, pdf + png):
`data/derivatives/group/model_RDM_correlations_DSR-contr_except_prev_but_29-08-2026/`

## 2026-08-29 — mid-HC diagnosis: the "future-only" DSR is not a separate test, and the location result is a control-stack artefact

Diagnostic pass on `DSR_RSA_simple_ROI/2026-08-27_19-18-20` (latest),
`2026-07-30_15-58-51-fixed_cells-fixed_perms` and `2026-07-30_11-11-36`, to
resolve why mid HC carries the strongest concurrent-future β (Fig 2d) while its
cells are tuned to now / just-past (Fig 3c). No new runs; all numbers read from
stored results.

### 1. `dsr_fmri_fut` is a re-run of `dsr_fmri`, not an independent test

On the 4,560 valid RDM cells (HC_mid, `split_halves_z` mask):

    corr(dsr_fmri, dsr_fmri_fut) = 0.980

Dropping lag 0 removes 1 of 12 lag-windows from a Hamming distance over the
rolled 144-int trajectory, so the geometry barely moves. Consequences visible
in both runs: `ctrl_dsrFULL` and `ctrl_dsrFUT` return **identical p_perm to 3
dp** for every ROI (mPFC .022/.022, mOFC .866/.866, PCC .666/.666, HC_ant
.103/.103, HC_mid .001/.001) and identical control βs to 4 dp.

Also, lags 1 and 11 (30°, 330°) coincide with the current location on 39% of
bins (the autocorrelation figure already in Methods), so "future only" still
contains the present. **`ctrl_dsrFUT` cannot be cited as evidence that mid HC
codes the future.** Either drop it or replace it with a genuinely disjoint
model (e.g. lags 3–9 only).

Related: `corr(dsr_fmri, location) = 0.408` but `corr(dsr_fmri_fut, location) =
0.218` — dropping lag 0 does halve the location confound, yet the HC_mid β only
moves 0.073 → 0.068. So the mid-HC effect is not simply location leaking in
through lag 0.

### 2. The lag decomposition resolves the Fig 2d / Fig 3c tension

Joint quarter fit, `ctrl_dsrQUARTERS` in `2026-07-30_11-11-36` (4 quarters
compete against each other plus location + bttn_curr). Quarter k = lags
{3k, 3k+1, 3k+2}, i.e. curr = 0–60°, next = 90–150°, next2 = 180–240°,
next3 = 270–330°:

| ROI | curr | next | next2 | next3 |
|---|---|---|---|---|
| mPFC | .021 (p=.298) | .003 (p=.626) | .021 (p=.204) | .031 (p=.104) |
| mOFC | .012 (p=.155) | −.020 | .005 | −.005 |
| PCC | .021 (p=.200) | −.053 | −.005 | **.048 (p=.005)** |
| HC_anterior | **.074 (p=.001)** | −.013 | .003 | .011 (p=.294) |
| HC_mid | .046 (p=.059) | .020 (p=.243) | .010 (p=.477) | **.073 (p=.001)** |

Single-model (no controls, latest run) shows the same shape for HC_mid:
curr .076 (p=.001), next .033 (p=.026), next2 .026 (p=.054), next3 .088 (p=.001).

**There is no contradiction.** Each quarter correlates 0.607 with the full DSR
by construction, so a region that matches on 2 of 4 quarters yields a large
full-DSR β. Mid HC's fit is carried by the two quarters flanking the present
(now and just-past) — exactly the cell-level profile (0°, p=.0131; 330°,
p=.0449). Anterior HC is purely present. The full-DSR regressor simply cannot
report which lags carry it.

### 3. Caveat: the decomposition does NOT confirm the mPFC 30/60° peak

`dsr_fmri_informed` (lags 1,2 = 30°+60°, the pre-registered mPFC window) in
`ctrl_dsrInformed`: mPFC β = +0.0196, **p = 0.130 (n.s.)**, while HC_anterior
β = +0.0557 (p=.001) and HC_mid β = +0.0436 (p=.006). This is because
`informed` correlates 0.873 with `curr_quarter` and 0.187 with `location` — it
is largely a near-present model, which is why the hippocampi load on it.

mPFC's DSR fit sits numerically in the later quarters (next2 p=.078, next3
p=.072 single-model; next3 p=.104 joint) but nothing survives. With n = 65 mPFC
cells this may be power, but at the RDM level the mPFC lag profile does **not**
independently reproduce the cell-level 30/60° result. This should be stated
rather than glossed — it is the weak point Jensen and Dorrell will find.

### 4. The mid-HC location effect is a control-stack artefact, not a data change

Same run, same cells (`2026-07-30_15-58-51-fixed_cells-fixed_perms`), location
as the read-out sub-model:

| combo | members | HC_mid | HC_ant |
|---|---|---|---|
| `ctrl_dsrFULL` | state, location, bttn_curr, dsr_fmri | **+.0330 (p=.025)** | **+.0403 (p=.009)** |
| `fmri_ctrl_dsrFULL` | + **l2_norm, bttn_next** | +.0145 (p=.223) | +.0356 (p=.040) |

Latest run carries only the second stack: HC_mid +.0153 (p=.215), HC_ant
+.0330 (p=.052).

**The culprit is `l2_norm`**: `corr(location, l2_norm) = 0.588`. It is a second
parameterisation of the same variable (graded negative distance to each of 9
grid locations vs categorical location). Singly in HC_mid: location .0643
(p=.001), l2_norm .0631 (p=.002) — near-identical. Jointly they split the
variance and neither clears.

Not a data change: the 07-31 and 08-27 runs return the same numbers despite
HC_mid n going 143 → 145, and 07-30's `fmri_ctrl_dsrFULL` (+.0145) matches
08-27's `ctrl_fMRI-state_dsrFULL` (+.0153).

**Manuscript consequence.** Fig 3g's caption describes exactly `ctrl_dsrFULL`
("controlling for future locations, position in sequence and current actions")
— no l2_norm, no next button — while Fig 2d uses the stack that includes them.
The two figures therefore use different control models, and the location claim
survives only under the leaner one. Options, in order of preference:

  (b) Test `location` + `l2_norm` jointly as one spatial-code contrast rather
      than pitting two parameterisations of one variable against each other.
      Correct given r = 0.59, and removes the arbitrariness.
  (a) Harmonise on the full stack and report mid HC as n.s. (p=.215), ant HC
      marginal (p=.052). The present-coding claim then rests on the cell-level
      result and on the quarter split (HC_ant curr p=.001, HC_mid next3
      p=.001), which is stronger evidence anyway.
  (c) Drop l2_norm from the location model only — hard to justify while it
      stays in the DSR model.

### Next step

`ctrl_dsrQUARTERS` is currently commented out in `RSA_DSR_ROIs_simple.py`
(combo_models). The joint quarter numbers above come from the pre-relabelling
cell set (mOFC 85 vs 74, HC_ant 162 vs 171, HC_mid 143 vs 145), so it needs a
re-run on the current cells before it can go in the paper. That single figure
answers [79], [102], [85], [107] and [109] in the co-author comments.


## 2026-08-28 (night) — `within_only` scope: the right fix for the instruction question

The user's suggestion -- drop the across-half block entirely -- is better than
the block nuisance regressor, and it is now implemented as
`data_rdm_scope = "within_only"`.

**Why it removes the artefact structurally.** The bias came from a similarity
offset BETWEEN the within-half and across-half blocks (data: 0.828 vs 0.863
dissimilarity, within < across in 89% of searchlights) which the instruction
regressors encode (r = +0.42 to +0.54 with an across-half indicator). Keep only
within-half cells and that contrast does not exist, so no regressor can absorb
it. Unlike the nuisance regressor, nothing has to be modelled away.

**It keeps the contrast the design was built for.** Of the 90 within-half
cells, 10 are same-task-letter pairs (the two directions of one task inside one
half). On exactly those cells the instruction dissimilarity is **0** (they saw
the same sequence) and the execution dissimilarity is **1** (they execute the
reverse). That is the instruction x execution dissociation, and it lives
entirely inside the within-half block.

**Empirical check** (sub-02, TR4, 3161 searchlights, mean single-subject t):

| combo | scope | instr | exec |
|-------|-------|-------|------|
| rewDSR_vs_instr | full_no_diag | +0.268 | -0.254 |
| rewDSR_vs_instr | **within_only** | -0.472 | **+0.058** |
| splitDSR_vs_instr | full_no_diag | +0.120 | -0.063 |
| splitDSR_vs_instr | **within_only** | **-0.081** | **+0.011** |

The mirror-image offset is gone. The residual -0.47 for `rewDSR_instr` in the
2-regressor combo is not obviously an artefact: a negative instruction beta on
within-half cells means same-instruction pairs are LESS similar than the
instruction model predicts, which is what you expect if the same-letter pair
(same instruction, reversed execution) is dominated by execution coding.

**No temporal-proximity confound.** Within a half, same-task pairs are closer
in time in TH1 (mean gap 362 vs 594 s) but FURTHER apart in TH2 (575 vs 528 s)
for sub-02 -- the ordering is not systematic, and the gaps are hundreds of
seconds, far outside BOLD autocorrelation.

**No recomputation needed.** `within_only` reads the same `data_RDM_full.npy`
cache as `full_no_diag` and keeps 90 of its 190 columns. Cell ordering is
`np.tril_indices(n_all, k=-1)` in both `get_full_instruction_RDM_per_searchlight`
and `_lower_tri_flat`, verified, so the mask aligns. If those caches survive on
the cluster this is a re-fit, not a re-run.

**Implemented** in `scripts/fMRI_run_RSA_instruction.py`: `within_half_mask()`,
`data_rdm_scope = "within_only"` accepted, model regressors and the cached data
RDM both subset by the same mask, and the `block` nuisance now asserts against
`within_only` (constant there, and unnecessary). Config
`condition_files/rsa_instruction_within_and_across_th.json` renamed to
`rsa_instruction_within_th_only.json`, `name_of_RSA =
within_th_only_intr-vs-exe`, scope `within_only`, block nuisance dropped.

Verified: all 10 single models and all 3 combos are full rank on 90 cells
(2/2, 3/3, 5/5, 9/9).

**The figures and printouts were WRONG for the new scope and are now fixed.**
Three places still described the across block regardless of `data_rdm_scope`:

1. The model-RDM figures always plotted `model_RDM_dir[model]`, the (n, n)
   across block, and the assembled (2n, 2n) figure only fired for
   `full_no_diag`. In `within_only` that meant the saved figure showed cells
   that are not fitted -- and for every instruction model it is a uniform block
   of 1.0, carrying no information at all. Replaced with `_display_RDM()`,
   which returns the matrix for the current scope with excluded cells set to
   NaN (`plot_instruction_RDM` already renders NaN white via
   `masked_invalid` + `set_bad`). Filenames now carry the scope:
   `{results_dir}_{model}_{data_rdm_scope}`.
2. The example data-RDM figure had the same problem; same fix.
3. The printed "execution vs instruction Pearson r" fell through to the
   across-block branch for any scope other than `full_no_diag`, so in
   `within_only` it would have reported a correlation against a constant
   vector. It now always uses `_model_regressor()`, i.e. exactly the vectors
   the OLS sees. `_model_regressor` was moved above the verification block so
   both share one definition.

Also `verify_instruction_rdm_blocks` now runs on the matrix actually fitted
(across block for `across_only`, the assembled matrix otherwise) and prints
which scope it checked.

Rendered all three scopes for `rewDSR_instr` and `rewDSR` to confirm: 100 / 400
/ 200 cells shown for across_only / full_no_diag / within_only, and
`rewDSR_instr` under `across_only` is `unique = [1.]` -- the degenerate uniform
block, now impossible to mistake for a real model RDM.

**Config as it will run:** `within_th_only_intr-vs-exe`, scope `within_only`,
TR set per run, 10 single models + 3 combos (`rewDSR_vs_instr` 2,
`instr_split` 4, `splitDSR_vs_instr` 8) = 24 beta/t/p map sets per subject per
TR. No block nuisance (nothing left to absorb).

**Status of the three scopes.**
- `across_only` -- correct for EXECUTION. Instruction is constant, so it cannot
  be tested there at all.
- `within_only` -- correct for INSTRUCTION, and it also carries the
  instruction x execution dissociation. Execution is estimable but from
  within-run cells only.
- `full_no_diag` -- mixes the two and introduces the block offset. Superseded;
  use it only with the block nuisance, and only as a control.

## 2026-08-28 (evening) — the full_no_diag t-bias diagnosed, and a block nuisance regressor

**The bias is real and it is not regressor collinearity.** Whole-brain t over
all voxels x 12 TRs, `instr_test_full`:

| map | mean t | % voxels > 0 |
|-----|--------|--------------|
| rewDSR_instr (single model) | **+2.07** | 97.0% |
| REWDSR_INSTR-rewDSR_vs_instr | +2.06 | 97.0% |
| rewDSR (single model) | **-1.04** | 15.7% |
| REWDSR-rewDSR_vs_instr | -1.03 | 16.0% |
| simple | -1.18 | 13.7% |

Single-model fits show the same offset, so it is not suppression between
regressors -- and r(rewDSR, rewDSR_instr) = -0.004 anyway. For comparison, the
`across_only` maps sit at -0.48 to +0.71.

**Cause: a within/across block offset in the DATA that the instruction models
encode.** Of the 190 lower-triangle cells, 90 are within-half and 100 across.

- Data (sub-02 TR4, 126 404 searchlights): mean cosine dissimilarity **0.828
  within vs 0.863 across**, a 4.0% offset, with within < across in **88.9%** of
  searchlights. Run-level noise.
- Regressors, correlation with an across-half indicator: every `_instr` model
  **+0.42 to +0.54** (within mean 0.711, across constant 1.000); every
  execution model **-0.13 to -0.17** (0.911 vs 0.820).

An instruction regressor therefore says exactly what the run noise says, and
collects a large positive beta in nearly every voxel; execution collects the
mirror-image negative one.

**The existing group maps CANNOT be corrected post hoc.** 82% of the block
indicator is a direction orthogonal to the old design `[1, instr, exec]`, i.e.
information the fit never computed. Predicting the correctly-fitted beta from
everything that was saved (beta and t for both regressors, 3152 searchlights):
R^2 = **0.577** for instruction, 0.936 for execution. Not a correction, and not
close enough to be one.

**But a re-fit may not need the expensive step.** The searchlight data RDMs are
cached as `data_RDM_full.npy` and `fMRI_run_RSA_instruction.py` already skips
computation when the cache exists. 99.8% of searchlights have no NaN cell (only
3 distinct NaN patterns among the other 293), and the OLS for all 126 111 clean
searchlights at one design takes **21 s** in the current per-searchlight loop
(0.6 s vectorised via `evaluate_model_vec`, 34x). So if those caches survive on
the cluster this is minutes per subject-TR, not days. Worth checking before
committing to a long re-run.

**Implemented:** `build_block_nuisance_RDM()` in
`scripts/fMRI_run_RSA_instruction.py`, reserved regressor name `block`, and a
config flag `add_block_nuisance: true` that appends it to every combo so it
cannot be forgotten in one. Asserts `full_no_diag` scope (in `across_only` it
would be constant). Enabled in
`condition_files/rsa_instruction_within_and_across_th.json`.

Verified on sub-02 TR4: all three combos stay full rank with it
(4/4, 6/6, 10/10), and over 3161 searchlights the mean single-subject t for
`splitDSR_vs_instr` moves from instr +0.120 / exec -0.063 to
**instr -0.101 / exec +0.003 / block +1.319** -- the nuisance takes the offset
and execution recentres on zero.

**What it fixes and what it does not.** After the nuisance absorbs the offset,
where each regressor's remaining variance lives:

| regressor | % within-half | % across-half |
|-----------|---------------|---------------|
| rewDSR_instr | **100.0%** | 0.0% |
| curr_rew_instr / two_next_rew_instr | **100.0%** | 0.0% |
| rewDSR | 21.4% | 78.6% |
| curr_rew / two_next_rew | 33.1% | 66.9% |

Execution keeps 67-79% of its variance in across-half cells, so the nuisance
makes those fits interpretable. Instruction keeps **zero** -- across halves the
same task is instructed in the reverse order, so those cells carry no
instruction information at all and the estimate is always a purely within-run
comparison, where "same instruction" is also "same stimulus, same run". No
nuisance regressor can change that; it is a property of the counterbalancing.

**Recommendation:** the re-run buys trustworthy execution numbers in
`full_no_diag` (which `across_only` already provides more cleanly) and a clean
demonstration that the instruction effect was the block artefact -- a good
supplementary control. It does NOT make the instruction models usable as
evidence for instruction coding.

## 2026-08-28 (later) — all three per-TR datasets on identical footing

All three now run through `scripts/per_TR_loso.py` / `mc.analyse.loso`, same 5
masks, same seeds, 10 000 SVC perms, LOSO k=50/100/200, whole-brain at 1000.

| # | dataset | maps | scope | output |
|---|---------|------|-------|--------|
| 1 | `instr_test_full` | 27 (25 usable) | `full_no_diag` | `per_TR_svc_instr_test_full_allTR_2026-08-28` |
| 2 | `split_rew_DSR_per_TR` | 8 | `across_only` | `per_TR_svc_split_rew_DSR_allTR_2026-08-27` |
| 3 | `instruction_per_TR` | 1 (rewDSR) | `across_only` | `per_TR_svc_instruction_rewDSR_allTR_2026-08-28` |

**The reported number reproduces exactly** through the new runner:

    reported   : t=5.079 TR4 MNI -6/32/18 p_FWE=.0407 n_vox=4181
    new runner : t=5.079 TR4 MNI -6/32/18 p_FWE=.0407 n_vox=4179

(The 2-voxel difference is the mask-intersection change; it does not move the p
at four decimals.) Its LOSO is p_FWE = .0144 at TR4. Nothing in MTL or visual
(best p = .47), nothing whole-brain (p = .451).

**Dataset 1 result: every single significant map is an instruction model, and
NO execution map is significant in ANY mask.**

| mask | sig maps | best instruction | best execution |
|------|----------|------------------|----------------|
| mPFC | 6/25 | curr_rew_instr t=6.94 TR2 -6/66/18 **p=.0002** | NEXT_REW-splitDSR_noInstr t=3.77 p=.313 |
| MTL_L | 8/25 | rewDSR_instr t=8.30 TR5 -32/-2/-34 **p<.0001** | THREE_NEXT_REW-splitDSR_noInstr t=3.20 p=.482 |
| MTL_R | 7/25 | curr_rew_instr t=6.41 TR0 22/0/-24 **p=.0002** | t=2.76 p=.665 |
| visual | 6/25 | CURR_REW_INSTR-splitDSR_vs_instr t=8.52 TR3 **p<.0001** | t=4.72 p=.136 |

0 of 10 execution maps reach p<.05 in any of the five masks. Whole brain: 7 of
25 significant, all instruction, peaking t=10.3 at TR2 (rewDSR_instr, 0/-32/-10).

**The scope, not the model, decides the answer.** The same subjects and the
same rewDSR construct give t=5.08, p=.041 in mPFC under `across_only`
(dataset 3) and nothing at all under `full_no_diag` (dataset 1, best mPFC
execution p=.313), while the instruction models go from structurally
impossible (constant regressor) to t=6.9-10.3. Adding the within-half cells
does not add power to the execution test -- it destroys it and replaces it
with a large instruction effect. That is what the within-half confound
predicts: within a half, "same instruction" IS the same visual stimulus in the
same run, so those cells inject stimulus-repetition structure that the
instruction regressor fits and the execution regressor does not.

**Timing supports the stimulus reading.** Peak TRs of the significant
instruction maps cluster early -- TR0:3, TR1:4, TR2:11, TR3:5, TR4:2, TR5:4,
TR6:6 (mode TR2) -- and the LOSO timecourses
(`per_TR_timecourses_instr.pdf`) rise at TR1-3 and decay, the shape of a
response to a screen that is on from 0 s, not of a plan assembled once all
four rewards are known at 6 s. Compare dataset 2, whose across-half execution
channels peak LATE (mPFC two_next TR4-6, left MTL next_rew TR7-8).

**Conclusion for the manuscript:** `across_only` is the defensible scope for
these questions. `full_no_diag` should not be used to compare instruction
against execution, because it is exactly the scope in which the two are
confounded. The instruction models in dataset 1 should not be reported as
evidence for instruction coding.

**Robustness fixes made while running this:**
- `_load_with_retry` -- the first 12-TR `instr_test_full` attempt died at model
  14/27 when `nib.load` transiently failed on an intact TR7 file (sync-backed
  storage). Reads now retry 4x with a 10 s wait; genuinely corrupt files still
  raise.
- `--resume` -- skips models whose outputs exist and rebuilds both summary
  tables from every per-model json on disk, so an interrupted run resumes
  without redoing finished work and still writes complete tables. Verified:
  "resume: 13 of 27 models already complete, 14 to run" -> "summary table
  covers 27 of 27 models".
- `load_ref` now intersects the group masks over the TRs that HAVE one and
  prints how many it used. The `instruction_per_TR` folders only ship
  `mask_all_32_subjects` for TR0 and TR3 -- which is why the original script
  read TR0's and stopped -- and this would otherwise have been a hard failure.
- `base_channel` strips a trailing `_instr`, so a reward channel keeps one
  colour across its execution and instruction variants.

## 2026-08-28 — per-TR LOSO analysis refactored into one runner + one library

The scripts folder had grown four files for one analysis. Consolidated:

**`mc/analyse/loso.py`** (new, registered in `mc/analyse/__init__.py`) holds
everything: inputs (`resolve_nii`, `load_ref`, `load_mask`, `load_masks`,
`discover_models`, `read_model_columns`), statistics (`tstat`, `null_max_t`,
`adaptive_pblock`, `voxel_fwe_p`, `perm_wholebrain`), tests (`run_svc`,
`run_loso`, `run_wholebrain`), volume writing (`vol_from_cols`,
`write_mask_maps`, `write_wholebrain_maps`), results loading (`load_loso`,
`load_settings`, `result_masks`) and plotting (`plot_per_TR_timecourses` plus
the CLAUDE.md palettes and the reward schedule).

**`scripts/per_TR_loso.py`** (new) is the only runner, with three modes:
`--mode run` (analyse, no figures), `--mode plot` (load an existing
`--out-dir` and plot, no recomputation), `--mode both` (default).
In plot mode it reads `settings.json` and defaults to the four reward channels
when present, else every model.

**Archived** to `scripts/old/per_TR_loso_pre_refactor/` with a README:
`svc_loso_test.py`, `svc_loso_batch.py`, `plot_per_TR_timecourses.py`,
`hemisphere_contrast.py` (the last retired at the user's request).

**Equivalence verified before archiving** — both implementations run on the
same input:

    tstat       bit-identical: True
    null_max_t  bit-identical (pblock = 250 and 1000): True
    run_loso    bit-identical to the inline LOSO of svc_loso_test.main(): True

One deliberate behavioural difference remains, as before: `load_ref` intersects
the group mask across all included TRs rather than taking TR0's alone (~21
voxels; mPFC 4181 -> 4182), so the reported `BA32-9-10` p will not reproduce
bit-for-bit. Documented in the archive README.

**References repointed:** `scripts/future_step_dominance_mPFC_lOFC.py` imported
`tstat, null_max_t` from `svc_loso_test` and now imports them from
`mc.analyse.loso`. Prose references updated in
`scripts/fMRI_run_RSA_instruction.py`, `scripts/mask_stats.py`,
`docs/rerun_after_roi_update.md` and
`scripts/old/instruction_phase_alternatives/README.md`.

**Still duplicating this code:** `scripts/mask_stats.py` and
`scripts/mask_stats_spyder.py` carry their own copies of `load_ref`,
`load_mask`, `extract_betas`, `tstat` and `null_max_t` (they compare voxel-wise
FDR against permutation FWE). They pre-date this work and were left alone; they
are the obvious next thing to fold into `mc.analyse.loso`.

## 2026-08-27 (evening, later) — HC/EC hemisphere split: next_rew is left-lateralised

**Masks:** `Garvert_MTL_2mm.nii.gz` split at MNI x = 0 into
`data/masks/Garvert_MTL_2mm_L.nii.gz` (1352 vox, 1332 in-brain) and
`..._R.nii.gz` (1364 vox, 1354 in-brain), provenance in
`Garvert_MTL_2mm_hemispheres.json`. No voxel sits at x = 0, and the two are
near-symmetric in size, so their FWE thresholds are comparable
(t_crit = 4.70 vs 4.67).

**Re-ran** `per_TR_svc_split_rew_DSR_allTR_2026-08-27` with 5 masks
(mPFC, MTL, MTL_L, MTL_R, visual). Same seeds/data, so mPFC / MTL / visual and
the whole-brain maps are unchanged; the folder now also holds the hemispheres.

**SVC, next_rew:**

| mask | peak t | TR | MNI | p_FWE | LOSO peak | LOSO p |
|------|--------|----|-----|-------|-----------|--------|
| MTL bilateral | 5.64 | 7 | -12/-38/-10 | .0116 | TR7 t=2.40 | .0510 |
| **MTL left** | 5.64 | 7 | -12/-38/-10 | **.0054** | TR8 t=3.07 | **.0124** |
| MTL right | 3.68 | 4 | 24/-36/2 | .3224 | TR2 t=0.53 | .6277 |

Splitting HELPS: the same peak voxel goes from p = .0116 bilaterally to
p = .0054 in the left mask, because halving the search volume lowers the
threshold while the effect is entirely on the left. LOSO likewise strengthens
(.051 -> .012, 3 significant seconds TR7/8/9).

**Direct L-R contrast** (`scripts/hemisphere_contrast.py`, new). "Significant
in L, not in R" is not a lateralisation test, so this tests L - R itself on the
per-subject LOSO held-out arrays (each hemisphere selected its own top-k on
n-1 subjects, so the paired difference stays unbiased), with the same
`null_max_t` sign-flip null corrected over the 12 seconds:

| channel | largest \|L-R\| | t(L-R) | p_FWE(L>R) |
|---------|---------------|--------|------------|
| **next_rew** | TR7 | **+2.83** | **.0243** |
| two_next_rew | TR10 | +1.91 | .1479 |
| three_next_rew | TR0 | -1.53 | .2648 (R>L) |
| curr_rew | TR0 | -1.22 | .4399 (R>L) |

So the left-lateralisation of `next_rew` is a real difference, not just a
difference in significance — L > R at TR7 (and TR8, p = .027), FWE-corrected
over seconds. No other channel is lateralised either way.

Left HC/EC t per second for next_rew: 0.14, 0.20, 0.51, 0.79, 0.99, 1.58,
2.36, 2.89, 3.07, 2.64, 1.85, 1.37 — a slow build peaking at TR7-8, i.e. during
the fast second pass, not when B is first shown at 1.5-3 s. Right HC/EC is flat
throughout (max \|t\| = 0.53).

**Figures:** `per_TR_timecourses_MTL_hemispheres.pdf/.jpeg` (+ `_peaks.csv`),
`hemisphere_contrast_MTL_L_vs_MTL_R.csv`.

**Caveat unchanged:** 8 models x 5 masks now, uncorrected across that family.
The lateralisation contrast is 4 tests; next_rew at .024 would not clear
Bonferroni over 4 (.0125). It confirms the direction of an effect selected on
other grounds rather than establishing it independently.

## 2026-08-27 (evening) — split_rew_DSR across all 12 TRs: no reveal staircase

**Data:** `group_RSA_split_rew_DSR_per_TR_glmbase_01-TR{0..11}_cropped` — all 12
TRs present and intact (unlike `instr_test_full`, which is still 11/12
truncated). Its `sub-XX_settings_summary.json` has no `data_rdm_scope` key, so
it ran the default **`across_only`**: every RDM cell is an across-half
comparison, which makes it free of the within-half same-stimulus confound that
limits the `full_no_diag` instruction models. Execution channels only — there
are no `_instr` models in this run.

**Analysis:** `svc_loso_batch.py`, 3 masks, 10 000 SVC perms, LOSO k=50/100/200,
whole-brain maps at 1000 perms. Output
`data/derivatives/group/per_TR_svc_split_rew_DSR_allTR_2026-08-27/`
(+ `per_TR_timecourses.pdf/.jpeg/_peaks.csv` from the new
`scripts/plot_per_TR_timecourses.py`).

**Timing ground truth.** `create_EVs_for_RDMs.py` builds the `01-TR{n}` EV as a
1-s boxcar at `instruct_start + n`, HRF-convolved by FEAT — so the TR axis is
neural seconds with no lag to add back. `show_rewards` in
`mc/latest_experiment/3x3_fMRI_part1.py` shows ONE reward at a time: A 0-1.5,
B 1.5-3, C 3-4.5, D 4.5-6, then a faster refresh A 6-7, B 7-8, C 8-9, D 9-12.

**Result: neither the sequential-reveal staircase nor a synchronous rise at
TR6.** LOSO t per second (k=100), mPFC:

| channel | TR0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| curr (A) | -1.25 | -2.08 | -1.66 | -0.92 | -0.51 | -0.59 | -0.81 | -0.92 | -0.90 | -1.13 | -1.46 | -1.60 |
| next (B) | -0.30 | -0.83 | -0.22 | 0.83 | 1.91 | **2.48** | 2.34 | 2.07 | 1.94 | 1.70 | 1.20 | 0.68 |
| two_next (C) | 0.10 | -0.03 | 0.97 | 2.27 | **3.45\*** | **3.47\*** | **2.82\*** | 2.15 | 1.23 | 0.16 | -0.36 | -0.37 |
| three_next (D) | 0.27 | 0.65 | 1.32 | **1.70** | 1.56 | 1.02 | 0.41 | 0.06 | -0.13 | -0.21 | -0.46 | -0.70 |

- `curr_rew` (reward A) is **never** represented — flat or negative at every
  second, in every mask.
- Peaks run D (TR3) -> C (TR4-6) -> B (TR5-8): if anything the REVERSE of the
  reveal order, and nothing like a double sweep.
- `two_next_rew` is the only channel with FWE-significant LOSO seconds
  (TR4/5/6, p < .05 corrected over the 12 seconds).
- MTL: only `next_rew`, SVC t = 5.64 at **TR7**, MNI -12/-38/-10,
  p_FWE = .012 — five seconds after B is first shown, but coincident with B's
  second appearance (7-8 s).
- Whole brain: only `next_rew` (t = 6.97, TR7, p_FWE = .014) and its combo
  regressor (t = 7.27, TR7, p_FWE = .010). Peak MNI -10/70/-4 is at the very
  anterior edge of the brain mask — check that it is not a smoothing/edge
  artefact before believing it.
- Occipital: nothing (best p_FWE = .144).

**The reported mPFC effect decomposes into the MIDDLE two channels.** The
published instruction-phase result is parent `rewDSR`, t = 5.08 at TR4,
MNI **-6/32/18**, p_FWE = .041. In the split:
`next_rew` t = 4.76 at TR4, MNI **-6/32/18** (p = .081), and `two_next_rew`
t = 4.51 at TR4, MNI **-8/32/18** (p = .113) — the same voxel and the same
second. `curr_rew` (t = 2.64, p = .93) and `three_next_rew` (t = 3.12, p = .74)
contribute nothing. So the mPFC effect is carried by rewards B and C, not by
where the subject is now and not by the last reward. Neither child alone beats
the parent, consistent with them contributing jointly rather than one driving it.

**Multiple comparisons:** 8 models x 3 masks plus 8 whole-brain tests, not
corrected across that family (per the standing request). Bonferroni over 8
models would need p < .00625; `next_rew` whole-brain (.010/.014) and MTL (.012)
do not clear that. Treat the decomposition as the robust part and the
individual p-values as suggestive.

**Still open:** the same split for the INSTRUCTION channels needs the
`instr_test_full` download to finish, and it will carry the within-half
confound, so it is not directly comparable to this run.

## 2026-08-27 (later still) — whole-brain t / FWE-p / uncorrected-p volumes added

**Script:** `scripts/svc_loso_batch.py`, new `--wholebrain` branch.

Motivation: the SVC test only ever reported numbers inside a mask, so there
was nothing to scroll through in fsleyes. `--wholebrain` now writes, per model,
in `wholebrain/`:

| file | what |
|------|------|
| `{model}_t.nii.gz` | observed group t, all brain voxels |
| `{model}_1minusFWEp.nii.gz` | 1-p, FWE over whole brain x all TRs (max-t null) |
| `{model}_1minusp_uncorr.nii.gz` | 1-p, uncorrected, that voxel's own permutation p |
| `{model}_summary.json`, `{model}_null_max_t.npy` | peak stats, the null itself |
| `wholebrain_summary_table.csv` | one row per model |

3-D for a single TR, **4-D (X, Y, Z, TR)** as soon as several TRs are passed —
so the fsleyes TR slider scrubs the instruction period. `--wholebrain-neg`
adds the negative-direction p maps; `--wholebrain-models` restricts which
models get volumes.

**Implementation notes.**
- One read per model still serves everything: with `--wholebrain` the
  extraction target becomes the whole brain mask and each ROI is a column
  subset of it, so adding whole-brain costs no extra I/O.
- `perm_wholebrain()` keeps only the max-t null plus a per-voxel exceedance
  tally, never the (n_perm x n_vox) null, so memory is set by the block size
  and not by n_perm. Blocks are sized to ~1e7 floats.
- The permutation t uses the same sign-flip identity as
  `svc_loso_test.null_max_t` (`var = (S2 - n*M^2)/(n-1)`). To make CLAUDE.md
  rule 4 checkable rather than assumed, the function **asserts** that the
  all-plus-one flip reproduces `tstat(D)` on the observed data — empirical and
  permutation statistic are therefore verifiably the same statistic.
- `n_p_FWE_lt_05` in the whole-brain summary is counted off the p map itself,
  not off the 95th-percentile t threshold; at low n_perm the two drift apart
  (1320 vs 1324 voxels at 200 perms) and only the former matches what you get
  by thresholding the saved map at 0.95.

**Interpretation guard rails, written into the docstrings.** The whole-brain
FWE null corrects over the entire brain mask AND every included TR at once, so
it is far stricter than the small-volume p in the mask folders and the two must
not be compared. The uncorrected map is for looking around, not for claims.
A voxel no permutation beat gets p = 0 / 1-p = 1; that means p < 1/n_perm, not
a real zero.

**Verified** (rewDSR_instr, TR5, 200-perm smoke run): map peak t = 8.8615 at
MNI -34/-2/-34 matches the summary json exactly; 147 358 in-brain voxels
match; uncorrected p <= FWE p at every brain voxel.

**TR5 re-run with the maps** (1000 whole-brain perms, 147 358 voxels; the SVC
mask numbers are unchanged from the earlier entry). 7 of 25 usable maps survive
whole-brain FWE, and every one of them is an `_instr` model:

| model | peak t | MNI | p_FWE | vox p_FWE<.05 |
|-------|--------|-----|-------|---------------|
| rewDSR_instr | 8.86 | -34/-2/-34 | <.001 | 1406 |
| REWDSR_INSTR-rewDSR_vs_instr | 8.86 | -34/-2/-34 | <.001 | 1406 |
| curr_rew_instr | 7.76 | -14/-22/10 | <.001 | 1086 |
| CURR_REW_INSTR-splitDSR_vs_instr | 7.10 | 46/-46/-14 | .001 | 394 |
| two_next_rew_instr | 6.77 | -12/-22/8 | .002 | 109 |
| next_rew_instr | 6.29 | -26/0/4 | .005 | 55 |
| three_next_rew_instr | 6.00 | -12/-22/10 | .008 | 68 |

No execution model comes close (best non-`_instr` whole-brain p = .129). Note
the peaks sit in thalamus / temporal pole / posterior insula rather than in any
of the a-priori regions — consistent with the within-half same-stimulus
confound noted in the entry below, and a further reason not to interpret these
until that control is run.

Output size: 33 MB of volumes for 27 models at one TR; expect ~0.4 GB and
roughly 45-90 min for the full 12-TR sweep.

## 2026-08-27 (later) — instruction models are degenerate in `across_only`, fine in `full_no_diag`

**Diagnosed on sub-02's actual model vectors** (`rewDSR` at the A_reward anchor,
`condition_files/rsa_instruction_full.json` settings).

**Why the across-half instruction RDM is constant.** Within one half, forw and
backw saw the SAME instructed sequence (backw reverses it mentally), so
`instruction_relabel_dict` is correct. But the same task letter is instructed in
the OPPOSITE order in the two halves — task A: half-1 instruction = 1,7,5,3,
half-2 instruction = 3,5,7,1. Reversing four distinct reward positions leaves no
slot matching, so every same-letter across-half cell is a Hamming mismatch (1),
and different-letter cells are 1 as well. The entire TH1 x TH2 block is
therefore uniformly 1.0 — zero variance, for `rewDSR_instr` and all four
`*_rew_instr` split channels alike.

**Consequence — `across_only` is dead for every instruction model.**
`evaluate_model_vec` zeroes a constant column, then `matrix_rank(XtX) < n_aug`
makes it return NaN for EVERY regressor in the design, and
`save_my_RSA_results` writes those NaNs as an all-zero map with no error. Ranks
in `across_only`: `rewDSR_vs_instr` 2/3, `instr_split` 1/5,
`splitDSR_vs_instr` 5/9 — all dead.

**`full_no_diag` is NOT ill-defined.** The within-half blocks W1/W2 carry the
instruction structure ("same letter within this half" = same stimulus = 0), so
the regressors have real variance (std 0.266 for `rewDSR_instr`, 0.344 for the
split channels) and every requested design is full rank: `rewDSR_vs_instr` 3/3,
`instr_split` 5/5, `splitDSR_vs_instr` 9/9, max |r| between regressors 0.644.
Instruction and execution are close to orthogonal there: r = -0.004 (rewDSR),
+0.020 / +0.109 / +0.109 / +0.020 (curr/next/two/three).

**But the identifying variance is entirely within-half.** All 100 across-half
cells sit at the same value, so no across-half pair can inform an instruction
beta — it is estimated only from the 90 within-half cells. Within a half,
"same instruction" is identical to "same task letter", i.e. the same visual
stimulus in the same run. That is precisely the shared-run-noise bias the
`data_rdm_scope` docstring flags and does not correct. The TR5 SVC result
(MTL t = 8.3, visual t = 6.2 for `rewDSR_instr`) is therefore most likely a
within-run same-stimulus effect and must not yet be read as instruction coding
in EC/HC. A within-half-only null (or splitting W1 vs W2) is the control to run.

**Cause of the two all-zero maps found in the earlier TR5 run.** Combo
`rewDSR_noInstr` = [`rewDSR`, `simple`]: `simple` is finite in only 30 of 190
lower-triangle cells, and on exactly those 30 cells `rewDSR` and `simple`
correlate at **r = 1.0** -> rank 2/3 -> NaN for both -> all-zero maps. Not a
bug in the group test. Do not re-run that combo as specified.

**Changes.**
- New `condition_files/rsa_instruction_within_and_across_th.json`
  (`name_of_RSA = within_and_across_th_intr-vs-exe`, `data_rdm_scope =
  full_no_diag`): 10 single models plus combos `rewDSR_vs_instr` [rewDSR,
  rewDSR_instr], `instr_split` [the four `*_rew_instr` channels], and
  `splitDSR_vs_instr` [all 8 exec+instr split channels, as in the old file].
  `simple` dropped.
- `scripts/fMRI_run_RSA_instruction.py`: new `design_rank_report()` plus a
  pre-flight loop that checks every single model and every combo for constant
  regressors / rank deficiency BEFORE any searchlight OLS runs, and raises with
  the offending list instead of silently writing zero maps. Verified: it passes
  all 13 designs of the new config and flags exactly `rewDSR_noInstr` in the old
  one.

## 2026-08-27 — SVC max-t + LOSO over all 27 maps of `instr_test_full`, TR5 only

**Scripts:** `scripts/svc_loso_batch.py` (new), `scripts/svc_loso_test.py` (patched
to resolve `.nii`/`.nii.gz`).
**Output:** `data/derivatives/group/per_TR_svc_instr_test_full_TR5_2026-08-27/`
— `settings.json`, `summary_table.csv`, `run.log`, and per mask/model:
`_svc_summary.json`, `_loso_results.json`, `_loso_k{K}.npy`, plus the volumes
`_t.nii.gz` (observed t), `_voxelFWEp.nii.gz` (voxel-wise FWE p against the
max-t null), `_voxel1minusFWEp.nii.gz` and `_voxel1minusFWEp_neg.nii.gz`
(threshold at 0.95 for p_FWE < .05), and `_null_max_t.npy` (the null itself,
for re-thresholding). 3-D here because one TR was analysed; 4-D (X,Y,Z,TR) when
several are. Neither `svc_loso_test.py` nor the first version of
`svc_loso_batch.py` wrote volumes — added 2026-08-27, verified to reproduce the
json peaks exactly.

**Test.** Identical to the reported instruction-phase test — one-sample t over
32 subjects, sign-flip max-t permutation (10 000 perms, seed 0) corrected over
all voxels in an a-priori mask, plus the LOSO cross-validated readout at
k = 50/100/200. `tstat`, `null_max_t` and the LOSO selection are imported from
`svc_loso_test.py`, so empirical and permutation values come from the same code.

**Masks (3).**
- `mPFC` = `masks/mask_PFC_LR_smoothed_resampled.nii.gz` (BA32/mBA9/mBA10), 4182 vox
- `MTL` = `masks/Garvert_MTL_2mm.nii.gz` (HC/EC), 2695 vox
- `visual` = `masks/visual_occipital_HO25_2mm.nii.gz` (NEW: union of the 8
  occipital Harvard-Oxford cortical labels at thr25 — LOC sup/inf,
  Intracalcarine, Cuneal, Lingual, Occipital Fusiform, Supracalcarine,
  Occipital Pole), 12 497 vox in-brain of 27 220 in the atlas. The 7T FOV
  truncates the occipital pole (3 % covered) and occipital fusiform (1 %), so
  this mask is effectively dorsal/medial occipital + LOC-superior.

**BLOCKER — only TR5 could be analysed.** Of the 12
`group_RSA_instr_test_full_glmbase_01-TR{n}_cropped` folders, only TR5 finished
downloading. All 26 beta maps in each of the other 11 folders fail `gzip -t`,
truncated at exact 256 KB boundaries (6–10 MB of an expected ~28 MB) — an
interrupted transfer, 275 corrupt files. So this run is `--trs 5`: max-t is
corrected over voxels ONLY, not over voxels x TRs, and there is no timecourse.
Re-run across all 12 TRs once the transfer completes.

**Two maps are entirely zero** and were dropped from the tables (25 of 27
remain): `REWDSR-rewDSR_noInstr` and `SIMPLE-rewDSR_noInstr` — i.e. the whole
`rewDSR_noInstr` combo model in `condition_files/rsa_instruction_full.json`
wrote all-zero output. `splitDSR_noInstr` is fine, so it is specific to that
combo. Upstream bug, not a bug in this test.

**Multiple comparisons.** By request, no correction across the family:
each p_FWE is corrected within its own mask only. 25 maps x 3 masks = 75 tests.

**Result — the `_instr` (visual instruction similarity) models dominate
everywhere.** All p_FWE below are within-mask, one-sided positive; LOSO p at
k=100.

| mask | top map | peak t | MNI | p_FWE | LOSO t | LOSO p |
|------|---------|--------|-----|-------|--------|--------|
| mPFC   | curr_rew_instr | 4.79 | -4/38/8 | **.0037** | 3.26 | .0010 |
| mPFC   | CURR_REW_INSTR-splitDSR_vs_instr | 4.70 | -8/42/10 | **.0051** | 3.02 | .0026 |
| mPFC   | rewDSR_instr (= REWDSR_INSTR-rewDSR_vs_instr) | 4.46 | 4/20/34 | **.0107** | 2.26 | .0151 |
| mPFC   | two_next_rew_instr | 4.28 | 4/48/36 | **.0208** | 3.56 | .0007 |
| MTL    | rewDSR_instr | 8.30 | -32/-2/-34 | **<.0001** | 7.18 | <.0001 |
| MTL    | two_next_rew_instr | 6.26 | -30/-24/-22 | **.0002** | 4.87 | <.0001 |
| MTL    | curr_rew_instr | 6.11 | -32/-22/-24 | **.0004** | 4.60 | <.0001 |
| MTL    | three_next_rew_instr | 5.75 | -24/-8/-24 | **.0006** | 4.04 | .0001 |
| MTL    | next_rew_instr | 5.32 | -22/-10/-28 | **.0023** | 4.53 | <.0001 |
| visual | curr_rew_instr | 6.06 | -10/-90/36 | **.0003** | 3.70 | <.0001 |
| visual | rewDSR_instr | 6.16 | 24/-84/26 | **.0006** | 3.82 | <.0001 |
| visual | two_next_rew_instr | 5.45 | -44/-62/2 | **.0030** | 3.81 | .0002 |

**Not significant anywhere:** every execution-similarity model on its own —
`rewDSR` (mPFC p = .74, MTL .86, visual .49), `simple` (.96 / .99 / .99),
`curr_rew`, `next_rew`, `two_next_rew`, `three_next_rew`, and all four
`*-splitDSR_noInstr` regressors. In the `*_vs_instr` combos the execution
regressor is likewise null while its `_INSTR` partner carries the effect —
i.e. at TR5 the instruction-similarity regressor explains the variance and
leaves nothing for the execution regressor.

**Caveat on the effect sizes.** MTL t = 8.3 and visual t = 6.2 for
`rewDSR_instr` are far above anything the execution models produce, and the
`_instr` models are uniform within (task_i, task_j) 2x2 sub-blocks by
construction. A block-structured regressor of that kind can be picked up by any
residual block structure in the data RDM (e.g. run/session or scanner-drift
structure aligned with task identity), so these should be read as "the
instruction-block model fits" and not yet as evidence about representation.
Worth a control before interpreting.

**Also recorded, descriptive only:** `peak_t_neg` / `p_FWE_neg` — the negative
peak against the same (symmetric) sign-flip null, a second one-sided test not
corrected for testing both directions. 11 of 75 cells have p_neg < .05, the
strongest being `NEXT_REW-splitDSR_noInstr` in MTL (t = -5.02, 32/0/-20,
p = .0035). At an uncorrected .05 with 75 tests, ~3.75 are expected by chance.

**Method note:** the group brain mask here is the intersection of
`mask_all_32_subjects` over all included TRs, not TR0's mask alone as in
`svc_loso_test.load_ref` (the per-TR group masks differ by ~21 voxels; a voxel
entering the max-t search must be valid at every TR the search runs over).

## 2026-08-09 (later) — YER micro positions reconstructed from macro probes

**Script:** `scripts/cell_to_roi_july26.py`

YER's v2026 file ships no `microwires` / `sEEG-micro` rows (202 rows, all
`Type == sEEG`, no `m`-prefixed labels, `NSxSource`/`NSxIndex` empty), so
its 26 cells were the last Baylor cells still on the pre-2026 big-table
macro position.

**The 3.15 mm constant.** Across **all 119 bundles in all 19 files that do
carry `microwires` rows**, the `microwires` position sits exactly
**3.15 mm** from the `sEEG-micro` position (min 3.15, max 3.15), always
*beyond* the probe tip along the insertion axis. Baylor is applying a
nominal Behnke-Fried protrusion, not localising wires individually.

**Validation.** Rebuilding each bundle from its macro probe as
`contact01 − 3.15 mm × unit(contact02 − contact01)` reproduces Baylor's
own supplied `microwires` coordinate to **median 0.25 mm, max 1.07 mm**
over the 113 checkable bundles (YEN excluded; its MNI152 is the corrupt
column). 112/113 within 1 mm. So this is Baylor's own construction, not
an approximation.
*(Sort key must be `Label` — `ElectrodeID` is a mixed int/str column and
sorts lexicographically, which silently picks the wrong contact.)*

YER's own MNI152 column is sound (median 0.00 mm from Fischl(MNI305),
sensible whole-head extent), so the rebuild uses it directly. The
supplied `YER_electrodes.pptx` independently confirms the probe
inventory: 6 probes labelled "microwire" — RT2cHbEb, RT2bHaEa, RF2aCa,
LT2cHbEb, LT2bHaEa, LF2Ca — matching the CSV `ProbeName`s.

Implemented as `reconstruct_micro_from_macro()`, applied only to files
with no micro rows at all, tagged
`baylor_v2026_micro_reconstructed_from_macro` with
`coord_verified = False` (inferred, not supplied).

**Result:** 26 YER cells move by median 3.15 mm (max 10.19 mm). **2 cells
change ROI** (`mLT2bHaEa03`, cells 1 & 2: HC_mid → HC_anterior; that
bundle moves 10.19 mm because the big-table macro coord was on a
different contact). No Baylor cell now uses the big table.

Cumulative vs the aug-09-2026 reference: 112/984 cells moved,
**3/984 changed `alt_final_roi`** (HC_anterior 275→276, HC_mid 232→231;
all other ROIs unchanged).

### How much does the MNI152 provenance actually matter?
Measured on the 78 cell-carrying bundles in the 18 subjects where both a
supplied MNI152 and MNI305 exist: supplied-152 vs Fischl(305→152) differ
by **median 1.79 mm, 95th pct 5.0 mm, max 9.0 mm**, and the ROI verdict
flips for **5/78 bundles = 50/608 cells (8.2 %)**. So coordinates derived
via the plain Fischl affine — YEN (broken 152, 35 cells), YER (26), and
the five affine-only files YEL/YEP/YEQ/YEU/YFT (146) = **207/669 Baylor
cells** — carry roughly a 2 mm positional and ~8 % ROI-label uncertainty
relative to a proper MNI152 normalisation.

## 2026-08-09 — Baylor v2026 electrode tables for YEL / YEN / YFT added to the cell→ROI pipeline

**Script:** `scripts/cell_to_roi_july26.py`
**Output:** `data/ephys_humans/derivatives/neurons_with_ROI_labels.csv`
**Reference (previous run):** `data/ephys_humans/derivatives/old_electrode_tables/aug-09-2026/neurons_with_ROI_labels.csv`
**Per-cell diff:** `data/ephys_humans/derivatives/ROI_assignment/cells_step8_change_vs_previous_run.csv`

### Source data
New `-electrodes_v2026.csv` files placed at the top level of
`data/ephys_humans/ABCD_pts_elecFilesForSvenja_v2026/`: **YEL, YEN, YFT**
(byte-identical re-sends of YEU and YFI arrived as `...[37].csv` /
`...[94].csv` and are ignored by the loader's `-electrodes_v2026.csv`
suffix filter — no content change). **YER** still contains no
`microwires` / `sEEG-micro` rows at all, so its 26 cells remain on
`baylor_bigtable_pre2026_macro_position`.

### Code changes
- `load_baylor_v2026` now reads micro-bundle rows of **either** `Type ==
  "microwires"` **or** `Type == "sEEG-micro"` (new helper
  `_micro_bundle_rows`). `microwires` (the bundle itself, label
  `mLT2bHb01`) wins; `sEEG-micro` (contact 01 of the carrying macro
  probe, label `LT2bHb01`, ~3 mm shallower) only fills bundles with no
  `microwires` row. Both are keyed on the m-prefixed bundle name.
  *In the current data this fallback never fires* — every bundle that
  has an `sEEG-micro` row also has a `microwires` row, so all coordinates
  still come from `microwires`.
- Fixed a pre-existing ordering bug: `alt_final_roi` was consumed by the
  step-7 RSA-ready plot but only assigned at the very bottom of the
  script (`KeyError`). It is now assigned in step 7 where
  `analysis_rois` is computed.
- New **step 8**: change report vs `REFERENCE_TABLE` (an archived copy of
  this script's previous output), printing coordinate shifts per subject,
  cells shifted > 10 mm, `alt_final_roi` counts old vs new, and the
  transition matrix.

### Result
984 cells, unchanged row count. 86 cells moved coordinate; all of them in
YEL (17), YEN (35), YFT (34).

| provenance | before | after |
|---|---|---|
| `baylor_v2026_bundle_micro` | 557 | 608 |
| `baylor_v2026_bundle_305to152_unreliable_file` | 0 | 35 |
| `baylor_bigtable_pre2026_macro_position` | 112 | 26 |

YEN's `MNI152_*` columns are internally inconsistent with its `MNI305_*`
columns (mean 55.6 mm, max 71.2 mm disagreement under the Fischl
305→152 transform), so the existing reliability gate correctly rejects
them and uses `MNI305 → 152` instead (`..._305to152_unreliable_file`).

**Coordinate sanity:** 85 / 86 moved cells shifted **3.13–3.15 mm** —
exactly the known macro-last-contact → micro-bundle offset, i.e. a small
local correction, not a relocation. The single exception is
`BY2-YEN`, electrode `mLT2bHb07`, cell idx 2, which moved 60.6 mm:
its big-table coordinate was `(32.3, -19.8, -17.9)` — the **right**-
hemisphere `mRT2bHb` coordinate — despite an `mL...` (left) electrode
label. The label-driven v2026 lookup places it at `(-28.1, -24.2, -14.9)`
together with its seven `mLT2bHb` siblings. This is a big-table
data-entry error being corrected, not a localisation change.

**ROI changes (`alt_final_roi`, the column read by
`mc.analyse.roi_relabel.relabel_per_cell`):** exactly **1 of 984** cells
changed label — the `mLT2bHb07` cell above, `HC_anterior → HC_mid`.

| ROI | before | after | Δ |
|---|---|---|---|
| HC_anterior | 275 | 274 | −1 |
| HC_mid | 232 | 233 | +1 |
| mOFC | 163 | 163 | 0 |
| mPFC | 155 | 155 | 0 |
| PCC | 61 | 61 | 0 |
| EC | 38 | 38 | 0 |
| (NaN / excluded) | 60 | 60 | 0 |

Downstream RSA results are therefore essentially unaffected; the value of
the update is 86 cells now sitting on verified micro-bundle coordinates
rather than inferred macro-contact positions.

---

## 2026-08-13 — Cell ↔ fMRI future-lag gradient: extensive exploration (mostly null)

**Question:** do human mPFC single units recapitulate the fMRI DSR "preferred
future angle" gradient — i.e. if we average cells, does anatomical position
predict their preferred spatial-tuning lag the way the fMRI angle map does?

**Scripts (all read `per_cell_ALL_ROIs.csv`, mPFC, 155 cells / 32 subjects):**
`cell_gradient_master_table.py` (master per-cell + group tables),
`gradient_brain_cells_by_lag.py` (MNE medial surfaces, cells coloured by lag),
`cell_gradient_principal_curve.py` (bent-axis sliding window + shift test),
`cell_gradient_split_table.py`, `cell_gradient_split_permgated.py`,
`cell_gradient_full_factorial.py` (320-row robustness grid).
Outputs under `data/ephys_humans/derivatives/group/cell_gradient_master/2026-08-13_09-45-28/`.

**Methods settled on:** pool the 12-lag profiles within a cell group, then read
the preferred lag off the pooled profile (argmax); per-cell argmax and the
continuous first-harmonic angle are both too noisy (harmonic vector length
≈ 0.07; per-cell harmonic angle ~ uniform). fMRI angle sampled at each cell
via symmetrised + 3 mm cos/sin smoothing + 6 mm sphere (quarters map).
Anatomical axis = PC1 of the gradient-mask voxels (folded x) = essentially
dorsoventral (loads [-0.04, -0.41, 0.91], r = 0.98 with MNI z).

**NULL / dead ends (do not re-run):**
- **Continuous gradient does not exist.** Spearman(future-score, arc-length on a
  bent principal curve) = +0.058, circular-shift p = 0.25, subject-bootstrap
  95% CI [-0.10, +0.19]. In the full factorial, 31/32 correlation configs are
  n.s. (r ≈ 0.0–0.17); ctrl-mode correlations ≈ 0 or negative.
- **Continuous first-harmonic angle** per cell is unusable (near-uniform;
  group bootstrap CIs span the whole circle).
- **Controlled tuning (`_ctrl` columns) shows nothing** reproducible — binned
  pooled lags are chaotic across weighting/gating (confirms prior expectation).
- **Subject-first vs cell averaging** does not stabilise; it only *diverges from*
  cell-weighting where the pooled profile is flat (peak r ≲ 0.05), i.e.
  divergence is a noise flag, not a fixable choice.
- **Perm-gating** (only cells significant at a lag contribute to that lag) does
  not sharpen results; it thins data (median 2–7 cells/lag) and only confirms
  the already-robust groups.
- A real **240° "backward" signal** sits mid-axis (pc1-Q2 / z-middle,
  peak r 0.085–0.10) — genuine, and it blocks any monotone ventral→dorsal ramp.

**The one robust positive:** under **noctrl**, the **future-end bin of the
gradient axis** pools to **60°**, invariant to cell-vs-subject weighting AND to
perm-gating: `all/pc1/half:end`, `all/pc1/quartile:Q3`, and `all/z/quartile:Q4`
all give 60/60/60/60, matching the local fMRI angle (~63°, err 3–9°; pooled
peak r ≈ 0.08–0.16). Interpretation is **local, not a gradient**: "cells in the
deep-future (dorsal/high-z) end of the DSR gradient prefer 60°, matching fMRI
there," NOT "a cell gradient mirrors the fMRI gradient."

**One nominally-significant correlation (EXPLORATORY — treat with caution):**
`noctrl / in_mask / z / perm-gated (≥1 sig lag) / cell-weighted`:
Spearman(future-score, MNI z) = **0.238, shift-p = 0.032, n = 54**. This is
1 hit in 32 tests and does **not survive subject-weighting** (r = 0.028,
p = 0.45), so it is not corrected-significant. Splitting those 54 cells at
median z: low-z (n=27) argmax 240°, future-score +0.00; high-z (n=27) argmax
**60°**, peak r 0.159, future-score +0.087 — the same "high-z → 60°" story.

## 2026-08-26 — SWR pipeline rewrite to Chen/Staresina standard: session audit (Milestone 0)

**Scripts:** `mc/analyse/swr_io.py` (new), `scripts/swr_audit_sessions.py` (new).
Outputs under `data/ephys_humans/derivatives/group/swr/`
(`session_manifest.csv`, `session_blocks.csv`, `session_manifest.json`, `settings.json`).

Motivation: comparison of the existing ripple pipeline against Chen, Staresina et al.
2025 (J Neurosci 45:e1502252025) found seven blocking defects. The most consequential
is that `identify_HPC_ripples.py:122` estimates the detection threshold *within each
cropped snippet*, which partially normalises ripple rate to be constant per snippet and
so suppresses exactly the between-window rate difference the planning hypothesis is
about. Also absent entirely: artifact/IED rejection, a line-noise notch, spectral
validation of candidate events. Full rationale in the plan file.

**Two loaders are currently broken and cannot have run.** `all_trial_times_{XX}.csv`
has **14 columns**, but `scripts/identify_HPC_ripples.py:57` and
`scripts/preprocess_LFP.py:128` assign 13 names, so `df.columns = column_names` raises
`ValueError: Length mismatch` on every session. The correct 14-name contract already
existed at `scripts/behaviour_summary.py:47` and `mc/analyse/helpers_human_cells.py:379`
— the 14th column is `correct`. `swr_io.BEH_COLS` now mirrors it.

**`correct` is per-repeat accuracy, NOT a "plan known" state.** Verified on s05: grids 2
and 9 contain errors *after* the first correct solve. The planning boundary is therefore
derived as a cumulative max within grid (`swr_io.load_behaviour` adds `plan_known`),
which is cleaner than the `found_first_D` heuristic at
`mc/analyse/ripple_helpers.py:69-72`, but the raw column must not be used as a
per-repeat planning state.

**The behavioural clock is continuous across recording blocks; the LFP files are not.**
**25 of 60 sessions are multi-block.** Block *k+1* always continues the block-*k* clock,
separated by a real recording gap: measured range **+7.3 s (s33) to +2910.1 s (s21, a
48-minute break)**; s18 and s27 have three blocks (+229.6 s, +148.8 s). This falsifies
the assumption at `scripts/preprocess_LFP.py:213`, which maps behaviour into block 2 by
subtracting the *file duration* — valid only if recording never stopped. Block offsets
must be estimated per session against three independent references and hard-validated
(every behavioural event inside `[0, duration_k]` with >=5 s margin, else the block
emits no ripples). This is now the highest-risk item in the build; a 20 s error
misassigns every block-2 ripple systematically, which would read as a null rather than
as noise.

**Subject clustering needs a normalised key.** 63 sessions map to 43 distinct
`Subject Label` values but fewer real subjects: s29 is `'UT1-202314'` and s30 is
`'UT202314'` — the same patient in two of the four Utah label formats. 16 labels span
more than one session (`BY2-YEK` = s07/08/09; `BY2-YEX` = s43/44/49). Clustering
robust SEs on the raw label is anticonservative. `swr_io.normalise_subject_key` collapses
them and the audit prints the full map for manual sign-off.

**Audit result (run locally, 60 sessions):** 24 `ok`, 4 `needs_review`, 32
`no_raw_files`. Sites: baylor 36, utah 18, ucla 6. The 32 are simply not on the laptop
— **the audit must be re-run on ceph**, where those sessions' config defects will
surface. The 4 genuine structural defects to resolve by hand: **s03** (UCLA, 2 `.ncs`
recording blocks not represented in the YAML at all), **s18** and **s28** (duplicate
block names in `blocks`), **s32** (yaml=2 blocks, behaviour=1, files=2).

**Config YAML is demoted to a hint.** It disagrees with behaviour and with disk for a
large minority of sessions (`segment: null` for 13; no `blocks` key for s57/58/59;
duplicate block names; block counts that do not match). `session_manifest.csv` is now
the authority for everything downstream.

**NULL / dead ends (do not re-run):**
- Do not use `scipy.signal.resample` on continuous traces — it is FFT-based and wraps
  the end of the recording into the beginning. Silent, and visible only at the two ends.
  Use `resample_poly`. This is also the source of the per-snippet edge ringing in the
  old pipeline.
- Do not downsample to 500 Hz (`preprocess_LFP.py:30-32`). It makes Chen's >250 Hz RMS
  artifact criterion and the 120-200 Hz spectral rejection impossible. Keep 1000 Hz.
- Do not store full stepwise TFR power. The existing `ripple_power_dict_s05` is **5 GB**
  and is not needed by anything.

## 2026-08-26 (later) — Block structure resolved from the raw data, not the config

**Script:** `scripts/swr_diagnose_blocks.py` (new). Reads every neo segment's duration
and channel count out of the raw files and matches them against behavioural block spans.
Written because excluding sessions on a config warning discards data that the recordings
themselves can disambiguate.

**s18 and s28 are fine — the "duplicate block names" warning was a false alarm.** The
duplicates are *filename labels*, not duplicate recordings. Matching by duration is
unambiguous:

| session | behavioural block span | matched recording | slack |
|---|---|---|---|
| s18 | 363.3 s / 4006.1 s / 3432.4 s | EMU-058 seg1 / EMU-059 seg1 / EMU-060 seg0 | +197.0 / +33.9 / +226.4 s |
| s28 | 1819.7 s / 654.6 s / 927.4 s | EMU-045 seg1 / EMU-047 seg1 / EMU-048 seg1 | +41.6 / +58.7 / +36.1 s |

Ordering by **EMU number** (the acquisition counter) is the reliable chronological key;
`blk-NN` labels are not. Note each Baylor file carries a ~2.3 s stub segment plus the
real recording; picking the longest segment per file independently reproduces the YAML
`segment` field for both sessions, so that field is trustworthy where it is populated.

**s32 has a real 444.8 s hole in the middle of a single behavioural block.** Behaviour is
one continuous block (101.8 -> 3883.0 s, 304 repeats). The recording is two files with
**non-zero `t_start` on a common amplifier clock**: 12042.7 s + 1977.5 s, then 14465.0 s
+ 1516.1 s. Recorded wall span 3938.4 s brackets the behavioural span 3781.2 s, but
~445 s of task has no LFP. Usable if the repeats falling in the hole are dropped.

**`t_start` is the amplifier wall clock and gives exact inter-block offsets where it is
populated** (s32: yes; s18/s28: zero). This is the "header wall-clock" offset estimator,
and it is the principled replacement for the file-duration subtraction at
`scripts/preprocess_LFP.py:213`. It is also what the `t_start > 10000` hack at
`preprocess_LFP.py:76-83` was groping at.

**s03 (UCLA) is a SINGLE-block session and is usable.** 164 of its 308 `.ncs` files are
**header-only 16384-byte stubs** with zero data records ("TimeClosed File was not closed
properly") — an aborted first recording. All real data is in the `_0001.ncs` files, which
include both macros (`LMH*`, `RMH*`, `LA*`, `LAC*`, `ROF*`, `RPH*`) and micros (`GA*-`).
Counting the stubs as a second block is what made the YAML look inconsistent.
`swr_io.discover_raw_files` now skips any `.ncs` of exactly the header size.

**UCLA sampling rate in the YAML is wrong.** `config_human_ABCD_iEEG.yaml` declares
`sampling_rate: 1000` for all six UCLA sessions. The s03 `.ncs` headers report
**macros at 2000 Hz** (micros at 32000 Hz). Read the rate from the file header, never
from the config.

**neo cannot read these `.ncs` files.** `NeuralynxRawIO.parse_header()` raises
`TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'` at
`neuralynxrawio.py:505` because `global_t_start`/`global_t_stop` are None. A direct
header/record reader (16 KB header, 1044-byte records of
`uint64 ts, uint32 ch, uint32 fs, uint32 n_valid, int16[512]`) works and additionally
exposes the Unix-epoch record timestamps needed for block alignment. UCLA support must
not depend on neo.

**Audit after these fixes: 26 `ok`, 2 `needs_review` (s03 informational, s32 the 445 s
hole), 32 `no_raw_files` (not on the laptop; re-run on ceph).** s18 and s28 moved to `ok`.

**NULL / dead ends (do not re-run):**
- Do not order Baylor blocks by the `blk-NN` label or by the YAML `blocks` list. Both
  contain duplicates. Order by EMU number.
- Do not use `neo.io.NeuralynxIO` / `NeuralynxRawIO` on the UCLA data in this env; it
  raises before returning anything. Parse the `.ncs` header directly.
- Do not trust `sampling_rate` in the YAML for UCLA (says 1000, actually 2000).

## 2026-08-26 (later still) — Anatomy loaders extracted to shared modules (Milestone 1)

**New:** `mc/analyse/anatomy_sources.py` (coordinate sources, all three sites),
`mc/analyse/anatomy_atlas.py` (MaxProbAtlas + the ROI rule ladder).
**Changed:** `scripts/cell_to_roi_july26.py` now imports them; 2514 -> 1880 lines.
**Unchanged:** every output of the cell pipeline.

Rationale: the LFP/ripple pipeline needs to pick hippocampal and mPFC *macro*
contacts by anatomy rather than by string matching (the old
`'H' in channel and 'T' in channel`, and `'1' in label` which also matches
HIP10/11/14). Rather than duplicating the rules, both pipelines now call one
implementation — so "the same anatomical criteria defined HC for cells and for LFP
contacts" is literally true, and a future change to `assign_atlas_roi` propagates to
both instead of silently diverging.

The move was verbatim: `sed`-extracted line ranges 221-617 (loaders) and 982-1029 +
1040-1245 (atlas layer), not retyped. Two structural edits only:
- `discover_utah_mats()` took two module globals of the script; it now takes them as
  keyword arguments defaulted to the same literal paths, so the bare call still works.
- The four atlas objects were built by `nldatasets.fetch_*` at import time; they are
  now built by `get_atlases()`, which populates the same module-level names.
  `assign_atlas_roi` still reads the module globals `juelich` and `HC_ANT_MID_Y`, so
  its body is byte-identical.

**Regression gate — PASSED.** Procedure: (1) confirm the *unmodified* script is
deterministic by re-running it and checking it reproduces its own output
(md5 `e0e758a303831cfc614a2490dcaf6aac`) — it does, so a post-refactor diff cannot be
blamed on pre-existing nondeterminism; (2) refactor; (3) re-run.

`derivatives/neurons_with_ROI_labels.csv` is **byte-identical** after the refactor, as
are all nine `ROI_assignment/cells_step*.csv` intermediates. Baseline copies kept at
`/tmp/swr_refactor_baseline/` for this session.

Note the gate must be against a *fresh run of the current script*, not against
`old_electrode_tables/aug-09-2026/` — that archive legitimately differs
(md5 `9ebfceda...`, 205394 bytes vs 204383) because new electrode files arrived since.

**NULL / dead ends (do not re-run):**
- A static AST check (names used by the script that are defined in the new modules but
  not imported) caught `_bundle_key`, which a syntax check cannot: `py_compile` passes
  on an undefined global. Use the AST check, not `py_compile`, when moving definitions
  out of a flat script.
- The new modules are deliberately NOT added to `mc/analyse/__init__.py`, which imports
  eagerly — adding nilearn/neo-heavy modules there would make every `import mc` in the
  repo pay for them. Import explicitly.

## 2026-08-26 (Milestone 2) — Macro-contact anatomy: hippocampal LFP contacts selected by anatomy, not string matching

**New:** `mc/analyse/contact_anatomy.py`, `scripts/swr_build_contacts.py`.
**Outputs:** `derivatives/s{XX}/LFP/macro_contacts_{XX}.csv` + `bipolar_pairs_{XX}.csv`
per session; `derivatives/group/swr/macro_contacts_all.csv`, `contact_qc.csv`,
`settings.json`.

Replaces contact selection by string matching — the old `'H' in channel and 'T' in
channel` (matches any label containing both letters) and `'1' in label` (also matches
HIP10/11/14) — and the hard-coded contact01−contact04 bipolar pair spanning ~15 mm.

**Result: 160 hippocampal contacts / 164 hippocampal bipolar pairs across 27 sessions**
(baylor 94, utah 60, ucla 6). The remaining 33 sessions have no raw data on the laptop;
re-run on ceph.

### Utah `ElecMapRaw` indexing — a trap that would not have raised
`ElecMapRaw` is an (n, 3) object array `[label, amplifier_channel, other_channel]`.
Coordinates must be read by the **direct row index** into `ElecXYZMNIRaw` /
`ElecAtlasRaw` / `ElecTypeRaw` — **not** by indexing with the channel number.
Verified on s02: row 41 is `bRAHIP2`, and `ElecAtlasRaw[41]` = "Right Hippocampus",
whereas indexing by column 1 gives "Right fusiform gyrus" — a different electrode
entirely, silently. Column 1 is the amplifier channel and matches
`utah_elec_labels_{XX}.csv` for all 114 rows where it is finite (the other 30 are
unlocalised and are dropped with a logged reason).

Also: `m*` = microwire, `b*` = the Behnke-Fried macro on the same shaft, bare = plain
sEEG depth. Macros use `ElecXYZMNIRaw`, not `ElecXYZMNIProj` — gray-matter projection
is right for a spiking micro tip and wrong for a 2 mm macro ring that legitimately
straddles the GM/WM boundary. Both are stored.

### ROI definition: native-space primary, shared atlas ladder as cross-check
Measured on s05/s02/s03, `anatomy_atlas.assign_atlas_roi` has **100% recall but 22%
precision** against the subjects' own segmentations: it calls 49 contacts hippocampal
where the native labels say 11, the excess being amygdala (`RAMG1-4`), lingual gyrus
(`RPHIP1-2`), VentralDC (`RT1cCM04`) and parahippocampal cortex. Expected — that ladder
was tuned for microwire *tips*, where ±2–3 mm neighbourhood rescue is desirable, and is
applied here to MNI152, where MTL registration error is large.

So a macro contact's ROI comes from the **subject's own segmentation**, matching Chen
et al., who identified hippocampal contacts "via visual inspection of postoperative
T1-weighted anatomical MRI scans". The shared ladder is still computed and stored as a
cross-check; because it is a strict superset, only the reverse disagreement is
meaningful. The cell pipeline is unaffected — `assign_atlas_roi` is untouched.

**Use the 3 mm-neighbourhood parcellation, not the single-voxel one.** For Baylor,
`ROI_DK2005_3mm` / `Matter_3mm`, not `Area_fs_vox` / `Matter_fs_vox`. A macro contact is
a ~2 mm ring recording from a volume, so the label must be sampled at that scale. On
YEJ the voxel column calls 1 contact hippocampal and the 3 mm column calls 4 — e.g.
`RT2bHaEa02` is `Right-Hippocampus` at 3 mm and `Right-Cerebral-White-Matter` at the
voxel. Voxel columns retained as `native_region_vox` / `matter_vox`.

**Pair rule:** a bipolar pair is hippocampal if **at least one** of its two contacts is
natively hippocampal — the standard montage (Chen take the most medial hippocampal
contact and its immediate neighbour, frequently white matter). Requiring both would
have given s05 zero pairs.

### Gate — PASSED (5/5)
1. **Probe names plausible.** baylor `RT2bHaEa02`, `LT2HbE02` (Ha/Hb); utah `LAHC1`,
   `bRAHC1`, `LPHC1`; ucla `LMH-1`, `LPH-1`, `RMH-1`.
2. **Atlas concordance 158/160.** The 2 exceptions are `LT2aA01` in s31 and s35 (same
   subject YFF): native says Left-Hippocampus, atlas says EC — an amygdala-probe
   contact 01 sitting on the HC/EC border. Borderline, not a defect.
3. **UCLA independent ground truth: 6/6.** Every UCLA HPC contact carries an ASHS
   hippocampal subfield (DG, CA1, SUB, CA2), and **0 contacts with an ASHS subfield
   were missed**. ASHS is a completely independent segmentation — this is the strongest
   validation available.
4. **Anterior/mid split clean.** HC_anterior y ∈ [−20.8, −3.8]; HC_mid y ∈ [−34.4,
   −21.4]. No overlap at HC_ANT_MID_Y = −21.0.
5. **Bipolar spans sane.** Median 4.68 mm (range 3.08–7.75), i.e. genuine adjacent sEEG
   spacing, versus ~15 mm for the old 01−04 pairing.

**Open point for Milestone 3/4:** adjacent pairs share contacts — pair (2,3) and pair
(3,4) both contain contact 3 — so the 164 pairs are not independent (median 5 per
session, max 11; Chen used ~2 per participant, one per probe). Decide before detection
whether to keep all pairs, or one anchor pair per probe.

**NULL / dead ends (do not re-run):**
- Do not index Utah coordinates via `ElecMapRaw` column 1. It returns another
  electrode's position and raises nothing.
- Do not use `Area_fs_vox` / `Matter_fs_vox` for macro contacts — single-voxel labels
  understate hippocampal coverage ~4x.
- Do not merge anatomy to channels on a key that can be NaN: pandas matches NaN to NaN,
  and s02 fanned 132 channels out to 248 rows before the null-key guard.
- Do not use the atlas ladder alone to select macro contacts (22% precision).

## 2026-08-26 (CORRECTION) — Block alignment is deterministic; the earlier "highest risk" entry was wrong

**This corrects the 2026-08-26 entry above**, which claimed the behavioural clock
includes recording gaps the LFP files do not, and that
`scripts/preprocess_LFP.py:213` (mapping block *k* by subtracting cumulative file
duration) was therefore "badly wrong". **That claim was mistaken. The old approach is
correct.** Verified rather than assumed, via `/tmp/blockfit.py`.

**The behavioural clock is cumulative file duration.** For block *k*,
`offset_k = sum(durations of files 0..k-1)`, and behavioural times map into file *k* as
`t - offset_k`. Tested on all **14 multi-block sessions with local raw data**: every
behavioural event lands inside its file, with head/tail margins of seconds.

s18 worked out exactly: file durations 560.3 / 4040.0 / 3658.8 s give offsets 0 /
560.3 / 4600.3; behavioural block 2 (580.8–4586.9 s) maps to in-file 20.5–4026.6 s
against a 4040.0 s file. The apparent 180.1 s "behavioural gap" I reported earlier is
just the 159.6 s tail of file 1 plus the 20.5 s head of file 2 — **not** a recording gap.

**Wall-clock timestamps are irrelevant, and would have been actively misleading.** The
NSx 2.2/2.3 basic header does carry a `TimeOrigin` SYSTEMTIME at byte offset 294 (neo
reports `rec_datetime = None`, but it parses directly). For s18 those wall clocks are
1943.4 s and 12459.2 s apart, versus file durations of 560.3 and 4040.0 s — i.e. real
elapsed gaps of 20–200 minutes between recordings. Had I anchored blocks to wall clock,
every block-2 ripple would have been misplaced by tens of minutes. The behaviour was
timestamped against the *concatenated* recording, not against wall time.

Two NSx variants exist: `NEURALCD` (2.2/2.3, has TimeOrigin) and `BRSMPGRP` (s32, s33 —
different layout, no TimeOrigin at that offset; neo reports a non-zero `t_start`
instead). Neither is needed for alignment.

**Three sessions overrun by 1–2 s, each by exactly one repeat.** s09 block 2 (−2.0 s),
s10 block 2 (−1.8 s), s33 block 1 (−0.9 s) — 1 repeat of 275, 274 and 296 respectively.
The amplifier was stopped a second or two before the last trial completed. Handle by
dropping the overrunning repeat and logging it; this is not a misalignment (a real one
would be off by tens or hundreds of seconds, not by one trial).

**Consequence for the plan:** block offsets need no three-way estimation and no
wall-clock anchoring. The rule is `cumsum(file durations)`, with a hard validation that
every behavioural event lands inside its file (margin ≥ 0, allowing a single trailing
repeat to be dropped). This removes what I had flagged as the highest-risk item.

**NULL / dead ends (do not re-run):**
- Do not align blocks by NSx `TimeOrigin` / wall clock. It is present and parseable but
  describes real elapsed time between recordings, which the behavioural clock excludes.
  Using it would misplace block 2+ by tens of minutes.
- Do not treat the behavioural inter-block "gap" as a recording gap. It is the tail of
  one file plus the head of the next.

## 2026-08-26 (Milestone 2, CORRECTION) — One bipolar pair per probe, not every adjacent pair

**Corrects the pair counts in the Milestone 2 entry above (164 pairs).** The final
number is **76 pairs across 27 sessions / 18 subjects**.

`build_bipolar_pairs` was generating *every* adjacent pair containing a hippocampal
contact. That over-generated (median 5 per session, max 11) and made the derivations
non-independent: contact 3 appeared in both pair (2,3) and pair (3,4), which the GLM
cannot account for.

Corrected to follow Chen et al. exactly — *"bipolar referencing was performed using the
most medial hippocampal contact and its immediate neighbour (i.e. the second-most
medial contact) **on each hippocampal probe**"* — i.e. **one derivation per probe**, so
no contact is ever reused. Chen report 34 contacts across 17 patients (~2 each); we get
76 across 27 sessions (~2.8 each).

**Anchor = most medial hippocampal contact on the probe, defined geometrically as
min |MNI x|**, not by contact number. Numbering conventions differ between sites
(Baylor contact 01 is deepest; Utah/UCLA differ), so a geometric definition of "medial"
is convention-free.

**Reference rule = immediate neighbour** (`scheme='neighbour'`, the default). Measured
alternatives on the full contact table:

| scheme | pairs | sessions | median span | ref in MTL grey | probes dropped |
|---|---|---|---|---|---|
| `neighbour` (Chen) | 76 | 27 | 5.28 mm | 49/76 | 0 |
| `white_matter` | 42 | 23 | 10.15 mm | 0/42 | 34 (no WM contact on probe) |

The white-matter montage gives a cleaner subtraction (no ripple signal in the
reference) but doubles the inter-contact distance, which enlarges the lead field,
degrades spatial specificity, and would *increase* volume-conduction confounds for the
H2 HC–mPFC analysis — the opposite of what that analysis needs. It also drops 34 probes
and 4 sessions. `scheme='white_matter'` remains available for a sensitivity analysis.

The 49/76 references that are themselves MTL grey are expected and are what Chen's own
montage produces (the second-most medial contact on a hippocampal probe is usually
still hippocampus). Ripples are spatially local at the millimetre scale, so adjacent
contacts 5 mm apart do not see identical events and the bipolar preserves them. If
common-mode cancellation turns out to matter, it will show up at the Milestone 4
checkpoint as reduced ripple amplitude or rate.

**NULL / dead ends (do not re-run):**
- Do not generate all adjacent pairs. It inflates n several-fold with non-independent
  derivations sharing contacts.
- Do not pick the anchor by contact number; numbering conventions differ per site.

## 2026-08-27 — Coordinate provenance audit of `cell_to_roi_july26.py`

Full trace of where every one of the 984 cell coordinates in
`neurons_with_ROI_labels.csv` comes from. Written up in
`docs/coordinate_provenance_audit.md`.

**Clean (906 / 984, 92 %):** 608 Baylor from the v2026 `microwires` row as shipped;
140 UCLA big-table coords independently corroborated ≤ 0.5 mm against the v2026
xlsx; 97 Utah reconstructed from the patient's own `Electrodes.mat`; 35 Baylor
MNI305 re-transformed (file failed its own 152-vs-305 gate); 26 Baylor inferred
from the macro probe with Baylor's own 3.15 mm protrusion constant.

**Not clean (78 cells, 7.9 %):** `utah_bigtable_recon_disagrees_gt3mm`. When the
reconstruction from the patient's own `.mat` disagrees with the hand-entered big
table by > 3 mm, the code keeps the **big table** and discards the reconstruction.
Median disagreement **37.9 mm**, max 62.3 mm; 73/78 exceed 10 mm. These currently
contribute mOFC 48, mPFC 18, HC_anterior 9, EC 3.

**Placeholder coordinate found:** 17 cells across UT1-202418 / UT1-202422b /
UT1-202503 all sit at the single point (4.55, 29.50, -20.63), which the atlas
calls mOFC. Those subjects (s54, s53, s55, plus s52 = UT202421) have **no
`Electrodes.mat` anywhere** — the four genuinely missing Utah files. s30 and s42
also lack one but are the same patients as s29 and s41.

**Root cause:** `discover_utah_mats()` coord-matches 3–16 big-table cells against
every folder's electrode pool with no uniqueness constraint, so s47's file was
assigned to six different patients. Measured: **8 of 12 Utah subjects match their
own `s{NN}` folder at 100 %** — the folder numbering is reliable and the
coord-matching was unnecessary. It is also circular: it validates the big table
against files using the big table's own (hand-entered) coordinates as the key.

**NOT changed:** the cell pipeline's behaviour is untouched — fixing items 1–3 in
the audit would change published cell ROIs and is the user's call. Only the
docstring was corrected (it claimed the reconstruction is preferred above 0.5 mm;
the code does the opposite at 3 mm) and an invented rationale was removed from
`discover_utah_mats()`.

**Already fixed in the SWR pipeline:** `mc.analyse.contact_anatomy.resolve_utah_mat`
resolves by folder (own → same patient → exclude with a stated reason), so no
session inherits another patient's electrodes.

### 2026-08-27 (addendum) — is the Utah .mat reconstruction trustworthy?

Validated `build_micro_map` against two independent signals: the microwire label
(`LabelMap`) vs the coordinate (`ElecXYZMNIProj/Raw` via `MicroElec`). Hemisphere
agreement **245/264 = 92.8 %**, region agreement **185/256 = 72.3 %**, against
chance of ~50 % and ~15 %. **The reconstruction method is sound** — but not
uniformly, and three subjects fail individually.

- **s23 (UT1_sj202309) is broken**: hemisphere 8/24, region 0/24. `MicroElec` empty,
  uses the `MicroElecRaw` fallback, labels misaligned to coords (`mLHIP1-8` land at
  x ≈ +8, which is not hippocampus in either hemisphere). Exclude, don't guess.
- **s47 and s39**: hemisphere 24/24 but region 8/24 — needs a look.
- **Resolving by folder fixes UT1_sj202308**: 1/16 → 16/16 hemisphere correct. The
  published laterality was right; the coord-matched file assignment was wrong.
- **MATLAB v7.3 (HDF5) files are silently unreadable**: `_load_mat` returns
  `LabelMap` as bare `None` because string cells are HDF5 object references that are
  never dereferenced. Affects s48, s52, s54, s55 — all four yield ZERO microwires
  and always fall back to the big table. Three of them are the placeholder-coord
  subjects.
- **Files after re-download**: s52, s54, s55 are present but under `Registered/` /
  `Registered-selected/`, which the loader never searches (it only looks in
  `electrodes/`). Only **s53 (UT202422b)** is genuinely absent. s30/s42 lack a file
  but are the same patients as s29/s41.

**UCLA is clean.** `load_ucla_v2026` reads the right sheet (`Sheet1`, the second
sheet, carrying `MNI_x/y/z` + `isMicro`). It does not filter on `isMicro`, but all
**140/140** UCLA cells match a microwire row at ≤ 0.5 mm and none match a macro —
so the missing filter is latent, not an actual error.

Full detail incl. per-subject tables: `docs/coordinate_provenance_audit.md`.
No code paths changed in this session; docstrings only.

### 2026-08-27 (addendum 2) — Utah coordinates: 168/175 now read directly

**Fixed the v7.3 read bug.** `_load_mat` returned `LabelMap` as a bare `None` for
MATLAB v7.3 files because cell-array strings are HDF5 object references that were
never dereferenced. s48, s52, s54, s55 therefore yielded ZERO microwires and always
fell back to the hand-entered big table. Now dereferenced.

**Files declare their own identity.** Every Utah `.mat` carries `Fname` (the original
acquisition path, e.g. `D:\Data\UIC202311\...`). Added `mat_patient_id()` to read it,
which removes the need for coord-matching entirely. Two mismatches found:
`s47` holds patient **202311**'s data (not 202302 — it is the v7 export of the same
165 electrodes as s48), and `s53`'s newly-downloaded file is a **duplicate of s52**
(202421, not 202422b). So UT1-202302 and UT1-202422b have no electrode file.
NOTE these files contain patient *names* in `PatientIDStr` — use the numeric ID.

**Removed the ordering assumption.** `build_micro_map` inferred the label↔coordinate
pairing by sorting microwires by amplifier channel against `MicroElec` (validated on
only s02/s06; failed on s23). Every file also has `MicroElecRaw` + `ElecMapRaw` +
`ElecXYZMNIRaw`, which indexed by the same row give label and coordinate together.
Added `build_micro_label_map()`. Validated across all 15 files: **352/360** microwires
have `sign(MNI_x)` matching the `mL`/`mR` in their own label (the 8 exceptions are
OFC within 2.5 mm of midline). **s23 goes from 8/24 to 24/24.**

**Census across all 984 cells:** 776 read directly from a site file; 140 (UCLA)
verified identical to the site file at ≤0.5 mm; 61 (Baylor) derived by a documented
transform; **7 guessed because no source file exists** (UT1-202302 ×3,
UT1-202422b ×4) = 0.7 %. Utah went from 97/175 to **168/175** readable.

**Not yet wired into `cell_to_roi_july26.py`** — that changes published cell ROIs.

### 2026-08-27 (addendum 3) — full coordinate rebuild from site files only

Re-derived every cell coordinate from the recording site's own electrode file.
Output: `derivatives/ROI_assignment/coordinate_rebuild_2026-08-27/`.

**Baylor and UCLA do not move (0.00 mm)** — the published table already used file
coordinates for both. Every change is Utah: UT1-202503 (52.8 mm), UT1-202418
(42.8), UT1-202421 (42.7), UT1_sj202308 (17.5), UT202314 (2.7), UT202413 (2.7),
UT1-202311 (1.3); all other Utah subjects 0.00.

**UT1_sj202309 (s23) does not move.** Its published coordinate already equalled the
direct read, so the hand-entered table was right for s23 and the old ordering-based
reconstruction was what was wrong — the old code kept the big table there for the
wrong reason.

**26 of 977 cells (2.7 %) change atlas ROI, all Utah:** mOFC→HC_anterior 12,
mOFC→mPFC 6, EC→HC_anterior 3, mPFC→mOFC 3, mOFC→HC_mid 2.

**New alt_final_roi counts (tier C, only the 7 no-file cells excluded):**
EC 38→35, HC_anterior 276→291, HC_mid 231→233, PCC 61→61, **mOFC 163→139**,
mPFC 155→158. 924→917 cells kept.

Only two non-as-shipped sources remain, both flagged: BY2-YEN (35 cells, the file's
own MNI305 through the Fischl affine) and BY2-YER (26 cells, macro + 3.15 mm, where
3.15 mm is Baylor's own constant, identical across all 119 ground-truth bundles and
reproducing them to 0.25 mm median). Dropping YER costs 16 mPFC cells (158→142).

**NULL / dead end (do not re-run):** coord-matching electrode files against the
big table. It is circular — the big table's coordinates are the thing in question —
and with 3-16 cells per subject it has no discriminative power. Use `Fname`.

## 2026-08-27 — Single-unit QC audit

`abcd_passed.mat` verified as `abcd_data_08-Sep-2025.mat` filtered to the QC-passing
units: 63 sessions, 1042 -> 984 (58 removed). Every retained spike train and every
`regionLabel` is byte-identical to the source — neural data and labels both untouched.

**Exclusions:** 36 for < 300 spikes (median 233, range 110-296); 22 as duplicates at
zero-lag r >= 0.50 on 100 ms bins. **All 22 duplicate pairs are within the same
microwire bundle and share a region label** (21/22 by bundle key; the 22nd is adjacent
contacts of one UCLA probe). That is the signature of one neuron on two wires, so the
criterion is doing what it claims — worth stating explicitly in the methods.

Retained population: median 4118 spikes, 10th pct 747, median FR 1.56 Hz, median RPV
0.00%. All three criteria assessed as sound; see `docs/cell_qc_methods.md`.

**Two problems found:**
1. The manuscript states duplicates were removed at *r = 70*. The run actually used
   **r >= 0.50**. At 0.70 only 9 units would have gone rather than 22. Methods text
   needs correcting.
2. **The QC code that produced `abcd_passed.mat` is not in the repo.** Its recorded
   settings include `MinOverallFR_Hz` and `SessionLowFR_Hz`, which appear in none of
   the four QC .m files and in no git commit. Decisions survive in
   `qc_all_sessions.mat`, but the pipeline is not reproducible as it stands.
   Separately, `qc_master_summary.txt` (Aug 2025) passed only 347/924 = 37.6% at
   nominally identical thresholds — almost certainly faulty; mark it superseded.

**New:** `scripts/plot_cell_qc_figure.py` -> `derivatives/group/cell_qc/`
(publication figure + per-cell metrics CSV + settings.json). Reads the stored QC
metrics; recomputes nothing.

Note: 924 in the manuscript is NOT the QC output (984) — it is what survives the
>= 3-subject ROI rule afterwards. The Aug QC run coincidentally had 924 as its
denominator; do not conflate them.

### 2026-08-27 — QC is reproducible again: `scripts/run_cell_qc.m`

Wrote a self-contained MATLAB script that regenerates the accepted-cell set and
`abcd_passed.mat` from `abcd_data_08-Sep-2025.mat`. Verified against the canonical
`qc_all_sessions.mat` (2026-04-16) cell by cell across all 1042 units:

    electrodeLabel differing : 0        RPV      max|diff| 0.00e+00
    n_spikes       differing : 0        corr_max max|diff| 0.00e+00
    accept/reject  differing : 0        fail reason differing : 0

1042 -> 984 (58 excluded), and `abcd_passed_rebuild.mat` matches the canonical file
in size, session count, cell count and electrode-label order.

**Resolved the missing-parameter question.** The canonical run recorded
`MinOverallFR_Hz = 0.1` and `SessionLowFR_Hz = 0.1`, which appear in no script. They
were never applied: 13 accepted units fire below 0.1 Hz (min 0.049 Hz). They are
deliberately NOT implemented in `run_cell_qc.m` — implementing them would change the
accepted set. The three criteria in the script (spike count, RPV, within-bundle
correlation) reproduce the canonical split exactly.

Outputs use an `OUT_SUFFIX` (default `_rebuild`) so nothing canonical is overwritten
until a rebuild has been verified.

### 2026-08-27 — new cell ROI table from site files only

`scripts/build_cell_roi_table.py` ->
`derivatives/ROI_assignment/cells_from_site_files_2026-08-27/neurons_with_ROI_labels_v2.csv`

Every coordinate read from the recording site's own electrode file. Utah files are
resolved by the patient ID each file declares in its own `Fname` (never by
coord-matching), and the Utah coordinate and microwire label are read from the SAME
row index, so no ordering assumption remains.

    baylor_file_micro         608     as shipped
    utah_file_micro           168     as shipped
    ucla_file_micro           140     as shipped
    baylor_file_305to152       35     BY2-YEN, same file's MNI305 + Fischl affine
    baylor_micro_from_macro    26     BY2-YER, macro + 3.15 mm (Baylor's own constant)
    no_electrode_file           7     UT1-202302 (3), UT1-202422b (4)

ROI counts, published -> rebuilt: EC 38->35, HC_anterior 276->291, HC_mid 231->233,
PCC 61->61, **mOFC 163->139**, mPFC 155->158. Cells with an ROI 924 -> 917.

**The 7 cells with no electrode file** take the collaborator's `regionLabel` from
`abcd_data_08-Sep-2025.mat`, flagged `roi_provisional=True`: UT1-202302 -> ROFC,
UT1-202422b -> RHC. Her labels carry hemisphere but not the medial/lateral or
anterior/mid distinction this taxonomy needs, so they are recorded coarsely
(`OFC_unsplit`, `HC_unsplit`) rather than invented. Both fall below the
>=3-subject rule (one subject each) and so carry `alt_final_roi = NaN` — they are
identifiable in the table but do not enter per-ROI analyses. Note this is a real
change for UT1-202422b: those 4 cells were previously counted as **mOFC** because
they sat on the placeholder coordinate; the collaborator calls them hippocampus.

Columns added: `has_coordinate`, `coord_source`, `roi_source`, `roi_provisional`,
`collaborator_regionLabel`, `published_alt_final_roi` (for diffing).

## 2026-08-27 — cell ROI pipeline consolidated onto site electrode files

`scripts/cell_to_roi_july26.py` is again the single script for cell ROIs. Its
coordinate section now reads every coordinate from the recording site's own
electrode file; `scripts/build_cell_roi_table.py` (a temporary standalone) was
deleted rather than left as a second entry point.

**s47 and s53 arrived from the collaborator and close the last gap.** Both declare
the correct patient in their own `Fname`: `s47/Electrodes.mat` -> 202302,
`s53/Electrodes.mat` -> 202422. **All 984 cells now have a coordinate from their own
patient's file — zero unresolved.** (The previously-used `s47/electrodes/` file is a
different patient, 202311; one session folder can hold two patients' files, which is
why folder position is not trusted.)

**The collaborator's labels were right.** With the real files:
  UT1-202302  chan97/102  she said ROFC -> file says mROFC1/mROFC6, atlas mOFC
  UT1-202422b chan116/119 she said RHC  -> file says mRHIP4/mRHIP7 at
                                           (23.5,-16.6,-17.4), atlas HC_anterior
The 4 UT1-202422b cells had been counted as **mOFC** off the placeholder coordinate.
She said hippocampus; she was correct.

**Final coordinate provenance (984 cells):**
    baylor_v2026_bundle_micro                     608
    utah_file_micro                               175
    ucla_file_micro                               140
    baylor_v2026_bundle_305to152_unreliable_file   35
    baylor_v2026_micro_reconstructed_from_macro    26   <- the only inference left

**alt_final_roi:** HC_anterior 275->295, HC_mid 232->233, mPFC 155->158,
**mOFC 163->142**, PCC 61->61, EC 38->35, NaN 60. 33/984 cells changed.

**Also changed, per the rule that anatomy read-in must match across analyses:**
- `load_ucla_v2026` now filters `isMicro`, so a micro cell cannot match a macro row
  (measured: it never did, so this is a guard, not a change).
- `contact_anatomy.resolve_utah_mat` (LFP pipeline) now resolves by declared patient
  ID via the same index. Utah LFP sessions resolved: **18/18**, up from partial.
- New in `anatomy_sources`: `subject_numeric_id`, `index_utah_files_by_id`,
  `index_utah_mats_by_id`, `utah_micro_coord`, `build_micro_label_map`,
  `mat_patient_id`, `mat_text`.

**NULL / dead end (do not re-run):** `discover_utah_mats()` coord-matching. No
uniqueness constraint, assigned s47's file to six patients, and circular — it
validates electrode files against the hand-entered coordinates it is meant to
replace.

**Script inventory after consolidation:**
    scripts/run_cell_qc.m            which units enter    -> abcd_passed.mat
    scripts/cell_to_roi_july26.py    cell anatomy + ROI   -> neurons_with_ROI_labels.csv
    scripts/swr_build_contacts.py    LFP contact anatomy  -> macro_contacts_all.csv

### 2026-08-27 — re-run readiness after the ROI rebuild

Audited all 18 analysis scripts against the rebuilt ROI table. Checklist:
`docs/rerun_after_roi_update.md`.

**Key fact making the re-run cheap:** the analysed cell set is unchanged — same 984
cells, the same 60 excluded by the >=3-subject rule, 0 cells entering or leaving,
33 moving between ROIs. Per-cell statistics therefore do not need recomputing; only
the ROI grouping does, which is what the existing `RELABEL_FROM` hooks do.

**One genuine trap found and fixed.** `RSA_DSR_ROIs_simple.py` had
`RELOAD_RUN = '2026-07-30_15-58-51-fixed_cells-fixed_perms'`, which skips the RSA
and permutation loop and re-renders plots from the old saved CSVs. A re-run would
have silently reproduced the previous result. Set to `None`; old tag kept in a
comment.

**Checked and left alone:** `REUSE_PERMS_FROM_PREVIOUS_RUNS = True` in the same
script is safe — the perm-cache fingerprint includes `cell_ids`, so any ROI whose
membership changed rebuilds its null and PCC (unchanged) legitimately reuses.
`per_lag_encoding.py` (RELOAD + RELABEL), `spatial_peaks_simple.py` and
`encoding_state_sustained_cv.py` (full run + RELABEL) were already correct.

**Four hardcoded upstream run directories** would otherwise mix old and new results.
Marked in source with `# >>> RERUN-CHECK` (`grep -rn "RERUN-CHECK" scripts/`):
`cell_gradient_master_table.py:CELL_TABLE`, `cell_fMRI_angle_match.py:MASTER_DIR`,
`overlay_double_dissociation.py:DEFAULT_PER_CELL_CSV` and `:DEFAULT_PER_LAG_CSV`.

**Unaffected (fMRI / behaviour / rodent only):** behaviour_summary,
create_fMRI_model_RDMs_on_clean_beh, analysis_rodents_complete_clean,
fMRI_run_RSA_without_rsatoolbox_clean, fMRI_mask_vs_cluster_extract,
harmonic_angle_maps, fMRI_run_RSA_instruction, svc_loso_test, plot_cell_qc_figure.

## 2026-08-28 — cell_to_roi_july26.py leaned; glass-brain plotting centralised

Output verified **byte-identical** to the pre-cleanup table (984 cells, mOFC 142,
EC 35, HC_anterior 295). 1790 -> 1633 lines.

**Deleted the text-based intent rescue (98 lines).** `RESCUE_MAX_DIST_MM`,
`INTENT_MAP` and the rescue bookkeeping (`n_rescued`, `rescue_rows`,
`rescue_source_atlas_label`, the unreachable rescue-breakdown report). It had been
disabled since 2026-07-29 and printed "[DISABLED]"; the 4 leftover cells stayed
leftover either way.

**Kept, deliberately:** `_neighborhood_search` and the amygdala/EC boundary pass
inside step 4. That pass is *not* the intent rescue — it is purely
coordinate-based (reassign Amygdala -> EC when Juelich entorhinal is within 3 mm)
and the amygdala pass calls the helper. It currently reassigns 0 cells, but it is
live, principled code, not dead weight. `rescue_dist_mm` is retained because it
records that pass and step 5 reads it.

**Glass-brain plotting moved to `mc/plotting/cell_results.glass_brain_cells()`**
with module constants `GLASS_MARKER_SIZE`, `GLASS_DISPLAY_MODE`, `GLASS_DPI`,
`GLASS_FIGSIZE`, `GLASS_DEFAULT_COLOUR`. Four of the nine figures now call it
(leftover scatter, per-hint leftovers, master ROI plot, per-ROI atlas overlay, and
the reusable `_master_plot`). The remaining figures build bespoke contour legends;
they now take their marker size and colours from the same constants rather than
hardcoding them.

**⚠ Corrected a palette conflict.** `mc/plotting/cell_results._EXTRA_ROI_COLORS`
had `HC_anterior = '#a30d6c'` (magenta) commented as a "CLAUDE.md override".
CLAUDE.md actually assigns **#23677E to HC_anterior** and **#a30d6c to lOFC**, and
`cell_to_roi_july26.py` followed CLAUDE.md. The module is now correct
(HC_anterior teal, lOFC magenta) and completed with PHC and the non-target ROIs,
so the script's local `ROI_COLORS` is derived from it instead of duplicating it.

**This changes figure colours in three other scripts** that call `get_roi_colour`:
`spatial_peaks_simple.py`, `roi_labelling_glassbrain_overview.py` and
`mpfc_coord_shift_glassbrain.py`. Anterior hippocampus will render teal rather
than magenta in those. Revert by swapping the two entries back if the published
figures need to match the old colours.

### 2026-08-28 — ⚠ the gradient analysis is running on stale coordinates

`cell_gradient_master_table.py:279` reads `MNI_x/y/z` from the per-lag
`per_cell_ALL_ROIs.csv`, not from `neurons_with_ROI_labels.csv`. Those coordinates
are frozen at the per-lag **base** run (2026-06-30): `relabel_per_cell` rewrites the
`roi` column only — it never touches coordinates. So the reload+relabel workflow
that is correct for every ROI-grouped analysis is **wrong for the gradient**, which
is about coordinates.

Measured against the canonical table: **140 of 158 mPFC cells carry stale
coordinates.**

    session 52 (Utah)    6 cells   42.1 mm off -- still at the placeholder
                                   (4.55, 29.50, -20.63) vs real (~3, ~19, +20.7);
                                   the z sign flips
    session  6           4 cells   10.4 mm
    sessions 45, 46      5 cells    7.0 mm
    sessions 43,44,49,
             61,62,7,8,9          3.3-4.8 mm  (Baylor - stale from an earlier
                                   coordinate change, not from this rebuild)

Both the old (2026-08-22) and new (2026-08-28) gradient runs use the same stale
source, so the old-vs-new comparison below is internally consistent but **neither
reflects the corrected anatomy**.

Old vs new gradient overlap (both on stale coordinates):

    mPFC cells             155 -> 158
    inside gradient mask    74 ->  72
    ventral / dorsal      42/32 -> 42/30
    PC1 median split     -13.81 -> -13.81 (unchanged)
    ventral pooled lag      30° ->  30°   (unchanged)
    dorsal  pooled lag      60° ->  60°   (unchanged)
    fMRI theta at sites  25-126° -> 25-126°
    recording sites          16 ->  15

**Fix required before trusting any gradient number:** take coordinates in
`cell_gradient_master_table.py` from `neurons_with_ROI_labels.csv` (join on
subject + cell idx, as `relabel_per_cell` does) rather than from the per-cell lag
CSV, which should supply only the lag statistics.

### 2026-08-28 — gradient analysis fixed and re-run on canonical coordinates

**Fix.** `cell_gradient_master_table.py` now refreshes `MNI_x/y/z` from
`neurons_with_ROI_labels.csv` (join on subject + cell idx, the keys
`relabel_per_cell` uses) via a new `refresh_coordinates()`, and raises rather than
proceeding if any cell is unmatched. `CELL_TABLE` supplies the per-cell lag
statistics only. Measured refresh: 158 mPFC cells, median shift 3.32 mm, max
43.1 mm, **143 cells moved > 1 mm**.

**`harmonic_maps_brain_overlay.py` checked and found correct** — it already maps
`MNI_*_final` onto `MNI_*` at load (lines 396-397). No change needed.

**Result (`cell_gradient_master/2026-08-28_15-19-35`):**

    run                      mPFC  in_mask  ventral  dorsal  vent_lag  dors_lag  sites
    paper / pre-rebuild       155       74       42      32       30        60      16
    new ROIs, stale coords    158       72       42      30       30        60      15
    new ROIs, FIXED coords    158       87       48      39       30        60      19

**The ventral-to-dorsal progression is unchanged: ventral 30°, dorsal 60°.** With
correct coordinates it now rests on more cells and more recording sites, not fewer.

Peak strengths shift: ventral r 0.059 -> 0.079, dorsal r 0.125 -> 0.081. The fMRI
gradient angle range sampled by the cells widens at the low end, 25-126° -> 5-120°.

**Paper numbers to update:** "74/155 mPFC units overlap the gradient" -> **87/158**;
"42 neurons at the ventral end and 32 slightly dorsal" -> **48 and 39**;
"n = 16 recording sites" -> **19**. The median split boundary and the 30°/60°
peaks stand.

### 2026-08-28 — inferential test of the ventral/dorsal gradient split

`scripts/gradient_split_stats.py`. Unit of inference is the recording site (87
in-mask cells sit at only **19 distinct coordinates**); the permutation shuffles
the ventral/dorsal label across sites and rebuilds the pooled profiles through the
same code path.

    pooled argmax difference   30 deg (30->60)   p = 0.37
    pooled circular-mean diff  23 deg (29->53)   p = 0.47
    circ-linear corr, angle vs axis position     r = 0.069, p = 0.88
    circ-circ corr, cell pref vs fMRI at site    r = -0.079, p = 0.74

Null argmax shift: median 0 deg, IQR 0-30 deg. A one-bin shift is the smallest
non-zero difference the 12-lag grid allows and occurs by chance in ~1/3 of
permutations. **The ventral/dorsal split is descriptive, not statistically
supported.**

Also: after the coordinate fix the fMRI gradient angle at the cells' own sites is
**58 deg (ventral) vs 61 deg (dorsal)** — a 3 deg difference. The manuscript clause
"consistent with the fMRI progression from 30-75 deg" is not supported at the cell
locations; before the fix these read 71 deg and 100 deg, which is where the
apparent correspondence came from.

Counts to update: 74/155 -> **87/158** in mask; 16 -> **19** recording sites;
42/32 -> **48/39** ventral/dorsal.

Write-up incl. smoothness-index suggestions for Fig 3b:
`docs/gradient_split_stats_and_smoothness.md`.

### 2026-08-28 (addendum) — two corrections on the gradient stats

**1. Tests against zero DO exist and are nominally significant.** I had tested only
the ventral-vs-dorsal contrast. At each group's own peak, one-sided:
ventral 30 deg r=0.079 t(47)=2.44 p=0.009 (cell) / t(11)=0.81 p=0.219 (site);
dorsal 60 deg r=0.081 t(38)=2.21 p=0.017 (cell) / t(6)=2.24 p=0.033 (site).
Caveats: testing at the argmax chosen from the same data is circular (Bonferroni
over 12 lags takes ventral to p=0.11), and it does not test the progression claim.

**2. The fMRI map DOES show a clear progression; my earlier number was wrong.**
Angle by MNI z in the gradient mask runs 59 deg (z -5) -> 95 deg (z +10) ->
158 deg (z +28) -> 333 deg (z +38), monotonic. My "58 vs 61 deg" came from the
pipeline's per-cell lookup, which is a **single voxel** (SPHERE_RADIUS_MM = 0)
averaged arithmetically rather than circularly. I quoted it without checking its
derivation.

The real issue is cell placement: ventral cells span z -4.6 to +4.1 (mean 0.8,
12 sites), dorsal z +1.2 to +9.8 (mean 2.8, 7 sites) -- **2 mm apart with
overlapping ranges**, sampling only the flattest ventral stretch. Recomputed with
the pipeline's constants, all three sampling schemes give dorsal <= ventral:
(a) COM vertex 52/46, (b) per-site vector mean 58/45, (c) mask voxels <=8mm 84/66.
Likely because PC1 = [-0.04, -0.41, 0.91] mixes y and z, and the dorsal group has
higher z but lower y.

**Recommended claim (option d):** the units lie at z -5 to +10 where the map reads
~60-95 deg, overlapping the immediate-future quarter (0-90 deg). Defensible;
the ventral-vs-dorsal progression is not.

### 2026-08-28 (addendum 2) — vs-zero tests stored, and the z-projection works

**New:** `scripts/gradient_split_vs_zero.py` -> `gradient_split_vs_zero.csv`
(cell / site / subject units x 12 lags + the pre-specified 30+60 window, one-sided
vs zero, BH-FDR across the 2 groups and across the 12 lags).
`scripts/gradient_fmri_z_projection.py` -> `fmri_angle_z_profile.csv`,
`fmri_angle_by_group.csv`.

Dorsal cluster is above zero at every unit (q = 0.033-0.067); ventral survives only
at cell level and neither survives the 12-lag correction, so the peak-lag test
should be reported as the pre-specified 30+60 window.

**The z-projection recovers the correspondence.** Vector-mean angle across
gradient-mask voxels per z-slab: 62 deg at z=-2 -> 71 (z=0) -> 82 (z=+2) ->
89 (z=+4) -> 97 (z=+10). Read off at the groups: **ventral z=0.8 -> 70.5 deg,
dorsal z=2.8 -> 81.6 deg** — an 11 deg progression in the predicted direction.

This supersedes the earlier "58 vs 61 deg", which came from the pipeline's
single-voxel lookup averaged arithmetically rather than circularly. Caveats: it is
descriptive, not a test, and the z-ranges overlap. The fMRI range over the cells'
z-span is **~70-97 deg**, not the 30-75 deg the manuscript states.

### 2026-08-28 (addendum 3) — consolidated gradient results table

**New:** `scripts/gradient_results_table.py` ->
`<run>/final_splits/gradient_results_summary.csv`. One tidy long-format table
holding every statistic quoted for Fig 3d, in three blocks:

    contrast  the ventral-vs-dorsal tests (site-level permutation, seed 42).
              These existed nowhere on disk before -- they had only ever run
              interactively -- so they are now reproducible.
    vs_zero   one-sided tests against zero at cell / site / subject level,
              merged from gradient_split_vs_zero.csv
    fmri_z    where each cluster sits on the fMRI gradient z-profile,
              merged from fmri_angle_by_group.csv

`scripts/gradient_split_stats.py` absorbed into it and deleted, so the contrast
tests have a single home.

Permutation values reproduce exactly from seed 42 (argmax p = 0.3722, circular
mean p = 0.4673, circ-linear r = 0.0688 p = 0.8696, circ-circular r = -0.0787
p = 0.7441).

Gradient scripts now, in run order:
    cell_gradient_master_table.py    per-cell master + canonical coordinates
    cell_fMRI_angle_match.py         ventral/dorsal splits
    gradient_split_vs_zero.py        vs-zero tests   -> gradient_split_vs_zero.csv
    gradient_fmri_z_projection.py    fMRI z-profile  -> fmri_angle_*.csv
    gradient_results_table.py        contrasts + merge -> gradient_results_summary.csv

### 2026-08-28 — manuscript gradient-section change list

`docs/manuscript_gradient_section_changes.md`. Every value in the
ventral-to-dorsal gradient section, its Figure 3 legend and the Methods
subsection, checked against the re-run results.

Changes: 74/155 -> **87/158** in mask; 16 -> **19** sites; 42/32 -> **48/39**;
median split -13.81 -> **-13.51 mm**; mPFC 60 deg t(31)=2.245 p=0.016 ->
**t(32)=2.156 p=0.019**; mid HC 0 deg r .055 t(34)=2.27 p=.0147 -> **r .060
t(35)=2.320 p=.0131**; mid HC 330 deg r .038 t(34)=2.072 p=.0229 -> **r .033
t(35)=1.745 p=.0449** (weakens); Fig 3 legend mPFC 155/32 -> **158/33**, HC mid
232/35 -> **233/36**, overlap 74/81 -> **87/71**.

Unchanged: the 30 deg / 60 deg peaks, PC1 = [-0.04, -0.41, 0.91].

Two items **unverified** because `harmonic_angle_maps.py` has not been re-run:
the subject-wise centre-of-mass linear trend (t(32) = 2.78), and the Fig 3b range
"~30 deg ventrally -> ~360/0 deg dorsally" -- my z-profile of the gradient mask
reads ~59-87 deg at the ventral end, so the ventral figure looks too low, but that
is a different computation (surface projection, per-sub-model t>=1.5) and should be
re-derived rather than taken from my number.

The sentence needing most work is the ventral/dorsal one: peaks unchanged, but the
difference is not significant (site-level permutation p = 0.37) and the "consistent
with the fMRI progression from 30-75 deg" clause should be replaced with the
z-projection read-out (ventral 71 deg, dorsal 82 deg). Suggested wording in the doc.

### 2026-08-28 (addendum 4) — gradient stats folded into cell_fMRI_angle_match.py

Deleted `gradient_results_table.py`, `gradient_split_vs_zero.py` and
`gradient_fmri_z_projection.py`. The two read-outs that are kept now live inside
`scripts/cell_fMRI_angle_match.py`, so this part of the analysis is still the
two scripts it always was (`cell_gradient_master_table.py` then
`cell_fMRI_angle_match.py`).

New outputs in `<run>/final_splits/`:

    lagwise_vs_zero.csv        one-sided t vs zero at ALL 12 lags, CELL and
                               SUBJECT level, for all three split schemes
                               (168 rows). BH-FDR across the groups of a scheme
                               within each (unit, lag).
    fmri_z_readout.csv         per group: n cells, n sites, z min/max/mean, the
                               fMRI angle at z mean, and the angle spanned across
                               the group's z range.
    fmri_angle_z_profile.csv   the map's vector-mean angle per 1 mm of MNI z
                               (z -10 to +48), for plotting.

Key values (pc1_ventral_dorsal), unchanged from the standalone scripts:

    cell    ventral 30 deg  r .079  t 2.44  p .0093  q .0185
    cell    dorsal  60 deg  r .081  t 2.21  p .0167  q .0334
    subject ventral 30 deg  r .051  t 1.20  p .1297  q .2595
    subject dorsal  60 deg  r .139  t 2.26  p .0226  q .0452
    ventral  z -4.6 to 4.1 (mean 0.8)  fMRI 70.5 deg at mean, spans 72-89 deg
    dorsal   z  1.2 to 9.8 (mean 2.8)  fMRI 81.6 deg at mean, spans 72-97 deg

The cross-lag rows also show the specificity: ventral is flat at 60 deg
(r = -0.002) and dorsal is weaker at 30 deg (r = 0.049) than at its own 60 deg.

**Dropped, per user decision:** the ventral-vs-dorsal permutation contrast
(argmax p = 0.37, circular mean p = 0.47) and the two continuous correlations.
Note this removes the basis for stating that the ventral/dorsal difference is
not significant -- that caveat now has no stored statistic behind it.

Also fixed while wiring this in: `load_master_provenance` is now called
unconditionally (the z-profile needs the harmonic-root and mask paths, not just
the brain rendering).

### 2026-08-28 (correction) — the fMRI 30 deg region is real; my read-outs were wrong

I twice reported the wrong fMRI angle for the ventral/dorsal cell groups and, on
that basis, wrongly advised that the manuscript's "consistent with the fMRI
progression from 30-75 deg" was unsupported. **It is supported.**

    circular median at the cells' own voxels   ventral  36 deg   dorsal  84 deg   <- correct
    same, eighths map                          ventral  37 deg   dorsal  75 deg
    vector mean at the cells' voxels           ventral  58 deg   dorsal  61 deg   <- misleading
    whole-mask z-profile at group mean z       ventral  70 deg   dorsal  82 deg   <- wrong region

Two compounding errors. (1) I summarised a broadly distributed circular variable
(0-120 deg at the recording sites) with its **vector mean**, which pulls towards
the middle and hid a 36->84 deg separation behind "58 vs 61". (2) Correcting that,
I switched to a **whole-mask z-profile**, which averages the entire y (16-70) and
x (+-14) extent of the gradient mask rather than the neighbourhood the electrodes
occupy -- that region is dominated by 90-120 deg voxels.

`fmri_z_readout.csv` now stores both, named so they cannot be confused:
`fmri_at_cells_median_deg` / `_vec_mean_deg` / `_min_deg` / `_max_deg` (quote
these) and `fmri_zprofile_at_z_*_deg` (context only). Added `_circ_median_deg`
to `cell_fMRI_angle_match.py`.

**Manuscript impact:** item 3 of the change list is withdrawn -- "theta = 30 deg to
75 deg" should stay (measured 36-84 deg); only 74/155 -> 87/158 changes. The
suggested rewording for the ventral/dorsal sentence now quotes 36 deg and 84 deg.

## 2026-08-29 — SWR macro contacts rebuilt on the corrected anatomy

`scripts/swr_build_contacts.py` re-run after the cell-anatomy work. Three fixes,
then 28 -> 32 sessions with a hippocampal bipolar pair (78 -> 98 pairs).

**1. Utah files resolved by declared identity** (from the 2026-08-27 work).
`s47` had been using patient 202311's electrode file and yielded **0** hippocampal
contacts; with its own file (202302) it yields 8 contacts / 3 pairs. `s24` moved
from s23's file to its own (10 -> 8 contacts). Net +1 session.

**2. `index_utah_mats_by_id` now prefers the file that carries the macro arrays.**
It took the first file per patient, which for s04 is a bare top-level
`ChannelMap.mat` with no `ElecMapRaw`/`ElecXYZMNIRaw`. Candidates are now scored on
those keys. Also fixed `merged.setdefault(...)` in `build_macro_table` -- a
DataFrame has no `setdefault`, a latent crash in the no-anatomy branch that only
fired once a mat without macro arrays reached it.

**3. Atlas fallback for `is_hpc` where the site segmentation is silent.**
Six Baylor subjects (YEL, YEP, YEQ, YER, YEU, YFT) ship an electrode file whose
`ROI_DK2005_3mm` column is **0 % populated**, against 91-98 % for every other
subject. Reading only that column concluded "no hippocampal contact" and dropped
**10 sessions** -- a property of the file, not the patient. `_hpc_with_atlas_fallback`
now uses the probabilistic atlas where the site said nothing.

    NULL / dead end: gating that fallback on `native_roi` (the MAPPED ROI) is
    wrong -- it is None for every non-target region, so the atlas overruled 110
    contacts the site had explicitly placed in white matter, fusiform gyrus and
    the inferior lateral ventricle. Gate on `native_region`, the raw
    segmentation string. Caught before it reached any result.

    Final scope: 69 fallback contacts, ALL with `native_region == 'Unknown'`,
    in 3 subjects (YEL, YER, YEU). Recovered s10, s11, s18, s25.
    `hpc_source` ('native' / 'atlas_fallback' / 'none') is written per contact
    so the fallback can be audited or excluded downstream.

**Current state:** 32/60 sessions usable locally (18 baylor, 1 ucla, 9 utah ->
now 22 baylor, 1 ucla, 9 utah), 98 hippocampal bipolar pairs.

**Still excluded:** 26 sessions have **zero raw LFP files on this machine** --
a data-location issue that should resolve on ceph, and the single biggest
remaining gain. s16 (BY2-YEP) reads only 4 channels from its raw header, which
looks like a wrong-nsx or header problem worth a look. s04 (UT1-202216) resolves
its electrode file but 0 of 132 channels join: the table carries `bLACG6`-style
anatomical labels while the raw header carries clinical names (`LCM1`...), so
that join needs a channel-index bridge.

### 2026-08-29 — hippocampal contacts: coordinate-only, one per electrode

Per user decision, macro-contact location is now determined **solely from the MNI
coordinate**. All inference of brain location from site-supplied region strings
was removed.

**Deleted:** `native_roi_label` (the region-string -> ROI mapper) and
`_hpc_with_atlas_fallback` (yesterday's native/atlas hybrid). `native_region` is
still carried in the table as metadata; nothing reads it. The white-matter
reference picker now filters on `atlas_roi` rather than `native_roi`.

**New:** `anatomy_atlas.hippocampal_probability(coords)` -- P(hippocampus) in per
cent from the Harvard-Oxford subcortical PROBABILITY maps. A max-prob atlas gives
a label, which cannot rank two contacts that are both "hippocampus"; picking one
contact per electrode needs a continuous measure, and probability makes
"deepest in the structure" the winner.

    NULL / dead end: `fetch_atlas_harvard_oxford('sub-prob-2mm')` returns 22
    labels but only 21 volumes -- 'Background' has no volume, so the 4-D index
    is label_index - 1. Indexing by label position returns 0 % everywhere,
    including at the hippocampal centroid. Caught by the sanity check.

**New:** `contact_anatomy.select_hpc_contacts()` -- per probe, rank contacts by
`hpc_prob` and keep the single highest, provided it clears `HPC_PROB_MIN = 25 %`
(matching the maxprob-thr25 atlases used elsewhere). Adds `hpc_prob`,
`hpc_rank_in_probe` and a strictly one-per-probe `is_hpc`.

**Result:** 108 hippocampal contacts across 108 distinct (session, probe) pairs --
**0 probes with more than one**, invariant verified. 32/60 sessions, 107 bipolar
pairs (baylor 76, ucla 3, utah 28; one Utah probe has no valid reference
partner). Selected contacts: P(hippocampus) median 61 %, IQR 42-84 %, min 26 %.

Of the 128 runner-up contacts on those same probes, 56 would themselves clear
25 % -- i.e. the old rule was admitting roughly twice as many contacts, several
per electrode, which is what the one-per-probe constraint now prevents.

`swr_build_contacts.py`: the atlas is no longer optional. Contact selection is
coordinate-based, so without it no contact can be chosen, and the run yields zero
pairs loudly rather than silently falling back to labels.

### 2026-08-29 — docs and preflight updated for the coordinate-only rule

`data/final_results/ripple_analysis/methods.md`:
- **§4.1 rewritten.** Was "ROI: native-space primary"; now "from the MNI coordinate
  alone". States why the earlier rule was dropped (0 %-populated column for six
  Baylor subjects costing 10 sessions; not comparable across sites; cannot rank
  contacts within a probe) and reports the concordance honestly: of 108 selected
  contacts, 90 have a site label and **71 (79 %)** of those agree. The 19
  disagreements are fusiform (6), amygdala (7), ventricle (5); they carry lower
  confidence (median P(HC) 51 % vs 63 %). Flags the five ventricle contacts as
  worth revisiting once ripple rates exist.
- **§4.2 rewritten.** Anchor is now the highest-P(hippocampus) contact per probe,
  not the most medial. Notes that 56 of 128 runner-ups would clear 25 %, so the
  one-per-probe constraint is doing real work.
- **§4.3** replaced with the current state: 108 contacts, 107 derivations, 32
  sessions, 22 subjects; invariant verified; span median 5.48 mm. Flags that this
  covers only 34 of 60 sessions locally, and that **s61-s63 are absent from the
  config entirely**.
- **§3.5** gains the Utah file-identity rule (`Fname` / `PatientIDStr`), why
  coord-matching was dropped, and that s47 holds two different patients.

`HOW_TO_RUN.md`: cluster prerequisites rewritten. **`nilearn_data` is now
REQUIRED** (the old text said it was not) -- rsync `~/nilearn_data/fsl` (~34 MB)
and set `NILEARN_DATA`, since `sub-prob-2mm` is a new dependency that will not be
in an older cache. Added the rsync for Utah `Electrodes.mat` files, which live in
three different subdirectory names and at the top level for s47/s53.

`scripts/swr_check_inputs.py`: nilearn reclassified from "not required" to
required, and the preflight now actively probes `hippocampal_probability()` at a
canonical hippocampal voxel, failing loudly with the rsync instruction if the
atlas is unreachable.

## 2026-08-29 — first full cluster run, and repo-hygiene fixes

**Cluster result: 50/60 sessions, 173 bipolar derivations** (baylor 131, ucla 11,
utah 34), against 32 sessions / 107 pairs on the development machine. The
one-per-electrode invariant holds at every site (`n_hpc == n_pairs`). Baylor's
131 contacts give 128 `n_hpc_pairs` because 3 anchors are assigned EC by the atlas
ladder despite clearing the P(hippocampus) threshold -- the HC/EC border case.

**⚠ NEVER write into the repository.** `scripts/batch_swr_on_ceph.sh` used
`logs_path="./logs/..."`, relative to the CWD, so SLURM `.out`/`.err`/`.sbatch`
files were written into the git tree -- and an earlier run's had been *committed*.
It now resolves the data root (`$SWR_DATA_ROOT`, then ceph, then the local path)
and writes to `<data_root>/derivatives/group/swr/slurm_logs/`. The tracked `logs/`
directory was removed and `logs/ *.log *.out *.err *.sbatch` added to `.gitignore`.
Swept the rest of the SWR scripts: no other relative output paths.

**Stale `bipolar_pairs` files could send stage 2 down the wrong path.**
`swr_check_inputs.py` counts pair FILES on disk, not sessions that produced pairs
in the latest build, so a file left over from an earlier run reads as "ready for
stage 2". That is the 52-vs-50 discrepancy in the cluster output: 52 files, 50
sessions with pairs, so 2 files are stale. `swr_build_contacts.py` now deletes a
session's pair file when that build produces none, and logs the removal.

**The Utah preflight warning was misleading.** It looked only in
`s{NN}/electrodes/` and reported sessions 30, 42 and 53 as missing an electrode
file. The pipeline resolves by the patient ID each file declares in its own
`Fname`, across several subdirectory names -- s30 and s42 have no file of their own
but share a patient with s29 and s41 and resolve fine. The check now uses
`index_utah_mats_by_id`, the same path the pipeline uses.

**Still to diagnose (9 sessions):** excluded as "no channel matched an electrode
table". Locally this was 2 (s04 Utah, s16 Baylor); on ceph it is 9, so more raw
data has exposed more channel-naming mismatches. Plus s39, which has no `.ns2`
files on ceph at all.

### 2026-09-01 — Utah channel join: two naming conventions, only one supported

The cluster run left 9 sessions at "no channel matched an electrode table", six of
them Utah with exactly 132 channels and **0** resolved (s04, s41, s42, s47, s48,
s53) -- including s47 and s48, which resolve fine on the development machine.

**Cause.** `build_macro_table` keyed the Utah join on `^chan(\d+)$` only. Utah
recordings use two naming conventions and both occur in this dataset:

    s01 / s02 / s23   chanN 128/132 | label 0/132     <- the old key works
    s04               chanN  15/132 | label 89/132    <- every channel dropped

The channel names and the electrode-table labels are the *same vocabulary*
(`LAMG`, `LANT`, `LCM`, `LINS`, `LPHIP`, `bLACG`, `bLAHIP`, `bLMCG` appear in
both), so this is a rename, not different anatomy.

**Fix.** The Utah branch now builds both candidate keys and uses whichever
resolves more channels. Trailing analog channels (`EyeX`, `EyeY`, `Pupil`, `BP`)
fail to match under either key and are reported unresolved -- never matched on
position, which the s02 132-names-vs-128-columns case rules out.

Locally: Utah resolved 809 -> **898**, HC contacts 29 -> 31, sessions **32 -> 33**
(s04 recovered). One-per-probe invariant still holds.

**New:** `scripts/swr_diagnose_channels.py` prints, per session, the channel
source, the first channel names, how many match each candidate key, and the
electrode table's labels -- so a join failure shows the mismatch instead of just
reporting zero. Prints only; writes nothing.

Not addressed yet: s16 (BY2-YEP) reads only 4 channels from its raw header;
s50/s51 (UCLA) are Blackrock but the UCLA join expects `.ncs` filename stems;
s39 has no `.ns2` files on ceph.

### 2026-09-01 — cluster diagnostic: the 9 failures split into four causes

`swr_diagnose_channels.py` on ceph resolved every remaining failure to a cause.

**(a) Clinical channel names, fixed by the dual-key join** — s04, s41, s42, s48.
Their channels are `LCM1` / `RMPFC1` / `RINS1`, not `chanN` (only 14-16 of 132
match). s04 verified: `join by chanN 0/132, join by label 89/132`.

**(b) Electrode file never reached ceph** — s47 and s53 report "electrode table:
NONE for patient 202302 / 202422". Both files exist locally
(`s47/Electrodes.mat`, `s53/Electrodes.mat`, 8 MB each) and are the TOP-LEVEL
Utah files. The rsync include pattern missed them. Nothing to fix in code.

**(c) Degenerate channel cache** — s16's `channels.npy` holds exactly
`['empty-064','empty-128','empty-192','empty-256']`, shadowing a 232-channel
recording. `_load_channels` now ignores a cache whose names are all placeholders
and falls through to the raw header.

**(d) UCLA Blackrock sessions have no bridge** — s50/s51 read `chan129..chan256`
from an `.ns3` header while their xlsx lists `LAI-1` / `LA1`. Unlike Utah, the two
vocabularies do NOT overlap, and the localizations xlsx carries no channel-number
column, so there is nothing to join on. The other UCLA sessions are Neuralynx,
where the `.ncs` filename IS the electrode name. **These two need a channel map
from UCLA; they cannot be recovered from the files we hold.**

Plus s39, which has no `.ns2` files on ceph at all.

Diagnostic now prints both candidate key match counts and says which wins, rather
than only testing `chanN`.

### 2026-09-01 — misfiled-file warning, subject-level coverage, UCLA verdict

**s47 is clean locally, not on ceph.** `s47/Electrodes.mat` declares 202302
(correct for session 47) and `s47/electrodes/Electrodes.mat` is now absent here,
so each folder holds one patient. ceph still carries the misfiled 202311 copy
under `s47/electrodes/`, which is why s48 was served from s47's folder -- both
declare 202311 and the index took whichever sorted first.

`index_utah_mats_by_id` now prints `[MISFILED] patient X declared by BOTH a and b`
when the two locations are in DIFFERENT session folders. Companion files in the
same folder (`Electrodes.mat` + `ChannelMap.mat`) are normal and no longer warn --
the first version flagged all 16 patients, which would have trained the reader to
ignore it.

**UCLA s50/s51 cannot be recovered from the files we hold.** The cell pipeline
never needed a channel bridge: UCLA cells are named `elec2`, `elec36`, and were
matched to the localizations xlsx **by coordinate** (`ucla_file_micro`), inheriting
`source_electrode` such as `RAI_micro-1`. The macro LFP has no coordinates of its
own to match with -- it has channel names (`chan129..chan256`) and needs a mapping
to xlsx electrode names (`LAI-1`, `LA1`). Those vocabularies do not overlap, and
the xlsx carries no channel-number column (checked). The other UCLA sessions are
Neuralynx, where the `.ncs` filename IS the electrode name, which is why they work.
**These two need a channel map from UCLA.**

**Subject-level coverage now reported by `swr_build_contacts.py`**, since several
subjects contribute 2-3 sessions and a session count overstates independent
coverage. Locally: 24 subjects, 24 with LFP, **23 with a usable hippocampal
derivation (96 %)**; the one exception is BY2-YEP, whose only session with data
(s16) has the degenerate channel cache.

s39 has no LFP file at all (confirmed by the user: never received). It is reported
as excluded rather than treated as a failure.

### 2026-09-01 — remaining SWR failures resolved or classified

After the dual-key join and the two rsyncs, the ceph diagnostic shows **6 of 9
recovered**:

    s04  join by label 89/132      s47  join by chanN 91/132  (own file now present)
    s41  join by label 88/132      s48  join by label 90/132  (own file, not s47's)
    s42  join by label 88/132      s53  join by chanN 82/132  (own file now present)

s47 now reads `s47/Electrodes.mat` (202302) and s48 reads
`s48/electrodes/Electrodes.mat` (202311) -- each session on its own patient.

**The remaining three are missing data, not defects:**

- **s16** -- its two `.ns3` files are both the **NSP-2** amplifier and each carries
  4 placeholder channels (`empty-064`...). The raw header gives the same 4 as the
  cache, so this was never a degenerate cache: the NSP-1 files simply are not
  present. Checked every session: **s16 is the only one with NSP-2 and no NSP-1**
  (all others are NSP-1 only), so `files[0]` is not picking the wrong amplifier
  anywhere else. Exclusion reason now says so rather than "channel list unreadable".
- **s39** -- no LFP file was ever received.
- **s50 / s51** -- need a channel map from UCLA (see the previous entry).

**Refactor:** `_load_channels` existed twice, in `swr_build_contacts.py` and
`swr_diagnose_channels.py`, and had already drifted -- the degenerate-cache check
was only in one, so the diagnostic reported s16 differently from the build. Moved
to `contact_anatomy.load_channel_list()`; both call it.

### 2026-09-01 — ripple QC figures

**New:** `scripts/swr_plot_ripples.py` -> `<session>/LFP-ripples/<run>/figures/`
  grand_average_ripple    mean waveform + ripple-locked TFR (the checkpoint)
  examples_best           clearest accepted events
  examples_borderline     accepted events nearest the threshold
  examples_rejected       what the spectral criterion discards

Validated on s02 (239 ripples, 0.168 Hz, Chen ~0.17-0.24): the TFR shows a tight
narrowband peak at ~95 Hz centred on t = 0, inside 80-120 Hz -- a real ripple, not
a broadband artifact.

Two plotting corrections made while building it:
- the mean of the BAND-PASSED signal is near zero however strong the ripples are,
  because ripple phase is not locked across events. Plot the mean ripple
  ENVELOPE; the first version showed a flat red line and looked like a failure.
- events too close to a recording edge were skipped after selection, leaving holes
  in the example grid. Select from cuttable events instead.

**Two things for the user to judge from these figures:**
1. The borderline accepted events (z = 3.0-3.1) show near-continuous 80-120 Hz
   activity with only a modest increase in the detected window. Whether
   `PEAK_SD = 3.0` is too permissive is a judgement call these panels make visible.
2. Several REJECTED events look like clean ripples (e.g. z = 7.6 at 90 Hz).
   Strict spectral rejection is 32.1% here against Chen's 23.4% +- 9.9%; the
   relaxed variant gives 18.2%. Worth deciding which is primary before the full run.

**PSD questions answered:** the notch is adaptive (`notch_ratio_threshold = 2.0`),
so sessions differ by design -- s02 had a 60 Hz line-noise ratio of **16958x**,
notched to a residual of **0.014x**, while a session with no line noise is left
untouched and shows a smooth 1/f. A narrowband peak INSIDE the ripple band (the
~90 Hz spike on one s01 contact) is not addressed by the notch and is a genuine
concern for that contact.

### 2026-09-01 — QC consolidated into swr_qc_report.py, plus a numeric checkpoint

**`swr_plot_ripples.py` deleted, merged into `swr_qc_report.py`.** That script
already produced the 6-panel checkpoint figure, so a second plotting script was
duplication. `swr_qc_report.py report --session=N` now also writes
`figures/examples_{best,borderline,rejected}.pdf`.

**New: a numeric checkpoint.** "Looks fine" does not scale to 56 sessions, so the
methods.md section 6 criteria are now evaluated numerically against Chen's
reference values, with FAIL (outside a hard range -- do not analyse as is) and
CHECK (outside the reference range -- look at it) verdicts:

    rate_hz              hard 0.05-0.60   ref 0.17-0.24
    spectral_reject_pct  hard 5-50        ref 13.5-33.3   (Chen 23.4 +- 9.9)
    peak_freq_hz         hard 80-120      ref 85-115
    duration_ms          hard 38-500      ref 40-120
    clean_frac           hard 0.33-1.0    ref 0.50-1.0
    ripple_gain          hard >1.2        ref >1.5

`ripple_gain` is the mean ripple-band envelope at the peak over its value at the
window edges -- a detector triggering on broadband noise gives ~1, so it turns
"the grand average looks like a ripple" into a number.

    swr_qc_report.py report  --session=N   figure + grids + metrics
    swr_qc_report.py metrics --session=N   metrics only (fast, for the cluster)
    swr_qc_report.py group                 aggregate every session -> triage table

**Bug found by the new metric:** `clean_frac` was medianed over *all* derivations
including ones already excluded for contamination, so s02 reported 0.361 (a CHECK)
when the analysed pair actually keeps 0.586. An excluded pair must not drag down the
session it was excluded from. `qc_metrics` now filters on `excluded` first.

s02 after the fix: rate 0.168 (CHECK, two ripples below Chen's 0.17), rejection
32.1%, peak 100 Hz, duration 56 ms, clean 0.586, ripple gain 2.98. No FAILs.

**PEAK_SD = 3.0 is kept, and the visual worry about it was wrong.** Events at the
threshold look unconvincing on the example grids, but that is a plotting artifact:
the band-passed trace is scaled x9-x34 to share an axis with broadband, which
magnifies the ongoing band activity as much as the burst. Measured per event
(envelope at peak / envelope at +-0.20-0.25 s), the z=3.0-3.5 bin has median gain
2.65 with 1% below 1.5 -- these are real bursts. Raising to 3.5 would drop 94 of 238
events (0.168 -> 0.098 Hz, well below Chen) for +0.26 gain. Recorded in methods.md
section 6.3 with the full table.

**NULL / dead ends (do not re-run):** rate over *total* recording time is 0.098 Hz
and is the wrong number -- Chen's denominator is artifact-free time, giving 0.168.

**Fixed a misleading plot title.** `qc_psd.png` said "Bipolar derivations after
notch" whether or not the notch had fired, so a session with no line noise looked
like a filter failure. It now reads either "notch applied at 60, 120, 180 Hz" or
"no notch needed (line-noise ratio below threshold)".

**Group QC across the 8-session development set — no FAILs, all usable.**

    session  clean  dur_ms  peak_Hz  rate_Hz  gain  reject%  verdict  n
      s02    0.586    56     100.0   0.168   2.98   32.1     CHECK   239
      s03    0.546    61      97.5   0.190   3.12   36.7     CHECK  1057
      s06    0.686    58     100.0   0.198   3.00   28.1     PASS    692
      s09    0.643    56      97.5   0.154   2.92   35.0     CHECK   592
      s12    0.685    58      97.5   0.158   2.94   41.1     CHECK   378
      s13    0.675    58      97.5   0.176   2.80   36.8     CHECK   327
      s14    0.564    58      95.0   0.180   2.87   34.3     CHECK   301
      s38    0.608    59      97.5   0.206   2.97   31.9     PASS    608

Rate 0.154-0.206 (Chen 0.17-0.24), peak frequency 95-100 Hz, duration 56-61 ms,
ripple gain 2.80-3.12. Those four are tight across sessions, sites and montages,
which is the real evidence that detection is stable.

The one systematic deviation is spectral rejection: 6 of 8 sessions above Chen's
upper bound, mean 34.5% strict vs 21.6% relaxed. Relaxed matches Chen almost
exactly -- and that is NOT a reason to switch, because picking the criterion that
best reproduces another paper's number after seeing which one does is post-hoc.
Strict stays primary as pre-declared; relaxed remains the sensitivity analysis on
19% more events (5006 vs 4194). See methods.md section 6.2.

### 2026-09-02 — Chen Fig 2 replication, and what it revealed about our averages

**`swr_qc_report.py report` now runs every session by default** (`--session=N` for
one), then draws the pooled group figure and prints the group triage table. New
`mc/plotting/ripple_figures.py` holds the publication figures.

**Three questions about ripple quality, answered with measurements.**

1. **Amplitude was never off by an order of magnitude.** The comparison had been
   between different traces. Chen's grand average is +-5 uV and their example
   ripple is +-20-25 uV broadband; ours are +-2 to +-9 uV and +-20-50 uV. Mean
   single-event ripple-band amplitude is 3.79 uV (range 1.46-9.07 across 14
   derivations). The large numbers are the broadband trace in both papers.

2. **Our grand average was cancelling itself.** Detection marks events at the
   peak of the RMS *envelope*, which carries no phase, so the 80-120 Hz
   oscillation sat at a random phase in every snippet:

       envelope-locked (was)  0.50 uV in the average --  7% of single-event
       trough-locked  (now)   7.18 uV in the average -- 95% of single-event

   Trough-locking reproduces Chen's figure and lands on their +-5 uV scale.
   DISPLAY ONLY -- detection, counts, rates and all statistics still use the
   envelope peak. `trough_lock` is called from plotting code and nowhere else.

3. **The sharp wave: present on some contacts, cancelled on others, absent on
   the rest.** Two findings, the second of which corrected the first.

   (a) The QC figure had been plotting an 8-40 Hz band-pass as "sharp wave",
   which high-passes away a deflection 40-100 ms wide. Median |SNR| 0.28 vs 1.37
   for a 20 Hz low-pass, now used.

   (b) New control `swr_qc_report.py sharpwave --session=N` re-reads the raw
   channels WITHOUT the bipolar subtraction (`preprocess_session(monopolar=True)`)
   and averages the same event times on each contact separately. Detection never
   runs on them. Ripple-locked deflection below 20 Hz, in uV:

       session  derivation             monopolar  bipolar  reduction
       s38      RT2aA03-RT2aA04             8.12     1.49      5.4x
       s38      RT2cHbE02-RT2cHbE03         7.29     2.13      3.4x
       s38      RT2bHa02-RT2bHa03           6.31     3.05      2.1x
       s06      LPHIP1-LPHIP2               7.87     6.24      1.3x
       s03      LMH-1-LMH-2                 3.42     6.58      0.5x
       s03      LPH-1-LPH-2                 1.89     1.01      1.9x
       s03      RMH-1-RMH-2                 1.07     0.61      1.8x
       s06      bLAHIP1-bLAHIP2             0.28     0.57      0.5x
       s02      bLHIP1-bLHIP2               0.11     0.15      0.7x

   5 of 9 derivations carry a ripple-locked deflection >=2 uV before subtraction;
   in 3 of those the bipolar difference removes most of it (2.1-5.4x); and in 4
   derivations a >=2 uV deflection SURVIVES into the analysed bipolar signal.
   The other 4 show nothing even monopolar -- s02, the development session, is
   one of them, which is why the first look suggested it was missing everywhere.
   Per-contact heterogeneity is what the anatomy predicts: the dipole reverses
   across the CA1 pyramidal layer.

   CAVEAT NOT YET RESOLVED: the "monopolar" trace is still referenced to the
   amplifier reference, so a deflection shared by two neighbours is equally
   consistent with a local sharp wave and with a global ripple-locked slow
   potential. The discriminating test is the same average on a NON-hippocampal
   contact from the same subject; `swr_build_contacts.py` does not emit those.

   Reporting: call the events **ripples**. The evidence is the ripple band, not
   morphology. Show the sharp wave per contact where it survives, never pooled
   (signs cancel).

   Two bugs fixed in this control before trusting it: the monopolar channel key
   followed "is ns_pos non-NaN" instead of the reader, silently missing every
   UCLA session; and the verdict compared SNR, which is not comparable across
   montages because bipolar shrinks the flank noise along with the signal -- it
   reported the opposite of what the microvolt amplitudes show.

**NULL / dead ends (do not re-run):** averaging raw snippets on the envelope
peak (cancels to 7%); using an 8-40 Hz band to show the sharp wave (SNR 0.28);
judging the monopolar-vs-bipolar sharp wave by SNR rather than microvolts.

**Speedup:** scipy `hilbert` FFTs at the signal's own length, which for a real
recording is an awkward number -- 1.08 s per frequency for s02 against 0.14 s at
`next_fast_len`, an 8x saving that matters because the TFR runs one pass per
frequency per derivation. The zero-padding perturbs only the edges (median
difference 8e-8 uV, interior max 1.1e-5 uV against a 0.35 uV median envelope).
Adopted in the plotting TFR. NOT applied to `swr_detect._event_baseline_spectra`,
which would change existing detection output -- available if detection is ever
re-run from scratch.

Per-session TFRs are cached to `ripple_stacks.npz`; the group figure reads those
rather than recomputing, since the TFR is the expensive part of the QC stage.

## 2026-09-02 — DSR main effect on the fsaverage surface; fixing the "two islands" artefact

**Problem.** The surface overlay in `scripts/harmonic_maps_brain_overlay.py`
drew the DSR main-effect boundary as two disconnected blobs, while the
volumetric (FSLeyes) view of the same cluster is one elongated medial-wall
band. Readers could not tell that the outline *is* the main effect.

**Cause (visualisation only — no analysis was affected).**
`nilearn.surface.vol_to_surf(vol, pial)` with a single mesh defaults to
`kind='line'`: it averages the volume along the vertex normal over ±3 mm.
Thresholding that mean at 0.5 demands that more than half of a 6 mm line lies
inside the cluster, which erodes an elongated cluster that only grazes the
cortical ribbon into disconnected gyral patches.

**Fix.** Binary masks (the ROI gate and the DSR outline) are now projected
with the FreeSurfer `--projfrac-max` recipe: sample at 11 depths strictly
between the white and pial surfaces, take the max, then close 2 mesh rings and
drop patches < 40 vertices (speckle only; counts printed and logged in the
settings JSON). Continuous cos/sin maps are unchanged. Implemented in
`scripts/dsr_main_effect_surface.py` and imported by
`scripts/harmonic_maps_brain_overlay.py`; new figures carry a `_ribbonmax` tag
so the old renderings are not overwritten.

**Effect on the projected cluster (cluster-mass FWE p < 0.05):**

| hemi | old vol_to_surf | ribbon-max raw | + 2-ring closing | final |
|------|-----------------|----------------|------------------|-------|
| lh   | 2333 vtx, 8 patches  | 3264 vtx, 13 patches | 3521 vtx, 6 patches | 3476 vtx, 4 patches |
| rh   |  864 vtx, 1 patch    | 1058 vtx,  4 patches | 1164 vtx, 2 patches | 1160 vtx, **1 patch** |

**Anatomical finding that resolves the figure.** The medial-wall part of the
cluster is right-lateralised in its dorsal extent. On the **right** hemisphere
the cortical ribbon intersects it as a *single* connected patch spanning
z = 14–56 mm — the elongated band seen in the FSLeyes sagittal view. On the
left, the ribbon intersection genuinely breaks in two (z = 15–36 and
z = 37–49), which is the "two islands" look. **The rh medial view is therefore
the honest and the visually clean panel**; the lh medial view additionally has
the lateral-OFC cluster shining through the translucent pial surface, which
adds spurious-looking speckle.

**New script.** `scripts/dsr_main_effect_surface.py` renders the main effect on
fsaverage in three modes — `cluster_t` (t inside the FWE cluster, red-yellow,
black boundary), `dual` (t > 1.7 faded + FWE cluster opaque, the FSLeyes
two-layer look, implemented as one MNE layer with per-vertex opacity because
MNE keeps a single `data` overlay), and `binary` (solid fill). Outputs +
`settings.json` in
`data/derivatives/group/Main_Results_fMRI/dsr_main_effect_surface/<date>/`.

The `dual` mode confirms the complaint about the FSLeyes screenshot: at
t > 1.7 the sub-threshold layer covers most of the brain and, on a translucent
surface, also shows through from the far hemisphere. Recommendation is
`cluster_t` on the pial glass brain, which needs no second threshold.

### 2026-09-02 (later) — diagnostic figures for everything we throw away

Standing requirement from SK: visualise what is removed, never just count it.

**New `figures/artifact_rejection.pdf`** (per session). Nothing previously
visualised the artifact stage at all, despite it deleting 35-54% of every
recording. Four panels: per-criterion flagged fraction; union + 1 s padding per
derivation against the 2/3 exclusion line; when in the recording; and the
removed stretches next to kept ones ON THE SAME Y-AXIS, marked by which criteria
fired. `swr_artifact.artifact_mask(..., return_per=True)` now optionally returns
the individual criterion masks so the marks are accurate.

s38: removed stretches are +-250 uV IED-shaped spikes, kept windows are +-25 uV
ordinary LFP. The 54% loss is real but it is not data. Individual criteria flag
only 0.04-3.2% each (`artifact_criteria.csv`); the total is almost entirely the
+-1 s padding around several hundred crossings.

**New `figures/sharp_wave_examples.pdf`.** Single ripples from the derivation
with the largest surviving low-frequency deflection, broadband + <20 Hz + ripple
band overlaid. On s38 RT2bHa02-RT2bHa03 the ripple sits on a clear slow wave.
Deliberately a best-case display and labelled as such -- the typical case is the
per-derivation table where 4 of 9 show nothing.

**Example-ripple selection fixed.** The Chen panel picked the event with the
largest RMS z, which reliably landed on the contact with the biggest slow waves,
i.e. the one where the ripple is least visible. Now scored by ripple envelope at
the peak / SD of the same window below 40 Hz -- how clearly the ripple stands out
from its background. The example panel also carries the band-passed overlay now,
because a single unaveraged hippocampal trace is busy (Chen's example is too).

**`HOW_TO_RUN.md` section 4 rewritten** (I started a second run doc, then
deleted it -- `HOW_TO_RUN.md` already covered this and better, including the
cluster rsync steps). It now carries the new entry points, a table of which
figure would invalidate what, and the warning not to expect a sharp wave under
the grand average. Two stale `swr_qc_report.py --session=N` invocations fixed:
the CLI now needs a verb.

**Follow-up (same day).** `cluster_t` on the translucent pial surface, medial
view, was chosen as the house style. `harmonic_maps_brain_overlay._make_brain`
now delegates to `dsr_main_effect_surface.make_brain`, so surface, size,
background, `cortex='low_contrast'` shading and `SURF_ALPHA` are defined once
and cannot drift between the main-effect panel and the gradient overlay; the
duplicated `_set_brain_surface_alpha` fallback moved there too. The gradient
overlay is rendered for lh + rh medial only (`RENDER_FILTER`).

### 2026-09-02 (later still) — repo cleanup, and the hypothesis tests

**Archived 7 scripts** to `derivatives/group/swr/archived_scripts/` (also still
in git history). Repo is down to 10 SWR scripts, all on the pipeline path.

    swr_diagnose_blocks.py       block structure is resolved, recorded in session_blocks.csv
    swr_diagnose_channels.py     the joins are fixed, contact_qc.csv reports the outcome
    swr_collect_figures.py       swr_qc_report.py report now does every session + the group
    swr_organise_box_download.py the download is done and sorted
    swr_plot_discovery.py        superseded by swr_hypotheses.py
    swr_build_windows.py         "
    swr_h1_stats.py              "

**Per-uncovering behaviour IS available** -- `windows_error_correct` carried a
caveat saying it was not, which was wrong. New `mc/analyse/swr_behaviour.py`
reconstructs every uncovering attempt and its outcome in session seconds from
the 25 ms derivatives:

  - clock: session_s = grid_onset + bin * 0.025, verified against
    all_trial_times on s02 to within one bin (max 0.02 s)
  - attempts: transitions into the uncover key in buttons_per_25ms_*
  - correctness: the 4 reward times per repeat are the correct ones

Validated against the stimulus PC's own BEH_*.mat on s47: truth 1903 attempts /
1288 correct / 615 errors, reconstruction 1891 / 1285 / 606 = 99.4% of attempts
and 99.8% of correct uncoverings. The raw .mat is NOT used directly: its
BUTTON_PRESS_TIMES are on the stimulus PC clock (s54 starts at 509637 s, machine
uptime), not the photodiode clock, and only 5 of those files exist locally.

Sanity: s47 discovery repeats average 16 errors / 20 uncoverings / 9.4 s; later
repeats 0.7 / 4.7 / 3.0 s.

**New `scripts/swr_hypotheses.py`** -- four hypotheses, one script. H1 (ripples
at the FIRST arrival at D) is primary and pre-declared, uncorrected; H2-H4 are
secondary and FDR-corrected across themselves. Confounds handled by a
log(artifact-free seconds) offset, a movement-key covariate in every model,
fixed window lengths within design, pooled rates as summed counts over summed
exposure, subject-level error bars, and circular-shift permutation as the
primary inference.

NOTE: the permutation null is NOT centred on zero (H1: -0.14). A log ratio of
small counts is biased downward and first-D windows are few. That is exactly why
inference is against the shifted null rather than against zero -- the null is
built by the identical code and carries the identical bias.

Bugs fixed while building: `n_events` vs `n_ripples`; `g.sem` returning the
DataFrame *method* rather than the column (attribute shadowing); H2 keying on a
`phase` column that windows_pauses does not emit (it is `phase_after`); H3
building a window frame without going through `_finalise`, so it lacked
duration_s.

**Development-set hypothesis results (8 sessions, 6 subjects, 200 perms) --
EXPLORATORY, not a result:**

    H1  first_D vs later_D          +24.5%   z=+2.84  p=0.005   (primary)
    H2  exploration vs later pause   -4.6%   z=-0.32  p=0.64   q=0.67
    H3  rate -> later errors      b=-0.79    z=+1.81  p=0.055  q=0.16
    H4  error vs correct feedback   -4.5%   z=-0.37  p=0.67   q=0.67

H1 is in the predicted direction: 0.217 Hz at the first arrival at D vs 0.173 Hz
at later arrivals in the same grid. Its GLM p is 0.22 -- the disagreement is
because a cluster-robust SE over 6 clusters is itself unreliable, and the
permutation is primary (section 13.2).

H3 points the right way (more ripples after first-D -> fewer subsequent errors,
holding discovery difficulty and pause length constant). WARNING for anyone
reading the GLM output: it reports p < 0.0001 while the permutation reports
0.055. The GLM treats 297 grid-derivations as independent when several share a
session and therefore share ripples. TRUST THE PERMUTATION. Averaging
derivations within a grid before the regression should be done before reporting.

H2 and H4 are flat (|z| < 0.4). If that holds on the full set the reading is
that the effect is specific to the moment the plan becomes knowable -- but that
must not be written up as if a null had been predicted there.

Figure bug fixed after the run: the H2 panel keyed on `phase` and collapsed to a
single "pause" bar; the column is `phase_after`.

### 2026-09-02 (cont.) — three more hypotheses, and a portable results bundle

**Three hypotheses I had left untested, added.** SK asked for them and one of
them (H6) already had its window builder written -- `windows_discovery` -- which
I should have run in the first pass.

    H5  explore / plan / execute as THREE phases. add_phase's "exploration"
        merges searching for unknown rewards with knowing all four but not yet
        executing reliably; new add_phase3 separates them.
    H6  is D special BECAUSE it completes the plan? The interaction
        (D - mean(A,B,C)) on the first traversal minus the same later.
    H7  feedback that could change behaviour vs feedback that could not:
        correct DURING discovery, error AFTER it, vs the other diagonal.

Development-set (8 sessions, 6 subjects, 60 perms):

    H5  plan vs execute pauses          -7.6%   z=-0.57  p=0.70
    H6  (D - ABC) first vs later       +38.7%   z=+3.12  p=0.016  q=0.049
    H7  informative vs uninformative    -3.0%   z=+0.27  p=0.43

**H6 is the discriminating test and it comes out the right way.** Within the
discovery traversal D carries +46% the ripple rate of A (GLM p=0.0016) and that
advantage is GONE at later traversals (interaction p=0.0020). "D is special
because it is last / ends the traversal" predicts a main effect of state with no
interaction. It should probably be the primary contrast for the full run rather
than H1, which is the same claim in a weaker form.

**New `swr_hypotheses.py export`** -- one pickle (+ the same tables as CSVs) with
everything needed to redo any of these statistics WITHOUT the LFP: ripple
timestamps per derivation, artifact-free intervals, pairs with coordinates,
behaviour per repeat, every uncovering attempt with its outcome, channel QC.
1.9 MB for 8 sessions, so ~14 MB for the full set against tens of GB of LFP.

WARNING recorded in the bundle's meta and in HOW_TO_RUN: a ripple rate is events
per ARTIFACT-FREE second. Anything that divides by wall-clock time is wrong --
that is why `intervals` is in the bundle.

CLI is now `swr_hypotheses.py run` / `swr_hypotheses.py export` (fire needs a
verb once there is more than one entry point).

Fixed: the summary's "role" column still read "FDR across H2-H4" after the
secondary set grew to six.

### 2026-09-02 (cont.) — the audit now says which warnings matter, and s09 fails

**Two cluster-run failures, both mine.**

1. `swr_qc_report.py report` on ceph errored with "no value for the required
   argument: session". The cluster copy is STALE -- the session=None default and
   the dispatcher are local and unpushed.
2. `batch_swr_on_ceph.sh qc_rep scripts/swr_qc_report.py report <condfile> ...`
   -> "condition file not found: report". The batch script's third argument IS
   the condition file; it runs `python <script> <condition line>` verbatim, so a
   fire VERB has to be inside the condition lines, not on the submit command.
   `swr_make_conditions.py` now also writes `swr_qc_<name>.txt` with the verb
   already in each line: `metrics --session=2 --analysis_name=swr_v1`.

**The audit could not tell a cosmetic warning from a dangerous one.** 17 of 60
sessions came back `needs_review`, which is unusable as a signal because almost
all of it is bookkeeping: a null `segment`, a missing `blocks` key, a YAML block
count disagreeing with the behaviour. The YAML is a hint, not the authority
(methods 3.3), so none of that can corrupt a result.

`swr_audit_sessions.py` now runs the CLOCK GATE the plan always called for:
read the raw file headers, map each behavioural block onto its recording, and
report the head/tail margins. New columns `clock_status`,
`min_head_margin_s`, `min_tail_margin_s`; new status `clock_failed`. The report
separates "the clock is fine, the rest is bookkeeping" from "this one can
corrupt a result".

**CORRECTION to the paragraph below, made the same day.** The first version of
the gate treated ANY overrun as a failure, which was wrong and would have thrown
away four good sessions. A behavioural clock can overrun the recording for two
unrelated reasons: the recording was stopped a second or two before the task
ended (truncation -- harmless), or the block mapping is wrong (corrupting). They
are told apart by magnitude. Across all 60 sessions: 1 genuine failure (s32,
-1905.5 s, files=2 vs beh_blocks=1) and 4 truncations of 0.9-2.0 s (s09, s10,
s26, s33). Truncation needs no action at all -- a window past the end of the
recording gets zero artifact-free exposure and `fit_count_glm` already filters
`exposure_s > 0` (both verified directly).

**So s09 does NOT invalidate the H1/H6 results.** It loses roughly its last two
seconds of behaviour, which is at most one window. The earlier claim that the
development-set numbers needed re-running without s09 was wrong.

**What the gate found (original wording, kept for the record): s09 FAILS.**
Block 2's offset is the
cumulative duration of block 1 (1527.4 s), which is only valid if recording
never stopped. Behaviour block 2 needs 1402.9 s and sits 53.4 s into a 1454.2 s
file, so it overruns the end by 2.0 s. Every block-2 event in s09 is misaligned
by up to ~2 s -- against 2 s analysis windows, that is total.

**s09 is in the 8-session development set**, so the H1/H6 numbers reported
earlier include one session with a broken clock. They need re-running without it.

**A false alarm I introduced and removed in the same pass:** the >= 5 s margin
criterion also flagged s38 as "tight" (3.0 s tail). s38 is SINGLE-block --
behaviour and LFP share t=0, there is no offset to get wrong, and a small
positive tail margin just means the recording ran on after the task. The
criterion exists to validate multi-block OFFSETS and now only applies when
there is more than one block.

**I archived `swr_diagnose_blocks.py` this morning saying "block structure is
resolved". That was wrong** -- s09 proves it. The capability is back, and better,
as part of the audit rather than a separate script to remember to run.

## 2026-09-03 — full-cohort SWR hypotheses (n=46): the development-set results do not replicate

Ran all seven hypotheses on the full cluster output: **46 sessions, 33 subjects,
52,129 accepted ripples, 166 bipolar derivations** (144 after exclusions), via a new
bundle-backed source so the statistics run on the laptop without the LFP.

### Two method defects fixed first (both affect every earlier p-value)

1. **The permutation was shifting in wall time, not on the artifact-free axis.**
   `methods.md` §specifies the clean axis and `assign_events_to_windows` has a
   `shift_s` parameter built for it, but `_shifted_counts_factory` did
   `t_sh = lo + mod(t - lo + sh, L)` on wall time. Shifted events could land inside
   artifact stretches where the detector could never have found them. Now uses the
   documented path. **All 8-session p-values came from the wall-time null.**
2. **`CleanAxis.to_clean` looped over intervals in Python** (~4,500 per pair) and
   dominated runtime. Vectorised with searchsorted; verified bit-identical against
   the loop on all 166 derivations (max abs difference 0.0).

### Results

| | contrast | observed | z | p (one-sided, pre-declared) | q FDR |
|---|---|---|---|---|---|
| H1 | first-D vs later-D | -5.5% | -1.78 | 0.966 | (primary, uncorrected) |
| H2 | exploration vs later pauses | -6.1% | -2.08 | 0.973 | 1.000 |
| H3 | rate after first-D -> later errors | b=-0.26 | +1.93 | 0.025 | 0.150 |
| H4 | error vs correct feedback | -7.9% | -5.20 | 1.000 | 1.000 |
| H5 | plan vs execute | -5.1% | -1.71 | 0.948 | 1.000 |
| H6 | D special x discovery | -6.2% | -1.35 | 0.916 | 1.000 |
| H7 | informative feedback | +0.4% | +0.77 | 0.218 | 0.653 |

**H1 does not replicate.** Development 8 sessions: +23.3%. The other 38: **-9.6%**.
Across all 46, 21/46 sessions positive, median log RR -0.023. The earlier +24.5%
(z=+2.84, p=0.005) was a small-sample artefact of a lucky draw (s38 +0.69, s06 +0.40,
s02 +0.40).

**H6, the discriminating test, is null** (-6.2%, p=0.916). Neither "D completes the
plan" nor its alternative is supported at n=46.

**H4 is the one well-powered effect, in the direction opposite to the pre-declared
one.** One-sided p=1.000 because the observed value is 5.2 SD on the wrong side of its
null; two-sided p ~ 2e-7. Ripple rate is **7.9% LOWER after an error than after a
correct uncovering**, confirmed independently by the GLM (`feedback[T.error]`
coef -0.159, p=0.0001) with an interaction (coef +0.106, p=0.036: suppression stronger
during discovery). The PETH shows the dip beginning BEFORE the press. Post-hoc
direction -- needs held-out confirmation before it is a claim.

### Definitional sweeps (GLM only, `sweep` verb, n_perms=0)

**H1 has no stable window.** The effect grows monotonically with window length
(0.5s +0.018 -> 5s -0.106), which is the opposite of what an event-locked response
does. The **largest effect is in the 1 s BEFORE D** (log RR -0.205, p=0.0001) --
before the event the hypothesis is about. Clipping at the next repeat halves the 2 s
effect (-0.057 -> -0.020), so part of it was spillover into the next traversal.
Conclusion: the first traversal has a **sustained lower baseline rate**, and the
window-averaged tests measure that offset, not a response at D.

**H3 is not robust.** Only 1 of 12 specifications reaches p<0.05 (variable pause,
per grid, all-later errors: p=0.019). All fixed-window variants null (p=0.29-0.99);
per-derivation p=0.078. `rate_hz = n/pause` is mechanically coupled to the
`log(pause_s)` covariate in the same model, and 36% of grids have zero ripples in
that pause. Do not report H3 as a finding.

### Peri-event time histograms (exposure-corrected, new `describe` verb)

Ripples are **phasically locked to reward uncovering**: a transient peak at
**+250-750 ms**, riding on a first-traversal baseline that is LOWER than later
traversals (~0.17 vs ~0.20 Hz). A 2 s window averages the transient away against the
depressed baseline, which is exactly how H1 becomes -5.5%. The remedy is
baseline-corrected, time-resolved testing (post vs pre within trial, cluster
permutation over time), NOT choosing a better window post hoc.

### Ripple-triggered single units (new: `mc/analyse/swr_units.py`)

24 sessions have both ripples and mPFC units. Row order taken from
`all_cells_region_labels_sub{XX}.txt`, which matches the firing-matrix row count in
every session; `neurons_with_ROI_labels.csv` has FEWER rows (s18 24 vs 25, s45 15 vs
18) and its `cell idx` must not be used for this -- it would mislabel HC cells as mPFC.

| region | units | subjects | mean z | p (units) | p (subjects) |
|---|---|---|---|---|---|
| HC (positive control) | 277 | 19 | +0.131 | 0.0004 | 0.029 |
| mPFC | 93 | 16 | +0.054 | 0.32 | 0.44 |
| PCC | 25 | 3 | +0.515 | 0.0004 | 0.21 |

**Positive control passes** (HC peaks at ripple onset, z~0.28), so spike and LFP
clocks are aligned. **No ripple-locked mPFC modulation** in the pre-declared
0-200 ms window. Caveat: HC's own effect is modest for the strongest ripple
signature in the brain -- if the 25 ms matrices are smoothed rather than raw counts,
fast coupling is blurred and the mPFC null is uninformative. Check before trusting it.

### Rejection bias is not a confound

chi2=10.32, p=0.0057 is significant only on n=66,435. Actual acceptance:
exploration 0.6666, first_correct 0.6638, later_repeats 0.6543 -- a **<=1.9% bias on
counts**, against effects of interest in the tens of percent, and running against H2
(which was null anyway). Artifact rejection differences are not a confound at all:
that is the denominator, already handled by artifact-free exposure.

**NULL / dead ends (do not re-run):**
- H1 as a window-averaged first-D vs later-D contrast, at any window from 0.5 to 5 s.
  The baseline differs; the window cannot fix it.
- H3 with the variable-length pause as both window and covariate.
- H2, H5, H6, H7 as pre-declared -- all null at n=46 with 1000 shifts.
- Peri-ripple mPFC firing in a 0-200 ms window at 25 ms resolution: null, and
  possibly under-powered by smoothing.

## 2026-09-04 — correction: "H1-H7" were never pre-registered

SK pointed out that the seven hypotheses in `swr_hypotheses.py` were written by
Claude from SK's verbal description of the idea. They were never declared in
advance by anyone. Labelling H1 "primary (pre-declared, uncorrected)" and
FDR-correcting H2-H7 as a registered family was therefore wrong on two counts:

* it gave those seven contrasts a confirmatory status they never had, and
* it made a set of exploratory probes read as a failed confirmatory study,
  which is a much stronger negative claim than the data support.

Relabelled to "exploratory" throughout; the q-values are retained but are
descriptive only. **The 2026-09-03 entry should be read with this in mind: the
n=46 nulls mean those particular contrasts do not show an effect, NOT that a
pre-registered prediction failed.**

The project is at the exploration stage: the open questions are when the effect
happens, over what window, and how ripples interact with movement. See
`swr_explore.py` (probes, nothing claimed) and `swr_findings.py` (curated).

N3 (no ripple response at grid onset) is also reframed: under SK's logic that is
the EXPECTED result, since at grid onset no reward has been uncovered and there
is nothing plan-relevant to communicate. It is a confirmation, not a null.

---

## 2026-09-07 — Methods figures for single-unit QC and ROI assignment

`scripts/qc_methods_figure.py` → `data/ephys_humans/derivatives/QC_methods_figure_2026-09-07/`

Two 18 × 4 cm panels (Arial, min 9 pt, 11 pt headings), pdf + png (300 dpi) + svg,
built purely from existing derivatives — no analysis re-run, nothing recomputed:

* `qc_all_sessions_rebuild.mat` (output of `scripts/call_cell_wise_QC.m`)
* `neurons_with_ROI_labels.csv` (output of `scripts/cell_to_roi_july26.py`)
* `abcd_data_08-Sep-2025.mat` — raw spike times, read **only** for the pooled
  ISI histogram; cached to `pooled_isi_histogram.npz` so the 6 GB file is
  touched once.

**Fig 1 — single-unit QC.** (a) spike count/unit, 300-spike cut-off;
(b) firing rate of the 36 spike-excluded vs the 1006 retained units;
(c) pooled ISI over the 984 final units with the 1.5 ms refractory window;
(d) max pairwise correlation (100 ms bins), r = 0.50 dedup threshold.
Panels follow the funnel: a/b cover all 1042 sorted units, d the 1006
base-accepted units, c the 984 final units.

**Fig 2 — from recorded to analysed units.** (a) yield 1042 → 984 → 924;
(b) hippocampal MNI-y with the anterior/mid split at y = −21 mm
(Poppenk & Moscovitch 2013); (c) units per ROI with contributing sessions.

Numbers (all reconcile exactly):

| stage | n |
|---|---|
| sorted units | 1042 |
| excluded, < 300 spikes | 36 |
| excluded, RPV ≥ 1 % | **0** |
| excluded, duplicate r ≥ 0.50 | 22 |
| QC-passed | 984 |
| excluded, ROI < 3 sessions | 60 |
| **analysed** | **924** |

Per ROI: HC_anterior 295 (52 sessions), HC_mid 233 (36), mPFC 158 (33),
mOFC 142 (27), PCC 61 (10), EC 35 (8). Not analysed: Visual 21, PHC 12,
medial_CC 8, Thalamus 8, Insula 7, leftover 4.

Two things worth knowing for the methods text:

1. **The RPV criterion never excluded anything.** Max RPV across all units is
   0.60 %, well under the 1 % threshold (median 0 %). The refractory criterion
   is real but was not binding — the manuscript should say units *satisfied*
   it rather than implying units were removed by it.
2. **No waveforms are stored** in `abcd_data_*.mat` (fields: cellID, channelNum,
   electrodeLabel, excludeCell, psth_trials*, region, regionLabel, roi,
   spikeTimes, spike_rate, unitNum). The Wave_clus waveform criteria
   (consistency, slope, amplitude, trough-to-peak) cannot be plotted from the
   current derivatives; the pooled ISI panel stands in for spike-isolation
   quality. Add waveform panels only if the sorted waveform templates are
   re-exported.

Direction check on the hippocampal split: `mc/analyse/anatomy_atlas.py:273`
assigns `y >= -21 → HC_anterior`, so larger y = anterior. The axis arrow
"posterior → anterior" pointing right is correct as drawn.

### Addendum (same day) — firing-rate descriptives

The 1.58 Hz quoted in Fig 1b is the **median**, not the mean. Panel b now
labels both values "mdn" so they cannot be misread. Firing rate over the whole
session, per unit:

| set | n | mean ± SD | median | IQR | range |
|---|---|---|---|---|---|
| all sorted | 1042 | 2.96 ± 4.06 | 1.50 | 0.52–3.71 | 0.041–32.98 |
| after < 300-spike exclusion | 1006 | 3.06 ± 4.10 | 1.58 | 0.59–3.82 | 0.048–32.98 |
| final QC-passed | 984 | 3.04 ± 4.11 | 1.56 | 0.58–3.82 | 0.048–32.98 |
| the 36 excluded units | 36 | 0.10 ± 0.04 | 0.10 | 0.08–0.13 | 0.041–0.23 |

The distribution is strongly right-skewed (mean ≈ 2× median, max 33 Hz), so
median + IQR is the honest descriptor for the manuscript; quoting the mean
alone overstates the typical unit.

---

## 2026-09-11 — left-hippocampus rewDSR instruction timecourse

Inspected the completed `per_TR_svc_instr_test_full_allTR_2026-08-28` run and
generated a LOSO k=50 plot for `rewDSR_instr` in its existing left Garvert MTL
mask. The broad-mask SVC peak is TR5, MNI [-32, -2, -34], t(31)=8.30,
p_FWE < .0001 (1326 voxels × 12 TRs). This coordinate is not hippocampus in
the Harvard–Oxford 50% atlas, so it should be called left MTL/HC-EC rather than
a hippocampal peak.

The anatomical hypothesis was a representation in hippocampus; its timing was
left open and is corrected across all 12 TRs. Repeated the same SVC and LOSO
pipeline in the Harvard–Oxford max-probability 50% left hippocampus mask.
Results are in
`data/derivatives/group/per_TR_svc_rewDSR_instr_HO50_HC_L_2026-09-11/`, with
the derived mask saved below its `masks/` directory and full parameters in
`settings.json`.

* n=32; 519 in-brain left-hippocampal voxels; 12 TRs; 10,000 sign flips.
* SVC peak: TR4, MNI [-30, -22, -20], t(31)=7.873, p_FWE < .0001, corrected
  jointly over hippocampal voxels × TRs.
* LOSO k=50: peak t at TR4, t(31)=6.743, p_FWE < .0001 across 12 TRs; all 12
  TRs significant. Held-out beta is already positive at TR0, reaches its raw
  maximum at TR3, and then declines. This supports a sustained representation,
  not a representation that first appears at D.
  The raw means are 0.05467 ± 0.00863 SEM at TR3 and 0.05151 ± 0.00764 at TR4.
  Their paired difference is not significant (TR4 - TR3 = -0.00316,
  t(31)=-0.906, two-sided p=.372, 95% CI [-0.01028, 0.00396]). TR4 is the
  inferential peak because its smaller between-subject variance gives the
  largest mean/SEM ratio, not because it has the largest raw mean.
* The original broad-mask results are bilateral (right MTL SVC t(31)=6.09,
  TR2, p_FWE=.0009), so no left-lateralisation claim follows without a direct
  hemisphere contrast.
* The older mPFC result currently stored in `per_TR_mask_stats` is n=32
  (therefore df=31, not 32): MNI [-6, 32, 18], TR4, t(31)=5.079,
  p_FWE=.042 from 2,000 permutations over 4,181 voxels × 12 TRs. A manuscript
  value of .041 does not match this saved JSON and should only be retained if
  it comes from a documented later run.

The plotting helper now supports aggregate models such as `rewDSR_instr` with
a compact legend/title and ROI colour, while retaining the fixed A/B/C/D state
colours for split reward-channel plots.

Plotting convention corrected after review: a TR label is the start of a
one-second interval (TR0 = 0--1 s), so numeric timecourse values are now placed
at interval centres (0.5, 1.5, ... s). The reward schedule is drawn at its
true boundaries: A/B/C/D for 1.5 s each on the first pass, then A/B/C for 1 s
each and D for 1 s.

Follow-up bilateral ROI comparison: repeated `rewDSR_instr` in the bilateral
Harvard--Oxford max-probability 50% hippocampus mask (1,049 in-brain voxels).
The bilateral hippocampal result remains significant, peaking at TR4, MNI
[-30, -22, -20], t(31)=7.873, p_FWE < .0001 (10,000 sign flips; correction
jointly over 1,049 voxels x 12 TRs). The existing bilateral Garvert MTL test is
also significant and has the larger corrected peak: TR5, MNI [-32, -2, -34],
t(31)=8.299, p_FWE < .0001 (2,678 voxels x 12 TRs). Per the pre-specified
selection rule, the final memory timecourse therefore uses the bilateral-MTL
search and correction family; the winning voxel is in the left MTL.

Added `scripts/plot_instruction_main_effects_t.py`, which plots replaceable
memory and plan group-t maps at their ROI peaks. It places conditions at their
one-second interval centres, draws the actual reward presentation strip, and
marks conditions significant in each supplied voxel-wise FWE map. It exports
PDF/SVG/PNG plus the exact plotted values and provenance as CSV/JSON. With the
current inputs, bilateral MTL conditions 1--9 are significant. The supplied
execution-model `rewDSR_instr_t.nii.gz` has no significant mPFC condition
(masked peak TR1, MNI [0, 48, 38], t(31)=3.632, p_FWE=.6243), so the orange
trace correctly has no significance bar pending a replacement final map.

### Correction after source-model audit

The publication mPFC effect was initially paired with the wrong later
execution-run map. Its correct source is the original
`group_RSA_instruction_per_TR_glmbase_01-TR{tr}_cropped` series and the mPFC
small-volume result in `per_TR_svc_instruction_rewDSR_allTR_2026-08-28`:
`rewDSR` peaks at TR4, MNI [-6, 32, 18], t(31)=5.079, p_FWE=.0407, corrected
jointly over 4,179 mPFC voxels x 12 TRs. Only TR4 is significant at the
selected peak. The publication plot has been regenerated from these inputs.

The previously selected bilateral-MTL `rewDSR_instr` result must not be
reported. It came from the `full_no_diag` RSA without a block nuisance. In
that design, instruction-model similarity is confounded with the systematic
within- versus across-task-half similarity difference. This is visibly a
whole-brain offset rather than a regional memory effect: at TR0 the mean
whole-brain t is 2.547 and 97.1% of brain voxels have positive t, even though
no reward location has yet been shown. Across TR0--TR11 the fraction positive
remains 90.2--97.2%. The earlier bilateral hippocampus/MTL max-t results are
therefore retained only as a record of a failed/confounded analysis.

Ran a post-hoc cluster-mass sensitivity analysis on the corrected
`within_only` model after subtracting each subject's whole-brain mean at each
TR. The one-sided cluster-forming threshold was p_uncorrected < .001
(t(31)>3.3749), using 26-voxel connectivity and 10,000 subject-wise sign
flips; the maximum spatial cluster mass was taken across all 12 TRs within
each bilateral ROI. No cluster survived:

* bilateral Harvard--Oxford 50% hippocampus: largest cluster at TR4, 10
  voxels, peak MNI [-34, -20, -18], peak t=3.806, cluster p_FWE=.186;
* bilateral Garvert MTL: largest cluster at TR4, 11 voxels, peak MNI
  [-34, -20, -18], peak t=3.806, cluster p_FWE=.373.

This cluster test is explicitly a sensitivity analysis selected after looking
at the voxelwise result; it does not provide evidence for a hippocampal memory
effect. Added `scripts/per_TR_cluster_test.py` so the empirical and permuted
cluster statistics use the identical function and the analysis can be rerun.

### Exact-source hippocampus/MTL double-check

Re-read all 12 requested
`group_RSA_within_th_only_intr-vs-exe_glmbase_01-TR{0..11}_cropped/`
`cropped_masked_smooth_fwhm5_rewDSR_instr_beta_std.nii.gz` files and reran
10,000-sign-flip inference from scratch in bilateral Harvard--Oxford 50%
hippocampus and bilateral Garvert MTL. Results reproduce the earlier
within-only outputs exactly and rule out a source-file mismatch:

* no demeaning: positive peak HC t(31)=2.264 at TR10, p_FWE=.8822; positive
  peak MTL t(31)=2.440 at TR1, p_FWE=.9383;
* subject/TR whole-brain demeaning: positive peak HC t(31)=3.806 at TR4,
  MNI [-34, -20, -18], p_FWE=.2854; positive peak MTL t(31)=4.699 at TR5,
  MNI [-28, -22, -32], p_FWE=.1110.

All p values above are one-sided and corrected jointly over the ROI voxels and
all 12 TRs. The demeaned maps therefore contain the visually plausible local
TR4--5 pattern, but it is not FWE significant. The matching non-demeaned
cluster-mass test has no positive voxels even crossing the p<.001
cluster-forming threshold; the demeaned cluster results remain p_FWE=.186 in
HC and .373 in MTL.

Also added `scripts/per_TR_roi_mean_test.py` and tested the entire bilateral
ROI mean, removing spatial selection before inference and correcting only
over 12 TRs. No positive whole-ROI effect survives either preprocessing:
without demeaning, HC peak t=-0.378 (p_FWE=.9260) and MTL peak t=-0.829
(p_FWE=.9640); with demeaning, HC peak t=2.038 at TR10 (p_FWE=.1268) and MTL
peak t=.550 at TR3 (p_FWE=.7103). Thus the apparent TR4--5 signal is local,
not a distributed whole-hippocampus effect, and remains subthreshold under
both peak-voxel and cluster-mass correction.

Restricted the demeaned bilateral-HC cluster correction to the a-priori TR4/5
window on request, while retaining the all-12-TR common brain mask for
whole-brain demeaning so that only the temporal correction family changed.
The TR4 cluster is still not significant: 10 voxels, mass=2.6245, peak
t(31)=3.806 at MNI [-34, -20, -18], p_FWE=.0530 across bilateral-HC space x
TR4/5 (10,000 sign flips; one-sided CFT p<.001, 26-connectivity). TR5 contains
one suprathreshold voxel, p_FWE=.1760. A preliminary run that allowed the
demeaning brain mask itself to change with the two-TR subset gave p=.0536;
that value is not the controlled comparison and should not be reported.

Rendered the requested combined group-t timecourse from the corrected inputs:
the demeaned bilateral-HC peak in blue and the original instruction-per-TR
mPFC plan peak in orange. The mPFC TR4 point carries the only p_FWE<.05 bar;
the HC curve is shown descriptively with no significance bar.

## 2026-09-13 — event-locked cumulative instruction overview

Inventoried the new `group_RSA_instr_cumrew_glmbase_instr_*_cropped` outputs
from `rsa_instruction_cumulative_rew.json`. There are 11 event definitions,
33 subjects and 41 maps per condition (not 42): 21 single-model maps and 20
coefficients from combo models. Added
`scripts/plot_instruction_condition_overview.py` and generated 26 matched
plan-memory comparisons in both raw and subject/condition whole-brain-demeaned
versions (52 two-panel PNGs total). The panels show the eight individual
reward presentations plus empty screen, and collapsed-first / collapsed-second
/ empty-screen. Every trace is a descriptive, uncorrected group-t timecourse
at one fixed ROI peak selected across the nine detailed conditions; it must not
be used for inference.

The 26 comparisons cover every stored map without summing across/within:
14 single-model comparisons (seven reward models x within/across plan), four
concurrent plan+instruction combo comparisons, and eight split-coefficient
comparisons (four rewards x within/across plan). Instruction maps are reused as
the within-only memory counterpart, as no across-half instruction map exists.
The common input mask has 144,237 voxels; the intersected ROIs contain 4,165
mPFC and 1,050 bilateral Harvard--Oxford 50% hippocampal voxels.

Demeaning audit on all 451 map-condition combinations: within-plan whole-brain
means never exceeded |t|=2 (0/121); across-plan did so in 6.6% (8/121), all
positive; within-memory did so in 12.0% (25/209), predominantly negative
(10.0% below -2, 1.9% above +2). These are occasional model/condition offsets,
not the near-whole-brain full-RDM confound seen in the obsolete
`full_no_diag` analysis. For this new `within_only` design, raw betas against
zero should therefore remain the primary estimand; whole-brain demeaning is a
regional-localisation sensitivity analysis, not an automatic requirement for
every within-half map. If across and within estimates are later combined 1:1,
combine/average the subject-level beta maps and recompute the group statistic;
never add t-statistics or mix a raw component with a demeaned component.

Descriptively, the raw single-model across-half ABCD plan effect reproduces the
expected mPFC timing and coordinate: the selected mPFC curve peaks at
`see-D-first`, t=4.880, MNI [-6, 32, 18]. The cumulative instruction-memory
sequence is not clean in hippocampus in this uncorrected screen, so no
inferential claim is made at this stage.

### Correction: across-half panels are plan-only

The initial overview incorrectly displayed each across-half plan contrast next
to the corresponding within-half instruction/memory contrast. Those traces do
not form a matched contrast: across-half estimates exist only for execution
(plan), whereas instruction (memory) was estimated only within task halves.
Updated `scripts/plot_instruction_condition_overview.py` so the 15 within-half
panels retain four ROI-by-role traces (mPFC/hippocampus x plan/memory), while
the 11 across-half panels contain only the two plan traces (mPFC and
hippocampus). The corrected v3 output supersedes the v1/v2 figures; its index
leaves `memory_model` empty for every across-half panel.

## 2026-09-14 — event-locked instruction group-t NIfTI export

Added `scripts/save_instruction_condition_tmaps.py` and exported uncorrected
group one-sample-t maps for all 41 stored event-locked instruction RSA models.
For both raw and subject/condition common-brain-demeaned beta maps, each model
has a 9-volume resolved NIfTI (A--D first, A--D second, empty screen) and a
3-volume collapsed NIfTI (collapsed first, collapsed second, empty screen).
The fourth dimension is a named condition axis rather than seconds. Outputs
use the 144,237-voxel mask shared by all 11 condition folders, contain 33
subjects, and carry NIfTI t-test intent metadata with 32 degrees of freedom.

Also exported 11 equally weighted within/across plan estimates under both
preprocessing choices and both condition orderings. Each was computed as
`0.5 * within beta + 0.5 * across beta` separately for every subject and voxel,
followed by the group t-test; group t-statistics were never averaged. The full
export contains 208 t-map NIfTIs and is indexed by `map_index.csv`, with exact
settings and source mappings in `settings.json`.

Corrected the automatic plan/memory classifier in the overview script. The
four plan coefficients from concurrent `*_exe_vs_instr` models had previously
been mislabeled as memory in the whole-brain-offset audit because the combo
suffix contains `instr`; the plotted traces were explicitly assigned and were
not affected. Correct counts and audit summaries are: across plan 121 maps
(mean whole-brain t=-0.079; 6.6% |t|>2), within plan 165 (mean=0.221; 0.6%),
and within memory 165 (mean=-0.884; 14.5%).

## 2026-09-14 — ABCD reward RSA methods and implementation audit

Wrote a supervisor-facing intermediate methods report at
`data/derivatives/group/instruction_cumulative_eventlocked_methods_audit_2026-09-14/ABCD_reward_RSA_methods_and_implementation_audit.md`.
It documents the event-locked GLMs, neural and model RDM construction,
`ABCD_rew` and `ABCD_rew_instr`, single versus concurrent fits, the exact
within/across cell sets, second-level beta-map tests, sign-flip SVC/LOSO,
whole-brain demeaning, and the subject-level 1:1 within/across plan average.

Focused diagnostics over the nine resolved conditions showed median raw to
demeaned group-t shifts of -0.805 to 0.014 for the single instruction-memory
map and -0.902 to 0.008 for its combo partial coefficient. The single
across-half plan map changed from -0.034 to 0.005. Raw/demeaned t-map
correlations were 0.953, 0.937 and 0.974, respectively. These are descriptive
pooled voxel-condition audits, not inferential tests; the report contains the
condition-wise brain-mean statistics and explicit caveats.

Added one shared 20-condition axis key for the subject-02 ABCD model-RDM
figures, listing each task-half label and its executed versus instructed
A--B--C--D grid locations. This was read from the existing pairing metadata
and `rewDSR` model vectors; no RSA maps or statistics were regenerated.

## 2026-09-14 — subject contribution and task-order audit of the instruction shift

Added `scripts/diagnose_instruction_subject_order.py`. The lightweight audit
uses the already merged raw 4-D maps and the actual behavioural task order for
all 33 subjects. It records each subject's spatial mean beta across the nine
resolved conditions and the exact change in the mean voxelwise group-t shift
when that subject is omitted. It also correlates the instruction-model
dissimilarity with temporal proximity (`exp(-lag/tau)`) for condition pairs;
cross-acquisition-part pairs have zero proximity. Positive values of that
order metric are the direction expected to bias an RSA coefficient negative
if temporally nearby patterns are spuriously similar.

For single-model `ABCD_rew_instr_within`, 20/33 subject means were negative.
The mean voxelwise group-t statistic across the nine conditions was -0.833;
it remained negative in every leave-one-subject-out sample (range -1.024 to
-0.704). Largest negative contributors were sub-27, sub-33, sub-02 and
sub-23, so no single subject generated the shift. The same subjects strongly
tracked the concurrent-model instruction coefficient (subject-mean Pearson
r=0.970).

There was no evidence that actual task order explained the subject shift. The
primary instruction-dissimilarity/temporal-proximity proxy correlated
Spearman rho=0.114 with the raw subject mean (two-sided permutation p=.516)
and rho=-0.086 with exact LOO negative-t contribution (p=.628). Initial
forward/backward direction (p=.784) and initial task set (p=.987) were also
unrelated. Sub-01 was the only subject whose matched task sets were mixed
between acquisition parts; its mean beta was positive and therefore attenuated
rather than caused the negative shift.

Successive instruction epochs were separated by 164.7 s on average, and
matched forward/backward task instructions by 614.5 s on average among the 32
subjects with within-part matched pairs. Thus this audit addresses order and
slow temporal structure, not short-lag HRF overlap. It cannot test whether
participants differ in residual autocorrelation strength because the current
event-locked FEAT design matrices and residuals are not present locally. The
merged NIfTIs also lack a subject-ID manifest, so their fourth-dimension order
is explicitly recorded as the standard 33-subject merge-order assumption.
Outputs and settings are in
`data/derivatives/group/instruction_memory_subject_order_audit_2026-09-14/`.

## 2026-09-14 — sEEG behaviour: all attempts + task stages

`scripts/behaviour_summary.py` extended on the cell-data (sEEG) side; fMRI
side unchanged. New helper `scripts/extract_uncovers_ephys.py` pulls
per-attempt uncover counts out of `abcd_data_08-Sep-2025.mat`
(`pressed_to_uncover == 1 & correct_uncover == 0`) into
`ephys_humans/derivatives/group/ephys_uncovers_per_attempt.csv`.

**Two levels.** Every cell measure now exists for all attempts
(`rep_overall` axis) and for error-free repeats only (`rep_correct` axis,
the subset the ABCD-code analyses use). n = 63 sessions, 18228 attempts,
1489 grids.
- all attempts: A→D 6.51 ± 2.76 s, whole attempt 9.15 ± 3.98 s,
  90.8 ± 3.6 % shortest walks, slope −0.59 s/attempt (t(62) = −14.4,
  p = 5.4e-20)
- correct only: A→D 4.43 ± 1.74 s, 97.4 ± 2.2 % shortest walks,
  slope −0.006 s/repeat (t(62) = −0.65, p = 0.52) — i.e. essentially all
  the speed-up happens before the first error-free loop.

**Stages**, defined by available information, not by performance:
explore = first attempt of a grid; learn = further attempts up to and
including the first error-free loop; execute = attempts after it.
7/1489 grids never reached an error-free loop.

| measure | explore | learn | execute | RM-ANOVA | learn−execute |
|---|---|---|---|---|---|
| attempt time [s] | 27.8 | 12.2 | 6.6 | F(2,124)=209, p=1.7e-40, ηp²=.77 | t(62)=9.1, p_holm=4.5e-13, dz=1.15 |
| A→D time [s] | 20.6 | 7.9 | 4.7 | F=270, p=7.0e-46, ηp²=.81 | t=6.9, p_holm=3.6e-09, dz=0.87 |
| time to A [s] | 7.2 | 4.3 | 1.9 | F=48, p=3.7e-16, ηp²=.44 | t=12.1, p_holm=1.5e-17, dz=1.53 |
| error fraction | 1.00 | 0.29 | 0.07 | F=1259, p=4.3e-83, ηp²=.95 | t=9.9, p_holm=2.1e-14, dz=1.25 |
| wrong uncovers/attempt | 13.7 | 1.84 | 0.17 | F=1249, p=7.0e-83, ηp²=.95 | t=7.1, p_holm=1.2e-09, dz=0.90 |
| % shortest walks | 31.1 | 86.8 | 96.3 | F=1151, p=8.6e-81, ηp²=.95 | t=−7.3, p_holm=6.0e-10, dz=−0.92 |
| extra steps/walk | 3.44 | 0.66 | 0.11 | F=485, p=2.3e-59, ηp²=.89 | t=7.3, p_holm=7.4e-10, dz=0.92 |

CAVEATS, logged so they are not over-claimed: `error_fraction` is ~1 in
explore by construction (the first attempt on a covered grid cannot be
error-free), and explore-stage uncovers are search, not memory failures.
The clean contrast is learn vs. execute — both stages where all four
rewards have already been seen — and it is significant for every measure.

**Learning speed** (per session, mean ± sd): explore duration
27.8 ± 12.8 s; 2.55 ± 0.58 attempts and 48.8 ± 27.5 s to the first
error-free loop; 17.5 ± 6.9 wrong uncovers before it. Across the ten
correct repeats loop time falls only 4.65 → 4.38 s (4.4 ± 9.3 %,
t(62) = 3.77, p = 3.7e-4).

**Data-quality finding.** 12 of 18228 attempts are flagged
`trial_correct = 1` although the move record contains an incorrect uncover.
Those 12 also incremented the correct-repeat counter, which fully explains
the previously unexplained `rep_correct == 10` overflow (same 9 sessions,
same 12 trials). The flag is left untouched; the discrepant attempts are
listed in `derivatives/group/ephys_uncover_flag_discrepancies.csv`.

New 2.5 × 2.5 cm panels (pdf + png) under `plots/`: `ephys_stage_*`
(6 measures + colour legend, phase ramp pastel-pink → bordeaux),
`ephys_loop_time_by_attempt_all`, `ephys_incorrect_uncovers_by_attempt`,
`ephys_attempts_to_criterion`, `ephys_speedup_percent`,
`ephys_explore_duration`. New tables: `ephys_per_stage.csv`,
`ephys_learning.csv`; `ephys_attempts.csv` gains stage / run_index /
uncover columns; `ephys_shortest_paths.csv` now covers all attempts.

### Addendum — time spent per stage

Added the missing "how much time does each stage take" read-out. Per-grid
stage durations partition the time on task (attempt starts are contiguous),
so shares sum to 100 %.

| | explore | learn | execute | RM-ANOVA | learn−execute |
|---|---|---|---|---|---|
| time per grid [s] | 27.8 | 21.1 | 64.7 | F(2,124)=171, p=2.0e-36, ηp²=.73 | t(62)=−15.4, p_holm=2.5e-22, dz=−1.93 |
| % of time on task | 25.3 | 16.7 | 58.0 | F(2,124)=408, p=2.8e-55, ηp²=.87 | t(62)=−22.4, p_holm=4.2e-31, dz=−2.82 |

Note the reversal against the per-attempt measures: execute is by far the
fastest stage per attempt (6.6 s vs 27.8 s) yet consumes most of the time on
task, because it contains ~9 of ~12 attempts per grid. Explore is a single
attempt and still takes a quarter of the time on task.

New panels: `ephys_stage_time_spent`, `ephys_stage_time_share`,
`ephys_stage_time_stacked` (single stacked bar, 25/17/58 %). `duration_per_grid`,
`total_duration` and `duration_percent` added to `ephys_per_stage.csv` and to
the stage tests.

### Bugfix — NaN loop times poisoned the sEEG learning slopes

Ten attempts across the dataset have no `t_D` (recording stopped
mid-attempt). Passing them to `scipy.stats.linregress` returned NaN for the
whole session, so two sessions (s02, s28 — which have 184 and 247 correct
trials each) were silently dropped from the correct-repeat slope, and eight
more from the all-attempt slope. New helper `_loop_time_slope` drops
non-finite pairs first. All 63 sessions now contribute.

- correct-repeat slope: was −0.006 s/repeat, t(60) = −0.65, p = 0.52 (n = 61)
  → now −0.008 s/repeat, t(62) = −0.97, p = 0.34 (n = 63). Still null.
- all-attempt slope: now −0.629 s/attempt, t(62) = −15.9, p = 1.4e-23 (n = 63)

This also corrects a methods claim: the two sessions missing from the slope
analysis were NOT sessions with "fewer than two correct trials".

Other facts checked against the draft methods while reviewing it:
- fMRI: 33 subjects (35 scanned, sub-21/sub-29 excluded), 10 layouts, 5
  repeats each, mean 2.599 steps per subpath — all as written.
- The fMRI gamma jitter (`3x3_fMRI_part1.py: jitter()`) draws the *subpath*
  duration (truncated 3–12 s, shape 5.75), then divides it across the steps
  of that subpath plus the reward wait. It is not a per-step draw.
- Runs per reward-layout within a session range 1–5, not 1–4
  (85/258/225/52/1 layouts at 1/2/3/4/5 runs).
- `fixed_grids` is set in 28 of 63 sessions, not 26.
- The single excluded s23 attempt is a hand-identified 314.6 s interruption,
  not a 3-SD rule: 357 attempts (82 correct ones) exceed mean + 3 SD and are
  retained.

## 2026-09-17 — ripple RSA in an expanded condition space (spikes + HFB): null result, and why the earlier positive was leverage

Results: `data/ephys_humans/derivatives/group/swr/ripple_rsa_expanded_2026-09-17/`
and `.../ripple_rsa_conditions_2026-09-17/`.
Code: `mc/analyse/ripple_rsa.py`, `scripts/swr_ripple_rsa_expanded.py`,
`scripts/swr_ripple_rsa_conditions.py`.

### (a) Standing decisions moved into code — `rrsa.DECISIONS`
pad 0.25 s; surrogate window as the PRIMARY null; interval scheme capped at
1 s after the press; firing/power read inside the ripple (peak ± duration/2);
positive rho = model encoded. Each entry carries its rationale. Every entry
point now inherits these rather than re-declaring them per script.

### (b) Three bugs fixed
1. **HFB lost ~half of every ROI.** `patterns_from_cache` applied `drop_silent`
   (`rates.sum() > 0`) to HFB. For a cell that means "fired ≥1 spike"; for a
   robust-z HFB derivation it means "above its own median more often than
   below" — true for ~half by construction. This is why mPFC (32 derivations,
   17 sessions) was absent from the HFB table: it fell under
   MIN_CELLS_PER_PAIR. Now auto-detected from a `kind` tag; mPFC returns with
   0 dropped.
2. `ripples_in_intervals` silently ignored its `window` argument; it now caps
   the lag, still clipped at the next discovery.
3. The surrogate draw was duplicated in four scheme-specific branches;
   both selectors now emit `win_lo_s`/`win_hi_s` and one `_surrogate_times`
   uses them.

### (c) Expanded condition space: 32 = config x uncover
448 usable pairs (48 same-config-across-state dropped: same grid seconds
apart, drift mimics `full_abcd`); state |k-k'| partialled out of every fit.
Chance SD falls 0.192 -> 0.047. Models become separable:
r(known_set, full_abcd) = 0.41 (was 0.92-1.00), and `current_location` is
fittable at all (constant within a state by counterbalancing; it lives
entirely in cross-state pairs). Observed surrogate-null SD 0.038-0.061,
matching the predicted floor — the estimator is calibrated.

### (d) RESULT: null, in both modalities. 500 surrogate-window draws, 2000 relabels.
Spikes (n=506 cells, 1185 ripple-condition assignments), family "all":
mPFC known_set +0.064 (z +1.48, p_surr 0.072) — largest positive, predicted
direction, not significant. PCC full_abcd +0.077 (p_surr 0.038) but
p_relabel 0.122; the two nulls disagree, not claimed. mOFC full_abcd -0.183
(z -2.97) — a NEGATIVE fit, and mOFC has only 17 cells/pair with 18.5 of 31
pairs estimable, so discounted as a half-empty RDM, not a finding.
HFB (n=295 derivations, 1209 assignments): **0 of 15 tests significant**.
mPFC known_set -0.096 (z -2.01), i.e. opposite in sign to spikes.
Visual (negative control) flat — contrast with the earlier 8-condition
analysis where Visual lit up as strongly as anything.
Neither modality survives FDR. The two do not converge.

### (e) Why this supersedes the earlier mPFC/D rho = +0.563 (8 conditions, pad 1.0)
Leave-one-condition-out (`swr_ripple_rsa_conditions.py`), model known_set:

  space          max leverage of ONE condition      chance SD
  8 conditions   0.280 - 0.517 (spikes)             0.192
                 0.332 - 0.506 (HFB)
  32 conditions  0.045 - 0.063 (spikes)             0.047
                 0.046 - 0.064 (HFB)

Single conditions moving rho by more than the chance SD of their own space:
**17 of 113 in the 8-condition space; 0 of 160 in the 32-condition space.**
In the 8-condition space one config touches 7 of 28 pairs — 25% of the
evidence — so a single condition can move rho by up to 0.52, MORE than the
entire mPFC/D effect that motivated this. That space was never conditioned
well enough to support the claim, independently of its p-value. Quadrupling
the resolution should have sharpened a real effect; instead mPFC fell to
+0.064. Read the earlier number as leverage, not signal.
Caveat: not the identical estimator (state partialled, 1 s cap, pad 0.25),
so this is strong evidence, not a formal refutation.
Sampling is NOT the explanation: 19-51 ripples per condition, evenly spread,
no empty conditions in HC/mPFC/PCC.

### (f) Nulls are now stored, and verified
`packs_surrogate_<modality>.npz` holds the pattern packs, not just the rho.
Everything downstream of a pack is milliseconds, so a NEW model refits
against the identical 500-draw null in ~10 s instead of 5 min (spikes) /
50 min (HFB). 4.3 MB per 500 spike draws. Regeneration is deterministic and
is checked against the reported null before being written
(`max |replay - stored| = 0.00e+00`); `load_packs(..., expect=)` refuses a
null built under a different pad/scheme/window instead of mixing two nulls.
Reusable for model/family/nuisance changes ONLY — pad/window/scheme/extent
changes correctly invalidate it.

### (g) Failed / superseded, do not repeat
- Do not fit the 8-condition space at a single state and treat rho against a
  0.192-SD null as evidence; check single-condition leverage first.
- `known_set(k=3)` IS `full_abcd` in the 8-condition space; nothing is
  distinguishable at D there. Use the 32-condition space to separate them.
- Do not apply `drop_silent` to any continuous (z-scored) signal.

## 2026-09-17 (later) — the RDM input vectors are NOT scaled, and for spikes that changes the answer

Results: `data/ephys_humans/derivatives/group/swr/ripple_rsa_inputs_2026-09-17/`
Code: `scripts/swr_ripple_rsa_inputs.py`

### (a) What the estimator actually does
`build_rdm` / `build_rdm_flat` CENTRE each feature across conditions
(`C = P - nanmean(P, axis=1)`) and then correlate across the FEATURE axis.
There is no division by SD. Centring removes a feature's overall level, so a
constantly-loud cell cannot make all conditions look alike -- but it does not
equalise how far a feature can MOVE the correlation. Influence is
proportional to across-condition variance, and firing rates here span two
orders of magnitude.

### (b) How concentrated the weighting is (participation ratio, (sum v)^2/sum v^2)
SPIKES -- effective n is 36-43% of the actual feature count:
  HC_ant 132 cells -> eff 48.5 (37%), top-5 cells = 22% of variance
  HC_mid 114 -> 41.5 (36%), top-5 = 24%
  mPFC    53 -> 19.7 (37%), top-5 = 40%
  mOFC    59 -> 25.3 (43%), top-5 = 33%
  PCC     46 -> 17.5 (38%), top-5 = 43%
HFB -- effective n is 82-89%, because `_robust_z` already normalised every
derivation over the whole session. The problem is spikes-only.

### (c) It changes which region looks best. Refit under both normalisations
against the IDENTICAL stored surrogate null (z vs surrogate):
                        centred (shipped)   z-scored
  HC_ant  known_set          +0.77           +2.27
  HC_ant  full_abcd          +0.20           +1.88
  HC_mid  full_abcd          +1.49           +0.24
  mPFC    known_set          +1.48           +0.54
  PCC     full_abcd          +1.80           +1.05
  mOFC    full_abcd          -2.97           -2.30
Centring favours mPFC/PCC/HC_mid; z-scoring favours HC_anterior. The ranking
of ROIs is not stable to a normalisation choice that was never justified.
NOTHING survives multiple-comparison correction under either scheme, so this
is a statement about fragility, not a new positive result. HFB is stable
under both (max |dz| ~ 0.8), as expected given it is already robust-z.

DO NOT pick the normalisation by which one gives the nicer answer. The
principled fix is multivariate noise normalisation (crossnobis): scale by the
NOISE SD estimated from within-condition variability ACROSS RIPPLES, which we
have, rather than by across-condition SD (centring over-weights high-rate
cells; z-scoring over-weights low-rate noisy ones). Not yet implemented.

### (d) mOFC's negative is a coverage artifact, confirmed
mOFC spikes: median 17 cells per pair and only 18.5 of 31 pairs estimable
(vs 30-31 in every other ROI); the example condition pair (2, 19) rests on
n = 9 features, i.e. below MIN_CELLS_PER_PAIR and correctly excluded. Its
z = -2.97 is a half-empty RDM, not a finding.

## 2026-09-17 (evening) — z-scoring adopted, crossnobis implemented, all-uncover scale-up, and a reliability ceiling that explains everything

Results: `.../ripple_rsa_expanded_zscore_2026-09-17/`,
`.../ripple_rsa_crossnobis_2026-09-17/`, `.../ripple_rsa_alluncovers_2026-09-17/`
Code: `mc/analyse/ripple_rsa.py`, `scripts/swr_ripple_rsa_crossnobis.py`,
`scripts/swr_ripple_rsa_alluncovers.py`

### (a) z-scoring is now the standing normalisation
`DECISIONS["normalise"] = "zscore"`, matching `_z_score_per_neuron`
(mc/analyse/rsa_perm_rdms.py) exactly, INCLUDING its sd==0 -> 1.0 handling so
a constant feature stays an all-zero column. Justified by prior project
convention (SK uses it for the other RSAs), NOT by which answer it gives --
it does change the answer and that is logged in the 2026-09-17 inputs entry.
Refit from stored packs (1 min, not 5/50). Under z-scoring HC_anterior
replaces mPFC as the strongest spike ROI: known_set +0.091 (z +2.27),
full_abcd +0.063 (z +1.88). Still FDR-negative.

### (b) Crossnobis implemented, following the project recipe
`crossnobis_rdm` in ripple_rsa.py: per session, noise covariance from
within-condition residuals, shrinkage 0.1, leave-one-fold-out, per-session
RDMs averaged. As scripts/RSA_human_cells_DSR_crossnobis.py, with ONE
deliberate deviation: per-session RDMs are divided by n_neurons before
averaging, because ROI cell counts per session run 3 to 40+ here and the
DSR script's plain mean would let the biggest session dominate.
Verified unbiased on synthetic data: d^2 = 0.015 when no true difference
exists, 4.37 when one does.
FEASIBILITY was the binding constraint on the discovery-only set: 32
conditions gave a median of 1 ripple per (session, condition) and 0 of 27
foldable sessions -- NOT ESTIMABLE. It ran only on 8 configs with states
pooled (28 pairs, chance SD 0.192), where it flipped the sign of HC_mid,
mPFC and mOFC relative to correlation distance and removed mOFC's negative
(-0.328 -> +0.115), supporting that negative having been a weighting artifact.
Nothing significant (best: HC_mid +0.322, z 1.59, p 0.054).

### (c) ALL correct uncovers, not just explore-phase discoveries
`all_uncover_events` + `ripples_after_uncovers` (intervals clipped at the next
uncover of the session, so they tile it once). 32688 uncovers, 12629 ripples
-- 10.9x the discovery-only set. Two spaces from one cache and one null:
  collapsed32  config x state, phase pooled, 448 pairs, chance SD 0.047
  phase64      config x state x phase, 1824 pairs, chance SD 0.0234
`phase64`'s `first_only` family IS the 448-pair discovery analysis, so they
nest. Phase is partialled out of every fit and never silently pooled: on a
REPEAT the subject already knows the whole configuration, so `known_set`
collapses onto `full_abcd` for every repeat condition -- verified in the
output (repeat_only gives identical values for the two models in all ROIs).
At this scale crossnobis IS estimable on 32 conditions (97-100% of
(session, condition) cells foldable, 20 of 24 complete sessions).

### (d) THE KEY RESULT: split-half reliability of the data RDM is ~0
20 splits, 12629 ripples, collapsed32. Asks nothing about any model -- only
whether an estimator recovers the same geometry from two independent halves.
                 corr_zscore        crossnobis
  HC_anterior    +0.017 (t 1.5)     +0.056 (t 3.8)   <- the only one > 0
  HC_mid         +0.003             -0.008
  PCC            -0.010             +0.028
  mOFC           +0.012             -0.013
  mPFC           -0.009             -0.010
Noise ceiling (Spearman-Brown to full length, then sqrt) -- the largest model
correlation that is even in principle recoverable:
  HC_anterior 0.33 (crossnobis) / 0.18 (corr);  mPFC, HC_mid, mOFC, PCC: 0.00-0.23,
  and ZERO for mPFC under both estimators.
This is with 10.9x more data, so it is not a sample-size problem that more
ripples from this dataset would fix. It retrospectively explains the pad
dependence, the normalisation dependence, the 8-condition leverage and the
mPFC rho=+0.563 evaporating: those were all fits to an RDM with no
reproducible structure.
It also settles crossnobis vs correlation: in the ROIs where they disagreed
(HC_mid, mPFC, mOFC) reliability is 0 for BOTH, so they were fitting models to
noise and noise has no sign. Crossnobis is better where anything is
measurable (HC_anterior) and is the better tool in principle, but it cannot
manufacture structure that is not there.

### (e) Model fits at scale, and a modality contradiction
SPIKES: only phase64 + corr_zscore survives FDR -- HC_anterior known_set
rho +0.086 (z +3.16, p<0.005), full_abcd +0.042 (z +2.28). Directionally
consistent everywhere asked (zscored-32 z +2.27, first_only z +2.42), in the
predicted direction, in the one ROI with non-zero reliability, and 0.086 sits
plausibly under its 0.18-0.33 ceiling. BUT crossnobis -- the MORE reliable
estimator in that exact ROI -- puts it at z +1.14, n.s.
HFB: 0/15 significant in three of four space x estimator combinations; NOTHING
survives FDR anywhere. Two findings that count against the spike result:
  * HC_anterior known_set in HFB is NEGATIVE (z -2.91, phase64 corr_zscore),
    directly contradicting the positive spike effect in the same region.
  * Visual, the negative control, is as high as any real ROI under crossnobis
    (known_set +0.126 phase64, +0.149 first_only z 2.23).
mPFC HFB, which was z -2.01 on discoveries only, is flat at scale -- i.e. that
negative was noise too.

### (f) Standing conclusion
Across 4 condition spaces, 3 normalisations, 2 distance estimators, 2
modalities and a 10.9x scale-up, there is no ripple-locked representation of
the reward configuration that survives correction and replicates across
modality. The binding constraint is the reliability of the ripple-locked
pattern itself, not the number of ripples, the model, or the estimator.
HC_anterior known_set is the only candidate worth pursuing and should be
treated as a hypothesis for an independent dataset, not a result.

### (g) HFB reliability — the negative control is the most reliable ROI
Same 20-split test, HFB (added after the entry above was written):
                 corr_zscore        crossnobis
  Visual         +0.044 (t 4.4)     +0.026 (t 1.5)   <- negative control, MOST reliable
  HC_mid         +0.027 (t 2.2)     +0.003
  HC_anterior    +0.001             +0.024 (t 1.5)
  mOFC           +0.015             +0.004
  mPFC           -0.013             +0.004
The only HFB ROI with clearly non-zero reliability is Visual, which is in the
analysis precisely because it should carry nothing task-representational. So
the reproducible structure HFB does contain is generic (arousal, or the
visual response to the uncover itself), not configuration coding. Combined
with Visual also scoring as high as any real ROI on the model fits, the HFB
arm should not be used to support a representational claim.
mPFC reliability is ~0 in BOTH modalities under BOTH estimators -- the ceiling
on any mPFC ripple effect in this dataset is zero.

## 2026-09-17 (j) — I11: rewarded vs non-rewarded squares during exploration

`scripts/swr_ripple_content_i11.py`, Figure 6. **The primary contrast is null.**

SK's question: during the first traversal, does a ripple carry more about a
square that is a reward? Run as a THREE-way split rather than the two-way one in
`POTENTIAL_IDEAS.md`, because a two-way split cannot separate reward from square:

| class | what the square underfoot is | role |
|---|---|---|
| `known_rew` | a reward, already uncovered | TARGET |
| `future_rew` | a reward, not yet uncovered | KNOWLEDGE CONTROL |
| `nonrew` | not a reward in this grid | BASELINE |

`future_rew` is the same physical square as `known_rew` — same template quality,
same geometry, same visit frequency across the session — minus the knowledge.

Controls: (1) matched flanks, so template quality cancels; (2) each target ripple
matched to one ripple of each other class on **dwell and latency from arrival**,
nearest neighbour without replacement, caliper 0.15 s / 10%; (3) two nulls —
template-label permutation for "is there content at all", class-label shuffle
within each matched triplet for the contrasts.

**Results** (n = 41 sessions matched, 1,235 triplets, sessions with ≥ 20):

| ROI | known−nonrew (PRIMARY) | future−nonrew | known−future |
|---|---|---|---|
| HC_all | **+0.241, p = 0.29** | −0.264, p = 0.24 | +0.503, p = 0.036 |
| HC_mid | +0.378, p = 0.34 (n=9) | +0.281, p = 0.42 | +0.092, p = 0.68 |
| HC_anterior | −0.015, p = 0.95 | −0.397, p = 0.13 | +0.389, p = 0.14 |
| mOFC / mPFC | null | null | null |

Per-class content (ripple minus flank, template-label null), HC_all:
`known_rew` **+0.580, p = 0.0094**; `future_rew` −0.164, p = 0.50;
`nonrew` +0.144, p = 0.48.

**Reading.** The per-class pattern looks like the hypothesis — content at known
rewards only — but the direct contrast that licenses that claim is not
significant, and "significant vs non-significant" is not itself a difference.
The one nominal hit (known−future, HC_all, p = 0.036) is 1 of ~50 tests in this
table and its sibling contrast is null, which it should not be if reward
knowledge were the driver. **I11 is not supported at this power.**

### Three things this run established that are worth keeping

1. **69% of known-reward ripples cannot be stillness-matched.** They sit in the
   post-reward pause (median dwell 2.31 s unmatched vs 0.75 s matched) and no
   other class ever pauses that long. I11 as posed is only answerable for the
   short-dwell subset. Any future reward-vs-non-reward ripple comparison
   inherits this ceiling.
2. **Ripple rate is flat across the three classes** — 0.491 / 0.506 / 0.484 Hz —
   despite the dwell difference. The stillness confound is real for dwell and
   not, here, for rate.
3. **Matching-order variance was large enough to change conclusions.** Over five
   seeds the HC_all known−nonrew contrast ranged +0.06 to +0.39 and known−future
   ranged p = 0.014 to p = 0.17. Fixed by repeating the matching 20× and
   averaging the per-session statistic; reporting a single draw would have been
   reporting the seed.

### Two bugs found and fixed on the way

- `scripts/swr_ripple_content_c2.py` called `swc.zscore_cells` without importing
  `mc.analyse.swr_content`. The C2 results on disk are from a run that had the
  import; the file as committed would `NameError`. Import restored, results
  unchanged.
- A single shared RNG across the ROI loop meant that ADDING an ROI changed the
  matching, and therefore the results, of every other ROI. Replaced with
  independent per-session and per-ROI streams (`default_rng([SEED, s, roi_i])`),
  so loop structure can no longer move a number. Same failure class as the
  random-window instability that forced `tiled_windows` in Stage 2.
- `mc.analyse.swr_content.occupancy` now carries `step_idx`, the row of the
  time-sorted step table each interval came from, so per-step fields survive the
  interval filtering.
- Figure 5's caption had hard-coded p-values from before matched flanks and
  per-cell z-scoring, and still called the flanks "±500 ms". It now computes its
  own caption from the run it is plotting.

### (h) Discovery-only crossnobis, HFB arm (completes the 2026-09-17 crossnobis entry)
8 configs, states pooled, model full_abcd, 500 surrogate draws.
**0 of 15 significant.** Best: Visual (the NEGATIVE CONTROL) crossnobis
+0.279 (z 1.36, p 0.090) and HC_anterior crossnobis +0.253 (z 1.47, p 0.072)
-- the control is second only to HC_anterior and ahead of every other real
ROI, the same pattern the reliability test later explained.
The estimators again disagree on SIGN for the same ROI: HC_anterior is -0.252
(corr_centre) / -0.153 (corr_zscore) / +0.253 (crossnobis); mPFC is +0.376
(corr_centre) / -0.001 (corr_zscore) / -0.017 (crossnobis). With a chance SD
of 0.192 and reliability ~0, these are three noisy readouts of nothing, which
is what entry (d) established.

## 2026-09-17 (k) — Pseudo-populations across sessions that ran the same task

`scripts/swr_pseudo_population.py`, Figure 7.

**`grid_id` is not a shared task identifier.** The same `grid_id` carries seven
different reward sets across sessions — it indexes configurations *within* a
session. That is exactly right for leave-one-configuration-out CV (so nothing
earlier is affected) and exactly wrong for pooling across subjects. Pooling here
matches on the reward tuple A-B-C-D itself: 94 distinct configurations, of which
**8 are run by 28 sessions each** and 14 by ≥ 10; 376 of 450 session-configurations
are shared with at least four others. That gives **316 hippocampal cells on one
configuration** where a single session has ~5.

Pseudo-trials: for a configuration and a square, draw one ripple independently
per cell from that cell's own session and stack the per-cell z-scored counts.
Templates are leave-this-configuration-out per cell. Two axes are swept — cells
pooled (2 … 316) and ripples averaged per cell (1, 4, 16) — with 10 cell
subsamples per point and a template-label permutation null.

### The read-out had to change, and that is itself a result

Nine-way argmax decoding stays near chance at every population size
(0.153 vs 0.109 null at 316 cells). This is not a failure of pooling. With every
cell's FULL mean profile — the best this design can do — argmax still gets 2 of
9, while the true square's evidence sits +0.42 SD above the other eight.
**Templates for neighbouring squares are correlated, so clearly elevated
evidence still peaks one square away.** Primary is therefore the continuous
statistic used everywhere else in this line (`target_minus_others`, plus the
scale-free `target_z`); accuracy is reported as a descriptive only.

### Results

**Pooling works, and the signal is distributed.** `target_z` in ripple windows,
hippocampus pooled, 16 ripples averaged: +0.037 (2 cells) → +0.063 (16) →
+0.103 (64) → +0.188 (316). It does not saturate. Averaging more ripples helps
on the same monotone pattern (+0.113 / +0.163 / +0.188 at 316 cells for 1 / 4 /
16). Evidence matrix, HC_all ripples: diagonal **+0.144** vs off-diagonal −0.018.

**Still not ripple-specific.** At each configuration's full population,
16 averaged:

| ROI | cells | configs | ripple z | target_z ripple | flank | difference | p |
|---|---|---|---|---|---|---|---|
| HC_all | 316 | 14 | +0.52 (p = 0.074) | +0.144 | +0.069 | +0.075 | 0.381 |
| HC_anterior | 171 | 14 | +0.60 (p = 0.032) | +0.160 | +0.060 | +0.099 | 0.165 |
| HC_mid | 145 | 8 | +0.39 (p = 0.306) | +0.100 | +0.080 | +0.020 | 0.806 |
| mOFC | 74 | 8 | −0.12 | −0.098 | +0.074 | −0.172 | 0.256 |
| mPFC | 65 | 8 | +0.31 | +0.036 | +0.065 | −0.029 | 0.842 |

Consistently positive, never significant. **The pseudo-population does not
rescue ripple-specificity** — it agrees with Stage 3 C1 and the 9-location
profile.

### NEW: the location signal fades across the traversal

Split by which reward is being sought, at the full population (target_z, ripple):

| ROI | A | B | C | D | slope/state | p | sign |
|---|---|---|---|---|---|---|---|
| HC_all | +0.234 | +0.221 | +0.171 | −0.076 | −0.098 | 0.017 | 11/14 neg |
| HC_anterior | +0.313 | +0.152 | +0.071 | −0.043 | −0.115 | **0.00017** | **14/14 neg** |
| HC_mid | +0.015 | +0.071 | +0.208 | +0.001 | +0.009 | 0.894 | — |

Flanks show a weaker version (HC_all p = 0.039, HC_anterior p = 0.103) and the
ripple-minus-flank slope difference is not significant, so this is a property of
hippocampal location coding in general, not of ripples.

**It is not the obvious confound.** Dwell is 0.417 s while seeking A and 0.350 s
for B, C and D alike — a step down after A, then flat, whereas the signal
declines monotonically all the way to D. Occupancy does not have the shape of
the effect.

⚠ Configurations share sessions, so the t across configurations is
anticonservative; 14/14 negative slopes in HC_anterior is the more honest
summary. ⚠ Independent per-cell sampling removes noise correlations and inflates
absolute levels — only the comparisons are interpretable. Not corrected across
ROIs. **Exploratory.**

## 2026-09-17 (l) — I12, I13, and the roles regression (square dummies added)

`scripts/swr_ripple_content_roles.py`, Figure 8c. One regression per window over
all nine squares, in ripples and in matched flanks:

    explore: E(L) ~ current + goal + next_step + known_rew + unknown_rew
                    + errors_here + visits_here + recent + SQUARE
    known:   E(L) ~ current + goal + next_step + reward + on_route
                    + errors_here + visits_here + recent + SQUARE

**SQUARE = eight location dummies, and adding them fixed C2.** Occupancy runs
from 8.9% at square 1 to 14.8% at square 5, so evidence carries a
square-identity baseline unrelated to any role. C2's unexplained knowledge
control `unknown_rew` = +0.218 (which must be zero) falls to **+0.033, p = 0.80**
once the dummies are in. The residual was square identity, not knowledge.

`on_route` is measured, not assumed: the squares actually walked on the correct
repeats of that grid run. Per run: 4 rewards, 3.51 on-route non-reward, 1.49
off-route squares.

**I12 (on_route, known phase): null.** HC_all +0.169 in ripple (p = 0.26),
ripple−flank +0.177 (p = 0.18); HC_anterior ripple−flank +0.266 (p = 0.082).
`reward` is flat (−0.042). No evidence for "sends the goals" OR "sends the route".

**I13 (errors_here, explore phase): null.** HC_all −0.118 (p = 0.37),
ripple−flank −0.152 (p = 0.17). Negative as SK predicted, not significant. Run
as a graded count and restricted to exploration because by the end of a first
traversal a median of 6 squares already carry an error history, so a binary
contrast has almost nothing to compare.

## 2026-09-17 (m) — The nulls were underpowered, and cell selection fixes half of it

**Every ripple-vs-flank null in this line could only have detected an effect of
60-173% of the ambient location signal** (80% power, session-level):

| test | n | effect | 95% CI | MDE | as % of ambient |
|---|---|---|---|---|---|
| Stage 3 C1, HC_anterior | 51 | +0.119 | [−0.156, +0.395] | 0.392 | 100% |
| Stage 3 C1, HC_mid | 35 | −0.050 | [−0.387, +0.286] | 0.477 | 157% |
| roles known/on_route, HC_all | 51 | +0.177 | [−0.087, +0.440] | 0.374 | 126% |
| roles explore/errors_here, HC_all | 51 | −0.152 | [−0.372, +0.069] | 0.314 | 60% |
| I11 known−nonrew, HC_all | 20 | +0.241 | [−0.220, +0.702] | 0.651 | 112% |
| pseudo-population, HC_all | 14 | +0.075 | [−0.103, +0.252] | 0.249 | 173% |

So "not ripple-specific" means "no effect that would roughly DOUBLE location
coding". It does NOT mean no effect.

### Cell selection: drop cells whose place map is not reliable

Added as a sweep inside `swr_ripple_content.py` and `swr_ripple_content_roles.py`
(`rel_min` = −inf, 0, 0.1, 0.2, 0.3 on the split-half, split-by-configuration
place-map correlation). Selection uses all task time, so it is shared by ripple
and flank alike and cannot manufacture a difference between them.

**It roughly doubles the location signal.** C0, in ripples:

| ROI | all cells | rel ≥ 0.2 | cells kept |
|---|---|---|---|
| HC_all | +0.457 (p = 0.0028) | **+0.864 (p = 1.1e-07)** | 10.1 → 5.0 |
| HC_mid | +0.339 (p = 0.093) | **+0.836 (p = 1.3e-04)** | 7.6 → 4.4 |
| HC_anterior | +0.387 (p = 0.0089) | +0.650 (p = 1.9e-04) | 6.5 → 3.3 |

**C1 rises but stays a trend:** HC_all +0.069 → +0.116 → +0.184 → +0.221
(p = 0.10) → +0.153 across the sweep; HC_anterior reaches +0.256 (p = 0.12).
Positive at 24 of 25 hippocampal cells of the sweep, never significant. In the
roles model, `current` ripple−flank reaches +0.301 (p = 0.057, HC_all, known).

**The mOFC negative artefact is a selection artefact.** mOFC C0 −0.633
(p = 0.016) with all cells, −0.142 (p = 0.59) at rel ≥ 0, +0.282 at rel ≥ 0.3.
It was driven by cells with no place map. Closes a known unknown.

### Two prospective-looking effects, both explained by approach

With tuned cells, two terms looked like planning:
- `next_step` (explore): HC_all **+0.346, p = 0.021**; HC_mid +0.538, p = 0.037
- `goal` (known, = `looking_for_loc`): HC_anterior **+0.505, p = 0.0041**

An occupancy interval runs from one move onset to the next, so a ripple late in
it happens while the subject is already moving. Splitting at the session median
latency:

| term | all | early half | late half |
|---|---|---|---|
| next_step, explore, HC_all | +0.346 (p = 0.021) | **−0.008 (p = 0.97)** | +0.314 (p = 0.047) |
| goal, known, HC_anterior | +0.505 (p = 0.0041) | +0.264 (p = 0.20) | +0.461 (p = 0.0057) |
| current, known, HC_all (reference) | +0.838 | +0.546 (p = 0.0041) | +0.680 |

`next_step` is **entirely** in the late half — an arrival artefact, not
prospection. `goal` loses significance early while the reference term survives,
so it is not clean evidence of goal coding either. **Neither is reported as a
prospective signal.**

## 2026-09-17 (n) — Peri-ripple pseudo-population time course

`scripts/swr_pseudo_timecourse.py`. Location signal in 50 ms windows from −500 to
+500 ms around the ripple peak, pooled across the sessions that ran the same
configuration, grouped by current / goal / next square.

HC_all current-square signal peaks at **+25 ms (+0.306)** against a baseline of
+0.086 at |t| ≥ 300 ms, but peak-vs-baseline is **p = 0.16** (10/14
configurations up). `goal` and `next_step` are flat. So there is a numerical
rise locked to the ripple that the data cannot resolve.

⚠ Each cell's window comes from a different ripple, so only ripple-LOCKED
structure survives the averaging. Classical random-start replay would average
away, and a null here does not exclude one. That remains untested in this
dataset and is the main thing this analysis line has not looked at.

### Inclusion change logged with (m)

Adding `ROI_SETS` to `swr_ripple_content.py` also changed the cell requirement
from "at least one cell" to **at least two**, because a one-cell "population"
cannot carry a population code and `zscore_cells` leaves it as a single
z-profile. This is not neutral and is logged rather than absorbed: HC_mid falls
35 → 29 sessions and the all-cells Stage 3 numbers move from C0 +0.304 / C1
−0.050 to **C0 +0.339 (p = 0.093) / C1 −0.169 (p = 0.34)**; HC_anterior is
51 → 44 sessions, C0 +0.391 → +0.387, C1 +0.119 → +0.126. The conclusion is
unchanged in both. Figure 5 now filters to the all-cells rows explicitly — the
sweep would otherwise have been averaged into it silently.

## 2026-09-17 (o) — CORRECTION to (m): the cell-selection gain was selection bias

**Entry (m) is wrong and is retracted here.** It reported that selecting
location-tuned cells "roughly doubles the location signal" (HC_all C0
+0.457 → +0.864). It does not. Two biases produced that number:

1. **The reliability peeked.** `spt.reliability` was computed over ALL
   configurations, including the one being scored. Cells were therefore chosen
   partly because their place map fitted the held-out data.
2. **Sessions were dropped.** A session whose cells all fell below the cut
   left the analysis entirely (HC_mid 35 → 21), so the surviving set was
   selected too.

`spt.reliability` now takes `exclude_grid` and `spt.weighted_loo` re-estimates it
for every held-out configuration. Decisive test, same estimator, same 41
sessions:

| | C0, HC_all |
|---|---|
| reported in (m) (peeking + session loss) | +0.864 |
| honest reliability, same 41 sessions | **+0.537** |
| honest reliability, all 51 sessions | **+0.444** |
| no weighting at all | **+0.457** |

**Weighting cells by location tuning buys nothing.** Across
`all / thresh_0.1 / thresh_0.2 / linear / square`, HC_all C0 = +0.457, +0.418,
+0.444, +0.424, +0.418 and C1 = +0.069, +0.089, +0.119, +0.041, +0.013. The MDE
moves 0.423 → 0.412. **I15's first lever does not work**; the power problem is
not untuned cells.

Also retracted from (m): "the mOFC negative artefact is a selection artefact".
With honest weights mOFC C0 goes −0.633 → −0.279 (thresh_0.2) → −0.182 (square)
— it softens but does not reverse, and the "+0.282" figure came from the biased
version. The mOFC negative is still an open problem.

The roles regression was re-run on the honest weighting. **`goal` collapses**:
HC_anterior known-phase was +0.505 (p = 0.0041) under the biased selection and
is **+0.149 (p = 0.36)** honestly. There is no goal coding. That finding existed
only because of the bias.

## 2026-09-17 (p) — Adjacency control, and why Figure 4 decodes the NEXT square

SK: *"what I find very suspicious is that you seem to be able to decode the next
location rather than the current one from the regression you ran in Fig 4."*
Right, and it has two causes, both now controlled.

**1. `next_loc` is always ADJACENT to the current square, and templates are
spatially smooth.** The pseudo-population evidence matrix already showed
neighbours sharing the diagonal's elevation. An `adjacent` term (4-connected
neighbours, column-major grid) is now in the roles model, and it is itself
positive: HC_all +0.255 (p = 0.057), HC_anterior +0.323 (p = 0.034). Some of
what looked like "coding where I go next" was the map being blurry.

`next_step` **survives** the adjacency control — HC_all explore +0.266
(p = 0.042) with all cells, +0.457 (p = 0.0012) with tuned weights, the largest
non-current term in the analysis — but it is NOT ripple-specific (ripple−flank
+0.183, p = 0.18), and the latency control that would say whether it is
prospection or approach **flips sign with the weighting scheme** (early/late =
−0.02/+0.28 unweighted, +0.32/+0.13 weighted). Unresolved, and the most
promising lead left.

**2. Figure 4's labels expire.** Measured, pooled over 61 sessions: at lag 0 the
subject is on the arrival square by definition; at **+0.44 s they are more often
on the NEXT square (median dwell 0.367 s)**; at +0.5 s it is 71% next vs 54%
current.

| lag | on arrival square | on previous | on next |
|---|---|---|---|
| −0.25 s | 0.29 | **0.85** | 0.03 |
| 0 s | **1.00** | 0.35 | 0.35 |
| +0.50 s | 0.54 | 0.18 | **0.71** |
| +1.00 s | 0.24 | 0.11 | **0.43** |

So HC_mid's `b_next` peaking at +0.4 s while `b_current` sits negative is the
subject having walked there — not prospection. `occupancy_by_lag.csv` is now
written beside the time course and plotted as row 3 of Figure 4, and the caption
states that only |t| < 0.25 s is interpretable.

**This also resolves the Figure 4 / Stage 2 discrepancy** flagged since
2026-09-17 (e): the three role regressors are a chain of mutually adjacent
squares whose labels drift out of date within one dwell, so the regression
splits one blurred bump between them. Figure 4 is no longer "unresolved" — it is
explained, and correspondingly narrowed.

### Figures 5 and 6 for location-encoding cells (SK's request)
Both now take a `scheme` argument and are written twice, `_tuned` using cells
whose place map generalises across configurations. Conclusions are unchanged:
Fig 5 tuned — HC_mid C0 +0.353 (p = 0.093), C1 −0.067; HC_anterior C0 +0.399
(p = 0.012), C1 +0.165 (p = 0.29). Fig 6 tuned — I11 known−nonrew −0.279
(p = 0.23). The tuned and untuned versions agree, which is the point of (o).

## 2026-09-17 (q) — Window audit, future goals, correct trials, phase-split errors

**Window audit (SK asked).** Confirmed: every content analysis uses each
ripple's OWN detected duration, `t_peak ± duration/2`. Durations are median
**60 ms**, IQR 46–80, range 38–441 (n = 100,737) — the "60 ms" in the write-ups
is the median, not a setting. Each flank inherits its own ripple's half-duration
(`w_fl = half[owner]`), so widths match exactly, and sits inside that ripple's
occupancy interval. `±500 ms` was never a width — it was the old flank
*position*, abandoned because it was location-matched only 53.7% of the time.
Fixed widths survive only in the Stage 2 positive-control sweep
(60/125/250/500/1000 ms), which is never compared with ripples.

**`scripts/swr_ripple_content_c2.py` deleted.** It was the last script still
placing flanks at ±500 ms, and it is superseded by the roles regression, which
does the same job with square dummies and matched flanks.

### New terms and splits in the roles regression
- `goal_1`, `goal_2` — the rewards sought AFTER the current one (the sequence
  loops). During exploration these are unknowable, so they double as a
  knowledge control.
- `subset` = all / correct trials.

**All three goal horizons are null** (ripple−flank, HC_all): `goal` +0.053
(p = 0.74), `goal_1` +0.022 (p = 0.88), `goal_2` −0.013 (p = 0.93) in the known
phase; same in exploration and in every ROI. **No prospective goal coding at any
horizon.**

### `on_route` in HC_mid — the strongest candidate, and it fails robustness
| variant | n | ripple−flank | p |
|---|---|---|---|
| all cells, all trials | 29 | **+0.530** | **0.0087** (subject-level 0.041) |
| all cells, correct traversals only | 29 | +0.424 | 0.032 |
| **location-tuned cells only** | 28 | **−0.172** | 0.44 |
| HC_all / HC_anterior | 51 / 44 | +0.150 / −0.044 | 0.35 / 0.80 |

Specific to HC_mid, phase-specific by construction, survives a correct-trials
restriction — but **reverses sign** under cell weighting, and is one cell of a
~200-test family (Bonferroni would need p < 0.0003). Not established. Deserves
one pre-specified replication, not more exploration.

### `errors_here` by phase — right direction, not significant
HC_all ripple−flank: explore −0.138 (p = 0.29), known +0.146 (p = 0.33),
interaction −0.284 (p = 0.19). More suppression during exploration and less once
everything is known, exactly as SK predicted, but underpowered.

### Presentation fixes
- **Figure 8c now plots in-ripple AND ripple-minus-flank side by side.** Showing
  only the in-ripple bar invited the misreading SK caught: the figure looked
  significant where the summary table was null, because they were different
  quantities. `current` is the clearest case — +0.526 (p < 0.0001) in ripple,
  +0.024 (p = 0.83) against its own flank.
- **Figure 9 added**: the peri-ripple pseudo-population time course, which had
  numbers but no plot. HC_all current-square signal peaks +0.306 at +25 ms
  against a +0.086 baseline; not significant (p = 0.16).
- The summary table in `SUMMARY_ripple_content.md` now carries ROI, phase, n and
  measure per row. Row 8 was **mislabelled** as "anything" — the
  pseudo-population tested the current square only, i.e. it is a better-powered
  row 1, not a test of rows 2–7.

## 2026-09-17 (r) — Route and error squares: controls, phase, and descriptives

### `on_route` (I12) in HC_mid — new controls, all passed but one

Added a **template-occupancy covariate**: for each square, log dwell at that
square across the OTHER configurations, i.e. the data its leave-one-out template
is built from. This was the suspected artefact — off-route squares are by
definition rarely visited, so their template entries are the worst estimated,
and `on_route` is scored against exactly them.

**The effect is unchanged**: HC_mid ripple−flank +0.530 (p = 0.0088), and the
covariate itself is null everywhere (HC_all +0.060, p = 0.67). Artefact ruled out.

**Phase dependence (SK asked):** split the known phase at the median
`trial_num_in_grid`:

| subset | HC_mid ripple−flank | p |
|---|---|---|
| all trials | +0.530 | 0.0088 |
| correct traversals only | +0.403 | 0.045 |
| **early repeats** | **+0.578** | **0.0043** |
| late repeats | +0.125 | 0.53 |
| location-tuned cells only | −0.156 | 0.49 |

The route is represented in ripples **when it has just been learned and not
later** — the same shape as He et al.'s 'stable→' replay decaying with
familiarity.

⚠ The cell-weighting failure is **systematic, not noise**: paired across the same
28 sessions, all-cells minus tuned-cells = +0.728, p = 0.00033, r = +0.64. The
effect is carried by cells whose place map does NOT generalise across
configurations. SK's reading is that non-local trajectory content need not live
in the most current-location-tuned cells, which is reasonable; the difficulty is
that in this estimator a cell's template is its only channel to contribute
location evidence, so an unreliable template should contribute noise, not
structure. **Unexplained. This is the thing to resolve before believing row 5.**

### ⚠ The raw templates carry an occupancy bias (Figure 10d)
Unadjusted within-window z by role, HC_mid, known phase: **off_route +0.039**,
current +0.014, reward +0.004, adjacent −0.003, **on_route −0.013**. The
rarely-visited squares score HIGHEST. This does not invalidate the
ripple-minus-flank contrasts, which compare the same square with itself, but it
does mean **no raw across-square comparison in this dataset is interpretable**,
and it is why the model needs square dummies and the occupancy covariate.

### `errors_here` (I13) by ROI and phase
Ripple−flank, all cells: HC_all explore −0.118 (p = 0.37) / known +0.132
(p = 0.38); HC_mid −0.084 / +0.036; HC_anterior −0.054 / +0.098. In-ripple
HC_anterior known +0.284 (p = 0.061). Interaction p = 0.19. The predicted
direction — suppression while exploring, elevation once known — is present in
all three ROIs and significant in none.

### New: `scripts/swr_roles_timecourse.py` and Figures 10, 11
Descriptive only: within-window z by square role in 100 ms windows from −750 to
+750 ms around the ripple peak, no null, no model. Both role contrasts are flat
across the peri-ripple second (on_route−off_route, HC_mid known: peri −0.053 vs
non-peri −0.049; error−no_error: −0.005 vs −0.007). **The regression effects are
partial coefficients and do not appear in the unadjusted data** — stated on the
figures so nobody reads c/d as the result.

Figure 10 = the walked route with every control; Figure 11 = error squares by
ROI and phase, time-resolved. Both mark the He et al. peri-ripple band (±250 ms).

## 2026-09-17 (s) — RESULT: a ripple-specific route signal in HC_mid

**Logged as a result at SK's request.** First and only ripple-specific effect in
this analysis line.

**Claim.** Once all four rewards are known, mid-hippocampal ripples carry more
information about the squares the subject actually walks through — over and
above the current square, its neighbours, the rewards, recency, visit count,
square identity and template occupancy — than matched flank windows in the same
occupancy interval do. The effect is present in the early repeats after learning
and absent in the later ones.

| | HC_mid, ripple − matched flank | p |
|---|---|---|
| **known phase, all trials** | **+0.530** | **0.0087** (subject-level 0.041) |
| correct traversals only | +0.403 | 0.045 |
| **early repeats** | **+0.578** | **0.0043** |
| late repeats | +0.125 | 0.53 |
| exploration phase | n/a — no route exists yet | — |
| HC_anterior / HC_all / mOFC / mPFC | −0.046 / +0.137 / −0.27 / −0.05 | all n.s. |

n = 29 sessions. `on_route` = squares walked on the correct repeats of that grid
run, excluding rewards, the current square, its 4-connected neighbours, the
goals and anything visited in the last 10 s. Reference = off-route non-reward
squares. Estimator, CV, flank matching and null as everywhere else in this line.

**Controls passed:** template-occupancy covariate (the suspected artefact — off-
route squares are the rarely visited ones; covariate itself null, effect
unchanged); correct-trials restriction; subject-level inference; square dummies;
adjacency term.

**Control failed:** weighting cells by cross-configuration place-map reliability
(−0.156, p = 0.49), and the disagreement is systematic (paired +0.728,
p = 0.00033). The effect lives in cells whose place maps do not generalise.
Unexplained.

**Statistical standing:** one cell of a ~200-test family; Bonferroni would
require p < 0.0003. **Exploratory. Needs a pre-specified replication.** The
honest one-line version: *"in mid-hippocampus, ripples recruit the newly-learned
route more than surrounding time does — exploratory, p = 0.0087 uncorrected,
and not robust to one of five controls."*

### Figure 10 rebuilt after SK asked why panel c looked flat
The old panel c plotted an unadjusted descriptive time course, which is NOT the
tested quantity, and read as though the result were absent. The figure now
shows: (a) per ROI, (b) every control, (c) **the test itself** — each session's
route coefficient in its ripple against its own matched flank, paired — (d) the
raw role means carrying the occupancy warning, (e) the descriptive time course
labelled as not-the-test, and (f) four reasons c and e differ: partial vs raw
coefficient, flank inside the occupancy interval (~100 ms) vs a ±250 ms band
that mostly sits on a different square, ripple-duration vs fixed 100 ms windows,
and z-vs-null vs raw across-session means.

## 2026-09-17 (t) — Do ripples cluster, and should the window be ±250 ms?

SK asked whether the He et al. ±250 ms peri-ripple window would give more signal
than the ripple's own ~60 ms. Measured, all 61 sessions:

**Ripples barely cluster.** Inter-ripple interval median 1.064 s (10th
percentile 0.146 s). 30.4% of ripples have another within 250 ms. A ±250 ms
window contains on average **1.39** ripples against a Poisson expectation of
1.27 at the observed 0.535 Hz — a clustering factor of **1.10×**. So widening the
window buys almost no extra ripples.

**Width does help a little, up to a point, and only in HC_mid** (Stage 2 sweep,
z vs permutation null): HC_mid +0.640 / +0.647 / **+0.684** / +0.267 / −0.013 at
60 / 125 / 250 / 500 / 1000 ms; HC_anterior declines monotonically +0.488 /
+0.439 / +0.315 / +0.168. The collapse at 500 ms is not mysterious: the median
dwell is 0.367 s, so a 500 ms window spans two squares and the label is wrong
for part of it.

**The binding constraint is the dwell, not the choice of window.** A ±250 ms
window stays inside the ripple's own occupancy interval only **29.9%** of the
time, against **88.7%** for ±30 ms. Since every ripple-specific claim in this
line rests on a flank matched on the same square, inside the same interval, a
500 ms-wide window makes that control impossible for 70% of ripples.

**Conclusion.** Keep the ripple's own duration for anything needing a
same-square flank — the current design is at the ceiling the behaviour allows.
Use ±250 ms where the window does not have to sit on one square: cortical
responses to ripples (what He et al. use it for, and what this project's HFB
branch already does) and the TDLM sequence analysis, where the peri-ripple
window is a period to search for order within, not a single feature window.

### (i) Consolidation — 12 scripts archived, summary written
The ripple-RSA work collapsed to two scripts plus the library:
  mc/analyse/ripple_rsa.py            the library, incl. `DECISIONS`
  scripts/swr_ripple_rsa_alluncovers.py   `run` + `reliability`
  scripts/swr_ripple_rsa_diagnostics.py   `leverage` + `weighting`
    (fused from the former conditions.py and inputs.py, same numbers)
Twelve superseded scripts moved to
`derivatives/group/swr/archived_scripts/ripple_rsa_2026-09-17/` with a README
giving, for each, why it was written and why it was archived. EIGHT OF THEM
WERE NEVER COMMITTED, so that folder is their only copy -- do not delete it
expecting git history to have them.
`swr_ripple_rsa_alluncovers.py run` now takes `--estimators`, so the
centre-vs-zscore-vs-crossnobis comparison is reproducible without the archived
crossnobis script.
Full write-up: `derivatives/group/swr/ripple_rsa_SUMMARY_2026-09-17.md`
(what was tested, what broke, the reliability ceiling, reproduction commands).

### (j) Diagnosis of the null — spike sparsity, not sample size
`scripts/swr_ripple_rsa_diagnostics.py power` (third subcommand).
Ripples are 60 ms; **87.0% of (cell, ripple) observations contain zero
spikes**; the median cell contributes **1 spike per condition** across all its
ripples. Observed across-condition CV barely exceeds the Poisson floor
(per-cell SNR 0.46-0.66 in every ROI).
Split-half reliability is FLAT from 1.5k to 12.6k ripples (8x, no trend) but
rises MONOTONICALLY with integration half-width in all five ROIs
(±10 ms ~0.02 -> ±500 ms 0.09-0.16, a 4-10x gain).
=> The binding constraint is spikes per observation, not number of
observations. More sessions cannot fix this. Caveat: ±500 ms is a 1 s window
and no longer ripple-specific, so that gain measures sustained firing, not
ripple coding.
Written up in the ADDENDUM of ripple_rsa_SUMMARY_2026-09-17.md.

## 2026-09-18 (u) — I14 gate: location is NOT decodable from iEEG band power

`scripts/swr_ieeg_location_decoder.py`. The prerequisite for any within-ripple
sequence analysis: TDLM needs a reactivation probability per state per timepoint,
so before running one, can a 9-way location decoder beat chance at all?

**Answer: no**, in every region and at both cross-validation levels. Per visit
(one occupancy interval, median 0.367 s — the same unit the spike analyses use,
where location IS decodable at z = +0.64):

| feature set | CV | bal. acc | null | p | above null |
|---|---|---|---|---|---|
| all contacts | configuration | 0.0994 | 0.1111 | 8.5e-07 | 5/61 |
| **all contacts** | **run** | **0.1082** | **0.1111** | **0.166** | 17/61 |
| HPC | configuration | 0.0944 | 0.1111 | 1.5e-14 | 3/61 |
| HPC | run | 0.1067 | 0.1111 | 6.2e-08 | 12/61 |
| mPFC | configuration / run | 0.0942 / 0.1056 | 0.1110 | 9.3e-10 / 3.3e-06 | 1/41, 7/41 |
| mOFC | configuration / run | 0.0931 / 0.1059 | 0.1111 | 1.1e-12 / 2.6e-09 | 0/47, 5/47 |
| Visual | configuration / run | 0.0955 / 0.1067 | 0.1112 | 2.3e-08 / 0.0039 | 2/36, 9/36 |

(All contacts, 61 sessions, 292,308 visits. Numbers are the canonical script
run; earlier quotes in conversation differed in the third decimal because the
RNG stream changed when the script was rewritten.)

The null lands at exactly 1/9, as balanced accuracy should. **SK's proposal did
what she predicted** — holding the configuration constant lifts every ROI
(0.093–0.099 → 0.106–0.108) and removes the below-chance artefact in the
best-powered set, confirming that a cross-configuration decoder learns each
configuration's occupancy and route and is then handed a different one. But it
reaches chance and stops there: the closest any cell of the table comes is
all-contacts/run at p = 0.166, and every focal ROI stays significantly BELOW
chance even with the leak. Band power carries no
location information at the unit where spikes carry plenty.

This is the same lesson as Figure 10d, where rarely-visited off-route squares
carried the HIGHEST raw evidence: **occupancy structure is the dominant signal
in this task, and any decoder trained across configurations finds it first.**

### ⚠ Three results were reported from this branch before it was validated, and
### all three were wrong. They are retracted here.

A first version reported a confident "clean negative, definitive" from a decoder
that could not decode **anything** — not location, not the 4-way state
(0.245 vs 0.240 chance), not even an uncover press against a window 1.5 s later
(0.487 vs 0.500, 0/14 sessions above 0.55, with ~1,000 events per session). A
recommendation to stop I14 was made on that null. All of it is withdrawn:
the below-chance config result, the run-level comparison, and a
regularisation × smoothing sweep.

**The fault was posing, not code.** The ripple-locked HFB response is about one
sample wide (+0.197 at 0 ms, +0.017 at ±100 ms). Averaging features over a
400 ms window diluted it ~40×, and sampling every 10 ms produced 24,000 rows
whose effective n was far smaller — band power is 0.94 autocorrelated at lag 1.
Both decoders asked band power to work at a timescale where its signal is
smeared or drowned.

**The loader was never the problem** and is now verified rather than assumed:
it reproduces this project's own established peri-ripple effect — HPC HFB peak
**+0.197 at exactly 0.00 s** (z = +7.3), frontal **+0.012 at −0.02 s**
(z = +3.2) — which also confirms sample 0 = session time 0 to within one sample.

**Process change, baked into the script.** `positive_control()` runs FIRST and
`main()` refuses to report a decoding null unless the HFB peak is clearly
positive and lands within one sample of zero. A null from an unvalidated
instrument says nothing about the brain, and this branch cost three retractions
to learn it.

### What this does and does not mean
It does **not** say replay is absent — nothing here tests replay. It says this
dataset cannot supply the decoder that a sequence analysis needs, on these
features. Remaining options are in `POTENTIAL_IDEAS.md` under I14.

## 2026-09-18 (v) — Every signal, one decoding design: argmax fails for all of them

`scripts/swr_decoder_comparison.py`, Figure 12. Built for SK's talk: the I14
gate tested band power, but the obvious question is whether spikes do better.
Same design throughout — one row per visit, 9-way multinomial, leave-one-
configuration-out, balanced accuracy against a label-permutation null.

| signal | sessions | features | bal. acc | null | p |
|---|---|---|---|---|---|
| HFB (high gamma) | 61 | 24.3 | 0.0958 | 0.1110 | 7.7e-11 |
| theta | 61 | 24.3 | 0.0948 | 0.1112 | 8.7e-17 |
| beta | 61 | 24.3 | 0.0938 | 0.1111 | 3.0e-16 |
| ripple band | 61 | 24.3 | 0.0953 | 0.1112 | 9.8e-14 |
| all bands | 61 | 73.0 | 0.0994 | 0.1111 | 9.7e-07 |
| **spikes (HC)** | **51** | **10.1** | **0.0955** | **0.1111** | **7.7e-12** |

**Spikes fail too.** That matters more than the band-power null, because the
continuous estimator finds location in exactly those spikes: HC_mid **+0.640**
(p = 0.0075), HC_anterior **+0.488** (p = 0.0015) in 60 ms windows.

**So the conclusion is not "there is no location signal" — it is that 9-way
argmax is the wrong read-out.** The evidence matrix shows why: diagonal +0.144
against off-diagonal −0.018, but neighbouring squares share the elevation, so
evidence that is clearly raised at the true square still peaks one square away.
Argmax discards a graded signal; the continuous score keeps it.

**Consequence for I14, restated precisely.** TDLM needs a per-state
reactivation probability, which is an argmax-family read-out. No signal in this
dataset supports one. The sequence analysis has no front end — not because
location is absent, but because the read-out it requires is the one that fails.

### Figures built for the presentation (2026-09-18 set)
- **Fig 12** — decoder comparison: every signal at chance (a), the same spikes
  with the continuous estimator (b), and the evidence matrix explaining the
  difference (c).
- **Fig 13** — reward locations vs the current location, in-ripple and
  ripple-minus-flank, with `reward not yet found` as the built-in knowledge
  control sitting at zero.
- **Fig 14** — every regressor in the route model, both measures, for
  HC_mid/known (where the effect is) and HC_all/explore.

## 2026-09-18 (w) — Collinearity audit of the roles model, and the post-uncover power limit

SK: *"a lot of the regressors are pretty overlapping, and worst case even fully
contain another... how correlated are they? What happens if you remove some
co-regressors?"* — `scripts/swr_roles_collinearity.py`.

**The design is not collinear in the damaging sense.** The binary roles are
mutually exclusive by construction, so no regressor contains another, and the
variance inflation factors are all **1.3–2.3** (problem threshold 5), including
`on_route` at 2.16 — the thin off-route reference category (~1.5 squares per
run) does not make the design ill-conditioned.

Largest correlations between role regressors: **`on_route` ↔ `recent` r = +0.625**,
then `errors_here` ↔ `visits_here` +0.362, `goal` ↔ `next_step` +0.305, all
others below 0.26.

**Leave-one-regressor-out on the route effect** (HC_mid, known, ripple−flank):

| model | z | p |
|---|---|---|
| full | +0.530 | 0.0088 |
| minus current | +0.622 | 0.0030 |
| minus goal_1 | +0.610 | 0.0030 |
| minus reward / goal_2 / adjacent / goal | +0.560…+0.569 | 0.0053–0.0067 |
| minus next_step / errors_here / visits_here / train_occ | +0.520…+0.530 | 0.0072–0.0100 |
| **minus recent** | **+0.369** | **0.046** |

Stable against dropping any rival except `recent`, where it halves but stays
significant. `recent` carries a NEGATIVE coefficient (≈ −0.30) and correlates
+0.625 with `on_route`: recently-visited squares carry less ripple-specific
evidence, and because route squares are usually recent ones the two partly
cancel until recency is controlled. **The route effect is therefore conditional
on separating "on the route" from "just been there"** — a real claim about the
model and the effect's one load-bearing covariate.

### The post-reward-uncover subset cannot be fitted, and the number says why

SK asked whether Fig 13 uses all ripples or only those after a reward uncover.
**All ripples in the phase**, wherever the subject was. Adding `after_reward` /
`after_error` subsets produced no usable sessions:

| subset | ripples/session (median) | sessions with ≥ 50 |
|---|---|---|
| exploration, all | 241 | 61/61 |
| **after a rewarded uncover** | **16** | **2/61** |
| after an erroneous uncover | 67 | 37/61 |

The regression has 21 parameters. A median of 16 ripples cannot support it.
That question is answerable only by a design built for small n — which is what
I11 is (matched triplets rather than regression), and its answer stands:
known-reward vs non-reward squares during exploration, +0.241, p = 0.29.
Figure 13c now shows that instead of an unfittable regression. ⚠ `after_error`
IS feasible in 37 sessions and has not been run.

### Model change: `known_rew` in exploration is now nominally negative
With `goal_1`, `goal_2` and `train_occ` added, HC_all exploration `known_rew` is
−0.350 in ripple (p = 0.016) and −0.349 ripple−flank (p = 0.018), up from −0.266.
Already-found reward squares are represented LESS than other squares. Wrong sign
for the hypothesis, one of ~200 tests, and not to be presented as a finding —
but it should not be hidden either.

## 2026-09-18 — He et al. peri-ripple windows, hippocampal units: still null

Results: `data/ephys_humans/derivatives/group/swr/ripple_rsa_periripple_2026-09-18/`
Code: `scripts/swr_ripple_rsa_periripple.py`, `rrsa.cache_band_rates`

Peri-ripple (-250 to +250 ms) vs non-peri-ripple (both flanks, -750/-250 and
+250/+750 combined), ripple-shuffled surrogate null (500 draws, pseudo-peaks
matched in number per event), hippocampal single units, 32 conditions
(config x uncover), z-scored, pad 0.25. Two event sets as requested:
explore-phase discoveries and all correct uncovers.

### (a) Two design problems had to be fixed first
1. Under the standing 1 s post-press selection cap, **100% of +-750 ms flanks
   fell outside their own inter-uncover interval** -- the early flank reaches
   back before the uncover. The cap and these windows are incompatible.
2. At the all-uncovers event density the **median inter-uncover gap is 1.35 s,
   shorter than the 1.5 s window itself** (explore: 5.45 s).
Fix: run on UNCAPPED intervals and require the full +-750 ms to lie inside one
interval, so both windows describe one condition. Keeps 3794 explore (71%) and
14091 all-uncover (47%) ripples. Surrogates drawn from the same
clearance-respecting sub-interval.

### (b) A duration confound in the He et al. design, worth knowing
Their baseline is BOTH flanks = 1000 ms against a 500 ms peri window.
Reliability scales with integration time, so the baseline is intrinsically
better measured. Split-half reliability of the data RDM (HC_anterior):
                     explore   all
  peri (500 ms)       +0.046  +0.072
  nonperi (1000 ms)   +0.122  +0.215
  nonperi_late (500)  +0.032  +0.144   <- duration-matched control, added here
So a peri-vs-nonperi contrast is biased AGAINST peri unless duration-matched.
`nonperi_late` (+250 to +750 ms alone) was added to `rrsa.BANDS` for this.

### (c) RESULT: 2 of 36 significant uncorrected, NOTHING survives FDR
  explore | peri         | HC_mid | full_abcd  rho +0.045  z 1.78  p 0.048
  all     | nonperi_late | HC_mid | known_set  rho +0.123  z 1.91  p 0.020
The larger of the two is in the NON-ripple window. No peri-ripple-specific
effect. Peri-minus-nonperi contrasts are small and inconsistent in sign across
ROIs and event sets.

### (d) What this does and does not change
CONFIRMS the sparsity diagnosis: the 500 ms peri window raises reliability
from ~0.02 (ripple-internal) to 0.046-0.072, exactly as the window sweep
predicted. The extra spikes are real.
DOES NOT rescue the result: duration-matched, peri is no better than ordinary
non-ripple time (worse in the all-uncovers set, 0.072 vs 0.144; marginally
better in explore, 0.046 vs 0.032). So the gain is INTEGRATION TIME, not
ripple alignment.
Caveat to carry: 500 ms is ~8x the 60 ms ripple, so this tests ripple-ALIGNED
rather than ripple-internal coding -- a weaker claim than the project's.
One reading worth testing rather than asserting: a ripple is a stereotyped
population burst, so peri-ripple firing may be LESS condition-discriminative
than ongoing task firing.

## 2026-09-19 — cell/fMRI gradient panels now share ONE gradient map

### The discrepancy
The cell<->fMRI gradient brain figure (`cell_fMRI_angle_match.render_brain`)
and the fMRI gradient figure in the manuscript
(`harmonic_maps_brain_overlay`, `angle_deg__gradient_thr1.5__circ_alphaR_
top100_opaque_g1.2_dsr_outline_ribbonmax`) were built from DIFFERENT volumes:

  cell figure : harmonic_angle_maps/unit_vector_derived/quarters, 0 mm smoothing,
                mask -> surface by vol_to_surf nearest >= 0.5   (2958 rh verts)
  fMRI figure : harmonic_angle_maps/quarters (magnitude-weighted), 3 mm FWHM,
                mask -> surface by ribbon projfrac-max (11 depths) + close 2
                rings + drop patches < 40 verts                 (3507 rh verts)

Angle difference where both masks overlap (rh pial): median 6.4 deg, mean 11.2,
p90 27.9. Decomposed: map source median 5.8 deg, 3 mm smoothing median 1.9 deg.
The bigger VISUAL difference is the mask projection (~550 vertices, fatter and
smoother patch edges), plus per-vertex alpha = (R/cutoff)^1.2 in the fMRI figure
vs uniform opacity in the cell figure.

### Which map is the honest one: magnitude-weighted
unit_vector_derived normalises every subject's vector to length 1 before
averaging (one vote per subject); magnitude-weighted is the plain group mean of
the per-subject RSA vectors. In the gradient mask the two angle maps differ by
median 13.3 deg (22% of voxels > 30 deg), so the choice is not cosmetic.
The usual reason to prefer unit-vector — one high-amplitude subject dragging
the group direction — does NOT apply here: across the 33 subjects the largest
share of total in-mask vector length is 4.3% (3.0% would be perfectly even) and
strongest/weakest is 2.5x. So the magnitude-weighted map is the less-transformed
one and is what the manuscript's fMRI gradient + DSR main effect already use.

### What changed
`harmonic_maps_brain_overlay.render_cell_gradient_panels()` renders four panels
per hemisphere from ONE projection of that map (magnitude-weighted, 3 mm,
gradient_thr1.5 ribbon-max gate, alpha-by-R):
  A gradient only | B + DSR main-effect outline | C + all cells grey
  D + cells coloured by group preferred lag, with split axis/boundary
Written to .../cell_gradient_master/2026-08-28_15-19-35/final_splits/ as
`cellpanel_{A..D}__quarters_gradient_thr1.5__{lh,rh}_medial.{png,pdf}`.
`_save_brain_figure` was factored out of `render_one` so both paths share one
colourbar. The earlier `backdrop_only_*` / `cells_all_grey_*` figures (built on
the unit-vector backdrop) were deleted.

### STILL OPEN
`per_cell_master.csv` (run 2026-08-28_15-19-35) sampled the per-cell fMRI angles
from unit_vector_derived at 0 mm. So the fMRI angles QUOTED for these cells
(`fmri_z_readout.csv`, the "fMRI58/fMRI61" subtitle) still come from the
unit-vector maps, while the panels now show the magnitude-weighted map. The cell
lags (30/60 deg) and the ventral/dorsal split are unaffected — the split axis is
mask GEOMETRY (PC1) and the lags are cell data only. Re-running
cell_gradient_master_table.py with harmonic_root = the magnitude-weighted maps
(+ 3 mm) would make the quoted numbers match the picture.

### CORRECTION (same day): panels use the UNIT-VECTOR maps after all
The recommendation above was made on the wrong criterion. "No single subject
dominates the group vector" (max 4.3% of in-mask vector length) rules out one
outlier subject distorting the map; it does NOT address what magnitude
weighting actually costs here, which is a weak-but-consistent direction being
outvoted by co-located stronger effects. The -90 deg territory is real and is
the weakest of the effects in this mask, so amplitude weighting suppresses
precisely the effect least able to defend itself:

  voxels in gradient mask with preferred angle in [-135, -45]
    magnitude-weighted   3.7%
    unit-vector          6.5%

Equal subject weight is the display-neutral choice, so
`CELL_PANEL_UNIT_VECTOR = True` and the four panels are now rendered from
`harmonic_angle_maps/unit_vector_derived/quarters` (3 mm, gradient_thr1.5
ribbon-max gate, alpha-by-R). Files carry the map source in the name:
`cellpanel_{A..D}__quarters_unitvec_gradient_thr1.5__{lh,rh}_medial.{png,pdf}`;
the magnitude-weighted set was deleted.

Consequence: the panels now match `per_cell_master.csv` (which sampled the
per-cell fMRI angles from unit_vector_derived) — the "STILL OPEN" item above is
resolved, apart from smoothing (panels 3 mm, per-cell sampling 0 mm; a median
1.9 deg difference). What no longer matches is the manuscript's standalone fMRI
gradient figure, which was rendered with USE_UNIT_VECTOR_MAPS = False. To make
that agree, re-run harmonic_maps_brain_overlay.main() with
USE_UNIT_VECTOR_MAPS = True.

## 2026-09-21 (x) — Ripples carry STATE as well as location, and the two look factorised

`mc/analyse/swr_explore/state_content.py`, run as
`python scripts/swr_content_explore.py state`. SK's question: ripples carry the
current square — do they also carry position in the ABCD sequence, and are the
two separable factors ("square 5 while heading to B")?

`state` = the reward being SOUGHT (the steps table's `state`, 1–4 = A–D), SK's
choice. Machinery identical to the location analysis throughout: four templates
per unit z-scored across states, leave-one-configuration-out, per-unit z-scored
counts, the ripple's own duration as the window, per-session permutation null,
t across sessions. ⚠ The state null is EXHAUSTIVE — only 24 label permutations
exist for four conditions, so all 23 non-identity ones are used.

### 1. State is represented, and it is not the drift confound

State advances monotonically within a traversal and correlates with
time-into-traversal at r = 0.40, so a unit that merely drifts upward would fake
a state-D template. Leave-one-configuration-out does NOT help — the drift is
inside every configuration. Two controls:

| ROI | raw | detrended | time-matched |
|---|---|---|---|
| HC_all | +0.349, p = 0.030 | **+0.777, p = 4.7e-07** | +0.384, p = 0.019 |
| HC_mid | +0.435, p = 0.019 | **+0.755, p = 0.00046** | +0.369, p = 0.035 |
| HC_anterior | +0.181, p = 0.24 | **+0.617, p = 1.8e-05** | +0.283, p = 0.093 |
| mOFC | +0.004, p = 0.99 | +0.497, p = 0.054 | +0.038, p = 0.87 |
| mPFC | +0.337, p = 0.12 | **+0.538, p = 0.016** | +0.235, p = 0.25 |

**Detrending makes state STRONGER, not weaker** (HC_all +0.349 → +0.777). A drift
artefact does the opposite: it inflates the raw value and dies under detrending.
Removing the time trend here removes noise. The assumption-free time-matched
control (states compared only within bands of time-into-traversal) also holds in
HC_all and HC_mid. **This is not the confound.**

**A dissociation:** mPFC carries state (+0.538, p = 0.016) but NOT location
(+0.126, p = 0.53), while hippocampus carries both. Consistent with the
project's framing of mPFC as task structure rather than place.

**Not ripple-specific**, same as location: state ripple−flank +0.254 (p = 0.11)
in HC_all, +0.333 (p = 0.038) in HC_anterior.

### 2. The conjunction is the sum of its parts — consistent with factorisation

36 templates, one per (square, state); each window's 36 evidence values
regressed on three non-overlapping indicators — `loc_only` (3 of 36),
`state_only` (8), `both` (1), reference (24).

⚠ **ERROR MADE AND CORRECTED.** The first run reported `both` at p = 2.5e-06 as
the conjunction result. It is not: with dummy coding `both` estimates
(true cell − reference), which is exactly what ADDITIVE location + state coding
already predicts. The test is the INTERACTION, `b_both − (b_loc + b_state)`,
now computed with the same permutation draws.

| ROI | loc_only | state_only | interaction | p | 95% CI |
|---|---|---|---|---|---|
| HC_all | +0.442 (0.0074) | +0.956 (2.6e-06) | +0.108 | 0.47 | [−0.19, +0.40] |
| HC_mid | +0.243 (0.34) | +1.009 (0.00063) | +0.287 | 0.15 | [−0.11, +0.68] |
| HC_anterior | +0.379 (0.035) | +0.675 (0.00065) | +0.016 | 0.92 | [−0.29, +0.32] |
| mPFC | +0.077 (0.74) | +0.660 (0.0077) | +0.147 | 0.39 | [−0.20, +0.49] |

Raw betas: HC_all b_loc +0.031 + b_state +0.046 = **+0.077 predicted** vs
**+0.111 observed**.

**Both marginals are present and the interaction is not — the pattern expected
if square and state are separable factors.** ⚠ But this rests on a null
interaction: the HC_all CI excludes an interaction above **+0.40**, which is
about the size of the LOCATION marginal (+0.442) and 42% of the state marginal.
So the claim is "no conjunction as large as the location effect", not "no
conjunction". **Exploratory; needs a pre-specified replication before it is
described as factorisation.**

### Zero-fill artefact checked and absent
With 36 conditions, unvisited (square, state) cells are zero-filled so the unit
abstains rather than making the window unscorable. Measured: **100% of the 36
conditions are observed in every leave-one-out template**, and the true
condition is no more likely to be observed than any other, so `both` is not
measuring observedness.

## 2026-09-27 — Robustness grid: phase split × baseline window × stillness control

**Question.** Do the four reported results survive (a) 3 vs 2 phases,
(b) a −1.5…−1.0 s baseline instead of −1.6…−1.1 s, (c) the stillness control?

**New scripts.** `scripts/swr_robustness_grid.py` (ripple rate; 3 phase schemes ×
5 readings, every number through `fin.window_stats` / `fin.cluster_test`),
`scripts/swr_hfb_phase_robustness.py` (HFB branch).
`scripts/swr_ripple_hfb_conditions.py`: stillness events now generated in EVERY
phase (were explore-only, so no phase-specific stillness control existed), and
the two-phase `*_learn` (explore+plan) collapse added beside `*_all`.

**Provenance finding (the important one).** All three reported ripple-rate values
come from the **own-baseline** reading, i.e. WITHOUT the stillness-matched
control, although `swr_final_ripple_analysis.py` documents the matched reading as
primary and the own-baseline one as "inflated here". The draft also mixes phase
splits: R1 uses `first uncovers`, R2/R3 use `exploring` (first + while learning).

**Results** (`swr/ripple_robustness_grid_2026-09-27/`, bundle_v2, pad 0.25, n=61
sessions, 0–0.5 s window):

| claim | (a) phase split | (b) baseline B | (c) stillness control |
|---|---|---|---|
| R1 rate rises at correct uncovers | holds for *first uncovers* (+0.0344); fails if pooled with *while learning* (+0.0128, p=0.24) | stronger (p 0.0068→0.0010) | **survives**: +0.0344, t(60)=2.15, p=0.032, cluster +0.35…+0.75 s p=0.003 |
| R2 early > late | 0.029 / 0.075 / 0.019 across the three splits | flips first/later 0.075→0.020 | **fails**: p = 0.075 / 0.149 / 0.323 |
| R3 valence × phase | significant in all three splits | invariant (0.008→0.009) | **fails**: p = 0.121 / 0.385 / 0.152 |
| R4 HFB specific to correct uncovers | fails under every split | n/a (no pre-event baseline) | untestable, n=5–19 sessions |

**Not previously reported and more robust than R2/R3:** `correct − incorrect`
within the exploring phase is significant in all five readings (matched raw
+0.0229 p=0.038; matched base A +0.0491 p=0.0011 Holm 0.005; own base B +0.0528
p=0.0003), because it is a main effect on 61 sessions rather than a difference
of differences on 49–52. The interaction's `once known` half contributes almost
nothing and `error, once known` exists in only 52 sessions.

**Caveat on the matched design.** `correct, first uncovers` has median post-press
stillness 1.05 s vs ~0.50 s elsewhere, and only 57% of its events find a matched
control (89–95% for every other cell). The matched R1 is therefore the 57% with
the shortest pauses.

**HFB (`swr/ripple_hfb_conditions_2026-09-27/hfb_phase_robustness.json`).**
No event-type contrast is significant in any phase under any split. Stillness
events with ≥15 clean ripples exist in only 14/53 sessions (explore), 7 (plan),
7 (known), 17 (learning); correct uncovers − stillness in explore is
−0.0001, t(13)=−0.02, p=0.98, and stillness alone is the numerically largest
explore effect (+0.0079, n=14). Collapsing explore+plan REMOVES the correct-
uncover effect (+0.0024, p=0.39) because `reward_plan` is ~0 (n=23) — and this
disagrees with `hfb_locking_contrasts_2026-09-23` (+0.0050, p=0.048), which
pools the saved traces instead of at extraction. Do not report a pooled
learning-phase HFB reward effect.

## 2026-09-27b — Phase definition FIXED; two branches were splitting the task differently

**Decision (user, 2026-09-27).** Two phases, per grid, with `fs` = the first
fully correct repeat:
  learning   `rep <  fs`   every traversal before the first correct one
  execution  `rep >= fs`   the first correct traversal and all that follow
Baseline window **−1.5 to −1.0 s**. Stillness-matched values reported wherever
they survive, and explicitly reported where they do not.

**Bug found.** `rip.stage_of_repeat` (ripple-rate branch) puts the first fully
correct repeat in `while learning` (`rep <= fs`); `swr_windows.add_phase3`
(HFB branch) puts it in `execute` (`rep < fs`). 1432 repeats — **39% of the
learning side**, 8.1% of all repeats — sat on opposite sides in the two
branches. The agreed definition is the `add_phase3` one. `session_events` in
`swr_ripple_hfb_conditions.py` now takes `--boundary` (inclusive/exclusive) and
records it in `result.json`. `add_phase3` itself is UNCHANGED, so
`swr_probes.py`, `swr_units.py`, the `phase3` window design and their consumers
(`swr_ripple_tests.py`, `swr_explore.py`, `swr_ripple_triggered_units.py`) are
untouched and still carry their original boundary.

**New scripts.** `swr_robustness_grid.py` rewritten as a full sweep: 2
boundaries × 3 groupings × 5 references, 51 unique tests computed once and
shared, with a markdown report. `swr_stillness_by_phase.py` (behavioural).
`swr_figure_phase_panels.py` (Fig 4f/4g at the agreed split, both references,
with a profile cache for fast redraws).

**Ripple rate at the agreed split** (`swr/figure4_panels_2026-09-27/`, n=61):

| test | own baseline −1.5…−1.0 | stillness-matched |
|---|---|---|
| correct, learning | +0.0394, t(60)=+3.71, p=0.0002; cl +0.25…+0.65 s p=0.010 | +0.0311, t(60)=+2.03, p=0.043; cl +0.35…+0.65 s p=0.006 |
| correct, execution | +0.0036, t(60)=+0.58, p=0.559 | +0.0067, t(60)=+0.70, p=0.487 |
| valence \| learning | +0.0558, t(60)=+4.10, p=0.0002, Holm 0.001; cl +0.15…+0.75 s p=0.004 | +0.0410, t(60)=+2.77, p=0.0067, Holm 0.033; cl +0.25…+0.65 s p=0.006 |
| phase \| correct | +0.0358, t(60)=+2.75, p=0.0057; **no cluster** | +0.0243, t(60)=+1.49, p=0.146 |
| INTERACTION | +0.0771, t(51)=+2.74, p=0.0083; cl +0.15…+0.75 s p=0.007 | +0.0495, t(48)=+1.96, **p=0.053**, no cluster |

**Survives stillness matching:** the learning-phase increase, and the
correct-vs-incorrect difference within learning. **Does not:** the phase
contrast and the interaction. `phase | correct` has no surviving cluster even
under the own baseline, so it is a window-test result only.

**Why the interaction fails the control** (`swr/stillness_by_phase_2026-09-27/`):
post-uncover stillness has a valence × phase crossover of its own, ~20x larger
than the neural one. Correct − incorrect stillness is **+0.953 ± 0.098 s**
(t(60)=+9.71, p=6.3e-14) in learning and **−0.474 ± 0.179 s** (t(51)=−2.65,
p=0.011) in execution. The draft sentence "subjects are not stationary for
longer during correct vs incorrect uncovers in the learning phase" is false.
Correct uncovers also speed up monotonically across the task (1.36 → 0.45 s)
while incorrect ones slow down (0.48 → 0.66 s). The boundary choice alone moves
the learning cell's median stillness from 0.775 s (inclusive) to 1.237 s
(exclusive).

**HFB at the agreed split** (`swr/ripple_hfb_conditions_2026-09-27/`): the
pooled learning cell is NOT significant (+0.0024 ± 0.0027, t(51)=+0.87,
p=0.39). The effect lives entirely in the first traversal (+0.0076, t(50)=+2.73,
p=0.0088, cl −130…+10 ms p=0.038); the rest of learning is +0.0002 (n=23,
p=0.97). Under the inclusive boundary it would be +0.0050, p=0.032, so the
claim is boundary-sensitive and must be made about the first traversal.
No condition contrast is significant in either phase; navigation is
numerically larger than correct uncovers in both.

**Failed/abandoned.** `ripple_robustness_grid_2026-09-27` and
`ripple_hfb_conditions_2026-09-27` (first version) used the old three-scheme
layout and a `first vs later` grouping that is not the agreed split; superseded.

## 2026-09-29 — Cleanup pass: status banners, inventory, canonical methods/results

No new analysis. Housekeeping so the branch can be handed over:

- **STATUS banners** in the module docstring of `swr_figure_phase_panels.py`
  (★ KEY), `swr_stillness_by_phase.py` (◆ SUPPORTING), `swr_robustness_grid.py`
  and `swr_hfb_phase_robustness.py` (⚠ EXPLORATORY), and
  `swr_stage_interaction.py` (✗ SUPERSEDED — wrong side of the phase boundary).
- **`ANALYSIS_INVENTORY.md`** (data/final_results/ripple_analysis/): every SWR
  script classified KEY / SUPPORTING / EXPLORATORY / SUPERSEDED with what it
  feeds, plus which output directories are superseded.
- **`HOW_TO_RUN.md` §6b**: the reported ripple-rate result, the fixed phase
  definition, the two references and what each zero means, and the redraw path.
- **`manuscript/methods_07_ripple_rate_by_phase.md`**: canonical methods.
  Adds what was missing — the 0–0.5 s window test itself (10,000 sign-flips),
  the session as unit and the 10-event floor that explains the varying df, the
  matching caliper max(0.15 s, 10%) and the resulting 58%/89% match rates, how
  contrasts are formed, Holm, the cluster p formula, and that correction is
  across window positions within a condition only. Also fixes: zero means the
  pre-event window for the own-baseline reading but the matched control for the
  stillness-matched reading — these are different tests.
- **`manuscript/results_ripple_rate_FINAL.md`**: canonical results paragraph
  (narrative form) plus the full value table and a "things not to write" list.
- `SUPERSEDED.md` dropped into seven stale output directories.
- **`mc/analyse/swr_phases.py`** (new): the phase definitions, `PHASE_Q`,
  `GROUPINGS`, `add_exclusive_stage`, `base`, `holm`. Extracted because the
  REPORTED script was importing the EXPLORATORY one — `swr_figure_phase_panels`
  imported `swr_robustness_grid`. Both now import from `mc/analyse/`.
- **Moved to `scripts/exploratory/`**: `swr_robustness_grid.py`,
  `swr_hfb_phase_robustness.py`, matching the convention already set for the
  HFB branch. Registered in `scripts/exploratory/README.md`.
- **`SCRIPTS_GUIDE_swr_ripple_rate.md`** (new, repo root), matching
  `SCRIPTS_GUIDE_swr_content.md`: one rule, `swr_figure_phase_panels.py` is the
  reported result. `ANALYSIS_INVENTORY.md` was trimmed to output-directory
  provenance only so the script classification lives in one place.
- Verified after the move: both scripts import and run, and the sweep's report
  reproduces the same numbers (interaction +0.0771, p=0.0083 own-baseline;
  +0.0495, p=0.053 stillness-matched).

**Still open.** (1) The HFB key result is spread over four scripts
(`swr_ripple_locked_hfb`, `swr_hfb_locking_contrasts`,
`swr_ripple_hfb_conditions`, `swr_hfb_panels_2cm`) and was not consolidated —
they are recent and restructuring them was not agreed. (2) The outcome column
for the eleven exploratory HFB scripts in the inventory is unfilled; their
results have not been read. (3) `add_phase3` is unchanged, so `swr_probes.py`,
`swr_units.py` and the `phase3` window design still carry the original boundary.

## 2026-09-29b — Two entry points, and a wrong default caught

**`scripts/swr_main.py`** — the reported result, one entry point. Lists the 11
steps with what each produces and which figure it is; runs one by name or `all`
in order. Two steps consume an upstream directory (`hfb_contrasts`,
`hfb_panels`); the dispatcher resolves those to the newest matching directory
that does **not** carry a `SUPERSEDED.md`, so a stale run cannot be wired in
silently.

**`scripts/exploratory/swr_archive.py`** — the archived work, one entry point,
mirroring `swr_content_explore.py`. Eight analyses, each with the question it
answered and what came of it.

Neither merges code: the steps keep their own `SEED`, `MIN_RIPPLES`, `N_PERM`
and smoothing constants, which differ deliberately, and one namespace would let
them redefine each other.

**⚠ BUG FOUND AND FIXED — wrong phase boundary by default.**
`swr_ripple_hfb_conditions.py` defaulted to `--boundary=inclusive`, which puts
the first fully correct repeat in the LEARNING phase. The agreed definition
(2026-09-27) puts it in EXECUTION, i.e. `exclusive`. The default was set on
2026-09-27, before the definition was agreed, and was then wrongly described in
`HOW_TO_RUN.md` and `SCRIPTS_GUIDE_swr_ripple_rate.md` as "the agreed one".

* default changed to `exclusive` in `session_events` and `run`
* the docstring now says which is agreed and which is sensitivity
* `HOW_TO_RUN.md` and the guide corrected, both carrying the warning that **any
  conditions run made between 2026-09-27 and 2026-09-29 without an explicit
  `--boundary` is on the wrong split**

Effect of the default, session 4: `reward_plan` 12 events (exclusive, correct)
vs 102 (inclusive), `reward_execute` 1017 vs 927.

The existing runs are unaffected — `ripple_hfb_conditions_2026-09-27/` predates
the flag and used `phase3` (= exclusive), and
`ripple_hfb_conditions_exclusive_2026-09-28/` passed it explicitly. Only
`ripple_hfb_conditions_incl_2026-09-27/` is on the inclusive split, and it is
labelled as the sensitivity run.

## 2026-10-02 — Position in sequence on phase- and ramp-residualised data

Replaces the earlier "two formats" (sustained vs phasic) framing. Goal progress
and the per-loop monotone ramp are both treated as signals of no interest and
removed from the traces before either analysis, so the population RSA and the
single-cell GLM are protected identically.

Runs (both `phase_residualise=cosine`, `drift_residualise=linear`):
- RSA  `DSR_RSA_simple_ROI/2026-10-02_13-57-52`, combo `S_P_ctrl`
  (phase is pruned from the combo because it is already out of the data,
  leaving state + location + l2_norm + dsr_fmri), 1000 permutations.
- GLM  `encoding_state_sustained_cv/2026-10-02_13-54-14`, 10000 permutations,
  924 cells.
- No-residualisation reference for the preferred-position comparison:
  `encoding_state_sustained_cv/2026-09-24_18-35-23`.

FDR is Benjamini-Hochberg across the 5 (RSA) or 6 (GLM) ROIs within each
sub-model. Figures + results JSON in
`group/position_index_2026-10-02/` (`scripts/plot_position_code_figures.py`).

**RSA, position-in-sequence (`split_halves_z`):** mOFC .089 q=.0025 (n=74),
HC_mid .051 q=.0033 (145), HC_ant .046 q=.0025 (171), mPFC .028 q=.033 (65),
PCC .010 q=.21 (51). EC is not testable at the population level: it has cells
in 1 of 28 RSA-eligible subjects, below `MIN_RSA_SUBJECTS=3`.

**GLM, held-out r for the position block** (Hodges-Lehmann + distribution-free
CI, the estimator the Wilcoxon signed-rank test actually tests -- a mean CI
crossed zero for HC_mid while the test was significant, which is what motivated
the switch): EC .0215 [.0071,.0352] q=.0029 (35); mOFC .0134 [.0059,.0208]
q=.0014 (142); mPFC .0100 [.0040,.0156] q=.0019 (158); HC_ant .0091
[.0035,.0141] q=.0019 (295); HC_mid .0091 [.0025,.0154] q=.0050 (233);
PCC .0026 [-.0100,.0149] q=.32 (61).

Every ROI the RSA finds is positive in the cells, and PCC is the only null in
both. EC and mOFC are the strongest (EC leads in cells, mOFC in the RSA).

**Cost of the ramp control (the honest caveat).** Preference for position 1 or 4
shares 60% of its variance with a full-loop ramp; positions 2/3 share 7%. Among
position-coding cells the preferred position is uniform without residualisation
(27/21/25/28%, n=206, chi2 p=.53) but position 4 is depleted after it
(26/30/30/13%, n=204, chi2 p=.00065) -- worst in mOFC (3% prefer position 4) and
HC_mid (5%). The count of position cells is unchanged (206 -> 204), so this is a
redistribution, not a loss.

**Ramp-robust subset (cells preferring position 2 or 3 only):** mPFC .0110
q=.0040 (99) and HC_mid .0131 q=.0040 (159) survive; EC .0197 q=.054 (20) and
mOFC .0084 q=.054 (91) sit at threshold with much reduced n; **HC_anterior does
not survive** (.0024 q=.29, 196 cells) -- its effect lives almost entirely in the
first/last positions (A/D cells: .0205 q=.0001), exactly where a residual drift
could produce it. HC_ant is therefore the least secure of the five results.

Failed/abandoned: windowing the trend fit to bins 0:270 instead of 0:360 does not
rescue identifiability -- it just moves the confound onto position 3 (67%
shared variance) instead of position 4.

### 2026-10-02 (addendum) — the ROIs do not differ from each other

`scripts/position_roi_differences.py` -> `group/position_index_2026-10-02/`
(`fig4_roi_differences`, `roi_differences_summary.json`).

Each ROI being individually significant is not the same as the ROIs differing.
Tested directly on the per-cell position effect (run `2026-10-02_13-54-14`):

- Kruskal-Wallis omnibus **H = 5.16, p_perm = .40** (10,000 ROI-label shuffles;
  null median H = 4.35, 95th pct = 10.91). Nowhere near significant.
- No pairwise contrast survives FDR across the 15 pairs; smallest q = .459
  (EC vs PCC, uncorrected p = .059).
- Planned one-sided contrasts vs the pooled rest: EC +.0119 [-.0025, +.0269]
  p = .053, q = .107; mOFC +.0046 [-.0033, +.0126] p = .124, q = .124.

So the ranking EC > mOFC > mPFC > HC is a description of point estimates only.
Between-cell spread within an ROI dwarfs any between-ROI difference. Claims of
regional concentration must rest on the fMRI, not the cells.

Note the two-sample shift CI must be the **Moses** interval (order statistics of
the pairwise differences indexed by the Mann-Whitney distribution). Taking the
2.5/97.5 percentiles of the raw pairwise differences is not a CI -- it is the
spread of the differences, and gave intervals ~10x too wide (+-0.1 vs +-0.015).

Also: mean+CI and the Wilcoxon p disagree for HC_mid (mean .0064 CI
[-.0009,+.0137] vs p = .0042). The HL estimate and its interval agree with the
test everywhere, which is why they are what is reported.

### 2026-10-02 (addendum 2) — RSA: mOFC beats mPFC and PCC, not hippocampus

`scripts/rsa_roi_bootstrap.py` -> `group/position_index_2026-10-02/`
(`fig5_rsa_roi_bootstrap`, `rsa_roi_bootstrap_summary.json`).

The RSA gives one beta per ROI, so there is no group to test. The regression's
own SE cannot be used for between-ROI contrasts: the 4560 RDM entries are not
independent (each of 96 conditions enters ~95 pairs), which is the same reason
the pipeline reports p_perm and not p_param. Uncertainty was therefore taken
over the unit that actually varies -- which cells were recorded. Cells are
resampled with replacement WITHIN each ROI (never across), the pseudo-population
RDM rebuilt and the combo refitted, 1000 draws, seed 42.

Fitting code is AST-extracted verbatim from `RSA_DSR_ROIs_simple.py`, and the
full-sample refit is asserted against the published beta before resampling
(reproduces to <=8.5e-17 for all five ROIs; the script exits otherwise).

**Significant after FDR:** mOFC > PCC (+.079 [.055,.140], q=.020); mOFC > mPFC
(+.061 [.032,.117], q=.030); HC_ant > PCC (+.036, q=.035); HC_mid > PCC (+.040,
q=.035). **Not significant:** mOFC vs HC_ant (+.044, q=.084), mOFC vs HC_mid
(+.039, q=.175), and all remaining pairs. So mOFC and hippocampus form an upper
tier and PCC the lower one; mOFC does NOT stand alone.

Note this contrasts with the per-cell GLM, where no ROI differs from any other
(H = 5.16, p_perm = .40). Not a contradiction: the RSA pools cells into one
population estimate per ROI, so it is not limited by between-cell variance,
which is what swamps the cell-level comparison.

**Gotcha: raw percentile bootstrap CIs are biased low here.** A resample holds
only ~63% distinct cells, and an RDM built with duplicates is less informative,
so every bootstrap beta is attenuated -- and proportionally to beta itself
(mOFC -.0228, mPFC -.0074, PCC -.0023). Reported intervals are therefore basic
(reverse-percentile), and the pairwise p-values use null-centred draws, which is
insensitive to that bias. Do not read the percentile interval as a CI on beta.

**Also verified (all five ROIs):** the combo permutation nulls are 0-centred.
Null beta means +.00019/+.00005/+.00034/-.00066/-.00050 against sd ~.0139;
null t means within +-.044 with sd ~0.93. Refitting the cached permuted data
RDMs reproduces every published p_perm exactly.

**Why the mOFC t is so much larger:** nothing but beta. The residual SE is
effectively constant across ROIs (.01482-.01488, a 0.4% spread), so
t = beta/.0148. t and beta carry identical information and the same ranking.

`RSA_DSR_ROIs_simple.py` gained two env-gated hooks (defaults unchanged):
`RSA_NO_PERM=1` skips the permutation loop, `RSA_DUMP_MAT=<dir>` writes each
ROI's stacked neuron x condition matrix. Safe because z-scoring there is per
neuron, so row subsets score identically to the full set.
