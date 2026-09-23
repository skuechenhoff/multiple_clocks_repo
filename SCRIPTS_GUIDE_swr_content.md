# Ripple content analysis — where everything is

**2026-09-18.** What each file does, what it produces, and what was archived.
Companion to `data/final_results/ripple_analysis/` (results prose) and
`CHANGELOG.md` (the running record, including failed attempts).

⚠ Results, figures and logs never live in this repo. Everything below writes to
`data/ephys_humans/derivatives/group/swr/`.

---

## The short version

| I want to… | run this |
|---|---|
| reproduce the reported result and its figure | `python scripts/swr_content_main.py` |
| re-run one step of it | `python scripts/swr_content_main.py contrast` |
| look at anything exploratory | `python scripts/swr_content_explore.py <analysis>` |
| regenerate every figure | `python scripts/swr_content_figures.py` |

---

## Library — `mc/analyse/`

Shared machinery and the exploratory analyses. Nothing in the first two files
runs on its own; everything else imports them.

    mc/analyse/swr_location.py      the location timeline
    mc/analyse/swr_content.py       estimator + templates
    mc/analyse/swr_explore/         the 12 exploratory analyses, one per module

### `mc/analyse/swr_location.py`
The real-time location timeline, straight from `abcd_passed.mat` → `trial_vars`.
One row per move: where the participant is, when, and what they uncovered.

⚠ The two obvious alternatives do not work, and both are dead ends that were
measured, not assumed: `all_location_snippets.csv` is time-warped, so only 7.2%
of ripples fall inside a snippet; integrating button presses reconstructs the
true square 62.9% of the time. Use `grid_onset_timestamp` /
`state_change_times` / `end_trial_timestamp` (trigger clock) and never
`button_pressed_timestamp` (Matlab clock, and the mat file itself renames it
`DONOTUSE_…`).

`build()` caches per-session step tables; `load()` reads them.

### `mc/analyse/swr_content.py`
The estimator, and the template machinery that used to sit in
`scripts/swr_place_templates.py` and be imported as if a script were a library.

- `occupancy` — location intervals per session. ⚠ `cv_group` is `grid_id`, NOT
  `grid_num`: a configuration recurs in ~3 blocks, so holding out one block
  leaves the same rewards and route in training.
- `place_map`, `zscore_map` — a unit's rate per square, z-scored to sum to zero
- `build_templates`, `reliability`, `weighted_loo` — templates, split-half
  reliability, and reliability-weighted templates
- `window_counts`, `score_windows`, `evidence`, `target_minus_others`,
  `target_z` — the estimator
- `zscore_cells` — per-unit z-scoring of counts, so loud units do not dominate
- `matched_flanks` — flanks matched on square, occupancy interval and width
- `tiled_windows` — deterministic non-ripple windows
- `session_data` — the per-session bundle every analysis starts from

---

## Entry points — `scripts/`

### `swr_content_main.py` — THE REPORTED RESULT
Everything behind the claim *location information is present in and around
hippocampal ripples, and is not ripple-specific*. Five steps, in dependency
order; `run("all")` does all of them.

| step | what it does | output folder |
|---|---|---|
| `timeline` | build/validate the location cache | `location_timeline/` |
| `control` | positive control on non-ripple windows, width sweep | `content_main_control_<date>/` |
| `contrast` | ripple vs matched flank (C0, C1) | `content_main_contrast_<date>/` |
| `timecourse` | 41 offsets, cluster-corrected | `content_main_timecourse_<date>/` |
| `panel` | the reported figure, 3 cm and full size | `ripple_content_figures_<date>/` |

`panel` runs LAST on purpose: figures drifted behind corrected numbers more than
once while this was being built.

### `swr_content_explore.py` — EVERYTHING ELSE
None of this is in the reported result. One entry point, one subcommand per
analysis. Run it with no arguments to print the table below with its findings.

The analysis code lives in **`mc/analyse/swr_explore/`**, one module each, and
is deliberately NOT merged into a single file: these were written separately and
their module-level constants differ (`SEED`, `N_PERM`, `ROI_SETS`,
`MIN_RIPPLES`), so a flat namespace would let one analysis silently redefine
another's. The dispatcher gives you the single entry point; the package keeps
them isolated.

| subcommand | question | verdict |
|---|---|---|
| `descriptives` | spike/ripple budget, peri-ripple histograms | — |
| `spike_rate` | is firing higher inside ripples? | yes, but only ~4% |
| `profile` | the full 9-square distance profile | identical in ripple and flank |
| `i11` | rewarded vs non-rewarded squares in exploration | null (+0.241, p = 0.29) |
| `roles` | the 12-term role regression (I12, I13, goals) | route effect only |
| `collinearity` | VIFs and leave-one-regressor-out | design is sound |
| `roles_timecourse` | role evidence over the peri-ripple second | descriptive, flat |
| `pseudopop` | cells pooled across sessions sharing a task | signal is distributed |
| `pseudo_timecourse` | pseudo-population peri-ripple time course | superseded by main |
| `decoders` | can location be decoded from iEEG bands or spikes? | no, from anything |
| `ieeg_decoder` | the decoder gate, with its positive control | gate closed |
| `location_timecourse` | the old arrival-locked time course | ⚠ labels expire, see below |

    python scripts/swr_content_explore.py            # the table, with findings
    python scripts/swr_content_explore.py i11

### `swr_content_figures.py`
Every figure, numbered. `python scripts/swr_content_figures.py` regenerates all
of them. Figure 15 is the reported panel.

---

## Results and prose — `data/final_results/ripple_analysis/`

| file | what it is |
|---|---|
| `SUMMARY_ripple_content.md` | the current position, rewritten rather than appended |
| `ripple_content.md` | the running log, in order, including what failed |
| `MANUSCRIPT_ripple_location.md` | Methods + Results for the reported panel |
| `POTENTIAL_IDEAS.md` | what is queued, and what has been closed |

---

## Things that will bite you

- **Cross-validate on the configuration** (`grid_id`), never the block
  (`grid_num`). ⚠ But `grid_id` is a WITHIN-session index — the same `grid_id`
  is a different task in another session. To pool across sessions, match on the
  reward tuple.
- **z-score counts per unit** before scoring, or a few loud units carry
  everything.
- **Flanks must be matched** on square, occupancy interval and width. A fixed
  ±500 ms flank is on the same square only 53.7% of the time.
- **Never average raw evidence across sessions** — |E| correlates r = 0.49 with
  unit count. z each session against its own null first.
- **Anything with a random serving order** (matching, window sampling) must be
  repeated and averaged, with an independent RNG per session and per ROI.
- **Report the continuous statistic, not decoding accuracy.** Neighbouring
  squares have correlated templates, so 9-way argmax sits at chance even where
  the signal is clear — for every signal in the dataset, spikes included.
- **Any selection or weighting of cells** must be estimated with the scored
  configuration held out, and must not drop sessions. Letting a reliability
  peek turned a null into "selection doubles the signal".
- **Peri-event labels expire.** Median dwell is 0.367 s, so beyond ~0.44 s from
  arrival the participant is more often on the NEXT square. Windows wider than
  that measure where they walked.
- **Validate a decoder before reporting its null.** `positive_control` in the
  decoder script is the template; a null from an unvalidated instrument says
  nothing.

---

## Archived — `scripts/archive/`

Five scripts, absorbed into `swr_content_main.py` and `mc/analyse/swr_content.py`:

| archived | where it went |
|---|---|
| `swr_place_templates.py` | `mc/analyse/swr_content.py` — it was a script being imported as a library, which is what made this tangle |
| `swr_ripple_content.py` | `swr_content_main.py` → `contrast` |
| `swr_stage3_timecourse.py` | `swr_content_main.py` → `timecourse` |
| `swr_place_width_sweep.py` | `swr_content_main.py` → `control` |
| `swr_build_location_timeline.py` | `swr_content_main.py` → `timeline` |

Kept rather than deleted because they are the exact code that produced numbers
quoted in `CHANGELOG.md`. **Not maintained**; their imports point at the old
`scripts.swr_place_templates` path and will not run as-is. Everything is also in
git history at `e548f71`.

⚠ The merged `contrast` step was verified against the archived
`swr_ripple_content.py`: identical session counts per ROI, per-session
correlation r = 0.993, group means agreeing to the third decimal. The residual
difference was permutation-null noise, and the merged version now uses a
deterministic RNG stream per session and ROI so loop order cannot move a number.

## Not part of this analysis line
The other ~35 `swr_*` scripts in `scripts/` belong to the ripple-detection,
HFB and RSA lines and were not touched: `swr_detect_session`, `swr_extract_*`,
`swr_build_*`, `swr_ripple_locked_hfb`, `swr_ripple_rsa_*`, `swr_stillness_*`,
`swr_matched_control_presses`, and so on.
