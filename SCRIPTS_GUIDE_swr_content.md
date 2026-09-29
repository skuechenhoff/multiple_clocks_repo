# Ripple content — where everything lives

Updated 2026-09-29. One rule: **`swr_content_main.py` is the reported result.
Everything else is exploratory or infrastructure.**

---

## The reported result

```bash
conda activate env_multiple_clocks
python scripts/swr_content_main.py                 # all steps, in order
python scripts/swr_content_main.py --step=phase    # just the reported analysis
```

| step | what it does | reported? |
|---|---|---|
| `timeline` | real-time location from `trial_vars`, cached once | infrastructure |
| `control` | positive control: the estimator on ordinary navigation windows | supporting |
| `contrast` | ripple vs matched flank, all ripples pooled | supporting |
| **`phase`** | **location in the ripple window, encode vs execute, counts matched** | **YES** |
| `panel` | regenerates every figure, last, so none can drift | — |
| `timecourse` | 41-offset time course, cluster corrected | retained, not reported |

The reported analysis is `mc/analyse/swr_phase_contrast.py`. It is
self-contained: spikes are counted in the ripple windows and their matched
flanks and nowhere else, which is exactly what the Methods describes.

**Claim:** during encoding, hippocampal ripples carry the square the
participant is standing on (pooled z = +0.57, P = 2.0e-4); once the sequence
is known this is not detectable (+0.21, P = 0.125); the difference survives
matching ripple counts (+0.46, P = 0.008). It is **not** ripple-specific
against a duration-matched flank from the same visit (+0.23, P = 0.115).

Figures: `fig21_phase_panel_2.5cm.pdf` (reported), `_4cm_with_pooled`,
`fig21_panel_phase_interaction`, `fig20_phase_effect_bars`.

---

## Exploratory

```bash
python scripts/swr_content_explore.py          # index + one-line verdicts
python scripts/swr_content_explore.py <name>
```

One module per analysis in `mc/analyse/swr_explore/`, deliberately not merged:
each has its own `SEED`, `N_PERM`, `ROI_SETS`, `MIN_RIPPLES`, and concatenating
them would let one analysis silently redefine another's constants.

`scripts/swr_content_figures.py` holds every figure (1–21) and is the only
place that reads result directories. `figure20`/`figure21*` read the reported
analysis via `_phase_contrast()`, which prefers `phase_contrast_*` and falls
back to the older module so nothing breaks.

**`phase_timecourse` is descriptive only.** Its fixed-window numbers differ
slightly from the reported ones because there the per-unit count normalisation
pools across all 41 offsets. It is kept because the cluster result (mHP
encoding, −100 to +75 ms, P = 0.005) exists nowhere else and may become a
supplementary panel.

---

## Library

| module | holds |
|---|---|
| `mc/analyse/swr_content.py` | the estimator: `place_map`, `zscore_map`, `build_templates`, `window_counts`, `zscore_cells`, `score_windows`, `occupancy`, `matched_flanks`, **`phase_labels`** |
| `mc/analyse/swr_phase_contrast.py` | the reported analysis |
| `mc/analyse/swr_explore/*` | one exploratory analysis each |

`phase_labels` defines encode/execute and lives in the library, not in an
exploratory module, because the reported contrast depends on it.

`scripts/swr_worked_example.py` runs the whole estimator on a toy session and
prints every intermediate (template, z-scored counts, dot product, the null,
count matching). Run it before editing anything here.

---

## Things that will bite you

1. **`cv_group` is `grid_id`, not `grid_num`.** A configuration recurs in ~3
   blocks; holding out one block leaves the same rewards in training.
2. **z carries a √n term.** SD(null) ∝ 1/√n, so z is not comparable between
   ripple sets of different size. Any phase or condition comparison must match
   counts. This is why the contrast uses subsampling.
3. **`first_traversal` is a SUBSET of `encode`** (99.95% inside it, 71% of
   it), not a third phase. Never pool them.
4. **The far-field / "surround" control is confounded.** Its windows are
   scored for the square occupied at the ripple, but at |t| ≥ 350 ms the
   subject is still on that square only 54.6% of the time. Use the matched
   flank, which holds the square fixed. See CHANGELOG 2026-09-29 (d).
5. **Argmax decoding fails structurally** — neighbouring-square templates are
   correlated. Every sequence/replay analysis is gated on a positive control
   that has never passed.
6. **Non-finite evidence is about the template, not spikes.** A location never
   visited outside the held-out configuration has a NaN template entry, which
   poisons the whole row.
7. **Seed the RNG per session and per curve.** A shared generator makes each
   result depend on what was computed before it — observed as a cluster P of
   0.048 then 0.051 on identical data.

---

## Write-ups

`data/final_results/ripple_analysis/`:

| file | what |
|---|---|
| `METHODS.md` | the Methods paragraph as it should appear |
| `RESULTS.md` | every reported number, what can and cannot be claimed, draft paragraph |
| `HOW_TO_RUN.md` | commands and outputs |
| `superseded/` | earlier drafts, kept as a record. ⚠ several report the confounded far-field control as positive — do not reuse |

---

## Archived scripts

`scripts/archive/` — superseded, kept for provenance:
`swr_build_location_timeline.py`, `swr_place_templates.py`,
`swr_place_width_sweep.py`, `swr_ripple_content.py`,
`swr_stage3_timecourse.py`. All were folded into `swr_content_main.py` or the
library; the main runner was verified against them (identical session counts,
per-session r = 0.993).
