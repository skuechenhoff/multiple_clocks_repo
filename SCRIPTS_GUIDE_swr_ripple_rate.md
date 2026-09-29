# Ripple rate by task phase — where everything lives

Updated 2026-09-29. One rule: **`swr_figure_phase_panels.py` is the reported
result. Everything else is exploratory, superseded, or infrastructure.**

---

## The phase definition, fixed 2026-09-27

With `fs` = a grid's first fully correct (error-free) repeat:

    encoding   rep <  fs     every traversal before the first correct loop
    execution  rep >= fs     the first correct loop and everything after

This is `swr_windows.add_phase3` collapsed to two. It is **not**
`ripples.stage_of_repeat`, which puts the first correct repeat on the encoding
side; the two disagree about **1432 repeats, 39% of encoding**. Both labellings
live in `mc/analyse/swr_phases.py` so the difference stays testable.

⚠ `add_phase3` itself is unchanged, so `swr_probes.py`, `swr_units.py` and the
`phase3` window design still carry the original boundary. Only
`swr_ripple_hfb_conditions.py` takes `--boundary`: `exclusive` is the agreed
definition and the default, `inclusive` is `stage_of_repeat` and is sensitivity
only. **The default was `inclusive` between 2026-09-27 and 2026-09-29** — a
conditions run made in that window without an explicit `--boundary` is on the
wrong split.

## The reported result

```bash
conda activate env_multiple_clocks
python scripts/swr_figure_phase_panels.py run                    # ~8 min
python scripts/swr_figure_phase_panels.py figures --out_dir=<run>  # redraw, ~5 s
```

Writes `figure4_panels_<date>/`: `panel_{f,g}_{ownbase,matched}.pdf`,
`figure4_panels.json` (every statistic), `profiles.npz` (what makes the redraw
fast).

| output | reported as |
|---|---|
| `panel_f_ownbase.pdf` | **Fig 4f** |
| `panel_g_ownbase.pdf` | **Fig 4g** |
| `panel_{f,g}_matched.pdf` | supplement — the stillness control |

**Two references, and they do not mean the same thing:**

| reading | zero means |
|---|---|
| `own baseline -1.5..-1.0` | no change from that event's own pre-event window |
| `stillness-matched` | no difference from an arrow press followed by the same stillness |

At the agreed split, the encoding increase and the correct-vs-incorrect
difference within encoding survive the stillness control; the phase contrast
(p = 0.146) and the valence × phase interaction (p = 0.053) do not. Report that
explicitly — see `manuscript/results_ripple_rate_FINAL.md`.

## Supporting

| script | supplies |
|---|---|
| `swr_stillness_by_phase.py` | the behavioural stillness numbers that justify the matched control (+0.95 s correct vs incorrect in encoding, reversing to −0.47 s in execution) |
| `mc/analyse/swr_phases.py` | the phase definitions and shared helpers; imported by both the reported and the exploratory scripts |
| `swr_final_ripple_analysis.py` | the machinery — matching, rates, window and cluster tests. Its own `between` block is **superseded** |

## Exploratory — in `scripts/exploratory/`

| script | question | outcome |
|---|---|---|
| `swr_robustness_grid.py` | every phase boundary × grouping × reference | fixed the phase definition; showed the interaction does not survive stillness matching |
| `swr_hfb_phase_robustness.py` | the same on the HFB branch | negative — no stillness control is available there (17/53 sessions) |

## ✗ Superseded — do not quote

| script | replaced by | why |
|---|---|---|
| `swr_stage_interaction.py` | `swr_figure_phase_panels.py` | its `explore_known` split puts the first correct repeat on the wrong side |
| `swr_valence_stage_interaction.py` | same | earlier 3-stage version |
| `swr_ripple_tests.py` `feedback_stage` | same | predates the agreed definition and the −1.5 s baseline |
| `swr_matched_control_presses.py`, `swr_feedback_vs_matched_stillness.py`, `swr_stillness_control.py` | `swr_final_ripple_analysis.py` + `swr_stillness_by_phase.py` | the matching they prototyped is now in the pipeline |

## Documentation

| what | where |
|---|---|
| how to run | `data/final_results/ripple_analysis/HOW_TO_RUN.md` §6b |
| methods | `.../manuscript/methods_07_ripple_rate_by_phase.md` |
| results | `.../manuscript/results_ripple_rate_FINAL.md` |
| which output dirs are current | `.../ANALYSIS_INVENTORY.md` |
