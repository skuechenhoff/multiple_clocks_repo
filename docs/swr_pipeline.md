# Hippocampal sharp-wave ripple pipeline

Detection and analysis of hippocampal sharp-wave ripples (SWRs) in human
intracranial recordings, following Chen et al. 2025 (*J Neurosci* 45:e1502252025)
with documented deviations.

This document is for **someone bringing their own data**. It describes what each
stage does, exactly what goes in and what comes out, and where you can join the
pipeline if you already have part of the work done. For the project-internal
cookbook — cluster commands, expected values, what to check after each job — see
`HOW_TO_RUN.md`; for why each choice was made, `methods.md`.

---

## 1. The short version

```
raw recording ──▶ bipolar derivations ──▶ artifact mask ──▶ ripple events ──▶ statistics
                  (stage 2)               (stage 3)          (stage 3)         (stage 6)
```

Three ideas run through the whole pipeline and explain most of its design:

1. **A ripple rate is events per *artifact-free* second.** Rejection is not
   uniform over a recording, so a rate per elapsed second reports artifact
   burden as if it were a ripple effect. Every rate, every window, and every
   permutation uses the artifact-free time axis.
2. **Nothing is dropped silently.** Each stage writes an inclusion report
   (`*_inclusion.md`) naming every unit it considered and why each one was kept
   or not.
3. **Anything removed must be visualisable, not just countable.** Artifact
   rejection discards ~40% of a typical recording; a percentage in a table is
   not evidence that what was discarded was artifact.

---

## 2. Where you can join

You do **not** have to start at stage 1. Each entry point below needs only the
files listed, in the formats given in §4.

| You already have | Enter at | You must provide |
|---|---|---|
| Raw recordings + electrode coordinates | stage 1 | raw files, a config entry, contact coordinates in MNI152 |
| Contact coordinates, no signal processing yet | stage 2 | `bipolar_pairs_NN.csv` |
| Preprocessed bipolar time series | **stage 3** | `continuous.npy` + `pairs.csv` + `meta.json` |
| Detected ripple times from your own detector | **stage 5/6** | `ripple_events.csv` + `clean_intervals.csv` + `channel_qc.csv` |
| A bundle from a collaborator | stage 6 | `bundle/` |

**Stage 3 is the most useful entry point** for most people: it takes a plain
`(n_derivations, n_samples)` float array and does artifact rejection plus
detection. Nothing above it is specific to this project's task or recording
sites.

If you only want the detector and not the file layout, skip to §6 — the library
functions take arrays and return DataFrames, with no filesystem contract at all.

---

## 3. The stages

Every script is `python scripts/<name>.py [verb] [--flags]` and uses
[`fire`](https://github.com/google/python-fire) for its command line. **Install
`fire`**: without it every script silently falls back to its defaults and
ignores `--session`, which is the single worst failure mode in the pipeline.

### Stage 0 — `swr_check_inputs.py`

Preflight. Reports missing Python packages, missing atlases, and unreadable
paths, separating blockers from warnings. Run it first; it costs seconds.

### Stage 1 — `swr_audit_sessions.py`

Reconciles the configuration file, the behavioural records and the raw files
actually present on disk, and writes the manifest that every later stage treats
as authoritative. Nothing here touches the signal.

- **out:** `group/swr/session_manifest.csv`, `audit_inclusion.md`

### Stage 2a — `swr_build_contacts.py`

Anatomy. Maps every recording channel to an MNI152 coordinate, assigns a region
from probabilistic atlases, selects the hippocampal contacts, and forms one
bipolar derivation per electrode.

Contacts are chosen from the **coordinate alone** — site-supplied region labels
are recorded as metadata but never consulted, because they are not comparable
across centres and cannot rank one contact against another. A contact enters if
`P(hippocampus) >= 25%` in the Harvard–Oxford subcortical probability atlas, and
each electrode contributes the single **most** hippocampal contact paired with
its immediate neighbour.

- **in:** `session_manifest.csv`, electrode tables, Harvard–Oxford atlases
- **out:** `s{NN}/LFP/macro_contacts_{NN}.csv`, `s{NN}/LFP/bipolar_pairs_{NN}.csv`,
  `group/swr/macro_contacts_all.csv`, `contact_qc.csv`, `contacts_inclusion.md`

> **Needs the atlases.** Without them no contact can be selected and the stage
> produces zero derivations *without erroring*. Set `NILEARN_DATA` to a directory
> holding the Harvard–Oxford set including `sub-prob-2mm`.

### Stage 2b — `swr_extract_continuous.py --session=N`

Signal preprocessing, one session at a time. The slow stage: it reads multi-GB
raw files.

Order is **resample → bipolar → notch**, and each step is deliberate:

- **Resample to 1000 Hz** by polyphase filtering (`resample_poly`, never
  `scipy.signal.resample`, which assumes periodicity and wraps the end of a
  recording into its beginning). 1000 Hz is the minimum that permits the >250 Hz
  artifact criterion and the 120–200 Hz spectral criterion used later.
- **Bipolar before notch**, because subtracting neighbouring contacts already
  removes most common-mode line noise, so the notch has far less to do.
- **Adaptive notch, per derivation.** For each of 60/120/180 Hz the
  peak-to-flank power ratio is measured *within each derivation*, and a
  zero-phase notch is applied only to derivations exceeding 2.0. A blanket notch
  would remove real 120 Hz signal — the upper edge of the ripple band — from
  clean recordings for nothing.

- **in:** `bipolar_pairs_{NN}.csv`, raw recordings
- **out:** `s{NN}/LFP-clean/{analysis}/continuous.npy`, `pairs.csv`, `meta.json`,
  `qc_psd.png`, `notch_psd.npz`

### Stage 3 — `swr_detect_session.py --session=N`

Artifact rejection and ripple detection. Reads `continuous.npy`, never the raw
files, so detector parameters can be re-tried cheaply. **Use `--analysis_name`
to keep variants apart**; never overwrite a completed set.

**Artifact rejection** flags a time point if any of five metrics exceeds four
interquartile ranges above its own median: signal amplitude; amplitude of the
first derivative (sharpness); RMS above 250 Hz; broadband 1–60 Hz power
(log-transformed *before* z-scoring — untransformed it flags many times more
samples than intended); and an automatic epileptiform-discharge detector (Janca
et al. 2015). Flags are padded by ±1 s, artifact-free islands under 1 s are
discarded, and derivations more than two-thirds contaminated are excluded.

**Detection** band-passes 80–120 Hz, takes a 20 ms moving RMS, and applies a
**dual threshold**: event extent (and therefore duration) at 1.5 SD, but the
peak must reach 3.0 SD. Mean and SD are estimated once per derivation over the
whole session from artifact-free samples. Survivors are then checked spectrally
against four criteria and flagged, never dropped — the accept/reject decision is
a column, so both scopes can be compared afterwards.

- **in:** `continuous.npy`, `pairs.csv`, `meta.json`
- **out:** `s{NN}/LFP-ripples/{analysis}/ripple_events.csv`, `channel_qc.csv`,
  `clean_intervals.csv`, `detector_diag.json`, `settings.json`,
  `detection_inclusion.md`

### Stage 4 — `swr_qc_report.py`

| verb | does |
|---|---|
| `metrics --session=N` | the numeric checkpoint only — fast |
| `report --session=N` | the same plus every per-session figure — ~10× slower |
| `group` | pools `qc_metrics.csv` into one triage table |
| `figure` | the group figures |

`metrics` grades six quantities against published reference ranges and returns
`PASS` / `CHECK` / `FAIL`. **`FAIL` is a stop**, `CHECK` means look at the
figures. Two of the six are traps worth knowing: a peak frequency sitting exactly
at 80 or 120 Hz means the filter band, not the signal, is choosing it; a duration
at 38 ms means the duration floor is doing the selecting. Both pass a range check
while telling you the detector is broken.

### Stage 5 — controls

- `swr_surrogate_control.py` fits each derivation's aperiodic (1/f) spectrum,
  simulates matched noise, runs the **identical** detector on it, and reports the
  false-positive fraction. Report excess-over-noise, not the raw rate: a large
  fraction of ripples in any human dataset fall inside the 1/f floor.
- `swr_rejection_bias.py` tests whether rejection varies with task phase. If it
  does, rejection is a confound and must enter the model.

### Stage 6 — `swr_ripple_tests.py --bundle=<dir>`

The tests. Each asks one question: **does ripple rate depart from that same
trial's baseline?** No window is chosen by hand — every position of a sliding
window is tested and corrected by a cluster permutation over positions.

Per test it writes `<test>.png`, `<test>_result.json` (the question, every
number, the conclusion) and `<test>_counts.csv` (subjects, sessions, derivations
and events per condition — never left implicit).

`swr_explore.py` is the scratch counterpart: PNG only, no correction, nothing
claimed. Things are tried there; what survives moves into `swr_ripple_tests.py`.

### Stage 7 — `swr_export.py`

| verb | does |
|---|---|
| `bundle` | a few MB that replace tens of GB of recordings |
| `numbers` | every value a manuscript would quote, as Markdown and JSON |

The bundle is the unit of collaboration: with it, every stage from 5 onward runs
on a laptop against identical inputs.

### Figures

`swr_group_figures.py all` builds the group figures **from a bundle**, so it
works wherever the bundle has been downloaded. `swr_contact_figure.py` (coverage)
and `swr_notch_figure.py` (line noise by site) are standalone.

---

## 4. File formats

Paths are relative to `<data_root>/derivatives/`. `{analysis}` is the
`--analysis_name` (default `swr_v1`), which keeps parameter variants apart.

### `continuous.npy` — the preprocessed signal

```
np.ndarray, shape (n_derivations, n_samples), dtype float32, microvolts
```

Row *i* corresponds to `pairs.csv` row *i* and to `meta["pair_ids"][i]`. Sample
*k* is at `k / fs` seconds on the session clock, so `sample = round(t * fs)`
exactly. Saved with `np.save`; loaded with `mmap_mode='r'` throughout, so it need
not fit in memory.

### `pairs.csv` — one row per derivation

Required by later stages:

| column | meaning |
|---|---|
| `session` | integer session id |
| `pair_id` | unique label, e.g. `LAHC2-LAHC3` |
| `pair_roi` / `pair_roi_atlas` | region of the anchor contact |
| `hemisphere` | `L` or `R` |
| `mni_x`, `mni_y`, `mni_z` | derivation midpoint, MNI152 mm |
| `subject_label` | patient id — **several sessions may share one** |

Everything else the anatomy stage writes (`anat_label_a/b`, `hpc_prob`,
`inter_contact_mm`, …) is provenance and is carried along but not required.

### `meta.json` — what preprocessing did

Stage 3 reads exactly one key, `fs` — derivation identity comes from the rows of
`pairs.csv`, not from here. Stage 4's figures additionally use `pair_ids`, so
write both:

```json
{"session": 38, "fs": 1000.0, "n_pairs": 2, "n_samples": 1660534,
 "pair_ids": ["RT2bHa02-RT2bHa03", "RT2cHbE02-RT2cHbE03"]}
```

Also written by stage 2b, and worth keeping for provenance: `recording_site`,
`duration_s`, `blocks`, `notch_applied_pairs`, `line_noise_ratio`,
`line_noise_residual`.

### `ripple_events.csv` — one row per candidate event

| column | meaning |
|---|---|
| `session`, `pair_id` | which derivation |
| `start_sample`, `peak_sample`, `stop_sample` | indices into `continuous.npy` |
| `t_start_s`, `t_peak_s`, `t_end_s` | the same in session seconds |
| `duration_s` | supra-threshold time of the RMS |
| `rms_peak`, `rms_peak_z` | peak RMS, raw and in SD of the session |
| `amp_peak_uv` | peak amplitude of the band-passed signal |
| `peak_freq_hz`, `peak_width_hz`, `peak_prominence` | spectral peak |
| `pass_duration`, `pass_peak`, `pass_amplitude` | time-domain gates |
| `spectral_passed_strict` / `_relaxed` | the two spectral scopes |
| `passed` | final accept — **the column downstream code filters on** |

Rejected candidates are kept as rows with `passed = False`, which is what makes
the rejection auditable.

### `clean_intervals.csv` — the artifact-free time axis

```
pair_id, start_s, stop_s
```

One row per contiguous artifact-free interval, in session seconds. **Not
optional**: a rate is events per artifact-free second, and any window analysis
without these is wrong.

### `channel_qc.csv` — one row per derivation

`contaminated_frac`, `clean_s`, `excluded`, the five `frac_*` per-criterion
fractions, `n_candidates`, `n_events`, `rate_hz`. **`excluded = True` means the
derivation exceeded two-thirds contamination and must not enter any analysis.**

### `bundle/` — everything downstream of detection, without the recordings

| file | contents |
|---|---|
| `ripples.csv` | accepted events only, with subject and site |
| `intervals.csv` | artifact-free intervals, all sessions |
| `pairs.csv` | derivations with coordinates |
| `channel_qc.csv` | per-derivation counts and exclusion flags |
| `behaviour.csv`, `uncover.csv` | task records (project-specific) |
| `swr_bundle.pkl` | all of the above in one object |
| `swr_bundle_figures.npz` | condensed waveforms for redrawing figures |
| `meta.json` | analysis name, creation time, session count |

### Behaviour (project-specific)

Stages 6 and the bundle's behaviour tables are specific to this task. If you
bring your own task, replace the condition functions in `swr_ripple_tests.py`
(`conditions_*`) — they take the event table and return labelled event times.
Everything below them is task-agnostic.

---

## 5. Bringing your own data

### If you have raw recordings

You need a reader. `mc/analyse/swr_preproc.py` handles Blackrock (`.ns2`/`.ns3`
via `neo`) and Neuralynx (`.ncs`, via a direct reader because `neo` cannot open
these files). For anything else, write a loader returning
`(n_channels, n_samples)` in microvolts plus its sampling rate, and call
`resample_to`, then subtract your pairs, then `notch_filter`.

### If you have preprocessed bipolar data — the usual case

Write three files per session and start at stage 3:

```python
import json, numpy as np, pandas as pd

fs = 1000.0                       # resample to this first
signal = ...                      # (n_derivations, n_samples) float32, microvolts
pair_ids = ["LAHC2-LAHC3", "LPHC2-LPHC3"]

out = f"{data_root}/derivatives/s01/LFP-clean/swr_v1"
np.save(f"{out}/continuous.npy", signal.astype(np.float32))
pd.DataFrame({
    "session": 1,
    "pair_id": pair_ids,
    "pair_roi": "HC",             # any label; used for grouping only
    "hemisphere": ["L", "L"],
    "mni_x": [-26.9, -29.1], "mni_y": [-12.5, -29.4], "mni_z": [-23.8, -14.6],
    "subject_label": "SUBJ01",
}).to_csv(f"{out}/pairs.csv", index=False)
json.dump({"session": 1, "fs": fs, "n_pairs": len(pair_ids),
           "n_samples": int(signal.shape[1]), "pair_ids": pair_ids},
          open(f"{out}/meta.json", "w"))
```

Then `python scripts/swr_detect_session.py --session=1`.

**Requirements.** Sampling rate ≥ 1000 Hz (the >250 Hz criterion is impossible
below it). Microvolts, so amplitude thresholds mean what they say. Bipolar or
another local reference — a widely spaced derivation imports distant activity by
volume conduction. And a **continuous** recording: thresholds are estimated over
the whole session, so concatenated epochs give a threshold that belongs to none
of them.

### If you already have ripple times

Write `ripple_events.csv` (at minimum `session`, `pair_id`, `t_peak_s`,
`duration_s`, `passed`), `clean_intervals.csv`, and `channel_qc.csv`, then go
straight to stage 5. Detection parameters will not match this pipeline's, so
report your own.

---

## 6. Using the library directly

No filesystem contract — arrays in, DataFrames out.

```python
import numpy as np
import mc.analyse.swr_artifact as art
import mc.analyse.swr_detect as det

fs = 1000.0
signal = ...                        # (n_samples,) float, microvolts, one derivation

# 1. artifact mask: True = reject this sample
bad, fractions = art.artifact_mask(signal, fs)
clean = ~bad
print(f"rejected {100 * bad.mean():.0f}% of the recording")

# 2. detect ripples on what is left.
#    Returns (events, diagnostics): the second holds the session threshold and
#    the count surviving each stage of the cascade.
events, diag = det.detect_channel(signal, fs, clean)
accepted = events[events.passed]
print(f"{diag['n_candidates']} candidates -> {len(accepted)} accepted")

# 3. a rate is per ARTIFACT-FREE second
rate_hz = len(accepted) / (clean.sum() / fs)
```

Useful entry points:

| function | does |
|---|---|
| `swr_artifact.artifact_mask(x, fs)` | the five criteria, padding, minimum island |
| `swr_artifact.clean_intervals(bad, fs)` | mask → `(n, 2)` array of intervals |
| `swr_detect.ripple_rms(x, fs)` | band-passed signal and its moving RMS |
| `swr_detect.detect_channel(raw, fs, clean)` | the full detector; returns `(events, diagnostics)` |
| `swr_detect.spectral_validate(events, raw, fs)` | the four spectral criteria as flags |
| `swr_bundle.RippleStore(bundle=...)` | read a bundle; `.get(session)` per session |
| `mc.plotting.ripple_figures` | every figure, each taking data rather than paths |

Detector constants live at the top of `mc/analyse/swr_detect.py`
(`LO_SD`, `PEAK_SD`, `HI_SD`, `DUR_MS`, `RIPPLE_BAND`) and artifact constants at
the top of `mc/analyse/swr_artifact.py` (`IQR_K`, `PAD_S`, `MIN_CLEAN_S`,
`MAX_CONTAM_FRAC`). Each carries the measurement that set it.

---

## 7. Where the code lives

| file | holds |
|---|---|
| `mc/analyse/swr_io.py` | paths, config, raw-file discovery, behaviour loading |
| `mc/analyse/swr_preproc.py` | readers, resampling, bipolar, adaptive notch, the session clock |
| `mc/analyse/swr_artifact.py` | the five rejection criteria, padding, clean intervals |
| `mc/analyse/swr_detect.py` | thresholds, dual-threshold detection, spectral criteria |
| `mc/analyse/swr_surrogate.py` | matched 1/f surrogates |
| `mc/analyse/swr_windows.py` | window designs, exposure, covariates |
| `mc/analyse/swr_stats.py` | sliding-window test and cluster permutation |
| `mc/analyse/swr_bundle.py` | building and reading the bundle |
| `mc/analyse/ripples.py` | shared analysis layer: rates, conditions, plots |
| `mc/plotting/ripple_figures.py` | every figure |

---

## 8. Deviations from Chen et al. 2025

Stated so results can be compared fairly.

| | Chen et al. | here | why |
|---|---|---|---|
| amplitude threshold | single, 1.5 SD | dual: 1.5 SD extent, 3.0 SD peak | 1.5 SD alone leaves 97% of detections inside the 1/f noise floor on this dataset |
| threshold estimated | per recording | per derivation, whole session, artifact-free only | a locally estimated threshold tracks the activity it should detect |
| RMS vs envelope | RMS of the band-passed signal | same | the Hilbert envelope has a different SD, so "1.5 SD" is not the same threshold |
| band-pass | 4th-order FIR | 4th-order Butterworth, zero-phase | zero-phase matters for measuring duration |
| notch | fixed, per recording | adaptive, **per derivation**, only above a measured ratio of 2 | a blanket notch removes real 120 Hz signal from clean recordings |
| visual confirmation | every detection inspected | standing diagnostic figures, no per-event inspection | 34 contacts vs ~180 derivations; claiming an inspection not performed would be worse |
| spectral criterion 2 | "30–200 Hz outside the band" | strict (literal) primary, relaxed 120–200 Hz as sensitivity | the literal reading rejects ~10 points more than they report |

---

## 9. Requirements

Python 3.10, `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-image`,
`statsmodels`, `nilearn`, `nibabel`, `neo`, `fire`, and `fooof` for the
surrogate control. 3-D coverage figures additionally need `mne` and `pyvista`.

Atlases are fetched through `nilearn`; set `NILEARN_DATA` to a warm cache if the
compute nodes have no network access.
