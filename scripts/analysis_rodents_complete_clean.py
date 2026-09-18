#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rodent DSR RSA — validation of the action-plan (DSR) model on El-Gaby et
al. (2024) mPFC recordings.

Purpose: show that the same RSA machinery used for the human fMRI and
human single units recovers the concurrent-future-location code that
El-Gaby et al. found at the level of individual mouse neurons, and that
it does NOT find an abstract position-in-sequence code — reproducing both
central properties of the original result.

WHAT IS COMPARED
----------------
Per recording day, the geometry of the recorded population across
timepoints × task configurations is compared against the geometry of four
simulated populations. All four enter ONE joint GLM, so the action-plan
model has to explain variance beyond the simpler codes:

    dsr_fmri  'Action Plan'        concurrent future locations (see below)
    stat      'Position in Seq.'   ordinal position A/B/C/D, config-general
    loc       'Physical Location'  place code (9 grid nodes)
    phas      'Subgoal Progress'   within-state phase (von Mises tuning)

`MODEL_ORDER_DSR` is the fitted combo. The rodent-native DSR variant
(`dsr`) and the location×phase model (`midn`) are computed but excluded:
`dsr` and `dsr_fmri` correlate at r ≈ 0.6 and cannibalise each other's β
in a joint fit (see diagnose_dsr_vs_dsr_fmri.py, 2026-07-27_22-28-27).
`dsr_fmri` is the primary regressor because it is built exactly like the
human pipeline (RSA_DSR_ROIs_simple.build_mode_path_dsr).
NEURAL DATA -> RDM
------------------
Normalised recordings, 360 bins per ABCD loop. Per recday:
  1. PHASE-RESIDUALISE each cell (see PHASE_RESIDUALISE below) — this is
     the step that removes the subgoal-progress signal, which is the
     strongest signal in this dataset and would otherwise dominate.
  2. Average across trials within a task configuration.
  3. Downsample 360 -> N_CONDS_PER_CONFIG conditions (mean over bins).
  4. Concatenate configurations, z-score each neuron across all columns.
  5. RDM = 1 − Pearson correlation across neurons, between every pair of
     (configuration, timepoint) columns.

MODEL RDMs
----------
Built from the MODE path across trials of a configuration (the single
most frequent trajectory, not an average — an averaged path is not a
valid trajectory), via `mc.simulation.predictions.model_DSR`.

`dsr_fmri` specifically: the mode path is downsampled to
N_CONDS_PER_CONFIG × 12 integer node IDs (12 × 12 = 144 with the current
settings), and for each of the N_CONDS_PER_CONFIG timepoints that vector
is rolled left by `pos × 12`, so it always reads
[current segment | next | ... | previous]. Every timepoint therefore
carries the whole remaining trajectory at once, in a frame anchored to
the present. Its RDM is a HAMMING distance — the fraction of future
positions at which two timepoints plan a different node — matching the
"overlap of locations" logic in the paper. The other three models are
cosine RDMs on their rate maps.

Settings that define the parameterisation (and are now written into
`key_analysis_stats.json['settings']`, because two runs differing only in
these are otherwise indistinguishable from the JSON alone):
N_CONDS_PER_CONFIG = 12 timepoints per configuration (30 bins = 30° each,
matching the human pipeline) and NUMBER_PHASE_NEURONS = 3.

TWO ANALYSES PER RECDAY
-----------------------
    1. `full_z`         — all configurations pooled; every off-diagonal
                          pair (within- and across-configuration).
    2. `across_halves`  — configurations recorded in TWO separate
                          sessions are split into those two sessions and
                          similarity is only ever computed across them.
                          Halves share no trials, so no within-session
                          autocorrelation can inflate the fit. Only
                          configurations with ≥2 source sessions qualify.

GROUP STATISTICS
----------------
Each recday's joint GLM is a standardised (z-scored) OLS; the per-recday β
for each model is carried to the group level and tested with a one-sided
one-sample t-test against zero across recdays, BH-FDR corrected across the
four models. Written to `key_analysis_stats.json`.

NOTE ON n: a "recday" is a RECORDING UNIT, not an animal — it is named
`{mouse}_{day1}_{day2}` and is two days that were spike-sorted together
(6 task configurations, 3 per day). The 8 recdays analysed here come from
5 animals:

    ah03 ×1, ah04 ×3, me08 ×1, me10 ×1, me11 ×2

so n = 8 recdays / 5 mice, and the group test treats recording DAYS, not
animals, as independent units.

These 8 are what the private Drive share provided — they are NOT the whole
dataset. The public release (https://osf.io/3d9r2/) has 25 combined ABCD
recdays from 7 mice; ab03 (×3) and ah07 (×3) are absent here entirely.
`scripts/download_rodent_ephys_data.py` fetches the missing 17 (~2.0 GB).

Before using them: OSF ships RAW recordings only, so the normalised 360-bin
view this analysis runs on has to be rebuilt from Neuron_raw + trialtimes for
the new recdays — and then for ALL recdays with the same function, otherwise
the preprocessing difference lines up with the mouse split and confounds the
group test. See the caveat at the top of the download script.

FIGURES (example recday = most neurons unless EXAMPLE_RECDAY is set)
    - fig 1: DSR overview (modelled neurons + DSR model RDM + group betas
             with FDR stars)   [per analysis]
    - fig 2: example-subject row (data + each model's activation and RDM)
             [per analysis]
    - fig 3: model schematics for one configuration  [variant-independent]

The full comparison-matrix sweep (trial filter × pool method × variant +
the continuous pipeline) lives in ``analysis_rodents_legacy_alternatives.py``.

@author: Svenja Kuechenhoff
"""

import os
import json
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed

import mc.analyse.analyse_ephys_clean as ae


# ── Settings ──────────────────────────────────────────────────────────
DATA_FOLDER = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_200423/'
OUT_BASE    = f"{DATA_FOLDER}derivatives/rodent_DSR_RSA/"

# Where the NORMALISED (90 bins/state) arrays come from.
#   None  -> the authors' released files in DATA_FOLDER: 8 recdays / 5 mice.
#   path  -> a self-normalised set from scripts/normalise_rodent_ephys.py:
#            up to 25 recdays / 7 mice, one uniform preprocessing.
# Set this to the derivatives/normalised_* folder to analyse the full release.
# The two must never be mixed — the authors' final normalisation differs from
# their published code and cannot be reproduced (r ~ 0.88), so a mixed set puts
# a preprocessing difference exactly on the mouse/recday split.
NORM_FOLDER = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_200423/derivatives/normalised_loc-max_2026-09-01_12-30-15'

NUMBER_PHASE_NEURONS = 3       # von Mises phase-tuned neurons per state (κ = 10/n = 3.33)
N_CONDS_PER_CONFIG   = 12      # timepoints per task config; 360/12 = 30 bins (30°) each.
                               # Matches the human cell pipeline
                               # (RSA_DSR_ROIs_simple.N_CONDS_PER_CONF = 12).
# Both DSR variants are computed for every recday:
#   'dsr'      — rodent-native DSR (predictions.model_DSR, cosine RDM)
#   'dsr_fmri' — human-pipeline DSR (mode-path rolled by bin, Hamming RDM;
#                same construction as RSA_DSR_ROIs_simple.build_mode_path_dsr)
# PIPELINE B — dsr_fmri (Hamming, fMRI-style) as the primary DSR regressor.
# The joint GLM fits ONLY these 4 regressors: dsr_fmri + 3 rodent controls.
# Rationale (see diagnose_dsr_vs_dsr_fmri.py output 2026-07-27_22-28-27):
# `dsr` and `dsr_fmri` are r≈0.6 correlated, so putting them in the same
# GLM makes them cannibalise each other's β. `dsr_fmri` survives the 4
# control stack on its own (see diagnostic drop-in trace) — this combo is
# the clean "dsr_fmri + controls" report.

#MODEL_ORDER_DSR      = ['dsr_fmri', 'stat', 'loc', 'phas']
MODEL_ORDER_DSR      = ['dsr_fmri', 'stat', 'loc']
# The schematic figures (fig 1 top-panel DSR display, fig 2 model row) show
# every model in MODEL_ORDER_DSR — dsr_fmri included, now that its display
# activation + Hamming RDM are built alongside the cosine-space models.
MODEL_ORDER_FIG      = list(MODEL_ORDER_DSR)
# Whichever DSR variant is present in MODEL_ORDER_DSR is used as the "DSR"
# panel at the top of fig 1. Pipeline A -> 'dsr'; Pipeline B -> 'dsr_fmri'.
DSR_DISPLAY_KEY      = 'dsr_fmri' if 'dsr_fmri' in MODEL_ORDER_DSR else 'dsr'

# Phase residualisation: per-cell OLS of firing rate against a within-state
# phase basis, subtracting the phase component before RDM computation.
#   'cosine'      — 2 basis functions (sin, cos of 2πφ). Same as the human
#                    RSA in RSA_DSR_ROIs_simple.py. Removes the smooth
#                    first-harmonic phase but leaves higher harmonics.
#   'cosine_2h'   — 4 basis functions (adds sin, cos of 4πφ). Removes
#                    first + second harmonic. This is what we use here.
#                    Chosen empirically: the criterion was that the
#                    `phas` (Subgoal Progress) regressor is driven to a
#                    null effect in the group GLM. A bare 'cosine' basis
#                    leaves `phas` significant, because the rodent phase
#                    tuning (von Mises, κ = 3.33) is sharp enough to carry
#                    non-trivial 2nd-harmonic content. In humans the
#                    phase model is already null after 'cosine', so the
#                    extra basis functions aren't needed there — i.e.
#                    this is the STRICTER of the two conventions.
#                    φ = (bin mod 90)/90, the within-state phase; only
#                    the phase component is subtracted, the cell's mean
#                    firing rate is preserved.
#   'categorical' — 3 boxcar indicators (early/middle/late). Tightest
#                    match to the 3-neuron structure of `phas`, but
#                    departs further from the human convention.
#   None          — no residualisation.
PHASE_RESIDUALISE    = 'cosine_2h'

# Phase-residualisation bases to compare, so the justification for using
# PHASE_RESIDUALISE is READ OUT OF THE SCRIPT rather than asserted. Each basis
# is run through the identical per-recday pipeline and group test; the result
# lands in key_analysis_stats.json['phase_residualisation_comparison'].
# The criterion is stated there: the chosen basis is the one that drives the
# Subgoal Progress regressor to a null group effect. Set to [] to skip (costs
# one extra full pass per basis that is not PHASE_RESIDUALISE).
PHASE_BASES_TO_COMPARE = ['cosine', 'cosine_2h']

# Additional model combinations to run and report alongside the primary one, so
# the supplementary analyses are reproducible from THIS script rather than from
# a separate legacy run. Each entry is a full re-run of every recday through the
# same pipeline, with only `combo_order` and `phase_residualise` changed.
# Results land in key_analysis_stats.json['supplementary_model_combos'].
#
# 'pipelineA_original_dsr' is El-Gaby's own action-plan parameterisation
# (`dsr`: 12 future lags x 9 locations x 3 phases = 324 simulated neurons),
# fitted WITHOUT phase residualisation and with the location x phase
# ('midnight') control, i.e. subgoal progress is handled as a GLM regressor
# rather than removed from the data. `dsr_fmri` is deliberately absent: it
# correlates r ~ 0.6 with `dsr` and the two cannibalise each other's beta.
# Set to {} to skip (costs one extra full pass per entry).
SUPPLEMENTARY_COMBOS = {
    'pipelineA_original_dsr': {
        'combo_order':       ['dsr', 'stat', 'loc', 'phas', 'midn'],
        'phase_residualise': None,
    },
}

EXAMPLE_RECDAY = None           # None -> pick the recday with the most neurons
N_JOBS         = -1             # joblib: -1 = all cores, 1 = serial (for debugging)

# Per-recday worker config.
ANALYSIS_CONFIG = {
    'number_phase_neurons': NUMBER_PHASE_NEURONS,
    'n_conds_per_config':   N_CONDS_PER_CONFIG,
    'dsr_pool_methods':     ['mode_path'],
    'run_continuous':       False,   # skip slow per-trial loop; not used here
    'phase_residualise':    PHASE_RESIDUALISE,
    # Combo the joint GLM actually fits. Must match MODEL_ORDER_DSR so the
    # stats aggregation, per-recday β column, and figure box-plot all use
    # the same set of regressors. Pipeline A (rodent-native) would be
    # ['dsr', 'stat', 'loc', 'phas', 'midn']. Pipeline B (this file) uses
    # ['dsr_fmri', 'stat', 'loc', 'phas'] — same list as MODEL_ORDER_DSR.
    'combo_order':          list(MODEL_ORDER_DSR),
    # The following are only consulted when run_continuous is True:
    'no_bins_per_state':    10,
    'mask_within':          True,
    'segmentation':         'reward_dwell',
    'plot_rsa_panels':      False,
}

np.random.seed(42)


# ── Output folder (timestamped) ───────────────────────────────────────
RUN_TAG = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUT_DIR = os.path.join(OUT_BASE, RUN_TAG)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Run output: {OUT_DIR}")


# ── Load + clean ──────────────────────────────────────────────────────
print("Loading mouse data (raw + normalised) ...")
# The recday list comes from whichever source provides the normalised view, so
# pointing NORM_FOLDER at a self-normalised set automatically widens the
# analysis to every recday that set contains.
RECDAYS = ae.discover_recdays(NORM_FOLDER or DATA_FOLDER)
print(f"  normalised view: {NORM_FOLDER or DATA_FOLDER}")
mouse_data      = ae.load_ephys_data(DATA_FOLDER, recdays=RECDAYS, raw=True)
mouse_data_norm = ae.load_ephys_data(DATA_FOLDER, recdays=RECDAYS, raw=False,
                                     norm_folder=NORM_FOLDER)
print(f"  loaded {len(mouse_data)} recdays: {list(mouse_data)}")

# A recday is `{mouse}_{day1}_{day2}`, so the mouse id is the first field.
# The group test runs over recdays, but the manuscript reports mice too —
# derive both here so they can never drift apart.
mouse_of_recday   = {rd: rd.split('_')[0] for rd in sorted(mouse_data)}
recdays_per_mouse = {}
for rd, m in mouse_of_recday.items():
    recdays_per_mouse.setdefault(m, []).append(rd)
print(f"  {len(mouse_data)} recdays from {len(recdays_per_mouse)} mice: "
      + ', '.join(f"{m} x{len(v)}" for m, v in sorted(recdays_per_mouse.items())))

# Sessions surviving into BOTH views — the original authors' implicit "bad
# session" flag is "missing from the normalised view".
keep_by_recday = ae.cross_view_session_ids(mouse_data, mouse_data_norm)

cleaned    = {'raw': {}, 'norm': {}}
clean_meta = {'raw': {}, 'norm': {}}
for recday in sorted(mouse_data):
    for view, src in (('raw', mouse_data), ('norm', mouse_data_norm)):
        cfg, loc, neu, tim, m = ae.clean_ephys_data(
            src[recday]['rewards_configs'], src[recday]['locations'],
            src[recday]['neurons'], src[recday]['timings'], recday,
            session_ids=src[recday]['session_ids'],
            keep_session_ids=keep_by_recday[recday], return_metadata=True)
        cleaned[view][recday]    = (cfg, loc, neu, tim)
        clean_meta[view][recday] = m


# ── Duplicate-config overview ─────────────────────────────────────────
# How many task configs in each recday have ≥2 source sessions (i.e. qualify
# for the across-halves analysis), and how many trials feed each half.
print("\nDuplicate-config overview (qualifying tasks for across-halves analysis):")
print(f"  {'recday':35s} {'n_qualifying':12s} {'configs (h1 trials / h2 trials)':40s}")
for recday in sorted(mouse_data):
    cfg, _, _, tim = cleaned['norm'][recday]
    sid = clean_meta['norm'][recday]['kept_session_ids']
    q   = ae.split_sessions_into_halves(cfg, sid, tim)
    summary = ', '.join(f"{tuple(g['config'])}={g['half1_n_trials']}/{g['half2_n_trials']}"
                        for g in q)
    print(f"  {recday:35s} {len(q):<12d} {summary}")


# ── Repeats overview ──────────────────────────────────────────────────
# Per-task trial counts, per-recday totals, and the average over tasks. Two
# scopes are computed so they can be archived in the stats JSON:
#   - all_data:       all task configs in the cleaned normalised view
#   - across_halves:  only qualifying configs (≥2 sessions, used by the
#                     across-task-halves analysis), split into h1 / h2.
print("\nRepeats overview (after cleaning, normalised view):")
print(f"  {'recday':35s} {'per-task trial counts':40s} "
      f"{'total':>7s} {'avg/task':>10s}")
repeats_all = {}     # per-recday: dict of summary stats for all data
total_per_recday    = {}
avg_per_task_recday = {}
for recday in sorted(mouse_data):
    _, _, _, tim = cleaned['norm'][recday]
    per_task = [int(len(tim[t])) for t in range(len(tim))]
    total = int(sum(per_task))
    avg   = total / max(len(per_task), 1)
    total_per_recday[recday]    = total
    avg_per_task_recday[recday] = avg
    repeats_all[recday] = {
        'n_tasks':            len(per_task),
        'per_task_repeats':   per_task,
        'total_repeats':      total,
        'avg_repeats_per_task': avg,
    }
    pt_str = ' '.join(str(x) for x in per_task)
    print(f"  {recday:35s} {pt_str:40s} {total:>7d} {avg:>10.1f}")

overall_total          = int(sum(total_per_recday.values()))
overall_avg_per_task   = float(np.mean(list(avg_per_task_recday.values())))
overall_avg_per_recday = overall_total / max(len(total_per_recday), 1)
print(f"  {'-' * 90}")
print(f"  {'OVERALL':35s} {'-':40s} {overall_total:>7d} "
      f"{overall_avg_per_task:>10.1f}  (avg/recday: {overall_avg_per_recday:.1f})")

# Across-halves view: only qualifying configs (≥2 sessions) and their
# half-1 / half-2 trial counts.
repeats_halves = {}
for recday in sorted(mouse_data):
    cfg, _, _, tim = cleaned['norm'][recday]
    sid = clean_meta['norm'][recday]['kept_session_ids']
    q   = ae.split_sessions_into_halves(cfg, sid, tim)
    if not q:
        repeats_halves[recday] = {'n_qualifying_configs': 0,
                                  'per_config': [], 'totals': {}}
        continue
    per_cfg = [{'config':  [int(x) for x in g['config']],
                'half1_n_trials': int(g['half1_n_trials']),
                'half2_n_trials': int(g['half2_n_trials'])}
               for g in q]
    h1 = sum(g['half1_n_trials'] for g in per_cfg)
    h2 = sum(g['half2_n_trials'] for g in per_cfg)
    repeats_halves[recday] = {
        'n_qualifying_configs': len(per_cfg),
        'per_config':           per_cfg,
        'totals': {'half1': h1, 'half2': h2,
                   'avg_per_config_half1': h1 / len(per_cfg),
                   'avg_per_config_half2': h2 / len(per_cfg)},
    }

repeats_overview = {
    'all_data': {
        'per_recday': repeats_all,
        'overall': {
            'total_repeats':         overall_total,
            'avg_repeats_per_task':  overall_avg_per_task,
            'avg_repeats_per_recday': overall_avg_per_recday,
            'n_recdays':             len(repeats_all),
        },
    },
    'across_halves': {
        'per_recday': repeats_halves,
        'overall': {
            'n_recdays_with_qualifying_configs': sum(
                1 for v in repeats_halves.values()
                if v['n_qualifying_configs'] > 0),
            'total_qualifying_configs': sum(
                v['n_qualifying_configs'] for v in repeats_halves.values()),
            'total_half1_trials': sum(
                v['totals'].get('half1', 0) for v in repeats_halves.values()),
            'total_half2_trials': sum(
                v['totals'].get('half2', 0) for v in repeats_halves.values()),
        },
    },
}


# ── Run both main analyses in parallel ────────────────────────────────
def run_all_recdays(config):
    """Dispatch every recday through `process_one_recday` with `config`.

    Factored out so the phase-basis comparison below runs through the IDENTICAL
    code path as the primary analysis — only `phase_residualise` differs.
    """
    return Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(ae.process_one_recday)(
            recday,
            cleaned['raw'][recday],  clean_meta['raw'][recday]['kept_session_ids'],
            cleaned['norm'][recday], clean_meta['norm'][recday]['kept_session_ids'],
            None, config,
        )
        for recday in sorted(mouse_data)
    )


print(f"\nDispatching {len(mouse_data)} recdays across {N_JOBS} workers ...")
per_recday = run_all_recdays(ANALYSIS_CONFIG)

full_z_results = {r['recday']: r['dsr_by_pool']['mode_path']['full_z'] for r in per_recday}
halves_results = {r['recday']: r['halves']['across_halves'] for r in per_recday
                  if 'across_halves' in r['halves']}

n_neurons_per_recday    = {r['recday']: int(cleaned['norm'][r['recday']][2][0].shape[0])
                           for r in per_recday}
n_pooled_per_recday     = {r['recday']: int(r['trim_pool_norm']['pooling']['n_configs_out'])
                           for r in per_recday}
n_qualifying_per_recday = {r['recday']: int(r['halves']['metadata']['n_qualifying_configs'])
                           for r in per_recday}


# ── Pick example recday for figures ───────────────────────────────────
example_recday = EXAMPLE_RECDAY or max(n_neurons_per_recday, key=n_neurons_per_recday.get)
print(f"\nExample recday: {example_recday} "
      f"({n_neurons_per_recday[example_recday]} neurons)")


# ── Stats helpers (small, local) ──────────────────────────────────────
def _print_stats(stats):
    print(f"\n=== {stats['pipeline']} ===")
    unit = 'mice' if 'within-mouse' in stats['pipeline'] else 'recdays'
    print(f"  n_{unit}: {stats['n_recdays']}"
          + ('' if unit == 'mice' else
             f" (from {len({rd.split('_')[0] for rd in stats['recdays']})} mice)"))
    print(f"  n_neurons: {stats['n_neurons_summary']}")
    print(f"  n_tasks:   {stats['n_tasks_summary']}")
    print(f"  {'model':18s} {'mean':>8s} {'sd':>8s} {'sem':>8s} "
          f"{'t':>7s} {'p_unc':>9s} {'p_fdr':>9s} {'sig':>4s}")
    for m in MODEL_ORDER_DSR:
        s = stats['models'][m]
        print(f"  {s['label']:18s} {s['mean']:>8.3f} {(s['sd'] or 0):>8.3f} "
              f"{(s['sem'] or 0):>8.3f} {(s['t_group'] or 0):>7.2f} "
              f"{(s['p_group_uncorrected'] or 1):>9.4f} "
              f"{(s['p_group_fdr'] or 1):>9.4f} {'*' if s['sig_fdr'] else '':>4s}")


def _print_stats_order(stats, order):
    """_print_stats for a combo whose model set differs from MODEL_ORDER_DSR."""
    print(f"\n=== {stats['pipeline']} ===")
    unit = 'mice' if 'within-mouse' in stats['pipeline'] else 'recdays'
    print(f"  n_{unit}: {stats['n_recdays']}")
    print(f"  {'model':18s} {'n':>3s} {'mean':>8s} {'t':>7s} {'p_unc':>10s} "
          f"{'p_fdr':>10s} {'sig':>4s}")
    for m in order:
        s = stats['models'][m]
        print(f"  {s['label']:18s} {s['n']:>3d} {s['mean']:>8.3f} "
              f"{(s['t_group'] or 0):>7.2f} {(s['p_group_uncorrected'] or 1):>10.4g} "
              f"{(s['p_group_fdr'] or 1):>10.4g} {'*' if s['sig_fdr'] else '':>4s}")


def _coefs_and_fdr_order(stats, order):
    """_coefs_and_fdr for a combo whose model set differs from MODEL_ORDER_DSR."""
    coefs = {m: np.asarray([stats['models'][m]['coefs_by_recday'][rd]
                            for rd in stats['recdays']])
             for m in order}
    fdr = {m: stats['models'][m]['p_group_fdr'] for m in order}
    return coefs, fdr


def _coefs_and_fdr(stats):
    coefs = {m: np.asarray([stats['models'][m]['coefs_by_recday'][rd]
                            for rd in stats['recdays']])
             for m in MODEL_ORDER_DSR}
    fdr   = {m: stats['models'][m]['p_group_fdr'] for m in MODEL_ORDER_DSR}
    return coefs, fdr


def _average_within_mouse(per_recday_results, n_neurons, n_tasks):
    """Collapse per-recday results to one entry per mouse (mean of the betas).

    Returns the same three structures `methods_results_stats` takes, so the
    n = 5 robustness test runs through the IDENTICAL code path as the primary
    n = 8 test — same one-sided ttest_1samp, same BH-FDR across the same four
    models. Only the unit of observation changes.

    Rationale: the 8 recdays are not exchangeable (ah04 contributes 3, me11 2),
    so a strong effect in one animal can carry the recday-level test. Averaging
    within animal first removes that. It is a ROBUSTNESS check, not a
    replacement: n = 5 is underpowered, so read the effect size, not a null.
    """
    by_mouse = {}
    for rd in per_recday_results:
        # A degenerate recday (all-NaN betas — see degenerate_data_vector)
        # must not drag its animal's mean to NaN.
        if not np.all(np.isfinite(np.asarray(per_recday_results[rd]['coefs'],
                                             dtype=float))):
            continue
        by_mouse.setdefault(rd.split('_')[0], []).append(rd)

    res, neu, tsk = {}, {}, {}
    for mouse, rds in sorted(by_mouse.items()):
        coefs = np.mean([np.asarray(per_recday_results[rd]['coefs'], dtype=float)
                         for rd in rds], axis=0)
        res[mouse] = {'coefs':      coefs,
                      'label_regs': per_recday_results[rds[0]]['label_regs']}
        # neurons are counted per neuron-day, so they sum across a mouse's recdays
        neu[mouse] = int(sum(n_neurons[rd] for rd in rds))
        tsk[mouse] = int(sum(n_tasks[rd]   for rd in rds))
    return res, neu, tsk


# ── Main analysis 1: full DSR z-scored RDM ────────────────────────────
stats_full_z = ae.methods_results_stats(
    full_z_results, n_neurons_per_recday, n_pooled_per_recday,
    model_label_order=MODEL_ORDER_DSR)
stats_full_z['pipeline'] = 'DSR mode_path / all_trials / full_z'
_print_stats(stats_full_z)

# Robustness: same test, one value per ANIMAL instead of per recday.
stats_full_z_by_mouse = ae.methods_results_stats(
    *_average_within_mouse(full_z_results, n_neurons_per_recday, n_pooled_per_recday),
    model_label_order=MODEL_ORDER_DSR)
stats_full_z_by_mouse['pipeline'] = ('DSR mode_path / all_trials / full_z '
                                     '(within-mouse average, n = mice)')
_print_stats(stats_full_z_by_mouse)

# Build example matrices from the pooled normalised view.
cfg_ex, loc_ex, neu_ex, tim_ex = cleaned['norm'][example_recday]
cfg_ex, loc_ex, neu_ex, tim_ex, _ = ae.pool_by_task_config(
    cfg_ex, loc_ex, neu_ex, tim_ex, kind='norm',
    session_ids=clean_meta['norm'][example_recday]['kept_session_ids'],
    return_metadata=True)
fz_data_act, fz_data_rdm, fz_model_acts, fz_model_rdms = (
    ae.dsr_example_recday_matrices(
        cfg_ex, loc_ex, neu_ex, tim_ex,
        n_conds_per_config=N_CONDS_PER_CONFIG,
        no_phase_neurons=NUMBER_PHASE_NEURONS))

fz_coefs, fz_fdr = _coefs_and_fdr(stats_full_z)
ae.pub_figure_dsr_overview(
    dsr_model_activation=fz_model_acts[DSR_DISPLAY_KEY],
    dsr_model_rdm=fz_model_rdms[DSR_DISPLAY_KEY],
    coefs_by_model=fz_coefs, model_order=MODEL_ORDER_FIG, fdr_pvals=fz_fdr,
    n_tasks=len(cfg_ex), n_conds_per_task=N_CONDS_PER_CONFIG,
    recday_label=example_recday,
    save_stem=os.path.join(OUT_DIR, 'fig1_full_z'))
ae.pub_figure_example_subject(
    data_activation=fz_data_act, data_rdm=fz_data_rdm,
    model_activations=fz_model_acts, model_rdms=fz_model_rdms,
    model_order=MODEL_ORDER_FIG,
    n_tasks=len(cfg_ex), n_conds_per_task=N_CONDS_PER_CONFIG,
    recday_label=example_recday,
    save_stem=os.path.join(OUT_DIR, 'fig2_full_z'))

# Same overview figure, but the box/scatter carries ONE POINT PER MOUSE. The
# model panels are identical (they depend on the task, not on the group test);
# only the betas and their FDR stars change.
fzm_coefs, fzm_fdr = _coefs_and_fdr(stats_full_z_by_mouse)
ae.pub_figure_dsr_overview(
    dsr_model_activation=fz_model_acts[DSR_DISPLAY_KEY],
    dsr_model_rdm=fz_model_rdms[DSR_DISPLAY_KEY],
    coefs_by_model=fzm_coefs, model_order=MODEL_ORDER_FIG, fdr_pvals=fzm_fdr,
    n_tasks=len(cfg_ex), n_conds_per_task=N_CONDS_PER_CONFIG,
    recday_label=f'{example_recday} (n = {stats_full_z_by_mouse["n_recdays"]} mice)',
    save_stem=os.path.join(OUT_DIR, 'fig1_full_z_by_mouse'))


# ── Main analysis 2: across-task-halves (mode_path) ───────────────────
stats_halves = ae.methods_results_stats(
    halves_results, n_neurons_per_recday, n_qualifying_per_recday,
    model_label_order=MODEL_ORDER_DSR)
stats_halves['pipeline'] = 'DSR mode_path / across-task-halves'
_print_stats(stats_halves)

stats_halves_by_mouse = ae.methods_results_stats(
    *_average_within_mouse(halves_results, n_neurons_per_recday,
                           n_qualifying_per_recday),
    model_label_order=MODEL_ORDER_DSR)
stats_halves_by_mouse['pipeline'] = ('DSR mode_path / across-task-halves '
                                     '(within-mouse average, n = mice)')
_print_stats(stats_halves_by_mouse)

# Example matrices: POST-clean PRE-pool (we need per-session structure for halves).
cfg_pre, loc_pre, neu_pre, tim_pre = cleaned['norm'][example_recday]
halves_mats = ae.dsr_across_halves_matrices(
    cfg_pre, loc_pre, neu_pre, tim_pre,
    session_ids=clean_meta['norm'][example_recday]['kept_session_ids'],
    n_conds_per_config=N_CONDS_PER_CONFIG,
    no_phase_neurons=NUMBER_PHASE_NEURONS)

if halves_mats is not None:
    h_data_act, h_data_rdm, h_model_acts, h_model_rdms, K_h = halves_mats
    h_coefs, h_fdr = _coefs_and_fdr(stats_halves)
    # Across-halves matrices stack [half-1 of all qualifying configs, half-2
    # of all qualifying configs], so the display axis has 2*K_h task columns.
    K_display = 2 * K_h
    ae.pub_figure_dsr_overview(
        dsr_model_activation=h_model_acts[DSR_DISPLAY_KEY],
        dsr_model_rdm=h_model_rdms[DSR_DISPLAY_KEY],
        coefs_by_model=h_coefs, model_order=MODEL_ORDER_DSR, fdr_pvals=h_fdr,
        n_tasks=K_display, n_conds_per_task=N_CONDS_PER_CONFIG,
        recday_label=f'{example_recday} (across-halves, K={K_h})',
        x_axis_groups=[('Task half 1', K_h), ('Task half 2', K_h)],
        save_stem=os.path.join(OUT_DIR, 'fig1_across_halves'))
    ae.pub_figure_example_subject(
        data_activation=h_data_act, data_rdm=h_data_rdm,
        model_activations=h_model_acts, model_rdms=h_model_rdms,
        model_order=MODEL_ORDER_FIG,
        n_tasks=K_display, n_conds_per_task=N_CONDS_PER_CONFIG,
        recday_label=f'{example_recday} (across-halves, K={K_h})',
        x_axis_groups=[('run 1', K_h), ('run 2', K_h)],
        save_stem=os.path.join(OUT_DIR, 'fig2_across_halves'))

    # One point per mouse (see the full_z by-mouse figure above).
    hm_coefs, hm_fdr = _coefs_and_fdr(stats_halves_by_mouse)
    ae.pub_figure_dsr_overview(
        dsr_model_activation=h_model_acts[DSR_DISPLAY_KEY],
        dsr_model_rdm=h_model_rdms[DSR_DISPLAY_KEY],
        coefs_by_model=hm_coefs, model_order=MODEL_ORDER_DSR, fdr_pvals=hm_fdr,
        n_tasks=K_display, n_conds_per_task=N_CONDS_PER_CONFIG,
        recday_label=(f'{example_recday} (across-halves, '
                      f'n = {stats_halves_by_mouse["n_recdays"]} mice)'),
        x_axis_groups=[('Task half 1', K_h), ('Task half 2', K_h)],
        save_stem=os.path.join(OUT_DIR, 'fig1_across_halves_by_mouse'))
else:
    print(f"\nExample recday {example_recday} has no qualifying configs; "
          f"skipping across-halves figures.")


# ── Phase-residualisation basis comparison ────────────────────────────
# WHY THIS IS IN THE SCRIPT: PHASE_RESIDUALISE = 'cosine_2h' is a choice, and
# the reason for it has to be checkable by whoever runs this next, not taken on
# trust. The criterion, fixed in advance: the residualisation must drive the
# Subgoal Progress regressor to a NULL group effect, because subgoal progress
# is the dominant signal in this dataset and any residue of it would inflate
# the action-plan fit. Whichever basis in PHASE_BASES_TO_COMPARE satisfies that
# is the one to use.
#
# Each basis goes through run_all_recdays -> methods_results_stats, i.e. the
# same functions as the primary analysis, so the numbers are directly
# comparable. The basis equal to PHASE_RESIDUALISE reuses the primary results
# instead of recomputing them.
phase_comparison = {}
if PHASE_BASES_TO_COMPARE:
    print(f"\n{'=' * 70}\nPhase-residualisation comparison: {PHASE_BASES_TO_COMPARE}")
    for basis in PHASE_BASES_TO_COMPARE:
        if basis == PHASE_RESIDUALISE:
            fz_res, hv_res = full_z_results, halves_results
        else:
            cfg_basis = dict(ANALYSIS_CONFIG, phase_residualise=basis)
            print(f"\n-- re-running all recdays with phase_residualise={basis!r} ...")
            pr = run_all_recdays(cfg_basis)
            fz_res = {r['recday']: r['dsr_by_pool']['mode_path']['full_z'] for r in pr}
            hv_res = {r['recday']: r['halves']['across_halves'] for r in pr
                      if 'across_halves' in r['halves']}

        st_fz = ae.methods_results_stats(fz_res, n_neurons_per_recday,
                                         n_pooled_per_recday,
                                         model_label_order=MODEL_ORDER_DSR)
        st_hv = ae.methods_results_stats(hv_res, n_neurons_per_recday,
                                         n_qualifying_per_recday,
                                         model_label_order=MODEL_ORDER_DSR)
        st_fz['pipeline'] = f'full_z / phase_residualise={basis}'
        st_hv['pipeline'] = f'across-halves / phase_residualise={basis}'
        _print_stats(st_fz)
        _print_stats(st_hv)
        phase_comparison[str(basis)] = {'full_z': st_fz, 'across_halves': st_hv}

    # Compact read-out: the Subgoal Progress line is the whole argument.
    print(f"\n{'-' * 70}\nSubgoal Progress (the residualisation criterion):")
    print(f"  {'basis':14s} {'analysis':16s} {'beta':>8s} {'t':>8s} "
          f"{'p_unc':>10s} {'p_fdr':>10s} {'sig':>5s}")
    for basis, blocks in phase_comparison.items():
        for name, st in blocks.items():
            m = st['models']['phas']
            print(f"  {basis:14s} {name:16s} {m['mean']:>8.4f} {m['t_group']:>8.3f} "
                  f"{m['p_group_uncorrected']:>10.4g} {m['p_group_fdr']:>10.4g} "
                  f"{'YES' if m['sig_fdr'] else 'no':>5s}")
    print("  criterion: use the basis whose Subgoal Progress effect is NOT "
          "significant.\n  Action Plan for reference:")
    for basis, blocks in phase_comparison.items():
        for name, st in blocks.items():
            m = st['models']['dsr_fmri']
            print(f"  {basis:14s} {name:16s} {m['mean']:>8.4f} {m['t_group']:>8.3f} "
                  f"{m['p_group_uncorrected']:>10.4g} {m['p_group_fdr']:>10.4g} "
                  f"{'YES' if m['sig_fdr'] else 'no':>5s}")


# ── Supplementary model combinations ──────────────────────────────────
# Same recdays, same cleaning, same group test — only the fitted model set (and
# whether the data were phase-residualised) differs. This is what makes the
# supplementary analyses auditable: nothing here comes from a separate script.
supplementary = {}
for combo_name, spec in SUPPLEMENTARY_COMBOS.items():
    order = list(spec['combo_order'])
    print(f"\n{'=' * 70}\nSupplementary combo {combo_name!r}: {order}, "
          f"phase_residualise={spec['phase_residualise']!r}")
    cfg_c = dict(ANALYSIS_CONFIG, combo_order=order,
                 phase_residualise=spec['phase_residualise'])
    pr = run_all_recdays(cfg_c)
    fz = {r['recday']: r['dsr_by_pool']['mode_path']['full_z'] for r in pr}
    hv = {r['recday']: r['halves']['across_halves'] for r in pr
          if 'across_halves' in r['halves']}

    blocks = {}
    for name, res, ntask in (('full_z', fz, n_pooled_per_recday),
                             ('across_halves', hv, n_qualifying_per_recday)):
        st = ae.methods_results_stats(res, n_neurons_per_recday, ntask,
                                      model_label_order=order)
        st['pipeline'] = f'{combo_name} / {name}'
        _print_stats_order(st, order)
        blocks[name] = st

        st_m = ae.methods_results_stats(
            *_average_within_mouse(res, n_neurons_per_recday, ntask),
            model_label_order=order)
        st_m['pipeline'] = f'{combo_name} / {name} (within-mouse average, n = mice)'
        _print_stats_order(st_m, order)
        blocks[f'{name}_by_mouse'] = st_m

    supplementary[combo_name] = {
        'combo_order':       order,
        'phase_residualise': spec['phase_residualise'],
        'results':           blocks,
    }

    # Figures, matching the primary ones so the supplement can show the same
    # panel. The DSR panel uses whichever action-plan variant this combo fits.
    disp = 'dsr' if 'dsr' in order else 'dsr_fmri'
    for name, K, groups in (('full_z', len(cfg_ex), None),
                            ('across_halves', None, None)):
        if name == 'across_halves':
            if halves_mats is None:
                continue
            acts, rdms, K, groups = (h_model_acts, h_model_rdms, K_display,
                                     [('Task half 1', K_h), ('Task half 2', K_h)])
        else:
            acts, rdms = fz_model_acts, fz_model_rdms
        for unit, key in (('', name), ('_by_mouse', f'{name}_by_mouse')):
            co, fd = _coefs_and_fdr_order(blocks[key], order)
            ae.pub_figure_dsr_overview(
                dsr_model_activation=acts[disp], dsr_model_rdm=rdms[disp],
                coefs_by_model=co, model_order=order, fdr_pvals=fd,
                n_tasks=K, n_conds_per_task=N_CONDS_PER_CONFIG,
                recday_label=(f'{example_recday} — {combo_name}'
                              + (f' (n = {blocks[key]["n_recdays"]} mice)'
                                 if unit else '')),
                x_axis_groups=groups,
                save_stem=os.path.join(OUT_DIR,
                                       f'figS_{combo_name}_{name}{unit}'))


# ── Figure 3: model schematics (variant-independent) ──────────────────
ae.pub_figure_model_schematics(
    walked_path=ae._mode_path_360(loc_ex[0]),
    task_config=cfg_ex[0],
    no_phase_neurons=NUMBER_PHASE_NEURONS,
    recday_label=example_recday,
    save_stem=os.path.join(OUT_DIR, 'fig3_model_schematics'))


# ── Write stats JSON ──────────────────────────────────────────────────
# The settings block is written FIRST and is not optional: two runs with
# different N_CONDS_PER_CONFIG / NUMBER_PHASE_NEURONS produce JSONs that are
# otherwise indistinguishable, which has already caused an 8-vs-12 mix-up.
run_settings = {
    'run_tag':                 RUN_TAG,
    'timestamp':               datetime.now().isoformat(timespec='seconds'),
    'data_folder':             DATA_FOLDER,
    'norm_folder':             NORM_FOLDER or DATA_FOLDER,
    'norm_source':             ('authors_released' if NORM_FOLDER is None
                                else 'self_normalised'),
    'n_conds_per_config':      N_CONDS_PER_CONFIG,
    'bins_per_cond':           360 // N_CONDS_PER_CONFIG,
    'degrees_per_cond':        360 / N_CONDS_PER_CONFIG,
    'number_phase_neurons':    NUMBER_PHASE_NEURONS,
    'von_mises_kappa':         10.0 / NUMBER_PHASE_NEURONS,
    'phase_residualise':       PHASE_RESIDUALISE,
    'model_order_dsr':         MODEL_ORDER_DSR,
    'dsr_display_key':         DSR_DISPLAY_KEY,
    'len_standardised_path_dsr_fmri': ae.LEN_STANDARDISED_PATH_DSR_FMRI,
    'dsr_fmri_matrix_shape':   [N_CONDS_PER_CONFIG,
                                N_CONDS_PER_CONFIG
                                * ae.LEN_STANDARDISED_PATH_DSR_FMRI],
    'random_seed':             42,
    'analysis_config':         ANALYSIS_CONFIG,
    # Sample description. n_recdays is the unit of the group t-test; n_mice is
    # what the manuscript quotes. A recday is `{mouse}_{day1}_{day2}`.
    'sample': {
        'n_recdays':         len(mouse_of_recday),
        'n_mice':            len(recdays_per_mouse),
        'recdays_per_mouse': recdays_per_mouse,
        'mouse_of_recday':   mouse_of_recday,
    },
}

stats_path = os.path.join(OUT_DIR, 'key_analysis_stats.json')
with open(stats_path, 'w') as f:
    json.dump({'settings':                run_settings,
               'full_z':                  stats_full_z,
               'across_halves':           stats_halves,
               # Robustness tests: identical machinery, one value per animal.
               'full_z_by_mouse':         stats_full_z_by_mouse,
               'across_halves_by_mouse':  stats_halves_by_mouse,
               'repeats_overview':        repeats_overview,
               # Justification for PHASE_RESIDUALISE, recomputed every run.
               'phase_residualisation_comparison': {
                   'criterion': ('Use the basis that drives the Subgoal Progress '
                                 'regressor to a null group effect; subgoal '
                                 'progress dominates this dataset and any residue '
                                 'would inflate the action-plan fit.'),
                   'bases_compared': [str(b) for b in PHASE_BASES_TO_COMPARE],
                   'basis_used':     str(PHASE_RESIDUALISE),
                   'results':        phase_comparison,
               },
               # El-Gaby's original DSR parameterisation and any other
               # supplementary model set, run through the same pipeline.
               'supplementary_model_combos': supplementary},
              f, indent=2)
print(f"\nWrote stats: {stats_path}")
print(f"  sample: {len(mouse_of_recday)} recdays from {len(recdays_per_mouse)} mice")
print(f"  settings: {N_CONDS_PER_CONFIG} conds/config "
      f"({360 // N_CONDS_PER_CONFIG} bins = {360 / N_CONDS_PER_CONFIG:.0f}° each), "
      f"{NUMBER_PHASE_NEURONS} phase neurons, "
      f"phase_residualise={PHASE_RESIDUALISE!r}")

# Variables persist in Spyder's variable explorer because the script runs at
# module scope. Uncomment if you want a hard pause before exit.
# import pdb; pdb.set_trace()
