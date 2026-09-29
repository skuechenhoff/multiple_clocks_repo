#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:52:57 2026

@author: Svenja Kuchenhoff

Population RSA on human single units: does the concurrent action-plan
(DSR) code explain the representational geometry of a human brain region,
over and above simpler codes? One ROI at a time, same machinery as the
rodent validation (analysis_rodents_complete_clean.py) and the fMRI
searchlight (fMRI_run_RSA_without_rsatoolbox_clean.py).

WHO IS ANALYSED
---------------
Only sessions in which the subject solved the SAME 8 task configurations
(listed in `configs`), read from
`all_sessions_dsrRSA_grouping_summary.json`. That shared task set is what
makes a pseudo-population across sessions possible. Cells are assigned to
ROIs from `neurons_with_ROI_labels.csv` (`alt_final_roi`, built by
cell_to_roi_july26.py); any ROI covered by fewer than MIN_RSA_SUBJECTS
sessions is dropped before anything is computed.

DATA RDM
--------
Per ROI, per session:
  1. PHASE-RESIDUALISE each cell's 360-bin firing rate against a
     within-state phase basis (PHASE_RESIDUALISE; 'cosine' =
     [sin 2piphi, cos 2piphi]) and subtract only the phase component, so the
     cell's mean rate is preserved. This is done at the DATA level rather
     than by adding a phase RDM regressor: RDM-level partialling with a
     block-structured phase RDM can remove DSR variance that merely shares
     that structure. It is why "DSR" here cannot be a relabelled
     goal-progress signal.
  2. Split correct trials into two independent runs using the per-subject
     grouping log. The split respects BLOCK boundaries (each `grid_no`
     goes wholly to run 1 or run 2, never split), so the two halves share
     no trials. With 3 blocks a 2+1 split is expected; the run-balance
     diagnostic checks that any imbalance is not systematic across
     configurations.
  3. Average across trials within (config, half), downsample 360 bins to
     N_CONDS_PER_CONF = 12 conditions per configuration, z-score per
     neuron. 8 configs x 12 conds = 96 conditions.
  4. RDM = cosine dissimilarity. Two variants are carried through:
       split_halves_z  (PRIMARY, = FDR_TEST) - built from the stacked
           run-1 / run-2 population matrix, so EVERY cell of the RDM is an
           across-run comparison of two independent sub-populations. Both
           the within-config block-diagonal (same config, run 1 vs run 2)
           and the between-config cells contribute.
       between_tasks_z (secondary) - repeats collapsed per config, so the
           block-diagonal would be autocorrelation and is masked out; only
           between-config cells contribute.
     The `within` variant is deliberately never used: it is exactly that
     autocorrelation trap.

MODEL RDMs
----------
Built from behaviour: the modal path across all repeats of a configuration
(per run, per ROI's contributing sessions), then simulated codes on top of
it. `dsr_fmri` is the canonical DSR - the modal trajectory downsampled to
N_CONDS_PER_CONF x LEN_STANDARDISED_PATH integer location IDs and rolled
left by `pos * LEN_STANDARDISED_PATH` at each condition, so every
timepoint carries the whole remaining trajectory anchored to the present
(identical construction to build_mode_path_dsr in the fMRI pipeline;
Hamming RDM). Controls are cosine or Hamming RDMs as appropriate:
`location`, `l2_norm`, `bttn_curr` / `bttn_next` / `bttn_prev`, `state`,
`phase`, `state_phase`, `midnight`, `uncover`, `repeat_counter`,
`reward_path`, plus lag-subset and state-quarter splits of dsr_fmri.

GLM AND INFERENCE
-----------------
`combo_models` defines the joint GLMs; each is one standardised OLS of the
data RDM on the stacked model RDMs, so every regressor's beta controls for
the rest of its combo. Significance comes from a permutation null
(N_PERMUTATIONS), NOT from the parametric p:

  SHUFFLE_MODE = 'shift_and_swap' (recommended, and what the reported
  numbers use). Per trial circular shift PLUS a per-cell re-assignment of
  trials to configurations. The circular shift alone ('shift', legacy)
  leaves each cell's per-configuration firing preference intact, which
  inflates the DSR null - the swap is what makes the null zero-centred
  for every regressor.

The permuted data RDMs are built once per ROI and cached (with a
fingerprint over cells/perms/seed/phase setting/configs), and can be
reused across runs. Perm OLS uses the SAME function as the empirical fit
(`evaluate_model_vec`), vectorised over all permutations - CLAUDE.md
rule #4. One-sided p_perm = fraction of permutation betas >= the
empirical beta.

BH-FDR is applied to one pre-specified confirmatory family:
FDR_SUBMODELS inside FDR_COMBOS at FDR_TEST, across every ROI analysed.
A second, per-combo FDR (`q_fdr_per_combo`) corrects across ROIs
separately within each (combo x sub-model).

RELOAD MODE
-----------
Set RELOAD_RUN to a previous run tag to skip the heavy RSA + permutation
loop entirely and only re-render figures / re-run FDR from that run's
saved CSVs. NOTE: in reload mode the `models` / `combo_models` /
`SHUFFLE_MODE` settings above are NOT applied - the numbers come from the
saved run, whose own settings are in its `config.json`. Set RELOAD_RUN =
None to recompute.

OUTPUTS (in OUT_BASE/<run_tag>/)
  results_summary.csv / results_summary_combos.csv   per-ROI betas + p_perm
  confirmatory_fdr.csv / confirmatory_fdr_per_combo.csv
  rdms/rdms_<ROI>.npz            data + model RDM vectors, for add-on replay
  perm_null_draws/perm_<ROI>.pkl full permutation null draws
  perm_data_rdms/                cached permuted data RDMs
  heatmap_roi_*, roi_*_glassbrains/, pub_figures/, rdm_diagnostics/
  run_balance_diagnostic.csv, data_loss_audit.csv
"""

import os
import glob
import sys
import io
import json
import contextlib
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
# Interactive-backend guard. The default backend on macOS is 'MacOSX', and
# this script calls plt.show() in several places (deliberately -- the
# non-publication figures are meant to be looked at during an interactive
# run). In a BACKGROUND / redirected run there is nobody to close those
# windows, so plt.show() blocks forever: a run launched here sat for 38
# minutes on the plt.show() after the phase-mask diagnostic, having used 55 s
# of CPU, looking exactly like slow permutations. Fall back to the headless
# backend only when stdout is not a terminal, so interactive use is
# unchanged, and always respect an explicit MPLBACKEND.
import matplotlib as _mpl
if not os.environ.get('MPLBACKEND') and not sys.stdout.isatty():
    _mpl.use('Agg')
from matplotlib import pyplot as plt
sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
import era_brewer

REPLOT_PUB_FIG3_ONLY = os.environ.get('RSA_REPLOT_PUB_FIG3_ONLY', '0') == '1'
if REPLOT_PUB_FIG3_ONLY:
    mc = None
    dsr_figs = None
else:
    import mc
    import mc.plotting.dsr_figures as dsr_figs   # shared rodent/human pub figures
from collections import Counter
# import pdb; pdb.set_trace()
# import pdb; pdb.set_trace()

# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR     = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_BASE     = os.path.join(DATA_DIR, 'group', 'DSR_RSA_simple_ROI')

# Canonical ROI colours used by the RSA ROI figures. The atlas/table labels
# remain unchanged below because they are also used for ROI mask lookup.
_ROI_PAL = era_brewer.era_brew('Showgirl2', n=7)
ROI_COLOURS = {
    'EC':          _ROI_PAL[0],
    'mOFC':        _ROI_PAL[4],
    'mPFC':        _ROI_PAL[1],
    'HC_anterior': '#a30d6c',
    'HC_mid':      _ROI_PAL[2],
    'PHC':         '#23677E',
    'PCC':         _ROI_PAL[3],
}

# The ROI table and atlas retain their historical labels, but all analysis
# outputs (tables, plots, titles, and filenames) use these canonical names.
ROI_DISPLAY_NAMES = {
    'ACC': 'mPFC',
    'medialOFC': 'mOFC',
    'medial OFC': 'mOFC',
    'Parahippocampal': 'PHC',
    'Parahippocampus': 'PHC',
}
def _canonical_roi_name(roi):
    """Return the project-wide ROI label used in analysis outputs."""
    return ROI_DISPLAY_NAMES.get(roi, roi)


def _atlas_roi_name(roi):
    """Table column uses canonical names directly — identity passthrough."""
    return roi

# Reload mode: set to the run tag of a previous run (e.g.
# '2026-05-18_16-33-05') to skip the heavy RSA + permutation loop and
# just re-render the overview plots from the saved
# results_summary*.csv files in OUT_BASE/<RELOAD_RUN>/.  None = run fresh.
# 2026-08-27: set to None for the re-run on the rebuilt ROI table. ROI
# membership determines which cells form each pseudo-population, so the
# data RDMs themselves change and a reload would silently re-plot the old
# result. Previous run kept for comparison:
#   '2026-07-30_15-58-51-fixed_cells-fixed_perms'
# 2026-08-31: None so the new `-noL2` combo is actually fitted. The heavy
# part (permuted data RDMs) is NOT recomputed: REUSE_PERMS_FROM_PREVIOUS_RUNS
# below matches on a fingerprint of (cells, n_perms, seed, phase, configs,
# shuffle_mode) - combo_models is not part of it - so the cached pickles
# from 2026-08-27_19-18-20 are symlinked in and only the OLS re-runs.
RELOAD_RUN = None

# ── Cross-run perm cache lookup ───────────────────────────────────────
# When True, before rebuilding any ROI's permutation RDMs the script
# scans every sibling run dir under OUT_BASE for a previously-cached
# perm pickle with the SAME fingerprint (cells, n_perms, seed,
# phase setting, configs, method). If found, it reuses that pickle
# (no recomputation) and symlinks it into the current run dir so
# downstream scripts that look for the canonical location still work.
# `link_reused=True` means the cache is shared on disk via symlink —
# saves ~30 MB per ROI × 7 ROIs = ~200 MB per rerun.
REUSE_PERMS_FROM_PREVIOUS_RUNS = True #False
LINK_REUSED_PERMS = True   # False → copy instead of symlink (safer on weird FS)

# Lightweight publication-figure refresh. Set
# RSA_REPLOT_PUB_FIG3_ONLY=1 to read the saved run configs below and rewrite
# only OUT_BASE/<run>/pub_figures/fig3_human_model_schematics.{pdf,jpg}.
# This deliberately does not load neurons, build data RDMs, or run RSA.
REPLOT_PUB_FIG3_RUNS = ['2026-09-24_17-32-11']
    
#     '2026-06-26_11-30-30-final-State',
#     '2026-06-22_16-17-15-final-DSR',
# ]
REPLOT_PUB_FIG3_MATRIX_CM = 2.0
# Columns in the model-schematic grid. Set so the panel count divides
# evenly (6 models -> 3 gives a tidy 3x2; the old default of 4 left a
# ragged 4+2).
REPLOT_PUB_FIG3_MAX_COLS = 3
# Example task for the model schematics. None = pick automatically, as the
# config whose mode path is most uniform across states (see
# `_pick_smoothest_config`). A schematic should show the mechanism, not a
# subject's messiest route; the criterion and full ranking are printed so
# the choice is reproducible rather than hand-picked. Set a config string
# here to override.
REPLOT_PUB_FIG3_EXAMPLE_CONFIG = None
REPLOT_PUB_FIG3_EXAMPLE_REPEAT = 8

# import pdb; pdb.set_trace()
configs = [
    '3-7-9-5', '8-2-6-7', '1-9-5-8', '4-8-1-3',
    '6-4-2-9', '9-1-3-4', '7-3-4-2', '2-5-7-6',
]


N_CONFIGS = len(configs)
N_CONDS_PER_CONF = 12
LEN_STANDARDISED_PATH = 12
N_PHASES = 3
states           = ['A', 'B', 'C', 'D']
RESOLUTIONx = 1
PLOT_FIGS = False
N_PERMUTATIONS = 1000 # 1000 # 500 # None or 300 — None skips the perm loop (parametric p only)

# Null-construction mode for the permutation loop. See
# `mc.analyse.rsa_perm_rdms.SHUFFLE_MODES` for the full list.
#   'shift'          — LEGACY.  Per-trial circular shift only. Preserves
#                      per-cell per-config firing preferences, which
#                      inflates the DSR null (see `publication_md/method.md`).
#   'shift_and_swap' — RECOMMENDED (default).  Circular shift + per-cell
#                      trial-to-config re-assignment.  Zero-centred null
#                      for every regressor (DSR, state, location, bttn).
SHUFFLE_MODE = 'shift_and_swap'
SPLIT_UNCV_BUTTONS = True

# Phase-based masking. Phase of a condition at position pos inside a config is
# (pos % N_CONDS_PER_CONF) % N_PHASES; with N_PHASES=3 this cycles
# early–middle–late across the 12 conds per config.
#   'full'         -> no masking (all pairs)
#   'within_phase' -> keep only same-phase pairs (e.g. early ↔ early)
#   'across_phase' -> keep only different-phase pairs (e.g. early ↔ middle)
# Drives the *primary* RSA pipeline (used by FDR, summary tables, glassbrains,
# permutations, pub figures).
PHASE_MASK_MODE = 'full'

# Per-cell phase residualisation applied at the 360-bin firing-rate level,
# BEFORE downsampling and RDM construction. Conceptually different from
# adding a phase RDM as a control regressor:
#   * RDM-level partialling can be 'too strong' when the phase RDM is
#     categorical / block-structured vs. graded per-cell phase tuning;
#     it may zero out DSR variance that shares phase's RDM structure.
#   * Data-level pre-residualisation only removes the part of each cell's
#     firing rate that's predicted by a continuous phase basis, more
#     conservative.
# Options: None / 'cosine' / 'cosine_2h' / 'categorical'.
# 2026-09-24: None for the fMRI-matched state round. The fMRI pipeline does
# NOT residualise anything -- it puts `path_rew` in the GLM as a regressor.
# Residualising phase out of the cells at the data level is therefore an
# extra step with no fMRI counterpart, and it is why the previous round
# found state_phase ~ 0 everywhere: the variance had already been removed
# before the RDMs were built. Handle phase with a REGRESSOR, as fMRI does.
PHASE_RESIDUALISE = None  # was 'cosine'

# Per-cell repeat-correct residualisation. Loads the per-cell-residualised
# files produced by scripts/residualise_data_by_repeat.py (regressing each
# cell's 360-bin firing rate against rep_correct, then subtracting the
# linear slope). Use this when you want to control for within-trial repeat
# drift at the DATA level, not via a 'repeat_counter' RDM regressor.
#   False -> load raw 'cell-*-360_bins_passed.csv' files (current default)
#   True  -> load 'cleaned_from_reps/cell-*-360_bins_residualised.csv' files
# Independent of PHASE_RESIDUALISE; the two stack additively.
RESIDUALISE_REPEATS = False

# If True, ALSO run the empirical (no-permutation) RSA in all three phase-mask
# modes and produce comparison heatmaps of the combo betas side-by-side per
# ROI. Cheap; uses the model + data RDMs already built. Does not change the
# primary results.
RUN_PHASE_MODE_COMPARISON = False

assert PHASE_MASK_MODE in ('full', 'within_phase', 'across_phase'), PHASE_MASK_MODE
# Safety: phase residualisation removes phase variance at the data level, so
# any phase-mask filter on RDM cells would be applied to data that no longer
# carries phase signal — double removal, and confusing to interpret. Force
# one or the other.
if PHASE_RESIDUALISE is not None and PHASE_MASK_MODE != 'full':
    raise AssertionError(
        f"PHASE_RESIDUALISE={PHASE_RESIDUALISE!r} implies phase is removed at "
        f"the data level; PHASE_MASK_MODE must therefore stay 'full' (no "
        f"RDM-cell filtering), got {PHASE_MASK_MODE!r}.")
# Same safety for the comparison plot: if data is phase-residualised, the
# 'within_phase' / 'across_phase' sweep has no biological meaning; turn it off.
if PHASE_RESIDUALISE is not None and RUN_PHASE_MODE_COMPARISON:
    print("[notice] PHASE_RESIDUALISE is set; forcing "
          "RUN_PHASE_MODE_COMPARISON=False (phase-mask sweep meaningless on "
          "phase-residualised data).")
    RUN_PHASE_MODE_COMPARISON = False
# Phase regressor pruning from combos happens below, after `combo_models`
# is defined (see "Phase-residualisation combo pruning" block).

PLOT_GLASSBRAINS = True
# ── Models / combos to evaluate per ROI this round ────────────────────
# All model RDMs are built each run (cheap). These lists only restrict the
# *expensive* per-ROI evaluation + permutation step.
# - `models`: base models evaluated per ROI. Use `[]` to skip single-model RSA.
# - `combo_models`: combos evaluated per ROI. Sub-models are pulled from the
#   always-built model_RDMs, so combos may reference any model regardless of
#   what's in `models`.
# dsr_old removed: after phase residualisation it flips sign (β goes from
# +0.029 to −0.029 in ACC between_tasks_z) — the clock-ring rotation fits
# phase variance that is no longer in the data.
# dsr_fmri (Hamming on rolled 144-int mode-trajectory) is the canonical
# DSR model. Reduced-lag splits dsr_fmri_lag01 / lag012 / lag0123 keep
# only the first K of 12 lag-windows so we can test the
# spatial-peaks +30/+60 prediction at the RDM level.
# 2026-09-24 STATE ROUND: single-model sweep trimmed to the two models this
# round reports; the full DSR-round list is restored by deleting the two
# lines below this comment block.
models = ['state', 'phase', 'reward_path', 'state_phase']
_models_full_dsr_round = [
    # DSR variants of interest (3 only)
    'dsr_fmri',           # full DSR  (control)
    'dsr_fmri_fut',       # drop lag 0 (control)
    'dsr_fmri_informed',  # lags 1,2 — pre-registered from independent fMRI prior (FDR target)
    # State-quarter split of dsr_fmri (mirrors the fMRI split4_DSR_keys in
    # create_fMRI_model_RDMs_on_clean_beh.py). Each quarter = 3 phases (36
    # bins) of the rolled trajectory — one state's contribution to the plan.
    'curr_quarter', 'next_quarter', 'next2_quarter', 'next3_quarter',
    # Confound controls + diagnostics
    'state', 'midnight', 'bttn_prev',
    'bttn_curr', 'bttn_next', 'location', 'l2_norm',
    'phase', 'repeat_counter', 'uncover', 'state_phase',
    'reward_path',
]

# Same control stack across every combo so the only thing that varies is the
# DSR feature subset — gives a clean head-to-head DSR-variant comparison.
# L2-norm is the negative-distance-from-current-location-to-each-of-9-grid-
# locations regressor, mirroring the fMRI version in
# create_fMRI_model_RDMs_on_clean_beh.py (cosine RDM, 9-feature vector).
# 2026-08-31: l2_norm taken OUT of the shared stack and named per combo, so
# that '-noL2' can drop it while every other combo keeps identical membership.
# Why it matters: l2_norm (graded negative distance to each of 9 grid nodes)
# correlates r = 0.588 with the categorical `location` RDM - two
# parameterisations of one variable. Fitting both splits the spatial variance
# and removes the mid-HC location effect (HC_mid location beta .033 p=.025
# without l2_norm -> .015 p=.223 with it). '-noL2' is the stack the published
# Fig 3g caption describes, so both can now be reported side by side.
_CTRLS_FINAL = ['location', 'bttn_curr', 'bttn_next']
# 2026-09-24 STATE ROUND. Original DSR-round combos are preserved below,
# commented out; restore them for the DSR analyses.
#   'ctrl_fMRI_dsrFULL':            _CTRLS_FINAL + ['dsr_fmri', 'l2_norm'],
#   'ctrl_fMRI-state_dsrFULL':      _CTRLS_FINAL + ['state', 'dsr_fmri', 'l2_norm'],
#   'ctrl_fMRI-state_dsrFULL-noL2': _CTRLS_FINAL + ['state', 'dsr_fmri'],
#   'ctrl_dsrFUT':                  _CTRLS_FINAL + ['dsr_fmri_fut', 'l2_norm'],
#
# `fmri_matched_state` mirrors the fMRI Abstract-task GLM in
# create_fMRI_model_RDMs_on_clean_beh.py:
#     state + location + l2_norm + A-state + DSR + path_rew + duration
# Cell-side equivalents exist for 5 of the 7: state, location, l2_norm,
# dsr_fmri (= DSR), reward_path (= path_rew). NOT matched:
#   * A-state — an fMRI-only visual control marking the A_reward bin, i.e.
#     the reward-feedback screen. No cell-side equivalent is built.
#   * duration — the fMRI bins vary in length; the cell traces are already
#     time-warped to 90 bins per state, so duration is constant by
#     construction and the regressor would be degenerate.
# Note the fMRI abstract-task GLM has NO button regressors (those belong to
# the DSR GLM), so _CTRLS_FINAL is deliberately not used here.
#
# `..._ctrlStatePhase` adds the state x phase RDM. This is the RSA analogue
# of what the fMRI model does by construction: the fMRI `state` model is a
# one-hot over A-D that is IDENTICAL for the path and reward bins of a
# state, so the fMRI GLM can only ever see the phase-invariant part of the
# position code. The cell model has 3 phases per state and no such
# constraint, so a cell whose position selectivity is phase-locked still
# loads on `state`. PHASE_RESIDUALISE='cosine' removes the phase MAIN
# effect at the data level but leaves state x phase untouched.
combo_models = {
    # 2026-09-25 ROUND: isolate how position, goal progress and their
    # conjunction share variance, with and without the sensory/action controls.
    # `state_phase` is the 12-way conjunction and correlates r = 0.62 with
    # `state` and r = 0.37 with `phase` on the model RDMs, so these are
    # deliberately collinear fits -- read them as a variance partition, not as
    # independent effects.
    'SxP_S_P':      ['state_phase', 'state', 'phase'],
    'SxP_S_P_ctrl': ['state_phase', 'state', 'phase',
                      'location', 'l2_norm', 'dsr_fmri'],
    'S_P_ctrl':     ['state', 'phase',
                      'location', 'l2_norm', 'dsr_fmri'],
    # Previous round kept for reference:
    #   'fmri_matched_state':               ['state','location','l2_norm','dsr_fmri','reward_path'],
    #   'fmri_matched_state_ctrlStatePhase': [... , 'state_phase'],
}
assert all(len(set(sm)) == len(sm) for sm in combo_models.values()), \
    f"Duplicate sub-model in combo_models: {combo_models}"


# ── Phase-residualisation combo pruning ────────────────────────────────
# When phase is removed at the data level, the 'phase' RDM regressor adds
# no signal and just clutters the combo output. Drop it from every combo.
if (not REPLOT_PUB_FIG3_ONLY) and PHASE_RESIDUALISE is not None:
    _stripped = 0
    for _name, _subs in list(combo_models.items()):
        if 'phase' in _subs:
            combo_models[_name] = [m for m in _subs if m != 'phase']
            _stripped += 1
    if _stripped:
        print(f"[notice] PHASE_RESIDUALISE={PHASE_RESIDUALISE!r} active — "
              f"dropped 'phase' from {_stripped} combo(s)' sub-model lists "
              f"(redundant after data-level residualisation).")

if not REPLOT_PUB_FIG3_ONLY:
    print(f"Base models evaluated this run ({len(models)}): {models}")
    print(f"Combos evaluated this run     ({len(combo_models)}): {list(combo_models.keys())}")


# ── Multiple-comparison correction (confirmatory family) ─────────────
# BH-FDR is applied to ONE pre-specified family: the effect(s) of interest
# inside the core combo model(s), for the theory-chosen primary test
# variant, across every ROI tested.  Everything else (single-model RSA,
# control regressors, other combos, other test variants) stays
# uncorrected / exploratory.
#   FDR_COMBOS differ only by `state` -> their dsr_old betas are
#   correlated; set FDR_COMBOS to a single combo if you consider them
#   one hypothesis (a 10-test family rather than 20).
FDR_TEST      = 'split_halves_z'    # primary variant. Data RDM is built
                                    # from a 2-half (run-1 × run-2)
                                    # population matrix; each RDM cell is
                                    # an ACROSS-RUN comparison between two
                                    # independent sub-populations. BOTH
                                    # within-task-across-runs (block-diagonal,
                                    # same config in run 1 vs run 2 — valid
                                    # because halves are independent) AND
                                    # between-task off-block cells contribute.
                                    # We use this variant as primary because
                                    # the spatial-peaks finding shows ACC's
                                    # signal is config-dependent: it survives
                                    # within-task generalisation but only
                                    # marginally between-task.
                                    # 'between_tasks_z' (run-averaged,
                                    # between-task only) remains computed
                                    # and reported as a secondary test.
# Confirmatory family: ONE primary combo × the effect of interest × all
# ROIs tested (≈ 7-9 tests). `MRI_combo-nofdb_midn` is treated as a
# robustness check rather than a second confirmatory test, since its
# `dsr_old` beta is highly correlated with the primary combo (the two
# differ only by the `state` regressor). This keeps the FDR family
# consistent with the publication panel (encoding_publication_panels.py).

# settings for state.
# FDR_COMBOS    = ['SxP_S_P', 'SxP_S_P_ctrl', 'S_P_ctrl']         
# FDR_SUBMODELS = ['state']       
# settings for DSR.
# 2026-09-24 STATE ROUND (DSR-round setting kept below for restoring):
#   FDR_COMBOS    = ['ctrl_fMRI_dsrFULL', 'ctrl_fMRI-state_dsrFULL',
#                    'ctrl_fMRI-state_dsrFULL-noL2', 'ctrl_dsrFUT']
#   FDR_SUBMODELS = ['dsr_fmri']
FDR_COMBOS    = ['SxP_S_P', 'SxP_S_P_ctrl', 'S_P_ctrl']
FDR_SUBMODELS = ['state']       

FDR_ALPHA     = 0.05
MIN_RSA_SUBJECTS = 3   # drop ROIs with fewer RSA-eligible subjects than this


# import pdb; pdb.set_trace()
# ── ROI assignment from MNI-based table (cell_to_roi_MNI.py output) ───
# Each neuron is matched by (subject, cell_idx) parsed from its label.
ROI_TABLE_PATH = os.path.join(
    DATA_DIR, 'neurons_with_ROI_labels.csv'
)

# Which ROI-label column of that table to use:
#   'final_roi'     -> original labelling
#   'alt_final_roi' -> alternative labelling (ACC split by y-cutoff,
#                      OFC11+OFC13+ventral_ACC collapsed into medialOFC)
ROI_LABEL_COLUMN = 'alt_final_roi' #'final_roi'

# Which ROI values to analyse this run — one list per labelling column.
# Precuneus added 2026-07-26 (Brainnetome A31 / dmPOS), matching the
# corresponding ROI added in cell_to_roi_MNI.py. Visual and medial_CC are
# NOT listed here: both have < 3 contributing subjects in the current
# dataset and are NaN'd out of `alt_final_roi` by the subject-count
# filter — including them would only produce empty rows.
ROIS_TO_ANALYZE_BY_COLUMN = {
    'final_roi': [
        'EC', 'Parahippocampal',
        'HC_anterior', 'HC_mid',
        'ventral_ACC', 'ACC',
        'OFC11', 'OFC13', 'PCC', 'Precuneus',
    ],
    'alt_final_roi': [
        'EC', 'mPFC', 'mOFC',
        'HC_anterior', 'HC_mid',
        'PCC',
    ],
}
ROIS_TO_ANALYZE = [
    _canonical_roi_name(roi)
    for roi in ROIS_TO_ANALYZE_BY_COLUMN[ROI_LABEL_COLUMN]
]

# ROI that gets the shared rodent-style publication figures (fig 2 + fig 3)
# saved into ``OUT_DIR/pub_figures/``. Set to None to disable, or to another
# ROI name from ROIS_TO_ANALYZE to pick a different reference ROI.
EXAMPLE_ROI_FOR_FIGS = 'mPFC'


def parse_neuron_label(label):
    """Parse '01_07-07-chan120-EC' into (sub:int, cell_idx:int)."""
    try:
        sub_str, rest = label.split('_', 1)
        cell_idx_str = rest.split('-', 1)[0]
        return int(sub_str), int(cell_idx_str)
    except (ValueError, IndexError):
        return None, None


def _load_roi_table(path, roi_col):
    df = pd.read_csv(path)
    # Prefer the corrected final coordinates; fall back to original MNI_x/y/z
    # for tables that pre-date the july26 pipeline.
    for ax in ('x', 'y', 'z'):
        final_col = f'MNI_{ax}_final'
        orig_col  = f'MNI_{ax}'
        if final_col in df.columns:
            df[orig_col] = df[final_col]
    needed = ['subject', 'cell idx', roi_col,
              'MNI_x', 'MNI_y', 'MNI_z', 'electrode label']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"ROI table {path} missing columns: {missing}  "
            f"(re-run scripts/cell_to_roi_MNI.py if {roi_col!r} is absent)")
    df = df.copy()
    df['subject']  = df['subject'].astype(int)
    df['cell idx'] = df['cell idx'].astype(int)
    return df.set_index(['subject', 'cell idx'])


if REPLOT_PUB_FIG3_ONLY:
    ROI_TABLE = None
else:
    ROI_TABLE = _load_roi_table(ROI_TABLE_PATH, ROI_LABEL_COLUMN)
    print(f"Loaded ROI table with {len(ROI_TABLE)} cells "
          f"({ROI_TABLE[ROI_LABEL_COLUMN].nunique()} distinct ROIs) from "
          f"{ROI_TABLE_PATH}  [column: {ROI_LABEL_COLUMN}]")
# the ROI_LABEL_COLUMN column indicates the correct ROI to take.


def get_neuron_roi(label):
    sub, cell_idx = parse_neuron_label(label)
    if sub is None:
        return None
    try:
        roi = ROI_TABLE.loc[(sub, cell_idx), ROI_LABEL_COLUMN]
    except KeyError:
        return None
    if isinstance(roi, pd.Series):
        roi = roi.dropna().iloc[0] if roi.notna().any() else None
    return None if (roi is None or pd.isna(roi)) else str(roi)


def get_neuron_mni(label):
    sub, cell_idx = parse_neuron_label(label)
    if sub is None:
        return (np.nan, np.nan, np.nan)
    try:
        row = ROI_TABLE.loc[(sub, cell_idx)]
    except KeyError:
        return (np.nan, np.nan, np.nan)
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return (float(row['MNI_x']), float(row['MNI_y']), float(row['MNI_z']))


def _make_roi_predicate(target_roi):
    def pred(label):
        return get_neuron_roi(label) == _atlas_roi_name(target_roi)
    return pred


ROI_RULES = {roi: _make_roi_predicate(roi) for roi in ROIS_TO_ANALYZE}

# ── Dynamic ROI filtering by RSA-subject count ────────────────────────
# After building the initial ROI_RULES from ROIS_TO_ANALYZE_BY_COLUMN,
# count how many unique RSA-eligible subjects have ≥1 cell in each ROI.
# Any ROI below MIN_RSA_SUBJECTS is dropped from both ROIS_TO_ANALYZE and
# ROI_RULES before any data are processed.  This runs even in RELOAD_RUN
# mode so that re-plotted figures also respect the threshold.
_grouping_json_path = os.path.join(DATA_DIR, 'all_sessions_dsrRSA_grouping_summary.json')
if os.path.exists(_grouping_json_path) and ROI_TABLE is not None:
    with open(_grouping_json_path) as _jf:
        _rsa_subj_ids = set(int(s) for s in json.load(_jf).keys())
    _rsa_roi_tbl = ROI_TABLE.reset_index()
    _rsa_roi_tbl = _rsa_roi_tbl[_rsa_roi_tbl['subject'].isin(_rsa_subj_ids)]
    _rsa_subj_counts = (
        _rsa_roi_tbl.groupby(ROI_LABEL_COLUMN)['subject'].nunique()
    )
    _rois_ok = set(_rsa_subj_counts[_rsa_subj_counts >= MIN_RSA_SUBJECTS].index)
    _dropped_rois = [r for r in ROIS_TO_ANALYZE if r not in _rois_ok]
    ROIS_TO_ANALYZE = [r for r in ROIS_TO_ANALYZE if r in _rois_ok]
    if _dropped_rois:
        print(f"[RSA filter] Dropped ROIs with < {MIN_RSA_SUBJECTS} RSA subjects "
              f"(out of {len(_rsa_subj_ids)} total): {_dropped_rois}")
    print(f"[RSA filter] ROI subject counts among RSA subjects:\n"
          + "\n".join(f"  {roi}: {_rsa_subj_counts.get(roi, 0)}"
                      for roi in list(_rois_ok) + _dropped_rois))
    print(f"[RSA filter] Final ROIs to analyze: {ROIS_TO_ANALYZE}")
    ROI_RULES = {roi: _make_roi_predicate(roi) for roi in ROIS_TO_ANALYZE}


def reward_locations_for_config(config_str):
    """Return the 4 reward location IDs (1..9) for a config like '3-7-9-5'.

    Order is A, B, C, D — matches the state ordering in the model.
    """
    return tuple(int(x) for x in config_str.split('-'))


def _reward_window_one_trial(loc_1d, btn_1d, reward_locs,
                              n_states=4, uncover_label='Return'):
    """Per-trial 0/1 reward window with CIRCULAR end-extension.

    For each state k, the window START is anchored to that state's
    state-change bin = the last bin of state k's slot, i.e.
    ``(k+1) * W - 1`` (bins 89 / 179 / 269 / 359 for the 4×90 layout).
    These are the warped-time equivalents of t_A / t_B / t_C / t_D from
    state_boundaries.csv, set per repeat by the MATLAB warping pipeline
    (save_humanABCDneurons_normalised.m). They are the best per-trial
    proxy for 'spacebar press at the reward location' — the actual
    Return-press samples aren't always aligned with the state-change
    moment (presumably because not every press is captured at every
    sample), but the time warp itself is anchored to that event.

    The window END extends forward CIRCULARLY mod ``n`` as long as
    ``loc`` stays equal to the reward. Essential for state D: the
    anchor at bin 359 is followed by the participant chilling at
    reward D through bins 0..~50 of the SAME trial's circular layout.
    Treating each per-repeat 360-bin trace as a closed loop captures
    that wrap-around exactly as the user spec asks for.

    ``btn_1d`` and ``uncover_label`` are kept in the signature for API
    stability but are not consulted — the state-change anchor already
    encodes the press event.

    Used as the building block; the model feature aggregates this across
    trials with mode-per-bin (see build_reward_path_label_360).
    """
    del btn_1d, uncover_label   # state-change anchor encodes the press
    n = len(loc_1d)
    label = np.zeros(n, dtype=int)
    W = n // n_states
    loc = np.asarray(loc_1d)
    for k in range(n_states):
        start = (k + 1) * W - 1          # bin 89 / 179 / 269 / 359
        target = float(reward_locs[k])
        # Skip degenerate trials where the warping anchor itself is
        # not at the reward — those have a real data gap and labelling
        # them would be guesswork.
        if not np.isclose(float(loc[start]), target):
            continue
        pos = start
        steps = 0
        while steps < n and np.isclose(float(loc[pos % n]), target):
            label[pos % n] = 1
            pos += 1
            steps += 1
    return label


def build_reward_path_label_360(loc_trials, btn_trials, config_str,
                                  n_states=4, uncover_label='Return',
                                  return_per_trial=False):
    """Per-bin binary reward-vs-path label aggregated across trials.

    Builds the reward window PER TRIAL using each trial's raw loc + btn
    (see ``_reward_window_one_trial``), then aggregates across trials by
    taking the per-bin mode (= 1 wherever ≥50% of trials were at reward
    in that bin, else 0). This mirrors how mode_locs / mode_buttons are
    themselves built and is the correct sibling of those arrays.

    Parameters
    ----------
    loc_trials, btn_trials : 2-D arrays of shape (n_trials, n_bins)
        Raw per-trial location and button traces (NOT the per-bin mode).
    config_str : str
        Config name like '3-7-9-5' giving the 4 reward location IDs.
    return_per_trial : bool
        If True, also return the (n_trials, n_bins) per-trial label array
        for diagnostics (e.g. flagging real data gaps).
    """
    loc_arr = np.asarray(loc_trials)
    btn_arr = np.asarray(btn_trials)
    assert loc_arr.shape == btn_arr.shape, (
        f"loc/btn shapes mismatch: {loc_arr.shape} vs {btn_arr.shape}")
    rewards = reward_locations_for_config(config_str)
    per_trial = np.stack([
        _reward_window_one_trial(loc_arr[i], btn_arr[i], rewards,
                                  n_states=n_states,
                                  uncover_label=uncover_label)
        for i in range(loc_arr.shape[0])
    ], axis=0)
    # Mode across trials per bin: 1 wherever majority of trials had reward.
    agg = (per_trial.mean(axis=0) >= 0.5).astype(int)
    if return_per_trial:
        return agg, per_trial
    return agg


def downsample_mode(x, target_len=10):
    """Mode-downsample x to ``target_len`` slots without discarding bins.

    Using ``block = len(x) // target_len`` truncated whenever target_len did
    not divide len(x): with len(x)=360 and target_len=144 (n_dsr_neurons or
    n_conds_per_config*len_per_bin) the last 72 raw bins were dropped and
    every cond's downsampled window was misaligned by 6 bins relative to
    the conceptual 30-bin layout. Similarly target_len=12 on a 30-bin
    subpath dropped 6 bins per cond.

    This version distributes input bins evenly across the output slots:
    slot i = x[(i*n)//target_len : ((i+1)*n)//target_len]. All bins are
    used; slot sizes differ by at most 1.
    """
    x = np.asarray(x, dtype=object)
    n = len(x)
    return np.array([
        Counter(x[(i * n) // target_len : ((i + 1) * n) // target_len])
            .most_common(1)[0][0]
        for i in range(target_len)
    ], dtype=object)


def make_phase_masks_for_cells(n_configs, n_conds_per_config, n_phases,
                                include_diagonal=False):
    """Phase-based boolean masks aligned with the cells RSA vector layouts.

    Phase of a condition at position ``pos`` inside a config is
    ``(pos % n_conds_per_config) % n_phases``; with N_PHASES=3 this cycles
    early–middle–late–early–middle–late–… across the 12 conds per config.

    Returns a dict ``{variant: {mode: 1d_bool_array}}`` plus a
    ``'_phase_per_condition'`` vector and a ``'_n'`` size, both for
    diagnostics.

      variants:
        'split_halves'   -> mask aligned with ``compute_crosscorr``'s output.
                            The DATA RDM here is built from a 2-half (run 1
                            vs run 2) population matrix: every RDM cell is
                            an ACROSS-RUN comparison between two independent
                            sub-populations. That includes BOTH:
                              * the within-task block-diagonal (same config,
                                run 1 vs run 2 — valid because independent
                                halves), AND
                              * between-task off-block cells (different
                                configs, run 1 vs run 2).
                            Mask shape: upper-tri (k=1 unless
                            include_diagonal) of a symmetrized cross-half
                            N×N block, where N = n_configs *
                            n_conds_per_config.
        'between_tasks'  -> mask aligned with ``compute_crosscorr_within``'s
                            between-block output. The DATA RDM here is
                            built from repeats pre-averaged per config (one
                            population vector per condition, all runs
                            collapsed), so the within-config block-diagonal
                            would be autocorrelations (same averaged vector
                            on both axes) and is EXCLUDED. ONLY between-task
                            cells of the upper-tri of the run-averaged N×N
                            RDM contribute.

      modes:
        'full'         -> all True (no masking).
        'within_phase' -> True where the two conds share a phase.
        'across_phase' -> True where the two conds have different phases.
    """
    n = n_configs * n_conds_per_config
    phase = np.tile(np.arange(n_conds_per_config) % n_phases, n_configs)
    k = 0 if include_diagonal else 1
    ii, jj = np.triu_indices(n, k=k)
    same_phase = phase[ii] == phase[jj]
    between_block = (ii // n_conds_per_config) != (jj // n_conds_per_config)
    split_halves = {
        'full':         np.ones_like(same_phase, dtype=bool),
        'within_phase': same_phase,
        'across_phase': ~same_phase,
    }
    same_phase_bt = same_phase[between_block]
    between_tasks = {
        'full':         np.ones_like(same_phase_bt, dtype=bool),
        'within_phase': same_phase_bt,
        'across_phase': ~same_phase_bt,
    }
    return {
        'split_halves':         split_halves,
        'between_tasks':        between_tasks,
        '_phase_per_condition': phase,
        '_n':                   n,
    }


# Layout-only — does not depend on data, so build once.
PHASE_MASKS = make_phase_masks_for_cells(
    N_CONFIGS, N_CONDS_PER_CONF, N_PHASES, include_diagonal=False)

ALL_PHASE_MODES = ('full', 'within_phase', 'across_phase')


def _phase_mask_for(test_name, mode):
    """Mask vector for (test_name, mode). Returns None when no masking is needed.

    test_name may end in '_z' (the permutation loop uses suffixed names);
    that suffix is stripped for the variant lookup.
    """
    if mode == 'full':
        return None
    base = test_name[:-2] if test_name.endswith('_z') else test_name
    return PHASE_MASKS[base][mode]


def _apply_phase_mask(arr, mask):
    """Slice a 1-D RDM vector or a (n_pairs, n_models) stacked design matrix."""
    if mask is None:
        return arr
    a = np.asarray(arr)
    return a[mask] if a.ndim == 1 else a[mask, :]


def _phase_mask_matrix(mode, n_configs, n_conds_per_config, n_phases):
    """Square N×N boolean mask (True = kept) for visualising RDMs.

    Same definition as the 1-D vector masks but expanded to a full N×N matrix
    (no upper-tri restriction) so it can be applied to a square RDM image.
    """
    n = n_configs * n_conds_per_config
    phase = np.tile(np.arange(n_conds_per_config) % n_phases, n_configs)
    same_phase_mat = phase[:, None] == phase[None, :]
    if mode == 'full':
        return np.ones((n, n), dtype=bool)
    if mode == 'within_phase':
        return same_phase_mat
    if mode == 'across_phase':
        return ~same_phase_mat
    raise ValueError(f"unknown phase mode {mode!r}")


# 3×3 grid coordinates in the same convention as
# create_fMRI_model_RDMs_on_clean_beh.py (location ID 1..9 → (x, y)).
# The L2-norm model uses the negative Euclidean distance from the current
# location to each of the 9 grid cells as a 9-feature condition vector
# (cosine similarity RDM, identical to the fMRI definition).
_LOC_COORD = {
    1: (-0.21,  0.29), 2: (0.0,  0.29), 3: (0.21,  0.29),
    4: (-0.21,  0.0 ), 5: (0.0,  0.0 ), 6: (0.21,  0.0 ),
    7: (-0.21, -0.29), 8: (0.0, -0.29), 9: (0.21, -0.29),
}
GRID_COORDS = np.array([_LOC_COORD[i] for i in range(1, 10)], dtype=float)
GRID_L2 = np.linalg.norm(
    GRID_COORDS[:, None, :] - GRID_COORDS[None, :, :], axis=-1
)  # (9, 9) pairwise L2; row i is distance from loc i+1 to each grid cell.


def l2_norm_row_for_loc(loc_id):
    """9-vector of NEGATIVE L2 distance from `loc_id` (1..9) to each grid cell."""
    return -GRID_L2[int(loc_id) - 1]


def _vec_to_square_rdm(vec_1d, n):
    """Reconstruct a symmetric n×n RDM from its upper-tri (k=1) 1-D vector.

    The compute_crosscorr / compute_hamming pipelines emit the upper-triangle
    above the main diagonal as a 1-D array (same layout as
    ``np.triu_indices(n, k=1)``). This helper inverts that so we can visualise
    exactly which cells fed evaluate_model under each RDM-mask variant.
    """
    M = np.full((n, n), np.nan, dtype=float)
    ii, jj = np.triu_indices(n, k=1)
    v = np.asarray(vec_1d, dtype=float)
    M[ii, jj] = v
    M[jj, ii] = v
    return M


def build_mode_path_dsr(mode_vec, n_conds_per_config, len_per_bin):
    """fMRI-style DSR: per-bin mode trajectory, flattened and rolled by bin.

    Mirrors the construction in create_fMRI_model_RDMs_on_clean_beh.py
    (EVs['DSR']): take the mode trajectory, downsample to
    n_conds_per_config * len_per_bin integer location IDs, then for each
    bin roll the flattened vector left by ``pos * len_per_bin`` so the
    "current" bin sits at the front. Returned matrix feeds compute_hamming_distance.
    """
    base = downsample_mode(mode_vec, target_len=n_conds_per_config * len_per_bin)
    return np.stack([np.roll(base, -pos * len_per_bin)
                     for pos in range(n_conds_per_config)], axis=0)


def make_empty(rows, cols, dtype=float):
    return np.zeros((rows, cols), dtype=dtype)

def _scalar(arr):
    """Safely extract a Python float from a size-1 array or scalar."""
    return float(np.asarray(arr, dtype=float).ravel()[0])


def _degenerate(x, atol=1e-12):
    """A 1-D vector is degenerate if it has any non-finite entry or zero variance."""
    x = np.asarray(x, dtype=float).ravel()
    return (not np.isfinite(x).all()) or (np.nanvar(x) <= atol)


def eval_tuple(rdm, data_rdm, label=''):
    """Single-model RSA eval. Returns (t, beta, p) as plain floats.

    Returns (NaN, NaN, NaN) when the model or data vector is degenerate
    (constant after masking, contains NaN/inf, etc.) — phase-mask modes
    can make e.g. the 'phase' regressor constant; we surface that with a
    print rather than letting statsmodels error out.
    """
    if _degenerate(rdm):
        print(f"  [skip-single] {label}: model RDM is constant/non-finite "
              f"under current mask → NaN")
        return (float('nan'), float('nan'), float('nan'))
    if _degenerate(data_rdm):
        return (float('nan'), float('nan'), float('nan'))
    try:
        return tuple(_scalar(v) for v in mc.analyse.my_RSA.evaluate_model(
            rdm, data_rdm))
    except Exception as exc:
        print(f"  [skip-single] {label}: {exc!s} → NaN")
        return (float('nan'), float('nan'), float('nan'))


def build_combo_rdm(rdm_dict, combo_list):
    """Stack several model RDMs into one multi-model design matrix."""
    return np.stack([rdm_dict[m][0] for m in combo_list], axis=1)


def evaluate_combo_safe(stacked, data_vec, sub_models, label=''):
    """Combo RSA eval that drops degenerate regressor columns.

    Each column of ``stacked`` corresponds to one ``sub_models`` entry. Any
    column that is non-finite or zero-variance after masking is removed from
    the design before calling evaluate_model; its slot in the returned arrays
    is set to NaN so the caller still sees one (t, beta, p) per sub-model.

    Returns (t_arr, beta_arr, p_arr) as 1-D numpy arrays of length ``len(sub_models)``.
    """
    X = np.asarray(stacked, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n_reg = X.shape[1]
    nan_arr = lambda: np.full(n_reg, np.nan, dtype=float)
    if _degenerate(data_vec):
        return nan_arr(), nan_arr(), nan_arr()

    finite_cols = np.isfinite(X).all(axis=0)
    var_cols = np.nanvar(X, axis=0) > 1e-12
    good = finite_cols & var_cols
    if not good.any():
        print(f"  [skip-combo] {label}: all {n_reg} regressors degenerate "
              f"under current mask → NaN")
        return nan_arr(), nan_arr(), nan_arr()
    if not good.all():
        dropped = [sub_models[i] for i in range(n_reg) if not good[i]]
        print(f"  [drop-combo] {label}: dropping {dropped} "
              f"(constant/non-finite under mask)")

    X_sub = X[:, good]
    try:
        t_sub, beta_sub, p_sub = mc.analyse.my_RSA.evaluate_model(X_sub, data_vec)
    except Exception as exc:
        print(f"  [skip-combo] {label}: {exc!s} → NaN")
        return nan_arr(), nan_arr(), nan_arr()

    t_full, beta_full, p_full = nan_arr(), nan_arr(), nan_arr()
    t_full[good] = np.asarray(t_sub, dtype=float).ravel()
    beta_full[good] = np.asarray(beta_sub, dtype=float).ravel()
    p_full[good] = np.asarray(p_sub, dtype=float).ravel()
    return t_full, beta_full, p_full


STATE_FIG3_COLORS = {
    'A': '#F15A29',
    'B': '#F7931E',
    'C': '#C7C6E2',
    'D': '#6B60AA',
}
PHASE_FIG3_COLORS = {
    'early': '#FCDDE3',
    'middle': '#D7657F',
    'late': '#5C1027',
}
LOCATION_FIG3_COLORS = {
    1: '#0a607a', 2: '#7eb1c4', 3: '#b6d4e0',
    4: '#175e62', 5: '#5b9b8d', 6: '#c8e0d0',
    7: '#0e3d3a', 8: '#3d8b7d', 9: '#a7d9b2',
}
BUTTON_FIG3_COLORS = {
    'stay': '#b8b2a7',
    'up': '#477998',
    'right': '#f2a65a',
    'down': '#8f5d46',
    'left': '#4f772d',
}
DSR_LAG_FIG3_COLORS = [
    '#fff7bc', '#fee391', '#fec44f', '#fe9929',
    '#ec7014', '#cc4c02', '#993404', '#662506',
    '#58151c', '#4a1019', '#3a0b15', '#2b0610',
]
MODEL_FIG3_LABELS = {
    'dsr_fmri': 'DSR\nfull',
    'dsr_fmri_fut': 'DSR\nfuture only',
    'dsr_fmri_informed': 'DSR\ninformed',
    'state': 'Position\nin sequence',
    'state_phase': 'Position x\ngoal progress',
    'phase': 'Goal\nprogress',
    'location': 'Physical\nlocation',
    'l2_norm': 'Grid L2\nlocation',
    'reward_path': 'Reward\npath',
    'repeat_counter': 'Repeat\ncounter',
    'bttn_curr': 'Current\naction',
    'bttn_next': 'Next\naction',
    'midnight': 'DSR\ncurrent only',
    'uncover': 'Uncover\nscreen',
    'curr_quarter':  'DSR\ncurrent state',
    'next_quarter':  'DSR\n+1 state',
    'next2_quarter': 'DSR\n+2 states',
    'next3_quarter': 'DSR\n+3 states',
}


def _fig3_grid_rc(loc_id):
    loc = int(loc_id)
    if loc < 1 or loc > 9:
        raise ValueError(f"location must be in 1..9, got {loc_id!r}")
    return divmod(loc - 1, 3)


def _fig3_grid_loc(row, col):
    row = int(np.clip(row, 0, 2))
    col = int(np.clip(col, 0, 2))
    return row * 3 + col + 1


FIG3_BEH_COLUMNS = [
    'rep_correct', 't_A', 't_B', 't_C', 't_D',
    'loc_A', 'loc_B', 'loc_C', 'loc_D',
    'rep_overall', 'new_grid_onset', 'session_no', 'grid_no', 'correct',
]
FIG3_BUTTON_LABELS = {
    'UpArrow': 'up',
    'RightArrow': 'right',
    'DownArrow': 'down',
    'LeftArrow': 'left',
    'Return': 'space',
}


def _fig3_bad_location_transitions(locs):
    bad = []
    locs = np.asarray(locs)
    for i in range(len(locs) - 1):
        r0, c0 = _fig3_grid_rc(locs[i])
        r1, c1 = _fig3_grid_rc(locs[i + 1])
        if abs(r0 - r1) + abs(c0 - c1) > 1:
            bad.append((i, int(locs[i]), int(locs[i + 1])))
    return bad


def _fig3_cleanup_buttons(buttons_12):
    """Remove Return/space from action panels; keep it for uncover."""
    raw = pd.Series([FIG3_BUTTON_LABELS.get(str(b), 'stay')
                     for b in buttons_12], dtype=object)
    uncover = np.where(raw.eq('space'), 'uncover', 'hidden')
    move = raw.mask(raw.eq('space')).ffill().bfill().fillna('stay')
    return move.to_numpy(dtype=object), uncover.astype(object)


def _fig3_load_single_trial(task_config_str, preferred_repeat=8,
                            n_conds=12):
    """Load one real correct loop and downsample it for fig3 schematics.

    We prefer a late repeat (default rep_correct==8), but a clean 12-bin
    schematic must not imply non-adjacent 3x3-grid moves. If the exact repeat
    has a coarse-bin jump after mode-downsampling, choose the closest valid
    later repeat instead and report the selected trial in the console.
    """
    candidates = []
    subj_dirs = sorted(
        d for d in os.listdir(DATA_DIR)
        if d.startswith('s') and d[1:].isdigit()
    )
    for sub_dir in subj_dirs:
        sub = sub_dir[1:]
        beh_dir = os.path.join(DATA_DIR, sub_dir, 'cells_and_beh')
        beh_path = os.path.join(beh_dir, f'all_trial_times_{sub}.csv')
        loc_path = os.path.join(beh_dir, 'locations.csv')
        btn_path = os.path.join(beh_dir, 'button_presses.csv')
        if not all(os.path.exists(p) for p in (beh_path, loc_path, btn_path)):
            continue

        beh = pd.read_csv(beh_path, header=None)
        beh.columns = FIG3_BEH_COLUMNS
        beh['config_str'] = (
            beh[['loc_A', 'loc_B', 'loc_C', 'loc_D']]
            .astype(int).astype(str).agg('-'.join, axis=1)
        )
        idxs = beh.index[(beh['config_str'] == task_config_str)
                         & (beh['correct'] == 1)]
        if len(idxs) == 0:
            continue

        locs_df = pd.read_csv(loc_path, header=None)
        btn_df = pd.read_csv(btn_path, header=None)
        for idx in idxs:
            loc_360 = locs_df.iloc[idx].to_numpy()
            btn_360 = btn_df.iloc[idx].to_numpy()
            loc_12 = downsample_mode(loc_360, target_len=n_conds).astype(int)
            btn_12 = downsample_mode(btn_360, target_len=n_conds)
            bad = _fig3_bad_location_transitions(loc_12)
            rep = int(beh.at[idx, 'rep_correct'])
            score = (
                len(bad),
                abs(rep - int(preferred_repeat)),
                0 if rep >= int(preferred_repeat) else 1,
                sub,
                int(idx),
            )
            candidates.append({
                'score': score,
                'subject': sub,
                'trial_index': int(idx),
                'grid_no': int(beh.at[idx, 'grid_no']),
                'rep_correct': rep,
                'loc_360': loc_360,
                'button_360': btn_360,
                'loc_12': loc_12,
                'button_12_raw': btn_12,
                'bad_transitions': bad,
            })

    if not candidates:
        raise RuntimeError(
            f"could not find any correct trial for fig3 task {task_config_str}")

    chosen = sorted(candidates, key=lambda c: c['score'])[0]
    buttons_curr, uncover = _fig3_cleanup_buttons(chosen['button_12_raw'])
    chosen['button_curr'] = buttons_curr
    chosen['button_next'] = np.roll(buttons_curr, -1)
    chosen['uncover'] = uncover
    # Reward is reached at the END of a state: the 360-bin warping anchors
    # each state's LAST bin to the reward-arrival time (verified: state A's
    # last bin is loc_A, B's is loc_B, ...). The previous version marked
    # np.arange(0, n_conds, 3) = conds 0/3/6/9, i.e. the EARLY phase of each
    # state — the schematic showed the reward at the start of a state while
    # the real `reward_path` model (build_reward_path_label_360, which walks
    # forward from bins 89/179/269/359) marks the end. Use the real model.
    _rp_360 = build_reward_path_label_360(
        chosen['loc_360'][None, :], chosen['button_360'][None, :],
        task_config_str)
    _w = len(_rp_360) // n_conds
    chosen['reward'] = np.array([
        int(Counter(_rp_360[i * _w:(i + 1) * _w]).most_common(1)[0][0])
        for i in range(n_conds)], dtype=int)

    print("[pub fig] fig3 example trial: "
          f"sub-{chosen['subject']}, trial row {chosen['trial_index']}, "
          f"grid {chosen['grid_no']}, rep_correct={chosen['rep_correct']}, "
          f"config {task_config_str}, loc12={chosen['loc_12'].tolist()}, "
          f"bad_transitions={chosen['bad_transitions']}")
    return chosen


def _fig3_active_spec(model_name, task_config_str, run_config, example_trial):
    n_phases = int(run_config.get('N_PHASES', N_PHASES))
    state_names = list(run_config.get('states', states))
    locs = np.asarray(example_trial['loc_12'], dtype=int)
    n_cond = len(locs)
    state_idx = np.repeat(np.arange(len(state_names)), n_phases)
    phase_idx = np.tile(np.arange(n_phases), len(state_names))
    phase_names = ['early', 'middle', 'late'][:n_phases]

    title = MODEL_FIG3_LABELS.get(model_name, model_name.replace('_', '\n'))

    if model_name in ('dsr_fmri', 'dsr_fmri_lag0123456789'):
        return {'kind': 'dsr', 'title': title, 'lags': list(range(n_cond)),
                'locs': locs}
    if model_name == 'dsr_fmri_fut':
        return {'kind': 'dsr', 'title': title, 'lags': list(range(1, n_cond)),
                'locs': locs}
    if model_name == 'dsr_fmri_informed':
        return {'kind': 'dsr', 'title': title, 'lags': [1, 2], 'locs': locs}
    if model_name == 'dsr_fmri_lag01':
        return {'kind': 'dsr', 'title': title, 'lags': [0, 1], 'locs': locs}
    if model_name == 'dsr_fmri_lag012':
        return {'kind': 'dsr', 'title': title, 'lags': [0, 1, 2], 'locs': locs}
    if model_name == 'dsr_fmri_lag0123':
        return {'kind': 'dsr', 'title': title, 'lags': [0, 1, 2, 3], 'locs': locs}
    if model_name in ('dsr_fmri_123', 'dsr_fmri_fut_123'):
        return {'kind': 'dsr', 'title': title, 'lags': [1, 2, 3], 'locs': locs}
    if model_name == 'dsr_fmri_345':
        return {'kind': 'dsr', 'title': title, 'lags': [3, 4, 5], 'locs': locs}
    if model_name == 'midnight':
        return {'kind': 'dsr', 'title': title, 'lags': [0], 'locs': locs}

    if model_name == 'state':
        rows = state_names
        active = state_idx
        colors = [STATE_FIG3_COLORS[s] for s in rows]
        return {'kind': 'active', 'title': title, 'rows': rows, 'active': active,
                'colors': colors, 'ylabel': 'state'}
    if model_name == 'phase':
        rows = phase_names
        active = phase_idx
        colors = [PHASE_FIG3_COLORS[p] for p in rows]
        return {'kind': 'active', 'title': title, 'rows': rows, 'active': active,
                'colors': colors, 'ylabel': 'phase'}
    if model_name == 'state_phase':
        rows = [f"{s}-{p[0]}" for s in state_names for p in phase_names]
        active = state_idx * n_phases + phase_idx
        colors = [PHASE_FIG3_COLORS[p] for _s in state_names for p in phase_names]
        return {'kind': 'active', 'title': title, 'rows': rows, 'active': active,
                'colors': colors, 'ylabel': 'state x phase'}
    if model_name == 'location':
        rows = [str(i) for i in range(1, 10)]
        active = locs - 1
        colors = [LOCATION_FIG3_COLORS[i] for i in range(1, 10)]
        return {'kind': 'active', 'title': title, 'rows': rows, 'active': active,
                'colors': colors, 'ylabel': 'grid loc'}
    if model_name == 'l2_norm':
        mat = np.stack([l2_norm_row_for_loc(loc) for loc in locs], axis=1)
        return {'kind': 'continuous', 'title': title, 'matrix': mat,
                'rows': [str(i) for i in range(1, 10)],
                'ylabel': 'grid loc', 'cmap': 'YlGnBu_r'}
    if model_name == 'reward_path':
        active = np.asarray(example_trial['reward'], dtype=int)
        return {'kind': 'active', 'title': title, 'rows': ['path', 'reward'],
                'active': active, 'colors': ['#d9d9d9', '#a6611a'],
                'ylabel': 'event'}
    if model_name == 'repeat_counter':
        rows = [f"rep {i + 1}" for i in range(len(state_names))]
        active = state_idx
        colors = ['#f6e8c3', '#dfc27d', '#bf812d', '#8c510a'][:len(rows)]
        return {'kind': 'active', 'title': title, 'rows': rows, 'active': active,
                'colors': colors, 'ylabel': 'repeat'}
    if model_name == 'uncover':
        labels = ['hidden', 'uncover']
        active = np.asarray([labels.index(x) for x in example_trial['uncover']],
                            dtype=int)
        return {'kind': 'active', 'title': title, 'rows': ['hidden', 'uncover'],
                'active': active, 'colors': ['#e6e6e6', '#2166ac'],
                'ylabel': 'screen'}
    if model_name in ('bttn_curr', 'bttn_next', 'bttn_prev'):
        labels = ['up', 'right', 'down', 'left', 'stay']
        if model_name == 'bttn_curr':
            active_labels = example_trial['button_curr']
        elif model_name == 'bttn_next':
            active_labels = example_trial['button_next']
        else:
            active_labels = np.roll(example_trial['button_curr'], 1)
        active = np.asarray([labels.index(x) for x in active_labels], dtype=int)
        colors = [BUTTON_FIG3_COLORS[x] for x in labels]
        return {'kind': 'active', 'title': title, 'rows': labels, 'active': active,
                'colors': colors, 'ylabel': 'button'}

    rows = [model_name]
    return {'kind': 'active', 'title': title, 'rows': rows,
            'active': np.zeros(n_cond, dtype=int), 'colors': ['#969696'],
            'ylabel': 'feature'}


def _fig3_model_order(run_config):
    preferred = [
        'dsr_fmri', 'dsr_fmri_fut', 'dsr_fmri_informed',
        'state', 'state_phase', 'phase',
        'location', 'l2_norm', 'reward_path', 'repeat_counter',
        'bttn_curr', 'bttn_next', 'uncover',
    ]
    legacy_dsr_panels = {
        'dsr', 'dsr_old', 'dsr_old_now_next', 'location_old',
        'midnight',  # old "DSR current only" schematic; not shown for fMRI DSR.
    }
    raw = []
    for sub_models in run_config.get('combo_models', {}).values():
        raw.extend(sub_models)
    raw.extend(run_config.get('models', []))
    raw = [m for m in raw if m not in legacy_dsr_panels]

    seen = set()
    ordered = []
    for model_name in preferred + raw:
        if model_name in raw and model_name not in seen:
            ordered.append(model_name)
            seen.add(model_name)
    return ordered


def _fig3_set_common_axis(ax, n_cols, row_labels, ylabel):
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(len(row_labels) - 0.5, -0.5)
    # Put each state's label where its REWARD actually sits. The 360-bin
    # warping anchors each state's LAST bin to reward arrival, so for a panel
    # with n_cols columns spanning 4 states the reward for state k is the last
    # column of that state's block: (k+1)*n_cols/4 - 1. The previous version
    # used np.arange(1, n_cols, 3), i.e. the MIDDLE of each 3-condition block
    # AND hard-coded a width of 3 columns per state, which silently mislabelled
    # any panel drawn at the model's true resolution (e.g. `location` is 12
    # sub-steps per condition = 36 columns per state, not 3).
    n_states_here = len(states)
    x_ticks = np.array([((k + 1) * n_cols) // n_states_here - 1
                        for k in range(n_states_here)])
    x_ticks = x_ticks[(x_ticks >= 0) & (x_ticks < n_cols)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(states[:len(x_ticks)], fontsize=9, fontname='Arial')
    ax.set_xlabel('task state (label at reward)', fontsize=9,
                  fontname='Arial', labelpad=1)
    ax.set_ylabel(ylabel, fontsize=9, fontname='Arial', labelpad=1)

    if len(row_labels) <= 9:
        ytick_idx = np.arange(len(row_labels))
    else:
        ytick_idx = np.linspace(0, len(row_labels) - 1, 5).round().astype(int)
    ax.set_yticks(ytick_idx)
    ax.set_yticklabels([row_labels[i] for i in ytick_idx],
                       fontsize=9, fontname='Arial')
    # White cell separators only while the cells are wide enough to survive
    # them. At the model's true resolution (144 sub-steps) a 0.5 pt white line
    # per column is wider than the column itself, which is what made the
    # `location` panel look mostly white and illegible.
    if n_cols <= 24:
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    if len(row_labels) <= 24:
        ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    if n_cols <= 24 or len(row_labels) <= 24:
        ax.grid(which='minor', color='white', linewidth=0.5)
    ax.tick_params(axis='both', which='major', length=2, pad=1)
    ax.tick_params(axis='both', which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color('#333333')


def _fig3_draw_active(ax, spec):
    import matplotlib.colors as mcolors

    active = np.asarray(spec['active'], dtype=int)
    row_labels = list(spec['rows'])
    n_rows, n_cols = len(row_labels), len(active)
    off_rgb = np.array([0.965, 0.965, 0.965, 1.0])
    img = np.tile(off_rgb, (n_rows, n_cols, 1))
    for col_i, row_i in enumerate(active):
        if 0 <= row_i < n_rows:
            rgba = mcolors.to_rgba(spec['colors'][row_i])
            img[row_i, col_i, :] = rgba
    ax.imshow(img, interpolation='nearest', aspect='auto')
    _fig3_set_common_axis(ax, n_cols, row_labels, spec.get('ylabel', 'feature'))


def _fig3_draw_continuous(ax, spec):
    mat = np.asarray(spec['matrix'], dtype=float)
    ax.imshow(mat, interpolation='nearest', aspect='auto',
              cmap=spec.get('cmap', 'Greys'))
    _fig3_set_common_axis(ax, mat.shape[1], list(spec['rows']),
                          spec.get('ylabel', 'feature'))


def _fig3_draw_dsr(ax, spec):
    from matplotlib.patches import Rectangle

    locs = np.asarray(spec['locs'], dtype=int)
    lags = list(spec['lags'])
    n_cols = len(locs)
    for row_i, lag in enumerate(lags):
        lag_color = DSR_LAG_FIG3_COLORS[min(lag, len(DSR_LAG_FIG3_COLORS) - 1)]
        for col_i in range(n_cols):
            future_loc = int(locs[(col_i + lag) % n_cols])
            ax.add_patch(Rectangle((col_i - 0.5, row_i - 0.5), 1.0, 1.0,
                                   facecolor=lag_color, edgecolor='white',
                                   linewidth=0.5))
            ax.add_patch(Rectangle((col_i - 0.25, row_i - 0.25), 0.5, 0.5,
                                   facecolor=LOCATION_FIG3_COLORS[future_loc],
                                   edgecolor='none'))
    labels = ['now' if lag == 0 else f'+{lag}' for lag in lags]
    _fig3_set_common_axis(ax, n_cols, labels, 'future lag')


def _save_pub_fig3_model_schematics(run_dir, run_config,
                                    task_config_str=None,
                                    matrix_cm=REPLOT_PUB_FIG3_MATRIX_CM,
                                    max_cols=4):
    """Save human RSA fig3 model schematics from a saved run configuration."""
    task_config_str = task_config_str or run_config.get('configs', configs)[0]
    example_trial = _fig3_load_single_trial(
        task_config_str=task_config_str,
        preferred_repeat=REPLOT_PUB_FIG3_EXAMPLE_REPEAT,
        n_conds=int(run_config.get('N_CONDS_PER_CONF', N_CONDS_PER_CONF)),
    )
    model_order = _fig3_model_order(run_config)
    specs = [_fig3_active_spec(m, task_config_str, run_config, example_trial)
             for m in model_order]

    matrix_in = matrix_cm / 2.54
    left_label_in = 0.54
    right_gap_in = 0.28
    bottom_label_in = 0.38
    top_title_in = 0.42
    panel_w = left_label_in + matrix_in + right_gap_in
    panel_h = bottom_label_in + matrix_in + top_title_in
    n_cols = min(max_cols, max(1, len(specs)))
    n_rows = int(np.ceil(len(specs) / n_cols))
    fig_w = n_cols * panel_w
    fig_h = n_rows * panel_h

    figs_dir = os.path.join(run_dir, 'pub_figures')
    os.makedirs(figs_dir, exist_ok=True)
    save_stem = os.path.join(figs_dir, 'fig3_human_model_schematics')

    with plt.rc_context({
        'font.family': 'Arial',
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }):
        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
        for panel_i, spec in enumerate(specs):
            row_i = panel_i // n_cols
            col_i = panel_i % n_cols
            x0 = col_i * panel_w + left_label_in
            y0 = fig_h - (row_i + 1) * panel_h + bottom_label_in
            ax = fig.add_axes([
                x0 / fig_w, y0 / fig_h,
                matrix_in / fig_w, matrix_in / fig_h,
            ])
            if spec['kind'] == 'dsr':
                _fig3_draw_dsr(ax, spec)
            elif spec['kind'] == 'continuous':
                _fig3_draw_continuous(ax, spec)
            else:
                _fig3_draw_active(ax, spec)
            ax.set_title(spec['title'], fontsize=11, fontname='Arial',
                         fontweight='normal', pad=4)

        fig.savefig(save_stem + '.pdf', dpi=300)
        fig.savefig(save_stem + '.jpg', dpi=300)
        plt.close(fig)

    print(f"[pub fig] wrote {save_stem}.pdf/.jpg "
          f"({len(specs)} models, task {task_config_str}, "
          f"{matrix_cm:g} cm matrices).")


def _pick_smoothest_config(z, cfgs, n_cond, len_path, verbose=True):
    """Pick the example task whose mode path reads most cleanly.

    The mode path is real behaviour, so some configs have states walked in 2
    steps and others in 5, with runs only a few sub-steps wide. On a schematic
    those slivers look like noise and invite questions about a nuance the
    figure is not making. We therefore pick the config with (a) the smallest
    spread in steps-per-state and (b) the widest thinnest step, and print the
    full ranking so the choice is reproducible and auditable.

    This selects an ILLUSTRATION only. No model, RDM or statistic depends on
    it -- every config enters the RSA identically.
    """
    loc = np.asarray(z['location'], dtype=int)
    scored = []
    for ci, c in enumerate(cfgs):
        blk = loc[ci * n_cond:(ci + 1) * n_cond].reshape(-1)
        per_state = len(blk) // len(states)
        n_steps, min_run = [], []
        for k in range(len(states)):
            w = blk[k * per_state:(k + 1) * per_state]
            runs = []
            for x in w:
                if not runs or runs[-1][0] != x:
                    runs.append([int(x), 1])
                else:
                    runs[-1][1] += 1
            n_steps.append(len(runs))
            min_run.append(min(b for _, b in runs))
        spread = max(n_steps) - min(n_steps)
        thinnest = min(min_run)
        scored.append(((spread, -thinnest), c, n_steps, thinnest))
    scored.sort()
    if verbose:
        print("[pub fig] example-task ranking (spread in steps/state, then "
              "widest thinnest step; illustration only):")
        for (sp, negt), c, ns, th in scored:
            print(f"           {c}  steps/state {ns}  spread {sp}  "
                  f"thinnest step {th}")
        print(f"[pub fig] chose {scored[0][1]}")
    return scored[0][1]


def _load_saved_design_matrices(run_dir):
    """Load the per-condition model design matrices the run actually fitted.

    Saved by the main run as `model_design_matrices/model_design_matrices.npz`.
    Rows are [half-1 conditions; half-2 conditions], each half
    n_configs * n_conds_per_conf rows.
    """
    f = os.path.join(run_dir, 'model_design_matrices',
                     'model_design_matrices.npz')
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"no saved design matrices at {f} -- a run must complete the model-"
            f"build stage first. Refusing to re-derive them: re-deriving from a "
            f"single trial is what made the earlier schematic disagree with the "
            f"model (the model uses the across-subject MODE path).")
    z = np.load(f, allow_pickle=True)
    return (z, int(z['__n_configs__']), int(z['__n_conds_per_conf__']),
            int(z['__len_standardised_path__']),
            [str(c) for c in z['__configs__']])


def _fig3_spec_from_design(model_name, z, cfg_i, n_cond, len_path):
    """Build a schematic spec straight from the model's own design matrix.

    Every panel is put on ONE common x-axis of `n_cond * len_path` sub-steps,
    so the same column means the same moment in the task in every panel and
    the A/B/C/D labels (drawn at each state's reward) line up across models.
    Models carrying one value per condition are expanded by repeating it
    `len_path` times; `location` and `reward_path` already have `len_path`
    sub-steps per condition and are used as-is.
    """
    blk = slice(cfg_i * n_cond, (cfg_i + 1) * n_cond)   # half-1 block
    title = MODEL_FIG3_LABELS.get(model_name, model_name)
    if model_name not in z.files:
        # Archive predates this model being added to model_concat. Skip the
        # panel rather than substituting a re-derived stand-in.
        return None

    if model_name == 'location':
        locs = np.asarray(z['location'][blk], dtype=int).reshape(-1)
        return {'kind': 'active', 'title': title,
                'rows': [str(i) for i in range(1, 10)], 'active': locs - 1,
                'colors': [LOCATION_FIG3_COLORS[i] for i in range(1, 10)],
                'ylabel': 'grid loc'}

    if model_name == 'reward_path':
        rp = np.asarray(z['reward_path'][blk], dtype=int).reshape(-1)
        return {'kind': 'active', 'title': title,
                'rows': ['path', 'reward'], 'active': rp,
                'colors': ['#d9d9d9', '#a6611a'], 'ylabel': 'event'}

    if model_name in ('state', 'phase', 'state_phase'):
        # Column order verified against the saved design matrices: `state`
        # gives argmax 0,0,0,1,1,1,... `phase` gives 0,1,2,0,1,2,... and
        # `state_phase` gives 0..11, i.e. state-major / phase-minor. Row
        # labels below follow that order.
        M = np.asarray(z[model_name][blk], dtype=float)      # (n_cond, k)
        active = np.repeat(M.argmax(axis=1), len_path)
        phase_names = ['early', 'middle', 'late'][:N_PHASES]
        if model_name == 'state':
            rows = list(states)
            colors = [STATE_FIG3_COLORS[r] for r in rows]
            ylabel = 'state'
        elif model_name == 'phase':
            rows = list(phase_names)
            colors = [PHASE_FIG3_COLORS[ph] for ph in phase_names]
            ylabel = 'goal progress'
        else:
            rows = [f"{st}-{ph[0]}" for st in states for ph in phase_names]
            colors = [PHASE_FIG3_COLORS[ph] for _st in states
                      for ph in phase_names]
            ylabel = 'state x phase'
        return {'kind': 'active', 'title': title, 'rows': rows,
                'active': active, 'colors': colors, 'ylabel': ylabel}

    if model_name == 'l2_norm':
        M = np.asarray(z['l2_norm'][blk], dtype=float)       # (n_cond, 9)
        mat = np.repeat(M, len_path, axis=0).T               # (9, n_cond*len)
        return {'kind': 'continuous', 'title': title, 'matrix': mat,
                'rows': [str(i) for i in range(1, 10)],
                'ylabel': 'grid loc', 'cmap': 'YlGnBu_r'}

    if model_name.startswith('dsr') or model_name == 'midnight':
        # DSR is (n_cond x n_cond*len_path): every condition carries the WHOLE
        # rolled trajectory. Draw it as the matrix it is, conditions on y.
        M = np.asarray(z[model_name][blk], dtype=float)
        return {'kind': 'continuous', 'title': title, 'matrix': M,
                'rows': [f'c{i}' for i in range(M.shape[0])],
                'ylabel': 'condition', 'cmap': 'YlGnBu_r'}

    return None


def _load_saved_model_rdms(run_dir, test_variant='split_halves'):
    """Load the model RDMs a run actually regressed, from `rdms/rdms_*.npz`.

    These are the regression's OWN inputs, so a figure drawn from them cannot
    drift from the model that was fitted -- nothing is re-derived here. The
    model RDMs are behaviour-derived and therefore ROI-independent (verified
    identical across all saved ROI archives), so the first archive is used and
    its ROI is reported.

    Returns (rdm_dict, n_configs, n_conds_per_conf, configs, roi_used) with
    each value a square (n x n) matrix, n = n_configs * n_conds_per_conf.
    """
    rdm_dir = os.path.join(run_dir, 'rdms')
    archives = sorted(glob.glob(os.path.join(rdm_dir, 'rdms_*.npz')))
    if not archives:
        raise FileNotFoundError(
            f"no saved model RDMs in {rdm_dir} -- the run must reach the RDM "
            f"stage before the RDM panels can be drawn from real data.")
    z = np.load(archives[0], allow_pickle=True)
    n_conf = int(z['__n_configs__'])
    n_cond = int(z['__n_conds_per_conf__'])
    n = n_conf * n_cond
    prefix = f'model__{test_variant}__'
    out = {}
    for key in z.keys():
        if key.startswith(prefix):
            out[key[len(prefix):]] = _vec_to_square_rdm(z[key], n)
    return (out, n_conf, n_cond, [str(c) for c in z['__configs__']],
            str(z['__roi__']))


def _draw_rdm(ax, M, title, n_cond=None, n_conf=None, kind='single'):
    """Draw one model RDM. `M` is the real matrix; nothing is rescaled.

    kind='single'  one task: tick each state's REWARD, i.e. the last condition
                   of each state block, matching the schematics' convention.
    kind='multi'   all tasks: mask the upper triangle (an RDM is symmetric, so
                   showing both halves doubles ink for no information) and tick
                   the centre of each task block.
    """
    M = np.array(M, dtype=float)
    if kind == 'multi':
        M[np.triu_indices_from(M, k=1)] = np.nan
    finite = M[np.isfinite(M)]
    if finite.size == 0:
        ax.set_axis_off(); return None
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('white')
    im = ax.imshow(np.ma.masked_invalid(M), interpolation='nearest',
                   aspect='equal', cmap=cmap,
                   vmin=float(np.nanmin(finite)), vmax=float(np.nanmax(finite)))
    if kind == 'single' and n_cond:
        per_state = n_cond // len(states)
        ticks = [(k + 1) * per_state - 1 for k in range(len(states))]
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels(states, fontsize=7, fontname='Arial')
        ax.set_yticklabels(states, fontsize=7, fontname='Arial')
    elif kind == 'multi' and n_cond and n_conf:
        ticks = [i * n_cond + n_cond / 2.0 - 0.5 for i in range(n_conf)]
        labs = [f'{i + 1}' for i in range(n_conf)]
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels(labs, fontsize=6, fontname='Arial')
        ax.set_yticklabels(labs, fontsize=6, fontname='Arial')
        ax.set_xlabel('task', fontsize=7, fontname='Arial', labelpad=1)
        ax.set_ylabel('task', fontsize=7, fontname='Arial', labelpad=1)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(axis='both', length=2, pad=1)
    ax.set_title(title, fontsize=8, fontname='Arial', pad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7); sp.set_color('#333333')
    return im


def _save_pub_fig3_models_with_rdms(run_dir, run_config, task_config_str=None,
                                     matrix_cm=REPLOT_PUB_FIG3_MATRIX_CM):
    """Methods figure: per model, its schematic + single-task + multi-task RDM.

    Every panel comes from real data: the schematic from one real correct loop,
    the two RDMs from the run's own saved model RDMs (see
    `_load_saved_model_rdms`). The single-task RDM is the diagonal block of the
    chosen config; the multi-task RDM is the full conditions x conditions
    matrix with white lines at the config boundaries.
    """
    rdms, n_conf, n_cond, cfg_list, roi_used = _load_saved_model_rdms(run_dir)
    task_config_str = task_config_str or run_config.get('configs', configs)[0]
    cfg_i = cfg_list.index(task_config_str) if task_config_str in cfg_list else 0
    blk = slice(cfg_i * n_cond, (cfg_i + 1) * n_cond)

    # Schematics come from the run's OWN design matrices, at the model's true
    # resolution, on one shared sub-step x-axis. No re-derivation from a single
    # trial: the model uses the across-subject mode path, and the two disagree.
    dz, _dn_conf, dn_cond, dlen, dcfgs = _load_saved_design_matrices(run_dir)
    if REPLOT_PUB_FIG3_EXAMPLE_CONFIG is None:
        task_config_str = _pick_smoothest_config(dz, dcfgs, dn_cond, dlen)
    else:
        task_config_str = REPLOT_PUB_FIG3_EXAMPLE_CONFIG
    cfg_i = cfg_list.index(task_config_str) if task_config_str in cfg_list else 0
    blk = slice(cfg_i * n_cond, (cfg_i + 1) * n_cond)
    d_cfg_i = dcfgs.index(task_config_str) if task_config_str in dcfgs else 0
    model_order = [m for m in _fig3_model_order(run_config) if m in rdms]
    specs = []
    for m in model_order:
        spec = _fig3_spec_from_design(m, dz, d_cfg_i, dn_cond, dlen)
        if spec is not None:
            specs.append((m, spec))
        else:
            print(f"[pub fig] no design-matrix panel for {m!r}; omitted "
                  f"rather than drawn from a re-derived source.")

    mi = matrix_cm / 2.54
    left_in, gap_in, top_in, bot_in = 0.62, 0.30, 0.34, 0.42
    col_w = [left_in + mi * 1.6 + gap_in, left_in * 0.4 + mi + gap_in,
             left_in * 0.4 + mi + gap_in]
    row_h = mi + top_in + bot_in
    fig_w, fig_h = sum(col_w), row_h * len(specs)

    figs_dir = os.path.join(run_dir, 'pub_figures')
    os.makedirs(figs_dir, exist_ok=True)
    stem = os.path.join(figs_dir, 'fig3_models_schematic_and_RDMs')

    with plt.rc_context({'font.family': 'Arial', 'font.size': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig = plt.figure(figsize=(fig_w, fig_h))
        for r, (mname, spec) in enumerate(specs):
            y0 = fig_h - (r + 1) * row_h + bot_in
            # --- column 0: model schematic (real example loop) ---
            ax = fig.add_axes([left_in / fig_w, y0 / fig_h,
                               (mi * 1.6) / fig_w, mi / fig_h])
            if spec['kind'] == 'dsr':
                _fig3_draw_dsr(ax, spec)
            elif spec['kind'] == 'continuous':
                _fig3_draw_continuous(ax, spec)
            else:
                _fig3_draw_active(ax, spec)
            ax.set_title(spec['title'].replace('\n', ' '), fontsize=9,
                         fontname='Arial', pad=3)
            # --- columns 1-2: the run's OWN model RDMs ---
            x1 = (col_w[0] + left_in * 0.4) / fig_w
            ax1 = fig.add_axes([x1, y0 / fig_h, mi / fig_w, mi / fig_h])
            im1 = _draw_rdm(ax1, rdms[mname][blk, blk],
                            f'1 task ({task_config_str})',
                            n_cond=n_cond, kind='single')
            x2 = (col_w[0] + col_w[1] + left_in * 0.4) / fig_w
            ax2 = fig.add_axes([x2, y0 / fig_h, mi / fig_w, mi / fig_h])
            im2 = _draw_rdm(ax2, rdms[mname], f'{n_conf} tasks',
                            n_cond=n_cond, n_conf=n_conf, kind='multi')
            if im2 is not None:
                cb = fig.colorbar(im2, ax=ax2, fraction=0.05, pad=0.04)
                cb.ax.tick_params(labelsize=7, length=2)
                cb.outline.set_linewidth(0.5)

        fig.savefig(stem + '.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(stem + '.jpg', dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f"[pub fig] wrote {stem}.pdf/.jpg ({len(specs)} models; RDMs from "
          f"{run_dir}/rdms [ROI {roi_used}, model RDMs are ROI-independent]; "
          f"single-task block = {task_config_str}).")


def _replot_saved_pub_fig3_only():
    for run_name in REPLOT_PUB_FIG3_RUNS:
        run_dir = os.path.join(OUT_BASE, run_name)
        cfg_path = os.path.join(run_dir, 'config.json')
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"saved RSA run config not found: {cfg_path}")
        with open(cfg_path, 'r') as f:
            run_config = json.load(f)
        _save_pub_fig3_model_schematics(
            run_dir=run_dir,
            run_config=run_config,
            task_config_str=run_config.get('configs', configs)[0],
            max_cols=REPLOT_PUB_FIG3_MAX_COLS,
        )
        try:
            _save_pub_fig3_models_with_rdms(
                run_dir=run_dir,
                run_config=run_config,
                task_config_str=run_config.get('configs', configs)[0],
            )
        except FileNotFoundError as exc:
            print(f"[pub fig] RDM panels skipped: {exc}")


if REPLOT_PUB_FIG3_ONLY:
    _replot_saved_pub_fig3_only()
    sys.exit(0)


with open(os.path.join(DATA_DIR, 'all_sessions_dsrRSA_grouping_summary.json'), 'r') as f:
    config_summary = json.load(f)
SUBJECTS = list(config_summary.keys())

# ── Per-run output folder (timestamped) + config dump ────────────────
if RELOAD_RUN is not None:
    RUN_TAG = RELOAD_RUN
    OUT_DIR = os.path.join(OUT_BASE, RELOAD_RUN)
    if not os.path.isdir(OUT_DIR):
        raise FileNotFoundError(f"Reload folder not found: {OUT_DIR}")
    print(f"RELOAD MODE — reading results from: {OUT_DIR}")
else:
    RUN_TAG = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    OUT_DIR = os.path.join(OUT_BASE, RUN_TAG)
    os.makedirs(OUT_BASE, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    run_config = {
        'run_tag':              RUN_TAG,
        'timestamp':            datetime.now().isoformat(timespec='seconds'),
        'data_dir':             DATA_DIR,
        'out_dir':              OUT_DIR,
        'configs':              configs,
        'N_CONFIGS':            N_CONFIGS,
        'N_CONDS_PER_CONF':     N_CONDS_PER_CONF,
        'LEN_STANDARDISED_PATH': LEN_STANDARDISED_PATH,
        'N_PHASES':             N_PHASES,
        'states':               states,
        'RESOLUTIONx':          RESOLUTIONx,
        'N_PERMUTATIONS':       N_PERMUTATIONS,
        'SHUFFLE_MODE':         SHUFFLE_MODE,
        'models':               models,
        'combo_models':         combo_models,
        'roi_label_column':     ROI_LABEL_COLUMN,
        'phase_residualise':    PHASE_RESIDUALISE,
        'residualise_repeats':  RESIDUALISE_REPEATS,
        'fdr_test':             FDR_TEST,
        'fdr_combos':           FDR_COMBOS,
        'fdr_submodels':        FDR_SUBMODELS,
        'fdr_alpha':            FDR_ALPHA,
        'phase_mask_mode':      PHASE_MASK_MODE,
        'run_phase_mode_comparison': RUN_PHASE_MODE_COMPARISON,
        'rois':                 list(ROI_RULES.keys()),
    }
    with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f"Run output: {OUT_DIR}")


if RELOAD_RUN is None:
    # Cache subject data once; loop ROIs over the cache afterwards.
    print("Loading subject data once for all ROIs...")
    if RESIDUALISE_REPEATS:
        print("  RESIDUALISE_REPEATS=True → loading per-cell rep_correct-residualised files")
    SUBJECT_DATA = {}
    for sub_str in SUBJECTS:
        SUBJECT_DATA[sub_str] = mc.analyse.helpers_human_cells.load_norm_data(
            DATA_DIR, [sub_str], res_data=RESIDUALISE_REPEATS,
        )
    print(f"Cached data for {len(SUBJECT_DATA)} subjects.")

    # ── Per-subject grouping logs ────────────────────────────────────────
    # ``s<sub>_dsr_grouping_log_two_runs.json`` holds, per config, the
    # non-spilling block-level allocation that the dsr-avg pipeline produced:
    # each ``grid_no`` (= one task block of ~10 trials) is assigned to either
    # run1_blocks or run2_blocks, and the partition is chosen to even out
    # trial counts as much as possible while respecting block boundaries.
    # Until now this script ignored the log and hard-sliced correct trials at
    # index [0:10] / [10:20], which (a) ignored block boundaries and (b) gave
    # different effective splits per subject × config depending on row order.
    # Load the logs once and consult them in the ROI loop.
    GROUPING_LOGS = {}
    for sub_str in SUBJECTS:
        gpath = os.path.join(
            DATA_DIR, f's{sub_str}', 'dsr_avg',
            f's{sub_str}_dsr_grouping_log_two_runs.json')
        if not os.path.exists(gpath):
            print(f"  [grouping] s{sub_str}: log not found at {gpath} — "
                  f"subject will be skipped per ROI when its configs are needed.")
            GROUPING_LOGS[sub_str] = {}
            continue
        with open(gpath) as f:
            _g = json.load(f)
        GROUPING_LOGS[sub_str] = {
            c['config']: c for c in _g.get('configs', [])
        }
    print(f"Loaded grouping logs for {len(GROUPING_LOGS)} subjects.")

    # ── Run-1 / run-2 separation + balance diagnostic ─────────────────────
    # Trial allocation rule (per subject × config): correct trials are
    # bucketed by grid_no into run1_blocks / run2_blocks from the grouping
    # log. The partition is non-spilling (each grid_no ∈ exactly one half)
    # and chosen by the dsr-avg pipeline to even out totals as much as block
    # boundaries allow.
    #
    # Balance philosophy: with 3-block configs the per-config split is
    # naturally 2+1 or 1+2 — this is EXPECTED and not a problem. What we
    # want to avoid is *systematic* over-loading of one half ACROSS configs
    # (e.g. always 2 blocks in run-1 and 1 in run-2 for every config of a
    # subject). The diagnostic therefore reports per-SUBJECT totals across
    # all configs, plus an "alternating-balance score". No cells are excluded.
    print("\n=== Run-1 / run-2 separation diagnostic ===")
    print("  Allocation: grouping-log blocks per config. Per-config 2+1 or "
          "1+2 splits are expected with 3 runs; what matters is that any "
          "over-loading is NOT systematic across configs within a subject.")
    diag_rows = []
    n_missing = 0
    for sub_str, sub_pack in SUBJECT_DATA.items():
        beh = sub_pack[f"sub-{sub_str}"]['beh'].copy()
        beh['config_str'] = (
            beh['loc_A'].astype(int).astype(str) + '-' +
            beh['loc_B'].astype(int).astype(str) + '-' +
            beh['loc_C'].astype(int).astype(str) + '-' +
            beh['loc_D'].astype(int).astype(str))
        beh['grid_no'] = beh['grid_no'].astype(int)
        glog = GROUPING_LOGS.get(sub_str, {})
        for c in configs:
            n_corr_total = int(((beh['config_str'] == c)
                                & (beh['correct'] == 1)).sum())
            cfg_entry = glog.get(c)
            if cfg_entry is None:
                n_missing += 1
                diag_rows.append({
                    'subject':         sub_str,
                    'config':          c,
                    'n_correct':       n_corr_total,
                    'n_run1':          0,
                    'n_run2':          0,
                    'n_blocks_run1':   0,
                    'n_blocks_run2':   0,
                    'run1_blocks':     '',
                    'run2_blocks':     '',
                    'log_present':     False,
                })
                continue
            run1_blocks = cfg_entry['run1_blocks']
            run2_blocks = cfg_entry['run2_blocks']
            n_run1 = int(((beh['config_str'] == c)
                          & (beh['correct'] == 1)
                          & beh['grid_no'].isin(run1_blocks)).sum())
            n_run2 = int(((beh['config_str'] == c)
                          & (beh['correct'] == 1)
                          & beh['grid_no'].isin(run2_blocks)).sum())
            diag_rows.append({
                'subject':         sub_str,
                'config':          c,
                'n_correct':       n_corr_total,
                'n_run1':          n_run1,
                'n_run2':          n_run2,
                'n_blocks_run1':   len(run1_blocks),
                'n_blocks_run2':   len(run2_blocks),
                'run1_blocks':     ';'.join(map(str, run1_blocks)),
                'run2_blocks':     ';'.join(map(str, run2_blocks)),
                'log_present':     True,
            })
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(os.path.join(OUT_DIR, 'run_balance_diagnostic.csv'), index=False)
    print(f"  cells missing grouping log entry: {n_missing}")
    print(f"  trial-overlap risk: 0 — every grid_no is in exactly one of "
          f"run1_blocks / run2_blocks (verified by block-partition construction).")

    # Per-subject totals + systematic-imbalance flag.
    # Definitions:
    #   total_run1   = sum of n_run1 across all configs of a subject
    #   total_run2   = sum of n_run2 across all configs of a subject
    #   balance_pct  = 100 * (1 - |total_run1 - total_run2| / total)
    #                  100% = perfectly balanced, 0% = all in one half.
    #   alternation  = fraction of configs that flipped which half had more
    #                  trials vs. the running cumulative excess. If the log
    #                  is genuinely alternating, this is near 1.0; if it
    #                  systematically over-loads one side, it drops toward 0.
    print("\n  Per-subject totals across configs (block-respecting allocation):")
    print(f"  {'sub':>4s}  {'n_run1':>7s}  {'n_run2':>7s}  "
          f"{'balance':>8s}  {'alternation':>11s}")
    n_subjects_unbalanced = 0
    n_subjects_systematic = 0
    sys_threshold_balance     = 70.0   # below this -> flag as unbalanced
    sys_threshold_alternation = 0.5    # below this -> flag as systematic
    summary_subject_rows = []
    for sub_str in diag_df['subject'].unique():
        sub_df = diag_df[(diag_df['subject'] == sub_str)
                         & (diag_df['log_present'])].sort_values('config')
        if sub_df.empty:
            continue
        t1 = int(sub_df['n_run1'].sum())
        t2 = int(sub_df['n_run2'].sum())
        total = max(t1 + t2, 1)
        balance_pct = 100.0 * (1.0 - abs(t1 - t2) / total)
        # alternation: how often does (n_run1 - n_run2) cross zero across
        # successive configs? Compute as fraction of non-zero diffs whose
        # sign differs from the running cumulative.
        diffs = (sub_df['n_run1'] - sub_df['n_run2']).to_numpy()
        nonzero = diffs[diffs != 0]
        if len(nonzero) <= 1:
            alternation = 1.0
        else:
            cum = np.cumsum(nonzero)
            # alternation reward: count configs whose diff opposes the cum sign
            opposes = np.sign(nonzero[1:]) != np.sign(cum[:-1])
            alternation = float(opposes.mean()) if len(opposes) else 1.0
        is_unbalanced = balance_pct < sys_threshold_balance
        is_systematic = (balance_pct < sys_threshold_balance
                         and alternation < sys_threshold_alternation)
        if is_unbalanced:
            n_subjects_unbalanced += 1
        if is_systematic:
            n_subjects_systematic += 1
        flag = ('  ⚠ SYSTEMATIC' if is_systematic
                else ('  ⚠ unbalanced' if is_unbalanced else ''))
        print(f"  {sub_str:>4s}  {t1:>7d}  {t2:>7d}  "
              f"{balance_pct:>7.1f}%  {alternation:>10.2f}{flag}")
        summary_subject_rows.append({
            'subject': sub_str,
            'total_run1': t1, 'total_run2': t2,
            'balance_pct': balance_pct,
            'alternation_score': alternation,
            'flag_unbalanced': is_unbalanced,
            'flag_systematic': is_systematic,
        })
    pd.DataFrame(summary_subject_rows).to_csv(
        os.path.join(OUT_DIR, 'run_balance_summary_by_subject.csv'), index=False)
    print(f"  -> {n_subjects_unbalanced} subjects flagged as unbalanced "
          f"(<{sys_threshold_balance:.0f}% balance), "
          f"{n_subjects_systematic} of those also systematic "
          f"(alternation <{sys_threshold_alternation:.2f}).")
    print(f"  No cells are excluded; flags are advisory.")
    print(f"  per-subject × config breakdown   -> "
          f"{os.path.join(OUT_DIR, 'run_balance_diagnostic.csv')}")
    print(f"  per-subject totals + flags       -> "
          f"{os.path.join(OUT_DIR, 'run_balance_summary_by_subject.csv')}")

    # Optional per-cell phase residualisation on raw 360-bin firing rate.
    # Applied to data ONLY (not models). Each cell's mean firing rate is
    # preserved; the within-state phase tuning component is subtracted.
    if PHASE_RESIDUALISE:
        from mc.analyse.future_spatial_peaks import phase_residualise as _residualise_phase
        n_cells_total = 0
        for sub_str, sub_pack in SUBJECT_DATA.items():
            neurons = sub_pack[f"sub-{sub_str}"]['normalised_neurons']
            for n_lab, n_df in neurons.items():
                arr_clean = _residualise_phase(
                    n_df.to_numpy(dtype=float), basis=PHASE_RESIDUALISE,
                )
                neurons[n_lab] = pd.DataFrame(
                    arr_clean, index=n_df.index, columns=n_df.columns,
                )
                n_cells_total += 1
        print(f"Applied phase residualisation ({PHASE_RESIDUALISE}) to "
              f"{n_cells_total} cells across {len(SUBJECT_DATA)} subjects.")


    # Containers for cross-ROI summary
    summary_rows = []
    summary_combo_rows = []
    roi_perm_results = {}
    roi_empirical_results = {}
    # mode -> roi -> test_name -> {raw_combo, z_combo, n_pairs_kept, n_pairs_total}
    # Populated only when RUN_PHASE_MODE_COMPARISON; used by the cross-ROI
    # phase-mode comparison plot at the end of the run.
    roi_mode_comparison = {}
    # Per-ROI electrode coordinates (deduplicated by (sub, cell_idx)) for the
    # schematic glass-brain shown alongside the overview heatmaps.
    roi_electrode_coords = {roi: {} for roi in ROI_RULES}

    # ── Phase-mask diagnostic (once, before ROI loop) ────────────────────
    # Print phase per condition + mask sizes per (variant, mode) so we can
    # verify which RDM cells are kept vs. excluded by each masking mode.
    # Also save a visual of the three mask matrices (96×96 each).
    print("\n=== Phase-mask diagnostic ===")
    print(f"Phase per condition (length {PHASE_MASKS['_n']}) — "
          f"{N_CONFIGS} configs × {N_CONDS_PER_CONF} conds each, "
          f"phase = (pos % {N_CONDS_PER_CONF}) % {N_PHASES} → cycles 0,1,2 = "
          f"early,middle,late within each state's sub-path.")
    _phase_vec = PHASE_MASKS['_phase_per_condition']
    for c_idx in range(N_CONFIGS):
        seg = _phase_vec[c_idx * N_CONDS_PER_CONF:(c_idx + 1) * N_CONDS_PER_CONF]
        print(f"  config {c_idx} ({configs[c_idx]:>8s}): {seg.tolist()}")
    print(f"\nMask sizes per RDM variant × mode "
          f"(kept = pairs entering evaluate_model):")
    print(f"  {'variant':<16s} {'mode':<14s} {'kept':>7s} / {'total':>7s} "
          f"{'(% kept)':>10s}")
    for _variant in ('split_halves', 'between_tasks'):
        for _mode in ALL_PHASE_MODES:
            _mask = PHASE_MASKS[_variant][_mode]
            _k, _t = int(_mask.sum()), int(_mask.size)
            print(f"  {_variant:<16s} {_mode:<14s} {_k:>7d} / {_t:>7d} "
                  f"{100.0*_k/_t:>9.1f}%")

    # Plot the three 96×96 mask matrices once.
    from mc.plotting.cell_results import plot_phase_mask_diagnostic
    plot_phase_mask_diagnostic(
        mask_matrices={
            m: _phase_mask_matrix(m, N_CONFIGS, N_CONDS_PER_CONF, N_PHASES)
            for m in ALL_PHASE_MODES
        },
        n_configs=N_CONFIGS, n_conds_per_config=N_CONDS_PER_CONF,
        save_path=os.path.join(OUT_DIR, 'phase_mask_diagnostic.png'),
        suptitle='Phase masks (96×96) — red lines = config boundaries',
    )
    print(f"Saved phase-mask diagnostic figure to "
          f"{os.path.join(OUT_DIR, 'phase_mask_diagnostic.png')}")
    plt.show()
    print(f"\nPrimary pipeline runs with PHASE_MASK_MODE = '{PHASE_MASK_MODE}'.")
    print(f"RUN_PHASE_MODE_COMPARISON = {RUN_PHASE_MODE_COMPARISON} — "
          f"{'will produce cross-mode comparison heatmaps.' if RUN_PHASE_MODE_COMPARISON else 'comparison disabled.'}\n")


    # ── Reward-vs-path model preview ─────────────────────────────────────
    # Standalone demonstration of the 'reward_path' model on 2 example
    # configs, before the ROI loop. Prints the per-bin label vector and
    # shows the per-condition downsampled matrix + the resulting model RDM.
    # Aggregates locations + buttons across subjects (purely behavioural —
    # ROI-independent) just for the requested configs.
    _PREVIEW_CONFIGS = configs[:2]
    print(f"\n=== reward_path model preview "
          f"(configs: {_PREVIEW_CONFIGS}) ===")
    _preview_locs_by_c    = {c: [] for c in _PREVIEW_CONFIGS}
    _preview_buttons_by_c = {c: [] for c in _PREVIEW_CONFIGS}
    for _sub_str, _sub_pack in SUBJECT_DATA.items():
        _beh = _sub_pack[f"sub-{_sub_str}"]['beh'].copy().reset_index(drop=True)
        _beh['config_str'] = (
            _beh['loc_A'].astype(int).astype(str) + '-' +
            _beh['loc_B'].astype(int).astype(str) + '-' +
            _beh['loc_C'].astype(int).astype(str) + '-' +
            _beh['loc_D'].astype(int).astype(str))
        _loc_df = _sub_pack[f"sub-{_sub_str}"]['locations']
        _btn_df = _sub_pack[f"sub-{_sub_str}"]['buttons']
        for _c in _PREVIEW_CONFIGS:
            _idx = (_beh['config_str'] == _c) & (_beh['correct'] == 1)
            if not _idx.any():
                continue
            _preview_locs_by_c[_c].append(_loc_df[_idx].to_numpy())
            _preview_buttons_by_c[_c].append(_btn_df[_idx].to_numpy())

    _preview_labels_360  = {}
    _preview_cond_matrix = {}
    _preview_per_trial   = {}
    for _c in _PREVIEW_CONFIGS:
        if not _preview_locs_by_c[_c]:
            print(f"  [preview] {_c}: no correct trials across subjects — skipping.")
            continue
        _stacked_loc = np.vstack(_preview_locs_by_c[_c])
        _stacked_btn = np.vstack(_preview_buttons_by_c[_c])
        _label_360, _per_trial = build_reward_path_label_360(
            loc_trials=_stacked_loc, btn_trials=_stacked_btn,
            config_str=_c, n_states=len(states),
            return_per_trial=True)
        _preview_labels_360[_c] = _label_360
        _preview_per_trial[_c] = _per_trial

        # Per-condition downsampled matrix (12 conds × 12 features).
        _W = len(_label_360) // N_CONDS_PER_CONF
        _cond_mat = np.array([
            downsample_mode(_label_360[i * _W:(i + 1) * _W],
                            target_len=LEN_STANDARDISED_PATH)
            for i in range(N_CONDS_PER_CONF)
        ], dtype=object)
        _preview_cond_matrix[_c] = _cond_mat

        _r_locs = reward_locations_for_config(_c)
        _n_reward_bins = int(np.asarray(_label_360, dtype=int).sum())
        _n_trials_preview = _stacked_loc.shape[0]
        print(f"\n  config {_c}: reward locations A,B,C,D = {_r_locs}  "
              f"(aggregated across {_n_trials_preview} correct trials)")
        print(f"    total reward bins in 360-bin vector: {_n_reward_bins}")
        _bins_per_state = len(_label_360) // len(states)
        for _k in range(len(states)):
            _s, _e = _k * _bins_per_state, (_k + 1) * _bins_per_state
            _n_k_agg = int(np.asarray(_label_360[_s:_e], dtype=int).sum())
            _per_trial_state = _per_trial[:, _s:_e].sum(axis=1)
            _mn, _md, _mx = (int(_per_trial_state.min()),
                              int(np.median(_per_trial_state)),
                              int(_per_trial_state.max()))
            print(f"    state {states[_k]} (loc {_r_locs[_k]}): "
                  f"{_n_k_agg} agg reward bins  | "
                  f"per-trial min/median/max = {_mn}/{_md}/{_mx}"
                  f"{'  ⚠ data gap (min=0)' if _mn == 0 else ''}")
        print(f"    per-condition labels (12 conds × {LEN_STANDARDISED_PATH} feats):")
        for _i in range(N_CONDS_PER_CONF):
            _row_int = np.asarray(_cond_mat[_i], dtype=int).tolist()
            _frac = sum(_row_int) / len(_row_int)
            print(f"      cond {_i:2d}  [{' '.join(map(str, _row_int))}]  "
                  f"frac_reward={_frac:.2f}")

    # # ── Preview figure: bin-level overlay + 24×24 model RDM ──────────────
    # if _preview_cond_matrix:
    #     _stacked = np.vstack([_preview_cond_matrix[_c]
    #                           for _c in _PREVIEW_CONFIGS
    #                           if _c in _preview_cond_matrix]).astype(int)
    #     _stacked_obj = _stacked.astype(object)
    #     _rdm_preview = mc.analyse.my_RSA.compute_hamming_distance(
    #         _stacked_obj, plotting=False, include_diagonal=False,
    #         model_name='reward_path[preview]',
    #         no_tasks=len(_preview_cond_matrix))
    #     _rdm_vec = np.asarray(_rdm_preview[0], dtype=float)
    #     _N = _stacked.shape[0]
    #     _ii, _jj = np.triu_indices(_N, k=1)
    #     _rdm_square = np.full((_N, _N), np.nan)
    #     _rdm_square[_ii, _jj] = _rdm_vec
    #     _rdm_square[_jj, _ii] = _rdm_vec

    #     _fig, _axes = plt.subplots(
    #         1 + len(_PREVIEW_CONFIGS), 1,
    #         figsize=(11, 2.0 * len(_PREVIEW_CONFIGS) + 5.5),
    #         gridspec_kw={'height_ratios': [1] * len(_PREVIEW_CONFIGS) + [3]})
    #     for _ax, _c in zip(_axes[:-1], _PREVIEW_CONFIGS):
    #         if _c not in _preview_labels_360:
    #             _ax.set_visible(False); continue
    #         _lbl = np.asarray(_preview_labels_360[_c], dtype=int)
    #         _ax.imshow(_lbl[None, :], aspect='auto',
    #                    cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    #         for _k in range(1, len(states)):
    #             _ax.axvline(_k * (len(_lbl) // len(states)) - 0.5,
    #                         color='black', lw=0.7)
    #         _ax.set_yticks([])
    #         _ax.set_title(
    #             f"config {_c} — per-bin reward (red) vs path (white)  "
    #             f"  reward locs A,B,C,D = {reward_locations_for_config(_c)}",
    #             fontsize=10)
    #         _ax.set_xlabel('bin (0..360)', fontsize=8)

    #     _ax_rdm = _axes[-1]
    #     _im = _ax_rdm.imshow(_rdm_square, cmap='RdBu_r', aspect='equal')
    #     _ax_rdm.set_title(
    #         f'reward_path model RDM — preview ({_N}×{_N}, '
    #         f'{len(_PREVIEW_CONFIGS)} configs × {N_CONDS_PER_CONF} conds)',
    #         fontsize=10)
    #     for _k in range(1, len(_PREVIEW_CONFIGS)):
    #         _ax_rdm.axvline(_k * N_CONDS_PER_CONF - 0.5,
    #                         color='black', lw=0.7)
    #         _ax_rdm.axhline(_k * N_CONDS_PER_CONF - 0.5,
    #                         color='black', lw=0.7)
    #     plt.colorbar(_im, ax=_ax_rdm, fraction=0.04, pad=0.02,
    #                  label='Hamming dist')
    #     _fig.tight_layout()
    #     _preview_path = os.path.join(OUT_DIR, 'reward_path_preview.png')
    #     _fig.savefig(_preview_path, dpi=150, bbox_inches='tight')
    #     plt.show()
    #     print(f"\n  Saved reward_path preview figure -> {_preview_path}\n")


    for roi_name, roi_pred in ROI_RULES.items():
        print(f"\n========== ROI: {roi_name} ==========")

        # Data-loss counters for this ROI. Anything that produces NaN entries
        # in the per-cell averages — either because the grouping log lacks an
        # entry for (subject, config) or because one of the run halves has
        # zero correct trials — gets logged here and reported at the end of
        # the ROI block. We *never* drop a cell silently.
        missing_log_subject_configs = []   # list of (sub_str, conf)
        empty_half_subject_configs  = []   # list of (sub_str, conf, half, n_other_half)

        # set up dicts and lists to load data
        acc_neurons, locs, buttons = {}, {}, {}
        acc_neurons_all, locs_all, buttons_all = {}, {}, {}
        # per_cell_trial_chunks accumulates the raw (n_trials, 360) firing
        # rates for every (cell, config) — consumed by
        # mc.analyse.rsa_perm_rdms to build / cache the permuted data RDMs.
        # Replaces the previous inline circular-shift loop that lived inside
        # the cell loop and rebuilt the perm population matrix N_PERMUTATIONS
        # times per cell × config.
        per_cell_trial_chunks = {}   # neuron_label → {'cell_id', 'per_config'}

        for conf in configs:
            acc_neurons[conf] = {}
            locs[conf] = {}
            buttons[conf] = {}
            buttons_all[conf] = []
            acc_neurons_all[conf] = []
            locs_all[conf] = []
            for th in [1,2]:
                acc_neurons[conf][th] = []
                locs[conf][th] = []
                buttons[conf][th] = []


        N_CONFIGS = len(configs)


        for sub_str in SUBJECTS:
            data_dict = SUBJECT_DATA[sub_str]
            # figure out correct config indices

            beh = data_dict[f"sub-{sub_str}"]['beh'].copy().reset_index(drop=True)
            beh['config'] = list(zip(
                beh['loc_A'].astype(int), beh['loc_B'].astype(int),
                beh['loc_C'].astype(int), beh['loc_D'].astype(int),
            ))
            beh['grid_no']    = beh['grid_no'].astype(int)
            beh['config_str'] = beh['config'].apply(
                lambda t: f'{t[0]}-{t[1]}-{t[2]}-{t[3]}')
            curr_neurons = data_dict[f"sub-{sub_str}"]['normalised_neurons']
            
            sub_glog = GROUPING_LOGS.get(sub_str, {})

            for conf in configs:
                cfg_entry = sub_glog.get(conf)
                # If the subject has no allocation for this config, treat it as
                # zero-trials in both halves: every matching cell will still
                # get one entry per config (NaN-filled) so the downstream
                # np.hstack alignment across configs stays consistent. We log
                # the case and report it at the end of the ROI block so the
                # silent NaN-fill is never invisible.
                if cfg_entry is None:
                    run1_blocks, run2_blocks = [], []
                    missing_log_subject_configs.append((sub_str, conf))
                    print(f"  [WARN] {roi_name}: sub {sub_str} has no grouping-log "
                          f"entry for config {conf} — cell entries for this "
                          f"(subject, config) will be NaN in both halves.")
                else:
                    run1_blocks = cfg_entry['run1_blocks']
                    run2_blocks = cfg_entry['run2_blocks']

                # Trial masks per RSA half, respecting block boundaries.
                # idx_all  : all correct trials of this config (for between_tasks avg)
                # idx_run1 : correct trials whose grid_no ∈ run1_blocks  (RSA half-1)
                # idx_run2 : correct trials whose grid_no ∈ run2_blocks  (RSA half-2)
                # By construction run1_blocks ∩ run2_blocks = ∅, so no spilling.
                idx_all  = (beh['config_str'] == conf) & (beh['correct'] == 1)
                idx_run1 = idx_all & beh['grid_no'].isin(run1_blocks)
                idx_run2 = idx_all & beh['grid_no'].isin(run2_blocks)

                # Empty-half check at the (subject, config) level. If either
                # half has zero correct trials for this subject × config, every
                # one of that subject's cells in this ROI will record NaN for
                # that half. Surface it once here rather than per-cell.
                n_r1 = int(idx_run1.sum())
                n_r2 = int(idx_run2.sum())
                if n_r1 == 0 and n_r2 > 0:
                    empty_half_subject_configs.append((sub_str, conf, 1, n_r2))
                    print(f"  [WARN] {roi_name}: sub {sub_str} config {conf} has 0 "
                          f"run1 trials (run2={n_r2}) — half-1 entries will be NaN.")
                elif n_r2 == 0 and n_r1 > 0:
                    empty_half_subject_configs.append((sub_str, conf, 2, n_r1))
                    print(f"  [WARN] {roi_name}: sub {sub_str} config {conf} has 0 "
                          f"run2 trials (run1={n_r1}) — half-2 entries will be NaN.")
                elif n_r1 == 0 and n_r2 == 0 and cfg_entry is not None:
                    empty_half_subject_configs.append((sub_str, conf, 0, 0))
                    print(f"  [WARN] {roi_name}: sub {sub_str} config {conf} has 0 "
                          f"correct trials in either half despite a grouping "
                          f"log entry — all entries will be NaN.")

                # locations + buttons
                loc_df  = data_dict[f"sub-{sub_str}"]['locations']
                btn_df  = data_dict[f"sub-{sub_str}"]['buttons']
                locs[conf][1].append(loc_df[idx_run1].to_numpy())
                locs[conf][2].append(loc_df[idx_run2].to_numpy())
                locs_all[conf].append(loc_df[idx_all].to_numpy())
                buttons[conf][1].append(btn_df[idx_run1].to_numpy())
                buttons[conf][2].append(btn_df[idx_run2].to_numpy())
                buttons_all[conf].append(btn_df[idx_all].to_numpy())

                # Row-level masks into the trial-stack (used inside the cell loop).
                # These map "row position in conf_neurons_all" -> {is run1, is run2}.
                # Computed once per (subject, config) and reused for every cell.
                _all_orig_idx = beh.index[idx_all].to_numpy()
                _r1_orig_idx  = set(beh.index[idx_run1].to_numpy().tolist())
                _r2_orig_idx  = set(beh.index[idx_run2].to_numpy().tolist())
                row_run1_mask = np.array(
                    [i in _r1_orig_idx for i in _all_orig_idx], dtype=bool)
                row_run2_mask = np.array(
                    [i in _r2_orig_idx for i in _all_orig_idx], dtype=bool)

                for n_lab in curr_neurons:
                    if roi_pred(n_lab):
                        # Record MNI coords (once per cell) for the ROI overview plot.
                        sub_int, cell_int = parse_neuron_label(n_lab)
                        if sub_int is not None:
                            key = (sub_int, cell_int)
                            if key not in roi_electrode_coords[roi_name]:
                                mni = get_neuron_mni(n_lab)
                                if all(np.isfinite(mni)):
                                    roi_electrode_coords[roi_name][key] = mni

                        conf_neurons_all = curr_neurons[n_lab][idx_all].to_numpy()
                        if conf_neurons_all.shape[0] == 0:
                            continue

                        # Store the raw (n_trials, 360) firing rates per
                        # (cell, config) so mc.analyse.rsa_perm_rdms can
                        # build / cache the permuted data RDMs in one place.
                        # The inline N_PERMUTATIONS-times shift loop that
                        # used to live here is gone — perms are now produced
                        # once per ROI via the central builder.
                        if n_lab not in per_cell_trial_chunks:
                            per_cell_trial_chunks[n_lab] = {
                                'cell_id':    n_lab,
                                'per_config': {},
                            }
                        per_cell_trial_chunks[n_lab]['per_config'][conf] = {
                            'trials_all': conf_neurons_all,
                            'run1_mask':  row_run1_mask.copy(),
                            'run2_mask':  row_run2_mask.copy(),
                        }

                        # Empirical means: between_tasks RDM uses all correct trials;
                        # split_halves RDM uses the run1 / run2 row masks.
                        avg_all = np.nanmean(conf_neurons_all, axis=0)
                        acc_neurons_all[conf].append(
                            avg_all.reshape(
                                N_CONDS_PER_CONF,
                                int(360 / N_CONDS_PER_CONF)).mean(axis=1))
                        for th, rmask in [(1, row_run1_mask),
                                          (2, row_run2_mask)]:
                            if not rmask.any():
                                acc_neurons[conf][th].append(
                                    np.full(N_CONDS_PER_CONF, np.nan))
                                continue
                            avg_th = np.nanmean(
                                conf_neurons_all[rmask], axis=0)
                            acc_neurons[conf][th].append(
                                avg_th.reshape(
                                    N_CONDS_PER_CONF,
                                    int(360 / N_CONDS_PER_CONF)).mean(axis=1))

        # import pdb; pdb.set_trace()

        n_neurons = len(acc_neurons[conf][th])

        if n_neurons == 0:
            print(f"[{roi_name}] no neurons matched — skipping.")
            continue

        # ── Data-loss summary for this ROI ──────────────────────────────
        # Every (subject, config) that produced NaN entries — either because
        # no grouping-log entry existed, or because a half had 0 correct
        # trials — is enumerated here. NaN cells are also counted in the
        # assembled trial matrices so we never lose visibility into
        # silent NaN-fill.
        print(f"\n[{roi_name}] === data-loss audit ===")
        if missing_log_subject_configs:
            print(f"  Missing grouping-log entries: "
                  f"{len(missing_log_subject_configs)} (subject, config) cases")
            for s, c in missing_log_subject_configs[:10]:
                print(f"    sub {s}  config {c}")
            if len(missing_log_subject_configs) > 10:
                print(f"    ... ({len(missing_log_subject_configs) - 10} more)")
        else:
            print(f"  Missing grouping-log entries: 0")
        if empty_half_subject_configs:
            print(f"  Empty-half (subject, config, half) cases: "
                  f"{len(empty_half_subject_configs)}")
            for s, c, h, n_other in empty_half_subject_configs[:10]:
                if h == 0:
                    print(f"    sub {s}  config {c}  both halves empty")
                else:
                    print(f"    sub {s}  config {c}  half {h} empty "
                          f"(other half has {n_other} trials)")
            if len(empty_half_subject_configs) > 10:
                print(f"    ... ({len(empty_half_subject_configs) - 10} more)")
        else:
            print(f"  Empty-half (subject, config, half) cases: 0")

        # Per-(config, half) NaN cell counts in the assembled trial matrices.
        nan_cells_by_half = {}
        for conf in configs:
            for th in [1, 2]:
                arr = np.asarray(acc_neurons[conf][th])
                if arr.size == 0:
                    nan_cells_by_half[(conf, th)] = 0
                    continue
                # A "NaN cell" = a cell that has any NaN across the 12
                # downsampled conditions of this config × half.
                nan_per_cell = np.isnan(arr).any(axis=1)
                nan_cells_by_half[(conf, th)] = int(nan_per_cell.sum())
        total_nan_cell_entries = sum(nan_cells_by_half.values())
        total_cell_entries = sum(
            len(acc_neurons[c][t]) for c in configs for t in (1, 2)
        )
        print(f"  NaN cell-config-half entries in assembled trial matrix: "
              f"{total_nan_cell_entries}/{total_cell_entries} "
              f"({100.0 * total_nan_cell_entries / max(total_cell_entries, 1):.2f}%)")
        if total_nan_cell_entries:
            print(f"  Per-(config, half) NaN cell counts "
                  f"(only non-zero entries shown):")
            for (c, th), n in nan_cells_by_half.items():
                if n:
                    print(f"    {c}  half {th}: {n} NaN cells")

        # Dump a per-ROI CSV so the audit is reproducible without re-running.
        audit_rows = [{'roi': roi_name, 'kind': 'missing_log',
                       'subject': s, 'config': c, 'half': 0, 'n_other_half': 0}
                      for s, c in missing_log_subject_configs] + [
            {'roi': roi_name, 'kind': 'empty_half',
             'subject': s, 'config': c, 'half': h, 'n_other_half': n}
            for s, c, h, n in empty_half_subject_configs
        ]
        audit_path = os.path.join(OUT_DIR, 'data_loss_audit.csv')
        if audit_rows:
            _df_audit = pd.DataFrame(audit_rows)
            if os.path.exists(audit_path):
                _df_audit.to_csv(audit_path, mode='a', header=False, index=False)
            else:
                _df_audit.to_csv(audit_path, index=False)
        print(f"  Audit rows (this ROI): {len(audit_rows)}  -> {audit_path}")

        print(f"[{roi_name}] {n_neurons} neurons collected.")

        rows = []
        row_labels = []

        # create the long neuron vector.
        for task_half in [1, 2]:
            for config in configs:
                neuron_values = acc_neurons[config][task_half]
                neuron_values = np.asarray(neuron_values)

                rows.append(neuron_values)
                row_labels.append((config, task_half))

        mat = np.hstack(rows)
        data_RDM = mc.analyse.my_RSA.compute_crosscorr(mat.T, plotting=PLOT_FIGS, include_diagonal=False, no_tasks=len(configs), model=f'data in {roi_name}')

        # z-scored ACC neurons.
        mu = np.nanmean(mat, axis=1)      # one mean per neuron
        sd = np.nanstd(mat, axis=1)       # one std per neuron
        mat_z = (mat.T - mu) / sd
        data_RDM_z = mc.analyse.my_RSA.compute_crosscorr(mat_z, plotting=PLOT_FIGS, include_diagonal=False, no_tasks=len(configs), model=f'data in z-scored {roi_name} neurons')



        # ── Run-averaged data matrix (feeds the within-run RDM family) ───
        # `acc_neurons_all[config]` holds, per cell, the mean firing rate
        # across ALL correct trials of that config — both runs collapsed
        # (idx_all = `correct == 1` without any run-half filter; see the
        # subject loop above). Stacking these per-config means gives the
        # run-averaged population matrix consumed by compute_crosscorr_within
        # to produce data_RDM_within / _across / _full and their _z variants.
        # → between_tasks (and the diagonal-masked between_tasks_z) is therefore
        # built from per-config averages of all 2 or 3 runs combined, NOT
        # from any single run.
        row_all = []
        row_labels_all = []
        for config in configs:
            all_neuron_values = acc_neurons_all[config]
            row_all.append(all_neuron_values)
            row_labels_all.append(config)
        mat_all = np.hstack(row_all)
        print(f"  [{roi_name}] within-run RDM source: "
              f"mean across all correct trials per config "
              f"(both/all session runs collapsed) — shape {mat_all.shape}")

        data_RDM_within, data_RDM_across, data_RDM_full = mc.analyse.my_RSA.compute_crosscorr_within(mat_all.T, plotting=PLOT_FIGS, include_diagonal=False, no_tasks=len(configs), model=f'data in {roi_name}', block_size=N_CONDS_PER_CONF)

        # z-scored ACC neurons.
        # z-scored ACC neurons.
        mu_all = np.nanmean(mat_all, axis=1)      # one mean per neuron
        sd_all = np.nanstd(mat_all, axis=1)       # one std per neuron
        mat_all_z = (mat_all.T - mu_all) / sd_all
        #data_RDM_within_z, data_RDM_across_z, data_RDM_full_z = mc.analyse.my_RSA.compute_crosscorr_within(mat_all_z, plotting=PLOT_FIGS, include_diagonal=False, no_tasks=len(configs), model=f'data in z-scored {roi_name} neurons', block_size=N_CONDS_PER_CONF)
        data_RDM_within_z, data_RDM_across_z, data_RDM_full_z = mc.analyse.my_RSA.compute_crosscorr_within(mat_all_z, plotting=True, include_diagonal=False, no_tasks=len(configs), model=f'data in z-scored {roi_name} neurons', block_size=N_CONDS_PER_CONF)
        
        #import pdb; pdb.set_trace()


        # create a mode path.
        mode_locs, mode_locs_all, mode_buttons, mode_buttons_all = {}, {}, {}, {}
        # reward_path is a SIBLING of mode_locs / mode_buttons (one 360-bin
        # array per (config, half)), but built by labelling each trial
        # individually first and then taking the mode across trials per bin.
        # Computing the reward window on the per-bin mode of loc + btn would
        # require both signals to coincide at the same bin position, which
        # rarely holds since trials are not bin-aligned (see commit message /
        # diagnostic) — that route under-reports reward bins for most states.
        mode_reward_path, mode_reward_path_all = {}, {}
        # Per-(config, half) min reward bins across trials per state — used
        # to warn about real data gaps (some half has a state with zero
        # reward bins in any trial) rather than algorithm bugs.
        reward_path_min_per_state = {}
        for c in configs:
            mode_locs[c] = {}
            mode_buttons[c] = {}
            mode_reward_path[c] = {}
            reward_path_min_per_state[c] = {}

            locs_all_per_conf = locs_all[c]
            stacked_all = np.vstack(locs_all_per_conf) # (n_trials_total, 360)
            m_all = stats.mode(stacked_all, axis=0, keepdims=False, nan_policy='omit')
            mode_locs_all[c] = m_all.mode.astype(float)

            buttons_all_per_conf = buttons_all[c]
            stacked_all_buttons = np.vstack(buttons_all_per_conf) # (n_trials_total, 360)
            b_m_all = stats.mode(stacked_all_buttons, axis=0, keepdims=False, nan_policy='omit')
            mode_buttons_all[c] = b_m_all.mode

            mode_reward_path_all[c] = build_reward_path_label_360(
                loc_trials=stacked_all, btn_trials=stacked_all_buttons,
                config_str=c, n_states=len(states))

            for th in [1, 2]:
                loc_per_config = locs[c][th]
                stacked = np.vstack(loc_per_config) # (n_trials_total, 360)
                m = stats.mode(stacked, axis=0, keepdims=False, nan_policy='omit')
                mode_locs[c][th] = m.mode.astype(float)

                # and for buttons
                button_per_config = buttons[c][th]
                stacked_b = np.vstack(button_per_config) # (n_trials_total, 360)
                m_b = stats.mode(stacked_b, axis=0, keepdims=False, nan_policy='omit')
                mode_buttons[c][th] = m_b.mode

                # reward_path: per-trial windows, then per-bin mode across trials.
                _rp_agg, _rp_per_trial = build_reward_path_label_360(
                    loc_trials=stacked, btn_trials=stacked_b,
                    config_str=c, n_states=len(states),
                    return_per_trial=True)
                mode_reward_path[c][th] = _rp_agg
                _W = stacked.shape[1] // len(states)
                reward_path_min_per_state[c][th] = [
                    int(_rp_per_trial[:, k * _W:(k + 1) * _W].sum(axis=1).min())
                    for k in range(len(states))
                ]


        print("Mode-location, mode-button, and mode-reward-path arrays built.")
        # Surface real data gaps: every state should visit its rewarded
        # location in every trial. A zero here means a trial in some half
        # had no Return-press-at-reward in that state's slot, which is a
        # behavioural gap, not an algorithm bug.
        _gap_rows = [(c, th, k_i, mn)
                     for c, by_th in reward_path_min_per_state.items()
                     for th, mins in by_th.items()
                     for k_i, mn in enumerate(mins) if mn == 0]
        if _gap_rows:
            print(f"  [reward_path] WARN {len(_gap_rows)} (config, half, state) "
                  f"cell(s) have at least one trial with zero reward bins "
                  f"in that state's slot — these are real behavioural gaps:")
            for c, th, k_i, mn in _gap_rows[:10]:
                print(f"    config {c}  half {th}  state {states[k_i]}: "
                      f"min reward bins/trial = 0")
            if len(_gap_rows) > 10:
                print(f"    ... and {len(_gap_rows) - 10} more (full list in "
                      f"reward_path_min_per_state if needed).")

        # build model rdms

        print("Building model matrices...")
        
        n_conditions = N_CONFIGS * N_CONDS_PER_CONF
        n_dsr_neurons = LEN_STANDARDISED_PATH * N_PHASES * len(states)
        
        # Initialise matrices for each half
        matrices = {
            run_id: {
                'loc':     make_empty(n_conditions, LEN_STANDARDISED_PATH),
                'dsr':     make_empty(n_conditions, n_dsr_neurons),
                'dsr_fmri': make_empty(n_conditions, N_CONDS_PER_CONF * LEN_STANDARDISED_PATH),
                'bttn_curr': make_empty(n_conditions, LEN_STANDARDISED_PATH, dtype=object),
                'bttn_prev': make_empty(n_conditions, LEN_STANDARDISED_PATH, dtype=object),
                'bttn_next': make_empty(n_conditions, LEN_STANDARDISED_PATH, dtype=object),
                # l2_norm: 9-feature negative-Euclidean-distance vector from
                # the current location to each of the 9 grid cells. Mirrors
                # create_fMRI_model_RDMs_on_clean_beh.py / models['l2_norm'].
                'l2_norm': make_empty(n_conditions, 9),
                # reward_path: per-bin binary label (1 = at uncovered reward,
                # 0 = path) downsampled per condition to LEN_STANDARDISED_PATH
                # features. Built via build_reward_path_label_360 from the
                # mode-button + mode-location vectors and the config string.
                'reward_path': make_empty(n_conditions, LEN_STANDARDISED_PATH,
                                          dtype=object),
            }
            for run_id in (1, 2)
        }
        
        for c_idx, c in enumerate(configs):
            row_start = c_idx * N_CONDS_PER_CONF
            for run_id, mats in matrices.items():
                mode_vec        = mode_locs[c][run_id]      # (360,)
                mode_vec_button = mode_buttons[c][run_id]
                LEN_OG_SUBPATH = int(len(mode_vec)/N_CONDS_PER_CONF)

                dsr_base = downsample_mode(mode_vec, target_len=n_dsr_neurons)

                # fMRI-style DSR: integer-ID mode trajectory, rolled per bin
                # (parallel to create_fMRI_model_RDMs_on_clean_beh.py / EVs['DSR']).
                # Same construct as 'dsr' above, just labelled separately so we
                # can compare the L=12-per-bin / fMRI-aligned wiring head-to-head.
                mats['dsr_fmri'][row_start:row_start + N_CONDS_PER_CONF] = (
                    build_mode_path_dsr(mode_vec, N_CONDS_PER_CONF, LEN_STANDARDISED_PATH))

                # Per-bin reward/path label for this (config, half), already
                # built per-trial and aggregated across trials in the
                # mode_reward_path block above. Slicing it here mirrors how
                # `loc` and `bttn_curr` slice their mode arrays.
                reward_path_360 = mode_reward_path[c][run_id]

                for n_subpath in range(N_CONDS_PER_CONF):
                    row = row_start + n_subpath

                    # --- location ---
                    subpath = mode_vec[n_subpath * LEN_OG_SUBPATH:(n_subpath + 1) * LEN_OG_SUBPATH]
                    mats['loc'][row] = downsample_mode(subpath, target_len=LEN_STANDARDISED_PATH)

                    # --- l2_norm: single curr-loc summary per condition ---
                    # Take the most common location in this subpath; convert to
                    # the 9-vector of negative L2 distances to each grid cell.
                    # If subpath is all-NaN (no behaviour at that time), zero row.
                    sub_clean = subpath[~np.isnan(subpath)] if subpath.dtype.kind == 'f' \
                        else subpath
                    if len(sub_clean) == 0:
                        mats['l2_norm'][row] = 0.0
                    else:
                        sub_mode = int(Counter(sub_clean.tolist())
                                       .most_common(1)[0][0])
                        mats['l2_norm'][row] = l2_norm_row_for_loc(sub_mode)

                    # --- dsr ---
                    mats['dsr'][row] = np.roll(dsr_base, -n_subpath * LEN_STANDARDISED_PATH)

                    # --- buttons (current / previous / next), shift by ±1 subpath ---
                    # --- buttons (current / previous / next), wraparound by ±1 subpath ---
                    for key, offset in [('bttn_curr', 0), ('bttn_prev', -1), ('bttn_next', +1)]:
                        # import pdb; pdb.set_trace()
                        shifted_n = (n_subpath + offset) % N_CONDS_PER_CONF
                        s = shifted_n * LEN_OG_SUBPATH
                        mats[key][row] = downsample_mode(mode_vec_button[s : s + LEN_OG_SUBPATH], target_len=LEN_STANDARDISED_PATH)

                    # --- reward_path: per-bin label → mode-downsampled per cond ---
                    rp_sub = reward_path_360[n_subpath * LEN_OG_SUBPATH:
                                              (n_subpath + 1) * LEN_OG_SUBPATH]
                    mats['reward_path'][row] = downsample_mode(
                        rp_sub, target_len=LEN_STANDARDISED_PATH)
                        

        # --- state / feedback / phase ---
        # Each state must span N_CONDS_PER_CONF // len(states) consecutive
        # conditions so the model covers ALL rows of every config block (not
        # just the first 12 with the legacy RESOLUTIONx * N_PHASES layout,
        # which left bins 12..N_CONDS_PER_CONF-1 zero for any subject with
        # N_CONDS_PER_CONF > N_PHASES * len(states)).
        # Phase divides each state's span into N_PHASES roughly-equal slices;
        # widths may differ by ±1 when STATE_WIDTH is not exactly divisible
        # by N_PHASES (e.g. STATE_WIDTH=5, N_PHASES=3 → [1, 1, 3] using
        # floor-division boundaries — every condition gets one phase).
        assert N_CONDS_PER_CONF % len(states) == 0, (
            f"N_CONDS_PER_CONF ({N_CONDS_PER_CONF}) must be divisible by "
            f"len(states) ({len(states)}) so state widths are equal.")
        STATE_WIDTH = N_CONDS_PER_CONF // len(states)
        state_config    = np.zeros((N_CONDS_PER_CONF, len(states)))
        feedback_config = np.zeros((N_CONDS_PER_CONF, len(states)))
        phase_config    = np.zeros((N_CONDS_PER_CONF, N_PHASES))

        for s_i, s in enumerate(states):
            s_start = s_i * STATE_WIDTH
            s_end   = (s_i + 1) * STATE_WIDTH
            state_config[s_start:s_end, s_i] = 1
            if s == 'A':
                # Feedback / repeat_counter pulse at the moment of reward A,
                # i.e. the very first condition of state A within each config.
                feedback_config[s_start:s_start + 1, s_i] = 1
            for p_i in range(N_PHASES):
                p_start = s_start + (p_i * STATE_WIDTH) // N_PHASES
                p_end   = s_start + ((p_i + 1) * STATE_WIDTH) // N_PHASES
                # Defensive: ensure every condition is tagged with exactly one
                # phase even when STATE_WIDTH is not divisible by N_PHASES.
                if p_end <= p_start:
                    p_end = min(p_start + 1, s_end)
                phase_config[p_start:p_end, p_i] = 1

        # Sanity: every condition must be claimed by exactly one state and
        # at least one phase, so the model isn't zero on any row of any
        # config block.
        _state_sum = state_config.sum(axis=1)
        _phase_sum = phase_config.sum(axis=1)
        assert np.all(_state_sum == 1), (
            f"state_config rows must sum to 1; got sums {_state_sum.tolist()}")
        assert np.all(_phase_sum >= 1), (
            f"phase_config rows must sum to ≥1 (every cond gets a phase); "
            f"got sums {_phase_sum.tolist()}")
        
        state_half    = np.tile(state_config,    (len(configs), 1))
        feedback_half = np.tile(feedback_config, (len(configs), 1))
        phase_half    = np.tile(phase_config,    (len(configs), 1))
        
        model_concat = {
            'location':   np.concatenate([matrices[1]['loc'],     matrices[2]['loc']],     axis=0),
            'dsr':        np.concatenate([matrices[1]['dsr'],     matrices[2]['dsr']],     axis=0),
            'dsr_fmri':   np.concatenate([matrices[1]['dsr_fmri'], matrices[2]['dsr_fmri']], axis=0),
            'bttn_curr':    np.concatenate([matrices[1]['bttn_curr'], matrices[2]['bttn_curr']], axis=0),
            'bttn_prev':    np.concatenate([matrices[1]['bttn_prev'], matrices[2]['bttn_prev']], axis=0),
            'bttn_next':    np.concatenate([matrices[1]['bttn_next'], matrices[2]['bttn_next']], axis=0),
            'l2_norm':    np.concatenate([matrices[1]['l2_norm'], matrices[2]['l2_norm']], axis=0),
            'reward_path': np.concatenate([matrices[1]['reward_path'], matrices[2]['reward_path']], axis=0),
            'state':      np.tile(state_half, (2, 1)),
            'repeat_counter': np.tile(feedback_half, (2,1))
        }

        
        if SPLIT_UNCV_BUTTONS == True:
            # now, as an additional step, split 'return' buttons from buttons.
            # also replace/forward fill the locations that had 'return' with the
            # previoys buttons that were pressed.
            # model_concat['uncover'] = (model_concat['bttn_curr'] == "Return").astype(int)
            model_concat['uncover'] = np.where(model_concat['bttn_curr'] == "Return","uncover","off")
            for button_type in ['bttn_curr', 'bttn_prev', 'bttn_next']:
                arr = model_concat[button_type].copy().astype(object)
            
                # Replace Return with missing
                arr[arr == "Return"] = np.nan
                # Forward-fill normally; backward-fill only affects leading Returns
                filled = (pd.Series(arr.ravel()).ffill().bfill().to_numpy().reshape(arr.shape))
                model_concat[button_type] = filled
        
        # old way
        dsr_old, midnight, dsr_old_now_next, state_phase, phase, loc_old = [], [], [], [], [], []
        for th in [1,2]:
            for c in configs:
                walked   = mode_locs[c][th]
                walked = [int(w-1) for w in walked]
                loc_og_matrix, phase_og_matrix, stat_matrix, midnight_matrix, dsr_matrix, phas_stat_matrix, dsr_now_next_matrix = mc.simulation.predictions.model_DSR(locations = walked, no_phase_neurons=N_PHASES)
                
                loc_old_downsampled = loc_og_matrix.reshape(loc_og_matrix.shape[0], N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis=2)
                state_phase_downsampled = phas_stat_matrix.reshape(phas_stat_matrix.shape[0], N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis=2)
                dsr_downsampled = dsr_matrix.reshape(dsr_matrix.shape[0], N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis=2)
                midn_downsampled = midnight_matrix.reshape(midnight_matrix.shape[0],N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis = 2)
                dsr_nownext_downsampled = dsr_now_next_matrix.reshape(dsr_now_next_matrix.shape[0], N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis = 2)
                phase_downsampled = phase_og_matrix.reshape(phase_og_matrix.shape[0], N_CONDS_PER_CONF, int(360/N_CONDS_PER_CONF)).mean(axis = 2)

                dsr_old.append(dsr_downsampled)
                midnight.append(midn_downsampled)
                dsr_old_now_next.append(dsr_nownext_downsampled)
                state_phase.append(state_phase_downsampled)
                phase.append(phase_downsampled)
                loc_old.append(loc_old_downsampled)
                


        model_concat['dsr_old'] = np.transpose(np.concatenate(dsr_old, axis = 1))
        model_concat['midnight'] = np.transpose(np.concatenate(midnight, axis = 1))
        model_concat['dsr_old_now_next'] = np.transpose(np.concatenate(dsr_old_now_next, axis = 1))
        model_concat['state_phase'] = np.transpose(np.concatenate(state_phase, axis = 1))
        model_concat['phase'] = np.transpose(np.concatenate(phase, axis = 1))
        model_concat['location_old'] = np.transpose(np.concatenate(loc_old, axis = 1))

        # Reduced-lag dsr_fmri variants: only the first K of 12 lag-windows
        # (LEN_STANDARDISED_PATH=12 columns per lag-window). Built by column-
        # truncating the full dsr_fmri matrix so the Hamming metric stays
        # identical and any difference reflects only the lag-truncation.
        _L = LEN_STANDARDISED_PATH
        model_concat['dsr_fmri_lag01']    = model_concat['dsr_fmri'][:, :2 * _L]   # current + 1
        model_concat['dsr_fmri_lag012']   = model_concat['dsr_fmri'][:, :3 * _L]   # + 2
        model_concat['dsr_fmri_lag0123']  = model_concat['dsr_fmri'][:, :4 * _L]   # + 3

        # ── Lag-selected (not just truncated) DSR variants ───────────────────
        # Pick arbitrary subsets of the 12 lag-windows by concatenating the
        # corresponding 12-column blocks. Used for the exploratory combo
        # family that swaps dsr_fmri for a future-only / proximal-future /
        # next-state-only / data-informed version, holding the control set
        # fixed.  Lag indexing: lag k = "location at subpath (current+k)";
        # lag 0 = current bin, lag 3 = first phase of next state, etc.
        def _lag_cols(lag_ids, L=_L):
            return np.concatenate([np.arange(k * L, (k + 1) * L) for k in lag_ids])
        model_concat['dsr_fmri_fut']      = model_concat['dsr_fmri'][:, _lag_cols(range(1, N_CONDS_PER_CONF))]
        model_concat['dsr_fmri_123']      = model_concat['dsr_fmri'][:, _lag_cols([1, 2, 3])]
        model_concat['dsr_fmri_345']      = model_concat['dsr_fmri'][:, _lag_cols([3, 4, 5])]
        model_concat['dsr_fmri_informed'] = model_concat['dsr_fmri'][:, _lag_cols([1, 2])]

        # State-quarter split of the full DSR trajectory. 12 lags = 4 states
        # × 3 phases per state, so each quarter is 3 consecutive lag windows
        # (3 * LEN_STANDARDISED_PATH = 36 columns). Mirrors split4_DSR_keys
        # in scripts/create_fMRI_model_RDMs_on_clean_beh.py.
        _QLEN = 3 * _L
        for _q, _name in enumerate(
                ['curr_quarter', 'next_quarter', 'next2_quarter', 'next3_quarter']):
            model_concat[_name] = model_concat['dsr_fmri'][:,
                                                             _q * _QLEN:(_q + 1) * _QLEN]


        # Persist the per-condition DESIGN MATRICES, not just the RDMs they
        # produce. Without this the only record was a PNG, so any figure
        # wanting a model at its true resolution had to re-derive it -- which
        # is how the fig3 `location` panel came to show one modal location per
        # condition (the model uses LEN_STANDARDISED_PATH sub-steps per
        # condition) from a SINGLE trial (the model uses the across-subject
        # mode path). Saved here, after every model has been added to
        # `model_concat` -- an earlier save point missed state_phase, phase,
        # midnight, dsr_old, location_old and the lag variants.
        try:
            _dm_dir = os.path.join(OUT_DIR, 'model_design_matrices')
            os.makedirs(_dm_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(_dm_dir, 'model_design_matrices.npz'),
                **{k: np.asarray(v) for k, v in model_concat.items()},
                __configs__=np.asarray(configs),
                __n_configs__=N_CONFIGS,
                __n_conds_per_conf__=N_CONDS_PER_CONF,
                __len_standardised_path__=LEN_STANDARDISED_PATH)
        except Exception as _exc:
            print(f"[warn] could not save model design matrices: {_exc}")

        # ── Model design-matrix plots ────────────────────────────────────
        # Visualise the (conditions × features) matrix that actually feeds
        # each model's RDM. One panel per model, run-1 half only (run-2 is
        # an identical mode-trajectory so plotting both halves is redundant).
        # Saved once, on the example ROI, into <OUT_DIR>/model_design_matrices/.
        if roi_name == EXAMPLE_ROI_FOR_FIGS:
            from mc.plotting.cell_results import plot_model_design_matrices
            md_dir = os.path.join(OUT_DIR, 'model_design_matrices')
            os.makedirs(md_dir, exist_ok=True)
            plot_model_design_matrices(
                model_concat=model_concat,
                models=['dsr_fmri', 'location', 'l2_norm',
                        'bttn_curr', 'bttn_next', 'state', 'midnight'],
                n_configs=N_CONFIGS, n_conds_per_config=N_CONDS_PER_CONF,
                save_path=os.path.join(md_dir,
                                        f'model_design_matrices_{roi_name}.png'),
                roi_label=f'example ROI {roi_name}',
            )
            plt.show()
            print(f"  Saved model design-matrix figure -> {md_dir}")


        model_RDMs = {}
        model_RDMs_within = {}
        model_RDMs_across = {}
        full = {}

        for m in model_concat:

            if m in ('location', 'dsr', 'dsr_fmri', 'bttn_prev', 'bttn_next', 'bttn_curr', 'uncover',
                     'dsr_fmri_lag01', 'dsr_fmri_lag012', 'dsr_fmri_lag0123',
                     'dsr_fmri_fut', 'dsr_fmri_123', 'dsr_fmri_345', 'dsr_fmri_informed',
                     'curr_quarter', 'next_quarter', 'next2_quarter', 'next3_quarter',
                     'reward_path'):
                model_RDMs[m] = mc.analyse.my_RSA.compute_hamming_distance(
                    model_concat[m], plotting=False, include_diagonal=False,
                    model_name=m, no_tasks=len(configs))

                model_RDMs_within[m], model_RDMs_across[m], full[m] = mc.analyse.my_RSA.compute_hamming_distance_within(
                    model_concat[m][0:len(mat_all_z)], plotting=PLOT_FIGS,
                    include_diagonal=False,
                    model_name=m, no_tasks=len(configs),
                    block_size=N_CONDS_PER_CONF)
            else:
                model_RDMs[m] = mc.analyse.my_RSA.compute_crosscorr(
                    model_concat[m], plotting=False, include_diagonal=False,
                    no_tasks=len(configs), model=m)
                model_RDMs_within[m], model_RDMs_across[m], full[m] = mc.analyse.my_RSA.compute_crosscorr_within(
                    model_concat[m][0:len(mat_all_z)], plotting=PLOT_FIGS,
                    include_diagonal=False,
                    no_tasks=len(configs), model=m,
                    block_size=N_CONDS_PER_CONF)


        # Repeat-counter rows are mostly zero, so the cosine RDM has NaN cells
        # wherever both endpoints are all-zero (norm=0). Replace with 1 (max
        # cosine dissimilarity) across ALL three RDM-vector dicts so combos
        # that include 'repeat_counter' don't blow up evaluate_model when run
        # on the split_halves / between_tasks variants.
        # (model_RDMs_within is the deprecated within-block-only variant —
        #  retained in the dict structure but never used downstream because
        #  it would be autocorrelation on the run-averaged RDM.)
        for _rdm_dict in (model_RDMs, model_RDMs_within, model_RDMs_across):
            _vec = _rdm_dict['repeat_counter'][0]
            _vec[np.isnan(_vec)] = 1
        # Defensive guard for the vector RDMs that actually feed evaluate_model.
        # Any remaining NaN/inf in any model would crash statsmodels deep
        # inside evaluate_model; replace with 0 and report which model produced
        # them so we can investigate rather than silently fudge results.
        for _label, _rdm_dict in (('split_halves',  model_RDMs),
                                  ('within',        model_RDMs_within),
                                  ('between_tasks', model_RDMs_across)):
            for _m in _rdm_dict:
                _arr = np.asarray(_rdm_dict[_m][0], dtype=float)
                if not np.isfinite(_arr).all():
                    n_bad = int((~np.isfinite(_arr)).sum())
                    print(f"  [{roi_name}] {_label} / {_m}: {n_bad} non-finite "
                          f"entries -> replaced with 0")
                    _arr[~np.isfinite(_arr)] = 0
                    _rdm_dict[_m][0] = _arr

        # ── Save data + model RDMs per ROI for cheap add-on replay ────────────
        # All downstream evaluate_model calls only need the 1-D upper-tri RDM
        # vectors (and the metadata to interpret them). Storing them lets a
        # separate add-on script (e.g. scripts/RSA_addon_analyses.py) replay
        # the phase-mask comparison, swap in extra combos, etc. without ever
        # touching the cell data again.
        rdm_save_dir = os.path.join(OUT_DIR, 'rdms')
        os.makedirs(rdm_save_dir, exist_ok=True)
        _rdm_payload = {
            # Test-variant identifying metadata
            '__roi__':              np.asarray(roi_name),
            '__n_neurons__':        np.asarray(n_neurons),
            '__configs__':          np.asarray(list(configs)),
            '__n_configs__':        np.asarray(N_CONFIGS),
            '__n_conds_per_conf__': np.asarray(N_CONDS_PER_CONF),
            '__n_phases__':         np.asarray(N_PHASES),
            # Data RDMs (1-D upper-tri vectors). Names mirror the in-script
            # variable names so an add-on script can pick the right one for
            # each test variant without inventing new keys.
            'data__split_halves':         np.asarray(data_RDM[0],          dtype=float),
            'data__split_halves_z':       np.asarray(data_RDM_z[0],        dtype=float),
            'data__between_tasks':        np.asarray(data_RDM_across[0],   dtype=float),
            'data__between_tasks_z':      np.asarray(data_RDM_across_z[0], dtype=float),
            'data__within':               np.asarray(data_RDM_within[0],   dtype=float),
            'data__within_z':             np.asarray(data_RDM_within_z[0], dtype=float),
        }
        # Model RDMs for each test variant. model_RDMs / model_RDMs_across /
        # model_RDMs_within are dicts keyed by model name; pickle them as
        # individual arrays under model__<variant>__<modelname>.
        for _m, _entry in model_RDMs.items():
            _rdm_payload[f'model__split_halves__{_m}']  = np.asarray(_entry[0], dtype=float)
        for _m, _entry in model_RDMs_across.items():
            _rdm_payload[f'model__between_tasks__{_m}'] = np.asarray(_entry[0], dtype=float)
        for _m, _entry in model_RDMs_within.items():
            _rdm_payload[f'model__within__{_m}']        = np.asarray(_entry[0], dtype=float)
        np.savez_compressed(
            os.path.join(rdm_save_dir, f'rdms_{roi_name}.npz'),
            **_rdm_payload,
        )
        print(f"  [{roi_name}] saved data + model RDMs -> "
              f"{os.path.join(rdm_save_dir, f'rdms_{roi_name}.npz')} "
              f"({len(_rdm_payload)} arrays)")

        # ── Optional smoke test: phase-residualise every RDM (data + models) ──
        # ── Publication figures 2 + 3 (shared with rodent pipeline) ────────
        # Built once for the example ROI: data + per-model activations and
        # RDMs across all 8 task configurations, using the mode-path models
        # (mode_locs_all, which already pools both halves). Mirrors the
        # rodent pipeline so a single change to the plotting helpers
        # propagates to both analyses.
        if roi_name == EXAMPLE_ROI_FOR_FIGS:
            DSR_MODEL_ORDER_HUMAN = ['dsr_old', 'state', 'location_old', 'phase']
            half1_slice = slice(0, N_CONFIGS * N_CONDS_PER_CONF)  # the across-blocks RDM is built from half-1
            fig_model_acts = {m: model_concat[m][half1_slice].T
                              for m in DSR_MODEL_ORDER_HUMAN}
            fig_model_rdms = {m: np.asarray(full[m], dtype=float).copy()
                              for m in DSR_MODEL_ORDER_HUMAN}
            fig_data_act   = mat_all_z.T      # (n_neurons, N_CONFIGS*N_CONDS_PER_CONF)
            fig_data_rdm   = np.asarray(data_RDM_full_z, dtype=float).copy()

            # CONFIRMATORY view: when the primary pipeline masks out pairs,
            # NaN those same cells in the pub figure RDMs so the figure shows
            # ONLY what entered evaluate_model. Diagonal blocks (within-config)
            # are already masked from the between-tasks variant, so for the
            # phase-mask we additionally NaN cross-phase OR within-phase cells
            # depending on the mode.
            if PHASE_MASK_MODE != 'full':
                _keep_phase = _phase_mask_matrix(
                    PHASE_MASK_MODE, N_CONFIGS, N_CONDS_PER_CONF, N_PHASES)
                _cfg_idx = np.repeat(np.arange(N_CONFIGS), N_CONDS_PER_CONF)
                _between_cfg = _cfg_idx[:, None] != _cfg_idx[None, :]
                _keep_for_fig = _keep_phase & _between_cfg
                fig_data_rdm = np.where(_keep_for_fig, fig_data_rdm, np.nan)
                for _m in DSR_MODEL_ORDER_HUMAN:
                    fig_model_rdms[_m] = np.where(_keep_for_fig,
                                                  fig_model_rdms[_m], np.nan)
                print(f"[pub fig] masked pub-figure RDMs to mode "
                      f"'{PHASE_MASK_MODE}' (between-config cells only).")

            figs_dir = os.path.join(OUT_DIR, 'pub_figures')
            os.makedirs(figs_dir, exist_ok=True)

            dsr_figs.pub_figure_example_subject(
                data_activation=fig_data_act, data_rdm=fig_data_rdm,
                model_activations=fig_model_acts, model_rdms=fig_model_rdms,
                model_order=DSR_MODEL_ORDER_HUMAN,
                n_tasks=N_CONFIGS, n_conds_per_task=N_CONDS_PER_CONF,
                recday_label=f'human cells / {roi_name} [{PHASE_MASK_MODE}]',
                save_stem=os.path.join(figs_dir,
                                       f'fig2_human_{roi_name}_{PHASE_MASK_MODE}'))

            # Fig 3 schematics: one deterministic example task, with one
            # panel for every model used by the current saved/run config.
            _save_pub_fig3_model_schematics(
                run_dir=OUT_DIR,
                run_config={
                    'models': models,
                    'combo_models': combo_models,
                    'configs': configs,
                    'N_PHASES': N_PHASES,
                    'states': states,
                },
                task_config_str=configs[0],
            )

        # import pdb; pdb.set_trace()
        print("Computing RSA...")

        # ── Empirical results ────────────────────────────────────────────────
        empirical_results = {}
        empirical_results_z = {}

        empirical_combo_results = {}
        empirical_combo_results_z = {}


        test_specs = [
            # 'split_halves' (= across-runs RDM):
            #   Built from a 2-half (10 reps each) population matrix; each
            #   RDM cell compares run-1 sub-population to run-2 sub-pop.
            #   That includes BOTH (a) the within-task block-diagonal
            #   (same config in run 1 vs same config in run 2 — valid
            #   because run 1 and run 2 are independent data), AND (b) the
            #   between-task off-block cells (different configs across
            #   runs). All upper-tri cells contribute.
            #
            # 'between_tasks' (= run-averaged, between-task-only):
            #   Built from per-config means (repeats pre-averaged into one
            #   population vector per condition). The within-config block-
            #   diagonal is MASKED because those cells would be auto-
            #   correlations (the same averaged vector compared to itself).
            #   Only between-task cells contribute.
            #
            # The deprecated 'within' variant (commented out below) would
            # have been "only the within-config block of the run-averaged
            # RDM" — that is exactly the autocorrelation trap and should
            # never be used. Kept commented as a reminder.
            ('split_halves',  model_RDMs,        data_RDM[0],        data_RDM_z[0]),
            #('within',       model_RDMs_within, data_RDM_within[0], data_RDM_within_z[0]),  # DO NOT USE: autocorrelation
            ('between_tasks', model_RDMs_across, data_RDM_across[0], data_RDM_across_z[0]),
        ]

        for test_name, rdm_dict, raw_data, z_data in test_specs:
            pmask = _phase_mask_for(test_name, PHASE_MASK_MODE)
            raw_m = _apply_phase_mask(raw_data, pmask)
            z_m   = _apply_phase_mask(z_data, pmask)

            empirical_results[test_name] = {
                m: eval_tuple(_apply_phase_mask(rdm_dict[m][0], pmask), raw_m,
                              label=f'[{roi_name}] {PHASE_MASK_MODE}/{test_name}/{m}')
                for m in models
            }
            empirical_results_z[test_name] = {
                m: eval_tuple(_apply_phase_mask(rdm_dict[m][0], pmask), z_m,
                              label=f'[{roi_name}] {PHASE_MASK_MODE}/{test_name}_z/{m}')
                for m in models
            }


            empirical_combo_results[test_name] = {
                combo: evaluate_combo_safe(
                    _apply_phase_mask(build_combo_rdm(rdm_dict, combo_models[combo]), pmask),
                    raw_m, combo_models[combo],
                    label=f'[{roi_name}] {PHASE_MASK_MODE}/{test_name}/{combo}')
                for combo in combo_models}

            empirical_combo_results_z[test_name] = {
                combo: evaluate_combo_safe(
                    _apply_phase_mask(build_combo_rdm(rdm_dict, combo_models[combo]), pmask),
                    z_m, combo_models[combo],
                    label=f'[{roi_name}] {PHASE_MASK_MODE}/{test_name}_z/{combo}')
                for combo in combo_models}
        # ── Block-diag / off-block / full breakdown (split_halves_z) ─────
        # Diagnostic: split the across-runs upper-tri RDM into
        #   * block_diag  : same-config cells (within-task across runs;
        #                   valid because runs are independent)
        #   * off_block   : different-config cells (between-task)
        #   * full        : both combined (= the primary split_halves_z test)
        # For each of the 3 combos in `combo_models`, re-run evaluate_model on
        # each subset. Saved as a per-ROI CSV row and printed inline so we can
        # see whether the FDR-family combo's signal is carried by within-task
        # or between-task pairs.
        N_RDM = N_CONFIGS * N_CONDS_PER_CONF
        _ii, _jj = np.triu_indices(N_RDM, k=1)
        _cfg_i = _ii // N_CONDS_PER_CONF
        _cfg_j = _jj // N_CONDS_PER_CONF
        _MASKS_BLOCK = {
            'block_diag': _cfg_i == _cfg_j,
            'off_block':  _cfg_i != _cfg_j,
            'full':       np.ones_like(_cfg_i, dtype=bool),
        }
        # Sanity check: vector length must match the 1-D RDM the regressions use.
        if len(data_RDM_z[0]) != _MASKS_BLOCK['full'].sum():
            print(f"  [block-breakdown] vector length mismatch "
                  f"({len(data_RDM_z[0])} vs {_MASKS_BLOCK['full'].sum()}) "
                  f"— skipping breakdown for {roi_name}.")
        else:
            block_break_rows = []
            print(f"\n  --- {roi_name}: block-diag / off-block / full breakdown "
                  f"(split_halves_z) ---")
            for mask_name, mvec in _MASKS_BLOCK.items():
                d_vec = np.asarray(data_RDM_z[0], dtype=float)[mvec]
                for combo_key, sub_models in combo_models.items():
                    stacked = build_combo_rdm(model_RDMs, sub_models)
                    stacked_m = np.asarray(stacked, dtype=float)[mvec, :]
                    t_arr, b_arr, p_arr = evaluate_combo_safe(
                        stacked_m, d_vec, sub_models,
                        label=f'[{roi_name}] {mask_name}/{combo_key}')
                    for s_i, sm in enumerate(sub_models):
                        block_break_rows.append({
                            'roi':       roi_name,
                            'mask':      mask_name,
                            'combo':     combo_key,
                            'sub_model': sm,
                            't':         float(t_arr[s_i]),
                            'beta':      float(b_arr[s_i]),
                            'p':         float(p_arr[s_i]),
                            'n_pairs':   int(mvec.sum()),
                        })
                    # Print only the dsr_fmri row to keep stdout terse.
                    if 'dsr_fmri' in sub_models:
                        _i = sub_models.index('dsr_fmri')
                        print(f"    {mask_name:>11s}  {combo_key:<18s} "
                              f"dsr_fmri  t={t_arr[_i]:+.2f}  "
                              f"beta={b_arr[_i]:+.4f}  p={p_arr[_i]:.4f}  "
                              f"(n_pairs={int(mvec.sum())})")
            # Append to a global CSV (created on first ROI, appended for the rest)
            block_break_path = os.path.join(
                OUT_DIR, 'rdm_block_breakdown_split_halves_z.csv')
            _df_bb = pd.DataFrame(block_break_rows)
            if os.path.exists(block_break_path):
                _df_bb.to_csv(block_break_path, mode='a', header=False, index=False)
            else:
                _df_bb.to_csv(block_break_path, index=False)

        # ── FDR-combo RDM diagnostic ─────────────────────────────────────
        # Show the EXACT 1-D vectors that feed evaluate_model for the
        # FDR-family combo, embedded back into a (N×N) display grid at the
        # positions they occupy in evaluate_model. Provenance per column:
        #
        #   across-run figure                 source vector
        #   ─────────────────────────  ───────────────────────────────────
        #   full        (= split_halves_z)   data_RDM_z[0]    / model_RDMs[m][0]
        #   block_diag  (within-task subset) same vectors, sub-indexed
        #   off_block   (across-task subset) same vectors, sub-indexed
        #
        #   within-run figure                 source vector
        #   ─────────────────────────  ───────────────────────────────────
        #   off_block   (= between_tasks_z)  data_RDM_across_z[0] /
        #                                    model_RDMs_across[m][0]
        #
        # Both compute_crosscorr and compute_crosscorr_within strip the
        # diagonal (include_diagonal=False, k=1), so the display has the
        # diagonal blanked AND a red diagonal line as an explicit marker.
        fdr_combo_key = FDR_COMBOS[0] if FDR_COMBOS else None
        if (fdr_combo_key and fdr_combo_key in combo_models
                and len(data_RDM_z[0]) == N_RDM * (N_RDM - 1) // 2):
            from mc.plotting.cell_results import plot_rdm_grid
            fdr_subs = combo_models[fdr_combo_key]
            rdm_diag_dir = os.path.join(OUT_DIR, 'rdm_diagnostics')
            os.makedirs(rdm_diag_dir, exist_ok=True)

            # Index arrays for the upper-tri positions (k=1: diagonal excluded)
            _ii_disp, _jj_disp = np.triu_indices(N_RDM, k=1)
            cfg_i_disp = _ii_disp // N_CONDS_PER_CONF
            cfg_j_disp = _jj_disp // N_CONDS_PER_CONF
            mask_block_diag_vec = cfg_i_disp == cfg_j_disp
            mask_off_block_vec  = cfg_i_disp != cfg_j_disp

            # ── Across-run figure (split_halves family) ────────────────
            # All three columns use the SAME source 1-D vector (the
            # split_halves_z input). 'block_diag' and 'off_block' are
            # subset views — sub-indexed from the full vector and placed at
            # the corresponding subset of upper-tri positions.
            full_pos_mask = np.ones_like(mask_block_diag_vec, dtype=bool)
            def _full_row(label, vec_full):
                return (label, {
                    'full':       (vec_full,                   full_pos_mask),
                    'block_diag': (vec_full[mask_block_diag_vec], mask_block_diag_vec),
                    'off_block':  (vec_full[mask_off_block_vec],  mask_off_block_vec),
                })
            rows_across = [_full_row(
                'data\n(split_halves_z\n= data_RDM_z[0])', data_RDM_z[0])]
            for sm in fdr_subs:
                if sm not in model_RDMs:
                    continue
                rows_across.append(_full_row(
                    f'{sm}\n(= model_RDMs[{sm!r}][0])',
                    model_RDMs[sm][0]))
            _save_a = os.path.join(rdm_diag_dir,
                                   f'fdr_rdm_across_run_{roi_name}.png')
            plot_rdm_grid(
                rows_to_plot=rows_across,
                col_specs=[
                    ('full',       '\n(= split_halves_z input)'),
                    ('block_diag', '\n(within-task subset)'),
                    ('off_block',  '\n(across-task subset)'),
                ],
                n_rdm=N_RDM, n_configs=N_CONFIGS,
                n_conds_per_config=N_CONDS_PER_CONF, configs=configs,
                suptitle=(f'FDR-family across-run RDMs — ROI {roi_name} '
                          f'(combo: {fdr_combo_key})'),
                save_path=_save_a)
            plt.show()
            print(f"  Saved FDR across-run RDM diagnostic -> {_save_a}")

            # ── Within-run figure (between_tasks family) ───────────────
            # Single column = the off-block 1-D vector that compute_crosscorr_within
            # returns directly (= between_tasks_z input). No mask hack: the
            # vector has exactly the cells evaluate_model sees, placed at
            # their off-block upper-tri positions.
            rows_within = []
            rows_within.append((
                'data\n(between_tasks_z\n= data_RDM_across_z[0])',
                {
                    'off_block': (data_RDM_across_z[0], mask_off_block_vec),
                },
            ))
            for sm in fdr_subs:
                if sm not in model_RDMs_across:
                    continue
                rows_within.append((
                    f'{sm}\n(= model_RDMs_across[{sm!r}][0])',
                    {
                        'off_block': (model_RDMs_across[sm][0], mask_off_block_vec),
                    },
                ))
            _save_w = os.path.join(rdm_diag_dir,
                                   f'fdr_rdm_within_run_{roi_name}.png')
            plot_rdm_grid(
                rows_to_plot=rows_within,
                col_specs=[
                    ('off_block',  '\n(= between_tasks_z input)'),
                ],
                n_rdm=N_RDM, n_configs=N_CONFIGS,
                n_conds_per_config=N_CONDS_PER_CONF, configs=configs,
                suptitle=(f'FDR-family within-run RDMs — ROI {roi_name} '
                          f'(combo: {fdr_combo_key})'),
                save_path=_save_w)
            plt.show()
            print(f"  Saved FDR within-run RDM diagnostic -> {_save_w}")

        # ── Mode comparison: empirical-only RSA for all 3 phase modes ────
        # Cheap (no permutations) — uses the SAME model + data RDMs and just
        # applies a different mask per mode. Stored separately from the primary
        # results above and consumed by the cross-ROI comparison heatmap.
        # When PHASE_MASK_MODE='full' (the usual case), the 'full' mode here
        # is bit-identical to the primary empirical_results — we alias rather
        # than recompute to remove a divergence risk.
        if RUN_PHASE_MODE_COMPARISON:
            for mode in ALL_PHASE_MODES:
                roi_mode_comparison.setdefault(mode, {})[roi_name] = {}
                for test_name, rdm_dict, raw_data, z_data in test_specs:
                    n_pairs_total = int(np.asarray(raw_data).size)
                    if mode == 'full' and PHASE_MASK_MODE == 'full':
                        roi_mode_comparison[mode][roi_name][test_name] = {
                            'raw_single':    empirical_results[test_name],
                            'z_single':      empirical_results_z[test_name],
                            'raw_combo':     empirical_combo_results[test_name],
                            'z_combo':       empirical_combo_results_z[test_name],
                            'n_pairs_kept':  n_pairs_total,
                            'n_pairs_total': n_pairs_total,
                        }
                        continue
                    pmask = _phase_mask_for(test_name, mode)
                    raw_m = _apply_phase_mask(raw_data, pmask)
                    z_m   = _apply_phase_mask(z_data, pmask)
                    roi_mode_comparison[mode][roi_name][test_name] = {
                        'raw_single': {
                            m: eval_tuple(
                                _apply_phase_mask(rdm_dict[m][0], pmask), raw_m,
                                label=f'[{roi_name}] {mode}/{test_name}/{m}')
                            for m in models},
                        'z_single': {
                            m: eval_tuple(
                                _apply_phase_mask(rdm_dict[m][0], pmask), z_m,
                                label=f'[{roi_name}] {mode}/{test_name}_z/{m}')
                            for m in models},
                        'raw_combo': {
                            combo: evaluate_combo_safe(
                                _apply_phase_mask(
                                    build_combo_rdm(rdm_dict, combo_models[combo]), pmask),
                                raw_m, combo_models[combo],
                                label=f'[{roi_name}] {mode}/{test_name}/{combo}')
                            for combo in combo_models},
                        'z_combo': {
                            combo: evaluate_combo_safe(
                                _apply_phase_mask(
                                    build_combo_rdm(rdm_dict, combo_models[combo]), pmask),
                                z_m, combo_models[combo],
                                label=f'[{roi_name}] {mode}/{test_name}_z/{combo}')
                            for combo in combo_models},
                        'n_pairs_kept': (int(pmask.sum()) if pmask is not None
                                          else n_pairs_total),
                        'n_pairs_total': n_pairs_total,
                    }



        # Active test variants:
        #   'split_halves'    = across-runs RDM (run 1 × run 2 population
        #                       matrix). Both within-task-across-runs and
        #                       between-task cells contribute. Raw r.
        #   'split_halves_z'  = same data RDM as 'split_halves' but z-scored
        #                       per neuron before RDM construction.
        #   'between_tasks'   = run-averaged RDM (repeats collapsed per
        #                       config). Within-config block-diagonal is
        #                       masked out (those cells would be autocorr).
        #                       Only between-task pairs contribute. Raw r.
        #   'between_tasks_z' = z-scored version of 'between_tasks'.
        # 'within' / 'within_z' are deprecated (commented below) — they
        # would have been "only within-config block of the run-averaged
        # RDM", which is exactly the autocorrelation trap.
        #tests = ['split_halves', 'split_halves_z', 'within', 'within_z', 'between_tasks', 'between_tasks_z']
        tests = ['split_halves', 'split_halves_z', 'between_tasks', 'between_tasks_z']
        perm_results = {test: {m: [] for m in models} for test in tests}
        perm_results_combo = {
            test: {combo: {'t': [], 'beta': [], 'p': []} for combo in combo_models}
            for test in tests
        }


        if N_PERMUTATIONS:
            # ── Cached perm-data-RDM build ──────────────────────────────────
            # mc.analyse.rsa_perm_rdms produces the n_perms permuted data
            # RDMs once per ROI: circular shifts per (cell, trial) → trial
            # average → downsample → z-score per neuron → crosscorr RDM.
            # Result is pickled at OUT_DIR/perm_data_rdms/perm_data_rdms_<ROI>.pkl
            # with a fingerprint covering every parameter that influences
            # the outcome (cells, n_perms, seed, phase setting, configs,
            # method version) — a mismatching pickle triggers a rebuild.
            #
            # Only the z-scored variants are cached (`split_halves_z` and
            # `between_tasks_z`). The non-z `split_halves` / `between_tasks`
            # rows in the summary CSV keep their empirical betas but their
            # `p_perm` columns become NaN — we no longer compute non-z
            # permutation nulls in the main pipeline.
            from mc.analyse.rsa_perm_rdms import (
                load_or_build_perm_rdms,
                fingerprint as _rsa_perm_fingerprint,
                TEST_VARIANTS as _PERM_TEST_VARIANTS,
            )

            per_cell_trials_list = list(per_cell_trial_chunks.values())
            _cell_ids = [c['cell_id'] for c in per_cell_trials_list]
            perm_pkl_path = os.path.join(
                OUT_DIR, 'perm_data_rdms',
                f'perm_data_rdms_{roi_name}.pkl',
            )
            _perm_fp = _rsa_perm_fingerprint(
                roi=roi_name,
                cell_ids=_cell_ids,
                n_perms=N_PERMUTATIONS,
                seed=42,
                phase_residualise=PHASE_RESIDUALISE,
                residualise_repeats=RESIDUALISE_REPEATS,
                configs=configs,
                n_conds_per_config=N_CONDS_PER_CONF,
                n_bins_per_trial=360,
                shuffle_mode=SHUFFLE_MODE,
            )
            # Build the cross-run search list: every sibling run dir
            # under OUT_BASE except the current one. The matcher walks
            # `<sibling>/perm_data_rdms/perm_data_rdms_<ROI>.pkl` and
            # validates the fingerprint before reusing.
            if REUSE_PERMS_FROM_PREVIOUS_RUNS:
                _search_dirs = sorted(
                    os.path.join(OUT_BASE, d) for d in os.listdir(OUT_BASE)
                    if (os.path.isdir(os.path.join(OUT_BASE, d))
                        and os.path.join(OUT_BASE, d) != OUT_DIR)
                )
            else:
                _search_dirs = None
            _, perm_data_rdms = load_or_build_perm_rdms(
                pickle_path=perm_pkl_path,
                per_cell_trials=per_cell_trials_list,
                fingerprint_data=_perm_fp,
                configs=configs,
                n_conds_per_config=N_CONDS_PER_CONF,
                n_bins_per_trial=360,
                verbose=True,
                search_dirs=_search_dirs,
                link_reused=LINK_REUSED_PERMS,
            )

            # ── Vectorised OLS across all perms ─────────────────────────────
            # One solve per (test × model) — replaces the inner Python loop
            # that used to run a separate OLS for every (perm × model × test).
            from scipy import stats as _scipy_stats

            # CLAUDE.md rule #4: the perm OLS uses the SAME function as the
            # empirical fit — mc.analyse.my_RSA.evaluate_model_vec —
            # called once per (test × model/combo) with the full
            # (n_perms, n_pairs) stack of permuted data RDMs as the target.
            # The function returns (t, beta, p) of shape (n_perms, n_feat)
            # in one vectorised solve, identical numerical convention as
            # the empirical evaluate_model wrapper.
            evaluate_model_vec = mc.analyse.my_RSA.evaluate_model_vec

            for test_name in tests:
                if test_name in _PERM_TEST_VARIANTS:
                    rdm_dict_for_test = (model_RDMs
                                          if test_name.startswith('split_halves')
                                          else model_RDMs_across)
                    Y_perms_full = perm_data_rdms[test_name]    # (n_perms, n_pairs)
                    pmask = _phase_mask_for(test_name, PHASE_MASK_MODE)
                    if pmask is not None:
                        Y_perms = Y_perms_full[:, pmask]
                    else:
                        Y_perms = Y_perms_full

                    for m in models:
                        x_full = np.asarray(rdm_dict_for_test[m][0], dtype=float)
                        x = x_full[pmask] if pmask is not None else x_full
                        # SAME function as the empirical fit — see CLAUDE.md rule #4.
                        # Y_perms is (n_perms, n_pairs); evaluate_model_vec
                        # returns (n_perms, 1) for a single regressor.
                        _, BETA_PERMS, _ = evaluate_model_vec(x, Y_perms)
                        betas = np.asarray(BETA_PERMS, dtype=float).ravel()
                        perm_results[test_name][m] = list(betas)

                    for combo, combo_list in combo_models.items():
                        X_full = np.stack(
                            [np.asarray(rdm_dict_for_test[m][0], dtype=float)
                             for m in combo_list],
                            axis=1,
                        )
                        X = X_full[pmask, :] if pmask is not None else X_full
                        T_PERMS, BETA_PERMS, P_PERMS = evaluate_model_vec(
                            X, Y_perms)
                        perm_results_combo[test_name][combo]['t']    = [
                            T_PERMS[i]    for i in range(N_PERMUTATIONS)]
                        perm_results_combo[test_name][combo]['beta'] = [
                            BETA_PERMS[i] for i in range(N_PERMUTATIONS)]
                        perm_results_combo[test_name][combo]['p']    = [
                            P_PERMS[i]    for i in range(N_PERMUTATIONS)]
                else:
                    # Non-z test variant — perms no longer computed; fill
                    # with NaN so downstream summary code sees an empty
                    # perm distribution and produces NaN p_perm cleanly.
                    nan_arr = np.array([np.nan])
                    for m in models:
                        perm_results[test_name][m] = [np.nan] * N_PERMUTATIONS
                    for combo, combo_list in combo_models.items():
                        nan_feat = np.full(len(combo_list), np.nan)
                        perm_results_combo[test_name][combo]['t']    = [
                            nan_feat.copy() for _ in range(N_PERMUTATIONS)]
                        perm_results_combo[test_name][combo]['beta'] = [
                            nan_feat.copy() for _ in range(N_PERMUTATIONS)]
                        perm_results_combo[test_name][combo]['p']    = [
                            nan_feat.copy() for _ in range(N_PERMUTATIONS)]

            print(f"  [{roi_name}] perm-RDM vectorised OLS done "
                  f"({N_PERMUTATIONS} perms × {len(models)} models + "
                  f"{len(combo_models)} combos × {len(_PERM_TEST_VARIANTS)} z-tests)")



        # ── Permutation null draws (pickle) + histogram (FDR combos only) ───
        # Save the FULL null draws — every single model and every combo —
        # so we never need to recompute permutations just to inspect the
        # null distribution. Histograms are still rendered only for
        # FDR_COMBOS to keep the figure count manageable; the pickle keeps
        # the rest accessible for any ad-hoc plot later.
        if N_PERMUTATIONS:
            import pickle as _pickle
            perm_pkl_dir = os.path.join(OUT_DIR, 'perm_null_draws')
            os.makedirs(perm_pkl_dir, exist_ok=True)
            _fdr_combos_present = [c for c in FDR_COMBOS if c in combo_models]
            with open(os.path.join(perm_pkl_dir, f'perm_{roi_name}.pkl'), 'wb') as _f:
                _pickle.dump({
                    'roi':                       roi_name,
                    'n_neurons':                 n_neurons,
                    'n_permutations':            N_PERMUTATIONS,
                    'tests':                     tests,
                    'models':                    list(models),
                    'combo_models':              {c: combo_models[c] for c in combo_models},
                    'fdr_combos':                _fdr_combos_present,
                    'perm_results_singles':      perm_results,
                    'perm_results_combo':        perm_results_combo,
                    'empirical_results':         empirical_results,
                    'empirical_results_z':       empirical_results_z,
                    'empirical_combo_results':   empirical_combo_results,
                    'empirical_combo_results_z': empirical_combo_results_z,
                }, _f)

            from mc.plotting.cell_results import plot_permutation_hist_combo_grid
            perm_hist_dir = os.path.join(OUT_DIR, 'perm_hist')
            os.makedirs(perm_hist_dir, exist_ok=True)
            for combo_key in _fdr_combos_present:
                fig_c, _ = plot_permutation_hist_combo_grid(
                    perm_results_combo=perm_results_combo,
                    empirical_combo_results=empirical_combo_results,
                    empirical_combo_results_z=empirical_combo_results_z,
                    combo_key=combo_key,
                    combo_models=combo_models,
                    tests=tests,
                    bins=30,
                    alpha=0.05,
                    suptitle=f'ROI: {roi_name} — combo {combo_key} '
                             f'(n={n_neurons} neurons, {N_PERMUTATIONS} perms)',
                )
                fig_c.savefig(
                    os.path.join(
                        perm_hist_dir,
                        f'permutation_hist_{roi_name}_combo_{combo_key}.png'),
                    dpi=150, bbox_inches='tight',
                )
                fig_c.savefig(
                    os.path.join(
                        perm_hist_dir,
                        f'permutation_hist_{roi_name}_combo_{combo_key}.svg'),
                    bbox_inches='tight',
                )
                plt.close(fig_c)
            print(f"  Saved perm null draws + hist for {roi_name} "
                  f"({len(_fdr_combos_present)} FDR combo[s])")

        # ── Cache for cross-ROI summary ──────────────────────────────────────
        roi_perm_results[roi_name] = perm_results
        roi_empirical_results[roi_name] = {
            'raw': empirical_results,
            'z':   empirical_results_z,
            'combo_raw': empirical_combo_results,
            'combo_z':   empirical_combo_results_z,
        }

        # ── Build per-ROI rows for the overview table ────────────────────────
        # tests for raw vs z follow the same naming convention used in perm_results
        test_pairs = [
            ('split_halves',    'raw'), ('split_halves_z',    'z'),
            #('within',         'raw'), ('within_z',          'z'),
            ('between_tasks',   'raw'), ('between_tasks_z',   'z'),
        ]
        for test_name, kind in test_pairs:
            # empirical_results / empirical_results_z are keyed by the BASE
            # test name (no '_z' suffix); perm_results carries the suffix.
            emp_key = test_name[:-2] if test_name.endswith('_z') else test_name
            emp_dict = empirical_results if kind == 'raw' else empirical_results_z
            for m in models:
                t_val, beta_val, p_param = emp_dict[emp_key][m]
                perm_betas = np.asarray(perm_results[test_name][m], dtype=float).ravel()
                if perm_betas.size > 0:
                    p_perm = (np.sum(perm_betas >= beta_val) + 1) / (perm_betas.size + 1)
                else:
                    p_perm = np.nan
                summary_rows.append({
                    'roi':    roi_name,
                    'n_neurons': n_neurons,
                    'test':   test_name,
                    'model':  m,
                    't':      t_val,
                    'beta':   beta_val,
                    'p_param': p_param,
                    'p_perm': p_perm,
                })

            # combo models
            combo_emp_dict = empirical_combo_results if kind == 'raw' else empirical_combo_results_z
            for combo in combo_models:
                t_arr    = np.asarray(combo_emp_dict[emp_key][combo][0], dtype=float).ravel()
                beta_arr = np.asarray(combo_emp_dict[emp_key][combo][1], dtype=float).ravel()
                p_arr    = np.asarray(combo_emp_dict[emp_key][combo][2], dtype=float).ravel()
                for sub_idx, sub_model in enumerate(combo_models[combo]):
                    perm_beta_list = perm_results_combo[test_name][combo]['beta']
                    if len(perm_beta_list) > 0:
                        perm_beta_sub = np.asarray(
                            [np.asarray(b, dtype=float).ravel()[sub_idx] for b in perm_beta_list],
                            dtype=float,
                        )
                        p_perm = (np.sum(perm_beta_sub >= beta_arr[sub_idx]) + 1) / (perm_beta_sub.size + 1)
                    else:
                        p_perm = np.nan
                    summary_combo_rows.append({
                        'roi':       roi_name,
                        'n_neurons': n_neurons,
                        'test':      test_name,
                        'combo':     combo,
                        'sub_model': sub_model,
                        't':         float(t_arr[sub_idx]),
                        'beta':      float(beta_arr[sub_idx]),
                        'p_param':   float(p_arr[sub_idx]),
                        'p_perm':    p_perm,
                    })


    # ── Regressor-collinearity check (model-RDM correlations) ───────────────
    # model_RDMs / _within / _across are identical across ROIs (depend only on
    # mode_locs). Use the last ROI's dict and plot once.
    for rdm_dict, label in [
        (model_RDMs,        'full'),
        #(model_RDMs_within, 'within'),
        (model_RDMs_across, 'between_tasks'),
    ]:
        mc.plotting.results.plot_model_rdm_correlation(
            rdm_dict,
            title=f'Model-RDM correlations ({label})',
            save_path=os.path.join(OUT_DIR, f'model_rdm_correlations_{label}.png'),
        )


    # ── Cross-ROI overview tables ────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_combo_df = pd.DataFrame(summary_combo_rows)

    summary_csv = os.path.join(OUT_DIR, 'results_summary.csv')
    summary_combo_csv = os.path.join(OUT_DIR, 'results_summary_combos.csv')
    summary_df.to_csv(summary_csv, index=False)
    summary_combo_df.to_csv(summary_combo_csv, index=False)
    print(f"\nSaved overview tables:\n  {summary_csv}\n  {summary_combo_csv}")

    # ── Phase-mode comparison: betas across full / within / across ────────
    # Empirical (no-permutation) betas for the primary test variant under each
    # phase-mask mode. One figure per single-model row (1 column wide) AND
    # one figure per combo (n_sub_models wide), so we can see whether a result
    # that looks negative in one mode is positive in another and by how much.
    if RUN_PHASE_MODE_COMPARISON and roi_mode_comparison:
        cmp_dir = os.path.join(OUT_DIR, 'phase_mode_comparison')
        os.makedirs(cmp_dir, exist_ok=True)
        cmp_test_name = FDR_TEST                          # e.g. 'between_tasks_z'
        cmp_base = (cmp_test_name[:-2]
                    if cmp_test_name.endswith('_z') else cmp_test_name)
        cmp_combo_key  = ('z_combo'
                          if cmp_test_name.endswith('_z') else 'raw_combo')
        cmp_single_key = ('z_single'
                          if cmp_test_name.endswith('_z') else 'raw_single')

        # Long-format CSVs: one for single-model rows, one for combo rows.
        single_rows, combo_rows = [], []
        for _mode, per_roi in roi_mode_comparison.items():
            for _roi, per_test in per_roi.items():
                _entry = per_test.get(cmp_base)
                if _entry is None:
                    continue
                # single models
                for m, (t_v, b_v, p_v) in _entry[cmp_single_key].items():
                    single_rows.append({
                        'mode':          _mode,
                        'roi':           _roi,
                        'model':         m,
                        't':             float(t_v) if np.isfinite(t_v) else np.nan,
                        'beta':          float(b_v) if np.isfinite(b_v) else np.nan,
                        'p_param':       float(p_v) if np.isfinite(p_v) else np.nan,
                        'n_pairs_kept':  _entry['n_pairs_kept'],
                        'n_pairs_total': _entry['n_pairs_total'],
                    })
                # combo models
                for combo, res in _entry[cmp_combo_key].items():
                    _t    = np.asarray(res[0], dtype=float).ravel()
                    _beta = np.asarray(res[1], dtype=float).ravel()
                    _p    = np.asarray(res[2], dtype=float).ravel()
                    for sub_idx, sub_model in enumerate(combo_models[combo]):
                        combo_rows.append({
                            'mode':       _mode,
                            'roi':        _roi,
                            'combo':      combo,
                            'sub_model':  sub_model,
                            't':          float(_t[sub_idx]),
                            'beta':       float(_beta[sub_idx]),
                            'p_param':    float(_p[sub_idx]),
                            'n_pairs_kept':  _entry['n_pairs_kept'],
                            'n_pairs_total': _entry['n_pairs_total'],
                        })

        single_df = pd.DataFrame(single_rows)
        combo_df  = pd.DataFrame(combo_rows)
        single_csv = os.path.join(cmp_dir,
                                  f'phase_mode_comparison_singles_{cmp_test_name}.csv')
        combo_csv  = os.path.join(cmp_dir,
                                  f'phase_mode_comparison_combos_{cmp_test_name}.csv')
        single_df.to_csv(single_csv, index=False)
        combo_df.to_csv(combo_csv, index=False)
        print(f"\nSaved phase-mode comparison CSVs:\n  {single_csv}\n  {combo_csv}")

        # Render one heatmap per phase-mask mode using the shared
        # plot_roi_model_heatmap helper. Each figure shows exactly what
        # evaluate_model produced for that mode — no beautification, no
        # silent deduplication. If a sub-model name appears twice in a
        # combo (= configuration bug) plot_roi_model_heatmap will raise.
        from mc.plotting.cell_results import (
            plot_roi_model_heatmap, CANONICAL_ROI_ORDER,
            CANONICAL_RSA_MODEL_ORDER,
        )

        # Cell counts per ROI come from the primary summary table.
        roi_n_neurons = (summary_df.drop_duplicates('roi')
                                  .set_index('roi')['n_neurons'].to_dict())

        def _render_mode_heatmap(df, models_for_cols, title, save_stem):
            df = df.copy()
            df['n_neurons'] = df['roi'].map(roi_n_neurons)
            plot_roi_model_heatmap(
                df,
                models=list(models_for_cols),
                rois=ROIS_TO_ANALYZE,
                value_col='beta', annot_col='p_param', sig_col='p_param',
                n_col='n_neurons', alpha=0.05,
                value_label='empirical beta (no permutation)',
                title=title,
                save_path=f'{save_stem}.png',
                base_fontsize=13,
            )
            plt.show()
            print(f"  saved {save_stem}.png")

        # Single-model heatmaps — one per mode.
        if not single_df.empty:
            for _mode in ALL_PHASE_MODES:
                _sdf = single_df[single_df['mode'] == _mode]
                if _sdf.empty:
                    continue
                _render_mode_heatmap(
                    _sdf, models_for_cols=models,
                    title=(f'Phase-mode comparison — single models — '
                           f'mode "{_mode}" ({cmp_test_name})'),
                    save_stem=os.path.join(
                        cmp_dir,
                        f'phase_mode_comparison_singles_{_mode}_{cmp_test_name}'))

        # Combo heatmaps — one figure per (combo, mode).
        if not combo_df.empty:
            for combo, sub_models in combo_models.items():
                _cdf_all = combo_df[combo_df['combo'] == combo].copy()
                if _cdf_all.empty:
                    continue
                _cdf_all = _cdf_all.rename(columns={'sub_model': 'model'})
                for _mode in ALL_PHASE_MODES:
                    _cdf = _cdf_all[_cdf_all['mode'] == _mode]
                    if _cdf.empty:
                        continue
                    _render_mode_heatmap(
                        _cdf, models_for_cols=sub_models,
                        title=(f'Phase-mode comparison — combo {combo} — '
                               f'mode "{_mode}" ({cmp_test_name})'),
                        save_stem=os.path.join(
                            cmp_dir,
                            f'phase_mode_comparison_combo_{combo}_{_mode}_{cmp_test_name}'))
else:
    # Reload mode: pull summary CSVs (and electrode coords if saved) from disk.
    summary_df = pd.read_csv(os.path.join(OUT_DIR, 'results_summary.csv'))
    summary_combo_df = pd.read_csv(
        os.path.join(OUT_DIR, 'results_summary_combos.csv'))
    # Older saved runs used ACC / medialOFC / Parahippocampal in their
    # summary tables. Rename them on reload so re-rendered outputs match
    # fresh runs.
    for df in (summary_df, summary_combo_df):
        if 'roi' in df:
            df['roi'] = df['roi'].map(_canonical_roi_name)
    summary_df.to_csv(os.path.join(OUT_DIR, 'results_summary.csv'), index=False)
    summary_combo_df.to_csv(
        os.path.join(OUT_DIR, 'results_summary_combos.csv'), index=False)
    # Restrict to ROIs that passed the MIN_RSA_SUBJECTS filter so that
    # re-rendered figures and FDR corrections are consistent with a fresh run.
    if 'roi' in summary_df.columns:
        summary_df = summary_df[summary_df['roi'].isin(ROIS_TO_ANALYZE)]
    if 'roi' in summary_combo_df.columns:
        summary_combo_df = summary_combo_df[
            summary_combo_df['roi'].isin(ROIS_TO_ANALYZE)]
    coords_csv = os.path.join(OUT_DIR, 'roi_electrode_coords.csv')
    if os.path.exists(coords_csv):
        ec_df = pd.read_csv(coords_csv)
        ec_df['roi'] = ec_df['roi'].map(_canonical_roi_name)
        roi_electrode_coords = {
            roi: {(int(r['subject']), int(r['cell_idx'])):
                  (float(r['MNI_x']), float(r['MNI_y']), float(r['MNI_z']))
                  for _, r in g.iterrows()}
            for roi, g in ec_df.groupby('roi')
        }
    else:
        # Fall back to deriving electrode positions from the ROI table.
        roi_electrode_coords = {}
        for roi in ROI_RULES:
            atlas_roi = _atlas_roi_name(roi)
            sub_tbl = ROI_TABLE[ROI_TABLE[ROI_LABEL_COLUMN] == atlas_roi]
            roi_electrode_coords[roi] = {
                (int(s), int(c)): (float(row['MNI_x']),
                                   float(row['MNI_y']),
                                   float(row['MNI_z']))
                for (s, c), row in sub_tbl.iterrows()
            }
    print(f"Loaded summary_df ({len(summary_df)} rows) and "
          f"summary_combo_df ({len(summary_combo_df)} rows) from {OUT_DIR}.")


# ── BH-FDR correction over the confirmatory family ───────────────────
def benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values (q-values).

    Returns an array the same length as `pvals`; NaN inputs stay NaN.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]   # step-up monotone
    q_ok = np.empty(n)
    q_ok[order] = np.clip(ranked, 0.0, 1.0)
    q[ok] = q_ok
    return q


_fam_cols = {'test', 'combo', 'sub_model', 'p_perm'}
if not summary_combo_df.empty and _fam_cols.issubset(summary_combo_df.columns):
    # Flag the confirmatory family and BH-correct only those p_perm values.
    summary_combo_df['in_fdr_family'] = (
        summary_combo_df['test'].eq(FDR_TEST)
        & summary_combo_df['combo'].isin(FDR_COMBOS)
        & summary_combo_df['sub_model'].isin(FDR_SUBMODELS)
    )
    summary_combo_df['p_fdr'] = np.nan
    _fam = summary_combo_df['in_fdr_family']
    if _fam.any():
        summary_combo_df.loc[_fam, 'p_fdr'] = benjamini_hochberg(
            summary_combo_df.loc[_fam, 'p_perm'].to_numpy())
        _fam_df = (summary_combo_df.loc[_fam,
                   ['roi', 'combo', 'sub_model', 'n_neurons',
                    'beta', 'p_perm', 'p_fdr']]
                   .sort_values(['combo', 'p_fdr'])
                   .reset_index(drop=True))
        n_sig = int((_fam_df['p_fdr'] < FDR_ALPHA).sum())
        print(f"\n=== BH-FDR confirmatory family "
              f"(test={FDR_TEST}, combos={FDR_COMBOS}, "
              f"effect={FDR_SUBMODELS}) ===")
        print(f"{len(_fam_df)} tests; {n_sig} significant at q < {FDR_ALPHA}")
        with pd.option_context('display.max_rows', None,
                               'display.width', 160):
            print(_fam_df.to_string(index=False))
        _fam_df.to_csv(os.path.join(OUT_DIR, 'confirmatory_fdr.csv'),
                       index=False)
    else:
        print("\nBH-FDR: confirmatory family is empty — check the FDR_* "
              "settings against the combo / test names actually run.")
    # Re-save the combo table so it carries p_fdr / in_fdr_family.
    summary_combo_df.to_csv(
        os.path.join(OUT_DIR, 'results_summary_combos.csv'), index=False)
else:
    print("\nBH-FDR: no combo results available to correct.")


# ── BH-FDR per combo, across ROIs, for each DSR sub-model ────────────
# Independent of the single-family confirmatory block above. For every
# (combo × DSR sub-model) at FDR_TEST, BH-FDR the 7 ROI p_perm values
# separately so each combo gets its own q-val per ROI. Writes
# `confirmatory_fdr_per_combo.csv` and adds a `q_fdr_per_combo` column to
# `results_summary_combos.csv`. Reload-friendly: requires only the saved
# summary_combo_df to exist.
if not summary_combo_df.empty and _fam_cols.issubset(summary_combo_df.columns):
    # Per-combo BH-FDR: previously restricted to DSR sub-models. Now applied
    # to EVERY sub-model in a combo (state / location / bttn_* / dsr_*) so
    # that `q_fdr_per_combo` is populated across the full row set. FDR
    # scope stays the same: 7 ROIs within each (combo × sub_model × test).
    _test_mask = summary_combo_df['test'].eq(FDR_TEST)
    _target = summary_combo_df[_test_mask].copy()
    summary_combo_df['q_fdr_per_combo'] = np.nan
    per_combo_rows = []
    if not _target.empty:
        for (combo, submodel), grp in _target.groupby(['combo', 'sub_model'],
                                                        sort=False):
            qs = benjamini_hochberg(grp['p_perm'].to_numpy())
            summary_combo_df.loc[grp.index, 'q_fdr_per_combo'] = qs
            fam_df = grp[['roi', 'combo', 'sub_model', 'n_neurons',
                           'beta', 't', 'p_perm']].copy()
            fam_df['q_fdr'] = qs
            fam_df = fam_df.sort_values('p_perm').reset_index(drop=True)
            n_sig = int((fam_df['q_fdr'] < FDR_ALPHA).sum())
            print(f"\n=== Per-combo BH-FDR  combo={combo}  submodel={submodel}"
                  f"  test={FDR_TEST}  ===")
            print(f"  {len(fam_df)} ROIs; {n_sig} significant at q < {FDR_ALPHA}")
            with pd.option_context('display.max_rows', None,
                                   'display.width', 160):
                print(fam_df.round(4).to_string(index=False))
            per_combo_rows.append(fam_df)
        if per_combo_rows:
            pd.concat(per_combo_rows, ignore_index=True).to_csv(
                os.path.join(OUT_DIR, 'confirmatory_fdr_per_combo.csv'),
                index=False)
            print(f"\nSaved: {os.path.join(OUT_DIR, 'confirmatory_fdr_per_combo.csv')}")
        # Re-save the combo table with the new column.
        summary_combo_df.to_csv(
            os.path.join(OUT_DIR, 'results_summary_combos.csv'), index=False)
    else:
        print("\nPer-combo BH-FDR: no DSR sub-models present in "
              f"summary_combo_df at test={FDR_TEST!r}.")

def _render_overview_plots(summary_df, summary_combo_df,
                           roi_electrode_coords, out_dir,
                           models, combo_models,
                           heatmap_test='across_z'):
    """Render the pivot tables, per-ROI heatmaps and electrode glass-brain.

    Pulled out into a function so both the fresh-run and the
    reload-only paths can produce the same set of overview figures.
    """
    # Pivoted view: beta per (roi, test) x model
    if not summary_df.empty:
        pivot_beta = summary_df.pivot_table(
            index=['roi', 'test'], columns='model', values='beta'
        )
        print("\n=== Beta overview (rows: ROI x test, cols: model) ===")
        print(pivot_beta.to_string())

        pivot_pperm = summary_df.pivot_table(
            index=['roi', 'test'], columns='model', values='p_perm'
        )
        print("\n=== Permutation p-value overview ===")
        print(pivot_pperm.to_string())

    # base models heatmap (only if any base models were evaluated)
    from mc.plotting.cell_results import (
        plot_roi_model_heatmap, CANONICAL_ROI_ORDER,
        CANONICAL_RSA_MODEL_ORDER,
    )
    if len(models) > 0 and not summary_df.empty:
        sub_df_base = summary_df[summary_df['test'] == heatmap_test].copy()
        if sub_df_base.empty:
            print(f"No rows for test={heatmap_test}; skipping base-model heatmap.")
        else:
            fig_h, ax_h = plot_roi_model_heatmap(
                sub_df_base,
                models=CANONICAL_RSA_MODEL_ORDER,
                rois=ROIS_TO_ANALYZE,
                value_col='beta', annot_col='p_perm', sig_col='p_perm',
                n_col='n_neurons', alpha=0.05,
                value_label='empirical beta',
                title=f'ROIs x base models — {heatmap_test}',
                save_path=os.path.join(
                    out_dir, f'heatmap_roi_models_{heatmap_test}.png'),
                base_fontsize=15,
            )
            if fig_h is not None:
                plt.show()
    else:
        print("No base models evaluated this run — skipping base-model heatmap.")

    # combo models heatmap
    if not summary_combo_df.empty:
        for combo_key, sub_models in combo_models.items():
            sub_df = summary_combo_df[
                (summary_combo_df['combo'] == combo_key)
                & (summary_combo_df['test'] == heatmap_test)
            ].copy()
            if sub_df.empty:
                continue
            sub_df['model'] = sub_df['sub_model']

            fig_hc, ax_hc = plot_roi_model_heatmap(
                sub_df,
                models=sub_models,
                rois=ROIS_TO_ANALYZE,
                value_col='beta', annot_col='p_perm', sig_col='p_perm',
                n_col='n_neurons', alpha=0.05,
                value_label='empirical beta',
                title=f'ROIs x {combo_key} sub-models — {heatmap_test}',
                save_path=os.path.join(
                    out_dir,
                    f'heatmap_roi_combo_{combo_key}_{heatmap_test}.png'),
                base_fontsize=15,
            )
            if fig_hc is not None:
                plt.show()

    # ROI electrode schematic
    if PLOT_GLASSBRAINS == True:
        #import pdb; pdb.set_trace()
        from mc.plotting.cell_results import (
            plot_roi_electrodes_glassbrain, plot_roi_beta_glassbrain,
        )
        electrodes_per_roi = {
            roi: np.array(list(coords.values()), dtype=float)
            for roi, coords in roi_electrode_coords.items()
            if coords
        }
        if electrodes_per_roi:
            plot_roi_electrodes_glassbrain(
                electrodes_per_roi,
                save_path=os.path.join(out_dir, 'roi_electrodes_glassbrain.png'),
                title='ROI electrode locations (one panel per ROI)',
                per_roi_panels=True,
                roi_colors=ROI_COLOURS,
            )
            plot_roi_electrodes_glassbrain(
                electrodes_per_roi,
                save_path=os.path.join(
                    out_dir, 'roi_electrodes_glassbrain_combined.png'),
                title='ROI electrode locations (all ROIs combined)',
                per_roi_panels=False,
                roi_colors=ROI_COLOURS,
            )
        else:
            print("No electrode coordinates collected — skipping ROI glass-brain.")

        # ── ROI-shaded glass-brain (heatmap colours on a brain) ──────────
        # Glass-brains are produced ONLY for the FDR-family combo (one per
        # sub-model). Single-model results are shown via the ROI x model
        # heatmap above; per-model glass-brains have been retired. Other
        # combo variants are inspected via the heatmaps only.
        rois_with_cells = sorted(electrodes_per_roi)
        if rois_with_cells and not summary_combo_df.empty:
            # t-vals
            glassbrain_tval_dir = os.path.join(out_dir, 'roi_t-vals_glassbrains')
            os.makedirs(glassbrain_tval_dir, exist_ok=True)
            sub_t = summary_combo_df[summary_combo_df['test'] == heatmap_test]
            for combo_key in FDR_COMBOS:
                if combo_key not in combo_models:
                    print(f"{combo_key} is not saved in {combo_models}")
                    import pdb; pdb.set_trace()
                    continue
                sub_models = combo_models[combo_key]
                for sm in sub_models:
                    print(f"Now plotting t-val glassbrain for {sm} in {combo_key}")
                    rows = sub_t[(sub_t['combo'] == combo_key)
                                 & (sub_t['sub_model'] == sm)]
                    if rows.empty:
                        print("rows were empty")
                        continue
                    # The new ROI table and atlas helper both use canonical
                    # labels (mPFC, mOFC, PHC, ...).
                    t_vals = {_atlas_roi_name(roi): beta for roi, beta in zip(rows['roi'], rows['t'])}
                    if 'q_fdr_per_combo' not in rows.columns:
                        raise ValueError(
                            'Glassbrain significance outlines require '
                            'q_fdr_per_combo; run the per-combo FDR step first.')
                    qvals = {
                        _atlas_roi_name(roi): qval
                        for roi, qval in zip(rows['roi'],
                                             rows['q_fdr_per_combo'])
                    }
                    atlas_electrodes = {
                        _atlas_roi_name(roi): coords
                        for roi, coords in electrodes_per_roi.items()
                    }
                    glassbrain_stem_t = os.path.join(
                        glassbrain_tval_dir,
                        f'roi_tval_glassbrain_{combo_key}_{sm}_{heatmap_test}')
                    fig_gb = plot_roi_beta_glassbrain(
                        roi_betas=t_vals, roi_pvals=qvals,
                        only_rois=list(atlas_electrodes),
                        roi_cell_coords=atlas_electrodes,
                        roi_label_column=ROI_LABEL_COLUMN,
                        roi_display_names=ROI_DISPLAY_NAMES,
                        significance_label='q_FDR (per combo)',
                        title=f'{combo_key} | {sm} t-val — {heatmap_test} (FDR family)',
                        save_path=f'{glassbrain_stem_t}.png', is_t_val=True
                    )
                    print(f"saved in roi_tval_glassbrain_{combo_key}_{sm}_{heatmap_test}")
                    # Preserve a vector version of the DSR glassbrains for
                    # figures/manuscripts; the other regressors stay PNG-only.
                    if fig_gb is not None and sm.startswith('dsr') or sm.startswith('location'):
                        pdf_path = f'{glassbrain_stem_t}.pdf'
                        fig_gb.savefig(pdf_path, bbox_inches='tight')
                        #print(f"  [roi-glassbrain_stem_t-glassbrain] saved → {pdf_path}")

                    
            # BETA
            glassbrain_dir = os.path.join(out_dir, 'roi_beta_glassbrains')
            os.makedirs(glassbrain_dir, exist_ok=True)
            sub_t = summary_combo_df[summary_combo_df['test'] == heatmap_test]
            for combo_key in FDR_COMBOS:
                if combo_key not in combo_models:
                    continue
                sub_models = combo_models[combo_key]
                for sm in sub_models:
                    rows = sub_t[(sub_t['combo'] == combo_key)
                                 & (sub_t['sub_model'] == sm)]
                    if rows.empty:
                        continue
                    # The new ROI table and atlas helper both use canonical
                    # labels (mPFC, mOFC, PHC, ...).
                    betas = {
                        _atlas_roi_name(roi): beta
                        for roi, beta in zip(rows['roi'], rows['beta'])
                    }
                    if 'q_fdr_per_combo' not in rows.columns:
                        raise ValueError(
                            'Glassbrain significance outlines require '
                            'q_fdr_per_combo; run the per-combo FDR step first.')
                    qvals = {
                        _atlas_roi_name(roi): qval
                        for roi, qval in zip(rows['roi'],
                                             rows['q_fdr_per_combo'])
                    }
                    atlas_electrodes = {
                        _atlas_roi_name(roi): coords
                        for roi, coords in electrodes_per_roi.items()
                    }
                    glassbrain_stem = os.path.join(
                        glassbrain_dir,
                        f'roi_beta_glassbrain_{combo_key}_{sm}_{heatmap_test}')
                    fig_gb = plot_roi_beta_glassbrain(
                        roi_betas=betas, roi_pvals=qvals,
                        only_rois=list(atlas_electrodes),
                        roi_cell_coords=atlas_electrodes,
                        roi_label_column=ROI_LABEL_COLUMN,
                        roi_display_names=ROI_DISPLAY_NAMES,
                        significance_label='q_FDR (per combo)',
                        title=f'{combo_key} | {sm} beta — {heatmap_test} (FDR family)',
                        save_path=f'{glassbrain_stem}.png',
                    )
                    # Preserve a vector version of the DSR glassbrains for
                    # figures/manuscripts; the other regressors stay PNG-only.
                    if fig_gb is not None and sm.startswith('dsr'):
                        pdf_path = f'{glassbrain_stem}.pdf'
                        fig_gb.savefig(pdf_path, bbox_inches='tight')
                        print(f"  [roi-beta-glassbrain] saved → {pdf_path}")
                    plt.show()
                


# ── Cross-ROI heatmap (rows=ROI, cols=model) ─────────────────────────────
# Heatmap rendering now lives in mc.plotting.cell_results
# (plot_roi_model_heatmap), shared with encoding_analysis_simple.py so
# fonts, ROI/model order and significance styling stay in sync.


HEATMAP_TEST = 'split_halves_z'   # across-runs cross-corr, z-scored
                                  # (= FDR primary variant). One of:
                                  # split_halves, split_halves_z,
                                  # within, within_z, between_tasks,
                                  # between_tasks_z

# Save the per-ROI electrode coordinates so that the reload path can
# reproduce the schematic glass-brain without rerunning the analysis.
_coord_rows = [
    {'roi': roi, 'subject': sub_int, 'cell_idx': cell_int,
     'MNI_x': x, 'MNI_y': y, 'MNI_z': z}
    for roi, d in roi_electrode_coords.items()
    for (sub_int, cell_int), (x, y, z) in d.items()
]
if _coord_rows:
    pd.DataFrame(_coord_rows).to_csv(
        os.path.join(OUT_DIR, 'roi_electrode_coords.csv'), index=False
    )

_render_overview_plots(
    summary_df, summary_combo_df, roi_electrode_coords, OUT_DIR,
    models=models, combo_models=combo_models, heatmap_test=HEATMAP_TEST,
)

# ── Publication-ready ROI × regressor summary figure ─────────────────
# Delegates to `scripts/RSA_DSR_replot_addon.make_publication_figures`,
# which is the same function the standalone re-plot uses — so a fresh
# run and a `python RSA_DSR_replot_addon.py <run_dir>` produce identical
# outputs. Reads results_summary_combos.csv + perm_null_draws/perm_*.pkl,
# no recomputation. Falls back silently if the addon script is missing.
try:
    _scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from RSA_DSR_replot_addon import make_publication_figures
    make_publication_figures(OUT_DIR, test_variant=HEATMAP_TEST)
except Exception as _pub_exc:
    print(f"[pub-figs] skipped ({type(_pub_exc).__name__}: {_pub_exc})")
