#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is a cell's abstract-state code SUSTAINED, or is it rhythmic?

A cell can look state-selective in two very different ways, and the
distinction decides what such a code could support. It may hold its rate
elevated for the whole of its preferred state — a stable index of task
position. Or it may fire rhythmically within that state, so its mean is
elevated while nothing about the moment-to-moment signal says which state
it is. Only the first can serve as a persistent index. This script
separates them, per cell, entirely on held-out data.

THE MODEL (fit on training configurations)
------------------------------------------
    y = Σ_s β_s · I[state=s]                       (4 one-hot columns)
      + Σ_{s,j} β_{sj} · I[state=s] · P_j          (4 × 2 sum-coded)
      + Σ_c α_c · I[config=c]                      (K_train − 1 intercepts)

`P_j` are SUM-CODED phase indicators (+1 for phase j, −1 for the
reference phase). There is deliberately NO separate phase main-effect
column: the sum-coded interactions absorb it, and adding one would make
the design rank-deficient. Sum coding is what buys the decomposition —
β_state[s] is the marginal across-phase mean for state s, not a
phase-reference cell mean, so the state block and the interaction block
are orthogonal contributions:
    ŷ_state       = X_state       @ β_state        "sustained" component
    ŷ_interaction = X_interaction @ β_interaction  "phase-modulated"

Preferred state for a fold = argmax over s of mean(y_train | state=s),
i.e. the highest marginal training FIRING RATE, not the largest β. This
is deliberate: a marginal mean is robust to the coding choice.

HELD-OUT SCORING (on the left-out configuration)
------------------------------------------------
  r_state        Pearson r between y_test and ŷ_state.
                 Does the across-phase state code generalise?
  r_interaction  Pearson r between y_test and ŷ_interaction.
                 Does the within-state phase modulation generalise?
  min_phase_contrast   For each phase p ∈ {early, middle, late}:
                 contrast[p] = mean(y_test | preferred state, phase p)
                             − mean(y_test | other states,   phase p)
                 averaged over folds; the statistic is the MINIMUM over
                 the three phases. A cell only scores high if its
                 preferred state beats the others at EVERY phase — which
                 is exactly what "sustained" means.

*** WHY min_phase_contrast IS NOT TESTED AGAINST ZERO ***
    The minimum of three noisy estimates is biased downward, so its null
    does not sit at zero — empirically the ROI means are NEGATIVE in
    every region. A one-sample t-test against 0 is therefore invalid here
    and will look like a null result no matter what the data say. Each
    cell is instead tested against ITS OWN permutation null, and the ROI
    is tested by asking whether that ROI's per-cell permutation-p values
    are shifted below 0.5 (Wilcoxon). Under H0 a permutation-p is
    Uniform[0,1] whatever the bias in the underlying statistic, so the
    population test is bias-free. Do not quote `t_min_phase_gt0`.

PERMUTATION NULL
    Circular shifts of the held-out trace y_test, N_PERMUTATIONS per
    fold, the SAME shifts reused for every statistic so the three are
    on a common null.

CELL LABELS
    sustained    p_perm(min_phase_contrast) < SIG_ALPHA
    any-state    p_perm(r_state)            < SIG_ALPHA
    phasic-only  sig_r_interaction AND NOT sig_r_state
    BH-FDR within ROI across cells gives the `*_fdr` variants.

ROI-LEVEL INFERENCE
    Per-ROI binomial of the sustained / phasic-only fraction vs the 5%
    chance rate; a χ² omnibus of homogeneity across ROIs; and a planned
    one-sided EC-vs-pooled-rest Fisher exact test (EC is singled out
    a priori by the fMRI result). All three land in
    diagnostic_figures/roi_inference.json.

    The population-shift tests (Wilcoxon of per-cell perm-p vs 0.5) are
    corrected across ROIs BOTH ways and stored in the ROI summary:
        q_wilcoxon_<stat>    BH-FDR       (less conservative)
        fwe_wilcoxon_<stat>  Bonferroni   (drives the figure stars)
    for <stat> ∈ {min_phase, r_state, r_interaction}.

NOTE ON ROI COUNT: ROI_ORDER lists 7, but PHC has no cells in the current
table, so runs produce 6 ROIs and every across-ROI correction is over 6.

OUTPUTS  (DATA_DIR/group/encoding_state_sustained_cv/<run_tag>/)
    state_sustained_cv_results.csv       per cell: statistics, perm-p,
                                         within-ROI q, labels
    state_sustained_cv_roi_summary.csv   per ROI: fractions, binomials,
                                         Wilcoxon p / q / FWE
    config.json
    diagnostic_figures/
        01_design_matrix              02_pipeline_run_stats
        03_perm_nulls_examples        05_dist_min_phase_contrast_per_roi
        06_dist_r_state_per_roi       06b_dist_r_interaction_per_roi
        07_per_phase_by_category      08_roi_fractions_overview
        08b_roi_fractions_grouped     10_pref_state_stacked
        11_effect_size_heatmap        captions.md
        roi_inference.json
"""
# import pdb; pdb.set_trace()
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
import textwrap
import hashlib
from datetime import datetime
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "multiple_clocks_mplconfig"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo")
import era_brewer
import mc.analyse.cell_selection as cell_selection
import mc.analyse.helpers_human_cells as hh
from mc.plotting.cell_results import plot_state_polar_clock, smooth_circular


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
DATA_DIR = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
OUT_BASE = os.path.join(DATA_DIR, "group", "encoding_state_sustained_cv")
# Canonical, current MNI-derived cell-to-ROI table.  Do not rely on
# ``cell_selection``'s default table here: that default is the older
# ``neurons_with_final_roi_labels.csv`` file.
ROI_TABLE_PATH = os.path.join(DATA_DIR, "neurons_with_ROI_labels.csv")

# FLAG ADDED — reload a previous run's per-cell CSV instead of recomputing.
# 2026-09-24: None — the balance statistic needs the per-permutation
# r_state and r_interaction values, which the reload path cannot
# recover from a saved per-cell CSV. Full re-run required. The run this
# supersedes (same cells, older ROI labels) was
#   RELOAD_OLD_RESULTS = '2026-08-17_08-41-10'
RELOAD_OLD_RESULTS = '2026-09-24_18-35-23'

# Reload-time ROI relabelling. Point at the fresh neurons ROI table (with
# an `alt_final_roi` column) to overwrite the per-cell `roi` column on
# reload; the ROI summary + figures are rewritten into a NEW sibling
# directory `<original>_relabelled_<timestamp>/`. Set to None to skip.
# Equivalent to passing `--relabel-from <path>` on the CLI.
RELABEL_FROM = None
# RELABEL_FROM = None

# RSA companion analysis — provides the 4th column of fig 11 (state RSA).
# Change the date here when re-running with a new RSA run.
# RSA companion run. Updated 2026-09-24 to the first run that POSTDATES the
# 2026-08-28 ROI-label update (the previous pointer, 2026-07-30_15-58-51, used
# the old labels). The combo was also renamed and redefined between the two:
# old `ctrl_dsrFULL` = [state, location, bttn_curr, dsr_fmri];
# new `ctrl_fMRI-state_dsrFULL` = [location, bttn_curr, bttn_next, state,
# dsr_fmri, l2_norm], i.e. it adds the FUTURE-action and spatial-similarity
# controls. The `-noL2` variant drops l2_norm and gives the same betas to
# three decimals.
RSA_STATE_RUN_DIR  = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data/'
                       'ephys_humans/derivatives/group/DSR_RSA_simple_ROI/2026-08-31_17-57-30')


RSA_TEST_VARIANT = 'split_halves_z'   # all-correct-repeats averaged per
                                        # config, z-scored per neuron,
                                        # between-task-config cells only.
RSA_SUB_MODEL    = 'state'
COMBO_MODEL_NAME = 'ctrl_fMRI-state_dsrFULL'

N_BINS_PER_TRIAL = 360
N_STATES = 4
N_PHASES = 3
STATE_LEN = N_BINS_PER_TRIAL // N_STATES
PHASE_LEN = STATE_LEN // N_PHASES
STATES = ["A", "B", "C", "D"]
PHASES = ["early", "middle", "late"]
PHASE_REF = 2

# 10,000, not 1,000: the per-cell p is thresholded at .05 and then counted
# across cells, so Monte-Carlo error (~.007 at p=.05 with 1,000 draws)
# flips borderline cells and moves the ROI fractions. Runtime was 93 s
# at 1,000, so this stays cheap.
N_PERMUTATIONS = 10000
N_JOBS = -1
SIG_ALPHA = 0.05

ROI_LABEL_COLUMN = "alt_final_roi"
TARGET_ROIS = None
SUBJECTS_TO_RUN = "all"

# Names in the current table's ``alt_final_roi`` column.
ROI_ORDER = ["mPFC", "mOFC", "PCC", "PHC",
             "HC_anterior", "HC_mid", "EC"]

DPI = 300                    # publication-quality export
CM = 1.0 / 2.54              # inches per cm — A4 sizing helper
FONT_BIG    = 11             # main titles
FONT_AXIS   = 10             # axis labels
FONT_TICK   = 9              # tick labels — project-wide MINIMUM 
FONT_SMALL  = 9              # annotations — bumped from 8 to honour the
                              # 9 pt floor required
# Back-compat shim (some old call sites still reference FONT_SIZE).
FONT_SIZE   = FONT_BIG

# State colours (orange / yellow / light purple / dark purple — see user ref image #1).
STATE_COLOURS = {
    'A': '#F15A29',
    'B': '#F7931E',
    'C': '#C7C6E2',
    'D': '#6B60AA',
}
# Phase colours: pastel pink → pink → bordeaux (early → middle → late).
PHASE_COLOURS = {
    'early':  '#FCDDE3',
    'middle': '#D7657F',
    'late':   '#5C1027',
}
# Index-form lists (legacy positional access).
COLOR_PALETTE  = era_brewer.era_brew("Showgirl2", n=7)
STATE_COLORS   = [STATE_COLOURS[s] for s in ['A', 'B', 'C', 'D']]
PHASE_COLORS   = [PHASE_COLOURS[p] for p in ['early', 'middle', 'late']]

# ROI colours — one stable hue per ROI, drawn from era_brewer 'Showgirl2'.
# Anywhere an ROI is the categorical variable, use these. Sustained-state
# project convention
_roi_palette_src = era_brewer.era_brew("Showgirl2", n=7)

ROI_COLOURS = {                 # matches CLAUDE.md `roi_colour_dict`
    'EC':          _roi_palette_src[0],
    'mPFC':        _roi_palette_src[1],
    'HC_anterior': _roi_palette_src[2],
    'PCC':         _roi_palette_src[3],
    'mOFC':        _roi_palette_src[4],
    'HC_mid':      '#a30d6c',   # magenta (CLAUDE.md override)
    'PHC':         '#23677E',   # teal    (CLAUDE.md override)
}
# Location colours (3 × 3 grid, dark teal top-left → light green bottom-right).
LOCATION_COLOURS = {
    1: '#0a607a',  2: '#7eb1c4',  3: '#b6d4e0',
    4: '#175e62',  5: '#5b9b8d',  6: '#c8e0d0',
    7: '#0e3d3a',  8: '#3d8b7d',  9: '#a7d9b2',
}
# Dark green for "observed value" markers on perm distributions (matches grid).
OBSERVED_GREEN = '#0e3d3a'


# ---------------------------------------------------------------------
# Bin-axis helpers
# ---------------------------------------------------------------------
def state_phase_labels(n_bins=N_BINS_PER_TRIAL):
    t = np.arange(n_bins)
    return (t // STATE_LEN).astype(int), ((t % STATE_LEN) // PHASE_LEN).astype(int)


STATE_IDX_360, PHASE_IDX_360 = state_phase_labels()


def cell_seed(*parts):
    """Deterministic per-cell permutation seed.

    The previous seed was built with Python's builtin str hash. Python 3 salts
    str hashing per process unless PYTHONHASHSEED is set, so EVERY RUN DREW
    DIFFERENT PERMUTATIONS and the analysis was not reproducible. Measured
    here: the same tuple gave 3855842339, 44291736 and 2509367741 in three
    consecutive interpreters. The consequence was material -- the planned
    EC-vs-rest Fisher test moved from P = 0.054 to P = 0.023 between two runs
    over the identical 924 cells, purely from permutation noise.

    blake2b is stable across processes, machines and Python versions.
    """
    key = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def make_circular_shifts(y, n_perms, rng):
    T = y.shape[0]
    shifts = rng.integers(0, T, size=n_perms)
    idx = (np.arange(T)[None, :] - shifts[:, None]) % T
    return y[idx]


def nan_safe_pearsonr(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2: return np.nan
    xm, ym = x[m], y[m]
    if np.std(xm) < 1e-12 or np.std(ym) < 1e-12: return np.nan
    return float(np.corrcoef(xm, ym)[0, 1])


def vectorized_pearsonr(Y, y):
    """Correlate each row of Y (n_perm, n_bins) with y (n_bins,)."""
    Y = np.asarray(Y, dtype=float); y = np.asarray(y, dtype=float)
    Y_c = Y - Y.mean(axis=1, keepdims=True)
    y_c = y - y.mean()
    num = Y_c @ y_c
    denom = np.sqrt((Y_c ** 2).sum(axis=1) * (y_c ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 1e-12, num / denom, np.nan)


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    good = np.isfinite(p)
    if not good.any(): return out
    p_good = p[good]
    order = np.argsort(p_good)
    ranked = p_good[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out_good = np.empty_like(q); out_good[order] = q
    out[good] = out_good
    return out


def one_sided_ttest_greater(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2: return np.nan, np.nan
    try:
        res = stats.ttest_1samp(x, 0.0, alternative="greater")
        return float(res.statistic), float(res.pvalue)
    except TypeError:
        t, p_two = stats.ttest_1samp(x, 0.0)
        return float(t), float(p_two / 2 if t > 0 else 1 - p_two / 2)


def _binom_vs_alpha(n_sig, n_total):
    if n_total == 0: return np.nan
    try:
        return stats.binomtest(n_sig, n_total, p=SIG_ALPHA, alternative="greater").pvalue
    except AttributeError:
        return stats.binom_test(n_sig, n_total, p=SIG_ALPHA, alternative="greater")


def parse_subjects(spec):
    if spec == "all":
        return [f"{i:02d}" for i in range(1, 64)]
    if spec == "dsr_subs":
        return cell_selection.load_rsa_subjects(data_dir=DATA_DIR)
    return [s.strip().zfill(2) for s in spec.split(",") if s.strip()]


def add_config_str(beh):
    beh = beh.copy().reset_index(drop=True)
    beh["config_str"] = (
        beh["loc_A"].astype(int).astype(str) + "-"
        + beh["loc_B"].astype(int).astype(str) + "-"
        + beh["loc_C"].astype(int).astype(str) + "-"
        + beh["loc_D"].astype(int).astype(str)
    )
    return beh


def _state_dict(arr):
    arr = np.asarray(arr, dtype=float)
    return {STATES[i]: (None if not np.isfinite(arr[i]) else float(arr[i])) for i in range(N_STATES)}


# ---------------------------------------------------------------------
# Trace per (cell, config)
# ---------------------------------------------------------------------
def build_y_per_config(beh, neurons, configs):
    config_labels, per_cfg_y = [], {n_lab: [] for n_lab in neurons}
    for c in configs:
        idx = beh.index[(beh["config_str"] == c) & (beh["correct"] == 1)].to_numpy()
        if len(idx) == 0: continue
        config_labels.append(c)
        for n_lab, df in neurons.items():
            arr = df.iloc[idx].to_numpy(dtype=float)
            per_cfg_y[n_lab].append(np.nanmean(arr, axis=0))
    Y = {n_lab: np.array(per_cfg_y[n_lab]) for n_lab in neurons}
    return Y, config_labels, STATE_IDX_360.copy(), PHASE_IDX_360.copy()


# ---------------------------------------------------------------------
# Design construction
# ---------------------------------------------------------------------
def build_design(state_idx, phase_idx, config_idx, ref_config_ids,
                 ref_phase=PHASE_REF):
    """Sum-coded GLM design (clean state vs state×phase split).

    Columns: 4 state one-hot   |  8 state × sum-coded-phase interactions
             | (K_train - 1) per-config intercepts.

    Sum coding for phase makes β_state[s] the marginal across-phase mean
    for state s (NOT a phase-reference cell mean). β_interaction captures
    the phase-specific deviations. The two predictions
        ŷ_state       = X_state       @ β_state
        ŷ_interaction = X_interaction @ β_interaction
    are then orthogonal contributions to the model — perfect for separating
    sustained-state generalisation (r_state) from phasic generalisation
    (r_interaction).

    No separate phase main effect column: the sum-coded interactions absorb
    it (and a separate phase main column would create rank deficiency since
    sum of interaction cols equals the phase sum-coded column).
    """
    n_bins = len(state_idx); slices = {}; offset = 0

    # State one-hot (4 cols)
    X_state = np.zeros((n_bins, N_STATES))
    for s in range(N_STATES):
        X_state[:, s] = (state_idx == s).astype(float)
    slices['state'] = slice(offset, offset + N_STATES); offset += N_STATES

    # Sum-coded phase indicators (used only to build interactions)
    phase_cols = [p for p in range(N_PHASES) if p != ref_phase]
    phase_sum = np.zeros((n_bins, len(phase_cols)))
    for j, p in enumerate(phase_cols):
        phase_sum[phase_idx == p, j] = 1.0
        phase_sum[phase_idx == ref_phase, j] = -1.0

    # state × sum-coded phase interactions (N_STATES × (N_PHASES - 1) cols)
    X_inter = np.zeros((n_bins, N_STATES * len(phase_cols)))
    k = 0
    for s in range(N_STATES):
        for j in range(len(phase_cols)):
            X_inter[:, k] = X_state[:, s] * phase_sum[:, j]
            k += 1
    slices['interaction'] = slice(offset, offset + X_inter.shape[1]); offset += X_inter.shape[1]
    # Phase slice kept empty for back-compat (no separate phase main effect).
    slices['phase'] = slice(offset, offset)

    # Per-train-config intercepts (drop one as reference)
    sorted_ids = sorted(ref_config_ids)
    cfg_cols = sorted_ids[1:] if len(sorted_ids) > 1 else []
    X_cfg = np.zeros((n_bins, len(cfg_cols)))
    for j, c in enumerate(cfg_cols):
        X_cfg[:, j] = (config_idx == c).astype(float)
    slices['config'] = slice(offset, offset + len(cfg_cols)); offset += len(cfg_cols)

    return np.hstack([X_state, X_inter, X_cfg]), slices


def ols_fit(X, y):
    keep = np.isfinite(y)
    if keep.sum() < X.shape[1] + 1: return None
    Xk, yk = X[keep], y[keep]
    beta, *_ = np.linalg.lstsq(Xk, yk, rcond=None)
    rss = float(((yk - Xk @ beta) ** 2).sum())
    df_resid = int(yk.size - Xk.shape[1])
    return beta, rss, df_resid


# ---------------------------------------------------------------------
# Per-cell analysis (TWO tests)
# ---------------------------------------------------------------------
def analyse_one_neuron(neuron_id, roi, y_per_cfg,
                       state_idx_full, phase_idx_full, config_idx_full,
                       all_config_ids, n_permutations, seed,
                       return_perm_dists=False):
    rng = np.random.default_rng(seed)
    n_folds = y_per_cfg.shape[0]
    bin_per_fold = N_BINS_PER_TRIAL
    y_full = y_per_cfg.reshape(-1)

    contrast_per_fold = np.full((n_folds, N_PHASES), np.nan)
    pref_state_per_fold = np.full(n_folds, -1, dtype=int)
    r_state_per_fold = np.full(n_folds, np.nan)
    r_interaction_per_fold = np.full(n_folds, np.nan)
    r_full_per_fold = np.full(n_folds, np.nan)
    beta_state_per_fold = np.full((n_folds, N_STATES), np.nan)
    perm_contrast_per_fold = np.full((n_permutations, n_folds, N_PHASES), np.nan)
    perm_r_state_per_fold = np.full((n_permutations, n_folds), np.nan)
    perm_r_interaction_per_fold = np.full((n_permutations, n_folds), np.nan)

    for fold_idx in range(n_folds):
        train_cfg_ids = [c for c in all_config_ids if c != fold_idx]
        X_fold, slices = build_design(
            state_idx_full, phase_idx_full, config_idx_full, train_cfg_ids
        )
        test_start = fold_idx * bin_per_fold
        test_end = test_start + bin_per_fold
        train_mask = np.ones(len(y_full), dtype=bool)
        train_mask[test_start:test_end] = False

        # Preferred state = argmax of training marginal state mean (robust)
        s_idx_tr = state_idx_full[train_mask]
        y_tr = y_full[train_mask]
        state_means = np.array([
            np.nanmean(y_tr[s_idx_tr == s]) if (s_idx_tr == s).any() else np.nan
            for s in range(N_STATES)
        ])
        if not np.isfinite(state_means).any():
            continue
        pref_state = int(np.nanargmax(state_means))
        pref_state_per_fold[fold_idx] = pref_state

        # Full OLS fit on training
        full = ols_fit(X_fold[train_mask], y_tr)
        if full is None: continue
        beta_full, _, _ = full
        beta_state_per_fold[fold_idx] = beta_full[slices['state']]

        beta_state = beta_full[slices['state']]
        beta_interaction = beta_full[slices['interaction']]

        # Held-out encoding predictions
        X_test = X_fold[test_start:test_end]
        y_test = y_full[test_start:test_end]
        y_pred_state = X_test[:, slices['state']] @ beta_state                 # marginal state
        y_pred_interaction = X_test[:, slices['interaction']] @ beta_interaction # phase modulation
        y_pred_full = X_test @ beta_full

        valid = (np.isfinite(y_test) & np.isfinite(y_pred_state)
                 & np.isfinite(y_pred_interaction) & np.isfinite(y_pred_full))
        if valid.sum() < 10: continue

        r_state_per_fold[fold_idx]       = nan_safe_pearsonr(y_test[valid], y_pred_state[valid])
        r_interaction_per_fold[fold_idx] = nan_safe_pearsonr(y_test[valid], y_pred_interaction[valid])
        r_full_per_fold[fold_idx]        = nan_safe_pearsonr(y_test[valid], y_pred_full[valid])

        # Per-phase contrast for preferred state (sustained test)
        state_test = state_idx_full[test_start:test_end]
        phase_test = phase_idx_full[test_start:test_end]
        for p in range(N_PHASES):
            pref_bins = (state_test == pref_state) & (phase_test == p) & np.isfinite(y_test)
            other_bins = (state_test != pref_state) & (phase_test == p) & np.isfinite(y_test)
            if pref_bins.any() and other_bins.any():
                contrast_per_fold[fold_idx, p] = float(
                    np.nanmean(y_test[pref_bins]) - np.nanmean(y_test[other_bins])
                )

        # Permutations (same shifts for both stat sets)
        if n_permutations > 0:
            Y_shifted = make_circular_shifts(y_test, n_permutations, rng)

            # r_state and r_interaction perm — vectorised correlations
            Y_v = Y_shifted[:, valid]
            if np.std(y_pred_state[valid]) > 1e-12:
                perm_r_state_per_fold[:, fold_idx] = vectorized_pearsonr(
                    Y_v, y_pred_state[valid]
                )
            if np.std(y_pred_interaction[valid]) > 1e-12:
                perm_r_interaction_per_fold[:, fold_idx] = vectorized_pearsonr(
                    Y_v, y_pred_interaction[valid]
                )

            # phase-contrast perm
            for p in range(N_PHASES):
                pref_bins = (state_test == pref_state) & (phase_test == p)
                other_bins = (state_test != pref_state) & (phase_test == p)
                if pref_bins.any() and other_bins.any():
                    perm_contrast_per_fold[:, fold_idx, p] = (
                        np.nanmean(Y_shifted[:, pref_bins], axis=1) -
                        np.nanmean(Y_shifted[:, other_bins], axis=1)
                    )

    # Aggregate
    phase_contrast_mean = np.nanmean(contrast_per_fold, axis=0)
    min_phase_contrast = (float(np.nanmin(phase_contrast_mean))
                          if np.isfinite(phase_contrast_mean).any() else np.nan)
    r_state       = (float(np.nanmean(r_state_per_fold))
                     if np.isfinite(r_state_per_fold).any() else np.nan)
    r_interaction = (float(np.nanmean(r_interaction_per_fold))
                     if np.isfinite(r_interaction_per_fold).any() else np.nan)
    r_full        = (float(np.nanmean(r_full_per_fold))
                     if np.isfinite(r_full_per_fold).any() else np.nan)

    if n_permutations > 0:
        perm_phase_mean = np.nanmean(perm_contrast_per_fold, axis=1)  # (n_perm, 3)
        perm_min = np.nanmin(perm_phase_mean, axis=1)
        v_m = np.isfinite(perm_min)
        p_min = ((np.sum(perm_min[v_m] >= min_phase_contrast) + 1) / (v_m.sum() + 1)
                 if v_m.any() and np.isfinite(min_phase_contrast) else np.nan)

        perm_r_state = np.nanmean(perm_r_state_per_fold, axis=1)
        v_s = np.isfinite(perm_r_state)
        p_r_state = ((np.sum(perm_r_state[v_s] >= r_state) + 1) / (v_s.sum() + 1)
                     if v_s.any() and np.isfinite(r_state) else np.nan)

        perm_r_int = np.nanmean(perm_r_interaction_per_fold, axis=1)
        v_i = np.isfinite(perm_r_int)
        p_r_int = ((np.sum(perm_r_int[v_i] >= r_interaction) + 1) / (v_i.sum() + 1)
                   if v_i.any() and np.isfinite(r_interaction) else np.nan)

        # BALANCE — how much of a cell's held-out predictable signal is carried
        # by the phase-INVARIANT state component versus the phase-MODULATED one.
        #
        # r_state comes from a 4-column block and r_interaction from an
        # 8-column block, so their permutation nulls need not sit at the same
        # place and a raw difference could be biased by block size alone. Two
        # guards: (a) `null_mean_*` are stored so the bias can be inspected
        # directly, and (b) the p-value is computed against the permutation
        # distribution OF THE DIFFERENCE, built from the SAME circular shifts
        # already used for both components (`Y_shifted` is shared), so any
        # common bias cancels within each permutation exactly as it does in
        # the empirical value.
        perm_balance = perm_r_state - perm_r_int
        v_b = np.isfinite(perm_balance)
        balance = (r_state - r_interaction
                   if np.isfinite(r_state) and np.isfinite(r_interaction)
                   else np.nan)
        p_balance = ((np.sum(perm_balance[v_b] >= balance) + 1) / (v_b.sum() + 1)
                     if v_b.any() and np.isfinite(balance) else np.nan)
        null_mean_state = float(np.nanmean(perm_r_state)) if v_s.any() else np.nan
        null_mean_inter = float(np.nanmean(perm_r_int)) if v_i.any() else np.nan
    else:
        perm_min = np.array([]); perm_r_state = np.array([]); perm_r_int = np.array([])
        perm_balance = np.array([])
        p_min = np.nan; p_r_state = np.nan; p_r_int = np.nan
        balance = (r_state - r_interaction
                   if np.isfinite(r_state) and np.isfinite(r_interaction)
                   else np.nan)
        p_balance = np.nan
        null_mean_state = np.nan; null_mean_inter = np.nan

    pref_valid = pref_state_per_fold[pref_state_per_fold >= 0]
    if pref_valid.size:
        pref_counts = np.bincount(pref_valid, minlength=N_STATES)
        pref_state_mode = int(np.argmax(pref_counts))
    else:
        pref_counts = np.zeros(N_STATES, dtype=int); pref_state_mode = -1

    all_phase_positive = (bool(np.all(phase_contrast_mean > 0))
                          if np.isfinite(phase_contrast_mean).all() else False)

    row = {
        "neuron": neuron_id, "roi": roi,
        # PRIMARY — sustained gate: min phase contrast
        "min_phase_contrast": min_phase_contrast,
        "p_perm_min_phase_contrast": float(p_min) if np.isfinite(p_min) else np.nan,
        "phase_contrast_early":  float(phase_contrast_mean[0]) if np.isfinite(phase_contrast_mean[0]) else np.nan,
        "phase_contrast_middle": float(phase_contrast_mean[1]) if np.isfinite(phase_contrast_mean[1]) else np.nan,
        "phase_contrast_late":   float(phase_contrast_mean[2]) if np.isfinite(phase_contrast_mean[2]) else np.nan,
        "all_phase_positive": all_phase_positive,
        "sig_sustained": bool(np.isfinite(p_min) and p_min < SIG_ALPHA),
        # GENERALISATION — encoding correlations (sum-coded GLM)
        "r_state":               r_state,
        "p_perm_r_state":        float(p_r_state) if np.isfinite(p_r_state) else np.nan,
        "sig_r_state":           bool(np.isfinite(p_r_state) and p_r_state < SIG_ALPHA),
        "r_interaction":         r_interaction,
        "p_perm_r_interaction":  float(p_r_int) if np.isfinite(p_r_int) else np.nan,
        "sig_r_interaction":     bool(np.isfinite(p_r_int) and p_r_int < SIG_ALPHA),
        "r_full":                r_full,
        # BALANCE — phase-invariant minus phase-modulated generalisation
        "balance":               balance,
        "p_perm_balance":        float(p_balance) if np.isfinite(p_balance) else np.nan,
        "sig_balance":           bool(np.isfinite(p_balance) and p_balance < SIG_ALPHA),
        "null_mean_r_state":     null_mean_state,
        "null_mean_r_interaction": null_mean_inter,
        # Null-centred variant: each component shifted by its OWN null mean
        # before differencing. Reported alongside `balance` so the two can be
        # compared; if they disagree the block-size bias is real and material.
        "balance_null_centred":  (
            float((r_state - null_mean_state) - (r_interaction - null_mean_inter))
            if all(np.isfinite([r_state, r_interaction,
                                null_mean_state, null_mean_inter])) else np.nan),
        # Descriptors
        "pref_state_mode": STATES[pref_state_mode] if pref_state_mode >= 0 else "",
        "pref_state_counts_json": json.dumps({STATES[i]: int(pref_counts[i]) for i in range(N_STATES)}),
        "mean_beta_state_json": json.dumps(_state_dict(np.nanmean(beta_state_per_fold, axis=0))),
        "contrast_per_fold_json": json.dumps([
            {PHASES[p]: (None if not np.isfinite(contrast_per_fold[f, p]) else float(contrast_per_fold[f, p]))
             for p in range(N_PHASES)}
            for f in range(n_folds)
        ]),
        "n_permutations": int(n_permutations),
        "n_folds_used": int(np.isfinite(r_full_per_fold).sum()),
    }
    if return_perm_dists:
        row["_perm_min_phase"]   = perm_min
        row["_perm_r_state"]     = perm_r_state
        row["_perm_r_interaction"] = perm_r_int
        row["_contrast_per_fold"] = contrast_per_fold
        row["_beta_state_per_fold"] = beta_state_per_fold
    return row


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------
def add_fdr_columns(df):
    df = df.copy()
    for col_p, col_q, col_sig in [
        ("p_perm_min_phase_contrast", "q_min_phase_within_roi",    "sig_sustained_fdr"),
        ("p_perm_r_state",            "q_r_state_within_roi",      "sig_r_state_fdr"),
        ("p_perm_r_interaction",      "q_r_interaction_within_roi","sig_r_interaction_fdr"),
    ]:
        df[col_q] = np.nan
        for _, idx in df.groupby("roi").groups.items():
            idx = list(idx)
            df.loc[idx, col_q] = bh_fdr(df.loc[idx, col_p].to_numpy())
        df[col_sig] = df[col_q] < SIG_ALPHA
    return df


def overrepresented_state(pref_states):
    pref = [s for s in pref_states if s in STATES]
    n = len(pref)
    if n == 0: return "", 0, 0, np.nan
    counts = pd.Series(pref).value_counts()
    top = str(counts.idxmax()); top_n = int(counts.iloc[0])
    try:
        p = stats.binomtest(top_n, n, p=1 / N_STATES, alternative="greater").pvalue
    except AttributeError:
        p = stats.binom_test(top_n, n, p=1 / N_STATES, alternative="greater")
    return top, top_n, n, float(p)


def wilcoxon_perm_p_per_roi(results, p_col):
    """Per-ROI Wilcoxon signed-rank test of per-cell perm-p values vs. 0.5
    (one-sided, alternative='less'). Under H_0, per-cell perm-p ~ Uniform[0, 1]
    regardless of any bias in the underlying empirical statistic, so the
    population median = 0.5. A significant result means the population is
    shifted toward smaller perm-p, i.e. the empirical distribution is shifted
    in the predicted direction. Bias-free for the min-of-3 statistic.
    Returns dict roi → p-value (NaN if too few valid cells)."""
    out = {}
    for roi, g in results.groupby('roi'):
        p_vals = g[p_col].dropna().to_numpy()
        if p_vals.size < 5:
            out[roi] = np.nan; continue
        try:
            res = stats.wilcoxon(p_vals - 0.5, alternative='less')
            out[roi] = float(res.pvalue)
        except ValueError:
            out[roi] = np.nan
    return out


def add_population_shift_tests(roi_summary, results):
    """For each per-cell perm-p column, run Wilcoxon-vs-uniform per ROI.
    Multiple comparisons across the 7 ROIs corrected BOTH ways:
      - BH-FDR (less conservative)         → `q_wilcoxon_<stat>`
      - Bonferroni FWE (more conservative) → `fwe_wilcoxon_<stat>`
    Figure stars are driven by the FWE-corrected p (stricter)."""
    spec = [
        ('p_perm_min_phase_contrast', 'min_phase'),
        ('p_perm_r_state',            'r_state'),
        ('p_perm_r_interaction',      'r_interaction'),
        ('p_perm_balance',            'balance'),
    ]
    n_rois = max(1, roi_summary.shape[0])
    for p_col, prefix in spec:
        if p_col not in results.columns:
            continue
        per_roi = wilcoxon_perm_p_per_roi(results, p_col)
        ps_raw = np.asarray([per_roi.get(r, np.nan) for r in roi_summary['roi']],
                             dtype=float)
        roi_summary[f'wilcoxon_p_{prefix}']    = ps_raw
        roi_summary[f'q_wilcoxon_{prefix}']    = bh_fdr(ps_raw)
        # Bonferroni FWE: raw_p × n_tests, capped at 1
        with np.errstate(invalid='ignore'):
            fwe = np.clip(ps_raw * n_rois, 0.0, 1.0)
        roi_summary[f'fwe_wilcoxon_{prefix}']  = fwe
    return roi_summary


def roi_omnibus_and_ec_tests(results, sig_col):
    """Two ROI-level inference tools for a binary `sig_col`:
      1. Chi-squared omnibus across ROIs (k×2 contingency).
      2. Planned 2×2 Fisher exact: EC vs all-other-ROIs pooled.
    Returns dict with both p-values.
    """
    tbl = (results.groupby('roi')[sig_col]
                   .agg(['sum', 'count'])
                   .rename(columns={'sum': 'k', 'count': 'n'}))
    tbl['not_k'] = tbl['n'] - tbl['k']
    obs_kx2 = tbl[['k', 'not_k']].to_numpy(dtype=int)
    chi2_p = np.nan; chi2_stat = np.nan; chi2_dof = np.nan
    if obs_kx2.shape[0] >= 2 and obs_kx2.sum() > 0:
        try:
            chi2_stat, chi2_p, chi2_dof, _ = stats.chi2_contingency(obs_kx2)
        except Exception:
            chi2_p = np.nan
    # EC vs rest
    ec_p = np.nan; ec_or = np.nan; ec_counts = None
    if 'EC' in tbl.index:
        ec_row  = tbl.loc['EC', ['k', 'not_k']].to_numpy(dtype=int)
        rest    = tbl.drop('EC')[['k', 'not_k']].sum(axis=0).to_numpy(dtype=int)
        ec_counts = {'EC_k': int(ec_row[0]), 'EC_n': int(ec_row.sum()),
                      'rest_k': int(rest[0]), 'rest_n': int(rest.sum())}
        try:
            ec_or, ec_p = stats.fisher_exact(np.vstack([ec_row, rest]),
                                              alternative='greater')
        except Exception:
            ec_p = np.nan
    out = {'chi2_omnibus_p': float(chi2_p) if np.isfinite(chi2_p) else np.nan,
           'fisher_EC_vs_rest_p': float(ec_p) if np.isfinite(ec_p) else np.nan,
           # reporting extras: the manuscript quotes chi2(dof) and the OR,
           # which were previously not persisted anywhere.
           'chi2_omnibus_stat': float(chi2_stat) if np.isfinite(chi2_stat) else np.nan,
           'chi2_omnibus_dof': int(chi2_dof) if np.isfinite(chi2_dof) else None,
           'fisher_EC_vs_rest_odds_ratio': float(ec_or) if np.isfinite(ec_or) else np.nan,
           'per_roi_counts': {str(r): {'k': int(tbl.loc[r, 'k']),
                                        'n': int(tbl.loc[r, 'n'])}
                               for r in tbl.index}}
    if ec_counts is not None:
        out['EC_vs_rest_counts'] = ec_counts
    return out


def balance_roi_omnibus_perm(results, n_perm=10000, seed=42, col='balance'):
    """Permutation omnibus: does `balance` differ across ROIs?

    The observed statistic is the Kruskal-Wallis H over ROI groups. The null
    is built by SHUFFLING ROI LABELS across cells and recomputing H, which
    holds the balance distribution fixed and tests only the regional
    assignment. Reported instead of the parametric chi-square p-value
    because the per-cell balance values are neither normal nor independent of
    cell count per ROI.
    """
    d = results[[col, 'roi']].dropna()
    if d['roi'].nunique() < 2 or len(d) < 10:
        return {'stat_H': np.nan, 'p_perm': np.nan, 'n_perm': 0, 'n_cells': len(d)}
    vals = d[col].to_numpy(dtype=float)
    labels = d['roi'].to_numpy()

    def _H(lab):
        groups = [vals[lab == r] for r in np.unique(lab)]
        groups = [g for g in groups if g.size > 0]
        if len(groups) < 2:
            return np.nan
        return float(stats.kruskal(*groups)[0])

    H_obs = _H(labels)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = _H(rng.permutation(labels))
    v = np.isfinite(null)
    p = ((np.sum(null[v] >= H_obs) + 1) / (v.sum() + 1)
         if v.any() and np.isfinite(H_obs) else np.nan)
    return {'stat_H': H_obs, 'p_perm': float(p) if np.isfinite(p) else np.nan,
            'n_perm': int(n_perm), 'n_cells': int(len(d)),
            'null_mean_H': float(np.nanmean(null)) if v.any() else np.nan}


def make_roi_summary(df):
    rows = []
    for roi, g in df.groupby("roi", sort=False):
        n = int(len(g))
        n_sustained = int(g["sig_sustained"].sum())
        n_state_enc = int(g["sig_r_state"].sum())
        n_phasic    = int(g["sig_r_interaction"].sum())
        n_sustained_fdr = int(g.get("sig_sustained_fdr", pd.Series([False] * n)).sum())
        n_state_enc_fdr = int(g.get("sig_r_state_fdr", pd.Series([False] * n)).sum())
        n_phasic_fdr    = int(g.get("sig_r_interaction_fdr", pd.Series([False] * n)).sum())
        # phasic-only: cells where the interaction contrib generalises but state main doesn't
        n_phasic_only = int(((g["sig_r_interaction"]) & (~g["sig_r_state"])).sum())
        t_min, p_min = one_sided_ttest_greater(g["min_phase_contrast"])
        t_rs,  p_rs  = one_sided_ttest_greater(g["r_state"])
        t_ri,  p_ri  = one_sided_ttest_greater(g["r_interaction"])
        ov_state, ov_count, ov_n, ov_p = overrepresented_state(
            g.loc[g["sig_sustained"], "pref_state_mode"].tolist()
        )
        rows.append({
            "roi": roi, "n_cells": n,
            # Sustained gate
            "mean_min_phase_contrast": float(np.nanmean(g["min_phase_contrast"])),
            "t_min_phase_gt0": t_min, "p_t_min_phase": p_min,
            "n_sustained": n_sustained, "frac_sustained": n_sustained / n,
            "n_sustained_fdr": n_sustained_fdr,
            "binom_p_sustained": float(_binom_vs_alpha(n_sustained, n)),
            # r_state (sustained encoding generalisation)
            "mean_r_state": float(np.nanmean(g["r_state"])),
            "t_r_state_gt0": t_rs, "p_t_r_state": p_rs,
            "n_sig_r_state": n_state_enc, "frac_sig_r_state": n_state_enc / n,
            "n_sig_r_state_fdr": n_state_enc_fdr,
            "binom_p_sig_r_state": float(_binom_vs_alpha(n_state_enc, n)),
            # r_interaction (phasic encoding generalisation)
            "mean_r_interaction": float(np.nanmean(g["r_interaction"])),
            "t_r_interaction_gt0": t_ri, "p_t_r_interaction": p_ri,
            "n_sig_r_interaction": n_phasic, "frac_sig_r_interaction": n_phasic / n,
            "n_sig_r_interaction_fdr": n_phasic_fdr,
            "binom_p_sig_r_interaction": float(_binom_vs_alpha(n_phasic, n)),
            # Composition
            "n_phasic_only": n_phasic_only,
            "frac_phasic_only": n_phasic_only / n,
            "binom_p_phasic_only": float(_binom_vs_alpha(n_phasic_only, n)),
            # BALANCE (phase-invariant minus phase-modulated generalisation)
            "mean_balance": float(g["balance"].mean(skipna=True))
                if "balance" in g else np.nan,
            "mean_balance_null_centred": float(
                g["balance_null_centred"].mean(skipna=True))
                if "balance_null_centred" in g else np.nan,
            "n_sig_balance": int(g.get("sig_balance",
                                        pd.Series([False] * n)).sum()),
            "frac_sig_balance": float(
                g.get("sig_balance", pd.Series([False] * n)).sum()) / n,
            # Null means of the two components, so any block-size bias in the
            # raw difference is visible in the summary rather than assumed away.
            "null_mean_r_state": float(g["null_mean_r_state"].mean(skipna=True))
                if "null_mean_r_state" in g else np.nan,
            "null_mean_r_interaction": float(
                g["null_mean_r_interaction"].mean(skipna=True))
                if "null_mean_r_interaction" in g else np.nan,
            # Pref-state overrep among sustained cells
            "overrep_state": ov_state, "overrep_state_count": ov_count,
            "overrep_state_n": ov_n, "overrep_state_p_binom": ov_p,
        })
    out = pd.DataFrame(rows)
    if out.empty: return out
    present = out["roi"].tolist()
    order = [ROI_ORDER.index(r) if r in ROI_ORDER else len(ROI_ORDER) + present.index(r)
             for r in present]
    out["__o"] = order
    out = out.sort_values("__o").drop(columns="__o").reset_index(drop=True)
    # Population shift tests (Wilcoxon perm-p vs 0.5, BH-FDR across ROIs)
    out = add_population_shift_tests(out, df)
    return out


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def _rc():
    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size':       FONT_TICK,
        'axes.labelsize':  FONT_AXIS,
        'axes.titlesize':  FONT_BIG,
        'xtick.labelsize': FONT_TICK,
        'ytick.labelsize': FONT_TICK,
        'legend.fontsize': FONT_TICK,
        'pdf.fonttype':    42,
        'ps.fonttype':     42,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def _strip(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def _roi_palette(rois):
    """Per-ROI colour cycle (state palette repeated as needed)."""
    return (STATE_COLORS + STATE_COLORS)[:len(rois)]


def _save(fig, save_path):
    """Save both PDF and PNG; close fig."""
    base = os.path.splitext(save_path)[0]
    fig.savefig(base + '.pdf', dpi=DPI, bbox_inches='tight')
    fig.savefig(base + '.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# Caption text storage: written to `captions.md` next to the figures so the
# user can copy them straight into the manuscript.
_FIGURE_CAPTIONS = {}


def _add_caption(fig, fig_key, text, draw=True):
    """Record the caption for captions.md, and (unless `draw=False`) draw it
    beneath the axes.

    The caption is HARD-wrapped with textwrap rather than matplotlib's
    `wrap=True`. matplotlib wraps against the figure width at draw time, but
    `bbox_inches='tight'` in `_save` then re-measures the unwrapped extent,
    so every captioned figure was being saved many times wider than its
    axes -- fig 11 came out 20,647 px wide for a 6 cm heatmap. Wrapping the
    string ourselves keeps the saved width equal to the figure width.

    `draw=False` for publication figures: their caption belongs in the
    manuscript, not burned into the vector file, but should still reach
    captions.md.
    """
    _FIGURE_CAPTIONS[fig_key] = text
    if not draw:
        return
    # ~0.5 em per character at FONT_SMALL; 1 pt = 1/72 in.
    char_in = 0.5 * FONT_SMALL / 72.0
    ncols = max(40, int(fig.get_size_inches()[0] / char_in))
    wrapped = "\n".join(textwrap.wrap(text, width=ncols))
    fig.text(0.5, -0.02, wrapped, ha='center', va='top',
             fontsize=FONT_SMALL, color='#222')


def write_captions_md(out_dir):
    """Write all collected captions to captions.md in `out_dir`."""
    lines = ["# Figure captions",
             "",
             "All figures generated by `scripts/encoding_state_sustained_cv.py` "
             "for the sustained-state encoding analysis.",
             ""]
    for key, txt in _FIGURE_CAPTIONS.items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append(txt)
        lines.append("")
    Path(out_dir).joinpath("captions.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------
# Figure 1 — Design matrix (5 cm × 3 cm)
# ---------------------------------------------------------------------
def fig01_design_matrix(X, slices, save_path):
    _rc()
    fig, ax = plt.subplots(figsize=(5 * CM, 3 * CM))
    ax.imshow(X, aspect='auto', cmap='Greys', interpolation='nearest')
    for name, s in slices.items():
        if s.stop > s.start:
            ax.axvline(s.stop - 0.5, color='red', lw=0.4, alpha=0.6)
            ax.text((s.start + s.stop - 1) / 2, -0.02 * X.shape[0],
                    name, ha='center', va='bottom', fontsize=FONT_SMALL)
    ax.set_xlabel('design columns', fontsize=FONT_AXIS)
    ax.set_ylabel('time bins', fontsize=FONT_AXIS)
    ax.set_xticks([]); ax.set_yticks([])
    _add_caption(fig, "01_design_matrix",
                  "GLM design matrix for one example training fold. Columns: "
                  "4 state one-hot indicators, 8 sum-coded state×phase "
                  "interactions, and (K_train − 1) per-config intercepts (one "
                  "configuration dropped as reference). Sum coding for phase "
                  "ensures β_state[s] is the marginal across-phase mean of "
                  "state s, not a phase-reference cell mean.")
    _save(fig, save_path)


# ---------------------------------------------------------------------
# Figure 2 — Pipeline run stats (14 cm × 3 cm, 4 panels)
# ---------------------------------------------------------------------
def fig02_pipeline_stats(results, save_path):
    _rc()
    fig, axes = plt.subplots(1, 4, figsize=(14 * CM, 3 * CM),
                              constrained_layout=True)
    rois = results.groupby('roi').size()
    rois = rois.reindex([r for r in ROI_ORDER if r in rois.index]
                        + [r for r in rois.index if r not in ROI_ORDER])

    ax = axes[0]
    ax.bar(range(len(rois)), rois.values,
           color=[ROI_COLOURS.get(r, '#888') for r in rois.index],
           edgecolor='black', linewidth=0.3, alpha=0.85)
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels(rois.index, rotation=45, ha='right',
                       fontsize=FONT_SMALL)
    ax.set_ylabel('# cells', fontsize=FONT_SMALL)
    ax.set_title('cells / ROI', fontsize=FONT_SMALL); _strip(ax)

    ax = axes[1]
    ax.hist(results['n_folds_used'].dropna(),
            bins=np.arange(0, 10) - 0.5,
            color='#888', edgecolor='black', linewidth=0.3)
    ax.set_xlabel('valid folds / cell', fontsize=FONT_SMALL)
    ax.set_title('CV validity', fontsize=FONT_SMALL); _strip(ax)

    ax = axes[2]
    ax.hist(results['r_full'].dropna(), bins=40,
            color=PHASE_COLOURS['middle'], edgecolor='black', linewidth=0.2)
    ax.axvline(0, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('held-out r_full', fontsize=FONT_SMALL)
    ax.set_title('full-model fit', fontsize=FONT_SMALL); _strip(ax)

    ax = axes[3]
    nans = [
        results['p_perm_min_phase_contrast'].isna().mean() * 100,
        results['p_perm_r_state'].isna().mean() * 100,
        results['p_perm_r_interaction'].isna().mean() * 100,
    ]
    ax.bar(['sus', 'r_s', 'r_i'], nans,
           color=[STATE_COLOURS['A'], STATE_COLOURS['B'], STATE_COLOURS['D']],
           edgecolor='black', linewidth=0.3)
    ax.set_ylabel('% NaN p', fontsize=FONT_SMALL)
    ax.set_title('NaN rate', fontsize=FONT_SMALL); _strip(ax)

    for ax in axes:
        ax.tick_params(axis='both', labelsize=FONT_SMALL, length=2, pad=1)

    _add_caption(fig, "02_pipeline_run_stats",
                  "Pipeline diagnostics. Left: cells per ROI in era_brewer "
                  "ROI colours. Centre-left: leave-one-config-out folds with "
                  "valid statistics per cell. Centre-right: distribution of "
                  "mean held-out full-model r across cells. Right: NaN rate "
                  "(% of cells with no perm-p) for each of the three tests "
                  "(sustained, r_state, r_interaction).")
    _save(fig, save_path)


# ---------------------------------------------------------------------
# Figure 3 — Permutation nulls (10 cm wide, ~2 cm tall per example cell)
# ---------------------------------------------------------------------
def fig03_perm_nulls(example_rows, save_path):
    """Permutation null + observed value per example cell.
    Caption: dark-green line = observed empirical value; gray histogram = null
    distribution from 100 circular shifts. p value below each panel."""
    _rc()
    n = len(example_rows)
    if n == 0: return
    fig, axes = plt.subplots(n, 2, figsize=(10 * CM, 2.4 * CM * n),
                              constrained_layout=True, squeeze=False)
    for i, (label, row) in enumerate(example_rows):
        # min_phase_contrast null (sustained statistic)
        ax = axes[i, 0]
        null = row.get("_perm_min_phase", np.array([]))
        obs = row.get("min_phase_contrast", np.nan)
        p   = row.get('p_perm_min_phase_contrast', np.nan)
        if isinstance(null, np.ndarray) and null.size:
            ax.hist(null[np.isfinite(null)], bins=30, color='lightgray',
                    edgecolor='gray', alpha=0.85, linewidth=0.3)
        if np.isfinite(obs):
            ax.axvline(obs, color=OBSERVED_GREEN, lw=1.5)
        ax.axvline(0, color='gray', ls='--', lw=0.4)
        ax.set_xlabel(f'min phase contrast\nobs = {obs:+.3f},  p = {p:.3f}',
                       fontsize=FONT_SMALL)
        ax.set_ylabel('# perm', fontsize=FONT_SMALL)
        ax.set_title(f'{label} — {row["neuron"][:18]}', fontsize=FONT_SMALL)
        ax.tick_params(axis='both', labelsize=FONT_SMALL, length=2, pad=1)
        _strip(ax)

        # r_state null (encoding-correlation, sustained generalisation)
        ax = axes[i, 1]
        null = row.get("_perm_r_state", np.array([]))
        obs = row.get("r_state", np.nan)
        p   = row.get('p_perm_r_state', np.nan)
        if isinstance(null, np.ndarray) and null.size:
            ax.hist(null[np.isfinite(null)], bins=30, color='lightgray',
                    edgecolor='gray', alpha=0.85, linewidth=0.3)
        if np.isfinite(obs):
            ax.axvline(obs, color=OBSERVED_GREEN, lw=1.5)
        ax.axvline(0, color='gray', ls='--', lw=0.4)
        ax.set_xlabel(f'r_state\nobs = {obs:+.3f},  p = {p:.3f}',
                       fontsize=FONT_SMALL)
        ax.set_title(f'{label} — {row["neuron"][:18]}', fontsize=FONT_SMALL)
        ax.tick_params(axis='both', labelsize=FONT_SMALL, length=2, pad=1)
        _strip(ax)

    _add_caption(fig, "03_perm_nulls_examples",
                  "Permutation null vs observed empirical value for example "
                  "cells across the significance spectrum. Each row = one "
                  "cell. Grey histograms = null distribution from 100 "
                  "circular shifts of the held-out trace. Dark-green vertical "
                  "line = observed empirical value, with p value reported "
                  "below each axis. Grey dashed line = reference at 0.")
    _save(fig, save_path)


# ---------------------------------------------------------------------
# Figure 4 — Perm-p uniformity check
# ---------------------------------------------------------------------
def fig04_perm_p_uniformity(results, save_path):
    _rc()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    # We expect non-effect cells to give uniform p-values.
    # Crude proxy: cells with observed stat below population median.
    for j, (p_col, title) in enumerate([
        ('p_perm_min_phase_contrast', 'sig_b — min_phase_contrast'),
        ('p_perm_r_state', 'sig_c — r_state'),
    ]):
        ax = axes[j]
        p = results[p_col].dropna().to_numpy()
        if p.size:
            p_sorted = np.sort(p)
            uniform_q = (np.arange(len(p_sorted)) + 1) / (len(p_sorted) + 1)
            ax.plot([0, 1], [0, 1], color='gray', ls='--', lw=0.6, label='uniform')
            ax.plot(uniform_q, p_sorted, color=COLOR_PALETTE[1 + j], lw=1.5,
                    label=f'observed p ({len(p_sorted)} cells)')
            # Excess at low p shows real signal departure from uniformity
            n_below_05 = int((p_sorted < 0.05).sum())
            ax.text(0.05, 0.95, f'cells with p < .05:\n{n_below_05} ({100*n_below_05/len(p_sorted):.1f}%)',
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=FONT_SIZE - 2)
        ax.set_xlabel('expected uniform quantile')
        ax.set_ylabel('observed perm-p quantile')
        ax.set_title(title)
        ax.legend(fontsize=FONT_SIZE - 2, frameon=False)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    fig.suptitle('Perm-p QQ vs uniform (deviation at low p indicates true effects)',
                 fontsize=FONT_SIZE + 1)
    fig.savefig(save_path, dpi=DPI, bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------------
# Figs 5/6/6b — Per-ROI histograms (subplot 3 cm × 2 cm, single shared legend)
# ---------------------------------------------------------------------
def _fig_stat_dist_per_roi(results, stat_col, stat_label, save_path,
                            color='lightgray', sig_col=None,
                            q_per_roi=None):
    """Per-ROI histograms of `stat_col`, sig cells overlaid in the ROI's colour.
    Subplots are flat (3 cm × 1.5 cm) with one shared legend on the right.
    Cell counts go in the caption, not on the panels.
    Caption: black solid line = mean of the distribution; dashed grey line = 0.
    Coloured bars = cells passing the perm test at α=.05; light grey bars = all
    other cells in that ROI.
    """
    _rc()
    rois = [r for r in ROI_ORDER if (results['roi'] == r).any()]
    n_cols = min(len(rois), 4)
    n_rows = int(np.ceil(len(rois) / n_cols))
    fig_w = (3 * n_cols + 2.2) * CM   # extra ~2 cm for side legend
    fig_h = (1.5 * n_rows + 0.7) * CM
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                              constrained_layout=True, squeeze=False)
    all_vals = results[stat_col].dropna().to_numpy()
    if all_vals.size:
        lo, hi = np.nanpercentile(all_vals, [1, 99])
        bins = np.linspace(lo, hi, 22)
    else:
        bins = 20

    for i, r in enumerate(rois):
        ax = axes[i // n_cols, i % n_cols]
        roi_df = results.loc[results['roi'] == r]
        v = roi_df[stat_col].dropna().to_numpy()
        if v.size:
            ax.hist(v, bins=bins, color=color, edgecolor='black',
                    linewidth=0.2, alpha=0.55)
            if sig_col is not None and sig_col in roi_df.columns:
                v_sig = roi_df.loc[roi_df[sig_col], stat_col].dropna().to_numpy()
                if v_sig.size:
                    ax.hist(v_sig, bins=bins,
                            color=ROI_COLOURS.get(r, '#444'),
                            edgecolor='black', linewidth=0.2, alpha=0.95)
            ax.axvline(float(np.mean(v)), color='black', lw=0.8)
        ax.axvline(0, color='gray', ls='--', lw=0.4)
        # ROI title with FDR-population-shift stars
        title_str = r
        if q_per_roi is not None and r in q_per_roi:
            q = q_per_roi[r]
            if np.isfinite(q):
                star = ('***' if q < .001 else '**' if q < .01
                        else '*' if q < .05 else '')
                if star:
                    title_str = f'{r} {star}'
        ax.set_title(title_str, fontsize=FONT_SMALL, pad=2)
        ax.tick_params(axis='both', labelsize=FONT_SMALL, length=1.5, pad=1)
        if i % n_cols == 0:
            ax.set_ylabel('# cells', fontsize=FONT_SMALL)
        if i // n_cols == n_rows - 1:
            ax.set_xlabel(stat_label, fontsize=FONT_SMALL)
        _strip(ax)
    for k in range(len(rois), n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis('off')

    # Single legend on the right
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=color, ec='black', lw=0.2, alpha=0.55),
        plt.Rectangle((0, 0), 1, 1, fc='#888', ec='black', lw=0.2, alpha=0.95),
        plt.Line2D([0], [0], color='black', lw=0.8),
        plt.Line2D([0], [0], color='gray', lw=0.4, ls='--'),
    ]
    legend_labels = ['all cells', 'perm-sig (ROI colour)', 'mean', '0']
    fig.legend(legend_handles, legend_labels,
               loc='center right', bbox_to_anchor=(1.0, 0.5),
               fontsize=FONT_SMALL, frameon=False)

    _add_caption(fig, f"dist_{stat_col}",
                  f"Per-ROI distribution of {stat_label}. Light grey = all "
                  f"cells in that ROI; ROI-coloured = cells passing the "
                  f"permutation test at α = .05. Black solid line = mean of "
                  f"the distribution. Grey dashed line at 0. Stars next to "
                  f"the ROI name = BH-FDR-corrected Wilcoxon perm-p < 0.5 "
                  f"(one-sided), testing whether the population is shifted "
                  f"in the predicted direction (* < .05, ** < .01, *** < "
                  f".001, BH-FDR across 7 ROIs; Bonferroni-FWE also "
                  f"available as `fwe_wilcoxon_<stat>` in roi_summary). "
                  f"Cell counts per ROI: ACC=159, medialOFC=167, PCC=64, "
                  f"Parahippocampal=61, HC_anterior=296, HC_mid=180, EC=28.")
    _save(fig, save_path)


# ---------------------------------------------------------------------
# Figure 7 — Per-phase contrast by category (14 cm × 4 cm, phase gradient)
# ---------------------------------------------------------------------
def fig07_per_phase_contrast_by_category(results, save_path):
    _rc()
    any_state_sig = results['sig_r_state'] | results['sig_r_interaction']
    categories = [
        ('sustained\n(min PC perm-sig)',
         results[results['sig_sustained']]),
        ('phasic-only\n(any state sig & ¬sustained)',
         results[any_state_sig & (~results['sig_sustained'])]),
        ('neither',
         results[(~results['sig_sustained']) & (~any_state_sig)]),
    ]
    # Figure caption: boxes coloured by phase (pastel pink → bordeaux).
    # Solid black line in each box = median; black DOTTED tick = mean of the
    # distribution. Grey dashed horizontal line at 0 marks zero contrast.
    fig, axes = plt.subplots(1, 3, figsize=(14 * CM, 2.5 * CM),
                              constrained_layout=True)
    for ax, (title, sub) in zip(axes, categories):
        data = [sub['phase_contrast_early'].dropna().to_numpy(),
                sub['phase_contrast_middle'].dropna().to_numpy(),
                sub['phase_contrast_late'].dropna().to_numpy()]
        positions = [1, 2, 3]
        bp = ax.boxplot(data, positions=positions, widths=0.55,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color='black', lw=0.8),
                        whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5),
                        boxprops=dict(lw=0.4))
        for patch, ph in zip(bp['boxes'], PHASES):
            patch.set_facecolor(PHASE_COLOURS[ph])
            patch.set_edgecolor('black')
        # Mean as a black DOTTED tick (median is the box black solid line)
        for i, d in enumerate(data):
            if d.size:
                m = float(np.mean(d))
                ax.plot([positions[i] - 0.25, positions[i] + 0.25], [m, m],
                        color='black', lw=1.0, ls=':')
        ax.axhline(0, color='gray', ls='--', lw=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(['early', 'middle', 'late'], fontsize=FONT_SMALL)
        ax.set_ylabel('pref state − others', fontsize=FONT_SMALL)
        ax.set_title(f'{title}  (n={len(sub)})', fontsize=FONT_SMALL)
        ax.tick_params(axis='both', labelsize=FONT_SMALL, length=2)
        _strip(ax)

    _add_caption(fig, "07_per_phase_by_category",
                  "Held-out per-phase contrast (preferred state − other "
                  "states) split by cell category. Categories are mutually "
                  "exclusive: sustained = min_phase_contrast perm-sig; "
                  "phasic-only = any state encoding perm-sig AND NOT sustained; "
                  "neither = no test sig. Boxes coloured by phase (pastel "
                  "pink = early → bordeaux = late). Black solid line = "
                  "median; black dotted tick = mean. Grey dashed line at 0 "
                  "marks no contrast.")
    _save(fig, save_path)


# ---------------------------------------------------------------------
# Figure 8 — MAIN: per-ROI fractions (2×2 of 4.5 × 3.5 cm subpanels)
# ---------------------------------------------------------------------
def fig08_roi_fractions_overview(roi_summary, save_path):
    _rc()
    rois = roi_summary['roi'].tolist()
    x = np.arange(len(rois))
    # Bar-plot area target: each subpanel ~2 cm tall. Outer figure: 2x2 grid
    # with the 11 pt titles, 10 pt x-tick labels at 35° rotation, and the
    # bold-11pt stars all needing room. Figsize bumped to ~12 × 9 cm.
    fig, axes = plt.subplots(2, 2, figsize=(12 * CM, 9 * CM),
                              constrained_layout=True)

    def _stars(p):
        if not np.isfinite(p): return ''
        if p < .001: return '***'
        if p < .01:  return '**'
        if p < .05:  return '*'
        return ''

    roi_bar_colours = [ROI_COLOURS.get(r, '#888') for r in rois]

    # Within the 9–11 pt window we use the UPPER half here so a
    # 2 cm tall subpanel still reads well on A4.
    f_lab  = FONT_AXIS         # 10 pt — tick labels + y-axis label
    f_ttl  = FONT_BIG          # 11 pt — panel titles
    f_star = FONT_BIG          # 11 pt bold — significance stars

    # (a) sustained — solid ROI-coloured bars
    ax = axes[0, 0]
    ax.bar(x, roi_summary['frac_sustained'], color=roi_bar_colours,
           edgecolor='black', linewidth=0.4, alpha=0.9)
    ax.axhline(SIG_ALPHA, color='gray', ls='--', lw=0.5)
    for i, p in enumerate(roi_summary['binom_p_sustained']):
        ax.text(i, roi_summary['frac_sustained'].iloc[i] + 0.005,
                _stars(p), ha='center', va='bottom',
                fontsize=f_star, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=35, ha='right', fontsize=f_lab)
    ax.set_ylabel('frac sustained', fontsize=f_lab)
    ax.set_title('(a) sustained', fontsize=f_ttl); _strip(ax)

    # (b) r_state
    ax = axes[0, 1]
    ax.bar(x, roi_summary['frac_sig_r_state'], color=roi_bar_colours,
           edgecolor='black', linewidth=0.4, alpha=0.9)
    ax.axhline(SIG_ALPHA, color='gray', ls='--', lw=0.5)
    for i, p in enumerate(roi_summary['binom_p_sig_r_state']):
        ax.text(i, roi_summary['frac_sig_r_state'].iloc[i] + 0.005,
                _stars(p), ha='center', va='bottom',
                fontsize=f_star, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=35, ha='right', fontsize=f_lab)
    ax.set_ylabel('frac sig r_state', fontsize=f_lab)
    ax.set_title('(b) state encoding\ngeneralises', fontsize=f_ttl); _strip(ax)

    # (c) phasic-only — hatched ROI-coloured bars
    ax = axes[1, 0]
    bars_c = ax.bar(x, roi_summary['frac_phasic_only'],
                     color=roi_bar_colours, edgecolor='black',
                     linewidth=0.4, alpha=0.9)
    for b in bars_c:
        b.set_hatch('///')
    ax.axhline(SIG_ALPHA, color='gray', ls='--', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=35, ha='right', fontsize=f_lab)
    ax.set_ylabel('frac phasic-only', fontsize=f_lab)
    ax.set_title('(c) phasic-only', fontsize=f_ttl); _strip(ax)

    # (d) Stacked composition — three mutually exclusive populations.
    ax = axes[1, 1]
    frac_sus = roi_summary['frac_sustained'].to_numpy()
    frac_pha = roi_summary['frac_phasic_only'].to_numpy()
    frac_nei = 1.0 - frac_sus - frac_pha
    ax.bar(x, frac_sus, color=roi_bar_colours,
           edgecolor='black', linewidth=0.4, alpha=0.9,
           label='sustained')
    bars_pha = ax.bar(x, frac_pha, bottom=frac_sus,
                       color=roi_bar_colours, edgecolor='black',
                       linewidth=0.4, alpha=0.9, label='phasic-only')
    for b in bars_pha: b.set_hatch('///')
    ax.bar(x, frac_nei, bottom=frac_sus + frac_pha,
           color='lightgray', edgecolor='black', linewidth=0.4,
           alpha=0.7, label='neither')
    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=35, ha='right', fontsize=f_lab)
    ax.set_ylabel('frac of cells', fontsize=f_lab)
    ax.set_ylim(0, 1.0)
    ax.set_title('(d) composition per ROI', fontsize=f_ttl)
    ax.legend(fontsize=f_lab, frameon=False, loc='upper right',
              handlelength=1.0)
    _strip(ax)

    for ax in axes.flatten():
        ax.tick_params(axis='both', labelsize=f_lab, length=2, pad=1)

    _add_caption(fig, "08_roi_fractions_overview",
                  "Per-ROI fractions of cells. (a) sustained = perm-sig on "
                  "min_phase_contrast (preferred-state effect positive in all "
                  "3 phases). (b) state encoding generalises = perm-sig on "
                  "r_state (marginal-state encoding correlates with held-out "
                  "activity). (c) phasic-only = (sig r_state OR sig "
                  "r_interaction) AND NOT sustained, /// hatched. (d) "
                  "stacked composition: each bar sums to 1.0, with sustained "
                  "(solid ROI colour), phasic-only (/// hatched), and neither "
                  "(grey). Bars in era_brewer ROI colours. Stars = binomial "
                  "p vs 5 % chance: * < .05, ** < .01, *** < .001. Grey "
                  "dashed = chance line at 0.05.")
    _save(fig, save_path)


# ---------------------------------------------------------------------
# Figure 9 — Per-cell scatter b vs c
# ---------------------------------------------------------------------
def fig09_scatter_b_vs_c(results, save_path):
    _rc()
    rois = [r for r in ROI_ORDER if (results['roi'] == r).any()]
    palette = _roi_palette(rois)
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    for i, r in enumerate(rois):
        sub = results[results['roi'] == r]
        ax.scatter(sub['r_state'], sub['min_phase_contrast'],
                   s=14, color=palette[i], alpha=0.45, edgecolor='none',
                   label=f'{r} (n={len(sub)})')
    sig = results[results['sig_sustained']]
    ax.scatter(sig['r_state'], sig['min_phase_contrast'],
               s=22, facecolor='none', edgecolor=COLOR_PALETTE[0], lw=1.0,
               label='sig_b (sustained)')
    ax.axhline(0, color='gray', ls='--', lw=0.7)
    ax.axvline(0, color='gray', ls='--', lw=0.7)
    ax.set_xlabel('held-out delta_R²_joint (any state)')
    ax.set_ylabel('held-out min phase contrast (sustained)')
    ax.set_title('Per-cell scatter — sustained vs. any state')
    ax.legend(fontsize=FONT_SIZE - 2, frameon=False, loc='upper left',
              bbox_to_anchor=(1.02, 1.0))
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    fig.savefig(save_path, dpi=DPI, bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------------
# Figure 8b — Per-ROI fractions, GROUPED bars (all-state, sustained, phasic)
# ---------------------------------------------------------------------
def fig08b_roi_fractions_grouped(roi_summary, save_path):
    """Same data as fig 8 but with bars *grouped per ROI*:
        ROI   →   [all-state | sustained | phasic-only]
    where all-state = frac of cells with any state encoding sig =
        sustained + phasic-only  (mutually exclusive by construction).
    Bars in ROI colour; three distinct visual treatments per ROI to tell
    them apart at a glance:
        all-state  = light fill   (alpha 0.45)
        sustained  = solid fill   (alpha 0.9)
        phasic-only = hatched fill (alpha 0.9, /// hatch)
    """
    _rc()
    rois = roi_summary['roi'].tolist()
    n_rois = len(rois)
    bar_colours = [ROI_COLOURS.get(r, '#888') for r in rois]

    # Derived "all state" = sustained + phasic-only (mutually exclusive).
    frac_all = (roi_summary['frac_sustained'].to_numpy()
                + roi_summary['frac_phasic_only'].to_numpy())
    frac_sus = roi_summary['frac_sustained'].to_numpy()
    frac_pha = roi_summary['frac_phasic_only'].to_numpy()

    width = 0.27
    x = np.arange(n_rois)
    fig, ax = plt.subplots(figsize=(13 * CM, 5 * CM), constrained_layout=True)

    b_all = ax.bar(x - width, frac_all, width=width, color=bar_colours,
                    edgecolor='black', linewidth=0.4, alpha=0.45,
                    label='all state')
    b_sus = ax.bar(x, frac_sus, width=width, color=bar_colours,
                    edgecolor='black', linewidth=0.4, alpha=0.9,
                    label='sustained')
    b_pha = ax.bar(x + width, frac_pha, width=width, color=bar_colours,
                    edgecolor='black', linewidth=0.4, alpha=0.9,
                    label='phasic-only')
    for b in b_pha: b.set_hatch('///')

    ax.axhline(SIG_ALPHA, color='gray', ls='--', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=35, ha='right', fontsize=FONT_SMALL)
    ax.set_ylabel('frac of cells', fontsize=FONT_AXIS)
    ax.set_title('Per-ROI state encoding — bars grouped by ROI',
                  fontsize=FONT_AXIS)
    ax.legend(fontsize=FONT_SMALL, frameon=False, loc='upper right',
              handlelength=1.0, ncol=3)
    ax.tick_params(axis='both', labelsize=FONT_SMALL, length=2, pad=1)
    _strip(ax)

    _add_caption(fig, "08b_roi_fractions_grouped",
                  "Same three fractions as fig 8 but grouped by ROI: each "
                  "ROI gets three bars side-by-side — all-state (light, left), "
                  "sustained (solid, middle), phasic-only (/// hatched, "
                  "right). Bars use era_brewer ROI colours. By construction, "
                  "all-state = sustained + phasic-only (mutually exclusive). "
                  "Grey dashed = chance line at 0.05.")
    _save(fig, save_path)


# ---------------------------------------------------------------------
# Figure 10 — Pref-state stacked per ROI (sig_b cells)
# ---------------------------------------------------------------------
def fig10_pref_state_stacked(results, roi_summary, save_path):
    """Stacked bars per ROI of preferred-state distribution among sustained cells.
    Letters (A/B/C/D) printed inside each segment when tall enough. Alternating
    grey background. Significance stars above each bar from the binomial test
    against uniform 1/4."""
    _rc()
    rois = roi_summary['roi'].tolist()
    if not rois: return
    counts = np.zeros((len(rois), N_STATES))
    for ri, r in enumerate(rois):
        sub = results.loc[(results['roi'] == r) & results['sig_sustained'],
                          'pref_state_mode']
        c = pd.Series(sub).value_counts()
        for si, s in enumerate(STATES):
            counts[ri, si] = int(c.get(s, 0))

    fig, ax = plt.subplots(figsize=(9 * CM, 5.5 * CM), constrained_layout=True)
    x = np.arange(len(rois)); bottom = np.zeros(len(rois))
    # Alternating grey background bands
    for i in range(len(rois)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color='#f5f5f5', zorder=-1)
    # Stacked bars, with state letter inside each segment when tall enough
    for si, s in enumerate(STATES):
        bars = ax.bar(x, counts[:, si], bottom=bottom, color=STATE_COLOURS[s],
                      edgecolor='black', linewidth=0.4, label=s)
        for i, b in enumerate(bars):
            seg_h = b.get_height()
            # Skip the in-bar letter when the segment isn't tall enough to hold
            # one — the colour-coded legend still tells the reader which state.
            if seg_h >= 2.5:
                ax.text(b.get_x() + b.get_width() / 2,
                        bottom[i] + seg_h / 2, s,
                        ha='center', va='center',
                        fontsize=FONT_AXIS, color='white', fontweight='bold')
        bottom += counts[:, si]

    # Chance line at uniform 1/4 of total per ROI (dotted)
    for i in range(len(rois)):
        n_sus = bottom[i]
        if n_sus > 0:
            chance = n_sus / 4.0
            ax.plot([i - 0.4, i + 0.4], [chance, chance],
                    color='black', ls=':', lw=0.6)

    # Significance stars above each bar
    def _stars(p):
        if not np.isfinite(p): return ''
        if p < .001: return '***'
        if p < .01:  return '**'
        if p < .05:  return '*'
        return ''
    for i in range(len(rois)):
        ov_p = roi_summary.iloc[i]['overrep_state_p_binom']
        star = _stars(ov_p)
        if star:
            ax.text(i, bottom[i] + 0.5, star,
                    ha='center', va='bottom', fontsize=FONT_BIG)

    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=35, ha='right', fontsize=FONT_SMALL)
    ax.set_ylabel('# cells', fontsize=FONT_AXIS)
    ax.set_title('preferred state per ROI (sustained cells)', fontsize=FONT_AXIS)
    ax.legend(title='state', ncol=N_STATES, fontsize=FONT_SMALL,
              frameon=False, loc='upper right', handlelength=1.0,
              columnspacing=0.6)
    ax.tick_params(axis='both', labelsize=FONT_SMALL, length=2)
    _strip(ax)
    _add_caption(fig, "10_pref_state_stacked",
                  "Preferred-state distribution among sustained cells per ROI. "
                  "Stacked bars in state colours (A=orange, B=yellow, "
                  "C=light purple, D=dark purple), with letters inside each "
                  "segment when tall enough. Dotted horizontal mark = uniform "
                  "chance level (n_sustained/4 per ROI). Stars above each bar "
                  "= binomial p of the most-overrepresented state vs uniform "
                  "1/4: * < .05, ** < .01, *** < .001. Alternating grey "
                  "backgrounds for visual separation between ROIs.")
    _save(fig, save_path)


# ---------------------------------------------------------------------
# Figure 11 — Effect-size heatmap (ROI × stat), 2 panels (RdBu_r style).
# Colour = one-sample t-stat of per-cell empirical values vs 0.
# Borders come from the bias-free Wilcoxon perm-p shift test:
#   solid black = BH-FDR-sig across 7 ROIs; dotted = raw p < .05.
# Two side-by-side panels because `min_phase_contrast`'s null is NOT at 0
# (downward bias ~ -0.846 σ_cell); plotting it on the same colour scale as
# the unbiased correlations would be misleading. Shared y-axis (ROIs);
# separate colourmap and vmax for the contrast panel.
# ---------------------------------------------------------------------
def _load_rsa_state_per_roi(rois):
    """Load `state` sub-model t-stats and p_perm from the RSA companion run
    (`RSA_STATE_RUN_DIR/results_summary_combos.csv`), filter to the chosen
    test variant, and reindex to the order in `rois`. BH-FDR across the 7
    ROIs is applied to `p_perm` here (the CSV's `p_fdr` is NaN for this
    sub_model — `in_fdr_family=False`). Returns (t, raw_p, fdr_q) arrays
    aligned to `rois`."""
    csv = os.path.join(RSA_STATE_RUN_DIR, 'results_summary_combos.csv')
    n = len(rois)
    #import pdb; pdb.set_trace()
    if not os.path.exists(csv):
        print(f"  fig11: RSA CSV not found at {csv} — skipping RSA column.")
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    df = pd.read_csv(csv)
    sub = df[(df['test'] == RSA_TEST_VARIANT)
              & (df['sub_model'] == RSA_SUB_MODEL)
              & (df['combo'] == COMBO_MODEL_NAME)].copy()
    by_roi = sub.set_index('roi')
    t   = np.array([float(by_roi.loc[r, 't'])       if r in by_roi.index else np.nan
                    for r in rois])
    p_r = np.array([float(by_roi.loc[r, 'p_perm']) if r in by_roi.index else np.nan
                    for r in rois])
    p_fdr = bh_fdr(p_r)
    return t, p_r, p_fdr


# ---------------------------------------------------------------------
# Main-result figure (publication)
# ---------------------------------------------------------------------
ROI_DISPLAY_NAMES = {
    'HC_anterior': 'HC ant',
    'HC_mid':      'HC mid',
    'ACC':         'mPFC',   # project rule: never display "ACC"
}


def _roi_label(roi):
    return ROI_DISPLAY_NAMES.get(roi, roi)


def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for a binomial proportion.

    Preferred over the Wald interval here because the fractions of interest
    are small (~0.05-0.3) and some ROIs have few cells (EC n=35), where the
    Wald interval is badly calibrated and can leave [0, 1].
    """
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    phat = k / n
    denom = 1.0 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return float(centre - half), float(centre + half)


def rank_biserial_below_half(p_vals):
    """Matched-pairs rank-biserial correlation for the population-shift test.

    This is the effect size belonging to `wilcoxon_perm_p_per_roi`: the same
    signed ranks of (perm-p - 0.5), expressed on [-1, 1]. POSITIVE means the
    population sits below 0.5, i.e. shifted in the predicted direction.

    Used instead of plotting the p-value itself, so the figure shows an
    effect size and reserves significance for the stars.
    """
    d = np.asarray(p_vals, dtype=float)
    d = d[np.isfinite(d)] - 0.5
    d = d[d != 0]
    if d.size == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(d))
    r_neg = float(ranks[d < 0].sum())   # evidence FOR the effect
    r_pos = float(ranks[d > 0].sum())
    tot = r_neg + r_pos
    return float((r_neg - r_pos) / tot) if tot > 0 else np.nan


def _boot_ci_rank_biserial(p_vals, n_boot=2000, seed=42, alpha=0.05):
    """Percentile bootstrap CI over CELLS for `rank_biserial_below_half`."""
    p_vals = np.asarray(p_vals, dtype=float)
    p_vals = p_vals[np.isfinite(p_vals)]
    if p_vals.size < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, p_vals.size, size=(n_boot, p_vals.size))
    vals = np.array([rank_biserial_below_half(p_vals[i]) for i in idx])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def _stars_from_p(p):
    if not np.isfinite(p):
        return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


def fig15_single_cell_summary(results, roi_summary, save_path, omnibus=None):
    """CHAPTER PANEL: how the position code is carried, per region.

    a  the two COMPONENTS of each cell's held-out prediction -- phase-invariant
       position vs phase-modulated -- as point estimates with 95% CIs over a
       box of the per-cell distribution.
    b  their per-cell difference (the balance contrast), same form, with the
       permutation omnibus across regions.
    c  fraction of cells classified sustained, with Wilson CIs and the 5%
       chance rate.

    Panels a and b deliberately do NOT use bars. A bar encodes magnitude by
    area from zero, which for a mean correlation of ~0.02 against a per-cell
    spread an order of magnitude larger implies a precision the data do not
    have. Point + CI over the distribution box states exactly what is
    estimated. Panel c keeps bars because a fraction genuinely is a
    quantity measured from zero, and the 5% chance line needs that baseline.
    """
    _rc()
    rs = roi_summary[roi_summary['n_cells'] > 0].copy()
    rs = rs.sort_values('mean_balance', ascending=False)
    rois = rs['roi'].tolist()
    x = np.arange(len(rois), dtype=float)
    colours = [ROI_COLOURS.get(r, '#888888') for r in rois]

    fig, axes = plt.subplots(1, 3, figsize=(18.0 * CM, 4.0 * CM),
                             gridspec_kw={'width_ratios': [1.35, 1.0, 1.0],
                                          'wspace': 0.60})
    ax_a, ax_b, ax_c = axes

    # ---- a: components --------------------------------------------------
    # Significance is carried by LINE WEIGHT and OPACITY rather than by stars:
    # 12 star annotations in panel a were unreadable and made it hard to see
    # which point belonged to which ROI. Solid error bar = phase-invariant,
    # dotted = phase-modulated; the within-pair offset is small relative to
    # the between-ROI gap so pairs group visually.
    off = 0.19
    for i, roi in enumerate(rois):
        cells = results.loc[results['roi'] == roi]
        row = rs.iloc[i]
        for j, (col, qcol, filled) in enumerate([
                ('r_state', 'q_wilcoxon_r_state', True),
                ('r_interaction', 'q_wilcoxon_r_interaction', False)]):
            v = cells[col].dropna().to_numpy()
            if v.size == 0:
                continue
            xi = x[i] + (j - 0.5) * 2 * off
            m = float(v.mean()); se = float(v.std(ddof=1) / np.sqrt(v.size))
            q = float(row[qcol]) if qcol in rs.columns else np.nan
            sig = np.isfinite(q) and q < SIG_ALPHA
            bar_c = '0.05' if sig else '0.55'
            alpha = 1.0 if sig else 0.45
            ax_a.plot([xi, xi], [m - 1.96 * se, m + 1.96 * se], color=bar_c,
                      lw=2.3 if sig else 1.1, zorder=3, alpha=alpha,
                      solid_capstyle='round')
            ax_a.scatter([xi], [m], s=80, zorder=4, alpha=alpha,
                         facecolor=colours[i],
                         edgecolor='0.05' if sig else '0.55',
                         linewidth=1.3 if sig else 0.9,
                         hatch=None if filled else '////')
    ax_a.axhline(0, color='0.45', lw=0.9, ls=':', zorder=0)
    ax_a.set_ylabel('held-out r', labelpad=2)
    ax_a.set_title('a   Components', loc='left', fontweight='bold')
    ax_a.legend(handles=[
        mpatches.Patch(facecolor='0.55', edgecolor='0.05',
                       label='position (phase-invariant)'),
        mpatches.Patch(facecolor='0.55', edgecolor='0.05', hatch='////',
                       label='phase-modulated'),
        plt.Line2D([], [], color='0.45', ls=':', lw=0.9,
                   label='permutation null')],
        frameon=False, loc='upper right', handlelength=1.0,
        borderaxespad=0.05, fontsize=FONT_SMALL - 2,
        labelspacing=0.3)

    # ---- b: balance ------------------------------------------------------
    rng = np.random.default_rng(42)
    for i, roi in enumerate(rois):
        v = results.loc[results['roi'] == roi, 'balance'].dropna().to_numpy()
        if v.size < 3:
            continue
        m = float(v.mean())
        boot = rng.choice(v, size=(2000, v.size), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        q = (float(rs.iloc[i]['q_wilcoxon_balance'])
             if 'q_wilcoxon_balance' in rs.columns else np.nan)
        sig = np.isfinite(q) and q < SIG_ALPHA
        ax_b.plot([x[i], x[i]], [lo, hi], color='0.05' if sig else '0.55',
                  lw=2.3 if sig else 1.1, zorder=3,
                  alpha=1.0 if sig else 0.45, solid_capstyle='round')
        ax_b.scatter([x[i]], [m], s=80, zorder=4,
                     alpha=1.0 if sig else 0.45, facecolor=colours[i],
                     edgecolor='0.05' if sig else '0.55',
                     linewidth=1.3 if sig else 0.9)
    ax_b.axhline(0, color='0.45', lw=0.9, ls=':', zorder=0)
    ax_b.set_ylabel('position − phase-mod. (r)', labelpad=2)
    ax_b.set_title('b   Balance', loc='left', fontweight='bold')
    if omnibus and np.isfinite(omnibus.get('p_perm', np.nan)):
        ax_b.text(0.98, 0.03, f"perm. omnibus\nP = {omnibus['p_perm']:.3g}",
                  transform=ax_b.transAxes, ha='right', va='bottom',
                  fontsize=FONT_SMALL - 1, color='0.25')

    # ---- c: sustained fraction ------------------------------------------
    for i, roi in enumerate(rois):
        row = rs.iloc[i]
        k = float(row['n_sustained']); n = float(row['n_cells'])
        lo, hi = wilson_ci(k, n)
        sig = float(row['binom_p_sustained']) < SIG_ALPHA
        ax_c.bar(x[i], k / n, width=0.78, facecolor=colours[i],
                 edgecolor=colours[i], lw=1.0, zorder=2,
                 alpha=1.0 if sig else 0.45)
        ax_c.plot([x[i], x[i]], [lo, hi], color='0.05' if sig else '0.55',
                  lw=2.3 if sig else 1.1, zorder=3,
                  alpha=1.0 if sig else 0.45, solid_capstyle='round')
    ax_c.axhline(0.05, ls='--', lw=0.9, color='0.55', zorder=1)
    ax_c.set_xlim(-0.9, len(rois) - 0.4)
    ax_c.text(-0.84, 0.056, 'chance', fontsize=FONT_SMALL - 2, color='0.45',
              ha='left', va='bottom')
    ax_c.set_ylabel('fraction sustained (95% CI)', labelpad=2)
    ax_c.set_title('c   Sustained cells', loc='left', fontweight='bold')
    ax_c.set_ylim(0, None)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([_roi_label(r) for r in rois], rotation=30,
                           ha='right')
        _strip(ax)

    _add_caption(fig, "15_single_cell_summary", draw=False, text=(
        "How the position-in-sequence code is carried, per region. (a) The two "
        "components of each cell's held-out prediction: from the "
        "phase-invariant position block alone (filled) and the phase-modulated "
        "block alone (open). Point and whisker give the mean and its 95% CI. "
        "The per-cell distributions are an order of magnitude wider than these "
        "means (SD approx. 0.05 for each component, 0.06 for the balance), so "
        "these are small mean shifts in a noisy population; the inference is "
        "about the population shift, not about individual cells. The dotted line at zero is "
        "the permutation null, which both components share (measured means "
        "-0.000013 and +0.000002). Stars are the bias-free population-shift "
        "test (Wilcoxon of per-cell permutation p against 0.5), BH-FDR across "
        "the six regions; significance is shown by line weight and opacity "
        "rather than by stars (thick black q < .05, thin pale n.s.), and the "
        "phase-modulated component is drawn with a hatched marker. (b) The per-cell difference, same conventions, with "
        "a percentile bootstrap 95% CI of the mean (2,000 draws, seed 42); "
        "positive means the position code is carried mainly by the "
        "phase-invariant component. The omnibus permutes region labels across "
        "cells and recomputes Kruskal-Wallis H. (c) Fraction of cells whose "
        "minimum-across-phases preferred-state contrast exceeds its own "
        "permutation null, with Wilson 95% CIs against the 5% chance rate. "
        "ROIs ordered by balance throughout. * p<.05, ** p<.01, *** p<.001."))
    _save(fig, save_path)


def fig14_balance_vs_sustained(results, save_path):
    """Do the two per-cell measures agree?

    `min_phase_contrast` asks an ABSOLUTE question -- is the preferred state
    elevated at every phase -- while `balance` asks a RELATIVE one -- does the
    phase-invariant component out-predict the phase-modulated one. They are
    not the same statistic, so how far they agree is worth showing rather
    than assuming.

    (a) paired plot, one line per cell, both measures z-scored across the
        sample so the slope is meaningful.
    (b) the same data as a scatter, with Spearman rho.
    """
    _rc()
    d = results[['min_phase_contrast', 'balance', 'roi']].dropna()
    if len(d) < 10:
        return
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    zm = z(d['min_phase_contrast'].to_numpy())
    zb = z(d['balance'].to_numpy())
    rho, p_rho = stats.spearmanr(d['min_phase_contrast'], d['balance'])
    cols = [ROI_COLOURS.get(r, '#888888') for r in d['roi']]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14.0 * CM, 6.5 * CM),
                                      gridspec_kw={'wspace': 0.45})
    for i in range(len(d)):
        ax_a.plot([0, 1], [zm[i], zb[i]], color='0.6', lw=0.3, alpha=0.18,
                   zorder=1, solid_capstyle='butt')
    ax_a.scatter(np.zeros(len(d)), zm, s=5, c=cols, alpha=0.55,
                  linewidths=0, zorder=2)
    ax_a.scatter(np.ones(len(d)), zb, s=5, c=cols, alpha=0.55,
                  linewidths=0, zorder=2)
    ax_a.set_xlim(-0.35, 1.35)
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(['sustained\n(min phase contrast)', 'balance'])
    ax_a.set_ylabel('z-scored across cells')
    ax_a.set_title('a   Same cell, both measures', loc='left', fontweight='bold')
    _strip(ax_a)

    ax_b.axhline(0, color='0.85', lw=0.7, zorder=0)
    ax_b.axvline(0, color='0.85', lw=0.7, zorder=0)
    ax_b.scatter(d['min_phase_contrast'], d['balance'], s=6, c=cols,
                  alpha=0.55, linewidths=0, zorder=2)
    ax_b.set_xlabel('min phase contrast (Hz)')
    ax_b.set_ylabel('balance (r_state − r_inter)')
    ax_b.set_title(f"b   Spearman ρ = {rho:.2f}, P = {p_rho:.2g}",
                    loc='left', fontweight='bold')
    _strip(ax_b)
    handles = [mpatches.Patch(facecolor=ROI_COLOURS.get(r, '#888888'),
                               edgecolor='none', label=_roi_label(r))
               for r in ROI_ORDER if r in set(d['roi'])]
    ax_b.legend(handles=handles, frameon=False, fontsize=FONT_SMALL - 2,
                 loc='upper left', handlelength=0.9, borderaxespad=0.1,
                 ncol=2)

    _add_caption(fig, "14_balance_vs_sustained", draw=False, text=(
        f"Agreement between the two per-cell measures. (a) One line per cell "
        f"({len(d)} cells), both measures z-scored across the sample so the "
        f"slope is interpretable; point colour is ROI. (b) The same data "
        f"unstandardised, with Spearman rho = {rho:.2f} (P = {p_rho:.2g}). "
        f"`min_phase_contrast` is the minimum across phases of the held-out "
        f"preferred-state-minus-others contrast, an ABSOLUTE test of whether "
        f"the preferred state is elevated at every phase. `balance` is "
        f"r_state - r_interaction, a RELATIVE measure of which component "
        f"out-predicts the other. They answer different questions, so partial "
        f"agreement is expected."))
    _save(fig, save_path)


def fig13_balance(results, roi_summary, save_path, omnibus=None):
    """Phase-invariant vs phase-modulated generalisation, per ROI.

    Panel a shows the two COMPONENTS separately (project rule: never plot a
    contrast in raw units without its parts), each with its permutation-null
    mean as a reference, so any block-size bias is visible. Panel b shows
    their per-cell difference with a bootstrap CI and the permutation omnibus.
    """
    _rc()
    rs = roi_summary[roi_summary['n_cells'] > 0].copy()
    rs = rs.sort_values('mean_balance', ascending=False)
    rois = rs['roi'].tolist()
    x = np.arange(len(rois), dtype=float)
    colours = [ROI_COLOURS.get(r, '#888888') for r in rois]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(16.5 * CM, 6.5 * CM),
        gridspec_kw={'width_ratios': [1.3, 1.0], 'wspace': 0.42})

    w = 0.36
    for i, roi in enumerate(rois):
        cells = results.loc[results['roi'] == roi]
        for j, (col, nullcol, filled) in enumerate([
                ('r_state', 'null_mean_r_state', True),
                ('r_interaction', 'null_mean_r_interaction', False)]):
            v = cells[col].dropna().to_numpy()
            if v.size == 0:
                continue
            m = float(v.mean()); se = float(v.std(ddof=1) / np.sqrt(v.size))
            xi = x[i] + (j - 0.5) * w
            ax_a.bar(xi, m, width=w * 0.92,
                      facecolor=colours[i] if filled else 'white',
                      edgecolor=colours[i], linewidth=1.1,
                      hatch=None if filled else '///', zorder=2)
            ax_a.plot([xi, xi], [m - 1.96 * se, m + 1.96 * se],
                       color='0.25', lw=1.0, zorder=3)
            if nullcol in cells.columns:
                nm = float(cells[nullcol].dropna().mean())
                ax_a.plot([xi - w * 0.46, xi + w * 0.46], [nm, nm],
                           color='0.35', lw=1.0, ls=':', zorder=4)
    ax_a.axhline(0, color='0.7', lw=0.8, zorder=0)
    ax_a.set_xticks(x); ax_a.set_xticklabels([_roi_label(r) for r in rois],
                                              rotation=30, ha='right')
    ax_a.set_ylabel('held-out r (mean ± 95% CI)')
    ax_a.set_title('a   Components', loc='left', fontweight='bold')
    _strip(ax_a)
    ax_a.legend(handles=[
        mpatches.Patch(facecolor='0.45', edgecolor='0.2',
                       label='phase-invariant (r_state)'),
        mpatches.Patch(facecolor='white', edgecolor='0.2', hatch='///',
                       label='phase-modulated (r_inter)'),
        plt.Line2D([], [], color='0.35', ls=':', lw=1.0,
                   label='permutation null')],
        frameon=False, loc='upper right', handlelength=1.3,
        borderaxespad=0.1, fontsize=FONT_SMALL - 1)

    rng = np.random.default_rng(42)
    for i, roi in enumerate(rois):
        v = results.loc[results['roi'] == roi, 'balance'].dropna().to_numpy()
        if v.size < 3:
            continue
        m = float(v.mean())
        boot = rng.choice(v, size=(2000, v.size), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ax_b.bar(x[i], m, width=0.6, facecolor=colours[i],
                  edgecolor=colours[i], linewidth=1.0, zorder=2)
        ax_b.plot([x[i], x[i]], [lo, hi], color='0.25', lw=1.0, zorder=3)
    ax_b.axhline(0, color='0.7', lw=0.8, zorder=0)
    ax_b.set_xticks(x); ax_b.set_xticklabels([_roi_label(r) for r in rois],
                                              rotation=30, ha='right')
    ax_b.set_ylabel('balance\n(phase-invariant − phase-modulated)')
    ttl = 'b   Balance'
    if omnibus and np.isfinite(omnibus.get('p_perm', np.nan)):
        ttl += (f"   (perm omnibus H = {omnibus['stat_H']:.1f}, "
                f"P = {omnibus['p_perm']:.3g})")
    ax_b.set_title(ttl, loc='left', fontweight='bold')
    _strip(ax_b)

    _add_caption(fig, "13_balance", draw=False, text=(
        "Phase-invariant versus phase-modulated generalisation of the "
        "position code, per ROI. (a) The two components separately: mean "
        "held-out correlation between the left-out configuration's activity "
        "and the prediction from the state block alone (filled) or the "
        "state x phase block alone (hatched); error bars 95% CI over cells; "
        "dotted lines the mean of each component's own circular-shift "
        "permutation null, shown so that any difference in null position "
        "between the 4-column and 8-column blocks is visible rather than "
        "assumed absent. (b) Their per-cell difference, with a percentile "
        "bootstrap 95% CI over cells (2,000 draws, seed 42). Positive means "
        "the position code is carried mainly by the phase-invariant "
        "component. The omnibus is a permutation test that shuffles ROI "
        "labels across cells and recomputes the Kruskal-Wallis statistic. "
        "ROIs ordered by mean balance."))
    _save(fig, save_path)


def fig12_main_result(results, roi_summary, save_path):
    """THE main-result figure: is the position code sustained or phase-driven?

    Panel a (cell level)       fraction of cells classified sustained vs
                               phasic-only, per ROI, with Wilson 95% CIs and
                               the 5% chance rate. CIs are the point of the
                               panel: EC's 31% rests on 35 cells, and the bare
                               bar in the old fig 8b invited the reader to
                               over-read it.
    Panel b (population level) rank-biserial effect size of the population
                               shift for the sustained statistic and for the
                               phase-modulated component, same ROI order, with
                               bootstrap CIs over cells. Every region shows
                               both EXCEPT EC, whose phase-modulation evidence
                               is absent -- which is the actual headline.

    ROIs are ordered by sustained fraction (descending) in BOTH panels; this
    is stated in the caption.
    """
    _rc()
    rs = roi_summary.copy()
    rs = rs[rs['n_cells'] > 0].sort_values('frac_sustained', ascending=False)
    rois = rs['roi'].tolist()
    n = len(rois)
    x = np.arange(n, dtype=float)
    colours = [ROI_COLOURS.get(r, '#888888') for r in rois]

    # ~16.5 cm x 6.5 cm total: a full-text-width two-panel main figure on A4.
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(16.5 * CM, 6.5 * CM),
        gridspec_kw={'width_ratios': [1.25, 1.0], 'wspace': 0.42})

    # ---------------- panel a: cell-level fractions ----------------
    w = 0.36
    for i, roi in enumerate(rois):
        row = rs.iloc[i]
        for j, (kcol, ncol, pcol, filled) in enumerate([
                ('n_sustained',   'n_cells', 'binom_p_sustained',   True),
                ('n_phasic_only', 'n_cells', 'binom_p_phasic_only', False)]):
            k = float(row[kcol]); nn = float(row[ncol])
            frac = k / nn if nn else np.nan
            lo, hi = wilson_ci(k, nn)
            xi = x[i] + (j - 0.5) * w
            ax_a.bar(xi, frac, width=w * 0.92,
                      facecolor=colours[i] if filled else 'white',
                      edgecolor=colours[i], linewidth=1.1,
                      hatch=None if filled else '///', zorder=2)
            ax_a.plot([xi, xi], [lo, hi], color='0.25', lw=1.0, zorder=3,
                       solid_capstyle='butt')
            star = _stars_from_p(row[pcol])
            if star:
                ax_a.text(xi, hi + 0.012, star, ha='center', va='bottom',
                           fontsize=FONT_SMALL - 1, color='0.25',
                           rotation=90 if star == 'n.s.' else 0)
    ax_a.axhline(0.05, ls='--', lw=0.9, color='0.55', zorder=1)
    ax_a.set_xlim(-0.95, n - 0.35)
    ax_a.text(-0.88, 0.052, 'chance', fontsize=FONT_SMALL - 1,
               color='0.45', ha='left', va='bottom')
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([_roi_label(r) for r in rois], rotation=30, ha='right')
    ax_a.set_ylabel('fraction of cells\n(95% Wilson CI)')
    ax_a.set_title('a   Cell level', loc='left', fontweight='bold')
    ax_a.set_ylim(0, None)
    _strip(ax_a)
    handles = [mpatches.Patch(facecolor='0.45', edgecolor='0.2',
                               label='sustained'),
               mpatches.Patch(facecolor='white', edgecolor='0.2',
                               hatch='///', label='phase-driven')]
    ax_a.legend(handles=handles, frameon=False, loc='upper right',
                 handlelength=1.2, borderaxespad=0.1)

    # ---------------- panel b: population-level effect sizes ----------------
    spec = [('p_perm_min_phase_contrast', 'q_wilcoxon_min_phase',   True),
            ('p_perm_r_interaction',      'q_wilcoxon_r_interaction', False)]
    for i, roi in enumerate(rois):
        cells = results.loc[results['roi'] == roi]
        pts = []
        for j, (pcol, qcol, filled) in enumerate(spec):
            vals = cells[pcol].dropna().to_numpy() if pcol in cells else np.array([])
            eff = rank_biserial_below_half(vals)
            lo, hi = _boot_ci_rank_biserial(vals)
            xi = x[i] + (j - 0.5) * 0.30
            pts.append((xi, eff))
            if np.isfinite(lo):
                ax_b.plot([xi, xi], [lo, hi], color='0.25', lw=1.0, zorder=2)
            ax_b.plot(xi, eff, marker='o', ms=6.0, zorder=3,
                       markerfacecolor=colours[i] if filled else 'white',
                       markeredgecolor=colours[i], markeredgewidth=1.4)
            q = float(rs.iloc[i][qcol]) if qcol in rs.columns else np.nan
            ax_b.text(xi, hi + 0.03 if np.isfinite(hi) else eff + 0.03,
                       _stars_from_p(q), ha='center', va='bottom',
                       fontsize=FONT_SMALL - 1, color='0.25',
                       rotation=90 if _stars_from_p(q) == 'n.s.' else 0)
        # connector makes the EC drop visible at a glance
        ax_b.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]],
                   color=colours[i], lw=1.0, alpha=0.55, zorder=1)
    ax_b.axhline(0.0, ls='-', lw=0.8, color='0.7', zorder=0)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([_roi_label(r) for r in rois], rotation=30, ha='right')
    ax_b.set_ylabel('population shift\n(rank-biserial, 95% CI)')
    ax_b.set_title('b   Population level', loc='left', fontweight='bold')
    _strip(ax_b)
    handles_b = [plt.Line2D([], [], marker='o', ls='none', ms=6,
                             markerfacecolor='0.45', markeredgecolor='0.2',
                             label='sustained'),
                 plt.Line2D([], [], marker='o', ls='none', ms=6,
                             markerfacecolor='white', markeredgecolor='0.2',
                             label='phase-driven')]
    ax_b.legend(handles=handles_b, frameon=False, loc='lower left',
                 handlelength=1.0, borderaxespad=0.1)

    _add_caption(fig, "12_main_result", draw=False, text=(
                  "Position-in-sequence coding is sustained in EC and "
                  "phase-driven elsewhere. ROIs are ordered by sustained "
                  "fraction (descending) in both panels. (a) Fraction of "
                  "cells classified sustained (filled) and phasic-only "
                  "(hatched) per ROI; error bars are Wilson 95% CIs, dashed "
                  "line the 5% rate expected by chance, stars per-ROI "
                  "binomial tests against that rate. (b) Effect size of the "
                  "population-shift test (matched-pairs rank-biserial "
                  "correlation of per-cell permutation-p against 0.5; "
                  "positive = shifted in the predicted direction) for the "
                  "sustained statistic and for the phase-modulated "
                  "component; error bars are percentile bootstrap 95% CIs "
                  "over cells (2,000 draws, seed 42), stars are BH-FDR "
                  "across the six ROIs. Every region carries both components "
                  "except EC, where the phase-modulated component is absent. "
                  "* p<.05, ** p<.01, *** p<.001."))
    _save(fig, save_path)


def fig11_wilcoxon_heatmap(results, roi_summary, save_path):
    _rc()
    rois = roi_summary['roi'].tolist()
    n_rows = len(rois)
    # Compute per-ROI one-sample t-vs-0 for all three within-script stats.
    t_enc  = np.full((n_rows, 2), np.nan)   # r_state, r_interaction
    t_minp = np.full((n_rows, 1), np.nan)   # min_phase_contrast
    raw_enc = np.full((n_rows, 2), np.nan); fdr_enc = np.full((n_rows, 2), np.nan)
    raw_mp  = np.full((n_rows, 1), np.nan); fdr_mp  = np.full((n_rows, 1), np.nan)
    for i, roi in enumerate(rois):
        rdf = results[results['roi'] == roi]
        for j, stat in enumerate(['r_state', 'r_interaction']):
            v = rdf[stat].dropna().to_numpy()
            if v.size >= 5:
                t_enc[i, j] = float(stats.ttest_1samp(v, 0.0).statistic)
        v = rdf['min_phase_contrast'].dropna().to_numpy()
        if v.size >= 5:
            t_minp[i, 0] = float(stats.ttest_1samp(v, 0.0).statistic)
        rs = roi_summary[roi_summary['roi'] == roi].iloc[0]
        raw_enc[i, 0] = rs.get('wilcoxon_p_r_state', np.nan)
        raw_enc[i, 1] = rs.get('wilcoxon_p_r_interaction', np.nan)
        raw_mp[i, 0]  = rs.get('wilcoxon_p_min_phase', np.nan)
        fdr_enc[i, 0] = rs.get('q_wilcoxon_r_state', np.nan)
        fdr_enc[i, 1] = rs.get('q_wilcoxon_r_interaction', np.nan)
        fdr_mp[i, 0]  = rs.get('q_wilcoxon_min_phase', np.nan)

    # RSA-state column from the companion run (4th panel).
    t_rsa, raw_rsa, fdr_rsa = _load_rsa_state_per_roi(rois)
    t_rsa  = t_rsa.reshape(-1, 1)
    raw_rsa = raw_rsa.reshape(-1, 1)
    fdr_rsa = fdr_rsa.reshape(-1, 1)

    # Total figure 15 × 5.5 cm. Cells marginally wider than the original
    # (not 2x). Colour-bars placed HORIZONTALLY below each panel so they
    # don't eat horizontal cell-width — leaves room for the 12 pt bold
    # in-cell stars without squeezing the cells.
    fig = plt.figure(figsize=(10 * CM, 13 * CM), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1.0, 1.0],
                          wspace=0.08)
    # gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.5, 0.5],
    #                       wspace=0.35)
    axE = fig.add_subplot(gs[0, 0])  # encoding correlations
    axM = fig.add_subplot(gs[0, 1])  # min_phase_contrast (biased)
    axR = fig.add_subplot(gs[0, 2])  # RSA state

    # Encoding panel (RdBu_r, null mean = 0).
    vmax_E = max(1.0, float(np.nanmax(np.abs(t_enc)))) if np.isfinite(t_enc).any() else 1.0
    imE = axE.imshow(t_enc, cmap='RdBu_r', vmin=-vmax_E, vmax=vmax_E, aspect='auto')
    # Min phase panel (PuOr_r, biased null mean < 0).
    vmax_M = max(1.0, float(np.nanmax(np.abs(t_minp)))) if np.isfinite(t_minp).any() else 1.0
    imM = axM.imshow(t_minp, cmap='PuOr_r', vmin=-vmax_M, vmax=vmax_M, aspect='auto')
    # RSA-state panel (PiYG_r — third distinct diverging map; null mean = 0).
    if np.isfinite(t_rsa).any():
        vmax_R = max(1.0, float(np.nanmax(np.abs(t_rsa))))
    else:
        vmax_R = 1.0
    imR = axR.imshow(t_rsa, cmap='PiYG_r', vmin=-vmax_R, vmax=vmax_R, aspect='auto')

    # FDR-only significance: bold stars inside each cell driven by the
    # BH-FDR-corrected q-value. No raw-p borders — they were visually
    # ambiguous next to the no-border ns cells.
    def _stars(q):
        if not np.isfinite(q): return ''
        if q < .001: return '***'
        if q < .01:  return '**'
        if q < .05:  return '*'
        return ''

    def _annotate(ax, fdr_q, n_cols, vmax):
        img_arr = ax.images[0].get_array() if ax.images else None
        for i in range(n_rows):
            for j in range(n_cols):
                st = _stars(fdr_q[i, j])
                if not st:
                    continue
                # White star on dark cells, black on light cells.
                raw_val = (float(img_arr[i, j]) if img_arr is not None
                           else 0.0)
                intensity = abs(raw_val) / max(vmax, 1e-9)
                star_colour = 'white' if intensity > 0.55 else 'black'
                ax.text(j, i, st, ha='center', va='center',
                        fontsize=FONT_BIG + 1, fontweight='bold',
                        color=star_colour, zorder=5)
    _annotate(axE, fdr_enc, 2, vmax_E)
    _annotate(axM, fdr_mp,  1, vmax_M)
    _annotate(axR, fdr_rsa, 1, vmax_R)

    # font window is 9 → 11 pt. Within that window we use the
    # UPPER half for this heatmap so the 2.5 cm panel is easy to read:
    f_lab = FONT_AXIS         # 10 pt — tick labels (ROI names, column labels)
    f_ttl = FONT_BIG          # 11 pt — panel titles
    f_cb  = FONT_AXIS         # 10 pt — colour-bar label + ticks

    def _style_panel(ax, im, xticklabels, title, cbar_label, frac=0.06):
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=30, ha='right', fontsize=f_lab)
        ax.set_yticks(range(n_rows))
        ax.set_title(title, fontsize=f_ttl, pad=2)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis='both', length=1.5, pad=1)
        # Horizontal colour-bar below the heatmap, so it doesn't eat width.
        cb = fig.colorbar(im, ax=ax, orientation='horizontal',
                          location='bottom', fraction=0.12, pad=0.55,
                          shrink=0.9, aspect=5)
        cb.set_label(cbar_label, fontsize=f_cb)
        cb.ax.tick_params(labelsize=f_cb)
        return cb

    _style_panel(axE, imE, ['r_state', 'r_inter'],
                  'encoding\n(null = 0)', 't vs 0')
    axE.set_yticklabels(rois, fontsize=f_lab)

    _style_panel(axM, imM, ['min phase\ncontrast'],
                  'sustained\n(null < 0)', 't vs 0  (biased)')
    axM.set_yticklabels([])

    _style_panel(axR, imR, ['state\n(RSA)'],
                  'RSA state\n(companion)', 't vs 0')
    axR.set_yticklabels([])
    #import pdb; pdb.set_trace()
    _add_caption(fig, "11_effect_size_heatmap",
                  "Effect-size heatmap, ROI × statistic. Colour = one-sample "
                  "t-statistic of per-cell empirical values vs 0 (a real "
                  "effect-size measure, not a p-value). LEFT panel "
                  "(`r_state`, `r_interaction`) uses RdBu_r — both stats are "
                  "correlations with null mean = 0, so t-vs-0 is interpretable "
                  "as effect size. MIDDLE panel (`min_phase_contrast`) uses "
                  "PuOr_r and its own scale because the perm null is biased "
                  "downward (≈ −0.846 σ_cell for min-of-3 noise) — t-vs-0 "
                  "here is mechanically biased toward negative, so only the "
                  "cross-ROI ordering is meaningful. RIGHT panel: companion "
                  "RSA analysis from "
                  f"`DSR_RSA_simple_ROI/{os.path.basename(RSA_STATE_RUN_DIR.rstrip('/'))}/results_"
                  f"summary_combos.csv` (test=`{RSA_TEST_VARIANT}`, "
                  f"sub_model=`{RSA_SUB_MODEL}`), in PiYG_r. Significance is "
                  "shown as BOLD STARS inside each cell (FDR-corrected ONLY; "
                  "raw-p markers were dropped because they are visually "
                  "ambiguous next to no-mark cells): * < .05, ** < .01, "
                  "*** < .001 (BH-FDR across 7 ROIs). Cells without a star "
                  "are not FDR-significant. Within-script panels use the "
                  "bias-free Wilcoxon perm-p shift test for the FDR p-values; "
                  "the RSA panel uses BH-FDR of the RSA permutation p "
                  "(`p_perm` re-corrected here across the 7 ROIs because "
                  "`in_fdr_family=False` in the RSA CSV).")
    _save(fig, save_path)


# ---------------------------------------------------------------------
# Figures 11-13 — Multiple example cells (polar) by category
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Example-cell gallery — one PDF per cell, many cells per category, so the
# user can browse and pick the cleanest visual exemplars.
# Each panel is ~3 × 3 cm with a 0.7 cm-radius polar and big A/B/C/D letters.
# Per-cell rescaling (no shared rlim) so each cell reads at its own scale.
# ---------------------------------------------------------------------
def _polar_panel_for_cell(trace, ax, panel_label):
    """One panel in the per-cell gallery page. Big A/B/C/D letters, polar grid
    kept (concentric circles) for spatial reference, no tick labels.
    Per-panel rescaling — scale varies by cell so each cell reads at its own
    level."""
    sm = smooth_circular(trace, sigma=4)
    plot_state_polar_clock(
        sm, title_string='', ax=ax,
        rlim=(float(np.nanmin(sm)), float(np.nanmax(sm))),
        fontsize_labels=FONT_BIG + 1, fontsize_title=FONT_SMALL,
        title_pad=2,
    )
    ax.text(0.5, -0.05, panel_label, transform=ax.transAxes,
            ha='center', va='top', fontsize=FONT_SMALL, color='#555')
    ax.set_yticks([]); ax.set_yticklabels([])
    ax.grid(True, lw=0.4, color='#ccc', alpha=0.7)


def _cell_polar_page(cell_row, cell_data, save_path):
    """One PDF for one cell: mean polar + one polar per configuration."""
    _rc()
    beh = cell_data['beh']; neuron_df = cell_data['neuron_trace']
    correct = (beh['correct'] == 1).to_numpy()
    configs = sorted(beh.loc[correct, 'config_str'].dropna().unique().tolist())
    per_cfg = []; cfg_labels = []
    for cfg in configs:
        idx = beh.index[(beh['config_str'] == cfg) & (beh['correct'] == 1)].to_numpy()
        if len(idx) == 0: continue
        per_cfg.append(np.nanmean(neuron_df.iloc[idx].to_numpy(dtype=float), axis=0))
        cfg_labels.append(cfg)
    if not per_cfg: return
    mean_trace = np.nanmean(np.stack(per_cfg, axis=0), axis=0)

    n_panels = 1 + len(per_cfg)
    n_cols = min(n_panels, 4)
    n_rows = int(np.ceil(n_panels / n_cols))
    # Each panel ~3 cm × 3 cm. Extra space for the title strip.
    fig_w = (3 * n_cols + 0.4) * CM
    fig_h = (3 * n_rows + 1.0) * CM
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    title = (f"{cell_row['neuron']}  [{cell_row['roi']}]    "
             f"pref={cell_row['pref_state_mode']}    "
             f"min_pc={cell_row['min_phase_contrast']:+.3f}  "
             f"(p={cell_row['p_perm_min_phase_contrast']:.3f})    "
             f"r_state={cell_row['r_state']:+.3f}  "
             f"(p={cell_row['p_perm_r_state']:.3f})    "
             f"r_int={cell_row['r_interaction']:+.3f}  "
             f"(p={cell_row['p_perm_r_interaction']:.3f})")
    fig.suptitle(title, fontsize=FONT_SMALL, y=0.97)

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n_rows, n_cols, figure=fig,
                  left=0.02, right=0.98, top=0.88, bottom=0.04,
                  wspace=0.05, hspace=0.05)

    # Panel 0: mean across configs
    ax = fig.add_subplot(gs[0, 0], projection='polar')
    _polar_panel_for_cell(mean_trace, ax, 'mean')

    # Per-config
    for i, (cfg, trace) in enumerate(zip(cfg_labels, per_cfg)):
        idx = i + 1
        r, c = idx // n_cols, idx % n_cols
        ax = fig.add_subplot(gs[r, c], projection='polar')
        _polar_panel_for_cell(trace, ax, f'cfg {cfg}')

    # Hide empty axes
    for k in range(n_panels, n_rows * n_cols):
        r, c = k // n_cols, k % n_cols
        fig.add_subplot(gs[r, c]).axis('off')

    fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def build_example_gallery(results, out_root, top_n=20):
    """Build gallery PDFs per category. Many cells per category, so the user
    can browse a folder of options."""
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    categories = [
        ('sustained',
         results.loc[results['sig_sustained']].sort_values(
             'min_phase_contrast', ascending=False)),
        ('phasic_only',
         results.loc[(results['sig_r_state'] | results['sig_r_interaction'])
                      & (~results['sig_sustained'])].sort_values(
             'r_interaction', ascending=False)),
        ('non_state',
         results.loc[(~results['sig_sustained']) & (~results['sig_r_state'])
                      & (~results['sig_r_interaction'])].sort_values(
             'min_phase_contrast', ascending=True)),
    ]
    for cat_name, df in categories:
        cat_dir = out_root / cat_name
        cat_dir.mkdir(exist_ok=True)
        rois_present = df['roi'].unique().tolist()
        df_by_roi = {r: df[df['roi'] == r].copy() for r in rois_present}
        picks = []
        # EC priority — pick up to 5 EC cells first so the most-interesting
        # ROI for sustained state is always represented in the gallery.
        EC_PRIORITY_N = 5
        if 'EC' in df_by_roi:
            ec_take = min(EC_PRIORITY_N, len(df_by_roi['EC']))
            for k in range(ec_take):
                picks.append(df_by_roi['EC'].iloc[0])
                df_by_roi['EC'] = df_by_roi['EC'].iloc[1:]
        # Round-robin across remaining ROIs to fill the rest.
        i = 0
        while len(picks) < top_n and any(len(v) > 0 for v in df_by_roi.values()):
            r = rois_present[i % len(rois_present)] if rois_present else None
            if r is not None and len(df_by_roi[r]) > 0:
                picks.append(df_by_roi[r].iloc[0])
                df_by_roi[r] = df_by_roi[r].iloc[1:]
            i += 1
        print(f"  gallery — {cat_name}: {len(picks)} cells")
        for rank, row in enumerate(picks, start=1):
            row_dict = row.to_dict() if hasattr(row, 'to_dict') else row
            cell_data = load_cell_data(row_dict['neuron'])
            if cell_data is None: continue
            cell_data = {'beh': cell_data['beh'],
                         'neuron_trace': cell_data['neuron_trace']}
            stem = (f"{rank:02d}_sub-{int(row_dict['subject_int']):02d}_"
                    f"{row_dict['roi']}_{row_dict['neuron'].replace('/', '_')[:30]}")
            try:
                _cell_polar_page(row_dict, cell_data, cat_dir / f"{stem}.pdf")
            except Exception as e:
                print(f"    failed {stem}: {e}")


# ---------------------------------------------------------------------
# Example-cell loading
# ---------------------------------------------------------------------
def parse_neuron_label(label):
    try:
        sub_str, rest = str(label).split("_", 1)
        cell_idx_str = rest.split("-", 1)[0]
        return int(sub_str), int(cell_idx_str)
    except (ValueError, IndexError):
        return None, None


def load_cell_data(neuron_id):
    sub, _ = parse_neuron_label(neuron_id)
    if sub is None: return None
    sub_str = f"{sub:02d}"
    try:
        data = hh.load_norm_data(DATA_DIR, [sub_str], res_data=False)
    except Exception as exc:
        print(f"  load failed for sub-{sub_str}: {exc}"); return None
    key = f"sub-{sub_str}"
    if key not in data: return None
    neurons = data[key]["normalised_neurons"]
    if neuron_id not in neurons: return None
    beh = add_config_str(data[key]["beh"])
    return {"neuron_id": neuron_id, "beh": beh,
            "neuron_trace": neurons[neuron_id].reset_index(drop=True)}


def pick_examples(results, sustained_n=9, phasic_n=9, none_n=9):
    """Pick example cells for each category, spreading across ROIs where possible."""
    def _spread_across_rois(df, n_want):
        df = df.copy()
        if df.empty: return df
        rois = df['roi'].unique().tolist()
        out = []
        i = 0
        df_sorted_by_roi = {r: df[df['roi'] == r].copy() for r in rois}
        # Round-robin across ROIs
        while len(out) < n_want and any(len(v) > 0 for v in df_sorted_by_roi.values()):
            r = rois[i % len(rois)]
            if len(df_sorted_by_roi[r]) > 0:
                out.append(df_sorted_by_roi[r].iloc[0])
                df_sorted_by_roi[r] = df_sorted_by_roi[r].iloc[1:]
            i += 1
        return pd.DataFrame(out)

    sustained = (results.loc[results['sig_sustained']]
                         .sort_values('min_phase_contrast', ascending=False))
    sustained_picks = _spread_across_rois(sustained, sustained_n)

    any_state = results['sig_r_state'] | results['sig_r_interaction']
    phasic = (results.loc[any_state & (~results['sig_sustained'])]
                      .sort_values('r_interaction', ascending=False))
    phasic_picks = _spread_across_rois(phasic, phasic_n)

    none = (results.loc[(~results['sig_sustained']) & (~results['sig_r_state']) & (~results['sig_r_interaction'])]
                     .sort_values('min_phase_contrast', ascending=True))   # most negative min_pc
    none_picks = _spread_across_rois(none, none_n)

    return sustained_picks, phasic_picks, none_picks


def pick_perm_validation_cells(results):
    """4 cells across the significance spectrum to plot perm nulls for."""
    sustained = (results.loc[results['sig_sustained']]
                         .sort_values('min_phase_contrast', ascending=False))
    near_b = (results.loc[(results['p_perm_min_phase_contrast'] >= SIG_ALPHA)
                           & (results['p_perm_min_phase_contrast'] < 0.2)]
                       .sort_values('min_phase_contrast', ascending=False))
    none_b = (results.loc[(~results['sig_sustained']) & (~results['sig_r_state'])]
                       .sort_values('min_phase_contrast', ascending=True))
    picks = []
    if len(sustained):
        picks.append(('strong sustained', sustained.iloc[0].to_dict()))
    if len(near_b):
        picks.append(('borderline (p≈0.1)', near_b.iloc[0].to_dict()))
    if len(none_b) > 2:
        picks.append(('null (median negative)', none_b.iloc[len(none_b) // 2].to_dict()))
    if len(none_b):
        picks.append(('null (most negative)', none_b.iloc[0].to_dict()))
    return picks


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default=SUBJECTS_TO_RUN)
    parser.add_argument("--rois", default=None)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--max-neurons-per-subject", type=int, default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument(
        "--load-old-results", default=RELOAD_OLD_RESULTS,
        help="Path to an existing run directory OR a run tag under OUT_BASE. "
             "Skips analysis and only generates the publication figures from "
             "the saved CSV in that run dir. Defaults to the module-level "
             "constant RELOAD_OLD_RESULTS (set to None to force a full run).")
    parser.add_argument(
        "--relabel-from", default=RELABEL_FROM,
        help="Optional. Path to a fresh neurons_with_ROI_labels.csv. "
             "Defaults to the module-level constant RELABEL_FROM (set to "
             "None there to skip). Only valid together with "
             "--load-old-results. If set, the per-cell CSV's `roi` column "
             "is overwritten by that table's `alt_final_roi` column "
             "(joined on subject + cell_idx), and the ROI summary + "
             "figures are rebuilt into a NEW sibling directory "
             "'<original>_relabelled_<timestamp>/' so the original run "
             "stays intact. Prints a full transitions audit.")
    parser.add_argument(
        "--gallery", action="store_true",
        help="Also build the per-cell example gallery (slow; loads raw data).")
    parser.add_argument(
        "--gallery-n", type=int, default=20,
        help="Number of cells per category in the gallery.")
    return parser.parse_args()


def _resolve_load_path(arg):
    """`--load-old-results foo` can be either a run tag under OUT_BASE or a path."""
    if arg is None: return None
    p = Path(arg)
    if p.exists() and p.is_dir():
        return p
    p_under_base = Path(OUT_BASE) / arg
    if p_under_base.exists():
        return p_under_base
    raise FileNotFoundError(f"Cannot resolve --load-old-results '{arg}'")


def _build_all_figures(results, roi_summary, diag_dir, n_permutations,
                       build_gallery=False, gallery_n=20):
    """Generate all publication figures from the in-memory results + summary."""
    diag_dir = Path(diag_dir); diag_dir.mkdir(parents=True, exist_ok=True)

    # Fig 01 — design matrix (build a small example from the data itself).
    try:
        # Use sub-01 (or first sub in results) to build a representative design.
        first_sub = f"{int(results.iloc[0]['subject_int']):02d}"
        data = hh.load_norm_data(DATA_DIR, [first_sub], res_data=False)
        sub_dict = data[f"sub-{first_sub}"]
        beh = add_config_str(sub_dict["beh"])
        configs = sorted(beh["config_str"].dropna().unique().tolist())
        n_cfgs = len(configs)
        state_idx_full  = np.tile(STATE_IDX_360, n_cfgs)
        phase_idx_full  = np.tile(PHASE_IDX_360, n_cfgs)
        config_idx_full = np.concatenate(
            [np.full(N_BINS_PER_TRIAL, i, dtype=int) for i in range(n_cfgs)])
        X_demo, slices_demo = build_design(
            state_idx_full, phase_idx_full, config_idx_full,
            list(range(n_cfgs - 1)))
        fig01_design_matrix(X_demo, slices_demo, str(diag_dir / "01_design_matrix.png"))
        print("  saved 01_design_matrix")
    except Exception as e:
        print(f"  fig01 failed: {e}")

    try:
        fig02_pipeline_stats(results, str(diag_dir / "02_pipeline_run_stats.png"))
        print("  saved 02_pipeline_run_stats")
    except Exception as e:
        print(f"  fig02 failed: {e}")

    # Fig 03 — needs cell refits to capture perm distributions.
    try:
        examples = pick_perm_validation_cells(results)
        refits = []
        for label, ex_row in examples:
            sub_str = f"{int(ex_row['subject_int']):02d}"
            try:
                data = hh.load_norm_data(DATA_DIR, [sub_str], res_data=False)
                sub_dict = data[f"sub-{sub_str}"]
                beh = add_config_str(sub_dict["beh"])
                configs = sorted(beh["config_str"].dropna().unique().tolist())
                neurons_raw = sub_dict["normalised_neurons"]
                if ex_row['neuron'] not in neurons_raw: continue
                neurons_used = {ex_row['neuron']:
                                neurons_raw[ex_row['neuron']].reset_index(drop=True)}
                Y_dict, _, state_block, phase_block = build_y_per_config(
                    beh, neurons_used, configs)
                n_cfgs_local = Y_dict[ex_row['neuron']].shape[0]
                if n_cfgs_local < 3: continue
                state_idx_full = np.tile(state_block, n_cfgs_local)
                phase_idx_full = np.tile(phase_block, n_cfgs_local)
                config_idx_full = np.concatenate(
                    [np.full(N_BINS_PER_TRIAL, i, dtype=int) for i in range(n_cfgs_local)])
                rf = analyse_one_neuron(
                    ex_row['neuron'], ex_row['roi'], Y_dict[ex_row['neuron']],
                    state_idx_full, phase_idx_full, config_idx_full, list(range(n_cfgs_local)),
                    n_permutations,
                    seed=cell_seed(sub_str, ex_row['neuron'], "state_sustained_v4"),
                    return_perm_dists=True,
                )
                refits.append((label, rf))
            except Exception as e:
                print(f"    refit failed for {label}: {e}")
        if refits:
            fig03_perm_nulls(refits, str(diag_dir / "03_perm_nulls_examples.png"))
            print("  saved 03_perm_nulls_examples")
    except Exception as e:
        print(f"  fig03 failed: {e}")

    # Pull the BH-FDR-corrected Wilcoxon perm-p stars per ROI (BH across the
    # 7 ROIs). Bonferroni-FWE is also stored as `fwe_wilcoxon_<stat>` but
    # the figure stars use BH-FDR (per user preference).
    def _q_dict(stat_prefix):
        col = f'q_wilcoxon_{stat_prefix}'
        if col not in roi_summary.columns: return None
        return dict(zip(roi_summary['roi'], roi_summary[col]))

    # Fig 05/06/6b — distributions per ROI
    try:
        _fig_stat_dist_per_roi(
            results, 'min_phase_contrast', 'min phase contrast',
            str(diag_dir / "05_dist_min_phase_contrast_per_roi.png"),
            color='lightgray', sig_col='sig_sustained',
            q_per_roi=_q_dict('min_phase'),
        )
        print("  saved 05")
    except Exception as e:
        print(f"  fig05 failed: {e}")
    try:
        _fig_stat_dist_per_roi(
            results, 'r_state', 'held-out r_state',
            str(diag_dir / "06_dist_r_state_per_roi.png"),
            color='lightgray', sig_col='sig_r_state',
            q_per_roi=_q_dict('r_state'),
        )
        print("  saved 06")
    except Exception as e:
        print(f"  fig06 failed: {e}")
    try:
        _fig_stat_dist_per_roi(
            results, 'r_interaction', 'held-out r_interaction',
            str(diag_dir / "06b_dist_r_interaction_per_roi.png"),
            color='lightgray', sig_col='sig_r_interaction',
            q_per_roi=_q_dict('r_interaction'),
        )
        print("  saved 06b")
    except Exception as e:
        print(f"  fig06b failed: {e}")

    try:
        fig07_per_phase_contrast_by_category(
            results, str(diag_dir / "07_per_phase_by_category.png"))
        print("  saved 07")
    except Exception as e:
        print(f"  fig07 failed: {e}")
    try:
        fig08_roi_fractions_overview(
            roi_summary, str(diag_dir / "08_roi_fractions_overview.png"))
        print("  saved 08")
    except Exception as e:
        print(f"  fig08 failed: {e}")
    try:
        fig08b_roi_fractions_grouped(
            roi_summary, str(diag_dir / "08b_roi_fractions_grouped.png"))
        print("  saved 08b")
    except Exception as e:
        print(f"  fig08b failed: {e}")
    try:
        fig10_pref_state_stacked(
            results, roi_summary, str(diag_dir / "10_pref_state_stacked.png"))
        print("  saved 10")
    except Exception as e:
        print(f"  fig10 failed: {e}")
    try:
        fig11_wilcoxon_heatmap(
            results, roi_summary,
            str(diag_dir / "11_effect_size_heatmap.png"))
        print("  saved 11")
    except Exception as e:
        print(f"  fig11 failed: {e}")
    try:
        fig12_main_result(
            results, roi_summary,
            str(diag_dir / "12_main_result.png"))
        print("  saved 12_main_result")
    except Exception as e:
        print(f"  fig12 failed: {e}")
    try:
        _omni = (balance_roi_omnibus_perm(results)
                 if 'balance' in results.columns else None)
        fig13_balance(results, roi_summary,
                      str(diag_dir / "13_balance.png"), omnibus=_omni)
        print("  saved 13_balance")
    except Exception as e:
        print(f"  fig13 failed: {e}")
    try:
        _om = (balance_roi_omnibus_perm(results)
               if 'balance' in results.columns else None)
        fig15_single_cell_summary(
            results, roi_summary,
            str(diag_dir / "15_single_cell_summary.png"), omnibus=_om)
        print("  saved 15_single_cell_summary")
    except Exception as e:
        print(f"  fig15 failed: {e}")
    try:
        fig14_balance_vs_sustained(
            results, str(diag_dir / "14_balance_vs_sustained.png"))
        print("  saved 14_balance_vs_sustained")
    except Exception as e:
        print(f"  fig14 failed: {e}")

    # ROI-level inference: omnibus chi-squared + EC vs rest (Fisher exact)
    try:
        roi_inference = {}
        for col in ['sig_sustained', 'sig_r_state', 'sig_r_interaction']:
            roi_inference[col] = roi_omnibus_and_ec_tests(results, col)
        # Phasic-only is derived; build a temporary boolean
        results_tmp = results.copy()
        results_tmp['_phasic_only'] = (
            (results_tmp['sig_r_state'] | results_tmp['sig_r_interaction'])
            & (~results_tmp['sig_sustained'])
        )
        roi_inference['_phasic_only'] = roi_omnibus_and_ec_tests(
            results_tmp, '_phasic_only')
        # Balance: permutation omnibus over ROI labels, plus the same
        # significant-cell-fraction tests as the other statistics.
        if 'balance' in results.columns:
            roi_inference['balance_omnibus_perm'] = balance_roi_omnibus_perm(
                results, col='balance')
            roi_inference['balance_null_centred_omnibus_perm'] = (
                balance_roi_omnibus_perm(results, col='balance_null_centred'))
            if 'sig_balance' in results.columns:
                roi_inference['sig_balance'] = roi_omnibus_and_ec_tests(
                    results, 'sig_balance')
        with open(diag_dir / "roi_inference.json", 'w') as f:
            json.dump(roi_inference, f, indent=2)
        print("  saved roi_inference.json")
        print("    sustained:      omnibus chi² p = {:.3g},  EC vs rest p = {:.3g}"
              .format(roi_inference['sig_sustained']['chi2_omnibus_p'],
                      roi_inference['sig_sustained']['fisher_EC_vs_rest_p']))
        print("    r_state:        omnibus chi² p = {:.3g},  EC vs rest p = {:.3g}"
              .format(roi_inference['sig_r_state']['chi2_omnibus_p'],
                      roi_inference['sig_r_state']['fisher_EC_vs_rest_p']))
        print("    r_interaction:  omnibus chi² p = {:.3g},  EC vs rest p = {:.3g}"
              .format(roi_inference['sig_r_interaction']['chi2_omnibus_p'],
                      roi_inference['sig_r_interaction']['fisher_EC_vs_rest_p']))
        print("    phasic-only:    omnibus chi² p = {:.3g},  EC vs rest p = {:.3g}"
              .format(roi_inference['_phasic_only']['chi2_omnibus_p'],
                      roi_inference['_phasic_only']['fisher_EC_vs_rest_p']))
    except Exception as e:
        print(f"  roi_inference failed: {e}")

    # Write captions.md alongside the figures
    try:
        write_captions_md(diag_dir)
        print("  saved captions.md")
    except Exception as e:
        print(f"  captions.md failed: {e}")

    if build_gallery:
        gallery_dir = diag_dir / "example_gallery"
        print(f"  building example gallery → {gallery_dir}")
        try:
            build_example_gallery(results, gallery_dir, top_n=gallery_n)
        except Exception as e:
            print(f"  gallery failed: {e}")


def main():
    args = parse_args()

    # --- LOAD-OLD-RESULTS branch ---------------------------------------
    if args.load_old_results is not None:
        run_dir = _resolve_load_path(args.load_old_results)
        print(f"Loading old results from: {run_dir}")
        results = pd.read_csv(run_dir / "state_sustained_cv_results.csv")

        # Optional relabel: swap the `roi` column in-memory from a fresh
        # roi table, write into a NEW sibling directory (keeps original run
        # canonical). See mc/analyse/roi_relabel.py for the join contract.
        if args.relabel_from is not None:
            from mc.analyse.roi_relabel import relabel_per_cell
            results, _audit = relabel_per_cell(
                results, roi_table_csv=args.relabel_from,
                roi_col_in_table=ROI_LABEL_COLUMN)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir_out = run_dir.parent / f"{run_dir.name}_relabelled_{ts}"
            run_dir_out.mkdir(parents=True, exist_ok=True)
            results.to_csv(run_dir_out / "state_sustained_cv_results.csv",
                            index=False)
            with open(run_dir_out / "relabel_config.json", "w") as f:
                json.dump({
                    "reloaded_from": str(run_dir),
                    "relabel_from": str(args.relabel_from),
                    "roi_column_used": ROI_LABEL_COLUMN,
                    "timestamp": ts,
                }, f, indent=2)
            run_dir = run_dir_out
            print(f"Relabelled reload will write into: {run_dir}")

        # Re-derive the ROI summary so newly-added columns (e.g.
        # binom_p_phasic_only) appear even with old CSVs.
        roi_summary = make_roi_summary(results)
        roi_summary.to_csv(run_dir / "state_sustained_cv_roi_summary.csv",
                            index=False)
        diag_dir = run_dir / "diagnostic_figures"
        diag_dir.mkdir(exist_ok=True)
        print(f"Building figures → {diag_dir}")
        _build_all_figures(results, roi_summary, diag_dir,
                            n_permutations=args.n_permutations,
                            build_gallery=args.gallery,
                            gallery_n=args.gallery_n)
        print("done.")
        return

    # --- FULL ANALYSIS branch ------------------------------------------
    subjects = parse_subjects(args.subjects)
    if args.max_subjects is not None:
        subjects = subjects[:args.max_subjects]
    target_rois = ([r.strip() for r in args.rois.split(",") if r.strip()]
                   if args.rois is not None else TARGET_ROIS)
    target_set = set(target_rois) if target_rois is not None else None

    run_tag = args.run_tag or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(OUT_BASE, run_tag)
    diag_dir = os.path.join(out_dir, "diagnostic_figures")
    os.makedirs(diag_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    config = {
        "run_tag": run_tag, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_dir": DATA_DIR, "out_dir": out_dir, "subjects": subjects,
        "target_rois": target_rois, "roi_label_column": ROI_LABEL_COLUMN,
        "roi_table_path": ROI_TABLE_PATH,
        "input_traces": ("cell-*-360_bins_passed.csv (res_data=False): binned "
                          "firing rate in Hz, 90 time-warped bins per state, "
                          "averaged over correct repeats within configuration. "
                          "NOT residualised — neither for phase nor for repeat number."),
        "fit_method": "OLS (numpy.linalg.lstsq)",
        "glm_terms": ["state (4 one-hot)",
                      "state × sum-coded phase (8 cols; late = reference)",
                      "NO separate phase main effect — it lies in the span of the "
                      "interaction block, so a separate column would be rank-deficient",
                      "per-config intercepts, train configs only (one dropped)"],
        "cross_validation": "leave-one-configuration-out",
        "tests": {
            "b_sustained":  ("held-out min_phase_contrast vs its own permutation null "
                              "(circular shifts); NOT tested against zero — the minimum "
                              "of three noisy contrasts is biased below zero"),
            "c_any_state":  ("held-out Pearson r between y_test and X_state @ beta_state "
                              "vs permutation null (circular shifts); a correlation, "
                              "not a nested-model comparison"),
            "d_interaction": ("held-out Pearson r between y_test and X_inter @ beta_inter "
                              "vs the same permutation null; captures common phase "
                              "modulation AND state-specific phase deviations"),
        },
        "n_permutations": args.n_permutations, "n_jobs": args.n_jobs,
        "sig_alpha": SIG_ALPHA, "states": STATES, "phases": PHASES,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    all_rows = []; saved_design_fig = False
    t0 = time.time()
    # Load exactly the current ROI table once and pass it explicitly below.
    # This makes the full-analysis path use the same table as the reload path.
    roi_table = cell_selection.load_roi_table(
        data_dir=DATA_DIR, table_name=os.path.basename(ROI_TABLE_PATH),
        roi_column=ROI_LABEL_COLUMN)
    print(f"ROI table: {ROI_TABLE_PATH} "
          f"[{ROI_LABEL_COLUMN}; {len(roi_table)} cells, "
          f"{roi_table[ROI_LABEL_COLUMN].nunique()} labelled ROIs]")

    for sub_i, sub_str in enumerate(subjects, start=1):
        print(f"\n========== sub-{sub_str} ({sub_i}/{len(subjects)}) ==========")
        try:
            data = hh.load_norm_data(DATA_DIR, [sub_str], res_data=False)
        except Exception as exc:
            print(f"  load failed: {exc}"); continue
        key = f"sub-{sub_str}"
        if key not in data:
            print("  no data; skipping"); continue
        sub_dict = data[key]
        beh = add_config_str(sub_dict["beh"])
        configs = sorted(beh["config_str"].dropna().unique().tolist())
        neurons_raw = sub_dict["normalised_neurons"]

        meta = cell_selection.attach_roi_to_neuron_labels(
            neurons_raw.keys(), roi_table=roi_table,
            roi_column=ROI_LABEL_COLUMN
        ).set_index("neuron_id")
        keep_labels = []
        for n_lab in neurons_raw:
            if n_lab not in meta.index: continue
            roi = meta.loc[n_lab, "roi"]
            if roi is None or pd.isna(roi): continue
            if target_set is not None and roi not in target_set: continue
            keep_labels.append(n_lab)
        if args.max_neurons_per_subject is not None:
            keep_labels = keep_labels[:args.max_neurons_per_subject]
        if not keep_labels:
            print("  no usable neurons; skipping"); continue

        neurons_used = {n: neurons_raw[n].reset_index(drop=True) for n in keep_labels}
        Y_dict, config_labels, state_block, phase_block = build_y_per_config(
            beh, neurons_used, configs
        )
        n_cfgs = len(config_labels)
        if n_cfgs < 3:
            print(f"  only {n_cfgs} configs; skipping"); continue

        state_idx_full = np.tile(state_block, n_cfgs)
        phase_idx_full = np.tile(phase_block, n_cfgs)
        config_idx_full = np.concatenate(
            [np.full(N_BINS_PER_TRIAL, i, dtype=int) for i in range(n_cfgs)]
        )
        all_config_ids = list(range(n_cfgs))

        if not saved_design_fig:
            try:
                X_demo, slices_demo = build_design(
                    state_idx_full, phase_idx_full, config_idx_full,
                    all_config_ids[:-1]
                )
                fig01_design_matrix(X_demo, slices_demo,
                                     os.path.join(diag_dir, "01_design_matrix.png"))
                print("  saved 01_design_matrix.png"); saved_design_fig = True
            except Exception as e:
                print(f"  design fig failed: {e}")

        print(f"  fitting {len(keep_labels)} cells × {args.n_permutations} perms")
        n_jobs = 1 if len(keep_labels) == 1 else args.n_jobs
        rows = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(analyse_one_neuron)(
                n_lab, meta.loc[n_lab, "roi"], Y_dict[n_lab],
                state_idx_full, phase_idx_full, config_idx_full, all_config_ids,
                args.n_permutations,
                seed=cell_seed(sub_str, n_lab, "state_sustained_v4"),
            )
            for n_lab in keep_labels
        )
        for row in rows:
            n_lab = row["neuron"]
            row["subject"] = sub_str
            row["subject_int"] = int(sub_str)
            row["cell_idx"] = int(meta.loc[n_lab, "cell_idx"])
            row["MNI_x"] = float(meta.loc[n_lab, "MNI_x"])
            row["MNI_y"] = float(meta.loc[n_lab, "MNI_y"])
            row["MNI_z"] = float(meta.loc[n_lab, "MNI_z"])
        all_rows.extend(rows)
        print(f"  done; cumulative cells = {len(all_rows)}")

    if not all_rows:
        raise RuntimeError("No cells analysed.")

    results = pd.DataFrame(all_rows)
    results = add_fdr_columns(results)
    results = results.sort_values(["subject_int", "cell_idx"]).reset_index(drop=True)
    results_csv = os.path.join(out_dir, "state_sustained_cv_results.csv")
    results.to_csv(results_csv, index=False)

    roi_summary = make_roi_summary(results)
    roi_summary_csv = os.path.join(out_dir, "state_sustained_cv_roi_summary.csv")
    roi_summary.to_csv(roi_summary_csv, index=False)

    # ---------------- Publication figures ----------------
    print(f"\nBuilding publication figures → {diag_dir}")
    _build_all_figures(results, roi_summary, diag_dir,
                        n_permutations=args.n_permutations,
                        build_gallery=args.gallery,
                        gallery_n=args.gallery_n)

    print(f"\nSaved per-cell results → {results_csv}")
    print(f"Saved ROI summary       → {roi_summary_csv}")
    print("\n=== ROI summary ===")
    if not roi_summary.empty:
        cols = [
            "roi", "n_cells",
            "n_sustained", "frac_sustained", "binom_p_sustained",
            "n_sig_r_state", "frac_sig_r_state", "binom_p_sig_r_state",
            "n_phasic_only", "frac_phasic_only",
            "overrep_state", "overrep_state_p_binom",
        ]
        pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
        print(roi_summary[cols].to_string(index=False))
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
