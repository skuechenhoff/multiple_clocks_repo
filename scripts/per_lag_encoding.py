#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-lag location encoding for human single units — rate-map version.

For each cell, at each lag k in {0,30,...,330}°, we build a 9-d
"lag-shifted rate map" per configuration: mean firing rate at bins
where the location AT LAG k is L, for L in 1..9. Leave-one-config-out
CV gives a weighted Pearson r between the dwell-weighted mean of the
training rate maps and the held-out rate map. Permutation null is a
circular shift of the HELD-OUT LOCATION SERIES (not of y), which
preserves dwell structure under the null.

WHY THE RATE-MAP REPLACED THE BIN-LEVEL GLM
    Pearson r between a step-function predictor (9 distinct values
    painted on 360 bins) and a noisy continuous firing trace is bounded
    above by how much of bin-by-bin variance the location code captures
    — and most of that variance is bin-level noise / residual phase
    jitter / trial-timing jitter / state. Pre-averaging y by location
    BEFORE inference removes that noise floor. This matches the design
    of `spatial_peaks_simple` (which is FDR-significant for ACC at
    30°/60°) and lets the same statistic be computed at every lag and
    with optional controls.

RELATION TO `spatial_peaks_simple.py`
    SAME CORE QUESTION / PREPROCESSING
      * Both phase-residualise firing rates, build 9-location rate maps at
        the same twelve 30° lags, use leave-one-group-out CV, score maps
        with dwell-weighted Pearson r, and test predicted lags against zero
        and against the other lags.

    UNIQUE TO THIS PER-LAG ENCODING ANALYSIS (manuscript primary)
      * The CV unit is ONE task configuration. Repetitions are averaged
        within each configuration before rate maps are built.
      * All training configurations are pooled into one dwell-weighted
        predicted map, which is correlated with the held-out map.
      * Requires >= 3 shared locations; permutations circularly shift the
        held-out configuration's location series once per fold/permutation.
      * Provides optional nuisance-control residualisation and the stacked
        dsrfull/dsrinf models. The manuscript's main result uses `noctrl`.

    UNIQUE TO FUTURE SPATIAL PEAKS (independent robustness analysis)
      * Configurations are paired into coverage-maximising grid groups.
        A held-out group is correlated separately with every training group
        and those correlations are averaged (training maps are not pooled).
      * Requires >= 5 shared locations and independently circular-shifts
        each repetition for its permutation null.
      * Also provides the train-selected/free-peak control and rate-map
        example plots. It should not be described as merely the same test
        with paired configurations: the CV estimator, QC, and null differ.

CONTROL MODES
    no_ctrl    Rate maps computed from raw (phase-residualised) y.
    with_ctrl  Rate maps computed from y RESIDUALISED against
               state (4) + bttn_curr (5) + bttn_next (5) + per-config
               intercepts, with location-at-lag-0° added when k != 0°.
               Controls are fit ON TRAINING CONFIGS only; the training
               betas are applied to all configs, so the held-out
               residual y is leakage-free.
               L2-norm is intentionally NOT a control — fully collinear
               with the categorical location dummies in encoding space.

*** THE MANUSCRIPT REPORTS `no_ctrl`. WHAT THAT MEANS, EXACTLY ***
    "Uncontrolled" does NOT mean unprocessed. Two things are already
    removed before a single rate map is built:
      * within-state PHASE, residualised per cell on the raw 360-bin
        firing rate with a cosine basis (see PHASE RESIDUALISATION
        below). This is what stops the result being a relabelled
        subgoal-progress signal.
      * unreliable location estimates, via the dwell-based QC
        (MIN_DWELL_BINS, MIN_SHARED_LOCS) and the dwell-weighted
        correlation.
    What `no_ctrl` omits is the nuisance-REGRESSION step: firing rate is
    not additionally residualised against task state, the current button,
    or the current location. The consequence is specific and worth stating
    in the paper: at lag k the estimate is not protected against the
    trajectory's own autocorrelation. Because subjects take stereotyped
    paths, the location 30 deg ahead coincides with the current location on
    ~39% of bins (~3.5x the 1/9 chance rate), decaying to chance by
    +-90 deg. A significant result at a single lag therefore demonstrates
    spatial structure at that lag, NOT that the structure is specific to
    the future. Future-specificity is established only by the SHAPE of the
    lag profile — mPFC significant at 30-60 deg while at chance at 0 deg,
    the mirror image of the hippocampal profile — and by the within-cell
    paired test (T2), which asks whether the predicted lags beat that same
    cell's other ten lags. `with_ctrl` exists to probe the same question by
    regression instead; it is a robustness check, not the headline.

*** WHERE THE REPORTED NUMBERS COME FROM ***
    This script writes per-cell CV r into `per_cell_ALL_ROIs.csv` and
    CELL-level statistics into `per_roi_stats.csv`. The main-text figures
    and statistics are NOT read off `per_roi_stats.csv`: they come from
    `scripts/overlay_double_dissociation.py`, run on this script's
    per-cell CSV, which additionally aggregates SUBJECT-WISE (average the
    Fisher-z CV r across cells within a session, then test across
    sessions). The two aggregations do not agree about which lag peaks in
    mPFC — cell-level peaks at 30 deg, subject-level at 60 deg — so always
    state which unit of analysis a quoted t and df refer to:
        cell-level    -> df = n_cells - 1     (mPFC 154, HC_mid 231)
        subject-level -> df = n_sessions - 1  (mPFC 31,  HC_mid 34)

AGGREGATE MODELS
    dsrfull   Stacked rate-maps across all "beyond-now" lags
              (30°…330°, 11 × 9 = 99-d vector per config).
    dsrinf    Stacked rate-maps across lags 30°/60°/90° (3 × 9 = 27-d).
    Both report CV r against zero — no winner's-curse comparison
    against best-single-lag (the previous T6 was misleading).

CELL COHORT (`CELL_SET`, same vocabulary as spatial_peaks_simple)
    'all_in_roi_table' is the main analysis. 'not_in_rsa' restricts to the
    sessions that could NOT enter the population-RSA pseudopopulation
    (they solved different reward layouts), giving a cell cohort with zero
    overlap with the RSA sample — the manuscript's independent replication.
    'rsa' is its complement. The flag is honoured on the reload path too,
    so a cohort is re-tested from the cached CV r and the cached per-cell
    permutation p; nothing is recomputed.

QUALITY CONTROLS (borrowed from spatial_peaks)
    MIN_DWELL_BINS         per-location dwell threshold (default 25 bins)
    WEIGHTED_CORRELATION   weight Pearson r by min(train_dwell, held_dwell)
                           per location, so under-visited locations are
                           downweighted automatically (replaces explicit
                           grid-group pairing).

PHASE RESIDUALISATION applied per cell BEFORE any design construction
(`mc.analyse.future_spatial_peaks._residualise_phase`, cosine basis).

ROI-LEVEL STATS (per_roi_stats.csv):
    T1   one-sample t-test of mean CV r > 0          (per lag, per variant)
    T2   within-cell paired: r at predicted lag(s) vs mean r at other lags
    T3   binomial: fraction of cells with perm-p < alpha exceeds alpha
    T4   Wilcoxon signed-rank of per-cell perm-p vs 0.5 (population shift)
    T5   dsrfull and dsrinf r > 0
    T6   paired: dsrinf r vs r at first predicted lag (replaces winner-curse)

FISHER-Z SENSITIVITY ANALYSIS (no-control results only):
    Alongside the original raw-r statistics, a separate set of two-sided
    t-tests is run on Fisher-transformed, fold-averaged per-cell CV
    correlations. Raw r remains the descriptive effect-size scale.

Predicted lags per ROI (used by T2 and Fig 01 outlines):
    ACC          (30°, 60°)
    HC_anterior  (0°,)
    HC_mid       (0°, 330°)

OUTPUT under DATA_DIR/group/per_lag_encoding/<run_tag>/:
    per_cell_<ROI>.csv
    per_cell_ALL_ROIs.csv
    per_roi_stats.csv
    per_roi_stats_fisher_z_noctrl.csv
    per_roi_single_lag_stats_fisher_z_noctrl.csv
    roi_lag_table_fisher_z_noctrl.csv
    methods.md
    results.md
    core_results_noctrl_fisher_z.md + .csv
    roi_lag_table.csv
    config.json
    figures/*.pdf + .png   (01 t-stat heatmap, 02 curves, 03 sig-fraction bar,
                            03b sig-fraction heatmap, 04 dsrfull vs dsrinf,
                            05 ctrl vs noctrl, 06 r distribution at predicted lag)

Reusable plotting helpers live in `mc.plotting.cell_results`.

@author: Svenja Küchenhoff
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
import era_brewer
import mc.analyse.cell_selection as cs
import mc.analyse.helpers_human_cells as hh
from mc.analyse.future_spatial_peaks import _residualise_phase
from mc.plotting.cell_results import (
    plot_roi_lag_tstat_heatmap,
    plot_roi_lag_curves,
    p_to_stars,
)

# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_BASE = os.path.join(DATA_DIR, 'group', 'per_lag_encoding')

# Reload mode: point at a previous run directory to skip the heavy
# CV + permutation compute and just re-run stats + plots from the
# cached per-cell CSVs. Outputs land in a fresh timestamped dir.
# The 2026-06-30 base run (old ROI labels, mode estimator) was archived on
# 2026-11-23; with MAP_ESTIMATOR = 'per_repeat' the maps must be recomputed.
RELOAD_FROM     = None

# Optional. Path to a fresh neurons_with_final_roi_labels.csv. Only used
# when RELOAD_FROM is set; on reload the per-cell CSV's `roi` column is
# overwritten by the fresh table's `alt_final_roi` (joined on subject_id
# + cell_idx). See mc.analyse.roi_relabel.relabel_per_cell.
RELABEL_FROM    = ("/Users/xpsy1114/Documents/projects/multiple_clocks/"
                    "data/ephys_humans/derivatives/"
                    "neurons_with_ROI_labels.csv")
# RELABEL_FROM  = None

# Canonicalize labels from older result files before statistics/plotting.
ROI_NAME_MAP = {
    'ACC': 'mPFC',
    'mPFC': 'mPFC',
    'medialOFC': 'mOFC',
    'mOFC': 'mOFC',
    'medial OFC': 'mOFC',
    'Parahippocampal': 'PHC',
    'Parahippocampus': 'PHC',
    'PHC': 'PHC',
}

# Kept as a separate display map because it is written into config.json.
ROI_DISPLAY_NAMES = {}
def _disp(roi):
    return ROI_DISPLAY_NAMES.get(roi, roi)

def _canonicalize_roi_names(per_cell):
    per_cell = per_cell.copy()
    per_cell['roi'] = per_cell['roi'].map(
        lambda roi: ROI_NAME_MAP.get(roi, roi))
    return per_cell

# Canonical ROIs from cell_to_roi_july26.py (matches CLAUDE.md palette).
# Old-script names are canonicalized by `_canonicalize_roi_names`.
ROIS_TO_RUN = ['mPFC', 'mOFC', 'PCC', 'PHC',
               'HC_anterior', 'HC_mid', 'EC']

PHASE_RESIDUALISE      = 'cosine'
TRIALS                 = 'all_minus_explore'

# How a configuration's rate map is built (SK, 2026-11-23):
#   'per_repeat' -- every repeat with its OWN visited locations; a bin counts
#                   for the location visited `lag` bins later in that repeat.
#                   Dwell = bins per repeat, so MIN_DWELL_BINS keeps its scale.
#   'mode'       -- average the repeats, then one location per bin (the most
#                   frequent one). The 2026-06-30 run (published until then).
# 'per_repeat' supports the noctrl variants only (the ctrl columns are NaN).
MAP_ESTIMATOR          = 'per_repeat'

# Which cells enter the analysis. Same vocabulary as
# `spatial_peaks_simple.CELL_SET`, resolved by `cs.load_cells`:
#   'all_in_roi_table' — every cell with an ROI label (the main analysis)
#   'rsa'              — only sessions that share the 8 RSA layouts
#   'not_in_rsa'       — only sessions that do NOT, i.e. the cohort with
#                        zero overlap with the population-RSA sample.
# Honoured on BOTH paths: on a full run it selects which cells are
# computed; under RELOAD_FROM it subsets the cached per-cell table, so a
# cohort can be re-tested without recomputing CV or permutations.
CELL_SET               = 'all_in_roi_table'
N_PERMUTATIONS         = 1000
N_JOBS                 = -1
RANDOM_SEED            = 42
ALPHA                  = 0.05
FISHER_Z_CLIP_EPS      = 1e-7

# Rate-map QC (mirrors spatial_peaks_simple settings)
MIN_DWELL_BINS         = 25
WEIGHTED_CORRELATION   = True
MIN_SHARED_LOCS        = 3        # rate-map r needs >= this many shared loc bins

# Lag grid (degrees); 1 deg = 1 bin in the standardised 360-bin trial.
N_BINS  = 360
N_LOC   = 9
N_STATE = 4
LAGS_DEG = list(range(0, 360, 30))     # 0,30,...,330

DSR_FULL_LAGS_DEG = [l for l in LAGS_DEG if l != 0]    # 11 "beyond-now" lags
DSR_INF_LAGS_DEG  = [30, 60, 90]                        # ACC-informed subset

ROI_PREDICTED_LAGS_DEG = {
    'mPFC':        (30, 60),
    'HC_anterior': (0, 330),
    'HC_mid':      (0, 330),
}

# Single lags at which we want a lag-agnostic per-ROI t-test on the
# per-cell CV r. Each lag is tested independently (no "vs other lags"
# structure), across EVERY ROI — makes it easy to see "which ROI has a
# signal AT lag X" without having to reason about lag sets.
SINGLE_LAGS_FOR_TESTS = [0, 30, 60, 330]

# bttn_next = button this many bins ahead. User-tuned (was 90 = one full
# state, now 30 = one position bin).
LAG_BINS_BTTN_NEXT = 30

# Which regressors enter the partial-out design in `with_ctrl` mode.
# Per-config intercepts are ALWAYS included (statistical glue, not a
# model). `loc_now` is additionally gated by lag (it is only added at
# lags != 0° to avoid being collinear with the target regressor).
# Set to an empty set to make `with_ctrl` equivalent to `no_ctrl`.
CONTROL_MODELS = {'state', 'bttn_curr'}
_VALID_CONTROL_MODELS = {'state', 'bttn_curr', 'bttn_next', 'loc_now'}
assert CONTROL_MODELS <= _VALID_CONTROL_MODELS, (
    f"CONTROL_MODELS contains unknown entries: "
    f"{CONTROL_MODELS - _VALID_CONTROL_MODELS}. "
    f"Allowed: {_VALID_CONTROL_MODELS}"
)

BTTN_CODES = [0, 1, 2, 3, 99]
N_BTTN     = len(BTTN_CODES)
BTTN_LABEL_TO_CODE = {'LeftArrow': 0, 'UpArrow': 1, 'RightArrow': 2,
                       'DownArrow': 3, 'Return': 99}

# Plotting --------------------------------------------------------------
DPI = 300
CM = 1.0 / 2.54
FONT_BIG, FONT_AXIS, FONT_TICK = 11, 10, 9
_ROI_PAL = era_brewer.era_brew('Showgirl2', n=7)
ROI_COLOURS = {                 # matches CLAUDE.md `roi_colour_dict`
    'EC':          _ROI_PAL[0],
    'mOFC':        _ROI_PAL[4],
    'mPFC':        _ROI_PAL[1],
    'HC_anterior': '#a30d6c',
    'HC_mid':      _ROI_PAL[2],
    'PHC':         '#23677E',
    'PCC':         _ROI_PAL[3],
}
OBSERVED_GREEN = '#0e3d3a'


# ── Generic helpers ───────────────────────────────────────────────────
def _set_rc():
    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size':       FONT_TICK,
        'pdf.fonttype':    42,
        'ps.fonttype':     42,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def _save(fig, save_stem):
    fig.savefig(save_stem + '.pdf', dpi=DPI, bbox_inches='tight')
    fig.savefig(save_stem + '.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def _bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    good = np.isfinite(p)
    if not good.any():
        return out
    pg = p[good]
    order = np.argsort(pg)
    ranked = pg[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out_good = np.empty_like(q); out_good[order] = q
    out[good] = out_good
    return out


def _weighted_pearson_r(x, y, w):
    """Weighted Pearson correlation. NaN-safe; returns NaN if degenerate."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if m.sum() < MIN_SHARED_LOCS:
        return np.nan
    x, y, w = x[m], y[m], w[m]
    wsum = w.sum()
    xm = (x * w).sum() / wsum
    ym = (y * w).sum() / wsum
    cov = (w * (x - xm) * (y - ym)).sum()
    vx  = (w * (x - xm) ** 2).sum()
    vy  = (w * (y - ym) ** 2).sum()
    if vx < 1e-12 or vy < 1e-12:
        return np.nan
    return float(cov / np.sqrt(vx * vy))


def _pearson_r(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < MIN_SHARED_LOCS:
        return np.nan
    if np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


# ── Per-config mode series ────────────────────────────────────────────
def _mode_per_bin_int(arr2d):
    out = np.zeros(arr2d.shape[1], dtype=int)
    for t in range(arr2d.shape[1]):
        col = arr2d[:, t]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        col = col.astype(int)
        vals, cnts = np.unique(col, return_counts=True)
        out[t] = int(vals[np.argmax(cnts)])
    return out


def _build_per_cfg_sequences(arr_clean, idx_cfg, locs, btns_int, n_cfg):
    if MAP_ESTIMATOR == 'per_repeat':
        # one (n_repeats, 360) block per configuration, nothing averaged
        Y_cfg   = [arr_clean[idx_cfg == c] for c in range(n_cfg)]
        loc_cfg = [locs[idx_cfg == c] for c in range(n_cfg)]
        btn_cfg = [btns_int[idx_cfg == c] for c in range(n_cfg)]
        return Y_cfg, loc_cfg, btn_cfg
    Y_cfg   = np.zeros((n_cfg, N_BINS))
    loc_cfg = np.zeros((n_cfg, N_BINS), dtype=int)
    btn_cfg = np.zeros((n_cfg, N_BINS), dtype=int)
    for c in range(n_cfg):
        mask = idx_cfg == c
        if not mask.any():
            continue
        Y_cfg[c]   = np.nanmean(arr_clean[mask], axis=0)
        loc_cfg[c] = _mode_per_bin_int(locs[mask])
        btn_cfg[c] = _mode_per_bin_int(btns_int[mask])
    return Y_cfg, loc_cfg, btn_cfg


# ── Rate-map helpers ──────────────────────────────────────────────────
def _lag_shifted_rate_map(y_360, loc_360, lag_bins):
    """Returns (9-d rate, 9-d dwell). rate[L-1] = mean firing at bins
    where the location AT LAG +lag_bins is L.

    y_360, loc_360: one sequence (360,) or one row per repeat (n_rep, 360);
    each repeat is shifted within itself. Dwell = such bins per repeat."""
    loc_at_lag = np.roll(loc_360, -lag_bins, axis=-1)
    n_rep = 1 if np.ndim(y_360) == 1 else len(y_360)
    rate = np.full(N_LOC, np.nan)
    dwell = np.zeros(N_LOC)
    for L in range(1, N_LOC + 1):
        m = (loc_at_lag == L) & np.isfinite(y_360)
        d = int(m.sum())
        dwell[L - 1] = d / n_rep
        if d > 0:
            rate[L - 1] = float(np.nanmean(y_360[m]))
    return rate, dwell


def _stacked_rate_map(y_360, loc_360, lags_deg):
    """Concatenate lag-shifted rate maps across `lags_deg`.
    Returns (9*len(lags),) rate and dwell."""
    rates, dwells = [], []
    for lag in lags_deg:
        r, d = _lag_shifted_rate_map(y_360, loc_360, lag)
        rates.append(r); dwells.append(d)
    return np.concatenate(rates), np.concatenate(dwells)


def _ratemap_cv_one_setup(rate_per_cfg, dwell_per_cfg,
                          loc_cfg, y_cfg_used, lags_deg,
                          n_perms, rng):
    """LOO CV over configs given pre-built per-config rate maps + dwell.
    `loc_cfg` and `y_cfg_used` are passed so we can recompute the held-out
    rate map after a circular shift of the held-out location series for
    the permutation null. Returns (mean_r, perm_p)."""
    n_cfg = rate_per_cfg.shape[0]
    if n_cfg < 2:
        return np.nan, np.nan
    fold_rs = []
    perm_r_per_fold = (np.full((n_perms, n_cfg), np.nan)
                       if n_perms > 0 else None)
    for held in range(n_cfg):
        idx_tr = [c for c in range(n_cfg) if c != held]
        train_r = rate_per_cfg[idx_tr]            # (n_tr, D)
        train_d = dwell_per_cfg[idx_tr]
        total_d = train_d.sum(axis=0)             # (D,)
        with np.errstate(invalid='ignore'):
            predicted = np.nansum(train_r * train_d, axis=0) / np.where(
                total_d > 0, total_d, np.nan)
        observed   = rate_per_cfg[held]
        held_d     = dwell_per_cfg[held]
        weights = (np.minimum(total_d, held_d)
                    if WEIGHTED_CORRELATION
                    else (np.minimum(total_d, held_d) >= 1).astype(float))
        keep = ((np.minimum(total_d, held_d) >= MIN_DWELL_BINS)
                & np.isfinite(predicted) & np.isfinite(observed))
        if keep.sum() < MIN_SHARED_LOCS:
            continue
        if WEIGHTED_CORRELATION:
            r = _weighted_pearson_r(predicted[keep], observed[keep], weights[keep])
        else:
            r = _pearson_r(predicted[keep], observed[keep])
        if np.isfinite(r):
            fold_rs.append(r)
        # Permutation: shift the held-out location series, rebuild its
        # rate map, weight by new dwell × training-total dwell, re-correlate.
        if n_perms > 0:
            for p in range(n_perms):
                # Avoid trivial near-zero shifts (would re-create the obs)
                shift = int(rng.integers(N_BINS // 12, N_BINS - N_BINS // 12))
                loc_sh = np.roll(loc_cfg[held], shift, axis=-1)
                obs_p, dwell_p = _stacked_rate_map(y_cfg_used[held], loc_sh,
                                                    lags_deg)
                w_p = (np.minimum(total_d, dwell_p)
                        if WEIGHTED_CORRELATION
                        else (np.minimum(total_d, dwell_p) >= 1).astype(float))
                kp = ((np.minimum(total_d, dwell_p) >= MIN_DWELL_BINS)
                       & np.isfinite(predicted) & np.isfinite(obs_p))
                if kp.sum() < MIN_SHARED_LOCS:
                    continue
                if WEIGHTED_CORRELATION:
                    rp = _weighted_pearson_r(predicted[kp], obs_p[kp], w_p[kp])
                else:
                    rp = _pearson_r(predicted[kp], obs_p[kp])
                perm_r_per_fold[p, held] = rp

    if not fold_rs:
        return np.nan, np.nan
    mean_r = float(np.mean(fold_rs))
    if n_perms == 0:
        return mean_r, np.nan
    null_means = np.nanmean(perm_r_per_fold, axis=1)
    null_means = null_means[np.isfinite(null_means)]
    if null_means.size == 0:
        return mean_r, np.nan
    pval = (np.sum(null_means >= mean_r) + 1) / (null_means.size + 1)
    return mean_r, float(pval)


# ── Control partial-out ───────────────────────────────────────────────
def _loc_onehot_at_lag_bin(loc_cfg, lag_bins):
    """For partial-out only: bin-level (n_cfg*360, 9) one-hot at lag."""
    n_cfg = loc_cfg.shape[0]
    out = np.zeros((n_cfg * N_BINS, N_LOC), dtype=float)
    for c in range(n_cfg):
        rolled = np.roll(loc_cfg[c], -lag_bins)
        for t in range(N_BINS):
            l = rolled[t]
            if 1 <= l <= N_LOC:
                out[c * N_BINS + t, l - 1] = 1.0
    return out


def _btn_onehot_at_lag_bin(btn_cfg, lag_bins):
    n_cfg = btn_cfg.shape[0]
    code_to_col = {c: i for i, c in enumerate(BTTN_CODES)}
    out = np.zeros((n_cfg * N_BINS, N_BTTN), dtype=float)
    for c in range(n_cfg):
        rolled = np.roll(btn_cfg[c], -lag_bins)
        for t in range(N_BINS):
            code = int(rolled[t])
            j = code_to_col.get(code, None)
            if j is not None:
                out[c * N_BINS + t, j] = 1.0
    return out


def _state_onehot_bin(n_cfg):
    state_per_bin = (np.arange(N_BINS) // (N_BINS // N_STATE)).astype(int)
    one_per_cfg = np.zeros((N_BINS, N_STATE))
    for s in range(N_STATE):
        one_per_cfg[state_per_bin == s, s] = 1.0
    return np.tile(one_per_cfg, (n_cfg, 1))


def _cfg_intercepts_bin(n_cfg, train_cfg_ids):
    cols = sorted(train_cfg_ids)[1:]
    X = np.zeros((n_cfg * N_BINS, len(cols)))
    for j, c in enumerate(cols):
        X[c * N_BINS:(c + 1) * N_BINS, j] = 1.0
    return X


def _partial_out_controls(Y_cfg, loc_cfg, btn_cfg, train_cfg_ids,
                            include_loc_now=True):
    """Fit y ~ <CONTROL_MODELS> + per-config intercepts on training
    configs, return residual Y_cfg (residuals for ALL configs). Held-out
    is leakage-free because betas are trained without it.

    `include_loc_now` gates the loc_now block by lag (it is suppressed
    at lag 0° to avoid being collinear with the target regressor). The
    loc_now block is only added if both this flag is True AND 'loc_now'
    is in `CONTROL_MODELS`.
    """
    n_cfg = Y_cfg.shape[0]
    blocks = []
    if 'state' in CONTROL_MODELS:
        blocks.append(_state_onehot_bin(n_cfg))
    if 'loc_now' in CONTROL_MODELS and include_loc_now:
        blocks.append(_loc_onehot_at_lag_bin(loc_cfg, 0))
    if 'bttn_curr' in CONTROL_MODELS:
        blocks.append(_btn_onehot_at_lag_bin(btn_cfg, 0))
    if 'bttn_next' in CONTROL_MODELS:
        blocks.append(_btn_onehot_at_lag_bin(btn_cfg, LAG_BINS_BTTN_NEXT))
    blocks.append(_cfg_intercepts_bin(n_cfg, train_cfg_ids))
    if len(blocks) == 1:
        # only the per-config intercepts → demean per cfg, no partialling
        X = blocks[0]
    else:
        X = np.hstack(blocks)
    y_full = Y_cfg.reshape(-1)
    keep = np.isfinite(y_full)
    train_mask = np.ones(X.shape[0], dtype=bool)
    held_out = set(range(n_cfg)) - set(train_cfg_ids)
    for c in held_out:
        train_mask[c * N_BINS:(c + 1) * N_BINS] = False
    train_keep = train_mask & keep
    if train_keep.sum() < X.shape[1] + 1:
        return None
    try:
        beta, *_ = np.linalg.lstsq(X[train_keep], y_full[train_keep], rcond=None)
    except np.linalg.LinAlgError:
        return None
    y_pred = X @ beta
    y_resid = y_full - y_pred
    return y_resid.reshape(n_cfg, N_BINS)


# ── Fit functions per (variant × ctrl-mode) ───────────────────────────
def _fit_variant(Y_cfg, loc_cfg, btn_cfg, lags_deg,
                  with_controls, include_loc_now_in_ctrl,
                  n_perms, rng):
    """Generic rate-map LOO CV. `lags_deg` may be a single-lag list
    (per-lag fit, 9-d) or multi-lag (stacked, 9*len(lags)-d).

    With controls: partial out state + buttons (+ location-now when
    include_loc_now_in_ctrl) PER FOLD — the residualisation is refit on
    each training set so the held-out config's residual is leakage-free.
    """
    n_cfg = len(Y_cfg)
    if n_cfg < 2:
        return np.nan, np.nan
    if with_controls and MAP_ESTIMATOR == 'per_repeat':
        return np.nan, np.nan          # ctrl is defined on averaged sequences only
    # We need to refit controls per fold for leakage-free residuals.
    # For efficiency, when with_controls=False the rate maps don't change
    # across folds, so build them ONCE and call the CV inline.
    if not with_controls:
        rate_per_cfg  = np.stack([
            _stacked_rate_map(Y_cfg[c], loc_cfg[c], lags_deg)[0]
            for c in range(n_cfg)
        ])
        dwell_per_cfg = np.stack([
            _stacked_rate_map(Y_cfg[c], loc_cfg[c], lags_deg)[1]
            for c in range(n_cfg)
        ])
        return _ratemap_cv_one_setup(
            rate_per_cfg, dwell_per_cfg, loc_cfg, Y_cfg, lags_deg,
            n_perms, rng,
        )

    # Controls mode: rate maps depend on training set → refit per fold.
    fold_rs = []
    perm_r_per_fold = (np.full((n_perms, n_cfg), np.nan)
                       if n_perms > 0 else None)
    for held in range(n_cfg):
        train_cfg = [c for c in range(n_cfg) if c != held]
        Y_resid = _partial_out_controls(
            Y_cfg, loc_cfg, btn_cfg, train_cfg,
            include_loc_now=include_loc_now_in_ctrl,
        )
        if Y_resid is None:
            continue
        # Build per-config rate maps on residual y
        rate_per_cfg = np.stack([
            _stacked_rate_map(Y_resid[c], loc_cfg[c], lags_deg)[0]
            for c in range(n_cfg)
        ])
        dwell_per_cfg = np.stack([
            _stacked_rate_map(Y_resid[c], loc_cfg[c], lags_deg)[1]
            for c in range(n_cfg)
        ])
        train_r = rate_per_cfg[train_cfg]
        train_d = dwell_per_cfg[train_cfg]
        total_d = train_d.sum(axis=0)
        with np.errstate(invalid='ignore'):
            predicted = np.nansum(train_r * train_d, axis=0) / np.where(
                total_d > 0, total_d, np.nan)
        observed = rate_per_cfg[held]
        held_d   = dwell_per_cfg[held]
        weights  = (np.minimum(total_d, held_d) if WEIGHTED_CORRELATION
                     else (np.minimum(total_d, held_d) >= 1).astype(float))
        keep = ((np.minimum(total_d, held_d) >= MIN_DWELL_BINS)
                & np.isfinite(predicted) & np.isfinite(observed))
        if keep.sum() < MIN_SHARED_LOCS:
            continue
        if WEIGHTED_CORRELATION:
            r = _weighted_pearson_r(predicted[keep], observed[keep], weights[keep])
        else:
            r = _pearson_r(predicted[keep], observed[keep])
        if np.isfinite(r):
            fold_rs.append(r)
        if n_perms > 0:
            for p in range(n_perms):
                shift = int(rng.integers(N_BINS // 12, N_BINS - N_BINS // 12))
                loc_sh = np.roll(loc_cfg[held], shift, axis=-1)
                obs_p, dwell_p = _stacked_rate_map(Y_resid[held], loc_sh,
                                                    lags_deg)
                w_p = (np.minimum(total_d, dwell_p) if WEIGHTED_CORRELATION
                        else (np.minimum(total_d, dwell_p) >= 1).astype(float))
                kp = ((np.minimum(total_d, dwell_p) >= MIN_DWELL_BINS)
                       & np.isfinite(predicted) & np.isfinite(obs_p))
                if kp.sum() < MIN_SHARED_LOCS:
                    continue
                if WEIGHTED_CORRELATION:
                    rp = _weighted_pearson_r(predicted[kp], obs_p[kp], w_p[kp])
                else:
                    rp = _pearson_r(predicted[kp], obs_p[kp])
                perm_r_per_fold[p, held] = rp

    if not fold_rs:
        return np.nan, np.nan
    mean_r = float(np.mean(fold_rs))
    if n_perms == 0:
        return mean_r, np.nan
    null_means = np.nanmean(perm_r_per_fold, axis=1)
    null_means = null_means[np.isfinite(null_means)]
    if null_means.size == 0:
        return mean_r, np.nan
    pval = (np.sum(null_means >= mean_r) + 1) / (null_means.size + 1)
    return mean_r, float(pval)


# ── Per-cell wrapper ──────────────────────────────────────────────────
def analyse_cell(neuron_id, arr_clean, idx_cfg, locs, btns_int, n_cfg,
                  roi, sub_str, mni, seed):
    rng = np.random.default_rng(seed)
    Y_cfg, loc_cfg, btn_cfg = _build_per_cfg_sequences(
        arr_clean, idx_cfg, locs, btns_int, n_cfg,
    )
    row = {
        'neuron': neuron_id, 'roi': roi, 'subject_id': sub_str,
        'MNI_x': float(mni[0]) if mni and np.isfinite(mni[0]) else np.nan,
        'MNI_y': float(mni[1]) if mni and np.isfinite(mni[1]) else np.nan,
        'MNI_z': float(mni[2]) if mni and np.isfinite(mni[2]) else np.nan,
        'n_cfg': int(n_cfg),
    }
    # PER-LAG sweep ---------------------------------------------------
    for lag in LAGS_DEG:
        for with_ctrl in (False, True):
            tag = f'lag{lag:03d}_{"ctrl" if with_ctrl else "noctrl"}'
            r, p = _fit_variant(
                Y_cfg, loc_cfg, btn_cfg, [lag],
                with_controls=with_ctrl,
                include_loc_now_in_ctrl=(with_ctrl and lag != 0),
                n_perms=N_PERMUTATIONS, rng=rng,
            )
            row[f'r_{tag}'] = r
            row[f'p_{tag}'] = p
    # AGGREGATE: dsrfull and dsrinf, both ctrl modes ----------------
    for variant, lags in (('dsrfull', DSR_FULL_LAGS_DEG),
                          ('dsrinf',  DSR_INF_LAGS_DEG)):
        for with_ctrl in (False, True):
            tag = f'{variant}_{"ctrl" if with_ctrl else "noctrl"}'
            r, p = _fit_variant(
                Y_cfg, loc_cfg, btn_cfg, lags,
                with_controls=with_ctrl,
                include_loc_now_in_ctrl=with_ctrl,   # 0 not in lags → safe
                n_perms=N_PERMUTATIONS, rng=rng,
            )
            row[f'r_{tag}'] = r
            row[f'p_{tag}'] = p
    return row


# ── ROI loop ──────────────────────────────────────────────────────────
def _filter_buttons(buttons_raw, keep_mask):
    btn_df = buttons_raw.loc[keep_mask.to_numpy()].reset_index(drop=True)
    vals = btn_df.values
    out = np.full(vals.shape, 99, dtype=int)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if isinstance(v, str):
                code = BTTN_LABEL_TO_CODE.get(v, None)
                if code is not None:
                    out[i, j] = code
            elif isinstance(v, (int, np.integer)):
                out[i, j] = int(v)
    return out


def run_roi(roi, cells_df, n_perms=N_PERMUTATIONS,
             n_jobs=N_JOBS, seed=RANDOM_SEED):
    print(f'\n══ ROI: {roi} ══')
    sub_list = sorted(cells_df.loc[cells_df['roi'] == roi, 'subject_id'].unique())
    print(f'  {(cells_df["roi"] == roi).sum()} cells across {len(sub_list)} subjects')

    tasks = []
    for sub_str in sub_list:
        try:
            data_raw = hh.load_norm_data(DATA_DIR, [sub_str], res_data=False)
        except Exception as exc:
            print(f'  sub-{sub_str} load failed: {exc}'); continue
        if not data_raw:
            continue
        beh_raw = data_raw[f'sub-{sub_str}']['beh']
        buttons_raw = data_raw[f'sub-{sub_str}']['buttons']
        keep_mask = beh_raw[['correct', 'rep_correct']].ne(0).any(axis=1)
        btns_int_all = _filter_buttons(buttons_raw, keep_mask)
        data = hh.filter_data(data_raw, int(sub_str), TRIALS)
        sub_dict = data[f'sub-{sub_str}']
        beh = sub_dict['beh'].copy().reset_index(drop=True)
        locs = sub_dict['locations'].to_numpy(dtype=float)
        if btns_int_all.shape[0] != locs.shape[0]:
            print(f'  sub-{sub_str} button/loc row mismatch '
                  f'({btns_int_all.shape[0]} vs {locs.shape[0]}); skipping')
            continue
        uniq, _, idx_cfg, _ = np.unique(
            beh[['loc_A', 'loc_B', 'loc_C', 'loc_D']].to_numpy(),
            axis=0, return_index=True, return_inverse=True, return_counts=True,
        )
        n_cfg = len(np.unique(idx_cfg))
        if n_cfg < 5:
            print(f'  sub-{sub_str} skipped ({n_cfg} configs)'); continue
        sub_keep = set(cells_df.loc[
            (cells_df['subject_id'] == sub_str) & (cells_df['roi'] == roi),
            'cell_idx'].tolist())
        for nid, n_df in sub_dict['normalised_neurons'].items():
            _, ci = cs.parse_neuron_label(nid)
            if ci not in sub_keep:
                continue
            arr = n_df.to_numpy(dtype=float)
            if PHASE_RESIDUALISE:
                arr = _residualise_phase(arr, basis=PHASE_RESIDUALISE)
            roi_row = cells_df[
                (cells_df['subject_id'] == sub_str)
                & (cells_df['cell_idx'] == ci)
            ].iloc[0]
            mni = (
                float(roi_row.get('MNI_x', np.nan)),
                float(roi_row.get('MNI_y', np.nan)),
                float(roi_row.get('MNI_z', np.nan)),
            )
            tasks.append(dict(
                neuron_id=nid, arr_clean=arr.copy(),
                idx_cfg=idx_cfg.copy(), locs=locs.copy(),
                btns_int=btns_int_all.copy(), n_cfg=int(n_cfg),
                roi=roi, sub_str=sub_str, mni=mni,
                seed=seed + (hash(nid) & 0x7FFFFFFF),
            ))

    if not tasks:
        return None
    print(f'  Fitting {len(tasks)} cells ({n_perms} perms each)…')
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(analyse_cell)(**t) for t in tasks
    )
    return pd.DataFrame(rows)


# ── Stats ─────────────────────────────────────────────────────────────
def _ttest_gt0(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan, np.nan
    try:
        res = stats.ttest_1samp(x, 0.0, alternative='greater')
        return float(res.statistic), float(res.pvalue)
    except TypeError:
        t, p2 = stats.ttest_1samp(x, 0.0)
        return float(t), float(p2 / 2 if t > 0 else 1 - p2 / 2)


def _fisher_z(r):
    """Fisher-transform finite correlations, guarding against |r| == 1."""
    r = np.asarray(r, dtype=float)
    z = np.full(r.shape, np.nan, dtype=float)
    m = np.isfinite(r)
    if m.any():
        z[m] = np.arctanh(np.clip(r[m], -1 + FISHER_Z_CLIP_EPS,
                                    1 - FISHER_Z_CLIP_EPS))
    return z


def _ttest_2sided_zero(x):
    """Two-sided one-sample t-test, returning statistic, p, and n."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan, np.nan, int(x.size)
    res = stats.ttest_1samp(x, 0.0)
    return float(res.statistic), float(res.pvalue), int(x.size)


def _ttest_2sided_paired(x, y):
    """Two-sided paired t-test, returning statistic, p, and complete-pair n."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan, int(m.sum())
    res = stats.ttest_rel(x[m], y[m])
    return float(res.statistic), float(res.pvalue), int(m.sum())


def _binom_gt_alpha(k, n, alpha=ALPHA):
    if n == 0:
        return np.nan
    try:
        return float(stats.binomtest(k, n, p=alpha, alternative='greater').pvalue)
    except AttributeError:
        return float(stats.binom_test(k, n, p=alpha, alternative='greater'))


def _wilcoxon_perm_p(p_vals):
    p = np.asarray(p_vals, dtype=float); p = p[np.isfinite(p)]
    if p.size < 5:
        return np.nan
    try:
        return float(stats.wilcoxon(p - 0.5, alternative='less').pvalue)
    except ValueError:
        return np.nan


def per_roi_stats(df, ctrl_mode):
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    rows = []
    for roi, g in df.groupby('roi'):
        n = len(g)
        rec = {'roi': roi, 'ctrl_mode': tag, 'n_cells': n}
        # T1 + T3 + T4 per lag ----------------------------------------
        for lag in LAGS_DEG:
            r = g[f'r_lag{lag:03d}_{tag}'].to_numpy(dtype=float)
            p = g[f'p_lag{lag:03d}_{tag}'].to_numpy(dtype=float)
            t1_t, t1_p = _ttest_gt0(r)
            k_sig = int(np.sum(np.isfinite(p) & (p < ALPHA)))
            rec[f'T1_t_lag{lag:03d}']     = t1_t
            rec[f'T1_p_lag{lag:03d}']     = t1_p
            rec[f'T1_meanR_lag{lag:03d}'] = float(np.nanmean(r))
            rec[f'T3_k_lag{lag:03d}']     = k_sig
            rec[f'T3_p_lag{lag:03d}']     = _binom_gt_alpha(k_sig, n)
            rec[f'T4_p_lag{lag:03d}']     = _wilcoxon_perm_p(p)
        # T1a — averaged-across-predicted-lags T1 (parallels the
        # `test1_meanR_*` test in spatial_peaks_simple: for each cell take
        # the mean CV r over the ROI's predicted lag set, then one-sample
        # t-test of that per-cell mean > 0 across cells). Only defined for
        # predicted-lag ROIs.
        # T2 — within-cell paired: predicted vs other lags.
        pred_lags = ROI_PREDICTED_LAGS_DEG.get(roi, None)
        if pred_lags:
            r_mat = np.stack([
                g[f'r_lag{lag:03d}_{tag}'].to_numpy(dtype=float)
                for lag in LAGS_DEG
            ], axis=1)
            idx_pred  = [LAGS_DEG.index(l) for l in pred_lags]
            idx_other = [i for i in range(len(LAGS_DEG)) if i not in idx_pred]
            tgt   = np.nanmean(r_mat[:, idx_pred],  axis=1)
            m_tgt = np.isfinite(tgt)
            if m_tgt.sum() >= 2:
                t1a_t, t1a_p = _ttest_gt0(tgt[m_tgt])
                rec.update({'T1a_avgPred_t':     t1a_t,
                            'T1a_avgPred_p':     t1a_p,
                            'T1a_avgPred_meanR': float(tgt[m_tgt].mean()),
                            'T1a_avgPred_n':     int(m_tgt.sum())})
            else:
                rec.update({'T1a_avgPred_t': np.nan, 'T1a_avgPred_p': np.nan,
                            'T1a_avgPred_meanR': np.nan, 'T1a_avgPred_n': 0})
            other = np.nanmean(r_mat[:, idx_other], axis=1)
            diff = tgt - other
            m = np.isfinite(diff)
            if m.sum() >= 2:
                res = stats.ttest_1samp(diff[m], 0.0, alternative='greater')
                rec.update({'T2_t': float(res.statistic),
                            'T2_p': float(res.pvalue),
                            'T2_meanDiff': float(diff[m].mean()),
                            'T2_n': int(m.sum())})
            else:
                rec.update({'T2_t': np.nan, 'T2_p': np.nan,
                            'T2_meanDiff': np.nan, 'T2_n': 0})
        else:
            rec.update({'T1a_avgPred_t': np.nan, 'T1a_avgPred_p': np.nan,
                        'T1a_avgPred_meanR': np.nan, 'T1a_avgPred_n': 0,
                        'T2_t': np.nan, 'T2_p': np.nan,
                        'T2_meanDiff': np.nan, 'T2_n': 0})
        # T5: dsrfull and dsrinf primary tests ------------------------
        for variant in ('dsrfull', 'dsrinf'):
            r = g[f'r_{variant}_{tag}'].to_numpy(dtype=float)
            p = g[f'p_{variant}_{tag}'].to_numpy(dtype=float)
            t, p_val = _ttest_gt0(r)
            k_sig = int(np.sum(np.isfinite(p) & (p < ALPHA)))
            rec[f'T5_{variant}_t']     = t
            rec[f'T5_{variant}_p']     = p_val
            rec[f'T5_{variant}_meanR'] = float(np.nanmean(r))
            rec[f'T3_{variant}_k']     = k_sig
            rec[f'T3_{variant}_p']     = _binom_gt_alpha(k_sig, n)
            rec[f'T4_{variant}_p']     = _wilcoxon_perm_p(p)
        # T6: dsrinf r vs r at first predicted lag (paired) ----------
        if pred_lags:
            r_inf  = g[f'r_dsrinf_{tag}'].to_numpy(dtype=float)
            r_pred = g[f'r_lag{pred_lags[0]:03d}_{tag}'].to_numpy(dtype=float)
            diff = r_inf - r_pred
            m = np.isfinite(diff)
            if m.sum() >= 2:
                res = stats.ttest_1samp(diff[m], 0.0, alternative='greater')
                rec.update({'T6_t': float(res.statistic),
                            'T6_p': float(res.pvalue),
                            'T6_meanDiff': float(diff[m].mean())})
            else:
                rec.update({'T6_t': np.nan, 'T6_p': np.nan,
                            'T6_meanDiff': np.nan})
        else:
            rec.update({'T6_t': np.nan, 'T6_p': np.nan, 'T6_meanDiff': np.nan})
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # BH-FDR within each family ---------------------------------------
    fam_t1 = [c for c in out.columns if c.startswith('T1_p_lag')]
    fam_t3 = [c for c in out.columns if c.startswith('T3_p_lag')]
    fam_t4 = [c for c in out.columns if c.startswith('T4_p_lag')]
    fam_t3_agg = [c for c in out.columns
                  if c.startswith('T3_') and c.endswith('_p') and 'lag' not in c]
    fam_t4_agg = [c for c in out.columns
                  if c.startswith('T4_') and c.endswith('_p') and 'lag' not in c]
    fam_t5 = [c for c in out.columns if c.startswith('T5_') and c.endswith('_p')]
    for fam in (fam_t1, fam_t3, fam_t4, fam_t3_agg, fam_t4_agg, fam_t5):
        for col in fam:
            out[col + '_fdr'] = _bh_fdr(out[col].to_numpy(dtype=float))
    for col in ('T1a_avgPred_p', 'T2_p', 'T6_p'):
        if col in out.columns:
            out[col + '_fdr'] = _bh_fdr(out[col].to_numpy(dtype=float))
    return out


def fisher_per_roi_stats_noctrl(df):
    """No-control Fisher-z sensitivity statistics.

    Each value entering this function is already a per-cell mean across
    leave-one-configuration-out folds. We transform those means (rather
    than individual folds), then use two-sided t-tests. The returned table
    deliberately keeps the ordinary T1/T2/T5/T6 column names so it can use
    the standard plotting code; ``meanR`` fields retain raw-r descriptives
    and ``meanZ``/``meanDiffZ`` identify the inferential scale.
    """
    rows = []
    for roi, g in df.groupby('roi'):
        rec = {'roi': roi, 'ctrl_mode': 'noctrl', 'n_cells': len(g)}
        z_mat = np.stack([
            _fisher_z(g[f'r_lag{lag:03d}_noctrl'].to_numpy(dtype=float))
            for lag in LAGS_DEG
        ], axis=1)
        r_mat = np.stack([
            g[f'r_lag{lag:03d}_noctrl'].to_numpy(dtype=float)
            for lag in LAGS_DEG
        ], axis=1)
        for li, lag in enumerate(LAGS_DEG):
            z = z_mat[:, li]
            t, p, n = _ttest_2sided_zero(z)
            rec.update({
                f'T1_t_lag{lag:03d}': t,
                f'T1_p_lag{lag:03d}': p,
                f'T1_n_lag{lag:03d}': n,
                f'T1_meanZ_lag{lag:03d}': float(np.nanmean(z)),
                f'T1_meanR_lag{lag:03d}': float(np.nanmean(r_mat[:, li])),
            })

        pred_lags = ROI_PREDICTED_LAGS_DEG.get(roi, None)
        if pred_lags:
            idx_pred = [LAGS_DEG.index(l) for l in pred_lags]
            idx_other = [i for i in range(len(LAGS_DEG)) if i not in idx_pred]
            tgt = np.nanmean(z_mat[:, idx_pred], axis=1)
            other = np.nanmean(z_mat[:, idx_other], axis=1)
            t, p, n = _ttest_2sided_zero(tgt)
            rec.update({'T1a_avgPred_t': t, 'T1a_avgPred_p': p,
                        'T1a_avgPred_n': n,
                        'T1a_avgPred_meanZ': float(np.nanmean(tgt))})
            diff = tgt - other
            t, p, n = _ttest_2sided_paired(tgt, other)
            rec.update({'T2_t': t, 'T2_p': p, 'T2_n': n,
                        'T2_meanDiffZ': float(np.nanmean(diff))})
        else:
            rec.update({'T1a_avgPred_t': np.nan, 'T1a_avgPred_p': np.nan,
                        'T1a_avgPred_n': 0, 'T1a_avgPred_meanZ': np.nan,
                        'T2_t': np.nan, 'T2_p': np.nan, 'T2_n': 0,
                        'T2_meanDiffZ': np.nan})

        for variant in ('dsrfull', 'dsrinf'):
            raw = g[f'r_{variant}_noctrl'].to_numpy(dtype=float)
            z = _fisher_z(raw)
            t, p, n = _ttest_2sided_zero(z)
            rec.update({f'T5_{variant}_t': t, f'T5_{variant}_p': p,
                        f'T5_{variant}_n': n,
                        f'T5_{variant}_meanZ': float(np.nanmean(z)),
                        f'T5_{variant}_meanR': float(np.nanmean(raw))})

        if pred_lags:
            z_inf = _fisher_z(g['r_dsrinf_noctrl'].to_numpy(dtype=float))
            z_pred = _fisher_z(g[
                f'r_lag{pred_lags[0]:03d}_noctrl'].to_numpy(dtype=float))
            diff = z_inf - z_pred
            t, p, n = _ttest_2sided_paired(z_inf, z_pred)
            rec.update({'T6_t': t, 'T6_p': p, 'T6_n': n,
                        'T6_meanDiffZ': float(np.nanmean(diff))})
        else:
            rec.update({'T6_t': np.nan, 'T6_p': np.nan, 'T6_n': 0,
                        'T6_meanDiffZ': np.nan})
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in [c for c in out.columns if c.startswith('T1_p_lag')]:
        out[col + '_fdr'] = _bh_fdr(out[col].to_numpy(dtype=float))
    for col in ('T1a_avgPred_p', 'T2_p', 'T5_dsrfull_p', 'T5_dsrinf_p',
                'T6_p'):
        out[col + '_fdr'] = _bh_fdr(out[col].to_numpy(dtype=float))
    return out


# ── Figures ───────────────────────────────────────────────────────────
def fig_roi_lag_heatmap(roi_stats, ctrl_mode, save_stem, fisher_z=False):
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    df = roi_stats[roi_stats['ctrl_mode'] == tag]
    rois = [r for r in ROIS_TO_RUN if r in df['roi'].values]
    T = np.full((len(rois), len(LAGS_DEG)), np.nan)
    Q = np.full((len(rois), len(LAGS_DEG)), np.nan)
    for ri, roi in enumerate(rois):
        rs = df[df['roi'] == roi].iloc[0]
        for li, lag in enumerate(LAGS_DEG):
            T[ri, li] = rs[f'T1_t_lag{lag:03d}']
            Q[ri, li] = rs.get(f'T1_p_lag{lag:03d}_fdr', np.nan)
    pred_str = ', '.join(f'{r}={list(ROI_PREDICTED_LAGS_DEG[r])}°'
                          for r in ROI_PREDICTED_LAGS_DEG)
    plot_roi_lag_tstat_heatmap(
        T, LAGS_DEG, rois, q_matrix=Q,
        predicted_lags_per_roi=ROI_PREDICTED_LAGS_DEG,
        save_stem=save_stem,
        t_label='t (two-sided)' if fisher_z else 't (one-sided > 0)',
        title=(f'Fig 3a — ROI × lag '
               f'{"two-sided Fisher-z t-stat" if fisher_z else "t-stat vs 0"}'
               f'  [{tag}]\n'
               f'Predicted lags (black outline): {pred_str}'),
    )


def fig_roi_lag_curves(per_cell, ctrl_mode, save_stem, fisher_z=False):
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    rois = [r for r in ROIS_TO_RUN if r in per_cell['roi'].unique()]
    curves = {}
    for roi in rois:
        g = per_cell[per_cell['roi'] == roi]
        curves[roi] = np.stack([(
            _fisher_z(g[f'r_lag{lag:03d}_{tag}'].to_numpy(dtype=float))
            if fisher_z else g[f'r_lag{lag:03d}_{tag}'].to_numpy(dtype=float)
        ) for lag in LAGS_DEG], axis=1)
    plot_roi_lag_curves(
        curves, LAGS_DEG,
        predicted_lags_per_roi=ROI_PREDICTED_LAGS_DEG,
        roi_colours=ROI_COLOURS,
        save_stem=save_stem,
        title=(f'Per-ROI mean {"Fisher z of fold-averaged CV r" if fisher_z else "CV r"}'
               f' across lags  [{tag}]'),
    )


def fig_perm_sig_fraction_bar(roi_stats, ctrl_mode, save_stem):
    """Best-of-predicted-lags perm-sig fraction per ROI (single number per ROI)."""
    _set_rc()
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    df = roi_stats[roi_stats['ctrl_mode'] == tag]
    rois = [r for r in ROIS_TO_RUN if r in df['roi'].values]
    fig, ax = plt.subplots(figsize=(12 * CM, 5 * CM), constrained_layout=True)
    x = np.arange(len(rois))
    fracs, q_best = [], []
    for roi in rois:
        rs = df[df['roi'] == roi].iloc[0]
        pred = ROI_PREDICTED_LAGS_DEG.get(roi, LAGS_DEG)
        ks = [rs[f'T3_k_lag{l:03d}'] for l in pred]
        qs = [rs.get(f'T3_p_lag{l:03d}_fdr', np.nan) for l in pred]
        fracs.append(max(ks) / rs['n_cells'] if rs['n_cells'] > 0 else np.nan)
        q_best.append(min([q for q in qs if np.isfinite(q)], default=np.nan))
    ax.bar(x, fracs, color=[ROI_COLOURS.get(r, '#888') for r in rois],
           edgecolor='black', linewidth=0.4, width=0.7)
    ax.axhline(ALPHA, color='black', lw=0.6, ls='--',
               label=f'chance = α = {ALPHA}')
    for xi, (f_, q_) in enumerate(zip(fracs, q_best)):
        s = p_to_stars(q_) if np.isfinite(q_) else ''
        if s and np.isfinite(f_):
            ax.text(xi, f_ + 0.005, s, ha='center', va='bottom',
                    fontsize=FONT_BIG, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=35, ha='right', fontsize=FONT_TICK)
    ax.set_ylabel(f'frac cells with p_perm < {ALPHA}', fontsize=FONT_AXIS)
    ax.set_title(f'Perm-sig fraction at predicted lag(s)  [{tag}]',
                 fontsize=FONT_TICK)
    ax.legend(fontsize=FONT_TICK - 1, frameon=False)
    _save(fig, save_stem)


def fig_perm_sig_fraction_heatmap(roi_stats, ctrl_mode, save_stem):
    """Fig 3b — ROI × lag fraction-of-perm-sig-cells heatmap with FDR stars
    and predicted-lag outlines.
    """
    _set_rc()
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    df = roi_stats[roi_stats['ctrl_mode'] == tag]
    rois = [r for r in ROIS_TO_RUN if r in df['roi'].values]
    M = np.full((len(rois), len(LAGS_DEG)), np.nan)
    Q = np.full((len(rois), len(LAGS_DEG)), np.nan)
    for ri, roi in enumerate(rois):
        rs = df[df['roi'] == roi].iloc[0]
        n = rs['n_cells']
        for li, lag in enumerate(LAGS_DEG):
            M[ri, li] = rs[f'T3_k_lag{lag:03d}'] / n if n > 0 else np.nan
            Q[ri, li] = rs.get(f'T3_p_lag{lag:03d}_fdr', np.nan)
    fig, ax = plt.subplots(figsize=(14 * CM, max(3.5, 0.55 * len(rois)) * CM),
                            constrained_layout=True)
    vmax = max(0.30, float(np.nanmax(M)) if np.isfinite(M).any() else 0.30)
    im = ax.imshow(M, cmap='Reds', vmin=0, vmax=vmax, aspect='auto')
    # Predicted-lag outlines + chance dashed level via text annotation
    for ri, roi in enumerate(rois):
        for tl in ROI_PREDICTED_LAGS_DEG.get(roi, ()):
            if tl in LAGS_DEG:
                ci = LAGS_DEG.index(tl)
                ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                            fill=False, edgecolor='black',
                                            lw=1.2))
    for ri in range(len(rois)):
        for ci in range(len(LAGS_DEG)):
            q = Q[ri, ci]
            if np.isfinite(q):
                s = p_to_stars(q)
                if s:
                    col = 'white' if M[ri, ci] > vmax * 0.55 else 'black'
                    ax.text(ci, ri, s, ha='center', va='center',
                            fontsize=FONT_TICK, fontweight='bold', color=col)
    ax.set_xticks(range(len(LAGS_DEG)))
    ax.set_xticklabels([str(l) for l in LAGS_DEG], fontsize=FONT_TICK)
    ax.set_yticks(range(len(rois)))
    ax.set_yticklabels(rois, fontsize=FONT_TICK)
    ax.set_xlabel('lag (°)', fontsize=FONT_AXIS)
    ax.set_title(f'Fig 3b — ROI × lag fraction of cells with p_perm < {ALPHA}\n'
                  f'[{tag}]  (chance = α = {ALPHA}; stars = BH-FDR binomial)',
                  fontsize=FONT_TICK)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cb.set_label('frac perm-sig', fontsize=FONT_TICK)
    cb.ax.tick_params(labelsize=FONT_TICK)
    _save(fig, save_stem)


def fig_dsrfull_vs_dsrinf_scatter(per_cell, ctrl_mode, save_stem, fisher_z=False):
    """Per-ROI scatter: dsrfull r vs dsrinf r. Replaces winner-curse Fig 4."""
    _set_rc()
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    rois = [r for r in ROIS_TO_RUN if r in per_cell['roi'].unique()]
    n_cols = min(len(rois), 4)
    n_rows = int(np.ceil(len(rois) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.5 * CM * n_cols, 3.5 * CM * n_rows),
                              constrained_layout=True, squeeze=False)
    for ax in axes.ravel()[len(rois):]:
        ax.axis('off')
    for ax, roi in zip(axes.ravel(), rois):
        g = per_cell[per_cell['roi'] == roi]
        full = g[f'r_dsrfull_{tag}'].to_numpy(dtype=float)
        inf_ = g[f'r_dsrinf_{tag}'].to_numpy(dtype=float)
        if fisher_z:
            full, inf_ = _fisher_z(full), _fisher_z(inf_)
        m = np.isfinite(full) & np.isfinite(inf_)
        col = ROI_COLOURS.get(roi, '#888')
        ax.scatter(full[m], inf_[m], s=6, color=col, alpha=0.7, edgecolor='none')
        if m.any():
            lim_lo = float(np.nanmin([full[m].min(), inf_[m].min()]))
            lim_hi = float(np.nanmax([full[m].max(), inf_[m].max()]))
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                    color='gray', lw=0.5, ls='--')
        ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
        ax.set_title(roi, fontsize=FONT_TICK)
        ax.tick_params(labelsize=FONT_TICK - 1, length=1.5, pad=1)
        ax.set_xlabel(f'dsr_full {"Fisher z" if fisher_z else "r"}', fontsize=FONT_TICK)
        ax.set_ylabel(f'dsr_inf {"Fisher z" if fisher_z else "r"}', fontsize=FONT_TICK)
    fig.suptitle(f'dsr_full (11 lags) vs dsr_inf (30/60/90) '
                 f'{"Fisher z of fold-averaged CV r" if fisher_z else "CV r"}  [{tag}]',
                 fontsize=FONT_AXIS)
    _save(fig, save_stem)


def fig_ctrl_vs_noctrl_scatter(per_cell, save_stem):
    _set_rc()
    rois = [r for r in ROIS_TO_RUN if r in per_cell['roi'].unique()]
    n_cols = min(len(rois), 4)
    n_rows = int(np.ceil(len(rois) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.5 * CM * n_cols, 3.5 * CM * n_rows),
                              constrained_layout=True, squeeze=False)
    for ax in axes.ravel()[len(rois):]:
        ax.axis('off')
    for ax, roi in zip(axes.ravel(), rois):
        g = per_cell[per_cell['roi'] == roi]
        pred = ROI_PREDICTED_LAGS_DEG.get(roi, [0])
        lag = pred[0]
        noc = g[f'r_lag{lag:03d}_noctrl'].to_numpy(dtype=float)
        ctl = g[f'r_lag{lag:03d}_ctrl'].to_numpy(dtype=float)
        m = np.isfinite(noc) & np.isfinite(ctl)
        col = ROI_COLOURS.get(roi, '#888')
        ax.scatter(noc[m], ctl[m], s=6, color=col, alpha=0.7, edgecolor='none')
        if m.any():
            lim_lo = float(np.nanmin([noc[m].min(), ctl[m].min()]))
            lim_hi = float(np.nanmax([noc[m].max(), ctl[m].max()]))
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                    color='gray', lw=0.5, ls='--')
        ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
        ax.set_title(f'{roi}  lag={lag}°', fontsize=FONT_TICK)
        ax.tick_params(labelsize=FONT_TICK - 1, length=1.5, pad=1)
        ax.set_xlabel('r without controls', fontsize=FONT_TICK)
        ax.set_ylabel('r with controls', fontsize=FONT_TICK)
    fig.suptitle('Per-cell CV r at first predicted lag — controls vs no-controls',
                 fontsize=FONT_AXIS)
    _save(fig, save_stem)


def single_lag_stats(per_cell, single_lags=SINGLE_LAGS_FOR_TESTS):
    """Per (ROI × ctrl_mode × single lag) one-sample t-test of CV r > 0,
    plus perm-sig fraction. BH-FDR is applied within each (ctrl_mode ×
    lag) family (across ROIs). This is the "lag-agnostic" test the user
    asked for — no lag sets, one number per (ROI, lag).
    """
    rows = []
    for ctrl in (False, True):
        tag = 'ctrl' if ctrl else 'noctrl'
        for roi, g in per_cell.groupby('roi'):
            n_cells = len(g)
            for lag in single_lags:
                r = g[f'r_lag{lag:03d}_{tag}'].to_numpy(dtype=float)
                p = g[f'p_lag{lag:03d}_{tag}'].to_numpy(dtype=float)
                t_val, p_val = _ttest_gt0(r)
                k_sig = int(np.sum(np.isfinite(p) & (p < ALPHA)))
                rows.append({
                    'roi': roi, 'ctrl_mode': tag, 'lag_deg': lag,
                    'n_cells': n_cells,
                    'mean_r': float(np.nanmean(r)),
                    't_vs_0': t_val, 'p_unc': p_val,
                    'k_perm_sig': k_sig,
                    'frac_perm_sig': k_sig / n_cells if n_cells else np.nan,
                    'p_binom': _binom_gt_alpha(k_sig, n_cells),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # BH-FDR within each (ctrl_mode × lag) family, across the reported ROIs.
    out['p_fdr']       = np.nan
    out['p_binom_fdr'] = np.nan
    for (tag, lag), idx in out.groupby(['ctrl_mode', 'lag_deg']).indices.items():
        idx = list(idx)
        out.loc[idx, 'p_fdr']       = _bh_fdr(out.loc[idx, 'p_unc'].to_numpy(float))
        out.loc[idx, 'p_binom_fdr'] = _bh_fdr(out.loc[idx, 'p_binom'].to_numpy(float))
    return out


def fisher_single_lag_stats_noctrl(per_cell, single_lags=SINGLE_LAGS_FOR_TESTS):
    """Two-sided Fisher-z tests for the no-control fixed-lag summaries."""
    rows = []
    for roi, g in per_cell.groupby('roi'):
        for lag in single_lags:
            raw = g[f'r_lag{lag:03d}_noctrl'].to_numpy(dtype=float)
            z = _fisher_z(raw)
            t, p, n = _ttest_2sided_zero(z)
            rows.append({
                'roi': roi, 'ctrl_mode': 'noctrl', 'lag_deg': lag,
                'n_cells': n, 'df': n - 1 if n else np.nan,
                'mean_r': float(np.nanmean(raw)), 'mean_z': float(np.nanmean(z)),
                't_vs_0': t, 'p_unc': p,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out['p_fdr'] = np.nan
    for _, idx in out.groupby('lag_deg').indices.items():
        idx = list(idx)
        out.loc[idx, 'p_fdr'] = _bh_fdr(out.loc[idx, 'p_unc'].to_numpy(float))
    return out


# ── Automatically written analysis reports ────────────────────────────
def _fmt_p(p):
    if not np.isfinite(p):
        return 'NA'
    return '< .001' if p < 0.001 else f'= {p:.3f}'


def _fmt_num(x, digits=3):
    return 'NA' if not np.isfinite(x) else f'{x:.{digits}f}'


def _core_result_rows(raw_roi_stats, fisher_roi_stats):
    """Build the compact, requested no-control result summary table."""
    raw = raw_roi_stats[raw_roi_stats['ctrl_mode'] == 'noctrl'].set_index('roi')
    fisher = fisher_roi_stats.set_index('roi')
    rows = []
    for roi, lags in ROI_PREDICTED_LAGS_DEG.items():
        if roi not in raw.index or roi not in fisher.index:
            continue
        r_raw, r_fisher = raw.loc[roi], fisher.loc[roi]
        for lag in lags:
            n = int(r_fisher[f'T1_n_lag{lag:03d}'])
            n_roi = int(r_raw['n_cells'])
            k = int(r_raw[f'T3_k_lag{lag:03d}'])
            rows.append({
                'result_type': 'individual_lag_vs_zero',
                'roi': roi, 'lag_deg': lag,
                'comparison': f'{lag}° lag vs 0',
                'test': 'two-sided one-sample t-test on Fisher z',
                'n': n, 'df': n - 1,
                'mean_raw_r': r_fisher[f'T1_meanR_lag{lag:03d}'],
                'mean_fisher_z': r_fisher[f'T1_meanZ_lag{lag:03d}'],
                't': r_fisher[f'T1_t_lag{lag:03d}'],
                'p_unc': r_fisher[f'T1_p_lag{lag:03d}'],
                'p_fdr': r_fisher[f'T1_p_lag{lag:03d}_fdr'],
                'perm_sig_cells': k,
                'roi_cells': n_roi,
                'perm_sig_percent': 100 * k / n_roi if n_roi else np.nan,
                'perm_binom_p_fdr': r_raw.get(f'T3_p_lag{lag:03d}_fdr', np.nan),
            })
        lag_label = '/'.join(str(l) for l in lags) + '°'
        n = int(r_fisher['T1a_avgPred_n'])
        rows.append({
            'result_type': 'mean_predicted_lags_vs_zero',
            'roi': roi, 'lag_deg': np.nan,
            'comparison': f'mean of predicted lags ({lag_label}) vs 0',
            'test': 'two-sided one-sample t-test on Fisher z',
            'n': n, 'df': n - 1,
            'mean_raw_r': np.nan,
            'mean_fisher_z': r_fisher['T1a_avgPred_meanZ'],
            't': r_fisher['T1a_avgPred_t'],
            'p_unc': r_fisher['T1a_avgPred_p'],
            'p_fdr': r_fisher['T1a_avgPred_p_fdr'],
            'perm_sig_cells': np.nan, 'roi_cells': np.nan,
            'perm_sig_percent': np.nan, 'perm_binom_p_fdr': np.nan,
        })
        n = int(r_fisher['T2_n'])
        rows.append({
            'result_type': 'predicted_lags_vs_other_lags',
            'roi': roi, 'lag_deg': np.nan,
            'comparison': f'predicted lags ({lag_label}) vs all other lags',
            'test': 'two-sided paired t-test on Fisher z',
            'n': n, 'df': n - 1,
            'mean_raw_r': np.nan,
            'mean_fisher_z': r_fisher['T2_meanDiffZ'],
            't': r_fisher['T2_t'], 'p_unc': r_fisher['T2_p'],
            'p_fdr': r_fisher['T2_p_fdr'],
            'perm_sig_cells': np.nan, 'roi_cells': np.nan,
            'perm_sig_percent': np.nan, 'perm_binom_p_fdr': np.nan,
        })
    return pd.DataFrame(rows)


def _write_methods_and_results(out_dir, raw_roi_stats, fisher_roi_stats):
    """Write methods, a readable results narrative, and a core-result table."""
    core = _core_result_rows(raw_roi_stats, fisher_roi_stats)
    core_csv = os.path.join(out_dir, 'core_results_noctrl_fisher_z.csv')
    core_md = os.path.join(out_dir, 'core_results_noctrl_fisher_z.md')
    methods_md = os.path.join(out_dir, 'methods.md')
    results_md = os.path.join(out_dir, 'results.md')
    core.to_csv(core_csv, index=False)

    methods = f"""# Methods: per-lag encoding

This analysis estimates location encoding at lags from 0° to 330° in 30°
steps. For each neuron, firing rate is averaged separately for each of the
{N_LOC} task locations within each task configuration, yielding a lag-shifted
rate map. A leave-one-configuration-out procedure predicts the held-out map
from dwell-time-weighted training maps and scores prediction with a
{'weighted' if WEIGHTED_CORRELATION else 'standard'} Pearson correlation.
The per-cell statistic is the mean correlation across valid held-out folds.

The no-control analysis uses phase-residualised firing rates without the
additional state/button/location partial-out controls. Permutation p-values
are obtained by circularly shifting the held-out location series, rebuilding
the held-out rate map, and comparing the observed mean correlation with the
permutation distribution ({N_PERMUTATIONS} permutations; alpha = {ALPHA}).

For the Fisher-z sensitivity analysis, each cell's fold-averaged correlation
is transformed as `z = arctanh(r)` (values at ±1 are clipped by
{FISHER_Z_CLIP_EPS:g} solely to keep the transform finite). The reported
Fisher-z tests are two-sided: individual-lag and predicted-lag-average tests
are one-sample tests against zero; predicted-versus-other-lag tests are paired
tests within cells. Benjamini-Hochberg FDR correction is applied across ROIs
within each test family. Raw correlations remain the descriptive effect-size
scale. The original raw-r, one-sided statistics and the Fisher-z sensitivity
results are both retained in this output directory.
"""
    with open(methods_md, 'w') as f:
        f.write(methods)

    lag_rows = core[core['result_type'] == 'individual_lag_vs_zero']
    summary_rows = core[core['result_type'] != 'individual_lag_vs_zero']
    core_lines = [
        '# Core no-control results (Fisher-z sensitivity analysis)',
        '',
        'Individual predicted-lag tests are two-sided one-sample t-tests of '
        'Fisher-transformed, fold-averaged CV correlations against zero. '
        'The permutation-significant percentage uses the original per-cell '
        f'permutation p-values (p < {ALPHA}).',
        '',
        '| ROI | Lag | n | Mean raw r | t(df) | p | FDR p | Permutation-significant cells |',
        '| --- | ---: | ---: | ---: | --- | --- | --- | --- |',
    ]
    for _, row in lag_rows.iterrows():
        core_lines.append(
            f"| {row.roi} | {int(row.lag_deg)}° | {int(row.n)} | "
            f"{_fmt_num(row.mean_raw_r)} | t({int(row.df)}) = {_fmt_num(row.t, 2)} | "
            f"{_fmt_p(row.p_unc)} | {_fmt_p(row.p_fdr)} | "
            f"{int(row.perm_sig_cells)}/{int(row.roi_cells)} "
            f"({_fmt_num(row.perm_sig_percent, 1)}%) |")
    core_lines += [
        '',
        '## Predicted-lag summary and specificity tests',
        '',
        '| ROI | Comparison | n | t(df) | p | FDR p |',
        '| --- | --- | ---: | --- | --- | --- |',
    ]
    for _, row in summary_rows.iterrows():
        core_lines.append(
            f"| {row.roi} | {row.comparison} ({row.test}) | {int(row.n)} | "
            f"t({int(row.df)}) = {_fmt_num(row.t, 2)} | {_fmt_p(row.p_unc)} | "
            f"{_fmt_p(row.p_fdr)} |")
    with open(core_md, 'w') as f:
        f.write('\n'.join(core_lines) + '\n')

    results_lines = [
        '# Results: per-lag encoding',
        '',
        'This report summarizes the requested no-control Fisher-z sensitivity '
        'analysis. Tests use Fisher-transformed, fold-averaged per-cell CV '
        'correlations and are two-sided; raw mean r is reported for effect-size '
        'interpretation. FDR p-values are corrected across ROIs within the '
        'relevant test family.',
        '',
    ]
    for roi in ROI_PREDICTED_LAGS_DEG:
        roi_lags = lag_rows[lag_rows['roi'] == roi]
        roi_summary = summary_rows[summary_rows['roi'] == roi]
        if roi_lags.empty:
            continue
        results_lines += [f'## {roi}', '']
        for _, row in roi_lags.iterrows():
            results_lines.append(
                f"At {int(row.lag_deg)}°, mean raw r = {_fmt_num(row.mean_raw_r)}; "
                f"t({int(row.df)}) = {_fmt_num(row.t, 2)}, p {_fmt_p(row.p_unc)}, "
                f"FDR p {_fmt_p(row.p_fdr)}. {int(row.perm_sig_cells)}/"
                f"{int(row.roi_cells)} cells ({_fmt_num(row.perm_sig_percent, 1)}%) "
                f"were permutation-significant.")
        for _, row in roi_summary.iterrows():
            label = ('The predicted-lag average versus zero' if
                     row.result_type == 'mean_predicted_lags_vs_zero' else
                     'The paired predicted-versus-other-lags specificity test')
            results_lines.append(
                f"{label}: t({int(row.df)}) = {_fmt_num(row.t, 2)}, "
                f"p {_fmt_p(row.p_unc)}, FDR p {_fmt_p(row.p_fdr)}.")
        results_lines.append('')
    results_lines += [
        'The complete machine-readable core table is '
        '[`core_results_noctrl_fisher_z.csv`](core_results_noctrl_fisher_z.csv); '
        'the corresponding compact table is '
        '[`core_results_noctrl_fisher_z.md`](core_results_noctrl_fisher_z.md).',
    ]
    with open(results_md, 'w') as f:
        f.write('\n'.join(results_lines) + '\n')
    return methods_md, results_md, core_md, core_csv


def fig_per_lag_r_hist_all_rois(per_cell, single_lag_df, ctrl_mode,
                                  fixed_lag_deg, save_stem, fisher_z=False):
    """One page per fixed lag: histogram of per-cell CV r across each ROI,
    annotated with the single-lag t vs 0 and its BH-FDR p. Mirrors the
    heatmap row-by-row as histograms."""
    _set_rc()
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    rois = [r for r in ROIS_TO_RUN if r in per_cell['roi'].unique()]
    n = len(rois)
    n_cols = min(n, 4)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.8 * CM * n_cols, 3.0 * CM * n_rows),
                              constrained_layout=True, squeeze=False)
    axes_flat = axes.ravel()
    for ax in axes_flat[n:]:
        ax.axis('off')
    # shared x-range across ROIs at this lag for comparability
    all_r = per_cell[f'r_lag{fixed_lag_deg:03d}_{tag}'].to_numpy(dtype=float)
    if fisher_z:
        all_r = _fisher_z(all_r)
    all_r = all_r[np.isfinite(all_r)]
    if all_r.size:
        lo, hi = np.nanpercentile(all_r, [1, 99])
        bins = np.linspace(lo, hi, 22)
    else:
        bins = 20
    for ax, roi in zip(axes_flat, rois):
        g = per_cell[per_cell['roi'] == roi]
        r = g[f'r_lag{fixed_lag_deg:03d}_{tag}'].to_numpy(dtype=float)
        p = g[f'p_lag{fixed_lag_deg:03d}_{tag}'].to_numpy(dtype=float)
        if fisher_z:
            r = _fisher_z(r)
        m = np.isfinite(r); r = r[m]; p = p[m]
        col = ROI_COLOURS.get(roi, '#888')
        if r.size:
            ax.hist(r, bins=bins, color='lightgray',
                    edgecolor='black', linewidth=0.2, alpha=0.7)
            sig = np.isfinite(p) & (p < ALPHA)
            if sig.any():
                ax.hist(r[sig], bins=bins, color=col,
                        edgecolor='black', linewidth=0.2, alpha=0.95)
            ax.axvline(r.mean(), color='black', lw=0.8)
        ax.axvline(0, color='gray', ls='--', lw=0.4)
        # pull t / p / p_fdr for this (ROI, ctrl_mode, lag) from the summary
        row = single_lag_df[
            (single_lag_df['roi'] == roi)
            & (single_lag_df['ctrl_mode'] == tag)
            & (single_lag_df['lag_deg'] == fixed_lag_deg)
        ]
        if not row.empty:
            rr = row.iloc[0]
            t_ = rr['t_vs_0']; p_ = rr['p_unc']; q_ = rr['p_fdr']
            title = (f'{_disp(roi)}\n'
                     f't = {t_:+.2f}  p = {p_:.3g}\np_FDR = {q_:.3g}')
        else:
            title = _disp(roi)
        ax.set_title(title, fontsize=FONT_TICK)
        ax.tick_params(labelsize=FONT_TICK - 1, length=1.5, pad=1)
        ax.set_xlabel('Fisher z (fold-averaged CV r)' if fisher_z else 'CV r',
                      fontsize=FONT_TICK)
        ax.set_ylabel('# cells', fontsize=FONT_TICK)
    fig.suptitle(
        f'Per-ROI {"Fisher z" if fisher_z else "CV r"} distribution at lag '
        f'{fixed_lag_deg}°  [{tag}]\n'
        f'({"two-sided Fisher-z t-test" if fisher_z else "one-sample t vs 0"}, '
        f'FDR across {n} ROIs; sig cells overlaid)',
        fontsize=FONT_BIG,
    )
    _save(fig, save_stem)


def lag_lag_correlation(per_cell, ctrl_mode):
    """Descriptive: across cells, how correlated are the per-lag CV r
    values? Returns a long-format DataFrame with one row per
    (roi, lag_i, lag_j). Pearson r is computed pairwise (masking non-
    finite entries per pair) so each entry uses the largest possible
    common cell set. No significance testing — this is a descriptive
    summary of how similar the per-lag single-cell curves are."""
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    rows = []
    for roi in sorted(per_cell['roi'].dropna().unique()):
        g = per_cell[per_cell['roi'] == roi]
        R = np.stack([
            g[f'r_lag{lag:03d}_{tag}'].to_numpy(dtype=float)
            for lag in LAGS_DEG
        ], axis=1)   # (n_cells, n_lags)
        for i, li in enumerate(LAGS_DEG):
            for j, lj in enumerate(LAGS_DEG):
                x, y = R[:, i], R[:, j]
                m = np.isfinite(x) & np.isfinite(y)
                n = int(m.sum())
                if n < 3 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
                    r = np.nan
                else:
                    r = float(np.corrcoef(x[m], y[m])[0, 1])
                rows.append({'roi': roi, 'ctrl_mode': tag,
                             'lag_i': li, 'lag_j': lj,
                             'pearson_r': r, 'n_cells': n})
    return pd.DataFrame(rows)


def fig_lag_lag_correlation_heatmap(per_cell, ctrl_mode, save_stem):
    """One panel per ROI: heatmap of Pearson r across cells between the
    per-cell CV r at every pair of lags. Descriptive only — diagonal = 1
    by construction. Off-diagonal blocks near the diagonal indicate
    smooth per-cell tuning across neighbouring lags; distant off-diagonal
    values close to 0 indicate lag-independent per-cell fits."""
    _set_rc()
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    corr_df = lag_lag_correlation(per_cell, ctrl_mode)
    rois = [r for r in ROIS_TO_RUN if r in corr_df['roi'].unique()]
    n_lags = len(LAGS_DEG)
    n_cols = min(len(rois), 4)
    n_rows = int(np.ceil(len(rois) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.6 * CM * n_cols, 3.6 * CM * n_rows),
                              constrained_layout=True, squeeze=False)
    axes_flat = axes.ravel()
    for ax in axes_flat[len(rois):]:
        ax.axis('off')
    for ax, roi in zip(axes_flat, rois):
        sub = corr_df[corr_df['roi'] == roi]
        M = sub.pivot(index='lag_i', columns='lag_j', values='pearson_r')
        M = M.reindex(index=LAGS_DEG, columns=LAGS_DEG).to_numpy()
        im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        ax.set_xticks(range(n_lags))
        ax.set_yticks(range(n_lags))
        ax.set_xticklabels([str(l) if (l % 60 == 0) else ''
                             for l in LAGS_DEG],
                            fontsize=FONT_TICK - 2, rotation=45)
        ax.set_yticklabels([str(l) if (l % 60 == 0) else ''
                             for l in LAGS_DEG],
                            fontsize=FONT_TICK - 2)
        ax.set_title(roi, fontsize=FONT_TICK)
        ax.tick_params(axis='both', length=1.5, pad=1)
    fig.suptitle(
        f'Across-cell Pearson r between per-cell CV r at each pair of lags  '
        f'[{tag}]\n'
        '(descriptive; diagonal = 1 by construction)',
        fontsize=FONT_AXIS,
    )
    cbar = fig.colorbar(im, ax=axes_flat.tolist(), fraction=0.03,
                         pad=0.02, shrink=0.7)
    cbar.set_label('Pearson r (across cells)', fontsize=FONT_TICK)
    cbar.ax.tick_params(labelsize=FONT_TICK - 1)
    _save(fig, save_stem)


def fig_test2_target_vs_others_lines(per_cell, roi_stats, ctrl_mode, save_stem,
                                      fisher_z=False):
    """TEST 2 — per-ROI mean CV r across lags (±SEM) with predicted lag(s)
    highlighted. Mirrors `plot_test2_target_vs_others_lines` in
    spatial_peaks_simple. Uses display names (ACC → mPFC)."""
    _set_rc()
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    df_r = roi_stats[roi_stats['ctrl_mode'] == tag]
    rois = [r for r in ROIS_TO_RUN if r in per_cell['roi'].unique()]
    rois_with_target = [r for r in rois if ROI_PREDICTED_LAGS_DEG.get(r)]
    rois_show = rois_with_target or rois
    n = len(rois_show)
    n_cols = min(n, 4)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.4 * CM * n_cols, 3.6 * CM * n_rows),
                              constrained_layout=True, squeeze=False)
    axes_flat = axes.ravel()
    for ax in axes_flat[n:]:
        ax.axis('off')
    x = np.asarray(LAGS_DEG)
    for ax, roi in zip(axes_flat, rois_show):
        g = per_cell[per_cell['roi'] == roi]
        curves = np.stack([(
            _fisher_z(g[f'r_lag{lag:03d}_{tag}'].to_numpy(dtype=float))
            if fisher_z else g[f'r_lag{lag:03d}_{tag}'].to_numpy(dtype=float)
        ) for lag in LAGS_DEG], axis=1)   # (n_cells, n_lags)
        col = ROI_COLOURS.get(roi, '#888')
        if curves.size:
            m = np.nanmean(curves, axis=0)
            s = np.nanstd(curves, axis=0, ddof=1) / np.sqrt(
                np.maximum(np.isfinite(curves).sum(axis=0), 1)
            )
            ax.fill_between(x, m - s, m + s, color=col, alpha=0.25, linewidth=0)
            ax.plot(x, m, color=col, lw=1.6, marker='o', ms=2.5)
        ax.axhline(0.0, color='black', lw=0.5, ls='--')
        for tl in ROI_PREDICTED_LAGS_DEG.get(roi, ()):
            ax.axvline(tl, color=OBSERVED_GREEN, lw=0.9, ls=':', alpha=0.7)
        rs_row = df_r[df_r['roi'] == roi]
        if not rs_row.empty:
            rs = rs_row.iloc[0]
            t_ = rs.get('T2_t', np.nan)
            p_ = rs.get('T2_p', np.nan)
            q_ = rs.get('T2_p_fdr', np.nan)
        else:
            t_ = p_ = q_ = np.nan
        target = list(ROI_PREDICTED_LAGS_DEG.get(roi, ()))
        ax.set_title(
            f'{_disp(roi)}   target = {target}°\n'
            f'paired t = {t_:+.2f}   p = {p_:.3g}\n'
            f'p_FDR = {q_:.3g}',
            fontsize=FONT_TICK,
        )
        ax.set_xlabel('lag (°)', fontsize=FONT_TICK)
        ax.set_ylabel('mean Fisher z' if fisher_z else 'mean CV r',
                      fontsize=FONT_TICK)
        ax.set_xticks(LAGS_DEG[::2])
        ax.tick_params(axis='both', labelsize=FONT_TICK, length=2, pad=1)
    fig.suptitle(
        'TEST 2 — within-cell predicted-lag vs other-lags\n'
        f'({"paired two-sided t-test on Fisher z" if fisher_z else "paired t-test, one-sided greater"}; FDR across '
        f'{sum(1 for r in rois if ROI_PREDICTED_LAGS_DEG.get(r))} '
        f'predicted-lag ROIs)  [{tag}]',
        fontsize=FONT_BIG,
    )
    _save(fig, save_stem)


def fig_per_roi_r_hist(per_cell, ctrl_mode, save_stem, fisher_z=False):
    _set_rc()
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    rois = [r for r in ROIS_TO_RUN if r in per_cell['roi'].unique()]
    n_cols = min(len(rois), 4)
    n_rows = int(np.ceil(len(rois) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.5 * CM * n_cols, 2.5 * CM * n_rows),
                              constrained_layout=True, squeeze=False)
    for ax in axes.ravel()[len(rois):]:
        ax.axis('off')
    for ax, roi in zip(axes.ravel(), rois):
        g = per_cell[per_cell['roi'] == roi]
        pred = ROI_PREDICTED_LAGS_DEG.get(roi, [0])
        lag = pred[0]
        r = g[f'r_lag{lag:03d}_{tag}'].to_numpy(dtype=float)
        p = g[f'p_lag{lag:03d}_{tag}'].to_numpy(dtype=float)
        if fisher_z:
            r = _fisher_z(r)
        m = np.isfinite(r); r = r[m]; p = p[m]
        col = ROI_COLOURS.get(roi, '#888')
        if r.size:
            lo, hi = np.nanpercentile(r, [1, 99])
            bins = np.linspace(lo, hi, 20)
            ax.hist(r, bins=bins, color='lightgray',
                    edgecolor='black', linewidth=0.2, alpha=0.7)
            sig = p < ALPHA
            if sig.any():
                ax.hist(r[sig], bins=bins, color=col,
                        edgecolor='black', linewidth=0.2, alpha=0.95)
            ax.axvline(r.mean(), color='black', lw=0.8)
        ax.axvline(0, color='gray', ls='--', lw=0.4)
        ax.set_title(f'{roi}  lag={lag}°', fontsize=FONT_TICK)
        ax.tick_params(labelsize=FONT_TICK - 1, length=1.5, pad=1)
        ax.set_xlabel('Fisher z' if fisher_z else 'CV r', fontsize=FONT_TICK)
        ax.set_ylabel('# cells', fontsize=FONT_TICK)
    fig.suptitle(f'Per-ROI {"Fisher z" if fisher_z else "CV r"} distribution '
                 f'at predicted lag  [{tag}]',
                 fontsize=FONT_AXIS)
    _save(fig, save_stem)


# ── Main ──────────────────────────────────────────────────────────────
def _load_cells():
    cells = cs.load_cells(cell_set=CELL_SET, rois_keep=ROIS_TO_RUN)
    cells = cells.copy()
    cells['subject_id'] = cells['subject_id'].astype(str).str.zfill(2)
    if 'cell_idx' not in cells.columns:
        raise ValueError('cells table missing cell_idx column')
    for c in ('MNI_x', 'MNI_y', 'MNI_z'):
        if c not in cells.columns:
            cells[c] = np.nan
    return cells


# ── Model-regressor lag-lag correlation ───────────────────────────────
def _identity_rate(loc_series, lag_i_bins, lag_j_bins):
    """Fraction of bins where loc_at_lag_i == loc_at_lag_j (∈ [0, 1])."""
    li = np.roll(loc_series, -lag_i_bins)
    lj = np.roll(loc_series, -lag_j_bins)
    return float(np.mean(li == lj))


def compute_lag_regressor_correlations(cells_df, out_dir):
    """Descriptive: how correlated are the lag-shifted LOCATION regressors
    themselves — the ones we plug into the rate-map at each lag — across
    task configurations and across subjects?

    For each subject × configuration, for every pair of lags (i, j) we
    compute the *identity rate*: fraction of bins where the location at
    lag_i equals the location at lag_j. This is the natural regressor
    correlation for a categorical location series and directly indexes
    how much one lag-regressor will absorb the variance of another in
    the OLS partial-out.

    Writes:
      lag_regressor_correlation_long.csv  (subject, config_id, lag_i,
                                            lag_j, identity_rate)
      lag_regressor_correlation_mean.csv  (lag_i × lag_j mean matrix
                                            across all subject-configs)
    Returns the long-format DataFrame.
    """
    sub_list = sorted(cells_df['subject_id'].unique())
    print(f'\n══ Lag-regressor correlation across {len(sub_list)} subjects ══')
    rows = []
    for sub_str in sub_list:
        try:
            data_raw = hh.load_norm_data(DATA_DIR, [sub_str], res_data=False)
        except Exception as exc:
            print(f'  sub-{sub_str} load failed: {exc}'); continue
        if not data_raw:
            continue
        data = hh.filter_data(data_raw, int(sub_str), TRIALS)
        sub_dict = data[f'sub-{sub_str}']
        beh = sub_dict['beh'].copy().reset_index(drop=True)
        locs = sub_dict['locations'].to_numpy(dtype=float)
        _, _, idx_cfg, _ = np.unique(
            beh[['loc_A', 'loc_B', 'loc_C', 'loc_D']].to_numpy(),
            axis=0, return_index=True, return_inverse=True, return_counts=True,
        )
        n_cfg = len(np.unique(idx_cfg))
        if n_cfg < 2:
            continue
        for c in range(n_cfg):
            mask = idx_cfg == c
            if not mask.any():
                continue
            loc_series = _mode_per_bin_int(locs[mask])
            for i, li in enumerate(LAGS_DEG):
                for j, lj in enumerate(LAGS_DEG):
                    rows.append({
                        'subject_id':    sub_str,
                        'config_id':     int(c),
                        'lag_i':         li,
                        'lag_j':         lj,
                        'identity_rate': _identity_rate(loc_series, li, lj),
                    })
    long = pd.DataFrame(rows)
    if long.empty:
        print('  no subject data loaded; skipping regressor-correlation output.')
        return long
    long.to_csv(
        os.path.join(out_dir, 'lag_regressor_correlation_long.csv'),
        index=False)
    mean_mat = (long.groupby(['lag_i', 'lag_j'])['identity_rate']
                    .mean().unstack())
    mean_mat.to_csv(
        os.path.join(out_dir, 'lag_regressor_correlation_mean.csv'))
    n_sub = long.subject_id.nunique()
    n_cfg = len(long.groupby(['subject_id', 'config_id']))
    print(f'  computed {n_cfg} subject-config pairs across {n_sub} subjects.')
    print(f'  mean identity rate off-diagonal (0-lag reference vs others):')
    ref = mean_mat.loc[0].drop(0).round(3)
    print(ref.to_string())
    return long


def fig_lag_regressor_correlation(long_df, out_dir):
    """Two panels: (a) heatmap of mean identity rate across all subject-
    configs; (b) violin/box of the distribution across subject-configs
    for lag 0 vs every other lag."""
    _set_rc()
    if long_df.empty:
        return
    mean_mat = (long_df.groupby(['lag_i', 'lag_j'])['identity_rate']
                    .mean().unstack()
                    .reindex(index=LAGS_DEG, columns=LAGS_DEG))
    n_lags = len(LAGS_DEG)

    fig, axes = plt.subplots(1, 2, figsize=(15 * CM, 6 * CM),
                              constrained_layout=True,
                              gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax = axes[0]
    im = ax.imshow(mean_mat.to_numpy(), cmap='Reds', vmin=0, vmax=1,
                    aspect='equal')
    ax.set_xticks(range(n_lags))
    ax.set_yticks(range(n_lags))
    ax.set_xticklabels([str(l) for l in LAGS_DEG],
                        fontsize=FONT_TICK - 2, rotation=45)
    ax.set_yticklabels([str(l) for l in LAGS_DEG], fontsize=FONT_TICK - 2)
    ax.set_xlabel('lag j (°)', fontsize=FONT_TICK)
    ax.set_ylabel('lag i (°)', fontsize=FONT_TICK)
    ax.set_title('Mean identity rate\n(across subject-configs)',
                  fontsize=FONT_TICK)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label('P(loc@i == loc@j)', fontsize=FONT_TICK - 1)
    cb.ax.tick_params(labelsize=FONT_TICK - 2)
    for i in range(n_lags):
        for j in range(n_lags):
            v = mean_mat.iat[i, j]
            if np.isfinite(v):
                col = 'white' if v > 0.55 else 'black'
                ax.text(j, i, f'{v:.2f}'.lstrip('0') if v < 1 else '1',
                        ha='center', va='center',
                        fontsize=FONT_TICK - 4, color=col)

    ax = axes[1]
    ref = long_df[long_df['lag_i'] == 0]
    lags_show = [l for l in LAGS_DEG if l != 0]
    data = [ref[ref['lag_j'] == l]['identity_rate'].dropna().to_numpy()
             for l in lags_show]
    bp = ax.boxplot(data, positions=range(len(lags_show)), widths=0.6,
                     showfliers=False, patch_artist=True,
                     medianprops=dict(color='black', lw=0.9))
    for patch in bp['boxes']:
        patch.set_facecolor('#FCDDE3'); patch.set_edgecolor('black'); patch.set_lw(0.4)
    rng = np.random.default_rng(42)
    for k, d in enumerate(data):
        if d.size == 0:
            continue
        ax.scatter(np.full(d.size, k) + 0.08 * rng.standard_normal(d.size),
                    d, s=3, color='black', alpha=0.35, zorder=3)
    ax.axhline(1/9, color='gray', ls='--', lw=0.6,
                label='chance = 1/9 (9 locations)')
    ax.set_xticks(range(len(lags_show)))
    ax.set_xticklabels([str(l) for l in lags_show],
                        fontsize=FONT_TICK - 2, rotation=45)
    ax.set_xlabel('lag j (°)   [reference: lag i = 0°]', fontsize=FONT_TICK)
    ax.set_ylabel('identity rate  P(loc@0 == loc@j)', fontsize=FONT_TICK)
    ax.set_title('Distribution across subject-configs\n(one dot per config)',
                  fontsize=FONT_TICK)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=FONT_TICK - 2, frameon=False, loc='upper right')
    ax.tick_params(axis='both', labelsize=FONT_TICK - 2, length=1.5, pad=1)
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    _save(fig, os.path.join(fig_dir, '10_lag_regressor_correlation'))


def _write_lagwise_both_units(per_cell, out_dir):
    """Lag-wise one-sample tests at CELL and SUBJECT level, for both
    control modes, plus the predicted-lag window tests.

    Cells from one session are not independent, so the subject-level
    version (average Fisher-z across that session's cells, then test
    across sessions) is the conservative read; the cell-level version is
    the sensitive one. Both are written, and the manuscript states which
    it quotes. See `mc.analyse.lagwise_aggregation` for the rationale.
    """
    from mc.analyse.lagwise_aggregation import (
        lagwise_tests_both_units, target_window_tests_both_units,
    )
    lag_tables, win_tables = [], []
    for ctrl_mode in ('noctrl', 'ctrl'):
        cols = [f'r_lag{a:03d}_{ctrl_mode}' for a in LAGS_DEG]
        if not all(c in per_cell.columns for c in cols):
            continue
        curves_by_roi = {}
        for roi, g in per_cell.groupby('roi'):
            if roi not in ROIS_TO_RUN:
                continue
            curves = g[cols].to_numpy(dtype=float)
            if curves.size == 0 or not np.isfinite(curves).any():
                continue
            curves_by_roi[roi] = (curves, g['subject_id'].to_numpy())
        if not curves_by_roi:
            continue
        label = f'per_lag_encoding_{ctrl_mode}'
        lag_tables.append(
            lagwise_tests_both_units(curves_by_roi, LAGS_DEG, label))
        win_tables.append(
            target_window_tests_both_units(
                curves_by_roi, LAGS_DEG, ROI_PREDICTED_LAGS_DEG, label))

    if lag_tables:
        lag_df = pd.concat(lag_tables, ignore_index=True)
        lag_path = os.path.join(out_dir, 'per_roi_lagwise_by_unit.csv')
        lag_df.to_csv(lag_path, index=False)
        print(f'Saved per_roi_lagwise_by_unit.csv ({lag_df.shape[0]} rows) '
              f'— lag-wise tests at cell AND subject level')
        # Console summary for the ROIs with an a-priori prediction.
        for roi in ROI_PREDICTED_LAGS_DEG:
            sub = lag_df[(lag_df.roi == roi)
                         & (lag_df.analysis == 'per_lag_encoding_noctrl')]
            if sub.empty:
                continue
            for unit in ('cell', 'subject'):
                s = sub[sub.analysis_unit == unit]
                if s.empty:
                    continue
                best = s.loc[s['t_fisher_z'].idxmax()]
                print(f"    {roi:12s} [{unit:7s}] n={int(best['n_units_valid']):3d}  "
                      f"peak lag {int(best['lag_deg']):3d}deg  "
                      f"r={best['mean_raw_r']:+.3f}  t={best['t_fisher_z']:+.2f}  "
                      f"p={best['p_one_sided']:.4f}  "
                      f"q_lags={best['p_one_sided_fdr_lags']:.3f}")
    if win_tables:
        win_df = pd.concat(win_tables, ignore_index=True)
        win_path = os.path.join(out_dir, 'per_roi_predicted_window_by_unit.csv')
        win_df.to_csv(win_path, index=False)
        print(f'Saved per_roi_predicted_window_by_unit.csv '
              f'({win_df.shape[0]} rows) — predicted-lag tests at both units')


def _stats_and_plots(per_cell, out_dir, fig_dir):
    """Recompute per-ROI stats + all figures from a pre-computed
    per_cell DataFrame. Shared by the full-run and reload paths."""
    per_cell = _canonicalize_roi_names(per_cell)
    stats_dfs = [per_roi_stats(per_cell, ctrl_mode=False),
                 per_roi_stats(per_cell, ctrl_mode=True)]
    roi_stats = pd.concat(stats_dfs, ignore_index=True)
    roi_stats.to_csv(os.path.join(out_dir, 'per_roi_stats.csv'), index=False)
    print(f'\nSaved per_roi_stats.csv ({roi_stats.shape[0]} rows × '
          f'{roi_stats.shape[1]} cols)')

    # ── Lag-wise stats at BOTH units of analysis (cell + subject) ──────
    # Written here so the manuscript numbers come straight out of this
    # script and do not require running overlay_double_dissociation.py.
    # Same shared implementation both use, so they cannot drift apart.
    _write_lagwise_both_units(per_cell, out_dir)

    # Lag-agnostic single-lag summary (one t-test per (ROI × ctrl × lag))
    single_lag_df = single_lag_stats(per_cell, SINGLE_LAGS_FOR_TESTS)
    single_lag_df.to_csv(
        os.path.join(out_dir, 'per_roi_single_lag_stats.csv'), index=False)
    print(f'Saved per_roi_single_lag_stats.csv '
          f'({single_lag_df.shape[0]} rows) for lags {SINGLE_LAGS_FOR_TESTS}')

    # Sensitivity analysis requested for the no-control condition only.
    # The original raw-r outputs above remain untouched.
    fisher_roi_stats = fisher_per_roi_stats_noctrl(per_cell)
    fisher_roi_stats.to_csv(
        os.path.join(out_dir, 'per_roi_stats_fisher_z_noctrl.csv'), index=False)
    fisher_single_lag_df = fisher_single_lag_stats_noctrl(
        per_cell, SINGLE_LAGS_FOR_TESTS)
    fisher_single_lag_df.to_csv(
        os.path.join(out_dir, 'per_roi_single_lag_stats_fisher_z_noctrl.csv'),
        index=False)
    print('Saved Fisher-z no-control statistics (two-sided t-tests)')
    methods_md, results_md, core_md, core_csv = _write_methods_and_results(
        out_dir, roi_stats, fisher_roi_stats)
    print('Saved methods, results, and core-result reports: '
          f'{os.path.basename(methods_md)}, {os.path.basename(results_md)}, '
          f'{os.path.basename(core_md)}, {os.path.basename(core_csv)}')

    long_rows = []
    for ctrl in (False, True):
        tag = 'ctrl' if ctrl else 'noctrl'
        rs = roi_stats[roi_stats['ctrl_mode'] == tag]
        for _, row in rs.iterrows():
            for lag in LAGS_DEG:
                long_rows.append({
                    'roi': row['roi'], 'ctrl_mode': tag, 'lag_deg': lag,
                    'mean_r':     row[f'T1_meanR_lag{lag:03d}'],
                    't_vs_0':     row[f'T1_t_lag{lag:03d}'],
                    'p_unc':      row[f'T1_p_lag{lag:03d}'],
                    'p_fdr':      row.get(f'T1_p_lag{lag:03d}_fdr', np.nan),
                    'k_perm_sig': int(row[f'T3_k_lag{lag:03d}']),
                    'p_binom':    row[f'T3_p_lag{lag:03d}'],
                    'p_binom_fdr':row.get(f'T3_p_lag{lag:03d}_fdr', np.nan),
                    'n_cells':    int(row['n_cells']),
                })
    pd.DataFrame(long_rows).to_csv(
        os.path.join(out_dir, 'roi_lag_table.csv'), index=False)

    fisher_long_rows = []
    for _, row in fisher_roi_stats.iterrows():
        for lag in LAGS_DEG:
            n = int(row[f'T1_n_lag{lag:03d}'])
            fisher_long_rows.append({
                'roi': row['roi'], 'ctrl_mode': 'noctrl', 'lag_deg': lag,
                'mean_r': row[f'T1_meanR_lag{lag:03d}'],
                'mean_z': row[f'T1_meanZ_lag{lag:03d}'],
                't_vs_0': row[f'T1_t_lag{lag:03d}'],
                'p_unc': row[f'T1_p_lag{lag:03d}'],
                'p_fdr': row.get(f'T1_p_lag{lag:03d}_fdr', np.nan),
                'n_cells': n, 'df': n - 1 if n else np.nan,
                'test': 'two-sided one-sample t-test of Fisher z vs 0',
            })
    pd.DataFrame(fisher_long_rows).to_csv(
        os.path.join(out_dir, 'roi_lag_table_fisher_z_noctrl.csv'), index=False)

    for ctrl in (False, True):
        tag = 'ctrl' if ctrl else 'noctrl'
        fig_roi_lag_heatmap(roi_stats, ctrl,
            os.path.join(fig_dir, f'01_roi_lag_heatmap_{tag}'))
        fig_roi_lag_curves(per_cell, ctrl,
            os.path.join(fig_dir, f'02_roi_lag_curves_{tag}'))
        fig_perm_sig_fraction_bar(roi_stats, ctrl,
            os.path.join(fig_dir, f'03_perm_sig_fraction_bar_{tag}'))
        fig_perm_sig_fraction_heatmap(roi_stats, ctrl,
            os.path.join(fig_dir, f'03b_perm_sig_fraction_heatmap_{tag}'))
        fig_dsrfull_vs_dsrinf_scatter(per_cell, ctrl,
            os.path.join(fig_dir, f'04_dsrfull_vs_dsrinf_scatter_{tag}'))
        fig_test2_target_vs_others_lines(per_cell, roi_stats, ctrl,
            os.path.join(fig_dir, f'07_test2_target_vs_others_lines_{tag}'))
        fig_per_roi_r_hist(per_cell, ctrl,
            os.path.join(fig_dir, f'06_per_roi_r_hist_{tag}'))
        # Per-fixed-lag histograms across all ROIs (one figure per lag).
        for fl in SINGLE_LAGS_FOR_TESTS:
            fig_per_lag_r_hist_all_rois(per_cell, single_lag_df, ctrl, fl,
                os.path.join(fig_dir,
                              f'08_per_lag_r_hist_all_rois_lag{fl:03d}_{tag}'))
        # Descriptive: across-cell Pearson r between per-cell CV r at
        # each pair of lags — one heatmap per ROI, plus a long-format CSV.
        corr_df = lag_lag_correlation(per_cell, ctrl_mode=ctrl)
        corr_df.to_csv(
            os.path.join(out_dir, f'lag_lag_correlation_{tag}.csv'),
            index=False,
        )
        fig_lag_lag_correlation_heatmap(per_cell, ctrl,
            os.path.join(fig_dir, f'09_lag_lag_correlation_{tag}'))

    # Figures that visualize correlation values or t-tests are recreated on
    # the Fisher-z scale for the no-control sensitivity analysis. Permutation
    # figures are unchanged because they do not use the parametric t-tests.
    fig_roi_lag_heatmap(fisher_roi_stats, False,
        os.path.join(fig_dir, '01_roi_lag_heatmap_noctrl_fisher_z'), fisher_z=True)
    fig_roi_lag_curves(per_cell, False,
        os.path.join(fig_dir, '02_roi_lag_curves_noctrl_fisher_z'), fisher_z=True)
    fig_dsrfull_vs_dsrinf_scatter(per_cell, False,
        os.path.join(fig_dir, '04_dsrfull_vs_dsrinf_scatter_noctrl_fisher_z'),
        fisher_z=True)
    fig_test2_target_vs_others_lines(per_cell, fisher_roi_stats, False,
        os.path.join(fig_dir, '07_test2_target_vs_others_lines_noctrl_fisher_z'),
        fisher_z=True)
    fig_per_roi_r_hist(per_cell, False,
        os.path.join(fig_dir, '06_per_roi_r_hist_noctrl_fisher_z'), fisher_z=True)
    for fl in SINGLE_LAGS_FOR_TESTS:
        fig_per_lag_r_hist_all_rois(
            per_cell, fisher_single_lag_df, False, fl,
            os.path.join(fig_dir,
                         f'08_per_lag_r_hist_all_rois_lag{fl:03d}_noctrl_fisher_z'),
            fisher_z=True)
    fig_ctrl_vs_noctrl_scatter(per_cell,
        os.path.join(fig_dir, '05_ctrl_vs_noctrl_scatter'))

    return roi_stats


def _subset_cell_set(per_cell, cell_set=None):
    """Restrict a cached per-cell table to `CELL_SET`.

    The reload path never touches `cs.load_cells`, so the cohort rule is
    applied here instead — on the SESSION, using the same RSA session
    list `cs.load_cells` uses, so the two paths cannot drift apart. This
    is what lets a cohort (e.g. the cells that could not enter the
    population-RSA pseudopopulation) be re-tested from cached CV r and
    cached permutation p, with no CV and no permutations recomputed.
    """
    cell_set = CELL_SET if cell_set is None else cell_set
    if cell_set == 'all_in_roi_table':
        return per_cell
    if cell_set not in ('rsa', 'not_in_rsa'):
        raise ValueError(f'Unknown CELL_SET {cell_set!r}')
    rsa_subjects = {int(s) for s in cs.load_rsa_subjects()}
    in_rsa = per_cell['subject_id'].astype(int).isin(rsa_subjects)
    keep = in_rsa if cell_set == 'rsa' else ~in_rsa
    out = per_cell[keep].copy()
    print(f'  cell_set={cell_set} — {len(out)}/{len(per_cell)} cells kept '
          f'from {out["subject_id"].nunique()} sessions')
    return out


def _load_reload_per_cell(reload_dir):
    """Read per_cell_ALL_ROIs.csv from a previous run directory."""
    csv = os.path.join(reload_dir, 'per_cell_ALL_ROIs.csv')
    if not os.path.exists(csv):
        raise FileNotFoundError(f'per_cell_ALL_ROIs.csv not found at {csv}')
    df = pd.read_csv(csv)
    print(f'  reload — loaded {len(df)} cells from {csv}')
    return df


def main():
    np.random.seed(RANDOM_SEED)
    run_tag = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if RELOAD_FROM is not None:
        run_tag += f'_reload_from_{os.path.basename(os.path.normpath(RELOAD_FROM))}'
    elif CELL_SET != 'all_in_roi_table':
        run_tag += f'_{CELL_SET}'
    out_dir = os.path.join(OUT_BASE, run_tag)
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    print(f'Output dir: {out_dir}')

    # ---- Reload branch: skip CV + perms, just recompute stats + plots ----
    if RELOAD_FROM is not None:
        if RELABEL_FROM is not None:
            run_tag += '_relabelled'
        if CELL_SET != 'all_in_roi_table':
            run_tag += f'_{CELL_SET}'
        if RELABEL_FROM is not None or CELL_SET != 'all_in_roi_table':
            out_dir = os.path.join(OUT_BASE, run_tag)
            fig_dir = os.path.join(out_dir, 'figures')
            os.makedirs(fig_dir, exist_ok=True)

        with open(os.path.join(out_dir, 'config.json'), 'w') as f:
            json.dump({'reload_from': RELOAD_FROM,
                       'relabel_from': RELABEL_FROM,
                       'cell_set': CELL_SET,
                       'n_permutations': N_PERMUTATIONS,
                       'fisher_z_sensitivity': {
                           'enabled': True,
                           'control_mode': 'noctrl',
                           'transform_order': 'mean CV r across folds, then arctanh',
                           't_test': 'two-sided',
                           'clip_epsilon': FISHER_Z_CLIP_EPS,
                       },
                       'roi_predicted_lags': {k: list(v) for k, v in
                                                ROI_PREDICTED_LAGS_DEG.items()},
                       'roi_display_names': ROI_DISPLAY_NAMES,
                       'lags_deg': LAGS_DEG}, f, indent=2)
        per_cell = _load_reload_per_cell(RELOAD_FROM)

        if RELABEL_FROM is not None:
            from mc.analyse.roi_relabel import relabel_per_cell
            # `neuron` format is "SS_CC-CC-..."; the number after '_' is cell_idx.
            per_cell = per_cell.copy()
            per_cell['cell_idx'] = (per_cell['neuron']
                                     .str.split('_').str[1]
                                     .str.split('-').str[0].astype(int))
            per_cell, _audit = relabel_per_cell(
                per_cell, roi_table_csv=RELABEL_FROM,
                roi_col_in_table='alt_final_roi',
                subject_key_per_cell='subject_id',
                cell_key_per_cell='cell_idx')

        per_cell = _canonicalize_roi_names(per_cell)
        per_cell = _subset_cell_set(per_cell)
        per_cell.to_csv(os.path.join(out_dir, 'per_cell_ALL_ROIs.csv'), index=False)
        _stats_and_plots(per_cell, out_dir, fig_dir)
        # Descriptive model-regressor correlation across subjects/configs.
        # Reloads only the per-subject location data (not cells).
        try:
            cells = _load_cells()
            long_reg = compute_lag_regressor_correlations(cells, out_dir)
            fig_lag_regressor_correlation(long_reg, out_dir)
        except Exception as exc:
            print(f'  lag-regressor correlation skipped: {exc}')
        print(f'\nDone (reload). Outputs in {out_dir}')
        return

    config = {
        'run_tag':              run_tag,
        'method':               'lag_shifted_rate_map_weighted_pearson',
        'rois_to_run':          ROIS_TO_RUN,
        'cell_set':             CELL_SET,
        'phase_residualise':    PHASE_RESIDUALISE,
        'trials':               TRIALS,
        'map_estimator':        MAP_ESTIMATOR,
        'lags_deg':             LAGS_DEG,
        'dsr_full_lags_deg':    DSR_FULL_LAGS_DEG,
        'dsr_inf_lags_deg':     DSR_INF_LAGS_DEG,
        'lag_bins_bttn_next':   LAG_BINS_BTTN_NEXT,
        'min_dwell_bins':       MIN_DWELL_BINS,
        'min_shared_locs':      MIN_SHARED_LOCS,
        'weighted_correlation': WEIGHTED_CORRELATION,
        'n_permutations':       N_PERMUTATIONS,
        'random_seed':          RANDOM_SEED,
        'alpha':                ALPHA,
        'fisher_z_sensitivity': {
            'enabled': True,
            'control_mode': 'noctrl',
            'transform_order': 'mean CV r across folds, then arctanh',
            't_test': 'two-sided',
            'clip_epsilon': FISHER_Z_CLIP_EPS,
        },
        'roi_predicted_lags':   {k: list(v) for k, v in
                                   ROI_PREDICTED_LAGS_DEG.items()},
        'controls':             sorted(CONTROL_MODELS),
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    cells = _load_cells()
    all_dfs = []
    for roi in ROIS_TO_RUN:
        df = run_roi(roi, cells)
        if df is None:
            continue
        df.to_csv(os.path.join(out_dir, f'per_cell_{roi}.csv'), index=False)
        all_dfs.append(df)
    if not all_dfs:
        print('No results.'); return
    per_cell = pd.concat(all_dfs, ignore_index=True)
    per_cell.to_csv(os.path.join(out_dir, 'per_cell_ALL_ROIs.csv'), index=False)

    _stats_and_plots(per_cell, out_dir, fig_dir)

    # Descriptive model-regressor correlation across subjects/configs.
    try:
        long_reg = compute_lag_regressor_correlations(cells, out_dir)
        fig_lag_regressor_correlation(long_reg, out_dir)
    except Exception as exc:
        print(f'  lag-regressor correlation skipped: {exc}')

    print(f'\nDone. Outputs in {out_dir}')


if __name__ == '__main__':
    main()
