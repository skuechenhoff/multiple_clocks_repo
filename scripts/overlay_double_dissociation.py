#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overlay ROI lag-curves onto one axis to show the mPFC-vs-HC double
dissociation: mPFC peaks at future lags (30-60°), HC_anterior/HC_mid at
0° (now). It writes both cell-weighted and subject-balanced versions:
(a) mean CV r ± SEM per lag, (b) one-sample t of Fisher-z CV r > 0 per
lag, and (c) a side-by-side weighting comparison. Both wrap lag 0 back at
the end (13 x-positions:
0, 30, ..., 330, 0) so the circular structure is obvious.

For Spyder, select the result family at the top of the file with
`CALL_RESULTS_FROM = 'per_lag'` or `'spatial_peaks'`. The per-lag encoding
results use `per_cell_ALL_ROIs.csv` and its `r_lagXXX_{noctrl,ctrl}` columns;
the spatial-peaks results use `per_cell.csv` and its
`per_lag_r_all_lags_json` column.

Subject-balanced curves first average cells within subject and then give every
subject equal weight. Never displays "ACC" — remapped to "mPFC".
"""
from __future__ import annotations
import argparse, json, os, sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mc.analyse.roi_labels import read_per_cell_table

sys.path.insert(0, "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo")
from mc.plotting.cell_results import (
    SHOWGIRL2_DISCRETE,
)


# ---- Spyder call settings --------------------------------------------
# Change this value when running the file with Spyder's "Run file" button.
CALL_RESULTS_FROM = 'per_lag'  # 'spatial_peaks'
CALL_PER_LAG_CTRL_MODE = 'noctrl'  # 'ctrl' is also available
CALL_WEIGHTING = 'both'  # 'both', 'cell', or 'subject'

ROIS_TO_OVERLAY = ['mPFC', 'HC_mid']
STATS_ROIS = ['mPFC', 'HC_anterior', 'HC_mid']

# Every t test in this script is a one-sample t on Fisher-z transformed CV r.
# The mean r printed next to a t is the *same* effect expressed in r units, so
# the reader always knows which r/z, p and q belong to which t.
TEST_DEFINITIONS = {
    "lagwise": (
        "One-sample t of the tested value > 0 at each of the 12 lags, run "
        "for every combination of unit_of_analysis (cell = one observation "
        "per cell; subject = cells averaged within subject first) and metric "
        "(raw_r = untransformed CV r; fisher_z = Fisher-z CV r). "
        "mean_tested_value is what the t was computed on; mean_raw_r, "
        "mean_fisher_z and r_from_fisher_z = tanh(mean_fisher_z) are the "
        "same effect in every unit, so no t is ever ambiguous."),
    "target_window": (
        "Pre-defined window test. target_mean_vs_zero: mean Fisher-z over "
        "the predicted lags vs 0. target_vs_other_lags: mean Fisher-z over "
        "the predicted lags minus the mean over the other 10 lags, vs 0 "
        "(this is the contrast the Fig 3c star refers to)."),
    "prespecified_lag": (
        "The same per-lag test, but only at the one lag fixed a priori per "
        "ROI (see PRESPECIFIED_LAG_DEG). Because no search was performed, "
        "the family is just the 3 ROIs."),
    "peak_lag_permutation": (
        "Every lag's observed t against a sign-flip permutation null of the "
        "MAX t across lags. Use this when the reported lag was chosen by "
        "looking at the curve; the FWE p already pays for that search."),
}

# Which correction is which. Each entry names the family of tests that were
# corrected together, so a q or FWE p can never be read out of context.
# In the lag-wise table every correction is computed INSIDE one
# unit_of_analysis x metric block, never across blocks: cell/subject and
# raw_r/fisher_z are four versions of the same test, not extra tests.
CORRECTION_FAMILIES = {
    "q_one_sided_across_12_lags": (
        "BH across the 12 lags of ONE ROI. 12 tests."),
    "q_one_sided_across_rois": (
        "BH across the 3 stats ROIs at ONE lag (lag-wise table), or across "
        "the 3 ROIs for one window / pre-specified-lag test. 3 tests."),
    "q_one_sided_across_rois_and_lags": (
        "BH across all 3 stats ROIs x 12 lags at once. 36 tests. "
        "The most conservative scope."),
    "p_fwe_maxt_within_roi_12_lags": (
        "Sign-flip permutation FWE across the 12 lags of one ROI: p is the "
        "fraction of permutations whose MAX t over lags beats the observed "
        "t. Correct scope for a lag chosen by looking at the curve, and less "
        "conservative than BH because it keeps the lag-lag correlation."),
    "p_fwe_maxt_across_rois_and_lags": (
        "Same permutation, but the null is the max t over all 3 ROIs x 12 "
        "lags. Scope for 'the strongest lag anywhere in the figure'."),
}

STAR_THRESHOLDS = ((0.001, "***"), (0.01, "**"), (0.05, "*"))

# The two analysis choices that get crossed in the lag-wise table:
# unit of analysis (cell vs subject) x metric (raw r vs Fisher z).
METRICS = {"raw_r": False, "fisher_z": True}   # name -> Fisher-transform flag

# ---- Single-lag reporting --------------------------------------------
# Reporting one lag instead of the two-lag window needs the multiple-
# comparisons scope to match how the lag was chosen.
#   (a) lag fixed a priori  -> correct across the 3 ROIs only (3 tests).
#   (b) lag read off the curve -> must pay for the search over 12 lags. BH
#       over 12 lags assumes near-independent tests, which is wrong here
#       (neighbouring lags correlate at r ~ .22-.33), so a sign-flip
#       permutation builds the null of the max t across lags instead. That
#       is FWE-correct and less conservative than BH, because it inherits
#       the real lag-lag correlation.
# Changing PRESPECIFIED_LAG_DEG is a scientific decision that has to be made
# before looking at the curve; the permutation route exists for when it wasn't.
PRESPECIFIED_LAG_DEG = {'mPFC': 30, 'HC_anterior': 0, 'HC_mid': 0}
N_PERM = 10000
PERM_SEED = 42


def _stars(q):
    """Significance marker for a q value (n.s. if nothing survives)."""
    if q is None or not np.isfinite(q):
        return "n.s."
    for cutoff, mark in STAR_THRESHOLDS:
        if q < cutoff:
            return mark
    return "n.s."

ROI_COLOURS = {
    'EC':              SHOWGIRL2_DISCRETE[0],
    'mPFC':            SHOWGIRL2_DISCRETE[1],
    'HC_anterior':     '#a30d6c',
    'PCC':             SHOWGIRL2_DISCRETE[3],
    'mOFC':             SHOWGIRL2_DISCRETE[4],
    'PHC':             '#23677E',
    'HC_mid':          SHOWGIRL2_DISCRETE[2],
}

# Normalize labels written by older analyses to the names used in the plot.
ROI_NAME_MAP = {
    'ACC': 'mPFC',
    'mPFC': 'mPFC',
    'medialOFC': 'mOFC',
    'mOFC': 'mOFC',
    'Parahippocampal': 'PHC',
    'Parahippocampus': 'PHC',
    'PHC': 'PHC',
}


#2026-07-01_10-35-40_reload_from_2026-06-26_18-47-11_phase_resid_paired_fixedlag-final
# ---- I/O defaults ----------------------------------------------------
# DEFAULT_PER_CELL_CSV = ("/Users/xpsy1114/Documents/projects/multiple_clocks/"
#                         "data/ephys_humans/derivatives/group/"
#                         "spatial_peaks_simple/"
#                         "2026-07-23_12-28-10_reload_from_2026-07-01_07-29-33_"
#                         "reload_from_replot_no_rsa_cells_relabelled/"
#                         "per_cell.csv")
# >>> RERUN-CHECK: hardcoded upstream run dir -- update after re-running spatial_peaks_simple.py
DEFAULT_PER_CELL_CSV = ("/Users/xpsy1114/Documents/projects/multiple_clocks/"
                        "data/ephys_humans/derivatives/group/"
                        "spatial_peaks_simple/"
                        "2026-08-28_10-23-56_phase_resid_paired_fixedlag/"
                        "per_cell.csv")
# >>> RERUN-CHECK: hardcoded upstream run dir -- update after re-running per_lag_encoding.py
# 2026-11-23: per-repeat rate maps (SK's choice), final ROI labels, correct
# (execute) loops = scripts/per_lag_encoding_loopsets.py. The previous source
# (mode estimator) was per_lag_encoding/2026-08-28_10-18-21_reload_from_
# 2026-06-30_18-21-57_relabelled/per_cell_ALL_ROIs.csv.
DEFAULT_PER_LAG_CSV = ("/Users/xpsy1114/Documents/projects/multiple_clocks/"
                       "data/ephys_humans/derivatives/group/"
                       "per_lag_loopsets_2026-11-23/per_repeat/execute/"
                       "per_cell_ALL_ROIs.csv")


LAGS_DEG_BASE = list(range(0, 360, 30))    # 12 lags used in spatial_peaks



# Publication sizing (see CLAUDE.md — subpanel ~7 cm at Arial 9-11 pt)
CM = 1 / 2.54
FONT_TICK, FONT_AXIS, FONT_BIG = 9, 10, 11
DPI = 300


def _display(roi):
    return roi


def _roi_colour(roi):
    """Return the requested colour for a canonical ROI name."""
    return ROI_COLOURS.get(roi, "#666")

def _load_curve_data(per_cell_csv, source="spatial_peaks", ctrl_mode="noctrl",
                     rois=ROIS_TO_OVERLAY):
    """Return per-cell curves and subject IDs for each ROI.

    The two analysis pipelines write the same conceptual result in different
    table formats, so the source-specific parsing is kept here rather than
    making callers know about either format.
    """
    df = (read_per_cell_table(per_cell_csv) if source == "per_lag"
          else pd.read_csv(per_cell_csv))
    if "roi" not in df:
        raise ValueError(f"{per_cell_csv} has no 'roi' column")
    df["roi"] = df["roi"].map(lambda roi: ROI_NAME_MAP.get(roi, roi))

    if source == "spatial_peaks":
        def curve_from_row(row):
            try:
                c = json.loads(row.get("per_lag_r_all_lags_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                c = []
            if len(c) != len(LAGS_DEG_BASE):
                return None
            return [np.nan if v is None else float(v) for v in c]
    elif source == "per_lag":
        columns = [f"r_lag{lag:03d}_{ctrl_mode}" for lag in LAGS_DEG_BASE]
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"{per_cell_csv} is missing per-lag columns: {', '.join(missing)}"
            )

        def curve_from_row(row):
            values = pd.to_numeric(row[columns], errors="coerce")
            return values.to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown source {source!r}")

    subject_col = "subject_id"
    if subject_col not in df:
        raise ValueError(f"{per_cell_csv} has no {subject_col!r} column")
    out = {}
    for roi in rois:
        g = df[df["roi"] == roi]
        curves, subjects = [], []
        for _, row in g.iterrows():
            c = curve_from_row(row)
            if c is not None:
                curves.append(c)
                subject = str(row[subject_col]).replace("sub-", "")
                if subject.endswith(".0"):
                    subject = subject[:-2]
                subjects.append(subject.zfill(2))
        out[roi] = {
            "curves": (np.asarray(curves, float) if curves else
                       np.empty((0, len(LAGS_DEG_BASE)))),
            "subjects": np.asarray(subjects, str),
        }
    return out


def _analysis_units(record, weighting="cell", fisher=False):
    """Return cell curves or equal-weight subject-mean curves."""
    curves = record["curves"]
    if fisher:
        curves = np.arctanh(np.clip(curves, -0.9999999, 0.9999999))
    if weighting == "cell":
        return curves
    if weighting != "subject":
        raise ValueError(f"Unknown weighting {weighting!r}")
    subjects = record["subjects"]
    return np.stack([
        np.nanmean(curves[subjects == subject], axis=0)
        for subject in np.unique(subjects)
    ])


def _bh_fdr(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    good = np.isfinite(p)
    if not good.any():
        return out
    pg = p[good]
    order = np.argsort(pg)
    ranked = pg[order]
    q = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(q)
    restored[order] = np.minimum(q, 1)
    out[good] = restored
    return out


def _wrap_lag(a):
    """Append the first column at the end to close the circle."""
    if a.ndim == 1:
        return np.concatenate([a, a[:1]])
    return np.concatenate([a, a[:, :1]], axis=1)


def _tstat_gt0(values):
    """One-sample one-sided t (> 0) at each lag column, and its p."""
    from scipy.stats import ttest_1samp
    n_lags = values.shape[1]
    t = np.full(n_lags, np.nan)
    p_one = np.full(n_lags, np.nan)
    p_two = np.full(n_lags, np.nan)
    for j in range(n_lags):
        x = values[:, j]; x = x[np.isfinite(x)]
        if x.size >= 2:
            res = ttest_1samp(x, 0.0, alternative="greater")
            t[j] = float(res.statistic)
            p_one[j] = float(res.pvalue)
            p_two[j] = float(ttest_1samp(x, 0.0).pvalue)
    return t, p_one, p_two


def _lagwise_tests(records, rois, weightings):
    """One row per unit-of-analysis x metric x ROI x lag, with every FDR scope.

    The same one-sample t (> 0) is run four ways -- cells or subjects as the
    unit of analysis, crossed with raw CV r or Fisher-z CV r as the tested
    quantity -- so the consequences of those two analysis choices can be read
    off one table. Every BH correction is computed inside one
    unit_of_analysis x metric block, because correcting across blocks would
    mix four versions of the same test.
    """
    rows = []
    for weighting in weightings:
        for metric, use_fisher in METRICS.items():
            for roi in rois:
                record = records[roi]
                if record["curves"].size == 0:
                    continue
                tested = _analysis_units(record, weighting=weighting,
                                         fisher=use_fisher)
                raw = _analysis_units(record, weighting=weighting, fisher=False)
                z = _analysis_units(record, weighting=weighting, fisher=True)
                t, p_one, _ = _tstat_gt0(tested)
                for j, lag in enumerate(LAGS_DEG_BASE):
                    n_valid = int(np.isfinite(tested[:, j]).sum())
                    rows.append({
                        "unit_of_analysis": weighting,
                        "metric": metric,
                        "roi": _display(roi),
                        "lag_deg": lag,
                        "n_units": n_valid,
                        "n_cells": record["curves"].shape[0],
                        "n_subjects": np.unique(record["subjects"]).size,
                        "df": n_valid - 1,
                        "mean_tested_value": float(np.nanmean(tested[:, j])),
                        "mean_raw_r": float(np.nanmean(raw[:, j])),
                        "mean_fisher_z": float(np.nanmean(z[:, j])),
                        "r_from_fisher_z": float(np.tanh(np.nanmean(z[:, j]))),
                        "t": t[j],
                        "p_one_sided": p_one[j],
                    })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    block = ["unit_of_analysis", "metric"]
    out["q_one_sided_across_12_lags"] = (
        out.groupby(block + ["roi"])["p_one_sided"].transform(_bh_fdr))
    out["q_one_sided_across_rois"] = (
        out.groupby(block + ["lag_deg"])["p_one_sided"].transform(_bh_fdr))
    out["q_one_sided_across_rois_and_lags"] = (
        out.groupby(block)["p_one_sided"].transform(_bh_fdr))
    return out


def _unit_label(record, roi, weighting):
    n_cells = record["curves"].shape[0]
    n_subjects = np.unique(record["subjects"]).size
    if weighting == "subject":
        return f"{_display(roi)} ({n_subjects} subjects; {n_cells} cells)"
    return f"{_display(roi)} ({n_cells} cells)"


def _draw_mean_overlay(ax, records, rois, weighting):
    x_wrap = np.asarray(LAGS_DEG_BASE + [360])
    for roi in rois:
        record = records[roi]
        C = _analysis_units(record, weighting=weighting, fisher=False)
        if C.size == 0:
            continue
        mean = np.nanmean(C, axis=0)
        sem = (np.nanstd(C, axis=0, ddof=1) /
               np.sqrt(np.maximum(np.isfinite(C).sum(axis=0), 1)))
        mean_w, sem_w = _wrap_lag(mean), _wrap_lag(sem)
        colour = _roi_colour(roi)
        ax.fill_between(x_wrap, mean_w - sem_w, mean_w + sem_w,
                        color=colour, alpha=0.18, linewidth=0)
        ax.plot(x_wrap, mean_w, "-o", color=colour, lw=1.5, ms=3,
                label=_unit_label(record, roi, weighting))
    ax.axhline(0, color="black", lw=0.5, ls="--")


def _draw_t_overlay(ax, records, rois, weighting):
    x_wrap = np.asarray(LAGS_DEG_BASE + [360])
    for roi in rois:
        record = records[roi]
        Z = _analysis_units(record, weighting=weighting, fisher=True)
        if Z.size == 0:
            continue
        t, _, _ = _tstat_gt0(Z)
        ax.plot(x_wrap, _wrap_lag(t), "-o", color=_roi_colour(roi),
                lw=1.5, ms=3, label=_unit_label(record, roi, weighting))
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.axhline(1.65, color="grey", lw=0.5, ls=":",
               label="approx. one-sided p = .05")


def _target_tests(records, rois, weightings):
    """Tests of each ROI's pre-defined lag window on Fisher-z units."""
    from scipy.stats import ttest_1samp
    targets = {"mPFC": (30, 60), "HC_mid": (0, 330),
               "HC_anterior": (0, 330)}
    rows = []
    for weighting in weightings:
        for roi in rois:
            if roi not in targets or records[roi]["curves"].size == 0:
                continue
            Z = _analysis_units(records[roi], weighting=weighting, fisher=True)
            target_idx = [LAGS_DEG_BASE.index(lag) for lag in targets[roi]]
            other_idx = [i for i in range(len(LAGS_DEG_BASE))
                         if i not in target_idx]
            target = np.nanmean(Z[:, target_idx], axis=1)
            contrast = target - np.nanmean(Z[:, other_idx], axis=1)
            for test, values in (("target_mean_vs_zero", target),
                                 ("target_vs_other_lags", contrast)):
                values = values[np.isfinite(values)]
                one = ttest_1samp(values, 0, alternative="greater")
                two = ttest_1samp(values, 0)
                rows.append({
                    "weighting": weighting, "roi": roi,
                    "target_lags_deg": "+".join(map(str, targets[roi])),
                    "test": test, "n_units": len(values),
                    "n_cells": records[roi]["curves"].shape[0],
                    "n_subjects": np.unique(records[roi]["subjects"]).size,
                    "mean_fisher_z_or_difference": float(np.mean(values)),
                    "r_from_fisher_z": float(np.tanh(np.mean(values))),
                    "t_fisher_z": float(one.statistic),
                    "df": len(values) - 1,
                    "p_one_sided": float(one.pvalue),
                    "p_two_sided": float(two.pvalue),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_one_sided_across_rois"] = (
            out.groupby(["weighting", "test"])["p_one_sided"]
            .transform(_bh_fdr))
        out["q_two_sided_across_rois"] = (
            out.groupby(["weighting", "test"])["p_two_sided"]
            .transform(_bh_fdr))
    return out


def _tstat_gt0_vectorised(values):
    """One-sample t vs 0 per lag; takes (n, lags) or (n_perm, n, lags)."""
    n = values.shape[-2]
    sd = values.std(axis=-2, ddof=1)
    return values.mean(axis=-2) / (sd / np.sqrt(n))


def _check_vectorised_t(values):
    """Permutations must use the empirical estimator -- verify, don't assume."""
    reference, _, _ = _tstat_gt0(values)
    if not np.allclose(reference, _tstat_gt0_vectorised(values),
                       equal_nan=True):
        raise AssertionError(
            "vectorised permutation t does not reproduce _tstat_gt0")


def _prespecified_lag_tests(records, rois, weightings):
    """Single lag per ROI, fixed a priori, corrected across the ROIs only."""
    from scipy.stats import ttest_1samp
    rows = []
    for weighting in weightings:
        for roi in rois:
            if (roi not in PRESPECIFIED_LAG_DEG
                    or records[roi]["curves"].size == 0):
                continue
            lag = PRESPECIFIED_LAG_DEG[roi]
            j = LAGS_DEG_BASE.index(lag)
            raw = _analysis_units(records[roi], weighting=weighting,
                                  fisher=False)[:, j]
            z = _analysis_units(records[roi], weighting=weighting,
                                fisher=True)[:, j]
            z = z[np.isfinite(z)]
            one = ttest_1samp(z, 0, alternative="greater")
            rows.append({
                "weighting": weighting, "roi": roi, "lag_deg": lag,
                "lag_choice": "pre-specified",
                "n_units": len(z),
                "n_cells": records[roi]["curves"].shape[0],
                "n_subjects": np.unique(records[roi]["subjects"]).size,
                "mean_raw_r": float(np.nanmean(raw)),
                "mean_fisher_z": float(np.mean(z)),
                "r_from_fisher_z": float(np.tanh(np.mean(z))),
                "t_fisher_z_vs_0": float(one.statistic),
                "df": len(z) - 1,
                "p_one_sided": float(one.pvalue),
                "p_two_sided": float(ttest_1samp(z, 0).pvalue),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_one_sided_across_rois"] = (
            out.groupby("weighting")["p_one_sided"].transform(_bh_fdr))
        out["q_two_sided_across_rois"] = (
            out.groupby("weighting")["p_two_sided"].transform(_bh_fdr))
    return out


def _peak_lag_permutation_tests(records, rois, weightings,
                                n_perm=N_PERM, seed=PERM_SEED):
    """FWE p per lag from the null of the max t across lags.

    Flipping the sign of a whole unit curve is the standard exchangeability
    for a one-sample test and leaves that unit's lag-lag correlation intact,
    so the max-t null already pays for the search across the 12 lags -- and,
    in the second column, across the three ROIs as well. Subjects that
    contribute cells to more than one ROI get the same sign flip in every
    ROI, so the across-ROI null does not pretend they are independent.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for weighting, (metric, use_fisher) in product(weightings, METRICS.items()):
        matrices, observed = {}, {}
        for roi in rois:
            if records[roi]["curves"].size == 0:
                continue
            z = _analysis_units(records[roi], weighting=weighting,
                                fisher=use_fisher)
            _check_vectorised_t(z)
            matrices[roi] = z
            observed[roi] = _tstat_gt0_vectorised(z)
        if not matrices:
            continue

        shared_index = None
        if weighting == "subject":
            unit_subjects = {roi: np.unique(records[roi]["subjects"])
                             for roi in matrices}
            all_subjects = np.unique(np.concatenate(list(unit_subjects.values())))
            shared_index = {roi: np.searchsorted(all_subjects, unit_subjects[roi])
                            for roi in matrices}

        null_within = {roi: [] for roi in matrices}
        null_across, done = [], 0
        while done < n_perm:
            size = min(500, n_perm - done)
            draws = (rng.choice([-1.0, 1.0], size=(size, all_subjects.size))
                     if weighting == "subject" else None)
            block_max = []
            for roi, z in matrices.items():
                signs = (draws[:, shared_index[roi]][:, :, None]
                         if weighting == "subject"
                         else rng.choice([-1.0, 1.0],
                                         size=(size, z.shape[0], 1)))
                t_null = _tstat_gt0_vectorised(z[None, :, :] * signs)
                null_within[roi].append(t_null.max(axis=1))
                block_max.append(t_null.max(axis=1))
            null_across.append(np.max(np.stack(block_max), axis=0))
            done += size
        null_across = np.concatenate(null_across)

        for roi, z in matrices.items():
            within = np.concatenate(null_within[roi])
            peak_lag = LAGS_DEG_BASE[int(np.argmax(observed[roi]))]
            for j, lag in enumerate(LAGS_DEG_BASE):
                t_obs = observed[roi][j]
                rows.append({
                    "unit_of_analysis": weighting, "metric": metric,
                    "roi": roi, "lag_deg": lag,
                    "n_units": z.shape[0],
                    "mean_tested_value": float(z[:, j].mean()),
                    "t": float(t_obs),
                    "df": z.shape[0] - 1,
                    "is_observed_peak_lag": lag == peak_lag,
                    "p_fwe_maxt_within_roi_12_lags": float(
                        (1 + np.sum(within >= t_obs)) / (n_perm + 1)),
                    "p_fwe_maxt_across_rois_and_lags": float(
                        (1 + np.sum(null_across >= t_obs)) / (n_perm + 1)),
                })
    return pd.DataFrame(rows)


def _json_safe(value):
    """numpy scalars and NaN are not valid JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value]
    return value


def _json_rows(table):
    """One dict per test, with p and every FDR q nested under the same row.

    Keeping p and all q variants inside the row (instead of as sibling
    columns) is what makes the file readable: whatever t you look at, its
    effect size, its p and each of its q values sit next to it.
    """
    rows = []
    for record in table.to_dict("records"):
        row, p_block, q_block = {}, {}, {}
        for key, value in record.items():
            value = _json_safe(value)
            if key.startswith("p_"):
                p_block[key[2:]] = value
            elif key.startswith("q_"):
                q_block[key[2:]] = value
            else:
                row[key] = value
        row["p"] = p_block
        row["fdr_q"] = q_block
        rows.append(row)
    return rows


def _md_table(table, columns, floatfmt=4):
    """Markdown table for the named columns, numbers right-aligned."""
    header = f"| {' | '.join(columns)} |"
    rule = f"| {' | '.join(['---:'] * len(columns))} |"
    lines = [header, rule]
    for record in table.to_dict("records"):
        cells = []
        for column in columns:
            value = record[column]
            if isinstance(value, (float, np.floating)):
                cells.append("n/a" if not np.isfinite(value)
                             else f"{value:.{floatfmt}f}")
            else:
                cells.append(str(value))
        lines.append(f"| {' | '.join(cells)} |")
    return lines


# Lag colour bar under the x axis: the circular wheel of the fMRI angle maps
# (same anchors as harmonic_maps_brain_overlay.CIRCULAR_ANCHORS_HEX):
# 0 deg yellow, 90 red, 180 blue, 270 green.
LAG_WHEEL_HEX = ['#FCE300', '#E53935', '#1E88E5', '#43A047', '#FCE300']
PANEL_LABELS = {"HC_mid": "mid HC", "mPFC": "mPFC",
                "HC_anterior": "anterior HC"}


def publication_panel(records, target, out_dir, rois=("HC_mid", "mPFC"),
                      weighting="subject"):
    """Fig 3c: mean CV r +- SEM per lag (session-balanced by default).

    Stars = each ROI's pre-defined window test (window mean minus the other
    lags, Fisher z, one-sided; `target_vs_other_lags`), one per ROI, drawn
    above the window lag with the highest mean (repeated at the wrapped 0 deg
    when that lag is 0 deg) -- the contrast the Fig 3c star refers to. The
    panel only draws `records` and `target`.
    """
    from matplotlib.colors import LinearSegmentedColormap
    x = np.asarray(LAGS_DEG_BASE + [360])
    fig = plt.figure(figsize=(7.5 * CM, 5.5 * CM))
    ax = fig.add_axes([0.0, 0.10, 1.0, 0.90])
    cax = fig.add_axes([0.0, 0.055, 1.0, 0.04])
    top = {}
    for roi in rois:
        C = _analysis_units(records[roi], weighting=weighting, fisher=False)
        m = np.nanmean(C, 0)
        se = np.nanstd(C, 0, ddof=1) / np.sqrt(np.isfinite(C).sum(0))
        m, se = _wrap_lag(m), _wrap_lag(se)
        col = _roi_colour(roi)
        ax.fill_between(x, m - se, m + se, color=col, alpha=0.2, lw=0)
        ax.plot(x, m, "-o", color=col, lw=1.6, ms=3.5,
                label=f"{PANEL_LABELS.get(roi, roi)} "
                      f"({np.unique(records[roi]['subjects']).size} sessions, "
                      f"{C.shape[0] if weighting == 'cell' else records[roi]['curves'].shape[0]} cells)")
        top[roi] = (m, se)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    lo, hi = ax.get_ylim()
    span = hi - lo
    for roi in rois:
        row = target[(target.weighting == weighting) & (target.roi == roi)
                     & (target.test == "target_vs_other_lags")]
        if row.empty:
            continue
        star = _stars(float(row.p_one_sided.iloc[0]))
        if star == "n.s.":
            continue
        m, se = top[roi]
        lags = [int(v) for v in row.target_lags_deg.iloc[0].split("+")]
        best = max((LAGS_DEG_BASE.index(l) for l in lags), key=lambda i: m[i])
        idx = [best] + ([len(LAGS_DEG_BASE)] if best == 0 else [])
        for i in idx:
            ax.text(x[i], m[i] + se[i] + 0.03 * span, star, ha="center",
                    va="bottom", fontsize=FONT_BIG, color=_roi_colour(roi))
    ax.set_ylim(lo, hi + 0.12 * span)
    ax.set_xlim(-10, 370)
    ax.set_xticks([])
    ax.set_ylabel("CV r (mean ± SEM)", fontsize=FONT_AXIS)
    ax.tick_params(labelsize=FONT_TICK, length=2, pad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    cmap = LinearSegmentedColormap.from_list("lag_wheel", LAG_WHEEL_HEX)
    cax.imshow(np.linspace(0, 1, 361)[None, :], aspect="auto", cmap=cmap,
               extent=(0, 360, 0, 1))
    cax.set_xlim(ax.get_xlim())
    cax.set_yticks([])
    for sp in cax.spines.values():
        sp.set_visible(False)
    cax.set_xticks([0, 60, 120, 180, 240, 300, 360])
    cax.set_xticklabels(["0°", "60°", "120°", "180°", "240°", "300°", "0°"],
                        fontsize=FONT_TICK)
    cax.tick_params(length=2, pad=1)
    cax.set_xlabel("Lag into the future", fontsize=FONT_AXIS, labelpad=1)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=1,
              frameon=False, fontsize=FONT_TICK, handlelength=1.5)
    suffix = "" if weighting == "subject" else "_cell_weighted"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"fig3c_lag_profile{suffix}.{ext}", dpi=DPI,
                    bbox_inches="tight")
    plt.close(fig)


def make_overlay(per_cell_csv, out_dir, rois=ROIS_TO_OVERLAY,
                 source="spatial_peaks", ctrl_mode="noctrl",
                 weighting="both"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    loaded_rois = list(dict.fromkeys(list(rois) + STATS_ROIS))
    records = _load_curve_data(per_cell_csv, source=source, ctrl_mode=ctrl_mode,
                               rois=loaded_rois)
    weightings = ["cell", "subject"] if weighting == "both" else [weighting]

    x_wrap = np.asarray(LAGS_DEG_BASE + [360])   # 13 positions, last shown as 360°
    tick_pos = [0, 30, 60, 120, 180, 240, 300, 360]
    tick_lab = ["0°", "30°", "60°", "120°", "180°", "240°", "300°", "0°"]

    # ---------- Separate mean and t-stat figures ------------------------
    for current_weighting in weightings:
        suffix = "" if current_weighting == "cell" else "_subject_weighted"
        unit_title = ("cell-weighted" if current_weighting == "cell"
                      else "subject-balanced")
        fig, ax = plt.subplots(figsize=(6 * CM, 4 * CM), constrained_layout=True)
        _draw_mean_overlay(ax, records, rois, current_weighting)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, fontsize=FONT_TICK, rotation=45)
        ax.set_xlabel("Lag (looking into the future)", fontsize=FONT_AXIS)
        ax.set_ylabel("Mean CV r", fontsize=FONT_AXIS)
        ax.tick_params(axis="both", labelsize=FONT_TICK, length=2, pad=1)
        ax.legend(fontsize=FONT_TICK - 1, frameon=False, loc="upper right")
        ax.set_title(f"mPFC–HC double dissociation ({unit_title})",
                     fontsize=FONT_BIG)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(out_dir / f"overlay_meanR_wrapped{suffix}.{ext}",
                        dpi=DPI, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6 * CM, 4 * CM), constrained_layout=True)
        _draw_t_overlay(ax, records, rois, current_weighting)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, fontsize=FONT_TICK, rotation=45)
        ax.set_xlabel("Lag (looking into the future)", fontsize=FONT_AXIS)
        ax.set_ylabel("t-stat (Fisher-z CV r > 0)", fontsize=FONT_AXIS)
        ax.tick_params(axis="both", labelsize=FONT_TICK, length=2, pad=1)
        ax.legend(fontsize=FONT_TICK - 1, frameon=False, loc="upper right")
        ax.set_title(f"Per-lag Fisher-z tests ({unit_title})", fontsize=FONT_BIG)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(out_dir / f"overlay_tstat_wrapped{suffix}.{ext}",
                        dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    # ---------- Direct publication comparison --------------------------
    if weighting == "both":
        fig, axes = plt.subplots(1, 2, figsize=(18 * CM, 6 * CM),
                                 sharex=True, sharey=True,
                                 constrained_layout=True)
        for ax, current_weighting, title in zip(
                axes, ("cell", "subject"),
                ("Cell-weighted", "Subject-balanced")):
            _draw_mean_overlay(ax, records, rois, current_weighting)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lab, fontsize=FONT_TICK, rotation=45)
            ax.set_xlabel("Lag (looking into the future)", fontsize=FONT_AXIS)
            ax.tick_params(axis="both", labelsize=FONT_TICK, length=2, pad=1)
            ax.set_title(title, fontsize=FONT_BIG)
            ax.legend(fontsize=FONT_TICK - 2, frameon=False, loc="upper right")
        axes[0].set_ylabel("Mean CV r", fontsize=FONT_AXIS)
        fig.suptitle("mPFC–HC double dissociation: effect of population weighting",
                     fontsize=FONT_BIG)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(out_dir / f"overlay_meanR_weighting_comparison.{ext}",
                        dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    # ---------- One consolidated lag-wise results table ------------------
    # cell x subject crossed with raw r x Fisher z, for all three stats ROIs
    # (HC_anterior is included even though it is not drawn in the overlay),
    # with all three FDR scopes side by side.
    lagwise = _lagwise_tests(records, STATS_ROIS, ["cell", "subject"])
    lagwise.to_csv(out_dir / "overlay_lagwise_tests.csv", index=False)

    target = _target_tests(records, STATS_ROIS, weightings)
    target.to_csv(out_dir / "overlay_target_window_tests.csv", index=False)
    for w in weightings:
        publication_panel(records, target, out_dir, weighting=w)

    # Two ways to report a single lag rather than the two-lag window.
    prespecified = _prespecified_lag_tests(records, STATS_ROIS, weightings)
    prespecified.to_csv(
        out_dir / "overlay_prespecified_lag_tests.csv", index=False)
    peak_perm = _peak_lag_permutation_tests(records, STATS_ROIS, weightings)
    peak_perm.to_csv(
        out_dir / "overlay_peak_lag_permutation.csv", index=False)

    # ---------- Settings, machine-readable and human-readable results ----
    settings = {
        "per_cell_csv": str(per_cell_csv), "source": source,
        "control_mode": ctrl_mode, "weighting": weighting,
        "overlay_rois": list(rois), "stats_rois": STATS_ROIS,
        "target_windows_deg": {"mPFC": [30, 60], "HC_anterior": [0, 330],
                               "HC_mid": [0, 330]},
        "prespecified_lag_deg": PRESPECIFIED_LAG_DEG,
        "permutation": {"n_perm": N_PERM, "seed": PERM_SEED,
                        "scheme": "sign-flip of whole unit curves, max t "
                                  "across lags"},
        "lags_deg": LAGS_DEG_BASE,
        "analysis_unit": {
            "cell": "one observation per cell",
            "subject": "cells averaged within subject, then one observation "
                       "per subject",
        },
        "subject_balancing": "mean Fisher z within subject, then t across subjects",
        "visual_subject_balancing": "mean raw r within subject, then mean across subjects",
    }
    (out_dir / "overlay_config.json").write_text(json.dumps(settings, indent=2))

    results = {
        "input": str(per_cell_csv),
        "settings": settings,
        "how_to_read": (
            "Every entry is one t test. mean_fisher_z (or "
            "mean_fisher_z_or_difference) is the quantity the t was computed "
            "on; r_from_fisher_z is that same effect back-transformed to r; "
            "mean_raw_r is the untransformed mean that the figure plots. "
            "'p' holds the uncorrected p values, 'fdr_q' the Benjamini-"
            "Hochberg q values -- one entry per correction family, described "
            "in 'correction_families'."),
        "test_definitions": TEST_DEFINITIONS,
        "correction_families": CORRECTION_FAMILIES,
        "target_window_tests": _json_rows(target),
        "prespecified_lag_tests": _json_rows(prespecified),
        "peak_lag_permutation_tests": _json_rows(peak_perm),
        "lagwise_tests": _json_rows(lagwise),
    }
    (out_dir / "overlay_results.json").write_text(json.dumps(results, indent=2))

    focus = target[target.test == "target_vs_other_lags"].copy()
    focus["t(df)"] = [f"{r.t_fisher_z:.2f}({int(r.df)})"
                      for r in focus.itertuples()]
    focus["star"] = [_stars(q) for q in focus["q_one_sided_across_rois"]]

    report = [
        "# Double dissociation overlay: results", "",
        f"Input: `{per_cell_csv}`", "",
        "All t tests are one-sample t tests on Fisher-z transformed CV r. "
        "`mean_fisher_z` is what the t was computed on, `r_from_fisher_z` is "
        "the same effect back-transformed to r, and `mean_raw_r` is the "
        "untransformed mean drawn in the figure. Cell-weighted rows treat "
        "each cell as an observation; subject-balanced rows average cells "
        "within a subject first, so n = subjects.", "",
        "## Multiple-comparisons families", "",
        "Several corrections are reported side by side; they differ only in "
        "which tests were corrected together. In the lag-wise table every "
        "correction runs inside one unit_of_analysis x metric block -- "
        "cell/subject and raw_r/fisher_z are four versions of the same test, "
        "not extra tests to correct for.", "",
        "| column | family |", "| --- | --- |",
    ]
    report += [f"| `{name}` | {text} |" for name, text in CORRECTION_FAMILIES.items()]
    report += [
        "", f"Stars: `***` q < .001, `**` q < .01, `*` q < .05.", "",
        "## Pre-defined target-window tests (the Fig 3c stars)", "",
        TEST_DEFINITIONS["target_window"], "",
        "Rows below are the `target_vs_other_lags` contrast, one-sided, "
        "FDR-corrected across the three ROIs.", "",
    ]
    report += _md_table(
        focus, ["weighting", "roi", "target_lags_deg", "n_units", "df",
                "mean_fisher_z_or_difference", "r_from_fisher_z",
                "t(df)", "p_one_sided", "q_one_sided_across_rois", "star"],
        floatfmt=5)
    report += ["", "Both window tests (against zero and against the other "
               "lags) are in `overlay_target_window_tests.csv` and in "
               "`overlay_results.json`.", ""]

    # ---- reporting one lag instead of the window ----------------------
    prespecified = prespecified.copy()
    prespecified["star"] = [_stars(q)
                            for q in prespecified["q_one_sided_across_rois"]]
    report += [
        "## Reporting a single lag instead of the window", "",
        "Which correction applies depends entirely on how the lag was "
        "chosen, so both routes are given.", "",
        "### (a) Lag fixed a priori", "",
        TEST_DEFINITIONS["prespecified_lag"], "",
        "Current pre-specified lags: "
        + ", ".join(f"{roi} = {lag}°"
                    for roi, lag in PRESPECIFIED_LAG_DEG.items())
        + ". This only holds if the choice really was made before looking.",
        "",
    ]
    report += _md_table(
        prespecified,
        ["weighting", "roi", "lag_deg", "n_units", "df", "mean_raw_r",
         "mean_fisher_z", "r_from_fisher_z", "t_fisher_z_vs_0",
         "p_one_sided", "q_one_sided_across_rois", "star"], floatfmt=5)

    peak_rows = peak_perm[peak_perm.is_observed_peak_lag].copy()
    peak_rows["star"] = [_stars(p) for p in
                         peak_rows["p_fwe_maxt_within_roi_12_lags"]]
    report += [
        "", "### (b) Lag read off the curve", "",
        TEST_DEFINITIONS["peak_lag_permutation"],
        f"{N_PERM} sign-flip permutations, seed {PERM_SEED}. Rows below are "
        "each ROI's observed peak lag; all 12 lags are in "
        "`overlay_peak_lag_permutation.csv`.", "",
    ]
    report += _md_table(
        peak_rows,
        ["unit_of_analysis", "metric", "roi", "lag_deg", "n_units", "df",
         "mean_tested_value", "t", "p_fwe_maxt_within_roi_12_lags",
         "p_fwe_maxt_across_rois_and_lags", "star"], floatfmt=5)
    report.append("")

    report += ["## Lag-wise tests: all four analysis choices", "",
               TEST_DEFINITIONS["lagwise"], "",
               "One block per unit_of_analysis x metric. Everything is in "
               "`overlay_lagwise_tests.csv`; the blocks below are only that "
               "file split up for reading.", ""]
    lag_columns = ["roi", "lag_deg", "n_units", "df", "mean_tested_value",
                   "mean_raw_r", "mean_fisher_z", "r_from_fisher_z", "t",
                   "p_one_sided", "q_one_sided_across_12_lags",
                   "q_one_sided_across_rois",
                   "q_one_sided_across_rois_and_lags"]
    for unit in ("cell", "subject"):
        for metric in METRICS:
            block = lagwise[(lagwise.unit_of_analysis == unit)
                            & (lagwise.metric == metric)]
            if block.empty:
                continue
            unit_text = ("one observation per cell" if unit == "cell"
                         else "cells averaged within subject first")
            metric_text = ("untransformed CV r" if metric == "raw_r"
                           else "Fisher-z CV r")
            report += [f"### {unit} x {metric}", "",
                       f"t on {metric_text}, {unit_text}.", ""]
            report += _md_table(block, lag_columns, floatfmt=4)
            report.append("")

    report += ["", "## Files", "",
               "- `overlay_results.json`: every test above, with its effect "
               "size, p and all q values in one record.",
               "- `overlay_config.json`: settings this run used.",
               "- `overlay_target_window_tests.csv`: predicted-window tests.",
               "- `overlay_prespecified_lag_tests.csv`: single a-priori lag "
               "per ROI, BH across the 3 ROIs.",
               "- `overlay_peak_lag_permutation.csv`: every lag with its "
               "sign-flip max-t FWE p, for reporting a peak lag.",
               "- `overlay_lagwise_tests.csv`: THE lag-wise table -- cell "
               "and subject units crossed with raw r and Fisher z, for all 3 "
               "stats ROIs x 12 lags, with all three FDR scopes.",
               "- `overlay_meanR_wrapped_subject_weighted.pdf`: standalone "
               "subject-balanced publication overlay.",
               "- `overlay_meanR_weighting_comparison.pdf`: matched cell- "
               "versus subject-weighted comparison.", ""]
    (out_dir / "RESULTS.md").write_text("\n".join(report) + "\n")
    for name in ("WEIGHTING_RESULTS.md", "overlay_per_lag_table.csv",
                 "overlay_subject_clustered_lagwise_ttests.csv"):
        stale = out_dir / name
        if stale.exists():
            stale.unlink()
    print("Wrote figures, RESULTS.md and overlay_results.json into", out_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=("spatial_peaks", "per_lag"),
                   default=None,
                   help="Optional override; otherwise CALL_RESULTS_FROM is used.")
    p.add_argument("--per-cell-csv", default=None,
                   help="Explicit per-cell CSV; overrides the source default.")
    p.add_argument("--ctrl-mode", choices=("noctrl", "ctrl"), default=None,
                   help="Optional override; otherwise CALL_PER_LAG_CTRL_MODE is used.")
    p.add_argument("--weighting", choices=("both", "cell", "subject"),
                   default=None,
                   help="Write cell-weighted, subject-balanced, or both figures.")
    p.add_argument("--out-dir", default=None,
                    help="Default: <per-cell dir>/overlay_double_dissociation/")
    args = p.parse_args()
    source = args.source or CALL_RESULTS_FROM
    ctrl_mode = args.ctrl_mode or CALL_PER_LAG_CTRL_MODE
    weighting = args.weighting or CALL_WEIGHTING
    if source not in ("spatial_peaks", "per_lag"):
        p.error("CALL_RESULTS_FROM must be 'per_lag' or 'spatial_peaks'")
    if args.per_cell_csv:
        csv = Path(args.per_cell_csv)
    elif source == "spatial_peaks":
        csv = Path(DEFAULT_PER_CELL_CSV)
    else:
        csv = Path(DEFAULT_PER_LAG_CSV)
    if not csv.is_file():
        p.error(f"Per-cell CSV does not exist: {csv}")
    default_out_name = "overlay_double_dissociation"
    if source == "per_lag":
        default_out_name += f"_{ctrl_mode}"
    out_dir = (Path(args.out_dir) if args.out_dir
                else csv.parent / default_out_name)
    make_overlay(csv, out_dir, source=source, ctrl_mode=ctrl_mode,
                 weighting=weighting)


if __name__ == "__main__":
    main()
