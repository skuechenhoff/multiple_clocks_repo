#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replot per_lag_encoding results from a previous run, with cell exclusion.

Reloads `per_cell_ALL_ROIs.csv` from a previous per_lag_encoding run,
filters cells based on exclusion / include-only criteria, re-runs the
per-ROI stats, and re-renders ALL six figures into a subdirectory
`replot_<EXCL_NAME>/figures/` within the same run directory. No heavy
re-computation — the per-cell rate-map fits and permutation nulls are
NOT re-run; we just re-aggregate and re-plot.

EXCLUSION SPECIFICATION (three layered options, applied in order):
    EXCL_NEURONS_LIST : explicit list of neuron IDs (overrides EXCL_CSV)
    EXCL_CSV          : path to a CSV with neuron IDs + a criterion column
                         (e.g. encoding_state_sustained_cv/<run>/results.csv
                          with column `sig_r_state` to drop state-tuned cells)
    EXCL_CRITERION_COL
    EXCL_CRITERION_VAL: drop cells where EXCL_CSV[EXCL_CRITERION_COL] == VAL
    EXCL_NEURON_COL   : column in EXCL_CSV holding the neuron ID
                         (defaults to 'neuron')
    EXCL_MODE         : 'exclude' (drop flagged cells) or 'include_only'
                         (keep ONLY flagged cells, drop the rest — useful
                          for the "cells used in DSR RSA" / "cells NOT used
                          in DSR RSA" complementary analyses)

USE CASES:
    1. Drop state-tuned cells: point EXCL_CSV at
       group/encoding_state_sustained_cv/<run>/results.csv, set
       EXCL_CRITERION_COL='sig_r_state' (or 'sig_sustained'), EXCL_VAL=True.
    2. Drop cells already in DSR RSA: build a CSV of the RSA cells'
       neuron IDs (from group/DSR_RSA_simple_ROI/<run>/results_summary.csv
       or roi_electrode_coords.csv + parse_neuron_label), set
       EXCL_MODE='exclude'.
    3. Keep ONLY cells NOT in DSR RSA (complementary set, parallel to the
       independent-cell replication in your manuscript): same as #2 but
       EXCL_MODE='include_only' with the COMPLEMENT CSV, OR pass the RSA
       cell list and set EXCL_MODE='exclude'.

OUTPUTS under <reload_run_dir>/replot_<EXCL_NAME>/:
    per_cell_ALL_ROIs_filtered.csv
    per_roi_stats_filtered.csv
    replot_manifest.json
    figures/01_..._{ctrl,noctrl}.{pdf,png}  through  figures/06_...

@author: Svenja Küchenhoff
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from mc.analyse.roi_labels import read_per_cell_table

REPO = '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))

# Importing the analysis module is safe (heavy work is behind __main__).
from per_lag_encoding import (
    per_roi_stats,
    fig_roi_lag_heatmap,
    fig_roi_lag_curves,
    fig_perm_sig_fraction_bar,
    fig_perm_sig_fraction_heatmap,
    fig_dsrfull_vs_dsrinf_scatter,
    fig_ctrl_vs_noctrl_scatter,
    fig_per_roi_r_hist,
    OUT_BASE,
)


# ── Settings ──────────────────────────────────────────────────────────
RELOAD_RUN = '2026-08-28_10-18-21_reload_from_2026-06-30_18-21-57_relabelled'  # final labels

# EXCL_NAME also dispatches to a named-mode auto-builder when it matches
# one of these reserved names; otherwise we fall back to the manual
# EXCL_NEURONS_LIST / EXCL_CSV settings below.
#   'no_rsa_cells'   → drop every cell listed in RSA_CELLS_SOURCE_CSV
#                       (matches by parsed (subject, cell_idx)).
#   'no_state_tuned' → drop every cell flagged by the perm test in
#                       STATE_TUNED_SOURCE_CSV. Match by 'neuron'.
EXCL_NAME = 'no_rsa_cells'   # 'no_state_tuned' / 'no_rsa_cells' / custom
EXCL_MODE = 'exclude'          # 'exclude' or 'include_only'

# ── Named-mode sources ───────────────────────────────────────────────
_DATA = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'

STATE_TUNED_SOURCE_CSV    = (f'{_DATA}/group/encoding_state_sustained_cv/'
                              '2026-09-24_18-35-23/state_sustained_cv_results.csv')   # final labels
STATE_TUNED_CRITERION_COL = 'sig_sustained'   # perm-sig sustained-state cell
STATE_TUNED_CRITERION_VAL = True               # use 'sig_sustained_fdr' for FDR-strict

RSA_CELLS_SOURCE_CSV      = (f'{_DATA}/group/DSR_RSA_simple_ROI/'
                              '2026-08-31_17-57-30/roi_electrode_coords.csv')   # Fig 2d RSA run, final labels

# ── Manual fallback (when EXCL_NAME is NOT a reserved name) ──────────
EXCL_NEURONS_LIST: list[str] = []
EXCL_CSV           = ''
EXCL_CRITERION_COL = 'sig_r_state'
EXCL_CRITERION_VAL = True
EXCL_NEURON_COL    = 'neuron'


def _parse_neuron_label(label):
    """'01_07-07-chan120-EC' → (subject_int=1, cell_idx=7)."""
    try:
        sub_str, rest = label.split('_', 1)
        cell_idx_str = rest.split('-', 1)[0]
        return int(sub_str), int(cell_idx_str)
    except (ValueError, IndexError, AttributeError):
        return None, None


# ── Helpers ───────────────────────────────────────────────────────────
def _build_flagged_set(per_cell_df):
    """Return the set of neuron IDs flagged for exclusion.

    Dispatches on EXCL_NAME (same reserved names as spatial_peaks_replot).
    """
    # Named-mode dispatchers --------------------------------------------
    if EXCL_NAME == 'no_rsa_cells':
        if not os.path.exists(RSA_CELLS_SOURCE_CSV):
            raise FileNotFoundError(
                f'RSA cells source CSV not found: {RSA_CELLS_SOURCE_CSV}')
        rsa = pd.read_csv(RSA_CELLS_SOURCE_CSV)
        for c in ('subject', 'cell_idx'):
            if c not in rsa.columns:
                raise ValueError(
                    f'{RSA_CELLS_SOURCE_CSV} missing column {c!r}')
        rsa_pairs = set(zip(rsa['subject'].astype(int),
                             rsa['cell_idx'].astype(int)))
        flagged = set()
        for nid in per_cell_df['neuron'].astype(str):
            sub, ci = _parse_neuron_label(nid)
            if sub is not None and (sub, ci) in rsa_pairs:
                flagged.add(nid)
        print(f"[no_rsa_cells] {len(rsa_pairs)} RSA pairs → "
              f"{len(flagged)} matching neurons in this run")
        return flagged

    if EXCL_NAME == 'no_state_tuned':
        if not os.path.exists(STATE_TUNED_SOURCE_CSV):
            raise FileNotFoundError(
                f'State CSV not found: {STATE_TUNED_SOURCE_CSV}')
        ext = pd.read_csv(STATE_TUNED_SOURCE_CSV)
        if STATE_TUNED_CRITERION_COL not in ext.columns:
            raise ValueError(
                f'{STATE_TUNED_SOURCE_CSV} missing column '
                f'{STATE_TUNED_CRITERION_COL!r}; has {list(ext.columns)}')
        if 'neuron' not in ext.columns:
            raise ValueError(
                f'{STATE_TUNED_SOURCE_CSV} missing column "neuron"')
        flagged = set(ext.loc[
            ext[STATE_TUNED_CRITERION_COL] == STATE_TUNED_CRITERION_VAL,
            'neuron',
        ].astype(str).unique().tolist())
        print(f"[no_state_tuned] {len(flagged)} neurons flagged where "
              f"{STATE_TUNED_CRITERION_COL}={STATE_TUNED_CRITERION_VAL}")
        return flagged

    # Manual fallback ---------------------------------------------------
    if EXCL_NEURONS_LIST:
        return set(EXCL_NEURONS_LIST)
    if not EXCL_CSV or not os.path.exists(EXCL_CSV):
        print(f"[warn] EXCL_NAME={EXCL_NAME!r} is not a reserved mode and "
              f"EXCL_CSV is unset/missing ({EXCL_CSV!r}); no cells "
              f"will be filtered.")
        return set()
    ext = pd.read_csv(EXCL_CSV)
    if EXCL_NEURON_COL not in ext.columns:
        raise ValueError(f"EXCL_CSV missing column {EXCL_NEURON_COL!r}; "
                         f"has {list(ext.columns)}")
    if EXCL_CRITERION_COL not in ext.columns:
        raise ValueError(f"EXCL_CSV missing column {EXCL_CRITERION_COL!r}; "
                         f"has {list(ext.columns)}")
    flagged = ext.loc[ext[EXCL_CRITERION_COL] == EXCL_CRITERION_VAL,
                       EXCL_NEURON_COL].astype(str).unique().tolist()
    return set(flagged)


def _apply_mask(per_cell_df, flagged):
    """Return filtered per_cell_df according to EXCL_MODE."""
    ids = per_cell_df['neuron'].astype(str)
    in_set = ids.isin(flagged).to_numpy()
    if EXCL_MODE == 'exclude':
        keep = ~in_set
    elif EXCL_MODE == 'include_only':
        keep = in_set
    else:
        raise ValueError(f"EXCL_MODE must be 'exclude' or 'include_only', "
                         f"got {EXCL_MODE!r}")
    return per_cell_df[keep].reset_index(drop=True), keep


# ── Main ──────────────────────────────────────────────────────────────
def main():
    run_dir = os.path.join(OUT_BASE, RELOAD_RUN)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    csv_path = os.path.join(run_dir, 'per_cell_ALL_ROIs.csv')
    print(f"Loading {csv_path}")
    per_cell = read_per_cell_table(csv_path)
    print(f"  {len(per_cell)} cells across {per_cell.roi.nunique()} ROIs")

    flagged = _build_flagged_set(per_cell)
    print(f"  {len(flagged)} flagged neuron IDs from exclusion source")
    per_cell_f, keep_mask = _apply_mask(per_cell, flagged)

    n_in, n_kept = int(len(per_cell)), int(len(per_cell_f))
    n_drop = n_in - n_kept
    print(f"After filter (mode={EXCL_MODE!r}): "
          f"{n_kept} kept, {n_drop} dropped (of {n_in})")
    print("  per-ROI cell counts:")
    for roi, n in per_cell_f.groupby('roi').size().items():
        in_n = int((per_cell['roi'] == roi).sum())
        print(f"    {roi:<18s} {n:>4d}  (was {in_n})")

    if n_kept < 5:
        print("\nFewer than 5 cells kept overall — aborting replot.")
        return

    out_dir = os.path.join(run_dir, f'replot_{EXCL_NAME}')
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    per_cell_f.to_csv(
        os.path.join(out_dir, 'per_cell_ALL_ROIs_filtered.csv'),
        index=False,
    )
    manifest = {
        'reload_run':       RELOAD_RUN,
        'replot_timestamp': datetime.now().isoformat(timespec='seconds'),
        'excl_name':        EXCL_NAME,
        'excl_csv':         None if EXCL_NEURONS_LIST else EXCL_CSV,
        'excl_criterion_col': None if EXCL_NEURONS_LIST else EXCL_CRITERION_COL,
        'excl_criterion_val': None if EXCL_NEURONS_LIST else EXCL_CRITERION_VAL,
        'excl_neuron_col':  EXCL_NEURON_COL,
        'excl_list_size':   len(EXCL_NEURONS_LIST),
        'excl_mode':        EXCL_MODE,
        'n_flagged':        len(flagged),
        'n_in':             n_in,
        'n_kept':           n_kept,
        'n_dropped':        n_drop,
        'kept_per_roi':     {k: int(v) for k, v in
                              per_cell_f.groupby('roi').size().items()},
    }
    with open(os.path.join(out_dir, 'replot_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    # Re-run stats and re-plot.
    stats_dfs = [per_roi_stats(per_cell_f, ctrl_mode=False),
                 per_roi_stats(per_cell_f, ctrl_mode=True)]
    roi_stats = pd.concat(stats_dfs, ignore_index=True)
    roi_stats.to_csv(os.path.join(out_dir, 'per_roi_stats_filtered.csv'),
                      index=False)

    for ctrl in (False, True):
        tag = 'ctrl' if ctrl else 'noctrl'
        fig_roi_lag_heatmap(roi_stats, ctrl,
            os.path.join(fig_dir, f'01_roi_lag_heatmap_{tag}'))
        fig_roi_lag_curves(per_cell_f, ctrl,
            os.path.join(fig_dir, f'02_roi_lag_curves_{tag}'))
        fig_perm_sig_fraction_bar(roi_stats, ctrl,
            os.path.join(fig_dir, f'03_perm_sig_fraction_bar_{tag}'))
        fig_perm_sig_fraction_heatmap(roi_stats, ctrl,
            os.path.join(fig_dir, f'03b_perm_sig_fraction_heatmap_{tag}'))
        fig_dsrfull_vs_dsrinf_scatter(per_cell_f, ctrl,
            os.path.join(fig_dir, f'04_dsrfull_vs_dsrinf_scatter_{tag}'))
        fig_per_roi_r_hist(per_cell_f, ctrl,
            os.path.join(fig_dir, f'06_per_roi_r_hist_{tag}'))
    fig_ctrl_vs_noctrl_scatter(per_cell_f,
        os.path.join(fig_dir, '05_ctrl_vs_noctrl_scatter'))

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == '__main__':
    main()
