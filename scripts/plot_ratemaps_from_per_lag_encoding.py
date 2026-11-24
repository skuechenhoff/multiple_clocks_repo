#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rate-map example plots derived from the LATEST `per_lag_encoding.py` run.

Same style as `spatial_peaks_simple`'s `rate_map_examples/` PDFs (a row of
per-configuration 3x3 rate maps + a dwell-weighted pooled mean), but:

  * cells are SELECTED from the per_lag_encoding per-cell CSV based on their
    cross-validated per-lag consistency (`r_lag###_noctrl`);
  * rate maps are BUILT exactly the way per_lag_encoding builds them
    (per configuration, phase-residualised firing rate, `_lag_shifted_rate_map`),
    not the paired-grid-group way of future_spatial_peaks.

Two selections (both from the noctrl per-lag CV r):
  * mPFC   — high r at 30 OR 60 deg, low r at 0 deg  (future action lookahead)
  * HC_mid — high r at 0 deg,        low r at 60 deg (current location)

For each cell we plot a two-row figure:
  row 1 = the cell's TARGET lag (mPFC: whichever of 30/60 is larger; HC: 0)
  row 2 = the cell's CONTRAST lag (mPFC: 0; HC: 60)
Columns = one 3x3 rate map per task configuration + a dwell-weighted mean.

@author: Svenja Kuchenhoff (script: Claude)
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from mc.analyse.roi_labels import read_per_cell_table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
import mc.analyse.helpers_human_cells as hh
import mc.analyse.cell_selection as cs
from mc.analyse.future_spatial_peaks import _residualise_phase

# Reuse per_lag_encoding's own machinery so the maps are identical to what
# the CV statistic was computed on.
from per_lag_encoding import (          # noqa: E402
    DATA_DIR, TRIALS, PHASE_RESIDUALISE, N_LOC, N_BINS, MIN_DWELL_BINS,
    _build_per_cfg_sequences, _lag_shifted_rate_map,
)

# ── Settings ──────────────────────────────────────────────────────────
RUN_DIR = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data/'
           'ephys_humans/derivatives/group/per_lag_encoding/'
           '2026-08-28_10-18-21_reload_from_2026-06-30_18-21-57_relabelled')
PER_CELL_CSV = os.path.join(RUN_DIR, 'per_cell_ALL_ROIs.csv')

N_CELLS_PER_SELECTION = 10
HI_THRESH = 0.15     # target lag r must exceed this
LO_THRESH = 0.10     # contrast lag r must be below this
# Coverage = mean number of grid locations (of 9) with dwell >= MIN_DWELL_BINS
# per configuration. Candidates (passing the CV-r gap thresholds) are ranked by
# a CONTINUOUS combined score = z(gap) + COVERAGE_WEIGHT * z(coverage); no hard
# coverage cutoff, so the trade-off between effect size and spatial coverage is
# a smooth moderator rather than a post-hoc gate. Increase COVERAGE_WEIGHT to
# push selection further toward well-covered cells; set to 0.0 to rank on the
# CV-r gap alone (recovers the original gap-only selection).
COVERAGE_WEIGHT = 0.0

# Which selections to run: (roi, target-lag options, contrast lag).
SELECTIONS = [
    dict(name='mPFC_high30or60_low0', roi='mPFC',
         target_lags=[30, 60], contrast_lag=0),
    dict(name='HCmid_high0_low60', roi='HC_mid',
         target_lags=[0], contrast_lag=60),
]

DPI = 300
CM = 1.0 / 2.54
FONT = 8
CMAP = 'coolwarm'
# Colour-scale window as percentiles of the finite rate values. A TIGHTER
# window (e.g. 20/80) spreads the colormap across the central range so the
# middle tones are easy to differentiate; the extreme low/high tails saturate
# to dark blue/red. Widen toward 5/95 for a less-clipped, more extreme look.
CLIP_PCT_LO = 10
CLIP_PCT_HI = 90

_STAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUT_DIR = os.path.join(RUN_DIR, f'rate_map_examples_per_lag_encoding_{_STAMP}')


def _set_rc():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': FONT,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


# ── Rate maps for one cell, per configuration, at a set of lags ───────
def _percfg_ratemaps(Y_cfg, loc_cfg, lag_deg):
    """(n_cfg, 9) rate maps and (n_cfg, 9) dwell counts at one lag."""
    n_cfg = Y_cfg.shape[0]
    rate = np.full((n_cfg, N_LOC), np.nan)
    dwell = np.zeros((n_cfg, N_LOC), dtype=int)
    for c in range(n_cfg):
        r, d = _lag_shifted_rate_map(Y_cfg[c], loc_cfg[c], int(lag_deg))
        rate[c] = r
        dwell[c] = d
    return rate, dwell


def _dwell_weighted_mean(rate, dwell):
    """Pooled dwell-weighted mean map across configs (per_lag_encoding's
    predicted map, but pooled over ALL configs for display)."""
    tot = dwell.sum(axis=0)
    with np.errstate(invalid='ignore'):
        m = np.nansum(rate * dwell, axis=0) / np.where(tot > 0, tot, np.nan)
    return m


def _plot_cell(cell_row, Y_cfg, loc_cfg, target_lag, contrast_lag, save_stem):
    """Two rows (target lag, contrast lag); columns = configs + pooled mean."""
    _set_rc()
    n_cfg = Y_cfg.shape[0]
    n_panels = n_cfg + 1
    lags = [target_lag, contrast_lag]
    fig, axes = plt.subplots(
        2, n_panels,
        figsize=(2.2 * CM * n_panels, 2 * 2.6 * CM),
        constrained_layout=True, squeeze=False,
    )
    for ri, lag in enumerate(lags):
        rate, dwell = _percfg_ratemaps(Y_cfg, loc_cfg, lag)
        vals = rate[np.isfinite(rate)]
        vmin = float(np.nanpercentile(vals, CLIP_PCT_LO)) if vals.size else 0.0
        vmax = float(np.nanpercentile(vals, CLIP_PCT_HI)) if vals.size else 1.0
        for ci in range(n_cfg):
            ax = axes[ri, ci]
            ax.imshow(rate[ci].reshape(3, 3), cmap=CMAP, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f'cfg {ci}\n(n={int(dwell[ci].sum())})',
                             fontsize=FONT)
        # pooled dwell-weighted mean
        ax = axes[ri, -1]
        im = ax.imshow(_dwell_weighted_mean(rate, dwell).reshape(3, 3),
                       cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title('pooled\nmean', fontsize=FONT)
        r_lag = cell_row.get(f'r_lag{lag:03d}_noctrl', np.nan)
        role = 'TARGET' if lag == target_lag else 'contrast'
        ax.set_ylabel('')
        axes[ri, 0].set_ylabel(f'lag {lag}°  ({role})\nCV r = {r_lag:.2f}',
                               fontsize=FONT, rotation=90, labelpad=2)
        cb = fig.colorbar(im, ax=axes[ri, -1], fraction=0.06, pad=0.03,
                          extend='both')
        cb.ax.tick_params(labelsize=FONT - 1)

    fig.suptitle(
        f"{cell_row['neuron']}   [{cell_row['roi']}]   "
        f"sub-{cell_row['subject_id']}   n_cfg={n_cfg}",
        fontsize=FONT,
    )
    fig.savefig(save_stem + '.pdf', dpi=DPI, bbox_inches='tight')
    fig.savefig(save_stem + '.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# ── Candidate pool from the per-cell CSV (gap only; coverage added later) ─
def _candidate_pool(df, roi, target_lags, contrast_lag):
    """All cells passing the hi/lo CV-r gap thresholds, ranked by gap.
    Coverage is added downstream once raw data is loaded."""
    g = df[df['roi'] == roi].copy()
    hi_cols = [f'r_lag{l:03d}_noctrl' for l in target_lags]
    lo_col = f'r_lag{contrast_lag:03d}_noctrl'
    g['hi'] = g[hi_cols].max(axis=1)
    g['hi_lag'] = g[hi_cols].idxmax(axis=1).str.extract(r'lag(\d+)').astype(int)
    g['lo'] = g[lo_col]
    g = g.dropna(subset=['hi', 'lo'])
    g = g[(g['hi'] > HI_THRESH) & (g['lo'] < LO_THRESH) & (g['n_cfg'] >= 6)]
    g['gap'] = g['hi'] - g['lo']
    return g.sort_values('gap', ascending=False)


def _coverage(loc_cfg):
    """Mean number of grid locations (of 9) with dwell >= MIN_DWELL_BINS per
    configuration. Dwell per location is lag-independent, so compute at lag 0."""
    per_cfg = []
    for c in range(loc_cfg.shape[0]):
        _, d = _lag_shifted_rate_map(np.ones(N_BINS), loc_cfg[c], 0)
        per_cfg.append(int(np.sum(d >= MIN_DWELL_BINS)))
    return float(np.mean(per_cfg)) if per_cfg else 0.0


def _rank_with_coverage(rows):
    """`rows` = list of dicts with 'gap' and 'coverage'. Rank by a continuous
    combined score z(gap) + COVERAGE_WEIGHT * z(coverage) — no hard cutoff, so
    both high CV-r gap and high spatial coverage are rewarded smoothly."""
    df = pd.DataFrame(rows).copy()

    def _z(x):
        x = np.asarray(x, float)
        s = x.std()
        return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)
    df['score'] = _z(df['gap']) + COVERAGE_WEIGHT * _z(df['coverage'])
    return df.sort_values('score', ascending=False).to_dict('records')


# ── Per-cell data assembly (mirrors per_lag_encoding.run_roi) ─────────
def _subject_sequences(sub_str):
    """Return per-subject (idx_cfg, locs, n_cfg, normalised_neurons dict)."""
    data_raw = hh.load_norm_data(DATA_DIR, [sub_str], res_data=False)
    if not data_raw:
        return None
    data = hh.filter_data(data_raw, int(sub_str), TRIALS)
    sub_dict = data[f'sub-{sub_str}']
    beh = sub_dict['beh'].copy().reset_index(drop=True)
    locs = sub_dict['locations'].to_numpy(dtype=float)
    _, _, idx_cfg, _ = np.unique(
        beh[['loc_A', 'loc_B', 'loc_C', 'loc_D']].to_numpy(),
        axis=0, return_index=True, return_inverse=True, return_counts=True,
    )
    n_cfg = len(np.unique(idx_cfg))
    return dict(idx_cfg=idx_cfg, locs=locs, n_cfg=n_cfg,
                neurons=sub_dict['normalised_neurons'])


def _build_cell_maps(seq, nid):
    """Phase-residualised per-config (Y_cfg, loc_cfg) for one cell, or None."""
    if nid not in seq['neurons']:
        return None
    arr = seq['neurons'][nid].to_numpy(dtype=float)
    if PHASE_RESIDUALISE:
        arr = _residualise_phase(arr, basis=PHASE_RESIDUALISE)
    btns_dummy = np.zeros_like(seq['locs'])
    Y_cfg, loc_cfg, _ = _build_per_cfg_sequences(
        arr, seq['idx_cfg'], seq['locs'], btns_dummy, seq['n_cfg'],
    )
    return Y_cfg, loc_cfg


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = read_per_cell_table(PER_CELL_CSV)
    df['subject_id'] = df['subject_id'].astype(str).str.zfill(2)

    # 1) Candidate pool per selection (gap thresholds only). Collect the
    #    union of subjects so we load each subject's raw data exactly once.
    pools = {sel['name']: _candidate_pool(df, sel['roi'], sel['target_lags'],
                                          sel['contrast_lag'])
             for sel in SELECTIONS}
    sel_by_name = {sel['name']: sel for sel in SELECTIONS}

    cand_by_sub = {}   # sub_str -> list of (sel_name, cell_row)
    for name, pool in pools.items():
        for _, row in pool.iterrows():
            cand_by_sub.setdefault(row['subject_id'], []).append((name, row))

    # 2) Load each subject once; build maps + coverage for every candidate.
    #    Cache the maps so the plotting pass needs no reload.
    cand_records = {name: [] for name in pools}   # name -> list of dicts
    map_cache = {}                                # neuron_id -> (Y_cfg, loc_cfg)
    for sub_str, items in sorted(cand_by_sub.items()):
        seq = _subject_sequences(sub_str)
        if seq is None:
            print(f'  sub-{sub_str}: load failed'); continue
        for name, row in items:
            nid = row['neuron']
            maps = map_cache.get(nid) or _build_cell_maps(seq, nid)
            if maps is None:
                print(f'  {nid}: not in normalised_neurons; skip'); continue
            map_cache[nid] = maps
            cov = _coverage(maps[1])
            cand_records[name].append({
                'neuron': nid, 'roi': row['roi'],
                'subject_id': sub_str, 'hi_lag': int(row['hi_lag']),
                'gap': float(row['gap']), 'hi': float(row['hi']),
                'lo': float(row['lo']), 'n_cfg': int(row['n_cfg']),
                'coverage': cov, '_row': row,
            })

    # 3) Rank each selection on gap + coverage, keep top 5, and plot.
    for name, recs in cand_records.items():
        if not recs:
            print(f'\n{name}: no candidates'); continue
        ranked = _rank_with_coverage(recs)[:N_CELLS_PER_SELECTION]
        sel = sel_by_name[name]
        sel_dir = os.path.join(OUT_DIR, name)
        os.makedirs(sel_dir, exist_ok=True)
        print(f"\n{name}: top {len(ranked)} by combined z(gap) + "
              f"{COVERAGE_WEIGHT}*z(coverage)")
        for rec in ranked:
            print(f"   {rec['neuron']:32s} hi_lag={rec['hi_lag']}° "
                  f"r_hi={rec['hi']:.2f} r_lo={rec['lo']:.2f} "
                  f"gap={rec['gap']:.2f} coverage={rec['coverage']:.1f}/9 "
                  f"n_cfg={rec['n_cfg']}")
            Y_cfg, loc_cfg = map_cache[rec['neuron']]
            safe = rec['neuron'].replace('/', '-')
            _plot_cell(rec['_row'], Y_cfg, loc_cfg, rec['hi_lag'],
                       sel['contrast_lag'], os.path.join(sel_dir, safe))
            print(f'     saved {name}/{safe}.pdf')

    print(f'\nDone. Figures in:\n  {OUT_DIR}')


if __name__ == '__main__':
    main()
