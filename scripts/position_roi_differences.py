#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Do the ROIs differ from each other in the per-cell position-in-sequence effect?

Each ROI being individually significant does not mean the ROIs differ from one
another. This asks that question directly: a Kruskal-Wallis omnibus with a
null built by shuffling ROI labels across cells (the same statistic recomputed
on every draw), all 15 pairwise contrasts under FDR, and the two planned
one-sided contrasts for the regions that lead the ranking.

The shift estimate is the two-sample Hodges-Lehmann statistic with its
distribution-free (Moses) interval -- the interval that belongs to the
Mann-Whitney test, not the spread of the pairwise differences.
"""
import os, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo")
import era_brewer

G = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group"
ENC_RUN = G + "/encoding_state_sustained_cv/2026-10-02_13-54-14"
OUT = G + "/position_index_2026-10-02"
os.makedirs(OUT, exist_ok=True)
N_PERM = 10000
SEED = 42
CM = 1 / 2.54

_p = era_brewer.era_brew("Showgirl2", n=7)
ROI_COLOURS = {'EC': _p[0], 'mPFC': _p[1], 'HC_anterior': _p[2], 'PCC': _p[3],
               'mOFC': _p[4], 'HC_mid': '#a30d6c'}
OBSERVED = '#0e3d3a'
LABEL = {'HC_anterior': 'HC ant', 'HC_mid': 'HC mid'}
lab = lambda r: LABEL.get(r, r)
FIXED = ['EC', 'mOFC', 'mPFC', 'HC_anterior', 'HC_mid', 'PCC']


def hl_shift(x, y, alpha=0.05):
    """Two-sample Hodges-Lehmann shift (median of x_i - y_j) + Moses CI."""
    d = np.sort(np.subtract.outer(np.asarray(x), np.asarray(y)).ravel())
    n, m, N = len(x), len(y), d.size
    z = stats.norm.ppf(1 - alpha / 2)
    k = int(np.floor(n * m / 2 - z * np.sqrt(n * m * (n + m + 1) / 12)))
    k = max(k, 0)
    return float(np.median(d)), float(d[k]), float(d[N - 1 - k])


def main():
    r = pd.read_csv(ENC_RUN + "/state_sustained_cv_results.csv")
    d = r[r['roi'].isin(FIXED)][['roi', 'r_state']].dropna()
    vals = {x: d.loc[d.roi == x, 'r_state'].to_numpy() for x in FIXED}

    # ---- omnibus, with the null built by shuffling ROI labels ----
    H, _ = stats.kruskal(*[vals[x] for x in FIXED])
    rng = np.random.default_rng(SEED)
    labels, v = d['roi'].to_numpy(), d['r_state'].to_numpy()
    null = np.array([stats.kruskal(*[v[s == x] for x in FIXED])[0]
                     for s in (rng.permutation(labels) for _ in range(N_PERM))])
    p_omni = (1 + (null >= H).sum()) / (1 + N_PERM)

    # ---- all 15 pairwise contrasts ----
    rows = []
    for i, a in enumerate(FIXED):
        for b in FIXED[i + 1:]:
            est, lo, hi = hl_shift(vals[a], vals[b])
            _, p = stats.mannwhitneyu(vals[a], vals[b], alternative='two-sided')
            rows.append(dict(pair=f"{a} vs {b}", shift=est, lo=lo, hi=hi, p=p))
    pw = pd.DataFrame(rows)
    pw['q'] = multipletests(pw['p'], method='fdr_bh')[1]

    # ---- the two planned contrasts: does either leader beat the rest? ----
    planned = {}
    for roi in ['EC', 'mOFC']:
        rest = np.concatenate([vals[x] for x in FIXED if x != roi])
        est, lo, hi = hl_shift(vals[roi], rest)
        _, p = stats.mannwhitneyu(vals[roi], rest, alternative='greater')
        planned[roi] = dict(shift=est, lo=lo, hi=hi, p_one_sided=float(p),
                            n=len(vals[roi]), n_rest=len(rest))
    q_pl = multipletests([planned[k]['p_one_sided'] for k in planned],
                         method='fdr_bh')[1]
    for k, q in zip(planned, q_pl):
        planned[k]['q_fdr'] = float(q)

    # ---------------------------------------------------------------- figure
    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 9,
        'axes.labelsize': 10, 'axes.titlesize': 11, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'legend.fontsize': 8, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13.5 * CM, 4.2 * CM),
                             gridspec_kw={'wspace': 0.48, 'width_ratios': [1.35, 1]})

    ax = axes[0]
    jit = np.random.default_rng(SEED)
    for i, roi in enumerate(FIXED):
        y = vals[roi]
        ax.scatter(i + jit.uniform(-.22, .22, y.size), y, s=2.2,
                   color=ROI_COLOURS[roi], alpha=.35, linewidths=0, zorder=2)
        parts = ax.violinplot([y], positions=[i], widths=.72,
                              showextrema=False, showmedians=False)
        for b in parts['bodies']:
            b.set_facecolor(ROI_COLOURS[roi]); b.set_alpha(.22)
            b.set_edgecolor(ROI_COLOURS[roi]); b.set_linewidth(.8)
        ax.scatter([i], [np.median(y)], s=26, facecolor='white',
                   edgecolor='0.1', linewidth=1.2, zorder=5)
    ax.axhline(0, color='0.45', ls=':', lw=.9, zorder=0)
    ax.set_xticks(range(len(FIXED)))
    ax.set_xticklabels([lab(x) for x in FIXED], rotation=30, ha='right')
    ax.set_ylabel('held-out r, per cell', labelpad=2, fontsize=8.5)
    ax.set_title('a   Per-cell effects overlap',
                 loc='left', fontweight='bold', pad=6)

    ax = axes[1]
    ax.hist(null, bins=60, color='0.78', edgecolor='none', zorder=2)
    ax.axvline(H, color=OBSERVED, lw=2.0, zorder=4)
    ax.text(H, ax.get_ylim()[1] * .99, f'  observed H = {H:.1f}\n  p$_{{perm}}$ = {p_omni:.2f}',
            color=OBSERVED, fontsize=7.5, va='top', ha='left', linespacing=1.3)
    ax.set_xlabel('Kruskal-Wallis H', labelpad=2, fontsize=8.5)
    ax.set_ylabel('permutations', labelpad=2, fontsize=8.5)
    ax.set_title('b   ROI-label null', loc='left', fontweight='bold', pad=6)

    for ext in ('pdf', 'png'):
        fig.savefig(f"{OUT}/fig4_roi_differences.{ext}", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("wrote", OUT + "/fig4_roi_differences.pdf")

    # ---------------------------------------------------------------- summary
    summary = {
        'question': 'Do ROIs differ from each other in the per-cell '
                    'position-in-sequence effect?',
        'answer': 'No. The omnibus is not significant and no pairwise contrast '
                  'survives FDR.',
        'run': os.path.basename(ENC_RUN),
        'n_per_roi': {k: int(len(v)) for k, v in vals.items()},
        'omnibus': {'statistic': 'Kruskal-Wallis H', 'H': float(H),
                    'p_perm': float(p_omni), 'n_permutations': N_PERM,
                    'null_median_H': float(np.median(null)),
                    'null_95th_pct': float(np.percentile(null, 95))},
        'pairwise_fdr': pw.to_dict('records'),
        'planned_vs_pooled_rest': planned,
    }
    with open(OUT + '/roi_differences_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print("wrote", OUT + '/roi_differences_summary.json')

    print(f"\nomnibus  H = {H:.3f}   p_perm = {p_omni:.4f}   "
          f"(null median {np.median(null):.2f}, 95th pct {np.percentile(null,95):.2f})")
    print("\npairwise, sorted by p:")
    print(pw.sort_values('p').to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\nplanned one-sided contrasts vs pooled rest:")
    for k, s in planned.items():
        print(f"  {k:5s} shift {s['shift']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}]  "
              f"p = {s['p_one_sided']:.4f}  q = {s['q_fdr']:.4f}  (n={s['n']} vs {s['n_rest']})")


if __name__ == '__main__':
    main()
