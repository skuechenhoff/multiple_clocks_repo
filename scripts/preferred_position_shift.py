#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Where does a cell's preferred position go when the ramp is removed?

The marginal shares per ROI show that cells preferring the last position are
depleted, but they cannot show WHERE those cells went, because the two runs
select partly different cells. Matching cells that are significant in both runs
answers that directly, and gives a far stronger version of the claim that the
removed ramp was mostly rising towards the end of the loop.
"""
import os, json, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo")

G = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group"
RAW = G + "/encoding_state_sustained_cv/2026-09-24_18-35-23"
RES = G + "/encoding_state_sustained_cv/2026-10-02_13-54-14"
OUT = G + "/position_index_2026-10-02"
CM = 1 / 2.54
FIXED = ['EC', 'mOFC', 'mPFC', 'HC_anterior', 'HC_mid', 'PCC']
LABEL = {'HC_anterior': 'HC ant', 'HC_mid': 'HC mid'}
lab = lambda r: LABEL.get(r, r)
STATE_COLOURS = {'A': '#F15A29', 'B': '#F7931E', 'C': '#C7C6E2', 'D': '#6B60AA'}


def main():
    raw = pd.read_csv(RAW + "/state_sustained_cv_results.csv")
    res = pd.read_csv(RES + "/state_sustained_cv_results.csv")
    m = (raw[raw.sig_r_state][['neuron', 'roi', 'pref_state_mode']]
         .merge(res[res.sig_r_state][['neuron', 'pref_state_mode']],
                on='neuron', suffixes=('_raw', '_res')))
    ct = (pd.crosstab(m.pref_state_mode_raw, m.pref_state_mode_res)
          .reindex(index=list('ABCD'), columns=list('ABCD')).fillna(0).astype(int))

    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 9,
        'axes.labelsize': 10, 'axes.titlesize': 11, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13.5 * CM, 4.6 * CM),
                             gridspec_kw={'wspace': 0.55, 'width_ratios': [1, 1.12]})

    ax = axes[0]
    M = ct.to_numpy()
    im = ax.imshow(M, cmap='Purples', vmin=0, vmax=M.max())
    for i in range(4):
        for j in range(4):
            ax.text(j, i, M[i, j], ha='center', va='center', fontsize=8.5,
                    color='white' if M[i, j] > M.max() * .55 else '0.15',
                    fontweight='bold' if i != j and M[i, j] >= 5 else 'normal')
    ax.set_xticks(range(4)); ax.set_xticklabels(list('ABCD'))
    ax.set_yticks(range(4)); ax.set_yticklabels(list('ABCD'))
    for k, st in enumerate('ABCD'):
        ax.get_xticklabels()[k].set_color(STATE_COLOURS[st])
        ax.get_yticklabels()[k].set_color(STATE_COLOURS[st])
    ax.set_xlabel('preferred position after', labelpad=2, fontsize=8.5)
    ax.set_ylabel('before', labelpad=2, fontsize=8.5)
    ax.set_title(f'a   Where cells move (n = {len(m)})', loc='left',
                 fontweight='bold', pad=6)
    for s in ax.spines.values():
        s.set_visible(False)

    ax = axes[1]
    rows = []
    for roi in FIXED:
        d = []
        for df in (raw, res):
            s = df[(df.roi == roi) & df.sig_r_state]
            c = s.pref_state_mode.value_counts().reindex(list('ABCD')).fillna(0)
            d.append(100 * c['D'] / c.sum() if c.sum() else np.nan)
        rows.append(dict(roi=roi, before=d[0], after=d[1], n_after=int(
            ((res.roi == roi) & res.sig_r_state).sum())))
    t = pd.DataFrame(rows)
    for i, r in t.iterrows():
        up = r['after'] >= r['before']
        ax.plot([0, 1], [r['before'], r['after']], color='0.6', lw=1.0, zorder=2)
        ax.scatter([0, 1], [r['before'], r['after']],
                   s=[26, 46], color=STATE_COLOURS['D'], zorder=3)
        ax.annotate(f"{lab(r['roi'])}", (1.04, r['after']), fontsize=7.5,
                    va='center', color='0.15' if up else '0.45',
                    fontweight='bold' if up else 'normal')
    ax.set_xticks([0, 1]); ax.set_xlim(-0.25, 1.75)
    ax.set_xticklabels(['no\nresidualisation', 'phase + ramp\nremoved'], fontsize=8)
    ax.set_ylabel('cells preferring\nposition 4 (%)', labelpad=2, fontsize=8.5)
    ax.set_title('b   Only EC is not depleted', loc='left',
                 fontweight='bold', pad=6)

    for ext in ('pdf', 'png'):
        fig.savefig(f"{OUT}/fig6_preferred_position_shift.{ext}", dpi=300,
                    bbox_inches='tight')
    plt.close(fig)
    print("wrote", OUT + "/fig6_preferred_position_shift.pdf")

    sig = res[res.sig_r_state]
    ec = sig[sig.roi == 'EC']; rest = sig[(sig.roi != 'EC') & sig.roi.isin(FIXED)]
    a = int((ec.pref_state_mode == 'D').sum())
    c_ = int((rest.pref_state_mode == 'D').sum())
    odds, p = stats.fisher_exact([[a, len(ec) - a], [c_, len(rest) - c_]],
                                 alternative='greater')
    lo, hi = stats.binomtest(a, len(ec)).proportion_ci(0.95)
    summary = {
        'question': 'Where does preferred position go when the ramp is removed, '
                    'and is EC special in retaining position-4 cells?',
        'paired_cells_significant_in_both_runs': int(len(m)),
        'n_significant_raw': int(raw.sig_r_state.sum()),
        'n_significant_residualised': int(res.sig_r_state.sum()),
        'preference_unchanged': int((m.pref_state_mode_raw == m.pref_state_mode_res).sum()),
        'transition_matrix_before_to_after': ct.to_dict(),
        'per_roi_position4_share': t.to_dict('records'),
        'EC_vs_rest_position4_after': {
            'ec_n': int(len(ec)), 'ec_prefer_D': a,
            'ec_share': float(a / len(ec)), 'ec_ci95': [float(lo), float(hi)],
            'rest_n': int(len(rest)), 'rest_prefer_D': c_,
            'rest_share': float(c_ / len(rest)),
            'fisher_one_sided_p': float(p), 'odds_ratio': float(odds),
            'verdict': 'EC is the only region where position-4 cells are not '
                       'depleted, but with 9 cells this is not statistically '
                       'distinguishable from the other regions.'},
    }
    with open(OUT + '/preferred_position_shift_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("wrote", OUT + '/preferred_position_shift_summary.json')


if __name__ == '__main__':
    main()
