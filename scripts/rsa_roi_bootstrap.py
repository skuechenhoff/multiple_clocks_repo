#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is the mOFC RSA effect bigger than the other ROIs', or is that sampling noise?

Each ROI yields a single RSA beta, fitted on a vector of 4560 RDM entries. The
regression's own standard error is not usable for comparing two ROIs, because
the RDM entries are not independent observations -- each of the 96 conditions
enters ~95 pairs. That is the same reason the main script reports p_perm rather
than p_param.

So the uncertainty is taken over the sampling unit that actually varies: the
neurons. Cells are resampled WITH REPLACEMENT WITHIN each ROI, that ROI's
pseudo-population RDM is rebuilt from scratch, and the combo is refitted. ROIs
are never mixed. This answers "had I recorded a different sample of mOFC cells,
would beta still be .09?", and the difference of two ROIs' bootstrap
distributions answers whether one is reliably larger.

The fitting code is lifted verbatim out of RSA_DSR_ROIs_simple.py by AST
extraction rather than reimplemented, so the bootstrap betas are produced by
exactly the function that produced the published ones. The full-sample fit is
asserted against the published beta before any resampling starts.
"""
import os, sys, ast, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

REPO = "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo"
sys.path.insert(0, REPO)
import mc
import era_brewer

G = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group"
RSA_RUN = G + "/DSR_RSA_simple_ROI/2026-10-02_13-57-52"
MAT_DIR = os.environ.get('RSA_MAT_DIR', '')
OUT = G + "/position_index_2026-10-02"
os.makedirs(OUT, exist_ok=True)

COMBO = ['state', 'location', 'l2_norm', 'dsr_fmri']
TEST = 'split_halves_z'
N_BOOT = int(os.environ.get("RSA_N_BOOT", "1000"))
SEED = 42
N_TASKS = 8
CM = 1 / 2.54

_p = era_brewer.era_brew("Showgirl2", n=7)
ROI_COLOURS = {'EC': _p[0], 'mPFC': _p[1], 'HC_anterior': _p[2], 'PCC': _p[3],
               'mOFC': _p[4], 'HC_mid': '#a30d6c'}
LABEL = {'HC_anterior': 'HC ant', 'HC_mid': 'HC mid'}
lab = lambda r: LABEL.get(r, r)
FIXED = ['mOFC', 'mPFC', 'HC_anterior', 'HC_mid', 'PCC']


def load_rsa_functions():
    """Pull the fitting helpers out of the main script without executing it."""
    src = open(os.path.join(REPO, 'scripts', 'RSA_DSR_ROIs_simple.py')).read()
    tree = ast.parse(src)
    want = {'_scalar', '_degenerate', 'build_combo_rdm', 'evaluate_combo_safe'}
    picked = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in want]
    missing = want - {n.name for n in picked}
    if missing:
        raise RuntimeError(f"could not find in source: {missing}")
    ns = {'np': np, 'mc': mc, 'print': print}
    exec(compile(ast.Module(body=picked, type_ignores=[]), '<rsa>', 'exec'), ns)
    return ns


def beta_for(mat, models, ns):
    """Rebuild the RDM from a (neuron x condition) matrix and fit the combo.

    Mirrors RSA_DSR_ROIs_simple.py exactly: z-scoring is per neuron, so a
    subset of rows is scored the same way the full set would be.
    """
    mu = np.nanmean(mat, axis=1)
    sd = np.nanstd(mat, axis=1)
    mat_z = (mat.T - mu) / sd
    data_rdm = mc.analyse.my_RSA.compute_crosscorr(
        mat_z, plotting=False, include_diagonal=False,
        no_tasks=N_TASKS, model='bootstrap')
    d_vec = np.asarray(data_rdm[0], dtype=float)
    stacked = ns['build_combo_rdm'](models, COMBO)
    t_arr, b_arr, p_arr = ns['evaluate_combo_safe'](stacked, d_vec, COMBO)
    return float(b_arr[COMBO.index('state')])


def main():
    if not MAT_DIR or not os.path.isdir(MAT_DIR):
        sys.exit("set RSA_MAT_DIR to the directory holding mat_<roi>.npz")
    ns = load_rsa_functions()
    published = pd.read_csv(RSA_RUN + "/results_summary_combos.csv")
    published = published[(published.test == TEST) & (published.combo == 'S_P_ctrl')
                          & (published.sub_model == 'state')].set_index('roi')

    rois = [r for r in FIXED if os.path.exists(f"{MAT_DIR}/mat_{r}.npz")]
    print(f"ROIs found: {rois}\n")

    boots, point = {}, {}
    for roi in rois:
        z = np.load(f"{MAT_DIR}/mat_{roi}.npz", allow_pickle=True)
        mat = z['mat']
        mods = np.load(f"{RSA_RUN}/rdms/rdms_{roi}.npz", allow_pickle=True)
        models = {m: (np.asarray(mods[f'model__split_halves__{m}'], float),)
                  for m in COMBO}

        # --- correctness gate: the full sample must reproduce the paper value
        b_full = beta_for(mat, models, ns)
        b_pub = float(published.loc[roi, 'beta'])
        print(f"{roi:12s} n={mat.shape[0]:4d}  rebuilt beta={b_full:+.6f}  "
              f"published={b_pub:+.6f}  diff={abs(b_full-b_pub):.2e}")
        if not np.isclose(b_full, b_pub, atol=1e-6):
            sys.exit(f"  REFUSING TO BOOTSTRAP: {roi} does not reproduce.")
        point[roi] = b_pub

        rng = np.random.default_rng(SEED)
        n = mat.shape[0]
        draws = np.empty(N_BOOT)
        for i in range(N_BOOT):
            draws[i] = beta_for(mat[rng.integers(0, n, n)], models, ns)
        boots[roi] = draws
        # Resampling with replacement duplicates cells, and a pseudo-population
        # carrying duplicates is less informative than one of the same size with
        # distinct cells, so every bootstrap beta is attenuated relative to the
        # real one. The percentile interval therefore sits too low and must not
        # be read as a CI on beta. Use the basic (reverse-percentile) interval,
        # which reflects the bootstrap spread back around the point estimate.
        lo_p, hi_p = np.percentile(draws, [2.5, 97.5])
        lo, hi = 2 * b_pub - hi_p, 2 * b_pub - lo_p
        print(f"{'':12s}   boot median={np.median(draws):+.4f} "
              f"(attenuated by {b_pub-np.median(draws):+.4f}); "
              f"basic 95% CI [{lo:+.4f}, {hi:+.4f}]  sd={draws.std(ddof=1):.4f}")

    # ---- pairwise differences from independent bootstrap draws ----
    rows = []
    for i, a in enumerate(rois):
        for b in rois[i+1:]:
            d_hat = point[a] - point[b]            # observed difference
            d = boots[a] - boots[b]                # bootstrap differences
            # basic bootstrap: reflect the spread around the observed value,
            # which removes the first-order attenuation bias noted above
            lo = 2 * d_hat - np.percentile(d, 97.5)
            hi = 2 * d_hat - np.percentile(d, 2.5)
            centred = d - d.mean()                 # null-centred draws
            p = (1 + (np.abs(centred) >= abs(d_hat)).sum()) / (1 + N_BOOT)
            rows.append(dict(pair=f"{a} vs {b}", diff=d_hat, lo=lo, hi=hi,
                             p_boot=float(p)))
    pw = pd.DataFrame(rows)
    pw['q'] = multipletests(pw['p_boot'], method='fdr_bh')[1]
    print("\npairwise (bootstrap difference, FDR across pairs):")
    print(pw.sort_values('p_boot').to_string(index=False,
          float_format=lambda v: f"{v:+.4f}"))

    # ---------------------------------------------------------------- figure
    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 9,
        'axes.labelsize': 10, 'axes.titlesize': 11, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(14.5 * CM, 4.2 * CM),
                             gridspec_kw={'wspace': 0.52, 'width_ratios': [1, 1.15]})

    ax = axes[0]
    for i, roi in enumerate(rois):
        lo_p, hi_p = np.percentile(boots[roi], [2.5, 97.5])
        lo, hi = 2 * point[roi] - hi_p, 2 * point[roi] - lo_p
        ax.plot([i, i], [lo, hi], color='0.15', lw=2.2, zorder=3,
                solid_capstyle='round')
        ax.scatter([i], [point[roi]], s=74, facecolor=ROI_COLOURS[roi],
                   edgecolor='0.1', linewidth=1.3, zorder=4)
    ax.axhline(0, color='0.45', ls=':', lw=.9, zorder=0)
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels([lab(r) for r in rois], rotation=30, ha='right')
    ax.set_ylabel(r'RSA $\beta$ (95% CI)', labelpad=2, fontsize=8.5)
    ax.set_title('a   Bootstrap over cells', loc='left', fontweight='bold', pad=6)
    ax.margins(y=0.18)

    ax = axes[1]
    pw2 = pw.sort_values('diff').reset_index(drop=True)
    for i, r in pw2.iterrows():
        sig = r['q'] < .05
        ax.plot([r['lo'], r['hi']], [i, i], color='0.1' if sig else '0.6',
                lw=2.2 if sig else 1.1, alpha=1 if sig else .5, zorder=3,
                solid_capstyle='round')
        ax.scatter([r['diff']], [i], s=34, color='0.1' if sig else '0.6',
                   alpha=1 if sig else .5, zorder=4)
    ax.axvline(0, color='0.45', ls=':', lw=.9, zorder=0)
    ax.set_yticks(range(len(pw2)))
    ax.set_yticklabels([p.replace('HC_anterior', 'HC ant').replace('HC_mid', 'HC mid')
                        for p in pw2['pair']], fontsize=7)
    ax.set_xlabel(r'$\Delta\beta$ (95% CI)', labelpad=2, fontsize=8.5)
    ax.set_title('b   Pairwise differences', loc='left', fontweight='bold', pad=6)

    for ext in ('pdf', 'png'):
        fig.savefig(f"{OUT}/fig5_rsa_roi_bootstrap.{ext}", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("\nwrote", OUT + "/fig5_rsa_roi_bootstrap.pdf")

    summary = {
        'question': 'Is the mOFC RSA effect larger than the other ROIs?',
        'method': 'Cells resampled with replacement WITHIN each ROI (never across); '
                  'the pseudo-population RDM rebuilt and the combo refitted each draw, '
                  'using the fitting functions extracted verbatim from '
                  'RSA_DSR_ROIs_simple.py. Full-sample fit verified against the '
                  'published beta before resampling.',
        'run': os.path.basename(RSA_RUN), 'combo': COMBO, 'test': TEST,
        'n_boot': N_BOOT, 'seed': SEED,
        'interval_type': 'basic (reverse-percentile) bootstrap; the raw '
                         'percentile interval is biased low because resampling '
                         'with replacement duplicates cells and an RDM built '
                         'from duplicates is less informative',
        'per_roi': {r: {'beta': point[r],
                        'boot_median': float(np.median(boots[r])),
                        'attenuation': float(point[r] - np.median(boots[r])),
                        'ci_lo': float(2 * point[r] - np.percentile(boots[r], 97.5)),
                        'ci_hi': float(2 * point[r] - np.percentile(boots[r], 2.5)),
                        'boot_sd': float(boots[r].std(ddof=1)),
                        'n_cells': int(np.load(f"{MAT_DIR}/mat_{r}.npz",
                                               allow_pickle=True)['mat'].shape[0])}
                    for r in rois},
        'pairwise': pw.to_dict('records'),
    }
    with open(OUT + '/rsa_roi_bootstrap_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print("wrote", OUT + '/rsa_roi_bootstrap_summary.json')


if __name__ == '__main__':
    main()
