"""
Figures for `discovery_cells` (N1, N2), drawn from its saved outputs only.

    from mc.analyse.swr_explore.discovery_cells_figures import draw
    draw(out_dir)

N1: per-cell firing around first discoveries (FD) and repeat-correct uncovers
(RC), both readings; the across-cell association with lag-0 tuning and with
depth over time. N2: evidence for the square just uncovered over time, both
readings, and its per-cell contributions against depth.

@author: Svenja Kuchenhoff
"""

import os
import json
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc.analyse.swr_explore.discovery_cells import CENTRES, N2_WIN

CM = 1 / 2.54
ROI_C = {"mPFC": "#448363", "mOFC": "#DC673E", "HC_all": "#4F4F4F"}
EV_C = {"FD": "#0e3d3a", "RC": "#9ec6b8"}
READ = {"own": "own baseline (−1.5…−1.0 s)", "matched": "stillness-matched"}
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.titlesize": 9})


def _clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _ms(X):
    X = np.stack(X)
    return X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X)), len(X)


def _bars(ax, res, col):
    for c in (res or {}).get("clusters", []):
        if c["p"] < 0.05:
            ax.plot([c["start_s"], c["stop_s"]], [0.96, 0.96], color=col, lw=3,
                    transform=ax.get_xaxis_transform(), solid_capstyle="butt")
            ax.text(c["stop_s"], 0.98, f" p={c['p']:.3f}", fontsize=6.5,
                    color=col, transform=ax.get_xaxis_transform(), va="bottom")


def draw(d):
    T = pd.read_pickle(os.path.join(d, "n1_cells.pkl"))
    R = json.load(open(os.path.join(d, "result.json")))
    n1 = R["n1"]
    fig, axes = plt.subplots(2, 3, figsize=(21 * CM, 13 * CM))
    for i, rd in enumerate(("own", "matched")):
        # a: FD and RC, cells in the top vs bottom third of lag-0 (display only)
        ax = axes[i, 0]
        S = T[T[f"{rd}_FD"].apply(lambda v: isinstance(v, np.ndarray))
              & T[f"{rd}_RC"].apply(lambda v: isinstance(v, np.ndarray))
              & np.isfinite(T.lag0)]
        lo, hi = np.nanpercentile(S.lag0, [33.3, 66.7])
        for grp, m, ls in (("top third lag-0", S.lag0 >= hi, "-"),
                           ("bottom third", S.lag0 <= lo, ":")):
            for ev in ("FD", "RC"):
                mu, se, n = _ms(S[m][f"{rd}_{ev}"])
                ax.plot(CENTRES, mu, color=EV_C[ev], ls=ls, lw=1.3,
                        label=f"{ev}, {grp} ({n})")
                if ls == "-":
                    ax.fill_between(CENTRES, mu - se, mu + se, color=EV_C[ev],
                                    alpha=0.2, lw=0)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("time from reward press (s)")
        ax.set_ylabel("firing change\n(fraction of mean rate)")
        ax.set_title(f"{'ad'[i]}  mPFC + mOFC cells — {READ[rd]}", fontsize=8.5)
        if i == 1:
            ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper center",
                      bbox_to_anchor=(0.5, -0.3))
        _clean(ax)
        # b: rho(t) with lag-0, FD - RC and ripple - no ripple
        ax = axes[i, 1]
        for key, lab, col in ((f"{rd} | FD-RC | lag0 | mPFC+mOFC",
                               "FD − RC", "#0e3d3a"),
                              (f"{rd} | rip-norip | lag0 | mPFC+mOFC",
                               "FD: ripple − none", "#a30d6c")):
            r = n1.get(key)
            if r:
                ax.plot(CENTRES, r["rho"], color=col, lw=1.3,
                        label=f"{lab} (n = {r['n']})")
                _bars(ax, r, col)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("time from reward press (s)")
        ax.set_ylabel("Spearman ρ with lag-0 tuning\n(across cells)")
        ax.set_title(f"{'be'[i]}  do lag-0 cells respond more?", fontsize=8.5)
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")
        _clean(ax)
        # c: rho(t) with depth, sites
        ax = axes[i, 2]
        for key, lab, col in ((f"{rd} | FD-RC | z (sites) | mPFC+mOFC",
                               "mPFC + mOFC sites", "#7E6B8F"),
                              (f"{rd} | FD-RC | z (sites) | mPFC",
                               "mPFC sites", ROI_C["mPFC"])):
            r = n1.get(key)
            if r:
                ax.plot(CENTRES, r["rho"], color=col, lw=1.3,
                        label=f"{lab} (n = {r['n']})")
                _bars(ax, r, col)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("time from reward press (s)")
        ax.set_ylabel("Spearman ρ of FD − RC\nwith MNI z (dorsal +)")
        ax.set_title(f"{'cf'[i]}  does it depend on depth?", fontsize=8.5)
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")
        _clean(ax)
    fig.suptitle("N1 — 'now' cells at the moment of discovery (reward-locked)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10.5, y=1.04)
    fig.text(0.01, -0.02, textwrap.fill(
             "FD = first discoveries (first traversal), RC = later correct "
             "uncovers of the same square. Rows: own baseline (the "
             "pre-registered primary) and stillness-matched (uncover minus an "
             "arrow press with the same following stillness). a, d: mean ± SEM "
             "over cells (lag-0 thirds for display only; the test is "
             "continuous). b, e: across-cell Spearman of lag-0 tuning with the "
             "response per 100 ms bin; bars = clusters p < .05 over 0–2 s, null "
             "= lag-0 permuted within session. c, f: the same with MNI z, cells "
             "averaged per site, null = site depth permuted.", 150),
             fontsize=6.5, color="0.3", va="top")
    fig.tight_layout()
    for e in (".png", ".pdf"):
        fig.savefig(os.path.join(d, "N1_discovery_cells" + e), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    # N2
    S2 = pd.read_csv(os.path.join(d, "n2_summary.csv"))
    sites = pd.read_csv(os.path.join(d, "n2_sites.csv"))
    fig, axes = plt.subplots(1, 4, figsize=(28 * CM, 7 * CM),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.9],
                                          "wspace": 0.55})
    win = S2[S2.stop - S2.start == N2_WIN].copy()
    win["mid"] = win.start + N2_WIN / 2
    for ax, roi in zip(axes[:3], ("mPFC", "mOFC", "HC_all")):
        for meas, lab, col, ls in (("z_FD", "FD", EV_C["FD"], "-"),
                                   ("z_RC", "RC", EV_C["RC"], "-"),
                                   ("z_FD_own", "FD, own baseline", "#0e3d3a",
                                    "--"),
                                   ("z_FD-RC_matched", "FD − RC, matched",
                                    "#a30d6c", "-")):
            g = win[(win.roi == roi) & (win.measure == meas)].sort_values("mid")
            if not len(g):
                continue
            ax.plot(g.mid, g["mean"], color=col, ls=ls, lw=1.3, marker="o",
                    ms=2.5, label=lab)
            ax.fill_between(g.mid, g["mean"] - g["sem"], g["mean"] + g["sem"],
                            color=col, alpha=0.15, lw=0)
            for _, r_ in g[g.p_perm < 0.05].iterrows():
                ax.text(r_.mid, r_["mean"] + r_["sem"], "*", color=col,
                        ha="center", fontsize=9)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("window centre from reward press (s)")
        ax.set_title(roi.replace("_all", ""), color=ROI_C[roi], pad=10)
        _clean(ax)
    axes[0].set_ylabel("evidence for the square\njust uncovered (z)")
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper center",
                   bbox_to_anchor=(1.9, -0.3), ncol=4)
    ax = axes[3]
    for roi in ("mPFC", "mOFC"):
        g = sites[sites.roi == roi]
        ax.scatter(g.z, g.z_cell, s=6 + 3 * g.n_cells, color=ROI_C[roi],
                   alpha=0.7, lw=0.3, edgecolor="0.2", label=roi)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    dep = R["n2_depth"]["mPFC+mOFC"]
    ax.set_xlabel("MNI z (mm)\nventral ← → dorsal")
    ax.set_ylabel("per-cell FD − RC\ncontribution (z), site mean")
    ax.set_title(f"by depth: ρ = {dep['rho']:+.2f}, p = {dep['p_perm']:.3f}",
                 fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False)
    _clean(ax)
    fig.suptitle("N2 — the newly found square, reward-locked and "
                 "time-resolved", x=0.01, ha="left", fontweight="bold",
                 fontsize=10.5, y=1.08)
    fig.text(0.01, -0.2, textwrap.fill(
             "Evidence for the square just uncovered (leave-one-configuration-"
             "out templates, z vs 1000 square-label permutations), 250 ms "
             "windows every 125 ms; mean ± SEM over sessions; * p < .05 "
             "(sign-flip, uncorrected across windows). FD own: post-press minus "
             "the same event's −0.5…−0.05 s. FD − RC matched: first discoveries "
             "vs repeat-correct uncovers matched on the following stillness. "
             "Right: per-cell contribution to FD − RC in 0.05–1.0 s, averaged "
             "per recording site; ρ with MNI z, site depth permuted.", 190),
             fontsize=6.5, color="0.3", va="top")
    for e in (".png", ".pdf"):
        fig.savefig(os.path.join(d, "N2_new_square" + e), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
