#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures for the ripple-content analysis -- one per stage, overview quality.

    python scripts/swr_content_figures.py

Fig 1  the data and the budget      (Stage 0 descriptives)
Fig 2  location tuning              (Stage 1a)
Fig 3  the positive control         (Stage 2 + width sweep)

These are OVERVIEW figures, not publication panels: saved as .pdf and .jpeg,
and every panel's numbers are written alongside as CSV so nothing plotted is
recomputed on the fly. Colours follow the project convention
(`mc.plotting.cell_results`): fixed era_brewer hue per ROI, teal->green for the
3x3 grid locations, dark green for an observed value against a null.

@author: Svenja Kuchenhoff
"""

import os
import glob
import json
import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mc.analyse.ripple_rsa as rrsa
from mc.plotting.cell_results import (get_roi_colour, roi_display,
                                      LOCATION_COLORS, OBSERVED_VALUE_COLOR)

ROIS = ["HC_anterior", "HC_mid", "mPFC", "mOFC", "PCC"]
CM = 1 / 2.54

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _latest(pattern):
    hits = sorted(glob.glob(os.path.join(rrsa._derivatives(), "group", "swr",
                                         pattern)))
    return hits[-1] if hits else None


def fig_dir():
    d = os.path.join(rrsa._derivatives(), "group", "swr",
                     f"ripple_content_figures_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    return d


def save(fig, d, name):
    for ext in ("pdf", "jpeg"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"))
    print(f"  {name}.pdf / .jpeg")


def _bar(ax, labels, vals, errs=None, colors=None, ylab=""):
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=errs, color=colors, capsize=2, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([roi_display(l) for l in labels], rotation=30, ha="right")
    ax.set_ylabel(ylab)


# ---------------------------------------------------------------- Figure 1
def figure1(d):
    src = _latest("ripple_content_descriptives_*")
    if not src:
        print("  Fig 1 skipped: no descriptives")
        return
    S = pd.read_csv(os.path.join(src, "session_descriptives.csv"))
    C = pd.read_csv(os.path.join(src, "cell_descriptives.csv"))
    DW = pd.read_csv(os.path.join(src, "location_dwell.csv"))
    CO = pd.read_csv(os.path.join(src, "coactivity.csv"))
    P = pd.read_csv(os.path.join(src, "peri_ripple_rate.csv"))

    fig, ax = plt.subplots(2, 3, figsize=(20 * CM, 12 * CM))

    a = ax[0, 0]
    a.hist(S.ripple_rate_hz, bins=20, color="#23677E", edgecolor="white")
    a.axvline(S.ripple_rate_hz.median(), color=OBSERVED_VALUE_COLOR, lw=2)
    a.set_xlabel("ripple rate (Hz, in-task)"); a.set_ylabel("sessions")
    a.set_title("a  Ripple rate", loc="left")

    a = ax[0, 1]
    a.hist(S.median_duration_s * 1000, bins=20, color="#23677E", edgecolor="white")
    a.set_xlabel("median ripple duration (ms)"); a.set_ylabel("sessions")
    a.set_title("b  Ripple duration", loc="left")

    a = ax[0, 2]
    g = C.groupby("roi").agg(cells=("cell", "size"),
                             sess=("session", "nunique")).reindex(ROIS).dropna()
    x = np.arange(len(g))
    a.bar(x - 0.2, g.cells, 0.4, label="cells",
          color=[get_roi_colour(r) for r in g.index], edgecolor="none")
    a.bar(x + 0.2, g.sess, 0.4, label="sessions",
          color=[get_roi_colour(r) for r in g.index], alpha=0.45, edgecolor="none")
    a.set_xticks(x); a.set_xticklabels([roi_display(i) for i in g.index],
                                       rotation=30, ha="right")
    a.set_ylabel("count"); a.legend(frameon=False)
    a.set_title("c  Coverage", loc="left")

    a = ax[1, 0]
    v = C.groupby("roi").spikes_per_ripple.agg(["mean", "sem"]).reindex(ROIS).dropna()
    _bar(a, list(v.index), v["mean"], v["sem"],
         [get_roi_colour(r) for r in v.index],
         "spikes per cell\nper ripple")
    a.set_title("d  The spike budget", loc="left")
    a.axhline(1.0, color="grey", ls=":", lw=1)
    a.text(0.02, 1.02, "1 spike", color="grey", fontsize=7,
           transform=a.get_yaxis_transform())

    a = ax[1, 1]
    for r in ROIS:
        c = CO[CO.roi == r].groupby("n_active").frac.mean()
        if not len(c):
            continue
        a.plot(c.index, c.values, "-o", ms=3, color=get_roi_colour(r),
               label=roi_display(r))
    a.set_xlabel("co-active cells in a ripple"); a.set_ylabel("fraction of ripples")
    a.legend(frameon=False); a.set_title("e  Co-activity", loc="left")

    a = ax[1, 2]
    # each ROI against ITS OWN pre-ripple baseline: a shared absolute y-axis
    # hides the modulation, which is the one thing this panel exists to show
    base = (P.t_s > -0.5) & (P.t_s < -0.25)
    for r in ROIS:
        if r in P.columns:
            b0 = P.loc[base, r].mean()
            a.plot(P.t_s * 1000, 100 * (P[r] - b0) / b0, color=get_roi_colour(r),
                   label=roi_display(r), lw=1.6)
    a.axvline(0, color=OBSERVED_VALUE_COLOR, lw=1, ls="--")
    a.axhline(0, color="grey", lw=.8)
    a.set_xlabel("time from ripple peak (ms)")
    a.set_ylabel("firing rate\n(% vs -500:-250 ms)")
    a.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    a.set_title("f  Peri-ripple firing", loc="left")

    fig.suptitle("Figure 1 — what the data looks like, and what a ripple can afford\n"
                 f"{len(S)} sessions, {len(C)} cells; ripple rate and duration in-task; "
                 "panel d is the constraint on everything downstream",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    save(fig, d, "fig1_data_and_budget")
    v.to_csv(os.path.join(d, "fig1d_spike_budget.csv"))
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
def figure2(d):
    src = _latest("ripple_content_templates_*")
    if not src:
        print("  Fig 2 skipped: no template results")
        return
    REL = pd.read_csv(os.path.join(src, "template_reliability.csv"))

    fig, ax = plt.subplots(1, 3, figsize=(20 * CM, 6 * CM))

    a = ax[0]
    for i, r in enumerate(ROIS):
        v = REL[REL.roi == r].r.dropna()
        if len(v) < 5:
            continue
        a.scatter(np.full(len(v), i) + np.random.uniform(-.15, .15, len(v)), v,
                  s=5, color=get_roi_colour(r), alpha=.45, edgecolors="none")
        a.plot([i - .3, i + .3], [v.mean()] * 2, color=OBSERVED_VALUE_COLOR, lw=2)
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(range(len(ROIS)))
    a.set_xticklabels([roi_display(r) for r in ROIS], rotation=30, ha="right")
    a.set_ylabel("split-half r\n(odd vs even grids)")
    a.set_title("a  Per-cell location tuning", loc="left")

    a = ax[1]
    rows = []
    for r in ROIS:
        v = REL[REL.roi == r].r.dropna()
        if len(v) < 5:
            continue
        t, p = stats.ttest_1samp(v, 0)
        ci = stats.t.interval(0.95, len(v) - 1, v.mean(), stats.sem(v))
        rows.append(dict(roi=r, n=len(v), mean_r=v.mean(), lo=ci[0], hi=ci[1],
                         t=t, p=p, frac_pos=float((v > 0).mean())))
    R = pd.DataFrame(rows)
    y = np.arange(len(R))
    a.errorbar(R.mean_r, y, xerr=[R.mean_r - R.lo, R.hi - R.mean_r], fmt="o",
               ms=6, color="k", ecolor="grey", capsize=3)
    for i, r in enumerate(R.roi):
        a.plot(R.mean_r.iloc[i], i, "o", ms=6, color=get_roi_colour(r))
    a.axvline(0, color="grey", lw=1)
    a.set_yticks(y); a.set_yticklabels([roi_display(r) for r in R.roi])
    a.set_xlabel("mean split-half r  [95% CI]")
    a.set_title("b  Group mean", loc="left")
    for i, (p_, n_) in enumerate(zip(R.p, R.n)):
        a.text(R.hi.iloc[i] + .004, i, f"n={n_}, p={p_:.3g}", va="center", fontsize=7)
    a.set_xlim(R.lo.min() - .02, R.hi.max() + .16)     # room for the labels

    a = ax[2]
    a.bar(np.arange(len(R)), R.frac_pos,
          color=[get_roi_colour(r) for r in R.roi], edgecolor="none")
    a.axhline(0.5, color=OBSERVED_VALUE_COLOR, ls="--", lw=1.5)
    a.text(0.98, 0.52, "chance", color=OBSERVED_VALUE_COLOR, fontsize=7,
           ha="right", transform=a.get_yaxis_transform())
    a.set_xticks(np.arange(len(R)))
    a.set_xticklabels([roi_display(r) for r in R.roi], rotation=30, ha="right")
    a.set_ylabel("fraction of cells with r > 0"); a.set_ylim(0, 1)
    a.set_title("c  How many cells", loc="left")

    fig.suptitle("Figure 2 — location tuning is real at the group level and tiny per cell\n"
                 "place maps built on the real-time timeline, leave-one-grid-out; "
                 "mPFC is flat, as a lag-tagged code should be",
                 fontsize=10, y=1.10)
    fig.tight_layout()
    save(fig, d, "fig2_location_tuning")
    R.to_csv(os.path.join(d, "fig2_reliability_stats.csv"), index=False)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def figure3(d):
    src = _latest("ripple_content_widthsweep_*")
    if not src:
        print("  Fig 3 skipped: no width sweep")
        return
    W = pd.read_csv(os.path.join(src, "width_sweep.csv"))

    fig, ax = plt.subplots(1, 3, figsize=(20 * CM, 6 * CM))

    a = ax[0]
    rows = []
    jit = {r: 1.0 + 0.03 * (i - 1.5) for i, r in enumerate(ROIS)}
    for r in ROIS:
        g = W[W.roi == r].dropna(subset=["z"])
        if not len(g):
            continue
        m = g.groupby("width").z.agg(["mean", "sem", "size"])
        m = m[m["size"] >= 5]
        if not len(m):
            continue
        a.errorbar(m.index * 1000 * jit[r], m["mean"], yerr=m["sem"], fmt="-o",
                   ms=4, lw=1.6, color=get_roi_colour(r), label=roi_display(r),
                   capsize=2)
        for w_, row in m.iterrows():
            rows.append(dict(roi=r, width=w_, z=row["mean"], sem=row["sem"],
                             n=row["size"]))
    a.axhline(0, color="grey", lw=1)
    a.set_xscale("log")
    a.set_xlabel("window width (ms, log)")
    a.set_ylabel("location signal\n(z vs own permutation null)")
    a.legend(frameon=False, fontsize=7, loc="upper left",
             bbox_to_anchor=(0.0, -0.28), ncol=4, columnspacing=1.0)
    a.set_title("a  Location signal vs window width", loc="left")

    a = ax[1]
    m = W.groupby(["roi", "width"]).spikes_per_cell.mean().reset_index()
    for r in ROIS:
        g = m[m.roi == r]
        if len(g):
            a.plot(g.width * 1000, g.spikes_per_cell, "-o", ms=4,
                   color=get_roi_colour(r), label=roi_display(r))
    a.axhline(1, color="grey", ls=":", lw=1)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("window width (ms, log)"); a.set_ylabel("spikes per cell (log)")
    a.set_title("b  Spikes available", loc="left")

    a = ax[2]
    peak = 0.125
    present = [r for r in ROIS
               if len(W[(W.roi == r) & (W.width == peak)].z.dropna()) >= 5]
    for i, r in enumerate(present):
        v = W[(W.roi == r) & (W.width == peak)].z.dropna()
        a.scatter(np.full(len(v), i) + np.random.uniform(-.15, .15, len(v)), v,
                  s=10, color=get_roi_colour(r), alpha=.6, edgecolors="none")
        a.plot([i - .3, i + .3], [v.mean()] * 2, color=OBSERVED_VALUE_COLOR, lw=2)
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(range(len(present)))
    a.set_xticklabels([roi_display(r) for r in present], rotation=30, ha="right")
    a.set_ylabel(f"z at {int(peak*1000)} ms")
    a.set_title("c  Per session, at the peak", loc="left")

    fig.suptitle("Figure 3 — HC_mid reads out current location at ripple width\n"
                 "every eligible window tiled; flat 60-250 ms, falling beyond 500 ms "
                 "as windows straddle locations (median dwell ~0.7 s)",
                 fontsize=10, y=1.10)
    fig.tight_layout()
    save(fig, d, "fig3_positive_control")
    pd.DataFrame(rows).to_csv(os.path.join(d, "fig3a_width_sweep.csv"), index=False)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2b
def figure2b(d, n_examples=4):
    """Example place maps: the most reliable cells per ROI, as 3x3 grids."""
    src = _latest("ripple_content_templates_*")
    if not src:
        print("  Fig 2b skipped: no template results")
        return
    import mc.analyse.swr_location as swl
    import mc.analyse.swr_content as swc
    REL = pd.read_csv(os.path.join(src, "template_reliability.csv")).dropna(subset=["r"])

    pick = (REL.sort_values("r", ascending=False)
               .groupby("roi").head(n_examples))
    pick = pick[pick.roi.isin(ROIS)]
    need = sorted(pick.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=need, verbose=False)
    steps = swl.load(need)

    rois_present = [r for r in ROIS if (pick.roi == r).any()]
    fig, ax = plt.subplots(len(rois_present), n_examples,
                           figsize=(n_examples * 3.2 * CM,
                                    len(rois_present) * 3.4 * CM),
                           squeeze=False)
    maps = []
    for i, r in enumerate(rois_present):
        sub = pick[pick.roi == r].reset_index(drop=True)
        for j in range(n_examples):
            a = ax[i][j]
            a.set_xticks([]); a.set_yticks([])
            if j >= len(sub):
                a.axis("off"); continue
            row = sub.iloc[j]
            occ = swc.occupancy(steps, int(row.session))
            m = swc.place_map(spk[int(row.session)]["spikes"][int(row.cell)], occ)
            maps.append(dict(roi=r, session=int(row.session), cell=int(row.cell),
                             r=row.r, **{f"loc{k+1}": m[k] for k in range(9)}))
            # locations are column-major on the 3x3 grid (loc = col*3 + row + 1)
            g = np.full((3, 3), np.nan)
            for k in range(9):
                g[k % 3, k // 3] = m[k]
            a.imshow(g, cmap="viridis")
            a.set_title(f"s{int(row.session)} c{int(row.cell)}\nr={row.r:.2f}",
                        fontsize=7, pad=2)
            if j == 0:
                a.set_ylabel(roi_display(r), fontsize=9)
    fig.suptitle("Figure 2b — example place maps (the 4 most reliable cells per ROI)\n"
                 "firing rate per grid location; colour scaled within cell. "
                 "These are the BEST cells, not typical ones",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    save(fig, d, "fig2b_example_place_maps")
    pd.DataFrame(maps).to_csv(os.path.join(d, "fig2b_example_maps.csv"), index=False)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 4
def figure4(d):
    """⚠ Read with the occupancy panel: the role labels expire after ~0.4 s."""
    src = _latest("ripple_content_timecourse_*")
    if not src:
        print("  Fig 4 skipped: no timecourse")
        return
    T = pd.read_csv(os.path.join(src, "location_timecourse.csv"))
    occ_f = os.path.join(src, "occupancy_by_lag.csv")
    OCC = pd.read_csv(occ_f) if os.path.exists(occ_f) else None
    rois_present = [r for r in ROIS if (T.roi == r).any()]
    has_reg = "b_current" in T.columns
    n_rows = (2 if has_reg else 1) + (1 if OCC is not None else 0)
    fig, ax = plt.subplots(n_rows, len(rois_present),
                           figsize=(len(rois_present) * 5.2 * CM,
                                    n_rows * 5.5 * CM), squeeze=False)
    for i, r in enumerate(rois_present):
        a = ax[0][i]
        g = T[T.roi == r]
        for col, lab, ls in (("new_loc", "location entered", "-"),
                             ("old_loc", "location left", "--")):
            m = g.groupby("offset")[col].agg(["mean", "sem"])
            a.plot(m.index, m["mean"], ls, color=get_roi_colour(r), lw=1.8,
                   label=lab)
            a.fill_between(m.index, m["mean"] - m["sem"], m["mean"] + m["sem"],
                           color=get_roi_colour(r), alpha=.18, lw=0)
        a.axvline(0, color=OBSERVED_VALUE_COLOR, ls=":", lw=1.2)
        a.axhline(0, color="grey", lw=.8)
        a.set_title(f"{roi_display(r)}  (n={g.session.nunique()})", loc="left")
        a.set_xlabel("time from arrival (s)")
        if i == 0:
            a.set_ylabel("location evidence\n(within-window z)")
        if i == len(rois_present) - 1:
            a.legend(frameon=False, fontsize=7, loc="upper left",
                     bbox_to_anchor=(1.02, 1.0))
        if has_reg:
            # row 2: regression over ALL NINE locations. `target_z` centres on
            # the mean of the nine, so if two locations are both elevated they
            # are elevated against a mean containing them -- these coefficients
            # are the unique contribution of each role with the others held.
            b = ax[1][i]
            g2 = T[T.roi == r]
            for col, lab, ls in (("b_current", "location entered", "-"),
                                 ("b_previous", "location left", "--"),
                                 ("b_next", "location coming next", ":")):
                m = g2.groupby("offset")[col].agg(["mean", "sem"])
                b.plot(m.index, m["mean"], ls, color=get_roi_colour(r), lw=1.8,
                       label=lab)
                b.fill_between(m.index, m["mean"] - m["sem"],
                               m["mean"] + m["sem"],
                               color=get_roi_colour(r), alpha=.15, lw=0)
            b.axvline(0, color=OBSERVED_VALUE_COLOR, ls=":", lw=1.2)
            b.axhline(0, color="grey", lw=.8)
            b.set_xlabel("time from arrival (s)")
            if i == 0:
                b.set_ylabel("regression beta\n(unique contribution)")
            if i == len(rois_present) - 1:
                b.legend(frameon=False, fontsize=7, loc="upper left",
                         bbox_to_anchor=(1.02, 1.0))
    if OCC is not None:
        # THE CONTROL. Without it the bottom row invites exactly the wrong
        # reading: `b_next` overtaking `b_current` after ~0.4 s is not the
        # hippocampus running ahead, it is the subject walking there.
        m = OCC.groupby("offset")[["at_current", "at_previous",
                                   "at_next"]].mean()
        for i in range(len(rois_present)):
            c = ax[n_rows - 1][i]
            for col, lab, ls in (("at_current", "on the location entered", "-"),
                                 ("at_previous", "still on the one left", "--"),
                                 ("at_next", "already on the next one", ":")):
                c.plot(m.index, m[col], ls, lw=1.8, color="#5C1027")
            c.axvline(0, color=OBSERVED_VALUE_COLOR, ls=":", lw=1.2)
            c.axhline(.5, color="grey", lw=.8, ls="--")
            c.set_xlabel("time from arrival (s)")
            c.set_ylim(0, 1)
            if i == 0:
                c.set_ylabel("P(subject is actually\nstanding there)")
            if i == len(rois_present) - 1:
                c.legend(["on the location entered", "still on the one left",
                          "already on the next one"], frameon=False,
                         fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        cross = m.index[(m.at_next > m.at_current) & (m.index > 0)]
        cross = cross[0] if len(cross) else np.nan
    else:
        cross = np.nan
    fig.suptitle("Figure 4 — location information over time, aligned to arriving "
                 "at a new location\n"
                 "rows 1-2: evidence and regression betas per role. ROW 3 IS "
                 "THE CONTROL and must be read first:\n"
                 + (f"the labels expire — from {cross:+.2f} s the subject is more "
                    "often on the NEXT square than the one they arrived at "
                    "(median dwell 0.367 s),\nso `next` overtaking `current` is "
                    "walking, not prospection. Only |t| < 0.25 s is "
                    "interpretable." if np.isfinite(cross) else ""),
                 fontsize=9.5, y=1.04)
    fig.tight_layout()
    save(fig, d, "fig4_location_timecourse")
    T.groupby(["roi", "offset"])[["new_loc", "old_loc"]].mean().to_csv(
        os.path.join(d, "fig4_timecourse_means.csv"))
    plt.close(fig)


# ---------------------------------------------------------------- Figure 5
def figure5(d, scheme="all", tag=""):
    """Stage 3: location content inside ripples, and against matched flanks.

    `scheme` selects the cell-weighting variant: "all" for every cell,
    "thresh_0.2" for cells whose place map generalises across configurations.
    """
    src = _latest("ripple_content_stage3_*")
    if not src:
        print("  Fig 5 skipped: no Stage 3 results")
        return
    S = pd.read_csv(os.path.join(src, "stage3_per_session.csv"))
    # the Stage 3 file now carries a cell-selection sweep; this figure is the
    # ALL-CELLS version, and Figure 8 carries the sweep. Without this filter the
    # panels would silently average across selection thresholds.
    if "scheme" in S.columns:
        S = S[S.scheme == scheme]
    order = [r for r in ["HC_mid", "HC_anterior", "mOFC", "mPFC"]
             if (S.roi == r).any()]
    fig, ax = plt.subplots(1, 3, figsize=(20 * CM, 6 * CM))

    a = ax[0]
    x = np.arange(len(order))
    for k, (col, lab, al) in enumerate([("z_ripple", "in ripple", 1.0),
                                        ("z_flank", "matched flank", 0.45)]):
        m = [S[S.roi == r][col].mean() for r in order]
        e = [S[S.roi == r][col].sem() for r in order]
        a.bar(x + (k - .5) * .4, m, .4, yerr=e, capsize=2, alpha=al,
              color=[get_roi_colour(r) for r in order], edgecolor="none",
              label=lab)
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(x); a.set_xticklabels([roi_display(r) for r in order],
                                       rotation=30, ha="right")
    a.set_ylabel("location signal\n(z vs permutation null)")
    a.legend(frameon=False, fontsize=7)
    a.set_title("a  C0 — inside the ripple", loc="left")

    a = ax[1]
    for i, r in enumerate(order):
        v = S[S.roi == r].z_diff.dropna()
        a.scatter(np.full(len(v), i) + np.random.uniform(-.15, .15, len(v)), v,
                  s=12, color=get_roi_colour(r), alpha=.6, edgecolors="none")
        a.plot([i - .3, i + .3], [v.mean()] * 2, color=OBSERVED_VALUE_COLOR, lw=2.5)
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(range(len(order)))
    a.set_xticklabels([roi_display(r) for r in order], rotation=30, ha="right")
    a.set_ylabel("ripple - flank (z)")
    a.set_title("b  C1 — ripple-specific?", loc="left")

    a = ax[2]
    g = S[S.roi == "HC_mid"]
    for _, row in g.iterrows():
        a.plot([0, 1], [row.z_flank, row.z_ripple], "-", color="#CCB178",
               alpha=.45, lw=1)
    a.plot([0, 1], [g.z_flank.mean(), g.z_ripple.mean()], "-o",
           color=OBSERVED_VALUE_COLOR, lw=2.5, ms=6)
    a.axhline(0, color="grey", lw=1)
    a.set_xticks([0, 1]); a.set_xticklabels(["flank", "ripple"])
    a.set_xlim(-.3, 1.3); a.set_ylabel("location signal (z)")
    a.set_title(f"c  HC_mid, per session (n={len(g)})", loc="left")

    # the caption states the numbers THIS run produced. An earlier version
    # hard-coded them and went stale the moment matched flanks and per-cell
    # z-scoring changed the answer.
    def _pp(roi, col):
        v = S[S.roi == roi][col].dropna()
        return v.mean(), stats.ttest_1samp(v, 0)[1]
    bits = []
    for r in order[:2]:
        m0, p0 = _pp(r, "z_ripple")
        m1, p1 = _pp(r, "z_diff")
        bits.append(f"{roi_display(r)} C0 {m0:+.3f} (p = {p0:.3g}), "
                    f"C1 {m1:+.3f} (p = {p1:.3g})")
    fig.suptitle("Figure 5 — Stage 3: location content in ripple windows "
                 + (" (ALL cells)" if scheme == "all" else
                    " (cells whose place map generalises)") + "\n"
                 + ";  ".join(bits) + "\n"
                 "Flanks matched on location, occupancy and width; counts "
                 "z-scored per cell; CV by configuration. EXPLORATORY.",
                 fontsize=10, y=1.16)
    fig.tight_layout()
    save(fig, d, f"fig5_stage3_ripple_content{tag}")
    S.groupby("roi")[["z_ripple", "z_flank", "z_diff"]].agg(["mean", "sem"]).to_csv(
        os.path.join(d, f"fig5_stage3_summary{tag}.csv"))
    plt.close(fig)


# ---------------------------------------------------------------- Figure 6

# Ordered by how much the subject knows about the square underfoot, so the
# project's 3-shade phase ramp is the right scale: pastel = nothing there,
# bordeaux = a reward already found.
I11_COLOUR = {"nonrew": "#FCDDE3", "future_rew": "#D7657F",
              "known_rew": "#5C1027"}
I11_LABEL = {"nonrew": "not a reward", "future_rew": "reward,\nnot yet found",
             "known_rew": "reward,\nalready found"}
I11_CLASSES = ["nonrew", "future_rew", "known_rew"]


def figure6(d, scheme="all", tag=""):
    """I11 — does a ripple carry more about a REWARDED square, during
    exploration? Primary contrast known_rew - nonrew, ripple minus flank."""
    src = _latest("ripple_content_i11_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 6 skipped: no I11 results")
        return
    R = pd.read_csv(os.path.join(src, "i11_per_session.csv"))
    B = pd.read_csv(os.path.join(src, "i11_matching_balance.csv"))
    prim = int(json.load(open(os.path.join(src, "settings.json")))["primary_min"])
    R = R[R.n_triplets >= prim]
    if "scheme" in R.columns:
        R = R[R.scheme == scheme]
    sets = [r for r in ["HC_all", "HC_mid", "HC_anterior", "mOFC", "mPFC"]
            if (R.roi == r).sum() >= 5]

    fig, ax = plt.subplots(2, 2, figsize=(18 * CM, 13 * CM))

    # a — the stillness confound, and that the matching removes it
    a = ax[0, 0]
    x = np.arange(3)
    for k, (pre, leg, fill) in enumerate([("alldwell_", "all ripples", False),
                                          ("dwell_", "after matching", True)]):
        m = [B[f"{pre}{c}"].median() for c in I11_CLASSES]
        a.bar(x + (k - .5) * .38, m, .38, label=leg, linewidth=.9,
              color=([I11_COLOUR[c] for c in I11_CLASSES] if fill else "white"),
              edgecolor=[I11_COLOUR[c] for c in I11_CLASSES] if fill
                        else "#5C1027")
    a.set_xticks(x)
    a.set_xticklabels([I11_LABEL[c] for c in I11_CLASSES], fontsize=7)
    a.set_ylabel("dwell at the square (s)\nmedian over sessions")
    a.legend(frameon=False, fontsize=7, loc="upper left")
    a.set_title("a  subjects pause after finding a reward —\n"
                "   every ripple is matched on dwell and latency", loc="left",
                fontsize=9)

    # b — is there location content at all, per class (HC_all)
    a = ax[0, 1]
    g = R[R.roi == "HC_all"]
    labb = []
    for i, c in enumerate(I11_CLASSES):
        v = g[f"z_diff_{c}"].dropna()
        a.scatter(np.full(len(v), i) + np.random.uniform(-.13, .13, len(v)), v,
                  s=11, color=I11_COLOUR[c], alpha=.75,
                  edgecolors="#5C1027", linewidths=.3)
        a.plot([i - .3, i + .3], [v.mean()] * 2, color=OBSERVED_VALUE_COLOR,
               lw=2.5)
        labb.append(f"p = {stats.ttest_1samp(v, 0)[1]:.3g}")
    a.axhline(0, color="grey", lw=1)
    top = a.get_ylim()[1]
    a.set_ylim(a.get_ylim()[0], top + .25 * (top - a.get_ylim()[0]))
    for i, t in enumerate(labb):
        a.text(i, top, t, ha="center", va="bottom", fontsize=7)
    a.set_xticks(x)
    a.set_xticklabels([I11_LABEL[c] for c in I11_CLASSES], fontsize=7)
    a.set_ylabel("ripple minus matched flank\n(z vs template-label null)")
    a.set_title(f"b  hippocampus (pooled), n = {len(g)} sessions", loc="left",
                fontsize=9)

    # c — THE PRIMARY CONTRAST, per ROI
    a = ax[1, 0]
    lab = []
    for i, r in enumerate(sets):
        v = R[R.roi == r]["z_diff_known_rew_vs_nonrew"].dropna()
        a.scatter(np.full(len(v), i) + np.random.uniform(-.13, .13, len(v)), v,
                  s=11, color=get_roi_colour(r), alpha=.6, edgecolors="none")
        a.plot([i - .3, i + .3], [v.mean()] * 2, color=OBSERVED_VALUE_COLOR,
               lw=2.5)
        lab.append(f"p = {stats.ttest_1samp(v, 0)[1]:.2g}\nn = {len(v)}")
    a.axhline(0, color="grey", lw=1)
    top = a.get_ylim()[1]
    a.set_ylim(a.get_ylim()[0], top + .45 * (top - a.get_ylim()[0]))
    for i, t in enumerate(lab):
        a.text(i, top, t, ha="center", va="bottom", fontsize=7)
    a.set_xticks(range(len(sets)))
    a.set_xticklabels([roi_display(r) for r in sets], rotation=30, ha="right")
    a.set_ylabel("reward minus non-reward\n(z vs class-label shuffle)")
    a.set_title("c  PRIMARY: is content bigger at a known reward?", loc="left",
                fontsize=9)

    # d — all three contrasts, hippocampus pooled
    a = ax[1, 1]
    cons = [("known_rew", "nonrew", "reward found\n- no reward"),
            ("future_rew", "nonrew", "reward unfound\n- no reward"),
            ("known_rew", "future_rew", "reward found\n- reward unfound")]
    for i, (ca, cb, lab) in enumerate(cons):
        v = g[f"z_diff_{ca}_vs_{cb}"].dropna()
        a.bar(i, v.mean(), .55, yerr=v.sem(), capsize=2,
              color=I11_COLOUR[ca], edgecolor="#5C1027", linewidth=.6)
        a.text(i, v.mean() + np.sign(v.mean()) * (v.sem() + .05),
               f"p = {stats.ttest_1samp(v, 0)[1]:.2g}", ha="center",
               va="bottom" if v.mean() > 0 else "top", fontsize=7)
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(range(3))
    a.set_xticklabels([c[2] for c in cons], fontsize=7)
    a.set_ylabel("contrast (z vs class-label shuffle)")
    a.set_title(f"d  hippocampus (pooled), all three contrasts", loc="left",
                fontsize=9)

    kept = 100 * B.n_triplets.sum() / B.n_known_rew.sum()
    fig.suptitle(
        "Figure 6 — I11: during the FIRST traversal, does a ripple carry more "
        "about a square that is a reward?\n"
        f"Primary contrast is NULL. {len(B)} sessions, "
        f"{int(B.n_triplets.sum())} matched triplets ({kept:.0f}% of "
        "known-reward ripples; the rest sit in post-reward pauses too long to "
        "match).\n"
        "Matching repeated 20x and averaged. EXPLORATORY, uncorrected across "
        "ROIs, contrasts and measures.\n"
        + ("All cells." if scheme == "all" else
           "Only cells whose place map generalises across configurations."),
        fontsize=9.5, y=1.12)
    fig.tight_layout()
    save(fig, d, f"fig6_i11_reward_vs_nonreward{tag}")
    R.groupby("roi")[[c for c in R.columns if c.startswith("z_diff")]].agg(
        ["mean", "sem"]).to_csv(os.path.join(d, f"fig6_i11_summary{tag}.csv"))
    plt.close(fig)



# ---------------------------------------------------------------- Figure 7

def figure7(d):
    """Pseudo-population: pooling cells across sessions that ran the same
    configuration. Does pooling help, and is the signal ripple-specific?"""
    src = _latest("ripple_pseudopop_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 7 skipped: no pseudo-population results")
        return
    R = pd.read_csv(os.path.join(src, "pseudopop_curve.csv"))
    M = pd.read_csv(os.path.join(src, "pseudopop_evidence_matrix.csv"))
    S = pd.read_csv(os.path.join(src, "pseudopop_by_state.csv"))
    cfg = json.load(open(os.path.join(src, "settings.json")))
    n_avg = cfg["n_avg"][-1] if "n_avg" in cfg else 16

    fig, ax = plt.subplots(2, 2, figsize=(18 * CM, 13 * CM))

    # a — does pooling cells help?
    a = ax[0, 0]
    g = R[(R.roi == "HC_all") & (R.n_avg == n_avg)]
    for col, lab, ls in [("tz_ripple", "ripple", "-"),
                         ("tz_flank", "matched flank", "--")]:
        v = g.groupby("n_cells")[col].agg(["mean", "sem", "count"])
        v = v[v["count"] >= 3]
        a.errorbar(v.index, v["mean"], yerr=v["sem"], ls=ls, marker="o", ms=3,
                   lw=1.6, color=get_roi_colour("HC_anterior"), label=lab,
                   capsize=2)
    a.axhline(0, color="grey", lw=1)
    a.set_xscale("log", base=2)
    a.set_xlabel("cells pooled across sessions")
    a.set_ylabel("location signal per pseudo-trial\n(target z across the 9 squares)")
    a.legend(frameon=False, fontsize=7, loc="upper left")
    a.set_title("a  the signal is distributed —\n   it keeps growing to 316 cells",
                loc="left", fontsize=9)

    # b — and does averaging more ripples help?
    a = ax[0, 1]
    g2 = R[R.roi == "HC_all"]
    mx = g2.n_cells.max()
    for m in sorted(g2.n_avg.unique()):
        v = g2[(g2.n_avg == m)].groupby("n_cells")["tz_ripple"].agg(
            ["mean", "count"])
        v = v[v["count"] >= 3]
        a.plot(v.index, v["mean"], "-o", ms=3, lw=1.6,
               color=plt.cm.viridis(0.15 + 0.35 * np.log2(m)),
               label=f"{int(m)} ripple{'s' if m > 1 else ''}")
    a.axhline(0, color="grey", lw=1)
    a.set_xscale("log", base=2)
    a.set_xlabel("cells pooled")
    a.set_ylabel("location signal (target z)")
    a.legend(frameon=False, fontsize=7, title="averaged per cell",
             title_fontsize=7, loc="upper left")
    a.set_title("b  more ripples per read-out also help", loc="left", fontsize=9)

    # c — where the evidence goes, not just whether argmax was right
    a = ax[1, 0]
    m = (M[M.roi == "HC_all"].groupby(["true", "pred"]).z_ripple.mean()
         .unstack().to_numpy())
    lim = np.nanmax(np.abs(m))
    im = a.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
    a.set_xticks(range(9)); a.set_xticklabels(range(1, 10), fontsize=7)
    a.set_yticks(range(9)); a.set_yticklabels(range(1, 10), fontsize=7)
    a.set_xlabel("candidate square"); a.set_ylabel("square the subject is on")
    cb = fig.colorbar(im, ax=a, fraction=.046, pad=.04)
    cb.set_label("evidence (z within pseudo-trial)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    dg = np.nanmean(np.diag(m))
    off = np.nanmean(m[~np.eye(9, dtype=bool)])
    a.set_title(f"c  diagonal {dg:+.3f} vs off-diagonal {off:+.3f}\n"
                "   — elevated, but neighbours share it", loc="left",
                fontsize=9)

    # d — across the traversal
    a = ax[1, 1]
    for col, lab, ls in [("tz_ripple", "ripple", "-"),
                         ("tz_flank", "matched flank", "--")]:
        for roiname in ("HC_all", "HC_anterior"):
            v = S[S.roi == roiname].groupby("state")[col].agg(["mean", "sem"])
            a.errorbar(v.index, v["mean"], yerr=v["sem"], ls=ls, marker="o",
                       ms=4, lw=1.6, capsize=2, alpha=1 if ls == "-" else .55,
                       color=get_roi_colour(
                           "HC_anterior" if roiname == "HC_anterior" else "HC_mid"),
                       label=f"{roi_display(roiname)}, {lab}")
    a.axhline(0, color="grey", lw=1)
    a.set_xticks([1, 2, 3, 4]); a.set_xticklabels(list("ABCD"))
    a.set_xlabel("reward being sought")
    a.set_ylabel("location signal (target z)")
    a.legend(frameon=False, fontsize=6.5, ncol=1, loc="upper right",
             bbox_to_anchor=(1.02, 1.02))
    top_d = a.get_ylim()[1]
    a.set_ylim(a.get_ylim()[0], top_d + .45 * (top_d - a.get_ylim()[0]))
    a.set_title("d  location signal fades across the traversal", loc="left",
                fontsize=9)

    top = R[(R.roi == "HC_all") & (R.n_avg == n_avg)]
    top = top[top.n_cells == top.groupby("config").n_cells.transform("max")]
    dz = (top.tz_ripple - top.tz_flank).dropna()
    fig.suptitle(
        "Figure 7 — pseudo-populations: cells pooled across the sessions that "
        "ran the SAME configuration (matched on the reward tuple, not grid_id)\n"
        f"Pooling recovers a location signal single sessions barely have, and "
        f"it does not saturate. Ripple minus flank {dz.mean():+.3f}, "
        f"p = {stats.ttest_1samp(dz, 0)[1]:.2g} — still NOT ripple-specific.\n"
        "Independent per-cell sampling removes noise correlations, so absolute "
        "levels are optimistic; the comparisons are the result. EXPLORATORY.",
        fontsize=9.5, y=1.10)
    fig.tight_layout()
    save(fig, d, "fig7_pseudopopulation")
    plt.close(fig)



# ---------------------------------------------------------------- Figure 8

_HC_COL = {"HC_all": OBSERVED_VALUE_COLOR, "HC_mid": None,
           "HC_anterior": None}


def figure8(d):
    """Power, cell selection, and what the roles regression finds once untuned
    cells are dropped."""
    src = _latest("ripple_content_stage3_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    rsrc = _latest("ripple_content_roles_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src or not rsrc:
        print("  Fig 8 skipped: no sweep results")
        return
    S = pd.read_csv(os.path.join(src, "stage3_per_session.csv"))
    RS = pd.read_csv(os.path.join(rsrc, "roles_summary.csv"))
    _HC_COL["HC_mid"] = get_roi_colour("HC_mid")
    _HC_COL["HC_anterior"] = get_roi_colour("HC_anterior")
    sets = ["HC_all", "HC_mid", "HC_anterior"]
    fig, ax = plt.subplots(2, 2, figsize=(18 * CM, 13 * CM))

    # a — dropping untuned cells roughly doubles the location signal
    a = ax[0, 0]
    for r in sets:
        g = S[S.roi == r]
        v = g.groupby("scheme", sort=False).z_ripple.agg(["mean", "sem"])
        x = np.arange(len(v))
        a.errorbar(x, v["mean"], yerr=v["sem"], marker="o", ms=4,
                   lw=1.6, capsize=2, color=get_roi_colour(
                       "HC_mid" if r == "HC_all" else r),
                   ls="-" if r != "HC_all" else "--", label=roi_display(r))
        lab = [str(t).replace("thresh_", "\u2265") for t in v.index]
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(np.arange(len(lab))); a.set_xticklabels(lab)
    a.set_xlabel("how cells are weighted by place-map reliability")
    a.set_ylabel("location signal in ripples\n(C0, z vs permutation null)")
    a.legend(frameon=False, fontsize=7, loc="lower right")
    a.set_title("a  weighting by location tuning buys\n   nothing once it "
                "cannot peek", loc="left", fontsize=9)

    # b — but ripple-specificity stays a trend, and here is the power to see it
    a = ax[0, 1]
    for r in sets:
        g = S[S.roi == r].dropna(subset=["z_diff"])
        v = g.groupby("scheme", sort=False).z_diff.agg(["mean", "sem"])
        x = np.arange(len(v))
        a.errorbar(x, v["mean"], yerr=v["sem"], marker="o", ms=4, lw=1.6,
                   capsize=2, color=_HC_COL[r], label=roi_display(r))
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(np.arange(len(lab))); a.set_xticklabels(lab)
    a.set_xlabel("how cells are weighted by place-map reliability")
    a.set_ylabel("ripple minus matched flank\n(C1, z)")
    a.legend(frameon=False, fontsize=7, loc="lower right")
    a.set_title("b  ripple-specificity: positive throughout,\n"
                "   never significant", loc="left", fontsize=9)

    # c — the roles, tuned cells, inside ripples
    a = ax[1, 0]
    show = [("explore", "current"), ("explore", "adjacent"),
            ("explore", "next_step"), ("explore", "errors_here"),
            ("known", "current"), ("known", "goal"), ("known", "goal_1"),
            ("known", "on_route")]
    g = RS[(RS.scheme == "all") & (RS.roi == "HC_all")
           & (RS.lat_half == "all") & (RS.subset == "all")]
    xs, ys, cs, ls = [], [], [], []
    for i, (ph, tm) in enumerate(show):
        v = g[(g.phase == ph) & (g.term == tm)]
        if not len(v):
            continue
        xs.append(i); ys.append(float(v.z_ripple.iloc[0]))
        cs.append("#5C1027" if ph == "known" else "#D7657F")
        ls.append(f"{tm.replace('_', ' ')}\n({ph})")
    ys2 = []
    for ph, tm in show:
        v = g[(g.phase == ph) & (g.term == tm)]
        ys2.append(float(v.z_diff.iloc[0]) if len(v) else np.nan)
    a.bar([x - .2 for x in xs], ys, .38, color=cs, edgecolor="none",
          label="in ripple (includes ambient)")
    a.bar([x + .2 for x in xs], ys2, .38, color=cs, edgecolor="#222",
          linewidth=.7, alpha=.45, label="ripple minus flank")
    for i, (ph, tm) in enumerate(show):
        v = g[(g.phase == ph) & (g.term == tm)]
        if len(v) and float(v.p_ripple.iloc[0]) < .05:
            a.text(xs[i] - .2, ys[i] + .03, "*", ha="center", fontsize=11)
        if len(v) and float(v.p_diff.iloc[0]) < .05:
            a.text(xs[i] + .2, ys2[i] + .03, "*", ha="center", fontsize=11)
    a.legend(frameon=False, fontsize=6)
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(xs)
    a.set_xticklabels(ls, fontsize=6, rotation=35, ha="right")
    a.set_ylabel("role coefficient in ripples (z)")
    lo = min([0] + [v for v in ys + ys2 if np.isfinite(v)])
    hi = max([v for v in ys + ys2 if np.isfinite(v)])
    a.set_ylim(lo - .08, hi * 1.35)
    a.set_title("c  what a square has to BE — and only the RIGHT bar\n"
                "   is ripple-specific (* p < 0.05 uncorr.)",
                loc="left", fontsize=9)

    # d — and the control that removes the prospective ones
    a = ax[1, 1]
    pairs = [("explore", "next_step", "HC_all"), ("known", "goal", "HC_anterior")]
    w = .35
    for pi, (ph, tm, roin) in enumerate(pairs):
        g2 = RS[(RS.scheme == "all") & (RS.roi == roin) & (RS.phase == ph)
                & (RS.term == tm)
                & (RS.subset == "all")].set_index("lat_half")
        for hi, h in enumerate(("all", "early", "late")):
            if h not in g2.index:
                continue
            a.bar(pi + (hi - 1) * w, g2.loc[h, "z_ripple"], w,
                  color=["#9E9E9E", "#23677E", "#CCB178"][hi],
                  edgecolor="none", label=h if pi == 0 else None)
            if g2.loc[h, "p_ripple"] < .05:
                a.text(pi + (hi - 1) * w, g2.loc[h, "z_ripple"] + .02, "*",
                       ha="center", fontsize=11)
    a.axhline(0, color="grey", lw=1)
    a.set_xticks([0, 1])
    a.set_xticklabels(["next step\n(explore, HC pooled)",
                       "goal square\n(known, HC anterior)"], fontsize=7)
    a.set_ylabel("role coefficient in ripples (z)")
    a.set_ylim(a.get_ylim()[0], a.get_ylim()[1] * 1.22)
    a.legend(frameon=False, fontsize=7, title="where in the\noccupancy interval",
             title_fontsize=6.5, loc="upper left")
    a.set_title("d  both prospective effects live LATE —\n"
                "   i.e. the subject is already moving there", loc="left",
                fontsize=9)

    fig.suptitle(
        "Figure 8 — the nulls are underpowered, and cell weighting does NOT fix "
        "it\n"
        "Reliability estimated with the scored configuration HELD OUT: the "
        "location signal is flat across weighting schemes. An earlier version "
        "let it peek and appeared to double.\n"
        "`next_step` survives an adjacency control; the latency control that "
        "would say whether it is prospection or approach is unstable. "
        "EXPLORATORY, uncorrected.", fontsize=9.5, y=1.12)
    fig.tight_layout()
    save(fig, d, "fig8_power_and_roles")
    plt.close(fig)



# ---------------------------------------------------------------- Figure 9

def figure9(d):
    """The peri-ripple second, pseudo-population. The closest thing in this
    analysis to a ripple-locked effect, and it does not reach significance."""
    src = _latest("ripple_pseudo_timecourse_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 9 skipped: no timecourse")
        return
    T = pd.read_csv(os.path.join(src, "timecourse.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(19 * CM, 6 * CM))

    for i, roin in enumerate(("HC_all", "HC_anterior")):
        a = ax[i]
        g = T[(T.roi == roin) & (T.grouping == "current")]
        v = g.groupby("offset_s").target_z.agg(["mean", "sem"])
        col = get_roi_colour("HC_mid" if roin == "HC_all" else roin)
        a.plot(v.index * 1000, v["mean"], "-", color=col, lw=1.8)
        a.fill_between(v.index * 1000, v["mean"] - v["sem"],
                       v["mean"] + v["sem"], color=col, alpha=.2, lw=0)
        base = g[np.abs(g.offset_s) >= .3].target_z.mean()
        a.axhline(base, color="grey", ls="--", lw=1)
        a.axvline(0, color=OBSERVED_VALUE_COLOR, ls=":", lw=1.2)
        a.axhline(0, color="grey", lw=.8)
        a.set_xlabel("time from ripple peak (ms)")
        a.set_ylabel("location signal\n(target z, current square)")
        pk = v["mean"].idxmax()
        a.set_title(f"{roi_display(roin)} — peak {v['mean'].max():+.3f} at "
                    f"{pk * 1000:+.0f} ms\n"
                    f"baseline {base:+.3f} (|t| \u2265 300 ms)", loc="left",
                    fontsize=8.5)

    a = ax[2]
    for gr, col, ls in (("current", "#5C1027", "-"),
                        ("goal", "#D7657F", "--"),
                        ("next_step", "#FCDDE3", ":")):
        g = T[(T.roi == "HC_all") & (T.grouping == gr)]
        v = g.groupby("offset_s").target_z.mean()
        a.plot(v.index * 1000, v.values, ls, color=col, lw=1.8,
               label=gr.replace("_", " "))
    a.axvline(0, color=OBSERVED_VALUE_COLOR, ls=":", lw=1.2)
    a.axhline(0, color="grey", lw=.8)
    a.set_xlabel("time from ripple peak (ms)")
    a.set_ylabel("location signal (target z)")
    a.legend(frameon=False, fontsize=7, title="square grouped by",
             title_fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    a.set_title("hippocampus pooled — all three groupings\n"
                "are noisy at this resolution", loc="left", fontsize=8.5)

    summ = pd.read_csv(os.path.join(src, "timecourse_summary.csv"))
    r = summ[(summ.roi == "HC_all") & (summ.grouping == "current")].iloc[0]
    fig.suptitle(
        "Figure 9 — the peri-ripple second, pooled across the sessions that ran "
        "the same configuration (50 ms windows)\n"
        f"The current-square signal peaks just after the ripple "
        f"({r.peak:+.3f} vs {r['base']:+.3f} baseline) but the rise is NOT "
        f"significant (p = {r.p:.2g}, {int(r.n_configs)} configurations).\n"
        "NOTE: each cell's window comes from a different ripple, so only "
        "ripple-LOCKED structure survives — random-start replay would average "
        "away. EXPLORATORY.", fontsize=9, y=1.16)
    fig.tight_layout()
    save(fig, d, "fig9_peri_ripple_timecourse")
    plt.close(fig)



# ---------------------------------------------------------------- Figures 10-11

PERI_S = 0.25          # He et al. peri-ripple band, marked for reference


def _roles_src():
    return (_latest("ripple_content_roles_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"),
            _latest("ripple_roles_timecourse_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"))


def figure10(d):
    """The walked route — the one ripple-specific candidate, and every control."""
    rsrc, tsrc = _roles_src()
    if not rsrc:
        print("  Fig 10 skipped: no roles results")
        return
    R = pd.read_csv(os.path.join(rsrc, "roles_per_session.csv"))
    base = R[(R.phase == "known") & (R.lat_half == "all")]
    fig, ax = plt.subplots(2, 3, figsize=(25 * CM, 13 * CM))

    # a — per ROI, sessions
    a = ax[0, 0]
    sets = ["HC_all", "HC_mid", "HC_anterior", "mOFC", "mPFC"]
    for i, rn in enumerate(sets):
        v = base[(base.roi == rn) & (base.scheme == "all")
                 & (base.subset == "all")].z_on_route_diff.dropna()
        if not len(v):
            continue
        a.scatter(np.full(len(v), i) + np.random.uniform(-.13, .13, len(v)), v,
                  s=11, color=get_roi_colour(rn if rn != "HC_all" else "HC_mid"),
                  alpha=.55, edgecolors="none")
        a.plot([i - .3, i + .3], [v.mean()] * 2, color=OBSERVED_VALUE_COLOR, lw=2.5)
        a.text(i, a.get_ylim()[1],
               f"p={stats.ttest_1samp(v, 0)[1]:.2g}\nn={len(v)}", ha="center",
               va="bottom", fontsize=6.5)
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(range(len(sets)))
    a.set_xticklabels([roi_display(x) for x in sets], rotation=30, ha="right")
    a.set_ylabel("route: ripple minus flank\n(z vs template-label null)")
    yl = a.get_ylim(); a.set_ylim(yl[0], yl[1] + .35 * (yl[1] - yl[0]))
    a.set_title("a  only HC_mid", loc="left", fontsize=9)

    # b — every control
    a = ax[0, 1]
    conds = [("all", "all", "all trials"), ("all", "correct", "correct only"),
             ("all", "early_known", "early repeats"),
             ("all", "late_known", "late repeats"),
             ("thresh_0.2", "all", "tuned cells only")]
    for i, (sch, sub, lab) in enumerate(conds):
        v = base[(base.roi == "HC_mid") & (base.scheme == sch)
                 & (base.subset == sub)].z_on_route_diff.dropna()
        if not len(v):
            continue
        m, sem = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
        p = stats.ttest_1samp(v, 0)[1]
        a.bar(i, m, .6, yerr=sem, capsize=2, edgecolor="none",
              color=get_roi_colour("HC_mid") if p < .05 else "#CCCCCC")
        a.text(i, m + sem + .04, f"{p:.2g}", ha="center", va="bottom",
               fontsize=6.5)
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(range(len(conds)))
    a.set_xticklabels([c[2] for c in conds], rotation=30, ha="right", fontsize=7)
    a.set_ylabel("route: ripple minus flank (z)")
    a.set_title("b  survives three controls, not the fourth", loc="left",
                fontsize=9)

    # c — THE TEST, visualised: each session's own ripple vs its own flank
    a = ax[0, 2]
    g = base[(base.roi == "HC_mid") & (base.scheme == "all")
             & (base.subset == "all")].dropna(
                 subset=["z_on_route_rip", "z_on_route_flank"])
    for _, row in g.iterrows():
        a.plot([0, 1], [row.z_on_route_flank, row.z_on_route_rip], "-",
               color="#CCB178", alpha=.45, lw=1)
    a.plot([0, 1], [g.z_on_route_flank.mean(), g.z_on_route_rip.mean()], "-o",
           color=OBSERVED_VALUE_COLOR, lw=2.5, ms=6)
    a.axhline(0, color="grey", lw=1)
    a.set_xticks([0, 1]); a.set_xticklabels(["matched flank", "ripple"])
    a.set_xlim(-.3, 1.3)
    a.set_ylabel("route coefficient (z)")
    a.set_title(f"c  THE TEST — paired, n = {len(g)} sessions", loc="left",
                fontsize=9)

    # d — raw role means, carrying the occupancy warning
    a = ax[1, 0]
    if tsrc:
        T = pd.read_csv(os.path.join(tsrc, "roles_timecourse.csv"))
        gg = T[(T.roi == "HC_mid") & (T.phase == "known")
               & (np.abs(T.offset_s) <= PERI_S)]
        names = ["current", "adjacent", "reward", "on_route", "off_route"]
        a.bar(range(len(names)), [gg[n].mean() for n in names], .6,
              yerr=[gg[n].sem() for n in names], capsize=2, edgecolor="none",
              color=["#5C1027", "#8E3550", "#D7657F", "#E9A0B0", "#FCDDE3"])
        a.axhline(0, color="grey", lw=1)
        a.set_xticks(range(len(names)))
        a.set_xticklabels([n.replace("_", " ") for n in names], rotation=30,
                          ha="right", fontsize=7)
        a.set_ylabel("evidence (within-window z)")
    a.set_title("d  RAW: off-route squares score HIGHEST —\n"
                "   rarely visited = noisiest template", loc="left", fontsize=9)

    # e — the descriptive time course, with the reason it looks flat
    a = ax[1, 1]
    if tsrc:
        gg = T[(T.roi == "HC_mid") & (T.phase == "known")]
        a.axvspan(-PERI_S * 1000, PERI_S * 1000, color="#FCDDE3", alpha=.6, lw=0)
        for col, lab, ls in (("on_route", "on the route", "-"),
                             ("off_route", "off the route", "--")):
            v = gg.groupby("offset_s")[col].agg(["mean", "sem"])
            a.plot(v.index * 1000, v["mean"], ls, lw=1.8,
                   color=get_roi_colour("HC_mid"), label=lab)
            a.fill_between(v.index * 1000, v["mean"] - v["sem"],
                           v["mean"] + v["sem"],
                           color=get_roi_colour("HC_mid"), alpha=.18, lw=0)
        a.axvline(0, color=OBSERVED_VALUE_COLOR, ls=":", lw=1.2)
        a.set_xlabel("time from ripple peak (ms)")
        a.set_ylabel("evidence (within-window z)")
        a.legend(frameon=False, fontsize=7)
    a.set_title("e  DESCRIPTIVE — and it is NOT panel c", loc="left", fontsize=9)

    # f — why they differ
    a = ax[1, 2]
    a.axis("off")
    a.text(0, 1.0,
           "Why e looks flat while c is significant\n\n"
           "1. c is a PARTIAL coefficient. It holds adjacency, reward\n"
           "   status, recency, visit count, square identity and\n"
           "   template occupancy fixed. e holds nothing fixed, so it\n"
           "   is dominated by the occupancy bias in d.\n\n"
           "2. The flank in c sits INSIDE the ripple's own occupancy\n"
           "   interval — median dwell 0.367 s — so it is typically\n"
           "   within ~100 ms. The shaded band in e is +-250 ms and\n"
           "   its edges are mostly a DIFFERENT square entirely.\n\n"
           "3. Different widths: c uses each ripple's own duration\n"
           "   (median 60 ms), e a fixed 100 ms so offsets compare.\n\n"
           "4. Different units: c is z against a per-session\n"
           "   permutation null, e is a raw mean across sessions\n"
           "   whose scales differ by orders of magnitude.\n\n"
           "e is here as a descriptive, not as the test.",
           va="top", ha="left", fontsize=7, family="monospace")

    fig.suptitle(
        "Figure 10 — the walked route (I12): a ripple-specific route signal in "
        "HC_mid, once the rewards are known\n"
        "Ripple minus matched flank +0.530 (p = 0.0087, subject-level 0.041); "
        "strongest in the EARLY repeats (+0.578, p = 0.0043), gone by the late "
        "ones (+0.125, p = 0.53).\n"
        "Survives a correct-trials restriction and a template-occupancy "
        "covariate; does NOT survive weighting cells by place-map reliability. "
        "One cell of a ~200-test family. EXPLORATORY.", fontsize=9, y=1.10)
    fig.tight_layout()
    save(fig, d, "fig10_walked_route")
    plt.close(fig)


def figure11(d):
    """Squares where the subject made an error."""
    rsrc, tsrc = _roles_src()
    if not rsrc:
        print("  Fig 11 skipped: no roles results")
        return
    R = pd.read_csv(os.path.join(rsrc, "roles_per_session.csv"))
    R = R[(R.lat_half == "all") & (R.subset == "all") & (R.scheme == "all")]
    fig, ax = plt.subplots(1, 3, figsize=(19 * CM, 6.5 * CM))

    # a — regression, ROI x phase
    a = ax[0]
    sets = ["HC_all", "HC_mid", "HC_anterior"]
    w = .35
    for i, rn in enumerate(sets):
        for pi, ph in enumerate(("explore", "known")):
            v = R[(R.roi == rn) & (R.phase == ph)].z_errors_here_diff.dropna()
            if not len(v):
                continue
            m, sem = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
            a.bar(i + (pi - .5) * w, m, w, yerr=sem, capsize=2, edgecolor="none",
                  color=["#D7657F", "#5C1027"][pi],
                  label=ph if i == 0 else None)
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(range(len(sets)))
    a.set_xticklabels([roi_display(x) for x in sets], rotation=30, ha="right")
    a.set_ylabel("error square: ripple minus flank (z)")
    a.legend(frameon=False, fontsize=7)
    a.set_title("a  suppressed while exploring,\n   elevated once known — n.s.",
                loc="left", fontsize=9)

    # b, c — descriptive time course per ROI
    if tsrc:
        T = pd.read_csv(os.path.join(tsrc, "roles_timecourse.csv"))
        for i, rn in enumerate(("HC_mid", "HC_anterior")):
            a = ax[1 + i]
            a.axvspan(-PERI_S * 1000, PERI_S * 1000, color="#FCDDE3",
                      alpha=.6, lw=0)
            for ph, ls in (("explore", "--"), ("known", "-")):
                g = T[(T.roi == rn) & (T.phase == ph)]
                if not len(g):
                    continue
                v = g.groupby("offset_s").apply(
                    lambda x: (x.error - x.no_error).mean())
                a.plot(v.index * 1000, v.values, ls, lw=1.8,
                       color=get_roi_colour(rn), label=ph)
            a.axhline(0, color="grey", lw=.8)
            a.axvline(0, color=OBSERVED_VALUE_COLOR, ls=":", lw=1.2)
            a.set_xlabel("time from ripple peak (ms)")
            a.set_ylabel("error minus no-error\n(within-window z)")
            a.legend(frameon=False, fontsize=7)
            a.set_title(f"{'bc'[i]}  {roi_display(rn)} — descriptive",
                        loc="left", fontsize=9)

    fig.suptitle(
        "Figure 11 — squares where an erroneous uncover was made (I13)\n"
        "The predicted direction is there — suppressed during exploration, "
        "elevated once everything is known — and it is not significant "
        "(HC_all −0.118 / +0.132; interaction p = 0.19).\n"
        "Shaded band is the He et al. peri-ripple window (±250 ms). "
        "EXPLORATORY.", fontsize=9, y=1.14)
    fig.tight_layout()
    save(fig, d, "fig11_error_squares")
    plt.close(fig)



# ---------------------------------------------------------------- Figures 12-14

TERM_LABEL = {"current": "current square", "adjacent": "a neighbour of it",
              "goal": "reward sought NOW", "goal_1": "reward sought next",
              "goal_2": "two rewards ahead", "next_step": "square stepped to next",
              "known_rew": "reward already found", "unknown_rew": "reward not yet found",
              "reward": "any reward square", "on_route": "on the walked route",
              "errors_here": "errors made here", "visits_here": "times visited",
              "recent": "visited in last 10 s", "train_occ": "template data volume"}


def figure13(d):
    """Reward locations against the current location — raw and ripple-specific."""
    rsrc, _ = _roles_src()
    if not rsrc:
        print("  Fig 13 skipped"); return
    S = pd.read_csv(os.path.join(rsrc, "roles_summary.csv"))
    S = S[(S.scheme == "all") & (S.lat_half == "all") & (S.subset == "all")]
    R = pd.read_csv(os.path.join(rsrc, "roles_per_session.csv"))
    R = R[(R.scheme == "all") & (R.lat_half == "all") & (R.subset == "all")]
    rois = ["HC_all", "HC_mid", "HC_anterior"]
    fig, ax = plt.subplots(1, 3, figsize=(24 * CM, 7 * CM))

    show = [("explore", "current"), ("explore", "known_rew"),
            ("explore", "unknown_rew"), ("known", "current"), ("known", "reward")]
    for pi, (meas, lab, ttl) in enumerate(
            [("z_ripple", "in ripple", "a  what a ripple contains"),
             ("z_diff", "ripple minus matched flank",
              "b  what is ripple-SPECIFIC")]):
        a = ax[pi]
        w = 0.26
        star = []
        for ri, rn in enumerate(rois):
            xs, ys, es, ps = [], [], [], []
            for i, (ph, tm) in enumerate(show):
                v = S[(S.roi == rn) & (S.phase == ph) & (S.term == tm)]
                if not len(v):
                    continue
                col = R[(R.roi == rn) & (R.phase == ph)][
                    f"{'z' if meas == 'z_ripple' else 'z'}_{tm}_"
                    f"{'rip' if meas == 'z_ripple' else 'diff'}"].dropna()
                xs.append(i); ys.append(float(v[meas].iloc[0]))
                es.append(col.std(ddof=1) / np.sqrt(len(col)) if len(col) else 0)
                ps.append(float(v["p_ripple" if meas == "z_ripple"
                                  else "p_diff"].iloc[0]))
            a.bar(np.array(xs) + (ri - 1) * w, ys, w, yerr=es, capsize=2,
                  color=get_roi_colour("HC_mid" if rn == "HC_all" else rn),
                  edgecolor="none", alpha=1 if rn != "HC_all" else .75,
                  label=roi_display(rn))
            star.extend([(x + (ri - 1) * w, yv, pv)
                         for x, yv, pv in zip(xs, ys, ps)])
        a.axhline(0, color="grey", lw=1)
        # headroom first, then put every star in the clear space above
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi + .30 * (hi - lo))
        top = hi + .08 * (hi - lo)
        for x, yv, pv in star:
            if pv < .05:
                a.text(x, top, "*", ha="center", va="bottom", fontsize=11)
        a.set_xticks(range(len(show)))
        a.set_xticklabels([f"{TERM_LABEL[t]}\n({ph})" for ph, t in show],
                          fontsize=6.5, rotation=25, ha="right")
        a.set_ylabel(f"{lab}\n(z vs template-label null)")
        if pi == 0:
            a.legend(frameon=False, fontsize=7, loc="upper right",
                     bbox_to_anchor=(1.0, 0.98))
        a.set_title(ttl, loc="left", fontsize=9)

    # c — the post-uncover question, answered by the design that can carry it
    a = ax[2]
    i11 = _latest("ripple_content_i11_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if i11:
        I = pd.read_csv(os.path.join(i11, "i11_per_session.csv"))
        prim = int(json.load(open(os.path.join(i11, "settings.json")))
                   ["primary_min"])
        I = I[(I.n_triplets >= prim) & (I.scheme == "all")]
        labs = [("z_diff_known_rew_vs_nonrew", "reward found\n- no reward"),
                ("z_diff_future_rew_vs_nonrew", "reward unfound\n- no reward"),
                ("z_diff_known_rew_vs_future_rew", "reward found\n- unfound")]
        g = I[I.roi == "HC_all"]
        lab_c = []
        for i, (col, lab) in enumerate(labs):
            v = g[col].dropna()
            if not len(v):
                continue
            m_, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
            p_ = stats.ttest_1samp(v, 0)[1]
            a.bar(i, m_, .6, yerr=se, capsize=2, edgecolor="none",
                  color="#5C1027" if p_ < .05 else "#D0C7CB")
            lab_c.append((i, p_))
        a.axhline(0, color="grey", lw=1)
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi + .30 * (hi - lo))
        for i, p_ in lab_c:
            a.text(i, hi + .06 * (hi - lo), f"p={p_:.2g}", ha="center",
                   va="bottom", fontsize=7)
        a.set_xticks(range(len(labs)))
        a.set_xticklabels([l for _, l in labs], fontsize=6.5)
        a.set_ylabel("ripple minus flank\n(z vs class-label shuffle)")
    a.set_title("c  ALL exploration ripples, split by what the\n"
                "   square underfoot IS (I11, matched triplets)", loc="left",
                fontsize=9)

    fig.suptitle(
        "Figure 13 — reward locations against the current location\n"
        "a, b use ALL ripples in the phase, wherever the subject was. A ripple "
        "carries the square underfoot (a) but no more than its matched flank "
        "does (b); the rewards are not carried at all.\n"
        "`reward not yet found` is the built-in knowledge control and sits at "
        "zero in b.\n"
        "c is also ALL exploration ripples, classified by the square underfoot, "
        "each matched to one of every other class on dwell and latency. "
        "EXPLORATORY, uncorrected.",
        fontsize=8.5, y=1.16)
    fig.tight_layout()
    save(fig, d, "fig13_reward_vs_current")
    plt.close(fig)


def figure14(d, roi="HC_mid", phase="known"):
    """Every regressor in the model that produced the route result."""
    rsrc, _ = _roles_src()
    if not rsrc:
        print("  Fig 14 skipped"); return
    S = pd.read_csv(os.path.join(rsrc, "roles_summary.csv"))
    S = S[(S.scheme == "all") & (S.lat_half == "all") & (S.subset == "all")
          & (S.roi == roi) & (S.phase == phase)]
    R = pd.read_csv(os.path.join(rsrc, "roles_per_session.csv"))
    R = R[(R.scheme == "all") & (R.lat_half == "all") & (R.subset == "all")
          & (R.roi == roi) & (R.phase == phase)]
    terms = [t for t in ["current", "adjacent", "goal", "goal_1", "goal_2",
                         "next_step", "reward", "on_route", "errors_here",
                         "visits_here", "recent", "train_occ"]
             if (S.term == t).any()]
    fig, ax = plt.subplots(1, 2, figsize=(22 * CM, 7.5 * CM))
    for pi, (mcol, pcol, scol, ttl) in enumerate(
            [("z_ripple", "p_ripple", "rip", "a  in the ripple"),
             ("z_diff", "p_diff", "diff", "b  ripple minus matched flank")]):
        a = ax[pi]
        ys, es, ps = [], [], []
        for t in terms:
            v = S[S.term == t]
            col = R[f"z_{t}_{scol}"].dropna()
            ys.append(float(v[mcol].iloc[0]))
            es.append(col.std(ddof=1) / np.sqrt(len(col)) if len(col) else 0)
            ps.append(float(v[pcol].iloc[0]))
        cols = ["#5C1027" if p_ < .05 else "#D0C7CB" for p_ in ps]
        a.barh(range(len(terms))[::-1], ys, .65, xerr=es, capsize=2,
               color=cols, edgecolor="none")
        for i, (yv, pv) in enumerate(zip(ys, ps)):
            a.text(yv + np.sign(yv) * (es[i] + .04), len(terms) - 1 - i,
                   f"p={pv:.3g}", va="center", fontsize=6,
                   ha="left" if yv > 0 else "right")
        a.axvline(0, color="grey", lw=1)
        a.set_yticks(range(len(terms))[::-1])
        a.set_yticklabels([TERM_LABEL.get(t, t) for t in terms], fontsize=7.5)
        a.set_xlabel("coefficient (z vs template-label null)")
        xl = a.get_xlim(); a.set_xlim(xl[0] - .25, xl[1] + .35)
        a.set_title(ttl, loc="left", fontsize=9)

    fig.suptitle(
        f"Figure 14 — every regressor in the model, {roi_display(roi)}, "
        f"{phase} phase (n = {len(R)} sessions)\n"
        "One regression per ripple window over all nine squares, plus eight "
        "square dummies. Filled bars p < 0.05 uncorrected.\n"
        "Only `on the walked route` is ripple-specific; the current square is "
        "carried strongly but equally by the matched flank.", fontsize=9, y=1.14)
    fig.tight_layout()
    save(fig, d, f"fig14_all_regressors_{roi}_{phase}")
    plt.close(fig)



def figure12(d):
    """Decoding accuracy fails for every signal -- including spikes. The
    continuous estimator does not. That contrast IS the result."""
    src = _latest("decoder_comparison_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    wsrc = _latest("ripple_content_widthsweep_*")
    if not src:
        print("  Fig 12 skipped"); return
    D = pd.read_csv(os.path.join(src, "decoder_comparison.csv"))
    order = ["HFB (high gamma)", "theta", "beta", "ripple band", "all bands",
             "spikes (HC)"]
    order = [o for o in order if (D.signal == o).any()]
    fig, ax = plt.subplots(1, 3, figsize=(24 * CM, 7 * CM))

    # a — accuracy, every signal
    a = ax[0]
    for i, sig in enumerate(order):
        g = D[D.signal == sig]
        col = "#5C1027" if "spike" in sig else get_roi_colour("HC_anterior")
        a.scatter(np.full(len(g), i) + np.random.uniform(-.15, .15, len(g)),
                  g.acc, s=9, color=col, alpha=.4, edgecolors="none")
        a.plot([i - .32, i + .32], [g.acc.mean()] * 2, color=col, lw=3)
    a.axhline(D["null"].mean(), color=OBSERVED_VALUE_COLOR, ls="--", lw=1.5)
    a.text(len(order) - .5, D["null"].mean(), "  chance (1/9)", va="center",
           fontsize=7, color=OBSERVED_VALUE_COLOR)
    a.set_xticks(range(len(order)))
    a.set_xticklabels(order, rotation=30, ha="right", fontsize=7)
    a.set_ylabel("9-way decoding accuracy\n(balanced, leave-one-config-out)")
    a.set_title("a  argmax decoding fails for EVERY signal", loc="left",
                fontsize=9)

    # b — the same spikes, scored the way this project actually scores them
    a = ax[1]
    if wsrc:
        W = pd.read_csv(os.path.join(wsrc, "width_sweep_summary.csv"))
        W = W[W.width == 0.060]
        rr = [r for r in ["HC_mid", "HC_anterior", "mPFC", "mOFC"]
              if (W.roi == r).any()]
        vals = [float(W[W.roi == r].z.iloc[0]) for r in rr]
        ps = [float(W[W.roi == r].p.iloc[0]) for r in rr]
        a.bar(range(len(rr)), vals, .6,
              color=[get_roi_colour(r) if p_ < .05 else "#D0C7CB"
                     for r, p_ in zip(rr, ps)], edgecolor="none")
        for i, (v, p_) in enumerate(zip(vals, ps)):
            a.text(i, v + .03, f"p={p_:.3g}", ha="center", fontsize=6.5)
        a.axhline(0, color="grey", lw=1)
        a.set_xticks(range(len(rr)))
        a.set_xticklabels([roi_display(r) for r in rr], rotation=30, ha="right")
    a.set_ylabel("location signal\n(continuous estimator, z vs null)")
    a.set_title("b  the SAME spikes, continuous read-out", loc="left",
                fontsize=9)

    # c — why argmax fails
    a = ax[2]
    psrc = _latest("ripple_pseudopop_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if psrc:
        M = pd.read_csv(os.path.join(psrc, "pseudopop_evidence_matrix.csv"))
        m = (M[M.roi == "HC_all"].groupby(["true", "pred"]).z_ripple.mean()
             .unstack().to_numpy())
        lim = np.nanmax(np.abs(m))
        im = a.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
        a.set_xticks(range(9)); a.set_xticklabels(range(1, 10), fontsize=7)
        a.set_yticks(range(9)); a.set_yticklabels(range(1, 10), fontsize=7)
        a.set_xlabel("candidate square"); a.set_ylabel("true square")
        cb = fig.colorbar(im, ax=a, fraction=.046, pad=.04)
        cb.set_label("evidence (z)", fontsize=7); cb.ax.tick_params(labelsize=7)
        dg = np.nanmean(np.diag(m)); off = np.nanmean(m[~np.eye(9, dtype=bool)])
        a.set_title(f"c  why: diagonal {dg:+.3f} vs off {off:+.3f}\n"
                    "   — elevated, but neighbours share it", loc="left",
                    fontsize=9)

    fig.suptitle(
        "Figure 12 — no signal in this dataset supports a 9-way location "
        "DECODER, and that is not the same as carrying no location\n"
        "Every signal sits at or below chance (a), spikes included. The same "
        "spikes scored with the continuous estimator carry location clearly "
        "(b).\n"
        "Argmax throws away a graded signal that neighbouring squares share "
        "(c). Consequence: no front end for a within-ripple sequence analysis.",
        fontsize=9, y=1.14)
    fig.tight_layout()
    save(fig, d, "fig12_decoder_comparison")
    plt.close(fig)



def figure15(d, xlim_ms=250, compact=False):
    """SINGLE SUBPANEL: location signal through the peri-ripple second.

    Figure 5's estimator at every offset, so the unit of inference is the
    SESSION (n = 29-44). An earlier version used the pseudo-population, which
    had to pool cells across sessions sharing a configuration and so dropped the
    unit to the CONFIGURATION (n = 8-14), units that share sessions.

    ⚠ The cluster test's family is the FULL +-500 ms window that was analysed.
    The axis is cropped for legibility only; no test was re-run on the crop.
    """
    src = _latest("stage3_timecourse_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 15 skipped: no stage-3 timecourse")
        return
    T = pd.read_csv(os.path.join(src, "stage3_timecourse.csv"))
    S = pd.read_csv(os.path.join(src, "stage3_timecourse_summary.csv"))
    cfile = os.path.join(src, "cluster_extents.csv")
    C = pd.read_csv(cfile) if os.path.exists(cfile) else None
    dfile = os.path.join(src, "ripple_durations_ms.npy")
    dur = np.load(dfile) if os.path.exists(dfile) else None
    bfile = os.path.join(src, "baseline_subtracted_cluster.csv")
    B = pd.read_csv(bfile) if os.path.exists(bfile) else None

    # A 3 x 3 cm panel cannot carry the project's usual 9-11 pt text and still
    # leave plotting area, so the compact variant drops to 7 pt -- Nature's
    # recommended minimum for final size -- and moves the legend and all
    # statistics out of the panel and into the figure legend.
    fs = 7 if compact else None
    if compact:
        plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5,
                             "axes.labelsize": 7, "xtick.labelsize": 6.5,
                             "ytick.labelsize": 6.5, "legend.fontsize": 6})
    fig, a = plt.subplots(figsize=((3.2, 3.2) if compact else
                                   (9.5 * CM, 7.5 * CM)))
    if compact:
        fig.set_size_inches(3.2 * CM / 0.3937 * 0.3937, 3.2 * CM / 0.3937 * 0.3937)
        fig.set_size_inches(3.2 / 2.54, 3.2 / 2.54)
    txt, base_lvl, curves = [], [], {}
    for roin in ("HC_anterior", "HC_mid"):
        g = T[T.roi == roin]
        W = g.pivot_table(index="session", columns="offset_s", values="z").dropna()
        if not len(W):
            continue
        off = W.columns.to_numpy() * 1000
        m = W.mean(0).to_numpy()
        se = (W.std(0, ddof=1) / np.sqrt(len(W))).to_numpy()
        col = get_roi_colour(roin)
        a.plot(off, m, "-", color=col, lw=1.0 if compact else 1.8,
               label=f"{roi_display(roin)} (n = {len(W)})")
        a.fill_between(off, m - se, m + se, color=col, alpha=.18, lw=0)
        base_lvl.append(W.to_numpy()[:, np.abs(W.columns.to_numpy()) >= .35].mean())
        curves[roin] = col
        r = S[S.roi == roin]
        bb = B[B.roi == roin] if B is not None else None
        if len(r):
            r = r.iloc[0]
            line = (f"{roi_display(roin)}: at ripple z = {r.z_at_zero:+.2f}, "
                    f"p = {r.p_at_zero:.3g}")
            if bb is not None and len(bb):
                line += (f";  vs surround p = "
                         f"{float(bb.p_cluster.iloc[0]):.2f} n.s.")
            txt.append(line)

    if base_lvl:
        a.axhline(float(np.mean(base_lvl)), color="grey", ls="--", lw=1)
    a.axvline(0, color=OBSERVED_VALUE_COLOR, ls=":", lw=1.2)
    a.axhline(0, color="grey", lw=.8)
    a.set_xlim(-xlim_ms, xlim_ms)
    lo, hi = a.get_ylim()
    span = hi - lo
    a.set_ylim(lo - .30 * span, hi + .30 * span)
    lo2, hi2 = a.get_ylim()

    # significant cluster, drawn only where it was actually significant
    if C is not None:
        for i, (roin, col) in enumerate(curves.items()):
            cc = C[(C.roi == roin) & C.is_largest & (C.p_cluster < .05)]
            for _, rr in cc.iterrows():
                y = hi2 - (.04 + .06 * i) * span
                a.plot([rr.start_s * 1000, rr.stop_s * 1000], [y, y], "-",
                       color=col, lw=2.5 if compact else 4,
                       solid_capstyle="butt")
                if not compact:
                    a.text(rr.stop_s * 1000 + 8, y, f"p = {rr.p_cluster:.3f}",
                           va="center", fontsize=6.5, color=col)

    # mean ripple extent, so the reader can see how much of the axis IS a ripple
    if dur is not None:
        h = float(np.mean(dur)) / 2.0
        yb = lo2 + .09 * span
        a.plot([-h, h], [yb, yb], "-", color="#333333",
               lw=2.5 if compact else 3.5, solid_capstyle="butt")
        if not compact:
            a.text(0, yb + .035 * span, f"mean ripple {2 * h:.0f} ms",
                   ha="center", va="bottom", fontsize=6.5, color="#333333")

    a.set_xlabel("time from ripple (ms)" if compact
                 else "time from ripple peak (ms)")
    a.set_ylabel("location signal (z)")
    if compact:
        a.set_xticks([-200, 0, 200])
        a.set_yticks([0, 0.5, 1.0])
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    else:
        a.legend(frameon=False, fontsize=7.5, loc="lower left",
                 handlelength=1.4, borderaxespad=0.4)
        a.set_title("Location information in hippocampal ripples", loc="left",
                    fontsize=10.5, pad=8)
        txt.append("bars = cluster-corrected significant encoding; "
                   "test family was the full \u00b1500 ms")
        fig.text(0.0, -0.06, "\n".join(txt), va="top", ha="left", fontsize=7)
    fig.tight_layout(pad=0.3 if compact else 1.08)
    save(fig, d, "fig15_panel_ripple_location_timecourse"
         + ("_3cm" if compact else ""))
    if compact:
        plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                             "axes.labelsize": 9, "xtick.labelsize": 8,
                             "ytick.labelsize": 8, "legend.fontsize": 8})
    out = []
    for roin in curves:
        W = T[T.roi == roin].pivot_table(index="session", columns="offset_s",
                                         values="z").dropna()
        out.append(pd.DataFrame(dict(
            roi=roin, offset_s=W.columns, mean=W.mean(0).to_numpy(),
            sem=(W.std(0, ddof=1) / np.sqrt(len(W))).to_numpy(),
            n_sessions=len(W))))
    pd.concat(out).to_csv(
        os.path.join(d, "fig15_panel_ripple_location_timecourse.csv"),
        index=False)
    plt.close(fig)



if __name__ == "__main__":
    d = fig_dir()
    print("writing figures ...")
    figure1(d); figure2(d); figure2b(d); figure3(d); figure4(d); figure5(d); figure5(d, "thresh_0.2", "_tuned")
    figure6(d); figure6(d, "thresh_0.2", "_tuned")
    figure7(d); figure8(d); figure9(d)
    figure10(d); figure11(d)
    figure12(d); figure13(d); figure15(d)
    figure14(d, 'HC_mid', 'known'); figure14(d, 'HC_all', 'explore')
    print(f"\n-> {d}")
