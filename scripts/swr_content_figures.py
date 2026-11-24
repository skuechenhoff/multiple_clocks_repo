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
                                      LOCATION_COLORS, OBSERVED_VALUE_COLOR,
                                      PHASE_COLORS)

ROIS = ["HC_anterior", "HC_mid", "mPFC", "mOFC", "PCC"]
SEED_FIG = 42     # jitter only; never affects a statistic
CM = 1 / 2.54

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    # Type 42 = TrueType, so text stays selectable/editable text in Illustrator.
    # matplotlib's default (Type 3) is what makes the fonts open as a "weird
    # format" -- some viewers turn them into outlines or lose the glyph names.
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
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


def _phase_contrast():
    """Per-session phase numbers for the REPORTED analysis.

    Prefers `phase_contrast` (ripple window only, which is what Methods
    describes) and falls back to the fixed-window slice of the older
    time-course module so existing figures still render.
    """
    src = _latest("phase_contrast_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if src:
        return pd.read_csv(os.path.join(src, "per_session.csv")), src
    src = _latest("timecourse_by_phase_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        return None, None
    F = pd.read_csv(os.path.join(src, "fixed_window_contrast.csv"))
    return F[F.window == "ripple"], src


SAVE_EXT = ("jpeg",)      # add "pdf" back when figures are final


def save(fig, d, name):
    for ext in SAVE_EXT:
        fig.savefig(os.path.join(d, f"{name}.{ext}"))
    print(f"  {name}." + " / .".join(SAVE_EXT))


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



def figure16(d):
    """State, location, their conjunction, and whether ripples bind them."""
    src = _latest("ripple_state_content_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 16 skipped: no state results")
        return
    R = pd.read_csv(os.path.join(src, "state_per_session.csv"))
    K = pd.read_csv(os.path.join(src, "conjunction_per_session.csv"))
    P = pd.read_csv(os.path.join(src, "coupling_per_session.csv"))
    rois = [r for r in ["HC_all", "HC_mid", "HC_anterior", "mOFC", "mPFC"]
            if (R.roi == r).any()]
    fig, ax = plt.subplots(2, 2, figsize=(19 * CM, 13 * CM))

    def _bars(a, get, labels, cols, ylab, title, hatch=None):
        w = .8 / len(labels)
        for i, lab in enumerate(labels):
            m, e, p = [], [], []
            for rn in rois:
                v = get(rn, lab).dropna()
                m.append(v.mean() if len(v) else np.nan)
                e.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
                p.append(stats.ttest_1samp(v, 0)[1] if len(v) >= 5 else np.nan)
            x = np.arange(len(rois)) + (i - (len(labels) - 1) / 2) * w
            a.bar(x, m, w, yerr=e, capsize=2, color=cols[i], edgecolor="none",
                  label=lab)
            for xx, mm, pp, ee in zip(x, m, p, e):
                if np.isfinite(pp) and pp < .05:
                    a.text(xx, mm + np.sign(mm) * (ee + .04), "*", ha="center",
                           va="bottom" if mm > 0 else "top", fontsize=9)
        a.axhline(0, color="grey", lw=1)
        a.set_xticks(range(len(rois)))
        a.set_xticklabels([roi_display(r) for r in rois], rotation=30,
                          ha="right", fontsize=7)
        a.set_ylabel(ylab)
        a.legend(frameon=False, fontsize=6.5)
        a.set_title(title, loc="left", fontsize=9)

    # a — state, and how it behaves under the two drift controls
    def g_state(rn, lab):
        if lab == "time-matched":
            return R[(R.roi == rn) & (~R.detrend)].get(
                "z_state_timematched", pd.Series(dtype=float))
        return R[(R.roi == rn) & (R.detrend == (lab == "detrended"))].z_state
    _bars(ax[0, 0], g_state, ["raw", "detrended", "time-matched"],
          ["#FCDDE3", "#D7657F", "#5C1027"], "state signal (z vs null)",
          "a  STATE survives both drift controls\n   (detrending makes it "
          "stronger, not weaker)")

    # b — location for comparison, same windows
    def g_loc(rn, lab):
        return R[(R.roi == rn) & (R.detrend == (lab == "detrended"))].z_location
    _bars(ax[0, 1], g_loc, ["raw", "detrended"], ["#7eb1c4", "#0a607a"],
          "location signal (z vs null)",
          "b  LOCATION, same ripples\n   mPFC has state but no location")

    # c — the conjunction test
    a = ax[1, 0]
    terms = [("z_loc_only", "location only", "#0a607a"),
             ("z_state_only", "state only", "#5C1027"),
             ("z_interaction", "INTERACTION", "#DC673E")]
    w = .8 / 3
    for i, (col, lab, c) in enumerate(terms):
        m, e, p = [], [], []
        for rn in rois:
            v = K[K.roi == rn][col].dropna()
            m.append(v.mean() if len(v) else np.nan)
            e.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
            p.append(stats.ttest_1samp(v, 0)[1] if len(v) >= 5 else np.nan)
        x = np.arange(len(rois)) + (i - 1) * w
        a.bar(x, m, w, yerr=e, capsize=2, color=c, edgecolor="none", label=lab)
        for xx, mm, pp, ee in zip(x, m, p, e):
            if np.isfinite(pp) and pp < .05:
                a.text(xx, mm + ee + .04, "*", ha="center", fontsize=9)
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(range(len(rois)))
    a.set_xticklabels([roi_display(r) for r in rois], rotation=30, ha="right",
                      fontsize=7)
    a.set_ylabel("coefficient (z vs label-permutation null)")
    a.legend(frameon=False, fontsize=6.5)
    a.set_title("c  both parts present, conjunction adds nothing\n"
                "   = separable factors, within our power", loc="left",
                fontsize=9)

    # d — binding
    a = ax[1, 1]
    for i, rn in enumerate(rois):
        g = P[P.roi == rn]
        if not len(g):
            continue
        for k, (col, c) in enumerate((("r_ripple", "#5C1027"),
                                      ("r_flank", "#D0C7CB"))):
            a.bar(i + (k - .5) * .35, g[col].mean(), .35,
                  yerr=g[col].std(ddof=1) / np.sqrt(len(g)), capsize=2,
                  color=c, edgecolor="none",
                  label=("ripple" if k == 0 else "matched flank")
                  if i == 0 else None)
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(range(len(rois)))
    a.set_xticklabels([roi_display(r) for r in rois], rotation=30, ha="right",
                      fontsize=7)
    a.set_ylabel("location-state coupling\n(partial r within a window)")
    a.legend(frameon=False, fontsize=6.5)
    v = P[P.roi == "HC_all"].dz.dropna()
    ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(),
                          scale=v.std(ddof=1) / np.sqrt(len(v)))
    a.set_title(f"d  ripples do NOT bind them more\n"
                f"   HC pooled \u0394z = {v.mean():+.3f}, "
                f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]", loc="left", fontsize=9)

    kk = K[K.roi == "HC_all"]
    ci_i = stats.t.interval(0.95, len(kk) - 1, loc=kk.z_interaction.mean(),
                            scale=kk.z_interaction.std(ddof=1) / np.sqrt(len(kk)))
    fig.suptitle(
        "Figure 16 — position in the ABCD sequence, and whether ripples bind it "
        "to location\n"
        "State is carried in ripples and survives both drift controls (a), but "
        "it is carried outside them too and is already known to be "
        "network-wide.\n"
        f"The conjunction adds nothing beyond the two parts (interaction "
        f"{kk.z_interaction.mean():+.2f}, 95% CI [{ci_i[0]:+.2f}, "
        f"{ci_i[1]:+.2f}]) and ripples do not increase their coupling (d). "
        "EXPLORATORY.", fontsize=9, y=1.10)
    fig.tight_layout()
    save(fig, d, "fig16_state_and_binding")
    plt.close(fig)



def figure17(d):
    """The two compositional tests this dataset can support. Both null."""
    src = _latest("ripple_compositional_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 17 skipped: no compositional results")
        return
    S = pd.read_csv(os.path.join(src, "sequence_per_session.csv"))
    X = pd.read_csv(os.path.join(src, "cross_region_per_session.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(25 * CM, 7 * CM))

    # a — forward vs reverse, and why the raw values are both negative
    a = ax[0]
    gaps = sorted(S.gap.unique())
    w = .35
    for i, (col, lab, c) in enumerate((("z_forward", "forward", "#5C1027"),
                                       ("z_reverse", "reverse", "#D7657F"))):
        m = [S[S.gap == g][col].mean() for g in gaps]
        e = [S[S.gap == g][col].sem() for g in gaps]
        a.bar(np.arange(len(gaps)) + (i - .5) * w, m, w, yerr=e, capsize=2,
              color=c, edgecolor="none", label=lab)
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(range(len(gaps)))
    a.set_xticklabels([f"{int(g * 1000)} ms" for g in gaps])
    a.set_xlabel("max gap between paired ripples")
    a.set_ylabel("transition score (z vs pairing shuffle)")
    a.legend(frameon=False, fontsize=7)
    a.set_title("a  both negative — a STRUCTURAL floor.\n"
                "   Residuals sum to zero, so a positive\n"
                "   diagonal forces the off-diagonal down",
                loc="left", fontsize=8.5)

    # b — the contrast that is free of it
    a = ax[1]
    for i, g in enumerate(gaps):
        v = S[S.gap == g].z_fwd_minus_rev.dropna()
        ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(),
                              scale=v.std(ddof=1) / np.sqrt(len(v)))
        a.scatter(np.full(len(v), i) + np.random.uniform(-.12, .12, len(v)), v,
                  s=9, color="#CCB178", alpha=.5, edgecolors="none")
        a.plot([i - .28, i + .28], [v.mean()] * 2, color=OBSERVED_VALUE_COLOR,
               lw=2.5)
        a.plot([i, i], ci, color=OBSERVED_VALUE_COLOR, lw=1.4)
        a.text(i, a.get_ylim()[1],
               f"p={stats.ttest_1samp(v, 0)[1]:.2g}\n{(v > 0).sum()}/{len(v)}",
               ha="center", va="bottom", fontsize=6.5)
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(range(len(gaps)))
    a.set_xticklabels([f"{int(g * 1000)} ms" for g in gaps])
    a.set_xlabel("max gap between paired ripples")
    a.set_ylabel("forward minus reverse (z)")
    yl = a.get_ylim(); a.set_ylim(yl[0], yl[1] + .35 * (yl[1] - yl[0]))
    a.set_title("b  forward > reverse: consistent\n   direction, never significant",
                loc="left", fontsize=8.5)

    # c — does hippocampus lead cortex?
    a = ax[2]
    for pname, col in (("mPFC", get_roi_colour("mPFC")),
                       ("mOFC", get_roi_colour("mOFC"))):
        g = X[X.partner == pname]
        if not len(g):
            continue
        lags, m, e = [], [], []
        for lag in sorted(g.lag_s.unique()):
            if lag == 0:
                continue
            aa = g[(g.lag_s == lag) & (g.direction == "hc_leads")
                   ].set_index("session").r
            bb = g[(g.lag_s == lag) & (g.direction == "pfc_leads")
                   ].set_index("session").r
            j = pd.concat([aa.rename("a"), bb.rename("b")], axis=1).dropna()
            if len(j) < 5:
                continue
            dif = j.a - j.b
            lags.append(lag * 1000); m.append(dif.mean())
            e.append(dif.std(ddof=1) / np.sqrt(len(dif)))
        a.errorbar(lags, m, yerr=e, marker="o", ms=3.5, lw=1.6, capsize=2,
                   color=col, label=f"{roi_display(pname)} (n={len(j)})")
    a.axhline(0, color="grey", lw=1)
    a.set_xlabel("lag from the ripple (ms)")
    a.set_ylabel("HC leads  minus  partner leads\n(partial r)")
    a.legend(frameon=False, fontsize=7)
    a.set_title("c  no directional asymmetry —\n"
                "   hippocampus does not lead cortex", loc="left", fontsize=8.5)

    v = S[S.gap == 0.25].z_fwd_minus_rev.dropna()
    ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(),
                          scale=v.std(ddof=1) / np.sqrt(len(v)))
    fig.suptitle(
        "Figure 17 — the two compositional-replay predictions this dataset CAN "
        "test, since a within-ripple sequence decode is impossible here\n"
        f"Ordered state structure across consecutive ripples, restricted to "
        f"pairs where the participant's own state does NOT change: forward "
        f"minus reverse {v.mean():+.2f}, 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}], "
        f"p = {stats.ttest_1samp(v, 0)[1]:.2g} over "
        f"{int(S[S.gap == 0.25].n_pairs.sum())} pairs.\n"
        "Hippocampus does not lead mPFC or mOFC at any lag. Neither test "
        "supports the account; both bounds are reported rather than a shrug. "
        "EXPLORATORY.", fontsize=8.5, y=1.16)
    fig.tight_layout()
    save(fig, d, "fig17_compositional_tests")
    plt.close(fig)



def figure18(d):
    """The current-square result split by phase and by what the square is."""
    src = _latest("location_by_phase_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 18 skipped: no phase-split results")
        return
    R = pd.read_csv(os.path.join(src, "per_session.csv"))
    N = pd.read_csv(os.path.join(src, "ripple_counts.csv"))
    PH = ["first_traversal", "encode", "execute"]
    PHL = ["first\ntraversal", "encode", "execute"]
    rois = [r for r in ["HC_all", "HC_mid", "HC_anterior"] if (R.roi == r).any()]
    fig, ax = plt.subplots(2, 2, figsize=(19 * CM, 13 * CM))

    def _panel(a, col, subs, cols, labs, ylab, title, roi="HC_all"):
        g0 = R[R.roi == roi]
        w = .8 / len(subs)
        for i, sub in enumerate(subs):
            m, e, p, n = [], [], [], []
            for ph in PH:
                v = g0[(g0.phase == ph) & (g0.subtype == sub)][col].dropna()
                m.append(v.mean() if len(v) else np.nan)
                e.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
                p.append(stats.ttest_1samp(v, 0)[1] if len(v) >= 5 else np.nan)
                n.append(len(v))
            x = np.arange(len(PH)) + (i - (len(subs) - 1) / 2) * w
            a.bar(x, m, w, yerr=e, capsize=2, color=cols[i], edgecolor="none",
                  label=labs[i])
            for xx, mm, pp, ee, nn in zip(x, m, p, e, n):
                if np.isfinite(pp) and pp < .05:
                    a.text(xx, mm + ee + .03, "*", ha="center", fontsize=9)
                if nn < 15:            # too few sessions to read anything into
                    a.text(xx, 0, "  ×", ha="center", va="center", fontsize=7,
                           color="#999999")
        a.axhline(0, color="grey", lw=1)
        a.set_xticks(range(len(PH))); a.set_xticklabels(PHL, fontsize=7.5)
        a.set_ylabel(ylab)
        a.legend(frameon=False, fontsize=6.5)
        a.set_title(title, loc="left", fontsize=9)

    # a — the headline: location in ripples by phase, all three ROIs
    a = ax[0, 0]
    w = .8 / len(rois)
    for i, rn in enumerate(rois):
        g0 = R[(R.roi == rn) & (R.subtype == "all")]
        m, e, p = [], [], []
        for ph in PH:
            v = g0[g0.phase == ph].z_c0.dropna()
            m.append(v.mean()); e.append(v.std(ddof=1) / np.sqrt(len(v)))
            p.append(stats.ttest_1samp(v, 0)[1])
        x = np.arange(len(PH)) + (i - (len(rois) - 1) / 2) * w
        a.bar(x, m, w, yerr=e, capsize=2,
              color=get_roi_colour("HC_mid" if rn == "HC_all" else rn),
              alpha=1 if rn != "HC_all" else .7, edgecolor="none",
              label=roi_display(rn))
        for xx, mm, pp, ee in zip(x, m, p, e):
            if pp < .05:
                a.text(xx, mm + ee + .03, "*", ha="center", fontsize=9)
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(range(len(PH))); a.set_xticklabels(PHL, fontsize=7.5)
    a.set_ylabel("location in ripples\n(C0, z vs permutation null)")
    a.legend(frameon=False, fontsize=6.5)
    a.set_title("a  present during encoding, not detectable\n"
                "   during execution (see b for the contrast)", loc="left",
                fontsize=9)

    # b — THE contrast, unmatched vs count-matched
    a = ax[0, 1]
    mfile = os.path.join(src, "count_matched_contrasts.csv")
    M = pd.read_csv(mfile) if os.path.exists(mfile) else None
    w2 = .38
    for i, (lab, c) in enumerate((("unmatched", "#D0C7CB"),
                                  ("count-matched", "#5C1027"))):
        m, e, p = [], [], []
        for rn in rois:
            if lab == "unmatched":
                t = R[(R.roi == rn) & (R.subtype == "all")].pivot_table(
                    index="session", columns="phase", values="z_c0")
                v = (t["encode"] - t["execute"]).dropna()
            elif M is not None:
                v = M[(M.roi == rn)
                      & (M.pair == "encode_minus_execute")].d_z_c0.dropna()
            else:
                v = pd.Series(dtype=float)
            m.append(v.mean() if len(v) else np.nan)
            e.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
            p.append(stats.ttest_1samp(v, 0)[1] if len(v) >= 5 else np.nan)
        x = np.arange(len(rois)) + (i - .5) * w2
        a.bar(x, m, w2, yerr=e, capsize=2, color=c, edgecolor="none", label=lab)
        for xx, mm, pp, ee in zip(x, m, p, e):
            if np.isfinite(pp):
                a.text(xx, mm + ee + .03, f"{pp:.3f}", ha="center",
                       fontsize=6, rotation=90, va="bottom")
    a.axhline(0, color="grey", lw=1)
    a.set_xticks(range(len(rois)))
    a.set_xticklabels([roi_display(r) for r in rois], rotation=20, ha="right",
                      fontsize=7)
    a.set_ylabel("encode minus execute\n(C0, z; paired within session)")
    yl = a.get_ylim(); a.set_ylim(yl[0], yl[1] + .3 * (yl[1] - yl[0]))
    a.legend(frameon=False, fontsize=6.5)
    a.set_title("b  the contrast. Execute has ~2x the ripples,\n"
                "   which inflates its z; matching counts helps",
                loc="left", fontsize=9)

    # c, d — the two subtype splits SK asked for
    _panel(ax[1, 0], "z_c0", ["reward", "nonreward"],
           ["#5C1027", "#E9A0B0"], ["reward square", "non-reward square"],
           "location in ripples (C0, z)",
           "c  reward vs non-reward square underfoot\n   — no difference in "
           "any phase")
    _panel(ax[1, 1], "z_c0", ["common", "off_route"],
           ["#0a607a", "#b6d4e0"], ["on common trajectory", "off it"],
           "location in ripples (C0, z)",
           "d  on vs off the common trajectory\n   \u00d7 = fewer than 15 "
           "sessions, do not read")

    g = R[(R.roi == "HC_all") & (R.subtype == "all")]
    t = g.pivot_table(index="session", columns="phase", values="z_c0")
    dif = (t["encode"] - t["execute"]).dropna()
    ci = stats.t.interval(0.95, len(dif) - 1, loc=dif.mean(),
                          scale=dif.std(ddof=1) / np.sqrt(len(dif)))
    n_off = int(N[(N.phase == "execute") & (N.subtype == "off_route")].n.sum())
    mm = (M[(M.roi == "HC_all") & (M.pair == "encode_minus_execute")].d_z_c0
          .dropna() if M is not None else pd.Series(dtype=float))
    mp = stats.ttest_1samp(mm, 0)[1] if len(mm) >= 5 else np.nan
    fig.suptitle(
        "Figure 18 — the current-square result, split by encode/execute and by "
        "what the square IS.  `first traversal` is a SUBSET of `encode` (71%), "
        "shown separately, not a third phase\n"
        f"Encode +0.55 (P = 0.0003), execute +0.22 (P = 0.13). The paired "
        f"contrast is {dif.mean():+.2f} (P = "
        f"{stats.ttest_1samp(dif, 0)[1]:.2g}) unmatched and {mm.mean():+.2f} "
        f"(P = {mp:.3f}) once ripple counts are matched — nominally "
        "significant, NOT surviving Bonferroni over the 6-test family.\n"
        f"Neither subtype split separates anything. \u26a0 Off-trajectory "
        f"ripples almost vanish during execution ({n_off} in total), so that "
        "cell is empty rather than null. EXPLORATORY.", fontsize=8.5, y=1.14)
    fig.tight_layout()
    save(fig, d, "fig18_location_by_phase")
    plt.close(fig)



if __name__ == "__main__":
    d = fig_dir()
    print("writing figures ...")
    figure1(d); figure2(d); figure2b(d); figure3(d); figure4(d); figure5(d); figure5(d, "thresh_0.2", "_tuned")
    figure6(d); figure6(d, "thresh_0.2", "_tuned")
    figure7(d); figure8(d); figure9(d)
    figure10(d); figure11(d); figure16(d); figure17(d); figure18(d)
    figure12(d); figure13(d); figure15(d)
    figure14(d, 'HC_mid', 'known'); figure14(d, 'HC_all', 'explore')
    print(f"\n-> {d}")


def figure19(d, xlim_ms=250, smooth=True):
    """SUBPANEL: the peri-ripple location signal, encoding vs once-known.

    Two 3 cm panels side by side, same estimator, colours and frame as the
    reported fig15 panel, so it drops straight in next to it.

    Ripple counts are matched WITHIN session between the two phases before z is
    taken (see `timecourse_by_phase`), because z carries a sqrt(n) term and
    execute has ~2x the ripples. Without matching the right panel is flattered.

    ⚠ The cluster test's family is the full +-500 ms window; the axis is
    cropped for legibility only. Smoothing is display-only (75 ms moving
    average, computed upstream and stored as its own column).
    """
    src = _latest("timecourse_by_phase_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 19 skipped: run `swr_content_explore.py phase_timecourse`")
        return
    T = pd.read_csv(os.path.join(src, "timecourse_by_phase.csv"))
    S = pd.read_csv(os.path.join(src, "cluster_summary.csv"))
    cfile = os.path.join(src, "cluster_extents.csv")
    C = pd.read_csv(cfile) if os.path.exists(cfile) else None
    dfile = _latest("stage3_timecourse_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    dur = None
    if dfile and os.path.exists(os.path.join(dfile, "ripple_durations_ms.npy")):
        dur = np.load(os.path.join(dfile, "ripple_durations_ms.npy"))
    col_z = "z_smooth" if smooth and "z_smooth" in T.columns else "z"
    # panel-h language. ⚠ the colour is the ROI's project colour, so
    # HC_mid IS the gold curve and HC_anterior IS the blue one.
    LAB = {"HC_anterior": "aHP", "HC_mid": "mHP", "HC_all": "HC"}

    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5,
                         "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6})
    fig, ax = plt.subplots(1, 2, figsize=(7.0 / 2.54, 3.4 / 2.54),
                           sharey=True)
    PH = (("encode", "encoding"), ("execute", "once known"))
    txt = []
    for pi, (ph, lab) in enumerate(PH):
        a = ax[pi]
        for roin in ("HC_anterior", "HC_mid"):
            W = (T[(T.roi == roin) & (T.phase == ph)]
                 .pivot_table(index="session", columns="offset_s",
                              values=col_z).dropna())
            if not len(W):
                continue
            off = W.columns.to_numpy() * 1000
            m = W.mean(0).to_numpy()
            se = (W.std(0, ddof=1) / np.sqrt(len(W))).to_numpy()
            col = get_roi_colour(roin)
            a.plot(off, m, "-", color=col, lw=1.0,
                   label=f"{LAB[roin]} (n = {len(W)})")
            a.fill_between(off, m - se, m + se, color=col, alpha=.18, lw=0)
            r = S[(S.roi == roin) & (S.curve == ph)]
            if len(r):
                txt.append(f"{LAB[roin]} {lab}: at ripple z = "
                           f"{r.z_at_zero.iloc[0]:+.2f}, p = "
                           f"{r.p_at_zero.iloc[0]:.3g}; cluster p = "
                           f"{r.p_cluster.iloc[0]:.3f}")
        a.axvline(0, color=OBSERVED_VALUE_COLOR, ls=":", lw=1.0)
        a.axhline(0, color="grey", lw=.8)
        a.set_xlim(-xlim_ms, xlim_ms)
        a.set_xticks([-200, 0, 200])
        a.set_title(lab, fontsize=7.5, pad=3)
        a.set_xlabel("ms from ripple")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel("location signal (z)")
    lo, hi = ax[0].get_ylim()
    span = hi - lo
    ax[0].set_ylim(lo - .22 * span, hi + .26 * span)
    lo2, hi2 = ax[0].get_ylim()

    for pi, (ph, _) in enumerate(PH):
        if C is None:
            continue
        for i, roin in enumerate(("HC_anterior", "HC_mid")):
            cc = C[(C.roi == roin) & (C.curve == ph) & C.is_largest
                   & (C.p_cluster < .05)]
            for _, rr in cc.iterrows():
                ax[pi].plot([rr.start_s * 1000, rr.stop_s * 1000],
                            [hi2 - (.05 + .08 * i) * span] * 2, "-",
                            color=get_roi_colour(roin), lw=2.2,
                            solid_capstyle="butt")
    if dur is not None:
        h = float(np.mean(dur)) / 2.0
        ax[0].plot([-h, h], [lo2 + .07 * span] * 2, "-", color="#333333",
                   lw=2.2, solid_capstyle="butt")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6, ncol=2, handlelength=1.2,
               loc="lower center", bbox_to_anchor=(0.55, -0.10),
               columnspacing=1.2)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    save(fig, d, "fig19_panel_location_timecourse_by_phase")

    for roin in ("HC_anterior", "HC_mid", "HC_all"):
        r = S[(S.roi == roin) & (S.curve == "encode_minus_execute")]
        if len(r):
            txt.append(f"{LAB[roin]} encode-minus-execute: at ripple "
                       f"z = {r.z_at_zero.iloc[0]:+.2f}, p = "
                       f"{r.p_at_zero.iloc[0]:.3g}; cluster p = "
                       f"{r.p_cluster.iloc[0]:.3f}")
    print("   " + "\n   ".join(txt))
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def figure20(d, rois=None, tag=None):
    """The phase effect as bars: why mHP looks stronger but tests weaker.

    Left: mean z per phase per ROI, +- s.e.m. across sessions.
    Right: the paired contrast per session, with every session shown, because
    the whole point is that mHP has the LARGER mean and the LARGER spread.
    """
    F, src = _phase_contrast()
    if F is None:
        print("  Fig 20 skipped: run `swr_content_explore.py phase_contrast`")
        return
    if rois is None:
        rois = ["HC_all", "HC_anterior", "HC_mid"]
    LAB = {"HC_all": "HC pooled", "HC_anterior": "aHP", "HC_mid": "mHP",
           "mPFC": "mPFC", "mOFC": "mOFC"}
    col = {"HC_all": "#4A4A55", "HC_anterior": get_roi_colour("HC_anterior"),
           "HC_mid": get_roi_colour("HC_mid"),
           "mPFC": get_roi_colour("mPFC"), "mOFC": get_roi_colour("mOFC")}

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7})
    fig, ax = plt.subplots(1, 2, figsize=(15 * CM, 6.5 * CM))

    def star(p):
        return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."

    # ── a: each phase against zero
    a, w = ax[0], .36
    suff = "_matched" if all(f"z_{p_}_matched" in F.columns
                             for p_ in ("encode", "execute")) else ""
    for i, ph in enumerate((f"z_encode{suff}", f"z_execute{suff}")):
        m, e, ps = [], [], []
        for r in rois:
            v = F[F.roi == r][ph].dropna()
            m.append(v.mean()); e.append(stats.sem(v))
            ps.append(stats.ttest_1samp(v, 0)[1])
        x = np.arange(len(rois)) + (i - .5) * w
        a.bar(x, m, w, yerr=e, capsize=3,
              color=[col[r] for r in rois], edgecolor="none",
              alpha=1.0 if i == 0 else .35,
              label="encoding" if i == 0 else "once known")
        for xx, mm, ee, pp in zip(x, m, e, ps):
            a.text(xx, mm + ee + .02, star(pp), ha="center", fontsize=7)
    a.axhline(0, color="grey", lw=.8)
    a.set_xticks(range(len(rois)))
    a.set_xticklabels([f"{LAB[r]}\n(n = {F[F.roi == r].shape[0]})" for r in rois])
    a.set_ylabel("location signal at the ripple (z)")
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.16),
             ncol=2, handlelength=1.3)
    a.set_title("a  location signal by phase", loc="left")

    # ── b: the paired contrast, every session shown
    a = ax[1]
    rng_j = np.random.default_rng(SEED_FIG)
    for i, r in enumerate(rois):
        v = F[F.roi == r].d_z.dropna().to_numpy()
        t, p = stats.ttest_1samp(v, 0)
        a.scatter(i + rng_j.uniform(-.17, .17, len(v)), v, s=7,
                  color=col[r], alpha=.35, edgecolor="none", zorder=2)
        a.errorbar(i, v.mean(), yerr=stats.sem(v), fmt="o", ms=6, capsize=4,
                   color=col[r], zorder=3, mec="white", mew=.8)
        a.text(i, a.get_ylim()[1], "", ha="center")
        a.annotate(f"{v.mean():+.2f}\ns.e.m. {stats.sem(v):.2f}\n"
                   f"P = {p:.3f} {star(p)}",
                   (i, 0), xytext=(i, -3.9), ha="center", fontsize=6.8,
                   va="bottom")
    a.axhline(0, color="grey", lw=.8)
    a.set_xticks(range(len(rois)))
    a.set_xticklabels([LAB[r] for r in rois])
    a.set_ylim(-4.6, 4.2)
    a.set_ylabel("encoding minus once known (z)")
    a.set_title("b  the contrast, one dot per session", loc="left")

    fig.suptitle(
        "Figure 20 — mHP has the LARGER effect and the LARGER spread; aHP "
        "reaches significance on n and consistency, not on size.\n"
        "Ripple window, counts matched within session, 1000 draws. Bars/points "
        "are session means; error bars s.e.m. across sessions.",
        fontsize=8, y=1.10)
    fig.tight_layout()
    save(fig, d, "fig20_phase_effect_bars" + (f"_{tag}" if tag else ""))
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def figure21(d, with_pooled=False):
    """SUBPANEL: location signal at the ripple, aHP and mHP across the two phases.

    An interaction-style panel: phase on x, one line per ROI, session-level
    mean +- s.e.m. Replaces the time course wherever the claim being made is
    "present during encoding, absent once known" rather than "and here is when
    within the ripple".

    Ripple window, counts matched within session between phases (1000 draws),
    so the two phases are directly comparable -- see `timecourse_by_phase`.
    """
    F, src = _phase_contrast()
    if F is None:
        print("  Fig 21 skipped: run `swr_content_explore.py phase_contrast`")
        return
    # prefer the COUNT-MATCHED per-phase levels: plotting unmatched levels
    # beside a matched contrast understates the gap, because z carries a
    # sqrt(n) term and execute has ~2x the ripples
    suff = "_matched" if all(f"z_{p_}_matched" in F.columns
                             for p_ in ("encode", "execute")) else ""
    rois = (["HC_all"] if with_pooled else []) + ["HC_anterior", "HC_mid"]
    LAB = {"HC_all": "HC pooled", "HC_anterior": "aHP", "HC_mid": "mHP"}
    col = {"HC_all": "#4A4A55", "HC_anterior": get_roi_colour("HC_anterior"),
           "HC_mid": get_roi_colour("HC_mid")}

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7, "legend.fontsize": 7})
    fig, a = plt.subplots(figsize=(5.6 / 2.54, 5.4 / 2.54))

    def star(p):
        return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."

    dodge = np.linspace(-.09, .09, len(rois)) if len(rois) > 1 else [0]
    sub = []
    for i, r in enumerate(rois):
        g = F[F.roi == r]
        m, e, ps = [], [], []
        for c in (f"z_encode{suff}", f"z_execute{suff}"):
            v = g[c].dropna()
            m.append(v.mean()); e.append(stats.sem(v))
            ps.append(stats.ttest_1samp(v, 0)[1])
        x = np.array([0, 1]) + dodge[i]
        a.plot(x, m, "-", color=col[r], lw=1.4, zorder=2)
        a.errorbar(x, m, yerr=e, fmt="o", ms=5.5, capsize=3, lw=1.2,
                   color=col[r], mec="white", mew=.8, zorder=3,
                   label=f"{LAB[r]} (n = {len(g)})")
        for xx, mm, ee, pp in zip(x, m, e, ps):
            a.text(xx, mm + ee + .045, star(pp), ha="center", fontsize=6.5,
                   color=col[r], va="bottom")
        dz = g.d_z.dropna()
        sub.append(f"{LAB[r]} {stats.ttest_1samp(dz, 0)[1]:.3f}")

    a.axhline(0, color="grey", lw=.8, zorder=1)
    a.set_xticks([0, 1]); a.set_xticklabels(["encoding", "known"])
    a.set_xlim(-.35, 1.35)
    a.set_ylabel("location signal in ripple (z)")
    lo, hi = a.get_ylim(); a.set_ylim(lo - .05 * (hi - lo), hi + .18 * (hi - lo))
    a.set_title("phase contrast  P = " + ", ".join(sub), fontsize=7, pad=4)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.20),
             ncol=len(rois), handlelength=1.1, columnspacing=1.0,
             handletextpad=.4)
    fig.tight_layout(pad=.3)
    save(fig, d, "fig21_panel_phase_interaction"
         + ("_with_pooled" if with_pooled else ""))
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def figure21_compact(d, size_cm=2.5, with_pooled=False, rois=None,
                     tag=None):
    """The phase panel at 2.5 x 2.5 cm, for use as a small subpanel.

    At this size there is no room for a legend or for "n.s." labels, so ROI
    identity is carried by inline coloured labels and only SIGNIFICANT marks
    are drawn. The yellow star marks a significant encode-minus-known contrast
    for that ROI. Everything else is identical to `figure21`.
    """
    F, src = _phase_contrast()
    if F is None:
        print("  Fig 21c skipped: run `swr_content_explore.py phase_contrast`")
        return
    suff = "_matched" if all(f"z_{p_}_matched" in F.columns
                             for p_ in ("encode", "execute")) else ""
    if rois is None:
        rois = (["HC_all"] if with_pooled else []) + ["HC_anterior", "HC_mid"]
    LAB = {"HC_all": "HC", "HC_anterior": "aHP", "HC_mid": "mHP",
           "mPFC": "mPFC", "mOFC": "mOFC"}
    col = {"HC_all": "#4A4A55", "HC_anterior": get_roi_colour("HC_anterior"),
           "HC_mid": get_roi_colour("HC_mid"),
           "mPFC": get_roi_colour("mPFC"), "mOFC": get_roi_colour("mOFC")}
    STAR = "#F7931E"          # project palette 'B'; the contrast marker

    plt.rcParams.update({"font.size": 6, "axes.labelsize": 6,
                         "xtick.labelsize": 6, "ytick.labelsize": 5.5})
    fig, a = plt.subplots(figsize=(size_cm / 2.54, size_cm / 2.54))
    dodge = np.linspace(-.08, .08, len(rois)) if len(rois) > 1 else [0]
    txt = []
    for i, r in enumerate(rois):
        g = F[F.roi == r]
        m, e, ps = [], [], []
        for c in (f"z_encode{suff}", f"z_execute{suff}"):
            v = g[c].dropna()
            m.append(v.mean()); e.append(stats.sem(v))
            ps.append(stats.ttest_1samp(v, 0)[1])
        x = np.array([0, 1]) + dodge[i]
        a.plot(x, m, "-", color=col[r], lw=2.0, zorder=2, solid_capstyle="round")
        a.errorbar(x, m, yerr=e, fmt="o", ms=4.5, capsize=2, elinewidth=1.5,
                   capthick=1.5, color=col[r], mec="white", mew=.7, zorder=3)
        a.text(x[0] - .20, m[0], LAB[r], color=col[r], fontsize=6,
               ha="right", va="center", fontweight="bold")
        dz = g.d_z.dropna()
        p_c = stats.ttest_1samp(dz, 0)[1]
        if p_c < .05:
            # a typeset asterisk, not a marker glyph: it stays selectable text
            # in the PDF and matches the rest of the figure set
            a.text(np.mean(x) + .10, np.mean(m) + .04, "*", color=col[r],
                   fontsize=10, ha="center", va="center", zorder=5)
        txt.append(f"{LAB[r]}: encode {m[0]:+.2f} (P={ps[0]:.3f}), "
                   f"known {m[1]:+.2f} (P={ps[1]:.3f}), "
                   f"contrast {dz.mean():+.2f} (P={p_c:.3f})")

    a.axhline(0, color="grey", lw=.8, zorder=1)
    a.set_xticks([0, 1]); a.set_xticklabels(["encode", "known"])
    a.set_xlim(-.72, 1.28)
    a.set_ylabel("location (z)", labelpad=2)
    lo, hi = a.get_ylim()
    a.set_ylim(min(lo, -.06), hi + .10 * (hi - lo))
    lo2, hi2 = a.get_ylim()
    a.set_yticks([v for v in (-1.0, -0.5, 0, 0.5, 1.0) if lo2 <= v <= hi2])
    a.tick_params(length=2.5, width=.8, pad=1.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_linewidth(.9)
    fig.tight_layout(pad=.15)
    save(fig, d, f"fig21_phase_panel_{size_cm:g}cm"
         + (f"_{tag}" if tag else "")
         + ("_with_pooled" if with_pooled and tag is None else ""))
    print("   " + "\n   ".join(txt))
    print(f"   yellow star = encode-minus-known contrast P < 0.05")
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    plt.close(fig)


def figure22(d, phase="encode"):
    """TEST (b): goal-sequence content in ripples, by how many rewards ahead.

    Left: evidence for the location at goal-lag g. Right: the same minus its
    matched flank (the ripple-specific part). One line per ROI, session mean
    +- s.e.m., stars = P < 0.05 uncorrected.
    """
    src = _latest("goal_lag_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 22 skipped: run `swr_content_explore.py goal_lag`")
        return
    R = pd.read_csv(os.path.join(src, "per_session.csv"))
    rois = ["HC_all", "mPFC", "mOFC"]
    LAB = {"HC_all": "HC pooled", "mPFC": "mPFC", "mOFC": "mOFC"}
    col = {"HC_all": "#4A4A55", "mPFC": get_roi_colour("mPFC"),
           "mOFC": get_roi_colour("mOFC")}
    LAGNAME = ["0\nseeking\nnow", "1\n+1 ahead\n(=3 back)",
               "2\n+2 ahead\n(=2 back)", "3\n+3 ahead\n= JUST\nCOLLECTED"]

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7})
    fig, ax = plt.subplots(1, 2, figsize=(15 * CM, 6.6 * CM), sharey=True)
    for pi, (col_p, ttl) in enumerate(
            ((f"z_{phase}", "a  evidence for the goal-lag location"),
             (f"c1_{phase}", "b  ripple minus its matched flank"))):
        a = ax[pi]
        dodge = np.linspace(-.10, .10, len(rois))
        for i, rn in enumerate(rois):
            m, e, ps = [], [], []
            for g in (0, 1, 2, 3):
                v = R[(R.roi == rn) & (R.lag == g)][col_p].dropna()
                m.append(v.mean()); e.append(stats.sem(v))
                ps.append(stats.ttest_1samp(v, 0)[1] if len(v) >= 5 else np.nan)
            x = np.arange(4) + dodge[i]
            n = R[(R.roi == rn) & (R.lag == 0)][col_p].dropna().shape[0]
            a.errorbar(x, m, yerr=e, fmt="o-", ms=4.5, lw=1.4, capsize=2.5,
                       color=col[rn], mec="white", mew=.7,
                       label=f"{LAB[rn]} (n = {n})")
            for xx, mm, ee, pp in zip(x, m, e, ps):
                if np.isfinite(pp) and pp < .05:
                    off = -(ee + .05) if mm < 0 else (ee + .05)
                    a.text(xx, mm + off, "*", ha="center",
                           va="top" if mm < 0 else "bottom",
                           fontsize=12, color=col[rn])
        a.axhline(0, color="grey", lw=.9)
        a.set_xticks(range(4)); a.set_xticklabels(LAGNAME)
        a.set_xlabel("position in the A-B-C-D cycle relative to the current goal\n"
                     "(the cycle has length 4, so +3 ahead IS 1 back)")
        a.set_title(ttl, loc="left")
        a.set_xlim(-.45, 3.45)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel(f"goal-sequence evidence (z)\n{phase} phase")
    lo, hi = ax[0].get_ylim(); ax[0].set_ylim(lo - .18 * (hi - lo), hi)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(.55, -.10), columnspacing=2.0)
    fig.suptitle(
        "Figure 22 — TEST (b): does a ripple carry the upcoming GOAL "
        "SEQUENCE?   goal_lag[g] = rew[(state−1+g) mod 4], read at the "
        "ripple's own moment.\n"
        "Lag 0 is the reward being sought NOW (it coincides with the square "
        "underfoot only 9.4% of the time, chance 11%) — so it is about "
        "where the subject is heading, not where they are.\n"
        "NOTE: the HC lag-2 negative is NOT explained by current-location "
        "coding: across sessions, current-location z does not predict lag-2 z "
        "(r = −0.09, P = 0.55). It is unexplained. EXPLORATORY.",
        fontsize=7.6, y=1.03)
    fig.tight_layout()
    save(fig, d, f"fig22_goal_lag_{phase}")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def figure23(d):
    """Where does the evidence actually GO? Role-collapsed confusion, per lag.

    Each row is a lag TEMPLATE; each column is the ROLE the scored square plays
    at that moment. The cell is mean evidence. Rows sum to ~0 by construction
    (templates are z-scored to sum to zero across the nine locations), so this
    shows how evidence is DISTRIBUTED, never its total. The diagonal-ish cell
    (template lag g scored at goal g) is the statistic reported elsewhere.
    """
    src = _latest("goal_lag_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src or not os.path.exists(os.path.join(src, "confusion.csv")):
        print("  Fig 23 skipped: no confusion.csv")
        return
    C = pd.read_csv(os.path.join(src, "confusion.csv"))
    R = C[C.kind == "role"]
    roles = ["current", "goal0", "goal1", "goal2", "goal3", "other"]
    RL = ["current", "goal 0\n(seeking)", "goal 1\n(next)",
          "goal 2\n(2 ahead)", "goal 3\n(just got)", "other"]
    rois = ["HC_all", "mPFC", "mOFC"]
    LAB = {"HC_all": "HC pooled", "mPFC": "mPFC", "mOFC": "mOFC"}
    phases = ["first_traversal", "encode_rest", "execute"]
    PL = {"first_traversal": "first traversal", "encode_rest": "rest of encode",
          "execute": "execute"}

    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8,
                         "xtick.labelsize": 6, "ytick.labelsize": 7})
    fig, ax = plt.subplots(len(rois), len(phases),
                           figsize=(19 * CM, 13 * CM))
    vmax = 0.15
    for ri, rn in enumerate(rois):
        for pi, ph in enumerate(phases):
            a = ax[ri, pi]
            g = R[(R.roi == rn) & (R.phase == ph)]
            M = np.full((4, len(roles)), np.nan)
            Pm = np.full((4, len(roles)), np.nan)
            for lag in range(4):
                gg = g[g.lag == lag]
                for ci, r_ in enumerate(roles):
                    v = gg[r_].dropna() if r_ in gg else pd.Series(dtype=float)
                    if len(v) < 5:
                        continue
                    M[lag, ci] = v.mean()
                    Pm[lag, ci] = stats.ttest_1samp(v, 0)[1]
            im = a.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          aspect="auto")
            for i in range(4):
                for j in range(len(roles)):
                    if np.isfinite(Pm[i, j]) and Pm[i, j] < .05:
                        a.text(j, i, "*", ha="center", va="center",
                               fontsize=11, color="k")
            a.set_xticks(range(len(roles)))
            a.set_xticklabels(RL if ri == len(rois) - 1 else [""] * len(roles),
                              rotation=45, ha="right", fontsize=6)
            a.set_yticks(range(4))
            a.set_yticklabels([f"lag {i}" for i in range(4)]
                              if pi == 0 else [""] * 4)
            n = int(g.n.median()) if len(g) else 0
            a.set_title(f"{LAB[rn]} — {PL[ph]}" if ri == 0 or pi == 0
                        else PL[ph], fontsize=7.5, loc="left")
            if pi == 0:
                a.set_ylabel(f"{LAB[rn]}\ntemplate lag", fontsize=7.5)
    fig.colorbar(im, ax=ax, shrink=.5, pad=.02,
                 label="mean evidence (arb.)")
    fig.suptitle(
        "Figure 23 — WHERE DOES THE EVIDENCE GO? Rows = which lag's "
        "template was used; columns = the role the scored square plays.\n"
        "Rows sum to ~0 by construction, so this shows the DISTRIBUTION of "
        "evidence, not its total. * = P < 0.05 uncorrected, 72 cells per "
        "panel-row. EXPLORATORY.",
        fontsize=8, y=1.02)
    save(fig, d, "fig23_confusion_roles")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    plt.close(fig)


def figure24(d):
    """TEST (c): the 12-bin warped future-lag profile, measured in ripples.

    Rows = ROI, columns = phase. Solid = evidence level (count-matched across
    phases); dashed = ripple minus its matched flank, i.e. the ripple-SPECIFIC
    part. Shaded band = the a-priori lag window registered in
    `per_lag_encoding.ROI_PREDICTED_LAGS_DEG` BEFORE this analysis existed.
    """
    src = _latest("warped_lag_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 24 skipped: run `swr_content_explore.py warped_lag`")
        return
    R = pd.read_csv(os.path.join(src, "per_session.csv"))
    rois = ["mPFC", "mOFC", "HC_all"]
    LAB = {"mPFC": "mPFC", "mOFC": "mOFC", "HC_all": "HC pooled"}
    col = {"mPFC": get_roi_colour("mPFC"), "mOFC": get_roi_colour("mOFC"),
           "HC_all": "#4A4A55"}
    AP = {"mPFC": (30, 60), "HC_all": (0, 330)}
    phases = ["first_traversal", "encode_rest", "execute"]
    PL = {"first_traversal": "first traversal", "encode_rest": "rest of encode",
          "execute": "execute"}
    lags = sorted(R.lag.unique())

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 7})
    fig, ax = plt.subplots(len(rois), len(phases), figsize=(18 * CM, 12 * CM),
                           sharex=True, sharey="row")
    for ri, rn in enumerate(rois):
        for pi, ph in enumerate(phases):
            a = ax[ri, pi]
            for cpat, style, lab in ((f"z_{ph}_matched", "-", "level (matched)"),
                                     (f"c1_{ph}", "--", "ripple − flank")):
                m, e, ps = [], [], []
                for lg in lags:
                    g = R[(R.roi == rn) & (R.lag == lg)]
                    v = g[cpat].dropna() if cpat in g else pd.Series(dtype=float)
                    if len(v) < 5:
                        m.append(np.nan); e.append(0); ps.append(np.nan); continue
                    m.append(v.mean()); e.append(stats.sem(v))
                    ps.append(stats.ttest_1samp(v, 0)[1])
                m, e, ps = np.array(m), np.array(e), np.array(ps)
                a.errorbar(lags, m, yerr=e, fmt="o" + style, ms=3, lw=1.2,
                           capsize=1.5, color=col[rn],
                           alpha=1.0 if style == "-" else .55, label=lab)
                for lg, mm, ee, pp in zip(lags, m, e, ps):
                    if np.isfinite(pp) and pp < .05:
                        a.text(lg, mm + (ee + .06) * np.sign(mm or 1), "*",
                               ha="center", fontsize=10, color=col[rn],
                               va="bottom" if (mm or 1) > 0 else "top")
            if rn in AP:
                for v_ in AP[rn]:
                    a.axvspan(v_ - 12, v_ + 12, color="#F7931E", alpha=.16,
                              lw=0, zorder=0)
            a.axhline(0, color="grey", lw=.8)
            a.set_xticks([0, 90, 180, 270])
            if ri == len(rois) - 1:
                a.set_xlabel("future lag θ (deg; 90° = one subgoal)")
            if pi == 0:
                a.set_ylabel(f"{LAB[rn]}\nevidence (z)")
            a.set_title(PL[ph] if ri == 0 else "", loc="left")
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
    ax[0, 0].legend(frameon=False, fontsize=6, loc="upper right")
    fig.suptitle(
        "Figure 24 — TEST (c): the 12-bin future-lag code measured with "
        "RIPPLE SPIKES ONLY. Orange bands = a-priori lags, registered before "
        "this analysis existed.\n"
        "Same subjects/cells as the per-lag paper analysis — this is a "
        "different MEASUREMENT WINDOW (~3% of task time), not a held-out "
        "cohort. * = P < 0.05 uncorrected; 216 tests, ~11 expected by chance, "
        "16 observed. EXPLORATORY.", fontsize=7.5, y=1.02)
    fig.tight_layout()
    save(fig, d, "fig24_warped_lag_profile")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8})
    plt.close(fig)


def figure25(d, xlim_ms=250, tag=None):
    """Time-resolved, both targets, all ROIs, WITH cluster-corrected extents.

    Bars under each curve mark cluster-corrected significant extents (the same
    sign-flip test the reported HC panel uses, 5000 permutations over the full
    +-500 ms window). `xlim_ms=500` gives the uncropped view -- worth looking
    at, because clusters outside +-250 ms exist and a cropped axis hides them.
    """
    import scripts.swr_content_main as mr
    srcs = {t: _latest(f"timecourse_by_phase_{t}_[0-9][0-9][0-9][0-9]-"
                       "[0-9][0-9]-[0-9][0-9]") for t in ("current", "goal0")}
    if not any(srcs.values()):
        print("  Fig 25 skipped: no timecourse_by_phase_<target>_* dirs")
        return
    rois = ["HC_all", "mPFC", "mOFC"]
    LAB = {"HC_all": "HC pooled", "mPFC": "mPFC", "mOFC": "mOFC"}
    col = {"HC_all": "#4A4A55", "mPFC": get_roi_colour("mPFC"),
           "mOFC": get_roi_colour("mOFC")}
    TL = {"current": "current square", "goal0": "next reward (goal 0)"}
    phases = ["encode", "execute"]
    PL = {"encode": "encode", "execute": "once known"}

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 7, "legend.fontsize": 6.5})
    fig, ax = plt.subplots(2, 2, figsize=(15 * CM, 11 * CM),
                           sharex=True, sharey="row")
    txt = []
    for ti, tgt in enumerate(("current", "goal0")):
        src = srcs[tgt]
        for pi, ph in enumerate(phases):
            a = ax[ti, pi]
            if not src:
                a.set_axis_off(); continue
            T = pd.read_csv(os.path.join(src, "timecourse_by_phase.csv"))
            colz = "z_smooth" if "z_smooth" in T.columns else "z"
            bars = []
            for rn in rois:
                W = (T[(T.roi == rn) & (T.phase == ph)]
                     .pivot_table(index="session", columns="offset_s",
                                  values=colz).dropna())
                Wr = (T[(T.roi == rn) & (T.phase == ph)]
                      .pivot_table(index="session", columns="offset_s",
                                   values="z").dropna())
                if not len(W):
                    continue
                off = W.columns.to_numpy() * 1000
                m = W.mean(0).to_numpy()
                se = (W.std(0, ddof=1) / np.sqrt(len(W))).to_numpy()
                a.plot(off, m, "-", color=col[rn], lw=1.3,
                       label=f"{LAB[rn]} (n = {len(W)})")
                a.fill_between(off, m - se, m + se, color=col[rn], alpha=.16,
                               lw=0)
                # cluster test on the UNSMOOTHED per-session curves
                o2 = Wr.columns.to_numpy()
                t_obs, mass, p_cl, thr = mr._cluster_test(
                    Wr.to_numpy(), np.random.default_rng(SEED_FIG),
                    n_clust=5000)
                if p_cl < .05:
                    for e in mr._extents(t_obs, o2, thr, mass):
                        if e["is_largest"]:
                            bars.append((rn, e["start_s"] * 1000,
                                         e["stop_s"] * 1000, p_cl))
                txt.append(f"{TL[tgt]}/{PL[ph]}/{LAB[rn]}: cluster p={p_cl:.3f}")
            lo, hi = a.get_ylim()
            for bi, (rn, x0, x1, p_cl) in enumerate(bars):
                y = hi - (.05 + .07 * bi) * (hi - lo)
                a.plot([x0, x1], [y, y], "-", color=col[rn], lw=3,
                       solid_capstyle="butt")
                a.text(x1 + 8, y, f"p={p_cl:.3f}", fontsize=5.5,
                       color=col[rn], va="center")
            a.axvline(0, color=OBSERVED_VALUE_COLOR, ls=":", lw=1)
            a.axhline(0, color="grey", lw=.8)
            a.set_xlim(-xlim_ms, xlim_ms)
            if ti == 1:
                a.set_xlabel("ms from ripple")
            if pi == 0:
                a.set_ylabel(f"{TL[tgt]}\nevidence (z)")
            if ti == 0:
                a.set_title(PL[ph], loc="left")
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
    ax[0, 1].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Figure 25 \u2014 time-resolved ripple content, both targets, with "
        "cluster-corrected extents (5000 sign-flip permutations over the full "
        "\u00b1500 ms).\n"
        "Bars = largest significant cluster for that ROI. Counts matched "
        "between phases within session. EXPLORATORY.",
        fontsize=7.6, y=1.04)
    fig.tight_layout()
    save(fig, d, "fig25_timecourse_both_targets" + (f"_{tag}" if tag else ""))
    print("   " + "\n   ".join(txt))
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def figure26(d):
    """TEST L1: does the representation step up AFTER a ripple? (It does not.)

    Paired post-minus-pre, 250 ms windows inside one subgoal interval. Filled =
    real ripples, open = sham events (random times in the same subgoal, away
    from ripples). The sham is the point of the figure: where ripple and sham
    agree, the effect is within-subgoal drift, not ripples.
    """
    src = _latest("loading_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 26 skipped: run `swr_content_explore.py loading`")
        return
    R = pd.read_csv(os.path.join(src, "per_session.csv"))
    rois = ["HC_all", "mPFC", "mOFC"]
    LAB = {"HC_all": "HC pooled", "mPFC": "mPFC", "mOFC": "mOFC"}
    col = {"HC_all": "#4A4A55", "mPFC": get_roi_colour("mPFC"),
           "mOFC": get_roi_colour("mOFC")}
    targets = ["goal0", "goal1", "current"]
    TL = {"goal0": "next reward", "goal1": "reward after", "current": "current sq"}
    phases = [("d_first_traversal", "first traversal"),
              ("d_encode_rest", "rest of encode"), ("d_execute", "execute")]

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 7, "legend.fontsize": 6.5})
    fig, ax = plt.subplots(1, len(targets), figsize=(17 * CM, 6 * CM),
                           sharey=True)
    for ti, tgt in enumerate(targets):
        a = ax[ti]
        x = np.arange(len(phases))
        for ri, rn in enumerate(rois):
            for kind, mk, alpha in (("ripple", "o", 1.0), ("sham", "s", .35)):
                m, e, ps = [], [], []
                for c, _ in phases:
                    g = R[(R.roi == rn) & (R.target == tgt)
                          & (R["kind"] == kind)]
                    v = g[c].dropna() if c in g else pd.Series(dtype=float)
                    if len(v) < 5:
                        m.append(np.nan); e.append(0); ps.append(np.nan); continue
                    m.append(v.mean()); e.append(stats.sem(v))
                    ps.append(stats.ttest_1samp(v, 0)[1])
                xx = x + (ri - 1) * .22 + (0 if kind == "ripple" else .09)
                a.errorbar(xx, m, yerr=e, fmt=mk, ms=4, capsize=2, lw=1,
                           color=col[rn], alpha=alpha, mfc=col[rn] if
                           kind == "ripple" else "white",
                           label=f"{LAB[rn]} {kind}" if ti == 0 else None)
                for x_, mm, ee, pp in zip(xx, m, e, ps):
                    if np.isfinite(pp) and pp < .05:
                        a.text(x_, mm - ee - .08, "*", ha="center", va="top",
                               fontsize=10, color=col[rn])
        a.axhline(0, color="grey", lw=.9)
        a.set_xticks(x); a.set_xticklabels([p[1] for p in phases],
                                           rotation=20, ha="right")
        a.set_title(TL[tgt], loc="left")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel("post minus pre (z)\npositive = loading")
    ax[0].legend(frameon=False, ncol=2, fontsize=5.5, loc="lower left")
    fig.suptitle(
        "Figure 26 — TEST L1: no loading. Post-minus-pre across a ripple, "
        "250 ms windows inside one subgoal interval, paired within event.\n"
        "Filled = ripples, open = SHAM events at random times in the same "
        "subgoal. Where the two agree the effect is within-subgoal drift, not "
        "ripples — which is what mPFC/next-reward/execute shows "
        "(−0.50 ripple vs −0.39 sham). 72 tests. EXPLORATORY.",
        fontsize=7.5, y=1.06)
    fig.tight_layout()
    save(fig, d, "fig26_loading_prepost")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def figure27(d):
    """SK's loading model tested directly: rewards indexed by discoveries-ago.

    The model predicts that during the FIRST TRAVERSAL all of back[1..3] are
    represented CONCURRENTLY in mPFC -- that is a pattern, not a single cell,
    so the figure shows all three together rather than a best point.
    """
    src = _latest("discovery_lag_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 27 skipped: run `swr_content_explore.py discovery_lag`")
        return
    R = pd.read_csv(os.path.join(src, "per_session.csv"))
    rois = ["mPFC", "mOFC", "HC_all"]
    LAB = {"mPFC": "mPFC", "mOFC": "mOFC", "HC_all": "HC pooled"}
    col = {"mPFC": get_roi_colour("mPFC"), "mOFC": get_roi_colour("mOFC"),
           "HC_all": "#4A4A55"}
    cols = [("z_first", "first traversal"),
            ("c1_first", "first traversal,\nripple − flank"),
            ("z_later", "later traversals")]

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 6.5})
    fig, ax = plt.subplots(1, len(cols), figsize=(16 * CM, 6 * CM), sharey=True)
    for ci, (c, ttl) in enumerate(cols):
        a = ax[ci]
        for ri, rn in enumerate(rois):
            m, e, ps, ns = [], [], [], []
            for bk in (1, 2, 3):
                g = R[(R.roi == rn) & (R.back == bk)]
                v = g[c].dropna() if c in g else pd.Series(dtype=float)
                if len(v) < 5:
                    m.append(np.nan); e.append(0); ps.append(np.nan); ns.append(0)
                    continue
                m.append(v.mean()); e.append(stats.sem(v))
                ps.append(stats.ttest_1samp(v, 0)[1]); ns.append(len(v))
            x = np.arange(3) + (ri - 1) * .14
            a.errorbar(x, m, yerr=e, fmt="o-", ms=4.5, lw=1.3, capsize=2.5,
                       color=col[rn],
                       label=f"{LAB[rn]} (n = {ns[0]})" if ci == 0 else None)
            for xx, mm, ee, pp in zip(x, m, e, ps):
                if np.isfinite(pp) and pp < .05:
                    off = -(ee + .06) if mm < 0 else (ee + .06)
                    a.text(xx, mm + off, "*", ha="center",
                           va="top" if mm < 0 else "bottom", fontsize=11,
                           color=col[rn])
        a.axhline(0, color="grey", lw=.9)
        a.set_xticks(range(3))
        a.set_xticklabels(["1\njust found", "2\none before", "3\ntwo before"])
        a.set_xlabel("discovered how many discoveries ago")
        a.set_title(ttl, loc="left", fontsize=7.5)
        a.set_xlim(-.45, 2.45)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel("evidence for that reward (z)")
    ax[0].legend(frameon=False, loc="lower left")
    fig.suptitle(
        "Figure 27 — the loading model tested directly: is each "
        "ALREADY-DISCOVERED reward represented at its recency index?\n"
        "Prediction was that mPFC shows all three POSITIVE and concurrently "
        "during the first traversal. It does not (+0.12, −0.12, −0.24; "
        "all n.s.). Every significant cell is NEGATIVE and is what "
        "current-location coding produces for other squares. 36 tests. "
        "EXPLORATORY.", fontsize=7.5, y=1.06)
    fig.tight_layout()
    save(fig, d, "fig27_discovery_lag")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def figure28(d):
    """The intention test: is the square ABOUT TO BE UNCOVERED represented?

    Locked to the uncover, not to ripples. `strict` keeps only events where
    that square is NEITHER the one the subject already stands on NOR the true
    goal -- so neither current-location leakage nor goal coding can produce it.
    """
    src = _latest("intention_ripple_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 28 skipped: run `swr_content_explore.py intention_ripple`")
        return
    R = pd.read_csv(os.path.join(src, "per_session.csv"))
    rois = ["HC_all", "mPFC", "mOFC"]
    LAB = {"HC_all": "HC pooled", "mPFC": "mPFC", "mOFC": "mOFC"}
    col = {"HC_all": "#4A4A55", "mPFC": get_roi_colour("mPFC"),
           "mOFC": get_roi_colour("mOFC")}
    phases = ["all", "encode", "execute"]
    PL = {"all": "all", "encode": "encode", "execute": "once known"}

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 6.5})
    fig, ax = plt.subplots(1, len(rois), figsize=(16 * CM, 6.2 * CM),
                           sharey=True)
    for ri, rn in enumerate(rois):
        a = ax[ri]
        for mi, (c, lab, ls) in enumerate(
                (("z_intent", "all uncovers", "-"),
                 ("z_intent_strict", "strict: not current, not goal", "--"))):
            m, e, ps, ns = [], [], [], []
            for ph in phases:
                g = R[(R.roi == rn) & (R.phase == ph)]
                v = g[c].dropna() if c in g else pd.Series(dtype=float)
                if len(v) < 5:
                    m.append(np.nan); e.append(0); ps.append(np.nan); ns.append(0)
                    continue
                m.append(v.mean()); e.append(stats.sem(v))
                ps.append(stats.ttest_1samp(v, 0)[1]); ns.append(len(v))
            x = np.arange(len(phases)) + (mi - .5) * .14
            a.errorbar(x, m, yerr=e, fmt="o" + ls, ms=4.5, lw=1.3, capsize=2.5,
                       color=col[rn], alpha=1.0 if mi == 0 else .6,
                       mfc=col[rn] if mi == 0 else "white", label=lab)
            for xx, mm, ee, pp in zip(x, m, e, ps):
                if np.isfinite(pp) and pp < .05:
                    a.text(xx, mm + ee + .05, "*", ha="center", va="bottom",
                           fontsize=11, color=col[rn])
        a.axhline(0, color="grey", lw=.9)
        a.set_xticks(range(len(phases)))
        a.set_xticklabels([PL[p_] for p_ in phases])
        n0 = R[(R.roi == rn) & (R.phase == "all")].shape[0]
        a.set_title(f"{LAB[rn]} (n = {n0})", loc="left")
        a.set_xlim(-.45, len(phases) - .55)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel("evidence for the square about\nto be uncovered (z)")
    ax[0].legend(frameon=False, loc="lower left")
    fig.suptitle(
        "Figure 28 \u2014 locked to the UNCOVER, not the ripple: before the "
        "press, is the square about to be uncovered represented?\n"
        "Dashed = strict (that square is neither the one underfoot nor the "
        "true goal), so neither current-location leakage nor goal coding can "
        "explain it.\n"
        "\u26a0 Even strict does not control spatial PROXIMITY \u2014 the next "
        "square is usually adjacent, and adjacent templates correlate. "
        "EXPLORATORY.", fontsize=7.5, y=1.10)
    fig.tight_layout()
    save(fig, d, "fig28_intention")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def figure29(d):
    """mPFC holds a newly discovered reward location -- only when it is new.

    a: evidence for the just-uncovered square, by what the uncover WAS. In all
       three the subject is standing on that square during the window, so
       "codes where you are" predicts all three equal.
    b: the same split by whether a ripple followed. The signal is WEAKER with a
       ripple, so ripples are not the delivery mechanism.
    """
    src = _latest("consolidation_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 29 skipped: run `swr_content_explore.py consolidation`")
        return
    R = pd.read_csv(os.path.join(src, "per_session.csv"))
    R = R[R.phase == "all"]
    rois = ["mPFC", "mOFC", "HC_all"]
    LAB = {"mPFC": "mPFC", "mOFC": "mOFC", "HC_all": "HC pooled"}
    col = {"mPFC": get_roi_colour("mPFC"), "mOFC": get_roi_colour("mOFC"),
           "HC_all": "#4A4A55"}
    evs = ["first_discovery", "repeat_correct", "incorrect"]
    EL = ["first discovery\n(new reward)", "correct,\nalready known",
          "incorrect\nuncover"]

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 6.5})
    fig, ax = plt.subplots(1, 2, figsize=(15 * CM, 6.4 * CM), sharey=True)

    a = ax[0]
    for ri, rn in enumerate(rois):
        m, e, ps = [], [], []
        for ev in evs:
            v = R[(R.roi == rn) & (R.event == ev)].z_all.dropna()
            if len(v) < 5:
                m.append(np.nan); e.append(0); ps.append(np.nan); continue
            m.append(v.mean()); e.append(stats.sem(v))
            ps.append(stats.ttest_1samp(v, 0)[1])
        x = np.arange(len(evs)) + (ri - 1) * .16
        n = R[(R.roi == rn) & (R.event == "first_discovery")].shape[0]
        a.errorbar(x, m, yerr=e, fmt="o-", ms=5, lw=1.4, capsize=3,
                   color=col[rn], label=f"{LAB[rn]} (n = {n})")
        for xx, mm, ee, pp in zip(x, m, e, ps):
            if np.isfinite(pp) and pp < .05:
                a.text(xx, mm + ee + .05, "*", ha="center", va="bottom",
                       fontsize=12, color=col[rn])
    a.axhline(0, color="grey", lw=.9)
    a.set_xticks(range(len(evs))); a.set_xticklabels(EL)
    a.set_ylabel("evidence for the square\njust uncovered (z)")
    a.set_title("a  only a NEW reward is held", loc="left")
    a.legend(frameon=False, loc="lower left")
    a.set_xlim(-.5, len(evs) - .5)

    a = ax[1]
    for ri, rn in enumerate(rois):
        for ki, (c, lab, mk) in enumerate(((f"z_without", "no ripple", "o"),
                                           (f"z_with", "ripple", "s"))):
            m, e, ps = [], [], []
            for ev in evs:
                v = R[(R.roi == rn) & (R.event == ev)][c].dropna()
                if len(v) < 5:
                    m.append(np.nan); e.append(0); ps.append(np.nan); continue
                m.append(v.mean()); e.append(stats.sem(v))
                ps.append(stats.ttest_1samp(v, 0)[1])
            x = np.arange(len(evs)) + (ri - 1) * .18 + (ki - .5) * .08
            a.errorbar(x, m, yerr=e, fmt=mk, ms=4, capsize=2, lw=1,
                       color=col[rn], mfc=col[rn] if ki else "white",
                       label=f"{LAB[rn]}, {lab}" if ri == 0 or True else None)
    a.axhline(0, color="grey", lw=.9)
    a.set_xticks(range(len(evs))); a.set_xticklabels(EL)
    a.set_title("b  weaker WITH a ripple, not stronger", loc="left")
    a.legend(frameon=False, fontsize=5.5, ncol=3, loc="lower left")
    a.set_xlim(-.5, len(evs) - .5)
    for aa in ax:
        for sp in ("top", "right"):
            aa.spines[sp].set_visible(False)
    fig.suptitle(
        "Figure 29 — after an uncover, does PFC hold the square that was "
        "just uncovered?  Window t+50 ms to t+1 s.\n"
        "The subject stands on that square in ALL THREE event types, so "
        "current-location coding predicts all three equal. mPFC first "
        "discovery vs repeat correct: +0.77, P = 0.0035.\n"
        "⚠ first discoveries also differ in trial number and novelty of "
        "the PLACE, not only of the reward. EXPLORATORY.",
        fontsize=7.5, y=1.10)
    fig.tight_layout()
    save(fig, d, "fig29_consolidation")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def figure30(d, xlim_ms=500):
    """The newly-discovered-reward signal, aligned to ripples. It does not move.

    Ripples between a first discovery and the next uncover; target is the
    square whose reward was just discovered. If a ripple delivered that
    location the curve would rise after 0; if it interrupted it, dip. Neither
    happens -- shown because a flat curve IS the result here, and a null is
    only informative if you can see the error bars.
    """
    src = _latest("discovery_locked_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 30 skipped: run `swr_content_explore.py discovery_locked`")
        return
    T = pd.read_csv(os.path.join(src, "timecourse.csv"))
    S = pd.read_csv(os.path.join(src, "summary.csv"))
    rois = ["mPFC", "mOFC", "HC_all"]
    LAB = {"mPFC": "mPFC", "mOFC": "mOFC", "HC_all": "HC pooled"}
    col = {"mPFC": get_roi_colour("mPFC"), "mOFC": get_roi_colour("mOFC"),
           "HC_all": "#4A4A55"}

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 7, "legend.fontsize": 6.5})
    fig, ax = plt.subplots(1, len(rois), figsize=(16 * CM, 6 * CM), sharey=True)
    for ri, rn in enumerate(rois):
        a = ax[ri]
        W = (T[T.roi == rn].pivot_table(index="session", columns="offset_s",
                                        values="z").dropna())
        if not len(W):
            a.set_axis_off(); continue
        off = W.columns.to_numpy() * 1000
        m = W.mean(0).to_numpy()
        se = (W.std(0, ddof=1) / np.sqrt(len(W))).to_numpy()
        a.plot(off, m, "-", color=col[rn], lw=1.4)
        a.fill_between(off, m - se, m + se, color=col[rn], alpha=.2, lw=0)
        a.axvline(0, color=OBSERVED_VALUE_COLOR, ls=":", lw=1.1)
        a.axhline(0, color="grey", lw=.9)
        a.set_xlim(-xlim_ms, xlim_ms)
        a.set_xlabel("ms from ripple")
        g = S[S.roi == rn]
        sub = ""
        if len(g):
            g = g.iloc[0]
            sub = (f"at ripple {g.z_at_ripple:+.2f} (P={g.p_at_ripple:.2f})\n"
                   f"post−pre {g.post_minus_pre:+.2f} "
                   f"(P={g.p_pp:.2f}); cluster P={g.p_cluster:.2f}")
        a.set_title(f"{LAB[rn]} (n = {len(W)})\n{sub}", loc="left", fontsize=7)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel("evidence for the square whose\nreward was just discovered (z)")
    med = int(T.n_events.median())
    fig.suptitle(
        "Figure 30 — does the newly-discovered-reward signal change around "
        f"a ripple? No. Ripples between a first discovery and the next uncover "
        f"(median {med} per session).\n"
        "A delivering ripple predicts a rise after 0; an interrupting one a "
        "dip. Neither occurs, and no cluster survives in any ROI. "
        "NULL RESULT, shown with its error bars. EXPLORATORY.",
        fontsize=7.5, y=1.08)
    fig.tight_layout()
    save(fig, d, "fig30_discovery_locked")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def figure31(d):
    """METHOD CHECK: the estimator recovers HC current location OUTSIDE ripples.

    Same templates, same statistic, window width matched to the median ripple
    and window COUNT matched between the two types -- so neither can explain a
    difference. If the non-ripple bars were flat, every null in this project
    would be uninterpretable; they are not.
    """
    src = _latest("method_check_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 31 skipped: run `swr_content_explore.py method_check`")
        return
    R = pd.read_csv(os.path.join(src, "per_session.csv"))
    rois = ["HC_all", "HC_anterior", "HC_mid", "mPFC", "mOFC"]
    LAB = {"HC_all": "HC pooled", "HC_anterior": "aHP", "HC_mid": "mHP",
           "mPFC": "mPFC", "mOFC": "mOFC"}
    col = {"HC_all": "#4A4A55", "HC_anterior": get_roi_colour("HC_anterior"),
           "HC_mid": get_roi_colour("HC_mid"), "mPFC": get_roi_colour("mPFC"),
           "mOFC": get_roi_colour("mOFC")}
    phases = ["encode", "execute"]
    PL = {"encode": "encode", "execute": "execute (once known)"}

    def star(p):
        return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8.5,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7})
    fig, ax = plt.subplots(1, 2, figsize=(16 * CM, 6.6 * CM), sharey=True)
    w = .36
    for pi, ph in enumerate(phases):
        a = ax[pi]
        for wi, (c, lab, alpha) in enumerate(
                (("z_nonripple", "NON-ripple windows", .45),
                 ("z_ripple", "ripple windows", 1.0))):
            m, e, ps = [], [], []
            for rn in rois:
                v = R[(R.roi == rn) & (R.phase == ph)][c].dropna()
                if len(v) < 5:
                    m.append(np.nan); e.append(0); ps.append(np.nan); continue
                m.append(v.mean()); e.append(stats.sem(v))
                ps.append(stats.ttest_1samp(v, 0)[1])
            x = np.arange(len(rois)) + (wi - .5) * w
            a.bar(x, m, w, yerr=e, capsize=2.5, edgecolor="none",
                  color=[col[rn] for rn in rois], alpha=alpha, label=lab)
            for xx, mm, ee, pp in zip(x, m, e, ps):
                if np.isfinite(pp) and pp < .05:
                    a.text(xx, mm + ee + .02, star(pp), ha="center",
                           va="bottom", fontsize=8)
        a.axhline(0, color="grey", lw=.9)
        a.set_xticks(range(len(rois)))
        a.set_xticklabels([LAB[rn] for rn in rois], rotation=20, ha="right")
        a.set_title(PL[ph], loc="left")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel("current-location evidence (z)")
    h, l = ax[0].get_legend_handles_labels()
    import matplotlib.patches as mpatches
    h = [mpatches.Patch(facecolor="#4A4A55", alpha=.45),
         mpatches.Patch(facecolor="#4A4A55", alpha=1.0)]
    ax[0].legend(h, l, frameon=False, loc="upper right")
    fig.suptitle(
        "Figure 31 — METHOD CHECK: the same estimator applied to ripple "
        "windows and to tiled NON-ripple windows (≥1 s from any ripple).\n"
        "Window width matched to the median ripple; window COUNT matched "
        "between types, so neither width nor n can explain a difference.\n"
        "Hippocampus encodes the current location during EXECUTE outside "
        "ripples (+0.145, P = 0.043) — the published result is recovered. "
        "Ripple minus non-ripple is n.s. in execute (+0.06, P = 0.68).",
        fontsize=7.6, y=1.11)
    fig.tight_layout()
    save(fig, d, "fig31_method_check")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def figure32(d):
    """Do the per-lag-encoding tags carry information? Two tests.

    a: does a cell fire more at its PREFERRED SQUARE (chosen on training
       configurations), split by whether its preferred lag is 0?
    b: is a cell more ripple-driven, by lag group and phase?
    c: the same modulation against all 12 preferred lags -- a real lag code
       should be a smooth harmonic, not a lag-0-vs-rest step.
    """
    s1 = _latest("preferred_lag_firing_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    s2 = _latest("lag_group_ripple_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not s1 or not s2:
        print("  Fig 32 skipped: run `preferred_lag` and `lag_group`")
        return
    C = pd.read_csv(os.path.join(s1, "per_cell.csv"))
    M = pd.read_csv(os.path.join(s2, "per_cell.csv"))
    rois = ["HC", "mPFC", "mOFC"]
    col = {"HC": "#4A4A55", "mPFC": get_roi_colour("mPFC"),
           "mOFC": get_roi_colour("mOFC")}
    phases = ["encode", "execute"]

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 6.5})
    fig, ax = plt.subplots(1, 3, figsize=(19 * CM, 6.2 * CM))

    # a -- preferred-square selectivity, ordinary task time
    a = ax[0]
    for ri, rn in enumerate(rois):
        for gi, (lab, sel) in enumerate((("lag 0", True), ("lag != 0", False))):
            m, e, ps = [], [], []
            for ph in phases:
                g = C[(C.roi == rn) & (C.phase == ph) & (C.window == "all")]
                v = (g[g.pref_lag == 0] if sel else g[g.pref_lag != 0]).sel.dropna()
                if len(v) < 5:
                    m.append(np.nan); e.append(0); ps.append(np.nan); continue
                m.append(v.mean()); e.append(stats.sem(v))
                ps.append(stats.ttest_1samp(v, 0)[1])
            x = np.arange(len(phases)) + (ri - 1) * .22 + (gi - .5) * .09
            a.errorbar(x, m, yerr=e, fmt="o" if gi == 0 else "s", ms=4,
                       capsize=2, lw=1, color=col[rn],
                       mfc=col[rn] if gi == 0 else "white",
                       label=f"{rn}, {lab}")
            for xx, mm, ee, pp in zip(x, m, e, ps):
                if np.isfinite(pp) and pp < .05:
                    a.text(xx, mm + ee + .004, "*", ha="center", va="bottom",
                           fontsize=10, color=col[rn])
    a.axhline(0, color="grey", lw=.9)
    a.set_xticks(range(len(phases))); a.set_xticklabels(phases)
    a.set_ylabel("selectivity for the\npreferred square")
    a.set_title("a  fires at its preferred square?", loc="left")
    a.legend(frameon=False, fontsize=5.5, ncol=2, loc="lower left")

    # b -- ripple modulation by lag group
    a = ax[1]
    for ri, rn in enumerate(rois):
        for gi, (lab, sel) in enumerate((("lag 0", True), ("lag != 0", False))):
            m, e, ps = [], [], []
            for ph in phases:
                g = M[(M.roi == rn) & (M.phase == ph)]
                v = (g[g.pref_lag == 0] if sel else g[g.pref_lag != 0]).ripmod.dropna()
                if len(v) < 5:
                    m.append(np.nan); e.append(0); ps.append(np.nan); continue
                m.append(v.mean()); e.append(stats.sem(v))
                ps.append(stats.ttest_1samp(v, 0)[1])
            x = np.arange(len(phases)) + (ri - 1) * .22 + (gi - .5) * .09
            a.errorbar(x, m, yerr=e, fmt="o" if gi == 0 else "s", ms=4,
                       capsize=2, lw=1, color=col[rn],
                       mfc=col[rn] if gi == 0 else "white")
            for xx, mm, ee, pp in zip(x, m, e, ps):
                if np.isfinite(pp) and pp < .05:
                    a.text(xx, mm + ee + .006, "*", ha="center", va="bottom",
                           fontsize=10, color=col[rn])
    a.axhline(0, color="grey", lw=.9)
    a.set_xticks(range(len(phases))); a.set_xticklabels(phases)
    a.set_ylabel("ripple modulation\n(in − out)/(in + out)")
    a.set_title("b  more driven by ripples?", loc="left")

    # c -- continuous, all 12 lags
    a = ax[2]
    for rn in rois:
        g = M[(M.roi == rn) & (M.phase == "execute")]
        prof = g.groupby("pref_lag").ripmod.agg(["mean", "sem"])
        a.errorbar(prof.index, prof["mean"], yerr=prof["sem"], fmt="o-",
                   ms=3, lw=1, capsize=1.5, color=col[rn], label=rn)
    a.axhline(0, color="grey", lw=.9)
    a.set_xticks([0, 90, 180, 270])
    a.set_xlabel("preferred lag (deg)")
    a.set_ylabel("ripple modulation")
    a.set_title("c  ripple-drive is not organised by lag", loc="left")
    a.legend(frameon=False, loc="lower right")
    for aa in ax:
        for sp in ("top", "right"):
            aa.spines[sp].set_visible(False)
    fig.suptitle(
        "Figure 32 — do the per-lag-encoding tags carry information? "
        "Preferred lag = argmax over the 12 stored r values, ALL 955 cells, no "
        "significance filter.\n"
        "a: HC cells tagged lag-0 are location-selective during execute "
        "(+0.051, P = 0.0011) and other-lag cells are not (difference +0.055, "
        "P = 0.0009) — the tag is real.\n"
        "⚠ the tag is a noisy argmax over 12 CORRELATED values and is "
        "near-uniform across cells (64-94 per bin, 80 by chance); mPFC has only "
        "9 lag-0 cells. EXPLORATORY.", fontsize=7.4, y=1.13)
    fig.tight_layout()
    save(fig, d, "fig32_preferred_lag")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def figure33(d):
    """Lag tuning predicts WHERE a cell fires, but not WHEN.

    a: continuous -- each cell's lag-0 encoding strength against how
       selectively it fires at its preferred square. Uses every cell instead of
       thresholding at the argmax, which discarded ~94% of them.
    b: the storage-vs-planning test. Null everywhere.
    """
    import glob as _g
    s1 = _latest("preferred_lag_firing_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    s2 = _latest("storage_vs_planning_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not s1 or not s2:
        print("  Fig 33 skipped: run `preferred_lag` and `storage_planning`")
        return
    from mc.analyse.swr_explore.preferred_lag_firing import preferred_lag_table
    PL = preferred_lag_table()[["session", "cell", "r_lag0"]]
    C = pd.read_csv(os.path.join(s1, "per_cell.csv")).merge(
        PL, on=["session", "cell"], how="left")
    S = pd.read_csv(os.path.join(s2, "per_cell.csv"))
    rois = ["HC", "mPFC", "mOFC"]
    col = {"HC": "#4A4A55", "mPFC": get_roi_colour("mPFC"),
           "mOFC": get_roi_colour("mOFC")}

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 6.5})
    fig, ax = plt.subplots(1, 4, figsize=(20 * CM, 5.6 * CM))

    # a-c: the continuous WHERE test, one scatter per ROI (execute)
    for i, rn in enumerate(rois):
        a = ax[i]
        g = C[(C.roi == rn) & (C.phase == "execute")
              & (C.window == "all")].dropna(subset=["sel", "r_lag0"])
        a.scatter(g.r_lag0, g.sel, s=5, color=col[rn], alpha=.35,
                  edgecolor="none")
        if len(g) > 5:
            r_, p_ = stats.pearsonr(g.r_lag0, g.sel)
            b1, b0 = np.polyfit(g.r_lag0, g.sel, 1)
            xx = np.linspace(g.r_lag0.min(), g.r_lag0.max(), 20)
            a.plot(xx, b0 + b1 * xx, "-", color=col[rn], lw=1.5)
            a.set_title(f"{rn}  r = {r_:+.2f}, P = {p_:.2g}\nn = {len(g)} cells",
                        loc="left", fontsize=7.5)
        a.axhline(0, color="grey", lw=.7); a.axvline(0, color="grey", lw=.7)
        a.set_xlabel("lag-0 encoding strength")
        if i == 0:
            a.set_ylabel("selectivity for the\npreferred square (execute)")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    # d: the WHEN test, null
    a = ax[3]
    preds = ["lag0_strength", "future_strength"]
    for pi, pred in enumerate(preds):
        m, e = [], []
        for rn in rois:
            g = S[S.roi == rn].dropna(subset=[pred, "store_minus_plan"])
            if len(g) < 20:
                m.append(np.nan); e.append(0); continue
            r_, p_ = stats.pearsonr(g[pred], g.store_minus_plan)
            m.append(r_)
            e.append(1.96 / np.sqrt(len(g) - 3))      # ~95% CI on r
        x = np.arange(len(rois)) + (pi - .5) * .22
        a.errorbar(x, m, yerr=e, fmt="o" if pi == 0 else "s", ms=5, capsize=3,
                   lw=1.2, color="#4A4A55",
                   mfc="#4A4A55" if pi == 0 else "white",
                   label="lag-0 strength" if pi == 0 else "future strength")
    a.axhline(0, color="grey", lw=.9)
    a.set_xticks(range(len(rois))); a.set_xticklabels(rois)
    a.set_ylabel("r with (storage \u2212 planning)")
    a.set_title("d  WHEN: null everywhere", loc="left")
    a.legend(frameon=False, fontsize=6)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

    fig.suptitle(
        "Figure 33 \u2014 lag tuning predicts WHERE a cell fires, not WHEN.\n"
        "a-c: a cell's lag-0 encoding strength predicts how selectively it "
        "fires at its preferred square during execute, in all three regions "
        "(continuous, every cell, preferred square cross-validated).\n"
        "d: the same tuning does NOT predict engagement at storage versus "
        "planning. The model predicts OPPOSITE signs for the two predictors; "
        "they are not obtained. EXPLORATORY.", fontsize=7.4, y=1.14)
    fig.tight_layout()
    save(fig, d, "fig33_where_not_when")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def figure34(d):
    """Are the lag-tuning contrasts ripple-dependent? No.

    a: does lag tuning predict a cell's engagement NEAR a ripple minus AWAY
       from one, at storage and at planning events?
    b: the most direct version -- post-ripple minus pre-ripple engagement.
    Plotted as correlations with 95% CIs, because the claim being tested is
    that these are non-zero, and a CI shows what the data can exclude.
    """
    src = _latest("storage_vs_planning_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 34 skipped: run `swr_content_explore.py storage_planning`")
        return
    R = pd.read_csv(os.path.join(src, "per_cell.csv"))
    rois = ["HC", "mPFC", "mOFC"]
    col = {"HC": "#4A4A55", "mPFC": get_roi_colour("mPFC"),
           "mOFC": get_roi_colour("mOFC")}
    preds = [("lag0_strength", "lag-0 strength", "o"),
             ("future_strength", "future strength", "s")]
    panels = [("storage_rip_minus_norip", "a  storage: near a ripple\n"
               "    minus away from one"),
              ("planning_rip_minus_norip", "b  planning: near a ripple\n"
               "    minus away from one"),
              ("ripple_post_minus_pre", "c  post-ripple minus\n"
               "    pre-ripple")]

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 6.5})
    fig, ax = plt.subplots(1, len(panels), figsize=(16 * CM, 6.2 * CM),
                           sharey=True)
    for pi, (meas, ttl) in enumerate(panels):
        a = ax[pi]
        for qi, (pred, plab, mk) in enumerate(preds):
            m, e, ps, ns = [], [], [], []
            for rn in rois:
                g = R[R.roi == rn].dropna(subset=[pred, meas]) \
                    if meas in R.columns else pd.DataFrame()
                if len(g) < 20:
                    m.append(np.nan); e.append(0); ps.append(np.nan); ns.append(0)
                    continue
                r_, p_ = stats.pearsonr(g[pred], g[meas])
                m.append(r_); ps.append(p_); ns.append(len(g))
                e.append(1.96 / np.sqrt(len(g) - 3))     # ~95% CI on r
            x = np.arange(len(rois)) + (qi - .5) * .20
            a.errorbar(x, m, yerr=e, fmt=mk, ms=5, capsize=3, lw=1.2,
                       color="#4A4A55", mfc="#4A4A55" if qi == 0 else "white",
                       label=plab if pi == 0 else None)
            for xx, mm, pp in zip(x, m, ps):
                if np.isfinite(pp) and pp < .05:
                    a.text(xx, mm + .035, "*", ha="center", va="bottom",
                           fontsize=11)
        a.axhline(0, color="grey", lw=1.0)
        a.set_xticks(range(len(rois))); a.set_xticklabels(rois)
        a.set_title(ttl, loc="left", fontsize=7.5)
        a.set_xlim(-.5, len(rois) - .5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel("correlation of lag tuning with\nthe ripple-dependent "
                     "contrast (r)")
    ax[0].legend(frameon=False, loc="lower left")
    n_hc = R[R.roi == "HC"].shape[0]
    fig.suptitle(
        "Figure 34 — are the lag-tuning contrasts RIPPLE-DEPENDENT? No. "
        "Each point is a correlation with its ~95% CI; a real effect would sit "
        "clear of zero.\n"
        "1 of 18 tests reaches P < 0.05 (HC, a, future strength, r = −0.089) "
        "against ~1 expected by chance. Panel c, the most direct test, is null "
        "everywhere.\n"
        f"mPFC leans positive at storage (+0.15, P = 0.065) and planning "
        f"(+0.13, P = 0.11) — a lead, not a result. n = {n_hc} HC, 148 mPFC, "
        "141 mOFC cells. EXPLORATORY.", fontsize=7.4, y=1.13)
    fig.tight_layout()
    save(fig, d, "fig34_ripple_dependent")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def figure35(d, xlim_ms=500):
    """fig30 split by WHICH reward was discovered (A, B, C, D).

    On the first traversal, A is discovered with nothing known, B with one
    known, and so on -- so the four curves differ in how much of the plan
    already exists when the ripple happens.

    ⚠ VERY underpowered: the pooled version has ~104 events per session and
    this divides them four ways. Shown because the A->D ordering is where a
    knowledge effect would live, not because any single curve is testable.
    """
    src = _latest("discovery_locked_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 35 skipped: run `swr_content_explore.py discovery_locked`")
        return
    T = pd.read_csv(os.path.join(src, "timecourse.csv"))
    if "reward" not in T.columns:
        print("  Fig 35 skipped: rerun discovery_locked for the reward split")
        return
    rois = ["mPFC", "mOFC", "HC_all"]
    LAB = {"mPFC": "mPFC", "mOFC": "mOFC", "HC_all": "HC pooled"}
    # project convention for states A-D
    RC = {1: "#F15A29", 2: "#F7931E", 3: "#C7C6E2", 4: "#6B60AA"}
    RL = {1: "A (nothing known)", 2: "B (1 known)", 3: "C (2 known)",
          4: "D (3 known)"}

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 7, "legend.fontsize": 6.5})
    fig, ax = plt.subplots(1, len(rois), figsize=(17 * CM, 6.2 * CM),
                           sharey=True)
    for ri, rn in enumerate(rois):
        a = ax[ri]
        # pooled, as the reference
        W0 = (T[(T.roi == rn) & (T.reward == 0)]
              .pivot_table(index="session", columns="offset_s",
                           values="z").dropna())
        if len(W0):
            a.plot(W0.columns.to_numpy() * 1000, W0.mean(0).to_numpy(), "-",
                   color="#999999", lw=2.2, alpha=.8, zorder=1,
                   label=f"all ({len(W0)})")
        ns = []
        for rw in (1, 2, 3, 4):
            W = (T[(T.roi == rn) & (T.reward == rw)]
                 .pivot_table(index="session", columns="offset_s",
                              values="z").dropna())
            if len(W) < 5:
                ns.append(0); continue
            off = W.columns.to_numpy() * 1000
            m = W.mean(0).to_numpy()
            se = (W.std(0, ddof=1) / np.sqrt(len(W))).to_numpy()
            a.plot(off, m, "-", color=RC[rw], lw=1.2,
                   label=f"{RL[rw]} ({len(W)})")
            a.fill_between(off, m - se, m + se, color=RC[rw], alpha=.12, lw=0)
            ns.append(int(T[(T.roi == rn) & (T.reward == rw)].n_events.median()))
        a.axvline(0, color=OBSERVED_VALUE_COLOR, ls=":", lw=1.1)
        a.axhline(0, color="grey", lw=.9)
        a.set_xlim(-xlim_ms, xlim_ms)
        a.set_xlabel("ms from ripple")
        med = ", ".join(str(n) for n in ns)
        a.set_title(f"{LAB[rn]}\nmedian events/session: {med}", loc="left",
                    fontsize=7)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel("evidence for the square whose\nreward was just discovered (z)")
    ax[0].legend(frameon=False, fontsize=5.8, loc="lower left")
    fig.suptitle(
        "Figure 35 — fig30 split by WHICH reward was just discovered. "
        "A is found with nothing known, D with three already known, so the "
        "curves differ in how much plan exists.\n"
        "⚠ SEVERELY underpowered — the pooled version has ~104 events "
        "per session and this divides them four ways (medians in each title). "
        "No test is reported because none would be meaningful.\n"
        "Shown to look at the A→D ordering only. EXPLORATORY.",
        fontsize=7.4, y=1.12)
    fig.tight_layout()
    save(fig, d, "fig35_discovery_by_reward")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def figure36(d):
    """Is the place template transferable across configurations? Barely.

    a: leave-one-out -- each configuration's rate map against the mean of the
       others, averaged over cells. This is exactly what the estimator's
       template relies on. ~0 means the template carries no transferable
       location information.
    b: SK's hypothesis -- does rate-map similarity fall as the two
       configurations time their squares more differently?
    c: the configuration x configuration similarity matrix for one example
       session, i.e. the 'confusion matrix' over tasks.
    """
    src = _latest("cross_task_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 36 skipped: run `swr_content_explore.py cross_task`")
        return
    R = pd.read_csv(os.path.join(src, "loo.csv"))
    PRS = pd.read_csv(os.path.join(src, "pairs.csv"))
    rois = ["HC", "mPFC", "mOFC"]
    col = {"HC": "#4A4A55", "mPFC": get_roi_colour("mPFC"),
           "mOFC": get_roi_colour("mOFC")}

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 6.5})
    fig, ax = plt.subplots(1, 3, figsize=(17 * CM, 6.2 * CM))

    # a: transferability
    a = ax[0]
    for i, rn in enumerate(rois):
        v = R[R.roi == rn].groupby("session").loo_r.mean().dropna()
        if len(v) < 5:
            continue
        t, p = stats.ttest_1samp(v, 0)
        a.bar(i, v.mean(), .6, yerr=stats.sem(v), capsize=3, color=col[rn],
              edgecolor="none")
        a.scatter(np.full(len(v), i) + np.random.default_rng(SEED_FIG)
                  .uniform(-.16, .16, len(v)), v, s=6, color="k", alpha=.25,
                  zorder=3)
        if p < .05:
            a.text(i, v.mean() + stats.sem(v) + .01, "*", ha="center",
                   fontsize=12)
        a.text(i, -0.14, f"r={v.mean():+.3f}\nP={p:.2g}", ha="center",
               fontsize=6)
    a.axhline(0, color="grey", lw=.9)
    a.set_xticks(range(len(rois))); a.set_xticklabels(rois)
    a.set_ylabel("rate map vs the mean of\nthe other configurations (r)")
    a.set_title("a  is the template transferable?", loc="left")
    a.set_ylim(-0.16, None)

    # b: lag divergence
    a = ax[1]
    for i, rn in enumerate(rois):
        g = PRS[PRS.roi == rn].dropna(subset=["lag_div", "map_r"])
        if len(g) < 20:
            continue
        per = [stats.pearsonr(gg.lag_div, gg.map_r)[0]
               for _, gg in g.groupby("session")
               if len(gg) >= 6 and gg.lag_div.std() > 0]
        rp, pp = stats.pearsonr(g.lag_div, g.map_r)
        tt, pw = stats.ttest_1samp(per, 0) if len(per) >= 5 else (np.nan, np.nan)
        a.errorbar(i - .13, rp, yerr=1.96 / np.sqrt(len(g) - 3), fmt="o", ms=5,
                   capsize=3, color=col[rn],
                   label="pooled over pairs" if i == 0 else None)
        # 95% CI for BOTH, not 1 SEM for one and 95% for the other -- the
        # mismatch made the within-session point look far tighter than it is
        ci_w = stats.t.ppf(.975, len(per) - 1) * stats.sem(per)
        a.errorbar(i + .13, np.mean(per), yerr=ci_w, fmt="s", ms=5,
                   capsize=3, color=col[rn], mfc="white",
                   label="within session, then across" if i == 0 else None)
        if pp < .05:
            a.text(i - .13, rp + .03, "*", ha="center", fontsize=11)
    a.axhline(0, color="grey", lw=.9)
    a.set_xticks(range(len(rois))); a.set_xticklabels(rois)
    a.set_ylabel("r(lag divergence, map similarity)")
    a.set_title("b  does lag divergence explain it?", loc="left")
    a.legend(frameon=False, fontsize=6, loc="lower left")

    # c: one session's configuration x configuration matrix
    a = ax[2]
    g = PRS[PRS.roi == "HC"]
    best = g.session.value_counts().idxmax() if len(g) else None
    gg = g[g.session == best]
    grids = sorted(set(gg.g1) | set(gg.g2))
    M = np.full((len(grids), len(grids)), np.nan)
    ix = {v: i for i, v in enumerate(grids)}
    for _, row in gg.iterrows():
        M[ix[row.g1], ix[row.g2]] = M[ix[row.g2], ix[row.g1]] = row.map_r
    np.fill_diagonal(M, np.nan)
    v = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
    im = a.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
    fig.colorbar(im, ax=a, shrink=.75, label="map similarity (r)")
    a.set_xticks(range(len(grids))); a.set_yticks(range(len(grids)))
    a.set_xticklabels([f"{int(x)}" for x in grids], fontsize=6)
    a.set_yticklabels([f"{int(x)}" for x in grids], fontsize=6)
    a.set_title(f"c  config x config, HC, session {int(best)}", loc="left")
    a.set_xlabel("configuration")
    for aa in ax[:2]:
        for sp in ("top", "right"):
            aa.spines[sp].set_visible(False)
    fig.suptitle(
        "Figure 36 — the place template barely transfers across "
        "configurations, and NOT because it is anti-correlated.\n"
        "a: HC +0.057 (P = 1.9e-5) but tiny; mPFC and mOFC are "
        "indistinguishable from ZERO — uninformative, not misleading.\n"
        "b: both error bars are 95% CIs. Lag divergence predicts lower "
        "similarity in mPFC when pooled (P = 0.034) but not within session "
        "(P = 0.16), which is the proper test. EXPLORATORY.",
        fontsize=7.4, y=1.12)
    fig.tight_layout()
    save(fig, d, "fig36_cross_task")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def figure37(d, session=None):
    """Confusion matrices: three TASK models beside the three neural ones.

    Top row: how similar are two configurations as TASKS -- by where the
    subject spends time, by which rewards they share, and by the movements
    they produce. Bottom row: how similar the rate maps are in each region.

    The question is whether the bottom row is explained by the top row. For HC
    it is (occupancy r = +0.113, trajectory +0.129, reward overlap +0.077, all
    P < 0.03 within session). For mPFC and mOFC nothing tracks any model.
    """
    src = _latest("cross_task_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 37 skipped: run `swr_content_explore.py cross_task`")
        return
    P = pd.read_csv(os.path.join(src, "pairs.csv"))
    if "occupancy_sim" not in P.columns:
        print("  Fig 37 skipped: rerun cross_task for the model matrices")
        return
    if session is None:
        g = P.groupby("session").agg(n=("map_r", "size"),
                                     k=("roi", "nunique"))
        g = g[g.k == 3].sort_values("n", ascending=False)
        session = int(g.index[0])
    S = P[P.session == session]
    grids = sorted(set(S.g1) | set(S.g2))
    ix = {v: i for i, v in enumerate(grids)}
    n = len(grids)

    def _mat(sub, col):
        M = np.full((n, n), np.nan)
        for _, r_ in sub.iterrows():
            if np.isfinite(r_[col]):
                M[ix[r_.g1], ix[r_.g2]] = M[ix[r_.g2], ix[r_.g1]] = r_[col]
        return M

    models = [("occupancy_sim", "TASK: occupancy\n(time per square)"),
              ("reward_overlap", "TASK: shared rewards\n(0-1 of four)"),
              ("trajectory_sim", "TASK: trajectory\n(9x9 transitions)")]
    neur = [("HC", "HC pooled"), ("mPFC", "mPFC"), ("mOFC", "mOFC")]
    stat = {"HC": "occ +0.11***  traj +0.13***  rew +0.08*",
            "mPFC": "all n.s.", "mOFC": "all n.s."}

    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5,
                         "xtick.labelsize": 6, "ytick.labelsize": 6})
    fig, ax = plt.subplots(2, 3, figsize=(15 * CM, 11 * CM))
    for j, (col, ttl) in enumerate(models):
        a = ax[0, j]
        M = _mat(S.drop_duplicates(subset=["g1", "g2"]), col)
        v = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
        im = a.imshow(M, cmap="PuOr_r", vmin=-v, vmax=v)
        fig.colorbar(im, ax=a, shrink=.72)
        a.set_title(ttl, loc="left", fontsize=7)
        a.set_xticks(range(n)); a.set_yticks(range(n))
        a.set_xticklabels([int(x) for x in grids]); a.set_yticklabels([int(x) for x in grids])
    for j, (rn, ttl) in enumerate(neur):
        a = ax[1, j]
        M = _mat(S[S.roi == rn], "map_r")
        v = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
        im = a.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
        fig.colorbar(im, ax=a, shrink=.72)
        a.set_title(f"NEURAL: {ttl}\n{stat[rn]}", loc="left", fontsize=7)
        a.set_xticks(range(n)); a.set_yticks(range(n))
        a.set_xticklabels([int(x) for x in grids]); a.set_yticklabels([int(x) for x in grids])
        a.set_xlabel("configuration")
    ax[0, 0].set_ylabel("configuration")
    ax[1, 0].set_ylabel("configuration")
    fig.suptitle(
        f"Figure 37 — configuration x configuration similarity: three TASK "
        f"models (top) and three regions (bottom). Session {session}, "
        f"{n} configurations.\n"
        "HC's cross-task similarity IS task similarity (occupancy +0.113, "
        "trajectory +0.129, reward overlap +0.077; within session then across, "
        "all P < 0.03).\n"
        "mPFC and mOFC track NONE of the three. ⚠ one session shown; the "
        "statistics are across all sessions. The three models are not "
        "independent of each other. EXPLORATORY.", fontsize=7.3, y=1.07)
    fig.tight_layout()
    save(fig, d, "fig37_confusion_task_vs_neural")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    plt.close(fig)


def _roi_row_axis(a, labels):
    a.set_xticks(range(len(labels)))
    a.set_xticklabels(labels, rotation=35, ha="right")
    a.axhline(0, color="grey", lw=.8)


def _p_row(a, xs, texts, y=.98, size=5.5):
    """P values in one row along the top of a panel, out of the data's way."""
    for x, t in zip(xs, texts):
        a.text(x, y, t, transform=a.get_xaxis_transform(), ha="center",
               va="top", fontsize=size)


def _headroom(a, frac=.45):
    lo, hi = a.get_ylim()
    a.set_ylim(lo, hi + frac * (hi - lo))


def figure38(d):
    """Across a ripple, does activity move from NOW to the UPCOMING squares?

    Same-block references (`swr_explore/future_shift`, pre-registered
    2026-10-07 FS). Every value is a per-session z against 1000 permutations of
    the nine square labels, then mean and 95% CI across sessions.

    a: gate -- evidence for the current square in ordinary still moments.
    b: PRIMARY  I = delta(FUT - NOW), as its components: ripple, the
       stratum-matched control, and ripple - control.
    c: J = delta(FUT - PAST), same layout. Needed before a positive I can be
       read as 'future' rather than 'less current'.
    d: the role profile, ripple - control. Single roles are relative to random
       squares, which include the current one, so only contrasts are read.
    e: I by phase, ripple counts matched across phases.
    f: I with all-other-traversals vs 4-nearest-traversals references.
    """
    from matplotlib.patches import Patch
    src = _latest("future_shift_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 38 skipped: run `swr_content_explore.py future_shift`")
        return
    R = pd.read_csv(os.path.join(src, "per_session.csv"))
    rois = [("spikes", "mPFC", "mPFC units"), ("spikes", "mOFC", "mOFC units"),
            ("spikes", "HC_all", "HC units"),
            ("hfb", "MedialFrontal", "med. frontal HFB"),
            ("hfb", "Visual", "visual HFB"), ("hfb", "HPC", "HC HFB (src)")]
    col = {"mPFC": get_roi_colour("mPFC"), "mOFC": get_roi_colour("mOFC"),
           "HC_all": "#4A4A55", "MedialFrontal": "#448363",
           "Visual": "#B0A8C0", "HPC": "#23677E"}
    primary = {("spikes", "mPFC"), ("hfb", "MedialFrontal")}
    labels = [r[2] for r in rois]

    def cell(sig, roi, q, phase="all", meas="z_diff", ref="block"):
        g = R[(R.signal == sig) & (R.roi == roi) & (R.quantity == q)
              & (R.phase == phase) & (R.ref == ref)]
        return g[meas].dropna() if meas in g else pd.Series(dtype=float)

    def ci(v):
        return stats.t.ppf(.975, len(v) - 1) * stats.sem(v) if len(v) > 2 \
            else np.nan

    def pval(v):
        return stats.ttest_1samp(v, 0)[1] if len(v) >= 5 else np.nan

    def pf(p):
        return f"{p:.2f}".lstrip("0") if p >= .01 else f"{p:.0e}"

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 7, "legend.fontsize": 6})
    fig, ax = plt.subplots(2, 3, figsize=(19 * CM, 13.5 * CM))
    jit = np.random.default_rng(SEED_FIG)

    # a: gate
    a = ax[0, 0]
    ptx = []
    for i, (sig, roi, lab) in enumerate(rois):
        v = cell(sig, roi, "now", meas="z_level_ctrl")
        a.bar(i, v.mean(), .6, yerr=ci(v), capsize=2, color=col[roi],
              edgecolor="k" if (sig, roi) in primary else "none", lw=.9)
        a.scatter(i + jit.uniform(-.18, .18, len(v)), v, s=3, color="k",
                  alpha=.3, zorder=3, lw=0)
        ptx.append(f"n={len(v)}\n{pf(pval(v))}")
    _roi_row_axis(a, labels)
    _headroom(a, .25)
    _p_row(a, range(len(rois)), ptx, size=5.2)
    a.set_ylabel("current-square evidence\nin still moments (z)")
    a.set_title("a  gate: is the current square coded?", loc="left")

    # b, c: components of I and J
    for a, q, ttl in ((ax[0, 1], "I", "b  PRIMARY: I = Δ(future − now)"),
                      (ax[0, 2], "J", "c  J = Δ(future − past)")):
        ptx = []
        for i, (sig, roi, lab) in enumerate(rois):
            for k, (meas, off) in enumerate((("z_rip", -.24), ("z_ctrl", 0),
                                              ("z_diff", .24))):
                v = cell(sig, roi, q, meas=meas)
                a.errorbar(i + off, v.mean(), yerr=ci(v),
                           fmt=("o", "o", "D")[k], ms=(3, 3, 4)[k],
                           mfc=("white", "#BDBDBD", col[roi])[k],
                           mec=(col[roi], "#7F7F7F", col[roi])[k],
                           ecolor="#666666", elinewidth=.7, capsize=1.5)
            ptx.append(pf(pval(cell(sig, roi, q))))
        _roi_row_axis(a, labels)
        _headroom(a, .2)
        _p_row(a, np.arange(len(rois)) + .24, ptx, size=5.2)
        a.set_ylabel("post − pre (z)")
        a.set_title(ttl, loc="left")
    fig.legend(handles=[
        plt.Line2D([], [], ls="", marker="o", mfc="white", mec="k", ms=3,
                   label="ripple"),
        plt.Line2D([], [], ls="", marker="o", mfc="#BDBDBD", mec="#7F7F7F",
                   ms=3, label="matched still moments"),
        plt.Line2D([], [], ls="", marker="D", mfc="k", mec="k", ms=4,
                   label="ripple − matched (its P above each ROI)")],
        frameon=False, loc="upper center", bbox_to_anchor=(.68, .875),
        ncol=3, fontsize=5.8, columnspacing=1.2, handletextpad=.3)

    # d: role profile
    a = ax[1, 0]
    roles = ["past3", "past2", "past1", "now", "fut1", "fut2", "fut3", "goal0"]
    x = np.r_[np.arange(7), 8]
    for (sig, roi, lab), ls in ((rois[0], "-"), (rois[3], "--"),
                                (rois[2], "-")):
        m = np.array([cell(sig, roi, q).mean() for q in roles])
        e = np.array([ci(cell(sig, roi, q)) for q in roles])
        a.plot(x[:7], m[:7], ls, marker="o", ms=2.8, color=col[roi], label=lab,
               lw=1.1)
        a.fill_between(x[:7], (m - e)[:7], (m + e)[:7], color=col[roi],
                       alpha=.12, lw=0)
        a.errorbar(x[7], m[7], yerr=e[7], fmt="s", ms=3.2, color=col[roi],
                   capsize=1.5)
    a.axhline(0, color="grey", lw=.8)
    a.axvline(3, color="grey", lw=.5, ls=":")
    a.set_xticks(x)
    a.set_xticklabels(["−3", "−2", "−1", "now", "+1", "+2", "+3", "next\ngoal"])
    a.set_xlabel("square along the route (visits from now)")
    a.set_ylabel("ripple − matched, post − pre (z)")
    a.set_title("d  where along the route does it move?", loc="left")
    _headroom(a, .3)
    a.legend(frameon=False, loc="upper left", ncol=2, fontsize=5.6)

    # e: phase, count matched
    a = ax[1, 1]
    phases = ["first_traversal", "encode_rest", "execute"]
    sel = (rois[0], rois[3], rois[2])
    w = .26
    xl = []
    for j, (sig, roi, lab) in enumerate(sel):
        n = 0
        for k, ph in enumerate(phases):
            v = cell(sig, roi, "I", phase=ph, meas="z_diff_matched")
            n = len(v)
            a.bar(j + (k - 1) * w, v.mean(), w, yerr=ci(v), capsize=1.5,
                  color=PHASE_COLORS[k], edgecolor=col[roi], lw=.9,
                  error_kw=dict(elinewidth=.7))
        xl.append(f"{lab.replace(' ', chr(10), 1)}\nn={n}")
    a.axhline(0, color="grey", lw=.8)
    a.set_xticks(range(len(sel))); a.set_xticklabels(xl)
    a.set_ylabel("I, ripple − matched (z)\nripple counts matched")
    a.set_title("e  I by learning phase", loc="left")
    _headroom(a, .3)
    a.legend(handles=[Patch(color=PHASE_COLORS[k], label=l) for k, l in
                      enumerate(["first traversal", "rest of encoding",
                                 "execution"])],
             frameon=False, loc="upper left", ncol=3, fontsize=5.6,
             columnspacing=.8, handlelength=1)

    # f: reference sensitivity
    a = ax[1, 2]
    for i, (sig, roi, lab) in enumerate(rois):
        for k, (ref, mk) in enumerate((("block", "o"), ("near4", "^"))):
            v = cell(sig, roi, "I", ref=ref)
            a.errorbar(i + (k - .5) * .3, v.mean(), yerr=ci(v), fmt=mk, ms=3.3,
                       color=col[roi], mfc=col[roi] if k == 0 else "white",
                       capsize=1.5, elinewidth=.7)
    _roi_row_axis(a, labels)
    _headroom(a, .3)
    a.legend(handles=[
        plt.Line2D([], [], ls="", marker="o", color="k", ms=3.3,
                   label="all other traversals of the block"),
        plt.Line2D([], [], ls="", marker="^", color="k", mfc="white", ms=3.3,
                   label="4 nearest traversals")],
        frameon=False, loc="upper left", fontsize=5.6)
    a.set_ylabel("I, ripple − matched (z)")
    a.set_title("f  reference built from …", loc="left")

    def line(sig, roi, q, meas="z_diff"):
        v = cell(sig, roi, q, meas=meas)
        t, p = stats.ttest_1samp(v, 0)
        return f"{v.mean():+.2f} z, t({len(v) - 1}) = {t:+.2f}, P = {p:.2g}"

    n_low = sum(pval(cell(s_, r_, "I", phase=ph, meas="z_diff_matched")) < .05
                for s_, r_, _ in sel for ph in phases)
    fig.suptitle(
        "Figure 38 — across a hippocampal ripple, does activity move from the "
        "current square towards the upcoming ones?\n"
        "Same-block references, pre-registered 2026-10-07 (FS). "
        f"PRIMARY I: mPFC units {line('spikes', 'mPFC', 'I')}; "
        f"medial frontal HFB {line('hfb', 'MedialFrontal', 'I')} "
        "(Bonferroni α = 0.025).\n"
        f"Gate, current square in still moments: mPFC units "
        f"{line('spikes', 'mPFC', 'now', 'z_level_ctrl')}; med. frontal HFB "
        f"{line('hfb', 'MedialFrontal', 'now', 'z_level_ctrl')}; HC units "
        f"{line('spikes', 'HC_all', 'now', 'z_level_ctrl')}.\n"
        "Both frontal signals FAIL the gate, so their nulls are uninterpretable; "
        "HC passes it and shows no shift either. "
        f"Panel e: {n_low} of {len(sel) * len(phases)} bars at P < 0.05 "
        "(sessions with >= 20 ripples in every phase only).\n"
        "Numbers above panels a-c = two-sided P, one-sample t across sessions "
        "(stored sign-flip P agrees). Mean ± 95% CI; dots = sessions; boxed "
        "bars = the two primaries. HC HFB contains the ripple itself: "
        "descriptive only.",
        fontsize=6.4, x=.01, ha="left", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .87), h_pad=1.2)
    save(fig, d, "fig38_future_shift")
    fig.savefig(os.path.join(d, "fig38_future_shift.pdf"))
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def figure39(d):
    """Is mPFC's 'now' code visible once its profile is built IN ripples?

    `swr_explore/ripple_templates`, pre-registered 2026-10-07 (RT).
    a-c: per ROI, current-square evidence (z vs 1000 label permutations, mean
       and 95% CI across sessions) for each template -- ripple-only, matched
       flank (same squares, dwell and data volume), all task time -- scored on
       ripple windows and on flank windows, encode and execute.
    d: the PRIMARY contrast, ripple-only minus flank template on encode ripple
       windows, with templates from all phases and from encode only.
    """
    src = _latest("ripple_templates_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
    if not src:
        print("  Fig 39 skipped: run `swr_content_explore.py ripple_templates`")
        return
    R = pd.read_csv(os.path.join(src, "per_session.csv"))
    rois = [("mPFC", "mPFC units"), ("mOFC", "mOFC units"),
            ("HC_all", "HC units")]
    col = {"mPFC": get_roi_colour("mPFC"), "mOFC": get_roi_colour("mOFC"),
           "HC_all": "#4A4A55"}
    tcol = {"rip": "#0e3d3a", "flank": "#BDBDBD", "all": "#7eb1c4"}
    tlab = {"rip": "ripple-only template", "flank": "matched-flank template",
            "all": "all-time template (≈40× data)"}

    def v_(roi, colname, tph="any"):
        g = R[(R.roi == roi) & (R.template_phase == tph)]
        return g[colname].dropna()

    def ci(v):
        return stats.t.ppf(.975, len(v) - 1) * stats.sem(v)

    def pf(p):
        return f"{p:.2f}".lstrip("0") if p >= .01 else f"{p:.0e}"

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 7, "legend.fontsize": 6})
    fig, ax = plt.subplots(1, 4, figsize=(20 * CM, 6.8 * CM),
                           gridspec_kw=dict(width_ratios=[1, 1, 1, .8]))
    groups = [("ripW", "encode", "ripple\nencode"),
              ("flankW", "encode", "flank\nencode"),
              ("ripW", "execute", "ripple\nexecute"),
              ("flankW", "execute", "flank\nexecute")]
    for a, (roi, lab) in zip(ax[:3], rois):
        for gi, (wn, ph, _) in enumerate(groups):
            for k, tn in enumerate(("rip", "flank", "all")):
                v = v_(roi, f"z_{tn}T_{wn}_{ph}")
                a.errorbar(gi + (k - 1) * .22, v.mean(), yerr=ci(v), fmt="o",
                           ms=3.2, color=tcol[tn], mec="k" if tn == "flank"
                           else tcol[tn], mew=.4, capsize=1.5, elinewidth=.7)
                p = stats.ttest_1samp(v, 0)[1]
                a.text(gi + (k - 1) * .22, .98, pf(p),
                       transform=a.get_xaxis_transform(), ha="center",
                       va="top", fontsize=4.6, rotation=90)
        a.axhline(0, color="grey", lw=.8)
        a.axvspan(-.5, .5, color=col[roi], alpha=.07, lw=0)
        a.set_xticks(range(len(groups)))
        a.set_xticklabels([g[2] for g in groups])
        a.set_xlim(-.5, len(groups) - .5)
        lo, hi = a.get_ylim(); a.set_ylim(lo, hi + .35 * (hi - lo))
        a.set_title(f"{'abc'[rois.index((roi, lab))]}  {lab} "
                    f"(n={len(v_(roi, 'z_ripT_ripW_encode'))})", loc="left")
        a.set_xlabel("test windows (held-out configuration)")
    ax[0].set_ylabel("current-square evidence (z)")
    fig.legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=3.2,
                                   color=tcol[t], mec="k" if t == "flank"
                                   else tcol[t], mew=.4, label=tlab[t])
                        for t in ("rip", "flank", "all")],
               frameon=False, loc="upper left", bbox_to_anchor=(.05, .80),
               ncol=3, fontsize=6)

    a = ax[3]
    jit = np.random.default_rng(SEED_FIG)
    for i, (roi, lab) in enumerate(rois):
        for k, tph in enumerate(("any", "encode")):
            v = v_(roi, "rip_minus_flank_T_ripW_encode", tph)
            x = i + (k - .5) * .34
            a.bar(x, v.mean(), .3, yerr=ci(v), capsize=1.5, color=col[roi],
                  alpha=1 if k == 0 else .45, edgecolor="k" if roi == "mPFC"
                  and k == 0 else "none", lw=.8, error_kw=dict(elinewidth=.7))
            a.scatter(x + jit.uniform(-.08, .08, len(v)), v, s=2.5, color="k",
                      alpha=.3, lw=0, zorder=3)
            a.text(x, .98, pf(stats.ttest_1samp(v, 0)[1]),
                   transform=a.get_xaxis_transform(), ha="center", va="top",
                   fontsize=5)
    a.axhline(0, color="grey", lw=.8)
    a.set_xticks(range(len(rois))); a.set_xticklabels([r[1] for r in rois],
                                                      rotation=30, ha="right")
    a.set_ylabel("ripple-only − flank (z)")
    a.set_title("d  PRIMARY contrast", loc="left")
    a.legend(handles=[Patch_(fc="k", alpha=1, label="templates: all phases"),
                      Patch_(fc="k", alpha=.45,
                             label="templates: encode only")],
             frameon=False, loc="lower left", fontsize=5.6)

    def line(roi, colname, tph="any"):
        v = v_(roi, colname, tph)
        t, p = stats.ttest_1samp(v, 0)
        return f"{v.mean():+.2f} z, t({len(v) - 1}) = {t:+.2f}, P = {p:.2g}"

    fig.suptitle(
        "Figure 39 — does mPFC code the current square once its spatial "
        "profile is built from ripple windows only? Pre-registered "
        "2026-10-07 (RT).\n"
        f"PRIMARY (mPFC, encode ripple windows, ripple-only − flank template): "
        f"{line('mPFC', 'rip_minus_flank_T_ripW_encode')}.  Gate (HC, "
        f"ripple-only template, encode ripple windows): "
        f"{line('HC_all', 'z_ripT_ripW_encode')}.\n"
        "Exploratory, uncorrected: mOFC ripple-only − flank "
        f"{line('mOFC', 'rip_minus_flank_T_ripW_encode')}; with encode-only "
        f"templates {line('mOFC', 'rip_minus_flank_T_ripW_encode', 'encode')}."
        "\nAll templates leave the test configuration out; ripple-only and "
        "flank templates come from the same ripples, squares and durations.\n"
        "Small numbers = two-sided P across sessions. Mean ± 95% CI. "
        "d: encode ripple windows; shaded columns in a-c are the windows d "
        "uses.",
        fontsize=6.4, x=.01, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, .74), w_pad=1.5)
    save(fig, d, "fig39_ripple_templates")
    fig.savefig(os.path.join(d, "fig39_ripple_templates.pdf"))
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    plt.close(fig)


def Patch_(**kw):
    from matplotlib.patches import Patch
    return Patch(**kw)
