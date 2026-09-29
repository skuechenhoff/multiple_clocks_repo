#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do the phase comparisons need correcting for how many ripples each phase has?

Ripple counts are wildly unequal across phases. Correct uncovers: 3750 ripples
on the first traversal against 20069 afterwards, a factor of 5.4. Navigation:
4810 against 28838, a factor of 6.0. Incorrect uncovers run the other way,
5963 against 3244. So the question is not academic.

WHAT COUNT DIFFERENCES CAN AND CANNOT DO
----------------------------------------
Each saved trace is the MEAN over that condition's ripples, and a mean is
unbiased in n. So unequal counts cannot by themselves push a phase's estimate
up or down -- there is no bias to correct.

What they do change is the VARIANCE. The mean of n ripples has noise ~1/sqrt(n),
so a phase with a fifth of the ripples has a per-session estimate roughly 2.2x
noisier, and every test on it has less power. That is not a bias but it IS an
asymmetry, and it cuts in a direction that depends on which way the counts go:

  * where a LOW-count phase shows the effect and a high-count phase does not,
    the count difference works AGAINST the finding and cannot explain it;
  * where a HIGH-count phase shows the effect and a low-count one does not,
    the count difference could be the whole story.

Those two cases need reporting differently, so this script labels every phase
contrast with which one it is.

WHAT IS CHECKED HERE
--------------------
1. Does the noise actually scale as 1/sqrt(n)? Regress log(flank SD) on
   log(n_ripples); the slope should be near -0.5. If it is flat, the estimate
   is dominated by between-session variance rather than ripple count and the
   whole concern is small.
2. Does the VALUE depend on n? Spearman of coupling against n_ripples across
   sessions. It should be ~0 -- if it is not, something other than sampling
   noise tracks ripple count.
3. Does the phase contrast survive when count is controlled? Two ways:
   the paired difference regressed on log(count ratio), and a count-matched
   subset of sessions where the two phases are within `max_ratio`.
4. Inverse-variance weighting, using n as the weight, against the unweighted
   mean.

WHAT THIS CANNOT DO
-------------------
A true count-matched analysis would SUBSAMPLE ripples so both phases average
over the same number, and that is impossible here: the saved traces are already
per-condition means and the individual ripple epochs are not in the bundle.
That needs a re-extraction with subsampling. Everything below is the strongest
control available without one, and the re-extraction is the recommendation if
any phase contrast ever becomes load-bearing.

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_hfb_ripple_count_control.py run \\
        --cond_dir=<ripple_hfb_conditions_exclusive_...>

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import datetime
import itertools

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))   # repo root
import mc.analyse.swr_io as swr_io
import mc.plotting.ripple_figures as rfig

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))    # scripts/
# archived scripts import BOTH maintained siblings (in scripts/) and each
# other (here), so both directories have to be importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # exploratory/
from swr_hfb_locking_contrasts import NONPERI_S, COL, LABEL
from swr_hfb_roi_partition import _bh
from swr_hfb_roi_conditions import PHASES, COND_KEYS, PERI_MS, MEDIAL_GROUP
from swr_hfb_roi_compare import _prep, _pooled

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "hfb_ripple_count_control"
ROIS = (MEDIAL_GROUP, "lOFC", "mPFC")
MIN_N = 5
MAX_RATIO = 2.0        # "count-matched" = within this factor, either way


def _per_session(t_ms, tr, ix, counts, roi, conds):
    """Coupling value, flank SD and ripple count per session."""
    cur = _pooled(t_ms, tr, ix[ix.roi == roi], counts, conds, "session")
    peri = np.abs(t_ms) <= PERI_MS
    fl = (np.abs(t_ms) >= NONPERI_S[0] * 1000) & \
         (np.abs(t_ms) < NONPERI_S[1] * 1000)
    n_by_sess = (counts[counts.cond.isin(conds)]
                 .groupby("session").n_ripples.sum())
    rows = []
    for u, c in cur.items():
        rows.append({"session": int(u), "value": float(c[peri].mean()),
                     "flank_sd": float(np.std(c[fl], ddof=1)),
                     "n_ripples": float(n_by_sess.get(int(u), np.nan))})
    return pd.DataFrame(rows).dropna(subset=["n_ripples"])


def run(cond_dir=None, partition_dir=None, out_dir=None,
        max_ratio=MAX_RATIO, save=True):
    from scipy import stats as st

    root = swr_io.derivatives_dir(swr_io.get_data_root())
    G = os.path.join(root, "group", "swr")
    if cond_dir is None:
        raise SystemExit("--cond_dir is required (the exclusive-boundary run)")
    partition_dir = partition_dir or sorted(
        os.path.join(G, x) for x in os.listdir(G)
        if x.startswith("hfb_roi_partition_"))[-1]
    t_ms, tr, ix, counts = _prep(cond_dir, partition_dir)

    # ---- the per-session table everything else is built from -------------
    tab = []
    for roi, ph, k in itertools.product(ROIS, PHASES, COND_KEYS):
        d = _per_session(t_ms, tr, ix, counts, roi, PHASES[ph][k])
        if len(d):
            tab.append(d.assign(roi=roi, phase=ph, condition=k))
    S = pd.concat(tab, ignore_index=True)

    # ---- 1. does noise scale as 1/sqrt(n)? -------------------------------
    scal = []
    for (roi, k), g in S.groupby(["roi", "condition"]):
        g = g[(g.n_ripples > 0) & (g.flank_sd > 0)]
        if len(g) < 10:
            continue
        sl, ic, r, p, se = st.linregress(np.log(g.n_ripples),
                                         np.log(g.flank_sd))
        scal.append({"roi": roi, "condition": k, "n": len(g),
                     "slope": float(sl), "se": float(se), "r": float(r),
                     "p": float(p)})
    SC = pd.DataFrame(scal)

    # ---- 2. does the VALUE depend on n? ----------------------------------
    dep = []
    for (roi, ph, k), g in S.groupby(["roi", "phase", "condition"]):
        if len(g) < MIN_N:
            continue
        rho, p = st.spearmanr(g.n_ripples, g.value)
        dep.append({"roi": roi, "phase": ph, "condition": k, "n": len(g),
                    "rho": float(rho), "p": float(p)})
    DP = pd.DataFrame(dep)
    DP["q"] = _bh(DP["p"]) if len(DP) else np.nan

    # ---- 3. phase contrasts, raw / count-covaried / count-matched --------
    con = []
    phase_pairs = [("first_traversal", "after_first"),
                   ("encoding", "after_encoding")]
    for roi, (pa, pb), k in itertools.product(ROIS, phase_pairs, COND_KEYS):
        A = S[(S.roi == roi) & (S.phase == pa) & (S.condition == k)] \
            .set_index("session")
        B = S[(S.roi == roi) & (S.phase == pb) & (S.condition == k)] \
            .set_index("session")
        sh = sorted(set(A.index) & set(B.index))
        if len(sh) < MIN_N:
            continue
        dv = (A.loc[sh, "value"] - B.loc[sh, "value"]).to_numpy()
        lr = np.log(A.loc[sh, "n_ripples"].to_numpy()
                    / B.loc[sh, "n_ripples"].to_numpy())
        t, p = st.ttest_1samp(dv, 0.0)
        r = {"roi": roi, "condition": k, "phase_a": pa, "phase_b": pb,
             "n": len(sh), "diff": float(dv.mean()),
             "sem": float(dv.std(ddof=1) / np.sqrt(len(dv))), "p": float(p),
             "median_count_a": float(A.loc[sh, "n_ripples"].median()),
             "median_count_b": float(B.loc[sh, "n_ripples"].median()),
             "median_ratio": float(np.exp(np.median(lr))),
             # Which way the power asymmetry cuts. If the phase with FEWER
             # ripples is the larger one, counts work against the difference.
             "counts_favour": (pa if np.median(lr) > 0 else pb),
             "effect_favours": (pa if dv.mean() > 0 else pb)}
        r["counts_explain_it"] = bool(r["counts_favour"] == r["effect_favours"])
        # covaried: intercept of diff ~ log(count ratio), i.e. the difference
        # predicted at equal counts
        if len(sh) >= 6 and np.ptp(lr) > 0:
            sl, ic, rr, pp, se = st.linregress(lr, dv)
            r.update({"cov_slope": float(sl), "cov_slope_p": float(pp),
                      "diff_at_equal_counts": float(ic)})
        # count-matched subset
        keep = np.abs(lr) <= np.log(max_ratio)
        if keep.sum() >= MIN_N:
            tm, pm = st.ttest_1samp(dv[keep], 0.0)
            r.update({"n_matched": int(keep.sum()),
                      "matched_diff": float(dv[keep].mean()),
                      "matched_p": float(pm)})
        # ---- 4. inverse-variance weighting, weight = n ------------------
        wv = A.loc[sh, "n_ripples"].to_numpy() + B.loc[sh, "n_ripples"].to_numpy()
        r["weighted_diff"] = float(np.average(dv, weights=wv))
        con.append(r)
    C = pd.DataFrame(con)
    for col in ("p", "matched_p"):
        if col in C:
            C[col.replace("p", "q")] = _bh(C[col])

    _report(S, SC, DP, C, max_ratio)

    if not save:
        return None
    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(G, f"{ANALYSIS_NAME}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    S.to_csv(os.path.join(out_dir, "per_session_values_and_counts.csv"),
             index=False)
    SC.to_csv(os.path.join(out_dir, "noise_vs_count_scaling.csv"), index=False)
    DP.to_csv(os.path.join(out_dir, "value_vs_count.csv"), index=False)
    C.to_csv(os.path.join(out_dir, "phase_contrasts_count_controlled.csv"),
             index=False)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"analysis": ANALYSIS_NAME, "created": stamp,
                   "cond_dir": cond_dir, "partition_dir": partition_dir,
                   "max_ratio_for_matching": max_ratio,
                   "note": ("counts cannot bias a mean, only its variance; a "
                            "true count-matched test needs ripple subsampling "
                            "and therefore a re-extraction"),
                   "noise_scaling": SC.to_dict("records"),
                   "value_vs_count": DP.to_dict("records"),
                   "phase_contrasts": C.to_dict("records")},
                  f, indent=2, default=str)
    _figure(S, SC, C, out_dir)
    print(f"\nsaved -> {out_dir}")
    return None


def _report(S, SC, DP, C, max_ratio):
    f4 = lambda v: "" if pd.isna(v) else f"{v:+.4f}"
    f3 = lambda v: "" if pd.isna(v) else f"{v:.3f}"
    print(f"\n{'='*100}\n1. DOES THE NOISE SCALE AS 1/sqrt(n)?  "
          f"log(flank SD) ~ log(n_ripples)\n{'='*100}")
    print(f"  {'ROI':<15s}{'condition':<12s}{'n':>4s}{'slope':>9s}{'se':>8s}"
          f"{'r':>8s}{'p':>9s}   (pure sampling noise = -0.50)")
    for _, r in SC.sort_values(["roi", "condition"]).iterrows():
        print(f"  {r.roi:<15s}{r.condition:<12s}{r.n:>4d}{r.slope:>+9.3f}"
              f"{r.se:>8.3f}{r.r:>+8.3f}{r.p:>9.4f}")

    print(f"\n{'='*100}\n2. DOES THE VALUE DEPEND ON n?  Spearman(coupling, "
          f"n_ripples) per cell\n{'='*100}")
    sig = DP[DP.q < 0.05]
    print(f"  {len(sig)} of {len(DP)} cells show a count-dependent VALUE at "
          f"q < 0.05")
    show = sig if len(sig) else DP.nsmallest(6, "p")
    print(f"  {'ROI':<15s}{'phase':<17s}{'cond':<9s}{'n':>4s}{'rho':>8s}"
          f"{'p':>8s}{'q':>8s}")
    for _, r in show.iterrows():
        print(f"  {r.roi:<15s}{r.phase:<17s}{r.condition:<9s}{r.n:>4d}"
              f"{r.rho:>+8.3f}{f3(r.p):>8s}{f3(r.q):>8s}")

    print(f"\n{'='*100}\n3. PHASE CONTRASTS WITH COUNT CONTROLLED\n{'='*100}")
    print(f"  {'ROI':<15s}{'cond':<8s}{'A vs B':<30s}{'n':>4s}{'raw':>10s}"
          f"{'p':>7s}{'ratio':>7s}{'matched':>10s}{'p':>7s}{'@equal':>10s}")
    for _, r in C.sort_values(["roi", "condition"]).iterrows():
        ab = f"{r.phase_a[:13]} vs {r.phase_b[:13]}"
        print(f"  {r.roi:<15s}{r.condition:<8s}{ab:<30s}{r.n:>4d}"
              f"{f4(r['diff']):>10s}{f3(r.p):>7s}{r.median_ratio:>7.2f}"
              f"{f4(r.get('matched_diff', np.nan)):>10s}"
              f"{f3(r.get('matched_p', np.nan)):>7s}"
              f"{f4(r.get('diff_at_equal_counts', np.nan)):>10s}")
    print(f"  'ratio' is median n_ripples(A)/n_ripples(B); 'matched' keeps "
          f"only sessions within {max_ratio}x;\n  '@equal' is the intercept of "
          "diff ~ log(count ratio), the difference predicted at equal counts.")

    print(f"\n{'='*100}\n   WHICH WAY DO THE COUNTS CUT?\n{'='*100}")
    for _, r in C.sort_values(["roi", "condition"]).iterrows():
        verdict = ("counts COULD explain it" if r.counts_explain_it
                   else "counts work AGAINST it")
        print(f"  {r.roi:<15s}{r.condition:<8s}"
              f"{r.phase_a[:13] + ' vs ' + r.phase_b[:13]:<30s}"
              f"effect favours {r.effect_favours[:14]:<15s}"
              f"counts favour {r.counts_favour[:14]:<15s}{verdict}")


def _figure(S, SC, C, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(rfig._rc()):
        fig, axes = plt.subplots(1, 3, figsize=(20 * rfig.CM, 7 * rfig.CM))
        # a) noise vs count
        ax = axes[0]
        for k in COND_KEYS:
            g = S[(S.condition == k) & (S.n_ripples > 0) & (S.flank_sd > 0)]
            ax.scatter(g.n_ripples, g.flank_sd, s=5, c=COL[k], alpha=0.5,
                       linewidths=0, label=LABEL[k])
        x = np.logspace(0.5, 3.5, 50)
        ax.plot(x, 0.09 * x ** -0.5, color="#3d3d3d", lw=1.0, ls="--",
                label="1/√n reference")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("ripples in that phase (per session)")
        ax.set_ylabel("flank SD of the curve\n(the estimator's noise)")
        ax.set_title("noise falls with ripple count", fontsize=rfig.FS_TITLE)
        ax.legend(frameon=False, fontsize=rfig.FS_TICK - 2, loc="lower left")
        # b) value vs count
        ax = axes[1]
        for k in COND_KEYS:
            g = S[(S.condition == k) & (S.n_ripples > 0)]
            ax.scatter(g.n_ripples, g.value, s=5, c=COL[k], alpha=0.5,
                       linewidths=0)
        ax.axhline(0, color="#3d3d3d", lw=0.7, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("ripples in that phase (per session)")
        ax.set_ylabel("ripple-HFB coupling")
        ax.set_title("but the VALUE does not", fontsize=rfig.FS_TITLE)
        # c) raw vs count-matched phase differences
        ax = axes[2]
        g = C.dropna(subset=["matched_diff"])
        if len(g):
            ax.scatter(g["diff"], g.matched_diff, s=18,
                       c=[COL[k] for k in g.condition], linewidths=0)
            lim = float(np.nanmax(np.abs(np.r_[g["diff"], g.matched_diff]))) * 1.15
            ax.plot([-lim, lim], [-lim, lim], color="#3d3d3d", lw=0.7, ls="--")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        ax.axhline(0, color="0.7", lw=0.5)
        ax.axvline(0, color="0.7", lw=0.5)
        ax.set_xlabel("phase difference, all sessions")
        ax.set_ylabel("phase difference,\ncount-matched sessions only")
        ax.set_title("matching barely moves them", fontsize=rfig.FS_TITLE)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "ripple_count_control.pdf"))
        fig.savefig(os.path.join(out_dir, "ripple_count_control.png"), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
