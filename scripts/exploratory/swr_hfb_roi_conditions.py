#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does ripple-HFB coupling depend on WHAT the subject just did, and does that
differ between lateral OFC and medial frontal cortex?

The coupling table (`swr_hfb_roi_partition.py`) says lOFC couples at least as
strongly as medial frontal. It says nothing about what the coupling is FOR.
This splits every partition ROI by condition -- correct uncovers, incorrect
uncovers, navigation presses -- and by learning phase, and then asks the two
questions that matter:

  WITHIN an ROI   is correct > navigation? i.e. is the coupling about reward
                  at all, or about pressing a key?
  BETWEEN ROIs    is (correct - navigation) BIGGER in lOFC than in medial
                  frontal? That is the interaction, and it is the only thing
                  that licenses "lOFC is more reward-linked". Two separate
                  within-ROI tests do not.

COVERAGE LIMIT -- READ THIS BEFORE ASKING FOR TEMPORAL ROIs
-----------------------------------------------------------
`swr_ripple_hfb_conditions.py` only ever extracted FRONTAL derivations: its
index holds 310 derivations, all in the mPFC / mOFC / MedialFrontal families.
So this can report lOFC, mPFC, mOFC, FrontalPole and vlPFC_IFG, and CANNOT
report MTG, STG, Auditory or any other partition ROI -- those traces do not
exist. Getting them means re-running the conditions extraction with the
cortical ROI list widened; that is a fresh pass over the 4 GB of HFB data, not
a re-read.

THE STATISTIC
-------------
Identical to the coupling table: the per-session curve is real minus
shifted-null, flank-normalised to the non-peri band, and the value is its mean
over the peri window (+-250 ms). Built by `pooled_cond_curves` from
`swr_hfb_locking_contrasts`, imported rather than reimplemented.

SMOOTHING IS OFF for the statistics. The 100 ms Gaussian in the figure
functions is cosmetic; leaving it on would correlate neighbouring samples
inside the peri window for no gain.

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_hfb_roi_conditions.py run \\
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
from swr_hfb_locking_contrasts import pooled_cond_curves, NONPERI_S, COL, LABEL
from swr_hfb_roi_partition import ROI_C, _bh, UNDERPOWERED_BELOW

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "hfb_roi_conditions"
PERI_MS = 250.0
MIN_UNITS = 5             # below this a paired t is not worth printing
COND_KEYS = ("reward", "error", "move")
REF_ROI = "mPFC"          # the medial frontal partition ROI

# Phase splits, in the EXCLUSIVE-boundary condition names. `encoding` is
# everything up to but not including the first fully correct ABCD loop.
# TWO different cuts, four phases plus the pooled one. They are not nested:
# `first_traversal` is repeat 1 only; `encoding` is everything before the first
# fully correct loop, which USUALLY contains several repeats. The first pair
# was in the 2 cm figure from the start and was missing here, which is why the
# first-traversal effect had no number next to it.
PHASES = {
    "all": {"reward": ("reward_explore", "reward_plan", "reward_execute"),
            "error": ("error_explore", "error_plan", "error_execute"),
            "move": ("move_explore", "move_plan", "move_execute")},
    "first_traversal": {"reward": ("reward_explore",),
                        "error": ("error_explore",),
                        "move": ("move_explore",)},
    "after_first": {"reward": ("reward_plan", "reward_execute"),
                    "error": ("error_plan", "error_execute"),
                    "move": ("move_plan", "move_execute")},
    "encoding": {"reward": ("reward_explore", "reward_plan"),
                 "error": ("error_explore", "error_plan"),
                 "move": ("move_explore", "move_plan")},
    "after_encoding": {"reward": ("reward_execute",),
                       "error": ("error_execute",),
                       "move": ("move_execute",)},
}

# The collapsed |x| <= 20 mm medial frontal group. It is what the 2 cm
# condition panels plot and what the manuscript means by "medial frontal", so
# leaving it out of this table -- as the first version did -- meant the figure
# and the statistics were describing different regions. It OVERLAPS the
# partition ROIs (it is a coordinate cut across mPFC+mOFC), so it is excluded
# from the ROI-vs-ROI comparisons and flagged everywhere else.
MEDIAL_GROUP = "MedialFrontal"


def _peri_values(cond_dir, roi_part_lut, roi, conds, unit, t_ms, tr, ix, counts):
    """Per-unit coupling value for one ROI x condition set."""
    sub = ix[ix._roi_part == roi]
    if not len(sub):
        return {}
    cur = pooled_cond_curves(t_ms, tr, sub, counts, conds, roi=roi,
                             unit=unit, smooth_ms=0.0)
    peri = np.abs(t_ms) <= PERI_MS
    return {u: float(c[peri].mean()) for u, c in cur.items()}


def run(cond_dir=None, partition_dir=None, out_dir=None, unit="session",
        ref_roi=REF_ROI, save=True):
    from scipy import stats as st
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = swr_io.derivatives_dir(swr_io.get_data_root())
    G = os.path.join(root, "group", "swr")
    if cond_dir is None:
        raise SystemExit(
            "--cond_dir is required, and should be the EXCLUSIVE-boundary run "
            "so that 'encoding' means up to but not including the first fully "
            "correct loop.")
    partition_dir = partition_dir or _latest(G, "hfb_roi_partition_")

    A = pd.read_csv(os.path.join(partition_dir, "derivation_assignment.csv"))
    part = A.set_index(["session", "cx_pair"])["roi_part"]

    z = np.load(os.path.join(cond_dir, "timecourses.npz"))
    t_ms, tr = z["t_ms"], z["traces"]
    ix = pd.read_csv(os.path.join(cond_dir, "timecourse_index.csv"))
    ix["is_real"] = ix.is_real.astype(bool)
    if "trace_row" not in ix.columns:
        ix["trace_row"] = np.arange(len(ix))
    # The family rows are the originals; the MedialFrontal rows are duplicates
    # of them under a derived name. Both are kept, the families relabelled to
    # partition ROIs and MedialFrontal kept as itself, so the table covers the
    # regions AND the group the figure plots.
    fam = ix[ix.roi.isin(("mPFC", "mOFC"))].copy()
    fam["_roi_part"] = pd.MultiIndex.from_arrays(
        [fam.session, fam.cx_pair]).map(part)
    mfr = ix[ix.roi == MEDIAL_GROUP].copy()
    mfr["_roi_part"] = MEDIAL_GROUP
    ix = pd.concat([fam, mfr], ignore_index=True)
    # `pooled_cond_curves` selects on a column called `roi`; the partition
    # label is written into it so the imported function needs no change.
    ix["roi"] = ix["_roi_part"]
    counts = pd.read_csv(os.path.join(cond_dir, "ripple_counts.csv"))

    present = [r for r in ix._roi_part.dropna().unique()
               if r not in ("Unassigned",)]
    n_der = (ix.dropna(subset=["_roi_part"])
               .groupby("_roi_part")[["session", "cx_pair"]]
               .apply(lambda g: len(g.drop_duplicates())))
    rois = [r for r in present if n_der.get(r, 0) >= 10]
    # MedialFrontal first: it is the comparison group the figure plots.
    rois = sorted(rois, key=lambda r: (r != MEDIAL_GROUP, -n_der[r]))
    print(f"conditions run covers {len(ix[['session','cx_pair']].drop_duplicates())} "
          f"derivations, all frontal")
    print("ROIs with >= 10 derivations: "
          + ", ".join(f"{r} ({n_der[r]})" for r in rois))

    # ---- values ---------------------------------------------------------
    vals = {}          # (roi, phase, cond) -> {unit: value}
    for roi, (ph, spec), k in itertools.product(rois, PHASES.items(),
                                                COND_KEYS):
        vals[(roi, ph, k)] = _peri_values(cond_dir, part, roi, spec[k], unit,
                                          t_ms, tr, ix, counts)

    rows = []
    for (roi, ph, k), v in vals.items():
        if len(v) < 2:
            continue
        a = np.array(list(v.values()))
        r = {"roi": roi, "phase": ph, "condition": k, "n_units": len(a),
             "coupling": float(a.mean()),
             "sem": float(a.std(ddof=1) / np.sqrt(len(a)))}
        if len(a) >= 3:
            t, p = st.ttest_1samp(a, 0.0)
            r["t"], r["p"] = float(t), float(p)
        rows.append(r)
    T = pd.DataFrame(rows)
    T["q"] = _bh(T["p"]) if "p" in T else np.nan

    # ---- within-ROI condition contrasts ---------------------------------
    def _paired(x, y):
        sh = sorted(set(x) & set(y))
        if len(sh) < MIN_UNITS:
            return None
        d = np.array([x[u] - y[u] for u in sh])
        t, p = st.ttest_1samp(d, 0.0)
        return dict(n=len(sh), diff=float(d.mean()),
                    sem=float(d.std(ddof=1) / np.sqrt(len(d))),
                    t=float(t), p=float(p))

    wrows = []
    for roi, ph in itertools.product(rois, PHASES):
        for a, b in (("reward", "move"), ("reward", "error"),
                     ("error", "move")):
            res = _paired(vals[(roi, ph, a)], vals[(roi, ph, b)])
            if res:
                wrows.append(dict(roi=roi, phase=ph, contrast=f"{a}-{b}", **res))
    W = pd.DataFrame(wrows)
    W["q"] = _bh(W["p"]) if len(W) else np.nan

    # ---- the interaction: (reward - move) here vs in the reference ------
    # This is the test the question actually needs. "reward > move in lOFC and
    # not in mPFC" is two tests and is not evidence of a difference.
    irows = []
    for roi, ph in itertools.product([r for r in rois if r != ref_roi], PHASES):
        for a, b in (("reward", "move"), ("reward", "error")):
            xa, xb = vals[(roi, ph, a)], vals[(roi, ph, b)]
            ya, yb = vals[(ref_roi, ph, a)], vals[(ref_roi, ph, b)]
            sh = sorted(set(xa) & set(xb) & set(ya) & set(yb))
            if len(sh) < MIN_UNITS:
                continue
            d = np.array([(xa[u] - xb[u]) - (ya[u] - yb[u]) for u in sh])
            t, p = st.ttest_1samp(d, 0.0)
            irows.append(dict(roi=roi, ref=ref_roi, phase=ph,
                              contrast=f"({a}-{b})", n=len(sh),
                              diff=float(d.mean()),
                              sem=float(d.std(ddof=1) / np.sqrt(len(d))),
                              t=float(t), p=float(p)))
    I = pd.DataFrame(irows)
    I["q"] = _bh(I["p"]) if len(I) else np.nan

    # ---- ROI vs ROI, WITHIN a condition ---------------------------------
    # The missing cell of the design. "Which region couples most to rewards"
    # is this comparison, not the within-ROI condition contrast and not the
    # interaction. Paired on the units that have both regions.
    rrows = []
    for ph, k in itertools.product(PHASES, COND_KEYS):
        # MedialFrontal is a coordinate cut across mPFC+mOFC, so comparing it
        # with them is a group against part of itself. Partition ROIs only.
        for a, b in itertools.combinations(
                [r for r in rois if r != MEDIAL_GROUP], 2):
            res = _paired(vals[(a, ph, k)], vals[(b, ph, k)])
            if res:
                rrows.append(dict(phase=ph, condition=k, roi_a=a, roi_b=b,
                                  **res))
    RR = pd.DataFrame(rrows)
    RR["q"] = _bh(RR["p"]) if len(RR) else np.nan

    # ---- phase contrast within ROI x condition --------------------------
    prows = []
    for roi, k in itertools.product(rois, COND_KEYS):
        res = _paired(vals[(roi, "encoding", k)],
                      vals[(roi, "after_encoding", k)])
        if res:
            prows.append(dict(roi=roi, condition=k,
                              contrast="encoding-after", **res))
    P = pd.DataFrame(prows)
    P["q"] = _bh(P["p"]) if len(P) else np.nan

    _report(T, W, I, P, RR, rois, unit, ref_roi)

    if not save:
        return None
    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(G, f"{ANALYSIS_NAME}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    T.to_csv(os.path.join(out_dir, "coupling_by_roi_condition.csv"), index=False)
    W.to_csv(os.path.join(out_dir, "within_roi_contrasts.csv"), index=False)
    I.to_csv(os.path.join(out_dir, "interaction_vs_ref.csv"), index=False)
    P.to_csv(os.path.join(out_dir, "phase_contrasts.csv"), index=False)
    RR.to_csv(os.path.join(out_dir, "roi_vs_roi_within_condition.csv"),
              index=False)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"analysis": ANALYSIS_NAME, "created": stamp,
                   "cond_dir": cond_dir, "partition_dir": partition_dir,
                   "unit": unit, "reference_roi": ref_roi,
                   "peri_ms": PERI_MS, "smoothing": "off for statistics",
                   "coverage_limit": ("the conditions run extracted FRONTAL "
                                      "derivations only; temporal/auditory "
                                      "ROIs need a new extraction"),
                   "rois": rois,
                   "coupling": T.to_dict("records"),
                   "within_roi": W.to_dict("records"),
                   "interaction_vs_ref": I.to_dict("records"),
                   "phase": P.to_dict("records"),
                   "roi_vs_roi_within_condition": RR.to_dict("records")}, f, indent=2, default=str)
    _figure(T, W, I, rois, out_dir, unit, ref_roi)
    print(f"\nsaved -> {out_dir}")
    return None


def _latest(G, prefix):
    d = sorted(x for x in os.listdir(G) if x.startswith(prefix))
    if not d:
        raise SystemExit(f"no {prefix}* directory in {G}")
    return os.path.join(G, d[-1])


def _report(T, W, I, P, RR, rois, unit, ref_roi):
    f4 = lambda v: "" if pd.isna(v) else f"{v:+.4f}"
    f3 = lambda v: "" if pd.isna(v) else f"{v:.3f}"
    print(f"\n{'='*94}\nCOUPLING BY ROI x CONDITION x PHASE "
          f"(mean over {unit}s, peri ±{PERI_MS:.0f} ms)\n{'='*94}")
    for ph in PHASES:
        print(f"  --- {ph} " + "-" * (70 - len(ph)))
        print(f"  {'ROI':<14s}" + "".join(f"{LABEL[k][:14]:>18s}"
                                          for k in COND_KEYS))
        for roi in rois:
            cells = []
            for k in COND_KEYS:
                r = T[(T.roi == roi) & (T.phase == ph) & (T.condition == k)]
                if not len(r):
                    cells.append(f"{'':>18s}")
                    continue
                r = r.iloc[0]
                star = "*" if (np.isfinite(r.get("q", np.nan))
                               and r["q"] < 0.05) else " "
                cells.append(f"{f4(r.coupling)}±{r['sem']:.4f}{star} n{r.n_units:<2d}"
                             .rjust(18))
            print(f"  {roi:<14s}" + "".join(cells))
    print(f"  * q < 0.05 vs zero (BH over the whole table)")

    print(f"\n{'='*94}\nWITHIN ROI: is the coupling about reward or about "
          f"pressing a key?\n{'='*94}")
    print(f"  {'ROI':<14s}{'phase':<16s}{'contrast':<16s}{'n':>4s}"
          f"{'difference':>13s}{'p':>8s}{'q':>8s}")
    for _, r in W.sort_values(["roi", "phase"]).iterrows():
        print(f"  {r.roi:<14s}{r.phase:<16s}{r.contrast:<16s}{r.n:>4d}"
              f"{f4(r['diff'])}±{r['sem']:.4f}{f3(r.p):>8s}{f3(r.q):>8s}")

    print(f"\n{'='*94}\nINTERACTION: is the condition difference BIGGER than "
          f"in {ref_roi}?\n{'='*94}")
    if not len(I):
        print("  (too few shared units)")
    else:
        print(f"  {'ROI':<14s}{'phase':<16s}{'contrast':<14s}{'n':>4s}"
              f"{'vs ' + ref_roi:>14s}{'p':>8s}{'q':>8s}   verdict")
        for _, r in I.sort_values(["roi", "phase"]).iterrows():
            v = ("DIFFERENT" if (np.isfinite(r.q) and r.q < 0.05)
                 else "no difference")
            print(f"  {r.roi:<14s}{r.phase:<16s}{r.contrast:<14s}{r.n:>4d}"
                  f"{f4(r['diff'])}±{r['sem']:.4f}{f3(r.p):>8s}{f3(r.q):>8s}   {v}")

    print(f"\n{'='*94}\nROI vs ROI WITHIN a condition: which region couples "
          f"more to the same event?\n{'='*94}")
    if not len(RR):
        print("  (too few shared units)")
    else:
        sig = RR[RR.q < 0.05]
        print(f"  {len(sig)} of {len(RR)} pairs survive BH at q < 0.05")
        show = sig if len(sig) else RR.nsmallest(8, "p")
        head = ("SIGNIFICANT" if len(sig)
                else "none significant; the 8 smallest p for orientation")
        print(f"  --- {head} ---")
        print(f"  {'phase':<16s}{'condition':<11s}{'A':<13s}{'B':<13s}{'n':>4s}"
              f"{'A - B':>13s}{'p':>8s}{'q':>8s}")
        for _, r in show.sort_values("p").iterrows():
            print(f"  {r.phase:<16s}{r.condition:<11s}{r.roi_a:<13s}"
                  f"{r.roi_b:<13s}{r.n:>4d}{f4(r['diff'])}±{r['sem']:.4f}"
                  f"{f3(r.p):>8s}{f3(r.q):>8s}")

    print(f"\n{'='*94}\nPHASE: encoding minus after-encoding, within "
          f"ROI x condition\n{'='*94}")
    print(f"  {'ROI':<14s}{'condition':<16s}{'n':>4s}{'difference':>13s}"
          f"{'p':>8s}{'q':>8s}")
    for _, r in P.sort_values(["roi", "condition"]).iterrows():
        print(f"  {r.roi:<14s}{r.condition:<16s}{r.n:>4d}"
              f"{f4(r['diff'])}±{r['sem']:.4f}{f3(r.p):>8s}{f3(r.q):>8s}")


def _figure(T, W, I, rois, out_dir, unit, ref_roi):
    """Grouped bars: ROI x condition, one panel per phase, * = q < 0.05 vs 0."""
    import matplotlib.pyplot as plt

    with plt.rc_context(rfig._rc()):
        n_ph = len(PHASES)
        fig, axes = plt.subplots(1, n_ph, sharey=True,
                                 figsize=(4.6 * n_ph * rfig.CM, 9.5 * rfig.CM))
        axes = np.atleast_1d(axes)
        w = 0.26
        for ax, ph in zip(axes, PHASES):
            for j, k in enumerate(COND_KEYS):
                x, y, e, q = [], [], [], []
                for i, roi in enumerate(rois):
                    r = T[(T.roi == roi) & (T.phase == ph) & (T.condition == k)]
                    if not len(r):
                        continue
                    r = r.iloc[0]
                    x.append(i + (j - 1) * w)
                    y.append(r.coupling)
                    e.append(r["sem"])
                    q.append(r.get("q", np.nan))
                ax.bar(x, y, w, yerr=e, color=COL[k], lw=0,
                       error_kw=dict(lw=0.6, ecolor="#3d3d3d"),
                       label=LABEL[k] if ax is axes[0] else None)
                for xi, yi, ei, qi in zip(x, y, e, q):
                    if np.isfinite(qi) and qi < 0.05:
                        ax.text(xi, yi + ei + 0.0008, "*", ha="center",
                                fontsize=rfig.FS_TICK, color="#0e3d3a")
            ax.axhline(0, color="#3d3d3d", lw=0.7)
            ax.set_xticks(range(len(rois)))
            ax.set_xticklabels(rois, rotation=55, ha="right",
                               fontsize=rfig.FS_TICK - 1)
            ax.set_title(ph.replace("_", " "), fontsize=rfig.FS_TITLE)
            # MedialFrontal is a coordinate cut across the other frontal rows,
            # so it is separated rather than read as one more region.
            ax.axvline(0.5, color="#bbbbbb", lw=0.7, ls=":")
        # The after-encoding error cells rest on 3-5 sessions and their bars
        # run to +-0.06, which flattens every other bar in the figure. The
        # y-limits follow the bars that carry the result; the few that overrun
        # are still drawn, and their numbers are in the table.
        vals = T[["coupling", "sem"]].dropna()
        # `vals["sem"]`, never `vals.sem` -- that attribute is the DataFrame
        # method, and it fails loudly here but silently elsewhere.
        lo = float((vals["coupling"] - vals["sem"]).quantile(0.02))
        hi = float((vals["coupling"] + vals["sem"]).quantile(0.98))
        pad = 0.25 * (hi - lo)
        axes[0].set_ylim(lo - pad, hi + pad)
        axes[0].set_ylabel(f"ripple-HFB coupling\n(peri ±{PERI_MS:.0f} ms, "
                           f"real − null)\nmean ± SEM over {unit}s",
                           fontsize=rfig.FS_TICK)
        axes[0].legend(frameon=False, fontsize=rfig.FS_TICK - 1,
                       loc="upper left", handlelength=1.1, labelspacing=0.2)
        fig.suptitle("Ripple-HFB coupling by ROI and condition — * q < 0.05 vs "
                     "zero (BH over all 90 cells)\nFRONTAL ROIs ONLY: the "
                     "conditions extraction never covered temporal or auditory "
                     "derivations.\nMedialFrontal (left of the dotted line) is "
                     "the |x| ≤ 20 mm cut across mPFC+mOFC and overlaps them.",
                     fontsize=rfig.FS_TICK - 1)
        fig.tight_layout()
        fig.subplots_adjust(top=1 - 2.3 / 9.5)
        stem = os.path.join(out_dir, "roi_condition_coupling")
        fig.savefig(stem + ".pdf")
        fig.savefig(stem + ".png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
