#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is the frontal HFB rise about RIPPLES, or just about uncovering a reward?

Everything so far locks HFB to the ripple. That cannot distinguish two stories:

  A  the hippocampus signals frontal cortex at the ripple, and the uncovering
     is just what brought the ripple about;
  B  frontal HFB rises after a rewarded uncovering whatever the hippocampus
     does, and the ripple is incidental.

This locks HFB to the EVENT instead, and splits the events by whether a ripple
followed. B predicts the same event-locked rise with and without a ripple. A
predicts a bigger rise when a ripple follows.

    reward_<phase>_rip   correct uncovering, >=1 ripple within RIPPLE_WIN_S
    reward_<phase>_norip same, no ripple in that window
    ... and the same for error_ and move_

WHY THE BASELINE IS DIFFERENT HERE
----------------------------------
The ripple-locked analysis normalises each epoch to its own 250-750 ms flanks,
which is symmetric around the ripple and fine there. Around an EVENT that
window sits squarely on top of the evoked response it is meant to baseline, so
it would subtract the effect. Event-locked epochs are therefore baselined to
-1500..-750 ms, PRE-event. Both estimators are written out -- `pre` (primary)
and `flank` (for comparability with the ripple-locked numbers) -- because the
choice changes the number and should be visible rather than buried.

The shifted null is kept exactly as elsewhere: the same SHIFT_RANGE_S, the same
seed, the event label always taken from the REAL event time and only the HFB
sampling point moved.

Derivations are different-shaft only. Volume conduction from the ripple is not
a concern when nothing is locked to a ripple, but keeping the same derivation
set is what makes the event-locked and ripple-locked numbers comparable.

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_hfb_event_locked.py run
    python scripts/swr_hfb_event_locked.py figure --results=<dir>

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import pickle
import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.swr_bundle as sb
import mc.plotting.ripple_figures as rfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swr_ripple_hfb_conditions import (session_events, _clean_mask,
                                       SHIFT_RANGE_S, POST_S)
from swr_hfb_locking_contrasts import COL, LABEL

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "hfb_event_locked"
HALF_S = 1.5                   # wider than the ripple epoch: a pre-event
                               # baseline needs room the +-1 s epoch lacks
PRE_BASE_S = (-1.5, -1.0)      # baseline for the event-locked estimator
POST_WIN_S = (0.0, 0.5)        # the response window, post-event
FLANK_S = (0.25, 0.75)         # the ripple-locked convention, for comparison
PERI_S = 0.25
RIPPLE_WIN_S = POST_S          # "a ripple followed" = one within this window
# THREE ripple labels per event, because "independent of ripples" has more than
# one reasonable meaning and they are not nested:
#   rip     >=1 ripple in (0, RIPPLE_WIN_S] after the event
#   norip   no ripple in that window -- says nothing about before the event
#   clean   NO RIPPLE ANYWHERE in the analysis epoch [-1.5, +0.5] s. This is
#           the strict reading: the epoch itself is ripple-free, so nothing in
#           the measured window can be ripple-driven.
CLEAN_WIN_S = (-1.5, 0.5)
MIN_EVENTS = 15
KINDS = ("reward", "error", "move")
PHASES3 = ("explore", "plan", "execute")


def run(bundle=None, out_dir=None, n_shifts=6, seed=42, boundary="exclusive",
        save=True):
    b_dir = bundle or os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                                   "group", "swr", "bundle_v2")
    with open(os.path.join(b_dir, "swr_bundle.pkl"), "rb") as f:
        B = pickle.load(f)
    rip, hfb_pairs, hiv, pairs = (B["ripples"], B["hfb_pairs"],
                                  B["hfb_intervals"], B["pairs"])
    hp = hfb_pairs[~hfb_pairs.excluded.fillna(False)].copy()
    key = pairs[["session", "pair_id", "same_probe_as_hpc"]].drop_duplicates()
    hp = hp.merge(key, on=["session", "pair_id"], how="left")
    hp = hp[~hp.same_probe_as_hpc.fillna(False).astype(bool)]

    R = swr_io.get_data_root()
    rng = np.random.default_rng(seed)
    lo, hi = SHIFT_RANGE_S
    shifts = [0.0] + [float(x) for x in
                      rng.uniform(lo, hi, n_shifts)
                      * rng.choice([-1, 1], n_shifts)]
    fs = float(B["hfb_index"].out_fs.iloc[0]) if len(B["hfb_index"]) else 100.0
    w = int(HALF_S * fs)
    off = np.arange(-w, w)
    t_s = off / fs
    peri = np.abs(t_s) < PERI_S
    post = (t_s >= POST_WIN_S[0]) & (t_s < POST_WIN_S[1])
    pre = (t_s >= PRE_BASE_S[0]) & (t_s < PRE_BASE_S[1])
    flank = (np.abs(t_s) >= FLANK_S[0]) & (np.abs(t_s) < FLANK_S[1])

    rows, tc, tc_ix, counts = [], [], [], []
    for sess in sorted(hp.session.unique()):
        g = hp[hp.session == sess]
        # ALL cortical families. The frontal-only restriction was inherited
        # from `swr_ripple_hfb_conditions.py`, where it was a cost decision,
        # and it is what left MTG, STG and Auditory with no event-locked data
        # at all. The HFB exists for every one of these derivations.
        cx = g[g.roi_family.isin(["mPFC", "mOFC", "TemporalLateral",
                                  "Auditory", "Visual"])]
        if not len(cx):
            continue
        ev = session_events(int(sess), R, boundary)
        if not len(ev):
            continue
        try:
            D = sb.load_hfb(b_dir, sess, arrays=["hfb"])
        except Exception:
            continue
        H, ids = D["hfb"], D["pair_ids"]
        n = H.shape[1]
        idx = {p: i for i, p in enumerate(ids)}
        masks = {p: _clean_mask(
            hiv[(hiv.session == sess) & (hiv.pair_id == p)]
            [["start_s", "stop_s"]].to_numpy(float), n, fs) for p in g.pair_id}

        rt = np.unique(np.round(
            rip.loc[rip.session == sess, "t_peak_s"].to_numpy(float), 3))
        # Did a ripple follow this event, within RIPPLE_WIN_S? Pooled over
        # hippocampal derivations, matching how the ripple-locked branch pools
        # them before assigning conditions.
        et = ev.t_s.to_numpy(float)
        j = np.searchsorted(rt, et, side="right")
        has_rip = np.zeros(len(et), bool)
        okj = j < len(rt)
        has_rip[okj] = (rt[np.clip(j, 0, len(rt) - 1)][okj] - et[okj]
                        ) <= RIPPLE_WIN_S
        # strict: no ripple anywhere in the epoch
        lo_c, hi_c = CLEAN_WIN_S
        il = np.searchsorted(rt, et + lo_c, side="left")
        ih = np.searchsorted(rt, et + hi_c, side="right")
        clean = (ih - il) == 0
        base = ev.cond.to_numpy(object)
        labs = [np.array([f"{c}_{'rip' if h else 'norip'}"
                          for c, h in zip(base, has_rip)], dtype=object),
                np.array([f"{c}_clean" if k else "" for c, k in
                          zip(base, clean)], dtype=object)]
        lab = np.concatenate(labs)
        et2 = np.concatenate([et, et])
        keep = lab != ""
        lab, et2 = lab[keep], et2[keep]

        for c in sorted(set(lab)):
            counts.append({"session": sess, "cond": c,
                           "n_events": int((lab == c).sum())})

        for sh in shifts:
            for c in sorted(set(lab)):
                t0 = et2[lab == c]
                if len(t0) < MIN_EVENTS:
                    continue
                ci = np.round((t0 + sh) * fs).astype(int)
                inb = (ci - w >= 0) & (ci + w < n)
                ci = ci[inb]
                if len(ci) < MIN_EVENTS:
                    continue
                win = ci[:, None] + off[None, :]
                for _, x in cx.iterrows():
                    if x.pair_id not in idx:
                        continue
                    ok = masks[x.pair_id][win].all(1)
                    if ok.sum() < MIN_EVENTS:
                        continue
                    m = H[idx[x.pair_id]][win[ok]].mean(0)
                    tc.append(m)
                    tc_ix.append({"session": sess, "subject": x.subject_label,
                                  "cx_pair": x.pair_id, "roi": x.roi_family,
                                  "cond": c, "is_real": sh == 0.0,
                                  "trace_row": len(tc) - 1,
                                  "n_events": int(ok.sum())})
                    rows.append({
                        "session": sess, "subject": x.subject_label,
                        "cx_pair": x.pair_id, "roi": x.roi_family, "cond": c,
                        "is_real": sh == 0.0, "shift_s": sh,
                        "n_events": int(ok.sum()),
                        # the requested estimator: 0-500 ms post-event minus
                        # the -1.5..-1.0 s pre-event baseline
                        "post_pre": float((m - m[pre].mean())[post].mean()),
                        "peri_pre": float((m - m[pre].mean())[peri].mean()),
                        "peri_flank": float((m - m[flank].mean())[peri].mean())})
        print(f"  s{int(sess):02d}: {len(set(lab))} conditions", end="\r")

    d = pd.DataFrame(rows)
    ix = pd.DataFrame(tc_ix)
    print(f"\n{len(ix)} traces, {d.session.nunique()} sessions, "
          f"{d.cx_pair.nunique()} cortical pair labels")
    if not save:
        return d, ix

    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(
        swr_io.derivatives_dir(swr_io.get_data_root()), "group", "swr",
        f"{ANALYSIS_NAME}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "timecourses.npz"),
                        t_ms=t_s * 1000.0, traces=np.stack(tc))
    ix.to_csv(os.path.join(out_dir, "timecourse_index.csv"), index=False)
    d.to_csv(os.path.join(out_dir, "per_pair.csv"), index=False)
    pd.DataFrame(counts).to_csv(os.path.join(out_dir, "event_counts.csv"),
                                index=False)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"analysis": ANALYSIS_NAME, "created": stamp,
                   "bundle": b_dir, "boundary": boundary,
                   "half_s": HALF_S, "peri_s": PERI_S,
                   "pre_baseline_s": list(PRE_BASE_S),
                   "post_window_s": list(POST_WIN_S),
                   "clean_window_s": list(CLEAN_WIN_S),
                   "flank_baseline_s": list(FLANK_S),
                   "ripple_window_s": RIPPLE_WIN_S,
                   "min_events": MIN_EVENTS, "n_shifts": n_shifts,
                   "seed": seed, "shift_range_s": list(SHIFT_RANGE_S),
                   "different_shaft_only": True,
                   "note": ("locked to the EVENT, split by whether a ripple "
                            "followed within ripple_window_s")}, f, indent=2)
    print(f"\nsaved -> {out_dir}")
    return None


def figure(results=None, cond_dir=None, partition_dir=None, out_stem=None,
           rois=("MedialFrontal", "lOFC"), phase="explore", smooth_ms=100.0):
    """Ripple-locked beside event-locked, and event-locked split by ripple.

    Three columns, one row per ROI:
      1  RIPPLE-locked, correct uncovers -- the published effect
      2  EVENT-locked, all correct uncovers -- "does HFB just rise after a
         reward?"
      3  EVENT-locked, split by whether a ripple followed -- "does the event
         response need the ripple?"

    Columns 1 and 2 use DIFFERENT baselines (flank vs pre-event) and different
    epoch widths, for the reasons in the module docstring, so their heights are
    not directly comparable. What is comparable is whether each is different
    from zero, which is what the panel is for.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from swr_hfb_roi_compare import _pooled

    G = os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                     "group", "swr")
    results = results or sorted(os.path.join(G, x) for x in os.listdir(G)
                                if x.startswith(f"{ANALYSIS_NAME}_"))[-1]
    cond_dir = cond_dir or sorted(
        os.path.join(G, x) for x in os.listdir(G)
        if x.startswith("ripple_hfb_conditions_exclusive_"))[-1]
    partition_dir = partition_dir or sorted(
        os.path.join(G, x) for x in os.listdir(G)
        if x.startswith("hfb_roi_partition_"))[-1]
    part = pd.read_csv(os.path.join(partition_dir,
                                    "derivation_assignment.csv")) \
        .set_index(["session", "cx_pair"])["roi_part"]

    def _load(d, relabel):
        z = np.load(os.path.join(d, "timecourses.npz"))
        ix = pd.read_csv(os.path.join(d, "timecourse_index.csv"))
        ix["is_real"] = ix.is_real.astype(bool)
        if "trace_row" not in ix.columns:
            ix["trace_row"] = np.arange(len(ix))
        if relabel:
            fam = ix[ix.roi.isin(("mPFC", "mOFC"))].copy()
            fam["roi"] = pd.MultiIndex.from_arrays(
                [fam.session, fam.cx_pair]).map(part)
            keep = ix[ix.roi == "MedialFrontal"]
            ix = pd.concat([fam, keep], ignore_index=True).dropna(subset=["roi"])
        return z["t_ms"], z["traces"], ix

    tR, trR, ixR = _load(cond_dir, True)
    cR = pd.read_csv(os.path.join(cond_dir, "ripple_counts.csv"))
    tE, trE, ixE = _load(results, False)
    ixE["roi"] = pd.MultiIndex.from_arrays(
        [ixE.session, ixE.cx_pair]).map(part)
    mfk = set(map(tuple, ixR[ixR.roi == "MedialFrontal"]
                  [["session", "cx_pair"]].drop_duplicates().to_numpy()))
    ixE.loc[[(s, c) in mfk for s, c in zip(ixE.session, ixE.cx_pair)],
            "roi"] = "MedialFrontal"
    cE = (pd.read_csv(os.path.join(results, "event_counts.csv"))
            .rename(columns={"n_events": "n_ripples"}))

    def _sm(A):
        if not smooth_ms:
            return A
        fs = 1000.0 / (tE[1] - tE[0])
        return gaussian_filter1d(A, (smooth_ms / 1000.0 * fs) / 2.355, axis=-1)

    def _ms(cur):
        A = _sm(np.stack(list(cur.values())))
        return A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A)), len(A)

    with plt.rc_context(rfig._rc()):
        fig, axes = plt.subplots(len(rois), 3,
                                 figsize=(17 * rfig.CM,
                                          5.4 * len(rois) * rfig.CM))
        axes = np.atleast_2d(axes)
        for r, roi in enumerate(rois):
            # 1 ripple-locked, correct uncovers
            cur = _pooled(tR, trR, ixR[ixR.roi == roi], cR,
                          (f"reward_{phase}",), "session")
            if cur:
                m, se, n = _ms(cur)
                ax = axes[r, 0]
                ax.fill_between(tR, m - se, m + se, color=COL["reward"],
                                alpha=0.2, lw=0)
                ax.plot(tR, m, color=COL["reward"], lw=1.3)
                ax.set_title(f"{roi} — RIPPLE-locked\ncorrect uncovers "
                             f"(n={n})", fontsize=rfig.FS_TICK)
            # 2 event-locked, all correct uncovers
            sub = ixE[ixE.roi == roi]
            cur = _pooled(tE, trE, sub, cE,
                          (f"reward_{phase}_rip", f"reward_{phase}_norip"),
                          "session")
            if cur:
                m, se, n = _ms(cur)
                ax = axes[r, 1]
                ax.fill_between(tE, m - se, m + se, color=COL["reward"],
                                alpha=0.2, lw=0)
                ax.plot(tE, m, color=COL["reward"], lw=1.3)
                ax.set_title(f"{roi} — EVENT-locked\nall correct uncovers "
                             f"(n={n})", fontsize=rfig.FS_TICK)
            # 3 event-locked, split by ripple
            ax = axes[r, 2]
            for suf, col, ls in (("rip", "#0e3d3a", "-"),
                                 ("norip", "#9ab5ad", (0, (4, 1.5)))):
                cur = _pooled(tE, trE, sub, cE,
                              (f"reward_{phase}_{suf}",), "session")
                if not cur:
                    continue
                m, se, n = _ms(cur)
                ax.fill_between(tE, m - se, m + se, color=col, alpha=0.18, lw=0)
                ax.plot(tE, m, color=col, lw=1.3, ls=ls,
                        label=("ripple followed" if suf == "rip"
                               else "no ripple") + f" ({n})")
            ax.legend(frameon=False, fontsize=rfig.FS_TICK - 2,
                      loc="upper left")
            ax.set_title(f"{roi} — EVENT-locked\nsplit by ripple",
                         fontsize=rfig.FS_TICK)
            for c in range(3):
                a = axes[r, c]
                a.axhline(0, color="0.6", lw=0.6)
                a.axvline(0, color="#3d3d3d", lw=0.6, ls=":")
                a.set_xlim(-750, 750)
                if r == len(rois) - 1:
                    a.set_xlabel("ms from ripple" if c == 0
                                 else "ms from uncovering")
            axes[r, 0].set_ylabel("HFB, real − null\n(z)")
        fig.suptitle("Is the frontal HFB rise about the ripple, or just about "
                     f"uncovering a reward?   ({phase} phase)\n"
                     "column 1 is flank-baselined and column 2-3 pre-event "
                     "baselined, so heights are NOT comparable across columns",
                     fontsize=rfig.FS_TICK)
        # The panel titles are two lines each, so the suptitle needs its own
        # band; tight_layout alone lets the two collide.
        fig.tight_layout()
        fig.subplots_adjust(top=1 - 2.4 / (5.4 * len(rois)), hspace=0.65)
        out_stem = out_stem or os.path.join(results, f"event_vs_ripple_{phase}")
        fig.savefig(out_stem + ".pdf")
        fig.savefig(out_stem + ".png", dpi=300)
        plt.close(fig)
    print(f"  -> {out_stem}.pdf / .png")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run, "figure": figure})
    else:
        run()
