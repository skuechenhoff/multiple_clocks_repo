#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Which regions respond to the EVENT itself, and which only around ripples?

A region whose HFB is ripple-coupled can get there two ways: the hippocampus
signals it at the ripple, or the region simply responds to whatever the subject
just did and ripples happen to follow. The separating test is to lock to the
EVENT and ask whether the response survives when no ripple is present.

`ripple` label, three definitions, none nested in the others:
  rip     >= 1 ripple in (0, +2 s] after the event
  norip   no ripple in that window -- says NOTHING about before the event, so
          an epoch can still contain one at -1 s
  clean   NO RIPPLE ANYWHERE in the analysis epoch, -1.5 to +0.5 s. This is the
          strict reading and the one to quote: the measured window is
          guaranteed ripple-free, so nothing in it can be ripple-driven.

ESTIMATOR: mean HFB over 0 to +500 ms minus the -1.5 to -1.0 s pre-event
baseline, real minus shifted-null, averaged within session. Same shifted null,
seed and shift range as every other branch.

CONDITIONS: correct uncovers, incorrect uncovers, ANY uncover (the two pooled
by event count, which equals the mean over their union), and navigation
presses. "Any uncover" exists because a region coding location rather than
value should care that a tile was turned over at all, not whether it paid out.

⚠ MTG here is MIDDLE temporal gyrus -- lateral temporal neocortex (Harvard-
Oxford "Middle Temporal Gyrus, anterior/posterior division", Brainnetome
A21r/A21c, FreeSurfer ctx-*-middletemporal). It is NOT medial temporal cortex,
not parahippocampal, and not part of the MTL.

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_hfb_event_specificity.py run
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
from swr_hfb_roi_compare import _pooled
from swr_hfb_roi_partition import _bh
from swr_hfb_locking_contrasts import COL
from swr_ripple_locked_hfb import _smooth

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "hfb_event_specificity"
PRE = (-1500.0, -1000.0)
POST = (0.0, 500.0)
PHASES3 = ("explore", "plan", "execute")
ROIS = ("MedialFrontal", "lOFC", "MTG")
# correct / incorrect / both pooled / navigation
KINDS = {"reward": "correct uncovers", "error": "incorrect uncovers",
         "anyuncover": "any uncover", "move": "navigation presses"}
KIND_SRC = {"reward": ("reward",), "error": ("error",),
            "anyuncover": ("reward", "error"), "move": ("move",)}
KCOL = {"reward": COL["reward"], "error": COL["error"], "move": COL["move"],
        "anyuncover": "#23677E"}
RIPLAB = {"clean": "no ripple in epoch", "rip": "ripple followed",
          "norip": "no ripple after"}


def _prep(ev_dir, part_dir, exclude_medial=True):
    z = np.load(os.path.join(ev_dir, "timecourses.npz"))
    ix = pd.read_csv(os.path.join(ev_dir, "timecourse_index.csv"))
    ix["is_real"] = ix.is_real.astype(bool)
    if "trace_row" not in ix.columns:
        ix["trace_row"] = np.arange(len(ix))
    part = pd.read_csv(os.path.join(part_dir, "derivation_assignment.csv")) \
        .set_index(["session", "cx_pair"])["roi_part"]
    # The partition directory is the EXCLUSIVE one, so MedialFrontal already
    # owns its derivations and no other ROI contains them. Relabelling here
    # would be wrong for the same reason it was wrong before: it would empty
    # whichever ROI MedialFrontal is a superset of.
    ix["roi"] = pd.MultiIndex.from_arrays([ix.session, ix.cx_pair]).map(part)
    ix = ix.dropna(subset=["roi"])
    cnt = (pd.read_csv(os.path.join(ev_dir, "event_counts.csv"))
             .rename(columns={"n_events": "n_ripples"}))
    return z["t_ms"], z["traces"], ix, cnt


def _series(t_ms, tr, ix, cnt, roi, kind, riplab, phases=PHASES3):
    conds = tuple(f"{k}_{p}_{riplab}"
                  for k in KIND_SRC[kind] for p in phases)
    cur = _pooled(t_ms, tr, ix[ix.roi == roi], cnt, conds, "session")
    if not cur:
        return pd.Series(dtype=float), {}
    pre = (t_ms >= PRE[0]) & (t_ms < PRE[1])
    post = (t_ms >= POST[0]) & (t_ms < POST[1])
    vals, curves = {}, {}
    for u, c in cur.items():
        cb = c - c[pre].mean()
        vals[u] = float(cb[post].mean())
        curves[u] = cb
    return pd.Series(vals), curves


def run(ev_dir=None, part_dir=None, out_dir=None, save=True):
    from scipy import stats as st

    G = os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                     "group", "swr")
    ev_dir = ev_dir or sorted(os.path.join(G, x) for x in os.listdir(G)
                              if x.startswith("hfb_event_locked_all_"))[-1]
    part_dir = part_dir or sorted(
        os.path.join(G, x) for x in os.listdir(G)
        if x.startswith("hfb_roi_partition_exclusive_"))[-1]
    t_ms, tr, ix, cnt = _prep(ev_dir, part_dir)
    print(f"events from {os.path.basename(ev_dir)}")
    print(f"partition   {os.path.basename(part_dir)} (MedialFrontal exclusive)")

    rows, store = [], {}
    for roi, kind, rl in itertools.product(ROIS, KINDS, ("clean", "rip",
                                                         "norip")):
        v, curves = _series(t_ms, tr, ix, cnt, roi, kind, rl)
        store[(roi, kind, rl)] = (v, curves)
        if len(v) < 5:
            continue
        t, p = st.ttest_1samp(v.to_numpy(), 0.0)
        rows.append({"roi": roi, "kind": kind, "ripple": rl, "n": len(v),
                     "effect": float(v.mean()),
                     "sem": float(v.std(ddof=1) / np.sqrt(len(v))),
                     "t": float(t), "p": float(p)})
    T = pd.DataFrame(rows)
    # THE specificity test: is the event response bigger when a ripple is in
    # the epoch than when the epoch is ripple-free? Paired on the sessions
    # that contribute both, so it is a within-session contrast.
    crows = []
    for roi, kind in itertools.product(ROIS, KINDS):
        a = store[(roi, kind, "rip")][0]
        b = store[(roi, kind, "clean")][0]
        sh = a.index.intersection(b.index)
        if len(sh) < 5:
            continue
        d = (a[sh] - b[sh]).to_numpy()
        t, p = st.ttest_1samp(d, 0.0)
        crows.append({"roi": roi, "kind": kind, "n": len(sh),
                      "rip": float(a[sh].mean()), "clean": float(b[sh].mean()),
                      "diff": float(d.mean()),
                      "sem": float(d.std(ddof=1) / np.sqrt(len(d))),
                      "t": float(t), "p": float(p)})
    C = pd.DataFrame(crows)
    C["q"] = _bh(C["p"]) if len(C) else np.nan
    # Correction over the CLEAN cells: that is the screen. rip/norip are
    # supporting detail, corrected separately.
    for rl, g in T.groupby("ripple"):
        T.loc[g.index, "q"] = _bh(g["p"])
    _report(T, C)

    if not save:
        return None
    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(G, f"{ANALYSIS_NAME}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    T.to_csv(os.path.join(out_dir, "event_specificity.csv"), index=False)
    C.to_csv(os.path.join(out_dir, "ripple_vs_clean.csv"), index=False)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"analysis": ANALYSIS_NAME, "created": stamp,
                   "event_dir": ev_dir, "partition_dir": part_dir,
                   "pre_baseline_ms": list(PRE), "post_window_ms": list(POST),
                   "ripple_labels": RIPLAB,
                   "note": ("MTG is MIDDLE temporal gyrus, lateral temporal "
                            "neocortex, NOT medial temporal cortex"),
                   "results": T.to_dict("records"),
                   "ripple_vs_clean": C.to_dict("records")}, f, indent=2)
    _figure(t_ms, store, T, out_dir)
    print(f"\nsaved -> {out_dir}")
    return None


def _report(T, C=None):
    print(f"\n{'='*104}\nEVENT-LOCKED HFB, {POST[0]:.0f}-{POST[1]:.0f} ms post-event "
          f"minus {PRE[0]:.0f}..{PRE[1]:.0f} ms baseline\n{'='*104}")
    for rl in ("clean", "rip", "norip"):
        g = T[T.ripple == rl]
        if not len(g):
            continue
        print(f"  --- ripple label: {rl} ({RIPLAB[rl]}) ---")
        print(f"  {'ROI':<15s}{'condition':<20s}{'n':>4s}{'effect':>11s}"
              f"{'t':>8s}{'p':>9s}{'q':>8s}")
        for _, r in g.iterrows():
            star = "*" if r.q < 0.05 else " "
            print(f"  {r.roi:<15s}{KINDS[r.kind]:<20s}{r.n:>4d}"
                  f"{r.effect:>+11.4f}{r.t:>8.2f}{r.p:>9.4f}{r.q:>7.3f}{star}")
        print()
    if C is not None and len(C):
        print(f"{'='*104}\nSPECIFICITY: (ripple in epoch) MINUS (epoch "
              f"ripple-free), paired within session\n{'='*104}")
        print(f"  {'ROI':<15s}{'condition':<20s}{'n':>4s}{'with ripple':>13s}"
              f"{'ripple-free':>13s}{'difference':>13s}{'p':>9s}{'q':>8s}")
        for _, r in C.iterrows():
            star = "*" if r.q < 0.05 else " "
            print(f"  {r.roi:<15s}{KINDS[r.kind]:<20s}{r.n:>4d}"
                  f"{r.rip:>+13.4f}{r.clean:>+13.4f}{r['diff']:>+13.4f}"
                  f"{r.p:>9.4f}{r.q:>7.3f}{star}")


def _figure(t_ms, store, T, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    kinds = list(KINDS)
    with plt.rc_context(rfig._rc()):
        fig, axes = plt.subplots(len(ROIS), len(kinds) + 1, figsize=(
            4.2 * (len(kinds) + 1) * rfig.CM, 4.6 * len(ROIS) * rfig.CM))
        axes = np.atleast_2d(axes)
        for r, roi in enumerate(ROIS):
            for c, kind in enumerate(kinds):
                ax = axes[r, c]
                for rl, col, ls in (("clean", "#23677E", "-"),
                                    ("rip", "#a30d6c", (0, (4, 1.5)))):
                    v, curves = store[(roi, kind, rl)]
                    if len(curves) < 5:
                        continue
                    A = _smooth(np.stack(list(curves.values())),
                                1000.0 / (t_ms[1] - t_ms[0]), 100.0)
                    m = A.mean(0)
                    se = A.std(0, ddof=1) / np.sqrt(len(A))
                    ax.fill_between(t_ms, m - se, m + se, color=col,
                                    alpha=0.18, lw=0)
                    ax.plot(t_ms, m, color=col, lw=1.2, ls=ls,
                            label=f"{RIPLAB[rl]} ({len(A)})")
                ax.axhline(0, color="0.6", lw=0.6)
                ax.axvline(0, color="#3d3d3d", lw=0.6, ls=":")
                ax.axvspan(POST[0], POST[1], color="#000000", alpha=0.05, lw=0)
                ax.set_xlim(-1500, 750)
                if r == 0:
                    ax.set_title(KINDS[kind], fontsize=rfig.FS_TICK)
                if c == 0:
                    ax.set_ylabel(f"{roi}\nHFB, real − null",
                                  fontsize=rfig.FS_TICK - 1)
                if r == len(ROIS) - 1:
                    ax.set_xlabel("ms from event", fontsize=rfig.FS_TICK - 1)
                if r == 0 and c == 0:
                    ax.legend(frameon=False, fontsize=rfig.FS_TICK - 3,
                              loc="upper left")
            # summary column: the clean-event effect for this ROI
            ax = axes[r, len(kinds)]
            g = T[(T.roi == roi) & (T.ripple == "clean")].set_index("kind")
            y = np.arange(len(kinds))
            for i, kind in enumerate(kinds):
                if kind not in g.index:
                    continue
                rr = g.loc[kind]
                ax.errorbar([rr.effect], [i], xerr=[rr["sem"]], fmt="o",
                            ms=4, color=KCOL[kind], elinewidth=1.0,
                            capsize=1.5)
                if rr.q < 0.05:
                    ax.text(rr.effect, i + 0.28, "*", ha="center",
                            fontsize=rfig.FS_TICK, color="#0e3d3a")
            ax.axvline(0, color="#3d3d3d", lw=0.7, ls="--")
            ax.set_yticks(y)
            ax.set_yticklabels([KINDS[k] for k in kinds],
                               fontsize=rfig.FS_TICK - 2)
            ax.set_ylim(-0.6, len(kinds) - 0.4)
            if r == 0:
                ax.set_title("ripple-free events\n(0–500 ms)",
                             fontsize=rfig.FS_TICK)
            if r == len(ROIS) - 1:
                ax.set_xlabel("HFB, real − null", fontsize=rfig.FS_TICK - 1)
        fig.suptitle("Event-locked HFB: does the response survive when the "
                     "epoch contains NO ripple?\n"
                     f"{POST[0]:.0f}–{POST[1]:.0f} ms post-event minus "
                     f"{PRE[0]:.0f} to {PRE[1]:.0f} ms baseline, real − "
                     "shifted null.  MTG = MIDDLE temporal gyrus (lateral "
                     "neocortex), not medial temporal.",
                     fontsize=rfig.FS_TICK - 1)
        fig.tight_layout()
        fig.subplots_adjust(top=1 - 1.7 / (4.6 * len(ROIS)))
        stem = os.path.join(out_dir, "event_specificity")
        fig.savefig(stem + ".pdf")
        fig.savefig(stem + ".png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
