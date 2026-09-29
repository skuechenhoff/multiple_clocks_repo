#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two axes: how much is a region locked to the RIPPLE, and how much to the EVENT?

The question this answers is "is medial frontal ripple-locked while lOFC is
reward-locked?". That is a statement about two different quantities, so it is
drawn as a plane rather than a contrast:

  x  RIPPLE-LOCKED coupling. Peri (+-250 ms) minus non-peri (250-750 ms) of the
     ripple-locked curve, real minus shifted null.
  y  EVENT-LOCKED response in RIPPLE-FREE epochs. 0 to +500 ms minus the
     -1.5..-1.0 s pre-event baseline, for events with NO ripple anywhere in
     the epoch, real minus shifted null.

THE UNITS ARE NOT THE SAME and the two are deliberately NOT subtracted. One is
symmetric around a ripple, the other causal around an event; a difference
between them would be an artefact of the window shapes. What the plane supports
is a statement about WHICH AXIS a region loads on, with each axis tested
against zero on its own terms.

A region that is purely ripple-driven sits on the x-axis: coupled, no residual
response when the ripple is removed. A region that is event-driven sits on the
y-axis. A region that is both sits off-diagonal.

The second figure is a SLIDING WINDOW over the event-locked epoch -- 250 ms
wide, 50 ms steps -- which shows when any event response actually happens
rather than assuming the 0-500 ms window is the right one. That window was
chosen from the literature, not from these data, and the sweep is how that
choice gets checked.

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_hfb_ripple_vs_event.py run
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
from swr_hfb_event_specificity import (_prep, _series, KINDS, KIND_SRC, PRE,
                                       POST, PHASES3)
from swr_hfb_roi_partition import _bh, ROI_C
from swr_hfb_roi_compare import _pooled
from swr_ripple_locked_hfb import _effect

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "hfb_ripple_vs_event"
PLANE_ROIS = ("MedialFrontal", "lOFC", "MTG", "FrontalPole", "Auditory",
              "STG", "TempOcc_Fusiform", "Occipital", "vlPFC_IFG")
SLIDE_ROIS = ("MedialFrontal", "lOFC", "MTG")
WIN_MS, STEP_MS = 250.0, 50.0


def _ripple_locked(part_dir, unit="session"):
    """Per-unit ripple-locked coupling for every partition ROI."""
    G = os.path.dirname(part_dir)
    R = os.path.join(G, "ripple_locked_hfb_2026-09-18")
    d = pd.read_csv(os.path.join(R, "per_pair.csv"))
    d = d[(d.pad_s == d.pad_s.min()) & (~d.same_shaft)]
    part = pd.read_csv(os.path.join(part_dir, "derivation_assignment.csv")) \
        .set_index(["session", "cx_pair"])["roi_part"]
    base = d[d.roi.isin(["mPFC", "mOFC", "TemporalLateral", "Auditory",
                         "Visual"])].copy()
    base["rp"] = pd.MultiIndex.from_arrays(
        [base.session, base.cx_pair]).map(part)
    return {roi: _effect(g, unit) for roi, g in base.groupby("rp")}


def run(ev_dir=None, part_dir=None, out_dir=None, kind="reward", save=True):
    from scipy import stats as st

    G = os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                     "group", "swr")
    ev_dir = ev_dir or sorted(os.path.join(G, x) for x in os.listdir(G)
                              if x.startswith("hfb_event_locked_all_"))[-1]
    part_dir = part_dir or sorted(
        os.path.join(G, x) for x in os.listdir(G)
        if x.startswith("hfb_roi_partition_exclusive_"))[-1]
    t_ms, tr, ix, cnt = _prep(ev_dir, part_dir)
    rip = _ripple_locked(part_dir)

    rows = []
    for roi in PLANE_ROIS:
        rv = rip.get(roi, pd.Series(dtype=float))
        ev, _ = _series(t_ms, tr, ix, cnt, roi, kind, "clean")
        if len(rv) < 5 or len(ev) < 5:
            continue
        tr_, pr_ = st.ttest_1samp(rv.to_numpy(), 0.0)
        te_, pe_ = st.ttest_1samp(ev.to_numpy(), 0.0)
        rows.append({"roi": roi,
                     "n_ripple": len(rv), "ripple": float(rv.mean()),
                     "ripple_sem": float(rv.std(ddof=1) / np.sqrt(len(rv))),
                     "ripple_p": float(pr_),
                     "n_event": len(ev), "event": float(ev.mean()),
                     "event_sem": float(ev.std(ddof=1) / np.sqrt(len(ev))),
                     "event_p": float(pe_)})
    P = pd.DataFrame(rows)
    P["ripple_q"] = _bh(P["ripple_p"])
    P["event_q"] = _bh(P["event_p"])

    # ---- sliding window over the event-locked epoch ---------------------
    edges = np.arange(-1000.0, 751.0 - WIN_MS + 1, STEP_MS)
    slide = []
    for roi, k, rl in itertools.product(SLIDE_ROIS, KINDS, ("clean", "rip")):
        conds = tuple(f"{a}_{p}_{rl}" for a in KIND_SRC[k] for p in PHASES3)
        cur = _pooled(t_ms, tr, ix[ix.roi == roi], cnt, conds, "session")
        if len(cur) < 5:
            continue
        pre = (t_ms >= PRE[0]) & (t_ms < PRE[1])
        A = np.stack([c - c[pre].mean() for c in cur.values()])
        for e in edges:
            w = (t_ms >= e) & (t_ms < e + WIN_MS)
            v = A[:, w].mean(1)
            t, p = st.ttest_1samp(v, 0.0)
            slide.append({"roi": roi, "kind": k, "ripple": rl,
                          "centre_ms": float(e + WIN_MS / 2), "n": len(v),
                          "effect": float(v.mean()),
                          "sem": float(v.std(ddof=1) / np.sqrt(len(v))),
                          "p": float(p)})
    S = pd.DataFrame(slide)
    # Correct ACROSS the sweep within each ROI x condition x label: a sliding
    # window is a family of overlapping tests and an uncorrected minimum is
    # guaranteed to look impressive.
    for keys, g in S.groupby(["roi", "kind", "ripple"]):
        S.loc[g.index, "q"] = _bh(g["p"])

    _report(P, S, kind)

    if not save:
        return None
    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(G, f"{ANALYSIS_NAME}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    P.to_csv(os.path.join(out_dir, f"plane_{kind}.csv"), index=False)
    S.to_csv(os.path.join(out_dir, "sliding_window.csv"), index=False)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"analysis": ANALYSIS_NAME, "created": stamp,
                   "event_dir": ev_dir, "partition_dir": part_dir,
                   "kind": kind, "window_ms": WIN_MS, "step_ms": STEP_MS,
                   "axes": {"x": "ripple-locked coupling",
                            "y": "event-locked response, ripple-free epochs"},
                   "note": ("the two axes have different window shapes and are "
                            "NOT subtracted"),
                   "plane": P.to_dict("records"),
                   "sliding": S.to_dict("records")}, f, indent=2)
    _figure(P, S, out_dir, kind)
    print(f"\nsaved -> {out_dir}")
    return None


def _report(P, S, kind):
    print(f"\n{'='*96}\nTHE PLANE: ripple-locked coupling vs event-locked "
          f"response in RIPPLE-FREE epochs\ncondition = {KINDS[kind]}\n{'='*96}")
    print(f"  {'ROI':<18s}{'ripple-locked':>22s}{'event (ripple-free)':>26s}")
    for _, r in P.sort_values("ripple", ascending=False).iterrows():
        s1 = f"{r.ripple:+.4f} q={r.ripple_q:.3f}"
        s2 = f"{r.event:+.4f} q={r.event_q:.3f}"
        print(f"  {r.roi:<18s}{s1:>22s}{s2:>26s}")
    print("  the two axes use different windows and are never subtracted")

    print(f"\n{'='*96}\nSLIDING WINDOW ({WIN_MS:.0f} ms, {STEP_MS:.0f} ms steps),"
          f" event-locked, cells surviving BH within sweep\n{'='*96}")
    sig = S[S.q < 0.05]
    if not len(sig):
        print("  none")
    for keys, g in sig.groupby(["roi", "kind", "ripple"]):
        roi, k, rl = keys
        lo, hi = g.centre_ms.min(), g.centre_ms.max()
        print(f"  {roi:<15s}{KINDS[k]:<20s}{rl:<7s} centres {lo:+.0f}..{hi:+.0f} ms,"
              f" peak {g.loc[g.p.idxmin(),'effect']:+.4f} at "
              f"{g.loc[g.p.idxmin(),'centre_ms']:+.0f} ms (q={g.q.min():.3f})")


def _figure(P, S, out_dir, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(rfig._rc()):
        fig = plt.figure(figsize=(18 * rfig.CM, 15 * rfig.CM))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.55,
                              wspace=0.35)
        # ---- the plane ----------------------------------------------
        ax = fig.add_subplot(gs[0, :2])
        for _, r in P.iterrows():
            col = ROI_C.get(r.roi, "#8C8C8C")
            ax.errorbar(r.ripple, r.event, xerr=r.ripple_sem,
                        yerr=r.event_sem, fmt="o", ms=5, color=col,
                        elinewidth=1.0, capsize=1.5, zorder=3)
            ax.annotate(r.roi, (r.ripple, r.event), fontsize=rfig.FS_TICK - 2,
                        xytext=(5, 4), textcoords="offset points", color=col)
        ax.axhline(0, color="#3d3d3d", lw=0.8, ls="--")
        ax.axvline(0, color="#3d3d3d", lw=0.8, ls="--")
        ax.set_xlabel("RIPPLE-locked coupling\n"
                      "peri (±250 ms) − non-peri, real − shifted null")
        ax.set_ylabel("EVENT-locked response, RIPPLE-FREE epochs\n"
                      "0–500 ms − (−1.5..−1.0 s), real − shifted null")
        ax.set_title(f"Which axis does each region load on?  ({KINDS[kind]})\n"
                     "axes have different window shapes and are never "
                     "subtracted", fontsize=rfig.FS_TITLE)
        # ---- legend panel: the quadrant reading ---------------------
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_axis_off()
        ax2.text(0, 1.0, "reading the plane", fontsize=rfig.FS_LABEL,
                 va="top", weight="bold")
        ax2.text(0, 0.86,
                 "right, on the line\n  ripple-coupled, and nothing\n"
                 "  left when the ripple is gone\n\n"
                 "up and right\n  both ripple- and event-driven\n\n"
                 "up, near zero on x\n  event-driven only\n\n"
                 "Error bars are SEM over sessions.\n"
                 "Each axis is tested against zero\n"
                 "on its own terms; the plane is\n"
                 "not a contrast.",
                 fontsize=rfig.FS_TICK - 1, va="top")
        # ---- sliding windows ----------------------------------------
        # SHARED y-axis across the three sweeps. With independent axes a
        # +0.004 rip-minus-clean gap fills half the MedialFrontal panel
        # (span 0.015) and a quarter of the MTG one (span 0.028), so the
        # same difference reads as 'massive' in one and 'identical' in
        # the other. That is a drawing artefact, not a result.
        sl_ax = None
        for i, roi in enumerate(SLIDE_ROIS):
            ax = fig.add_subplot(gs[1, i], sharey=sl_ax)
            sl_ax = sl_ax or ax
            for k, col, ls in (("reward", "#0e3d3a", "-"),
                               ("move", "#6E6E6E", (0, (4, 1.5)))):
                for rl, alpha in (("clean", 1.0), ("rip", 0.45)):
                    g = S[(S.roi == roi) & (S.kind == k) & (S.ripple == rl)]
                    if not len(g):
                        continue
                    ax.plot(g.centre_ms, g.effect, color=col, lw=1.1, ls=ls,
                            alpha=alpha,
                            label=f"{KINDS[k].split()[0]}, {rl}"
                            if i == 0 else None)
                    s = g[g.q < 0.05]
                    if len(s):
                        ax.scatter(s.centre_ms, s.effect, s=10, color=col,
                                   alpha=alpha, zorder=4, linewidths=0)
            ax.axhline(0, color="0.6", lw=0.6)
            ax.axvline(0, color="#3d3d3d", lw=0.6, ls=":")
            ax.axvspan(POST[0], POST[1], color="#000000", alpha=0.05, lw=0)
            ax.set_title(roi, fontsize=rfig.FS_TICK)
            ax.set_xlabel("window centre (ms from event)",
                          fontsize=rfig.FS_TICK - 1)
            if i == 0:
                ax.set_ylabel(f"event-locked HFB\n({WIN_MS:.0f} ms window)",
                              fontsize=rfig.FS_TICK - 1)
            else:
                ax.tick_params(labelleft=False)
                ax.legend(frameon=False, fontsize=rfig.FS_TICK - 3,
                          loc="upper left")
        # No suptitle: the plane already carries a two-line title and the two
        # collide at this figure height. The sweep note goes on its own panel.
        fig.text(0.5, 0.012, "Filled dots on the sliding windows are q < 0.05 "
                 "within that sweep (BH over the 30 overlapping windows).",
                 ha="center", fontsize=rfig.FS_TICK - 1, color="#3d3d3d")
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        stem = os.path.join(out_dir, f"ripple_vs_event_{kind}")
        fig.savefig(stem + ".pdf")
        fig.savefig(stem + ".png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
