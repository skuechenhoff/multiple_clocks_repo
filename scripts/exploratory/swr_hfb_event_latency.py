#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
When did the button press happen, relative to the ripple it was assigned to?

WHY THIS MATTERS
----------------
Both lOFC and medial frontal peak SLIGHTLY BEFORE the ripple. Every condition
in this analysis is defined by an event that PRECEDES its ripple -- see the
allocation rule below -- so a pre-ripple HFB rise has an obvious alternative
explanation: it is the response to the button press, and the ripple merely
follows it. Deciding between "ripple-locked" and "press-locked, ripple
incidental" needs the press-to-ripple latency distribution drawn on the same
axis as the HFB, which is what this produces.

THE ALLOCATION RULE, exactly (`swr_ripple_hfb_conditions.session_events` and
`.assign`)
------------------------------------------------------------------------
Events are laid down first, each at its own time:
  * every UNCOVERING, correct or not, in every phase -> `reward_<phase>` or
    `error_<phase>`;
  * every MOVEMENT PRESS, in every phase, inside the grid's A..D window ->
    `move_<phase>`;
  * pseudo-events every 0.5 s with no press or uncover within +-2 s ->
    `still_<phase>`.

Then each ripple takes the label of the single NEAREST PRECEDING event, if that
event is within POST_S = 2.0 s. Ripples with nothing in the preceding 2 s are
unlabelled and drop out.

Three consequences worth stating plainly:
  * the event ALWAYS precedes the ripple. The latency is bounded to (0, 2] s
    by construction, so on a ripple-locked axis the triggering press is ALWAYS
    at negative time. A pre-ripple bump is exactly where a press response
    would land;
  * conditions are mutually exclusive and exhaustive over labelled ripples --
    no ripple is counted twice;
  * a condition is "the most recent thing that happened", not "an event that
    happened nearby". A ripple 1.9 s after a correct uncovering counts as
    `reward_*` even though 1.9 s is a long time in this task.

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_hfb_event_latency.py run
    python scripts/swr_hfb_event_latency.py run --boundary=inclusive

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import pickle
import datetime

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
from swr_ripple_hfb_conditions import session_events, assign, POST_S
from swr_hfb_locking_contrasts import COL, LABEL
from swr_hfb_roi_partition import PHASES_EVENT as PHASES, COND_KEYS

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "hfb_event_latency"


def latencies(bundle_dir, data_root, boundary="exclusive", cache=None):
    """(session, cond, latency) for every labelled ripple.

    Latency = ripple peak minus the time of the event it was assigned to, so
    it is positive by construction and at most POST_S.
    """
    if cache and os.path.exists(cache):
        print(f"  reusing {cache}")
        return pd.read_csv(cache)
    with open(os.path.join(bundle_dir, "swr_bundle.pkl"), "rb") as f:
        rip = pickle.load(f)["ripples"]
    rows = []
    for sess, g in rip.groupby("session"):
        ev = session_events(int(sess), data_root, boundary)
        if not len(ev):
            continue
        t = np.sort(g.t_peak_s.to_numpy(float))
        c = assign(t, ev, POST_S)
        et = ev.t_s.to_numpy(float)
        j = np.clip(np.searchsorted(et, t, side="right") - 1, 0, len(et) - 1)
        lat = t - et[j]
        rows.extend((int(sess), cc, float(ll))
                    for cc, ll in zip(c, lat) if cc is not None)
        print(f"  s{int(sess):02d}: {sum(x is not None for x in c)} labelled",
              end="\r")
    d = pd.DataFrame(rows, columns=["session", "cond", "lat_s"])
    if cache:
        d.to_csv(cache, index=False)
    return d


def first_ripple(bundle_dir, data_root, boundary="exclusive", cache=None,
                 window_s=POST_S):
    """For each EVENT, the latency to the first ripple that follows it.

    This inverts the question `latencies()` answers. There, each RIPPLE takes
    its nearest preceding event, so one uncovering owns every ripple in the
    next 2 s and the distribution is shaped by how many ripples happen to
    occur. Here each EVENT contributes exactly one number -- how long until the
    next ripple -- so the denominator is events and the answer is about ripple
    TIMING rather than ripple count.

    Computed per hippocampal derivation, because a session with four
    hippocampal probes would otherwise contribute four times as many "first"
    ripples and the earliest of them would dominate.

    Two variants per event, and the difference between them matters:
      `any`      the first ripple after the event, full stop;
      `assigned` the first ripple after the event with NO other event in
                 between -- i.e. the one that would actually carry the event's
                 label. Navigation presses come in fast succession, so for them
                 the two differ a lot.

    Events with no ripple in `window_s` are kept as censored rows, because the
    fraction of events that get no ripple at all is part of the answer.
    """
    if cache and os.path.exists(cache):
        print(f"  reusing {cache}")
        return pd.read_csv(cache)
    with open(os.path.join(bundle_dir, "swr_bundle.pkl"), "rb") as f:
        rip = pickle.load(f)["ripples"]
    rows = []
    for sess, g in rip.groupby("session"):
        ev = session_events(int(sess), data_root, boundary)
        if not len(ev):
            continue
        et = ev.t_s.to_numpy(float)
        ec = ev.cond.to_numpy(object)
        order = np.argsort(et)
        et, ec = et[order], ec[order]
        # next event after each event: the boundary for the `assigned` variant
        nxt = np.r_[et[1:], np.inf]
        for pair, gp in g.groupby("pair_id"):
            rt = np.sort(gp.t_peak_s.to_numpy(float))
            if not len(rt):
                continue
            j = np.searchsorted(rt, et, side="right")   # first ripple after
            ok = j < len(rt)
            lat = np.where(ok, rt[np.clip(j, 0, len(rt) - 1)] - et, np.inf)
            gap = nxt - et          # time until the NEXT event
            for i in range(len(et)):
                if lat[i] > window_s:
                    rows.append((int(sess), str(pair), ec[i], np.nan, np.nan,
                                 float(gap[i])))
                    continue
                assigned = (et[i] + lat[i]) < nxt[i]
                rows.append((int(sess), str(pair), ec[i], float(lat[i]),
                             float(lat[i]) if assigned else np.nan,
                             float(gap[i])))
        print(f"  s{int(sess):02d}", end="\r")
    # `gap_to_next_event_s` is recorded because it is the obvious confound in
    # the ripple-centred view: an event that is followed by a long quiet gap
    # OWNS every ripple in that gap, so its ripples appear at long latencies
    # for a reason that has nothing to do with when ripples occur.
    d = pd.DataFrame(rows, columns=["session", "pair_id", "cond",
                                    "first_any_s", "first_assigned_s",
                                    "gap_to_next_event_s"])
    if cache:
        d.to_csv(cache, index=False)
    return d


def first(bundle=None, out_dir=None, boundary="exclusive", cache=None,
          save=True):
    """Event-centred view: time from each uncover/press to the NEXT ripple."""
    root = swr_io.derivatives_dir(swr_io.get_data_root())
    G = os.path.join(root, "group", "swr")
    bundle = bundle or os.path.join(G, "bundle_v2")
    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(G, f"{ANALYSIS_NAME}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    cache = cache or os.path.join(out_dir, f"first_ripple_{boundary}.csv")

    D = first_ripple(bundle, swr_io.get_data_root(), boundary, cache)
    rows = []
    for ph, spec in PHASES.items():
        for k in COND_KEYS:
            g = D[D.cond.isin(spec[k])]
            if not len(g):
                continue
            a = g.first_any_s.to_numpy(float) * 1000.0
            b = g.first_assigned_s.to_numpy(float) * 1000.0
            fin_a, fin_b = np.isfinite(a), np.isfinite(b)
            rows.append({
                "phase": ph, "condition": k,
                "n_event_probe_pairs": int(len(g)),
                "frac_with_ripple_within_2s": float(fin_a.mean()),
                "median_first_any_ms": (float(np.median(a[fin_a]))
                                        if fin_a.any() else np.nan),
                "q25_first_any_ms": (float(np.percentile(a[fin_a], 25))
                                     if fin_a.any() else np.nan),
                "q75_first_any_ms": (float(np.percentile(a[fin_a], 75))
                                     if fin_a.any() else np.nan),
                "frac_assigned": float(fin_b.mean()),
                "median_first_assigned_ms": (float(np.median(b[fin_b]))
                                             if fin_b.any() else np.nan),
                "median_gap_to_next_event_ms": float(np.median(
                    np.minimum(g.gap_to_next_event_s.to_numpy(float),
                               1e4) * 1000.0))})
    T = pd.DataFrame(rows)
    print(f"\n{'='*104}\nEVENT-CENTRED: time from each event to the NEXT "
          f"ripple (per hippocampal derivation)\n{'='*104}")
    print(f"  {'phase':<17s}{'condition':<11s}{'n ev x probe':>13s}"
          f"{'any ripple <2s':>16s}{'median':>9s}{'IQR':>18s}"
          f"{'median (assigned)':>19s}")
    for _, r in T.iterrows():
        iqr = (f"[{r.q25_first_any_ms:.0f}, {r.q75_first_any_ms:.0f}]"
               if np.isfinite(r.q25_first_any_ms) else "")
        print(f"  {r.phase:<17s}{r.condition:<11s}{r.n_event_probe_pairs:>13d}"
              f"{r.frac_with_ripple_within_2s:>16.2f}"
              f"{r.median_first_any_ms:>8.0f}ms{iqr:>18s}"
              f"{r.median_first_assigned_ms:>18.0f}ms"
              f"{r.median_gap_to_next_event_ms:>13.0f}ms")
    print("  'assigned' = no other event intervened, so that ripple actually "
          "carries this\n  event's label. Last column is the median gap to "
          "the NEXT event -- the window\n  this event owns in the "
          "ripple-centred view, and the confound in it.")

    if not save:
        return None
    T.to_csv(os.path.join(out_dir, "first_ripple_summary.csv"), index=False)
    with open(os.path.join(out_dir, "first_ripple.json"), "w") as f:
        json.dump({"analysis": ANALYSIS_NAME + "_first", "created": stamp,
                   "boundary": boundary, "window_s": POST_S,
                   "unit": "event x hippocampal derivation",
                   "summary": T.to_dict("records")}, f, indent=2, default=str)
    _first_figure(D, out_dir)
    print(f"\nsaved -> {out_dir}")
    return None


def _first_figure(D, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = [p for p in PHASES if p != "all"]
    with plt.rc_context(rfig._rc()):
        fig, axes = plt.subplots(1, len(phases), sharey=True, sharex=True,
                                 figsize=(4.8 * len(phases) * rfig.CM,
                                          7.0 * rfig.CM))
        axes = np.atleast_1d(axes)
        bins = np.arange(0, 2001, 50)
        for ax, ph in zip(axes, phases):
            for k in COND_KEYS:
                v = (D[D.cond.isin(PHASES[ph][k])].first_any_s.to_numpy(float)
                     * 1000.0)
                v = v[np.isfinite(v)]
                if not len(v):
                    continue
                ax.hist(v, bins=bins, density=True, histtype="step",
                        color=COL[k], lw=1.0,
                        label=LABEL[k] if ax is axes[0] else None)
            ax.set_title(ph.replace("_", " "), fontsize=rfig.FS_TITLE)
        fig.supxlabel("time from event to the NEXT ripple (ms)",
                      fontsize=rfig.FS_LABEL)
        axes[0].set_ylabel("density of events")
        axes[0].legend(frameon=False, fontsize=rfig.FS_TICK - 1,
                       loc="upper right")
        fig.suptitle("Event-centred: how long until the next ripple?\n"
                     "one value per event x hippocampal derivation, so this is "
                     "about ripple TIMING, not ripple count",
                     fontsize=rfig.FS_TICK)
        fig.tight_layout()
        fig.subplots_adjust(top=1 - 1.6 / 7.0, bottom=0.20)
        fig.savefig(os.path.join(out_dir, "first_ripple.pdf"))
        fig.savefig(os.path.join(out_dir, "first_ripple.png"), dpi=300)
        plt.close(fig)


def run(bundle=None, out_dir=None, boundary="exclusive", cache=None,
        save=True):
    root = swr_io.derivatives_dir(swr_io.get_data_root())
    G = os.path.join(root, "group", "swr")
    bundle = bundle or os.path.join(G, "bundle_v2")
    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(G, f"{ANALYSIS_NAME}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    cache = cache or os.path.join(out_dir, f"latencies_{boundary}.csv")

    D = latencies(bundle, swr_io.get_data_root(), boundary, cache)
    D["lat_ms"] = D.lat_s * 1000.0
    print(f"\nlabelled ripples: {len(D)}, {D.session.nunique()} sessions, "
          f"boundary = {boundary}, POST_S = {POST_S} s")

    # Per condition x phase, on the ripple-locked axis the event sits at
    # MINUS the latency.
    rows = []
    for ph, spec in PHASES.items():
        for k in COND_KEYS:
            v = D[D.cond.isin(spec[k])].lat_ms.to_numpy()
            if not len(v):
                continue
            rows.append({
                "phase": ph, "condition": k, "n_ripples": int(len(v)),
                "median_ms": float(np.median(v)),
                "q25_ms": float(np.percentile(v, 25)),
                "q75_ms": float(np.percentile(v, 75)),
                "q10_ms": float(np.percentile(v, 10)),
                "frac_within_500ms": float((v <= 500).mean()),
                "frac_within_250ms": float((v <= 250).mean()),
                # the panel axis runs -500..+500, so this is the share of
                # presses that are even visible on it
                "event_x_median_ms": float(-np.median(v))})
    T = pd.DataFrame(rows)
    _report(T, D)

    if not save:
        return None
    T.to_csv(os.path.join(out_dir, "event_latency_summary.csv"), index=False)
    with open(os.path.join(out_dir, "event_latency.json"), "w") as f:
        json.dump({"analysis": ANALYSIS_NAME, "created": stamp,
                   "boundary": boundary, "post_s": POST_S,
                   "rule": ("each ripple takes the label of the nearest "
                            "PRECEDING event within POST_S; latency is "
                            "positive by construction"),
                   "n_labelled_ripples": int(len(D)),
                   "summary": T.to_dict("records")}, f, indent=2)
    _figure(D, T, out_dir)
    print(f"\nsaved -> {out_dir}")
    return None


def _report(T, D):
    print(f"\n{'='*92}\nEVENT-TO-RIPPLE LATENCY (event always PRECEDES the "
          f"ripple)\n{'='*92}")
    print(f"  {'phase':<17s}{'condition':<11s}{'n':>8s}{'median':>9s}"
          f"{'IQR':>18s}{'<=250ms':>9s}{'<=500ms':>9s}")
    for _, r in T.iterrows():
        print(f"  {r.phase:<17s}{r.condition:<11s}{r.n_ripples:>8d}"
              f"{r.median_ms:>8.0f}ms"
              f"{('[' + format(r.q25_ms, '.0f') + ', ' + format(r.q75_ms, '.0f') + ']'):>18s}"
              f"{r.frac_within_250ms:>9.2f}{r.frac_within_500ms:>9.2f}")
    print("  On a ripple-locked axis the press sits at MINUS these values.")


def _figure(D, T, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = [p for p in PHASES if p != "all"]
    with plt.rc_context(rfig._rc()):
        fig, axes = plt.subplots(1, len(phases), sharey=True, sharex=True,
                                 figsize=(4.8 * len(phases) * rfig.CM,
                                          7.0 * rfig.CM))
        axes = np.atleast_1d(axes)
        bins = np.arange(-2000, 1, 40)
        for ax, ph in zip(axes, phases):
            for k in COND_KEYS:
                v = -D[D.cond.isin(PHASES[ph][k])].lat_ms.to_numpy()
                if not len(v):
                    continue
                # Density, not counts: the conditions differ in n by up to 6x
                # and the question is about SHAPE, not how many there are.
                ax.hist(v, bins=bins, density=True, histtype="step",
                        color=COL[k], lw=1.0,
                        label=LABEL[k] if ax is axes[0] else None)
            ax.axvspan(-500, 0, color="#000000", alpha=0.05, lw=0)
            ax.axvline(0, color="#3d3d3d", lw=0.8, ls=":")
            ax.set_title(ph.replace("_", " "), fontsize=rfig.FS_TITLE)
        # One shared x-label: four copies of it overlap at this panel width.
        fig.supxlabel("event time relative to ripple (ms)",
                      fontsize=rfig.FS_LABEL)
        axes[0].set_ylabel("density of assigned ripples")
        axes[0].legend(frameon=False, fontsize=rfig.FS_TICK - 1,
                       loc="upper left")
        fig.suptitle("When did the event happen, relative to its ripple?\n"
                     "shaded = the −500 to 0 ms window the trace panels show; "
                     "everything left of it is off those panels",
                     fontsize=rfig.FS_TICK)
        fig.tight_layout()
        fig.subplots_adjust(top=1 - 1.6 / 7.0, bottom=0.20)
        fig.savefig(os.path.join(out_dir, "event_latency.pdf"))
        fig.savefig(os.path.join(out_dir, "event_latency.png"), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run, "first": first})
    else:
        run()
