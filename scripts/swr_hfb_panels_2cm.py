#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PUBLICATION panels at 2 x 2 cm: the medial frontal ripple-locked HFB trace, and
the condition split under two different definitions of "early".

Five panels, each its own file, each exactly 2 x 2 cm, x from -500 to +500 ms:

  medial_frontal      medial frontal, ripple-locked HFB
  first_traversal     correct / incorrect uncovers / navigation, FIRST traversal
  after_first         the same three, every repeat after the first traversal
  encoding            the same three, everything up to but EXCLUDING the first
                      fully correct ABCD loop
  after_encoding      the same three, from that loop onward

THE TWO SPLITS ARE NOT THE SAME CUT
-----------------------------------
`first_traversal` / `after_first` cut after repeat 1. `encoding` /
`after_encoding` cut at the first fully correct loop. The second pair needs the
EXCLUSIVE phase boundary -- the first fully correct repeat belongs on the LATE
side -- which is not the boundary the project's other figures use: `ripples.
stage_of_repeat` puts that repeat in `while learning` because it is where the
route is first demonstrated. So `--cond_dir_excl` must point at a
`swr_ripple_hfb_conditions.py run --boundary=exclusive` run. Passing the
inclusive run would silently shift 39% of the learning-side repeats.

WHAT IS PLOTTED
---------------
Per unit: real minus shifted-null, flank-normalised to the non-peri band
(250-750 ms), Gaussian-smoothed. Conditions spanning several stages are pooled
by ripple count, which equals the mean over their union because each saved
trace is already a per-ripple mean. All of that is imported from
`swr_hfb_locking_contrasts`, not reimplemented, so these panels and the
published contrast figures are the same numbers.

SIZE AND TYPE
-------------
The figure is 2 x 2 cm INCLUDING its axis labels, because that is the footprint
the panel occupies on the page. At that size CLAUDE.md's 9 pt minimum leaves
roughly 1 cm of axes, so both are written: `_7pt` (the default, legible at 2 cm)
and `_9pt` (the house minimum, which needs the panel placed at ~2.6 cm to look
the same). Vector .pdf plus 600 dpi .png. Legends do not fit in 2 cm and are
written once to `legend_conditions.pdf`; a `_labelled` variant of every panel
carries titles and a legend for checking what is what.

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_hfb_panels_2cm.py run \\
        --cond_dir_excl=<ripple_hfb_conditions_exclusive_...>

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swr_hfb_locking_contrasts import (NONPERI_S, MEDIAL, COL, LABEL,
                                       SMOOTH_MS, ripple_duration, N_PERM,
                                       SEED)
from swr_hfb_locking_contrasts import pooled_curves_by as _pooled
import mc.analyse.swr_sakon as sk
from swr_ripple_locked_hfb import _smooth

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

CM = 1 / 2.54
PANEL_CM = 2.0
XLIM_MS = (-500.0, 500.0)
XTICKS = (-500, 0, 500)
COND_KEYS = ("reward", "error", "move")

# Which saved conditions make up each side of each split. The condition names
# are stage-suffixed, so "after the first traversal" is plan + execute pooled,
# and "encoding" is explore + plan pooled -- but only under the EXCLUSIVE
# boundary does `plan` stop before the first fully correct loop.
#
# ALL FOUR come from the EXCLUSIVE run. `explore` is bit-identical across the
# two boundaries, so `first_traversal` could be read from either -- but
# `after_first` cannot. It looks as if it should: the boundary only moves one
# repeat per grid, from `plan` to `execute`, and pooling the two back together
# should recover the same union either way. It does not, for two reasons the
# invariance check below measures rather than assumes:
#
#   * `pooled_cond_curves` weights each derivation's trace by a SESSION-level
#     ripple count, not by the ripples that derivation actually contributed to
#     that condition. Moving ripples between plan and execute changes the
#     per-derivation composition without changing the session weight, so the
#     pooled mean shifts (mean |diff| 5e-4 against a ~1.6e-2 signal, but up to
#     1.3e-2 in one session);
#   * MIN_RIPPLES is applied per (derivation, condition), so a few error
#     derivations clear the threshold under one boundary and not the other
#     (error_plan 124 vs 126, error_execute 83 vs 80).
#
# Neither is a reason to prefer one boundary; both are a reason not to MIX
# them across panels of one figure. One run, all four panels.
SPLITS = {
    "first_traversal": dict(
        source="excl", title="first traversal",
        conds={"reward": ("reward_explore",), "error": ("error_explore",),
               "move": ("move_explore",)}),
    "after_first": dict(
        source="excl", title="later repeats",
        conds={"reward": ("reward_plan", "reward_execute"),
               "error": ("error_plan", "error_execute"),
               "move": ("move_plan", "move_execute")}),
    "encoding": dict(
        source="excl", title="encoding",
        conds={"reward": ("reward_explore", "reward_plan"),
               "error": ("error_explore", "error_plan"),
               "move": ("move_explore", "move_plan")}),
    "after_encoding": dict(
        source="excl", title="after encoding",
        conds={"reward": ("reward_execute",), "error": ("error_execute",),
               "move": ("move_execute",)}),
}


def _rc(font_pt):
    return {"font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": font_pt,
            "axes.titlesize": font_pt, "axes.labelsize": font_pt,
            "xtick.labelsize": font_pt, "ytick.labelsize": font_pt,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6, "ytick.major.width": 0.6,
            "xtick.major.size": 1.8, "ytick.major.size": 1.8,
            "xtick.major.pad": 1.2, "ytick.major.pad": 1.2,
            "axes.spines.top": False, "axes.spines.right": False,
            "savefig.bbox": None}


def _region_curves(region_dir, unit, roi=MEDIAL, smooth_ms=SMOOTH_MS,
                   part=None):
    """Per-unit region curve, built exactly as the region figure does.

    `part` maps (session, pair) to a partition ROI. The saved index carries the
    run's own `roi_family` plus the derived MedialFrontal label, so without the
    map only those names can be asked for -- lOFC does not exist there.
    """
    z = np.load(os.path.join(region_dir, "timecourses.npz"))
    t_ms, tr = z["t_ms"], z["traces"]
    fs = 1000.0 / (t_ms[1] - t_ms[0])
    flank = (np.abs(t_ms) >= NONPERI_S[0] * 1000) & \
            (np.abs(t_ms) < NONPERI_S[1] * 1000)
    ix = pd.read_csv(os.path.join(region_dir, "timecourse_index.csv"))
    ix["same_shaft"] = ix.same_shaft.astype(bool)
    ix["is_real"] = ix.is_real.astype(bool)
    if "trace_row" not in ix.columns:
        ix["trace_row"] = np.arange(len(ix))
    if part is not None and roi != MEDIAL:
        ix = ix[ix.roi.isin(("mPFC", "mOFC", "TemporalLateral", "Auditory",
                             "Visual"))].copy()
        ix["roi"] = pd.MultiIndex.from_arrays(
            [ix.session, ix.cx_pair]).map(part)
    sel = ix[(ix.roi == roi) & (~ix.same_shaft)]
    if not len(sel):
        raise SystemExit(f"no different-shaft derivations for roi '{roi}'")
    out = []
    for _, g in sel.groupby(unit):
        rr = g.trace_row.to_numpy()
        r, n = tr[rr[g.is_real.to_numpy()]], tr[rr[~g.is_real.to_numpy()]]
        if not len(r) or not len(n):
            continue
        c = r.mean(0) - n.mean(0)
        out.append(c - c[flank].mean())
    A = _smooth(np.stack(out), fs, smooth_ms)
    return t_ms, A


def _cond_block(cond_dir, unit, smooth_ms=SMOOTH_MS, part=None):
    """Everything the pooler needs from one conditions run.

    The family rows are relabelled to partition ROIs and the MedialFrontal
    rows kept as they are, so both `MedialFrontal` and `lOFC` can be asked for
    from the same index.
    """
    z = np.load(os.path.join(cond_dir, "timecourses.npz"))
    ix = pd.read_csv(os.path.join(cond_dir, "timecourse_index.csv"))
    ix["is_real"] = ix.is_real.astype(bool)
    if "trace_row" not in ix.columns:
        ix["trace_row"] = np.arange(len(ix))
    if part is not None:
        fam = ix[ix.roi.isin(("mPFC", "mOFC"))].copy()
        fam["roi"] = pd.MultiIndex.from_arrays(
            [fam.session, fam.cx_pair]).map(part)
        ix = pd.concat([fam, ix[ix.roi == MEDIAL]],
                       ignore_index=True).dropna(subset=["roi"])
    counts = pd.read_csv(os.path.join(cond_dir, "ripple_counts.csv"))
    return dict(t_ms=z["t_ms"], tr=z["traces"], ix=ix, counts=counts,
                unit=unit, smooth_ms=smooth_ms)


def _mean_sem(curves_by_unit, t_ms=None, clusters=True):
    """mean, SEM, n, and the surviving clusters of the curve against zero.

    Same test as every other panel in this project: `swr_sakon.
    cluster_perm_time`, sign-flip null, same n_perm and seed. The curve is
    already real-minus-shifted-null, so zero is the right reference.
    """
    A = np.stack(list(curves_by_unit.values()))
    cl = []
    if clusters and t_ms is not None and len(A) >= 5:
        _, cc, pv, _ = sk.cluster_perm_time(A, n_perm=N_PERM, seed=SEED)
        cl = [{"start_ms": float(t_ms[a]), "stop_ms": float(t_ms[b - 1]),
               "p": float(p)} for (a, b), p in zip(cc, pv) if p < 0.05]
    return A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A)), len(A), cl


def _panel(ax, t_ms, series, lw, font_pt, ylabel=True, xlabel=True,
           title=None, legend=False, dur=None, press=None):
    """One 2 cm panel. `series` is {key: (mean, sem, n, clusters)}.

    Under the traces, in order: WHEN THE TRIGGERING EVENT HAPPENED (median and
    IQR of the event-to-ripple latency, one row per condition), the mean ripple
    extent (peak +- duration/2, with its SE as end caps), then one bar per
    surviving cluster. All below the data because a 2 cm panel has no room
    above it -- the y-axis is extended downward to make that room rather than
    letting the annotation sit on the traces.

    The event row is not decoration. Every condition is defined by an event
    that PRECEDES its ripple, so a pre-ripple HFB bump could be the response to
    that event rather than to the ripple. Drawing the press distribution on the
    same axis is what lets a reader tell the two apart. A median that falls
    off the left edge is drawn as a caret there, with the IQR clipped -- not
    silently omitted, which would make the press look closer than it is.
    """
    keep = (t_ms >= XLIM_MS[0]) & (t_ms <= XLIM_MS[1])
    t = t_ms[keep]
    ax.axhline(0, color="0.55", lw=0.5, zorder=0)
    ax.axvline(0, color="0.35", lw=0.5, ls=":", zorder=0)
    for k, v in series.items():
        m, se, n = v[0], v[1], v[2]
        ax.fill_between(t, (m - se)[keep], (m + se)[keep], color=COL[k],
                        alpha=0.20, lw=0, zorder=1)
        ax.plot(t, m[keep], color=COL[k], lw=lw, zorder=2,
                label=f"{LABEL.get(k, k)} ({n})" if legend else None)

    lo, hi = ax.get_ylim()
    span = hi - lo
    # Only conditions that HAVE a cluster get a row. Reserving a row for every
    # condition regardless cost 40% of a 2 cm panel to draw nothing.
    with_cl = [k for k, v in series.items() if len(v) > 3 and v[3]]
    pr_keys = [k for k in series if press and k in press]
    n_rows = len(with_cl)
    n_extra = (1 if dur else 0) + len(pr_keys)
    ax.set_ylim(lo - span * (0.05 + 0.07 * (n_rows + n_extra)), hi)
    lo2 = ax.get_ylim()[0]
    step = span * 0.07
    for i, k in enumerate(pr_keys):
        q = press[k]
        y = lo2 + step * (n_rows + (1 if dur else 0) + i + 0.45)
        a = max(-q["q75_ms"], XLIM_MS[0])      # IQR on the ripple-locked axis
        b = min(-q["q25_ms"], XLIM_MS[1])
        if b > a:
            ax.plot([a, b], [y, y], color=COL[k], lw=lw * 0.8, alpha=0.75,
                    solid_capstyle="butt", zorder=4)
        med = -q["median_ms"]
        if med >= XLIM_MS[0]:
            ax.plot([med], [y], marker="|", color=COL[k], ms=3.4,
                    mew=lw * 1.1, zorder=5)
        else:
            ax.plot([XLIM_MS[0]], [y], marker="<", color=COL[k], ms=2.4,
                    mew=0, zorder=5)
    if dur:
        y = lo2 + step * (n_rows + 0.45)
        h, sehw = dur["mean_ms"] / 2.0, dur["sem_ms"] / 2.0
        ax.plot([-h, h], [y, y], color="0.15", lw=lw * 1.9,
                solid_capstyle="butt", zorder=5)
        ax.plot([-h - sehw, -h + sehw], [y, y], color="0.15", lw=lw * 0.7,
                zorder=5)
        ax.plot([h - sehw, h + sehw], [y, y], color="0.15", lw=lw * 0.7,
                zorder=5)
    for i, k in enumerate(with_cl):
        for c in series[k][3]:
            ax.plot([c["start_ms"], c["stop_ms"]],
                    [lo2 + step * (n_rows - 1 - i + 0.35)] * 2,
                    color=COL[k], lw=lw * 1.9, solid_capstyle="butt", zorder=5)
    ax.set_xlim(*XLIM_MS)
    ax.set_xticks(list(XTICKS))
    # Two decimals would need four characters per tick and there is no room;
    # the values are ~1e-2, so the axis is written in units of 0.01.
    ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune="both"))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
    if xlabel:
        ax.set_xlabel("ms", labelpad=0.8)
    if ylabel:
        ax.set_ylabel("HFB (z × 100)", labelpad=1.0)
    if title:
        ax.set_title(title, pad=1.5)
    if legend:
        ax.legend(fontsize=font_pt - 1, frameon=False, loc="upper left",
                  handlelength=1.0, borderpad=0.05, labelspacing=0.15,
                  borderaxespad=0.1)


# Margins around the 2 cm axes box, in cm, at the default font. A rotated
# one-line y-label plus its ticks needs about 1 cm; two lines would need half
# the panel width again, which is why the label is written on one line.
# `right` is not decoration: the last x tick sits ON the axis end, so half
# of "500" hangs outside it and is clipped with a hairline margin.
MARGIN_CM = dict(left=1.00, bottom=0.80, right=0.38, top=0.12)


def _make_fig(mode, font_pt, labelled=False):
    """A figure whose AXES BOX is 2 x 2 cm ('axes'), or whose whole canvas is
    ('total').

    Two conventions, because "a 2 cm panel" means different things when you are
    placing it: `axes` keeps the data area at 2 cm and lets the labels sit
    outside it, which is what makes a row of panels line up; `total` keeps the
    whole footprint at 2 cm, which is what fits a fixed slot. At 7 pt the
    labels cost about 1 cm, so in `total` mode the data area is only ~1.1 cm --
    real, but small.
    """
    scale = 2.6 if labelled else 1.0        # proofs get room for title+legend
    box = PANEL_CM * scale
    if mode == "axes":
        m = {k: v * (1.0 if not labelled else 1.15)
             for k, v in MARGIN_CM.items()}
        W = box + m["left"] + m["right"]
        H = box + m["bottom"] + m["top"] + (0.45 if labelled else 0.0)
        fig = plt.figure(figsize=(W * CM, H * CM))
        ax = fig.add_axes([m["left"] / W, m["bottom"] / H, box / W, box / H])
    else:
        W = H = box
        m = MARGIN_CM
        fig = plt.figure(figsize=(W * CM, H * CM))
        ax = fig.add_axes([m["left"] / W, m["bottom"] / H,
                           (W - m["left"] - m["right"]) / W,
                           (H - m["bottom"] - m["top"]) / H])
    return fig, ax


def _save(fig, stem):
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=600)


def run(region_dir=None, cond_dir=None, cond_dir_excl=None, out_dir=None,
        unit="session", smooth_ms=SMOOTH_MS, save=True,
        rois=(MEDIAL, "lOFC"), partition_dir=None, bundle_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    global plt
    import matplotlib.pyplot as plt

    root = swr_io.derivatives_dir(swr_io.get_data_root())
    G = os.path.join(root, "group", "swr")
    region_dir = region_dir or os.path.join(G, "ripple_locked_hfb_2026-09-18")
    cond_dir = cond_dir or os.path.join(G, "ripple_hfb_conditions_2026-09-21")
    if cond_dir_excl is None:
        raise SystemExit(
            "--cond_dir_excl is required: the encoding split needs the "
            "EXCLUSIVE phase boundary (the first fully correct repeat on the "
            "late side). Run:\n  python scripts/swr_ripple_hfb_conditions.py "
            "run --boundary=exclusive --out_dir=<dir>")

    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(G, f"hfb_panels_2cm_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    partition_dir = partition_dir or sorted(
        os.path.join(G, x) for x in os.listdir(G)
        if x.startswith("hfb_roi_partition_"))[-1]
    part = pd.read_csv(os.path.join(partition_dir,
                                    "derivation_assignment.csv")) \
        .set_index(["session", "cx_pair"])["roi_part"]
    bundle_dir = bundle_dir or os.path.join(G, "bundle_v2")
    dur = ripple_duration(bundle_dir)
    # Event-to-ripple latency, if `swr_hfb_event_latency.py` has been run. The
    # panels are readable without it; they are INTERPRETABLE with it.
    lat_dirs = sorted(os.path.join(G, x) for x in os.listdir(G)
                      if x.startswith("hfb_event_latency_"))
    press = {}
    if lat_dirs:
        L = pd.read_csv(os.path.join(lat_dirs[-1],
                                     "event_latency_summary.csv"))
        press = {ph: {r.condition: {"median_ms": float(r.median_ms),
                                    "q25_ms": float(r.q25_ms),
                                    "q75_ms": float(r.q75_ms)}
                      for _, r in g.iterrows()}
                 for ph, g in L.groupby("phase")}
        print(f"event latencies from {os.path.basename(lat_dirs[-1])}")
    else:
        print("no hfb_event_latency_* run found -- press markers omitted")
    print(f"ripple duration {dur['mean_ms']:.1f} ± {dur['sem_ms']:.1f} ms "
          f"({dur['n_ripples']} ripples, {dur['n_sessions']} sessions)")

    blocks = {"incl": _cond_block(cond_dir, unit, smooth_ms, part),
              "excl": _cond_block(cond_dir_excl, unit, smooth_ms, part)}

    # roi -> {"region": (t, series), "<split>": (t, series, spec)}
    by_roi, report = {}, {}
    for roi in rois:
        t_reg, A_reg = _region_curves(region_dir, unit, roi=roi,
                                      smooth_ms=smooth_ms, part=part)
        m, se, n, cl = _mean_sem({i: a for i, a in enumerate(A_reg)}, t_reg)
        print(f"\n{roi}: region trace {n} {unit}s, "
              f"{len(cl)} cluster(s) p<0.05")
        for c in cl:
            print(f"    {c['start_ms']:+.0f} to {c['stop_ms']:+.0f} ms, "
                  f"p = {c['p']:.4f}")
        panels = {"region": (t_reg, {"reward": (m, se, n, cl)}, None)}
        rep = {"region": {"n_units": n, "clusters": cl}}
        for name, spec in SPLITS.items():
            b = blocks[spec["source"]]
            series = {}
            for k in COND_KEYS:
                cur = _pooled(b["t_ms"], b["tr"], b["ix"][b["ix"].roi == roi],
                              b["counts"], spec["conds"][k], b["unit"])
                if len(cur) < 2:
                    print(f"  [{roi}/{name}/{k}: {len(cur)} {unit}s -- skipped]")
                    continue
                if b["smooth_ms"]:
                    keys = list(cur)
                    A = _smooth(np.stack([cur[x] for x in keys]),
                                1000.0 / (b["t_ms"][1] - b["t_ms"][0]),
                                b["smooth_ms"])
                    cur = {x: A[i] for i, x in enumerate(keys)}
                series[k] = _mean_sem(cur, b["t_ms"])
            panels[name] = (b["t_ms"], series, spec)
            rep[name] = {"boundary": spec["source"], "title": spec["title"],
                         "conditions": spec["conds"],
                         "n_units": {k: int(v[2]) for k, v in series.items()},
                         "clusters": {k: v[3] for k, v in series.items()}}
            sig = [f"{k}:{len(v[3])}" for k, v in series.items() if v[3]]
            print(f"  {name:<16s} "
                  + ", ".join(f"{k} n={v[2]}" for k, v in series.items())
                  + (f"   clusters {', '.join(sig)}" if sig else ""))
        by_roi[roi] = panels
        report[roi] = rep

    # How much the two boundary runs disagree on the panels that describe the
    # SAME repeats. `first_traversal` must be exactly 0 -- `explore` cannot
    # move. `after_first` is not 0, for the pooling reasons in the SPLITS note;
    # it is measured here so the size of that approximation is on record next
    # to the figure instead of being assumed away.
    check = {}
    for name in ("first_traversal", "after_first"):
        spec = SPLITS[name]
        d = []
        for k in COND_KEYS:
            B1, B2 = blocks["incl"], blocks["excl"]
            a = _pooled(B1["t_ms"], B1["tr"], B1["ix"][B1["ix"].roi == MEDIAL],
                        B1["counts"], spec["conds"][k], unit)
            b = _pooled(B2["t_ms"], B2["tr"], B2["ix"][B2["ix"].roi == MEDIAL],
                        B2["counts"], spec["conds"][k], unit)
            sh = sorted(set(a) & set(b))
            if sh:
                d.append(float(np.abs(np.stack([a[u] for u in sh])
                                      - np.stack([b[u] for u in sh])).max()))
        check[name] = max(d) if d else float("nan")
    print("\nboundary sensitivity (max |inclusive - exclusive| over units;"
          "\n  all four panels are drawn from the EXCLUSIVE run):")
    for k, v in check.items():
        note = ("  identical, as it must be"
                if (np.isfinite(v) and v < 1e-9)
                else "  pooling approximation, see SPLITS note")
        print(f"  {k:<18s}{v:.3e}{note}")

    if not save:
        return None

    font_pt, lw = 7, 1.1
    with plt.rc_context(_rc(font_pt)):
        for mode in ("axes", "total"):
            for roi, panels in by_roi.items():
                sub = os.path.join(out_dir, f"{mode}_2cm", roi)
                os.makedirs(sub, exist_ok=True)
                for labelled in (False, True):
                    suf = "_labelled" if labelled else ""
                    for name, (t_ms, series, spec) in panels.items():
                        fig, ax = _make_fig(mode, font_pt, labelled)
                        n0 = list(series.values())[0][2]
                        ttl = (f"{roi} ({n0} {unit}s)" if spec is None
                               else f"{roi} — {spec['title']}")
                        _panel(ax, t_ms, series, lw, font_pt, dur=dur,
                               press=(press.get(name) if spec is not None
                                      else None),
                               title=ttl if labelled else None,
                               legend=labelled and spec is not None)
                        _save(fig, os.path.join(sub, f"{name}{suf}"))
                        plt.close(fig)

        # The legend the 2 cm panels cannot carry, once, to be placed by hand.
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots(figsize=(4.4 * CM, 1.5 * CM))
        ax.set_axis_off()
        ax.legend(handles=[Line2D([], [], color=COL[k], lw=lw, label=LABEL[k])
                           for k in COND_KEYS],
                  frameon=False, loc="center", handlelength=1.4,
                  labelspacing=0.3, fontsize=font_pt)
        fig.savefig(os.path.join(out_dir, "legend_conditions.pdf"),
                    bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, "legend_conditions.png"), dpi=600,
                    bbox_inches="tight")
        plt.close(fig)

    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"analysis": "hfb_panels_2cm", "created": stamp,
                   "region_dir": region_dir,
                   "cond_dir_inclusive": cond_dir,
                   "cond_dir_exclusive": cond_dir_excl,
                   "unit": unit, "smooth_ms": smooth_ms,
                   "panel_cm": PANEL_CM, "xlim_ms": list(XLIM_MS),
                   "font_pt": font_pt, "line_width_pt": lw,
                   "sizes": {"axes_2cm": "the AXES BOX is 2x2 cm, labels "
                                         "outside it",
                             "total_2cm": "the whole canvas is 2x2 cm, data "
                                          "area ~1.1x1.1 cm"},
                   "rois": list(rois), "different_shaft_only": True,
                   "y_units": "HFB (z), plotted x10^-2",
                   "ripple_duration": dur,
                   "event_latency": press,
                   "event_latency_note": ("median and IQR of the event-to-"
                                          "ripple latency, drawn on the "
                                          "ripple-locked axis at MINUS the "
                                          "latency; a caret at the left edge "
                                          "means the median is off-panel"),
                   "cluster_test": {"function": "swr_sakon.cluster_perm_time",
                                    "n_perm": N_PERM, "seed": SEED,
                                    "against": "zero (curve is already "
                                               "real minus shifted null)"},
                   "panels": report,
                   "boundary_invariance_max_abs_diff": check}, f, indent=2)
    print(f"\nsaved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
