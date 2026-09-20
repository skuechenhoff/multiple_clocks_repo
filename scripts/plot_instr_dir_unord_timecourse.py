#!/usr/bin/env python3
"""Instruction-period timecourse panels for the direction-controlled RSA.

Three 3.5 x 3.5 cm subpanels:

  1. plan model in mPFC                    (raw betas)      -- the result
  2. memory model in bilateral Garvert MTL (demeaned)       -- descriptive
  3. memory model in left/right hippocampus (demeaned)      -- descriptive

Masks: hippocampus_left/right.nii.gz are PROBABILISTIC (0-100), so loading them
with a >0 rule would take every voxel with >=1% probability and give a "left
hippocampus" larger than bilateral MTL. The binarised hippocampus_{side}_bin50
masks (>50%) are used instead; their union reproduces hippocampus_bin exactly
(4274 + 4445 = 8719 voxels at 1 mm, 539 + 542 = 1081 in the group brain mask).

Values are the LEAVE-ONE-SUBJECT-OUT readout from per_TR_loso.py, not the peak
voxel of the group map: for each held-out subject the ROI voxels are ranked by
their maximum t over conditions on the OTHER 32 subjects, and that subject's
mean beta over the top k=100 is read at every condition. Plotting a peak voxel
selected on all subjects would be circular. p_FWE is a subject-wise sign-flip
max-t null corrected over the nine conditions, one family per (mask, model).

That LOSO family (9 conditions) is SMALLER than the small-volume family the
`*_svc_summary.json` files report (mask voxels x 9 conditions), so the two give
different p-values for the same data and must not be quoted interchangeably.

Conditions sit at their true onset midpoints in the 12 s instruction period, so
the x axis is time and the joining line is meaningful.
"""
from __future__ import annotations

import json
import os
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

PROJECT = "/Users/xpsy1114/Documents/projects/multiple_clocks"
GROUP = os.path.join(PROJECT, "data/derivatives/group")
SVC_RAW = os.path.join(GROUP, "instr_dir_unord_svc_loso_2026-09-16")
SVC_DEMEAN = os.path.join(GROUP, "instr_dir_unord_svc_loso_DEMEANED_2026-09-17")
OUT = os.path.join(GROUP, f"instr_dir_unord_timecourse_figure_{date.today().isoformat()}")
K = "100"
PANEL_CM = 3.5

EPOCHS = [
    ("instr_see-A-first",   0.0,  1.5, "A"),
    ("instr_see-B-first",   1.5,  3.0, "B"),
    ("instr_see-C-first",   3.0,  4.5, "C"),
    ("instr_see-D-first",   4.5,  6.0, "D"),
    ("instr_see-A-second",  6.0,  7.0, "A"),
    ("instr_see-B-second",  7.0,  8.0, "B"),
    ("instr_see-C-second",  8.0,  9.0, "C"),
    ("instr_see-D-second",  9.0, 10.0, "D"),
    ("instr_empty-screen", 10.0, 12.0, ""),
]
STATE_COLOURS = {"A": "#F15A29", "B": "#F7931E", "C": "#C7C6E2", "D": "#6B60AA",
                 "": "#E8E8E8"}
MPFC = "#DC673E"        # era_brewer Showgirl2 index 1
MEMORY_DARK = "#23677E"  # project HC colour
MEMORY_LIGHT = "#7eb1c4"

def memory_model(level):
    return f"{level}_REW_INSTR-{level}_instr_vs_dir_within"


def plan_model(level):
    return f"{level}_REW-{level}_unord_vs_exe_dir_across"


# One panel per model per ROI view. The plan family has no A level: there is no
# A_unord_vs_exe_dir combo, because the k=1 set model was dropped as an exact
# duplicate of A_rew_instr (with one location, order cannot matter) and is in
# any case constant across task halves.
PANELS = (
    [{"name": f"plan_mPFC_{lvl}", "svc": SVC_RAW,
      "traces": [("mPFC", plan_model(lvl), MPFC, None)]}
     for lvl in ("AB", "ABC", "ABCD")]
    + [{"name": f"memory_MTL_{lvl}_demeaned", "svc": SVC_DEMEAN,
        "traces": [("MTL", memory_model(lvl), MEMORY_DARK, None)]}
       for lvl in ("A", "AB", "ABC", "ABCD")]
    + [{"name": f"memory_HC_LR_{lvl}_demeaned", "svc": SVC_DEMEAN,
        "traces": [("HC_left", memory_model(lvl), MEMORY_DARK, "L"),
                   ("HC_right", memory_model(lvl), MEMORY_LIGHT, "R")]}
       for lvl in ("A", "AB", "ABC", "ABCD")]
    # The memory model in the SAME mask as the plan effect, raw betas, to show
    # it is a null there: once beside the plan trace, once on its own.
    + [{"name": "plan_and_memory_mPFC_ABCD", "svc": SVC_RAW,
        "traces": [("mPFC", plan_model("ABCD"), MPFC, "plan"),
                   ("mPFC", memory_model("ABCD"), MEMORY_DARK, "memory")]},
       {"name": "memory_mPFC_ABCD", "svc": SVC_RAW,
        "traces": [("mPFC", memory_model("ABCD"), MEMORY_DARK, None)]}]
)


def integer_yticks(values):
    """Whole-number ticks spanning the data, at most four of them.

    Hardcoding ticks per panel does not survive adding model levels with
    different ranges, and a panel whose ticks miss its own data reads as an
    error."""
    lo, hi = int(np.floor(min(values))), int(np.ceil(max(values)))
    step = max(1, int(np.ceil((hi - lo) / 3)))
    return list(range(lo, hi + 1, step))


def load_trace(svc_dir, mask, model):
    path = os.path.join(svc_dir, mask, f"{model}_loso_results.json")
    with open(path) as handle:
        block = json.load(handle)[K]
    order = [block["trs"].index(name) for name, *_ in EPOCHS]
    return (np.asarray(block["t"])[order], np.asarray(block["p_FWE"])[order],
            float(block["t_crit_FWE05"]), path)


def draw_panel(panel, mid):
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
    CM = 1 / 2.54
    fig, (strip, ax) = plt.subplots(
        2, 1, figsize=(PANEL_CM * CM, PANEL_CM * CM), sharex=True,
        gridspec_kw={"height_ratios": [1, 9], "hspace": .45})
    # No bbox_inches="tight" on save, so the panel is exactly PANEL_CM square.
    # At 3.5 cm a 9 pt axis label longer than ~20 characters is wider than the
    # panel itself, so the labels stay short and the readout goes in the caption.
    fig.subplots_adjust(left=.325, right=.98, top=.94, bottom=.26)

    for _n, start, stop, state in EPOCHS:
        strip.add_patch(Rectangle((start, 0), stop - start, 1,
                                  facecolor=STATE_COLOURS[state],
                                  edgecolor="white", linewidth=.5))
        if state:
            strip.text((start + stop) / 2, .45, state, ha="center", va="center",
                       fontsize=5,
                       color="white" if state in ("A", "B", "D") else "black")
    strip.set_xlim(0, 12); strip.set_ylim(0, 1); strip.axis("off")

    traces = {}
    for row, (mask, model, colour, label) in enumerate(panel["traces"]):
        t_values, p_fwe, t_crit, path = load_trace(panel["svc"], mask, model)
        traces[(mask, model)] = (t_values, p_fwe, t_crit, path)
        # one significance rail per trace, stacked so two traces never overlap
        for (_n, start, stop, _s), p in zip(EPOCHS, p_fwe):
            if p < .05:
                strip.plot([start + .1, stop - .1], [-.75 - 1.05 * row] * 2,
                           color=colour, linewidth=2.0, solid_capstyle="butt",
                           clip_on=False)

    ax.axhline(0, color="grey", linewidth=.6, zorder=1)
    for mask, model, colour, label in panel["traces"]:
        t_values = traces[(mask, model)][0]
        ax.plot(mid, t_values, "-o", color=colour, markersize=2.6,
                linewidth=1.3, zorder=3, clip_on=False)

    # Inline trace labels: there is no room for a legend at 3.5 cm. They are
    # placed at the condition where the traces are furthest apart, and offset
    # away from each other, so they land inside the axes rather than clipping
    # at an edge or colliding with the significance rails above.
    lower_anchor = None
    labelled = [((m, mod), c, l) for m, mod, c, l in panel["traces"] if l]
    if len(labelled) > 1:
        stack = np.array([traces[key][0] for key, _c, _l in labelled])
        split = int(np.argmax(stack.max(0) - stack.min(0)))
        order = np.argsort(-stack[:, split])
        for rank, index in enumerate(order):
            _key, colour, label = labelled[index]
            # Anchor to the trace's own extreme, not to its value at `split`.
            # A label sits to the RIGHT of its anchor, so anchoring at the local
            # value lets a long label run into the trace further along; anchoring
            # at the series max (top label) or min (bottom label) cannot.
            top = rank == 0
            # The label sits to the RIGHT of its anchor and spans roughly the
            # left half of the axis, so it is anchored on the extreme of THAT
            # segment only. Anchoring on the whole trace's extreme drags the
            # lower label down to a minimum that may occur far right, pushing it
            # onto the x axis; anchoring on the local value lets the trace catch
            # up with it further along. Left half, own extreme, is free of both.
            span = stack[index][:len(mid) // 2 + 1]
            if not top:
                lower_anchor = float(span.min())
            ax.annotate(label,
                        (mid[split] if top else mid[0],
                         stack[index].max() if top else span.min()),
                        textcoords="offset points", xytext=(3.0, 3.0 if top else -10.0),
                        fontsize=8, color=colour, fontweight="bold")

    ax.margins(y=.18 if len(panel["traces"]) > 1 else .12)
    if lower_anchor is not None:
        # The bottom label is offset in POINTS, so whether it clears the axis
        # depends on the data range -- a deep anchor put it on the spine. Reserve
        # the space explicitly instead of relying on the margin.
        low, high = ax.get_ylim()
        needed = lower_anchor - .30 * (high - low)
        if needed < low:
            ax.set_ylim(needed, high)
    ax.set_xlim(0, 12)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_xlabel("time (s)", fontsize=9, labelpad=1.0)
    ax.set_ylabel("group $t$", fontsize=9, labelpad=1.0)
    ax.set_yticks(integer_yticks(np.concatenate(
        [traces[(m, mod)][0] for m, mod, _c, _l in panel["traces"]] + [[0.0]])))
    ax.tick_params(labelsize=8, length=2, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)

    stem = os.path.join(OUT, f"instr_dir_unord_{panel['name']}_3p5cm")
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=600)
    plt.close(fig)
    return stem, traces


def main():
    os.makedirs(OUT, exist_ok=True)
    mid = np.array([(a + b) / 2 for _, a, b, _ in EPOCHS])
    stats = {"date": date.today().isoformat(), "panel_cm": PANEL_CM,
             "readout": f"leave-one-subject-out, top k={K} voxels per mask",
             "correction": ("subject-wise sign-flip max-t over the nine "
                            "conditions, one family per (mask, model); "
                            "10000 permutations, seed 0"),
             "conditions": [e[0] for e in EPOCHS], "panels": {}}
    for panel in PANELS:
        stem, traces = draw_panel(panel, mid)
        stats["panels"][panel["name"]] = {
            "file": stem + ".pdf",
            "betas": "demeaned" if panel["svc"] is SVC_DEMEAN else "raw",
            "traces": {f"{mask}|{model}": {
                          "mask": mask, "model": model,
                          "t": traces[(mask, model)][0].round(4).tolist(),
                          "p_FWE": traces[(mask, model)][1].round(4).tolist(),
                          "t_crit_FWE05": round(traces[(mask, model)][2], 4),
                          "source_file": traces[(mask, model)][3]}
                       for mask, model, _c, _l in panel["traces"]}}
        print("wrote", os.path.basename(stem) + ".pdf")
    with open(os.path.join(OUT, "figure_stats.json"), "w") as handle:
        json.dump(stats, handle, indent=2)


if __name__ == "__main__":
    main()
