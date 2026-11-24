#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One overview of every existing result that bears on SK's gradient hypothesis:

  1  the ventral end (mOFC/OFC) is ripple-coupled during encoding, mPFC during
     execution
  2  'now' neurons fire when the subject identifies a correct location
  3  in execution mPFC plays the immediate future, independent of ripples
  4  lOFC encodes reward events, before / independent of ripples
  5  the representation changes from exploration to execution

Nothing is recomputed: the panels are the saved figures, and the one new panel
(lOFC uncover response) is drawn from the saved T1b curves with the same
baseline function the T1b statistics used.

    conda activate env_multiple_clocks
    python scripts/exploratory/plot_gradient_hypothesis_overview.py

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "exploratory"))
import mc.analyse.swr_io as swr_io

CM = 1 / 2.54
SWR = os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()), "group", "swr")
GROUP = os.path.dirname(SWR)
PVS = os.path.join(SWR, "plan_vs_store_2026-10-29")

CLAIMS = [
    ("1  Ventral end (mOFC/OFC) ripple-coupled in encoding, mPFC in execution",
     "PARTLY. First traversal: mOFC HFB rises −220…+50 ms (p = .001), mPFC "
     "none (p = .32); execution: mPFC rises −110…+300 (p < .001). But the "
     "mOFC rise sits BEFORE the ripple, and on the same ripple mOFC-before does "
     "not predict mPFC-after (p = .42). mOFC cells: spatial code inverts "
     "inside encoding ripples (−0.54, p = .028).",
     [os.path.join(SWR, "hfb_phases_50shifts_2026-10-08", "mOFC_vs_mPFC",
                   "first_traversal", "sliding_panel_a.png"),
      os.path.join(SWR, "hfb_mofc_mpfc_phases_2026-10-08", "execution",
                   "sliding_panel_a.png"),
      os.path.join(SWR, "hfb_ripple_coupling_chain_2026-10-07",
                   "chain_ripple_locked.png"),
      os.path.join(SWR, "ripple_content_figures_2026-10-07",
                   "fig39_ripple_templates.jpeg")]),
    ("2  'Now' neurons fire when a correct location is identified",
     "PROMISING. mPFC holds the newly found reward square (+0.77, p = .0006), "
     "not repeats or errors, but less so with a ripple (−0.51, p = .08). Lag-0 "
     "cells lean towards firing more at storage events near a ripple (r = "
     "+0.15, p = .065, cells). ⚠ Both lock to ARRIVAL, not reward; the ripple "
     "split misses post-reward ripples. Cell lag profiles peak at 30° "
     "(ventral) / 60° (dorsal); 2c: yellow = ventral, orange = dorsal, grey = "
     "outside the fMRI gradient.",
     [os.path.join(SWR, "ripple_content_figures_2026-10-01",
                   "fig29_consolidation.jpeg"),
      os.path.join(SWR, "ripple_content_figures_2026-10-01",
                   "fig34_ripple_dependent.jpeg"),
      os.path.join(GROUP, "cell_gradient_master", "2026-08-28_15-19-35",
                   "final_splits_2026-10-06", "pc1_ventral_dorsal_profiles.png")]),
    ("3  In execution, mPFC plays the immediate future, independent of ripples",
     "CONSISTENT, NOT TESTED DIRECTLY. Inside ripples cells fire LESS at their "
     "preferred square (mPFC −0.08 to −0.13, p ≤ .005); ripples do not shift "
     "mPFC towards the future (future_shift; mPFC fails the gate). The forward "
     "content in execution is in HC ripples (T4, row 5).",
     [os.path.join(SWR, "ripple_content_figures_2026-10-01",
                   "fig32_preferred_lag.jpeg"),
      os.path.join(SWR, "ripple_content_figures_2026-10-07",
                   "fig38_future_shift.jpeg")]),
    ("4  lOFC encodes reward events, before / independent of ripples",
     "CONSISTENT. lOFC responds to correct uncovers in the first traversal "
     "(+0.029 z, p = .006), not in execution (p = .24); it leads the ripple "
     "(direction −0.007, p = .009). Open: whether the lead is just the uncover "
     "response (ripples follow presses by ~0.4 s).",
     ["LOFC_PANEL",
      os.path.join(SWR, "hfb_direction_sliding_lOFC_2026-10-08",
                   "sliding_panel_a.png"),
      os.path.join(SWR, "hfb_ripple_coupling_chain_lOFC_2026-10-07",
                   "chain_ripple_locked.png")]),
    ("5  The representation changes from exploration to execution",
     "IN HC. Post-correct HC ripples carry the path just walked in the first "
     "traversal (PAST − FUT +0.44, p = .043) and the path ahead in execution "
     "(FUT +0.35, p = .006; flip post hoc p = .0003). mPFC: lag tags predict "
     "WHERE a cell fires in execution only; nothing in mPFC ripples.",
     [os.path.join(PVS, "T4_path_content", "T4_path_content.png"),
      os.path.join(SWR, "ripple_content_figures_2026-10-01",
                   "fig33_where_not_when.jpeg")]),
]


def lofc_panel(path):
    """lOFC uncover-locked HFB, first traversal vs execution, from T1b."""
    from swr_plan_vs_store import _base_smooth
    z = np.load(os.path.join(PVS, "T1b_hfb_state", "curves.npz"))
    t = z["t_event_ms"]
    with open(os.path.join(PVS, "T1b_hfb_state", "result.json")) as f:
        st = json.load(f)["stats"]
    fig, ax = plt.subplots(figsize=(9 * CM, 6 * CM))
    for key, lab, col in (("FT correct", "correct, first traversal", "#0e3d3a"),
                          ("exec correct", "correct, execution", "#9ec6b8")):
        X = np.stack([_base_smooth(c, t) for c in z[f"event|{key}|lOFC"]])
        m, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))
        r = st[f"event | lOFC | {key}"]
        ax.plot(t, m, color=col, lw=1.4,
                label=f"{lab} (n = {len(X)}): {r['mean']:+.3f} z, "
                      f"p = {r['p_perm']:.3f}")
        ax.fill_between(t, m - se, m + se, color=col, alpha=0.25, lw=0)
    ax.axvspan(-1500, -1000, color="0.92", lw=0)
    ax.axvspan(0, 500, color="0.96", lw=0)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("time from correct uncover (ms)")
    ax.set_ylabel("lOFC HFB (z)")
    ax.set_title("lOFC, uncover-locked (T1b)", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.46))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def claim_pages(rows, pdf_path):
    """One A4-landscape page per claim, at most two figures per row, so that
    every panel stays legible."""
    from matplotlib.backends.backend_pdf import PdfPages
    W = 29.7 * CM
    with PdfPages(pdf_path) as pdf:
        for i, (title, verdict, paths) in enumerate(rows):
            imgs = [plt.imread(p) for p in paths]
            pairs = [imgs[k:k + 2] for k in range(0, len(imgs), 2)]
            # each row's height follows from fitting its images side by side
            row_h = [min(W / sum(im.shape[1] / im.shape[0] for im in pr),
                         9.0 * CM) for pr in pairs]
            head = 4.0 * CM
            fig = plt.figure(figsize=(W, head + sum(row_h)))
            g = gridspec.GridSpec(1 + len(pairs), 1, figure=fig, hspace=0.04,
                                  height_ratios=[head] + row_h)
            ax = fig.add_subplot(g[0])
            ax.axis("off")
            ax.text(0, 0.95, title, fontsize=14, fontweight="bold", va="top")
            ax.text(0, 0.5, verdict, fontsize=9.5, va="top", wrap=True,
                    color="0.2")
            k = 0
            for r, pr in enumerate(pairs):
                inner = gridspec.GridSpecFromSubplotSpec(
                    1, len(pr), subplot_spec=g[1 + r], wspace=0.02,
                    width_ratios=[im.shape[1] / im.shape[0] for im in pr])
                for j, im in enumerate(pr):
                    a = fig.add_subplot(inner[j])
                    a.imshow(im)
                    a.axis("off")
                    a.text(0, 1.0, f"{i + 1}{'abcd'[k]}", transform=a.transAxes,
                           fontsize=12, fontweight="bold", va="bottom")
                    k += 1
            pdf.savefig(fig, dpi=200)
            plt.close(fig)


def main():
    out = os.path.join(SWR, f"gradient_hypothesis_overview_{datetime.date.today()}")
    os.makedirs(out, exist_ok=True)
    lofc = os.path.join(out, "lofc_uncover_panel.png")
    lofc_panel(lofc)
    rows = [(t, v, [lofc if p == "LOFC_PANEL" else p for p in ps])
            for t, v, ps in CLAIMS]
    W = 48 * CM
    row_h = 11 * CM
    head_h = 2.2 * CM
    fig = plt.figure(figsize=(W, len(rows) * (row_h + head_h)))
    outer = gridspec.GridSpec(2 * len(rows), 1, figure=fig, hspace=0.05,
                              height_ratios=[head_h, row_h] * len(rows))
    sources = []
    for i, (title, verdict, paths) in enumerate(rows):
        ax = fig.add_subplot(outer[2 * i])
        ax.axis("off")
        ax.text(0, 0.95, title, fontsize=15, fontweight="bold", va="top")
        ax.text(0, 0.42, verdict, fontsize=10.5, va="top", wrap=True,
                color="0.2")
        imgs = [plt.imread(p) for p in paths]
        ratios = [im.shape[1] / im.shape[0] for im in imgs]
        inner = gridspec.GridSpecFromSubplotSpec(
            1, len(imgs), subplot_spec=outer[2 * i + 1], width_ratios=ratios,
            wspace=0.02)
        for j, (im, p) in enumerate(zip(imgs, paths)):
            a = fig.add_subplot(inner[j])
            a.imshow(im)
            a.axis("off")
            sources.append(dict(claim=title, panel=f"{i + 1}{'abcd'[j]}",
                                file=p))
            a.text(0, -0.01, f"{i + 1}{'abcd'[j]}", transform=a.transAxes,
                   fontsize=12, fontweight="bold", va="bottom")
    fig.savefig(os.path.join(out, "gradient_hypothesis_overview_index.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    claim_pages(rows, os.path.join(out, "gradient_hypothesis_overview.pdf"))
    with open(os.path.join(out, "sources.json"), "w") as f:
        json.dump(dict(created=str(datetime.date.today()),
                       script=os.path.relpath(__file__, REPO),
                       note="assembled from saved figures; nothing recomputed",
                       panels=sources), f, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
