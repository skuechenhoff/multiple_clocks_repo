#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure for `tuned_cofiring_hc_mpfc.py`. Plots `result.json` as is
(session mean, 95% CI from the stored SEM, two-sided t-test p):

    a  HC-mPFC population co-firing (r of the mean z of all cells)
    b  tuned coupling, HC now x mPFC now
    c  tuned coupling, HC now x mPFC future (30 + 60 deg)

    python scripts/tuned_cofiring_hc_mpfc_figures.py <folder>

@author: Svenja Kuchenhoff
"""
import os
import sys
import json

import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from now_coupling_hc_mpfc_figures import panel, MPFC, GREY, CM


def main(folder):
    R = json.load(open(os.path.join(folder, "result.json")))
    n = R["n_sessions"]
    T = R["tests"]
    fig, axes = plt.subplots(1, 3, figsize=(18 * CM, 8.5 * CM))
    panel(axes[0], lambda k, ph: T[f"population co-firing r | {k} | {ph}"],
          f"all cells\nHC × mPFC\n({n} sessions)",
          "r(HC, mPFC) across events\n(session mean, 95% CI)")
    for ax, pair, lab in ((axes[1], "HC now x mPFC now", "HC now × mPFC now"),
                          (axes[2], "HC now x mPFC future",
                           "HC now × mPFC 30–60°")):
        panel(ax, lambda k, ph, pair=pair: T[f"{pair} | {k} | {ph}"],
              f"tuned cells\n{lab}\n({n} sessions)",
              "r − shuffled-tuning r\n(session mean, 95% CI)")
    for ax, l in zip(axes, "abc"):
        ax.text(-0.42, 1.2, l, transform=ax.transAxes, fontsize=11,
                fontweight="bold")
    h = [plt.Rectangle((0, 0), 1, 1, color=MPFC),
         plt.Rectangle((0, 0), 1, 1, color=GREY)]
    fig.legend(h, ["ripples (window = ripple duration)",
                   "ordinary moments (no ripple within 0.5 s)"],
               loc="lower center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(folder, f"tuned_cofiring_hc_mpfc.{ext}"),
                    dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1])
