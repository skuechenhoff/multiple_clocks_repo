#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is the layout's execution pattern reinstated as its rewards are discovered?

    python scripts/cells_plan_reinstatement.py            # spikes
    python scripts/cells_plan_reinstatement.py hfb        # or theta, beta, ripple
    python scripts/cells_plan_reinstatement.py ripple premove   # or ripmask, ripshift<k>

LFP signals use one feature per bipolar derivation (mean envelope in the window)
in place of one per cell; everything downstream is identical.

The cell analogue of the fMRI loading result; see
`mc/analyse/plan_reinstatement.py` for the estimator and CHANGELOG.md
(2026-10-01, pre-registration) for the declared tests.

Output: data/ephys_humans/derivatives/group/plan_reinstatement_<signal>_<date>/
    per_event.csv, per_session.csv, summary.csv, results.json, settings.json,
    plan_reinstatement.jpeg

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import datetime

import numpy as np
import matplotlib.pyplot as plt

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.plan_reinstatement as pr
from mc.plotting.cell_results import STATE_QUADRANT_COLORS

np.random.seed(pr.SEED)
signal = sys.argv[1] if len(sys.argv) > 1 else "spikes"
variant = sys.argv[2] if len(sys.argv) > 2 else "full"
tag = signal if variant == "full" else f"{signal}_{variant}"

out_dir = os.path.join(rrsa._derivatives(), "group",
                       f"plan_reinstatement_{tag}_{datetime.date.today()}")
os.makedirs(out_dir, exist_ok=True)

E = pr.run(out_dir, signal, variant)
S = pr.session_table(E)
S.to_csv(os.path.join(out_dir, "per_session.csv"), index=False)
G = pr.group_tests(S)
G.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
pr.save_settings(out_dir, S.session.unique(), signal, variant)
json.dump({f"{r.roi}|{r.test}": r.dropna().to_dict()
           for _, r in G.iterrows()},
          open(os.path.join(out_dir, "results.json"), "w"), indent=2,
          default=float)

print(f"\n{'ROI':7s} {'test':38s} {'n':>3s} {'mean':>8s} {'t':>6s} "
      f"{'P(>0)':>7s} {'subj t':>7s} {'subj P':>7s}")
for _, r in G[G.test.isin(pr.TESTS)].iterrows():
    print(f"{r.roi:7s} {r.test:38s} {r.n:3d} {r['mean']:+8.4f} {r.t:+6.2f} "
          f"{r.p_one_sided:7.4f} {r.subj_t:+7.2f} {r.subj_p_one_sided:7.4f}")
print(f"-> {out_dir}")


# ---------------------------------------------------------------------------
# Overview figure: Δ by discovered state (encode) next to execution, per ROI
rois = list(S.roi.unique())
fig, axes = plt.subplots(1, len(rois), figsize=(4 * len(rois), 4.6), sharey=True)
x = np.arange(4)
for ax, roiname in zip(axes, rois):
    g = S[S.roi == roiname]
    for cond, dx, face in (("discovery", -0.12, True), ("execute", 0.12, False)):
        M = g[[f"{cond}_all_{s}" for s in pr.STATES]].to_numpy()
        n = np.isfinite(M).sum(axis=0)
        mu, se = np.nanmean(M, axis=0), np.nanstd(M, axis=0, ddof=1) / np.sqrt(n)
        if cond == "discovery":
            for row in M:
                ax.plot(x + dx, row, color="0.85", lw=0.6, zorder=1)
        ax.plot(x + dx, mu, color="0.3" if face else "0.6", lw=1.2, zorder=2,
                ls="-" if face else "--")
        for k in range(4):
            ax.errorbar(x[k] + dx, mu[k], se[k], fmt="o", ms=8, capsize=3,
                        color=STATE_QUADRANT_COLORS[k],
                        mfc=STATE_QUADRANT_COLORS[k] if face else "white",
                        zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    res = G[(G.roi == roiname)].set_index("test")
    fmt = lambda p: "P < 0.001" if p < 0.001 else f"P = {p:.3f}"
    p_main = res.loc["PRIMARY: discovery D - A"]
    p_ctrl = res.loc["positive control: execute Δ > 0"]
    p_abc = res.loc["heading-free slope over A, B, C"]
    ax.set_title(f"{roiname} ({p_main.n} sessions)\n"
                 f"D−A: t = {p_main.t:+.2f}, {fmt(p_main.p_one_sided)}\n"
                 f"slope A→C: t = {p_abc.t:+.2f}, {fmt(p_abc.p_one_sided)}\n"
                 f"execute: t = {p_ctrl.t:+.2f}, {fmt(p_ctrl.p_one_sided)}",
                 fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{k} known" for k, s in enumerate(pr.STATES)])
axes[0].set_ylabel("Δ similarity to same-layout execution\n"
                   "minus other layouts (z·z / feature)")
axes[0].plot([], [], "o-", color="0.3", label="first discovery (encode)")
axes[0].plot([], [], "o--", color="0.6", mfc="white",
             label="execute uncovers (positive control)")
axes[0].legend(fontsize=9, loc="lower left", frameon=False)
fig.supxlabel("reward uncovered (number of future rewards known at that moment)",
              fontsize=10)
fig.suptitle(f"[{tag}] Plan reinstatement: is this layout's execution pattern present "
             "when its rewards are discovered? (mean ± SEM over sessions; "
             "grey = single sessions)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "plan_reinstatement.jpeg"), dpi=150)
plt.show()
