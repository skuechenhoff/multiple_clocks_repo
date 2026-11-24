#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is the first-run medial frontal D - A effect heading, or ripple-locked?

    python scripts/plan_reinstatement_control_figures.py

Reads the per-session tables of `cells_plan_reinstatement.py` for the variants
    full      [t+0.05, t+1.0] s after the reward
    premove   window ends at the first move after the reward
    ripmask   samples within ±0.2 s of a hippocampal ripple dropped
    ripshift  the same mask on a ripple train shifted ±2-5 s (5 shifts; the
              per-session values are averaged over shifts)
Pre-registered in CHANGELOG.md (2026-10-01).

Panels
    a  ripple band, first-run D - A, per variant
    b  HFB, the same
    c  ripple band Δ at A-D (first run): full / ripmask / ripshift
    d  per session: ripmask vs ripshift D - A (ripple band)
    e  fraction of each window the ripple mask removes, by event
    f  mPFC spikes: full vs premove

Statistics are computed first (same `signflip_t` as the analysis) and written
to `figure_stats.json`; plotting only displays them.

@author: Svenja Kuchenhoff
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc
import mc.analyse.plan_reinstatement as pr
from mc.plotting.cell_results import STATE_QUADRANT_COLORS

DATE = "2026-10-01"
GROUP = os.path.join(rrsa._derivatives(), "group")
OUT = os.path.join(GROUP, f"plan_reinstatement_controls_{DATE}")
os.makedirs(OUT, exist_ok=True)
N_SHIFT = 5
VARIANTS = {"full": ("full window", "#B74C2D"),
            "premove": ("before first\nmove", "#DC673E"),
            "ripmask": ("ripples\nmasked", "#448363"),
            "ripshift": ("shifted mask\n(control)", "#7BB594")}


def table(signal, variant):
    tag = signal if variant == "full" else f"{signal}_{variant}"
    return pd.read_csv(os.path.join(GROUP, f"plan_reinstatement_{tag}_{DATE}",
                                    "per_session.csv"))


def shifted(signal, roi):
    """Per-session values averaged over the N_SHIFT shifted-mask runs."""
    T = pd.concat([table(signal, f"ripshift{k}") for k in range(N_SHIFT)])
    T = T[T.roi == roi]
    num = T.select_dtypes("number").columns.drop("session")
    return T.groupby("session")[num].mean().reset_index()


# ---------------------------------------------------------------------------
# data and statistics
D = {}
for sig in ("ripple", "hfb"):
    for v in ("full", "premove", "ripmask"):
        t = table(sig, v)
        D[sig, v] = t[t.roi == "MedialFrontal"]
    D[sig, "ripshift"] = shifted(sig, "MedialFrontal")
for v in ("full", "premove"):
    t = table("spikes", v)
    D["spikes", v] = t[t.roi == "mPFC"]

stat = {f"{s}|{v}|{c}": pr.signflip_t(g[c])
        for (s, v), g in D.items()
        for c in ("D_minus_A_run0", "D_minus_A_all", "execute_all_mean")}
pair = D["ripple", "ripmask"][["session", "D_minus_A_run0"]].merge(
    D["ripple", "ripshift"][["session", "D_minus_A_run0"]], on="session",
    suffixes=("_mask", "_shift")).dropna()
stat["ripple|ripmask_minus_ripshift|D_minus_A_run0"] = pr.signflip_t(
    pair.D_minus_A_run0_mask - pair.D_minus_A_run0_shift)

# how much of each window does the ripple mask remove?
rip = pd.read_csv(os.path.join(GROUP, "swr", "bundle_v2", "ripples.csv"),
                  usecols=["session", "t_peak_s"])
sessions = sorted(D["ripple", "full"].session.unique())
steps = swl.load(sessions)
cov = []
for s in sessions:
    ev = pr.uncover_table(steps[steps.session == s], swc.occupancy(steps, s))
    t_rip = np.sort(rip[rip.session == s].t_peak_s.to_numpy(float))
    t_rip = t_rip[np.concatenate([[True], np.diff(t_rip) > 0.05])]
    t0, _ = pr.event_windows(ev, "full")
    grid = t0[:, None] + np.arange(0, pr.WINDOW[1] - pr.WINDOW[0], 0.01)
    j = np.searchsorted(t_rip, grid)
    near = np.minimum(np.abs(grid - t_rip[np.clip(j - 1, 0, len(t_rip) - 1)]),
                      np.abs(t_rip[np.clip(j, 0, len(t_rip) - 1)] - grid))
    frac = (near <= pr.RIPPLE_MASK_S).mean(axis=1)
    ev["masked"] = frac
    for k, S in enumerate(pr.STATES):
        m = ev.first_discovery & (ev.state == k) & (ev.run == 0)
        cov.append(dict(session=s, event=f"discovery {S}",
                        frac=ev.masked[m].mean()))
    cov.append(dict(session=s, event="execute",
                    frac=ev.masked[ev.phase == "execute"].mean()))
COV = pd.DataFrame(cov)
COV.to_csv(os.path.join(OUT, "mask_coverage.csv"), index=False)
stat["mask_coverage_mean"] = COV.groupby("event").frac.mean().to_dict()
json.dump(stat, open(os.path.join(OUT, "figure_stats.json"), "w"), indent=2)


# ---------------------------------------------------------------------------
# plotting
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


def ptxt(d):
    p = d["p_one_sided"]
    return (f"t = {d['t']:+.2f}\nP = {p:.3f}" if p >= .001
            else f"t = {d['t']:+.2f}\nP < .001") + f"\nn = {d['n']}"


def strip(ax, sig, variants, title):
    for i, v in enumerate(variants):
        lab, c = VARIANTS[v]
        x = D[sig, v].D_minus_A_run0.dropna().to_numpy()
        jit = np.random.default_rng(pr.SEED).uniform(-0.15, 0.15, len(x))
        ax.scatter(i + jit, x, s=10, color=c, alpha=0.35, lw=0)
        d = stat[f"{sig}|{v}|D_minus_A_run0"]
        ax.errorbar(i, d["mean"], d["sem"], fmt="o", ms=7, color=c, mec="k",
                    mew=0.6, capsize=3, zorder=3)
        ax.text(i, 1.0, ptxt(d), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([VARIANTS[v][0] for v in variants], fontsize=8)
    ax.set_xlim(-0.6, len(variants) - 0.4)
    ax.set_ylabel("first-run D − A of Δ, per session")
    ax.set_title(title, pad=46, fontsize=10)


fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
allv = ["full", "premove", "ripmask", "ripshift"]
strip(axes[0, 0], "ripple", allv,
      "a  medial frontal ripple band (80-120 Hz)")
strip(axes[0, 1], "hfb", allv, "b  medial frontal HFB (70-150 Hz)")

ax = axes[0, 2]
x = np.arange(4)
for v, dx in (("full", -0.1), ("ripmask", 0.0), ("ripshift", 0.1)):
    lab, c = VARIANTS[v]
    M = D["ripple", v][[f"discovery_run0_{k}" for k in pr.STATES]].to_numpy()
    n = np.isfinite(M).sum(axis=0)
    mu = np.nanmean(M, axis=0)
    se = np.nanstd(M, axis=0, ddof=1) / np.sqrt(n)
    ax.plot(x + dx, mu, color=c, lw=1.5, label=lab.replace("\n", " "))
    for k in range(4):
        ax.errorbar(x[k] + dx, mu[k], se[k], fmt="o", ms=6, capsize=3,
                    color=c, mfc=STATE_QUADRANT_COLORS[k], mec=c, mew=1.5)
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{k}\n{i} known" for i, k in enumerate(pr.STATES)])
ax.set_xlabel("reward discovered (future rewards known)")
ax.set_ylabel("Δ similarity, same − other layout\n(z·z / contact)")
ax.set_title("c  ripple band, first run: Δ at each discovery\n"
             "(marker fill = state; mean ± SEM)", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper left")

ax = axes[1, 0]
lim = np.nanmax(np.abs(pair[["D_minus_A_run0_mask",
                             "D_minus_A_run0_shift"]].to_numpy())) * 1.1
ax.scatter(pair.D_minus_A_run0_shift, pair.D_minus_A_run0_mask, s=18,
           color=VARIANTS["ripmask"][1], alpha=0.7, lw=0)
ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.8, ls="--")
ax.axhline(0, color="0.4", lw=0.6)
ax.axvline(0, color="0.4", lw=0.6)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
d = stat["ripple|ripmask_minus_ripshift|D_minus_A_run0"]
ax.set_title("d  ripple band, first-run D − A per session\n"
             f"masked − shifted: t = {d['t']:+.2f}, "
             f"P(two-sided) = {d['p_two_sided']:.3f}", fontsize=10)
ax.set_xlabel("shifted mask (control): D − A")
ax.set_ylabel("ripples masked: D − A")

ax = axes[1, 1]
evs = [f"discovery {S}" for S in pr.STATES] + ["execute"]
for i, e in enumerate(evs):
    v = COV[COV.event == e].frac.dropna().to_numpy() * 100
    c = STATE_QUADRANT_COLORS[i] if i < 4 else "0.5"
    jit = np.random.default_rng(pr.SEED).uniform(-0.15, 0.15, len(v))
    ax.scatter(i + jit, v, s=10, color=c, alpha=0.4, lw=0)
    ax.errorbar(i, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), fmt="o", ms=7,
                color=c, mec="k", mew=0.6, capsize=3, zorder=3)
ax.set_xticks(range(len(evs)))
ax.set_xticklabels(["A", "B", "C", "D", "execute"])
ax.set_xlabel("first discovery (first run) / execution uncovers")
ax.set_ylabel("% of the 1 s window within ±0.2 s\nof a hippocampal ripple")
ax.set_title("e  how much the ripple mask removes", fontsize=10)

strip(axes[1, 2], "spikes", ["full", "premove"],
      "f  mPFC spikes (for reference)")

fig.suptitle("Is the first-run medial frontal D − A effect heading, or "
             "ripple-locked?\nMedialFrontal = mPFC + mOFC within |MNI x| ≤ "
             "20 mm. One-sided sign-flip P (d: two-sided), n = sessions. "
             "EXPLORATORY.", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(OUT, "plan_reinstatement_controls.jpeg"), dpi=200)
fig.savefig(os.path.join(OUT, "plan_reinstatement_controls.pdf"))
print(json.dumps(stat, indent=1, default=float))
print(f"-> {OUT}")
plt.show()
