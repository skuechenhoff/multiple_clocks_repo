#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure: learning-phase hippocampal ripples and the mPFC plan code, per lag and
over all future lags, measured on the FIRST 3 execution loops of each block vs
on ALL of them. Own reading (no still-time covariate). Exploratory follow-up of
B1 (CHANGELOG 2026-11-14 (b)).

WHAT THE PLAN CODE IS, per block (one run of a configuration) and lag k:
  per mPFC cell, the rate map at lag k (mean phase-residualised firing over
  every (loop, bin) whose location k bins AHEAD is L) is built from the
  execution loops of every OTHER configuration (dwell-weighted mean) and
  correlated (dwell-weighted Pearson r) with the same map from THIS block's
  execution loops (its first 3, or all); block code = mean Fisher z over the
  session's mPFC cells. Read from B1's saved `cell_block_codes.csv`.
Future lags collapsed: the block code averaged over lags 30..330 (all 11
'beyond-now' lags) BEFORE the regression.

Per session (>= 8 blocks): OLS of the z-scored block code on z-scored
log(1 + learning ripples), log learning duration, n learning loops, block
order, run index; one ripple beta per session; sign-flip over sessions.

Usage:
    python scripts/plan_code_vs_ripples_learning_figure.py

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import datetime
import textwrap

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "exploratory"))
from plan_code_vs_ripples import session_beta, design
from per_lag_encoding_by_phase import fz, sign_flip, DATA_DIR
from plot_gradient_claims import plt, _clean, _p

SEED = 42
WHICHES = {"first3": ("first 3 execution loops", "#2E5E45"),
           "all": ("all execution loops", "#9CC5A8")}
LAGS = list(range(0, 360, 30))
FUTURE = list(range(30, 360, 30))
MIN_CELLS, MIN_LAGS = 2, 8
B1_DIR = os.path.join(DATA_DIR, "group", "swr", "plan_code_vs_ripples_2026-11-14")
OUT = os.path.join(DATA_DIR, "group", "swr",
                   f"plan_code_vs_ripples_learning_{datetime.date.today()}")
CM = 1 / 2.54


def block_codes(C, which):
    G = C[(C.roi == "mPFC") & (C.which == which)
          & C.lag.str.startswith("lag")].copy()
    G["z"] = fz(G.r)
    G["lagdeg"] = G.lag.str[3:].astype(int)
    A = G.groupby(["session", "grid_no", "lagdeg"]).z.agg(["mean", "count"])
    A = A[A["count"] >= MIN_CELLS]["mean"].unstack("lagdeg")
    v = A[FUTURE]
    A["future"] = v.mean(1).where(v.notna().sum(1) >= MIN_LAGS)
    return A.reset_index()


def partial_xy(D, col):
    xs, ys = [], []
    for _, d in D.groupby("session"):
        d = d.dropna(subset=[col])
        if len(d) < 8:
            continue
        cov = np.column_stack([np.ones(len(d)), np.log(d.learn_s),
                               d.n_learn_loops, d.order, d.run]).astype(float)
        cov = cov[:, np.r_[True, cov[:, 1:].std(0) > 0]]
        res = lambda v: v - cov @ np.linalg.lstsq(cov, v, rcond=None)[0]
        xs.append(res(np.log1p(d.n_rip_learn.to_numpy(float))))
        ys.append(res(d[col].to_numpy(float)))
    return np.concatenate(xs), np.concatenate(ys)


def compute(B, C, which):
    D = B.merge(block_codes(C, which), on=["session", "grid_no"])
    rows = []
    for s, d in D.groupby("session"):
        X = design(d, "own", "ripples")
        rec = dict(session=int(s), n_blocks=len(d))
        for o in LAGS + ["future"]:
            rec[str(o)] = session_beta(d[o].to_numpy(float), X)
        rows.append(rec)
    S = pd.DataFrame(rows)
    res = {str(o): sign_flip(S[str(o)].to_numpy(float))
           for o in LAGS + ["future"]}
    return D, S, res


def main():
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    B = pd.read_csv(os.path.join(B1_DIR, "blocks.csv"))
    C = pd.read_csv(os.path.join(B1_DIR, "cell_block_codes.csv"))
    out, Ds, Ss = {}, {}, {}
    for w in WHICHES:
        D, S, res = compute(B, C, w)
        D.to_csv(os.path.join(OUT, f"blocks_with_codes_{w}.csv"), index=False)
        S.to_csv(os.path.join(OUT, f"session_betas_{w}.csv"), index=False)
        Ds[w], Ss[w], out[w] = D, S, res
        print(f"\n  {w}")
        for o in LAGS + ["future"]:
            r = res[str(o)]
            print(f"    {str(o):7s} n={r['n']} beta {r['mean']:+.3f} ± "
                  f"{r['sem']:.3f}  p={r['p_perm']:.4f}")
    pa = Ss["first3"].set_index("session")["future"] - \
        Ss["all"].set_index("session")["future"]
    out["first3 - all | future"] = sign_flip(pa.dropna().to_numpy())
    print("\n  first3 - all, future:", out["first3 - all | future"])

    # ── figure ───────────────────────────────────────────────────────────
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(18 * CM, 13 * CM))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.3, 1.0, 1.5], wspace=0.6,
                          hspace=0.6)
    rng = np.random.default_rng(SEED)
    # a: per lag, both loop sets
    ax = fig.add_subplot(gs[0, :2])
    ax.axvspan(15, 345, color="#F2F2F2", lw=0, zorder=0)
    ax.text(180, 0.02, "future lags (averaged in b)", ha="center",
            va="bottom", fontsize=7, color="0.4",
            transform=ax.get_xaxis_transform())
    for k, (w, (lab, col)) in enumerate(WHICHES.items()):
        res = out[w]
        off = -4 if k == 0 else 4
        m = np.array([res[str(l)]["mean"] for l in LAGS])
        se = np.array([res[str(l)]["sem"] for l in LAGS])
        ax.errorbar(np.array(LAGS) + off, m, se, color=col, lw=1.4, marker="o",
                    ms=4, capsize=2, zorder=3, label=lab)
        for l, mm, ss in zip(LAGS, m, se):
            if res[str(l)]["p_perm"] < 0.05:
                ax.text(l + off, mm + ss + 0.03, "*", ha="center",
                        fontsize=11, color=col)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xticks(LAGS, [str(l) if l % 60 == 0 else "" for l in LAGS])
    ax.set_xlim(-15, 345)
    ax.set_xlabel("lag of the mPFC code (°)")
    ax.set_ylabel("effect of learning ripples\n(standardised β, mean ± SEM)")
    ax.set_title("a  per lag", loc="left", pad=14)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    _clean(ax)
    # b: future collapsed, both, with session lines
    ax = fig.add_subplot(gs[0, 2])
    f3 = Ss["first3"].set_index("session")["future"]
    fa = Ss["all"].set_index("session")["future"]
    for s in sorted(set(f3.dropna().index) & set(fa.dropna().index)):
        ax.plot([0, 1], [f3[s], fa[s]], color="0.85", lw=0.6, zorder=1)
    for x, (w, (lab, col)) in enumerate(WHICHES.items()):
        v = Ss[w]["future"].dropna()
        r = out[w]["future"]
        ax.scatter(x + rng.uniform(-0.12, 0.12, len(v)), v, s=9, color=col,
                   alpha=0.6, lw=0, zorder=2)
        ax.errorbar(x, r["mean"], r["sem"], fmt="o", ms=6, mfc=col, mec="k",
                    ecolor="k", capsize=3, zorder=3)
        ax.text(x, 1.01, f"{r['mean']:+.2f}\n{_p(r['p_perm'])}",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7)
    pr = out["first3 - all | future"]
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1], ["first 3\nloops", "all\nloops"])
    ax.set_ylabel("standardised β per session")
    ax.set_title(f"b  all future lags (30–330°)\nfirst 3 − all: "
                 f"{_p(pr['p_perm'])}", loc="left", pad=24, fontsize=9)
    _clean(ax)
    # c, d: partial regression for each loop set
    for k, (w, (lab, col)) in enumerate(WHICHES.items()):
        ax = fig.add_subplot(gs[1, 0] if k == 0 else gs[1, 1:])
        x, y = partial_xy(Ds[w], "future")
        ax.scatter(x, y, s=5, color=col, alpha=0.45, lw=0)
        b = np.polyfit(x, y, 1)
        xx = np.linspace(np.percentile(x, 2), np.percentile(x, 98), 5)
        ax.plot(xx, np.polyval(b, xx), color="k", lw=1.4)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("learning ripples, log(1 + n)\n(within session, "
                      "| covariates)")
        ax.set_ylabel("mPFC future code (Fisher z)\n(within session, "
                      "| covariates)")
        ax.set_title(f"{'cd'[k]}  {lab}: {len(x)} blocks", loc="left", pad=6,
                     fontsize=9)
        _clean(ax)
    fig.suptitle("Learning-phase hippocampal ripples and the mPFC plan code: "
                 "first 3 vs all execution loops", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold", y=1.06)
    r3 = out["first3"]["future"]
    fig.text(0.01, -0.03, textwrap.fill(
        "Plan code per block (one run of a configuration) and lag: per mPFC "
        "cell, the lag-shifted rate map from the execution loops of all OTHER "
        "configurations correlated (dwell-weighted r) with the same map from "
        "this block's first 3 execution loops (dark) or all of them (light); "
        "Fisher z averaged over the session's cells. Per session, the z-scored "
        "code regressed on z-scored log(1 + hippocampal ripples during the "
        "block's learning phase), with learning duration, number of learning "
        "loops, block order and re-encounter as covariates. a: mean ± SEM "
        "over sessions of the ripple β per lag; * sign-flip p < .05, "
        "uncorrected over 12 lags. b: the code averaged over all 11 future "
        "lags before the regression (one test per loop set); lines join "
        "sessions; first 3 − all = paired sign-flip. c, d: the same model as "
        "partial-regression plots, all blocks pooled. mPFC, "
        f"{r3['n']} sessions; no still-time covariate; exploratory.", 150),
        fontsize=7, color="0.3", va="top")
    for e in (".png", ".pdf"):
        fig.savefig(os.path.join(OUT, "learning_ripples_plan_code" + e),
                    dpi=300, bbox_inches="tight")
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(dict(settings=dict(
            analysis="learning ripples vs mPFC plan code (B1), first 3 vs all "
                     "execution loops",
            created=str(datetime.date.today()),
            script=os.path.relpath(__file__, REPO), seed=SEED, roi="mPFC",
            loop_sets={k: v[0] for k, v in WHICHES.items()},
            codes=os.path.join(B1_DIR, "cell_block_codes.csv"),
            future_lags=FUTURE, min_lags=MIN_LAGS, min_cells=MIN_CELLS,
            model="per session OLS, z-scored: code ~ log1p(learning ripples) "
                  "+ log(learning duration) + n learning loops + order + run",
            reading="own (no still-time covariate)",
            note="exploratory follow-up of B1 (CHANGELOG 2026-11-14 (b))"),
            tests=out), f, indent=1, default=str)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
