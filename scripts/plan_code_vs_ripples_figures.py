"""
Figure for `plan_code_vs_ripples.py` (B1), drawn from its saved outputs.

@author: Svenja Kuchenhoff
"""

import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "exploratory"))
from plot_gradient_claims import (plt, layout, caption, save, dots, zero,
                                  _clean, _p, COL, SEED)
from per_lag_encoding_by_phase import fz

READ_C = {"own": "#0e3d3a", "matched": "#a30d6c"}
READ_L = {"own": "own (duration covariate)",
          "matched": "stillness (still-time covariate)"}


def partial_xy(D):
    """Session-wise residuals of code and log1p(ripples) on the covariates of
    the own reading -- the partial-regression view of the primary model."""
    xs, ys = [], []
    for _, d in D.groupby("session"):
        d = d.dropna(subset=["mean"])
        if len(d) < 8:
            continue
        cov = np.column_stack([np.ones(len(d)), np.log(d.learn_s),
                               d.n_learn_loops, d.order, d.run]).astype(float)
        cov = cov[:, np.r_[True, cov[:, 1:].std(0) > 0]]
        def res(v):
            return v - cov @ np.linalg.lstsq(cov, v, rcond=None)[0]
        xs.append(res(np.log1p(d.n_rip_learn.to_numpy(float))))
        ys.append(res(d["mean"].to_numpy(float)))
    return np.concatenate(xs), np.concatenate(ys)


def draw(d):
    B = pd.read_csv(os.path.join(d, "blocks.csv"))
    C = pd.read_csv(os.path.join(d, "cell_block_codes.csv"))
    S = pd.read_csv(os.path.join(d, "session_betas.csv"))
    R = json.load(open(os.path.join(d, "result.json")))["tests"]
    rng = np.random.default_rng(SEED)
    fig, axes = layout([1, 1, 1, 2, 1])
    # a: what the predictor looks like
    ax = axes[0]
    ax.hist(B.n_rip_learn, bins=np.arange(0, 80, 2), color="#7E6B8F", lw=0)
    ax.set_xlabel("hippocampal ripples while learning\n(per block)")
    ax.set_ylabel("blocks")
    ax.set_title(f"a  {len(B)} blocks, median "
                 f"{int(B.n_rip_learn.median())} ripples\n(axis cut at 80; "
                 f"{int((B.n_rip_learn >= 80).sum())} blocks beyond)",
                 pad=10, fontsize=8)
    _clean(ax)
    # b: partial regression, mPFC 60 deg, first 3 loops
    ax = axes[1]
    G = C[(C.roi == "mPFC") & (C.lag == "lag060") & (C.which == "first3")].copy()
    G["z"] = fz(G.r)
    P = G.groupby(["session", "grid_no"]).z.agg(["mean", "count"]).reset_index()
    D = B.merge(P[P["count"] >= 2], on=["session", "grid_no"])
    x, y = partial_xy(D)
    ax.scatter(x, y, s=5, color=COL["mPFC"], alpha=0.4, lw=0)
    b = np.polyfit(x, y, 1)
    xx = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 5)
    ax.plot(xx, np.polyval(b, xx), color="k", lw=1.4)
    zero(ax)
    r = R["mPFC | lag060 | first3 | own | ripples"]
    ax.set_xlabel("log(1 + ripples while learning)\n| covariates, within session")
    ax.set_ylabel("mPFC 60° plan code, first 3\nexecution loops | covariates")
    ax.set_title(f"b  PRIMARY: {r['n']} sessions\nmean β = {r['mean']:+.3f}, "
                 f"{_p(r['p_perm'])}", pad=10)
    _clean(ax)
    # c: per-session betas, the key rows, both readings
    ax = axes[2]
    keys = (("mPFC", "lag060"), ("mPFC", "lag000"), ("mPFC", "stack30-90"),
            ("mOFC", "lag060"), ("HC", "lag000"))
    for i, (g, lag) in enumerate(keys):
        for j, rd in enumerate(("own", "matched")):
            v = S[(S.group == g) & (S.lag == lag) & (S.which == "first3")
                  & (S.reading == rd) & (S.predictor == "ripples")].beta
            rr = R[f"{g} | {lag} | first3 | {rd} | ripples"]
            dots(ax, i * 2.5 + j, v, READ_C[rd], rng=rng)
            ax.text(i * 2.5 + j, 1.0 + 0.07 * j, _p(rr["p_perm"]),
                    transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=5.8, color=READ_C[rd])
    zero(ax)
    ax.set_xticks([i * 2.5 + 0.5 for i in range(len(keys))],
                  [f"{g}\n{l.replace('lag0', '').replace('lag', '')}"
                   .replace("stack30-90", "30–90°").replace("60", "60°")
                   .replace("00", "0°") for g, l in keys], fontsize=6.3)
    ax.set_ylabel("ripple β per session\n(standardised)")
    ax.set_ylim(-1.6, 1.8)
    ax.set_title("c  ripple effect by region and lag", pad=22)
    _clean(ax)
    # d: beta across lags, mPFC
    ax = axes[3]
    lags = list(range(0, 360, 30))
    for rd in ("own", "matched"):
        for which, ls in (("first3", "-"), ("all", ":")):
            m, se, sig = [], [], []
            for l in lags:
                rr = R[f"mPFC | lag{l:03d} | {which} | {rd} | ripples"]
                m.append(rr.get("mean", np.nan))
                se.append(rr.get("sem", np.nan))
                sig.append(rr.get("p_perm", 1) < 0.05)
            m, se = np.array(m), np.array(se)
            ax.plot(lags, m, color=READ_C[rd], ls=ls, lw=1.3, marker="o",
                    ms=2.5, label=f"{READ_L[rd]}, "
                                  f"{'first 3 loops' if which == 'first3' else 'all loops'}")
            if ls == "-":
                ax.fill_between(lags, m - se, m + se, color=READ_C[rd],
                                alpha=0.15, lw=0)
            for l, mm, s_ in zip(lags, m, sig):
                if s_:
                    ax.text(l, mm, "*", color=READ_C[rd], ha="center",
                            fontsize=9)
    zero(ax)
    ax.set_xticks(lags)
    ax.set_xlabel("lag of the plan code (°)")
    ax.set_ylabel("mPFC: mean ripple β\nover sessions")
    ax.set_title("d  mPFC: which lag depends on learning ripples?", pad=10)
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.32))
    _clean(ax)
    # e: negative control
    ax = axes[4]
    for j, rd in enumerate(("own", "matched")):
        for i, pred in enumerate(("ripples", "prev_exe_ripples")):
            v = S[(S.group == "mPFC") & (S.lag == "lag060")
                  & (S.which == "first3") & (S.reading == rd)
                  & (S.predictor == pred)].beta
            rr = R[f"mPFC | lag060 | first3 | {rd} | {pred}"]
            dots(ax, i * 2.5 + j, v, READ_C[rd], rng=rng)
            ax.text(i * 2.5 + j, 1.0 + 0.07 * j, _p(rr["p_perm"]),
                    transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=5.8, color=READ_C[rd])
    zero(ax)
    ax.set_xticks([0.5, 3.0], ["this block's\nlearning ripples",
                               "previous block's\nexecution ripples"],
                  fontsize=6.3)
    ax.set_ylabel("ripple β (mPFC, 60°)")
    ax.set_title("e  negative control", pad=22)
    _clean(ax)
    mx = R.get("MixedLM | mPFC | lag060 | first3 | own", {})
    fig.suptitle("B1 — do more ripples while learning give a stronger mPFC "
                 "plan code?", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", y=1.0)
    caption(fig, "One block = one run of a configuration. Learning = grid "
            "onset to the start of the first fully correct repeat; ripples = "
            "hippocampal ripples in that window. Plan code = per cell, the "
            "execution rate maps of all OTHER configurations predict this "
            "block's first 3 execution loops (F2 estimator, dwell-weighted r), "
            "Fisher z averaged over the session's cells of the ROI — no cell "
            "is classified. Per session, OLS of the code on z-scored "
            "log(1 + ripples) with log learning duration (own) or log still "
            "time (stillness), number of learning loops, block order and run "
            "index as covariates; dots = sessions' ripple β; p = sign-flip "
            "over sessions, uncorrected (* in d). b: partial-regression view "
            "of the primary model, all blocks pooled. e: the same model with "
            "the previous block's execution ripples as predictor. MixedLM "
            f"(all blocks, session-centred): β = {mx.get('beta', np.nan):+.4f},"
            f" p = {mx.get('p', np.nan):.3f}, {mx.get('n_blocks')} blocks.",
            below=-0.07)
    save(fig, d, "B1_plan_code_vs_ripples")
