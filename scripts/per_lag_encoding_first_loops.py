#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How does the mPFC lag profile (F2, panel a) look when execution is restricted
to the FIRST 3 execution loops of every block? Exploratory, descriptive.

F2's estimator unchanged (`per_lag_encoding_by_phase`: per configuration, rate
maps pooled over its loops, phase residualised within phase, leave-one-
configuration-out, dwell-weighted r), run within four loop sets:

    exe_all     every execution loop                      (= F2 r_exe)
    exe_first3  each block's first 3 execution loops      (what B1 measures)
    exe_last3   each block's last 3 execution loops       (same amount of data
                                                           as first3: control)
    disc        discovery loops                           (= F2 r_disc)

A configuration's map pools the chosen loops of all its blocks (runs). Fewer
loops make maps noisier, so first3 vs all mixes 'early' with 'less data';
first3 vs last3 does not. Per session mean Fisher z over mPFC cells;
sign-flip over sessions per lag, and paired first3 - last3, first3 - all.
Execution loops of a block = as in B1 (published execution rows, the block's
first traversal excluded).

Usage:
    python scripts/per_lag_encoding_first_loops.py

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import zlib
from datetime import datetime

import numpy as np
import pandas as pd

from mc.analyse.roi_labels import read_per_cell_table
from joblib import Parallel, delayed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "exploratory"))
import mc.analyse.helpers_human_cells as hh
from mc.analyse.future_spatial_peaks import _residualise_phase
from per_lag_encoding_by_phase import (pooled_map, predicted, score, fz,
                                       sign_flip, PER_CELL, ROI_NAME_MAP,
                                       DATA_DIR, LAGS_DEG, MIN_CFG)

SEED = 42
N_LOOPS = 3
SETS = {"exe_all": ("all execution loops", "#5C1027"),
        "exe_first3": ("first 3 execution loops", "#2E5E45"),
        "exe_last3": ("last 3 execution loops", "#9CC5A8"),
        "disc": ("discovery loops", "#D7657F")}
OUT = os.path.join(DATA_DIR, "group", "per_lag_encoding_by_phase",
                   f"first_loops_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")


def row_sets(beh):
    exe = ~((beh.rep_correct == 0) & (beh.correct == 0)).to_numpy()
    first3, last3 = np.zeros(len(beh), bool), np.zeros(len(beh), bool)
    for g, gb in beh.groupby("grid_no", sort=True):
        r = np.flatnonzero((beh.grid_no == g).to_numpy() & exe)
        r = r[r != gb.index[0]]
        first3[r[:N_LOOPS]] = True
        last3[r[-N_LOOPS:]] = True
    return dict(exe_all=exe, exe_first3=first3, exe_last3=last3, disc=~exe), exe


def loco(Y, L, cfg, rows):
    out = {}
    cfgs = [c for c in np.unique(cfg) if (rows & (cfg == c)).any()]
    if len(cfgs) < 2:
        return {l: np.nan for l in LAGS_DEG}
    for lag in LAGS_DEG:
        maps = {c: pooled_map(Y[rows & (cfg == c)], L[rows & (cfg == c)], lag)
                for c in cfgs}
        rs = [score(*predicted(maps, [x for x in cfgs if x != c]), *maps[c])
              for c in cfgs]
        rs = np.asarray(rs, float)
        out[lag] = float(np.nanmean(rs)) if np.isfinite(rs).any() else np.nan
    return out


def cell(meta, X, L, cfg, sets, exe):
    Y = np.full_like(X, np.nan)
    for m in (exe, ~exe):
        if m.sum():
            Y[m] = _residualise_phase(X[m], basis="cosine")
    rec = dict(meta)
    for k, rows in sets.items():
        for lag, r in loco(Y, L, cfg, rows).items():
            rec[f"{k}_lag{lag:03d}"] = r
    return rec


def main(n_jobs=-1):
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    cells = read_per_cell_table(PER_CELL)
    cells["roi"] = cells.roi.map(lambda r: ROI_NAME_MAP.get(r, r))
    cells = cells[cells.roi == "mPFC"]
    tasks = []
    for s in sorted(cells.subject_id.unique()):
        d = hh.load_norm_data(DATA_DIR, [f"{int(s):02d}"], res_data=False)
        if not d:
            continue
        S = d[f"sub-{int(s):02d}"]
        beh = S["beh"].reset_index(drop=True)
        L = S["locations"].to_numpy(float)
        sets, exe = row_sets(beh)
        _, cfg = np.unique(beh[["loc_A", "loc_B", "loc_C", "loc_D"]]
                           .to_numpy(), axis=0, return_inverse=True)
        cfg = np.asarray(cfg).ravel()
        if len(np.unique(cfg[exe])) < MIN_CFG:
            continue
        keep = cells[cells.subject_id == s].set_index("neuron")
        for nid, df in S["normalised_neurons"].items():
            if nid in keep.index:
                tasks.append(dict(meta=dict(neuron=nid, session=int(s)),
                                  X=df.to_numpy(float), L=L, cfg=cfg,
                                  sets=sets, exe=exe))
    print(f"  {len(tasks)} mPFC cells")
    rows = Parallel(n_jobs=n_jobs, verbose=2)(delayed(cell)(**t) for t in tasks)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "per_cell.csv"), index=False)
    # session means of Fisher z, then tests
    Z = T.copy()
    cols = [c for c in T.columns if "_lag" in c]
    Z[cols] = fz(Z[cols].to_numpy(float))
    Sm = Z.groupby("session")[cols].mean()
    Sm.to_csv(os.path.join(OUT, "per_session.csv"))
    res = {}
    for k in SETS:
        for l in LAGS_DEG:
            res[f"{k} | {l}"] = sign_flip(Sm[f"{k}_lag{l:03d}"].to_numpy())
    for a, b in (("exe_first3", "exe_last3"), ("exe_first3", "exe_all")):
        for l in LAGS_DEG:
            res[f"{a} - {b} | {l}"] = sign_flip(
                (Sm[f"{a}_lag{l:03d}"] - Sm[f"{b}_lag{l:03d}"]).to_numpy())
    for l in LAGS_DEG:
        print(f"  {l:3d}  " + "  ".join(
            f"{k} {res[f'{k} | {l}'].get('mean', np.nan):+.3f}"
            f"{'*' if res[f'{k} | {l}'].get('p_perm', 1) < .05 else ' '}"
            for k in SETS) + f"   first3-last3 "
            f"{res[f'exe_first3 - exe_last3 | {l}'].get('mean', np.nan):+.3f} "
            f"p={res[f'exe_first3 - exe_last3 | {l}'].get('p_perm', np.nan):.3f}")
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(dict(settings=dict(
            analysis="mPFC lag profile, first 3 vs last 3 vs all execution "
                     "loops (F2 estimator)",
            created=datetime.now().isoformat(timespec="seconds"),
            script=os.path.relpath(__file__, REPO), seed=SEED,
            n_loops=N_LOOPS, sets={k: v[0] for k, v in SETS.items()},
            estimator="F2: pooled rate maps per configuration, LOCO, "
                      "dwell-weighted r; phase residualised within phase",
            unit="session mean Fisher z over mPFC cells",
            note="exploratory, descriptive"), tests=res), f, indent=1,
            default=str)
    draw(OUT, Sm, res)
    print(f"-> {OUT}")


def draw(d, Sm, res):
    from plot_gradient_claims import plt, _clean, _p
    import textwrap
    CM = 1 / 2.54
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(18 * CM, 7 * CM),
                             gridspec_kw=dict(width_ratios=[1.6, 1],
                                              wspace=0.4))
    ax = axes[0]
    for k, (lab, col) in SETS.items():
        m = np.array([res[f"{k} | {l}"]["mean"] for l in LAGS_DEG])
        se = np.array([res[f"{k} | {l}"]["sem"] for l in LAGS_DEG])
        ax.plot(LAGS_DEG, m, color=col, lw=1.5, marker="o", ms=3.5,
                label=lab, ls="--" if k == "disc" else "-")
        ax.fill_between(LAGS_DEG, m - se, m + se, color=col, alpha=0.12, lw=0)
        for l, mm, ss in zip(LAGS_DEG, m, se):
            if res[f"{k} | {l}"]["p_perm"] < 0.05:
                ax.text(l, mm + ss + 0.004, "*", color=col, ha="center",
                        fontsize=10)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xticks(LAGS_DEG, [str(l) if l % 60 == 0 else "" for l in LAGS_DEG])
    ax.set_xlabel("future lag (°)")
    ax.set_ylabel("spatial consistency\n(CV r, Fisher z; mean ± SEM)")
    n = res["exe_all | 0"]["n"]
    ax.set_title(f"a  mPFC lag profile by loop set ({n} sessions)", loc="left",
                 pad=8)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    _clean(ax)
    ax = axes[1]
    for k, col, lab in (("exe_first3 - exe_last3", "#2E5E45",
                         "first 3 − last 3"),
                        ("exe_first3 - exe_all", "#5C1027",
                         "first 3 − all")):
        m = np.array([res[f"{k} | {l}"]["mean"] for l in LAGS_DEG])
        se = np.array([res[f"{k} | {l}"]["sem"] for l in LAGS_DEG])
        ax.plot(LAGS_DEG, m, color=col, lw=1.4, marker="o", ms=3, label=lab)
        ax.fill_between(LAGS_DEG, m - se, m + se, color=col, alpha=0.12, lw=0)
        for l, mm, ss in zip(LAGS_DEG, m, se):
            if res[f"{k} | {l}"]["p_perm"] < 0.05:
                ax.text(l, mm + ss + 0.004, "*", color=col, ha="center",
                        fontsize=10)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xticks(LAGS_DEG, [str(l) if l % 60 == 0 else "" for l in LAGS_DEG])
    ax.set_xlabel("future lag (°)")
    ax.set_ylabel("difference (Fisher z)")
    ax.set_title("b  paired differences", loc="left", pad=8)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    _clean(ax)
    fig.suptitle("mPFC lag profile: early vs late execution loops", x=0.01,
                 ha="left", fontsize=10.5, fontweight="bold", y=1.05)
    fig.text(0.01, -0.08, textwrap.fill(
        "F2 estimator (per configuration, rate maps pooled over the chosen "
        "loops of all its blocks; leave-one-configuration-out, dwell-weighted "
        "r; phase residualised within phase), run separately on every "
        "execution loop (= F2 'execution'), each block's first 3 execution "
        "loops (what B1 measures), each block's last 3 (same amount of data "
        "as the first 3), and the discovery loops (= F2). Session = mean "
        "Fisher z over mPFC cells; * sign-flip p < .05, uncorrected over 12 "
        "lags. Fewer loops give noisier maps, so 'first 3 − all' mixes early "
        "with less data; 'first 3 − last 3' does not. Exploratory.", 150),
        fontsize=7, color="0.3", va="top")
    for e in (".png", ".pdf"):
        fig.savefig(os.path.join(d, "lag_profile_first_vs_last_loops" + e),
                    dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
