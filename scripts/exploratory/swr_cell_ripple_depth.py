#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C4 -- are ventral medial-frontal cells more ripple-modulated during encoding?

Pre-registered in CHANGELOG.md, 2026-10-30 (C4).

Per cell, the ripple modulation index from `lag_group_ripple`
    m = (rate inside ripples - rate outside) / (sum)
for the encode and the execute phase. Cells on one microwire bundle share a
coordinate, so the unit is the RECORDING SITE (session x MNI coordinate):
cells are averaged per site, and the null permutes site depth.

PRIMARY: Spearman across sites of [encode - execute] with MNI z; SK predicts
< 0 (ventral sites more ripple-modulated in encoding, relative to execution).

Usage:
    conda activate env_multiple_clocks
    python scripts/exploratory/swr_cell_ripple_depth.py

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import datetime

import numpy as np
import pandas as pd

from mc.analyse.roi_labels import read_per_cell_table
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "exploratory"))
import mc.analyse.swr_io as swr_io
from mc.analyse.swr_explore.preferred_lag_firing import PER_CELL
from swr_hfb_direction import ROI_COL

SEED = 42
N_PERM = 10000
ROIS = ("mPFC", "mOFC")
CM = 1 / 2.54


def site_table():
    group = os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                         "group")
    m = pd.read_csv(os.path.join(group, "swr", "lag_group_ripple_2026-10-01",
                                 "per_cell.csv"))
    m = m[m.roi.isin(ROIS)]
    w = m.pivot_table(index=["session", "cell", "roi"], columns="phase",
                      values="ripmod").reset_index()
    c = read_per_cell_table(PER_CELL).dropna(subset=["cell_idx", "subject_id"])
    c = c.assign(session=c.subject_id.astype(int),
                 cell=c.cell_idx.astype(int) - 1)[
        ["session", "cell", "MNI_x", "MNI_y", "MNI_z"]]
    w = w.merge(c, on=["session", "cell"], how="inner")
    w["site"] = (w.session.astype(str) + "|" + w.MNI_x.round(1).astype(str)
                 + "|" + w.MNI_y.round(1).astype(str) + "|"
                 + w.MNI_z.round(1).astype(str))
    S = w.groupby(["site", "session", "roi"]).agg(
        z=("MNI_z", "first"), encode=("encode", "mean"),
        execute=("execute", "mean"), n_cells=("cell", "size")).reset_index()
    S["enc_minus_exe"] = S.encode - S.execute
    return w, S


def spearman_perm(x, y):
    """Spearman rho and a two-sided p from permuting x (site depth); the same
    function gives the observed value and every null draw."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = stats.rankdata(x[ok]), stats.rankdata(y[ok])
    rho = lambda a: np.corrcoef(a, y)[0, 1]
    obs = rho(x)
    rng = np.random.default_rng(SEED)
    null = np.array([rho(rng.permutation(x)) for _ in range(N_PERM)])
    return dict(n_sites=int(ok.sum()), rho=float(obs),
                p_perm=float((1 + np.sum(np.abs(null) >= abs(obs)))
                             / (1 + N_PERM)))


def label_perm(a, b):
    v = np.r_[a, b]
    lab = np.r_[np.ones(len(a), bool), np.zeros(len(b), bool)]
    diff = lambda l: v[l].mean() - v[~l].mean()
    obs = diff(lab)
    rng = np.random.default_rng(SEED)
    null = np.array([diff(rng.permutation(lab)) for _ in range(N_PERM)])
    return dict(n_a=len(a), n_b=len(b), diff=float(obs),
                p_perm=float((1 + np.sum(np.abs(null) >= abs(obs)))
                             / (1 + N_PERM)))


def main():
    group = os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                         "group", "swr")
    d = os.path.join(group, f"cell_ripple_depth_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    W, S = site_table()
    S = S.dropna(subset=["encode", "execute"])
    res = {"PRIMARY enc - exe vs z (mPFC + mOFC sites)":
           spearman_perm(S.z.to_numpy(), S.enc_minus_exe.to_numpy())}
    for col in ("encode", "execute"):
        res[f"{col} vs z"] = spearman_perm(S.z.to_numpy(), S[col].to_numpy())
    m = S[S.roi == "mPFC"]
    res["mPFC sites only: enc - exe vs z"] = spearman_perm(
        m.z.to_numpy(), m.enc_minus_exe.to_numpy())
    for col in ("encode", "execute", "enc_minus_exe"):
        res[f"mOFC - mPFC sites: {col}"] = label_perm(
            S[S.roi == "mOFC"][col].to_numpy(), S[S.roi == "mPFC"][col].to_numpy())
    for k, r in res.items():
        print(f"  {k:46s} {r}")
    S.to_csv(os.path.join(d, "sites.csv"), index=False)
    W.to_csv(os.path.join(d, "cells.csv"), index=False)
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(dict(settings=dict(
            analysis="C4_cell_ripple_depth", created=str(datetime.date.today()),
            script=os.path.relpath(__file__, REPO), seed=SEED, n_perm=N_PERM,
            source="lag_group_ripple_2026-10-01/per_cell.csv (ripmod)",
            unit="site = session x MNI coordinate, cells averaged",
            n_cells=int(len(W)), n_sites=int(len(S)),
            preregistration="CHANGELOG.md 2026-10-30 C4"), tests=res), f,
            indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(21 * CM, 6.5 * CM), sharex=True)
    for ax, col, lab in zip(axes, ("encode", "execute", "enc_minus_exe"),
                            ("encoding", "execution", "encoding − execution")):
        for roi in ROIS:
            g = S[S.roi == roi]
            ax.scatter(g.z, g[col], s=8 + 4 * g.n_cells, color=ROI_COL[roi],
                       alpha=0.75, lw=0.3, edgecolor="0.2", label=roi)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        key = ("PRIMARY enc - exe vs z (mPFC + mOFC sites)"
               if col == "enc_minus_exe" else f"{col} vs z")
        r = res[key]
        ax.set_title(f"{lab}\nρ = {r['rho']:+.2f}, p = {r['p_perm']:.3f} "
                     f"({r['n_sites']} sites)", fontsize=9)
        ax.set_xlabel("MNI z (mm)\nventral ← → dorsal")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("ripple modulation\n(in − out)/(in + out)")
    axes[0].legend(fontsize=7, frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, -0.42), ncol=2,
                   title="marker size = cells per site", title_fontsize=7)
    fig.text(0.01, -0.22,
             "One marker per recording site (cells sharing a session and MNI "
             "coordinate, averaged). Ripple modulation from lag_group_ripple: "
             "firing inside hippocampal ripples vs outside ±1 s.\nρ: Spearman "
             "across sites; p: 10,000 permutations of site depth.",
             fontsize=7, color="0.3")
    fig.tight_layout()
    for e in (".png", ".pdf"):
        fig.savefig(os.path.join(d, "cell_ripple_depth" + e), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"-> {d}")


if __name__ == "__main__":
    main()
