#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Position-in-sequence index, LEARNING vs EXECUTION, with the published
definition (chapter 5; group/position_index_2026-10-02).

Everything is the published single-cell pipeline in
`scripts/encoding_state_sustained_cv.py`, imported, not re-implemented:
    traces      360 warped bins per loop, 90 per state
    remove      goal progress (within-state cosine basis), then the per-loop
                linear ramp (trial mean preserved)
    model       4 one-hot states + 8 state x sum-coded goal-progress
                interactions + config intercepts, leave-one-configuration-out
    score       r_state = held-out r(y_test, X_state @ beta_state)
    null        10,000 circular shifts of the held-out trace per fold
    ROI test    one-sided Wilcoxon of per-cell r_state > 0, BH-FDR across
                ROIs (plot_position_code_figures.cell_position); Hodges-
                Lehmann location + interval. The permutation-p shift test of
                the Methods text is kept as p_permshift.

The ONLY thing that changes is which loops are averaged per configuration:
    learning           loops of a block BEFORE its first correct loop
    execution          correct loops (correct == 1)  -- the published set;
                       reproduces the published r_state per cell
    execution_matched  the first n correct loops of each configuration,
                       n = that configuration's number of learning loops,
                       so a learning-vs-execution difference cannot come
                       from averaging fewer loops

Paired tests (same cell, same configurations): per ROI Wilcoxon signed-rank
of r_state(execution) - r_state(learning), and of execution_matched -
learning. Region x phase: Mann-Whitney of that per-cell difference, mOFC vs
HC (HC_anterior + HC_mid pooled, as in D4).

    python scripts/position_index_by_phase.py

@author: Svenja Kuchenhoff
"""
import os
import sys
import json
import datetime

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import encoding_state_sustained_cv as enc
from plot_position_code_figures import hodges_lehmann
import mc.analyse.helpers_human_cells as hh
import mc.analyse.cell_selection as cell_selection
from mc.analyse.future_spatial_peaks import phase_residualise

N_PERM = 10000
SEED_TAG = "state_sustained_v4"          # the published per-cell seed tag
VARIANTS = ("learning", "execution", "execution_matched")
ROIS = ["EC", "mOFC", "mPFC", "HC_anterior", "HC_mid", "PCC"]
OUT = os.path.join(enc.DATA_DIR, "group",
                   f"position_index_by_phase_{datetime.date.today()}")


def loop_sets(beh):
    """Row indices per (variant, configuration)."""
    beh = beh.reset_index(drop=True)
    learn = np.zeros(len(beh), bool)
    for _, g in beh.groupby("grid_no"):
        c = g.correct.to_numpy()
        first = np.argmax(c == 1) if (c == 1).any() else len(c)
        learn[g.index[:first]] = True
    out = {v: {} for v in VARIANTS}
    for cfg, g in beh.groupby("config_str"):
        L = g.index[learn[g.index]].to_numpy()
        E = g.index[(g.correct == 1).to_numpy()].to_numpy()
        if len(L) and len(E):
            out["learning"][cfg] = L
            out["execution"][cfg] = E
            out["execution_matched"][cfg] = E[:len(L)]
    return out


def y_per_config(neurons, rows, configs):
    return {n: np.array([np.nanmean(df.iloc[rows[c]].to_numpy(float), axis=0)
                         for c in configs]) for n, df in neurons.items()}


def run():
    os.makedirs(OUT, exist_ok=True)
    roi_table = cell_selection.load_roi_table(
        data_dir=enc.DATA_DIR, table_name=os.path.basename(enc.ROI_TABLE_PATH),
        roi_column=enc.ROI_LABEL_COLUMN)
    rows = {v: [] for v in VARIANTS}
    for sub in enc.parse_subjects("all"):
        data = hh.load_norm_data(enc.DATA_DIR, [sub], res_data=False)
        if f"sub-{sub}" not in data:
            continue
        d = data[f"sub-{sub}"]
        beh = enc.add_config_str(d["beh"])
        meta = cell_selection.attach_roi_to_neuron_labels(
            d["normalised_neurons"].keys(), roi_table=roi_table,
            roi_column=enc.ROI_LABEL_COLUMN).set_index("neuron_id")
        keep = [n for n in d["normalised_neurons"] if n in meta.index
                and isinstance(meta.loc[n, "roi"], str)]
        if not keep:
            continue
        # published order: goal progress first, then the per-loop ramp
        neurons = {n: pd.DataFrame(phase_residualise(
            d["normalised_neurons"][n].reset_index(drop=True).to_numpy(float),
            basis="cosine")) for n in keep}
        neurons = enc.drift_residualise_traces(neurons)
        sets = loop_sets(beh)
        configs = sorted(sets["execution"])
        if len(configs) < 3:
            continue
        n_cfg = len(configs)
        st = np.tile(enc.STATE_IDX_360, n_cfg)
        ph = np.tile(enc.PHASE_IDX_360, n_cfg)
        cf = np.repeat(np.arange(n_cfg), enc.N_BINS_PER_TRIAL)
        for v in VARIANTS:
            Y = y_per_config(neurons, sets[v], configs)
            res = Parallel(n_jobs=-1)(delayed(enc.analyse_one_neuron)(
                n, meta.loc[n, "roi"], Y[n], st, ph, cf, list(range(n_cfg)),
                N_PERM, seed=enc.cell_seed(sub, n, SEED_TAG)) for n in keep)
            for r in res:
                r.update(subject=sub, variant=v,
                         n_loops=int(sum(len(sets[v][c]) for c in configs)))
            rows[v].extend(res)
        print(f"sub-{sub}: {len(keep)} cells, {n_cfg} configs", flush=True)
    keep_cols = ["neuron", "roi", "subject", "variant", "n_loops", "r_state",
                 "p_perm_r_state", "sig_r_state", "pref_state_mode",
                 "null_mean_r_state"]
    df = pd.concat([pd.DataFrame(rows[v])[keep_cols] for v in VARIANTS])
    df.to_csv(os.path.join(OUT, "per_cell.csv"), index=False)
    return df


def summarise(df):
    out = {"levels": {}, "paired": {}, "mOFC_vs_HC": {}}
    for v in VARIANTS:
        g = df[df.variant == v]
        # published test (plot_position_code_figures.cell_position): one-sided
        # Wilcoxon of per-cell r_state > 0, BH across ROIs. The permutation-p
        # shift test (Methods text) is kept as p_permshift.
        p_shift = enc.wilcoxon_perm_p_per_roi(g, "p_perm_r_state")
        ps = np.array([stats.wilcoxon(g[g.roi == r].r_state.dropna(),
                                      alternative="greater").pvalue
                       for r in ROIS])
        q = enc.bh_fdr(ps)
        out["levels"][v] = {}
        for r, p_, q_ in zip(ROIS, ps, q):
            x = g[g.roi == r].r_state.dropna().to_numpy()
            hl, lo, hi = hodges_lehmann(x)
            out["levels"][v][r] = dict(n=int(x.size), hl=hl, lo=lo, hi=hi,
                                       p=float(p_), q=float(q_),
                                       p_permshift=float(p_shift[r]),
                                       frac_sig=float((g[g.roi == r].p_perm_r_state
                                                       < .05).mean()))
    W = df.pivot_table(index=["neuron", "roi"], columns="variant",
                       values="r_state").reset_index()
    for contrast, (a, b) in {"execution - learning": ("execution", "learning"),
                             "execution_matched - learning":
                                 ("execution_matched", "learning")}.items():
        W[contrast] = W[a] - W[b]
        out["paired"][contrast] = {}
        ps = []
        for r in ROIS:
            x = W.loc[W.roi == r, contrast].dropna().to_numpy()
            hl, lo, hi = hodges_lehmann(x)
            p = float(stats.wilcoxon(x).pvalue) if x.size >= 6 else np.nan
            ps.append(p)
            out["paired"][contrast][r] = dict(n=int(x.size), hl=hl, lo=lo,
                                              hi=hi, p=p)
        for r, q in zip(ROIS, enc.bh_fdr(np.array(ps))):
            out["paired"][contrast][r]["q"] = float(q)
        hc = W.loc[W.roi.isin(["HC_anterior", "HC_mid"]), contrast].dropna()
        mo = W.loc[W.roi == "mOFC", contrast].dropna()
        shift = np.median(np.subtract.outer(hc.to_numpy(), mo.to_numpy()))
        out["mOFC_vs_HC"][contrast] = dict(
            n_HC=int(hc.size), n_mOFC=int(mo.size),
            hl_shift_HC_minus_mOFC=float(shift),
            p_mannwhitney=float(stats.mannwhitneyu(hc, mo).pvalue))
    W.to_csv(os.path.join(OUT, "per_cell_wide.csv"), index=False)
    return out


if __name__ == "__main__":
    np.random.seed(42)
    f = os.path.join(OUT, "per_cell.csv")
    df = pd.read_csv(f) if os.path.exists(f) else run()
    res = summarise(df)
    res["settings"] = dict(
        definition="published chapter-5 single-cell index "
                   "(group/position_index_2026-10-02), split by loop set",
        residualisation="goal progress (within-state cosine) then per-loop "
                        "linear ramp, trial mean preserved",
        learning="loops of a block before its first correct loop",
        execution="correct loops (correct == 1)",
        execution_matched="first n correct loops per configuration, n = its "
                          "number of learning loops",
        n_perm=N_PERM, seed_tag=SEED_TAG, rois=ROIS,
        script="scripts/position_index_by_phase.py")
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: res[k] for k in ("paired", "mOFC_vs_HC")}, indent=1))
    for v in VARIANTS:
        print(v, {r: (round(x["hl"], 4), round(x["q"], 4))
                  for r, x in res["levels"][v].items()})
