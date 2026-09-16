#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ripple-triggered single-unit RSA -- the cell analogue of the instruction-phase
fMRI analysis, on RAW spike times.

Subjects discover A, B, C, D one at a time, so "how much of the plan is
assembled" is indexed by WHICH reward was just uncovered rather than by time
within the 12 s fMRI instruction period. For every first discovery of a reward,
the ripples in the second before and the second after the press are collected;
each cell's firing rate INSIDE each ripple (t_peak +- duration/2) is averaged
per (config, state); cells are pooled across the 28 shared-config sessions; and
an 8 x 8 config RDM per state is compared with models of what the subject knows.

Spikes come from `abcd_passed.mat`, ROIs from `neurons_with_ROI_labels.csv`
(`atlas_roi`). See `mc.analyse.ripple_rsa` for why the position-locked fMRI
model cannot be fitted here and why "current location" is controlled for free.

    pre  [-1.0, 0.0]   control
    post [ 0.0, +1.0]  key

PRE-DECLARED PRIMARY: state D, post window, HC_anterior / HC_mid / mPFC,
`known_set`. Everything else is a secondary family, FDR-corrected as one.

    python scripts/swr_ripple_rsa_spikes.py run
    python scripts/swr_ripple_rsa_spikes.py run --rois=HC_anterior,mPFC

@author: Svenja Kuchenhoff
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.ripple_rsa as rrsa
from mc.plotting.cell_results import roi_display, get_roi_colour

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

STAMP = datetime.now().strftime("%Y-%m-%d")

# CLAUDE.md figure rules: these panels are ~5-6 cm wide on an A4 page, so type
# has to be set for that size -- Arial, 11 pt for titles down to 9 pt for tick
# labels -- and lines heavy enough to survive the reduction.
CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "savefig.dpi": 300,
})

# Every ROI with enough cells to form a pattern. Visual is carried as a
# negative control -- it has no reason to represent a reward plan. EC has 3
# cells under `atlas_roi` (51 under neurons_MNI_latest.csv) and is shown so
# that disagreement stays visible rather than being quietly resolved.
ROIS = ["HC_anterior", "HC_mid", "mPFC", "mOFC", "PCC", "EC", "Visual"]

N_PIPELINE_PERM = 100
MODELS = ["known_set", "known_seq", "full_abcd"]
PRIMARY = {"state": "D", "window": "post", "model": "known_set",
           "rois": ["HC_anterior", "HC_mid", "mPFC"]}


def _out(data_root=None):
    d = os.path.join(swr_io.derivatives_dir(data_root), "group", "swr",
                     f"ripple_rsa_spikes_{STAMP}")
    os.makedirs(d, exist_ok=True)
    return d


def run(bundle=None, rois=None, data_root=None, figures=True, out=None):
    rois = ROIS if rois is None else (rois.split(",") if isinstance(rois, str)
                                      else list(rois))
    out_dir = out or _out(data_root)
    swr_io.start_log(out_dir, "swr_ripple_rsa_spikes")
    bundle_dir = bundle or os.path.join(swr_io.derivatives_dir(data_root),
                                        "group", "swr", "bundle")
    print(f"bundle : {bundle_dir}\noutput : {out_dir}\n")

    # ---- 0. what the design allows -----------------------------------------
    var = rrsa.verify_model_variance()
    var.to_csv(os.path.join(out_dir, "model_variance.csv"), index=False)
    print("MODEL VARIANCE (sd = 0 means the model cannot be fitted at that state)")
    print(var.to_string(index=False), "\n")

    bd = rrsa.load_bundle(bundle_dir)
    events = rrsa.discovery_events(bd)
    roi_tab = rrsa.cell_roi_table()
    roi_tab = roi_tab[roi_tab.roi.isin(rois)]
    spikes = rrsa.load_spike_times()
    print(f"{len(events)} discovery events, {len(roi_tab)} cells in {rois}\n")

    # ---- 1. positive control, on raw spikes --------------------------------
    print("POSITIVE CONTROL -- firing inside vs beside the ripple")
    ctrl = rrsa.peri_ripple_control(bd, roi_tab, spikes)
    ctrl.to_csv(os.path.join(out_dir, "positive_control_sessions.csv"),
                index=False)
    ctrl_sum = rrsa.summarise_control(ctrl)
    ctrl_sum.to_csv(os.path.join(out_dir, "positive_control.csv"), index=False)
    print(ctrl_sum.to_string(index=False), "\n")

    # ---- 2. patterns --------------------------------------------------------
    packs, packs_half = {}, {}
    for wname, w in rrsa.PRESS_WINDOWS.items():
        print(f"[{wname}] {w}")
        packs[wname] = rrsa.collect_spike_patterns(bd, events, roi_tab, spikes, w)
        packs_half[wname] = rrsa.collect_spike_patterns(
            bd, events, roi_tab, spikes, w, split_halves=True, verbose=False)

    # ---- 3. RDMs, models, exact permutation --------------------------------
    rows, rdms, nulls, cov = [], {}, {}, []
    iu = np.triu_indices(rrsa.N_CONFIG, 1)
    for wname, pack in packs.items():
        for roi in rois:
            for state in rrsa.STATES:
                rdm, n_used = rrsa.rdm_for(pack, roi, state)
                rdms[(wname, roi, state)] = rdm
                cov.append({"window": wname, "roi": roi, "state": state,
                            "n_cells": int((pack["roi"] == roi).sum()),
                            "n_ripples": int(np.nansum(
                                pack["counts"][:, :, rrsa.STATES.index(state)])),
                            "rdm_pairs_missing": int((~np.isfinite(rdm[iu])).sum()),
                            "median_cells_per_pair": float(np.median(n_used[iu]))})
                for mname in MODELS:
                    M = rrsa.model_rdms(state)[mname]
                    if M[iu].std() < 1e-12:
                        rows.append({"window": wname, "roi": roi, "state": state,
                                     "model": mname, "rho": np.nan,
                                     "p": np.nan, "p_one_sided": np.nan,
                                     "note": "model constant -- not fittable"})
                        continue
                    res = rrsa.fit_model(rdm, M)
                    nulls[(wname, roi, state, mname)] = res.pop("null")
                    rows.append({"window": wname, "roi": roi, "state": state,
                                 "model": mname, **res})
    fits, cov = pd.DataFrame(rows), pd.DataFrame(cov)
    cov.to_csv(os.path.join(out_dir, "coverage.csv"), index=False)
    print("RDM COVERAGE")
    print(cov.to_string(index=False), "\n")

    is_prim = ((fits.state == PRIMARY["state"]) & (fits.window == PRIMARY["window"])
               & (fits.model == PRIMARY["model"]) & fits.roi.isin(PRIMARY["rois"]))
    fits["family"] = np.where(is_prim, "primary", "secondary")
    fits["p_fdr"] = np.nan
    for fam in ("primary", "secondary"):
        sel = (fits.family == fam) & fits.p_one_sided.notna()
        if sel.sum():
            fits.loc[sel, "p_fdr"] = _fdr(fits.loc[sel, "p_one_sided"].values)
    fits.to_csv(os.path.join(out_dir, "model_fits.csv"), index=False)
    print("PRIMARY (state D, post, known_set)")
    print(fits[is_prim][["roi", "rho", "p_one_sided", "p_fdr"]].to_string(index=False))
    print("\nALL FITS (post window)")
    print(fits[(fits.window == "post") & fits.rho.notna()]
          [["roi", "state", "model", "rho", "p_one_sided", "p_fdr"]]
          .to_string(index=False))

    # ---- 4. reliability -----------------------------------------------------
    rel = []
    for wname in rrsa.PRESS_WINDOWS:
        for roi in rois:
            for state in rrsa.STATES:
                r = rrsa.split_half_reliability(packs_half[wname], roi, state)
                rel.append({"window": wname, "roi": roi, "state": state, **r})
    rel = pd.DataFrame(rel)
    rel.to_csv(os.path.join(out_dir, "split_half_reliability.csv"), index=False)
    print("\nSPLIT-HALF RELIABILITY (post window)")
    print(rel[rel.window == "post"].to_string(index=False))

    # ---- 5. full-pipeline permutation --------------------------------------
    print(f"\nPIPELINE PERMUTATION ({N_PIPELINE_PERM} relabellings, whole "
          "estimator re-run)")
    pipe = []
    for roi in rois:
        for state in rrsa.STATES:
            M = rrsa.model_rdms(state)[PRIMARY["model"]]
            if M[np.triu_indices(rrsa.N_CONFIG, 1)].std() < 1e-12:
                continue
            obs = fits[(fits.roi == roi) & (fits.state == state)
                       & (fits.window == "post")
                       & (fits.model == PRIMARY["model"])].rho
            if obs.empty or not np.isfinite(obs.iloc[0]):
                continue
            null = rrsa.pipeline_null(bd, events, roi_tab, spikes,
                                      rrsa.PRESS_WINDOWS["post"], roi, state,
                                      PRIMARY["model"], n_perm=N_PIPELINE_PERM)
            null = null[np.isfinite(null)]
            o = float(obs.iloc[0])
            pipe.append({"roi": roi, "state": state, "model": PRIMARY["model"],
                         "rho": o, "n_perm": len(null),
                         "null_mean": float(np.mean(null)) if len(null) else np.nan,
                         "null_sd": float(np.std(null)) if len(null) else np.nan,
                         "null_p5": float(np.percentile(null, 5)) if len(null) else np.nan,
                         # upper tail: positive rho is the predicted direction
                         "p_pipeline": (float((null >= o).mean())
                                        if len(null) else np.nan),
                         "z": ((o - np.mean(null)) / np.std(null)
                               if len(null) and np.std(null) > 0 else np.nan)})
            print(f"  {roi:12s} {state}: rho={o:+.3f}  null {np.mean(null):+.3f}"
                  f" +- {np.std(null):.3f}  z={pipe[-1]['z']:+.2f}"
                  f"  p={pipe[-1]['p_pipeline']:.3f}")
    pipe = pd.DataFrame(pipe)
    pipe.to_csv(os.path.join(out_dir, "pipeline_permutation.csv"), index=False)

    swr_io.write_settings(out_dir, {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": os.path.basename(__file__),
        "spike_source": "abcd_passed.mat raw spikeTimes",
        "roi_source": f"{rrsa.ROI_TABLE} column '{rrsa.ROI_COLUMN}'",
        "ripple_extent": "t_peak +- duration_s/2",
        "press_windows_s": {k: list(v) for k, v in rrsa.PRESS_WINDOWS.items()},
        "silent_cells_dropped": {k: int(v["n_silent_dropped"])
                                 for k, v in packs.items()},
        "zero_counts": "kept -- only cells silent in EVERY ripple are dropped",
        "cell_centring": "each cell centred across the 8 configs",
        "permutation": "exhaustive, all 8! = 40320 config relabellings",
        "primary": PRIMARY, "rois": rois, "seed": 42,
        "partial_rdms": "fitted on the observed pairs; the model, not the "
                        "data, is permuted so the observed-pair mask is fixed",
        "min_pairs_to_fit": rrsa.MIN_PAIRS_TO_FIT,
        "pipeline_permutation": f"{N_PIPELINE_PERM} per-session config "
                                "relabellings, whole estimator re-run",
    })

    if figures:
        _figure_model_rdms(out_dir)
        _figure_data_rdms(rdms, rois, out_dir)
        _figure_timecourse(fits, nulls, rel, rois, out_dir)
        _figure_overview(fits, cov, pipe, rel, rois, out_dir)
        plt.show(block=False)
    print(f"\nwritten to {out_dir}")


def _fdr(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0, 1)
    return out


# =============================================================================
# FIGURES
# =============================================================================

def _roi_colour(roi):
    return get_roi_colour(roi)


def _figure_model_rdms(out_dir):
    """Every model RDM, at every state, with its variance printed on it."""
    iu = np.triu_indices(rrsa.N_CONFIG, 1)
    names = MODELS + ["position_locked"]
    fig, axes = plt.subplots(len(names), 4, figsize=(13.5, 13),
                             squeeze=False)
    for r, mname in enumerate(names):
        for c, state in enumerate(rrsa.STATES):
            ax = axes[r][c]
            M = rrsa.model_rdms(state)[mname]
            sd = M[iu].std()
            im = ax.imshow(M, cmap="Greys", vmin=0, vmax=max(M.max(), 1e-9))
            ax.set_xticks(range(rrsa.N_CONFIG))
            ax.set_yticks(range(rrsa.N_CONFIG))
            ax.set_xticklabels(rrsa.CONFIG_LABELS, rotation=90, fontsize=6)
            ax.set_yticklabels(rrsa.CONFIG_LABELS, fontsize=6)
            dead = sd < 1e-12
            ax.set_title(f"uncover {state}   sd = {sd:.3f}"
                         + ("\nCONSTANT — not fittable" if dead else ""),
                         fontsize=9, color="#b03030" if dead else "k")
            if dead:
                for sp in ax.spines.values():
                    sp.set_color("#b03030")
                    sp.set_linewidth(2)
            if c == 0:
                ax.set_ylabel(mname, fontsize=11, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Model RDMs (dissimilarity between the 8 reward configurations)\n"
                 "knowledge-gated models use only the locations uncovered so "
                 "far; full_abcd is the same at every state", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "model_RDMs.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "model_RDMs.pdf"), bbox_inches="tight")


def _figure_data_rdms(rdms, rois, out_dir):
    for roi in rois:
        fig, axes = plt.subplots(2, 4, figsize=(13, 7))
        for r, wname in enumerate(rrsa.PRESS_WINDOWS):
            for c, state in enumerate(rrsa.STATES):
                ax = axes[r][c]
                rdm = rdms[(wname, roi, state)]
                im = ax.imshow(rdm, cmap="RdYlBu_r",
                               vmin=np.nanpercentile(rdm, 5),
                               vmax=np.nanpercentile(rdm, 95))
                ax.set_title(f"{wname} · uncover {state}", fontsize=10)
                ax.set_xticks(range(rrsa.N_CONFIG))
                ax.set_yticks(range(rrsa.N_CONFIG))
                ax.set_xticklabels(rrsa.CONFIG_LABELS, rotation=90, fontsize=6)
                ax.set_yticklabels(rrsa.CONFIG_LABELS, fontsize=6)
                fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{roi_display(roi)} — ripple-triggered config RDMs "
                     f"(correlation distance)", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"data_RDMs_{roi}.jpeg"), dpi=150)


def _figure_timecourse(fits, nulls, rel, rois, out_dir):
    """Model fit across A -> B -> C -> D, per model, per ROI, with the
    pre-press window as an explicit contrast.

    Panels are ~5 cm wide on the page, so type is set for that size.
    """
    x = np.arange(len(rrsa.STATES))
    ncol = 3
    fig, axes = plt.subplots(len(MODELS), ncol,
                             figsize=(3 * 6.2 * CM, len(MODELS) * 5.4 * CM),
                             squeeze=False, sharex=True)
    col = {"post": 0, "pre": 1}
    for r, model in enumerate(MODELS):
        rows = {}
        for wname in ("post", "pre"):
            ax = axes[r][col[wname]]
            for roi in rois:
                sub = (fits[(fits.roi == roi) & (fits.model == model)
                            & (fits.window == wname)].set_index("state"))
                y = np.array([sub.rho.get(s, np.nan) for s in rrsa.STATES],
                             dtype=float)
                rows[(wname, roi)] = y
                if not np.isfinite(y).any():
                    continue
                ax.plot(x, y, "o-", color=_roi_colour(roi),
                        label=roi_display(roi), lw=2.2, ms=6)
                for k, st in enumerate(rrsa.STATES):
                    p = sub.p_one_sided.get(st, np.nan)
                    if np.isfinite(p) and p < 0.05:
                        ax.annotate("*", (x[k], y[k]), textcoords="offset points",
                                    xytext=(0, 8), ha="center", fontsize=13,
                                    color=_roi_colour(roi))
            ax.axhline(0, color="k", lw=1.0)
            ax.set_title(f"{wname}-press", pad=4)
            if col[wname] == 0:
                ax.set_ylabel(f"{model}\nSpearman $\\rho$")
        # contrast column
        ax = axes[r][2]
        for roi in rois:
            d = rows.get(("post", roi))
            e = rows.get(("pre", roi))
            if d is None or e is None or not np.isfinite(d - e).any():
                continue
            ax.plot(x, d - e, "o-", color=_roi_colour(roi), lw=2.2, ms=6)
        ax.axhline(0, color="k", lw=1.0)
        ax.set_title("post − pre", pad=4)
        for c in range(ncol):
            axes[r][c].set_xticks(x)
            axes[r][c].set_xticklabels(list(rrsa.STATES))
            axes[r][c].tick_params(length=3)
            if model in ("known_set", "known_seq"):
                axes[r][c].axvspan(-0.45, 0.45, color="#b03030", alpha=0.08,
                                   lw=0)
    axes[-1][1].set_xlabel("reward uncovered")
    handles = [Line2D([], [], color=_roi_colour(r), lw=2.2, marker="o", ms=5,
                      label=roi_display(r)) for r in rois]
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=len(rois),
               loc="lower center", bbox_to_anchor=(0.5, -0.01),
               handlelength=1.4, columnspacing=1.2)
    fig.suptitle("Model fit across the four reward discoveries\n"
                 "positive $\\rho$ = predicted (configurations sharing locations "
                 "have more similar patterns).  Shaded = model constant at A.  "
                 "* p < 0.05 uncorrected", fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 0.91])
    fig.savefig(os.path.join(out_dir, "fit_timecourse_A_to_D.png"),
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fit_timecourse_A_to_D.pdf"),
                bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7.5 * CM, 5.6 * CM))
    for roi in rois:
        g = rel[(rel.roi == roi) & (rel.window == "post")].set_index("state")
        y = [g.rho.get(s, np.nan) for s in rrsa.STATES]
        if not np.isfinite(np.array(y, dtype=float)).any():
            continue
        ax.plot(x, y, "o-", color=_roi_colour(roi), label=roi_display(roi),
                lw=2.2, ms=6)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(list(rrsa.STATES))
    ax.set_xlabel("reward uncovered")
    ax.set_ylabel("split-half reliability\n(Spearman $\\rho$)")
    ax.set_title("Is the RDM signal at all?", pad=4)
    ax.legend(frameon=False, fontsize=8, ncol=2, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rdm_reliability.png"),
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "rdm_reliability.pdf"),
                bbox_inches="tight")


def _figure_overview(fits, cov, pipe, rel, rois, out_dir):
    """One page: every ROI, the effect against its own re-estimated null, how
    much of each RDM was observed, and whether the RDM is reliable."""
    fig, axes = plt.subplots(2, 2, figsize=(2 * 8.0 * CM, 2 * 6.4 * CM))

    # a. primary model fit, all ROIs, post vs pre
    ax = axes[0][0]
    x = np.arange(len(rrsa.STATES))
    for roi in rois:
        for wname, style in (("post", "o-"), ("pre", "o--")):
            sub = (fits[(fits.roi == roi) & (fits.model == PRIMARY["model"])
                        & (fits.window == wname)].set_index("state"))
            y = np.array([sub.rho.get(s, np.nan) for s in rrsa.STATES],
                         dtype=float)
            if not np.isfinite(y).any():
                continue
            ax.plot(x, y, style, color=_roi_colour(roi), lw=2.2, ms=5,
                    alpha=1.0 if wname == "post" else 0.5)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(list(rrsa.STATES))
    ax.set_xlabel("reward uncovered")
    ax.set_ylabel("known_set fit\n(Spearman $\\rho$)")
    ax.set_title("a  Plan model, all ROIs", pad=4)
    ax.legend(handles=[Line2D([], [], color="#555", lw=2.2, label="post"),
                       Line2D([], [], color="#555", lw=2.2, ls="--", alpha=0.5,
                              label="pre")],
              frameon=False, fontsize=8, handlelength=1.6, loc="upper left")
    handles = [Line2D([], [], color=_roi_colour(r), lw=2.2, marker="o", ms=5,
                      label=roi_display(r)) for r in rois]
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=len(rois),
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               handlelength=1.4, columnspacing=1.0)

    # b. observed effect against the re-estimated null
    ax = axes[0][1]
    if len(pipe):
        pos = {r: i for i, r in enumerate(rois)}
        for _, row in pipe.iterrows():
            i = pos[row.roi] + 0.22 * (rrsa.STATES.index(row.state) - 1.5)
            ax.errorbar(i, row.null_mean, yerr=row.null_sd, fmt="_",
                        color="#999999", lw=2.0, capsize=2, ms=10)
            ax.plot(i, row.rho, "o", color=_roi_colour(row.roi), ms=6)
        ax.axhline(0, color="k", lw=1.0)
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels([roi_display(r) for r in rois], rotation=35,
                           ha="right")
    ax.set_ylabel("known_set fit\n(Spearman $\\rho$)")
    ax.set_title(f"b  vs re-estimated null ({N_PIPELINE_PERM} perms)", pad=4)

    # c. how much of each RDM was actually observed
    ax = axes[1][0]
    sub = cov[cov.window == "post"]
    for i, roi in enumerate(rois):
        g = sub[sub.roi == roi]
        if g.empty:
            continue
        pct = 100 * (28 - g.rdm_pairs_missing) / 28
        ax.bar(i, pct.mean(), 0.66, yerr=pct.std(), color=_roi_colour(roi),
               edgecolor="k", linewidth=0.8, capsize=2)
    ax.axhline(100 * rrsa.MIN_PAIRS_TO_FIT / 28, color="#b03030", ls="--",
               lw=1.6)
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels([roi_display(r) for r in rois], rotation=35, ha="right")
    ax.set_ylabel("config pairs observed (%)")
    ax.set_ylim(0, 105)
    ax.set_title("c  How complete is each RDM", pad=4)

    # d. reliability
    ax = axes[1][1]
    for i, roi in enumerate(rois):
        g = rel[(rel.roi == roi) & (rel.window == "post")
                & (rel.n_pairs >= rrsa.MIN_PAIRS_TO_FIT)]
        v = g.rho.astype(float).dropna()
        if v.empty:
            continue
        ax.bar(i, v.mean(), 0.66, yerr=v.sem() if len(v) > 1 else None,
               color=_roi_colour(roi), edgecolor="k", linewidth=0.8, capsize=2)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels([roi_display(r) for r in rois], rotation=35, ha="right")
    ax.set_ylabel("split-half reliability\n(mean over states)")
    ax.set_title(f"d  Is any of it signal?\n"
                 f"(states with $\\geq${rrsa.MIN_PAIRS_TO_FIT} pairs only)",
                 fontsize=10, pad=4)

    fig.suptitle("Ripple-triggered plan coding: overview across all ROIs",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(os.path.join(out_dir, "overview_all_ROIs.png"),
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "overview_all_ROIs.pdf"),
                bbox_inches="tight")


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
