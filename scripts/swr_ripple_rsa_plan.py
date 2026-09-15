#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is the action plan loaded into the population at the ripples that follow a
reward discovery? -- the cell analogue of the instruction-phase fMRI RSA.

Subjects discover A, B, C, D one at a time, so "how much of the plan is
assembled" is indexed by WHICH reward was just uncovered, not by time on a
clock as in the 12 s fMRI instruction period. Ripple rate rises after those
discoveries, so the population state at those ripples is where a freshly
loaded plan should show up.

For each first discovery of a reward (explore stage, correct), the ripples in
a window around the press are collected, each cell's firing 0-200 ms after
each ripple peak is averaged within (config, state), cells are pooled across
the 28 shared-config sessions, and an 8 x 8 config RDM is compared with models
of what the subject now knows. See `mc.analyse.ripple_rsa` for why the
position-locked fMRI model cannot be used here and why "current location" is
controlled for free.

    pre  [-0.35, 0]     control
    post [+0.15, +0.70] key

PRE-DECLARED PRIMARY: state D, post window, HC and mPFC, `known_set`.
Everything else is a family of secondary tests and is FDR-corrected as one.

    python scripts/swr_ripple_rsa_plan.py run
    python scripts/swr_ripple_rsa_plan.py run --rois=HC,ACC --no_figures

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.ripple_rsa as rrsa
from mc.plotting.cell_results import roi_display, get_roi_colour, OBSERVED_VALUE_COLOR

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

STAMP = datetime.now().strftime("%Y-%m-%d")

# Pre-declared primary test. Named here, before any result exists, so the
# remaining 40-odd cells of the grid cannot be promoted after the fact.
PRIMARY = {"state": "D", "window": "post", "rois": ["HC", "ACC"],
           "model": "known_set"}


# =============================================================================
def _out_dir(data_root=None):
    d = os.path.join(swr_io.group_dir(data_root) if hasattr(swr_io, "group_dir")
                     else os.path.join(swr_io.derivatives_dir(data_root), "group"),
                     "swr", f"ripple_rsa_plan_{STAMP}")
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    return d


def _bundle_dir(data_root=None):
    return os.path.join(swr_io.derivatives_dir(data_root), "group", "swr", "bundle")


# =============================================================================
def run(bundle=None, rois=None, data_root=None, figures=True, out=None):
    rois = rrsa.ROIS if rois is None else (
        rois.split(",") if isinstance(rois, str) else list(rois))
    bundle_dir = bundle or _bundle_dir(data_root)
    out_dir = out or _out_dir(data_root)
    print(f"bundle : {bundle_dir}\noutput : {out_dir}\n")

    # ---- 0. what the task design allows -------------------------------------
    var = rrsa.verify_model_variance()
    print("MODEL VARIANCE (a model with sd = 0 cannot be fitted at that state)")
    print(var.to_string(index=False), "\n")
    var.to_csv(os.path.join(out_dir, "model_variance.csv"), index=False)

    # ---- 1. events ----------------------------------------------------------
    bd = rrsa.load_bundle(bundle_dir)
    events = rrsa.discovery_events(bd)
    print(f"{len(events)} discovery events, {events.session.nunique()} sessions, "
          f"{events.cfg.nunique()} configs")
    print(events.groupby("state").size().to_string(), "\n")
    events.to_csv(os.path.join(out_dir, "discovery_events.csv"), index=False)

    # ---- 2. ripple-triggered patterns --------------------------------------
    print("collecting ripple-triggered patterns (this reads every grid matrix)")
    packs = rrsa.collect_patterns(bd, events, data_root=data_root)
    packs_half = rrsa.collect_patterns(bd, events, data_root=data_root,
                                       split_halves=True, verbose=False)

    cov = []
    for wname, pack in packs.items():
        for roi in rois:
            n_cells = int((pack["roi"] == roi).sum())
            for si, state in enumerate(rrsa.STATES):
                n_rip = int(np.nansum(pack["counts"][:, :, si]))
                cov.append({"window": wname, "roi": roi, "state": state,
                            "n_cells": n_cells, "n_ripples": n_rip})
    cov = pd.DataFrame(cov)
    cov.to_csv(os.path.join(out_dir, "coverage.csv"), index=False)

    # ---- 3. RDMs, models, exact permutation --------------------------------
    rows, rdms, nulls = [], {}, {}
    for wname, pack in packs.items():
        for roi in rois:
            for state in rrsa.STATES:
                rdm, n_used = rrsa.rdm_for(pack, roi, state)
                rdms[(wname, roi, state)] = (rdm, n_used)
                models = rrsa.model_rdms(state)
                for mname, M in models.items():
                    iu = np.triu_indices(rrsa.N_CONFIG, 1)
                    if M[iu].std() < 1e-12:
                        rows.append({"window": wname, "roi": roi, "state": state,
                                     "model": mname, "rho": np.nan, "p": np.nan,
                                     "p_one_sided": np.nan,
                                     "note": "model constant -- not fittable"})
                        continue
                    res = rrsa.fit_model(rdm, M)
                    nulls[(wname, roi, state, mname)] = res.pop("null")
                    rows.append({"window": wname, "roi": roi, "state": state,
                                 "model": mname, **res})
    fits = pd.DataFrame(rows)

    # ---- 4. multiple comparisons -------------------------------------------
    is_primary = ((fits.state == PRIMARY["state"])
                  & (fits.window == PRIMARY["window"])
                  & (fits.model == PRIMARY["model"])
                  & (fits.roi.isin(PRIMARY["rois"])))
    fits["family"] = np.where(is_primary, "primary", "secondary")
    fits["p_fdr"] = np.nan
    for fam in ("primary", "secondary"):
        sel = (fits.family == fam) & fits.p_one_sided.notna()
        if sel.sum():
            fits.loc[sel, "p_fdr"] = _fdr(fits.loc[sel, "p_one_sided"].values)
    fits.to_csv(os.path.join(out_dir, "model_fits.csv"), index=False)

    print("\nPRIMARY (pre-declared: state D, post window, known_set)")
    print(fits[is_primary][["roi", "rho", "p_one_sided", "p", "p_fdr"]]
          .to_string(index=False))

    # ---- 5. the controls the design gives for free -------------------------
    ctrl = []
    for wname in rrsa.PRESS_WINDOWS:
        for roi in rois:
            for state in ("C", "D"):
                rdm, _ = rdms[(wname, roi, state)]
                r = rrsa.rank_offset_test(rdm, state)
                ctrl.append({"test": "rank_offset", "window": wname, "roi": roi,
                             "state": state, **r})
            rel = rrsa.split_half_reliability(packs_half[wname], roi, "D")
            ctrl.append({"test": "split_half_D", "window": wname, "roi": roi,
                         "state": "D", **rel})
    ctrl = pd.DataFrame(ctrl)
    ctrl.to_csv(os.path.join(out_dir, "controls.csv"), index=False)
    print("\nCONTROLS")
    print(ctrl.to_string(index=False))

    jk = []
    for roi in PRIMARY["rois"]:
        if roi not in rois:
            continue
        d = rrsa.jackknife_sessions(packs[PRIMARY["window"]], roi,
                                    PRIMARY["state"],
                                    rrsa.model_rdms(PRIMARY["state"])[PRIMARY["model"]])
        d["roi"] = roi
        jk.append(d)
    if jk:
        jk = pd.concat(jk)
        jk.to_csv(os.path.join(out_dir, "jackknife_primary.csv"), index=False)
        print("\nJACKKNIFE (primary), rho range per ROI")
        print(jk.groupby("roi").rho.agg(["min", "median", "max"]).to_string())

    # ---- 6. settings --------------------------------------------------------
    settings = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": os.path.basename(__file__),
        "bundle": bundle_dir,
        "sessions": rrsa.DSR_SESSIONS,
        "n_sessions": len(rrsa.DSR_SESSIONS),
        "events": "correct & is_discovery & explore stage, 8 shared configs",
        "n_events": int(len(events)),
        "peri_ripple_window_s": list(rrsa.PERI_WIN_S),
        "press_windows_s": {k: list(v) for k, v in rrsa.PRESS_WINDOWS.items()},
        "ripple_dedup_s": rrsa.DEDUP_S,
        "rois": rois,
        "min_cells_per_pair": rrsa.MIN_CELLS_PER_PAIR,
        "cell_centring": "each cell centred across the 8 configs before correlating",
        "permutation": "exhaustive, all 8! = 40320 config relabellings",
        "primary": PRIMARY,
        "seed": 42,
        "notes": [
            "position-locked (fMRI rewDSR Hamming) model is constant at every "
            "state by task counterbalancing and is reported but never fitted",
            "current location is constant within every within-state RDM, so a "
            "within-state effect cannot be a place code",
            "known_set and known_seq correlate r=1.00/0.97/0.92 at B/C/D and "
            "are NOT run as a horse race; rank_offset_test asks the sequence "
            "question instead",
        ],
    }
    with open(os.path.join(out_dir, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)

    # ---- 7. figures ---------------------------------------------------------
    if figures:
        _figure_rdms(rdms, rois, out_dir)
        _figure_fits(fits, nulls, rois, out_dir)
        _figure_loading_curve(fits, rois, out_dir)
        plt.show(block=False)

    print(f"\nwritten to {out_dir}")
    return fits


def _fdr(p):
    """Benjamini-Hochberg."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0, 1)
    return out


# =============================================================================
# FIGURES -- overview only, not publication panels
# =============================================================================

def _figure_rdms(rdms, rois, out_dir):
    for roi in rois:
        fig, axes = plt.subplots(2, 4, figsize=(13, 7))
        for r, wname in enumerate(rrsa.PRESS_WINDOWS):
            for c, state in enumerate(rrsa.STATES):
                ax = axes[r, c]
                rdm, n_used = rdms[(wname, roi, state)]
                im = ax.imshow(rdm, cmap="RdYlBu_r", vmin=np.nanpercentile(rdm, 5),
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
        fig.savefig(os.path.join(out_dir, f"rdms_{roi}.jpeg"), dpi=150)


def _figure_fits(fits, nulls, rois, out_dir):
    models = ["known_set", "known_seq", "full_abcd"]
    fig, axes = plt.subplots(len(models), len(rois),
                             figsize=(3.2 * len(rois), 2.8 * len(models)),
                             squeeze=False, sharey=True)
    x = np.arange(len(rrsa.STATES))
    for mi, model in enumerate(models):
        for ri, roi in enumerate(rois):
            ax = axes[mi][ri]
            colour = get_roi_colour("mPFC" if roi == "ACC" else
                                    "HC_anterior" if roi == "HC" else
                                    "mOFC" if roi == "OFC" else roi)
            for wi, wname in enumerate(rrsa.PRESS_WINDOWS):
                sub = fits[(fits.roi == roi) & (fits.model == model)
                           & (fits.window == wname)].set_index("state")
                y = [sub.rho.get(s, np.nan) for s in rrsa.STATES]
                lo = []
                for s in rrsa.STATES:
                    n = nulls.get((wname, roi, s, model))
                    lo.append(np.nanpercentile(n, 5) if n is not None and len(n)
                              else np.nan)
                ax.plot(x + 0.12 * wi, y, "o-", color=colour,
                        alpha=1.0 if wname == "post" else 0.35,
                        label=f"{wname}", markersize=5)
                ax.plot(x + 0.12 * wi, lo, "_", color="#999999", markersize=9)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([f"uncover {s}" for s in rrsa.STATES], fontsize=8)
            if ri == 0:
                ax.set_ylabel(f"{model}\nSpearman rho", fontsize=9)
            if mi == 0:
                ax.set_title(roi_display(roi), fontsize=11, color=colour)
            if mi == 0 and ri == 0:
                ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Model fit to the ripple-triggered config RDM\n"
                 "(negative rho = configs sharing locations are more similar; "
                 "grey dashes = 5th percentile of the exact null)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "model_fits.jpeg"), dpi=150)


def _figure_loading_curve(fits, rois, out_dir):
    """The knowledge-accumulation prediction: the fixed full-ABCD model cannot
    be represented at A and can at D, so its fit should grow A -> D."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = np.arange(len(rrsa.STATES))
    for roi in rois:
        sub = fits[(fits.roi == roi) & (fits.model == "full_abcd")
                   & (fits.window == "post")].set_index("state")
        colour = get_roi_colour("mPFC" if roi == "ACC" else
                                "HC_anterior" if roi == "HC" else
                                "mOFC" if roi == "OFC" else roi)
        ax.plot(x, [sub.rho.get(s, np.nan) for s in rrsa.STATES], "o-",
                color=colour, label=roi_display(roi), markersize=5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"after\nuncover {s}" for s in rrsa.STATES], fontsize=9)
    ax.set_ylabel("fit to full ABCD model\n(Spearman rho)", fontsize=9)
    ax.set_title("Knowledge accumulation: the whole plan is only\n"
                 "knowable once D is uncovered", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loading_curve.jpeg"), dpi=150)


# =============================================================================
if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
