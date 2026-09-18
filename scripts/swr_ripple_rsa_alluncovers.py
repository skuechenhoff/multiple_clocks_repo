#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The ripple RSA on ALL correct uncovers, not just explore-phase discoveries.

Two condition spaces, from one cache and one set of surrogate draws:

  collapsed32   config x state. All As together, all Bs together, and so on,
                pooling first traversals with repeats. 448 pairs.
  phase64       config x state x phase (first vs repeat). 1824 pairs, chance
                SD 0.0234. Its `first_only` family is exactly the 448-pair
                discovery analysis, so the two nest and can be compared
                directly.

Scale: 32688 uncovers and ~12600 ripples, 10.9x the discovery-only set, and
~154 ripples per 64-condition cell instead of ~37 per 32-condition cell.

WHY PHASE IS A FACTOR AND NOT POOLED AWAY: on a repeat the subject already
knows the whole configuration, so `known_set` collapses onto `full_abcd` for
every repeat condition. The two models therefore differ ONLY in the `first`
half. Pooling phase would quietly dilute exactly the contrast the analysis is
about, so phase is partialled out of every fit in both spaces and kept as its
own factor in phase64.

TWO ESTIMATORS, adjudicated by reliability rather than by preference:
  corr_zscore  1 - Pearson after per-feature z-scoring (project convention)
  crossnobis   cross-validated Mahalanobis, per session, shrinkage 0.1
Crossnobis was NOT estimable on the discovery-only 32-condition space (median
1 ripple per session x condition). At this scale it is, so the two can finally
be compared where both are well posed. Split-half reliability of the data RDM
decides which one is extracting reproducible structure at all -- a fit from an
estimator whose own RDM does not replicate across halves of the ripples means
nothing, whatever its p-value.

    python scripts/swr_ripple_rsa_alluncovers.py run
    python scripts/swr_ripple_rsa_alluncovers.py run --modality=hfb --n_surr=300
    python scripts/swr_ripple_rsa_alluncovers.py reliability

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.ripple_rsa as rrsa
from mc.plotting.cell_results import roi_display, get_roi_colour

try:
    import fire
except ImportError:
    fire = None

STAMP = datetime.now().strftime("%Y-%m-%d")
SEED = 42
CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 9, "legend.fontsize": 8,
    "axes.linewidth": 1.0, "lines.linewidth": 2.2, "savefig.dpi": 300,
})

SPIKE_ROIS = ["HC_anterior", "HC_mid", "mPFC", "mOFC", "PCC"]
HFB_ROIS = ["HC_anterior", "HC_mid", "mPFC", "mOFC", "Visual"]
MODELS = ["current_location", "known_set", "full_abcd"]
MODEL_COLOURS = {"current_location": "#5b9b8d", "known_set": "#23677E",
                 "full_abcd": "#DC673E"}
# default estimators. "corr_centre" is available but NOT default: it is the
# pre-2026-09-17 normalisation, kept so the centre-vs-zscore sensitivity that
# motivated adopting z-scoring can be reproduced with
#   --estimators=corr_centre,corr_zscore,crossnobis
ALL_EST = ["corr_centre", "corr_zscore", "crossnobis"]
EST = ["corr_zscore", "crossnobis"]
FAM32 = ["all", "within_state", "cross_state"]
FAM64 = ["all", "within_phase", "first_only", "repeat_only", "cross_phase"]


def _out(data_root=None, name=None):
    d = os.path.join(swr_io.derivatives_dir(data_root), "group", "swr",
                     name or f"ripple_rsa_alluncovers_{STAMP}")
    os.makedirs(d, exist_ok=True)
    return d


def _build(modality, bd, ev, tab, spikes, bundle_dir, rng=None):
    D = rrsa.DECISIONS
    if modality == "spikes":
        return rrsa.cache_ripple_rates(bd, ev, tab, spikes, D["window_s"],
                                       extent=D["extent"], scheme="uncovers",
                                       surrogate_rng=rng)
    return rrsa.cache_hfb_rates(bd, ev, tab, D["window_s"], scheme="uncovers",
                                extent=D["extent"], bundle_dir=bundle_dir,
                                surrogate_rng=rng, verbose=False)


def _rdms(cache, rois, est=None):
    """Every data RDM this analysis needs, from one cache. The whole estimator,
    so a surrogate draw calls exactly this."""
    est = est or EST
    p32 = rrsa.patterns_uncover(cache, collapse_phase=True)
    p64 = rrsa.patterns_uncover(cache, collapse_phase=False)
    out = {}
    for roi in rois:
        for e in est:
            if e == "crossnobis":
                out[(roi, "collapsed32", e)] = rrsa.crossnobis_rdm(
                    cache, roi, conditions="config_state")[0]
                out[(roi, "phase64", e)] = _crossnobis64(cache, roi)
            else:
                norm = "centre" if e == "corr_centre" else "zscore"
                out[(roi, "collapsed32", e)] = rrsa.rdm_for_flat(
                    p32, roi, normalise=norm)[0]
                out[(roi, "phase64", e)] = rrsa.rdm_for_64(
                    p64, roi, normalise=norm)[0]
    return out, p32, p64


def _crossnobis64(cache, roi, n_folds=2):
    """Crossnobis over the 64 conditions -- same recipe, phase in the index."""
    mats = []
    for blk in cache:
        sel = np.asarray(blk["roi"]) == roi
        if sel.sum() < 2:
            continue
        rates = np.asarray(blk["rates"], float)[sel]
        cond = (blk["pi"] * rrsa.N_CONFIG * rrsa.N_STATE
                + blk["si"] * rrsa.N_CONFIG + blk["ci"])
        X, R = rrsa._fold_means(rates, cond, rrsa.N_COND64, n_folds)
        if R.shape[0] < 2 or np.isfinite(X).all(axis=(1, 2)).sum() < 2:
            continue
        S = np.atleast_2d(np.cov(R, rowvar=False))
        S = (1 - rrsa.CROSSNOBIS_SHRINKAGE) * S \
            + rrsa.CROSSNOBIS_SHRINKAGE * np.eye(S.shape[0])
        rdm, _ = rrsa._crossnobis_from_X(X, np.linalg.pinv(S))
        if np.isfinite(rdm).any():
            mats.append(rdm)
    if not mats:
        return np.full((rrsa.N_COND64, rrsa.N_COND64), np.nan)
    M = np.stack(mats)
    with np.errstate(invalid="ignore"):
        rdm = np.nanmean(M, axis=0)
    rdm[np.isfinite(M).sum(axis=0) == 0] = np.nan
    np.fill_diagonal(rdm, 0.0)
    return rdm


def _fit_all(rdms, rois, est=None):
    """rho for every (roi, space, estimator, model, family)."""
    M32, M64 = rrsa.model_rdms_flat(), rrsa.model_rdms_64()
    est = est or EST
    out = {}
    for roi in rois:
        for e in est:
            r32 = rdms[(roi, "collapsed32", e)]
            for name, M in M32.items():
                for fam in FAM32:
                    out[(roi, "collapsed32", e, name, fam)] = \
                        rrsa.fit_rho_flat(r32, M, family=fam)
            r64 = rdms[(roi, "phase64", e)]
            for name, M in M64.items():
                for fam in FAM64:
                    out[(roi, "phase64", e, name, fam)] = \
                        rrsa.fit_rho_64(r64, M, family=fam)
    return out


def run(bundle=None, data_root=None, modality="spikes", n_surr=300, out=None,
        estimators=None):
    D = rrsa.DECISIONS
    out_dir = out or _out(data_root)
    swr_io.start_log(out_dir, "swr_ripple_rsa_alluncovers")
    bundle_dir = bundle or rrsa.default_bundle_dir(data_root)
    np.random.seed(SEED)

    est = (EST if estimators is None else
           (estimators.split(",") if isinstance(estimators, str)
            else list(estimators)))
    bad = [e for e in est if e not in ALL_EST]
    if bad:
        raise ValueError(f"unknown estimator(s) {bad}; choose from {ALL_EST}")
    bd = rrsa.load_bundle(bundle_dir, pad_s=D["pad_s"])
    ev = rrsa.all_uncover_events(bd)
    spikes = rrsa.load_spike_times(verbose=False)
    if modality == "spikes":
        tab = rrsa.cell_roi_table()
        rois = [r for r in SPIKE_ROIS if (tab.roi == r).sum() >= 5]
    else:
        tab = rrsa.hfb_roi_table(bundle_dir)
        rois = [r for r in HFB_ROIS if (tab.roi == r).sum() >= 5]
    tab = tab[tab.roi.isin(rois)]
    print(f"{modality}: {len(ev)} uncovers "
          f"({ev.phase.value_counts().to_dict()}), {len(tab)} features")

    cache = _build(modality, bd, ev, tab, spikes, bundle_dir)
    n_rip = sum(b["rates"].shape[1] for b in cache)
    feas = rrsa.crossnobis_feasibility(cache, rois)
    feas.to_csv(os.path.join(out_dir, f"feasibility_{modality}.csv"),
                index=False)
    print(f"  {n_rip} ripples")
    print(feas[feas.conditions == "config_state"].round(1).to_string(index=False))

    rdms, p32, p64 = _rdms(cache, rois, est)
    obs = _fit_all(rdms, rois, est)

    rng = np.random.default_rng(SEED)
    null = {k: [] for k in obs}
    for i in range(n_surr):
        c = _build(modality, bd, ev, tab, spikes, bundle_dir, rng=rng)
        f = _fit_all(_rdms(c, rois, est)[0], rois, est)
        for k, v in f.items():
            null[k].append(v)
        if (i + 1) % 25 == 0:
            print(f"    surrogate {i + 1}/{n_surr}", flush=True)

    rows = []
    for k, o in obs.items():
        roi, space, est, model, fam = k
        s = np.array(null[k], float)
        s = s[np.isfinite(s)]
        rows.append({
            "modality": modality, "roi": roi, "space": space, "estimator": est,
            "model": model, "family": fam, "rho": o,
            "surr_mean": s.mean() if len(s) else np.nan,
            "surr_sd": s.std() if len(s) else np.nan,
            "z_surr": ((o - s.mean()) / s.std()
                       if len(s) and s.std() > 0 and np.isfinite(o) else np.nan),
            "p_surr": (float((s >= o - 1e-12).mean())
                       if len(s) and np.isfinite(o) else np.nan),
            "n_surr": len(s)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, f"alluncovers_{modality}.csv"),
               index=False)
    np.savez_compressed(
        os.path.join(out_dir, f"nulls_{modality}.npz"),
        keys=np.array(["|".join(k) for k in obs]),
        null=np.array([np.asarray(null[k], float) for k in obs]))

    _report(res, modality)
    _figure(res, rois, modality, out_dir)
    _overview(res, feas, rois, modality, out_dir, n_surr, n_rip, len(ev))
    swr_io.write_settings(out_dir, {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": os.path.basename(__file__), "bundle": bundle_dir,
        "modality": modality, "events": "all correct uncovers, 8 shared configs",
        "n_events": int(len(ev)), "n_ripples": int(n_rip),
        "scheme": "uncovers (clipped at the next uncover of the session)",
        "spaces": {"collapsed32": "config x state, phase pooled, 448 pairs",
                   "phase64": "config x state x phase, 1824 pairs"},
        "estimators": est, "n_surr": n_surr, "seed": SEED,
        "decisions": {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in D.items()}})
    print(f"\nwritten to {out_dir}")


def _report(res, modality):
    for space, fam in (("collapsed32", "all"), ("phase64", "all"),
                       ("phase64", "first_only"), ("phase64", "repeat_only")):
        t = res[(res.space == space) & (res.family == fam) & res.rho.notna()]
        if t.empty:
            continue
        print(f"\n##### {modality} | {space} | family {fam} #####")
        print(t.pivot_table(index=["roi", "model"], columns="estimator",
                            values=["rho", "z_surr"]).round(3).to_string())


def _figure(res, rois, modality, out_dir):
    for space, fams in (("collapsed32", ["all"]),
                        ("phase64", ["all", "first_only", "repeat_only"])):
        t = res[res.space == space]
        n_est = max(1, t.estimator.nunique())
        fig, axes = plt.subplots(
            len(fams) * n_est, len(rois),
            figsize=(len(rois) * 3.9 * CM, len(fams) * n_est * 3.9 * CM),
            squeeze=False, sharey="row", sharex=True)
        r = 0
        ests = [e for e in ALL_EST if (t.estimator == e).any()]
        for fam in fams:
            for est in ests:
                for c, roi in enumerate(rois):
                    ax = axes[r][c]
                    for i, m in enumerate(MODELS):
                        g = t[(t.roi == roi) & (t.estimator == est)
                              & (t.model == m) & (t.family == fam)]
                        if not len(g) or not np.isfinite(g.rho.iloc[0]):
                            continue
                        sd = g.surr_sd.iloc[0]
                        mu = g.surr_mean.iloc[0]
                        ax.add_patch(plt.Rectangle(
                            (i - 0.38, mu - sd), 0.76, 2 * sd,
                            facecolor="#d9d9d9", edgecolor="none", zorder=0))
                        ax.plot([i - 0.38, i + 0.38], [g.rho.iloc[0]] * 2,
                                color=MODEL_COLOURS[m], lw=2.6,
                                solid_capstyle="butt", zorder=5)
                        if g.p_surr.iloc[0] < 0.05:
                            ax.annotate("*", (i, g.rho.iloc[0]), ha="center",
                                        va="bottom", fontsize=13, zorder=6,
                                        textcoords="offset points",
                                        xytext=(0, 2), color=MODEL_COLOURS[m])
                    ax.axhline(0, color="k", lw=0.9)
                    ax.set_xticks(range(len(MODELS)))
                    ax.set_xticklabels(["curr\nloc", "known\nset", "full\nABCD"],
                                       fontsize=7.5)
                    ax.set_xlim(-0.6, len(MODELS) - 0.4)
                    if r == 0:
                        ax.set_title(roi_display(roi), fontsize=10,
                                     color=get_roi_colour(roi))
                    if c == 0:
                        ax.set_ylabel(f"{est.replace('_', ' ')}\n"
                                      f"{fam}", fontsize=8)
                r += 1
        fig.suptitle(f"All correct uncovers — {modality}, {space}\n"
                     f"grey = ±1 SD of the surrogate-window null   ·   "
                     f"* p < 0.05 uncorrected", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        for e in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir,
                                     f"alluncovers_{modality}_{space}.{e}"),
                        bbox_inches="tight")
        plt.close(fig)


def _overview(res, feas, rois, modality, out_dir, n_surr, n_rip, n_ev):
    top = {}
    for space in ("collapsed32", "phase64"):
        for est in sorted(res.estimator.unique()):
            t = res[(res.space == space) & (res.estimator == est)
                    & (res.family == "all") & res.rho.notna()]
            if t.empty:
                continue
            t = t.sort_values("z_surr", ascending=False)
            top[f"{space}|{est}"] = {
                "strongest": [{"roi": r.roi, "model": r.model,
                               "rho": round(r.rho, 3),
                               "z": round(r.z_surr, 2),
                               "p": round(r.p_surr, 4)}
                              for _, r in t.head(3).iterrows()],
                "n_sig_p05_uncorrected": int((t.p_surr < 0.05).sum()),
                "n_tests": int(len(t))}
    with open(os.path.join(out_dir, f"results_overview_{modality}.json"),
              "w") as f:
        json.dump({
            "created": datetime.now().isoformat(timespec="seconds"),
            "modality": modality, "n_uncovers": int(n_ev),
            "n_ripples": int(n_rip),
            "vs_discovery_only": "10.9x more ripples than the explore-only set",
            "spaces": {"collapsed32": {"pairs": 448, "chance_sd": 0.0473},
                       "phase64": {"pairs": 1824, "chance_sd": 0.0234}},
            "estimators": sorted(res.estimator.unique()), "n_surr": n_surr,
            "crossnobis_now_estimable": "yes on 32 conditions at this scale; "
                                        "it was not on discoveries only",
            "feasibility": feas.to_dict(orient="records"),
            "results": top}, f, indent=2)
    print("\n--- overview ---")
    print(json.dumps(top, indent=2))


def reliability(modality="spikes", bundle=None, data_root=None, n_splits=20,
                out=None):
    """Split-half reliability of the data RDM, per estimator.

    THE ADJUDICATION. Split the ripples of every condition into two halves,
    build the RDM on each half independently, and correlate the two. This asks
    nothing about any model: it asks whether the estimator recovers the SAME
    representational geometry from two independent halves of the same data.

    An estimator whose own RDM does not replicate cannot support a model fit,
    whatever p-value that fit carries -- so this, not the size of rho, is what
    decides which estimator to believe.
    """
    D = rrsa.DECISIONS
    out_dir = out or _out(data_root)
    swr_io.start_log(out_dir, "swr_ripple_rsa_alluncovers_reliability")
    bundle_dir = bundle or rrsa.default_bundle_dir(data_root)
    bd = rrsa.load_bundle(bundle_dir, pad_s=D["pad_s"])
    ev = rrsa.all_uncover_events(bd)
    spikes = rrsa.load_spike_times(verbose=False)
    if modality == "spikes":
        tab = rrsa.cell_roi_table()
        rois = [r for r in SPIKE_ROIS if (tab.roi == r).sum() >= 5]
    else:
        tab = rrsa.hfb_roi_table(bundle_dir)
        rois = [r for r in HFB_ROIS if (tab.roi == r).sum() >= 5]
    tab = tab[tab.roi.isin(rois)]
    cache = _build(modality, bd, ev, tab, spikes, bundle_dir)

    rng = np.random.default_rng(SEED)
    iu = np.triu_indices(rrsa.N_COND, 1)
    rows = []
    for rep in range(n_splits):
        halves = []
        for h in (0, 1):
            sub = []
            for blk in cache:
                n = blk["rates"].shape[1]
                if rep == 0 and h == 0:
                    blk["_perm"] = rng.permutation(n)
                elif h == 0:
                    blk["_perm"] = rng.permutation(n)
                idx = blk["_perm"][h::2]
                b = {k: v for k, v in blk.items() if k not in
                     ("rates", "ci", "si", "pi", "_perm")}
                b["rates"] = blk["rates"][:, idx]
                b["ci"], b["si"] = blk["ci"][idx], blk["si"][idx]
                if "pi" in blk:
                    b["pi"] = blk["pi"][idx]
                sub.append(b)
            halves.append(sub)
        for roi in rois:
            for est in EST:
                v = []
                for sub in halves:
                    if est == "crossnobis":
                        r = rrsa.crossnobis_rdm(sub, roi,
                                                conditions="config_state")[0]
                    else:
                        p = rrsa.patterns_uncover(sub, collapse_phase=True)
                        r = rrsa.rdm_for_flat(p, roi, normalise="zscore")[0]
                    v.append(r[iu])
                a, b = v
                ok = np.isfinite(a) & np.isfinite(b)
                rows.append({"modality": modality, "roi": roi,
                             "estimator": est, "split": rep,
                             "reliability": (
                                 float(stats.spearmanr(a[ok], b[ok]).correlation)
                                 if ok.sum() > 10 else np.nan),
                             "n_pairs": int(ok.sum())})
        print(f"  split {rep + 1}/{n_splits}", flush=True)
    rel = pd.DataFrame(rows)
    rel.to_csv(os.path.join(out_dir, f"reliability_{modality}.csv"),
               index=False)
    g = rel.groupby(["roi", "estimator"]).reliability.agg(["mean", "std"])
    print(f"\n=== split-half reliability of the DATA RDM ({modality}, "
          f"{n_splits} splits, collapsed32) ===")
    print(g.round(3).to_string())
    _fig_reliability(rel, rois, modality, out_dir)
    print(f"\nwritten to {out_dir}")


def _fig_reliability(rel, rois, modality, out_dir):
    fig, ax = plt.subplots(figsize=(2.0 * len(rois) * CM + 2 * CM, 6.2 * CM))
    x = np.arange(len(rois))
    w = 0.34
    cols = {"corr_zscore": "#23677E", "crossnobis": "#DC673E"}
    for i, est in enumerate(EST):
        g = rel[rel.estimator == est].groupby("roi").reliability
        mu = g.mean().reindex(rois)
        sd = g.std().reindex(rois)
        ax.bar(x + (i - 0.5) * w, mu.values, w, color=cols[est],
               label=est.replace("_", " "))
        ax.errorbar(x + (i - 0.5) * w, mu.values, yerr=sd.values, fmt="none",
                    ecolor="k", elinewidth=1.2, capsize=2.5)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([roi_display(r) for r in rois], rotation=25, ha="right")
    ax.set_ylabel("split-half reliability\nof the data RDM (Spearman $\\rho$)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title(f"Which estimator recovers a reproducible geometry?\n"
                 f"{modality}, all uncovers, 32 conditions", fontsize=10)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"reliability_{modality}.{e}"),
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run, "reliability": reliability})
    else:
        run()
