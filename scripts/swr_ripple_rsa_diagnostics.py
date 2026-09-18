#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostics for the ripple RSA: why a fit is or is not trustworthy.

Two questions, two subcommands. Neither fits a model to test a hypothesis --
both ask whether the measurement is sound enough for a fit to mean anything.

  leverage    How much can ONE condition move the fit? Leave-one-condition-out
              in both condition spaces. This is what showed the 8-condition
              space was unusable: a single config touches 7 of 28 pairs, so it
              can move rho by up to 0.52 -- more than the entire mPFC effect
              that originally motivated the analysis. 17 of 113 single
              conditions there move rho by more than the space's own chance
              SD; in the 32-condition space, 0 of 160 do.

  power       WHY the analysis is underpowered. Spike sparsity, the Poisson
              floor on a condition estimate, and whether more ripples or a
              longer integration window would fix it. This is the diagnosis
              behind the ~0 reliability ceiling.

  weighting   Which FEATURES carry the RDM? An RDM entry is a correlation
              across features, and the shipped estimator centred them without
              scaling, so influence was proportional to across-condition
              variance. With firing rates spanning two orders of magnitude the
              effective number of contributing cells was 36-43% of the actual
              count (top 5 cells = up to 43% of an ROI's RDM). HFB was fine
              (82-89%) because it is already robust-z per derivation. This is
              why `DECISIONS["normalise"]` is now "zscore".

Fused from the former swr_ripple_rsa_conditions.py and swr_ripple_rsa_inputs.py.

    python scripts/swr_ripple_rsa_diagnostics.py power
    python scripts/swr_ripple_rsa_diagnostics.py leverage --modality=spikes
    python scripts/swr_ripple_rsa_diagnostics.py weighting --modality=hfb

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
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
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 1.0, "lines.linewidth": 2.2, "savefig.dpi": 300,
})

SPIKE_ROIS = ["HC_anterior", "HC_mid", "mPFC", "mOFC", "PCC"]
HFB_ROIS = ["HC_anterior", "HC_mid", "mPFC", "mOFC", "Visual"]
MODELS = ["current_location", "known_set", "full_abcd"]
MODEL_COLOURS = {"current_location": "#5b9b8d", "known_set": "#23677E",
                 "full_abcd": "#DC673E"}
STATE_COLOURS = ["#F15A29", "#F7931E", "#C7C6E2", "#6B60AA"]
PAIR = (2, 19)
TOPN = 5


def _out(data_root=None):
    d = os.path.join(swr_io.derivatives_dir(data_root), "group", "swr",
                     f"ripple_rsa_diagnostics_{STAMP}")
    os.makedirs(d, exist_ok=True)
    return d


def _pack(modality, bundle_dir, data_root):
    D = rrsa.DECISIONS
    bd = rrsa.load_bundle(bundle_dir, pad_s=D["pad_s"])
    ev = rrsa.discovery_events(bd)
    if modality == "spikes":
        tab = rrsa.cell_roi_table()
        rois = [r for r in SPIKE_ROIS if (tab.roi == r).sum() >= 5]
        tab = tab[tab.roi.isin(rois)]
        cache = rrsa.cache_ripple_rates(
            bd, ev, tab, rrsa.load_spike_times(verbose=False), D["window_s"],
            extent=D["extent"], scheme=D["scheme"])
    else:
        tab = rrsa.hfb_roi_table(bundle_dir)
        rois = [r for r in HFB_ROIS if (tab.roi == r).sum() >= 5]
        tab = tab[tab.roi.isin(rois)]
        cache = rrsa.cache_hfb_rates(
            bd, ev, tab, D["window_s"], scheme=D["scheme"],
            extent=D["extent"], bundle_dir=bundle_dir, verbose=False)
    return rrsa.patterns_from_cache(cache), rois


def leverage(bundle=None, data_root=None, modality="spikes", out=None):
    out_dir = out or _out(data_root)
    swr_io.start_log(out_dir, "swr_ripple_rsa_diagnostics_leverage")
    bundle_dir = bundle or rrsa.default_bundle_dir(data_root)
    np.random.seed(SEED)
    D = rrsa.DECISIONS

    pack, rois = _pack(modality, bundle_dir, data_root)
    models32 = rrsa.model_rdms_flat()
    print(f"{modality}: {pack['patterns'].shape[0]} features, "
          f"{int(np.nansum(pack['counts']))} ripple-condition assignments")

    # ---------------- 32 conditions ----------------
    rows32, cov = [], []
    for roi in rois:
        rdm, _ = rrsa.rdm_for_flat(pack, roi)
        c = rrsa.condition_coverage(pack, roi)
        c["roi"] = roi
        cov.append(c)
        for name, M in models32.items():
            delta, full = rrsa.loco_flat(rdm, M, family="all")
            for i in range(rrsa.N_COND):
                rows32.append({
                    "roi": roi, "model": name, "cond": i,
                    "label": rrsa.COND_LABELS[i],
                    "config": rrsa.CONFIG_LABELS[rrsa.COND_CFG[i]],
                    "state": rrsa.STATES[rrsa.COND_STATE[i]],
                    "delta": delta[i], "rho_full": full})
    d32 = pd.DataFrame(rows32)
    cov = pd.concat(cov, ignore_index=True)

    # ---------------- 8 conditions, one RDM per state ----------------
    rows8 = []
    for roi in rois:
        for st in rrsa.STATES:
            rdm, _ = rrsa.rdm_for(pack, roi, st)
            for name in ("known_set", "full_abcd"):
                M = rrsa.model_rdms(st)[name]
                if M[np.triu_indices(8, 1)].std() < 1e-12:
                    continue
                delta, full = rrsa.loco_configs(rdm, M)
                for c in range(rrsa.N_CONFIG):
                    rows8.append({
                        "roi": roi, "state": st, "model": name, "config_i": c,
                        "config": rrsa.CONFIG_LABELS[c],
                        "delta": delta[c], "rho_full": full})
    d8 = pd.DataFrame(rows8)

    d32.to_csv(os.path.join(out_dir, f"loco_32_{modality}.csv"), index=False)
    d8.to_csv(os.path.join(out_dir, f"loco_8_{modality}.csv"), index=False)
    cov.to_csv(os.path.join(out_dir, f"coverage_{modality}.csv"), index=False)

    _report(d32, d8, cov, modality)
    _fig_32(d32, cov, rois, modality, out_dir)
    _fig_8(d8, rois, modality, out_dir)
    _fig_compare(d32, d8, rois, modality, out_dir)
    _overview_leverage(d32, d8, cov, rois, modality, out_dir, D)
    print(f"\nwritten to {out_dir}")


def _report(d32, d8, cov, modality):
    print(f"\n===== {modality}: most / least informative of the 32 "
          f"conditions (model = known_set) =====")
    t = d32[d32.model == "known_set"]
    for roi, g in t.groupby("roi", sort=False):
        g = g.dropna(subset=["delta"]).sort_values("delta", ascending=False)
        top = " | ".join(f"{r.label}({r.delta:+.3f})" for _, r in g.head(3).iterrows())
        bot = " | ".join(f"{r.label}({r.delta:+.3f})" for _, r in g.tail(3).iterrows())
        print(f"  {roi_display(roi):<12} rho={g.rho_full.iloc[0]:+.3f}"
              f"   carries: {top}\n{'':<16}against: {bot}")
    print(f"\n  spread of delta (max-min) -- how concentrated the fit is:")
    a = (d32[d32.model == "known_set"].groupby("roi").delta
         .agg(lambda x: np.nanmax(x) - np.nanmin(x)))
    b = (d8[d8.model == "known_set"].groupby(["roi", "state"]).delta
         .agg(lambda x: np.nanmax(x) - np.nanmin(x)).groupby("roi").median())
    print(pd.DataFrame({"32-cond": a.round(3),
                        "8-cond (median over states)": b.round(3)}).to_string())


def _fig_32(d32, cov, rois, modality, out_dir):
    """Heatmap ROI x 32 conditions of the leave-one-out contribution."""
    fig, axes = plt.subplots(
        len(MODELS) + 1, 1, figsize=(17 * CM, 12.5 * CM),
        gridspec_kw={"height_ratios": [1] * len(MODELS) + [1.15],
                     "hspace": 0.32})
    order = np.arange(rrsa.N_COND)
    labels = [rrsa.COND_LABELS[i].split("|")[0] for i in order]
    states = [rrsa.COND_LABELS[i].split("|")[1] for i in order]

    for r, model in enumerate(MODELS):
        ax = axes[r]
        Mx = np.array([[d32[(d32.roi == roi) & (d32.model == model)
                            & (d32.cond == i)].delta.iloc[0] for i in order]
                       for roi in rois], float)
        v = np.nanmax(np.abs(Mx)) or 1e-6
        im = ax.imshow(Mx, cmap="RdBu_r", aspect="auto",
                       norm=TwoSlopeNorm(0, -v, v))
        ax.set_yticks(range(len(rois)))
        ax.set_yticklabels([roi_display(x) for x in rois])
        ax.set_xticks(order if r == len(MODELS) - 1 else [])
        if r == len(MODELS) - 1:
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_title(f"{model.replace('_', ' ')}", fontsize=10, pad=3,
                     color=MODEL_COLOURS[model])
        for k in range(1, 4):
            ax.axvline(k * 8 - 0.5, color="k", lw=1.4)
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
        cb.set_label("Δρ if dropped", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    ax = axes[-1]
    n = cov.groupby("cond").n_ripples.max().reindex(order).values
    ax.bar(order, n, color=[STATE_COLOURS[rrsa.COND_STATE[i]] for i in order],
           width=0.82)
    ax.set_xticks(order)
    ax.set_xticklabels([f"{l}|{s}" for l, s in zip(labels, states)],
                       rotation=90, fontsize=6)
    ax.set_ylabel("ripples\nin condition")
    ax.set_xlim(-0.6, rrsa.N_COND - 0.4)
    for k in range(1, 4):
        ax.axvline(k * 8 - 0.5, color="k", lw=1.4)
    ax.set_title("sampling per condition (colour = uncover A/B/C/D)",
                 fontsize=10, pad=3)
    fig.suptitle(
        f"Which of the 32 conditions carry the fit — {modality}\n"
        f"red = dropping it LOWERS ρ (it carries the fit)   ·   "
        f"blue = dropping it RAISES ρ (it works against the model)",
        fontsize=11)
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.11, right=0.93)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"conditions_32_{modality}.{e}"),
                    bbox_inches="tight")
    plt.close(fig)


def _fig_8(d8, rois, modality, out_dir):
    """The same, in the old 8-condition space: ROI x config, one row per state."""
    model = "known_set"
    sts = [s for s in rrsa.STATES
           if len(d8[(d8.model == model) & (d8.state == s)].dropna(subset=["delta"]))]
    fig, axes = plt.subplots(len(sts), 1,
                             figsize=(10 * CM, len(sts) * 4.3 * CM),
                             squeeze=False, sharex=True,
                             gridspec_kw={"hspace": 0.30})
    Mall = []
    for st in sts:
        Mall.append(np.array(
            [[d8[(d8.roi == roi) & (d8.model == model) & (d8.state == st)
                 & (d8.config_i == c)].delta.iloc[0]
              if len(d8[(d8.roi == roi) & (d8.model == model)
                        & (d8.state == st) & (d8.config_i == c)]) else np.nan
              for c in range(rrsa.N_CONFIG)] for roi in rois], float))
    v = np.nanmax(np.abs(np.array(Mall))) or 1e-6
    for r, st in enumerate(sts):
        ax = axes[r][0]
        im = ax.imshow(Mall[r], cmap="RdBu_r", aspect="auto",
                       norm=TwoSlopeNorm(0, -v, v))
        ax.set_yticks(range(len(rois)))
        ax.set_yticklabels([roi_display(x) for x in rois])
        ax.set_xticks(range(rrsa.N_CONFIG))
        if r == len(sts) - 1:
            ax.set_xticklabels(rrsa.CONFIG_LABELS, rotation=90, fontsize=7)
        ax.set_title(f"uncover {st}", fontsize=10, pad=3,
                     color=STATE_COLOURS[rrsa.STATES.index(st)])
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
        cb.set_label("Δρ if dropped", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"The old 8-condition space — {modality}, known set\n"
                 f"one config touches 7 of 28 pairs, so a single condition "
                 f"moves ρ far\n(note the colour scale: ±{v:.2f}, "
                 f"~10× the 32-condition one)", fontsize=10)
    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.24, right=0.88)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"conditions_8_{modality}.{e}"),
                    bbox_inches="tight")
    plt.close(fig)


def _fig_compare(d32, d8, rois, modality, out_dir):
    """How concentrated the fit is, 8 vs 32 conditions."""
    model = "known_set"
    a = (d32[d32.model == model].groupby("roi").delta
         .agg(lambda x: np.nanmax(x) - np.nanmin(x)).reindex(rois))
    b = (d8[d8.model == model].groupby(["roi", "state"]).delta
         .agg(lambda x: np.nanmax(x) - np.nanmin(x))
         .groupby("roi").median().reindex(rois))
    sd8 = 1 / np.sqrt(28 - 1)
    sd32 = 1 / np.sqrt(448 - 1)
    fig, ax = plt.subplots(figsize=(9 * CM, 6.2 * CM))
    x = np.arange(len(rois))
    ax.bar(x - 0.2, b.values, 0.38, color="#C7C6E2", label="8 conditions")
    ax.bar(x + 0.2, a.values, 0.38, color="#23677E", label="32 conditions")
    # the noise floor each space is working against. A design is only usable
    # when one condition moves rho by LESS than chance already does.
    ax.axhline(sd8, color="#C7C6E2", ls="--", lw=2.0)
    ax.axhline(sd32, color="#23677E", ls="--", lw=2.0)
    ax.annotate(f"chance SD, 8 cond ({sd8:.2f})", (len(rois) - 0.45, sd8),
                ha="right", va="bottom", fontsize=8, color="#8f8cb8")
    ax.annotate(f"chance SD, 32 cond ({sd32:.2f})", (len(rois) - 0.45, sd32),
                ha="right", va="bottom", fontsize=8, color="#23677E")
    ax.set_xticks(x)
    ax.set_xticklabels([roi_display(r) for r in rois], rotation=30, ha="right")
    ax.set_ylabel("spread of Δρ\n(max − min across conditions)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("How far one single condition\ncan move the fit", fontsize=11)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"conditions_spread_{modality}.{e}"),
                    bbox_inches="tight")
    plt.close(fig)


def _overview_leverage(d32, d8, cov, rois, modality, out_dir, D):
    o = {"created": datetime.now().isoformat(timespec="seconds"),
         "modality": modality,
         "what": "leave-one-condition-out contribution to the ripple RSA fit",
         "reading": {"positive_delta": "condition CARRIES the fit",
                     "negative_delta": "condition works AGAINST the model",
                     "zero": "contributes nothing"},
         "decisions": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in D.items()},
         "per_roi": {}}
    t = d32[d32.model == "known_set"]
    for roi in rois:
        g = t[t.roi == roi].dropna(subset=["delta"]).sort_values(
            "delta", ascending=False)
        c = cov[cov.roi == roi]
        o["per_roi"][roi] = {
            "rho_full_32cond_known_set": round(float(g.rho_full.iloc[0]), 4),
            "most_informative": [{"condition": r.label,
                                  "delta": round(r.delta, 4)}
                                 for _, r in g.head(3).iterrows()],
            "least_informative": [{"condition": r.label,
                                   "delta": round(r.delta, 4)}
                                  for _, r in g.tail(3).iterrows()],
            "delta_spread_32cond": round(
                float(np.nanmax(g.delta) - np.nanmin(g.delta)), 4),
            "delta_spread_8cond_median_over_states": round(float(
                d8[(d8.model == "known_set") & (d8.roi == roi)]
                .groupby("state").delta
                .agg(lambda x: np.nanmax(x) - np.nanmin(x)).median()), 4),
            "conditions_with_no_estimable_pairs": int(
                (c.n_pairs_estimable == 0).sum()),
            "min_ripples_in_a_condition": int(c.n_ripples.min()),
            "max_ripples_in_a_condition": int(c.n_ripples.max()),
        }
    with open(os.path.join(out_dir,
                           f"leverage_overview_{modality}.json"), "w") as f:
        json.dump(o, f, indent=2)
    swr_io.write_settings(out_dir, {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": os.path.basename(__file__), "modality": modality,
        "decisions": {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in D.items()}, "seed": SEED})


def _observed(modality, expanded_dir, bundle_dir, data_root):
    """The observed pattern pack -- from the stored run if it is there."""
    f = os.path.join(expanded_dir or "", f"packs_observed_{modality}.npz")
    if expanded_dir and os.path.isfile(f):
        pk = rrsa.load_packs(f)[0][0]
        pk["patterns"] = np.asarray(pk["patterns"], float)
        pk["roi"] = np.asarray(pk["roi"])
        return pk
    D = rrsa.DECISIONS
    bd = rrsa.load_bundle(bundle_dir, pad_s=D["pad_s"])
    ev = rrsa.discovery_events(bd)
    if modality == "spikes":
        tab = rrsa.cell_roi_table()
        tab = tab[tab.roi.isin(SPIKE_ROIS)]
        cache = rrsa.cache_ripple_rates(
            bd, ev, tab, rrsa.load_spike_times(verbose=False), D["window_s"],
            extent=D["extent"], scheme=D["scheme"])
    else:
        tab = rrsa.hfb_roi_table(bundle_dir)
        tab = tab[tab.roi.isin(HFB_ROIS)]
        cache = rrsa.cache_hfb_rates(
            bd, ev, tab, D["window_s"], scheme=D["scheme"],
            extent=D["extent"], bundle_dir=bundle_dir, verbose=False)
    return rrsa.patterns_from_cache(cache)


def _stats(X):
    """Centred patterns, z-scored patterns, per-cell variance, effective n."""
    X = np.asarray(X, float)
    C = X - np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(C, axis=1, keepdims=True)
    Z = np.where(sd > 0, C / np.where(sd == 0, np.nan, sd), np.nan)
    v = np.nanvar(C, axis=1)
    ok = np.isfinite(v) & (v > 0)
    vv = v[ok]
    eff = float((vv.sum() ** 2) / (vv ** 2).sum()) if len(vv) else np.nan
    return C, Z, v, eff


def weighting(bundle=None, data_root=None, modality="spikes", expanded_dir=None,
        out=None):
    out_dir = out or _out(data_root)
    swr_io.start_log(out_dir, "swr_ripple_rsa_diagnostics_weighting")
    bundle_dir = bundle or rrsa.default_bundle_dir(data_root)
    np.random.seed(SEED)
    pk = _observed(modality, expanded_dir, bundle_dir, data_root)
    F = rrsa.flat_patterns(pk)
    roiv = np.asarray(pk["roi"])
    rois = [r for r in (SPIKE_ROIS if modality == "spikes" else HFB_ROIS)
            if (roiv == r).sum() >= 5]

    per_roi, rows = {}, []
    for roi in rois:
        X = np.asarray(F[roiv == roi], float)
        C, Z, v, eff = _stats(X)
        order = np.argsort(np.where(np.isfinite(v), v, -np.inf))[::-1]
        s = np.sort(v[np.isfinite(v) & (v > 0)])[::-1]
        mu = np.nanmean(X, axis=1)
        mu = mu[np.isfinite(mu)]
        i, j = PAIR
        ok = np.isfinite(C[:, i]) & np.isfinite(C[:, j])
        okz = np.isfinite(Z[:, i]) & np.isfinite(Z[:, j])
        r_c = (np.corrcoef(C[ok, i], C[ok, j])[0, 1] if ok.sum() > 2
               else np.nan)
        r_z = (np.corrcoef(Z[okz, i], Z[okz, j])[0, 1] if okz.sum() > 2
               else np.nan)
        per_roi[roi] = {"X": X, "C": C, "Z": Z, "v": v, "order": order,
                        "ok": ok, "okz": okz, "r_c": r_c, "r_z": r_z}
        rows.append({
            "modality": modality, "roi": roi, "n_cells": int(np.isfinite(v).sum()),
            "effective_n_cells": round(eff, 1),
            "effective_pct": round(100 * eff / max(np.isfinite(v).sum(), 1), 1),
            "top1_variance_share_pct": round(100 * s[0] / s.sum(), 1) if len(s) else np.nan,
            "top5_variance_share_pct": round(100 * s[:TOPN].sum() / s.sum(), 1) if len(s) else np.nan,
            "rate_median": round(float(np.median(mu)), 4),
            "rate_p95": round(float(np.percentile(mu, 95)), 4),
            "rate_max": round(float(mu.max()), 4),
            "example_pair_r_centred": round(float(r_c), 3),
            "example_pair_r_zscored": round(float(r_z), 3),
        })
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, f"input_weighting_{modality}.csv"),
               index=False)
    print(f"\n=== {modality}: how evenly are features weighted in the RDM ===")
    print(res.to_string(index=False))

    _fig_inputs(per_roi, rois, res, modality, out_dir)
    with open(os.path.join(out_dir, f"weighting_overview_{modality}.json"),
              "w") as f:
        json.dump({
            "created": datetime.now().isoformat(timespec="seconds"),
            "modality": modality,
            "normalisation_currently_used": (
                "per-feature CENTRING across conditions, no scaling "
                "(C = P - nanmean(P, axis=1)); correlation is then taken "
                "across the feature axis"),
            "consequence": (
                "features are NOT equally weighted -- a feature's influence "
                "is proportional to its across-condition variance, and firing "
                "rates span two orders of magnitude"),
            "effective_n_is": "participation ratio (sum v)^2 / sum v^2",
            "hfb_note": ("HFB is already robust-z per derivation over the "
                         "whole session, so its effective n is much closer to "
                         "its actual n"),
            "per_roi": res.set_index("roi").to_dict(orient="index"),
        }, f, indent=2)
    swr_io.write_settings(out_dir, {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": os.path.basename(__file__), "modality": modality,
        "example_pair": list(PAIR), "seed": SEED,
        "decisions": {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in rrsa.DECISIONS.items()}})
    print(f"\nwritten to {out_dir}")


def _fig_inputs(per_roi, rois, res, modality, out_dir):
    fig, axes = plt.subplots(3, len(rois),
                             figsize=(len(rois) * 4.0 * CM, 14.5 * CM),
                             squeeze=False,
                             gridspec_kw={"height_ratios": [1.25, 1, 1],
                                          "hspace": 0.78, "wspace": 0.45})
    i, j = PAIR
    for c, roi in enumerate(rois):
        d = per_roi[roi]
        # --- a: the pattern matrix, cells sorted by how much they can move it
        ax = axes[0][c]
        Cm = d["C"][d["order"]]
        v = np.nanpercentile(np.abs(Cm), 98) or 1e-9
        ax.imshow(Cm, cmap="RdBu_r", aspect="auto",
                  norm=TwoSlopeNorm(0, -v, v), interpolation="nearest")
        for k in range(1, 4):
            ax.axvline(k * 8 - 0.5, color="k", lw=0.8)
        ax.axhline(TOPN - 0.5, color="#0e3d3a", lw=1.6)
        ax.set_xticks([3.5, 11.5, 19.5, 27.5])
        ax.set_xticklabels(rrsa.STATES)
        ax.set_title(roi_display(roi), fontsize=10, color=get_roi_colour(roi))
        if c == 0:
            ax.set_ylabel("features,\nsorted by variance")

        # --- b: Lorenz curve of across-condition variance
        ax = axes[1][c]
        s = np.sort(d["v"][np.isfinite(d["v"]) & (d["v"] > 0)])[::-1]
        cum = np.cumsum(s) / s.sum()
        x = np.arange(1, len(s) + 1) / len(s)
        ax.plot(x, cum, color=get_roi_colour(roi), lw=2.2)
        ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1.4)
        e = res.loc[res.roi == roi, "effective_pct"].iloc[0]
        ax.annotate(f"eff. {e:.0f}%", xy=(0.96, 0.08), xycoords="axes fraction",
                    ha="right", fontsize=8, color=get_roi_colour(roi),
                    fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("feature rank\n(fraction)")
        if c == 0:
            ax.set_ylabel("cumulative share\nof variance")

        # --- c: one RDM entry as the scatter it really is
        ax = axes[2][c]
        ok = d["ok"]
        top = np.zeros(len(ok), bool)
        top[d["order"][:TOPN]] = True
        ax.scatter(d["C"][ok & ~top, i], d["C"][ok & ~top, j], s=9,
                   color="#bdbdbd", edgecolor="none")
        ax.scatter(d["C"][ok & top, i], d["C"][ok & top, j], s=26,
                   color="#0e3d3a", edgecolor="none")
        ax.axhline(0, color="k", lw=0.7)
        ax.axvline(0, color="k", lw=0.7)
        # inside the axes: a title here collides with the row above
        ax.annotate(f"r={d['r_c']:+.2f}\nz-scored {d['r_z']:+.2f}\nn={ok.sum()}",
                    xy=(0.03, 0.97), xycoords="axes fraction", va="top",
                    ha="left", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#cccccc", lw=0.6))
        ax.set_xlabel(f"cond {i}")
        if c == 0:
            ax.set_ylabel(f"cond {j}\n(centred rate)")

    fig.suptitle(
        f"What goes into one RDM entry — {modality}\n"
        f"features are CENTRED across conditions but NOT scaled, so "
        f"influence is proportional to across-condition variance\n"
        f"dark green = the {TOPN} highest-variance features   ·   "
        f"grey dashed = every feature weighted equally", fontsize=10)
    fig.subplots_adjust(top=0.84, bottom=0.10, left=0.10, right=0.97)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"rdm_inputs_{modality}.{e}"),
                    bbox_inches="tight")
    plt.close(fig)




def _rdm_norm(P, mode, min_cells=rrsa.MIN_CELLS_PER_PAIR):
    """`build_rdm_flat`, with the per-feature scaling made switchable.

    mode "centre" reproduces the shipped estimator exactly. mode "zscore"
    additionally divides each feature by its own across-condition SD, so every
    feature contributes equally to the correlation regardless of firing rate.
    """
    P = np.asarray(P, float)
    n = P.shape[1]
    C = P - np.nanmean(P, axis=1, keepdims=True)
    if mode == "zscore":
        sd = np.nanstd(C, axis=1, keepdims=True)
        C = np.where(sd > 0, C / np.where(sd == 0, np.nan, sd), np.nan)
    rdm = np.full((n, n), np.nan)
    np.fill_diagonal(rdm, 0.0)
    ok_all = np.isfinite(C)
    for a in range(n):
        for b in range(a + 1, n):
            ok = ok_all[:, a] & ok_all[:, b]
            if ok.sum() < min_cells:
                continue
            u, w = C[ok, a], C[ok, b]
            if u.std() == 0 or w.std() == 0:
                continue
            rdm[a, b] = rdm[b, a] = 1.0 - np.corrcoef(u, w)[0, 1]
    return rdm


def normalisation_sensitivity(expanded_dir, modality="spikes", out=None,
                              data_root=None, family="all"):
    """Does the answer depend on whether features are scaled?

    Refits every ROI x model under both normalisations against the IDENTICAL
    stored surrogate null, so the only thing that changes is the weighting of
    features. Reported because it turns out to matter: the two schemes do not
    rank the ROIs the same way.
    """
    out_dir = out or _out(data_root)
    packs, meta = rrsa.load_packs(
        os.path.join(expanded_dir, f"packs_surrogate_{modality}.npz"))
    obs, _ = rrsa.load_packs(
        os.path.join(expanded_dir, f"packs_observed_{modality}.npz"))
    for p in packs + obs:
        p["patterns"] = np.asarray(p["patterns"], float)
        p["roi"] = np.asarray(p["roi"])
    rois = meta["rois"]
    models = rrsa.model_rdms_flat()

    Fo, ro = rrsa.flat_patterns(obs[0]), obs[0]["roi"]
    Fs = [(rrsa.flat_patterns(p), p["roi"]) for p in packs]
    rows = []
    for mode in ("centre", "zscore"):
        for roi in rois:
            # build each RDM once and fit all models to it -- rebuilding per
            # model was what made the first pass take ten minutes
            rd_o = _rdm_norm(Fo[ro == roi], mode)
            rd_n = [_rdm_norm(F[r == roi], mode) for F, r in Fs]
            for name, M in models.items():
                o = rrsa.fit_rho_flat(rd_o, M, family=family)
                null = np.array([rrsa.fit_rho_flat(x, M, family=family)
                                 for x in rd_n])
                null = null[np.isfinite(null)]
                rows.append({
                    "modality": modality, "norm": mode, "roi": roi,
                    "model": name, "family": family, "rho": o,
                    "z_surr": (o - null.mean()) / null.std(),
                    "p_surr": float((null >= o - 1e-12).mean()),
                    "null_sd": null.std(), "n_surr": len(null)})
            print(f"  {mode} {roi} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, f"norm_sensitivity_{modality}.csv"),
               index=False)
    piv = res.pivot_table(index=["roi", "model"], columns="norm",
                          values=["rho", "z_surr"])
    print(f"\n=== {modality}: centring vs z-scoring, identical null ===")
    print(piv.round(3).to_string())
    _fig_sensitivity(res, rois, modality, out_dir)
    print(f"\nwritten to {out_dir}")


def _fig_sensitivity(res, rois, modality, out_dir):
    models = ["current_location", "known_set", "full_abcd"]
    colours = {"current_location": "#5b9b8d", "known_set": "#23677E",
               "full_abcd": "#DC673E"}
    fig, axes = plt.subplots(1, len(rois),
                             figsize=(len(rois) * 3.9 * CM, 6.0 * CM),
                             squeeze=False, sharey=True)
    for c, roi in enumerate(rois):
        ax = axes[0][c]
        for i, m in enumerate(models):
            g = res[(res.roi == roi) & (res.model == m)].set_index("norm")
            y = [g.z_surr.get("centre", np.nan), g.z_surr.get("zscore", np.nan)]
            ax.plot([0, 1], y, "o-", color=colours[m], ms=6,
                    label=m.replace("_", " ") if c == 0 else None)
        ax.axhline(0, color="k", lw=0.9)
        for lv in (-1.96, 1.96):
            ax.axhline(lv, color="#999999", ls=":", lw=1.2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["centred\n(shipped)", "z-scored"], fontsize=8)
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(roi_display(roi), fontsize=10, color=get_roi_colour(roi))
        if c == 0:
            ax.set_ylabel("z vs surrogate-window null")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(
        f"Does the answer depend on feature scaling? — {modality}\n"
        f"identical surrogate null; only the per-feature weighting changes   "
        f"·   dotted = ±1.96", fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.84])
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"norm_sensitivity_{modality}.{e}"),
                    bbox_inches="tight")
    plt.close(fig)




# =============================================================================
# POWER -- why the ripple-locked pattern is not measurable
# =============================================================================

HALF_WIDTHS_S = [0.010, 0.030, 0.050, 0.125, 0.250, 0.500]
FRACTIONS = [0.125, 0.25, 0.5, 1.0]


def _subset(cache, idx_fn):
    out = []
    for b in cache:
        i = idx_fn(b)
        d = {k: v for k, v in b.items()
             if k not in ("rates", "ci", "si", "pi", "_perm")}
        d["rates"] = b["rates"][:, i]
        d["ci"], d["si"] = b["ci"][i], b["si"][i]
        if "pi" in b:
            d["pi"] = b["pi"][i]
        out.append(d)
    return out


def _reliability(cache, roi, rng, n_splits):
    """Split-half reliability of the 32-condition RDM -- the ceiling metric."""
    iu = np.triu_indices(rrsa.N_COND, 1)
    vals = []
    for _ in range(n_splits):
        perms = {id(b): rng.permutation(b["rates"].shape[1]) for b in cache}
        halves = [_subset(cache, lambda b, h=h: perms[id(b)][h::2])
                  for h in (0, 1)]
        v = []
        for sub in halves:
            pk = rrsa.patterns_uncover(sub, collapse_phase=True)
            v.append(rrsa.rdm_for_flat(pk, roi, normalise="zscore")[0][iu])
        a, b2 = v
        ok = np.isfinite(a) & np.isfinite(b2)
        if ok.sum() > 10:
            vals.append(float(stats.spearmanr(a[ok], b2[ok]).correlation))
    return float(np.mean(vals)) if vals else np.nan


def power(bundle=None, data_root=None, modality="spikes", n_splits=8,
          out=None):
    """Why this analysis is underpowered, in numbers and in figures."""
    D = rrsa.DECISIONS
    out_dir = out or _out(data_root)
    swr_io.start_log(out_dir, "swr_ripple_rsa_diagnostics_power")
    bundle_dir = bundle or rrsa.default_bundle_dir(data_root)
    np.random.seed(SEED)
    bd = rrsa.load_bundle(bundle_dir, pad_s=D["pad_s"])
    ev = rrsa.all_uncover_events(bd)
    tab = rrsa.cell_roi_table()
    rois = [r for r in SPIKE_ROIS if (tab.roi == r).sum() >= 5]
    tab = tab[tab.roi.isin(rois)]
    spikes = rrsa.load_spike_times(verbose=False)
    rng = np.random.default_rng(SEED)

    # ---- raw spike counts per (cell, ripple) -------------------------------
    counts, durs, cell_roi = [], [], []
    for s in sorted(set(tab.session)):
        rip = rrsa.ripples_after_uncovers(bd, s, ev, D["window_s"])
        cells = tab[tab.session == s]
        if rip.empty or cells.empty:
            continue
        half = rip.duration_s.values / 2.0
        C = np.empty((len(cells), len(rip)))
        for r, (_, c) in enumerate(cells.iterrows()):
            C[r] = rrsa.spike_counts_in_windows(
                spikes[s]["spikes"][int(c.cell)], rip.t_peak_s.values, half)
        counts.append(C)
        durs.append(rip.duration_s.values)
        cell_roi.extend(cells.roi.tolist())
    flat = np.concatenate([c.ravel() for c in counts])
    dur = np.concatenate(durs)
    cell_roi = np.array(cell_roi)

    # ---- Poisson floor on a per-condition estimate -------------------------
    # A condition mean rests on k spikes. Poisson CV of that estimate is
    # 1/sqrt(k). The observed across-condition CV contains BOTH that noise and
    # any real signal, so signal_cv^2 = obs_cv^2 - noise_cv^2; where that is
    # negative there is no detectable signal in the cell at all.
    cache = rrsa.cache_ripple_rates(bd, ev, tab, spikes, D["window_s"],
                                    extent=D["extent"], scheme="uncovers")
    # K and the rates are built from the SAME raw count matrices, so they stay
    # aligned with `cell_roi`. Going via `patterns_uncover` would not: it drops
    # silent cells, which is right for an RDM but breaks the row alignment
    # needed here (503 vs 506).
    K_blocks, R_blocks = [], []
    for C, d, b in zip(counts, durs, cache):
        n_cond = rrsa.N_CONFIG * rrsa.N_STATE
        cond = b["si"] * rrsa.N_CONFIG + b["ci"]
        k = np.zeros((C.shape[0], n_cond))
        np.add.at(k.T, cond, C.T)
        T = np.zeros(n_cond)
        np.add.at(T, cond, d)
        K_blocks.append(k)
        with np.errstate(invalid="ignore", divide="ignore"):
            R_blocks.append(k / np.where(T == 0, np.nan, T))
    K = np.concatenate(K_blocks)
    R = np.concatenate(R_blocks)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_k = np.nanmean(np.where(np.isfinite(R), K, np.nan), axis=1)
        obs_cv = np.nanstd(R, axis=1) / np.nanmean(R, axis=1)
    noise_cv = 1.0 / np.sqrt(np.maximum(mean_k, 1e-9))
    sig_var = obs_cv ** 2 - noise_cv ** 2
    assert len(sig_var) == len(cell_roi), (len(sig_var), len(cell_roi))
    rows = []
    for roi in rois:
        m = cell_roi == roi
        sv = sig_var[m]
        sv = sv[np.isfinite(sv)]
        rows.append({"roi": roi, "n_cells": int(m.sum()),
                     "median_spikes_per_condition": float(np.nanmedian(K[m])),
                     "median_noise_cv": float(np.nanmedian(noise_cv[m])),
                     "median_observed_cv": float(np.nanmedian(obs_cv[m])),
                     "pct_cells_with_detectable_signal": float(
                         100 * (sv > 0).mean()) if len(sv) else np.nan,
                     # obs_cv^2 = noise_cv^2 + signal_cv^2, so this is the
                     # per-cell signal-to-noise of a single condition estimate.
                     # A percentage of cells "above the floor" flatters the
                     # data; the RATIO says how far above, and it is ~0.5.
                     "median_signal_to_noise": float(np.nanmedian(
                         np.sqrt(np.maximum(sig_var[m], 0))
                         / noise_cv[m]))})
    poisson = pd.DataFrame(rows)

    # ---- does a longer integration window help? ----------------------------
    win = []
    for hw in HALF_WIDTHS_S:
        c = rrsa.cache_ripple_rates(bd, ev, tab, spikes, D["window_s"],
                                    extent="fixed", fixed_half_s=hw,
                                    scheme="uncovers")
        for roi in rois:
            win.append({"half_width_s": hw, "roi": roi,
                        "reliability": _reliability(c, roi, rng, n_splits)})
        print(f"  window +-{hw * 1000:.0f} ms done", flush=True)
    win = pd.DataFrame(win)

    # ---- would more ripples help? ------------------------------------------
    frac = []
    for f in FRACTIONS:
        sub = _subset(cache, lambda b, f=f: rng.choice(
            b["rates"].shape[1], max(2, int(b["rates"].shape[1] * f)),
            replace=False))
        n = sum(b["rates"].shape[1] for b in sub)
        for roi in rois:
            frac.append({"fraction": f, "n_ripples": n, "roi": roi,
                         "reliability": _reliability(sub, roi, rng, n_splits)})
        print(f"  fraction {f:.3g} done", flush=True)
    frac = pd.DataFrame(frac)

    for name, t in (("spike_counts", pd.DataFrame({
                        "spikes_per_cell_ripple": flat})),
                    ("poisson_floor", poisson),
                    ("reliability_vs_window", win),
                    ("reliability_vs_nripples", frac)):
        t.to_csv(os.path.join(out_dir, f"power_{name}_{modality}.csv"),
                 index=False)

    print(f"\n=== {modality}: why this is underpowered ===")
    print(f"  ripple duration: median {np.median(dur) * 1000:.0f} ms")
    print(f"  (cell, ripple) observations: {flat.size}")
    print(f"  ... with ZERO spikes: {100 * (flat == 0).mean():.1f}%")
    print(f"  spikes per (cell, condition): median {np.median(K):.1f}")
    print()
    print(poisson.round(1).to_string(index=False))
    print("\n  reliability vs integration window:")
    print(win.pivot_table(index="half_width_s", columns="roi",
                          values="reliability").round(3).to_string())
    print("\n  reliability vs number of ripples:")
    print(frac.pivot_table(index="n_ripples", columns="roi",
                           values="reliability").round(3).to_string())

    _fig_power(flat, dur, K, poisson, win, frac, rois, modality, out_dir)
    with open(os.path.join(out_dir, f"power_overview_{modality}.json"),
              "w") as f:
        json.dump({
            "created": datetime.now().isoformat(timespec="seconds"),
            "modality": modality,
            "diagnosis": "single-unit firing inside a ~60 ms ripple is too "
                         "sparse to define a population pattern",
            "median_ripple_duration_ms": float(np.median(dur) * 1000),
            "pct_cell_ripple_observations_with_zero_spikes": float(
                100 * (flat == 0).mean()),
            "median_spikes_per_cell_per_condition": float(np.median(K)),
            "poisson_floor": poisson.to_dict(orient="records"),
            "reliability_vs_window": win.to_dict(orient="records"),
            "reliability_vs_nripples": frac.to_dict(orient="records"),
        }, f, indent=2)
    swr_io.write_settings(out_dir, {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": os.path.basename(__file__) + " power",
        "modality": modality, "n_splits": n_splits,
        "half_widths_s": HALF_WIDTHS_S, "fractions": FRACTIONS, "seed": SEED,
        "decisions": {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in rrsa.DECISIONS.items()}})
    print(f"\nwritten to {out_dir}")


def _fig_power(flat, dur, K, poisson, win, frac, rois, modality, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(19 * CM, 12.5 * CM),
                             gridspec_kw={"hspace": 0.95, "wspace": 0.42})

    ax = axes[0][0]
    mx = 5
    h = [np.mean(flat == k) for k in range(mx)] + [np.mean(flat >= mx)]
    ax.bar(range(mx + 1), [100 * x for x in h], color="#23677E")
    ax.set_xticks(range(mx + 1))
    ax.set_xticklabels([str(k) for k in range(mx)] + [f"{mx}+"])
    ax.set_xlabel("spikes in the ripple")
    ax.set_ylabel("% of (cell, ripple)\nobservations")
    ax.set_title(f"a  {100 * (flat == 0).mean():.0f}% of observations\n"
                 f"are empty", fontsize=10, loc="left")

    ax = axes[0][1]
    ax.hist(dur * 1000, bins=40, color="#5b9b8d")
    ax.axvline(np.median(dur) * 1000, color="#0e3d3a", lw=2.0)
    ax.set_xlabel("ripple duration (ms)")
    ax.set_ylabel("ripples")
    ax.set_title(f"b  median {np.median(dur) * 1000:.0f} ms of\n"
                 f"observation per event", fontsize=10, loc="left")

    ax = axes[0][2]
    kk = K[np.isfinite(K)]
    ax.hist(np.clip(kk, 0, 20), bins=np.arange(0, 21) - 0.5, color="#DC673E")
    ax.axvline(np.median(kk), color="#0e3d3a", lw=2.0)
    ax.set_xlabel("spikes per (cell, condition)")
    ax.set_ylabel("count")
    _n = np.median(kk)
    ax.set_title(f"c  median {_n:.0f} spike{'' if _n == 1 else 's'} behind\n"
                 f"each RDM entry", fontsize=10, loc="left")

    ax = axes[1][0]
    x = np.arange(len(poisson))
    ax.bar(x - 0.2, poisson.median_noise_cv.values, 0.38, color="#bdbdbd",
           label="Poisson floor")
    ax.bar(x + 0.2, poisson.median_observed_cv.values, 0.38, color="#6B60AA",
           label="observed")
    top = max(poisson.median_observed_cv.max(), poisson.median_noise_cv.max())
    for i, v in enumerate(poisson.median_signal_to_noise.values):
        ax.annotate(f"{v:.2f}", (i, max(
            poisson.median_observed_cv.values[i],
            poisson.median_noise_cv.values[i]) + top * 0.04), ha="center",
            va="bottom", fontsize=7, color="#0e3d3a")
    ax.set_xticks(x)
    ax.set_xticklabels([roi_display(r) for r in poisson.roi],
                       rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("CV of a\ncondition estimate")
    ax.set_ylim(0, top * 1.55)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.02), handlelength=1.1,
              columnspacing=0.9)
    ax.set_title("d  observed variation barely exceeds\ncounting noise",
                 fontsize=10, loc="left")

    ax = axes[1][1]
    for roi in rois:
        g = win[win.roi == roi]
        ax.plot(g.half_width_s * 1000, g.reliability, "o-",
                color=get_roi_colour(roi), ms=5, label=roi_display(roi))
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks([10, 30, 100, 300])
    ax.get_xaxis().set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.get_xaxis().set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.set_xlabel("integration half-width (ms, log)")
    ax.set_ylabel("split-half reliability")
    ax.set_title("e  does a longer window help?", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1][2]
    for roi in rois:
        g = frac[frac.roi == roi].sort_values("n_ripples")
        ax.plot(g.n_ripples, g.reliability, "o-", color=get_roi_colour(roi),
                ms=5)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks([1500, 3000, 6000, 12000])
    ax.get_xaxis().set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    ax.get_xaxis().set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.set_xlabel("ripples included (log)")
    ax.set_ylabel("split-half reliability")
    # SAME y-scale as e. On its own axis this panel's noise looks like a
    # trend; against e's range it is visibly flat, which is the whole point:
    # integration time buys reliability, more events do not.
    ax.set_ylim(axes[1][1].get_ylim())
    ax.set_title("f  would more ripples help?", fontsize=10, loc="left")

    fig.suptitle(
        "Why the ripple-locked RSA is underpowered — single units\n"
        "each RDM entry is a correlation across cells whose condition means "
        "rest on ~1 spike", fontsize=11)
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.09, right=0.98)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"power_{modality}.{e}"),
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"leverage": leverage, "weighting": weighting,
                   "power": power})
    else:
        power()
