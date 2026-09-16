#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controls and a properly powered permutation test for the ripple-triggered RSA.

Four questions, four different nulls. They are NOT interchangeable, and the
first one is the reason the others are needed.

WHY THE NULL IS AS WIDE AS IT IS
--------------------------------
An 8 x 8 RDM has 28 unique pairs, so a chance Spearman correlation over it has
SD = 1/sqrt(27) = 0.19. That is arithmetic, not a property of ripples or of the
permutation scheme: with 28 numbers you need |rho| > ~0.37 before p < 0.05.
Every null below reproduces ~0.19, which is the check that they are correct.
The way to shrink it is more RDM cells, not more permutations -- hence C3.

    C1  CONFIG RELABEL (2000 draws)
        Each session gets its own random relabelling of the 8 configurations,
        and the whole estimator is re-run. Destroys the cross-session config
        alignment that pooling cells depends on -- which is the signal.
        Answers: "is the fit about configuration identity?"

    C2  SURROGATE WINDOW (200 draws)   <- SK's suggestion
        Each ripple's window is moved to a random time inside the SAME +-1 s
        press window of the SAME uncover event, keeping its duration. Same
        events, same configs, same states, same cells, same number of windows
        -- the only thing removed is that the window sat on a ripple.
        Answers: "does this require a ripple, or would any window do?"

    C3  POOLED ACROSS B, C, D
        The knowledge-gated model makes the same claim at every state, so the
        three states can be fitted together: up to 84 pairs instead of 28, and
        a null SD of ~0.11 instead of ~0.19. Answers the same question as C1
        with better resolution -- IF the effect is present at all three states.

    C4  POST - PRE CONTRAST
        A difference of two Spearman rhos has no standard sampling
        distribution, so it is permuted directly: ONE per-session relabelling
        applied to BOTH windows, then differenced. Answers: "is the fit
        specific to the window after the press?"

    python scripts/swr_ripple_rsa_controls.py run
    python scripts/swr_ripple_rsa_controls.py run --n_config=2000 --n_surrogate=200

@author: Svenja Kuchenhoff
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.ripple_rsa as rrsa
from mc.plotting.cell_results import (roi_display, get_roi_colour,
                                      OBSERVED_VALUE_COLOR)

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)
STAMP = datetime.now().strftime("%Y-%m-%d")

CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.linewidth": 1.0, "lines.linewidth": 2.2, "lines.markersize": 6,
    "savefig.dpi": 300,
})

ROIS = ["HC_anterior", "HC_mid", "mPFC", "mOFC", "PCC"]
MODEL = "known_set"
STATES_POOLED = ("B", "C", "D")
SEED = 42


def _out(data_root=None):
    d = os.path.join(swr_io.derivatives_dir(data_root), "group", "swr",
                     f"ripple_rsa_controls_{STAMP}")
    os.makedirs(d, exist_ok=True)
    return d


def figures_only(out=None, data_root=None, rois=None):
    """Redraw from `control_results.csv` without re-running the permutations."""
    rois = ROIS if rois is None else (rois.split(",") if isinstance(rois, str)
                                      else list(rois))
    out_dir = out or _out(data_root)
    res = pd.read_csv(os.path.join(out_dir, "control_results.csv"))
    _figure(res, rois, out_dir)
    print(f"figures rewritten in {out_dir}")


def run(bundle=None, data_root=None, n_config=2000, n_surrogate=200,
        n_contrast=1000, rois=None, out=None, figures=True):
    rois = ROIS if rois is None else (rois.split(",") if isinstance(rois, str)
                                      else list(rois))
    out_dir = out or _out(data_root)
    swr_io.start_log(out_dir, "swr_ripple_rsa_controls")
    bundle_dir = bundle or os.path.join(swr_io.derivatives_dir(data_root),
                                        "group", "swr", "bundle")

    bd = rrsa.load_bundle(bundle_dir)
    events = rrsa.discovery_events(bd)
    roi_tab = rrsa.cell_roi_table()
    roi_tab = roi_tab[roi_tab.roi.isin(rois)]
    spikes = rrsa.load_spike_times(verbose=False)
    sessions = sorted(set(roi_tab.session))
    print(f"{len(events)} events, {len(roi_tab)} cells, {len(sessions)} sessions\n")

    cache = {w: rrsa.cache_ripple_rates(bd, events, roi_tab, spikes, win)
             for w, win in rrsa.PRESS_WINDOWS.items()}
    obs_pack = {w: rrsa.patterns_from_cache(c) for w, c in cache.items()}

    iu = np.triu_indices(rrsa.N_CONFIG, 1)
    rng = np.random.default_rng(SEED)

    # ---------- C1 : config relabel ----------------------------------------
    print(f"C1  CONFIG RELABEL ({n_config} draws)")
    perms = [{s: rng.permutation(rrsa.N_CONFIG) for s in sessions}
             for _ in range(n_config)]
    null_packs = [rrsa.patterns_from_cache(cache["post"], config_perm=p)
                  for p in perms]

    rows = []
    for roi in rois:
        for state in rrsa.STATES:
            M = rrsa.model_rdms(state)[MODEL]
            if M[iu].std() < 1e-12:
                continue
            rdm, _ = rrsa.rdm_for(obs_pack["post"], roi, state)
            o = rrsa.fit_rho(rdm, M)
            if not np.isfinite(o):
                continue
            null = np.array([rrsa.fit_rho(rrsa.rdm_for(p, roi, state)[0], M)
                             for p in null_packs])
            null = null[np.isfinite(null)]
            rows.append(_summarise("C1_config_relabel", roi, state, o, null))
            print(f"  {roi:12s} {state}: rho={o:+.3f}  null {null.mean():+.3f}"
                  f" +- {null.std():.3f}  z={rows[-1]['z']:+.2f}"
                  f"  p={rows[-1]['p_upper']:.4f}")

    # ---------- C2 : surrogate window --------------------------------------
    print(f"\nC2  SURROGATE WINDOW ({n_surrogate} draws) -- same events, "
          "window moved off the ripple")
    srng = np.random.default_rng(SEED + 1)
    sur_packs = [rrsa.patterns_from_cache(
        rrsa.cache_ripple_rates(bd, events, roi_tab, spikes,
                                rrsa.PRESS_WINDOWS["post"], surrogate_rng=srng))
        for _ in range(n_surrogate)]
    for roi in rois:
        for state in rrsa.STATES:
            M = rrsa.model_rdms(state)[MODEL]
            if M[iu].std() < 1e-12:
                continue
            rdm, _ = rrsa.rdm_for(obs_pack["post"], roi, state)
            o = rrsa.fit_rho(rdm, M)
            if not np.isfinite(o):
                continue
            null = np.array([rrsa.fit_rho(rrsa.rdm_for(p, roi, state)[0], M)
                             for p in sur_packs])
            null = null[np.isfinite(null)]
            rows.append(_summarise("C2_surrogate_window", roi, state, o, null))
            print(f"  {roi:12s} {state}: rho={o:+.3f}  surrogate "
                  f"{null.mean():+.3f} +- {null.std():.3f}  "
                  f"z={rows[-1]['z']:+.2f}  p={rows[-1]['p_upper']:.4f}")

    # ---------- C3 : pooled across B, C, D ---------------------------------
    print(f"\nC3  POOLED over {STATES_POOLED} ({n_config} draws)")
    for roi in rois:
        o = rrsa.fit_pooled(obs_pack["post"], roi, MODEL, STATES_POOLED)
        if not np.isfinite(o["rho"]):
            continue
        null = np.array([rrsa.fit_pooled(p, roi, MODEL, STATES_POOLED)["rho"]
                         for p in null_packs])
        null = null[np.isfinite(null)]
        r = _summarise("C3_pooled_BCD", roi, "B+C+D", o["rho"], null)
        r["n_pairs"] = o["n_pairs"]
        rows.append(r)
        print(f"  {roi:12s}: rho={o['rho']:+.3f} on {o['n_pairs']} pairs  "
              f"null {null.mean():+.3f} +- {null.std():.3f}  z={r['z']:+.2f}"
              f"  p={r['p_upper']:.4f}")

    # ---------- C4 : post - pre contrast -----------------------------------
    print(f"\nC4  POST - PRE CONTRAST ({n_contrast} draws)")
    crng = np.random.default_rng(SEED + 2)
    cperms = [{s: crng.permutation(rrsa.N_CONFIG) for s in sessions}
              for _ in range(n_contrast)]
    cnull_packs = {w: [rrsa.patterns_from_cache(cache[w], config_perm=p)
                       for p in cperms] for w in ("post", "pre")}
    for roi in rois:
        for state in rrsa.STATES:
            M = rrsa.model_rdms(state)[MODEL]
            if M[iu].std() < 1e-12:
                continue
            o = {}
            for w in ("post", "pre"):
                rdm, _ = rrsa.rdm_for(obs_pack[w], roi, state)
                o[w] = rrsa.fit_rho(rdm, M)
            if not np.isfinite(o["post"] - o["pre"]):
                continue
            null = np.array([
                rrsa.fit_rho(rrsa.rdm_for(cnull_packs["post"][k], roi, state)[0], M)
                - rrsa.fit_rho(rrsa.rdm_for(cnull_packs["pre"][k], roi, state)[0], M)
                for k in range(n_contrast)])
            null = null[np.isfinite(null)]
            rows.append(_summarise("C4_post_minus_pre", roi, state,
                                   o["post"] - o["pre"], null))
            print(f"  {roi:12s} {state}: post-pre={o['post']-o['pre']:+.3f}  "
                  f"null {null.mean():+.3f} +- {null.std():.3f}  "
                  f"z={rows[-1]['z']:+.2f}  p={rows[-1]['p_upper']:.4f}")

    res = pd.DataFrame(rows)
    for test, g in res.groupby("test"):
        res.loc[g.index, "p_fdr"] = _fdr(g.p_upper.values)
    res.to_csv(os.path.join(out_dir, "control_results.csv"), index=False)

    swr_io.write_settings(out_dir, {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": os.path.basename(__file__),
        "model": MODEL, "rois": rois, "seed": SEED,
        "n_config_relabel": n_config, "n_surrogate": n_surrogate,
        "n_contrast": n_contrast,
        "sign": "both matrices are dissimilarities; POSITIVE rho is predicted",
        "null_width_note": "an 8x8 RDM has 28 pairs, so a chance rho has "
                           "SD = 1/sqrt(27) = 0.19 by construction",
        "surrogate_window": "ripple time replaced by a uniform draw from the "
                            "same press window of the same event, duration kept",
        "fdr": "within each test family separately",
    })
    print(f"\nwritten to {out_dir}")
    if figures:
        _figure(res, rois, out_dir)
        plt.show(block=False)


def _summarise(test, roi, state, obs, null):
    sd = float(np.std(null))
    return {"test": test, "roi": roi, "state": state, "rho": float(obs),
            "n_perm": int(len(null)), "null_mean": float(np.mean(null)),
            "null_sd": sd, "z": (obs - np.mean(null)) / sd if sd > 0 else np.nan,
            "p_upper": float((null >= obs).mean())}


def _fdr(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0, 1)
    return out


def _figure(res, rois, out_dir):
    """Each control, every ROI, observed against its own null.

    Panels ~7 cm wide on the page; one dot per state, ordered A..D left to
    right within each ROI, so a D-specific effect is visible as a rightward
    outlier.
    """
    spec = [("C1_config_relabel", "a  Config relabel\n(is it about config identity?)"),
            ("C2_surrogate_window", "b  Surrogate window\n(does it need a ripple?)"),
            ("C4_post_minus_pre", "c  Post − pre contrast\n(is it after the press?)"),
            ("C3_pooled_BCD", "d  Pooled B+C+D\n(84 pairs, tighter null)")]
    fig, axes = plt.subplots(2, 2, figsize=(2 * 8.2 * CM, 2 * 7.0 * CM))
    for ax, (test, title) in zip(axes.ravel(), spec):
        sub = res[res.test == test]
        for i, roi in enumerate(rois):
            g = sub[sub.roi == roi]
            for _, row in g.iterrows():
                off = (0.0 if row.state == "B+C+D"
                       else 0.24 * (rrsa.STATES.index(row.state) - 1.5))
                ax.errorbar(i + off, row.null_mean, yerr=row.null_sd, fmt="_",
                            color="#b0b0b0", lw=1.8, capsize=2, ms=10, zorder=1)
                ax.plot(i + off, row.rho, "o", color=_roi_colour(roi), ms=7,
                        zorder=3, markeredgecolor="white", markeredgewidth=0.6)
                if row.p_upper < 0.05:
                    ax.annotate("*", (i + off, row.rho), fontsize=15,
                                ha="center", va="bottom",
                                textcoords="offset points", xytext=(0, 4),
                                color=OBSERVED_VALUE_COLOR, zorder=4)
                if row.state in ("D", "B+C+D"):
                    ax.annotate(row.state if row.state == "D" else "",
                                (i + off, row.rho), fontsize=7,
                                ha="center", va="top",
                                textcoords="offset points", xytext=(0, -9))
        ax.axhline(0, color="k", lw=1.0)
        ax.margins(y=0.16)          # headroom so a significance star stays inside
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels([roi_display(r) for r in rois], rotation=30,
                           ha="right")
        ax.set_ylabel("known_set fit (Spearman $\\rho$)")
        ax.set_title(title, fontsize=10, pad=5)
    fig.suptitle("Controls for the ripple-triggered plan fit\n"
                 "grey = null mean ± SD   ·   dots = observed, states A→D left "
                 "to right within each ROI   ·   * p < 0.05 uncorrected",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(out_dir, "controls_overview.png"),
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "controls_overview.pdf"),
                bbox_inches="tight")


def _roi_colour(roi):
    return get_roi_colour(roi)


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run, "figures_only": figures_only})
    else:
        run()
