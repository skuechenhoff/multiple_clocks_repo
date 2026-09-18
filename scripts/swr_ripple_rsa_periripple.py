#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hippocampal peri-ripple RSA, He et al. windows. The final figure.

He et al. analyse ripple-aligned compositional encoding in a PERI-RIPPLE
window of -250 to +250 ms around the ripple peak, against a NON-PERI-RIPPLE
baseline built from the two flanks (-750 to -250 ms and +250 to +750 ms)
combined, plus a ripple-shuffled surrogate that preserves ripple count and
temporal structure while destroying true ripple timing.

This applies that design to hippocampal single units and asks what, if
anything, is represented about the rewards:

    current_location   the reward just uncovered
    known_set          the locations uncovered so far
    full_abcd          the whole configuration

Event sets, both requested:
    a  explore    first discoveries only (the ripple-rate increase was
                  established on these); inter-uncover gap median 5.5 s
    b  all        every correct uncover, first traversals and repeats;
                  inter-uncover gap median 1.4 s

WHY THIS IS WORTH RUNNING AFTER A NULL RESULT. The ripple-internal analysis
cannot work: a ripple is 60 ms and 87% of (cell, ripple) observations contain
no spikes, so each condition mean rested on ~1 spike and split-half
reliability was ~0. A 500 ms peri-ripple window collects ~8x the spikes, and
the reliability-vs-window sweep rose monotonically (~0.02 at the ripple to
~0.09 at +-250 ms). The peri-vs-non-peri contrast keeps a real test of ripple
alignment. The cost is that 500 ms is ~8x the ripple, so this is
ripple-ALIGNED, not ripple-internal -- a weaker claim than the original one.

CLEARANCE. Only ripples whose full +-750 ms extent lies inside their own
inter-uncover interval are used, so both windows describe ONE condition.
Without it the design does not fit this task's timing at all: under the 1 s
post-press cap 100% of flanks reached outside their interval, and at the
all-uncovers event density the median gap (1.35 s) is shorter than the 1.5 s
window itself. Run on uncapped intervals the clearance keeps 71% of explore
ripples (3794) and 47% of all-uncover ones (14091).

    python scripts/swr_ripple_rsa_periripple.py run
    python scripts/swr_ripple_rsa_periripple.py run --n_surr=1000

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

HC_ROIS = ["HC_anterior", "HC_mid"]
MODELS = ["current_location", "known_set", "full_abcd"]
MODEL_LABELS = {"current_location": "current\nlocation",
                "known_set": "known\nset", "full_abcd": "full\nABCD"}
MODEL_COLOURS = {"current_location": "#5b9b8d", "known_set": "#23677E",
                 "full_abcd": "#DC673E"}
BAND_LABEL = {"peri": "peri-ripple (±250 ms)",
              "nonperi": "non-peri-ripple (both flanks, 1000 ms)",
              "nonperi_late": "non-peri, duration-matched (+250–750 ms)"}
BAND_ALPHA = {"peri": 1.0, "nonperi": 0.42, "nonperi_late": 0.7}
# Every bar gets its OWN x position, models separated by a gap. Packing the
# three windows inside one model tick left no room for their labels, and alpha
# alone was too weak an encoding -- the triplets read as if they were the four
# uncovers, which they are not (the uncovers are a dimension INSIDE the RDM).
BAND_X = {"peri": 0, "nonperi": 1, "nonperi_late": 2}
GROUP_W = 4          # 3 bars + 1 blank between models
# each model is one GROUP of three bars, one per window. Alpha alone was too
# weak an encoding -- the three bars in a group share the model's colour and
# read as if they were the four uncovers. The window is now spelled out under
# every bar and the model name sits above the group.
BAND_TICK = {"peri": "peri", "nonperi": "non-peri", "nonperi_late": "matched"}
BAND_OFF = {"peri": -0.24, "nonperi": 0.0, "nonperi_late": 0.24}
# kept short on purpose: rotated row labels any longer than this spill
# vertically into the neighbouring row
EVENT_LABEL = {"explore": "a  explore", "all": "b  all uncovers"}


def _out(data_root=None):
    d = os.path.join(swr_io.derivatives_dir(data_root), "group", "swr",
                     f"ripple_rsa_periripple_{STAMP}")
    os.makedirs(d, exist_ok=True)
    return d


def _events(bd, which):
    if which == "explore":
        return rrsa.discovery_events(bd), "interval", None
    return rrsa.all_uncover_events(bd), "uncovers", (0.0, 1e9)


def _fit(cache, rois):
    """rho for every (roi, model) -- the whole estimator, so surrogate draws
    call exactly this."""
    pk = rrsa.patterns_uncover(cache, collapse_phase=True)
    M = rrsa.model_rdms_flat()
    out = {}
    for roi in rois:
        rdm = rrsa.rdm_for_flat(pk, roi, normalise="zscore")[0]
        for name, mod in M.items():
            out[(roi, name)] = rrsa.fit_rho_flat(rdm, mod, family="all")
    return out


def _reliability(cache, roi, rng, n_splits=10):
    iu = np.triu_indices(rrsa.N_COND, 1)
    vals = []
    for _ in range(n_splits):
        perms = {id(b): rng.permutation(b["rates"].shape[1]) for b in cache}
        v = []
        for h in (0, 1):
            sub = []
            for b in cache:
                i = perms[id(b)][h::2]
                d = {k: x for k, x in b.items()
                     if k not in ("rates", "ci", "si", "pi")}
                d["rates"] = b["rates"][:, i]
                d["ci"], d["si"] = b["ci"][i], b["si"][i]
                if "pi" in b:
                    d["pi"] = b["pi"][i]
                sub.append(d)
            pk = rrsa.patterns_uncover(sub, collapse_phase=True)
            v.append(rrsa.rdm_for_flat(pk, roi, normalise="zscore")[0][iu])
        a, b2 = v
        ok = np.isfinite(a) & np.isfinite(b2)
        if ok.sum() > 10:
            vals.append(float(stats.spearmanr(a[ok], b2[ok]).correlation))
    return float(np.mean(vals)) if vals else np.nan


def run(bundle=None, data_root=None, n_surr=500, out=None):
    D = rrsa.DECISIONS
    out_dir = out or _out(data_root)
    swr_io.start_log(out_dir, "swr_ripple_rsa_periripple")
    bundle_dir = bundle or rrsa.default_bundle_dir(data_root)
    np.random.seed(SEED)

    bd = rrsa.load_bundle(bundle_dir, pad_s=D["pad_s"])
    tab = rrsa.cell_roi_table()
    rois = [r for r in HC_ROIS if (tab.roi == r).sum() >= 5]
    tab = tab[tab.roi.isin(rois)]
    spikes = rrsa.load_spike_times(verbose=False)
    print(f"hippocampal units: {len(tab)} in {rois}")

    rows, nulls, meta = [], {}, {}
    for which in ("explore", "all"):
        ev, scheme, win = _events(bd, which)
        caches = dict(zip(rrsa.BANDS, rrsa.cache_band_rates(
            bd, ev, tab, spikes, win, band="both", scheme=scheme)))
        n_rip = sum(b["rates"].shape[1] for b in caches["peri"])
        rng = np.random.default_rng(SEED)
        meta[which] = {"n_events": int(len(ev)), "n_ripples_used": int(n_rip),
                       "scheme": scheme,
                       "reliability": {b: _reliability(caches[b], rois[0], rng)
                                       for b in rrsa.BANDS}}
        print(f"\n{which}: {len(ev)} uncovers, {n_rip} clearance-ok ripples")
        print(f"  split-half reliability ({rois[0]}): "
              + "  ".join(f"{b} {meta[which]['reliability'][b]:+.3f}"
                          for b in rrsa.BANDS))

        obs = {b: _fit(caches[b], rois) for b in rrsa.BANDS}
        null = {(b,) + k: [] for b in rrsa.BANDS for k in obs[b]}
        rng = np.random.default_rng(SEED)
        for i in range(n_surr):
            sp, sn, sl = rrsa.cache_band_rates(bd, ev, tab, spikes, win,
                                               band="both", scheme=scheme,
                                               surrogate_rng=rng)
            for b, c in zip(rrsa.BANDS, (sp, sn, sl)):
                for k, v in _fit(c, rois).items():
                    null[(b,) + k].append(v)
            if (i + 1) % 50 == 0:
                print(f"    surrogate {i + 1}/{n_surr}", flush=True)

        for b in rrsa.BANDS:
            for (roi, model), o in obs[b].items():
                s = np.array(null[(b, roi, model)], float)
                s = s[np.isfinite(s)]
                nulls[f"{which}|{b}|{roi}|{model}"] = s
                rows.append({
                    "events": which, "band": b, "roi": roi, "model": model,
                    "rho": o, "surr_mean": s.mean() if len(s) else np.nan,
                    "surr_sd": s.std() if len(s) else np.nan,
                    "z_surr": ((o - s.mean()) / s.std()
                               if len(s) and s.std() > 0 and np.isfinite(o)
                               else np.nan),
                    "p_surr": (float((s >= o - 1e-12).mean())
                               if len(s) and np.isfinite(o) else np.nan),
                    "n_surr": len(s), "n_ripples": int(n_rip)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "periripple_rsa.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "nulls.npz"),
                        keys=np.array(list(nulls)),
                        null=np.array([nulls[k] for k in nulls], dtype=object))

    print("\n=== peri-ripple vs non-peri-ripple, hippocampal units ===")
    print(res.pivot_table(index=["events", "roi", "model"], columns="band",
                          values=["rho", "z_surr"]).round(3).to_string())
    _report_contrast(res)

    _figure(res, nulls, rois, out_dir)
    _overview(res, meta, rois, out_dir, n_surr)
    swr_io.write_settings(out_dir, {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": os.path.basename(__file__), "bundle": bundle_dir,
        "windows": {"peri_s": [-rrsa.PERI_HALF_S, rrsa.PERI_HALF_S],
                    "nonperi_s": [[-rrsa.NONPERI_OUTER_S, -rrsa.PERI_HALF_S],
                                  [rrsa.PERI_HALF_S, rrsa.NONPERI_OUTER_S]]},
        "source": "He et al. peri-ripple windows",
        "clearance": "full +-750 ms must lie inside the inter-uncover interval",
        "rois": rois, "models": MODELS, "n_surr": n_surr, "seed": SEED,
        "conditions": "32 = 8 configs x 4 uncovers, phase pooled",
        "normalise": D["normalise"], "pad_s": D["pad_s"],
        "null": "ripple-shuffled surrogate: pseudo-peaks matched in number per "
                "event, drawn from the same clearance-respecting interval",
        "meta": meta})
    print(f"\nwritten to {out_dir}")


def _report_contrast(res):
    print("\n  peri MINUS non-peri (the ripple-alignment contrast):")
    p = res.pivot_table(index=["events", "roi", "model"], columns="band",
                        values="rho")
    p["peri_minus_nonperi"] = p["peri"] - p["nonperi"]
    print(p.round(3).to_string())


def _figure(res, nulls, rois, out_dir):
    ev_order = ["explore", "all"]
    fig, axes = plt.subplots(len(ev_order), len(rois),
                             figsize=(len(rois) * 7.6 * CM,
                                      len(ev_order) * 5.8 * CM),
                             squeeze=False, sharey=True)
    for r, which in enumerate(ev_order):
        for c, roi in enumerate(rois):
            ax = axes[r][c]
            for i, model in enumerate(MODELS):
                for b in rrsa.BANDS:
                    s = nulls.get(f"{which}|{b}|{roi}|{model}", np.array([]))
                    s = np.asarray(s, float)
                    s = s[np.isfinite(s)]
                    x = i * GROUP_W + BAND_X[b]
                    if len(s) > 2:
                        v = ax.violinplot([s], positions=[x], widths=0.85,
                                          showextrema=False)
                        for bd_ in v["bodies"]:
                            bd_.set_facecolor("#d9d9d9")
                            bd_.set_alpha(0.9)
                            bd_.set_edgecolor("none")
                    g = res[(res.events == which) & (res.band == b)
                            & (res.roi == roi) & (res.model == model)]
                    if not len(g) or not np.isfinite(g.rho.iloc[0]):
                        continue
                    col = MODEL_COLOURS[model]
                    ax.plot([x - 0.40, x + 0.40], [g.rho.iloc[0]] * 2,
                            color=col, lw=2.6, solid_capstyle="butt",
                            zorder=5, alpha=BAND_ALPHA[b])
                    if g.p_surr.iloc[0] < 0.05:
                        ax.annotate("*", (x, g.rho.iloc[0]), ha="center",
                                    va="bottom", fontsize=13, zorder=6,
                                    textcoords="offset points", xytext=(0, 2),
                                    color=col)
            ax.axhline(0, color="k", lw=0.9)
            for i in range(1, len(MODELS)):
                ax.axvline(i * GROUP_W - 1, color="#cccccc", lw=0.9, zorder=0)
            ax.set_xticks([i * GROUP_W + BAND_X[b] for i in range(len(MODELS))
                           for b in rrsa.BANDS])
            ax.set_xticklabels([BAND_TICK[b] for _ in MODELS
                                for b in rrsa.BANDS], fontsize=6.5,
                               rotation=90)
            ax.tick_params(axis="x", length=2, pad=1)
            for i, mdl in enumerate(MODELS):
                ax.annotate(MODEL_LABELS[mdl],
                            xy=(i * GROUP_W + 1, -0.80),
                            xycoords=("data", "axes fraction"),
                            ha="center", va="top", fontsize=8,
                            linespacing=0.95,
                            fontweight="bold", color=MODEL_COLOURS[mdl],
                            annotation_clip=False)
            ax.set_xlim(-1, (len(MODELS) - 1) * GROUP_W + 3)
            if r == 0:
                ax.set_title(roi_display(roi), fontsize=11, pad=6,
                             color=get_roi_colour(roi))
            if c == 0:
                ax.set_ylabel("partial $\\rho$", labelpad=2)
                # the event-set label sits OUTSIDE the ylabel, not above the
                # axes: above it collided with the ROI titles on the top row
                ax.annotate(EVENT_LABEL[which], xy=(-0.46, 0.5),
                            xycoords="axes fraction", rotation=90,
                            ha="center", va="center", fontsize=9.5,
                            fontweight="bold")
    h = [plt.Line2D([], [], color="#444444", lw=2.6, alpha=BAND_ALPHA[b],
                    label=BAND_LABEL[b]) for b in rrsa.BANDS]
    h.append(plt.Line2D([], [], color="#d9d9d9", lw=7,
                        label="ripple-shuffled null"))
    fig.legend(handles=h, frameon=False, fontsize=7.5, ncol=2,
               loc="lower center", bbox_to_anchor=(0.55, -0.015))
    fig.suptitle(
        "What hippocampal firing represents around ripples (He et al. windows)\n"
        "each model = 3 bars, one per WINDOW (not per uncover)   ·   "
        "grey = ripple-shuffled null   ·   * p < 0.05",
        fontsize=11, y=0.985)
    fig.subplots_adjust(top=0.79, bottom=0.38, left=0.205, right=0.985,
                        hspace=1.15, wspace=0.10)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"periripple_hippocampus.{e}"),
                    bbox_inches="tight")
    plt.close(fig)


def _overview(res, meta, rois, out_dir, n_surr):
    sig = res[res.p_surr < 0.05]
    o = {"created": datetime.now().isoformat(timespec="seconds"),
         "design": "He et al. peri-ripple (+-250 ms) vs non-peri-ripple "
                   "(+-250-750 ms), hippocampal single units",
         "caveat": "500 ms is ~8x the 60 ms ripple, so this is ripple-ALIGNED, "
                   "not ripple-internal",
         "clearance": "only ripples whose full +-750 ms lies inside their "
                      "inter-uncover interval",
         "rois": rois, "n_surr": n_surr, "data": meta,
         "n_significant_p05_uncorrected": int(len(sig)),
         "n_tests": int(res.p_surr.notna().sum()),
         "significant": sig[["events", "band", "roi", "model", "rho",
                             "z_surr", "p_surr"]].round(4)
                           .to_dict(orient="records"),
         "results": res[["events", "band", "roi", "model", "rho", "z_surr",
                         "p_surr"]].round(4).to_dict(orient="records")}
    with open(os.path.join(out_dir, "results_overview.json"), "w") as f:
        json.dump(o, f, indent=2)


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
