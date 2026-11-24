#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig 3d check: do the ventral and dorsal mPFC cell groups really prefer
different future lags (30 deg vs 60 deg)?  (2026-11-23)

The manuscript tests each group against zero at each lag
(`cell_fMRI_angle_match.py`, lagwise_vs_zero.csv). It does not test the
split itself. This script adds that, for one or more master runs
(e.g. the old 'mode' and the new 'per_repeat' rate-map estimator):

  1. SHIFT. Per cell d = z(r60) - z(r30) (Fisher z). Statistic
     S = mean d(dorsal) - mean d(ventral), at three units:
        cell     cells pooled (the manuscript's unit)
        site     cells on one recording site (shared MNI coordinate)
                 averaged first -- sites are the effective N
        session  cells of one session within a group averaged first
     Tests: Welch t, one-sided (dorsal > ventral); and an EXACT site
     permutation: every way of choosing which sites are 'dorsal' (group
     sizes in sites kept), the same `shift_stat` function as the
     empirical value.
  2. PEAK STABILITY. Bootstrap over sites within each group (2000 draws,
     seed 42): pooled unweighted profile -> argmax lag, as in
     `cell_fMRI_angle_match.group_angle`.
  3. Within-group peak vs its neighbour: paired t on z(r60) - z(r30).

Splits: the reported PC1 median split (labels read from
final_splits_per_cell.csv), and an MNI-z median split of the same in-mask
cells (same rule, `cell_fMRI_angle_match.make_labels`) as robustness.

    python scripts/fig3d_gradient_split_check.py <master_dir> [<master_dir> ...]

Output: group/fig3d_gradient_check_<date>/ (result.json, split_check.png)

@author: Svenja Kuchenhoff
"""
import os
import sys
import json
import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from cell_fMRI_angle_match import make_labels, L, angles_to_colours

GROUP = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group"
OUT = os.path.join(GROUP, f"fig3d_gradient_check_{datetime.date.today()}")
N_BOOT = 2000
SEED = 42
I30, I60 = list(L).index(30), list(L).index(60)
CM = 1 / 2.54


def fz(r):
    return np.arctanh(np.clip(r, -0.999, 0.999))


def load(master_dir):
    c = pd.read_csv(os.path.join(master_dir, "per_cell_master.csv"))
    s = pd.read_csv(os.path.join(master_dir, "final_splits",
                                 "final_splits_per_cell.csv"))
    assert (c.neuron.values == s.neuron.values).all()
    R = c[[f"r_lag{a:03d}_noctrl" for a in L]].to_numpy(float)
    site = c[["MNI_x", "MNI_y", "MNI_z"]].round(3).astype(str).agg("_".join, axis=1)
    inm = c.in_gradient_mask.to_numpy(bool)
    z_lab = make_labels(c.MNI_z.to_numpy(float), inm, 2)
    labels = {"pc1_median (reported)": s.pc1_ventral_dorsal_group.to_numpy(),
              "MNI_z_median": np.array([{"g0": "ventral", "g1": "dorsal"}
                                        .get(x, "outside") for x in z_lab])}
    return R, site.to_numpy(), c.subject_id.to_numpy(), labels


def shift_stat(dsum, n, dorsal):
    """Cell-unit S for site-level sums. dorsal: (n_assign, n_sites) bool."""
    dorsal = np.atleast_2d(dorsal)
    m_d = (dorsal @ dsum) / (dorsal @ n)
    m_v = ((~dorsal) @ dsum) / ((~dorsal) @ n)
    return m_d - m_v


def shift_stat_site(dmean, dorsal):
    """Site-unit S: mean of site means, dorsal minus ventral."""
    dorsal = np.atleast_2d(dorsal)
    return (dorsal @ dmean) / dorsal.sum(1) - ((~dorsal) @ dmean) / (~dorsal).sum(1)


def welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return dict(diff=float(np.mean(a) - np.mean(b)), t=float(t),
                p_one_sided=float(p / 2 if t > 0 else 1 - p / 2),
                n_dorsal=int(len(a)), n_ventral=int(len(b)))


def paired(z, i_hi, i_lo):
    d = z[:, i_hi] - z[:, i_lo]
    d = d[np.isfinite(d)]
    t, p = stats.ttest_1samp(d, 0)
    return dict(mean=float(d.mean()), t=float(t), p_two_sided=float(p), n=int(len(d)))


def peak(R):
    return int(L[np.nanargmax(np.nanmean(R, 0))])


def boot_peaks(R, site, sel, rng):
    sites = np.unique(site[sel])
    rows = {s: np.where(sel & (site == s))[0] for s in sites}
    out = np.empty(N_BOOT, int)
    for b in range(N_BOOT):
        take = rng.choice(sites, sites.size, replace=True)
        out[b] = peak(R[np.concatenate([rows[s] for s in take])])
    return out


def check_split(R, site, subj, lab, rng):
    z = fz(R)
    d = z[:, I60] - z[:, I30]
    inm = lab != "outside"
    res = {"n_cells": {g: int((lab == g).sum())
                       for g in ("ventral", "dorsal", "outside")},
           "n_sites": {g: int(np.unique(site[lab == g]).size)
                       for g in ("ventral", "dorsal")},
           "n_sessions": {g: int(np.unique(subj[lab == g]).size)
                          for g in ("ventral", "dorsal")}}
    # profiles (unweighted mean +- SEM across cells), exactly what is tested
    res["profile"] = {g: dict(mean=np.nanmean(R[lab == g], 0).tolist(),
                              sem=stats.sem(R[lab == g], 0, nan_policy="omit")
                              .tolist(), peak_deg=peak(R[lab == g]))
                      for g in ("ventral", "dorsal", "outside")}
    # 1. shift, three units
    df = pd.DataFrame(dict(d=d, g=lab, site=site, subj=subj))[inm]
    by_site = df.groupby(["g", "site"]).d.mean()
    by_sess = df.groupby(["g", "subj"]).d.mean()
    res["shift_60_minus_30"] = {
        "cell": welch(df.d[df.g == "dorsal"], df.d[df.g == "ventral"]),
        "site": welch(by_site["dorsal"], by_site["ventral"]),
        "session": welch(by_sess["dorsal"], by_sess["ventral"]),
        "group_means_cell": {g: float(df.d[df.g == g].mean())
                             for g in ("ventral", "dorsal")}}
    # exact site permutation (same functions for empirical and null)
    s_tab = df.groupby("site").agg(dsum=("d", "sum"), n=("d", "size"),
                                   dmean=("d", "mean"),
                                   g=("g", lambda x: x.iloc[0]))
    assert (df.groupby("site").g.nunique() == 1).all()
    obs = (s_tab.g == "dorsal").to_numpy()
    k, n_sites = int(obs.sum()), len(s_tab)
    combos = np.array(list(combinations(range(n_sites), k)))
    A = np.zeros((len(combos), n_sites), bool)
    A[np.arange(len(combos))[:, None], combos] = True
    for unit, fn, args in (("cell", shift_stat, (s_tab.dsum.values, s_tab.n.values)),
                           ("site", shift_stat_site, (s_tab.dmean.values,))):
        emp = float(fn(*args, obs)[0])
        null = fn(*args, A)
        res["shift_60_minus_30"][f"site_permutation_{unit}_unit"] = dict(
            S=emp, p_one_sided=float(np.mean(null >= emp - 1e-12)),
            n_assignments=int(len(A)))
    # 2. peak stability (site bootstrap)
    pk = {g: boot_peaks(R, site, lab == g, rng) for g in ("ventral", "dorsal")}
    res["peak_bootstrap"] = {
        g: {int(a): float(np.mean(pk[g] == a)) for a in L if np.mean(pk[g] == a) > 0}
        for g in pk}
    res["peak_bootstrap"]["P(ventral=30 and dorsal=60)"] = float(
        np.mean((pk["ventral"] == 30) & (pk["dorsal"] == 60)))
    # 3. within-group 60 vs 30 (cell unit and session unit)
    res["within_group_60_minus_30"] = {}
    for g in ("ventral", "dorsal"):
        zs = pd.DataFrame(z[lab == g]).groupby(subj[lab == g]).mean().to_numpy()
        res["within_group_60_minus_30"][g] = dict(
            cell=paired(z[lab == g], I60, I30), session=paired(zs, I60, I30))
    return res


def figure(results, out):
    tags = list(results)
    fig, axes = plt.subplots(len(tags), 3, figsize=(20 * CM, 8 * CM * len(tags)),
                             squeeze=False)
    for row, tag in zip(axes, tags):
        r = results[tag]["pc1_median (reported)"]
        ax = row[0]
        for g, col in (("outside", "0.6"),
                       ("ventral", angles_to_colours([30])[0]),
                       ("dorsal", angles_to_colours([60])[0])):
            p = r["profile"][g]
            m, e = np.array(p["mean"]), np.array(p["sem"])
            ax.fill_between(L, m - e, m + e, color=col, alpha=0.25, lw=0)
            ax.plot(L, m, color=col, lw=1.4,
                    label=f"{g} (n = {r['n_cells'][g]}"
                          f"{' cells' if g == 'outside' else ', ' + str(r['n_sites'][g]) + ' sites'}"
                          f"), peak {p['peak_deg']}°")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(L[::2])
        ax.set_xlabel("future lag (deg)")
        ax.set_ylabel("CV r (mean ± SEM, cells)")
        ax.set_title(f"{tag}: PC1 median split", fontsize=10)
        ax.legend(fontsize=6.5, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.28))
        # shift bars, both splits
        ax = row[1]
        for i, split in enumerate(results[tag]):
            s = results[tag][split]
            for j, g in enumerate(("ventral", "dorsal")):
                w = s["within_group_60_minus_30"][g]["cell"]
                ci = stats.t.ppf(0.975, w["n"] - 1) * abs(w["mean"] / w["t"])
                ax.bar(i + (j - 0.5) * 0.38, w["mean"], 0.34, yerr=ci,
                       color=angles_to_colours([30 if g == "ventral" else 60])[0],
                       error_kw=dict(lw=0.8, capsize=2),
                       label=g if i == 0 else None)
            sp = s["shift_60_minus_30"]
            ax.text(i, 1.0, f"t p = {sp['cell']['p_one_sided']:.3f}"
                    f"\nperm p = {sp['site_permutation_cell_unit']['p_one_sided']:.3f}",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=7)
        ax.set_title("dorsal > ventral, one-sided:\ncell t-test (t), site permutation"
                     " (perm)", fontsize=8, pad=24)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(len(results[tag])))
        ax.set_xticklabels([x.replace(" (reported)", "\n(reported)")
                            .replace("_median", " median") for x in results[tag]])
        ax.set_ylabel("z(r 60°) − z(r 30°)\n(cell mean, 95% CI)")
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        # peak bootstrap
        ax = row[2]
        pb = r["peak_bootstrap"]
        for j, g in enumerate(("ventral", "dorsal")):
            p = np.array([pb[g].get(int(a), pb[g].get(str(int(a)), 0)) for a in L])
            ax.bar(L + (j - 0.5) * 11, p, 10,
                   color=angles_to_colours([30 if g == "ventral" else 60])[0],
                   label=g)
        ax.set_xticks(L[::2])
        ax.set_xlabel("peak lag of the pooled profile (deg)")
        ax.set_ylabel("fraction of site bootstraps")
        ax.set_title(f"P(ventral 30° & dorsal 60°) = "
                     f"{pb['P(ventral=30 and dorsal=60)']:.2f}", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    for ax in axes.flat:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "split_check.png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for master in sys.argv[1:]:
        tag = os.path.basename(master.rstrip("/")).split("_", 2)[-1]
        R, site, subj, labels = load(master)
        results[tag] = {}
        for split, lab in labels.items():
            rng = np.random.default_rng(SEED)
            results[tag][split] = check_split(R, site, subj, lab, rng)
        results[tag]["_master"] = master
    masters = {t: results[t].pop("_master") for t in results}
    json.dump(dict(masters=masters, n_boot=N_BOOT, seed=SEED, results=results),
              open(os.path.join(OUT, "result.json"), "w"), indent=1)
    figure(results, OUT)
    for tag in results:
        for split, r in results[tag].items():
            sh = r["shift_60_minus_30"]
            print(f"\n== {tag} | {split} | cells {r['n_cells']} sites {r['n_sites']}"
                  f" sessions {r['n_sessions']}")
            print("  peaks:", {g: r["profile"][g]["peak_deg"] for g in r["profile"]})
            print("  shift d(60-30) group means:", {k: round(v, 3) for k, v in
                                                     sh["group_means_cell"].items()})
            for u in ("cell", "site", "session"):
                print(f"  shift {u:8s} diff {sh[u]['diff']:+.3f} t {sh[u]['t']:.2f}"
                      f" p1 {sh[u]['p_one_sided']:.3f}  (n {sh[u]['n_dorsal']}/{sh[u]['n_ventral']})")
            for u in ("cell", "site"):
                pp = sh[f"site_permutation_{u}_unit"]
                print(f"  site-perm ({u} unit) S {pp['S']:+.3f} p1 {pp['p_one_sided']:.4f}"
                      f" ({pp['n_assignments']} assignments)")
            print("  peak bootstrap:", r["peak_bootstrap"])
            for g, w in r["within_group_60_minus_30"].items():
                print(f"  {g} 60-30: cell {w['cell']['mean']:+.3f} p {w['cell']['p_two_sided']:.3f}"
                      f" | session {w['session']['mean']:+.3f} p {w['session']['p_two_sided']:.3f}")
    print("\nwrote", OUT)
