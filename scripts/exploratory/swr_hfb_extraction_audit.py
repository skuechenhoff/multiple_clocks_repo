#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Which averaging choice moves the ripple-locked OFC vs mPFC result?

Two extractions of the same thing -- ripple-locked HFB, every ripple, OFC
(mOFC + lOFC) and mPFC, real minus shifted null, per session -- gave the same
curves but different p for the timing test (mPFC − OFC direction index):
`ripple_locked_hfb_2026-09-18` (the region run) p = 0.032 / paired 0.015,
`swr_hfb_ripple_coupling.py phases` p = 0.072 / paired 0.007. Both start from
the SAME bundle: the same ripple detections and the same HFB envelopes.
Nothing is re-detected here. They differ only in how epochs are averaged:

  weighting  region: each hippocampal channel's ripples separately, a trace per
             (HC channel x cortical derivation) pair, pairs averaged with equal
             weight. phases: ripples pooled over HC channels, per ripple the
             mean over the ROI's derivations, ripples averaged.
  artifact   region: the ripple's HC channel AND the cortical derivation clean
             for the +-1 s window. phases: the cortical derivations only.
  shifts     region: 8 shifted nulls (seed 42). phases: the conditions run's 6.
  dedup      phases merges ripples whose peaks round to the same millisecond;
             a +-50 ms cross-channel merge is added as a further variant.

Each variant switches ONE of these, starting from the region method; the
first variant must reproduce the region run's numbers, the fifth the phases
run's. Unit = session throughout.

Usage:
    conda activate env_multiple_clocks
    python scripts/exploratory/swr_hfb_extraction_audit.py run --show=False

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import pickle
import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))   # repo root
import mc.analyse.swr_io as swr_io
import mc.analyse.swr_bundle as sb
import mc.plotting.ripple_figures as rfig

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))    # scripts/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # exploratory/
from swr_ripple_hfb_conditions import _clean_mask, SHIFT_RANGE_S, HALF_S
from swr_hfb_direction import (direction, pair_test, sliding_curves, _p_txt,
                               PARTITION, ROI_COL, REMOTE_MS)

try:
    import fire
except ImportError:
    fire = None

ANALYSIS_NAME = "hfb_extraction_audit"
MIN_RIPPLES = 20
ROIS = {"OFC": ("mOFC", "lOFC"), "mPFC": ("mPFC",)}
DEDUP_MS = 50.0

# (key, label, weighting, hc_clean, shifts, dedup)
VARIANTS = [
    ("region", "region method", "pair", True, "region", None),
    ("cortical_only", "+ artifact: cortical only", "pair", False, "region",
     None),
    ("six_shifts", "+ 6 conditions-run shifts", "pair", True, "cond", None),
    ("ripple_weighted", "+ ripples pooled, ripple-weighted", "ripple", True,
     "region", "exact"),
    ("phases", "phases method (all three)", "ripple", False, "cond", "exact"),
    ("phases_dedup50", f"phases + ±{DEDUP_MS:.0f} ms dedup", "ripple", False,
     "cond", "50ms"),
]


def _shifts(n, seed=42):
    rng = np.random.default_rng(seed)
    lo, hi = SHIFT_RANGE_S
    return [0.0] + [float(s) for s in rng.uniform(lo, hi, n)
                    * rng.choice([-1, 1], n)]


def _dedup(t, ch, how):
    """Pooled ripple list. 'exact': peaks equal at 1 ms (as the phases run);
    '50ms': a ripple within DEDUP_MS of a kept ripple on ANOTHER channel is the
    same event and dropped (the earlier one is kept)."""
    o = np.argsort(t, kind="stable")
    t, ch = t[o], ch[o]
    if how == "exact":
        _, first = np.unique(np.round(t, 3), return_index=True)
        return t[first], ch[first]
    keep = np.ones(len(t), bool)
    for i in range(1, len(t)):
        j = i - 1
        while j >= 0 and t[i] - t[j] <= DEDUP_MS / 1000.0:
            if keep[j] and ch[j] != ch[i]:
                keep[i] = False
                break
            j -= 1
    return t[keep], ch[keep]


def session_curves(H, idx, masks, cx, hc_ripples, fs, n, off, shifts,
                   weighting, hc_clean, dedup):
    """{roi: real − null curve} for one session under one variant."""
    w = -off[0]
    out = {}
    if weighting == "pair":
        real = {r: [] for r in ROIS}
        null = {r: [] for r in ROIS}
        for h, t0 in hc_ripples.items():
            if len(t0) < MIN_RIPPLES:
                continue
            for sh in shifts:
                c = np.round((t0 + sh) * fs).astype(int)
                c = c[(c - w >= 0) & (c + w < n)]
                if len(c) < MIN_RIPPLES:
                    continue
                win = c[:, None] + off[None, :]
                hc_ok = masks[h][win].all(1) if hc_clean \
                    else np.ones(len(c), bool)
                for roi, pids in cx.items():
                    for x in pids:
                        ok = hc_ok & masks[x][win].all(1)
                        if ok.sum() < MIN_RIPPLES:
                            continue
                        (real if sh == 0.0 else null)[roi].append(
                            H[idx[x]][win[ok]].mean(0))
        for roi in ROIS:
            if real[roi] and null[roi]:
                out[roi] = np.mean(real[roi], 0) - np.mean(null[roi], 0)
        return out
    # ripple-weighted: pooled over HC channels
    t = np.concatenate(list(hc_ripples.values()))
    ch = np.concatenate([[h] * len(v) for h, v in hc_ripples.items()])
    t, ch = _dedup(t, ch, dedup)
    for roi, pids in cx.items():
        if not pids:
            continue
        means = []
        for sh in shifts:
            c = np.round((t + sh) * fs).astype(int)
            inb = (c - w >= 0) & (c + w < n)
            cc = np.where(inb, c, w)
            win = cc[:, None] + off[None, :]
            ok_r = inb.copy()
            if hc_clean:
                ok_r &= np.array([masks[h_][wi].all() for h_, wi
                                  in zip(ch, win)])
            s = np.zeros(win.shape)
            k = np.zeros(len(c))
            for x in pids:
                okx = masks[x][win].all(1) & ok_r
                s[okx] += H[idx[x]][win[okx]]
                k += okx
            if (k > 0).sum() < MIN_RIPPLES:
                break
            means.append((s[k > 0] / k[k > 0, None]).mean(0))
        if len(means) == len(shifts):
            out[roi] = means[0] - np.mean(means[1:], 0)
    return out


def run(bundle=None, partition_dir=None, out_dir=None, show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    R = swr_io.get_data_root()
    G = os.path.join(swr_io.derivatives_dir(R), "group", "swr")
    b_dir = bundle or os.path.join(G, "bundle_v2")
    partition_dir = partition_dir or os.path.join(G, PARTITION)
    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(G, f"{ANALYSIS_NAME}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(b_dir, "swr_bundle.pkl"), "rb") as f:
        B = pickle.load(f)
    rip, hiv, pairs = B["ripples"], B["hfb_intervals"], B["pairs"]
    hp = B["hfb_pairs"][~B["hfb_pairs"].excluded.fillna(False)].copy()
    key = pairs[["session", "pair_id", "same_probe_as_hpc"]].drop_duplicates()
    hp = hp.merge(key, on=["session", "pair_id"], how="left")
    hp["same_shaft"] = hp.same_probe_as_hpc.fillna(False).astype(bool)
    A = pd.read_csv(os.path.join(partition_dir, "derivation_assignment.csv"))
    part = A.set_index(["session", "cx_pair"])["roi_part"]
    hp["roi"] = pd.MultiIndex.from_arrays([hp.session, hp.pair_id]).map(part)
    fs = float(B["hfb_index"].out_fs.iloc[0])
    w = int(HALF_S * fs)
    off = np.arange(-w, w)
    t_ms = off / fs * 1000.0
    shift_sets = {"region": _shifts(8), "cond": _shifts(6)}

    curves = {v[0]: {r: {} for r in ROIS} for v in VARIANTS}
    n_rip = []
    for sess in sorted(hp.session.unique()):
        g = hp[hp.session == sess]
        hc = g[g.roi_family == "HPC"]
        cxg = g[(g.roi_family != "HPC") & ~g.same_shaft]
        cx = {r: cxg[cxg.roi.isin(fam)].pair_id.tolist()
              for r, fam in ROIS.items()}
        if not len(hc) or not any(cx.values()):
            continue
        D = sb.load_hfb(b_dir, sess, arrays=["hfb"])
        H, ids = D["hfb"], D["pair_ids"]
        n = H.shape[1]
        idx = {p: i for i, p in enumerate(ids)}
        cx = {r: [p for p in v if p in idx] for r, v in cx.items()}
        masks = {p: _clean_mask(hiv[(hiv.session == sess)
                                    & (hiv.pair_id == p)]
                                [["start_s", "stop_s"]].to_numpy(float), n, fs)
                 for p in g.pair_id}
        hc_rip = {h: rip.loc[(rip.session == sess) & (rip.pair_id == h),
                             "t_peak_s"].to_numpy(float) for h in hc.pair_id}
        tt = np.concatenate(list(hc_rip.values()))
        cc = np.concatenate([[h] * len(v) for h, v in hc_rip.items()])
        n_rip.append(dict(session=int(sess), n_hc=len(hc), rows=len(tt),
                          exact=len(_dedup(tt, cc, "exact")[0]),
                          dedup50=len(_dedup(tt, cc, "50ms")[0])))
        for key_, _, wt, hcc, shk, dd in VARIANTS:
            c = session_curves(H, idx, masks, cx, hc_rip, fs, n, off,
                               shift_sets[shk], wt, hcc, dd)
            for roi, v in c.items():
                curves[key_][roi][int(sess)] = v
        print(f"  s{int(sess):02d} done", end="\r")

    # ---- the numbers per variant ----------------------------------------
    rows = []
    ref = curves["region"]
    test = np.abs(t_ms) < REMOTE_MS
    for key_, label, *_ in VARIANTS:
        reg = curves[key_]
        dirs = {r: pd.Series({u: d[0] for u, d in
                              direction(reg[r], t_ms).items()}) for r in ROIS}
        tst = pair_test(dirs["mPFC"], dirs["OFC"])
        row = dict(variant=key_, label=label, n_OFC=len(reg["OFC"]),
                   n_mPFC=len(reg["mPFC"]), diff=tst["diff"],
                   p_unpaired=tst["p_perm"], n_paired=tst["n_paired"],
                   paired_diff=tst.get("paired_diff"),
                   p_paired=tst.get("p_paired"))
        for r in ROIS:
            _, Xs = sliding_curves(reg, r, t_ms)
            row[f"peak_{r}"] = float(Xs.mean(0)[test].max())
            # how far each session's curve moved from the region method
            sh = sorted(set(reg[r]) & set(ref[r]))
            row[f"r_vs_region_{r}"] = float(np.median(
                [np.corrcoef(reg[r][u][test], ref[r][u][test])[0, 1]
                 for u in sh]))
            dref = pd.Series({u: d[0] for u, d in
                              direction(ref[r], t_ms).items()})
            row[f"dir_r_vs_region_{r}"] = float(np.corrcoef(
                dirs[r][sh], dref[sh])[0, 1])
        rows.append(row)
    T = pd.DataFrame(rows)
    N = pd.DataFrame(n_rip)
    pd.set_option("display.width", 200)
    print("\n" + T.round(4).to_string(index=False))
    print(f"\n  ripple rows {N.rows.sum()}, after exact-ms merge "
          f"{N.exact.sum()}, after ±{DEDUP_MS:.0f} ms cross-channel merge "
          f"{N.dedup50.sum()} ({100 * (1 - N.dedup50.sum() / N.rows.sum()):.1f}"
          f"% removed)")

    T.to_csv(os.path.join(out_dir, "variants.csv"), index=False)
    N.to_csv(os.path.join(out_dir, "ripple_counts.csv"), index=False)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"settings": dict(
                       analysis=ANALYSIS_NAME, created=stamp,
                       script=os.path.relpath(__file__), bundle=b_dir,
                       partition_dir=partition_dir, unit="session",
                       variants=[dict(zip(("key", "label", "weighting",
                                           "hc_clean", "shifts", "dedup"), v))
                                 for v in VARIANTS],
                       shifts=shift_sets, min_ripples=MIN_RIPPLES,
                       dedup_ms=DEDUP_MS,
                       saved_reference=dict(region_run_p=0.032,
                                            region_run_paired_p=0.015,
                                            phases_run_p=0.072,
                                            phases_run_paired_p=0.007)),
                   "variants": T.to_dict("records"),
                   "ripples": dict(rows=int(N.rows.sum()),
                                   exact=int(N.exact.sum()),
                                   dedup50=int(N.dedup50.sum()))},
                  f, indent=2, default=str)
    _figure(T, curves, t_ms, out_dir)
    print(f"\n  saved -> {out_dir}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()


def _figure(T, curves, t_ms, out_dir):
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    with plt.rc_context(rfig._rc()):
        fig, axes = plt.subplots(1, 3, figsize=(19 * rfig.CM, 6.5 * rfig.CM),
                                 gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
        ax = axes[0]
        y = np.arange(len(T))[::-1]
        for col, mk, lab in (("p_unpaired", "o", "unpaired (35 vs 27)"),
                             ("p_paired", "s", "paired (shared sessions)")):
            ax.plot(T[col], y, mk, color="#1a1a1a" if mk == "o" else "#8C8C8C",
                    ms=5, label=lab)
        ax.axvline(0.05, color="#a30d6c", lw=0.8, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(T.label, fontsize=rfig.FS_TICK - 2)
        ax.set_xlabel("p, mPFC − OFC timing", fontsize=rfig.FS_TICK)
        ax.set_xlim(0, max(0.12, T[["p_unpaired", "p_paired"]].max().max()
                           * 1.1))
        fig.legend(*ax.get_legend_handles_labels(), frameon=False,
                   fontsize=rfig.FS_TICK - 2, loc="lower center", ncol=1,
                   bbox_to_anchor=(0.17, 0.0))
        ax.set_title("timing test per variant", fontsize=rfig.FS_TICK)
        k = 5
        for ax, roi in zip(axes[1:], ROIS):
            for i, (key_, label, *_) in enumerate(VARIANTS):
                X = np.stack(list(curves[key_][roi].values()))
                m = uniform_filter1d(X.mean(0), k)
                ax.plot(t_ms, m - m[np.abs(t_ms) >= REMOTE_MS].mean(),
                        lw=1.0, alpha=0.85,
                        color=plt.cm.viridis(i / (len(VARIANTS) - 1)),
                        label=label)
            ax.axhline(0, color="#3d3d3d", lw=0.6, ls="--")
            ax.axvline(0, color="#1a1a1a", lw=0.6, ls=":")
            ax.set_xlim(-750, 750)
            ax.set_xlabel("time from ripple (ms)", fontsize=rfig.FS_TICK)
            ax.set_ylabel(f"{roi} HFB (z)", fontsize=rfig.FS_TICK,
                          color=ROI_COL.get(roi, "#1a1a1a"))
            ax.set_title(f"{roi}: mean curve per variant",
                         fontsize=rfig.FS_TICK)
            ax.tick_params(labelsize=rfig.FS_TICK - 2)
        fig.legend(*axes[2].get_legend_handles_labels(), frameon=False,
                   fontsize=rfig.FS_TICK - 2, loc="lower center", ncol=3,
                   handlelength=1.2, bbox_to_anchor=(0.62, 0.0))
        axes[0].tick_params(axis="x", labelsize=rfig.FS_TICK - 2)
        fig.suptitle("Same ripples, same HFB: which averaging choice moves the "
                     "result?", fontsize=rfig.FS_LABEL)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"extraction_audit.{ext}"),
                        dpi=300)


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
