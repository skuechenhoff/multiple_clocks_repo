#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cortical high-frequency broadband locked to hippocampal ripples (idea I1/I3).

He et al. 2026 (Nat Neurosci 29:1711) Fig 4a/7a: mPFC HFB rises around
hippocampal ripple peaks and visual cortex does not. This asks the same
question of the ABCD dataset, with their windows -- peri = +-250 ms, non-peri =
(-750:-250) U (+250:+750).

Runs entirely from the bundle. No LFP, no cluster.

THREE THINGS THIS DOES THAT THE NAIVE VERSION DOES NOT, each because the naive
version gave a misleading answer when it was run:

1. **The reference is a SHIFTED NULL, not zero.** Ripples cluster in states
   (stillness; see SWR_SUMMARY F1) and cortical HFB varies with state too, so
   peri-minus-non-peri is positive in EVERY region -- controls included --
   before anything is controlled. Shifting ripple times by tens of seconds
   destroys the locking while preserving the state, and the shifted effect is
   ~0 everywhere. The real-minus-shifted difference is the estimate.

2. **Same-shaft derivations are EXCLUDED from the anatomical claim.** A
   temporal depth electrode carries its own outer contacts in lateral temporal
   cortex, and those sit millimetres from the hippocampal contact. Measured
   here, the ripple-locked effect on same-shaft contacts is ~3x the
   different-shaft effect in lateral temporal (+0.0068 vs +0.0023), and the
   entire Visual effect comes from 9 same-shaft derivations (+0.0081 same
   shaft, +0.0018 and null on the other 97). That is volume conduction, not
   anatomy. Same-shaft contacts remain an excellent control for RECORDING
   QUALITY -- shared amplifier, reference and noise -- and a bad one for
   anatomical specificity. Both are reported; the primary set is
   different-shaft.

3. **Every ROI is reported.** The effect is not mPFC-selective and saying so is
   the finding, not a footnote.

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_ripple_locked_hfb.py run --bundle=<bundle_v2 dir>
    python scripts/swr_ripple_locked_hfb.py run --bundle=<dir> --n_shifts=12
    python scripts/swr_ripple_locked_hfb.py figure --results=<out dir>

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.swr_bundle as sb

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "ripple_locked_hfb"
HALF_S = 0.75                  # epoch half-width = He's non-peri outer edge
PERI_S = 0.25                  # He: peri = +-250 ms
MIN_RIPPLES = 20               # per (HC x cortical) pair, for a stable mean
SHIFT_RANGE_S = (5.0, 120.0)   # shifted-null offsets, either sign
# Artifact-pad stability. swr_v2 detects at 0.1 s and every event carries
# dist_to_artifact_s, so larger pads are nested SUBSETS of the same detection --
# no re-run needed. A real effect should not grow as data is thrown away; the
# ripple-RSA mPFC effect (CHANGELOG 2026-09-16) failed exactly this check, being
# largest at the pad with fewest ripples. Applying the same standard here.
#
# The sweep STARTS at HALF_S, not at the detection pad, and that is not a
# choice. A peri/non-peri epoch spans +-750 ms and must be artifact-free, so an
# event closer than 750 ms to a crossing can never enter regardless of the pad
# it was detected at -- 26.7% of the bundle is excluded on that ground alone.
# Sweeping 0.10/0.25/0.50 therefore compares three identical subsets and looks
# reassuringly flat while testing nothing. The informative range is above the
# epoch half-width, where 0.75 -> 3.0 s takes the usable set from 100% to ~40%.
PAD_SWEEP = (0.75, 1.00, 1.50, 2.00, 3.00)
ROI_ORDER = ["mPFC", "mOFC", "TemporalLateral", "Auditory", "Visual"]


def _clean_mask(rows, n, fs):
    m = np.zeros(n, bool)
    for a, b in rows:
        i0, i1 = max(0, int(a * fs)), min(n, int(b * fs))
        if i1 > i0:
            m[i0:i1] = True
    return m


def run(bundle=None, n_shifts=8, seed=42, out_dir=None, save=True,
        pad_sweep=PAD_SWEEP):
    b_dir = bundle or os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                                   "group", "swr", "bundle_v2")
    with open(os.path.join(b_dir, "swr_bundle.pkl"), "rb") as f:
        B = pickle.load(f)
    rip, hfb_pairs, hiv, pairs = (B["ripples"], B["hfb_pairs"],
                                  B["hfb_intervals"], B["pairs"])
    hp = hfb_pairs[~hfb_pairs.excluded.fillna(False)].copy()

    # same_probe_as_hpc lives on `pairs`, not `hfb_pairs`
    key = pairs[["session", "pair_id", "same_probe_as_hpc"]].drop_duplicates()
    hp = hp.merge(key, on=["session", "pair_id"], how="left")
    hp["same_shaft"] = hp.same_probe_as_hpc.fillna(False).astype(bool)

    rng = np.random.default_rng(seed)
    lo, hi = SHIFT_RANGE_S
    shifts = [0.0] + [float(s) for s in
                      rng.uniform(lo, hi, n_shifts) * rng.choice([-1, 1], n_shifts)]

    fs = float(B["hfb_index"].out_fs.iloc[0]) if len(B["hfb_index"]) else 100.0
    w, wp = int(HALF_S * fs), int(PERI_S * fs)
    off = np.arange(-w, w)
    print(f"\nbundle {b_dir}\n  {len(hp)} usable derivations, fs={fs:.0f} Hz, "
          f"{n_shifts} shifted nulls in +-[{lo:.0f},{hi:.0f}] s")

    rows, tc, tc_ix = [], [], []
    for sess in sorted(hp.session.unique()):
        g = hp[hp.session == sess]
        if not (g.roi_family == "HPC").any():
            continue
        try:
            D = sb.load_hfb(b_dir, sess, arrays=["hfb"])
        except Exception as e:
            print(f"  s{sess:02d}: {type(e).__name__}: {e}")
            continue
        H, ids = D["hfb"], D["pair_ids"]
        n = H.shape[1]
        idx = {p: i for i, p in enumerate(ids)}
        masks = {p: _clean_mask(
            hiv[(hiv.session == sess) & (hiv.pair_id == p)]
            [["start_s", "stop_s"]].to_numpy(float), n, fs) for p in g.pair_id}
        hc, cx = g[g.roi_family == "HPC"], g[g.roi_family != "HPC"]

        for _, h in hc.iterrows():
            _r = rip.loc[(rip.session == sess) & (rip.pair_id == h.pair_id)]
            t0 = _r["t_peak_s"].to_numpy(float)
            d0 = (_r["dist_to_artifact_s"].to_numpy(float)
                  if "dist_to_artifact_s" in _r else np.full(len(t0), np.inf))
            if len(t0) < MIN_RIPPLES:
                continue
            for k, sh in enumerate(shifts):
                c_all = np.round((t0 + sh) * fs).astype(int)
                inb = (c_all - w >= 0) & (c_all + w < n)
                c, dist = c_all[inb], d0[inb]
                if len(c) < MIN_RIPPLES:
                    continue
                win = c[:, None] + off[None, :]
                hc_ok = masks[h.pair_id][win].all(1)
                for _, x in cx.iterrows():
                    if x.pair_id not in idx:
                        continue
                    ok = hc_ok & masks[x.pair_id][win].all(1)
                    if ok.sum() < MIN_RIPPLES:
                        continue
                    stack = H[idx[x.pair_id]][win[ok]]
                    d_ok = dist[ok]
                    # Larger pads are nested subsets of the SAME epochs, so the
                    # whole sweep costs one extraction rather than four runs.
                    for pad in pad_sweep:
                        keep = d_ok >= pad
                        if keep.sum() < MIN_RIPPLES:
                            continue
                        m = stack[keep].mean(0)
                        rows.append({
                            "session": sess, "subject": x.subject_label,
                            "hc_pair": h.pair_id, "cx_pair": x.pair_id,
                            "roi": x.roi_family, "same_shaft": bool(x.same_shaft),
                            "shift_s": sh, "is_real": sh == 0.0, "pad_s": pad,
                            "n_ripples": int(keep.sum()),
                            "peri": float(m[w - wp:w + wp].mean()),
                            "nonperi": float(np.r_[m[:w - wp], m[w + wp:]].mean()),
                        })
                    m = stack.mean(0)
                    tc.append(m)
                    tc_ix.append({"subject": x.subject_label,
                                  "roi": x.roi_family,
                                  "same_shaft": bool(x.same_shaft),
                                  "is_real": sh == 0.0,
                                  "cx_pair": x.pair_id})
        print(f"  s{sess:02d}: {len(hc)} HC x {len(cx)} cortical", end="\r")

    d = pd.DataFrame(rows)
    d["diff"] = d.peri - d.nonperi
    print(f"\n\n{len(d)} rows, {d.cx_pair.nunique()} cortical derivations, "
          f"{d.session.nunique()} sessions, {d.subject.nunique()} subjects")

    native = min(pad_sweep)
    res = _stats(d[d.pad_s == native])
    res["pad_sweep"] = _pad_sweep(d, pad_sweep)
    _report(res, d[d.pad_s == native])
    _report_sweep(res["pad_sweep"], pad_sweep)

    if save:
        out_dir = out_dir or os.path.join(
            swr_io.derivatives_dir(swr_io.get_data_root()), "group", "swr",
            f"{ANALYSIS_NAME}_{datetime.now():%Y-%m-%d}")
        os.makedirs(out_dir, exist_ok=True)
        d.to_csv(os.path.join(out_dir, "per_pair.csv"), index=False)
        # Flat stack plus an index, rather than one array per (roi, shaft,
        # real) cell. The keyed form forced the figure to average over
        # derivations while the statistics average over subjects, and for
        # Visual -- whose coverage is concentrated, max 14 derivations in one
        # subject -- those differ fourfold (+0.0044 vs +0.0011). The two panels
        # then told different stories. With the index, the figure aggregates
        # exactly as the test does.
        np.savez_compressed(os.path.join(out_dir, "timecourses.npz"),
                            t_ms=off / fs * 1000.0, traces=np.stack(tc))
        pd.DataFrame(tc_ix).to_csv(
            os.path.join(out_dir, "timecourse_index.csv"), index=False)
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump({"analysis": ANALYSIS_NAME, "bundle": b_dir,
                       "peri_s": PERI_S, "half_s": HALF_S,
                       "n_shifts": n_shifts, "shift_range_s": list(SHIFT_RANGE_S),
                       "seed": seed, "min_ripples": MIN_RIPPLES,
                       "created": datetime.now().isoformat(timespec="seconds"),
                       "results": res}, f, indent=2, default=str)
        swr_io.write_settings(out_dir, {"analysis_name": ANALYSIS_NAME,
                                        "bundle": b_dir, "n_shifts": n_shifts,
                                        "seed": seed})
        print(f"\nsaved -> {out_dir}")
        try:
            figure(results=out_dir)
        except Exception as e:
            print(f"  [figure skipped: {type(e).__name__}: {e}]")
    return None


def _stats(d):
    """Subject-level real-minus-shifted-null, per ROI, split by shaft."""
    from scipy import stats as st
    out = {}
    for same in (False, True):
        g0 = d[d.same_shaft == same]
        lab = "same_shaft" if same else "different_shaft"
        out[lab] = {}
        for roi in ROI_ORDER:
            g = g0[g0.roi == roi]
            if not len(g):
                continue
            real = g[g.is_real].groupby("subject")["diff"].mean()
            null = g[~g.is_real].groupby("subject")["diff"].mean()
            c = real.index.intersection(null.index)
            if len(c) < 5:
                continue
            v = (real[c] - null[c]).to_numpy()
            t, p = st.ttest_1samp(v, 0.0)
            out[lab][roi] = {
                "n_subjects": int(len(c)),
                "n_derivations": int(g.cx_pair.nunique()),
                "n_ripple_alignments": int(g[g.is_real].n_ripples.sum()),
                "real": float(real[c].mean()), "null": float(null[c].mean()),
                "effect": float(v.mean()), "sem": float(v.std(ddof=1)/np.sqrt(len(v))),
                "t": float(t), "p": float(p)}
    # ROI contrasts, different-shaft only
    g0 = d[~d.same_shaft]
    eff = (g0[g0.is_real].groupby(["subject", "roi"])["diff"].mean()
           - g0[~g0.is_real].groupby(["subject", "roi"])["diff"].mean()).reset_index()
    piv = eff.pivot(index="subject", columns="roi", values="diff")
    out["contrasts_different_shaft"] = {}
    for a, b in [("mOFC", "Visual"), ("mPFC", "Visual"),
                 ("mOFC", "TemporalLateral"), ("mPFC", "TemporalLateral"),
                 ("mOFC", "mPFC")]:
        if a not in piv or b not in piv:
            continue
        s = piv[[a, b]].dropna()
        if len(s) < 5:
            continue
        t, p = st.ttest_rel(s[a], s[b])
        out["contrasts_different_shaft"][f"{a}_vs_{b}"] = {
            "n_subjects": int(len(s)), "diff": float((s[a] - s[b]).mean()),
            "t": float(t), "p": float(p)}
    return out


def _pad_sweep(d, pads):
    """Effect at each artifact pad. Larger pad = fewer, cleaner ripples.

    A real effect should be stable or STRENGTHEN with more data. One that grows
    as ripples are discarded is a small-n artefact -- the failure mode the
    ripple-RSA mPFC result showed (CHANGELOG 2026-09-16).

    Pads below HALF_S are not tested: the epoch requirement already excludes
    every event nearer than HALF_S to a crossing, so they would compare
    identical subsets. See the PAD_SWEEP comment.
    """
    from scipy import stats as st
    g0 = d[~d.same_shaft]
    out = {}
    for pad in pads:
        g1 = g0[g0.pad_s == pad]
        out[f"{pad:.2f}"] = {}
        for roi in ROI_ORDER:
            g = g1[g1.roi == roi]
            if not len(g):
                continue
            real = g[g.is_real].groupby("subject")["diff"].mean()
            null = g[~g.is_real].groupby("subject")["diff"].mean()
            c = real.index.intersection(null.index)
            if len(c) < 5:
                continue
            v = (real[c] - null[c]).to_numpy()
            t, p = st.ttest_1samp(v, 0.0)
            out[f"{pad:.2f}"][roi] = {
                "n_subjects": int(len(c)),
                "n_ripple_alignments": int(g[g.is_real].n_ripples.sum()),
                "effect": float(v.mean()), "t": float(t), "p": float(p)}
    return out


def _report_sweep(sw, pads):
    print("\n" + "=" * 78)
    print(" ARTIFACT-PAD STABILITY   (different shaft; larger pad = fewer ripples)")
    print(" a real effect should NOT grow as ripples are discarded")
    print(f" sweep starts at the epoch half-width ({HALF_S:.2f}s): events nearer than"
          f" that to a\n crossing cannot enter at ANY detection pad, so smaller"
          f" pads test nothing")
    print("=" * 78)
    keys = [f"{p:.2f}" for p in pads]
    print(f"\n  {'ROI':<16s}" + "".join(f"{'pad ' + k:>16s}" for k in keys))
    for roi in ROI_ORDER:
        if not any(roi in sw.get(k, {}) for k in keys):
            continue
        line = f"  {roi:<16s}"
        for k in keys:
            v = sw.get(k, {}).get(roi)
            line += (f"{v['effect']:>8.4f} p={v['p']:<6.3f}" if v
                     else f"{'--':>16s}")
        print(line)
    print(f"\n  {'alignments':<16s}" + "".join(
        f"{max((sw[k][r]['n_ripple_alignments'] for r in sw[k]), default=0):>16,d}"
        for k in keys))


def _report(res, d):
    print("\n" + "=" * 78)
    print(" RIPPLE-LOCKED CORTICAL HFB   (real minus shifted null, subject-level)")
    print("=" * 78)
    for lab in ("different_shaft", "same_shaft"):
        if not res.get(lab):
            continue
        note = ("PRIMARY -- volume-conduction free"
                if lab == "different_shaft" else
                "volume conduction: same electrode as the hippocampal contact")
        print(f"\n  {lab}   ({note})")
        print(f"    {'ROI':<17s}{'n_cx':>6s}{'subj':>6s}{'align':>9s}"
              f"{'effect':>10s}{'t':>7s}{'p':>9s}")
        for roi, v in res[lab].items():
            star = "*" if v["p"] < 0.05 else " "
            print(f"    {roi:<17s}{v['n_derivations']:>6d}{v['n_subjects']:>6d}"
                  f"{v['n_ripple_alignments']:>9d}{v['effect']:>10.4f}"
                  f"{v['t']:>7.2f}{v['p']:>9.4f}{star}")
    if res.get("contrasts_different_shaft"):
        print("\n  ROI contrasts (different shaft only, paired within subject)")
        for k, v in res["contrasts_different_shaft"].items():
            star = "*" if v["p"] < 0.05 else " "
            print(f"    {k:<30s} n={v['n_subjects']:>3d}  "
                  f"{v['diff']:+.4f}  t={v['t']:+.2f}  p={v['p']:.4f}{star}")
    print("\n" + "=" * 78)


def figure(results=None, out_stem=None):
    """Time courses and the effect per ROI, different-shaft set."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mc.plotting.ripple_figures as rfig

    R = results
    z = np.load(os.path.join(R, "timecourses.npz"))
    res = json.load(open(os.path.join(R, "result.json")))["results"]
    t_ms = z["t_ms"]
    C = rfig.MONTAGE_C
    CM = 1 / 2.54

    SHORT = {"TemporalLateral": "Lat. temporal", "Auditory": "Auditory",
             "Visual": "Visual", "mPFC": "mPFC", "mOFC": "mOFC"}
    tr = z["traces"]
    ix = pd.read_csv(os.path.join(R, "timecourse_index.csv"))
    ix["same_shaft"] = ix.same_shaft.astype(bool)
    ix["is_real"] = ix.is_real.astype(bool)
    present = [r for r in ROI_ORDER if r in res.get("different_shaft", {})
               and ((ix.roi == r) & ~ix.same_shaft).any()]

    fig, axes = plt.subplots(1, 2, figsize=(18.0 * CM, 7.0 * CM),
                             gridspec_kw={"width_ratios": [1.45, 1.0]},
                             constrained_layout=True)

    ax = axes[0]
    flank = np.abs(t_ms) >= PERI_S * 1000          # He's non-peri window
    for roi in present:
        sel = (ix.roi == roi) & (~ix.same_shaft)
        # SUBJECT-level, exactly as the test: mean within subject first, then
        # across subjects. Averaging over derivations instead lets a subject
        # with 14 Visual contacts outweigh one with 1.
        curves = []
        for subj, gi in ix[sel].groupby("subject"):
            r = tr[gi.index[gi.is_real.to_numpy()]]
            n = tr[gi.index[~gi.is_real.to_numpy()]]
            if not len(r) or not len(n):
                continue
            c = r.mean(0) - n.mean(0)
            curves.append(c - c[flank].mean())
        if not curves:
            continue
        A = np.stack(curves)
        m, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))
        ax.plot(t_ms, m, color=C.get(roi, "#888"), lw=1.5,
                label=f"{SHORT.get(roi, roi)} ({len(A)} subj)")
        ax.fill_between(t_ms, m - se, m + se, color=C.get(roi, "#888"),
                        alpha=0.15, lw=0)
    ax.axvline(0, color="0.35", lw=0.8, ls="--")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.axvspan(-PERI_S * 1000, PERI_S * 1000, color="#F15A29", alpha=0.07, lw=0)
    ax.set_xlabel("Time from hippocampal ripple peak (ms)", fontsize=9)
    ax.set_ylabel("HFB, real − null, vs non-peri (z)", fontsize=9)
    ax.set_title("Ripple-locked cortical HFB\n(subject-level, different shaft)",
                 fontsize=10.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", handlelength=1.4,
              labelspacing=0.3)
    ax.tick_params(labelsize=8)
    ax.margins(x=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax = axes[1]
    vals = [res["different_shaft"][r] for r in present]
    top = max(v["effect"] + v["sem"] for v in vals)
    bot = min(0.0, min(v["effect"] - v["sem"] for v in vals))
    for i, (roi, v) in enumerate(zip(present, vals)):
        ax.bar(i, v["effect"], yerr=v["sem"], color=C.get(roi, "#888"),
               width=0.68, capsize=3, error_kw=dict(lw=0.9))
        if v["p"] < 0.05:
            ax.text(i, v["effect"] + v["sem"] + 0.03 * top,
                    "**" if v["p"] < 0.01 else "*", ha="center", fontsize=10)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylim(bot - 0.06 * top, top * 1.25)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([f"{SHORT.get(r, r)}\n(n={v['n_subjects']})"
                        for r, v in zip(present, vals)], fontsize=8,
                       rotation=30, ha="right")
    ax.set_ylabel("peri − non-peri, minus null (z)", fontsize=9)
    ax.set_title("Effect per region\n(subject-level, mean ± s.e.m.)", fontsize=11)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    stem = out_stem or os.path.join(R, "ripple_locked_hfb")
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    print(f"figure -> {stem}.pdf / .png")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run, "figure": figure})
    else:
        run()
