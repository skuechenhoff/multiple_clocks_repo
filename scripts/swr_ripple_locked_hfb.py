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
# Epoch is +-1 s: Norman et al. 2021 plot and test over that range, and He et al.
# plot it even though their windows stop at 750 ms. The STATISTIC keeps He's
# windows exactly, so it stays comparable to them; the extra 250 ms either side
# is for looking at.
HALF_S = 1.00                  # epoch half-width (plotting)
PERI_S = 0.25                  # He: peri = +-250 ms
NONPERI_S = (0.25, 0.75)       # He: non-peri = (-750:-250) U (+250:+750)
SMOOTH_MS = 50.0               # time-course smoothing; 0 disables
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
# SK asked to carry 0.25 and 0.5 forward. They are recorded in the settings and
# swept, but for THIS design they are inert and that must not be mistaken for
# stability: a +-1 s epoch has to be artifact-free, so every event nearer than
# 1 s to a crossing is excluded whatever the detection pad. Pads at or below the
# epoch half-width all select the identical subset. They do matter for the rate
# and RSA analyses, which have no epoch requirement.
PAD_SWEEP = (0.25, 0.50, 1.00, 1.50, 2.00, 3.00)
PAD_INERT_BELOW = HALF_S
ROI_ORDER = ["MedialFrontal", "A24_z-10_to_5", "mPFC", "mOFC",
             "TemporalLateral", "Auditory", "Visual"]

# MEDIAL frontal, by coordinate rather than by label. The `mOFC` label is not
# reliably medial: only 36 of 225 contacts carrying it were assigned by the
# Brainnetome medial-OFC rule (A11m/A13/A14m, |x| median 9.7 mm, max 13.1);
# the other 189 came from a 1-3 mm neighbourhood rescue and reach |x| = 52.8 mm,
# which is lateral orbitofrontal cortex. 40% of "mOFC" sits beyond |x| = 25 mm.
# mPFC is clean by comparison (|x| median 10.4, max 17.0).
#
# This group therefore takes mPFC and mOFC derivations within MEDIAL_MAX_ABS_X
# of the midline and reports them as one medial frontal region. The threshold
# is anatomical, not chosen for effect: the estimate is flat across it --
# +0.0068 at 12 mm, +0.0071 at 15, +0.0073 at 20, +0.0068 at 25, +0.0067 with
# no cut -- so the cut buys an honest label, not a bigger number.
#
# ⚠ |x| here is the pair MIDPOINT, a median 2.2 mm lateral of the source
# contact, so the effective anchor threshold is ~18 mm.
MEDIAL_MAX_ABS_X = 20.0
MEDIAL_NAME = "MedialFrontal"

# A z-defined medial-frontal group cutting across the mPFC/mOFC label boundary.
# Area 24 is a principal hippocampal input to prefrontal cortex and straddles
# that boundary as this project draws it, so the labels may be splitting one
# functional zone. Derivations in mPFC or mOFC whose midpoint z falls in the
# band are ADDED under this name; their original rows are left untouched, so
# mPFC and mOFC statistics are unchanged and the groups OVERLAP -- never sum
# them or treat them as independent.
#
# ⚠ The band was chosen after seeing the gradient scatter. It is defensible on
# anatomy (perigenual/subgenual area 24 is the classic input zone) but it is
# NOT independent of the data that suggested it. The continuous z regression in
# `_z_gradient` is the non-circular version of the same question -- quote that
# one when the distinction matters.
A24_BAND = (-10.0, 5.0)
A24_NAME = "A24_z-10_to_5"


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
    t_s = off / fs
    peri_ix = np.abs(t_s) < PERI_S
    nonperi_ix = (np.abs(t_s) >= NONPERI_S[0]) & (np.abs(t_s) < NONPERI_S[1])
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
                            "peri": float(m[peri_ix].mean()),
                            "nonperi": float(m[nonperi_ix].mean()),
                        })
                    m = stack.mean(0)
                    tc.append(m)
                    tc_ix.append({"subject": x.subject_label,
                                  "session": sess,
                                  "roi": x.roi_family,
                                  "same_shaft": bool(x.same_shaft),
                                  "is_real": sh == 0.0,
                                  "cx_pair": x.pair_id})
        print(f"  s{sess:02d}: {len(hc)} HC x {len(cx)} cortical", end="\r")

    d = pd.DataFrame(rows)
    d["diff"] = d.peri - d.nonperi

    # Duplicate the medial-frontal rows that fall in the A24 band under the new
    # label. The traces are indexed, not copied, so the figure gets it too.
    # Derived, OVERLAPPING ROI groups: MedialFrontal (by |x|) and A24 (by z).
    # Both are built from the ORIGINAL rows and appended in one step. Building
    # them sequentially meant the second mask was computed against a frame the
    # first had already grown, and pandas refused the misaligned indexer.
    #
    # Keyed on (session, pair_id), never pair_id alone: a pair label recurs
    # across sessions of one patient, so a label-only index is non-unique --
    # the same trap that produced the sites-vs-derivations miscount.
    mx = (pairs.drop_duplicates(["session", "pair_id"])
               .set_index(["session", "pair_id"])["mni_x"])
    mz = (pairs.drop_duplicates(["session", "pair_id"])
               .set_index(["session", "pair_id"])["mni_z"])

    def _key(frame):
        return pd.MultiIndex.from_arrays([frame.session, frame.cx_pair])

    def _masks(frame):
        ax = np.abs(pd.to_numeric(_key(frame).map(mx), errors="coerce"))
        z = pd.to_numeric(_key(frame).map(mz), errors="coerce")
        front = frame.roi.isin(["mPFC", "mOFC"]).to_numpy()
        return (pd.Series(front & (ax <= MEDIAL_MAX_ABS_X), index=frame.index),
                pd.Series(front & (z >= A24_BAND[0]) & (z <= A24_BAND[1]),
                          index=frame.index))

    med, band = _masks(d)
    extras = []
    for mask, name in ((med, MEDIAL_NAME), (band, A24_NAME)):
        if mask.any():
            e = d[mask].copy()
            e["roi"] = name
            extras.append(e)
            print(f"  {name}: {int(e.cx_pair.nunique())} sites / "
                  f"{len(e[['session', 'cx_pair']].drop_duplicates())} "
                  f"derivations -- OVERLAPS mPFC/mOFC, never sum them")
    if extras:
        d = pd.concat([d] + extras, ignore_index=True)

    ti = pd.DataFrame(tc_ix)
    ti["trace_row"] = ti.index
    tmed, tband = _masks(ti)
    t_extras = []
    for mask, name in ((tmed, MEDIAL_NAME), (tband, A24_NAME)):
        if mask.any():
            e = ti[mask].copy()
            e["roi"] = name          # trace_row already points at the real trace
            t_extras.append(e)
    tc_ix = pd.concat([ti] + t_extras, ignore_index=True).to_dict("records")

    # `subject_label` is the site's label and is not unique per patient -- one
    # Utah patient carries both 'UT1-202314' and 'UT202314', so counting labels
    # says 42 where the manuscript says 41. Patients are counted on the
    # manifest's subject_key, which merges them.
    try:
        sk = rip[["session", "subject_key"]].drop_duplicates()
        n_pat = d[["session"]].drop_duplicates().merge(
            sk, on="session", how="left").subject_key.nunique()
    except Exception:
        n_pat = d.subject.nunique()
    print(f"\n\n{len(d)} rows, {d.cx_pair.nunique()} cortical sites, "
          f"{len(d[['session', 'cx_pair']].drop_duplicates())} derivations, "
          f"{d.session.nunique()} sessions, {n_pat} patients")

    native = min(p for p in pad_sweep if p >= PAD_INERT_BELOW)
    dn = d[d.pad_s == native]
    res = {}
    for lvl in ("session", "subject"):
        res[lvl] = _stats(dn, unit=lvl)
        res[f"contrasts_{lvl}"] = _contrasts(dn, unit=lvl)
    res["pad_sweep"] = {lvl: _pad_sweep(d, pad_sweep, unit=lvl)
                        for lvl in ("session", "subject")}
    res["z_gradient"] = _z_gradient(dn, hp)
    res["native_pad_s"] = native
    _report(res, pad_sweep)

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
                       "nonperi_s": list(NONPERI_S), "smooth_ms": SMOOTH_MS,
                       "pad_sweep": list(pad_sweep),
                       "pad_inert_below": PAD_INERT_BELOW,
                       "pad_note": ("pads <= the epoch half-width select the "
                                    "identical subset: a +-HALF_S epoch must be "
                                    "artifact-free, so nearer events are excluded "
                                    "at any detection pad"),
                       "levels": ["session (PRIMARY)", "subject (control)"],
                       "a24_band": list(A24_BAND),
                       "a24_note": ("overlaps mPFC/mOFC by construction; "
                                    "band chosen after seeing the gradient, "
                                    "so the continuous z regression is the "
                                    "non-circular version"),
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


def _effect(g, unit):
    """Real-minus-shifted-null per `unit` (session or subject)."""
    real = g[g.is_real].groupby(unit)["diff"].mean()
    null = g[~g.is_real].groupby(unit)["diff"].mean()
    c = real.index.intersection(null.index)
    return (real[c] - null[c])


def _stats(d, unit="subject"):
    """Effect per ROI, split by shaft. `unit` is the level of inference.

    Both levels are reported because they can disagree and the disagreement is
    informative: a session-level test treats two sessions from one patient as
    independent, which they are not, while a subject-level test throws away the
    within-patient replication. Where they agree, the result does not depend on
    that choice.
    """
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
            v = _effect(g, unit)
            if len(v) < 5:
                continue
            t, p = st.ttest_1samp(v.to_numpy(), 0.0)
            out[lab][roi] = {
                "unit": unit, "n_units": int(len(v)),
                # `pair_id` is an electrode label and recurs across sessions of
                # the same patient (27 of 51 mPFC labels do), so nunique()
                # counts SITES, not derivation-instances. Both are reported:
                # sites is the He-et-al.-comparable "n contacts", instances is
                # what the average is actually taken over.
                "n_sites": int(g.cx_pair.nunique()),
                "n_derivations": int(
                    len(g[["session", "cx_pair"]].drop_duplicates())),
                "n_ripple_alignments": int(g[g.is_real].n_ripples.sum()),
                "effect": float(v.mean()),
                "sem": float(v.std(ddof=1) / np.sqrt(len(v))),
                "t": float(t), "p": float(p)}
    return out


def _contrasts(d, unit="subject"):
    from scipy import stats as st
    g0 = d[~d.same_shaft]
    eff = (g0[g0.is_real].groupby([unit, "roi"])["diff"].mean()
           - g0[~g0.is_real].groupby([unit, "roi"])["diff"].mean()).reset_index()
    piv = eff.pivot(index=unit, columns="roi", values="diff")
    out = {}
    for a, b in [(MEDIAL_NAME, "Visual"), (MEDIAL_NAME, "TemporalLateral"),
                 (MEDIAL_NAME, "Auditory"),
                 ("mOFC", "Visual"), ("mPFC", "Visual"),
                 ("mOFC", "TemporalLateral"), ("mPFC", "TemporalLateral"),
                 ("mOFC", "mPFC")]:
        if a not in piv or b not in piv:
            continue
        x = piv[[a, b]].dropna()
        if len(x) < 5:
            continue
        t, p = st.ttest_rel(x[a], x[b])
        out[f"{a}_vs_{b}"] = {"unit": unit, "n_units": int(len(x)),
                              "diff": float((x[a] - x[b]).mean()),
                              "t": float(t), "p": float(p)}
    return out


def _z_gradient(d, hfb_pairs):
    """Does the medial-frontal effect vary along the dorsoventral (MNI z) axis?

    Area 24 is a principal hippocampal input to prefrontal cortex and straddles
    the mPFC/mOFC boundary as this project draws it, so the ROI labels may be
    cutting one functional gradient in two. Regressing the per-derivation effect
    on MNI z asks that directly, without depending on where the label boundary
    sits.

    Per derivation (not per subject) because z varies WITHIN subject -- that is
    the whole point -- with subject as a random effect.
    """
    import statsmodels.formula.api as smf
    from scipy import stats as st
    g0 = d[(~d.same_shaft) & d.roi.isin(["mPFC", "mOFC"])]
    if not len(g0):
        return {}
    eff = (g0[g0.is_real].groupby(["cx_pair", "subject", "roi"])["diff"].mean()
           - g0[~g0.is_real].groupby(["cx_pair", "subject", "roi"])["diff"].mean())
    e = eff.reset_index().rename(columns={0: "effect", "diff": "effect"})
    coord = hfb_pairs[["pair_id", "mni_x", "mni_y", "mni_z"]].drop_duplicates("pair_id")
    e = e.merge(coord, left_on="cx_pair", right_on="pair_id", how="left").dropna(
        subset=["mni_z", "effect"])
    if len(e) < 10:
        return {}
    out = {"n_derivations": int(len(e)),
           "n_subjects": int(e.subject.nunique()),
           "z_range": [float(e.mni_z.min()), float(e.mni_z.max())],
           "roi_counts": e.roi.value_counts().to_dict()}
    rho, p = st.spearmanr(e.mni_z, e.effect)
    out["spearman_z_vs_effect"] = {"rho": float(rho), "p": float(p)}
    try:
        m = smf.mixedlm("effect ~ mni_z", e, groups=e["subject"].astype(str)).fit(
            reml=True, method="nm", maxiter=2000)
        out["lme_effect_on_z"] = {"beta_per_mm": float(m.params.get("mni_z", np.nan)),
                                  "p": float(m.pvalues.get("mni_z", np.nan))}
    except Exception as ex:
        out["lme_effect_on_z"] = {"error": f"{type(ex).__name__}: {ex}"}
    # also y, since area 24 runs anteroposteriorly too
    rho_y, p_y = st.spearmanr(e.mni_y, e.effect)
    out["spearman_y_vs_effect"] = {"rho": float(rho_y), "p": float(p_y)}
    out["_table"] = e[["cx_pair", "subject", "roi", "mni_x", "mni_y", "mni_z",
                       "effect"]].to_dict("records")
    return out


def _pad_sweep(d, pads, unit="subject"):
    """Effect at each artifact pad. Larger pad = fewer, cleaner ripples.

    A real effect should be stable or STRENGTHEN with more data. One that grows
    as ripples are discarded is a small-n artefact -- the failure mode the
    ripple-RSA mPFC effect showed (CHANGELOG 2026-09-16).

    Pads at or below the epoch half-width select the IDENTICAL subset, because a
    +-HALF_S epoch must be artifact-free. They are swept anyway so the table
    shows it rather than leaving it to be assumed.
    """
    from scipy import stats as st
    g0 = d[~d.same_shaft]
    out = {}
    for pad in pads:
        g1 = g0[g0.pad_s == pad]
        k = f"{pad:.2f}"
        out[k] = {"_inert": bool(pad <= PAD_INERT_BELOW)}
        for roi in ROI_ORDER:
            g = g1[g1.roi == roi]
            if not len(g):
                continue
            v = _effect(g, unit)
            if len(v) < 5:
                continue
            t, p = st.ttest_1samp(v.to_numpy(), 0.0)
            out[k][roi] = {"n_units": int(len(v)),
                           "n_ripple_alignments": int(g[g.is_real].n_ripples.sum()),
                           "effect": float(v.mean()), "t": float(t), "p": float(p)}
    return out


def _report(res, pads):
    print("\n" + "=" * 80)
    print(f" RIPPLE-LOCKED CORTICAL HFB   (real minus shifted null, "
          f"pad {res['native_pad_s']:.2f}s)")
    print(" peri = +-250 ms; non-peri = (-750:-250) U (+250:+750); epoch +-1 s")
    print("=" * 80)
    for lvl in ("session", "subject"):
        tag = "PRIMARY" if lvl == "session" else "control"
        print(f"\n  --- {lvl.upper()}-level [{tag}], different shaft "
              f"(volume-conduction free) ---")
        print(f"    {'ROI':<17s}{'sites':>6s}{'deriv':>6s}{'n_' + lvl:>8s}"
              f"{'align':>9s}{'effect':>10s}{'t':>7s}{'p':>9s}")
        for roi, v in res[lvl].get("different_shaft", {}).items():
            star = "*" if v["p"] < 0.05 else " "
            print(f"    {roi:<17s}{v.get('n_sites', 0):>6d}"
                  f"{v['n_derivations']:>6d}{v['n_units']:>8d}"
                  f"{v['n_ripple_alignments']:>9d}{v['effect']:>10.4f}"
                  f"{v['t']:>7.2f}{v['p']:>9.4f}{star}")
        cs = res.get(f"contrasts_{lvl}", {})
        if cs:
            print(f"    contrasts: " + ";  ".join(
                f"{k} {v['diff']:+.4f} p={v['p']:.3f}" for k, v in cs.items()))

    print("\n  --- SAME shaft (volume conduction; not for anatomical claims) ---")
    for roi, v in res["subject"].get("same_shaft", {}).items():
        print(f"    {roi:<17s}{v.get('n_sites', 0):>6d}"
              f"{v['n_derivations']:>6d}{v['n_units']:>8d}"
              f"{'':>9s}{v['effect']:>10.4f}{v['t']:>7.2f}{v['p']:>9.4f}")

    zg = res.get("z_gradient") or {}
    if zg:
        print(f"\n  --- MEDIAL FRONTAL dorsoventral gradient "
              f"({zg['n_derivations']} derivations, {zg['n_subjects']} subjects, "
              f"z {zg['z_range'][0]:.0f} to {zg['z_range'][1]:.0f} mm) ---")
        sp = zg["spearman_z_vs_effect"]
        print(f"    effect vs MNI z : Spearman rho = {sp['rho']:+.3f}, "
              f"p = {sp['p']:.4f}")
        lme = zg.get("lme_effect_on_z", {})
        if "beta_per_mm" in lme:
            print(f"                      LME beta = {lme['beta_per_mm']:+.5f} "
                  f"per mm, p = {lme['p']:.4f}")
        sy = zg["spearman_y_vs_effect"]
        print(f"    effect vs MNI y : Spearman rho = {sy['rho']:+.3f}, "
              f"p = {sy['p']:.4f}")
        print(f"    ROI mix: {zg['roi_counts']}")

    for lvl in ("session", "subject"):
        sw = res["pad_sweep"][lvl]
        keys = [f"{p:.2f}" for p in pads]
        print(f"\n  --- PAD STABILITY, {lvl}-level "
              f"(pads <= {PAD_INERT_BELOW:.2f}s are the SAME subset) ---")
        print(f"    {'ROI':<16s}" + "".join(f"{'pad ' + k:>15s}" for k in keys))
        for roi in ROI_ORDER:
            if not any(roi in sw.get(k, {}) for k in keys):
                continue
            line = f"    {roi:<16s}"
            for k in keys:
                v = sw.get(k, {}).get(roi)
                line += (f"{v['effect']:>8.4f} p={v['p']:<4.2f}" if v
                         else f"{'--':>15s}")
            print(line)
        print(f"    {'alignments':<16s}" + "".join(
            f"{max((sw[k][r]['n_ripple_alignments'] for r in sw[k] if r != '_inert'), default=0):>15,d}"
            for k in keys))
    print("\n" + "=" * 80)


def _smooth(y, fs, ms=SMOOTH_MS):
    """Gaussian smoothing of a time course, in milliseconds.

    The HFB is already a sub-band-averaged envelope at 100 Hz, so this is
    cosmetic rather than a filter with consequences -- but without it a
    +-1 s trace of 200 points is dominated by sample-to-sample wobble and only
    the largest effect is visible. He et al. and Norman et al. both plot
    visibly smoothed traces. The STATISTICS are computed on unsmoothed data;
    smoothing is applied only here.
    """
    if not ms:
        return y
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(y, (ms / 1000.0 * fs) / 2.355, axis=-1)


def _curves(tr, ix, roi, unit, flank, fs, smooth_ms):
    """Per-unit real-minus-null trace for one ROI, baselined to the non-peri band."""
    sel = (ix.roi == roi) & (~ix.same_shaft)
    out = []
    for _, gi in ix[sel].groupby(unit):
        rr = gi.trace_row.to_numpy()
        r = tr[rr[gi.is_real.to_numpy()]]
        n = tr[rr[~gi.is_real.to_numpy()]]
        if not len(r) or not len(n):
            continue
        c = r.mean(0) - n.mean(0)
        out.append(c - c[flank].mean())          # non-peri IS the baseline
    return _smooth(np.stack(out), fs, smooth_ms) if out else None


def figure(results=None, out_stem=None, smooth_ms=SMOOTH_MS):
    """Session-level primary; subject-level as a control figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mc.plotting.ripple_figures as rfig

    R = results
    z = np.load(os.path.join(R, "timecourses.npz"))
    res = json.load(open(os.path.join(R, "result.json")))["results"]
    t_ms = z["t_ms"]
    fs = 1000.0 / (t_ms[1] - t_ms[0])
    C = dict(rfig.MONTAGE_C)
    C[A24_NAME] = "#7B3294"                     # a distinct hue: it is a new group
    CM = 1 / 2.54
    SHORT = {"TemporalLateral": "Lat. temporal", A24_NAME: "A24 (z −10:5)",
             "Auditory": "Auditory", "Visual": "Visual",
             "mPFC": "mPFC", "mOFC": "mOFC"}
    tr = z["traces"]
    ix = pd.read_csv(os.path.join(R, "timecourse_index.csv"))
    ix["same_shaft"] = ix.same_shaft.astype(bool)
    ix["is_real"] = ix.is_real.astype(bool)
    if "trace_row" not in ix.columns:
        ix["trace_row"] = np.arange(len(ix))
    flank = (np.abs(t_ms) >= NONPERI_S[0] * 1000) & (np.abs(t_ms) < NONPERI_S[1] * 1000)

    for unit, tag in (("session", "PRIMARY"), ("subject", "control")):
        present = [r for r in ROI_ORDER if r in res[unit].get("different_shaft", {})]
        n = len(present)
        fig = plt.figure(figsize=(22.0 * CM, 14.0 * CM), constrained_layout=True)
        gs = fig.add_gridspec(2, n, height_ratios=[1.0, 1.25])

        # --- row 1: one panel per ROI, EACH WITH ITS OWN Y-SCALE ------------
        # Sharing an axis lets mOFC set the scale and makes every other region
        # look flat, which is a plotting artefact rather than a result. The
        # traces are already baselined to the non-peri band, so 0 on each axis
        # is that region's own non-peri level and the peak is directly readable.
        for i, roi in enumerate(present):
            ax = fig.add_subplot(gs[0, i])
            A = _curves(tr, ix, roi, unit, flank, fs, smooth_ms)
            if A is None:
                continue
            m, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))
            ax.plot(t_ms, m, color=C.get(roi, "#888"), lw=1.4)
            ax.fill_between(t_ms, m - se, m + se, color=C.get(roi, "#888"),
                            alpha=0.2, lw=0)
            ax.axvline(0, color="0.4", lw=0.7, ls="--")
            ax.axhline(0, color="0.7", lw=0.6)
            ax.axvspan(-PERI_S * 1000, PERI_S * 1000, color="#F15A29",
                       alpha=0.07, lw=0)
            v = res[unit]["different_shaft"][roi]
            ax.set_title(f"{SHORT.get(roi, roi)}\nn={v['n_units']}, "
                         f"p={v['p']:.3f}", fontsize=8, pad=3)
            ax.tick_params(labelsize=7)
            ax.set_xticks([-1000, 0, 1000])
            ax.margins(x=0)
            if i == 0:
                ax.set_ylabel("HFB, real − null\n(vs non-peri, z)", fontsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

        # --- row 2 left: overlay; right: bars ------------------------------
        half = max(1, n // 2)
        ax = fig.add_subplot(gs[1, :half])
        for roi in present:
            A = _curves(tr, ix, roi, unit, flank, fs, smooth_ms)
            if A is None:
                continue
            m = A.mean(0)
            ax.plot(t_ms, m, color=C.get(roi, "#888"), lw=1.5,
                    label=f"{SHORT.get(roi, roi)} ({len(A)})")
        ax.axvline(0, color="0.35", lw=0.8, ls="--")
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvspan(-PERI_S * 1000, PERI_S * 1000, color="#F15A29", alpha=0.07, lw=0)
        ax.set_xlabel("Time from ripple peak (ms)", fontsize=9)
        ax.set_ylabel("HFB, real − null, vs non-peri (z)", fontsize=9)
        ax.set_title("all regions, shared scale", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper left", handlelength=1.3,
                  labelspacing=0.28)
        ax.tick_params(labelsize=8)
        ax.margins(x=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        ax = fig.add_subplot(gs[1, half:])
        vals = [res[unit]["different_shaft"][r] for r in present]
        top = max(v["effect"] + v["sem"] for v in vals)
        bot = min(0.0, min(v["effect"] - v["sem"] for v in vals))
        for i, (roi, v) in enumerate(zip(present, vals)):
            ax.bar(i, v["effect"], yerr=v["sem"], color=C.get(roi, "#888"),
                   width=0.68, capsize=3, error_kw=dict(lw=0.9))
            if v["p"] < 0.05:
                ax.text(i, v["effect"] + v["sem"] + 0.03 * top,
                        "**" if v["p"] < 0.01 else "*", ha="center", fontsize=10)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_ylim(bot - 0.08 * top, top * 1.28)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([f"{SHORT.get(r, r)}\n({v['n_units']})"
                            for r, v in zip(present, vals)], fontsize=7.5,
                           rotation=30, ha="right")
        ax.set_ylabel("peri − non-peri, minus null (z)", fontsize=9)
        ax.set_title(f"effect ± s.e.m.", fontsize=10)
        ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        fig.suptitle(f"Ripple-locked cortical HFB — {unit}-level [{tag}], "
                     f"different-shaft derivations", fontsize=11.5)
        stem = (out_stem or os.path.join(R, "ripple_locked_hfb")) + (
            "" if unit == "session" else "_subject_control")
        fig.savefig(stem + ".pdf")
        fig.savefig(stem + ".png", dpi=300)
        plt.close(fig)
        print(f"figure -> {stem}.pdf / .png")

    # ---- medial-frontal dorsoventral gradient -----------------------------
    zg = res.get("z_gradient") or {}
    if zg.get("_table"):
        e = pd.DataFrame(zg["_table"])
        fig, ax = plt.subplots(figsize=(9.0 * CM, 7.0 * CM), constrained_layout=True)
        ax.axvspan(A24_BAND[0], A24_BAND[1], color="#7B3294", alpha=0.10, lw=0,
                   label=f"A24 band ({A24_BAND[0]:.0f} to {A24_BAND[1]:.0f})")
        for roi, g in e.groupby("roi"):
            ax.scatter(g.mni_z, g.effect, s=16, alpha=0.8,
                       color=C.get(roi, "#888"), edgecolors="none",
                       label=f"{roi} ({len(g)})")
        if len(e) > 2:
            b = np.polyfit(e.mni_z, e.effect, 1)
            xs = np.linspace(e.mni_z.min(), e.mni_z.max(), 50)
            ax.plot(xs, np.polyval(b, xs), color="0.25", lw=1.2, ls="--")
        sp = zg["spearman_z_vs_effect"]
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_xlabel("MNI z (mm)   ventral → dorsal", fontsize=9)
        ax.set_ylabel("ripple-locked HFB effect (z)", fontsize=9)
        ax.set_title(f"Medial frontal dorsoventral gradient\n"
                     f"ρ = {sp['rho']:+.3f}, p = {sp['p']:.3f}, n = {len(e)}",
                     fontsize=10)
        ax.legend(fontsize=7, frameon=False)
        ax.tick_params(labelsize=8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        gs2 = os.path.join(R, "medial_frontal_z_gradient")
        fig.savefig(gs2 + ".pdf")
        fig.savefig(gs2 + ".png", dpi=300)
        plt.close(fig)
        print(f"figure -> {gs2}.pdf / .png")
    return None


# Overlay colours. Medial frontal takes the project's Showgirl2 green (the
# mPFC hue, CLAUDE.md); the two temporal controls are greys, so target and
# control are separable without reading the key.
OVERLAY_C = {"MedialFrontal": "#448363", "mPFC": "#448363", "mOFC": "#DC673E",
             "TemporalLateral": "#8C8C8C", "Auditory": "#4F4F4F",
             "Visual": "#B0A8C0"}
OVERLAY_LABEL = {"MedialFrontal": "medial frontal", "TemporalLateral": "lat. temporal",
                 "Auditory": "auditory", "Visual": "visual",
                 "mPFC": "mPFC", "mOFC": "mOFC"}


def overlay_figure(results=None, out_stem=None,
                   rois=("MedialFrontal", "TemporalLateral", "Auditory"),
                   width_cm=2.5, height_cm=2.5, font_pt=6.0, smooth_ms=100.0,
                   n_perm=2000, seed=42, unit="session", legend=False):
    """One small panel overlaying the target ROI with its control regions.

    Significance bars come from the same cluster-based permutation the rest of
    this project uses (`swr_sakon.cluster_perm_time`, sign-flipping across the
    unit of inference), run on the same smoothed curves that are drawn.

    At 2.5 cm a legend is wider than the panel, so it is off by default and
    belongs in the caption; `_labelled` is written alongside with legend, axis
    titles and the cluster statistics.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    import mc.analyse.swr_sakon as sk

    R = results
    z = np.load(os.path.join(R, "timecourses.npz"))
    t_ms, tr = z["t_ms"], z["traces"]
    fs = 1000.0 / (t_ms[1] - t_ms[0])
    ix = pd.read_csv(os.path.join(R, "timecourse_index.csv"))
    ix["same_shaft"] = ix.same_shaft.astype(bool)
    ix["is_real"] = ix.is_real.astype(bool)
    if "trace_row" not in ix.columns:
        ix["trace_row"] = np.arange(len(ix))
    flank = (np.abs(t_ms) >= NONPERI_S[0] * 1000) & (np.abs(t_ms) < NONPERI_S[1] * 1000)

    curves, stats_out = {}, {}
    for roi in rois:
        sel = (ix.roi == roi) & (~ix.same_shaft)
        if not sel.any():
            continue
        rows = []
        for _, gi in ix[sel].groupby(unit):
            rr = gi.trace_row.to_numpy()
            r = tr[rr[gi.is_real.to_numpy()]]
            n = tr[rr[~gi.is_real.to_numpy()]]
            if not len(r) or not len(n):
                continue
            c = r.mean(0) - n.mean(0)
            rows.append(c - c[flank].mean())
        if len(rows) < 5:
            continue
        A = np.stack(rows)
        if smooth_ms:
            A = gaussian_filter1d(A, (smooth_ms / 1000.0 * fs) / 2.355, axis=-1)
        curves[roi] = A
        _, cl, pv, _ = sk.cluster_perm_time(A, n_perm=n_perm, seed=seed)
        sig = [(float(t_ms[a]), float(t_ms[b - 1]), float(pp))
               for (a, b), pp in zip(cl, pv) if pp < 0.05]
        stats_out[roi] = {"n_units": int(len(A)), "unit": unit, "clusters": sig,
                          "peak_z": float(A.mean(0).max()),
                          "peak_at_ms": float(t_ms[int(np.argmax(A.mean(0)))])}
        print(f"  {OVERLAY_LABEL.get(roi, roi):<16s} n={len(A):>3d} {unit}s  "
              f"peak {A.mean(0).max():+.4f} z at {t_ms[int(np.argmax(A.mean(0)))]:+.0f} ms")
        for a, b, pp in sig:
            print(f"      cluster {a:+.0f} to {b:+.0f} ms, p = {pp:.4f}")
        if not sig:
            print("      no cluster survives correction")
    if not curves:
        raise RuntimeError("no ROI had enough units")

    hi = max(float((A.mean(0) + A.std(0, ddof=1) / np.sqrt(len(A))).max())
             for A in curves.values())
    lo = min(float((A.mean(0) - A.std(0, ddof=1) / np.sqrt(len(A))).min())
             for A in curves.values())
    span = hi - lo

    for suffix, wcm, hcm, fpt, lab in (("", width_cm, height_cm, font_pt, legend),
                                       ("_labelled", 9.0, 7.0, 9.0, True)):
        fig, ax = plt.subplots(figsize=(wcm / 2.54, hcm / 2.54),
                               constrained_layout=True)
        for i, (roi, A) in enumerate(curves.items()):
            m = A.mean(0)
            se = A.std(0, ddof=1) / np.sqrt(len(A))
            col = OVERLAY_C.get(roi, "#888")
            ax.plot(t_ms, m, color=col, lw=1.1 if not lab else 1.5,
                    solid_capstyle="round",
                    label=f"{OVERLAY_LABEL.get(roi, roi)} ({len(A)})")
            ax.fill_between(t_ms, m - se, m + se, color=col, alpha=0.16, lw=0)
            y = hi + span * (0.10 + 0.085 * i)
            for a, b, _pp in stats_out[roi]["clusters"]:
                ax.plot([a, b], [y, y], color=col, lw=1.8 if not lab else 2.6,
                        solid_capstyle="butt", clip_on=False)
        ax.axvline(0, color="0.45", lw=0.6, ls=(0, (2, 2)))
        ax.axhline(0, color="0.75", lw=0.5)
        ax.set_xlim(t_ms[0], t_ms[-1])
        ax.set_ylim(lo - span * 0.10, hi + span * (0.10 + 0.085 * len(curves)))
        ax.set_xticks([-500, 0, 500] if wcm >= 4 else [-500, 500])
        ax.set_yticks([0.0, float(np.round(hi, 2))])
        ax.tick_params(labelsize=fpt - 0.5, length=2.0, pad=1.0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if lab:
            ax.set_xlabel("Time from hippocampal ripple peak (ms)", fontsize=fpt)
            ax.set_ylabel("HFB, real − shifted null, vs non-peri (z)", fontsize=fpt)
            ax.set_title("bars = cluster-corrected p < 0.05", fontsize=fpt + 1)
            ax.legend(fontsize=fpt - 1, frameon=False, loc="upper left")
        else:
            ax.set_xlabel("ms from ripple", fontsize=fpt, labelpad=1)
            ax.set_ylabel("HFB (z)", fontsize=fpt, labelpad=1)
        stem = (out_stem or os.path.join(R, "hfb_overlay")) + suffix
        fig.savefig(stem + ".pdf")
        fig.savefig(stem + ".png", dpi=600)
        plt.close(fig)
        print(f"figure -> {stem}.pdf / .png   ({wcm:.1f} x {hcm:.1f} cm, {fpt} pt)")

    with open(os.path.join(R, "hfb_overlay_clusters.json"), "w") as f:
        json.dump({"rois": list(curves), "unit": unit, "smooth_ms": smooth_ms,
                   "n_perm": n_perm, "seed": seed, "colours": OVERLAY_C,
                   "stats": stats_out}, f, indent=2)
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run, "figure": figure,
                   "overlay_figure": overlay_figure})
    else:
        run()
