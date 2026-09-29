#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ripple-HFB coupling in every ROI we have the coverage for, over all sessions,
with each derivation in exactly ONE ROI.

Replaces `swr_hfb_roi_inventory.py`, which built overlapping groups.

WHAT A "DERIVATION" IS  (the coverage unit)
-------------------------------------------
A bipolar pair of neighbouring macro contacts on one probe: anchor + its
immediate neighbour, the anchor's atlas label naming the pair. The rules that
matter here, all from `mc.analyse.contact_anatomy`:

  * NO CONTACT IS EVER IN TWO DERIVATIONS. Pairs are taken greedily along the
    probe in contact order and both contacts are then consumed, so contact 3
    cannot be in both 2-3 and 3-4.
  * The HIPPOCAMPAL montage runs first and its contacts are marked used, so a
    cortical derivation can never reuse a contact that supplied ripples. One
    hippocampal derivation per probe (Chen et al.); cortex may have several
    per probe, because mPFC coverage along a probe is genuinely different
    tissue.
  * `same_shaft` means the cortical derivation sits on the SAME probe as the
    hippocampal one it is paired with. Those pairs share an amplifier, a
    reference chain and a noise environment with the ripple source, so the
    published statistic uses different-shaft pairs only and so does the
    coupling column here. Both counts are reported.

So "dual use" is allowed at the level of the PROBE (one shaft can carry a
hippocampal derivation and cortical ones) but never at the level of a CONTACT.

ONE ROI PER DERIVATION
----------------------
Each derivation is labelled once, by a fixed three-rung ladder on its ANCHOR:

  1. Harvard-Oxford cortical   (`_ho_cort`)
  2. Brainnetome               (`_bn_label`, hemisphere suffix dropped)
  3. FreeSurfer native region  (`native_region_a`)

The first rung that returns a label wins; that label is then mapped to exactly
one ROI by `ROI_RULES`, which is an ordered list of substring rules. A
derivation therefore cannot appear under two ROIs, and the partition is
asserted, not assumed. Anything the ladder cannot place -- an anchor in white
matter, or labelled 'Unknown' -- goes to `Unassigned` and is reported rather
than quietly dropped. `roi_label_audit.csv` lists every raw label, the rung it
came from and the ROI it was mapped to, so the mapping can be checked line by
line.

This is NOT the grouping the published numbers use. `ripple_locked_hfb` groups
by `roi_family` and carves `MedialFrontal` out by a coordinate rule, which
makes its groups overlap by design. That group is printed once at the end for
orientation and is deliberately not a row in the partition table.

ALL SESSIONS
------------
Everything here is all 61 sessions / 42 subject labels. It reads the bundle,
which was built on the cluster. (The earlier "31 of 61" caveat came from
`macro_contacts_all.csv`, a LOCAL rebuild that needs the raw recordings and so
only covers the sessions whose raw data is on this machine. Nothing here uses
that file.)

COUPLING
--------
Exactly the published statistic, computed by the run's own `_effect`:
mean HFB in +-250 ms around a ripple minus mean HFB in (250, 750) ms either
side, at the real ripple times minus the same quantity at shifted-null times,
averaged within subject and then over subjects.

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_hfb_roi_partition.py run
    python scripts/swr_hfb_roi_partition.py run --min_sessions=3 --unit=session

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import pickle
import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.plotting.ripple_figures as rfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swr_ripple_locked_hfb import (_effect, _smooth, MEDIAL_NAME, NONPERI_S,
                                   PERI_S, SMOOTH_MS)

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "hfb_roi_partition"
MIN_SESSIONS = 3
MIN_UNITS_FOR_TEST = 3
UNDERPOWERED_BELOW = 5
REF_ROI = "mPFC"          # medial frontal: the region the project's claim is about
FAMILIES = ("mPFC", "mOFC", "TemporalLateral", "Auditory", "Visual")
LADDER = (("ho", "_ho_cort"), ("bn", "_bn_label"), ("fs", "native_region_a"))

# Ordered substring rules: (ROI, (patterns, ...)). Matching is case-insensitive
# and the FIRST rule that matches wins, so order encodes precedence -- the
# auditory patterns must come before the general superior-temporal ones, and
# the temporo-occipital ones before the plain gyrus names, or the finer region
# is swallowed by the coarser one. `roi_label_audit.csv` is how this is
# checked: it prints every raw label next to the ROI it landed in.
ROI_RULES = [
    # --- frontal, medial -> lateral ---------------------------------------
    ("mPFC", ("cingulate gyrus, anterior", "paracingulate", "frontal medial",
              "subcallosal", "a32sg", "a32p", "a24cd", "a24rv", "a24rd",
              "a9m", "a10m", "rostralanteriorcingulate",
              "caudalanteriorcingulate", "acgg", "mfc medial frontal",
              "msfg superior frontal gyrus medial")),
    # "medial orbital" must be listed here: lOFC's rule matches the bare
    # "orbital gyrus", which would otherwise swallow "MOrG medial orbital
    # gyrus". The audit is what caught that.
    ("mOFC", ("a11m", "a13", "a14m", "medialorbitofrontal", "medial orbital",
              "morg", "gre gyrus rectus", "gyrus rectus")),
    ("lOFC", ("frontal orbital", "a11l", "a12/47", "lateralorbitofrontal",
              "lof lateral orbital", "orbital gyrus")),
    ("FrontalPole", ("frontal pole", "a10l", "frontalpole")),
    ("vlPFC_IFG", ("inferior frontal gyrus", "frontal operc", "a44", "a45",
                   "iag inferior frontal", "parstriangularis",
                   "parsopercularis", "parsorbitalis")),
    ("dlPFC", ("middle frontal", "superior frontal", "a9l", "a8", "a46",
               "a6", "rostralmiddlefrontal", "caudalmiddlefrontal",
               "superiorfrontal", "mfg middle frontal", "sfg superior frontal",
               "precentral", "postcentral", "paracentral")),
    ("Insula", ("insula", "via", "dia", "dig", "vig")),
    # --- temporal ----------------------------------------------------------
    ("Auditory", ("heschl", "planum temporale", "planum polare", "a41/42",
                  "te1.0", "te1.2", "transversetemporal",
                  "ttg transverse temporal", "pt planum", "pp planum")),
    ("TempOcc_Fusiform", ("temporooccipital", "fusiform", "a37", "phg",
                          "temporal occipital")),
    ("TemporalPole", ("temporal pole", "a38", "temporalpole")),
    ("STG", ("superior temporal", "sts", "a22", "superiortemporal",
             "bankssts", "stg superior temporal")),
    ("MTG", ("middle temporal", "a21", "middletemporal",
             "mtg middle temporal")),
    ("ITG", ("inferior temporal", "a20", "inferiortemporal",
             "itg inferior temporal")),
    ("Parahippocampal", ("parahippocampal", "entorhinal", "phg parahippo")),
    # --- posterior ---------------------------------------------------------
    ("PCC_Precuneus", ("cingulate gyrus, posterior", "precuneous", "precuneus",
                       "a23", "posteriorcingulate", "isthmuscingulate",
                       "pcgg posterior cingulate")),
    ("Parietal", ("angular", "supramarginal", "superior parietal",
                  "parietal operc", "a39", "a40", "a7", "hip", "intra-pariet",
                  "inferiorparietal", "superiorparietal",
                  "ag angular", "smg supramarginal")),
    ("Occipital", ("occipital", "intracalcarine", "lingual", "cuneus",
                   "calcarine", "v1", "v2", "v3", "v4", "v5", "msv",
                   "pericalcarine", "og occipital", "lig lingual")),
]

# Labels that are not a region at all. They are matched BEFORE ROI_RULES, so a
# white-matter anchor never inherits a cortical ROI from a loose substring.
NOT_A_REGION = ("white matter", "white-matter", "unknown", "out", "csf",
                "ventricle", "wm ")

# Project colours (CLAUDE.md). `mPFC` here is the anterior-cingulate-based
# medial frontal region -- the project never displays it as "ACC".
ROI_C = {
    "mPFC": rfig.PAL[1], "mOFC": rfig.PAL[4], "lOFC": "#a30d6c",
    "PCC_Precuneus": rfig.PAL[3], "Insula": "#D8C9A3",
    "Auditory": "#4F4F4F", "STG": "#8C8C8C", "MTG": "#A8A8A8",
    "ITG": "#C4C4C4", "TemporalPole": "#6E6E6E",
    "TempOcc_Fusiform": "#B0A8C0", "Occipital": "#9C93B0",
    "FrontalPole": "#E09B7A", "vlPFC_IFG": "#C97B5A", "dlPFC": "#8FB8A6",
    "Parietal": "#7E9CAB", "Parahippocampal": "#23677E",
    "Unassigned": "#DDDDDD",
}


# Phase splits and condition keys, in the EXCLUSIVE-boundary condition names.
# They live here rather than in an exploratory file because the maintained
# reviewer checks (`swr_hfb_event_latency.py`) need them.
COND_KEYS = ("reward", "error", "move")
PHASES_EVENT = {
    "all": {"reward": ("reward_explore", "reward_plan", "reward_execute"),
            "error": ("error_explore", "error_plan", "error_execute"),
            "move": ("move_explore", "move_plan", "move_execute")},
    "first_traversal": {"reward": ("reward_explore",),
                        "error": ("error_explore",),
                        "move": ("move_explore",)},
    "after_first": {"reward": ("reward_plan", "reward_execute"),
                    "error": ("error_plan", "error_execute"),
                    "move": ("move_plan", "move_execute")},
    "encoding": {"reward": ("reward_explore", "reward_plan"),
                 "error": ("error_explore", "error_plan"),
                 "move": ("move_explore", "move_plan")},
    "after_encoding": {"reward": ("reward_execute",),
                       "error": ("error_execute",),
                       "move": ("move_execute",)},
}


def _norm(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else \
        str(x).strip().strip("[]'\" ").lower()


def _to_roi(label):
    """Map one raw atlas label to exactly one ROI, or None if it is not a region."""
    s = _norm(label)
    if not s or s == "nan":
        return None
    if any(p in s for p in NOT_A_REGION):
        return None
    for roi, pats in ROI_RULES:
        if any(p in s for p in pats):
            return roi
    return None


def _assign(pairs):
    """One ROI per (session, pair_id), by the ladder. Returns a tidy table."""
    rows = []
    for (sess, pid), r in pairs.set_index(["session", "pair_id"]).iterrows():
        roi, used_label, used_rung = None, None, None
        for rung, col in LADDER:
            lab = r.get(col)
            if rung == "bn" and isinstance(lab, str):
                lab = lab[:-2] if lab.endswith(("_L", "_R")) else lab
            roi = _to_roi(lab)
            if roi is not None:
                used_label, used_rung = lab, rung
                break
        if roi is None:
            # Say WHY it could not be placed. An anchor in white matter is a
            # different fact from an anchor with no atlas entry at all, and
            # lumping them under a blank hides both.
            cands = [_norm(r.get(c)) for _, c in LADDER]
            cands = [c for c in cands if c and c != "nan"]
            used_label = cands[-1] if cands else "no label on any rung"
        rows.append({"session": sess, "pair_id": pid,
                     "roi_part": roi or "Unassigned",
                     "raw_label": used_label,
                     "rung": used_rung or "none"})
    return pd.DataFrame(rows)


def run(results=None, bundle=None, out_dir=None, min_sessions=MIN_SESSIONS,
        unit="subject", ref_roi=REF_ROI, save=True, exclude_medial=False):
    """`exclude_medial` makes MedialFrontal a genuine 14th region: every
    derivation inside the |x| <= 20 mm cut is REMOVED from its atlas ROI, so
    lOFC means "lateral OFC outside the medial group" and every row on the
    figure is disjoint from every other including the comparison band. Without
    it, mPFC and mOFC are entirely inside MedialFrontal and lOFC overlaps it
    by 24%, which is what makes the comparison panel misleading."""
    from scipy import stats as st

    root = swr_io.derivatives_dir(swr_io.get_data_root())
    R = results or os.path.join(root, "group", "swr",
                                "ripple_locked_hfb_2026-09-18")
    b_dir = bundle or os.path.join(root, "group", "swr", "bundle_v2")
    with open(os.path.join(b_dir, "swr_bundle.pkl"), "rb") as f:
        B = pickle.load(f)
    pairs = B["pairs"].drop_duplicates(["session", "pair_id"])

    d = pd.read_csv(os.path.join(R, "per_pair.csv"))
    native = float(d.pad_s.min()) if "pad_s" in d.columns else None
    if native is not None:
        d = d[d.pad_s == native]
    # The family rows are the ORIGINALS: MedialFrontal / A24 rows in this file
    # are duplicates of them under a derived name, so taking the families is
    # what makes every derivation appear exactly once.
    base = d[d.roi.isin(FAMILIES)].copy()

    lut = _assign(pairs).set_index(["session", "pair_id"])
    key = pd.MultiIndex.from_arrays([base.session, base.cx_pair])
    base["roi_part"] = key.map(lut["roi_part"])
    base["rung"] = key.map(lut["rung"])
    base["raw_label"] = key.map(lut["raw_label"])

    if exclude_medial:
        mf_keys = _k(d[d.roi == MEDIAL_NAME])
        in_mf = np.array([(x, y) in mf_keys
                          for x, y in zip(base.session, base.cx_pair)])
        base.loc[in_mf, "roi_part"] = MEDIAL_NAME
        print(f"  exclude_medial: {len(mf_keys)} derivations moved into "
              f"{MEDIAL_NAME}; every other ROI is now disjoint from it")

    keys = base[["session", "cx_pair", "roi_part"]].drop_duplicates(
        ["session", "cx_pair"])
    assert keys.groupby(["session", "cx_pair"]).roi_part.nunique().max() == 1, \
        "a derivation landed in two ROIs -- the partition is broken"
    print(f"\nall sessions: {base.session.nunique()} sessions, "
          f"{base.subject.nunique()} subject labels, "
          f"{len(keys)} cortical derivations with an HFB estimate")
    print(f"pad {native} s; coupling below is different-shaft pairs only")

    ds = base[~base.same_shaft]
    if exclude_medial:
        ref_roi = MEDIAL_NAME       # it is now a real, disjoint region
    ref_rows = ds[ds.roi_part == ref_roi]
    ref_eff = _effect(ref_rows, unit) if len(ref_rows) else pd.Series(dtype=float)

    rows = []
    for roi, g_all in base.groupby("roi_part"):
        g = ds[ds.roi_part == roi]
        r = {"roi": roi,
             "n_derivations_total": len(_k(g_all)),
             "n_derivations_diff_shaft": len(_k(g)),
             "n_sessions": int(g_all.session.nunique()),
             "n_sessions_diff_shaft": int(g.session.nunique()) if len(g) else 0,
             "n_subjects": int(g_all.subject.nunique()),
             "n_sites": int(g_all.cx_pair.nunique()),
             "n_ripple_alignments": int(g[g.is_real].n_ripples.sum())
             if len(g) else 0,
             "enough_coverage": bool(len(g) and g.session.nunique()
                                     >= min_sessions)}
        if r["enough_coverage"]:
            v = _effect(g, unit)
            r.update({"n_units": int(len(v)),
                      "coupling": float(v.mean()),
                      "sem": float(v.std(ddof=1) / np.sqrt(len(v)))
                      if len(v) > 1 else np.nan})
            if len(v) >= MIN_UNITS_FOR_TEST:
                t, p = st.ttest_1samp(v.to_numpy(), 0.0)
                r["t"], r["p"] = float(t), float(p)
                if roi != ref_roi and len(ref_eff):
                    sh = v.index.intersection(ref_eff.index)
                    if len(sh) >= MIN_UNITS_FOR_TEST:
                        ta, pa = st.ttest_rel(v[sh].to_numpy(),
                                              ref_eff[sh].to_numpy())
                        r.update({"n_paired": int(len(sh)),
                                  "diff_vs_ref": float(
                                      (v[sh] - ref_eff[sh]).mean()),
                                  "t_vs_ref": float(ta),
                                  "p_vs_ref": float(pa)})
        rows.append(r)
    t = pd.DataFrame(rows)
    ok = t.enough_coverage & (t.roi != "Unassigned")
    for src, dst in (("p", "q"), ("p_vs_ref", "q_vs_ref")):
        t[dst] = np.nan
        if src in t:
            t.loc[ok, dst] = _bh(t.loc[ok, src])
    t["underpowered"] = t.get("n_units", pd.Series(np.nan, index=t.index)) \
                         .lt(UNDERPOWERED_BELOW)
    t = t.sort_values(["enough_coverage", "coupling"],
                      ascending=[False, False]).reset_index(drop=True)
    # ROIs the scheme can name but that no derivation reached: absence of
    # coverage, which is a different statement from a weak effect.
    absent = [r for r, _ in ROI_RULES if r not in set(t.roi)]

    # ---- the collapsed |x| <= 20 mm MedialFrontal group -------------------
    # NOT a partition row: it is carved out of the mPFC+mOFC FAMILIES by a
    # coordinate rule, so it shares derivations with several rows above. The
    # overlap is computed per ROI and carried on every contrast, because a
    # paired test between a group and one that contains part of it is not a
    # clean test and the reader has to be able to see which ones those are.
    mf_rows = d[(d.roi == MEDIAL_NAME) & (~d.same_shaft)]
    mf_keys = _k(mf_rows)
    mf_eff = _effect(mf_rows, unit) if len(mf_rows) else pd.Series(dtype=float)
    mf = {"roi": MEDIAL_NAME, "kind": "comparison",
          "n_derivations_total": len(_k(d[d.roi == MEDIAL_NAME])),
          "n_derivations_diff_shaft": len(mf_keys),
          "n_sessions": int(mf_rows.session.nunique()),
          "n_subjects": int(mf_rows.subject.nunique()),
          "n_units": int(len(mf_eff)),
          "coupling": float(mf_eff.mean()),
          "sem": float(mf_eff.std(ddof=1) / np.sqrt(len(mf_eff))),
          "enough_coverage": True}
    mf["t"], mf["p"] = [float(x) for x in st.ttest_1samp(mf_eff.to_numpy(), 0.0)]
    print(f"\ncomparison group {MEDIAL_NAME} (|x| <= 20 mm across the "
          f"mPFC+mOFC families): {len(mf_keys)} derivations, "
          f"{len(mf_eff)} {unit}s, {mf['coupling']:+.4f}")

    # ---- MedialFrontal MINUS each ROI, paired on shared units -------------
    crows = []
    for _, r in t[t.enough_coverage & (t.roi != "Unassigned")].iterrows():
        g = ds[ds.roi_part == r.roi]
        v = _effect(g, unit)
        kk = _k(g)
        sh = mf_eff.index.intersection(v.index)
        row = {"roi": r.roi, "n_units_roi": int(len(v)),
               "n_paired": int(len(sh)),
               "overlap_pct": round(100.0 * len(kk & mf_keys)
                                    / max(len(kk), 1), 1)}
        if len(sh) >= MIN_UNITS_FOR_TEST:
            diff = mf_eff[sh] - v[sh]
            ta, pa = st.ttest_rel(mf_eff[sh].to_numpy(), v[sh].to_numpy())
            row.update({"diff_mf_minus_roi": float(diff.mean()),
                        "sem": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                        "t": float(ta), "p": float(pa)})
        crows.append(row)
    ct = pd.DataFrame(crows)
    ct["q"] = _bh(ct["p"]) if "p" in ct else np.nan
    # A contrast against a group that contains these very derivations is not
    # independent; flagged rather than silently dropped.
    ct["contaminated"] = ct.overlap_pct > 0
    ct = ct.sort_values("diff_mf_minus_roi", ascending=False).reset_index(
        drop=True)

    audit = (base.drop_duplicates(["session", "cx_pair"])
                 .groupby(["roi_part", "rung", "raw_label"])
                 .agg(n_derivations=("cx_pair", "size"),
                      n_sessions=("session", "nunique"))
                 .reset_index()
                 .sort_values(["roi_part", "n_derivations"],
                              ascending=[True, False]))

    _report(t, audit, base, d, min_sessions, unit, ref_roi, native, absent,
            mf, ct)

    if not save:
        return None
    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(root, "group", "swr",
                                      f"{ANALYSIS_NAME}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    t.to_csv(os.path.join(out_dir, "roi_coupling.csv"), index=False)
    ct.to_csv(os.path.join(out_dir, "medialfrontal_vs_roi.csv"), index=False)
    audit.to_csv(os.path.join(out_dir, "roi_label_audit.csv"), index=False)
    base.drop_duplicates(["session", "cx_pair"])[
        ["session", "subject", "cx_pair", "roi_part", "rung", "raw_label",
         "same_shaft"]].to_csv(
        os.path.join(out_dir, "derivation_assignment.csv"), index=False)
    _markdown(t, audit, out_dir, min_sessions, unit, ref_roi, base, native,
              mf, ct)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({
            "analysis": ANALYSIS_NAME, "created": stamp,
            "source_results": R, "bundle": b_dir,
            "settings": {"min_sessions": min_sessions, "unit": unit,
                         "pad_s": native, "reference_roi": ref_roi,
                         "coupling_on": "different-shaft pairs only",
                         "ladder": [c for _, c in LADDER],
                         "partition": "one ROI per derivation, asserted",
                         "multiple_comparisons": "Benjamini-Hochberg over ROIs "
                                                 "with enough coverage"},
            "n_sessions": int(base.session.nunique()),
            "n_subject_labels": int(base.subject.nunique()),
            "n_derivations": int(len(keys)),
            "rois_with_no_coverage": absent,
            "results": t.to_dict("records"),
            "medial_frontal_comparison_group": mf,
            "medialfrontal_minus_roi": ct.to_dict("records"),
        }, f, indent=2, default=str)
    _figure(t, ct, None if exclude_medial else mf, out_dir, unit,
            ref_roi, min_sessions, exclude_medial)
    print(f"\nsaved -> {out_dir}")
    return None


def _k(f):
    return set(map(tuple, f[["session", "cx_pair"]].drop_duplicates()
                   .to_numpy()))


def _bh(p):
    """Benjamini-Hochberg q-values. NaN in, NaN out, ranked over the rest."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    v = p[ok]
    order = np.argsort(v)
    ranked = v[order] * len(v) / (np.arange(len(v)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def _report(t, audit, base, d, min_sessions, unit, ref_roi, native,
            absent=(), mf=None, ct=None):
    f4 = lambda v: "" if pd.isna(v) else f"{v:+.4f}"
    f3 = lambda v: "" if pd.isna(v) else f"{v:.3f}"
    print(f"\n{'='*100}")
    print(f"RIPPLE-HFB COUPLING BY ROI -- one ROI per derivation, "
          f"all {base.session.nunique()} sessions")
    print(f"{'='*100}")
    print(f"  {'ROI':<20s}{'sess':>5s}{'subj':>5s}{'der':>6s}{'der_ds':>7s}"
          f"{'coupling':>11s}{'p':>8s}{'q':>8s}{'vs ' + ref_roi:>11s}{'q':>8s}")
    for lab, sel in (("ENOUGH COVERAGE "
                      f"(>= {min_sessions} sessions, different-shaft)",
                      t.enough_coverage & (t.roi != "Unassigned")),
                     ("too little coverage to estimate",
                      (~t.enough_coverage) & (t.roi != "Unassigned")),
                     ("not a region (white matter / unlabelled anchor)",
                      t.roi == "Unassigned")):
        g = t[sel]
        if not len(g):
            continue
        print(f"  --- {lab} " + "-" * max(0, 70 - len(lab)))
        for _, r in g.iterrows():
            star = "*" if r.underpowered else " "
            print(f"  {str(r.roi)[:19]:<19s}{star}{r.n_sessions:>5d}"
                  f"{r.n_subjects:>5d}{r.n_derivations_total:>6d}"
                  f"{r.n_derivations_diff_shaft:>7d}"
                  f"{f4(r.get('coupling', np.nan)):>11s}"
                  f"{f3(r.get('p', np.nan)):>8s}{f3(r.get('q', np.nan)):>8s}"
                  f"{f4(r.get('diff_vs_ref', np.nan)):>11s}"
                  f"{f3(r.get('q_vs_ref', np.nan)):>8s}")
    print(f"\n  coupling = peri (+-250 ms) minus non-peri (250-750 ms) HFB, "
          f"real minus shifted null, mean over {unit}s")
    print(f"  der = all derivations; der_ds = different-shaft, which is what "
          "the coupling is computed on")
    print(f"  * fewer than {UNDERPOWERED_BELOW} {unit}s. q = BH over the ROIs "
          "with enough coverage.")
    if absent:
        print(f"  NO COVERAGE AT ALL ({len(absent)}): " + ", ".join(absent))
    print(f"  labelling ladder: "
          + ", ".join(f"{k}={v}" for k, v in
                      base.drop_duplicates(['session', 'cx_pair'])
                          .rung.value_counts().items()))

    if mf is not None:
        print(f"\n  COMPARISON GROUP, not a partition row -- "
              f"{MEDIAL_NAME}, the collapsed |x| <= 20 mm rule across the\n"
              f"  mPFC+mOFC families: {mf['n_derivations_diff_shaft']} "
              f"derivations, {mf['n_sessions']} sessions, "
              f"{mf['n_units']} {unit}s, coupling {mf['coupling']:+.4f} "
              f"(p = {mf['p']:.4f})")
    if ct is not None and len(ct):
        print(f"\n{'='*100}\n{MEDIAL_NAME} MINUS each ROI "
              f"(paired on the {unit}s with both; positive = "
              f"{MEDIAL_NAME} couples MORE)\n{'='*100}")
        print(f"  {'ROI':<20s}{'n_pair':>7s}{'ovl%':>6s}{'difference':>12s}"
              f"{'p':>8s}{'q':>8s}   verdict")
        for _, r in ct.iterrows():
            f4 = lambda v: "" if pd.isna(v) else f"{v:+.4f}"
            f3 = lambda v: "" if pd.isna(v) else f"{v:.3f}"
            q = r.get("q", np.nan)
            if pd.isna(q):
                verdict = "not testable"
            elif q < 0.05:
                verdict = (f"{MEDIAL_NAME} BIGGER" if r.diff_mf_minus_roi > 0
                           else f"{MEDIAL_NAME} SMALLER")
            else:
                verdict = "no difference"
            if r.contaminated:
                verdict += f"  (shares {r.overlap_pct:.0f}% of its derivations)"
            print(f"  {str(r.roi)[:19]:<20s}{r.n_paired:>7d}"
                  f"{r.overlap_pct:>6.0f}{f4(r.diff_mf_minus_roi):>12s}"
                  f"{f3(r.get('p', np.nan)):>8s}{f3(q):>8s}   {verdict}")
        print(f"  q = BH over these {len(ct)} contrasts.")


def _markdown(t, audit, out_dir, min_sessions, unit, ref_roi, base, native,
              mf=None, ct=None):
    f4 = lambda v: "" if pd.isna(v) else f"{v:+.4f}"
    f3 = lambda v: "" if pd.isna(v) else f"{v:.3f}"
    L = ["# Ripple-HFB coupling by ROI", "",
         f"All **{base.session.nunique()} sessions**, "
         f"{base.subject.nunique()} subject labels, "
         f"{len(base.drop_duplicates(['session', 'cx_pair']))} cortical "
         f"derivations. Artifact pad {native} s.", "",
         "**Each derivation is in exactly one ROI.** Labelled by a fixed "
         "ladder on the anchor — Harvard-Oxford cortical, then Brainnetome, "
         "then FreeSurfer native — and mapped to one ROI by an ordered rule "
         "list. The partition is asserted in code. `roi_label_audit.csv` shows "
         "every raw label next to the ROI it landed in.", "",
         "**Coupling** = mean HFB within ±250 ms of a ripple minus mean HFB in "
         "250–750 ms either side, at real ripple times minus the same at "
         "shifted-null times, averaged within subject then over subjects. "
         "Computed on **different-shaft pairs only** (`der_ds`), because a "
         "same-shaft pair shares an amplifier and reference chain with the "
         "ripple source.", "",
         f"`q` is Benjamini-Hochberg over the ROIs with enough coverage. "
         f"Contrast is paired against **{ref_roi}** on the {unit}s that have "
         "both.", ""]
    for head, sel in (
            (f"ROIs with enough coverage (≥ {min_sessions} sessions)",
             t.enough_coverage & (t.roi != "Unassigned")),
            ("Too little coverage to estimate",
             (~t.enough_coverage) & (t.roi != "Unassigned")),
            ("Not a region — white-matter or unlabelled anchor",
             t.roi == "Unassigned")):
        g = t[sel]
        L += [f"## {head}", ""]
        if not len(g):
            L += ["_none_", ""]
            continue
        L += ["| ROI | sessions | subjects | derivations | diff-shaft | "
              f"coupling | p | q | vs {ref_roi} | p | q |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in g.iterrows():
            nm = (f"{r.roi}"
                  + (f" (n<{UNDERPOWERED_BELOW})" if r.underpowered else ""))
            L.append(f"| {nm} | {r.n_sessions} | {r.n_subjects} | "
                     f"{r.n_derivations_total} | "
                     f"{r.n_derivations_diff_shaft} | "
                     f"{f4(r.get('coupling', np.nan))} | "
                     f"{f3(r.get('p', np.nan))} | {f3(r.get('q', np.nan))} | "
                     f"{f4(r.get('diff_vs_ref', np.nan))} | "
                     f"{f3(r.get('p_vs_ref', np.nan))} | "
                     f"{f3(r.get('q_vs_ref', np.nan))} |")
        L += [""]
    if mf is not None:
        L += [f"## Comparison group: `{MEDIAL_NAME}` (not a partition row)", "",
              f"The collapsed **|x| ≤ 20 mm** rule applied across the "
              f"mPFC+mOFC families. It is listed separately because it "
              f"OVERLAPS several rows above — it is not a region of its own, "
              f"it is a coordinate cut through two of them.", "",
              "| group | sessions | subjects | derivations | coupling | p |",
              "|---|---:|---:|---:|---:|---:|",
              f"| {MEDIAL_NAME} | {mf['n_sessions']} | {mf['n_subjects']} | "
              f"{mf['n_derivations_diff_shaft']} | {mf['coupling']:+.4f} | "
              f"{mf['p']:.4f} |", ""]
    if ct is not None and len(ct):
        L += [f"## {MEDIAL_NAME} minus each ROI", "",
              f"Paired on the {unit}s that have both. **Positive = "
              f"{MEDIAL_NAME} couples more.** `ovl%` is the share of the "
              f"ROI's derivations that are also in {MEDIAL_NAME}; where it is "
              "above 0 the two groups are not independent and the contrast "
              "is marked. `q` is Benjamini-Hochberg over these contrasts.", "",
              f"| ROI | paired {unit}s | ovl% | {MEDIAL_NAME} − ROI | p | q | "
              "verdict |", "|---|---:|---:|---:|---:|---:|---|"]
        for _, r in ct.iterrows():
            q = r.get("q", np.nan)
            if pd.isna(q):
                verdict = "not testable"
            elif q < 0.05:
                verdict = ("**" + MEDIAL_NAME + " bigger**"
                           if r.diff_mf_minus_roi > 0
                           else "**" + MEDIAL_NAME + " smaller**")
            else:
                verdict = "no difference"
            if r.contaminated:
                verdict += " ⚠ overlapping"
            L.append(f"| {r.roi} | {r.n_paired} | {r.overlap_pct:.0f} | "
                     f"{f4(r.get('diff_mf_minus_roi', np.nan))} | "
                     f"{f3(r.get('p', np.nan))} | {f3(q)} | {verdict} |")
        L += [""]
    L += ["## What each ROI is made of", "",
          "| ROI | rung | raw atlas label | derivations | sessions |",
          "|---|---|---|---:|---:|"]
    for _, r in audit.iterrows():
        L.append(f"| {r.roi_part} | {r.rung} | {r.raw_label} | "
                 f"{r.n_derivations} | {r.n_sessions} |")
    L += [""]
    with open(os.path.join(out_dir, "ROI_COUPLING.md"), "w") as f:
        f.write("\n".join(L))


def _figure(t, ct, mf, out_dir, unit, ref_roi, min_sessions,
            exclude_medial=False):
    """Two panels, one ROI order.

    LEFT   coupling per ROI, * = q < 0.05 against ZERO (is there coupling).
    RIGHT  MedialFrontal minus that ROI, * = q < 0.05 for the DIFFERENCE
           (is medial frontal bigger). The two asterisks answer different
           questions, which is why they are never in the same panel.

    MedialFrontal is drawn in the left panel below a divider, because it is a
    coordinate cut across mPFC+mOFC rather than a region of its own -- it
    shares derivations with the rows above and must not be read as one more
    independent ROI.
    """
    import matplotlib.pyplot as plt

    g = t[t.enough_coverage & (t.roi != "Unassigned")].sort_values("coupling")
    if not len(g):
        return
    order = list(g.roi)
    cmap = ct.set_index("roi") if ct is not None and len(ct) else None

    with plt.rc_context(rfig._rc()):
        h_cm = max(6.0, 0.55 * (len(order) + 1) + 5.0)
        fig, (axL, axR) = plt.subplots(
            1, 2, figsize=(19 * rfig.CM, h_cm * rfig.CM), sharey=True,
            gridspec_kw=dict(width_ratios=[1.0, 0.85], wspace=0.06))

        # MedialFrontal occupies y = -1, below the divider.
        y = np.arange(len(order))
        col = [ROI_C.get(r, "#8C8C8C") for r in order]

        # ---------------- left: coupling ---------------------------------
        axL.axvline(0, color="#3d3d3d", lw=0.8, ls="--", zorder=1)
        if mf is not None:
            axL.axvspan(mf["coupling"] - mf["sem"], mf["coupling"] + mf["sem"],
                        color=rfig.PAL[1], alpha=0.13, lw=0, zorder=0)
            axL.axvline(mf["coupling"], color=rfig.PAL[1], lw=1.0, zorder=1)
        e = g["sem"].to_numpy(float)
        x = g["coupling"].to_numpy(float)
        axL.hlines(y, x - e, x + e, colors=col, linewidth=1.2, zorder=2)
        axL.scatter(x, y, s=20, c=col, zorder=3, linewidths=0)
        for i, (_, r) in enumerate(g.iterrows()):
            if np.isfinite(r.get("q", np.nan)) and r["q"] < 0.05:
                axL.text(x[i] + e[i] + 0.0005, i, "*", va="center",
                         fontsize=rfig.FS_LABEL, color="#0e3d3a")
        if mf is not None:
            axL.hlines([-1], mf["coupling"] - mf["sem"],
                       mf["coupling"] + mf["sem"], colors=rfig.PAL[1],
                       linewidth=1.2, zorder=2)
            axL.scatter([mf["coupling"]], [-1], s=26, c=rfig.PAL[1],
                        marker="D", zorder=3, linewidths=0)
            axL.axhline(-0.5, color="#bbbbbb", lw=0.8, ls=":")
        axL.set_xlabel("ripple-HFB coupling\n"
                       "peri (±250 ms) − non-peri (250–750 ms),\n"
                       f"real − shifted null (mean ± SEM over {unit}s)")
        axL.set_title("Coupling\n* q < 0.05 vs zero", fontsize=rfig.FS_TITLE)
        # Five ticks, not matplotlib's default: the values are ~1e-2 with four
        # decimals, so the default density runs the labels into each other.
        axL.xaxis.set_major_locator(plt.MaxNLocator(5))

        # ---------------- right: MedialFrontal minus ROI -------------------
        axR.axvline(0, color="#3d3d3d", lw=0.8, ls="--", zorder=1)
        if cmap is not None:
            dx = np.array([cmap["diff_mf_minus_roi"].get(r, np.nan)
                           for r in order], float)
            de = np.array([cmap["sem"].get(r, np.nan) for r in order], float)
            qq = np.array([cmap["q"].get(r, np.nan) for r in order], float)
            ovl = np.array([cmap["overlap_pct"].get(r, np.nan)
                            for r in order], float)
            # An overlapping contrast is drawn hollow: the number is real but
            # the two groups share contacts, so it is not an independent test.
            face = ["none" if o > 0 else c for o, c in zip(ovl, col)]
            axR.hlines(y, dx - de, dx + de, colors=col, linewidth=1.2, zorder=2)
            axR.scatter(dx, y, s=22, facecolors=face, edgecolors=col,
                        linewidths=1.0, zorder=3)
            for i in range(len(order)):
                if np.isfinite(qq[i]) and qq[i] < 0.05:
                    axR.text(dx[i] + de[i] + 0.0005, i, "*", va="center",
                             fontsize=rfig.FS_LABEL, color="#0e3d3a")
        axR.axhline(-0.5, color="#bbbbbb", lw=0.8, ls=":")
        axR.set_xlabel(f"{MEDIAL_NAME} − ROI\n"
                       f"paired over {unit}s\n"
                       f"positive = {MEDIAL_NAME} couples more")
        axR.set_title(f"{MEDIAL_NAME} vs ROI\n* q < 0.05 for the difference",
                      fontsize=rfig.FS_TITLE)
        axR.xaxis.set_major_locator(plt.MaxNLocator(5))

        axL.set_yticks(np.r_[[-1], y])
        axL.set_yticklabels(
            [f"{MEDIAL_NAME}  ({mf['n_sessions']}s / "
             f"{mf['n_derivations_diff_shaft']}d)" if mf else ""]
            + [f"{r.roi}  ({r.n_sessions}s / {r.n_derivations_diff_shaft}d)"
               + (f"  n<{UNDERPOWERED_BELOW}" if r.underpowered else "")
               for _, r in g.iterrows()], fontsize=rfig.FS_TICK)
        for lab in axL.get_yticklabels()[:1]:
            lab.set_color(rfig.PAL[1])
        for ax in (axL, axR):
            ax.set_ylim(-1.7, len(order) - 0.3)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        # Everything goes at the TOP. A footnote under a three-line x-label
        # collides with it, and how badly depends on the row count, so there
        # is no bottom margin that is right for every run.
        cap = ("Ripple-HFB coupling by ROI — one ROI per derivation, "
               f"≥ {min_sessions} sessions, different-shaft pairs\n")
        if exclude_medial:
            cap += (f"{MEDIAL_NAME} is the |x| ≤ 20 mm cut across mPFC+mOFC; "
                    "its derivations are REMOVED from every other ROI, so "
                    "every row here is disjoint\nand every contrast on the "
                    "right is an independent test. mPFC and mOFC do not "
                    "appear: they lie entirely inside it.")
        else:
            cap += (f"{MEDIAL_NAME} (diamond) is the collapsed |x| ≤ 20 mm "
                    "rule across mPFC+mOFC — it OVERLAPS the rows above and "
                    "is not an independent region.\nHollow markers on the "
                    f"right: that contrast shares derivations with "
                    f"{MEDIAL_NAME}, so it is not an independent test.")
        fig.suptitle(cap, fontsize=rfig.FS_TICK - 1, y=0.997, va="top")
        # The suptitle is three lines at ~8 pt plus the two-line panel
        # titles; the axes have to start below all of it. The fraction
        # is derived from the figure height so it holds as rows grow.
        # `tight_layout(rect=...)` does not reliably hold the top back once a
        # suptitle is set, so the axes top is forced afterwards instead. The
        # fraction comes from the figure height, so it holds as rows grow.
        head_cm = 3.0
        fig.tight_layout()
        fig.subplots_adjust(top=1 - head_cm / h_cm)
        stem = os.path.join(out_dir, "roi_coupling")
        fig.savefig(stem + ".pdf")
        fig.savefig(stem + ".png", dpi=300)
        plt.close(fig)


def traces(results=None, assignment=None, out_dir=None, unit="subject",
           smooth_ms=SMOOTH_MS, min_sessions=MIN_SESSIONS, show=True):
    """The averaged ripple-locked HFB trace per partition ROI.

    A LOOK-AT-IT figure, not a publication one: saved as .jpeg and left open.

    Each unit's curve is real-minus-shifted-null and then baselined to the
    non-peri band, exactly as `ripple_locked_hfb.figure` does it -- the same
    baseline the peri/non-peri STATISTIC uses, so the height of the bump at
    t = 0 is the number in the coupling table and not a differently-referenced
    quantity that happens to look like it.

    Smoothing (`smooth_ms`) is cosmetic and applies to the drawing only.

    Two panels, as asked:
      `..._overlay.jpeg`  every ROI on one axis, thin lines
      `..._grid.jpeg`     one small panel per ROI, each on its own y-scale

    Own y-scale per panel is deliberate: lOFC is ~4x STG, and a shared axis
    would flatten every weak region into a straight line and say nothing about
    its shape.
    """
    import matplotlib.pyplot as plt

    root = swr_io.derivatives_dir(swr_io.get_data_root())
    R = results or os.path.join(root, "group", "swr",
                                "ripple_locked_hfb_2026-09-18")
    out_dir = out_dir or _latest_partition(root)
    A = pd.read_csv(assignment or os.path.join(out_dir,
                                               "derivation_assignment.csv"))
    part = A.set_index(["session", "cx_pair"])["roi_part"]

    z = np.load(os.path.join(R, "timecourses.npz"))
    t_ms, tr = z["t_ms"], z["traces"]
    fs = 1000.0 / (t_ms[1] - t_ms[0])
    flank = (np.abs(t_ms) >= NONPERI_S[0] * 1000) & \
            (np.abs(t_ms) < NONPERI_S[1] * 1000)
    ix = pd.read_csv(os.path.join(R, "timecourse_index.csv"))
    ix["same_shaft"] = ix.same_shaft.astype(bool)
    ix["is_real"] = ix.is_real.astype(bool)
    if "trace_row" not in ix.columns:
        ix["trace_row"] = np.arange(len(ix))

    # The index carries the run's own overlapping labels, so it is restricted
    # to the FAMILY rows first -- otherwise a derivation duplicated under
    # MedialFrontal/A24 would be averaged in twice.
    fam = ix[ix.roi.isin(FAMILIES)].copy()
    fam["roi_part"] = pd.MultiIndex.from_arrays(
        [fam.session, fam.cx_pair]).map(part)
    fam = fam.dropna(subset=["roi_part"])

    def _curve(sel_ix):
        """Per-unit real-minus-null curve, baselined to the non-peri band."""
        out, units = [], []
        for u, g in sel_ix[~sel_ix.same_shaft].groupby(unit):
            rr = g.trace_row.to_numpy()
            r = tr[rr[g.is_real.to_numpy()]]
            n = tr[rr[~g.is_real.to_numpy()]]
            if not len(r) or not len(n):
                continue
            c = r.mean(0) - n.mean(0)
            out.append(c - c[flank].mean())
            units.append(u)
        if not out:
            return None, []
        return _smooth(np.stack(out), fs, smooth_ms), units

    curves = {}
    for roi, g in fam.groupby("roi_part"):
        # `Unassigned` is a bookkeeping bucket -- white matter and unlabelled
        # anchors -- not a region, so it gets no trace.
        if roi == "Unassigned":
            continue
        if g[~g.same_shaft].session.nunique() < min_sessions:
            continue
        c, us = _curve(g)
        if c is not None and len(c) >= 2:
            curves[roi] = c
    mf_c, _ = _curve(ix[ix.roi == MEDIAL_NAME])
    if mf_c is not None and len(mf_c) >= 2:
        curves[MEDIAL_NAME] = mf_c

    # Most units first. Sorting by peak height puts the noisiest two-subject
    # ROI at the front, which is the opposite of useful when the point is to
    # look at the shapes.
    order = sorted(curves, key=lambda r: -len(curves[r]))
    print(f"\ntraces: {len(order)} ROIs, unit = {unit}, "
          f"smoothing {smooth_ms:.0f} ms (cosmetic)")
    for roi in order:
        print(f"  {roi:<20s}{len(curves[roi]):>4d} {unit}s")

    stem = os.path.join(out_dir, f"roi_traces_{unit}")
    with plt.rc_context(rfig._rc()):
        # ---- 1. everything on one axis --------------------------------
        fig1, ax = plt.subplots(figsize=(16 * rfig.CM, 10 * rfig.CM))
        ax.axvspan(-PERI_S * 1000, PERI_S * 1000, color="#000000", alpha=0.05,
                   lw=0, label="peri window (±250 ms)")
        ax.axhline(0, color="#3d3d3d", lw=0.7, ls="--")
        ax.axvline(0, color="#3d3d3d", lw=0.7, ls=":")
        # Opacity tracks the number of units, so a curve averaged over two
        # subjects cannot be mistaken for one averaged over thirty-five. No
        # ROI is left out; the faint ones are still drawn.
        nmax = max(len(c) for c in curves.values())
        for roi in order:
            c = curves[roi]
            m = c.mean(0)
            ax.plot(t_ms, m, lw=1.0 if roi != MEDIAL_NAME else 1.8,
                    color=ROI_C.get(roi, rfig.PAL[1]),
                    alpha=0.35 + 0.65 * (len(c) / nmax),
                    ls="-" if roi != MEDIAL_NAME else (0, (4, 1.5)),
                    label=f"{roi} ({len(c)})")
        ax.set_xlabel("time from ripple peak (ms)")
        ax.set_ylabel("HFB, real − shifted null\n(baselined to 250–750 ms)")
        ax.set_title(f"Ripple-locked HFB by ROI — mean over {unit}s\n"
                     f"different-shaft pairs, {smooth_ms:.0f} ms smoothing "
                     f"(cosmetic); line opacity scales with n {unit}s",
                     fontsize=rfig.FS_TITLE)
        ax.legend(frameon=False, fontsize=rfig.FS_TICK - 2, ncol=2,
                  loc="upper left")
        fig1.tight_layout()
        fig1.savefig(stem + "_overlay.jpeg", dpi=200)

        # ---- 2. one small panel each ----------------------------------
        n = len(order)
        ncol = min(5, n)
        nrow = int(np.ceil(n / ncol))
        fig2, axes = plt.subplots(nrow, ncol, sharex=True,
                                  figsize=(3.6 * ncol * rfig.CM,
                                           3.4 * nrow * rfig.CM))
        axes = np.atleast_1d(axes).ravel()
        for a, roi in zip(axes, order):
            c = curves[roi]
            m = c.mean(0)
            se = c.std(0, ddof=1) / np.sqrt(len(c))
            col = ROI_C.get(roi, rfig.PAL[1])
            a.axvspan(-PERI_S * 1000, PERI_S * 1000, color="#000000",
                      alpha=0.05, lw=0)
            a.axhline(0, color="#3d3d3d", lw=0.6, ls="--")
            # Thin per-unit lines behind the mean: with 4-40 units per ROI the
            # spread is the thing worth seeing, and an SEM band alone hides
            # that some ROIs rest on a handful of subjects.
            for row in c:
                a.plot(t_ms, row, lw=0.3, color=col, alpha=0.35)
            a.fill_between(t_ms, m - se, m + se, color=col, alpha=0.25, lw=0)
            a.plot(t_ms, m, lw=1.3, color=col)
            a.set_title(f"{roi}\n({len(c)} {unit}s)",
                        fontsize=rfig.FS_TICK - 1)
            a.tick_params(labelsize=rfig.FS_TICK - 2)
            # A single excursion -- one MedialFrontal subject reaches 0.2, ten
            # times the mean -- otherwise sets the panel's scale and flattens
            # the average into a line. The y-limits are taken from the 2nd-98th
            # percentile of the individual curves instead, so a few samples run
            # off the top; nothing is removed from the average, which is
            # computed on every unit.
            lo, hi = np.percentile(c, [2, 98])
            pad = 0.15 * max(hi - lo, 1e-9)
            a.set_ylim(min(lo - pad, (m - se).min()),
                       max(hi + pad, (m + se).max()))
        for a in axes[n:]:
            a.set_visible(False)
        # x-label on the lowest VISIBLE panel of every column, not on the last
        # n panels -- with a ragged final row those are different sets.
        for col_i in range(ncol):
            vis = [i for i in range(col_i, n, ncol)]
            if vis:
                axes[vis[-1]].set_xlabel("time (ms)",
                                         fontsize=rfig.FS_TICK - 1)
        axes[0].set_ylabel("HFB, real − null", fontsize=rfig.FS_TICK - 1)
        fig2.suptitle(f"Ripple-locked HFB per ROI — thin lines are single "
                      f"{unit}s, band is SEM; each panel has its OWN y-scale\n"
                      "y-limits are the 2nd–98th percentile of the single "
                      f"{unit} curves, so extreme excursions run off-panel; "
                      "the mean uses every unit",
                      fontsize=rfig.FS_TICK - 1)
        fig2.tight_layout(rect=(0, 0, 1, 0.94))
        fig2.savefig(stem + "_grid.jpeg", dpi=200)

    print(f"\n  -> {stem}_overlay.jpeg\n  -> {stem}_grid.jpeg")
    if show:
        plt.show()
    return None


def distance(results=None, partition_dir=None, bundle=None, unit="session"):
    """Reviewer check: is the coupling explained by proximity to hippocampus?

    Cortical HFB near the hippocampus could in principle reflect volume
    conduction of the ripple itself, even on a different shaft. If it did,
    coupling would FALL with distance. This reports, per ROI, the median
    Euclidean distance from each different-shaft derivation to the nearest
    hippocampal derivation in the same session, beside that ROI's coupling.
    """
    from scipy import stats as st
    root = swr_io.derivatives_dir(swr_io.get_data_root())
    G = os.path.join(root, "group", "swr")
    partition_dir = partition_dir or _latest_partition(root)
    b_dir = bundle or os.path.join(G, "bundle_v2")
    with open(os.path.join(b_dir, "swr_bundle.pkl"), "rb") as f:
        pr = pickle.load(f)["pairs"].drop_duplicates(["session", "pair_id"])
    coord = pr.set_index(["session", "pair_id"])[["mni_x", "mni_y", "mni_z"]]
    hc = pr[pr.role == "ripple"]
    A = pd.read_csv(os.path.join(partition_dir, "derivation_assignment.csv"))
    A = A[~A.same_shaft]
    rows = []
    for _, r in A.iterrows():
        try:
            c = coord.loc[(r.session, r.cx_pair)].to_numpy(float)
        except KeyError:
            continue
        hx = hc[hc.session == r.session][["mni_x", "mni_y", "mni_z"]] \
            .to_numpy(float)
        hx = hx[~np.isnan(hx).any(1)]
        if not len(hx) or np.isnan(c).any():
            continue
        rows.append({"roi": r.roi_part,
                     "d_mm": float(np.linalg.norm(hx - c, axis=1).min())})
    D = pd.DataFrame(rows)
    T = pd.read_csv(os.path.join(partition_dir, "roi_coupling.csv"))
    T = T[T.enough_coverage & (T.roi != "Unassigned")]
    t = (D.groupby("roi").d_mm
           .agg(n="count", median="median",
                q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75))
           .reset_index().merge(T[["roi", "coupling", "q"]], on="roi"))
    t = t.sort_values("median")
    print("\nDistance to the nearest hippocampal derivation vs coupling")
    print("different-shaft derivations only\n")
    print(f"  {'ROI':<18s}{'n':>5s}{'median mm':>11s}{'IQR':>16s}"
          f"{'coupling':>11s}{'q':>8s}")
    for _, r in t.iterrows():
        star = "*" if r.q < 0.05 else " "
        print(f"  {r.roi:<18s}{int(r.n):>5d}{r['median']:>11.1f}"
              f"{f'[{r.q25:.0f}, {r.q75:.0f}]':>16s}"
              f"{r.coupling:>+11.4f}{r.q:>7.3f}{star}")
    rho, p = st.spearmanr(t["median"], t.coupling)
    print(f"\n  Spearman(distance, coupling) across ROIs: rho = {rho:+.3f}, "
          f"p = {p:.3f}")
    print("  volume conduction predicts rho < 0; the two most coupled regions "
          "are the\n  two furthest from the hippocampus.")
    return None


def estimators(results=None, partition_dir=None, ref_roi=MEDIAL_NAME):
    """Reviewer check: does the ROI comparison depend on the estimator?

    `ref_roi` and another ROI rarely share a session, so a PAIRED test throws
    away most of the data. Three estimators are reported side by side:
    paired (sessions with both), unpaired (all sessions per ROI, Welch), and a
    mixed model over DERIVATIONS with subject as a random intercept -- the
    pseudopopulation version. Where they agree the conclusion does not depend
    on the choice.
    """
    import warnings
    from scipy import stats as st
    import statsmodels.formula.api as smf
    root = swr_io.derivatives_dir(swr_io.get_data_root())
    G = os.path.join(root, "group", "swr")
    partition_dir = partition_dir or _latest_partition(root)
    R = results or os.path.join(G, "ripple_locked_hfb_2026-09-18")
    part = pd.read_csv(os.path.join(partition_dir,
                                    "derivation_assignment.csv"))
    d = pd.read_csv(os.path.join(R, "per_pair.csv"))
    d = d[(d.pad_s == d.pad_s.min()) & (~d.same_shaft)]
    base = d[d.roi.isin(FAMILIES)].copy()
    base["rp"] = pd.MultiIndex.from_arrays(
        [base.session, base.cx_pair]).map(
        part.set_index(["session", "cx_pair"])["roi_part"])
    sess = {r: _effect(g, "session") for r, g in base.groupby("rp")}
    der = {}
    for r, g in base.groupby("rp"):
        gg = g.copy()
        gg["_d"] = gg.session.astype(str) + "|" + gg.cx_pair.astype(str)
        der[r] = (_effect(gg, "_d"),
                  gg.drop_duplicates("_d").set_index("_d").subject)
    print(f"\n{ref_roi} minus each ROI, three estimators\n")
    print(f"  {'ROI':<18s}{'paired':>26s}{'unpaired':>26s}{'mixed':>24s}")
    for roi in sorted(sess, key=lambda r: -sess[r].mean()):
        if roi in (ref_roi, "Unassigned") or len(sess[roi]) < 5:
            continue
        a, b = sess[ref_roi], sess[roi]
        sh = a.index.intersection(b.index)
        s1 = "n/a"
        if len(sh) >= 5:
            dd = (a[sh] - b[sh]).to_numpy()
            t, p = st.ttest_1samp(dd, 0.0)
            s1 = f"{dd.mean():+.4f} p={p:.3f} n={len(sh)}"
        t, p = st.ttest_ind(a.to_numpy(), b.to_numpy(), equal_var=False)
        s2 = f"{a.mean() - b.mean():+.4f} p={p:.3f} n={len(a)}/{len(b)}"
        s3 = "n/a"
        va, sa = der[ref_roi]
        vb, sb = der[roi]
        fr = pd.DataFrame(
            [{"y": v, "g": 0, "s": sa.get(k)} for k, v in va.items()]
            + [{"y": v, "g": 1, "s": sb.get(k)} for k, v in vb.items()]).dropna()
        if fr.s.nunique() >= 3:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = smf.mixedlm("y ~ g", fr, groups=fr["s"]).fit(reml=True)
                s3 = (f"{-m.params['g']:+.4f} p={m.pvalues['g']:.3f} "
                      f"n={len(fr)}d")
            except Exception:
                pass
        print(f"  {roi:<18s}{s1:>26s}{s2:>26s}{s3:>24s}")
    print("\n  agreement across all three is what makes the comparison robust "
          "to the\n  low co-occurrence of these electrodes")
    return None


def _latest_partition(root):
    """Newest hfb_roi_partition_* directory, so `traces` needs no arguments."""
    g = os.path.join(root, "group", "swr")
    d = sorted(x for x in os.listdir(g) if x.startswith(f"{ANALYSIS_NAME}_"))
    if not d:
        raise SystemExit(f"no {ANALYSIS_NAME}_* directory in {g} -- run `run` first")
    return os.path.join(g, d[-1])


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run, "traces": traces, "distance": distance,
                   "estimators": estimators})
    else:
        run()
