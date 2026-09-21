#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coverage figure for the ripple-locked HFB analysis: hippocampal derivations
that supplied the ripples, and the medial frontal derivations whose HFB was
measured, in one 3-D rendering.

Positions are the **SOURCE contact** of each bipolar derivation -- the anchor,
whose atlas label defines the region. A bipolar derivation does not describe
the midpoint between its contacts: it describes the signal at the anchor,
cleaned by subtracting the neighbour. Plotting midpoints displaces every
cortical marker a median 2.2 mm laterally for no reason.

The anchor's coordinate is not carried in the exported bundle, so it is looked
up in `group/swr/macro_contacts_all.csv`. **That file must be the one built on
the cluster** -- a local build covers only the sessions whose raw data is on
this machine, and the figure will silently lose the rest. Coverage is reported;
if it is short, copy the cluster file:

    rsync <user>@ssh.swc.ucl.ac.uk:/ceph/behrens/svenja/human_ABCD_ephys/\
derivatives/group/swr/macro_contacts_all.csv  <local group/swr>/

Only derivations that entered the analysis are drawn: different-shaft
(Methods §5.1) with enough clean peri-ripple epochs to yield an estimate.

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_hfb_coverage_figure.py run --results=<ripple_locked_hfb dir>
    python scripts/swr_hfb_coverage_figure.py run --width_cm=4 --views="['left','dorsal']"

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.plotting.ripple_figures as rfig

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

XYZ = ["mni_x", "mni_y", "mni_z"]
# Project ROI colours (CLAUDE.md): mPFC = Showgirl2[1], mOFC = Showgirl2[4].
GROUP_C = {"mPFC": rfig.PAL[1], "mOFC": rfig.PAL[4]}


def run(results=None, bundle=None, out_stem=None, width_cm=12.0,
        views=("left", "right", "dorsal"), contact_scale=0.20,
        frontal_scale=0.22, font_pt=None, legend=True, view_labels=True,
        medial_max_abs_x=None):
    R = results or os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                                "group", "swr", "ripple_locked_hfb_2026-09-16")
    b_dir = bundle or os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                                   "group", "swr", "bundle_v2")
    with open(os.path.join(b_dir, "swr_bundle.pkl"), "rb") as f:
        B = pickle.load(f)
    pairs = B["pairs"].drop_duplicates(subset=["session", "pair_id"])
    # anchor label per derivation, then its own coordinate from the contact table
    anchor = pairs.set_index(["session", "pair_id"])[
        ["anat_label_a", "hemisphere"]]
    mc_p = os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                        "group", "swr", "macro_contacts_all.csv")
    mc = pd.read_csv(mc_p)
    cpos = (mc.dropna(subset=XYZ)
              .drop_duplicates(["session", "anat_label"])
              .set_index(["session", "anat_label"])[XYZ])
    print(f"contact table: {mc_p}\n  {mc.session.nunique()} sessions")

    d = pd.read_csv(os.path.join(R, "per_pair.csv"))
    d = d[~d.same_shaft]
    if medial_max_abs_x is not None:
        # Restrict the cortical sets to genuinely medial contacts. The `mOFC`
        # label is not reliably medial: 40% of the contacts carrying it sit
        # beyond |x| = 25 mm, in lateral orbitofrontal cortex, because they were
        # assigned by a neighbourhood rescue rather than by the Brainnetome
        # medial-OFC rule. Keyed on (session, pair_id) -- a pair label recurs
        # across sessions of one patient.
        mx = (pairs.drop_duplicates(["session", "pair_id"])
                   .set_index(["session", "pair_id"])["mni_x"])
        ax = np.abs(pd.to_numeric(pd.MultiIndex.from_arrays(
            [d.session, d.cx_pair]).map(mx), errors="coerce"))
        keep = pd.Series(ax, index=d.index).le(float(medial_max_abs_x))
        n0 = len(d[["session", "cx_pair"]].drop_duplicates())
        d = d[keep | ~d.roi.isin(["mPFC", "mOFC"])]
        print(f"medial restriction |x| <= {medial_max_abs_x} mm: "
              f"{len(d[['session','cx_pair']].drop_duplicates())} of {n0} "
              f"cortical derivations kept")
    if "pad_s" in d.columns:                      # the native pad only
        d = d[d.pad_s == d.pad_s.min()]

    def _coords(keys):
        """Anchor (source) coordinate for each (session, pair_id)."""
        ix = pd.MultiIndex.from_tuples(sorted(set(keys)))
        a = anchor.reindex(ix).dropna(subset=["anat_label_a"])
        want = pd.MultiIndex.from_arrays(
            [a.index.get_level_values(0), a["anat_label_a"].to_numpy()])
        xyz = cpos.reindex(want)
        ok = xyz[XYZ].notna().all(axis=1).to_numpy()
        return (xyz[XYZ].to_numpy(float)[ok], a["hemisphere"].to_numpy()[ok],
                int(len(a)), int(ok.sum()))

    hc_xyz, hc_hemi, hc_want, hc_got = _coords(zip(d.session, d.hc_pair))
    miss = [("hippocampal", hc_want, hc_got)]
    groups, counts = [], {}
    for roi in ("mPFC", "mOFC"):
        g = d[d.roi == roi]
        if not len(g):
            continue
        xyz, hemi, want, got = _coords(zip(g.session, g.cx_pair))
        miss.append((roi, want, got))
        counts[roi] = len(xyz)
        groups.append({"coords": xyz, "hemispheres": hemi,
                       "color": GROUP_C[roi], "scale": frontal_scale,
                       "label": f"{roi} ({len(xyz)})"})

    print(f"\nsource contacts drawn (different-shaft, entered the analysis):")
    print(f"  {'set':<16s}{'drawn':>7s}{'wanted':>8s}{'resolved':>10s}")
    for lab, want, got in miss:
        flag = "" if got == want else "   <-- INCOMPLETE"
        print(f"  {lab:<16s}{got:>7d}{want:>8d}{100*got/max(want,1):>9.0f}%{flag}")
    print(f"  hippocampal L {int((hc_hemi == 'L').sum())}, "
          f"R {int((hc_hemi == 'R').sum())}")
    print(f"  sessions {d.session.nunique()}, subjects {d.subject.nunique()}")
    if any(g_ < w_ for _, w_, g_ in miss):
        print("\n  ⚠ some anchors have no coordinate in macro_contacts_all.csv.")
        print("    That file is probably a LOCAL build covering only the sessions")
        print("    whose raw data is on this machine. Copy the cluster one:")
        print("    rsync <user>@ssh.swc.ucl.ac.uk:/ceph/behrens/svenja/"
              "human_ABCD_ephys/derivatives/group/swr/macro_contacts_all.csv .")

    out_stem = out_stem or os.path.join(R, "hfb_coverage_3d")
    rfig.contact_coverage_3d_figure(
        hc_xyz, excluded=None, out_stem=out_stem, hemispheres=hc_hemi,
        views=list(views), width_cm=width_cm, contact_scale=contact_scale,
        font_pt=font_pt, legend=legend,
        contact_label=f"hippocampal ({len(hc_xyz)})",
        extra_groups=groups,
        view_labels=({"left": "left hemisphere", "right": "right hemisphere",
                      "dorsal": "dorsal", "ventral": "ventral",
                      "anterior": "anterior"} if view_labels else None))
    print(f"\n  -> {out_stem}.pdf / .png")

    pd.DataFrame(
        [{"set": "hippocampal", "n_derivations": len(hc_xyz),
          "n_left": int((hc_hemi == "L").sum()),
          "n_right": int((hc_hemi == "R").sum())}]
        + [{"set": roi, "n_derivations": n} for roi, n in counts.items()]
    ).to_csv(out_stem + "_counts.csv", index=False)
    return None


def distance(results=None, bundle=None, out_csv=None):
    """Euclidean distance from each cortical derivation to the hippocampus.

    The control this supports: if the cortical HFB response were volume
    conduction of the ripple itself, it would be LARGER for contacts closer to
    the hippocampus. HFB (70-150 Hz) overlaps the ripple band (80-120 Hz), so
    that is a real alternative and proximity is the variable that would drive it.

    **Euclidean, not geodesic, and deliberately so.** Volume conduction spreads
    through the volume, not along the cortical sheet, so straight-line distance
    is the quantity that governs it. A geodesic distance is also not well
    defined here: the hippocampus is not on the cortical surface, so there is no
    surface path from it to a cortical contact. Geodesic distance would be the
    right measure for a cortico-cortical connectivity argument; it is the wrong
    one for this control.

    Distance is measured to the NEAREST hippocampal derivation in the same
    session -- the nearest one bounds the volume-conduction risk -- with the
    mean over that session's hippocampal derivations reported alongside.
    """
    import pickle
    from scipy import stats as st

    R = results or os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                                "group", "swr", "ripple_locked_hfb_2026-09-18")
    b_dir = bundle or os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                                   "group", "swr", "bundle_v2")
    with open(os.path.join(b_dir, "swr_bundle.pkl"), "rb") as f:
        pr = pickle.load(f)["pairs"].drop_duplicates(["session", "pair_id"])
    coord = pr.set_index(["session", "pair_id"])[XYZ]

    # The NATIVE pad, taken from the run's own settings -- not pad_s.max(),
    # which is the sparsest pad in the stability sweep (3.0 s) and answers a
    # different question with a third of the data.
    with open(os.path.join(R, "result.json")) as f:
        native = float(json.load(f)["results"].get("native_pad_s", 1.0))
    d = pd.read_csv(os.path.join(R, "per_pair.csv"))
    d = d[(~d.same_shaft) & (d.pad_s == native) & d.is_real]
    print(f"  artifact pad: {native:.2f} s (the run's native pad)")
    rows = []
    for (sess, cx, roi), g in d.groupby(["session", "cx_pair", "roi"]):
        try:
            c = coord.loc[(sess, cx)].to_numpy(float)
        except KeyError:
            continue
        hcs = []
        for h in g.hc_pair.unique():
            try:
                hcs.append(coord.loc[(sess, h)].to_numpy(float))
            except KeyError:
                pass
        if not hcs or np.isnan(c).any():
            continue
        hcs = np.asarray(hcs)
        hcs = hcs[~np.isnan(hcs).any(1)]
        if not len(hcs):
            continue
        dd = np.linalg.norm(hcs - c, axis=1)
        rows.append({"session": sess, "cx_pair": cx, "roi": roi,
                     "d_nearest_mm": float(dd.min()),
                     "d_mean_mm": float(dd.mean()),
                     "n_hc": int(len(dd))})
    t = pd.DataFrame(rows)

    print("\nEuclidean distance to the nearest hippocampal derivation "
          "(same session, different-shaft only)\n")
    print(f"  {'ROI':<17s}{'n':>5s}{'nearest (mm)':>26s}{'mean (mm)':>12s}")
    for roi in ["MedialFrontal", "mPFC", "mOFC", "TemporalLateral", "Auditory",
                "Visual"]:
        g = t[t.roi == roi]
        if not len(g):
            continue
        print(f"  {roi:<17s}{len(g):>5d}   median {g.d_nearest_mm.median():5.1f} "
              f"(IQR {g.d_nearest_mm.quantile(.25):.1f}-"
              f"{g.d_nearest_mm.quantile(.75):.1f})"
              f"{g.d_mean_mm.median():>12.1f}")

    # does proximity predict the effect? volume conduction says it should
    eff = (d.groupby(["session", "cx_pair", "roi"])["diff"].mean()).reset_index()
    nul = pd.read_csv(os.path.join(R, "per_pair.csv"))
    nul = nul[(~nul.same_shaft) & (nul.pad_s == native) & (~nul.is_real)]
    nul = nul.groupby(["session", "cx_pair", "roi"])["diff"].mean().reset_index()
    e = eff.merge(nul, on=["session", "cx_pair", "roi"], suffixes=("_r", "_n"))
    e["effect"] = e["diff_r"] - e["diff_n"]
    e = e.merge(t, on=["session", "cx_pair", "roi"]).dropna(
        subset=["effect", "d_nearest_mm"])
    base = e[e.roi.isin(["mPFC", "mOFC", "TemporalLateral", "Auditory", "Visual"])]
    rho, p = st.spearmanr(base.d_nearest_mm, base.effect)
    print(f"\n  effect vs distance, pooled over cortical derivations "
          f"(n = {len(base)}): rho = {rho:+.3f}, p = {p:.3g}")
    print("  " + ("FURTHER from hippocampus = LARGER effect, i.e. the opposite "
                  "of volume conduction" if rho > 0 else
                  "closer = larger, consistent with volume conduction -- "
                  "INVESTIGATE"))

    out_csv = out_csv or os.path.join(R, "hc_distance_by_derivation.csv")
    e.to_csv(out_csv, index=False)
    print(f"\n  -> {out_csv}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run, "distance": distance})
    else:
        run()
