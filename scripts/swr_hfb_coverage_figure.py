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
        frontal_scale=0.22, font_pt=None, legend=True, view_labels=True):
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


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
