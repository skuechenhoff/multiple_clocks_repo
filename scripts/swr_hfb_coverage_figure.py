#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coverage figure for the ripple-locked HFB analysis: hippocampal derivations
that supplied the ripples, and the medial frontal derivations whose HFB was
measured, in one 3-D rendering.

Positions are **bipolar pair midpoints**, not contact centres. A bipolar
derivation is a spatial derivative and is sensitive between its two contacts,
so the midpoint is where it actually samples; plotting one of the two contacts
would place it up to half an inter-contact distance away. The hippocampal
coverage figure in `swr_contact_figure.py` plots contact SITES instead, because
its question is which tissue was implanted rather than which derivation was
measured -- so the two figures legitimately show different counts.

Only derivations that actually entered the analysis are drawn: different-shaft
(Methods §5.1) and enough clean peri-ripple epochs to yield an estimate.

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_hfb_coverage_figure.py run --results=<ripple_locked_hfb dir>

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
        frontal_scale=0.22):
    R = results or os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                                "group", "swr", "ripple_locked_hfb_2026-09-16")
    b_dir = bundle or os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                                   "group", "swr", "bundle_v2")
    with open(os.path.join(b_dir, "swr_bundle.pkl"), "rb") as f:
        B = pickle.load(f)
    pairs = B["pairs"].drop_duplicates(subset=["session", "pair_id"])
    coord = pairs.set_index(["session", "pair_id"])[XYZ + ["hemisphere"]]

    d = pd.read_csv(os.path.join(R, "per_pair.csv"))
    d = d[~d.same_shaft]
    if "pad_s" in d.columns:                      # the native pad only
        d = d[d.pad_s == d.pad_s.min()]

    def _coords(keys):
        ix = pd.MultiIndex.from_tuples(sorted(set(keys)))
        g = coord.reindex(ix).dropna(subset=XYZ)
        return g[XYZ].to_numpy(float), g["hemisphere"].to_numpy()

    hc_xyz, hc_hemi = _coords(zip(d.session, d.hc_pair))
    groups, counts = [], {}
    for roi in ("mPFC", "mOFC"):
        g = d[d.roi == roi]
        if not len(g):
            continue
        xyz, hemi = _coords(zip(g.session, g.cx_pair))
        counts[roi] = len(xyz)
        groups.append({"coords": xyz, "hemispheres": hemi,
                       "color": GROUP_C[roi], "scale": frontal_scale,
                       "label": f"{roi} ({len(xyz)})"})

    print(f"\nderivations drawn (different-shaft, entered the analysis):")
    print(f"  hippocampal      {len(hc_xyz):3d}  "
          f"(L {int((hc_hemi == 'L').sum())}, R {int((hc_hemi == 'R').sum())})")
    for roi, n in counts.items():
        print(f"  {roi:<16s} {n:3d}")
    print(f"  sessions {d.session.nunique()}, subjects {d.subject.nunique()}")

    out_stem = out_stem or os.path.join(R, "hfb_coverage_3d")
    rfig.contact_coverage_3d_figure(
        hc_xyz, excluded=None, out_stem=out_stem, hemispheres=hc_hemi,
        views=list(views), width_cm=width_cm, contact_scale=contact_scale,
        contact_label=f"hippocampal ({len(hc_xyz)})",
        extra_groups=groups,
        view_labels={"left": "left hemisphere", "right": "right hemisphere",
                     "dorsal": "dorsal", "ventral": "ventral",
                     "anterior": "anterior"})
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
