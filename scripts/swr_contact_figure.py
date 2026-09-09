#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manuscript figure: hippocampal contacts inside a translucent brain.

The companion to `swr_build_contacts.py`, which decides *which* contacts these
are. This script only draws them, on an fsaverage pial surface with the two
hippocampi rendered as solid bodies, so a reader can see that every analysed
derivation sits inside the structure the paper claims it does.

Two versions are written, because they answer different questions:

    hpc_contacts_3d            the 144 derivations that entered the analysis
    hpc_contacts_3d_excluded   the same, plus every hippocampal contact that
                               did not (artifact-contaminated, or in a session
                               whose behaviour could not be placed in the
                               recording)

Coordinates are deduplicated by subject x contact: several subjects contributed
two or three sessions from the same implant, and plotting those spheres on top
of each other would suggest coverage that is not there. Both counts are
printed, and land in the settings file next to the figure.

Outputs (under derivatives/group/swr/figures/):
    hpc_contacts_3d.pdf / .png            the panel
    hpc_contacts_3d_<view>.png            each view at full render resolution
    hpc_contacts_3d_counts.csv            what is drawn, per category
    settings_swr_contact_figure.json

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_contact_figure.py
    python scripts/swr_contact_figure.py --group_dir=<dir with the cluster run>
    python scripts/swr_contact_figure.py --views="['left','dorsal','ventral']"

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io

try:
    import fire
except ImportError:
    fire = None

XYZ = ["mni_x", "mni_y", "mni_z"]


def _load(group_dir):
    """The tables the figure needs.

    Two shapes are accepted, because a bundle downloaded from the cluster does
    not carry `macro_contacts_all.csv`:

      full     macro_contacts_all.csv + bundle/  -- coordinates are the ANCHOR
               contact, i.e. the hippocampal contact itself
      bundle   bundle/ only -- coordinates are the derivation MIDPOINT, which is
               where a bipolar derivation is actually sensitive

    The two differ by about half the inter-contact distance (~2.6 mm) and the
    figure says which it used. `group_dir` may be the group folder or the bundle
    folder itself.
    """
    if os.path.isfile(os.path.join(group_dir, "pairs.csv")):
        bdir = group_dir
    else:
        bdir = os.path.join(group_dir, "bundle")
    need = {"pairs": os.path.join(bdir, "pairs.csv"),
            "qc": os.path.join(bdir, "channel_qc.csv")}
    missing = [f"{k}: {v}" for k, v in need.items() if not os.path.isfile(v)]
    if missing:
        raise FileNotFoundError(
            "cannot draw the figure without the analysed set --\n  "
            + "\n  ".join(missing)
            + "\nPoint --group_dir at a run's group folder or its bundle/.")
    out = {k: pd.read_csv(v) for k, v in need.items()}

    mc = os.path.join(group_dir, "macro_contacts_all.csv")
    out["contacts"] = pd.read_csv(mc) if os.path.isfile(mc) else None
    return out


def build_contact_sets(group_dir):
    """Split the hippocampal contacts into analysed and not, with their coords.

    A contact counts as analysed if it anchored a bipolar derivation that
    survived artifact rejection in *any* session -- a site that was usable once
    and contaminated in a repeat session is coverage that existed, and the
    figure says so.
    """
    t = _load(group_dir)
    pairs = t["pairs"].merge(
        t["qc"][["session", "pair_id", "excluded"]],
        on=["session", "pair_id"], how="left")
    pairs["excluded"] = pairs["excluded"].fillna(True).astype(bool)

    if t["contacts"] is not None:
        coord_source = "anchor contact (macro_contacts_all.csv)"
        hpc = t["contacts"][t["contacts"]["is_hpc"].fillna(False).astype(bool)].copy()
        hpc = hpc.dropna(subset=XYZ)
        kept = set(zip(pairs.loc[~pairs.excluded, "session"],
                       pairs.loc[~pairs.excluded, "anat_label_a"]))
        hpc["analysed"] = [(s, l) in kept
                           for s, l in zip(hpc.session, hpc.anat_label)]
        dedup_on = ["subject_label", "anat_label"]
    else:
        coord_source = "derivation midpoint (bundle pairs.csv)"
        hpc = pairs.dropna(subset=XYZ).copy()
        hpc["analysed"] = ~hpc["excluded"]
        hpc["anat_label"] = hpc["anat_label_a"]
        dedup_on = ["subject_label", "pair_id"]

    # A site is analysed if any of its sessions was; take that session's row so
    # the coordinate belongs to the derivation actually used.
    hpc = hpc.sort_values("analysed", ascending=False)
    sites = hpc.drop_duplicates(dedup_on, keep="first")

    analysed_sessions = set(pairs.session)
    return {
        "included": sites[sites.analysed],
        "excluded": sites[~sites.analysed],
        "n_derivations": int((~pairs.excluded).sum()),
        "n_sessions": int(pairs.loc[~pairs.excluded, "session"].nunique()),
        "n_subjects": int(pairs.loc[~pairs.excluded, "subject_label"].nunique()),
        "n_derivations_dropped": int(pairs.excluded.sum()),
        "coord_source": coord_source,
        "n_hpc_contacts": int(len(hpc)),
        "n_hpc_unanalysed_sessions": int((~hpc.session.isin(analysed_sessions)).sum()),
    }


def make_figure(group_dir=None, out_dir=None,
                views=("left", "right", "dorsal"),
                width_cm=12.0, contact_scale=0.20, hpc_color=None,
                hpc_source="atlas", verbose=True):
    import matplotlib
    matplotlib.use("Agg")
    import mc.plotting.ripple_figures as rfig

    data_root = swr_io.get_data_root()
    group_dir = group_dir or os.path.join(
        swr_io.derivatives_dir(data_root), "group", "swr")
    out_dir = out_dir or os.path.join(
        swr_io.derivatives_dir(data_root), "group", "swr", "figures")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(os.path.join(out_dir, ".."), "swr_contact_figure")

    s = build_contact_sets(group_dir)
    inc, exc = s["included"], s["excluded"]
    views = list(views)
    kw = dict(views=views, width_cm=width_cm, contact_scale=contact_scale,
              hpc_source=hpc_source,
              view_labels={"left": "left hemisphere", "right": "right hemisphere",
                           "dorsal": "dorsal", "ventral": "ventral",
                           "oblique": "oblique", "anterior": "anterior"})
    if hpc_color:
        kw["hpc_color"] = hpc_color

    print(f"coordinates       : {s['coord_source']}")
    print(f"contacts analysed : {len(inc):3d} sites  "
          f"({s['n_derivations']} derivations, {s['n_sessions']} sessions, "
          f"{s['n_subjects']} subjects)")
    print(f"contacts not used : {len(exc):3d} sites  "
          f"(of {s['n_hpc_contacts']} hippocampal contacts overall: "
          f"{s['n_derivations_dropped']} derivations dropped for artifact, "
          f"{s['n_hpc_unanalysed_sessions']} in sessions that were not analysed; "
          f"a site analysed in any session counts as analysed)")

    stem_a = os.path.join(out_dir, "hpc_contacts_3d")
    rfig.contact_coverage_3d_figure(
        inc[XYZ].to_numpy(float), excluded=None, out_stem=stem_a,
        hemispheres=inc.hemisphere.to_numpy(),
        counts=f"{len(inc)} sites", **kw)
    print(f"  -> {stem_a}.pdf / .png")

    stem_b = os.path.join(out_dir, "hpc_contacts_3d_excluded")
    rfig.contact_coverage_3d_figure(
        inc[XYZ].to_numpy(float), excluded=exc[XYZ].to_numpy(float),
        out_stem=stem_b, hemispheres=inc.hemisphere.to_numpy(),
        excluded_hemispheres=exc.hemisphere.to_numpy(),
        counts=f"{len(inc)} sites", **kw)
    print(f"  -> {stem_b}.pdf / .png")

    counts = pd.DataFrame([
        {"set": "analysed", "n_sites": len(inc),
         "n_subjects": s["n_subjects"], "n_sessions": s["n_sessions"],
         "n_left": int((inc.hemisphere == "L").sum()),
         "n_right": int((inc.hemisphere == "R").sum()),
         "note": f"{s['n_derivations']} bipolar derivations, "
                 f"{len(inc)} distinct sites"},
        {"set": "not_analysed", "n_sites": len(exc),
         "n_subjects": int(exc.subject_label.nunique()),
         "n_sessions": int(exc.session.nunique()),
         "n_left": int((exc.hemisphere == "L").sum()),
         "n_right": int((exc.hemisphere == "R").sum()),
         "note": f"{s['n_hpc_contacts']} hippocampal contacts in total; "
                 f"{s['n_derivations_dropped']} derivations >2/3 artifact, "
                 f"{s['n_hpc_unanalysed_sessions']} in unanalysed sessions"}])
    counts.to_csv(os.path.join(out_dir, "hpc_contacts_3d_counts.csv"), index=False)
    if verbose:
        print("\n" + counts.to_string(index=False))

    with open(os.path.join(out_dir, "settings_swr_contact_figure.json"), "w") as f:
        json.dump({"analysis_name": "swr_contact_figure",
                   "group_dir": group_dir, "views": views,
                   "width_cm": width_cm, "contact_scale": contact_scale,
                   "surface": "fsaverage pial, both hemispheres",
                   "hippocampus": ("Harvard-Oxford subcortical probability map at the "
                                   "same threshold that selected the contacts"),
                   "hpc_source": hpc_source,
                   "projection": "parallel (orthographic)",
                   "lateral_views": "show only that hemisphere's contacts",
                   "coords": "MNI152 -> MNI305 (fsaverage surface RAS)",
                   "coord_source": s["coord_source"],
                   "dedup": "one sphere per subject x contact",
                   "created": datetime.now().isoformat(timespec="seconds")},
                  f, indent=2)
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(make_figure)
    else:
        make_figure()
