#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overview of high-contributing datapoints: where (locations) subjects are when
conditions are active, and which condition-pairs are compared.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect conditions for high-contributing datapoints."
    )
    parser.add_argument(
        "--summary-csv",
        default="data/derivatives/group/RDM_plots/summary_group_contrib_DSR_ortho_location.csv",
        help="Group summary CSV from inspect_group_contrib_ortho_DSR.py",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=200,
        help="Top-N datapoints by mean_contrib (positive only).",
    )
    parser.add_argument(
        "--subjects",
        default="",
        help="Comma-separated subject IDs (e.g., sub-01,sub-02).",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory (defaults next to summary csv).",
    )
    return parser.parse_args()


def extract_evs_from_label(label):
    """Return list of EV strings from a label like 'A1... | A2... vs B1... | B2...'"""
    if not label:
        return []
    parts = label.split(" vs ")
    evs = []
    for part in parts:
        for ev in part.split(" | "):
            evs.append(ev.strip())
    return [e for e in evs if e]


def main():
    args = parse_args()

    source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
    if not os.path.isdir(source_dir):
        source_dir = "/home/fs0/xpsy1114/scratch"

    summary_path = args.summary_csv
    if not os.path.isabs(summary_path):
        summary_path = os.path.join(source_dir, summary_path)

    out_dir = args.out_dir
    if not out_dir:
        out_dir = os.path.dirname(summary_path)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(summary_path)
    df = df[df["mean_contrib"] > 0].sort_values("mean_contrib", ascending=False)
    top_df = df.head(args.top_n)

    labels = top_df["label"].fillna("").tolist()
    top_evs = sorted(set(sum((extract_evs_from_label(l) for l in labels), [])))

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        # default: infer from data directory
        data_dir = os.path.join(source_dir, "data/derivatives")
        subjects = sorted([d for d in os.listdir(data_dir) if d.startswith("sub-")])
    exclude_subs = {"sub-10", "sub-21", "sub-39"}
    subjects = [s for s in subjects if s not in exclude_subs]

    locs = list(range(1, 10))
    subj_loc_counts = []
    subj_names = []

    for sub in subjects:
        beh_dir = os.path.join(source_dir, "data/derivatives", sub, "beh")
        beh_path = os.path.join(beh_dir, f"{sub}_beh_fmri_clean.csv")
        if not os.path.exists(beh_path):
            continue
        beh_df = pd.read_csv(beh_path)
        mask = beh_df["unique_time_bin_type"].isin(top_evs)
        curr_locs = beh_df.loc[mask, "curr_loc"]
        counts = curr_locs.value_counts().to_dict()
        subj_loc_counts.append([counts.get(loc, 0) for loc in locs])
        subj_names.append(sub)

    if subj_loc_counts:
        heat = np.asarray(subj_loc_counts)
        fig, ax = plt.subplots(figsize=(6, 0.4 * len(subj_names) + 2))
        im = ax.imshow(heat, aspect="auto", interpolation="None")
        ax.set_title(f"Locations for top {len(top_df)} high-contrib datapoints", fontsize=12)
        ax.set_xlabel("Location (curr_loc)", fontsize=10)
        ax.set_ylabel("Subject", fontsize=10)
        ax.set_xticks(range(len(locs)))
        ax.set_xticklabels(locs)
        ax.set_yticks(range(len(subj_names)))
        ax.set_yticklabels(subj_names, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "heat_locations_top_contrib.png"), dpi=200)
        plt.close(fig)

    # Plot condition-pair labels (top 30)
    top_label_counts = top_df["label"].value_counts().head(30)
    if len(top_label_counts) > 0:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.barh(top_label_counts.index[::-1], top_label_counts.values[::-1])
        ax2.set_title("Top condition-pair labels", fontsize=12)
        ax2.set_xlabel("Count", fontsize=10)
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "bar_top_condition_pairs.png"), dpi=200)
        plt.close(fig2)

    # Save EV list
    with open(os.path.join(out_dir, "top_contrib_EVs.txt"), "w") as f:
        for ev in top_evs:
            f.write(ev + "\n")


if __name__ == "__main__":
    main()
