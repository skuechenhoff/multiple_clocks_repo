#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The one source of cell -> ROI labels.

The final assignment is `derivatives/ROI_assignment/cells_step5_final_labels.csv`
(column `atlas_roi`), written by `scripts/cell_to_roi_july26.py` on 2026-08-28.
`derivatives/neurons_with_ROI_labels.csv` is the same run's export and carries
identical labels; the loaders that read it (`ripple_rsa.cell_roi_table`,
`cell_selection.load_roi_table`) check that with `assert_final`.

Saved per-cell RESULT tables carry the labels of the day they were written
(e.g. the per-lag run `2026-08-04_..._relabelled-final` is stale on 33 cells).
Never trust a saved `roi` column: read such tables with `read_per_cell_table`,
which re-applies the final labels.

@author: Svenja Kuchenhoff
"""
import pandas as pd

FINAL_ROI_TABLE = ("/Users/xpsy1114/Documents/projects/multiple_clocks/data/"
                   "ephys_humans/derivatives/ROI_assignment/"
                   "cells_step5_final_labels.csv")
ROI_COLUMN = "atlas_roi"


def final_labels():
    """One row per cell: subject (int), cell_idx (1-based int), roi."""
    t = pd.read_csv(FINAL_ROI_TABLE)
    return pd.DataFrame({"subject": t["subject"].astype(int),
                         "cell_idx": t["cell idx"].astype(int),
                         "roi": t[ROI_COLUMN]})


def apply_final_labels(df, subject_col="subject_id", cell_col="cell_idx",
                       roi_col="roi"):
    """Overwrite `roi_col` with each cell's final label (row order kept).

    The label the table was saved with is kept as `<roi_col>_saved`. A cell
    missing from the final table gets NaN, so it drops out of any ROI filter.
    """
    key = pd.DataFrame({
        "subject": pd.to_numeric(df[subject_col]).astype(int).to_numpy(),
        "cell_idx": pd.to_numeric(df[cell_col]).astype(int).to_numpy()})
    final = key.merge(final_labels(), on=["subject", "cell_idx"], how="left")
    assert len(final) == len(df)
    out = df.copy()
    if roi_col in out:
        out[f"{roi_col}_saved"] = out[roi_col]
    out[roi_col] = final["roi"].to_numpy()
    return out


def read_per_cell_table(path, subject_col="subject_id", cell_col="cell_idx"):
    """Read a saved per-cell result table, with the final ROI labels.

    `cell_idx` is parsed from `neuron` ('SS_CC-...') when the table lacks it.
    """
    df = pd.read_csv(path)
    if cell_col not in df:
        df[cell_col] = (df["neuron"].astype(str).str.split("_").str[1]
                        .str.split("-").str[0].astype(int))
    df = df.dropna(subset=[subject_col, cell_col])
    return apply_final_labels(df, subject_col, cell_col)


def assert_final(table, subject_col="subject", cell_col="cell idx",
                 roi_col="atlas_roi"):
    """Raise if a ROI table disagrees with the final assignment anywhere."""
    t = apply_final_labels(table, subject_col, cell_col, roi_col="_check")
    t["_given"] = table[roi_col].to_numpy()
    bad = t[t["_check"].astype(str) != t["_given"].astype(str)]
    if len(bad):
        raise ValueError(
            f"{len(bad)} cells differ from the final ROI assignment "
            f"({FINAL_ROI_TABLE}); first: "
            f"{bad[[subject_col, cell_col, '_given', '_check']].head().to_dict('records')}")
