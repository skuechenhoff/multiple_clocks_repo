"""Shared cell-selection utilities used by the RSA, encoding and
spatial-peaks pipelines.

Single source of truth for:
    * which subjects to include (RSA grouping JSON)
    * the MNI/ROI lookup (neurons_with_final_roi_labels.csv)
    * label parsing (`01_07-07-chan120-EC` -> subject=1, cell_idx=7)

Lightweight by design: this module does NOT load neuron data. It only
returns the per-cell registry (one row per (subject, cell_idx)).
Use mc.analyse.helpers_human_cells.load_norm_data for the per-subject
data load step.
"""

import json
import os

import numpy as np
import pandas as pd

import mc.analyse.roi_labels as roi_labels


DEFAULT_DATA_DIR = (
    "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/"
    "derivatives"
)
DEFAULT_RSA_SUBJECTS_JSON = "all_sessions_dsrRSA_grouping_summary.json"
# Canonical per-cell ROI table, produced by scripts/cell_to_roi_july26.py.
# The previous name (`neurons_with_final_roi_labels.csv`, from the older
# cell_to_roi_MNI.py) now exists only inside archive folders; it is kept as
# a fallback so callers pointing at an archived copy still work.
DEFAULT_ROI_TABLE = "neurons_with_ROI_labels.csv"
LEGACY_ROI_TABLE = "neurons_with_final_roi_labels.csv"
DEFAULT_ROI_COLUMN = "alt_final_roi"


def parse_neuron_label(label):
    """Parse '01_07-07-chan120-EC' into (subject:int, cell_idx:int).

    Returns (None, None) if the label cannot be parsed.
    """
    try:
        sub_str, rest = label.split("_", 1)
        cell_idx_str = rest.split("-", 1)[0]
        return int(sub_str), int(cell_idx_str)
    except (ValueError, IndexError):
        return None, None


def load_rsa_subjects(data_dir=DEFAULT_DATA_DIR,
                      json_name=DEFAULT_RSA_SUBJECTS_JSON):
    """Return the canonical RSA subject list (zero-padded strings)."""
    path = os.path.join(data_dir, json_name)
    with open(path, "r") as f:
        summary = json.load(f)
    return list(summary.keys())


def load_roi_table(data_dir=DEFAULT_DATA_DIR,
                   table_name=DEFAULT_ROI_TABLE,
                   roi_column=DEFAULT_ROI_COLUMN):
    """Load the per-cell ROI/MNI table indexed by (subject, cell_idx).

    Falls back to the legacy filename if the canonical one is absent, and
    raises with both paths listed rather than a bare FileNotFoundError, so
    a stale default is obvious from the message.
    """
    path = os.path.join(data_dir, table_name)
    if not os.path.exists(path):
        legacy = os.path.join(data_dir, LEGACY_ROI_TABLE)
        if os.path.exists(legacy):
            print(f"[cell_selection] {table_name} not found; falling back to "
                  f"the legacy table {LEGACY_ROI_TABLE}. ROI labels may be "
                  f"out of date — re-run scripts/cell_to_roi_july26.py.")
            path = legacy
        else:
            raise FileNotFoundError(
                f"No per-cell ROI table found in {data_dir}. Looked for "
                f"{table_name!r} (canonical, written by "
                f"scripts/cell_to_roi_july26.py) and {LEGACY_ROI_TABLE!r} "
                f"(legacy, written by scripts/cell_to_roi_MNI.py)."
            )
    df = pd.read_csv(path)
    # The canonical table stores the corrected coordinates as MNI_*_final and
    # keeps the untouched originals as MNI_*. Downstream code asks for MNI_x/
    # y/z and means "the coordinate to use", so prefer the final ones — this
    # is the same precedence RSA_DSR_ROIs_simple._load_roi_table applies.
    for ax in ("x", "y", "z"):
        if f"MNI_{ax}_final" in df.columns:
            df[f"MNI_{ax}"] = df[f"MNI_{ax}_final"]
    needed = ["subject", "cell idx", roi_column,
              "MNI_x", "MNI_y", "MNI_z", "electrode label"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"ROI table {path} missing columns: {missing}")
    roi_labels.assert_final(df)
    df = df.copy()
    df["subject"] = df["subject"].astype(int)
    df["cell idx"] = df["cell idx"].astype(int)
    return df.set_index(["subject", "cell idx"])


def _lookup_roi(roi_table, subject, cell_idx, roi_column):
    try:
        roi = roi_table.loc[(subject, cell_idx), roi_column]
    except KeyError:
        return None
    if isinstance(roi, pd.Series):
        roi = roi.dropna().iloc[0] if roi.notna().any() else None
    if roi is None or pd.isna(roi):
        return None
    return str(roi)


def _lookup_mni(roi_table, subject, cell_idx):
    try:
        row = roi_table.loc[(subject, cell_idx)]
    except KeyError:
        return (np.nan, np.nan, np.nan)
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return (float(row["MNI_x"]), float(row["MNI_y"]), float(row["MNI_z"]))


def load_cells(cell_set="all_in_roi_table",
               subjects=None,
               data_dir=DEFAULT_DATA_DIR,
               roi_column=DEFAULT_ROI_COLUMN,
               rois_keep=None,
               include_unassigned=False):
    """Return per-cell registry as a DataFrame.

    Parameters
    ----------
    cell_set : {'rsa', 'all_in_roi_table'}
        'rsa'              -> subjects from the RSA grouping JSON.
        'all_in_roi_table' -> every (subject, cell_idx) row in the ROI
                              table that has a value in roi_column.
    subjects : list[str] or None
        Override the subject list. Zero-padded strings ('01', '07', ...).
    rois_keep : list[str] or None
        If set, drop rows whose ROI is not in this list. None = keep all.
    include_unassigned : bool
        If False (default), drop cells with no ROI assignment.

    Returns
    -------
    DataFrame with one row per cell. Columns:
        subject_id    (str, zero-padded)
        subject_int   (int)
        cell_idx      (int)
        roi           (str or NaN)
        MNI_x, MNI_y, MNI_z
        electrode_label
    """
    roi_table = load_roi_table(data_dir=data_dir, roi_column=roi_column)

    if cell_set == "rsa":
        if subjects is None:
            subjects = load_rsa_subjects(data_dir=data_dir)
        subj_ints = sorted({int(s) for s in subjects})
        roi_rows = roi_table.loc[roi_table.index.get_level_values(0).isin(subj_ints)]
    elif cell_set == "not_in_rsa":
        if subjects is None:
            subjects = load_rsa_subjects(data_dir=data_dir)
        subj_ints = sorted({int(s) for s in subjects})
        roi_rows = roi_table.loc[~roi_table.index.get_level_values(0).isin(subj_ints)]
    elif cell_set == "all_in_roi_table":
        roi_rows = roi_table
    else:
        raise ValueError(f"Unknown cell_set {cell_set!r}")

    rows = []
    for (sub_int, cell_idx), row in roi_rows.iterrows():
        roi = row[roi_column]
        if pd.isna(roi):
            roi = None
        if roi is None and not include_unassigned:
            continue
        if rois_keep is not None and roi not in rois_keep:
            continue
        rows.append({
            "subject_id":  f"{sub_int:02d}",
            "subject_int": int(sub_int),
            "cell_idx":    int(cell_idx),
            "roi":         roi,
            "MNI_x":       float(row["MNI_x"]) if pd.notna(row["MNI_x"]) else np.nan,
            "MNI_y":       float(row["MNI_y"]) if pd.notna(row["MNI_y"]) else np.nan,
            "MNI_z":       float(row["MNI_z"]) if pd.notna(row["MNI_z"]) else np.nan,
            "electrode_label": str(row["electrode label"]) if pd.notna(row["electrode label"]) else "",
        })
    out = pd.DataFrame(rows)
    out = out.sort_values(["subject_int", "cell_idx"]).reset_index(drop=True)
    return out


def attach_roi_to_neuron_labels(neuron_labels, roi_table=None,
                                data_dir=DEFAULT_DATA_DIR,
                                roi_column=DEFAULT_ROI_COLUMN):
    """Given an iterable of normalised_neurons labels (e.g.
    '01_07-07-chan120-EC'), return a DataFrame with columns
    [neuron_id, subject_id, cell_idx, roi, MNI_x, MNI_y, MNI_z].
    """
    if roi_table is None:
        roi_table = load_roi_table(data_dir=data_dir, roi_column=roi_column)
    rows = []
    for label in neuron_labels:
        sub, cell_idx = parse_neuron_label(label)
        if sub is None:
            rows.append({"neuron_id": label, "subject_id": None,
                         "cell_idx": None, "roi": None,
                         "MNI_x": np.nan, "MNI_y": np.nan, "MNI_z": np.nan})
            continue
        roi = _lookup_roi(roi_table, sub, cell_idx, roi_column)
        mni = _lookup_mni(roi_table, sub, cell_idx)
        rows.append({
            "neuron_id":  label,
            "subject_id": f"{sub:02d}",
            "cell_idx":   cell_idx,
            "roi":        roi,
            "MNI_x":      mni[0],
            "MNI_y":      mni[1],
            "MNI_z":      mni[2],
        })
    return pd.DataFrame(rows)
