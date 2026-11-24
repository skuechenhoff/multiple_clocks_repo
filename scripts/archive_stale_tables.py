#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Move every OLD (superseded) label table and every result carrying old ROI
labels out of the way, into one archive folder. (2026-11-23, SK: "once and
for all".)

What moves (relative paths are kept, so everything can be moved back):
  1. result folders under group/ holding a per-cell table whose ROI labels
     differ from the final assignment -- from the audit
     (group/roi_label_audit_2026-11-23/stale_tables.csv). The unit moved is
     the run folder (group/<analysis>/<run>); a stale table directly in a
     dated analysis folder moves that folder; one directly in an undated
     folder moves on its own.
  2. outputs of 2026-11-23 built on the old labels that hold no per-cell
     table themselves (STALE_TODAY).
  3. superseded label / anatomy tables at the top of derivatives/
     (ROOT_OLD). Kept: the final table (ROI_assignment/), its copy
     neurons_with_ROI_labels.csv, and the inputs the ROI pipeline still
     reads (neurons_MNI_latest.csv, old_electrode_tables/).

    python scripts/archive_stale_tables.py          # dry run: list only
    python scripts/archive_stale_tables.py --move   # move + manifest

Output: derivatives/_archive_old_labels_<date>/ with the moved tree,
manifest.csv (original -> archived, why) and README.md.

@author: Svenja Kuchenhoff
"""
import os
import re
import sys
import shutil
import datetime

import pandas as pd

DERIV = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
AUDIT = os.path.join(DERIV, "group", "roi_label_audit_2026-11-23", "stale_tables.csv")
ARCHIVE = os.path.join(DERIV, f"_archive_old_labels_{datetime.date.today()}")
STALE_TODAY = [
    "group/per_lag_estimator_comparison_2026-11-23",
    "group/per_lag_cross_phase_2026-11-23",
    "group/per_lag_encode_vs_execute_2026-11-23",
    "group/swr/plan_code_vs_ripples_lagfree_2026-11-23",
]
ROOT_OLD = [
    "ROI_assignment_27thofAug_2026",          # superseded by ROI_assignment/
    "human_cell_labels.xlsx", "human_cell_labels-latest.xlsx",
    "human_cell_labels-INBETWEENSAVE.xlsx", "human_cell_labels-INBETWEENSAV2E.xlsx",
    "human_cell_labels-INBETWEENSAVE3.xlsx",
    "neurons_with_final_roi_labels.numbers", "neurons_MNI.csv",
    "neurons_MNI_latest.numbers", "label_qc.mat", "label_qc_flagged.csv",
    "label_qc_table.csv", "leftover_unassigned_cells.csv",
    "encoding_results-ACC-sig.csv",
]
DATED = re.compile(r"20\d\d-\d\d-\d\d")


def targets():
    S = pd.read_csv(AUDIT)
    rows = {}
    for f, n, tr, mod in S[["file", "n_stale", "transitions", "modified"]].itertuples(index=False):
        parts = f.split("/")                    # group/<analysis>/...
        if len(parts) >= 4:
            t = "/".join(parts[:3])
        elif DATED.search(parts[1]):
            t = "/".join(parts[:2])
        else:
            t = f
        r = rows.setdefault(t, dict(original=t, kind="result with old ROI labels",
                                    n_stale_cells=0, transitions=set(), modified=mod))
        r["n_stale_cells"] = max(r["n_stale_cells"], int(n))
        r["transitions"].add(str(tr))
        r["modified"] = max(r["modified"], mod)
    for t in STALE_TODAY:
        rows.setdefault(t, dict(original=t, kind="2026-11-23 output built on old labels",
                                n_stale_cells=None, transitions=set(), modified="2026-11-23"))
    for t in ROOT_OLD:
        rows[t] = dict(original=t, kind="superseded label / anatomy table",
                       n_stale_cells=None, transitions=set(), modified="")
    T = pd.DataFrame(rows.values())
    T["transitions"] = T.transitions.map(lambda s: " | ".join(sorted(s)))
    # a target inside another target moves with its parent
    T = T[[not any(o != x and x.startswith(o + "/") for o in T.original)
           for x in T.original]]
    T["exists"] = T.original.map(lambda p: os.path.exists(os.path.join(DERIV, p)))
    T["archived"] = T.original.map(lambda p: os.path.relpath(
        os.path.join(ARCHIVE, p), DERIV))
    return T.sort_values("original").reset_index(drop=True)


if __name__ == "__main__":
    T = targets()
    pd.set_option("display.width", 200, "display.max_rows", 400,
                  "display.max_colwidth", 90)
    print(T[["original", "kind", "exists"]].to_string())
    print(f"\n{len(T)} targets, {T.exists.sum()} exist")
    if "--move" not in sys.argv:
        print("dry run -- pass --move to move them")
        sys.exit()
    os.makedirs(ARCHIVE, exist_ok=True)
    for p in T[T.exists].original:
        dst = os.path.join(ARCHIVE, p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(os.path.join(DERIV, p), dst)
    T.to_csv(os.path.join(ARCHIVE, "manifest.csv"), index=False)
    print(f"moved {T.exists.sum()} -> {ARCHIVE}")
