#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit: which SAVED per-cell result tables carry ROI labels that differ from
the final assignment (`mc.analyse.roi_labels.FINAL_ROI_TABLE`), and which
scripts still point at them. (2026-11-23)

Scans every .csv under ephys_humans/derivatives/group (< 60 MB) that has an
ROI column and cell keys:
    (subject_id | subject, neuron)     cell index parsed from the neuron label
    (subject_id | subject, cell_idx | cell idx)   1-based
    (session, cell)                    0-based (ripple pipeline)
Saved labels are canonicalised (ACC -> mPFC, medialOFC -> mOFC, ...); a saved
'HC' / 'HC_all' matches either HC subfield. A row is stale when the saved
label is an analysis ROI and differs from the final one.

Output: group/roi_label_audit_<date>/stale_tables.csv, all_tables.csv,
scripts_pointing_at_stale.csv, summary.md

    python scripts/audit_roi_labels.py

@author: Svenja Kuchenhoff
"""
import os
import re
import sys
import glob
import datetime

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from mc.analyse.roi_labels import final_labels, FINAL_ROI_TABLE

DERIV = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
OUT = os.path.join(DERIV, "group", f"roi_label_audit_{datetime.date.today()}")
ANALYSIS = {"EC", "HC_anterior", "HC_mid", "mPFC", "mOFC", "PCC", "PHC"}
OLD = {"ACC": "mPFC", "medialOFC": "mOFC", "medial OFC": "mOFC",
       "Parahippocampal": "PHC", "Parahippocampus": "PHC"}
ROI_COLS = ("roi", "ROI", "roi_ripple_pipeline", "alt_final_roi", "atlas_roi")


def keys(df):
    """(subject, 1-based cell index) arrays, or None."""
    subj = next((c for c in ("subject_id", "subject", "session")
                 if c in df), None)
    if subj is None:
        return None
    if "neuron" in df and df["neuron"].astype(str).str.match(r"^\d+_\d+").all():
        ci = df["neuron"].astype(str).str.split("_").str[1].str.split("-").str[0]
        return df[subj], ci.astype(int)
    for c in ("cell_idx", "cell idx"):
        if c in df:
            return df[subj], df[c]
    if subj == "session" and "cell" in df:
        return df[subj], df["cell"] + 1
    return None


def audit(path):
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    rc = next((c for c in ROI_COLS if c in df), None)
    k = keys(df) if rc else None
    if k is None:
        return None
    s, c = (pd.to_numeric(x, errors="coerce") for x in k)
    ok = s.notna() & c.notna()
    if not ok.any():
        return None
    t = pd.DataFrame({"subject": s[ok].astype(int), "cell_idx": c[ok].astype(int),
                      "saved": df.loc[ok, rc].map(lambda r: OLD.get(r, r))})
    t = t.drop_duplicates(["subject", "cell_idx", "saved"])
    t = t.merge(final_labels(), on=["subject", "cell_idx"], how="left")
    hc = t.saved.isin(["HC", "HC_all"])
    same = (t.saved.astype(str) == t.roi.astype(str)) | (
        hc & t.roi.astype(str).str.startswith("HC_"))
    judged = t.saved.isin(ANALYSIS) | hc
    bad = t[judged & ~same]
    trans = (bad.groupby(["saved", "roi"], dropna=False).size()
             .rename("n").reset_index())
    return dict(file=os.path.relpath(path, DERIV), roi_column=rc,
                n_cells=int(judged.sum()), n_stale=int(len(bad)),
                transitions="; ".join(f"{a}->{b}: {n}" for a, b, n in
                                      trans.itertuples(index=False)),
                modified=datetime.datetime.fromtimestamp(
                    os.path.getmtime(path)).strftime("%Y-%m-%d"))


def scripts_pointing_at(stale_dirs):
    rows = []
    files = [f for f in glob.glob(os.path.join(REPO, "**", "*.py"),
                                  recursive=True)
             if "/old/" not in f and "/archived/" not in f]
    for f in files:
        txt = open(f, errors="ignore").read()
        for d in stale_dirs:
            if d in txt:
                rows.append(dict(script=os.path.relpath(f, REPO), points_at=d))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for p in glob.glob(os.path.join(DERIV, "group", "**", "*.csv"),
                       recursive=True):
        if os.path.getsize(p) < 60e6 and "roi_label_audit" not in p:
            r = audit(p)
            if r is not None:
                rows.append(r)
    A = pd.DataFrame(rows).sort_values(["n_stale", "file"],
                                       ascending=[False, True])
    A.to_csv(os.path.join(OUT, "all_tables.csv"), index=False)
    S = A[A.n_stale > 0]
    S.to_csv(os.path.join(OUT, "stale_tables.csv"), index=False)
    # result folders holding a stale table (the folder name is what scripts cite)
    dirs = sorted({os.path.basename(os.path.dirname(os.path.join(DERIV, f)))
                   for f in S.file})
    P = scripts_pointing_at([d for d in dirs if len(d) > 12])
    P.to_csv(os.path.join(OUT, "scripts_pointing_at_stale.csv"), index=False)
    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write(f"# ROI-label audit, {datetime.date.today()}\n\n"
                 f"Final labels: `{FINAL_ROI_TABLE}`.\n\n"
                 f"- per-cell tables scanned: {len(A)}\n"
                 f"- tables with >= 1 stale label: {len(S)}\n"
                 f"- live scripts citing a folder that holds a stale table: "
                 f"{P.script.nunique() if len(P) else 0}\n\n"
                 "A stale SAVED table is harmless when the script that reads it "
                 "re-applies the final labels (`mc.analyse.roi_labels."
                 "read_per_cell_table`) or recomputes from the live loaders. "
                 "See `scripts_pointing_at_stale.csv`.\n")
    print(f"{len(A)} tables, {len(S)} stale")
    print(S[["file", "n_cells", "n_stale", "modified"]].head(60).to_string())
    print(P.to_string() if len(P) else "no live script points at a stale folder")
