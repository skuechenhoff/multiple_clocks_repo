#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ucla_label_vs_atlas_roi.py — do UCLA's own anatomical labels agree with the
atlas ROIs we assigned in `cell_to_roi_july26.py`?

UCLA ship `sub-{NNN}_localizations.xlsx`, whose `Sheet1` carries, per contact,
several independent anatomical verdicts. `cell_to_roi_july26.py` uses only the
MNI coordinate from that file and throws every label away; this script puts the
labels back and asks how often they land on the same ROI.

The label columns compared (most subject-specific first):

  ASHS_ABC             ASHS segmentation of the subject's own T2 -> MTL
                       subfields (CA1/DG/ERC/PHC). Only defined near the MTL.
  aparc+aseg           FreeSurfer segmentation of the subject's own T1,
                       native space. The 'precise' method.
  aparc.DKTatlas+aseg  same, DKT parcellation.
  Anat                 SPM Anatomy toolbox (Juelich cytoarchitecture) at the
                       MNI coordinate.
  NMM                  Neuromorphometrics atlas label.
  AnatMacro_1          SPM Anatomy macro label.
  region               the electrode's implantation-target code (LEC, RAH,
                       LAC ...). This is the clinical intent label that
                       cell_to_roi_july26.py deliberately never uses.

Comparison rules
----------------
* Every label is mapped onto our ROI vocabulary by `map_ucla_label`. Labels
  that name tissue rather than a region (`Cerebral White Matter`, `Unknown`)
  map to NaN and are counted separately as "no verdict" -- they are not
  scored as disagreements, because they carry no anatomical claim.
* Our HC_anterior / HC_mid split is our own y = -21 rule; no UCLA column makes
  that distinction, so agreement is scored on the collapsed ROI (both -> HC).
  The raw ROI is kept in the output table.
* Nothing here changes the ROI table. Read-only comparison.

Outputs -> derivatives/ROI_assignment/ucla_label_agreement_<date>/
    per_cell_labels.csv     one row per UCLA cell, all label columns + verdicts
    per_bundle_labels.csv   one row per microwire bundle (the unit UCLA label)
    agreement_summary.csv   match rate per label column
    confusion_<col>.csv     our ROI x UCLA ROI, for the main columns
    settings.json           paths + counts of this run
"""
import os
import re
import json
import datetime
import numpy as np
import pandas as pd

PATH_V2026 = ("/Users/xpsy1114/Documents/projects/multiple_clocks/"
              "data/ephys_humans/ABCD_pts_elecFilesForSvenja_v2026")
PATH_CELLS = ("/Users/xpsy1114/Documents/projects/multiple_clocks/"
              "data/ephys_humans/derivatives/neurons_with_ROI_labels.csv")
OUT_ROOT = ("/Users/xpsy1114/Documents/projects/multiple_clocks/"
            "data/ephys_humans/derivatives/ROI_assignment")

UCLA_SUBJECT_TO_FILE = {
    "UC3-0559": "sub-559", "UC3-0573": "sub-573", "UC2-0576": "sub-576",
    "UC3-0577": "sub-577", "UC2-0578": "sub-578", "UC3-0582": "sub-582",
}

LABEL_COLS = ["ASHS_ABC", "aparc+aseg", "aparc.DKTatlas+aseg",
              "Anat", "NMM", "AnatMacro_1", "region"]

# Substrings that say "no anatomical claim" rather than "a different region".
# FreeSurfer writes `Left-Cerebral-White-Matter` / `wm-lh-insula`, ASHS writes
# `(extra-axial)`, all three atlases write `Unknown` -- none of these is a
# competing regional verdict, so they are scored as "no verdict", not as a
# disagreement.
NO_VERDICT = ("unknown", "white matter", "extra axial", "wm ", "nan")


def map_ucla_label(text):
    """One UCLA label string -> our ROI vocabulary, or NaN if it makes no
    anatomical claim (white matter / Unknown). Matching is on substrings, so
    the same function handles FreeSurfer (`ctx-lh-precuneus`), Neuromorpho-
    metrics (`Left PCu precuneus`) and SPM Anatomy (`CA1 (Hippocampus)`).
    Hyphens/underscores are flattened to spaces first so that the FreeSurfer
    and prose spellings of the same region hit the same rule."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return np.nan
    t = re.sub(r"[-_/]", " ", str(text).strip().lower())
    t = re.sub(r"\s+", " ", t)
    if any(k in t for k in NO_VERDICT) or not t:
        return np.nan

    # --- MTL ------------------------------------------------------------
    if "entorhinal" in t or "_erc" in t or t.endswith("erc"):
        return "EC"
    if ("hippocamp" in t and "para" not in t) or "subiculum" in t \
            or "hippocampal_sulcus" in t or t.startswith(("ca1", "ca2", "ca3")) \
            or t in ("left dg", "right dg") or " dg" in t:
        return "HC"
    if "parahippocampal" in t or " phg " in t or t.endswith(" phc") \
            or "phc" in t:
        return "PHC"
    if "amygdala" in t:
        return "Amygdala"

    # --- medial frontal --------------------------------------------------
    # our rules split anterior cingulate at y = 10 (mPFC vs medial_CC); no
    # UCLA column encodes that, so anterior cingulate -> mPFC and middle
    # cingulate -> medial_CC.
    if "anterior cingulate" in t or "anteriorcingulate" in t \
            or "acc" in t.split() or "rostralanterior" in t \
            or "caudalanteriorcingulate" in t:
        return "mPFC"
    if "medialorbitofrontal" in t or "mid orbital" in t or "orbitofrontal gyrus" in t \
            or "gyrus rectus" in t or "morg" in t:
        return "mOFC"
    if "superiorfrontal" in t or "superior frontal gyrus medial" in t \
            or "msfg" in t or "superiorfrontal" in t:
        return "SFG"
    if "middle cingulate" in t or "middlecingulate" in t or "mcgg" in t or t.strip() in ("l mcc", "r mcc") \
            or "mcc" in t.split():
        return "medial_CC"
    if "posterior cingulate" in t or "posteriorcingulate" in t \
            or "pcgg" in t or "precuneus" in t \
            or "pcu " in t or t.strip() in ("l pcc", "r pcc") or "isthmus" in t:
        return "PCC"

    # --- other -----------------------------------------------------------
    if "lingual" in t or "calcarine" in t or "cuneus" in t or "fusiform" in t \
            or "occipital" in t or "hoc" in t or "fg3" in t or "fg4" in t \
            or "lig " in t or "calc " in t or "fug " in t:
        return "Visual"
    if "insula" in t or "operculum" in t or "insular" in t or "op3" in t:
        return "Insula"
    if "thalam" in t or t.startswith("thal:") or "pulvinar" in t:
        return "Thalamus"
    if "putamen" in t or "pallidum" in t or "caudate" in t or "accumbens" in t:
        return "Striatum"
    if "temporal" in t or "planum" in t or "heschl" in t or " te " in t:
        return "Temporal"
    if "supramarginal" in t or "parietal" in t or "postcentral" in t:
        return "Parietal"
    if "motor" in t or "precentral" in t or "sma" in t or "rolandic" in t:
        return "Motor"
    if "inferior frontal" in t or "oriflg" in t or "orifg" in t \
            or "lateralorbitofrontal" in t or "frontal pole" in t or "fp2" in t:
        return "lateral_frontal"
    if "frontal" in t:
        return "other_frontal"
    return "other"


# electrode-name codes -> ROI (the `region` column is a code, not prose)
REGION_CODE_MAP = {
    "AC": "mPFC", "MH": "HC", "AH": "HC", "SUB": "HC", "A": "Amygdala",
    "EC": "EC", "PHG": "PHC", "OF": "mOFC", "AI": "Insula", "MI": "Insula",
    "HG": "Temporal", "SMA": "Motor", "FSG": "Visual", "SMGA": "Parietal",
    "TO": "Visual", "TP": "PCC", "O": "Visual", "Oa": "Visual", "Op": "Visual",
    "Os": "Visual", "PO": "PCC", "P": "Thalamus", "Pv": "Thalamus",
}


def map_region_code(code):
    """`LAC` / `RPHG` / `LOa` -> ROI. Strips the leading L/R hemisphere letter."""
    if not isinstance(code, str):
        return np.nan
    c = code.strip()
    stem = c[1:] if len(c) > 1 and c[0] in "LR" else c
    return REGION_CODE_MAP.get(stem, np.nan)


def collapse(roi):
    """HC_anterior / HC_mid -> HC (UCLA labels make no such distinction)."""
    if not isinstance(roi, str):
        return np.nan
    return "HC" if roi.startswith("HC") else roi


# =============================================================================
# LOAD
# =============================================================================
print("Loading UCLA localization files...")
micro_rows = []
for subj, prefix in UCLA_SUBJECT_TO_FILE.items():
    fpath = os.path.join(PATH_V2026, f"{prefix}_localizations.xlsx")
    d = pd.read_excel(fpath, sheet_name="Sheet1")
    d = d[d["isMicro"].astype(str).str.upper().isin(["TRUE", "1", "1.0"])].copy()
    d["subject_label"] = subj
    micro_rows.append(d)
micro = pd.concat(micro_rows, ignore_index=True)
print(f"  {len(micro)} microwire rows across {micro['subject_label'].nunique()} subjects")

cells = pd.read_csv(PATH_CELLS)
ucla = cells[cells["Recording Site"].astype(str).str.lower() == "ucla"].copy()
ucla["subject_label"] = ucla["Subject Label"].astype(str).str.strip("'\" ")
print(f"  {len(ucla)} UCLA cells in the ROI table")

merged = ucla.merge(
    micro[["subject_label", "electrode"] + LABEL_COLS],
    left_on=["subject_label", "source_electrode"],
    right_on=["subject_label", "electrode"], how="left", validate="m:1")
assert merged["electrode"].notna().all(), "some UCLA cells did not match a row"

# =============================================================================
# MAP LABELS -> ROI
# =============================================================================
for col in LABEL_COLS:
    out = f"ucla_roi__{col}"
    if col == "region":
        merged[out] = merged[col].apply(map_region_code)
    else:
        merged[out] = merged[col].apply(map_ucla_label)

merged["our_roi_collapsed"] = merged["atlas_roi"].apply(collapse)
merged["our_alt_roi_collapsed"] = merged["alt_final_roi"].apply(collapse)

# a single "best available UCLA verdict": most subject-specific column that
# actually makes a claim at this contact.
PRIORITY = ["ASHS_ABC", "aparc+aseg", "Anat", "NMM", "AnatMacro_1"]
def best(row):
    for col in PRIORITY:
        v = row[f"ucla_roi__{col}"]
        if isinstance(v, str):
            return pd.Series([v, col])
    return pd.Series([np.nan, ""])
merged[["ucla_roi__best", "ucla_best_source"]] = merged.apply(best, axis=1)

# any column agreeing counts as agreement for the permissive measure
def any_agree(row):
    ours = row["our_roi_collapsed"]
    if not isinstance(ours, str):
        return np.nan
    vals = [row[f"ucla_roi__{c}"] for c in PRIORITY]
    vals = [v for v in vals if isinstance(v, str)]
    if not vals:
        return np.nan
    return ours in vals
merged["any_ucla_col_agrees"] = merged.apply(any_agree, axis=1)

# =============================================================================
# SCORE
# =============================================================================
def score(df, unit):
    rows = []
    for col in LABEL_COLS + ["best"]:
        c = f"ucla_roi__{col}"
        sub = df[df["our_roi_collapsed"].notna()]
        has = sub[c].notna()
        agree = (sub[c] == sub["our_roi_collapsed"]) & has
        rows.append(dict(
            unit=unit, label_column=col,
            n_total=len(sub), n_with_verdict=int(has.sum()),
            n_agree=int(agree.sum()),
            pct_of_verdicts=round(100 * agree.sum() / max(has.sum(), 1), 1),
            pct_of_all=round(100 * agree.sum() / max(len(sub), 1), 1)))
    sub = df[df["our_roi_collapsed"].notna()]
    a = sub["any_ucla_col_agrees"]
    rows.append(dict(unit=unit, label_column="ANY_of_5_atlases",
                     n_total=len(sub), n_with_verdict=int(a.notna().sum()),
                     n_agree=int((a == True).sum()),
                     pct_of_verdicts=round(100 * (a == True).sum() / max(a.notna().sum(), 1), 1),
                     pct_of_all=round(100 * (a == True).sum() / max(len(sub), 1), 1)))
    return pd.DataFrame(rows)


bundles = merged.drop_duplicates(subset=["subject_label", "source_electrode"]).copy()
summary = pd.concat([score(merged, "cell"), score(bundles, "bundle")],
                    ignore_index=True)

print("\n=== AGREEMENT: UCLA label vs our atlas_roi (HC_ant/HC_mid collapsed) ===")
print(summary.to_string(index=False))

# same, but restricted to the six ROIs that survive into analyses
kept = merged[merged["alt_final_roi"].notna()]
kept_bundles = kept.drop_duplicates(subset=["subject_label", "source_electrode"])
summary_kept = pd.concat([score(kept, "cell_analysis_ROIs_only"),
                          score(kept_bundles, "bundle_analysis_ROIs_only")],
                         ignore_index=True)
print("\n=== restricted to cells kept in analyses (alt_final_roi not NaN) ===")
print(summary_kept.to_string(index=False))

# =============================================================================
# SAVE
# =============================================================================
stamp = datetime.date.today().isoformat()
out_dir = os.path.join(OUT_ROOT, f"ucla_label_agreement_{stamp}")
os.makedirs(out_dir, exist_ok=True)

keep_cols = (["subject_label", "cell idx", "electrode label", "source_electrode",
              "MNI_x_final", "MNI_y_final", "MNI_z_final",
              "atlas_roi", "alt_final_roi", "atlas_source_label", "atlas_reason",
              "our_roi_collapsed"] + LABEL_COLS +
             [f"ucla_roi__{c}" for c in LABEL_COLS] +
             ["ucla_roi__best", "ucla_best_source", "any_ucla_col_agrees"])
merged[keep_cols].to_csv(os.path.join(out_dir, "per_cell_labels.csv"), index=False)
bundles[keep_cols + ["subject"]].assign(
    n_cells=bundles.apply(lambda r: int(((merged["subject_label"] == r["subject_label"]) &
                                         (merged["source_electrode"] == r["source_electrode"])).sum()), axis=1)
).to_csv(os.path.join(out_dir, "per_bundle_labels.csv"), index=False)
pd.concat([summary, summary_kept], ignore_index=True).to_csv(
    os.path.join(out_dir, "agreement_summary.csv"), index=False)

for col in ["best", "aparc+aseg", "NMM", "Anat", "region"]:
    ct = pd.crosstab(merged["our_roi_collapsed"], merged[f"ucla_roi__{col}"],
                     dropna=False)
    ct.to_csv(os.path.join(out_dir, f"confusion_{col.replace('+','_')}.csv"))

with open(os.path.join(out_dir, "settings.json"), "w") as f:
    json.dump(dict(
        date=stamp, cell_table=PATH_CELLS, ucla_files=PATH_V2026,
        n_ucla_cells=int(len(merged)), n_bundles=int(len(bundles)),
        n_cells_in_analysis_ROIs=int(len(kept)),
        label_columns=LABEL_COLS, priority_for_best=PRIORITY,
        hc_collapsed="HC_anterior and HC_mid scored as HC",
        no_verdict_labels=list(NO_VERDICT)), f, indent=2)

print(f"\nSaved -> {out_dir}")

# --- readable per-bundle table for eyeballing -------------------------------
print("\n=== per bundle ===")
show = bundles[["subject_label", "source_electrode", "region", "atlas_roi",
                "ucla_roi__best", "ucla_best_source",
                "ucla_roi__NMM", "ucla_roi__aparc+aseg", "ucla_roi__Anat",
                "any_ucla_col_agrees"]].copy()
show["n_cells"] = [int(((merged["subject_label"] == r.subject_label) &
                        (merged["source_electrode"] == r.source_electrode)).sum())
                   for r in bundles.itertuples()]
pd.set_option("display.width", 250)
print(show.sort_values(["subject_label", "source_electrode"]).to_string(index=False))


# =============================================================================
# FOCUSED REPORT
# =============================================================================
# (a) Sanity check on the native-space FreeSurfer columns. `aparc+aseg` is the
#     column UCLA describe as precise, but it is only as good as the T1<->MNI
#     bookkeeping in each subject's own file. Compare it against NMM per
#     subject: a subject where the two never agree has a broken column, not a
#     precise one.
print("\n=== aparc+aseg vs NMM, per subject (broken-column check) ===")
rows = []
for subj, g in micro.assign(
        a=micro["aparc+aseg"].apply(map_ucla_label),
        n=micro["NMM"].apply(map_ucla_label)).groupby("subject_label"):
    both = g[g["a"].notna() & g["n"].notna()]
    rows.append(dict(subject=subj, n_micro=len(g), n_both_verdict=len(both),
                     n_aparc_eq_nmm=int((both["a"] == both["n"]).sum()),
                     pct=round(100 * (both["a"] == both["n"]).mean(), 1)
                     if len(both) else np.nan))
fs_check = pd.DataFrame(rows)
print(fs_check.to_string(index=False))
fs_check.to_csv(os.path.join(out_dir, "freesurfer_column_sanity.csv"), index=False)

# (b) Bundle-level verdict for the cells that actually enter analyses.
print("\n=== bundles carrying analysis cells: ours vs UCLA ===")
kb = merged[merged["alt_final_roi"].notna()].groupby(
    ["subject_label", "source_electrode"]).agg(
        n_cells=("cell idx", "size"),
        our_roi=("alt_final_roi", "first"),
        ucla_best=("ucla_roi__best", "first"),
        ucla_NMM=("ucla_roi__NMM", "first"),
        ucla_aparc=("ucla_roi__aparc+aseg", "first"),
        ucla_Anat=("ucla_roi__Anat", "first"),
        ucla_ASHS=("ucla_roi__ASHS_ABC", "first"),
        agree=("any_ucla_col_agrees", "first")).reset_index()
kb = kb.sort_values("n_cells", ascending=False)
print(kb.to_string(index=False))
kb.to_csv(os.path.join(out_dir, "analysis_bundles_verdict.csv"), index=False)

n_dis = int((kb["agree"] == False).sum())
print(f"\n{len(kb) - n_dis}/{len(kb)} analysis bundles have at least one UCLA "
      f"column agreeing; {n_dis} disagree on every column "
      f"({int(kb.loc[kb['agree'] == False, 'n_cells'].sum())} cells).")


# (c) Does agreement track how the ROI was assigned? Rule 12 of
#     `assign_atlas_roi` probes a +-1/2/3 mm cube for cells whose exact voxel
#     is white matter (`atlas_reason` contains `neighbor@`). Those are the
#     assignments we should trust least, so split agreement by that flag.
print("\n=== agreement by assignment mode (exact voxel vs neighbourhood probe) ===")
merged["assign_mode"] = np.where(
    merged["atlas_reason"].astype(str).str.contains("neighbor@"),
    "neighbourhood probe", "exact voxel")
rows = []
for mode, g in merged[merged["our_roi_collapsed"].notna()].groupby("assign_mode"):
    gb = g.drop_duplicates(subset=["subject_label", "source_electrode"])
    a = g["any_ucla_col_agrees"]
    ab = gb["any_ucla_col_agrees"]
    rows.append(dict(assign_mode=mode, n_cells=len(g),
                     n_cells_with_verdict=int(a.notna().sum()),
                     n_cells_agree=int((a == True).sum()),
                     pct_cells=round(100 * (a == True).sum() / max(a.notna().sum(), 1), 1),
                     n_bundles=len(gb), n_bundles_agree=int((ab == True).sum())))
mode_tbl = pd.DataFrame(rows)
print(mode_tbl.to_string(index=False))
mode_tbl.to_csv(os.path.join(out_dir, "agreement_by_assignment_mode.csv"), index=False)


# (d) Per-analysis-ROI verdict: for each of the six ROIs that survive into
#     analyses, how many UCLA cells does UCLA's own labelling confirm?
print("\n=== per analysis ROI: UCLA cells confirmed ===")
rows = []
for roi, g in merged[merged["alt_final_roi"].notna()].groupby("alt_final_roi"):
    strict = g[[f"ucla_roi__{c}" for c in PRIORITY]].eq(
        g["our_roi_collapsed"], axis=0)
    has = g[[f"ucla_roi__{c}" for c in PRIORITY]].notna()
    rows.append(dict(
        roi=roi, n_cells=len(g),
        n_bundles=g.groupby(["subject_label", "source_electrode"]).ngroups,
        n_any_col_agrees=int((g["any_ucla_col_agrees"] == True).sum()),
        n_all_cols_agree=int((strict.sum(1) == has.sum(1)).sum()),
        n_best_col_agrees=int((g["ucla_roi__best"] == g["our_roi_collapsed"]).sum()),
        ucla_alternatives="; ".join(sorted(set(
            f"{v}" for v in g["ucla_roi__best"].dropna().unique()
            if v != collapse(roi))))))
roi_tbl = pd.DataFrame(rows).sort_values("n_cells", ascending=False)
print(roi_tbl.to_string(index=False))
roi_tbl.to_csv(os.path.join(out_dir, "per_analysis_roi_verdict.csv"), index=False)


# =============================================================================
# (e) WHICH PROBE PATH, AND WHAT DID IT OVERRIDE?
# =============================================================================
# There are two neighbourhood probes, and they are not equally safe:
#
#   path A  the HC probe *inside* rule 4 (`hippocampal_subfield_..._neighbor@`)
#           fires whenever the exact voxel is not hippocampus, and returns
#           before rules 5-11 are ever consulted. It can therefore override a
#           perfectly good exact-voxel verdict from a lower-priority rule.
#   path B  rule 12 proper (`atlas_neighbor@`) is reached only when NO rule
#           matched at the exact voxel, so it can only promote `leftover`.
#           It cannot contradict an exact-voxel label.
#
# Path A is confined to HC_anterior/HC_mid by construction (it is inside rule
# 4), which is why no other ROI is exposed to this failure mode. This block
# re-queries the exact voxel of every path-A cell to see what was overridden.
print("\n=== probe path by ROI ===")
r = merged_all_reason = cells["atlas_reason"].astype(str)
cells["probe_path"] = np.where(
    r.str.contains(r"hippocampal_subfield.*neighbor@"), "A: HC probe (rule 4)",
    np.where(r.str.contains("atlas_neighbor@"), "B: rule 12", "exact voxel"))
in_analysis = cells[cells["alt_final_roi"].notna()]
print(pd.crosstab(in_analysis["alt_final_roi"], in_analysis["probe_path"]).to_string())

from mc.analyse.anatomy_atlas import get_atlases  # noqa: E402
ho_cort, ho_sub, _juelich, _bn = get_atlases(
    "/Users/xpsy1114/Documents/toolboxes/Brainnatome")

pathA = in_analysis[in_analysis["probe_path"] == "A: HC probe (rule 4)"].copy()


def exact_voxel_verdict(row):
    """What rules 5-11 would have said at the cell's own voxel."""
    xyz = np.array([row["MNI_x_final"], row["MNI_y_final"],
                    row["MNI_z_final"]], float)
    s = (ho_sub.label_at(xyz) or "").lower()
    c = (ho_cort.label_at(xyz) or "").lower()
    if "thalamus" in s:
        return "Thalamus"
    if "amygdala" in s:
        return "Amygdala"
    if "parahippocampal" in c:
        return "PHC (intended override)"
    if "white matter" in s or (not s and not c):
        return "white matter (probe justified)"
    return f"other: {c or s}"


pathA["overridden_exact_label"] = pathA.apply(exact_voxel_verdict, axis=1)
print("\n=== what the rule-4 HC probe overrode (cells in analysis ROIs) ===")
ov = pd.crosstab(pathA["overridden_exact_label"],
                 [pathA["alt_final_roi"], pathA["Recording Site"]], margins=True)
print(ov.to_string())
ov.to_csv(os.path.join(out_dir, "hc_probe_overrides.csv"))
pathA.to_csv(os.path.join(out_dir, "hc_probe_cells_all_sites.csv"), index=False)
print(f"\nSaved override tables -> {out_dir}")
