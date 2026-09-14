#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atlas lookup and ROI-assignment rules, shared by the cell pipeline and the
LFP/ripple pipeline.

EXTRACTED VERBATIM from scripts/cell_to_roi_july26.py: `HC_ANT_MID_Y` (953),
`load_brainnetome_labels` (982), `MaxProbAtlas` (1004), `_contains_any` (1044)
and `assign_atlas_roi` (1051). One implementation of the ROI rules, so that
"the same anatomical criteria defined HC for cells and for LFP contacts" is
literally true.

The only structural change: the four atlas objects were module-level globals
built by `nldatasets.fetch_*` calls at import time. They are now built by
`get_atlases()`, which populates the same module-level names. `assign_atlas_roi`
still reads the module globals `juelich` and `HC_ANT_MID_Y` exactly as before,
so its body is unchanged -- callers must simply call `get_atlases()` first.

Regression gate: cell_to_roi_july26.py must still reproduce
derivatives/neurons_with_ROI_labels.csv byte-for-byte
(md5 e0e758a303831cfc614a2490dcaf6aac as of 2026-08-26).

@author: Svenja Kuchenhoff
"""

import os
import re

import numpy as np
import nibabel as nib


# Split y (mm). Poppenk & Moscovitch 2013 (anterior HC is preferentially
# connected to mPFC/mOFC via the uncinate fasciculus). Same value used in
# cell_to_roi_MNI.py so results remain comparable.
HC_ANT_MID_Y = -21.0

# Brainnetome lives outside the data tree, so it needs searching rather than
# hardcoding: the original path was a laptop-only location and broke on ceph.
# Override with $BRAINNETOME_DIR.
_BRAINNETOME_CANDIDATES = [
    os.environ.get("BRAINNETOME_DIR", ""),
    "/Users/xpsy1114/Documents/toolboxes/Brainnatome",
    "/ceph/behrens/svenja/analysis/toolboxes/Brainnatome",
    "/ceph/behrens/svenja/toolboxes/Brainnatome",
    os.path.expanduser("~/toolboxes/Brainnatome"),
    os.path.expanduser("~/Brainnatome"),
]


def find_brainnetome_dir():
    """First candidate directory that actually holds the atlas, else None."""
    for d in _BRAINNETOME_CANDIDATES:
        if d and os.path.isfile(os.path.join(d, "BN_Atlas_246_LUT.txt")):
            return d
    return None


DEFAULT_BRAINNETOME_DIR = find_brainnetome_dir()

# Populated by get_atlases(). `assign_atlas_roi` reads `juelich` from here.
ho_cort = None
ho_sub = None
juelich = None
brainnetome = None


def get_atlases(brainnetome_dir=DEFAULT_BRAINNETOME_DIR, verbose=True):
    """Fetch/instantiate the four atlases once and populate the module
    globals. Returns (ho_cort, ho_sub, juelich, brainnetome).

    Identical to the import-time block that previously lived at
    cell_to_roi_july26.py:973-979 and :1032-1035.
    """
    global ho_cort, ho_sub, juelich, brainnetome
    if ho_cort is not None:
        return ho_cort, ho_sub, juelich, brainnetome

    from nilearn import datasets as nldatasets

    if verbose:
        print("Loading atlases (already cached by nilearn / local)...")

    ho_cort_atlas = nldatasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    ho_sub_atlas = nldatasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    juelich_atlas = nldatasets.fetch_atlas_juelich("maxprob-thr25-2mm")

    if not brainnetome_dir:
        raise FileNotFoundError(
            "Brainnetome atlas not found. It is only needed for the atlas "
            "CROSS-CHECK -- contact selection uses native-space labels and is "
            "unaffected. Either set BRAINNETOME_DIR, copy the 182 KB folder "
            "(BN_Atlas_246_1mm.nii.gz + BN_Atlas_246_LUT.txt), or run with "
            "--use_atlas=False.")
    brainnetome_nii = os.path.join(brainnetome_dir, "BN_Atlas_246_1mm.nii.gz")
    brainnetome_lut = os.path.join(brainnetome_dir, "BN_Atlas_246_LUT.txt")

    ho_cort = MaxProbAtlas(ho_cort_atlas)
    ho_sub = MaxProbAtlas(ho_sub_atlas)
    juelich = MaxProbAtlas(juelich_atlas)
    brainnetome = MaxProbAtlas(brainnetome_nii,
                               labels=load_brainnetome_labels(brainnetome_lut))
    if verbose:
        print(f"  HO cortical: {len(ho_cort.labels)} labels")
        print(f"  HO subcortical: {len(ho_sub.labels)} labels")
        print(f"  Juelich: {len(juelich.labels)} labels")
        print(f"  Brainnetome: {len(brainnetome.labels)} labels")
    return ho_cort, ho_sub, juelich, brainnetome


def load_brainnetome_labels(lut_path):
    labels = {0: "background"}
    with open(lut_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            idx = None
            for p in parts:
                try:
                    idx = int(p); break
                except ValueError:
                    pass
            if idx is None:
                continue
            text_parts = [p for p in parts
                          if not re.match(r"^-?\d+(\.\d+)?$", p)]
            labels[idx] = " ".join(text_parts).strip() or f"BN_{idx}"
    return labels


class MaxProbAtlas:
    """Single-label-per-voxel lookup. For nilearn Harvard-Oxford/Juelich
    the `maxprob-thr25-2mm` variant assigns each voxel the label of the
    region with highest probability *only if that probability is >= 25 %*
    — voxels below threshold are 0 (background). So a non-background
    return already means the cell passed the 25 % criterion."""
    def __init__(self, atlas_or_path, labels=None):
        maps = getattr(atlas_or_path, "maps", atlas_or_path)
        self.img = nib.load(maps) if isinstance(maps, str) else maps
        self.data = self.img.get_fdata()
        self.inv_affine = np.linalg.inv(self.img.affine)
        if labels is None:
            self.labels = list(atlas_or_path.labels)
        else:
            self.labels = labels

    def label_at(self, xyz):
        v = np.round(nib.affines.apply_affine(self.inv_affine, xyz)).astype(int)
        if np.any(v < 0) or np.any(v >= self.data.shape):
            return None
        idx = int(self.data[tuple(v)])
        if idx == 0:
            return None
        if isinstance(self.labels, dict):
            return self.labels.get(idx)
        return self.labels[idx] if idx < len(self.labels) else None




# =============================================================================
# ROI ASSIGNMENT RULES (priority order)
# =============================================================================

def _contains_any(label, patterns):
    if not label:
        return False
    low = label.lower()
    return any(p.lower() in low for p in patterns)


def assign_atlas_roi(row):
    """Return (atlas_roi, atlas_source_label, atlas_reason). Priority
    order — first rule that matches wins. atlas_roi == 'leftover' if
    nothing matches at the 25 % threshold."""
    y = row["MNI_y_final"]
    jl = row["_juelich"]        # single label or None
    hc = row["_ho_cort"]
    hs = row["_ho_sub"]
    bn = row["_bn_label"]

    # 1. EC — Juelich histological entorhinal cortex, provided y >= -18.
    #    The Juelich EC probability map has a dense core at y ~= -6 to
    #    -12 mm (Amunts et al. 2005, Insausti et al. 1998) and tails off
    #    posteriorly at y ~= -18 to -20 mm where it transitions into the
    #    subiculum. Cells with y < -18 whose Juelich label at 25% is
    #    still "entorhinal" are at the far posterior EC border, where
    #    the atlas overlaps heavily with the subicular complex — those
    #    fall through to rule 4 (Hippocampus), which captures them as
    #    HC via the Juelich subiculum/CA neighborhood extension. Same
    #    -18 mm cutoff is used in the LEC/REC rescue path.
    if _contains_any(jl, ["entorhinal"]) and float(y) >= -18:
        return ("EC", jl, "juelich_entorhinal_y_ge_-18")

    # 2. mPFC — matched by EITHER
    #    (a) Brainnetome A32sg / A32p / A24cd / A24rv / A10m / A9m
    #        (fine cytoarchitectonic parcellation of medial PFC), OR
    #    (b) Harvard-Oxford cortical `Cingulate Gyrus, anterior
    #        division` (coarser gyral label covering the whole ACC).
    #    A32sg (subgenual) stays with mPFC EXCEPT for cells with
    #    z <= -10 which are on the orbital surface (assigned mOFC).
    #    Only cells with y >= 10 stay mPFC (anterior cingulate proper);
    #    more posterior hits become `medial_CC` (mid-cingulate, following
    #    the old cell_to_roi_MNI.py `acc_y_cutoff = 10` convention).
    _mpfc_bn = _contains_any(bn, ["a32sg", "a32p", "a24cd", "a24rv",
                                    "a10m", "a9m"])
    _mpfc_hc = (hc is not None
                and "cingulate gyrus, anterior" in hc.lower())
    if _mpfc_bn or _mpfc_hc:
        z = row["MNI_z_final"]
        if _contains_any(bn, ["a32sg"]) and float(z) <= -10:
            return ("mOFC", bn, "a32sg_ventral_z_le_-10")
        atlas_lab = bn if _mpfc_bn else hc
        src = "brainnetome_mpfc" if _mpfc_bn else "ho_cort_anterior_cingulate"
        if float(y) >= 10:
            return ("mPFC", atlas_lab, src)
        return ("medial_CC", atlas_lab, src + "_but_y_lt_10")

    # 3. mOFC / vOFC — Brainnetome A14m / A13 / A11m
    if _contains_any(bn, ["a14m", "a13", "a11m"]):
        return ("mOFC", bn, "brainnetome_mofc")

    # 4. Hippocampus — assigned when EITHER
    #    (a) HO subcortical returns "Hippocampus" at 25 %, OR
    #    (b) Juelich histological atlas returns one of the non-EC
    #        hippocampal subfields at 25 %: `subiculum` or `cornu
    #        ammonis`, OR
    #    (c) Juelich subiculum / cornu ammonis is within a ±3 mm
    #        cube neighborhood of the coord (extension of rule b to
    #        cover voxels that sit in the immediately adjacent white
    #        matter or transitional cortex — Juelich MTL parcels are
    #        thin and 25 %-thresholded, so a small neighborhood probe
    #        captures the perihippocampal WM that anatomically belongs
    #        to the hippocampal formation but falls outside the strict
    #        maxprob mask).
    #    Rule (b)+(c) matter because HO subcortical uses a coarser
    #    hippocampal mask, and the strict 25 % threshold on Juelich
    #    otherwise leaves subicular voxels to be captured by the
    #    coarser parahippocampal-gyrus rule (rule 5) despite being
    #    anatomically hippocampal.
    _hc_hs   = _contains_any(hs, ["hippocampus"])
    _hc_jl0  = _contains_any(jl, ["subiculum", "cornu ammonis"])
    _hc_jlN  = None            # (label, dist) if found in ±3 mm neighborhood
    if not (_hc_hs or _hc_jl0):
        xyz_here = np.array([row["MNI_x_final"], row["MNI_y_final"],
                              row["MNI_z_final"]], dtype=float)
        if not np.any(np.isnan(xyz_here)):
            for r_probe in range(1, 4):
                found = False
                for dx in (-r_probe, 0, r_probe):
                    for dy in (-r_probe, 0, r_probe):
                        for dz in (-r_probe, 0, r_probe):
                            if dx == dy == dz == 0:
                                continue
                            lab_probe = juelich.label_at(
                                xyz_here + np.array([dx, dy, dz], float))
                            if lab_probe and (
                                "subiculum" in lab_probe.lower()
                                or "cornu ammonis" in lab_probe.lower()):
                                _hc_jlN = (lab_probe, r_probe); found = True; break
                        if found: break
                    if found: break
                if found: break
    if _hc_hs or _hc_jl0 or _hc_jlN:
        if _hc_hs:
            atlas_label = hs; suffix = ""
        elif _hc_jl0:
            atlas_label = jl; suffix = ""
        else:
            atlas_label = _hc_jlN[0]; suffix = f"_neighbor@{_hc_jlN[1]}mm"
        roi = "HC_anterior" if float(y) >= HC_ANT_MID_Y else "HC_mid"
        return (roi, atlas_label,
                f"hippocampal_subfield_split_y{HC_ANT_MID_Y}{suffix}")

    # 5. PHC (parahippocampal cortex) — HO cortical
    if _contains_any(hc, ["parahippocampal"]):
        return ("PHC", hc, "ho_cort_parahippocampal")

    # 6. PCC — Brainnetome A23 (PCC proper) + A31/dmPOS (Precuneus).
    #    Both are DMN posterior-medial hubs (Andrews-Hanna 2010,
    #    Utevsky 2014); collapsed under the CLAUDE.md name `PCC`.
    if _contains_any(bn, ["a23", "a31", "dmpos"]):
        return ("PCC", bn, "brainnetome_pcc_precuneus")
    if hc and "cingulate" in hc.lower() and "posterior" in hc.lower():
        return ("PCC", hc, "ho_cort_pcc")
    if _contains_any(hc, ["precuneous"]):
        return ("PCC", hc, "ho_cort_precuneous")

    # 8. Visual — Juelich V1/V2/V3 or HO cortical occipital
    if _contains_any(jl, ["v1", "v2", "v3", "visual", "calcarine"]):
        return ("Visual", jl, "juelich_visual")
    if _contains_any(hc, ["occipital", "cuneal", "lingual",
                          "intracalcarine", "supracalcarine"]):
        return ("Visual", hc, "ho_cort_occipital")

    # 9. Amygdala — HO subcortical
    if _contains_any(hs, ["amygdala"]):
        return ("Amygdala", hs, "ho_sub_amygdala")

    # 10. Thalamus — HO subcortical
    if _contains_any(hs, ["thalamus"]):
        return ("Thalamus", hs, "ho_sub_thalamus")

    # 11. Insula — HO cortical `Insular Cortex`
    if _contains_any(hc, ["insular"]):
        return ("Insula", hc, "ho_cort_insula")

    # 12. Atlas-neighborhood extension for cells whose exact voxel
    #    sits in white matter or falls outside any 25 %-thresholded
    #    atlas mask, but a target-ROI atlas region is immediately
    #    adjacent (±3 mm cube grid). Purely atlas-based — probes the
    #    same atlases at slightly-offset voxels, no text lookup.
    #    Priority ORDER matches the priority order of the exact-coord
    #    rules 1-11 above (EC first, then mPFC/mOFC, then PHC, PCC,
    #    Visual, Amygdala, Thalamus, Insula). mPFC is checked here
    #    via the HO cortical anterior-cingulate probe used for the
    #    perigenual cingulum bundle.
    xyz_here = np.array([row["MNI_x_final"], row["MNI_y_final"],
                          row["MNI_z_final"]], dtype=float)
    if np.any(np.isnan(xyz_here)):
        return ("leftover", "", "")

    NEIGH_PROBES = [
        # (target_roi, atlas_obj, patterns, extra_y_constraint)
        ("EC",       juelich, ["entorhinal"],
                                     ("y_ge", -18)),   # EC only if y >= -18
        ("mPFC",     ho_cort, ["cingulate gyrus, anterior"],
                                     ("y_ge",  10)),   # mPFC if y >= 10
        ("medial_CC",ho_cort, ["cingulate gyrus, anterior"],
                                     ("y_lt",  10)),   # else medial_CC
        ("mOFC",     brainnetome, ["a11m", "a13", "a14m"], None),
        ("mOFC",     ho_cort, ["frontal orbital", "subcallosal"], None),
        ("PHC",      ho_cort, ["parahippocampal"], None),
        ("PCC",      brainnetome, ["a23", "a31", "dmpos"], None),
        ("PCC",      ho_cort, ["cingulate gyrus, posterior", "precuneous"], None),
        ("Visual",   ho_cort, ["occipital", "cuneal", "lingual",
                                "intracalcarine", "supracalcarine"], None),
        ("Amygdala", ho_sub,  ["amygdala"], None),
        ("Thalamus", ho_sub,  ["thalamus"], None),
        ("Insula",   ho_cort, ["insular"], None),
    ]

    for r_probe in range(1, 4):    # 1, 2, 3 mm
        # Enumerate all offsets at radius r_probe (cube grid).
        offsets = [(dx, dy, dz)
                   for dx in (-r_probe, 0, r_probe)
                   for dy in (-r_probe, 0, r_probe)
                   for dz in (-r_probe, 0, r_probe)
                   if not (dx == dy == dz == 0)]
        for target_roi, atlas_obj, patterns, ycon in NEIGH_PROBES:
            # skip if y-constraint fails at the cell's own coord
            if ycon is not None:
                op, thr = ycon
                if op == "y_ge" and float(y) < thr: continue
                if op == "y_lt" and float(y) >= thr: continue
            for dx, dy, dz in offsets:
                lab_probe = atlas_obj.label_at(
                    xyz_here + np.array([dx, dy, dz], float))
                if lab_probe and any(p.lower() in lab_probe.lower()
                                       for p in patterns):
                    return (target_roi, lab_probe,
                            f"atlas_neighbor@{r_probe}mm")

    return ("leftover", "", "")


# =============================================================================
# HIPPOCAMPAL PROBABILITY (for ranking contacts, not labelling them)
# =============================================================================
# The max-prob atlases give a label, which cannot rank two contacts that are
# both "hippocampus". Selecting ONE contact per electrode needs a continuous
# measure, so this reads the Harvard-Oxford subcortical PROBABILITY maps and
# returns P(hippocampus) at a coordinate: 0-100, the percentage of subjects in
# whom that voxel was hippocampus. Deepest-in-the-structure therefore scores
# highest, which is what "most hippocampal" should mean.

_hc_prob_img = None
_hc_prob_idx = None


def _load_hc_prob():
    """Harvard-Oxford subcortical probability maps, hippocampus volumes only."""
    global _hc_prob_img, _hc_prob_idx
    if _hc_prob_img is not None:
        return _hc_prob_img, _hc_prob_idx
    from nilearn import datasets as nldatasets
    import nibabel as nib
    atlas = nldatasets.fetch_atlas_harvard_oxford("sub-prob-2mm")
    names = [str(n) for n in atlas.labels]
    img = atlas.maps if hasattr(atlas.maps, "get_fdata") else nib.load(atlas.maps)
    # `labels` carries a leading 'Background' entry that has no volume, so the
    # 4-D index is label_index - 1. Indexing with the label position directly
    # silently reads a neighbouring structure (it returned 0 % everywhere,
    # including at the hippocampal centroid, which is how this was caught).
    n_vol = img.shape[3] if img.ndim == 4 else 1
    off = 1 if len(names) == n_vol + 1 else 0
    idx = [i - off for i, n in enumerate(names) if "hippocampus" in n.lower()]
    idx = [i for i in idx if 0 <= i < n_vol]
    if not idx:
        raise RuntimeError("No hippocampus volume in the HO subcortical "
                           f"probability atlas; labels were {names}")
    _hc_prob_img, _hc_prob_idx = img, idx
    return _hc_prob_img, _hc_prob_idx


_jl_prob_img = None
_jl_prob_idx = None

# Juelich's hippocampal subregions. It separates the cornu ammonis from the
# dentate gyrus and the subiculum, but does NOT separate CA1 from CA2/CA3 --
# "cornu ammonis" is one volume covering all fields. No atlas in this project's
# stack resolves CA1 from CA3, and at 2 mm macro contacts with a lead field of
# several mm it is doubtful that any would. Treat the output as CA / DG / SUB,
# and say so rather than calling the CA volume "CA1".
HC_SUBFIELDS = {
    "CA": "cornu ammonis",
    "DG": "dentate gyrus",
    "SUB": "subiculum",
    "HATA": "hippocampal-amygdaloid transition area",
    "EC_jl": "entorhinal cortex",
}


def _load_hc_subfield_prob():
    """Juelich histological probability maps, hippocampal subregions only."""
    global _jl_prob_img, _jl_prob_idx
    if _jl_prob_img is not None:
        return _jl_prob_img, _jl_prob_idx
    from nilearn import datasets as nldatasets
    import nibabel as nib
    atlas = nldatasets.fetch_atlas_juelich("prob-2mm")
    names = [str(n) for n in atlas.labels]
    img = atlas.maps if hasattr(atlas.maps, "get_fdata") else nib.load(atlas.maps)
    n_vol = img.shape[3] if img.ndim == 4 else 1
    # Same background-offset trap as the Harvard-Oxford loader above: a leading
    # 'Background' entry with no volume makes the 4-D index label_index - 1.
    off = 1 if len(names) == n_vol + 1 else 0
    idx = {}
    for short, token in HC_SUBFIELDS.items():
        hits = [i - off for i, n in enumerate(names)
                if "hippocampus" in n.lower() and token in n.lower()]
        hits = [i for i in hits if 0 <= i < n_vol]
        if hits:
            idx[short] = hits
    if not idx:
        raise RuntimeError("No hippocampal subregion volumes in the Juelich "
                           f"probability atlas; labels were {names[:10]}")
    _jl_prob_img, _jl_prob_idx = img, idx
    return _jl_prob_img, _jl_prob_idx


def hippocampal_subfield_probability(coords):
    """P(subregion) in per cent at each MNI152 coordinate, per Juelich.

    Returns a dict {short_name: (N,) array}. Left and right volumes are
    combined with `max`, as in `hippocampal_probability`: a contact is in one
    hemisphere or the other, never both.

    These are PROBABILITIES, not a classification, and that is deliberate. The
    cornu ammonis volume dominates the hippocampus by size, so a max-prob label
    calls almost every contact "CA" and a hard CA-vs-DG split would be mostly an
    artefact of volume. Carry P(CA) and P(DG) as continuous moderators instead.
    """
    import nibabel as nib
    img, idx = _load_hc_subfield_prob()
    data = img.get_fdata()
    inv = np.linalg.inv(img.affine)
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    out = {k: np.zeros(len(coords)) for k in idx}
    for i, mni in enumerate(coords):
        if np.any(np.isnan(mni)):
            for k in out:
                out[k][i] = np.nan
            continue
        v = np.round(nib.affines.apply_affine(inv, mni)).astype(int)
        if np.any(v < 0) or np.any(v >= np.array(data.shape[:3])):
            continue
        for k, vols in idx.items():
            out[k][i] = float(max(data[v[0], v[1], v[2], j] for j in vols))
    return out


def hippocampal_probability(coords):
    """P(hippocampus) in per cent at each MNI152 coordinate.

    `coords` is (N, 3) in mm. Left and right hippocampus volumes are combined
    with `max`, since a contact is in one or the other. Coordinates outside the
    volume return 0.
    """
    import nibabel as nib
    img, idx = _load_hc_prob()
    data = img.get_fdata()
    inv = np.linalg.inv(img.affine)
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    out = np.zeros(len(coords))
    for i, mni in enumerate(coords):
        if np.any(np.isnan(mni)):
            out[i] = np.nan
            continue
        v = np.round(nib.affines.apply_affine(inv, mni)).astype(int)
        if np.any(v < 0) or np.any(v >= np.array(data.shape[:3])):
            continue
        out[i] = float(max(data[v[0], v[1], v[2], k] for k in idx))
    return out
