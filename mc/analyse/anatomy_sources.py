#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electrode coordinate sources for all three recording sites.

EXTRACTED VERBATIM from scripts/cell_to_roi_july26.py (lines 221-617) so that
the cell pipeline and the LFP/ripple pipeline resolve electrode anatomy through
exactly one implementation. If these rules change, both analyses change
together -- which is the point.

The only edit made during extraction: `discover_utah_mats` took two module
globals of the original script (`path_to_subject_folders`, `path_to_cell_table`)
and now takes them as keyword arguments defaulted to the same literal paths, so
the original bare call `discover_utah_mats()` behaves identically.

Regression gate: after wiring cell_to_roi_july26.py to import from here, that
script must still reproduce derivatives/neurons_with_ROI_labels.csv
byte-for-byte (md5 e0e758a303831cfc614a2490dcaf6aac as of 2026-08-26).

@author: Svenja Kuchenhoff
"""

import os
import re

import numpy as np
import pandas as pd
import scipy.io as sio

try:
    import h5py
    HAVE_H5 = True
except ImportError:
    HAVE_H5 = False


# Default source locations (were module globals in cell_to_roi_july26.py).
# Resolved local-then-ceph rather than hardcoded: the literal Mac paths broke
# every cluster run. Same precedence as swr_io.get_data_root(), kept here as a
# tiny local helper so this module stays importable on its own.
_ROOT_LOCAL = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
_ROOT_CEPH = "/ceph/behrens/svenja/human_ABCD_ephys"


def _default_root():
    return _ROOT_LOCAL if os.path.isdir(_ROOT_LOCAL) else _ROOT_CEPH


DEFAULT_SUBJECT_FOLDERS = _default_root()
DEFAULT_CELL_TABLE = os.path.join(_default_root(), "derivatives",
                                  "neurons_MNI_latest.csv")


COORD_MATCH_TOL_MM = 0.5
BAYLOR_RELIABILITY_TOL_MM = 8.0


# =============================================================================
# MNI305 -> MNI152 (Fischl affine)
# =============================================================================

MNI305_TO_MNI152 = np.array([
    [ 0.9975, -0.0073,  0.0176, -0.0429],
    [ 0.0146,  1.0009, -0.0024,  1.5496],
    [-0.0130, -0.0093,  0.9971,  1.1840],
    [ 0.0000,  0.0000,  0.0000,  1.0000],
])


def mni305_to_mni152(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords[None, :]
    coords_h = np.c_[coords, np.ones(len(coords))]
    return (coords_h @ MNI305_TO_MNI152.T)[:, :3]


# =============================================================================
# BAYLOR v2026 LOADER
# =============================================================================

def _bundle_key(label):
    """mRT2bHaEa04 -> mrt2bhaea (strip trailing 2 digits, lowercase)."""
    if pd.isna(label):
        return None
    return re.sub(r"\d{2}$", "", str(label)).lower()


# Row types in the v2026 CSVs that describe a microwire bundle. Each
# bundle usually appears twice: once as `microwires` (the bundle itself,
# label `mLT2bHb01`) and once as `sEEG-micro` (contact 01 of the macro
# probe carrying it, label `LT2bHb01`, ~3 mm shallower). Preference
# order matters — index 0 wins.
MICRO_TYPES = ["microwires", "sEEG-micro"]


def _micro_bundle_rows(d):
    """Return one row per micro bundle, keyed on the m-prefixed bundle
    name in a new `bundle_key` column. `microwires` rows win over
    `sEEG-micro` rows for the same bundle; `sEEG-micro` only fills in
    bundles that have no `microwires` row at all."""
    micro = d[d["Type"].isin(MICRO_TYPES)].copy()
    if micro.empty:
        return micro
    micro["bundle_key"] = micro["Label"].apply(_bundle_key)
    # sEEG-micro labels lack the leading 'm' — add it so both row types
    # land on the same key as the big table's `electrode label`.
    is_seeg = micro["Type"].eq("sEEG-micro")
    micro.loc[is_seeg, "bundle_key"] = "m" + micro.loc[is_seeg, "bundle_key"]
    micro["_pref"] = micro["Type"].map({t: i for i, t in enumerate(MICRO_TYPES)})
    return (micro.sort_values("_pref")
                 .drop_duplicates("bundle_key", keep="first"))


# Distance the microwire bundle protrudes beyond the deepest macro
# contact, along the probe insertion axis. This is not a per-subject
# measurement: in all 119 bundles across all 19 v2026 files that carry
# `microwires` rows, the offset is exactly 3.15 mm, i.e. Baylor applies a
# nominal Behnke-Fried protrusion rather than localising the wires
# individually. Reconstructing a bundle from its macro probe with this
# constant reproduces Baylor's own `microwires` row to a median of
# 0.25 mm (max 1.07 mm) over those 113 checkable bundles.
MICRO_PROTRUSION_MM = 3.15


def reconstruct_micro_from_macro(d, probe, cols):
    """Rebuild a micro-bundle position from its macro probe alone, for
    subject files that ship no `microwires` / `sEEG-micro` rows at all
    (YER as of 2026-08). Contact 01 is the deepest contact and
    01 -> 02 points back out along the shaft, so the bundle sits at
    `contact01 - MICRO_PROTRUSION_MM * unit(contact02 - contact01)`.

    NB `Label` (zero-padded contact number) is the sort key — the
    `ElectrodeID` column is mixed int/str and sorts lexicographically."""
    pr = d[(d["ProbeName"].astype(str) == probe)
           & (d["Type"].isin(["sEEG", "sEEG-micro"]))].sort_values("Label")
    if len(pr) < 2:
        return None
    p1 = pr.iloc[0][cols].to_numpy(float)
    p2 = pr.iloc[1][cols].to_numpy(float)
    if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
        return None
    v = p2 - p1
    if np.linalg.norm(v) == 0:
        return None
    return p1 - MICRO_PROTRUSION_MM * (v / np.linalg.norm(v))


def _file_152_is_selfconsistent(d):
    """True if the file's MNI152 column agrees with its own MNI305
    column under the Fischl transform (same gate as the micro-bundle
    reliability check, applied to every contact in the file)."""
    chk = d[["MNI305_x", "MNI305_y", "MNI305_z",
             "MNI152_x", "MNI152_y", "MNI152_z"]].dropna()
    if chk.empty:
        return False
    a305 = chk[["MNI305_x", "MNI305_y", "MNI305_z"]].to_numpy(float)
    a152 = chk[["MNI152_x", "MNI152_y", "MNI152_z"]].to_numpy(float)
    return bool(np.mean(np.linalg.norm(a152 - mni305_to_mni152(a305), axis=1))
                <= BAYLOR_RELIABILITY_TOL_MM)


def load_baylor_v2026(folder):
    """Return `(reliable, fallback_305, reconstructed, reliability_df)`.

    * reliable: {(subj_code, bundle_key): (x, y, z)} in MNI152 from
      subjects whose file passed the reliability check.
    * fallback_305: same shape, but coords are the file's MNI305 for
      subjects that failed the check (caller applies 305->152).
    * reconstructed: same shape, in MNI152, for subjects whose file has
      no micro rows at all — rebuilt from the macro probe geometry (see
      `reconstruct_micro_from_macro`). Flagged separately downstream
      because it is inferred, not supplied.
    """
    reliable = {}
    fallback = {}
    reconstructed = {}
    rel_rows = []
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith("-electrodes_v2026.csv"):
            continue
        subj_code = fn.split("-")[0]
        d = pd.read_csv(os.path.join(folder, fn))
        micro = _micro_bundle_rows(d)
        if micro.empty:
            # No micro rows shipped at all. The macro probes are still
            # there, and Baylor's own micro positions are a fixed
            # 3.15 mm extension of them, so rebuild every probe. Use the
            # file's MNI152 when it is self-consistent, else rebuild in
            # MNI305 and transform.
            use_152 = _file_152_is_selfconsistent(d)
            cols = (["MNI152_x", "MNI152_y", "MNI152_z"] if use_152
                    else ["MNI305_x", "MNI305_y", "MNI305_z"])
            n_rec = 0
            for probe in d["ProbeName"].dropna().astype(str).unique():
                xyz = reconstruct_micro_from_macro(d, probe, cols)
                if xyz is None:
                    continue
                if not use_152:
                    xyz = mni305_to_mni152(xyz)[0]
                reconstructed[(subj_code, "m" + probe.lower())] = tuple(xyz)
                n_rec += 1
            rel_rows.append({"subject_code": subj_code, "n_bundles": 0,
                             "n_microwires_rows": 0, "n_seegmicro_rows": 0,
                             "n_reconstructed_probes": n_rec,
                             "mean_mm": np.nan, "max_mm": np.nan,
                             "reliable": False})
            continue

        chk = micro[["MNI305_x", "MNI305_y", "MNI305_z",
                     "MNI152_x", "MNI152_y", "MNI152_z"]].dropna()
        if chk.empty:
            mean_d = max_d = np.nan
            is_rel = False
        else:
            a305 = chk[["MNI305_x", "MNI305_y", "MNI305_z"]].to_numpy(float)
            a152 = chk[["MNI152_x", "MNI152_y", "MNI152_z"]].to_numpy(float)
            recomp = mni305_to_mni152(a305)
            d152 = np.linalg.norm(a152 - recomp, axis=1)
            mean_d = float(np.mean(d152))
            max_d = float(np.max(d152))
            is_rel = mean_d <= BAYLOR_RELIABILITY_TOL_MM

        rel_rows.append({
            "subject_code": subj_code,
            "n_bundles": len(micro),
            "n_microwires_rows": int(micro["Type"].eq("microwires").sum()),
            "n_seegmicro_rows": int(micro["Type"].eq("sEEG-micro").sum()),
            "n_reconstructed_probes": 0,
            "mean_mm": mean_d, "max_mm": max_d, "reliable": is_rel})

        for _, r in micro.iterrows():
            key = (subj_code, r["bundle_key"])
            if is_rel:
                try:
                    x, y, z = float(r["MNI152_x"]), float(r["MNI152_y"]), float(r["MNI152_z"])
                except (TypeError, ValueError):
                    continue
                if any(np.isnan(v) for v in (x, y, z)):
                    continue
                reliable[key] = (x, y, z)
            else:
                try:
                    x, y, z = float(r["MNI305_x"]), float(r["MNI305_y"]), float(r["MNI305_z"])
                except (TypeError, ValueError):
                    continue
                if any(np.isnan(v) for v in (x, y, z)):
                    continue
                fallback[key] = (x, y, z)

    return reliable, fallback, reconstructed, pd.DataFrame(rel_rows)


# =============================================================================
# UCLA v2026 xlsx LOADER
# =============================================================================

# All 6 UCLA subjects now have a v2026 xlsx (UC3-0559 moved in by the user).
UCLA_SUBJECT_TO_FILE = {
    "UC3-0559": "sub-559",
    "UC3-0573": "sub-573",
    "UC2-0576": "sub-576",
    "UC3-0577": "sub-577",
    "UC2-0578": "sub-578",
    "UC3-0582": "sub-582",
}


def load_ucla_v2026(folder):
    """Return `{subject_label: DataFrame}` where each df has columns
    `electrode`, `region_hint`, `MNI_x`, `MNI_y`, `MNI_z`. Used to
    coord-match big-table cells (no label-based rules)."""
    tables = {}
    for subj, prefix in UCLA_SUBJECT_TO_FILE.items():
        fpath = os.path.join(folder, f"{prefix}_localizations.xlsx")
        if not os.path.exists(fpath):
            continue
        d = pd.read_excel(fpath, sheet_name="Sheet1")
        # Sheet1 is the second sheet and the only one carrying MNI coordinates.
        # Cells are microwire recordings, so macro contacts are excluded: a
        # micro cell must not be able to match a macro row.
        if "isMicro" in d.columns:
            d = d[d["isMicro"].astype(str).str.upper().isin(["TRUE", "1", "1.0"])]
        d = d[["electrode", "MNI_x", "MNI_y", "MNI_z", "NMM"]].copy()
        d = d.dropna(subset=["MNI_x", "MNI_y", "MNI_z"]).reset_index(drop=True)
        d = d.rename(columns={"NMM": "region_hint"})
        tables[subj] = d
    return tables


# =============================================================================
# UTAH .mat LOADER
# =============================================================================


def _obj_array(items):
    """1-D object array built element-by-element.

    `np.array(list_of_arrays, dtype=object)` tries to broadcast when the
    elements share a leading dimension; filling an empty object array
    sidesteps that.
    """
    out = np.empty(len(items), dtype=object)
    for i, v in enumerate(items):
        out[i] = v
    return out


def _h5_deref(f, val):
    """Resolve one h5py value into a plain python object.

    MATLAB v7.3 stores a cell array of strings as a dataset of HDF5 object
    references; each reference points at a `char` dataset of uint16 code
    points. Reading such a dataset with a plain `np.array(...)` yields
    reference handles, not text — which is why `LabelMap` came back as a
    bare `None` for every v7.3 file and those subjects silently produced
    zero microwires. Dereference explicitly instead.
    """
    if not isinstance(val, h5py.Reference):
        return val
    try:
        d = f[val]
    except Exception:
        return None
    cls = d.attrs.get("MATLAB_class", b"")
    if isinstance(cls, bytes):
        cls = cls.decode("utf-8", "ignore")
    if d.attrs.get("MATLAB_empty", 0):
        return None
    a = np.array(d)
    if cls == "char":
        return "".join(chr(int(c)) for c in a.ravel() if int(c) > 0)
    if cls == "cell":
        return _obj_array([_h5_deref(f, r) for r in a.ravel()])
    if a.size == 0:
        return None
    if a.size == 1:
        return a.ravel()[0]
    return a


def _load_mat(fpath):
    """Load .mat as a plain dict, supporting both v7 (scipy) and v7.3 (h5)."""
    try:
        return sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        if not HAVE_H5:
            return None
        try:
            out = {}
            with h5py.File(fpath, "r") as f:
                for k in f.keys():
                    d = f[k]
                    arr = np.array(d)
                    if arr.dtype == object:
                        # cell array -> dereference every element
                        flat = _obj_array([_h5_deref(f, r)
                                           for r in arr.ravel()])
                        arr = flat.reshape(arr.shape)
                    # h5py returns column-major; transpose 2D arrays.
                    if arr.ndim == 2:
                        arr = arr.T
                    out[k] = arr
            return out
        except Exception as e:
            print(f"  [mat load] h5 failed for {fpath}: {e}")
            return None
    except Exception as e:
        print(f"  [mat load] scipy failed for {fpath}: {e}")
        return None


def mat_text(val):
    """Decode a MATLAB char field (v7 str, or v7.3 uint16 code points)."""
    a = np.asarray(val)
    if a.dtype.kind in "US":
        return str(a.ravel()[0]) if a.size else ""
    try:
        return "".join(chr(int(c)) for c in a.ravel() if int(c) > 0)
    except Exception:
        return ""


def mat_patient_id(mat):
    """Patient ID this electrode file actually belongs to, read from its own
    `Fname` (the original acquisition path, e.g. `D:\\Data\\UIC202311\\...`).

    This is the file's own statement of identity, so it replaces coord-matching
    the file against a hand-entered coordinate table -- which was circular, had
    no uniqueness constraint, and assigned one patient's file to six others.
    Returns None if `Fname` is absent or carries no recognisable ID.
    """
    if mat is None or "Fname" not in mat:
        return None
    m = re.search(r"(?:UIC|UT1?[-_]?)(\d{6}[a-z]?)", mat_text(mat["Fname"]), re.I)
    return m.group(1) if m else None


def build_micro_label_map(mat):
    """Return `{micro_label: (mni_xyz, source_field)}` read DIRECTLY.

    `MicroElecRaw` holds 1-based row indices into `ElecMapRaw` (column 0 is the
    electrode label) and `ElecXYZMNIRaw` (the coordinate). Label and coordinate
    are therefore taken from the *same row index* -- there is no ordering
    assumption, unlike `build_micro_map`, which infers the pairing by sorting
    microwires by amplifier channel value and matching that against `MicroElec`.

    Every Utah file carries these three arrays, including the ones whose
    `MicroElec` is empty and whose `LabelMap`/`ChannelMap` are absent
    altogether (the MATLAB v7.3 exports). Validated on all 15 files:
    352/360 microwires have `sign(MNI_x)` consistent with the `mL`/`mR` in
    their own label; the 8 exceptions are OFC contacts within 2.5 mm of the
    midline, where the sign carries no information.
    """
    if mat is None:
        return {}
    for k in ("MicroElecRaw", "ElecMapRaw", "ElecXYZMNIRaw"):
        if k not in mat:
            return {}
    idx = np.asarray(mat["MicroElecRaw"]).ravel()
    emap = np.asarray(mat["ElecMapRaw"], dtype=object)
    xyz = np.asarray(mat["ElecXYZMNIRaw"], dtype=float)
    if emap.ndim != 2 or xyz.ndim != 2 or len(emap) != len(xyz):
        return {}
    out = {}
    for i in idx:
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        if not (1 <= i <= len(emap)):
            continue
        label = str(emap[i - 1, 0]).strip()
        coord = xyz[i - 1]
        if label and not np.any(np.isnan(coord)):
            out[label] = (coord, "ElecXYZMNIRaw")
    return out


# NB: macOS is case-insensitive, so listing both "Registered" and "registered"
# would scan the same directory twice.
# Utah exports are not laid out consistently.  Most registered files sit at
# sXX/Registered, while s39 keeps the complete set one level deeper at
# s39/electrodes/Registered.  Scan both layouts; identity is still established
# from each MAT file's own Fname, never from its directory name.
UTAH_SEARCH_SUBDIRS = (
    "", "electrodes", "Registered-selected", "Registered",
    os.path.join("electrodes", "Registered-selected"),
    os.path.join("electrodes", "Registered"),
)
UTAH_MAT_NAMES = ("Electrodes.mat", "ChannelMap.mat", "ChannelMap2.mat")


def subject_numeric_id(label):
    """Numeric patient ID from a subject label, e.g. `UT1-202422b` -> `202422`.

    Utah labels carry four formats and sometimes a trailing session letter that
    the electrode file's own `Fname` does not (`UT1-202422b` vs `UIC202422`), so
    the letter is dropped for matching.
    """
    m = re.search(r"(\d{6})", str(label))
    return m.group(1) if m else None


def index_utah_files_by_id(path_to_subject_folders=DEFAULT_SUBJECT_FOLDERS,
                           verbose=False):
    """Index every Utah electrode file by the patient ID it declares itself.

    Returns `{patient_id: (location, label_map, chan_map)}` where `label_map`
    is `build_micro_label_map` and `chan_map` is `build_micro_map`, merged
    across `Electrodes.mat` and its sibling `ChannelMap*.mat` (the MATLAB v7.3
    exports keep the `chanN` bridge only in the latter).

    Every candidate location is scanned, not just the first that contains a
    file: one session folder can hold two different patients' files (s47 has
    202302 at its top level and 202311 under `electrodes/`). Identity comes
    from each file's own `Fname`, so a misfiled copy cannot be mis-assigned --
    which is what coord-matching against the hand-entered cell table could not
    prevent.
    """
    by_id = {}
    for folder in sorted(os.listdir(path_to_subject_folders)):
        if not (folder.startswith("s") and folder[1:].isdigit()):
            continue
        for sd in UTAH_SEARCH_SUBDIRS:
            d = (os.path.join(path_to_subject_folders, folder, sd) if sd
                 else os.path.join(path_to_subject_folders, folder))
            if not os.path.isdir(d):
                continue
            group = {}                      # pid -> list of mats found here
            for name in UTAH_MAT_NAMES:
                fp = os.path.join(d, name)
                if not os.path.isfile(fp):
                    continue
                mat = _load_mat(fp)
                if mat is None:
                    continue
                pid = mat_patient_id(mat)
                if pid is None:
                    continue
                group.setdefault(pid, []).append(mat)
            for pid, mats in group.items():
                lmap, cmap = {}, {}
                for mat in mats:
                    lmap.update(build_micro_label_map(mat))
                    cmap.update(build_micro_map(mat))
                if not lmap and not cmap:
                    continue
                loc = f"{folder}/{sd}" if sd else folder
                # merge across every location holding this patient's files:
                # s04 keeps a bare ChannelMap.mat at its top level and the
                # microwire coordinates one level down, and neither alone is
                # sufficient.
                if pid in by_id:
                    old_loc, old_l, old_c = by_id[pid]
                    old_l.update(lmap)
                    old_c.update(cmap)
                    by_id[pid] = (old_loc if old_l else loc, old_l, old_c)
                else:
                    by_id[pid] = (loc, lmap, cmap)
                if verbose:
                    print(f"    {pid} -> {loc}  ({len(lmap)} microwires)")
    return by_id


def index_utah_mats_by_id(path_to_subject_folders=DEFAULT_SUBJECT_FOLDERS):
    """`{patient_id: (location, mat_dict)}` for Utah `Electrodes.mat` files.

    Same identity rule as `index_utah_files_by_id` -- each file is filed under
    the patient it names in its own `Fname` -- but returns the loaded mat, for
    callers that need the macro arrays rather than the microwire maps.
    """
    # Keys `load_utah_macros` needs. A folder can hold several files that all
    # declare the same patient -- s04 keeps a bare ChannelMap.mat at its top
    # level and the macro arrays one level down -- so score candidates and keep
    # the most complete one rather than the first one found.
    MACRO_KEYS = ("ElecMapRaw", "ElecXYZMNIRaw", "ElecTypeRaw", "ElecAtlasRaw")

    best = {}
    for folder in sorted(os.listdir(path_to_subject_folders)):
        if not (folder.startswith("s") and folder[1:].isdigit()):
            continue
        for sd in UTAH_SEARCH_SUBDIRS:
            d = (os.path.join(path_to_subject_folders, folder, sd) if sd
                 else os.path.join(path_to_subject_folders, folder))
            if not os.path.isdir(d):
                continue
            for name in ("Electrodes.mat", "ChannelMap.mat"):
                fp = os.path.join(d, name)
                if not os.path.isfile(fp):
                    continue
                mat = _load_mat(fp)
                if mat is None:
                    continue
                pid = mat_patient_id(mat)
                if not pid:
                    continue
                score = sum(k in mat for k in MACRO_KEYS)
                loc = f"{folder}/{sd}/{name}" if sd else f"{folder}/{name}"
                # A session folder is supposed to hold ONE patient's files. When
                # two locations declare the same patient it is usually a misfiled
                # copy (ceph had 202311 in both s47/electrodes/ and s48/electrodes/,
                # so s48 was served from s47's folder). Prefer the file whose own
                # folder number matches the session that patient was recorded in,
                # and say so, rather than silently taking whichever sorts first.
                if pid in best:
                    prev_score, prev_loc, prev_mat = best[pid]
                    # companion files in the SAME session folder are normal
                    # (Electrodes.mat + ChannelMap.mat); only a copy under a
                    # DIFFERENT session folder means a file has been misfiled.
                    if prev_loc.split("/")[0] != folder:
                        print(f"  [MISFILED] patient {pid} declared by BOTH "
                              f"{prev_loc} and {loc} -- a session folder should "
                              f"hold one patient; remove the copy that does not "
                              f"belong to its folder")
                    if score <= prev_score:
                        continue
                best[pid] = (score, loc, mat)
    return {pid: (loc, mat) for pid, (score, loc, mat) in best.items()
            if score > 0}


def utah_micro_coord(patient_id, electrode_label, utah_index):
    """Coordinate + provenance for one Utah cell, read straight from the file.

    `(xyz, source)` or `(None, "")`. The big-table `electrode label` is either
    the microwire label itself (`mLHIP4`) or an amplifier channel (`chan113`);
    the latter is bridged `ChannelMap1/2 -> LabelMap -> label`, and the
    coordinate is then taken by that label so that label and coordinate always
    come from the same row of the same file.
    """
    entry = utah_index.get(patient_id)
    if entry is None:
        return None, ""
    loc, lmap, cmap = entry
    lab = str(electrode_label).strip()
    if lab in lmap:
        return lmap[lab][0], f"{loc}:{lab}"
    m = re.match(r"^chan(\d+)$", lab, re.IGNORECASE)
    if m and int(m.group(1)) in cmap:
        micro_label = str(cmap[int(m.group(1))][1])
        if micro_label in lmap:
            return lmap[micro_label][0], f"{loc}:{micro_label}"
    return None, ""


def build_micro_map(mat):
    """Build {chan_value: (mni_xyz, micro_label)} for microwires.

    Two attempts:
      (A) MicroElec + ElecXYZMNIProj (gray-matter-projected).
      (B) MicroElecRaw + ElecXYZMNIRaw (raw electrode positions, used
          when MicroElec is empty — happens for some subjects, e.g. s23).

    In each attempt, LabelMap entries starting with 'm' mark
    microwires; each such (r, c) carries an amplifier chan value in
    ChannelMap1 or ChannelMap2. Sort micros by chan-value ascending —
    that ordering matches the sorted MicroElec[i] entries (validated on
    s02, s06). Return a map keyed by chan_value so we can look up any
    big-table `chan{N}` directly."""
    lm = mat.get("LabelMap")
    cm1 = mat.get("ChannelMap1")
    cm2 = mat.get("ChannelMap2")
    if lm is None:
        return {}

    entries = []  # (chan, r, c, label)
    for r in range(lm.shape[0]):
        for c in range(lm.shape[1]):
            lab = lm[r, c]
            if not (isinstance(lab, str) and lab.startswith("m")):
                continue
            chan = None
            for cm in (cm1, cm2):
                if cm is None:
                    continue
                v = cm[r, c]
                if v is not None and not np.isnan(v) and v > 0:
                    chan = int(v)
                    break
            if chan is not None:
                entries.append((chan, r, c, lab))
    entries.sort(key=lambda x: x[0])
    if not entries:
        return {}

    for micro_key, coord_key in (("MicroElec", "ElecXYZMNIProj"),
                                 ("MicroElecRaw", "ElecXYZMNIRaw")):
        micro = np.atleast_1d(mat.get(micro_key, np.array([]))).astype(int).ravel()
        coords = mat.get(coord_key)
        if not len(micro) or coords is None or len(coords) == 0:
            continue
        if len(entries) != len(micro):
            continue  # try the next attempt
        coords = np.asarray(coords, dtype=float)
        out = {}
        for i, (chan, r, c, lab) in enumerate(entries):
            row = int(micro[i]) - 1
            if 0 <= row < len(coords):
                out[chan] = (coords[row], lab, coord_key)
        if out:
            return out
    return {}


def channel_position(mat, chan):
    """Find (r, c) where `ChannelMap1[r, c] == chan`, else fall back to
    ChannelMap2. Returns None if not found."""
    for key in ("ChannelMap1", "ChannelMap2"):
        cm = mat.get(key)
        if cm is None:
            continue
        pos = np.argwhere(cm == chan)
        if len(pos):
            return int(pos[0][0]), int(pos[0][1])
    return None


def discover_utah_mats(path_to_subject_folders=DEFAULT_SUBJECT_FOLDERS,
                       path_to_cell_table=DEFAULT_CELL_TABLE):
    """For each Utah subject, find which `s{NN}/electrodes/*.mat` file
    corresponds to it, by coord-matching: pool every folder's
    `ElecXYZMNIProj` coords and pick the folder that covers
    >= 50% of the subject's big-table cells at <= 0.5 mm. Returns
    {subject_label: (folder_name, mat_dict)} — the whole mat dict is
    kept so the independent-reconstruction step below can use it."""
    folder_mats = {}
    for f in sorted(os.listdir(path_to_subject_folders)):
        if not (f.startswith("s") and f[1:].isdigit()):
            continue
        edir = os.path.join(path_to_subject_folders, f, "electrodes")
        if not os.path.isdir(edir):
            continue
        # Prefer Electrodes.mat; fall back to ChannelMap.mat.
        chosen = None
        for name in ("Electrodes.mat", "ChannelMap.mat"):
            p = os.path.join(edir, name)
            if os.path.exists(p):
                chosen = p
                break
        if not chosen:
            continue
        mat = _load_mat(chosen)
        if mat is None or "ElecXYZMNIProj" not in mat:
            continue
        folder_mats[f] = (chosen, mat)

    bt = pd.read_csv(path_to_cell_table)
    bt = bt[bt["Recording Site"].astype(str).str.lower() == "utah"]
    mapping = {}
    for subj_label, grp in bt.groupby("Subject Label"):
        pts = grp[["MNI_x", "MNI_y", "MNI_z"]].dropna().to_numpy(float)
        if not len(pts):
            continue
        best = None
        for folder, (fpath, mat) in folder_mats.items():
            # Big-table coords may match Proj OR Raw depending on subject;
            # pool both for the folder-picking step (reconstruction below
            # always uses Proj).
            pool = []
            for k in ("ElecXYZMNIProj", "ElecXYZMNIRaw"):
                if k in mat:
                    a = np.asarray(mat[k], dtype=float)
                    if a.ndim == 2 and a.shape[1] == 3:
                        pool.append(a)
            if not pool:
                continue
            coords = np.vstack(pool)
            n_hit = 0
            for p in pts:
                d = np.linalg.norm(coords - p, axis=1)
                if np.any(~np.isnan(d)) and np.nanmin(d) <= COORD_MATCH_TOL_MM:
                    n_hit += 1
            if best is None or n_hit > best[1]:
                best = (folder, n_hit)
        if best and best[1] >= len(pts) * 0.5:
            mapping[subj_label.strip("'\" ")] = (
                best[0], folder_mats[best[0]][1])
    return mapping

