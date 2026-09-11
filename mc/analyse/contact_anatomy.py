#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anatomy for LFP MACRO contacts, across all three recording sites.

Produces one row per LFP channel with an MNI152 coordinate, grey/white, a
native-atlas region label and an ROI assigned by the SAME rule ladder the cell
pipeline uses (`mc.analyse.anatomy_atlas.assign_atlas_roi`). Then pairs contacts
into bipolar derivations.

This replaces contact selection by string matching. The old code used
`'H' in channel and 'T' in channel` (matches any label containing both letters)
and `'1' in label` (matches HIP10, HIP11, HIP14), plus a hard-coded
contact01-contact04 bipolar pair spanning ~15 mm.

Site specifics, all verified against the data rather than assumed
-----------------------------------------------------------------
Baylor
    `channels.npy` entries look like `RF1bIb01-001`. Strip the trailing
    `-\\d+` and the remainder matches the v2026 CSV `Label` exactly. The
    trailing number is a Blackrock electrode ID, NOT the array position
    (s05 runs to 256 for 156 channels), so data is always indexed by
    position in `channels.npy`. Macro rows are `Type == 'sEEG'`; they
    carry MNI152 directly.

Utah
    `Electrodes.mat` -> `ElecMapRaw`, an (n, 3) object array of
    `[label, amplifier_channel, other_channel]`.

    IMPORTANT: coordinates come from the **direct row index** into
    `ElecXYZMNIRaw` / `ElecAtlasRaw` / `ElecTypeRaw`, not from indexing by
    the channel number. Verified on s02: row 41 is `bRAHIP2` and
    `ElecAtlasRaw[41]` is "Right Hippocampus", whereas indexing by column 1
    gives "Right fusiform gyrus" -- a different electrode entirely, and it
    would not raise. Column 1 is the amplifier channel; it matches
    `utah_elec_labels_{XX}.csv` for all 114 rows where it is finite (the
    other 30 are unlocalised and are dropped with a logged reason).

    Macros are labels NOT starting with 'm'. Naming: `m*` = microwire,
    `b*` = the Behnke-Fried macro on the same shaft as those micros,
    bare = plain sEEG depth contact.

    We use `ElecXYZMNIRaw`, whereas cell_to_roi_july26.py uses
    `ElecXYZMNIProj` for micros. Gray-matter projection is right for a
    spiking micro tip and wrong for a 2 mm macro ring that legitimately
    straddles the GM/WM boundary. Both are stored so the choice is
    reversible.

UCLA
    `sub-NNN_localizations.xlsx` Sheet1, rows with `isMicro == False`.
    Carries `grayWhite`, `MNI_x/y/z`, `NMM`, `AnatMacro_1`, and
    `ASHS_ABC` hippocampal subfields -- the last is independent ground
    truth for validating the atlas ROI assignment. `electrode` is
    `LMH-1`; the .ncs file is `LMH1.ncs`, i.e. drop the hyphen.

@author: Svenja Kuchenhoff
"""

import os
import re

import numpy as np
import pandas as pd

import mc.analyse.anatomy_sources as anat_src
import mc.analyse.anatomy_atlas as anat_atlas


HPC_ROIS = ("HC_anterior", "HC_mid")


# =============================================================================
# NATIVE-SPACE ROI  (primary criterion for macro contacts)
# =============================================================================
#
# A contact's ROI is taken from the SUBJECT'S OWN segmentation, not from the
# group-space atlas. Each site supplies one: Baylor `Area_fs_vox` (FreeSurfer
# aseg), Utah `ElecAtlasRaw` NMM, UCLA `NMM` (+ `ASHS_ABC` subfields).
#
# Why native rather than the shared atlas ladder: measured on s05/s02/s03, the
# atlas rules have 100% recall but only 22% precision against native
# segmentation -- they label 49 contacts hippocampal where the subjects' own
# MRIs say 11, the excess being amygdala, lingual gyrus, VentralDC and
# parahippocampal cortex. That is expected: the ladder was tuned for microwire
# TIPS (where +-2-3 mm neighbourhood rescue is desirable) and is applied here to
# MNI152 coordinates, where MTL registration error is large. Chen et al.
# identified hippocampal contacts "via visual inspection of postoperative
# T1-weighted anatomical MRI scans" -- i.e. in native space.
#
# `atlas_roi` is still computed and stored as a cross-check. Because the atlas
# ladder is a strict superset, disagreement in the HC direction is impossible;
# `roi_concordant` therefore flags only the reverse case.

_NATIVE_PATTERNS = [
    ("PHC",      ("parahippocamp",)),          # must precede the HC test
    ("HC",       ("hippocamp",)),
    ("Amygdala", ("amygdala",)),
    ("EC",       ("entorhinal", "ent ")),
    ("Insula",   ("insula",)),
    ("Thalamus", ("thalamus",)),
    ("WM",       ("cerebral white matter", "white matter")),
]


# Label parsing (probe + contact number). This is string handling, but it is
# NOT anatomical inference -- it only splits `RT2bHa04` into probe `RT2bHa` and
# contact 4 so contacts can be grouped by electrode. Where a contact *is* is
# decided from its MNI coordinate alone.
_CONTACT_RE = re.compile(r'^(?P<probe>.*?)[-_]?(?P<num>\d+)$')
_NS_SUFFIX_RE = re.compile(r'-\d+$')


def split_contact(label):
    """'LT2HbE02' -> ('LT2HbE', 2); 'LMH-1' -> ('LMH', 1). (None, nan) if
    the label carries no trailing contact number."""
    m = _CONTACT_RE.match(str(label).strip())
    if not m:
        return None, np.nan
    return m.group('probe'), int(m.group('num'))


def _hemisphere(label, fallback=None):
    lab = str(label).lstrip('mb').upper()
    if lab.startswith('L'):
        return 'L'
    if lab.startswith('R'):
        return 'R'
    return fallback


# =============================================================================
# PER-SITE MACRO LOADERS
# =============================================================================

def load_baylor_macros(v2026_folder):
    """{subject_code: DataFrame of sEEG (macro) contacts}.

    Applies the same MNI152 self-consistency gate as the micro loader: if a
    file's MNI152 column disagrees with its own MNI305 under the Fischl
    transform, use MNI305 transformed instead and flag it.
    """
    out = {}
    for fn in sorted(os.listdir(v2026_folder)):
        if not fn.endswith("-electrodes_v2026.csv"):
            continue
        code = fn.split("-")[0]
        d = pd.read_csv(os.path.join(v2026_folder, fn))
        macro = d[d["Type"].astype(str) == "sEEG"].copy()
        if macro.empty:
            continue

        use_152 = anat_src._file_152_is_selfconsistent(d)
        if use_152:
            xyz = macro[["MNI152_x", "MNI152_y", "MNI152_z"]].to_numpy(float)
            src = "baylor_v2026_macro_mni152"
        else:
            a305 = macro[["MNI305_x", "MNI305_y", "MNI305_z"]].to_numpy(float)
            xyz = np.full_like(a305, np.nan)
            ok = ~np.any(np.isnan(a305), axis=1)
            if ok.any():
                xyz[ok] = anat_src.mni305_to_mni152(a305[ok])
            src = "baylor_v2026_macro_305to152"

        # Native region uses the 3 mm-neighbourhood parcellation
        # (ROI_DK2005_3mm / Matter_3mm), NOT the single-voxel Area_fs_vox.
        # A macro contact is a ~2 mm ring recording from a volume, so the
        # label should be sampled at that scale. Measured on YEJ: the voxel
        # column calls 1 contact hippocampal, the 3 mm column calls 4 -- e.g.
        # RT2bHaEa02 is "Right-Hippocampus" at 3 mm and
        # "Right-Cerebral-White-Matter" at the voxel. The voxel columns are
        # kept alongside so the choice is reversible.
        res = pd.DataFrame({
            "anat_label": macro["Label"].astype(str).str.strip(),
            "mni_x": xyz[:, 0], "mni_y": xyz[:, 1], "mni_z": xyz[:, 2],
            "matter": macro.get("Matter_3mm"),
            "native_region": macro.get("ROI_DK2005_3mm"),
            "native_region_vox": macro.get("Area_fs_vox"),
            "matter_vox": macro.get("Matter_fs_vox"),
            "probe_src": macro.get("ProbeName"),
            "hemisphere_src": macro.get("Hemisphere"),
            "coord_source": src,
        }).reset_index(drop=True)
        out[code] = res
    return out


def load_utah_macros(mat):
    """Macro contacts from an Electrodes.mat. See the module docstring for
    why coordinates use the direct row index and not the channel number."""
    em = np.asarray(mat.get("ElecMapRaw"), dtype=object)
    if em is None or em.ndim != 2 or em.shape[1] < 2:
        return pd.DataFrame()

    xyz_raw = np.asarray(mat.get("ElecXYZMNIRaw"), dtype=float)
    xyz_proj = mat.get("ElecXYZMNIProjRaw", mat.get("ElecXYZMNIProj"))
    xyz_proj = (np.asarray(xyz_proj, dtype=float)
                if xyz_proj is not None else None)
    matter = np.atleast_1d(mat.get("ElecTypeRaw"))
    atlas = mat.get("ElecAtlasRaw")
    atlas = np.asarray(atlas, dtype=object) if atlas is not None else None
    atlas_names = [str(a) for a in np.atleast_1d(mat.get("AtlasNames", []))]
    nmm_col = atlas_names.index("NMM") if "NMM" in atlas_names else 0

    rows = []
    for i in range(len(em)):
        label = str(em[i][0]).strip()
        if label.startswith('m'):
            continue                              # microwire, not an LFP macro
        chan = em[i][1]
        try:
            chan = int(chan) if np.isfinite(float(chan)) else None
        except (TypeError, ValueError):
            chan = None

        c = xyz_raw[i] if i < len(xyz_raw) else np.full(3, np.nan)
        cp = (xyz_proj[i] if xyz_proj is not None and i < len(xyz_proj)
              else np.full(3, np.nan))
        rows.append({
            "anat_label": label,
            "utah_chan": chan,
            "mni_x": c[0], "mni_y": c[1], "mni_z": c[2],
            "mni_proj_x": cp[0], "mni_proj_y": cp[1], "mni_proj_z": cp[2],
            "matter": (str(matter[i]) if i < len(matter) else None),
            "native_region": (str(atlas[i][nmm_col])
                              if atlas is not None and i < len(atlas) else None),
            "coord_source": "utah_elecmapraw_mni_raw",
        })
    return pd.DataFrame(rows)


def load_ucla_macros(v2026_folder):
    """{subject_label: DataFrame of macro contacts} from the v2026 xlsx."""
    out = {}
    for subj, prefix in anat_src.UCLA_SUBJECT_TO_FILE.items():
        fpath = os.path.join(v2026_folder, f"{prefix}_localizations.xlsx")
        if not os.path.exists(fpath):
            continue
        d = pd.read_excel(fpath, sheet_name="Sheet1")
        if "isMicro" in d.columns:
            d = d[d["isMicro"] == False]                          # noqa: E712
        d = d.dropna(subset=["MNI_x", "MNI_y", "MNI_z"])
        res = pd.DataFrame({
            "anat_label": d["electrode"].astype(str).str.strip(),
            "ncs_stem": d["electrode"].astype(str).str.replace("-", "", regex=False),
            "mni_x": d["MNI_x"].to_numpy(float),
            "mni_y": d["MNI_y"].to_numpy(float),
            "mni_z": d["MNI_z"].to_numpy(float),
            "matter": d.get("grayWhite"),
            "native_region": d.get("NMM"),
            "ashs_subfield": d.get("ASHS_ABC"),
            "coord_source": "ucla_v2026_xlsx",
        }).reset_index(drop=True)
        out[subj] = res
    return out


# UCLA amplifier channel numbers are offset by one 128-channel bank: the micros
# occupy 1-128 and the macros 129-256, which is why s50/s51's .ns3 reports
# `chan129 ... chan256`. The montage JSON numbers its macro channels from 1, so
# macro channel c is recorded as `chan{MACRO_BANK_OFFSET + c}`. Verified against
# both patients: every mapped name exists in the recording, and the declared
# macro count (77 for p576, 95 for p578) fits inside the 128-channel bank.
MACRO_BANK_OFFSET = 128


def load_ucla_montage(v2026_folder, offset=MACRO_BANK_OFFSET):
    """{subject_label: DataFrame[ns_label, anat_label, ncs_stem]} from montage JSON.

    s50 and s51 are recorded on Blackrock, whose header carries no electrode
    names -- every channel is `chan129 ... chan256`. Without a key from those
    numbers to the localised contacts both sessions were unusable, since ROI is
    assigned from the MNI coordinate and there was no way to reach it.

    The montage file supplies exactly that key. `macroChannels` is a list of
    `[probe_label, first_channel, last_channel]`, contiguous and in recording
    order, so channel `c` on probe P is contact `c - first + 1` of P:

        macroChannels: [["RAI", 1, 7], ["RFO", 8, 13], ["RPI", 14, 21], ...]
        -> chan129 = RAI-1, chan135 = RAI-7, chan136 = RFO-1, ...

    Contact 1 is the deepest: |MNI_x| increases monotonically along every probe
    at 5-6 mm spacing (LMH-1..7 runs 29.8 -> 64.7 mm). The localisation tables
    list contacts 1..n with no gaps and fewer than the montage declares, so the
    contacts that were never localised are the OUTERMOST ones; they simply find
    no anatomy row and are reported unresolved. Nothing is renumbered.

    `ncs_stem` is built with the same dash-stripping `load_ucla_macros` applies,
    so `LOF-AC` contact 1 becomes `LOFAC1` on both sides of the join.
    """
    import glob
    import json as _json
    out = {}
    for subj, prefix in anat_src.UCLA_SUBJECT_TO_FILE.items():
        pat = prefix.split("-")[-1]                       # 'sub-576' -> '576'
        hits = sorted(glob.glob(os.path.join(v2026_folder,
                                            f"montage_Patient-{pat}_*.json")))
        if not hits:
            continue
        with open(hits[0]) as f:
            mont = _json.load(f)
        rows = []
        for entry in mont.get("macroChannels", []):
            if len(entry) < 3:
                continue
            label, first, last = str(entry[0]).strip(), int(entry[1]), int(entry[2])
            for c in range(first, last + 1):
                k = c - first + 1
                rows.append({"ns_label": f"chan{offset + c}",
                             "anat_label": f"{label}-{k}",
                             "ncs_stem": f"{label}{k}".replace("-", "")})
        if rows:
            out[subj] = pd.DataFrame(rows)
    return out


# =============================================================================
# JOIN ANATOMY TO ACTUAL LFP CHANNELS
# =============================================================================

def load_channel_list(session, data_root, verbose=False):
    """Channel names in LFP-array order, and where they came from.

    Prefers the cached `channels.npy`, falling back to the raw file header so
    contact anatomy can be built for every session with raw data, not only the
    handful the old preprocessing ever ran on.

    A cache whose names are ALL placeholders is treated as absent: s16 stores
    exactly ['empty-064','empty-128','empty-192','empty-256'], which shadowed a
    232-channel recording and cost the whole session. Returns
    `(names, source, path)`.
    """
    import mc.analyse.swr_io as swr_io

    lfp_dir = os.path.join(swr_io.session_deriv_dir(session, data_root), "LFP")
    cached_path = os.path.join(lfp_dir, "channels.npy")
    if os.path.isfile(cached_path):
        cached = [str(c) for c in np.load(cached_path, allow_pickle=True)]
        if cached and not all(str(c).lower().startswith("empty") for c in cached):
            return cached, "channels.npy", cached_path
        if verbose:
            print(f"    s{int(session):02d}: ignoring degenerate channels.npy "
                  f"({len(cached)} names, all placeholders) -- reading the raw header")

    cfg_s = swr_io.session_config(session, data_root=data_root)
    files, kind, _ = swr_io.discover_raw_files(session, cfg_s, data_root=data_root)
    if not files:
        return [], "none", ""
    if kind == "neuralynx":
        return ([os.path.splitext(os.path.basename(f))[0] for f in files[0]],
                "ncs filenames", os.path.dirname(files[0][0]))
    try:
        import mc.analyse.swr_preproc as pre
        # Share the extraction reader rather than reimplementing it here.  It
        # trusts the stream actually present on disk (important for s39: YAML
        # says ns2, recovered file is .ns6) and tolerates a companion .nev whose
        # segment structure disagrees with the NSx file.
        ext = os.path.splitext(files[0])[1].lstrip(".")
        nsx = pre._nsx_number(files[0], cfg_s)
        reader = pre._blackrock_reader(files[0], nsx)
        names = [str(e) for e in reader.header["signal_channels"]]
        return ([n.split(",")[0].strip("('") for n in names],
                f"raw header ({ext})", files[0])
    except Exception as e:
        if verbose:
            print(f"    s{int(session):02d}: could not read channel names from raw: "
                  f"{type(e).__name__}: {e}")
        return [], "none", ""


def build_macro_table(session, recording_site, subject_label, channels,
                      baylor_macros=None, utah_mat=None, ucla_macros=None,
                      ucla_montage=None):
    """One row per LFP channel in `channels` (the `channels.npy` order).

    Never drops a channel: unmatched ones come back with `resolved=False`
    and an `unresolved_reason`, so the QC report can account for every
    channel in the recording rather than silently losing some.

    `ns_pos` is the 0-based column index into the LFP array. For Baylor the
    trailing `-NNN` in a channel name is a Blackrock electrode ID, not the
    array position, so it must never be used to index data.
    """
    channels = [str(c).strip() for c in channels]
    site = str(recording_site).lower()
    subject_key = re.sub(r'^BY\d*[-_]?', '', str(subject_label).strip().strip("'"))

    base = pd.DataFrame({"ns_pos": np.arange(len(channels)),
                         "ns_label": channels})

    if site == 'baylor':
        table = (baylor_macros or {}).get(subject_key)
        if table is None:
            base["unresolved_reason"] = f"no v2026 macro table for '{subject_key}'"
            anat = pd.DataFrame()
        else:
            base["anat_label"] = base["ns_label"].str.replace(
                _NS_SUFFIX_RE, "", regex=True)
            anat = table
        key = "anat_label"

    elif site == 'utah':
        if utah_mat is None:
            base["unresolved_reason"] = "no Electrodes.mat found"
            anat = pd.DataFrame()
        else:
            anat = load_utah_macros(utah_mat)
            # Utah recordings use TWO naming conventions, and both occur here.
            # Most sessions name channels `chan1..chanN`, which join to the
            # amplifier-channel column; others carry the clinical label instead
            # (`LAMG1`, `bLACG6`), which joins to the electrode label directly.
            # The two vocabularies are the same set of electrodes, so this is a
            # rename, not different anatomy.
            #
            # Measured: s01/s02/s23 match `chanN` 128/132 and labels 0/132;
            # s04 matches `chanN` 15/132 and labels 89/132. Requiring `chanN`
            # dropped every channel of the label-named sessions, which is why
            # six Utah sessions resolved 0 of 132 channels on the cluster.
            #
            # channels.npy also carries trailing analog channels (EyeX, EyeY,
            # Pupil, BP); under either key those simply fail to match and are
            # reported unresolved, never matched on position.
            base["utah_chan"] = base["ns_label"].str.extract(
                r'^chan(\d+)$', expand=False).astype(float)
            base["utah_label"] = base["ns_label"].astype(str).str.strip()
            n_by_chan = (int(base["utah_chan"].isin(anat["utah_chan"].dropna()).sum())
                         if "utah_chan" in anat.columns else 0)
            _al = anat["anat_label"].astype(str).str.strip()
            n_by_label = int(base["utah_label"].isin(set(_al)).sum())
            if n_by_label > n_by_chan:
                anat = anat.assign(utah_label=_al)
                key = "utah_label"
            else:
                key = "utah_chan"
        if utah_mat is None:
            key = "utah_chan"

    elif site == 'ucla':
        subj_lab = str(subject_label).strip().strip("'")
        table = (ucla_macros or {}).get(subj_lab)
        if table is None:
            base["unresolved_reason"] = f"no v2026 xlsx for '{subj_lab}'"
            anat = pd.DataFrame()
        else:
            base["ncs_stem"] = base["ns_label"].str.replace(
                r'\.ncs$|_\d{4}$', "", regex=True)
            # Blackrock-recorded UCLA sessions (s50, s51) have no electrode
            # names -- `chan129..chan256`. The montage JSON maps those numbers
            # to probe contacts; without it there is no route from a channel to
            # its MNI coordinate and the session is unusable.
            if base["ncs_stem"].str.match(r'^chan\d+$').all():
                mont = (ucla_montage or {}).get(subj_lab)
                if mont is None or not len(mont):
                    base["unresolved_reason"] = (
                        f"channels are positional (chanN) and no montage JSON "
                        f"for '{subj_lab}' -- cannot reach an MNI coordinate")
                    anat = pd.DataFrame()
                else:
                    known = set(base["ns_label"])
                    missing = sorted(set(mont.ns_label) - known)
                    if missing:
                        print(f"    s{int(session):02d}: {len(missing)} montage "
                              f"channels absent from the recording, e.g. "
                              f"{missing[:4]} -- check MACRO_BANK_OFFSET")
                    base = base.merge(mont[["ns_label", "ncs_stem"]],
                                      on="ns_label", how="left",
                                      suffixes=("_raw", ""))
                    base["ncs_stem"] = base["ncs_stem"].fillna(
                        base.pop("ncs_stem_raw"))
                    anat = table
            else:
                anat = table
        key = "ncs_stem"

    else:
        base["unresolved_reason"] = f"unknown recording site '{recording_site}'"
        anat = pd.DataFrame()
        key = None

    if anat is not None and len(anat) and key in base.columns:
        # pandas merges NaN keys to each other, so a channel with no join key
        # (Utah's trailing EyeX/Pupil/BP analog channels) would fan out against
        # every unlocalised anatomy row. s02 went 132 -> 248 rows before this
        # guard. Drop null keys on both sides, then reattach the unmatched
        # channels so no channel is ever silently lost.
        anat_j = anat[anat[key].notna()].drop_duplicates(subset=[key])
        joinable = base[base[key].notna()]
        merged = joinable.merge(anat_j, on=key, how="left", suffixes=("", "_anat"))
        leftover = base[base[key].isna()].copy()
        if len(leftover):
            leftover["unresolved_reason"] = leftover.get(
                "unresolved_reason", pd.Series([None] * len(leftover))
            ).fillna("channel has no anatomy join key (e.g. analog channel)")
            merged = pd.concat([merged, leftover], ignore_index=True)
        merged = merged.sort_values("ns_pos").reset_index(drop=True)
    else:
        merged = base.copy()
        for c in ["anat_label", "mni_x", "mni_y", "mni_z", "matter",
                  "native_region", "coord_source"]:
            if c not in merged.columns:      # DataFrame has no .setdefault
                merged[c] = np.nan

    merged["session"] = int(session)
    merged["subject_label"] = str(subject_label).strip().strip("'")
    merged["recording_site"] = site

    probe_num = merged["anat_label"].apply(
        lambda l: split_contact(l) if isinstance(l, str) else (None, np.nan))
    merged["probe"] = [p for p, _ in probe_num]
    merged["contact_no"] = [n for _, n in probe_num]
    if "probe_src" in merged.columns:
        merged["probe"] = merged["probe_src"].fillna(merged["probe"])
    merged["hemisphere"] = [
        _hemisphere(l, h) if isinstance(l, str) else None
        for l, h in zip(merged["anat_label"],
                        merged.get("hemisphere_src", pd.Series([None] * len(merged))))]

    has_xyz = merged[["mni_x", "mni_y", "mni_z"]].notna().all(axis=1)
    merged["resolved"] = has_xyz
    if "unresolved_reason" not in merged.columns:
        merged["unresolved_reason"] = None
    merged.loc[~has_xyz & merged["unresolved_reason"].isna(),
               "unresolved_reason"] = "no coordinate for this channel"
    merged.loc[has_xyz, "unresolved_reason"] = None
    return merged


# =============================================================================
# ATLAS LABELLING  (identical rules to the cell pipeline)
# =============================================================================

HPC_PROB_MIN = 25.0     # per cent; matches the maxprob-thr25 atlases


def select_hpc_contacts(df, prob_min=HPC_PROB_MIN):
    """Exactly ONE hippocampal contact per electrode, chosen by coordinate.

    Location comes only from the MNI coordinate: `hpc_prob` is P(hippocampus)
    from the Harvard-Oxford subcortical probability maps, so a contact deep in
    the structure scores above one clipping its edge. Per probe the single
    highest-probability contact is kept, provided it clears `prob_min`; every
    other contact on that probe is `is_hpc = False` however hippocampal it looks.

    No site-supplied region string is consulted. Those strings were previously
    the primary criterion, but they are not comparable across sites, they are
    absent altogether for six Baylor subjects, and they cannot rank two contacts
    that are both "Hippocampus" -- which is what picking one per electrode needs.
    `native_region` is still carried through the table as metadata; nothing reads
    it.

    Adds:
        hpc_prob            P(hippocampus) in per cent at the contact
        is_hpc              True for the single selected contact per probe
        hpc_rank_in_probe   1 = selected, 2 = runner-up, ... (NaN if prob is NaN)
    """
    df = df.copy()
    xyz = df[["mni_x", "mni_y", "mni_z"]].to_numpy(float)
    df["hpc_prob"] = anat_atlas.hippocampal_probability(xyz)

    df["is_hpc"] = False
    df["hpc_rank_in_probe"] = np.nan
    if "probe" not in df.columns:
        return df

    for probe, grp in df.groupby("probe", dropna=True):
        cand = grp[grp["hpc_prob"].notna()]
        if "resolved" in cand.columns:
            cand = cand[cand["resolved"].fillna(False)]
        if cand.empty:
            continue
        order = cand.sort_values("hpc_prob", ascending=False)
        df.loc[order.index, "hpc_rank_in_probe"] = np.arange(1, len(order) + 1)
        best = order.iloc[0]
        if float(best["hpc_prob"]) >= prob_min:
            df.loc[order.index[0], "is_hpc"] = True
    return df


def label_contacts_with_atlas(df, atlases=None):
    """Add `_juelich/_ho_cort/_ho_sub/_bn_label`, then `atlas_roi`,
    `atlas_source_label`, `atlas_reason` via anatomy_atlas.assign_atlas_roi.

    Mirrors the per-cell query block of cell_to_roi_july26.py exactly, so a
    contact and a cell at the same coordinate get the same ROI.
    """
    # Location comes only from the MNI coordinate (see select_hpc_contacts).
    # So if the
    # atlases are unavailable -- e.g. on a cluster node with no nilearn_data and
    # no network -- degrade gracefully rather than failing. This removes a 314 MB
    # dependency from the cluster run; contact selection is unaffected.
    # `atlases=False` means "explicitly none" (caller already knows they are
    # unavailable); `atlases=None` means "not supplied, try to load". Without
    # that distinction a caller passing None would retry the failed load once
    # per session and print the same error 60 times.
    if atlases is None:
        try:
            atlases = anat_atlas.get_atlases()
        except Exception as e:
            print(f"  [atlas unavailable: {type(e).__name__}: {e}]")
            print("  [continuing with native-space ROI only; atlas_roi will be NaN]")
            atlases = False
    if not atlases:
        df = df.reset_index(drop=True).copy()
        for c in ["_juelich", "_ho_cort", "_ho_sub", "_bn_label",
                  "atlas_roi", "atlas_source_label", "atlas_reason"]:
            df[c] = np.nan
        df["atlas_available"] = False
        # Hippocampal selection is coordinate-based, so without an atlas it
        # cannot be made at all. Fail visibly rather than fall back to labels.
        for c in ("atlas_roi", "hpc_prob"):
            if c not in df.columns:
                df[c] = np.nan
        df["is_hpc"] = False
        df["hpc_rank_in_probe"] = np.nan
        df["roi_concordant"] = np.nan          # cannot be assessed without atlas
        return df
    ho_cort, ho_sub, juelich, brainnetome = atlases

    if df.empty:
        for c in ["_juelich", "_ho_cort", "_ho_sub", "_bn_label",
                  "atlas_roi", "atlas_source_label", "atlas_reason"]:
            df[c] = pd.Series(dtype=object)
        return df

    xyz_all = df[["mni_x", "mni_y", "mni_z"]].to_numpy(float)
    records = []
    for xyz in xyz_all:
        if np.any(np.isnan(xyz)):
            records.append({"_juelich": None, "_ho_cort": None,
                            "_ho_sub": None, "_bn_label": None})
            continue
        records.append({
            "_juelich": juelich.label_at(xyz),
            "_ho_cort": ho_cort.label_at(xyz),
            "_ho_sub": ho_sub.label_at(xyz),
            "_bn_label": brainnetome.label_at(xyz),
        })
    df = pd.concat([df.reset_index(drop=True),
                    pd.DataFrame(records)], axis=1)

    # assign_atlas_roi reads MNI_y_final and the four _atlas columns.
    df["MNI_x_final"] = df["mni_x"]
    df["MNI_y_final"] = df["mni_y"]
    df["MNI_z_final"] = df["mni_z"]

    assigns = df.apply(anat_atlas.assign_atlas_roi, axis=1, result_type="expand")
    assigns.columns = ["atlas_roi", "atlas_source_label", "atlas_reason"]
    df = pd.concat([df, assigns], axis=1)
    df = df.drop(columns=["MNI_x_final", "MNI_y_final", "MNI_z_final"])

    df["atlas_available"] = True
    df = select_hpc_contacts(df)
    return df


# =============================================================================
# BIPOLAR PAIRING
# =============================================================================

_WM_TOKENS = ("white",)
# Grey structures a reference contact must NOT sit in: referencing a
# hippocampal contact against another MTL grey contact subtracts out the very
# signal we are trying to measure.
_MTL_GREY = ("HC_anterior", "HC_mid", "HC", "PHC", "EC", "Amygdala")


def _pick_anchor(hpc_rows):
    """The most medial contact of the probe, i.e. the one closest to the
    midline. Defined geometrically as min |MNI x| rather than by contact
    number, so it does not depend on each site's numbering convention
    (Baylor 01 = deepest; Utah/UCLA differ)."""
    return hpc_rows.loc[hpc_rows["mni_x"].abs().idxmin()]


def _pick_reference(probe_rows, anchor, scheme, max_gap_mm):
    """Choose the reference contact for `anchor` on the same probe."""
    others = probe_rows[probe_rows["anat_label"] != anchor["anat_label"]].copy()
    if others.empty:
        return None, "no other contact on probe"

    a_xyz = np.array([anchor["mni_x"], anchor["mni_y"], anchor["mni_z"]], float)
    xyz = others[["mni_x", "mni_y", "mni_z"]].to_numpy(float)
    others["_dist"] = np.linalg.norm(xyz - a_xyz, axis=1)
    others = others[np.isfinite(others["_dist"]) & (others["_dist"] <= max_gap_mm)]
    if others.empty:
        return None, f"no contact within {max_gap_mm} mm"

    if scheme == "neighbour":
        # Chen et al.: the immediate neighbour (second-most medial). Prefer
        # the next contact outward; fall back to the one inward.
        if np.isfinite(anchor.get("contact_no", np.nan)):
            n = int(anchor["contact_no"])
            for cand in (n + 1, n - 1):
                hit = others[others["contact_no"] == cand]
                if len(hit):
                    return hit.iloc[0], "immediate neighbour"
        return others.nsmallest(1, "_dist").iloc[0], "nearest contact"

    if scheme == "white_matter":
        wm = others[
            others["matter"].astype(str).str.lower().str.contains("|".join(_WM_TOKENS))
            & ~others["atlas_roi"].isin(_MTL_GREY)
        ]
        if len(wm):
            return wm.nsmallest(1, "_dist").iloc[0], "nearest white matter"
        return None, "no white-matter contact on probe"

    raise ValueError(f"unknown scheme '{scheme}'")


def build_bipolar_pairs(contacts, target_rois=HPC_ROIS,
                        scheme="neighbour", max_gap_mm=12.0):
    """**One** bipolar derivation per probe -- not every adjacent pair.

    Follows Chen et al.: "bipolar referencing was performed using the most
    medial hippocampal contact and its immediate neighbour (i.e. the
    second-most medial contact) on each hippocampal probe". That yields ~2
    derivations per participant (they report 34 contacts across 17 patients),
    and every derivation is independent -- no contact appears in two pairs.

    Generating all adjacent pairs instead would inflate the count several-fold
    and make the derivations non-independent (contact 3 would appear in both
    pair 2-3 and pair 3-4), which the GLM cannot account for.

    scheme:
      'neighbour'    -- Chen: anchor + immediate neighbour (default; always
                        available).
      'white_matter' -- anchor + nearest same-probe white-matter contact that
                        is not itself in MTL grey. Cleaner subtraction where a
                        suitable contact exists, but not every probe has one.

    Returns one row per (probe, target ROI family) with the anchor's native
    ROI as `pair_roi_atlas`. The pair coordinate is the midpoint, which is
    where the derivation is actually sensitive.
    """
    rows, skipped = [], []
    for probe, grp in contacts.groupby("probe", dropna=True):
        grp = grp[grp["resolved"]].dropna(subset=["contact_no"])
        if grp.empty:
            continue
        hpc = grp[grp["is_hpc"].fillna(False)]
        if hpc.empty:
            continue

        anchor = _pick_anchor(hpc)
        ref, why = _pick_reference(grp, anchor, scheme, max_gap_mm)
        if ref is None:
            skipped.append({"probe": probe, "anchor": anchor["anat_label"],
                            "reason": why})
            continue

        d = float(np.linalg.norm(
            np.array([anchor["mni_x"], anchor["mni_y"], anchor["mni_z"]], float)
            - np.array([ref["mni_x"], ref["mni_y"], ref["mni_z"]], float)))
        rows.append({
            "pair_id": f"{anchor['anat_label']}-{ref['anat_label']}",
            "probe": probe,
            "anat_label_a": anchor["anat_label"], "anat_label_b": ref["anat_label"],
            "ns_pos_a": anchor.get("ns_pos"), "ns_pos_b": ref.get("ns_pos"),
            "ns_label_a": anchor.get("ns_label"), "ns_label_b": ref.get("ns_label"),
            "contact_no_a": anchor.get("contact_no"),
            "contact_no_b": ref.get("contact_no"),
            "hemisphere": anchor.get("hemisphere"),
            "matter_a": anchor.get("matter"), "matter_b": ref.get("matter"),
            "hpc_prob_a": anchor.get("hpc_prob"),
            "hpc_prob_b": ref.get("hpc_prob"),
            "atlas_roi_a": anchor.get("atlas_roi"), "atlas_roi_b": ref.get("atlas_roi"),
            "native_region_a": anchor.get("native_region"),
            "native_region_b": ref.get("native_region"),
            "pair_roi_atlas": anchor.get("atlas_roi"),
            "n_hpc_on_probe": int(len(hpc)),
            "mni_x": (anchor["mni_x"] + ref["mni_x"]) / 2.0,
            "mni_y": (anchor["mni_y"] + ref["mni_y"]) / 2.0,
            "mni_z": (anchor["mni_z"] + ref["mni_z"]) / 2.0,
            "inter_contact_mm": d,
            "ref_rule": scheme,
            "ref_reason": why,
        })
    out = pd.DataFrame(rows)
    out.attrs["skipped"] = skipped
    return out


# =============================================================================
# UTAH ELECTRODE FILE RESOLUTION
# =============================================================================

_UTAH_MAT_INDEX = {}


def resolve_utah_mat(session, subject_key, data_root, manifest=None,
                     verbose=False):
    """Find the Electrodes.mat that genuinely belongs to `session`.

    Resolution is by the patient ID each file declares in its own `Fname`
    (`D:\\Data\\UIC202302\\...`), which is the file's own statement of identity.

    This replaces coord-matching each subject's cells against every folder's
    electrode pool. That had no uniqueness constraint, and with only 3-16 cells
    per Utah subject the match was weak: measured on this dataset, **s47's mat
    was assigned to six different patients**. It was also circular, since the
    coordinates it matched against are the hand-entered ones under question.

    Folder position is not trusted either, and it should not be: `s47` holds
    patient 202302 at its top level and 202311 under `electrodes/`.

    Returns `(mat_dict, provenance)`, or `(None, reason)` so the caller can
    exclude the session with a stated reason rather than borrow another
    patient's anatomy. The same index backs the cell pipeline
    (`scripts/cell_to_roi_july26.py`), so both analyses read anatomy identically.
    """
    global _UTAH_MAT_INDEX
    if not _UTAH_MAT_INDEX:
        _UTAH_MAT_INDEX = anat_src.index_utah_mats_by_id(data_root)

    pid = anat_src.subject_numeric_id(subject_key)
    if pid is None and manifest is not None:
        row = manifest[manifest.session == int(session)]
        if len(row):
            pid = anat_src.subject_numeric_id(row.iloc[0].get("subject_label"))
    if pid and pid in _UTAH_MAT_INDEX:
        loc, mat = _UTAH_MAT_INDEX[pid]
        if verbose:
            print(f"    s{int(session):02d}: patient {pid} -> {loc}")
        return mat, f"declared id {pid} ({loc})"
    return None, f"no electrode file declares patient id {pid}"
