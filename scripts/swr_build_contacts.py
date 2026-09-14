#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 2 of the SWR pipeline: build the macro-contact anatomy table.

One row per LFP channel per session, with an MNI152 coordinate, grey/white,
a native-atlas region label, and an ROI assigned by the SAME rule ladder the
cell pipeline uses. Then adjacent-bipolar pairs for the contacts we care about.

Group-level on purpose: the nilearn fetches plus four MaxProbAtlas
instantiations cost ~30 s, and paying that once per session across 60 sessions
would be pointless.

Outputs, per session:
    derivatives/s{XX}/LFP/macro_contacts_{XX}.csv
    derivatives/s{XX}/LFP/bipolar_pairs_{XX}.csv
and group-level:
    derivatives/group/swr/macro_contacts_all.csv
    derivatives/group/swr/contact_qc.csv
    derivatives/group/swr/settings.json

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_build_contacts.py
    python scripts/swr_build_contacts.py --sessions="[2,5,26]" --verbose=True

@author: Svenja Kuchenhoff
"""

import os
import sys
import glob
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.anatomy_sources as anat_src
import mc.analyse.anatomy_atlas as anat_atlas
import mc.analyse.contact_anatomy as ca
import mc.analyse.swr_report as swr_report

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "swr_contacts"
V2026_DIRNAME = "ABCD_pts_elecFilesForSvenja_v2026"


def _settings_dict(sessions, v2026_folder, cortical_rois=()):
    return {
        "analysis_name": ANALYSIS_NAME,
        "sessions": list(sessions),
        "v2026_folder": v2026_folder,
        "hpc_rois": list(ca.HPC_ROIS),
        "cortical_rois": list(cortical_rois or []),
        "cortical_scheme": ("non-overlapping adjacent pairs anchored on each "
                            "in-ROI contact; one contact is never reused"),
        "bipolar_scheme": "neighbour (Chen: most medial HC contact + immediate neighbour, ONE pair per probe)",
        "utah_coords": "ElecXYZMNIRaw (direct row index, NOT via channel no.)",
        "atlas_rules": "mc.analyse.anatomy_atlas.assign_atlas_roi (shared with cells)",
        "hc_ant_mid_y": anat_atlas.HC_ANT_MID_Y,
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def build_contacts(sessions=None, save_all=True, verbose=False,
                   use_atlas=True, cortical_rois=ca.CORTICAL_ROIS):
    """`cortical_rois` adds high-frequency-broadband derivations to the SAME
    bipolar_pairs table as the hippocampal ones, so stage 2 reads them in one
    raw-file pass. Pass `--cortical_rois="[]"` for the hippocampus-only table.
    Hippocampal rows are unaffected either way -- cortical rows are appended."""
    swr_io.start_log(os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()), "group", "swr"), "swr_build_contacts")
    data_root = swr_io.get_data_root()
    v2026 = os.path.join(data_root, V2026_DIRNAME)

    manifest_p = os.path.join(swr_io.derivatives_dir(data_root),
                              "group", "swr", "session_manifest.csv")
    if not os.path.isfile(manifest_p):
        raise FileNotFoundError(
            f"{manifest_p} not found -- run scripts/swr_audit_sessions.py first")
    manifest = pd.read_csv(manifest_p)

    if sessions is not None:
        manifest = manifest[manifest.session.isin([int(s) for s in sessions])]

    print(f"Loading atlases and source tables ({len(manifest)} sessions)...")
    # The atlas is REQUIRED: hippocampal contacts are selected from the MNI
    # coordinate alone (mc.analyse.contact_anatomy.select_hpc_contacts), so
    # without it no contact can be chosen. Site-supplied region strings are not
    # consulted. A missing atlas therefore yields zero pairs, loudly.
    atlases = False          # False = explicitly none (see label_contacts_with_atlas)
    if use_atlas:
        try:
            atlases = anat_atlas.get_atlases()
        except Exception as e:
            print(f"  [atlas unavailable] {e}")
            print("  [continuing with native-space ROI only -- contact selection "
                  "is unaffected; atlas_roi will be NaN]")
    else:
        print("  [--use_atlas=False: native-space ROI only]")
    baylor_macros = ca.load_baylor_macros(v2026)
    ucla_macros = ca.load_ucla_macros(v2026)
    ucla_montage = ca.load_ucla_montage(v2026)
    print(f"  baylor macro tables: {len(baylor_macros)} subjects")
    print(f"  ucla macro tables:   {len(ucla_macros)} subjects")
    print(f"  ucla montage maps:   {len(ucla_montage)} subjects "
          f"(chanN -> contact, for the Blackrock-recorded UCLA sessions)")

    # Utah electrode files are resolved PER SESSION (own folder, then same
    # patient), not by coord-matching cells against every folder -- see
    # ca.resolve_utah_mat for why that was unsafe here.

    all_rows, qc_rows, all_pairs = [], [], []
    for _, m in manifest.iterrows():
        session = int(m.session)
        site = str(m.recording_site).lower()
        label = str(m.subject_label).strip().strip("'") if pd.notna(m.subject_label) else None

        channels, chan_src, _ = ca.load_channel_list(
            session, data_root, verbose=verbose)
        if not channels:
            qc_rows.append({"session": session, "recording_site": site,
                            "n_channels": 0, "n_resolved": 0, "n_hpc": 0,
                            "n_pairs": 0, "n_hpc_pairs": 0,
                            "note": "no channel list available"})
            if verbose:
                print(f"  s{session:02d}: no channel list")
            continue

        utah_mat, utah_src = None, ""
        if site == 'utah':
            utah_mat, utah_src = ca.resolve_utah_mat(
                session, m.subject_key, data_root, manifest=manifest,
                verbose=verbose)
            if utah_mat is None and verbose:
                print(f"  s{session:02d}: {utah_src}")

        contacts = ca.build_macro_table(
            session, site, label, channels,
            baylor_macros=baylor_macros, utah_mat=utah_mat,
            ucla_macros=ucla_macros, ucla_montage=ucla_montage)
        contacts = ca.label_contacts_with_atlas(contacts, atlases=atlases)

        pairs = ca.build_bipolar_pairs(contacts, cortical_rois=cortical_rois)
        if len(pairs):
            pairs = ca.label_contacts_with_atlas(pairs, atlases=atlases)
            pairs = pairs.rename(columns={"atlas_roi": "pair_roi"})
            pairs.insert(0, "session", session)
            pairs.insert(1, "subject_label", label)

        n_res = int(contacts["resolved"].sum())
        n_hpc = int(contacts["is_hpc"].sum())
        n_hpc_pairs = int(pairs["pair_roi_atlas"].isin(ca.HPC_ROIS).sum()) if len(pairs) else 0
        n_cx_pairs = int((pairs["role"] == "hfb").sum()) if len(pairs) else 0
        n_mpfc_pairs = (int(pairs["pair_roi_atlas"].isin(["mPFC", "mOFC"]).sum())
                        if len(pairs) else 0)
        qc_rows.append({
            "session": session, "recording_site": site, "subject_label": label,
            "channel_source": chan_src, "n_channels": len(contacts),
            "n_resolved": n_res,
            "frac_resolved": round(n_res / max(len(contacts), 1), 3),
            "n_hpc": n_hpc, "n_pairs": len(pairs), "n_hpc_pairs": n_hpc_pairs,
            "n_cortical_pairs": n_cx_pairs, "n_mpfc_pairs": n_mpfc_pairs,
            "note": "",
        })
        if verbose:
            print(f"  s{session:02d} {site:7s} ch={len(contacts):4d} "
                  f"resolved={n_res:4d} hpc={n_hpc:3d} pairs={len(pairs):3d} "
                  f"hpc_pairs={n_hpc_pairs:3d} cortical={n_cx_pairs:3d} "
                  f"(mPFC/mOFC {n_mpfc_pairs:3d})")

        all_rows.append(contacts)
        if len(pairs):
            all_pairs.append(pairs)
        if save_all:
            out_dir = os.path.join(swr_io.session_deriv_dir(session, data_root), "LFP")
            os.makedirs(out_dir, exist_ok=True)
            contacts.to_csv(os.path.join(out_dir, f"macro_contacts_{session:02d}.csv"),
                            index=False)
            pair_path = os.path.join(out_dir, f"bipolar_pairs_{session:02d}.csv")
            if len(pairs):
                pairs.to_csv(pair_path, index=False)
            elif os.path.isfile(pair_path):
                # A session that yields no pairs now must not keep a file from an
                # earlier run: swr_check_inputs counts files, so a stale one reads
                # as "ready for stage 2" and stage 2 would run on outdated contacts.
                os.remove(pair_path)
                print(f"    s{session:02d}: removed stale {os.path.basename(pair_path)} "
                      f"(this build produced no pairs)")

    qc = pd.DataFrame(qc_rows)

    # inclusion report: every session accounted for, with a reason
    rep = swr_report.InclusionReport(
        "contacts", ANALYSIS_NAME,
        "Sessions yielding at least one hippocampal bipolar derivation.")
    for _, r in qc.iterrows():
        u = f"s{int(r.session):02d}"
        if int(r.get("n_hpc_pairs", 0)) > 0:
            rep.include(u, "", site=r.recording_site,
                        channels=int(r.n_channels), resolved=int(r.n_resolved),
                        hpc=int(r.n_hpc), pairs=int(r.n_hpc_pairs))
        elif int(r.get("n_channels", 0)) == 0:
            rep.exclude(u, "no channel list could be read "
                           "(no channels.npy and raw header unreadable)",
                        site=r.recording_site)
        elif int(r.get("n_channels", 0)) < 8:
            # s16 is the only session whose LFP is the NSP-2 amplifier with no
            # NSP-1 counterpart; both its files carry 4 placeholder channels
            # ('empty-064'...). That is missing data, not a join failure.
            rep.exclude(u, f"only {int(r.n_channels)} channels in the recording "
                           "(placeholder/second-amplifier file; the real "
                           "amplifier's data is not present)",
                        site=r.recording_site)
        elif int(r.get("n_resolved", 0)) == 0:
            rep.exclude(u, "no channel matched an electrode table "
                           "(missing/mismatched anatomy source)",
                        site=r.recording_site, channels=int(r.n_channels))
        elif int(r.get("n_hpc", 0)) == 0:
            rep.exclude(u, "no contact in hippocampus by native segmentation",
                        site=r.recording_site, resolved=int(r.n_resolved))
        else:
            rep.exclude(u, "hippocampal contacts found but no valid bipolar "
                           "partner on the same probe",
                        site=r.recording_site, hpc=int(r.n_hpc))
    print("\n" + "=" * 74)
    print(" CONTACT ANATOMY SUMMARY")
    print("=" * 74)
    print(qc.groupby("recording_site")[
        ["n_channels", "n_resolved", "n_hpc", "n_pairs", "n_hpc_pairs",
         "n_cortical_pairs", "n_mpfc_pairs"]
    ].sum().to_string())
    print(f"\nsessions with >=1 hippocampal bipolar pair: "
          f"{int((qc.n_hpc_pairs > 0).sum())}/{len(qc)}")

    # Subject-level availability. Sessions are the analysis unit but subjects are
    # the sample: several subjects contributed 2-3 sessions, so a session count
    # overstates independent coverage.
    sub = qc.dropna(subset=["subject_label"]).copy()
    sub["has_lfp"] = sub["n_channels"] > 0
    by_sub = sub.groupby("subject_label").agg(
        sessions=("session", "size"),
        sessions_with_lfp=("has_lfp", "sum"),
        sessions_with_hpc=("n_hpc_pairs", lambda x: int((x > 0).sum())),
        hpc_pairs=("n_hpc_pairs", "sum"))
    n_lfp = int((by_sub.sessions_with_lfp > 0).sum())
    n_hpc = int((by_sub.sessions_with_hpc > 0).sum())
    print("\n--- subject-level coverage (the sample, not the analysis unit) ---")
    print(f"  subjects in the manifest              : {len(by_sub)}")
    print(f"  with LFP available                    : {n_lfp}")
    print(f"  of those, with a usable HPC derivation: {n_hpc}"
          f"  ({100 * n_hpc / max(n_lfp, 1):.0f}% of subjects with LFP)")
    lost = by_sub[(by_sub.sessions_with_lfp > 0) & (by_sub.sessions_with_hpc == 0)]
    if len(lost):
        print(f"  subjects with LFP but NO hippocampal contact ({len(lost)}): "
              + ", ".join(sorted(lost.index.astype(str))))

    if all_rows:
        allc = pd.concat(all_rows, ignore_index=True)
        print("\n--- ROI counts across all resolved contacts ---")
        print(allc.loc[allc.resolved, "atlas_roi"].value_counts().head(12).to_string())

        if save_all:
            gdir = os.path.join(swr_io.derivatives_dir(data_root), "group", "swr")
            os.makedirs(gdir, exist_ok=True)
            allc.to_csv(os.path.join(gdir, "macro_contacts_all.csv"), index=False)
            qc.to_csv(os.path.join(gdir, "contact_qc.csv"), index=False)
            rep.write(gdir)
            swr_io.write_settings(gdir, _settings_dict(
                list(manifest.session.astype(int)), v2026, cortical_rois))
            # The cross-regional sample, which is what decides whether any
            # HC->mPFC analysis is possible at all. Printed here rather than
            # discovered three stages later.
            both = qc[(qc.n_hpc_pairs > 0) & (qc.n_mpfc_pairs > 0)]
            print(f"\nsessions with BOTH a hippocampal and an mPFC/mOFC "
                  f"derivation: {len(both)}  "
                  f"(subjects: {both.subject_label.nunique()})")
            print(f"  mPFC/mOFC derivations in those sessions: "
                  f"{int(both.n_mpfc_pairs.sum())}")
            print(f"\nSaved -> {gdir}")

            # Publication coverage figure. Optional: it needs nilearn plotting
            # and the MNI templates, which a compute node without a warm
            # nilearn_data cache does not have. Contact building must not fail
            # because a figure could not be drawn.
            fdir = os.path.join(gdir, "figures")
            os.makedirs(fdir, exist_ok=True)
            stem = os.path.join(fdir, "contact_coverage")
            try:
                import matplotlib
                matplotlib.use("Agg")
                import mc.plotting.ripple_figures as rfig
                rfig.contact_coverage_figure(allc, out_stem=stem)
                print(f"Coverage figure -> {stem}.pdf / .jpg")
                # Where every DERIVATION sits, coloured by ROI, with the
                # same-shaft controls ringed. This is the figure to look at
                # before spending an extraction: it shows at a glance whether
                # the cortical coverage supports the question being asked.
                if all_pairs:
                    ap = pd.concat(all_pairs, ignore_index=True)
                    mstem = os.path.join(fdir, "montage_coverage")
                    rfig.montage_figure(
                        ap, out_stem=mstem, height_cm=5.0,
                        title="SWR montage: hippocampus, mPFC/mOFC, controls")
                    print(f"Montage figure  -> {mstem}.pdf / .jpg")
            except Exception as e:
                print(f"  [coverage figure skipped: {type(e).__name__}: {e}]")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(build_contacts)
    else:
        build_contacts()
