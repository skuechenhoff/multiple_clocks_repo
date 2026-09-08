#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate SLURM condition files listing only the sessions that are actually
runnable, so array jobs do not burn tasks on sessions with no data.

Reads `group/swr/session_manifest.csv` (and `input_check.csv` if present) and
writes one `--session=N ...` line per usable session.

Usage:
    python scripts/swr_make_conditions.py                       # both stages
    python scripts/swr_make_conditions.py --stage=detect        # only stage 3
    python scripts/swr_make_conditions.py --analysis_name=swr_v2

@author: Svenja Kuchenhoff
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

# Condition files are GENERATED STATE, not code: their contents depend on which
# stages have already finished on the machine you are running on. Keeping them
# in the git repo meant a laptop's short list could be pulled over the cluster's
# correct one. They now live with the data.
OUT_DIR = None


def make(stage="both", analysis_name="swr_v1", out_dir=OUT_DIR,
         include_needs_review=True):
    R = swr_io.get_data_root()
    if out_dir is None:
        out_dir = os.path.join(swr_io.derivatives_dir(R), "group", "swr",
                               "condition_files")
    gdir = os.path.join(swr_io.derivatives_dir(R), "group", "swr")
    mf_p = os.path.join(gdir, "session_manifest.csv")
    if not os.path.isfile(mf_p):
        raise FileNotFoundError(f"{mf_p} -- run scripts/swr_audit_sessions.py first")
    mf = pd.read_csv(mf_p)

    ok = mf[mf.n_raw_files > 0].copy()
    if not include_needs_review:
        ok = ok[ok.status == "ok"]
    print(f"\n{len(mf)} sessions in manifest; {len(ok)} have raw files")

    # stage 2 needs bipolar_pairs; stage 3 needs continuous.npy
    rows_pre, rows_det, rows_qc, rows_fig = [], [], [], []
    for _, r in ok.iterrows():
        s = int(r.session)
        pairs = os.path.join(swr_io.session_deriv_dir(s, R), "LFP",
                             f"bipolar_pairs_{s:02d}.csv")
        cont = os.path.join(swr_io.session_deriv_dir(s, R), "LFP-clean",
                            analysis_name, "continuous.npy")
        arg = f"--session={s} --analysis_name={analysis_name}"
        if os.path.isfile(pairs):
            rows_pre.append(arg)
        if os.path.isfile(cont):
            rows_det.append(arg)
        # swr_qc_report takes a VERB before its flags, and the batch script
        # passes each condition line verbatim to python, so the verb belongs in
        # the line rather than on the submit command.
        ev = os.path.join(swr_io.session_deriv_dir(s, R), "LFP-ripples",
                          analysis_name, "ripple_events.csv")
        if os.path.isfile(ev):
            rows_qc.append(f"metrics {arg}")
            # `metrics` prints the checkpoint numbers and nothing else; the
            # per-session FIGURES -- artifact_rejection, artifact_padding,
            # ied_vs_ripple, chen_fig2, the example grids -- come from `report`,
            # which is ~10x slower and so gets its own list rather than being
            # folded into the qc one.
            rows_fig.append(f"report {arg}")

    os.makedirs(out_dir, exist_ok=True)
    written = []
    if stage in ("both", "extract"):
        p = os.path.join(out_dir, f"swr_extract_{analysis_name}.txt")
        with open(p, "w") as f:
            f.write("\n".join(rows_pre) + "\n")
        written.append((p, len(rows_pre), "stage 2 (extract)"))
    if stage in ("both", "detect"):
        p = os.path.join(out_dir, f"swr_detect_{analysis_name}.txt")
        with open(p, "w") as f:
            f.write("\n".join(rows_det) + "\n")
        written.append((p, len(rows_det), "stage 3 (detect)"))
    if stage in ("both", "qc") and not rows_qc:
        print("\n  NOTE stage-4 list is empty: no ripple_events.csv yet.")
        print("       Run stage 3 first, then re-run this to build the qc list.")
    if stage in ("both", "qc") and rows_qc:
        p = os.path.join(out_dir, f"swr_qc_{analysis_name}.txt")
        with open(p, "w") as f:
            f.write("\n".join(rows_qc) + "\n")
        written.append((p, len(rows_qc), "stage 4 (qc metrics)"))
    if stage in ("both", "qcfig", "report") and rows_fig:
        p = os.path.join(out_dir, f"swr_qcfig_{analysis_name}.txt")
        with open(p, "w") as f:
            f.write("\n".join(rows_fig) + "\n")
        written.append((p, len(rows_fig), "stage 4b (qc FIGURES)"))

    print()
    for p, n, what in written:
        print(f"  {what:20s} {n:3d} sessions -> {p}")
    print("\n  Each list contains only the sessions whose PREVIOUS stage has")
    print("  already finished on this machine, so the lists grow as you go:")
    print("     extract  needs bipolar_pairs_NN.csv   (swr_build_contacts.py)")
    print("     detect   needs continuous.npy         (swr_extract_continuous.py)")
    print("     qc       needs ripple_events.csv      (swr_detect_session.py)")
    print("     qcfig    needs ripple_events.csv      (same, but draws figures)")
    print("  Re-run this script after each stage to pick up what just finished.")
    if stage in ("both", "detect") and not rows_det:
        print("\n  NOTE stage-3 list is empty: no continuous.npy yet.")
        print("       Run stage 2 first, then re-run this to build the detect list.")
    missing_pairs = len(ok) - len(rows_pre)
    if missing_pairs > 0:
        print(f"\n  {missing_pairs} sessions have raw data but no bipolar_pairs_*.csv")
        print("       -> run scripts/swr_build_contacts.py")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(make)
    else:
        make()
