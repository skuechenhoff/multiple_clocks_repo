#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bring the SWR analysis home from the cluster, and write down its numbers.

Two verbs:

    bundle    ripples, artifact-free intervals, derivations, behaviour and
              per-derivation QC for every session, in a few MB. Everything
              downstream of detection can then be redone on a laptop.

    numbers   every value the manuscript quotes, in one small file:
              `manuscript_numbers.md` to read, `manuscript_numbers.json` to
              compute with. Sectioned to match the manuscript, so a number in
              the text can be traced to the run that produced it.

`numbers` never recomputes a statistic. It reads what the pipeline already
wrote -- the contact tables, the QC summary, the surrogate control, the test
result JSONs -- and reports it. If a stage has not run, its section says so
rather than quietly leaving the number out.

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_export.py bundle
    python scripts/swr_export.py numbers
    python scripts/swr_export.py numbers --group_dir=<a downloaded run>
    python scripts/swr_export.py numbers --tests_dir=<ripple_tests_YYYY-MM-DD/...>

On the cluster, run both after the tests finish; then
`scp <cluster>:$SWR/bundle/* <cluster>:$SWR/manuscript_numbers.* .`

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import glob
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.swr_bundle as swr_bundle

try:
    import fire
except ImportError:
    fire = None

SITE_LABEL = {"baylor": "Baylor", "utah": "Utah", "ucla": "UCLA"}


# ------------------------------------------------------------- helpers ----
def _read(path, what):
    """Read a table, or return None and say which stage has not run."""
    if not os.path.isfile(path):
        print(f"  [missing] {what}: {path}")
        return None
    return pd.read_csv(path)


def _q(x, digits=2):
    """median [IQR] of a series, as a string and as numbers."""
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if not len(x):
        return None
    lo, hi = np.percentile(x, [25, 75])
    return {"median": round(float(x.median()), digits),
            "iqr": [round(float(lo), digits), round(float(hi), digits)],
            "min": round(float(x.min()), digits),
            "max": round(float(x.max()), digits),
            "n": int(len(x)),
            "text": f"{x.median():.{digits}f} "
                    f"(IQR {lo:.{digits}f}–{hi:.{digits}f})"}


# ------------------------------------------------- sections of the report --
def section_sessions(group_dir):
    """Which sessions and subjects the analysis rests on, and why the rest not."""
    man = _read(os.path.join(group_dir, "session_manifest.csv"), "session manifest")
    if man is None:
        return None
    out = {"n_sessions_considered": int(len(man)),
           "by_site": man.recording_site.str.lower().value_counts().to_dict(),
           "status": man.status.value_counts().to_dict()}
    if "clock_status" in man.columns:
        out["clock_status"] = man.clock_status.value_counts().to_dict()
    return out


def section_contacts(group_dir):
    """Localisation and montage: the numbers in Methods 'Recording system'."""
    mc_all = _read(os.path.join(group_dir, "macro_contacts_all.csv"),
                   "macro contact table")
    if mc_all is None:
        return None
    res = mc_all[mc_all.get("resolved", pd.Series(True, index=mc_all.index))
                 .fillna(False).astype(bool)]
    hpc = mc_all[mc_all["is_hpc"].fillna(False).astype(bool)]

    out = {
        "n_channels_total": int(len(mc_all)),
        "n_contacts_localised": int(len(res)),
        "n_sessions_with_contacts": int(mc_all.session.nunique()),
        "n_hpc_contacts": int(len(hpc)),
        "n_sessions_with_hpc": int(hpc.session.nunique()),
        "n_subjects_with_hpc": int(hpc.subject_label.nunique()),
        "hpc_by_site": hpc.recording_site.value_counts().to_dict(),
        "hpc_probability_pct": _q(hpc.get("hpc_prob"), 0),
        "hpc_roi": hpc.get("atlas_roi", pd.Series(dtype=object))
                      .value_counts().to_dict(),
        "frac_channels_localised_by_site": {
            k: round(float(v), 2) for k, v in
            mc_all.groupby("recording_site")["resolved"].mean().items()},
    }

    # Concordance with the site's own segmentation, where it exists at all.
    if "native_region" in hpc.columns:
        # "Unknown" is what a site writes when its own segmentation had no
        # answer -- that is an absent label, not a disagreement, and counting
        # it as one understates the concordance.
        lab = hpc.native_region.astype(str).str.lower().str.strip("[]' ")
        has = ~lab.isin(["nan", "", "unknown", "none"])
        agrees = has & lab.str.contains("hippocamp")
        out["native_label_concordance"] = {
            "n_with_site_label": int(has.sum()),
            "n_site_also_says_hippocampus": int(agrees.sum()),
            "pct": (round(100 * agrees.sum() / has.sum(), 0)
                    if has.sum() else None),
            "site_says_otherwise": hpc.loc[has & ~agrees, "native_region"]
                                      .value_counts().to_dict()}

    pairs = _read(os.path.join(group_dir, "bundle", "pairs.csv"), "bundle pairs")
    qc = _read(os.path.join(group_dir, "bundle", "channel_qc.csv"),
               "bundle channel QC")
    if pairs is not None and qc is not None:
        p = pairs.merge(qc[["session", "pair_id", "excluded"]],
                        on=["session", "pair_id"], how="left")
        p["excluded"] = p.excluded.fillna(True).astype(bool)
        inc = p[~p.excluded]
        per_session = inc.groupby("session").size()
        out["montage"] = {
            "n_derivations_built": int(len(p)),
            "n_derivations_excluded_artifact": int(p.excluded.sum()),
            "n_derivations_analysed": int(len(inc)),
            "n_sessions_analysed": int(inc.session.nunique()),
            "n_subjects_analysed": int(inc.subject_label.nunique()),
            "n_distinct_sites": int(inc.drop_duplicates(
                ["subject_label", "pair_id"]).shape[0]),
            "hemisphere": inc.hemisphere.value_counts().to_dict(),
            "derivations_per_session": {
                "median": float(per_session.median()),
                "range": [int(per_session.min()), int(per_session.max())]},
            "inter_contact_mm": _q(inc.get("inter_contact_mm"), 2),
            "pair_roi": inc.get("pair_roi_atlas", pd.Series(dtype=object))
                           .value_counts().to_dict(),
        }
    return out


def section_detection(group_dir):
    """Ripple properties and artifact burden: Methods 'Ripple detection'."""
    out = {}
    qcm = _read(os.path.join(group_dir, "qc_metrics_all_sessions.csv"),
                "QC metrics")
    if qcm is not None:
        # Kept separate from the per-event numbers below: these are one value
        # per session, so a median here is a median over sessions, not over
        # ripples, and the two must never be quoted as if they were the same.
        per_session = {}
        for metric, g in qcm.groupby("metric"):
            per_session[metric] = _q(g.value, 3)
            per_session[metric]["verdicts"] = g.verdict.value_counts().to_dict()
        per_session["n_sessions"] = int(qcm.session.nunique())
        out["per_session"] = per_session

    qc = _read(os.path.join(group_dir, "bundle", "channel_qc.csv"),
               "bundle channel QC")
    if qc is not None:
        out["n_ripples_total"] = int(qc.loc[~qc.excluded.fillna(False), "n_events"].sum())
        out["contaminated_frac"] = _q(qc.loc[~qc.excluded.fillna(False),
                                             "contaminated_frac"], 3)
        out["clean_seconds_per_derivation"] = _q(
            qc.loc[~qc.excluded.fillna(False), "clean_s"], 0)
        out["rate_hz_per_derivation"] = _q(
            qc.loc[~qc.excluded.fillna(False), "rate_hz"], 3)

    rip = os.path.join(group_dir, "bundle", "ripples.csv")
    if os.path.isfile(rip):
        e = pd.read_csv(rip)
        # The bundle holds ACCEPTED ripples only, so the spectral rejection
        # rate cannot be recovered from it -- every event here passed by
        # construction. It is `per_session.spectral_reject_pct` above.
        out["per_event"] = {
            "n_ripples": int(len(e)),
            "duration_ms": _q(e.duration_s * 1000.0, 1),
            "peak_frequency_hz": _q(e.peak_freq_hz, 1),
            "amplitude_uv": _q(e.amp_peak_uv, 1)}
    return out or None


def section_controls(group_dir):
    """The 1/f surrogate floor and the rejection-bias check."""
    out = {}
    sur = _read(os.path.join(group_dir, "surrogate_noise_floor.csv"),
                "surrogate control")
    if sur is not None:
        out["false_positive_frac"] = _q(sur.false_positive_frac, 3)
        out["rate_excess_hz"] = _q(sur.rate_excess_hz, 3)
        out["aperiodic_exponent"] = _q(sur.aperiodic_exponent, 2)
        out["aperiodic_fit_r2"] = _q(sur.aperiodic_r2, 3)
        out["n_derivations"] = int(len(sur))
    for name in ("rejection_bias_windows.csv", "rejection_bias_events.csv"):
        p = os.path.join(group_dir, name)
        if os.path.isfile(p):
            out.setdefault("rejection_bias_files", []).append(name)
    return out or None


def section_tests(tests_dir):
    """Every ripple test, with its question and its numbers, verbatim.

    Nothing is re-tested and nothing is re-thresholded here: the JSON the test
    wrote is the record, and this only flattens it so a p-value can be found
    without opening six files.
    """
    if not tests_dir or not os.path.isdir(tests_dir):
        print(f"  [missing] ripple tests: {tests_dir}")
        return None
    out = {}
    for p in sorted(glob.glob(os.path.join(tests_dir, "*_result.json"))):
        with open(p) as f:
            out[os.path.basename(p).replace("_result.json", "")] = json.load(f)
    counts = {}
    for p in sorted(glob.glob(os.path.join(tests_dir, "*_counts.csv"))):
        counts[os.path.basename(p).replace("_counts.csv", "")] = \
            pd.read_csv(p).to_dict("records")
    return {"results": out, "counts": counts, "tests_dir": tests_dir} or None


# --------------------------------------------------------- the markdown ----
def _md_kv(d, indent=0):
    """Render a nested dict as an indented markdown list, numbers intact."""
    pad = "  " * indent
    lines = []
    for k, v in d.items():
        if isinstance(v, dict) and "text" in v and "median" in v:
            lines.append(f"{pad}- **{k}**: {v['text']}  "
                         f"[min {v['min']}, max {v['max']}, n {v['n']}]")
        elif isinstance(v, dict):
            lines.append(f"{pad}- **{k}**:")
            lines += _md_kv(v, indent + 1)
        elif isinstance(v, list):
            lines.append(f"{pad}- **{k}**: {v}")
        else:
            lines.append(f"{pad}- **{k}**: {v}")
    return lines


def _test_md(tests):
    """One table per test: condition, window, n, effect, p, permutation p."""
    if not tests:
        return ["*(the ripple tests have not been run, or --tests_dir is wrong)*"]
    lines = []
    for name, res in tests["results"].items():
        lines.append(f"### `{name}`")
        lines.append(f"*{res.get('hypothesis', '')}*")
        lines.append("")
        lines.append("| condition | window (s) | n subj | effect (Hz) | t | p | p (perm) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for block in res.get("tests", []):
            for cond, d in block.items():
                for wname, w in d.get("windows", {}).items():
                    lines.append(
                        f"| {cond} | {wname} | {w.get('n_subjects', '')} | "
                        f"{w.get('mean_hz', float('nan')):.4f} | "
                        f"{w.get('t', float('nan')):.2f} | "
                        f"{w.get('p', float('nan')):.4f} | "
                        f"{w.get('p_perm', float('nan')):.4f} |")
        if res.get("conclusion"):
            lines.append("")
            lines.append(f"**Conclusion:** {res['conclusion']}")
        lines.append("")
    return lines


def numbers(group_dir=None, tests_dir=None, out_dir=None, data_root=None):
    """Collect every manuscript number into one file that fits in an email."""
    R = data_root or swr_io.get_data_root()
    group_dir = group_dir or os.path.join(swr_io.derivatives_dir(R), "group", "swr")
    out_dir = out_dir or group_dir
    if tests_dir is None:
        cand = sorted(glob.glob(os.path.join(group_dir, "ripple_tests_*", "*")))
        cand = [c for c in cand if glob.glob(os.path.join(c, "*_result.json"))]
        tests_dir = cand[-1] if cand else None

    print(f"reading   {group_dir}")
    print(f"tests     {tests_dir}")

    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "group_dir": group_dir,
        "sessions": section_sessions(group_dir),
        "contacts": section_contacts(group_dir),
        "detection": section_detection(group_dir),
        "controls": section_controls(group_dir),
    }
    tests = section_tests(tests_dir)

    md = [f"# SWR analysis — every number the manuscript quotes",
          "",
          f"Generated {report['created']} from `{group_dir}`.",
          "",
          "Read straight off the pipeline's own output — nothing here is "
          "recomputed. A section that says *not available* means that stage "
          "has not been run in this directory.",
          ""]
    titles = {"sessions": "Sessions and subjects",
              "contacts": "Recording system and electrode localisation",
              "detection": "Ripple detection",
              "controls": "Controls (1/f surrogate, rejection bias)"}
    for key, title in titles.items():
        md.append(f"## {title}")
        md.append("")
        if report[key] is None:
            md.append("*not available in this directory*")
        else:
            md += _md_kv(report[key])
        md.append("")
    md.append("## Ripple tests")
    md.append("")
    md += _test_md(tests)

    os.makedirs(out_dir, exist_ok=True)
    md_p = os.path.join(out_dir, "manuscript_numbers.md")
    js_p = os.path.join(out_dir, "manuscript_numbers.json")
    with open(md_p, "w") as f:
        f.write("\n".join(md) + "\n")
    with open(js_p, "w") as f:
        json.dump({**report, "tests": tests}, f, indent=2, default=str)
    print(f"\n-> {md_p}\n-> {js_p}")
    return None


def bundle(analysis_name="swr_v1", data_root=None, out_dir=None,
           with_hfb=True, hfb_arrays=None):
    """Write the bundle: the MB that replace the LFP on a laptop.

    `with_hfb` also copies the continuous 100 Hz HFB / band-envelope stores into
    `bundle/hfb/`, one npz per session -- the cortical half, and the larger
    half. Set `--with_hfb=False` for the original ripples-only bundle, or
    narrow it with e.g. `--hfb_arrays="['hfb','ripple']"` if the transfer is
    the constraint. Nothing is epoched either way.
    """
    return swr_bundle.export_bundle(analysis_name=analysis_name,
                                    data_root=data_root, out_dir=out_dir,
                                    with_hfb=with_hfb,
                                    hfb_arrays=hfb_arrays) and None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"bundle": bundle, "numbers": numbers})
    else:
        print("fire is not installed -- `pip install fire`, then re-run.")
