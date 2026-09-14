#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 4 of the SWR pipeline: artifact rejection + ripple detection.

Reads `continuous.npy` from swr_extract_continuous, applies Chen's five
artifact criteria plus the Janca IED detector, estimates the ripple threshold
over the whole artifact-free session, detects events and flags them against
Chen's four spectral criteria.

Fast to re-run (~seconds to a minute per session) because it never touches the
raw files -- which is the entire reason extraction and detection are separate
stages.

Outputs, per session:
    derivatives/s{XX}/LFP-ripples/{name}/ripple_events.csv
    derivatives/s{XX}/LFP-ripples/{name}/channel_qc.csv
    derivatives/s{XX}/LFP-ripples/{name}/clean_intervals.csv
    derivatives/s{XX}/LFP-ripples/{name}/detector_diag.json
    derivatives/s{XX}/LFP-ripples/{name}/settings.json

Usage:
    python scripts/swr_detect_session.py --session=38

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.swr_preproc as pp
import mc.analyse.swr_artifact as art
import mc.analyse.swr_detect as det
import mc.analyse.swr_report as swr_report

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "swr_v1"


def _settings_dict(session, analysis_name, pad_s=None, clean_name=None):
    pad_s = art.PAD_S if pad_s is None else float(pad_s)
    return {
        "analysis_name": analysis_name, "session": int(session),
        # Which extraction this run READ. swr_bundle.export_bundle needs it to
        # find pairs.csv for a variant run (a pad sweep reads swr_v1 and writes
        # under its own name); without it the bundle's pairs table comes back
        # empty and the session's behaviour is skipped with it.
        "clean_name": clean_name or analysis_name,
        "ripple_band_hz": list(det.RIPPLE_BAND),
        "threshold_extent_sd": det.LO_SD,
        "threshold_peak_sd": det.PEAK_SD,
        "threshold_ceiling_sd": det.HI_SD,
        "threshold_rationale": ("3.0 SD chosen by maximising excess over the 1/f "
                                "surrogate noise floor; Chen used 1.5 SD, which "
                                "here leaves 97% of detections inside the floor"),
        "sensitivity_analysis": "swr_lo1.5_sensitivity (Chen's 1.5 SD)",
        "threshold_scope": "whole artifact-free session (NOT per snippet)",
        "duration_ms": list(det.DUR_MS),
        "merge_gap_ms": det.MERGE_GAP_MS,
        "rms_window_ms": det.RMS_WIN_MS,
        "artifact_iqr_k": art.IQR_K,
        "artifact_pad_s": pad_s,
        "artifact_pad_s_default": art.PAD_S,
        "exclusion_pad_s": art.EXCLUSION_PAD_S,
        "pad_rationale": ("pad shrunk from 1.0 s to 0.25 s on 2026-09-13: the five "
                          "criteria flag ~2.6% of samples and the +-1 s dilation was "
                          "removing 43%. Contamination is still JUDGED at 1.0 s so the "
                          "set of included derivations does not move with the pad."),
        "min_clean_s": art.MIN_CLEAN_S,
        "max_contaminated_frac": art.MAX_CONTAM_FRAC,
        "near_artifact_s": art.NEAR_ARTIFACT_S,
        "repad_note": ("artifact_intervals.csv holds the UNPADDED criterion "
                       "crossings; art.clean_intervals_at_pad rebuilds the "
                       "exposure denominator at any pad, and events carry "
                       "dist_to_artifact_s. Both are needed -- filtering "
                       "events without shrinking the denominator is wrong."),
        "ied_detector": "Janca et al. 2015 (log-normal envelope model)",
        "spectral_criteria": "Chen et al. 2025, 4 peak-based flags (bitmask)",
        "criterion2_scope_primary": "strict = Chen literal, 30-200 Hz outside band",
        "criterion2_scope_sensitivity": "relaxed = 120-200 Hz only",
        "primary_analysis": "strict (pre-declared)",
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def detect_session(session, analysis_name=ANALYSIS_NAME, save_all=True,
                   verbose=True, pad_s=None, clean_name=None):
    """`pad_s` overrides art.PAD_S for this run -- the whole point of keeping
    detection separate from extraction is that a pad sweep costs no raw I/O.
    Always pair it with a distinct `--analysis_name`.

    `clean_name` is which LFP-clean/ extraction to READ (default: the same as
    `analysis_name`). Without it a sensitivity run could only read an extraction
    of its own name, so every pad variant would need its own copy of
    continuous.npy -- hours of I/O to change one dilation width.
    """
    swr_io.start_log(os.path.join(swr_io.session_deriv_dir(int(session), swr_io.get_data_root()), "LFP-ripples", analysis_name), "swr_detect_session")
    session = int(session)
    pad_s = art.PAD_S if pad_s is None else float(pad_s)
    data_root = swr_io.get_data_root()
    clean_name = analysis_name if clean_name is None else str(clean_name)
    clean_dir = os.path.join(swr_io.session_deriv_dir(session, data_root),
                             "LFP-clean", clean_name)
    sig_p = os.path.join(clean_dir, "continuous.npy")
    if not os.path.isfile(sig_p):
        print(f"s{session:02d}: no continuous.npy under LFP-clean/{clean_name} "
              f"-- run swr_extract_continuous first, or pass --clean_name")
        return None

    sig = np.load(sig_p, mmap_mode='r')
    pairs = pd.read_csv(os.path.join(clean_dir, "pairs.csv"))
    with open(os.path.join(clean_dir, "meta.json")) as f:
        meta = json.load(f)
    fs = float(meta["fs"])
    total_s = sig.shape[1] / fs

    # Ripple detection runs ONLY on the hippocampal derivations. The cortical
    # rows share this file so that extraction reads every channel in one pass,
    # but this detector's band, thresholds and duration gate are Chen's
    # hippocampal ones. Running them on mPFC would emit a table that looks
    # exactly like a ripple table and means nothing -- cortical ripples are a
    # separate literature with separate parameters. The HFB stage is what reads
    # the cortical rows.
    n_all = sig.shape[0]
    if "role" in pairs.columns:
        keep_idx = [i for i, r in enumerate(pairs.role.astype(str)) if r == "ripple"]
    else:
        keep_idx = list(range(len(pairs)))          # pre-cortical pair table
    n_skip = n_all - len(keep_idx)

    print(f"\ns{session:02d}: {n_all} derivations, {total_s:.0f}s @ {fs:.0f}Hz  "
          f"[pad {pad_s:.2f}s, exclusion judged at {art.EXCLUSION_PAD_S:.2f}s]")
    if n_skip:
        print(f"  detecting on {len(keep_idx)} hippocampal derivation(s); "
              f"{n_skip} cortical row(s) are for swr_extract_hfb.py, not this stage")
    if not keep_idx:
        print(f"  s{session:02d}: no hippocampal derivation in this montage, "
              f"nothing to detect")
        return None

    all_ev, qc, diags, iv_rows, bad_rows = [], [], {}, [], []
    for i in keep_idx:
        p = pairs.iloc[i]
        x = np.asarray(sig[i], float)

        # Criteria once; combined twice. `pad_s` sets the analysis mask, but
        # contamination is judged at the FIXED EXCLUSION_PAD_S so that changing
        # the pad cannot silently change which derivations enter the sample --
        # otherwise "more exposure" and "dirtier contacts added" are confounded.
        bad, astats, per = art.artifact_mask(x, fs, pad_s=pad_s, return_per=True)
        clean = ~bad
        contam = float(bad.mean())
        contam_excl = float(art.combine_criteria(
            per, fs, pad_s=art.EXCLUSION_PAD_S).mean())
        excluded = contam_excl > art.MAX_CONTAM_FRAC

        iv = art.clean_intervals(bad, fs)
        clean_s = float(np.diff(iv, axis=1).sum()) if len(iv) else 0.0

        row = {"session": session, "pair_id": p.pair_id,
               "pair_roi": p.get("pair_roi_atlas"),
               "hemisphere": p.get("hemisphere"),
               "contaminated_frac": round(contam, 4),
               "contaminated_frac_at_exclusion_pad": round(contam_excl, 4),
               "pad_s": pad_s,
               "clean_s": round(clean_s, 1),
               "excluded": excluded,
               **{f"frac_{k}": round(v, 4) for k, v in astats.items()}}

        if excluded:
            row.update({"n_events": 0, "rate_hz": np.nan})
            qc.append(row)
            if verbose:
                print(f"  {p.pair_id:24s} EXCLUDED ({contam_excl:.0%} "
                      f"contaminated at the {art.EXCLUSION_PAD_S:.2f}s exclusion pad)")
            continue

        ev, diag = det.detect_channel(x, fs, clean)
        diags[p.pair_id] = diag

        n_pass = int(ev.passed.sum()) if len(ev) else 0
        rate = n_pass / clean_s if clean_s > 0 else np.nan
        # The pad exists to keep IED ringing out of the accepted events, so a
        # pad change is judged on THIS, not on the exposure it recovers.
        # Stored per event (not just summarised) so any pad can be re-imposed
        # downstream by filtering the column -- see art.artifact_distance.
        if len(ev):
            ev["dist_to_artifact_s"] = np.round(
                art.artifact_distance(ev.t_peak_s.to_numpy(), per, fs), 4)
        near = (float((ev.loc[ev.passed, "dist_to_artifact_s"]
                       < art.NEAR_ARTIFACT_S).mean()) if n_pass else np.nan)
        row.update({"n_candidates": diag.get("n_candidates", 0),
                    "frac_near_artifact": round(near, 4) if np.isfinite(near) else np.nan,
                    "n_events": n_pass,
                    "rate_hz": round(rate, 4) if np.isfinite(rate) else np.nan})
        qc.append(row)

        if len(ev):
            ev.insert(0, "session", session)
            ev.insert(1, "pair_id", p.pair_id)
            ev["pair_roi"] = p.get("pair_roi_atlas")
            ev["hemisphere"] = p.get("hemisphere")
            ev["subject_label"] = p.get("subject_label")
            all_ev.append(ev)
        for a, b in iv:
            iv_rows.append({"pair_id": p.pair_id, "start_s": a, "stop_s": b})
        # UNPADDED criterion crossings. These are what let the pad be re-chosen
        # on the laptop: art.clean_intervals_at_pad rebuilds the exposure
        # denominator at any pad from them, exactly (verified bit-for-bit
        # against the cluster-side mask at 0.1/0.25/0.5/1.0 s).
        for a, b in art.bad_intervals(per, fs):
            bad_rows.append({"pair_id": p.pair_id, "start_s": round(a, 4),
                             "stop_s": round(b, 4)})

        if verbose:
            print(f"  {p.pair_id:24s} clean={clean_s:7.0f}s "
                  f"cand={diag.get('n_candidates',0):5d} "
                  f"pass={n_pass:5d}  rate={rate:.3f} Hz")

    events = pd.concat(all_ev, ignore_index=True) if all_ev else pd.DataFrame()
    qc_df = pd.DataFrame(qc)

    if len(events):
        print(f"\n  total {int(events.passed.sum())} ripples "
              f"({len(events)} candidates) across {qc_df.excluded.eq(False).sum()} pairs")
        good = qc_df[~qc_df.excluded & qc_df.rate_hz.notna()]
        if len(good):
            print(f"  rate: median {good.rate_hz.median():.3f} Hz "
                  f"(range {good.rate_hz.min():.3f}-{good.rate_hz.max():.3f}); "
                  f"Chen reference ~0.17-0.24 Hz")
            rs = 1 - events.spectral_passed_strict.mean()
            rr = 1 - events.spectral_passed_relaxed.mean()
            print(f"  spectral rejection: strict {rs:.1%} (PRIMARY) | "
                  f"relaxed {rr:.1%}   [Chen: 23.4% +- 9.9%]")
            if "frac_near_artifact" in good and good.frac_near_artifact.notna().any():
                print(f"  events within {art.NEAR_ARTIFACT_S:.2f}s of an unpadded "
                      f"crossing: median {good.frac_near_artifact.median():.1%}  "
                      f"<- what a pad change is judged on; these are the events "
                      f"a 1s pad would have discarded")
            print(f"  clean exposure: {good.clean_s.sum()/3600:.2f} h at pad {pad_s:.2f}s")
            n_s = int(events.passed_strict.sum()); n_r = int(events.passed_relaxed.sum())
            cs = float(qc_df.loc[~qc_df.excluded, "clean_s"].sum())
            if cs > 0:
                print(f"  pooled rate: strict {n_s/cs:.3f} Hz | relaxed {n_r/cs:.3f} Hz")

    rep = swr_report.InclusionReport(
        "detection", analysis_name,
        f"s{session:02d}: bipolar derivations entering ripple detection.")
    for _, r in qc_df.iterrows():
        u = f"s{session:02d}/{r.pair_id}"
        if bool(r.excluded):
            rep.exclude(u, f">2/3 of the recording contaminated "
                           f"({r.contaminated_frac:.0%})", roi=r.pair_roi)
        elif float(r.get("clean_s", 0)) <= 0:
            rep.exclude(u, "no artifact-free time", roi=r.pair_roi)
        else:
            rep.include(u, "", roi=r.pair_roi, n_events=int(r.get("n_events", 0)),
                        clean_s=r.clean_s, rate_hz=r.get("rate_hz"))

    if save_all:
        out_dir = os.path.join(swr_io.session_deriv_dir(session, data_root),
                               "LFP-ripples", analysis_name)
        os.makedirs(out_dir, exist_ok=True)
        rep.write(out_dir)
        if len(events):
            events.to_csv(os.path.join(out_dir, "ripple_events.csv"), index=False)
        qc_df.to_csv(os.path.join(out_dir, "channel_qc.csv"), index=False)
        pd.DataFrame(iv_rows).to_csv(
            os.path.join(out_dir, "clean_intervals.csv"), index=False)
        pd.DataFrame(bad_rows).to_csv(
            os.path.join(out_dir, "artifact_intervals.csv"), index=False)
        with open(os.path.join(out_dir, "detector_diag.json"), "w") as f:
            json.dump(diags, f, indent=2, default=str)
        swr_io.write_settings(out_dir, _settings_dict(session, analysis_name, pad_s, clean_name))
        print(f"  saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(detect_session)
    else:
        detect_session(38)
