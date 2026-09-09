#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 3 of the SWR pipeline: continuous preprocessing, per session.

raw -> 1000 Hz (resample_poly) -> bipolar -> notch 60/120/180, written out on
the behavioural clock so that `sample = round(t_session * fs)`.

Separate from detection on purpose: this stage is I/O-bound on multi-GB raw
files, while detection is CPU-bound and gets re-run whenever a detector
parameter changes. Fusing them would force a re-read of the raw data on every
parameter tweak (~4 h across all sessions instead of ~15 min).

Outputs (per session, keyed by ANALYSIS_NAME rather than a timestamp so a
60-task array job produces one joinable set):
    derivatives/s{XX}/LFP-clean/{name}/continuous.npy    float32 (n_pairs, n_samples)
    derivatives/s{XX}/LFP-clean/{name}/pairs.csv
    derivatives/s{XX}/LFP-clean/{name}/meta.json
    derivatives/s{XX}/LFP-clean/{name}/settings.json
    derivatives/s{XX}/LFP-clean/{name}/qc_psd.png

Usage:
    python scripts/swr_extract_continuous.py --session=38
    python scripts/swr_extract_continuous.py --session=38 --analysis_name=swr_v1

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

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "swr_v1"

# A behavioural event may fall at most this far outside its recording before
# the block is rejected. Three sessions (s09, s10, s33) overrun by 0.9-2.0 s,
# always by exactly one trailing repeat, because the amplifier was stopped just
# before the final trial completed.
CLOCK_TOLERANCE_S = 5.0


def _settings_dict(session, analysis_name):
    return {
        "analysis_name": analysis_name,
        "session": int(session),
        "target_fs": pp.TARGET_FS,
        "notch_hz": list(pp.LINE_FREQS),
        "notch_q": pp.NOTCH_Q,
        "resample": "scipy.signal.resample_poly (NOT scipy.signal.resample)",
        "montage": "bipolar, one pair per probe, anchor + immediate neighbour",
        "clock": "behavioural = cumulative file duration",
        "clock_tolerance_s": CLOCK_TOLERANCE_S,
        "chunk_s": pp.CHUNK_S,
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def _qc_psd(sig, fs, pair_ids, out_png, notch_hz=None):
    """Welch PSD of every derivation, with the notch frequencies marked.

    The check that matters: no residual peak at 60/120/180, and no edge
    artifact from resampling.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    f, P = welch(sig, fs=fs, nperseg=int(4 * fs), axis=-1)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for i in range(P.shape[0]):
        ax.semilogy(f, P[i], lw=0.7, alpha=0.75)
    for lf in pp.LINE_FREQS:
        ax.axvline(lf, color="0.5", ls=":", lw=0.8)
    ax.axvspan(80, 120, color="#F15A29", alpha=0.12, label="ripple band 80–120 Hz")
    ax.set_xlim(0, 250)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD ($\mu$V$^2$/Hz)")
    # Say what actually happened. The notch is adaptive, so on a session with no
    # line noise nothing is removed -- titling that "after notch" made it look
    # as though a filter had failed rather than not been needed.
    if notch_hz:
        applied = ", ".join(f"{float(f):.0f}" for f in notch_hz)
        note = f"notch applied at {applied} Hz"
    else:
        note = "no notch needed (line-noise ratio below threshold)"
    ax.set_title(f"Bipolar derivations, {note} (n={P.shape[0]})", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    # residual line noise: power at the notch vs the local neighbourhood
    res = {}
    for lf in pp.LINE_FREQS:
        i = int(np.argmin(np.abs(f - lf)))
        side = np.r_[P[:, max(0, i - 12):i - 4], P[:, i + 5:i + 13]]
        res[f"{lf:.0f}Hz_ratio"] = float(np.median(P[:, i]) / max(np.median(side), 1e-30))
    return res


def extract_session(session, analysis_name=ANALYSIS_NAME, save_all=True,
                    verbose=True):
    swr_io.start_log(os.path.join(swr_io.session_deriv_dir(int(session), swr_io.get_data_root()), "LFP-clean", analysis_name), "swr_extract_continuous")
    session = int(session)
    data_root = swr_io.get_data_root()

    pairs_p = os.path.join(swr_io.session_deriv_dir(session, data_root),
                           "LFP", f"bipolar_pairs_{session:02d}.csv")
    if not os.path.isfile(pairs_p):
        raise FileNotFoundError(
            f"{pairs_p} not found -- run scripts/swr_build_contacts.py first")
    pairs = pd.read_csv(pairs_p)
    if not len(pairs):
        print(f"s{session:02d}: no bipolar pairs, nothing to do")
        return None

    # ---- hard gate: does every behavioural event land inside its recording?
    blocks, bt, warn = pp.session_block_table(session, data_root=data_root)
    print(f"\ns{session:02d}: {len(blocks)} block(s), {len(pairs)} pairs")
    for w in warn:
        print(f"  note: {w}")
    # Emptiness first: the empty frame carries no `block`/`offset_s` columns,
    # so printing the table before this check turns a clean skip into a KeyError.
    if not len(blocks):
        reason = "no readable recording -- see the notes above"
        print(f"  SKIPPED s{session:02d}: {reason}")
        if save_all:
            out_dir = os.path.join(swr_io.session_deriv_dir(session, data_root),
                                   "LFP-clean", analysis_name)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "SKIPPED.json"), "w") as f:
                json.dump({"session": session, "reason": reason,
                           "warnings": warn}, f, indent=2)
        return None
    print(blocks[["block", "duration_s", "fs_raw", "offset_s",
                  "head_margin_s", "tail_margin_s"]].to_string(index=False))

    # The count gate that used to sit here -- "n recordings vs m behavioural
    # blocks, recording gap inside a block" -- skipped seven sessions, and for
    # five of them the diagnosis was simply wrong: the directory held
    # recordings that are not part of the task (another day's session, an
    # intervening task, an aborted start, a 4.3 s stub). `resolve_run` now
    # picks the run that carries the behaviour, so a count mismatch is normal
    # and no longer interesting.
    #
    # A short recording is a different thing and is NOT fatal. The trailing
    # windows get exposure_s = 0 and are dropped by the GLM, exactly as the
    # already-tolerated sub-5 s truncations are. Raising here instead threw
    # away whole sessions over a fraction of their length: s58 loses 1.2%,
    # s62 1.6%, s32 10%, s63 34%.
    head = float(np.nanmin(blocks.head_margin_s.astype(float)))
    tail = float(np.nanmin(blocks.tail_margin_s.astype(float)))
    truncated_s = max(0.0, -tail) + max(0.0, -head)
    if truncated_s > 0:
        span = float(bt.beh_end_s.max() - bt.beh_start_s.min())
        print(f"  TRUNCATED s{session:02d}: {truncated_s:.1f}s of behaviour "
              f"({100 * truncated_s / span:.1f}% of the task) has no recording "
              f"(head {head:.1f}s, tail {tail:.1f}s). Those windows get zero "
              f"exposure; the session is carried as a partial.")

    sig, meta = pp.preprocess_session(session, pairs, data_root=data_root,
                                      verbose=verbose)
    meta["behaviour_blocks"] = bt.to_dict("records")
    meta["truncated_s"] = truncated_s          # behaviour with no recording
    meta["discovery_warnings"] = warn

    print(f"  -> {sig.shape[0]} pairs x {sig.shape[1]} samples "
          f"({sig.shape[1]/pp.TARGET_FS:.0f}s) "
          f"= {sig.nbytes/1e6:.0f} MB float32")

    if save_all:
        out_dir = os.path.join(swr_io.session_deriv_dir(session, data_root),
                               "LFP-clean", analysis_name)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "continuous.npy"), sig)
        pairs.to_csv(os.path.join(out_dir, "pairs.csv"), index=False)

        # The before/after spectra, for the adaptive-notch figure. Popped from
        # meta because it holds arrays and meta is written as JSON.
        psd = meta.pop("_psd", None)
        if psd is not None:
            np.savez_compressed(
                os.path.join(out_dir, "notch_psd.npz"),
                pair_ids=np.asarray(list(pairs.pair_id), dtype=object),
                recording_site=str(meta.get("recording_site")),
                session=int(session),
                notch_applied_hz=np.asarray(
                    sorted(float(k) for k in meta.get("notch_applied_hz", {}))),
                # how many derivations were notched at each harmonic -- the
                # decision is per derivation now, so "notched" is a count
                n_notched=np.asarray(
                    [meta.get("notch_applied_hz", {}).get(f"{f:.0f}", 0)
                     for f in pp.LINE_FREQS], dtype=int),
                n_pairs=int(sig.shape[0]),
                # every measured ratio, not only the notched ones: the point
                # of the figure is that some harmonics were left alone, and
                # that is only legible next to the ratio that spared them.
                # Column 1 is the MEDIAN across derivations; the full
                # per-derivation set is in `ratios_per_pair`.
                ratios=np.asarray(
                    [[float(k), float(v)] for k, v in
                     sorted(meta.get("line_noise_ratio_median", {}).items(),
                            key=lambda kv: float(kv[0]))],
                    dtype=float).reshape(-1, 2),
                ratios_per_pair=np.asarray(
                    [meta.get("line_noise_ratio", {}).get(f"{f:.0f}",
                                                          [np.nan] * sig.shape[0])
                     for f in pp.LINE_FREQS], dtype=float),
                ratio_threshold=float(meta.get("notch_ratio_threshold", 2.0)),
                **psd)

        res = _qc_psd(sig, pp.TARGET_FS, list(pairs.pair_id),
                      os.path.join(out_dir, "qc_psd.png"),
                      notch_hz=meta.get("notch_applied_hz"))
        meta["line_noise_residual"] = res
        print("  residual line noise (1.0 = fully removed): "
              + ", ".join(f"{k}={v:.2f}" for k, v in res.items()))

        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        swr_io.write_settings(out_dir, _settings_dict(session, analysis_name))
        print(f"  saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(extract_session)
    else:
        extract_session(38)
