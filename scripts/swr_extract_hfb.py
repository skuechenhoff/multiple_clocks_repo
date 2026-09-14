#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per session: high-frequency broadband and band envelopes for EVERY derivation,
hippocampal and cortical alike.

This is the cortical half of the SWR pipeline -- the stage that makes
hippocampus-to-mPFC questions askable. It sits after extraction and beside
detection: detection turns the hippocampal derivations into ripple times, this
turns every derivation into a continuous power time course on the same clock.

Deliberately NOT epoched. Everything here stays continuous at 100 Hz so that
the peri-ripple window, the baseline, the phase split and the task alignment
are all still free choices when the bundle reaches the laptop. Epoching on the
cluster would bake in exactly the parameters that are still open questions, and
re-running this stage costs a cluster job rather than a re-read of the raw
files.

Inputs   derivatives/s{XX}/LFP-clean/{clean_name}/continuous.npy + pairs.csv + meta.json
Outputs  derivatives/s{XX}/LFP-hfb/{name}/hfb.npz          float32 (n_pairs, n_samples_100Hz)
         derivatives/s{XX}/LFP-hfb/{name}/hfb_pairs.csv
         derivatives/s{XX}/LFP-hfb/{name}/hfb_intervals.csv   artifact-free, PER DERIVATION
         derivatives/s{XX}/LFP-hfb/{name}/meta.json, settings.json, qc_hfb.png

Usage:
    python scripts/swr_extract_hfb.py --session=38
    python scripts/swr_extract_hfb.py --session=38 --analysis_name=swr_v2 --clean_name=swr_v2

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
import mc.analyse.swr_artifact as art
import mc.analyse.swr_hfb as hfb

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "swr_v2"


def _settings_dict(session, analysis_name, clean_name, pad_s):
    return {
        "analysis_name": analysis_name,
        "clean_name": clean_name,
        "session": int(session),
        "out_fs": hfb.OUT_FS,
        "hfb_band": list(hfb.HFB_BAND),
        "hfb_sub_bw": hfb.HFB_SUB_BW,
        "hfb_recipe": ("per sub-band Hilbert amplitude, log, z-scored on "
                       "artifact-free samples, then averaged across sub-bands"),
        "notch_guard_hz": hfb.NOTCH_GUARD_HZ,
        "bands": {k: list(v) for k, v in hfb.BANDS.items()},
        "phase_bands": list(hfb.PHASE_BANDS),
        "artifact_pad_s": pad_s,
        "exclusion_pad_s": art.EXCLUSION_PAD_S,
        "artifact_iqr_k": art.IQR_K,
        "epoched": False,
        "epoching_note": ("continuous on purpose -- windows stay free choices "
                          "on the laptop"),
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def _qc_figure(arrs, pairs, out_png, out_fs):
    """What each derivation's HFB actually looks like: its distribution and a
    60 s excerpt. A flat trace or an all-zero row is the failure this catches,
    and it is invisible in any summary number."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = np.asarray(arrs["hfb"], np.float32)
    n = h.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 0.34 * n + 2.4))
    ax = axes[0]
    ax.boxplot([h[i][np.isfinite(h[i])] for i in range(n)], vert=False,
               widths=0.6, showfliers=False)
    ax.set_yticklabels([f"{r.pair_id[:20]} [{r.roi_family}]"
                        for _, r in pairs.iterrows()], fontsize=7)
    ax.set_xlabel("HFB (z, log sub-band mean)")
    ax.set_title("HFB distribution per derivation", fontsize=10)
    ax.axvline(0, color="0.6", lw=0.8, ls=":")

    ax = axes[1]
    k = min(int(60 * out_fs), h.shape[1])
    t = np.arange(k) / out_fs
    for i in range(n):
        ax.plot(t, h[i, :k] + 4.0 * i, lw=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([4.0 * i for i in range(n)])
    ax.set_yticklabels([r.roi_family for _, r in pairs.iterrows()], fontsize=7)
    ax.set_title("first 60 s, offset per derivation", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _peri_ripple_qc(arrs, pairs, session, analysis_name, data_root, out_fs,
                    half_s=1.0):
    """Positive control, run here rather than discovered on the laptop.

    Hippocampal HFB MUST rise at its own ripple peaks -- that is the most
    reproducible fact about ripples there is. If it does not, the LFP and the
    event times are not on the same clock and every cortical number downstream
    is meaningless. Same logic as the 'read the HC panel first' rule in
    swr_ripple_triggered_units.py.

    Returns None when detection has not run yet; this stage does not depend on
    it.
    """
    rip_p = os.path.join(swr_io.session_deriv_dir(session, data_root),
                         "LFP-ripples", analysis_name, "ripple_events.csv")
    if not os.path.isfile(rip_p):
        return None
    ev = pd.read_csv(rip_p)
    if "passed" in ev:
        ev = ev[ev.passed.astype(bool)]
    if not len(ev):
        return None

    h = np.asarray(arrs["hfb"], np.float32)
    w = int(round(half_s * out_fs))
    out = {}
    for i, (_, p) in enumerate(pairs.iterrows()):
        t = ev.loc[ev.pair_id == p.pair_id, "t_peak_s"].to_numpy(float) \
            if p.role == "ripple" else ev.t_peak_s.to_numpy(float)
        if not len(t):
            continue
        idx = np.round(t * out_fs).astype(int)
        idx = idx[(idx - w >= 0) & (idx + w < h.shape[1])]
        if not len(idx):
            continue
        stack = np.stack([h[i, j - w:j + w] for j in idx])
        peri = float(np.nanmean(stack[:, w - int(0.25 * out_fs):
                                         w + int(0.25 * out_fs)]))
        flank = float(np.nanmean(np.c_[stack[:, :int(0.25 * out_fs)],
                                       stack[:, -int(0.25 * out_fs):]]))
        out[p.pair_id] = {"roi_family": p.roi_family, "role": p.role,
                          "n_ripples": int(len(idx)),
                          "peri_250ms": round(peri, 4),
                          "flank_250ms": round(flank, 4),
                          "peri_minus_flank": round(peri - flank, 4)}
    return out


def extract_hfb(session, analysis_name=ANALYSIS_NAME, clean_name=None,
                pad_s=None, save_all=True, verbose=True):
    session = int(session)
    clean_name = analysis_name if clean_name is None else str(clean_name)
    pad_s = art.PAD_S if pad_s is None else float(pad_s)
    data_root = swr_io.get_data_root()
    out_dir = os.path.join(swr_io.session_deriv_dir(session, data_root),
                           "LFP-hfb", analysis_name)
    swr_io.start_log(out_dir, "swr_extract_hfb")

    clean_dir = os.path.join(swr_io.session_deriv_dir(session, data_root),
                             "LFP-clean", clean_name)
    sig_p = os.path.join(clean_dir, "continuous.npy")
    if not os.path.isfile(sig_p):
        print(f"s{session:02d}: no continuous.npy under LFP-clean/{clean_name} "
              f"-- run swr_extract_continuous.py first")
        return None

    sig = np.load(sig_p, mmap_mode='r')
    pairs = pd.read_csv(os.path.join(clean_dir, "pairs.csv"))
    with open(os.path.join(clean_dir, "meta.json")) as f:
        meta = json.load(f)
    fs = float(meta["fs"])

    # Older contact tables have no role/roi_family; treat them as all-ripple so
    # this stage still runs on a hippocampus-only extraction.
    if "role" not in pairs.columns:
        pairs["role"] = "ripple"
    if "roi_family" not in pairs.columns:
        pairs["roi_family"] = pairs.get("pair_roi_atlas", "unknown")

    n_out = int(np.ceil(sig.shape[1] * hfb.OUT_FS / fs))
    print(f"\ns{session:02d}: {sig.shape[0]} derivations, "
          f"{sig.shape[1]/fs:.0f}s @ {fs:.0f}Hz -> {hfb.OUT_FS:.0f}Hz  "
          f"[pad {pad_s:.2f}s]")
    print("  " + pairs.roi_family.value_counts().to_string().replace("\n", "\n  "))

    keys = ["hfb"] + list(hfb.BANDS) + [f"{b}_phase" for b in hfb.PHASE_BANDS]
    arrs = {k: np.full((len(pairs), n_out), np.nan, hfb.STORE_DTYPE)
        for k in keys}
    rows, iv_rows, infos = [], [], {}

    for i, (_, p) in enumerate(pairs.iterrows()):
        x = np.asarray(sig[i], float)
        per = art.criteria_masks(x, fs)
        bad = art.combine_criteria(per, fs, pad_s=pad_s)
        clean = ~bad
        contam_excl = float(art.combine_criteria(
            per, fs, pad_s=art.EXCLUSION_PAD_S).mean())

        drop = hfb.notched_harmonics_for_pair(meta, str(p.pair_id))
        h, info = hfb.hfb_amplitude(x, fs, clean=clean, drop_hz=drop)
        m = min(len(h), n_out)
        arrs["hfb"][i, :m] = h[:m]
        for name, (lo, hi) in hfb.BANDS.items():
            v = hfb.band_envelope(x, fs, lo, hi, clean=clean)
            arrs[name][i, :len(v)] = v[:n_out]
        for name in hfb.PHASE_BANDS:
            lo, hi = hfb.BANDS[name]
            v = hfb.band_phase(x, fs, lo, hi)
            arrs[f"{name}_phase"][i, :len(v)] = v[:n_out]

        iv = art.clean_intervals(bad, fs)
        clean_s = float(np.diff(iv, axis=1).sum()) if len(iv) else 0.0
        for a, b in iv:
            iv_rows.append({"pair_id": p.pair_id, "start_s": a, "stop_s": b})

        infos[str(p.pair_id)] = info
        rows.append({
            "session": session, "pair_id": p.pair_id,
            "roi_family": p.roi_family, "role": p.role,
            "pair_roi_atlas": p.get("pair_roi_atlas"),
            "pair_roi": p.get("pair_roi"),
            "hemisphere": p.get("hemisphere"),
            "subject_label": p.get("subject_label"),
            "mni_x": p.get("mni_x"), "mni_y": p.get("mni_y"),
            "mni_z": p.get("mni_z"),
            "n_sub_bands": info["n_sub_bands"],
            "notched_hz": ",".join(f"{f:.0f}" for f in drop),
            "clean_s": round(clean_s, 1),
            "contaminated_frac": round(float(bad.mean()), 4),
            "contaminated_frac_at_exclusion_pad": round(contam_excl, 4),
            # Same fixed-pad rule as detection, so a derivation is in or out of
            # the sample for the same reason in both branches.
            "excluded": contam_excl > art.MAX_CONTAM_FRAC,
        })
        if verbose:
            print(f"  {str(p.pair_id)[:26]:26s} {p.roi_family:12s} "
                  f"sub_bands={info['n_sub_bands']} "
                  f"notched={drop if drop else '-'} clean={clean_s:6.0f}s")

    qc = pd.DataFrame(rows)
    print(f"\n  {len(qc)} derivations, {int(qc.excluded.sum())} excluded, "
          f"{qc.loc[~qc.excluded,'clean_s'].sum()/3600:.2f} h clean")
    if (qc.n_sub_bands < 8).any():
        print(f"  {int((qc.n_sub_bands < 8).sum())} derivation(s) lost sub-bands "
              f"to the adaptive notch -- HFB is still comparable (z per band) "
              f"but n_sub_bands is carried per derivation, use it as a covariate")

    peri = _peri_ripple_qc(arrs, pairs, session, analysis_name, data_root,
                           hfb.OUT_FS)
    if peri:
        print("\n  POSITIVE CONTROL -- peri-ripple HFB (+-250ms) minus flanks:")
        n_hpc_ok = 0
        for pid, v in peri.items():
            flag = ""
            if v.get("role") == "ripple":
                ok = v["peri_minus_flank"] > 0
                n_hpc_ok += int(ok)
                flag = ("  <- HC, must be > 0  OK" if ok else
                        "  <- HC, NEGATIVE: check the clock before "
                        "trusting any cortical number")
            print(f"    {pid[:26]:26s} {str(v['roi_family'])[:15]:15s} "
                  f"n={v['n_ripples']:5d}  {v['peri_minus_flank']:+.3f}{flag}")
        n_hpc = sum(1 for v in peri.values() if v.get("role") == "ripple")
        if n_hpc:
            print(f"  -> {n_hpc_ok}/{n_hpc} hippocampal derivations positive"
                  + ("" if n_hpc_ok == n_hpc else
                     "   ⛔ INVESTIGATE before using the cortical rows"))
    else:
        print("\n  [peri-ripple positive control skipped: detection has not run "
              "for this analysis_name yet -- re-run this stage after it to get it]")

    if save_all:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "hfb.npz"),
                            pair_ids=np.asarray(list(pairs.pair_id), dtype=object),
                            out_fs=np.float64(hfb.OUT_FS), **arrs)
        qc.to_csv(os.path.join(out_dir, "hfb_pairs.csv"), index=False)
        pd.DataFrame(iv_rows).to_csv(
            os.path.join(out_dir, "hfb_intervals.csv"), index=False)
        meta_out = {"session": session, "out_fs": hfb.OUT_FS,
                    "dtype": str(np.dtype(hfb.STORE_DTYPE)),
                    "n_samples": n_out, "fs_in": fs,
                    "duration_s": float(sig.shape[1] / fs),
                    "arrays": keys, "per_pair": infos,
                    "peri_ripple_qc": peri}
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta_out, f, indent=2, default=str)
        swr_io.write_settings(out_dir, _settings_dict(
            session, analysis_name, clean_name, pad_s))
        try:
            _qc_figure(arrs, pairs, os.path.join(out_dir, "qc_hfb.png"),
                       hfb.OUT_FS)
        except Exception as e:
            print(f"  [qc figure skipped: {type(e).__name__}: {e}]")
        sz = os.path.getsize(os.path.join(out_dir, "hfb.npz")) / 1e6
        print(f"  saved -> {out_dir}  (hfb.npz {sz:.0f} MB)")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(extract_hfb)
    else:
        extract_hfb(38)
