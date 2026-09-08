#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The bundle: ripples, intervals and channel QC for every session, from one place.

The cluster holds the raw recordings; the analysis of what the ripples MEAN does
not need them. `export_bundle` writes a few MB that reproduces every result in
this project on a laptop, and `RippleStore` reads either that bundle or the
per-session detection output behind one interface -- so the same analysis code
runs on the cluster and locally and cannot diverge.

    import mc.analyse.swr_bundle as swb_
    store = swb_.RippleStore(bundle="<bundle dir>")
    ev, intervals, qc = store.get(18)

@author: Svenja Kuchenhoff
"""

import os
import glob

import numpy as np
import pandas as pd

import mc.analyse.swr_io as swr_io

ANALYSIS_NAME = "swr_v1"


class RippleStore:
    """Where the detected ripples come from, behind one interface.

    Two sources:
      `sessions`  the per-session detection output on this machine. What the
                  cluster has.
      `bundle`    a bundle downloaded from the cluster, carrying the same three
                  tables for every session in a few MB. What the laptop has.

    The bundle exists so these statistics can be redone without moving the LFP.
    Everything that reads ripples goes through here, so both paths run the
    IDENTICAL analysis code -- the source cannot change a result.
    """

    def __init__(self, analysis_name=ANALYSIS_NAME, data_root=None, bundle=None):
        self.analysis_name = analysis_name
        self.R = data_root or swr_io.get_data_root()
        self.bundle_path = None
        if bundle is None:
            self.source = "sessions"
            paths = sorted(glob.glob(os.path.join(
                swr_io.derivatives_dir(self.R), "s*", "LFP-ripples",
                analysis_name, "ripple_events.csv")))
            self._sessions = [int(p.split(os.sep)[-4][1:]) for p in paths]
            self._dirs = {s: os.path.dirname(p)
                          for s, p in zip(self._sessions, paths)}
        else:
            self.source = "bundle"
            self._load_bundle(bundle)

    def _load_bundle(self, bundle):
        """`bundle` is the bundle directory, or the .pkl inside it."""
        if os.path.isdir(bundle):
            d = bundle
            tabs = {k: pd.read_csv(os.path.join(d, f"{k}.csv"))
                    for k in ("ripples", "intervals", "channel_qc")}
        else:
            import pickle
            with open(bundle, "rb") as f:
                b = pickle.load(f)
            d = os.path.dirname(bundle)
            tabs = {k: b[k] for k in ("ripples", "intervals", "channel_qc")}
        self.bundle_path = d
        self._rip = {s: g for s, g in tabs["ripples"].groupby("session")}
        self._iv = {s: g for s, g in tabs["intervals"].groupby("session")}
        self._qc = {s: g.set_index("pair_id")
                    for s, g in tabs["channel_qc"].groupby("session")}
        self._sessions = sorted(self._rip)
        # the bundle carries its own subject key, so a stale local manifest
        # cannot silently re-cluster the robust standard errors
        r = tabs["ripples"]
        self._subj = (r[["session", "subject_key", "recording_site"]]
                      .drop_duplicates("session").set_index("session")
                      .to_dict("index"))

    def sessions(self):
        return list(self._sessions)

    def get(self, sess):
        """(accepted events, artifact-free intervals, channel QC) for a session.

        Events are already filtered to those that passed detection, in both
        sources -- the bundle stores only accepted ripples.
        """
        sess = int(sess)
        if self.source == "bundle":
            return (self._rip[sess], self._iv[sess], self._qc[sess])
        d = self._dirs[sess]
        ev = pd.read_csv(os.path.join(d, "ripple_events.csv"))
        ev = ev[ev.passed.fillna(False)]
        iv = pd.read_csv(os.path.join(d, "clean_intervals.csv"))
        qc = pd.read_csv(os.path.join(d, "channel_qc.csv")).set_index("pair_id")
        return ev, iv, qc

    def subject_map(self):
        if self.source == "bundle":
            return self._subj
        m = pd.read_csv(os.path.join(swr_io.derivatives_dir(self.R), "group",
                                     "swr", "session_manifest.csv"))
        return m.set_index("session")[["subject_key",
                                       "recording_site"]].to_dict("index")

    def describe(self):
        n = sum(len(self._rip[s]) for s in self._sessions) \
            if self.source == "bundle" else None
        where = self.bundle_path or f"{swr_io.derivatives_dir(self.R)}/s*/LFP-ripples"
        print(f"  source: {self.source}  ({len(self._sessions)} sessions"
              + (f", {n} accepted ripples" if n is not None else "") + ")")
        print(f"          {where}")


def load_figure_data(bundle_dir, out_name="swr_bundle"):
    """(arrays, index) for methods figures, from a downloaded bundle.

        arrays, idx = swb_.load_figure_data("<bundle dir>")
        row = idx[(idx.session == 21) & (idx["rank"] == 3)].iloc[0]
        trace = arrays[row.key_raw][row["rank"] - 1]      # one sharp-wave example
        fs    = float(arrays[f"s{row.session:02d}_sw_fs"])

    `rank` is 1-based and indexes into the stacked array in the same order.
    """
    import pickle
    z = np.load(os.path.join(bundle_dir, f"{out_name}_figures.npz"),
                allow_pickle=False)
    arrays = {k: z[k] for k in z.files}
    idx = pd.DataFrame()
    pkl = os.path.join(bundle_dir, f"{out_name}.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            idx = pickle.load(f).get("figure_index", pd.DataFrame())
    return arrays, idx


def collect_figure_data(sessions, analysis_name=ANALYSIS_NAME):
    """Per-session waveform arrays for METHODS FIGURES, small enough to travel.

    The statistics only need event times, but a methods figure needs the traces,
    and fetching those file-by-file from the cluster does not work in practice.
    So they ride along in the bundle.

    Only condensed arrays are taken, never per-event stacks:

        mean       ripple-triggered mean waveform, ONE PER DERIVATION -- keeps
                   the option of a per-ROI or per-site grand average locally
        tfr_mean   ripple-triggered TFR averaged across the session's
                   derivations. The per-derivation TFR is (n_pairs, n_freq,
                   n_time) and would dominate the bundle; the session mean is
                   ~0.1 MB and is what the pooled figure uses anyway
        ex_*       the single clearest ripple the QC report found, raw,
                   band-passed and as a TFR
        sw_*       the sharp-wave example candidates (`qc_report examples`),
                   which are chosen by eye for the publication figure
        art_*      the artifact panels' inputs: one interictal discharge, one
                   ripple from the same derivation, and the representative
                   excerpt with each criterion's crossings and the final padded
                   mask. Without these the two artifact figures cannot be
                   redrawn off-cluster at all -- they are the only figures that
                   need the raw trace rather than event-triggered averages

    Returns (arrays, index) -- a dict of numeric arrays keyed `s{NN}_{what}`,
    and a DataFrame naming what each sharp-wave candidate is, since the labels
    are strings and do not belong in an .npz.
    """
    import mc.plotting.ripple_figures as rfig

    arrays, index, index_art = {}, [], []
    for sess, rip_dir in sessions:
        tag = f"s{sess:02d}"
        sw = os.path.join(rip_dir, "sharpwave_examples_best.npz")
        if os.path.exists(sw):
            z = np.load(sw, allow_pickle=True)
            arrays[f"{tag}_sw_raw"] = np.asarray(z["raw"], np.float32)
            arrays[f"{tag}_sw_bip"] = np.asarray(z["bip"], np.float32)
            arrays[f"{tag}_sw_fs"] = np.asarray(z["fs"], float)
            for k in range(len(z["raw"])):
                index.append({"session": sess, "rank": k + 1,
                              "pair_id": str(z["pair_id"][k]),
                              "contact": str(z["contact"][k]),
                              "t_peak_s": float(z["t_peak_s"][k]),
                              "score": float(z["score"][k]),
                              "key_raw": f"{tag}_sw_raw",
                              "key_bip": f"{tag}_sw_bip"})
        art = os.path.join(rip_dir, "artifact_examples.npz")
        if os.path.exists(art):
            z = np.load(art, allow_pickle=True)
            # Masks are stored as bool: an int8 cast would quadruple them for
            # no gain, and there are five per session.
            for k in z.files:
                if k in ("session", "pair_id"):
                    continue
                v = np.asarray(z[k])
                if v.dtype == bool:
                    arrays[f"{tag}_art_{k}"] = v
                else:
                    arrays[f"{tag}_art_{k}"] = v.astype(
                        np.float32 if v.ndim else float)
            index_art.append({"session": sess, "pair_id": str(z["pair_id"]),
                              "t0_s": float(z["t0_s"]), "fs": float(z["fs"]),
                              "key_prefix": f"{tag}_art_"})

        st = os.path.join(rip_dir, "ripple_stacks.npz")
        if os.path.exists(st):
            z = np.load(st, allow_pickle=True)
            if "mean" in z:
                arrays[f"{tag}_mean"] = np.asarray(z["mean"], np.float32)
            if "tfr" in z and np.size(z["tfr"]):
                arrays[f"{tag}_tfr_mean"] = np.asarray(z["tfr"], float).mean(0).astype(np.float32)
            for k in ("t_ms", "n_events", "ex_raw", "ex_bp", "ex_tfr", "fs"):
                if k in z and np.size(z[k]):
                    arrays[f"{tag}_{k}"] = np.asarray(z[k], np.float32)
    arrays["tfr_freqs"] = np.asarray(rfig.TFR_FREQS, float)
    return arrays, pd.DataFrame(index), pd.DataFrame(index_art)


def export_bundle(analysis_name=ANALYSIS_NAME, data_root=None,
                  out_name="swr_bundle", out_dir=None):
    """Everything needed to redo any of these statistics WITHOUT the LFP.

    The cluster holds the raw recordings; the analysis of what the ripples mean
    does not need them. This writes one pickle -- and the same tables as CSVs,
    so it is readable without this repo -- containing, for every session:

        ripples    one row per accepted ripple: session, subject, pair, ROI,
                   MNI, t_peak_s, duration, peak frequency, amplitude
        intervals  the artifact-free intervals per derivation. REQUIRED, not
                   optional: a rate is ripples per artifact-free second, and any
                   window analysis without them is wrong
        pairs      the bipolar derivations with their coordinates
        behaviour  all_trial_times per session, with phase labels
        uncover    every uncovering attempt with its outcome, in session seconds
        channel_qc per-derivation counts, clean time and exclusion flags

    A few MB, against tens of GB of LFP. This is the file to bring home.

    Lived in `scripts/archived/swr_hypotheses.py` until 2026-09-07; it is here
    now because it is the one part of that script still in the pipeline, and
    `HOW_TO_RUN` was pointing users at an archived file.
    """
    import json
    import pickle
    from datetime import datetime

    import mc.analyse.swr_behaviour as swb
    import mc.analyse.swr_windows as win

    R = data_root or swr_io.get_data_root()
    out_dir = out_dir or os.path.join(swr_io.derivatives_dir(R), "group",
                                      "swr", "bundle")
    os.makedirs(out_dir, exist_ok=True)

    man = pd.read_csv(os.path.join(swr_io.derivatives_dir(R), "group", "swr",
                                   "session_manifest.csv"))
    subj = man.set_index("session")[["subject_key", "recording_site"]].to_dict("index")

    paths = sorted(glob.glob(os.path.join(
        swr_io.derivatives_dir(R), "s*", "LFP-ripples", analysis_name,
        "ripple_events.csv")))
    sessions = [(int(p.split(os.sep)[-4][1:]), os.path.dirname(p)) for p in paths]

    rip, iv, pr, beh_all, unc_all, qc_all = [], [], [], [], [], []
    for sess, rip_dir in sessions:
        meta = subj.get(sess, {})
        clean_dir = os.path.join(swr_io.session_deriv_dir(sess, R), "LFP-clean",
                                 analysis_name)
        try:
            e = pd.read_csv(os.path.join(rip_dir, "ripple_events.csv"))
            e = e[e.passed.fillna(False)]
            keep = [c for c in ("pair_id", "t_peak_s", "duration_s",
                                "peak_freq_hz", "amp_peak_uv", "rms_peak_z",
                                "spectral_passed_strict",
                                "spectral_passed_relaxed")
                    if c in e.columns]
            e = e[keep].copy()
            e["session"] = sess
            e["subject_key"] = meta.get("subject_key")
            e["recording_site"] = meta.get("recording_site")
            rip.append(e)

            i = pd.read_csv(os.path.join(rip_dir, "clean_intervals.csv"))
            i["session"] = sess
            iv.append(i)

            q = pd.read_csv(os.path.join(rip_dir, "channel_qc.csv"))
            q["session"] = sess
            qc_all.append(q)

            pr.append(pd.read_csv(os.path.join(clean_dir, "pairs.csv")))
        except FileNotFoundError as err:
            print(f"  s{sess:02d}: {err}")
            continue

        try:
            b = win.add_phase3(win.add_phase(swr_io.load_behaviour(sess, data_root=R)))
            b["session"] = sess
            b["subject_key"] = meta.get("subject_key")
            beh_all.append(b)
            u = swb.uncover_events(sess, data_root=R)
            if len(u):
                u["subject_key"] = meta.get("subject_key")
                unc_all.append(u)
        except Exception as err:
            print(f"  s{sess:02d}: behaviour skipped "
                  f"({type(err).__name__}: {err})")

    def cat(x):
        return pd.concat(x, ignore_index=True) if x else pd.DataFrame()

    fig_arrays, fig_index, fig_index_art = collect_figure_data(
        sessions, analysis_name)

    bundle = {"ripples": cat(rip), "intervals": cat(iv), "pairs": cat(pr),
              "figure_index": fig_index,
              "figure_index_artifact": fig_index_art,
              "behaviour": cat(beh_all), "uncover": cat(unc_all),
              "channel_qc": cat(qc_all),
              "meta": {"analysis_name": analysis_name,
                       "created": datetime.now().isoformat(timespec="seconds"),
                       "data_root": R,
                       "n_sessions": len(sessions),
                       "figure_data": f"{out_name}_figures.npz",
                       "note": "rates must use intervals for exposure; a ripple "
                               "rate is events per ARTIFACT-FREE second"}}

    fig_path = os.path.join(out_dir, f"{out_name}_figures.npz")
    np.savez_compressed(fig_path, **fig_arrays)

    with open(os.path.join(out_dir, f"{out_name}.pkl"), "wb") as f:
        pickle.dump(bundle, f, protocol=4)
    for k, v in bundle.items():
        if isinstance(v, pd.DataFrame) and len(v):
            v.to_csv(os.path.join(out_dir, f"{k}.csv"), index=False)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(bundle["meta"], f, indent=2)

    print("\n" + "=" * 74)
    print(" EXPORT BUNDLE")
    print("=" * 74)
    for k, v in bundle.items():
        if isinstance(v, pd.DataFrame):
            print(f"  {k:12s} {len(v):7d} rows")
    mb = os.path.getsize(fig_path) / 1e6
    n_sw = len(fig_index)
    print(f"  {'figure_data':12s} {len(fig_arrays):7d} arrays  ({mb:.1f} MB, "
          f"{n_sw} sharp-wave candidates, "
          f"{len(fig_index_art)} artifact excerpts)")
    print(f"\nSaved -> {out_dir}")
    print(f"  bring home: {out_name}.pkl + {out_name}_figures.npz")
    return bundle
