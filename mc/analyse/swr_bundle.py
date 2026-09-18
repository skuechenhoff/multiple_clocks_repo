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


def repad_bundle(bundle, pad_s):
    """Re-impose an artifact pad on a loaded bundle, at home, without the LFP.

    The bundle stores artifact intervals UNPADDED (`artifact_intervals`) and the
    distance from every ripple to the nearest one (`dist_to_artifact_s`), so the
    pad is a laptop-side decision rather than something fixed on the cluster.

    BOTH halves are required and this does both:

      numerator    events are filtered on `dist_to_artifact_s >= pad_s`
      denominator  exposure is rebuilt with `clean_intervals_at_pad`

    Filtering events alone inflates the rate, because a rate is events per
    artifact-free second and the denominator has to shrink with the numerator.
    Verified: at pad_s = 1.0 this reproduces the cluster-side run to 0.2%
    (64,895 vs 64,760 ripples, 91.4 vs 91.2 h).

    `channel_qc.excluded` is deliberately NOT recomputed. It is decided at a
    fixed reference pad (`contaminated_frac_at_exclusion_pad`) so that the set
    of derivations stays constant as the analysis pad changes; otherwise a pad
    sweep would confound the pad with the sample.
    """
    import mc.analyse.swr_artifact as swr_artifact

    need = {'artifact_intervals', 'ripples', 'channel_qc'}
    missing = need - set(bundle)
    if missing:
        raise KeyError(f"bundle cannot be re-padded, missing {sorted(missing)}. "
                       f"This needs a swr_v2 bundle or later.")
    rips = bundle['ripples']
    if 'dist_to_artifact_s' not in rips:
        raise KeyError("ripples has no `dist_to_artifact_s`; re-export the bundle")

    qc = bundle['channel_qc']
    duration = {}
    for _, q in qc.iterrows():
        frac = float(q.contaminated_frac)
        duration[(int(q.session), str(q.pair_id))] = (
            float(q.clean_s) / max(1e-9, 1.0 - frac))

    rows = []
    for (sess, pair), g in bundle['artifact_intervals'].groupby(['session', 'pair_id']):
        dur = duration.get((int(sess), str(pair)))
        if dur is None:
            continue
        iv = swr_artifact.clean_intervals_at_pad(
            g[['start_s', 'stop_s']].to_numpy(float), dur, 1000.0, pad_s=pad_s)
        for a, b in iv:
            rows.append((int(sess), str(pair), float(a), float(b)))

    out = dict(bundle)
    out['ripples'] = rips[rips.dist_to_artifact_s >= pad_s].copy()
    out['intervals'] = pd.DataFrame(rows, columns=['session', 'pair_id',
                                                   'start_s', 'stop_s'])
    out['meta'] = dict(bundle.get('meta', {}))
    out['meta']['repadded_to_s'] = float(pad_s)
    return out


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


def collect_hfb(analysis_name, data_root, out_dir, arrays=None):
    """Copy each session's continuous HFB store into the bundle.

    Kept as one .npz PER SESSION next to the pickle rather than inside it. The
    HFB is orders of magnitude larger than everything else in the bundle, and
    folding it into the pickle would mean the ripple tables -- which are a few
    MB and are what most analyses need -- could no longer be loaded without it.
    This way `swr_bundle.pkl` stays laptop-sized and `bundle/hfb/` is an
    optional second download.

    `arrays` selects which of hfb/ripple/theta/beta/theta_phase to carry. None
    means all of them: the point of this stage is that the windows and the
    bands are still open questions when the data reaches the laptop, so the
    default does not decide either.
    """
    hfb_dir = os.path.join(out_dir, "hfb")
    os.makedirs(hfb_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(
        swr_io.derivatives_dir(data_root), "s*", "LFP-hfb", analysis_name,
        "hfb.npz")))
    idx, qc_rows, iv_rows, total = [], [], [], 0
    for p in paths:
        sess = int(p.split(os.sep)[-4][1:])
        src = os.path.dirname(p)
        z = np.load(p, allow_pickle=True)
        keep = {k: z[k] for k in z.files
                if arrays is None or k in tuple(arrays)
                or k in ("pair_ids", "out_fs")}
        dst = os.path.join(hfb_dir, f"s{sess:02d}_hfb.npz")
        np.savez_compressed(dst, **keep)
        total += os.path.getsize(dst)
        for name, sink in (("hfb_pairs.csv", qc_rows),
                           ("hfb_intervals.csv", iv_rows)):
            f = os.path.join(src, name)
            if os.path.isfile(f):
                t = pd.read_csv(f)
                t["session"] = sess
                sink.append(t)
        n_pairs = len(np.atleast_1d(z["pair_ids"]))
        idx.append({"session": sess, "file": f"hfb/s{sess:02d}_hfb.npz",
                    "n_derivations": n_pairs,
                    "out_fs": float(np.atleast_1d(z["out_fs"])[0]),
                    "arrays": ",".join(k for k in keep
                                       if k not in ("pair_ids", "out_fs"))})
    return (pd.DataFrame(idx),
            pd.concat(qc_rows, ignore_index=True) if qc_rows else pd.DataFrame(),
            pd.concat(iv_rows, ignore_index=True) if iv_rows else pd.DataFrame(),
            total)


def load_hfb(bundle_dir, session, arrays=None, as_float32=True):
    """One session's HFB store, as {array_name: (n_derivations, n_samples)}.

    Cast to float32 by default: the store is float16 to survive the copy home,
    and float16 accumulates visible error over long sums -- which is exactly
    what averaging a few thousand peri-ripple epochs is.

    `pair_ids` gives the row order, matching `hfb_pairs` for that session.
    """
    p = os.path.join(bundle_dir, "hfb", f"s{int(session):02d}_hfb.npz")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{p} -- was the bundle exported with_hfb=True?")
    z = np.load(p, allow_pickle=True)
    out = {"pair_ids": [str(v) for v in np.atleast_1d(z["pair_ids"])],
           "out_fs": float(np.atleast_1d(z["out_fs"])[0])}
    for k in z.files:
        if k in ("pair_ids", "out_fs"):
            continue
        if arrays is not None and k not in tuple(arrays):
            continue
        out[k] = np.asarray(z[k], np.float32) if as_float32 else z[k]
    return out


def export_bundle(analysis_name=ANALYSIS_NAME, data_root=None,
                  out_name="swr_bundle", out_dir=None, with_hfb=True,
                  hfb_arrays=None):
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
        hfb_pairs  every derivation with an HFB time course, hippocampal and
                   cortical, with its ROI, coordinate and sub-band count
        hfb_intervals  artifact-free intervals for those derivations. The
                   cortical mask is INDEPENDENT of the hippocampal one, and a
                   peri-ripple window needs both clean -- so this is required,
                   not decorative

    With `with_hfb`, the continuous 100 Hz power time courses are written to
    `bundle/hfb/s{NN}_hfb.npz`, one file per session, and read back with
    `load_hfb`. Nothing is epoched: every window choice stays open.

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
    bad_iv = []
    for sess, rip_dir in sessions:
        meta = subj.get(sess, {})
        # Detection may have READ a differently-named extraction (--clean_name,
        # used by every pad variant so a sweep costs no raw I/O). Without this
        # the pairs table silently comes back empty for such a run -- and the
        # FileNotFoundError below then skips the session's behaviour too.
        clean_name = analysis_name
        set_p = os.path.join(rip_dir, "settings.json")
        if os.path.isfile(set_p):
            try:
                with open(set_p) as _f:
                    clean_name = json.load(_f).get("clean_name") or analysis_name
            except Exception:
                pass
        clean_dir = os.path.join(swr_io.session_deriv_dir(sess, R), "LFP-clean",
                                 clean_name)
        try:
            e = pd.read_csv(os.path.join(rip_dir, "ripple_events.csv"))
            e = e[e.passed.fillna(False)]
            # dist_to_artifact_s is NOT optional here: without it the pad
            # cannot be revisited on the laptop, which is the whole point of
            # storing it per event.
            keep = [c for c in ("pair_id", "t_peak_s", "duration_s",
                                "peak_freq_hz", "amp_peak_uv", "rms_peak_z",
                                "spectral_passed_strict",
                                "spectral_passed_relaxed",
                                "dist_to_artifact_s")
                    if c in e.columns]
            e = e[keep].copy()
            e["session"] = sess
            e["subject_key"] = meta.get("subject_key")
            e["recording_site"] = meta.get("recording_site")
            rip.append(e)

            i = pd.read_csv(os.path.join(rip_dir, "clean_intervals.csv"))
            i["session"] = sess
            iv.append(i)

            # The unpadded crossings, so a stricter pad can be imposed at home.
            bp = os.path.join(rip_dir, "artifact_intervals.csv")
            if os.path.isfile(bp):
                bi = pd.read_csv(bp)
                bi["session"] = sess
                bad_iv.append(bi)

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

    hfb_index, hfb_pairs, hfb_iv, hfb_bytes = (
        collect_hfb(analysis_name, R, out_dir, hfb_arrays) if with_hfb
        else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0))

    bundle = {"ripples": cat(rip), "intervals": cat(iv), "pairs": cat(pr),
              "figure_index": fig_index,
              "figure_index_artifact": fig_index_art,
              "behaviour": cat(beh_all), "uncover": cat(unc_all),
              "channel_qc": cat(qc_all),
              "artifact_intervals": cat(bad_iv),
              "hfb_index": hfb_index, "hfb_pairs": hfb_pairs,
              "hfb_intervals": hfb_iv,
              "meta": {"analysis_name": analysis_name,
                       "created": datetime.now().isoformat(timespec="seconds"),
                       "data_root": R,
                       "n_sessions": len(sessions),
                       "figure_data": f"{out_name}_figures.npz",
                       "hfb_dir": "hfb/ (one npz per session, load_hfb)",
                       "hfb_mb": round(hfb_bytes / 1e6, 1),
                       "note": "rates must use intervals for exposure; a ripple "
                               "rate is events per ARTIFACT-FREE second",
                       "repad": ("to impose a larger pad at home: filter events on "
                                 "dist_to_artifact_s >= pad, AND rebuild exposure "
                                 "with swr_artifact.clean_intervals_at_pad on "
                                 "`artifact_intervals`. Doing only the first "
                                 "inflates the rate.")}}

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
    if with_hfb and len(hfb_index):
        n_mpfc = int(hfb_pairs.roi_family.isin(["mPFC", "mOFC"]).sum()) \
            if "roi_family" in hfb_pairs else 0
        print(f"  {'hfb':12s} {len(hfb_index):7d} sessions "
              f"({hfb_bytes/1e6:.0f} MB in hfb/, "
              f"{len(hfb_pairs)} derivations, {n_mpfc} mPFC/mOFC)")
    print(f"\nSaved -> {out_dir}")
    print(f"  bring home: {out_name}.pkl + {out_name}_figures.npz"
          + (f" + hfb/  ({hfb_bytes/1e6:.0f} MB)" if with_hfb and hfb_bytes else ""))
    return bundle
