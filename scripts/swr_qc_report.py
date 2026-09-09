#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 4 CHECKPOINT: does the detector actually find ripples?

Produces the figure that has to be judged before any statistics are written.
Items 1, 2 and 6 are the ones that cannot be fixed later by re-thresholding:

  1. grand-average ripple-triggered waveform (should show the ripple riding on
     a sharp wave)
  2. ripple-triggered time-frequency (should show an isolated 80-100 Hz blob,
     not a broadband smear)
  3. example single events
  4. peak-frequency distribution
  5. rate per derivation against Chen's reference range
  6. rejection cascade, and a white-matter control derivation whose rate should
     be far lower than hippocampus -- if it is not, the detector is finding
     artifacts rather than ripples

Usage:
    python scripts/swr_qc_report.py --session=38

@author: Svenja Kuchenhoff
"""

import os
import sys
import json

import numpy as np
from scipy import signal
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import label as ndimage_label

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc.analyse.swr_io as swr_io
import mc.analyse.swr_detect as det
import mc.plotting.ripple_figures as rfig
import mc.analyse.swr_preproc as pre
import mc.analyse.swr_artifact as art
import era_brewer

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "swr_v1"
PAL = era_brewer.era_brew("Showgirl2", n=7)
HC_COL, ACC_COL = PAL[0], "#0e3d3a"
WIN_S = 0.5

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
})


def _snippets(sig, peaks, half):
    keep = (peaks - half >= 0) & (peaks + half < sig.shape[-1])
    p = peaks[keep]
    if not len(p):
        return np.zeros((0, 2 * half))
    return np.stack([sig[q - half:q + half] for q in p])


PAD_S = 0.25          # window either side of the ripple peak, seconds
RIPPLE_C = "#B74C2D"  # era_brewer Showgirl2 rust
BROAD_C = "#3d3d3d"
SHADE_C = "#F3D9D2"


def _bandpass(x, fs, lo, hi):
    b, a = signal.butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return signal.filtfilt(b, a, x)


def _load(session, data_root, analysis_name):
    clean = os.path.join(swr_io.session_deriv_dir(session, data_root),
                         "LFP-clean", analysis_name)
    rip = os.path.join(swr_io.session_deriv_dir(session, data_root),
                       "LFP-ripples", analysis_name)
    sig = np.load(os.path.join(clean, "continuous.npy"), mmap_mode="r")
    meta = json.load(open(os.path.join(clean, "meta.json")))
    ev = pd.read_csv(os.path.join(rip, "ripple_events.csv"))
    return sig, meta, ev, rip


def _cut(sig, row, pair_idx, fs, pad_s=PAD_S):
    """Broadband snippet centred on the RMS peak, and where the event sits."""
    half = int(round(pad_s * fs))
    pk = int(row.peak_sample)
    lo, hi = pk - half, pk + half
    if lo < 0 or hi > sig.shape[1]:
        return None, None, None
    seg = np.asarray(sig[pair_idx, lo:hi], dtype=float)
    t = (np.arange(len(seg)) - half) / fs
    win = ((int(row.start_sample) - pk) / fs, (int(row.stop_sample) - pk) / fs)
    return t, seg, win


def _panel(ax, t, seg, win, fs, title, sub, show_right_label=False):
    """Broadband and ripple band, each on its OWN axis.

    The ripple band is 1-2 uV against tens of uV of broadband LFP, so the two
    cannot share a y-axis. They used to be drawn on one, with the ripple
    multiplied by a factor printed in small grey text in the corner -- which is
    easy to miss and invites reading a ripple as ten times its real amplitude.
    A second axis in the ripple's own colour states the scale instead of hiding
    it in a caption.
    """
    ax.axvspan(win[0], win[1], color=SHADE_C, lw=0, zorder=0)
    ax.plot(t, seg, color=BROAD_C, lw=0.6, zorder=2, label="broadband")
    ax.set_title(title, fontsize=8, pad=3)
    ax.text(0.02, 0.02, sub, transform=ax.transAxes,
            fontsize=6.5, va="bottom", ha="left", color="#666")
    ax.set_xlim(t[0], t[-1])
    ax.set_xticks([-0.2, 0, 0.2])
    ax.tick_params(labelsize=7)

    rip = _bandpass(seg, fs, *det.RIPPLE_BAND) if hasattr(det, "RIPPLE_BAND") \
        else _bandpass(seg, fs, 80.0, 120.0)
    ax2 = ax.twinx()
    ax2.plot(t, rip, color=RIPPLE_C, lw=0.8, zorder=3, label="80-120 Hz")
    lim = 1.15 * np.abs(rip).max()
    ax2.set_ylim(-lim, lim)
    ax2.tick_params(axis="y", colors=RIPPLE_C, labelsize=6.5)
    ax2.spines["right"].set_color(RIPPLE_C)
    ax2.spines["top"].set_visible(False)
    if show_right_label:
        ax2.set_ylabel(r"80–120 Hz ($\mu$V)", color=RIPPLE_C, fontsize=7)
    return ax2


def _grid(sig, ev, pairs, fs, out_stem, suptitle, n=9):
    # Select from events that can actually be cut: one too close to a recording
    # edge used to leave a hole in the grid.
    ok = []
    for _, r in ev.iterrows():
        if r.pair_id not in pairs:
            continue
        t, seg, win = _cut(sig, r, pairs.index(r.pair_id), fs)
        if t is not None:
            ok.append(r)
        if len(ok) == n:
            break
    ev = pd.DataFrame(ok)
    if not len(ev):
        print(f"    (no events for {os.path.basename(out_stem)})")
        return
    ncol = 3
    nrow = int(np.ceil(len(ev) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(17.4 / 2.54, 3.6 * nrow / 2.54),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (_, r) in enumerate(ev.iterrows()):
        pi = pairs.index(r.pair_id)
        t, seg, win = _cut(sig, r, pi, fs)
        if t is None:
            continue
        ax = axes.ravel()[k]
        ax.axis("on")
        ax2 = _panel(ax, t, seg, win, fs, f"{r.pair_id}  t={r.t_peak_s:.1f}s",
                     f"z={r.rms_peak_z:.1f}  {1000*r.duration_s:.0f} ms  "
                     f"{r.peak_freq_hz:.0f} Hz",
                     show_right_label=(k % ncol == ncol - 1))
        if k % ncol == 0:
            ax.set_ylabel(r"broadband ($\mu$V)", fontsize=7)
        if k >= len(ev) - ncol:
            ax.set_xlabel("time from ripple peak (s)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color=BROAD_C, lw=1.2, label="broadband (left axis)"),
        Line2D([], [], color=RIPPLE_C, lw=1.2, label="80–120 Hz (right axis)")],
        frameon=False, fontsize=7.5, ncol=2,
        loc="lower center", bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(suptitle, fontsize=9, y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {os.path.basename(out_stem)}.png/.pdf  ({len(ev)} events)")



def _example_grids(session, clean_dir, ev, n=9):
    """Best / borderline / rejected example events, as separate grids.

    The 6-panel QC figure shows a handful of events; these three grids show the
    events whose classification is actually a judgement call -- the ones nearest
    the threshold, and the ones the spectral criterion threw out.
    """
    import json as _json
    sig = np.load(os.path.join(clean_dir, "continuous.npy"), mmap_mode="r")
    meta = _json.load(open(os.path.join(clean_dir, "meta.json")))
    fs, pairs = float(meta["fs"]), list(meta["pair_ids"])
    out = os.path.join(os.path.dirname(clean_dir.rstrip("/")), "")  # unused
    figdir = _FIGDIR[0]
    acc = ev[ev.passed.fillna(False)] if "passed" in ev.columns else ev
    rej = ev[~ev.passed.fillna(False)] if "passed" in ev.columns else ev.iloc[:0]
    _grid(sig, acc.sort_values("rms_peak_z", ascending=False), pairs, fs,
          os.path.join(figdir, "examples_best"),
          f"s{session:02d} - clearest accepted ripples (highest RMS z)", n)
    _grid(sig, acc.sort_values("rms_peak_z"), pairs, fs,
          os.path.join(figdir, "examples_borderline"),
          f"s{session:02d} - accepted ripples CLOSEST TO THRESHOLD "
          f"- judge the threshold on these", n)
    if len(rej):
        _grid(sig, rej.sort_values("rms_peak_z", ascending=False), pairs, fs,
              os.path.join(figdir, "examples_rejected"),
              f"s{session:02d} - REJECTED by the spectral criterion", n)



# =============================================================================
# QUANTITATIVE CHECKPOINT
# =============================================================================
# The figure has to be looked at, but "looks fine" does not scale to 56 sessions.
# These are the same checkpoint criteria as methods.md section 6, evaluated
# numerically so a cluster run can be triaged before anyone opens a PDF.
#
# Reference values are Chen et al. 2025 (J Neurosci 45:e1502252025):
#   ripple rate                0.17-0.24 Hz
#   spectral rejection         23.4% +- 9.9%   -> 13.5-33.3%
#   duration                   38-500 ms by construction; median ~50-90 ms
#   peak frequency             inside the 80-120 Hz detection band
#
# A FAIL means the session should not enter the statistics as it stands. A CHECK
# means it is outside the reference range but not obviously broken -- look at it.

QC_RULES = {
    "rate_hz":            (0.05, 0.60, 0.17, 0.24),
    "spectral_reject_pct": (5.0, 50.0, 13.5, 33.3),
    "peak_freq_hz":       (80.0, 120.0, 85.0, 115.0),
    "duration_ms":        (38.0, 500.0, 40.0, 120.0),
    "clean_frac":         (0.33, 1.00, 0.50, 1.00),
    "ripple_gain":        (1.20, 99.0, 1.50, 99.0),
}


def _verdict(name, value):
    """FAIL outside the hard range, CHECK outside the reference range, else PASS."""
    if value is None or not np.isfinite(value):
        return "FAIL", "not computable"
    hard_lo, hard_hi, ref_lo, ref_hi = QC_RULES[name]
    if value < hard_lo or value > hard_hi:
        return "FAIL", f"outside {hard_lo}-{hard_hi}"
    if value < ref_lo or value > ref_hi:
        return "CHECK", f"outside reference {ref_lo}-{ref_hi}"
    return "PASS", ""


def _ripple_gain(sig, ev, pairs, fs, n=200):
    """Mean ripple-band envelope at the peak divided by its value at the window
    edges. A real ripple population gives a clear bump; a detector triggering on
    broadband noise gives ~1."""
    from scipy.signal import hilbert
    half = int(round(PAD_S * fs))
    env = []
    for _, r in ev.head(n).iterrows():
        if r.pair_id not in pairs:
            continue
        t, seg, _ = _cut(sig, r, pairs.index(r.pair_id), fs)
        if t is None:
            continue
        env.append(np.abs(hilbert(_bandpass(seg, fs, 80.0, 120.0))))
    if len(env) < 5:
        return np.nan
    env = np.array(env).mean(0)
    edge = max(int(0.05 * fs), 1)
    base = np.concatenate([env[:edge], env[-edge:]]).mean()
    return float(env[half] / base) if base > 0 else np.nan


def qc_metrics(session, analysis_name=ANALYSIS_NAME, save=True):
    """Numeric checkpoint for one session -> qc_metrics.csv (one row per rule)."""
    import json as _json
    session = int(session)
    R = swr_io.get_data_root()
    clean_dir = os.path.join(swr_io.session_deriv_dir(session, R),
                             "LFP-clean", analysis_name)
    rip_dir = os.path.join(swr_io.session_deriv_dir(session, R),
                           "LFP-ripples", analysis_name)
    ev_p = os.path.join(rip_dir, "ripple_events.csv")
    if not os.path.isfile(ev_p):
        return None
    ev = pd.read_csv(ev_p)
    acc = ev[ev.passed.fillna(False)] if "passed" in ev.columns else ev
    if not len(acc):
        return None
    meta = _json.load(open(os.path.join(clean_dir, "meta.json")))
    fs, pairs = float(meta["fs"]), list(meta["pair_ids"])

    ch_p = os.path.join(rip_dir, "channel_qc.csv")
    ch = pd.read_csv(ch_p) if os.path.isfile(ch_p) else pd.DataFrame()
    # a pair that is already excluded must not drag the session metrics down --
    # its whole point is that it was too contaminated to analyse
    if "excluded" in ch.columns:
        ch = ch[~ch.excluded.fillna(False)]
    rate = float(ch.rate_hz.median()) if "rate_hz" in ch.columns else np.nan
    clean = np.nan
    for c in ("clean_frac", "clean_fraction", "frac_clean"):
        if c in ch.columns:
            clean = float(ch[c].median()); break
    if not np.isfinite(clean) and "clean_s" in ch.columns:
        clean = float((ch.clean_s / float(meta["duration_s"])).median())

    sig = np.load(os.path.join(clean_dir, "continuous.npy"), mmap_mode="r")
    vals = {
        "rate_hz": rate,
        "spectral_reject_pct": 100.0 * float((~ev.passed.fillna(False)).mean()),
        "peak_freq_hz": float(acc.peak_freq_hz.median()),
        "duration_ms": float(acc.duration_s.median() * 1000.0),
        "clean_frac": clean,
        "ripple_gain": _ripple_gain(sig, acc, pairs, fs),
    }
    rows = []
    for k, v in vals.items():
        verdict, why = _verdict(k, v)
        rows.append(dict(session=session, metric=k, value=v,
                         verdict=verdict, note=why,
                         hard_lo=QC_RULES[k][0], hard_hi=QC_RULES[k][1],
                         ref_lo=QC_RULES[k][2], ref_hi=QC_RULES[k][3]))
    out = pd.DataFrame(rows)
    out["n_ripples"] = len(acc)
    out["n_derivations"] = len(ch) if len(ch) else np.nan

    print(f"\n  quantitative checkpoint  (s{session:02d}, {len(acc)} ripples, "
          f"{len(ch)} derivation(s)):")
    for _, r in out.iterrows():
        mark = {"PASS": "  ok  ", "CHECK": " check", "FAIL": " FAIL "}[r.verdict]
        print(f"   [{mark}] {r.metric:22s} {r.value:8.3f}   {r.note}")
    if save:
        out.to_csv(os.path.join(rip_dir, "qc_metrics.csv"), index=False)
        print(f"   -> {os.path.join(rip_dir, 'qc_metrics.csv')}")
    return out


def qc_group(analysis_name=ANALYSIS_NAME, sessions=None, save=True):
    """Aggregate every session's qc_metrics.csv into one triage table."""
    R = swr_io.get_data_root()
    if sessions is None:
        cfg = swr_io.load_config(R)
        sessions = sorted(int(k) for k in cfg.keys())
    frames = []
    for s in sessions:
        p = os.path.join(swr_io.session_deriv_dir(int(s), R), "LFP-ripples",
                         analysis_name, "qc_metrics.csv")
        if os.path.isfile(p):
            frames.append(pd.read_csv(p))
    if not frames:
        print("no qc_metrics.csv found -- run qc_metrics per session first")
        return None
    allm = pd.concat(frames, ignore_index=True)
    wide = allm.pivot(index="session", columns="metric", values="value")
    verd = allm.pivot(index="session", columns="metric", values="verdict")
    wide["worst"] = verd.apply(
        lambda r: "FAIL" if (r == "FAIL").any() else
                  ("CHECK" if (r == "CHECK").any() else "PASS"), axis=1)
    wide["n_ripples"] = allm.groupby("session").n_ripples.first()

    print("\n" + "=" * 78)
    print(" QUANTITATIVE QC ACROSS SESSIONS")
    print("=" * 78)
    print(wide.round(3).to_string())
    print("\n verdicts: " + str(dict(wide.worst.value_counts())))
    for m in QC_RULES:
        bad = verd.index[verd[m].isin(["FAIL"])].tolist() if m in verd else []
        if bad:
            print(f"   FAIL on {m}: " + ", ".join(f"s{int(x):02d}" for x in bad))
    if save:
        out = os.path.join(swr_io.derivatives_dir(R), "group", "swr")
        os.makedirs(out, exist_ok=True)
        allm.to_csv(os.path.join(out, "qc_metrics_all_sessions.csv"), index=False)
        wide.to_csv(os.path.join(out, "qc_metrics_summary.csv"))
        print(f"\n saved -> {out}/qc_metrics_summary.csv")
    return None      # fire renders a returned DataFrame as an attribute listing


ARTIFACT_EXCERPT_S = 60.0        # length of the padding-figure excerpt
ARTIFACT_SNIP_S = 0.30           # half-width of the IED / ripple snippets


def _artifact_panels(session, raw_by_pair, fs, masks, passed, rip_dir):
    """The IED-vs-ripple and padding panels, and the cache to rebuild them.

    Both need the raw signal, which lives only on the cluster, so the snippets
    they use are written to `artifact_examples.npz` alongside. That file is what
    `swr_export.py bundle` brings home, and it is what lets these panels be
    redrawn -- restyled, recoloured, recomposed -- without the LFP.

    The excerpt is the minute whose rejected fraction is CLOSEST TO THE
    DERIVATION'S OWN -- not the busiest one. The panel exists to explain where
    ~40% of a recording goes, and picking the worst minute would explain it with
    a number the rest of the paper never uses. Both fractions are printed on the
    panel so a reader can see the excerpt is representative.
    """
    pid = max(raw_by_pair, key=lambda k: masks[k]["bad"].mean())
    x = raw_by_pair[pid]
    bad, per = masks[pid]["bad"], masks[pid]["per"]
    half = int(round(ARTIFACT_SNIP_S * fs))

    # excerpt: the window whose removed fraction best matches the whole
    # derivation's, among windows that actually contain crossings
    overall = float(bad.mean())
    n_win = int(round(ARTIFACT_EXCERPT_S * fs))
    if len(x) > n_win:
        raw_flag = np.zeros(len(x), bool)
        for v in per.values():
            raw_flag |= v
        step = max(1, n_win // 10)
        starts = np.arange(0, len(x) - n_win, step)
        fracs = np.array([bad[a:a + n_win].mean() for a in starts])
        has = np.array([raw_flag[a:a + n_win].any() for a in starts])
        score = np.where(has, np.abs(fracs - overall), np.inf)
        a0 = int(starts[int(np.argmin(score))]) if np.isfinite(score).any() else 0
    else:
        a0 = 0
    sl = slice(a0, a0 + min(n_win, len(x)))
    ex_x = np.asarray(x[sl], float)
    ex_per = {k: np.asarray(v[sl], bool) for k, v in per.items()}
    ex_bad = np.asarray(bad[sl], bool)

    # The discharge with the largest RIPPLE-BAND envelope -- not the largest raw
    # deflection. Measured across 12 derivations, IEDs beat ripples on raw
    # amplitude in 12/12 (median 34 vs 9 uV) but on peak ripple-band envelope in
    # 11/12 (20 vs 4 uV), and it is the second number that matters: the panel
    # exists to show what the detector would have called a ripple, not what is
    # merely big. Ranking by raw amplitude picked visually unconvincing examples
    # on clean derivations.
    xb = np.abs(hilbert(sosfiltfilt(
        butter(4, [det.RIPPLE_BAND[0] / (fs / 2), det.RIPPLE_BAND[1] / (fs / 2)],
               btype="band", output="sos"), x)))

    ied_snip = np.zeros(0)
    lab, n = ndimage_label(per["ied_janca"])
    if n:
        best, best_env = None, -np.inf
        for i in range(1, n + 1):
            idx = np.flatnonzero(lab == i)
            c = int(idx[len(idx) // 2])
            if c - half < 0 or c + half >= len(x):
                continue
            e_ = xb[idx].max()
            if e_ > best_env:
                best, best_env = c, e_
        if best is not None:
            ied_snip = np.asarray(x[best - half:best + half], float)

    # a ripple from the same derivation, from CLEAN time, of typical amplitude
    # Ranked by the ripple-band envelope at the event, not by `amp_peak_uv`:
    # the panel has to show something a reader recognises as a ripple, and the
    # raw amplitude of a ripple is mostly the background it sits on. The 90th
    # percentile rather than the maximum, so it is a good example and not a
    # freak one.
    rip_snip = np.zeros(0)
    ev = passed[passed.pair_id == pid]
    cand = []
    for _, r in ev.iterrows():
        c = int(r.peak_sample) if "peak_sample" in r else int(r.t_peak_s * fs)
        if c - half >= 0 and c + half < len(x) and not bad[c]:
            w = int(0.025 * fs)
            cand.append((float(xb[max(0, c - w):c + w].max()), c))
    if cand:
        cand.sort(key=lambda z: z[0])
        c = cand[int(0.90 * (len(cand) - 1))][1]
        rip_snip = np.asarray(x[c - half:c + half], float)

    np.savez_compressed(
        os.path.join(rip_dir, "artifact_examples.npz"),
        fs=fs, session=session, pair_id=pid, t0_s=a0 / fs,
        excerpt=ex_x, excerpt_bad=ex_bad,
        ied=ied_snip, ripple=rip_snip,
        **{f"excerpt_per__{k}": v for k, v in ex_per.items()})

    rfig.padding_figure(
        ex_x, fs, ex_per, ex_bad, t0_s=a0 / fs, overall_frac=overall,
        out_stem=os.path.join(_FIGDIR[0], "artifact_padding"),
        title=f"s{session:02d} {pid}: criterion crossings, then ±1 s padding")
    if ied_snip.size and rip_snip.size:
        rfig.ied_vs_ripple_figure(
            ied_snip, rip_snip, fs,
            out_stem=os.path.join(_FIGDIR[0], "ied_vs_ripple"),
            title=f"s{session:02d} {pid}")
    else:
        print(f"  [ied_vs_ripple skipped: "
              f"{'no IED' if not ied_snip.size else 'no clean ripple'} on {pid}]")
    print(f"  saved -> artifact_padding, ied_vs_ripple, artifact_examples.npz")


_FIGDIR = [None]


def _qc_report_one(session, analysis_name=ANALYSIS_NAME, max_events=800):
    swr_io.start_log(os.path.join(swr_io.session_deriv_dir(int(session), swr_io.get_data_root()), "LFP-ripples", analysis_name), "swr_qc_report")
    session = int(session)
    R = swr_io.get_data_root()
    clean_dir = os.path.join(swr_io.session_deriv_dir(session, R),
                             "LFP-clean", analysis_name)
    rip_dir = os.path.join(swr_io.session_deriv_dir(session, R),
                           "LFP-ripples", analysis_name)
    ev_p = os.path.join(rip_dir, "ripple_events.csv")
    if not os.path.isfile(ev_p):
        print(f"s{session:02d}: no ripple_events.csv"); return None

    sig = np.load(os.path.join(clean_dir, "continuous.npy"), mmap_mode='r')
    pairs = pd.read_csv(os.path.join(clean_dir, "pairs.csv"))
    events = pd.read_csv(ev_p)
    qc = pd.read_csv(os.path.join(rip_dir, "channel_qc.csv"))
    with open(os.path.join(clean_dir, "meta.json")) as f:
        fs = float(json.load(f)["fs"])

    passed = events[events.passed]
    if not len(passed):
        print(f"s{session:02d}: no passing events"); return None
    half = int(WIN_S * fs)

    # gather snippets across derivations
    raw_sn, bp_sn, sw_sn = [], [], []
    sos_r = butter(4, [det.RIPPLE_BAND[0]/(fs/2), det.RIPPLE_BAND[1]/(fs/2)],
                   btype='band', output='sos')
    # The sharp wave is a slow deflection ~40-100 ms wide, so most of its energy
    # sits below 20 Hz. The 8-40 Hz band-pass used previously high-passed it away:
    # measured across the development set it gave a median |SNR| of 0.28 against
    # 1.37 for a 20 Hz low-pass, i.e. it was showing essentially nothing.
    sos_s = butter(4, rfig.SW_BAND_HZ/(fs/2), btype='low', output='sos')
    for i, p in pairs.iterrows():
        ev = passed[passed.pair_id == p.pair_id]
        if not len(ev):
            continue
        x = np.asarray(sig[i], float)
        # Average on the ripple trough, not the envelope peak: the envelope
        # carries no phase, so an envelope-locked average cancels to ~7% of the
        # single-event amplitude. Display only -- detection and stats are
        # unaffected. See mc.plotting.ripple_figures.trough_lock.
        pk = rfig.trough_lock(ev, sosfiltfilt(sos_r, x))
        if len(pk) > max_events:
            pk = np.random.RandomState(42).choice(pk, max_events, replace=False)
        raw_sn.append(_snippets(x, pk, half))
        bp_sn.append(_snippets(sosfiltfilt(sos_r, x), pk, half))
        sw_sn.append(_snippets(sosfiltfilt(sos_s, x), pk, half))
    raw_sn = np.concatenate(raw_sn); bp_sn = np.concatenate(bp_sn)
    sw_sn = np.concatenate(sw_sn)
    t = (np.arange(-half, half) / fs) * 1000.0

    fig = plt.figure(figsize=(11.0, 7.2))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.32)

    # 1. grand average -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    m, se = raw_sn.mean(0), raw_sn.std(0)/np.sqrt(len(raw_sn))
    ax.plot(t, m, color=ACC_COL, lw=1.2, label="raw")
    ax.fill_between(t, m-se, m+se, color=ACC_COL, alpha=0.25, lw=0)
    ax.plot(t, sw_sn.mean(0), color=HC_COL, lw=1.2,
            label=f"< {rfig.SW_BAND_HZ:g} Hz (sharp wave)")
    ax.axvline(0, color='0.6', ls=':', lw=0.8)
    ax.set_xlim(-250, 250); ax.set_xlabel("Time from ripple peak (ms)")
    ax.set_ylabel(r"Amplitude ($\mu$V)")
    ax.set_title(f"Grand average (n={len(raw_sn)})")
    ax.legend(frameon=False, loc="lower right")

    # 2. band-passed average ----------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, bp_sn.mean(0), color=ACC_COL, lw=1.0)
    ax.axvline(0, color='0.6', ls=':', lw=0.8)
    ax.set_xlim(-120, 120); ax.set_xlabel("Time from ripple peak (ms)")
    ax.set_ylabel(r"80–120 Hz ($\mu$V)")
    ax.set_title("Band-passed average")

    # 3. ripple-triggered TFR ---------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    freqs = np.arange(30, 181, 2.5)
    tfr = np.zeros((len(freqs), raw_sn.shape[1]))
    for j, f0 in enumerate(freqs):
        bw = max(2.0, f0/6.0*2)
        s = butter(3, [max(1, f0-bw/2)/(fs/2), min(fs/2-1, f0+bw/2)/(fs/2)],
                   btype='band', output='sos')
        tfr[j] = np.abs(hilbert(sosfiltfilt(s, raw_sn, axis=-1), axis=-1)).mean(0)
    base = tfr[:, (t < -300)].mean(1, keepdims=True)
    tfr_pct = (tfr - base) / base * 100
    im = ax.pcolormesh(t, freqs, tfr_pct, cmap="magma", shading="auto")
    ax.axhline(det.RIPPLE_BAND[0], color='w', ls=':', lw=0.7)
    ax.axhline(det.RIPPLE_BAND[1], color='w', ls=':', lw=0.7)
    ax.set_xlim(-250, 250); ax.set_xlabel("Time from ripple peak (ms)")
    ax.set_ylabel("Frequency (Hz)"); ax.set_title("Ripple-triggered TFR")
    plt.colorbar(im, ax=ax, label="% change")

    # 4. examples ----------------------------------------------------------
    # Twin axes: the band-passed ripple is ~1-2 uV while the raw trace swings
    # +-50 uV, so plotting both on one axis makes the ripple invisible.
    # Examples are the largest-amplitude events, which is what one inspects.
    order = np.argsort(np.abs(bp_sn).max(axis=1))[::-1]
    for k in range(3):
        ax = fig.add_subplot(gs[1, k])
        idx = order[k * max(1, len(order) // 30)]
        ax.plot(t, raw_sn[idx], color='0.45', lw=0.9)
        ax.set_xlim(-150, 150); ax.set_xlabel("Time (ms)")
        ax.set_ylabel(r"raw ($\mu$V)", color='0.45')
        ax.tick_params(axis='y', colors='0.45')
        ax2 = ax.twinx()
        ax2.plot(t, bp_sn[idx], color=HC_COL, lw=0.9)
        ax2.set_ylabel(r"80–120 Hz ($\mu$V)", color=HC_COL)
        ax2.tick_params(axis='y', colors=HC_COL)
        ax.axvline(0, color='0.6', ls=':', lw=0.8)
        ax.set_title(f"Example {k+1}", fontsize=9)

    # 5. peak frequency ----------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    ax.hist(passed.peak_freq_hz.dropna(), bins=np.arange(30, 185, 5),
            color=HC_COL, edgecolor='w', lw=0.4)
    ax.axvspan(*det.RIPPLE_BAND, color=ACC_COL, alpha=0.12)
    ax.set_xlabel("Peak frequency (Hz)"); ax.set_ylabel("Events")
    ax.set_title("Peak frequency")

    # 6. rate per derivation ----------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    g = qc[~qc.excluded & qc.rate_hz.notna()]
    ax.bar(range(len(g)), g.rate_hz, color=HC_COL, edgecolor='w')
    ax.axhspan(0.17, 0.24, color=ACC_COL, alpha=0.18, label="Chen 0.17–0.24 Hz")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.pair_id, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel("Ripple rate (Hz)"); ax.set_title("Rate per derivation")
    ax.legend(frameon=False, fontsize=7)

    # 7. rejection cascade -------------------------------------------------
    ax = fig.add_subplot(gs[2, 2])
    n_cand = int(qc.n_candidates.sum()) if 'n_candidates' in qc else len(events)
    stages = ["candidates", "dur+amp", "spectral"]
    vals = [n_cand, len(events), int(events.passed.sum())]
    ax.bar(stages, vals, color=[PAL[2], PAL[1], HC_COL], edgecolor='w')
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v}", ha='center', va='bottom', fontsize=8)
    ax.set_ylabel("Events"); ax.set_title("Rejection cascade")

    fig.suptitle(f"s{session:02d} ripple detection QC — "
                 f"{int(passed.shape[0])} ripples, "
                 f"{len(g)} derivations", fontsize=12, y=0.98)
    _FIGDIR[0] = os.path.join(rip_dir, "figures")
    os.makedirs(_FIGDIR[0], exist_ok=True)
    out = os.path.join(rip_dir, "qc_ripples")
    fig.savefig(out + ".png", dpi=300, bbox_inches='tight')
    fig.savefig(out + ".pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  saved -> {out}.png / .pdf")

    # numeric summary for the checkpoint
    print(f"  peak freq: median {passed.peak_freq_hz.median():.1f} Hz "
          f"(IQR {passed.peak_freq_hz.quantile(.25):.1f}–"
          f"{passed.peak_freq_hz.quantile(.75):.1f})")
    print(f"  duration : median {passed.duration_s.median()*1000:.0f} ms")
    print(f"  rate     : median {g.rate_hz.median():.3f} Hz across {len(g)} derivations")

    _example_grids(session, clean_dir, events)

    # Chen et al. 2025 Fig 2a-b, and the sharp-wave assessment
    raw_by_pair, coords, rois = {}, [], []
    for i, p_ in pairs.iterrows():
        if i >= sig.shape[0]:
            continue
        if not len(passed[passed.pair_id == p_.pair_id]):
            continue
        raw_by_pair[p_.pair_id] = np.asarray(sig[i], float)
        if np.isfinite([p_.get("mni_x"), p_.get("mni_y"), p_.get("mni_z")]).all():
            coords.append([p_.mni_x, p_.mni_y, p_.mni_z])
            rois.append(p_.get("pair_roi", "HC_mid"))
    if raw_by_pair:
        st = session_stacks(session, analysis_name)      # computes and caches
        if st is not None:
            ok = np.isfinite(st["coords"]).all(1)
            rfig.chen_panels(
                st["mean"], np.nanmean(st["tfr"], axis=0), st["t_ms"],
                st["ex_raw"] if st["ex_raw"].size else None,
                st["ex_tfr"] if st["ex_tfr"].size else None,
                coords=st["coords"][ok] if ok.any() else None,
                rois=list(st["rois"][ok]) if ok.any() else None,
                out_stem=os.path.join(_FIGDIR[0], "chen_fig2"),
                title=f"s{session:02d}", n_contacts=len(st["mean"]),
                ex_bp=st["ex_bp"] if st["ex_bp"].size else None)
        sw = rfig.sharp_wave_figure(raw_by_pair, fs, passed,
                                    out_stem=os.path.join(_FIGDIR[0], "sharp_wave"),
                                    title=f"s{session:02d} sharp-wave assessment")
        if len(sw):
            sw.insert(0, "session", session)
            sw.to_csv(os.path.join(rip_dir, "sharp_wave.csv"), index=False)
            rfig.sharp_wave_examples(
                raw_by_pair, fs, passed, sw,
                out_stem=os.path.join(_FIGDIR[0], "sharp_wave_examples"),
                title=f"s{session:02d}: ripples with a visible sharp wave")

        # What the artifact rejection removed. 35-54% of a recording is deleted
        # here, so it gets looked at rather than trusted.
        masks, astats = {}, {}
        for pid, x in raw_by_pair.items():
            bad, st, per = art.artifact_mask(x, fs, return_per=True)
            masks[pid] = {"bad": bad, "per": per}
            astats[pid] = st
        rfig.artifact_figure(raw_by_pair, fs, masks, astats,
                             out_stem=os.path.join(_FIGDIR[0], "artifact_rejection"),
                             title=f"s{session:02d}: what artifact rejection removed")
        pd.DataFrame(astats).T.rename_axis("pair_id").reset_index().assign(
            session=session).to_csv(
            os.path.join(rip_dir, "artifact_criteria.csv"), index=False)

        # The two manuscript panels, plus the cache that lets them be redrawn
        # on a laptop. Everything above needs `continuous.npy`, which stays on
        # the cluster; `artifact_examples.npz` is ~1 MB and comes home.
        _artifact_panels(session, raw_by_pair, fs, masks, passed, rip_dir)

    qc_metrics(session, analysis_name)      # prints its own checkpoint
    return None


def sessions_with_ripples(analysis_name=ANALYSIS_NAME, data_root=None):
    """Every session that has a ripple_events.csv on this machine."""
    R = data_root or swr_io.get_data_root()
    cfg = swr_io.load_config(R)
    out = []
    for k in sorted(int(x) for x in cfg.keys()):
        p = os.path.join(swr_io.session_deriv_dir(k, R), "LFP-ripples",
                         analysis_name, "ripple_events.csv")
        if os.path.isfile(p):
            out.append(k)
    return out


STACK_KEYS = ("mean", "tfr", "t_ms", "coords", "rois", "pair_id",
              "n_events", "ex_raw", "ex_bp", "ex_tfr", "fs")


def session_stacks(session, analysis_name=ANALYSIS_NAME, win_s=WIN_S,
                   use_cache=True):
    """Per-derivation ripple-locked mean trace and TFR for one session.

    Cached to `ripple_stacks.npz` beside the events, because the TFR is the
    expensive part of the whole QC stage -- one bandpass+Hilbert pass over the
    full recording per frequency per derivation -- and the group figure would
    otherwise repeat it for every session each time it is drawn.
    """
    R = swr_io.get_data_root()
    clean_dir = os.path.join(swr_io.session_deriv_dir(session, R),
                             "LFP-clean", analysis_name)
    rip_dir = os.path.join(swr_io.session_deriv_dir(session, R),
                           "LFP-ripples", analysis_name)
    cache = os.path.join(rip_dir, "ripple_stacks.npz")
    if use_cache and os.path.isfile(cache):
        z = np.load(cache, allow_pickle=True)
        got = {k: z[k] for k in z.files}
        # A cache written before a format change is missing keys the callers
        # index directly; rebuild rather than KeyError halfway through a run.
        if all(k in got for k in STACK_KEYS):
            return got
        print("    (stale stacks cache, rebuilding)")

    sig = np.load(os.path.join(clean_dir, "continuous.npy"), mmap_mode="r")
    pairs = pd.read_csv(os.path.join(clean_dir, "pairs.csv"))
    with open(os.path.join(clean_dir, "meta.json")) as f:
        fs = float(json.load(f)["fs"])
    ev = pd.read_csv(os.path.join(rip_dir, "ripple_events.csv"))
    ev = ev[ev.passed.fillna(False)]
    half = int(round(win_s * fs))

    means, tfrs, coords, rois, ids, ns = [], [], [], [], [], []
    ex_raw = ex_tfr = ex_bp = best_ex = None
    best_z = -np.inf
    t_ms = np.linspace(-win_s, win_s, 2 * half) * 1000
    for i, p_ in pairs.iterrows():
        if i >= sig.shape[0]:
            continue
        e = ev[ev.pair_id == p_.pair_id]
        if len(e) < 30:
            continue
        raw = np.asarray(sig[i], float)
        rb = rfig._bp(raw, fs, *rfig.RIPPLE_BAND)
        pk = rfig.trough_lock(e, rb)
        sn = rfig._snips(raw, pk, half)
        if not len(sn):
            continue
        means.append(sn.mean(0)); ns.append(len(sn)); ids.append(p_.pair_id)
        tf, t_ms, _ = rfig.ripple_tfr(raw, fs, pk, win_s=win_s)
        tfrs.append(tf)
        xyz = [p_.get("mni_x"), p_.get("mni_y"), p_.get("mni_z")]
        coords.append(xyz if np.isfinite(xyz).all() else [np.nan] * 3)
        rois.append(str(p_.get("pair_roi", "HC_mid")))
        # Which event to FEATURE in the figure. A display choice only -- it
        # changes no count, rate or statistic -- but the wrong score picks an
        # unconvincing example, and the featured event is what most readers
        # take as the evidence that these are ripples.
        #
        # Scoring on envelope / slow-background alone (the previous rule) is not
        # enough: it rewards any event on a flat contact, including one sitting
        # in continuous ripple-band activity, where nothing stands out. A ripple
        # is a DISCRETE burst, so focality has to be in the score, together with
        # the two properties that make an event look canonical rather than
        # marginal: a duration in the middle of the accepted range, and a peak
        # frequency away from the band edges where the filter, not the signal,
        # sets it.
        env = np.abs(hilbert(rb))
        slow = sosfiltfilt(butter(4, 40 / (fs / 2), btype="low", output="sos"), raw)
        cand_ev = e[(e.peak_sample >= half) &
                    (e.peak_sample < len(raw) - half)]
        if len(cand_ev):
            sc, cs = [], []
            for _, r in cand_ev.iterrows():
                c = int(r.peak_sample)
                a, b = int(r.start_sample), int(r.stop_sample)
                nbr = np.r_[env[c - half:max(a, c - half)],
                            env[min(b, c + half):c + half]]
                focal = env[c] / (np.median(nbr) + 1e-9) if nbr.size else 1.0
                contrast = env[c] / (slow[c - half:c + half].std() + 1e-9)
                dur_ms = 1000.0 * float(r.duration_s)
                # triangular preferences, peaked at 65 ms and 97 Hz
                w_dur = max(0.0, 1.0 - abs(dur_ms - 65.0) / 45.0)
                w_frq = max(0.0, 1.0 - abs(float(r.peak_freq_hz) - 97.0) / 20.0)
                sc.append(focal * contrast * w_dur * w_frq)
                cs.append(c)
            j = int(np.argmax(sc))
            if sc[j] > best_z:
                best_z = sc[j]
                best_ex = (i, int(cs[j]))

    if best_ex is not None:
        i, bpk = best_ex
        raw = np.asarray(sig[i], float)
        ex_raw = raw[bpk - half:bpk + half]
        ex_bp = rfig._bp(raw, fs, *rfig.RIPPLE_BAND)[bpk - half:bpk + half]
        ex_tfr, _, _ = rfig.ripple_tfr(raw, fs, [bpk], win_s=win_s)

    if not means:
        return None
    out = dict(mean=np.stack(means), tfr=np.stack(tfrs), t_ms=t_ms,
               coords=np.array(coords, float), rois=np.array(rois),
               pair_id=np.array(ids), n_events=np.array(ns),
               ex_raw=np.asarray(ex_raw if ex_raw is not None else []),
               ex_bp=np.asarray(ex_bp if ex_bp is not None else []),
               ex_tfr=np.asarray(ex_tfr if ex_tfr is not None else []),
               fs=np.array([fs]))
    os.makedirs(rip_dir, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def group_figure(sessions=None, analysis_name=ANALYSIS_NAME, win_s=WIN_S,
                 use_cache=True):
    """Chen Fig 2a-b pooled across sessions.

    Averaged with one weight per derivation, not per event: a single contact
    with a high ripple rate would otherwise dominate the grand average, and the
    claim is about hippocampal contacts in general.
    """
    R = swr_io.get_data_root()
    sessions = sessions or sessions_with_ripples(analysis_name, R)
    means, tfrs, coords, rois = [], [], [], []
    ex_raw = ex_tfr = ex_bp = t_ms = None
    best_n = -1
    for sess in sessions:
        try:
            st = session_stacks(sess, analysis_name, win_s, use_cache)
        except (FileNotFoundError, OSError) as e:
            print(f"  s{sess:02d}: skipped ({type(e).__name__})"); continue
        if st is None:
            continue
        means.append(st["mean"]); tfrs.append(st["tfr"])
        coords.append(st["coords"]); rois.append(st["rois"])
        t_ms = st["t_ms"]
        if st["ex_raw"].size and int(st["n_events"].sum()) > best_n:
            best_n = int(st["n_events"].sum())
            ex_raw, ex_tfr = st["ex_raw"], st["ex_tfr"]
            ex_bp = st["ex_bp"] if st["ex_bp"].size else None
        print(f"  s{sess:02d}: {len(st['mean'])} derivations")

    if not means:
        print("group_figure: nothing to pool"); return None
    means = np.concatenate(means); tfrs = np.concatenate(tfrs)
    coords = np.concatenate(coords); rois = np.concatenate(rois)
    ok = np.isfinite(coords).all(1)
    out_dir = os.path.join(swr_io.derivatives_dir(R), "group", "swr", "figures")
    rfig.chen_panels(
        means, np.nanmean(tfrs, axis=0), t_ms, ex_raw, ex_tfr,
        coords=coords[ok] if ok.any() else None,
        rois=list(rois[ok]) if ok.any() else None,
        out_stem=os.path.join(out_dir, "chen_fig2_group"),
        title=f"Hippocampal ripples, {len(means)} derivations "
              f"across {len(sessions)} sessions",
        n_contacts=len(means), ex_bp=ex_bp)
    print(f"  pooled {len(means)} derivations from {len(sessions)} sessions")

    # --- artifact contamination across every derivation in the study --------
    qcs = []
    for sess in sessions:
        p_ = os.path.join(swr_io.session_deriv_dir(int(sess), R), "LFP-ripples",
                          analysis_name, "channel_qc.csv")
        if os.path.isfile(p_):
            q = pd.read_csv(p_); q["session"] = sess; qcs.append(q)
    if qcs:
        qc_all = pd.concat(qcs, ignore_index=True)
        rfig.contamination_group_figure(
            qc_all, out_stem=os.path.join(out_dir, "contamination_group"),
            title=f"Artifact rejection across {len(qc_all)} derivations, "
                  f"{len(sessions)} sessions")
        n_exc = int(qc_all.excluded.fillna(False).sum())
        print(f"  contamination figure: {len(qc_all)} derivations, "
              f"{n_exc} above 2/3")

    # --- where the hippocampal contacts are --------------------------------
    # Built here rather than only in swr_contact_figure.py so that one command
    # produces every group figure the manuscript needs; the script remains for
    # rebuilding it alone against a downloaded run.
    try:
        import swr_contact_figure as scf
        gdir = os.path.join(swr_io.derivatives_dir(R), "group", "swr")
        scf.make_figure(group_dir=gdir, out_dir=out_dir, verbose=False)
    except Exception as e:
        print(f"  [contact figure skipped: {type(e).__name__}: {e}]")
    return None


def sharpwave_control(session, analysis_name=ANALYSIS_NAME, win_s=WIN_S):
    """Is the sharp wave there before the bipolar subtraction?

    The bipolar montage is a spatial derivative. The ripple is spatially focal
    and survives it; the sharp wave is a broad low-frequency dipole largely
    common to both contacts of a pair, so subtraction should remove it. That is
    a testable claim rather than an excuse, and this tests it: take the events
    already detected on the bipolar signal, and average the SAME time points on
    each contact separately, before subtraction.

    If a slow deflection appears in the monopolar traces and cancels in their
    difference, the montage explains the missing sharp wave. If it is absent
    monopolar too, it was never recorded and the events should not be called
    sharp-wave ripples on any montage.

    Detection is untouched -- this only re-averages existing event times.
    """
    session = int(session)
    R = swr_io.get_data_root()
    clean_dir = os.path.join(swr_io.session_deriv_dir(session, R),
                             "LFP-clean", analysis_name)
    rip_dir = os.path.join(swr_io.session_deriv_dir(session, R),
                           "LFP-ripples", analysis_name)
    pairs = pd.read_csv(os.path.join(clean_dir, "pairs.csv"))
    with open(os.path.join(clean_dir, "meta.json")) as f:
        fs = float(json.load(f)["fs"])
    ev = pd.read_csv(os.path.join(rip_dir, "ripple_events.csv"))
    ev = ev[ev.passed.fillna(False)]
    bip = np.load(os.path.join(clean_dir, "continuous.npy"), mmap_mode="r")

    cfg_s = swr_io.session_config(session, data_root=R)
    _, kind, _ = swr_io.discover_raw_files(session, cfg_s, data_root=R)
    print(f"  s{session:02d}: re-reading raw WITHOUT bipolar subtraction "
          f"({kind}) ...")
    mono, mmeta = pre.preprocess_session(session, pairs, data_root=R,
                                         verbose=False, monopolar=True)
    ch_ids = list(mmeta["pair_ids"])
    n = min(mono.shape[1], bip.shape[1])
    half = int(round(win_s * fs))

    rows, prof = [], {}
    for i, p_ in pairs.iterrows():
        if i >= bip.shape[0]:
            continue
        e = ev[ev.pair_id == p_.pair_id]
        if len(e) < 30:
            continue
        raw_b = np.asarray(bip[i], float)[:n]
        pk = rfig.trough_lock(e, rfig._bp(raw_b, fs, *rfig.RIPPLE_BAND))
        pk = pk[(pk - half >= 0) & (pk + half < n)]
        # The monopolar rows are keyed the way the reader keys them: Blackrock
        # by nsx position, Neuralynx by .ncs stem. Guessing from "is ns_pos
        # non-NaN" silently missed every UCLA session.
        if kind == "blackrock":
            ka, kb = str(int(p_.ns_pos_a)), str(int(p_.ns_pos_b))
        else:
            ka, kb = str(p_.ns_label_a), str(p_.ns_label_b)
        for tag, key, x in (("contact A", ka, None), ("contact B", kb, None),
                            ("bipolar A-B", None, raw_b)):
            if x is None:
                if key not in ch_ids:
                    print(f"    {p_.pair_id}: channel {key} not in monopolar set")
                    continue
                x = np.asarray(mono[ch_ids.index(key)], float)[:n]
            pr = rfig.sharp_wave_profiles(x, fs, pk, win_s=win_s)
            if pr is None:
                continue
            prof[f"{p_.pair_id} | {tag}"] = pr
            rows.append(dict(session=session, pair_id=p_.pair_id, trace=tag,
                             channel=key or p_.pair_id, n=pr["n"],
                             deflection_uv=pr["deflection_uv"],
                             flank_sd=pr["flank_sd"], snr=pr["snr"]))
    if not rows:
        print("  nothing to compare"); return None
    tab = pd.DataFrame(rows)
    out_dir = os.path.join(rip_dir, "figures")
    rfig.monopolar_sharpwave_figure(
        prof, out_stem=os.path.join(out_dir, "sharp_wave_monopolar"),
        title=f"s{session:02d}: is the sharp wave present before bipolar subtraction?")
    tab.to_csv(os.path.join(rip_dir, "sharp_wave_monopolar.csv"), index=False)

    print("\n" + tab[["pair_id", "trace", "n", "deflection_uv", "snr"]]
          .round(2).to_string(index=False))
    # Compare deflection in microvolts, NOT SNR. Bipolar subtraction shrinks the
    # flank noise as well as the signal, so its SNR can exceed the monopolar SNR
    # even while the deflection itself collapses -- which is exactly the effect
    # being tested. An earlier version compared SNR and concluded the opposite
    # of what the amplitudes show.
    mono = (tab[tab.trace != "bipolar A-B"].groupby("pair_id")
            .deflection_uv.apply(lambda v: v.abs().mean()))
    bip = (tab[tab.trace == "bipolar A-B"].set_index("pair_id")
           .deflection_uv.abs())
    comp = pd.DataFrame({"monopolar_uv": mono, "bipolar_uv": bip}).dropna()
    if not len(comp):
        print("\n  -> monopolar channels could not be matched; NO CONCLUSION.")
        return None
    comp["reduction_x"] = comp.monopolar_uv / comp.bipolar_uv.replace(0, np.nan)
    print("\n  slow deflection, monopolar vs bipolar (microvolts):")
    print("  " + comp.round(2).to_string().replace("\n", "\n  "))

    HAS_SW_UV = 2.0        # a deflection this small is not a sharp wave
    strong = comp[comp.monopolar_uv >= HAS_SW_UV]
    if not len(strong):
        print(f"\n  -> no contact shows a slow deflection above {HAS_SW_UV} uV even "
              "before subtraction;\n     on these contacts there is no sharp wave "
              "to lose.")
    elif strong.reduction_x.median() >= 2.0:
        print(f"\n  -> {len(strong)} contact(s) carry a slow deflection before "
              f"subtraction\n     (median {strong.monopolar_uv.median():.1f} uV), "
              f"reduced {strong.reduction_x.median():.1f}x by the bipolar "
              "difference:\n     the montage explains the missing sharp wave.")
    else:
        print(f"\n  -> {len(strong)} contact(s) carry a slow deflection that the "
              "bipolar difference\n     does NOT remove "
              f"({strong.reduction_x.median():.1f}x): the montage is not the "
              "explanation here.")
    comp.to_csv(os.path.join(rip_dir, "sharp_wave_monopolar_summary.csv"))
    print(f"  -> {os.path.join(rip_dir, 'sharp_wave_monopolar.csv')}")
    return None


def sharpwave_examples(session, analysis_name=ANALYSIS_NAME, n=12, win_s=WIN_S):
    """The clearest single ripples in a session, ranked by sharp-wave depth.

    The grand average is trough-locked, so the ripple survives averaging while
    the sharp wave -- whose polarity depends on which side of the CA1 pyramidal
    layer a contact sits (SS6.4) -- partly cancels. A publication figure wants a
    SINGLE event where both are visible on the same trace, and that has to be
    chosen by eye from candidates rather than by averaging.

    Scores every detected event on the MONOPOLAR contacts, before the bipolar
    subtraction removes the spatially broad sharp wave, and keeps the best `n`.

        score = sharp-wave trough depth at the ripple, in flank SD

    Writes, per session, into LFP-ripples/{name}/:
        sharpwave_examples_best.pdf   contact sheet, one panel per candidate
        sharpwave_examples_best.npz   the raw snippets, to replot the chosen one
        sharpwave_examples_best.csv   pair_id, t_peak_s, score -- the index

    Nothing here feeds detection or statistics; it only re-reads existing event
    times to choose a figure.
    """
    session = int(session)
    R = swr_io.get_data_root()
    clean_dir = os.path.join(swr_io.session_deriv_dir(session, R),
                             "LFP-clean", analysis_name)
    rip_dir = os.path.join(swr_io.session_deriv_dir(session, R),
                           "LFP-ripples", analysis_name)
    pairs = pd.read_csv(os.path.join(clean_dir, "pairs.csv"))
    with open(os.path.join(clean_dir, "meta.json")) as f:
        fs = float(json.load(f)["fs"])
    ev = pd.read_csv(os.path.join(rip_dir, "ripple_events.csv"))
    ev = ev[ev.passed.fillna(False)]
    if not len(ev):
        print(f"  s{session:02d}: no passed events"); return None
    bip = np.load(os.path.join(clean_dir, "continuous.npy"), mmap_mode="r")

    cfg_s = swr_io.session_config(session, data_root=R)
    _, kind, _ = swr_io.discover_raw_files(session, cfg_s, data_root=R)
    mono, mmeta = pre.preprocess_session(session, pairs, data_root=R,
                                         verbose=False, monopolar=True)
    ch_ids = list(mmeta["pair_ids"])
    n_s = min(mono.shape[1], bip.shape[1])
    half = int(round(win_s * fs))
    edge = int(round(0.05 * fs))

    cands = []
    for i, p_ in pairs.iterrows():
        if i >= bip.shape[0]:
            continue
        e = ev[ev.pair_id == p_.pair_id]
        if not len(e):
            continue
        raw_b = np.asarray(bip[i], float)[:n_s]
        pk = rfig.trough_lock(e, rfig._bp(raw_b, fs, *rfig.RIPPLE_BAND))
        keep = (pk - half >= 0) & (pk + half < n_s)
        pk, e = pk[keep], e.iloc[keep]
        if not len(pk):
            continue
        if kind == "blackrock":
            keys = (str(int(p_.ns_pos_a)), str(int(p_.ns_pos_b)))
        else:
            keys = (str(p_.ns_label_a), str(p_.ns_label_b))
        for key in keys:
            if key not in ch_ids:
                continue
            x = np.asarray(mono[ch_ids.index(key)], float)[:n_s]
            sw = rfig._lp(x, fs, rfig.SW_BAND_HZ)
            for j, c in enumerate(pk):
                seg = sw[c - half:c + half]
                flank = np.r_[seg[:edge], seg[-edge:]]
                sd = float(np.std(flank))
                if sd <= 0:
                    continue
                mid = seg[half - int(0.03 * fs):half + int(0.03 * fs)]
                depth = float(np.max(np.abs(mid - np.median(flank)))) / sd
                cands.append({"pair_id": p_.pair_id, "contact": key,
                              "t_peak_s": float(e.t_peak_s.iloc[j]),
                              "score": depth, "centre": int(c),
                              "raw": x[c - half:c + half].copy(),
                              "bip": raw_b[c - half:c + half].copy()})
    if not cands:
        print(f"  s{session:02d}: no candidates"); return None

    cands.sort(key=lambda d: -d["score"])
    best = cands[:int(n)]
    idx = pd.DataFrame([{k: b[k] for k in ("pair_id", "contact", "t_peak_s", "score")}
                        for b in best])
    idx.insert(0, "session", session)
    idx.insert(1, "rank", np.arange(1, len(best) + 1))
    idx.to_csv(os.path.join(rip_dir, "sharpwave_examples_best.csv"), index=False)
    np.savez_compressed(
        os.path.join(rip_dir, "sharpwave_examples_best.npz"),
        raw=np.array([b["raw"] for b in best]),
        bip=np.array([b["bip"] for b in best]),
        fs=fs, win_s=win_s,
        pair_id=np.array([b["pair_id"] for b in best]),
        contact=np.array([b["contact"] for b in best]),
        t_peak_s=np.array([b["t_peak_s"] for b in best]),
        score=np.array([b["score"] for b in best]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = (np.arange(-half, half) / fs) * 1000.0
    ncol = 3
    nrow = int(np.ceil(len(best) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.1 * nrow),
                             sharex=True)
    for ax, b in zip(np.atleast_1d(axes).ravel(), best):
        sw = rfig._lp(b["raw"], fs, rfig.SW_BAND_HZ)
        rip = rfig._bp(b["raw"], fs, *rfig.RIPPLE_BAND)
        ax.plot(t, b["raw"], color="0.35", lw=0.6)
        ax.plot(t, sw, color="#0e3d3a", lw=1.6, label=f"sharp wave (<{rfig.SW_BAND_HZ:.0f} Hz)")
        ax.plot(t, rip * 3 + np.max(b["raw"]) * 0.9, color="#F15A29", lw=0.7,
                label="80–120 Hz (×3)")
        ax.set_title(f"#{best.index(b)+1} {b['pair_id']} ch{b['contact']}\n"
                     f"t={b['t_peak_s']:.1f}s  score={b['score']:.1f}", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in np.atleast_1d(axes).ravel()[len(best):]:
        ax.axis("off")
    np.atleast_1d(axes).ravel()[0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"s{session:02d}: clearest sharp-wave ripples (monopolar, "
                 f"pre-subtraction)", fontsize=10)
    fig.supxlabel("Time from ripple trough (ms)", fontsize=9)
    fig.supylabel(r"Voltage ($\mu$V)", fontsize=9)
    fig.tight_layout()
    out = os.path.join(rip_dir, "sharpwave_examples_best.pdf")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  s{session:02d}: {len(best)} candidates -> {out}")
    return idx


def _metrics_cli(session, analysis_name=ANALYSIS_NAME):
    """qc_metrics returns a DataFrame so qc_report can reuse it; fire would render
    that as an attribute listing instead of the printed table."""
    qc_metrics(int(session), analysis_name)


def qc_report(session=None, analysis_name=ANALYSIS_NAME, max_events=800,
              group=True):
    """QC every session, or just one.

    session : None (default) runs every session that has detection output on
              this machine, then the group figure and the group triage table.
              Pass --session=N for a single session.
    """
    if session is not None:
        return _qc_report_one(int(session), analysis_name, max_events)

    sess = sessions_with_ripples(analysis_name)
    if not sess:
        print("no sessions with ripple_events.csv found"); return None
    print(f"QC across {len(sess)} sessions: "
          + ", ".join(f"s{s:02d}" for s in sess) + "\n")
    for s_ in sess:
        print("=" * 74); print(f" s{s_:02d}"); print("=" * 74)
        try:
            _qc_report_one(s_, analysis_name, max_events)
        except Exception as e:                    # one bad session must not
            print(f"  s{s_:02d} FAILED: {type(e).__name__}: {e}")   # stop the rest
    if group:
        print("\n" + "=" * 74); print(" GROUP"); print("=" * 74)
        group_figure(sess, analysis_name)
        qc_group(analysis_name, sess)
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({'report': qc_report, 'metrics': _metrics_cli,
                   'examples': sharpwave_examples,
                   'group': qc_group, 'figure': group_figure,
                   'sharpwave': sharpwave_control})
    else:
        qc_report(38)
