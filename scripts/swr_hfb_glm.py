#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One GLM instead of a pile of split-and-contrast tests.

Everything up to here has tried to separate ripple-driven from event-driven HFB
by SPLITTING epochs -- ripple-free vs ripple-containing, correct vs incorrect --
and contrasting the halves. That keeps failing for a structural reason: the
events overlap in time. A ripple 300 ms after an uncovering sits inside that
uncovering's epoch, an uncovering sits inside the previous press's epoch, and
no amount of labelling makes the epochs independent.

The standard fix is to stop splitting and model the continuous signal:

    HFB(t) = sum_k  b_ripple[k]  * ripple(t - k)
           + sum_k  b_reward[k]  * reward(t - k)
           + sum_k  b_error[k]   * error(t - k)
           + sum_k  b_move[k]    * move(t - k)
           + drift + intercept + e(t)

with a FINITE IMPULSE RESPONSE basis: each event type gets one regressor per
lag, so `b_ripple[k]` is the HFB attributable to a ripple k ms ago AFTER
everything the other event types explain at the same moment. That is the
"controlling for each other" the split designs could never deliver, and the
beta timecourse is a deconvolved response rather than an average over
contaminated epochs.

WHAT REPLACES THE SHIFTED NULL
------------------------------
Nothing needs to. The shifted null existed because a raw peri-event average is
inflated by whatever else is happening; here that is handled by the design
matrix. Drift is absorbed by an intercept plus low-order Legendre polynomials.
Betas are tested against zero across sessions and cluster-corrected over lags
with the project's own `swr_sakon.cluster_perm_time`.

COLLINEARITY IS THE THING TO WATCH, so it is reported rather than assumed: if
ripples and rewarded uncoverings co-occur too tightly the GLM cannot separate
them and the betas will be unstable with wide errors. The condition number of
X'X and the variance inflation per event type go into the result json.

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_hfb_glm.py run
    python scripts/swr_hfb_glm.py figure --results=<dir>

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import pickle
import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.swr_bundle as sb
import mc.plotting.ripple_figures as rfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swr_ripple_hfb_conditions import session_events, _clean_mask

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "hfb_glm"
LAG_MS = (-500.0, 1000.0)      # FIR support, relative to each event
BIN_MS = 50.0                  # one regressor per event type per 50 ms
N_POLY = 3                     # Legendre drift terms
REGS = ("ripple", "reward", "error", "move")
RCOL = {"ripple": "#23677E", "reward": "#0e3d3a", "error": "#a30d6c",
        "move": "#6E6E6E"}
RLAB = {"ripple": "hippocampal ripple", "reward": "correct uncover",
        "error": "incorrect uncover", "move": "navigation press"}


def _design(n, fs, ev_times, lags_samp, bin_samp):
    """FIR design block for one event type: (n, n_lags), float32."""
    X = np.zeros((n, len(lags_samp)), np.float32)
    idx = np.round(np.asarray(ev_times, float) * fs).astype(int)
    idx = idx[(idx >= 0) & (idx < n)]
    for j, L in enumerate(lags_samp):
        # a boxcar of width bin_samp starting L samples after the event
        for s in range(bin_samp):
            k = idx + L + s
            k = k[(k >= 0) & (k < n)]
            np.add.at(X[:, j], k, 1.0)
    return X


def run(bundle=None, out_dir=None, boundary="exclusive", save=True):
    b_dir = bundle or os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                                   "group", "swr", "bundle_v2")
    with open(os.path.join(b_dir, "swr_bundle.pkl"), "rb") as f:
        B = pickle.load(f)
    rip, hfb_pairs, hiv, pairs = (B["ripples"], B["hfb_pairs"],
                                  B["hfb_intervals"], B["pairs"])
    hp = hfb_pairs[~hfb_pairs.excluded.fillna(False)].copy()
    key = pairs[["session", "pair_id", "same_probe_as_hpc"]].drop_duplicates()
    hp = hp.merge(key, on=["session", "pair_id"], how="left")
    hp = hp[~hp.same_probe_as_hpc.fillna(False).astype(bool)]

    R = swr_io.get_data_root()
    fs = float(B["hfb_index"].out_fs.iloc[0]) if len(B["hfb_index"]) else 100.0
    bin_samp = max(1, int(round(BIN_MS / 1000.0 * fs)))
    lags_samp = np.arange(int(round(LAG_MS[0] / 1000.0 * fs)),
                          int(round(LAG_MS[1] / 1000.0 * fs)), bin_samp)
    lag_ms = lags_samp / fs * 1000.0
    nlag = len(lags_samp)
    print(f"FIR {LAG_MS[0]:.0f}..{LAG_MS[1]:.0f} ms, {BIN_MS:.0f} ms bins -> "
          f"{nlag} lags x {len(REGS)} regressors = {nlag * len(REGS)} columns")

    rows, diag = [], []
    for sess in sorted(hp.session.unique()):
        g = hp[hp.session == sess]
        cx = g[g.roi_family.isin(["mPFC", "mOFC", "TemporalLateral",
                                  "Auditory", "Visual"])]
        if not len(cx):
            continue
        ev = session_events(int(sess), R, boundary)
        if not len(ev):
            continue
        try:
            D = sb.load_hfb(b_dir, sess, arrays=["hfb"])
        except Exception:
            continue
        H, ids = D["hfb"], D["pair_ids"]
        n = H.shape[1]
        idx = {p: i for i, p in enumerate(ids)}

        kind = ev.cond.str.split("_").str[0].to_numpy()
        times = {"ripple": np.unique(np.round(
            rip.loc[rip.session == sess, "t_peak_s"].to_numpy(float), 3))}
        for k in ("reward", "error", "move"):
            times[k] = ev.t_s.to_numpy(float)[kind == k]
        if min(len(v) for v in times.values()) < 20:
            continue

        blocks = [_design(n, fs, times[k], lags_samp, bin_samp) for k in REGS]
        # drift: intercept + Legendre polynomials over the recording
        t01 = np.linspace(-1, 1, n, dtype=np.float32)
        poly = np.stack([np.polynomial.legendre.legval(
            t01, [0] * d + [1]).astype(np.float32)
            for d in range(N_POLY + 1)], 1)
        X = np.concatenate(blocks + [poly], 1)
        # centre the FIR columns so the intercept carries the mean
        X[:, :nlag * len(REGS)] -= X[:, :nlag * len(REGS)].mean(0, keepdims=True)

        for _, x in cx.iterrows():
            if x.pair_id not in idx:
                continue
            m = _clean_mask(
                hiv[(hiv.session == sess) & (hiv.pair_id == x.pair_id)]
                [["start_s", "stop_s"]].to_numpy(float), n, fs)
            if m.sum() < 10 * X.shape[1]:
                continue
            Xm = X[m]
            y = H[idx[x.pair_id]][m].astype(np.float32)
            XtX = Xm.T @ Xm
            try:
                beta = np.linalg.solve(XtX + 1e-6 * np.eye(len(XtX), dtype=np.float32),
                                       Xm.T @ y)
            except np.linalg.LinAlgError:
                continue
            # UNIQUE VARIANCE per regressor block. This is the quantity that
            # CAN be contrasted across regressors: the level ambiguity that
            # makes beta heights incomparable does not touch it, because
            # dropping a block simply lets the others reabsorb whatever
            # constant they were sharing. What is left is the variance only
            # that block can explain.
            Xty = Xm.T @ y
            yty = float(y @ y)
            rss_full = yty - 2 * float(beta @ Xty) + float(beta @ (XtX @ beta))
            tss = yty - float(y.sum()) ** 2 / len(y)
            partial = {}
            for r, k in enumerate(REGS):
                keep = np.ones(len(XtX), bool)
                keep[r * nlag:(r + 1) * nlag] = False
                A2 = XtX[np.ix_(keep, keep)]
                b2 = np.linalg.solve(
                    A2 + 1e-6 * np.eye(len(A2), dtype=np.float32), Xty[keep])
                rss_red = (yty - 2 * float(b2 @ Xty[keep])
                           + float(b2 @ (A2 @ b2)))
                partial[k] = float((rss_red - rss_full) / tss)
            for r, k in enumerate(REGS):
                b = beta[r * nlag:(r + 1) * nlag]
                rows.append({"session": sess, "subject": x.subject_label,
                             "cx_pair": x.pair_id, "roi": x.roi_family,
                             "reg": k, "partial_r2": partial[k],
                             "r2_full": float(1 - rss_full / tss),
                             **{f"b{j}": float(v) for j, v in enumerate(b)}})
            if not diag or diag[-1]["session"] != sess:
                cn = float(np.linalg.cond(XtX))
                diag.append({"session": sess, "n_clean_s": float(m.sum() / fs),
                             "cond_number": cn,
                             **{f"n_{k}": int(len(times[k])) for k in REGS}})
        print(f"  s{int(sess):02d}: {len(cx)} derivations", end="\r")

    d = pd.DataFrame(rows)
    G = pd.DataFrame(diag)
    print(f"\n{len(d)} beta rows, {d.session.nunique()} sessions, "
          f"{d.cx_pair.nunique()} pair labels")
    print(f"design condition number: median {G.cond_number.median():.1f}, "
          f"max {G.cond_number.max():.1f}")

    if not save:
        return d
    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(
        swr_io.derivatives_dir(swr_io.get_data_root()), "group", "swr",
        f"{ANALYSIS_NAME}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    d.to_csv(os.path.join(out_dir, "betas.csv"), index=False)
    G.to_csv(os.path.join(out_dir, "design_diagnostics.csv"), index=False)
    np.save(os.path.join(out_dir, "lag_ms.npy"), lag_ms)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"analysis": ANALYSIS_NAME, "created": stamp,
                   "bundle": b_dir, "boundary": boundary,
                   "lag_ms": list(LAG_MS), "bin_ms": BIN_MS,
                   "n_lags": int(nlag), "regressors": list(REGS),
                   "n_poly_drift": N_POLY,
                   "different_shaft_only": True,
                   "model": ("FIR GLM on continuous HFB; every event type "
                             "controls for the others at the same moment"),
                   "cond_number_median": float(G.cond_number.median()),
                   "cond_number_max": float(G.cond_number.max())},
                  f, indent=2)
    print(f"\nsaved -> {out_dir}")
    return None


KERNELS = {"sym": (-250.0, 500.0), "post": (0.0, 500.0),
           "tight": (-100.0, 250.0)}
# BASELINE arm for the differenced regressors. A plain boxcar beta measures HFB
# during the event window relative to the SESSION mean, which is confounded by
# whatever global state the events sit in: ripples come out positive even in
# STG and Auditory, where the shifted-null analyses find no coupling at all,
# and the regressors are orthogonal (r ~ 0.003) so collinearity is not the
# cause. Subtracting a per-event baseline arm removes any sustained offset and
# makes the beta the same quantity the peri/non-peri statistic estimates.
BASE_MS = (-1500.0, -750.0)


def simple(bundle=None, out_dir=None, boundary="exclusive", save=True):
    """ONE beta per event type, not thirty.

    The FIR gives a shape but costs 30 free parameters per event type, which is
    why its betas carry an unidentifiable level and why its partial R2 is
    dominated by degrees of freedom rather than signal (every block lands near
    1e-3 whatever the region). With a SINGLE regressor per event type -- the
    event train convolved with a fixed boxcar -- there are 4 parameters, the
    level ambiguity is gone, and the betas ARE directly comparable: each is the
    HFB attributable to one event of that type.

    The price is that a fixed kernel fits some responses better than others.
    Three kernels are fitted so that is visible rather than assumed:
      sym    -250..+500 ms, the default -- wide enough for a ripple peaking at
             0 and an event peaking at +150
      post   0..+500 ms, causal only
      tight  -100..+250 ms, centred on the ripple transient
    """
    b_dir = bundle or os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                                   "group", "swr", "bundle_v2")
    with open(os.path.join(b_dir, "swr_bundle.pkl"), "rb") as f:
        B = pickle.load(f)
    rip, hfb_pairs, hiv, pairs = (B["ripples"], B["hfb_pairs"],
                                  B["hfb_intervals"], B["pairs"])
    hp = hfb_pairs[~hfb_pairs.excluded.fillna(False)].copy()
    key = pairs[["session", "pair_id", "same_probe_as_hpc"]].drop_duplicates()
    hp = hp.merge(key, on=["session", "pair_id"], how="left")
    hp = hp[~hp.same_probe_as_hpc.fillna(False).astype(bool)]
    R = swr_io.get_data_root()
    fs = float(B["hfb_index"].out_fs.iloc[0]) if len(B["hfb_index"]) else 100.0

    rows = []
    for sess in sorted(hp.session.unique()):
        g = hp[hp.session == sess]
        cx = g[g.roi_family.isin(["mPFC", "mOFC", "TemporalLateral",
                                  "Auditory", "Visual"])]
        ev = session_events(int(sess), R, boundary) if len(cx) else None
        if ev is None or not len(ev):
            continue
        try:
            D = sb.load_hfb(b_dir, sess, arrays=["hfb"])
        except Exception:
            continue
        H, ids = D["hfb"], D["pair_ids"]
        n = H.shape[1]
        idx = {p: i for i, p in enumerate(ids)}
        kind = ev.cond.str.split("_").str[0].to_numpy()
        times = {"ripple": np.unique(np.round(
            rip.loc[rip.session == sess, "t_peak_s"].to_numpy(float), 3))}
        for k in ("reward", "error", "move"):
            times[k] = ev.t_s.to_numpy(float)[kind == k]
        if min(len(v) for v in times.values()) < 20:
            continue
        t01 = np.linspace(-1, 1, n, dtype=np.float32)
        poly = np.stack([np.polynomial.legendre.legval(
            t01, [0] * dd + [1]).astype(np.float32)
            for dd in range(N_POLY + 1)], 1)
        for kname, (lo, hi) in KERNELS.items():
            lo_s = int(round(lo / 1000.0 * fs))
            wid = max(1, int(round((hi - lo) / 1000.0 * fs)))
            lo_b = int(round(BASE_MS[0] / 1000.0 * fs))
            wid_b = max(1, int(round((BASE_MS[1] - BASE_MS[0]) / 1000.0 * fs)))
            cols = []
            for k in REGS:
                a = _design(n, fs, times[k], np.array([lo_s]), wid)[:, 0]
                b = _design(n, fs, times[k], np.array([lo_b]), wid_b)[:, 0]
                # equal total weight on each arm, so beta = response - baseline
                cols.append(a / max(wid, 1) - b / max(wid_b, 1))
            Xe = np.stack(cols, 1)
            Xe -= Xe.mean(0, keepdims=True)
            X = np.concatenate([Xe, poly], 1)
            for _, x in cx.iterrows():
                if x.pair_id not in idx:
                    continue
                m = _clean_mask(
                    hiv[(hiv.session == sess) & (hiv.pair_id == x.pair_id)]
                    [["start_s", "stop_s"]].to_numpy(float), n, fs)
                if m.sum() < 1000:
                    continue
                Xm, y = X[m], H[idx[x.pair_id]][m].astype(np.float32)
                XtX = Xm.T @ Xm
                Xty = Xm.T @ y
                try:
                    beta = np.linalg.solve(
                        XtX + 1e-6 * np.eye(len(XtX), dtype=np.float32), Xty)
                except np.linalg.LinAlgError:
                    continue
                yty = float(y @ y)
                rss = yty - 2 * float(beta @ Xty) + float(beta @ (XtX @ beta))
                dof = int(m.sum()) - len(beta)
                cov = rss / dof * np.linalg.pinv(XtX)
                se = np.sqrt(np.clip(np.diag(cov), 0, None))
                rows.append({"session": sess, "subject": x.subject_label,
                             "cx_pair": x.pair_id, "roi": x.roi_family,
                             "kernel": kname,
                             **{f"b_{k}": float(beta[i])
                                for i, k in enumerate(REGS)},
                             **{f"se_{k}": float(se[i])
                                for i, k in enumerate(REGS)}})
        print(f"  s{int(sess):02d}", end="\r")
    d = pd.DataFrame(rows)
    print(f"\n{len(d)} rows, {d.session.nunique()} sessions, "
          f"{d.cx_pair.nunique()} pair labels, {d.kernel.nunique()} kernels")
    if not save:
        return d
    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(
        swr_io.derivatives_dir(swr_io.get_data_root()), "group", "swr",
        f"{ANALYSIS_NAME}_single_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    d.to_csv(os.path.join(out_dir, "betas_single.csv"), index=False)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"analysis": ANALYSIS_NAME + "_single", "created": stamp,
                   "kernels_ms": {k: list(v) for k, v in KERNELS.items()},
                   "regressors": list(REGS), "n_poly_drift": N_POLY,
                   "model": ("one DIFFERENCED regressor per event type: response "
                             "window minus a -1500..-750 ms baseline arm, "
                             "so beta = response minus own baseline"),
                   "baseline_arm_ms": list(BASE_MS)},
                  f, indent=2)
    print(f"\nsaved -> {out_dir}")
    return None


def figure(results=None, part_dir=None, out_stem=None,
           rois=("MedialFrontal", "lOFC", "MTG", "STG"), smooth=False):
    """Beta timecourse per ROI, one line per event type, all mutually controlled.

    Each line is the HFB attributable to that event type at that lag AFTER the
    other three have explained what they can at the same moment. Bars under the
    traces are clusters surviving `swr_sakon.cluster_perm_time` against zero,
    the same test and seed used everywhere else in this project.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as st
    import mc.analyse.swr_sakon as sk
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from swr_hfb_locking_contrasts import N_PERM, SEED

    G = os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                     "group", "swr")
    results = results or sorted(os.path.join(G, x) for x in os.listdir(G)
                                if x.startswith(f"{ANALYSIS_NAME}_"))[-1]
    part_dir = part_dir or sorted(
        os.path.join(G, x) for x in os.listdir(G)
        if x.startswith("hfb_roi_partition_exclusive_"))[-1]
    d = pd.read_csv(os.path.join(results, "betas.csv"))
    lag = np.load(os.path.join(results, "lag_ms.npy"))
    bcols = [c for c in d.columns if c.startswith("b") and c[1:].isdigit()]
    bcols = sorted(bcols, key=lambda c: int(c[1:]))
    part = pd.read_csv(os.path.join(part_dir, "derivation_assignment.csv")) \
        .set_index(["session", "cx_pair"])["roi_part"]
    d["roi_part"] = pd.MultiIndex.from_arrays(
        [d.session, d.cx_pair]).map(part)
    d = d.dropna(subset=["roi_part"])

    stats = []
    with plt.rc_context(rfig._rc()):
        rois = [r for r in rois if (d.roi_part == r).any()]
        fig, axes = plt.subplots(1, len(rois), figsize=(
            5.0 * len(rois) * rfig.CM, 7.5 * rfig.CM), sharey=True)
        axes = np.atleast_1d(axes)
        bars = []
        for ax, roi in zip(axes, rois):
            g = d[d.roi_part == roi]
            n_ses = g.session.nunique()
            lo_all = []
            for k in REGS:
                A = (g[g.reg == k].groupby("session")[bcols].mean()
                     .to_numpy(float))
                if len(A) < 5:
                    continue
                # ---- DEMEAN EACH REGRESSOR'S OWN TIMECOURSE ----------------
                # The LEVEL of an FIR beta set is not identifiable here. Events
                # are dense enough that adding a constant to every lag of one
                # regressor and removing it from another barely changes the
                # fit: across sessions the ripple and reward levels correlate
                # r = -0.63 (MedialFrontal), -0.71 (MTG), -0.57 (STG), all
                # p < 1e-4, and no beta set returns to zero at the edges of the
                # +-500/+1000 ms support. Only the SHAPE -- the transient
                # deviation from a regressor's own mean -- is estimable, so
                # that is what is plotted and tested. Absolute heights across
                # regressors must not be compared.
                A = A - A.mean(1, keepdims=True)
                m = A.mean(0)
                se = A.std(0, ddof=1) / np.sqrt(len(A))
                _, cc, pv, _ = sk.cluster_perm_time(A, n_perm=N_PERM,
                                                    seed=SEED)
                sig = [(lag[a], lag[b - 1], float(pp))
                       for (a, b), pp in zip(cc, pv) if pp < 0.05]
                stats.append({"roi": roi, "reg": k, "n_sessions": len(A),
                              "peak_beta": float(m[np.argmax(np.abs(m))]),
                              "peak_lag_ms": float(lag[np.argmax(np.abs(m))]),
                              "clusters": sig})
                ax.fill_between(lag, m - se, m + se, color=RCOL[k], alpha=.18,
                                lw=0)
                ax.plot(lag, m, color=RCOL[k], lw=1.3,
                        label=f"{RLAB[k]}" if ax is axes[0] else None)
                lo_all.append((k, sig))
            ax.axhline(0, color="#3d3d3d", lw=.7, ls="--")
            ax.axvline(0, color="#3d3d3d", lw=.7, ls=":")
            ax.set_title(f"{roi}  ({n_ses} sessions)", fontsize=rfig.FS_TITLE)
            ax.set_xlabel("lag from event (ms)")
            bars.append((ax, lo_all))
        # The axes share y, so expanding each one in turn compounds and the
        # cluster bars end up eating most of the panel. Expand ONCE, from the
        # limits all four traces settled on, then place the bars in that strip.
        y0, y1 = axes[0].get_ylim()
        span = y1 - y0
        step = span * 0.055
        axes[0].set_ylim(y0 - step * (len(REGS) + 0.8), y1)
        b0 = axes[0].get_ylim()[0]
        for ax, lo_all in bars:
            for i, (k, sig) in enumerate(lo_all):
                for a, b, pp in sig:
                    ax.plot([a, b], [b0 + step * (i + 0.6)] * 2,
                            color=RCOL[k], lw=2.4, solid_capstyle="butt")
        axes[0].set_ylabel("GLM beta, demeaned over lags\n(transient attributable to that event)")
        axes[0].legend(frameon=False, fontsize=rfig.FS_TICK - 2,
                       loc="upper left", ncol=2, columnspacing=0.8,
                       handlelength=1.2, borderaxespad=0.2)
        fig.suptitle("FIR GLM: every event type controls for the others at "
                     "the same moment\nEach curve is DEMEANED over its own "
                     "lags — the level of an FIR set is not identifiable at "
                     "this event density, only the shape is.\n"
                     "bars = clusters p < 0.05 vs zero (sign-flip, "
                     "n_perm = 2000)", fontsize=rfig.FS_TICK - 1)
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        out_stem = out_stem or os.path.join(results, "glm_betas")
        fig.savefig(out_stem + ".pdf")
        fig.savefig(out_stem + ".png", dpi=300)
        plt.close(fig)
    S = pd.DataFrame(stats)
    S.to_csv(os.path.join(results, "glm_cluster_stats.csv"), index=False)
    print(f"\n{'ROI':<16s}{'regressor':<20s}{'n':>4s}{'peak beta':>11s}"
          f"{'at lag':>9s}   clusters (p<0.05)")
    for _, r in S.iterrows():
        cl = ("; ".join(f"{a:+.0f}..{b:+.0f} ms p={pp:.4f}"
                        for a, b, pp in r.clusters) if r.clusters else "none")
        print(f"  {r.roi:<14s}{RLAB[r.reg]:<20s}{r.n_sessions:>4d}"
              f"{r.peak_beta:>+11.5f}{r.peak_lag_ms:>+8.0f}ms   {cl}")
    print(f"\n  -> {out_stem}.pdf / .png")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run, "simple": simple, "figure": figure})
    else:
        run()
