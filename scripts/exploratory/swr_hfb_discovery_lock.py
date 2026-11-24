#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mOFC vs mPFC HFB at discovery: locked to the hippocampal ripple, or to the
reward uncover?

Pre-registered in CHANGELOG.md, 2026-11-14 (F4). Follow-up to claim 1 panel a
(`hfb_mofc_mpfc_phases_2026-10-08`, first-traversal ripples).

Sets, per session (partition ROIs as claim 1a; 20 shifted nulls):
    R_FT       ripples in the first traversal (grid onset -> first t_D)
    R_FD       ripples in the pause after a first discovery,
               (press, press + min(pause, 2 s)]
    R_ALL      every in-task ripple (summary set, 2026-11-14 (b))
    U_COR / U_ERR / U_FT_ERR  every correct / incorrect uncover, incorrect
               uncovers of the first traversal (summary sets)
    U_FT       every uncover in the first traversal (correct and incorrect)
    U_FD       first discoveries (correct uncovers in the first traversal)
    U_FD_rip   ... with a ripple in their post-reward pause
    U_FD_norip ... without

TWO READINGS
    own      real - mean of the shifted nulls, minus the epoch's own baseline
             (ripple: |t| >= 750 ms; uncover: -1.5..-1.0 s) -- claim 1a's
             reading (`swr_hfb_direction.analyse_pair` conventions)
    matched  uncovers: minus arrow presses followed by the same stillness
             (`fin.matched_sets`), raw. Ripples: minus ripple-free still
             moments (no ripple within 0.5 s, 50 ms grid) of the SAME period
             (first-traversal time for R_FT, post-FD pauses for R_FD), in the
             same stratum of time since the last press x time to the next one
             (`future_shift` bin edges), raw

THE GLM -- what the epoch views cannot do: post-FD ripples sit inside the FD
epochs. `swr_hfb_glm.simple`'s differenced single-beta model (kernel
-250..+500 ms minus a -1500..-750 ms arm) on the continuous HFB, per
derivation, with the events split so that they compete:
    rip_postFD, rip_FT_other, rip_later,
    FD, FT_error, correct_later, error_later, move
    own      as above
    still    + three boxcars for pause state (0.5-1, 1-2, > 2 s since the
             last press), so a sustained stillness level cannot load on the
             event betas
PRIMARY (Q2): per ROI, beta(FD) - beta(rip_postFD); > 0 uncover-locked,
< 0 ripple-locked. PRIMARY (Q1): R_FD, +-250 ms, mOFC - mPFC, own reading.

Usage:
    conda activate env_multiple_clocks
    python scripts/exploratory/swr_hfb_discovery_lock.py all
    python scripts/exploratory/swr_hfb_discovery_lock.py epochs
    python scripts/exploratory/swr_hfb_discovery_lock.py glm

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import pickle
import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter1d

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "exploratory"))
import mc.analyse.swr_sakon as sk
import scripts.swr_final_ripple_analysis as fin
from swr_plan_vs_store import (_paths, _hfb_shifts, value_stats, _save_json,
                               SEED, N_SHIFTS_HFB)
from swr_hfb_ripple_coupling import (load_inputs, session_data,
                                     session_ripples, roi_sums)
from swr_hfb_direction import PARTITION, label_perm, SLIDE_MS, REMOTE_MS
from swr_hfb_gradient import phase_windows
from mc.analyse.swr_explore.discovery_cells import FD_Q, MAX_PAUSE_S
from mc.analyse.swr_explore.future_shift import BIN_EDGES

try:
    import fire
except ImportError:
    fire = None

ANALYSIS_NAME = "hfb_discovery_lock"
ROIS = ("mOFC", "mPFC")
FT_Q = "kind == 'uncover' and stage == 'first uncovers'"
LOCK = {"R_FT": "ripple", "R_FD": "ripple", "U_FT": "event", "U_FD": "event",
        "U_FD_rip": "event", "U_FD_norip": "event",
        # summary sets (2026-11-14 (b))
        "R_ALL": "ripple", "U_COR": "event", "U_ERR": "event",
        "U_FT_ERR": "event"}
TITLE = {"R_FT": "ripples, first traversal",
         "R_FD": "ripples after a first discovery",
         "U_FT": "uncovers, first traversal",
         "U_FD": "first discoveries",
         "U_FD_rip": "first discoveries, ripple follows",
         "U_FD_norip": "first discoveries, no ripple",
         "R_ALL": "every in-task ripple", "U_COR": "every correct uncover",
         "U_ERR": "every incorrect uncover",
         "U_FT_ERR": "incorrect uncovers, first traversal"}
QUERY = {"U_FT": FT_Q, "U_FD": FD_Q,
         "U_FD_rip": FD_Q + " and post_ripple == 1",
         "U_FD_norip": FD_Q + " and post_ripple == 0",
         "U_COR": "kind == 'uncover' and valence == 'correct'",
         "U_ERR": "kind == 'uncover' and valence == 'error'",
         "U_FT_ERR": FT_Q + " and valence == 'error'"}
EPOCH_S = {"ripple": (-1.0, 1.0), "event": (-1.5, 1.5)}
EV_BASE_S = (-1.5, -1.0)
WINDOW_MS = {"ripple": (-250.0, 250.0), "event": (0.0, 500.0)}
MIN_N = {"R_FT": 20, "R_FD": 10, "U_FT": 10, "U_FD": 10, "U_FD_rip": 10,
         "U_FD_norip": 10, "R_ALL": 20, "U_COR": 10, "U_ERR": 10,
         "U_FT_ERR": 10}
CTRL_STEP_S, CTRL_PAD_S, MIN_CTRL = 0.05, 0.5, 5
EDGES = np.r_[0.0, BIN_EDGES]
N_CLUSTER = 10000
# GLM
KERNEL_MS, BASE_MS, N_POLY = (-250.0, 500.0), (-1500.0, -750.0), 3
REGS = ("rip_postFD", "rip_FT_other", "rip_later", "FD", "FT_error",
        "correct_later", "error_later", "move")
STILL_BINS_S = ((0.5, 1.0), (1.0, 2.0), (2.0, np.inf))
MIN_EV_GLM = 5


def _out(out_date=None):
    group, _ = _paths()
    d = os.path.join(group, f"{ANALYSIS_NAME}_{out_date or datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    return d


def _inputs():
    from mc.analyse.swr_explore.discovery_cells import load_inputs as cl_inputs
    data, presses, _ = cl_inputs()          # press table + post_ripple flag
    group, b_dir = _paths()
    rip, hiv, hp = load_inputs(b_dir, os.path.join(group, PARTITION))
    return data, presses, rip, hiv, hp, b_dir


def _windows_mask(t, W):
    m = np.zeros(len(t), bool)
    for a, b in W:
        m |= (t > a) & (t <= b)
    return m


def strata(t, tp):
    """time since the last press x time to the next one, binned."""
    j = np.searchsorted(tp, t, side="right") - 1
    tsp = np.where(j >= 0, t - tp[np.clip(j, 0, None)], np.inf)
    k = j + 1
    tnp = np.where(k < len(tp), tp[np.clip(k, None, len(tp) - 1)] - t, np.inf)
    nb = len(EDGES) - 1
    a = np.clip(np.searchsorted(EDGES, tsp, side="right") - 1, 0, nb - 1)
    b = np.clip(np.searchsorted(EDGES, tnp, side="right") - 1, 0, nb - 1)
    return a * nb + b


def session_sets(P, t_rip, beh):
    """Event times per set, and for the ripple sets the period their
    stillness-matched controls are drawn from."""
    W = phase_windows(beh)
    ft = [(a, b) for ph, a, b in W.itertuples(index=False)
          if ph == "first_traversal"]
    fd = P.query(FD_Q).dropna(subset=["still_next_s"])
    pause = [(t, t + min(s, MAX_PAUSE_S))
             for t, s in zip(fd.t_s.to_numpy(float),
                             fd.still_next_s.to_numpy(float))]
    task = [(a, b) for _, a, b in W.itertuples(index=False)]
    sets = {"R_FT": t_rip[_windows_mask(t_rip, ft)],
            "R_FD": t_rip[_windows_mask(t_rip, pause)],
            "R_ALL": t_rip[_windows_mask(t_rip, task)]}
    for k, q in QUERY.items():
        sets[k] = P.query(q).t_s.to_numpy(float)
    return sets, {"R_FT": ft, "R_FD": pause, "R_ALL": task}


def control_times(period, t_rip):
    g = [np.arange(a + CTRL_STEP_S, b, CTRL_STEP_S) for a, b in period
         if b - a > CTRL_STEP_S]
    t = np.unique(np.round(np.concatenate(g), 3)) if g else np.empty(0)
    r = np.sort(t_rip)
    near = (np.searchsorted(r, t + CTRL_PAD_S) - np.searchsorted(r, t - CTRL_PAD_S))
    return t[near == 0]


def roi_epochs(H, idx, masks, pids, t, off, n, fs, shift=0.0):
    ci = np.round((np.asarray(t) + shift) * fs).astype(int)
    s_, k_ = roi_sums(H, idx, masks, pids, ci, off, n)
    ok = k_ > 0
    return s_[ok] / k_[ok, None], ok


def matched_ripple_curve(X, s_rip, C, s_ctl):
    """mean over ripples of (ripple - mean of its stratum's controls)."""
    out, w = [], []
    for s in np.unique(s_rip):
        c = s_ctl == s
        if c.sum() < MIN_CTRL:
            continue
        r = s_rip == s
        out.append(X[r].mean(0) - C[c].mean(0))
        w.append(r.sum())
    if not w:
        return None, 0
    w = np.asarray(w, float)
    return (np.stack(out) * w[:, None]).sum(0) / w.sum(), int(w.sum())


# ── epochs ───────────────────────────────────────────────────────────────
def epochs(out_date=None):
    d = _out(out_date)
    data, presses, rip, hiv, hp, b_dir = _inputs()
    shifts = _hfb_shifts()
    matched = {k: fin.matched_sets(presses, q, fin.CALIPER_S, fin.REL_CALIPER,
                                   SEED)[:2] for k, q in QUERY.items()}
    beh_all = data["behaviour"]
    curves, counts = {}, []
    sessions = sorted(hp[hp.roi.isin(ROIS)].session.unique())
    for sess in sessions:
        H, idx, masks, g, fs, n = session_data(sess, b_dir, hp, hiv, ROIS)
        pids = {r: g[g.roi == r].pair_id.tolist() for r in ROIS}
        P = presses[presses.session == sess].sort_values("t_s")
        if not len(P):
            continue
        tp = np.sort(P.t_s.to_numpy(float))
        t_rip = session_ripples(rip, sess)
        sets, periods = session_sets(P, t_rip,
                                     beh_all[beh_all.session == sess])
        for key, t in sets.items():
            lock = LOCK[key]
            off = np.arange(int(round(EPOCH_S[lock][0] * fs)),
                            int(round(EPOCH_S[lock][1] * fs)))
            for roi in ROIS:
                if not pids[roi] or len(t) < MIN_N[key]:
                    continue
                # own: real - mean(shifted)
                M = []
                for sh in shifts:
                    X, ok = roi_epochs(H, idx, masks, pids[roi], t, off, n,
                                       fs, sh)
                    if ok.sum() < MIN_N[key]:
                        M = None
                        break
                    M.append(X.mean(0))
                    if sh == 0.0:
                        n_real = int(ok.sum())
                if M is not None:
                    M = np.stack(M)
                    curves[(key, roi, "own", int(sess))] = M[0] - M[1:].mean(0)
                    counts.append(dict(set=key, roi=roi, reading="own",
                                       session=int(sess), n=n_real))
                # matched
                if lock == "event":
                    tg = matched[key][0].get(int(sess))
                    ct = matched[key][1].get(int(sess))
                    if tg is None or len(tg) < MIN_N[key]:
                        continue
                    Xt, okt = roi_epochs(H, idx, masks, pids[roi], tg, off,
                                         n, fs)
                    Xc, okc = roi_epochs(H, idx, masks, pids[roi], ct, off,
                                         n, fs)
                    if okt.sum() < MIN_N[key] or okc.sum() < MIN_N[key]:
                        continue
                    curves[(key, roi, "matched", int(sess))] = \
                        Xt.mean(0) - Xc.mean(0)
                    counts.append(dict(set=key, roi=roi, reading="matched",
                                       session=int(sess), n=int(okt.sum()),
                                       n_ctrl=int(okc.sum())))
                else:
                    tc = control_times(periods[key], t_rip)
                    Xr, okr = roi_epochs(H, idx, masks, pids[roi], t, off, n,
                                         fs)
                    Xc, okc = roi_epochs(H, idx, masks, pids[roi], tc, off,
                                         n, fs)
                    if okr.sum() < MIN_N[key] or not okc.sum():
                        continue
                    c, n_used = matched_ripple_curve(
                        Xr, strata(t[okr], tp), Xc, strata(tc[okc], tp))
                    if c is not None and n_used >= MIN_N[key]:
                        curves[(key, roi, "matched", int(sess))] = c
                        counts.append(dict(set=key, roi=roi,
                                           reading="matched",
                                           session=int(sess), n=n_used,
                                           n_ctrl=int(okc.sum())))
        print(f"  s{int(sess):02d}", flush=True)
    t_ms = {lk: np.arange(int(round(EPOCH_S[lk][0] * fs)),
                          int(round(EPOCH_S[lk][1] * fs))) / fs * 1000.0
            for lk in EPOCH_S}
    with open(os.path.join(d, "curves.pkl"), "wb") as f:
        pickle.dump(dict(curves=curves, t_ms=t_ms), f)
    pd.DataFrame(counts).to_csv(os.path.join(d, "counts.csv"), index=False)
    _save_json(d, "settings_epochs.json", dict(
        analysis=ANALYSIS_NAME, part="epochs",
        created=str(datetime.date.today()),
        script=os.path.relpath(__file__, REPO), seed=SEED,
        n_shifts=N_SHIFTS_HFB, shifts_s=shifts, partition=PARTITION,
        rois=ROIS, sets=TITLE, queries=QUERY, fd=FD_Q,
        post_fd_pause_s=f"(press, press + min(pause, {MAX_PAUSE_S}))",
        epoch_s=EPOCH_S, own_baseline=dict(ripple=f"|t| >= {REMOTE_MS} ms",
                                           event=EV_BASE_S),
        matched=dict(event="fin.matched_sets: arrow presses, same following "
                           f"stillness (caliper {fin.CALIPER_S} s / "
                           f"{fin.REL_CALIPER})",
                     ripple=f"ripple-free still moments (no ripple within "
                            f"{CTRL_PAD_S} s, {CTRL_STEP_S} s grid) of the same "
                            "period, same stratum of time since last press x "
                            f"time to next press, edges {EDGES.tolist()}, "
                            f">= {MIN_CTRL} controls per stratum"),
        min_n=MIN_N, slide_ms=SLIDE_MS, window_ms=WINDOW_MS,
        preregistration="CHANGELOG.md 2026-11-14 F4"))
    return d


def _test_mask(t_ms, lock):
    if lock == "ripple":
        return np.abs(t_ms) < REMOTE_MS
    return t_ms >= EV_BASE_S[1] * 1000


def prepared(curves, t_ms, key, roi, reading):
    """{session: curve} after the reading's baseline and the sliding mean."""
    lock = LOCK[key]
    tt = t_ms[lock]
    if lock == "ripple":
        base = np.abs(tt) >= REMOTE_MS
    else:
        base = (tt >= EV_BASE_S[0] * 1000) & (tt < EV_BASE_S[1] * 1000)
    k = int(round(SLIDE_MS / (tt[1] - tt[0])))
    out = {}
    for (k_, r_, rd_, s), c in curves.items():
        if (k_, r_, rd_) != (key, roi, reading):
            continue
        x = c - c[base].mean() if reading == "own" else c
        out[s] = uniform_filter1d(x, k, mode="nearest")
    return out


def clusters(X, tt, test):
    t_obs, cl, pv, info = sk.cluster_perm_time(X[:, test], n_perm=N_CLUSTER,
                                               seed=SEED)
    ts = tt[test]
    return [dict(start_ms=float(ts[a]), stop_ms=float(ts[b - 1]),
                 sign="increase" if np.nanmean(t_obs[a:b]) > 0 else "decrease",
                 mass=float(np.abs(t_obs[a:b]).sum()), p=float(p))
            for (a, b), p in zip(cl, pv)]


def epoch_tests(out_date=None):
    d = _out(out_date)
    D = pickle.load(open(os.path.join(d, "curves.pkl"), "rb"))
    curves, t_ms = D["curves"], D["t_ms"]
    res, rows = {}, []
    for key in TITLE:
        lock = LOCK[key]
        tt = t_ms[lock]
        test = _test_mask(tt, lock)
        w = (tt >= WINDOW_MS[lock][0]) & (tt <= WINDOW_MS[lock][1])
        for reading in ("own", "matched"):
            vals = {}
            for roi in ROIS:
                C = prepared(curves, t_ms, key, roi, reading)
                if len(C) < 5:
                    continue
                S = sorted(C)
                X = np.stack([C[s] for s in S])
                vals[roi] = pd.Series(X[:, w].mean(1), index=S)
                for s in S:
                    rows.append(dict(set=key, roi=roi, reading=reading,
                                     session=s, window=float(vals[roi][s])))
                res[f"{key} | {reading} | {roi}"] = dict(
                    window=value_stats(vals[roi].to_numpy()),
                    clusters=clusters(X, tt, test),
                    mean=X.mean(0).tolist(),
                    sem=(X.std(0, ddof=1) / np.sqrt(len(X))).tolist())
            if len(vals) == 2:
                a, b = vals["mOFC"], vals["mPFC"]
                diff, p = label_perm(a.to_numpy(), b.to_numpy())
                both = sorted(set(a.index) & set(b.index))
                pr = value_stats((a[both] - b[both]).to_numpy())
                Cm = prepared(curves, t_ms, key, "mOFC", reading)
                Cp = prepared(curves, t_ms, key, "mPFC", reading)
                cl = clusters(np.stack([Cm[s] - Cp[s] for s in both]), tt,
                              test) if len(both) >= 5 else []
                res[f"{key} | {reading} | mOFC - mPFC"] = dict(
                    unpaired=dict(diff=diff, p_perm=p, n_mOFC=len(a),
                                  n_mPFC=len(b)),
                    paired=pr, paired_clusters=cl)
    pd.DataFrame(rows).to_csv(os.path.join(d, "window_values.csv"),
                              index=False)
    _save_json(d, "result_epochs.json", dict(
        t_ms={k: v.tolist() for k, v in t_ms.items()}, tests=res))
    for k, r in res.items():
        if "window" in r:
            w_ = r["window"]
            sig = "; ".join(f"{c['sign'][:3]} {c['start_ms']:+.0f}..{c['stop_ms']:+.0f}"
                            f" p={c['p']:.3f}" for c in r["clusters"]
                            if c["p"] < 0.05) or "-"
            print(f"  {k:42s} n={w_['n']:2d} win {w_.get('mean', np.nan):+.4f} "
                  f"p={w_.get('p_perm', np.nan):.3f}  clusters: {sig}")
        else:
            print(f"  {k:42s} unpaired {r['unpaired']['diff']:+.4f} "
                  f"p={r['unpaired']['p_perm']:.3f} | paired n="
                  f"{r['paired'].get('n')} {r['paired'].get('mean', np.nan):+.4f}"
                  f" p={r['paired'].get('p_perm', np.nan):.3f}")
    return res


# ── GLM ──────────────────────────────────────────────────────────────────
def _box(n, fs, t, lo_ms, hi_ms):
    """Event train convolved with a boxcar over [lo, hi) ms, height 1/width."""
    x = np.zeros(n, np.float32)
    lo = int(round(lo_ms / 1000 * fs))
    wid = max(1, int(round((hi_ms - lo_ms) / 1000 * fs)))
    idx = np.round(np.asarray(t, float) * fs).astype(int) + lo
    for s in range(wid):
        k = idx + s
        k = k[(k >= 0) & (k < n)]
        np.add.at(x, k, 1.0)
    return x / wid


def glm_events(P, t_rip, beh):
    sets, _ = session_sets(P, t_rip, beh)
    ft_rip = sets["R_FT"]
    post = np.isin(t_rip, sets["R_FD"])
    in_ft = np.isin(t_rip, ft_rip)
    unc = "kind == 'uncover'"
    later = "stage != 'first uncovers'"
    ev = {"rip_postFD": t_rip[post],
          "rip_FT_other": t_rip[in_ft & ~post],
          "rip_later": t_rip[~in_ft & ~post],
          "FD": P.query(FD_Q).t_s.to_numpy(float),
          "FT_error": P.query(FT_Q + " and valence == 'error'").t_s.to_numpy(float),
          "correct_later": P.query(f"{unc} and valence == 'correct' and {later}")
          .t_s.to_numpy(float),
          "error_later": P.query(f"{unc} and valence == 'error' and {later}")
          .t_s.to_numpy(float),
          "move": P.query("kind == 'move'").t_s.to_numpy(float)}
    return ev


def still_regressors(n, fs, tp):
    t = np.arange(n) / fs
    j = np.searchsorted(tp, t, side="right") - 1
    tsp = np.where(j >= 0, t - tp[np.clip(j, 0, None)], -1.0)
    return [((tsp >= a) & (tsp < b)).astype(np.float32) for a, b in STILL_BINS_S]


def glm(out_date=None):
    d = _out(out_date)
    data, presses, rip, hiv, hp, b_dir = _inputs()
    from swr_ripple_hfb_conditions import _clean_mask
    import mc.analyse.swr_bundle as sb
    beh_all = data["behaviour"]
    rows, counts = [], []
    for sess in sorted(hp[hp.roi.isin(ROIS)].session.unique()):
        g = hp[(hp.session == sess) & hp.roi.isin(ROIS)]
        P = presses[presses.session == sess].sort_values("t_s")
        if not len(P):
            continue
        D = sb.load_hfb(b_dir, sess, arrays=["hfb"])
        H, fs, n = D["hfb"], D["out_fs"], D["hfb"].shape[1]
        idx = {p: i for i, p in enumerate(D["pair_ids"])}
        t_rip = session_ripples(rip, sess)
        ev = glm_events(P, t_rip, beh_all[beh_all.session == sess])
        counts.append(dict(session=int(sess), **{k: len(v) for k, v in ev.items()}))
        use = [k for k in REGS if len(ev[k]) >= MIN_EV_GLM]
        Xe = np.stack([_box(n, fs, ev[k], *KERNEL_MS) - _box(n, fs, ev[k], *BASE_MS)
                       for k in use], 1)
        Xe -= Xe.mean(0, keepdims=True)
        t01 = np.linspace(-1, 1, n, dtype=np.float32)
        poly = np.stack([np.polynomial.legendre.legval(t01, [0] * k + [1])
                         .astype(np.float32) for k in range(N_POLY + 1)], 1)
        St = np.stack(still_regressors(n, fs, np.sort(P.t_s.to_numpy(float))), 1)
        St -= St.mean(0, keepdims=True)
        for variant, X in (("own", np.concatenate([Xe, poly], 1)),
                           ("still", np.concatenate([Xe, St, poly], 1))):
            for _, x in g.iterrows():
                if x.pair_id not in idx:
                    continue
                m = _clean_mask(hiv[(hiv.session == sess)
                                    & (hiv.pair_id == x.pair_id)]
                                [["start_s", "stop_s"]].to_numpy(float), n, fs)
                if m.sum() < 1000:
                    continue
                Xm, y = X[m], H[idx[x.pair_id]][m].astype(np.float32)
                beta = np.linalg.solve(Xm.T @ Xm + 1e-6 * np.eye(X.shape[1],
                                                                 dtype=np.float32),
                                       Xm.T @ y)
                rows.append(dict(session=int(sess), pair_id=x.pair_id,
                                 roi=x.roi, variant=variant,
                                 **{f"b_{k}": float(beta[i])
                                    for i, k in enumerate(use)}))
        print(f"  GLM s{int(sess):02d}", flush=True)
    B = pd.DataFrame(rows)
    B.to_csv(os.path.join(d, "glm_betas.csv"), index=False)
    pd.DataFrame(counts).to_csv(os.path.join(d, "glm_event_counts.csv"),
                                index=False)
    _save_json(d, "settings_glm.json", dict(
        analysis=ANALYSIS_NAME, part="glm", created=str(datetime.date.today()),
        script=os.path.relpath(__file__, REPO), kernel_ms=KERNEL_MS,
        baseline_arm_ms=BASE_MS, n_poly=N_POLY, regressors=REGS,
        min_events=MIN_EV_GLM, still_bins_s=[list(b) for b in STILL_BINS_S],
        variants=dict(own="event regressors + drift",
                      still="+ pause-state boxcars by time since last press"),
        partition=PARTITION, rois=ROIS,
        preregistration="CHANGELOG.md 2026-11-14 F4"))
    return glm_tests(out_date)


def glm_tests(out_date=None):
    d = _out(out_date)
    B = pd.read_csv(os.path.join(d, "glm_betas.csv"))
    S = B.groupby(["variant", "roi", "session"]).mean(numeric_only=True)\
        .reset_index()
    S.to_csv(os.path.join(d, "glm_session_betas.csv"), index=False)
    res = {}
    for variant in ("own", "still"):
        for roi in ROIS:
            g = S[(S.variant == variant) & (S.roi == roi)]
            for k in REGS:
                res[f"{variant} | {roi} | {k}"] = value_stats(
                    g[f"b_{k}"].to_numpy(float))
            for a, b in (("FD", "rip_postFD"), ("FD", "rip_FT_other"),
                         ("rip_postFD", "rip_FT_other"),
                         ("FT_error", "rip_FT_other")):
                lab = (" PRIMARY" if (a, b) == ("FD", "rip_postFD")
                       and variant == "own" else "")
                res[f"{variant} | {roi} | {a} - {b}{lab}"] = value_stats(
                    (g[f"b_{a}"] - g[f"b_{b}"]).to_numpy(float))
        for k in REGS + ("FD - rip_postFD",):
            def v(roi):
                g = S[(S.variant == variant) & (S.roi == roi)].set_index("session")
                if k == "FD - rip_postFD":
                    return (g.b_FD - g.b_rip_postFD).dropna()
                return g[f"b_{k}"].dropna()
            a, b = v("mOFC"), v("mPFC")
            diff, p = label_perm(a.to_numpy(), b.to_numpy())
            both = sorted(set(a.index) & set(b.index))
            res[f"{variant} | mOFC - mPFC | {k}"] = dict(
                unpaired=dict(diff=diff, p_perm=p, n_mOFC=len(a),
                              n_mPFC=len(b)),
                paired=value_stats((a[both] - b[both]).to_numpy()))
    _save_json(d, "result_glm.json", res)
    for k, r in res.items():
        if "unpaired" in r:
            print(f"  {k:46s} unpaired {r['unpaired']['diff']:+.4f} "
                  f"p={r['unpaired']['p_perm']:.3f}")
        else:
            print(f"  {k:46s} n={r.get('n')} {r.get('mean', np.nan):+.4f} "
                  f"p={r.get('p_perm', np.nan):.3f}")
    return res


def run_all(out_date=None):
    d = epochs(out_date)
    epoch_tests(out_date)
    glm(out_date)
    from swr_hfb_discovery_lock_figures import draw
    draw(d)
    print(f"-> {d}")


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"all": run_all, "epochs": epochs, "epoch_tests": epoch_tests,
                   "glm": glm, "glm_tests": glm_tests})
    else:
        run_all()
