#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D1, D2 (pre-registered CHANGELOG 2026-11-15): who leads around a ripple, and
does that depend on context? Does the cortical response to an uncover predict
the ripple that follows?

SK's model: at uncovers mOFC (position) / lOFC (reward) signal the
hippocampus, which then ripples to store the current position (cortex -> HC);
later ripples run HC -> cortex to load the plan.

Ripple contexts (x phase: learning = grid onset -> start of the first fully
correct repeat; execution = the rest):
    UNCOVER  in (press, press + min(pause, 2 s)] after an uncover press
    AWAY     no uncover press in [t - 2 s, t + 0.5 s]
    -> U_L, U_E, A_L, A_E

D1  ripple-locked HFB per ROI (mOFC, lOFC, mPFC; 2026-09-28 partition),
    direction index = post (0, +250] - pre [-250, 0) ms.
      own      real - mean of 20 shifted nulls, minus the |t| >= 750 ms mean
      matched  ripple - ripple-free moments of the same context and phase in
               the same stratum of time since last / to next press
    PRIMARY: direction(U_L) - direction(A_E) < 0, own, mOFC and lOFC.
D2  per uncover: single-trial ROI HFB 0-500 ms minus -1.5..-1.0 s; outcome =
    a ripple in (+0.6 s, press + min(pause, 2.5 s)]; events with a ripple in
    [-1.5, +0.6] s or a pause < 1 s dropped. Per session logistic regression
    of the outcome on z(HFB) + log(window) + correct + first traversal (+
    phase); the stillness reading adds log(pause) + log(time since the
    previous press). PRIMARY: beta > 0 in learning, mOFC and lOFC.

Usage:
    python scripts/exploratory/swr_hfb_direction_context.py all

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import pickle
import datetime
import warnings

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "exploratory"))
import mc.analyse.helpers_human_cells as hh
import mc.analyse.swr_sakon as sk
from swr_plan_vs_store import _paths, _hfb_shifts, value_stats, _save_json, SEED
from swr_hfb_ripple_coupling import (load_inputs, session_data,
                                     session_ripples, roi_sums)
from swr_hfb_direction import PARTITION, REMOTE_MS, SLIDE_MS
from swr_hfb_gradient import phase_windows
from swr_hfb_discovery_lock import (strata, control_times, roi_epochs,
                                    _windows_mask, MIN_CTRL)

try:
    import fire
except ImportError:
    fire = None

NAME = "hfb_direction_context"
ROIS = ("mOFC", "lOFC", "mPFC")
SETS = ("U_L", "U_E", "A_L", "A_E")
SET_L = {"U_L": "uncover, learning", "U_E": "uncover, execution",
         "A_L": "away, learning", "A_E": "away, execution"}
PAUSE_MAX_S, AWAY_BEFORE_S, AWAY_AFTER_S = 2.0, 2.0, 0.5
MIN_EV = 15
WIN_MS = 250.0
EPOCH_S = (-1.0, 1.0)
EV_EPOCH_S = (-1.5, 1.5)
RESP_S, BASE_S = (0.0, 0.5), (-1.5, -1.0)
EXCL_S = (-1.5, 0.6)
OUT_START, OUT_MAX_S, MIN_PAUSE_S = 0.6, 2.5, 1.0
MIN_Y = 10
N_CLUSTER = 10000


def _out(out_date=None):
    group, _ = _paths()
    d = os.path.join(group, f"{NAME}_{out_date or datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    return d


def session_context(s, P, beh):
    W = phase_windows(beh)
    learn = [(a, b) for ph, a, b in W.itertuples(index=False)
             if ph in ("first_traversal", "rest_encoding")]
    exe = [(a, b) for ph, a, b in W.itertuples(index=False)
           if ph == "execution"]
    U = P[P.kind == "uncover"]
    tu = np.sort(U.t_s.to_numpy(float))
    Us = U.dropna(subset=["still_next_s"])
    pause = [(t, t + min(x, PAUSE_MAX_S)) for t, x in
             zip(Us.t_s.to_numpy(float), Us.still_next_s.to_numpy(float))]

    def away(t):
        n = (np.searchsorted(tu, t + AWAY_AFTER_S, side="right")
             - np.searchsorted(tu, t - AWAY_BEFORE_S, side="left"))
        return n == 0
    return dict(learn=learn, exe=exe, pause=pause, away=away, tu=tu)


def ripple_sets(t, ctx):
    in_l, in_e = _windows_mask(t, ctx["learn"]), _windows_mask(t, ctx["exe"])
    in_u, aw = _windows_mask(t, ctx["pause"]), ctx["away"](t)
    return {"U_L": in_u & in_l, "U_E": in_u & in_e,
            "A_L": aw & in_l, "A_E": aw & in_e}


def control_sets(t_rip, ctx):
    """Ripple-free moments, labelled with the same contexts."""
    tc_u = control_times(ctx["pause"], t_rip)
    tc_a = control_times(ctx["learn"] + ctx["exe"], t_rip)
    out = {}
    for k, tc in (("U", tc_u), ("A", tc_a)):
        m = ripple_sets(tc, ctx)
        for ph in ("L", "E"):
            out[f"{k}_{ph}"] = tc[m[f"{k}_{ph}"]]
    return out


# ── D1 ───────────────────────────────────────────────────────────────────
def d1(out_date=None):
    d = _out(out_date)
    group, b_dir = _paths()
    rip, hiv, hp = load_inputs(b_dir, os.path.join(group, PARTITION))
    P_all = pd.read_csv(os.path.join(group, "press_categories.csv"))
    shifts = _hfb_shifts()
    curves, counts = {}, []
    for sess in sorted(hp[hp.roi.isin(ROIS)].session.unique()):
        dd = hh.load_norm_data(hh_dir(), [f"{int(sess):02d}"], res_data=False)
        if not dd:
            continue
        beh = dd[f"sub-{int(sess):02d}"]["beh"]
        P = P_all[P_all.session == sess].sort_values("t_s")
        if not len(P):
            continue
        H, idx, masks, g, fs, n = session_data(sess, b_dir, hp, hiv, ROIS)
        pids = {r: g[g.roi == r].pair_id.tolist() for r in ROIS}
        ctx = session_context(sess, P, beh)
        tp = np.sort(P.t_s.to_numpy(float))
        t_rip = session_ripples(rip, sess)
        rs = ripple_sets(t_rip, ctx)
        cs = control_sets(t_rip, ctx)
        off = np.arange(int(round(EPOCH_S[0] * fs)), int(round(EPOCH_S[1] * fs)))
        for key in SETS:
            t = t_rip[rs[key]]
            for roi in ROIS:
                if not pids[roi] or len(t) < MIN_EV:
                    continue
                M = []
                for sh in shifts:
                    X, ok = roi_epochs(H, idx, masks, pids[roi], t, off, n, fs, sh)
                    if ok.sum() < MIN_EV:
                        M = None
                        break
                    M.append(X.mean(0))
                if M is not None:
                    M = np.stack(M)
                    curves[(key, roi, "own", int(sess))] = M[0] - M[1:].mean(0)
                Xr, okr = roi_epochs(H, idx, masks, pids[roi], t, off, n, fs)
                tc = cs[key]
                if okr.sum() < MIN_EV or not len(tc):
                    continue
                Xc, okc = roi_epochs(H, idx, masks, pids[roi], tc, off, n, fs)
                sr, sc = strata(t[okr], tp), strata(tc[okc], tp)
                acc, w = [], []
                for q in np.unique(sr):
                    cq = sc == q
                    if cq.sum() < MIN_CTRL:
                        continue
                    rq = sr == q
                    acc.append(Xr[rq].mean(0) - Xc[cq].mean(0))
                    w.append(rq.sum())
                if sum(w) >= MIN_EV:
                    w = np.asarray(w, float)
                    curves[(key, roi, "matched", int(sess))] = \
                        (np.stack(acc) * w[:, None]).sum(0) / w.sum()
                counts.append(dict(session=int(sess), set=key, roi=roi,
                                   n_ripples=int(okr.sum()),
                                   n_matched=int(sum(w)), n_ctrl=int(okc.sum())))
        print(f"  D1 s{int(sess):02d}", flush=True)
    t_ms = off / fs * 1000.0
    with open(os.path.join(d, "d1_curves.pkl"), "wb") as f:
        pickle.dump(dict(curves=curves, t_ms=t_ms), f)
    pd.DataFrame(counts).to_csv(os.path.join(d, "d1_counts.csv"), index=False)
    return d1_tests(out_date)


def hh_dir():
    from per_lag_encoding_by_phase import DATA_DIR
    return DATA_DIR


def prepared(curves, t_ms, key, roi, reading):
    from scipy.ndimage import uniform_filter1d
    base = np.abs(t_ms) >= REMOTE_MS
    k = int(round(SLIDE_MS / (t_ms[1] - t_ms[0])))
    out = {}
    for (k_, r_, rd_, s), c in curves.items():
        if (k_, r_, rd_) == (key, roi, reading):
            x = c - c[base].mean() if reading == "own" else c
            out[s] = uniform_filter1d(x, k, mode="nearest")
    return out


def direction(c, t_ms):
    pre = (t_ms >= -WIN_MS) & (t_ms < 0)
    post = (t_ms > 0) & (t_ms <= WIN_MS)
    return float(c[post].mean() - c[pre].mean())


def d1_tests(out_date=None):
    d = _out(out_date)
    D = pickle.load(open(os.path.join(d, "d1_curves.pkl"), "rb"))
    curves, t_ms = D["curves"], D["t_ms"]
    test = np.abs(t_ms) < REMOTE_MS
    res, rows = {}, []
    for roi in ROIS:
        for rd in ("own", "matched"):
            dirs = {}
            for key in SETS:
                C = prepared(curves, t_ms, key, roi, rd)
                if len(C) < 5:
                    continue
                S = sorted(C)
                X = np.stack([C[s] for s in S])
                dirs[key] = pd.Series({s: direction(C[s], t_ms) for s in S})
                for s in S:
                    rows.append(dict(roi=roi, reading=rd, set=key, session=s,
                                     direction=dirs[key][s],
                                     window=float(C[s][np.abs(t_ms) <= WIN_MS]
                                                  .mean())))
                t_obs, cl, pv, _ = sk.cluster_perm_time(X[:, test],
                                                        n_perm=N_CLUSTER,
                                                        seed=SEED)
                tt = t_ms[test]
                res[f"{roi} | {rd} | {key}"] = dict(
                    direction=value_stats(dirs[key].to_numpy()),
                    window=value_stats(np.array([C[s][np.abs(t_ms) <= WIN_MS]
                                                 .mean() for s in S])),
                    clusters=[dict(start_ms=float(tt[a]), stop_ms=float(tt[b - 1]),
                                   sign="increase" if np.nanmean(t_obs[a:b]) > 0
                                   else "decrease", p=float(p))
                              for (a, b), p in zip(cl, pv)],
                    mean=X.mean(0).tolist(),
                    sem=(X.std(0, ddof=1) / np.sqrt(len(X))).tolist())
            for a, b in (("U_L", "A_E"), ("U_L", "A_L"), ("U_L", "U_E"),
                         ("A_L", "A_E")):
                if a in dirs and b in dirs:
                    both = sorted(set(dirs[a].index) & set(dirs[b].index))
                    lab = " PRIMARY" if (a, b) == ("U_L", "A_E") and rd == "own" \
                        and roi in ("mOFC", "lOFC") else ""
                    res[f"{roi} | {rd} | {a} - {b}{lab}"] = value_stats(
                        (dirs[a][both] - dirs[b][both]).to_numpy())
            if all(k in dirs for k in SETS):
                both = sorted(set.intersection(*[set(dirs[k].index)
                                                 for k in SETS]))
                v = ((dirs["U_L"] - dirs["A_L"]) - (dirs["U_E"] - dirs["A_E"]))
                res[f"{roi} | {rd} | interaction (U-A)L - (U-A)E"] = \
                    value_stats(v[both].to_numpy())
    pd.DataFrame(rows).to_csv(os.path.join(d, "d1_session_values.csv"),
                              index=False)
    _save_json(d, "d1_result.json", dict(t_ms=t_ms.tolist(), tests=res))
    for k, r in res.items():
        if "direction" in r:
            v = r["direction"]
            print(f"  {k:28s} n={v.get('n')} dir {v.get('mean', np.nan):+.4f} "
                  f"p={v.get('p_perm', np.nan):.3f} | win "
                  f"{r['window'].get('mean', np.nan):+.4f} "
                  f"p={r['window'].get('p_perm', np.nan):.3f}")
        else:
            print(f"  {k:45s} n={r.get('n')} {r.get('mean', np.nan):+.4f} "
                  f"p={r.get('p_perm', np.nan):.3f}")
    return res


# ── D2 ───────────────────────────────────────────────────────────────────
def d2(out_date=None):
    import statsmodels.api as sm
    d = _out(out_date)
    group, b_dir = _paths()
    rip, hiv, hp = load_inputs(b_dir, os.path.join(group, PARTITION))
    P_all = pd.read_csv(os.path.join(group, "press_categories.csv"))
    rows, betas = [], []
    for sess in sorted(hp[hp.roi.isin(ROIS)].session.unique()):
        dd = hh.load_norm_data(hh_dir(), [f"{int(sess):02d}"], res_data=False)
        if not dd:
            continue
        beh = dd[f"sub-{int(sess):02d}"]["beh"]
        P = P_all[P_all.session == sess].sort_values("t_s")
        if not len(P):
            continue
        H, idx, masks, g, fs, n = session_data(sess, b_dir, hp, hiv, ROIS)
        pids = {r: g[g.roi == r].pair_id.tolist() for r in ROIS}
        ctx = session_context(sess, P, beh)
        t_rip = np.sort(session_ripples(rip, sess))
        tp = np.sort(P.t_s.to_numpy(float))
        U = P[(P.kind == "uncover") & (P.still_next_s >= MIN_PAUSE_S)]
        t = U.t_s.to_numpy(float)
        pre_rip = (np.searchsorted(t_rip, t + EXCL_S[1], side="right")
                   - np.searchsorted(t_rip, t + EXCL_S[0], side="left")) > 0
        U, t = U[~pre_rip], t[~pre_rip]
        end = t + np.minimum(U.still_next_s.to_numpy(float), OUT_MAX_S)
        y = ((np.searchsorted(t_rip, end, side="right")
              - np.searchsorted(t_rip, t + OUT_START, side="right")) > 0)
        j = np.searchsorted(tp, t, side="left") - 1
        since = np.where(j >= 0, t - tp[np.clip(j, 0, None)], np.nan)
        E = pd.DataFrame(dict(
            t=t, y=y.astype(int), lwin=np.log(end - t - OUT_START),
            correct=(U.valence == "correct").to_numpy().astype(int),
            ft=(U.stage == "first uncovers").to_numpy().astype(int),
            learning=_windows_mask(t, ctx["learn"]).astype(int),
            lpause=np.log(U.still_next_s.to_numpy(float)),
            lsince=np.log(np.clip(since, 0.05, None))))
        off = np.arange(int(round(EV_EPOCH_S[0] * fs)),
                        int(round(EV_EPOCH_S[1] * fs)))
        t_ms = off / fs * 1000
        resp = (t_ms >= RESP_S[0] * 1000) & (t_ms < RESP_S[1] * 1000)
        base = (t_ms >= BASE_S[0] * 1000) & (t_ms < BASE_S[1] * 1000)
        for roi in ROIS:
            if not pids[roi]:
                continue
            X, ok = roi_epochs(H, idx, masks, pids[roi], t, off, n, fs)
            if ok.sum() < 2 * MIN_Y:
                continue
            Ei = E[ok].copy()
            Ei["hfb"] = X[:, resp].mean(1) - X[:, base].mean(1)
            for _, r_ in Ei.iterrows():
                rows.append(dict(session=int(sess), roi=roi, **r_.to_dict()))
            for subset, m in (("all", np.ones(len(Ei), bool)),
                              ("learning", Ei.learning == 1),
                              ("execution", Ei.learning == 0)):
                Es = Ei[m]
                if Es.y.sum() < MIN_Y or (1 - Es.y).sum() < MIN_Y:
                    continue
                for reading, cov in (("own", ["lwin", "correct", "ft"]
                                      + (["learning"] if subset == "all" else [])),
                                     ("stillness", ["lwin", "correct", "ft",
                                                    "lpause", "lsince"]
                                      + (["learning"] if subset == "all" else []))):
                    Z = Es[["hfb"] + cov].astype(float)
                    Z = Z.loc[:, Z.std() > 0]
                    if "hfb" not in Z:
                        continue
                    Z = (Z - Z.mean()) / Z.std()
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            f = sm.Logit(Es.y.to_numpy(float),
                                         sm.add_constant(Z.to_numpy())).fit(
                                disp=0, maxiter=200)
                        b = float(f.params[1 + list(Z.columns).index("hfb")])
                    except Exception:
                        b = np.nan
                    betas.append(dict(session=int(sess), roi=roi, subset=subset,
                                      reading=reading, beta=b, n=len(Es),
                                      n_ripple=int(Es.y.sum())))
        print(f"  D2 s{int(sess):02d}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(d, "d2_events.csv"), index=False)
    B = pd.DataFrame(betas)
    B.to_csv(os.path.join(d, "d2_betas.csv"), index=False)
    res = {}
    for (roi, sub, rd), g in B.groupby(["roi", "subset", "reading"]):
        lab = " PRIMARY" if sub == "learning" and rd == "own" and \
            roi in ("mOFC", "lOFC") else ""
        res[f"{roi} | {sub} | {rd}{lab}"] = value_stats(g.beta.to_numpy(float))
    for (roi, rd), g in B.groupby(["roi", "reading"]):
        w = g.pivot_table(index="session", columns="subset", values="beta")
        if {"learning", "execution"} <= set(w.columns):
            res[f"{roi} | learning - execution | {rd}"] = value_stats(
                (w.learning - w.execution).dropna().to_numpy())
    _save_json(d, "d2_result.json", res)
    for k, r in res.items():
        print(f"  {k:46s} n={r.get('n')} {r.get('mean', np.nan):+.3f} "
              f"p={r.get('p_perm', np.nan):.3f}")
    return res


def settings(out_date=None):
    d = _out(out_date)
    _save_json(d, "settings.json", dict(
        analysis=NAME, created=str(datetime.date.today()),
        script=os.path.relpath(__file__, REPO), seed=SEED, rois=ROIS,
        partition=PARTITION, n_shifts=len(_hfb_shifts()) - 1,
        contexts=dict(uncover=f"(press, press + min(pause, {PAUSE_MAX_S}))",
                      away=f"no uncover press in [t - {AWAY_BEFORE_S}, "
                           f"t + {AWAY_AFTER_S}]"),
        phases="learning = grid onset -> start of the first fully correct "
               "repeat; execution = the rest",
        d1=dict(epoch_s=EPOCH_S, direction=f"post (0, {WIN_MS}] - pre "
                                          f"[-{WIN_MS}, 0) ms",
                own=f"real - shifted, minus |t| >= {REMOTE_MS} ms",
                matched="ripple - stratum-matched ripple-free moments of the "
                        "same context and phase", min_events=MIN_EV,
                timing="ripple PEAK (no onset in the bundle)"),
        d2=dict(predictor=f"HFB {RESP_S} s minus {BASE_S} s, single trial",
                excluded=f"ripple in {EXCL_S} s, pause < {MIN_PAUSE_S} s",
                outcome=f"ripple in (+{OUT_START}, press + min(pause, "
                        f"{OUT_MAX_S})] s",
                own="z(HFB) + log(window) + correct + first traversal "
                    "(+ phase)",
                stillness="+ log(pause) + log(time since previous press)",
                min_each_class=MIN_Y),
        preregistration="CHANGELOG.md 2026-11-15 D1, D2"))
    return d


def run_all(out_date=None):
    settings(out_date)
    d1(out_date)
    d2(out_date)
    from swr_hfb_direction_context_figures import draw
    draw(_out(out_date))


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"all": run_all, "d1": d1, "d1_tests": d1_tests, "d2": d2,
                   "settings": settings})
    else:
        run_all()
