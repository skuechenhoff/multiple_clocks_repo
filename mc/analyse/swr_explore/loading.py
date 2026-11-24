#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST L1: does the representation become MORE PLAN-LIKE AFTER a ripple?

    python scripts/swr_content_explore.py loading

Content and loading are different questions. "Is the future represented more
during ripples than outside?" is a LEVEL. Loading is a CHANGE: the plan should
be better represented AFTER a ripple than BEFORE it. No during/outside contrast
can show that, so this one is paired in time.

    pre  = [t - half - gap - win,  t - half - gap]
    post = [t + half + gap,        t + half + gap + win]
    statistic = z( mean over ripples of (S_post - S_pre) )

BOTH sides are required, so every ripple contributes a paired difference and
sampling differences between the sides cannot masquerade as a step.

MATCHING UNIT. Both windows must sit inside ONE SUBGOAL INTERVAL, because that
is the unit over which the PLAN is constant (median 1.85 s). The occupancy
interval used elsewhere holds the CURRENT LOCATION constant instead, and at a
median 0.45 s is too short to fit a pre/post pair. Getting this wrong would
compare plan evidence across a change of plan.

TWO CONTROLS, both mandatory:
  * `current` target -- the hippocampal current-location code is carried
    continuously, so it must show NO step. A step there means the contrast is
    picking up something generic rather than plan loading.
  * sham events -- random times inside subgoal intervals, away from real
    ripples, same count, same construction. A step at sham times means the
    contrast reflects within-subgoal drift, not ripples.

@author: Svenja Kuchenhoff
"""

import os
import json
import datetime

import numpy as np
import pandas as pd
from scipy import stats

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc
import mc.analyse.swr_warp as sww
from mc.analyse.swr_content import phase_labels
from mc.analyse.swr_explore.goal_lag import goal_lag_locations

SEED = 42
N_PERM = 1000
ROI_SETS = {"mPFC": ["mPFC"], "mOFC": ["mOFC"],
            "HC_all": ["HC_anterior", "HC_mid"]}
TARGETS = ("goal0", "goal1", "current")
PHASES = ("first_traversal", "encode_rest", "execute")
WIN_S = 0.25
GAP_S = 0.05
MIN_EVENTS = 20
MIN_CELLS = 2


def _stat(E, li, P):
    E = np.nan_to_num(E)
    H = np.take_along_axis(E, P[:, li].T, axis=1)
    return H - (E.sum(axis=1)[:, None] - H) / 8.0


def _z(M):
    sd = M[..., 1:].std(axis=-1)
    return np.where(sd > 0, (M[..., 0] - M[..., 1:].mean(axis=-1)) / sd, np.nan)


def _sham(t_rip, seg, rng, n, min_gap=0.5):
    """Random times inside subgoal intervals, away from any real ripple."""
    if not len(seg) or not n:
        return np.empty(0)
    w = seg[:, 1] - seg[:, 0]
    p = w / w.sum()
    out = []
    for _ in range(6):
        i = rng.choice(len(seg), size=n * 3, p=p)
        t = seg[i, 0] + rng.random(len(i)) * (seg[i, 1] - seg[i, 0])
        if len(t_rip):
            d = np.abs(t[:, None] - t_rip[None, :]).min(axis=1)
            t = t[d > min_gap]
        out.append(t)
        if sum(len(o) for o in out) >= n:
            break
    t = np.concatenate(out) if out else np.empty(0)
    return np.sort(t[:n])


def main():
    rng0 = np.random.default_rng(SEED)
    P = np.vstack([np.arange(9)] + [rng0.permutation(9) for _ in range(N_PERM)])
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s",
                               "subject_key"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)

    rows = []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < MIN_EVENTS:
            continue
        st_s = steps[steps.session == s]
        seg = sww.subgoal_intervals(st_s)
        if not len(seg):
            continue
        subj = rip[rip.session == s].subject_key.iloc[0]
        ph_i, first_i, _, _ = phase_labels(st_s, occ)
        GL = goal_lag_locations(st_s, occ)
        grids = np.unique(occ.cv_group.to_numpy())
        rng = np.random.default_rng([SEED, s])

        for kind in ("ripple", "sham"):
            if kind == "ripple":
                t0, half = t_rip, d_rip / 2.0
            else:
                t0 = _sham(t_rip, seg, rng, len(t_rip))
                if len(t0) < MIN_EVENTS:
                    continue
                half = np.full(len(t0), float(np.median(d_rip)) / 2.0)
            pl, ph_, ql, qh, ok = sww.prepost_windows(
                t0, half, st_s, other_t=t_rip, win_s=WIN_S, gap_s=GAP_S)
            if ok.sum() < MIN_EVENTS:
                continue
            # which occupancy interval each event sits in
            o = occ.sort_values("start_s")
            a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
            j = np.searchsorted(a, t0, side="right") - 1
            jj = np.clip(j, 0, len(a) - 1)
            ins = (j >= 0) & (t0 <= b[jj]) & ok
            if ins.sum() < MIN_EVENTS:
                continue
            k = o.index.to_numpy()[jj][ins]
            grid = occ.cv_group.to_numpy()[k]
            phs, fst = ph_i[k], first_i[k]
            masks = {"first_traversal": fst,
                     "encode_rest": (phs == "encode") & ~fst,
                     "execute": (phs == "execute")}
            W = dict(pre_lo=pl[ins], pre_hi=ph_[ins],
                     post_lo=ql[ins], post_hi=qh[ins])

            for roiname, members in ROI_SETS.items():
                cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                         if c < len(spk[s]["spikes"])]
                if len(cells) < MIN_CELLS:
                    continue
                C_pre, C_post = swc.zscore_cells([
                    swc.window_counts(spk[s], cells, W["pre_lo"], W["pre_hi"]),
                    swc.window_counts(spk[s], cells, W["post_lo"],
                                      W["post_hi"])])
                for tgt_name in TARGETS:
                    if tgt_name == "current":
                        cond = occ["loc"].to_numpy().astype(int) - 1
                    else:
                        cond = GL[:, int(tgt_name[-1])] - 1
                    if (cond >= 0).sum() < 50:
                        continue
                    loo = {gr: np.array([
                        swc.zscore_map(swc.cond_map(spk[s]["spikes"][c], occ,
                                                    cond, 9, exclude_grid=gr))
                        for c in cells]) for gr in grids}
                    tg = cond[k]
                    E_pre = swc.score_windows(C_pre, W["pre_lo"], grid, loo)
                    E_post = swc.score_windows(C_post, W["post_lo"], grid, loo)
                    good = (np.isfinite(E_pre).all(axis=1)
                            & np.isfinite(E_post).all(axis=1) & (tg >= 0))
                    li = np.clip(tg, 0, 8)
                    D = _stat(E_post, li, P) - _stat(E_pre, li, P)
                    rec = dict(session=s, subject=subj, roi=roiname,
                               kind=kind, target=tgt_name)
                    for p_ in PHASES + ("all",):
                        m = good if p_ == "all" else (good & masks[p_])
                        sel = np.flatnonzero(m)
                        rec[f"n_{p_}"] = len(sel)
                        rec[f"d_{p_}"] = (float(_z(D[sel].mean(axis=0)))
                                          if len(sel) >= MIN_EVENTS else np.nan)
                    rows.append(rec)
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr", f"loading_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "per_session.csv"), index=False)

    print(f"\n=== LOADING: post minus pre, {WIN_S * 1000:.0f} ms windows "
          "inside one subgoal interval ===")
    print("    positive = the representation is stronger AFTER the event\n")
    out, ov = [], {}
    print(f"  {'ROI':10s} {'target':8s} {'kind':7s} {'phase':16s} {'n':>3s} "
          f"{'d':>7s} {'t':>6s} {'P':>9s}")
    for roiname in ROI_SETS:
        for tgt_name in TARGETS:
            for kind in ("ripple", "sham"):
                for p_ in ("all",) + PHASES:
                    g = R[(R.roi == roiname) & (R.target == tgt_name)
                          & (R["kind"] == kind)]
                    v = g[f"d_{p_}"].dropna() if f"d_{p_}" in g else \
                        pd.Series(dtype=float)
                    if len(v) < 5:
                        continue
                    t, p = stats.ttest_1samp(v, 0)
                    star = " *" if p < .05 else ""
                    print(f"  {roiname:10s} {tgt_name:8s} {kind:7s} {p_:16s} "
                          f"{len(v):3d} {v.mean():+7.3f} {t:+6.2f} "
                          f"{p:9.4g}{star}")
                    rec = dict(roi=roiname, target=tgt_name, kind=kind,
                               phase=p_, n_sessions=int(len(v)),
                               d=float(v.mean()), t=float(t),
                               df=int(len(v) - 1), p=float(p))
                    out.append(rec)
                    ov[f"{roiname}|{tgt_name}|{kind}|{p_}"] = rec
        print()
    pd.DataFrame(out).to_csv(os.path.join(d, "summary.csv"), index=False)
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"), indent=2)
    json.dump(dict(seed=SEED, n_perm=N_PERM, roi_sets=ROI_SETS,
                   targets=list(TARGETS), phases=list(PHASES),
                   win_s=WIN_S, gap_s=GAP_S, min_events=MIN_EVENTS,
                   matching_unit="subgoal interval",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"-> {d}")


if __name__ == "__main__":
    main()
