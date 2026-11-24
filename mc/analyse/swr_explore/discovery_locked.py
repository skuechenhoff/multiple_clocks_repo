#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the newly-discovered-reward signal in mPFC CHANGE around a ripple?

    python scripts/swr_content_explore.py discovery_locked

`consolidation` showed mPFC holds the square that was just discovered (+0.77,
P = 0.0006), and that it does so specifically for NEW rewards -- not for the
same square uncovered again later, and not for errors. It also showed the
signal is WEAKER on trials where a ripple followed. This asks the time-resolved
version: take the ripples that occur while that signal is alive, and track the
evidence for the just-discovered square across the peri-ripple second.

    target  = the square whose reward was just discovered (fixed per event)
    events  = ripples between the discovery and the NEXT uncover, so the
              subject is still in the same situation and the target is still
              'the thing just learned'
    offsets = -500..+500 ms, the same 41-offset frame used elsewhere

A ripple that DELIVERS the location predicts a rise after 0. A ripple that
INTERRUPTS it predicts a dip. The `consolidation` split predicts a dip, so this
is a real test rather than a confirmation.

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

SEED = 42
N_PERM = 500
ROI_SETS = {"mPFC": ["mPFC"], "mOFC": ["mOFC"],
            "HC_all": ["HC_anterior", "HC_mid"]}
OFFSETS = np.round(np.arange(-0.50, 0.5001, 0.025), 4)
WIN_S = 0.25              # window width at each offset
MIN_EVENTS = 10
MIN_CELLS = 2
MAX_AFTER_S = 5.0         # ignore ripples more than this long after a discovery


def _stat(E, li, P):
    E = np.nan_to_num(E)
    H = np.take_along_axis(E, P[:, li].T, axis=1)
    return H - (E.sum(axis=1)[:, None] - H) / 8.0


def _z(M):
    sd = M[..., 1:].std(axis=-1)
    return np.where(sd > 0, (M[..., 0] - M[..., 1:].mean(axis=-1)) / sd, np.nan)


def main():
    rng = np.random.default_rng(SEED)
    P = np.vstack([np.arange(9)] + [rng.permutation(9) for _ in range(N_PERM)])
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
        occ, r, _, _ = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r):
            continue
        st = steps[steps.session == s].sort_values("t_s").reset_index(drop=True)
        subj = rip[rip.session == s].subject_key.iloc[0]
        t_all = st.t_s.to_numpy(float)
        unc = np.flatnonzero(st.is_uncover.to_numpy(float) == 1)
        if len(unc) < 10:
            continue
        t_u = t_all[unc]
        loc_u = st["loc"].to_numpy(float)[unc].astype(int)
        cu = st.correct_uncover.to_numpy(float)[unc]
        gn = st.grid_num.to_numpy(float)[unc]
        gid = st.grid_id.to_numpy(float)[unc]
        seen, is_first = set(), np.zeros(len(t_u), bool)
        for i in np.flatnonzero(cu == 1):
            key = (gn[i], loc_u[i])
            if key not in seen:
                seen.add(key); is_first[i] = True
        t_rip = np.sort(rip[rip.session == s].t_peak_s.to_numpy())

        # which reward each uncover was (state 1..4 -> A..D), so the time
        # course can be split by how much is already known at that point
        state_u = st.state.to_numpy(float)[unc]
        # ripples that fall between a FIRST DISCOVERY and the next uncover
        ev_t, ev_loc, ev_grid, ev_rew = [], [], [], []
        for i in np.flatnonzero(is_first):
            hi = t_u[i + 1] if i + 1 < len(t_u) else t_u[i] + MAX_AFTER_S
            hi = min(hi, t_u[i] + MAX_AFTER_S)
            m = (t_rip > t_u[i]) & (t_rip < hi)
            for tt in t_rip[m]:
                ev_t.append(tt); ev_loc.append(loc_u[i]); ev_grid.append(gid[i])
                ev_rew.append(state_u[i])
        if len(ev_t) < MIN_EVENTS:
            continue
        ev_t = np.asarray(ev_t, float)
        ev_loc = np.asarray(ev_loc, int)
        ev_grid = np.asarray(ev_grid, float)
        ev_rew = np.asarray(ev_rew, float)
        grids = np.unique(occ.cv_group.to_numpy())

        for roiname, members in ROI_SETS.items():
            cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                     if c < len(spk[s]["spikes"])]
            if len(cells) < MIN_CELLS:
                continue
            _, loo = swc.build_templates(spk[s], cells, occ, grids)
            C = swc.zscore_cells([swc.window_counts(
                spk[s], cells, ev_t + off - WIN_S / 2, ev_t + off + WIN_S / 2)
                for off in OFFSETS])
            li = np.clip(ev_loc - 1, 0, 8)
            for oi, off in enumerate(OFFSETS):
                E = swc.score_windows(C[oi], ev_t, ev_grid, loo)
                ok0 = np.isfinite(E).all(axis=1)
                for rw in (0, 1, 2, 3, 4):     # 0 = all rewards pooled
                    ok = ok0 if rw == 0 else (ok0 & (ev_rew == rw))
                    if ok.sum() < (MIN_EVENTS if rw == 0 else 5):
                        continue
                    S = _stat(E[ok], li[ok], P)
                    rows.append(dict(session=s, subject=subj, roi=roiname,
                                     offset_s=off, reward=int(rw),
                                     n_events=int(ok.sum()),
                                     z=float(_z(S.mean(axis=0)))))
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"discovery_locked_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "timecourse.csv"), index=False)

    import scripts.swr_content_main as mr
    print("\n=== THE JUST-DISCOVERED SQUARE, AROUND A RIPPLE ===")
    print(f"    ripples between a first discovery and the next uncover; "
          f"{WIN_S * 1000:.0f} ms windows\n")
    out, ov = [], {}
    for roiname in ROI_SETS:
        W = (R[(R.roi == roiname) & (R.reward == 0)]
             .pivot_table(index="session", columns="offset_s", values="z")
             .dropna())
        if len(W) < 5:
            print(f"  {roiname}: too few sessions ({len(W)})")
            continue
        off = W.columns.to_numpy(); M = W.to_numpy()
        t_obs, mass, p_cl, thr = mr._cluster_test(
            M, np.random.default_rng(SEED), n_clust=5000)
        i0 = int(np.argmin(np.abs(off)))
        t0, p0 = stats.ttest_1samp(M[:, i0], 0)
        pre = off < -0.05; post = off > 0.05
        dpp = M[:, post].mean(1) - M[:, pre].mean(1)
        tpp, ppp = stats.ttest_1samp(dpp, 0)
        print(f"  {roiname:8s} n = {len(W):3d} events/session median "
              f"{R[(R.roi == roiname) & (R.reward == 0)].n_events.median():.0f}")
        print(f"      at the ripple   z = {M[:, i0].mean():+.3f}  P = {p0:.4g}")
        print(f"      post minus pre  d = {dpp.mean():+.3f}  t = {tpp:+.2f}  "
              f"P = {ppp:.4g}")
        print(f"      cluster         p = {p_cl:.4g}")
        rec = dict(roi=roiname, n_sessions=int(len(W)),
                   z_at_ripple=float(M[:, i0].mean()), p_at_ripple=float(p0),
                   post_minus_pre=float(dpp.mean()), t_pp=float(tpp),
                   p_pp=float(ppp), p_cluster=float(p_cl))
        out.append(rec); ov[roiname] = rec
        print()
    pd.DataFrame(out).to_csv(os.path.join(d, "summary.csv"), index=False)
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"), indent=2)
    json.dump(dict(seed=SEED, n_perm=N_PERM, win_s=WIN_S,
                   offsets_s=OFFSETS.tolist(), roi_sets=ROI_SETS,
                   max_after_s=MAX_AFTER_S, min_events=MIN_EVENTS,
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"-> {d}")


if __name__ == "__main__":
    main()
