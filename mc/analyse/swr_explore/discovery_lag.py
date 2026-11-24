#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SK's loading model: rewards are stored by HOW MANY DISCOVERIES AGO they were found.

    python scripts/swr_content_explore.py discovery_lag

    find A -> ripple -> A is 'now'
    find B -> ripple -> A moves back one, B is 'now'
    find C -> ripple -> A moves back two, B back one, C is 'now'
    ... and once all four are known the index becomes the A-B-C-D cycle.

So during the FIRST TRAVERSAL the representable content is entirely BEHIND the
subject: the future is unknown by construction. Indexing by "how far ahead in
the plan" (what `goal_lag` and `warped_lag` do) is therefore the wrong index
here -- it mixes squares the subject has found with squares they cannot
possibly know. Measured: goal0 is known 7.3% of the time on the first
traversal, goal1 25.3%, goal2 49.3%, goal3 75.5%.

This module indexes by DISCOVERY RECENCY instead:

    back[m] = the reward discovered m discoveries ago   (m = 1 is the newest)

defined only once at least m rewards have been found, so every tested square is
one the subject demonstrably knows. That is the whole point: it is the only
indexing under which a "loading" claim is even coherent during search.

The model also predicts the index MOVES: one reward occupies back[1], then
back[2], then back[3] as new rewards are found. Because the identity of the
square is constant and only its index changes, that shift is a within-item,
within-configuration contrast -- every static template bias cancels.

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
from mc.analyse.swr_content import phase_labels

SEED = 42
N_PERM = 1000
ROI_SETS = {"mPFC": ["mPFC"], "mOFC": ["mOFC"],
            "HC_all": ["HC_anterior", "HC_mid"]}
BACKS = (1, 2, 3)
MIN_RIPPLES = 15
MIN_CELLS = 2
FLANK_GAP_S = 0.05
MAX_FLANKS = 2


def discovery_back(steps_s, occ):
    """(n_occ, 3) location discovered m discoveries ago, m = 1..3; -1 if undefined.

    A "discovery" is the FIRST correct uncover of a reward within its
    configuration. Before m discoveries have happened, back[m] does not exist
    and the interval is excluded rather than given a placeholder -- a
    placeholder would silently test an unknowable square.
    """
    st = steps_s.sort_values("t_s").reset_index(drop=True)
    k = occ.step_idx.to_numpy()
    cu = st.correct_uncover.to_numpy(float)
    loc = st["loc"].to_numpy(float)
    gn = st.grid_num.to_numpy(float)
    t = st.t_s.to_numpy(float)
    out = np.full((len(occ), len(BACKS)), -1, int)
    t_occ = occ.start_s.to_numpy(float)
    gn_occ = gn[k]
    for g in np.unique(gn[np.isfinite(gn)]):
        m_g = (gn == g) & (cu == 1)
        if not m_g.any():
            continue
        # first correct uncover of each reward, in discovery order
        seen, seq = set(), []
        for i in np.flatnonzero(m_g):
            if loc[i] in seen:
                continue
            seen.add(loc[i]); seq.append((t[i], int(loc[i])))
        seq.sort()
        rows = np.flatnonzero(gn_occ == g)
        for r_ in rows:
            n_found = int(np.searchsorted([s_[0] for s_ in seq],
                                          t_occ[r_], side="right"))
            for mi, m in enumerate(BACKS):
                if n_found >= m:
                    out[r_, mi] = seq[n_found - m][1]
    return out


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
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < MIN_RIPPLES:
            continue
        st_s = steps[steps.session == s]
        subj = rip[rip.session == s].subject_key.iloc[0]
        BK = discovery_back(st_s, occ)
        ph_i, first_i, _, _ = phase_labels(st_s, occ)
        o = occ.sort_values("start_s")
        a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
        j = np.searchsorted(a, t_rip, side="right") - 1
        jj = np.clip(j, 0, len(a) - 1)
        k = o.index.to_numpy()[jj]
        inside = (j >= 0) & (t_rip <= b[jj])
        if inside.sum() < MIN_RIPPLES:
            continue
        t_rip, d_rip, k = t_rip[inside], d_rip[inside], k[inside]
        half = d_rip / 2.0
        grid = occ.cv_group.to_numpy()[k]
        fst = first_i[k]
        grids = np.unique(occ.cv_group.to_numpy())
        t_fl, owner = swc.matched_flanks(t_rip, half, occ, other_t=t_rip,
                                         gap_s=FLANK_GAP_S, max_per=MAX_FLANKS)
        if not len(t_fl):
            continue
        w_fl = half[owner]

        for roiname, members in ROI_SETS.items():
            cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                     if c < len(spk[s]["spikes"])]
            if len(cells) < MIN_CELLS:
                continue
            C_rip, C_fl = swc.zscore_cells([
                swc.window_counts(spk[s], cells, t_rip - half, t_rip + half),
                swc.window_counts(spk[s], cells, t_fl - w_fl, t_fl + w_fl)])
            for mi, m in enumerate(BACKS):
                cond = BK[:, mi] - 1
                if (cond >= 0).sum() < 50:
                    continue
                loo = {gr: np.array([
                    swc.zscore_map(swc.cond_map(spk[s]["spikes"][c], occ, cond,
                                                9, exclude_grid=gr))
                    for c in cells]) for gr in grids}
                tgt = cond[k]
                E_r = swc.score_windows(C_rip, t_rip, grid, loo)
                E_f = swc.score_windows(C_fl, t_fl, grid[owner], loo)
                ok_r = np.isfinite(E_r).all(axis=1) & (tgt >= 0)
                ok_f = np.isfinite(E_f).all(axis=1) & (tgt[owner] >= 0)
                S = _stat(E_r, np.clip(tgt, 0, 8), P)
                S_f = _stat(E_f, np.clip(tgt[owner], 0, 8), P)
                acc = np.zeros_like(S); cnt = np.zeros(len(S))
                np.add.at(acc, owner[ok_f], S_f[ok_f])
                np.add.at(cnt, owner[ok_f], 1.0)
                has = cnt > 0
                D = np.zeros_like(S)
                D[has] = S[has] - acc[has] / cnt[has, None]
                rec = dict(session=s, subject=subj, roi=roiname, back=m)
                for nm_, msk in (("first", fst), ("later", ~fst)):
                    sel = np.flatnonzero(ok_r & msk)
                    rec[f"n_{nm_}"] = len(sel)
                    rec[f"z_{nm_}"] = (float(_z(S[sel].mean(axis=0)))
                                       if len(sel) >= MIN_RIPPLES else np.nan)
                    selc = np.flatnonzero(ok_r & msk & has)
                    rec[f"c1_{nm_}"] = (float(_z(D[selc].mean(axis=0)))
                                        if len(selc) >= MIN_RIPPLES else np.nan)
                rows.append(rec)
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"discovery_lag_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "per_session.csv"), index=False)

    print("\n=== REWARDS INDEXED BY HOW MANY DISCOVERIES AGO THEY WERE FOUND ===")
    print("    back 1 = most recently found; only ALREADY-KNOWN squares tested\n")
    out, ov = [], {}
    print(f"  {'ROI':10s} {'back':>4s} {'test':22s} {'n':>3s} {'z':>7s} "
          f"{'t':>6s} {'P':>9s}")
    for roiname in ROI_SETS:
        for m in BACKS:
            g = R[(R.roi == roiname) & (R.back == m)]
            for col, nm in (("z_first", "first traversal"),
                            ("c1_first", "ripple-flank, first trav"),
                            ("z_later", "later traversals"),
                            ("c1_later", "ripple-flank, later")):
                v = g[col].dropna() if col in g else pd.Series(dtype=float)
                if len(v) < 5:
                    continue
                t, p = stats.ttest_1samp(v, 0)
                star = " *" if p < .05 else ""
                print(f"  {roiname:10s} {m:4d} {nm:22s} {len(v):3d} "
                      f"{v.mean():+7.3f} {t:+6.2f} {p:9.4g}{star}")
                rec = dict(roi=roiname, back=int(m), test=nm,
                           n_sessions=int(len(v)), z=float(v.mean()),
                           t=float(t), df=int(len(v) - 1), p=float(p))
                out.append(rec); ov[f"{roiname}|back{m}|{nm}"] = rec
        print()
    pd.DataFrame(out).to_csv(os.path.join(d, "summary.csv"), index=False)
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"), indent=2)
    json.dump(dict(seed=SEED, n_perm=N_PERM, roi_sets=ROI_SETS,
                   backs=list(BACKS), min_ripples=MIN_RIPPLES,
                   definition="back[m] = reward discovered m discoveries ago",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"-> {d}")


if __name__ == "__main__":
    main()
