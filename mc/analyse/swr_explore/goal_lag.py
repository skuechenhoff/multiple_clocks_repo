#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST (b): do ripples carry the upcoming GOAL SEQUENCE, not just where you are?

    python scripts/swr_content_explore.py goal_lag

The coarse, real-time version of the future-lag question. No time warping: the
plan is indexed by how many REWARDS ahead a location will be visited.

    goal_lag[g] = rew[(state - 1 + g) mod 4]

so g = 0 is the reward being sought right now, g = 1 the one after it, and so
on. g is NOT the physical current location -- that is the existing analysis in
`swr_phase_contrast.py`, and the two are different quantities.

WHY THIS INDEX IS CLEAN. Within a configuration the four reward locations are
fixed, but which lag each one occupies rotates as the subject cycles A->B->C->D.
Measured: a fixed reward location occupies 4.00 of the 4 possible goal-lags,
in 100% of 672 location x configuration cells. So "location L at lag 0" and
"location L at lag 2" share location, configuration, cells and templates, and
differ only in where L sits in the plan. Every static bias cancels.

Everything else is the reported ripple estimator unchanged: leave-one-
configuration-out templates, nine-label permutation null, matched flanks drawn
from inside the ripple's own occupancy interval, session as the unit of
inference, two-tailed t-tests.

⚠ Templates for different lags are correlated (the next goal is often adjacent
to the current one, and adjacent-square templates correlate). Only the
ripple-minus-flank contrast is confound-free; the raw per-lag levels are
descriptive.

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
            "HC_all": ["HC_anterior", "HC_mid"],
            "HC_anterior": ["HC_anterior"], "HC_mid": ["HC_mid"]}
LAGS = (0, 1, 2, 3)          # rewards ahead
# THREE-WAY split. first_traversal is a strict subset of `encode`, so the
# middle phase is encode-minus-first-traversal -- the three are then mutually
# exclusive, which they must be for a matched comparison.
PHASES = ("first_traversal", "encode_rest", "execute")
N_MATCH = 1000
MIN_RIPPLES = 20
MIN_CELLS = 2
FLANK_GAP_S = 0.05
MAX_FLANKS = 2


def goal_lag_locations(steps_s, occ):
    """(n_occ, 4) int: the location that is g rewards ahead, for g = 0..3.

    -1 where undefined (state or rewards missing).
    """
    st = steps_s.sort_values("t_s").reset_index(drop=True)
    k = occ.step_idx.to_numpy()
    state = st.state.to_numpy(float)[k]
    rew = st[["rew_A", "rew_B", "rew_C", "rew_D"]].to_numpy(float)[k]
    out = np.full((len(occ), len(LAGS)), -1, int)   # locations 1-9, -1 = undefined
    ok = np.isfinite(state) & np.isfinite(rew).all(axis=1)
    idx = np.arange(len(occ))
    for g in LAGS:
        col = ((state - 1 + g) % 4)
        good = ok & np.isfinite(col)
        out[good, g] = rew[idx[good], col[good].astype(int)].astype(int)
    return out


def _templates(spk_s, cells, occ, cond, grids):
    """{grid: (n_cells, 9)} leave-one-configuration-out templates for one lag."""
    loo = {}
    for gr in grids:
        loo[gr] = np.array([
            swc.zscore_map(swc.cond_map(spk_s["spikes"][c], occ, cond, 9,
                                        exclude_grid=gr))
            for c in cells])
    return loo


def _stat(E, li, P):
    """(n_windows, 1 + N_PERM): observed statistic then its permutation null."""
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

    rows, conf = [], []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < MIN_RIPPLES:
            continue
        subj = rip[rip.session == s].subject_key.iloc[0]
        st_s = steps[steps.session == s]
        GL = goal_lag_locations(st_s, occ)             # (n_occ, 4)
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
        cur_loc = occ["loc"].to_numpy()[k].astype(int)
        ph, fst = ph_i[k], first_i[k]
        grids = np.unique(occ.cv_group.to_numpy())
        masks = {"first_traversal": fst,
                 "encode_rest": (ph == "encode") & ~fst,
                 "execute": (ph == "execute")}

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
            for g in LAGS:
                # cond_map wants 0..8 (negative = skip); GL holds locations 1-9
                cond_all = GL[:, g] - 1
                if (cond_all >= 0).sum() < 50:
                    continue
                loo = _templates(spk[s], cells, occ, cond_all, grids)
                tgt = cond_all[k]        # 0-based target index at each ripple
                good = tgt >= 0
                E_r = swc.score_windows(C_rip, t_rip, grid, loo)
                E_f = swc.score_windows(C_fl, t_fl, grid[owner], loo)
                ok_r = np.isfinite(E_r).all(axis=1) & good
                ok_f = np.isfinite(E_f).all(axis=1) & good[owner]
                S = _stat(E_r, np.clip(tgt, 0, 8), P)
                S_f = _stat(E_f, np.clip(tgt[owner], 0, 8), P)
                acc = np.zeros_like(S); cnt = np.zeros(len(S))
                np.add.at(acc, owner[ok_f], S_f[ok_f])
                np.add.at(cnt, owner[ok_f], 1.0)
                has = cnt > 0
                D = np.zeros_like(S)
                D[has] = S[has] - acc[has] / cnt[has, None]
                rec = dict(session=s, subject=subj, roi=roiname, lag=g)
                for p_ in PHASES:
                    sel = np.flatnonzero(masks[p_] & ok_r)
                    rec[f"n_{p_}"] = len(sel)
                    rec[f"z_{p_}"] = (float(_z(S[sel].mean(axis=0)))
                                      if len(sel) >= MIN_RIPPLES else np.nan)
                    selc = np.flatnonzero(masks[p_] & ok_r & has)
                    rec[f"c1_{p_}"] = (float(_z(D[selc].mean(axis=0)))
                                       if len(selc) >= MIN_RIPPLES else np.nan)
                # ---- COUNT MATCHING. z carries a sqrt(n) term, and the three
                # phases differ in ripple count for purely structural reasons
                # (execute is longest, first_traversal shortest), so raw
                # levels are not comparable. One set of draws, all three
                # phases down to the common minimum, so the levels AND every
                # pairwise contrast come from the same subsamples.
                idx = {p_: np.flatnonzero(masks[p_] & ok_r) for p_ in PHASES}
                nmin = min(len(v) for v in idx.values())
                if nmin >= MIN_RIPPLES:
                    rs = np.random.default_rng([SEED, s, hash(roiname) % 997, g])
                    zd = {}
                    for p_ in PHASES:
                        A = np.zeros((N_MATCH, len(S)))
                        for di in range(N_MATCH):
                            A[di, rs.choice(idx[p_], nmin,
                                            replace=False)] = 1.0 / nmin
                        zd[p_] = _z(A @ S)
                    rec["n_matched"] = int(nmin)
                    for p_ in PHASES:
                        rec[f"z_{p_}_matched"] = float(np.nanmean(zd[p_]))
                    for pa, pb in (("first_traversal", "execute"),
                                   ("encode_rest", "execute"),
                                   ("first_traversal", "encode_rest")):
                        rec[f"d_{pa}_minus_{pb}"] = float(
                            np.nanmean(zd[pa] - zd[pb]))
                rows.append(rec)

                # ---- CONFUSION: where does the evidence actually GO?
                # E_r is raw evidence per location. Because templates are
                # z-scored to sum to zero, each row sums to ~0 by
                # construction -- the matrix shows how evidence is
                # DISTRIBUTED, not its total.
                for p_ in PHASES:
                    m = masks[p_] & ok_r
                    if m.sum() < MIN_RIPPLES:
                        continue
                    Em, tt = E_r[m], tgt[m]
                    # (i) 9 x 9 in absolute grid locations
                    for tl in range(9):
                        sel = tt == tl
                        if sel.sum() < 5:
                            continue
                        conf.append(dict(
                            session=s, roi=roiname, lag=g, phase=p_,
                            kind="location", row=tl + 1, n=int(sel.sum()),
                            **{f"E{l + 1}": float(Em[sel, l].mean())
                               for l in range(9)}))
                    # (ii) collapsed by the ROLE the scored square plays --
                    # far more interpretable than absolute grid identity
                    role_idx = {"current": cur_loc[m] - 1}
                    for gg in LAGS:
                        role_idx[f"goal{gg}"] = GL[k, gg][m] - 1
                    ar = np.arange(m.sum())
                    row = dict(session=s, roi=roiname, lag=g, phase=p_,
                               kind="role", row=0, n=int(m.sum()))
                    used = np.zeros((m.sum(), 9), bool)
                    for nm_, v in role_idx.items():
                        good_ = (v >= 0) & (v < 9)
                        if good_.sum() < 5:
                            continue
                        row[nm_] = float(Em[ar[good_], v[good_]].mean())
                        used[ar[good_], v[good_]] = True
                    oth = np.where(~used, Em, np.nan)
                    row["other"] = float(np.nanmean(oth))
                    conf.append(row)
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"goal_lag_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "per_session.csv"), index=False)
    pd.DataFrame(conf).to_csv(os.path.join(d, "confusion.csv"), index=False)

    print("\n=== GOAL-SEQUENCE CONTENT IN RIPPLES (coarse, real time) ===")
    print("    lag = rewards ahead; lag 0 = the reward being sought now\n")
    out, ov = [], {}
    print(f"  {'ROI':12s} {'lag':>3s} {'test':26s} {'n':>3s} {'z':>7s} "
          f"{'t':>7s} {'P':>9s} {'95% CI':>18s}")
    for roiname in ROI_SETS:
        for g in LAGS:
            gg = R[(R.roi == roiname) & (R.lag == g)]
            tests = [(f"z_{p_}_matched", f"{p_} (matched)") for p_ in PHASES]
            tests += [(f"z_{p_}", f"{p_} (raw)") for p_ in PHASES]
            tests += [(f"c1_{p_}", f"ripple-flank, {p_}") for p_ in PHASES]
            tests += [("d_first_traversal_minus_execute", "1st trav - execute"),
                      ("d_encode_rest_minus_execute", "enc rest - execute"),
                      ("d_first_traversal_minus_encode_rest",
                       "1st trav - enc rest")]
            for col, nm in tests:
                v = gg[col].dropna() if col in gg else pd.Series(dtype=float)
                if len(v) < 5:
                    continue
                t, p = stats.ttest_1samp(v, 0)
                ci = stats.t.interval(.95, len(v) - 1, v.mean(), stats.sem(v))
                star = " *" if p < .05 else ""
                print(f"  {roiname:12s} {g:3d} {nm:26s} {len(v):3d} "
                      f"{v.mean():+7.3f} {t:+7.2f} {p:9.4g} "
                      f"[{ci[0]:+.2f}, {ci[1]:+.2f}]{star}")
                rec = dict(roi=roiname, lag=int(g), test=nm,
                           n_sessions=int(len(v)), z=float(v.mean()),
                           t=float(t), df=int(len(v) - 1), p=float(p),
                           ci95=[float(ci[0]), float(ci[1])])
                out.append(rec); ov[f"{roiname}|lag{g}|{nm}"] = rec
        print()
    pd.DataFrame(out).to_csv(os.path.join(d, "summary.csv"), index=False)
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"), indent=2)
    json.dump(dict(seed=SEED, n_perm=N_PERM, roi_sets=ROI_SETS,
                   lags=list(LAGS), phases=list(PHASES), n_match=N_MATCH,
                   min_ripples=MIN_RIPPLES, min_cells=MIN_CELLS,
                   definition="goal_lag[g] = rew[(state-1+g) mod 4]",
                   window="each ripple's own detection duration; no warping",
                   unit="session",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"-> {d}")


if __name__ == "__main__":
    main()
