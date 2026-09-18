#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I12 and I13 -- what ROLE does a square have to play to appear in a ripple?

    python scripts/swr_ripple_content_roles.py

C0-C2 asked about the square underfoot and about the reward set. These two ask
about the other eight squares, and they are the sharpest form of SK's question
-- "is hippocampus sending the goals, or the route?" and "does a square where
you made a mistake get actively suppressed?"

  I12  Once all four rewards are known, is the WHOLE trajectory represented, or
       only the rewards? A set representation and a trajectory representation
       differ exactly here.
  I13  Is a square where an erroneous uncover was made BLOCKED -- pushed below
       the squares with no such history -- rather than merely not represented?
       A negative coefficient is the interesting result.

ONE REGRESSION PER WINDOW, ALL NINE SQUARES, because the roles are correlated by
construction: a reward is a place you have been, a place you have been recently
is a place you walked through, an error square is a square you visited. Scoring
a role on its own would credit it with every other role it travels with.

    explore (first traversal)
      E(L) ~ current + known_reward + unknown_reward + errors_here
             + visits_here + recent + SQUARE

    known (every later traversal)
      E(L) ~ current + reward + on_route + errors_here
             + visits_here + recent + SQUARE
      -- reference category: a non-reward square OFF the walked route

`on_route` is not a guess at the optimal path: it is the set of squares the
subject actually walked on the correct repeats of that grid run, minus the
rewards themselves. Measured, per run: 4 reward squares, 3.51 on-route
non-reward squares, 1.49 off-route squares, and 69% of runs have at least one
off-route square. The off-route reference is the thin class and I12's power sits
on it.

`errors_here` is a COUNT, not a flag, and I13 is run in the explore phase where
it varies: by the end of a first traversal a median of 6 squares carry an error
history, so a binary contrast has almost nothing left to compare. Early in the
traversal it is graded, and 45% of steps sit in the 1-6 range.

SQUARE = eight location dummies. This is new here and it is the fix for C2's
unexplained residual (`unknown_rew` = +0.218 where it must be zero). A square's
template is estimated from however much time the subject spent there --
measured, occupancy runs from 8.9% at square 1 to 14.8% at square 5 -- so
evidence has a square-identity baseline that has nothing to do with any role.
The dummies are identifiable because the same square plays different roles in
different configurations.

Everything is computed in ripple windows AND in matched flanks (same square,
same occupancy interval, same width), and the reported contrast is ripple minus
flank. Null: the nine template labels are permuted, which reorders `E` within
each window; the design is untouched, so the whole null is ONE matmul against a
precomputed pseudo-inverse.

Unit of inference: SESSION. Exploratory; nothing here is pre-registered.

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
N_PERM = 100
ROI_SETS = {"HC_all": ["HC_anterior", "HC_mid"],
            "HC_mid": ["HC_mid"], "HC_anterior": ["HC_anterior"],
            "mOFC": ["mOFC"], "mPFC": ["mPFC"]}
FLANK_GAP_S = 0.05
MAX_FLANKS = 2
RECENT_S = 10.0
MIN_WINDOWS = 50
# Untuned cells dilute every term in the model. Selecting on split-half place-map
# reliability uses all task time, so it is shared by ripple and flank alike and
# cannot manufacture a difference between them -- it only lifts the signal any
# role effect would have to stand out from. At rel >= 0.2 the ambient
# current-location signal roughly doubles (CHANGELOG 2026-09-17 m).
# See `swc.weighted_loo`. An earlier version selected cells on a reliability
# computed over ALL configurations, including the one being scored, and dropped
# sessions whose cells fell below the cut. Both inflated the result
# (CHANGELOG 2026-09-17 o). Weights are now estimated with the scored
# configuration held out and every session is kept.
SCHEMES = ["all", "thresh_0.2"]
# An occupancy interval runs from one move onset to the next, so a ripple late
# in it happens while the subject is already moving towards `next_loc`.
# Splitting at the session's median latency separates "the map codes where I am
# going" from "I have half-arrived": a prospective signal should survive in the
# EARLY half, an arrival artefact should not.
LAT_HALVES = ("early", "late")
# `goal` is the reward being sought NOW; `goal_1` and `goal_2` are the ones
# sought after it (the sequence loops, so D is followed by A). During
# exploration the subject cannot know `goal_1`/`goal_2`, which makes them a
# built-in knowledge control: they must be null there and are free to be
# non-null once everything is known.
TERMS = {"explore": ["current", "adjacent", "goal", "goal_1", "goal_2",
                     "next_step", "known_rew", "unknown_rew", "errors_here",
                     "visits_here", "recent", "train_occ"],
         "known": ["current", "adjacent", "goal", "goal_1", "goal_2",
                   "next_step", "reward", "on_route", "errors_here",
                   "visits_here", "recent", "train_occ"]}
# (latency half, trial subset). The correct-trials restriction answers whether
# any of this depends on the subject actually executing the sequence properly.
# ⚠ Every subset except the two `after_*` ones takes ALL ripples in the phase,
# wherever the subject happened to be. `after_reward` / `after_error` restrict to
# ripples in the occupancy interval that BEGINS with an uncover press -- i.e. the
# subject has just pressed and is standing on what they uncovered. That is the
# moment reward information actually arrives, and it is a different question
# from "any ripple during exploration".
SPLITS = [("all", "all"), ("early", "all"), ("late", "all"), ("all", "correct"),
          ("all", "early_known"), ("all", "late_known"),
          ("all", "after_reward"), ("all", "after_error")]
LOC = np.arange(1, 10)


def _adjacency():
    """(9, 9) 4-connected neighbours. loc = col*3 + row + 1 (column-major).

    ⚠ This is why `adjacent` has to be in the model. `next_loc` is ALWAYS one
    step from the current square, and place templates are spatially smooth --
    the pseudo-population evidence matrix shows neighbours sharing the
    diagonal's elevation. Without an adjacency term, "the map codes where I am
    going next" is indistinguishable from "the map is blurry".
    """
    A = np.zeros((9, 9), bool)
    for i in range(9):
        ri, ci = i % 3, i // 3
        for j in range(9):
            rj, cj = j % 3, j // 3
            if abs(ri - rj) + abs(ci - cj) == 1:
                A[i, j] = True
    return A


ADJ = _adjacency()


def run_history(steps_s, occ):
    """Per occupancy interval, the state of the world in its own grid run.

    Returns, aligned to `occ` rows: the four rewards, which are known, how many
    erroneous uncovers each square has collected SO FAR in this run, how many
    times each square has been visited so far, the time since each was last
    visited, whether the interval is a first traversal, and the set of squares
    walked on the correct repeats of the run.
    """
    st = steps_s.sort_values("t_s").reset_index(drop=True)
    k = occ.step_idx.to_numpy()
    n = len(occ)
    goal = st.looking_for_loc.to_numpy(float)[k]
    nxt = st.next_loc.to_numpy(float)[k]
    rew = st[["rew_A", "rew_B", "rew_C", "rew_D"]].to_numpy(float)[k]
    correct = st.trial_correct.to_numpy(float)[k] == 1
    state = st.state.to_numpy(float)[k]
    known = np.arange(1, 5)[None, :] < state[:, None]
    # the next two rewards in the loop after the one being sought
    si = np.clip(state.astype(int), 1, 4) - 1
    goal1 = rew[np.arange(n), (si + 1) % 4]
    goal2 = rew[np.arange(n), (si + 2) % 4]
    first = st.first_trial.to_numpy(float)[k] == 1

    # how much data the TEMPLATE for each square rests on: dwell at that square
    # in every OTHER configuration, which is what the leave-one-out template is
    # built from. ⚠ This is the control `on_route` needs. Off-route squares are
    # by definition the rarely-visited ones, so their template entries are the
    # worst estimated, and `on_route` is scored against exactly them. Per-session
    # square occupancy would be absorbed by the square dummies; this varies
    # across configurations within a session, so it is not.
    occ_sq = np.zeros((n, 9))
    dwell_o = (occ.stop_s.to_numpy() - occ.start_s.to_numpy())
    loc_o = occ["loc"].to_numpy().astype(int)
    cv_o = occ.cv_group.to_numpy()
    tot = np.zeros(9)
    for L in range(1, 10):
        tot[L - 1] = dwell_o[loc_o == L].sum()
    err = np.zeros((n, 9)); vis = np.zeros((n, 9))
    last = np.full((n, 9), np.inf)
    route = np.zeros((n, 9), bool)

    t_all = st.t_s.to_numpy(float)
    is_err = (st.is_uncover.to_numpy(float) == 1) & \
             (st.correct_uncover.to_numpy(float) == 0)
    loc_all = st["loc"].to_numpy(float)
    run_all = st.grid_num.to_numpy(float)
    ok_rep = (st.first_trial.to_numpy(float) != 1) & \
             (st.trial_correct.to_numpy(float) == 1)

    start = occ.start_s.to_numpy()
    run = occ.grid_num.to_numpy()
    for g in np.unique(cv_o):
        rows_g = np.flatnonzero(cv_o == g)
        here = np.zeros(9)
        for L in range(1, 10):
            here[L - 1] = dwell_o[(loc_o == L) & (cv_o == g)].sum()
        occ_sq[rows_g] = np.log1p(tot - here)[None, :]
    sd = occ_sq.std()
    occ_sq = (occ_sq - occ_sq.mean()) / sd if sd > 0 else occ_sq * 0
    for g in np.unique(run):
        rows = np.flatnonzero(run == g)
        m_all = run_all == g
        # the route this run settles on: squares walked on correct repeats
        r_sq = np.unique(loc_all[m_all & ok_rep]).astype(int)
        route[np.ix_(rows, r_sq - 1)] = True
        te, le = t_all[m_all & is_err], loc_all[m_all & is_err]
        tv, lv = t_all[m_all], loc_all[m_all]
        for L in range(1, 10):
            a = np.sort(te[le == L])
            err[rows, L - 1] = np.searchsorted(a, start[rows], side="left")
            b = np.sort(tv[lv == L])
            c = np.searchsorted(b, start[rows], side="left")
            vis[rows, L - 1] = c
            has = c > 0
            last[rows[has], L - 1] = (start[rows[has]]
                                      - b[np.clip(c[has] - 1, 0, len(b) - 1)])
    trial_n = st.trial_num_in_grid.to_numpy(float)[k]
    # does this interval start with an uncover press, and was it rewarded?
    unc = st.is_uncover.to_numpy(float)[k] == 1
    cor_unc = st.correct_uncover.to_numpy(float)[k] == 1
    return (rew, known, err, vis, last, first, route, goal, nxt,
            goal1, goal2, correct, occ_sq, trial_n, unc, cor_unc)


def design(phase, loc, rew, known, err, vis, last, route, goal, nxt,
           goal1, goal2, occ_sq):
    """(n_windows, 9, n_terms + 8 square dummies + 1) -- see the module docstring."""
    n = len(loc)
    L = LOC[None, :]
    is_cur = L == loc[:, None]
    # the square the subject is trying to REACH, and the square they step to
    # next. These are the two prospective roles -- "tell prefrontal what to do"
    # -- and they are the most on-hypothesis terms in the model.
    is_goal = (L == goal[:, None]) & ~is_cur
    is_g1 = (L == goal1[:, None]) & ~is_cur & ~is_goal
    is_g2 = (L == goal2[:, None]) & ~is_cur & ~is_goal & ~is_g1
    is_next = (L == nxt[:, None]) & ~is_cur
    # every square one step from the current one, minus the ones that already
    # have a role of their own
    is_adj = ADJ[loc - 1] & ~is_cur
    is_rew = np.zeros((n, 9), bool)
    is_unk = np.zeros((n, 9), bool)
    for j in range(4):
        m = L == rew[:, j][:, None]
        is_rew |= m & known[:, j][:, None]
        is_unk |= m & ~known[:, j][:, None]
    is_rew &= ~is_cur & ~is_goal & ~is_g1 & ~is_g2
    is_unk &= ~is_cur & ~is_goal & ~is_g1 & ~is_g2
    on_route = (route & ~is_rew & ~is_unk & ~is_cur & ~is_goal & ~is_g1
                & ~is_g2 & ~is_next & ~is_adj)
    recent = (last < RECENT_S) & ~is_cur & ~is_goal & ~is_g1 & ~is_g2 \
        & ~is_next & ~is_adj

    def z(a):
        s = a.std()
        return (a - a.mean()) / s if s > 0 else np.zeros_like(a)

    is_adj = is_adj & ~is_goal & ~is_g1 & ~is_g2 & ~is_next
    cols = ([is_cur.astype(float), is_adj.astype(float),
             is_goal.astype(float), is_g1.astype(float), is_g2.astype(float),
             is_next.astype(float)]
            + ([is_rew.astype(float), is_unk.astype(float)] if phase == "explore"
               else [is_rew.astype(float), on_route.astype(float)])
            + [z(err), z(vis), recent.astype(float), occ_sq]
            + [np.tile((LOC == j).astype(float), (n, 1))
               for j in range(2, 10)]          # square dummies, square 1 = ref
            + [np.ones((n, 9))])
    return np.stack(cols, axis=2)


def fit_all(X, E, perms):
    """Betas for the true labelling and every permutation, in two matmuls.

    The design never changes -- a permutation only reorders `E` within a window
    -- so the pseudo-inverse is computed ONCE and every null draw is a matmul
    against it. Same code path as the empirical value, as CLAUDE.md requires.
    """
    n, _, k = X.shape
    A = X.reshape(-1, k)
    good = np.isfinite(A).all(axis=1) & np.isfinite(E).all(axis=1).repeat(9)
    A = A[good]
    if len(A) < 10 * k or np.linalg.matrix_rank(A) < k:
        return None
    B = np.linalg.pinv(A)                                   # (k, rows)
    cols = [E] + [E[:, p] for p in perms]
    Y = np.stack(cols, axis=2).reshape(-1, len(cols))[good]   # (rows, 1 + P)
    return B @ Y                                              # (k, 1 + P)


def main():
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(9) for _ in range(N_PERM)]
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s", "subject_key"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)

    rows = []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or not len(t_rip):
            continue
        subj = rip[rip.session == s].subject_key.iloc[0]
        st_s = steps[steps.session == s]
        (rew_i, known_i, err_i, vis_i, last_i, first_i, route_i,
         goal_i, nxt_i, g1_i, g2_i, corr_i, occq_i,
         trialn_i, unc_i, corunc_i) = run_history(st_s, occ)

        o = occ.sort_values("start_s")
        row_of = o.index.to_numpy()
        a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
        j = np.searchsorted(a, t_rip, side="right") - 1
        jj = np.clip(j, 0, len(o) - 1)
        k = row_of[jj]
        inside = (j >= 0) & (t_rip <= b[jj])
        if inside.sum() < MIN_WINDOWS:
            continue
        t_rip, d_rip, k = t_rip[inside], d_rip[inside], k[inside]
        loc = occ["loc"].to_numpy()[k].astype(int)
        grid = occ.cv_group.to_numpy()[k]
        lat_in = t_rip - occ.start_s.to_numpy()[k]   # how far into the interval
        half = d_rip / 2.0

        t_fl, owner = swc.matched_flanks(t_rip, half, occ, other_t=t_rip,
                                         gap_s=FLANK_GAP_S, max_per=MAX_FLANKS)
        if not len(t_fl):
            continue
        w_fl = half[owner]
        grids = np.unique(occ.cv_group.to_numpy())

        for roiname, members in ROI_SETS.items():
            all_cells = r[r.roi.isin(members)].cell.to_numpy()
            all_cells = [c for c in all_cells if c < len(spk[s]["spikes"])]
            if len(all_cells) < 2:
                continue
            cells = all_cells
            for scheme in SCHEMES:
                loo = swc.weighted_loo(spk[s], cells, occ, grids, scheme)
                if not any(np.nansum(np.abs(T)) > 0 for T in loo.values()):
                    continue
                C_rip, C_fl = swc.zscore_cells([
                    swc.window_counts(spk[s], cells, t_rip - half, t_rip + half),
                    swc.window_counts(spk[s], cells, t_fl - w_fl, t_fl + w_fl)])
                E_rip = swc.score_windows(C_rip, t_rip, grid, loo)
                E_fl = swc.score_windows(C_fl, t_fl, grid[owner], loo)

                for phase in ("explore", "known"):
                  ph_m = first_i[k] if phase == "explore" else ~first_i[k]
                  for lat_half, subset in SPLITS:
                      if lat_half == "all":
                          m = ph_m
                      else:
                          cut = np.median(lat_in[ph_m])
                          m = ph_m & ((lat_in <= cut) if lat_half == "early"
                                      else (lat_in > cut))
                      if subset == "correct":
                          m = m & corr_i[k]
                      elif subset == "after_reward":
                          m = m & unc_i[k] & corunc_i[k]
                      elif subset == "after_error":
                          m = m & unc_i[k] & ~corunc_i[k]
                      elif subset in ("early_known", "late_known"):
                          if phase != "known":
                              continue
                          tn = trialn_i[k]
                          cut = np.median(tn[ph_m])
                          m = m & ((tn <= cut) if subset == "early_known"
                                   else (tn > cut))
                      mf = m[owner]
                      if m.sum() < MIN_WINDOWS or mf.sum() < MIN_WINDOWS:
                          continue
                      Xr = design(phase, loc[m], rew_i[k][m], known_i[k][m],
                                  err_i[k][m], vis_i[k][m], last_i[k][m],
                                  route_i[k][m], goal_i[k][m], nxt_i[k][m],
                                  g1_i[k][m], g2_i[k][m], occq_i[k][m])
                      kf = k[owner][mf]
                      Xf = design(phase, loc[owner][mf], rew_i[kf], known_i[kf],
                                  err_i[kf], vis_i[kf], last_i[kf], route_i[kf],
                                  goal_i[kf], nxt_i[kf], g1_i[kf], g2_i[kf],
                                  occq_i[kf])
                      Br = fit_all(Xr, E_rip[m], perms)
                      Bf = fit_all(Xf, E_fl[mf], perms)
                      if Br is None or Bf is None:
                          continue
                      out = dict(session=s, subject=subj, roi=roiname,
                                 phase=phase, scheme=scheme,
                                 lat_half=lat_half, subset=subset,
                                 n_cells=len(cells),
                                 n_windows=int(m.sum()))
                      for ti, tname in enumerate(TERMS[phase]):
                          for nm, B in (("rip", Br), ("flank", Bf)):
                              v, nul = B[ti, 0], B[ti, 1:]
                              sd = np.nanstd(nul)
                              out[f"b_{tname}_{nm}"] = float(v)
                              out[f"z_{tname}_{nm}"] = (float((v - np.nanmean(nul)) / sd)
                                                        if sd > 0 else np.nan)
                          dv = Br[ti, 0] - Bf[ti, 0]
                          dn = Br[ti, 1:] - Bf[ti, 1:]
                          sd = np.nanstd(dn)
                          out[f"b_{tname}_diff"] = float(dv)
                          out[f"z_{tname}_diff"] = (float((dv - np.nanmean(dn)) / sd)
                                                    if sd > 0 else np.nan)
                      rows.append(out)
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"ripple_content_roles_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "roles_per_session.csv"), index=False)
    report(R, d)
    json.dump(dict(seed=SEED, n_perm=N_PERM, roi_sets=ROI_SETS,
                   terms=TERMS, recent_s=RECENT_S, min_windows=MIN_WINDOWS,
                   schemes=SCHEMES,
                   square_dummies="8 location dummies, square 1 as reference "
                                  "-- absorbs the square-identity baseline that "
                                  "left C2's unknown_rew control at +0.218",
                   on_route="squares actually walked on the correct repeats of "
                            "that grid run, minus the rewards",
                   targets="I12 = on_route (known phase); "
                           "I13 = errors_here (explore phase, graded count); "
                           "goal = the square currently being sought "
                           "(looking_for_loc); next_step = the square stepped "
                           "to next",
                   primary="ripple minus matched flank, HC_all",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"\n-> {d}")


def report(R, d):
    print("\n=== role coefficients: is a square represented because of what it "
          "IS? ===")
    print("    square-identity dummies are IN the model, so these are role "
          "effects only.\n")
    summ = []
    for phase in ("explore", "known"):
      for scheme in SCHEMES:
       for lat_half, subset in SPLITS:
        print(f"  --- {phase}, weights = {scheme}, {lat_half} in the "
              f"interval, {subset} trials " + "-" * 6)
        for roiname in ROI_SETS:
            g = R[(R.roi == roiname) & (R.phase == phase)
                  & (R.scheme == scheme) & (R.lat_half == lat_half)
                  & (R.subset == subset)]
            if len(g) < 5:
                continue
            print(f"  {roiname}  (n = {len(g)} sessions, "
                  f"{g.n_cells.mean():.1f} cells, "
                  f"{int(g.n_windows.sum())} ripples)")
            print(f"    {'term':<14s} {'z in ripple':>12s} {'p':>9s} "
                  f"{'z rip-flank':>12s} {'p':>9s}")
            for t_ in TERMS[phase]:
                a_ = g[f"z_{t_}_rip"].dropna()
                b_ = g[f"z_{t_}_diff"].dropna()
                if len(a_) < 5:
                    continue
                pa = stats.ttest_1samp(a_, 0)[1]
                pb = stats.ttest_1samp(b_, 0)[1]
                star = ""
                if (phase == "known" and t_ == "on_route") or \
                   (phase == "explore" and t_ == "errors_here") or \
                   t_ in ("goal", "goal_1", "goal_2", "next_step",
                          "adjacent"):
                    star = "  <--"
                print(f"    {t_:<14s} {a_.mean():+12.3f} {pa:9.3g} "
                      f"{b_.mean():+12.3f} {pb:9.3g}{star}")
                summ.append(dict(phase=phase, scheme=scheme, lat_half=lat_half,
                                 subset=subset,
                                 roi=roiname, term=t_,
                                 sessions=len(a_), z_ripple=float(a_.mean()),
                                 p_ripple=float(pa), z_diff=float(b_.mean()),
                                 p_diff=float(pb)))
            print()
    pd.DataFrame(summ).to_csv(os.path.join(d, "roles_summary.csv"), index=False)


if __name__ == "__main__":
    main()
