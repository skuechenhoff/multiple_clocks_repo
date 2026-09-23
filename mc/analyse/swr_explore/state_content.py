#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do ripples carry STATE (A/B/C/D), and is it factorised from location?

    python scripts/swr_content_explore.py state

SK's question: we know ripples carry the current location. Do they also carry
position in the ABCD sequence, and -- the interesting part -- are the two
represented as separable factors, so that "square 5 while heading to B" is built
from a square code and a state code rather than from a single conjunctive
pattern?

THREE TESTS, in this order.

  1  STATE, on its own. Four templates per unit (mean rate while seeking A, B,
     C, D, z-scored across the four). Evidence for the state actually being
     sought, minus the mean of the other three. Identical machinery to the
     location analysis: leave-one-configuration-out templates, per-unit
     z-scored counts, the ripple's own duration as the window, a per-session
     permutation null, t across sessions.

  2  BOTH AT ONCE. Location and state evidence from the same ripples, so
     "concurrent" is a measurement rather than an assumption.

  3  CONJUNCTION. 36 templates, one per (square, state). For each window the
     36 evidence values are regressed on three non-overlapping indicators:

         loc_only    same square, different state   (3 of 36)
         state_only  same state, different square   (8 of 36)
         both        the true conjunction            (1 of 36)
         reference   neither                        (24 of 36)

     ⚠ `both` on its own is NOT the conjunction test. With dummy coding it
     estimates (true cell - reference), which is exactly what additive
     location + state coding already predicts. The test is the INTERACTION

         b_both - (b_loc_only + b_state_only)

     FACTORISED  both marginals > 0, interaction ~ 0 -- the conjunction is the
                 sum of its parts, i.e. separable building blocks.
     CONJUNCTIVE interaction > 0 -- (square, state) carries more than square
                 and state do separately.

  4  BINDING. The compositional-replay proposals (Kurth-Nelson et al. 2023;
     He et al. 2026) do not claim that entities and roles are merely both
     present -- state coding is already known to be present everywhere in this
     dataset, in and out of ripples. They claim that a replay event BINDS an
     entity to its role. The observable version of that is COUPLING: within a
     single window, does strong evidence for the current square go together
     with strong evidence for the current state, and is that coupling stronger
     in ripples than in matched flanks?

     Both scores come from the same spike counts, so they are correlated by
     construction. That baseline is identical in ripples and flanks -- same
     units, same templates, same arithmetic -- so the ripple-minus-flank
     difference is the interpretable quantity, not the raw correlation. Total
     window activity is partialled out, because a window with more spikes
     inflates both scores at once.

⚠ THE CONFOUND, and why half this file is about it. State advances
monotonically within a traversal, so it correlates with time-into-traversal at
r = 0.40 (measured). A unit whose rate merely drifts upward across a traversal
would produce a perfect-looking "state D" template. Leave-one-configuration-out
does NOT help, because the drift happens inside every configuration. Two
controls, both reported:

  detrended  the linear effect of time-into-traversal removed from the template
             side AND the window side before anything is scored
  matched    state pairs compared only within matched bands of
             time-into-traversal -- assumption-free, and it costs data

A state effect that survives neither is a time effect.

NULL. With four conditions there are only 24 label permutations, so the state
null is EXHAUSTIVE (all 23 non-identity permutations) rather than sampled. The
36-condition conjunction null is sampled at N_PERM.

@author: Svenja Kuchenhoff
"""

import os
import json
import datetime
import itertools

import numpy as np
import pandas as pd
from scipy import stats

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc

SEED = 42
N_PERM = 200
ROI_SETS = {"HC_all": ["HC_anterior", "HC_mid"],
            "HC_mid": ["HC_mid"], "HC_anterior": ["HC_anterior"],
            "mOFC": ["mOFC"], "mPFC": ["mPFC"]}
N_STATE, N_LOC = 4, 9
N_CONJ = N_LOC * N_STATE
MIN_RIPPLES = 30
MIN_CELLS = 2
FLANK_GAP_S = 0.05
MAX_FLANKS = 2
TIME_BINS = np.array([0, 2, 4, 7, 12, 1e9])      # time-into-traversal, seconds
STATE_PERMS = [np.array(p) for p in itertools.permutations(range(N_STATE))
               if list(p) != list(range(N_STATE))]


def _time_into_traversal(steps_s, occ):
    """Seconds since the start of the current traversal, per occupancy interval."""
    st = steps_s.sort_values("t_s").reset_index(drop=True)
    k = occ.step_idx.to_numpy()
    trial = st.trial_idx.to_numpy(float)[k]
    start = occ.start_s.to_numpy()
    out = np.zeros(len(occ))
    for tr in np.unique(trial):
        m = trial == tr
        out[m] = start[m] - start[m].min()
    return out, st.state.to_numpy(float)[k]


def _templates(spk_s, cells, occ, cond, n_cond, grids, detrend_t=None):
    """{held-out configuration: (n_cells, n_cond)}, or None per cell if unusable."""
    out = {}
    counts = None
    if detrend_t is not None:
        dur = (occ.stop_s - occ.start_s).to_numpy()
        counts = {}
        for c in cells:
            n = swc.count_in(spk_s["spikes"][c], occ.start_s.to_numpy(),
                             occ.stop_s.to_numpy())
            rate = np.divide(n, dur, out=np.zeros_like(n, float), where=dur > 0)
            # residual rate x duration puts it back on the "counts" scale that
            # cond_map divides by duration again
            counts[c] = swc.detrend_on(rate, detrend_t) * dur
    for gr in grids:
        T = []
        for c in cells:
            m = swc.cond_map(spk_s["spikes"][c], occ, cond, n_cond,
                             exclude_grid=gr,
                             weights=None if counts is None else counts[c])
            z = swc.zscore_cond(m)
            T.append(np.zeros(n_cond) if z is None else z)
        out[gr] = np.array(T)
    return out


def _score(C, win_grid, loo):
    E = np.full((C.shape[1], next(iter(loo.values())).shape[1]), np.nan)
    for gr in np.unique(win_grid):
        T = loo.get(gr)
        if T is None:
            continue
        m = win_grid == gr
        E[m] = C[:, m].T @ T
    return E


def _target_minus_others(E, target):
    ar = np.arange(len(E))
    n = E.shape[1]
    h = E[ar, target]
    return h - (E.sum(axis=1) - h) / (n - 1)


def _partial_r(x, y, z):
    """Correlation of x and y with z regressed out of both."""
    X = np.column_stack([np.ones(len(z)), z])
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _z(v, nul):
    sd = np.nanstd(nul)
    return float((v - np.nanmean(nul)) / sd) if sd > 0 else np.nan


def main():
    rng = np.random.default_rng(SEED)
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s",
                               "subject_key"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)
    conj_perms = [rng.permutation(N_CONJ) for _ in range(N_PERM)]

    rows, conj_rows, couple_rows = [], [], []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < MIN_RIPPLES:
            continue
        subj = rip[rip.session == s].subject_key.iloc[0]
        tit_i, state_i = _time_into_traversal(steps[steps.session == s], occ)
        if not np.isfinite(state_i).any():
            continue
        loc_i = occ["loc"].to_numpy()
        cond_state = np.where(np.isfinite(state_i), state_i - 1, -1).astype(int)
        cond_loc = (loc_i - 1).astype(int)
        cond_conj = np.where(cond_state >= 0, cond_loc * N_STATE + cond_state, -1)

        o = occ.sort_values("start_s")
        row_of = o.index.to_numpy()
        a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
        j = np.searchsorted(a, t_rip, side="right") - 1
        jj = np.clip(j, 0, len(a) - 1)
        k = row_of[jj]
        inside = (j >= 0) & (t_rip <= b[jj]) & (cond_state[k] >= 0)
        if inside.sum() < MIN_RIPPLES:
            continue
        t_rip, d_rip, k = t_rip[inside], d_rip[inside], k[inside]
        half = d_rip / 2.0
        grid = occ.cv_group.to_numpy()[k]
        w_state, w_loc = cond_state[k], cond_loc[k]
        w_conj = cond_conj[k]
        w_tit = tit_i[k] + (t_rip - occ.start_s.to_numpy()[k])
        grids = np.unique(occ.cv_group.to_numpy())

        t_fl, owner = swc.matched_flanks(t_rip, half, occ, other_t=t_rip,
                                         gap_s=FLANK_GAP_S, max_per=MAX_FLANKS)
        has_fl = len(t_fl) > 0

        for ri, (roiname, members) in enumerate(ROI_SETS.items()):
            cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                     if c < len(spk[s]["spikes"])]
            if len(cells) < MIN_CELLS:
                continue
            sets = [swc.window_counts(spk[s], cells, t_rip - half, t_rip + half)]
            if has_fl:
                w_fl_h = half[owner]
                sets.append(swc.window_counts(spk[s], cells, t_fl - w_fl_h,
                                              t_fl + w_fl_h))
            Z = swc.zscore_cells(sets)
            C_rip = Z[0]
            C_fl = Z[1] if has_fl else None

            for detrend in (False, True):
                dt = tit_i if detrend else None
                loo_s = _templates(spk[s], cells, occ, cond_state, N_STATE,
                                   grids, detrend_t=dt)
                loo_l = _templates(spk[s], cells, occ, cond_loc, N_LOC,
                                   grids, detrend_t=dt)
                Cr = swc.detrend_on(C_rip, w_tit) if detrend else C_rip
                out = dict(session=s, subject=subj, roi=roiname,
                           detrend=detrend, n_cells=len(cells),
                           n_ripples=int(len(t_rip)))
                for tag, loo, tgt, perms in (
                        ("state", loo_s, w_state, STATE_PERMS),
                        ("location", loo_l, w_loc,
                         [rng.permutation(N_LOC) for _ in range(50)])):
                    E = _score(Cr, grid, loo)
                    ok = np.isfinite(E).all(axis=1)
                    if ok.sum() < MIN_RIPPLES:
                        continue
                    Eo, to = E[ok], tgt[ok]
                    v = float(np.mean(_target_minus_others(Eo, to)))
                    nul = np.array([float(np.mean(
                        _target_minus_others(Eo, pm[to]))) for pm in perms])
                    out[f"{tag}_score"] = v
                    out[f"z_{tag}"] = _z(v, nul)
                    if C_fl is not None:
                        Cf = swc.detrend_on(C_fl, w_tit[owner]) if detrend \
                            else C_fl
                        Ef = _score(Cf, grid[owner], loo)
                        okf = np.isfinite(Ef).all(axis=1)
                        if okf.sum() >= MIN_RIPPLES:
                            tf = tgt[owner][okf]
                            vf = float(np.mean(
                                _target_minus_others(Ef[okf], tf)))
                            nf = np.array([float(np.mean(_target_minus_others(
                                Ef[okf], pm[tf]))) for pm in perms])
                            out[f"z_{tag}_flank"] = _z(vf, nf)
                            out[f"z_{tag}_diff"] = _z(v - vf, nul - nf)
                # ---- time-matched control, state only, no detrending needed
                if not detrend:
                    E = _score(C_rip, grid, loo_s)
                    ok = np.isfinite(E).all(axis=1)
                    if ok.sum() >= MIN_RIPPLES:
                        bins = np.digitize(w_tit[ok], TIME_BINS) - 1
                        Eo, to = E[ok], w_state[ok]
                        vals, nulls = [], []
                        for bn in np.unique(bins):
                            mb = bins == bn
                            if mb.sum() < 20 or len(np.unique(to[mb])) < 2:
                                continue
                            vals.append(np.mean(
                                _target_minus_others(Eo[mb], to[mb])))
                            nulls.append([np.mean(_target_minus_others(
                                Eo[mb], pm[to[mb]])) for pm in STATE_PERMS])
                        if vals:
                            out["z_state_timematched"] = _z(
                                float(np.mean(vals)),
                                np.mean(np.array(nulls), axis=0))
                rows.append(out)

            # ---- binding: do the two scores covary WITHIN a window, more so
            #      in ripples than in matched flanks?
            if C_fl is not None:
                loo_s = _templates(spk[s], cells, occ, cond_state, N_STATE,
                                   grids, detrend_t=tit_i)
                loo_l = _templates(spk[s], cells, occ, cond_loc, N_LOC,
                                   grids, detrend_t=tit_i)
                Cr = swc.detrend_on(C_rip, w_tit)
                Cf = swc.detrend_on(C_fl, w_tit[owner])
                cpl = dict(session=s, subject=subj, roi=roiname,
                           n_cells=len(cells))
                for wname, Cx, gx, sx, lx in (
                        ("ripple", Cr, grid, w_state, w_loc),
                        ("flank", Cf, grid[owner], w_state[owner],
                         w_loc[owner])):
                    Es = _score(Cx, gx, loo_s)
                    El = _score(Cx, gx, loo_l)
                    ok = np.isfinite(Es).all(axis=1) & np.isfinite(El).all(axis=1)
                    if ok.sum() < MIN_RIPPLES:
                        continue
                    ss = _target_minus_others(Es[ok], sx[ok])
                    ll = _target_minus_others(El[ok], lx[ok])
                    act = np.abs(Cx[:, ok]).sum(axis=0)   # total window activity
                    cpl[f"r_{wname}"] = _partial_r(ll, ss, act)
                    cpl[f"n_{wname}"] = int(ok.sum())
                if "r_ripple" in cpl and "r_flank" in cpl:
                    cpl["dz"] = (np.arctanh(np.clip(cpl["r_ripple"], -.999, .999))
                                 - np.arctanh(np.clip(cpl["r_flank"], -.999, .999)))
                    couple_rows.append(cpl)

            # ---- conjunction, on the detrended data
            loo_c = _templates(spk[s], cells, occ, cond_conj, N_CONJ, grids,
                               detrend_t=tit_i)
            Cr = swc.detrend_on(C_rip, w_tit)
            E = _score(Cr, grid, loo_c)
            ok = np.isfinite(E).all(axis=1)
            if ok.sum() >= MIN_RIPPLES:
                Eo = E[ok]
                tl, ts = w_loc[ok], w_state[ok]
                ll = np.arange(N_CONJ) // N_STATE
                ss = np.arange(N_CONJ) % N_STATE
                same_l = ll[None, :] == tl[:, None]
                same_s = ss[None, :] == ts[:, None]
                X = np.stack([(same_l & ~same_s).astype(float),
                              (~same_l & same_s).astype(float),
                              (same_l & same_s).astype(float),
                              np.ones_like(same_l, float)], axis=2)
                A = X.reshape(-1, X.shape[2])
                B = np.linalg.pinv(A)
                Y = np.stack([Eo] + [Eo[:, pm] for pm in conj_perms],
                             axis=2).reshape(-1, 1 + N_PERM)
                Bet = B @ Y
                cr = dict(session=s, subject=subj, roi=roiname,
                          n_cells=len(cells), n_windows=int(ok.sum()))
                for ti, tname in enumerate(("loc_only", "state_only", "both")):
                    cr[f"b_{tname}"] = float(Bet[ti, 0])
                    cr[f"z_{tname}"] = _z(Bet[ti, 0], Bet[ti, 1:])
                # THE conjunction test: does the true cell exceed what the two
                # marginals already predict? Same permutation draws, so the
                # null is the identical linear combination.
                inter = Bet[2] - Bet[0] - Bet[1]
                cr["b_interaction"] = float(inter[0])
                cr["z_interaction"] = _z(inter[0], inter[1:])
                conj_rows.append(cr)
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    K = pd.DataFrame(conj_rows)
    P = pd.DataFrame(couple_rows)
    d = os.path.join(deriv, "group", "swr",
                     f"ripple_state_content_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "state_per_session.csv"), index=False)
    K.to_csv(os.path.join(d, "conjunction_per_session.csv"), index=False)
    P.to_csv(os.path.join(d, "coupling_per_session.csv"), index=False)
    overview = report(R, K, P)
    json.dump(overview, open(os.path.join(d, "results_overview.json"), "w"),
              indent=2)
    json.dump(dict(seed=SEED, n_perm=N_PERM, roi_sets=ROI_SETS,
                   state="the reward being SOUGHT (steps `state`, 1-4 = A-D)",
                   drift_control="linear time-into-traversal removed from both "
                                 "template and window; plus a time-matched "
                                 "variant",
                   time_bins_s=TIME_BINS[:-1].tolist(),
                   state_null="exhaustive, all 23 non-identity permutations",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"\n-> {d}")


def _p(v):
    v = v.dropna()
    return stats.ttest_1samp(v, 0)[1] if len(v) >= 5 else np.nan


def report(R, K, P):
    """Print the tables AND return the machine-readable overview."""
    ov = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
          "state_definition": "the reward being SOUGHT (steps `state`, 1-4=A-D)",
          "tests": {}}

    print("\n=== 1-2. STATE and LOCATION in the same ripples ===")
    print("    `detrend` = linear time-into-traversal removed from template "
          "AND window\n")
    print(f"{'ROI':12s} {'detrend':>8s} {'sess':>5s} | "
          f"{'z state':>8s} {'p':>8s} | {'z loc':>8s} {'p':>8s} | "
          f"{'state rip-flank':>15s} {'p':>8s}")
    marg = {}
    for roiname in ROI_SETS:
        for dt in (False, True):
            g = R[(R.roi == roiname) & (R.detrend == dt)]
            if len(g) < 5:
                continue
            dif = g.get("z_state_diff", pd.Series(dtype=float))
            print(f"{roiname:12s} {str(dt):>8s} {len(g):5d} | "
                  f"{g.z_state.mean():+8.3f} {_p(g.z_state):8.3g} | "
                  f"{g.z_location.mean():+8.3f} {_p(g.z_location):8.3g} | "
                  f"{dif.mean():+15.3f} {_p(dif):8.3g}")
            marg[f"{roiname}_detrend={dt}"] = dict(
                n_sessions=int(len(g)),
                z_state=float(g.z_state.mean()), p_state=float(_p(g.z_state)),
                z_location=float(g.z_location.mean()),
                p_location=float(_p(g.z_location)),
                z_state_ripple_minus_flank=float(dif.mean()),
                p_state_ripple_minus_flank=float(_p(dif)))
    print("\n    time-matched control (state only, no model-based adjustment):")
    for roiname in ROI_SETS:
        g = R[(R.roi == roiname) & (~R.detrend)]
        v = g.get("z_state_timematched", pd.Series(dtype=float)).dropna()
        if len(v) >= 5:
            print(f"      {roiname:12s} n = {len(v):3d}  {v.mean():+.3f}  "
                  f"p = {_p(v):.3g}")
            marg.setdefault(f"{roiname}_timematched", {}).update(
                dict(n_sessions=int(len(v)), z_state=float(v.mean()),
                     p=float(_p(v))))
    ov["tests"]["state_and_location"] = marg

    print("\n=== 3. CONJUNCTION: is (square, state) more than its parts? ===")
    print("    ⚠ `both` is NOT the test -- additive coding predicts it. The")
    print("      test is the INTERACTION, b_both - (b_loc + b_state).\n")
    print(f"{'ROI':12s} {'sess':>5s} | {'loc only':>9s} {'p':>8s} | "
          f"{'state only':>10s} {'p':>8s} || {'INTERACTION':>12s} {'p':>8s} "
          f"{'95% CI':>18s}")
    conj = {}
    for roiname in ROI_SETS:
        g = K[K.roi == roiname]
        if len(g) < 5:
            continue
        v = g.z_interaction.dropna()
        sem = v.std(ddof=1) / np.sqrt(len(v))
        ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=sem)
        print(f"{roiname:12s} {len(g):5d} | "
              f"{g.z_loc_only.mean():+9.3f} {_p(g.z_loc_only):8.3g} | "
              f"{g.z_state_only.mean():+10.3f} {_p(g.z_state_only):8.3g} || "
              f"{v.mean():+12.3f} {_p(v):8.3g} "
              f"{f'[{ci[0]:+.2f}, {ci[1]:+.2f}]':>18s}")
        conj[roiname] = dict(
            n_sessions=int(len(g)),
            z_loc_only=float(g.z_loc_only.mean()), p_loc_only=float(_p(g.z_loc_only)),
            z_state_only=float(g.z_state_only.mean()),
            p_state_only=float(_p(g.z_state_only)),
            z_interaction=float(v.mean()), p_interaction=float(_p(v)),
            ci95_interaction=[float(ci[0]), float(ci[1])],
            b_loc_only=float(g.b_loc_only.mean()),
            b_state_only=float(g.b_state_only.mean()),
            b_both=float(g.b_both.mean()),
            b_additive_prediction=float((g.b_loc_only + g.b_state_only).mean()))
    ov["tests"]["conjunction"] = conj

    print("\n=== 4. BINDING: is location coupled to state MORE during ripples? ===")
    print("    partial correlation of the two scores within a window, total")
    print("    window activity removed. Ripple minus flank is the test.\n")
    print(f"{'ROI':12s} {'sess':>5s} | {'r ripple':>9s} {'r flank':>8s} | "
          f"{'dz(rip-flank)':>14s} {'p':>8s} {'95% CI':>18s}")
    cpl = {}
    for roiname in ROI_SETS:
        g = P[P.roi == roiname] if len(P) else P
        if len(g) < 5:
            continue
        v = g.dz.dropna()
        sem = v.std(ddof=1) / np.sqrt(len(v))
        ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=sem)
        print(f"{roiname:12s} {len(g):5d} | {g.r_ripple.mean():+9.3f} "
              f"{g.r_flank.mean():+8.3f} | {v.mean():+14.3f} {_p(v):8.3g} "
              f"{f'[{ci[0]:+.3f}, {ci[1]:+.3f}]':>18s}")
        cpl[roiname] = dict(
            n_sessions=int(len(g)), r_ripple=float(g.r_ripple.mean()),
            r_flank=float(g.r_flank.mean()),
            dz_ripple_minus_flank=float(v.mean()), p=float(_p(v)),
            ci95=[float(ci[0]), float(ci[1])])
    ov["tests"]["binding_coupling"] = cpl
    return ov




if __name__ == "__main__":
    main()
