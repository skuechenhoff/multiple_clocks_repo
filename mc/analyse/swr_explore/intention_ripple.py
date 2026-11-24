#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SK's idea: lock mPFC location coding to UNCOVERS, then ask what ripples do.

    python scripts/swr_content_explore.py intention_ripple

    mPFC encodes 'loc 2'  ->  subject uncovers loc 2  ->  correct?  -> ripples
                                                       ->  wrong?   -> fewer?

Everything else in this project locks to ripples and asks what is represented.
That cannot see a sequence. This locks to the BEHAVIOURAL event instead:

  1. before each uncover, measure how strongly mPFC represents the square that
     is about to be uncovered (a prospective INTENTION signal -- the square is
     read from what the subject does, not from what the task wants);
  2. after the uncover, count ripples;
  3. ask whether (1) predicts (2), and whether that depends on the uncover
     being correct.

The pre-window ends BEFORE the press so the measurement cannot contain the
outcome, and the ripple count starts AT the press so it cannot contain the
measurement -- the two windows never overlap.

Trial-level statistics are computed WITHIN session (Spearman, so a few large
ripple counts cannot drive it), then the per-session correlations are tested
across sessions. Session stays the unit of inference, as everywhere else here.

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
from mc.analyse.swr_explore.goal_lag import goal_lag_locations

SEED = 42
N_PERM = 1000
ROI_SETS = {"mPFC": ["mPFC"], "mOFC": ["mOFC"],
            "HC_all": ["HC_anterior", "HC_mid"]}
PRE_S = (0.50, 0.05)      # window before the press: [t-0.50, t-0.05]
POST_S = (0.0, 1.0)       # ripple count window after the press
MIN_EVENTS = 15
MIN_CELLS = 2


def _z(M):
    """z of a mean statistic against its own permutation null."""
    sd = M[..., 1:].std(axis=-1)
    return np.where(sd > 0, (M[..., 0] - M[..., 1:].mean(axis=-1)) / sd, np.nan)


def _stat(E, li, P):
    E = np.nan_to_num(E)
    H = np.take_along_axis(E, P[:, li].T, axis=1)
    return H - (E.sum(axis=1)[:, None] - H) / 8.0


def main():
    rng = np.random.default_rng(SEED)
    P = np.vstack([np.arange(9)] + [rng.permutation(9) for _ in range(N_PERM)])
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
        occ, r, _, _ = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r):
            continue
        st = steps[steps.session == s].sort_values("t_s").reset_index(drop=True)
        subj = rip[rip.session == s].subject_key.iloc[0]
        t_all = st.t_s.to_numpy(float)
        unc = np.flatnonzero(st.is_uncover.to_numpy(float) == 1)
        if len(unc) < MIN_EVENTS:
            continue
        t_ev = t_all[unc]
        loc_ev = st["loc"].to_numpy(float)[unc].astype(int)
        corr_ev = st.correct_uncover.to_numpy(float)[unc]
        grid_ev = st.grid_id.to_numpy(float)[unc]
        # phase of each uncover, via the occupancy interval it sits in
        ph_i, first_i, _, _ = phase_labels(st, occ)
        o = occ.sort_values("start_s")
        a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
        j = np.clip(np.searchsorted(a, t_ev, side="right") - 1, 0, len(a) - 1)
        k = o.index.to_numpy()[j]
        ph_ev = ph_i[k]
        # ripple count AFTER each uncover
        t_rip = np.sort(rip[rip.session == s].t_peak_s.to_numpy())
        n_rip = (np.searchsorted(t_rip, t_ev + POST_S[1])
                 - np.searchsorted(t_rip, t_ev + POST_S[0])).astype(float)
        grids = np.unique(occ.cv_group.to_numpy())
        # where was the subject DURING the pre-window, and what was the true
        # goal? Both are needed for the strict intention mask below.
        t_pre = t_ev - (PRE_S[0] + PRE_S[1]) / 2.0
        jp = np.clip(np.searchsorted(a, t_pre, side="right") - 1, 0, len(a) - 1)
        in_pre = (jp >= 0) & (t_pre <= b[jp])
        kp = o.index.to_numpy()[jp]
        cur_pre = np.where(in_pre, occ["loc"].to_numpy()[kp], -1).astype(int)
        goal0 = goal_lag_locations(st, occ)[kp, 0]
        # STRICT intention: the square about to be uncovered is neither where
        # the subject already stands nor the correct answer. Current-location
        # leakage and true-goal coding are both excluded by construction.
        strict = in_pre & (loc_ev != cur_pre) & (loc_ev != goal0)

        for roiname, members in ROI_SETS.items():
            cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                     if c < len(spk[s]["spikes"])]
            if len(cells) < MIN_CELLS:
                continue
            _, loo = swc.build_templates(spk[s], cells, occ, grids)
            C, = swc.zscore_cells([swc.window_counts(
                spk[s], cells, t_ev - PRE_S[0], t_ev - PRE_S[1])])
            E = swc.score_windows(C, t_ev, grid_ev, loo)
            ok = np.isfinite(E).all(axis=1) & (loc_ev >= 1) & (loc_ev <= 9)
            if ok.sum() < MIN_EVENTS:
                continue
            Sfull = _stat(E, np.clip(loc_ev - 1, 0, 8), P)     # (n_ev, 1+N_PERM)
            S = Sfull[:, 0]
            for phname, msk in (("all", np.ones(len(t_ev), bool)),
                                ("encode", ph_ev == "encode"),
                                ("execute", ph_ev == "execute")):
                m = ok & msk
                if m.sum() < MIN_EVENTS:
                    continue
                rec = dict(session=s, subject=subj, roi=roiname, phase=phname,
                           n_events=int(m.sum()))
                # is the about-to-be-uncovered square represented at all?
                rec["z_intent"] = float(_z(Sfull[m].mean(axis=0)))
                ms = m & strict
                rec["n_strict"] = int(ms.sum())
                rec["z_intent_strict"] = (float(_z(Sfull[ms].mean(axis=0)))
                                          if ms.sum() >= MIN_EVENTS else np.nan)
                rec["rho_strict"] = (float(stats.spearmanr(S[ms],
                                                           n_rip[ms])[0])
                                     if ms.sum() >= MIN_EVENTS else np.nan)
                # does pre-uncover intention strength predict ripples after?
                rec["rho_all"] = float(stats.spearmanr(S[m], n_rip[m])[0])
                for cname, cm in (("correct", corr_ev == 1),
                                  ("incorrect", corr_ev == 0)):
                    mm = m & cm
                    if mm.sum() < MIN_EVENTS:
                        rec[f"rho_{cname}"] = np.nan
                        rec[f"S_{cname}"] = np.nan
                        rec[f"nrip_{cname}"] = np.nan
                        continue
                    rec[f"rho_{cname}"] = float(
                        stats.spearmanr(S[mm], n_rip[mm])[0])
                    rec[f"S_{cname}"] = float(np.mean(S[mm]))
                    rec[f"nrip_{cname}"] = float(np.mean(n_rip[mm]))
                rows.append(rec)
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"intention_ripple_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "per_session.csv"), index=False)

    print("\n=== PRE-UNCOVER INTENTION vs RIPPLES AFTER THE UNCOVER ===")
    print(f"    pre-window [t-{PRE_S[0]}, t-{PRE_S[1]}] s; ripples counted "
          f"[t+{POST_S[0]}, t+{POST_S[1]}] s; windows never overlap\n")
    out, ov = [], {}
    nst = R.groupby(["roi", "phase"]).n_strict.median() if "n_strict" in R \
        else None
    if nst is not None:
        print("  median strict events per session:")
        for (rr, pp_), v in nst.items():
            print(f"      {rr:10s} {pp_:8s} {v:.0f}")
        print()
    print(f"  {'ROI':10s} {'phase':8s} {'measure':30s} {'n':>3s} {'val':>8s} "
          f"{'t':>6s} {'P':>9s}")
    for roiname in ROI_SETS:
        for phname in ("all", "encode", "execute"):
            g = R[(R.roi == roiname) & (R.phase == phname)]
            for c, nm in (("z_intent", "intention: square represented?"),
                          ("z_intent_strict", "  STRICT (not current, not goal)"),
                          ("rho_strict", "  STRICT rho(intention, ripples)"),
                          ("rho_all", "rho(intention, n ripples)"),
                          ("rho_correct", "  ... correct uncovers only"),
                          ("rho_incorrect", "  ... incorrect uncovers only"),
                          ("S_correct", "intention before CORRECT"),
                          ("S_incorrect", "intention before INCORRECT")):
                v = g[c].dropna() if c in g else pd.Series(dtype=float)
                if len(v) < 5:
                    continue
                t, p = stats.ttest_1samp(v, 0)
                star = " *" if p < .05 else ""
                print(f"  {roiname:10s} {phname:8s} {nm:30s} {len(v):3d} "
                      f"{v.mean():+8.4f} {t:+6.2f} {p:9.4g}{star}")
                rec = dict(roi=roiname, phase=phname, measure=nm,
                           n_sessions=int(len(v)), value=float(v.mean()),
                           t=float(t), df=int(len(v) - 1), p=float(p))
                out.append(rec); ov[f"{roiname}|{phname}|{c}"] = rec
            # correct minus incorrect, paired
            gg = g.dropna(subset=["S_correct", "S_incorrect"])
            if len(gg) >= 5:
                dd = gg.S_correct - gg.S_incorrect
                t, p = stats.ttest_1samp(dd, 0)
                star = " *" if p < .05 else ""
                print(f"  {roiname:10s} {phname:8s} "
                      f"{'intention CORRECT - INCORRECT':30s} {len(dd):3d} "
                      f"{dd.mean():+8.4f} {t:+6.2f} {p:9.4g}{star}")
                ov[f"{roiname}|{phname}|S_diff"] = dict(
                    roi=roiname, phase=phname, measure="correct - incorrect",
                    n_sessions=int(len(dd)), value=float(dd.mean()),
                    t=float(t), p=float(p))
        print()
    pd.DataFrame(out).to_csv(os.path.join(d, "summary.csv"), index=False)
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"), indent=2)
    json.dump(dict(seed=SEED, n_perm=N_PERM, roi_sets=ROI_SETS,
                   pre_s=PRE_S, post_s=POST_S, min_events=MIN_EVENTS,
                   note="trial-level Spearman within session, then t across "
                        "sessions; windows never overlap",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"-> {d}")


if __name__ == "__main__":
    main()
