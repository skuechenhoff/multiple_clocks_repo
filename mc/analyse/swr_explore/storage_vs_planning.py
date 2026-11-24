#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WHEN does a cell fire -- at storage, or at planning? By its lag tuning.

    python scripts/swr_content_explore.py storage_planning

Not "does a cell fire at its square" (that is `preferred_lag_firing`). SK's
actual question: a cell tuned to lag 0 codes the PRESENT, so it should be
engaged when something is being stored; a cell tuned to a future lag codes the
PLAN, so it should be engaged when the plan is being used.

    storage       = the first discovery of a reward (new information arrives)
    planning      = ANY subgoal onset during execute
    planning_first= the FIRST subgoal onset of each traversal during execute

The two planning definitions matter. A plan plausibly has to be ASSEMBLED only
once, at the start of a traversal; the remaining three subgoal onsets are then
its execution. Pooling all four would dilute an assembly signal with three
parts motor follow-through, which is the most likely reason a null here would
be uninformative rather than evidence against the model.

Storage only exists during encode and planning only during execute, so the two
are measured against their OWN phase baseline:

    modulation = (rate in the event window - rate in that phase's baseline)
                 / (sum)

which removes any phase-level difference in overall rate before the two are
compared.

CONTINUOUS PREDICTORS, not an argmax. A per-cell argmax over 12 correlated
values is near-uniformly distributed (64-94 cells per bin, 80 by chance) and
throws away ~94% of the cells to form the group of interest. Instead each cell
carries two numbers straight from the per-lag file:

    lag0_strength   = r_lag000_noctrl
    future_strength = mean of r_lag090 .. r_lag270   (the far half of the circle)

PREDICTION. If the tuning means what the model says, `lag0_strength` should
predict a storage preference and `future_strength` a planning preference, so
the two correlations with (storage - planning) should have OPPOSITE signs.
That double prediction is much harder to satisfy by chance than either alone.

@author: Svenja Kuchenhoff
"""

import os
import json
import datetime

import numpy as np
import pandas as pd

from mc.analyse.roi_labels import read_per_cell_table
from scipy import stats

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc
from mc.analyse.swr_content import phase_labels
from mc.analyse.swr_explore.preferred_lag_firing import PER_CELL

SEED = 42
ROI_GROUPS = {"HC": ["HC_anterior", "HC_mid"], "mPFC": ["mPFC"],
              "mOFC": ["mOFC"]}
EVENT_WIN = (0.05, 1.00)       # after the event
# Ripple-dependent split of the SAME events. Everything else in this module is
# ripple-blind; these ask whether the lag-tuning relationship exists only when
# a ripple is involved.
RIP_PAD_S = 0.5                # 'near a ripple' = within this of a ripple peak
FUTURE_LAGS = [90, 120, 150, 180, 210, 240, 270]
MIN_EVENTS = 8
MIN_SPIKES = 10


def lag_strengths():
    """Per cell: lag-0 encoding strength and mean far-future strength."""
    d = read_per_cell_table(PER_CELL).dropna(subset=["cell_idx", "subject_id"])
    fut = [f"r_lag{l:03d}_noctrl" for l in FUTURE_LAGS]
    return pd.DataFrame(dict(
        session=d.subject_id.astype(int).to_numpy(),
        cell=(d.cell_idx.astype(int) - 1).to_numpy(),
        lag0_strength=d["r_lag000_noctrl"].to_numpy(float),
        future_strength=d[fut].mean(axis=1).to_numpy(float)))


def _rate(stimes, lo, hi):
    if not len(lo):
        return np.nan, 0.0
    n = swc.count_in(stimes, lo, hi).sum()
    t = float(np.sum(hi - lo))
    return (n / t if t > 0 else np.nan), t


def main():
    LS = lag_strengths()
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])
    sessions = sorted(set(rip.session.unique()) & set(LS.session.unique()))
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
        ph_i, _, _, _ = phase_labels(st, occ)
        t_all = st.t_s.to_numpy(float)
        loc = st["loc"].to_numpy(float)
        gn = st.grid_num.to_numpy(float)
        cu = st.correct_uncover.to_numpy(float)
        state = st.state.to_numpy(float)

        # STORAGE: first correct uncover of each reward in its configuration
        seen, t_store = set(), []
        for i in np.flatnonzero(cu == 1):
            k = (gn[i], loc[i])
            if k not in seen:
                seen.add(k); t_store.append(t_all[i])
        t_store = np.asarray(t_store)
        # PLANNING: subgoal onsets during EXECUTE (state changes)
        j = np.flatnonzero(np.diff(state) != 0) + 1
        o = occ.sort_values("start_s")
        a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
        def _phase_of(t):
            k = np.clip(np.searchsorted(a, t, side="right") - 1, 0, len(a) - 1)
            return ph_i[o.index.to_numpy()[k]]
        t_plan = t_all[j]
        t_plan = t_plan[_phase_of(t_plan) == "execute"]
        # FIRST subgoal onset of each traversal: where a plan would be
        # assembled rather than followed
        tnum = st.trial_num_in_grid.to_numpy(float)
        gnum = st.grid_num.to_numpy(float)
        first_rows = []
        for key in pd.unique(list(zip(gnum, tnum))):
            m = (gnum == key[0]) & (tnum == key[1])
            if m.any():
                first_rows.append(t_all[np.flatnonzero(m)[0]])
        t_plan_first = np.asarray(sorted(first_rows))
        if len(t_plan_first):
            t_plan_first = t_plan_first[_phase_of(t_plan_first) == "execute"]
        t_store = t_store[_phase_of(t_store) == "encode"]
        if len(t_store) < MIN_EVENTS or len(t_plan) < MIN_EVENTS:
            continue
        # --- ripple-dependent splits of the SAME events
        t_rip_all = np.sort(rip[rip.session == s].t_peak_s.to_numpy())

        def _near(t):
            if not len(t_rip_all) or not len(t):
                return np.zeros(len(t), bool)
            i = np.searchsorted(t_rip_all, t)
            lo = np.where(i > 0, t - t_rip_all[np.clip(i - 1, 0, None)], np.inf)
            hi = np.where(i < len(t_rip_all),
                          t_rip_all[np.clip(i, None, len(t_rip_all) - 1)] - t,
                          np.inf)
            return np.minimum(lo, hi) <= RIP_PAD_S

        ns = _near(t_store); npl = _near(t_plan)
        t_store_r, t_store_n = t_store[ns], t_store[~ns]
        t_plan_r, t_plan_n = t_plan[npl], t_plan[~npl]
        # ripples themselves, for a pre/post contrast
        t_pre = t_post = t_rip_all[(t_rip_all > a.min()) & (t_rip_all < b.max())]
        # phase baselines: all task time in that phase
        base = {ph: (a[ph_i[o.index.to_numpy()] == ph],
                     b[ph_i[o.index.to_numpy()] == ph]) for ph in
                ("encode", "execute")}

        for _, cr in LS[LS.session == s].iterrows():
            c = int(cr.cell)
            if c >= len(spk[s]["spikes"]):
                continue
            rr = r[r.cell == c]
            if not len(rr):
                continue
            grp = next((g for g, mem in ROI_GROUPS.items()
                        if rr.roi.iloc[0] in mem), None)
            if grp is None:
                continue
            stimes = spk[s]["spikes"][c]
            if len(stimes) < MIN_SPIKES:
                continue
            rec = dict(session=s, cell=c, roi=grp,
                       lag0_strength=float(cr.lag0_strength),
                       future_strength=float(cr.future_strength))
            ok = True
            for nm, tev, ph in (("storage", t_store, "encode"),
                                ("planning", t_plan, "execute"),
                                ("planning_first", t_plan_first, "execute"),
                                ("storage_rip", t_store_r, "encode"),
                                ("storage_norip", t_store_n, "encode"),
                                ("planning_rip", t_plan_r, "execute"),
                                ("planning_norip", t_plan_n, "execute"),
                                ("pre_ripple", t_pre, None),
                                ("post_ripple", t_post, None)):
                if tev is None or len(tev) < MIN_EVENTS:
                    rec[f"mod_{nm}"] = np.nan
                    continue
                if ph is None:
                    ph = "encode"
                if nm == "pre_ripple":
                    re_, _ = _rate(stimes, tev - 0.30, tev - 0.05)
                elif nm == "post_ripple":
                    re_, _ = _rate(stimes, tev + 0.05, tev + 0.30)
                else:
                    re_, _ = _rate(stimes, tev + EVENT_WIN[0],
                                   tev + EVENT_WIN[1])
                rb, _ = _rate(stimes, *base[ph])
                if not np.isfinite(re_) or not np.isfinite(rb) or re_ + rb <= 0:
                    rec[f"mod_{nm}"] = np.nan
                    if nm in ("storage", "planning"):
                        ok = False
                    continue
                rec[f"mod_{nm}"] = (re_ - rb) / (re_ + rb)
                rec[f"n_{nm}"] = len(tev)
            if ok:
                rec["store_minus_plan"] = rec["mod_storage"] - rec["mod_planning"]
                pf = rec.get("mod_planning_first")
                rec["store_minus_planfirst"] = (
                    rec["mod_storage"] - pf
                    if pf is not None and np.isfinite(pf) else np.nan)
                for nm in ("storage", "planning"):
                    a_, b_ = rec.get(f"mod_{nm}_rip"), rec.get(f"mod_{nm}_norip")
                    rec[f"{nm}_rip_minus_norip"] = (
                        a_ - b_ if a_ is not None and b_ is not None
                        and np.isfinite(a_) and np.isfinite(b_) else np.nan)
                pr, po = rec.get("mod_pre_ripple"), rec.get("mod_post_ripple")
                rec["ripple_post_minus_pre"] = (
                    po - pr if pr is not None and po is not None
                    and np.isfinite(pr) and np.isfinite(po) else np.nan)
                rows.append(rec)
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"storage_vs_planning_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "per_cell.csv"), index=False)

    print("\n=== DOES LAG TUNING PREDICT *WHEN* A CELL FIRES? ===")
    print("    storage  = first discovery of a reward (encode)")
    print("    planning = subgoal onset during execute")
    print("    each modulated against its OWN phase baseline\n")
    out, ov = [], {}
    print(f"  {'ROI':6s} {'predictor':16s} {'measure':22s} {'n':>4s} "
          f"{'r':>8s} {'P':>10s}")
    for grp in ROI_GROUPS:
        g = R[R.roi == grp]
        if len(g) < 20:
            continue
        for pred in ("lag0_strength", "future_strength"):
            for meas in ("mod_storage", "mod_planning", "mod_planning_first",
                         "store_minus_plan", "store_minus_planfirst",
                         "storage_rip_minus_norip",
                         "planning_rip_minus_norip",
                         "ripple_post_minus_pre"):
                gg = g.dropna(subset=[pred, meas])
                if len(gg) < 20:
                    continue
                rr_, p = stats.pearsonr(gg[pred], gg[meas])
                star = " *" if p < .05 else ""
                print(f"  {grp:6s} {pred:16s} {meas:22s} {len(gg):4d} "
                      f"{rr_:+8.3f} {p:10.4g}{star}")
                rec = dict(roi=grp, predictor=pred, measure=meas,
                           n_cells=int(len(gg)), r=float(rr_), p=float(p))
                out.append(rec); ov[f"{grp}|{pred}|{meas}"] = rec
        # the double prediction: opposite signs on store_minus_plan
        for meas, lab in (("store_minus_plan", "all subgoal onsets"),
                          ("store_minus_planfirst", "FIRST onset only")):
            gg = g.dropna(subset=["lag0_strength", "future_strength", meas])
            if len(gg) < 20:
                continue
            r0, p0 = stats.pearsonr(gg.lag0_strength, gg[meas])
            rf, pf_ = stats.pearsonr(gg.future_strength, gg[meas])
            print(f"  {grp:6s} {'--> opposite signs?':16s} {lab:22s} "
                  f"lag0 {r0:+.3f} (P={p0:.3f}) vs future {rf:+.3f} "
                  f"(P={pf_:.3f}) : {'YES' if r0 * rf < 0 else 'no'}")
            ov[f"{grp}|double|{meas}"] = dict(
                roi=grp, measure=meas, r_lag0=float(r0), p_lag0=float(p0),
                r_future=float(rf), p_future=float(pf_),
                opposite=bool(r0 * rf < 0))
        print()
    pd.DataFrame(out).to_csv(os.path.join(d, "summary.csv"), index=False)
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"), indent=2)
    json.dump(dict(seed=SEED, roi_groups=ROI_GROUPS, event_win=EVENT_WIN,
                   future_lags=FUTURE_LAGS, min_events=MIN_EVENTS,
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"-> {d}")


if __name__ == "__main__":
    main()
