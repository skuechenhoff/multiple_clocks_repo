#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ripples as CONSOLIDATION, not planning: 'this was the relevant square, store it'.

    python scripts/swr_content_explore.py consolidation

Every other analysis here locks to ripples and asks what is represented, or
indexes squares by how far AHEAD they are. SK's reframe: during search there is
no plan to retrieve, so a ripple at a discovery is more plausibly telling PFC
to STORE the square that just paid off. Note the arithmetic that made this easy
to miss -- the task cycle has length 4, so '+3 ahead' IS '1 back': `goal3` was
never a distant-future goal, it was the reward just collected.

DESIGN. Lock to the uncover, then compare windows AFTER it that do or do not
contain a ripple:

    evidence for the JUST-UNCOVERED square, in [t+0.05, t+1.0]
      ... on trials where a ripple occurred in that window
      ... versus trials where none did

If ripples deliver the just-found location, the first should exceed the second.
Splitting within event type holds the behaviour fixed, so this is not the
trivial comparison of discoveries against other moments.

THREE EVENT TYPES, because the consolidation account makes different
predictions for each:
    first_discovery  -- the reward is NEW information. The account predicts the
                        largest ripple-dependent gain here.
    repeat_correct   -- correct, but nothing new to store.
    incorrect        -- nothing to store at all.

⚠ Ripple occurrence is not randomised. Trials with and without a ripple may
differ in arousal, stillness or speed, and those could move the evidence on
their own. `n_rip` and post-event stillness are recorded per session so the
comparison can be re-run matched if anything survives.

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
POST = (0.05, 1.00)       # window after the uncover
MIN_EVENTS = 15
MIN_CELLS = 2


def _stat(E, li, P):
    E = np.nan_to_num(E)
    H = np.take_along_axis(E, P[:, li].T, axis=1)
    return H - (E.sum(axis=1)[:, None] - H) / 8.0


def _z(M):
    sd = M[..., 1:].std(axis=-1)
    return np.where(sd > 0, (M[..., 0] - M[..., 1:].mean(axis=-1)) / sd, np.nan)


def main(keep=None, out_dir=None):
    """`keep`: optional set of (session, cell) -- only these cells enter (e.g.
    the top lag-0 cells, `lag0_cells`); `out_dir` overrides the dated folder."""
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
        if len(unc) < MIN_EVENTS:
            continue
        t_ev = t_all[unc]
        loc_ev = st["loc"].to_numpy(float)[unc].astype(int)
        cu = st.correct_uncover.to_numpy(float)[unc]
        gn = st.grid_num.to_numpy(float)[unc]
        grid_ev = st.grid_id.to_numpy(float)[unc]
        # first discovery of each reward within its configuration
        seen, is_first = set(), np.zeros(len(t_ev), bool)
        for i in np.flatnonzero(cu == 1):
            key = (gn[i], loc_ev[i])
            if key not in seen:
                seen.add(key); is_first[i] = True
        t_rip = np.sort(rip[rip.session == s].t_peak_s.to_numpy())
        n_rip = (np.searchsorted(t_rip, t_ev + POST[1])
                 - np.searchsorted(t_rip, t_ev + POST[0])).astype(float)
        has_rip = n_rip > 0
        ph_i, first_i, _, _ = phase_labels(st, occ)
        o = occ.sort_values("start_s")
        a = o.start_s.to_numpy()
        j = np.clip(np.searchsorted(a, t_ev, side="right") - 1, 0, len(a) - 1)
        ph_ev = ph_i[o.index.to_numpy()[j]]
        grids = np.unique(occ.cv_group.to_numpy())

        EV = {"first_discovery": is_first,
              "repeat_correct": (cu == 1) & ~is_first,
              "incorrect": cu == 0}
        for roiname, members in ROI_SETS.items():
            cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                     if c < len(spk[s]["spikes"])
                     and (keep is None or (int(s), int(c)) in keep)]
            if len(cells) < MIN_CELLS:
                continue
            _, loo = swc.build_templates(spk[s], cells, occ, grids)
            C, = swc.zscore_cells([swc.window_counts(
                spk[s], cells, t_ev + POST[0], t_ev + POST[1])])
            E = swc.score_windows(C, t_ev, grid_ev, loo)
            ok = np.isfinite(E).all(axis=1) & (loc_ev >= 1) & (loc_ev <= 9)
            S = _stat(E, np.clip(loc_ev - 1, 0, 8), P)
            for evname, evm in EV.items():
                for phname, pm in (("all", np.ones(len(t_ev), bool)),
                                   ("encode", ph_ev == "encode"),
                                   ("execute", ph_ev == "execute")):
                    base = ok & evm & pm
                    if base.sum() < MIN_EVENTS:
                        continue
                    rec = dict(session=s, subject=subj, roi=roiname,
                               event=evname, phase=phname,
                               n=int(base.sum()),
                               n_with=int((base & has_rip).sum()),
                               n_without=int((base & ~has_rip).sum()),
                               mean_n_rip=float(n_rip[base].mean()))
                    rec["z_all"] = float(_z(S[base].mean(axis=0)))
                    for lab, mm in (("with", base & has_rip),
                                    ("without", base & ~has_rip)):
                        rec[f"z_{lab}"] = (float(_z(S[mm].mean(axis=0)))
                                           if mm.sum() >= MIN_EVENTS else np.nan)
                    rows.append(rec)
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    R["d_ripple"] = R.z_with - R.z_without
    d = out_dir or os.path.join(deriv, "group", "swr",
                                f"consolidation_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "per_session.csv"), index=False)

    print(f"\n=== IS THE JUST-UNCOVERED SQUARE REPRESENTED AFTER THE EVENT? ===")
    print(f"    window [t+{POST[0]}, t+{POST[1]}] s; split by whether a ripple "
          "occurred in it\n")
    out, ov = [], {}
    print(f"  {'ROI':10s} {'event':16s} {'phase':8s} {'measure':22s} {'n':>3s} "
          f"{'z':>7s} {'t':>6s} {'P':>9s}")
    for roiname in ROI_SETS:
        for evname in EV:
            for phname in ("all", "encode", "execute"):
                g = R[(R.roi == roiname) & (R.event == evname)
                      & (R.phase == phname)]
                for c, nm in (("z_all", "represented at all"),
                              ("z_with", "  trials WITH a ripple"),
                              ("z_without", "  trials WITHOUT"),
                              ("d_ripple", "  WITH minus WITHOUT")):
                    v = g[c].dropna() if c in g else pd.Series(dtype=float)
                    if len(v) < 5:
                        continue
                    t, p = stats.ttest_1samp(v, 0)
                    star = " *" if p < .05 else ""
                    print(f"  {roiname:10s} {evname:16s} {phname:8s} {nm:22s} "
                          f"{len(v):3d} {v.mean():+7.3f} {t:+6.2f} "
                          f"{p:9.4g}{star}")
                    rec = dict(roi=roiname, event=evname, phase=phname,
                               measure=nm, n_sessions=int(len(v)),
                               z=float(v.mean()), t=float(t), p=float(p))
                    out.append(rec)
                    ov[f"{roiname}|{evname}|{phname}|{c}"] = rec
            print()
    pd.DataFrame(out).to_csv(os.path.join(d, "summary.csv"), index=False)
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"), indent=2)
    json.dump(dict(seed=SEED, n_perm=N_PERM, roi_sets=ROI_SETS, post_s=POST,
                   min_events=MIN_EVENTS,
                   cell_filter=None if keep is None else f"{len(keep)} cells",
                   caveat="ripple occurrence is not randomised; arousal and "
                          "stillness are not matched between with/without",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"-> {d}")


if __name__ == "__main__":
    main()
