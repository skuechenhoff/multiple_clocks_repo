#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two tests of the compositional-replay account that this dataset CAN support.

    python scripts/swr_content_explore.py compositional

The proposals (Kurth-Nelson et al. 2023; He et al. 2026; Jensen et al. 2024) are
about SEQUENCES -- a compound is entities strung together, and hippocampal
ripples are supposed to orchestrate cortical reassembly. This dataset cannot
test a sequence WITHIN a ripple: the decoder gate closed on every signal we
have, spikes included (`decoder_comparison`). Two versions of the claim survive
that, because neither needs a within-event decode.

  1  SEQUENCE ACROSS EVENTS. Ripples cluster -- 30.4% have another within
     250 ms. If ripple n over-represents state i and ripple n+1 over-represents
     state i+1, that is an ordered compound spread over events instead of over
     milliseconds.

     ⚠ THE CONFOUND, and the reason for the central restriction. The
     participant's own state advances with time, so consecutive ripples
     naturally carry k then k+1 for a wholly uninteresting reason. Pairs are
     therefore taken ONLY where the participant's ACTUAL state is unchanged
     between the two ripples, and each ripple's evidence vector has its
     state-conditional mean removed first. What is left is fluctuation around
     what the participant is actually doing, so any forward structure cannot be
     the participant advancing.

  2  HIPPOCAMPUS LEADING CORTEX. He et al.'s mechanism is that a hippocampal
     ripple reorganises cortical activity. Testable directly here: does
     hippocampal state evidence AT the ripple predict mPFC state evidence at a
     lag AFTER it, more than the reverse direction and more than at matched
     flanks? 22 sessions carry both populations.

Both tests use the state estimator from `state_content.py` unchanged --
detrended for time-into-traversal, leave-one-configuration-out templates,
per-unit z-scored counts.

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
from mc.analyse.swr_explore.state_content import (
    _time_into_traversal, _templates, _score, _partial_r, _z,
    N_STATE, N_LOC, MIN_CELLS, FLANK_GAP_S, MAX_FLANKS)

SEED = 42
N_PERM = 200
GAPS = (0.25, 0.5, 1.0)          # max inter-ripple interval for a pair
PRIMARY_GAP = 0.25
LAGS = np.round(np.arange(0.0, 0.501, 0.05), 3)
MIN_PAIRS = 20
MIN_RIPPLES = 30
HC = ["HC_anterior", "HC_mid"]
PARTNERS = {"mPFC": ["mPFC"], "mOFC": ["mOFC"]}


def _hyp_matrices():
    """Forward, reverse and same-state hypothesis matrices over ABCD.

    The sequence loops (D is followed by A on the next traversal), so `forward`
    is the cyclic shift. The diagonal is excluded from both directional
    matrices so that simple autocorrelation between neighbouring ripples cannot
    masquerade as a transition.
    """
    fwd = np.zeros((N_STATE, N_STATE))
    rev = np.zeros((N_STATE, N_STATE))
    for i in range(N_STATE):
        fwd[i, (i + 1) % N_STATE] = 1
        rev[i, (i - 1) % N_STATE] = 1
    return {"forward": fwd, "reverse": rev}


HYP = _hyp_matrices()


def _session_state_evidence(s, spk, roi, steps, rip, members):
    """Per-ripple state-evidence matrix, detrended, plus the bookkeeping."""
    occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
    if not len(occ) or not len(r) or len(t_rip) < MIN_RIPPLES:
        return None
    cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
             if c < len(spk[s]["spikes"])]
    if len(cells) < MIN_CELLS:
        return None
    tit, state = _time_into_traversal(steps[steps.session == s], occ)
    if not np.isfinite(state).any():
        return None
    cond_state = np.where(np.isfinite(state), state - 1, -1).astype(int)
    o = occ.sort_values("start_s")
    a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
    j = np.searchsorted(a, t_rip, side="right") - 1
    jj = np.clip(j, 0, len(a) - 1)
    k = o.index.to_numpy()[jj]
    inside = (j >= 0) & (t_rip <= b[jj]) & (cond_state[k] >= 0)
    if inside.sum() < MIN_RIPPLES:
        return None
    t_rip, d_rip, k = t_rip[inside], d_rip[inside], k[inside]
    half = d_rip / 2.0
    grid = occ.cv_group.to_numpy()[k]
    w_state = cond_state[k]
    w_tit = tit[k] + (t_rip - occ.start_s.to_numpy()[k])
    grids = np.unique(occ.cv_group.to_numpy())
    loo = _templates(spk[s], cells, occ, cond_state, N_STATE, grids,
                     detrend_t=tit)
    C = swc.zscore_cells([swc.window_counts(spk[s], cells,
                                            t_rip - half, t_rip + half)])[0]
    E = _score(swc.detrend_on(C, w_tit), grid, loo)
    ok = np.isfinite(E).all(axis=1)
    if ok.sum() < MIN_RIPPLES:
        return None
    return dict(t=t_rip[ok], half=half[ok], E=E[ok], state=w_state[ok],
                grid=grid[ok], tit=w_tit[ok], occ=occ, cells=cells,
                loo=loo, r=r)


def _residual_evidence(E, state):
    """Within-window z, then the state-conditional mean removed.

    Both steps matter. The z makes windows comparable regardless of how many
    spikes they contain; removing the state-conditional mean removes exactly
    the thing the participant is actually doing, leaving only fluctuation.
    """
    Z = (E - E.mean(axis=1, keepdims=True))
    sd = E.std(axis=1, keepdims=True)
    Z = np.divide(Z, sd, out=np.zeros_like(Z), where=sd > 0)
    R = Z.copy()
    for st in np.unique(state):
        m = state == st
        if m.sum() > 1:
            R[m] -= Z[m].mean(axis=0, keepdims=True)
    return R


def test_sequence(spk, roi, steps, rip, sessions, rng):
    rows = []
    for s in sessions:
        if s not in spk:
            continue
        D = _session_state_evidence(s, spk, roi, steps, rip, HC)
        if D is None:
            continue
        R = _residual_evidence(D["E"], D["state"])
        t, stt = D["t"], D["state"]
        order = np.argsort(t)
        t, stt, R = t[order], stt[order], R[order]
        for gap in GAPS:
            dt = np.diff(t)
            pair = np.flatnonzero((np.diff(stt) == 0) & (dt <= gap))
            if len(pair) < MIN_PAIRS:
                continue
            A, B = R[pair], R[pair + 1]

            def mat(idx):
                return A.T @ B[idx] / len(idx)

            ident = np.arange(len(pair))
            obs = mat(ident)
            # null: keep every ripple, destroy only WHICH ripple follows which
            nul = [mat(rng.permutation(len(pair))) for _ in range(N_PERM)]
            row = dict(session=s, gap=gap, n_pairs=int(len(pair)))
            for name, H in HYP.items():
                v = float(np.sum(obs * H))
                row[f"{name}"] = v
                row[f"z_{name}"] = _z(v, np.array(
                    [float(np.sum(n * H)) for n in nul]))
            row["z_fwd_minus_rev"] = row["z_forward"] - row["z_reverse"]
            rows.append(row)
        print(f"  seq s{s:02d}", flush=True)
    return pd.DataFrame(rows)


def test_cross_region(spk, roi, steps, rip, sessions, rng):
    rows = []
    for s in sessions:
        if s not in spk:
            continue
        H = _session_state_evidence(s, spk, roi, steps, rip, HC)
        if H is None:
            continue
        for pname, members in PARTNERS.items():
            cells = [c for c in H["r"][H["r"].roi.isin(members)].cell.to_numpy()
                     if c < len(spk[s]["spikes"])]
            if len(cells) < MIN_CELLS:
                continue
            occ = H["occ"]
            tit, state = _time_into_traversal(steps[steps.session == s], occ)
            cond_state = np.where(np.isfinite(state), state - 1, -1).astype(int)
            grids = np.unique(occ.cv_group.to_numpy())
            loo_p = _templates(spk[s], cells, occ, cond_state, N_STATE, grids,
                               detrend_t=tit)
            hc_score = (H["E"][np.arange(len(H["E"])), H["state"]]
                        - (H["E"].sum(axis=1)
                           - H["E"][np.arange(len(H["E"])), H["state"]]) / 3)
            hc_act = None
            for lag in LAGS:
                for direction in ("hc_leads", "pfc_leads"):
                    sign = 1.0 if direction == "hc_leads" else -1.0
                    c = swc.window_counts(
                        spk[s], cells, H["t"] + sign * lag - H["half"],
                        H["t"] + sign * lag + H["half"])
                    Cz = swc.zscore_cells([c])[0]
                    Ep = _score(swc.detrend_on(Cz, H["tit"]), H["grid"], loo_p)
                    ok = np.isfinite(Ep).all(axis=1)
                    if ok.sum() < MIN_RIPPLES:
                        continue
                    ar = np.arange(ok.sum())
                    tgt = H["state"][ok]
                    p_score = (Ep[ok][ar, tgt]
                               - (Ep[ok].sum(axis=1) - Ep[ok][ar, tgt]) / 3)
                    act = np.abs(Cz[:, ok]).sum(axis=0)
                    rows.append(dict(
                        session=s, partner=pname, lag_s=float(lag),
                        direction=direction, n=int(ok.sum()),
                        r=_partial_r(hc_score[ok], p_score, act)))
        print(f"  xreg s{s:02d}", flush=True)
    return pd.DataFrame(rows)


def _p(v):
    v = pd.Series(v).dropna()
    return stats.ttest_1samp(v, 0)[1] if len(v) >= 5 else np.nan


def report(S, X, d):
    """Print the tables and return the overview. Callable on saved CSVs."""
    ov = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
          "primary_gap_s": PRIMARY_GAP, "n_perm": N_PERM, "lags_s": LAGS.tolist(),
          "restriction": "ripple pairs with UNCHANGED participant state; "
                         "state-conditional mean removed from each evidence "
                         "vector",
          "tests": {}}

    print("\n=== 1. ORDERED STATE STRUCTURE ACROSS CONSECUTIVE RIPPLES ===")
    print("    pairs with the participant's actual state UNCHANGED, so a")
    print("    forward effect cannot be the participant advancing\n")
    print(f"{'gap':>6s} {'sess':>5s} {'pairs':>7s} | {'forward':>8s} {'p':>8s} | "
          f"{'reverse':>8s} {'p':>8s} | {'fwd-rev':>8s} {'p':>8s}")
    seq = {}
    for gap in GAPS:
        g = S[S.gap == gap]
        if len(g) < 5:
            continue
        print(f"{gap:6.2f} {len(g):5d} {int(g.n_pairs.sum()):7d} | "
              f"{g.z_forward.mean():+8.3f} {_p(g.z_forward):8.3g} | "
              f"{g.z_reverse.mean():+8.3f} {_p(g.z_reverse):8.3g} | "
              f"{g.z_fwd_minus_rev.mean():+8.3f} {_p(g.z_fwd_minus_rev):8.3g}")
        v = g.z_fwd_minus_rev.dropna()
        ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(),
                              scale=v.std(ddof=1) / np.sqrt(len(v)))
        seq[f"gap_{gap}"] = dict(
            n_sessions=int(len(g)), n_pairs=int(g.n_pairs.sum()),
            z_forward=float(g.z_forward.mean()), p_forward=float(_p(g.z_forward)),
            z_reverse=float(g.z_reverse.mean()), p_reverse=float(_p(g.z_reverse)),
            z_forward_minus_reverse=float(v.mean()), p_diff=float(_p(v)),
            ci95_diff=[float(ci[0]), float(ci[1])])
    ov["tests"]["sequence_across_ripples"] = seq

    print("\n=== 2. DOES HIPPOCAMPUS LEAD CORTEX? ===")
    print("    partial correlation of HC state evidence at the ripple with the")
    print("    partner region's state evidence at a lag, activity removed\n")
    xr = {}
    for pname in PARTNERS:
        g = X[X.partner == pname]
        if not len(g):
            continue
        print(f"  {pname}  ({g.session.nunique()} sessions)")
        print(f"    {'lag':>7s} {'HC leads':>10s} {'p':>8s} | "
              f"{pname + ' leads':>12s} {'p':>8s}")
        for lag in LAGS:
            a = g[(g.lag_s == lag) & (g.direction == "hc_leads")].r
            b = g[(g.lag_s == lag) & (g.direction == "pfc_leads")].r
            if len(a) < 5:
                continue
            print(f"    {lag * 1000:6.0f}ms {a.mean():+10.4f} {_p(a):8.3g} | "
                  f"{b.mean():+12.4f} {_p(b):8.3g}")
            xr[f"{pname}_lag{int(lag * 1000)}"] = dict(
                n_sessions=int(len(a)), r_hc_leads=float(a.mean()),
                p_hc_leads=float(_p(a)), r_partner_leads=float(b.mean()),
                p_partner_leads=float(_p(b)))
    ov["tests"]["cross_region"] = xr


    # THE directional test: HC-leads minus partner-leads at each lag. The raw
    # correlations are not the evidence -- two regions both tracking state
    # correlate in BOTH directions. Only the asymmetry speaks to who leads.
    print("\n    directional asymmetry (HC leads MINUS partner leads):")
    asym = {}
    for pname in PARTNERS:
        g = X[X.partner == pname]
        if not len(g):
            continue
        print(f"      {pname}: ", end="")
        hits = []
        for lag in sorted(g.lag_s.unique()):
            if lag == 0:                      # identical windows by definition
                continue
            aa = g[(g.lag_s == lag) & (g.direction == "hc_leads")
                   ].set_index("session").r
            bb = g[(g.lag_s == lag) & (g.direction == "pfc_leads")
                   ].set_index("session").r
            j = pd.concat([aa.rename("a"), bb.rename("b")], axis=1).dropna()
            if len(j) < 5:
                continue
            dif = j.a - j.b
            pv = stats.ttest_1samp(dif, 0)[1]
            ci = stats.t.interval(0.95, len(dif) - 1, loc=dif.mean(),
                                  scale=dif.std(ddof=1) / np.sqrt(len(dif)))
            asym[f"{pname}_lag{int(lag * 1000)}"] = dict(
                n_sessions=int(len(dif)), asymmetry=float(dif.mean()),
                p=float(pv), ci95=[float(ci[0]), float(ci[1])])
            if pv < .05:
                hits.append(f"{int(lag * 1000)}ms {dif.mean():+.4f} p={pv:.3g}")
        print(", ".join(hits) if hits else "nothing below p = 0.05")
    ov["tests"]["cross_region_asymmetry"] = asym
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"), indent=2)
    return ov


def main():
    rng = np.random.default_rng(SEED)
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)

    S = test_sequence(spk, roi, steps, rip, sessions, rng)
    X = test_cross_region(spk, roi, steps, rip, sessions, rng)

    d = os.path.join(deriv, "group", "swr",
                     f"ripple_compositional_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    S.to_csv(os.path.join(d, "sequence_per_session.csv"), index=False)
    X.to_csv(os.path.join(d, "cross_region_per_session.csv"), index=False)

    ov = report(S, X, d)
    json.dump(dict(seed=SEED, n_perm=N_PERM, gaps=list(GAPS),
                   lags_s=LAGS.tolist(), min_pairs=MIN_PAIRS,
                   hypotheses="cyclic forward and reverse over ABCD, diagonal "
                              "excluded",
                   null="pairing shuffled: every ripple kept, only WHICH "
                        "ripple follows which is destroyed",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"\n-> {d}")


if __name__ == "__main__":
    main()
