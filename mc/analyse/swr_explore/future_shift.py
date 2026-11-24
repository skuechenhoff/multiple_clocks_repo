#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FUTURE SHIFT: across a hippocampal ripple, does frontal activity move from the
square the subject is ON towards the squares they are ABOUT TO VISIT?

    python scripts/swr_content_explore.py future_shift

WHY NOT THE USUAL TEMPLATE. Every earlier content test scored a window against
a template built from OTHER configurations. If mPFC codes 'now' early in a block
and a lagged future later, that template averages two codes -- and mPFC place
maps do not transfer across configurations at all (CHANGELOG 2026-10-02,
r = -0.002). Here every reference comes from the SAME BLOCK:

    R(L) = mean activity of each feature while the subject stood on square L,
           over the OTHER traversals of this block, z-scored across squares

R(L) is 'what this population does when standing on L, in this block, now'. It
assumes nothing about which lag is coded: if mPFC is lagged, R(L) is lagged
too, so moving towards R(future square) means the state jumped ahead along the
route, whatever the code.

    E_w(L) = x_w . R(L)              x_w = z-scored window (counts or mean HFB)
    I = delta(FUT - NOW)             post minus pre: more future than current?
    J = delta(FUT - PAST)            towards the future, not just away from now

FUT / PAST = mean over the next / previous three distinct squares of the
block's visit sequence. Both windows sit on ONE square (inside the ripple's own
occupancy interval), so the subject neither moves nor presses in between.

THE CONTROL IS THE TEST. Even without ripples, a still period drifts towards
the next move, so post looks more 'future' than pre at every moment. Each
ripple is therefore compared with every ordinary still moment (tiled every
50 ms, same geometry, no ripple near) in its stratum: phase x reward square x
time since arrival x time to departure. The same label permutation is applied
to both terms, so observed and null go through one function.

Pre-registered in CHANGELOG.md, 2026-10-07 (FS).

@author: Svenja Kuchenhoff
"""

import os
import json
import pickle
import datetime

import numpy as np
import pandas as pd

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc
import mc.analyse.swr_bundle as sb
import mc.analyse.plan_reinstatement as pr
from mc.analyse.swr_explore.goal_lag import goal_lag_locations

SEED = 42
N_PERM = 1000
N_FLIP = 10000
N_MATCH = 1000
WIN_S = 0.25              # each window
INNER_S = 0.10            # ripple peak -> inner edge of each window
MAX_RIP_DUR_S = 0.15      # longer ripples would reach into the windows
OTHER_RIP_PAD_S = 0.05    # no other ripple this close to a ripple's windows
CTRL_STEP_S = 0.05        # control events tile still time at this step
CTRL_RIP_PAD_S = 0.25     # no ripple this close to a control's windows
MIN_DWELL_S = 0.2         # a square counts as visited in a reference
MIN_CLEAN_SAMPLES = 20    # HFB: clean samples for a square's reference value
MIN_EVENTS = 20
MIN_FEATURES = 2
PHASES = ("first_traversal", "encode_rest", "execute")
BIN_EDGES = np.array([0.35, 0.5, 0.75, 1.0, 1.5, 2.5, np.inf])
N_STRATA = len(PHASES) * 2 * (len(BIN_EDGES) - 1) ** 2   # x reward square
SPIKE_ROIS = {"mPFC": ["mPFC"], "mOFC": ["mOFC"],
              "HC_all": ["HC_anterior", "HC_mid"]}
HFB_ROIS = ("MedialFrontal", "Visual", "HPC")
REF_SCHEMES = {"block": None, "near4": 4}
ROLES = ("past3", "past2", "past1", "now", "fut1", "fut2", "fut3", "goal0")
QUANTITIES = ROLES + ("FUT", "PAST", "I", "J")
MATCHED = ("now", "FUT", "PAST", "I", "J")   # count-matched across phases
PRIMARY = {"spikes": "mPFC", "hfb": "MedialFrontal"}
ALPHA_PRIMARY = 0.025     # Bonferroni over the two primaries


# --------------------------------------------------------------------- task
def role_table(steps_s, occ):
    """Per occupancy interval: block, traversal, phase, reward square, and the
    square playing each role, from the block's own visit sequence.

    A visit is a run of steps on one square; the sequence runs on across
    traversals within a block but never across blocks. -1 = undefined, which
    includes any role that is the current square itself.
    """
    st = steps_s.sort_values("t_s").reset_index(drop=True)
    k = occ.step_idx.to_numpy()
    loc = st["loc"].to_numpy(float).astype(int)
    block = st.grid_num.to_numpy(float)
    fut = np.full((len(st), 3), -1)
    past = np.full((len(st), 3), -1)
    for gn in np.unique(block):
        rows = np.flatnonzero(block == gn)
        l = loc[rows]
        new = np.r_[True, l[1:] != l[:-1]]
        vid = np.cumsum(new) - 1
        vloc = l[new]
        for j in (1, 2, 3):
            ok = vid + j < len(vloc)
            fut[rows[ok], j - 1] = vloc[vid[ok] + j]
            ok = vid - j >= 0
            past[rows[ok], j - 1] = vloc[vid[ok] - j]
    ph, first, is_rew, _ = swc.phase_labels(st, occ)
    phase3 = np.where(first, "first_traversal",
                      np.where(ph == "encode", "encode_rest",
                               np.where(ph == "execute", "execute", "")))
    R = pd.DataFrame({"block": block[k],
                      "trial": st.trial_num_in_grid.to_numpy(float)[k],
                      "phase": phase3, "is_rew": is_rew.astype(int),
                      "now": occ["loc"].to_numpy().astype(int),
                      "goal0": goal_lag_locations(st, occ)[:, 0]})
    R["key"] = R.block * 1000 + R.trial      # one traversal of one block
    for j in (1, 2, 3):
        R[f"fut{j}"] = fut[k, j - 1]
        R[f"past{j}"] = past[k, j - 1]
    for c in ROLES[:3] + ROLES[4:]:
        R.loc[R[c] == R["now"], c] = -1
    return R


def place_events(t, occ, t_avoid, pad):
    """Occupancy interval of each event, whether both windows fit on that one
    square, how many ripples in `t_avoid` lie within `pad` of the windows, and
    time since arrival / time to departure."""
    a, b = occ.start_s.to_numpy(), occ.stop_s.to_numpy()
    j = np.searchsorted(a, t, side="right") - 1
    jj = np.clip(j, 0, len(a) - 1)
    span = INNER_S + WIN_S
    fits = (j >= 0) & (t - span >= a[jj]) & (t + span <= b[jj])
    av = np.sort(np.asarray(t_avoid, float))
    n_near = (np.searchsorted(av, t + span + pad, side="right")
              - np.searchsorted(av, t - span - pad, side="left"))
    return jj, fits, n_near, t - a[jj], b[jj] - t


def control_times(occ):
    """Every still moment with room for both windows, every CTRL_STEP_S.
    Deterministic: tiling, not sampling, so nothing moves with a seed."""
    span = INNER_S + WIN_S
    a = occ.start_s.to_numpy() + span
    b = occ.stop_s.to_numpy() - span
    out = [np.arange(x, y + 1e-9, CTRL_STEP_S) for x, y in zip(a, b) if y >= x]
    return np.concatenate(out) if out else np.empty(0)


def stratum(phase, is_rew, tsa, ttd):
    """phase x reward square x time since arrival x time to departure."""
    nb = len(BIN_EDGES) - 1
    p = np.array([PHASES.index(x) for x in phase])
    bt = np.clip(np.searchsorted(BIN_EDGES, tsa, side="right") - 1, 0, nb - 1)
    bd = np.clip(np.searchsorted(BIN_EDGES, ttd, side="right") - 1, 0, nb - 1)
    return ((p * 2 + np.asarray(is_rew)) * nb + bt) * nb + bd


def session_events(steps, s, t_rip, d_rip):
    """Ripple and control events of one session, with roles and strata."""
    occ = swc.occupancy(steps, s)
    if not len(occ):
        return None
    RT = role_table(steps[steps.session == s], occ)
    out = {"occ": occ, "roles": RT}
    for kind in ("ripple", "control"):
        if kind == "ripple":
            t = t_rip[d_rip <= MAX_RIP_DUR_S]
            jj, fits, n_near, tsa, ttd = place_events(t, occ, t_rip,
                                                      OTHER_RIP_PAD_S)
            ok = fits & (n_near == 1)            # only the ripple itself
        else:
            t = control_times(occ)
            jj, fits, n_near, tsa, ttd = place_events(t, occ, t_rip,
                                                      CTRL_RIP_PAD_S)
            ok = fits & (n_near == 0)
        ok &= RT.phase.to_numpy()[jj] != ""
        ev = RT.iloc[jj[ok]].reset_index(drop=True)
        ev["t"] = t[ok]
        ev["occ_idx"] = jj[ok]
        ev["stratum"] = stratum(ev.phase, ev.is_rew, tsa[ok], ttd[ok])
        out[kind] = ev
    return out


# ----------------------------------------------------------------- features
def spike_signal(spk_s, cells):
    """(interval sums, window values) for spikes: counts, with dwell as the
    exposure of every cell."""
    def sums(occ):
        a, b = occ.start_s.to_numpy(), occ.stop_s.to_numpy()
        C = swc.window_counts(spk_s, cells, a, b)
        return C, np.broadcast_to(b - a, C.shape)

    def windows(t0, t1):
        return swc.window_counts(spk_s, cells, t0, t1)
    return sums, windows, MIN_DWELL_S


def hfb_signal(store, pairs, clean_iv):
    """(interval sums, window values) for HFB: clean samples only. A window
    is NaN unless every sample in it is clean, as in `plan_reinstatement`."""
    fs = store["out_fs"]
    X = store["hfb"]
    row = {p: i for i, p in enumerate(store["pair_ids"])}
    n = X.shape[1]
    CX, CC = [], []
    for p in pairs:
        clean = np.zeros(n)
        for a, b in clean_iv.get(p, []):
            clean[max(0, int(a * fs)):min(n, int(b * fs))] = 1
        CX.append(np.r_[0, np.cumsum(X[row[p]] * clean, dtype=np.float64)])
        CC.append(np.r_[0, np.cumsum(clean)])
    CX, CC = np.array(CX), np.array(CC)

    def _idx(t0, t1):
        i0 = np.clip(np.round(np.asarray(t0) * fs).astype(int), 0, n)
        i1 = np.clip(np.round(np.asarray(t1) * fs).astype(int), 0, n)
        return i0, i1

    def sums(occ):
        i0, i1 = _idx(occ.start_s.to_numpy(), occ.stop_s.to_numpy())
        return CX[:, i1] - CX[:, i0], CC[:, i1] - CC[:, i0]

    def windows(t0, t1):
        i0, i1 = _idx(t0, t1)
        k = CC[:, i1] - CC[:, i0]
        v = (CX[:, i1] - CX[:, i0]) / np.maximum(k, 1)
        return np.where((k == (i1 - i0)) & (i1 > i0), v, np.nan)
    return sums, windows, MIN_CLEAN_SAMPLES


def zscore_features(sets):
    """z per feature, pooled over every window of every event; NaN -> 0, so an
    artifact window abstains exactly as a silent cell would."""
    pooled = np.concatenate(sets, axis=1)
    mu = np.nanmean(pooled, axis=1, keepdims=True)
    sd = np.nanstd(pooled, axis=1, keepdims=True)
    sd = np.where(sd > 0, sd, np.inf)
    return [np.nan_to_num((x - mu) / sd) for x in sets]


# --------------------------------------------------------------- references
def references(S, X, occ, roles, scheme, min_expo):
    """{traversal key: (T (F, 9), visited (9,))}, built from OTHER traversals
    of the same block -- all of them (`None`) or the `scheme` nearest."""
    dwell = (occ.stop_s - occ.start_s).to_numpy()
    uk, tau = np.unique(roles.key.to_numpy(), return_inverse=True)
    blk_of = np.array([roles.block.to_numpy()[tau == i][0]
                       for i in range(len(uk))])
    tr_of = np.array([roles.trial.to_numpy()[tau == i][0]
                      for i in range(len(uk))])
    L = roles.now.to_numpy() - 1
    F = S.shape[0]
    St = np.zeros((len(uk), F, 9)); Xt = np.zeros((len(uk), F, 9))
    Dt = np.zeros((len(uk), 9))
    np.add.at(St, (tau, slice(None), L), S.T)
    np.add.at(Xt, (tau, slice(None), L), X.T)
    np.add.at(Dt, (tau, L), dwell)
    out = {}
    for i in range(len(uk)):
        same = [j for j in np.flatnonzero(blk_of == blk_of[i]) if j != i]
        if scheme is not None:
            same = sorted(same, key=lambda j: (abs(tr_of[j] - tr_of[i]),
                                               tr_of[j]))[:scheme]
        if not same:
            continue
        s_, x_, d_ = St[same].sum(0), Xt[same].sum(0), Dt[same].sum(0)
        vis = d_ >= MIN_DWELL_S
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where((x_ >= min_expo) & vis[None, :], s_ / x_, np.nan)
        T = np.zeros((F, 9))
        for f in range(F):
            ok = np.isfinite(rate[f])
            if ok.sum() < 3 or np.std(rate[f, ok]) == 0:
                continue                      # this feature abstains
            z = (rate[f, ok] - rate[f, ok].mean()) / rate[f, ok].std()
            T[f, ok] = z - z.mean()
        out[uk[i]] = (T, vis)
    return out


def evidence(Xw, ev, refs):
    """E (n, 9), visited (n, 9), and the reference used (n, F, 9)."""
    n, F = len(ev), Xw.shape[0]
    E = np.full((n, 9), np.nan)
    V = np.zeros((n, 9), bool)
    T_all = np.zeros((n, F, 9))
    for k, m in ev.groupby("key").indices.items():
        if k not in refs:
            continue
        T, vis = refs[k]
        E[m] = Xw[:, m].T @ T
        V[m] = vis
        T_all[m] = T
    return E, V, T_all


def quantity_weights(ev, V):
    """{q: (weights over the 9 squares (n, 9), defined (n,))}."""
    n = len(ev)
    one = {}
    for r in ROLES:
        L = ev[r].to_numpy().astype(int)
        ok = (L >= 1) & np.isfinite(ev[r].to_numpy(float))
        ok[ok] = V[np.flatnonzero(ok), L[ok] - 1]
        W = np.zeros((n, 9))
        W[np.flatnonzero(ok), L[ok] - 1] = 1.0
        one[r] = (W, ok)
    out = dict(one)
    for name, rs in (("FUT", ("fut1", "fut2", "fut3")),
                     ("PAST", ("past1", "past2", "past3"))):
        W = sum(one[r][0] for r in rs)
        k = sum(one[r][1].astype(float) for r in rs)
        out[name] = (W / np.maximum(k, 1)[:, None], k > 0)
    out["I"] = (out["FUT"][0] - one["now"][0], out["FUT"][1] & one["now"][1])
    out["J"] = (out["FUT"][0] - out["PAST"][0],
                out["FUT"][1] & out["PAST"][1])
    return out


# ---------------------------------------------------------------- statistic
def _z(M):
    sd = M[..., 1:].std(axis=-1)
    return np.where(sd > 0, (M[..., 0] - M[..., 1:].mean(axis=-1)) / sd,
                    np.nan)


def permuted(W, G):
    """sum_l W[e, l] * E_e[perm(l)] for every event and permutation.
    G = E[:, P] of shape (n, N_PERM + 1, 9); row 0 of P is the identity."""
    return np.einsum("el,epl->ep", W, G)


def stratum_means(W, X, strat, ok, n_strata, P):
    """Per stratum, the permuted mean of sum_l W_l X[perm(l)] over its events.
    Linear in the events, so it is aggregated as a 9 x 9 outer product first
    -- the control set is too large for an (events x permutations) array."""
    M = np.zeros((n_strata, 9, 9))
    cnt = np.zeros(n_strata)
    e = np.flatnonzero(ok)
    np.add.at(M, strat[e], W[e, :, None] * X[e, None, :])
    np.add.at(cnt, strat[e], 1.0)
    Mbar = M / np.maximum(cnt, 1)[:, None, None]
    return Mbar[:, np.arange(9)[None, :], P].sum(-1), cnt


def roi_stats(ev, Xr_pre, Xr_post, Xc_pre, Xc_post, refs, P, seed_key):
    """Everything for one session x ROI x reference scheme."""
    rip, ctl = ev["ripple"], ev["control"]
    Er_pre, Vr, Tr = evidence(Xr_pre, rip, refs)
    Er_post, _, _ = evidence(Xr_post, rip, refs)
    Ec_pre, Vc, Tc = evidence(Xc_pre, ctl, refs)
    Ec_post, _, _ = evidence(Xc_post, ctl, refs)
    hr = np.isfinite(Er_pre).all(1)
    hc = np.isfinite(Ec_pre).all(1)
    dEr = np.nan_to_num(Er_post - Er_pre)
    dEc = np.nan_to_num(Ec_post - Ec_pre)
    mid_c = np.nan_to_num((Ec_pre + Ec_post) / 2)
    Gr = dEr[:, P]                                   # (n_r, N_PERM + 1, 9)
    n_str = N_STRATA
    sr, sc = rip.stratum.to_numpy(), ctl.stratum.to_numpy()
    wr, wc = quantity_weights(rip, Vr), quantity_weights(ctl, Vc)
    recs = {}
    for qi, q in enumerate(QUANTITIES):
        W_r, ok_r = wr[q]
        W_c, ok_c = wc[q]
        ok_r = ok_r & hr
        ok_c = ok_c & hc
        Cs, cnt = stratum_means(W_c, dEc, sc, ok_c, n_str, P)
        valid = ok_r & (cnt[sr] > 0)
        Rp = permuted(W_r[valid], Gr[valid])          # ripple, per event
        Cp = Cs[sr[valid]]                            # its matched control
        Vd = Rp - Cp
        ph_r = rip.phase.to_numpy()[valid]
        for p_ in ("all",) + PHASES:
            m = np.ones(len(Vd), bool) if p_ == "all" else ph_r == p_
            rec = dict(n_rip=int(m.sum()))
            if m.sum() >= MIN_EVENTS:
                rec.update(z_diff=float(_z(Vd[m].mean(0))),
                           z_rip=float(_z(Rp[m].mean(0))),
                           z_ctrl=float(_z(Cp[m].mean(0))),
                           raw_rip=float(Rp[m, 0].mean()),
                           raw_ctrl=float(Cp[m, 0].mean()))
            # positive control: level in ordinary still moments
            mc_ = ok_c if p_ == "all" else ok_c & (ctl.phase.to_numpy() == p_)
            rec["n_ctrl"] = int(mc_.sum())
            if mc_.sum() >= MIN_EVENTS:
                Lv, _ = stratum_means(W_c, mid_c, np.zeros(len(ctl), int),
                                      mc_, 1, P)
                rec["z_level_ctrl"] = float(_z(Lv[0]))
            recs[(q, p_)] = rec
        # count-matched phase levels: z carries sqrt(n)
        idx = {p_: np.flatnonzero(ph_r == p_) for p_ in PHASES}
        nmin = min(len(v) for v in idx.values())
        if q in MATCHED and nmin >= MIN_EVENTS:
            rs = np.random.default_rng([SEED] + list(seed_key) + [qi])
            for p_ in PHASES:
                pick = rs.random((N_MATCH, len(idx[p_]))).argsort(1)[:, :nmin]
                A = np.zeros((N_MATCH, len(Vd)))
                A[np.arange(N_MATCH)[:, None], idx[p_][pick]] = 1.0 / nmin
                recs[(q, p_)]["z_diff_matched"] = float(np.nanmean(_z(A @ Vd)))
                recs[(q, p_)]["n_matched"] = int(nmin)

    # per-feature contributions to I (identity labels), and the gain check
    W_r, ok_r = wr["I"]
    W_c, ok_c = wc["I"]
    ok_r, ok_c = ok_r & hr, ok_c & hc
    dXr, dXc = Xr_post - Xr_pre, Xc_post - Xc_pre
    cr = dXr.T * np.einsum("efl,el->ef", Tr, W_r)           # (n_r, F)
    cc = dXc.T * np.einsum("efl,el->ef", Tc, W_c)
    F = dXr.shape[0]
    Ms = np.zeros((n_str, F)); ns = np.zeros(n_str)
    np.add.at(Ms, sc[ok_c], cc[ok_c]); np.add.at(ns, sc[ok_c], 1.0)
    valid = ok_r & (ns[sr] > 0)
    unit = (cr[valid] - Ms[sr[valid]] / ns[sr[valid], None]).mean(0) \
        if valid.sum() >= MIN_EVENTS else np.full(F, np.nan)
    g_r, g_c = dXr.mean(0), dXc.mean(0)
    Gs = np.zeros(n_str); np.add.at(Gs, sc[hc], g_c[hc])
    nG = np.zeros(n_str); np.add.at(nG, sc[hc], 1.0)
    vg = hr & (nG[sr] > 0)
    gain = float((g_r[vg] - Gs[sr[vg]] / nG[sr[vg]]).mean()) \
        if vg.sum() >= MIN_EVENTS else np.nan
    return recs, unit, gain


# --------------------------------------------------------------------- run
def _load_hfb_inputs(b_dir):
    with open(os.path.join(b_dir, "swr_bundle.pkl"), "rb") as f:
        B = pickle.load(f)
    contacts = pr.lfp_contacts(B)
    hiv = B["hfb_intervals"]
    return contacts, hiv


def _ripples(rip, s):
    g = rip[rip.session == s].sort_values("t_peak_s")
    t, d = g.t_peak_s.to_numpy(float), g.duration_s.to_numpy(float)
    keep = np.r_[True, np.diff(t) > 0.05] if len(t) else np.zeros(0, bool)
    return t[keep], d[keep]


def main():
    rng0 = np.random.default_rng(SEED)
    P = np.vstack([np.arange(9)] + [rng0.permutation(9)
                                    for _ in range(N_PERM)])
    deriv = rrsa._derivatives()
    b_dir = os.path.join(deriv, "group", "swr", "bundle_v2")
    rip = pd.read_csv(os.path.join(b_dir, "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s",
                               "subject_key"])
    subject = dict(zip(rip.session, rip.subject_key))
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)
    contacts, hiv = _load_hfb_inputs(b_dir)
    roi_order = list(SPIKE_ROIS) + list(HFB_ROIS)

    rows, units, counts = [], [], []
    for s in sessions:
        t_rip, d_rip = _ripples(rip, s)
        ev = session_events(steps, s, t_rip, d_rip)
        if ev is None or len(ev["ripple"]) < MIN_EVENTS \
                or len(ev["control"]) < MIN_EVENTS:
            continue
        counts.append(dict(session=s, n_ripples_task=len(t_rip),
                           n_ripples_eligible=len(ev["ripple"]),
                           n_controls=len(ev["control"]),
                           **{f"n_rip_{p_}": int((ev["ripple"].phase == p_)
                                                 .sum()) for p_ in PHASES}))
        W = {}
        for kind in ("ripple", "control"):
            t = ev[kind].t.to_numpy()
            W[kind] = ((t - INNER_S - WIN_S, t - INNER_S),
                       (t + INNER_S, t + INNER_S + WIN_S))

        signals = []
        if s in spk:
            r = roi[roi.session == s]
            for name, members in SPIKE_ROIS.items():
                cells = [c for c in r[r.roi.isin(members)].cell
                         if c < len(spk[s]["spikes"])]
                if len(cells) >= MIN_FEATURES:
                    signals.append(("spikes", name, cells,
                                    spike_signal(spk[s], cells)))
        c_s = contacts[contacts.session == s]
        if len(c_s):
            store = sb.load_hfb(b_dir, s, arrays=["hfb"])
            h = hiv[hiv.session == s]
            clean_iv = {p: g[["start_s", "stop_s"]].to_numpy(float)
                        for p, g in h.groupby("pair_id")}
            for name in HFB_ROIS:
                pairs = [p for p in c_s[c_s.roi == name].pair_id
                         if p in store["pair_ids"]]
                if len(pairs) >= MIN_FEATURES:
                    signals.append(("hfb", name, pairs,
                                    hfb_signal(store, pairs, clean_iv)))

        for sig, name, feats, (sums_fn, win_fn, min_expo) in signals:
            S_, X_ = sums_fn(ev["occ"])
            Xr_pre, Xr_post, Xc_pre, Xc_post = zscore_features([
                win_fn(*W["ripple"][0]), win_fn(*W["ripple"][1]),
                win_fn(*W["control"][0]), win_fn(*W["control"][1])])
            for ref_i, (ref, K) in enumerate(REF_SCHEMES.items()):
                refs = references(S_, X_, ev["occ"], ev["roles"], K, min_expo)
                recs, unit, gain = roi_stats(
                    ev, Xr_pre, Xr_post, Xc_pre, Xc_post, refs, P,
                    seed_key=(s, roi_order.index(name), ref_i))
                for (q, p_), rec in recs.items():
                    rows.append(dict(session=s, subject=subject[s],
                                     signal=sig, roi=name, ref=ref,
                                     quantity=q, phase=p_,
                                     n_features=len(feats), gain=gain, **rec))
                if ref == "block":
                    units += [dict(session=s, subject=subject[s], signal=sig,
                                   roi=name, feature=str(f), I_contrib=u)
                              for f, u in zip(feats, unit)]
        print(f"  s{s:02d}  ripples {len(ev['ripple'])}  controls "
              f"{len(ev['control'])}  signals {len(signals)}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"future_shift_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "per_session.csv"), index=False)
    pd.DataFrame(units).to_csv(os.path.join(d, "per_unit.csv"), index=False)
    pd.DataFrame(counts).to_csv(os.path.join(d, "event_counts.csv"),
                                index=False)
    summary = group_tests(R)
    summary.to_csv(os.path.join(d, "summary.csv"), index=False)
    overview(R, summary, pd.DataFrame(units), pd.DataFrame(counts), d)
    save_settings(d)
    print(f"-> {d}")
    return d


MEASURES = ("z_diff", "z_rip", "z_ctrl", "z_diff_matched", "z_level_ctrl")


def group_tests(R):
    """Sign-flip t across sessions (and across subjects) for every cell."""
    out = []
    for keys, g in R.groupby(["signal", "roi", "ref", "quantity", "phase"]):
        for meas in MEASURES:
            if meas not in g or g[meas].notna().sum() < 5:
                continue
            res = pr.signflip_t(g[meas], n_flip=N_FLIP, seed=SEED)
            subj = pr.signflip_t(g.groupby("subject")[meas].mean(),
                                 n_flip=N_FLIP, seed=SEED)
            out.append(dict(zip(["signal", "roi", "ref", "quantity", "phase"],
                                keys), measure=meas, n=res["n"],
                            mean=res["mean"], sem=res["sem"], t=res["t"],
                            p=res["p_two_sided"], subj_n=subj["n"],
                            subj_t=subj["t"], subj_p=subj["p_two_sided"]))
    return pd.DataFrame(out)


def _pick(S, **kw):
    m = np.ones(len(S), bool)
    for k, v in kw.items():
        m &= S[k] == v
    g = S[m]
    if not len(g):
        return None
    r = g.iloc[0]
    return {k: (float(r[k]) if isinstance(r[k], (float, np.floating))
                else int(r[k]) if isinstance(r[k], (int, np.integer))
                else r[k])
            for k in ("n", "mean", "sem", "t", "p", "subj_n", "subj_t",
                      "subj_p")}


def overview(R, S, U, C, d):
    """results_overview.json: the numbers that get quoted."""
    ov = {"preregistration": "CHANGELOG.md 2026-10-07 (FS)",
          "alpha_primary": ALPHA_PRIMARY, "events": {
              "sessions": int(len(C)),
              "ripples_in_task": int(C.n_ripples_task.sum()),
              "ripples_eligible": int(C.n_ripples_eligible.sum()),
              "controls": int(C.n_controls.sum()),
              **{f"ripples_{p_}": int(C[f"n_rip_{p_}"].sum())
                 for p_ in PHASES}}}
    for sig, roiname in PRIMARY.items():
        key = f"{sig}|{roiname}"
        ov[key] = {
            "PRIMARY_I_ripple_minus_control": _pick(
                S, signal=sig, roi=roiname, ref="block", quantity="I",
                phase="all", measure="z_diff"),
            "gate_now_level_in_controls": _pick(
                S, signal=sig, roi=roiname, ref="block", quantity="now",
                phase="all", measure="z_level_ctrl"),
            "J_ripple_minus_control": _pick(
                S, signal=sig, roi=roiname, ref="block", quantity="J",
                phase="all", measure="z_diff"),
            "I_near4": _pick(S, signal=sig, roi=roiname, ref="near4",
                             quantity="I", phase="all", measure="z_diff"),
            "I_by_phase_matched": {p_: _pick(
                S, signal=sig, roi=roiname, ref="block", quantity="I",
                phase=p_, measure="z_diff_matched") for p_ in PHASES},
        }
    for sig, roiname in [("spikes", r) for r in SPIKE_ROIS] + \
                        [("hfb", r) for r in HFB_ROIS]:
        key = f"{sig}|{roiname}"
        ov.setdefault(key, {})
        ov[key]["profile_ripple_minus_control"] = {
            q: _pick(S, signal=sig, roi=roiname, ref="block", quantity=q,
                     phase="all", measure="z_diff") for q in QUANTITIES}
        ov[key]["gate_now_level_in_controls"] = _pick(
            S, signal=sig, roi=roiname, ref="block", quantity="now",
            phase="all", measure="z_level_ctrl")
        g = R[(R.signal == sig) & (R.roi == roiname) & (R.ref == "block")
              & (R.quantity == "I") & (R.phase == "all")]
        ov[key]["gain_ripple_minus_control"] = (
            pr.signflip_t(g.gain, n_flip=N_FLIP, seed=SEED)
            if g.gain.notna().sum() >= 5 else None)
        u = U[(U.signal == sig) & (U.roi == roiname)].I_contrib.dropna() \
            if len(U) else pd.Series(dtype=float)
        ov[key]["per_unit_I_contrib"] = (
            pr.signflip_t(u, n_flip=N_FLIP, seed=SEED) if len(u) >= 5
            else None)
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"),
              indent=2, default=float)


def save_settings(d):
    json.dump(dict(
        seed=SEED, n_perm=N_PERM, n_flip=N_FLIP, n_match=N_MATCH,
        count_matched_quantities=list(MATCHED), n_strata=N_STRATA,
        win_s=WIN_S, inner_s=INNER_S, max_ripple_duration_s=MAX_RIP_DUR_S,
        other_ripple_pad_s=OTHER_RIP_PAD_S, control_step_s=CTRL_STEP_S,
        control_ripple_pad_s=CTRL_RIP_PAD_S, min_dwell_s=MIN_DWELL_S,
        min_clean_samples=MIN_CLEAN_SAMPLES, min_events=MIN_EVENTS,
        min_features=MIN_FEATURES, strata_bin_edges_s=BIN_EDGES.tolist(),
        spike_rois=SPIKE_ROIS, hfb_rois=list(HFB_ROIS),
        hfb_medial_max_abs_x_mm=pr.MEDIAL_MAX_ABS_X,
        ref_schemes={k: ("all other traversals of the block" if v is None
                         else f"{v} nearest other traversals of the block")
                     for k, v in REF_SCHEMES.items()},
        roles=list(ROLES), quantities=list(QUANTITIES), phases=list(PHASES),
        primary=PRIMARY, alpha_primary=ALPHA_PRIMARY,
        statistic="E_w(L) = x_w . R(L); I = delta(FUT - NOW), "
                  "J = delta(FUT - PAST); ripple minus stratum-matched "
                  "control; per-session z vs label permutations; sign-flip "
                  "across sessions, two-sided",
        preregistration="CHANGELOG.md 2026-10-07 (FS)",
        created=datetime.datetime.now().isoformat(timespec="seconds")),
        open(os.path.join(d, "settings.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
