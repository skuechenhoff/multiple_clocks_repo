#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hippocampal location content in ripples vs ordinary moments, encode vs known:
the current square, the path behind and ahead, and the reward squares.
(2026-11-23)

Built on the REPORTED analysis (`mc/analyse/swr_phase_contrast.py`), imported:
same sessions, ripples (peak inside a visit), cells, encode / execute split,
each cell's map from the OTHER CONFIGURATIONS (leave-one-grid_id-out,
`swc.build_templates`), window = the ripple's own duration, counts z-scored
per cell (ripples together with their flanks, as reported), 1000
square-label permutations, 1000 count-matched draws with the reported seed.
The CURRENT-square statistic in ripples must reproduce the reported
per-session values (checked, written to result.json).

ADDED
1. Ordinary moments ("non-ripple"): window centres tiled every 50 ms inside
   visits, the window fully inside the visit, width = the session's median
   ripple duration, no ripple peak within 0.5 s, at most 15,000 per session
   (seeded). Counts z-scored per cell across these windows. Same templates,
   statistics and permutations. Levels are count-matched: in every session
   each of the four cells (ripple / non-ripple x encode / execute) is a
   draw of n* events, n* = the reported matched count (fewest eligible
   ripples of the two phases); 1000 draws.
2. More squares than the current one. Per event, from the task (labels only;
   the templates stay other-configuration):
     past3 past2 past1 fut1 fut2 fut3   the previous / next three distinct
                                        squares of the block's visit sequence
                                        (`future_shift.role_table`)
     NEXT_REW    the next reward of the sequence that is not the current
                 square (`goal_lag_locations`, g = 0, else 1)
     OTHER_REW   the configuration's other reward squares (not current, not
                 next)
     ANY_REW     every reward square that is not the current square
   Each is scored DISTANCE-CONTROLLED: evidence at the square minus the
   mean evidence at the other squares at the same Manhattan distance from
   the current square (undefined if it has none), because the place code is
   spatially smooth and path/reward squares sit at typical distances.
     REW-NOW         mean evidence at the non-current reward squares minus
                     evidence at the current square (no distance control:
                     > 0 = rewards represented MORE than the current square)
     NEXT-OTHER_REW  NEXT_REW minus OTHER_REW (both distance-controlled)
   Under each permutation the same weights are applied to the permuted
   evidence, so observed and null go through one function.

Session = unit; per-session quantities tested with the reported one-sample
t-test and with a sign-flip test (context_content.group), both written.

    python scripts/location_content_ripples_by_phase.py

@author: Svenja Kuchenhoff
"""
import os
import sys
import json
import datetime

import numpy as np
import pandas as pd
from scipy import sparse
from joblib import Parallel, delayed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc
import mc.analyse.swr_phase_contrast as pcn
import mc.analyse.swr_explore.future_shift as fs
from mc.analyse.swr_explore.goal_lag import goal_lag_locations
from mc.analyse.swr_explore.context_content import group

SEED = pcn.SEED
PHASES = pcn.PHASES
ROIS = ("HC_all", "HC_anterior", "HC_mid")
CTRL_STEP_S, CTRL_PAD_S, MAX_CTRL = fs.CTRL_STEP_S, 0.5, 15000
ROLES = ("past3", "past2", "past1", "fut1", "fut2", "fut3")
STATS = ("CURRENT",) + ROLES + ("NEXT_REW", "OTHER_REW", "ANY_REW",
                                "REW-NOW", "NEXT-OTHER_REW")
_rc = np.array(divmod(np.arange(9), 3)).T
DIST = np.abs(_rc[:, None, :] - _rc[None, :, :]).sum(-1)       # (9, 9)
OUT = os.path.join(rrsa._derivatives(), "group", "swr",
                   f"location_content_by_phase_{datetime.date.today()}")
REPORTED = os.path.join(rrsa._derivatives(), "group", "swr",
                        "phase_contrast_2026-09-30", "per_session.csv")


# ── weights per occupancy row ────────────────────────────────────────────
def ring(now, L):
    """(9,) evidence at L minus the mean at the other squares at L's
    distance from `now`; None if undefined."""
    if L < 1 or L == now:
        return None
    d = DIST[now - 1]
    mates = [m for m in range(9) if d[m] == d[L - 1] and m != L - 1]
    if not mates:
        return None
    w = np.zeros(9)
    w[L - 1] = 1.0
    w[mates] -= 1.0 / len(mates)
    return w


def _mean(ws):
    ws = [w for w in ws if w is not None]
    return np.mean(ws, axis=0) if ws else None


def occ_weights(steps_s, occ):
    """{stat: (W (n_occ, 9), defined (n_occ,))}."""
    RT = fs.role_table(steps_s, occ)
    G = goal_lag_locations(steps_s, occ)
    st = steps_s.sort_values("t_s").reset_index(drop=True)
    rew = st[["rew_A", "rew_B", "rew_C", "rew_D"]].to_numpy(float)[
        occ.step_idx.to_numpy()]
    out = {k: (np.zeros((len(occ), 9)), np.zeros(len(occ), bool)) for k in STATS}
    for i in range(len(occ)):
        now = int(RT.now.iloc[i])
        w = np.full(9, -1.0 / 8)
        w[now - 1] = 1.0
        W = {"CURRENT": w}
        for r in ROLES:
            W[r] = ring(now, int(RT[r].iloc[i]))
        R = [int(x) for x in rew[i] if np.isfinite(x)]
        nxt = next((int(g) for g in G[i, :2] if g >= 1 and g != now), -1)
        others = [L for L in set(R) if L not in (now, nxt)]
        W["NEXT_REW"] = ring(now, nxt)
        W["OTHER_REW"] = _mean([ring(now, L) for L in others])
        W["ANY_REW"] = _mean([ring(now, L) for L in set(R) if L != now])
        nr = [L for L in set(R) if L != now]
        if len(R) == 4 and nr:
            w = np.zeros(9)
            w[np.array(nr) - 1] = 1.0 / len(nr)
            w[now - 1] -= 1.0
            W["REW-NOW"] = w
        else:
            W["REW-NOW"] = None
        W["NEXT-OTHER_REW"] = (W["NEXT_REW"] - W["OTHER_REW"]
                               if W["NEXT_REW"] is not None
                               and W["OTHER_REW"] is not None else None)
        for k in STATS:
            if W.get(k) is not None:
                out[k][0][i] = W[k]
                out[k][1][i] = True
    return out


# ── statistic under every permutation ────────────────────────────────────
def stat_perm(E, W, P, block=2000):
    """(n, 1 + N_PERM): sum_L W[i, L] E[i, P[p, L]]."""
    E = np.nan_to_num(E)
    out = np.empty((len(E), P.shape[0]))
    for a in range(0, len(E), block):
        G = E[a:a + block][:, P]                      # (b, n_perm + 1, 9)
        out[a:a + block] = np.einsum("il,ipl->ip", W[a:a + block], G)
    return out


def matched(S, idx, n, rs):
    """Reported count-matched draws: per phase, N_MATCH subsets of n."""
    zd = {}
    for p_ in PHASES:
        rows = np.concatenate([rs.choice(idx[p_], n, replace=False)
                               for _ in range(pcn.N_MATCH)])
        A = sparse.csr_matrix(
            (np.full(rows.size, 1.0 / n),
             (np.repeat(np.arange(pcn.N_MATCH), n), rows)),
            shape=(pcn.N_MATCH, len(S)))
        zd[p_] = pcn._z(A @ S)
    return zd


# ── one session ──────────────────────────────────────────────────────────
def events(s, occ, t_all, d_all, steps_s):
    """Ripples inside visits (+ flanks) and ordinary moments of one session,
    exactly as the reported analysis defines them; None if too few."""
    ph_i, _, _, _ = swc.phase_labels(steps_s, occ)
    o = occ.sort_values("start_s")
    a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
    j = np.searchsorted(a, t_all, side="right") - 1
    jj = np.clip(j, 0, len(a) - 1)
    k = o.index.to_numpy()[jj]
    inside = (j >= 0) & (t_all <= b[jj])
    if inside.sum() < pcn.MIN_RIPPLES:
        return None
    t_rip, d_rip, k = t_all[inside], d_all[inside], k[inside]
    half = d_rip / 2.0
    grid = occ.cv_group.to_numpy()[k]
    loc = occ["loc"].to_numpy()[k].astype(int)
    ph = ph_i[k]
    grids = np.unique(occ.cv_group.to_numpy())
    masks = {p_: ph == p_ for p_ in PHASES}
    if min(m.sum() for m in masks.values()) < pcn.MIN_RIPPLES:
        return None
    t_fl, owner = swc.matched_flanks(t_rip, half, occ, other_t=t_rip,
                                     gap_s=pcn.FLANK_GAP_S,
                                     max_per=pcn.MAX_FLANKS)
    w_fl = half[owner] if len(t_fl) else np.zeros(0)

    # ordinary moments
    h = float(np.median(half))
    occ_ok = np.isin(ph_i, PHASES)
    tc, kc = [], []
    for row in np.flatnonzero(occ_ok):
        x = np.arange(occ.start_s.iat[row] + h, occ.stop_s.iat[row] - h + 1e-9,
                      CTRL_STEP_S)
        tc.append(x); kc.append(np.full(len(x), row))
    tc, kc = np.concatenate(tc), np.concatenate(kc)
    rs_ = np.sort(t_all)
    near = (np.searchsorted(rs_, tc + CTRL_PAD_S)
            - np.searchsorted(rs_, tc - CTRL_PAD_S))
    tc, kc = tc[near == 0], kc[near == 0]
    if len(tc) > MAX_CTRL:
        pick = np.sort(np.random.default_rng([SEED, s, 9]).choice(
            len(tc), MAX_CTRL, replace=False))
        tc, kc = tc[pick], kc[pick]
    ph_c = ph_i[kc]
    masks_c = {p_: ph_c == p_ for p_ in PHASES}
    return dict(t_rip=t_rip, half=half, k=k, grid=grid, loc=loc, grids=grids,
                masks=masks, t_fl=t_fl, w_fl=w_fl, tc=tc, kc=kc, h=h,
                masks_c=masks_c)


def session(s, spk_s, r, occ, t_all, d_all, steps_s, P):
    pcn._stat.P = P                 # the reported statistic reads it here
    ev = events(s, occ, t_all, d_all, steps_s)
    if ev is None:
        return []
    t_rip, half, k, grid, loc, grids = (ev[x] for x in
                                        ("t_rip", "half", "k", "grid", "loc",
                                         "grids"))
    masks, t_fl, w_fl, tc, kc, h, masks_c = (ev[x] for x in
                                             ("masks", "t_fl", "w_fl", "tc",
                                              "kc", "h", "masks_c"))
    OW = occ_weights(steps_s, occ)
    rows = []
    for roi in ROIS:
        ri = list(pcn.ROI_SETS).index(roi)
        cells = [c for c in r[r.roi.isin(pcn.ROI_SETS[roi])].cell.to_numpy()
                 if c < len(spk_s["spikes"])]
        if len(cells) < pcn.MIN_CELLS:
            continue
        _, loo = swc.build_templates(spk_s, cells, occ, grids)
        C_rip, _ = swc.zscore_cells([
            swc.window_counts(spk_s, cells, t_rip - half, t_rip + half),
            swc.window_counts(spk_s, cells, t_fl - w_fl, t_fl + w_fl)])
        C_ctl, = swc.zscore_cells([
            swc.window_counts(spk_s, cells, tc - h, tc + h)])
        E_r = swc.score_windows(C_rip, t_rip, grid, loo)
        E_c = swc.score_windows(C_ctl, tc, occ.cv_group.to_numpy()[kc], loo)
        ok_r, ok_c = np.isfinite(E_r).all(1), np.isfinite(E_c).all(1)
        for stat in STATS:
            W, dfn = OW[stat]
            S_r = stat_perm(E_r, W[k], P)
            S_c = stat_perm(E_c, W[kc], P)
            if stat == "CURRENT":          # identical to the reported _stat
                assert np.allclose(S_r, pcn._stat(E_r, loc - 1))
            er, ec = ok_r & dfn[k], ok_c & dfn[kc]
            idx_r = {p_: np.flatnonzero(masks[p_] & er) for p_ in PHASES}
            idx_c = {p_: np.flatnonzero(masks_c[p_] & ec) for p_ in PHASES}
            rec = dict(session=int(s), roi=roi, stat=stat)
            for p_ in PHASES:
                rec[f"n_rip_{p_}"] = len(idx_r[p_])
                rec[f"n_ctl_{p_}"] = len(idx_c[p_])
                if len(idx_r[p_]) >= pcn.MIN_RIPPLES:
                    rec[f"z_rip_{p_}"] = float(pcn._z(S_r[idx_r[p_]].mean(0)))
                if len(idx_c[p_]) >= pcn.MIN_RIPPLES:
                    rec[f"z_ctl_{p_}"] = float(pcn._z(S_c[idx_c[p_]].mean(0)))
            n = min(len(v) for v in idx_r.values())
            if n >= pcn.MIN_RIPPLES and min(len(v) for v in idx_c.values()) >= n:
                zr = matched(S_r, idx_r, n, np.random.default_rng([SEED, s, ri, 7]))
                zc = matched(S_c, idx_c, n, np.random.default_rng([SEED, s, ri, 8]))
                rec["n_matched"] = int(n)
                for p_ in PHASES:
                    rec[f"zm_rip_{p_}"] = float(np.nanmean(zr[p_]))
                    rec[f"zm_ctl_{p_}"] = float(np.nanmean(zc[p_]))
                    rec[f"rip-ctl_{p_}"] = rec[f"zm_rip_{p_}"] - rec[f"zm_ctl_{p_}"]
                rec["dz_rip"] = float(np.nanmean(zr["encode"] - zr["execute"]))
                rec["dz_ctl"] = float(np.nanmean(zc["encode"] - zc["execute"]))
                rec["(rip-ctl) enc - exe"] = (rec["rip-ctl_encode"]
                                              - rec["rip-ctl_execute"])
            rows.append(rec)
    return rows


def run():
    rng = np.random.default_rng(SEED)
    P = np.vstack([np.arange(9)] + [rng.permutation(9) for _ in range(pcn.N_PERM)])
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)
    jobs = []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < pcn.MIN_RIPPLES:
            continue
        jobs.append(delayed(session)(s, spk[s], r, occ, t_rip, d_rip,
                                     steps[steps.session == s], P))
    out = Parallel(n_jobs=8, verbose=5)(jobs)
    return pd.DataFrame([x for o in out for x in o])


QUANT = ("z_rip_encode", "z_rip_execute", "z_ctl_encode", "z_ctl_execute",
         "zm_rip_encode", "zm_rip_execute", "zm_ctl_encode", "zm_ctl_execute",
         "dz_rip", "dz_ctl", "rip-ctl_encode", "rip-ctl_execute",
         "(rip-ctl) enc - exe")


def summarise(R):
    res = {}
    for (roi, stat), g in R.groupby(["roi", "stat"]):
        for q in QUANT:
            if q in g:
                res[f"{roi} | {stat} | {q}"] = group(g[q])
    rep = pd.read_csv(REPORTED)
    chk = {}
    for roi in ROIS:
        a = rep[rep.roi == roi].set_index("session")
        m = R[(R.roi == roi) & (R.stat == "CURRENT")].set_index("session")
        both = sorted(set(a.index) & set(m.index))
        chk[roi] = dict(n=[len(both), len(a), len(m)], **{
            mine: float(np.nanmax(np.abs(a.loc[both, theirs]
                                         - m.loc[both, mine])))
            for mine, theirs in (("z_rip_encode", "z_encode"),
                                 ("z_rip_execute", "z_execute"),
                                 ("zm_rip_encode", "z_encode_matched"),
                                 ("zm_rip_execute", "z_execute_matched"),
                                 ("dz_rip", "d_z"))})
    res["reproduces reported (max |diff|)"] = chk
    return res


if __name__ == "__main__":
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, "per_session.csv")
    R = pd.read_csv(f) if os.path.exists(f) else run()
    R.to_csv(f, index=False)
    res = summarise(R)
    settings = dict(
        created=str(datetime.date.today()),
        script="scripts/location_content_ripples_by_phase.py",
        base="mc/analyse/swr_phase_contrast.py (reported): sessions, "
             "ripples, cells, phases, other-configuration templates, "
             "window, normalisation, statistic, permutations, count matching",
        rois=ROIS, stats=STATS, non_ripple=dict(
            step_s=CTRL_STEP_S, pad_s=CTRL_PAD_S, max_per_session=MAX_CTRL,
            width="session median ripple duration, window inside the visit"),
        distance_control="evidence minus mean of the other squares at the "
                         "same Manhattan distance from the current square",
        p="'p' = one-sample t-test (as reported); 'p_perm' = sign-flip")
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(dict(settings=settings, tests=res), fh, indent=1,
                  default=float)
    print(json.dumps(res["reproduces reported (max |diff|)"], indent=1))
