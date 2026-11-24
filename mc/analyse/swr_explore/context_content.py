"""
CONTEXT CONTENT (D3, D4; pre-registered + amended CHANGELOG 2026-11-15).

    python scripts/swr_content_explore.py context_content

SK's model: at uncovers, cortex (mOFC: position in sequence; lOFC: reward)
signals the hippocampus to store the current position; later ripples run
HC -> cortex to load the plan; ripples away from uncovers carry the rewarded
squares, more once the sequence is known. Alternative: location and sequence
are played together (bound). Son, Azab et al. (same dataset): during learning
HC cells RAMP across the four positions; position-SELECTIVE cells appear only
after learning.

Ripple contexts x phase (learning = first traversal + rest of encoding;
execution), from the occupancy role table:
    UNCOVER  in (press, press + min(pause, 2 s)] after an uncover press
    AWAY     no uncover press in [t - 2 s, t + 0.5 s]

D3  HC location content, T4 machinery (same-block references, +-125 ms):
    evidence for the configuration's REWARD squares vs non-rewarded ROUTE
    squares at the same Manhattan distance from the current square (current
    square and neighbours excluded), and vs all non-rewarded squares.
    matched = ripple - T4 stratum controls of the same context; own = ripple
    - its own flanks. PRIMARY: HC, AWAY ripples, execution, REW - ROUTE > 0.
    Added 2026-11-19 (descriptive): CURRENT square, and ROUTE - OTHER
    (non-rewarded route squares vs squares that are neither, same distance).

D4  position in sequence (state), templates WITHIN phase (leave-one-
    configuration-out over that phase's occupancy), each cell's template
    split into a RAMP part (projection on -1.5,-0.5,+0.5,+1.5) and a
    SELECTIVE part (the rest).
    D4a direction: across ripples, rho(frontal pre, HC post) vs rho(HC pre,
        frontal post), Fisher z difference, minus random pairing (own) or
        minus the same for matched moments (matched).
    D4b symmetric option: (i) HC location x state conjunction interaction,
        ripple - matched; (ii) partial r(HC location, frontal state | activity)
        in the same window, ripple - matched; (iii) the same within HC.
    D4c timeline: region x phase x code (ramp / selective) in ripples and in
        still moments. PRIMARY: HC selective evidence, learning ripples -
        matched learning moments > 0.

D5  (2026-11-19, exploratory, SK: "are location and sequence co-active
    BEFORE a ripple, and does that elicit it?") Every D4 code in the window
    BEFORE the ripple (-500..-50 ms) and AFTER it (+50..+500 ms): level in
    ripples and in the same window around still moments, ripple - matched,
    and ripple - its own farther window (-1000..-550 / +550..+1000 ms).
    And the D4b coupling (HC location x frontal state | activity) in the
    pre-ripple window. These windows are z-scored separately from D4a's, so
    D4a is unchanged.

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import datetime
import warnings
import itertools

import numpy as np
import pandas as pd
from scipy import stats

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc
import mc.analyse.swr_explore.future_shift as fs
import mc.analyse.swr_explore.path_content as pc
from mc.analyse.swr_explore.state_content import (_templates, _score,
                                                  _time_into_traversal,
                                                  _partial_r)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "scripts", "exploratory"))

SEED = 42
N_FLIP = 10000
N_PERM_CONJ = 200
N_PERM_LOC = 100
N_SHUF = 200
MIN_EV = 15
MIN_CELLS = 2
MAX_CTRL = 15000
PAUSE_MAX_S, AWAY_BEFORE_S, AWAY_AFTER_S = 2.0, 2.0, 0.5
CTRL_PAD_S = 0.5
HALF_S = 0.125
PRE, POST = (-0.5, -0.05), (0.05, 0.5)
FLANK = (0.5, 0.75)
# D5 own baselines for PRE / POST: the equally long window just beyond them
WIN = {"in": (-HALF_S, HALF_S), "pre": PRE, "post": POST,
       "farpre": (-1.0, -0.55), "farpost": (0.55, 1.0)}
REGIONS = {"HC": ["HC_anterior", "HC_mid"], "frontal": ["mPFC", "mOFC"],
           "mPFC": ["mPFC"], "mOFC": ["mOFC"]}
RAMP = np.array([-1.5, -0.5, 0.5, 1.5])
N_STATE, N_LOC = 4, 9
SETS = ("U_L", "U_E", "A_L", "A_E", "all_L", "all_E", "all")
LEARN = ("first_traversal", "encode_rest")


# ── context labels ───────────────────────────────────────────────────────
def contexts(P):
    U = P[P.kind == "uncover"]
    tu = np.sort(U.t_s.to_numpy(float))
    Us = U.dropna(subset=["still_next_s"])
    a = Us.t_s.to_numpy(float)
    b = a + np.minimum(Us.still_next_s.to_numpy(float), PAUSE_MAX_S)
    o = np.argsort(a)
    a, b = a[o], b[o]

    def uncover(t):
        k = np.searchsorted(a, t, side="left") - 1
        kk = np.clip(k, 0, None)
        return (k >= 0) & (t <= b[kk]) & (t > a[kk]) if len(a) else \
            np.zeros(len(t), bool)

    def away(t):
        n = (np.searchsorted(tu, t + AWAY_AFTER_S, side="right")
             - np.searchsorted(tu, t - AWAY_BEFORE_S, side="left"))
        return n == 0
    return uncover, away


def set_masks(t, phase, uncover, away):
    L = np.isin(phase, LEARN)
    E = phase == "execute"
    u, w = uncover(t), away(t)
    return {"U_L": u & L, "U_E": u & E, "A_L": w & L, "A_E": w & E,
            "all_L": L, "all_E": E, "all": L | E}


def group(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size < 3:
        return dict(n=int(v.size))
    t, p = stats.ttest_1samp(v, 0.0)
    rng = np.random.default_rng(SEED)
    null = (rng.choice([-1.0, 1.0], size=(N_FLIP, v.size)) * v).mean(1)
    return dict(n=int(v.size), mean=float(v.mean()),
                sem=float(v.std(ddof=1) / np.sqrt(v.size)), t=float(t),
                p=float(p), p_perm=float((1 + np.sum(np.abs(null)
                                                     >= abs(v.mean())))
                                         / (1 + N_FLIP)))


def _z(M):
    sd = M[..., 1:].std(axis=-1)
    return np.where(sd > 0, (M[..., 0] - M[..., 1:].mean(axis=-1)) / sd,
                    np.nan)


# ── D3: rewarded squares in HC ripples ───────────────────────────────────
def grid_dist(a, b):
    a, b = np.asarray(a) - 1, np.asarray(b) - 1
    return np.abs(a // 3 - b // 3) + np.abs(a % 3 - b % 3)


def reward_route(steps_s, roles, occ):
    """{block: (reward squares set, route squares set)}."""
    out = {}
    for blk, g in steps_s.groupby("grid_num"):
        rew = set(int(x) for x in g[["rew_A", "rew_B", "rew_C", "rew_D"]]
                  .iloc[0].to_numpy() if np.isfinite(x))
        m = (roles.block.to_numpy() == blk) & (roles.phase.to_numpy() == "execute")
        route = set(occ["loc"].to_numpy().astype(int)[m])
        if route:
            out[blk] = (rew, route)
    return out


D3_QUANTITIES = ("CURRENT", "REW-ROUTE", "REW-OTHER", "ROUTE-OTHER")


def d3_weights(ev, V, rr):
    """{quantity: (weights (n, 9), defined (n,))}.

    CURRENT      the occupied square minus the mean of the other squares
    REW-ROUTE    reward squares vs non-rewarded squares of the block's route
    REW-OTHER    reward squares vs all non-rewarded squares
    ROUTE-OTHER  non-rewarded route squares vs squares that are neither
    The last three compare squares at the SAME Manhattan distance (2-4) from
    the current square, so the spatial gradient of the location code cannot
    produce them."""
    n = len(ev)
    out = {k: (np.zeros((n, 9)), np.zeros(n, bool)) for k in D3_QUANTITIES}
    now = ev.now.to_numpy().astype(int)
    blk = ev.block.to_numpy()
    sq = np.arange(1, 10)
    for i in range(n):
        if blk[i] not in rr:
            continue
        rew, route = rr[blk[i]]
        d = grid_dist(sq, now[i])
        vis = V[i]
        others = [s for s in sq if s != now[i] and vis[s - 1]]
        if vis[now[i] - 1] and others:
            W = np.zeros(9)
            W[now[i] - 1] = 1.0
            W[np.array(others) - 1] = -1.0 / len(others)
            out["CURRENT"][0][i] = W
            out["CURRENT"][1][i] = True
        for name, this, other in (
                ("REW-ROUTE", lambda s: s in rew,
                 lambda s: s in route and s not in rew),
                ("REW-OTHER", lambda s: s in rew, lambda s: s not in rew),
                ("ROUTE-OTHER", lambda s: s in route and s not in rew,
                 lambda s: s not in route and s not in rew)):
            W, k = np.zeros(9), 0
            for dd in (2, 3, 4):
                r_ = [s for s in sq if d[s - 1] == dd and this(s) and vis[s - 1]]
                o_ = [s for s in sq if d[s - 1] == dd and other(s) and vis[s - 1]]
                if r_ and o_:
                    for s in r_:
                        W[s - 1] += 1.0 / len(r_)
                    for s in o_:
                        W[s - 1] -= 1.0 / len(o_)
                    k += 1
            if k:
                out[name][0][i] = W / k
                out[name][1][i] = True
    return out


def d3_session(s, spk, roi, steps, rip, P_s, P9):
    t_rip, d_rip = fs._ripples(rip, s)
    U = P_s[P_s.kind == "uncover"].dropna(subset=["still_next_s"])
    ev = pc.session_events(steps, s, t_rip, d_rip, U.t_s.to_numpy(float),
                           U.still_next_s.to_numpy(float))
    if ev is None or len(ev["ripple"]) < MIN_EV:
        return []
    uncover, away = contexts(P_s)
    rr = reward_route(steps[steps.session == s], ev["roles"], ev["occ"])
    rip_ev, ctl_ev = ev["ripple"], ev["control"]
    mr = set_masks(rip_ev.t.to_numpy(), rip_ev.phase.to_numpy(), uncover, away)
    mc = set_masks(ctl_ev.t.to_numpy(), ctl_ev.phase.to_numpy(), uncover, away)
    rows = []
    r = roi[roi.session == s]
    for name, members in fs.SPIKE_ROIS.items():
        cells = [c for c in r[r.roi.isin(members)].cell if c < len(spk[s]["spikes"])]
        if len(cells) < MIN_CELLS:
            continue
        sums, win, min_expo = fs.spike_signal(spk[s], cells)
        S_, X_ = sums(ev["occ"])
        refs = fs.references(S_, X_, ev["occ"], ev["roles"], None, min_expo)
        tr, tc = rip_ev.t.to_numpy(), ctl_ev.t.to_numpy()
        Xr, Xc = fs.zscore_features([win(tr - HALF_S, tr + HALF_S),
                                     win(tc - HALF_S, tc + HALF_S)])
        Xr2, Xf = fs.zscore_features([
            win(tr - HALF_S, tr + HALF_S),
            (win(tr - pc.FLANK_S[1], tr - pc.FLANK_S[0])
             + win(tr + pc.FLANK_S[0], tr + pc.FLANK_S[1])) / 2.0])
        Er, Vr, _ = fs.evidence(Xr, rip_ev, refs)
        Ec, Vc, _ = fs.evidence(Xc, ctl_ev, refs)
        Er2, _, _ = fs.evidence(Xr2, rip_ev, refs)
        Ef, _, _ = fs.evidence(Xf, rip_ev, refs)
        hr, hc = np.isfinite(Er).all(1), np.isfinite(Ec).all(1)
        hf = np.isfinite(Er2).all(1) & np.isfinite(Ef).all(1) & \
            rip_ev.flank_ok.to_numpy()
        Er, Ec = np.nan_to_num(Er), np.nan_to_num(Ec)
        Er2, Ef = np.nan_to_num(Er2), np.nan_to_num(Ef)
        wr, wc = d3_weights(rip_ev, Vr, rr), d3_weights(ctl_ev, Vc, rr)
        sr, sc = rip_ev.stratum.to_numpy(), ctl_ev.stratum.to_numpy()
        for q in D3_QUANTITIES:
            W_r, ok_r = wr[q]
            W_c, ok_c = wc[q]
            for key in SETS:
                rec = dict(session=int(s), roi=name, quantity=q, set=key,
                           n_cells=len(cells))
                okc = ok_c & hc & mc[key]
                Cs, cnt = fs.stratum_means(W_c, Ec, sc, okc, fs.N_STRATA, P9)
                valid = ok_r & hr & mr[key] & (cnt[sr] > 0)
                rec["n_rip"] = int(valid.sum())
                if valid.sum() >= MIN_EV:
                    Rp = fs.permuted(W_r[valid], Er[valid][:, P9])
                    Cp = Cs[sr[valid]]
                    rec["z_matched"] = float(_z((Rp - Cp).mean(0)))
                    rec["z_rip"] = float(_z(Rp.mean(0)))
                    rec["z_ctrl"] = float(_z(Cp.mean(0)))
                vf = ok_r & hf & mr[key]
                rec["n_rip_flank"] = int(vf.sum())
                if vf.sum() >= MIN_EV:
                    rec["z_own"] = float(_z(fs.permuted(
                        W_r[vf], (Er2[vf] - Ef[vf])[:, P9]).mean(0)))
                rows.append(rec)
    return rows


# ── D4: state (ramp / selective), location, conjunction ──────────────────
def perm_weights(K, perms, kind):
    """W[p, label, K]: the statistic for a window with true `label` under
    label permutation p is W[p, label] . E (row 0 = identity)."""
    W = np.zeros((len(perms), K, K))
    for p, pm in enumerate(perms):
        for lab in range(K):
            if kind in ("state", "loc"):
                W[p, lab] -= 1.0 / (K - 1)
                W[p, lab, pm[lab]] += 1.0 + 1.0 / (K - 1)
            else:                                  # conjunction interaction
                l0, s0 = lab // N_STATE, lab % N_STATE
                for c in range(K):
                    l, st = c // N_STATE, c % N_STATE
                    if l == l0 and st == s0:
                        w = 1.0
                    elif l == l0:
                        w = -1.0 / 3
                    elif st == s0:
                        w = -1.0 / 8
                    else:
                        w = 1.0 / 24
                    W[p, lab, pm[c]] += w
    return W


def stat_perm(E, lab, W):
    """(n, n_perm) statistic per window and permutation."""
    out = np.full((len(E), W.shape[0]), np.nan)
    for l in np.unique(lab):
        m = lab == l
        out[m] = E[m] @ W[:, l, :].T
    return out


def split_templates(loo):
    ramp, sel = {}, {}
    for g, T in loo.items():
        coef = T @ RAMP / (RAMP @ RAMP)
        ramp[g] = np.outer(coef, RAMP)
        sel[g] = T - ramp[g]
    return ramp, sel


def _levels(Sr, Sc, Sf, ev, key):
    """Code level in one window type: ripples (z_rip), control moments
    (z_still), ripple minus its stratum's control mean (z_matched), ripple
    minus its own baseline window (z_own). S* are (n_events, n_perm + 1)."""
    rec = {}
    mr = ev["rip"]["masks"][key] & np.isfinite(Sr[:, 0])
    mc_ = ev["ctl"]["masks"][key] & np.isfinite(Sc[:, 0])
    rec["n_rip"], rec["n_ctl"] = int(mr.sum()), int(mc_.sum())
    if mc_.sum() >= MIN_EV:
        rec["z_still"] = float(_z(Sc[mc_].mean(0)))
    if mr.sum() < MIN_EV:
        return rec
    rec["z_rip"] = float(_z(Sr[mr].mean(0)))
    strat_r, strat_c = ev["rip"]["strat"], ev["ctl"]["strat"]
    diffs, ok = [], []
    for q in np.unique(strat_r[mr]):
        cq = mc_ & (strat_c == q)
        if cq.sum() < 5:
            continue
        rq = mr & (strat_r == q)
        diffs.append((Sr[rq] - Sc[cq].mean(0)).sum(0))
        ok.append(rq.sum())
    if sum(ok) >= MIN_EV:
        rec["z_matched"] = float(_z(np.sum(diffs, 0) / sum(ok)))
    mf = mr & np.isfinite(Sf[:, 0])
    if mf.sum() >= MIN_EV:
        rec["z_own"] = float(_z((Sr[mf] - Sf[mf]).mean(0)))
    return rec


def d4_session(s, spk, roi, steps, rip, P_s, perms):
    occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
    if not len(occ) or len(t_rip) < MIN_EV:
        return None
    st_s = steps[steps.session == s]
    roles = fs.role_table(st_s, occ)
    _, state_i = _time_into_traversal(st_s, occ)
    phase_i = roles.phase.to_numpy()
    cond_state = np.where(np.isfinite(state_i), state_i - 1, -1).astype(int)
    cond_loc = (occ["loc"].to_numpy() - 1).astype(int)
    cond_conj = np.where(cond_state >= 0, cond_loc * N_STATE + cond_state, -1)
    grids = np.unique(occ.cv_group.to_numpy())
    uncover, away = contexts(P_s)
    tp = np.sort(P_s.t_s.to_numpy(float))
    from swr_hfb_discovery_lock import strata

    # events: ripples and still controls, placed in occupancy
    o = occ.sort_values("start_s")
    row_of, a, b = o.index.to_numpy(), o.start_s.to_numpy(), o.stop_s.to_numpy()

    def place(t):
        j = np.searchsorted(a, t, side="right") - 1
        jj = np.clip(j, 0, len(a) - 1)
        k = row_of[jj]
        ok = (j >= 0) & (t <= b[jj]) & (cond_state[k] >= 0) & \
            np.isin(phase_i[k], LEARN + ("execute",))
        return k, ok
    k_r, ok_r = place(t_rip)
    tr, kr = t_rip[ok_r], k_r[ok_r]
    tc = pc.controls(occ)
    rs = np.sort(t_rip)
    near = (np.searchsorted(rs, tc + CTRL_PAD_S) - np.searchsorted(rs, tc - CTRL_PAD_S))
    tc = tc[near == 0]
    k_c, ok_c = place(tc)
    tc, kc = tc[ok_c], k_c[ok_c]
    if len(tc) > MAX_CTRL:
        pick = np.sort(np.random.default_rng(SEED + int(s)).choice(
            len(tc), MAX_CTRL, replace=False))
        tc, kc = tc[pick], kc[pick]
    if len(tr) < MIN_EV:
        return None
    ev = {"rip": dict(t=tr, k=kr), "ctl": dict(t=tc, k=kc)}
    for e in ev.values():
        k = e["k"]
        e["phase"] = np.where(np.isin(phase_i[k], LEARN), "L", "E")
        e["grid"] = occ.cv_group.to_numpy()[k]
        e["state"], e["loc"] = cond_state[k], cond_loc[k]
        e["conj"] = e["loc"] * N_STATE + e["state"]
        e["strat"] = strata(e["t"], tp)
        e["masks"] = set_masks(e["t"], phase_i[k], uncover, away)

    rr = r[r.cell < len(spk[s]["spikes"])]
    regions = {nm: [c for c in rr[rr.roi.isin(m)].cell] for nm, m in REGIONS.items()}
    regions = {k: v for k, v in regions.items() if len(v) >= MIN_CELLS}
    if not regions:
        return None
    out = dict(levels=[], direction=[], binding=[], conj=[], prepost=[],
               binding_pre=[])
    scal = {}                                    # per region: real-label scalars
    for reg, cells in regions.items():
        # windows: in / flank (0.25 s) z-scored together; pre / post (0.45 s)
        cnt = {}
        for nm, e in ev.items():
            t = e["t"]
            cnt[(nm, "in")] = swc.window_counts(spk[s], cells, t - HALF_S, t + HALF_S)
            cnt[(nm, "pre")] = swc.window_counts(spk[s], cells, t + PRE[0], t + PRE[1])
            cnt[(nm, "post")] = swc.window_counts(spk[s], cells, t + POST[0], t + POST[1])
        t = ev["rip"]["t"]
        cnt[("rip", "flank")] = (
            swc.window_counts(spk[s], cells, t - FLANK[1], t - FLANK[0])
            + swc.window_counts(spk[s], cells, t + FLANK[0], t + FLANK[1])) / 2.0
        for w in ("farpre", "farpost"):
            cnt[("rip", w)] = swc.window_counts(spk[s], cells, t + WIN[w][0],
                                                t + WIN[w][1])
        ks = [("rip", "in"), ("ctl", "in"), ("rip", "flank")]
        Z1 = swc.zscore_cells([cnt[x] for x in ks])
        ks2 = [("rip", "pre"), ("rip", "post"), ("ctl", "pre"), ("ctl", "post")]
        Z2 = swc.zscore_cells([cnt[x] for x in ks2])
        Z = dict(zip(ks, Z1))
        Z.update(zip(ks2, Z2))
        # D5: the same pre / post windows z-scored together with the ripple's
        # own far windows, so pre - farpre is on one scale. Kept apart from Z2
        # so that D4a is unchanged.
        ks3 = ks2 + [("rip", "farpre"), ("rip", "farpost")]
        Z3 = swc.zscore_cells([cnt[x] for x in ks3])
        Z.update({(nm, w if w.startswith("far") else w + "5"): z
                  for (nm, w), z in zip(ks3, Z3)})
        # within-phase templates
        tpl = {}
        for ph, phs in (("L", LEARN), ("E", ("execute",))):
            keep = np.isin(phase_i, phs)
            for code, cond, K in (("state", cond_state, N_STATE),
                                  ("loc", cond_loc, N_LOC),
                                  ("conj", cond_conj, N_STATE * N_LOC)):
                if code == "conj" and reg != "HC":
                    continue
                tpl[(ph, code)] = _templates(spk[s], cells, occ,
                                             np.where(keep, cond, -1), K, grids)
            tpl[(ph, "ramp")], tpl[(ph, "sel")] = split_templates(tpl[(ph, "state")])
        codes = [("state", "state", N_STATE), ("ramp", "state", N_STATE),
                 ("sel", "state", N_STATE), ("loc", "loc", N_LOC)] + \
            ([("conj", "conj", N_STATE * N_LOC)] if reg == "HC" else [])
        scal[reg] = {}
        for code, lab_key, K in codes:
            Wp = perms[lab_key]
            for (nm, win), C in Z.items():
                e = ev[nm]
                S = np.full((len(e["t"]), Wp.shape[0]), np.nan)
                for ph in ("L", "E"):
                    m = e["phase"] == ph
                    if not m.any():
                        continue
                    E = _score(C[:, m], e["grid"][m], tpl[(ph, code)])
                    okE = np.isfinite(E).all(1)
                    idx = np.flatnonzero(m)[okE]
                    S[idx] = stat_perm(E[okE], e[lab_key][idx], Wp)
                scal[reg][(code, nm, win)] = S
            # levels (D4c, and conj for D4b-i): per set, matched + own + still;
            # D5: the same in the windows before and after the ripple
            for win, own in (("in", "flank"), ("pre5", "farpre"),
                             ("post5", "farpost")):
                Sr = scal[reg][(code, "rip", win)]
                Sc = scal[reg][(code, "ctl", win)]
                Sf = scal[reg][(code, "rip", own)]
                for key in SETS:
                    rec = dict(session=int(s), region=reg, code=code, set=key,
                               n_cells=len(cells))
                    rec.update(_levels(Sr, Sc, Sf, ev, key))
                    if win != "in":
                        out["prepost"].append(dict(rec, win=win[:-1]))
                    else:
                        (out["conj"] if code == "conj" else
                         out["levels"]).append(rec)
    # D4a direction and D4b (ii), (iii): real-label scalars
    rng = np.random.default_rng(SEED + int(s))
    if "HC" in regions:
        fronts = [f for f in ("frontal", "mPFC", "mOFC") if f in regions]
        for fr in fronts:
            for hc_code in ("state", "ramp", "sel", "loc"):
                for nm in ("rip", "ctl"):
                    e = ev[nm]
                    Fp = scal[fr][("state", nm, "pre")][:, 0]
                    Fq = scal[fr][("state", nm, "post")][:, 0]
                    Hp = scal["HC"][(hc_code, nm, "pre")][:, 0]
                    Hq = scal["HC"][(hc_code, nm, "post")][:, 0]
                    for key in SETS:
                        m = e["masks"][key] & np.isfinite(Fp) & np.isfinite(Fq) \
                            & np.isfinite(Hp) & np.isfinite(Hq)
                        if m.sum() < MIN_EV:
                            continue
                        def asym(fp, hq, hp, fq):
                            r1 = stats.spearmanr(fp, hq)[0]
                            r2 = stats.spearmanr(hp, fq)[0]
                            return np.arctanh(np.clip(r1, -.999, .999)) - \
                                np.arctanh(np.clip(r2, -.999, .999))
                        a = asym(Fp[m], Hq[m], Hp[m], Fq[m])
                        sh = [asym(Fp[m], rng.permutation(Hq[m]),
                                   Hp[m], rng.permutation(Fq[m]))
                              for _ in range(N_SHUF if nm == "rip" else 20)]
                        out["direction"].append(dict(
                            session=int(s), frontal=fr, hc_code=hc_code,
                            events=nm, set=key, n=int(m.sum()), asym=float(a),
                            asym_shuffled=float(np.nanmean(sh))))
        # binding: same window, ripple vs control (matched) and vs the
        # ripple's own baseline (own). D4b in the ripple window; D5 in the
        # window BEFORE the ripple ("are location and sequence co-active
        # before a ripple?")
        pairs = [(f"HC loc x {fr} state", "HC", "loc", fr, "state")
                 for fr in fronts] + [("HC loc x HC state", "HC", "loc", "HC",
                                       "state")]
        fz = lambda r: np.arctanh(np.clip(r, -.999, .999))
        for dest, win, own in (("binding", "in", "flank"),
                               ("binding_pre", "pre5", "farpre")):
            wins = (("rip", win), ("ctl", win), ("rip", own))
            for name, ra, ca, rb, cb in pairs:
                act = {(nm, w): Z_all(ev, spk, s, regions, ra, rb, nm,
                                      w).sum(0) for nm, w in wins}
                for key in SETS:
                    rec = dict(session=int(s), pair=name, set=key)
                    vals = {}
                    for nm, w in wins:
                        x = scal[ra][(ca, nm, w)][:, 0]
                        y = scal[rb][(cb, nm, w)][:, 0]
                        m = ev[nm]["masks"][key] & np.isfinite(x) & np.isfinite(y)
                        if m.sum() >= MIN_EV:
                            vals[(nm, w)] = _partial_r(x[m], y[m],
                                                       act[(nm, w)][m])
                            rec[f"r_{nm}_{w.rstrip('5')}"] = vals[(nm, w)]
                    if ("rip", win) in vals and ("ctl", win) in vals:
                        rec["dz_matched"] = float(fz(vals[("rip", win)])
                                                  - fz(vals[("ctl", win)]))
                    if ("rip", win) in vals and ("rip", own) in vals:
                        rec["dz_own"] = float(fz(vals[("rip", win)])
                                              - fz(vals[("rip", own)]))
                    out[dest].append(rec)
    return out


def Z_all(ev, spk, s, regions, ra, rb, nm, win):
    """Raw counts of both regions' cells in one window type (for the total
    activity covariate)."""
    t = ev[nm]["t"]
    cells = regions[ra] + ([c for c in regions[rb] if c not in regions[ra]])
    if win == "flank":
        return (swc.window_counts(spk[s], cells, t - FLANK[1], t - FLANK[0])
                + swc.window_counts(spk[s], cells, t + FLANK[0],
                                    t + FLANK[1])) / 2.0
    a, b = WIN[win.rstrip("5")]
    return swc.window_counts(spk[s], cells, t + a, t + b)


# ── run ──────────────────────────────────────────────────────────────────
def main():
    deriv = rrsa._derivatives()
    d = os.path.join(deriv, "group", "swr",
                     f"context_content_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    swr = os.path.join(deriv, "group", "swr")
    rip = pd.read_csv(os.path.join(swr, "bundle_v2", "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])
    P = pd.read_csv(os.path.join(swr, "press_categories.csv"))
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)
    rng = np.random.default_rng(SEED)
    P9 = np.vstack([np.arange(9)] + [rng.permutation(9) for _ in range(fs.N_PERM)])
    st_perms = [np.arange(4)] + [np.array(p) for p in
                                 itertools.permutations(range(4))
                                 if list(p) != [0, 1, 2, 3]]
    loc_perms = [np.arange(9)] + [rng.permutation(9) for _ in range(N_PERM_LOC)]
    conj_perms = [np.arange(36)] + [rng.permutation(36) for _ in range(N_PERM_CONJ)]
    perms = {"state": perm_weights(4, st_perms, "state"),
             "loc": perm_weights(9, loc_perms, "loc"),
             "conj": perm_weights(36, conj_perms, "conj")}
    d3_rows, lv, dr, bd, cj, pp, bp = [], [], [], [], [], [], []
    for s in sessions:
        if s not in spk:
            continue
        P_s = P[P.session == s].sort_values("t_s")
        if not len(P_s):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d3_rows += d3_session(s, spk, roi, steps, rip, P_s, P9)
            o = d4_session(s, spk, roi, steps, rip, P_s, perms)
        if o:
            lv += o["levels"]; dr += o["direction"]
            bd += o["binding"]; cj += o["conj"]
            pp += o["prepost"]; bp += o["binding_pre"]
        print(f"  context content s{s:02d}", flush=True)
    T3, L4 = pd.DataFrame(d3_rows), pd.DataFrame(lv)
    D4, B4, C4 = pd.DataFrame(dr), pd.DataFrame(bd), pd.DataFrame(cj)
    P5, B5 = pd.DataFrame(pp), pd.DataFrame(bp)
    for name, T in (("d3_per_session", T3), ("d4c_levels", L4),
                    ("d4a_direction", D4), ("d4b_binding", B4),
                    ("d4b_conjunction", C4), ("d5_prepost_levels", P5),
                    ("d5_binding_pre", B5)):
        T.to_csv(os.path.join(d, f"{name}.csv"), index=False)
    res = summarise(T3, L4, D4, B4, C4, P5, B5)
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(dict(settings=_settings(), tests=res), f, indent=1,
                  default=str)
    from mc.analyse.swr_explore.context_content_figures import draw
    draw(d)
    print(f"-> {d}")
    return d


def _settings():
    return dict(
        analysis="context_content (D3, D4)",
        created=str(datetime.date.today()),
        module="mc/analyse/swr_explore/context_content.py", seed=SEED,
        contexts=dict(uncover=f"(press, press + min(pause, {PAUSE_MAX_S}))",
                      away=f"no uncover press in [t - {AWAY_BEFORE_S}, "
                           f"t + {AWAY_AFTER_S}]"),
        phases="occupancy role table: learning = first_traversal + "
               "encode_rest; execution = execute",
        d3="T4 machinery; REW vs ROUTE / OTHER at equal Manhattan distance "
           "(>= 2) from the current square; matched = T4 stratum controls "
           "of the same context; own = flanks",
        d4=dict(templates="leave-one-configuration-out WITHIN phase; no "
                          "detrending",
                ramp_split=RAMP.tolist(), windows=dict(
                    in_s=[-HALF_S, HALF_S], pre_s=PRE, post_s=POST,
                    flank_s=FLANK), ctrl_pad_s=CTRL_PAD_S,
                max_controls=MAX_CTRL, n_perm_conj=N_PERM_CONJ,
                n_perm_loc=N_PERM_LOC, n_shuffle=N_SHUF,
                state_perms="all 23 non-identity"),
        min_events=MIN_EV, min_cells=MIN_CELLS, regions=REGIONS,
        preregistration="CHANGELOG.md 2026-11-15 D3, D4 (+ amendment)")


def from_saved(d):
    """Summary and figures from the saved per-session tables."""
    T = {n: pd.read_csv(os.path.join(d, f"{n}.csv")) for n in
         ("d3_per_session", "d4c_levels", "d4a_direction", "d4b_binding",
          "d4b_conjunction", "d5_prepost_levels", "d5_binding_pre")}
    res = summarise(*T.values())
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(dict(settings=_settings(), tests=res), f, indent=1,
                  default=str)
    from mc.analyse.swr_explore.context_content_figures import draw
    draw(d)
    return res


def summarise(T3, L4, D4, B4, C4, P5, B5):
    res = {}
    for (roi_, q, key), g in T3.groupby(["roi", "quantity", "set"]):
        for meas in ("z_matched", "z_own", "z_rip", "z_ctrl"):
            if meas in g:
                res[f"D3 | {roi_} | {q} | {key} | {meas}"] = group(g[meas])
    for (roi_, q), g in T3.groupby(["roi", "quantity"]):
        for meas in ("z_matched", "z_own"):
            w = g.pivot_table(index="session", columns="set", values=meas)
            if {"A_E", "A_L"} <= set(w.columns):
                res[f"D3 | {roi_} | {q} | A_E - A_L | {meas}"] = group(
                    (w.A_E - w.A_L).dropna())
    for (reg, code, key), g in L4.groupby(["region", "code", "set"]):
        for meas in ("z_matched", "z_own", "z_rip", "z_still"):
            if meas in g:
                res[f"D4c | {reg} | {code} | {key} | {meas}"] = group(g[meas])
    for key, g in C4.groupby("set"):
        for meas in ("z_matched", "z_own", "z_rip", "z_still"):
            if meas in g:
                res[f"D4b-i | HC conj | {key} | {meas}"] = group(g[meas])
    if len(D4):
        for (fr, hcc, key), g in D4.groupby(["frontal", "hc_code", "set"]):
            rip = g[g.events == "rip"].set_index("session")
            ctl = g[g.events == "ctl"].set_index("session")
            res[f"D4a | {fr} -> HC {hcc} | {key} | own"] = group(
                rip.asym - rip.asym_shuffled)
            both = sorted(set(rip.index) & set(ctl.index))
            res[f"D4a | {fr} -> HC {hcc} | {key} | matched"] = group(
                rip.asym[both] - ctl.asym[both])
        for (fr, hcc), g in D4[D4.events == "rip"].groupby(["frontal", "hc_code"]):
            w = g.assign(v=g.asym - g.asym_shuffled).pivot_table(
                index="session", columns="set", values="v")
            if {"U_L", "A_E"} <= set(w.columns):
                res[f"D4a | {fr} -> HC {hcc} | U_L - A_E | own"] = group(
                    (w.U_L - w.A_E).dropna())
    # region x phase pattern of each code in still moments (pre-registered
    # secondary of D4c): (region L - E) - (HC L - E), within session
    for code in ("sel", "ramp", "state"):
        w = L4[(L4.code == code) & L4.set.isin(["all_L", "all_E"])].pivot_table(
            index="session", columns=["region", "set"], values="z_still")
        for reg in ("HC", "frontal", "mOFC", "mPFC"):
            if (reg, "all_L") in w and (reg, "all_E") in w:
                res[f"D4c | {reg} | {code} | L - E | z_still"] = group(
                    (w[(reg, "all_L")] - w[(reg, "all_E")]).dropna())
                if reg != "HC":
                    res[f"D4c | ({reg} L-E) - (HC L-E) | {code} | z_still"] = \
                        group(((w[(reg, "all_L")] - w[(reg, "all_E")])
                               - (w[("HC", "all_L")] - w[("HC", "all_E")]))
                              .dropna())
    fz = lambda r: np.arctanh(np.clip(r, -.999, .999))
    for tag, B in (("D4b", B4), ("D5 pre", B5)):
        for (pair, key), g in B.groupby(["pair", "set"]):
            for meas in ("dz_matched", "dz_own"):
                if meas in g:
                    res[f"{tag} | {pair} | {key} | {meas}"] = group(g[meas])
            # coupling LEVELS (Fisher z vs 0): ripples and control moments
            for col in [c for c in g if c.startswith("r_")]:
                res[f"{tag} | {pair} | {key} | level {col}"] = group(fz(g[col]))
        # phase-specific binding: learning - execution, within session
        for pair, g in B.groupby("pair"):
            for col in [c for c in g if c.startswith("r_")]:
                w = g.pivot_table(index="session", columns="set", values=col)
                if {"all_L", "all_E"} <= set(w.columns):
                    res[f"{tag} | {pair} | L - E | level {col}"] = group(
                        (fz(w.all_L) - fz(w.all_E)).dropna())
    # D5: code levels before / after the ripple
    for (reg, code, key, win), g in P5.groupby(["region", "code", "set", "win"]):
        for meas in ("z_matched", "z_own", "z_rip", "z_still"):
            if meas in g:
                res[f"D5 {win} | {reg} | {code} | {key} | {meas}"] = group(g[meas])
    return res
