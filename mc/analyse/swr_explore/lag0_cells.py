"""
LAG-0 CELLS AT DISCOVERY: are the cells that hold the CURRENT square during
execution the ones the plan is loaded into, through ripples, when a reward is
found?

    python scripts/swr_content_explore.py lag0_cells

Pre-registered in CHANGELOG.md, 2026-11-14 (F1, F3). Follow-up to claim 2
(`swr/gradient_claims_2026-10-30/`). SK's refinement of N1: whether lag-0
cells are MORE ACTIVE at discovery is not the hypothesis. The question is
whether they are ripple-locked, and whether they are the cells that carry the
reward square at discovery.

lag-0 tuning = `r_lag000_noctrl` (published per-lag table), computed on
execution repeats only. "lag-0 cells" = top 30% within ROI (mPFC, mOFC,
HC_all) -- for display; every test uses the continuous value.

F1   claim 2 a/b in lag-0 cells
     (i)  the published `consolidation` run, unchanged but for the cell filter
          (arrival-locked)
     (ii) reward-locked, the N2 estimator, all cells and top-30% side by side:
          first discovery (FD), repeat correct (RC), incorrect (INC); FD with
          / without a ripple in the post-reward pause. Readings: raw, own
          (post - pre-press), matched (FD vs RC and with vs without, 1:1 on the
          following stillness)
F3a  ripple-locked firing vs lag-0, across mPFC + mOFC cells. Per cell, rate
     around ripples (50 ms bins, -1..+1 s) / mean rate.
       own      minus the ripple's own flanks (0.5-1.0 s both sides)
       matched  minus ripple-free moments of the same period in the same
                stratum of time since the last press x time to the next one
     Sets: first-traversal ripples (PRIMARY), post-FD-pause ripples, execution
     ripples, first traversal - execution. Spearman with lag-0 per bin,
     cluster mass over -0.5..+0.5 s, null = lag-0 permuted within session
     (`discovery_cells.cluster_corr`, the N1 function); and the +-150 ms window.
F3b  per cell, its term of the evidence for the square just discovered:
     own (post - pre) at FD; FD - RC matched; FD with - without a ripple,
     matched. Spearman with lag-0, within-session permutation.

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import datetime
import warnings

import numpy as np
import pandas as pd

from mc.analyse.roi_labels import read_per_cell_table
from scipy import stats

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc
from mc.analyse.swr_explore.consolidation import _stat, _z
from mc.analyse.swr_explore.preferred_lag_firing import PER_CELL
from mc.analyse.swr_explore.discovery_cells import (
    load_inputs, cluster_corr, group, FD_Q, RC_Q, MAX_PAUSE_S, N2_PRIMARY,
    N2_PRE, MIN_EVENTS, MIN_CELLS)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "scripts", "exploratory"))

SEED = 42
N_PERM_LABEL = 1000
N_PERM = 10000
TOP = 0.30
ROI_SETS = {"mPFC": ["mPFC"], "mOFC": ["mOFC"],
            "HC_all": ["HC_anterior", "HC_mid"]}
INC_Q = "kind == 'uncover' and valence == 'error'"
# F3a
R_EDGES = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.05), 3)
R_CENTRES = R_EDGES[:-1] + 0.025
R_FLANK = (0.5, 1.0)
R_TEST = (-0.5, 0.5)
R_WIN = 0.15
MIN_RIP = {"R_FT": 20, "R_FD": 10, "R_EXE": 20}
F3_ROIS = ("mPFC", "mOFC")


# ── cells ────────────────────────────────────────────────────────────────
def lag0_cells():
    """(session, cell) -> lag0, and the top-30% set within each ROI group."""
    T = read_per_cell_table(PER_CELL).dropna(subset=["cell_idx", "subject_id"])
    T = T.assign(session=T.subject_id.astype(int),
                 cell=T.cell_idx.astype(int) - 1,
                 lag0=T.r_lag000_noctrl.astype(float))
    T["roi_set"] = T.roi.map({m: k for k, v in ROI_SETS.items() for m in v})
    T = T.dropna(subset=["roi_set", "lag0"])
    thr = T.groupby("roi_set").lag0.quantile(1 - TOP)
    T["top"] = T.lag0 >= T.roi_set.map(thr)
    keep = set(zip(T.session[T.top], T.cell[T.top]))
    return T, keep, thr.to_dict()


def spearman_within(x, y, g, n_perm=N_PERM):
    """Spearman of x with y across cells; null = x permuted within g."""
    x, y, g = np.asarray(x, float), np.asarray(y, float), np.asarray(g)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], g[ok]
    if len(x) < 6:
        return dict(n=int(len(x)))
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    obs = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.default_rng(SEED)
    blocks = [np.flatnonzero(g == k) for k in np.unique(g)]
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = np.arange(len(x))
        for b in blocks:
            p[b] = rng.permutation(b)
        null[i] = np.corrcoef(rx[p], ry)[0, 1]
    return dict(n=int(len(x)), rho=float(obs),
                p_perm=float((1 + np.sum(np.abs(null) >= abs(obs)))
                             / (1 + n_perm)))


def _match(t_a, s_a, t_b, s_b):
    import scripts.swr_final_ripple_analysis as fin
    if not len(t_a) or not len(t_b):
        return np.empty(0), np.empty(0)
    tm, cm, _ = fin.match_session(t_a, s_a, t_b, s_b, fin.CALIPER_S,
                                  fin.REL_CALIPER, np.random.default_rng(SEED))
    return tm, cm


# ── F1 (ii) and F3b: reward-locked evidence ──────────────────────────────
def cell_terms(C, T_ev, li, P):
    """(n_cells, n_events, n_perm + 1): each cell's term of `_stat`."""
    H = np.take_along_axis(T_ev, P[:, li].T[None], axis=2)
    return (C[:, :, None] * (H - (T_ev.sum(2)[:, :, None] - H) / 8.0)).astype(np.float32)


def evidence(presses, spk, roi, steps, rip, keep):
    rng = np.random.default_rng(SEED)
    P = np.vstack([np.arange(9)] + [rng.permutation(9)
                                    for _ in range(N_PERM_LABEL)])
    rows, per_cell = [], []
    for s in sorted(presses.session.unique()):
        if s not in spk:
            continue
        occ, r, _, _ = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r):
            continue
        blk = occ.groupby("grid_num").grid_id.first()
        g = presses[(presses.session == s) & (presses.kind == "uncover")]\
            .dropna(subset=["loc"])
        t_ev = g.t_s.to_numpy(float)
        loc = g["loc"].to_numpy(float).astype(int)
        grid = g.grid_no.map(blk).to_numpy(float)
        ok = np.isfinite(grid) & (loc >= 1) & (loc <= 9)
        still = g.still_next_s.to_numpy(float)
        M = {"FD": g.eval(FD_Q).to_numpy(), "RC": g.eval(RC_Q).to_numpy(),
             "INC": g.eval(INC_Q).to_numpy()}
        M["FD_rip"] = M["FD"] & (g.post_ripple == 1).to_numpy()
        M["FD_norip"] = M["FD"] & (g.post_ripple == 0).to_numpy()
        pairs = {}
        for name, a, b in (("FD-RC", "FD", "RC"),
                           ("rip-norip", "FD_rip", "FD_norip")):
            ma, mb = M[a] & np.isfinite(still), M[b] & np.isfinite(still)
            tm, cm = _match(t_ev[ma], still[ma], t_ev[mb], still[mb])
            pairs[name] = (np.isin(t_ev, tm), np.isin(t_ev, cm))
        li = np.clip(loc - 1, 0, 8)
        grids = np.unique(occ.cv_group.to_numpy())
        for rname, members in ROI_SETS.items():
            base_cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                          if c < len(spk[s]["spikes"])]
            for cset in ("all", "top30"):
                cl = [c for c in base_cells
                      if cset == "all" or (int(s), int(c)) in keep]
                if len(cl) < MIN_CELLS:
                    continue
                _, loo = swc.build_templates(spk[s], cl, occ, grids)
                T_ev = np.full((len(cl), len(t_ev), 9), np.nan)
                for gr in np.unique(grid[np.isfinite(grid)]):
                    if gr in loo:
                        T_ev[:, grid == gr] = loo[gr][:, None, :]
                good = ok & np.isfinite(T_ev).all((0, 2))
                Tn = np.nan_to_num(T_ev)
                Cpost, = swc.zscore_cells([swc.window_counts(
                    spk[s], cl, t_ev + N2_PRIMARY[0], t_ev + N2_PRIMARY[1])])
                Cpre, = swc.zscore_cells([swc.window_counts(
                    spk[s], cl, t_ev + N2_PRE[0], t_ev + N2_PRE[1])])
                K_post = cell_terms(Cpost, Tn, li, P)
                K_pre = cell_terms(Cpre, Tn, li, P)
                S_post, S_pre = K_post.sum(0), K_pre.sum(0)
                rec = dict(session=int(s), roi=rname, cells=cset,
                           n_cells=len(cl))
                for ev in ("FD", "RC", "INC", "FD_rip", "FD_norip"):
                    m = good & M[ev]
                    rec[f"n_{ev}"] = int(m.sum())
                    if m.sum() >= MIN_EVENTS:
                        rec[f"z_{ev}"] = float(_z(S_post[m].mean(0)))
                        rec[f"z_{ev}_own"] = float(_z((S_post[m]
                                                       - S_pre[m]).mean(0)))
                for name, a, b in (("FD-RC", "FD", "RC"),
                                   ("rip-norip", "FD_rip", "FD_norip")):
                    ma, mb = good & M[a], good & M[b]
                    if ma.sum() >= MIN_EVENTS and mb.sum() >= MIN_EVENTS:
                        rec[f"z_{name}"] = float(_z(S_post[ma].mean(0)
                                                    - S_post[mb].mean(0)))
                        rec[f"z_{name}_own"] = float(_z(
                            (S_post[ma] - S_pre[ma]).mean(0)
                            - (S_post[mb] - S_pre[mb]).mean(0)))
                    pa, pb = good & pairs[name][0], good & pairs[name][1]
                    if pa.sum() >= MIN_EVENTS and pb.sum() >= MIN_EVENTS:
                        rec[f"z_{name}_matched"] = float(_z(
                            S_post[pa].mean(0) - S_post[pb].mean(0)))
                rows.append(rec)
                # F3b: per-cell terms, all cells of mPFC / mOFC
                if cset != "all" or rname not in F3_ROIS:
                    continue
                fd = good & M["FD"]
                if fd.sum() < MIN_EVENTS:
                    continue
                pa, pb = good & pairs["FD-RC"][0], good & pairs["FD-RC"][1]
                ra, rb = good & pairs["rip-norip"][0], \
                    good & pairs["rip-norip"][1]
                for k, c in enumerate(cl):
                    has = np.isfinite(T_ev[k]).all(1)
                    out = dict(session=int(s), cell=int(c), roi=rname,
                               FD_raw=float(_z(K_post[k][fd & has].mean(0))),
                               FD_own=float(_z((K_post[k] - K_pre[k])
                                               [fd & has].mean(0))))
                    if (pa & has).sum() >= MIN_EVENTS and \
                            (pb & has).sum() >= MIN_EVENTS:
                        out["FD-RC_matched"] = float(_z(
                            K_post[k][pa & has].mean(0)
                            - K_post[k][pb & has].mean(0)))
                    if (ra & has).sum() >= MIN_EVENTS and \
                            (rb & has).sum() >= MIN_EVENTS:
                        out["rip-norip_matched"] = float(_z(
                            K_post[k][ra & has].mean(0)
                            - K_post[k][rb & has].mean(0)))
                    per_cell.append(out)
        print(f"  evidence s{s:02d}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(per_cell)


# ── F3a: ripple-locked firing ────────────────────────────────────────────
def rpsth(st, t):
    if not len(t):
        return np.empty((0, len(R_CENTRES)))
    idx = np.searchsorted(st, t[:, None] + R_EDGES[None, :])
    return np.diff(idx, axis=1) / 0.05


def ripple_sets(P, t_rip, beh):
    from swr_hfb_gradient import phase_windows
    from swr_hfb_discovery_lock import _windows_mask
    W = phase_windows(beh)
    per = {"R_FT": [(a, b) for ph, a, b in W.itertuples(index=False)
                    if ph == "first_traversal"],
           "R_EXE": [(a, b) for ph, a, b in W.itertuples(index=False)
                     if ph == "execution"]}
    fd = P.query(FD_Q).dropna(subset=["still_next_s"])
    per["R_FD"] = [(t, t + min(s_, MAX_PAUSE_S)) for t, s_ in
                   zip(fd.t_s.to_numpy(float), fd.still_next_s.to_numpy(float))]
    return {k: t_rip[_windows_mask(t_rip, v)] for k, v in per.items()}, per


def ripple_firing(presses, spk, roi, data, cells):
    from swr_hfb_discovery_lock import strata, control_times, MIN_CTRL
    beh = data["behaviour"]
    rip = data["ripples"]
    flank = (np.abs(R_CENTRES) >= R_FLANK[0]) & (np.abs(R_CENTRES) < R_FLANK[1])
    rows = []
    for s in sorted(presses.session.unique()):
        if s not in spk:
            continue
        P = presses[presses.session == s].sort_values("t_s")
        tp = np.sort(P.t_s.to_numpy(float))
        t_rip = np.unique(np.round(rip[rip.session == s].t_peak_s
                                   .to_numpy(float), 3))
        sets, periods = ripple_sets(P, t_rip, beh[beh.session == s])
        ctl = {k: control_times(periods[k], t_rip) for k in sets}
        st_r = {k: strata(v, tp) for k, v in sets.items()}
        st_c = {k: strata(v, tp) for k, v in ctl.items()}
        r = roi[(roi.session == s) & roi.roi.isin(F3_ROIS)]
        t0, t1 = tp.min(), tp.max()
        for _, c in r.iterrows():
            if c.cell >= len(spk[s]["spikes"]):
                continue
            st = np.sort(np.asarray(spk[s]["spikes"][c.cell], float))
            mu = ((st >= t0) & (st <= t1)).sum() / (t1 - t0)
            if mu <= 0:
                continue
            rec = dict(session=int(s), cell=int(c.cell), roi=c.roi,
                       mean_rate=float(mu))
            for k, t in sets.items():
                if len(t) < MIN_RIP[k]:
                    continue
                R = rpsth(st, t)
                m = R.mean(0)
                rec[f"own_{k}"] = (m - m[flank].mean()) / mu
                C = rpsth(st, ctl[k])
                out, w = [], []
                for q in np.unique(st_r[k]):
                    cq = st_c[k] == q
                    if cq.sum() < MIN_CTRL:
                        continue
                    rq = st_r[k] == q
                    out.append(R[rq].mean(0) - C[cq].mean(0))
                    w.append(rq.sum())
                if sum(w) >= MIN_RIP[k]:
                    w = np.asarray(w, float)
                    rec[f"matched_{k}"] = ((np.stack(out) * w[:, None]).sum(0)
                                           / w.sum()) / mu
                rec[f"n_{k}"] = int(len(t))
            rows.append(rec)
        print(f"  ripple firing s{s:02d}", flush=True)
    T = pd.DataFrame(rows).merge(cells[["session", "cell", "lag0", "top"]],
                                 on=["session", "cell"], how="left")
    for rd in ("own", "matched"):
        a, b = T.get(f"{rd}_R_FT"), T.get(f"{rd}_R_EXE")
        T[f"{rd}_FT-EXE"] = [x - y if isinstance(x, np.ndarray)
                             and isinstance(y, np.ndarray) else np.nan
                             for x, y in zip(a, b)]
    return T


def f3a_tests(T):
    test = (R_CENTRES >= R_TEST[0]) & (R_CENTRES < R_TEST[1])
    win = np.abs(R_CENTRES) <= R_WIN
    res = {}
    for rd in ("own", "matched"):
        for k in ("R_FT", "R_FD", "R_EXE", "FT-EXE"):
            col = f"{rd}_{k}"
            for name, sub in (("mPFC+mOFC", T), ("mPFC", T[T.roi == "mPFC"]),
                              ("mOFC", T[T.roi == "mOFC"])):
                S = sub[sub[col].apply(lambda v: isinstance(v, np.ndarray))
                        & np.isfinite(sub.lag0)]
                if len(S) < 6:
                    continue
                Y = np.stack(S[col])
                r = cluster_corr(S.lag0.to_numpy(float), Y[:, test],
                                 S.session.to_numpy(), n_perm=N_PERM)
                if r is not None:
                    # cluster_corr reports edges on the N1 grid (-1.5 s, 0.1 s
                    # steps); map its bin indices back onto these bins
                    tb = R_EDGES[:-1][test]
                    r["t_s"] = R_CENTRES[test].tolist()
                    for c in r["clusters"]:
                        a = int(round((c["start_s"] + 1.5) / 0.1))
                        b = int(round((c["stop_s"] + 1.5) / 0.1))
                        c["start_s"] = float(tb[a])
                        c["stop_s"] = float(tb[b - 1] + 0.05)
                res[f"{rd} | {k} | lag0 (time) | {name}"] = r
                res[f"{rd} | {k} | lag0 (±150 ms) | {name}"] = spearman_within(
                    S.lag0, Y[:, win].mean(1), S.session)
                res[f"{rd} | {k} | modulation ±150 ms vs 0 | {name}"] = group(
                    pd.Series(Y[:, win].mean(1)).groupby(
                        S.session.to_numpy()).mean().to_numpy())
    return res


# ── run ──────────────────────────────────────────────────────────────────
def main():
    import mc.analyse.swr_explore.consolidation as cons
    deriv = rrsa._derivatives()
    d = os.path.join(deriv, "group", "swr",
                     f"lag0_cells_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    cells, keep, thr = lag0_cells()
    print(f"  top-30% lag-0 thresholds: {thr}; {len(keep)} cells")

    print("\n  F1 (i): the published consolidation run, lag-0 cells only")
    cons.main(keep=keep, out_dir=os.path.join(d, "F1_arrival_locked_top30"))

    data, presses, _ = load_inputs()
    sessions = sorted(presses.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])

    print("\n  F1 (ii) + F3b: reward-locked evidence")
    E, PC = evidence(presses, spk, roi, steps, rip, keep)
    E.to_csv(os.path.join(d, "F1_reward_locked_per_session.csv"), index=False)
    S1 = []
    for (rname, cset), g in E.groupby(["roi", "cells"]):
        for col in [c for c in g.columns if c.startswith("z_")]:
            S1.append(dict(roi=rname, cells=cset, measure=col,
                           **group(g[col].to_numpy(float))))
    S1 = pd.DataFrame(S1)
    S1.to_csv(os.path.join(d, "F1_reward_locked_summary.csv"), index=False)
    PC = PC.merge(cells[["session", "cell", "lag0", "top"]],
                  on=["session", "cell"], how="left")
    PC.to_csv(os.path.join(d, "F3b_per_cell.csv"), index=False)
    f3b = {}
    for col in ("FD_raw", "FD_own", "FD-RC_matched", "rip-norip_matched"):
        for name, sub in (("mPFC+mOFC", PC), ("mPFC", PC[PC.roi == "mPFC"]),
                          ("mOFC", PC[PC.roi == "mOFC"])):
            if col in sub:
                f3b[f"{col} | {name}"] = spearman_within(sub.lag0, sub[col],
                                                         sub.session)

    print("\n  F3a: ripple-locked firing")
    T = ripple_firing(presses, spk, roi, data, cells)
    T.to_pickle(os.path.join(d, "F3a_cells.pkl"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f3a = f3a_tests(T)

    def js(o):
        return o.item() if hasattr(o, "item") else str(o)
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(dict(
            settings=dict(analysis="lag0_cells (F1, F3)",
                          created=str(datetime.date.today()),
                          module="mc/analyse/swr_explore/lag0_cells.py",
                          seed=SEED, top=TOP, thresholds=thr,
                          n_top_cells=len(keep), lag0="r_lag000_noctrl",
                          per_cell_table=PER_CELL, fd=FD_Q, rc=RC_Q,
                          inc=INC_Q, n2_window=N2_PRIMARY, n2_pre=N2_PRE,
                          min_events=MIN_EVENTS, min_cells=MIN_CELLS,
                          f3a=dict(bins_s=0.05, epoch_s=(-1, 1),
                                   flank_s=R_FLANK, test_s=R_TEST,
                                   window_s=R_WIN, min_ripples=MIN_RIP,
                                   matched="ripple-free moments of the same "
                                           "period, same stratum of time since "
                                           "last press x time to next press"),
                          n_perm=N_PERM, n_perm_label=N_PERM_LABEL,
                          preregistration="CHANGELOG.md 2026-11-14 F1, F3"),
            f1_reward_locked=S1.to_dict("records"), f3b=f3b, f3a=f3a),
            f, indent=1, default=js)
    for k, v in f3b.items():
        print(f"   F3b {k:36s} {v}")
    for k, v in f3a.items():
        if "(time)" in k:
            cl = "; ".join(f"{c['start_s']:+.2f}s p={c['p']:.3f}"
                           for c in (v or {}).get("clusters", [])) or "-"
            print(f"   F3a {k:46s} n={(v or {}).get('n')} {cl}")
        else:
            print(f"   F3a {k:46s} {v}")
    from mc.analyse.swr_explore.lag0_cells_figures import draw
    draw(d)
    print(f"-> {d}")
    return d
