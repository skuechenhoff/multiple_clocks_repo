"""
DISCOVERY CELLS: do 'now' (lag-0) cells fire when a reward is found, over
time -- and is that tied to depth and to ripples?

    python scripts/swr_content_explore.py discovery_cells

Pre-registered in CHANGELOG.md, 2026-10-30 (N1, N2). Replaces the two
arrival-locked analyses it grew out of (`storage_vs_planning`, whose lag-0 x
ripple lead was r = +0.15, p = .065, and `consolidation`):

  * events are locked to the REWARD PRESS (bundle press table), not to arrival
    on the square, which comes 0.25-0.5 s earlier;
  * the ripple split uses ripples in the post-reward pause (0, min(pause, 2 s)],
    where the ripple increase lives, not +-0.5 s around arrival;
  * time-resolved with a cluster test, not one fixed window;
  * the moderator test keeps session structure (lag-0 permuted within session).

FD = correct uncovers in the first traversal (first discoveries); RC = correct
uncovers on later repeats of the same block (same square and action, reward
already known).

TWO READINGS, for everything (SK, 2026-10-30):
  own        each event against its own -1.5..-1.0 s baseline (N1, the
             pre-registered primary); for N2, evidence after the press minus
             evidence just before it
  matched    N1: each uncover minus an arrow press followed by the same
             stillness (`fin.matched_sets`), raw; N2: FD against RC uncovers
             matched 1:1 on the stillness that follows

N1  per cell, firing (100 ms bins, -1.5..+2.0 s) / the cell's task-wide mean
    rate; FD - RC. PRIMARY: across mPFC + mOFC cells, Spearman of lag-0 tuning
    (r_lag000_noctrl) with FD - RC per bin, cluster mass over 0-2 s, null =
    10,000 within-session permutations of lag-0 (one function for observed and
    null). Secondary: mPFC, mOFC; FD alone; FD with - without a post-reward
    ripple; MNI z at the site level (null = permuted site z).
N2  the `consolidation` estimator (evidence for the square just uncovered,
    leave-one-configuration-out templates, square-label permutation z),
    reward-locked, in 250 ms windows every 125 ms from -0.5 s; FD, RC and
    FD - RC per ROI; per-cell FD - RC contributions (0.05-1.0 s) against z at
    the site level.

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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "exploratory"))

SEED = 42
N_PERM_CLUSTER = 10000
N_PERM_LABEL = 1000
N_FLIP = 10000
BIN_S, EDGES = 0.1, np.round(np.arange(-1.5, 2.0 + 1e-9, 0.1), 3)
CENTRES = EDGES[:-1] + BIN_S / 2
BASE = (-1.5, -1.0)
TEST = (0.0, 2.0)
MAX_PAUSE_S = 2.0
MIN_EVENTS = 15
ROIS = ("mPFC", "mOFC")
N2_ROIS = {"mPFC": ["mPFC"], "mOFC": ["mOFC"], "HC_all": ["HC_anterior",
                                                         "HC_mid"]}
N2_WIN = 0.25
N2_STARTS = np.round(np.arange(-0.5, 1.75 + 1e-9, 0.125), 3)
N2_PRIMARY = (0.05, 1.0)
N2_PRE = (-0.5, -0.05)            # own baseline for N2: just before the press
MIN_CELLS = 2

FD_Q = "kind == 'uncover' and valence == 'correct' and stage == 'first uncovers'"
RC_Q = "kind == 'uncover' and valence == 'correct' and stage != 'first uncovers'"


# ── inputs ───────────────────────────────────────────────────────────────
def load_inputs():
    from swr_plan_vs_store import load
    data, presses = load()
    presses = presses.sort_values(["session", "t_s"]).reset_index(drop=True)
    rip = data["ripples"]
    # ripple in the post-reward pause, per correct uncover
    presses["post_ripple"] = np.nan
    for s, g in presses.groupby("session"):
        t_r = np.unique(np.round(rip[rip.session == s].t_peak_s.to_numpy(float),
                                 3))
        u = g.kind == "uncover"
        t = g.t_s[u].to_numpy(float)
        w = np.minimum(g.still_next_s[u].to_numpy(float), MAX_PAUSE_S)
        n = (np.searchsorted(t_r, t + w, side="right")
             - np.searchsorted(t_r, t, side="right"))
        presses.loc[g.index[u], "post_ripple"] = np.where(np.isfinite(w),
                                                          (n > 0).astype(float),
                                                          np.nan)
    cells = read_per_cell_table(PER_CELL).dropna(subset=["cell_idx", "subject_id"])
    cells = cells.assign(session=cells.subject_id.astype(int),
                         cell=cells.cell_idx.astype(int) - 1,
                         lag0=cells.r_lag000_noctrl.astype(float))[
        ["session", "cell", "lag0", "MNI_x", "MNI_y", "MNI_z"]]
    return data, presses, cells


def psth(st, t):
    """(n_events, n_bins) rate around each event."""
    if not len(t):
        return np.empty((0, len(CENTRES)))
    idx = np.searchsorted(st, t[:, None] + EDGES[None, :])
    return np.diff(idx, axis=1) / BIN_S


def own(R):
    b = (CENTRES >= BASE[0]) & (CENTRES < BASE[1])
    return R.mean(0) - R.mean(0)[b].mean()


# ── N1 ───────────────────────────────────────────────────────────────────
def n1_table(presses, spk, roi, cells):
    import scripts.swr_final_ripple_analysis as fin
    sets = {"FD": FD_Q, "RC": RC_Q,
            "FD_rip": FD_Q + " and post_ripple == 1",
            "FD_norip": FD_Q + " and post_ripple == 0"}
    matched = {}
    for k, q in sets.items():
        tgt, ctl, _, _ = fin.matched_sets(presses, q, fin.CALIPER_S,
                                          fin.REL_CALIPER, SEED)
        matched[k] = (tgt, ctl)
    rows = []
    for s in sorted(presses.session.unique()):
        if s not in spk:
            continue
        P = presses[presses.session == s]
        r = roi[(roi.session == s) & roi.roi.isin(ROIS)]
        t0, t1 = P.t_s.min(), P.t_s.max()
        ev = {k: P.query(q).t_s.to_numpy(float) for k, q in sets.items()}
        for _, c in r.iterrows():
            if c.cell >= len(spk[s]["spikes"]):
                continue
            st = np.sort(np.asarray(spk[s]["spikes"][c.cell], float))
            mu = ((st >= t0) & (st <= t1)).sum() / (t1 - t0)
            if mu <= 0:
                continue
            rec = dict(session=int(s), cell=int(c.cell), roi=c.roi,
                       mean_rate=float(mu))
            for k, t in ev.items():
                if len(t) >= MIN_EVENTS:
                    rec[f"own_{k}"] = own(psth(st, t)) / mu
                tg, ct = matched[k][0].get(int(s)), matched[k][1].get(int(s))
                if tg is not None and len(tg) >= MIN_EVENTS:
                    rec[f"matched_{k}"] = (psth(st, tg).mean(0)
                                           - psth(st, ct).mean(0)) / mu
            rows.append(rec)
    T = pd.DataFrame(rows).merge(cells, on=["session", "cell"], how="left")
    for rd in ("own", "matched"):
        T[f"{rd}_FD-RC"] = [a - b if isinstance(a, np.ndarray)
                            and isinstance(b, np.ndarray) else np.nan
                            for a, b in zip(T.get(f"{rd}_FD"), T.get(f"{rd}_RC"))]
        T[f"{rd}_rip-norip"] = [a - b if isinstance(a, np.ndarray)
                                and isinstance(b, np.ndarray) else np.nan
                                for a, b in zip(T.get(f"{rd}_FD_rip"),
                                                T.get(f"{rd}_FD_norip"))]
    return T


def _clusters(t, thr, mask):
    out, i, idx = [], 0, np.flatnonzero(mask)
    tt = t[idx]
    while i < len(tt):
        if abs(tt[i]) > thr:
            j = i
            while j + 1 < len(tt) and abs(tt[j + 1]) > thr \
                    and np.sign(tt[j + 1]) == np.sign(tt[i]):
                j += 1
            out.append((idx[i], idx[j] + 1, float(tt[i:j + 1].sum())))
            i = j + 1
        else:
            i += 1
    return out


def cluster_corr(x, Y, groups=None, n_perm=N_PERM_CLUSTER):
    """Spearman of x with each column of Y, cluster mass over TEST. Null:
    x permuted within `groups` (None = freely). One function computes the
    observed statistic and every null draw."""
    ok = np.isfinite(x) & np.isfinite(Y).all(1)
    x, Y = x[ok], Y[ok]
    g = np.zeros(len(x), int) if groups is None else np.asarray(groups)[ok]
    n = len(x)
    if n < 6:
        return None

    def zs(a):
        a = stats.rankdata(a, axis=0)
        return (a - a.mean(0)) / a.std(0)
    zx, ZY = zs(x), zs(Y)
    mask = (CENTRES >= TEST[0]) & (CENTRES < TEST[1]) if Y.shape[1] == len(
        CENTRES) else np.ones(Y.shape[1], bool)
    thr = stats.t.ppf(0.975, n - 2)

    def tvals(zx_):
        r = np.clip(zx_ @ ZY / n, -0.9999, 0.9999)
        return r, r * np.sqrt((n - 2) / (1 - r ** 2))
    rho, t = tvals(zx)
    obs = _clusters(t, thr, mask)
    rng = np.random.default_rng(SEED)
    blocks = [np.flatnonzero(g == k) for k in np.unique(g)]
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = np.arange(n)
        for b in blocks:
            perm[b] = rng.permutation(b)
        _, tp = tvals(zx[perm])
        cl = _clusters(tp, thr, mask)
        null[i] = max([abs(c[2]) for c in cl], default=0.0)
    return dict(n=int(n), rho=rho.tolist(), t=t.tolist(), threshold_t=float(thr),
                clusters=[dict(start_s=float(EDGES[a]), stop_s=float(EDGES[b]),
                               mass=m, sign=int(np.sign(m)),
                               p=float((1 + np.sum(null >= abs(m)))
                                       / (1 + n_perm))) for a, b, m in obs])


def site_mean(T, col):
    T = T.dropna(subset=["MNI_z"])
    T = T[T[col].apply(lambda v: isinstance(v, np.ndarray))]
    T = T.assign(site=T.session.astype(str) + "|" + T.MNI_x.round(1).astype(str)
                 + "|" + T.MNI_y.round(1).astype(str) + "|"
                 + T.MNI_z.round(1).astype(str))
    rows = []
    for site, g in T.groupby("site"):
        rows.append(dict(site=site, session=int(g.session.iloc[0]),
                         roi=g.roi.iloc[0], z=float(g.MNI_z.iloc[0]),
                         n_cells=len(g), Y=np.mean(np.stack(g[col]), 0)))
    return pd.DataFrame(rows)


def n1(T):
    res = {}
    for rd in ("own", "matched"):
        for meas in ("FD-RC", "FD", "rip-norip"):
            col = f"{rd}_{meas}"
            if col not in T:
                continue
            for name, sub in (("mPFC+mOFC", T), ("mPFC", T[T.roi == "mPFC"]),
                              ("mOFC", T[T.roi == "mOFC"])):
                S = sub[sub[col].apply(lambda v: isinstance(v, np.ndarray))
                        & np.isfinite(sub.lag0)]
                if len(S) < 6:
                    continue
                res[f"{rd} | {meas} | lag0 | {name}"] = cluster_corr(
                    S.lag0.to_numpy(float), np.stack(S[col]),
                    S.session.to_numpy())
            st = site_mean(T, col)
            if len(st) >= 6:
                res[f"{rd} | {meas} | z (sites) | mPFC+mOFC"] = cluster_corr(
                    st.z.to_numpy(float), np.stack(st.Y))
                m = st[st.roi == "mPFC"]
                if len(m) >= 6:
                    res[f"{rd} | {meas} | z (sites) | mPFC"] = cluster_corr(
                        m.z.to_numpy(float), np.stack(m.Y))
    return res


# ── N2 ───────────────────────────────────────────────────────────────────
def n2(presses, spk, roi, steps, rip, cells):
    import scripts.swr_final_ripple_analysis as fin
    rng = np.random.default_rng(SEED)
    P = np.vstack([np.arange(9)] + [rng.permutation(9)
                                    for _ in range(N_PERM_LABEL)])
    # stillness-matched FD <-> RC pairs: FD targets, RC as the control pool
    pairs = {}
    for s, g in presses.groupby("session"):
        fd = g.query(FD_Q).dropna(subset=["still_next_s"])
        rc = g.query(RC_Q).dropna(subset=["still_next_s"])
        if len(fd) and len(rc):
            t_m, c_m, _ = fin.match_session(
                fd.t_s.to_numpy(float), fd.still_next_s.to_numpy(float),
                rc.t_s.to_numpy(float), rc.still_next_s.to_numpy(float),
                fin.CALIPER_S, fin.REL_CALIPER, np.random.default_rng(SEED))
            pairs[int(s)] = (t_m, c_m)
    rows, per_cell = [], []
    for s in sorted(presses.session.unique()):
        if s not in spk:
            continue
        occ, r, _, _ = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r):
            continue
        blk = occ.groupby("grid_num").grid_id.first()
        g = presses[(presses.session == s) & (presses.kind == "uncover")
                    & (presses.valence == "correct")].dropna(subset=["loc"])
        t_ev = g.t_s.to_numpy(float)
        loc = g["loc"].to_numpy(float).astype(int)
        grid = g.grid_no.map(blk).to_numpy(float)
        fd = (g.stage == "first uncovers").to_numpy()
        tm, cm = pairs.get(int(s), (np.empty(0), np.empty(0)))
        fd_m, rc_m = np.isin(t_ev, tm), np.isin(t_ev, cm)
        ok = np.isfinite(grid) & (loc >= 1) & (loc <= 9)
        grids = np.unique(occ.cv_group.to_numpy())
        for rname, members in N2_ROIS.items():
            cl = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                  if c < len(spk[s]["spikes"])]
            if len(cl) < MIN_CELLS:
                continue
            _, loo = swc.build_templates(spk[s], cl, occ, grids)

            def S_of(a, b):
                C, = swc.zscore_cells([swc.window_counts(spk[s], cl,
                                                         t_ev + a, t_ev + b)])
                E = swc.score_windows(C, t_ev, grid, loo)
                good = ok & np.isfinite(E).all(1)
                return _stat(E, np.clip(loc - 1, 0, 8), P), good, C, E
            S_pre, g_pre, _, _ = S_of(*N2_PRE)
            wins = [(a, a + N2_WIN) for a in N2_STARTS] + [N2_PRIMARY]
            for a, b in wins:
                S_, good, C, E = S_of(a, b)
                rec = dict(session=int(s), roi=rname, start=a, stop=b,
                           primary=(a, b) == N2_PRIMARY)
                for nm, m in (("FD", fd), ("RC", ~fd)):
                    mm = good & m
                    if mm.sum() >= MIN_EVENTS:
                        rec[f"z_{nm}"] = float(_z(S_[mm].mean(0)))
                        mp = mm & g_pre
                        if mp.sum() >= MIN_EVENTS:
                            rec[f"z_{nm}_own"] = float(
                                _z((S_[mp] - S_pre[mp]).mean(0)))
                if (good & fd).sum() >= MIN_EVENTS and \
                        (good & ~fd).sum() >= MIN_EVENTS:
                    rec["z_FD-RC"] = float(_z(S_[good & fd].mean(0)
                                              - S_[good & ~fd].mean(0)))
                if (good & fd_m).sum() >= MIN_EVENTS and \
                        (good & rc_m).sum() >= MIN_EVENTS:
                    rec["z_FD-RC_matched"] = float(_z(S_[good & fd_m].mean(0)
                                                      - S_[good & rc_m].mean(0)))
                rows.append(rec)
                if (a, b) == N2_PRIMARY and (good & fd).sum() >= MIN_EVENTS \
                        and (good & ~fd).sum() >= MIN_EVENTS:
                    per_cell += _cell_contrib(s, rname, cl, C, t_ev, grid,
                                              loo, loc, good, fd, P)
        print(f"  N2 s{s:02d}", flush=True)
    R = pd.DataFrame(rows)
    PC = pd.DataFrame(per_cell).merge(cells, on=["session", "cell"], how="left")
    return R, PC


def _cell_contrib(s, rname, cl, C, t_ev, grid, loo, loc, good, fd, P):
    """Per cell: its term of the FD - RC statistic, z against the same square
    permutations (the population statistic is the sum of these)."""
    out = []
    li = np.clip(loc - 1, 0, 8)
    for k, c in enumerate(cl):
        T = np.full((len(t_ev), 9), np.nan)
        for gr in np.unique(grid[np.isfinite(grid)]):
            if gr in loo:
                T[grid == gr] = loo[gr][k]
        Tm = np.nan_to_num(T)
        H = np.take_along_axis(Tm, P[:, li].T, axis=1)          # (n, perms)
        contrib = C[k][:, None] * (H - (Tm.sum(1)[:, None] - H) / 8.0)
        okc = good & np.isfinite(T).any(1)
        if (okc & fd).sum() < MIN_EVENTS or (okc & ~fd).sum() < MIN_EVENTS:
            continue
        D = contrib[okc & fd].mean(0) - contrib[okc & ~fd].mean(0)
        out.append(dict(session=int(s), cell=int(c), roi=rname,
                        z_cell=float(_z(D))))
    return out


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


def spearman_perm(x, y, n_perm=N_PERM_CLUSTER):
    ok = np.isfinite(x) & np.isfinite(y)
    rx, ry = stats.rankdata(x[ok]), stats.rankdata(y[ok])
    obs = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.default_rng(SEED)
    null = np.array([np.corrcoef(rng.permutation(rx), ry)[0, 1]
                     for _ in range(n_perm)])
    return dict(n=int(ok.sum()), rho=float(obs),
                p_perm=float((1 + np.sum(np.abs(null) >= abs(obs)))
                             / (1 + n_perm)))


# ── run ──────────────────────────────────────────────────────────────────
def main():
    deriv = rrsa._derivatives()
    d = os.path.join(deriv, "group", "swr",
                     f"discovery_cells_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    data, presses, cells = load_inputs()
    sessions = sorted(presses.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)

    print("  N1: building per-cell responses")
    T = n1_table(presses, spk, roi, cells)
    T.to_pickle(os.path.join(d, "n1_cells.pkl"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        R1 = n1(T)
    for k, r in R1.items():
        if r is None:
            continue
        cl = "; ".join(f"{c['start_s']:+.1f}..{c['stop_s']:+.1f}s "
                       f"{'+' if c['sign'] > 0 else '-'} p={c['p']:.3f}"
                       for c in r["clusters"]) or "no cluster"
        print(f"   {k:46s} n={r['n']:3d}  {cl}")

    print("\n  N2: template evidence, reward-locked")
    steps = swl.load(sessions)
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])
    R2, PC = n2(presses, spk, roi, steps, rip, cells)
    R2.to_csv(os.path.join(d, "n2_per_session.csv"), index=False)
    PC.to_csv(os.path.join(d, "n2_per_cell.csv"), index=False)
    S2 = []
    for (rname, a, b), g in R2.groupby(["roi", "start", "stop"]):
        for col in ("z_FD", "z_RC", "z_FD-RC", "z_FD-RC_matched", "z_FD_own",
                    "z_RC_own"):
            if col in g:
                S2.append(dict(roi=rname, start=a, stop=b, measure=col,
                               **group(g[col].to_numpy(float))))
    S2 = pd.DataFrame(S2)
    S2.to_csv(os.path.join(d, "n2_summary.csv"), index=False)
    prim = S2[(S2.start == N2_PRIMARY[0]) & (S2.stop == N2_PRIMARY[1])]
    print(prim.round(4).to_string(index=False))
    # per-cell contributions against depth, at the site level
    PC = PC.dropna(subset=["MNI_z"])
    PC["site"] = (PC.session.astype(str) + "|" + PC.MNI_z.round(1).astype(str)
                  + "|" + PC.MNI_y.round(1).astype(str))
    sites = PC[PC.roi.isin(ROIS)].groupby(["site", "roi"]).agg(
        z=("MNI_z", "first"), z_cell=("z_cell", "mean"),
        n_cells=("cell", "size")).reset_index()
    sites.to_csv(os.path.join(d, "n2_sites.csv"), index=False)
    depth = {"mPFC+mOFC": spearman_perm(sites.z.to_numpy(float),
                                        sites.z_cell.to_numpy(float))}
    m = sites[sites.roi == "mPFC"]
    depth["mPFC"] = spearman_perm(m.z.to_numpy(float), m.z_cell.to_numpy(float))
    print("  N2 per-cell FD - RC vs depth:", depth)

    def js(o):
        return o.item() if hasattr(o, "item") else str(o)
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(dict(
            settings=dict(analysis="discovery_cells (N1, N2)",
                          created=str(datetime.date.today()),
                          module="mc/analyse/swr_explore/discovery_cells.py",
                          seed=SEED, bins_s=BIN_S, baseline_s=BASE,
                          test_s=TEST, max_pause_s=MAX_PAUSE_S,
                          min_events=MIN_EVENTS,
                          n_perm_cluster=N_PERM_CLUSTER,
                          n_perm_label=N_PERM_LABEL, n2_window_s=N2_WIN,
                          n2_starts=N2_STARTS.tolist(), n2_primary=N2_PRIMARY,
                          n2_pre=N2_PRE, fd=FD_Q, rc=RC_Q,
                          readings=dict(
                              own="event vs its own -1.5..-1.0 s baseline "
                                  "(N1); post-press evidence minus the same "
                                  "event's pre-press evidence (N2)",
                              matched="uncover minus arrow press with the same "
                                      "following stillness (N1); FD vs RC "
                                      "matched 1:1 on stillness (N2)"),
                          preregistration="CHANGELOG.md 2026-10-30 N1, N2"),
            n1=R1, n2_primary_window=prim.to_dict("records"),
            n2_depth=depth), f, indent=1, default=js)
    from mc.analyse.swr_explore.discovery_cells_figures import draw
    draw(d)
    print(f"-> {d}")
    return d
