#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do cells change their lag tuning between DISCOVERY and EXECUTION loops?

Pre-registered in CHANGELOG.md, 2026-11-14 (F2).

The published per-lag tuning (`per_lag_encoding.py`, `r_lag*_noctrl`) is
computed on `all_minus_explore` repeats only -- from the first fully correct
repeat of a configuration onwards. The discovery loops (the first traversal and
every repeat before the first fully correct one) never entered it. Here the
same rate-map estimator is run on each phase separately and ACROSS them.

    discovery  repeats with rep_correct == 0 and correct == 0
    execution  the rest (exactly the published set)

ONE CHANGE TO THE ESTIMATOR, for both phases. The published version averages
each configuration's repeats into one 360-bin trace and takes the per-bin MODE
location. Execution paths are stereotyped, so that is harmless; discovery paths
differ from repeat to repeat, so the mode would blur them. Here a
configuration's rate map pools every (repeat, bin) instead:
    rate[L] = mean firing over all (repeat, bin) whose location at lag k is L
    dwell[L] = number of such (repeat, bin) / number of repeats
(dwell per repeat, so MIN_DWELL_BINS means what it means in the published
pipeline). Execution r is checked against the published `r_lag*_noctrl`.

Per cell, per lag (leave-one-configuration-out, dwell-weighted Pearson):
    r_exe   execution maps of the other configurations -> held-out execution
    r_disc  discovery maps of the other configurations -> held-out discovery
    r_x     EXECUTION maps of the other configurations -> held-out DISCOVERY
    r_x_n   the same execution maps -> k EXECUTION repeats of the held-out
            configuration, k = its number of discovery repeats (10 random
            draws, seed 42 + a stable per-cell offset, r averaged); only
            folds with >= k execution repeats
    change  r_x_n - r_x, on the folds where both exist: same training map,
            same amount of held-out data, so > 0 only if discovery firing
            departs from the execution tuning

PRIMARY (mPFC, lag 0, session = mean Fisher z over the session's cells):
    (a) r_disc > 0       (b) change != 0
Sign-flip over sessions (10,000), the same function for every row.

Usage:
    conda activate env_multiple_clocks
    python scripts/per_lag_encoding_by_phase.py

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import zlib
from datetime import datetime

import numpy as np
import pandas as pd

from mc.analyse.roi_labels import read_per_cell_table
from joblib import Parallel, delayed
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import mc.analyse.helpers_human_cells as hh
from mc.analyse.future_spatial_peaks import _residualise_phase
from mc.analyse.swr_explore.preferred_lag_firing import PER_CELL
from per_lag_encoding import (_weighted_pearson_r, MIN_DWELL_BINS,
                              MIN_SHARED_LOCS, N_LOC, LAGS_DEG, DATA_DIR,
                              ROI_NAME_MAP)

SEED = 42
N_DRAWS = 50
N_FLIP = 10000
MIN_CFG = 5                      # as published
ROIS = ["mPFC", "mOFC", "HC_anterior", "HC_mid", "EC", "PCC"]
GROUPS = {"mPFC": ["mPFC"], "mOFC": ["mOFC"], "HC": ["HC_anterior", "HC_mid"],
          "EC": ["EC"], "PCC": ["PCC"]}
MEASURES = ("r_exe", "r_disc", "r_x", "r_x_n", "change")
RELIABILITY_LAGS = (0, 60)
OUT_BASE = os.path.join(DATA_DIR, "group", "per_lag_encoding_by_phase")


# ── rate maps ────────────────────────────────────────────────────────────
def pooled_map(Y, L, lag):
    """(9,) rate and (9,) dwell-per-repeat over all (repeat, bin) of one
    configuration and phase; location taken `lag` bins ahead in each repeat."""
    loc = np.roll(L, -lag, axis=1)
    rate, dwell = np.full(N_LOC, np.nan), np.zeros(N_LOC)
    ok = np.isfinite(Y)
    for k in range(N_LOC):
        m = (loc == k + 1) & ok
        if m.any():
            rate[k] = Y[m].mean()
            dwell[k] = m.sum() / len(Y)
    return rate, dwell


def predicted(maps, idx):
    R = np.stack([maps[c][0] for c in idx])
    D = np.stack([maps[c][1] for c in idx])
    tot = D.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pred = np.nansum(R * D, 0) / np.where(tot > 0, tot, np.nan)
    return pred, tot


def score(pred, tot, obs, d_obs):
    w = np.minimum(tot, d_obs)
    keep = (w >= MIN_DWELL_BINS) & np.isfinite(pred) & np.isfinite(obs)
    if keep.sum() < MIN_SHARED_LOCS:
        return np.nan
    return _weighted_pearson_r(pred[keep], obs[keep], w[keep])


def cell_lags(Y, L, cfg, disc, seed):
    """All measures at all lags for one cell. Y, L: (n_rep, 360); cfg: config
    id per repeat; disc: True for discovery repeats."""
    rng = np.random.default_rng(seed)
    cfgs = np.unique(cfg)
    rows_e = {c: np.flatnonzero((cfg == c) & ~disc) for c in cfgs}
    rows_d = {c: np.flatnonzero((cfg == c) & disc) for c in cfgs}
    # one fixed set of subsamples per configuration, shared by every lag
    subs = {c: [rng.choice(rows_e[c], len(rows_d[c]), replace=False)
                for _ in range(N_DRAWS)]
            for c in cfgs if 0 < len(rows_d[c]) <= len(rows_e[c])}
    out = {}
    # reliability of the cell property itself: execution r from odd vs even
    # execution repeats of every configuration (independent halves)
    for h in (0, 1):
        for lag in RELIABILITY_LAGS:
            mh = {c: pooled_map(Y[r[h::2]], L[r[h::2]], lag)
                  for c, r in rows_e.items() if len(r) >= 2}
            v = [score(*predicted(mh, [x for x in mh if x != c]), *mh[c])
                 for c in mh if len(mh) > 1]
            v = np.asarray(v, float)
            v = v[np.isfinite(v)]
            out[f"r_exe_half{h + 1}_lag{lag:03d}"] = float(v.mean()) \
                if v.size else np.nan
    for lag in LAGS_DEG:
        me = {c: pooled_map(Y[r], L[r], lag) for c, r in rows_e.items()
              if len(r)}
        md = {c: pooled_map(Y[r], L[r], lag) for c, r in rows_d.items()
              if len(r)}
        f = {m: [] for m in MEASURES}
        for c in cfgs:
            tr_e = [x for x in me if x != c]
            tr_d = [x for x in md if x != c]
            pe = predicted(me, tr_e) if len(tr_e) else None
            if pe is not None and c in me:
                f["r_exe"].append(score(*pe, *me[c]))
            if len(tr_d) and c in md:
                f["r_disc"].append(score(*predicted(md, tr_d), *md[c]))
            if pe is not None and c in md:
                rx = score(*pe, *md[c])
                f["r_x"].append(rx)
                if c in subs:
                    rs = [score(*pe, *pooled_map(Y[s], L[s], lag))
                          for s in subs[c]]
                    rn = np.nanmean(rs) if np.isfinite(rs).any() else np.nan
                    f["r_x_n"].append(rn)
                    f["change"].append(rn - rx)
        for m in MEASURES:
            v = np.asarray(f[m], float)
            v = v[np.isfinite(v)]
            out[f"{m}_lag{lag:03d}"] = float(v.mean()) if v.size else np.nan
            if lag == 0:
                out[f"n_folds_{m}"] = int(v.size)
    return out


# ── per session ──────────────────────────────────────────────────────────
def session_tasks(sub, cells):
    d = hh.load_norm_data(DATA_DIR, [sub], res_data=False)
    if not d:
        return []
    S = d[f"sub-{sub}"]
    beh = S["beh"].reset_index(drop=True)
    L = S["locations"].to_numpy(float)
    disc = ((beh.rep_correct == 0) & (beh.correct == 0)).to_numpy()
    _, cfg = np.unique(beh[["loc_A", "loc_B", "loc_C", "loc_D"]].to_numpy(),
                       axis=0, return_inverse=True)
    cfg = np.asarray(cfg).ravel()
    if len(np.unique(cfg[~disc])) < MIN_CFG:
        return []
    keep = cells[cells.subject_id == int(sub)].set_index("neuron")
    tasks = []
    for nid, df in S["normalised_neurons"].items():
        if nid not in keep.index:
            continue
        X = df.to_numpy(float)
        Y = np.full_like(X, np.nan)
        # phase residualised within each phase subset, as the published
        # pipeline does within its own subset
        for m in (disc, ~disc):
            if m.sum():
                Y[m] = _residualise_phase(X[m], basis="cosine")
        tasks.append(dict(nid=nid, Y=Y, L=L, cfg=cfg, disc=disc,
                          meta=dict(neuron=nid, session=int(sub),
                                    roi=keep.loc[nid, "roi"],
                                    lag0_published=float(
                                        keep.loc[nid, "r_lag000_noctrl"]),
                                    n_disc_reps=int(disc.sum()),
                                    n_exe_reps=int((~disc).sum()),
                                    n_cfg=int(len(np.unique(cfg))))))
    return tasks


def _run(t):
    return {**t["meta"], **cell_lags(t["Y"], t["L"], t["cfg"], t["disc"],
                                     SEED + zlib.crc32(t["nid"].encode()) % 100000)}


# ── stats ────────────────────────────────────────────────────────────────
def fz(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def sign_flip(v):
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


def spearman_within(x, y, g, n_perm=N_FLIP):
    """Across cells, Spearman of x with y; null = x permuted within session."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], np.asarray(g)[ok]
    if len(x) < 6:
        return dict(n=int(len(x)))
    obs = stats.spearmanr(x, y)[0]
    rng = np.random.default_rng(SEED)
    blocks = [np.flatnonzero(g == k) for k in np.unique(g)]
    ry = stats.rankdata(y)
    null = np.empty(n_perm)
    for i in range(n_perm):
        xp = x.copy()
        for b in blocks:
            xp[b] = x[rng.permutation(b)]
        null[i] = np.corrcoef(stats.rankdata(xp), ry)[0, 1]
    return dict(n=int(len(x)), rho=float(obs),
                p_perm=float((1 + np.sum(np.abs(null) >= abs(obs)))
                             / (1 + n_perm)))


def tests(T):
    res, sess_rows = {}, []
    for gname, members in GROUPS.items():
        G = T[T.roi.isin(members)]
        for m in MEASURES:
            for lag in LAGS_DEG:
                col = f"{m}_lag{lag:03d}"
                v = G[col].to_numpy(float)
                val = v if m == "change" else fz(v)
                per_s = pd.Series(val).groupby(G.session.to_numpy()).mean()
                for s, x in per_s.items():
                    sess_rows.append(dict(group=gname, measure=m, lag=lag,
                                          session=int(s), value=float(x)))
                cell = val[np.isfinite(val)]
                res[f"{gname} | {m} | lag {lag}"] = dict(
                    session=sign_flip(per_s.to_numpy()),
                    cell=dict(n=int(cell.size),
                              mean=float(cell.mean()) if cell.size else None,
                              p=float(stats.ttest_1samp(cell, 0)[1])
                              if cell.size > 2 else None))
        # are the execution lag-0 cells the discovery lag-0 cells?
        # (independent repeats; r_disc never shares data with r_exe)
        res[f"{gname} | r_disc vs r_exe, lag 0 (across cells)"] = \
            spearman_within(G.r_exe_lag000.to_numpy(float),
                            G.r_disc_lag000.to_numpy(float), G.session)
        res[f"{gname} | r_disc vs published lag-0 (across cells)"] = \
            spearman_within(G.lag0_published.to_numpy(float),
                            G.r_disc_lag000.to_numpy(float), G.session)
        res[f"{gname} | sanity: pooled r_exe vs published, lag 0"] = \
            spearman_within(G.lag0_published.to_numpy(float),
                            G.r_exe_lag000.to_numpy(float), G.session)
        for lag in RELIABILITY_LAGS:
            res[f"{gname} | reliability: odd vs even execution repeats, "
                f"lag {lag}"] = spearman_within(
                G[f"r_exe_half1_lag{lag:03d}"].to_numpy(float),
                G[f"r_exe_half2_lag{lag:03d}"].to_numpy(float), G.session)
    return res, pd.DataFrame(sess_rows)


# ── run ──────────────────────────────────────────────────────────────────
def main(n_jobs=-1):
    np.random.seed(SEED)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d = os.path.join(OUT_BASE, stamp)
    os.makedirs(d, exist_ok=True)
    cells = read_per_cell_table(PER_CELL)
    cells["roi"] = cells.roi.map(lambda r: ROI_NAME_MAP.get(r, r))
    cells = cells[cells.roi.isin(ROIS)]
    tasks = []
    for sub in sorted(cells.subject_id.unique()):
        t = session_tasks(f"{int(sub):02d}", cells)
        tasks += t
        print(f"  s{int(sub):02d}: {len(t)} cells", flush=True)
    print(f"  {len(tasks)} cells -> fitting")
    rows = Parallel(n_jobs=n_jobs, verbose=2)(delayed(_run)(t) for t in tasks)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(d, "per_cell.csv"), index=False)
    res, S = tests(T)
    S.to_csv(os.path.join(d, "per_session.csv"), index=False)
    for k in ("mPFC | r_disc | lag 0", "mPFC | change | lag 0",
              "mPFC | r_exe | lag 0", "mPFC | r_x | lag 0",
              "mPFC | r_x_n | lag 0",
              "mPFC | change | lag 60",
              "mPFC | r_disc vs r_exe, lag 0 (across cells)",
              "mPFC | sanity: pooled r_exe vs published, lag 0",
              "mPFC | reliability: odd vs even execution repeats, lag 0",
              "HC | reliability: odd vs even execution repeats, lag 0"):
        print(f"  {k:52s} {res[k]}")
    cfg = dict(analysis="per_lag_encoding_by_phase", created=stamp,
               script=os.path.relpath(__file__, REPO), seed=SEED,
               cells=PER_CELL, rois=ROIS, groups=GROUPS,
               discovery="rep_correct == 0 and correct == 0 (first traversal "
                         "and every repeat before the first fully correct)",
               execution="the rest = all_minus_explore (published set)",
               rate_map="pooled over (repeat, bin); dwell per repeat",
               min_dwell_bins=MIN_DWELL_BINS, min_shared_locs=MIN_SHARED_LOCS,
               min_cfg=MIN_CFG, n_draws=N_DRAWS, lags_deg=LAGS_DEG,
               phase_residualise="cosine, within each phase subset",
               measures=dict(r_exe="LOCO within execution",
                             r_disc="LOCO within discovery",
                             r_x="execution train -> discovery held-out",
                             r_x_n="execution train -> k execution repeats "
                                   "held-out, k = n discovery repeats",
                             change="r_x_n - r_x, matched folds"),
               unit="session (mean Fisher z over cells); change in raw r",
               preregistration="CHANGELOG.md 2026-11-14 F2")
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(dict(settings=cfg, tests=res), f, indent=1, default=str)
    from per_lag_encoding_by_phase_figures import draw
    draw(d)
    print(f"-> {d}")
    return d


if __name__ == "__main__":
    main()
