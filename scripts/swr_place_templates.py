#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1a + Stage 2 of the ripple-content analysis.

    python scripts/swr_place_templates.py run

Stage 1a -- build a location template per cell from the REAL-TIME location
timeline (`mc.analyse.swr_location`) and raw spike times, leave-one-grid-out,
and measure how reliable those templates are (split-half over odd/even grids).

Stage 2 -- THE POSITIVE CONTROL, and the gate for everything that follows.
Score ordinary navigation windows, drawn at the SAME widths as that session's
real ripples and at least 1 s away from any ripple. If `E(current location)`
does not exceed the other eight locations HERE, on plentiful ordinary data,
then nothing measured at ripple time can mean anything.

The statistic is `target_minus_others` per window, averaged within session, then
tested across sessions -- session is the unit of inference, as everywhere else
in this project. The null permutes the 9 location labels of every template and
recomputes THE SAME statistic through the same code path (CLAUDE.md rule 4).

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import datetime

import numpy as np
import pandas as pd
from scipy import stats

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc

SEED = 42
N_PERM = 200
ROIS = ["HC_anterior", "HC_mid", "mPFC", "mOFC", "PCC"]
AVOID_PAD_S = 1.0          # navigation windows must be this far from any ripple


def out_dir():
    d = os.path.join(rrsa._derivatives(), "group", "swr",
                     f"ripple_content_templates_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    return d


def session_data(s, spk, roi, steps, rip):
    occ = swc.occupancy(steps, s)
    r = roi[roi.session == s]
    r = r[r.cell < len(spk[s]["spikes"])]
    g = rip[rip.session == s]
    t = np.sort(g.t_peak_s.to_numpy(float))
    d = g.duration_s.to_numpy(float)[np.argsort(g.t_peak_s.to_numpy(float))]
    keep = np.concatenate([[True], np.diff(t) > 0.05]) if len(t) else np.zeros(0, bool)
    return occ, r, t[keep], d[keep]


def build_templates(spk_s, cells, occ, grids):
    """(n_cells, 9) full template, and {grid: (n_cells, 9)} leave-one-out."""
    full = np.array([swc.zscore_map(swc.place_map(spk_s["spikes"][c], occ))
                     for c in cells])
    loo = {}
    for gr in grids:
        loo[gr] = np.array([
            swc.zscore_map(swc.place_map(spk_s["spikes"][c], occ, exclude_grid=gr))
            for c in cells])
    return full, loo


def reliability(spk_s, cells, occ, exclude_grid=None):
    """Split-half correlation of the raw place map, split by CONFIGURATION.

    Splitting on grid_num would put different runs of the same configuration in
    opposite halves, which inflates the correlation -- the two halves would
    share reward locations and trajectories.

    `exclude_grid` drops one configuration first, so that a reliability used to
    WEIGHT cells when scoring that configuration never saw it. For the
    ripple-minus-flank contrast the weights cancel either way, but C0 is scored
    against a permutation null and would otherwise be mildly optimistic.
    """
    if exclude_grid is not None:
        occ = occ[occ.cv_group != exclude_grid]
    gr = occ.cv_group.to_numpy()
    out = []
    for c in cells:
        a = swc.place_map(spk_s["spikes"][c], occ[gr % 2 == 0])
        b = swc.place_map(spk_s["spikes"][c], occ[gr % 2 == 1])
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 6 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            out.append(np.nan)
        else:
            out.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return np.array(out)


# How a cell's location tuning becomes its say in the read-out. `thresh_*` is
# the sweep from CHANGELOG 2026-09-17 (m) expressed as a 0/1 weight, so that
# thresholding and weighting run through one code path and are comparable.
WEIGHTS = {
    "all": lambda rel: np.ones_like(rel),
    "thresh_0.1": lambda rel: (rel >= 0.1).astype(float),
    "thresh_0.2": lambda rel: (rel >= 0.2).astype(float),
    "linear": lambda rel: np.clip(rel, 0, None),
    "square": lambda rel: np.clip(rel, 0, None) ** 2,
}


def weighted_loo(spk_s, cells, occ, grids, scheme):
    """{grid: (n_cells, 9) template}, each cell scaled by its location tuning.

    Scaling a cell's TEMPLATE is exactly weighting that cell in
    `E(L) = sum_c n_c z_c(L)`, so no other part of the estimator changes and
    `scheme = "all"` reproduces the unweighted analysis bit for bit.

    A cell with a negative split-half correlation is given weight zero, not a
    negative weight: an unreliable map is no evidence, not evidence against.
    Reliability is recomputed for every held-out configuration.
    """
    f = WEIGHTS[scheme]
    out = {}
    for gr in grids:
        T = np.array([swc.zscore_map(swc.place_map(spk_s["spikes"][c], occ,
                                                   exclude_grid=gr))
                      for c in cells])
        if scheme == "all":
            out[gr] = T
            continue
        rel = reliability(spk_s, cells, occ, exclude_grid=gr)
        w = f(np.nan_to_num(rel, nan=0.0))
        out[gr] = T * w[:, None]
    return out


def window_counts(spk_s, cells, t0, t1):
    """(n_cells, n_windows) spike counts -- invariant to label permutations, so
    computed ONCE and reused by every null draw."""
    return np.array([swc.count_in(spk_s["spikes"][c], t0, t1) for c in cells],
                    float)


def score_windows(C, t0, win_grid, loo, perm=None):
    """E per window, from precomputed counts."""
    E = np.full((len(t0), 9), np.nan)
    for gr in np.unique(win_grid):
        m = win_grid == gr
        T = loo.get(gr)
        if T is None:
            continue
        if perm is not None:
            T = T[:, perm]
        E[m] = swc.evidence_matrix(C[:, m], T)
    return E


def main():
    rng = np.random.default_rng(SEED)
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])
    sessions = sorted(rip.session.unique().tolist())
    print(f"loading {len(sessions)} sessions ...")
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)

    rel_rows, sess_rows, null_rows = [], [], []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or not len(t_rip):
            continue
        grids = np.unique(occ.cv_group.to_numpy())

        # --- navigation windows: same widths as this session's ripples -------
        n_want = len(t_rip)
        w = rng.choice(d_rip, size=n_want)
        t0, loc = swc.random_windows(occ, float(np.median(d_rip)), n_want, rng,
                                     avoid=t_rip, avoid_pad_s=AVOID_PAD_S)
        if len(t0) < 50:
            continue
        w = rng.choice(d_rip, size=len(t0))
        t1 = t0 + w
        # which grid does each window belong to
        o = occ.sort_values("start_s")
        j = np.searchsorted(o.start_s.to_numpy(), t0, side="right") - 1
        win_grid = o.cv_group.to_numpy()[np.clip(j, 0, len(o) - 1)]

        for roiname in ROIS:
            cells = r[r.roi == roiname].cell.to_numpy()
            if len(cells) < 1:
                continue
            full, loo = build_templates(spk[s], cells, occ, grids)
            rel = reliability(spk[s], cells, occ)
            for c, rr in zip(cells, rel):
                rel_rows.append(dict(session=s, roi=roiname, cell=int(c), r=rr))

            C = window_counts(spk[s], cells, t0, t1)
            E = score_windows(C, t0, win_grid, loo)
            sc = swc.target_minus_others(E, loc)
            ok = np.isfinite(sc)
            if ok.sum() < 50:
                continue
            acc = np.nanmean(np.nanargmax(E[ok], 1) + 1 == loc[ok])
            sess_rows.append(dict(session=s, roi=roiname, n_cells=len(cells),
                                  n_windows=int(ok.sum()),
                                  score=float(np.nanmean(sc[ok])),
                                  acc=float(acc),
                                  mean_rel=float(np.nanmean(rel))))
            for p in range(N_PERM):
                perm = rng.permutation(9)
                Ep = score_windows(C, t0, win_grid, loo, perm=perm)
                scp = swc.target_minus_others(Ep, loc)
                null_rows.append(dict(session=s, roi=roiname, perm=p,
                                      score=float(np.nanmean(scp[np.isfinite(scp)]))))
        print(f"  s{s:02d}: {len(t0)} windows, {len(r)} cells", flush=True)

    REL = pd.DataFrame(rel_rows)
    S = pd.DataFrame(sess_rows)
    N = pd.DataFrame(null_rows)
    d = out_dir()
    REL.to_csv(os.path.join(d, "template_reliability.csv"), index=False)
    S.to_csv(os.path.join(d, "session_scores.csv"), index=False)
    N.to_csv(os.path.join(d, "session_null.csv"), index=False)

    print("\n=== Stage 1a: template reliability (split-half, odd vs even grids) ===")
    for roiname in ROIS:
        v = REL[REL.roi == roiname].r.dropna()
        if len(v) < 5:
            continue
        t, p = stats.ttest_1samp(v, 0)
        print(f"  {roiname:12s} n={len(v):4d}  mean r={v.mean():+.4f}  "
              f"t={t:+.2f}  p={p:.3g}")

    print("\n=== Stage 2: POSITIVE CONTROL, navigation windows (non-ripple) ===")
    print(f"{'ROI':12s} {'sess':>5s} {'cells':>6s} {'windows':>8s} "
          f"{'score':>9s} {'t':>7s} {'p':>9s} {'perm p':>8s} {'acc':>6s}")
    summary = []
    for roiname in ROIS:
        s_ = S[S.roi == roiname]
        if len(s_) < 5:
            continue
        t, p = stats.ttest_1samp(s_.score, 0)
        nu = N[N.roi == roiname].groupby("perm").score.mean()
        obs = s_.score.mean()
        pp = float((np.abs(nu) >= abs(obs)).mean()) if len(nu) else np.nan
        print(f"{roiname:12s} {s_.session.nunique():5d} {s_.n_cells.sum():6d} "
              f"{s_.n_windows.sum():8d} {obs:+9.4f} {t:+7.2f} {p:9.3g} "
              f"{pp:8.3f} {s_.acc.mean():6.3f}")
        summary.append(dict(roi=roiname, sessions=int(s_.session.nunique()),
                            cells=int(s_.n_cells.sum()),
                            windows=int(s_.n_windows.sum()),
                            score=obs, t=float(t), p=float(p), perm_p=pp,
                            acc=float(s_.acc.mean())))
    pd.DataFrame(summary).to_csv(os.path.join(d, "stage2_summary.csv"), index=False)
    json.dump(dict(seed=SEED, n_perm=N_PERM, rois=ROIS,
                   avoid_pad_s=AVOID_PAD_S,
                   window_widths="sampled from the session's own ripple durations",
                   cv="leave-one-grid-out templates",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"\n-> {d}")


if __name__ == "__main__":
    main()
