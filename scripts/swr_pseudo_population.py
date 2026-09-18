#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pseudo-population read-out of location from ripple windows.

    python scripts/swr_pseudo_population.py

SK's question: "for the sessions in which the same tasks are solved, can we
build pseudo populations ... and look at the pattern during the ripples across
the entire ROI, instead of just for the few cells that are recorded for a single
subject?" -- and the reason it should help: "the location signal is carried by
the population, not by individual cells".

WHAT MAKES THIS POSSIBLE, AND WHAT ALMOST BROKE IT

`grid_id` is NOT a shared task identifier. The same `grid_id` carries seven
different reward sets across sessions -- it indexes the configurations WITHIN a
session, which is exactly what leave-one-configuration-out cross-validation
needs and exactly not what pooling across subjects needs. Configurations are
matched here on the reward tuple itself (A-B-C-D), which gives 94 distinct
configurations, of which 8 are run by 28 sessions each and 14 by at least 10.
376 of 450 session-configurations are shared with at least four other sessions.

THE PSEUDO-TRIAL

Cells recorded in different patients never fire at the same moment, so a
population vector has to be assembled: for a configuration and a location, draw
one ripple INDEPENDENTLY per cell from that cell's own session, and stack the
per-cell z-scored spike counts into one vector. Decoding is then the same
estimator the rest of this analysis uses,

    E(L) = sum_c  n_c * z_c(L),   prediction = argmax_L E(L),

with each cell's template estimated with its own session's runs of the tested
configuration held out, so no cell is ever scored against a template that saw
the trials being decoded.

⚠ Independent sampling destroys noise correlations, which is known to INFLATE
pseudo-population decoding. Absolute accuracy here is therefore not a meaningful
estimate of what a real simultaneous population would achieve. What is
meaningful is the comparison between conditions measured the same way:

  the CURVE   accuracy against the number of pooled cells, which is the actual
              answer to "few loud cells or many quiet ones" -- a signal carried
              by a handful saturates immediately, a distributed one keeps
              climbing.
  RIPPLE vs FLANK, paired: each drawn ripple brings its own matched flank
              (same square, same occupancy interval, same width), so the two
              curves use the same cells, the same draws and the same templates
              and differ only in where the window sits.
  BY STATE    ripples split by which reward is being sought (A/B/C/D), at a
              fixed cell count so the comparison is not a cell-count artefact.

Null: the nine template labels are permuted, through the same scoring function,
which puts chance where the data says it is rather than at a nominal 1/9.

Unit of inference: the CONFIGURATION (n = 14 with >= 10 sessions). Configurations
share sessions, so they are not independent, and the per-configuration spread is
descriptive rather than a licence for a t-test -- the permutation null carries
the inference.

@author: Svenja Kuchenhoff
"""

import os
import json
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc
import scripts.swr_place_templates as spt

SEED = 42
N_PERM = 100
N_DRAW = 300              # pseudo-trials per configuration x location
N_SUB = 10                # cell subsamples averaged at each point on the curve
N_AVG = (1, 4, 16)        # ripples averaged per cell per pseudo-trial
MIN_SESSIONS = 10         # configurations pooled across at least this many
MIN_EVENTS = 1            # ripples a cell's session needs at a location to join
CURVE = [2, 4, 8, 16, 32, 64, 128, 256]
FLANK_GAP_S = 0.05
MAX_FLANKS = 2
ROI_SETS = {"HC_all": ["HC_anterior", "HC_mid"],
            "HC_mid": ["HC_mid"], "HC_anterior": ["HC_anterior"],
            "mOFC": ["mOFC"], "mPFC": ["mPFC"]}
LOCATIONS = np.arange(1, 10)


def _key_seed(key):
    """Stable integer from a configuration key -- `hash` is salted per process
    and would make the run irreproducible."""
    return int.from_bytes(key.encode(), "little") % 100_000


def config_key(steps):
    """{(session, grid_id): 'A-B-C-D'} -- the configuration's reward tuple."""
    c = (steps.dropna(subset=["rew_A", "rew_B", "rew_C", "rew_D"])
         .groupby(["session", "grid_id"])[["rew_A", "rew_B", "rew_C", "rew_D"]]
         .first())
    return {k: "-".join(map(str, map(int, v)))
            for k, v in zip(c.index, c.to_numpy())}


# ── 1) collect, per ROI and configuration, one record per cell ─────────

def collect(sessions, spk, roi, steps, rip, keys):
    """pool[roi][config] = list of dicts, one per cell.

    Each record carries the cell's leave-this-configuration-out template and,
    per location, the z-scored ripple counts and the matching flank counts.
    """
    pool = {r: defaultdict(list) for r in ROI_SETS}
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = spt.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or not len(t_rip):
            continue
        st = steps[steps.session == s].sort_values("t_s").reset_index(drop=True)
        o = occ.sort_values("start_s")
        row_of = o.index.to_numpy()
        a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
        j = np.searchsorted(a, t_rip, side="right") - 1
        jj = np.clip(j, 0, len(o) - 1)
        k = row_of[jj]
        inside = (j >= 0) & (t_rip <= b[jj])
        if inside.sum() < 20:
            continue
        t_rip, d_rip, k = t_rip[inside], d_rip[inside], k[inside]
        loc = occ["loc"].to_numpy()[k].astype(int)
        grid = occ.cv_group.to_numpy()[k]
        state = st.state.to_numpy(float)[occ.step_idx.to_numpy()[k]]
        half = d_rip / 2.0

        t_fl, owner = swc.matched_flanks(t_rip, half, occ, other_t=t_rip,
                                         gap_s=FLANK_GAP_S, max_per=MAX_FLANKS)
        if not len(t_fl):
            continue
        w_fl = half[owner]
        grids = np.unique(occ.cv_group.to_numpy())
        cfg = np.array([keys.get((s, g), "") for g in grid])

        for roiname, members in ROI_SETS.items():
            cells = r[r.roi.isin(members)].cell.to_numpy()
            cells = [c for c in cells if c < len(spk[s]["spikes"])]
            if not cells:
                continue
            _, loo = spt.build_templates(spk[s], cells, occ, grids)
            C_rip, C_fl = swc.zscore_cells([
                spt.window_counts(spk[s], cells, t_rip - half, t_rip + half),
                spt.window_counts(spk[s], cells, t_fl - w_fl, t_fl + w_fl)])
            # each ripple's flanks averaged, so ripple and flank are PAIRED and
            # a draw picks the same event in both
            F = np.full_like(C_rip, np.nan)
            cnt = np.zeros(C_rip.shape[1])
            acc = np.zeros_like(C_rip)
            np.add.at(cnt, owner, 1.0)
            np.add.at(acc.T, owner, C_fl.T)
            ok = cnt > 0
            F[:, ok] = acc[:, ok] / cnt[ok]

            for g in grids:
                key = keys.get((s, g), "")
                m = (grid == g) & ok
                T = loo.get(g)
                if not key or T is None or m.sum() < 9:
                    continue
                for ci, c in enumerate(cells):
                    if not np.isfinite(T[ci]).all():
                        continue
                    ev = {}
                    for L in LOCATIONS:
                        sel = m & (loc == L)
                        if sel.sum() >= MIN_EVENTS:
                            ev[L] = (C_rip[ci, sel], F[ci, sel],
                                     state[sel])
                    if len(ev) >= 2:
                        pool[roiname][key].append(
                            dict(session=s, template=T[ci], events=ev))
        print(f"  s{s:02d}", flush=True)
    return pool


# ── 2) decode ──────────────────────────────────────────────────────────

def draw(recs, rng, n_draw, n_avg=1, states=None):
    """Pseudo-trials: (X_rip, X_fl, y, templates) for one configuration.

    For every location, cells whose session has a ripple there each contribute
    `n_avg` randomly drawn events per pseudo-trial, averaged. A cell with
    nothing at that location contributes 0, which is the neutral value for
    z-scored counts and scores exactly 0 evidence -- it abstains rather than
    biasing a location.

    `n_avg` is the question "how many ripples does a read-out need?". At
    n_avg = 1 this is single-ripple decoding, which is the strictest form and
    the one that sits near chance; averaging trades temporal resolution for
    signal without changing anything else about the estimator.
    """
    n_c = len(recs)
    Xr, Xf, y = [], [], []
    for L in LOCATIONS:
        who = [i for i, rc in enumerate(recs) if L in rc["events"]
               and (states is None
                    or np.isin(rc["events"][L][2], states).any())]
        if len(who) < 2:
            continue
        xr = np.zeros((n_draw, n_c)); xf = np.zeros((n_draw, n_c))
        for i in who:
            r_, f_, st_ = recs[i]["events"][L]
            pick = (np.flatnonzero(np.isin(st_, states)) if states is not None
                    else np.arange(len(r_)))
            idx = pick[rng.integers(0, len(pick), (n_draw, n_avg))]
            xr[:, i] = r_[idx].mean(axis=1)
            xf[:, i] = f_[idx].mean(axis=1)
        Xr.append(xr); Xf.append(xf); y.append(np.full(n_draw, L))
    if not Xr:
        return None
    return (np.vstack(Xr), np.vstack(Xf), np.concatenate(y),
            np.array([rc["template"] for rc in recs]))


def scores(X, T, y, perms):
    """(target_minus_others, accuracy) for the true labelling and the null.

    PRIMARY is `target_minus_others`, the statistic Stage 2 and Stage 3 use, so
    the pseudo-population number is on the same scale as the per-session ones.

    Accuracy is kept as a descriptive only. Nine-way argmax is a bad read-out
    here and the reason is structural, not a failure of the pooling: templates
    for neighbouring squares are correlated, so evidence that is clearly
    elevated at the true location can still peak one square away. Measured on
    the largest configuration with every cell's FULL mean profile -- the best
    this design can possibly do -- argmax gets 2 of 9 while the true location
    sits +0.42 SD above the other eight. The continuous statistic sees that;
    argmax throws it away.

    Both come from one matmul: permuting the template's location labels
    reorders the columns of `E = X @ T`, so the null is an index, never a
    rescoring -- the same trick as Stage 3.
    """
    E = X @ T
    rows = np.arange(len(y))
    li = y - 1
    tot = E.sum(axis=1)
    amax = np.argmax(E, axis=1)
    mu = E.mean(axis=1); sd = E.std(axis=1)
    sd = np.where(sd > 0, sd, np.nan)

    def one(perm):
        h = E[rows, perm[li]]
        inv = np.empty(9, int)
        inv[perm] = np.arange(9)
        return (float(np.mean(h - (tot - h) / 8.0)),
                # scale-free: how far the true square stands out WITHIN a
                # pseudo-trial. Unlike the permutation z this grows as the
                # population carries more information, so it is the one that
                # answers "does pooling cells help?"
                float(np.nanmean((h - mu) / sd)),
                float(np.mean(LOCATIONS[inv[amax]] == y)))

    obs = one(np.arange(9))
    nul = np.array([one(pm) for pm in perms])
    return obs, nul


def evidence_matrix(X, T, y):
    """(9, 9) mean within-trial z of the evidence, by true x candidate square.

    Richer than a confusion matrix: it shows WHERE the evidence goes, not only
    whether the winner was right.
    """
    E = X @ T
    Z = (E - E.mean(axis=1, keepdims=True)) / E.std(axis=1, keepdims=True)
    M = np.full((9, 9), np.nan)
    for i, L in enumerate(LOCATIONS):
        m = y == L
        if m.any():
            M[i] = Z[m].mean(axis=0)
    return M


def main():
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s", "subject_key"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)
    keys = config_key(steps)

    print("collecting ...")
    pool = collect(sessions, spk, roi, steps, rip, keys)

    d = os.path.join(deriv, "group", "swr",
                     f"ripple_pseudopop_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)

    rows, conf_rows, state_rows = [], [], []
    for roiname in ROI_SETS:
        for key, recs in pool[roiname].items():
            n_sess = len({rc["session"] for rc in recs})
            if n_sess < MIN_SESSIONS:
                continue
            rng = np.random.default_rng([SEED, _key_seed(key),
                                         list(ROI_SETS).index(roiname)])
            n_cells = len(recs)
            perms = [rng.permutation(9) for _ in range(N_PERM)]
            grid = sorted({c for c in CURVE if c <= n_cells} | {n_cells})

            for m in N_AVG:
                got = draw(recs, rng, N_DRAW, n_avg=m)
                if got is None:
                    continue
                Xr, Xf, y, T = got
                for n in grid:
                    # several cell subsamples per point, because one draw of
                    # `n` cells out of a few hundred is itself a coin flip
                    ar, af, nulls = [], [], []
                    for _ in range(N_SUB if n < n_cells else 1):
                        sub = rng.choice(n_cells, n, replace=False)
                        o1, n1 = scores(Xr[:, sub], T[sub], y, perms)
                        o2, _ = scores(Xf[:, sub], T[sub], y, perms)
                        ar.append(o1); af.append(o2); nulls.append(n1)
                    # the null of the SUBSAMPLE-AVERAGED value, so the z is on
                    # the same footing as the number it standardises
                    N = np.mean(nulls, axis=0)          # (n_perm, 2)
                    A = np.mean(ar, axis=0); B = np.mean(af, axis=0)
                    row = dict(roi=roiname, config=key, sessions=n_sess,
                               n_avg=m, n_cells=n)
                    for qi, q in enumerate(("score", "tz", "acc")):
                        sd = float(N[:, qi].std()); mu = float(N[:, qi].mean())
                        row[f"{q}_ripple"] = float(A[qi])
                        row[f"{q}_flank"] = float(B[qi])
                        row[f"{q}_null"] = mu
                        row[f"z_{q}_ripple"] = ((A[qi] - mu) / sd if sd > 0
                                                else np.nan)
                        row[f"z_{q}_flank"] = ((B[qi] - mu) / sd if sd > 0
                                               else np.nan)
                    rows.append(row)
                if m == N_AVG[-1]:
                    Mr = evidence_matrix(Xr, T, y)
                    Mf = evidence_matrix(Xf, T, y)
                    for i, L in enumerate(LOCATIONS):
                        for jx, P in enumerate(LOCATIONS):
                            conf_rows.append(dict(roi=roiname, config=key,
                                                  true=L, pred=P,
                                                  z_ripple=Mr[i, jx],
                                                  z_flank=Mf[i, jx]))

            # by which reward is being sought, at a fixed cell count so this is
            # not a cell-count effect
            for k in (1, 2, 3, 4):
                gk = draw(recs, rng, N_DRAW, n_avg=N_AVG[-1], states=[k])
                if gk is None:
                    continue
                Xr2, Xf2, y2, T2 = gk
                o1, n1 = scores(Xr2, T2, y2, perms)
                o2, _ = scores(Xf2, T2, y2, perms)
                sd = float(n1[:, 0].std()); mu = float(n1[:, 0].mean())
                state_rows.append(dict(
                    roi=roiname, config=key, state=k, n_cells=n_cells,
                    n_avg=N_AVG[-1], score_ripple=o1[0], score_flank=o2[0],
                    tz_ripple=o1[1], tz_flank=o2[1], score_null=mu,
                    z_ripple=(o1[0] - mu) / sd if sd > 0 else np.nan,
                    z_flank=(o2[0] - mu) / sd if sd > 0 else np.nan))
        print(f"  {roiname} done", flush=True)

    R = pd.DataFrame(rows); C = pd.DataFrame(conf_rows); S = pd.DataFrame(state_rows)
    R.to_csv(os.path.join(d, "pseudopop_curve.csv"), index=False)
    C.to_csv(os.path.join(d, "pseudopop_evidence_matrix.csv"), index=False)
    S.to_csv(os.path.join(d, "pseudopop_by_state.csv"), index=False)
    report(R, S, d)
    json.dump(dict(seed=SEED, n_perm=N_PERM, n_draw=N_DRAW,
                   min_sessions=MIN_SESSIONS, curve=CURVE, roi_sets=ROI_SETS,
                   config="matched on the reward tuple A-B-C-D, NOT grid_id "
                          "(grid_id indexes configurations within a session "
                          "and carries different rewards in different sessions)",
                   templates="leave-this-configuration-out, per cell",
                   caveat="independent per-cell sampling removes noise "
                          "correlations and inflates absolute accuracy; only "
                          "the ripple-vs-flank and cell-count comparisons are "
                          "interpretable",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"\n-> {d}")


def report(R, S, d):
    print("\n=== pseudo-population location signal in ripple windows ===")
    print("    PRIMARY = target minus the other eight squares, z against the")
    print("    template-label null. Accuracy is descriptive only: 9-way argmax")
    print("    is a poor read-out because neighbouring squares have correlated")
    print("    templates (see `scores`).\n")
    for roiname in ROI_SETS:
        g = R[R.roi == roiname]
        if not len(g):
            continue
        print(f"  {roiname}  ({g.config.nunique()} configurations, "
              f"up to {int(g.n_cells.max())} cells)")
        for m, gm in g.groupby("n_avg"):
            print(f"    {int(m)} ripple(s) averaged per cell")
            print(f"      {'cells':>6s} {'tz rip':>7s} {'tz flank':>9s} "
                  f"{'diff':>7s} {'p':>8s} | {'z rip':>6s} {'acc':>6s}")
            for n, v in gm.groupby("n_cells"):
                if len(v) < 3:
                    continue
                dz = (v.tz_ripple - v.tz_flank).dropna()
                p = stats.ttest_1samp(dz, 0)[1] if len(dz) >= 3 else np.nan
                print(f"      {int(n):6d} {v.tz_ripple.mean():+7.3f} "
                      f"{v.tz_flank.mean():+9.3f} {dz.mean():+7.3f} "
                      f"{p:8.3g} | {v.z_score_ripple.mean():+6.2f} "
                      f"{v.acc_ripple.mean():6.3f}")
        print()
    # each configuration's OWN full population, not only the biggest one
    top = R[R.n_cells == R.groupby(["roi", "n_avg", "config"])
            .n_cells.transform("max")]
    print("=== at the full population, "
          f"{N_AVG[-1]} ripples averaged ===")
    for roiname in ROI_SETS:
        v = top[(top.roi == roiname) & (top.n_avg == N_AVG[-1])]
        if len(v) < 3:
            continue
        p = stats.ttest_1samp(v.z_score_ripple.dropna(), 0)[1]
        pd_ = stats.ttest_1samp((v.tz_ripple - v.tz_flank).dropna(), 0)[1]
        print(f"  {roiname:12s} {int(v.n_cells.median()):4d} cells (median), "
              f"{len(v):2d} configs: ripple z {v.z_score_ripple.mean():+.2f} "
              f"(p = {p:.3g});  target_z ripple {v.tz_ripple.mean():+.3f} "
              f"flank {v.tz_flank.mean():+.3f}  diff "
              f"{(v.tz_ripple - v.tz_flank).mean():+.3f} (p = {pd_:.3g})")

    print(f"\n=== by which reward is being sought "
          f"({N_AVG[-1]} averaged, all cells) ===")
    for roiname in ROI_SETS:
        g = S[S.roi == roiname]
        if not len(g):
            continue
        print(f"  {roiname:12s} " + "  ".join(
            f"{'ABCD'[int(k) - 1]} {v.tz_ripple.mean():+.3f}/{v.tz_flank.mean():+.3f}"
            for k, v in g.groupby("state")) + "   (target_z ripple/flank)")

    # Is the decline across the traversal real? Linear slope per configuration.
    # ⚠ configurations share sessions, so this t is anticonservative -- the sign
    # count is the more honest summary and is printed beside it.
    print("\n    linear trend A -> D, slope per configuration:")
    trend = []
    for roiname in ROI_SETS:
        g = S[S.roi == roiname]
        for col in ("tz_ripple", "tz_flank"):
            sl = [np.polyfit(v.sort_values("state").state,
                             v.sort_values("state")[col], 1)[0]
                  for _, v in g.groupby("config") if len(v) == 4]
            if len(sl) < 5:
                continue
            t, p = stats.ttest_1samp(sl, 0)
            neg = sum(x < 0 for x in sl)
            print(f"      {roiname:12s} {col:10s} n = {len(sl):2d} configs, "
                  f"slope {np.mean(sl):+.4f}/state, p = {p:.3g}, "
                  f"{neg}/{len(sl)} negative")
            trend.append(dict(roi=roiname, measure=col, n_configs=len(sl),
                              slope=float(np.mean(sl)), p=float(p),
                              n_negative=neg))
    pd.DataFrame(trend).to_csv(os.path.join(d, "pseudopop_state_trend.csv"),
                               index=False)


if __name__ == "__main__":
    main()
