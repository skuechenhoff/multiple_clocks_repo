#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The current-square result of Figure 13, split by phase and by what the square IS.

    python scripts/swr_content_explore.py phase_splits

SK asked for two splits of the SAME quantity Figure 13 plots -- location
evidence in a ripple (C0) and ripple minus its matched flank (C1). This is not
the roles regression: nothing is partialled out, the ripples are simply divided
into subsets and the identical statistic recomputed in each.

PHASES (SK's revised definition)

  first_traversal  the very first traversal of a grid run (`first_trial`), i.e.
                   the discovery pass. Unchanged from earlier figures.
  encode           every traversal BEFORE the first error-free ABCD loop of that
                   grid run, exclusive of it.
  execute          the first error-free loop and everything after.

  ⚠ `encode` is NOT the same as `first_traversal`. The first error-free
  traversal is trial 2 in 73% of runs, but trial 3 or later in the remaining
  27%, so `encode` then spans several traversals. Measured on the real data and
  printed below.

  ⚠ The first error-free loop itself is counted as `execute`. "Until, but
  exclusive of" fixes the encode side; this is the reading that leaves no
  traversal in neither.

SUBTYPES of the square underfoot

  reward / nonreward     is the current square one of the four rewards?
  common / off route     was it visited on at least `COMMON_FRAC` of the
                         error-free traversals of that grid run? This is the
                         "most common trajectory", defined by frequency rather
                         than by any single modal path, which would be brittle.

Every cell is the same estimator as everywhere else: leave-one-configuration-out
templates, per-unit z-scored counts, the ripple's own duration, evidence for the
occupied square minus the mean of the other eight, per-session permutation null,
t across sessions.

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
from mc.analyse.swr_content import phase_labels as _labels

SEED = 42
N_PERM = 200
ROI_SETS = {"HC_all": ["HC_anterior", "HC_mid"],
            "HC_mid": ["HC_mid"], "HC_anterior": ["HC_anterior"]}
COMMON_FRAC = 0.5
MIN_RIPPLES = 20
MIN_CELLS = 2
FLANK_GAP_S = 0.05
MAX_FLANKS = 2
PHASES = ("first_traversal", "encode", "execute")
N_MATCH = 20            # subsamples per count-matched contrast
PHASE_PAIRS = (("first_traversal", "execute"), ("encode", "execute"),
               ("first_traversal", "encode"))
SUBTYPES = ("all", "reward", "nonreward", "common", "off_route")


def main():
    rng = np.random.default_rng(SEED)
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s",
                               "subject_key"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)
    perms = [rng.permutation(9) for _ in range(N_PERM)]

    rows, counts, matched = [], [], []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < MIN_RIPPLES:
            continue
        subj = rip[rip.session == s].subject_key.iloc[0]
        ph_i, first_i, rew_i, com_i = _labels(steps[steps.session == s], occ)
        o = occ.sort_values("start_s")
        a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
        j = np.searchsorted(a, t_rip, side="right") - 1
        jj = np.clip(j, 0, len(a) - 1)
        k = o.index.to_numpy()[jj]
        inside = (j >= 0) & (t_rip <= b[jj])
        if inside.sum() < MIN_RIPPLES:
            continue
        t_rip, d_rip, k = t_rip[inside], d_rip[inside], k[inside]
        half = d_rip / 2.0
        loc = occ["loc"].to_numpy()[k].astype(int)
        grid = occ.cv_group.to_numpy()[k]
        ph, fst, rw, cm = ph_i[k], first_i[k], rew_i[k], com_i[k]
        grids = np.unique(occ.cv_group.to_numpy())

        t_fl, owner = swc.matched_flanks(t_rip, half, occ, other_t=t_rip,
                                         gap_s=FLANK_GAP_S, max_per=MAX_FLANKS)
        if not len(t_fl):
            continue
        w_fl = half[owner]

        for phase in PHASES:
            pm = fst if phase == "first_traversal" else (ph == phase)
            for sub in SUBTYPES:
                sm = {"all": np.ones(len(pm), bool), "reward": rw,
                      "nonreward": ~rw, "common": cm,
                      "off_route": ~cm}[sub]
                counts.append(dict(session=s, phase=phase, subtype=sub,
                                   n=int((pm & sm).sum())))

        for roiname, members in ROI_SETS.items():
            cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                     if c < len(spk[s]["spikes"])]
            if len(cells) < MIN_CELLS:
                continue
            _, loo = swc.build_templates(spk[s], cells, occ, grids)
            C_rip, C_fl = swc.zscore_cells([
                swc.window_counts(spk[s], cells, t_rip - half, t_rip + half),
                swc.window_counts(spk[s], cells, t_fl - w_fl, t_fl + w_fl)])
            E_rip = swc.score_windows(C_rip, t_rip, grid, loo)
            E_fl = swc.score_windows(C_fl, t_fl, grid[owner], loo)
            ok_r = np.isfinite(E_rip).all(axis=1)
            ok_f = np.isfinite(E_fl).all(axis=1)
            li_r, li_f = loc - 1, loc[owner] - 1
            tot_r, tot_f = E_rip.sum(axis=1), E_fl.sum(axis=1)
            ar, af = np.arange(len(E_rip)), np.arange(len(E_fl))

            def stat_for(keep_r, keep_f, p_):
                """(C0, C1) for an arbitrary subset of ripples."""
                hr = E_rip[ar, p_[li_r]]
                sr = hr - (tot_r - hr) / 8.0
                hf = E_fl[af, p_[li_f]]
                sf = hf - (tot_f - hf) / 8.0
                acc = np.zeros(len(E_rip)); cnt = np.zeros(len(E_rip))
                np.add.at(acc, owner[keep_f], sf[keep_f])
                np.add.at(cnt, owner[keep_f], 1.0)
                good = keep_r & (cnt > 0)
                c0 = float(np.mean(sr[keep_r])) if keep_r.any() else np.nan
                c1 = (float(np.mean(sr[good] - acc[good] / cnt[good]))
                      if good.sum() >= MIN_RIPPLES else np.nan)
                return c0, c1

            def z_for(keep_r, keep_f):
                obs = stat_for(keep_r, keep_f, np.arange(9))
                nl = np.array([stat_for(keep_r, keep_f, p_) for p_ in perms],
                              float)
                out = []
                for i in range(2):
                    sd = np.nanstd(nl[:, i])
                    out.append(float((obs[i] - np.nanmean(nl[:, i])) / sd)
                               if sd > 0 else np.nan)
                return out

            # ---- COUNT-MATCHED phase contrasts.
            # ⚠ z = (obs - null_mean)/null_SD and the null SD shrinks as
            # 1/sqrt(n), so z carries a sqrt(n) term. `execute` has ~2x the
            # ripples per session of `encode` (median 704 vs 331) simply
            # because it lasts longer, which inflates ITS z for the same
            # per-ripple effect. Subsampling both phases to the same count
            # removes that term entirely.
            masks = {p_: (fst if p_ == "first_traversal" else (ph == p_))
                     for p_ in PHASES}
            for pa, pb in PHASE_PAIRS:
                ia = np.flatnonzero(ok_r & masks[pa])
                ib = np.flatnonzero(ok_r & masks[pb])
                nmin = min(len(ia), len(ib))
                if nmin < MIN_RIPPLES:
                    continue
                acc = []
                for _ in range(N_MATCH):
                    sa = rng.choice(ia, nmin, replace=False)
                    sb = rng.choice(ib, nmin, replace=False)
                    za, zb = [], []
                    for sel, dest in ((sa, za), (sb, zb)):
                        kr = np.zeros(len(E_rip), bool); kr[sel] = True
                        kf = ok_f & kr[owner]
                        dest.extend(z_for(kr, kf))
                    acc.append([za[0] - zb[0], za[1] - zb[1]])
                acc = np.array(acc, float)
                matched.append(dict(
                    session=s, roi=roiname, pair=f"{pa}_minus_{pb}",
                    n_matched=int(nmin),
                    d_z_c0=float(np.nanmean(acc[:, 0])),
                    d_z_c1=float(np.nanmean(acc[:, 1]))))

            for phase in PHASES:
                pm = fst if phase == "first_traversal" else (ph == phase)
                for sub in SUBTYPES:
                    sm = {"all": np.ones(len(pm), bool), "reward": rw,
                          "nonreward": ~rw, "common": cm,
                          "off_route": ~cm}[sub]
                    keep_r = ok_r & pm & sm
                    keep_f = ok_f & (pm & sm)[owner]
                    if keep_r.sum() < MIN_RIPPLES:
                        continue

                    def stat(p_):
                        hr = E_rip[ar, p_[li_r]]
                        sr = hr - (tot_r - hr) / 8.0
                        hf = E_fl[af, p_[li_f]]
                        sf = hf - (tot_f - hf) / 8.0
                        acc = np.zeros(len(E_rip)); cnt = np.zeros(len(E_rip))
                        np.add.at(acc, owner[keep_f], sf[keep_f])
                        np.add.at(cnt, owner[keep_f], 1.0)
                        good = keep_r & (cnt > 0)
                        c0 = float(np.mean(sr[keep_r]))
                        c1 = (float(np.mean(sr[good] - acc[good] / cnt[good]))
                              if good.sum() >= MIN_RIPPLES else np.nan)
                        return c0, c1, int(good.sum())

                    obs = stat(np.arange(9))
                    nul = np.array([stat(p_)[:2] for p_ in perms], float)
                    out = dict(session=s, subject=subj, roi=roiname,
                               phase=phase, subtype=sub,
                               n_ripples=int(keep_r.sum()),
                               n_paired=obs[2], n_cells=len(cells),
                               c0=obs[0], c1=obs[1])
                    for i, nm in enumerate(("c0", "c1")):
                        sd = np.nanstd(nul[:, i])
                        out[f"z_{nm}"] = (float((obs[i] - np.nanmean(nul[:, i]))
                                                / sd) if sd > 0 else np.nan)
                    rows.append(out)
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    N = pd.DataFrame(counts)
    M = pd.DataFrame(matched)
    d = os.path.join(deriv, "group", "swr",
                     f"location_by_phase_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "per_session.csv"), index=False)
    N.to_csv(os.path.join(d, "ripple_counts.csv"), index=False)
    M.to_csv(os.path.join(d, "count_matched_contrasts.csv"), index=False)
    ov = report(R, N, d, M)
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"), indent=2)
    json.dump(dict(seed=SEED, n_perm=N_PERM, roi_sets=ROI_SETS,
                   common_frac=COMMON_FRAC, min_ripples=MIN_RIPPLES,
                   phases={"first_traversal": "first traversal of a grid run",
                           "encode": "before the first error-free ABCD loop",
                           "execute": "the first error-free loop and after"},
                   statistic="evidence for the occupied square minus the mean "
                             "of the other eight; z against a per-session "
                             "template-label permutation null",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"\n-> {d}")


def _p(v):
    v = pd.Series(v).dropna()
    return stats.ttest_1samp(v, 0)[1] if len(v) >= 5 else np.nan


def report(R, N, d, M=None):
    ov = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
          "tests": {}}
    print("\n=== ripples per phase and subtype (summed over sessions) ===")
    piv = N.groupby(["phase", "subtype"]).n.sum().unstack()
    print(piv.reindex(index=list(PHASES), columns=list(SUBTYPES)).to_string())

    # the contrast the table makes obvious but does not test: does location
    # coding in ripples fall away once the participant knows the route?
    # ALL pairwise phase contrasts, paired within session. ⚠ This is a family
    # of 3 ROIs x 3 phase pairs x 2 measures = 18; the uncorrected p is printed
    # beside the Bonferroni threshold so a lucky one cannot be read as a result.
    print("\n=== PHASE CONTRASTS, paired within session (subtype = all) ===")
    pairs = [("first_traversal", "execute"), ("encode", "execute"),
             ("first_traversal", "encode")]
    n_fam = len(ROI_SETS) * len(pairs) * 2
    print(f"    family = {n_fam} contrasts; Bonferroni threshold "
          f"p < {0.05 / n_fam:.4f}\n")
    print(f"{'ROI':12s} {'contrast':30s} {'meas':5s} {'n':>4s} {'diff':>8s} "
          f"{'p':>9s} {'95% CI':>18s}")
    for roiname in ROI_SETS:
        g0 = R[(R.roi == roiname) & (R.subtype == "all")]
        for pa, pb in pairs:
            for col in ("z_c0", "z_c1"):
                t = g0.pivot_table(index="session", columns="phase", values=col)
                if pa not in t or pb not in t:
                    continue
                dif = (t[pa] - t[pb]).dropna()
                if len(dif) < 5:
                    continue
                pv = _p(dif)
                ci = stats.t.interval(0.95, len(dif) - 1, loc=dif.mean(),
                                      scale=dif.std(ddof=1) / np.sqrt(len(dif)))
                star = " *" if pv < 0.05 else ""
                star += "*" if pv < 0.05 / n_fam else ""
                print(f"{roiname:12s} {pa + ' - ' + pb:30s} {col:5s} "
                      f"{len(dif):4d} {dif.mean():+8.3f} {pv:9.4g} "
                      f"{f'[{ci[0]:+.2f}, {ci[1]:+.2f}]':>18s}{star}")
                ov["tests"][f"{roiname}|{pa}_minus_{pb}|{col}"] = dict(
                    n_sessions=int(len(dif)), diff=float(dif.mean()),
                    p=float(pv), ci95=[float(ci[0]), float(ci[1])],
                    survives_bonferroni=bool(pv < 0.05 / n_fam))

    if M is not None and len(M):
        print("\n=== the SAME contrasts with ripple COUNT matched ===")
        print("    both phases subsampled to the same number of ripples, "
              f"{N_MATCH} draws averaged\n")
        print(f"{'ROI':12s} {'contrast':30s} {'meas':5s} {'n':>4s} "
              f"{'matched':>8s} {'diff':>8s} {'p':>9s} {'95% CI':>18s}")
        for roiname in ROI_SETS:
            for pair in M.pair.unique():
                g = M[(M.roi == roiname) & (M.pair == pair)]
                if len(g) < 5:
                    continue
                for col in ("d_z_c0", "d_z_c1"):
                    v = g[col].dropna()
                    if len(v) < 5:
                        continue
                    pv = _p(v)
                    ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(),
                                          scale=v.std(ddof=1) / np.sqrt(len(v)))
                    print(f"{roiname:12s} {pair:30s} {col[2:]:5s} {len(v):4d} "
                          f"{g.n_matched.median():8.0f} {v.mean():+8.3f} "
                          f"{pv:9.4g} {f'[{ci[0]:+.2f}, {ci[1]:+.2f}]':>18s}")
                    ov["tests"][f"{roiname}|{pair}|{col}|count_matched"] = dict(
                        n_sessions=int(len(v)),
                        median_matched_ripples=float(g.n_matched.median()),
                        diff=float(v.mean()), p=float(pv),
                        ci95=[float(ci[0]), float(ci[1])])

    for roiname in ROI_SETS:
        g0 = R[R.roi == roiname]
        if not len(g0):
            continue
        print(f"\n=== {roiname} ===")
        print(f"{'phase':16s} {'subtype':11s} {'sess':>5s} {'ripples':>8s} | "
              f"{'C0 in ripple':>12s} {'p':>8s} | {'C1 rip-flank':>12s} {'p':>8s}")
        for phase in PHASES:
            for sub in SUBTYPES:
                g = g0[(g0.phase == phase) & (g0.subtype == sub)]
                if len(g) < 5:
                    continue
                print(f"{phase:16s} {sub:11s} {len(g):5d} "
                      f"{int(g.n_ripples.sum()):8d} | "
                      f"{g.z_c0.mean():+12.3f} {_p(g.z_c0):8.3g} | "
                      f"{g.z_c1.mean():+12.3f} {_p(g.z_c1):8.3g}")
                ov["tests"][f"{roiname}|{phase}|{sub}"] = dict(
                    n_sessions=int(len(g)), n_ripples=int(g.n_ripples.sum()),
                    z_c0=float(g.z_c0.mean()), p_c0=float(_p(g.z_c0)),
                    z_c1=float(g.z_c1.mean()), p_c1=float(_p(g.z_c1)))
            # paired within-session contrasts between the subtype pairs
            for sa, sb in (("reward", "nonreward"), ("common", "off_route")):
                w = g0[(g0.phase == phase) & (g0.subtype.isin((sa, sb)))]
                for col in ("z_c0", "z_c1"):
                    t = w.pivot_table(index="session", columns="subtype",
                                      values=col)
                    if sa not in t or sb not in t:
                        continue
                    dif = (t[sa] - t[sb]).dropna()
                    if len(dif) < 5:
                        continue
                    pv = _p(dif)
                    ci = stats.t.interval(0.95, len(dif) - 1, loc=dif.mean(),
                                          scale=dif.std(ddof=1) / np.sqrt(len(dif)))
                    print(f"{'  -> ' + sa + ' - ' + sb:27s} {len(dif):5d} "
                          f"{'':8s} | {col:>12s} {dif.mean():+8.3f} "
                          f"p = {pv:.3g}  95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]")
                    ov["tests"][f"{roiname}|{phase}|{sa}_minus_{sb}|{col}"] = \
                        dict(n_sessions=int(len(dif)), diff=float(dif.mean()),
                             p=float(pv), ci95=[float(ci[0]), float(ci[1])])
    return ov


if __name__ == "__main__":
    main()
