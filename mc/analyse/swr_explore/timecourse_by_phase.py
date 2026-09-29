#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPLORATORY: the peri-ripple location time course, split into ENCODE/EXECUTE.

⚠ NOT the reported analysis. The reported phase contrast is
`mc/analyse/swr_phase_contrast.py` (ripple window only). This module is kept
for the descriptive time course and its cluster test only; its fixed-window
numbers differ slightly because here the per-unit count normalisation pools
across all 41 offsets.

    python scripts/swr_content_explore.py timecourse_by_phase

This is the main runner's `timecourse` step (41 offsets, -500..+500 ms, each
ripple's own duration as the window, template-label permutation null, session
as the unit of inference) run separately inside each phase, so the encoding
effect can be shown in the same temporal frame as the reported panel.

⚠ Ripple COUNT, not rate, is the confound between phases. z is
(obs - null_mean) / null_SD and the null SD shrinks as 1/sqrt(n), so z carries
a sqrt(n) term. `execute` has ~2x the ripples per session of `encode` purely
because it lasts longer, which inflates ITS z for the same per-ripple effect.
Every curve here is therefore computed on subsamples of both phases drawn down
to the SAME count (N_MATCH draws, averaged). The unmatched curves are saved
alongside so the size of the correction stays visible.

The same ripple set is used at every offset (intersection of the per-offset
finite-evidence masks), so a change across the time course is a change in the
signal and not in which ripples contributed.

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
N_PERM = 1000
N_MATCH = 20            # draws for the DISPLAY time course (41 offsets)
N_MATCH_SCALAR = 1000   # draws for the reported fixed-window contrast
SCALAR_WINDOWS = {"ripple": 0.0, "pm250ms": 0.25, "far": None}
#   "far" is the complement |t| >= FAR_FIELD_S: the surround baseline
ROI_SETS = {"HC_mid": ["HC_mid"], "HC_anterior": ["HC_anterior"],
            "HC_all": ["HC_anterior", "HC_mid"]}
OFFSETS = np.round(np.arange(-0.50, 0.5001, 0.025), 4)
PHASES = ("encode", "execute")
MIN_RIPPLES = 20
MIN_CELLS = 2
SMOOTH = 3              # offsets in the display-only moving average (75 ms)
FAR_FIELD_S = 0.35      # |t| beyond this is the "surround" baseline
N_CLUST = 10000         # sign-flip permutations for the cluster test


def _smooth(y, k=SMOOTH):
    """Centred k-point moving average along the last axis, edges shortened."""
    pad = k // 2
    yp = np.pad(y, [(0, 0)] * (y.ndim - 1) + [(pad, pad)], mode="edge")
    ker = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(v, ker, "valid"), -1, yp)


def summarise(d):
    """Cluster stats from the saved per-session time courses.

    Split out of `main` so the tests can be re-run (or extended) without
    repeating the spike-counting loop.
    """
    import scripts.swr_content_main as main_run   # same cluster test as the panel

    R = pd.read_csv(os.path.join(d, "timecourse_by_phase.csv"))
    print("\n=== PERI-RIPPLE LOCATION SIGNAL, BY PHASE "
          "(ripple counts matched within session) ===")
    print(f"    session = unit of inference; cluster-corrected over "
          f"{len(OFFSETS)} offsets\n")
    summ, ext = [], []
    ov = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
          "tests": {}}
    for ri, roiname in enumerate(ROI_SETS):
        W = {p_: (R[(R.roi == roiname) & (R.phase == p_)]
                  .pivot_table(index="session", columns="offset_s", values="z"))
             for p_ in PHASES}
        common = W["encode"].dropna().index.intersection(
            W["execute"].dropna().index)
        if len(common) < 5:
            continue
        print(f"  {roiname:12s} n = {len(common)} sessions")
        off = W["encode"].columns.to_numpy()
        far = np.abs(off) >= FAR_FIELD_S
        curves = (PHASES + ("encode_minus_execute",)
                  + tuple(f"{q}_vs_surround" for q in PHASES))
        for ci, p_ in enumerate(curves):
            if p_ in PHASES:
                M = W[p_].loc[common].to_numpy()
            elif p_.endswith("_vs_surround"):
                # each session's own peri-ripple far field removed: does the
                # signal PEAK at the ripple, or is it just present throughout?
                M = W[p_[:-len("_vs_surround")]].loc[common].to_numpy()
                M = M - M[:, far].mean(axis=1, keepdims=True)
            else:
                M = (W["encode"].loc[common] - W["execute"].loc[common]).to_numpy()
            # seeded per ROI and per curve: a shared generator would make
            # every curve's p depend on which curves were computed before it
            # (0.048 vs 0.051 for the same data, observed).
            t_obs, mass, p_cl, thr = main_run._cluster_test(
                M, np.random.default_rng([SEED, ri, ci]), n_clust=N_CLUST)
            i0 = int(np.argmin(np.abs(off)))
            t0, p0 = stats.ttest_1samp(M[:, i0], 0)
            print(f"      {p_:22s} at peak z = {M[:, i0].mean():+.3f} "
                  f"(p = {p0:.3f}); largest cluster mass {mass:6.1f}, "
                  f"p = {p_cl:.3f}")
            e = main_run._extents(t_obs, off, thr, mass)
            summ.append(dict(roi=roiname, curve=p_, n_sessions=len(common),
                             z_at_zero=float(M[:, i0].mean()),
                             t_at_zero=float(t0), p_at_zero=float(p0),
                             cluster_mass=float(mass), p_cluster=float(p_cl)))
            ov["tests"][f"{roiname}|{p_}"] = summ[-1]
            for ee in e:
                ext.append(dict(roi=roiname, curve=p_, p_cluster=float(p_cl),
                                **ee))
        print()
    pd.DataFrame(summ).to_csv(os.path.join(d, "cluster_summary.csv"),
                              index=False)
    pd.DataFrame(ext).to_csv(os.path.join(d, "cluster_extents.csv"),
                             index=False)
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"), indent=2)


def summarise_scalar(d):
    """The REPORTED contrast: one fixed window, t-test across sessions.

    Also reports how far the averaged contrast had converged after 20, 100 and
    1000 subsample draws, which is what says whether the draw count matters.
    """
    F = pd.read_csv(os.path.join(d, "fixed_window_contrast.csv"))
    print("\n=== FIXED-WINDOW PHASE CONTRAST (counts matched within session) ===\n")
    out, ov = [], {}
    for w in F.window.unique():
        print(f"  --- window: {w} ---")
        print(f"      {'ROI':12s} {'n':>3s} {'n/phase':>8s} {'encode':>8s} "
              f"{'execute':>8s} {'diff':>8s} {'t':>7s} {'P':>9s} "
              f"{'95% CI':>18s} {'draw SD':>8s}")
        for roiname in ROI_SETS:
            g = F[(F.roi == roiname) & (F.window == w)]
            v = g.d_z.dropna()
            if len(v) < 5:
                continue
            t, p = stats.ttest_1samp(v, 0)
            ci = stats.t.interval(.95, len(v) - 1, v.mean(),
                                  stats.sem(v))
            print(f"      {roiname:12s} {len(v):3d} {g.n_matched.median():8.0f} "
                  f"{g.z_encode.mean():+8.3f} {g.z_execute.mean():+8.3f} "
                  f"{v.mean():+8.3f} {t:+7.2f} {p:9.4g} "
                  f"[{ci[0]:+.2f}, {ci[1]:+.2f}]  {g.d_z_sd_across_draws.mean():7.3f}")
            extra = {}
            for cname in ("z_encode", "z_execute", "z_encode_minus_surround",
                          "z_execute_minus_surround"):
                if cname not in g.columns:
                    continue
                vv = g[cname].dropna()
                if len(vv) < 5:
                    continue
                tt, pp = stats.ttest_1samp(vv, 0)
                cc = stats.t.interval(.95, len(vv) - 1, vv.mean(),
                                      stats.sem(vv))
                extra[cname] = dict(mean=float(vv.mean()), t=float(tt),
                                    p=float(pp),
                                    ci95=[float(cc[0]), float(cc[1])])
                print(f"          {cname:26s} {vv.mean():+7.3f} "
                      f"t({len(vv) - 1}) = {tt:+5.2f}  P = {pp:.4g}  "
                      f"[{cc[0]:+.2f}, {cc[1]:+.2f}]")
            rec = dict(roi=roiname, window=w, n_sessions=int(len(v)),
                       per_curve=extra,
                       median_matched_ripples=float(g.n_matched.median()),
                       z_encode=float(g.z_encode.mean()),
                       z_execute=float(g.z_execute.mean()),
                       diff=float(v.mean()), t=float(t), p=float(p),
                       ci95=[float(ci[0]), float(ci[1])],
                       mean_sd_across_draws=float(g.d_z_sd_across_draws.mean()))
            for n_d in (20, 100, 1000):
                vv = g[f"d_z_at_{n_d}"].dropna()
                tt, pp = stats.ttest_1samp(vv, 0)
                rec[f"diff_at_{n_d}_draws"] = float(vv.mean())
                rec[f"p_at_{n_d}_draws"] = float(pp)
            out.append(rec)
            ov[f"{roiname}|{w}"] = rec
        print()
    print("  --- does the number of draws matter? ---")
    print(f"      {'ROI':12s} {'window':9s} {'20 draws':>18s} "
          f"{'100 draws':>18s} {'1000 draws':>18s}")
    for rec in out:
        cells = "".join(
            f"{rec[f'diff_at_{n}_draws']:+9.3f} (p={rec[f'p_at_{n}_draws']:.3f})"
            for n in (20, 100, 1000))
        print(f"      {rec['roi']:12s} {rec['window']:9s} {cells}")
    pd.DataFrame(out).to_csv(os.path.join(d, "fixed_window_summary.csv"),
                             index=False)
    json.dump(ov, open(os.path.join(d, "fixed_window_overview.json"), "w"),
              indent=2)


def main():
    import scripts.swr_content_main as main_run   # same cluster test as the panel

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
    P = np.vstack([np.arange(9)] + [rng.permutation(9) for _ in range(N_PERM)])

    rows, scal = [], []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < MIN_RIPPLES:
            continue
        subj = rip[rip.session == s].subject_key.iloc[0]
        ph_i, _, _, _ = _labels(steps[steps.session == s], occ)
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
        ph = ph_i[k]
        grids = np.unique(occ.cv_group.to_numpy())
        masks = {p_: (ph == p_) for p_ in PHASES}
        if min(m.sum() for m in masks.values()) < MIN_RIPPLES:
            continue

        for ri, (roiname, members) in enumerate(ROI_SETS.items()):
            cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                     if c < len(spk[s]["spikes"])]
            if len(cells) < MIN_CELLS:
                continue
            _, loo = swc.build_templates(spk[s], cells, occ, grids)
            C = swc.zscore_cells([swc.window_counts(
                spk[s], cells, t_rip + off - half, t_rip + off + half)
                for off in OFFSETS])
            # Evidence per offset is small ((n_rip, 9)); the per-ripple
            # statistic crossed with 1000 permutations is not, so the offsets
            # are STREAMED below rather than stacked -- the largest session has
            # 7633 ripples, which would otherwise be ~2.5 GB.
            Es = [swc.score_windows(C[oi], t_rip, grid, loo)
                  for oi in range(len(OFFSETS))]
            ok_all = np.ones(len(t_rip), bool)
            for E in Es:
                ok_all &= np.isfinite(E).all(axis=1)
            idx = {p_: np.flatnonzero(masks[p_] & ok_all) for p_ in PHASES}
            nmin = min(len(v) for v in idx.values())
            if nmin < MIN_RIPPLES:
                continue
            # subsamples are drawn ONCE and reused at every offset, so a change
            # across the time course is a change in the signal, not in which
            # ripples contributed
            draws = {p_: [rng.choice(idx[p_], nmin, replace=False)
                          for _ in range(N_MATCH)] for p_ in PHASES}

            zm = {p_: np.full((N_MATCH, len(OFFSETS)), np.nan) for p_ in PHASES}
            zu = {p_: np.full(len(OFFSETS), np.nan) for p_ in PHASES}
            li = loc - 1
            # fixed-window statistics, accumulated as the offsets stream past.
            # "ripple" is the ripple window itself (offset 0); "pm250ms" is the
            # per-ripple statistic AVERAGED over every offset within +-250 ms,
            # so the null passes through the same averaging as the observed
            # value and the wider window is a genuine estimator, not a mean of
            # z values.
            Sw = {w: None for w in SCALAR_WINDOWS}
            nw = {w: 0 for w in SCALAR_WINDOWS}
            for oi, E in enumerate(Es):
                E = np.nan_to_num(E)
                # S is (n_ripples, 1 + N_PERM): column 0 is the observed
                # statistic per ripple, the rest its permutation null. A
                # subsample is a row selection, never a recomputation, so
                # empirical and null go through the same code.
                H = np.take_along_axis(E, P[:, li].T, axis=1)
                S = H - (E.sum(axis=1)[:, None] - H) / 8.0

                def z_of(sel):
                    m = S[sel].mean(axis=0)      # (1 + N_PERM,)
                    sd = m[1:].std()
                    return (m[0] - m[1:].mean()) / sd if sd > 0 else np.nan

                for w, halfw in SCALAR_WINDOWS.items():
                    inw = (abs(OFFSETS[oi]) >= FAR_FIELD_S if halfw is None
                           else abs(OFFSETS[oi]) <= halfw + 1e-9)
                    if inw:
                        Sw[w] = S.copy() if Sw[w] is None else Sw[w] + S
                        nw[w] += 1
                for p_ in PHASES:
                    zu[p_][oi] = z_of(idx[p_])
                    for di, sel in enumerate(draws[p_]):
                        zm[p_][di, oi] = z_of(sel)
            for p_ in PHASES:
                zz = np.nanmean(zm[p_], axis=0)
                for oi, off in enumerate(OFFSETS):
                    rows.append(dict(session=s, subject=subj, roi=roiname,
                                     phase=p_, offset_s=off,
                                     n_ripples=len(idx[p_]), n_matched=nmin,
                                     z=zz[oi], z_unmatched=zu[p_][oi]))

            # ---- the REPORTED contrast: one fixed window, many draws.
            # Draws are applied as a (n_draws x n_ripples) selection matrix so
            # all 1000 subsample means are one matmul rather than a loop.
            rs = np.random.default_rng([SEED, s, ri, 7])
            # ONE set of draws, reused for every window, so that a
            # ripple-minus-surround difference is paired draw by draw
            A = {}
            for p_ in PHASES:
                A[p_] = np.zeros((N_MATCH_SCALAR, len(t_rip)))
                for di in range(N_MATCH_SCALAR):
                    A[p_][di, rs.choice(idx[p_], nmin,
                                        replace=False)] = 1.0 / nmin
            z = {}
            for w in SCALAR_WINDOWS:
                Sm = Sw[w] / nw[w]
                for p_ in PHASES:
                    M = A[p_] @ Sm                # (n_draws, 1 + N_PERM)
                    sd = M[:, 1:].std(axis=1)
                    z[(w, p_)] = np.where(
                        sd > 0, (M[:, 0] - M[:, 1:].mean(axis=1)) / sd, np.nan)
            for w in SCALAR_WINDOWS:
                dz = z[(w, "encode")] - z[(w, "execute")]
                # running means answer "were 20 draws enough?" for free
                run = np.cumsum(dz) / np.arange(1, len(dz) + 1)
                rec = dict(
                    session=s, subject=subj, roi=roiname, window=w,
                    n_matched=nmin, n_offsets_in_window=nw[w],
                    z_encode=float(np.nanmean(z[(w, "encode")])),
                    z_execute=float(np.nanmean(z[(w, "execute")])),
                    d_z=float(np.nanmean(dz)),
                    d_z_sd_across_draws=float(np.nanstd(dz)),
                    d_z_at_20=float(run[19]), d_z_at_100=float(run[99]),
                    d_z_at_1000=float(run[-1]))
                if w != "far":
                    # ripple-specificity: this window minus the same draws'
                    # far field, so the subtraction is paired
                    for p_ in PHASES:
                        rec[f"z_{p_}_minus_surround"] = float(np.nanmean(
                            z[(w, p_)] - z[("far", p_)]))
                scal.append(rec)
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(rrsa._derivatives(), "group", "swr",
                     f"timecourse_by_phase_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)

    # display-only smoothing, done here so the figure plots exactly what it gets
    R = R.sort_values(["roi", "phase", "session", "offset_s"])
    R["z_smooth"] = (R.groupby(["roi", "phase", "session"])["z"]
                     .transform(lambda v: _smooth(v.to_numpy()[None, :])[0]))
    R.to_csv(os.path.join(d, "timecourse_by_phase.csv"), index=False)
    pd.DataFrame(scal).to_csv(os.path.join(d, "fixed_window_contrast.csv"),
                              index=False)
    summarise(d)
    summarise_scalar(d)

    json.dump(dict(seed=SEED, n_perm=N_PERM, n_match=N_MATCH,
                   n_cluster_perm=N_CLUST,
                   n_match_scalar=N_MATCH_SCALAR,
                   scalar_windows=SCALAR_WINDOWS,
                   offsets_s=OFFSETS.tolist(), roi_sets=ROI_SETS,
                   phases=list(PHASES), smooth_offsets=SMOOTH,
                   window="each ripple's own duration at every offset",
                   matching="both phases subsampled to the smaller phase's "
                            "ripple count within session, N_MATCH draws, "
                            "z averaged across draws",
                   unit="session",
                   null="template-label permutation within session; cluster "
                        "correction by sign-flipping whole session time "
                        "courses",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"-> {d}")


if __name__ == "__main__":
    main()
