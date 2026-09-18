#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Location information in and around hippocampal ripples -- the reported result.

    python scripts/swr_content_main.py            # everything, in order
    python scripts/swr_content_main.py --step=timecourse

THE CLAIM THIS SCRIPT SUPPORTS, and nothing more: hippocampal population
activity carries the square the participant is standing on, this is measurable
inside a ripple window, and it is NOT specific to ripples -- a window matched on
square, dwell and width, taken from the same visit, carries just as much.

Everything exploratory -- reward locations, the walked route, error squares,
pseudo-populations, decoders -- is in `swr_content_explore.py` and is not part
of the reported result.

FOUR STEPS, in dependency order:

  timeline    real-time location from `trial_vars`, cached once. Everything
              downstream needs it and nothing else provides it: the time-warped
              snippets in `all_location_snippets.csv` put only 7.2% of ripples
              inside a snippet, and integrating button presses reconstructs the
              true square 62.9% of the time.
  control     the positive control. The same estimator on ordinary navigation
              windows >= 1 s from any ripple, swept over window width. If this
              is null the ripple numbers mean nothing.
  contrast    ripple versus matched flank (Stage 3). The confirmatory test of
              ripple-specificity.
  timecourse  the same estimator at 41 offsets from -500 to +500 ms, cluster
              corrected. Produces the reported panel.

ESTIMATOR, once, since all four steps share it. For every unit, a place
template: firing rate at each of the nine squares, z-scored across squares so a
location-independent unit contributes exactly zero. Templates are estimated
leave-one-CONFIGURATION-out -- the configuration, not the block, because each
configuration recurs in ~3 blocks and holding out one block leaves the same
rewards and route in training. Spike counts are z-scored per unit so loud units
do not dominate. Evidence E(L) = sum_c n_c * z_c(L); the statistic is the
evidence for the occupied square minus the mean of the other eight. Inference is
per-session z against a label-permutation null, then t across sessions.

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

try:
    import fire
except ImportError:
    fire = None

SEED = 42
N_PERM = 100
N_CLUST = 1000
ROI_SETS = {"HC_all": ["HC_anterior", "HC_mid"],
            "HC_mid": ["HC_mid"], "HC_anterior": ["HC_anterior"],
            "mOFC": ["mOFC"], "mPFC": ["mPFC"]}
WIDTHS = [0.060, 0.125, 0.250, 0.500]      # positive-control sweep
OFFSETS = np.round(np.arange(-0.50, 0.5001, 0.025), 4)
FLANK_GAP_S = 0.05
MAX_FLANKS = 2
MIN_RIPPLES = 30
MIN_CELLS = 2


def out_dir(tag):
    d = os.path.join(rrsa._derivatives(), "group", "swr",
                     f"content_main_{tag}_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    return d


def _inputs():
    rip = pd.read_csv(os.path.join(rrsa._derivatives(), "group", "swr",
                                   "bundle_v2", "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s",
                               "subject_key"])
    sessions = sorted(rip.session.unique().tolist())
    return (rip, sessions, rrsa.load_spike_times(sessions=sessions,
                                                 verbose=False),
            rrsa.cell_roi_table(sessions=sessions), swl.load(sessions))


def _ripples_in_task(occ, t_rip, d_rip):
    """Ripples inside a resolvable occupancy interval, with their square."""
    o = occ.sort_values("start_s")
    row_of = o.index.to_numpy()
    a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
    j = np.searchsorted(a, t_rip, side="right") - 1
    jj = np.clip(j, 0, len(a) - 1)
    k = row_of[jj]
    inside = (j >= 0) & (t_rip <= b[jj])
    return (t_rip[inside], d_rip[inside], k[inside],
            occ["loc"].to_numpy()[k[inside]].astype(int),
            occ.cv_group.to_numpy()[k[inside]])


def _z(v, null):
    sd = np.nanstd(null)
    return float((v - np.nanmean(null)) / sd) if sd > 0 else np.nan


# ── step 1 ─────────────────────────────────────────────────────────────

def timeline(rebuild=False):
    """Cache the real-time location timeline, and validate it."""
    d = swl.cache_dir()
    if rebuild or not os.path.exists(d) or not os.listdir(d):
        print("building the location timeline (slow, once) ...")
        swl.build()
    steps = swl.load()
    print(f"location timeline: {steps.session.nunique()} sessions, "
          f"{len(steps)} steps, {int((steps.is_uncover == 1).sum())} uncovers")
    return steps


# ── step 2 ─────────────────────────────────────────────────────────────

def control():
    """Positive control: the estimator on ordinary navigation windows.

    Windows are TILED -- every non-overlapping window of the given width inside
    task time, at least 1 s from any ripple. Sampling them at random instead
    made the answer move by 0.2 z between seeds (CHANGELOG 2026-09-17 c/d).
    """
    rip, sessions, spk, roi, steps = _inputs()
    rows = []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, _ = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r):
            continue
        grids = np.unique(occ.cv_group.to_numpy())
        o = occ.sort_values("start_s")
        for width in WIDTHS:
            t0, loc = swc.tiled_windows(occ, width, avoid=t_rip,
                                        avoid_pad_s=1.0)
            if len(t0) < 200:
                continue
            j = np.searchsorted(o.start_s.to_numpy(), t0, side="right") - 1
            win_grid = o.cv_group.to_numpy()[np.clip(j, 0, len(o) - 1)]
            for ri, (roiname, members) in enumerate(ROI_SETS.items()):
                rng_p = np.random.default_rng([SEED, s, ri, int(width * 1000)])
                cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                         if c < len(spk[s]["spikes"])]
                if len(cells) < MIN_CELLS:
                    continue
                _, loo = swc.build_templates(spk[s], cells, occ, grids)
                C_raw = swc.window_counts(spk[s], cells, t0, t0 + width)
                C = swc.zscore_cells([C_raw])[0]
                E = swc.score_windows(C, t0, win_grid, loo)
                ok = np.isfinite(E).all(axis=1)
                if ok.sum() < 200:
                    continue
                v, nul = _score_and_null(E[ok], loc[ok].astype(int) - 1,
                                         rng_p)
                rows.append(dict(session=s, roi=roiname, width=width,
                                 n_windows=int(ok.sum()),
                                 spikes_per_cell=float(C_raw.mean()),
                                 score=v, z=_z(v, nul)))
        print(f"  s{s:02d}", flush=True)
    R = pd.DataFrame(rows)
    d = out_dir("control")
    R.to_csv(os.path.join(d, "positive_control.csv"), index=False)
    print("\n=== POSITIVE CONTROL: location in ordinary navigation windows ===")
    print("    (>= 1 s from any ripple; if this is null the ripple numbers "
          "mean nothing)\n")
    print(f"{'ROI':12s} {'width':>6s} {'sess':>5s} {'spk/cell':>9s} "
          f"{'z':>8s} {'p':>9s}")
    for roiname in ROI_SETS:
        for w in WIDTHS:
            g = R[(R.roi == roiname) & (R.width == w)]
            v = g.z.dropna()
            if len(v) < 5:
                continue
            print(f"{roiname:12s} {w:6.3f} {len(v):5d} "
                  f"{g.spikes_per_cell.mean():9.3f} {v.mean():+8.3f} "
                  f"{stats.ttest_1samp(v, 0)[1]:9.3g}")
    print(f"\n-> {d}")


# ── step 3 ─────────────────────────────────────────────────────────────

def contrast():
    """Ripple versus matched flank -- the test of ripple-specificity.

    The flank sits INSIDE the ripple's own occupancy interval, so it shares the
    square, the dwell and the task phase, and has the same width. A fixed
    +-500 ms flank was rejected: it falls on the same square only 53.7% of the
    time, and when it does not it is scored for the wrong square, which
    manufactures a positive ripple-minus-flank difference.
    """
    rip, sessions, spk, roi, steps = _inputs()
    rows = []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < MIN_RIPPLES:
            continue
        subj = rip[rip.session == s].subject_key.iloc[0]
        t_rip, d_rip, k, loc, grid = _ripples_in_task(occ, t_rip, d_rip)
        if len(t_rip) < MIN_RIPPLES:
            continue
        half = d_rip / 2.0
        grids = np.unique(occ.cv_group.to_numpy())
        t_fl, owner = swc.matched_flanks(t_rip, half, occ, other_t=t_rip,
                                         gap_s=FLANK_GAP_S, max_per=MAX_FLANKS)
        if not len(t_fl):
            continue
        w_fl = half[owner]
        for ri, (roiname, members) in enumerate(ROI_SETS.items()):
            # an independent, deterministic stream per session and ROI. With a
            # single shared stream, adding or reordering an ROI moves every
            # other ROI's number -- that bit this analysis once already.
            rng_p = np.random.default_rng([SEED, s, ri])
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
            if ok_r.sum() < MIN_RIPPLES or ok_f.sum() < MIN_RIPPLES:
                continue
            li_r, li_f = loc - 1, loc[owner] - 1
            tot_r, tot_f = E_rip.sum(axis=1), E_fl.sum(axis=1)
            ar, af = np.arange(len(E_rip)), np.arange(len(E_fl))

            def stat(pm):
                hr = E_rip[ar, pm[li_r]]
                sr = hr - (tot_r - hr) / 8.0
                hf = E_fl[af, pm[li_f]]
                sf = hf - (tot_f - hf) / 8.0
                acc = np.zeros(len(E_rip)); cnt = np.zeros(len(E_rip))
                np.add.at(acc, owner[ok_f], sf[ok_f])
                np.add.at(cnt, owner[ok_f], 1.0)
                good = ok_r & (cnt > 0)
                fl = np.full(len(E_rip), np.nan)
                fl[good] = acc[good] / cnt[good]
                return (float(np.nanmean(sr[good])), float(np.nanmean(fl[good])),
                        float(np.nanmean(sr[good] - fl[good])), int(good.sum()))

            obs = stat(np.arange(9))
            if obs[3] < MIN_RIPPLES:
                continue
            nul = np.array([stat(rng_p.permutation(9))[:3]
                            for _ in range(N_PERM)], float)
            out = dict(session=s, subject=subj, roi=roiname,
                       n_cells=len(cells), n_ripples=obs[3])
            for i, nm in enumerate(("ripple", "flank", "diff")):
                out[nm] = obs[i]
                out[f"z_{nm}"] = _z(obs[i], nul[:, i])
            rows.append(out)
        print(f"  s{s:02d}", flush=True)
    R = pd.DataFrame(rows)
    d = out_dir("contrast")
    R.to_csv(os.path.join(d, "ripple_vs_flank.csv"), index=False)
    print("\n=== RIPPLE vs MATCHED FLANK ===")
    print("    C0 = content inside the ripple;  C1 = ripple minus its own "
          "matched flank\n")
    print(f"{'ROI':12s} {'sess':>5s} {'ripples':>8s} {'C0':>8s} {'p':>9s} "
          f"{'C1':>8s} {'p':>9s}")
    for roiname in ROI_SETS:
        g = R[R.roi == roiname].dropna(subset=["z_ripple"])
        if len(g) < 5:
            continue
        gd = g.dropna(subset=["z_diff"])
        print(f"{roiname:12s} {len(g):5d} {int(g.n_ripples.sum()):8d} "
              f"{g.z_ripple.mean():+8.3f} "
              f"{stats.ttest_1samp(g.z_ripple, 0)[1]:9.3g} "
              f"{gd.z_diff.mean():+8.3f} "
              f"{stats.ttest_1samp(gd.z_diff, 0)[1]:9.3g}")
    print(f"\n-> {d}")


# ── step 4 ─────────────────────────────────────────────────────────────

def _cluster_test(Z, rng, n_clust=N_CLUST):
    """Largest cluster of |t| above the pointwise threshold, vs a sign-flip null.

    Flipping the sign of a whole session's time course is exact under "no effect
    at any offset" and preserves each session's temporal autocorrelation.
    """
    n = Z.shape[0]
    thr = stats.t.ppf(0.975, n - 1)

    def mass(M):
        t = M.mean(0) / (M.std(0, ddof=1) / np.sqrt(n))
        best = cur = 0.0
        for v in t:
            cur = cur + abs(v) if abs(v) >= thr else 0.0
            best = max(best, cur)
        return best, t

    obs, t_obs = mass(Z)
    nul = np.array([mass(Z * rng.choice([-1.0, 1.0], size=(n, 1)))[0]
                    for _ in range(n_clust)])
    return t_obs, obs, float((nul >= obs).mean()), thr


def _extents(t_obs, off, thr, mass_obs):
    runs, i = [], 0
    above = np.abs(t_obs) >= thr
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            runs.append((off[i], off[j], float(np.abs(t_obs[i:j + 1]).sum())))
            i = j + 1
        else:
            i += 1
    return [dict(start_s=a, stop_s=b, mass=m,
                 is_largest=abs(m - mass_obs) < 1e-9) for a, b, m in runs]


def timecourse():
    """The estimator at 41 offsets around the ripple peak, cluster corrected.

    Window is each ripple's OWN duration at every offset, so it is identical to
    the one `contrast` uses and constant across the time course. Counts are
    z-scored per unit across ALL offsets pooled, so a difference between offsets
    is a real difference and not a by-product of the normalisation.

    At offset 0 this IS `contrast`'s C0 up to that normalisation, and the
    printout says so, so a silent divergence would be visible.
    """
    rip, sessions, spk, roi, steps = _inputs()
    rows = []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < MIN_RIPPLES:
            continue
        subj = rip[rip.session == s].subject_key.iloc[0]
        t_rip, d_rip, k, loc, grid = _ripples_in_task(occ, t_rip, d_rip)
        if len(t_rip) < MIN_RIPPLES:
            continue
        half = d_rip / 2.0
        grids = np.unique(occ.cv_group.to_numpy())
        for ri, (roiname, members) in enumerate(ROI_SETS.items()):
            cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                     if c < len(spk[s]["spikes"])]
            if len(cells) < MIN_CELLS:
                continue
            _, loo = swc.build_templates(spk[s], cells, occ, grids)
            C = swc.zscore_cells([swc.window_counts(
                spk[s], cells, t_rip + off - half, t_rip + off + half)
                for off in OFFSETS])
            rng_p = np.random.default_rng([SEED, s, ri])
            perms = [rng_p.permutation(9) for _ in range(N_PERM)]
            for oi, off in enumerate(OFFSETS):
                E = swc.score_windows(C[oi], t_rip, grid, loo)
                ok = np.isfinite(E).all(axis=1)
                if ok.sum() < MIN_RIPPLES:
                    continue
                Eo, li = E[ok], (loc - 1)[ok]
                tot, ar = Eo.sum(axis=1), np.arange(len(Eo))

                def one(pm):
                    h = Eo[ar, pm[li]]
                    return float(np.mean(h - (tot - h) / 8.0))

                v = one(np.arange(9))
                nul = np.array([one(pm) for pm in perms])
                rows.append(dict(session=s, subject=subj, roi=roiname,
                                 offset_s=off, n_ripples=int(ok.sum()),
                                 score=v, z=_z(v, nul)))
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = out_dir("timecourse")
    R.to_csv(os.path.join(d, "timecourse.csv"), index=False)

    print("\n=== LOCATION SIGNAL THROUGH THE PERI-RIPPLE SECOND ===")
    print("    unit of inference = SESSION; cluster-corrected over "
          f"{len(OFFSETS)} offsets\n")
    summ, ext = [], []
    for roiname in ROI_SETS:
        W = (R[R.roi == roiname]
             .pivot_table(index="session", columns="offset_s", values="z")
             .dropna())
        if len(W) < 5:
            continue
        off = W.columns.to_numpy()
        t_obs, mass, p_cl, thr = _cluster_test(W.to_numpy(), rng)
        i0 = int(np.argmin(np.abs(off)))
        z0 = W.to_numpy()[:, i0]
        t0, p0 = stats.ttest_1samp(z0, 0)
        # ripple-specificity in the temporal frame: each session's own far field
        far = np.abs(off) >= 0.35
        Mb = W.to_numpy() - W.to_numpy()[:, far].mean(axis=1, keepdims=True)
        _, mass_b, p_b, _ = _cluster_test(Mb, rng)
        print(f"  {roiname:12s} n = {len(W)} sessions")
        print(f"      at the ripple  z = {z0.mean():+.3f}, "
              f"t({len(W) - 1}) = {t0:+.2f}, p = {p0:.4g}")
        print(f"      vs zero        largest cluster mass {mass:.1f}, "
              f"p = {p_cl:.3f}")
        print(f"      vs surround    largest cluster mass {mass_b:.1f}, "
              f"p = {p_b:.3f}   <- ripple-specificity")
        summ.append(dict(roi=roiname, n_sessions=len(W),
                         z_at_zero=float(z0.mean()), t_at_zero=float(t0),
                         p_at_zero=float(p0), cluster_mass=float(mass),
                         p_cluster=float(p_cl),
                         cluster_mass_baselined=float(mass_b),
                         p_cluster_baselined=float(p_b)))
        for e in _extents(t_obs, off, thr, mass):
            ext.append(dict(roi=roiname, p_cluster=p_cl, **e))
    pd.DataFrame(summ).to_csv(os.path.join(d, "timecourse_summary.csv"),
                              index=False)
    pd.DataFrame(ext).to_csv(os.path.join(d, "cluster_extents.csv"),
                             index=False)
    # the durations the panel annotates
    dur = []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < MIN_RIPPLES:
            continue
        if not len(r[r.roi.isin(["HC_anterior", "HC_mid"])]):
            continue
        _, dd, _, _, _ = _ripples_in_task(occ, t_rip, d_rip)
        dur.append(dd * 1000)
    if dur:
        np.save(os.path.join(d, "ripple_durations_ms.npy"),
                np.concatenate(dur))
    json.dump(dict(seed=SEED, n_perm=N_PERM, n_cluster_perm=N_CLUST,
                   offsets_s=OFFSETS.tolist(), roi_sets=ROI_SETS,
                   window="each ripple's own duration at every offset",
                   unit="session",
                   null="template-label permutation per session; cluster "
                        "correction by sign-flipping whole sessions",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"\n-> {d}")


# ── step 5 ─────────────────────────────────────────────────────────────

def panel():
    """The reported figure. Regenerated LAST so it cannot drift from the numbers."""
    import scripts.swr_content_figures as fig
    d = fig.fig_dir()
    fig.figure15(d)
    fig.figure15(d, compact=True)
    print(f"-> {d}")


def run(step="all", rebuild_timeline=False):
    steps = {"timeline": lambda: timeline(rebuild_timeline),
             "control": control, "contrast": contrast,
             "timecourse": timecourse, "panel": panel}
    if step == "all":
        for name in ("timeline", "control", "contrast", "timecourse", "panel"):
            print(f"\n{'=' * 70}\n{name.upper()}\n{'=' * 70}")
            steps[name]()
    elif step in steps:
        steps[step]()
    else:
        print(f"unknown step {step!r}; choose from {list(steps)} or 'all'")


if __name__ == "__main__":
    if fire is not None and len(sys.argv) > 1:
        fire.Fire({"run": run, "timeline": timeline, "control": control,
                   "contrast": contrast, "timecourse": timecourse,
                   "panel": panel})
    else:
        run("all")
