#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3's estimator, resolved in time around the ripple peak.

    python scripts/swr_stage3_timecourse.py

Figure 15 showed the peri-ripple shape but had to pool cells across the sessions
that ran the same configuration, which drops the unit of inference to the
CONFIGURATION (n = 8-14) and makes those units share sessions. Figure 5 has the
honest unit -- the SESSION (n = 29-51) -- but only two time points, the ripple
and its matched flank.

This is Figure 5's estimator at every offset: same score, same per-session
permutation null, same unit of inference, now as a time course. Every session
contributes; no configuration matching is needed, because nothing is pooled
across sessions.

WINDOW. Each ripple's OWN duration (median 60 ms) at every offset, so the window
is identical to the one Stage 3 uses and constant across the time course. Spike
counts are z-scored per cell across ALL offsets pooled, so a difference between
offsets is a real difference and not a by-product of normalising each offset
separately.

AT OFFSET 0 THIS IS STAGE 3's C0, and the script checks that: the printed value
at 0 ms should reproduce the Stage 3 table.

MULTIPLE COMPARISONS. 41 offsets is a family. The group test is therefore
cluster-based: contiguous runs of offsets with |t| above the pointwise 0.05
threshold are summed, and the largest cluster is compared against the null built
by flipping the sign of each session's whole time course (1,000 draws). That
null is exact under the hypothesis of no effect at any offset.

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
import scripts.swr_place_templates as spt

SEED = 42
N_PERM = 100
N_CLUST = 1000
OFFSETS = np.round(np.arange(-0.50, 0.5001, 0.025), 4)
ROI_SETS = {"HC_all": ["HC_anterior", "HC_mid"],
            "HC_mid": ["HC_mid"], "HC_anterior": ["HC_anterior"],
            "mOFC": ["mOFC"], "mPFC": ["mPFC"]}
MIN_RIPPLES = 30


def cluster_test(Z, rng):
    """Largest cluster of |t| above the pointwise threshold, vs a sign-flip null.

    Z is (n_sessions, n_offsets) of per-session z values. Flipping the sign of a
    whole session's time course is the exact randomisation under "no effect at
    any offset", and it preserves each session's temporal autocorrelation.
    """
    n = Z.shape[0]
    thr = stats.t.ppf(0.975, n - 1)

    def mass(M):
        t = M.mean(0) / (M.std(0, ddof=1) / np.sqrt(n))
        best, cur = 0.0, 0.0
        for v in t:
            if abs(v) >= thr:
                cur += abs(v)
                best = max(best, cur)
            else:
                cur = 0.0
        return best, t

    obs, t_obs = mass(Z)
    nul = np.empty(N_CLUST)
    for i in range(N_CLUST):
        sgn = rng.choice([-1.0, 1.0], size=(n, 1))
        nul[i] = mass(Z * sgn)[0]
    return t_obs, obs, float((nul >= obs).mean()), thr


def main():
    rng = np.random.default_rng(SEED)
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s", "subject_key"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)

    rows = []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = spt.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < MIN_RIPPLES:
            continue
        subj = rip[rip.session == s].subject_key.iloc[0]
        o = occ.sort_values("start_s")
        row_of = o.index.to_numpy()
        a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
        j = np.searchsorted(a, t_rip, side="right") - 1
        jj = np.clip(j, 0, len(a) - 1)
        k = row_of[jj]
        inside = (j >= 0) & (t_rip <= b[jj])
        if inside.sum() < MIN_RIPPLES:
            continue
        t_rip, d_rip, k = t_rip[inside], d_rip[inside], k[inside]
        loc = occ["loc"].to_numpy()[k].astype(int)
        grid = occ.cv_group.to_numpy()[k]
        half = d_rip / 2.0
        grids = np.unique(occ.cv_group.to_numpy())

        for ri, (roiname, members) in enumerate(ROI_SETS.items()):
            cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                     if c < len(spk[s]["spikes"])]
            if len(cells) < 2:
                continue
            _, loo = spt.build_templates(spk[s], cells, occ, grids)
            # one pooled z per cell over every offset, so offsets stay comparable
            raw = [spt.window_counts(spk[s], cells, t_rip + off - half,
                                     t_rip + off + half) for off in OFFSETS]
            C = swc.zscore_cells(raw)
            rng_p = np.random.default_rng([SEED, s, ri])
            perms = [rng_p.permutation(9) for _ in range(N_PERM)]
            li = loc - 1
            for oi, off in enumerate(OFFSETS):
                E = spt.score_windows(C[oi], t_rip, grid, loo)
                ok = np.isfinite(E).all(axis=1)
                if ok.sum() < MIN_RIPPLES:
                    continue
                Eo, lo_ = E[ok], li[ok]
                tot = Eo.sum(axis=1)
                ar = np.arange(len(Eo))

                def score(pm):
                    h = Eo[ar, pm[lo_]]
                    return float(np.mean(h - (tot - h) / 8.0))

                v = score(np.arange(9))
                nul = np.array([score(pm) for pm in perms])
                sd = nul.std()
                rows.append(dict(session=s, subject=subj, roi=roiname,
                                 offset_s=off, n_ripples=int(ok.sum()),
                                 score=v,
                                 z=(v - nul.mean()) / sd if sd > 0 else np.nan))
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"stage3_timecourse_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "stage3_timecourse.csv"), index=False)

    print("\n=== Stage 3's estimator, time-locked to the ripple peak ===")
    print("    unit of inference = SESSION; cluster-corrected over 41 offsets\n")
    summ = []
    for roiname in ROI_SETS:
        g = R[R.roi == roiname]
        if g.session.nunique() < 5:
            continue
        W = g.pivot_table(index="session", columns="offset_s", values="z")
        W = W.dropna(axis=0, how="any")
        if len(W) < 5:
            continue
        t_obs, mass_obs, p_clust, thr = cluster_test(W.to_numpy(), rng)
        i0 = int(np.argmin(np.abs(W.columns.to_numpy())))
        z0 = W.to_numpy()[:, i0]
        p0 = stats.ttest_1samp(z0, 0)[1]
        pk = int(np.argmax(np.abs(t_obs)))
        print(f"  {roiname:12s} n = {len(W)} sessions")
        print(f"      at 0 ms: z = {z0.mean():+.3f}, p = {p0:.4g}   "
              f"(Stage 3 C0 should match)")
        print(f"      peak |t| = {abs(t_obs[pk]):.2f} at "
              f"{W.columns[pk] * 1000:+.0f} ms;  largest cluster mass "
              f"{mass_obs:.1f}, cluster p = {p_clust:.3f}")
        summ.append(dict(roi=roiname, n_sessions=len(W),
                         z_at_zero=float(z0.mean()), p_at_zero=float(p0),
                         peak_offset_s=float(W.columns[pk]),
                         peak_t=float(t_obs[pk]), cluster_mass=float(mass_obs),
                         p_cluster=float(p_clust)))
    pd.DataFrame(summ).to_csv(os.path.join(d, "stage3_timecourse_summary.csv"),
                              index=False)
    json.dump(dict(seed=SEED, n_perm=N_PERM, n_cluster_perm=N_CLUST,
                   offsets_s=OFFSETS.tolist(), roi_sets=ROI_SETS,
                   window="each ripple's own duration at every offset",
                   unit="session", null="template-label permutation per session; "
                        "cluster correction by sign-flipping whole sessions",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"\n-> {d}")


if __name__ == "__main__":
    main()
