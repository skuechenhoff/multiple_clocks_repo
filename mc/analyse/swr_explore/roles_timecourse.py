#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descriptive time course of location evidence BY SQUARE ROLE, around ripples.

    python scripts/swr_roles_timecourse.py

`swr_ripple_content_roles.py` answers "is role X carried more in a ripple than
beside it" with a regression at two time points (the ripple and its matched
flank). SK asked what that looks like resolved in time. This is the descriptive
companion: no regression, no null, no model -- at each offset from the ripple
peak, the within-window z of the evidence for every square, averaged by the role
that square plays. Exactly the numbers, plotted.

Roles, computed per occupancy interval by `run_history`:

  current                 the square underfoot
  on_route / off_route    non-reward squares the subject does (does not) walk on
                          the correct repeats of that grid run -- known phase only
  reward                  a reward square, not the current one
  error / no_error        squares with (without) an erroneous uncover so far in
                          this grid run

Windows are `WIN_S` wide and slide across the peri-ripple second, so a point at
0 ms is the ripple and points beyond +-300 ms are ordinary navigation. The
project's ripple windows are the ripple's own duration (median 60 ms); a fixed
width is used HERE only so that every offset is comparable to every other.

⚠ Descriptive. There is no inference in this file. The tests are in
`swr_ripple_content_roles.py`, and the He et al. peri (|t| <= 250 ms) versus
non-peri (250 < |t| <= 750 ms) bands are marked on the figures for reference.

@author: Svenja Kuchenhoff
"""

import os
import json
import datetime

import numpy as np
import pandas as pd

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc
from mc.analyse.swr_explore.ripple_content_roles import run_history, ADJ

WIN_S = 0.100
OFFSETS = np.round(np.arange(-0.75, 0.7501, 0.05), 4)
ROI_SETS = {"HC_all": ["HC_anterior", "HC_mid"],
            "HC_mid": ["HC_mid"], "HC_anterior": ["HC_anterior"]}
MIN_RIPPLES = 30
LOC = np.arange(1, 10)


def main():
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)

    rows = []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, _ = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < MIN_RIPPLES:
            continue
        (rew, known, err, vis, last, first, route, goal, nxt,
         g1, g2, corr, occ_sq, trial_n) = run_history(steps[steps.session == s],
                                                      occ)
        o = occ.sort_values("start_s")
        row_of = o.index.to_numpy()
        a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
        j = np.searchsorted(a, t_rip, side="right") - 1
        jj = np.clip(j, 0, len(o) - 1)
        k = row_of[jj]
        inside = (j >= 0) & (t_rip <= b[jj])
        if inside.sum() < MIN_RIPPLES:
            continue
        t_rip, k = t_rip[inside], k[inside]
        loc = occ["loc"].to_numpy()[k].astype(int)
        grid = occ.cv_group.to_numpy()[k]
        grids = np.unique(occ.cv_group.to_numpy())

        # (n_ripples, 9) role masks
        L = LOC[None, :]
        is_cur = L == loc[:, None]
        is_rew = np.zeros_like(is_cur)
        for c in range(4):
            is_rew |= L == rew[k][:, c][:, None]
        is_rew &= ~is_cur
        on_r = route[k] & ~is_rew & ~is_cur
        off_r = ~route[k] & ~is_rew & ~is_cur
        has_e = (err[k] > 0) & ~is_cur
        no_e = (err[k] == 0) & ~is_cur
        adj = ADJ[loc - 1] & ~is_cur
        phase = np.where(first[k], "explore", "known")

        for roiname, members in ROI_SETS.items():
            cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                     if c < len(spk[s]["spikes"])]
            if len(cells) < 2:
                continue
            _, loo = swc.build_templates(spk[s], cells, occ, grids)
            C = np.stack([swc.window_counts(spk[s], cells,
                                            t_rip + off - WIN_S / 2,
                                            t_rip + off + WIN_S / 2)
                          for off in OFFSETS])
            # one pooled z per cell across all offsets, so differences BETWEEN
            # offsets survive the normalisation
            mu = C.mean(axis=(0, 2), keepdims=True)
            sd = C.std(axis=(0, 2), keepdims=True)
            C = (C - mu) / np.where(sd > 0, sd, np.nan)

            for oi, off in enumerate(OFFSETS):
                E = swc.score_windows(C[oi], t_rip, grid, loo)
                ok = np.isfinite(E).all(axis=1)
                if ok.sum() < MIN_RIPPLES:
                    continue
                Z = ((E - E.mean(axis=1, keepdims=True))
                     / np.where(E.std(axis=1, keepdims=True) > 0,
                                E.std(axis=1, keepdims=True), np.nan))
                for ph in ("explore", "known"):
                    m = ok & (phase == ph)
                    if m.sum() < MIN_RIPPLES:
                        continue
                    d = dict(session=s, roi=roiname, phase=ph, offset_s=off,
                             n=int(m.sum()))
                    for name, mask in (("current", is_cur), ("reward", is_rew),
                                       ("on_route", on_r), ("off_route", off_r),
                                       ("error", has_e), ("no_error", no_e),
                                       ("adjacent", adj)):
                        v = np.where(mask[m], Z[m], np.nan)
                        d[name] = float(np.nanmean(v)) if np.isfinite(v).any() \
                            else np.nan
                    rows.append(d)
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"ripple_roles_timecourse_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "roles_timecourse.csv"), index=False)
    json.dump(dict(win_s=WIN_S, offsets_s=OFFSETS.tolist(),
                   roi_sets=ROI_SETS, min_ripples=MIN_RIPPLES,
                   note="DESCRIPTIVE ONLY -- no null, no inference. Fixed "
                        "window width so offsets are comparable; the tests use "
                        "each ripple's own duration.",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"\n-> {d}")
    for roiname in ROI_SETS:
        for ph in ("explore", "known"):
            g = R[(R.roi == roiname) & (R.phase == ph)]
            if not len(g):
                continue
            peri = g[np.abs(g.offset_s) <= .25]
            non = g[np.abs(g.offset_s) > .25]
            print(f"  {roiname:12s} {ph:8s} n={g.session.nunique():3d} | "
                  f"on_route-off_route peri {(peri.on_route - peri.off_route).mean():+.4f} "
                  f"non-peri {(non.on_route - non.off_route).mean():+.4f} | "
                  f"error-no_error peri {(peri.error - peri.no_error).mean():+.4f} "
                  f"non-peri {(non.error - non.no_error).mean():+.4f}")


if __name__ == "__main__":
    main()
