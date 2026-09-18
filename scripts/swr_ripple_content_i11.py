#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I11 -- during exploration, does a ripple carry more about a REWARDED location?

    python scripts/swr_ripple_content_i11.py

SK's reasoning, from `POTENTIAL_IDEAS.md`: the hippocampus should only bother
communicating the rewarded locations. A non-rewarded square is meaningless until
the trajectory is known, so there is nothing worth sending to prefrontal cortex
about it.

Stage 3 found no ripple-specific location content when ALL ripples were pooled
against matched flanks. That does not answer this question and does not rule it
out: I11 is a DIFFERENTIAL prediction. If ripples at rewarded locations carry
more and ripples elsewhere carry less, the pooled average is exactly the zero
Stage 3 measured.

THE THREE-WAY SPLIT, AND WHY IT IS THREE AND NOT TWO

Restricted to the FIRST TRAVERSAL of each grid -- the only time a subject can be
standing somewhere without knowing whether it is rewarded -- every moment is one
of three kinds, by what the square underfoot is:

  known_rew   a reward, already uncovered.  THE TARGET.
  future_rew  a reward, not yet uncovered.  THE KNOWLEDGE CONTROL.
  nonrew      not a reward in this grid.    THE BASELINE.

`future_rew` is what makes the design work. It is the same physical square as
`known_rew` -- visited on every later traversal, so its place template is
estimated from the same amount of data, and it sits in the same part of the grid
-- but the subject does not yet know it is rewarded. So:

  known_rew > nonrew AND future_rew ~ nonrew   -> the difference is KNOWLEDGE.
  known_rew > nonrew AND future_rew > nonrew   -> the difference is the SQUARE
                                                  (template quality, geometry,
                                                  visit frequency), not reward.

A two-way rewarded/non-rewarded split cannot tell those apart, which is why the
version written in the ideas log is not the version run here.

TWO CONFOUNDS, TWO CONTROLS

  Stillness. Measured here: median dwell 0.860 s at a known reward against
  0.417 s at a non-rewarded square and 0.367 s at an undiscovered one -- the
  pause SK warned about, and the thing ripple rate is known to track. Every
  target ripple is therefore matched to one ripple of each other class in the
  SAME session on both the dwell of the square it sits in and the latency from
  arrival, nearest-neighbour, without replacement, seeded, with a caliper; the
  achieved balance and every drop are reported. (Ripple RATE turns out to be
  flat across the three classes -- 0.491 / 0.506 / 0.484 Hz -- so the stillness
  difference does not by itself produce more ripples. The matching is still
  applied, because content per ripple could depend on dwell even where rate
  does not.)

  Template quality. A reward square is visited more often across the session, so
  its template is better estimated, so its evidence score is higher for reasons
  that have nothing to do with ripples. The PRIMARY statistic is therefore
  ripple MINUS its own matched flanks (`swc.matched_flanks`: same square, same
  occupancy interval, same width), in which template quality cancels exactly.
  The plain in-ripple score is reported second, carrying that confound.

TWO NULLS, because the two questions need different ones and CLAUDE.md requires
the empirical and null values come out of the same function:

  template-label permutation   destroys location identity in every class at
                               once. Answers "is there content in this class at
                               all", comparable to Stage 2 and Stage 3.
  class-label shuffle          permutes the three labels WITHIN each matched
                               triplet. An exact randomisation test for the
                               contrast, holding content, location and stillness
                               fixed. PRIMARY for known_rew - nonrew.

Unit of inference: SESSION (the project's convention for ephys); subject-level
printed as a sensitivity check.

PRIMARY ROI: HC_mid, the only region that passed the Stage 2 current-location
control on raw counts. HC_anterior passes it too once counts are z-scored per
cell and is reported as primary-adjacent; mOFC and mPFC are for completeness.

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

SEED = int(os.environ.get("I11_SEED", 42))
N_PERM = int(os.environ.get("I11_NPERM", 100))
N_MATCH = int(os.environ.get("I11_NMATCH", 20))   # matchings averaged over
# HC_all pools the two hippocampal sets: I11 loses most of its sessions to the
# matching, and HC_mid alone survives in too few to test anything
ROI_SETS = {"HC_all": ["HC_anterior", "HC_mid"],
            "HC_mid": ["HC_mid"], "HC_anterior": ["HC_anterior"],
            "mOFC": ["mOFC"], "mPFC": ["mPFC"]}
ROIS = list(ROI_SETS)
CLASSES = ["known_rew", "future_rew", "nonrew"]
FLANK_GAP_S = 0.05
MAX_FLANKS = 2
CALIPER_S = 0.15          # same caliper as swr_matched_control_presses.py
REL_CALIPER = 0.10
# Cells weighted by how well their place map generalises across configurations
# (`spt.weighted_loo`), so Figure 6 can be read for location-encoding cells only.
SCHEMES = ["all", "thresh_0.2"]
MIN_MATCHES = 10          # inclusion, set low so sensitivity can run BOTH ways
PRIMARY_MIN = 20          # the threshold declared before the first run
SENSITIVITY = (10, 20, 30, 40)


# ── 1) what the subject is standing on, during the first traversal ──────

def interval_classes(steps, occ, session):
    """Per occupancy interval: class label, and whether it is a first traversal.

    A reward of rank r is DISCOVERED once the sought state has moved past it,
    i.e. r < `state`. On the first traversal that makes states 1..state-1 known
    and state..4 unknown -- the same rule C2 uses.
    """
    st = steps[steps.session == session].sort_values("t_s").reset_index(drop=True)
    k = occ.step_idx.to_numpy()
    first = st.first_trial.to_numpy(float)[k] == 1
    loc = occ["loc"].to_numpy(float)
    rew = st[["rew_A", "rew_B", "rew_C", "rew_D"]].to_numpy(float)[k]
    state = st.state.to_numpy(float)[k]
    known = np.arange(1, 5)[None, :] < state[:, None]
    is_rew = rew == loc[:, None]
    cls = np.where((is_rew & known).any(1), "known_rew",
                   np.where((is_rew & ~known).any(1), "future_rew", "nonrew"))
    return cls, first


# ── 2) matching ────────────────────────────────────────────────────────

def _nearest(i, cand, free, dwell, lat):
    """Index into `cand` of the closest free ripple, or None."""
    tol_d = max(CALIPER_S, REL_CALIPER * dwell[i])
    tol_l = max(CALIPER_S, REL_CALIPER * lat[i])
    dd = np.abs(dwell[cand] - dwell[i])
    dl = np.abs(lat[cand] - lat[i])
    ok = free & (dd <= tol_d) & (dl <= tol_l)
    if not ok.any():
        return None
    # closest on the two together, each in units of its own caliper
    return int(np.argmin(np.where(ok, dd / tol_d + dl / tol_l, np.inf)))


def match_sets(cls, dwell, lat, rng, targets, controls):
    """Match every `targets` ripple to one ripple of each control class.

    Nearest neighbour on dwell and latency, WITHOUT replacement, targets served
    in a seeded random order so the first-served do not take all the best
    partners. A target is kept only if it finds a partner in EVERY control class
    -- with two controls that makes a complete triplet, so the two contrasts
    rest on the same events rather than on two different subsets.

    Returns (n, 1 + len(controls)) row indices and the indices of the targets
    that found nobody, which are reported rather than counted away.
    """
    idx = {c: np.flatnonzero(cls == c) for c in set([targets] + list(controls))}
    free = {c: np.ones(len(idx[c]), bool) for c in controls}
    tgt = idx[targets]
    out, missed = [], []
    for i in tgt[rng.permutation(len(tgt))]:
        pick = {}
        for c in controls:
            j = _nearest(i, idx[c], free[c], dwell, lat)
            if j is None:
                break
            pick[c] = j
        if len(pick) < len(controls):
            missed.append(i)
            continue
        for c, j in pick.items():
            free[c][j] = False
        out.append([i] + [idx[c][pick[c]] for c in controls])
    return (np.array(out, int).reshape(-1, 1 + len(controls)),
            np.array(missed, int))


# ── 3) the statistic ───────────────────────────────────────────────────

def make_stat(E_rip, li_r, E_fl, li_f, owner):
    """Closure giving per-class (in-ripple, ripple-minus-flank) means.

    `perm` permutes the nine template labels. `trip` is the (n, 3) table of
    ripple rows, one column per class -- so the class-label shuffle is applied
    by reordering each row of `trip` and calling this same function, which is
    what CLAUDE.md means by deriving the null through the empirical code path.
    """
    n_r = len(E_rip)
    tot_r, tot_f = E_rip.sum(axis=1), E_fl.sum(axis=1)
    ar, af = np.arange(n_r), np.arange(len(E_fl))
    acc0 = np.zeros(n_r); cnt = np.zeros(n_r)
    np.add.at(cnt, owner, 1.0)
    has_fl = cnt > 0

    def stat(perm, trip):
        hr = E_rip[ar, perm[li_r]]
        sr = hr - (tot_r - hr) / 8.0
        hf = E_fl[af, perm[li_f]]
        sf = hf - (tot_f - hf) / 8.0
        acc = acc0.copy()
        np.add.at(acc, owner, sf)
        diff = np.full(n_r, np.nan)
        diff[has_fl] = sr[has_fl] - acc[has_fl] / cnt[has_fl]
        out = np.empty((trip.shape[1], 2))
        for c in range(trip.shape[1]):
            rows = trip[:, c]
            out[c, 0] = np.mean(sr[rows])
            out[c, 1] = np.nanmean(diff[rows]) if has_fl[rows].any() else np.nan
        return out

    return stat


def _z(v, null):
    sd = np.nanstd(null)
    return (v - np.nanmean(null)) / sd if sd > 0 else np.nan


def _shuffle_rows(tab, rng):
    """Relabel the classes independently within every matched set."""
    return np.take_along_axis(tab, np.argsort(rng.random(tab.shape), axis=1),
                              axis=1)


CONTRASTS = (("known_rew", "nonrew"), ("future_rew", "nonrew"),
             ("known_rew", "future_rew"))


def main():
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s", "subject_key"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)

    rows, bal = [], []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = spt.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or not len(t_rip):
            continue
        subj = rip[rip.session == s].subject_key.iloc[0]
        cls_i, first_i = interval_classes(steps, occ, s)
        # one independent stream per session (and, below, per ROI set) so that
        # adding or removing an ROI cannot change the matching or the nulls of
        # anything else. A single shared stream made the whole table move when
        # HC_all was added -- the same instability that forced `tiled_windows`
        # on Stage 2.
        rng = np.random.default_rng([SEED, s])

        # each ripple to its occupancy interval, keeping only the first
        # traversal -- the only phase in which `future_rew` exists
        o = occ.sort_values("start_s")
        row_of = o.index.to_numpy()
        a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
        j = np.searchsorted(a, t_rip, side="right") - 1
        jj = np.clip(j, 0, len(o) - 1)
        k = row_of[jj]
        inside = (j >= 0) & (t_rip <= b[jj]) & first_i[k]
        if inside.sum() < 3 * MIN_MATCHES:
            continue
        t_rip, d_rip, k = t_rip[inside], d_rip[inside], k[inside]
        loc = occ["loc"].to_numpy()[k].astype(int)
        grid = occ.cv_group.to_numpy()[k]
        cls = cls_i[k]
        dwell = (occ.stop_s.to_numpy() - occ.start_s.to_numpy())[k]
        lat = t_rip - occ.start_s.to_numpy()[k]
        half = d_rip / 2.0

        # PRIMARY: complete triplets, so both contrasts rest on the same
        # events. SECONDARY: pairwise matches, which keep the targets that
        # could not find a partner in BOTH control classes -- a known reward is
        # dwelt on far longer than anything else, so the strict triplet loses
        # exactly the long post-reward pauses.
        #
        # The matching serves targets in a random order, and with a control pool
        # this thin that choice moves the answer: over five seeds the HC_all
        # known-vs-nonrew contrast ranged +0.06 to +0.39. Reporting one draw
        # would be reporting the seed. So the matching is repeated N_MATCH times
        # and the per-session statistic is AVERAGED over repeats, which removes
        # the nuisance variance instead of hiding it.
        reps = []
        for _ in range(N_MATCH):
            trip, missed = match_sets(cls, dwell, lat, rng, "known_rew",
                                      ["future_rew", "nonrew"])
            if len(trip) < MIN_MATCHES:
                continue
            reps.append((trip, missed, {
                ("known_rew", "nonrew"):
                    match_sets(cls, dwell, lat, rng, "known_rew",
                               ["nonrew"])[0],
                ("future_rew", "nonrew"):
                    match_sets(cls, dwell, lat, rng, "future_rew",
                               ["nonrew"])[0]}))
        if len(reps) < N_MATCH / 2:
            continue
        trip, missed, _ = reps[0]
        bal.append(dict(
            session=s, n_triplets=float(np.mean([len(t) for t, _, _ in reps])),
            n_pairs_known_nonrew=float(np.mean(
                [len(p[("known_rew", "nonrew")]) for _, _, p in reps])),
            n_pairs_future_nonrew=float(np.mean(
                [len(p[("future_rew", "nonrew")]) for _, _, p in reps])),
            **{f"n_{c}": int((cls == c).sum()) for c in CLASSES},
            **{f"dwell_{c}": float(np.median(dwell[trip[:, i]]))
               for i, c in enumerate(CLASSES)},
            **{f"lat_{c}": float(np.median(lat[trip[:, i]]))
               for i, c in enumerate(CLASSES)},
            **{f"alldwell_{c}": float(np.median(dwell[cls == c]))
               for c in CLASSES},
            dwell_known_unmatched=(float(np.median(dwell[missed]))
                                   if len(missed) else np.nan)))

        t_fl, owner = swc.matched_flanks(t_rip, half, occ, other_t=t_rip,
                                         gap_s=FLANK_GAP_S, max_per=MAX_FLANKS)
        if not len(t_fl):
            continue
        w_fl = half[owner]
        grids = np.unique(occ.cv_group.to_numpy())

        for ri, (roiname, members) in enumerate(ROI_SETS.items()):
          cells = r[r.roi.isin(members)].cell.to_numpy()
          cells = [c for c in cells if c < len(spk[s]["spikes"])]
          if len(cells) < 2:
              continue
          for si, scheme in enumerate(SCHEMES):
            rng_p = np.random.default_rng([SEED, s, ri, si])
            loo = spt.weighted_loo(spk[s], cells, occ, grids, scheme)
            if not any(np.nansum(np.abs(T)) > 0 for T in loo.values()):
                continue
            C_rip, C_fl = swc.zscore_cells([
                spt.window_counts(spk[s], cells, t_rip - half, t_rip + half),
                spt.window_counts(spk[s], cells, t_fl - w_fl, t_fl + w_fl)])
            E_rip = spt.score_windows(C_rip, t_rip, grid, loo)
            E_fl = spt.score_windows(C_fl, t_fl, grid[owner], loo)

            ok_f = np.isfinite(E_fl).all(axis=1)
            good = np.isfinite(E_rip).all(axis=1)
            if not ok_f.any():
                continue
            stat = make_stat(E_rip, loc - 1, E_fl[ok_f],
                             loc[owner[ok_f]] - 1, owner[ok_f])

            acc, n_used = [], []
            for trip_r, _, pair_r in reps:
                # an unscorable window (no template for its held-out grid)
                # takes its whole matched set with it, so every class always
                # rests on the same events
                tr = trip_r[good[trip_r].all(axis=1)]
                if len(tr) < MIN_MATCHES:
                    continue
                obs = stat(np.arange(9), tr)
                nul_t = np.array([stat(rng_p.permutation(9), tr)
                                  for _ in range(N_PERM)], float)
                nul_c = np.array([stat(np.arange(9), _shuffle_rows(tr, rng_p))
                                  for _ in range(N_PERM)], float)
                z = {}
                for ci, c in enumerate(CLASSES):
                    for mi, m in enumerate(("rip", "diff")):
                        z[f"{m}_{c}"] = obs[ci, mi]
                        z[f"z_{m}_{c}"] = _z(obs[ci, mi], nul_t[:, ci, mi])
                for mi, m in enumerate(("rip", "diff")):
                    for ca, cb in CONTRASTS:
                        ia, ib = CLASSES.index(ca), CLASSES.index(cb)
                        v = obs[ia, mi] - obs[ib, mi]
                        z[f"d_{m}_{ca}_vs_{cb}"] = v
                        z[f"z_{m}_{ca}_vs_{cb}"] = _z(
                            v, nul_c[:, ia, mi] - nul_c[:, ib, mi])
                for (ca, cb), tab in pair_r.items():
                    tb = tab[good[tab].all(axis=1)]
                    if len(tb) < MIN_MATCHES:
                        continue
                    o2 = stat(np.arange(9), tb)
                    n2 = np.array([stat(np.arange(9), _shuffle_rows(tb, rng_p))
                                   for _ in range(N_PERM)], float)
                    for mi, m in enumerate(("rip", "diff")):
                        v = o2[0, mi] - o2[1, mi]
                        z[f"pair_d_{m}_{ca}_vs_{cb}"] = v
                        z[f"pair_z_{m}_{ca}_vs_{cb}"] = _z(
                            v, n2[:, 0, mi] - n2[:, 1, mi])
                acc.append(z)
                n_used.append(len(tr))
            if len(acc) < N_MATCH / 2:
                continue
            A = pd.DataFrame(acc)
            out = dict(session=s, subject=subj, roi=roiname, scheme=scheme,
                       n_cells=len(cells), n_triplets=float(np.mean(n_used)),
                       n_matchings=len(acc))
            out.update(A.mean(numeric_only=True).to_dict())
            rows.append(out)
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    B = pd.DataFrame(bal)
    d = os.path.join(deriv, "group", "swr",
                     # a seed/permutation sweep must NOT land next to the canonical run: the
                     # figure script globs for the newest match and picked a sweep folder
                     # once already
                     (f"ripple_content_i11_{datetime.date.today()}"
                      if (SEED, N_PERM, N_MATCH) == (42, 100, 20) else
                      f"ripple_content_i11_stability_{datetime.date.today()}"
                      f"_seed{SEED}_p{N_PERM}_m{N_MATCH}"))
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "i11_per_session.csv"), index=False)
    B.to_csv(os.path.join(d, "i11_matching_balance.csv"), index=False)
    report(R, B, d)
    print(f"\n-> {d}")


# ── 4) reporting ───────────────────────────────────────────────────────

def report(R, B, d):
    print("\n=== stillness matching: median over sessions (s) ===")
    print(f"{'class':<12s} {'dwell, all':>11s} {'dwell, matched':>15s} "
          f"{'latency, matched':>17s}")
    for c in CLASSES:
        print(f"{c:<12s} {B[f'alldwell_{c}'].median():11.3f} "
              f"{B[f'dwell_{c}'].median():15.3f} {B[f'lat_{c}'].median():17.3f}")
    kept = B.n_triplets.sum() / B.n_known_rew.sum()
    print(f"\n{len(B)} sessions matched; {int(B.n_triplets.sum())} complete "
          f"triplets from {int(B.n_known_rew.sum())} known-reward ripples "
          f"({100 * kept:.0f}%).")
    print(f"Unmatched known-reward ripples sit in LONGER dwells "
          f"(median {B.dwell_known_unmatched.median():.2f} s vs "
          f"{B.dwell_known_rew.median():.2f} s matched): the post-reward pause "
          f"has no counterpart in a class that never pauses that long, so the "
          f"strict design can only speak for the short-dwell subset.")
    print(f"Pairwise matching keeps more: "
          f"{int(B.n_pairs_known_nonrew.sum())} known-vs-nonrew and "
          f"{int(B.n_pairs_future_nonrew.sum())} future-vs-nonrew pairs.")

    R = R[R.n_triplets >= PRIMARY_MIN]
    print(f"\n=== I11, sessions with >= {PRIMARY_MIN} triplets ===")
    for scheme in sorted(R.scheme.unique()):
      print(f"\n########## cell weights: {scheme} ##########")
      R_s = R[R.scheme == scheme]
      summ = []
      for roiname in ROIS:
        g = R_s[R_s.roi == roiname]
        if len(g) < 5:
            continue
        print(f"\n  {roiname}  (n = {len(g)} sessions, "
              f"{int(g.n_triplets.sum())} triplets)")
        print(f"    {'':<20s} {'in ripple':>10s} {'p':>8s} "
              f"{'rip-flank':>10s} {'p':>8s}")
        print("    -- is there location content at all? (template-label null)")
        for c in CLASSES:
            line = f"    {c:<20s}"
            for m in ("rip", "diff"):
                v = g[f"z_{m}_{c}"].dropna()
                p = stats.ttest_1samp(v, 0)[1] if len(v) >= 5 else np.nan
                line += f" {v.mean():+10.3f} {p:8.3g}"
            print(line)
        print("    -- is it bigger for one class? (class-label shuffle)")
        for ca, cb in CONTRASTS:
            for pre in ("", "pair_"):
                col = f"{pre}z_{{m}}_{ca}_vs_{cb}"
                if col.format(m="diff") not in g.columns:
                    continue
                tag = "triplet" if pre == "" else "pairs  "
                line = f"    {ca[:5]}-{cb[:6]} {tag:<8s}"
                row = dict(roi=roiname, contrast=f"{ca}_vs_{cb}",
                           design="triplet" if pre == "" else "pair",
                           sessions=len(g))
                for m in ("rip", "diff"):
                    v = g[col.format(m=m)].dropna()
                    p = (stats.ttest_1samp(v, 0)[1] if len(v) >= 5 else np.nan)
                    line += f" {v.mean():+10.3f} {p:8.3g}"
                    row[f"z_{m}"] = float(v.mean())
                    row[f"p_{m}"] = float(p)
                    row["sessions"] = int(len(v))
                sub = g.groupby("subject")[col.format(m="diff")].mean().dropna()
                row["p_diff_subject"] = float(stats.ttest_1samp(sub, 0)[1]
                                              if len(sub) >= 5 else np.nan)
                print(line)
                summ.append(row)
    pd.DataFrame(summ).to_csv(os.path.join(d, "i11_summary.csv"), index=False)

    print("\n=== sensitivity: HC_mid / HC_anterior primary contrast "
          "(known_rew - nonrew, ripple minus flank, triplets) ===")
    R2 = pd.read_csv(os.path.join(d, "i11_per_session.csv"))
    for roiname in ("HC_all", "HC_mid", "HC_anterior"):
        for th in SENSITIVITY:
            v = R2[(R2.roi == roiname) & (R2.n_triplets >= th)
                   ]["z_diff_known_rew_vs_nonrew"].dropna()
            if len(v) >= 5:
                mark = "  <- declared" if th == PRIMARY_MIN else ""
                print(f"  {roiname:<12s} >= {th:3d} triplets: n = {len(v):2d}, "
                      f"{v.mean():+.3f}, p = "
                      f"{stats.ttest_1samp(v, 0)[1]:.3g}{mark}")

    json.dump(dict(seed=SEED, n_perm=N_PERM, roi_sets=ROI_SETS, classes=CLASSES,
                   phase="first traversal of each grid only",
                   caliper_s=CALIPER_S, rel_caliper=REL_CALIPER,
                   matched_on="dwell of the occupancy interval and latency "
                              "from arrival, nearest neighbour without "
                              "replacement; triplets complete, pairs secondary",
                   min_matches=MIN_MATCHES, primary_min=PRIMARY_MIN, n_matchings=N_MATCH,
                   sensitivity=list(SENSITIVITY),
                   primary="known_rew - nonrew, ripple minus matched flank, "
                           "HC_all, against the class-label shuffle",
                   controls="future_rew - nonrew (knowledge control); "
                            "matched flanks (template quality); "
                            "dwell+latency matching (stillness)",
                   family="4 ROIs x 3 contrasts x 2 measures x 2 designs, "
                          "uncorrected; only the primary is confirmatory",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
