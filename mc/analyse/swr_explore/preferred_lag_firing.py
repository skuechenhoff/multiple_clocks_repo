#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do cells TAGGED as lag-0 location coders actually fire at their square?

    python scripts/swr_content_explore.py preferred_lag

Everything else in this project is a population read-out. This is a cell-level
check of the per-lag encoding tags: take each neuron's preferred lag from
`per_cell_ALL_ROIs.csv` (argmax over r_lag000..330_noctrl, ALL cells, no
significance filter), find the square it prefers, and ask whether it fires more
there than elsewhere -- first in ordinary task time, then inside ripples.

    preferred lag = 0 deg  -> should fire at its square when the subject is there
    preferred lag != 0 deg -> should not, or less so

TWO THINGS THAT WOULD MAKE THIS CIRCULAR, AND HOW THEY ARE AVOIDED
  * The preferred SQUARE is not stored anywhere, so it is recomputed -- but
    always from the TRAINING configurations, and the firing is then measured on
    the HELD-OUT one. Picking the square and testing it on the same data would
    guarantee a positive result.
  * No cells are dropped for significance. Selecting cells by how well they
    encode and then testing whether they encode is the same circle one step
    removed.

CELL MATCHING. `per_cell_ALL_ROIs.csv` indexes cells as (subject_id, cell_idx)
and the ripple pipeline as (session, cell). Verified: `cell_idx - 1 -> cell`
matches the ELECTRODE name for 97.9% of 955 cells (the rest are a parsing
artefact of anatomical labels containing hyphens, not a misalignment). ROI
labels disagree for 64 cells, 58 of which are flagged `roi_changed` -- a
relabelling, not a mismatch. The ripple pipeline's ROI is used throughout.

⚠ A cell's preferred lag is itself a noisy argmax over 12 CORRELATED values, so
"preferred lag = 0" is a noisy label. If every lag group shows the effect, the
label is not separating anything and the comparison is uninformative.

@author: Svenja Kuchenhoff
"""

import os
import glob
import json
import datetime

import numpy as np
import pandas as pd

from mc.analyse.roi_labels import read_per_cell_table
from scipy import stats

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc
from mc.analyse.swr_content import phase_labels

SEED = 42
PER_CELL = ("/Users/xpsy1114/Documents/projects/multiple_clocks/data/"
            "ephys_humans/derivatives/group/per_lag_encoding/"
            "2026-08-28_10-18-21_reload_from_2026-06-30_18-21-57_relabelled/"
            "per_cell_ALL_ROIs.csv")
LAGS_DEG = list(range(0, 360, 30))
ROI_GROUPS = {"HC": ["HC_anterior", "HC_mid"], "mPFC": ["mPFC"],
              "mOFC": ["mOFC"]}
PHASES = ("encode", "execute")
MIN_VISITS = 5


def preferred_lag_table():
    """(session, cell) -> preferred lag in degrees, for every cell in the file."""
    d = read_per_cell_table(PER_CELL).dropna(subset=["cell_idx", "subject_id"])
    cols = [f"r_lag{l:03d}_noctrl" for l in LAGS_DEG]
    have = [c for c in cols if c in d.columns]
    if len(have) < len(cols):
        raise RuntimeError(f"missing lag columns: {set(cols) - set(have)}")
    R = d[cols].to_numpy(float)
    ok = np.isfinite(R).any(axis=1)
    pref = np.full(len(d), np.nan)
    pref[ok] = np.array(LAGS_DEG)[np.nanargmax(R[ok], axis=1)]
    out = pd.DataFrame(dict(session=d.subject_id.astype(int).to_numpy(),
                            cell=(d.cell_idx.astype(int) - 1).to_numpy(),
                            pref_lag=pref,
                            r_lag0=d["r_lag000_noctrl"].to_numpy(float),
                            neuron=d.neuron.to_numpy()))
    return out.dropna(subset=["pref_lag"])


def main():
    rng = np.random.default_rng(SEED)
    PL = preferred_lag_table()
    print(f"  preferred lag read for {len(PL)} cells")
    print("  distribution: "
          + ", ".join(f"{int(k)}deg:{v}" for k, v in
                      PL.pref_lag.value_counts().sort_index().items()))

    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])
    sessions = sorted(set(rip.session.unique()) & set(PL.session.unique()))
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)

    rows = []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r):
            continue
        st = steps[steps.session == s]
        ph_i, _, _, _ = phase_labels(st, occ)
        loc = occ["loc"].to_numpy().astype(int)
        cvg = occ.cv_group.to_numpy()
        a, b = occ.start_s.to_numpy(), occ.stop_s.to_numpy()
        dur = b - a
        grids = np.unique(cvg)
        # ripple seconds inside each occupancy interval
        half = d_rip / 2.0
        rlo, rhi = t_rip - half, t_rip + half
        rip_s = np.array([np.clip(np.minimum(rhi, b[i])
                                  - np.maximum(rlo, a[i]), 0, None).sum()
                          for i in range(len(occ))])
        pl_s = PL[PL.session == s]
        for _, cr in pl_s.iterrows():
            c = int(cr.cell)
            if c >= len(spk[s]["spikes"]):
                continue
            rr = r[r.cell == c]
            if not len(rr):
                continue
            roi_name = rr.roi.iloc[0]
            grp = next((g for g, mem in ROI_GROUPS.items()
                        if roi_name in mem), None)
            if grp is None:
                continue
            stimes = spk[s]["spikes"][c]
            n_all = swc.count_in(stimes, a, b)
            # spikes inside ripples, per occupancy interval
            n_in = np.zeros(len(occ))
            j0 = np.searchsorted(a, t_rip, side="right") - 1
            jj = np.clip(j0, 0, len(a) - 1)
            ins = (j0 >= 0) & (t_rip <= b[jj])
            cnt = swc.count_in(stimes, rlo[ins], rhi[ins])
            np.add.at(n_in, jj[ins], cnt)
            for ph in PHASES:
                pm = ph_i == ph
                if pm.sum() < 20:
                    continue
                # LEAVE-ONE-CONFIGURATION-OUT preferred square
                for gr in grids:
                    tr = pm & (cvg != gr)
                    te = pm & (cvg == gr)
                    if te.sum() < MIN_VISITS or tr.sum() < MIN_VISITS:
                        continue
                    rate_tr = np.array([
                        n_all[tr & (loc == L)].sum() / dur[tr & (loc == L)].sum()
                        if (tr & (loc == L)).any()
                        and dur[tr & (loc == L)].sum() > 0 else np.nan
                        for L in range(1, 10)])
                    if np.isfinite(rate_tr).sum() < 3:
                        continue
                    best = int(np.nanargmax(rate_tr)) + 1
                    for lab, nsp, tsec in (("all", n_all, dur),
                                           ("ripple", n_in, rip_s)):
                        at = te & (loc == best); oth = te & (loc != best)
                        if tsec[at].sum() <= 0 or tsec[oth].sum() <= 0:
                            continue
                        rows.append(dict(
                            session=s, cell=c, roi=grp, phase=ph,
                            pref_lag=float(cr.pref_lag), window=lab,
                            fold=float(gr), best_loc=best,
                            n_at=float(nsp[at].sum()),
                            t_at=float(tsec[at].sum()),
                            n_other=float(nsp[oth].sum()),
                            t_other=float(tsec[oth].sum()),
                            rate_at=nsp[at].sum() / tsec[at].sum(),
                            rate_other=nsp[oth].sum() / tsec[oth].sum()))
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    # ⚠ the per-fold ratio is NOT usable: in a held-out configuration there is
    # very little ripple time at any one square, so 56% of folds have zero
    # ripple spikes at the preferred square and the ratio is pinned to -1 by
    # construction. Pool counts and dwell ACROSS folds first, then form one
    # ratio per cell, so empty folds contribute exposure rather than a floor.
    R["sel_fold_biased"] = ((R.rate_at - R.rate_other)
                            / (R.rate_at + R.rate_other + 1e-9))
    d = os.path.join(deriv, "group", "swr",
                     f"preferred_lag_firing_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "per_fold.csv"), index=False)
    # one value per cell, from POOLED counts and dwell
    C = (R.groupby(["session", "cell", "roi", "phase", "pref_lag", "window"])
         [["n_at", "t_at", "n_other", "t_other"]].sum().reset_index())
    ra = C.n_at / C.t_at.replace(0, np.nan)
    ro = C.n_other / C.t_other.replace(0, np.nan)
    C["sel"] = (ra - ro) / (ra + ro)
    C = C[(C.t_at > 0) & (C.t_other > 0) & ((C.n_at + C.n_other) >= 5)]
    C.to_csv(os.path.join(d, "per_cell.csv"), index=False)

    print("\n=== DOES A CELL FIRE MORE AT ITS PREFERRED SQUARE? ===")
    print("    selectivity = (rate at best - rate elsewhere) / (sum), from "
          "counts POOLED over held-out folds;")
    print("    best square chosen on TRAINING configurations only; cells with "
          "< 5 total spikes dropped\n")
    out, ov = [], {}
    print(f"  {'ROI':6s} {'phase':8s} {'window':7s} {'group':16s} {'cells':>5s} "
          f"{'sel':>8s} {'t':>6s} {'P':>10s}")
    for grp in ROI_GROUPS:
        for ph in PHASES:
            for win in ("all", "ripple"):
                g = C[(C.roi == grp) & (C.phase == ph) & (C.window == win)]
                if not len(g):
                    continue
                grps = {"pref lag = 0": g[g.pref_lag == 0],
                        "pref lag != 0": g[g.pref_lag != 0]}
                vals = {}
                for nm, gg in grps.items():
                    v = gg.sel.dropna()
                    if len(v) < 5:
                        continue
                    t, p = stats.ttest_1samp(v, 0)
                    vals[nm] = v
                    star = " *" if p < .05 else ""
                    print(f"  {grp:6s} {ph:8s} {win:7s} {nm:16s} {len(v):5d} "
                          f"{v.mean():+8.4f} {t:+6.2f} {p:10.4g}{star}")
                    rec = dict(roi=grp, phase=ph, window=win, group=nm,
                               n_cells=int(len(v)), sel=float(v.mean()),
                               t=float(t), p=float(p))
                    out.append(rec); ov[f"{grp}|{ph}|{win}|{nm}"] = rec
                if len(vals) == 2:
                    a_, b_ = vals["pref lag = 0"], vals["pref lag != 0"]
                    t, p = stats.ttest_ind(a_, b_, equal_var=False)
                    star = " *" if p < .05 else ""
                    print(f"  {grp:6s} {ph:8s} {win:7s} "
                          f"{'lag0 MINUS others':16s} {len(a_) + len(b_):5d} "
                          f"{a_.mean() - b_.mean():+8.4f} {t:+6.2f} "
                          f"{p:10.4g}{star}")
                    ov[f"{grp}|{ph}|{win}|diff"] = dict(
                        roi=grp, phase=ph, window=win, group="lag0 - others",
                        sel=float(a_.mean() - b_.mean()), t=float(t),
                        p=float(p))
            print()
    pd.DataFrame(out).to_csv(os.path.join(d, "summary.csv"), index=False)
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"), indent=2)
    json.dump(dict(seed=SEED, per_cell_source=PER_CELL, lags_deg=LAGS_DEG,
                   roi_groups=ROI_GROUPS, min_visits=MIN_VISITS,
                   note="preferred square from TRAINING configs, firing from "
                        "the held-out one; no significance filter on cells",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"-> {d}")


if __name__ == "__main__":
    main()
