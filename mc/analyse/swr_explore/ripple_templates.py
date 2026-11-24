#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does mPFC code the CURRENT square once its spatial profile is measured DURING
ripples only?

    python scripts/swr_content_explore.py ripple_templates

SK's idea: ripples inform mPFC about 'now', while outside them it codes future
lags. A profile built from all task time is then mostly a lagged profile -- the
wrong template for a ripple window. A profile built from ripple windows alone
should score the current square better.

THREE TEMPLATES per cell, all leave-one-configuration-out:
    rip    rate per square from ripple windows only
    flank  rate from each ripple's ONE matched flank: same occupancy interval,
           same width -- so the same squares, dwell and data volume. Only
           'inside a ripple' differs. THIS is the fair comparison.
    all    the standard all-time profile (~40x the data; reference only)

A ripple-only template is noisy by construction (median mPFC cell: 4 ripple
spikes per square), so comparing it with `all` would mostly measure data
volume. `flank` holds the volume fixed.

The 2 x 2 (template x test window) separates two readings: a ripple-specific
'now' code predicts rip > flank on ripple windows but NOT on flank windows.

Pre-registered in CHANGELOG.md, 2026-10-07 (RT).

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
import mc.analyse.plan_reinstatement as pr
from mc.analyse.swr_content import phase_labels

SEED = 42
N_PERM = 1000
N_FLIP = 10000
ROI_SETS = {"mPFC": ["mPFC"], "mOFC": ["mOFC"],
            "HC_all": ["HC_anterior", "HC_mid"]}
PHASES = ("encode", "execute")
TEMPLATES = ("rip", "flank", "all")
WINDOWS = ("rip", "flank")
TEMPLATE_PHASE = ("any", "encode")      # which windows build the template
MIN_RIPPLES = 20
MIN_CELLS = 2
FLANK_GAP_S = 0.05


def _stat(E, li, P):
    """(n_windows, 1 + N_PERM): current square minus the other 8, then null."""
    E = np.nan_to_num(E)
    H = np.take_along_axis(E, P[:, li].T, axis=1)
    return H - (E.sum(axis=1)[:, None] - H) / 8.0


def _z(M):
    sd = M[..., 1:].std(axis=-1)
    return np.where(sd > 0, (M[..., 0] - M[..., 1:].mean(axis=-1)) / sd,
                    np.nan)


def window_templates(C, dur, loc, grid, grids, use):
    """{grid: (n_cells, 9)} from window counts: rate per square over the
    windows of every OTHER configuration (and only those flagged `use`)."""
    out = {}
    for gr in grids:
        m = use & (grid != gr)
        n = np.zeros((C.shape[0], 9)); t = np.zeros(9)
        np.add.at(n.T, loc[m] - 1, C[:, m].T)
        np.add.at(t, loc[m] - 1, dur[m])
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(t > 0, n / t, np.nan)
        out[gr] = _zrows(rate)
    return out


def all_time_templates(spk_s, cells, occ, grids, use_occ):
    """{grid: (n_cells, 9)}: the standard place map, same normalisation."""
    o = occ[use_occ]
    return {gr: _zrows(np.array([swc.place_map(spk_s["spikes"][c], o,
                                               exclude_grid=gr)
                                 for c in cells]))
            for gr in grids}


def _zrows(rate):
    """z across squares per cell, sum to zero; unobserved squares and cells
    without a usable profile abstain (0), so no template drops a cell that
    another keeps."""
    T = np.zeros_like(rate)
    for i, m in enumerate(rate):
        z = swc.zscore_cond(m, min_obs=3, fill=0.0)
        if z is not None:
            T[i] = z
    return T


def main():
    rng = np.random.default_rng(SEED)
    P = np.vstack([np.arange(9)] + [rng.permutation(9) for _ in range(N_PERM)])
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s",
                               "subject_key"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)

    rows = []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or len(t_rip) < MIN_RIPPLES:
            continue
        subj = rip[rip.session == s].subject_key.iloc[0]
        ph_occ, _, _, _ = phase_labels(steps[steps.session == s], occ)
        a, b = occ.start_s.to_numpy(), occ.stop_s.to_numpy()
        j = np.searchsorted(a, t_rip, side="right") - 1
        jj = np.clip(j, 0, len(a) - 1)
        inside = (j >= 0) & (t_rip <= b[jj])
        t_rip, half, k = t_rip[inside], d_rip[inside] / 2.0, jj[inside]
        # ONE flank per ripple; only ripples with a flank are used at all, so
        # both templates come from the same ripples
        t_fl, owner = swc.matched_flanks(t_rip, half, occ, other_t=t_rip,
                                         gap_s=FLANK_GAP_S, max_per=1)
        if len(owner) < MIN_RIPPLES:
            continue
        t_rip, half, k = t_rip[owner], half[owner], k[owner]
        loc = occ["loc"].to_numpy()[k].astype(int)
        grid = occ.cv_group.to_numpy()[k]
        ph = ph_occ[k]
        grids = np.unique(occ.cv_group.to_numpy())
        win = {"rip": (t_rip - half, t_rip + half),
               "flank": (t_fl - half, t_fl + half)}
        dur = 2 * half

        for roiname, members in ROI_SETS.items():
            cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                     if c < len(spk[s]["spikes"])]
            if len(cells) < MIN_CELLS:
                continue
            raw = {w: swc.window_counts(spk[s], cells, *win[w])
                   for w in WINDOWS}
            Z = dict(zip(WINDOWS, swc.zscore_cells([raw[w] for w in WINDOWS])))
            for tph in TEMPLATE_PHASE:
                use = np.ones(len(t_rip), bool) if tph == "any" \
                    else ph == "encode"
                use_occ = np.ones(len(occ), bool) if tph == "any" \
                    else ph_occ == "encode"
                tmpl = {
                    "rip": window_templates(raw["rip"], dur, loc, grid,
                                            grids, use),
                    "flank": window_templates(raw["flank"], dur, loc, grid,
                                              grids, use),
                    "all": all_time_templates(spk[s], cells, occ, grids,
                                              use_occ)}
                rec = dict(session=s, subject=subj, roi=roiname,
                           template_phase=tph, n_cells=len(cells))
                for tn in TEMPLATES:
                    for wn in WINDOWS:
                        E = swc.score_windows(Z[wn], win[wn][0], grid,
                                              tmpl[tn])
                        S = _stat(E, loc - 1, P)
                        for p_ in PHASES:
                            sel = ph == p_
                            rec[f"n_{p_}"] = int(sel.sum())
                            rec[f"z_{tn}T_{wn}W_{p_}"] = (
                                float(_z(S[sel].mean(axis=0)))
                                if sel.sum() >= MIN_RIPPLES else np.nan)
                for p_ in PHASES:
                    g = lambda tn, wn: rec[f"z_{tn}T_{wn}W_{p_}"]
                    rec[f"rip_minus_flank_T_ripW_{p_}"] = g("rip", "rip") \
                        - g("flank", "rip")
                    rec[f"rip_minus_flank_T_flankW_{p_}"] = \
                        g("rip", "flank") - g("flank", "flank")
                    rec[f"interaction_{p_}"] = \
                        rec[f"rip_minus_flank_T_ripW_{p_}"] \
                        - rec[f"rip_minus_flank_T_flankW_{p_}"]
                rows.append(rec)
        print(f"  s{s:02d}  paired ripples {len(t_rip)}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"ripple_templates_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "per_session.csv"), index=False)

    measures = [c for c in R.columns if c.startswith(("z_", "rip_minus",
                                                      "interaction"))]
    out = []
    for (roiname, tph), g in R.groupby(["roi", "template_phase"]):
        for col in measures:
            v = g[col].dropna()
            if len(v) < 5:
                continue
            t, p = stats.ttest_1samp(v, 0)
            sf = pr.signflip_t(v, n_flip=N_FLIP, seed=SEED)
            ci = stats.t.interval(.95, len(v) - 1, v.mean(), stats.sem(v))
            out.append(dict(roi=roiname, template_phase=tph, measure=col,
                            n=int(len(v)), mean=float(v.mean()),
                            sem=float(stats.sem(v)), t=float(t), p=float(p),
                            p_signflip=sf["p_two_sided"],
                            ci_lo=float(ci[0]), ci_hi=float(ci[1])))
    S = pd.DataFrame(out)
    S.to_csv(os.path.join(d, "summary.csv"), index=False)

    def pick(roiname, col, tph="any"):
        g = S[(S.roi == roiname) & (S.template_phase == tph)
              & (S.measure == col)]
        return g.iloc[0].to_dict() if len(g) else None

    ov = {"preregistration": "CHANGELOG.md 2026-10-07 (RT)",
          "PRIMARY_mPFC_encode_rip_minus_flank_template":
              pick("mPFC", "rip_minus_flank_T_ripW_encode"),
          "gate_HC_all_encode_ripT_ripW": pick("HC_all", "z_ripT_ripW_encode"),
          "mPFC_encode_ripT_ripW": pick("mPFC", "z_ripT_ripW_encode"),
          "mPFC_encode_flankT_ripW": pick("mPFC", "z_flankT_ripW_encode"),
          "mPFC_encode_allT_ripW": pick("mPFC", "z_allT_ripW_encode"),
          "mPFC_execute_rip_minus_flank_template":
              pick("mPFC", "rip_minus_flank_T_ripW_execute"),
          "mPFC_encode_interaction": pick("mPFC", "interaction_encode"),
          "mPFC_encode_templates_from_encode_only":
              pick("mPFC", "rip_minus_flank_T_ripW_encode", "encode"),
          "all_cells": {f"{r_.roi}|{r_.template_phase}|{r_.measure}":
                        {k: r_[k] for k in ("n", "mean", "t", "p",
                                            "p_signflip")}
                        for _, r_ in S.iterrows()}}
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"),
              indent=2, default=float)
    json.dump(dict(seed=SEED, n_perm=N_PERM, n_flip=N_FLIP,
                   roi_sets=ROI_SETS, phases=list(PHASES),
                   templates={"rip": "ripple windows only",
                              "flank": "one matched flank per ripple "
                                       "(same interval, same width)",
                              "all": "all task time (place_map)"},
                   template_phase=list(TEMPLATE_PHASE),
                   normalisation="template z across squares, sum to zero, "
                                 "unobserved = 0 (zscore_cond, min_obs=3)",
                   cv="leave-one-configuration-out (grid_id)",
                   windows="each ripple's detection duration; its one flank",
                   min_ripples=MIN_RIPPLES, min_cells=MIN_CELLS,
                   flank_gap_s=FLANK_GAP_S,
                   preregistration="CHANGELOG.md 2026-10-07 (RT)",
                   created=datetime.datetime.now().isoformat(
                       timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"-> {d}")
    return d


if __name__ == "__main__":
    main()
