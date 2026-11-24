#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controls for the HC-mPFC ripple co-firing (`tuned_cofiring_hc_mpfc.py`,
panel a) and the content coupling (`now_coupling_hc_mpfc.py`). (2026-11-23)

Concern. Ripple windows are each ripple's OWN duration; ordinary moments all
have ONE width (the session's median ripple duration). A longer window holds
more spikes in every cell of both regions, so HC and mPFC counts correlate
across ripples for that reason alone -- and ordinary moments cannot show it.
The same shared gain, multiplied by square preferences both regions happen to
share, could leak into the content coupling (the same-square shuffle keeps
the square, not the gain).

Variants (same sessions, events, z-scoring and functions as the originals):
    own        ripple windows = own duration (as reported)
    fixed      ripple and flank windows = peak +- the ordinary-moment
               half-width, i.e. the same width as ordinary moments
    ordinary   ordinary moments, one width (the session's median ripple
               duration) -- the baseline for `fixed`
    ordinary_matched
               the same ordinary moments, each with a width drawn (seeded,
               with replacement) from the durations of the same session's
               ripples in the same phase -- the baseline for `own`: both
               then carry the same window-length variability (2026-11-24)
    *_gainfree the per-event mean z across a region's cells is removed
               before the location read-out (content coupling only)
Population co-firing r_pop = Pearson r across events of the mean z of all
HC cells and of all mPFC cells; for `own` and `ordinary_matched` also with
window duration partialled out of both.
Contrasts (per session, then across sessions): own - ordinary_matched,
fixed - ordinary, and own - ordinary (unmatched; kept for reference). Content coupling = r - same-square-shuffle null (imported
`coupling`, unchanged).

    python scripts/cofiring_window_controls.py
    -> group/swr/cofiring_window_controls_<date>/

@author: Svenja Kuchenhoff
"""
import os
import sys
import json
import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import location_content_ripples_by_phase as lc
from now_coupling_hc_mpfc import coupling, corr_rows, PAIRS
from mc.analyse.swr_explore.context_content import group

rrsa, swl, swc, pcn = lc.rrsa, lc.swl, lc.swc, lc.pcn
SEED = pcn.SEED
PHASES = pcn.PHASES
KINDS = ("own", "fixed", "ordinary", "ordinary_matched")
CONTRASTS = (("own", "ordinary_matched"), ("fixed", "ordinary"), ("own", "ordinary"))
OUT = os.path.join(rrsa._derivatives(), "group", "swr",
                   f"cofiring_window_controls_{datetime.date.today()}")


def resid(y, x):
    """y with its linear dependence on x removed."""
    X = np.column_stack([np.ones_like(x), x])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def matched_halfwidths(ev, s):
    """One half-width per ordinary moment, drawn from the same session's
    ripples of the same phase."""
    rng = np.random.default_rng([SEED, s, 11])
    hm = np.full(len(ev["tc"]), ev["h"])
    for ph in PHASES:
        hm[ev["masks_c"][ph]] = rng.choice(ev["half"][ev["masks"][ph]],
                                           ev["masks_c"][ph].sum())
    return hm


def region_counts(spk_s, cells, ev, hm):
    """z-scored (cells, events) counts for ripples (own / fixed) and
    ordinary moments (fixed / duration-matched), each z-scored exactly like
    the reported analysis."""
    h = ev["h"]
    own, _ = swc.zscore_cells([
        swc.window_counts(spk_s, cells, ev["t_rip"] - ev["half"], ev["t_rip"] + ev["half"]),
        swc.window_counts(spk_s, cells, ev["t_fl"] - ev["w_fl"], ev["t_fl"] + ev["w_fl"])])
    fixed, _ = swc.zscore_cells([
        swc.window_counts(spk_s, cells, ev["t_rip"] - h, ev["t_rip"] + h),
        swc.window_counts(spk_s, cells, ev["t_fl"] - h, ev["t_fl"] + h)])
    ctl, = swc.zscore_cells([
        swc.window_counts(spk_s, cells, ev["tc"] - h, ev["tc"] + h)])
    ctl_m, = swc.zscore_cells([
        swc.window_counts(spk_s, cells, ev["tc"] - hm, ev["tc"] + hm)])
    return {"own": own, "fixed": fixed, "ordinary": ctl,
            "ordinary_matched": ctl_m}


def session(s, spk_s, r, occ, t_all, d_all, steps_s):
    ev = lc.events(s, occ, t_all, d_all, steps_s)
    if ev is None:
        return []
    lc.STATS = ("CURRENT", "fut1")
    OW = lc.occ_weights(steps_s, occ)
    P0 = np.arange(9)[None]
    kc = ev["kc"]
    hm = matched_halfwidths(ev, s)
    sq_c = occ["loc"].to_numpy()[kc].astype(int)
    idx = {"own": ev["k"], "fixed": ev["k"], "ordinary": kc, "ordinary_matched": kc}
    sq = {"own": ev["loc"], "fixed": ev["loc"], "ordinary": sq_c,
          "ordinary_matched": sq_c}
    grid = {"own": ev["grid"], "fixed": ev["grid"],
            "ordinary": occ.cv_group.to_numpy()[kc],
            "ordinary_matched": occ.cv_group.to_numpy()[kc]}
    tt = {"own": ev["t_rip"], "fixed": ev["t_rip"], "ordinary": ev["tc"],
          "ordinary_matched": ev["tc"]}
    masks = {"own": ev["masks"], "fixed": ev["masks"], "ordinary": ev["masks_c"],
             "ordinary_matched": ev["masks_c"]}
    dur = {"own": 2 * ev["half"], "ordinary_matched": 2 * hm}
    Z, V = {}, {}
    for roi in ("HC_all", "mPFC"):
        cells = [c for c in r[r.roi.isin(pcn.ROI_SETS[roi])].cell.to_numpy()
                 if c < len(spk_s["spikes"])]
        if len(cells) < pcn.MIN_CELLS:
            return []
        _, loo = swc.build_templates(spk_s, cells, occ, ev["grids"])
        C = region_counts(spk_s, cells, ev, hm)
        for kind, Ck in C.items():
            Z[(roi, kind)] = Ck.mean(0)                      # population z
            for gf in (False, True):
                Cg = Ck - Ck.mean(0, keepdims=True) if gf else Ck
                E = swc.score_windows(Cg, tt[kind], grid[kind], loo)
                for stat in ("CURRENT", "fut1"):
                    W, dfn = OW[stat]
                    v = lc.stat_perm(E, W[idx[kind]], P0)[:, 0]
                    ok = np.isfinite(E).all(1) & dfn[idx[kind]]
                    V[(roi, stat, kind, gf)] = np.where(ok, v, np.nan)
    rows = []
    for kind in KINDS:
        for ph in PHASES:
            mk = masks[kind][ph]
            if mk.sum() < 20:
                continue
            h_, m_ = Z[("HC_all", kind)][mk], Z[("mPFC", kind)][mk]
            row = dict(session=int(s), kind=kind, phase=ph, n=int(mk.sum()),
                       r_pop=float(corr_rows(h_, m_[None])[0]))
            if kind in dur:
                d = dur[kind][mk]
                row["r_pop_dur_partial"] = float(corr_rows(resid(h_, d), resid(m_, d)[None])[0])
                row["r_dur_hc"] = float(corr_rows(d, h_[None])[0])
                row["r_dur_mpfc"] = float(corr_rows(d, m_[None])[0])
            for gf in (False, True):
                for pair, (sh, sm) in PAIRS.items():
                    rng = np.random.default_rng([SEED, s, PHASES.index(ph),
                                                 0 if kind in ("own", "fixed") else 1])
                    c = coupling(V[("HC_all", sh, kind, gf)][mk],
                                 V[("mPFC", sm, kind, gf)][mk], sq[kind][mk], rng)
                    if c is not None:
                        row[f"{pair}{' | gainfree' if gf else ''}"] = c["coupling"]
            rows.append(row)
    return rows


def run():
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2", "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)
    jobs = []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < pcn.MIN_RIPPLES:
            continue
        jobs.append(delayed(session)(s, spk[s], r, occ, t_rip, d_rip,
                                     steps[steps.session == s]))
    out = Parallel(n_jobs=6, verbose=5)(jobs)
    return pd.DataFrame([x for o in out for x in o])


def summarise(D):
    res = {"n_sessions": int(D.session.nunique()), "tests": {}, "contrasts": {}}
    cols = ["r_pop", "r_pop_dur_partial", "r_dur_hc", "r_dur_mpfc"] + \
        [c for c in D.columns if " x " in c]
    for kind in KINDS:
        for ph in PHASES:
            g = D[(D.kind == kind) & (D.phase == ph)]
            for c in cols:
                if c in g and g[c].notna().any():
                    res["tests"][f"{c} | {kind} | {ph}"] = group(g[c].to_numpy(float))
    piv = D.pivot_table(index="session", columns=["kind", "phase"], values=cols)
    for c in cols:
        for a, b in CONTRASTS:
            for ph in PHASES:
                if (c, a, ph) in piv and (c, b, ph) in piv:
                    d = (piv[(c, a, ph)] - piv[(c, b, ph)]).dropna()
                    res["contrasts"][f"{c} | {a} - {b} | {ph}"] = group(d.to_numpy())
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    D = run()
    D.to_csv(os.path.join(OUT, "per_session.csv"), index=False)
    res = summarise(D)
    res["config"] = dict(variants=list(KINDS), contrasts=CONTRASTS, gainfree=True,
                         seed=SEED, events="lc.events",
                         ordinary_matched="half-widths drawn with replacement from "
                                          "the same session's ripples of the same phase")
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1, default=float)
    for k, v in list(res["tests"].items()) + list(res["contrasts"].items()):
        print(f"{k:70s} {v.get('mean', np.nan):+.4f}  p {v.get('p', np.nan):.3f}  n {v['n']}")
