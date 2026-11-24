#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COLLAPSED test: are lag-0 cells more ripple-driven than future-lag cells?

    python scripts/swr_content_explore.py lag_group

The per-cell preferred-LOCATION test is underpowered by construction: it needs a
location per cell, cross-validated, from few visits. This asks a cruder but far
better powered question that needs no location at all --

    does a cell fire MORE inside ripples than outside,
    and does that depend on its preferred LAG?

SK's reasoning: at any instant there is exactly one 'now' but several planned
futures. If ripples carry the present, lag-0 cells should be the ripple-driven
ones; if ripples carry the plan, the future-lag cells should be, and the
balance should shift between encode and execute.

MEASURE. Per cell, a ripple modulation index

    m = (rate inside ripples - rate outside) / (rate inside + rate outside)

bounded in [-1, +1], so a few high-rate cells cannot dominate, and independent
of each cell's baseline rate. 'Outside' excludes +-1 s around any ripple.

TWO UNITS OF INFERENCE, because they answer different questions and disagreeing
would matter:
  * CELL level -- all cells, maximal power, but cells within a session are NOT
    independent, so P values are anticonservative;
  * SESSION level -- the per-session mean per lag group, tested across sessions,
    consistent with every other analysis in this project.

Also reported continuously: modulation against all 12 preferred lags, which
uses the whole axis instead of an arbitrary 0-vs-rest cut.

⚠ Preferred lag is an argmax over 12 CORRELATED noisy values and is almost
uniformly distributed across cells (64-94 per bin, 80 expected by chance), so
the grouping may carry little information. A null here is at least as likely to
be about the tag as about the brain.

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
from mc.analyse.swr_content import phase_labels
from mc.analyse.swr_explore.preferred_lag_firing import preferred_lag_table

SEED = 42
ROI_GROUPS = {"HC": ["HC_anterior", "HC_mid"], "mPFC": ["mPFC"],
              "mOFC": ["mOFC"]}
PHASES = ("encode", "execute")
PAD_S = 1.0               # 'outside' must be this far from any ripple
MIN_RIP_S = 1.0           # minimum ripple seconds per cell per phase
MIN_CELLS = 3


def main():
    PL = preferred_lag_table()
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
        if not len(occ) or not len(r) or not len(t_rip):
            continue
        st = steps[steps.session == s]
        ph_i, _, _, _ = phase_labels(st, occ)
        a, b = occ.start_s.to_numpy(), occ.stop_s.to_numpy()
        half = d_rip / 2.0
        rlo, rhi = t_rip - half, t_rip + half
        for ph in PHASES:
            m = ph_i == ph
            if m.sum() < 20:
                continue
            A, B = a[m], b[m]
            # ripple segments inside this phase's intervals
            seg_lo, seg_hi, out_lo, out_hi = [], [], [], []
            for lo, hi in zip(A, B):
                k = (rhi > lo) & (rlo < hi)
                rl = np.clip(rlo[k], lo, hi); rh = np.clip(rhi[k], lo, hi)
                seg_lo.extend(rl); seg_hi.extend(rh)
                # 'outside' = the interval minus a +-PAD_S guard around ripples
                bad = np.sort(np.column_stack(
                    [np.clip(rlo[k] - PAD_S, lo, hi),
                     np.clip(rhi[k] + PAD_S, lo, hi)]), axis=0) \
                    if k.any() else np.empty((0, 2))
                cur = lo
                for bl, bh in (bad if len(bad) else []):
                    if bl > cur:
                        out_lo.append(cur); out_hi.append(bl)
                    cur = max(cur, bh)
                if cur < hi:
                    out_lo.append(cur); out_hi.append(hi)
            seg_lo = np.asarray(seg_lo); seg_hi = np.asarray(seg_hi)
            out_lo = np.asarray(out_lo); out_hi = np.asarray(out_hi)
            t_in = float((seg_hi - seg_lo).sum())
            t_out = float((out_hi - out_lo).sum())
            if t_in < MIN_RIP_S or t_out < 5.0:
                continue
            for _, cr in PL[PL.session == s].iterrows():
                c = int(cr.cell)
                if c >= len(spk[s]["spikes"]):
                    continue
                rr = r[r.cell == c]
                if not len(rr):
                    continue
                grp = next((g for g, mem in ROI_GROUPS.items()
                            if rr.roi.iloc[0] in mem), None)
                if grp is None:
                    continue
                stimes = spk[s]["spikes"][c]
                n_in = swc.count_in(stimes, seg_lo, seg_hi).sum()
                n_out = swc.count_in(stimes, out_lo, out_hi).sum()
                ri_, ro = n_in / t_in, n_out / t_out
                if ri_ + ro <= 0:
                    continue
                rows.append(dict(session=s, cell=c, roi=grp, phase=ph,
                                 pref_lag=float(cr.pref_lag),
                                 rate_in=ri_, rate_out=ro,
                                 ripmod=(ri_ - ro) / (ri_ + ro)))
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    R["lag0"] = R.pref_lag == 0
    d = os.path.join(deriv, "group", "swr",
                     f"lag_group_ripple_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "per_cell.csv"), index=False)

    print("\n=== RIPPLE MODULATION BY PREFERRED LAG ===")
    print("    m = (rate in ripples - rate outside) / (sum); outside excludes "
          f"+-{PAD_S} s around any ripple\n")
    out, ov = [], {}
    print(f"  {'ROI':6s} {'phase':8s} {'unit':8s} {'group':18s} {'n':>5s} "
          f"{'mod':>8s} {'t':>6s} {'P':>10s}")
    for grp in ROI_GROUPS:
        for ph in PHASES:
            g = R[(R.roi == grp) & (R.phase == ph)]
            if len(g) < MIN_CELLS:
                continue
            # --- cell level
            a_ = g[g.lag0].ripmod.dropna(); b_ = g[~g.lag0].ripmod.dropna()
            for nm, v in (("lag 0", a_), ("lag != 0", b_)):
                if len(v) < MIN_CELLS:
                    continue
                t, p = stats.ttest_1samp(v, 0)
                print(f"  {grp:6s} {ph:8s} {'cell':8s} {nm:18s} {len(v):5d} "
                      f"{v.mean():+8.4f} {t:+6.2f} {p:10.4g}"
                      f"{' *' if p < .05 else ''}")
                out.append(dict(roi=grp, phase=ph, unit="cell", group=nm,
                                n=int(len(v)), mod=float(v.mean()),
                                t=float(t), p=float(p)))
            if len(a_) >= MIN_CELLS and len(b_) >= MIN_CELLS:
                t, p = stats.ttest_ind(a_, b_, equal_var=False)
                print(f"  {grp:6s} {ph:8s} {'cell':8s} "
                      f"{'lag0 MINUS others':18s} {len(a_) + len(b_):5d} "
                      f"{a_.mean() - b_.mean():+8.4f} {t:+6.2f} {p:10.4g}"
                      f"{' *' if p < .05 else ''}")
                ov[f"{grp}|{ph}|cell|diff"] = dict(
                    roi=grp, phase=ph, unit="cell", group="lag0 - others",
                    mod=float(a_.mean() - b_.mean()), t=float(t), p=float(p))
            # --- session level: per-session difference between the groups
            w = (g.groupby(["session", "lag0"]).ripmod.mean().unstack())
            if {True, False} <= set(w.columns):
                dd = (w[True] - w[False]).dropna()
                if len(dd) >= 5:
                    t, p = stats.ttest_1samp(dd, 0)
                    print(f"  {grp:6s} {ph:8s} {'session':8s} "
                          f"{'lag0 MINUS others':18s} {len(dd):5d} "
                          f"{dd.mean():+8.4f} {t:+6.2f} {p:10.4g}"
                          f"{' *' if p < .05 else ''}")
                    ov[f"{grp}|{ph}|session|diff"] = dict(
                        roi=grp, phase=ph, unit="session",
                        group="lag0 - others", n=int(len(dd)),
                        mod=float(dd.mean()), t=float(t), p=float(p))
        # --- the phase interaction, session level
        w = (R[R.roi == grp].groupby(["session", "phase", "lag0"]).ripmod.mean()
             .unstack())
        if {True, False} <= set(w.columns):
            dd = (w[True] - w[False]).unstack()
            if {"encode", "execute"} <= set(dd.columns):
                di = (dd["encode"] - dd["execute"]).dropna()
                if len(di) >= 5:
                    t, p = stats.ttest_1samp(di, 0)
                    print(f"  {grp:6s} {'INTERACTION':8s} {'session':8s} "
                          f"{'(lag0-others) enc-exe':18s} {len(di):5d} "
                          f"{di.mean():+8.4f} {t:+6.2f} {p:10.4g}"
                          f"{' *' if p < .05 else ''}")
                    ov[f"{grp}|interaction"] = dict(
                        roi=grp, measure="(lag0-others) encode-execute",
                        n=int(len(di)), mod=float(di.mean()), t=float(t),
                        p=float(p))
        print()

    print("  --- continuous: modulation against all 12 preferred lags ---")
    for grp in ROI_GROUPS:
        for ph in PHASES:
            g = R[(R.roi == grp) & (R.phase == ph)]
            if len(g) < 30:
                continue
            prof = g.groupby("pref_lag").ripmod.agg(["mean", "sem", "count"])
            # circular first harmonic: is modulation tuned to lag at all?
            th = np.deg2rad(prof.index.to_numpy())
            C = np.nansum(prof["mean"] * np.cos(th)) / len(prof)
            Sn = np.nansum(prof["mean"] * np.sin(th)) / len(prof)
            print(f"    {grp:6s} {ph:8s} harmonic length {np.hypot(C, Sn):.4f} "
                  f"at {np.rad2deg(np.arctan2(Sn, C)) % 360:5.1f} deg | "
                  f"range {prof['mean'].min():+.3f} to {prof['mean'].max():+.3f}")
    pd.DataFrame(out).to_csv(os.path.join(d, "summary.csv"), index=False)
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"), indent=2)
    json.dump(dict(seed=SEED, roi_groups=ROI_GROUPS, pad_s=PAD_S,
                   min_rip_s=MIN_RIP_S,
                   measure="(rate_in - rate_out)/(rate_in + rate_out)",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"\n-> {d}")


if __name__ == "__main__":
    main()
