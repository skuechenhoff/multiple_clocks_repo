#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The medial-frontal gradient in HFB: where along the ventral-dorsal axis is
frontal cortex ripple-locked, and where does it respond to rewards?

Pre-registered in CHANGELOG.md, 2026-10-30 (G1-G3).

SK's hypothesis: medial PFC is ordered ventral = now / immediate future ->
dorsal = distant future. During ENCODING, hippocampal ripples deliver the 'now'
to the ventral end (perhaps down to mOFC); during EXECUTION the more dorsal part
plays the immediate future, independent of ripples. lOFC codes reward events,
before or independent of ripples.

Cells cannot test a gradient (87 in-mask mPFC cells on 19 sites, SD 2.2 mm along
the axis). HFB contacts can: 179 medial contacts in 53 sessions span MNI z
-18..+26, and 35 sessions span > 10 mm, so the gradient is tested WITHIN
session -- one slope per session, session as the unit.

  g1   ripple-locked HFB per medial contact, by phase, against z
  g2   uncover-locked reward response per frontal contact, against z and |x|
  g3   lOFC: is its pre-ripple rise the event response? does its reward
       response depend on whether a ripple follows?
  all  one extraction pass, then g1-g3

Per contact: the epoch averaged over the events at which THAT contact is clean
for the whole window (`roi_sums` on one derivation), real minus the mean of 20
shifted nulls.

Usage:
    conda activate env_multiple_clocks
    python scripts/exploratory/swr_hfb_gradient.py all
    python scripts/exploratory/swr_hfb_gradient.py tests     # from saved rows

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import datetime

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "exploratory"))
import mc.analyse.swr_bundle as sb
from swr_plan_vs_store import (load, _paths, _hfb_shifts, value_stats,
                               _save_json, _clean_ax, _dots, SEED, CM,
                               N_SHIFTS_HFB)

try:
    import fire
except ImportError:
    fire = None

ANALYSIS_NAME = "gradient_hfb"
MEDIAL_MAX_ABS_X = 20.0
FRONTAL_PART = ("lOFC", "FrontalPole", "vlPFC_IFG")
RIP_EPOCH_S, EV_EPOCH_S = (-1.0, 1.0), (-1.5, 1.5)
PERI_MS, FLANK_MS, PREPOST_MS = 250.0, (250.0, 750.0), 250.0
BASE_S, WIN_S = (-1.5, -1.0), (0.0, 0.5)
MIN_RIP, MIN_EV = 20, 10
QUIET_S = 2.0                   # G3a: no press this long before the ripple
FOLLOW_S = (0.5, 2.0)           # G3b: a ripple this long after the uncover
MIN_SPAN_MM = 5.0               # a session's contacts must span this in z
PHASES = ("first_traversal", "rest_encoding", "execution")
PHASE_C = {"first_traversal": "#FCDDE3", "rest_encoding": "#D7657F",
           "execution": "#5C1027"}
EVENT_SETS = {
    "FT correct": "kind == 'uncover' and stage == 'first uncovers' "
                  "and valence == 'correct'",
    "FT error": "kind == 'uncover' and stage == 'first uncovers' "
                "and valence == 'error'",
    "EXE correct": "kind == 'uncover' and stage_exc == 'once known' "
                   "and valence == 'correct'",
}


def _out(out_date=None):
    group, _ = _paths()
    d = os.path.join(group, f"{ANALYSIS_NAME}_{out_date or datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    return d


# ── phases ───────────────────────────────────────────────────────────────
def phase_windows(beh):
    """[start, end) per phase and grid, with the agreed exclusive boundary:
    execution starts with the first fully correct repeat."""
    rows = []
    for _, g in beh.groupby("grid_no"):
        g = g.sort_values("rep_overall").reset_index(drop=True)
        t0 = float(g.new_grid_onset.iloc[0])
        t_d = g.t_D.to_numpy(float)
        if not np.isfinite(t_d[0]):
            continue
        start = np.r_[t0, t_d[:-1]]               # start of each repeat
        ok = np.flatnonzero(g.correct.astype(int).to_numpy() == 1)
        ex0 = max(start[ok[0]], t_d[0]) if len(ok) else np.inf
        end = np.nanmax(t_d)
        for name, a, b in (("first_traversal", t0, t_d[0]),
                           ("rest_encoding", t_d[0], min(ex0, end)),
                           ("execution", ex0, end)):
            if np.isfinite(a) and np.isfinite(b) and b > a:
                rows.append((name, a, b))
    return pd.DataFrame(rows, columns=["phase", "start_s", "end_s"])


def phase_of(t, W):
    lab = np.full(len(t), "", dtype=object)
    for ph, a, b in W.itertuples(index=False):
        lab[(t >= a) & (t < b)] = ph
    return lab


# ── extraction ───────────────────────────────────────────────────────────
def contact_curve(H, idx, masks, p, t, off, n, fs, shifts, min_rows):
    """Real - mean(shifted) epoch of ONE contact over the events where it is
    clean for the whole window; None below `min_rows`."""
    from swr_hfb_ripple_coupling import roi_sums
    M = []
    for sh in shifts:
        ci = np.round((np.asarray(t) + sh) * fs).astype(int)
        s_, k_ = roi_sums(H, idx, masks, [p], ci, off, n)
        ok = k_ > 0
        if ok.sum() < min_rows:
            return None, 0
        M.append(s_[ok].mean(0))
        if sh == 0.0:
            n_real = int(ok.sum())
    M = np.stack(M)
    return M[0] - M[1:].mean(0), n_real


def ripple_values(c, t_ms):
    flank = (np.abs(t_ms) >= FLANK_MS[0]) & (np.abs(t_ms) < FLANK_MS[1])
    x = c - c[flank].mean()
    pre = x[(t_ms >= -PREPOST_MS) & (t_ms < 0)].mean()
    post = x[(t_ms > 0) & (t_ms <= PREPOST_MS)].mean()
    return dict(value=float(x[np.abs(t_ms) <= PERI_MS].mean()),
                pre=float(pre), post=float(post), direction=float(post - pre))


def event_value(c, t_ms):
    x = c - c[(t_ms >= BASE_S[0] * 1000) & (t_ms < BASE_S[1] * 1000)].mean()
    return dict(value=float(x[(t_ms >= WIN_S[0] * 1000)
                              & (t_ms < WIN_S[1] * 1000)].mean()))


def extract(out_date=None):
    from swr_hfb_ripple_coupling import load_inputs, session_ripples
    from swr_ripple_hfb_conditions import _clean_mask
    from swr_hfb_direction import PARTITION
    d = _out(out_date)
    group, b_dir = _paths()
    data, presses = load()
    rip_t, hiv, hp = load_inputs(b_dir, os.path.join(group, PARTITION))
    hp["medial"] = (hp.roi_family.isin(["mPFC", "mOFC"])
                    & (hp.mni_x.abs() <= MEDIAL_MAX_ABS_X))
    hp["frontal"] = hp.medial | hp.roi.isin(FRONTAL_PART)
    hp["lofc"] = hp.roi == "lOFC"
    shifts = _hfb_shifts()
    beh = data["behaviour"]
    rows = []
    for sess in sorted(hp[hp.frontal].session.unique()):
        g = hp[(hp.session == sess) & hp.frontal]
        D = sb.load_hfb(b_dir, sess, arrays=["hfb"])
        H, fs, n = D["hfb"], D["out_fs"], D["hfb"].shape[1]
        idx = {p: i for i, p in enumerate(D["pair_ids"])}
        g = g[g.pair_id.isin(idx)]
        masks = {p: _clean_mask(hiv[(hiv.session == sess) & (hiv.pair_id == p)]
                                [["start_s", "stop_s"]].to_numpy(float), n, fs)
                 for p in g.pair_id}
        W = phase_windows(beh[beh.session == sess])
        t_rip = session_ripples(rip_t, sess)
        P = presses[presses.session == sess].sort_values("t_s")
        tp = P.t_s.to_numpy(float)
        j = np.searchsorted(tp, t_rip, side="right") - 1
        gap = np.where(j >= 0, t_rip - tp[np.clip(j, 0, None)], np.inf)
        rip_sets = {ph: t_rip[phase_of(t_rip, W) == ph] for ph in PHASES}
        rip_sets["quiet"] = t_rip[gap > QUIET_S]
        rip_sets["after press"] = t_rip[gap <= QUIET_S]
        ev_sets = {k: P.query(q).t_s.to_numpy(float)
                   for k, q in EVENT_SETS.items()}
        ft = ev_sets["FT correct"]
        r_s = np.sort(t_rip)
        n_follow = (np.searchsorted(r_s, ft + FOLLOW_S[1], side="right")
                    - np.searchsorted(r_s, ft + FOLLOW_S[0], side="right"))
        n_early = (np.searchsorted(r_s, ft + FOLLOW_S[0], side="right")
                   - np.searchsorted(r_s, ft, side="right"))
        ev_sets["FT correct, ripple follows"] = ft[(n_follow > 0) & (n_early == 0)]
        ev_sets["FT correct, no ripple"] = ft[(n_follow == 0) & (n_early == 0)]
        for _, c in g.iterrows():
            p = c.pair_id
            base = dict(session=int(sess), pair_id=p, roi=c.roi,
                        roi_family=c.roi_family, medial=bool(c.medial),
                        lofc=bool(c.lofc), x=float(c.mni_x), y=float(c.mni_y),
                        z=float(c.mni_z))
            want_r = (list(PHASES) if c.medial else []) + \
                (["quiet", "after press"] if c.lofc else [])
            off = np.arange(int(round(RIP_EPOCH_S[0] * fs)),
                            int(round(RIP_EPOCH_S[1] * fs)))
            t_ms = off / fs * 1000
            for k in want_r:
                cur, nr = contact_curve(H, idx, masks, p, rip_sets[k], off, n,
                                        fs, shifts, MIN_RIP)
                if cur is not None:
                    rows.append(dict(base, lock="ripple", set=k, n=nr,
                                     **ripple_values(cur, t_ms)))
            off = np.arange(int(round(EV_EPOCH_S[0] * fs)),
                            int(round(EV_EPOCH_S[1] * fs)))
            t_ms = off / fs * 1000
            want_e = list(EVENT_SETS) + (["FT correct, ripple follows",
                                          "FT correct, no ripple"]
                                         if c.lofc else [])
            for k in want_e:
                cur, nr = contact_curve(H, idx, masks, p, ev_sets[k], off, n,
                                        fs, shifts, MIN_EV)
                if cur is not None:
                    rows.append(dict(base, lock="event", set=k, n=nr,
                                     **event_value(cur, t_ms)))
        print(f"  s{int(sess):02d}: {len(g)} contacts", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(d, "contact_values.csv"), index=False)
    _save_json(d, "settings.json", dict(
        analysis="gradient_hfb", created=str(datetime.date.today()),
        script=os.path.relpath(__file__, REPO), seed=SEED,
        n_shifts=N_SHIFTS_HFB, shifts_s=shifts, partition=PARTITION,
        medial=f"roi_family mPFC|mOFC and |MNI x| <= {MEDIAL_MAX_ABS_X}",
        frontal=f"medial + partition {list(FRONTAL_PART)}",
        phases="first traversal = onset -> first t_D; rest of encoding -> "
               "start of the first fully correct repeat; execution after",
        ripple=dict(epoch_s=RIP_EPOCH_S, peri_ms=PERI_MS, flank_ms=FLANK_MS,
                    prepost_ms=PREPOST_MS, min_ripples=MIN_RIP),
        event=dict(epoch_s=EV_EPOCH_S, baseline_s=BASE_S, window_s=WIN_S,
                   min_events=MIN_EV, sets=EVENT_SETS),
        quiet_s=QUIET_S, follow_s=FOLLOW_S, min_span_mm=MIN_SPAN_MM,
        unit="session; contacts within session give one slope",
        preregistration="CHANGELOG.md 2026-10-30 G1-G3"))
    return d


# ── tests ────────────────────────────────────────────────────────────────
def session_slopes(R, value_of, axis="z"):
    """One OLS slope of value on `axis` per session with >= 2 contacts
    spanning >= MIN_SPAN_MM (on z, whatever the axis)."""
    out = {}
    for s, g in R.groupby("session"):
        v = value_of(g)
        if v is None:
            continue
        a, val, zz = v
        ok = np.isfinite(val) & np.isfinite(a)
        if ok.sum() < 2 or np.ptp(zz[ok]) < MIN_SPAN_MM or np.ptp(a[ok]) == 0:
            continue
        out[int(s)] = float(np.polyfit(a[ok], val[ok], 1)[0])
    return out


def _pivot(g, lock, sets, col="value"):
    w = g[g.lock == lock].pivot_table(index=["pair_id", "z", "x", "y"],
                                      columns="set", values=col)
    w = w.reset_index()
    return w if all(s in w for s in sets) else None


def slope_test(R, lock, a_set, b_set=None, col="value", axis="z"):
    def value_of(g):
        w = _pivot(g, lock, [a_set] + ([b_set] if b_set else []), col)
        if w is None:
            return None
        v = w[a_set] - (w[b_set] if b_set else 0)
        ax = w["x"].abs() if axis == "absx" else w[axis]
        return ax.to_numpy(float), v.to_numpy(float), w["z"].to_numpy(float)
    sl = session_slopes(R, value_of, axis)
    r = value_stats(list(sl.values()))
    return r, sl


def mixed_model(R, lock, sets, col="value"):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {"error": "statsmodels not installed"}
    M = R[(R.lock == lock) & R.set.isin(sets)].copy()
    M["phase"] = pd.Categorical(M.set, categories=list(sets))
    f = smf.mixedlm(f"{col} ~ z * phase", M, groups=M["session"]).fit()
    return {k: dict(coef=float(f.params[k]), se=float(f.bse[k]),
                    p=float(f.pvalues[k])) for k in f.params.index
            if k != "Group Var"}


def tests(out_date=None, show=False):
    d = _out(out_date)
    R = pd.read_csv(os.path.join(d, "contact_values.csv"))
    med, fro, lof = R[R.medial], R, R[R.lofc]
    res = {}

    # G1
    g1 = {}
    g1["PRIMARY first_traversal - execution, slope on z"], sl_g1 = slope_test(
        med, "ripple", "first_traversal", "execution")
    for ph in PHASES:
        g1[f"{ph}, slope on z"], _ = slope_test(med, "ripple", ph)
        g1[f"{ph}, direction slope on z"], _ = slope_test(
            med, "ripple", ph, col="direction")
    g1["rest_encoding - execution, slope on z"], _ = slope_test(
        med, "ripple", "rest_encoding", "execution")
    for ax in ("absx", "y"):
        g1[f"first_traversal - execution, slope on {ax}"], _ = slope_test(
            med, "ripple", "first_traversal", "execution", axis=ax)
    g1["mixed model (all medial contacts)"] = mixed_model(
        med, "ripple", PHASES)
    res["G1"] = g1

    # G2
    g2 = {}
    g2["PRIMARY FT correct, slope on z"], sl_g2 = slope_test(
        fro, "event", "FT correct")
    g2["FT correct, slope on |x|"], _ = slope_test(fro, "event", "FT correct",
                                                   axis="absx")
    g2["FT correct - FT error, slope on z"], _ = slope_test(
        fro, "event", "FT correct", "FT error")
    g2["EXE correct, slope on z"], _ = slope_test(fro, "event", "EXE correct")
    g2["FT correct - EXE correct, slope on z"], _ = slope_test(
        fro, "event", "FT correct", "EXE correct")
    for roi, gg in fro[fro.lock == "event"].groupby(
            np.where(fro[fro.lock == "event"].medial, "medial",
                     fro[fro.lock == "event"].roi)):
        for k in ("FT correct", "EXE correct"):
            v = gg[gg.set == k].groupby("session").value.mean()
            g2[f"{roi}: {k} vs 0"] = value_stats(v.to_numpy())
    res["G2"] = g2

    # G3: lOFC, session = mean over its lOFC contacts
    def per_sess(df, lock, k, col="value"):
        return df[(df.lock == lock) & (df.set == k)].groupby("session")[col].mean()
    g3 = {}
    q_pre = per_sess(lof, "ripple", "quiet", "pre")
    a_pre = per_sess(lof, "ripple", "after press", "pre")
    g3["PRIMARY quiet ripples, pre-ripple [-250, 0)"] = value_stats(q_pre)
    g3["after-press ripples, pre-ripple [-250, 0)"] = value_stats(a_pre)
    sh = q_pre.index.intersection(a_pre.index)
    g3["after-press - quiet, pre-ripple"] = value_stats(
        (a_pre[sh] - q_pre[sh]).to_numpy())
    g3["quiet ripples, peri - flank"] = value_stats(
        per_sess(lof, "ripple", "quiet"))
    for k in ("quiet", "after press"):
        g3[f"{k} ripples, direction"] = value_stats(
            per_sess(lof, "ripple", k, "direction"))
    f_ = per_sess(lof, "event", "FT correct, ripple follows")
    n_ = per_sess(lof, "event", "FT correct, no ripple")
    sh = f_.index.intersection(n_.index)
    g3["G3b ripple follows - no ripple, uncover response"] = value_stats(
        (f_[sh] - n_[sh]).to_numpy())
    g3["G3b ripple follows, uncover response"] = value_stats(f_)
    g3["G3b no ripple, uncover response"] = value_stats(n_)
    res["G3"] = g3

    counts = dict(
        medial_contacts=int(med.pair_id.nunique()),
        medial_sessions=int(med.session.nunique()),
        g1_sessions_with_slope=len(sl_g1), g2_sessions_with_slope=len(sl_g2),
        lofc_contacts=int(lof.pair_id.nunique()),
        lofc_sessions=int(lof.session.nunique()))
    for k, v in res.items():
        print(f"\n  ══ {k} ══")
        for name, r in v.items():
            if "mean" in r:
                print(f"   {name:56s} n={r['n']:2d} {r['mean']:+.5f} ± "
                      f"{r['sem']:.5f} t={r['t']:+.2f} p_perm={r['p_perm']:.4f}")
            elif "n" in r:
                print(f"   {name:56s} n={r['n']} (too few)")
            else:
                for term, m in r.items():
                    print(f"   {name[:30]:30s} {term:26s} {m}")
    print("\n  " + json.dumps(counts))
    _save_json(d, "result.json", dict(counts=counts, tests=res,
                                      slopes=dict(G1=sl_g1, G2=sl_g2)))
    _figure(d, R, res, sl_g1, sl_g2, show)
    print(f"\n  saved -> {d}")


def _figure(d, R, res, sl_g1, sl_g2, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 4, figsize=(28 * CM, 7.5 * CM),
                             gridspec_kw={"width_ratios": [1.4, 0.7, 1.4, 1.1]})
    # a) G1: per-contact ripple-locked HFB against z, by phase
    ax = axes[0]
    med = R[R.medial & (R.lock == "ripple")]
    for ph in PHASES:
        m = med[med.set == ph]
        # session-centred so the plot shows the within-session relation tested
        v = m.value - m.groupby("session").value.transform("mean")
        zc = m.z - m.groupby("session").z.transform("mean")
        ax.scatter(zc, v, s=9, color=PHASE_C[ph], edgecolor="0.3", lw=0.3,
                   label=ph.replace("_", " "), alpha=0.85)
        ok = np.isfinite(zc) & np.isfinite(v)
        b = np.polyfit(zc[ok], v[ok], 1)
        xx = np.linspace(zc.min(), zc.max(), 10)
        ax.plot(xx, np.polyval(b, xx), color=PHASE_C[ph], lw=1.6,
                path_effects=None)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("MNI z − session mean (mm)\nventral ← → dorsal")
    ax.set_ylabel("ripple-locked HFB,\nperi − flank (z), session-centred")
    ax.set_title("G1: medial contacts", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.38), ncol=3)
    _clean_ax(ax)
    # b) G1 primary: per-session slopes
    ax = axes[1]
    _dots(ax, 0, list(sl_g1.values()), PHASE_C["rest_encoding"], rng)
    r = res["G1"]["PRIMARY first_traversal - execution, slope on z"]
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xticks([0], ["first trav. −\nexecution"], fontsize=7)
    ax.set_ylabel("slope on z (HFB z / mm)")
    ax.set_title(f"G1 primary\np = {r.get('p_perm', np.nan):.3f}, "
                 f"n = {r['n']}", fontsize=9)
    _clean_ax(ax)
    # c) G2: FT correct uncover response against z, all frontal contacts
    ax = axes[2]
    ev = R[(R.lock == "event") & (R.set == "FT correct")]
    for lab, m, col, mk in (("medial", ev[ev.medial], "#448363", "o"),
                            ("lOFC", ev[ev.roi == "lOFC"], "#a30d6c", "s"),
                            ("other frontal",
                             ev[~ev.medial & (ev.roi != "lOFC")], "#8C8C8C",
                             "^")):
        ax.scatter(m.z, m.value, s=10, color=col, marker=mk, alpha=0.7, lw=0,
                   label=f"{lab} ({len(m)})")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    r = res["G2"]["PRIMARY FT correct, slope on z"]
    ax.set_xlabel("MNI z (mm)")
    ax.set_ylabel("uncover response 0–0.5 s (z)")
    ax.set_title(f"G2: correct first-traversal uncovers\nwithin-session slope "
                 f"p = {r.get('p_perm', np.nan):.3f} (n = {r['n']})",
                 fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.32), ncol=3)
    _clean_ax(ax)
    # d) G3
    ax = axes[3]
    lof = R[R.lofc]
    vals = [lof[(lof.lock == "ripple") & (lof.set == k)].groupby("session").pre
            .mean() for k in ("quiet", "after press")]
    vals += [lof[(lof.lock == "event") & (lof.set == k)].groupby("session")
             .value.mean() for k in ("FT correct, ripple follows",
                                     "FT correct, no ripple")]
    for i, (v, c) in enumerate(zip(vals, ["#a30d6c", "#d4749f",
                                          "#0e3d3a", "#9ec6b8"])):
        _dots(ax, i, v.to_numpy(), c, rng)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.axvline(1.5, color="0.6", lw=0.6, ls=":")
    ax.set_xticks(range(4), ["quiet\nripples", "after-press\nripples",
                             "uncover,\nripple\nfollows", "uncover,\nno ripple"],
                  fontsize=7)
    ax.set_ylabel("lOFC HFB (z)")
    p1 = res["G3"]["PRIMARY quiet ripples, pre-ripple [-250, 0)"]
    p2 = res["G3"]["G3b ripple follows - no ripple, uncover response"]
    ax.set_title(f"G3: lOFC\nquiet pre-ripple p = {p1.get('p_perm', np.nan):.3f}"
                 f"; follows − none p = {p2.get('p_perm', np.nan):.3f}",
                 fontsize=8)
    _clean_ax(ax)
    fig.text(0.01, -0.2,
             "HFB per contact = real − mean of 20 shifted nulls, over the events "
             "at which that contact is clean. a: per-contact ripple-locked HFB "
             "(±250 ms minus 250–750 ms flanks), centred within session, with "
             "pooled\nfits for display; the test (b) is one slope per session "
             "(≥ 2 contacts spanning ≥ 5 mm). c: uncover response vs MNI z. d: "
             "lOFC per session; left of the dotted line pre-ripple [−250, 0) ms, "
             "right uncover response 0–0.5 s.", fontsize=7, color="0.3")
    fig.tight_layout()
    for e in (".png", ".pdf"):
        fig.savefig(os.path.join(d, "gradient_hfb" + e), dpi=300,
                    bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def run_all(out_date=None, show=False):
    extract(out_date)
    tests(out_date, show)


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"all": run_all, "extract": extract, "tests": tests})
    else:
        run_all()
