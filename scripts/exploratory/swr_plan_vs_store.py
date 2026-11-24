#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STORAGE or NEXT-STEP PLANNING: what is the post-correct ripple increase tied to?

Pre-registered in CHANGELOG.md, 2026-10-29 (tests T1-T5; T4 lives in
`mc/analyse/swr_explore/path_content.py`).

A co-author reads the ripple increase after correct uncovers in the first
traversal as consolidation, not next-step planning. SK: 'loading' the current
state into mPFC is itself storage -- it predicts communication at EVERY new
discovery, not at D. So these tests split STORAGE (tied to new information
arriving) from NEXT-STEP PLANNING (tied to the upcoming move and goal):

  knowledge    T1a  ripple rate after the first D (next goal known for the first
                    time) vs after the first A-C (next goal unknown), with the
                    minimum detectable effect
  hfb_state    T1b  the same split in mPFC HFB, uncover- and ripple-locked; and
                    execution errors vs execution correct uncovers
  two_clocks   T2   is the increase locked to the uncover or to the departure
                    (the next press)?
  behaviour    T3   do pause ripples predict later retrieval of the item just
                    found (storage) or the efficiency of the next step (planning)?
  novelty      T5   do ripples rise after a NEW square without a reward?

Everything about ripple rate goes through the reported pipeline:
`fin.matched_sets` (arrow presses matched on the stillness that follows),
`rip.peri_event_rate`, `fin.window_stats` and `fin.cluster_test`. The primary
reading is stillness-matched (uncover minus its matched arrow press, raw); the
own-baseline reading (-1.5..-1.0 s) is secondary.

Usage:
    conda activate env_multiple_clocks
    python scripts/exploratory/swr_plan_vs_store.py knowledge
    python scripts/exploratory/swr_plan_vs_store.py hfb_state
    python scripts/exploratory/swr_plan_vs_store.py two_clocks
    python scripts/exploratory/swr_plan_vs_store.py behaviour
    python scripts/exploratory/swr_plan_vs_store.py novelty

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import datetime
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "exploratory"))
import mc.analyse.swr_io as swr_io
import mc.analyse.ripples as rip
import mc.analyse.swr_phases as grid
import mc.analyse.swr_bundle as sbu
import scripts.swr_final_ripple_analysis as fin

try:
    import fire
except ImportError:
    fire = None

ANALYSIS_NAME = "plan_vs_store"
SEED = 42
N_FLIP = 10000                 # sign flips for every per-session value test
PAD_S = 0.25                   # artifact pad, as the reported panels
WIN = fin.PRIMARY_WINDOW       # (0.0, 0.5) s after the press
BASE = grid.BASE_B             # (-1.5, -1.0) s, the agreed own baseline
FT = "kind == 'uncover' and stage == 'first uncovers'"
MAX_PAUSE_S = 2.0              # T3: pause window capped here
MIN_EVENTS_T3 = 20
STATE_C = {"A": "#F15A29", "B": "#F7931E", "C": "#C7C6E2", "D": "#6B60AA"}
POOL_C = "#6E6E6E"             # pooled A-C, a group rather than a state
CORRECT_C, ERROR_C = "#0e3d3a", "#a30d6c"
CM = 1 / 2.54


# ── loading ──────────────────────────────────────────────────────────────
def _paths():
    root = swr_io.get_data_root()
    group = os.path.join(swr_io.derivatives_dir(root), "group", "swr")
    return group, os.path.join(group, "bundle_v2")


def _out(sub, out_date=None):
    group, _ = _paths()
    d = os.path.join(group, f"{ANALYSIS_NAME}_{out_date or datetime.date.today()}", sub)
    os.makedirs(d, exist_ok=True)
    return d


def load(pad_s=PAD_S):
    """Bundle and press table exactly as `swr_figure_phase_panels.run`, plus
    each uncover's square, repeat and departure, and first-traversal novelty."""
    group, bundle = _paths()
    data = sbu.repad_bundle(rip.load_bundle(bundle), float(pad_s))
    presses = fin.press_table(data, bundle,
                              os.path.join(group, "press_categories.csv"),
                              rebuild=False)
    presses = grid.add_exclusive_stage(data, presses)
    return data, add_uncover_info(data, presses)


def add_uncover_info(data, presses):
    unc = data["uncover"][["session", "t_s", "loc", "rep_overall"]]
    out = []
    for s, p in presses.groupby("session"):
        u = unc[unc.session == s].drop(columns="session") \
            .rename(columns={"t_s": "t_unc"}).sort_values("t_unc")
        p = pd.merge_asof(p.sort_values("t_s"), u, left_on="t_s",
                          right_on="t_unc", direction="nearest",
                          tolerance=0.03)
        p.loc[p.kind == "move", ["loc", "rep_overall"]] = np.nan
        out.append(p.drop(columns="t_unc"))
    P = pd.concat(out, ignore_index=True)
    P["departure_s"] = P.t_s + P.still_next_s
    # NOVEL = this square has not been uncovered before in this block. The
    # first traversal is the block's first repeat, so 'before in this block'
    # is 'before in this first traversal'.
    ft = P.query(FT).dropna(subset=["loc"]).sort_values("t_s")
    seen = ft.groupby(["session", "grid_no"])["loc"].transform(
        lambda s: s.duplicated()).astype(bool)
    P["novel"] = np.nan
    P.loc[ft.index, "novel"] = (~seen).astype(float)
    return P


# ── shared statistics ────────────────────────────────────────────────────
def value_stats(v):
    """Mean, SEM, t and a two-sided sign-flip p of per-session values -- the
    same arithmetic as `fin.window_stats`, for values that are not a window of
    one profile (differences of windows, correlations)."""
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size < 3:
        return dict(n=int(v.size))
    t, p = stats.ttest_1samp(v, 0.0)
    rng = np.random.default_rng(SEED)
    null = (rng.choice([-1.0, 1.0], size=(N_FLIP, v.size)) * v).mean(1)
    sem = float(v.std(ddof=1) / np.sqrt(v.size))
    crit = float(stats.t.ppf(0.975, v.size - 1))
    return dict(n=int(v.size), mean=float(v.mean()), sem=sem,
                ci_low=float(v.mean() - crit * sem),
                ci_high=float(v.mean() + crit * sem),
                t=float(t), df=int(v.size - 1), p=float(p),
                p_perm=float((1 + np.sum(np.abs(null) >= abs(v.mean())))
                             / (1 + N_FLIP)))


def mde(v, power=0.8, alpha=0.05):
    """Smallest true mean a two-sided one-sample t test on these sessions
    detects with `power`, given their spread."""
    v = np.asarray(v, float)
    n = v.size
    return float((stats.t.ppf(1 - alpha / 2, n - 1) + stats.t.ppf(power, n - 1))
                 * v.std(ddof=1) / np.sqrt(n))


def window_values(profiles, centres, window):
    m = (centres >= window[0]) & (centres < window[1])
    return {u: float(np.nanmean(p[m])) for u, p in profiles.items()}


def profile_stats(prof, centres, label, n_perm=rip.N_SIGN_FLIPS):
    """`fin.window_stats` on WIN plus `fin.cluster_test`, as the panels do."""
    if len(prof) < 3:
        return None
    s = fin.window_stats(prof, centres, WIN)
    _, cl = fin.cluster_test(prof, centres, label, n_perm)
    return dict(s, clusters=cl)


def _fmt(r):
    if r is None or "mean_hz" not in r:
        return "n/a"
    cl = "; ".join(f"{c['start_s']:+.2f}..{c['stop_s']:+.2f}s p={c['p']:.3f}"
                   for c in r.get("clusters", [])) or "no cluster"
    return (f"n={r['n_units']:2d} {r['mean_hz']:+.4f} Hz "
            f"[{r['ci_low_hz']:+.4f}, {r['ci_high_hz']:+.4f}] "
            f"t({r['df']})={r['t']:+.2f} p_perm={r['p_perm']:.4f} [{cl}]")


def _fmtv(r, unit=""):
    if "mean" not in r:
        return f"n={r['n']} (too few)"
    return (f"n={r['n']:2d} {r['mean']:+.4f}{unit} ± {r['sem']:.4f} "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] t({r['df']})="
            f"{r['t']:+.2f} p_perm={r['p_perm']:.4f}")


def _save_json(d, name, payload):
    with open(os.path.join(d, name), "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: o.item()
                  if hasattr(o, "item") else str(o))


def _clean_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _trace(ax, prof, centres, colour, label, ls="-"):
    X = np.vstack([prof[u] for u in sorted(prof)])
    m = gaussian_filter1d(np.nanmean(X, 0), 1.0)      # display only
    se = gaussian_filter1d(np.nanstd(X, 0, ddof=1) / np.sqrt(len(X)), 1.0)
    ax.plot(centres, m, color=colour, lw=1.4, ls=ls,
            label=f"{label} (n = {len(X)})")
    ax.fill_between(centres, m - se, m + se, color=colour, alpha=0.2, lw=0)


def _cluster_bars(ax, res, colour, y):
    for c in (res or {}).get("clusters", []):
        ax.plot([c["start_s"], c["stop_s"]], [y, y], color=colour, lw=3,
                solid_capstyle="butt")


def _dots(ax, x, vals, colour, rng, marker="o"):
    v = np.asarray([u for u in vals if np.isfinite(u)])
    if not v.size:
        return
    ax.scatter(x + rng.uniform(-0.12, 0.12, v.size), v, s=9, color=colour,
               alpha=0.55, lw=0, zorder=2)
    ax.errorbar(x, v.mean(), v.std(ddof=1) / np.sqrt(v.size), fmt=marker,
                ms=5.5, mfc=colour, mec="#1a1a1a", ecolor="#1a1a1a",
                capsize=2.5, zorder=3)


# ── stillness-matched cells (T1a, T5) ────────────────────────────────────
def cell(data, presses, query, min_events=fin.MIN_EVENTS):
    """One condition through the reported pipeline: the stillness-matched
    excess (target - matched arrow press, raw) and the own-baseline reading."""
    tgt, ctl, _, bal = fin.matched_sets(presses, query, fin.CALIPER_S,
                                        fin.REL_CALIPER, SEED)
    every = {int(s): g.t_s.to_numpy(float)
             for s, g in presses.query(query).groupby("session")}
    centres, raw_t, _ = fin.rates(data, tgt, "session", min_events)
    _, raw_c, _ = fin.rates(data, ctl, "session", min_events)
    c_a, raw_a, cnt = fin.rates(data, every, "session", min_events)
    centres = centres if centres is not None else c_a
    return centres, dict(
        matched=fin.difference(raw_t, raw_c),
        own=grid.base(raw_a, c_a, BASE) if raw_a else {},
        n_events=int(cnt["n_events_used"]),
        n_matched=int(bal.n_matched.sum()) if len(bal) else 0,
        n_unmatched=int(bal.n_unmatched.sum()) if len(bal) else 0,
        still_s=float(bal.still_target_s.median()) if len(bal) else np.nan)


def run_cells(data, presses, cells, contrasts, min_events=fin.MIN_EVENTS):
    """Every cell and contrast, both readings. Contrasts are paired on the
    sessions that have every named cell (`rip.contrast_profiles`)."""
    prof, meta, centres = {}, {}, None
    for name, q in cells.items():
        c, out = cell(data, presses, q, min_events)
        centres = c if c is not None else centres
        prof[name] = {"matched": out.pop("matched"), "own": out.pop("own")}
        meta[name] = out
        print(f"    {name:18s} {out['n_events']:6d} ev | matched "
              f"{out['n_matched']:5d} / {out['n_matched'] + out['n_unmatched']:5d}"
              f" | still {out['still_s']:.2f} s")
    res = {}
    for reading in ("matched", "own"):
        P = {k: v[reading] for k, v in prof.items()}
        res[reading] = {k: profile_stats(v, centres, k) for k, v in P.items()}
        for cname, w in contrasts.items():
            res[reading][cname] = profile_stats(
                rip.contrast_profiles(P, w), centres, cname)
    return centres, prof, meta, res


# ── T1a: does the increase need a known next goal? ───────────────────────
T1_CELLS = {
    "A-C": FT + " and valence == 'correct' and reward != 'D'",
    "D": FT + " and valence == 'correct' and reward == 'D'",
    **{x: FT + f" and valence == 'correct' and reward == '{x}'"
       for x in "ABC"},
}
T1_CONTRASTS = {"D - (A-C)": {"D": 1.0, "A-C": -1.0}}


def knowledge(show=False, out_date=None):
    d = _out("T1a_knowledge", out_date)
    np.random.seed(SEED)
    data, presses = load()
    print("\n  T1a: correct first-traversal uncovers, D vs A-C")
    centres, prof, meta, res = run_cells(data, presses, T1_CELLS, T1_CONTRASTS)

    # power: how big a D - (A-C) difference could these sessions have shown?
    con = rip.contrast_profiles({k: v["matched"] for k, v in prof.items()},
                                T1_CONTRASTS["D - (A-C)"])
    v_con = np.array(list(window_values(con, centres, WIN).values()))
    v_ac = np.array(list(window_values(prof["A-C"]["matched"], centres,
                                       WIN).values()))
    power = dict(n_sessions=int(v_con.size),
                 sd_contrast_hz=float(v_con.std(ddof=1)),
                 mde80_hz=mde(v_con), ac_effect_hz=float(v_ac.mean()),
                 mde80_over_ac_effect=float(mde(v_con) / v_ac.mean()))
    for reading in ("matched", "own"):
        print(f"\n  ══ {reading} ══")
        for k, r in res[reading].items():
            print(f"   {k:12s} {_fmt(r)}")
    print(f"\n  power: D - (A-C) MDE(80%) = {power['mde80_hz']:+.4f} Hz on "
          f"{power['n_sessions']} sessions; the A-C effect itself is "
          f"{power['ac_effect_hz']:+.4f} Hz (MDE = "
          f"{power['mde80_over_ac_effect']:.2f} x it)")

    pd.DataFrame([dict(reading=rd, cell=k, **{x: r[x] for x in
                                              ("n_units", "mean_hz", "sem_hz",
                                               "ci_low_hz", "ci_high_hz", "t",
                                               "df", "p", "p_perm")})
                  for rd in res for k, r in res[rd].items() if r]) \
        .to_csv(os.path.join(d, "stats.csv"), index=False)
    _save_json(d, "result.json", dict(
        settings=dict(analysis="T1a_knowledge", created=str(datetime.date.today()),
                      script=os.path.relpath(__file__, REPO), seed=SEED,
                      pad_s=PAD_S, window_s=list(WIN), baseline_s=list(BASE),
                      caliper_s=fin.CALIPER_S, rel_caliper=fin.REL_CALIPER,
                      min_events=fin.MIN_EVENTS, cells=T1_CELLS,
                      contrasts=T1_CONTRASTS, unit="session",
                      preregistration="CHANGELOG.md 2026-10-29 T1a"),
        cell_counts=meta, stats=res, power=power))
    _figure_t1a(d, centres, prof, res, power, show)
    print(f"\n  saved -> {d}")


def _figure_t1a(d, centres, prof, res, power, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 3, figsize=(19 * CM, 6.5 * CM),
                             gridspec_kw={"width_ratios": [1.5, 1, 1.3]})
    ax = axes[0]
    ax.axvspan(*WIN, color="0.92", lw=0)
    for x in "ABCD":
        _trace(ax, prof[x]["matched"], centres, STATE_C[x], f"first {x}")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlim(-1.5, 2.0)
    ax.set_xlabel("time from correct uncover (s)")
    ax.set_ylabel("uncover − matched\narrow press (Hz)")
    ax.set_title("ripple rate, first traversal", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.42))
    _clean_ax(ax)

    ax = axes[1]
    for i, x in enumerate(["A", "B", "C", "D"]):
        _dots(ax, i, list(window_values(prof[x]["matched"], centres,
                                        WIN).values()), STATE_C[x], rng)
    _dots(ax, 4.3, list(window_values(prof["A-C"]["matched"], centres,
                                      WIN).values()), POOL_C, rng, "s")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xticks([0, 1, 2, 3, 4.3], ["A", "B", "C", "D", "A–C"])
    ax.set_ylabel("0–0.5 s, uncover −\nmatched press (Hz)")
    ax.set_title("per state", fontsize=10)
    _clean_ax(ax)

    ax = axes[2]
    con = rip.contrast_profiles({k: v["matched"] for k, v in prof.items()},
                                T1_CONTRASTS["D - (A-C)"])
    ax.axvspan(*WIN, color="0.92", lw=0)
    _trace(ax, con, centres, STATE_C["D"], "D − (A–C)")
    r = res["matched"]["D - (A-C)"]
    _cluster_bars(ax, r, STATE_C["D"], ax.get_ylim()[0])
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlim(-1.5, 2.0)
    ax.set_xlabel("time from correct uncover (s)")
    ax.set_ylabel("D − (A–C) (Hz)")
    ax.set_title(f"D − (A–C): {r['mean_hz']:+.3f} Hz, p = {r['p_perm']:.2f}\n"
                 f"MDE(80%) = {power['mde80_hz']:.3f} Hz "
                 f"= {power['mde80_over_ac_effect']:.1f}× the A–C effect",
                 fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.42))
    _clean_ax(ax)
    fig.text(0.01, -0.26,
             "Stillness-matched reading: each uncover minus an arrow press "
             "followed by the same stillness; mean ± SEM over sessions "
             "(line smoothed for display only).\nGrey band: the tested 0–0.5 s "
             "window. Dots: sessions; marker: mean ± SEM. Bar: cluster p < 0.05 "
             "(sign-flip, 0.3 s sliding window).", fontsize=7, color="0.3")
    fig.tight_layout()
    for e in (".png", ".pdf"):
        fig.savefig(os.path.join(d, "T1a_knowledge" + e), dpi=300,
                    bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ── T1b: the same split in frontal HFB ───────────────────────────────────
HFB_ROIS = ("mPFC", "mOFC", "lOFC", "MedialFrontal")
PART_ROIS = ("mPFC", "mOFC", "lOFC")
MEDIAL_MAX_ABS_X = 20.0        # MedialFrontal as in swr_ripple_hfb_conditions
N_SHIFTS_HFB = 20
EV_EPOCH_S, RIP_EPOCH_S = (-1.5, 1.5), (-1.0, 1.0)
PERI_MS, FLANK_MS = 250.0, (250.0, 750.0)
MIN_EV_HFB, MIN_RIP_HFB = 10, 10
POST_S = 2.0                   # a ripple belongs to the press before it
HFB_SETS = {
    "D": FT + " and valence == 'correct' and reward == 'D'",
    "A-C": FT + " and valence == 'correct' and reward != 'D'",
    "FT correct": FT + " and valence == 'correct'",
    "exec correct": "kind == 'uncover' and valence == 'correct' "
                    "and stage_exc == 'once known'",
    "exec error": "kind == 'uncover' and valence == 'error' "
                  "and stage_exc == 'once known'",
}
HFB_CONTRASTS = {"D - (A-C)": ("D", "A-C"),
                 "exec error - exec correct": ("exec error", "exec correct")}


def _hfb_shifts():
    from swr_hfb_ripple_coupling import SHIFT_RANGE_S
    rng = np.random.default_rng(SEED)
    lo, hi = SHIFT_RANGE_S
    return [0.0] + [float(x) for x in rng.uniform(lo, hi, N_SHIFTS_HFB)
                    * rng.choice([-1, 1], N_SHIFTS_HFB)]


def _epoch_means(H, idx, masks, pids, t, off, fs, n, shifts, min_rows):
    """(n_shifts + 1, T) mean epoch over events for one ROI, or None.
    `roi_sums` keeps only derivations clean for the whole window."""
    from swr_hfb_ripple_coupling import roi_sums
    out = []
    for sh in shifts:
        ci = np.round((t + sh) * fs).astype(int)
        s_, k_ = roi_sums(H, idx, masks, pids, ci, off, n)
        ok = k_ > 0
        if ok.sum() < min_rows:
            return None
        out.append((s_[ok] / k_[ok, None]).mean(0))
    return np.stack(out)


def hfb_state(show=False, out_date=None):
    import pickle
    import mc.analyse.swr_bundle as sb
    from swr_hfb_ripple_coupling import load_inputs, session_ripples
    from swr_ripple_hfb_conditions import _clean_mask
    from swr_hfb_direction import PARTITION
    d = _out("T1b_hfb_state", out_date)
    group, b_dir = _paths()
    _, presses = load()
    rip_t, hiv, hp = load_inputs(b_dir, os.path.join(group, PARTITION))
    hp["MedialFrontal"] = (hp.roi_family.isin(["mPFC", "mOFC"])
                           & (hp.mni_x.abs() <= MEDIAL_MAX_ABS_X))
    shifts = _hfb_shifts()
    sessions = sorted(hp[hp.roi.isin(PART_ROIS) | hp.MedialFrontal]
                      .session.unique())
    print(f"\n  T1b: {len(sessions)} sessions, {N_SHIFTS_HFB} shifted nulls")

    E = {"event": {}, "ripple": {}}     # lock -> (set, roi) -> {sess: curve}
    counts = []
    for sess in sessions:
        g = hp[(hp.session == sess) & (hp.roi.isin(PART_ROIS) | hp.MedialFrontal)]
        D = sb.load_hfb(b_dir, sess, arrays=["hfb"])
        H, fs, n = D["hfb"], D["out_fs"], D["hfb"].shape[1]
        idx = {p: i for i, p in enumerate(D["pair_ids"])}
        g = g[g.pair_id.isin(idx)]
        masks = {p: _clean_mask(hiv[(hiv.session == sess) & (hiv.pair_id == p)]
                                [["start_s", "stop_s"]].to_numpy(float), n, fs)
                 for p in g.pair_id}
        pids = {r: g[g.roi == r].pair_id.tolist() for r in PART_ROIS}
        pids["MedialFrontal"] = g[g.MedialFrontal].pair_id.tolist()
        P = presses[presses.session == sess].sort_values("t_s")
        # ripple -> its nearest preceding press within POST_S (conditions rule)
        t_rip = session_ripples(rip_t, sess)
        tp = P.t_s.to_numpy(float)
        j = np.searchsorted(tp, t_rip, side="right") - 1
        has = (j >= 0) & (t_rip - tp[np.clip(j, 0, None)] <= POST_S)
        for name, q in HFB_SETS.items():
            ev = P.query(q).t_s.to_numpy(float)
            in_set = np.isin(tp, ev)
            r_set = t_rip[has & in_set[np.clip(j, 0, None)]]
            for lock, t, ep, mn in (("event", ev, EV_EPOCH_S, MIN_EV_HFB),
                                    ("ripple", r_set, RIP_EPOCH_S, MIN_RIP_HFB)):
                off = np.arange(int(round(ep[0] * fs)), int(round(ep[1] * fs)))
                for roi in HFB_ROIS:
                    if not pids[roi] or len(t) < mn:
                        continue
                    M = _epoch_means(H, idx, masks, pids[roi], t, off, fs, n,
                                     shifts, mn)
                    if M is None:
                        continue
                    E[lock].setdefault((name, roi), {})[int(sess)] = \
                        M[0] - M[1:].mean(0)
                    counts.append(dict(session=int(sess), set=name, lock=lock,
                                       roi=roi, n=int(len(t))))
        print(f"  s{int(sess):02d} done", end="\r")
    t_ms = {"event": np.arange(int(EV_EPOCH_S[0] * fs), int(EV_EPOCH_S[1] * fs))
            / fs * 1000, "ripple": np.arange(int(RIP_EPOCH_S[0] * fs),
                                             int(RIP_EPOCH_S[1] * fs)) / fs * 1000}
    pd.DataFrame(counts).to_csv(os.path.join(d, "counts.csv"), index=False)

    # per-session values
    def value(lock, curve):
        t = t_ms[lock]
        if lock == "event":
            c = curve - curve[(t >= BASE[0] * 1000) & (t < BASE[1] * 1000)].mean()
            return float(c[(t >= WIN[0] * 1000) & (t < WIN[1] * 1000)].mean())
        peri = np.abs(t) <= PERI_MS
        flank = (np.abs(t) >= FLANK_MS[0]) & (np.abs(t) < FLANK_MS[1])
        return float(curve[peri].mean() - curve[flank].mean())

    V = {lock: {k: {s: value(lock, c) for s, c in v.items()}
                for k, v in E[lock].items()} for lock in E}
    rows, res = [], {}
    for lock in V:
        for (name, roi), vals in V[lock].items():
            r = value_stats(list(vals.values()))
            res[f"{lock} | {roi} | {name}"] = r
            rows.append(dict(lock=lock, roi=roi, test=name, **r))
        for cname, (a, b) in HFB_CONTRASTS.items():
            for roi in HFB_ROIS:
                va, vb = V[lock].get((a, roi), {}), V[lock].get((b, roi), {})
                sh = sorted(set(va) & set(vb))
                r = value_stats([va[s] - vb[s] for s in sh])
                if "mean" in r:
                    r["mde80"] = mde([va[s] - vb[s] for s in sh])
                res[f"{lock} | {roi} | {cname}"] = r
                rows.append(dict(lock=lock, roi=roi, test=cname, **r))
    St = pd.DataFrame(rows)
    St.to_csv(os.path.join(d, "stats.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\n" + St[["lock", "roi", "test", "n", "mean", "sem", "t", "p_perm"]]
          .round(4).to_string(index=False))

    # time-resolved PRIMARY: mPFC uncover-locked D - (A-C), cluster test
    from mc.analyse.swr_sakon import cluster_perm_time
    cl_out = {}
    for lock, key in (("event", "mPFC"), ("event", "MedialFrontal")):
        a, b = E[lock].get(("D", key), {}), E[lock].get(("A-C", key), {})
        sh = sorted(set(a) & set(b))
        t = t_ms[lock]
        keep = t >= BASE[1] * 1000
        X = np.stack([_base_smooth(a[s] - b[s], t) for s in sh])[:, keep]
        _, clusters, pvals, _ = cluster_perm_time(X, n_perm=N_FLIP, seed=SEED)
        cl_out[f"{lock} | {key} | D - (A-C)"] = [
            dict(start_ms=float(t[keep][i0]), stop_ms=float(t[keep][i1 - 1]),
                 p=float(p)) for (i0, i1), p in zip(clusters, pvals)]
    print("\n  clusters, uncover-locked D - (A-C):",
          json.dumps(cl_out, indent=1))
    _save_json(d, "result.json", dict(
        settings=dict(analysis="T1b_hfb_state", created=str(datetime.date.today()),
                      script=os.path.relpath(__file__, REPO), seed=SEED,
                      n_shifts=N_SHIFTS_HFB, shifts_s=shifts,
                      partition=PARTITION, rois=HFB_ROIS,
                      medialfrontal=f"roi_family mPFC|mOFC and |MNI x| <= "
                                    f"{MEDIAL_MAX_ABS_X}",
                      sets=HFB_SETS, contrasts=HFB_CONTRASTS,
                      event=dict(epoch_s=EV_EPOCH_S, baseline_s=BASE,
                                 window_s=WIN, min_events=MIN_EV_HFB),
                      ripple=dict(epoch_s=RIP_EPOCH_S, peri_ms=PERI_MS,
                                  flank_ms=FLANK_MS, min_ripples=MIN_RIP_HFB,
                                  assignment=f"nearest preceding press within "
                                             f"{POST_S} s"),
                      estimate="real - mean of the shifted nulls, per session",
                      unit="session",
                      preregistration="CHANGELOG.md 2026-10-29 T1b"),
        stats=res, clusters=cl_out))
    np.savez_compressed(os.path.join(d, "curves.npz"),
                        **{f"{lock}|{k[0]}|{k[1]}": np.stack(
                            [v[s] for s in sorted(v)])
                           for lock in E for k, v in E[lock].items()},
                        **{f"{lock}|{k[0]}|{k[1]}|sessions": np.array(sorted(v))
                           for lock in E for k, v in E[lock].items()},
                        t_event_ms=t_ms["event"], t_ripple_ms=t_ms["ripple"])
    _figure_t1b(d, E, V, t_ms, res, cl_out, show)
    print(f"\n  saved -> {d}")


def _base_smooth(c, t_ms):
    """Own baseline (-1.5..-1.0 s) removed, 50 ms sliding mean -- the sliding
    analysis' curve treatment."""
    c = c - c[(t_ms >= BASE[0] * 1000) & (t_ms < BASE[1] * 1000)].mean()
    return np.convolve(c, np.ones(5) / 5, mode="same")


def _figure_t1b(d, E, V, t_ms, res, cl_out, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from swr_hfb_direction import ROI_COL
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 4, figsize=(26 * CM, 7 * CM),
                             gridspec_kw={"width_ratios": [1.4, 1, 1, 1]})
    ax = axes[0]
    t = t_ms["event"]
    for name, col in (("A-C", POOL_C), ("D", STATE_C["D"])):
        cur = E["event"].get((name, "mPFC"), {})
        X = np.stack([_base_smooth(c, t) for c in cur.values()])
        m, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))
        ax.plot(t, m, color=col, lw=1.4,
                label=f"first {name.replace('-', '–')} (n = {len(X)})")
        ax.fill_between(t, m - se, m + se, color=col, alpha=0.2, lw=0)
    for c in cl_out.get("event | mPFC | D - (A-C)", []):
        if c["p"] < 0.05:
            ax.plot([c["start_ms"], c["stop_ms"]], [ax.get_ylim()[0]] * 2,
                    color=STATE_C["D"], lw=3)
    ax.axvspan(BASE[0] * 1000, BASE[1] * 1000, color="0.92", lw=0)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("time from correct uncover (ms)")
    ax.set_ylabel("mPFC HFB (z, real − null)")
    ax.set_title("mPFC, uncover-locked", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.42))
    _clean_ax(ax)
    panels = (("event", "D - (A-C)", "uncover-locked, 0–0.5 s\nD − (A–C)"),
              ("ripple", "D - (A-C)", "ripple-locked, peri − flank\nD − (A–C)"),
              ("event", "exec error - exec correct",
               "uncover-locked, execution\nerror − correct"))
    for ax, (lock, cname, title) in zip(axes[1:], panels):
        a, b = HFB_CONTRASTS[cname]
        for i, roi in enumerate(HFB_ROIS):
            va, vb = V[lock].get((a, roi), {}), V[lock].get((b, roi), {})
            sh = sorted(set(va) & set(vb))
            _dots(ax, i, [va[s] - vb[s] for s in sh], ROI_COL.get(roi, "#888"),
                  rng, "D" if roi == "MedialFrontal" else "o")
            r = res[f"{lock} | {roi} | {cname}"]
            if "p_perm" in r:
                ax.text(i, 1.0, f"p={r['p_perm']:.2f}", fontsize=6.5,
                        ha="center", transform=ax.get_xaxis_transform())
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_xticks(range(len(HFB_ROIS)), HFB_ROIS, rotation=45, ha="right",
                      fontsize=8)
        ax.set_ylabel("HFB difference (z)")
        ax.set_title(title, fontsize=9, pad=12)
        _clean_ax(ax)
    fig.text(0.01, -0.26,
             "HFB = real − mean of 20 shifted nulls per session. Uncover-locked: "
             "own baseline −1.5…−1.0 s (grey), window 0–0.5 s; ripple-locked: "
             "±250 ms minus 250–750 ms flanks,\nripples whose nearest preceding "
             "press (≤ 2 s) is the uncover. Dots: sessions; marker: mean ± SEM; "
             "p: sign-flip. MedialFrontal (diamond) overlaps mPFC + mOFC.",
             fontsize=7, color="0.3")
    fig.tight_layout()
    for e in (".png", ".pdf"):
        fig.savefig(os.path.join(d, "T1b_hfb_state" + e), dpi=300,
                    bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ── T2: the reward clock or the departure clock? ─────────────────────────
def rate_profiles(data, events, min_events=fin.MIN_EVENTS):
    """`rip.rate_by_unit` (unit = session) WITHOUT its 2 s de-duplication.
    Departures of de-duplicated uncovers can sit < 2 s apart, and dropping
    some would break the uncover <-> departure pairing; the uncovers themselves
    were de-duplicated by `fin.matched_sets` already, so for them this is
    identical to `rate_by_unit`."""
    out, centres = {}, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s, t in events.items():
            t = np.sort(np.asarray(t, float))
            if t.size < min_events:
                continue
            prof = []
            for _, ripples, iv in rip.derivations(data, s):
                centres, p = rip.peri_event_rate(t, ripples, iv)
                prof.append(np.nanmean(p, axis=0))
            if prof:
                out[int(s)] = np.nanmean(np.vstack(prof), axis=0)
    return centres, out


def _clock_run(data, presses, min_still, win_s):
    q = FT + f" and valence == 'correct' and still_next_s >= {min_still}"
    # controls must also be followed by >= min_still, or their two windows
    # would overlap
    pool = presses[~(presses.key.isin(fin.CONTROL_KEYS)
                     & (presses.still_next_s < min_still))]
    tgt, ctl, _, bal = fin.matched_sets(pool, q, fin.CALIPER_S,
                                        fin.REL_CALIPER, SEED)
    dep = {}
    for side, ev in (("t", tgt), ("c", ctl)):
        dep[side] = {}
        for s, t in ev.items():
            p = presses[presses.session == s].drop_duplicates("t_s") \
                .set_index("t_s")
            dep[side][s] = t + p.still_next_s.loc[t].to_numpy(float)
    centres, u_t = rate_profiles(data, tgt)
    _, u_c = rate_profiles(data, ctl)
    _, d_t = rate_profiles(data, dep["t"])
    _, d_c = rate_profiles(data, dep["c"])
    post = fin.difference(u_t, u_c)          # excess, uncover-locked
    pre = fin.difference(d_t, d_c)           # excess, departure-locked
    v_post = window_values(post, centres, (0.0, win_s))
    v_pre = window_values(pre, centres, (-win_s, 0.0))
    sh = sorted(set(v_post) & set(v_pre))
    res = dict(
        n_matched=int(bal.n_matched.sum()), n_unmatched=int(bal.n_unmatched.sum()),
        still_target_s=float(bal.still_target_s.median()),
        still_control_s=float(bal.still_control_s.median()),
        primary_post_minus_predeparture=value_stats(
            [v_post[s] - v_pre[s] for s in sh]),
        excess_post=value_stats([v_post[s] for s in sh]),
        excess_predeparture=value_stats([v_pre[s] for s in sh]),
        # the same windows without a control: the uncover alone
        raw_post_minus_predeparture=value_stats(
            [window_values({s: u_t[s]}, centres, (0.0, win_s))[s]
             - window_values({s: d_t[s]}, centres, (-win_s, 0.0))[s]
             for s in sh]))
    res["cluster_uncover_locked"] = profile_stats(post, centres, "post")
    res["cluster_departure_locked"] = profile_stats(pre, centres, "pre")
    return centres, dict(post=post, pre=pre, u_t=u_t, u_c=u_c, d_t=d_t,
                         d_c=d_c, v_post=v_post, v_pre=v_pre), res


def two_clocks(show=False, out_date=None):
    d = _out("T2_two_clocks", out_date)
    np.random.seed(SEED)
    data, presses = load()
    settings = {"primary": (1.0, 0.5), "long pauses": (1.5, 0.75)}
    out, keep = {}, {}
    for name, (min_still, win_s) in settings.items():
        print(f"\n  T2 {name}: stillness >= {min_still} s, windows {win_s} s")
        centres, P, r = _clock_run(data, presses, min_still, win_s)
        out[name] = dict(min_still_s=min_still, window_s=win_s, **r)
        keep[name] = P
        print(f"    matched {r['n_matched']} (unmatched {r['n_unmatched']}), "
              f"still {r['still_target_s']:.2f} / {r['still_control_s']:.2f} s")
        for k in ("primary_post_minus_predeparture", "excess_post",
                  "excess_predeparture", "raw_post_minus_predeparture"):
            print(f"    {k:34s} {_fmtv(r[k], ' Hz')}")
    _save_json(d, "result.json", dict(
        settings=dict(analysis="T2_two_clocks", created=str(datetime.date.today()),
                      script=os.path.relpath(__file__, REPO), seed=SEED,
                      pad_s=PAD_S, events="correct first-traversal uncovers",
                      control="arrow presses with the same following stillness "
                              "(fin.matched_sets), themselves >= min_still",
                      departure="the next press of any kind",
                      variants=settings, unit="session",
                      preregistration="CHANGELOG.md 2026-10-29 T2"),
        results=out))
    _figure_t2(d, centres, keep, out, show)
    print(f"\n  saved -> {d}")


def _figure_t2(d, centres, keep, out, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(SEED)
    P, r = keep["primary"], out["primary"]
    w = r["window_s"]
    fig, axes = plt.subplots(1, 3, figsize=(20 * CM, 6.5 * CM),
                             gridspec_kw={"width_ratios": [1.3, 1.3, 0.8]})
    for ax, (key, lock, xl, win, xlim) in zip(axes[:2], (
            ("post", "uncover", "time from correct uncover (s)", (0, w),
             (-1.0, 2.0)),
            ("pre", "departure", "time from the next press (s)", (-w, 0),
             (-2.0, 1.0)))):
        ax.axvspan(*win, color="0.92", lw=0)
        t_, c_ = (P["u_t"], P["u_c"]) if key == "post" else (P["d_t"], P["d_c"])
        _trace(ax, t_, centres, CORRECT_C, "correct uncover")
        _trace(ax, c_, centres, "#8C8C8C", "matched arrow press", ls="--")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_xlim(*xlim)
        ax.set_xlabel(xl)
        ax.set_ylabel("ripple rate (Hz)")
        res = r["cluster_uncover_locked" if key == "post"
                else "cluster_departure_locked"]
        _cluster_bars(ax, res, CORRECT_C, ax.get_ylim()[0])
        ax.set_title(f"{lock}-locked", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.42))
        _clean_ax(ax)
    ax = axes[2]
    sh = sorted(set(P["v_post"]) & set(P["v_pre"]))
    a = np.array([P["v_post"][s] for s in sh])
    b = np.array([P["v_pre"][s] for s in sh])
    for x, y in zip(a, b):
        ax.plot([0, 1], [x, y], color="0.8", lw=0.6, zorder=1)
    _dots(ax, 0, a, CORRECT_C, rng)
    _dots(ax, 1, b, CORRECT_C, rng)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xticks([0, 1], [f"0–{w:g} s after\nuncover",
                           f"{w:g} s before\ndeparture"], fontsize=7)
    ax.set_ylabel("uncover − matched press (Hz)")
    pr = r["primary_post_minus_predeparture"]
    ax.set_title(f"after − before: {pr['mean']:+.3f} Hz\np = {pr['p_perm']:.3f}",
                 fontsize=8)
    _clean_ax(ax)
    fig.text(0.01, -0.26,
             f"Correct first-traversal uncovers followed by ≥ "
             f"{r['min_still_s']:g} s of stillness, each matched to an arrow "
             f"press followed by the same stillness (n = {r['n_matched']} "
             f"pairs).\nMean ± SEM over sessions (lines smoothed for display "
             f"only). Grey: tested windows. Bars: excess clusters p < 0.05. "
             f"Right: per-session excess, paired.", fontsize=7, color="0.3")
    fig.tight_layout()
    for e in (".png", ".pdf"):
        fig.savefig(os.path.join(d, "T2_two_clocks" + e), dpi=300,
                    bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ── T3: behaviour ────────────────────────────────────────────────────────
STATES = ("A", "B", "C", "D")


def _manhattan(a, b):
    a, b = np.asarray(a, float) - 1, np.asarray(b, float) - 1
    return np.abs(a // 3 - b // 3) + np.abs(a % 3 - b % 3)


def pause_rate(t0, w, derivs):
    """HC ripples per clean second in [t0, t0 + w], averaged over derivations;
    NaN on a derivation whose window is less than half clean."""
    R = []
    for _, ripples, iv in derivs:
        r = np.sort(ripples)
        n = (np.searchsorted(r, t0 + w, side="right")
             - np.searchsorted(r, t0, side="left")).astype(float)
        expo = rip.clean_seconds(iv, t0, t0 + w)
        R.append(np.where(expo >= rip.MIN_CLEAN_FRAC * w,
                          n / np.maximum(expo, 1e-12), np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.vstack(R), axis=0) if R else np.full(len(t0), np.nan)


def segment_table(data, presses, s):
    """Every correct uncover of one session with its pause ripple rate, the
    segment INTO it (since the previous correct uncover) and the segment OUT
    of it (to the next one): wrong uncovers and path excess."""
    U = data["uncover"][data["uncover"].session == s].sort_values("t_s")
    P = presses[presses.session == s]
    still = P[P.kind == "uncover"].set_index("t_s").still_next_s
    U = pd.merge_asof(U, still.reset_index().sort_values("t_s"), on="t_s",
                      direction="nearest", tolerance=0.03)
    moves = np.sort(P[P.kind == "move"].t_s.to_numpy(float))
    beh = data["behaviour"][data["behaviour"].session == s]
    rows = []
    for gn, g in U.groupby("grid_no"):
        g = g.sort_values("t_s").reset_index(drop=True)
        r0 = int(g.rep_overall.min())
        b = beh[beh.grid_no == gn]
        solved = b.loc[b.correct.astype(int) == 1, "rep_overall"]
        fs = int(solved.min()) if len(solved) else np.inf
        cor = np.flatnonzero(g.correct.to_numpy(int) == 1)
        t = g.t_s.to_numpy(float)
        for k, i in enumerate(cor):
            prev = cor[k - 1] if k > 0 else -1
            nxt = cor[k + 1] if k + 1 < len(cor) else None

            def seg(a, b_):
                """wrong uncovers and path excess for (a, b_]"""
                t_a = t[a] if a >= 0 else -np.inf
                wrong = int((g.correct.to_numpy(int)[a + 1:b_] == 0).sum())
                n_mv = int(np.searchsorted(moves, t[b_])
                           - np.searchsorted(moves, t_a, side="right"))
                ex = (n_mv - _manhattan(g["loc"].iloc[a], g["loc"].iloc[b_])
                      if a >= 0 else np.nan)
                return wrong, ex
            w_in, ex_in = seg(prev, i)
            w_out, ex_out = seg(i, nxt) if nxt is not None else (np.nan, np.nan)
            rows.append(dict(session=int(s), grid_no=int(gn), t_s=t[i],
                             rep=int(g.rep_overall.iloc[i]), r0=r0, fs=fs,
                             state=g.state.iloc[i],
                             still_s=float(g.still_next_s.iloc[i]),
                             wrong_in=w_in, excess_in=ex_in,
                             wrong_out=w_out, excess_out=ex_out))
    T = pd.DataFrame(rows)
    if not len(T):
        return T
    T["pause_s"] = np.minimum(T.still_s, MAX_PAUSE_S)
    ok = np.isfinite(T.pause_s) & (T.pause_s > 0)
    T["rate"] = np.nan
    T.loc[ok, "rate"] = pause_rate(T.t_s[ok].to_numpy(float),
                                   T.pause_s[ok].to_numpy(float),
                                   rip.derivations(data, s))
    # the same reward on the NEXT repeat of this grid: its segment INTO it
    nxt = T.drop_duplicates(["grid_no", "rep", "state"]) \
        .set_index(["grid_no", "rep", "state"])
    key = list(zip(T.grid_no, T.rep + 1, T.state))
    T["next_rep_wrong_in"] = [nxt.wrong_in.get(k_, np.nan) for k_ in key]
    T["next_rep_excess_in"] = [nxt.excess_in.get(k_, np.nan) for k_ in key]
    return T


def partial_spearman(y, x, cont, dummies):
    """Spearman of y and x with `cont` (ranked) and `dummies` partialled out:
    ranks residualised on the covariates by OLS, then correlated."""
    Z = [np.ones(len(y))] + [stats.rankdata(c) for c in cont] + list(dummies)
    Z = np.column_stack(Z)

    def res(v):
        r = stats.rankdata(v)
        return r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]
    ey, ex = res(y), res(x)
    if ey.std() == 0 or ex.std() == 0:
        return np.nan
    return float(np.corrcoef(ey, ex)[0, 1])


def _state_dummies(st):
    return [(st == x).astype(float) for x in STATES[1:]
            if (st == x).any() and not (st == x).all()]


T3_TESTS = {
    # name: (event mask, outcome, continuous covariates, extra label)
    "T3a storage: first discovery -> same reward next repeat, wrong uncovers":
        ("first", "next_rep_wrong_in", ("pause_s", "wrong_in")),
    "T3b planning: later repeats -> next segment, wrong uncovers":
        ("later", "wrong_out", ("pause_s", "rep_idx")),
    "T3a sec: first discovery -> same reward next repeat, path excess":
        ("first", "next_rep_excess_in", ("pause_s", "wrong_in")),
    "T3b sec: later repeats -> next segment, path excess":
        ("later", "excess_out", ("pause_s", "rep_idx")),
    "T3b sec: learning repeats -> next segment, wrong uncovers":
        ("learning", "wrong_out", ("pause_s", "rep_idx")),
    "T3b sec: execution repeats -> next segment, wrong uncovers":
        ("execution", "wrong_out", ("pause_s", "rep_idx")),
    "T3a control: first discovery A-C -> next first-traversal segment (explore)":
        ("first_AC", "wrong_out", ("pause_s", "wrong_in")),
}


def behaviour(show=False, out_date=None):
    d = _out("T3_behaviour", out_date)
    data, presses = load()
    T = pd.concat([segment_table(data, presses, s)
                   for s in sorted(presses.session.unique())], ignore_index=True)
    T["rep_idx"] = T.rep - T.r0
    T.to_csv(os.path.join(d, "events.csv"), index=False)
    masks = {"first": T.rep == T.r0,
             "first_AC": (T.rep == T.r0) & (T.state != "D"),
             "later": T.rep > T.r0,
             "learning": (T.rep > T.r0) & (T.rep < T.fs),
             "execution": T.rep >= T.fs}
    rows, res, per = [], {}, {}
    for name, (mk, outcome, cont) in T3_TESTS.items():
        z = {}
        for s, g in T[masks[mk]].groupby("session"):
            cols = ["rate", outcome] + list(cont)
            g = g.dropna(subset=cols)
            if len(g) < MIN_EVENTS_T3 or g[outcome].nunique() < 2 \
                    or g.rate.nunique() < 2:
                continue
            r = partial_spearman(g[outcome].to_numpy(float),
                                 g.rate.to_numpy(float),
                                 [g[c].to_numpy(float) for c in cont],
                                 _state_dummies(g.state.to_numpy()))
            if np.isfinite(r):
                z[int(s)] = float(np.arctanh(np.clip(r, -0.999, 0.999)))
        r = value_stats(list(z.values()))
        r["n_events"] = int(masks[mk].sum())
        res[name], per[name] = r, z
        rows.append(dict(test=name, outcome=outcome, covariates=",".join(cont),
                         **r))
        print(f"  {name:76s} {_fmtv(r)}")
    pd.DataFrame(rows).to_csv(os.path.join(d, "stats.csv"), index=False)
    pd.DataFrame([dict(test=k, session=s, z=v) for k, z in per.items()
                  for s, v in z.items()]).to_csv(
        os.path.join(d, "per_session.csv"), index=False)
    desc = dict(
        events=int(len(T)),
        p_ripple_in_pause_first=float((T[masks["first"]].rate > 0).mean()),
        p_ripple_in_pause_later=float((T[masks["later"]].rate > 0).mean()),
        median_pause_first_s=float(T[masks["first"]].pause_s.median()),
        median_pause_later_s=float(T[masks["later"]].pause_s.median()),
        mean_next_rep_wrong_in=float(T[masks["first"]].next_rep_wrong_in.mean()),
        mean_wrong_out_later=float(T[masks["later"]].wrong_out.mean()))
    print("\n  " + json.dumps(desc, indent=1))
    _save_json(d, "result.json", dict(
        settings=dict(analysis="T3_behaviour", created=str(datetime.date.today()),
                      script=os.path.relpath(__file__, REPO), seed=SEED,
                      pad_s=PAD_S, max_pause_s=MAX_PAUSE_S,
                      predictor="HC ripples per clean second from the correct "
                                "uncover to the next press (capped), mean over "
                                "derivations",
                      statistic="partial Spearman within session (ranks "
                                "residualised on covariates + state dummies), "
                                "Fisher z, sign-flip over sessions",
                      min_events=MIN_EVENTS_T3,
                      tests={k: dict(events=v[0], outcome=v[1],
                                     covariates=list(v[2]))
                             for k, v in T3_TESTS.items()},
                      unit="session",
                      preregistration="CHANGELOG.md 2026-10-29 T3"),
        descriptives=desc, stats=res))
    _figure_t3(d, per, res, show)
    print(f"\n  saved -> {d}")


def _figure_t3(d, per, res, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(SEED)
    names = list(T3_TESTS)
    short = ["storage:\nnext-repeat\nerrors", "planning:\nnext-segment\nerrors",
             "storage:\npath\nexcess", "planning:\npath\nexcess",
             "planning:\nlearning", "planning:\nexecution",
             "control:\nexploration"]
    cols = [CORRECT_C, ERROR_C, CORRECT_C, ERROR_C, ERROR_C, ERROR_C, POOL_C]
    fig, ax = plt.subplots(figsize=(17 * CM, 7 * CM))
    for i, (n, c) in enumerate(zip(names, cols)):
        _dots(ax, i, list(per[n].values()), c, rng,
              "o" if i < 2 else "s")
        r = res[n]
        if "p_perm" in r:
            ax.text(i, 1.0, f"p={r['p_perm']:.2f}\nn={r['n']}", fontsize=6.5,
                    ha="center", va="bottom",
                    transform=ax.get_xaxis_transform())
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.axvline(1.5, color="0.6", lw=0.6, ls=":")
    ax.set_xticks(range(len(names)), short, fontsize=7)
    ax.set_ylabel("partial Spearman ρ (Fisher z)\npause ripples vs outcome")
    ax.set_title("Do pause ripples predict later retrieval (storage) or the "
                 "next step (planning)?\nnegative = more ripples, better "
                 "behaviour", fontsize=9, pad=24)
    _clean_ax(ax)
    fig.text(0.01, -0.12,
             "Per session: Spearman of HC ripple rate in the pause after a "
             "correct uncover with the outcome, pause length, state and "
             "difficulty partialled out; dots: sessions,\nmarker: mean ± SEM; "
             "p: sign-flip. Left of the dotted line: the two pre-registered "
             "primaries.", fontsize=7, color="0.3")
    fig.tight_layout()
    for e in (".png", ".pdf"):
        fig.savefig(os.path.join(d, "T3_behaviour" + e), dpi=300,
                    bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ── T5: novelty without reward ───────────────────────────────────────────
T5_CELLS = {
    "error, novel": FT + " and valence == 'error' and novel == 1",
    "error, familiar": FT + " and valence == 'error' and novel == 0",
    "correct, novel": FT + " and valence == 'correct' and novel == 1",
    "correct, familiar": FT + " and valence == 'correct' and novel == 0",
}
T5_CONTRASTS = {
    "error: novel - familiar": {"error, novel": 1.0, "error, familiar": -1.0},
    "correct: novel - familiar": {"correct, novel": 1.0,
                                  "correct, familiar": -1.0},
    "novel: correct - error": {"correct, novel": 1.0, "error, novel": -1.0},
}
T5_C = {"error, novel": ERROR_C, "error, familiar": "#d4749f",
        "correct, novel": CORRECT_C, "correct, familiar": "#9ec6b8"}


def novelty(show=False, out_date=None):
    d = _out("T5_novelty", out_date)
    np.random.seed(SEED)
    data, presses = load()
    print("\n  T5: first traversal, novel vs familiar squares")
    centres, prof, meta, res = run_cells(data, presses, T5_CELLS, T5_CONTRASTS)
    for reading in ("matched", "own"):
        print(f"\n  ══ {reading} ══")
        for k, r in res[reading].items():
            print(f"   {k:26s} {_fmt(r)}")
    pd.DataFrame([dict(reading=rd, cell=k, **{x: r[x] for x in
                                              ("n_units", "mean_hz", "sem_hz",
                                               "ci_low_hz", "ci_high_hz", "t",
                                               "df", "p", "p_perm")})
                  for rd in res for k, r in res[rd].items() if r]) \
        .to_csv(os.path.join(d, "stats.csv"), index=False)
    _save_json(d, "result.json", dict(
        settings=dict(analysis="T5_novelty", created=str(datetime.date.today()),
                      script=os.path.relpath(__file__, REPO), seed=SEED,
                      pad_s=PAD_S, window_s=list(WIN), baseline_s=list(BASE),
                      novel="square not uncovered before in this block",
                      cells=T5_CELLS, contrasts=T5_CONTRASTS, unit="session",
                      preregistration="CHANGELOG.md 2026-10-29 T5"),
        cell_counts=meta, stats=res))
    _figure_t5(d, centres, prof, res, show)
    print(f"\n  saved -> {d}")


def _figure_t5(d, centres, prof, res, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 6.5 * CM),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    ax.axvspan(*WIN, color="0.92", lw=0)
    for k, c in T5_C.items():
        _trace(ax, prof[k]["matched"], centres, c, k,
               ls="-" if "novel" in k else "--")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlim(-1.5, 2.0)
    ax.set_xlabel("time from uncover (s)")
    ax.set_ylabel("uncover − matched\narrow press (Hz)")
    ax.set_title("first traversal: novel vs familiar square", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.42))
    _clean_ax(ax)
    ax = axes[1]
    for i, (k, c) in enumerate(T5_C.items()):
        _dots(ax, i, list(window_values(prof[k]["matched"], centres,
                                        WIN).values()), c, rng)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xticks(range(4), ["error\nnovel", "error\nfamiliar",
                             "correct\nnovel", "correct\nfamiliar"], fontsize=7)
    ax.set_ylabel("0–0.5 s (Hz)")
    r1 = res["matched"]["error: novel - familiar"]
    r2 = res["matched"]["correct: novel - familiar"]
    ax.set_title(f"error novel − familiar p = {r1['p_perm']:.2f}\n"
                 f"correct novel − familiar p = "
                 f"{r2['p_perm'] if r2 else float('nan'):.2f}", fontsize=8)
    _clean_ax(ax)
    fig.text(0.01, -0.26,
             "Stillness-matched reading (uncover minus an arrow press followed "
             "by the same stillness); NOVEL = square not uncovered before in "
             "this block. Mean ± SEM over sessions\n(lines smoothed for display "
             "only); dots: sessions; marker: mean ± SEM; p: sign-flip, 0–0.5 s.",
             fontsize=7, color="0.3")
    fig.tight_layout()
    for e in (".png", ".pdf"):
        fig.savefig(os.path.join(d, "T5_novelty" + e), dpi=300,
                    bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"knowledge": knowledge, "hfb_state": hfb_state,
                   "two_clocks": two_clocks, "behaviour": behaviour,
                   "novelty": novelty})
    else:
        knowledge()
