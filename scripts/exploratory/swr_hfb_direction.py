#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does frontal HFB lock BEFORE or AFTER the ripple -- and does mPFC differ from
mOFC in that?

The per-ROI trace grid (`swr_hfb_roi_partition.py traces`) looks as if mPFC HFB
peaks slightly AFTER the ripple and mOFC slightly BEFORE. This tests exactly
that, with one number per session:

    direction = mean HFB over (0, +250] ms  -  mean HFB over [-250, 0) ms

on the real-minus-shifted-null curve, UNSMOOTHED. Positive = locking stronger
after the ripple peak, negative = before. The two windows are the existing peri
window split at the peak (t = 0 itself is in neither). The flank baseline is a
constant and cancels in the difference, so the index does not depend on it.

THREE SETS OF RIPPLES
  a) all ripples in the task -- the region run, the same curves the trace grid
     draws.
  b) all ripples within a phase -- every condition (reward, error, move, still)
     of the conditions run, pooled by ripple count. Phases are PHASES_EVENT's
     first_traversal / encoding / after_encoding. `encoding` CONTAINS the
     first traversal, so those two are not independent.
  c) first traversal, by the event the ripple follows -- correct uncovers,
     incorrect uncovers, navigation presses; `still` (no press within +-2 s) as
     the press-free view; and correct - navigation, incorrect - navigation.
  d) c) in EVERY phase (first_traversal / encoding / after_encoding) and with
     lOFC as a third ROI: mPFC - mOFC, lOFC - mOFC, mPFC - lOFC. Exploratory,
     so p is reported uncorrected with BH q alongside. Cells resting on few
     sessions (error after encoding) are tested only from MIN_UNITS up.

THE CONFOUND THAT SHAPES THE DESIGN
A ripple's condition is the event that PRECEDES it by <= 2 s, so the press
response always sits at negative time on a ripple-locked axis
(`swr_hfb_event_latency.py`). That pushes every ROI towards "before". Hence:
  * the primary test is mPFC vs mOFC on the index -- a press response common to
    both cancels;
  * c) also reports each condition minus navigation, which removes the generic
    press response within ROI;
  * `still` ripples have no press within +-2 s.

STATISTICS
  * within ROI: sign-flip permutation of the per-session index against 0;
  * mPFC vs mOFC: ROI-label permutation of the per-session index (unpaired --
    only 11 sessions have both), with the paired test on those 11 reported as
    a check. Holm over the 10 ripple sets: this is the a-priori family.
  * every other ROI vs 0 is exploratory, BH over the whole table.
The permutation statistic is computed by the SAME function for the empirical
value and the null (`_signed_mean`, `_mean_diff`).

ROIs are the 2026-09-28 partition -- the one the trace grid was drawn from.
In the EXCLUSIVE partition mPFC and mOFC do not exist (they lie entirely inside
MedialFrontal), so it cannot answer this question.

Usage:
    conda activate env_multiple_clocks
    python scripts/exploratory/swr_hfb_direction.py run
    python scripts/exploratory/swr_hfb_direction.py run --show=False

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import datetime
import itertools

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))   # repo root
import mc.analyse.swr_io as swr_io
import mc.plotting.ripple_figures as rfig

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))    # scripts/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # exploratory/
from swr_hfb_locking_contrasts import (load, unit_curves, pooled_cond_curves,
                                       holm, COL)
from swr_hfb_roi_partition import ROI_C, FAMILIES, PHASES_EVENT, _bh
from swr_hfb_roi_conditions import load_partitioned, MEDIAL_GROUP
from swr_ripple_locked_hfb import _smooth, SMOOTH_MS

try:
    import fire
except ImportError:
    fire = None

ANALYSIS_NAME = "hfb_direction"
SEED, N_PERM = 42, 10000
WIN_MS = 250.0            # pre = [-250, 0), post = (0, +250]
MIN_UNITS = 5             # below this no test is run (UNDERPOWERED_BELOW)
PAIR = ("mPFC", "mOFC")   # the a-priori comparison
OTHER_ROIS = ("lOFC", "FrontalPole", "vlPFC_IFG", MEDIAL_GROUP)
ROI_COL = dict(ROI_C, **{MEDIAL_GROUP: COL[MEDIAL_GROUP]})

# d) the condition x phase grid
GRID_PHASES = (("first_traversal", "first traversal"), ("encoding", "encoding"),
               ("after_encoding", "after encoding"))
GRID_CONDS = (("reward", "correct uncovers"), ("error", "incorrect uncovers"),
              ("move", "navigation presses"), ("still", "no press ±2 s"),
              ("reward-move", "correct − navigation"),
              ("error-move", "incorrect − navigation"))
GRID_ROIS = ("mPFC", "mOFC", "lOFC")
GRID_PAIRS = (("mPFC", "mOFC"), ("lOFC", "mOFC"), ("mPFC", "lOFC"))

REGION_RUN = "ripple_locked_hfb_2026-09-18"
COND_RUN = "ripple_hfb_conditions_exclusive_2026-09-28"
PARTITION = "hfb_roi_partition_2026-09-28"


def _phase_conds(ph):
    """{reward, error, move, still: condition names} for one phase."""
    spec = dict(PHASES_EVENT[ph])
    spec["still"] = tuple(c.replace("reward", "still") for c in spec["reward"])
    return spec


def _phase_all(ph):
    """Every condition of a phase, `still` included: all its labelled ripples."""
    return sum(_phase_conds(ph).values(), ())


# (key, panel title, how the curves are built). `region` = the region run,
# a tuple = pooled conditions, ('minus', a, b) = paired difference of two sets.
SETS = [
    ("all", "a) all ripples", "region"),
    ("first_traversal", "b) first traversal", _phase_all("first_traversal")),
    ("encoding", "b) encoding", _phase_all("encoding")),
    ("after_encoding", "b) after encoding", _phase_all("after_encoding")),
    ("ft_reward", "c) correct uncovers", ("reward_explore",)),
    ("ft_error", "c) incorrect uncovers", ("error_explore",)),
    ("ft_move", "c) navigation presses", ("move_explore",)),
    ("ft_still", "c) no press ±2 s", ("still_explore",)),
    ("ft_reward-move", "c) correct − navigation", ("minus", "ft_reward", "ft_move")),
    ("ft_error-move", "c) incorrect − navigation", ("minus", "ft_error", "ft_move")),
]


# ── the index and its tests ──────────────────────────────────────────────
def direction(curves, t_ms):
    """{unit: post - pre} from {unit: curve}. Also returns pre and post."""
    pre = (t_ms >= -WIN_MS) & (t_ms < 0)
    post = (t_ms > 0) & (t_ms <= WIN_MS)
    return {u: (float(c[post].mean() - c[pre].mean()),
                float(c[pre].mean()), float(c[post].mean()))
            for u, c in curves.items()}


def _signed_mean(v, signs):
    return (v * signs).mean(-1)


def _mean_diff(v, is_a):
    return ((v * is_a).sum(-1) / is_a.sum(-1)
            - (v * ~is_a).sum(-1) / (~is_a).sum(-1))


def sign_flip(v):
    """Two-sided sign-flip permutation of mean(v) against 0."""
    rng = np.random.default_rng(SEED)
    obs = _signed_mean(v, np.ones_like(v))
    null = _signed_mean(v, rng.choice([-1.0, 1.0], size=(N_PERM, len(v))))
    return float(obs), float((1 + np.sum(np.abs(null) >= abs(obs))) / (N_PERM + 1))


def label_perm(a, b):
    """Two-sided ROI-label permutation of mean(a) - mean(b)."""
    rng = np.random.default_rng(SEED)
    v = np.concatenate([a, b])
    is_a = np.zeros(len(v), bool)
    is_a[:len(a)] = True
    obs = _mean_diff(v, is_a)
    perm = np.stack([rng.permutation(is_a) for _ in range(N_PERM)])
    null = _mean_diff(v, perm)
    return float(obs), float((1 + np.sum(np.abs(null) >= abs(obs))) / (N_PERM + 1))


def vs_zero(v):
    """Mean, SEM and the sign-flip test of one cell's per-unit index."""
    r = dict(n=len(v), mean=float(v.mean()),
             sem=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan)
    if len(v) >= MIN_UNITS:
        r["p_perm"] = sign_flip(v)[1]
        r["t"], r["p_t"] = map(float, stats.ttest_1samp(v, 0.0))
    return r


def pair_test(a, b):
    """ROI A vs ROI B on the index: unpaired label permutation (+ Welch), and
    the paired sign-flip on the units that have both, as a check."""
    if min(len(a), len(b)) < MIN_UNITS:
        return None
    diff, p = label_perm(a.to_numpy(), b.to_numpy())
    tw, pw = stats.ttest_ind(a, b, equal_var=False)
    sh = a.index.intersection(b.index)
    r = dict(n_a=len(a), n_b=len(b), diff=diff, p_perm=p, t_welch=float(tw),
             p_welch=float(pw), n_paired=len(sh))
    if len(sh) >= MIN_UNITS:
        d = (a[sh] - b[sh]).to_numpy()
        r["paired_diff"] = float(d.mean())
        r["p_paired"] = sign_flip(d)[1]
    return r


def _index_rows(curves, t_ms, keys):
    """Long table of the per-unit index; `keys` names the curve-dict key."""
    rows = []
    for k, cur in curves.items():
        for u, (d, pre, post) in direction(cur, t_ms).items():
            rows.append(dict(zip(keys, k), unit=u, direction=d, pre=pre,
                             post=post))
    return pd.DataFrame(rows)


# ── curves ───────────────────────────────────────────────────────────────
def region_curves(region_dir, partition_dir, rois, unit, merge=None):
    """Region-run curves per partition ROI, as `swr_hfb_roi_partition.traces`.
    `merge` maps partition ROIs onto a collapsed name, e.g. both OFC ROIs onto
    "OFC"; a unit's curve then averages over all its derivations in either."""
    t_ms, tr, ix = load(region_dir)
    A = pd.read_csv(os.path.join(partition_dir, "derivation_assignment.csv"))
    part = A.set_index(["session", "cx_pair"])["roi_part"]
    fam = ix[ix.roi.isin(FAMILIES)].copy()
    fam["roi"] = pd.MultiIndex.from_arrays([fam.session, fam.cx_pair]).map(part)
    if merge:
        fam["roi"] = fam.roi.replace(merge)
    ix = pd.concat([fam, ix[ix.roi == MEDIAL_GROUP]], ignore_index=True)
    ix = ix[~ix.same_shaft]
    return t_ms, {roi: unit_curves(t_ms, tr, ix, ix.roi == roi, unit=unit,
                                   smooth_ms=0.0) for roi in rois}


def grid_curves(t_ms, tr, ix, counts, unit):
    """d): {(phase, condition, roi): {unit: curve}}, same builder as c)."""
    out = {}
    for (ph, _), roi in itertools.product(GRID_PHASES, GRID_ROIS):
        conds = _phase_conds(ph)
        for k in ("reward", "error", "move", "still"):
            out[(ph, k, roi)] = pooled_cond_curves(
                t_ms, tr, ix, counts, conds[k], roi=roi, unit=unit,
                smooth_ms=0.0)
        for k in ("reward", "error"):
            a, b = out[(ph, k, roi)], out[(ph, "move", roi)]
            out[(ph, f"{k}-move", roi)] = {u: a[u] - b[u]
                                           for u in set(a) & set(b)}
    return out


def labelled_fraction(cond_dir, bundle_dir):
    """Share of all ripples that carry a condition label (the b) pool)."""
    c = pd.read_csv(os.path.join(cond_dir, "ripple_counts.csv"))
    lab = c[c.cond.isin(_phase_all("all"))].groupby("session").n_ripples.sum()
    tot = pd.read_csv(os.path.join(bundle_dir, "ripples.csv"),
                      usecols=["session"]).groupby("session").size()
    s = lab.index.intersection(tot.index)
    return {"labelled": int(lab[s].sum()), "all": int(tot[s].sum()),
            "fraction": float(lab[s].sum() / tot[s].sum()),
            "n_sessions": int(len(s))}


# ── run ──────────────────────────────────────────────────────────────────
def run(region_dir=None, cond_dir=None, partition_dir=None, out_dir=None,
        unit="session", show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")

    G = os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                     "group", "swr")
    region_dir = region_dir or os.path.join(G, REGION_RUN)
    cond_dir = cond_dir or os.path.join(G, COND_RUN)
    partition_dir = partition_dir or os.path.join(G, PARTITION)
    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(G, f"{ANALYSIS_NAME}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    rois = list(PAIR) + list(OTHER_ROIS)
    print(f"\n  regions    : {region_dir}\n  conditions : {cond_dir}\n"
          f"  partition  : {partition_dir}\n  unit = {unit}, pre [-{WIN_MS:.0f}, 0)"
          f" vs post (0, +{WIN_MS:.0f}] ms, unsmoothed, {N_PERM} perms, seed {SEED}")

    frac = labelled_fraction(cond_dir, os.path.join(G, "bundle_v2"))
    print(f"  b) pools the LABELLED ripples: {frac['labelled']} of {frac['all']}"
          f" ({100 * frac['fraction']:.0f}%) over {frac['n_sessions']} sessions")

    # ---- curves per set x ROI ------------------------------------------
    t_reg, reg = region_curves(region_dir, partition_dir, rois, unit)
    t_ms, tr, ix, counts = load_partitioned(cond_dir, partition_dir)
    assert np.allclose(t_reg, t_ms)
    curves = {}                                   # (set, roi) -> {unit: curve}
    for key, _, how in SETS:
        for roi in rois:
            if how == "region":
                curves[(key, roi)] = reg[roi]
            elif how[0] == "minus":
                a, b = curves[(how[1], roi)], curves[(how[2], roi)]
                curves[(key, roi)] = {u: a[u] - b[u] for u in set(a) & set(b)}
            else:
                curves[(key, roi)] = pooled_cond_curves(
                    t_ms, tr, ix, counts, how, roi=roi, unit=unit, smooth_ms=0.0)

    # ---- per-unit index -------------------------------------------------
    V = _index_rows(curves, t_ms, ("set", "roi"))

    # ---- within ROI vs 0 (BH over the table) ----------------------------
    T = pd.DataFrame([dict(set=key, roi=roi, **vs_zero(g.direction.to_numpy()),
                           pre=float(g.pre.mean()), post=float(g.post.mean()))
                      for (key, roi), g in V.groupby(["set", "roi"], sort=False)])
    ok = T.p_perm.notna()
    T.loc[ok, "q_bh"] = _bh(T.loc[ok, "p_perm"].to_numpy())

    # ---- mPFC vs mOFC (Holm over the sets) -------------------------------
    prows = []
    for key, _, _ in SETS:
        r = pair_test(*[V[(V.set == key) & (V.roi == roi)]
                        .set_index("unit").direction for roi in PAIR])
        if r:
            prows.append(dict(set=key, **r))
    P = pd.DataFrame(prows)
    P["p_holm"] = holm(P.p_perm.to_numpy())

    _report(T, P, unit)

    # ---- d) condition x phase, with lOFC ---------------------------------
    gcur = grid_curves(t_ms, tr, ix, counts, unit)
    GV = _index_rows(gcur, t_ms, ("phase", "cond", "roi"))
    GZ = pd.DataFrame([dict(phase=ph, cond=k, roi=roi,
                            **vs_zero(g.direction.to_numpy()),
                            pre=float(g.pre.mean()), post=float(g.post.mean()))
                       for (ph, k, roi), g in GV.groupby(["phase", "cond", "roi"],
                                                         sort=False)])
    ok = GZ.p_perm.notna()
    GZ.loc[ok, "q_bh"] = _bh(GZ.loc[ok, "p_perm"].to_numpy())
    gprows = []
    for (ph, _), (k, _), (ra, rb) in itertools.product(GRID_PHASES, GRID_CONDS,
                                                       GRID_PAIRS):
        sel = (GV.phase == ph) & (GV.cond == k)
        r = pair_test(GV[sel & (GV.roi == ra)].set_index("unit").direction,
                      GV[sel & (GV.roi == rb)].set_index("unit").direction)
        if r:
            gprows.append(dict(phase=ph, cond=k, pair=f"{ra} − {rb}", **r))
    GP = pd.DataFrame(gprows)
    GP["q_bh"] = _bh(GP.p_perm.to_numpy())
    _report_grid(GZ, GP)

    V.to_csv(os.path.join(out_dir, "direction_per_unit.csv"), index=False)
    T.to_csv(os.path.join(out_dir, "direction_vs_zero.csv"), index=False)
    P.to_csv(os.path.join(out_dir, "direction_mpfc_vs_mofc.csv"), index=False)
    GV.to_csv(os.path.join(out_dir, "grid_per_unit.csv"), index=False)
    GZ.to_csv(os.path.join(out_dir, "grid_vs_zero.csv"), index=False)
    GP.to_csv(os.path.join(out_dir, "grid_roi_pairs.csv"), index=False)
    settings = dict(analysis=ANALYSIS_NAME, created=stamp,
                    script=os.path.relpath(__file__),
                    region_dir=region_dir, cond_dir=cond_dir,
                    partition_dir=partition_dir, unit=unit,
                    index="mean(0, +250] - mean[-250, 0) ms of the unsmoothed "
                          "real-minus-null curve",
                    win_ms=WIN_MS, n_perm=N_PERM, seed=SEED,
                    min_units=MIN_UNITS, shafts="different-shaft only",
                    sets={k: (h if isinstance(h, str) else list(h))
                          for k, _, h in SETS},
                    labelled_fraction=frac,
                    multiple_comparisons={
                        "mPFC_vs_mOFC": "Holm over the sets (a-priori family)",
                        "vs_zero": "BH over every ROI x set (exploratory)",
                        "grid": "uncorrected p reported; BH q over the grid "
                                "pairs table and over the grid vs-0 table"},
                    grid=dict(phases=[p for p, _ in GRID_PHASES],
                              conds=[c for c, _ in GRID_CONDS],
                              rois=list(GRID_ROIS),
                              pairs=[list(p) for p in GRID_PAIRS]),
                    curve_smoothing_ms_figure_only=SMOOTH_MS)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"settings": settings,
                   "mpfc_vs_mofc": P.to_dict("records"),
                   "vs_zero": T.to_dict("records"),
                   "grid_roi_pairs": GP.to_dict("records"),
                   "grid_vs_zero": GZ.to_dict("records")},
                  f, indent=2, default=str)

    _figure_index(V, T, P, rois, out_dir, unit)
    _figure_curves(curves, t_ms, T, out_dir, unit)
    _figure_grid(GV, GZ, GP, out_dir, unit)
    _figure_grid_curves(gcur, t_ms, out_dir, unit)
    print(f"\n  saved -> {out_dir}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()


def _report(T, P, unit):
    print(f"\n{'=' * 92}\nmPFC − mOFC on the direction index (post − pre), "
          f"per {unit}; + = mPFC later\n{'=' * 92}")
    print(f"  {'set':<18s}{'nA':>4s}{'nB':>4s}{'diff':>10s}{'p_perm':>9s}"
          f"{'p_holm':>9s}{'p_welch':>9s}{'n_pair':>8s}{'p_pair':>8s}")
    for _, r in P.iterrows():
        pp = r.get("p_paired", np.nan)
        print(f"  {r.set:<18s}{r.n_a:>4d}{r.n_b:>4d}{r['diff']:>+10.4f}"
              f"{r.p_perm:>9.4f}{r.p_holm:>9.4f}{r.p_welch:>9.4f}"
              f"{r.n_paired:>8d}{'' if pd.isna(pp) else f'{pp:.4f}':>8s}")
    print(f"\n{'=' * 92}\nEach ROI vs 0 (sign-flip); + = after, − = before\n"
          f"{'=' * 92}")
    for key in T.set.unique():
        print(f"  --- {key}")
        for _, r in T[T.set == key].iterrows():
            q = r.get("q_bh", np.nan)
            print(f"    {r.roi:<14s}n={r.n:>3d}  {r['mean']:+.4f} ± {r['sem']:.4f}"
                  f"  (pre {r.pre:+.4f}, post {r.post:+.4f})"
                  + ("" if pd.isna(q) else f"  p={r.p_perm:.4f} q={q:.3f}"))


def _report_grid(GZ, GP):
    print(f"\n{'=' * 92}\nd) condition x phase -- ROI vs ROI on the index "
          f"(uncorrected p, BH q over the table)\n{'=' * 92}")
    for (ph, _), (k, lab) in itertools.product(GRID_PHASES, GRID_CONDS):
        z = GZ[(GZ.phase == ph) & (GZ.cond == k)].set_index("roi")
        p = GP[(GP.phase == ph) & (GP.cond == k)]
        roi_txt = "  ".join(f"{roi} {z.loc[roi, 'mean']:+.4f} (n={z.loc[roi, 'n']})"
                            for roi in GRID_ROIS if roi in z.index)
        print(f"  {ph:<16s}{lab:<24s}{roi_txt}")
        for _, r in p.iterrows():
            print(f"  {'':40s}{r.pair:<14s}{r['diff']:+.4f}  p={r.p_perm:.3f}"
                  f"  q={r.q_bh:.2f}")


# ── figures (overview, not publication) ──────────────────────────────────
def _p_txt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _figure_index(V, T, P, rois, out_dir, unit):
    """Per-session direction index per ROI, one panel per ripple set."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)        # dot jitter only
    with plt.rc_context(rfig._rc()):
        fig, axes = plt.subplots(3, 4, figsize=(4 * 5.2 * rfig.CM,
                                                3 * 7.3 * rfig.CM))
        axes = axes.ravel()
        for ax, (key, title, _) in zip(axes, SETS):
            for i, roi in enumerate(rois):
                v = V[(V.set == key) & (V.roi == roi)].direction.to_numpy()
                if not len(v):
                    continue
                col = ROI_COL.get(roi, "#888888")
                mk = "D" if roi == MEDIAL_GROUP else "o"
                ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=7,
                           color=col, alpha=0.55, lw=0, marker=mk, zorder=2)
                r = T[(T.set == key) & (T.roi == roi)].iloc[0]
                ax.errorbar(i, r["mean"], yerr=r["sem"], fmt=mk, ms=5.5,
                            color="#1a1a1a", mfc=col, mec="#1a1a1a", lw=1.2,
                            capsize=2.5, zorder=4)
                if r.get("q_bh", np.nan) < 0.05:
                    ax.text(i, 1.0, "*", transform=ax.get_xaxis_transform(),
                            ha="center", va="top", fontsize=rfig.FS_TITLE,
                            color="#0e3d3a")
            ax.axhline(0, color="#3d3d3d", lw=0.7, ls="--", zorder=1)
            ax.axvline(1.5, color="#bbbbbb", lw=0.7, ls=":")
            ax.set_xticks(range(len(rois)))
            ax.set_xticklabels([r.replace("_", " ") for r in rois],
                               rotation=50, ha="right", fontsize=rfig.FS_TICK)
            ax.set_xlim(-0.6, len(rois) - 0.4)
            # 2nd-98th percentile limits, as in the trace grid: a few
            # sessions run off-panel, every one is in the mean.
            v = V[V.set == key].direction
            lo, hi = np.percentile(v, [2, 98])
            ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.22 * (hi - lo))
            # mPFC vs mOFC goes in the title, so it can never sit on the dots
            # or the tick labels; the bracket only marks which pair it is.
            pr = P[P.set == key]
            if len(pr):
                pr = pr.iloc[0]
                y = hi + 0.12 * (hi - lo)
                ax.plot([0, 0, 1, 1], [y - 0.04 * (hi - lo), y, y,
                                       y - 0.04 * (hi - lo)],
                        color="#1a1a1a", lw=0.8)
                title = (f"{title}\nmPFC vs mOFC: {_p_txt(pr.p_perm)}"
                         f"\nHolm-corrected p = {pr.p_holm:.2f}")
            ax.set_title(title, fontsize=rfig.FS_TICK)
        for ax in axes[len(SETS):]:
            ax.set_visible(False)
        for ax in axes[::4]:
            ax.set_ylabel("direction (z)\npost − pre, ±250 ms",
                          fontsize=rfig.FS_LABEL)
        fig.suptitle("Does ripple-locked HFB rise AFTER (> 0) or BEFORE (< 0) "
                     "the ripple peak?", fontsize=rfig.FS_TITLE)
        fig.text(0.62, 0.17,
                 f"Dots: single {unit}s. Marker: mean ± SEM.\n"
                 "Bracket: mPFC vs mOFC, ROI-label permutation;\n"
                 "Holm over the 10 sets.\n"
                 "* sign-flip vs 0, BH q < 0.05 over all cells\n"
                 "(exploratory).\n"
                 "Right of the dotted line: exploratory ROIs;\n"
                 "MedialFrontal (diamond) overlaps mPFC + mOFC.\n"
                 "c) uses the first traversal only. The event always\n"
                 "precedes its ripple, so a press response biases\n"
                 "every ROI towards < 0; it cancels in mPFC − mOFC\n"
                 "and in the − navigation contrasts.",
                 ha="left", va="center", fontsize=rfig.FS_TICK - 1,
                 color="#3d3d3d")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        stem = os.path.join(out_dir, "direction_index")
        fig.savefig(stem + ".png", dpi=300)
        fig.savefig(stem + ".pdf")


def _curves_panel(ax, by_roi, t_ms):
    """One ripple set: mean ± SEM per ROI, pre/post windows shaded."""
    fs = 1000.0 / (t_ms[1] - t_ms[0])
    ax.axvspan(-WIN_MS, 0, color="#000000", alpha=0.06, lw=0)
    ax.axvspan(0, WIN_MS, color="#000000", alpha=0.12, lw=0)
    ax.axhline(0, color="#3d3d3d", lw=0.6, ls="--")
    ax.axvline(0, color="#3d3d3d", lw=0.6, ls=":")
    for roi, cur in by_roi.items():
        if len(cur) < 2:
            continue
        c = _smooth(np.stack([cur[u] for u in sorted(cur)]), fs, SMOOTH_MS)
        m, se = c.mean(0), c.std(0, ddof=1) / np.sqrt(len(c))
        ax.fill_between(t_ms, m - se, m + se, color=ROI_COL[roi], alpha=0.25,
                        lw=0)
        ax.plot(t_ms, m, color=ROI_COL[roi], lw=1.5, label=roi)
    for x, lab in ((-WIN_MS / 2, "pre"), (WIN_MS / 2, "post")):
        ax.text(x, 0.02, lab, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=rfig.FS_TICK - 2, color="#3d3d3d")
    ax.set_xlim(-750, 750)
    ax.set_xticks([-500, 0, 500])
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.12 * (hi - lo), hi)


def _figure_curves(curves, t_ms, T, out_dir, unit):
    """The components: mPFC and mOFC curves with the two windows marked."""
    import matplotlib.pyplot as plt

    with plt.rc_context(rfig._rc()):
        fig, axes = plt.subplots(3, 4, sharex=True,
                                 figsize=(4 * 5.0 * rfig.CM, 3 * 5.6 * rfig.CM))
        axes = axes.ravel()
        for ax, (key, title, _) in zip(axes, SETS):
            _curves_panel(ax, {roi: curves[(key, roi)] for roi in PAIR}, t_ms)
            n = {roi: len(curves[(key, roi)]) for roi in PAIR}
            ax.set_title(f"{title}\n(n = {n['mPFC']} / {n['mOFC']} {unit}s)",
                         fontsize=rfig.FS_TICK)
        for ax in axes[len(SETS):]:
            ax.axis("off")
        # one legend for the figure, in the empty slot -- per-panel legends
        # sat on the traces
        axes[-1].legend(*axes[0].get_legend_handles_labels(), frameon=False,
                        loc="center", fontsize=rfig.FS_LABEL,
                        title="ROI (mean ± SEM)", title_fontsize=rfig.FS_TICK)
        for ax in axes[::4]:
            ax.set_ylabel("HFB, real − null (z)", fontsize=rfig.FS_LABEL)
        for ax in axes[len(SETS) - 4:len(SETS)]:
            ax.set_xlabel("time from ripple (ms)", fontsize=rfig.FS_LABEL)
            ax.xaxis.set_tick_params(labelbottom=True)
        fig.suptitle(f"The components: mPFC and mOFC ripple-locked HFB, mean ± "
                     f"SEM over {unit}s (n = mPFC / mOFC)\nshaded: pre [−250, 0) and post (0, +250] "
                     f"ms; {SMOOTH_MS:.0f} ms smoothing for drawing only, the "
                     "index uses the raw curve", fontsize=rfig.FS_TICK)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        stem = os.path.join(out_dir, "direction_curves")
        fig.savefig(stem + ".png", dpi=300)
        fig.savefig(stem + ".pdf")


def _figure_grid(GV, GZ, GP, out_dir, unit):
    """d): per-session index for mPFC / mOFC / lOFC, one panel per condition,
    phases along x; the ROI-vs-ROI tests in a table under each panel."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(SEED)        # dot jitter only
    off = {roi: (j - 1) * 0.26 for j, roi in enumerate(GRID_ROIS)}
    with plt.rc_context(rfig._rc()):
        fig = plt.figure(figsize=(3 * 7.2 * rfig.CM, 2 * 9.2 * rfig.CM))
        gs = fig.add_gridspec(4, 3, height_ratios=[3.3, 0.8, 3.3, 0.8],
                              hspace=0.5, wspace=0.3)
        for ci, (k, lab) in enumerate(GRID_CONDS):
            ax = fig.add_subplot(gs[2 * (ci // 3), ci % 3])
            tax = fig.add_subplot(gs[2 * (ci // 3) + 1, ci % 3])
            for i, (ph, _) in enumerate(GRID_PHASES):
                for roi in GRID_ROIS:
                    sel = (GV.phase == ph) & (GV.cond == k) & (GV.roi == roi)
                    v = GV[sel].direction.to_numpy()
                    if not len(v):
                        continue
                    x = i + off[roi]
                    ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=5,
                               color=ROI_COL[roi], alpha=0.5, lw=0, zorder=2)
                    z = GZ[(GZ.phase == ph) & (GZ.cond == k)
                           & (GZ.roi == roi)].iloc[0]
                    ax.errorbar(x, z["mean"], yerr=z["sem"], fmt="o", ms=4.5,
                                mfc=ROI_COL[roi], mec="#1a1a1a",
                                color="#1a1a1a", lw=1.0, capsize=2, zorder=4)
                    if z.get("p_perm", np.nan) < 0.05:
                        ax.text(x, 1.0, "*", transform=ax.get_xaxis_transform(),
                                ha="center", va="top", fontsize=rfig.FS_LABEL,
                                color="#0e3d3a")
            ax.axhline(0, color="#3d3d3d", lw=0.7, ls="--", zorder=1)
            for b in (0.5, 1.5):
                ax.axvline(b, color="#dddddd", lw=0.7)
            ax.set_xticks(range(len(GRID_PHASES)))
            ax.set_xticklabels([p.replace(" ", "\n") for _, p in GRID_PHASES],
                               fontsize=rfig.FS_TICK)
            ax.set_xlim(-0.5, len(GRID_PHASES) - 0.5)
            v = GV[GV.cond == k].direction
            lo, hi = np.percentile(v, [2, 98])
            ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.15 * (hi - lo))
            ax.set_title(lab, fontsize=rfig.FS_LABEL)
            if ci % 3 == 0:
                ax.set_ylabel("direction (z)\npost − pre, ±250 ms",
                              fontsize=rfig.FS_LABEL)

            # the ROI-vs-ROI tests, written under their own phase column
            # (same x-axis as the dots) -- difference and uncorrected p
            tax.set_xlim(ax.get_xlim())
            tax.set_ylim(len(GRID_PAIRS) - 0.4, -0.6)
            tax.axis("off")
            for ri, (ra, rb) in enumerate(GRID_PAIRS):
                if ci % 3 == 0:
                    tax.text(-0.62, ri, f"{ra} − {rb}", ha="right",
                             va="center", fontsize=rfig.FS_TICK - 2)
                for i, (ph, _) in enumerate(GRID_PHASES):
                    r = GP[(GP.phase == ph) & (GP.cond == k)
                           & (GP.pair == f"{ra} − {rb}")]
                    if len(r):
                        r = r.iloc[0]
                        txt = f"{r['diff']:+.3f}, p = {r.p_perm:.2f}"
                        fw = "bold" if r.p_perm < 0.05 else "normal"
                    else:
                        txt, fw = "n < 5", "normal"
                    tax.text(i, ri, txt, ha="center", va="center",
                             fontsize=rfig.FS_TICK - 3, fontweight=fw,
                             color="#1a1a1a")
        handles = [Line2D([], [], marker="o", ls="", mfc=ROI_COL[r],
                          mec="#1a1a1a", ms=6, label=r) for r in GRID_ROIS]
        fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False,
                   fontsize=rfig.FS_TICK, bbox_to_anchor=(0.98, 0.985))
        fig.suptitle("Direction of ripple-locked HFB by condition and phase "
                     "(> 0 = after the ripple, < 0 = before)",
                     fontsize=rfig.FS_TITLE, x=0.42, y=0.995)
        fig.text(0.5, 0.005,
                 f"Dots: single {unit}s; marker: mean ± SEM. * sign-flip vs 0, "
                 "p < 0.05 UNCORRECTED. Under each panel: ROI A − ROI B on the index, "
                 "ROI-label\npermutation, uncorrected p (bold < 0.05); BH q for "
                 "every cell is in grid_roi_pairs.csv. Encoding contains the "
                 "first traversal. The event always precedes its ripple.",
                 ha="center", va="bottom", fontsize=rfig.FS_TICK - 2,
                 color="#3d3d3d")
        fig.subplots_adjust(left=0.15, right=0.98, top=0.92, bottom=0.07)
        stem = os.path.join(out_dir, "direction_phase_condition")
        fig.savefig(stem + ".png", dpi=300)
        fig.savefig(stem + ".pdf")


def _figure_grid_curves(gcur, t_ms, out_dir, unit):
    """d) components: the curves behind every grid cell, three ROIs each."""
    import matplotlib.pyplot as plt

    with plt.rc_context(rfig._rc()):
        nr, nc = len(GRID_CONDS), len(GRID_PHASES)
        fig, axes = plt.subplots(nr, nc, sharex=True,
                                 figsize=(nc * 5.4 * rfig.CM,
                                          nr * 4.6 * rfig.CM))
        for (ri, (k, lab)), (cj, (ph, plab)) in itertools.product(
                enumerate(GRID_CONDS), enumerate(GRID_PHASES)):
            ax = axes[ri, cj]
            by_roi = {roi: gcur[(ph, k, roi)] for roi in GRID_ROIS}
            _curves_panel(ax, by_roi, t_ms)
            n = " / ".join(str(len(by_roi[r])) for r in GRID_ROIS)
            ax.set_title(f"{lab}, {plab}\n(n = {n})", fontsize=rfig.FS_TICK - 1)
            if cj == 0:
                ax.set_ylabel("HFB, real − null (z)", fontsize=rfig.FS_TICK)
        for ax in axes[-1]:
            ax.set_xlabel("time from ripple (ms)", fontsize=rfig.FS_TICK)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right",
                   ncol=3, frameon=False, fontsize=rfig.FS_TICK)
        fig.suptitle(f"Components of the condition x phase grid: mean ± SEM "
                     f"over {unit}s (n = {' / '.join(GRID_ROIS)})\nshaded: pre "
                     f"and post windows; {SMOOTH_MS:.0f} ms smoothing for drawing "
                     "only", fontsize=rfig.FS_TICK, x=0.4)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        stem = os.path.join(out_dir, "direction_phase_condition_curves")
        fig.savefig(stem + ".png", dpi=200)
        fig.savefig(stem + ".pdf")


# ── panel a) again, with a sliding-window test per ROI ───────────────────
# Each ROI against its OWN baseline remote from the ripple: the curve is real
# minus shifted null, re-baselined to the epoch edges (|t| >= 750 ms -- as far
# from the ripple as the saved +-1 s epochs reach). The flank baseline of the
# other panels (250-750 ms) would subtract the very window being tested.
# The test is a 50 ms sliding mean, cluster-permuted over |t| < 750 ms with
# `swr_sakon.cluster_perm_time` (sign-flip over sessions, cluster mass,
# two-sided) -- the same function as the locking contrasts.
SLIDE_MS = 50.0
REMOTE_MS = 750.0
CLUSTER_PERM = 10000


OFC_MERGE = {"mOFC": "OFC", "lOFC": "OFC"}
ROI_COL.setdefault("OFC", ROI_COL["mOFC"])
ROI_SHOW = {"OFC": "OFC (mOFC+lOFC)"}


# What the time axis is locked to decides where a baseline can sit.
#   ripple  epoch +-1 s; baseline = both edges, |t| >= 750 ms
#   event   epoch -1.5..+1.5 s; baseline = -1.5..-1.0 s BEFORE the event (the
#           project's agreed event baseline, `swr_phases.BASE_B`) -- after the
#           event the response lives, so no post-event window can be baseline
# Everything that is not baseline is tested.
LOCKS = {
    "ripple": dict(base=lambda t: np.abs(t) >= REMOTE_MS, label="ripple",
                   base_txt=f"its own mean ≥ {REMOTE_MS:.0f} ms from the ripple",
                   xticks=(-1000, -500, 0, 500, 1000)),
    "event": dict(base=lambda t: (t >= -1500) & (t < -1000),
                  label="correct uncover",
                  base_txt="its own mean 1.5–1.0 s before the uncover",
                  xticks=(-1500, -1000, -500, 0, 500, 1000, 1500)),
}


def _lock(t_ms, lock):
    base = LOCKS[lock]["base"](t_ms)
    return base, ~base


def sliding_curves(reg, roi, t_ms, base=None):
    """(units, curves): real − null per unit, minus its own baseline mean,
    then a SLIDE_MS sliding mean. What panel a) draws."""
    from scipy.ndimage import uniform_filter1d
    base = _lock(t_ms, "ripple")[0] if base is None else base
    units = sorted(reg[roi])
    X = np.stack([reg[roi][u] for u in units])
    X = X - X[:, base].mean(1, keepdims=True)
    k = int(round(SLIDE_MS / (t_ms[1] - t_ms[0])))
    return units, uniform_filter1d(X, k, axis=1, mode="nearest")


# ── the analyses (no I/O) ────────────────────────────────────────────────
def analyse_pair(reg, t_ms, pair, lock="ripple", unit="session"):
    """Per ROI: sliding curve, cluster test against its own baseline
    (`swr_sakon.cluster_perm_time`), peak, direction index; then the timing
    test, mPFC − left on the direction index (`pair_test`)."""
    import mc.analyse.swr_sakon as sk
    base, test = _lock(t_ms, lock)
    tt = t_ms[test]
    left = pair[0]
    res = {}
    for roi in pair:
        _, Xs = sliding_curves(reg, roi, t_ms, base)
        t_obs, cl, pv, info = sk.cluster_perm_time(
            Xs[:, test], n_perm=CLUSTER_PERM, seed=SEED)
        m = Xs.mean(0)
        i_pk = np.flatnonzero(test)[np.argmax(m[test])]
        # the timing test the clusters cannot give: the direction index on
        # the unsmoothed curve, exactly as in `run` (baseline cancels)
        dirs = pd.Series({u: d[0] for u, d in direction(reg[roi], t_ms).items()})
        res[roi] = dict(
            n_units=len(Xs), mean=m, sem=Xs.std(0, ddof=1) / np.sqrt(len(Xs)),
            direction=dirs, peak=float(m[i_pk]), peak_ms=float(t_ms[i_pk]),
            threshold_t=info["threshold_t"],
            clusters=[dict(start_ms=float(tt[a]), stop_ms=float(tt[b - 1]),
                           sign="increase" if t_obs[a:b].mean() > 0
                           else "decrease",
                           mass=float(np.abs(t_obs[a:b]).sum()), p=float(p))
                      for (a, b), p in zip(cl, pv)])
        print(f"    {roi}: n = {len(Xs)} {unit}s, peak {m[i_pk]:+.4f} z at "
              f"{t_ms[i_pk]:+.0f} ms; " + ("; ".join(
                  f"{c['sign']} {c['start_ms']:+.0f}…{c['stop_ms']:+.0f} ms "
                  f"p={c['p']:.4f}" for c in res[roi]["clusters"]
                  if c["p"] < 0.05) or "no significant cluster"))
    timing = dict(test=f"direction index (post − pre, ±{WIN_MS:.0f} ms): "
                       f"mPFC − {left}",
                  **{f"{r}_mean": float(res[r]["direction"].mean())
                     for r in pair},
                  **(pair_test(res["mPFC"]["direction"],
                               res[left]["direction"]) or {}))
    if "p_perm" in timing:
        print(f"    timing, mPFC − {left}: {timing['diff']:+.4f}, "
              f"p = {timing['p_perm']:.3f} (paired on {timing['n_paired']}: "
              f"p = {timing.get('p_paired', np.nan):.3f})")
    return res, timing


def _contrast_stat(A, B, test, scale):
    """Welch t (B − A) per bin over `test`, after the chosen scaling."""
    if scale == "peak":
        A = A / A.mean(0)[test].max()
        B = B / B.mean(0)[test].max()
    return stats.ttest_ind(B[:, test], A[:, test], equal_var=False).statistic


def _clusters(t, thr):
    out, i = [], 0
    sig = np.abs(t) > thr
    while i < len(sig):
        if sig[i]:
            j = i
            while j + 1 < len(sig) and sig[j + 1]:
                j += 1
            out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def contrast_perm(A, B, test, scale, n_perm=CLUSTER_PERM, seed=SEED):
    """Cluster-mass test of B − A with ROI labels permuted across units."""
    rng = np.random.default_rng(seed)
    thr = stats.t.ppf(0.975, len(A) + len(B) - 2)
    t_obs = _contrast_stat(A, B, test, scale)
    obs = _clusters(t_obs, thr)
    mass = [np.abs(t_obs[a:b]).sum() for a, b in obs]
    pool, nA = np.r_[A, B], len(A)
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(len(pool))
        tk = _contrast_stat(pool[p[:nA]], pool[p[nA:]], test, scale)
        null[k] = max([np.abs(tk[a:b]).sum() for a, b in _clusters(tk, thr)],
                      default=0.0)
    pv = [float((1 + np.sum(null >= m)) / (1 + n_perm)) for m in mass]
    return t_obs, obs, pv


def analyse_contrast(reg, t_ms, pair, lock="ripple", unit="session"):
    """mPFC − left over time, z and peak-scaled: Welch t, cluster mass, ROI
    labels permuted over units, scaling recomputed in every permutation;
    paired check on the shared units (sign-flip, `cluster_perm_time`)."""
    import mc.analyse.swr_sakon as sk
    base, test = _lock(t_ms, lock)
    tt = t_ms[test]
    left = pair[0]
    uA, A = sliding_curves(reg, left, t_ms, base)
    uB, B = sliding_curves(reg, "mPFC", t_ms, base)
    shared = sorted(set(uA) & set(uB))
    res = {}
    for scale in ("z", "peak"):
        t_obs, cl, pv = contrast_perm(A, B, test, scale)
        sa = A.mean(0)[test].max() if scale == "peak" else 1.0
        sb = B.mean(0)[test].max() if scale == "peak" else 1.0
        fmt = lambda t_, c_, p_: [dict(
            start_ms=float(tt[a]), stop_ms=float(tt[b - 1]),
            bigger="mPFC" if t_[a:b].mean() > 0 else left,
            mass=float(np.abs(t_[a:b]).sum()), p=float(p))
            for (a, b), p in zip(c_, p_)]
        r = dict(scale_A=float(sa), scale_B=float(sb),
                 clusters=fmt(t_obs, cl, pv), n_paired=len(shared),
                 paired_clusters=[])
        if len(shared) >= MIN_UNITS:
            D = (np.stack([B[uB.index(u)] for u in shared]) / sb
                 - np.stack([A[uA.index(u)] for u in shared]) / sa)
            tp, clp, pvp, _ = sk.cluster_perm_time(
                D[:, test], n_perm=CLUSTER_PERM, seed=SEED)
            r["paired_clusters"] = fmt(tp, clp, pvp)
        res[scale] = r
        sig = [c for c in r["clusters"] if c["p"] < 0.05]
        sigp = [c for c in r["paired_clusters"] if c["p"] < 0.05]
        print(f"    contrast ({scale}): " + ("; ".join(
            f"{c['bigger']} bigger {c['start_ms']:+.0f}…{c['stop_ms']:+.0f} "
            f"p={c['p']:.4f}" for c in sig) or "none") + " | paired: " + (
            "; ".join(f"{c['bigger']} bigger {c['start_ms']:+.0f}…"
                      f"{c['stop_ms']:+.0f} p={c['p']:.4f}" for c in sigp)
            or "none"))
    return dict(A=A, B=B, uA=uA, uB=uB, res=res)


# ── drawing, onto a given axis ───────────────────────────────────────────
def draw_sliding(ax, res, t_ms, pair, lock="ripple", title=None, legend=True,
                 fs_scale=0):
    """Left ROI on the left axis, mPFC on the right, each scaled to its own
    peak (peaks and zero lines coincide); significant clusters as bars."""
    base, _ = _lock(t_ms, lock)
    left = pair[0]
    lo = min(((d["mean"] - d["sem"]).min() / d["peak"]) for d in res.values())
    hi = max(((d["mean"] + d["sem"]).max() / d["peak"]) for d in res.values())
    lo, hi = lo - 0.3 * (hi - lo), hi + 0.35 * (hi - lo)
    axes = {left: ax, "mPFC": ax.twinx()}
    for a_, b_ in _runs(base):
        ax.axvspan(t_ms[a_], t_ms[b_ - 1] + (t_ms[1] - t_ms[0]),
                   color="#000000", alpha=0.06, lw=0)
    ax.axhline(0, color="#3d3d3d", lw=0.6, ls="--")
    ax.axvline(0, color="#1a1a1a", lw=0.7, ls=":")
    handles, row = [], 0          # one row per significant cluster
    for j, roi in enumerate((left, "mPFC")):
        d, a = res[roi], axes[roi]
        col = ROI_COL[roi]
        a.fill_between(t_ms, d["mean"] - d["sem"], d["mean"] + d["sem"],
                       color=col, alpha=0.22, lw=0)
        h, = a.plot(t_ms, d["mean"], color=col, lw=1.5,
                    label=f"{ROI_SHOW.get(roi, roi)} (n = {d['n_units']})")
        handles.append(h)
        a.set_ylim(lo * d["peak"], hi * d["peak"])
        a.tick_params(axis="y", colors=col, labelsize=rfig.FS_TICK - fs_scale)
        a.set_ylabel(f"{roi} HFB (z)", color=col,
                     fontsize=rfig.FS_LABEL - fs_scale)
        for c in d["clusters"]:
            if c["p"] < 0.05:
                y = 0.05 + 0.065 * row
                row += 1
                ax.plot([c["start_ms"], c["stop_ms"]], [y, y],
                        transform=ax.get_xaxis_transform(), color=col,
                        lw=3.0, solid_capstyle="butt",
                        alpha=1.0 if c["sign"] == "increase" else 0.4)
                ax.text(c["stop_ms"] + 25, y, _p_txt(c["p"]),
                        transform=ax.get_xaxis_transform(), ha="left",
                        va="center", fontsize=rfig.FS_TICK - 2 - fs_scale,
                        color=col)
    axes["mPFC"].spines["right"].set_visible(True)
    axes["mPFC"].spines["right"].set_color(ROI_COL["mPFC"])
    ax.spines["left"].set_color(ROI_COL[left])
    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.set_xticks(LOCKS[lock]["xticks"])
    ax.tick_params(axis="x", labelsize=rfig.FS_TICK - fs_scale)
    ax.set_xlabel(f"time from {LOCKS[lock]['label']} (ms)",
                  fontsize=rfig.FS_LABEL - fs_scale)
    if title:
        ax.set_title(title, fontsize=rfig.FS_TITLE - fs_scale)
    if legend:
        ax.legend(handles=handles, frameon=False,
                  fontsize=rfig.FS_TICK - 1.5 - fs_scale, loc="upper left",
                  handlelength=1.0, borderaxespad=0.3, labelspacing=0.2)


def _runs(mask):
    """(start, stop) index pairs of the True runs of a boolean array."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def draw_timing(ax, res, timing, pair, lock="ripple", title=None, fs_scale=0):
    """Direction index per unit, both ROIs, lines joining shared units."""
    rng = np.random.default_rng(SEED)        # dot jitter only
    left = pair[0]
    d = {r: pd.Series(res[r]["direction"]) for r in (left, "mPFC")}
    x = {r: pd.Series(i + rng.uniform(-0.12, 0.12, len(d[r])),
                      index=d[r].index)
         for i, r in enumerate((left, "mPFC"))}
    for u in d[left].index.intersection(d["mPFC"].index):
        ax.plot([x[left][u], x["mPFC"][u]], [d[left][u], d["mPFC"][u]],
                color="#bbbbbb", lw=0.6, zorder=1)
    for i, r in enumerate((left, "mPFC")):
        v = d[r].to_numpy()
        ax.scatter(x[r], v, s=10, color=ROI_COL[r], alpha=0.7, lw=0, zorder=2)
        ax.errorbar(i, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)),
                    fmt="o", ms=5.5, mfc=ROI_COL[r], mec="#1a1a1a",
                    color="#1a1a1a", lw=1.2, capsize=3, zorder=4)
    ax.axhline(0, color="#3d3d3d", lw=0.6, ls="--", zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{left}\nn = {len(d[left])}",
                        f"mPFC\nn = {len(d['mPFC'])}"],
                       fontsize=rfig.FS_TICK - 1 - fs_scale)
    ax.set_xlim(-0.55, 1.55)
    lo, hi = np.percentile(np.r_[d[left], d["mPFC"]], [1, 99])
    ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.45 * (hi - lo))
    if "p_perm" in timing:
        ax.plot([0, 0, 1, 1], [hi + 0.12 * (hi - lo), hi + 0.17 * (hi - lo),
                               hi + 0.17 * (hi - lo), hi + 0.12 * (hi - lo)],
                color="#1a1a1a", lw=0.8)
        ax.text(0.5, hi + 0.19 * (hi - lo),
                f"{_p_txt(timing['p_perm'])}\npaired {_p_txt(timing['p_paired'])}"
                if np.isfinite(timing.get("p_paired", np.nan))
                else _p_txt(timing["p_perm"]),
                ha="center", va="bottom", fontsize=rfig.FS_TICK - 2 - fs_scale)
    ax.set_ylabel(f"after − before {LOCKS[lock]['label']} (z)\n"
                  f"(0, +{WIN_MS:.0f}] − [−{WIN_MS:.0f}, 0) ms",
                  fontsize=rfig.FS_TICK - fs_scale)
    ax.set_title(title or ("Which rises later?" + (
        f"\n{left} = mOFC + lOFC" if left == "OFC" else "")),
        fontsize=rfig.FS_LABEL - 1 - fs_scale)
    ax.tick_params(axis="y", labelsize=rfig.FS_TICK - 1 - fs_scale)


def draw_contrast(ax, C, t_ms, pair, lock="ripple", scale="peak", title=None,
                  legend=True, fs_scale=0):
    """Both components on one axis, the gap filled by which one is bigger;
    contrast clusters as bars (thick: unpaired, thin: paired check)."""
    base, _ = _lock(t_ms, lock)
    left = pair[0]
    cA, cB = ROI_COL[left], ROI_COL["mPFC"]
    r = C["res"][scale]
    a, b = C["A"] / r["scale_A"], C["B"] / r["scale_B"]
    ma, mb = a.mean(0), b.mean(0)
    for m, X, c, lab, n in ((ma, a, cA, ROI_SHOW.get(left, left), len(C["uA"])),
                            (mb, b, cB, "mPFC", len(C["uB"]))):
        se = X.std(0, ddof=1) / np.sqrt(len(X))
        ax.fill_between(t_ms, m - se, m + se, color=c, alpha=0.15, lw=0)
        ax.plot(t_ms, m, color=c, lw=1.4, label=f"{lab} (n = {n})")
    ax.fill_between(t_ms, ma, mb, where=mb >= ma, color=cB, alpha=0.45, lw=0,
                    interpolate=True)
    ax.fill_between(t_ms, ma, mb, where=mb < ma, color=cA, alpha=0.45, lw=0,
                    interpolate=True)
    ax.axhline(0, color="#3d3d3d", lw=0.6, ls="--")
    ax.axvline(0, color="#1a1a1a", lw=0.7, ls=":")
    for a_, b_ in _runs(base):
        ax.axvspan(t_ms[a_], t_ms[b_ - 1] + (t_ms[1] - t_ms[0]),
                   color="#000000", alpha=0.06, lw=0)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))
    for i, (cl, key) in enumerate(((r["clusters"], "unpaired"),
                                   (r["paired_clusters"], "paired"))):
        y = 0.06 + 0.075 * (1 - i)
        for c in cl:
            if c["p"] >= 0.05:
                continue
            col = cB if c["bigger"] == "mPFC" else cA
            ax.plot([c["start_ms"], c["stop_ms"]], [y, y],
                    transform=ax.get_xaxis_transform(), color=col,
                    lw=3.0 if key == "unpaired" else 1.6,
                    solid_capstyle="butt")
            ax.text(c["stop_ms"] + 25, y, _p_txt(c["p"]) + (
                " (paired)" if key == "paired" else ""),
                transform=ax.get_xaxis_transform(), ha="left", va="center",
                fontsize=rfig.FS_TICK - 3 - fs_scale, color=col)
    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.set_xticks(LOCKS[lock]["xticks"])
    ax.set_xlabel(f"time from {LOCKS[lock]['label']} (ms)",
                  fontsize=rfig.FS_LABEL - fs_scale)
    ax.set_ylabel("HFB (z)" if scale == "z" else "HFB / own peak",
                  fontsize=rfig.FS_LABEL - fs_scale)
    sig = [c for c in r["clusters"] if c["p"] < 0.05]
    head = title or ("as measured (z)" if scale == "z"
                     else "scaled to each ROI's peak")
    ax.set_title(f"{head}\n" + ("; ".join(
        f"{c['bigger']} bigger {c['start_ms']:+.0f}…{c['stop_ms']:+.0f} ms"
        for c in sig) or "no significant difference"),
        fontsize=rfig.FS_TICK - fs_scale)
    ax.tick_params(labelsize=rfig.FS_TICK - 1 - fs_scale)
    if legend:
        ax.legend(frameon=False, fontsize=rfig.FS_TICK - 2 - fs_scale,
                  loc="upper left", handlelength=1.0, labelspacing=0.2)


# ── writing: figures, captions, json ─────────────────────────────────────
def _cl_txt(cl, key="sign"):
    s_ = [f"{c[key]} {'bigger ' if key == 'bigger' else ''}"
          f"{c['start_ms']:+.0f} to {c['stop_ms']:+.0f} ms, {_p_txt(c['p'])}"
          for c in cl if c["p"] < 0.05]
    return "; ".join(s_) or "none"


def report_pair(reg, t_ms, pair, out_dir, lock="ripple", unit="session",
                parts=("sliding", "timing", "contrast"), title=None,
                settings=None):
    """Run the analyses for one set of curves and write every output."""
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    left, label = pair[0], LOCKS[lock]["label"]
    title = title or f"{label.capitalize()}-locked HFB"
    out = {"settings": dict(settings or {}, rois=list(pair), lock=lock,
                            unit=unit, sliding_window_ms=SLIDE_MS,
                            baseline=LOCKS[lock]["base_txt"],
                            n_perm=CLUSTER_PERM, seed=SEED,
                            curve="real − shifted null per unit"),
           "t_ms": t_ms.tolist()}
    res = timing = C = None
    if "sliding" in parts or "timing" in parts:
        res, timing = analyse_pair(reg, t_ms, pair, lock, unit)
        out["timing"] = timing
        out["results"] = {r: {k: (v.tolist() if isinstance(v, np.ndarray)
                                  else v.to_dict() if isinstance(v, pd.Series)
                                  else v) for k, v in d.items()}
                          for r, d in res.items()}
    if "contrast" in parts:
        C = analyse_contrast(reg, t_ms, pair, lock, unit)
        out["contrast"] = C["res"]
    with plt.rc_context(rfig._rc()):
        if "sliding" in parts:
            fig, ax = plt.subplots(figsize=(8.5 * rfig.CM, 6.4 * rfig.CM))
            draw_sliding(ax, res, t_ms, pair, lock, title=title)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(out_dir, f"sliding_panel_a.{ext}"),
                            dpi=300)
            with open(os.path.join(out_dir, "caption.txt"), "w") as f:
                f.write(
                    f"{title} (real − shifted null) for "
                    f"{ROI_SHOW.get(left, left)} (left axis) and mPFC (right "
                    f"axis), mean ± SEM over {unit}s, {SLIDE_MS:.0f} ms "
                    f"sliding mean. Each axis is scaled to that ROI's peak, so "
                    f"the two peaks and zero lines coincide. Each ROI is "
                    f"baselined to {LOCKS[lock]['base_txt']} (grey). Bars: "
                    f"clusters vs that baseline (cluster-based sign-flip "
                    f"permutation over the rest of the epoch, {CLUSTER_PERM} "
                    f"permutations, p < 0.05). " + " ".join(
                        f"{r}: {_cl_txt(d['clusters'])}." for r, d in
                        res.items()) + "\n")
        if "timing" in parts:
            fig, ax = plt.subplots(figsize=(5.0 * rfig.CM, 6.4 * rfig.CM))
            draw_timing(ax, res, timing, pair, lock)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(out_dir, f"sliding_timing.{ext}"),
                            dpi=300)
            with open(os.path.join(out_dir, "caption_timing.txt"), "w") as f:
                f.write(
                    f"Direction index (mean HFB (0, +{WIN_MS:.0f}] minus "
                    f"[−{WIN_MS:.0f}, 0) ms around the {label}, real − shifted "
                    f"null, unsmoothed) per {unit}; marker: mean ± SEM; grey "
                    f"lines join {unit}s with both ROIs. " + (
                        f"mPFC − {left} = {timing['diff']:+.4f}, ROI-label "
                        f"permutation {_p_txt(timing['p_perm'])}; paired over "
                        f"the {timing['n_paired']} shared {unit}s "
                        f"{_p_txt(timing.get('p_paired', np.nan))}."
                        if "p_perm" in timing else "Too few units to test.")
                    + "\n")
        if "contrast" in parts:
            fig, axes = plt.subplots(1, 2, figsize=(16 * rfig.CM,
                                                    6.6 * rfig.CM))
            for ax, scale in zip(axes, ("z", "peak")):
                draw_contrast(ax, C, t_ms, pair, lock, scale,
                              legend=scale == "z")
            fig.suptitle(f"Where is mPFC bigger, where is {left} bigger? "
                         f"{title}", fontsize=rfig.FS_LABEL)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(out_dir, f"contrast_panel_a.{ext}"),
                            dpi=300)
            r = C["res"]
            with open(os.path.join(out_dir, "caption_contrast.txt"), "w") as f:
                f.write(
                    f"{title} (real − shifted null, baselined to "
                    f"{LOCKS[lock]['base_txt']}, {SLIDE_MS:.0f} ms sliding "
                    f"mean) for {ROI_SHOW.get(left, left)} and mPFC, mean ± SEM "
                    f"over {unit}s. Fill: green where mPFC is above, orange "
                    f"where {left} is above. Left: as measured; right: each ROI "
                    f"divided by its own group-mean peak. Thick bars: contrast "
                    f"clusters (Welch t, ROI-label permutation over {unit}s, "
                    f"{CLUSTER_PERM} permutations, scaling recomputed in every "
                    f"permutation); thin bars: paired check on the "
                    f"{r['z']['n_paired']} {unit}s with both ROIs (sign-flip). "
                    f"As measured: {_cl_txt(r['z']['clusters'], 'bigger')} "
                    f"(paired: {_cl_txt(r['z']['paired_clusters'], 'bigger')})."
                    f" Peak-scaled: {_cl_txt(r['peak']['clusters'], 'bigger')}"
                    f" (paired: "
                    f"{_cl_txt(r['peak']['paired_clusters'], 'bigger')}).\n")
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    return res, timing, C


# ── the two commands on the saved all-ripple run ─────────────────────────
def _region_inputs(region_dir, partition_dir, left, unit):
    G = os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                     "group", "swr")
    region_dir = region_dir or os.path.join(G, REGION_RUN)
    partition_dir = partition_dir or os.path.join(G, PARTITION)
    merge = OFC_MERGE if left == "OFC" else None
    t_ms, reg = region_curves(region_dir, partition_dir, (left, "mPFC"), unit,
                              merge=merge)
    return G, t_ms, reg, dict(script=os.path.relpath(__file__),
                              created=datetime.date.today().isoformat(),
                              region_dir=region_dir,
                              partition_dir=partition_dir, merge=merge,
                              ripples="all ripples (region run), "
                                      "different-shaft")


def sliding(region_dir=None, partition_dir=None, out_dir=None, unit="session",
            show=True, left="mOFC"):
    """Panel a) with a sliding-window test per ROI and the timing test.
    `left`: mOFC, lOFC, or OFC (both collapsed per session)."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    G, t_ms, reg, st = _region_inputs(region_dir, partition_dir, left, unit)
    out_dir = out_dir or os.path.join(
        G, f"{ANALYSIS_NAME}_sliding_{left}_{st['created']}")
    report_pair(reg, t_ms, (left, "mPFC"), out_dir, "ripple", unit,
                parts=("sliding", "timing"),
                title="Ripple-locked HFB, all ripples",
                settings=dict(st, analysis=f"{ANALYSIS_NAME}_sliding"))
    print(f"\n  saved -> {out_dir}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()


def contrast(region_dir=None, partition_dir=None, out_dir=None,
             unit="session", show=True, left="OFC"):
    """Panel a) as a contrast: where is mPFC bigger, where is `left`?"""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    G, t_ms, reg, st = _region_inputs(region_dir, partition_dir, left, unit)
    out_dir = out_dir or os.path.join(
        G, f"{ANALYSIS_NAME}_contrast_{left}_{st['created']}")
    report_pair(reg, t_ms, (left, "mPFC"), out_dir, "ripple", unit,
                parts=("contrast",), title="Ripple-locked HFB, all ripples",
                settings=dict(st, analysis=f"{ANALYSIS_NAME}_contrast"))
    print(f"\n  saved -> {out_dir}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()



if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run, "sliding": sliding, "contrast": contrast})
    else:
        run()
