#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the ripple-locked frontal HFB response depend on WHAT the ripple follows?

`swr_ripple_locked_hfb.py` established that medial frontal HFB rises around
hippocampal ripples. This asks whether that rise is bigger for ripples that
follow a REWARDED uncovering during exploration -- the moment new plan-relevant
information arrives -- than for matched control events.

That is SK's mechanism made testable: the hippocampus should be telling mPFC
something when there is something new to tell.

CONDITIONS. Each ripple is assigned to the single NEAREST preceding labelled
event within `post_s`, so no ripple is counted twice:

    reward_explore   TARGET -- correct uncovering during the first traversal
    error_explore    same phase, same action, opposite feedback
    reward_plan      same feedback and action, route known but not reliable
    reward_execute   same feedback and action, route known and reliable
    move_explore     same phase, a keypress, NO feedback
    still_explore    same phase, no keypress within +-post_s at all

WHY A SHIFTED NULL IS STILL NEEDED. The conditions differ in state by
construction -- SWR_SUMMARY F1 shows ripple rate rises with stillness and F3
that the first traversal is stiller -- so the state confound does NOT cancel in
a between-condition contrast. Each condition gets its own shifted null and the
estimate is real-minus-null within condition. The condition label always comes
from the REAL ripple time; only the HFB sampling point moves.

WHY COUNTS ARE EQUALISED. Conditions differ enormously in ripple count, and
SWR_SUMMARY §5 records that comparing across unequal counts once produced a
spurious effect in this project. For every direct contrast both sides are
subsampled to the smaller n within session, with a fixed seed.

Usage:
    python scripts/swr_ripple_hfb_conditions.py run --bundle=<bundle_v2 dir>
    python scripts/swr_ripple_hfb_conditions.py run --bundle=<dir> --post_s=1.5

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.swr_bundle as sb
import mc.analyse.swr_behaviour as swb
import mc.analyse.swr_windows as win

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "ripple_hfb_conditions"
HALF_S, PERI_S, NONPERI_S = 1.00, 0.25, (0.25, 0.75)
POST_S = 2.0                  # a ripple belongs to an event this long after it
MIN_RIPPLES = 15              # per (session, condition, derivation)
SHIFT_RANGE_S = (5.0, 120.0)
TARGET = "reward_explore"
CONDITIONS = [TARGET, "error_explore", "reward_plan", "reward_execute",
              "move_explore", "still_explore"]
FRONTAL = {"mPFC": ["mPFC"], "mOFC": ["mOFC"], "Frontal": ["mPFC", "mOFC"]}
SMOOTH_MS = 50.0

# Feedback-valence colours. This is a new categorical variable -- CLAUDE.md
# fixes colours for states, phases, locations and ROIs, but not for feedback --
# so these are taken from the documented families: the two greens are grid
# positions 7 and 5 (also the project's "observed value" dark green), the pink
# is the lOFC magenta. Change them here, nowhere else.
VALENCE_C = {
    "reward_explore": "#0e3d3a",    # dark green  -- positive feedback, explore
    "reward_execute": "#5b9b8d",    # bright green -- positive feedback, execute
    "error_explore":  "#a30d6c",    # dark pink   -- negative feedback, explore
}

# Display names. The internal keys stay machine-readable; these are what a
# reader sees, and they say what the event actually is rather than what the
# code calls it.
COND_LABEL = {
    "reward_explore": "first reward uncovers",
    "reward_execute": "reward uncovers during execution",
    "reward_plan": "reward uncovers during planning",
    "error_explore": "error, explore",
    "move_explore": "movement press, explore",
    "still_explore": "stillness, explore",
}


def _clean_mask(rows, n, fs):
    m = np.zeros(n, bool)
    for a, b in rows:
        i0, i1 = max(0, int(a * fs)), min(n, int(b * fs))
        if i1 > i0:
            m[i0:i1] = True
    return m


def session_events(sess, data_root):
    """Labelled behavioural events for one session: (time, condition)."""
    try:
        beh = win.add_phase3(swr_io.load_behaviour(sess, data_root=data_root))
        unc = swb.uncover_events(sess, data_root=data_root)
    except Exception:
        return pd.DataFrame(columns=["t_s", "cond"])
    if not len(unc):
        return pd.DataFrame(columns=["t_s", "cond"])
    ph = beh[["grid_no", "rep_overall", "phase3"]].drop_duplicates()
    u = unc.merge(ph, on=["grid_no", "rep_overall"], how="left")
    rows = []
    for _, r in u.iterrows():
        p, ok = r.get("phase3"), int(r.correct)
        if p == "explore":
            rows.append((r.t_s, "reward_explore" if ok else "error_explore"))
        elif p == "plan" and ok:
            rows.append((r.t_s, "reward_plan"))
        elif p == "execute" and ok:
            rows.append((r.t_s, "reward_execute"))

    # movement presses during explore, and stillness during explore
    expl = beh[beh.phase3 == "explore"]
    press_t = []
    for grid, g in expl.groupby("grid_no"):
        t, is_move = swb.movement_series(sess, int(grid), data_root=data_root)
        if t is None:
            continue
        onset = float(g.new_grid_onset.iloc[0])
        mv = onset + t[is_move]
        lo, hi = float(g.t_A.min()), float(g.t_D.max())
        # A grid can carry NaN bounds when its last repeat was never completed.
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            continue
        mv = mv[(mv >= lo) & (mv <= hi)]
        press_t.extend(mv.tolist())
        rows.extend((x, "move_explore") for x in mv)
    # stillness: explore time at least POST_S from ANY press
    allp = np.sort(np.r_[u.t_s.to_numpy(float), np.asarray(press_t, float)])
    for grid, g in expl.groupby("grid_no"):
        lo, hi = float(g.t_A.min()), float(g.t_D.max())
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            continue
        cand = np.arange(lo, hi, 0.5)
        if allp.size:
            j = np.clip(np.searchsorted(allp, cand), 1, len(allp) - 1)
            near = np.minimum(np.abs(cand - allp[j - 1]), np.abs(cand - allp[j]))
            cand = cand[near >= POST_S]
        rows.extend((x, "still_explore") for x in cand)
    return pd.DataFrame(rows, columns=["t_s", "cond"]).sort_values("t_s")


def assign(ripple_t, ev, post_s=POST_S):
    """Condition of each ripple = nearest preceding event within post_s."""
    out = np.array([None] * len(ripple_t), dtype=object)
    if not len(ev):
        return out
    et, ec = ev.t_s.to_numpy(float), ev.cond.to_numpy(object)
    j = np.searchsorted(et, ripple_t, side="right") - 1
    ok = (j >= 0)
    jj = np.clip(j, 0, len(et) - 1)
    ok &= (ripple_t - et[jj]) <= post_s
    out[ok] = ec[jj[ok]]
    return out


def run(bundle=None, post_s=POST_S, n_shifts=6, seed=42, save=True, out_dir=None):
    b_dir = bundle or os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                                   "group", "swr", "bundle_v2")
    with open(os.path.join(b_dir, "swr_bundle.pkl"), "rb") as f:
        B = pickle.load(f)
    rip, hfb_pairs, hiv, pairs = (B["ripples"], B["hfb_pairs"],
                                  B["hfb_intervals"], B["pairs"])
    hp = hfb_pairs[~hfb_pairs.excluded.fillna(False)].copy()
    key = pairs[["session", "pair_id", "same_probe_as_hpc"]].drop_duplicates()
    hp = hp.merge(key, on=["session", "pair_id"], how="left")
    hp["same_shaft"] = hp.same_probe_as_hpc.fillna(False).astype(bool)
    hp = hp[~hp.same_shaft]                       # volume conduction, as before

    R = swr_io.get_data_root()
    rng = np.random.default_rng(seed)
    lo, hi = SHIFT_RANGE_S
    shifts = [0.0] + [float(x) for x in
                      rng.uniform(lo, hi, n_shifts) * rng.choice([-1, 1], n_shifts)]
    fs = float(B["hfb_index"].out_fs.iloc[0]) if len(B["hfb_index"]) else 100.0
    w = int(HALF_S * fs)
    off = np.arange(-w, w)
    t_s = off / fs
    peri_ix = np.abs(t_s) < PERI_S
    non_ix = (np.abs(t_s) >= NONPERI_S[0]) & (np.abs(t_s) < NONPERI_S[1])

    rows, counts, bal_rows = [], [], []
    tc, tc_ix = [], []
    for sess in sorted(hp.session.unique()):
        g = hp[hp.session == sess]
        cx = g[g.roi_family.isin(["mPFC", "mOFC"])]
        if not len(cx):
            continue
        ev = session_events(int(sess), R)
        if not len(ev):
            continue
        try:
            D = sb.load_hfb(b_dir, sess, arrays=["hfb"])
        except Exception:
            continue
        H, ids = D["hfb"], D["pair_ids"]
        n = H.shape[1]
        idx = {p: i for i, p in enumerate(ids)}
        masks = {p: _clean_mask(
            hiv[(hiv.session == sess) & (hiv.pair_id == p)]
            [["start_s", "stop_s"]].to_numpy(float), n, fs) for p in g.pair_id}

        t_all = rip.loc[rip.session == sess, "t_peak_s"].to_numpy(float)
        t_all = np.unique(np.round(t_all, 3))     # pool HC derivations, dedup
        cond = assign(t_all, ev, post_s)
        for c in CONDITIONS:
            counts.append({"session": sess, "cond": c,
                           "n_ripples": int((cond == c).sum())})

        # Count equalisation. Conditions differ ~10-fold in ripple count
        # (reward_execute 97k vs reward_explore 9.5k), and a mean over 10x more
        # events is a quieter estimate, which flatters whichever side has more.
        # Each condition present in this session is subsampled to the smallest
        # of them, with a fixed seed, and BOTH the full and balanced estimates
        # are stored so the two can be compared rather than one being asserted.
        # PAIRWISE, not global. Balancing all six conditions to the smallest of
        # them cut the target from 46 ripples to 30 even when the control it was
        # being compared against had 68 -- power thrown away for nothing. Each
        # contrast now subsamples only the LARGER of its two sides.
        avail = {c: t_all[cond == c] for c in CONDITIONS}
        avail = {c: v for c, v in avail.items() if len(v) >= MIN_RIPPLES}
        srng = np.random.default_rng(seed + int(sess))
        pair_bal = {}
        if TARGET in avail:
            for c in CONDITIONS:
                if c == TARGET or c not in avail:
                    continue
                k = min(len(avail[TARGET]), len(avail[c]))
                if k < MIN_RIPPLES:
                    continue
                pair_bal[c] = {
                    TARGET: np.sort(srng.choice(avail[TARGET], k, replace=False)),
                    c: np.sort(srng.choice(avail[c], k, replace=False)), "n": k}

        for _, x in cx.iterrows():
            if x.pair_id not in idx:
                continue
            for c in CONDITIONS:
                tt = t_all[cond == c]
                if len(tt) < MIN_RIPPLES:
                    continue
                for sh in shifts:
                    ci = np.round((tt + sh) * fs).astype(int)
                    ci = ci[(ci - w >= 0) & (ci + w < n)]
                    if len(ci) < MIN_RIPPLES:
                        continue
                    wi = ci[:, None] + off[None, :]
                    ok = masks[x.pair_id][wi].all(1)
                    if ok.sum() < MIN_RIPPLES:
                        continue
                    m = H[idx[x.pair_id]][wi[ok]].mean(0)
                    rows.append({
                        "session": sess, "subject": x.subject_label,
                        "cx_pair": x.pair_id, "roi": x.roi_family,
                        "cond": c, "shift_s": sh, "is_real": sh == 0.0,
                        "n_ripples": int(ok.sum()),
                        "diff": float(m[peri_ix].mean() - m[non_ix].mean())})
                    tc.append(m.astype(np.float32))
                    tc_ix.append({"session": sess, "subject": x.subject_label,
                                  "cx_pair": x.pair_id, "roi": x.roi_family,
                                  "cond": c, "is_real": sh == 0.0})

            # pairwise count-balanced estimates, one row per contrast per side
            for ctrl, sets in pair_bal.items():
                for sh in shifts:
                    vals = {}
                    for side in (TARGET, ctrl):
                        cb = np.round((sets[side] + sh) * fs).astype(int)
                        cb = cb[(cb - w >= 0) & (cb + w < n)]
                        if not len(cb):
                            break
                        wb = cb[:, None] + off[None, :]
                        okb = masks[x.pair_id][wb].all(1)
                        if okb.sum() < MIN_RIPPLES:
                            break
                        mb = H[idx[x.pair_id]][wb[okb]].mean(0)
                        vals[side] = (float(mb[peri_ix].mean()
                                            - mb[non_ix].mean()), int(okb.sum()))
                    if len(vals) == 2:
                        for side, (dv, nn) in vals.items():
                            bal_rows.append({
                                "session": sess, "subject": x.subject_label,
                                "cx_pair": x.pair_id, "roi": x.roi_family,
                                "contrast": ctrl, "side": side,
                                "shift_s": sh, "is_real": sh == 0.0,
                                "n_ripples": nn, "diff": dv})
        print(f"  s{sess:02d}", end="\r")

    d = pd.DataFrame(rows)
    cnt = pd.DataFrame(counts)
    db = pd.DataFrame(bal_rows)
    if not len(d):
        raise RuntimeError("no (session, condition, derivation) cell had enough ripples")
    print(f"\n\n{len(d)} rows | {d.session.nunique()} sessions | "
          f"{d.cx_pair.nunique()} frontal derivations")
    print("\nripples per condition (pooled over sessions):")
    print(cnt.groupby("cond").n_ripples.agg(["sum", "median"]).to_string())

    res = _stats(d, db)
    _report(res, cnt)

    if save:
        out_dir = out_dir or os.path.join(
            swr_io.derivatives_dir(R), "group", "swr",
            f"{ANALYSIS_NAME}_{datetime.now():%Y-%m-%d}")
        os.makedirs(out_dir, exist_ok=True)
        d.to_csv(os.path.join(out_dir, "per_cell.csv"), index=False)
        if len(db):
            db.to_csv(os.path.join(out_dir, "per_cell_balanced.csv"),
                      index=False)
        np.savez_compressed(os.path.join(out_dir, "timecourses.npz"),
                            t_ms=off / fs * 1000.0, traces=np.stack(tc))
        pd.DataFrame(tc_ix).to_csv(
            os.path.join(out_dir, "timecourse_index.csv"), index=False)
        cnt.to_csv(os.path.join(out_dir, "ripple_counts.csv"), index=False)
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump({"analysis": ANALYSIS_NAME, "bundle": b_dir,
                       "post_s": post_s, "peri_s": PERI_S,
                       "nonperi_s": list(NONPERI_S), "half_s": HALF_S,
                       "n_shifts": n_shifts, "seed": seed,
                       "min_ripples": MIN_RIPPLES, "target": TARGET,
                       "unit": "session (primary)",
                       "created": datetime.now().isoformat(timespec="seconds"),
                       "results": res}, f, indent=2, default=str)
        print(f"\nsaved -> {out_dir}")
        try:
            figure(results=out_dir)
        except Exception as e:
            print(f"  [figure skipped: {type(e).__name__}: {e}]")
    return None


def _eff(g, unit="session", col="diff"):
    g = g.dropna(subset=[col])
    real = g[g.is_real].groupby(unit)[col].mean()
    null = g[~g.is_real].groupby(unit)[col].mean()
    c = real.index.intersection(null.index)
    return real[c] - null[c]


def _stats(d, db=None):
    from scipy import stats as st
    out = {}
    for name, rois in FRONTAL.items():
        g0 = d[d.roi.isin(rois)]
        out[name] = {"per_condition": {}, "vs_target": {},
                     "vs_target_unbalanced": {}}
        for c in CONDITIONS:
            v = _eff(g0[g0.cond == c])
            if len(v) < 5:
                continue
            t, p = st.ttest_1samp(v.to_numpy(), 0.0)
            out[name]["per_condition"][c] = {
                "n_sessions": int(len(v)), "effect": float(v.mean()),
                "sem": float(v.std(ddof=1) / np.sqrt(len(v))),
                "t": float(t), "p": float(p),
                "n_ripples": int(g0[(g0.cond == c) & g0.is_real].n_ripples.sum())}
        # paired contrasts against the target, within session -- on the
        # COUNT-BALANCED estimate, with the unbalanced one reported beside it
        # unbalanced, for reference
        tv = _eff(g0[g0.cond == TARGET])
        for c in CONDITIONS:
            if c == TARGET:
                continue
            cv = _eff(g0[g0.cond == c])
            common = tv.index.intersection(cv.index)
            if len(common) < 5:
                continue
            t, p = st.ttest_rel(tv[common], cv[common])
            out[name]["vs_target_unbalanced"][c] = {
                "n_sessions": int(len(common)),
                "diff": float((tv[common] - cv[common]).mean()),
                "t": float(t), "p": float(p)}
        # PRIMARY: pairwise count-balanced
        if db is not None and len(db):
            b0 = db[db.roi.isin(rois)]
            for c in CONDITIONS:
                if c == TARGET:
                    continue
                gb = b0[b0.contrast == c]
                if not len(gb):
                    continue
                tvb = _eff(gb[gb.side == TARGET])
                cvb = _eff(gb[gb.side == c])
                common = tvb.index.intersection(cvb.index)
                if len(common) < 5:
                    continue
                dd = (tvb[common] - cvb[common]).to_numpy()
                t, p = st.ttest_rel(tvb[common], cvb[common])
                out[name]["vs_target"][c] = {
                    "n_sessions": int(len(common)), "diff": float(dd.mean()),
                    "sem": float(dd.std(ddof=1) / np.sqrt(len(dd))),
                    "t": float(t), "p": float(p),
                    "n_ripples_per_side": int(
                        gb[gb.is_real].groupby("side").n_ripples.sum().min())}
    return out


def _report(res, cnt):
    print("\n" + "=" * 78)
    print(f" PERI-RIPPLE FRONTAL HFB BY CONDITION   (session-level, "
          f"real minus shifted null)")
    print(f" target = {TARGET}; each ripple assigned to its nearest preceding event")
    print("=" * 78)
    for name in FRONTAL:
        pc = res[name]["per_condition"]
        if not pc:
            continue
        print(f"\n  --- {name} ---")
        print(f"    {'condition':<18s}{'n_sess':>7s}{'ripples':>9s}"
              f"{'effect':>10s}{'t':>7s}{'p':>9s}")
        for c in CONDITIONS:
            v = pc.get(c)
            if not v:
                continue
            star = "*" if v["p"] < 0.05 else " "
            mark = "  <- TARGET" if c == TARGET else ""
            print(f"    {c:<18s}{v['n_sessions']:>7d}{v['n_ripples']:>9d}"
                  f"{v['effect']:>10.4f}{v['t']:>7.2f}{v['p']:>9.4f}{star}{mark}")
        vt, vu = res[name].get("vs_target", {}), res[name].get(
            "vs_target_unbalanced", {})
        if vt:
            print(f"    {TARGET} MINUS each control, COUNT-BALANCED "
                  f"(unbalanced in brackets):")
            for c, v in vt.items():
                star = "*" if v["p"] < 0.05 else " "
                u = vu.get(c)
                ub = (f"   [unbal {u['diff']:+.4f} p={u['p']:.3f}]" if u else "")
                print(f"      vs {c:<16s} n={v['n_sessions']:>3d}  "
                      f"{v['diff']:+.4f} ± {v['sem']:.4f}  "
                      f"t={v['t']:+.2f}  p={v['p']:.4f}{star}{ub}")
    print("\n" + "=" * 78)


def figure(results=None, out_stem=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = results
    res = json.load(open(os.path.join(R, "result.json")))["results"]
    CM = 1 / 2.54
    COND_C = {TARGET: "#F15A29", "error_explore": "#6B60AA",
              "reward_plan": "#D7657F", "reward_execute": "#5C1027",
              "move_explore": "#8C8C8C", "still_explore": "#C7C6E2"}
    SHORT = {TARGET: "reward\nexplore", "error_explore": "error\nexplore",
             "reward_plan": "reward\nplan", "reward_execute": "reward\nexecute",
             "move_explore": "move\nexplore", "still_explore": "still\nexplore"}
    names = [n for n in FRONTAL if res.get(n, {}).get("per_condition")]
    fig, axes = plt.subplots(2, len(names),
                             figsize=(6.6 * len(names) * CM, 13.5 * CM),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for j, name in enumerate(names):
        pc = res[name]["per_condition"]
        cs = [c for c in CONDITIONS if c in pc]
        ax = axes[0, j]
        top = max(pc[c]["effect"] + pc[c]["sem"] for c in cs)
        for i, c in enumerate(cs):
            v = pc[c]
            ax.bar(i, v["effect"], yerr=v["sem"], color=COND_C.get(c, "#888"),
                   width=0.7, capsize=2.5, error_kw=dict(lw=0.8),
                   edgecolor="black" if c == TARGET else "none",
                   linewidth=1.1 if c == TARGET else 0)
            if v["p"] < 0.05:
                ax.text(i, v["effect"] + v["sem"] + 0.04 * top, "*",
                        ha="center", fontsize=11)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xticks(range(len(cs)))
        ax.set_xticklabels([SHORT.get(c, c) for c in cs], fontsize=7,
                           rotation=45, ha="right")
        ax.set_title(f"{name}  (n={max(v['n_sessions'] for v in pc.values())} sess)",
                     fontsize=10)
        ax.tick_params(labelsize=7.5)
        if j == 0:
            ax.set_ylabel("peri − non-peri,\nminus null (z)", fontsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # --- contrasts, count-balanced pairwise --------------------------
        ax = axes[1, j]
        vt = res[name].get("vs_target", {})
        cs2 = [c for c in CONDITIONS if c in vt]
        if not cs2:
            ax.axis("off")
            continue
        for i, c in enumerate(cs2):
            v = vt[c]
            ax.barh(i, v["diff"], xerr=v["sem"], color=COND_C.get(c, "#888"),
                    height=0.65, capsize=2.5, error_kw=dict(lw=0.8))
            if v["p"] < 0.05:
                ax.text(v["diff"] + np.sign(v["diff"]) * (v["sem"] + 0.0012),
                        i, "*", va="center", fontsize=11)
        ax.axvline(0, color="0.4", lw=0.8)
        ax.set_yticks(range(len(cs2)))
        ax.set_yticklabels([SHORT.get(c, c).replace("\n", " ") for c in cs2],
                           fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("reward_explore − control (z)", fontsize=8.5)
        ax.set_title("contrasts, count-balanced", fontsize=9.5)
        ax.tick_params(labelsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("Peri-ripple frontal HFB by what the ripple follows "
                 "(session-level, different-shaft)", fontsize=11)
    stem = out_stem or os.path.join(R, "ripple_hfb_conditions")
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    print(f"figure -> {stem}.pdf / .png")
    return None


def timecourse_figure(results=None, out_stem=None, rois=("mPFC", "mOFC"),
                      smooth_ms=100.0,
                      conds=("reward_explore", "reward_execute", "error_explore",
                             "move_explore", "still_explore")):
    """Peri-ripple HFB time course per condition, per region.

    Session-level, real minus each condition's own shifted null, baselined to
    the non-peri band -- the same quantity the bars report, so the two agree.

    Smoothed harder than the pooled figure (100 ms, not 50) because splitting by
    condition costs an order of magnitude of ripples: the pooled analysis
    averages thousands per cell, a single condition ~46. The y-limits are set
    from the conditions with >= 20 sessions, so `stillness, explore` (11-12
    sessions, and correspondingly wild) cannot decide the scale for everything
    else -- it is still drawn, dotted, just not allowed to hide the rest.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    R = results
    z = np.load(os.path.join(R, "timecourses.npz"))
    t_ms, tr = z["t_ms"], z["traces"]
    fs = 1000.0 / (t_ms[1] - t_ms[0])
    ix = pd.read_csv(os.path.join(R, "timecourse_index.csv"))
    ix["is_real"] = ix.is_real.astype(bool)
    res = json.load(open(os.path.join(R, "result.json")))["results"]
    flank = (np.abs(t_ms) >= NONPERI_S[0] * 1000) & (np.abs(t_ms) < NONPERI_S[1] * 1000)
    CM = 1 / 2.54
    COND_C = {"reward_explore": "#F15A29", "error_explore": "#6B60AA",
              "reward_plan": "#D7657F", "reward_execute": "#5C1027",
              "move_explore": "#8C8C8C", "still_explore": "#C7C6E2"}

    rois = [r for r in rois if (ix.roi == r).any()]
    fig, axes = plt.subplots(1, len(rois), figsize=(10.0 * len(rois) * CM,
                                                   8.5 * CM),
                             constrained_layout=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, roi in zip(axes, rois):
        pc = res.get(roi, {}).get("per_condition", {})
        well_powered = []
        for c in conds:
            sel = (ix.roi == roi) & (ix.cond == c)
            if not sel.any():
                continue
            curves = []
            for _, gi in ix[sel].groupby("session"):
                rr = gi.index.to_numpy()
                r = tr[rr[gi.is_real.to_numpy()]]
                n = tr[rr[~gi.is_real.to_numpy()]]
                if not len(r) or not len(n):
                    continue
                cc = r.mean(0) - n.mean(0)
                curves.append(cc - cc[flank].mean())
            if len(curves) < 5:
                continue
            A = np.stack(curves)
            if smooth_ms:
                A = gaussian_filter1d(A, (smooth_ms / 1000.0 * fs) / 2.355, axis=-1)
            m, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))
            v = pc.get(c, {})
            star = " *" if v.get("p", 1) < 0.05 else ""
            thin = len(A) < 20
            ax.plot(t_ms, m, color=COND_C.get(c, "#888"), lw=1.6,
                    alpha=0.55 if thin else 1.0, ls=":" if thin else "-",
                    label=f"{COND_LABEL.get(c, c)} ({len(A)} sess){star}")
            ax.fill_between(t_ms, m - se, m + se, color=COND_C.get(c, "#888"),
                            alpha=0.10 if thin else 0.15, lw=0)
            if not thin:
                well_powered.append(m)
        if well_powered:
            lim = 1.6 * max(float(np.abs(np.stack(well_powered)).max()), 1e-6)
            ax.set_ylim(-lim, lim)
        ax.axvline(0, color="0.35", lw=0.8, ls="--")
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvspan(-PERI_S * 1000, PERI_S * 1000, color="#F15A29", alpha=0.06, lw=0)
        ax.set_xlabel("Time from hippocampal ripple peak (ms)", fontsize=9)
        ax.set_title(roi, fontsize=11)
        ax.tick_params(labelsize=8)
        ax.margins(x=0)
        ax.legend(fontsize=7, frameon=False, loc="upper left", handlelength=1.4,
                  labelspacing=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("HFB, real − shifted null,\nvs non-peri (z)", fontsize=9)
    fig.suptitle("Peri-ripple frontal HFB by what the ripple follows "
                 "(session-level, different-shaft)", fontsize=11)
    stem = out_stem or os.path.join(R, "ripple_hfb_conditions_timecourse")
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    print(f"figure -> {stem}.pdf / .png")
    return None


def frontal_figure(results=None, out_stem=None, smooth_ms=100.0, n_perm=2000,
                   seed=42, width_cm=3.0, height_cm=3.0,
                   conds=("reward_explore", "reward_execute", "error_explore")):
    """One small panel: peri-ripple HFB in COLLAPSED frontal (mPFC + mOFC).

    Significance is a cluster-based permutation over time bins
    (`swr_sakon.cluster_perm_time`, sign-flipping across sessions) -- the same
    function the hippocampal branch uses, so the correction is identical. The
    test runs on the SAME smoothed traces that are drawn: sign-flipping
    preserves the smoothing, so the null carries the same autocorrelation and
    the correction stays valid.

    Two files are written: the bare panel at `width_cm` x `height_cm` for
    dropping into a figure, and a `_labelled` version with legend, stats and
    axis titles, because at 3 cm nothing but the traces fits.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    import mc.analyse.swr_sakon as sk

    R = results
    z = np.load(os.path.join(R, "timecourses.npz"))
    t_ms, tr = z["t_ms"], z["traces"]
    fs = 1000.0 / (t_ms[1] - t_ms[0])
    ix = pd.read_csv(os.path.join(R, "timecourse_index.csv"))
    ix["is_real"] = ix.is_real.astype(bool)
    flank = (np.abs(t_ms) >= NONPERI_S[0] * 1000) & (np.abs(t_ms) < NONPERI_S[1] * 1000)
    CM = 1 / 2.54

    # collapsed frontal: average a session's mPFC and mOFC derivations together
    curves, stats_out = {}, {}
    for c in conds:
        sel = (ix.roi.isin(["mPFC", "mOFC"])) & (ix.cond == c)
        if not sel.any():
            continue
        rows = []
        for _, gi in ix[sel].groupby("session"):
            rr = gi.index.to_numpy()
            r = tr[rr[gi.is_real.to_numpy()]]
            n = tr[rr[~gi.is_real.to_numpy()]]
            if not len(r) or not len(n):
                continue
            cc = r.mean(0) - n.mean(0)
            rows.append(cc - cc[flank].mean())
        if len(rows) < 5:
            continue
        A = np.stack(rows)
        if smooth_ms:
            A = gaussian_filter1d(A, (smooth_ms / 1000.0 * fs) / 2.355, axis=-1)
        curves[c] = A
        t_obs, cl, pv, _ = sk.cluster_perm_time(A, n_perm=n_perm, seed=seed)
        sig = [(float(t_ms[a]), float(t_ms[b - 1]), float(pp))
               for (a, b), pp in zip(cl, pv) if pp < 0.05]
        stats_out[c] = {"n_sessions": int(len(A)), "clusters": sig,
                        "peak_z": float(A.mean(0).max()),
                        "peak_at_ms": float(t_ms[int(np.argmax(A.mean(0)))])}
        lab = COND_LABEL.get(c, c)
        print(f"  {lab:<34s} n={len(A):>3d} sessions  peak {A.mean(0).max():+.4f} z "
              f"at {t_ms[int(np.argmax(A.mean(0)))]:+.0f} ms")
        for a, b, pp in sig:
            print(f"      significant cluster {a:+.0f} to {b:+.0f} ms, p = {pp:.4f}")
        if not sig:
            print(f"      no cluster survives correction")

    if not curves:
        raise RuntimeError("no condition had enough sessions")

    lo = min(float((A.mean(0) - A.std(0, ddof=1) / np.sqrt(len(A))).min())
             for A in curves.values())
    hi = max(float((A.mean(0) + A.std(0, ddof=1) / np.sqrt(len(A))).max())
             for A in curves.values())
    span = hi - lo

    # Three sizes, because 3 cm and CLAUDE.md's 9 pt minimum are in tension.
    # An eighth of A4 is ~7.4 x 5.2 cm, which is what 11 pt is calibrated for;
    # 9 pt on a 3 cm panel is proportionally the same as ~22 pt there, and it
    # swallows the axes. The 3 cm panel therefore runs at 6.5 pt -- below the
    # documented floor, deliberately, because the alternative is a panel that is
    # all label and no data. `panel_9pt` is the smallest size that keeps the
    # rule, and is the one to use if the floor matters more than the width.
    VARIANTS = [("", width_cm, height_cm, 6.5, False),
                ("_9pt", 5.2, 4.6, 9.0, False),
                ("_labelled", 10.0, 7.0, 9.0, True)]
    for suffix, wcm, hcm, fpt, labelled in VARIANTS:
        fig, ax = plt.subplots(figsize=(wcm * CM, hcm * CM),
                               constrained_layout=True)
        for i, (c, A) in enumerate(curves.items()):
            m = A.mean(0)
            se = A.std(0, ddof=1) / np.sqrt(len(A))
            col = VALENCE_C.get(c, "#888")
            ax.plot(t_ms, m, color=col, lw=1.1 if not labelled else 1.4,
                    solid_capstyle="round",
                    label=f"{COND_LABEL.get(c, c)} ({len(A)})")
            ax.fill_between(t_ms, m - se, m + se, color=col, alpha=0.16, lw=0)
            y = hi + span * (0.10 + 0.085 * i)
            for a, b, _pp in stats_out[c]["clusters"]:
                ax.plot([a, b], [y, y], color=col,
                        lw=2.0 if not labelled else 2.6,
                        solid_capstyle="butt", clip_on=False)
        ax.axvline(0, color="0.45", lw=0.7, ls=(0, (2, 2)))
        ax.axhline(0, color="0.75", lw=0.6)
        ax.set_xlim(t_ms[0], t_ms[-1])
        ax.set_ylim(lo - span * 0.10, hi + span * (0.10 + 0.085 * len(curves)))
        # 0 is already marked by the dashed line, so labelling it too just
        # collides with +-500 on a 3 cm axis.
        ax.set_xticks([-500, 0, 500] if wcm >= 5 else [-500, 500])
        ax.set_yticks([0.0, float(np.round(hi, 2))])
        ax.tick_params(labelsize=fpt - 0.5, length=2.2, pad=1.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if labelled:
            ax.set_xlabel("Time from hippocampal ripple peak (ms)", fontsize=fpt)
            ax.set_ylabel("HFB, real − shifted null, vs non-peri (z)",
                          fontsize=fpt)
            ax.set_title("Frontal (mPFC + mOFC), session-level\n"
                         "bars = cluster-corrected p < 0.05", fontsize=fpt + 1)
            ax.legend(fontsize=fpt - 1, frameon=False, loc="upper center",
                      bbox_to_anchor=(0.5, -0.22), handlelength=1.6)
        else:
            ax.set_xlabel("ms from ripple", fontsize=fpt, labelpad=1)
            ax.set_ylabel("HFB (z)", fontsize=fpt, labelpad=1)
        stem = (out_stem or os.path.join(R, "frontal_hfb_timecourse")) + suffix
        fig.savefig(stem + ".pdf")
        fig.savefig(stem + ".png", dpi=600)
        plt.close(fig)
        print(f"figure -> {stem}.pdf / .png   ({wcm:.1f} x {hcm:.1f} cm, {fpt} pt)")

    with open(os.path.join(R, "frontal_timecourse_clusters.json"), "w") as f:
        json.dump({"smooth_ms": smooth_ms, "n_perm": n_perm, "seed": seed,
                   "roi": "Frontal = mPFC + mOFC, different-shaft",
                   "unit": "session", "colours": VALENCE_C,
                   "stats": stats_out}, f, indent=2)
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run, "figure": figure,
                   "timecourse_figure": timecourse_figure,
                   "frontal_figure": frontal_figure})
    else:
        run()
