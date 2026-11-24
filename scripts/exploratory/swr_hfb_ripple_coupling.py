#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ripple by ripple: does mOFC activity BEFORE a ripple predict mPFC activity
AFTER the same ripple?

The direction index (`swr_hfb_direction.py`) found mPFC HFB leaning later than
mOFC around hippocampal ripples, on session averages. SK's mechanism is a chain
through the ripple: mOFC signals "there is a reward", the hippocampus ripples,
mPFC takes it up for the plan. A chain makes a prediction session averages
cannot test -- it is about the SAME ripple: on ripples where mOFC is high
before, mPFC should be high after.

THE STATISTIC, per session with both ROIs:

    asymmetry = z( rho(mOFC pre, mPFC post) ) - z( rho(mPFC pre, mOFC post) )

rho = Spearman across ripples, z = Fisher. pre = [-250, 0), post = (0, +250] ms
of the flank-normalised HFB epoch, exactly the windows of the direction index.
Positive = mOFC-before predicts mPFC-after better than the reverse.

WHY THE REVERSE DIRECTION IS SUBTRACTED. Slow shared state (arousal, task
engagement) correlates the two ROIs at every lag in both directions; it cancels
in the difference.

WHY A SHIFTED NULL. A general mOFC -> mPFC lead that has nothing to do with
ripples would also give asymmetry > 0. The same statistic is computed at the
shifted times of the conditions run (same 6 shifts, 5-120 s either sign, seed
42), with the ripple-set label from the REAL ripple, and subtracted:
estimate = real - mean(shifted). Group test: sign-flip over sessions, the same
function as the direction index.

PRE-SPECIFIED. Primary = mOFC -> mPFC, FIRST TRAVERSAL ripples (SK's hypothesis,
stated before this was run): reported uncorrected. Everything else -- other
ripple sets, lOFC -> mPFC -- is exploratory, BH over those.

ROI EPOCH. Per ripple, the mean over that ROI's different-shaft derivations
that are artifact-free for the whole +-1 s window; ripples with no clean
derivation in either ROI are dropped. Ripples are the session's hippocampal
ripples pooled over HC derivations and de-duplicated, as in the conditions run,
so both ROIs are locked to the same ripples. ROIs: 2026-09-28 partition.

Usage:
    conda activate env_multiple_clocks
    python scripts/exploratory/swr_hfb_ripple_coupling.py run
    python scripts/exploratory/swr_hfb_ripple_coupling.py run --show=False

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import pickle
import datetime

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))   # repo root
import mc.analyse.swr_io as swr_io
import mc.analyse.swr_bundle as sb
import mc.analyse.swr_windows as win
import mc.analyse.swr_behaviour as swb
import mc.plotting.ripple_figures as rfig

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))    # scripts/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # exploratory/
from swr_ripple_hfb_conditions import (session_events, assign, _clean_mask,
                                       HALF_S, NONPERI_S, POST_S,
                                       SHIFT_RANGE_S)
from swr_hfb_roi_partition import _bh
from swr_hfb_direction import (_phase_all, sign_flip, label_perm, WIN_MS,
                               PARTITION, ROI_COL)

try:
    import fire
except ImportError:
    fire = None

ANALYSIS_NAME = "hfb_ripple_coupling"
SEED, N_SHIFTS = 42, 6          # identical to the conditions run
MIN_RIPPLES = 20                # per session x set, for a correlation
BIN_MS, MAP_HALF_MS = 50.0, 500.0
PAIRS = (("mOFC", "mPFC"), ("lOFC", "mPFC"))    # (sender, receiver)
PRIMARY = ("mOFC", "mPFC", "first_traversal")

# None = every ripple of the session, labelled or not
SETS = [("all", "all ripples", None),
        ("first_traversal", "first traversal", _phase_all("first_traversal")),
        ("encoding", "encoding", _phase_all("encoding")),
        ("after_encoding", "after encoding", _phase_all("after_encoding")),
        ("ft_reward", "correct uncovers (FT)", ("reward_explore",)),
        ("ft_error", "incorrect uncovers (FT)", ("error_explore",)),
        ("ft_move", "navigation presses (FT)", ("move_explore",))]


# ── per-ripple epochs and features ───────────────────────────────────────
def roi_sums(H, idx, masks, pids, ci, off, n):
    """Per event: the summed epoch over the ROI's derivations that are clean
    for the whole window, and how many there were (0 = none, or the window
    leaves the recording). Sums, so ROIs can be combined before dividing."""
    # first and last sample of the window must both be inside the recording
    # (until 2026-10-08 this read `ci - off[0] >= 0`, which is always true, so
    # a window running off the START wrapped round to the end of the array)
    inside = (ci + off[0] >= 0) & (ci + off[-1] < n)
    wi = np.where(inside, ci, -off[0])[:, None] + off[None, :]
    s = np.zeros(wi.shape)
    k = np.zeros(len(ci))
    for p in pids:
        ok = masks[p][wi].all(1) & inside
        s[ok] += H[idx[p]][wi[ok]]
        k += ok
    return s, k


def roi_epochs(H, idx, masks, pids, ci, off, n):
    """Per-event ROI epoch: mean over the ROI's derivations that are clean
    for the whole window; a NaN row where none is."""
    s, k = roi_sums(H, idx, masks, pids, ci, off, n)
    X = np.full(s.shape, np.nan)
    X[k > 0] = s[k > 0] / k[k > 0, None]
    return X


def features(X, t_ms):
    """Flank-normalised epoch -> [pre, post, 50 ms bins over +-500 ms]."""
    flank = (np.abs(t_ms) >= NONPERI_S[0] * 1000) & \
            (np.abs(t_ms) < NONPERI_S[1] * 1000)
    X = X - X[:, flank].mean(1, keepdims=True)
    pre = X[:, (t_ms >= -WIN_MS) & (t_ms < 0)].mean(1)
    post = X[:, (t_ms > 0) & (t_ms <= WIN_MS)].mean(1)
    edges = np.arange(-MAP_HALF_MS, MAP_HALF_MS + BIN_MS, BIN_MS)
    bins = [X[:, (t_ms >= a) & (t_ms < b)].mean(1)
            for a, b in zip(edges[:-1], edges[1:])]
    return np.column_stack([pre, post] + bins)


def coupling(FA, FB):
    """Fisher-z Spearman matrix, sender features x receiver features."""
    def rz(F):
        R = stats.rankdata(F, axis=0)
        return (R - R.mean(0)) / R.std(0)
    rho = rz(FA).T @ rz(FB) / len(FA)
    return np.arctanh(np.clip(rho, -0.999999, 0.999999))


def asymmetry(M):
    """z(sender pre, receiver post) - z(receiver pre, sender post)."""
    return M[0, 1] - M[1, 0], M[0, 1], M[1, 0]


# ── loading, shared by `run` and `timeline` ──────────────────────────────
def load_inputs(b_dir, partition_dir):
    """Ripples, clean intervals and the cortical derivations exactly as the
    conditions run takes them (not excluded, different shaft from the
    hippocampal derivation), each labelled with its partition ROI."""
    with open(os.path.join(b_dir, "swr_bundle.pkl"), "rb") as f:
        B = pickle.load(f)
    rip, hfb_pairs, hiv, pairs = (B["ripples"], B["hfb_pairs"],
                                  B["hfb_intervals"], B["pairs"])
    hp = hfb_pairs[~hfb_pairs.excluded.fillna(False)].copy()
    key = pairs[["session", "pair_id", "same_probe_as_hpc"]].drop_duplicates()
    hp = hp.merge(key, on=["session", "pair_id"], how="left")
    hp = hp[~hp.same_probe_as_hpc.fillna(False).astype(bool)]
    A = pd.read_csv(os.path.join(partition_dir, "derivation_assignment.csv"))
    part = A.set_index(["session", "cx_pair"])["roi_part"]
    hp["roi"] = pd.MultiIndex.from_arrays([hp.session, hp.pair_id]).map(part)
    return rip, hiv, hp


def session_data(sess, b_dir, hp, hiv, rois):
    """One session's HFB, its derivation rows for `rois`, and clean masks."""
    g = hp[(hp.session == sess) & hp.roi.isin(rois)]
    D = sb.load_hfb(b_dir, sess, arrays=["hfb"])
    H, ids = D["hfb"], D["pair_ids"]
    fs, n = D["out_fs"], D["hfb"].shape[1]
    idx = {p: i for i, p in enumerate(ids)}
    g = g[g.pair_id.isin(idx)]
    masks = {p: _clean_mask(hiv[(hiv.session == sess) & (hiv.pair_id == p)]
                            [["start_s", "stop_s"]].to_numpy(float), n, fs)
             for p in g.pair_id}
    return H, idx, masks, g, fs, n


def session_ripples(rip, sess):
    """The session's ripple peaks, pooled over HC derivations, de-duplicated."""
    t_all = rip.loc[rip.session == sess, "t_peak_s"].to_numpy(float)
    return np.unique(np.round(t_all, 3))


# ── run ──────────────────────────────────────────────────────────────────
def run(bundle=None, partition_dir=None, out_dir=None, show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")

    R = swr_io.get_data_root()
    G = os.path.join(swr_io.derivatives_dir(R), "group", "swr")
    b_dir = bundle or os.path.join(G, "bundle_v2")
    partition_dir = partition_dir or os.path.join(G, PARTITION)
    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(G, f"{ANALYSIS_NAME}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    rip, hiv, hp = load_inputs(b_dir, partition_dir)

    rng = np.random.default_rng(SEED)
    lo, hi = SHIFT_RANGE_S
    shifts = [0.0] + [float(x) for x in rng.uniform(lo, hi, N_SHIFTS)
                      * rng.choice([-1, 1], N_SHIFTS)]
    rois = sorted({r for p in PAIRS for r in p})
    has = hp.groupby("session").roi.agg(set)
    sessions = [s for s, r in has.items()
                if any(a in r and b in r for a, b in PAIRS)]
    print(f"\n  partition : {partition_dir}\n  sessions with a pair: "
          f"{len(sessions)}; shifts {['%.1f' % s for s in shifts]}")

    rows, maps = [], {}
    for sess in sessions:
        H, idx, masks, g, fs, n = session_data(sess, b_dir, hp, hiv, rois)
        w = int(HALF_S * fs)
        off = np.arange(-w, w)
        t_ms = off / fs * 1000.0

        t_all = session_ripples(rip, sess)
        cond = assign(t_all, session_events(int(sess), R), POST_S)
        subject = g.subject_label.iloc[0]

        # features per ROI per shift, computed once
        F = {}
        for sh in shifts:
            ci = np.round((t_all + sh) * fs).astype(int)
            for roi in rois:
                pids = g[g.roi == roi].pair_id.tolist()
                if pids:
                    F[(roi, sh)] = features(roi_epochs(H, idx, masks, pids, ci,
                                                       off, n), t_ms)
        for (ra, rb), (key_, _, conds) in [(p, s) for p in PAIRS for s in SETS]:
            if (ra, 0.0) not in F or (rb, 0.0) not in F:
                continue
            in_set = np.ones(len(t_all), bool) if conds is None \
                else np.isin(cond.astype(str), conds)
            out = {}
            for sh in shifts:
                FA, FB = F[(ra, sh)], F[(rb, sh)]
                ok = in_set & np.isfinite(FA).all(1) & np.isfinite(FB).all(1)
                if ok.sum() < MIN_RIPPLES:
                    break
                M = coupling(FA[ok], FB[ok])
                out[sh] = dict(M=M, n=int(ok.sum()),
                               dir_a=float((FA[ok, 1] - FA[ok, 0]).mean()),
                               dir_b=float((FB[ok, 1] - FB[ok, 0]).mean()))
            if len(out) < len(shifts):
                continue
            real = out[0.0]
            null_M = np.mean([out[s]["M"] for s in shifts[1:]], 0)
            a_r, f_r, r_r = asymmetry(real["M"])
            a_n, f_n, r_n = asymmetry(null_M)
            rows.append(dict(sender=ra, receiver=rb, set=key_, session=int(sess),
                             subject=subject, n_ripples=real["n"],
                             asym=a_r - a_n, asym_real=a_r, asym_null=a_n,
                             fwd_real=f_r, rev_real=r_r, fwd_null=f_n,
                             rev_null=r_n,
                             dir_sender=real["dir_a"]
                             - np.mean([out[s]["dir_a"] for s in shifts[1:]]),
                             dir_receiver=real["dir_b"]
                             - np.mean([out[s]["dir_b"] for s in shifts[1:]])))
            maps.setdefault((ra, rb, key_), []).append(
                (real["M"] - null_M)[2:, 2:])
        print(f"  s{int(sess):02d} done", end="\r")

    S = pd.DataFrame(rows)

    # ---- group tests ----------------------------------------------------
    grows = []
    for (ra, rb, key_), g in S.groupby(["sender", "receiver", "set"], sort=False):
        v = g.asym.to_numpy()
        r = dict(sender=ra, receiver=rb, set=key_, n_sessions=len(v),
                 n_subjects=int(g.subject.nunique()),
                 median_ripples=float(g.n_ripples.median()),
                 asym=float(v.mean()),
                 sem=float(v.std(ddof=1) / np.sqrt(len(v))),
                 asym_real=float(g.asym_real.mean()),
                 asym_null=float(g.asym_null.mean()),
                 fwd=float((g.fwd_real - g.fwd_null).mean()),
                 rev=float((g.rev_real - g.rev_null).mean()),
                 primary=(ra, rb, key_) == PRIMARY)
        if len(v) >= 5:
            r["p_perm"] = sign_flip(v)[1]
        grows.append(r)
    Gt = pd.DataFrame(grows)
    expl = ~Gt.primary & Gt.p_perm.notna()
    Gt.loc[expl, "q_bh_exploratory"] = _bh(Gt.loc[expl, "p_perm"].to_numpy())

    _report(Gt, S)

    S.to_csv(os.path.join(out_dir, "coupling_per_session.csv"), index=False)
    Gt.to_csv(os.path.join(out_dir, "coupling_group.csv"), index=False)
    edges = np.arange(-MAP_HALF_MS, MAP_HALF_MS + BIN_MS, BIN_MS)
    np.savez_compressed(os.path.join(out_dir, "coupling_maps.npz"),
                        bin_edges_ms=edges,
                        **{f"{a}_{b}_{k}": np.stack(v)
                           for (a, b, k), v in maps.items()})
    settings = dict(analysis=ANALYSIS_NAME, created=stamp,
                    script=os.path.relpath(__file__), bundle=b_dir,
                    partition_dir=partition_dir,
                    statistic="z(rho(sender pre, receiver post)) - "
                              "z(rho(receiver pre, sender post)), Spearman "
                              "across ripples, Fisher z; real - mean(shifted)",
                    pre_post_ms=[[-WIN_MS, 0], [0, WIN_MS]],
                    flank_s=list(NONPERI_S), half_s=HALF_S,
                    shifts_s=shifts, seed=SEED, min_ripples=MIN_RIPPLES,
                    map_bin_ms=BIN_MS, map_half_ms=MAP_HALF_MS,
                    pairs=[list(p) for p in PAIRS],
                    sets={k: (None if c is None else list(c))
                          for k, _, c in SETS},
                    primary=list(PRIMARY),
                    multiple_comparisons="primary uncorrected (pre-specified);"
                                         " BH over all other cells",
                    roi_epoch="mean over the ROI's different-shaft derivations"
                              " clean for the whole +-1 s window",
                    unit="session")
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"settings": settings, "group": Gt.to_dict("records")},
                  f, indent=2, default=str)
    _figure(S, Gt, maps, out_dir)
    print(f"\n  saved -> {out_dir}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()


def _report(Gt, S):
    print(f"\n{'=' * 96}\nRipple-by-ripple coupling asymmetry, real − shifted "
          f"(+ = sender-before predicts receiver-after)\n{'=' * 96}")
    print(f"  {'pair':<14s}{'set':<18s}{'n':>4s}{'subj':>5s}{'rip':>6s}"
          f"{'asym':>9s}{'sem':>8s}{'fwd':>8s}{'rev':>8s}{'p':>8s}{'q':>7s}")
    for _, r in Gt.iterrows():
        q = r.get("q_bh_exploratory", np.nan)
        print(f"  {r.sender + '→' + r.receiver:<14s}{r.set:<18s}"
              f"{r.n_sessions:>4d}{r.n_subjects:>5d}{r.median_ripples:>6.0f}"
              f"{r.asym:>+9.4f}{r['sem']:>8.4f}{r.fwd:>+8.4f}{r.rev:>+8.4f}"
              f"{r.get('p_perm', np.nan):>8.3f}"
              + (" PRIMARY" if r.primary else f"{q:>7.2f}"))


# ── figure (overview) ────────────────────────────────────────────────────
def _figure(S, Gt, maps, out_dir):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)        # dot jitter only
    edges = np.arange(-MAP_HALF_MS, MAP_HALF_MS + BIN_MS, BIN_MS)
    with plt.rc_context(rfig._rc()):
        fig = plt.figure(figsize=(19 * rfig.CM, 17.5 * rfig.CM))
        outer = fig.add_gridspec(2, 1, height_ratios=[1, 0.95], hspace=0.62)
        top = outer[0].subgridspec(1, 2, width_ratios=[2.6, 1], wspace=0.45)
        bot = outer[1].subgridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.75)

        # A: the asymmetry per ripple set, both pairs
        ax = fig.add_subplot(top[0])
        off = {p: (i - 0.5) * 0.3 for i, p in enumerate(PAIRS)}
        for p in PAIRS:
            col = ROI_COL[p[0]]
            for i, (key_, _, _) in enumerate(SETS):
                v = S[(S.sender == p[0]) & (S.receiver == p[1])
                      & (S.set == key_)].asym.to_numpy()
                if not len(v):
                    continue
                x = i + off[p]
                ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=8,
                           color=col, alpha=0.55, lw=0, zorder=2)
                r = Gt[(Gt.sender == p[0]) & (Gt.receiver == p[1])
                       & (Gt.set == key_)].iloc[0]
                ax.errorbar(x, r.asym, yerr=r["sem"], fmt="o", ms=5,
                            mfc=col, mec="#1a1a1a", color="#1a1a1a", lw=1.1,
                            capsize=2, zorder=4,
                            label=f"{p[0]} → {p[1]}" if i == 0 else None)
                # one row of p-values per pair above the axes, so they never
                # collide with each other or with the dots
                if np.isfinite(r.get("p_perm", np.nan)):
                    ax.text(i, 1.02 + 0.07 * (1 - PAIRS.index(p)),
                            f"{r.p_perm:.2f}",
                            transform=ax.get_xaxis_transform(), ha="center",
                            va="bottom", fontsize=rfig.FS_TICK - 2,
                            fontweight="bold" if r.primary else "normal",
                            color=col)
        ax.axhline(0, color="#3d3d3d", lw=0.7, ls="--", zorder=1)
        # 2nd-98th percentile limits: a few sessions run off-panel, every one
        # is in the mean
        lo, hi = np.percentile(S.asym, [2, 98])
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        ax.set_xlim(-0.5, len(SETS) - 0.5)
        ax.set_xticks(range(len(SETS)))
        ax.set_xticklabels([t.replace(" (", "\n(") for _, t, _ in SETS],
                           rotation=40, ha="right", fontsize=rfig.FS_TICK - 1)
        ax.set_ylabel("coupling asymmetry (Δz)\nreal − shifted",
                      fontsize=rfig.FS_LABEL)
        ax.set_title("Does sender activity BEFORE a ripple predict receiver "
                     "activity AFTER it?\np = sign-flip over sessions, one row "
                     "per pair; bold = pre-specified primary",
                     fontsize=rfig.FS_TICK, pad=24)
        ax.legend(frameon=False, fontsize=rfig.FS_TICK - 1, loc="lower left",
                  handletextpad=0.2)

        # B: the components of the primary cell, session by session
        ax = fig.add_subplot(top[1])
        ra, rb, key_ = PRIMARY
        g = S[(S.sender == ra) & (S.receiver == rb) & (S.set == key_)]
        fwd = (g.fwd_real - g.fwd_null).to_numpy()
        rev = (g.rev_real - g.rev_null).to_numpy()
        for a_, b_ in zip(fwd, rev):
            ax.plot([0, 1], [a_, b_], color="#9a9a9a", lw=0.7, zorder=1)
        ax.scatter(np.zeros(len(fwd)), fwd, s=12, color=ROI_COL[ra], zorder=2)
        ax.scatter(np.ones(len(rev)), rev, s=12, color=ROI_COL[rb], zorder=2)
        for x, v in ((0, fwd), (1, rev)):
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)),
                        fmt="_", ms=14, color="#1a1a1a", lw=1.4, capsize=3,
                        zorder=3)
        ax.axhline(0, color="#3d3d3d", lw=0.7, ls="--")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{ra}\n→ {rb}", f"{rb}\n→ {ra}"],
                           fontsize=rfig.FS_TICK - 1)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel("coupling, A pre → B post\n(Fisher z, real − shifted)",
                      fontsize=rfig.FS_TICK)
        ax.set_title(f"components, {key_.replace('_', ' ')}\n"
                     f"(n = {len(g)} sessions)", fontsize=rfig.FS_TICK)

        # C, D: cross-temporal coupling map and its asymmetry, primary cell
        M = np.stack(maps[PRIMARY]).mean(0)
        for j, (Z, title, lab) in enumerate((
                (M, f"coupling map, {key_.replace('_', ' ')}\n(real − shifted)",
                 "Δz"),
                (M - M.T, f"asymmetry: map − transpose\n(> 0 above diagonal = "
                          f"{ra} leads)", "Δz"))):
            ax = fig.add_subplot(bot[j])
            lim = np.nanmax(np.abs(Z))
            im = ax.pcolormesh(edges, edges, Z, cmap="RdYlBu_r", vmin=-lim,
                               vmax=lim, shading="flat")
            ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]],
                    color="#1a1a1a", lw=0.6, ls=":")
            ax.axhline(0, color="#1a1a1a", lw=0.5)
            ax.axvline(0, color="#1a1a1a", lw=0.5)
            ax.set_aspect("equal")
            ax.set_xticks([-500, 0, 500])
            ax.set_yticks([-500, 0, 500])
            ax.set_xlabel(f"{rb} time from ripple (ms)", fontsize=rfig.FS_TICK)
            if j == 0:       # both maps share the axes; one label is enough
                ax.set_ylabel(f"{ra} time from ripple (ms)",
                              fontsize=rfig.FS_TICK)
            ax.set_title(title, fontsize=rfig.FS_TICK)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05, shrink=0.8)
            cb.set_label(lab, fontsize=rfig.FS_TICK - 2)
            cb.ax.tick_params(labelsize=rfig.FS_TICK - 2)
        ax = fig.add_subplot(bot[2])
        ax.axis("off")
        ax.text(0.0, 0.5,
                "Asymmetry = z(ρ(sender pre, receiver post))\n"
                "− z(ρ(receiver pre, sender post)), Spearman\n"
                "across ripples within a session; pre/post =\n"
                "[−250, 0) / (0, +250] ms of the flank-\n"
                "normalised HFB epoch. Shifted null: the\n"
                "same statistic at 6 shifted times (5–120 s),\n"
                "subtracted. Dots: sessions; marker: mean ±\n"
                "SEM. Maps: mean over sessions, 50 ms bins;\n"
                "above the dotted diagonal the sender's bin\n"
                "is EARLIER than the receiver's.\n"
                "Primary (pre-specified): mOFC → mPFC,\n"
                "first traversal, uncorrected. Others are\n"
                "exploratory (BH q in coupling_group.csv).",
                ha="left", va="center", fontsize=rfig.FS_TICK - 2,
                color="#3d3d3d", transform=ax.transAxes)
        fig.suptitle("Ripple-by-ripple coupling between orbitofrontal cortex "
                     "and mPFC", fontsize=rfig.FS_TITLE, y=0.995)
        stem = os.path.join(out_dir, "ripple_coupling")
        fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(stem + ".pdf", bbox_inches="tight")


# ── timeline: the single trials behind it, in two alignments ─────────────
# Same rows, same baseline, two clocks. Ripple-locked rows are sorted by how
# long before the ripple its event happened; event-locked rows by how long
# after the event its ripple came. If an ROI follows the EVENT, its activity
# runs along the event marker in the ripple-locked image; if it follows the
# RIPPLE, it sits at 0 there and along the ripple marker in the other.
TL_SETS = (("ft_all", "first traversal, every event",
            ("reward_explore", "error_explore", "move_explore")),
           ("ft_reward", "first traversal, correct uncovers",
            ("reward_explore",)))
TL_ROIS = ("mOFC", "mPFC")
RIP_WIN_S = (-2.75, 1.0)       # ripple-locked: room for an event up to -2 s
EV_WIN_S = (-1.0, 2.75)        # event-locked: room for a ripple up to +2 s
BASE_S = (-0.75, -0.25)        # per row, relative to the EVENT, both views
LAT_BINS_S = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0))   # event -> ripple

# ── the chain, link by link (first traversal) ────────────────────────────
#   link 1  mOFC response to the uncover (0-500 ms) -> does a ripple follow?
#   link 2  ripple -> mPFC HFB in the 250 ms after it, against the same
#           moment after an uncover that had NO ripple
#   chain   mOFC uncover response -> mPFC after the ripple, event by event
# A ripple "follows" an event if the first ripple after it comes within 2 s,
# whatever happens in between. Only ripples 0.5-2 s after the event count, so
# the ripple's own HFB rise cannot sit inside the 0-500 ms response window;
# events whose first ripple comes earlier are left out of the tests.
# Pre-specified (SK, before any of this was run): link 1 = mOFC, link 2 =
# mPFC, both on correct uncovers. Every other cell is exploratory (BH).
RESP_S = (0.0, 0.5)
FOLLOW_S = (0.5, 2.0)
POSTRIP_S = 0.25
CHAIN_CONDS = (("reward_explore", "correct uncovers"),
               ("error_explore", "incorrect uncovers"),
               ("move_explore", "navigation presses"))
LINK_PRIMARY = {"link1": ("mOFC", "reward_explore"),
                "link2": ("mPFC", "reward_explore")}


def preceding_event(ripple_t, ev_t, post_s=POST_S):
    """Index of each ripple's nearest preceding event within post_s, -1 if
    none -- the same rule as `assign`, which only returns the label."""
    j = np.searchsorted(ev_t, ripple_t, side="right") - 1
    jj = np.clip(j, 0, len(ev_t) - 1)
    ok = (j >= 0) & ((ripple_t - ev_t[jj]) <= post_s)
    return np.where(ok, j, -1)


def _epochs(H, idx, masks, pids, t_s, win_s, fs, n):
    off = np.arange(int(round(win_s[0] * fs)), int(round(win_s[1] * fs)))
    ci = np.round(np.asarray(t_s) * fs).astype(int)
    return roi_epochs(H, idx, masks, pids, ci, off, n), off / fs


def timeline(bundle=None, partition_dir=None, out_dir=None, show=True,
             reuse=False):
    """`reuse=True` reads the extracted rows back from `out_dir` instead of
    re-extracting (~6 min) -- for redrawing only."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")

    R = swr_io.get_data_root()
    G = os.path.join(swr_io.derivatives_dir(R), "group", "swr")
    b_dir = bundle or os.path.join(G, "bundle_v2")
    partition_dir = partition_dir or os.path.join(G, PARTITION)
    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(G, f"{ANALYSIS_NAME}_timeline_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "timeline_rows.pkl")
    if reuse and os.path.isfile(cache):
        with open(cache, "rb") as f:
            rows_r, rows_e, t_r, t_e = pickle.load(f)
        sessions = []
    else:
        rip, hiv, hp = load_inputs(b_dir, partition_dir)
        sessions = sorted(hp[hp.roi.isin(TL_ROIS)].session.unique())
        rows_r, rows_e = [], []      # one dict per ripple / per event
    for sess in sessions:
        H, idx, masks, g, fs, n = session_data(sess, b_dir, hp, hiv, TL_ROIS)
        ev = session_events(int(sess), R)
        if not len(ev):
            continue
        ev_t, ev_c = ev.t_s.to_numpy(float), ev.cond.to_numpy(object)
        t_all = session_ripples(rip, sess)
        j = preceding_event(t_all, ev_t)
        # the labels must be exactly the conditions run's
        lab = np.where(j >= 0, ev_c[np.clip(j, 0, None)], None)
        assert (lab == assign(t_all, ev, POST_S)).all()
        k_ = np.searchsorted(t_all, ev_t, side="right")
        first_after = np.full(len(ev_t), np.nan)
        has = k_ < len(t_all)
        first_after[has] = t_all[k_[has]] - ev_t[has]
        first_after[first_after > POST_S] = np.nan
        for roi in TL_ROIS:
            pids = g[g.roi == roi].pair_id.tolist()
            if not pids:
                continue
            for key_, _, conds in TL_SETS:
                # ripple-locked: every ripple whose event is in the set
                rsel = np.flatnonzero(np.isin(lab.astype(str), conds))
                lat = t_all[rsel] - ev_t[j[rsel]]
                X, t_r = _epochs(H, idx, masks, pids, t_all[rsel], RIP_WIN_S,
                                 fs, n)
                for x, l in zip(X, lat):
                    if np.isfinite(x).all():
                        b = (t_r >= -l + BASE_S[0]) & (t_r < -l + BASE_S[1])
                        rows_r.append(dict(roi=roi, set=key_, session=int(sess),
                                           lat=float(l), x=x - x[b].mean()))
                # event-locked: every event in the set, with the first ripple
                # after it within 2 s (whatever happens in between)
                esel = np.flatnonzero(np.isin(ev_c.astype(str), conds))
                X, t_e = _epochs(H, idx, masks, pids, ev_t[esel], EV_WIN_S,
                                 fs, n)
                b = (t_e >= BASE_S[0]) & (t_e < BASE_S[1])
                for x, e_i, l in zip(X, esel, first_after[esel]):
                    if np.isfinite(x).all():
                        rows_e.append(dict(roi=roi, set=key_, session=int(sess),
                                           cond=ev_c[e_i], ev_i=int(e_i),
                                           lat=float(l), x=x - x[b].mean()))
        print(f"  s{int(sess):02d} done", end="\r")
    if sessions:
        with open(cache, "wb") as f:
            pickle.dump((rows_r, rows_e, t_r, t_e), f)

    Rr, Re = pd.DataFrame(rows_r), pd.DataFrame(rows_e)

    # ---- session-level mean traces (unit = session) ----------------------
    def sess_means(D, groups):
        out = []
        for (roi, key_, gname), sel in groups(D):
            for sess, gs_ in D[sel].groupby("session"):
                if len(gs_) >= 3:
                    out.append(dict(roi=roi, set=key_, group=gname,
                                    session=sess, n=len(gs_),
                                    x=np.stack(gs_.x).mean(0)))
        return pd.DataFrame(out)

    def rip_groups(D):
        for roi, key_ in D[["roi", "set"]].drop_duplicates().itertuples(False):
            base = (D.roi == roi) & (D.set == key_)
            for a, b in LAT_BINS_S:
                yield (roi, key_, f"{a:.1f}-{b:.1f} s"), \
                    base & (D.lat > a) & (D.lat <= b)

    def ev_groups(D):
        for roi, key_ in D[["roi", "set"]].drop_duplicates().itertuples(False):
            base = (D.roi == roi) & (D.set == key_)
            yield (roi, key_, "ripple follows"), base & D.lat.notna()
            yield (roi, key_, "no ripple"), base & D.lat.isna()

    Mr, Me = sess_means(Rr, rip_groups), sess_means(Re, ev_groups)

    summary = []
    for (roi, key_), g_ in Re.groupby(["roi", "set"]):
        gr = Rr[(Rr.roi == roi) & (Rr.set == key_)]
        summary.append(dict(roi=roi, set=key_, n_sessions=int(g_.session.nunique()),
                            n_events=len(g_), n_ripples=len(gr),
                            frac_events_with_ripple=float(g_.lat.notna().mean()),
                            median_latency_s=float(gr.lat.median())))
    Sm = pd.DataFrame(summary)
    print("\n" + Sm.to_string(index=False))

    np.savez_compressed(
        os.path.join(out_dir, "timeline_traces.npz"),
        t_ripple_s=t_r, t_event_s=t_e,
        **{f"ripple_{r.roi}_{r.set}_{r.group}_s{r.session}": r.x
           for r in Mr.itertuples()},
        **{f"event_{r.roi}_{r.set}_{r.group}_s{r.session}": r.x
           for r in Me.itertuples()})
    Sm.to_csv(os.path.join(out_dir, "timeline_counts.csv"), index=False)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"settings": dict(
                       analysis=f"{ANALYSIS_NAME}_timeline", created=stamp,
                       script=os.path.relpath(__file__), bundle=b_dir,
                       partition_dir=partition_dir,
                       sets={k: list(c) for k, _, c in TL_SETS},
                       rois=list(TL_ROIS), ripple_window_s=list(RIP_WIN_S),
                       event_window_s=list(EV_WIN_S),
                       baseline_s_rel_event=list(BASE_S),
                       latency_bins_s=[list(b) for b in LAT_BINS_S],
                       hfb="raw z-scored HFB, NO shifted null; per-row "
                           "pre-event baseline; ROI epoch = mean of the "
                           "derivations clean for the whole window",
                       ripple_event_rule="ripple-locked view: ripple belongs "
                                         "to its nearest preceding event "
                                         "within 2 s (= conditions run); "
                                         "event-locked view: first ripple "
                                         "after the event within 2 s",
                       unit_for_traces="session (>= 3 rows per group)"),
                   "counts": Sm.to_dict("records")}, f, indent=2, default=str)

    for key_, title, _ in TL_SETS:
        _figure_timeline(Rr, Re, Mr, Me, t_r, t_e, key_, title, out_dir)

    # ---- the chain tests ------------------------------------------------
    L, CH, tr_ = chain_tests(Re[Re.set == "ft_all"], t_e)
    L.to_csv(os.path.join(out_dir, "chain_per_session.csv"), index=False)
    CH.to_csv(os.path.join(out_dir, "chain_group.csv"), index=False)
    _report_chain(CH)
    with open(os.path.join(out_dir, "chain_result.json"), "w") as f:
        json.dump({"settings": dict(
                       response_window_s=list(RESP_S),
                       ripple_follows_s=list(FOLLOW_S),
                       post_ripple_s=POSTRIP_S, baseline_s=list(BASE_S),
                       primary={k: list(v) for k, v in LINK_PRIMARY.items()},
                       link1="per session AUC(response | ripple 0.5-2 s vs no "
                             "ripple within 2 s) - 0.5; sign-flip over sessions",
                       link2="per session mean HFB in (lat, lat+0.25] s, "
                             "ripple events minus no-ripple events at a "
                             "pseudo-latency drawn from the session's ripple "
                             "latencies (seed 42 + session); the same in "
                             "[lat-0.25, lat) reported as the pre component",
                       chain="per session Spearman (Fisher z) of mOFC 0-0.5 s "
                             "response with mPFC (lat, lat+0.25], ripple "
                             "events minus pseudo-latency no-ripple events",
                       multiple_comparisons="primaries uncorrected; BH over "
                                            "all other cells"),
                   "group": CH.to_dict("records")}, f, indent=2, default=str)
    _figure_chain(L, CH, tr_, Re[Re.set == "ft_all"], t_e, out_dir)
    print(f"\n  saved -> {out_dir}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()


def chain_tests(E, t_e):
    """Link 1, link 2 and the chain, per session, then over sessions."""
    resp_w = (t_e >= RESP_S[0]) & (t_e < RESP_S[1])
    rows, traces = [], []
    for (roi, cond, sess), g in E.groupby(["roi", "cond", "session"]):
        lat = g.lat.to_numpy()
        X = np.stack(g.x)
        rip_ = (lat > FOLLOW_S[0]) & (lat <= FOLLOW_S[1])
        nor = np.isnan(lat)
        resp = X[:, resp_w].mean(1)
        r = dict(roi=roi, cond=cond, session=sess, n_ripple=int(rip_.sum()),
                 n_none=int(nor.sum()), n_early=int((lat <= FOLLOW_S[0]).sum()))
        if rip_.sum() >= 3 and nor.sum() >= 3:
            u = stats.mannwhitneyu(resp[rip_], resp[nor]).statistic
            r["link1_auc"] = float(u / (rip_.sum() * nor.sum()) - 0.5)
            r["link1_diff"] = float(resp[rip_].mean() - resp[nor].mean())
            # link 2: the no-ripple events get a pseudo-latency drawn from
            # this session's ripple latencies, so both sides are read at the
            # same distance from the event
            rng = np.random.default_rng(SEED + int(sess))
            plat = rng.choice(lat[rip_], nor.sum(), replace=True)
            L_ = np.r_[lat[rip_], plat]
            XX = np.r_[X[rip_], X[nor]]
            is_r = np.r_[np.ones(rip_.sum(), bool), np.zeros(nor.sum(), bool)]
            post = np.array([x[(t_e > l) & (t_e <= l + POSTRIP_S)].mean()
                             for x, l in zip(XX, L_)])
            pre = np.array([x[(t_e >= l - POSTRIP_S) & (t_e < l)].mean()
                            for x, l in zip(XX, L_)])
            r["link2_post"] = float(post[is_r].mean() - post[~is_r].mean())
            r["link2_pre"] = float(pre[is_r].mean() - pre[~is_r].mean())
            # the trace around the (pseudo-)ripple, for the figure
            k = int(round(0.5 / (t_e[1] - t_e[0])))
            for flag, name in ((True, "ripple"), (False, "no ripple")):
                seg = [x[np.searchsorted(t_e, l) - k:np.searchsorted(t_e, l) + k]
                       for x, l in zip(XX[is_r == flag], L_[is_r == flag])]
                traces.append(dict(roi=roi, cond=cond, session=sess,
                                   group=name, x=np.mean(seg, 0)))
            r["_post"], r["_resp"], r["_isr"], r["_ev"] = \
                post, np.r_[resp[rip_], resp[nor]], is_r, \
                np.r_[g.ev_i.to_numpy()[rip_], g.ev_i.to_numpy()[nor]]
        rows.append(r)
    L = pd.DataFrame(rows)

    # chain: mOFC response x mPFC after the ripple, same events
    crow = []
    for cond, _ in CHAIN_CONDS:
        for sess in sorted(set(L.session)):
            a = L[(L.roi == "mOFC") & (L.cond == cond) & (L.session == sess)]
            b = L[(L.roi == "mPFC") & (L.cond == cond) & (L.session == sess)]
            if not len(a) or not len(b) or "_ev" not in a or \
                    not isinstance(a._ev.iloc[0], np.ndarray) or \
                    not isinstance(b._ev.iloc[0], np.ndarray):
                continue
            a, b = a.iloc[0], b.iloc[0]
            out = {}
            for flag, name in ((True, "ripple"), (False, "none")):
                ea = dict(zip(a._ev[a._isr == flag], a._resp[a._isr == flag]))
                eb = dict(zip(b._ev[b._isr == flag], b._post[b._isr == flag]))
                sh = sorted(set(ea) & set(eb))
                if len(sh) < 8:
                    break
                rho = stats.spearmanr([ea[e] for e in sh],
                                      [eb[e] for e in sh])[0]
                out[name] = (np.arctanh(np.clip(rho, -0.999999, 0.999999)),
                             len(sh))
            if len(out) == 2:
                crow.append(dict(roi="mOFC→mPFC", cond=cond, session=sess,
                                 chain_z=out["ripple"][0] - out["none"][0],
                                 chain_z_ripple=out["ripple"][0],
                                 chain_z_none=out["none"][0],
                                 n_events_ripple=out["ripple"][1]))
    C = pd.DataFrame(crow)
    L = L.drop(columns=[c for c in L.columns if c.startswith("_")])

    grows = []
    for test, col, D in (("link1", "link1_auc", L), ("link2", "link2_post", L),
                         ("link2_pre", "link2_pre", L),
                         ("chain", "chain_z", C)):
        if not len(D) or col not in D:
            continue
        for (roi, cond), g in D.dropna(subset=[col]).groupby(["roi", "cond"]):
            v = g[col].to_numpy()
            r = dict(test=test, roi=roi, cond=cond, n_sessions=len(v),
                     mean=float(v.mean()),
                     sem=float(v.std(ddof=1) / np.sqrt(len(v)))
                     if len(v) > 1 else np.nan,
                     primary=LINK_PRIMARY.get(test) == (roi, cond))
            if len(v) >= 5:
                r["p_perm"] = sign_flip(v)[1]
            grows.append(r)
    CH = pd.DataFrame(grows)
    ex = ~CH.primary & CH.p_perm.notna() & (CH.test != "link2_pre")
    CH.loc[ex, "q_bh_exploratory"] = _bh(CH.loc[ex, "p_perm"].to_numpy())
    L = L.merge(C, on=["roi", "cond", "session"], how="outer") if len(C) else L
    return L, CH, pd.DataFrame(traces)


def _report_chain(CH):
    print(f"\n{'=' * 90}\nTHE CHAIN, link by link (first traversal; "
          f"sign-flip over sessions)\n{'=' * 90}")
    for _, r in CH.iterrows():
        q = r.get("q_bh_exploratory", np.nan)
        tag = "PRIMARY" if r.primary else ("" if pd.isna(q) else f"q={q:.2f}")
        print(f"  {r.test:<10s}{r.roi:<11s}{r.cond:<16s}n={r.n_sessions:>3d}"
              f"  {r['mean']:+.4f} ± {r['sem']:.4f}"
              f"  p={r.get('p_perm', np.nan):.3f}  {tag}")


def _dots(ax, D, col, CH, test, rng, rois, ylab):
    """Per-session values per condition, one offset per ROI, p above."""
    off = {r: (i - (len(rois) - 1) / 2) * 0.28 for i, r in enumerate(rois)}
    for i, (cond, _) in enumerate(CHAIN_CONDS):
        for j, roi in enumerate(rois):
            v = D[(D.roi == roi) & (D.cond == cond)][col].dropna().to_numpy()
            if not len(v):
                continue
            c = ROI_COL.get(roi.split("→")[0], "#555555")
            x = i + off[roi]
            ax.scatter(x + rng.uniform(-0.06, 0.06, len(v)), v, s=7, color=c,
                       alpha=0.55, lw=0, zorder=2)
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v))
                        if len(v) > 1 else 0, fmt="o", ms=4.5, mfc=c,
                        mec="#1a1a1a", color="#1a1a1a", lw=1.0, capsize=2,
                        zorder=4)
            r = CH[(CH.test == test) & (CH.roi == roi) & (CH.cond == cond)]
            if len(r) and np.isfinite(r.iloc[0].get("p_perm", np.nan)):
                ax.text(i, 1.02 + 0.08 * (len(rois) - 1 - j),
                        f"{r.iloc[0].p_perm:.2f}",
                        transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=rfig.FS_TICK - 2.5, color=c,
                        fontweight="bold" if r.iloc[0].primary else "normal")
    ax.axhline(0, color="#3d3d3d", lw=0.7, ls="--", zorder=1)
    ax.set_xticks(range(len(CHAIN_CONDS)))
    ax.set_xticklabels([l.replace(" ", "\n") for _, l in CHAIN_CONDS],
                       fontsize=rfig.FS_TICK - 1)
    ax.set_xlim(-0.5, len(CHAIN_CONDS) - 0.5)
    ax.set_ylabel(ylab, fontsize=rfig.FS_TICK - 1)


def _trace_pair(ax, groups, t, roi, shade=None):
    """Mean ± SEM over sessions for each (label, colour, session x trace)."""
    for lab, col, X in groups:
        if len(X) < 2:
            continue
        m, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))
        ax.fill_between(t, m - se, m + se, color=col, alpha=0.25, lw=0)
        ax.plot(t, m, color=col, lw=1.3, label=f"{lab} (n={len(X)})")
    if shade:
        ax.axvspan(*shade, color="#000000", alpha=0.07, lw=0)
    ax.axhline(0, color="#3d3d3d", lw=0.6, ls="--")
    ax.axvline(0, color="#1a1a1a", lw=0.8, ls=":")
    ax.legend(frameon=False, fontsize=rfig.FS_TICK - 3, loc="upper left",
              handlelength=1.0, labelspacing=0.15, borderpad=0.1)


def _figure_chain(L, CH, TR, E, t_e, out_dir):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)        # dot jitter only
    cond0 = "reward_explore"
    with plt.rc_context(rfig._rc()):
        fig, axes = plt.subplots(2, 3, figsize=(19 * rfig.CM, 13.5 * rfig.CM),
                                 gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
        # ---- link 1 ----
        _dots(axes[0, 0], L, "link1_auc", CH, "link1", rng, TL_ROIS,
              "AUC − 0.5\n(response | ripple vs none)")
        axes[0, 0].set_title("LINK 1: event response (0–500 ms)\npredicts a "
                             "ripple 0.5–2 s later?", fontsize=rfig.FS_TICK,
                             pad=18)
        for c, roi in enumerate(TL_ROIS):
            groups = []
            for lab, sel, col in (
                    ("ripple 0.5–2 s", lambda d: (d.lat > FOLLOW_S[0])
                     & (d.lat <= FOLLOW_S[1]), ROI_COL[roi]),
                    ("no ripple ≤ 2 s", lambda d: d.lat.isna(), "#8C8C8C")):
                d = E[(E.roi == roi) & (E.cond == cond0)]
                X = [np.stack(gs.x).mean(0) for _, gs in d[sel(d)].groupby("session")
                     if len(gs) >= 3]
                groups.append((lab, col, np.array(X)))
            ax = axes[0, 1 + c]
            _trace_pair(ax, groups, t_e, roi, shade=RESP_S)
            ax.set_xlim(-0.75, 2.25)
            ax.set_xlabel("time from correct uncover (s)",
                          fontsize=rfig.FS_TICK - 1)
            ax.set_ylabel(f"{roi} HFB (z)", fontsize=rfig.FS_TICK - 1)
            ax.set_title(f"{roi}, correct uncovers\n(shaded: response window)",
                         fontsize=rfig.FS_TICK)
        # ---- link 2 ----
        ax = axes[1, 0]
        _dots(ax, L, "link2_post", CH, "link2", rng, TL_ROIS,
              "HFB after the ripple (z)\nripple − no ripple")
        ax.set_title(f"LINK 2: HFB in the {int(POSTRIP_S * 1000)} ms after "
                     "the ripple,\nvs the same moment without one",
                     fontsize=rfig.FS_TICK, pad=18)
        ax = axes[1, 1]
        k = TR.x.iloc[0].shape[0] // 2 if len(TR) else 50
        tt = (np.arange(-k, k)) * (t_e[1] - t_e[0])
        roi = LINK_PRIMARY["link2"][0]
        groups = [(lab, col, np.stack(TR[(TR.roi == roi) & (TR.cond == cond0)
                                         & (TR.group == g_)].x))
                  for lab, g_, col in (("ripple", "ripple", ROI_COL[roi]),
                                       ("no ripple, same latency", "no ripple",
                                        "#8C8C8C"))]
        _trace_pair(ax, groups, tt, roi, shade=(0, POSTRIP_S))
        ax.set_xlabel("time from (pseudo-)ripple (s)", fontsize=rfig.FS_TICK - 1)
        ax.set_ylabel(f"{roi} HFB (z)", fontsize=rfig.FS_TICK - 1)
        ax.set_title(f"{roi}, correct uncovers\n(shaded: post-ripple window)",
                     fontsize=rfig.FS_TICK)
        # ---- the chain ----
        ax = axes[1, 2]
        if "chain_z" in L:
            _dots(ax, L, "chain_z", CH, "chain", rng, ("mOFC→mPFC",),
                  "Δ Fisher z\n(ripple − no-ripple events)")
        ax.set_title("CHAIN: mOFC response predicts\nmPFC after the ripple?",
                     fontsize=rfig.FS_TICK, pad=10)
        for a in axes.ravel():
            a.tick_params(labelsize=rfig.FS_TICK - 1)
        fig.suptitle("Reward uncover → mOFC → ripple → mPFC, first traversal",
                     fontsize=rfig.FS_TITLE)
        fig.text(0.5, 0.0,
                 "Dots: sessions; marker: mean ± SEM; p: sign-flip over "
                 "sessions, one row per ROI (bold = pre-specified primary: "
                 "link 1 mOFC, link 2 mPFC, correct uncovers).\nOnly ripples "
                 "0.5–2 s after the event count, so the ripple's own rise is "
                 "never inside the 0–500 ms response window. No-ripple events "
                 "are read at a latency drawn from\nthe session's ripple "
                 "latencies. Raw z-scored HFB, baselined −750…−250 ms before "
                 "the event. Pre-ripple components and BH q: chain_group.csv.",
                 ha="center", va="top", fontsize=rfig.FS_TICK - 2.5,
                 color="#3d3d3d")
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
        stem = os.path.join(out_dir, "chain")
        fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(stem + ".pdf", bbox_inches="tight")


def _sorted_image(ax, D, t, marker_sign, cmap, vlim, fs):
    """Rows sorted by latency; smoothed over rows and 50 ms; marker per row."""
    from scipy.ndimage import uniform_filter1d, gaussian_filter1d
    D = D.assign(_l=D.lat.fillna(np.inf)).sort_values("_l")
    X = np.stack(D.x)
    k = max(1, len(X) // 60)              # ~60 independent rows on screen
    X = uniform_filter1d(gaussian_filter1d(X, 0.05 * fs / 2.355, axis=1), k,
                         axis=0)
    im = ax.imshow(X, aspect="auto", cmap=cmap, vmin=-vlim, vmax=vlim,
                   extent=(t[0], t[-1] + 1 / fs, len(X), 0),
                   interpolation="nearest")
    lat = D.lat.to_numpy()
    y = np.arange(len(X)) + 0.5
    ok = np.isfinite(lat)
    ax.plot(marker_sign * lat[ok], y[ok], color="#1a1a1a", lw=0.9)
    if (~ok).any():
        ax.axhline(ok.sum(), color="#1a1a1a", lw=0.8, ls="--")
        ax.text(t[-1], ok.sum() + 0.5 * (~ok).sum(), "no ripple ",
                ha="right", va="center", fontsize=rfig.FS_TICK - 2,
                color="#1a1a1a")
    ax.axvline(0, color="#1a1a1a", lw=0.8, ls=":")
    return im, k


def _mean_panel(ax, M, roi, key_, groups, colours, t):
    for gname, col in zip(groups, colours):
        g = M[(M.roi == roi) & (M.set == key_) & (M.group == gname)]
        if len(g) < 2:
            continue
        X = np.stack(g.x)
        m, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))
        ax.fill_between(t, m - se, m + se, color=col, alpha=0.25, lw=0)
        ax.plot(t, m, color=col, lw=1.3, label=f"{gname} (n={len(g)})")
    ax.axhline(0, color="#3d3d3d", lw=0.6, ls="--")
    ax.axvline(0, color="#1a1a1a", lw=0.8, ls=":")
    ax.legend(frameon=False, fontsize=rfig.FS_TICK - 3, loc="upper left",
              handlelength=1.0, labelspacing=0.15, borderpad=0.1)


def _figure_timeline(Rr, Re, Mr, Me, t_r, t_e, key_, title, out_dir):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    fs = 1.0 / (t_r[1] - t_r[0])
    rr, re_ = Rr[Rr.set == key_], Re[Re.set == key_]
    vlim = np.percentile(np.abs(np.concatenate(
        [np.stack(rr.x).ravel(), np.stack(re_.x).ravel()])), 90)
    with plt.rc_context(rfig._rc()):
        fig, axes = plt.subplots(4, 2, sharex="row",
                                 figsize=(2 * 7.5 * rfig.CM, 23 * rfig.CM),
                                 gridspec_kw=dict(height_ratios=[2.2, 1, 2.2, 1]))
        for c, roi in enumerate(TL_ROIS):
            col = ROI_COL[roi]
            shades = [tuple(1 - a * (1 - np.array(to_rgb(col))))
                      for a in (0.35, 0.65, 1.0)]
            # ripple-locked
            ax = axes[0, c]
            im, k = _sorted_image(ax, rr[rr.roi == roi], t_r, -1, "RdYlBu_r",
                                  vlim, fs)
            ax.set_title(f"{roi} — locked to the RIPPLE\n"
                         f"{(rr.roi == roi).sum()} ripples, "
                         f"{rr[rr.roi == roi].session.nunique()} sessions",
                         fontsize=rfig.FS_TICK)
            ax.set_ylabel(f"ripples, sorted by\nevent → ripple latency\n"
                          f"(smoothed over {k} rows)", fontsize=rfig.FS_TICK - 1)
            ax = axes[1, c]
            _mean_panel(ax, Mr, roi, key_,
                        [f"{a:.1f}-{b:.1f} s" for a, b in LAT_BINS_S],
                        shades, t_r)
            for (a, b), sh in zip(LAT_BINS_S, shades):
                ax.axvspan(-b, -a, ymin=0.0, ymax=0.06, color=sh, lw=0)
            ax.set_xlabel("time from ripple (s)", fontsize=rfig.FS_TICK)
            ax.set_ylabel("HFB (z)\nmean ± SEM", fontsize=rfig.FS_TICK - 1)
            # event-locked
            ax = axes[2, c]
            sub = re_[re_.roi == roi]
            im, k = _sorted_image(ax, sub, t_e, 1, "RdYlBu_r", vlim, fs)
            ax.set_title(f"{roi} — locked to the EVENT\n{len(sub)} events, "
                         f"{100 * sub.lat.notna().mean():.0f}% followed by a "
                         "ripple", fontsize=rfig.FS_TICK)
            ax.set_ylabel(f"events, sorted by\nevent → ripple latency\n"
                          f"(smoothed over {k} rows)", fontsize=rfig.FS_TICK - 1)
            ax = axes[3, c]
            _mean_panel(ax, Me, roi, key_, ["ripple follows", "no ripple"],
                        [col, "#8C8C8C"], t_e)
            ax.set_xlabel("time from event (s)", fontsize=rfig.FS_TICK)
            ax.set_ylabel("HFB (z)\nmean ± SEM", fontsize=rfig.FS_TICK - 1)
        for ax in axes.ravel():
            ax.tick_params(labelsize=rfig.FS_TICK - 1)
        cax = fig.add_axes([0.92, 0.80, 0.015, 0.1])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("HFB (z)", fontsize=rfig.FS_TICK - 1)
        cb.ax.tick_params(labelsize=rfig.FS_TICK - 2)
        fig.suptitle(f"{title}: does HFB follow the event or the ripple?\n"
                     "black line = the event (top) / the first ripple "
                     "(bottom) on each row; dotted = 0", fontsize=rfig.FS_TICK)
        fig.text(0.5, 0.005,
                 "Raw z-scored HFB, no shifted null; every row baselined to "
                 "−750…−250 ms before its EVENT. Rows pooled over sessions;\n"
                 "mean traces: mean over sessions (≥ 3 rows each) ± SEM. "
                 "Ripple-locked: a ripple belongs to its nearest preceding "
                 "event within 2 s;\nevent-locked: the first ripple after the "
                 "event within 2 s. Single-trial noise (±0.5 z) is ~50× the "
                 "ripple-locked effects (~0.01 z).\n"
                 "Ripple-locked traces split by event → ripple latency (shade "
                 "bar at the bottom = where those events sit).",
                 ha="center", va="bottom", fontsize=rfig.FS_TICK - 2,
                 color="#3d3d3d")
        fig.tight_layout(rect=(0, 0.05, 0.91, 0.95))
        stem = os.path.join(out_dir, f"timeline_{key_}")
        fig.savefig(stem + ".png", dpi=250)
        fig.savefig(stem + ".pdf")


# ── chain: everything ripple-locked, every first-traversal ripple ────────
# No event anywhere. First traversal = grid onset -> the first t_D
# (`swr_windows.windows_phase3_whole`, SK's definition), and EVERY ripple inside
# it counts, labelled or not. Three links, each real - shifted null, each
# pre-specified by SK before this was run:
#   link 1  mOFC HFB is up BEFORE the ripple      (pre  [-250, 0) ms > 0)
#   link 2  mPFC HFB is up AFTER the ripple       (post (0, +250] ms > 0)
#   chain   on the same ripple, mOFC-before predicts mPFC-after better than
#           the reverse (the `run` asymmetry), sessions with both ROIs
# The chain holds only if all three do, so each is tested at 0.05 without
# correction (intersection-union). `--sender=lOFC` runs the same chain with
# lOFC as the leading region. The ordering -- mPFC's direction index minus
# mOFC's -- is reported alongside, as in `swr_hfb_direction.py`.


def ft_mask(t_all, sess, R):
    """Ripples inside a first-traversal window (grid onset -> first t_D)."""
    w = win.windows_phase3_whole(swr_io.load_behaviour(sess, data_root=R))
    w = w[w.phase3 == "explore"]
    inside = np.zeros(len(t_all), bool)
    for a, b in zip(w.start_s, w.end_s):
        inside |= (t_all >= a) & (t_all < b)
    return inside, float((w.end_s - w.start_s).sum())


def chain(bundle=None, partition_dir=None, out_dir=None, show=True,
          sender="mOFC"):
    """`sender` is the ROI tested as the one that leads (mOFC or lOFC)."""
    snd = sender
    rois = (snd, "mPFC")
    import matplotlib
    if not show:
        matplotlib.use("Agg")

    R = swr_io.get_data_root()
    G = os.path.join(swr_io.derivatives_dir(R), "group", "swr")
    b_dir = bundle or os.path.join(G, "bundle_v2")
    partition_dir = partition_dir or os.path.join(G, PARTITION)
    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(
        G, f"{ANALYSIS_NAME}_chain_{snd}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    rip, hiv, hp = load_inputs(b_dir, partition_dir)
    rng = np.random.default_rng(SEED)
    lo, hi = SHIFT_RANGE_S
    shifts = [0.0] + [float(x) for x in rng.uniform(lo, hi, N_SHIFTS)
                      * rng.choice([-1, 1], N_SHIFTS)]
    sessions = sorted(hp[hp.roi.isin(rois)].session.unique())

    rows, crows, curves, maps = [], [], [], []
    for sess in sessions:
        H, idx, masks, g, fs, n = session_data(sess, b_dir, hp, hiv, rois)
        w = int(HALF_S * fs)
        off = np.arange(-w, w)
        t_ms = off / fs * 1000.0
        flank = (np.abs(t_ms) >= NONPERI_S[0] * 1000) & \
                (np.abs(t_ms) < NONPERI_S[1] * 1000)
        t_all = session_ripples(rip, sess)
        ft, ft_dur = ft_mask(t_all, int(sess), R)
        t_ft = t_all[ft]
        F, E = {}, {}
        for roi in rois:
            pids = g[g.roi == roi].pair_id.tolist()
            if not pids:
                continue
            for sh in shifts:
                X = roi_epochs(H, idx, masks, pids,
                               np.round((t_ft + sh) * fs).astype(int), off, n)
                F[(roi, sh)] = features(X, t_ms)
                E[(roi, sh)] = X - X[:, flank].mean(1, keepdims=True)
        # links 1 and 2: each ROI on its own, all its sessions
        for roi in rois:
            if (roi, 0.0) not in F:
                continue
            ok = {sh: np.isfinite(F[(roi, sh)]).all(1) for sh in shifts}
            if min(o.sum() for o in ok.values()) < MIN_RIPPLES:
                continue
            pre = {sh: F[(roi, sh)][ok[sh], 0].mean() for sh in shifts}
            post = {sh: F[(roi, sh)][ok[sh], 1].mean() for sh in shifts}
            cur = {sh: E[(roi, sh)][ok[sh]].mean(0) for sh in shifts}
            null = lambda d: np.mean([d[sh] for sh in shifts[1:]], 0)
            rows.append(dict(roi=roi, session=int(sess),
                             subject=g.subject_label.iloc[0],
                             n_ripples=int(ok[0.0].sum()),
                             n_ft_ripples=int(ft.sum()), ft_s=ft_dur,
                             pre=pre[0.0] - null(pre),
                             post=post[0.0] - null(post)))
            curves.append(dict(roi=roi, session=int(sess),
                               x=cur[0.0] - null(cur)))
        # the chain: same ripples in both ROIs
        if all((r, 0.0) in F for r in rois):
            out = {}
            for sh in shifts:
                FA, FB = F[(snd, sh)], F[("mPFC", sh)]
                ok = np.isfinite(FA).all(1) & np.isfinite(FB).all(1)
                if ok.sum() < MIN_RIPPLES:
                    break
                out[sh] = (coupling(FA[ok], FB[ok]), int(ok.sum()))
            if len(out) == len(shifts):
                nM = np.mean([out[sh][0] for sh in shifts[1:]], 0)
                a_r, f_r, r_r = asymmetry(out[0.0][0])
                a_n, f_n, r_n = asymmetry(nM)
                crows.append(dict(session=int(sess),
                                  subject=g.subject_label.iloc[0],
                                  n_ripples=out[0.0][1], asym=a_r - a_n,
                                  fwd=f_r - f_n, rev=r_r - r_n))
                maps.append((out[0.0][0] - nM)[2:, 2:])
        print(f"  s{int(sess):02d} done", end="\r")

    L, C = pd.DataFrame(rows), pd.DataFrame(crows)
    L["direction"] = L.post - L.pre

    # ---- tests ----------------------------------------------------------
    def one(test, v, primary, **kw):
        v = np.asarray(v, float)
        return dict(test=test, n_sessions=len(v), mean=float(v.mean()),
                    sem=float(v.std(ddof=1) / np.sqrt(len(v))),
                    p_perm=sign_flip(v)[1], primary=primary, **kw)
    T = []
    for roi in rois:
        d = L[L.roi == roi]
        for col in ("pre", "post", "direction"):
            prim = (roi, col) in ((snd, "pre"), ("mPFC", "post"))
            name = {(snd, "pre"): f"link 1: {snd} before the ripple",
                    ("mPFC", "post"): "link 2: mPFC after the ripple"}.get(
                (roi, col), f"{roi} {col}")
            T.append(one(name, d[col], prim, roi=roi, measure=col,
                         n_subjects=int(d.subject.nunique())))
    a = L[L.roi == "mPFC"].direction.to_numpy()
    b = L[L.roi == snd].direction.to_numpy()
    diff, p = label_perm(a, b)
    T.append(dict(test=f"ordering: mPFC − {snd} direction", roi=f"mPFC−{snd}",
                  measure="direction", n_sessions=f"{len(a)}/{len(b)}",
                  mean=diff, sem=np.nan, p_perm=p, primary=False))
    for col, prim in (("asym", True), ("fwd", False), ("rev", False)):
        T.append(one({"asym": f"chain: {snd} pre → mPFC post, asymmetry",
                      "fwd": f"chain component: {snd} pre → mPFC post",
                      "rev": f"chain component: mPFC pre → {snd} post"}[col],
                     C[col], prim, roi=f"{snd}→mPFC", measure=col,
                     n_subjects=int(C.subject.nunique())))
    T = pd.DataFrame(T)

    print(f"\n{'=' * 92}\nTHE CHAIN, ripple-locked, every first-traversal "
          f"ripple (real − shifted; sign-flip over sessions)\n{'=' * 92}")
    for _, r in T.iterrows():
        print(f"  {r.test:<44s}n={str(r.n_sessions):>6s}  {r['mean']:+.4f}"
              + ("" if pd.isna(r['sem']) else f" ± {r['sem']:.4f}")
              + f"  p={r.p_perm:.3f}" + ("  PRIMARY" if r.primary else ""))
    print(f"  first-traversal ripples per session: median "
          f"{L.n_ripples.median():.0f} ({snd}/mPFC rows), chain sessions "
          f"median {C.n_ripples.median():.0f}")

    L.to_csv(os.path.join(out_dir, "chain_links_per_session.csv"), index=False)
    C.to_csv(os.path.join(out_dir, "chain_coupling_per_session.csv"),
             index=False)
    T.to_csv(os.path.join(out_dir, "chain_tests.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "chain_curves.npz"), t_ms=t_ms,
                        **{f"{c['roi']}_s{c['session']}": c["x"]
                           for c in curves},
                        coupling_maps=np.stack(maps))
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"settings": dict(
                       analysis=f"{ANALYSIS_NAME}_chain", created=stamp,
                       script=os.path.relpath(__file__), bundle=b_dir,
                       partition_dir=partition_dir,
                       ripples="every ripple inside a first-traversal window "
                               "(grid onset -> first t_D, "
                               "swr_windows.windows_phase3_whole)",
                       pre_post_ms=[[-WIN_MS, 0], [0, WIN_MS]],
                       flank_s=list(NONPERI_S), shifts_s=shifts, seed=SEED,
                       min_ripples=MIN_RIPPLES,
                       sender=snd,
                       primary=f"link 1 {snd} pre, link 2 mPFC post, chain "
                               "asymmetry; each at 0.05 uncorrected "
                               "(intersection-union: the chain needs all "
                               "three)", unit="session"),
                   "tests": T.to_dict("records")}, f, indent=2, default=str)
    _figure_chain_ripple(L, C, T, curves, maps, t_ms, out_dir, snd)
    print(f"\n  saved -> {out_dir}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()


def _figure_chain_ripple(L, C, T, curves, maps, t_ms, out_dir, snd="mOFC"):
    rois = (snd, "mPFC")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)        # dot jitter only
    edges = np.arange(-MAP_HALF_MS, MAP_HALF_MS + BIN_MS, BIN_MS)
    fs = 1000.0 / (t_ms[1] - t_ms[0])
    ptxt = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    with plt.rc_context(rfig._rc()):
        fig, axes = plt.subplots(2, 3, figsize=(19 * rfig.CM, 13 * rfig.CM))
        # A: the curves
        ax = axes[0, 0]
        ax.axvspan(-WIN_MS, 0, color="#000000", alpha=0.06, lw=0)
        ax.axvspan(0, WIN_MS, color="#000000", alpha=0.12, lw=0)
        for roi in rois:
            X = np.stack([c["x"] for c in curves if c["roi"] == roi])
            from scipy.ndimage import gaussian_filter1d
            Xs = gaussian_filter1d(X, (50.0 / 1000 * fs) / 2.355, axis=1)
            m, se = Xs.mean(0), Xs.std(0, ddof=1) / np.sqrt(len(Xs))
            ax.fill_between(t_ms, m - se, m + se, color=ROI_COL[roi],
                            alpha=0.25, lw=0)
            ax.plot(t_ms, m, color=ROI_COL[roi], lw=1.5,
                    label=f"{roi} (n={len(X)})")
        ax.axhline(0, color="#3d3d3d", lw=0.6, ls="--")
        ax.axvline(0, color="#1a1a1a", lw=0.7, ls=":")
        ax.set_xlim(-750, 750)
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.4 * (hi_ - lo_))     # room for the legend
        ax.set_xlabel("time from ripple (ms)", fontsize=rfig.FS_TICK - 1)
        ax.set_ylabel("HFB, real − shifted (z)", fontsize=rfig.FS_TICK - 1)
        ax.set_title("ripple-locked HFB, every\nfirst-traversal ripple",
                     fontsize=rfig.FS_TICK)
        ax.legend(frameon=False, fontsize=rfig.FS_TICK - 2, loc="upper left",
                  ncol=2, handlelength=1.0, columnspacing=0.8)

        # B: links 1 and 2 with their components
        ax = axes[0, 1]
        cells = [(snd, "pre"), (snd, "post"), ("mPFC", "pre"),
                 ("mPFC", "post")]
        for i, (roi, col) in enumerate(cells):
            v = L[L.roi == roi][col].to_numpy()
            r = T[(T.roi == roi) & (T.measure == col)].iloc[0]
            c = ROI_COL[roi]
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=8, color=c,
                       alpha=0.5, lw=0, zorder=2)
            ax.errorbar(i, v.mean(), yerr=r["sem"], fmt="o", ms=5, mfc=c,
                        mec="#1a1a1a", color="#1a1a1a", lw=1.1, capsize=2,
                        zorder=4)
            ax.text(i, 1.02, "< 0.001" if r.p_perm < 0.001 else f"{r.p_perm:.3f}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=rfig.FS_TICK - 2.5,
                    fontweight="bold" if r.primary else "normal", color=c)
        lo_, hi_ = np.percentile(L[["pre", "post"]].to_numpy(), [2, 98])
        ax.set_ylim(lo_ - 0.1 * (hi_ - lo_), hi_ + 0.1 * (hi_ - lo_))
        ax.axhline(0, color="#3d3d3d", lw=0.7, ls="--", zorder=1)
        ax.axvline(1.5, color="#bbbbbb", lw=0.7, ls=":")
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"{r}\n{c}" for r, c in cells],
                           fontsize=rfig.FS_TICK - 1)
        ax.set_ylabel("HFB, real − shifted (z)", fontsize=rfig.FS_TICK - 1)
        ax.set_title(f"LINK 1: {snd} before  |  LINK 2: mPFC after\n(bold p = "
                     "the link; the other window shown for scale)",
                     fontsize=rfig.FS_TICK - 1, pad=14)

        # C: the ordering
        ax = axes[0, 2]
        for i, roi in enumerate(rois):
            v = L[L.roi == roi].direction.to_numpy()
            c = ROI_COL[roi]
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=8, color=c,
                       alpha=0.5, lw=0, zorder=2)
            ax.errorbar(i, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)),
                        fmt="o", ms=5, mfc=c, mec="#1a1a1a", color="#1a1a1a",
                        lw=1.1, capsize=2, zorder=4)
        r = T[T.measure.eq("direction") & T.roi.eq(f"mPFC−{snd}")].iloc[0]
        lo_, hi_ = np.percentile(L.direction, [2, 98])
        ax.set_ylim(lo_ - 0.1 * (hi_ - lo_), hi_ + 0.1 * (hi_ - lo_))
        ax.axhline(0, color="#3d3d3d", lw=0.7, ls="--", zorder=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(list(rois), fontsize=rfig.FS_TICK - 1)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel("direction, post − pre (z)", fontsize=rfig.FS_TICK - 1)
        ax.set_title(f"ORDERING: mPFC later than {snd}?\nmPFC − {snd} "
                     f"{r['mean']:+.4f}, {ptxt(r.p_perm)}",
                     fontsize=rfig.FS_TICK - 1)

        # D: chain components, session by session
        ax = axes[1, 0]
        for a_, b_ in zip(C.fwd, C.rev):
            ax.plot([0, 1], [a_, b_], color="#9a9a9a", lw=0.7, zorder=1)
        ax.scatter(np.zeros(len(C)), C.fwd, s=12, color=ROI_COL[snd],
                   zorder=2)
        ax.scatter(np.ones(len(C)), C.rev, s=12, color=ROI_COL["mPFC"],
                   zorder=2)
        for x, v in ((0, C.fwd.to_numpy()), (1, C.rev.to_numpy())):
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)),
                        fmt="_", ms=14, color="#1a1a1a", lw=1.4, capsize=3,
                        zorder=3)
        r = T[T.measure.eq("asym")].iloc[0]
        ax.axhline(0, color="#3d3d3d", lw=0.7, ls="--")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{snd} pre\n→ mPFC post", f"mPFC pre\n→ {snd} post"],
                           fontsize=rfig.FS_TICK - 1)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel("ripple-by-ripple coupling\n(Fisher z, real − shifted)",
                      fontsize=rfig.FS_TICK - 1)
        ax.set_title(f"CHAIN: same ripple ({len(C)} sessions)\nasymmetry "
                     f"{r['mean']:+.3f} ± {r['sem']:.3f}, {ptxt(r.p_perm)}",
                     fontsize=rfig.FS_TICK - 1)

        # E: the coupling map
        ax = axes[1, 1]
        M = np.stack(maps).mean(0)
        lim = np.nanmax(np.abs(M))
        im = ax.pcolormesh(edges, edges, M, cmap="RdYlBu_r", vmin=-lim,
                           vmax=lim, shading="flat")
        ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], color="#1a1a1a",
                lw=0.6, ls=":")
        ax.axhline(0, color="#1a1a1a", lw=0.5)
        ax.axvline(0, color="#1a1a1a", lw=0.5)
        ax.set_aspect("equal")
        ax.set_xticks([-500, 0, 500])
        ax.set_yticks([-500, 0, 500])
        ax.set_xlabel("mPFC time from ripple (ms)", fontsize=rfig.FS_TICK - 1)
        ax.set_ylabel(f"{snd} time from ripple (ms)", fontsize=rfig.FS_TICK - 1)
        ax.set_title(f"coupling map (real − shifted)\nabove diagonal = {snd} "
                     "earlier", fontsize=rfig.FS_TICK - 1)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
        cb.set_label("Δz", fontsize=rfig.FS_TICK - 2)
        cb.ax.tick_params(labelsize=rfig.FS_TICK - 2)

        ax = axes[1, 2]
        ax.axis("off")
        ax.text(0.0, 0.5,
                "Every ripple inside a first-traversal\nwindow (grid onset → "
                "first D), no event\nlabel needed. HFB real − 6 shifted\n"
                "nulls (5–120 s), flank-normalised.\npre = [−250, 0), post = "
                "(0, +250] ms.\nDots: sessions; marker: mean ± SEM;\n"
                "p: sign-flip over sessions (ordering:\nROI-label permutation). "
                "Pre-specified:\nlink 1, link 2, chain; the chain needs\nall "
                "three, so each is at 0.05\nuncorrected (intersection–union).\n"
                "Curves: 50 ms smoothing, drawing only.",
                ha="left", va="center", fontsize=rfig.FS_TICK - 2,
                color="#3d3d3d", transform=ax.transAxes)
        for a in axes.ravel():
            a.tick_params(labelsize=rfig.FS_TICK - 1)
        fig.suptitle(f"{snd} → ripple → mPFC, ripple-locked, first traversal",
                     fontsize=rfig.FS_TITLE)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        stem = os.path.join(out_dir, "chain_ripple_locked")
        fig.savefig(stem + ".png", dpi=300)
        fig.savefig(stem + ".pdf")


# ── OFC / mOFC vs mPFC by phase, ripple- and uncover-locked ──────────────
# The `swr_hfb_direction.py` analyses -- per-ROI cluster test against its own
# baseline, the timing test, the time-resolved contrast -- on sets the saved
# all-ripple run cannot split. Ripple-locked sets (epoch +-1 s):
#   all              every ripple
#   first_traversal  grid onset -> the first t_D
#   learning         grid onset -> END of the first fully correct repeat,
#                    that repeat INCLUDED (SK's definition here; the project
#                    default since 2026-09-27 is EXCLUSIVE, see swr_phases)
#   execution        after the first fully correct repeat -> grid end
# Locked to correct uncovers (epoch +-1.5 s), phase by the uncover's REPEAT:
#   reward_uncover            every correct uncover
#   reward_uncover_learning   repeat <= the first fully correct one (so the
#                             first traversal and the D that completes the
#                             first correct loop are in)
#   reward_uncover_execution  repeat >  the first fully correct one
# Per event, the mean over the ROI's derivations clean for the whole window
# (OFC = its mOFC and lOFC derivations together); per session the mean over
# events, real − mean of the shifted nulls. mOFC, lOFC and mPFC come from ONE
# pass, so every pairing sees the same events.
# SHIFTED NULLS. With 6 of them, a different draw alone moved the unpaired
# timing p from 0.032 to 0.069 (2026-10-08 (c)). `n_shifts` sets how many;
# `stability=True` re-runs the tests on each half of them and reports the
# remaining spread.
PH_SETS = {
    "all": ("all ripples", "ripple"),
    "first_traversal": ("first traversal", "ripple"),
    "learning": ("learning (to the first correct repeat, incl.)", "ripple"),
    "execution": ("execution (after the first correct repeat)", "ripple"),
    "reward_uncover": ("correct uncovers", "event"),
    "reward_uncover_learning": ("correct uncovers, learning (incl.)", "event"),
    "reward_uncover_execution": ("correct uncovers, execution", "event"),
}
PH_WIN_S = {"ripple": (-1.0, 1.0), "event": (-1.5, 1.5)}
PH_MIN = {"ripple": MIN_RIPPLES, "event": 10}
PH_FAMS = ("mOFC", "lOFC", "mPFC")
PH_LEFT = {"OFC": ("mOFC", "lOFC"), "mOFC": ("mOFC",), "lOFC": ("lOFC",),
           "mPFC": ("mPFC",)}


def phase_windows(beh):
    """first_traversal / learning (first correct repeat INCLUDED) /
    execution, as continuous spans per grid."""
    rows = []
    for _, g in beh.groupby(["session_no", "grid_no"]):
        g = g.sort_values("rep_overall")
        t0 = float(g.new_grid_onset.iloc[0])
        t_ft = float(g.t_D.iloc[0])
        t_end = float(g.t_D.max())
        solved = g[g.correct.astype(int) == 1]
        t_learn = float(solved.t_D.iloc[0]) if len(solved) else t_end
        if np.isfinite(t_ft):
            t_learn = max(t_learn, t_ft)
        for name, a, b in (("first_traversal", t0, t_ft),
                           ("learning", t0, t_learn),
                           ("execution", t_learn, t_end)):
            if np.isfinite(a) and np.isfinite(b) and b > a:
                rows.append((name, a, b))
    return pd.DataFrame(rows, columns=["phase", "start_s", "end_s"])


def _inside(t, W):
    m = np.zeros(len(t), bool)
    for a, b in zip(W.start_s, W.end_s):
        m |= (t >= a) & (t < b)
    return m


def uncover_phase(unc, beh):
    """'learning' when the uncover's repeat is at or before its grid's first
    fully correct repeat, else 'execution'."""
    first_ok = {}
    for grid, g in beh.groupby("grid_no"):
        ok = g.loc[g.correct.astype(int) == 1, "rep_overall"]
        first_ok[int(grid)] = int(ok.min()) if len(ok) else np.inf
    return np.array(["learning" if r <= first_ok.get(int(gr), np.inf)
                     else "execution"
                     for gr, r in zip(unc.grid_no, unc.rep_overall)])


def phases(bundle=None, partition_dir=None, out_dir=None, show=True,
           lefts=("OFC",), n_shifts=N_SHIFTS, sets=tuple(PH_SETS),
           stability=False):
    """`lefts`: the ROIs set against mPFC -- OFC (mOFC + lOFC), mOFC, lOFC.
    `sets`: keys of PH_SETS. `n_shifts`: shifted nulls."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    from swr_hfb_direction import report_pair, analyse_pair

    lefts = (lefts,) if isinstance(lefts, str) else tuple(lefts)
    sets = (sets,) if isinstance(sets, str) else tuple(sets)
    R = swr_io.get_data_root()
    G = os.path.join(swr_io.derivatives_dir(R), "group", "swr")
    b_dir = bundle or os.path.join(G, "bundle_v2")
    partition_dir = partition_dir or os.path.join(G, PARTITION)
    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(
        G, f"hfb_phases_{n_shifts}shifts_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    rip, hiv, hp = load_inputs(b_dir, partition_dir)
    rng = np.random.default_rng(SEED)
    lo, hi = SHIFT_RANGE_S
    shifts = [0.0] + [float(x) for x in rng.uniform(lo, hi, n_shifts)
                      * rng.choice([-1, 1], n_shifts)]
    rois = list(dict.fromkeys(list(lefts) + ["mPFC"]))
    sessions = sorted(hp[hp.roi.isin(PH_FAMS)].session.unique())
    print(f"\n  {len(sessions)} sessions, {n_shifts} shifted nulls, rois "
          f"{rois}, sets {list(sets)}")

    # per (set, roi): {session: (n_shifts + 1, T) per-shift means}
    M = {(k, r): {} for k in sets for r in rois}
    counts, t_axis = [], {}
    for sess in sessions:
        H, idx, masks, g, fs, n = session_data(sess, b_dir, hp, hiv, PH_FAMS)
        pids = {f: g[g.roi == f].pair_id.tolist() for f in PH_FAMS}
        beh = swr_io.load_behaviour(int(sess), data_root=R)
        W = phase_windows(beh)
        t_rip = session_ripples(rip, sess)
        unc = swb.uncover_events(int(sess), data_root=R)
        unc = unc[unc.correct.astype(int) == 1]
        uph = uncover_phase(unc, beh)
        locks = {
            "ripple": (t_rip, {
                "all": np.ones(len(t_rip), bool),
                **{ph: _inside(t_rip, W[W.phase == ph])
                   for ph in ("first_traversal", "learning", "execution")}}),
            "event": (unc.t_s.to_numpy(float), {
                "reward_uncover": np.ones(len(unc), bool),
                "reward_uncover_learning": uph == "learning",
                "reward_uncover_execution": uph == "execution"})}
        for lock, (tt, set_mask) in locks.items():
            need = [k for k in sets if PH_SETS[k][1] == lock]
            if not need or not len(tt):
                continue
            off = np.arange(int(round(PH_WIN_S[lock][0] * fs)),
                            int(round(PH_WIN_S[lock][1] * fs)))
            t_axis[lock] = off / fs * 1000.0
            per = {(k, r): [] for k in need for r in rois}
            n_real = {}
            for sh in shifts:
                ci = np.round((tt + sh) * fs).astype(int)
                S = {f: roi_sums(H, idx, masks, pids[f], ci, off, n)
                     for f in PH_FAMS if pids[f]}
                for r in rois:
                    fam = [f for f in PH_LEFT[r] if f in S]
                    if not fam:
                        continue
                    s_ = sum(S[f][0] for f in fam)
                    k_ = sum(S[f][1] for f in fam)
                    for key in need:
                        sel = (k_ > 0) & set_mask[key]
                        per[(key, r)].append(
                            (s_[sel] / k_[sel, None]).mean(0)
                            if sel.sum() >= PH_MIN[lock] else None)
                        if sh == 0.0:
                            n_real[(key, r)] = int(sel.sum())
            for (key, r), lst in per.items():
                if len(lst) == len(shifts) and all(x is not None
                                                   for x in lst):
                    M[(key, r)][int(sess)] = np.stack(lst)
                    counts.append(dict(set=key, roi=r, session=int(sess),
                                       n=n_real[(key, r)]))
        print(f"  s{int(sess):02d} done", end="\r")
    pd.DataFrame(counts).to_csv(os.path.join(out_dir, "counts.csv"),
                                index=False)

    def reg_of(key, roi, part=slice(1, None)):
        return {s: m[0] - m[part].mean(0) for s, m in M[(key, roi)].items()}

    def best(cl):
        return min([c["p"] for c in cl], default=np.nan)

    halves = (slice(1, 1 + n_shifts // 2), slice(1 + n_shifts // 2, None))
    stab = []
    for left in lefts:
        pair = (left, "mPFC")
        sub = os.path.join(out_dir, f"{left}_vs_mPFC")
        out = {}
        for key in sets:
            title, lock = PH_SETS[key]
            reg = {r: reg_of(key, r) for r in pair}
            print(f"\n  ── {left} vs mPFC, {key}: {len(reg[left])} / "
                  f"{len(reg['mPFC'])} sessions ──")
            res, timing, C = report_pair(
                reg, t_axis[lock], pair, os.path.join(sub, key), lock,
                title=f"{'Uncover' if lock == 'event' else 'Ripple'}-locked "
                      f"HFB, {title}",
                settings=dict(analysis="hfb_phases", set=key, created=stamp,
                              script=os.path.relpath(__file__), bundle=b_dir,
                              partition_dir=partition_dir,
                              left=f"{left} = {' + '.join(PH_LEFT[left])} "
                                   "derivations, mean per event",
                              n_shifts=n_shifts, shifts_s=shifts,
                              min_rows=PH_MIN[lock],
                              phase_boundary="first fully correct repeat "
                                             "INCLUDED in learning"))
            out[key] = (res, timing, C)
            rows = [("all", res, timing)]
            if stability:
                for h, part in zip(("first half", "second half"), halves):
                    print(f"    [{h} of the shifts]")
                    rh, th = analyse_pair({r: reg_of(key, r, part)
                                           for r in pair},
                                          t_axis[lock], pair, lock)
                    rows.append((h, rh, th))
            for h, rr, tm in rows:
                stab.append(dict(
                    left=left, set=key, shifts=h,
                    **{f"best_cluster_p_{r}": best(
                        [c for c in rr[r]["clusters"]
                         if c["sign"] == "increase"]) for r in pair},
                    timing_diff=tm.get("diff"), timing_p=tm.get("p_perm"),
                    timing_p_paired=tm.get("p_paired"),
                    n_left=rr[left]["n_units"], n_mPFC=rr["mPFC"]["n_units"]))
        _figure_phases(out, t_axis, pair, sub, sets)
    St = pd.DataFrame(stab)
    St.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\n" + St.round(4).to_string(index=False))
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"settings": dict(analysis="hfb_phases", created=stamp,
                                    script=os.path.relpath(__file__),
                                    n_shifts=n_shifts, shifts_s=shifts,
                                    lefts=list(lefts), sets=list(sets),
                                    stability=stability, unit="session"),
                   "summary": St.to_dict("records")}, f, indent=2,
                  default=str)
    print(f"\n  saved -> {out_dir}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()


def _figure_phases(out, t_axis, pair, out_dir, sets):
    """One row per set: twin-axis panel | timing | peak-scaled contrast."""
    import matplotlib.pyplot as plt
    from swr_hfb_direction import draw_sliding, draw_timing, draw_contrast, \
        ROI_SHOW

    with plt.rc_context(rfig._rc()):
        fig, axes = plt.subplots(len(sets), 3, squeeze=False,
                                 figsize=(20 * rfig.CM,
                                          len(sets) * 6.0 * rfig.CM),
                                 gridspec_kw=dict(width_ratios=[1.5, 0.75,
                                                                1.3]))
        for i, key in enumerate(sets):
            title, lock = PH_SETS[key]
            res, timing, C = out[key]
            t_ms = t_axis[lock]
            draw_sliding(axes[i, 0], res, t_ms, pair, lock, title=title,
                         legend=True, fs_scale=1.5)
            draw_timing(axes[i, 1], res, timing, pair, lock, title=" ",
                        fs_scale=1.5)
            draw_contrast(axes[i, 2], C, t_ms, pair, lock, "peak",
                          title="contrast, peak-scaled", legend=False,
                          fs_scale=1.5)
        fig.suptitle(f"{ROI_SHOW.get(pair[0], pair[0])} vs mPFC: "
                     "ripple-locked and locked to correct uncovers",
                     fontsize=rfig.FS_TITLE)
        fig.text(0.5, 0.0,
                 "Left: each ROI against its own baseline (grey), each axis "
                 "scaled to that ROI's peak; bars = clusters p < 0.05. "
                 "Middle: after − before index per session (± 250 ms), "
                 "ROI-label permutation and paired check. Right: the two "
                 "components, peak-scaled,\nfill = which is bigger; bars = "
                 "contrast clusters (thick unpaired, thin paired). Learning "
                 "INCLUDES the first fully correct repeat. Uncorrected across "
                 "sets.", ha="center", va="bottom",
                 fontsize=rfig.FS_TICK - 2.5, color="#3d3d3d")
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"overview.{ext}"), dpi=250)


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run, "timeline": timeline, "chain": chain,
                   "phases": phases})
    else:
        run()
