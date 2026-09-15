#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ripple-triggered RSA on human single units -- the cell analogue of the
instruction-phase fMRI analysis.

THE QUESTION
------------
In fMRI the four reward locations are *shown* during a 12 s instruction period,
and `fMRI_run_RSA_instruction.py` asks how much of the plan is assembled as that
period unfolds (the CUMULATIVE family: A_rew, AB_rew, ABC_rew, ABCD_rew). In the
cell task nothing is shown: subjects *discover* A, B, C, D one at a time. The
same "how much of the plan is assembled" axis therefore falls out of the task
itself -- it is indexed by WHICH reward was just uncovered rather than by time
on a clock. Ripple rate rises after those discoveries
(`swr_final_ripple_analysis.py`), so the population state at those ripples is
where a freshly loaded plan should be visible.

WHAT THE TASK DESIGN ALLOWS, AND WHAT IT FORBIDS
------------------------------------------------
The 8 shared configs are counterbalanced so that WITHIN ANY ONE STATE all eight
reward locations are distinct:

    state A: 3 8 1 4 6 9 7 2        state C: 9 6 5 1 2 3 4 7
    state B: 7 2 9 8 4 1 3 5        state D: 5 7 8 3 9 4 2 6

Three consequences, all verified in `verify_model_variance`:

1. The position-locked Hamming model used in the fMRI (`rewDSR`) is CONSTANT
   inside a within-state RDM -- sd = 0.000 at A, B, C and D. It cannot be
   fitted here. The generalisation that survives is set overlap of the
   locations known so far.
2. At state A the known set has ONE element and all eight are distinct, so
   every knowledge-gated model is constant. State A is not a weak test, it is
   an impossible one -- for knowledge-gated models. It is still testable with
   the fixed full-ABCD model (below), where it is the knowledge null.
3. **The confound control is free.** The subject stands on loc_k when uncovering
   state k, and all eight loc_k are distinct, so "current location" is constant
   inside every within-state RDM. A within-state effect cannot be a place code.
   This holds unconditionally and does not depend on the pre-window.

THE MODELS
----------
    known_set   unordered overlap of locations 1..k -- working memory
                accumulating the rewards found so far, no plan structure.
    known_seq   the same shared locations, weighted by how far apart in the
                sequence they sit (1/(1+|rank_i - rank_j|)) -- sequence
                sensitive, i.e. relevant to how the route will be executed.
    full_abcd   set overlap of the WHOLE ABCD config, identical at every state.
                The subject cannot know it at A and can at D, so its fit should
                grow A -> D. Its correlation with `known_set` is 0.19 at B,
                0.60 at C and 1.00 at D, so at B and C it doubles as a
                confound detector: a fit to locations not yet uncovered is a
                red flag, not a result.

`known_set` and `known_seq` are correlated at r = 1.00 (B), 0.97 (C), 0.92 (D).
Over 28 config pairs that is not separable in a regression, so DO NOT run them
as a horse race -- `rank_offset_test` asks the sequence question directly
instead, by testing whether similarity depends on the rank offset of the shared
location among the pairs that share one.

INFERENCE
---------
Only 8 conditions, so the config-label permutation is EXHAUSTIVE: all
8! = 40,320 relabellings, no sampling and no null-SD-shrinkage trap. The same
estimator computes the observed value and every permutation value, per
CLAUDE.md rule 4.

@author: Svenja Kuchenhoff
"""

import os
import glob
import itertools

import numpy as np
import pandas as pd
from scipy import stats

import mc.analyse.swr_units as swu


# =============================================================================
# SETTINGS -- all pre-declared, none tuned on these data
# =============================================================================

BIN_S = 0.025                      # the derivatives' bin, as in swr_units

# Firing is read 0 to +200 ms after the ripple peak. Pre-declared in
# scripts/swr_ripple_triggered_units.py as the standard window for
# ripple-locked cortical activity; reused unchanged so nothing is tuned here.
PERI_WIN_S = (0.0, 0.200)

# Windows relative to the uncover press in which ripples are collected.
# `pre` is short because the median gap from the PREVIOUS press to an uncover
# press is only 0.35 s -- a longer pre-window would sit before the subject
# arrived at the square and would encode the previous location instead.
PRESS_WINDOWS = {"pre": (-0.35, 0.0), "post": (0.15, 0.70)}

DEDUP_S = 0.05                     # one ripple on several derivations is one event

STATES = ["A", "B", "C", "D"]

# The 8 configs shared across sessions, in the fixed project order used by
# prep_human_cells_RSA-2026.py and RSA_human_cells_DSR.py.
CONFIGS = [(3, 7, 9, 5), (8, 2, 6, 7), (1, 9, 5, 8), (4, 8, 1, 3),
           (6, 4, 2, 9), (9, 1, 3, 4), (7, 3, 4, 2), (2, 5, 7, 6)]
CONFIG_LABELS = ["-".join(map(str, c)) for c in CONFIGS]
N_CONFIG = len(CONFIGS)

DSR_SESSIONS = [27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 40, 43, 44, 45, 46,
                49, 50, 51, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63]

# Native cell labels are already the ROI names in these sessions (verified:
# HC 236, OFC 68, ACC 67, EC 51 across the 28 sessions).
PRIMARY_ROIS = ["HC", "ACC"]
SECONDARY_ROIS = ["OFC", "EC"]
ROIS = PRIMARY_ROIS + SECONDARY_ROIS

# A config pair whose correlation rests on fewer cells than this is not
# estimated. Kept low on purpose: dropping pairs breaks the exhaustive
# permutation, so the whole (roi, state, window) cell is reported as unusable
# instead of being silently patched.
MIN_CELLS_PER_PAIR = 10


# =============================================================================
# EVENTS
# =============================================================================

def load_bundle(bundle_dir):
    """The three bundle tables this analysis needs."""
    return {
        "uncover": pd.read_csv(os.path.join(bundle_dir, "uncover.csv")),
        "ripples": pd.read_csv(os.path.join(bundle_dir, "ripples.csv")),
        "behaviour": pd.read_csv(os.path.join(bundle_dir, "behaviour.csv")),
    }


def discovery_events(bundle, sessions=None):
    """First discovery of each reward: correct, `is_discovery`, explore stage.

    One row per (session, grid, state) with the grid's config label attached.
    These are the events the ripple-rate increase was established on
    (`correct, first uncovers` in the results draft: +0.035 Hz, cluster
    +0.35 to +0.75 s).
    """
    sessions = DSR_SESSIONS if sessions is None else sessions
    u = bundle["uncover"]
    ev = u[(u.session.isin(sessions)) & (u.correct == 1)
           & (u.is_discovery == 1) & u.state.notna()].copy()

    beh = bundle["behaviour"]
    cfg = beh[beh.session.isin(sessions)].drop_duplicates(["session", "grid_no"])
    cfg = cfg[["session", "grid_no", "loc_A", "loc_B", "loc_C", "loc_D"]].copy()
    cfg["cfg"] = (cfg[["loc_A", "loc_B", "loc_C", "loc_D"]]
                  .astype(int).astype(str).agg("-".join, axis=1))

    ev = ev.merge(cfg[["session", "grid_no", "cfg"]], on=["session", "grid_no"],
                  how="left")
    # only the 8 shared configs -- a session may also have run others
    ev = ev[ev.cfg.isin(CONFIG_LABELS)]
    return ev[["session", "grid_no", "rep_overall", "t_s", "state", "cfg"]] \
             .reset_index(drop=True)


def session_ripples(bundle, session):
    """Deduplicated ripple peak times for one session."""
    r = bundle["ripples"]
    return swu.dedup_ripples(r.loc[r.session == session, "t_peak_s"].values,
                             tol_s=DEDUP_S)


# =============================================================================
# RIPPLE-TRIGGERED PATTERNS
# =============================================================================

def session_patterns(session, events, ripple_t, beh, window, data_root=None,
                     split_halves=False):
    """Ripple-triggered firing per cell, per (config, state), for one session.

    For every discovery event, the ripples falling inside `window` relative to
    the press are collected; each contributes the cell's mean firing over
    `PERI_WIN_S` after the ripple peak; those are averaged within
    (config, state).

    Returns (labels, sums, counts) where
        sums    (n_cells, N_CONFIG, 4)   summed peri-ripple firing
        counts  (N_CONFIG, 4)            ripples contributing -- shared by all
                                         cells of the session, since they see
                                         the same ripples
    With `split_halves`, sums/counts gain a leading axis of length 2 holding
    alternate ripples (for the split-half reliability check).
    """
    labels = swu.unit_labels(session, data_root)
    if labels is None:
        return None
    n_cells = len(labels)
    n_half = 2 if split_halves else 1

    lo = int(round(PERI_WIN_S[0] / BIN_S))
    hi = int(round(PERI_WIN_S[1] / BIN_S))

    sums = np.zeros((n_half, n_cells, N_CONFIG, len(STATES)))
    counts = np.zeros((n_half, N_CONFIG, len(STATES)))

    ev = events[events.session == session]
    seen = 0
    for grid, g in ev.groupby("grid_no"):
        M = swu.grid_firing(session, grid, data_root)
        if M is None or M.shape[0] != n_cells:
            continue
        onset = float(beh.loc[(beh.session == session) & (beh.grid_no == grid),
                              "new_grid_onset"].iloc[0])
        n_bins = M.shape[1]
        for _, row in g.iterrows():
            ci = CONFIG_LABELS.index(row.cfg)
            si = STATES.index(row.state)
            t0, t1 = row.t_s + window[0], row.t_s + window[1]
            for t in ripple_t[(ripple_t >= t0) & (ripple_t < t1)]:
                b = int(round((t - onset) / BIN_S))
                if b + lo < 0 or b + hi > n_bins:
                    continue
                h = seen % n_half
                sums[h, :, ci, si] += M[:, b + lo:b + hi].mean(axis=1)
                counts[h, ci, si] += 1
                seen += 1

    if not split_halves:
        return labels, sums[0], counts[0]
    return labels, sums, counts


def collect_patterns(bundle, events, sessions=None, data_root=None,
                     split_halves=False, verbose=True):
    """Pool ripple-triggered patterns over sessions into one cell x config x state
    array per press window.

    Cells are pooled ACROSS sessions -- legitimate only because these 28
    sessions share the same 8 configs, which is what makes the feature space
    236 HC / 67 mPFC cells instead of the ~9 / ~3 available within a session.
    A cell whose session contributed no ripple to a (config, state) cell is NaN
    there; nothing is imputed.
    """
    sessions = DSR_SESSIONS if sessions is None else sessions
    beh = bundle["behaviour"]
    out = {}
    for wname, window in PRESS_WINDOWS.items():
        P, rois, sess_of_cell, N = [], [], [], []
        for s in sessions:
            rip = session_ripples(bundle, s)
            res = session_patterns(s, events, rip, beh, window, data_root,
                                   split_halves=split_halves)
            if res is None:
                if verbose:
                    print(f"  [{wname}] s{s}: no unit labels -- skipped")
                continue
            labels, sums, counts = res
            # counts are shared by every cell of the session, so they need a
            # cell axis to broadcast against `sums`
            denom = counts[:, None] if split_halves else counts
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = sums / np.where(denom == 0, np.nan, denom)
            P.append(mean)
            rois.extend(labels)
            sess_of_cell.extend([s] * len(labels))
            N.append(counts)
        axis = 1 if split_halves else 0
        out[wname] = {
            "patterns": np.concatenate(P, axis=axis),
            "roi": np.array(rois),
            "session": np.array(sess_of_cell),
            "counts": np.stack(N),          # (n_sessions, [2,] config, state)
            "sessions": np.array(sessions),
        }
        if verbose:
            tot = int(np.nansum(np.stack(N)))
            print(f"  [{wname}] {tot} ripples, {len(rois)} cells "
                  f"over {len(P)} sessions")
    return out


# =============================================================================
# DATA RDMs
# =============================================================================

def build_rdm(P):
    """Correlation-distance RDM over the 8 configs from a (n_cells, 8) pattern.

    Each cell is centred across the configs it has data for, so a cell's
    overall firing rate -- which is the same in every condition -- cannot make
    all configs look alike. Each config pair then uses the cells present in
    both (no imputation), so the RDM is estimated pairwise-complete.

    Returns (rdm, n_cells_per_pair). Entries resting on fewer than
    MIN_CELLS_PER_PAIR cells are NaN.
    """
    P = np.asarray(P, float)
    C = P - np.nanmean(P, axis=1, keepdims=True)

    rdm = np.full((N_CONFIG, N_CONFIG), np.nan)
    n_used = np.zeros((N_CONFIG, N_CONFIG), int)
    np.fill_diagonal(rdm, 0.0)
    for i in range(N_CONFIG):
        for j in range(i + 1, N_CONFIG):
            ok = np.isfinite(C[:, i]) & np.isfinite(C[:, j])
            n_used[i, j] = n_used[j, i] = ok.sum()
            if ok.sum() < MIN_CELLS_PER_PAIR:
                continue
            a, b = C[ok, i], C[ok, j]
            if a.std() == 0 or b.std() == 0:
                continue
            rdm[i, j] = rdm[j, i] = 1.0 - np.corrcoef(a, b)[0, 1]
    return rdm, n_used


def rdm_for(pack, roi, state, min_cells=MIN_CELLS_PER_PAIR):
    """Data RDM for one ROI at one state, from a `collect_patterns` entry."""
    sel = pack["roi"] == roi
    P = pack["patterns"][sel][:, :, STATES.index(state)]
    return build_rdm(P)


# =============================================================================
# MODEL RDMs
# =============================================================================

def _known(cfg, k):
    """The locations the subject knows once state k (0-based) is uncovered."""
    return set(cfg[:k + 1])


def known_set_rdm(k):
    """Unordered overlap of the locations known so far -- Jaccard distance."""
    M = np.zeros((N_CONFIG, N_CONFIG))
    for i in range(N_CONFIG):
        for j in range(N_CONFIG):
            a, b = _known(CONFIGS[i], k), _known(CONFIGS[j], k)
            M[i, j] = 1.0 - len(a & b) / len(a | b)
    return M


def known_seq_rdm(k):
    """Shared locations weighted by their rank offset in the sequence.

    A location that both configs visit at adjacent steps counts more than one
    they visit three steps apart, so the model is sensitive to WHERE in the
    route a shared location sits, not just that it is shared. The
    position-locked version of this (offset 0 only) is constant by design --
    see the module docstring -- so the weighting is what makes it fittable.
    """
    S = np.zeros((N_CONFIG, N_CONFIG))
    for i in range(N_CONFIG):
        for j in range(N_CONFIG):
            s = 0.0
            for x in range(k + 1):
                for y in range(k + 1):
                    if CONFIGS[i][x] == CONFIGS[j][y]:
                        s += 1.0 / (1.0 + abs(x - y))
            S[i, j] = s
    return 1.0 - S / S.max()


def full_abcd_rdm():
    """Set overlap of the whole ABCD config -- the same RDM at every state."""
    return known_set_rdm(3)


def position_locked_rdm(k):
    """The fMRI `rewDSR` Hamming form. Provided so its emptiness is explicit."""
    M = np.zeros((N_CONFIG, N_CONFIG))
    for i in range(N_CONFIG):
        for j in range(N_CONFIG):
            M[i, j] = np.mean([CONFIGS[i][x] != CONFIGS[j][x]
                               for x in range(k + 1)])
    return M


def model_rdms(state):
    """The models fittable at one state, plus the ones that are not."""
    k = STATES.index(state)
    return {
        "known_set": known_set_rdm(k),
        "known_seq": known_seq_rdm(k),
        "full_abcd": full_abcd_rdm(),
        "position_locked": position_locked_rdm(k),   # constant; never fitted
    }


def verify_model_variance():
    """Which models carry variance at which state -- printed, not assumed."""
    iu = np.triu_indices(N_CONFIG, 1)
    rows = []
    for state in STATES:
        for name, M in model_rdms(state).items():
            v = M[iu]
            rows.append({"state": state, "model": name, "sd": v.std(),
                         "n_distinct": len(np.unique(np.round(v, 6))),
                         "fittable": v.std() > 1e-12})
    return pd.DataFrame(rows)


# =============================================================================
# INFERENCE -- exhaustive config-label permutation
# =============================================================================

_PERMS = None


def all_permutations():
    """All 8! = 40,320 config relabellings, cached."""
    global _PERMS
    if _PERMS is None:
        _PERMS = np.array(list(itertools.permutations(range(N_CONFIG))))
    return _PERMS


def fit_model(rdm, model):
    """Spearman correlation between a data RDM and a model RDM, with an EXACT
    permutation test over config labels.

    The permutation relabels the configs of the data RDM, which is the null
    "the population pattern carries no information about which config this is".
    Because the observed value and all 40,320 null values come from the same
    two lines of code, the CLAUDE.md rule on permutations is satisfied by
    construction.

    Returns dict with rho, p_two_sided, p_one_sided (negative rho = similar
    configs are less distant = the predicted direction) and the null.
    """
    iu = np.triu_indices(N_CONFIG, 1)
    ok = np.isfinite(rdm[iu]) & np.isfinite(model[iu])
    if ok.sum() < N_CONFIG:                       # not enough pairs to say anything
        return {"rho": np.nan, "p": np.nan, "p_one_sided": np.nan,
                "n_pairs": int(ok.sum()), "null": np.array([])}
    if not np.isfinite(rdm[iu]).all():
        # a NaN pair would move between model entries under relabelling, so the
        # permutation would not be exchangeable. Report rather than patch.
        return {"rho": np.nan, "p": np.nan, "p_one_sided": np.nan,
                "n_pairs": int(ok.sum()), "null": np.array([]),
                "note": "incomplete RDM -- no exact permutation possible"}

    # ranks are invariant to relabelling, so rank once and permute the matrix
    R = np.zeros((N_CONFIG, N_CONFIG))
    R[iu] = stats.rankdata(rdm[iu])
    R = R + R.T
    m = stats.rankdata(model[iu])
    m = (m - m.mean()) / m.std()

    perms = all_permutations()
    Rp = R[perms[:, :, None], perms[:, None, :]][:, iu[0], iu[1]]
    Rp = (Rp - Rp.mean(axis=1, keepdims=True)) / Rp.std(axis=1, keepdims=True)
    null = Rp @ m / m.size

    obs = null[0]                                  # identity is the first perm
    assert np.allclose(perms[0], np.arange(N_CONFIG))
    p_two = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    p_one = float((null <= obs + 1e-12).mean())    # negative rho predicted
    return {"rho": float(obs), "p": p_two, "p_one_sided": p_one,
            "n_pairs": int(ok.sum()), "null": null}


def rank_offset_test(rdm, state="D"):
    """Does similarity depend on WHERE in the sequence a shared location sits?

    Among the config pairs that share at least one location, the shared
    location sits at some rank offset (0 = same step, which never happens by
    design; 1, 2 or 3 otherwise). A sequence-sensitive code predicts that
    pairs sharing a location at a small offset are more similar. This replaces
    the `known_set` vs `known_seq` horse race, which is not identifiable at
    r = 0.92 over 28 pairs.

    At state D the offsets are populated 21 / 14 / 7 (offset 1 / 2 / 3).
    """
    k = STATES.index(state)
    offs, dists = [], []
    for i in range(N_CONFIG):
        for j in range(i + 1, N_CONFIG):
            shared = [abs(x - y) for x in range(k + 1) for y in range(k + 1)
                      if CONFIGS[i][x] == CONFIGS[j][y]]
            if not shared or not np.isfinite(rdm[i, j]):
                continue
            offs.append(min(shared))
            dists.append(rdm[i, j])
    offs, dists = np.array(offs), np.array(dists)
    if len(offs) < 6 or len(np.unique(offs)) < 2:
        return {"rho": np.nan, "p": np.nan, "n_pairs": len(offs)}
    rho, p = stats.spearmanr(offs, dists)
    return {"rho": float(rho), "p": float(p), "n_pairs": int(len(offs)),
            "n_per_offset": {int(o): int((offs == o).sum())
                             for o in np.unique(offs)}}


def split_half_reliability(pack, roi, state):
    """Spearman between two RDMs built from alternate ripples.

    Not a gate on the analysis -- the permutation null already controls false
    positives, and reliability only bears on how a NULL should be read. With
    ~1.3 ripples per cell per condition this is expected to be low; it is
    reported so a null is not mistaken for an absence of signal.
    """
    sel = pack["roi"] == roi
    si = STATES.index(state)
    halves = [build_rdm(pack["patterns"][h][sel][:, :, si])[0] for h in (0, 1)]
    iu = np.triu_indices(N_CONFIG, 1)
    a, b = halves[0][iu], halves[1][iu]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 6:
        return {"rho": np.nan, "n_pairs": int(ok.sum())}
    rho, p = stats.spearmanr(a[ok], b[ok])
    return {"rho": float(rho), "p": float(p), "n_pairs": int(ok.sum())}


def jackknife_sessions(pack, roi, state, model):
    """Leave-one-session-out refit, so no single session can carry the result."""
    sel = pack["roi"] == roi
    si = STATES.index(state)
    P = pack["patterns"][sel][:, :, si]
    sess = pack["session"][sel]
    out = []
    for s in np.unique(sess):
        rdm, _ = build_rdm(P[sess != s])
        out.append({"left_out": int(s), "rho": fit_model(rdm, model)["rho"]})
    return pd.DataFrame(out)
