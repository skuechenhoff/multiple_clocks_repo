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
# One second either side. The earlier, narrower windows ([-0.35, 0] and
# [+0.15, +0.70], taken from the ripple-rate cluster) left a median of 8
# ripples per (config x state) in the pre window, too few for any RDM cell to
# be estimated. A symmetric second either side gives 732 / 907 ripples
# (median 22 / 28 per condition) and makes the before-vs-after comparison
# estimable at all. The cost is that the post window no longer sits only on
# the rate increase.
PRESS_WINDOWS = {"pre": (-1.0, 0.0), "post": (0.0, 1.0)}

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

# A config pair whose correlation rests on fewer cells than this is left
# unestimated (NaN). The RDM is then partial, which `fit_model` handles by
# permuting the model rather than the data, so the observed-pair mask stays
# fixed.
MIN_CELLS_PER_PAIR = 10

# Partial RDMs are fitted -- see `fit_model`. This is the floor on how many
# config pairs must survive before a fit is attempted at all; below it the
# correlation is a description of a handful of numbers, not an estimate.
MIN_PAIRS_TO_FIT = 10


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


def fit_model(rdm, model, exact=True):
    """Spearman between a data RDM and a model RDM, with an EXACT permutation
    test over config labels -- and it tolerates a PARTIAL data RDM.

    Partial RDMs are the normal case here: a config pair whose two conditions
    share too few recorded cells cannot be estimated. Permuting the DATA labels
    would then move the holes around, so the set of compared pairs would change
    from permutation to permutation. Permuting the MODEL instead keeps the
    observed-pair mask fixed and is exactly as valid a null -- "the plan
    structure is assigned to configurations at random" -- so that is what is
    done. Observed and null values come from the identical two lines.

    Returns rho, the two-sided and one-sided p, and the null.

    SIGN: both matrices are DISSIMILARITIES, so they co-vary POSITIVELY when
    the region encodes the model. Two configurations that share few locations
    are far apart in the model AND should be far apart in the data. Verified by
    simulation: patterns built to encode the reward set give rho = +0.92.
    The one-sided p is therefore the upper tail.
    """
    iu = np.triu_indices(N_CONFIG, 1)
    d, m = rdm[iu], model[iu]
    mask = np.isfinite(d) & np.isfinite(m)
    n_obs = int(mask.sum())
    if n_obs < MIN_PAIRS_TO_FIT or np.unique(m[mask]).size < 2 \
            or np.unique(d[mask]).size < 2:
        return {"rho": np.nan, "p": np.nan, "p_one_sided": np.nan,
                "n_pairs": n_obs, "null": np.array([])}

    dr = stats.rankdata(d[mask])
    dr = (dr - dr.mean()) / dr.std()

    perms = all_permutations() if exact else _random_permutations()
    Mp = model[perms[:, :, None], perms[:, None, :]][:, iu[0], iu[1]][:, mask]
    Mr = stats.rankdata(Mp, axis=1)
    Mr = (Mr - Mr.mean(axis=1, keepdims=True)) / Mr.std(axis=1, keepdims=True)
    null = Mr @ dr / dr.size

    obs = float(null[0])          # the first permutation is the identity
    p_two = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    p_one = float((null >= obs - 1e-12).mean())     # upper tail: see SIGN above
    return {"rho": obs, "p": p_two, "p_one_sided": p_one, "n_pairs": n_obs,
            "null": null}


def fit_rho(rdm, model):
    """Just the Spearman rho between a (possibly partial) data RDM and a model.

    `fit_model` builds a whole permutation null on every call, which is wasted
    work inside a permutation loop -- there, only the point estimate of each
    permuted dataset is wanted. Same masking and same statistic, no null.
    """
    iu = np.triu_indices(N_CONFIG, 1)
    d, m = rdm[iu], model[iu]
    mask = np.isfinite(d) & np.isfinite(m)
    if mask.sum() < MIN_PAIRS_TO_FIT:
        return np.nan
    d, m = d[mask], m[mask]
    if np.unique(d).size < 2 or np.unique(m).size < 2:
        return np.nan
    return float(stats.spearmanr(d, m).correlation)


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


# =============================================================================
# RAW SPIKE TIMES -- from abcd_passed.mat, not the 25 ms binned derivatives
# =============================================================================
#
# The per-grid `all_cells_firing_rate_grid*.csv` matrices are 25 ms histograms
# of exactly these spike times (`scripts/save_iEEG_as_csv.m`:
# `edges = 0:0.025:end_time_behaviour`, `histcounts(spikeTimes, edges)`).
# A ripple is ~60 ms long, so a peri-ripple window of +-10 ms is SMALLER THAN
# ONE BIN and cannot be formed from them at all. Anything that needs firing
# resolved inside a ripple has to come from here.
#
# Spike times are in seconds on the same session clock as the behaviour and
# the ripple times -- `save_iEEG_as_csv.m` bins them from 0 with no offset, and
# the behavioural timestamps are on that same clock.

MAT_PATH = "abcd_passed.mat"
ROI_TABLE = "neurons_with_ROI_labels.csv"

# `neurons_with_ROI_labels.csv` carries the project's canonical ROI names in
# `atlas_roi` (HC_anterior / HC_mid / mPFC / mOFC / EC / PCC), indexed by
# `cell idx` within a subject. Verified 2026-09-15: that order matches the
# `electrodeLabel` order in abcd_passed.mat for all 28 sessions, so the join is
# positional and safe -- unlike `neurons_MNI_latest.csv`, which is NOT
# row-aligned with the labels file for s27/s40/s50/s57/s60.
ROI_COLUMN = "atlas_roi"


def _derivatives(data_root=None):
    import mc.analyse.swr_io as swr_io
    return swr_io.derivatives_dir(data_root)


def _mat_str(f, ref):
    return "".join(chr(c) for c in np.array(f[ref]).ravel())


def load_spike_times(sessions=None, data_root=None, cache_dir=None,
                     verbose=True):
    """Raw spike times per cell, per session, straight from abcd_passed.mat.

    Returns {session: {"spikes": [np.ndarray, ...], "electrode": [str, ...]}}.
    The mat file is 5.8 GB and HDF5-v7.3, so extracted spike times are cached
    as one .npz per session; delete the cache to force a re-read.
    """
    import h5py
    sessions = DSR_SESSIONS if sessions is None else sessions
    deriv = _derivatives(data_root)
    cache_dir = cache_dir or os.path.join(deriv, "group", "swr", "spike_cache")
    os.makedirs(cache_dir, exist_ok=True)

    out, need = {}, []
    for s in sessions:
        p = os.path.join(cache_dir, f"spikes_s{s:02d}.npz")
        if os.path.isfile(p):
            z = np.load(p, allow_pickle=True)
            out[s] = {"spikes": list(z["spikes"]),
                      "electrode": [str(x) for x in z["electrode"]]}
        else:
            need.append(s)
    if need:
        if verbose:
            print(f"  reading {len(need)} session(s) from {MAT_PATH} "
                  f"(cached afterwards)")
        with h5py.File(os.path.join(deriv, MAT_PATH), "r") as f:
            nd = f["abcd_passed/abcd_data"]["neural_data"]
            for s in need:
                g = f[nd[s - 1, 0]]
                st = g["spikeTimes"]
                spikes = [np.sort(np.array(f[st[i, 0]]).ravel().astype(float))
                          for i in range(st.shape[0])]
                elec = [_mat_str(f, g["electrodeLabel"][i, 0])
                        for i in range(st.shape[0])]
                np.savez_compressed(
                    os.path.join(cache_dir, f"spikes_s{s:02d}.npz"),
                    spikes=np.array(spikes, dtype=object),
                    electrode=np.array(elec, dtype=object))
                out[s] = {"spikes": spikes, "electrode": elec}
                if verbose:
                    print(f"    s{s}: {len(spikes)} cells, "
                          f"{sum(len(x) for x in spikes):,} spikes")
    return out


def cell_roi_table(sessions=None, data_root=None):
    """One row per cell: session, 0-based cell index, electrode, ROI.

    Rows are in the SAME order as `load_spike_times` returns cells, so the two
    can be zipped without a join.
    """
    sessions = DSR_SESSIONS if sessions is None else sessions
    t = pd.read_csv(os.path.join(_derivatives(data_root), ROI_TABLE))
    t = t[t.subject.isin(sessions)].copy()
    t = t.sort_values(["subject", "cell idx"]).reset_index(drop=True)
    return pd.DataFrame({
        "session": t.subject.astype(int),
        "cell": t["cell idx"].astype(int) - 1,
        "electrode": t["electrode label"].astype(str),
        "roi": t[ROI_COLUMN].astype(str),
    })


def spike_counts_in_windows(spikes, centres, half_widths):
    """Spikes per window for one cell -- vectorised over windows.

    `half_widths` may be a scalar or one value per centre (e.g. each ripple's
    own duration/2).
    """
    centres = np.asarray(centres, float)
    hw = np.broadcast_to(np.asarray(half_widths, float), centres.shape)
    lo = np.searchsorted(spikes, centres - hw, side="left")
    hi = np.searchsorted(spikes, centres + hw, side="right")
    return hi - lo


def ripples_near_events(bundle, session, events, window, with_duration=True):
    """Ripples falling in `window` around each discovery press of one session.

    Returns a DataFrame with one row per ripple: t_peak_s, duration_s, and the
    config/state of the event it belongs to.

    NOTE ON RIPPLE EXTENT: `ripple_events.csv` on the cluster carries
    `t_start_s` / `t_peak_s` / `t_end_s`, but the bundle export keeps only the
    peak and `duration_s`. On the one session held locally (s38) the peak sits
    essentially at the centre of the event -- median 23 ms before, 24 ms after
    -- so `peak +- duration/2` reconstructs the extent well. Re-exporting the
    bundle with the two existing columns removes the approximation; no
    re-detection is needed.
    """
    r = bundle["ripples"]
    r = r[r.session == session]
    t = r.t_peak_s.values
    dur = r.duration_s.values if with_duration else np.full(len(r), np.nan)
    order = np.argsort(t)
    t, dur = t[order], dur[order]

    rows = []
    ev = events[events.session == session]
    for _, e in ev.iterrows():
        lo = np.searchsorted(t, e.t_s + window[0])
        hi = np.searchsorted(t, e.t_s + window[1])
        for k in range(lo, hi):
            rows.append({"session": session, "t_peak_s": t[k],
                         "duration_s": dur[k], "cfg": e.cfg, "state": e.state,
                         "grid_no": e.grid_no, "press_t_s": e.t_s})
    cols = ["session", "t_peak_s", "duration_s", "cfg", "state", "grid_no",
            "press_t_s"]
    out = pd.DataFrame(rows, columns=cols)
    if len(out):
        # one ripple detected on several derivations is one event
        out = out.sort_values("t_peak_s")
        keep = np.concatenate([[True], np.diff(out.t_peak_s.values) > DEDUP_S])
        out = out[keep]
    return out.reset_index(drop=True)


# =============================================================================
# RIPPLE-TRIGGERED PATTERNS, FROM RAW SPIKES
# =============================================================================

def collect_spike_patterns(bundle, events, roi_tab, spikes, window,
                           extent="duration", fixed_half_s=0.010,
                           split_halves=False, drop_silent=True, verbose=True,
                           rng=None, config_perm=None):
    """Firing rate inside each ripple, averaged per (config, state), per cell.

    A ripple's extent is taken as `t_peak +- duration_s/2` (`extent="duration"`),
    which is what the bundle allows; `extent="fixed"` uses `+- fixed_half_s`
    instead. Rate is spikes divided by the window's own width, so ripples of
    different length are comparable.

    Cells are pooled ACROSS sessions -- legitimate only because these 28
    sessions share the same 8 configs. A cell whose session contributed no
    ripple to a (config, state) cell is NaN there; nothing is imputed.

    `drop_silent` removes cells that fire NO spike in ANY ripple of this
    window. Those cells are an all-zero column: after the per-cell centring in
    `build_rdm` they contribute exactly nothing to any correlation, but they do
    inflate the apparent feature count, so they are dropped and counted.
    NOTE: individual zero counts are NOT dropped -- "this neuron was silent in
    this ripple" is data, and discarding it would bias every rate upward.
    """
    n_half = 2 if split_halves else 1
    P, rois, sess_of_cell, N, silent = [], [], [], [], 0

    for s in sorted(set(roi_tab.session)):
        rip = ripples_near_events(bundle, s, events, window)
        cells = roi_tab[roi_tab.session == s]
        if rip.empty or cells.empty:
            continue
        ci = np.array([CONFIG_LABELS.index(c) for c in rip.cfg])
        if config_perm is not None:
            # a FIXED relabelling, supplied by the caller. Needed for the
            # post-minus-pre contrast: both windows must be relabelled the same
            # way inside one permutation, or the contrast null is inflated by
            # two independent shuffles instead of one.
            ci = config_perm[s][ci]
        elif rng is not None:
            # NULL: relabel this session's configurations at random. Ripple
            # counts, states, cell coverage and the missing-data pattern are
            # all preserved exactly; only the identity of which configuration
            # a ripple belongs to is destroyed -- and because each session gets
            # its OWN relabelling, the cross-session config alignment that
            # pooling cells depends on is destroyed too. That alignment is the
            # signal, so this is the null the design calls for.
            ci = rng.permutation(N_CONFIG)[ci]
        si = np.array([STATES.index(x) for x in rip.state])
        half = (rip.duration_s.values / 2.0 if extent == "duration"
                else np.full(len(rip), fixed_half_s))
        width = 2 * half
        hix = np.arange(len(rip)) % n_half

        sums = np.zeros((n_half, len(cells), N_CONFIG, len(STATES)))
        counts = np.zeros((n_half, N_CONFIG, len(STATES)))
        for h in range(n_half):
            m = hix == h
            np.add.at(counts[h], (ci[m], si[m]), 1)

        keep = []
        for row, (_, c) in enumerate(cells.iterrows()):
            n = spike_counts_in_windows(spikes[s]["spikes"][int(c.cell)],
                                        rip.t_peak_s.values, half)
            if drop_silent and n.sum() == 0:
                silent += 1
                continue
            rate = n / width
            for h in range(n_half):
                m = hix == h
                np.add.at(sums[h, row], (ci[m], si[m]), rate[m])
            keep.append(row)

        if not keep:
            continue
        sums = sums[:, keep]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sums / np.where(counts[:, None] == 0, np.nan, counts[:, None])
        P.append(mean if split_halves else mean[0])
        rois.extend(cells.iloc[keep].roi.tolist())
        sess_of_cell.extend([s] * len(keep))
        N.append(counts if split_halves else counts[0])

    axis = 1 if split_halves else 0
    pack = {"patterns": np.concatenate(P, axis=axis),
            "roi": np.array(rois), "session": np.array(sess_of_cell),
            "counts": np.stack(N), "n_silent_dropped": silent}
    if verbose:
        print(f"  {int(np.nansum(np.stack(N)))} ripples, {len(rois)} cells kept, "
              f"{silent} silent cells dropped")
    return pack


def peri_ripple_control(bundle, roi_tab, spikes, extent="duration",
                        fixed_half_s=0.010):
    """Do units fire more inside ripples than just outside them?

    Peri = the ripple's own extent. Non-peri = two windows of the SAME total
    width, one on each side, offset by 250 ms -- equal width matters, because
    comparing counts from windows of different length is the trap recorded in
    `POTENTIAL_IDEAS.md`. One row per (session, ROI).

    The doctrine from `swr_ripple_triggered_units.py` stands: hippocampal units
    must show an increase, or nothing representational from these two clocks is
    interpretable. It is a positive control, not a result.
    """
    rows = []
    for s in sorted(set(roi_tab.session)):
        r = bundle["ripples"]
        r = r[r.session == s]
        if r.empty:
            continue
        t = np.sort(r.t_peak_s.values)
        dur = r.duration_s.values[np.argsort(r.t_peak_s.values)]
        t = swu.dedup_ripples(t)
        dur = dur[:len(t)] if len(dur) >= len(t) else np.pad(
            dur, (0, len(t) - len(dur)), constant_values=np.median(dur))
        half = dur / 2.0 if extent == "duration" else np.full(len(t), fixed_half_s)
        cells = roi_tab[roi_tab.session == s]
        for roi, g in cells.groupby("roi"):
            peri, non = [], []
            for _, c in g.iterrows():
                st = spikes[s]["spikes"][int(c.cell)]
                p = spike_counts_in_windows(st, t, half) / (2 * half)
                a = spike_counts_in_windows(st, t - 0.25 - half, half) / (2 * half)
                b = spike_counts_in_windows(st, t + 0.25 + half, half) / (2 * half)
                peri.append(p.mean())
                non.append((a.mean() + b.mean()) / 2)
            rows.append({"session": int(s), "roi": roi, "n_units": len(g),
                         "n_ripples": len(t), "peri": float(np.mean(peri)),
                         "non_peri": float(np.mean(non))})
    d = pd.DataFrame(rows)
    d["pct_change"] = 100 * (d.peri - d.non_peri) / d.non_peri
    return d


def summarise_control(ctrl):
    """Paired test plus a sign test per ROI -- the sign test matters here
    because the per-session percentages are heavy-tailed at these unit counts."""
    out = []
    for roi, g in ctrl.groupby("roi"):
        t, p = stats.ttest_rel(g.peri, g.non_peri)
        n_pos = int((g.peri > g.non_peri).sum())
        out.append({"roi": roi, "n_sessions": len(g),
                    "n_units": int(g.n_units.sum()),
                    "mean_pct_change": g["pct_change"].mean(),
                    "sem_pct_change": g["pct_change"].sem(),
                    "t": t, "p_paired": p, "n_sessions_positive": n_pos,
                    "p_sign": stats.binomtest(n_pos, len(g), 0.5).pvalue})
    return pd.DataFrame(out)


def pipeline_null(bundle, events, roi_tab, spikes, window, roi, state, model,
                  n_perm=100, seed=42, **kwargs):
    """Null distribution of the model fit, re-estimated through the WHOLE
    pipeline rather than by relabelling a finished RDM.

    Each permutation draws a fresh per-session configuration relabelling, then
    rebuilds the patterns, the RDM and the fit with the identical code that
    produced the observed value -- CLAUDE.md rule 4. That makes it sensitive to
    everything the RDM-level permutation cannot see: the cell centring, the
    missing-data structure, and the fact that a condition's pattern rests on a
    handful of spikes.

    100 permutations is enough to see how big the effect is relative to chance;
    it is not enough for a precise p (resolution 0.01).
    """
    rng = np.random.default_rng(seed)
    M = model_rdms(state)[model]
    out = []
    for _ in range(n_perm):
        pack = collect_spike_patterns(bundle, events, roi_tab, spikes, window,
                                      verbose=False, rng=rng, **kwargs)
        rdm, _ = rdm_for(pack, roi, state)
        out.append(fit_model(rdm, M, exact=False)["rho"])
    return np.array(out, float)


_RAND_PERMS = None


def _random_permutations(n=2000, seed=42):
    """A fixed random subset of the 8! relabellings.

    Used inside `pipeline_null`, where the exact 40,320 would be recomputed
    100 times over for no gain -- the quantity wanted there is the observed
    rho of each permuted dataset, not its own p.
    """
    global _RAND_PERMS
    if _RAND_PERMS is None:
        rng = np.random.default_rng(seed)
        P = np.array([rng.permutation(N_CONFIG) for _ in range(n - 1)])
        _RAND_PERMS = np.vstack([np.arange(N_CONFIG), P])
    return _RAND_PERMS


def pipeline_null_contrast(bundle, events, roi_tab, spikes, roi, state, model,
                           n_perm=100, seed=42, **kwargs):
    """Null for the post-minus-pre contrast of model fits.

    A difference of two Spearman rhos has no standard sampling distribution, so
    it is only interpretable against a null built the same way. Each
    permutation draws ONE per-session configuration relabelling and applies it
    to BOTH windows, then takes the difference of the two fits -- exactly the
    arithmetic used on the real data. Anything the two windows share (session
    composition, cell coverage, ripple counts) survives the permutation, so the
    null isolates the config identity, which is what the contrast claims.
    """
    rng = np.random.default_rng(seed)
    sessions = sorted(set(roi_tab.session))
    M = model_rdms(state)[model]
    out = []
    for _ in range(n_perm):
        perm = {s: rng.permutation(N_CONFIG) for s in sessions}
        rho = {}
        for wname, w in PRESS_WINDOWS.items():
            pack = collect_spike_patterns(bundle, events, roi_tab, spikes, w,
                                          verbose=False, config_perm=perm,
                                          **kwargs)
            rdm, _ = rdm_for(pack, roi, state)
            rho[wname] = fit_model(rdm, M, exact=False)["rho"]
        out.append(rho["post"] - rho["pre"])
    return np.array(out, float)


# =============================================================================
# FAST PATH: cache the (cell x ripple) rates once, permute by re-indexing
# =============================================================================
#
# `collect_spike_patterns` re-reads every spike train on every call, which caps
# a permutation test at ~100 draws. But a permutation only changes which
# (config, state) bin a ripple falls into -- the spike counts themselves never
# change. Caching the (cell x ripple) rate matrix per session therefore makes a
# permutation a re-indexing operation, and thousands of draws become cheap.
# `patterns_from_cache` reproduces `collect_spike_patterns` exactly.

def cache_ripple_rates(bundle, events, roi_tab, spikes, window,
                       extent="duration", fixed_half_s=0.010,
                       surrogate_rng=None, scheme="window"):
    """Per session: the rate of every cell in every ripple, plus its labels.

    `surrogate_rng` replaces each ripple time with a random time drawn from the
    SAME press window of the SAME event, keeping its duration. Everything else
    -- which event, which config, which state, how many windows, which cells --
    is untouched, so a comparison against it isolates one thing: whether the
    window had to be at a ripple.
    """
    get = SCHEMES[scheme]
    out = []
    for s in sorted(set(roi_tab.session)):
        rip = get(bundle, s, events, window)
        cells = roi_tab[roi_tab.session == s]
        if rip.empty or cells.empty:
            continue
        t = rip.t_peak_s.values.copy()
        if surrogate_rng is not None:
            if scheme == "interval":
                # a surrogate must stay inside the SAME interval, or it would
                # change which knowledge state the window belongs to
                lo = rip.press_t_s.values
                hi = rip.interval_end_s.values
                t = lo + surrogate_rng.uniform(0, 1, len(rip)) * (hi - lo)
            else:
                t = (rip.press_t_s.values
                     + surrogate_rng.uniform(window[0], window[1], len(rip)))
        half = (rip.duration_s.values / 2.0 if extent == "duration"
                else np.full(len(rip), fixed_half_s))
        width = 2 * half
        rates = np.empty((len(cells), len(rip)))
        for row, (_, c) in enumerate(cells.iterrows()):
            n = spike_counts_in_windows(spikes[s]["spikes"][int(c.cell)], t, half)
            rates[row] = n / width
        out.append({
            "session": s,
            "rates": rates,
            "ci": np.array([CONFIG_LABELS.index(c) for c in rip.cfg]),
            "si": np.array([STATES.index(x) for x in rip.state]),
            "roi": cells.roi.to_numpy(),
        })
    return out


def patterns_from_cache(cache, config_perm=None, split_halves=False,
                        drop_silent=True):
    """Rebuild a pattern pack from cached rates -- same output as
    `collect_spike_patterns`, but a permutation costs no spike lookups."""
    n_half = 2 if split_halves else 1
    P, rois, sess, N, silent = [], [], [], [], 0
    for blk in cache:
        ci = blk["ci"]
        if config_perm is not None:
            ci = config_perm[blk["session"]][ci]
        si, rates = blk["si"], blk["rates"]
        hix = np.arange(rates.shape[1]) % n_half

        counts = np.zeros((n_half, N_CONFIG, len(STATES)))
        for h in range(n_half):
            m = hix == h
            np.add.at(counts[h], (ci[m], si[m]), 1)

        keep = np.ones(rates.shape[0], bool)
        if drop_silent:
            keep = rates.sum(axis=1) > 0
            silent += int((~keep).sum())
        if not keep.any():
            continue
        sums = np.zeros((n_half, int(keep.sum()), N_CONFIG, len(STATES)))
        R = rates[keep]
        for h in range(n_half):
            m = hix == h
            np.add.at(sums[h].transpose(1, 2, 0), (ci[m], si[m]), R[:, m].T)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sums / np.where(counts[:, None] == 0, np.nan, counts[:, None])
        P.append(mean if split_halves else mean[0])
        rois.extend(blk["roi"][keep].tolist())
        sess.extend([blk["session"]] * int(keep.sum()))
        N.append(counts if split_halves else counts[0])

    axis = 1 if split_halves else 0
    return {"patterns": np.concatenate(P, axis=axis), "roi": np.array(rois),
            "session": np.array(sess), "counts": np.stack(N),
            "n_silent_dropped": silent}


def fit_pooled(pack, roi, model, states=("B", "C", "D")):
    """One fit across several states instead of one fit per state.

    The knowledge-gated model makes the SAME claim at B, C and D -- only the
    known set grows -- so the states can be fitted together. Each state's pairs
    are ranked within that state (their dissimilarity scales differ) and then
    concatenated, which turns 28 observations into up to 84 and drops the null
    SD from ~0.19 to ~0.11. State A is excluded: the model is constant there.
    """
    iu = np.triu_indices(N_CONFIG, 1)
    D, M = [], []
    for st in states:
        rdm, _ = rdm_for(pack, roi, st)
        m = model_rdms(st)[model][iu]
        d = rdm[iu]
        ok = np.isfinite(d) & np.isfinite(m)
        if ok.sum() < 3 or np.unique(m[ok]).size < 2:
            continue
        D.append(stats.rankdata(d[ok]) / ok.sum())
        M.append(stats.rankdata(m[ok]) / ok.sum())
    if not D:
        return {"rho": np.nan, "n_pairs": 0}
    D, M = np.concatenate(D), np.concatenate(M)
    if D.std() == 0 or M.std() == 0:
        return {"rho": np.nan, "n_pairs": len(D)}
    return {"rho": float(np.corrcoef(D, M)[0, 1]), "n_pairs": int(len(D))}


# =============================================================================
# RIPPLE -> CONDITION ASSIGNMENT: two schemes
# =============================================================================

def ripples_in_intervals(bundle, session, events, window=None):
    """Every ripple between one reward discovery and the next.

    Scheme suggested by SK's supervisor. A ripple is assigned to state k if it
    falls in [uncover_k, uncover_{k+1}); state D runs until the NEXT repeat's
    t_A. Three properties make this better than a fixed window around the
    press:

    1. **Coverage.** 4163 ripples instead of 885 (4.7x), median 126 per
       (config x state) RDM cell instead of 28.
    2. **No double counting.** The intervals tile the first traversal exactly
       once, so no ripple can land in two conditions -- which a +-1 s window
       around consecutive presses cannot guarantee.
    3. **The knowledge state is constant throughout.** Between uncovering A and
       uncovering B the subject knows exactly {A}, for the whole interval. That
       is precisely what the knowledge-gated model describes, so the interval
       is a better match to the model than a window that happens to sit near
       the press.

    The cost: it is no longer "the moment of discovery" -- most of the interval
    is spent searching for the next reward. `window` is accepted and ignored,
    so the two schemes are interchangeable at the call site.

    Returns the same columns as `ripples_near_events`.
    """
    beh = bundle["behaviour"]
    b = beh[beh.session == session]
    ev = events[events.session == session]
    t = np.sort(swu.dedup_ripples(
        bundle["ripples"].loc[bundle["ripples"].session == session,
                              "t_peak_s"].values))
    dur = bundle["ripples"].loc[bundle["ripples"].session == session]
    dur = dur.sort_values("t_peak_s")
    # durations are matched to the deduplicated peaks by nearest time
    idx = np.searchsorted(dur.t_peak_s.values, t)
    idx = np.clip(idx, 0, len(dur) - 1)
    dur = dur.duration_s.values[idx]

    rows = []
    for grid, g in ev.groupby("grid_no"):
        gi = g.set_index("state")
        gb = b[b.grid_no == grid].sort_values("rep_overall")
        if gb.empty:
            continue
        later = gb[gb.rep_overall > gb.rep_overall.min()]
        end_D = (float(later.t_A.iloc[0])
                 if len(later) and np.isfinite(later.t_A.iloc[0]) else np.nan)
        ts = {k: float(gi.t_s[k]) for k in STATES if k in gi.index}
        for i, k in enumerate(STATES):
            if k not in ts:
                continue
            t0 = ts[k]
            t1 = ts[STATES[i + 1]] if (i < 3 and STATES[i + 1] in ts) else end_D
            if not np.isfinite(t1) or t1 <= t0:
                continue
            lo, hi = np.searchsorted(t, t0), np.searchsorted(t, t1)
            for j in range(lo, hi):
                rows.append({"session": session, "t_peak_s": t[j],
                             "duration_s": dur[j], "cfg": gi.cfg[k],
                             "state": k, "grid_no": grid,
                             "press_t_s": t0, "interval_end_s": t1})
    cols = ["session", "t_peak_s", "duration_s", "cfg", "state", "grid_no",
            "press_t_s", "interval_end_s"]
    return pd.DataFrame(rows, columns=cols).reset_index(drop=True)


SCHEMES = {"window": ripples_near_events, "interval": ripples_in_intervals}
