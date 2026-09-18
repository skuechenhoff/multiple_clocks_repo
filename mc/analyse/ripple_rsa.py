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

import json
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

def default_bundle_dir(data_root=None):
    """Newest bundle on disk, preferring swr_v2 over the 2026-09-08 swr_v1."""
    base = os.path.join(_derivatives(data_root), "group", "swr")
    for name in ("bundle_v2", "bundle", "bundle_08.09.2026"):
        p = os.path.join(base, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "ripples.csv")):
            return p
    raise FileNotFoundError(f"no bundle under {base}")


def load_bundle(bundle_dir=None, pad_s=None, data_root=None):
    """The three bundle tables this analysis needs.

    `pad_s` re-imposes an artifact pad at home, as `meta.json` of the swr_v2
    bundle describes: events are kept only if `dist_to_artifact_s >= pad_s`.
    The bundle is detected at its smallest pad (0.1 s), so a larger pad is a
    pure subset and the sweep is nested.

    NOTE the meta.json caveat -- "doing only the first inflates the rate" --
    applies to ripple RATES, which need artifact-free exposure rebuilt from
    `artifact_intervals`. This analysis never divides by exposure: it measures
    each neuron's firing DURING a ripple. Filtering the events is therefore
    sufficient and complete here.
    """
    bundle_dir = bundle_dir or default_bundle_dir(data_root)
    rip = pd.read_csv(os.path.join(bundle_dir, "ripples.csv"))
    if pad_s is not None:
        if "dist_to_artifact_s" not in rip.columns:
            raise ValueError(f"{bundle_dir} has no dist_to_artifact_s; "
                             "pad sweeps need the swr_v2 bundle")
        rip = rip[rip.dist_to_artifact_s >= pad_s]
    return {
        "uncover": pd.read_csv(os.path.join(bundle_dir, "uncover.csv")),
        "ripples": rip.reset_index(drop=True),
        "behaviour": pd.read_csv(os.path.join(bundle_dir, "behaviour.csv")),
        "bundle_dir": bundle_dir, "pad_s": pad_s,
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
                         "grid_no": e.grid_no, "press_t_s": e.t_s,
                         "win_lo_s": e.t_s + window[0],
                         "win_hi_s": e.t_s + window[1]})
    cols = ["session", "t_peak_s", "duration_s", "cfg", "state", "grid_no",
            "press_t_s", "win_lo_s", "win_hi_s"]
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

def _surrogate_times(rip, rng):
    """A random time in the same admissible window as each ripple.

    The window is the one the ripple was selected from (`win_lo_s`,
    `win_hi_s`), so the surrogate keeps the event, the config, the state and
    the knowledge state, and differs in exactly one thing: it is not at a
    ripple. This is the primary null of the whole analysis -- "during a ripple,
    does the pattern become similar to the model" -- so it must be a matched
    window, not a shuffled label.
    """
    lo = rip.win_lo_s.values
    hi = rip.win_hi_s.values
    return lo + rng.uniform(0, 1, len(rip)) * (hi - lo)


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
            t = _surrogate_times(rip, surrogate_rng)
        half = (rip.duration_s.values / 2.0 if extent == "duration"
                else np.full(len(rip), fixed_half_s))
        width = 2 * half
        rates = np.empty((len(cells), len(rip)))
        for row, (_, c) in enumerate(cells.iterrows()):
            n = spike_counts_in_windows(spikes[s]["spikes"][int(c.cell)], t, half)
            rates[row] = n / width
        blk = {
            "session": s,
            "kind": "spikes",
            "rates": rates,
            "ci": np.array([CONFIG_LABELS.index(c) for c in rip.cfg]),
            "si": np.array([STATES.index(x) for x in rip.state]),
            "roi": cells.roi.to_numpy(),
        }
        if "phase" in rip.columns:
            blk["pi"] = np.array([PHASES.index(x) for x in rip.phase])
        out.append(blk)
    return out


def patterns_from_cache(cache, config_perm=None, split_halves=False,
                        drop_silent=None):
    """Rebuild a pattern pack from cached rates -- same output as
    `collect_spike_patterns`, but a permutation costs no spike lookups.

    `drop_silent=None` (the default) decides from the block's `kind`: on for
    spikes, off for HFB. It MUST be off for HFB. The test is
    `rates.sum() > 0`, which for a cell means "fired at least one spike", but
    for a robust-z HFB derivation means "was above its own session median more
    often than below" -- true for about half of them by construction. Applying
    it to HFB silently discarded ~50% of every ROI's derivations, which is what
    pushed mPFC (32 derivations) under MIN_CELLS_PER_PAIR and made it vanish
    from the HFB results table.
    """
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

        ds = (blk.get("kind", "spikes") == "spikes") if drop_silent is None \
            else drop_silent
        keep = np.ones(rates.shape[0], bool)
        if ds:
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
    is spent searching for the next reward. `window` caps the lag after the
    press (e.g. (0, 1) keeps only the first second), still clipped at the next
    discovery, so the two schemes are interchangeable at the call site.

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
            # `window` caps the lag after the press, still clipped at the next
            # discovery, so the interval never leaks into the next condition.
            w0 = t0 if window is None else t0 + window[0]
            w1 = t1 if window is None else min(t1, t0 + window[1])
            if w1 <= w0:
                continue
            lo, hi = np.searchsorted(t, w0), np.searchsorted(t, w1)
            for j in range(lo, hi):
                rows.append({"session": session, "t_peak_s": t[j],
                             "duration_s": dur[j], "cfg": gi.cfg[k],
                             "state": k, "grid_no": grid,
                             "press_t_s": t0, "interval_end_s": t1,
                             "win_lo_s": w0, "win_hi_s": w1})
    cols = ["session", "t_peak_s", "duration_s", "cfg", "state", "grid_no",
            "press_t_s", "interval_end_s", "win_lo_s", "win_hi_s"]
    return pd.DataFrame(rows, columns=cols).reset_index(drop=True)


SCHEMES = {"window": ripples_near_events, "interval": ripples_in_intervals}


# =============================================================================
# HFB VARIANT -- same RSA, continuous power instead of spikes
# =============================================================================
#
# Why this is the better-powered version of the same question:
#   * no spike-count floor. A single unit contributes ~0.2 spikes to a ripple;
#     an HFB derivation contributes a real-valued power estimate.
#   * more independent sites. The 65 mPFC units sit on ~8 microwire bundles;
#     mPFC HFB has 32 derivations across 17 sessions.
#   * a proper negative control exists: 81 Visual derivations.
# The cost: HFB is a population proxy, not spiking, and a derivation is a much
# coarser spatial sample than a unit.
#
# CIRCULARITY: ripples are detected ON the hippocampal derivations, so
# hippocampal HFB at its own ripple times is partly the ripple itself
# (80-120 Hz sits inside the 70-150 Hz HFB band). Hippocampal HFB rows are
# therefore a positive control at best, never a result -- `hfb_roi_table`
# flags them with `is_ripple_source`.

HFB_BANDS = ("hfb", "ripple", "theta", "beta")


def hfb_roi_table(bundle_dir=None, sessions=None, data_root=None,
                  drop_excluded=True):
    """One row per usable HFB derivation: session, pair_id, ROI, MNI, role."""
    bundle_dir = bundle_dir or default_bundle_dir(data_root)
    sessions = DSR_SESSIONS if sessions is None else sessions
    p = pd.read_csv(os.path.join(bundle_dir, "hfb_pairs.csv"))
    if drop_excluded and "excluded" in p.columns:
        p = p[~p.excluded.astype(bool)]
    p = p[p.session.isin(sessions)].copy()
    p["is_ripple_source"] = p.get("role", pd.Series(index=p.index)).eq("ripple")
    return p.rename(columns={"pair_roi": "roi"})[
        ["session", "pair_id", "roi", "role", "is_ripple_source",
         "mni_x", "mni_y", "mni_z"]].reset_index(drop=True)


def load_hfb(session, bundle_dir=None, band="hfb", data_root=None):
    """(pair_ids, signal, fs) for one session. Signal is (n_pairs, n_samples).

    Stored as float16 at 100 Hz on the SAME session clock as behaviour and
    ripple times: sample i is t = i / fs seconds. Returned as float32 --
    float16 cannot be z-scored without losing precision.
    """
    bundle_dir = bundle_dir or default_bundle_dir(data_root)
    z = np.load(os.path.join(bundle_dir, "hfb", f"s{int(session):02d}_hfb.npz"),
                allow_pickle=True)
    return ([str(x) for x in z["pair_ids"]], z[band].astype(np.float32),
            float(z["out_fs"]))


def _robust_z(x):
    """Median / IQR z-score, per derivation, over the whole session.

    Derivations differ in impedance and gain by orders of magnitude, so raw
    power is not comparable across them and the RDM would be dominated by a
    handful of loud channels. Median and IQR rather than mean and SD because
    the tail is artifact.
    """
    med = np.median(x, axis=1, keepdims=True)
    q1, q3 = np.percentile(x, [25, 75], axis=1, keepdims=True)
    iqr = np.maximum(q3 - q1, 1e-9)
    return (x - med) / iqr


def cache_hfb_rates(bundle, events, hfb_tab, window, scheme="window",
                    band="hfb", bundle_dir=None, extent="duration",
                    surrogate_rng=None, data_root=None, verbose=True):
    """Mean HFB power inside each ripple, per derivation -- the HFB analogue of
    `cache_ripple_rates`, and interchangeable with it downstream.

    Returns the same block structure, so `patterns_from_cache`, `rdm_for`,
    `fit_rho` and every null work unchanged.
    """
    get = SCHEMES[scheme]
    out = []
    for s in sorted(set(hfb_tab.session)):
        rip = get(bundle, s, events, window)
        rows = hfb_tab[hfb_tab.session == s]
        if rip.empty or rows.empty:
            continue
        try:
            pair_ids, sig, fs = load_hfb(s, bundle_dir, band, data_root)
        except FileNotFoundError:
            if verbose:
                print(f"  s{s}: no HFB file")
            continue
        idx = {p: i for i, p in enumerate(pair_ids)}
        keep = [(i, p) for i, p in enumerate(rows.pair_id) if p in idx]
        if not keep:
            continue
        sig = _robust_z(sig[[idx[p] for _, p in keep]])

        t = rip.t_peak_s.values.copy()
        if surrogate_rng is not None:
            t = _surrogate_times(rip, surrogate_rng)
        half = (rip.duration_s.values / 2.0 if extent == "duration"
                else np.full(len(rip), 0.010))

        a = np.clip(np.round((t - half) * fs).astype(int), 0, sig.shape[1] - 1)
        b = np.clip(np.round((t + half) * fs).astype(int) + 1, 1, sig.shape[1])
        power = np.empty((sig.shape[0], len(t)), np.float32)
        cs = np.cumsum(sig, axis=1)
        cs = np.concatenate([np.zeros((sig.shape[0], 1), np.float32), cs], 1)
        n = np.maximum(b - a, 1)
        power[:] = (cs[:, b] - cs[:, a]) / n          # windowed mean, O(1) each

        blk = {
            "session": s, "kind": "hfb", "rates": power,
            "ci": np.array([CONFIG_LABELS.index(c) for c in rip.cfg]),
            "si": np.array([STATES.index(x) for x in rip.state]),
            "roi": rows.iloc[[i for i, _ in keep]].roi.to_numpy(),
        }
        if "phase" in rip.columns:
            blk["pi"] = np.array([PHASES.index(x) for x in rip.phase])
        out.append(blk)
    if verbose:
        tot = sum(b["rates"].shape[1] for b in out)
        nder = sum(b["rates"].shape[0] for b in out)
        print(f"  HFB[{band}]: {tot} ripples, {nder} derivations, "
              f"{len(out)} sessions")
    return out


# =============================================================================
# STANDING DECISIONS -- the choices this pipeline has already settled
# =============================================================================
#
# Recorded here rather than in each script so that every entry point inherits
# the same choice and a change has to be made once, visibly.

DECISIONS = {
    "pad_s": 0.25,
    "pad_why": (
        "Artifact pad swept over 0.1/0.25/0.5/0.75/1.0 s. The pad is a near-"
        "uniform ~30% thinning of ripples (Spearman lag vs dist_to_artifact "
        "= +0.005, p = 0.69) and a random 33% thinning moves a single ROI's "
        "rho over a range of ~1.0 (SD 0.20), i.e. the pad dependence IS "
        "resampling noise. 0.25 s is fixed a priori so no result can pick it."),
    "primary_null": "surrogate_window",
    "primary_null_why": (
        "The question is what is encoded DURING a ripple, so the comparison "
        "must be ripple vs not-ripple. The surrogate window keeps the event, "
        "the config, the state, the knowledge state, the number of windows and "
        "the cells, and moves only the window off the ripple peak. A label "
        "shuffle answers a different question and is kept as a secondary "
        "check."),
    "scheme": "interval",
    "window_s": (0.0, 1.0),
    "window_why": (
        "Ripples in the first second after the uncover press, clipped at the "
        "next discovery so nothing leaks between conditions. The time-resolved "
        "sweep showed the fit stops growing at ~1 s, and the ripple-HFB "
        "literature this is being compared against uses -1 to 1 s."),
    "extent": "duration",
    "extent_why": "firing/power is read inside the ripple, t_peak +- duration/2.",
    "sign": "both matrices are dissimilarities, so POSITIVE rho = model encoded",
    "min_cells_per_pair": MIN_CELLS_PER_PAIR,
    "normalise": "zscore",
    "normalise_why": (
        "Features are z-scored across conditions before the correlation "
        "distance, matching `_z_score_per_neuron` in mc/analyse/"
        "rsa_perm_rdms.py -- the convention already used by the other RSAs in "
        "this project. Centring alone (what this pipeline shipped until "
        "2026-09-17) leaves a feature's influence proportional to its "
        "across-condition variance, and firing rates span two orders of "
        "magnitude: effective n was 36-43% of the actual cell count, with the "
        "top 5 cells carrying up to 43% of an ROI's RDM. The choice is made "
        "on prior project convention, NOT on which normalisation gives the "
        "nicer answer -- it does change which ROI looks best, and that "
        "sensitivity is reported in ripple_rsa_inputs_*/."),
    "seed": 42,
}


# =============================================================================
# EXPANDED CONDITION SPACE -- config x state
# =============================================================================
#
# The 8-config RDM has 28 unique pairs, so chance rho has SD = 1/sqrt(27) =
# 0.192 no matter how many ripples go in: the noise floor is arithmetic, not
# statistical, and more data cannot lower it. More CONDITIONS can. Treating
# each (config, state) as its own condition gives 32 conditions and 496 pairs,
# a floor of 1/sqrt(495) = 0.045.
#
# Two pair families have to be handled, not merged blindly:
#   within-state  (k == k')   -- the old 8-condition analysis, four times over.
#   cross-state   (k != k')   -- new, and where the extra power comes from.
# Cross-state pairs of the SAME config are dropped: those two conditions come
# from the same grid a few seconds apart, so any slow drift in firing makes
# them look alike, and that is exactly the pattern `full_abcd` predicts. They
# are 48 of the 496 and removing them costs little.
# The state model (|k - k'|) is partialled out of every fit, because the known
# set grows with k and firing drifts with time-on-task, so a region that only
# tracked "how far into the grid am I" would otherwise fit `known_set`.

N_STATE = len(STATES)
N_COND = N_CONFIG * N_STATE
# condition index is state-major: i = k * N_CONFIG + c
COND_CFG = np.tile(np.arange(N_CONFIG), N_STATE)
COND_STATE = np.repeat(np.arange(N_STATE), N_CONFIG)
COND_LABELS = [f"{CONFIG_LABELS[c]}|{STATES[k]}"
               for k, c in zip(COND_STATE, COND_CFG)]


def flat_patterns(pack):
    """(n_cells, 32) from a pack's (n_cells, 8, 4), in COND order."""
    P = pack["patterns"]
    return P.transpose(0, 2, 1).reshape(P.shape[0], N_COND)


def normalise_features(P, mode="zscore"):
    """Per-feature normalisation across conditions, project convention.

    "centre"  subtract each feature's mean across conditions.
    "zscore"  additionally divide by its across-condition SD, so every feature
              contributes equally to the correlation regardless of firing rate.

    The zscore branch reproduces `_z_score_per_neuron` in
    mc/analyse/rsa_perm_rdms.py exactly, INCLUDING its handling of a constant
    feature: SD 0 is replaced by 1.0 rather than NaN, so such a feature stays
    an all-zero column instead of dropping out. Matching that matters -- an
    all-zero column is not inert in a pairwise correlation.
    """
    P = np.asarray(P, float)
    C = P - np.nanmean(P, axis=1, keepdims=True)
    if mode == "centre":
        return C
    if mode != "zscore":
        raise ValueError(f"unknown normalisation {mode!r}")
    sd = np.nanstd(C, axis=1, keepdims=True)
    return C / np.where(sd > 0, sd, 1.0)


def build_rdm_flat(P, min_cells=MIN_CELLS_PER_PAIR, normalise=None):
    """Correlation-distance RDM over an arbitrary number of conditions.

    Per-feature normalisation across conditions (see `normalise_features`;
    default from DECISIONS), then pairwise-complete correlation distance.
    Written separately from `build_rdm` so the 8-condition results stay
    bit-identical.
    """
    P = np.asarray(P, float)
    n = P.shape[1]
    C = normalise_features(P, normalise or DECISIONS["normalise"])
    rdm = np.full((n, n), np.nan)
    n_used = np.zeros((n, n), int)
    np.fill_diagonal(rdm, 0.0)
    ok_all = np.isfinite(C)
    for i in range(n):
        for j in range(i + 1, n):
            ok = ok_all[:, i] & ok_all[:, j]
            n_used[i, j] = n_used[j, i] = ok.sum()
            if ok.sum() < min_cells:
                continue
            a, b = C[ok, i], C[ok, j]
            if a.std() == 0 or b.std() == 0:
                continue
            rdm[i, j] = rdm[j, i] = 1.0 - np.corrcoef(a, b)[0, 1]
    return rdm, n_used


def rdm_for_flat(pack, roi, min_cells=MIN_CELLS_PER_PAIR, normalise=None):
    """32 x 32 data RDM for one ROI, all four states at once."""
    sel = np.asarray(pack["roi"]) == roi
    P = np.asarray(flat_patterns(pack), float)[sel]
    return build_rdm_flat(P, min_cells, normalise)


# ---------------------------------------------------------------- models

def _cond_known(i):
    c, k = COND_CFG[i], COND_STATE[i]
    return set(CONFIGS[c][:k + 1])


def current_location_rdm():
    """Is the reward just uncovered in the same place? 0 = same, 1 = different.

    Within a state this is constant (the 8 configs put every state in a
    different location by counterbalancing), so it is carried entirely by the
    cross-state pairs -- which is the reason to build the 32-condition space.
    """
    loc = np.array([CONFIGS[c][k] for c, k in zip(COND_CFG, COND_STATE)])
    return (loc[:, None] != loc[None, :]).astype(float)


def known_set_rdm_flat():
    """Jaccard distance between the sets of locations known so far."""
    S = [_cond_known(i) for i in range(N_COND)]
    M = np.zeros((N_COND, N_COND))
    for i in range(N_COND):
        for j in range(N_COND):
            M[i, j] = 1.0 - len(S[i] & S[j]) / len(S[i] | S[j])
    return M


def full_abcd_rdm_flat():
    """Jaccard distance between the whole configurations -- state-invariant."""
    S = [set(CONFIGS[c]) for c in COND_CFG]
    M = np.zeros((N_COND, N_COND))
    for i in range(N_COND):
        for j in range(N_COND):
            M[i, j] = 1.0 - len(S[i] & S[j]) / len(S[i] | S[j])
    return M


def state_rdm_flat():
    """|k - k'| -- the nuisance that is partialled out of every fit."""
    return np.abs(COND_STATE[:, None] - COND_STATE[None, :]).astype(float) / 3.0


def model_rdms_flat():
    return {"current_location": current_location_rdm(),
            "known_set": known_set_rdm_flat(),
            "full_abcd": full_abcd_rdm_flat()}


NUISANCE_FLAT = "state"


def pair_mask(family="all", drop_same_config=True):
    """Boolean mask over the upper triangle of the 32 x 32 RDM."""
    iu = np.triu_indices(N_COND, 1)
    ki, kj = COND_STATE[iu[0]], COND_STATE[iu[1]]
    ci, cj = COND_CFG[iu[0]], COND_CFG[iu[1]]
    m = np.ones(len(iu[0]), bool)
    if family == "within_state":
        m &= ki == kj
    elif family == "cross_state":
        m &= ki != kj
    elif family != "all":
        raise ValueError(family)
    if drop_same_config:
        m &= ~((ci == cj) & (ki != kj))
    return iu, m


def _partial_rank(d, m, z):
    """Spearman partial correlation of d and m given z, on matched vectors."""
    R = np.vstack([stats.rankdata(v) for v in (d, m, z)]).astype(float)
    R -= R.mean(axis=1, keepdims=True)
    sd = R.std(axis=1)
    if np.any(sd == 0):
        return np.nan
    d_, m_, z_ = R
    d_ = d_ - (d_ @ z_) / (z_ @ z_) * z_
    m_ = m_ - (m_ @ z_) / (z_ @ z_) * z_
    if d_.std() == 0 or m_.std() == 0:
        return np.nan
    return float(np.corrcoef(d_, m_)[0, 1])


def fit_rho_flat(rdm, model, family="all", partial=True,
                 min_pairs=MIN_PAIRS_TO_FIT):
    """Spearman (partial) correlation between a 32 x 32 data RDM and a model.

    Both matrices are dissimilarities, so a region encoding the model gives a
    POSITIVE rho.
    """
    iu, m = pair_mask(family)
    d = rdm[iu][m]
    mv = model[iu][m]
    zv = state_rdm_flat()[iu][m]
    ok = np.isfinite(d) & np.isfinite(mv)
    if ok.sum() < min_pairs or np.unique(mv[ok]).size < 2:
        return np.nan
    if not partial:
        if np.unique(d[ok]).size < 2:
            return np.nan
        return float(stats.spearmanr(d[ok], mv[ok]).correlation)
    if np.unique(zv[ok]).size < 2:          # within-state: nothing to partial
        return float(stats.spearmanr(d[ok], mv[ok]).correlation)
    return _partial_rank(d[ok], mv[ok], zv[ok])


def relabel_flat(model, perm):
    """Apply a permutation of the 8 configs to a 32-condition model RDM.

    The configs are relabelled consistently across states, so the state
    structure -- and therefore the nuisance being partialled out -- is
    untouched. Permuting the MODEL rather than the data keeps the observed-pair
    mask fixed, which is what makes partial RDMs fittable.
    """
    idx = COND_STATE * N_CONFIG + np.asarray(perm)[COND_CFG]
    return model[np.ix_(idx, idx)]


# =============================================================================
# STORING NULLS -- so a model change does not cost another surrogate run
# =============================================================================
#
# A surrogate draw is expensive for the reason that it is the right null: it
# re-runs the WHOLE estimator, which for HFB means re-reading a session's
# 100 Hz signal off disk (~6 s per draw, ~50 min for 500). But almost all of
# that work is independent of the model being fitted. The expensive part ends
# at the pattern pack -- (n_features, 8, 4) mean rate per condition -- and
# everything after it (RDM, model, family, partialling) is milliseconds.
#
# So the packs are stored. A new model, a new pair family, a different
# nuisance, a different min_cells: all re-fittable in seconds against the
# identical stored null.
#
# WHAT INVALIDATES A STORED NULL: anything upstream of the pack -- the pad,
# the bundle, the ripple->condition scheme, the window, the extent, the ROI
# table, the surrogate definition. Those are recorded in the sidecar `meta`
# and `load_packs` refuses to return packs whose meta does not match what the
# caller expects, rather than silently mixing two nulls.

def save_packs(path, packs, meta):
    """Store a list of pattern packs (one per surrogate draw) plus its meta.

    Packs are ragged -- `drop_silent` can keep a different number of cells in
    different draws -- so patterns and roi go in as object arrays.
    """
    np.savez_compressed(
        path,
        patterns=np.array([p["patterns"].astype(np.float32) for p in packs],
                          dtype=object),
        roi=np.array([p["roi"] for p in packs], dtype=object),
        session=np.array([p["session"] for p in packs], dtype=object),
        counts=np.array([p["counts"] for p in packs], dtype=object),
        meta=json.dumps(meta))


def load_packs(path, expect=None):
    """Return (packs, meta). `expect` is checked key by key against `meta`."""
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    if expect:
        bad = {k: (meta.get(k), v) for k, v in expect.items()
               if meta.get(k) != v}
        if bad:
            raise ValueError(
                f"{os.path.basename(path)} was built with different settings, "
                f"so its null does not apply here: {bad}")
    packs = [{"patterns": p, "roi": r, "session": s, "counts": c}
             for p, r, s, c in zip(z["patterns"], z["roi"], z["session"],
                                   z["counts"])]
    return packs, meta


def null_meta(decisions=None, **extra):
    """The settings a stored null is only valid for."""
    d = dict(decisions or DECISIONS)
    m = {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()
         if k in ("pad_s", "scheme", "window_s", "extent",
                  "min_cells_per_pair", "seed")}
    m.update(extra)
    return m


# =============================================================================
# WHICH CONDITIONS CARRY THE FIT
# =============================================================================

def loco_flat(rdm, model, family="all", partial=True):
    """Leave-one-condition-out contribution of each of the 32 conditions.

    `delta[i] = rho(all conditions) - rho(without condition i)`.

    POSITIVE delta = dropping the condition HURTS, i.e. it was carrying the
    fit. Negative delta = dropping it HELPS, i.e. it was working against the
    model. Near zero = it contributed nothing either way, which for a condition
    with almost no ripples is what you would expect and is the cheapest way to
    see whether the fit rests on a handful of well-sampled conditions.
    """
    iu, base = pair_mask(family)
    full = fit_rho_flat(rdm, model, family=family, partial=partial)
    d = rdm[iu]
    mv = model[iu]
    zv = state_rdm_flat()[iu]
    out = np.full(N_COND, np.nan)
    for i in range(N_COND):
        keep = base & (iu[0] != i) & (iu[1] != i)
        ok = keep & np.isfinite(d) & np.isfinite(mv)
        if ok.sum() < MIN_PAIRS_TO_FIT or np.unique(mv[ok]).size < 2:
            continue
        if partial and np.unique(zv[ok]).size > 1:
            r = _partial_rank(d[ok], mv[ok], zv[ok])
        else:
            r = float(stats.spearmanr(d[ok], mv[ok]).correlation)
        out[i] = full - r
    return out, full


def loco_configs(rdm, model, min_pairs=MIN_PAIRS_TO_FIT):
    """The same, for one 8-config RDM at one state. `delta[c]` over 8 configs."""
    iu = np.triu_indices(N_CONFIG, 1)
    d, mv = rdm[iu], model[iu]
    ok0 = np.isfinite(d) & np.isfinite(mv)
    if ok0.sum() < min_pairs or np.unique(mv[ok0]).size < 2:
        return np.full(N_CONFIG, np.nan), np.nan
    full = float(stats.spearmanr(d[ok0], mv[ok0]).correlation)
    out = np.full(N_CONFIG, np.nan)
    for c in range(N_CONFIG):
        ok = ok0 & (iu[0] != c) & (iu[1] != c)
        if ok.sum() < min_pairs or np.unique(mv[ok]).size < 2 \
                or np.unique(d[ok]).size < 2:
            continue
        out[c] = full - float(stats.spearmanr(d[ok], mv[ok]).correlation)
    return out, full


def condition_coverage(pack, roi):
    """Per-condition sampling: ripples, features with data, estimable pairs."""
    sel = pack["roi"] == roi
    P = flat_patterns(pack)[sel]
    rdm, n_used = build_rdm_flat(P)
    iu = np.triu_indices(N_COND, 1)
    est = np.zeros(N_COND, int)
    for a, b in zip(*iu):
        if np.isfinite(rdm[a, b]):
            est[a] += 1
            est[b] += 1
    return pd.DataFrame({
        "cond": np.arange(N_COND), "label": COND_LABELS,
        "config": COND_CFG, "state": [STATES[k] for k in COND_STATE],
        "n_ripples": pack["counts"].sum(axis=0).T.reshape(-1),
        "n_features_with_data": np.isfinite(P).sum(axis=0),
        "n_pairs_estimable": est,
        "median_cells_per_pair": np.nanmedian(
            np.where(np.eye(N_COND, dtype=bool), np.nan,
                     n_used.astype(float)), axis=1)})


# =============================================================================
# CROSSNOBIS -- cross-validated Mahalanobis distance
# =============================================================================
#
# Follows the project's existing recipe in
# scripts/RSA_human_cells_DSR_crossnobis.py: per SESSION, estimate the noise
# covariance from residuals, shrink toward the identity, take a
# leave-one-fold-out cross-validated Mahalanobis distance, then average the
# per-session RDMs. Per session is not a stylistic choice -- cells in
# different sessions are never recorded together, so their covariance is not
# estimable and a pooled full-Sigma crossnobis does not exist.
#
# Why bother: crossnobis is the principled answer to "which features get
# weighted". Correlation distance after centring weights a feature by its
# across-condition variance (so loud cells dominate); after z-scoring it
# weights every feature equally (so quiet noisy cells are amplified).
# Crossnobis divides by the NOISE, estimated from how much a feature varies
# between ripples WITHIN a condition, so a feature counts in proportion to its
# signal-to-noise. It is also cross-validated, hence unbiased: the expected
# distance between two conditions that do not differ is 0, not positive, and
# individual entries may legitimately be negative.
#
# FEASIBILITY IS THE BINDING CONSTRAINT HERE. Crossnobis needs at least
# `n_folds` ripples per condition per session. Measured on the pad-0.25,
# 0-1 s interval window:
#     32 conditions (config x state): median 1 ripple per (session, condition);
#         0 of 27 sessions have >=2 in every condition. NOT ESTIMABLE.
#      8 conditions (config, states pooled): median 5; 17 of 27 sessions
#         complete. Estimable.
# So crossnobis runs on the 8-config space, where the only fittable model is
# `full_abcd`. It is a different, weaker question than the 32-condition RSA --
# not a drop-in replacement for it.

CROSSNOBIS_SHRINKAGE = 0.1        # as SHRINKAGE_ALPHA in the DSR script


def _fold_means(rates, cond, n_cond, n_folds):
    """X[cond, fold, cell] and the residuals used for the noise covariance.

    Folds interleave ripples in their recorded order (i % n_folds), which is
    deterministic and balanced -- no RNG, so a surrogate draw and the observed
    value are folded the same way.
    """
    n_cells = rates.shape[0]
    X = np.full((n_cond, n_folds, n_cells), np.nan)
    resid = []
    for c in range(n_cond):
        idx = np.flatnonzero(cond == c)
        if len(idx) < n_folds:
            continue
        f = np.arange(len(idx)) % n_folds
        for k in range(n_folds):
            take = idx[f == k]
            if not len(take):
                break
            m = rates[:, take].mean(axis=1)
            X[c, k] = m
            resid.append(rates[:, take] - m[:, None])
    R = np.concatenate(resid, axis=1).T if resid else np.zeros((0, n_cells))
    return X, R


def _crossnobis_from_X(X, sigma_inv, per_neuron=True):
    """Leave-one-fold-out crossnobis over the conditions that are estimable."""
    n_cond, K, n_cells = X.shape
    ok = np.flatnonzero(np.isfinite(X).all(axis=(1, 2)))
    out = np.full((n_cond, n_cond), np.nan)
    if len(ok) < 2:
        return out, ok
    Xo = X[ok]
    acc = np.zeros((len(ok), len(ok)))
    for k in range(K):
        A = Xo[:, k, :]
        B = Xo[:, np.delete(np.arange(K), k), :].mean(axis=1)
        A_S = A @ sigma_inv
        AB = A_S @ B.T
        d = np.einsum("ij,ij->i", A_S, B)
        acc += d[:, None] + d[None, :] - AB - AB.T
    acc /= K
    if per_neuron:
        # DELIBERATE DEVIATION from the DSR script, which averages raw
        # per-session RDMs. A whitened d^2 grows with the number of neurons,
        # and ROI cell counts here range from 3 to 40+ per session, so without
        # this the biggest session would dominate the average.
        acc /= n_cells
    out[np.ix_(ok, ok)] = acc
    return out, ok


def crossnobis_rdm(cache, roi, conditions="config", n_folds=2,
                   shrinkage=CROSSNOBIS_SHRINKAGE, per_neuron=True):
    """Crossnobis RDM for one ROI, averaged over the sessions that support it.

    `conditions` is "config" (8, states pooled -- the estimable space) or
    "config_state" (32; kept so the infeasibility is reproducible rather than
    asserted). Returns (rdm, n_sessions_per_pair).
    """
    n_cond = N_CONFIG if conditions == "config" else N_COND
    mats = []
    for blk in cache:
        sel = np.asarray(blk["roi"]) == roi
        if sel.sum() < 2:
            continue
        rates = np.asarray(blk["rates"], float)[sel]
        cond = (blk["ci"] if conditions == "config"
                else blk["si"] * N_CONFIG + blk["ci"])
        X, R = _fold_means(rates, cond, n_cond, n_folds)
        if R.shape[0] < 2 or not np.isfinite(X).all(axis=(1, 2)).sum() >= 2:
            continue
        S = np.cov(R, rowvar=False)
        S = np.atleast_2d(S)
        S = (1 - shrinkage) * S + shrinkage * np.eye(S.shape[0])
        rdm, _ = _crossnobis_from_X(X, np.linalg.pinv(S), per_neuron)
        if np.isfinite(rdm).any():
            mats.append(rdm)
    if not mats:
        return np.full((n_cond, n_cond), np.nan), np.zeros((n_cond, n_cond), int)
    M = np.stack(mats)
    n_sess = np.isfinite(M).sum(axis=0)
    with np.errstate(invalid="ignore"):
        rdm = np.nanmean(M, axis=0)
    rdm[n_sess == 0] = np.nan
    np.fill_diagonal(rdm, 0.0)
    return rdm, n_sess


def crossnobis_feasibility(cache, rois, n_folds=2):
    """How many (session, condition) cells carry enough ripples to fold."""
    rows = []
    for name, n_cond, key in (("config", N_CONFIG, "ci"),
                              ("config_state", N_COND, None)):
        for roi in rois:
            per_sess, complete = [], 0
            for blk in cache:
                if (np.asarray(blk["roi"]) == roi).sum() < 2:
                    continue
                cond = (blk["ci"] if key else blk["si"] * N_CONFIG + blk["ci"])
                n = np.bincount(cond, minlength=n_cond)
                per_sess.append(n)
                complete += int((n >= n_folds).all())
            if not per_sess:
                continue
            A = np.array(per_sess)
            rows.append({"conditions": name, "roi": roi, "n_sessions": len(A),
                         "median_ripples_per_cell": float(np.median(A)),
                         "pct_cells_foldable": float(100 * (A >= n_folds).mean()),
                         "n_sessions_complete": complete})
    return pd.DataFrame(rows)


# =============================================================================
# ALL UNCOVERS -- not just the explore-phase discoveries
# =============================================================================
#
# The discovery-only event set is 2665 uncovers and 1185 ripples. Every
# correct uncover on the 8 shared configs is 32688 uncovers and 12902 ripples,
# 10.9x more. That matters for two reasons:
#   * the 32-condition RDM goes from ~37 to ~400 ripples per condition;
#   * crossnobis becomes ESTIMABLE on the 32-condition space (~14 ripples per
#     session x condition, against a median of 1 for discoveries only), so the
#     estimator question can finally be asked where the design is strong.
#
# The cost is that a repeat is not a discovery. The subject already knows the
# configuration, the ripple-rate increase was established on discoveries, and
# firing differs between the two. So the phase is never silently pooled: the
# 64-condition space keeps it as its own factor, and phase is partialled out
# of every fit in the collapsed space.

PHASES = ["first", "repeat"]


def all_uncover_events(bundle, sessions=None, phases=None):
    """Every CORRECT uncover with a state, on the 8 shared configs.

    `is_discovery == 1` is exactly `rep_overall == 1` in this table, so phase
    is "first" for the initial traversal of a grid and "repeat" afterwards.
    """
    sessions = DSR_SESSIONS if sessions is None else sessions
    u = bundle["uncover"]
    ev = u[(u.session.isin(sessions)) & (u.correct == 1)
           & u.state.notna()].copy()

    beh = bundle["behaviour"]
    cfg = beh[beh.session.isin(sessions)].drop_duplicates(["session", "grid_no"])
    cfg = cfg[["session", "grid_no", "loc_A", "loc_B", "loc_C", "loc_D"]].copy()
    cfg["cfg"] = (cfg[["loc_A", "loc_B", "loc_C", "loc_D"]]
                  .astype(int).astype(str).agg("-".join, axis=1))
    ev = ev.merge(cfg[["session", "grid_no", "cfg"]],
                  on=["session", "grid_no"], how="left")
    ev = ev[ev.cfg.isin(CONFIG_LABELS)].copy()
    ev["phase"] = np.where(ev.is_discovery == 1, "first", "repeat")
    if phases is not None:
        ev = ev[ev.phase.isin(phases)]
    return ev[["session", "grid_no", "rep_overall", "t_s", "state", "cfg",
               "phase"]].reset_index(drop=True)


def ripples_after_uncovers(bundle, session, events, window):
    """Ripples after each uncover, clipped at the NEXT uncover of that session.

    Generalises `ripples_in_intervals` to repeats. The interval logic there
    walks A->B->C->D within one traversal, which assumes one row per state per
    grid; with repeats there are many. Here the cap is simply the next correct
    uncover in the session, whichever grid or repeat it belongs to, so the
    intervals tile the session exactly once and no ripple can be counted in
    two conditions.
    """
    t = np.sort(swu.dedup_ripples(
        bundle["ripples"].loc[bundle["ripples"].session == session,
                              "t_peak_s"].values, tol_s=DEDUP_S))
    d = bundle["ripples"].loc[bundle["ripples"].session == session]
    d = d.sort_values("t_peak_s")
    if not len(t):
        return pd.DataFrame(columns=[
            "session", "t_peak_s", "duration_s", "cfg", "state", "phase",
            "grid_no", "press_t_s", "win_lo_s", "win_hi_s"])
    idx = np.clip(np.searchsorted(d.t_peak_s.values, t), 0, len(d) - 1)
    dur = d.duration_s.values[idx]

    ev = events[events.session == session].sort_values("t_s")
    ut = ev.t_s.values.astype(float)
    nxt = np.r_[ut[1:], np.inf]
    lo = ut + window[0]
    hi = np.minimum(nxt, ut + window[1])

    a, b = np.searchsorted(t, lo), np.searchsorted(t, hi)
    keep = b > a
    rows = []
    for k in np.flatnonzero(keep):
        e = ev.iloc[k]
        for j in range(a[k], b[k]):
            rows.append({"session": session, "t_peak_s": t[j],
                         "duration_s": dur[j], "cfg": e.cfg, "state": e.state,
                         "phase": e.phase, "grid_no": e.grid_no,
                         "press_t_s": ut[k], "win_lo_s": lo[k],
                         "win_hi_s": hi[k]})
    cols = ["session", "t_peak_s", "duration_s", "cfg", "state", "phase",
            "grid_no", "press_t_s", "win_lo_s", "win_hi_s"]
    return pd.DataFrame(rows, columns=cols).reset_index(drop=True)


SCHEMES["uncovers"] = ripples_after_uncovers


# =============================================================================
# 64-CONDITION SPACE -- config x state x phase
# =============================================================================

N_PHASE = len(PHASES)
N_COND64 = N_CONFIG * N_STATE * N_PHASE
# index is phase-major, then state, then config: i = p*32 + k*8 + c
C64_CFG = np.tile(np.arange(N_CONFIG), N_STATE * N_PHASE)
C64_STATE = np.tile(np.repeat(np.arange(N_STATE), N_CONFIG), N_PHASE)
C64_PHASE = np.repeat(np.arange(N_PHASE), N_CONFIG * N_STATE)
COND64_LABELS = [f"{CONFIG_LABELS[c]}|{STATES[k]}|{PHASES[p]}"
                 for p, k, c in zip(C64_PHASE, C64_STATE, C64_CFG)]


def _known64(i):
    """What the subject knows at condition i.

    On a REPEAT the subject has already completed the grid, so the known set
    is the whole configuration regardless of which reward is being uncovered.
    This is what makes `known_set` and `full_abcd` differ only in the `first`
    half of the 64-condition space -- and it is the reason splitting by phase
    is informative rather than merely doubling the conditions.
    """
    c, k, p = C64_CFG[i], C64_STATE[i], C64_PHASE[i]
    return set(CONFIGS[c]) if PHASES[p] == "repeat" else set(CONFIGS[c][:k + 1])


def _jaccard(sets):
    n = len(sets)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = 1.0 - len(sets[i] & sets[j]) / len(sets[i] | sets[j])
    return M


def model_rdms_64():
    loc = np.array([CONFIGS[c][k] for c, k in zip(C64_CFG, C64_STATE)])
    return {"current_location": (loc[:, None] != loc[None, :]).astype(float),
            "known_set": _jaccard([_known64(i) for i in range(N_COND64)]),
            "full_abcd": _jaccard([set(CONFIGS[c]) for c in C64_CFG])}


def nuisance_rdms_64():
    return {"state": np.abs(C64_STATE[:, None] - C64_STATE[None, :]) / 3.0,
            "phase": (C64_PHASE[:, None] != C64_PHASE[None, :]).astype(float)}


def pair_mask_64(family="all", drop_same_config_cross_state=True):
    """Upper-triangle mask over the 64 x 64 RDM.

    Families are about PHASE here: "within_phase" compares first with first
    and repeat with repeat; "cross_phase" compares the two.
    """
    iu = np.triu_indices(N_COND64, 1)
    pi, pj = C64_PHASE[iu[0]], C64_PHASE[iu[1]]
    ki, kj = C64_STATE[iu[0]], C64_STATE[iu[1]]
    ci, cj = C64_CFG[iu[0]], C64_CFG[iu[1]]
    m = np.ones(len(iu[0]), bool)
    if family == "within_phase":
        m &= pi == pj
    elif family == "first_only":
        m &= (pi == 0) & (pj == 0)
    elif family == "repeat_only":
        m &= (pi == 1) & (pj == 1)
    elif family == "cross_phase":
        m &= pi != pj
    elif family != "all":
        raise ValueError(family)
    if drop_same_config_cross_state:
        m &= ~((ci == cj) & (ki != kj))
    return iu, m


def _partial_rank_multi(d, m, Z):
    """Spearman partial correlation of d and m given several nuisances."""
    R = np.vstack([stats.rankdata(v) for v in ([d, m] + list(Z))]).astype(float)
    R -= R.mean(axis=1, keepdims=True)
    if np.any(R.std(axis=1) == 0):
        return np.nan
    dv, mv, N = R[0], R[1], R[2:].T
    beta, *_ = np.linalg.lstsq(N, np.vstack([dv, mv]).T, rcond=None)
    res = np.vstack([dv, mv]).T - N @ beta
    if res[:, 0].std() == 0 or res[:, 1].std() == 0:
        return np.nan
    return float(np.corrcoef(res[:, 0], res[:, 1])[0, 1])


def fit_rho_64(rdm, model, family="all", partial=True,
               min_pairs=MIN_PAIRS_TO_FIT):
    """Partial Spearman on the 64-condition space, state AND phase removed."""
    iu, m = pair_mask_64(family)
    d, mv = rdm[iu][m], model[iu][m]
    nz = [n[iu][m] for n in nuisance_rdms_64().values()]
    ok = np.isfinite(d) & np.isfinite(mv)
    if ok.sum() < min_pairs or np.unique(mv[ok]).size < 2:
        return np.nan
    Z = [z[ok] for z in nz if np.unique(z[ok]).size > 1]
    if not partial or not Z:
        if np.unique(d[ok]).size < 2:
            return np.nan
        return float(stats.spearmanr(d[ok], mv[ok]).correlation)
    return _partial_rank_multi(d[ok], mv[ok], Z)


def flat_patterns_64(pack):
    """(n_features, 64) from a pack's (n_features, 8, 4, 2)."""
    P = np.asarray(pack["patterns"], float)
    return P.transpose(0, 3, 2, 1).reshape(P.shape[0], N_COND64)


def patterns_uncover(cache, collapse_phase=False, config_perm=None,
                     drop_silent=None):
    """Pattern pack from an "uncovers" cache, with or without the phase axis.

    `collapse_phase=True` gives the (n_features, 8, 4) shape every existing
    32-condition function already expects, with first and repeat pooled.
    `False` gives (n_features, 8, 4, 2) for the 64-condition space.

    Pooling is a mean over RIPPLES, not a mean of the two phase means, so a
    condition is not dragged toward whichever phase happens to be rarer.
    """
    shape = (N_CONFIG, N_STATE) if collapse_phase else (N_CONFIG, N_STATE,
                                                        N_PHASE)
    P, rois, sess, N, silent = [], [], [], [], 0
    for blk in cache:
        ci = blk["ci"]
        if config_perm is not None:
            ci = config_perm[blk["session"]][ci]
        rates = blk["rates"]
        key = ((ci, blk["si"]) if collapse_phase
               else (ci, blk["si"], blk["pi"]))
        counts = np.zeros(shape)
        np.add.at(counts, key, 1)

        ds = (blk.get("kind", "spikes") == "spikes") if drop_silent is None \
            else drop_silent
        keep = rates.sum(axis=1) > 0 if ds else np.ones(rates.shape[0], bool)
        silent += int((~keep).sum())
        if not keep.any():
            continue
        R = rates[keep]
        sums = np.zeros((int(keep.sum()),) + shape)
        np.add.at(sums.transpose(*range(1, len(shape) + 1), 0), key, R.T)
        with np.errstate(invalid="ignore", divide="ignore"):
            P.append(sums / np.where(counts == 0, np.nan, counts))
        rois.extend(blk["roi"][keep].tolist())
        sess.extend([blk["session"]] * int(keep.sum()))
        N.append(counts)
    return {"patterns": np.concatenate(P, axis=0), "roi": np.array(rois),
            "session": np.array(sess), "counts": np.stack(N),
            "n_silent_dropped": silent}


def rdm_for_64(pack, roi, min_cells=MIN_CELLS_PER_PAIR, normalise=None):
    """64 x 64 data RDM for one ROI."""
    sel = np.asarray(pack["roi"]) == roi
    return build_rdm_flat(flat_patterns_64(pack)[sel], min_cells, normalise)


# =============================================================================
# PERI-RIPPLE BANDS -- He et al. windows
# =============================================================================
#
# He et al. analyse compositional encoding in a PERI-RIPPLE window of
# -250 to +250 ms around the ripple peak, against a NON-PERI-RIPPLE baseline
# made of the two flanks, -750 to -250 ms and +250 to +750 ms, combined.
#
# This is a different question from "what is encoded DURING a ripple": 500 ms
# is ~8x the 60 ms ripple itself, so it is ripple-ALIGNED rather than
# ripple-internal. It is worth running because the sparsity diagnosis says the
# ripple-internal version cannot work -- 87% of (cell, ripple) observations
# are empty -- while a 500 ms window collects ~8x the spikes, and the
# peri-vs-non-peri contrast keeps a genuine test of ripple alignment.
#
# The two bands are computed from the SAME cumulative spike counts, so the
# non-peri rate is exactly "everything in +-750 ms that is not in +-250 ms":
#     peri     = n(+-0.25) / 0.5
#     nonperi  = (n(+-0.75) - n(+-0.25)) / 1.0

PERI_HALF_S = 0.25
NONPERI_OUTER_S = 0.75
# "nonperi" is He et al.'s baseline: BOTH flanks, so 1000 ms against the peri
# window's 500 ms. That asymmetry is not harmless here -- reliability scales
# with integration time, and the baseline measured 2.5-3x MORE reliable than
# the signal window purely for being twice as long. "nonperi_late" is the
# duration-matched control: the +250 to +750 ms flank alone, 500 ms, so peri
# and baseline carry the same counting noise.
BANDS = ("peri", "nonperi", "nonperi_late")


def _band_rate(spikes, t, band):
    """Firing rate in the peri- or non-peri-ripple band around times `t`."""
    inner = spike_counts_in_windows(spikes, t, PERI_HALF_S)
    if band == "peri":
        return inner / (2 * PERI_HALF_S)
    outer = spike_counts_in_windows(spikes, t, NONPERI_OUTER_S)
    return (outer - inner) / (2 * (NONPERI_OUTER_S - PERI_HALF_S))


def cache_band_rates(bundle, events, roi_tab, spikes, window, band="both",
                     scheme="uncovers", surrogate_rng=None,
                     require_clearance=True):
    """`cache_ripple_rates` with the He et al. bands instead of the ripple extent.

    `require_clearance` keeps only ripples whose FULL +-750 ms extent lies
    inside their own inter-uncover interval, so both the peri window and the
    non-peri flanks describe one condition. This is not optional book-keeping:
    with the 1 s post-press selection cap, 100% of flanks reached outside their
    interval (the early flank lands before the uncover), and at the
    all-uncovers event density the median gap is 1.35 s -- shorter than the
    1.5 s window itself. Run uncapped (`window=None` for the interval scheme)
    the clearance is met by 71% of explore ripples and 47% of all-uncover ones.

    `band="both"` returns (peri, nonperi, nonperi_late) from one pass, sharing
    the same spike lookups and -- crucially -- the same surrogate peaks, so the
    two bands of a given draw are the same pseudo-events.

    Surrogates are drawn from the clearance-respecting sub-interval, matched in
    number per event, which is the ripple-shuffle He et al. describe: temporal
    structure and ripple count preserved, true ripple timing removed.
    """
    if band not in BANDS + ("both",):
        raise ValueError(f"band must be one of {BANDS + ('both',)}")
    get = SCHEMES[scheme]
    out_p, out_n, out_l = [], [], []
    for s in sorted(set(roi_tab.session)):
        rip = get(bundle, s, events, window)
        cells = roi_tab[roi_tab.session == s]
        if rip.empty or cells.empty:
            continue
        lo_ok = rip.press_t_s.values + NONPERI_OUTER_S
        hi_ok = rip.win_hi_s.values - NONPERI_OUTER_S
        if require_clearance:
            keep = ((rip.t_peak_s.values >= lo_ok)
                    & (rip.t_peak_s.values <= hi_ok))
            rip = rip[keep]
            lo_ok, hi_ok = lo_ok[keep], hi_ok[keep]
        if rip.empty:
            continue
        if surrogate_rng is not None:
            t = lo_ok + surrogate_rng.uniform(0, 1, len(rip)) * (hi_ok - lo_ok)
        else:
            t = rip.t_peak_s.values.copy()

        n_p = np.empty((len(cells), len(rip)))
        n_n = np.empty((len(cells), len(rip)))
        n_l = np.empty((len(cells), len(rip)))
        flank = (NONPERI_OUTER_S - PERI_HALF_S) / 2.0     # 0.25 s half-width
        t_late = t + PERI_HALF_S + flank                  # centre of +250-750
        for row, (_, c) in enumerate(cells.iterrows()):
            sp = spikes[s]["spikes"][int(c.cell)]
            inner = spike_counts_in_windows(sp, t, PERI_HALF_S)
            outer = spike_counts_in_windows(sp, t, NONPERI_OUTER_S)
            n_p[row] = inner / (2 * PERI_HALF_S)
            n_n[row] = (outer - inner) / (2 * (NONPERI_OUTER_S - PERI_HALF_S))
            n_l[row] = spike_counts_in_windows(sp, t_late, flank) / (2 * flank)
        meta = {"session": s, "kind": "spikes",
                "ci": np.array([CONFIG_LABELS.index(x) for x in rip.cfg]),
                "si": np.array([STATES.index(x) for x in rip.state]),
                "roi": cells.roi.to_numpy()}
        if "phase" in rip.columns:
            meta["pi"] = np.array([PHASES.index(x) for x in rip.phase])
        out_p.append(dict(meta, rates=n_p))
        out_n.append(dict(meta, rates=n_n))
        out_l.append(dict(meta, rates=n_l))
    if band == "peri":
        return out_p
    if band == "nonperi":
        return out_n
    if band == "nonperi_late":
        return out_l
    return out_p, out_n, out_l


def band_window_overlap(bundle, events, roi_tab, window, scheme="uncovers"):
    """Fraction of ripples whose +-750 ms flanks cross the next uncover.

    Reported rather than corrected: the flanks are what He et al. use, and the
    same overlap applies to the surrogate draws, so the contrast stays fair.
    But a flank that crosses into the next inter-uncover interval carries
    firing from a different condition, which dilutes rather than inflates.
    """
    get = SCHEMES[scheme]
    n_tot = n_cross = 0
    for s in sorted(set(roi_tab.session)):
        rip = get(bundle, s, events, window)
        if rip.empty or "win_hi_s" not in rip.columns:
            continue
        lag_end = rip.t_peak_s.values + NONPERI_OUTER_S
        n_tot += len(rip)
        n_cross += int((lag_end > rip.win_hi_s.values).sum())
    return (n_cross / n_tot if n_tot else np.nan), n_tot
