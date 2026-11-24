#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-ROI permutation data-RDM builder for the DSR RSA pipeline.

Centralised so the main RSA script and any diagnostic script (e.g. the
control-dropout sweep) compute permutation null RDMs the SAME way and
share a pickle cache: each per-cell, per-trial firing rate is circularly
shifted by a random per-trial offset, the population matrix is rebuilt,
and the population-level z-scored RDM is computed once. The resulting
``(n_perms, n_pairs)`` matrix is cached on disk together with a
fingerprint of every parameter that influences the result; load-time
validation refuses to reuse a stale pickle.

Only the two z-scored across-run RDMs are produced:

    * ``split_halves_z``  — cross-half cosine RDM (run-1 vs run-2 cells of the
                            population matrix), z-scored per neuron across all
                            halves. Built by `mc.analyse.my_RSA.compute_crosscorr`.
    * ``between_tasks_z`` — between-task block of the run-averaged cosine RDM,
                            z-scored per neuron. Built by
                            `mc.analyse.my_RSA.compute_crosscorr_within`.

The non-z variants are intentionally dropped: every downstream test uses
the z-scored RDMs.

Hot-loop strategy
-----------------
Per-perm Python overhead is tiny because every inner step is pure numpy:
per-cell trial-shifts apply via ``np.take_along_axis`` in one call, trial
averages collapse the trial axis with a single ``np.nanmean``, the
downsample-to-12-conds is a ``reshape().mean()``, and the data-RDM step
is one ``compute_crosscorr`` matmul. With 1000 perms × ~150 cells × 8
configs the loop completes in a few seconds; the OLS that used to happen
inside this loop has been moved out and is run once per combo, vectorised
against the full ``(n_perms, n_pairs)`` matrix.

@author: Svenja Küchenhoff
"""
from __future__ import annotations

import os
import pickle
from typing import Iterable

import numpy as np

import mc.analyse.my_RSA as _rsa


# ── Test variants produced (z-only) ──────────────────────────────────
TEST_VARIANTS = ('split_halves_z', 'between_tasks_z')

# Supported shuffle modes for null construction. Both preserve
# within-trial autocorrelation (they only permute WHICH trial goes WHERE):
#   'shift'          — each trial gets an independent circular shift in
#                      time. Preserves per-cell per-config firing
#                      preferences. Historical default.
#   'shift_and_swap' — as 'shift', PLUS: within each cell, the trials are
#                      pooled across configs and re-assigned to config
#                      slots uniformly at random. Preserves per-config
#                      trial counts and run1/run2 mask patterns, but
#                      breaks any per-cell per-config firing preference.
#                      Recommended for DSR-family regressors — see
#                      `publication_md/method.md`.
SHUFFLE_MODES = ('shift', 'shift_and_swap')

# Method identifier baked into the cache fingerprint. Bump when the
# shift / swap logic, downsampling, or RDM construction changes — older
# pickles will then be rejected as stale.
PERM_METHOD = 'circular_shift_per_trial_v1'


# ── Population-matrix construction (one perm or empirical) ──────────
def _shift_and_pool_one_cell_config(trials_all, run1_mask, run2_mask,
                                    rng, n_conds_per_config, bin_per_cond,
                                    n_bins_per_trial):
    """Build the three average vectors (run1, run2, all) for one
    (cell, config) chunk.  ``rng=None`` means no shift (empirical).

    Each row of ``trials_all`` is one trial of this (cell, config); rows
    get independent circular shifts when ``rng`` is given. Trial-average
    happens AFTER the shift, separately for run1 / run2 / all.  Each
    average is then reshaped from ``n_bins_per_trial`` down to
    ``n_conds_per_config`` slots by averaging consecutive ``bin_per_cond``
    bins.

    Returns
    -------
    avg_r1, avg_r2, avg_all : 1-D arrays of length ``n_conds_per_config``;
        any half with no contributing trials is filled with NaN.
    """
    n_trials = trials_all.shape[0]
    if n_trials == 0:
        nan = np.full(n_conds_per_config, np.nan)
        return nan, nan.copy(), nan.copy()

    if rng is None:
        shifted = trials_all
    else:
        shifts = rng.integers(0, n_bins_per_trial, size=n_trials)
        idx = (np.arange(n_bins_per_trial) - shifts[:, None]) % n_bins_per_trial
        shifted = np.take_along_axis(trials_all, idx, axis=1)

    def _avg_then_downsample(mask):
        if mask is None:
            sel = shifted
        else:
            sel = shifted[mask]
        if sel.shape[0] == 0:
            return np.full(n_conds_per_config, np.nan)
        avg = np.nanmean(sel, axis=0)
        return avg.reshape(n_conds_per_config, bin_per_cond).mean(axis=1)

    return (_avg_then_downsample(run1_mask),
            _avg_then_downsample(run2_mask),
            _avg_then_downsample(None))


def _swap_config_assignment(per_cfg, configs, rng):
    """Return a new per_config dict where the cell's trials have been pooled
    across configs and randomly re-assigned to configs. Per-config trial
    counts and per-config run1/run2 masks are preserved, so downstream code
    treats the result exactly like the original per_cfg. Only the mapping
    trial-content → config is broken.
    """
    slabs = []
    sizes = []
    for cfg in configs:
        entry = per_cfg.get(cfg)
        if entry is None or entry['trials_all'].shape[0] == 0:
            sizes.append(0)
        else:
            slabs.append(entry['trials_all'])
            sizes.append(entry['trials_all'].shape[0])
    if not slabs:
        return per_cfg

    all_trials = np.vstack(slabs)
    perm = rng.permutation(all_trials.shape[0])
    reshuffled = all_trials[perm]

    out = {}
    off = 0
    for cfg, n in zip(configs, sizes):
        entry = per_cfg.get(cfg)
        if entry is None or n == 0:
            out[cfg] = entry
            continue
        out[cfg] = {
            'trials_all': reshuffled[off:off + n],
            'run1_mask':  entry['run1_mask'],
            'run2_mask':  entry['run2_mask'],
        }
        off += n
    return out


def _build_population_matrices(per_cell_trials, configs, rng,
                                n_conds_per_config, n_bins_per_trial,
                                shuffle_mode='shift'):
    """Returns ``(pop_split, pop_between)`` for ONE perm (or empirical).

    ``pop_split``   shape: ``(n_cells, n_configs * n_conds_per_config * 2)``
                    layout: [run1_cfg0 ... run1_cfgN | run2_cfg0 ... run2_cfgN]
    ``pop_between`` shape: ``(n_cells, n_configs * n_conds_per_config)``
                    layout: [all_cfg0 ... all_cfgN]

    ``shuffle_mode`` picks the null construction (see ``SHUFFLE_MODES``).
    ``rng=None`` bypasses all shuffling — used to build the empirical
    population matrix.
    """
    if shuffle_mode not in SHUFFLE_MODES:
        raise ValueError(
            f"unsupported shuffle_mode {shuffle_mode!r}; "
            f"expected one of {SHUFFLE_MODES}")

    do_swap = (shuffle_mode == 'shift_and_swap') and (rng is not None)

    bin_per_cond = n_bins_per_trial // n_conds_per_config
    n_cells = len(per_cell_trials)
    n_cfg = len(configs)
    n_half = n_cfg * n_conds_per_config

    pop_split = np.zeros((n_cells, n_half * 2), dtype=np.float64)
    pop_between = np.zeros((n_cells, n_half), dtype=np.float64)

    for ci, cell_chunk in enumerate(per_cell_trials):
        per_cfg = cell_chunk['per_config']
        if do_swap:
            per_cfg = _swap_config_assignment(per_cfg, configs, rng)
        for cfg_i, cfg in enumerate(configs):
            slot_lo = cfg_i * n_conds_per_config
            slot_hi = slot_lo + n_conds_per_config
            entry = per_cfg.get(cfg, None)
            if entry is None or entry['trials_all'].shape[0] == 0:
                pop_split[ci, slot_lo:slot_hi] = np.nan
                pop_split[ci, n_half + slot_lo:n_half + slot_hi] = np.nan
                pop_between[ci, slot_lo:slot_hi] = np.nan
                continue
            avg_r1, avg_r2, avg_all = _shift_and_pool_one_cell_config(
                entry['trials_all'], entry['run1_mask'], entry['run2_mask'],
                rng, n_conds_per_config, bin_per_cond, n_bins_per_trial,
            )
            pop_split[ci, slot_lo:slot_hi]                 = avg_r1
            pop_split[ci, n_half + slot_lo:n_half + slot_hi] = avg_r2
            pop_between[ci, slot_lo:slot_hi]               = avg_all
    return pop_split, pop_between


def _z_score_per_neuron(pop):
    mu = np.nanmean(pop, axis=1, keepdims=True)
    sd = np.nanstd(pop, axis=1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return (pop - mu) / sd


def _rdm_pair_from_pop(pop_split_z, pop_between_z, n_conds_per_config, n_configs):
    """One (split_halves_z RDM vector, between_tasks_z RDM vector) pair."""
    # compute_crosscorr expects (2N, n_cells) — i.e. transpose.
    rdm_split = _rsa.compute_crosscorr(
        pop_split_z.T, plotting=False, include_diagonal=False,
        no_tasks=n_configs,
    )[0]
    # compute_crosscorr_within returns (within_list, between_list, full_rdm);
    # we only keep the between-task block here.
    _within_list, rdm_between_list, _full = _rsa.compute_crosscorr_within(
        pop_between_z.T, plotting=False, include_diagonal=False,
        no_tasks=n_configs, block_size=n_conds_per_config,
    )
    return rdm_split, rdm_between_list[0]


# ── Public API ──────────────────────────────────────────────────────
def build_perm_data_rdms(
    per_cell_trials,
    configs,
    n_perms: int,
    seed: int,
    n_conds_per_config: int = 12,
    n_bins_per_trial: int = 360,
    verbose: bool = True,
    shuffle_mode: str = 'shift',
):
    """Compute empirical + per-perm z-scored RDMs.

    Parameters
    ----------
    per_cell_trials : list[dict]
        One dict per cell; each dict has:
          * ``'cell_id'`` (str)
          * ``'per_config'`` (dict) keyed by config string. Each entry holds:
              - ``'trials_all'``: ``(n_trials, n_bins_per_trial)`` raw firing
                rates for the correct trials of this (cell, config).
              - ``'run1_mask'``: bool array of length ``n_trials``.
              - ``'run2_mask'``: bool array of length ``n_trials``.
        Cells without any matching trials in any config can be omitted.

    configs : iterable[str]
        Ordered list of configuration strings; the RDM layout depends on
        this order.

    n_perms : int       number of permutations to compute.
    seed    : int       RNG seed (deterministic shifts → reproducible RDMs).
    shuffle_mode : str  null construction; one of ``SHUFFLE_MODES``.

    Returns
    -------
    empirical : dict {TEST_VARIANT: (n_pairs,) np.ndarray}
    perms     : dict {TEST_VARIANT: (n_perms, n_pairs) np.ndarray}
    """
    if shuffle_mode not in SHUFFLE_MODES:
        raise ValueError(
            f"unsupported shuffle_mode {shuffle_mode!r}; "
            f"expected one of {SHUFFLE_MODES}")

    configs = list(configs)
    n_cfg = len(configs)

    rng = np.random.default_rng(seed)

    # Empirical RDMs (no shuffling). rng=None short-circuits both shift
    # and swap in _build_population_matrices.
    pop_split, pop_between = _build_population_matrices(
        per_cell_trials, configs, rng=None,
        n_conds_per_config=n_conds_per_config,
        n_bins_per_trial=n_bins_per_trial,
        shuffle_mode=shuffle_mode,
    )
    emp_split, emp_between = _rdm_pair_from_pop(
        _z_score_per_neuron(pop_split),
        _z_score_per_neuron(pop_between),
        n_conds_per_config, n_cfg,
    )

    n_pairs_split = emp_split.shape[0]
    n_pairs_between = emp_between.shape[0]
    perm_split   = np.zeros((n_perms, n_pairs_split),   dtype=np.float64)
    perm_between = np.zeros((n_perms, n_pairs_between), dtype=np.float64)

    log_every = max(1, n_perms // 10) if verbose else None
    for p in range(n_perms):
        pop_split, pop_between = _build_population_matrices(
            per_cell_trials, configs, rng=rng,
            n_conds_per_config=n_conds_per_config,
            n_bins_per_trial=n_bins_per_trial,
            shuffle_mode=shuffle_mode,
        )
        rdm_split, rdm_between = _rdm_pair_from_pop(
            _z_score_per_neuron(pop_split),
            _z_score_per_neuron(pop_between),
            n_conds_per_config, n_cfg,
        )
        perm_split[p]   = rdm_split
        perm_between[p] = rdm_between
        if log_every and (p + 1) % log_every == 0:
            print(f"    perm-rdm: {p + 1}/{n_perms}")

    return ({'split_halves_z':  emp_split,
             'between_tasks_z': emp_between},
            {'split_halves_z':  perm_split,
             'between_tasks_z': perm_between})


# ── Cache fingerprint + load-or-build ─────────────────────────────────
def fingerprint(
    roi: str,
    cell_ids: Iterable[str],
    n_perms: int,
    seed: int,
    phase_residualise,        # str or None
    residualise_repeats: bool,
    configs: Iterable[str],
    n_conds_per_config: int = 12,
    n_bins_per_trial: int = 360,
    shuffle_mode: str = 'shift',
    drift_residualise=None,   # str or None
):
    """Build the dict baked into the perm-RDM pickle.

    Any caller-visible parameter that influences the data RDM goes in
    here; the cache loader insists on an exact match before reusing.

    Backward compatibility: when ``shuffle_mode == 'shift'`` we OMIT the
    key from the fingerprint so pre-existing caches (which were built
    without the shuffle_mode field) still match. Any non-default mode is
    baked in explicitly, so caches from different modes never alias.
    ``drift_residualise`` follows the same rule: omitted when None, so
    caches predating it still match, and baked in whenever it is set, so a
    drift-residualised run can never reuse a non-residualised null.
    """
    if shuffle_mode not in SHUFFLE_MODES:
        raise ValueError(
            f"unsupported shuffle_mode {shuffle_mode!r}; "
            f"expected one of {SHUFFLE_MODES}")

    fp = {
        'roi':                str(roi),
        'cell_ids':           tuple(sorted(map(str, cell_ids))),
        'n_cells':            int(len(list(cell_ids))),
        'n_perms':            int(n_perms),
        'seed':               int(seed),
        'phase_residualise':  None if phase_residualise is None else str(phase_residualise),
        'residualise_repeats': bool(residualise_repeats),
        'configs':            tuple(configs),
        'n_conds_per_config': int(n_conds_per_config),
        'n_bins_per_trial':   int(n_bins_per_trial),
        'tests_included':     TEST_VARIANTS,
        'method':             PERM_METHOD,
    }
    if shuffle_mode != 'shift':
        fp['shuffle_mode'] = shuffle_mode
    if drift_residualise is not None:
        fp['drift_residualise'] = str(drift_residualise)
    return fp


def _diff_fingerprint(cached, want):
    """Yield (key, cached_val, want_val) for each field that differs."""
    keys = sorted(set(cached.keys()) | set(want.keys()))
    for k in keys:
        c = cached.get(k, '<MISSING>')
        w = want.get(k, '<MISSING>')
        if c != w:
            yield k, c, w


def _scan_search_dirs_for_match(target_filename, search_dirs, fingerprint_data,
                                 verbose=True):
    """Walk every directory in ``search_dirs`` looking for a file named
    ``target_filename`` (typically ``perm_data_rdms_<ROI>.pkl``) whose
    stored fingerprint matches ``fingerprint_data`` exactly. Returns the
    matching ``(absolute_path, cached_dict)`` or ``(None, None)``.

    Searches each directory recursively up to one level deep so users can
    pass either the run-dir itself or its OUT_BASE parent.
    """
    if not search_dirs:
        return None, None
    for sd in search_dirs:
        if not os.path.isdir(sd):
            continue
        # candidates: directly under sd, or under sd/perm_data_rdms/
        candidates = [
            os.path.join(sd, target_filename),
            os.path.join(sd, 'perm_data_rdms', target_filename),
        ]
        # also scan one level of subdirectories (e.g. OUT_BASE/<run>/perm_data_rdms/)
        try:
            for entry in os.listdir(sd):
                sub = os.path.join(sd, entry)
                if os.path.isdir(sub):
                    candidates.append(os.path.join(sub, target_filename))
                    candidates.append(os.path.join(sub, 'perm_data_rdms', target_filename))
        except OSError:
            pass
        for cand in candidates:
            if not os.path.isfile(cand):
                continue
            try:
                with open(cand, 'rb') as f:
                    cached = pickle.load(f)
            except Exception as exc:
                if verbose:
                    print(f"[perm-rdm cache] failed to read {cand}: {exc}")
                continue
            if cached.get('fingerprint') == fingerprint_data:
                return os.path.abspath(cand), cached
    return None, None


def load_or_build_perm_rdms(
    pickle_path: str,
    per_cell_trials,
    fingerprint_data: dict,
    configs,
    n_conds_per_config: int = 12,
    n_bins_per_trial: int = 360,
    force: bool = False,
    verbose: bool = True,
    search_dirs=None,
    link_reused: bool = True,
):
    """Return ``(empirical, perms)``, reading from cache when valid.

    ``pickle_path`` is the on-disk cache for the current run. If the file
    exists AND its stored fingerprint matches ``fingerprint_data``
    exactly, we load and return.

    ``search_dirs`` is an optional list of OTHER directories (typically
    previous run dirs or their OUT_BASE parent) to scan for a matching
    cached pickle BEFORE rebuilding. If a match is found in any
    search_dir AND ``link_reused`` is True, we symlink the matching file
    to ``pickle_path`` so downstream tools that expect the cache in the
    canonical location keep working — without duplicating ~tens of MB
    per ROI on disk.

    A fingerprint mismatch / missing pickle anywhere triggers a fresh
    build, which is then saved to ``pickle_path``.
    """
    n_perms = fingerprint_data['n_perms']
    seed    = fingerprint_data['seed']

    # 1) Check the canonical pickle_path first
    if (not force) and os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                cached = pickle.load(f)
            if cached.get('fingerprint') == fingerprint_data:
                if verbose:
                    print(f"[perm-rdm cache] HIT: {pickle_path}")
                return cached['empirical'], cached['perms']
            if verbose:
                print(f"[perm-rdm cache] MISMATCH at {pickle_path} — checking "
                      f"search_dirs (if any) before rebuilding")
                for k, c, w in _diff_fingerprint(
                        cached.get('fingerprint', {}), fingerprint_data):
                    cs = repr(c)[:80]; ws = repr(w)[:80]
                    print(f"   {k}: cached={cs}  want={ws}")
        except Exception as exc:
            if verbose:
                print(f"[perm-rdm cache] failed to read {pickle_path}: {exc} — "
                      f"checking search_dirs (if any) before rebuilding")

    # 2) Cross-run cache lookup
    if (not force) and search_dirs:
        match_path, cached = _scan_search_dirs_for_match(
            target_filename=os.path.basename(pickle_path),
            search_dirs=search_dirs,
            fingerprint_data=fingerprint_data,
            verbose=verbose,
        )
        if cached is not None:
            if verbose:
                print(f"[perm-rdm cache] REUSE from previous run: {match_path}")
            if link_reused:
                os.makedirs(os.path.dirname(os.path.abspath(pickle_path)) or '.',
                            exist_ok=True)
                # Skip if pickle_path is already the same file (or a symlink to it)
                try:
                    same = (os.path.exists(pickle_path)
                            and os.path.samefile(pickle_path, match_path))
                except OSError:
                    same = False
                if not same:
                    # Remove stale file/symlink at pickle_path first
                    if os.path.lexists(pickle_path):
                        os.remove(pickle_path)
                    try:
                        os.symlink(match_path, pickle_path)
                        if verbose:
                            print(f"   symlinked → {pickle_path}")
                    except OSError as exc:
                        # symlink may fail on some platforms; fall back to copy
                        import shutil
                        shutil.copy2(match_path, pickle_path)
                        if verbose:
                            print(f"   symlink failed ({exc}); copied instead → "
                                  f"{pickle_path}")
            return cached['empirical'], cached['perms']

    # 3) Build fresh
    shuffle_mode = fingerprint_data.get('shuffle_mode', 'shift')
    if verbose:
        print(f"[perm-rdm cache] BUILD: {n_perms} perms × "
              f"{fingerprint_data['n_cells']} cells in ROI "
              f"{fingerprint_data['roi']!r}  (shuffle_mode={shuffle_mode!r})")
    empirical, perms = build_perm_data_rdms(
        per_cell_trials, configs, n_perms, seed,
        n_conds_per_config=n_conds_per_config,
        n_bins_per_trial=n_bins_per_trial,
        verbose=verbose,
        shuffle_mode=shuffle_mode,
    )

    os.makedirs(os.path.dirname(os.path.abspath(pickle_path)) or '.', exist_ok=True)
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'fingerprint': fingerprint_data,
            'empirical':   empirical,
            'perms':       perms,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"[perm-rdm cache] SAVED: {pickle_path}")
    return empirical, perms
