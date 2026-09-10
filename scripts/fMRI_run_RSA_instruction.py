#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 07:36:34 2026
Instruction-phase searchlight RSA, run separately for every second of the
instruction period. Per subject, per TR, per model: one whole-brain beta map.
RQ


------------
Is the action plan assembled BEFORE any action is executed? The task was
designed so that each layout is instructed once forwards and once backwards
(1-2-3-4 and 4-3-2-1), and subjects either followed the instructed order or
mentally reversed it, with reversal trials marked from the first reward cue.
That gives a 2x2 of visual instruction x execution, which is what lets a
representation of what was just SEEN be separated from a representation of
what will be DONE. The reward-DSR models describe the similarity of future
executions anchored at 'A' and are deliberately different from the observed
instructed order, so a fit to them cannot be explained by the visual input.

PAIRING AND THE TWO MODEL FAMILIES
----------------------------------
Conditions are paired by same EXECUTION (A1_forw <-> A2_backw), so the RDM
compares two independent task halves that share a plan but not a stimulus.
Any model can then be built in two variants:
  <model>          execution similarity — the plan the subject will carry out
  <model>_instr    instruction similarity — built by `instruction_relabel_dict`,
                   which overwrites each '_backw_' entry with its '_forw_'
                   counterpart so that both directions share the vector they
                   were SHOWN. Under the execution-based pairing this makes
                   every (task_i, task_j) 2x2 sub-block internally uniform,
                   which `verify_instruction_rdm_blocks` checks and prints.
The two variants are also correlated with each other and reported, so the
shared variance between "what was seen" and "what will be done" is explicit.

`rewDSR` at the A_reward anchor is four equal chunks, one per reward step
(A, B, C, D). Two families of models are sliced out of it, both literal
slices so that Hamming behaves identically to Hamming on rewDSR itself
rather than on the 9-dim one-hot stored under those names:
  SPLIT       one chunk each — A_rew, B_rew, C_rew, D_rew.
              "which single future reward is this".
  CUMULATIVE  the first k chunks — A_rew (A), AB_rew (A+B), ABC_rew (A+B+C),
              ABCD_rew (A+B+C+D, i.e. the whole rewDSR vector). "how much of
              the plan is already assembled". Because the chunks are equally
              long, Hamming on a cumulative channel is exactly the mean of the
              per-step Hammings it contains, so the family is strictly nested
              and ABCD_rew IS rewDSR.
Each of those has an `_instr` counterpart built the same way. The pre-rename
names (curr_rew / next_rew / two_next_rew / three_next_rew / two_rew /
three_rew) are rejected with a pointer to the new one — see
`LEGACY_MODEL_NAMES`.

DATA RDM SCOPE (config: `data_rdm_scope`)
    'across_only'  (default) only the TH1 x TH2 off-block, so every cell is
                   an across-half comparison and run-level noise cannot
                   inflate similarity.
    'full_no_diag' the symmetric (2n x 2n) matrix, strict lower triangle.
                   Nearly doubles the pairs the OLS sees, but the within-half
                   cells share run-level noise and are therefore biased
                   toward looking similar; that bias is NOT corrected.
    'within_only'  only the two within-half blocks. Required for anything
                   instruction-based: across halves a task is instructed in
                   the reverse order, so instruction dissim is constant there
                   and carries no variance at all.

Individual models can override this and be fitted in more than one scope:
  combo models    a `"scope"` key on the combo entry (string or list) — this
                  is how `exe_split` is fitted once within and once across
                  task halves in the same run.
  single models   `"single_model_scopes"` in the config, e.g.
                      {"execution": ["within_only", "across_only"],
                       "instruction": ["within_only"]}
                  applied by name: anything ending in `_instr` is
                  'instruction', everything else 'execution'. Execution models
                  carry variance in both blocks and are worth having both ways;
                  instruction models are constant across halves and can only be
                  fitted within.
Outputs that come from such an override are suffixed `_within` / `_across` /
`_full` so the variants never overwrite each other. All scopes are subsets of
the same cached (2n x 2n) data RDM, so running several costs no extra
searchlight work. (The legacy top-level `'across_only'` mode keeps its own
(n x n) cache and cannot be mixed with per-model or per-combo scopes.)

WHAT HAPPENS NEXT (this script does NOT do group statistics)
------------------------------------------------------------
Per-subject maps are smoothed (FWHM 5 mm), masked to voxels present in all
subjects and stacked into
    group_RSA_instruction_per_TR_glmbase_01-TR{tr}_cropped/
        cropped_masked_smooth_fwhm5_{model}_beta_std.nii   (X, Y, Z, n_subj)
Group inference for the reported instruction-phase result is then done by
`scripts/per_TR_loso.py` (statistics in `mc.analyse.loso`): a max-t
sign-flip permutation test inside an
a-priori BA32 / medial BA9 / medial BA10 mask, correcting over voxels and
seconds jointly, plus a leave-one-subject-out cross-validated timecourse.

USAGE
    python fMRI_run_RSA_instruction.py <subj_no> [config.json]
    (default config: rsa_instruction_full.json; the TR is set in the config,
     so one run of this script = one second of the instruction phase)

@author: Svenja Küchenhoff, 2026
"""


from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from nilearn.image import load_img
from rsatoolbox.util.searchlight import get_volume_searchlight
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import mc
import pickle
import sys
from datetime import date
import json
#import pdb; pdb.set_trace() 


def pair_correct_tasks(data_dict, keys_list):
    """
    data_dict: dict with keys like 'A1_forw_A_reward'
    keys_list: ordered list of keys you want to include and in what order
    Returns two matrices: one for the first element of each pair, one for its match.
    """
    # Define task pairing relationships
    task_pairs = {'1_forw': '2_backw', '1_backw': '2_forw'}
    th_1, th_2, paired_list_control  = [], [], []
    # Loop through keys in the *specified order*
    for key in keys_list:
        assert key in data_dict, "Missmatch between model rdm keys and data RDM keys"
        task, direction, state, phase = key.split('_')  # e.g. ['A1', 'forw', 'A', 'reward']
        # Create the pairing suffix (e.g. from '1_forw' → '2_backw')
        pair_suffix = task_pairs.get(f"{task[-1]}_{direction}")
        # Build the paired key (e.g. 'A2_backw_A_reward')
        pair_key = f"{task[0]}{pair_suffix}_{state}_{phase}"
        # Only add if both keys exist
        if pair_key in data_dict:
            th_1.append(np.asarray(data_dict[key]))
            th_2.append(np.asarray(data_dict[pair_key]))
            paired_list_control.append(f"{key} with {pair_key}")

    # sanity check: every task-half-1 key in keys_list must have found a partner
    n_th1_expected = sum(1 for k in keys_list if k.split('_')[0].endswith('1'))
    assert len(paired_list_control) == n_th1_expected, (
        f"Expected {n_th1_expected} pairs, got {len(paired_list_control)}. "
        "Some task-half-1 keys did not find their same-execution partner in the dict."
    )
    th_1 = np.vstack(th_1)
    th_2 = np.vstack(th_2)
    # print(paired_list_control)
    return th_1, th_2, paired_list_control


# Names, in canonical order, of the four channels that constitute a "literal split"
# of rewDSR at the A_reward anchor — one chunk each, so the name IS the reward
# step it encodes (A_rew = chunk 0, B_rew = 1, C_rew = 2, D_rew = 3).
REWDSR_SPLIT_CHANNELS = ('A_rew', 'B_rew', 'C_rew', 'D_rew')

# Cumulative "first k rewards" channels: {name: number of leading chunks kept}.
# k=1 is A_rew, already a split channel and the very same vector, so it is not
# repeated here. k=4 (ABCD_rew) is the whole rewDSR vector. The family
# A_rew -> AB_rew -> ABC_rew -> ABCD_rew is therefore strictly nested, and the
# name spells out which rewards each model knows about.
REWDSR_PREFIX_CHANNELS = {'AB_rew': 2, 'ABC_rew': 3, 'ABCD_rew': 4}

# Everything that is sliced out of the rewDSR A_reward vector rather than read
# from its own entry in model_EVs.
REWDSR_DERIVED_CHANNELS = tuple(REWDSR_SPLIT_CHANNELS) + tuple(REWDSR_PREFIX_CHANNELS)

# Names these models used to carry. They are NOT harmless typos: 'curr_rew',
# 'next_rew', 'two_next_rew' and 'three_next_rew' are also real keys in
# model_EVs, holding the 9-dim ONE-HOT version of the same idea. Left
# unguarded, an old config would silently fall through to the standard path,
# build the one-hot model instead of the rewDSR slice (mismatch 2/9 rather
# than 1) and produce different numbers under an unchanged name. So they raise.
LEGACY_MODEL_NAMES = {
    'curr_rew': 'A_rew', 'next_rew': 'B_rew',
    'two_next_rew': 'C_rew', 'three_next_rew': 'D_rew',
    'two_rew': 'AB_rew', 'three_rew': 'ABC_rew',
}


def check_no_legacy_names(model_names, where):
    """Raise if any name in `model_names` is a pre-rename model name."""
    hits = {}
    for m in model_names:
        base, _ = strip_instr(m)
        if base in LEGACY_MODEL_NAMES:
            hits[m] = m.replace(base, LEGACY_MODEL_NAMES[base], 1)
    if hits:
        raise ValueError(
            f"{where} uses pre-rename model name(s). Rename them in the config:\n  "
            + "\n  ".join(f"{old!r} -> {new!r}" for old, new in sorted(hits.items()))
            + "\n(the old split-channel names are also one-hot entries in "
              "model_EVs, so leaving them in place would silently build a "
              "DIFFERENT model rather than fail)")


# models in .json can be
# instr = visual instruction similarity
# "selected_models": [
#       "DSR", "rewDSR", "simple",
#       "A_rew",    "A_rew_instr",        # split:      one reward step each
#       "B_rew",    "B_rew_instr",
#       "C_rew",    "C_rew_instr",
#       "D_rew",    "D_rew_instr",
#       "AB_rew",   "AB_rew_instr",       # cumulative: the first k reward steps
#       "ABC_rew",  "ABC_rew_instr",
#       "ABCD_rew", "ABCD_rew_instr"      # == rewDSR
#   ]


# Suffix that marks the instruction-similarity variant of a model. Any model in
# `selected_models` ending in this suffix is built by first substituting each
# '_backw_' key's value with its '_forw_' counterpart so that A1_forw and
# A1_backw share the same model vector (they saw the same instruction).
INSTR_SUFFIX = '_instr'


def strip_instr(name):
    """Return (base_name, is_instr). 'rewDSR_instr' -> ('rewDSR', True)."""
    if name.endswith(INSTR_SUFFIX):
        return name[:-len(INSTR_SUFFIX)], True
    return name, False


def instruction_relabel_dict(model_subdict):
    """Replace each '<task>_backw_<state>_<phase>' key's value with the
    corresponding '<task>_forw_<state>_<phase>' value. Forward keys stay
    unchanged. Turns an execution-similarity model dict into an
    instruction-similarity model dict — under the current execution-based
    data pairing, this yields uniform 2x2 sub-blocks per (task_letter_i,
    task_letter_j) in the resulting model RDM."""
    out = dict(model_subdict)
    for k in list(model_subdict.keys()):
        parts = k.split('_')
        if len(parts) >= 2 and parts[1] == 'backw':
            forw_key = k.replace('_backw_', '_forw_', 1)
            if forw_key in model_subdict:
                out[k] = model_subdict[forw_key]
    return out


def verify_instruction_rdm_blocks(rdm, th1_labels, th2_labels, tol=1e-9):
    """Instruction models should be uniform inside each (task_letter_i,
    task_letter_j) 2x2 sub-block. Returns (block_df, all_uniform) — the
    df has one row per (task_i, task_j) with the unique block value and a
    uniformity flag."""
    from collections import defaultdict
    import pandas as _pd
    def _task_id(lbl):
        return lbl.split('_')[0]     # e.g. 'A1_forw_A_reward' -> 'A1'
    g1, g2 = defaultdict(list), defaultdict(list)
    for i, l in enumerate(th1_labels): g1[_task_id(l)].append(i)
    for j, l in enumerate(th2_labels): g2[_task_id(l)].append(j)
    rows = []
    all_uniform = True
    for t1, idx1 in sorted(g1.items()):
        for t2, idx2 in sorted(g2.items()):
            block = np.asarray(rdm)[np.ix_(idx1, idx2)]
            first = float(block.flat[0])
            uniform = np.allclose(block, first, atol=tol)
            if not uniform:
                all_uniform = False
            rows.append({'task1': t1, 'task2': t2,
                          'shape': block.shape,
                          'value': first,
                          'min':  float(block.min()),
                          'max':  float(block.max()),
                          'uniform': uniform})
    return _pd.DataFrame(rows), all_uniform


def _lower_tri_flat(mat):
    """Return the strict lower triangle (k=-1) of a square matrix, flattened.
    Used when ``data_rdm_scope == 'full_no_diag'``. Cosine dissim is symmetric,
    so the upper triangle is redundant; the diagonal (self-pairs) is 0 and
    would be pure autocorrelation, so we drop it too. Row-major ordering
    follows ``np.tril_indices(N, k=-1)`` — the matching model regressor must
    use the same ordering."""
    mat = np.asarray(mat)
    assert mat.ndim == 2 and mat.shape[0] == mat.shape[1], (
        f"_lower_tri_flat expects a square matrix, got {mat.shape}")
    i, j = np.tril_indices(mat.shape[0], k=-1)
    return mat[i, j]


def within_half_mask(n_pairs):
    """Boolean over the strict lower triangle of the (2n, 2n) block RDM, True
    for the WITHIN-half cells (the W1 and W2 blocks), False for across-half.

    Ordering matches `_lower_tri_flat`, so the same mask selects the matching
    cells of a model regressor and of a data RDM built in 'full_no_diag' mode.

    This is what `data_rdm_scope = 'within_only'` keeps. It exists because the
    across-half block is where the instruction models are degenerate (across
    halves the same task is instructed in the reverse order, so every
    across-half cell is a mismatch and the block is constant at 1.0). Keeping
    only within-half cells removes the within/across contrast entirely, so no
    regressor can absorb the run-level similarity offset between the two
    blocks -- and it retains the pair the design was built around: within a
    half, the two directions of one task saw the SAME instruction (instruction
    dissim 0) but execute REVERSED sequences (execution dissim 1)."""
    N = 2 * n_pairs
    blk = np.zeros((N, N))
    blk[:n_pairs, n_pairs:] = 1
    blk[n_pairs:, :n_pairs] = 1
    return _lower_tri_flat(blk) == 0


def within_half_mask_2d(n_pairs):
    """(2n, 2n) boolean, True for the two WITHIN-half blocks. Display companion
    to `within_half_mask`, which returns the same selection over the flattened
    strict lower triangle."""
    N = 2 * n_pairs
    m = np.zeros((N, N), dtype=bool)
    m[:n_pairs, :n_pairs] = True
    m[n_pairs:, n_pairs:] = True
    return m


def assemble_full_rdm_from_blocks(W1, A, W2):
    """Build the (n1+n2, n1+n2) block RDM from three (n,n) blocks:

        +----------------+----------------+
        |   W1 (within)  |   A (across)   |
        +----------------+----------------+
        |   A.T          |   W2 (within)  |
        +----------------+----------------+

    The lower-left block is A.T, giving a symmetric assembled matrix
    (cosine/hamming dissim are symmetric)."""
    W1 = np.asarray(W1); A = np.asarray(A); W2 = np.asarray(W2)
    n1, n2 = W1.shape[0], W2.shape[0]
    assert A.shape == (n1, n2), (
        f"A block shape {A.shape} does not match ({n1}, {n2})")
    R = np.empty((n1 + n2, n1 + n2), dtype=float)
    R[:n1, :n1] = W1
    R[:n1, n1:] = A
    R[n1:, :n1] = A.T
    R[n1:, n1:] = W2
    return R


BLOCK_NUISANCE = 'block'


def build_block_nuisance_RDM(n_pairs):
    """(2n, 2n) indicator: 1 for across-task-half cells, 0 for within-half.

    Why this is needed in 'full_no_diag'. Within-half pairs share run-level
    noise and come out systematically MORE similar than across-half pairs --
    measured on sub-02 TR4, mean cosine dissimilarity 0.828 within vs 0.863
    across, in 89% of searchlights. Any regressor correlated with the
    within/across split therefore gets a large beta everywhere for a reason
    that has nothing to do with the model it encodes.

    The instruction models are exactly such regressors: across halves the same
    task is instructed in the reverse order, so their across-half block is
    constant at 1.0 while their within-half block averages 0.711 -- r = +0.42
    to +0.54 with this indicator. The execution models lean the other way
    (r = -0.13 to -0.17). That is what put every instruction t-map at a mean of
    +2.07 (97% of voxels positive) and every execution t-map at -1.04.

    Including this indicator as a nuisance regressor absorbs the offset so the
    real regressors compete only for the residual structure. It does NOT make
    the instruction models clean: with the offset removed, 100% of an
    instruction regressor's remaining variance still lies in within-half cells
    (execution keeps 67-79% across-half), because the design gives instruction
    similarity no across-half variance at all."""
    N = 2 * n_pairs
    R = np.zeros((N, N))
    R[:n_pairs, n_pairs:] = 1
    R[n_pairs:, :n_pairs] = 1
    return R


def slice_rewDSR_channels(model_EVs, EV_keys, use_instruction=False):
    """
    Build (th1, th2) matrices for every model that is a slice of rewDSR at the
    A_reward anchor.

    That vector is four equal chunks, one per reward step, each holding the raw
    location value repeated (e.g. [5]*12 for A, then B, C, D). Two families are
    cut out of it:
      * SPLIT — one chunk each: A_rew, B_rew, C_rew, D_rew.
      * CUMULATIVE — the first k chunks: AB_rew (A+B), ABC_rew (A+B+C),
        ABCD_rew (the whole vector, i.e. rewDSR). A_rew (k=1) is literally the
        first split channel, so it is not repeated in the cumulative dict.

    Slicing rather than reading model_EVs['curr_rew'] etc. keeps every channel a
    LITERAL piece of rewDSR: each chunk carries the raw location value, so
    hamming behaves the same way as on rewDSR (match -> 0, mismatch -> 1)
    instead of the 2/9 that the 9-dim one-hot stored under those names gives.
    Because the chunks are equally long, hamming on a cumulative channel is
    exactly the mean of the per-step hammings it contains.

    Parameters
    ----------
    use_instruction : bool
        If True, apply ``instruction_relabel_dict`` to the rewDSR sub-dict
        before pairing/slicing — produces the instruction-similarity variant
        of every channel.

    Returns
    -------
    dict : {channel_name: (th1_mat, th2_mat)}
        Split channels have shape (n_pairs, chunk); cumulative channels
        (n_pairs, k * chunk).
    """
    rewDSR_sub = {k: v for k, v in model_EVs['rewDSR'].items() if k.endswith('_A_reward')}
    if use_instruction:
        rewDSR_sub = instruction_relabel_dict(rewDSR_sub)
    rewDSR_keys = [k.replace('_instruction_onset', '_A_reward') for k in EV_keys]
    th1_full, th2_full, _ = pair_correct_tasks(rewDSR_sub, rewDSR_keys)

    n_pairs, n_total = th1_full.shape
    assert n_total % 4 == 0, (
        f"rewDSR at A_reward has {n_total} elements, not divisible by 4."
    )
    CHUNK = n_total // 4
    out = {}
    for i, name in enumerate(REWDSR_SPLIT_CHANNELS):
        out[name] = (th1_full[:, i * CHUNK:(i + 1) * CHUNK],
                     th2_full[:, i * CHUNK:(i + 1) * CHUNK])
    for name, k in REWDSR_PREFIX_CHANNELS.items():
        out[name] = (th1_full[:, :k * CHUNK], th2_full[:, :k * CHUNK])
    return out


# ── Scope names ───────────────────────────────────────────────────────────
# Canonical scope names plus the short aliases a config may use. A combo model
# can carry its own "scope" (string or list) to be fitted in a scope other than
# the config-level `data_rdm_scope` — used to run the same execution model once
# within and once across task halves in a single pass.
SCOPE_ALIASES = {
    'across_only': 'across_only', 'across': 'across_only',
    'within_only': 'within_only', 'within': 'within_only',
    'full_no_diag': 'full_no_diag', 'full': 'full_no_diag',
}
# Suffix appended to the output map names of a combo that declares "scope",
# so the within- and across-half fits of one combo never overwrite each other.
SCOPE_TAGS = {'across_only': 'across', 'within_only': 'within', 'full_no_diag': 'full'}


def normalise_scope(name):
    assert name in SCOPE_ALIASES, (
        f"unknown scope {name!r} — use one of {sorted(set(SCOPE_ALIASES))}")
    return SCOPE_ALIASES[name]


def combo_scopes(combo, default_scope):
    """(scopes, tag_outputs) for one combo dict. A combo without a "scope" key
    runs in the config-level scope and keeps its plain output names."""
    if "scope" not in combo:
        return [default_scope], False
    raw = combo["scope"]
    raw = [raw] if isinstance(raw, str) else list(raw)
    return [normalise_scope(x) for x in raw], True


def single_model_scopes(model, cfg_scopes, default_scope):
    """(scopes, tag_outputs) for one SINGLE model.

    `cfg_scopes` is the config's optional `single_model_scopes`, e.g.

        "single_model_scopes": {"execution":   ["within_only", "across_only"],
                                "instruction": ["within_only"]}

    A model is 'instruction' iff its name ends in `_instr`, else 'execution'.
    The asymmetry is forced by the design, not chosen: an execution model has
    variance both within and across task halves and is worth fitting in both,
    whereas an instruction model is constant across halves (there the same task
    is instructed in the reverse order) and can only be fitted within-half.

    Without the key, every single model runs in the config-level scope and keeps
    its plain output name, exactly as before."""
    if not cfg_scopes:
        return [default_scope], False
    key = 'instruction' if model.endswith(INSTR_SUFFIX) else 'execution'
    raw = cfg_scopes.get(key, default_scope)
    raw = [raw] if isinstance(raw, str) else list(raw)
    return [normalise_scope(x) for x in raw], True


def design_rank_report(names, X):
    """Report on a design that `evaluate_model` would silently fail on.

    `evaluate_model_vec` zeroes a constant regressor column and returns NaN
    for EVERY regressor when the (NaN-dropped, z-scored) design is
    rank-deficient. `save_my_RSA_results` then writes those NaNs out as an
    all-zero map with no error anywhere — so this has to be caught before the
    OLS runs. Two things trigger it with these models:
      * any `_instr` model in 'across_only' scope — the instruction RDM is
        constant across task halves by design (both halves saw the same
        sequence in opposite order, so every across-half cell is a mismatch),
        so it carries no variance there and needs 'full_no_diag';
      * a combo whose regressors are collinear on the rows surviving the NaN
        drop — e.g. ['rewDSR', 'simple'], which correlate at r = 1.0 on the
        30 cells where `simple` is not NaN.

    Returns (ok, message)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Xa = np.column_stack([np.ones(X.shape[0]), X])
    fin = np.isfinite(Xa).all(axis=1)
    Z = Xa[fin]
    for c in range(1, Z.shape[1]):
        sd = Z[:, c].std()
        Z[:, c] = (Z[:, c] - Z[:, c].mean()) / sd if sd > 0 else 0.0
    const = [n for i, n in enumerate(names) if Z[:, i + 1].std() == 0]
    rank = int(np.linalg.matrix_rank(Z.T @ Z))
    ok = rank == Z.shape[1]
    msg = (f"{int(fin.sum())}/{X.shape[0]} rows finite, rank {rank}/{Z.shape[1]}"
           + (f", constant: {const}" if const else ""))
    return ok, msg


#
#
# import pdb; pdb.set_trace() 
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
    print("Running on laptop.")
    
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"
    print(f"Running on Cluster, setting {source_dir} as data directory")

# --- Load configuration ---
config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_instruction_full.json"
with open(f"{config_path}/{config_file}", "r") as f:
    config = json.load(f)

# SETTINGS
EV_string = config.get("load_EVs_from")
regression_version = config.get("regression_version")
TR = config.get("TR")
# The per-TR GLMs are '01-TR4'; the instruction-epoch GLMs are named after the
# epoch they measure ('instr_see-A-first') and set no TR, so the version is
# already the full GLM name.
regression_version_full = regression_version if TR is None else f"{regression_version}-TR{TR}"


name_RSA = config.get("name_of_RSA")
RDM_version = f"{name_RSA}"


# Subjects
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'  
subjects = [f"sub-{subj_no}"]

# Flags
smoothing = config.get("smoothing", True)
fwhm = config.get("fwhm", 5)
PLOTTING = True

# --- Scope of the data/model RDM ------------------------------------------
# 'across_only'   : classic behaviour — n_conds x n_conds cross-block only
#                   (TH1 vs TH2). Each cell is a pure across-runs comparison,
#                   so run-level noise cannot inflate similarity.
# 'full_no_diag'  : symmetric (2n_conds x 2n_conds) full RDM (within-run-1,
#                   across, within-run-2), strict lower triangle only (k=-1),
#                   diagonal dropped. Nearly doubles the number of pairs the
#                   OLS sees. Off-diagonal within-run cells share run-level
#                   noise → within-run pairs tend to look more similar than
#                   across-run pairs at the same true stimulus similarity;
#                   this bias is known but not corrected here.
# 'within_only'   : the strict lower triangle of the two WITHIN-half blocks
#                   only. Built from the same (2n x 2n) matrix as
#                   'full_no_diag' and then subset, so it reuses the same
#                   cached data RDM. Use this for instruction-similarity
#                   questions: the across-half block carries no instruction
#                   information (constant by design) and its similarity offset
#                   against the within-half block is what biases every
#                   instruction regressor upward in 'full_no_diag'.
data_rdm_scope = normalise_scope(config.get("data_rdm_scope", "across_only"))
# The legacy 'across_only' mode fits the (n x n) TH1 x TH2 block directly and
# keeps its own data RDM cache. Every other scope is a subset of the strict
# lower triangle of the assembled (2n x 2n) matrix, which is what makes
# per-combo scope overrides free: they reuse the one cached full data RDM.
legacy_across_block = (data_rdm_scope == "across_only")
print(f"data_rdm_scope = {data_rdm_scope}")

# this should better be: what kind of searchlight_mask do you want?
# make sure to change this in the config files!
#load_searchlights = config.get("load_searchlights", False)
searchlight_mask = config.get("searchlight_mask", None)

print(f"Now running RSA based on subj GLM {regression_version} for subj {subj_no}")


for sub in subjects:
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
    if os.path.isdir(data_dir):
        print("Running on laptop.")
        only_load_labels = False 
    else:
        data_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}"
        only_load_labels = False
        print(f"Running on Cluster, setting {data_dir} as data directory")
      
    modelled_conditions_dir = f"{data_dir}/beh/modelled_EVs"
    data_rdm_dir = f"{data_dir}/func/data_RDMs_glmbase_{regression_version_full}_{searchlight_mask}"
    results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version_full}/results"
    if smoothing == True:
       results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version_full}_smooth{fwhm}/results"
    os.makedirs(results_dir, exist_ok=True)

    # get a reference image to later project the results onto. This is usually
    # example_func from half 1, as this is where the data is corrected to.
    ref_img = load_img(f"{data_dir}/func/preproc_clean_01.feat/example_func.nii.gz")
    
    
    # Step 1: creating the searchlights
    # mask will define the searchlight positions, in pt01 space because that is 
    # where the functional files have been registered to.
    if searchlight_mask:
        if searchlight_mask == 'no_CSF':
            mask_file = load_img(f"{data_dir}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")
            mask_name = '_no_CSF' # Found 166.240 searchlights with no CSF mask
        elif searchlight_mask == 'grey_matter':
            mask_file = load_img(f"{data_dir}/anat/grey_matter_mask_func_01.nii.gz")
            mask_name = '_grey_matter'  # Found 126.404 searchlights with gm mask 
    else:
        mask_file = ref_img.copy() # full BOLD Found 175.483 searchlights
        mask_name = ''
    mask = mask_file.get_fdata()  
    path_to_searchlight_centers = f"{data_dir}/func/searchlight_centers{mask_name}.pkl"
    path_to_searchlight_neighbours = f"{data_dir}/func/searchlight_neighbors{mask_name}.pkl"
    if os.path.exists(path_to_searchlight_centers):
        with open(path_to_searchlight_centers, "rb") as f:
            centers = pickle.load(f)
        with open(path_to_searchlight_neighbours, "rb") as f:
            neighbors = pickle.load(f)
    else:
        centers, neighbors = get_volume_searchlight(mask, radius=3, threshold=0.5)
        with open(path_to_searchlight_centers, 'wb') as file:
            pickle.dump(centers, file)
            print("stored searchlight centres")
        with open(path_to_searchlight_neighbours, 'wb') as file:
            pickle.dump(neighbors, file)   
            print("stored searchlight neighbors")

    #
    # Step 2: loading conditions for model and data RDMs
    #
    # Model EVs — full dict (used to build DSR / rewDSR / simple below).
    with open(f"{modelled_conditions_dir}/{sub}_modelled_EVs_{EV_string}.pkl", 'rb') as file:
        model_EVs = pickle.load(file)
    # Which models to build + evaluate. Driven by the config so we can swap
    # between the original ['DSR', 'rewDSR', 'simple'] analysis, the split
    # ['A_rew', 'B_rew', 'C_rew', 'D_rew'] one and the cumulative
    # ['A_rew', 'AB_rew', 'ABC_rew', 'ABCD_rew'] one without touching the script.
    selected_models = config.get("selected_models", ['DSR', 'rewDSR', 'simple'])
    check_no_legacy_names(selected_models, "config 'selected_models'")
    for _c in config.get("combo_models", []):
        check_no_legacy_names(_c["regressors"], f"combo model {_c['name']!r}")
    # Data EVs — one PE per instruction-phase condition at this TR, per task half.
    data_EVs, all_EV_keys = mc.analyse.my_RSA.load_data_EVs_instr_TRwise(
        data_dir, regression_version=regression_version, TR=TR,
        only_load_labels=only_load_labels,
    )
    EV_keys = sorted(all_EV_keys)
    print(f"including the following EVs in the RDMs: {EV_keys}")

    # Pair task halves by same-execution (A1_forw <-> A2_backw, etc.).
    data_th1, data_th2, paired_labels = pair_correct_tasks(data_EVs, EV_keys)
    data_concat = np.concatenate((data_th1, data_th2), axis=0)
    
    #
    # Step 3: compute the model RDMs.
    # Labels aligned with data pairing (row -> TH1 label, col -> TH2 label).
    th1_labels = [p.split(' with ')[0].replace('_instruction_onset', '') for p in paired_labels]
    th2_labels = [p.split(' with ')[1].replace('_instruction_onset', '') for p in paired_labels]
    # Cells of the (2n, 2n) strict lower triangle that are WITHIN-half. Used to
    # subset both the model regressors and the cached data RDM when
    # data_rdm_scope == 'within_only'.
    within_mask = within_half_mask(len(th1_labels))

    model_RDM_dir = {}
    # In 'full_no_diag' mode we additionally store the assembled (2n, 2n)
    # block matrix for each model. The OLS then uses its strict lower
    # triangle via _lower_tri_flat(). Keeping model_RDM_dir[model] as the
    # (n, n) across block preserves the existing PLOTTING / verification code.
    model_RDM_full_dir = {}

    # Sources for the models sliced out of rewDSR (split channels + cumulative
    # first-k-rewards channels) — one dict for execution, one for instruction.
    # 'A_rew' and 'A_rew_instr' are the same slice drawn from the two.
    split_th_by_channel = {}
    split_th_by_channel_instr = {}
    _base_of = [strip_instr(m) for m in selected_models]
    if any(base in REWDSR_DERIVED_CHANNELS and not is_instr for base, is_instr in _base_of):
        split_th_by_channel = slice_rewDSR_channels(
            model_EVs, EV_keys, use_instruction=False)
    if any(base in REWDSR_DERIVED_CHANNELS and is_instr for base, is_instr in _base_of):
        split_th_by_channel_instr = slice_rewDSR_channels(
            model_EVs, EV_keys, use_instruction=True)

    # Every non-'simple' model in `selected_models` is built the same way:
    # hamming dissim over its A_reward vectors, TH1 x TH2, full off-block.
    # Models ending in '_instr' are built with instruction relabelling first.
    for model in selected_models:
        base_name, is_instr = strip_instr(model)
        if base_name == 'simple':
            continue
        if base_name in REWDSR_DERIVED_CHANNELS:
            source = split_th_by_channel_instr if is_instr else split_th_by_channel
            m_th1, m_th2 = source[base_name]
        else:
            # standard path: filter model_EVs[base_name] to its _A_reward keys
            a_rew_sub = {k: v for k, v in model_EVs[base_name].items()
                          if k.endswith('_A_reward')}
            if is_instr:
                a_rew_sub = instruction_relabel_dict(a_rew_sub)
            a_rew_keys = [k.replace('_instruction_onset', '_A_reward') for k in EV_keys]
            m_th1, m_th2, _ = pair_correct_tasks(a_rew_sub, a_rew_keys)
        model_RDM_dir[model] = mc.analyse.my_RSA.compute_hamming_instruction_RDM(m_th1, m_th2)
        if not legacy_across_block:
            W1 = mc.analyse.my_RSA.compute_hamming_instruction_RDM(m_th1, m_th1)
            W2 = mc.analyse.my_RSA.compute_hamming_instruction_RDM(m_th2, m_th2)
            model_RDM_full_dir[model] = assemble_full_rdm_from_blocks(
                W1, model_RDM_dir[model], W2)

    # Simple — {-1, +1, NaN} based on same/different execution within the same task letter.
    if 'simple' in selected_models:
        model_RDM_dir['simple'] = mc.analyse.my_RSA.build_simple_instruction_RDM(th1_labels, th2_labels)
        if not legacy_across_block:
            all_labels = th1_labels + th2_labels
            model_RDM_full_dir['simple'] = mc.analyse.my_RSA.build_simple_instruction_RDM(
                all_labels, all_labels)

    # Block nuisance regressor. Available as the reserved name 'block' in any
    # combo; `add_block_nuisance: true` in the config appends it to every combo
    # automatically so it cannot be forgotten in one of them.
    add_block_nuisance = config.get("add_block_nuisance", False)
    combo_cfg = config.get("combo_models", [])
    if add_block_nuisance:
        for combo in combo_cfg:
            if BLOCK_NUISANCE not in combo["regressors"]:
                combo["regressors"] = list(combo["regressors"]) + [BLOCK_NUISANCE]
    _block_wanted = (BLOCK_NUISANCE in selected_models or
                     any(BLOCK_NUISANCE in c["regressors"] for c in combo_cfg))
    if _block_wanted:
        assert data_rdm_scope == "full_no_diag", (
            "the 'block' nuisance regressor only exists in 'full_no_diag' scope "
            "-- in 'across_only' every cell is across-half and in 'within_only' "
            "every cell is within-half, so it is constant either way and the "
            "design would be rank-deficient. In 'within_only' it is also "
            "unnecessary: there is no block contrast left to absorb.")
        n_pairs = len(th1_labels)
        model_RDM_full_dir[BLOCK_NUISANCE] = build_block_nuisance_RDM(n_pairs)
        # keep the (n, n) across block too, so the combo `missing` check and the
        # plotting path see it like any other model
        model_RDM_dir[BLOCK_NUISANCE] = np.ones((n_pairs, n_pairs))
        print(f"[block nuisance] added: {int(build_block_nuisance_RDM(n_pairs).sum())} "
              f"across-half cells of {n_pairs*2}x{n_pairs*2}, "
              f"appended to {sum(BLOCK_NUISANCE in c['regressors'] for c in combo_cfg)} "
              f"combo model(s)")

    # ── Which scopes is each model actually fitted in? ───────────────────
    # Union over its single-model scopes and every combo it takes part in.
    # Resolved here, before anything is plotted, so the figures show exactly
    # the cells that go into an OLS and nothing else.
    single_scope_cfg = config.get("single_model_scopes", None)
    run_single_models = config.get("run_single_models", True)
    run_combo_models = config.get("run_combo_models", bool(combo_cfg))
    scopes_per_model = {m: [] for m in selected_models}
    if run_single_models:
        for m in selected_models:
            for sc in single_model_scopes(m, single_scope_cfg, data_rdm_scope)[0]:
                if sc not in scopes_per_model[m]:
                    scopes_per_model[m].append(sc)
    if run_combo_models:
        for c in combo_cfg:
            for sc in combo_scopes(c, data_rdm_scope)[0]:
                for m in c["regressors"]:
                    if m in scopes_per_model and sc not in scopes_per_model[m]:
                        scopes_per_model[m].append(sc)
    for m, scs in scopes_per_model.items():
        if not scs:                      # built but never fitted — still plot it
            scopes_per_model[m] = [data_rdm_scope]
    all_scopes_used = [sc for sc in ("within_only", "across_only", "full_no_diag")
                       if any(sc in v for v in scopes_per_model.values())]
    print("\n[scopes] " + ", ".join(
        f"{m}: {'+'.join(SCOPE_TAGS[sc] for sc in scs)}"
        for m, scs in scopes_per_model.items()))

    def _scope_cells(scope):
        """Boolean over the strict lower triangle of the (2n, 2n) matrix,
        selecting the cells a given scope fits. The three scopes partition that
        triangle: 'within_only' keeps the two within-half blocks, 'across_only'
        their complement (which is exactly the n**2 cells of the across block),
        'full_no_diag' keeps everything. Because they are masks over one and the
        same vector, the model regressor and the cached data RDM are always
        subset in the same order."""
        if scope == "within_only":
            return within_mask
        if scope == "across_only":
            return ~within_mask
        return np.ones(within_mask.shape, dtype=bool)

    def _model_regressor(m, scope=None):
        """The 1-D vector the OLS actually sees, for `scope` (default: the
        config-level scope)."""
        scope = data_rdm_scope if scope is None else scope
        if legacy_across_block:
            assert scope == "across_only", (
                f"combo scope {scope!r} needs the assembled (2n x 2n) model/data "
                "RDMs, which the legacy top-level 'across_only' mode never "
                "builds. Set data_rdm_scope to 'within_only' or 'full_no_diag' "
                "and give the combo an explicit scope instead.")
            return np.asarray(model_RDM_dir[m]).ravel()
        return _lower_tri_flat(model_RDM_full_dir[m])[_scope_cells(scope)]

    def _mask_2d_for_scope(scope):
        """(2n, 2n) boolean, True for the cells `scope` fits. Display companion
        to `_scope_cells`, which selects the same cells over the flat triangle."""
        w = within_half_mask_2d(len(th1_labels))
        return w if scope == "within_only" else (~w if scope == "across_only"
                                                 else np.ones_like(w))

    def _display_RDM(model, scope):
        """(matrix, row labels, col labels, caption) for plotting ONE model in
        ONE scope.

        Shows the cells that ENTER THE OLS for that scope; everything excluded
        is NaN, which `plot_instruction_RDM` renders white. Plotting the (n, n)
        across block regardless of scope -- as this script used to -- is
        misleading in 'within_only', where that block is not fitted at all and
        is a constant 1.0 for every instruction model."""
        if legacy_across_block:
            return (np.asarray(model_RDM_dir[model], dtype=float),
                    th1_labels, th2_labels, "across block (TH1 x TH2)")
        M = np.array(model_RDM_full_dir[model], dtype=float)
        all_lab = th1_labels + th2_labels
        keep = _mask_2d_for_scope(scope)
        M[~keep] = np.nan
        caption = {"within_only": "within-half blocks only; white = not fitted",
                   "across_only": "across-half block only; white = not fitted",
                   "full_no_diag": "full 2n x 2n (W1 | A | W2)"}[scope]
        # Both axes of the assembled matrix run over ALL conditions of BOTH
        # halves, so the function's default 'task half 1/2' labels would lie.
        return M, all_lab, all_lab, caption

    # ── Instruction-model verification + correlation with execution ──────
    # For every model ending in `_instr`, verify the 2x2 sub-block
    # uniformity property (every (task_letter_i, task_letter_j) block should
    # be internally constant) and print the block table. Then, if the
    # execution counterpart is also present, report Pearson r between the
    # two RDMs so you can see how much variance they share.
    instr_models_present = [m for m in selected_models if m.endswith(INSTR_SUFFIX)]
    exec_instr_correlations = {}
    for im in instr_models_present:
        # Verify on the matrix that is actually fitted. The 2x2 uniformity
        # property is a statement about the across block; in 'within_only' the
        # meaningful check is on the assembled matrix instead, so report which.
        if data_rdm_scope == "across_only":
            rdm, rlab, clab = np.asarray(model_RDM_dir[im]), th1_labels, th2_labels
        else:
            rdm = np.asarray(model_RDM_full_dir[im])
            rlab = clab = th1_labels + th2_labels
        block_df, all_uniform = verify_instruction_rdm_blocks(rdm, rlab, clab)
        print(f"\n[instr-model verify, scope={data_rdm_scope}] {im}: "
              f"RDM shape = {rdm.shape}, all 2x2 blocks uniform = {all_uniform}")
        with pd.option_context('display.width', 200,
                                'display.max_rows', 100):
            print(block_df.round(4).to_string(index=False))
        # Correlation with execution counterpart, reported once per scope the
        # instruction model is actually fitted in, and always over the vectors
        # the OLS sees -- otherwise the number describes cells nobody fits.
        base_name = im[:-len(INSTR_SUFFIX)]
        if base_name in model_RDM_dir:
            for sc in scopes_per_model[im]:
                exec_flat = _model_regressor(base_name, sc)
                instr_flat = _model_regressor(im, sc)
                m = np.isfinite(exec_flat) & np.isfinite(instr_flat)
                r_pearson = float(np.corrcoef(exec_flat[m], instr_flat[m])[0, 1])
                exec_instr_correlations[f"{base_name}_vs_{im}_{SCOPE_TAGS[sc]}"] = {
                    "execution": base_name, "instruction": im, "scope": sc,
                    "pearson_r": round(r_pearson, 4), "n_cells": int(m.sum())}
                print(f"[correlation, {sc}] {base_name} (execution) vs {im} "
                      f"(instruction): Pearson r = {r_pearson:+.4f}  "
                      f"(n_cells = {int(m.sum())})")
        else:
            print(f"[correlation] {base_name} not in selected_models — "
                  f"skipping execution-vs-instruction correlation.")
    
    if PLOTTING == True:
        # One figure per model PER SCOPE it is fitted in — plotted straight from
        # the stored arrays, no recomputation. Cells outside the scope are NaN
        # (white), so what you see is exactly what that OLS regressor contains.
        # A model fitted both within and across halves therefore gets two
        # figures, and they should look structured in both.
        # Titles/labels are kept short on purpose: these panels are 4 x 4 cm at
        # 9 pt Arial, so a two-line sentence would swamp the matrix. The long
        # form lives in the print-out and the settings json instead.
        SCOPE_CAPTION = {"within_only": "within-half",
                         "across_only": "across-half",
                         "full_no_diag": "full"}
        for model in selected_models:
            vmin, vmax = (-1, 1) if model == 'simple' else (0, 1)
            for scope in scopes_per_model[model]:
                M, rlab, clab, caption = _display_RDM(model, scope)
                # Stats over the REGRESSOR (strict lower triangle of the fitted
                # cells), not over the symmetric matrix the figure shows —
                # otherwise every cell is counted twice. A regressor with one
                # distinct value is constant and would sink the design.
                reg = _model_regressor(model, scope)
                reg = reg[np.isfinite(reg)]
                print(f"[model RDM] {model:22s} [{SCOPE_TAGS[scope]:6s}] "
                      f"{reg.size:4d} fitted cells, "
                      f"{len(np.unique(np.round(reg, 6))):2d} distinct values, "
                      f"range [{reg.min():.3f}, {reg.max():.3f}], "
                      f"sd {reg.std():.3f}")
                mc.analyse.my_RSA.plot_instruction_RDM(
                    M, rlab, clab,
                    title=f'{model}\n{SCOPE_CAPTION[scope]}',
                    vmin=vmin, vmax=vmax,
                    xlabel='', ylabel='',
                    n_first_half=len(th1_labels),
                    save_path=f"{results_dir}_{model}_{SCOPE_TAGS[scope]}")

        # Optional inspection plot: cosine dissim from one random searchlight.
        plot_example_data_RDM = config.get("plot_example_data_RDM", False)
        if plot_example_data_RDM and not only_load_labels:
            rng = np.random.default_rng(42)
            sl_idx = int(rng.integers(0, len(centers)))
            vox_ids = np.asarray(neighbors[sl_idx])
            sl_data = data_concat[:, vox_ids]
            n_conds = sl_data.shape[0] // 2
            # Same convention as the model figures: one panel per scope in use,
            # showing the cells that enter that OLS and whiting out the rest.
            if legacy_across_block:
                ex_full = None
                ex = mc.analyse.my_RSA.compute_cosine_instruction_RDM(
                    sl_data[:n_conds], sl_data[n_conds:])
                mc.analyse.my_RSA.plot_instruction_RDM(
                    ex, th1_labels, th2_labels,
                    title=f'example data RDM (searchlight #{sl_idx}, cosine dissim)\n'
                          'across block (TH1 x TH2)',
                    save_path=f"{results_dir}_data_across")
            else:
                ex_full = np.array(mc.analyse.my_RSA.compute_cosine_instruction_RDM(
                    sl_data, sl_data), dtype=float)
                all_lab = th1_labels + th2_labels
                for scope in all_scopes_used:
                    ex = ex_full.copy()
                    ex[~_mask_2d_for_scope(scope)] = np.nan
                    caption = {"within_only": "within-half blocks only; white = not fitted",
                               "across_only": "across-half block only; white = not fitted",
                               "full_no_diag": "full 2n x 2n (W1 | A | W2)"}[scope]
                    mc.analyse.my_RSA.plot_instruction_RDM(
                        ex, all_lab, all_lab,
                        title=f'data, SL #{sl_idx}\n{SCOPE_CAPTION[scope]}',
                        xlabel='', ylabel='',
                        n_first_half=len(th1_labels),
                        save_path=f"{results_dir}_data_{SCOPE_TAGS[scope]}")
    
        plt.show(block=False)
    
    #
    # Step 4: compute the data RDM per searchlight (cosine dissim).
    #   legacy 'across_only' : (n_conds, n_conds) off-block, ravel'd
    #   everything else      : (2n, 2n) symmetric, strict lower tri
    # The cache filename carries a '_full' suffix in the second case so the
    # two variants live side-by-side and never overwrite each other. The
    # assembled version is computed ONCE; 'within_only' and 'across_only' are
    # then column subsets of it (`_scope_cells`), which is what lets one combo
    # be fitted in several scopes without recomputing any searchlight.
    os.makedirs(data_rdm_dir, exist_ok=True)
    cache_tag = "" if legacy_across_block else "_full"
    data_rdm_name = f"data_RDM{cache_tag}"
    data_rdm_npy = f"{data_rdm_dir}/{data_rdm_name}.npy"
    if not os.path.exists(data_rdm_npy):
        if legacy_across_block:
            data_RDMs_cached = mc.analyse.my_RSA.get_instruction_RDM_per_searchlight(
                data_concat, centers, neighbors)
        else:
            data_RDMs_cached = mc.analyse.my_RSA.get_full_instruction_RDM_per_searchlight(
                data_concat, centers, neighbors)
        mc.analyse.handle_MRI_files.save_data_RDM_as_nifti(
            data_RDMs_cached, data_rdm_dir, data_rdm_name, ref_img, centers)
    else:
        data_RDMs_cached = np.load(data_rdm_npy)

    if smoothing == True:
        smooth_name = f"data_RDM_smooth_fwhm{fwhm}{cache_tag}"
        smooth_npy = f"{data_rdm_dir}/{smooth_name}.npy"
        if not os.path.exists(smooth_npy):
            path_to_save_smooth = f"{data_rdm_dir}/{smooth_name}"
            print(f"now smoothing the RDM and saving it here: {path_to_save_smooth}")
            data_RDMs_cached = mc.analyse.handle_MRI_files.smooth_RDMs(
                data_RDMs_cached, ref_img, fwhm, use_rsa_toolbox=False,
                path_to_save=path_to_save_smooth, centers=centers)
        else:
            data_RDMs_cached = np.load(smooth_npy)

    if not legacy_across_block:
        assert data_RDMs_cached.shape[1] == within_mask.size, (
            f"cached data RDM has {data_RDMs_cached.shape[1]} cells but the "
            f"(2n x 2n) lower triangle has {within_mask.size}")

    _data_RDM_by_scope = {}
    def get_data_RDMs(scope):
        """Data RDMs for one scope, cut from the single cached matrix. Cached
        per scope so a combo fitted in two scopes still loads nothing twice."""
        if scope not in _data_RDM_by_scope:
            if legacy_across_block:
                assert scope == "across_only", (
                    f"scope {scope!r} is not available in the legacy top-level "
                    "'across_only' mode — set data_rdm_scope to 'within_only' "
                    "or 'full_no_diag'.")
                _data_RDM_by_scope[scope] = data_RDMs_cached
            else:
                _data_RDM_by_scope[scope] = data_RDMs_cached[:, _scope_cells(scope)]
            print(f"[data RDM] scope={scope}, cells per searchlight = "
                  f"{_data_RDM_by_scope[scope].shape[1]}")
        return _data_RDM_by_scope[scope]

    data_RDMs = get_data_RDMs(data_rdm_scope)

    #
    # Step 5: evaluate each single model against every searchlight data RDM.
    # NaN cells in the simple model automatically drop the corresponding data
    # cells from the OLS (see evaluate_model_vec). Both X and Y are z-scored.
    #
    # Helper: pick the correct model matrix and flatten it in a way that
    # matches the current data RDM layout — full ravel of the (n, n) A block
    # in 'across_only' mode, strict lower-tri of the (2n, 2n) full block in
    # 'full_no_diag' mode.
    # ── Resume: skip maps that already ran to completion ─────────────────
    # A killed job (walltime, memory) leaves a partial results/ folder. Without
    # this, every resubmission recomputes all 28 fits from scratch and dies at
    # the same wall-clock point, so it never converges.
    #
    # Skipping is only safe if the maps already there were made with the SAME
    # settings -- otherwise a rerun silently mixes two analyses in one folder.
    # The settings summary is written at the very END of a run, so a killed job
    # never leaves one; this writes a `_run_settings.json` at the START instead,
    # and resumes only when it matches. A folder with maps but no such file
    # predates this mechanism and cannot be verified, so by default it is NOT
    # resumed -- set "resume_unverified": true in the config to trust it anyway.
    run_settings = {
        "EV_string": EV_string, "regression_version": regression_version,
        "TR": TR, "RDM_version": RDM_version, "data_rdm_scope": data_rdm_scope,
        "smoothing": smoothing, "fwhm": fwhm,
        "searchlight_mask": searchlight_mask,
        "selected_models": selected_models,
        "single_model_scopes": single_scope_cfg,
        "run_single_models": run_single_models,
        "run_combo_models": run_combo_models,
        "combo_models": combo_cfg,
    }
    run_settings_path = os.path.join(results_dir, f"{sub}_run_settings.json")
    resume = config.get("resume", True)
    resume_unverified = config.get("resume_unverified", False)
    previous = None
    if os.path.exists(run_settings_path):
        try:
            with open(run_settings_path) as f:
                previous = json.load(f)
        except ValueError:
            previous = None
    else:
        # No run-settings file, but a folder left by a run that FINISHED has a
        # settings summary, which records the same settings under its own key
        # names. Comparing against that is what stops `resume_unverified` from
        # trusting maps that belong to a different analysis: it only ever
        # applies when there is no evidence at all.
        summary_path = os.path.join(results_dir, f"{sub}_settings_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path) as f:
                    old = json.load(f)
                shared = [k for k in run_settings if k != "single_model_scopes"]
                key_in_summary = {"selected_models": "models_evaluated"}
                if all(key_in_summary.get(k, k) in old for k in shared):
                    previous = dict(run_settings)
                    for k in shared:
                        previous[k] = old[key_in_summary.get(k, k)]
            except ValueError:
                pass

    if not resume:
        resume_ok, why = False, "resume disabled in config"
    elif previous == run_settings:
        resume_ok, why = True, "settings match the previous run"
    elif previous is not None:
        resume_ok, why = False, ("settings DIFFER from the previous run -- every "
                                 "map is recomputed so the folder cannot end up "
                                 "holding two analyses")
    elif resume_unverified:
        resume_ok, why = True, ("no run-settings file, but resume_unverified is "
                                "set -- trusting the existing maps")
    else:
        resume_ok, why = False, ("no run-settings file to verify the existing "
                                 "maps against; set resume_unverified to trust them")
    print(f"\n[resume] {'ON' if resume_ok else 'OFF'} -- {why}")
    with open(run_settings_path, "w") as f:
        json.dump(run_settings, f, indent=2)

    skipped_maps = []

    def already_done(out_name):
        """True if this map's three volumes are all present and non-empty."""
        if not resume_ok:
            return False
        for suffix in ("beta", "t_val", "p_val"):
            f = os.path.join(results_dir, f"{out_name}_{suffix}.nii.gz")
            if not os.path.exists(f) or os.path.getsize(f) == 0:
                return False
        return True

    # Pre-flight: every design that is about to be fitted must be full rank,
    # otherwise the OLS returns NaN and the maps are written as all-zero.
    # Checked here, once, before any searchlight OLS runs.
    print("\n=== design checks ===")
    bad = []
    _to_check = []
    if run_single_models:
        for m in selected_models:
            for sc in single_model_scopes(m, single_scope_cfg, data_rdm_scope)[0]:
                _to_check.append((f"single '{m}' [{sc}]", [m], sc))
    if run_combo_models:
        for c in combo_cfg:
            for sc in combo_scopes(c, data_rdm_scope)[0]:
                _to_check.append((f"combo '{c['name']}' [{sc}]", c["regressors"], sc))
    for label, regs, sc in _to_check:
        ok, msg = design_rank_report(
            regs, np.stack([_model_regressor(m, sc) for m in regs], axis=1))
        print(f"  {'OK  ' if ok else 'FAIL'} {label:44s} {msg}")
        if not ok:
            bad.append(f"{label}: {msg}")
    if bad:
        raise ValueError(
            "Rank-deficient design(s) — evaluate_model would return NaN for "
            "every regressor and the saved maps would be all-zero:\n  "
            + "\n  ".join(bad)
            + f"\n(current data_rdm_scope = {data_rdm_scope!r}; instruction "
              "models carry no variance in the across-half block, so they must "
              "be fitted in 'within_only' — either as the config-level scope or "
              "via a per-combo \"scope\")")

    RSA_results = {}
    if run_single_models == True:
        for model in selected_models:
            # With `single_model_scopes` in the config, each model is fitted once
            # per scope it is entitled to — execution models within AND across
            # task halves, instruction models within only — and the outputs carry
            # a `_within` / `_across` / `_full` suffix so they cannot collide.
            scopes, tag_outputs = single_model_scopes(
                model, single_scope_cfg, data_rdm_scope)
            for scope in scopes:
                out_name = (f"{model}_{SCOPE_TAGS[scope]}"
                            if tag_outputs else model)
                if already_done(out_name):
                    print(f"  [resume] skipping {out_name}, already complete")
                    skipped_maps.append(out_name)
                    continue
                model_flat = _model_regressor(model, scope)
                RSA_results[out_name] = Parallel(n_jobs=3)(
                    delayed(mc.analyse.my_RSA.evaluate_model)(model_flat, d)
                    for d in tqdm(get_data_RDMs(scope),
                                  desc=f"running GLM for all searchlights in {out_name}")
                )
                mc.analyse.handle_MRI_files.save_my_RSA_results(
                    result_file=RSA_results[out_name], centers=centers,
                    file_path=results_dir, file_name=f"{out_name}",
                    mask=mask, number_regr=0, ref_image_for_affine_path=ref_img,
                )

    # import pdb; pdb.set_trace()
    # Regressor correlations of every combo, per scope — printed AND written to
    # the settings summary json, so the collinearity of a fit is recoverable per
    # subject without re-reading stdout.
    combo_regressor_correlations = {}
    if run_combo_models:
        combo_list = combo_cfg
        for combo in combo_list:
            combo_model_name = combo["name"]
            models_to_combine = combo["regressors"]
            # check if these models have been computed in model_EVs
            missing = [m for m in models_to_combine if m not in model_RDM_dir]
            if missing:
                for m_int in missing:
                    if m_int.endswith('interaction'):
                        curr_m = m_int.split('_interaction')[0]
                        z = lambda v: (v - np.nanmean(v)) / np.nanstd(v)
                        model_RDM_dir[m_int] = [z(model_RDM_dir[curr_m][0]) * z(model_RDM_dir['path_rew'][0])]
                        # model_RDM_dir[m_int] = [model_RDM_dir[curr_m][0]*model_RDM_dir['path_rew'][0]]
                    else:
                        raise ValueError(f"Combo model {combo_model_name} not possible, as {missing} not computed")

            # A combo may declare its own "scope" (string or list) and is then
            # fitted once per scope — e.g. the pure-execution split model once
            # within and once across task halves. Those outputs carry a
            # `_within` / `_across` / `_full` suffix so they cannot collide.
            scopes, tag_outputs = combo_scopes(combo, data_rdm_scope)
            for scope in scopes:
                combo_out_name = (f"{combo_model_name}_{SCOPE_TAGS[scope]}"
                                  if tag_outputs else combo_model_name)
                print(f"running combo model {combo_out_name} (scope = {scope})")
                combo_data_RDMs = get_data_RDMs(scope)

                # Each regressor is subset from the (2n, 2n) block matrix with the
                # same cell mask as the data RDM, so model and data always line up.
                stacked_model_RDMs = np.stack(
                    [_model_regressor(m, scope) for m in models_to_combine],
                    axis=1,
                )

                # How correlated are the regressors of this combo model with each other?
                # NaN-safe pearson via pandas; then print a compact upper-triangle summary
                # and save a heatmap alongside the results.
                import pandas as _pd
                corr = _pd.DataFrame(stacked_model_RDMs, columns=models_to_combine).corr().to_numpy()
                pairwise = {}
                print(f"\n[{combo_out_name}] pairwise Pearson r between regressor RDMs:")
                for i in range(len(models_to_combine)):
                    for j in range(i + 1, len(models_to_combine)):
                        pairwise[f"{models_to_combine[i]}~{models_to_combine[j]}"] = round(float(corr[i, j]), 4)
                        print(f"    {models_to_combine[i]:>16s} vs {models_to_combine[j]:<16s}: r = {corr[i, j]:+.3f}")
                combo_regressor_correlations[combo_out_name] = {
                    "combo": combo_model_name,
                    "scope": scope,
                    "regressors": list(models_to_combine),
                    "n_cells": int(stacked_model_RDMs.shape[0]),
                    "pairwise_pearson_r": pairwise,
                    "correlation_matrix": np.round(corr, 4).tolist(),
                }
                mc.analyse.my_RSA.plot_model_correlations(
                    stacked_model_RDMs, models_to_combine,
                    save_path=f"{results_dir}_{combo_out_name}_regressor_corr",
                    show=True,
                )

                # One OLS produces every regressor's map, so the combo is only
                # skipped when ALL of them are already there. The correlation
                # record above is computed either way -- it is cheap and belongs
                # in the settings summary regardless.
                combo_out_names = [f"{m.upper()}-{combo_out_name}"
                                   for m in models_to_combine]
                if all(already_done(n) for n in combo_out_names):
                    print(f"  [resume] skipping combo {combo_out_name}, "
                          f"all {len(combo_out_names)} maps already complete")
                    skipped_maps.extend(combo_out_names)
                    continue

                estimates_combined_model_rdms = Parallel(n_jobs=3)(delayed(mc.analyse.my_RSA.evaluate_model)(stacked_model_RDMs, d) for d in tqdm(combo_data_RDMs, desc=f"running GLM for all searchlights in {combo_out_name}"))
                for i, model in enumerate(models_to_combine):
                    # TODO: Change the type of similarity to not throw away half of the matrix.
                    mc.analyse.handle_MRI_files.save_my_RSA_results(result_file=estimates_combined_model_rdms, centers=centers, file_path = results_dir, file_name= f"{model.upper()}-{combo_out_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
            
    

    # --- SETTINGS SUMMARY (per subject) ---
    summary = {
        "subject": sub,
        "EV_string": EV_string,
        "regression_version": regression_version,
        "TR": TR,
        "regression_version_full": regression_version_full,
        "RDM_version": RDM_version,
        "paired_labels": paired_labels,
        "smoothing": smoothing,
        "fwhm": fwhm,
        "searchlight_mask": searchlight_mask,
        "data_rdm_scope": data_rdm_scope,
        "n_cells_per_searchlight": int(data_RDMs.shape[1]),
        "n_all_EVs": len(all_EV_keys),
        "n_selected_EVs": len(EV_keys),
        "models_evaluated": selected_models,
        "run_single_models": run_single_models,
        "scopes_per_model": scopes_per_model,
        "run_combo_models": run_combo_models,
        "combo_models": combo_cfg,
        "combo_scopes": {c["name"]: combo_scopes(c, data_rdm_scope)[0] for c in combo_cfg},
        # Collinearity of every fit that was actually run, per combo per scope.
        "combo_regressor_correlations": combo_regressor_correlations,
        # Execution-vs-instruction shared variance, per model pair per scope.
        "exec_vs_instr_correlations": exec_instr_correlations,
        # Which maps this invocation reused rather than recomputed. Empty on a
        # first run; non-empty means the folder was completed across several
        # jobs, all of them under the settings recorded above.
        "resume_enabled": bool(resume_ok),
        "resumed_maps": skipped_maps,
        "n_resumed_maps": len(skipped_maps),
        "data_dir": data_dir,
        "results_dir": results_dir
    }

    print("\n=== SETTINGS SUMMARY ===")
    for k, v in summary.items():
        if k in ("combo_regressor_correlations", "exec_vs_instr_correlations"):
            print(f"{k:>20}: {len(v)} entries (see json)")
        else:
            print(f"{k:>20}: {v}")
    
    # Save a copy alongside results for provenance
    with open(os.path.join(results_dir, f"{sub}_settings_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"(Saved summary → {os.path.join(results_dir, f'{sub}_settings_summary.json')})\n")
            

