#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit the instruction-phase RSA pipeline: which (subject x epoch x model x stage)
outputs exist, and which jobs still need running.

`fMRI_run_RSA_instruction.py` is expensive (one searchlight pass per subject per
epoch) and it reads the first-level GLMs directly, so submitting it blindly
wastes queue time two ways: on subjects whose epoch GLM never finished (the job
dies on a missing PE, or worse, reads a half-written one), and on subjects that
are already done. This walks the (subject x epoch) grid and sorts it into:

    NOT_STARTED      all inputs present, no maps yet             -> submit
    RESULTS_INCOMPLETE  some beta maps are there and some are not, or all are
                     there but the settings summary is not (the run died
                     before finishing)                          -> submit
    RERUN_CHANGED    a result exists but was made with DIFFERENT settings
                     than this config asks for                  -> submit
                     (it overwrites the old maps; --skip-changed leaves it)
    SMOOTHED_INCOMPLETE / STANDARD_INCOMPLETE
                     the RSA is finished but a DOWNSTREAM stage is not. These
                     do NOT need the RSA rerun -- rerun the smoothing or the
                     applywarp wrapper, which skip what already exists.
    DONE             every stage complete and the settings match -> skip
    GLM_NOT_READY    the epoch GLM has no complete run in the base directory
                     the RSA reads                              -> skip
    MISSING_INPUT    modelled EVs / reference image / mask absent -> skip

THE PIPELINE, per model. Three per-subject stages and two group stages:
    results/                {map}_beta.nii.gz
    smoothed/               smooth_fwhm{N}_{map}_beta.nii.gz
    standard-space-smooth/  smooth_fwhm{N}_{map}_beta_std.nii.gz
    group/..._glmbase_{epoch}/          masked_smooth_fwhm{N}_{map}_beta_std.nii
    group/..._glmbase_{epoch}_cropped/  cropped_masked_..._beta_std.nii.gz
The last name is loso.py's BETA_STEM, which closes the loop to the group
statistics. Only the beta map is tracked: it is what the group merge collects
(`*beta_std.nii.gz`) and what loso.py finally reads, so a missing beta is what
actually breaks the pipeline; the t_val / p_val maps are written in the same
call and travel with it. `{map}` is every name the config implies --
`{model}`, `{model}_within` / `_across` when `single_model_scopes` entitles a
model to several scopes, and `{REGRESSOR}-{combo}` for each combo regressor --
built by `expected_map_names`, which mirrors fMRI_run_RSA_instruction.py.

GLM readiness is not re-implemented here: it calls `check_GLMs_ran.check_one`,
the same function that produces the FEAT audit, so 'complete' means exactly the
same thing in both places. Note that PROMOTE_TWIN does NOT count as ready --
that status means the finished run is sitting in a '+' twin while the base
directory the RSA reads is broken, so the cleanup has to run first.

'ALREADY DONE' MEANS DONE *WITH THESE SETTINGS*
    The completion marker is `{sub}_settings_summary.json`, which
    fMRI_run_RSA_instruction.py writes as its very last action -- so its
    presence means the whole run finished, not just some of the maps. Its
    contents are then compared field by field against what this config asks
    for (`COMPARED_KEYS`). A result built with other models, another scope,
    different smoothing or a different GLM is NOT treated as done. A summary
    written by an older version of the RSA script that lacks a compared field
    also counts as changed: the settings cannot be shown to match.

WHAT IT WRITES  (--out-dir, default analysis/logs_mid_sept/rsa_audit_<name>_<date>/)
    report.txt        the per-stage table, and every (subject, epoch) that is
                      not complete, with the reason
    missing_maps.txt  one line per individual map that is not on disk
                      (subject, epoch, stage, filename)
    todo_rsa.txt      'subject epoch_config' per RSA job to submit
    resubmit_plan.sh  the commands that fill every gap, per wrapper, in
                      pipeline order. Each stage lists only work whose
                      PREDECESSOR is complete, and a run that is going back
                      through the RSA is kept out of the downstream stages (its
                      maps are about to be rewritten). The stages are dependent,
                      so run one, wait for the queue, then re-run this audit for
                      the next plan -- it is not a script to run top to bottom.
    settings.json   what this audit was run with
  and one per-epoch config snapshot per epoch, next to the base config:
    <base>_<epoch>.json   = the base config with regression_version set to the
                            epoch GLM and TR set to null, which is what makes
                            fMRI_run_RSA_instruction.py read
                            glm_instr_<epoch>_pt0{1,2}.feat.

USAGE
    python3 check_RSA_ran.py                      # audit, write the todo list
    python3 check_RSA_ran.py --skip-changed       # leave existing results alone
    python3 check_RSA_ran.py --subjects 01 02     # a subset
    bash submit_RSA_instruction_epochs.sh         # runs this, then submits

@author: Svenja Kuechenhoff
"""

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_GLMs_ran as glmcheck

# A GLM is usable by the RSA only if the BASE directory is complete: that is the
# only name my_RSA.load_data_EVs_instr_TRwise ever opens. 'PROMOTE_TWIN' means a
# twin finished but the base did not, so it is deliberately not in this set.
GLM_READY = {'OK', 'DUPLICATES'}

# statuses that are fixed by rerunning the RSA itself, rather than one of the
# cheap downstream wrappers
NEEDS_RSA = {'NOT_STARTED', 'RESULTS_INCOMPLETE', 'RERUN_CHANGED'}

# Settings that must match for an existing result to count as done. These are
# the keys of fMRI_run_RSA_instruction.py's summary that come from the config
# rather than from the data (n_cells_per_searchlight, paired_labels etc. are
# outputs, not settings, so they are not compared).
COMPARED_KEYS = ['EV_string', 'regression_version', 'TR', 'regression_version_full',
                 'RDM_version', 'smoothing', 'fwhm', 'searchlight_mask',
                 'data_rdm_scope', 'models_evaluated', 'run_single_models',
                 'run_combo_models', 'combo_models']

# Mirrors SCOPE_ALIASES in fMRI_run_RSA_instruction.py: the config may use the
# short form, the summary always stores the canonical one.
SCOPE_ALIASES = {'across_only': 'across_only', 'across': 'across_only',
                 'within_only': 'within_only', 'within': 'within_only',
                 'full_no_diag': 'full_no_diag', 'full': 'full_no_diag'}


# Suffix the RSA appends to a map name when a model is fitted in more than one
# scope, and the marker of an instruction-similarity model. Both mirror
# fMRI_run_RSA_instruction.py.
SCOPE_TAGS = {'across_only': 'across', 'within_only': 'within',
              'full_no_diag': 'full'}
INSTR_SUFFIX = '_instr'

# The three per-subject stages, in pipeline order. Each is a directory under
# the RSA folder plus the name one model's map carries there. Only the BETA map
# is tracked: it is what the group merge collects (`*beta_std.nii.gz`) and what
# loso.py finally reads, so a missing beta is what actually breaks the pipeline.
# The t_val / p_val maps are written in the same call and travel with it.
STAGE_ORDER = ('results', 'smoothed', 'standard')
STAGE_DIRS = {'results': 'results', 'smoothed': 'smoothed',
              'standard': 'standard-space-smooth'}
STAGE_MADE_BY = {
    'results': 'fMRI_run_RSA_instruction.py (submit_RSA_instruction_epochs.sh)',
    'smoothed': 'smooth_subject_space.py (wrapper_smooth_stat_maps_subj.sh)',
    'standard': 'applywarp (transform_smooth_subject_res_to_standard.sh)'}


# The two GROUP stages, after the per-subject ones. These are per (epoch, map)
# rather than per subject: merge_subj_to_group.sh stacks all subjects into one
# 4-D file, mask_subj_by_missingvoxels.sh then crops it. Names follow those
# scripts: masked_${filename} and cropped_masked_${filename}, where ${filename}
# is the subject-level standard-space name. The cropped name is exactly
# loso.py's BETA_STEM, which is what closes the loop to the group statistics.
GROUP_STAGE_ORDER = ('merged', 'cropped')
GROUP_STAGE_MADE_BY = {'merged': 'merge_subj_to_group.sh',
                       'cropped': 'mask_subj_by_missingvoxels.sh'}


def group_dir_for(config, epoch, cropped=False):
    """derivatives/group/group_RSA_{name}_glmbase_{epoch}[_cropped]"""
    want = expected_settings(config)
    d = (f"{glmcheck.data_dir_deriv}/group/"
         f"group_RSA_{want['RDM_version']}_glmbase_{epoch}")
    return f"{d}_cropped" if cropped else d


def group_stage_filename(stage, map_name, fwhm):
    base = f"masked_smooth_fwhm{fwhm}_{map_name}_beta_std.nii.gz"
    return base if stage == 'merged' else f"cropped_{base}"


def check_group_stages(config, epoch, map_names, fwhm):
    """Per group stage: how many of the expected maps exist for this epoch."""
    out = {}
    for stage in GROUP_STAGE_ORDER:
        d = group_dir_for(config, epoch, cropped=(stage == 'cropped'))
        listing = set(os.listdir(d)) if os.path.isdir(d) else set()
        missing = [m for m in map_names
                   if not _present(listing, group_stage_filename(stage, m, fwhm))]
        out[stage] = dict(dir=d, n_expected=len(map_names),
                          n_missing=len(missing), missing=missing)
    return out


def stage_filename(stage, map_name, fwhm):
    """What one model's beta map is called at one stage."""
    if stage == 'results':
        return f"{map_name}_beta.nii.gz"
    if stage == 'smoothed':
        return f"smooth_fwhm{fwhm}_{map_name}_beta.nii.gz"
    return f"smooth_fwhm{fwhm}_{map_name}_beta_std.nii.gz"


def _scopes_for_single(model, cfg_scopes, default):
    """Mirrors single_model_scopes() in fMRI_run_RSA_instruction.py."""
    if not cfg_scopes:
        return [default], False
    key = 'instruction' if model.endswith(INSTR_SUFFIX) else 'execution'
    raw = cfg_scopes.get(key, default)
    raw = [raw] if isinstance(raw, str) else list(raw)
    return [SCOPE_ALIASES[x] for x in raw], True


def _scopes_for_combo(combo, default):
    """Mirrors combo_scopes() in fMRI_run_RSA_instruction.py."""
    if "scope" not in combo:
        return [default], False
    raw = combo["scope"]
    raw = [raw] if isinstance(raw, str) else list(raw)
    return [SCOPE_ALIASES[x] for x in raw], True


def expected_map_names(config):
    """Every map name the RSA writes, for this config.

    Mirrors the output naming of fMRI_run_RSA_instruction.py: a single model is
    `{model}`, or `{model}_{within|across|full}` when `single_model_scopes`
    entitles it to more than one scope; a combo regressor is
    `{REGRESSOR}-{combo}` with the same optional scope suffix on the combo
    name. Keep in step with that script -- it is the only duplicated logic
    here, and a mismatch shows up as a stage that never looks complete."""
    want = expected_settings(config)
    default = want['data_rdm_scope']
    cfg_scopes = config.get("single_model_scopes", None)
    names = []
    if want['run_single_models']:
        for m in want['models_evaluated']:
            scopes, tagged = _scopes_for_single(m, cfg_scopes, default)
            for sc in scopes:
                names.append(f"{m}_{SCOPE_TAGS[sc]}" if tagged else m)
    if want['run_combo_models']:
        for combo in want['combo_models']:
            scopes, tagged = _scopes_for_combo(combo, default)
            for sc in scopes:
                out = (f"{combo['name']}_{SCOPE_TAGS[sc]}" if tagged
                       else combo['name'])
                for m in combo['regressors']:
                    names.append(f"{m.upper()}-{out}")
    return names


def _present(listing, fname):
    """Accept the .nii twin of a .nii.gz name, as the rest of the code does."""
    return fname in listing or fname[:-3] in listing


def check_stages(rsa_dir, map_names, fwhm):
    """Per stage: how many of the expected maps are on disk, and which are not.
    One listdir per stage directory rather than a stat per map."""
    out = {}
    for stage in STAGE_ORDER:
        d = f"{rsa_dir}/{STAGE_DIRS[stage]}"
        listing = set(os.listdir(d)) if os.path.isdir(d) else set()
        missing = [m for m in map_names
                   if not _present(listing, stage_filename(stage, m, fwhm))]
        out[stage] = dict(dir=d, exists=bool(listing), n_expected=len(map_names),
                          n_missing=len(missing), missing=missing)
    return out


def expected_settings(config):
    """What fMRI_run_RSA_instruction.py would write into its summary, given this
    config. Defaults repeated from the script -- keep them in step with it."""
    TR = config.get("TR")
    regression_version = config.get("regression_version")
    combo_cfg = [dict(c) for c in config.get("combo_models", [])]
    if config.get("add_block_nuisance", False):
        for combo in combo_cfg:
            if 'block' not in combo["regressors"]:
                combo["regressors"] = list(combo["regressors"]) + ['block']
    return {
        'EV_string': config.get("load_EVs_from"),
        'regression_version': regression_version,
        'TR': TR,
        'regression_version_full': (regression_version if TR is None
                                    else f"{regression_version}-TR{TR}"),
        'RDM_version': config.get("name_of_RSA"),
        'smoothing': config.get("smoothing", True),
        'fwhm': config.get("fwhm", 5),
        'searchlight_mask': config.get("searchlight_mask", None),
        'data_rdm_scope': SCOPE_ALIASES[config.get("data_rdm_scope", "across_only")],
        'models_evaluated': config.get("selected_models", ['DSR', 'rewDSR', 'simple']),
        'run_single_models': config.get("run_single_models", True),
        'run_combo_models': config.get("run_combo_models", bool(combo_cfg)),
        'combo_models': combo_cfg,
    }


def rsa_dir_for(data_dir, config):
    """The RSA folder itself -- results/, smoothed/ and standard-space-smooth/
    all live under it."""
    want = expected_settings(config)
    base = f"{data_dir}/func/RSA_{want['RDM_version']}_glmbase_{want['regression_version_full']}"
    if want['smoothing']:
        base = f"{base}_smooth{want['fwhm']}"
    return base


def results_dir_for(data_dir, config):
    """The same path fMRI_run_RSA_instruction.py builds."""
    return f"{rsa_dir_for(data_dir, config)}/results"


def settings_differences(summary, want):
    """Which compared settings disagree. A key missing from the summary counts
    as a difference: an older run cannot be shown to have used these settings."""
    diffs = []
    for k in COMPARED_KEYS:
        if k not in summary:
            diffs.append(f"{k}: absent from summary")
        elif summary[k] != want[k]:
            diffs.append(f"{k}: {summary[k]!r} != {want[k]!r}")
    return diffs


def missing_inputs(data_dir, sub, config):
    """Inputs the RSA opens directly, other than the GLM itself."""
    missing = []
    EV_string = config.get("load_EVs_from")
    pkl = f"{data_dir}/beh/modelled_EVs/{sub}_modelled_EVs_{EV_string}.pkl"
    if not os.path.exists(pkl):
        missing.append(f"modelled EVs: {pkl}")
    ref = f"{data_dir}/func/preproc_clean_01.feat/example_func.nii.gz"
    if not os.path.exists(ref):
        missing.append(f"reference image: {ref}")
    mask_kind = config.get("searchlight_mask", None)
    if mask_kind == 'grey_matter':
        m = f"{data_dir}/anat/grey_matter_mask_func_01.nii.gz"
    elif mask_kind == 'no_CSF':
        m = f"{data_dir}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz"
    else:
        m = None
    if m and not os.path.exists(m):
        missing.append(f"searchlight mask: {m}")
    return missing


def check_one_rsa(sub_tag, glm, config, map_names=None, fwhm=5):
    """(status, detail, stages) for one (subject, epoch).

    `stages` is the per-stage record from check_stages, or None when the run
    never got far enough for the stages to mean anything (inputs missing)."""
    sub = f"sub-{sub_tag}"
    data_dir = f"{glmcheck.data_dir_deriv}/{sub}"

    # 1. the GLM the RSA reads, both task halves, via the FEAT audit itself
    for th in (1, 2):
        status, detail, _ = glmcheck.check_one(sub_tag, th, glm)
        if status not in GLM_READY:
            if status == 'PROMOTE_TWIN':
                detail = ("the finished run is in a '+' twin, the base directory "
                          "the RSA reads is not usable -- run the FEAT cleanup first")
            return 'GLM_NOT_READY', f"pt{th} {status}: {detail}", None

    # 2. everything else the script opens
    missing = missing_inputs(data_dir, sub, config)
    if missing:
        return 'MISSING_INPUT', '; '.join(missing), None

    # 3. walk the three stages, per model
    if map_names is None:
        map_names = expected_map_names(config)
    rsa_dir = rsa_dir_for(data_dir, config)
    stages = check_stages(rsa_dir, map_names, fwhm)
    res = stages['results']

    if res['n_missing'] == res['n_expected']:
        return 'NOT_STARTED', f"no maps under {rsa_dir}", stages
    if res['n_missing']:
        return 'RESULTS_INCOMPLETE', (
            f"{res['n_missing']}/{res['n_expected']} beta maps missing "
            f"(first: {res['missing'][0]})"), stages

    # The settings summary is the RSA's last action, so results can be complete
    # while the run still died before writing it.
    summary_path = f"{rsa_dir}/results/{sub}_settings_summary.json"
    if not os.path.exists(summary_path):
        return 'RESULTS_INCOMPLETE', ("all beta maps present but no settings "
                                      "summary -- the run did not finish"), stages
    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except (ValueError, OSError) as e:
        return 'RERUN_CHANGED', f"summary unreadable ({e}), treating as not done", stages
    diffs = settings_differences(summary, expected_settings(config))
    if diffs:
        return 'RERUN_CHANGED', '; '.join(diffs[:3]) + (
            f" (+{len(diffs) - 3} more)" if len(diffs) > 3 else ""), stages

    # 4. the two downstream stages. These do not need the RSA rerun -- they need
    # their own wrapper rerun, which is a different and much cheaper fix.
    for stage in ('smoothed', 'standard'):
        st = stages[stage]
        if st['n_missing']:
            return f"{stage.upper()}_INCOMPLETE", (
                f"{st['n_missing']}/{st['n_expected']} maps missing "
                f"(first: {st['missing'][0]})"), stages
    return 'DONE', summary_path, stages


PLAN_HEADER = """#!/bin/sh
# Resubmission plan written by check_RSA_ran.py on {stamp}.
#
# The stages are ORDERED and each depends on the one before it, so this is not
# a script to run top to bottom in one go. Run one stage, wait for the queue to
# drain, then re-run the audit -- it will write the next plan from what is then
# on disk. Only work that is actually missing appears here, and only where the
# preceding stage is already complete, so nothing is submitted that would read
# half-written inputs.
#
# repo   {repo}
# audit  {audit}
set -e
cd "{fmri_dir}"

"""


def build_plan(stage_rows, group_rows, epochs, subjects, todo_path, fmri_dir,
               smooth_configs):
    """The commands that would fill every gap, per wrapper, in pipeline order.

    Returns [(stage, human summary, [command lines])]. A stage only lists work
    whose predecessor is complete: submitting the smoothing for a subject whose
    RSA has not finished would just read an empty results/ directory."""
    plan = []

    # 1. the RSA itself -- one job per (subject, epoch), driven by todo_rsa.txt
    n_rsa = sum(1 for _ in open(todo_path)
                if _.strip() and not _.startswith('#')) if os.path.exists(todo_path) else 0
    if n_rsa:
        plan.append(('results', f"{n_rsa} RSA job(s)",
                     [f"bash submit_RSA_instruction_epochs.sh {todo_path}"]))

    # 2. smoothing -- results complete, smoothed not. Submitted per (subject,
    #    epoch) through the job wrapper, which is what activates conda. The
    #    wrapper_smooth_stat_maps_subj.sh loop would redo every subject of the
    #    epoch instead: smooth_subject_space.py has no skip-if-exists.
    smooth_cmds = [
        f"fsl_sub -q short bash update_fMRI/wrapper_python_fMRI_RSA_clean_config.sh "
        f"{sub} {smooth_configs[glm]} smooth_subject_space.py"
        for sub, glm in stage_rows['smoothed']]
    if smooth_cmds:
        plan.append(('smoothed', f"{len(smooth_cmds)} (subject, epoch) to smooth",
                     smooth_cmds))

    # 3. standard space -- per epoch: that wrapper loops subjects itself and
    #    skips outputs that already exist, so naming the epochs is enough.
    std_epochs = sorted({glm for _, glm in stage_rows['standard']})
    if std_epochs:
        plan.append(('standard',
                     f"{len(stage_rows['standard'])} (subject, epoch) missing, "
                     f"in {len(std_epochs)} epoch(s)",
                     ["bash transform_smooth_subject_res_to_standard.sh \\\n    "
                      + " \\\n    ".join(std_epochs)]))

    # 4/5. group stages -- only for epochs with NO subject-level gap left,
    #      otherwise the merge stacks an incomplete set and
    #      mask_subj_by_missingvoxels.sh rejects it on required_n.
    for stage, script in (('merged', 'merge_subj_to_group.sh'),
                          ('cropped', 'mask_subj_by_missingvoxels.sh')):
        eps = sorted(group_rows[stage])
        if eps:
            plan.append((stage, f"{len(eps)} epoch(s)",
                         [f"bash {script} \\\n    " + " \\\n    ".join(eps)]))
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base-config', default='rsa_instruction_cumulative_rew.json',
                    help='RSA config the per-epoch snapshots are derived from')
    ap.add_argument('--ev-config', default='EV_config_instruction.json',
                    help='EV config the epoch names come from')
    ap.add_argument('--epochs', nargs='+', default=None,
                    help='full GLM names, e.g. instr_see-A-first. Default: all from --ev-config')
    ap.add_argument('--subjects', nargs='+', default=glmcheck.DEFAULT_SUBJECTS)
    ap.add_argument('--data-dir', default=None,
                    help='derivatives directory. Default: laptop path if it exists, else cluster')
    ap.add_argument('--config-dir', default=None,
                    help='where the per-epoch config snapshots are written. '
                         'Default: the repo condition_files directory')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--smooth-config', default='smooth5_config.json',
                    help='base config for smooth_subject_space.py; a per-epoch '
                         'snapshot is written next to it, so the resubmission '
                         'plan can smooth one (subject, epoch) at a time without '
                         'the shared config being rewritten under a running job')
    ap.add_argument('--smooth-fwhm', type=int, default=None,
                    help='FWHM in the smoothed/standard-space filenames '
                         '(smooth_fwhm{N}_...). Default: the fwhm in the config.')
    ap.add_argument('--skip-changed', action='store_true',
                    help='do NOT resubmit results whose settings differ (default is to '
                         'resubmit them, which overwrites the old maps)')
    ap.add_argument('--no-write', action='store_true', help='print only, write nothing')
    args = ap.parse_args()

    if args.data_dir:
        glmcheck.data_dir_deriv = args.data_dir.rstrip('/')
    config_dir = args.config_dir or glmcheck.config_path
    stamp = datetime.date.today().isoformat()

    with open(f"{config_dir}/{args.base_config}") as f:
        base_config = json.load(f)
    name_RSA = base_config.get("name_of_RSA")
    out_dir = args.out_dir or f"{glmcheck.logs_dir}/rsa_audit_{name_RSA}_{stamp}"

    epochs = args.epochs or glmcheck.glm_names_from_config(args.ev_config)
    print(f"RSA '{name_RSA}' from {args.base_config}")
    print(f"{len(epochs)} epoch(s) x {len(args.subjects)} subject(s) = "
          f"{len(epochs) * len(args.subjects)} jobs")
    print(f"under {glmcheck.data_dir_deriv}\n")

    # One config snapshot per epoch: the base config pointed at that epoch's GLM.
    # TR = None makes the RSA treat regression_version as the full GLM name.
    epoch_configs = {}
    for glm in epochs:
        epoch_tag = glm.split('_', 1)[1] if '_' in glm else glm
        cfg = dict(base_config)
        cfg['regression_version'] = glm
        cfg['TR'] = None
        epoch_configs[glm] = (f"{os.path.splitext(args.base_config)[0]}_{epoch_tag}.json", cfg)

    map_names = expected_map_names(base_config)
    fwhm = args.smooth_fwhm if args.smooth_fwhm is not None else base_config.get("fwhm", 5)
    print(f"{len(map_names)} map(s) per subject x epoch, checked at each of "
          f"{len(STAGE_ORDER)} stages (fwhm{fwhm})")
    print(f"  {' -> '.join(STAGE_DIRS[st] for st in STAGE_ORDER)}\n")

    rows, todo, counts_all = [], [], {}
    # per stage: how many (subject, epoch) runs are complete / partial / absent,
    # and every individual map that is missing
    stage_tally = {st: dict(complete=0, partial=0, absent=0, n_missing_maps=0)
                   for st in STAGE_ORDER}
    missing_maps = []
    # (subject, epoch) pairs whose PREDECESSOR stage is complete but which are
    # themselves incomplete -- i.e. the work that can actually be submitted now
    stage_rows = {st: [] for st in STAGE_ORDER}
    # epochs with no subject-level gap left, per epoch: safe to take to group
    epoch_subject_complete = {glm: True for glm in epochs}
    for glm in epochs:
        cfg_name, cfg = epoch_configs[glm]
        counts = {}
        for sub in args.subjects:
            status, detail, stages = check_one_rsa(sub, glm, cfg, map_names, fwhm)
            counts[status] = counts.get(status, 0) + 1
            counts_all[status] = counts_all.get(status, 0) + 1
            if status != 'DONE':
                rows.append((glm, sub, status, detail))
            if status in NEEDS_RSA and not (status == 'RERUN_CHANGED' and args.skip_changed):
                todo.append((sub, cfg_name))
            if stages is None:
                epoch_subject_complete[glm] = False
                continue
            if status != 'DONE':
                epoch_subject_complete[glm] = False
            # Work that is ready to submit: this stage incomplete, the one
            # before it complete. A run that is going back through the RSA is
            # excluded from the downstream stages entirely -- its maps are
            # about to be rewritten, so smoothing them now is wasted work.
            prev_ok = status not in NEEDS_RSA
            for st in STAGE_ORDER:
                if stages[st]['n_missing'] and prev_ok:
                    stage_rows[st].append((sub, glm))
                prev_ok = prev_ok and stages[st]['n_missing'] == 0
            for st in STAGE_ORDER:
                rec = stages[st]
                if rec['n_missing'] == 0:
                    stage_tally[st]['complete'] += 1
                elif rec['n_missing'] == rec['n_expected']:
                    stage_tally[st]['absent'] += 1
                else:
                    stage_tally[st]['partial'] += 1
                stage_tally[st]['n_missing_maps'] += rec['n_missing']
                for m in rec['missing']:
                    missing_maps.append((sub, glm, st, stage_filename(st, m, fwhm)))
        n_sub = sum(v for k, v in counts.items()
                    if k in NEEDS_RSA
                    and not (k == 'RERUN_CHANGED' and args.skip_changed))
        other = ', '.join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"  {glm:<40} submit {n_sub:>3}/{len(args.subjects)}   [{other}]")

    # group stages, per epoch. Only an epoch whose subjects are all complete is
    # offered to the merge; a partial set would give the wrong volume count.
    group_tally = {st: dict(complete=0, partial=0, absent=0) for st in GROUP_STAGE_ORDER}
    group_rows = {st: [] for st in GROUP_STAGE_ORDER}
    for glm in epochs:
        gs = check_group_stages(epoch_configs[glm][1], glm, map_names, fwhm)
        prev_ok = epoch_subject_complete[glm]
        for st in GROUP_STAGE_ORDER:
            rec = gs[st]
            key = ('complete' if rec['n_missing'] == 0 else
                   'absent' if rec['n_missing'] == rec['n_expected'] else 'partial')
            group_tally[st][key] += 1
            if rec['n_missing'] and prev_ok:
                group_rows[st].append(glm)
            prev_ok = prev_ok and rec['n_missing'] == 0
            for m in rec['missing']:
                missing_maps.append(('-', glm, st, group_stage_filename(st, m, fwhm)))

    n_runs = len(epochs) * len(args.subjects)
    print(f"\n=== stages, over {n_runs} (subject x epoch) runs ===")
    print(f"    {'stage':<22}{'complete':>9}{'partial':>9}{'none':>7}"
          f"{'maps missing':>14}   made by")
    for st in STAGE_ORDER:
        t = stage_tally[st]
        print(f"    {STAGE_DIRS[st]:<22}{t['complete']:>9}{t['partial']:>9}"
              f"{t['absent']:>7}{t['n_missing_maps']:>14}   {STAGE_MADE_BY[st]}")
    print(f"\n=== group stages, over {len(epochs)} epoch(s) ===")
    for st in GROUP_STAGE_ORDER:
        t = group_tally[st]
        print(f"    {st:<22}{t['complete']:>9}{t['partial']:>9}{t['absent']:>7}"
              f"{'':>14}   {GROUP_STAGE_MADE_BY[st]}")

    print(f"\n=== {len(todo)} job(s) to submit ===")
    if counts_all.get('DONE'):
        print(f"    {counts_all['DONE']} already done with these exact settings -- skipped")
    if counts_all.get('RERUN_CHANGED'):
        what = "SKIPPED (--skip-changed)" if args.skip_changed else "WILL BE OVERWRITTEN"
        print(f"    {counts_all['RERUN_CHANGED']} existing result(s) built with different "
              f"settings -- {what}")
    for glm, sub, status, detail in rows:
        if status == 'RERUN_CHANGED':
            print(f"        sub-{sub} {glm}: {detail}")
            break
    if counts_all.get('GLM_NOT_READY'):
        print(f"    {counts_all['GLM_NOT_READY']} blocked: the epoch GLM is not complete "
              f"(run check_GLMs_ran.py)")
    if counts_all.get('MISSING_INPUT'):
        print(f"    {counts_all['MISSING_INPUT']} blocked: inputs missing")
    # These two do NOT need the RSA rerun -- the downstream wrapper skips what
    # already exists, so rerunning it is cheap and only fills the gaps.
    for st, script in (('SMOOTHED_INCOMPLETE', 'wrapper_smooth_stat_maps_subj.sh'),
                       ('STANDARD_INCOMPLETE',
                        'transform_smooth_subject_res_to_standard.sh')):
        if counts_all.get(st):
            print(f"    {counts_all[st]} run(s) have complete results but an "
                  f"incomplete {st.split('_')[0].lower()} stage -- rerun {script}, "
                  f"not the RSA")

    by_status = {}
    for glm, sub, status, detail in rows:
        by_status.setdefault(status, []).append((glm, sub, detail))
    for status in sorted(by_status):
        rws = by_status[status]
        print(f"\n{status} ({len(rws)}):")
        for glm, sub, detail in rws[:10]:
            print(f"    sub-{sub} {glm}" + (f"  --  {detail}" if detail else ""))
        if len(rws) > 10:
            print(f"    ... and {len(rws) - 10} more (see report.txt)")

    if args.no_write:
        return 0

    os.makedirs(out_dir, exist_ok=True)
    for cfg_name, cfg in epoch_configs.values():
        with open(f"{config_dir}/{cfg_name}", 'w') as f:
            json.dump(cfg, f, indent=2)

    # One smoothing config per epoch, so a per-(subject, epoch) smoothing job
    # cannot have its regression_version rewritten by a sibling job.
    smooth_configs = {}
    smooth_base_path = f"{config_dir}/{args.smooth_config}"
    if os.path.exists(smooth_base_path):
        smooth_base = json.load(open(smooth_base_path))
    else:
        smooth_base = {"fwhm": fwhm, "searchlight_mask":
                       base_config.get("searchlight_mask", "grey_matter")}
    for glm in epochs:
        c = dict(smooth_base)
        c['name_of_RSA'] = name_RSA
        c['regression_version'] = glm
        c['fwhm'] = fwhm
        name = f"{os.path.splitext(args.smooth_config)[0]}_{glm}.json"
        json.dump(c, open(f"{config_dir}/{name}", 'w'), indent=2)
        smooth_configs[glm] = name

    todo_path = f"{out_dir}/todo_rsa.txt"
    with open(todo_path, 'w') as f:
        f.write("# subject epoch_config -- RSA jobs still to run.\n")
        f.write("# feed to: bash submit_RSA_instruction_epochs.sh todo_rsa.txt\n")
        for sub, cfg_name in todo:
            f.write(f"{sub} {cfg_name}\n")

    with open(f"{out_dir}/report.txt", 'w') as f:
        f.write(f"RSA audit {stamp} -- {name_RSA} ({args.base_config})\n")
        f.write(f"{len(map_names)} maps per (subject, epoch); {n_runs} runs\n")
        f.write(f"{len(todo)} job(s) to submit\n\n")
        f.write("stage                  complete  partial   none   maps missing\n")
        for st in STAGE_ORDER:
            t = stage_tally[st]
            f.write(f"{STAGE_DIRS[st]:<22}{t['complete']:>9}{t['partial']:>9}"
                    f"{t['absent']:>7}{t['n_missing_maps']:>15}\n")
        f.write("\nsub\tepoch\tstatus\tdetail\n")
        for glm, sub, status, detail in rows:
            f.write(f"sub-{sub}\t{glm}\t{status}\t{detail}\n")

    # Exactly which map is absent where -- the thing you actually need when a
    # stage is partial and you want to know whether it is one model or all of
    # them.
    with open(f"{out_dir}/missing_maps.txt", 'w') as f:
        f.write("# subject\tepoch\tstage\tfilename -- maps not on disk\n")
        for sub, glm, st, fname in missing_maps:
            f.write(f"sub-{sub}\t{glm}\t{st}\t{fname}\n")

    # The plan: what to run, per wrapper, in pipeline order.
    fmri_dir = os.path.dirname(os.path.abspath(__file__))
    plan = build_plan(stage_rows, group_rows, epochs, args.subjects,
                      todo_path, fmri_dir, smooth_configs)
    plan_path = f"{out_dir}/resubmit_plan.sh"
    with open(plan_path, 'w') as f:
        f.write(PLAN_HEADER.format(stamp=stamp, repo=fmri_dir, audit=out_dir,
                                   fmri_dir=fmri_dir))
        if not plan:
            f.write("echo 'nothing to do -- every stage is complete.'\n")
        for i, (stage, summary, cmds) in enumerate(plan, 1):
            f.write(f"# ---- stage {i}: {stage} -- {summary} "
                    f"({GROUP_STAGE_MADE_BY.get(stage) or STAGE_MADE_BY[stage]})\n")
            for c in cmds:
                f.write(c + "\n")
            f.write("\n")
    os.chmod(plan_path, 0o755)

    print("\n=== what to submit, per wrapper ===")
    if not plan:
        print("    nothing -- every stage is complete.")
    for i, (stage, summary, cmds) in enumerate(plan, 1):
        print(f"  {i}. {stage:<10} {summary}")
        for c in cmds[:2]:
            print("       " + c.replace("\\\n    ", " "))
        if len(cmds) > 2:
            print(f"       ... and {len(cmds) - 2} more (see resubmit_plan.sh)")
    if len(plan) > 1:
        print("\n  These are ordered and dependent: run stage 1, wait for the "
              "queue,\n  then re-run this audit for the next plan.")

    with open(f"{out_dir}/settings.json", 'w') as f:
        json.dump({'date': stamp, 'base_config': args.base_config,
                   'data_dir_deriv': glmcheck.data_dir_deriv,
                   'epochs': epochs, 'subjects': args.subjects,
                   'skip_changed': args.skip_changed,
                   'compared_keys': COMPARED_KEYS,
                   'smooth_fwhm': fwhm,
                   'stages': {st: STAGE_DIRS[st] for st in STAGE_ORDER},
                   'stage_tally': stage_tally,
                   'group_stage_tally': group_tally,
                   'n_maps_expected_per_run': len(map_names),
                   'expected_map_names': map_names,
                   'expected_settings': expected_settings(base_config),
                   'counts': counts_all, 'n_todo': len(todo)}, f, indent=2)

    print(f"\nwritten to {out_dir}:")
    print(f"    todo_rsa.txt      {len(todo)} job(s)")
    print(f"    report.txt        per-stage table + why each run is not complete")
    print(f"    missing_maps.txt  {len(missing_maps)} individual map(s) not on disk")
    print(f"    resubmit_plan.sh  the commands above, in order")
    print(f"    settings.json     what this audit was run with, incl. the "
          f"{len(map_names)} expected map names")
    print(f"    ({len(epoch_configs)} per-epoch config snapshot(s) in {config_dir})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
