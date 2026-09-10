#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit first-level FEAT GLMs: what finished, what is duplicated, what to rerun.

A FEAT job can die in ways that leave a plausible-looking .feat directory
behind: killed mid-FILM (stats/ half full), submitted against a missing EV
folder, or out of disk quota. And FEAT never overwrites: re-running the same
design writes glm_..._pt01+.feat, the run after that glm_..._pt01++.feat, and so
on, while every downstream script keeps reading the plain glm_..._pt01.feat.
So the same run can exist several times over, in any mix of complete and
broken, and 'is this GLM done?' is really a question about a *set* of folders.

This walks the expected (subject x task half x GLM) grid, evaluates EVERY
'+'-generation of each run, and sorts the whole grid into:

    OK             base folder complete, no twins -- nothing to do
    DUPLICATES     base complete AND twins exist -- delete the twins
    PROMOTE_TWIN   base missing/broken but a twin is complete -- rename the
                   good twin onto the base name (no rerun needed)
    <failure>      no complete copy anywhere -- delete what is there, rerun
    NO_EV_FOLDER / NO_DRAFT_FSF
                   cannot even be submitted; rerun create_EVs_instruction_period.py

It then writes three files into --out-dir (default:
derivatives/group/glm_audit_<version>_<date>/), which are the three things you
need to clean this up:

    report.txt              every run that is not plain OK, with the reason
    cleanup_feat_dirs.sh    the deletions and renames, DRY-RUN BY DEFAULT.
                            Run it to see what it would do; run it with --apply
                            to actually do it. It re-checks every path against
                            the expected pattern before touching it and refuses
                            anything else, so it cannot wander off the grid.
    todo_submit.txt         'sub half glm' for every run that genuinely still
                            needs FEAT. Feed it straight to
                            subject_GLM_instruction_epochs.sh, which then
                            submits only these.

Order matters: clean up FIRST, then submit. FEAT adds another '+' generation
for every rerun that starts while the old directory is still in place.

WHAT IS CHECKED, per .feat folder
    1. the folder exists
    2. stats/ exists
    3. design.mat is there and says how many regressors to expect (/NumWaves,
       which includes the motion/PNM confound columns FEAT appends)
    4. pe1..pe{NumWaves}.nii.gz all exist and are non-empty
    5. FILM's end-of-run outputs are there (dof, smoothness, sigmasquareds,
       threshac1) -- these are written last, so they are the completion marker
    6. every PE the RSA will actually read exists. The RSA maps EV -> PE through
       EVs_{glm}_pt0{th}/task-to-EV.txt, so a GLM can be complete in itself and
       still be unusable if that file disagrees with the design.

    --check-data additionally loads every PE the RSA reads and flags all-zero or
    non-finite maps. That catches a GLM that ran to completion on a degenerate
    design. It is slow (one nifti read per PE), so it is off by default.

USAGE
    # audit the 11 instruction-epoch GLMs, all subjects, both halves
    python3 check_GLMs_ran.py

    # then, from the printed out-dir:
    sh cleanup_feat_dirs.sh              # dry run: prints what it would do
    sh cleanup_feat_dirs.sh --apply      # actually delete / rename
    bash subject_GLM_instruction_epochs.sh todo_submit.txt

    # other GLMs, or a subset, or with the PE data verified too
    python3 check_GLMs_ran.py --glms all-paths-fixed_stickrews_split-buttons
    python3 check_GLMs_ran.py --subjects 01 02 35 --check-data

@author: Svenja Kuechenhoff
"""

import argparse
import datetime
import json
import os
import re
import sys

# --- where things live ----------------------------------------------------
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    data_dir_deriv = f"{source_dir}/data/derivatives"
    config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
    logs_dir = f"{source_dir}/analysis/logs_mid_sept"
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    data_dir_deriv = f"{source_dir}/data/derivatives"
    config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"
    logs_dir = f"{source_dir}/analysis/logs_mid_sept"
# Audits are logs, not results: they go next to the analysis code, never inside
# the git repo and not under derivatives/group.

DEFAULT_SUBJECTS = [f"{i:02}" for i in range(1, 36) if i not in (21, 29)]
# written by FILM at the very end of a first-level run
FILM_DONE_MARKERS = ['dof', 'smoothness', 'sigmasquareds.nii.gz', 'threshac1.nii.gz']
# EV names that are nuisance regressors, skipped by the RSA loader
NUISANCE_EVS = {'press_EV', 'up', 'down', 'left', 'right'}
# a run in one of these states has no complete copy, but also cannot be
# submitted: its inputs are missing, so it does not belong on the todo list
BLOCKED = {'NO_EV_FOLDER', 'NO_DRAFT_FSF'}
# nothing to do about these two
FINE = {'OK', 'DUPLICATES', 'PROMOTE_TWIN'}


def glm_names_from_config(config_file):
    """The same names create_EVs_instruction_period.py builds its folders from,
    so this checker and the EV script cannot drift apart."""
    with open(f"{config_path}/{config_file}", "r") as f:
        config = json.load(f)
    version = config['name']
    return [f"{version}_{seg['name'].replace('_', '-')}" for seg in config['segments']]


def num_waves(design_mat):
    """Regressor count FEAT actually fitted, from the design it wrote."""
    with open(design_mat, 'r') as f:
        for line in f:
            if line.startswith('/NumWaves'):
                return int(line.split()[1])
    return None


def rsa_pe_indices(EV_folder):
    """(pe number, EV name) for every EV the RSA reads, mirroring
    mc.analyse.my_RSA.load_data_EVs_instr_TRwise: line i of task-to-EV.txt maps
    to pe{i+1}, nuisance regressors skipped."""
    EV_txt = os.path.join(EV_folder, 'task-to-EV.txt')
    if not os.path.exists(EV_txt):
        return None
    out = []
    with open(EV_txt, 'r') as f:
        for line in f:
            index, name_ev = line.strip().split(' ', 1)
            name = name_ev.replace('ev_', '')
            if name in NUISANCE_EVS:
                continue
            out.append((int(index) + 1, name))
    return out


_listing_cache = {}


def list_func_dir(func):
    """os.listdir once per subject/half folder -- this gets asked the same
    question 11 times, once per instruction GLM."""
    if func not in _listing_cache:
        _listing_cache[func] = sorted(os.listdir(func)) if os.path.isdir(func) else []
    return _listing_cache[func]


def feat_variants(func, glm, th):
    """Every folder FEAT may have written for this run, oldest first, as
    (generation, path). Generation is the number of '+' FEAT appended: 0 is the
    original directory and the only name the RSA ever reads, 1 is the second
    run, 2 the third. Anything other than '+' after the run name is a different
    GLM and is not matched."""
    pattern = re.compile(rf"^glm_{re.escape(glm)}_pt0{th}(\+*)\.feat$")
    found = []
    for name in list_func_dir(func):
        match = pattern.match(name)
        if match and os.path.isdir(f"{func}/{name}"):
            found.append((len(match.group(1)), f"{func}/{name}"))
    return sorted(found)


def check_feat_dir(feat, EV_folder, check_data=False):
    """Did THIS one .feat directory finish? (status, detail); 'OK' means the
    PEs are all there and readable by the RSA."""
    stats = f"{feat}/stats"
    if not os.path.isdir(stats):
        return 'NO_STATS', 'FEAT started but never got to FILM'

    design = f"{feat}/design.mat"
    if not os.path.exists(design):
        return 'NO_DESIGN', 'design.mat missing'
    n_waves = num_waves(design)
    if n_waves is None:
        return 'BAD_DESIGN', 'design.mat has no /NumWaves'

    missing_pe = [i for i in range(1, n_waves + 1)
                  if not os.path.exists(f"{stats}/pe{i}.nii.gz")
                  or os.path.getsize(f"{stats}/pe{i}.nii.gz") == 0]
    if missing_pe:
        return 'INCOMPLETE_PES', (f"{len(missing_pe)}/{n_waves} PEs missing or empty "
                                  f"(first: pe{missing_pe[0]})")

    # prune_feat_dirs.py strips everything but the PEs from old GLMs, which
    # removes two of the FILM markers. The PEs are still the real ones.
    pruned = os.path.exists(f"{feat}/PRUNED.json")
    missing_marker = [m for m in FILM_DONE_MARKERS if not os.path.exists(f"{stats}/{m}")]
    if missing_marker and not pruned:
        return 'FILM_UNFINISHED', f"all PEs present but missing {', '.join(missing_marker)}"

    rsa_pes = rsa_pe_indices(EV_folder)
    if rsa_pes is None:
        return 'NO_TASK_TO_EV', 'task-to-EV.txt missing, RSA cannot map EVs to PEs'
    beyond = [f"pe{i} ({n})" for i, n in rsa_pes if i > n_waves]
    if beyond:
        return 'EV_PE_MISMATCH', (f"task-to-EV.txt points past the design "
                                  f"({n_waves} waves): {', '.join(beyond[:3])}")

    if check_data:
        import numpy as np
        import nibabel as nib
        for i, name in rsa_pes:
            d = np.asarray(nib.load(f"{stats}/pe{i}.nii.gz").dataobj)
            if not np.isfinite(d).any() or not np.any(d):
                return 'EMPTY_PE', f"pe{i} ({name}) is all-zero or non-finite"

    return 'OK', ''


def check_one(sub, th, glm, check_data=False):
    """Verdict over ALL '+'-generations of one run.

    Returns (status, detail, actions) where actions is
        {'delete_duplicate': [...],   complete copies that are simply redundant
         'delete_incomplete': [...],  copies that never finished
         'promote': (twin, base)}     the good twin to move onto the base name
    """
    func = f"{data_dir_deriv}/sub-{sub}/func"
    base = f"{func}/glm_{glm}_pt0{th}.feat"
    EV_folder = f"{func}/EVs_{glm}_pt0{th}"
    draft_fsf = f"{func}/sub-{sub}_draft_GLM_0{th}_{glm}.fsf"
    nothing = {'delete_duplicate': [], 'delete_incomplete': [], 'promote': None}

    variants = feat_variants(func, glm, th)

    # Inputs first, but only report them as the verdict when there is no
    # finished output already sitting there: a GLM that ran back when the EVs
    # were present is still perfectly usable.
    if not os.path.isdir(EV_folder):
        return 'NO_EV_FOLDER', f"missing {EV_folder}", nothing
    if not variants and not os.path.exists(draft_fsf):
        return 'NO_DRAFT_FSF', f"missing {draft_fsf}", nothing
    if not variants:
        return 'NO_FEAT_DIR', base, nothing

    checked = [(gen, path) + check_feat_dir(path, EV_folder, check_data)
               for gen, path in variants]
    complete = [(gen, path) for gen, path, status, _ in checked if status == 'OK']

    def plus(gen):
        """How to call a generation in the output."""
        return 'base' if gen == 0 else "'" + '+' * gen + "'"

    if not complete:
        # Nothing usable. Report the base's problem if the base is there, else
        # the newest attempt's, and offer every copy up for deletion so the
        # rerun starts from a clean name.
        gen, path, status, detail = checked[0] if checked[0][0] == 0 else checked[-1]
        if len(checked) > 1:
            detail = (f"{detail} -- all {len(checked)} copies "
                      f"({', '.join(plus(g) for g, _, _, _ in checked)}) failed")
        if not os.path.exists(draft_fsf):
            return 'NO_DRAFT_FSF', (f"{detail}; and the draft fsf is gone, so this "
                                    f"cannot be resubmitted either"), nothing
        actions = dict(nothing, delete_incomplete=[p for _, p, _, _ in checked])
        return status, detail, actions

    keep_gen, keep_path = complete[0]
    dupes = [p for g, p in complete if g != keep_gen]
    broken = [p for g, p, s, _ in checked if s != 'OK']

    if keep_gen == 0 and not dupes and not broken:
        return 'OK', '', nothing

    if keep_gen == 0:
        detail = (f"base is complete; {len(dupes) + len(broken)} extra copy(s) "
                  f"({', '.join(plus(g) for g, _, s, _ in checked if g != 0)}) "
                  f"can go")
        return 'DUPLICATES', detail, {'delete_duplicate': dupes,
                                      'delete_incomplete': broken,
                                      'promote': None}

    # The base is the one the RSA reads and it is not the good one. The broken
    # base is not listed for deletion: the rename replaces it, and deleting it
    # a second time afterwards would take the promoted run with it.
    base_status = next((s for g, _, s, _ in checked if g == 0), 'missing')
    detail = (f"base is {base_status}, but {plus(keep_gen)} is complete -- "
              f"rename it onto the base name, no rerun needed")
    return 'PROMOTE_TWIN', detail, {'delete_duplicate': dupes,
                                    'delete_incomplete': [p for p in broken if p != base],
                                    'promote': (keep_path, base)}


# --- the cleanup script ----------------------------------------------------
# Guards live in the generated script, not here, so that what gets run is the
# same text you can read before running it.
CLEANUP_HEADER = r'''#!/bin/sh
# Delete the redundant / unfinished FEAT directories found by check_GLMs_ran.py,
# and move finished '+' twins onto the base name the RSA reads.
#
#   sh cleanup_feat_dirs.sh            DRY RUN -- only prints what it would do
#   sh cleanup_feat_dirs.sh --apply    actually delete and rename
#
# Nothing is deleted unless --apply is given. Every path is re-checked here,
# at the moment of deletion, against the shape of a first-level FEAT directory
# (.../derivatives/sub-XX/func/glm_*.feat, containing a stats/ or design.fsf).
# Anything that does not match is REFUSED and left alone, so a stale or
# hand-edited path in this file cannot delete something else.

APPLY=0
[ "$1" = "--apply" ] && APPLY=1
if [ "$APPLY" = "0" ]; then
  echo "=== DRY RUN -- nothing will be changed. Re-run with --apply to do it. ==="
fi
echo

n_del=0; n_promoted=0; n_refused=0; n_gone=0

# A path is only touched if it looks exactly like a first-level FEAT directory.
check_path () {
  case "$1" in
    */derivatives/sub-*/func/glm_*.feat) ;;
    *) echo "  REFUSED (not a first-level FEAT path): $1"; n_refused=$((n_refused+1)); return 1;;
  esac
  case "$1" in
    *..*) echo "  REFUSED (path contains ..): $1"; n_refused=$((n_refused+1)); return 1;;
  esac
  [ -d "$1" ] || return 2                       # already gone: not an error
  if [ ! -d "$1/stats" ] && [ ! -f "$1/design.fsf" ]; then
    echo "  REFUSED (does not look like a FEAT dir): $1"; n_refused=$((n_refused+1)); return 1
  fi
  return 0
}

drop () {
  check_path "$1"; rc=$?
  [ $rc -eq 1 ] && return 1
  if [ $rc -eq 2 ]; then
    echo "  already gone: $1"; n_gone=$((n_gone+1)); return 0
  fi
  if [ "$APPLY" = "1" ]; then
    rm -rf "$1" && echo "  deleted:      $1" && n_del=$((n_del+1))
  else
    echo "  would delete: $1"; n_del=$((n_del+1))
  fi
}

# Rename a finished twin onto the base name. The broken base is moved aside
# first and only removed once the twin is safely in place, so an interrupted
# run can never leave the good data deleted.
promote () {
  twin="$1"; base="$2"
  check_path "$twin"; rc=$?
  [ $rc -eq 1 ] && return 1
  if [ $rc -eq 2 ]; then
    echo "  MISSING twin, skipped: $twin"; n_refused=$((n_refused+1)); return 1
  fi
  if [ ! -d "$twin/stats" ]; then
    echo "  REFUSED (twin has no stats/): $twin"; n_refused=$((n_refused+1)); return 1
  fi
  if [ "$APPLY" = "0" ]; then
    echo "  would promote: $twin"
    echo "              -> $base"
    n_promoted=$((n_promoted+1)); return 0
  fi
  if [ -d "$base" ]; then
    check_path "$base" || return 1
    mv "$base" "$base.SUPERSEDED" || { echo "  FAILED to move old base aside: $base"; return 1; }
  fi
  if mv "$twin" "$base"; then
    rm -rf "$base.SUPERSEDED"
    echo "  promoted:     $twin -> $base"
    n_promoted=$((n_promoted+1))
  else
    echo "  FAILED to promote $twin; restoring old base"
    [ -d "$base.SUPERSEDED" ] && mv "$base.SUPERSEDED" "$base"
    return 1
  fi
}

'''

CLEANUP_FOOTER = r'''
echo
if [ "$APPLY" = "1" ]; then
  echo "done: $n_del deleted, $n_promoted promoted, $n_gone already gone, $n_refused refused."
else
  echo "dry run: $n_del would be deleted, $n_promoted would be promoted, $n_gone already gone, $n_refused refused."
  echo "Re-run with --apply to carry this out."
fi
'''


def write_cleanup(path, to_promote, to_delete_dupe, to_delete_incomplete):
    n_actions = len(to_promote) + len(to_delete_dupe) + len(to_delete_incomplete)
    with open(path, 'w') as f:
        # Machine-readable count on line 2: subject_GLM_instruction_epochs.sh
        # reads it to decide whether there is cleanup outstanding. Counting the
        # drop/promote CALLS by grep would also match their definitions.
        f.write(CLEANUP_HEADER.replace('#!/bin/sh\n',
                                       f'#!/bin/sh\n# N_ACTIONS: {n_actions}\n', 1))
        if to_promote:
            f.write(f'echo "--- {len(to_promote)} finished twin(s) to move onto the base name ---"\n')
            for src, dst in to_promote:
                f.write(f"promote '{src}' '{dst}'\n")
            f.write('echo\n\n')
        if to_delete_dupe:
            f.write(f'echo "--- {len(to_delete_dupe)} complete but redundant copy(s) ---"\n')
            for p in to_delete_dupe:
                f.write(f"drop '{p}'\n")
            f.write('echo\n\n')
        if to_delete_incomplete:
            f.write(f'echo "--- {len(to_delete_incomplete)} copy(s) that never finished ---"\n')
            for p in to_delete_incomplete:
                f.write(f"drop '{p}'\n")
            f.write('echo\n')
        f.write(CLEANUP_FOOTER)
    os.chmod(path, 0o755)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--glms', nargs='+', default=None,
                    help='GLM names to check. Default: the epochs in --ev-config.')
    ap.add_argument('--ev-config', default='EV_config_instruction.json',
                    help='EV config the default GLM names are derived from.')
    ap.add_argument('--subjects', nargs='+', default=DEFAULT_SUBJECTS)
    ap.add_argument('--task-halves', nargs='+', type=int, default=[1, 2])
    ap.add_argument('--check-data', action='store_true',
                    help='also load every PE the RSA reads and flag all-zero / non-finite maps (slow)')
    ap.add_argument('--data-dir', default=None,
                    help='derivatives directory to check. Default: the laptop path if it '
                         'exists, else the cluster one.')
    ap.add_argument('--out-dir', default=None,
                    help='where report.txt, cleanup_feat_dirs.sh and todo_submit.txt go. '
                         'Default: analysis/logs_mid_sept/glm_audit_<version>_<date>/')
    ap.add_argument('--no-write', action='store_true',
                    help='only print the overview, write no files')
    args = ap.parse_args()

    if args.data_dir:
        global data_dir_deriv
        data_dir_deriv = args.data_dir.rstrip('/')

    glms = args.glms if args.glms else glm_names_from_config(args.ev_config)
    stamp = datetime.date.today().isoformat()
    tag = os.path.splitext(args.ev_config)[0].replace('EV_config_', '') if not args.glms else 'custom'
    out_dir = args.out_dir or f"{logs_dir}/glm_audit_{tag}_{stamp}"

    print(f"checking {len(glms)} GLM(s) x {len(args.subjects)} subjects x "
          f"{len(args.task_halves)} halves = "
          f"{len(glms) * len(args.subjects) * len(args.task_halves)} runs")
    print(f"under {data_dir_deriv}\n")

    problems, todo, blocked = [], [], []
    to_delete_dupe, to_delete_incomplete, to_promote = [], [], []
    counts_all = {}
    for glm in glms:
        counts = {}
        for sub in args.subjects:
            for th in args.task_halves:
                status, detail, actions = check_one(sub, th, glm, args.check_data)
                counts[status] = counts.get(status, 0) + 1
                counts_all[status] = counts_all.get(status, 0) + 1
                if status != 'OK':
                    problems.append((glm, sub, th, status, detail))
                if status in BLOCKED:
                    blocked.append((glm, sub, th, status, detail))
                elif status not in FINE:
                    # no complete copy anywhere, and the inputs are there
                    todo.append((sub, th, glm))
                to_delete_dupe += actions['delete_duplicate']
                to_delete_incomplete += actions['delete_incomplete']
                if actions['promote']:
                    to_promote.append(actions['promote'])
        n_ok = counts.get('OK', 0)
        total = sum(counts.values())
        other = ', '.join(f"{k}:{v}" for k, v in sorted(counts.items()) if k != 'OK')
        print(f"  {glm:<40} {n_ok:>3}/{total} OK" + (f"   [{other}]" if other else ""))

    n_runs = sum(counts_all.values())
    n_usable = sum(v for k, v in counts_all.items() if k in FINE)
    print(f"\n{n_usable}/{n_runs} runs have a complete GLM "
          f"({counts_all.get('OK', 0)} of them already under the right name).")

    if problems:
        print(f"\n=== {len(problems)} run(s) need attention ===")
        by_status = {}
        for glm, sub, th, status, detail in problems:
            by_status.setdefault(status, []).append((glm, sub, th, detail))
        for status, rows in sorted(by_status.items()):
            print(f"\n{status} ({len(rows)}):")
            for glm, sub, th, detail in rows[:12]:
                print(f"    sub-{sub} pt{th} {glm}" + (f"  --  {detail}" if detail else ""))
            if len(rows) > 12:
                print(f"    ... and {len(rows) - 12} more (see report.txt)")

    n_cleanup = len(to_delete_dupe) + len(to_delete_incomplete)
    print(f"\n=== 1. cleanup: {len(to_promote)} rename(s), {n_cleanup} folder(s) to delete ===")
    print(f"    {len(to_promote):>4} finished '+' twins to move onto the base name")
    print(f"    {len(to_delete_dupe):>4} complete but redundant copies")
    print(f"    {len(to_delete_incomplete):>4} copies that never finished")

    print(f"\n=== 2. still to submit: {len(todo)} run(s) ===")
    per_glm = {}
    for sub, th, glm in todo:
        per_glm.setdefault(glm, set()).add(sub)
    for glm in glms:
        if glm in per_glm:
            print(f"    {glm:<40} {len(per_glm[glm])} subject(s): "
                  f"{' '.join(sorted(per_glm[glm]))}")
    if not todo:
        print("    nothing -- every run exists somewhere")

    if blocked:
        print(f"\n=== 3. cannot be submitted: {len(blocked)} run(s) ===")
        print("    inputs are missing -- rerun create_EVs_instruction_period.py for these")
        by_sub = {}
        for glm, sub, th, status, detail in blocked:
            by_sub.setdefault((sub, status), []).append(f"{glm} pt{th}")
        for (sub, status), items in sorted(by_sub.items()):
            print(f"    sub-{sub} {status}: {len(items)} run(s), e.g. {items[0]}")

    if args.no_write:
        return 0 if not problems else 1

    os.makedirs(out_dir, exist_ok=True)
    report_path = f"{out_dir}/report.txt"
    cleanup_path = f"{out_dir}/cleanup_feat_dirs.sh"
    todo_path = f"{out_dir}/todo_submit.txt"

    with open(report_path, 'w') as f:
        f.write(f"FEAT audit {stamp}\n{n_usable}/{n_runs} runs have a complete GLM; "
                f"{len(problems)} need attention\n\n")
        f.write("sub\tpt\tglm\tstatus\tdetail\n")
        for glm, sub, th, status, detail in problems:
            f.write(f"sub-{sub}\t{th}\t{glm}\t{status}\t{detail}\n")

    write_cleanup(cleanup_path, to_promote, to_delete_dupe, to_delete_incomplete)

    with open(todo_path, 'w') as f:
        f.write("# subject task_half glm -- runs with no complete copy anywhere.\n")
        f.write("# feed to: bash subject_GLM_instruction_epochs.sh todo_submit.txt\n")
        for sub, th, glm in sorted(todo):
            f.write(f"{sub} {th} {glm}\n")

    with open(f"{out_dir}/settings.json", 'w') as f:
        json.dump({'date': stamp,
                   'data_dir_deriv': data_dir_deriv,
                   'glms': glms,
                   'ev_config': args.ev_config,
                   'subjects': args.subjects,
                   'task_halves': args.task_halves,
                   'check_data': args.check_data,
                   'film_done_markers': FILM_DONE_MARKERS,
                   'counts': counts_all,
                   'n_to_promote': len(to_promote),
                   'n_delete_duplicate': len(to_delete_dupe),
                   'n_delete_incomplete': len(to_delete_incomplete),
                   'n_todo': len(todo),
                   'n_blocked': len(blocked)}, f, indent=2)

    print(f"\nwritten to {out_dir}:")
    print(f"    report.txt             what is wrong with each run")
    print(f"    cleanup_feat_dirs.sh   sh cleanup_feat_dirs.sh          (dry run, changes nothing)")
    print(f"                           sh cleanup_feat_dirs.sh --apply  (do it)")
    print(f"    todo_submit.txt        bash subject_GLM_instruction_epochs.sh {todo_path}")
    print(f"    settings.json          what this audit was run with")
    print("\nClean up BEFORE submitting: FEAT adds another '+' generation for every "
          "rerun\nthat starts while the old directory is still there.")
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
