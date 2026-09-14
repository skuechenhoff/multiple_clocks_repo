#!/bin/bash
# Submit the instruction-phase searchlight RSA, one job per (subject, epoch),
# but ONLY for jobs that can actually run and are not already done.
#
#   bash submit_RSA_instruction_epochs.sh                 # audit, then submit what is missing
#   bash submit_RSA_instruction_epochs.sh todo_rsa.txt    # submit exactly this list
#   DRYRUN=1 bash submit_RSA_instruction_epochs.sh        # print what would be submitted
#   SKIP_CHANGED=1 bash submit_RSA_instruction_epochs.sh  # leave results whose settings differ
#
# WHICH CONFIG (env RSA_CONFIG, default rsa_instruction_samedirection.json)
#   RSA_CONFIG=rsa_instruction_cumulative_rew.json bash submit_RSA_instruction_epochs.sh
# The default is the same-direction masked instruction config: the *_rew_instr
# models fitted only on RDM cells that do NOT cross the forward/backward cue.
# Every *_instr model forces dissimilarity 0 on the same-task forward/backward
# pairs, but the instruction screen differs there ('please backwards' text), and
# no nuisance regressor can absorb that -- see the 2026-09-14 CHANGELOG entries.
# The two configs write to different RSA folders (name_of_RSA) and get separate
# audit dirs, so they never collide and either can be rerun independently.
#
# The GLMs are not named by a TR index ('01-TR4') but by the epoch they measure
# ('instr_see-A-first'), so each job gets a config snapshot with
# regression_version set to the epoch GLM and TR set to null --
# fMRI_run_RSA_instruction.py then treats regression_version as the full GLM
# name (see load_data_EVs_instr_TRwise). check_RSA_ran.py writes those snapshots.
#
# WHAT IS DIFFERENT FROM A PLAIN GRID LOOP
# This used to submit 33 subjects x 11 epochs unconditionally. Most of those
# jobs cannot succeed or need not run: the RSA reads glm_instr_<epoch>_pt0{1,2}
# .feat directly, so a subject whose epoch GLM never finished produces a job
# that dies on a missing PE, and a subject that is already done just recomputes
# the same maps for an hour. check_RSA_ran.py therefore decides the list first:
#   - the epoch GLM must be complete in the BASE directory the RSA reads
#     (same check as check_GLMs_ran.py; a finished '+' twin does not count,
#      the FEAT cleanup has to run first)
#   - the modelled EVs, reference image and searchlight mask must exist
#   - a finished result with THESE settings means the job is skipped; a result
#     built with different settings is rerun (SKIP_CHANGED=1 to leave it)

scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
scriptname="fMRI_run_RSA_instruction.py"
base_config="${RSA_CONFIG:-rsa_instruction_samedirection.json}"
configTag=$(basename "${base_config}" .json)

# Wall-time estimate in minutes for one RSA job. 240 was sized for the full
# cumulative-reward config: 28 whole-brain searchlight OLS models (21 single +
# 7 combo) over ~126k searchlights. The same-direction config fits 7 single
# models on 40 RDM cells instead, and reuses the searchlight data-RDM cache
# (keyed on regression_version + searchlight_mask, not on name_of_RSA), so it
# is far cheaper -- start at 120 and escalate only if jobs are killed.
# Under-requesting is cheap here: fMRI_run_RSA_instruction.py resumes, so a job
# that runs out of time picks up where it stopped instead of starting over.
# Override with FSLSUB_T=<minutes>; use 240 for the cumulative-reward config.
jobTime="${FSLSUB_T:-120}"

# Audits are logs, not results: keep them next to the analysis code.
logDir="${analysisDir}/logs_mid_sept"

# this script, check_RSA_ran.py and the job wrapper all live in the repo
scriptDir=$(dirname "$0")
jobWrapper="${scriptDir}/update_fMRI/wrapper_python_fMRI_RSA_clean_config.sh"

module load fsl

todoFile="$1"

if [ -z "$todoFile" ]; then
    auditDir="${logDir}/rsa_audit_${configTag}_$(date +%F)"
    echo "No list given -- auditing first, so that finished RSAs are not rerun."
    pythonBin=$(command -v python3 || command -v python)
    if [ -z "$pythonBin" ]; then
        echo "ERROR: no python found. Pass a todo list instead."
        exit 1
    fi
    skipArg=""
    [ -n "${SKIP_CHANGED:-}" ] && skipArg="--skip-changed"
    $pythonBin "${scriptDir}/check_RSA_ran.py" \
        --base-config "${base_config}" \
        --data-dir "${scratchDir}/derivatives" \
        --config-dir "${analysisDir}/multiple_clocks_repo/condition_files" \
        --out-dir "${auditDir}" $skipArg || exit 1
    todoFile="${auditDir}/todo_rsa.txt"
    if [ ! -f "$todoFile" ]; then
        echo "ERROR: the audit wrote no todo list ($todoFile)"
        exit 1
    fi
fi

runList=$(mktemp "${TMPDIR:-/tmp}/rsa_runlist.XXXXXX")
trap 'rm -f "$runList"' EXIT
grep -v '^#' "$todoFile" | grep -v '^[[:space:]]*$' > "$runList"

n_runs=$(wc -l < "$runList" | tr -d "[:space:]")
if [ "$n_runs" -eq 0 ]; then
    echo "Nothing to submit -- every RSA is either done or blocked (see the report)."
    exit 0
fi
echo "Submitting $n_runs job(s) from $todoFile  (config: ${base_config}, -T ${jobTime})"

n_submitted=0
while read -r subjectTag epoch_config; do
    [ -z "$subjectTag" ] && continue
    if [ -n "${DRYRUN:-}" ]; then
        echo "DRYRUN would submit (-T ${jobTime}): sub-${subjectTag}  ${epoch_config}"
    else
        echo "submitting sub-${subjectTag}  ${epoch_config}"
        fsl_sub -T "${jobTime}" bash "${jobWrapper}" \
            "${subjectTag}" "${epoch_config}" "${scriptname}"
    fi
    n_submitted=$((n_submitted+1))
done < "$runList"

echo
echo "Done: $n_submitted job(s) submitted."
