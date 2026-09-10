#!/bin/sh
# Run subject-level GLMs for the instruction-phase epochs.
#
#   bash subject_GLM_instruction_epochs.sh                  # audit first, submit only what is missing
#   bash subject_GLM_instruction_epochs.sh todo_submit.txt  # submit exactly this list
#   DRYRUN=1 bash subject_GLM_instruction_epochs.sh         # print what would be submitted
#   ALL=1 bash subject_GLM_instruction_epochs.sh            # ignore the audit, submit the whole grid
#
# requires EV directories with EVs and a subject fsf file before (made by
# scripts/create_EVs_instruction_period.py), the filtered_func dataset, and the
# nuisance regs.
#
# This is subject_GLM_RDM_conds.sh with one extra loop: instead of a single
# $version there is one GLM per instruction epoch, named after what it measures.
# create_EVs_instruction_period.py prints exactly this list when it runs.
#
# WHAT IS DIFFERENT FROM A PLAIN GRID LOOP
# FEAT never overwrites: submitting a run whose glm_..._pt01.feat already exists
# writes glm_..._pt01+.feat instead, and the RSA keeps reading the old one. So
# by default this script does not loop over the full grid at all. It first runs
# check_GLMs_ran.py, which works out which (subject, half, GLM) combinations
# have no complete GLM anywhere, and submits only those. Completed runs are
# never resubmitted, so no '+' twins get created.
#
# If that audit reports directories to clean up, run its cleanup_feat_dirs.sh
# BEFORE submitting -- a leftover broken glm_..._pt01.feat means this run lands
# in a '+' twin again.

version="instr"
glm_names="see-A-first see-B-first see-C-first see-D-first see-A-second see-B-second see-C-second see-D-second collapsed-first-instruction collapsed-second-instruction empty-screen"

# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
module load fsl

# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
  analysisDir="/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo/mc/fmri_analysis"
  fslDir="/Users/xpsy1114/fsl"
fi

# check_GLMs_ran.py sits next to this script in the repo, wherever that is
scriptDir=$(dirname "$0")

echo this is version $version
echo Scratch directory is $scratchDir

# ---------------------------------------------------------------------------
# Work out what to submit
# ---------------------------------------------------------------------------
todoFile="$1"

if [ -z "$todoFile" ] && [ -z "$ALL" ]; then
    # Audits are logs, not results. On the cluster they sit next to the
    # analysis code; on a laptop $analysisDir IS the repo, so they go under the
    # data tree instead -- nothing generated ever belongs in the repo.
    logDir="${analysisDir}/logs_mid_sept"
    case "$analysisDir" in
      *multiple_clocks_repo*) logDir="$scratchDir/derivatives/group/logs_mid_sept";;
    esac
    auditDir="${logDir}/glm_audit_${version}_$(date +%F)"
    echo "No list given -- auditing first, so that finished GLMs are not run again."
    echo "Audit goes to $auditDir"
    pythonBin=$(command -v python3 || command -v python)
    if [ -z "$pythonBin" ]; then
        echo "ERROR: no python found, cannot audit. Pass a todo list, or use ALL=1."
        exit 1
    fi
    $pythonBin "$scriptDir/check_GLMs_ran.py" \
        --data-dir "$scratchDir/derivatives" --out-dir "$auditDir"
    todoFile="$auditDir/todo_submit.txt"
    if [ ! -f "$todoFile" ]; then
        echo "ERROR: the audit wrote no todo list ($todoFile)"
        exit 1
    fi
    # Refuse to submit while there are leftover directories: a broken
    # glm_..._pt01.feat still in place means this submission lands in a '+'
    # twin, which is the mess we are trying to get out of.
    cleanupScript="$auditDir/cleanup_feat_dirs.sh"
    # The audit stamps the number of outstanding actions on line 2 of the
    # cleanup script. Counting drop/promote lines by grep would also match the
    # shell functions of those names that the script defines.
    n_cleanup=$(sed -n 's/^# N_ACTIONS: //p' "$cleanupScript" 2>/dev/null)
    [ -z "$n_cleanup" ] && n_cleanup=0
    if [ "$n_cleanup" -gt 0 ]; then
        echo
        echo "STOPPING: $n_cleanup leftover directory(s) must be cleaned up first."
        echo "    sh $cleanupScript           # dry run, changes nothing"
        echo "    sh $cleanupScript --apply   # do it"
        echo "Then rerun this script. To submit anyway, pass the list explicitly:"
        echo "    bash $0 $todoFile"
        exit 1
    fi
fi

# Either a list of 'subject half glm' lines, or the full grid under ALL=1.
# It goes to a temp file rather than a pipe so that the loop below runs in this
# shell and can keep count.
runList=$(mktemp "${TMPDIR:-/tmp}/glm_runlist.XXXXXX")
trap 'rm -f "$runList"' EXIT

if [ -n "$todoFile" ]; then
    echo "Submitting the runs listed in $todoFile"
    grep -v '^#' "$todoFile" | grep -v '^[[:space:]]*$' | sort > "$runList"
else
    echo "ALL=1: submitting the full grid, including GLMs that already finished."
    for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35; do
        for task_half in 1 2; do
            for glm_name in $glm_names; do
                echo "$subjectTag $task_half ${version}_${glm_name}" >> "$runList"
            done
        done
    done
fi

n_runs=$(wc -l < "$runList" | tr -d "[:space:]")
if [ "$n_runs" -eq 0 ]; then
    echo "Nothing to submit -- every GLM already has a complete run."
    exit 0
fi
echo "$n_runs run(s) to submit."
echo Now entering the loop ....

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
lastSubHalf=""
n_submitted=0
n_skipped=0

while read -r subjectTag task_half version_full; do
    [ -z "$subjectTag" ] && continue

    derivDir=$scratchDir/derivatives/sub-$subjectTag
    funcDir=$derivDir/func

    # The volume and voxel counts are a property of the (subject, half) data,
    # not of the GLM, so they only get recomputed when those change. The list
    # is sorted, so this is one fslval pass per subject and half.
    if [ "$lastSubHalf" != "$subjectTag $task_half" ]; then
        echo "=== sub-${subjectTag} pt${task_half} ==="
        numVols=$(fslval $funcDir/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz dim4)
        dim1=$(fslval $funcDir/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz dim1)
        dim2=$(fslval $funcDir/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz dim2)
        dim3=$(fslval $funcDir/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz dim3)
        dim4=$(fslval $funcDir/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz dim4)
        numVoxels=$((dim1*dim2*dim3*dim4))
        echo "Found $numVols volumes, $numVoxels voxels"
        lastSubHalf="$subjectTag $task_half"
    fi

    nuisanceFile="/motion/nuisance_0${task_half}/combined.txt"
    glmDir="$funcDir/glm_${version_full}_pt0${task_half}.feat"

    # FEAT would write a '+' twin rather than overwrite this, and the RSA would
    # go on reading the old directory. The audit is supposed to have cleaned
    # these away; if one is still here, say so instead of making another copy.
    if [ -d "$glmDir" ]; then
        echo "SKIP sub-${subjectTag} pt${task_half} ${version_full}: $glmDir still exists."
        echo "     Delete it first (cleanup_feat_dirs.sh), otherwise FEAT writes a '+' twin."
        n_skipped=$((n_skipped+1))
        continue
    fi

    EV_dir="$funcDir/EVs_${version_full}_pt0${task_half}"
    if [ ! -d "$EV_dir" ]; then
        echo "SKIP sub-${subjectTag} pt${task_half} ${version_full}: no EV folder $EV_dir"
        n_skipped=$((n_skipped+1))
        continue
    fi

    draftFsf="${funcDir}/sub-${subjectTag}_draft_GLM_0${task_half}_${version_full}.fsf"
    if [ ! -f "$draftFsf" ]; then
        echo "SKIP sub-${subjectTag} pt${task_half} ${version_full}: no draft fsf $draftFsf"
        n_skipped=$((n_skipped+1))
        continue
    fi

    designFsf="$funcDir/sub-${subjectTag}_design_glm_${version_full}_pt${task_half}.fsf"

    cat ${draftFsf} | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01:${derivDir}:g" | sed "s:/func/preproc_clean_01:/func/preproc_clean_0${task_half}:g" | sed "s:/func/nuisance_01/combined.txt:${nuisanceFile}:g" | sed "s/filtered_func_data_clean/filtered_func_data/g" | sed "s/1246648320/${numVoxels}/g" | sed "s/1670/${numVols}/g" | sed "s:/Users/xpsy1114/fsl:${fslDir}:g" | sed "s:/func/glm_02.feat:/func/glm_${version_full}_pt0${task_half}.feat:g" | sed "s:/motion/sub-01_1_evlist.txt:/motion/sub-${subjectTag}_${task_half}_evlist.txt:g" | sed "s/sub-01/sub-$subjectTag/g" > $designFsf

    if [ -n "$DRYRUN" ]; then
        echo "DRYRUN would submit: sub-${subjectTag} pt${task_half} ${version_full}  -> $glmDir"
    else
        echo "submitting sub-${subjectTag} pt${task_half} ${version_full}"
        fsl_sub -q short feat $designFsf
    fi
    n_submitted=$((n_submitted+1))
done < "$runList"

echo
echo "Done: $n_submitted submitted, $n_skipped skipped."
if [ "$n_skipped" -gt 0 ]; then
    echo "The skipped ones need their leftover .feat removed or their EVs rebuilt."
fi
