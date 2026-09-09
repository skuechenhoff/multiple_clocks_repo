#!/usr/bin/env python
"""
Single runner for the instruction-phase group analysis, per TR or per condition.

All statistics and plotting live in `mc.analyse.loso`; this file only wires
them to a command line. Three modes:

    --mode run    analyse and write results (no figures)
    --mode plot   load an existing --out-dir and plot it, no recomputation
    --mode both   run, then plot (default)

The test is a small-volume corrected max-t sign-flip permutation over voxels
x levels of the third axis inside each a-priori mask, plus the LOSO
cross-validated readout, and optionally whole-brain t / FWE-p / uncorrected-p
volumes for visualisation. See `mc/analyse/loso.py` for what each of those
means and what it does not.

--axis picks what that third axis is:
    tr         (default) the ordered seconds of the instruction period, from
               --trs. Folders are group_..._01-TR{tr}_cropped, peaks are
               reported as TR numbers, and the figure is a timecourse.
    condition  named conditions, from --conditions -- the instruction-epoch
               GLMs (instr_see-A-first, ...), whose folders are
               group_..._glmbase_{condition}_cropped. The statistics are
               IDENTICAL: max-t over voxels x levels, one family per mask.
               What changes is the reporting: peak_TR fields hold the condition
               name, and the figure is drawn as points with no joining line,
               because these conditions have no ordering and a line would
               assert a progression the design does not contain.

Multiple comparisons across models are NOT corrected: each map's peak_p_FWE is
corrected over its own voxels x TRs only. The family size is recorded in
settings.json.

Examples
--------
    conda activate env_multiple_clocks

    # analyse one per-TR RSA folder in three masks, with whole-brain maps
    python per_TR_loso.py --mode both \
        --dir-pattern "group_RSA_instr_test_full_glmbase_01-TR{tr}_cropped" \
        --mask mPFC=../../data/masks/mask_PFC_LR_smoothed_resampled.nii.gz \
        --mask MTL=../../data/masks/Garvert_MTL_2mm.nii.gz \
        --wholebrain \
        --out-dir .../per_TR_svc_instr_test_full_allTR_2026-08-28

    # the same thing over the instruction-epoch conditions
    python per_TR_loso.py --mode both --axis condition \
        --dir-pattern "group_RSA_instr_cumrew_glmbase_{condition}_cropped" \
        --conditions instr_see-A-first,instr_see-B-first,instr_see-C-first,instr_see-D-first \
        --mask mPFC=../../data/masks/mask_PFC_LR_smoothed_resampled.nii.gz \
        --out-dir .../per_condition_svc_instr_cumrew_2026-09-09

    # re-plot a finished run without touching the statistics
    python per_TR_loso.py --mode plot \
        --out-dir .../per_TR_svc_split_rew_DSR_allTR_2026-08-27 \
        --plot-models curr_rew,next_rew,two_next_rew,three_next_rew

Outputs (into --out-dir)
    settings.json                              every parameter + resolved inputs
    summary_table.csv                          one row per (mask, model)
    {mask}/{model}_svc_summary.json            peak stats, both signs, per-TR traces
    {mask}/{model}_loso_results.json           LOSO timecourse per k
    {mask}/{model}_loso_k{K}.npy               per-subject held-out beta x TR
    {mask}/{model}_t.nii.gz                    t inside the mask (4-D over TRs)
    {mask}/{model}_voxel1minusFWEp.nii.gz      1-p; threshold 0.95 for p<.05
    {mask}/{model}_voxel1minusFWEp_neg.nii.gz  same, negative direction
    {mask}/{model}_voxelFWEp.nii.gz            raw FWE p
    {mask}/{model}_null_max_t.npy              the permutation null itself
  with --wholebrain, additionally:
    wholebrain/{model}_t.nii.gz                observed t, whole brain
    wholebrain/{model}_1minusFWEp.nii.gz       1-p, FWE over whole brain x TRs
    wholebrain/{model}_1minusp_uncorr.nii.gz   1-p, uncorrected
    wholebrain/{model}_summary.json
    wholebrain_summary_table.csv
  with plotting:
    per_TR_timecourses.pdf / .jpeg / _peaks.csv
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import mc
import mc.analyse.loso as L


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["run", "plot", "both"], default="both")
    ap.add_argument("--out-dir", required=True)
    # inputs (run modes)
    ap.add_argument("--root", default="/Users/xpsy1114/Documents/projects/multiple_clocks/"
                                      "data/derivatives/group/per_TR")
    ap.add_argument("--dir-pattern",
                    default="group_RSA_instr_test_full_glmbase_01-TR{tr}_cropped")
    ap.add_argument("--mask", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--models", default="", help="comma-separated; default = all found")
    ap.add_argument("--axis", choices=L.AXIS_KINDS, default="tr",
                    help="what the third axis of the maps is. 'tr' (default): "
                         "ordered seconds of the instruction period, taken from "
                         "--trs. 'condition': named conditions, taken from "
                         "--conditions -- the instruction-epoch GLMs, which have "
                         "no ordering. The statistics are identical (max-t over "
                         "voxels x levels); what changes is that peaks are "
                         "reported by condition name and the figure is drawn as "
                         "points rather than as a timecourse.")
    ap.add_argument("--trs", default="0,1,2,3,4,5,6,7,8,9,10,11",
                    help="axis levels when --axis tr")
    ap.add_argument("--conditions", default="",
                    help="axis levels when --axis condition, comma-separated, "
                         "e.g. instr_see-A-first,instr_see-B-first. Substituted "
                         "into --dir-pattern as {condition} (or {tr}/{level}).")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", default="50,100,200")
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip models whose outputs already exist in --out-dir; "
                         "the summary tables are still rebuilt from every model "
                         "present on disk")
    ap.add_argument("--demean", action="store_true",
                    help="subtract each subject's whole-brain mean, separately "
                         "per TR, before any statistic is computed. Turns the "
                         "test from 'is this voxel's beta above zero' into 'is "
                         "it above this subject's own brain-wide level at this "
                         "TR', which is what you want when a model carries a "
                         "global offset. Implies reading the whole brain.")
    ap.add_argument("--skip-input-check", action="store_true",
                    help="do not verify that every beta map is complete before "
                         "starting; only useful if the check is misfiring")
    ap.add_argument("--wholebrain", action="store_true",
                    help="also write whole-brain t / FWE-p / uncorrected-p volumes")
    ap.add_argument("--n-perm-wholebrain", type=int, default=1000)
    ap.add_argument("--wholebrain-neg", action="store_true")
    ap.add_argument("--wholebrain-models", default="",
                    help="subset to write whole-brain volumes for (default: all)")
    # plotting
    ap.add_argument("--plot-models", default="",
                    help="comma-separated models to plot; default = the four "
                         "reward channels if present, else every model")
    ap.add_argument("--plot-masks", default="", help="default: every mask folder")
    ap.add_argument("--plot-k", default="100")
    ap.add_argument("--plot-name", default="per_TR_timecourses")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    if args.mode in ("run", "both") and not args.mask:
        ap.error("--mask is required when --mode is 'run' or 'both'")
    if args.axis == "condition" and not args.conditions:
        ap.error("--axis condition needs --conditions (comma-separated names)")
    return args


def level_str(v):
    """How one axis level is printed: 'TR4' for a numeric level, the bare name
    for a condition. Padded to 4 so the TR form lines up exactly as before."""
    return f"TR{v}" if isinstance(v, int) else str(v)


def axis_levels(args):
    """The levels of the third axis: ints for 'tr', names for 'condition'."""
    if args.axis == "condition":
        return [c for c in args.conditions.split(",") if c]
    return [int(x) for x in args.trs.split(",")]


def do_run(args):
    trs = axis_levels(args)
    k_values = [int(x) for x in args.k.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    ref, brain = L.load_ref(args.root, args.dir_pattern, trs)
    masks = L.load_masks(args.mask, ref, brain)

    # One read per model serves everything: the whole brain when asked for,
    # else just the union of the masks. Each mask is a column subset of it.
    if args.wholebrain or args.demean:
        union = brain.copy()
    else:
        union = np.zeros(brain.shape, bool)
        for v in masks.values():
            union |= v["bool"]
    union_ijk = np.where(union)
    cols_of = {n: np.flatnonzero(v["bool"][union_ijk]) for n, v in masks.items()}
    print(f"[info] extracting {union.sum()} voxels per model", file=sys.stderr)

    models = ([m for m in args.models.split(",") if m] if args.models
              else L.discover_models(args.root, args.dir_pattern, trs[0]))
    wb_models = ([m for m in args.wholebrain_models.split(",") if m]
                 if args.wholebrain_models else models)
    if not args.skip_input_check:
        L.check_inputs(args.root, args.dir_pattern, models, trs)
    todo = models
    if args.resume:
        todo = [m for m in models
                if not L.model_is_done(args.out_dir, m, list(masks), not args.no_cv,
                                       args.wholebrain and m in wb_models)]
        print(f"[info] resume: {len(models) - len(todo)} of {len(models)} models "
              f"already complete, {len(todo)} to run", file=sys.stderr)
    print(f"[info] {len(todo)} maps x {len(masks)} masks x {len(trs)} "
          f"{'TRs' if args.axis == 'tr' else 'conditions'}", file=sys.stderr)

    json.dump(dict(
        root=args.root, dir_pattern=args.dir_pattern, trs=trs,
        axis=args.axis, levels=trs,
        axis_note=("the third axis is ordered time (seconds of the instruction "
                   "period); peaks are reported as TR numbers"
                   if args.axis == "tr" else
                   "the third axis is a set of NAMED CONDITIONS with no "
                   "ordering; peak_TR fields hold the condition name, and "
                   "max-t is corrected over voxels x conditions jointly -- the "
                   "same family-wise correction, but it must not be read as a "
                   "timecourse"),
        masks={k: dict(path=v["path"], n_vox_in_brain=v["n_vox"])
               for k, v in masks.items()},
        models=models, n_models=len(models), n_perm=args.n_perm, seed=args.seed,
        k_values=k_values, cv=not args.no_cv, numpy_seed=42,
        demean=bool(args.demean),
        demean_note=("each subject's whole-brain mean was subtracted separately "
                     "per TR before any statistic, so t tests regional deviation "
                     "from that subject's global level rather than deviation "
                     "from zero. Applied to the data, hence identically to every "
                     "permutation. A model-wide additive offset -- e.g. one "
                     "induced by correlated instruction/execution regressors -- "
                     "cannot survive it; a genuinely regional effect can."
                     if args.demean else "not applied; t is against zero"),
        wholebrain=bool(args.wholebrain), n_perm_wholebrain=args.n_perm_wholebrain,
        wholebrain_neg=bool(args.wholebrain_neg),
        brain_mask_note=("group brain mask = intersection of mask_all_32_subjects "
                         f"over all included TRs ({int(brain.sum())} voxels)"),
        wholebrain_note=("whole-brain maps are corrected over the entire brain mask "
                         "x all included TRs; that FWE p is much stricter than, and "
                         "not comparable to, the small-volume p in the mask folders. "
                         "The uncorrected p maps are for visualisation only."),
        multiple_comparison_note=(
            "peak_p_FWE is corrected over voxels x TRs WITHIN each mask only. "
            f"{len(models)} maps x {len(masks)} masks were tested; no correction "
            "across that family was applied."),
        negative_direction_note=(
            "peak_p_FWE_neg reuses the same sign-flip null (symmetric) as a second "
            "one-sided test; it is not corrected for testing both signs."),
    ), open(os.path.join(args.out_dir, "settings.json"), "w"), indent=2)

    for name in masks:
        os.makedirs(os.path.join(args.out_dir, name), exist_ok=True)
    if args.wholebrain:
        os.makedirs(os.path.join(args.out_dir, "wholebrain"), exist_ok=True)

    for mi, model in enumerate(todo, 1):
        t0 = time.time()
        Du = L.read_model_columns(args.root, args.dir_pattern, model, trs, union_ijk)
        if args.demean:
            # Per subject, per TR, over the whole brain mask. Done here, on the
            # data, so run_svc / run_loso / run_wholebrain and every one of
            # their sign-flip permutations see exactly the same array
            # (CLAUDE.md rule 4) -- there is no separate permutation path.
            Du -= Du.mean(axis=1, keepdims=True)

        if args.wholebrain and model in wb_models:
            wb_dir = os.path.join(args.out_dir, "wholebrain")
            Tw, nmw, p_fwe, p_unc, p_unc_neg, wsum = L.run_wholebrain(
                Du, ref, union_ijk, n_perm=args.n_perm_wholebrain,
                seed=args.seed, want_neg=args.wholebrain_neg, trs=trs)
            wsum["model"] = model
            L.write_wholebrain_maps(wb_dir, model, Tw, p_fwe, p_unc, p_unc_neg,
                                    nmw, ref, union_ijk, len(trs))
            json.dump(wsum, open(os.path.join(wb_dir, f"{model}_summary.json"), "w"),
                      indent=2)
            print(f"  {'*' if wsum['peak_p_FWE'] < 0.05 else ' '} [{mi:2d}/{len(todo)}]"
                  f" WHOLEBRAIN {model:42s} t={wsum['peak_t']:5.2f} "
                  f"{level_str(wsum['peak_TR']):<4s} p={wsum['peak_p_FWE']:.4f} "
                  f"MNI={wsum['peak_mni']}", flush=True)
            del Tw, nmw, p_fwe, p_unc, p_unc_neg

        for name, v in masks.items():
            cols = cols_of[name]
            D = np.ascontiguousarray(Du[:, cols, :])
            ijk = tuple(a[cols] for a in union_ijk)
            Tobs, nmt, svc = L.run_svc(D, ref, ijk, n_perm=args.n_perm,
                                       seed=args.seed, trs=trs)
            svc["model"], svc["mask"] = model, name
            L.write_mask_maps(os.path.join(args.out_dir, name), model, Tobs, nmt,
                              ref, ijk, len(trs))
            json.dump(svc, open(os.path.join(args.out_dir, name,
                                             f"{model}_svc_summary.json"), "w"), indent=2)
            if not args.no_cv:
                lo, held_by_k = L.run_loso(D, k_values, n_perm=args.n_perm,
                                           seed=args.seed, trs=trs)
                for kk, held in held_by_k.items():
                    np.save(os.path.join(args.out_dir, name,
                                         f"{model}_loso_k{kk}.npy"), held)
                json.dump(lo, open(os.path.join(args.out_dir, name,
                                                f"{model}_loso_results.json"), "w"),
                          indent=2)
            print(f"{'*' if svc['peak_p_FWE'] < 0.05 else ' '} [{mi:2d}/{len(todo)}] "
                  f"{name:7s} {model:42s} t={svc['peak_t']:5.2f} "
                  f"{level_str(svc['peak_TR']):<4s} p={svc['peak_p_FWE']:.4f}   "
                  f"(neg t={svc['peak_t_neg']:6.2f} p={svc['peak_p_FWE_neg']:.4f})",
                  flush=True)
        del Du
        print(f"    -- {model} done in {time.time()-t0:.0f}s", file=sys.stderr, flush=True)

    # Tables are rebuilt from every per-model json on disk, so a resumed run
    # still produces complete tables covering the models an earlier invocation
    # finished.
    rows, wb_rows = L.collect_rows(args.out_dir, list(masks), models,
                                   cv=not args.no_cv, wholebrain=args.wholebrain)
    if wb_rows:
        print("-> " + str(L.write_table(
            os.path.join(args.out_dir, "wholebrain_summary_table.csv"), wb_rows)))
    print(f"\n[info] summary table covers {len(set(r['model'] for r in rows))} "
          f"of {len(models)} models")
    print("-> " + str(L.write_table(
        os.path.join(args.out_dir, "summary_table.csv"), rows)))


def do_plot(args):
    # The axis kind is taken from the finished run's settings.json when there is
    # one, so `--mode plot` on an existing folder draws it the way it was run
    # (and an older folder with no 'axis' key stays a timecourse).
    try:
        axis_kind = L.load_settings(args.out_dir).get("axis", "tr")
    except (OSError, ValueError):
        axis_kind = args.axis
    if args.plot_models:
        models = [m for m in args.plot_models.split(",") if m]
    else:
        settings = L.load_settings(args.out_dir)
        # the four split reward channels, under whatever names this run used
        four = [m for m in settings["models"]
                if L.base_channel(m) in L.CHANNEL_COLOURS
                and not m.endswith("_instr")]
        models = four or settings["models"]
        print(f"[info] plotting {len(models)} models "
              f"({'reward channels' if four else 'all models'})", file=sys.stderr)
    masks = [m for m in args.plot_masks.split(",") if m] or None
    L.plot_per_TR_timecourses(args.out_dir, models, masks=masks, k=args.plot_k,
                              out_name=args.plot_name, show=not args.no_show,
                              axis_kind=axis_kind)


def main():
    args = parse_args()
    np.random.seed(42)
    if args.mode in ("run", "both"):
        do_run(args)
    if args.mode in ("plot", "both"):
        do_plot(args)


if __name__ == "__main__":
    main()
