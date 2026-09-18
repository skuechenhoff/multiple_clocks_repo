#!/usr/bin/env python
"""Save event-locked instruction RSA group-t maps as ordered 4-D NIfTIs.

For every stored subject-level beta-map series, this script writes two group
one-sample-t NIfTIs:

* ``resolved``: A--D first presentation, A--D second presentation, empty;
* ``collapsed``: collapsed first presentation, collapsed second, empty.

Both the stored (``raw``) betas and subject/condition whole-brain-demeaned
betas are evaluated.  For plan models available within and across task halves,
an additional 1:1 estimate is formed by averaging the TWO SUBJECT BETA MAPS
before computing the group t-statistic.  T-statistics are never averaged.

The fourth NIfTI dimension is a condition axis, not time in seconds.  Its exact
order is recorded in ``settings.json`` and ``map_index.csv`` in the output
directory.  All outputs are uncorrected t-statistic maps.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
from datetime import date

import nibabel as nib
import numpy as np


PROJECT = "/Users/xpsy1114/Documents/projects/multiple_clocks"
DEFAULT_ROOT = os.path.join(PROJECT, "data/derivatives/group/per_TR")
# DEFAULT_PATTERN = "group_RSA_instr_cumrew_glmbase_instr_{condition}_cropped"
# DEFAULT_OUT = os.path.join(
#     PROJECT, "data/derivatives/group",
#     f"instruction_cumulative_eventlocked_group_tmaps_{date.today().isoformat()}")
DEFAULT_PATTERN = "group_RSA_instr_dir_unord_glmbase_instr_{condition}_cropped"
DEFAULT_OUT = os.path.join(
    PROJECT, "data/derivatives/group",
    f"instruction_dir_unord_eventlocked_group_tmaps_{date.today().isoformat()}")


RESOLVED = [
    "see-A-first", "see-B-first", "see-C-first", "see-D-first",
    "see-A-second", "see-B-second", "see-C-second", "see-D-second",
    "empty-screen",
]
COLLAPSED = [
    "collapsed-first-instruction", "collapsed-second-instruction",
    "empty-screen",
]
ALL_CONDITIONS = RESOLVED[:-1] + COLLAPSED

MODEL_RE = re.compile(
    r"^cropped_masked_smooth_fwhm5_(.+)_beta_std\.nii(?:\.gz)?$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--dir-pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument(
        "--preprocessing", choices=("raw", "demeaned", "both"),
        default="both")
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="optional exact stored model names (mainly for testing)")
    return parser.parse_args()


def condition_dir(args, condition):
    return os.path.join(
        args.root, args.dir_pattern.format(condition=condition))


def resolve_nii(stem):
    for path in (stem, stem + ".gz"):
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(stem + "[.gz]")


def beta_path(args, condition, model):
    return resolve_nii(os.path.join(
        condition_dir(args, condition),
        f"cropped_masked_smooth_fwhm5_{model}_beta_std.nii"))


def beta_path_or_none(args, condition, model):
    """`beta_path` without the exception, for inventorying what is present."""
    try:
        return beta_path(args, condition, model)
    except FileNotFoundError:
        return None


def find_missing_betas(args, models):
    """{model: set(conditions whose beta map is absent)}.

    A run that died on one epoch leaves that epoch's beta maps missing while
    every other epoch is complete. Those inputs are skipped rather than fatal,
    but only the VIEWS that actually need them are dropped: `collapsed-second-
    instruction` feeds the collapsed view only, so the resolved view is
    unaffected and still written.
    """
    missing = {}
    for model in models:
        absent = {c for c in ALL_CONDITIONS
                  if beta_path_or_none(args, c, model) is None}
        if absent:
            missing[model] = absent
    return missing


def views_available(model, missing):
    """Which of the two views can still be built for this model."""
    absent = missing.get(model, set())
    return [view for view, conditions in
            (("resolved", RESOLVED), ("collapsed", COLLAPSED))
            if not (absent & set(conditions))]


def discover_models(args):
    names = []
    for path in glob.glob(os.path.join(
            condition_dir(args, ALL_CONDITIONS[0]),
            "cropped_masked_smooth_fwhm5_*_beta_std.nii*")):
        match = MODEL_RE.match(os.path.basename(path))
        if match:
            names.append(match.group(1))
    models = sorted(set(names))
    if args.models is not None:
        missing = sorted(set(args.models) - set(models))
        if missing:
            raise ValueError(f"requested models not present: {missing}")
        models = list(args.models)
    if not models:
        raise RuntimeError("no input beta maps found")
    return models


def group_mask_path(directory):
    matches = sorted(glob.glob(os.path.join(
        directory, "mask_all_*_subjects.nii*")))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one mask_all_*_subjects NIfTI in {directory}: {matches}")
    return matches[0]


def common_reference_and_mask(args):
    reference = None
    common = None
    paths = {}
    for condition in ALL_CONDITIONS:
        path = group_mask_path(condition_dir(args, condition))
        image = nib.load(path)
        values = np.asarray(image.dataobj) > 0
        if reference is None:
            reference = image
            common = values
        else:
            if (image.shape != reference.shape or
                    not np.allclose(image.affine, reference.affine, atol=1e-3)):
                raise ValueError(f"group-mask grid differs: {path}")
            common &= values
        paths[condition] = path
    return reference, common, paths


def load_subject_map(path, reference, expected_subjects=None):
    image = nib.load(path)
    if (image.shape[:3] != reference.shape or
            not np.allclose(image.affine, reference.affine, atol=1e-3)):
        raise ValueError(f"beta-map grid differs: {path}")
    data = image.get_fdata(dtype=np.float32)
    if data.ndim == 3:
        data = data[..., None]
    if expected_subjects is not None and data.shape[-1] != expected_subjects:
        raise ValueError(
            f"subject count differs: {path} has {data.shape[-1]}, "
            f"expected {expected_subjects}")
    return data


def t_map(subject_maps, brain, demean):
    """One-sample t across subjects, restricted to the common brain mask."""
    values = np.asarray(subject_maps[brain, :].T, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("non-finite beta in common brain")
    if demean:
        values = values - values.mean(axis=1, keepdims=True)
    n_subjects = values.shape[0]
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    statistic = np.divide(
        mean * np.sqrt(n_subjects), sd,
        out=np.zeros_like(mean), where=sd > 0)
    output = np.zeros(brain.shape, dtype=np.float32)
    output[brain] = statistic
    return output


def empty_t_series(reference, n_conditions):
    return np.empty(reference.shape + (n_conditions,), dtype=np.float32)


def header_for_t(reference, n_subjects, description):
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    header.set_intent("t test", (float(n_subjects - 1),), name="group t")
    header["descrip"] = description[:79]
    return header


def save_series(path, series, reference, n_subjects, description):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = nib.Nifti1Image(
        series, reference.affine,
        header=header_for_t(reference, n_subjects, description))
    nib.save(image, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_pairs(models):
    model_set = set(models)
    pairs = []
    for within in models:
        if within.endswith("_rew_within"):
            across = within[:-len("_within")] + "_across"
        elif within.endswith("-exe_split_within"):
            across = within[:-len("_within")] + "_across"
        else:
            continue
        if across not in model_set:
            raise RuntimeError(
                f"plan model {within} has no expected across partner {across}")
        combined = within[:-len("_within")] + "_within-plus-across_1to1"
        pairs.append((within, across, combined))
    return pairs


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def output_paths(args, preprocessing, family, model):
    base = os.path.join(args.out_dir, preprocessing, family)
    return {
        "resolved": os.path.join(
            base, "resolved", f"{model}_group-t_resolved.nii.gz"),
        "collapsed": os.path.join(
            base, "collapsed", f"{model}_group-t_collapsed.nii.gz"),
    }


def save_condition_views(args, preprocessing, family, model, full_series,
                         reference, n_subjects, source_models, index_rows,
                         views=("resolved", "collapsed")):
    paths = output_paths(args, preprocessing, family, model)
    selections = {
        "resolved": [ALL_CONDITIONS.index(c) for c in RESOLVED],
        "collapsed": [ALL_CONDITIONS.index(c) for c in COLLAPSED],
    }
    labels = {"resolved": RESOLVED, "collapsed": COLLAPSED}
    for view in views:
        description = (
            f"uncorrected group t({n_subjects - 1}); {preprocessing}; {view}")
        save_series(
            paths[view], full_series[..., selections[view]], reference,
            n_subjects, description)
        index_rows.append({
            "preprocessing": preprocessing,
            "family": family,
            "output_model": model,
            "source_models": " + ".join(source_models),
            "subject_beta_operation": (
                "identity" if len(source_models) == 1
                else "0.5 * within + 0.5 * across"),
            "view": view,
            "n_volumes": len(labels[view]),
            "volume_labels": " | ".join(labels[view]),
            "df": n_subjects - 1,
            "path": paths[view],
        })


def compute_single_model(args, model, preprocessings, reference, brain,
                         expected_subjects):
    output = {
        prep: empty_t_series(reference, len(ALL_CONDITIONS))
        for prep in preprocessings}
    n_subjects = expected_subjects
    for condition_index, condition in enumerate(ALL_CONDITIONS):
        path = beta_path_or_none(args, condition, model)
        if path is None:
            # NaN, not 0: a zero t-value would read as a real null result.
            # Any view needing this condition is dropped before it is written.
            for preprocessing in preprocessings:
                output[preprocessing][..., condition_index] = np.nan
            continue
        data = load_subject_map(path, reference, n_subjects)
        if n_subjects is None:
            n_subjects = data.shape[-1]
        for preprocessing in preprocessings:
            output[preprocessing][..., condition_index] = t_map(
                data, brain, preprocessing == "demeaned")
        del data
    return output, n_subjects


def compute_plan_pair(args, within, across, preprocessings, reference, brain,
                      expected_subjects):
    output = {
        model: {
            prep: empty_t_series(reference, len(ALL_CONDITIONS))
            for prep in preprocessings}
        for model in (within, across, "combined")}
    n_subjects = expected_subjects
    for condition_index, condition in enumerate(ALL_CONDITIONS):
        within_data = load_subject_map(
            beta_path(args, condition, within), reference, n_subjects)
        if n_subjects is None:
            n_subjects = within_data.shape[-1]
        across_data = load_subject_map(
            beta_path(args, condition, across), reference, n_subjects)
        combined_data = 0.5 * (within_data + across_data)
        for preprocessing in preprocessings:
            demean = preprocessing == "demeaned"
            output[within][preprocessing][..., condition_index] = t_map(
                within_data, brain, demean)
            output[across][preprocessing][..., condition_index] = t_map(
                across_data, brain, demean)
            output["combined"][preprocessing][..., condition_index] = t_map(
                combined_data, brain, demean)
        del within_data, across_data, combined_data
    return output, n_subjects


def main():
    args = parse_args()
    np.random.seed(42)
    models = discover_models(args)
    missing = find_missing_betas(args, models)
    reference, brain, group_masks = common_reference_and_mask(args)
    preprocessings = ([args.preprocessing] if args.preprocessing != "both"
                      else ["raw", "demeaned"])
    # A model with no buildable view is dropped entirely; a plan pair is dropped
    # if EITHER half is incomplete, because the 1:1 average needs both.
    unbuildable = [m for m in models if not views_available(m, missing)]
    models = [m for m in models if m not in unbuildable]
    pairs = [pair for pair in plan_pairs(models)
             if not (missing.get(pair[0]) or missing.get(pair[1]))]
    dropped_pairs = [pair for pair in plan_pairs(models)
                     if missing.get(pair[0]) or missing.get(pair[1])]
    paired_models = {name for within, across, _ in pairs
                     for name in (within, across)}
    unpaired = [model for model in models if model not in paired_models]

    if missing:
        print("\n*** MISSING INPUT BETA MAPS -- these results are INCOMPLETE ***",
              flush=True)
        for model in sorted(missing):
            views = views_available(model, missing)
            print(f"  {model}: no beta for {sorted(missing[model])} "
                  f"-> writing {views or 'NOTHING (model dropped)'}", flush=True)
        if dropped_pairs:
            print(f"  plan pairs dropped (need both halves complete): "
                  f"{[p[2] for p in dropped_pairs]}", flush=True)
        print("*** see missing_inputs.csv in the output directory ***\n", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(
        brain.astype(np.uint8), reference.affine, reference.header),
        os.path.join(args.out_dir, "common_brain_mask.nii.gz"))

    print(f"models={len(models)}; plan pairs={len(pairs)}; "
          f"unpaired={len(unpaired)}; brain voxels={int(brain.sum())}",
          flush=True)
    index_rows = []
    n_subjects = None

    for job_index, (within, across, combined) in enumerate(pairs, 1):
        print(f"[plan pair {job_index}/{len(pairs)}] {within} + {across}",
              flush=True)
        series, n_subjects = compute_plan_pair(
            args, within, across, preprocessings, reference, brain, n_subjects)
        for preprocessing in preprocessings:
            for model in (within, across):
                save_condition_views(
                    args, preprocessing, "stored_models", model,
                    series[model][preprocessing], reference, n_subjects,
                    [model], index_rows, views_available(model, missing))
            save_condition_views(
                args, preprocessing, "plan_within-plus-across_1to1", combined,
                series["combined"][preprocessing], reference, n_subjects,
                [within, across], index_rows,
                views_available(within, missing))
        del series

    for job_index, model in enumerate(unpaired, 1):
        print(f"[unpaired {job_index}/{len(unpaired)}] {model}", flush=True)
        series, n_subjects = compute_single_model(
            args, model, preprocessings, reference, brain, n_subjects)
        for preprocessing in preprocessings:
            save_condition_views(
                args, preprocessing, "stored_models", model,
                series[preprocessing], reference, n_subjects,
                [model], index_rows, views_available(model, missing))
        del series

    write_csv(os.path.join(args.out_dir, "map_index.csv"), index_rows)
    missing_rows = [
        {"model": model, "missing_condition": condition,
         "views_still_written": " | ".join(views_available(model, missing)) or "none",
         "expected_path": os.path.join(
             condition_dir(args, condition),
             f"cropped_masked_smooth_fwhm5_{model}_beta_std.nii[.gz]")}
        for model in sorted(missing) for condition in sorted(missing[model])]
    if missing_rows:
        write_csv(os.path.join(args.out_dir, "missing_inputs.csv"), missing_rows)
    settings = {
        "analysis": "event-locked instruction RSA group t-map export",
        "date": date.today().isoformat(),
        "input_root": os.path.abspath(args.root),
        "input_directory_pattern": args.dir_pattern,
        "input_beta_pattern": (
            "cropped_masked_smooth_fwhm5_{model}_beta_std.nii[.gz]"),
        "n_subjects": n_subjects,
        "degrees_of_freedom": n_subjects - 1,
        "incomplete_inputs": {m: sorted(c) for m, c in missing.items()},
        "models_dropped_entirely": sorted(unbuildable),
        "views_not_written": {
            m: sorted(set(("resolved", "collapsed")) - set(views_available(m, missing)))
            for m in sorted(missing)},
        "n_stored_models": len(models),
        "stored_models": models,
        "n_plan_within_across_pairs": len(pairs),
        "plan_within_across_pairs": [
            {"within": within, "across": across, "combined": combined}
            for within, across, combined in pairs],
        "combined_plan_operation": (
            "For each subject and voxel: 0.5 * within beta + "
            "0.5 * across beta; then one-sample group t against zero. "
            "Group t-statistics were not averaged."),
        "preprocessing": {
            "raw": "stored subject beta maps",
            "demeaned": (
                "subtract each subject/condition/model common-brain spatial "
                "mean before the group t-test")},
        "resolved_volume_order": RESOLVED,
        "collapsed_volume_order": COLLAPSED,
        "condition_axis_note": (
            "NIfTI dimension 4 indexes named event conditions, not seconds"),
        "inference_note": (
            "All images are uncorrected descriptive group t-statistics"),
        "common_brain_voxels": int(brain.sum()),
        "common_brain_mask": os.path.join(
            os.path.abspath(args.out_dir), "common_brain_mask.nii.gz"),
        "source_group_masks": group_masks,
        "subject_order_assumption": (
            "The fourth-dimension subject order is identical across the "
            "upstream merged within/across maps, as produced by the common "
            "group merge pipeline."),
        "script": os.path.abspath(__file__),
        "script_sha256": file_sha256(__file__),
    }
    with open(os.path.join(args.out_dir, "settings.json"), "w") as handle:
        json.dump(settings, handle, indent=2)
    print(f"wrote {len(index_rows)} t-map NIfTIs -> {args.out_dir}")


if __name__ == "__main__":
    main()
