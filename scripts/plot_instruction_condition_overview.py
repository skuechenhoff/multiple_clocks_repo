#!/usr/bin/env python
"""Create uncorrected overview plots for event-locked instruction RSA maps.

The input contains one group 4-D subject map per condition and model. This
script computes group one-sample t maps inside mPFC and bilateral hippocampus,
both on the stored betas and after subject/condition whole-brain demeaning.

For every model family, it writes a two-panel PNG:

* nine event-locked conditions: A--D first, A--D second, empty screen;
* three conditions: collapsed first, collapsed second, empty screen.

Within-half figures contain four traces (mPFC/hippocampus x plan/memory).
Across-half figures contain only the two plan traces because instruction-memory
models have no across-half variance and no valid across map. Each trace uses
one fixed voxel, selected as that trace's largest positive t across the nine
detailed conditions. The collapsed panel reads the same voxel. There is
deliberately no p-value or multiple-comparison correction: these are
descriptive screening plots, and peak selection makes their heights circular
for inference.

The 41 input maps form 15 within plan-memory comparisons and 11 across
plan-only views. No within/across maps are summed here.
"""
import argparse
import csv
import hashlib
import json
import os
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np

import mc.analyse.loso as L


PROJECT = "/Users/xpsy1114/Documents/projects/multiple_clocks"
DEFAULT_ROOT = os.path.join(PROJECT, "data/derivatives/group/per_TR")
DEFAULT_PATTERN = "group_RSA_instr_cumrew_glmbase_instr_{condition}_cropped"
DEFAULT_MPFC_MASK = os.path.join(
    PROJECT, "data/masks/mask_PFC_LR_smoothed_resampled.nii.gz")
DEFAULT_HC_MASK = os.path.join(
    PROJECT, "data/derivatives/group/"
    "per_TR_svc_rewDSR_instr_HO50_HC_bilateral_2026-09-11/masks/"
    "HarvardOxford_bilateral_hippocampus_maxprob_thr50_2mm.nii.gz")
DEFAULT_CONFIG = os.path.join(
    PROJECT, "multiple_clocks_repo/condition_files/rsa_instruction_cumulative_rew.json")

DETAILED = [
    ("see-A-first", "A\nfirst"), ("see-B-first", "B\nfirst"),
    ("see-C-first", "C\nfirst"), ("see-D-first", "D\nfirst"),
    ("see-A-second", "A\nsecond"), ("see-B-second", "B\nsecond"),
    ("see-C-second", "C\nsecond"), ("see-D-second", "D\nsecond"),
    ("empty-screen", "empty\nscreen"),
]
COLLAPSED = [
    ("collapsed-first-instruction", "first\ncollapsed"),
    ("collapsed-second-instruction", "second\ncollapsed"),
    ("empty-screen", "empty\nscreen"),
]
ALL_CONDITIONS = [name for name, _ in DETAILED[:-1] + COLLAPSED]

TRACE_STYLE = {
    ("mPFC", "plan"): ("#E67E22", "mPFC plan"),
    ("HC", "plan"): ("#F2B447", "hippocampus plan"),
    ("mPFC", "memory"): ("#448363", "mPFC memory"),
    ("HC", "memory"): ("#23677E", "hippocampus memory"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--dir-pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--mPFC-mask", default=DEFAULT_MPFC_MASK)
    parser.add_argument("--HC-mask", default=DEFAULT_HC_MASK)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--preprocessing", choices=("raw", "demeaned", "both"),
                        default="both")
    parser.add_argument(
        "--reuse-cache", default="",
        help="existing roi_t_maps_raw_and_demeaned.npz; skips all beta-map reads")
    return parser.parse_args()


def input_dir(args, condition):
    return os.path.join(args.root, args.dir_pattern.format(condition=condition))


def beta_path(args, condition, model):
    directory = input_dir(args, condition)
    stem = os.path.join(
        directory, f"cropped_masked_smooth_fwhm5_{model}_beta_std.nii")
    if os.path.exists(stem):
        return stem
    if os.path.exists(stem + ".gz"):
        return stem + ".gz"
    raise FileNotFoundError(stem + "[.gz]")


def discover_models(args):
    directory = input_dir(args, ALL_CONDITIONS[0])
    pattern = re.compile(
        r"^cropped_masked_smooth_fwhm5_(.+)_beta_std\.nii(?:\.gz)?$")
    models = sorted(match.group(1) for match in
                    (pattern.match(name) for name in os.listdir(directory)) if match)
    for condition in ALL_CONDITIONS:
        present = {model for model in models
                   if os.path.exists(beta_path(args, condition, model))}
        if present != set(models):
            missing = sorted(set(models) - present)
            raise FileNotFoundError(f"{condition} is missing models: {missing}")
    return models


def group_mask_path(directory):
    candidates = sorted(
        os.path.join(directory, name) for name in os.listdir(directory)
        if re.match(r"mask_all_\d+_subjects\.nii(?:\.gz)?$", name))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one group subject mask in {directory}, got {candidates}")
    return candidates[0]


def common_reference_and_brain(args):
    reference = None
    brain = None
    mask_paths = {}
    for condition in ALL_CONDITIONS:
        path = group_mask_path(input_dir(args, condition))
        image = nib.load(path)
        values = np.asarray(image.dataobj) > 0
        if reference is None:
            reference, brain = image, values
        else:
            if image.shape != reference.shape or not np.allclose(
                    image.affine, reference.affine, atol=1e-3):
                raise ValueError(f"condition group-mask grid differs: {path}")
            brain &= values
        mask_paths[condition] = path
    return reference, brain, mask_paths


def one_sample_t(subject_by_voxel):
    return np.asarray(L.tstat(subject_by_voxel), dtype=np.float32)


def load_t_maps(args, models, reference, brain, masks):
    statistics = {
        preprocessing: {
            roi: {model: np.empty((len(ALL_CONDITIONS), int(mask.sum())), np.float32)
                  for model in models}
            for roi, mask in masks.items()}
        for preprocessing in ("raw", "demeaned")
    }
    global_rows = []
    expected_subjects = None
    for condition_index, condition in enumerate(ALL_CONDITIONS):
        print(f"[{condition_index + 1}/{len(ALL_CONDITIONS)}] {condition}", flush=True)
        for model_index, model in enumerate(models):
            path = beta_path(args, condition, model)
            image = nib.load(path)
            if image.shape[:3] != reference.shape or not np.allclose(
                    image.affine, reference.affine, atol=1e-3):
                raise ValueError(f"beta-map grid differs: {path}")
            data = image.get_fdata(dtype=np.float32)
            if data.ndim == 3:
                data = data[..., None]
            if expected_subjects is None:
                expected_subjects = data.shape[-1]
            if data.shape[-1] != expected_subjects:
                raise ValueError(f"subject count differs: {path} has {data.shape[-1]}")
            if not np.isfinite(data[brain]).all():
                raise ValueError(f"non-finite in group brain: {path}")

            whole_brain_mean = data[brain, :].mean(axis=0)
            whole_t = float(one_sample_t(whole_brain_mean[:, None])[0])
            global_rows.append(dict(
                condition=condition, model=model, scope=model_scope(model),
                role=model_role(model), n_subjects=expected_subjects,
                whole_brain_mean=float(whole_brain_mean.mean()),
                whole_brain_sem=float(
                    whole_brain_mean.std(ddof=1) / np.sqrt(expected_subjects)),
                whole_brain_t=whole_t))

            for roi, mask in masks.items():
                subject_by_voxel = data[mask, :].T
                statistics["raw"][roi][model][condition_index] = one_sample_t(
                    subject_by_voxel)
                statistics["demeaned"][roi][model][condition_index] = one_sample_t(
                    subject_by_voxel - whole_brain_mean[:, None])
            del data
            if (model_index + 1) % 10 == 0 or model_index + 1 == len(models):
                print(f"  maps {model_index + 1}/{len(models)}", flush=True)
    return statistics, global_rows, expected_subjects


def model_scope(model):
    if "_across" in model:
        return "across"
    return "within"


def model_role(model):
    # The coefficient name, not the combo suffix, determines its role.  For
    # example ``A_REW-first_exe_vs_instr`` is the PLAN coefficient from a
    # concurrent model even though the combo name itself contains ``instr``.
    return "memory" if "rew_instr" in model.lower() else "plan"


def comparison_specs():
    specs = []
    singles = ["A", "AB", "ABC", "ABCD", "B", "C", "D"]
    for base in singles:
        memory = f"{base}_rew_instr_within"
        for scope in ("within", "across"):
            specs.append(dict(
                name=f"single_{base}_{scope}",
                title=f"Single {base} reward model — plan {scope}",
                kind="single", plan_scope=scope,
                plan=f"{base}_rew_{scope}",
                memory=memory if scope == "within" else None))

    concurrent = [
        ("A", "first_exe_vs_instr"), ("AB", "two_exe_vs_instr"),
        ("ABC", "three_exe_vs_instr"), ("ABCD", "four_exe_vs_instr"),
    ]
    for base, combo in concurrent:
        specs.append(dict(
            name=f"concurrent_{base}_within",
            title=f"Concurrent {base} plan + memory model — within",
            kind="concurrent_combo", plan_scope="within",
            plan=f"{base}_REW-{combo}", memory=f"{base}_REW_INSTR-{combo}"))

    for base in ("A", "B", "C", "D"):
        memory = f"{base}_REW_INSTR-instr_split"
        for scope in ("within", "across"):
            specs.append(dict(
                name=f"split_{base}_{scope}",
                title=f"Split-model {base} coefficient — plan {scope}",
                kind="split_combo", plan_scope=scope,
                plan=f"{base}_REW-exe_split_{scope}",
                memory=memory if scope == "within" else None))
    return specs


def trace_at_peak(statistics, preprocessing, roi, model):
    all_t = statistics[preprocessing][roi][model]
    detailed_indices = [ALL_CONDITIONS.index(name) for name, _ in DETAILED]
    detailed_t = all_t[detailed_indices]
    peak_condition, peak_voxel = np.unravel_index(np.argmax(detailed_t), detailed_t.shape)
    collapsed_indices = [ALL_CONDITIONS.index(name) for name, _ in COLLAPSED]
    return dict(
        peak_voxel=int(peak_voxel), peak_condition=int(peak_condition),
        peak_t=float(detailed_t[peak_condition, peak_voxel]),
        detailed=detailed_t[:, peak_voxel],
        collapsed=all_t[collapsed_indices, peak_voxel])


def plot_comparison(spec, preprocessing, statistics, masks, reference, out_dir):
    traces = []
    role_models = [("plan", spec["plan"])]
    if spec["memory"] is not None:
        role_models.append(("memory", spec["memory"]))
    for role, model in role_models:
        for roi in ("mPFC", "HC"):
            trace = trace_at_peak(statistics, preprocessing, roi, model)
            ijk = np.column_stack(np.where(masks[roi]))[trace["peak_voxel"]]
            mni = nib.affines.apply_affine(reference.affine, ijk)
            trace.update(
                role=role, roi=roi, model=model,
                mni=[int(round(value)) for value in mni])
            traces.append(trace)

    centimetre = 1 / 2.54
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 7, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": 0.7,
    })
    figure, axes = plt.subplots(
        1, 2, figsize=(16 * centimetre, 8.5 * centimetre),
        gridspec_kw={"width_ratios": [3, 1.18]}, sharey=True)
    figure.subplots_adjust(left=.09, right=.985, bottom=.29, top=.75, wspace=.12)
    panels = [
        (axes[0], "detailed event windows", "detailed", DETAILED),
        (axes[1], "collapsed windows", "collapsed", COLLAPSED),
    ]
    for axis, title, value_key, conditions in panels:
        x = np.arange(len(conditions))
        for boundary in np.arange(-.5, len(conditions), 1):
            axis.axvline(boundary, color="#E1E1E1", lw=.5, zorder=0)
        axis.axhline(0, color="#999999", lw=.7, zorder=1)
        for trace in traces:
            colour, _ = TRACE_STYLE[(trace["roi"], trace["role"])]
            axis.plot(x, trace[value_key], "-o", color=colour,
                      lw=1.5, ms=3.2, zorder=3)
        axis.set_xticks(x)
        axis.set_xticklabels([label for _, label in conditions])
        axis.set_xlim(-.5, len(conditions) - .5)
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("group t-statistic")

    legend_keys = [("mPFC", "plan"), ("HC", "plan")]
    if spec["memory"] is not None:
        legend_keys += [("mPFC", "memory"), ("HC", "memory")]
    handles = [
        Line2D([], [], color=TRACE_STYLE[key][0], marker="o", lw=1.5,
               ms=3.2, label=TRACE_STYLE[key][1])
        for key in legend_keys]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .045),
                  frameon=False, ncol=2)
    figure.suptitle(spec["title"], y=.97, fontsize=11)
    figure.text(
        .5, .845,
        f"{preprocessing}; uncorrected fixed peaks selected over detailed conditions",
        ha="center", va="center", fontsize=7, color="#555555")

    target = os.path.join(out_dir, preprocessing, spec["name"] + ".png")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    figure.savefig(target, dpi=300, facecolor="white")
    plt.close(figure)
    return traces, target


def cache_statistics(path, statistics, models):
    arrays = {}
    for preprocessing in statistics:
        for roi in statistics[preprocessing]:
            for model_index, model in enumerate(models):
                arrays[f"{preprocessing}_{roi}_{model_index:02d}"] = (
                    statistics[preprocessing][roi][model])
    np.savez_compressed(path, **arrays)


def load_cached_statistics(path, models):
    cached = np.load(path)
    return {
        preprocessing: {
            roi: {model: cached[f"{preprocessing}_{roi}_{model_index:02d}"]
                  for model_index, model in enumerate(models)}
            for roi in ("mPFC", "HC")}
        for preprocessing in ("raw", "demeaned")}


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    np.random.seed(42)
    models = discover_models(args)
    specs = comparison_specs()
    expected = {spec[key] for spec in specs for key in ("plan", "memory")
                if spec[key] is not None}
    if expected != set(models):
        raise RuntimeError(
            f"comparison mapping does not cover inputs; missing={sorted(set(models)-expected)}, "
            f"invented={sorted(expected-set(models))}")
    reference, brain, group_masks = common_reference_and_brain(args)
    masks = {
        "mPFC": L.load_mask(args.mPFC_mask, reference) & brain,
        "HC": L.load_mask(args.HC_mask, reference) & brain,
    }
    print(f"41 input maps x 11 conditions; common brain={int(brain.sum())}; "
          f"mPFC={int(masks['mPFC'].sum())}; HC={int(masks['HC'].sum())}")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.reuse_cache:
        statistics = load_cached_statistics(args.reuse_cache, models)
        source_dir = os.path.dirname(os.path.abspath(args.reuse_cache))
        with open(os.path.join(source_dir, "settings.json")) as handle:
            cached_settings = json.load(handle)
        n_subjects = cached_settings["n_subjects"]
        with open(os.path.join(source_dir, "whole_brain_offsets.csv"), newline="") as handle:
            global_rows = list(csv.DictReader(handle))
    else:
        statistics, global_rows, n_subjects = load_t_maps(
            args, models, reference, brain, masks)
        cache_statistics(
            os.path.join(args.out_dir, "roi_t_maps_raw_and_demeaned.npz"),
            statistics, models)
    write_csv(os.path.join(args.out_dir, "whole_brain_offsets.csv"), global_rows)

    preprocessings = ([args.preprocessing] if args.preprocessing != "both"
                      else ["raw", "demeaned"])
    trace_rows = []
    figure_rows = []
    for preprocessing in preprocessings:
        for figure_index, spec in enumerate(specs, 1):
            traces, path = plot_comparison(
                spec, preprocessing, statistics, masks, reference, args.out_dir)
            figure_rows.append(dict(
                preprocessing=preprocessing, comparison=spec["name"],
                kind=spec["kind"], plan_scope=spec["plan_scope"],
                plan_model=spec["plan"], memory_model=spec["memory"], figure=path))
            for trace in traces:
                for panel, conditions in (("detailed", DETAILED),
                                          ("collapsed", COLLAPSED)):
                    values = trace[panel]
                    for condition_index, ((condition, _), value) in enumerate(
                            zip(conditions, values)):
                        trace_rows.append(dict(
                            preprocessing=preprocessing,
                            comparison=spec["name"], kind=spec["kind"],
                            plan_scope=spec["plan_scope"], role=trace["role"],
                            roi=trace["roi"], model=trace["model"],
                            peak_mni="/".join(map(str, trace["mni"])),
                            selected_peak_condition=DETAILED[
                                trace["peak_condition"]][0],
                            selected_peak_t=trace["peak_t"], panel=panel,
                            condition_index=condition_index, condition=condition,
                            t=float(value)))
            print(f"[{preprocessing}] figure {figure_index}/{len(specs)}: {spec['name']}")

    write_csv(os.path.join(args.out_dir, "figure_index.csv"), figure_rows)
    write_csv(os.path.join(args.out_dir, "peak_timecourses.csv"), trace_rows)
    settings = dict(
        input_root=args.root, dir_pattern=args.dir_pattern,
        config=os.path.abspath(args.config), config_sha256=file_sha256(args.config),
        n_subjects=n_subjects, common_brain_voxels=int(brain.sum()),
        masks={name: dict(
            path=os.path.abspath(args.mPFC_mask if name == "mPFC" else args.HC_mask),
            n_voxels=int(mask.sum())) for name, mask in masks.items()},
        conditions=dict(detailed=[name for name, _ in DETAILED],
                        collapsed=[name for name, _ in COLLAPSED]),
        group_masks=group_masks, models=models, n_models=len(models),
        comparisons=specs, preprocessing=preprocessings,
        reused_t_cache=(os.path.abspath(args.reuse_cache) if args.reuse_cache else None),
        statistic="one-sample group t at a fixed ROI peak; no correction",
        peak_selection=(
            "separately for each trace, maximum positive t over ROI voxels x "
            "nine detailed conditions; same voxel read in collapsed panel"),
        inference_warning=(
            "descriptive screen only; peak curves are circular and uncorrected; "
            "do not interpret as inferential p values"))
    with open(os.path.join(args.out_dir, "settings.json"), "w") as handle:
        json.dump(settings, handle, indent=2)
    print(f"-> {args.out_dir} ({len(preprocessings) * len(specs)} PNG figures)")


if __name__ == "__main__":
    main()
