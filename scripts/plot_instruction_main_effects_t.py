#!/usr/bin/env python
"""Plot the two instruction-period main effects as peak-voxel group t curves.

By default the figure overlays:

* memory: the instruction-model effect in bilateral MTL (blue), and
* plan: the execution-model effect in mPFC (orange).

Each point is the group t-statistic at one ROI peak voxel and one one-second
instruction condition. Condition 1 (TR0) is plotted at x=0.5, condition 2 at
x=1.5, and so on; thin vertical lines show condition boundaries. The coloured
strip above the axes shows the rewards actually on screen: A--D for 1.5 s each,
then A--D for 1 s each. Coloured horizontal bars above the curves mark
conditions whose supplied voxel-wise FWE p-value is < alpha.

Pass ``--omit-memory`` to render the plan effect alone. The t maps, FWE maps,
masks and optional fixed MNI coordinates are command-line arguments, so
replacing either effect does not require editing this file.
When no coordinate is supplied, the positive peak is selected jointly over
all voxels in that effect's mask and all conditions.

Examples
--------
    conda activate env_multiple_clocks

    # Current defaults; save PDF, SVG, PNG, CSV, JSON and a caption text file.
    python scripts/plot_instruction_main_effects_t.py \
        --out-stem /path/to/results/instruction_memory_and_plan_group_t

    # Replace the plan result while keeping the rest of the figure unchanged.
    python scripts/plot_instruction_main_effects_t.py \
        --plan-t-map /path/to/new/rewDSR_t.nii.gz \
        --plan-fwe-map /path/to/new/rewDSR_voxelFWEp.nii.gz \
        --out-stem /path/to/results/instruction_memory_and_plan_group_t

    # Lock peaks to manuscript coordinates instead of finding them from maps.
    python scripts/plot_instruction_main_effects_t.py \
        --memory-mni=-32,-2,-34 --plan-mni=-6,32,18 \
        --out-stem /path/to/results/instruction_memory_and_plan_group_t

FWE input conventions
---------------------
Files named ``*voxelFWEp*`` are interpreted as p. Files named
``*1minusFWEp*`` or ``*corrp*`` are interpreted as 1-p. Significance is read
at the plotted voxel; it is never inferred from the height of the t curve.
"""
import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


PROJECT = "/Users/xpsy1114/Documents/projects/multiple_clocks"
MEMORY_ROOT = os.path.join(
    PROJECT, "data/derivatives/group/within_th_only_intr-vs-exe_allTRs_DEMEANED_2026-09-02")
PLAN_ROOT = os.path.join(
    PROJECT, "data/derivatives/group/per_TR_svc_instruction_rewDSR_allTR_2026-08-28")

DEFAULT_MEMORY_T = os.path.join(MEMORY_ROOT, "MTL/rewDSR_instr_t.nii.gz")
DEFAULT_MEMORY_FWE = os.path.join(MEMORY_ROOT, "MTL/rewDSR_instr_voxelFWEp.nii.gz")
DEFAULT_MEMORY_MASK = os.path.join(PROJECT, "data/masks/Garvert_MTL_2mm.nii.gz")
DEFAULT_PLAN_T = os.path.join(PLAN_ROOT, "mPFC/rewDSR_t.nii.gz")
DEFAULT_PLAN_FWE = os.path.join(PLAN_ROOT, "mPFC/rewDSR_voxelFWEp.nii.gz")
DEFAULT_PLAN_MASK = os.path.join(
    PROJECT, "data/masks/mask_PFC_LR_smoothed_resampled.nii.gz")

MEMORY_COLOUR = "#23677E"
PLAN_COLOUR = "#E67E22"
SIGNIFICANCE_ALPHA = 0.05

# Seconds from instruction onset. The blank final two seconds are deliberately
# left blank: every reward is repeated for one second, including D.
REWARD_SCHEDULE = [
    (0.0, 1.5, "A", "#F15A29"), (1.5, 3.0, "B", "#F7931E"),
    (3.0, 4.5, "C", "#C7C6E2"), (4.5, 6.0, "D", "#6B60AA"),
    (6.0, 7.0, "A", "#F15A29"), (7.0, 8.0, "B", "#F7931E"),
    (8.0, 9.0, "C", "#C7C6E2"), (9.0, 10.0, "D", "#6B60AA"),
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory-t-map", default=DEFAULT_MEMORY_T)
    ap.add_argument("--memory-fwe-map", default=DEFAULT_MEMORY_FWE)
    ap.add_argument("--memory-mask", default=DEFAULT_MEMORY_MASK)
    ap.add_argument("--memory-mni", default="",
                    help="optional fixed peak as x,y,z; default: positive mask peak")
    ap.add_argument("--memory-label", default="Memory: bilateral MTL")
    ap.add_argument("--omit-memory", action="store_true",
                    help="render only the plan effect (current publication choice)")
    ap.add_argument("--plan-t-map", default=DEFAULT_PLAN_T)
    ap.add_argument("--plan-fwe-map", default=DEFAULT_PLAN_FWE)
    ap.add_argument("--plan-mask", default=DEFAULT_PLAN_MASK)
    ap.add_argument("--plan-mni", default="",
                    help="optional fixed peak as x,y,z; default: positive mask peak")
    ap.add_argument("--plan-label", default="Plan: mPFC")
    ap.add_argument("--alpha", type=float, default=SIGNIFICANCE_ALPHA)
    ap.add_argument("--title", default="Plan and memory representations during instruction")
    ap.add_argument("--out-stem", required=True,
                    help="output path without extension; parent directory is created")
    ap.add_argument("--show", action="store_true")
    return ap.parse_args()


def load_4d(path, what):
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 3:
        data = data[..., None]
    if data.ndim != 4:
        raise ValueError(f"{what} must be 3-D or 4-D, got {data.shape}: {path}")
    if not np.isfinite(data).all():
        raise ValueError(f"{what} contains non-finite values: {path}")
    return img, data


def load_mask(path, reference):
    img = nib.load(path)
    same_grid = (img.shape[:3] == reference.shape[:3]
                 and np.allclose(img.affine, reference.affine, atol=1e-3))
    if not same_grid:
        img = resample_from_to(img, (reference.shape[:3], reference.affine), order=0)
    mask = np.asarray(img.dataobj) > 0
    if not mask.any():
        raise ValueError(f"mask is empty on the t-map grid: {path}")
    return mask


def fwe_p_values(path, reference, n_conditions):
    img, values = load_4d(path, "FWE map")
    same_grid = (img.shape[:3] == reference.shape[:3]
                 and np.allclose(img.affine, reference.affine, atol=1e-3))
    if not same_grid:
        volumes = []
        for condition in range(values.shape[-1]):
            one = nib.Nifti1Image(values[..., condition], img.affine, img.header)
            volumes.append(np.asarray(resample_from_to(
                one, (reference.shape[:3], reference.affine), order=0).dataobj))
        values = np.stack(volumes, axis=-1)
    if values.shape[-1] != n_conditions:
        raise ValueError(
            f"FWE map has {values.shape[-1]} conditions, t map has {n_conditions}: {path}")
    name = os.path.basename(path).lower()
    if "1minusfwep" in name or "corrp" in name:
        values = 1.0 - values
    elif "fwep" not in name:
        raise ValueError(
            "cannot infer whether FWE map stores p or 1-p; name it with "
            f"'FWEp', '1minusFWEp', or 'corrp': {path}")
    return np.clip(values, 0.0, 1.0)


def parse_mni(text):
    if not text:
        return None
    values = [float(v) for v in text.split(",")]
    if len(values) != 3:
        raise ValueError(f"MNI coordinate must be x,y,z, got {text!r}")
    return np.asarray(values)


def effect_from_maps(label, colour, t_path, fwe_path, mask_path, mni_text, alpha):
    t_img, t_data = load_4d(t_path, "t map")
    mask = load_mask(mask_path, t_img)
    p_data = fwe_p_values(fwe_path, t_img, t_data.shape[-1])
    requested_mni = parse_mni(mni_text)

    if requested_mni is None:
        masked_t = np.where(mask[..., None], t_data, -np.inf)
        peak = np.unravel_index(np.argmax(masked_t), masked_t.shape)
        ijk, peak_condition = peak[:3], int(peak[3])
    else:
        ijk = tuple(np.rint(nib.affines.apply_affine(
            np.linalg.inv(t_img.affine), requested_mni)).astype(int))
        if any(i < 0 or i >= n for i, n in zip(ijk, t_img.shape[:3])):
            raise ValueError(f"requested {label} MNI coordinate is outside the map: {requested_mni}")
        if not mask[ijk]:
            raise ValueError(f"requested {label} MNI coordinate is outside its ROI mask: {requested_mni}")
        peak_condition = int(np.argmax(t_data[ijk]))

    mni = nib.affines.apply_affine(t_img.affine, ijk)
    t_curve = np.asarray(t_data[ijk], dtype=float)
    p_curve = np.asarray(p_data[ijk], dtype=float)
    return dict(
        label=label, colour=colour, t_map=os.path.abspath(t_path),
        fwe_map=os.path.abspath(fwe_path), mask=os.path.abspath(mask_path),
        ijk=[int(v) for v in ijk], mni=[float(v) for v in mni],
        peak_condition=peak_condition, peak_t=float(t_curve[peak_condition]),
        peak_p_FWE=float(p_curve[peak_condition]), n_mask_voxels=int(mask.sum()),
        t=t_curve, p_FWE=p_curve, significant=p_curve < alpha)


def contiguous_runs(flags):
    padded = np.r_[False, np.asarray(flags, dtype=bool), False].astype(int)
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def readable_text_colour(hex_colour):
    rgb = np.asarray([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    luminance = float(np.dot(linear, [.2126, .7152, .0722]))
    return "white" if luminance < 0.45 else "black"


def plot_effects(effects, title, alpha, out_stem, show=False):
    n_conditions = len(effects[0]["t"])
    if any(len(effect["t"]) != n_conditions for effect in effects):
        raise ValueError("both t maps must contain the same number of conditions")
    centres = np.arange(n_conditions) + 0.5

    cm = 1 / 2.54
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 11,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(9.0 * cm, 7.5 * cm))
    # Leave room for the reward strip above and the three-entry legend below.
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.34, top=0.70)

    for boundary in range(n_conditions + 1):
        ax.axvline(boundary, color="#D8D8D8", lw=0.45, zorder=0)
    ax.axhline(0, color="#999999", lw=0.65, zorder=1)
    for effect in effects:
        ax.plot(centres, effect["t"], "-o", color=effect["colour"],
                lw=1.6, ms=3.5, zorder=3, label=effect["label"])

    ax.set_xlim(0, n_conditions)
    ax.set_xticks(centres)
    ax.set_xticklabels([str(i) for i in range(1, n_conditions + 1)])
    ax.set_xlabel("instruction condition (1 s each)")
    ax.set_ylabel("group t-statistic")
    ax.spines[["top", "right"]].set_visible(False)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    sig_y = [1.035, 1.095]
    for effect, y in zip(effects, sig_y):
        for start, stop in contiguous_runs(effect["significant"]):
            ax.plot([start + 0.08, stop - 0.08], [y, y], transform=trans,
                    color=effect["colour"], lw=2.5, solid_capstyle="butt",
                    clip_on=False, zorder=5)

    strip_lo, strip_hi = 1.17, 1.36
    for start, stop, reward, colour in REWARD_SCHEDULE:
        if start >= n_conditions:
            continue
        stop = min(stop, n_conditions)
        ax.add_patch(plt.Rectangle(
            (start, strip_lo), stop - start, strip_hi - strip_lo,
            transform=trans, facecolor=colour, edgecolor="white", lw=0.4,
            clip_on=False, zorder=4))
        ax.text((start + stop) / 2, (strip_lo + strip_hi) / 2, reward,
                transform=trans, ha="center", va="center", fontsize=8,
                color=readable_text_colour(colour), clip_on=False, zorder=5)

    handles = [Line2D([], [], color=e["colour"], marker="o", lw=1.6,
                      ms=3.5, label=e["label"]) for e in effects]
    handles.append(Line2D([], [], color="#333333", lw=2.5,
                          label=rf"condition $p_{{FWE}} < {alpha:g}$"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              frameon=False, ncol=1)
    fig.suptitle(title, y=0.97, fontsize=11)

    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(f"{out_stem}.{extension}", dpi=600)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def write_outputs(effects, args):
    def formatted_p(p_value):
        if p_value == 0:
            return "<0.0001"
        return f"={p_value:.4g}"

    rows = []
    for effect in effects:
        for condition, (t_value, p_value, significant) in enumerate(
                zip(effect["t"], effect["p_FWE"], effect["significant"])):
            rows.append(dict(
                effect=effect["label"], condition=condition + 1, TR=condition,
                interval_start_s=condition, interval_end_s=condition + 1,
                plot_x=condition + 0.5, t=float(t_value), p_FWE=float(p_value),
                significant_FWE=bool(significant)))
    with open(f"{args.out_stem}.csv", "w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    settings = dict(
        figure="peak-voxel group t-statistics across one-second instruction conditions",
        alpha=args.alpha,
        condition_note="TR0 is 0--1 s and is plotted at 0.5; TR1 is plotted at 1.5; etc.",
        significance_note=(
            "bars use the supplied voxel-wise FWE map at the plotted voxel; "
            "the correction family is inherited from that input map"),
        reward_schedule=[dict(start_s=a, end_s=b, reward=r, colour=c)
                         for a, b, r, c in REWARD_SCHEDULE],
        effects=[{key: value.tolist() if isinstance(value, np.ndarray) else value
                  for key, value in effect.items()} for effect in effects],
        command_line_options=vars(args))
    with open(f"{args.out_stem}.json", "w") as file_handle:
        json.dump(settings, file_handle, indent=2)

    lines = [
        "Points show the group t-statistic at each effect's peak voxel for each ",
        "one-second instruction condition. Conditions are plotted at interval centres; ",
        "thin vertical lines mark their boundaries. The upper strip shows the reward ",
        "visible on screen. Horizontal coloured bars mark conditions with voxel-wise ",
        f"FWE-corrected p < {args.alpha:g}, using the supplied correction maps. ",
    ]
    for effect in effects:
        xyz = ", ".join(str(int(round(v))) for v in effect["mni"])
        lines.append(
            f"{effect['label']}: MNI [{xyz}], peak condition "
            f"{effect['peak_condition'] + 1} (TR{effect['peak_condition']}), "
            f"t={effect['peak_t']:.2f}, "
            f"p_FWE{formatted_p(effect['peak_p_FWE'])}. ")
    with open(f"{args.out_stem}_caption.txt", "w") as file_handle:
        file_handle.write("".join(lines) + "\n")


def main():
    args = parse_args()
    effects = []
    if not args.omit_memory:
        effects.append(effect_from_maps(
            args.memory_label, MEMORY_COLOUR, args.memory_t_map,
            args.memory_fwe_map, args.memory_mask, args.memory_mni, args.alpha))
    effects.append(effect_from_maps(
        args.plan_label, PLAN_COLOUR, args.plan_t_map,
        args.plan_fwe_map, args.plan_mask, args.plan_mni, args.alpha))
    plot_effects(effects, args.title, args.alpha, args.out_stem, show=args.show)
    write_outputs(effects, args)
    for effect in effects:
        xyz = [int(round(v)) for v in effect["mni"]]
        significant = np.flatnonzero(effect["significant"]) + 1
        print(f"{effect['label']}: MNI {xyz}, peak condition "
              f"{effect['peak_condition'] + 1} (TR{effect['peak_condition']}), "
              f"t={effect['peak_t']:.3f}, p_FWE={effect['peak_p_FWE']:.4g}; "
              f"significant conditions={significant.tolist()}")
    print(f"-> {args.out_stem}.pdf (+ .svg, .png, .csv, .json, _caption.txt)")


if __name__ == "__main__":
    main()
