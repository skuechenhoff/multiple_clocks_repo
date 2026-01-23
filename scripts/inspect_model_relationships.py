#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two model RDMs (e.g., location vs DSR), highlight strongest positive
contributors to their correlation, orthogonalize DSR against location, and
inspect how data RDMs relate to the orthogonalized DSR.
"""

import argparse
import json
import os
import pickle
import collections
import textwrap
from fnmatch import fnmatch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

import mc


def pair_correct_tasks(data_dict, keys_list):
    """
    data_dict: dict with keys like 'A1_forw_A_reward'
    keys_list: ordered list of keys you want to include and in what order
    Returns two matrices: one for the first element of each pair, one for its match.
    """
    task_pairs = {"1_forw": "2_backw", "1_backw": "2_forw"}
    th_1, th_2, paired_list_control = [], [], []
    for key in keys_list:
        assert key in data_dict, "Missmatch between model rdm keys and data RDM keys"
        task, direction, state, phase = key.split("_")
        pair_suffix = task_pairs.get(f"{task[-1]}_{direction}")
        pair_key = f"{task[0]}{pair_suffix}_{state}_{phase}"
        if pair_key in data_dict:
            th_1.append(np.asarray(data_dict[key]))
            th_2.append(np.asarray(data_dict[pair_key]))
            paired_list_control.append(f"{key} | {pair_key}")

    th_1 = np.vstack(th_1)
    th_2 = np.vstack(th_2)
    return th_1, th_2, paired_list_control


def build_ev_keys(all_EV_keys, parts_to_use):
    for _p in ("task", "direction", "state", "phase"):
        if _p not in parts_to_use:
            raise ValueError(f"Missing selection.parts['{_p}'] in config.")

    EV_keys = []
    for ev in sorted(all_EV_keys):
        task, direction, state, phase = ev.split("_")
        for name, value in zip(
            ["task", "direction", "state", "phase"],
            [task, direction, state, phase],
        ):
            part = parts_to_use[name]
            includes = part.get("include", [])
            excludes = part.get("exclude", [])
            if any(fnmatch(value, pat) for pat in excludes):
                break
            if includes and not any(fnmatch(value, pat) for pat in includes):
                break
        else:
            EV_keys.append(ev)

    return EV_keys


def build_point_labels(paired_labels, include_diagonal):
    n = len(paired_labels)
    k = 0 if include_diagonal else 1
    iu = np.triu_indices(n, k=k)
    point_labels = [
        f"{paired_labels[i]} vs {paired_labels[j]}" for i, j in zip(iu[0], iu[1])
    ]
    return point_labels


def parse_label_events(label):
    parts = label.split(" vs ")
    events = []
    for side in parts:
        halves = side.split(" | ")
        for ev in halves:
            tokens = ev.split("_")
            if len(tokens) == 4:
                events.append(
                    {
                        "task": tokens[0],
                        "direction": tokens[1],
                        "state": tokens[2],
                        "phase": tokens[3],
                    }
                )
    return events


def summarize_top_labels(top_labels):
    phase_counts = collections.Counter()
    task_counts = collections.Counter()
    task_combo_counts = collections.Counter()
    direction_counts = collections.Counter()
    combo_phase_counts = collections.Counter()
    label_direction_presence = collections.Counter()

    for label in top_labels:
        if not label:
            continue
        events = parse_label_events(label)
        for ev in events:
            phase_counts[ev["phase"]] += 1
            task_counts[ev["task"][0]] += 1
            direction_counts[ev["direction"]] += 1

        sides = label.split(" vs ")
        if len(sides) == 2:
            left_events = parse_label_events(sides[0])
            right_events = parse_label_events(sides[1])
            if left_events and right_events:
                left_phase = left_events[0]["phase"]
                right_phase = right_events[0]["phase"]
                combo = "-".join(sorted([left_phase, right_phase]))
                combo_phase_counts[combo] += 1
                left_task = left_events[0]["task"][0]
                right_task = right_events[0]["task"][0]
                combo_task = "-".join(sorted([left_task, right_task]))
                task_combo_counts[combo_task] += 1

        if any(ev["direction"] == "backw" for ev in events):
            label_direction_presence["labels_with_backw"] += 1
        if any(ev["direction"] == "forw" for ev in events):
            label_direction_presence["labels_with_forw"] += 1

    return {
        "phase_counts": phase_counts,
        "task_counts": task_counts,
        "task_combo_counts": task_combo_counts,
        "direction_counts": direction_counts,
        "phase_combo_counts": combo_phase_counts,
        "label_direction_presence": label_direction_presence,
    }


def sort_counter(counter_obj):
    return collections.OrderedDict(
        sorted(counter_obj.items(), key=lambda kv: kv[1], reverse=True)
    )


def analysis_string_from_labels(top_labels):
    summary = summarize_top_labels(top_labels)
    return (
        f"phase={dict(sort_counter(summary['phase_counts']))} | "
        f"task={dict(sort_counter(summary['task_counts']))} | "
        f"task_combo={dict(sort_counter(summary['task_combo_counts']))} | "
        f"direction={dict(sort_counter(summary['direction_counts']))} | "
        f"phase_combo={dict(sort_counter(summary['phase_combo_counts']))} | "
        f"label_dir={dict(sort_counter(summary['label_direction_presence']))}"
    )


def compute_contributions(model_vec, data_vec):
    mask = np.isfinite(model_vec) & np.isfinite(data_vec)
    if mask.sum() == 0:
        return None, mask
    m = model_vec[mask]
    d = data_vec[mask]
    m_std = np.std(m)
    d_std = np.std(d)
    if m_std == 0 or d_std == 0:
        return None, mask
    m_z = (m - np.mean(m)) / m_std
    d_z = (d - np.mean(d)) / d_std
    contrib = np.full_like(model_vec, np.nan, dtype=float)
    contrib[mask] = m_z * d_z
    return contrib, mask


def orthogonalize(A, B):
    A = np.asarray(A, float)
    B = np.asarray(B, float)

    norm = np.linalg.norm(B)
    if norm == 0:
        raise ValueError("B must be non-zero")

    B_hat = B / norm
    return A - np.dot(A, B_hat) * B_hat


def select_top_positive_contrib(
    contrib,
    min_abs=1e-6,
    min_cum=0.30,
    fallback_cum=0.80,
    min_top_n=10,
    smooth_window=25,
    max_points=200,
):
    pos_idx = np.where(np.isfinite(contrib) & (contrib > 0))[0]
    if pos_idx.size == 0:
        return np.array([], dtype=int)

    pos_vals = contrib[pos_idx]
    order = np.argsort(pos_vals)[::-1]
    vals = pos_vals[order]

    if smooth_window > 1 and vals.size >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        vals_s = np.convolve(vals, kernel, mode="same")
    else:
        vals_s = vals

    x = np.arange(vals_s.size)
    y = vals_s
    if y.size > 1 and y.max() != y.min():
        x0, y0 = x[0], y[0]
        x1, y1 = x[-1], y[-1]
        denom = np.hypot(x1 - x0, y1 - y0)
        if denom == 0:
            d = np.zeros_like(y)
        else:
            d = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
        knee_idx = int(np.argmax(d))
    else:
        knee_idx = min(vals.size - 1, min_top_n - 1)

    if vals[knee_idx] < min_abs:
        knee_idx = min_top_n - 1

    cum = np.cumsum(vals)
    total = cum[-1]
    knee_cum = cum[knee_idx] / total if total > 0 else 0

    if knee_cum < min_cum:
        cutoff = fallback_cum * total
        knee_idx = int(np.searchsorted(cum, cutoff, side="left"))

    knee_idx = max(knee_idx, min_top_n - 1)
    knee_idx = min(knee_idx, vals.size - 1, max_points - 1)

    return pos_idx[order[: knee_idx + 1]]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare model RDMs and relate data to orthogonalized DSR."
    )
    parser.add_argument(
        "--data-npy",
        default=(
            "data/derivatives/group/RDM_plots/"
            "vox_33_81_25_data_RDM_DSR_rew-vs-path_stepwise_combos_glmbase_"
            "all-paths-fixed_stickrews_split-buttons.npy"
        ),
        help="Path to subject-by-RDM numpy array (.npy).",
    )
    parser.add_argument(
        "--config",
        default="rsa_config_DSR_rew_vs_path_stepwise_combos.json",
        help="RSA config file (in condition_files).",
    )
    parser.add_argument(
        "--model-a",
        default="location",
        help="First model RDM name (e.g., location).",
    )
    parser.add_argument(
        "--model-b",
        default="DSR",
        help="Second model RDM name (e.g., DSR).",
    )
    parser.add_argument(
        "--subjects",
        default="",
        help="Comma-separated subject IDs (e.g., sub-01,sub-02).",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory for plots (defaults next to data-npy).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
    if os.path.isdir(source_dir):
        config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
    else:
        source_dir = "/home/fs0/xpsy1114/scratch"
        config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"

    data_npy_path = args.data_npy
    if not os.path.isabs(data_npy_path):
        data_npy_path = os.path.join(source_dir, data_npy_path)

    with open(os.path.join(config_path, args.config), "r") as f:
        config = json.load(f)

    EV_string = config.get("load_EVs_from")
    regression_version = config.get("regression_version")
    include_diagonal = config.get("diagonal_included", True)
    conditions = config.get("EV_condition_selection")
    parts_to_use = conditions["parts"]

    data_rdms = np.load(data_npy_path, allow_pickle=True)
    if data_rdms.ndim == 1:
        data_rdms = data_rdms[None, :]

    base_name = os.path.basename(data_npy_path)
    voxel_tag = base_name.split("_data_RDM_")[0]

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = [f'sub-{i:02}' for i in range(1,15)]
        # subjects = [f"sub-{i + 1:02d}" for i in range(data_rdms.shape[0])]

    if data_rdms.shape[0] == 0:
        raise ValueError("No data rows found in the data RDM file.")

    subj_indices = []
    for s in subjects:
        try:
            subj_no = int(s.split("-")[-1])
        except ValueError as exc:
            raise ValueError(f"Cannot parse subject id from '{s}'") from exc
        subj_indices.append(subj_no - 1)

    if max(subj_indices) >= data_rdms.shape[0]:
        raise ValueError(
            f"Requested subject index {max(subj_indices)} exceeds data rows "
            f"({data_rdms.shape[0]})."
        )

    data_rdms = np.stack([data_rdms[i] for i in subj_indices], axis=0)

    out_dir = args.out_dir
    if not out_dir:
        out_dir = os.path.dirname(data_npy_path)
    os.makedirs(out_dir, exist_ok=True)

    first_data_dir = os.path.join(source_dir, "data/derivatives", subjects[0])
    data_EVs, all_EV_keys = mc.analyse.my_RSA.load_data_EVs(
        first_data_dir,
        regression_version=regression_version,
        only_load_labels=True,
    )
    EV_keys = build_ev_keys(all_EV_keys, parts_to_use)
    _, _, paired_labels = pair_correct_tasks(data_EVs, EV_keys)
    point_labels = build_point_labels(paired_labels, include_diagonal)
    block_size = 8
    n_cond = len(paired_labels)

    def apply_block_lines(ax):
        for b in range(block_size, n_cond, block_size):
            ax.axhline(b - 0.5, color="white", lw=1.0)
            ax.axvline(b - 0.5, color="white", lw=1.0)
        ax.set_ylabel("Blocks", fontsize=12)

    for sub, data_vec in zip(subjects, data_rdms):
        data_dir = os.path.join(source_dir, "data/derivatives", sub)
        modelled_conditions_dir = os.path.join(data_dir, "beh/modelled_EVs")

        with open(
            os.path.join(
                modelled_conditions_dir, f"{sub}_modelled_EVs_{EV_string}.pkl"
            ),
            "rb",
        ) as file:
            model_EVs = pickle.load(file)

        if args.model_a not in model_EVs or args.model_b not in model_EVs:
            print(f"{sub}: missing model(s) {args.model_a}/{args.model_b}")
            continue

        model_a_th1, model_a_th2, _ = pair_correct_tasks(model_EVs[args.model_a], EV_keys)
        model_b_th1, model_b_th2, _ = pair_correct_tasks(model_EVs[args.model_b], EV_keys)

        model_a_concat = np.concatenate((model_a_th1, model_a_th2), axis=0)
        model_b_concat = np.concatenate((model_b_th1, model_b_th2), axis=0)

        model_a_vec = np.asarray(
            mc.analyse.my_RSA.compute_crosscorr(
                model_a_concat, plotting=False, include_diagonal=include_diagonal
            )[0]
        )
        model_b_vec = np.asarray(
            mc.analyse.my_RSA.compute_crosscorr(
                model_b_concat, plotting=False, include_diagonal=include_diagonal
            )[0]
        )
        mm_mask = np.isfinite(model_a_vec) & np.isfinite(model_b_vec)
        mm_corr = (
            np.corrcoef(model_a_vec[mm_mask], model_b_vec[mm_mask])[0, 1]
            if mm_mask.any()
            else np.nan
        )

        contrib_mm, _ = compute_contributions(model_a_vec, model_b_vec)
        if contrib_mm is None:
            print(f"{sub}: cannot compute model-model contributions.")
            continue
        top_pos_idx_mm = select_top_positive_contrib(contrib_mm)
        labels_mm = [
            point_labels[i] if i < len(point_labels) else "" for i in top_pos_idx_mm
        ]
        analysis_mm = analysis_string_from_labels(labels_mm)

        n = len(paired_labels)
        k = 0 if include_diagonal else 1
        iu = np.triu_indices(n, k=k)
        lower_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)

        contrib_mm_mat = np.full((n, n), np.nan, dtype=float)
        contrib_mm_mat[iu] = 0.0
        contrib_mm_mat[iu[0][top_pos_idx_mm], iu[1][top_pos_idx_mm]] = 1.0
        contrib_mm_mat = np.where(np.isnan(contrib_mm_mat), contrib_mm_mat.T, contrib_mm_mat)

        model_a_mat = np.full((n, n), np.nan, dtype=float)
        model_a_mat[iu] = model_a_vec
        model_a_mat = np.where(np.isnan(model_a_mat), model_a_mat.T, model_a_mat)
        model_b_mat = np.full((n, n), np.nan, dtype=float)
        model_b_mat[iu] = model_b_vec
        model_b_mat = np.where(np.isnan(model_b_mat), model_b_mat.T, model_b_mat)

        contrib_mm_upper = np.ma.array(contrib_mm_mat, mask=lower_mask)
        model_a_upper = np.ma.array(model_a_mat, mask=lower_mask)
        model_b_upper = np.ma.array(model_b_mat, mask=lower_mask)

        model_vals = np.concatenate(
            [model_a_vec[np.isfinite(model_a_vec)], model_b_vec[np.isfinite(model_b_vec)]]
        )
        rdm_vmin = float(np.min(model_vals)) if model_vals.size else 0.0
        rdm_vmax = float(np.max(model_vals)) if model_vals.size else 2.0
        norm_rdm = mpl.colors.TwoSlopeNorm(vmin=rdm_vmin, vcenter=1.0, vmax=rdm_vmax)
        cmap_rdm = plt.get_cmap("RdBu").copy()
        cmap_rdm.set_bad("white")

        cmap_contrib = mpl.colors.ListedColormap(["white", "#d7301f"])
        cmap_contrib.set_bad("white")
        norm_contrib = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap_contrib.N)

        fig_mm, axes_mm = plt.subplots(1, 3, figsize=(18, 6))

        axes_mm[0].imshow(
            contrib_mm_upper,
            cmap=cmap_contrib,
            norm=norm_contrib,
            interpolation="None",
            aspect="equal",
        )
        axes_mm[0].set_title(
            f"{sub} {voxel_tag}: top {len(top_pos_idx_mm)} contrib (model-model)\n"
            f"r={mm_corr:.3f}",
            fontsize=14,
        )

        im_a = axes_mm[1].imshow(
            model_a_upper,
            cmap=cmap_rdm,
            norm=norm_rdm,
            interpolation="None",
            aspect="equal",
        )
        axes_mm[1].set_title(
            f"{sub} {voxel_tag}: {args.model_a} (min={rdm_vmin:.3g}, max={rdm_vmax:.3g})",
            fontsize=14,
        )

        im_b = axes_mm[2].imshow(
            model_b_upper,
            cmap=cmap_rdm,
            norm=norm_rdm,
            interpolation="None",
            aspect="equal",
        )
        axes_mm[2].set_title(
            f"{sub} {voxel_tag}: {args.model_b} (min={rdm_vmin:.3g}, max={rdm_vmax:.3g})",
            fontsize=14,
        )

        for ax in axes_mm:
            ax.set_xticks([])
            ax.set_yticks([])
            apply_block_lines(ax)
            for idx in top_pos_idx_mm:
                i, j = iu[0][idx], iu[1][idx]
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#d7301f",
                        linewidth=0.8,
                    )
                )

        fig_mm.tight_layout()
        fig_mm.text(
            0.5,
            0.01,
            textwrap.fill(analysis_mm, width=160),
            ha="center",
            va="bottom",
            fontsize=11,
        )
        fig_mm.savefig(
            os.path.join(out_dir, f"{sub}_model_model_contrib_{args.model_a}-{args.model_b}.png"),
            dpi=200,
        )
        # plt.close(fig_mm)

        # Orthogonalize model B against model A
        ortho_b_vec = orthogonalize(model_b_vec, model_a_vec)
        ortho_b_mat = np.full((n, n), np.nan, dtype=float)
        ortho_b_mat[iu] = ortho_b_vec
        ortho_b_mat = np.where(np.isnan(ortho_b_mat), ortho_b_mat.T, ortho_b_mat)
        ortho_b_upper = np.ma.array(ortho_b_mat, mask=lower_mask)

        ortho_vals = ortho_b_vec[np.isfinite(ortho_b_vec)]
        ortho_vmin = float(np.min(ortho_vals)) if ortho_vals.size else -1.0
        ortho_vmax = float(np.max(ortho_vals)) if ortho_vals.size else 1.0
        norm_ortho = mpl.colors.TwoSlopeNorm(
            vmin=ortho_vmin, vcenter=0.0, vmax=ortho_vmax
        )

        fig_ortho, ax_ortho = plt.subplots(1, 1, figsize=(6, 6))
        ax_ortho.imshow(
            ortho_b_upper,
            cmap=cmap_rdm,
            norm=norm_ortho,
            interpolation="None",
            aspect="equal",
        )
        ax_ortho.set_title(
            f"{sub} {voxel_tag}: ortho {args.model_b} (min={ortho_vmin:.3g}, max={ortho_vmax:.3g})",
            fontsize=14,
        )
        ax_ortho.set_xticks([])
        ax_ortho.set_yticks([])
        apply_block_lines(ax_ortho)
        fig_ortho.tight_layout()
        fig_ortho.savefig(
            os.path.join(out_dir, f"{sub}_ortho_{args.model_b}_by_{args.model_a}.png"),
            dpi=200,
        )
        # plt.close(fig_ortho)

        # Data vs orthogonalized model
        contrib_do, _ = compute_contributions(ortho_b_vec, data_vec)
        if contrib_do is None:
            print(f"{sub}: cannot compute data-ortho contributions.")
            continue
        top_pos_idx_do = select_top_positive_contrib(contrib_do)
        labels_do = [
            point_labels[i] if i < len(point_labels) else "" for i in top_pos_idx_do
        ]
        analysis_do = analysis_string_from_labels(labels_do)
        do_mask = np.isfinite(ortho_b_vec) & np.isfinite(data_vec)
        do_corr = (
            np.corrcoef(ortho_b_vec[do_mask], data_vec[do_mask])[0, 1]
            if do_mask.any()
            else np.nan
        )
        # Print top 10 condition pairs for ortho DSR vs data (Pearson r contributions)
        top_do_order = np.argsort(contrib_do[top_pos_idx_do])[::-1]
        top_do_idx = top_pos_idx_do[top_do_order][:10]
        print(f"{sub}: top {len(top_do_idx)} ortho-{args.model_b} vs data contributors")
        for rank, idx in enumerate(top_do_idx, start=1):
            label = point_labels[idx] if idx < len(point_labels) else ""
            print(f"{rank:03d} idx={idx:04d} contrib={contrib_do[idx]: .4f} label={label}")

        contrib_do_mat = np.full((n, n), np.nan, dtype=float)
        contrib_do_mat[iu] = 0.0
        contrib_do_mat[iu[0][top_pos_idx_do], iu[1][top_pos_idx_do]] = 1.0
        contrib_do_mat = np.where(np.isnan(contrib_do_mat), contrib_do_mat.T, contrib_do_mat)
        contrib_do_upper = np.ma.array(contrib_do_mat, mask=lower_mask)

        data_mat = np.full((n, n), np.nan, dtype=float)
        data_mat[iu] = data_vec
        data_mat = np.where(np.isnan(data_mat), data_mat.T, data_mat)
        data_upper = np.ma.array(data_mat, mask=lower_mask)

        data_vals = data_vec[np.isfinite(data_vec)]
        data_vmin = float(np.min(data_vals)) if data_vals.size else 0.0
        data_vmax = float(np.max(data_vals)) if data_vals.size else 2.0
        norm_data = mpl.colors.TwoSlopeNorm(vmin=data_vmin, vcenter=1.0, vmax=data_vmax)

        fig_do, axes_do = plt.subplots(1, 3, figsize=(18, 6))

        axes_do[0].imshow(
            contrib_do_upper,
            cmap=cmap_contrib,
            norm=norm_contrib,
            interpolation="None",
            aspect="equal",
        )
        axes_do[0].set_title(
            f"{sub} {voxel_tag}: top {len(top_pos_idx_do)} contrib (data vs ortho)\n"
            f"r={do_corr:.3f}",
            fontsize=14,
        )

        axes_do[1].imshow(
            ortho_b_upper,
            cmap=cmap_rdm,
            norm=norm_ortho,
            interpolation="None",
            aspect="equal",
        )
        axes_do[1].set_title(
            f"{sub} {voxel_tag}: ortho {args.model_b}",
            fontsize=14,
        )

        axes_do[2].imshow(
            data_upper,
            cmap=cmap_rdm,
            norm=norm_data,
            interpolation="None",
            aspect="equal",
        )
        axes_do[2].set_title(
            f"{sub} {voxel_tag}: data RDM (min={data_vmin:.3g}, max={data_vmax:.3g})",
            fontsize=14,
        )

        for ax in axes_do:
            ax.set_xticks([])
            ax.set_yticks([])
            apply_block_lines(ax)
            for idx in top_pos_idx_do:
                i, j = iu[0][idx], iu[1][idx]
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#d7301f",
                        linewidth=0.8,
                    )
                )

        fig_do.tight_layout()
        fig_do.text(
            0.5,
            0.01,
            textwrap.fill(analysis_do, width=160),
            ha="center",
            va="bottom",
            fontsize=11,
        )
        fig_do.savefig(
            os.path.join(out_dir, f"{sub}_data_vs_ortho_{args.model_b}.png"),
            dpi=200,
        )
        #plt.close(fig_do)


if __name__ == "__main__":
    main()
