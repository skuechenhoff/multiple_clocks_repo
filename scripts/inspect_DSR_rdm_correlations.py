#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect which parts of the RDM drive correlations by comparing subject data RDMs
against model RDMs (e.g., DSR) and plotting the RDM vectors as lines.
"""

import argparse
import csv
import collections
import textwrap
import json
import os
import pickle
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


def parse_paired_label(lbl):
    if " | " in lbl:
        left, right = lbl.split(" | ")
    else:
        left, right = lbl.split(" with ")
    l = left.split("_")
    r = right.split("_")
    arrow = {"backw": "<-", "forw": "->"}
    block = f"{l[0]}{arrow.get(l[1], '')}|{r[0]}{arrow.get(r[1], '')}"
    within = f"{l[2]}-{l[3]}".replace("reward", "rew")
    return block, within


def parse_label_events(label):
    parts = label.split(" vs ")
    events = []
    for side in parts:
        if " | " in side:
            halves = side.split(" | ")
        else:
            halves = side.split(" with ")
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
        norm_label = normalize_label_for_analysis(label)
        events = parse_label_events(norm_label)
        for ev in events:
            phase_counts[ev["phase"]] += 1
            task_counts[ev["task"][0]] += 1
            direction_counts[ev["direction"]] += 1

        # phase combo across the two sides (left vs right)
        sides = norm_label.split(" vs ")
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

        # whether any backw/forw appears in label
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


def normalize_label_for_analysis(label):
    return label


def select_top_positive_contrib(
    contrib,
    min_abs=1e-6,
    min_cum=0.30,
    fallback_cum=0.80,
    min_top_n=10,
    smooth_window=25,
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
    knee_idx = min(knee_idx, vals.size - 1, 200 - 1)

    return pos_idx[order[: knee_idx + 1]]


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Correlate subject data RDMs with model RDMs and plot vectors."
    )
    parser.add_argument(
        "--data-npy",
        default=(
            "data/derivatives/group/RDM_plots/"
            "vox_33_19_48_data_RDM_DSR_rew-vs-path_stepwise_combos_glmbase_"
            "all-paths-fixed_stickrews_split-buttons.npy"
        ),
        help="Path to the subject-by-RDM numpy array (.npy).",
    )
    # "vox_33_81_25_data_RDM_DSR_rew-vs-path_stepwise_combos_glmbase_"
        
    parser.add_argument(
        "--config",
        default="rsa_config_DSR_rew_vs_path_stepwise_combos.json",
        help="RSA config file (in condition_files).",
    )
    parser.add_argument(
        "--model",
        default="DSR",
        help="Model RDM of interest for correlation.",
    )
    parser.add_argument(
        "--subjects",
        default="",
        help="Comma-separated subject IDs (e.g., sub-01,sub-02).",
    )
    parser.add_argument(
        "--plot-all-models",
        action="store_true",
        help="Plot all model RDMs (not just the model of interest).",
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
        subjects = [f'sub-{i:02}' for i in range(1, 16)]
        # subjects = ["sub-01", "sub-02", "sub-03"]

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

    # Build labels once (from first subject)
    first_data_dir = os.path.join(source_dir, "data/derivatives", subjects[0])
    data_EVs, all_EV_keys = mc.analyse.my_RSA.load_data_EVs(
        first_data_dir,
        regression_version=regression_version,
        only_load_labels=True,
    )
    EV_keys = build_ev_keys(all_EV_keys, parts_to_use)
    _, _, paired_labels = pair_correct_tasks(data_EVs, EV_keys)
    point_labels = build_point_labels(paired_labels, include_diagonal)

    summary_rows = []

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

        selected_models = config.get("models", list(model_EVs.keys()))
        if args.model not in selected_models and args.model in model_EVs:
            selected_models.append(args.model)

        model_RDM_dir = {}
        for model in selected_models:
            if model not in model_EVs:
                continue
            model_th1, model_th2, _ = pair_correct_tasks(model_EVs[model], EV_keys)
            model_concat = np.concatenate((model_th1, model_th2), axis=0)
            if model == "path_rew":
                model_RDM_dir[model] = mc.analyse.my_RSA.make_categorical_RDM(
                    model_concat, plotting=False, include_diagonal=include_diagonal
                )
            elif model == "duration":
                model_RDM_dir[model] = mc.analyse.my_RSA.make_distance_RDM(
                    model_concat, plotting=False, include_diagonal=include_diagonal
                )
            else:
                model_RDM_dir[model] = mc.analyse.my_RSA.compute_crosscorr(
                    model_concat, plotting=False, include_diagonal=include_diagonal
                )

        if args.model not in model_RDM_dir:
            print(f"{sub}: model '{args.model}' not found; skipping correlation.")
            continue

        model_vec = np.asarray(model_RDM_dir[args.model][0])
        data_vec = np.asarray(data_vec)
        if model_vec.shape[0] != data_vec.shape[0]:
            print(
                f"{sub}: length mismatch model={model_vec.shape[0]} "
                f"data={data_vec.shape[0]}"
            )
            continue

        mask = np.isfinite(model_vec) & np.isfinite(data_vec)
        if mask.sum() == 0:
            print(f"{sub}: no finite values to correlate.")
            continue
        corr = np.corrcoef(model_vec[mask], data_vec[mask])[0, 1]
        print(f"{sub}: {args.model} vs data r = {corr:.4f} (n={mask.sum()})")

        contrib, contrib_mask = compute_contributions(model_vec, data_vec)
        if contrib is None:
            print(f"{sub}: cannot compute contributions (zero variance).")
            continue

        top_print_n = 50
        top_pos_idx = select_top_positive_contrib(
            contrib,
            min_abs=1e-6,
            min_cum=0.30,
            fallback_cum=0.80,
            min_top_n=10,
            smooth_window=25,
        )
        if top_pos_idx.size == 0:
            print(f"{sub}: no positive contributions to select.")
            continue
        top_idx = top_pos_idx

        for rank, idx in enumerate(top_idx[:top_print_n], start=1):
            label = point_labels[idx] if idx < len(point_labels) else ""
            summary_rows.append(
                [
                    sub,
                    rank,
                    idx,
                    label,
                    float(model_vec[idx]),
                    float(data_vec[idx]),
                    float(contrib[idx]),
                ]
            )

        print(
            f"{sub}: top {top_print_n} of {len(top_pos_idx)} positive contributors "
            f"(knee+guardrails, {args.model})"
        )
        for rank, idx in enumerate(top_idx[:top_print_n], start=1):
            label = point_labels[idx] if idx < len(point_labels) else ""
            print(
                f"{rank:03d} idx={idx:04d} contrib={contrib[idx]: .4f} "
                f"model={model_vec[idx]: .4f} data={data_vec[idx]: .4f} "
                f"label={label}"
            )

        # Label analysis for top 200 by absolute contribution
        top_labels = [
            point_labels[i] if i < len(point_labels) else "" for i in top_pos_idx
        ]
        summary = summarize_top_labels(top_labels)

        print(
            f"{sub}: label analysis for top {len(top_pos_idx)} positive contributions"
        )
        print("phase counts:", dict(sort_counter(summary["phase_counts"])))
        print("task counts (letter only):", dict(sort_counter(summary["task_counts"])))
        print("task combo counts:", dict(sort_counter(summary["task_combo_counts"])))
        print("direction counts:", dict(sort_counter(summary["direction_counts"])))
        print("phase combo counts:", dict(sort_counter(summary["phase_combo_counts"])))
        print(
            "label direction presence:",
            dict(sort_counter(summary["label_direction_presence"])),
        )
        analysis_str = (
            f"phase={dict(sort_counter(summary['phase_counts']))} | "
            f"task={dict(sort_counter(summary['task_counts']))} | "
            f"task_combo={dict(sort_counter(summary['task_combo_counts']))} | "
            f"direction={dict(sort_counter(summary['direction_counts']))} | "
            f"phase_combo={dict(sort_counter(summary['phase_combo_counts']))} | "
            f"label_dir={dict(sort_counter(summary['label_direction_presence']))}"
        )

        # Visual map of top positive contributions
        n = len(paired_labels)
        k = 0 if include_diagonal else 1
        iu = np.triu_indices(n, k=k)
        contrib_mat = np.full((n, n), np.nan, dtype=float)
        contrib_mat[iu] = 0.0
        contrib_mat[iu[0][top_pos_idx], iu[1][top_pos_idx]] = 1.0
        contrib_mat = np.where(np.isnan(contrib_mat), contrib_mat.T, contrib_mat)

        # block size from paired labels (same logic as store_group_RDM.py)
        first_block_str = paired_labels[0].split("_")[0] + "_" + paired_labels[0].split("_")[1]
        block_size = None
        for i, l in enumerate(paired_labels):
            if first_block_str in l:
                block_size = i + 1
        if block_size is None:
            block_size = n

        parsed = [parse_paired_label(l) for l in paired_labels]
        block_labels = [parsed[i][0] for i in range(0, n, block_size)]
        centers = np.arange(block_size / 2 - 0.5, n, block_size)
        within = [parsed[i][1] for i in range(0, block_size)]

        cmap_contrib = mpl.colors.ListedColormap(["#4575b4", "white", "#d7301f"])
        cmap_contrib.set_bad("white")
        norm_contrib = mpl.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap_contrib.N)

        def apply_block_labels(ax):
            for b in range(block_size, n, block_size):
                ax.axhline(b - 0.5, color="white", lw=1.0)
                ax.axvline(b - 0.5, color="white", lw=1.0)
            ax.set_xticks([])
            ax.set_yticks(centers)
            ax.set_yticklabels(block_labels, fontsize=8)
            ax.yaxis.tick_right()
            ax.tick_params(length=0, pad=1, labelsize=8)
            ax.set_xlabel("Within-block: " + " | ".join(within), fontsize=8, labelpad=8)

        # Build model/data RDM matrices
        model_mat = np.full((n, n), np.nan, dtype=float)
        model_mat[iu] = model_vec
        model_mat = np.where(np.isnan(model_mat), model_mat.T, model_mat)
        data_mat = np.full((n, n), np.nan, dtype=float)
        data_mat[iu] = data_vec
        data_mat = np.where(np.isnan(data_mat), data_mat.T, data_mat)

        cmap_rdm = plt.get_cmap("RdBu").copy()
        cmap_rdm.set_bad("white")
        model_vals = model_vec[np.isfinite(model_vec)]
        data_vals = data_vec[np.isfinite(data_vec)]
        if model_vals.size == 0 or data_vals.size == 0:
            rdm_vmin, rdm_vmax = 0.0, 2.0
        else:
            model_vmin, model_vmax = float(np.min(model_vals)), float(np.max(model_vals))
            data_vmin, data_vmax = float(np.min(data_vals)), float(np.max(data_vals))
            rdm_vmin, rdm_vmax = min(model_vmin, data_vmin), max(model_vmax, data_vmax)
        norm_rdm = mpl.colors.TwoSlopeNorm(vmin=rdm_vmin, vcenter=1.0, vmax=rdm_vmax)

        lower_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
        contrib_upper = np.ma.array(contrib_mat, mask=lower_mask)
        model_upper = np.ma.array(model_mat, mask=lower_mask)
        data_upper = np.ma.array(data_mat, mask=lower_mask)

        def draw_contrib_boxes(ax):
            for idx in top_pos_idx:
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
            # negative contributions not plotted

        fig_c, axes_c = plt.subplots(1, 3, figsize=(18, 6))

        axes_c[0].imshow(
            contrib_upper,
            cmap=cmap_contrib,
            norm=norm_contrib,
            interpolation="None",
            aspect="equal",
        )
        axes_c[0].set_title(
            f"{sub} {voxel_tag}: \n top {len(top_pos_idx)} pos contrib (knee)",
            fontsize=15,
        )
        apply_block_labels(axes_c[0])
        draw_contrib_boxes(axes_c[0])

        im_model = axes_c[1].imshow(
            model_upper,
            cmap=cmap_rdm,
            norm=norm_rdm,
            interpolation="None",
            aspect="equal",
        )
        axes_c[1].set_title(
            f"{sub} {voxel_tag}: \n model RDM \n (min={rdm_vmin:.3g}, max={rdm_vmax:.3g})",
            fontsize=15,
        )
        apply_block_labels(axes_c[1])
        draw_contrib_boxes(axes_c[1])
        im_data = axes_c[2].imshow(
            data_upper,
            cmap=cmap_rdm,
            norm=norm_rdm,
            interpolation="None",
            aspect="equal",
        )
        axes_c[2].set_title(
            f"{sub} {voxel_tag}: \n data RDM \n (min={rdm_vmin:.3g}, max={rdm_vmax:.3g})",
            fontsize=15,
        )
        apply_block_labels(axes_c[2])
        draw_contrib_boxes(axes_c[2])
        analysis_wrapped = textwrap.fill(analysis_str, width=160)
        fig_c.text(
            0.5,
            0.01,
            analysis_wrapped,
            ha="center",
            va="bottom",
            fontsize=11,
        )
        map_path = os.path.join(
            out_dir, f"{sub}_toppos{len(top_pos_idx)}_contrib_panels_{args.model}.png"
        )
        fig_c.tight_layout()
        fig_c.savefig(map_path, dpi=200)
        plt.close(fig_c)

        # # Plot
        # fig, ax = plt.subplots(figsize=(22, 6))
        # x = np.arange(len(data_vec))
        # ax.plot(x, data_vec, label="data", linewidth=1.2, color="black")

        # models_to_plot = (
        #     [m for m in selected_models if m in model_RDM_dir]
        #     if args.plot_all_models
        #     else [args.model]
        # )
        # for model in models_to_plot:
        #     ax.plot(x, model_RDM_dir[model][0], label=model, linewidth=0.9)

        # ax.set_title(f"{sub}: RDM vectors (model of interest = {args.model})")
        # ax.set_xlabel("RDM element")
        # ax.set_ylabel("Dissimilarity")
        # ax.legend(fontsize=8, ncol=2)

        # if len(point_labels) == len(data_vec):
        #     ax.set_xticks(x)
        #     ax.set_xticklabels(point_labels, rotation=90, fontsize=4)
        # else:
        #     print(
        #         f"{sub}: label count mismatch labels={len(point_labels)} "
        #         f"data={len(data_vec)} (skipping x labels)"
        #     )

        # fig.tight_layout()
        # out_path = os.path.join(out_dir, f"{sub}_rdm_lineplot_{args.model}.png")
        # fig.savefig(out_path, dpi=200)
        # plt.close(fig)

        # Plot top contributing positive bins
        top_bins_n = min(50, len(top_pos_idx))
        top_pos_bins = top_pos_idx[:top_bins_n]

        fig_t, ax_t = plt.subplots(1, 1, figsize=(22, 6), sharey=True)
        if top_pos_bins.size == 0:
            ax_t.set_title(f"{sub}: top {top_bins_n} positive bins (none)")
            ax_t.axis("off")
        else:
            x_t = np.arange(len(top_pos_bins))
            data_t = data_vec[top_pos_bins]
            model_t = model_vec[top_pos_bins]
            labels_t = [
                point_labels[i] if i < len(point_labels) else "" for i in top_pos_bins
            ]

            ax_t.plot(x_t, data_t, label="data", linewidth=1.2, color="black")
            ax_t.plot(x_t, model_t, label=args.model, linewidth=1.0)
            ax_t.set_title(
                f"{sub}: top {top_bins_n} positive bins (85% cutoff)",
                fontsize=14,
            )
            ax_t.set_xlabel("Selected RDM element", fontsize=12)
            ax_t.set_ylabel("Dissimilarity", fontsize=12)
            ax_t.legend(fontsize=10)
            ax_t.set_xticks(x_t)
            ax_t.set_xticklabels(labels_t, rotation=90, fontsize=8)

        fig_t.tight_layout()
        top_path = os.path.join(
            out_dir, f"{sub}_toppos{top_bins_n}_bins_lineplot_{args.model}.png"
        )
        fig_t.savefig(top_path, dpi=200)
        plt.close(fig_t)

    summary_path = os.path.join(
        out_dir, f"summary_toppos_contrib_{args.model}.csv"
    )
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["subject", "rank", "index", "label", "model_value", "data_value", "contribution"]
        )
        writer.writerows(summary_rows)

    print("done with analysis.")


if __name__ == "__main__":
    main()
