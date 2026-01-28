#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group-level analysis: how datapoints contribute to explaining the DSR model
(after orthogonalization vs location, and optionally vs path_rew) across subjects.
"""

import argparse
import json
import os
import pickle
from fnmatch import fnmatch

import numpy as np
import matplotlib.pyplot as plt

import mc


def _to_numeric_array(arr, mapping=None):
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float), mapping
    flat = arr.ravel()
    if mapping is None:
        uniques = sorted(set(flat.tolist()))
        mapping = {v: i for i, v in enumerate(uniques)}
    mapped = np.vectorize(mapping.get)(arr)
    return mapped.astype(float), mapping


def pair_correct_tasks(data_dict, keys_list):
    task_pairs = {"1_forw": "2_backw", "1_backw": "2_forw"}
    th_1, th_2, paired_list_control = [], [], []
    for key in keys_list:
        assert key in data_dict, "Missmatch between model rdm keys and data RDM keys"
        task, direction, state, phase = key.split("_")
        pair_suffix = task_pairs.get(f"{task[-1]}_{direction}")
        pair_key = f"{task[0]}{pair_suffix}_{state}_{phase}"
        if pair_key in data_dict:
            try:
                v1, mapping = _to_numeric_array(data_dict[key], mapping=None)
                v2, _ = _to_numeric_array(data_dict[pair_key], mapping=mapping)
                th_1.append(v1)
                th_2.append(v2)
            except Exception as exc:
                raise ValueError(
                    f"Non-numeric EV values for {key} or {pair_key}"
                ) from exc
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


def compute_contributions(model_vec, data_vec):
    mask = np.isfinite(model_vec) & np.isfinite(data_vec)
    if mask.sum() == 0:
        return None
    m = model_vec[mask]
    d = data_vec[mask]
    m_std = np.std(m)
    d_std = np.std(d)
    if m_std == 0 or d_std == 0:
        return None
    m_z = (m - np.mean(m)) / m_std
    d_z = (d - np.mean(d)) / d_std
    contrib = np.full_like(model_vec, np.nan, dtype=float)
    contrib[mask] = m_z * d_z
    return contrib


def orthogonalize(A, B):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    norm = np.linalg.norm(B)
    if norm == 0:
        raise ValueError("B must be non-zero")
    B_hat = B / norm
    return A - np.dot(A, B_hat) * B_hat


def parse_args():
    parser = argparse.ArgumentParser(
        description="Group contribution analysis for orthogonalized DSR model."
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
        "--model",
        default="DSR",
        help="Target model (DSR).",
    )
    parser.add_argument(
        "--orthos",
        default="location,path_rew",
        help="Comma-separated orthogonalizers (e.g., location,path_rew).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k cells for per-subject fraction of contribution.",
    )
    parser.add_argument(
        "--top-n-cells",
        type=int,
        default=200,
        help="Top-N cells (by group mean |contribution|) for heatmap.",
    )
    parser.add_argument(
        "--subjects",
        default="",
        help="Comma-separated subject IDs (e.g., sub-01,sub-02).",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory (defaults next to data-npy).",
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

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = [f"sub-{i + 1:02d}" for i in range(data_rdms.shape[0])]
        subjects = [f"sub-{i + 1:02d}" for i in range(0,28)]
    exclude_subs = {"sub-10", "sub-21"}
    subjects = [s for s in subjects if s not in exclude_subs]

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

    orthos = ["location"]

    for ortho_name in orthos:
        all_contribs = []
        used_subjects = []
        subject_stats = []
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

            if args.model not in model_EVs or ortho_name not in model_EVs:
                print(f"{sub}: missing model(s) {args.model}/{ortho_name}")
                continue

            model_t_th1, model_t_th2, _ = pair_correct_tasks(model_EVs[args.model], EV_keys)
            model_o_th1, model_o_th2, _ = pair_correct_tasks(model_EVs[ortho_name], EV_keys)

            model_t_concat = np.concatenate((model_t_th1, model_t_th2), axis=0)
            model_o_concat = np.concatenate((model_o_th1, model_o_th2), axis=0)

            model_t_vec = np.asarray(
                mc.analyse.my_RSA.compute_crosscorr(
                    model_t_concat, plotting=False, include_diagonal=include_diagonal
                )[0]
            )
            model_o_vec = np.asarray(
                mc.analyse.my_RSA.compute_crosscorr(
                    model_o_concat, plotting=False, include_diagonal=include_diagonal
                )[0]
            )

            ortho_vec = orthogonalize(model_t_vec, model_o_vec)
            contrib = compute_contributions(ortho_vec, data_vec)
            if contrib is None:
                print(f"{sub}: invalid contributions for {ortho_name}")
                continue

            mask = np.isfinite(ortho_vec) & np.isfinite(data_vec)
            if mask.any():
                r_sub = np.corrcoef(ortho_vec[mask], data_vec[mask])[0, 1]
            else:
                r_sub = np.nan
            pos_total = np.nansum(contrib[contrib > 0])
            vals = contrib[np.isfinite(contrib)]
            if vals.size:
                order = np.argsort(np.abs(vals))[::-1]
                top_k = min(args.top_k, vals.size)
                frac_top_k = (
                    np.nansum(np.abs(vals[order[:top_k]])) / np.nansum(np.abs(vals))
                    if np.nansum(np.abs(vals)) > 0
                    else np.nan
                )
            else:
                frac_top_k = np.nan

            subject_stats.append((sub, r_sub, pos_total, frac_top_k))
            all_contribs.append(contrib)
            used_subjects.append(sub)

        if not all_contribs:
            print(f"No contributions computed for orthogonalizer {ortho_name}")
            continue

        all_contribs = np.stack(all_contribs, axis=0)
        mean_contrib = np.nanmean(all_contribs, axis=0)
        std_contrib = np.nanstd(all_contribs, axis=0)
        n_sub = all_contribs.shape[0]
        n_pos = np.sum(all_contribs > 0, axis=0)
        n_neg = np.sum(all_contribs < 0, axis=0)

        summary_path = os.path.join(
            out_dir, f"summary_group_contrib_{args.model}_ortho_{ortho_name}.csv"
        )
        with open(summary_path, "w") as f:
            f.write("index,label,mean_contrib,std_contrib,n_pos,n_neg,n_sub\n")
            for i in range(len(mean_contrib)):
                label = point_labels[i] if i < len(point_labels) else ""
                f.write(
                    f"{i},{label},{mean_contrib[i]:.6f},{std_contrib[i]:.6f},"
                    f"{n_pos[i]},{n_neg[i]},{n_sub}\n"
                )

        # also save a compact npy for further stats
        np.save(
            os.path.join(
                out_dir, f"group_contrib_{args.model}_ortho_{ortho_name}.npy"
            ),
            all_contribs,
        )

        # Per-subject stats plots
        if subject_stats:
            r_vals = np.array([s[1] for s in subject_stats], dtype=float)
            frac_vals = np.array([s[3] for s in subject_stats], dtype=float)

            fig_r, ax_r = plt.subplots(1, 1, figsize=(5, 4))
            ax_r.hist(r_vals[np.isfinite(r_vals)], bins=15, color="#4c78a8")
            ax_r.set_title(f"{ortho_name}: r_sub histogram", fontsize=12)
            ax_r.set_xlabel("r_sub", fontsize=10)
            ax_r.set_ylabel("Count", fontsize=10)
            fig_r.tight_layout()
            fig_r.savefig(
                os.path.join(out_dir, f"hist_r_sub_{args.model}_ortho_{ortho_name}.png"),
                dpi=200,
            )
            #plt.close(fig_r)

            fig_sc, ax_sc = plt.subplots(1, 1, figsize=(5, 4))
            ax_sc.scatter(r_vals, frac_vals, s=20, color="#f58518", alpha=0.7)
            ax_sc.set_title(
                f"{ortho_name}: r_sub vs frac_top_{args.top_k}", fontsize=12
            )
            ax_sc.set_xlabel("r_sub", fontsize=10)
            ax_sc.set_ylabel(f"frac_top_{args.top_k}", fontsize=10)
            fig_sc.tight_layout()
            fig_sc.savefig(
                os.path.join(
                    out_dir,
                    f"scatter_r_vs_frac_top{args.top_k}_{args.model}_ortho_{ortho_name}.png",
                ),
                dpi=200,
            )
            #plt.close(fig_sc)

        # Subject x cell heatmap for top-N by group mean |contrib|
        top_n = min(args.top_n_cells, mean_contrib.size)
        order = np.argsort(np.abs(mean_contrib))[::-1][:top_n]
        heat = all_contribs[:, order]
        fig_h, ax_h = plt.subplots(1, 1, figsize=(8, 6))
        im = ax_h.imshow(heat, aspect="auto", cmap="coolwarm", interpolation="None")
        ax_h.set_title(
            f"{ortho_name}: subjects x top-{top_n} cells (|mean contrib|)",
            fontsize=12,
        )
        ax_h.set_xlabel("Cells (sorted)", fontsize=10)
        ax_h.set_ylabel("Subjects", fontsize=10)
        fig_h.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
        fig_h.tight_layout()
        fig_h.savefig(
            os.path.join(
                out_dir,
                f"heat_subjects_x_cells_{args.model}_ortho_{ortho_name}.png",
            ),
            dpi=200,
        )
        #plt.close(fig_h)
        import pdb; pdb.set_trace()
        print(
            f"Saved group summary for ortho {ortho_name}: {summary_path}"
        )


if __name__ == "__main__":
    main()
