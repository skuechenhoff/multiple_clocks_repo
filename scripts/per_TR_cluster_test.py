#!/usr/bin/env python
"""Cluster-mass permutation test for a per-TR group RSA model.

The test is one-sided (positive) within each supplied anatomical mask. Voxels
are first thresholded at an uncorrected cluster-forming p value. Spatial
clusters are formed separately at every TR, and the largest cluster mass over
all clusters and all TRs is retained for each subject-wise sign flip. Thus the
cluster p values are FWE-corrected jointly across space and the supplied TRs.

Cluster mass is sum(t - cluster-forming t) over the cluster. The identical
``clusters`` function is used for the observed map and every permutation.
Input maps are already searchlight-smoothed; this script adds no smoothing.

This is intended as a documented sensitivity analysis. Choosing cluster
inference after inspecting voxelwise results must not be presented as the
original confirmatory test.
"""
import argparse
import csv
import json
import os

import nibabel as nib
import numpy as np
from scipy import ndimage, stats

import mc.analyse.loso as L


DEFAULT_ROOT = ("/Users/xpsy1114/Documents/projects/multiple_clocks/data/"
                "derivatives/group/per_TR")
DEFAULT_PATTERN = "group_RSA_within_th_only_intr-vs-exe_glmbase_01-TR{tr}_cropped"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--dir-pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--model", default="rewDSR_instr")
    parser.add_argument("--trs", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument(
        "--brain-mask-trs", default="",
        help=("optional TRs used only to construct the common brain mask for "
              "whole-brain demeaning, e.g. 0,1,...,11 while testing TR4,5; "
              "default: the tested TRs"))
    parser.add_argument("--mask", action="append", required=True,
                        help="name=path; repeat for multiple correction families")
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cluster-forming-p", type=float, default=0.001,
                        help="one-sided uncorrected voxel p (default: .001)")
    parser.add_argument("--connectivity", type=int, choices=(6, 18, 26), default=26)
    parser.add_argument("--demean", action="store_true",
                        help="subtract each subject/TR whole-brain mean before testing")
    return parser.parse_args()


def connectivity_structure(connectivity):
    rank = {6: 1, 18: 2, 26: 3}[connectivity]
    return ndimage.generate_binary_structure(3, rank)


class ClusterFinder:
    """Find clusters in a compact bounding box around one ROI."""

    def __init__(self, ijk, threshold, connectivity):
        coords = np.column_stack(ijk).astype(int)
        self.minimum = coords.min(axis=0)
        local = coords - self.minimum
        self.local = tuple(local[:, axis] for axis in range(3))
        self.shape = tuple((local.max(axis=0) + 1).tolist())
        self.threshold = float(threshold)
        self.structure = connectivity_structure(connectivity)

    def clusters(self, t_values, with_members=False):
        """Return clusters across TRs; used unchanged for data/permutations."""
        found = []
        volume = np.zeros(self.shape, dtype=np.float32)
        for tr_index in range(t_values.shape[1]):
            volume.fill(0)
            volume[self.local] = t_values[:, tr_index]
            labels, n_labels = ndimage.label(
                volume > self.threshold, structure=self.structure)
            if not n_labels:
                continue
            label_at_voxel = labels[self.local]
            masses = np.bincount(
                labels.ravel(),
                weights=np.maximum(volume - self.threshold, 0).ravel(),
                minlength=n_labels + 1)
            sizes = np.bincount(labels.ravel(), minlength=n_labels + 1)
            for cluster_id in range(1, n_labels + 1):
                members = np.flatnonzero(label_at_voxel == cluster_id)
                if members.size == 0:
                    continue
                peak_member = members[np.argmax(t_values[members, tr_index])]
                record = dict(
                    tr_index=int(tr_index), mass=float(masses[cluster_id]),
                    size=int(sizes[cluster_id]), peak_member=int(peak_member),
                    peak_t=float(t_values[peak_member, tr_index]))
                if with_members:
                    record["members"] = members
                found.append(record)
        return found

    def maximum_mass(self, t_values):
        clusters = self.clusters(t_values, with_members=False)
        return max((cluster["mass"] for cluster in clusters), default=0.0)


def permutation_null(D, finder, n_perm, seed):
    """Maximum cluster-mass null from subject-wise sign flips."""
    n_subjects, n_voxels, n_trs = D.shape
    flat = D.reshape(n_subjects, n_voxels * n_trs)
    sum_squares = (flat ** 2).sum(axis=0)
    observed = L.tstat(D)

    def t_from_flips(flips):
        means = (flips @ flat) / n_subjects
        variances = ((sum_squares[None, :] - n_subjects * means ** 2)
                     / (n_subjects - 1))
        values = np.where(
            variances > 0,
            means * np.sqrt(n_subjects) / np.sqrt(np.where(variances > 0, variances, 1)),
            0.0)
        return values.reshape(len(flips), n_voxels, n_trs)

    identity = t_from_flips(np.ones((1, n_subjects)))[0]
    if not np.allclose(identity, observed, atol=1e-4):
        raise AssertionError("permutation t does not reproduce the observed t")

    rng = np.random.RandomState(seed)
    null = np.empty(n_perm, dtype=np.float32)
    block_size = max(1, min(100, 10_000_000 // (n_voxels * n_trs)))
    for start in range(0, n_perm, block_size):
        stop = min(start + block_size, n_perm)
        flips = rng.choice([-1.0, 1.0], size=(stop - start, n_subjects))
        permuted_t = t_from_flips(flips)
        for offset, values in enumerate(permuted_t):
            null[start + offset] = finder.maximum_mass(values)
        if stop == n_perm or stop % 1000 == 0:
            print(f"  permutations {stop}/{n_perm}", flush=True)
    return observed, null


def save_roi_results(out_dir, name, D, ref, ijk, trs, args, threshold):
    finder = ClusterFinder(ijk, threshold, args.connectivity)
    observed_t, null = permutation_null(D, finder, args.n_perm, args.seed)
    clusters = finder.clusters(observed_t, with_members=True)
    cluster_p = np.ones(observed_t.shape, dtype=np.float32)
    rows = []
    for cluster_number, cluster in enumerate(
            sorted(clusters, key=lambda item: item["mass"], reverse=True), 1):
        p_fwe = ((np.count_nonzero(null >= cluster["mass"]) + 1)
                 / (args.n_perm + 1))
        members = cluster.pop("members")
        cluster_p[members, cluster["tr_index"]] = p_fwe
        peak_ijk = [axis[cluster["peak_member"]] for axis in ijk]
        peak_mni = nib.affines.apply_affine(ref.affine, peak_ijk)
        rows.append(dict(
            mask=name, cluster=cluster_number,
            TR=int(trs[cluster["tr_index"]]),
            size_voxels=cluster["size"], cluster_mass=cluster["mass"],
            peak_t=cluster["peak_t"],
            peak_mni=[int(round(value)) for value in peak_mni],
            p_FWE=float(p_fwe), significant_FWE=bool(p_fwe < 0.05)))

    os.makedirs(out_dir, exist_ok=True)
    header = ref.header.copy()
    for suffix, values, fill in (
            ("t", observed_t, 0.0), ("clusterFWEp", cluster_p, 1.0)):
        volume = L.vol_from_cols(values, fill, ref, ijk, len(trs))
        nib.save(nib.Nifti1Image(volume, ref.affine, header),
                 os.path.join(out_dir, f"{args.model}_{suffix}.nii.gz"))
    np.save(os.path.join(out_dir, f"{args.model}_null_max_cluster_mass.npy"), null)

    significant = [row for row in rows if row["significant_FWE"]]
    summary = dict(
        mask=name, model=args.model, n_subjects=int(D.shape[0]),
        n_voxels=int(D.shape[1]), trs=trs, n_trs=len(trs),
        test="one-sided positive sign-flip maximum cluster mass",
        cluster_forming_p_uncorrected_one_sided=args.cluster_forming_p,
        cluster_forming_t=float(threshold), connectivity=args.connectivity,
        correction_family="maximum spatial cluster mass across all supplied TRs",
        n_perm=args.n_perm, seed=args.seed, demean=args.demean,
        n_observed_clusters=len(rows), n_significant_clusters=len(significant),
        clusters=rows)
    with open(os.path.join(out_dir, f"{args.model}_cluster_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary, rows


def main():
    args = parse_args()
    np.random.seed(42)
    trs = [int(value) for value in args.trs.split(",")]
    brain_mask_trs = ([int(value) for value in args.brain_mask_trs.split(",")]
                      if args.brain_mask_trs else trs)
    L.check_inputs(args.root, args.dir_pattern, [args.model], trs)
    ref, brain = L.load_ref(args.root, args.dir_pattern, brain_mask_trs)
    masks = L.load_masks(args.mask, ref, brain)

    union = brain.copy() if args.demean else np.logical_or.reduce(
        [record["bool"] for record in masks.values()])
    union_ijk = np.where(union)
    data = L.read_model_columns(
        args.root, args.dir_pattern, args.model, trs, union_ijk)
    if args.demean:
        data -= data.mean(axis=1, keepdims=True)

    threshold = stats.t.ppf(
        1.0 - args.cluster_forming_p, df=data.shape[0] - 1)
    all_rows = []
    summaries = {}
    for name, record in masks.items():
        columns = np.flatnonzero(record["bool"][union_ijk])
        roi_data = np.ascontiguousarray(data[:, columns, :])
        roi_ijk = tuple(axis[columns] for axis in union_ijk)
        print(f"[{name}] {len(columns)} voxels; cluster-forming t={threshold:.4f}")
        summary, rows = save_roi_results(
            os.path.join(args.out_dir, name), name, roi_data, ref, roi_ijk,
            trs, args, threshold)
        summaries[name] = summary
        all_rows.extend(rows)
        print(f"[{name}] {summary['n_significant_clusters']} significant cluster(s)")

    os.makedirs(args.out_dir, exist_ok=True)
    settings = dict(vars(args))
    settings.update(
        trs=trs, brain_mask_trs=brain_mask_trs, n_subjects=int(data.shape[0]),
        cluster_forming_t=float(threshold),
        inference_note=(
            "post-hoc sensitivity analysis; cluster inference does not repair "
            "a misspecified or confounded first-level/RSA model"),
        masks={name: dict(path=record["path"], n_voxels=record["n_vox"])
               for name, record in masks.items()})
    with open(os.path.join(args.out_dir, "settings.json"), "w") as handle:
        json.dump(settings, handle, indent=2)
    if all_rows:
        fields = list(all_rows[0])
        with open(os.path.join(args.out_dir, "cluster_table.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)


if __name__ == "__main__":
    main()
