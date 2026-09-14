#!/usr/bin/env python
"""Test a per-TR RSA effect averaged over each anatomical ROI.

For every subject and TR, voxel values are averaged across the complete ROI
before inference. A subject-wise sign-flip maximum-t test then corrects across
the supplied TRs only; there is no spatial peak search. Optional whole-brain
demeaning is performed separately for every subject and TR before ROI
averaging, giving a test of regional deviation from that whole-brain mean.
"""
import argparse
import csv
import json
import os

import numpy as np

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
    parser.add_argument("--mask", action="append", required=True,
                        help="name=path; repeat for multiple ROI tests")
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demean", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(42)
    trs = [int(value) for value in args.trs.split(",")]
    L.check_inputs(args.root, args.dir_pattern, [args.model], trs)
    ref, brain = L.load_ref(args.root, args.dir_pattern, trs)
    masks = L.load_masks(args.mask, ref, brain)
    union = brain.copy() if args.demean else np.logical_or.reduce(
        [record["bool"] for record in masks.values()])
    union_ijk = np.where(union)
    data = L.read_model_columns(
        args.root, args.dir_pattern, args.model, trs, union_ijk)
    if args.demean:
        data -= data.mean(axis=1, keepdims=True)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    summaries = {}
    for name, record in masks.items():
        columns = np.flatnonzero(record["bool"][union_ijk])
        subject_by_tr = data[:, columns, :].mean(axis=1)
        observed_t = L.tstat(subject_by_tr)
        null_max = L.null_max_t(
            subject_by_tr, n_perm=args.n_perm, seed=args.seed,
            pblock=L.adaptive_pblock(len(trs)))
        p_fwe = np.asarray([(np.count_nonzero(null_max >= value) + 1)
                            / (args.n_perm + 1) for value in observed_t])
        peak_index = int(np.argmax(observed_t))
        summary = dict(
            mask=name, model=args.model, n_subjects=int(data.shape[0]),
            n_voxels=int(len(columns)), trs=trs, demean=args.demean,
            test="one-sided positive ROI-mean sign-flip maximum t across TRs",
            n_perm=args.n_perm, seed=args.seed,
            mean=[float(value) for value in subject_by_tr.mean(axis=0)],
            sem=[float(value) for value in
                 subject_by_tr.std(axis=0, ddof=1) / np.sqrt(data.shape[0])],
            t=[float(value) for value in observed_t],
            p_FWE=[float(value) for value in p_fwe],
            peak_TR=trs[peak_index], peak_t=float(observed_t[peak_index]),
            peak_p_FWE=float(p_fwe[peak_index]),
            significant_TRs=[trs[index] for index in np.flatnonzero(p_fwe < 0.05)])
        summaries[name] = summary
        np.save(os.path.join(args.out_dir, f"{name}_{args.model}_subject_by_TR.npy"),
                subject_by_tr.astype(np.float32))
        with open(os.path.join(args.out_dir, f"{name}_{args.model}_summary.json"), "w") as handle:
            json.dump(summary, handle, indent=2)
        for index, tr in enumerate(trs):
            rows.append(dict(
                mask=name, TR=tr, mean=summary["mean"][index],
                sem=summary["sem"][index], t=summary["t"][index],
                p_FWE=summary["p_FWE"][index],
                significant_FWE=summary["p_FWE"][index] < 0.05))
        print(f"{name}: peak TR{summary['peak_TR']}, "
              f"t={summary['peak_t']:.3f}, p_FWE={summary['peak_p_FWE']:.4f}; "
              f"significant TRs={summary['significant_TRs']}")

    settings = dict(vars(args))
    settings.update(
        trs=trs, n_subjects=int(data.shape[0]),
        correction_family="12 TRs within each separately tested whole-ROI mean",
        masks={name: dict(path=record["path"], n_voxels=record["n_vox"])
               for name, record in masks.items()})
    with open(os.path.join(args.out_dir, "settings.json"), "w") as handle:
        json.dump(settings, handle, indent=2)
    with open(os.path.join(args.out_dir, "roi_mean_table.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
