#!/usr/bin/env python3
"""Build one manuscript-facing summary of the human-cell spatial analyses.

This report keeps four conceptually distinct analyses separate:

1. ``per_lag_encoding`` / full cohort / no controls
   Manuscript-primary single-cell lag analysis (single-config CV).
2. ``spatial_peaks_simple`` / full cohort
   Paired-grid-group robustness analysis.
3. The same two estimators in subjects outside the DSR-RSA cohort.
4. ``RSA_DSR_ROIs_simple``
   Population-RDM analysis; not a single-cell lag test. Its supported DSR
   effect is the all-future-lags-except-now model under ``shift_and_swap``.

The script auto-detects the newest compatible relabelled result folders,
writes a timestamped archive, and refreshes stable ``LATEST_*`` files under
``derivatives/group/manuscript_cell_results_core``.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.per_lag_encoding as ple


DATA_DIR = Path(
    "/Users/xpsy1114/Documents/projects/multiple_clocks/"
    "data/ephys_humans/derivatives"
)
PER_LAG_BASE = DATA_DIR / "group/per_lag_encoding"
SPATIAL_BASE = DATA_DIR / "group/spatial_peaks_simple"
OUT_BASE = DATA_DIR / "group/manuscript_cell_results_core"
RSA_SUBJECTS_JSON = DATA_DIR / "all_sessions_dsrRSA_grouping_summary.json"
RSA_RUN = (DATA_DIR / "group/DSR_RSA_simple_ROI/"
           "2026-08-31_17-57-30")   # Fig 2d run, final labels

CORE_ROIS = ("mPFC", "HC_anterior", "HC_mid")


def _latest_per_lag():
    candidates = [p for p in PER_LAG_BASE.iterdir() if p.is_dir()
                  and (p / "core_results_noctrl_fisher_z.csv").exists()
                  and (p / "per_cell_ALL_ROIs.csv").exists()]
    if not candidates:
        raise FileNotFoundError("No compatible per-lag result folder found")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _latest_spatial(no_rsa):
    candidates = []
    for p in SPATIAL_BASE.iterdir():
        stats_path, settings_path = p / "per_roi_stats.csv", p / "settings.json"
        if not p.is_dir() or not stats_path.exists() or not settings_path.exists():
            continue
        try:
            settings = json.loads(settings_path.read_text())
            stats_df = pd.read_csv(stats_path)
        except Exception:
            continue
        source_text = (str(settings.get("reload_from", "")) + " " + p.name).lower()
        is_no_rsa = "no_rsa" in source_text or "not_in_rsa" in source_text
        if is_no_rsa != no_rsa or "mPFC" not in set(stats_df.get("roi", [])):
            continue
        candidates.append(p)
    if not candidates:
        label = "RSA-excluded" if no_rsa else "full-cohort"
        raise FileNotFoundError(f"No compatible {label} spatial-peaks folder found")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _norm_subject(series):
    return (series.astype(str).str.replace(r"\.0$", "", regex=True)
            .str.zfill(2))


def _per_lag_tables(run_dir):
    full_core = pd.read_csv(run_dir / "core_results_noctrl_fisher_z.csv")
    # Add the manuscript-critical mPFC current-location result. The compact
    # per-lag core file otherwise contains only each ROI's predicted lags.
    lag_table = pd.read_csv(run_dir / "roi_lag_table_fisher_z_noctrl.csv")
    raw_stats = pd.read_csv(run_dir / "per_roi_stats.csv")
    now = lag_table[(lag_table["roi"] == "mPFC")
                    & (lag_table["lag_deg"] == 0)].iloc[0]
    raw_now = raw_stats[(raw_stats["roi"] == "mPFC")
                        & (raw_stats["ctrl_mode"] == "noctrl")].iloc[0]
    k_now, n_now = int(raw_now["T3_k_lag000"]), int(raw_now["n_cells"])
    full_core = pd.concat([full_core, pd.DataFrame([{
        "result_type": "individual_lag_vs_zero",
        "roi": "mPFC", "lag_deg": 0,
        "comparison": "0° lag vs 0",
        "test": "two-sided one-sample t-test on Fisher z",
        "n": int(now["n_cells"]), "df": int(now["df"]),
        "mean_raw_r": now["mean_r"], "mean_fisher_z": now["mean_z"],
        "t": now["t_vs_0"], "p_unc": now["p_unc"], "p_fdr": now["p_fdr"],
        "perm_sig_cells": k_now, "roi_cells": n_now,
        "perm_sig_percent": 100 * k_now / n_now,
        "perm_binom_p_fdr": raw_now.get("T3_p_lag000_fdr", np.nan),
    }])], ignore_index=True)
    per_cell = pd.read_csv(run_dir / "per_cell_ALL_ROIs.csv")
    rsa_subjects = set(json.loads(RSA_SUBJECTS_JSON.read_text()).keys())
    no_rsa = per_cell.loc[~_norm_subject(per_cell["subject_id"])
                          .isin(rsa_subjects)].copy()
    raw_no_rsa = ple.per_roi_stats(no_rsa, ctrl_mode=False)
    fz_no_rsa = ple.fisher_per_roi_stats_noctrl(no_rsa)
    no_rsa_core = ple._core_result_rows(raw_no_rsa, fz_no_rsa)
    return full_core, no_rsa_core, per_cell, no_rsa


def _spatial_rows(run_dir, cohort):
    roi_stats = pd.read_csv(run_dir / "per_roi_stats.csv")
    lag_stats = pd.read_csv(run_dir / "per_roi_single_lag_stats.csv")
    rows = []
    for roi in CORE_ROIS:
        rr = roi_stats[roi_stats["roi"] == roi]
        if rr.empty:
            continue
        rr = rr.iloc[0]
        for lag in ple.ROI_PREDICTED_LAGS_DEG[roi]:
            lr = lag_stats[(lag_stats["roi"] == roi)
                           & (lag_stats["lag_deg"] == lag)]
            if lr.empty:
                continue
            lr = lr.iloc[0]
            n = int(lr["n_cells_finite"])
            rows.append({
                "analysis": "spatial_peaks_paired_groups",
                "cohort": cohort,
                "roi": roi,
                "result_type": "individual_lag_vs_zero",
                "lag_deg": lag,
                "comparison": f"{lag}° lag vs zero",
                "scale": "raw r",
                "alternative": "one-sided greater",
                "n": n, "df": n - 1,
                "mean_raw_r": lr["mean_r"],
                "t": lr["t_vs_0"], "p_unc": lr["p_unc"],
                "p_fdr": lr["p_fdr"],
                "perm_sig_k": np.nan, "perm_sig_n": np.nan,
                "perm_sig_percent": np.nan, "perm_binom_p_fdr": np.nan,
                "source": str(run_dir),
            })
        n = int(rr["test1_meanR_n"])
        rows.append({
            "analysis": "spatial_peaks_paired_groups", "cohort": cohort,
            "roi": roi, "result_type": "mean_predicted_lags_vs_zero",
            "lag_deg": np.nan,
            "comparison": f"mean predicted lags {rr['target_lags_deg']} vs zero",
            "scale": "raw r", "alternative": "one-sided greater",
            "n": n, "df": n - 1, "mean_raw_r": rr["test1_meanR_mean"],
            "t": rr["test1_meanR_t"], "p_unc": rr["test1_meanR_p_unc"],
            "p_fdr": rr["test1_meanR_p_fdr"],
            "perm_sig_k": np.nan, "perm_sig_n": np.nan,
            "perm_sig_percent": np.nan, "perm_binom_p_fdr": np.nan,
            "source": str(run_dir),
        })
        n = int(rr["test2_targetVsOther_n"])
        rows.append({
            "analysis": "spatial_peaks_paired_groups", "cohort": cohort,
            "roi": roi, "result_type": "predicted_lags_vs_other_lags",
            "lag_deg": np.nan,
            "comparison": "predicted lags vs all other lags",
            "scale": "raw r", "alternative": "one-sided greater",
            "n": n, "df": n - 1,
            "mean_raw_r": rr["test2_targetVsOther_mean_diff"],
            "t": rr["test2_targetVsOther_t"],
            "p_unc": rr["test2_targetVsOther_p_unc"],
            "p_fdr": rr["test2_targetVsOther_p_fdr"],
            "perm_sig_k": np.nan, "perm_sig_n": np.nan,
            "perm_sig_percent": np.nan, "perm_binom_p_fdr": np.nan,
            "source": str(run_dir),
        })
        k, n_perm = int(rr["test3_permSig_k"]), int(rr["test3_permSig_n"])
        rows.append({
            "analysis": "spatial_peaks_paired_groups", "cohort": cohort,
            "roi": roi,
            "result_type": "permutation_sig_fraction_predicted_lags",
            "lag_deg": np.nan,
            "comparison": "fraction significant for predicted-lag average",
            "scale": "per-cell permutation p", "alternative": "greater than 5%",
            "n": n_perm, "df": np.nan, "mean_raw_r": np.nan,
            "t": np.nan, "p_unc": rr["test3_permSig_p_unc"],
            "p_fdr": rr["test3_permSig_p_fdr"],
            "perm_sig_k": k, "perm_sig_n": n_perm,
            "perm_sig_percent": 100 * k / n_perm if n_perm else np.nan,
            "perm_binom_p_fdr": rr["test3_permSig_p_fdr"],
            "source": str(run_dir),
        })
    return pd.DataFrame(rows), roi_stats, lag_stats


def _convert_per_lag(core, cohort, source):
    out = core.copy()
    out.insert(0, "cohort", cohort)
    out.insert(0, "analysis", "per_lag_single_config_noctrl")
    out["scale"] = "Fisher z"
    out["alternative"] = "two-sided"
    out["perm_sig_k"] = out.pop("perm_sig_cells")
    out["perm_sig_n"] = out.pop("roi_cells")
    out["source"] = str(source)
    return out


def _gradient_overlap():
    import scripts.cell_mask_overlap as cmo
    cells = pd.read_csv(cmo.CELL_TABLE_PATH)
    for axis in "xyz":
        final = f"MNI_{axis}_final"
        if final in cells:
            cells[f"MNI_{axis}"] = cells[final]
    cells = cells.dropna(subset=[cmo.ROI_COL, "MNI_x", "MNI_y", "MNI_z"])
    cells = cells[cells[cmo.ROI_COL].isin(cmo.ROI_ORDER)].copy()
    union = cmo.build_gradient_union_mask(
        cmo.GRADIENT_TSTAT_DIR, cmo.GRADIENT_TSTAT_MAPS,
        cmo.GRADIENT_TSTAT_THRESHOLDS,
        prebuilt_path=cmo.GRADIENT_PREBUILT_MASK,
    )
    counts = cmo.gradient_overlap_counts(cells, union, cmo.ROI_COL)
    row = counts[counts["roi"] == "mPFC"].iloc[0]
    return int(row["n_inside"]), int(row["n_total"])


def _p(p):
    if not np.isfinite(p):
        return "NA"
    return "< .001" if p < .001 else f"= {p:.3f}"


def _stat(row):
    if not np.isfinite(row.get("t", np.nan)):
        return "NA"
    return f"t({int(row['df'])}) = {row['t']:.2f}"


def _one(df, analysis, cohort, roi, result_type, lag=None):
    sel = ((df["analysis"] == analysis) & (df["cohort"] == cohort)
           & (df["roi"] == roi) & (df["result_type"] == result_type))
    if lag is not None:
        sel &= df["lag_deg"].eq(lag)
    rows = df[sel]
    if rows.empty:
        raise KeyError((analysis, cohort, roi, result_type, lag))
    return rows.iloc[0]


def _write_report(all_results, paths, gradient_count, out_dir):
    pl = "per_lag_single_config_noctrl"
    sp = "spatial_peaks_paired_groups"
    m30 = _one(all_results, pl, "full", "mPFC", "individual_lag_vs_zero", 30)
    m0 = _one(all_results, pl, "full", "mPFC", "individual_lag_vs_zero", 0)
    m60 = _one(all_results, pl, "full", "mPFC", "individual_lag_vs_zero", 60)
    mavg = _one(all_results, pl, "full", "mPFC", "mean_predicted_lags_vs_zero")
    mpair = _one(all_results, pl, "full", "mPFC", "predicted_lags_vs_other_lags")
    spavg = _one(all_results, sp, "full", "mPFC", "mean_predicted_lags_vs_zero")
    sppair = _one(all_results, sp, "full", "mPFC", "predicted_lags_vs_other_lags")
    spfrac = _one(all_results, sp, "full", "mPFC",
                  "permutation_sig_fraction_predicted_lags")
    nravg = _one(all_results, sp, "not_in_rsa_subjects", "mPFC",
                 "mean_predicted_lags_vs_zero")
    nrpair = _one(all_results, sp, "not_in_rsa_subjects", "mPFC",
                  "predicted_lags_vs_other_lags")
    nrfrac = _one(all_results, sp, "not_in_rsa_subjects", "mPFC",
                  "permutation_sig_fraction_predicted_lags")
    pl_nravg = _one(all_results, pl, "not_in_rsa_subjects", "mPFC",
                    "mean_predicted_lags_vs_zero")
    pl_nrpair = _one(all_results, pl, "not_in_rsa_subjects", "mPFC",
                     "predicted_lags_vs_other_lags")

    primary_lag_lines = [
        "| ROI | Lag | Mean raw r | t(df) | p | FDR p | Perm-sig cells |",
        "| --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for roi in CORE_ROIS:
        for lag in ple.ROI_PREDICTED_LAGS_DEG[roi]:
            row = _one(all_results, pl, "full", roi,
                       "individual_lag_vs_zero", lag)
            primary_lag_lines.append(
                f"| {roi} | {lag}° | {row['mean_raw_r']:.3f} | {_stat(row)} | "
                f"{_p(row['p_unc'])} | {_p(row['p_fdr'])} | "
                f"{int(row['perm_sig_k'])}/{int(row['perm_sig_n'])} "
                f"({row['perm_sig_percent']:.1f}%) |")

    hc_summary_lines = [
        "| ROI | Test | t(df) | p | FDR p |",
        "| --- | --- | --- | --- | --- |",
    ]
    for roi in ("HC_anterior", "HC_mid"):
        avg = _one(all_results, pl, "full", roi,
                   "mean_predicted_lags_vs_zero")
        pair = _one(all_results, pl, "full", roi,
                    "predicted_lags_vs_other_lags")
        hc_summary_lines += [
            f"| {roi} | mean 0°/330° vs zero | {_stat(avg)} | "
            f"{_p(avg['p_unc'])} | {_p(avg['p_fdr'])} |",
            f"| {roi} | 0°/330° vs other lags | {_stat(pair)} | "
            f"{_p(pair['p_unc'])} | {_p(pair['p_fdr'])} |",
        ]

    spatial_fraction_lines = [
        "| ROI | Permutation-significant cells | Binomial FDR p |",
        "| --- | ---: | --- |",
    ]
    for roi in CORE_ROIS:
        row = _one(all_results, sp, "full", roi,
                   "permutation_sig_fraction_predicted_lags")
        spatial_fraction_lines.append(
            f"| {roi} | {int(row['perm_sig_k'])}/{int(row['perm_sig_n'])} "
            f"({row['perm_sig_percent']:.1f}%) | {_p(row['p_fdr'])} |")

    lines = [
        "# Core manuscript results: human-cell future-location analyses",
        "",
        "## Which analysis supports which claim",
        "",
        "| Name to use | What it estimates | Use in manuscript |",
        "| --- | --- | --- |",
        "| **Per-lag encoding (primary; no controls)** | Leave one single task "
        "configuration out; pool all training configurations into one predicted "
        "rate map. | Primary lag profile, individual 0°/30°/60°/330° tests, "
        "and the mPFC-versus-HC dissociation. |",
        "| **Future spatial peaks (paired-group robustness)** | Pair configurations "
        "for coverage; correlate each held-out group separately with each training "
        "group. | Independent estimator / robustness statement, not the source of "
        "the claim that mPFC is at chance at 0°. |",
        "| **RSA-excluded cohort** | Either estimator restricted to subjects outside "
        "the DSR-RSA-eligible cohort. | Independence check from the population RSA. "
        "Always state which estimator is being quoted. |",
        "| **Population DSR RSA** | Population RDM for all future lags except lag 0. "
        "| Separate population-level result. It is supported under the compound "
        "`shift_and_swap` null: per-trial circular shifts plus trial-to-configuration "
        "reassignment. It is not a single-cell lag or permutation-fraction test. |",
        "",
        "## What is shared and what is genuinely different",
        "",
        "Both single-cell analyses phase-residualise firing, build twelve lagged "
        "9-location rate maps, use dwell-weighted Pearson correlations, and evaluate "
        "predicted lags against zero and against other lags. Paired configurations "
        "are **not** the only difference: spatial peaks averages held-out-versus-each-"
        "training-group correlations, requires five shared locations, and shifts each "
        "repetition independently; per-lag encoding pools training maps, requires "
        "three shared locations, and shifts the held-out configuration series once "
        "per fold/permutation.",
        "",
        "## Manuscript-primary numbers: per-lag encoding, no controls",
        "",
        "All t-tests below use Fisher-transformed fold-averaged correlations and are "
        "two-sided. Raw mean r is retained as the effect size.",
        "",
        "| Result | Mean raw r | Statistic | p | FDR p |",
        "| --- | ---: | --- | --- | --- |",
        f"| mPFC 30° vs zero | {m30['mean_raw_r']:.3f} | {_stat(m30)} | "
        f"{_p(m30['p_unc'])} | {_p(m30['p_fdr'])} |",
        f"| mPFC 60° vs zero | {m60['mean_raw_r']:.3f} | {_stat(m60)} | "
        f"{_p(m60['p_unc'])} | {_p(m60['p_fdr'])} |",
        f"| mPFC 0° vs zero | {m0['mean_raw_r']:.3f} | {_stat(m0)} | "
        f"{_p(m0['p_unc'])} | {_p(m0['p_fdr'])} |",
        f"| mPFC mean 30°/60° vs zero | — | {_stat(mavg)} | "
        f"{_p(mavg['p_unc'])} | {_p(mavg['p_fdr'])} |",
        f"| mPFC 30°/60° vs other ten lags | — | {_stat(mpair)} | "
        f"{_p(mpair['p_unc'])} | {_p(mpair['p_fdr'])} |",
        "",
        "The individual-lag FDR is across reported ROIs within each lag (not across "
        "all ROI × lag cells). The predicted-average and paired-test FDR families "
        "contain the three ROIs with a priori lag sets.",
        "",
        "### All a priori lag tests and cell percentages",
        "",
        *primary_lag_lines,
        "",
        "The permutation percentages in this table are lag-specific per-lag-encoding "
        "null tests. They are not the same quantity as the spatial-peaks percentage "
        "for the average across an ROI's predicted lag set.",
        "",
        "### Hippocampal predicted-lag summaries",
        "",
        *hc_summary_lines,
        "",
        "The combined 0°/330° hippocampal tests are significant, although the "
        "individual-lag FDR results are weaker (see the table above). This should be "
        "reported as a combined current/just-past profile, not as both individual "
        "lags being independently significant.",
        "",
        "## Paired-group spatial-peaks robustness",
        "",
        f"Full-cohort mPFC predicted-lag mean: {_stat(spavg)}, p {_p(spavg['p_unc'])}, "
        f"FDR p {_p(spavg['p_fdr'])}. Paired specificity: {_stat(sppair)}, "
        f"p {_p(sppair['p_unc'])}, FDR p {_p(sppair['p_fdr'])}. "
        f"Permutation-significant fraction: {int(spfrac['perm_sig_k'])}/"
        f"{int(spfrac['perm_sig_n'])} ({spfrac['perm_sig_percent']:.1f}%), "
        f"binomial FDR p {_p(spfrac['p_fdr'])}.",
        "",
        "Important: spatial peaks also finds a positive mPFC 0° estimate in the full "
        "cohort, so the statement that mPFC is at chance at the current location must "
        "be attributed to the primary per-lag estimator, not to both analyses.",
        "",
        "### Paired-group permutation-significant percentages by ROI",
        "",
        *spatial_fraction_lines,
        "",
        "## RSA-excluded cohort",
        "",
        f"The paired-group spatial-peaks estimator reproduces all three mPFC summary "
        f"tests: predicted mean {_stat(nravg)}, FDR p {_p(nravg['p_fdr'])}; "
        f"paired specificity {_stat(nrpair)}, FDR p {_p(nrpair['p_fdr'])}; "
        f"and {int(nrfrac['perm_sig_k'])}/{int(nrfrac['perm_sig_n'])} "
        f"({nrfrac['perm_sig_percent']:.1f}%) permutation-significant cells, "
        f"binomial FDR p {_p(nrfrac['p_fdr'])}.",
        "",
        f"Under the manuscript-primary per-lag/Fisher estimator, the same excluded "
        f"cohort has a significant paired specificity test ({_stat(pl_nrpair)}, FDR "
        f"p {_p(pl_nrpair['p_fdr'])}), but its predicted-lag mean does not survive "
        f"FDR ({_stat(pl_nravg)}, FDR p {_p(pl_nravg['p_fdr'])}). Therefore the phrase "
        "‘all three tests replicated’ is valid only when explicitly labelled as the "
        "paired-group spatial-peaks robustness analysis.",
        "",
        "## Anatomical overlap update",
        "",
        f"Using the current ROI table and the same thresholded gradient union mask, "
        f"{gradient_count[0]}/{gradient_count[1]} mPFC units overlap the gradient "
        "mask. Replace the older 53/159 count with this value.",
        "",
        "## Suggested revised cells-results text",
        "",
        f"We built spatial rate maps at twelve lags and quantified their consistency "
        f"with leave-one-configuration-out cross-validation. In mPFC, consistency "
        f"was significant at the next visited location (30°: {_stat(m30)}, "
        f"p {_p(m30['p_unc'])}, FDR p {_p(m30['p_fdr'])}; mean raw r = "
        f"{m30['mean_raw_r']:.3f}), but not at 60° ({_stat(m60)}, p "
        f"{_p(m60['p_unc'])}) or the current location ({_stat(m0)}, p "
        f"{_p(m0['p_unc'])}). Averaged across the two a priori future lags, "
        f"the population mean exceeded zero ({_stat(mavg)}, p {_p(mavg['p_unc'])}, "
        f"FDR p {_p(mavg['p_fdr'])}), and exceeded the other ten lags in a "
        f"within-cell paired test ({_stat(mpair)}, p {_p(mpair['p_unc'])}, FDR p "
        f"{_p(mpair['p_fdr'])}). A complementary paired-configuration analysis "
        f"reproduced the predicted-lag mean and specificity effects ({_stat(spavg)} "
        f"and {_stat(sppair)}, respectively). In subjects outside the DSR-RSA "
        f"cohort, this paired-configuration estimator again showed a predicted-lag "
        f"mean above zero ({_stat(nravg)}), predicted-lag specificity "
        f"({_stat(nrpair)}), and an excess of permutation-significant mPFC cells "
        f"({int(nrfrac['perm_sig_k'])}/{int(nrfrac['perm_sig_n'])}, "
        f"{nrfrac['perm_sig_percent']:.1f}%).",
        "",
        "## Sources",
        "",
        f"- Per-lag full: `{paths['per_lag']}`",
        f"- Spatial peaks full: `{paths['spatial_full']}`",
        f"- Spatial peaks RSA-excluded: `{paths['spatial_no_rsa']}`",
        f"- Population RSA: `{paths['rsa']}`",
        "",
        "The complete machine-readable table is `LATEST_CORE_RESULTS.csv`.",
    ]
    report = "\n".join(lines) + "\n"
    (out_dir / "CORE_RESULTS.md").write_text(report)
    return report


def main():
    per_lag_dir = _latest_per_lag()
    spatial_full_dir = _latest_spatial(no_rsa=False)
    spatial_no_rsa_dir = _latest_spatial(no_rsa=True)
    pl_full, pl_no_rsa, per_cell, no_rsa = _per_lag_tables(per_lag_dir)
    pl_full = _convert_per_lag(pl_full, "full", per_lag_dir)
    pl_no_rsa = _convert_per_lag(pl_no_rsa, "not_in_rsa_subjects", per_lag_dir)
    sp_full, _, _ = _spatial_rows(spatial_full_dir, "full")
    sp_no_rsa, _, _ = _spatial_rows(spatial_no_rsa_dir,
                                     "not_in_rsa_subjects")
    all_results = pd.concat([pl_full, pl_no_rsa, sp_full, sp_no_rsa],
                            ignore_index=True, sort=False)

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    out_dir = OUT_BASE / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(out_dir / "CORE_RESULTS.csv", index=False)
    gradient_count = _gradient_overlap()
    paths = {
        "per_lag": per_lag_dir,
        "spatial_full": spatial_full_dir,
        "spatial_no_rsa": spatial_no_rsa_dir,
        "rsa": RSA_RUN,
    }
    report = _write_report(all_results, paths, gradient_count, out_dir)
    config = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "sources": {k: str(v) for k, v in paths.items()},
        "per_lag_full_rows": len(per_cell),
        "per_lag_not_in_rsa_rows": len(no_rsa),
        "gradient_overlap_mPFC": {
            "inside": gradient_count[0], "total": gradient_count[1]},
        "rsa_interpretation": {
            "level": "population RDM",
            "model": "all future lags except lag 0 (now)",
            "permutation": "shift_and_swap: per-trial circular shift plus config reassignment",
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Stable, easy-to-find copies refreshed on every run.
    (OUT_BASE / "LATEST_CORE_RESULTS.md").write_text(report)
    shutil.copy2(out_dir / "CORE_RESULTS.csv", OUT_BASE / "LATEST_CORE_RESULTS.csv")
    shutil.copy2(out_dir / "config.json", OUT_BASE / "LATEST_config.json")
    print(f"Wrote {out_dir}")
    print(f"Refreshed {OUT_BASE / 'LATEST_CORE_RESULTS.md'}")


if __name__ == "__main__":
    main()
