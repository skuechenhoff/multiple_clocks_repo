#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 13:39:32 2025

@author: xpsy1114
"""

import os
import mc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from patsy import dmatrices, build_design_matrices
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from itertools import combinations
from itertools import combinations
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import binomtest



from matplotlib.ticker import PercentFormatter
import math

import numpy as np
from matplotlib import cm
from matplotlib.colors import to_hex

def colors_from_palette(labels, cmap="YlGnBu", lo=0.25, hi=0.95):
    """Return {label: hexcolor} with N shades sampled from a blue/turquoise/green palette."""
    cmap = cm.get_cmap(cmap)
    xs = np.linspace(lo, hi, len(labels))
    return {lab: to_hex(cmap(x)) for lab, x in zip(labels, xs)}



def get_data(sub, trials):
    data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
    if not os.path.isdir(data_folder):
        print("running on ceph")
        data_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives"
    if trials == 'residualised':
        res_data = True
    else:
        res_data = False
    data_norm = mc.analyse.helpers_human_cells.load_norm_data(data_folder, [f"{sub:02}"], res_data = res_data)
    return data_norm, data_folder 


def compute_perm_pvalues_groupby(df):
    group_cols = ["session", "neuron", "roi", "model", "term"]

    real  = df[df["permuted"] == False].copy()
    perms = df[df["permuted"] == True].copy()

    perm_groups = perms.groupby(group_cols)

    real["p_perm_t"] = np.nan
    real["p_perm_F"] = np.nan

    for key, real_row in real.groupby(group_cols):
        # key is a tuple (session, neuron, roi, model, term)
        if key not in perm_groups.groups:
            continue

        perm_group = perm_groups.get_group(key)

        t_real = real_row["t"].iloc[0]
        F_real = real_row["F"].iloc[0]

        t_perm = perm_group["t"].to_numpy()
        F_perm = perm_group["F"].to_numpy()

        p_t = (np.sum(np.abs(t_perm) >= abs(t_real)) + 1) / (len(t_perm) + 1)
        p_F = (np.sum(F_perm >= F_real) + 1) / (len(F_perm) + 1)

        # assign same p-values to that real row (usually just 1 row anyway)
        real.loc[real_row.index, "p_perm_t"] = p_t
        real.loc[real_row.index, "p_perm_F"] = p_F

    return real

# some stats helpers
def _p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

def add_sig_brackets(ax, comps_df, xpos, y0, step=0.08, h=0.015, fs=9):
    # xpos: function(term) -> x coordinate (works for overall or ROI-shifted bars)
    # ax = subplot 
    for i, r in enumerate(comps_df.sort_values("p").itertuples(index=False)):
        x1, x2 = sorted([xpos(r.term1), xpos(r.term2)])
        y = y0 + i*step
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], c="k", lw=1.5, clip_on=False)
        ax.text((x1+x2)/2, y+h, _p_to_stars(r.p), ha="center", va="bottom", fontsize=fs, clip_on=False)


# load the latest results.
source_dir = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
group_dir_state = f"{source_dir}/group/state_lin_regs"
os.makedirs(group_dir_state, exist_ok=True)

# trials = 'all_minus_explore'
trials = 'all_correct'
#name_combo = f"{group_dir_state}/Combo_perm_emp_state_rep_int_{trials}.csv"
name_combo = f"{group_dir_state}/Combo_circ-perm_emp_state_rep_int_{trials}.csv"

combo_df = pd.read_csv(name_combo)
combo_df.head()


pperm_path = f"{group_dir_state}/p_circ_perms_state_rep_int_{trials}.csv"
df = pd.read_csv(pperm_path)


# this plots model no 1: main effect state, once for all cells across the entire brain, and once per ROI.
terms = ["A", "B", "C", "D"]
model = "M1_main"

# Colors for each term: orange → purple
base = {"A":"#d95f02","B":"#fe9929","C":"#9e9ac8","D":"#542788"}
colors = {t: base[t.split("_", 1)[0]] for t in terms}  # keys match `terms`

bg_color = "#f0f0f0"  # very light grey for 100% bar

# --- Filter to empirical rows for model M1_main and terms A–D ---
real = df[
    (df["permuted"] == False) &
    (df["model"] == model) &
    (df["term"].isin(terms))
].copy()


# Ensure one row per (session, neuron, roi, model, term)
real = real.drop_duplicates(["session", "neuron", "roi", "model", "term"])
# Define "cell" as (session, neuron) with its ROI
cells = real[["session", "neuron", "roi"]].drop_duplicates()

# ---------------- OVERALL PANEL (using p_perm_t) ----------------

# Total unique cells (across all ROIs)
# --- overall counts / proportions per term ---
total_cells_all = int(cells.shape[0])
sig_cells_t = (
    real.loc[real["p_perm_t"] < 0.05, ["term", "session", "neuron"]]
        .drop_duplicates()
)
sig_counts_overall = sig_cells_t.groupby("term").size().reindex(terms, fill_value=0)
props_overall = (sig_counts_overall / total_cells_all).reindex(terms)


# --- per-term binomial tests (one-sided > 0.05) with Bonferroni across terms ---
alpha = 0.05
overall_rows = []
for term in terms:
    n_sig = int(sig_counts_overall[term])
    res = binomtest(n_sig, total_cells_all, p=alpha, alternative="greater")
    pval = res.pvalue
    p_bonf = min(pval * len(terms), 1.0)
    overall_rows.append({
        "scope": "overall",
        "model": model,
        "roi": "all",
        "term": term,
        "n_cells": total_cells_all,
        "n_sig": n_sig,
        "prop_sig": float(props_overall[term]),
        "pval_binom": pval,
        "p_bonf": p_bonf,
        "sig_bonf": p_bonf < alpha,
    })
overall_stats = pd.DataFrame(overall_rows)

# --- paired t-tests between terms (within-cell comparisons) ---
# pivot to wide form indexed by (session,neuron) so comparisons are paired
wide = real.pivot_table(index=["session", "neuron"], columns="term", values="p_perm_t")[terms]

pairwise_rows = []
for a, b in combinations(terms, 2):
    joined = wide[[a, b]].dropna()
    n_pairs = int(joined.shape[0])
    if n_pairs == 0:
        t_stat, p_val = np.nan, np.nan
    else:
        t_stat, p_val = ttest_rel(joined[a], joined[b], nan_policy="omit")
    pairwise_rows.append({
        "model": model,
        "term1": a,
        "term2": b,
        "t": t_stat,
        "p": p_val,
        "p_bonf": min(pval * len(terms), 1.0),
        "n_cells": n_pairs,
        "df": n_pairs - 1 if n_pairs > 0 else np.nan
    })
overall_pairwise = pd.DataFrame(pairwise_rows).sort_values("p").reset_index(drop=True)

# add to one summary.
overall_summary = pd.concat([overall_stats, overall_pairwise], ignore_index = True, sort = False)


# ---------------- BY-ROI PANEL (using p_perm_t) ----------------

# Total cells per ROI (denominator)
cells_per_roi = (
    cells[["roi", "session", "neuron"]]
    .drop_duplicates()
    .groupby("roi")
    .size()
)
rois = sorted(cells_per_roi.index.to_list())

# Significant cells per (ROI, term)
sig_cells_t_roi = (
    real[real["p_perm_t"] < 0.05][["roi", "term", "session", "neuron"]]
    .drop_duplicates()
)
sig_counts_roi = (
    sig_cells_t_roi.groupby(["roi", "term"])
    .size()
    .unstack("term", fill_value=0)
)


# Proportions by ROI and term: N_sig(R,T) / N_total(R)
props_by_roi = sig_counts_roi.div(cells_per_roi, axis=0)  # broadcast over index

# one-sided test: greater than chance (p=0.05)
by_roi_rows = []
for roi in rois:
    for term in terms:
        n_sig = int(sig_counts_roi[term].loc[roi])
        bin_res = binomtest(sig_counts_roi[term].loc[roi], cells_per_roi.loc[roi], p=alpha, alternative="greater")
        pval = bin_res.pvalue
        p_bonf = min(pval * len(terms), 1.0)
        by_roi_rows.append({
            "scope": "by_roi",
            "model": model,
            "roi": roi,
            "term": term,
            "n_cells": cells_per_roi.loc[roi],
            "n_sig": n_sig,
            "prop_sig": float(props_overall[term]),
            "pval_binom": pval,
            "p_bonf": p_bonf,
            "sig_bonf": p_bonf < alpha,
        })
overall_stats_per_roi = pd.DataFrame(by_roi_rows)



# signficance between terms per roi
wide_roi = real.pivot_table(index=["roi","session","neuron"], columns="term", values="p_perm_t")[terms]

pairwise_by_roi = pd.concat(
    [pd.DataFrame(
        [(roi, term_a, term_b,
          *ttest_rel(perm_p[term_a], perm_p[term_b], nan_policy="omit"),
          int(perm_p[[term_a, term_b]].dropna().shape[0]))   # n_pairs
         for term_a, term_b in combinations(terms, 2)],
        columns=["roi", "term1", "term2", "t", "p", "n_cells"])
     for roi, perm_p in wide_roi.groupby(level="roi")],
    ignore_index=True
).assign(df=lambda d: d["n_cells"] - 1).sort_values(["roi", "p"])

# compute significance across ROI
rows = []
for r1, r2 in combinations(real['roi'].unique(), 2):
    v1 = real.loc[real['roi'] == r1, 'p_perm_t'].dropna()
    v2 = real.loc[real['roi'] == r2, 'p_perm_t'].dropna()
    if len(v1) == 0 or len(v2) == 0:
        t, p = float('nan'), float('nan')
    else:
        t, p = ttest_ind(v1, v2, equal_var=False)   # Welch's t-test
    rows.append((r1, r2, t, p, len(v1), len(v2)))

roi_comp_pairwise = (
    pd.DataFrame(rows, columns=['roi1','roi2','t','p','n1','n2'])
      .assign(n_total=lambda d: d.n1 + d.n2)
      .sort_values('p')
      .reset_index(drop=True)
)



# create stats reporting .csv
# ---- REPORTING TABLES ----
# overall counts/props per term
overall_summary = pd.concat([overall_summary, overall_stats_per_roi, pairwise_by_roi], ignore_index = True, sort = False)
out_csv = f"{group_dir_state}/report_{model}_{trials}.csv"
overall_summary.to_csv(out_csv, index=False)
print("Saved report:", out_csv)



# ---------------- PLOTTING ----------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(11.7, 4.2),                 # ~A4 width in inches, short height
    constrained_layout=True,
    gridspec_kw={"width_ratios": [1, 2]} # 1/3 vs 2/3
)


# ===== Left panel: overall =====
ax1 = axes[0]
x1 = np.arange(len(terms))

# background 100% bars
ax1.bar(x1, [1] * len(terms), color=bg_color, width=0.8, zorder=0)

# colored bars for proportions
bar_colors = [colors[t] for t in terms]
ax1.bar(x1, props_overall.values, color=bar_colors, width=0.5, zorder=1)

star_y = 0.7
for term_i, term in enumerate(terms):
    # # and stars for significance per proportion
    if overall_summary.loc[(overall_summary["term"] == term) & (overall_summary["roi"] == "all"), "sig_bonf"].iat[0]:
        ax1.text(x1[term_i], star_y, "*", ha="center", va="center", fontsize=12, weight="bold")


# add signficance comparisons, but Bonferroni correct for MC (6 tests)
term_to_idx = {t:i for i,t in enumerate(terms)}
sig_pairwise = overall_pairwise[overall_pairwise['p'] < 0.05/len(overall_pairwise)]
add_sig_brackets(ax1, sig_pairwise, xpos=lambda t: term_to_idx[t],
                 y0=float(props_overall.max()) + 0.05, step = 0.07)

#add_sig_brackets(ax1, sig_pairwise, terms, props_overall.values)
    
ax1.set_xticks(x1)
ax1.set_xticklabels(terms)
ax1.set_ylim(0, 1.25)
ax1.set_ylabel("Proportion of cells significant\n(p < 0.05)")
ax1.set_title("Overall (all cells)")
ax1.yaxis.grid(True, linestyle="--", alpha=0.3)


# ===== Right panel: by ROI =====
ax2 = axes[1]
x2 = np.arange(len(rois))
bar_width = 0.18

# term-specific offsets so bars sit side-by-side per ROI
base_offset = {"A":-1.5* bar_width,"B":-0.5* bar_width,"C":0.5* bar_width,"D":1.5* bar_width}
offsets = {t: base_offset[t.split("_", 1)[0]] for t in terms}  # keys match `terms`

star_y = 0.7

for term in terms:
    positions = x2 + offsets[term]
    vals = props_by_roi[term].values  # proportion for each ROI for this term

    # background 100% bar for each term/ROI
    ax2.bar(positions, [1] * len(rois), color=bg_color,
            width=bar_width, zorder=0)

    # colored bar with actual proportion
    ax2.bar(positions, vals, color=colors[term],
            width=bar_width, zorder=1, label=term)
    

for roi_i, roi in enumerate(rois):
    for term in terms:
        if overall_summary.loc[(overall_summary["term"] == term) & (overall_summary["roi"] == roi), "sig_bonf"].iat[0]:
            ax2.text(roi_i + offsets[term], star_y, "*", ha="center", va="center", fontsize=12, weight="bold")
    

    
for roi in rois:
    roi_i = rois.index(roi)
    # add signficance comparisons, but Bonferroni correct for MC (6 tests)
    sig_pairwise_roi = pairwise_by_roi[(pairwise_by_roi['p'] < 0.05/len(pairwise_by_roi)) & (pairwise_by_roi['roi'] == roi)]
    add_sig_brackets(
        ax2, sig_pairwise_roi,
        xpos=lambda t, roi_i=roi_i: roi_i + offsets[t],   # shifted x positions
        y0=float(props_by_roi.loc[roi, terms].max()) + 0.045,
        step=0.07, h=0.012, fs=8
    )
        

ax2.set_xticks(x2)
ax2.set_xticklabels(rois, rotation=45, ha="right")
ax2.set_ylim(0, 1.25)
ax2.set_ylabel("Proportion of cells significant\n(p_perm_F < 0.05)")
ax2.set_title("By ROI")
ax2.yaxis.grid(True, linestyle="--", alpha=0.3)
ax2.legend(title="Term", frameon=False)

plt.suptitle(f"Permutation-based significance (A–D), model {model}", fontsize=12)

out_pdf = f"{group_dir_state}/fig_{model}_ABCDeffect_{trials}.pdf"
out_svg = f"{group_dir_state}/fig_{model}_ABCDeffect_{trials}.svg"
out_png = f"{group_dir_state}/fig_{model}_ABCDeffect_{trials}.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight")
print("Saved figure:", out_pdf, "and", out_svg)

plt.show()
import pdb; pdb.set_trace()

# -------------------------------------------------------------------


# ---------------- OVERALL PANEL (using p_perm_F) ----------------

# # examples
# roi_colors   = colors_from_palette(rois,  cmap="GnBu")     # green→turquoise→blue
# state_colors = colors_from_palette(terms, cmap="PuBuGn")   # purple-blue→greenish



# # Colors for each term: orange → purple
# base = {"A":"#d95f02","B":"#fe9929","C":"#9e9ac8","D":"#542788"}
# colors = {t: base[t.split("_", 1)[0]] for t in terms}  # keys match `terms`

bg_color = "#f0f0f0"  # very light grey for 100% bar

# --- Filter to empirical rows for model M1_main and terms A–D ---
real = df[(df["permuted"] == False) & (df["model"] == model)].copy()
# Ensure one row per (session, neuron, roi, model, term)
real = real.drop_duplicates(["session", "neuron", "roi", "model"])
# Define "cell" as (session, neuron) with its ROI
cells = real[["session", "neuron", "roi"]].drop_duplicates()

# ---------------- OVERALL PANEL (using p_perm_F) ----------------

# Total unique cells (across all ROIs)
# all_cells = cells[["session", "neuron"]].drop_duplicates()
total_cells_all = cells.shape[0]

# For each term, count cells with p_perm_t < 0.05
sig_cells_F = (
    real[real["p_perm_F"] < 0.05][["session", "neuron"]]
    .drop_duplicates()
)

sig_counts_overall = len(sig_cells_F)
props_overall = sig_counts_overall / total_cells_all


# one-sided test: greater than chance (p=0.05)
res_bin_all_cells = binomtest(sig_counts_overall, total_cells_all, p=0.05, alternative="greater")


# ---------------- BY-ROI PANEL (using p_perm_F) ----------------

# Total cells per ROI (denominator)
cells_per_roi = (
    cells[["roi", "session", "neuron"]]
    .drop_duplicates()
    .groupby("roi")
    .size()
)
rois = sorted(cells_per_roi.index.to_list())
# color per roi
roi_colors = colors_from_palette(rois,  cmap="GnBu") 

# Significant cells per (ROI, term)
sig_cells_F_roi = (
    real[real["p_perm_F"] < 0.05][["roi", "session", "neuron"]]
    .drop_duplicates()
)

sig_counts_roi = (
    sig_cells_F_roi.groupby(["roi"])
    .size()
)

# ---------------- proportion significance tests (per ROI) ----------------
rows = []
for roi in rois:
    n_total = int(cells_per_roi.loc[roi])
    n_sig = int(sig_counts_roi.reindex([roi], fill_value=0).iat[0])
    # one-sided test: greater than chance (p=0.05)
    res = binomtest(n_sig, n_total, p=0.05, alternative="greater")
    rows.append((roi, n_total, n_sig, n_sig / n_total if n_total else 0.0, res.pvalue))

roi_prop_tests = pd.DataFrame(rows, columns=["roi", "n_total", "n_sig", "prop", "pval"])
# Bonferroni correction (simple)
roi_prop_tests["p_bonf"] = roi_prop_tests["pval"] * len(rois)
roi_prop_tests["p_bonf"] = roi_prop_tests["p_bonf"].clip(upper=1.0)
alpha = 0.05
roi_prop_tests["sig_bonf"] = roi_prop_tests["p_bonf"] < alpha


# Proportions by ROI : N_sig(R,T) / N_total(R)
props_by_roi = sig_counts_roi.div(cells_per_roi, axis=0)  # broadcast over index

# signficance between rois
# not paired because i'm comparing cells now
pairwise_rois = pd.DataFrame(
    [(r1, r2, *ttest_ind(real.loc[real.roi==r1, "p_perm_F"],
                        real.loc[real.roi==r2, "p_perm_F"],
                        equal_var=False))   # Welch t-test (recommended)
     for r1, r2 in combinations(rois, 2)],
    columns=["roi1","roi2","t","p"]
).sort_values("p")

# ---- REPORTING TABLES (simplified: main effect only) ----

# overall: single row
overall_summary = pd.DataFrame([{
    "scope": "overall",
    "roi": "all",
    "n_total": int(total_cells_all),
    "n_sig": int(sig_cells_F_roi[["session","neuron"]].drop_duplicates().shape[0]),
    "prop_sig": float(sig_cells_F_roi[["session","neuron"]].drop_duplicates().shape[0] / total_cells_all),
}])

# roi-level: one row per ROI
roi_summary = pd.DataFrame({
    "scope": "roi",
    "roi": rois,
    "n_total": cells_per_roi.reindex(rois).astype(int).values,
    "n_sig": sig_counts_roi.reindex(rois, fill_value=0).astype(int).values,
})
roi_summary["prop_sig"] = roi_summary["n_sig"] / roi_summary["n_total"]

# tests: roi vs roi (Welch t-test on p_perm_F)
tests_rois = pairwise_rois.copy()
tests_rois.insert(0, "scope", "roi_vs_roi")

# combine to one CSV
report = pd.concat(
    [overall_summary.assign(section="props_counts"),
     roi_summary.assign(section="props_counts"),
     tests_rois.assign(section="ttests")],
    ignore_index=True,
    sort=False
)

out_csv = f"{group_dir_state}/report_{model}_F_{trials}.csv"
report.to_csv(out_csv, index=False)
print("Saved report:", out_csv)




# ---------------- PLOTTING ----------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(11.7, 4.2),                 # ~A4 width in inches, short height
    constrained_layout=True,
    gridspec_kw={"width_ratios": [1, 2]} # 1/3 vs 2/3
)


# ===== Left panel: overall =====
ax1 = axes[0]
x1 = [0]

# background 100% bars
ax1.bar(x1, [1], color=bg_color, width=0.6, zorder=0)


# colored bars for proportions
bar_color = [roi_colors['mixed']]
ax1.bar(x1, props_overall, color=bar_color, width=0.5, zorder=1)

# add big fat stars at y=0.7 for ROIs significant vs chance
star_y = 0.7
if res_bin_all_cells.pvalue < 0.05:
    ax1.text(0, star_y, "*", ha="center", va="center", fontsize=16, weight="bold")


ax1.set_xticks(x1)
ax1.set_xticklabels(['All neurons'])
ax1.set_ylim(0, 1.25)
ax1.set_ylabel("Proportion of cells significant\n(p < 0.05)")
ax1.set_title("Overall (all cells)")
ax1.yaxis.grid(True, linestyle="--", alpha=0.3)


# ===== Right panel: by ROI =====
ax2 = axes[1]
x2 = np.arange(len(rois))
bar_width = 0.18


    

for r_i, roi in enumerate(rois):
    # background 100% bar for each ROI
    ax2.bar(r_i, [1] * len(rois), color=bg_color,
            width=0.6, zorder=0)

    # colored bar with actual proportion
    ax2.bar(r_i, props_by_roi[roi], color=roi_colors[roi],
            width=0.5, zorder=1, label=roi)
    
    
# add big fat stars at y=0.7 for ROIs significant vs chance
star_y = 0.7
for roi_i, roi in enumerate(rois):
    if roi_prop_tests.loc[roi_prop_tests.roi == roi, "sig_bonf"].iat[0]:
        ax2.text(roi_i, star_y, "*", ha="center", va="center", fontsize=16, weight="bold")


rows = []
for r1, r2 in combinations(real['roi'].unique(), 2):
    v1 = real.loc[real['roi'] == r1, 'p_perm_F'].dropna()
    v2 = real.loc[real['roi'] == r2, 'p_perm_F'].dropna()
    if len(v1) == 0 or len(v2) == 0:
        t, p = float('nan'), float('nan')
    else:
        t, p = ttest_ind(v1, v2, equal_var=False)   # Welch's t-test
    rows.append((r1, r2, t, p, len(v1), len(v2)))

roi_comp_pairwise = (
    pd.DataFrame(rows, columns=['roi1','roi2','t','p','n1','n2'])
      .assign(n_total=lambda d: d.n1 + d.n2)
      .sort_values('p')
      .reset_index(drop=True)
)
     
num_pairs = len(roi_comp_pairwise)

for roi_i, roi in enumerate(rois):
    # add signficance comparisons, but Bonferroni correct for MC (6 tests)
    sig_pairwise_roi = roi_comp_pairwise[(roi_comp_pairwise['p'] < 0.05/num_pairs) & (roi_comp_pairwise['roi1'] == roi)]
    add_sig_brackets(
        ax2, sig_pairwise_roi,
        xpos = roi_i,
        y0=float(props_by_roi.loc[roi].max()) + 0.045,
        step=0.07, h=0.012, fs=8)
    

ax2.set_xticks(x2)
ax2.set_xticklabels(rois, rotation=45, ha="right")
ax2.set_ylim(0, 1.25)
ax2.set_ylabel("Proportion of cells significant\n(p_perm_F < 0.05)")
ax2.set_title("By ROI")
ax2.yaxis.grid(True, linestyle="--", alpha=0.3)
# ax2.legend(title="Roi", frameon=False)

plt.suptitle(f"Permutation-based significance per ROI, model {model}", fontsize=12)

out_pdf = f"{group_dir_state}/fig_{model}_{trials}.pdf"
out_svg = f"{group_dir_state}/fig_{model}_{trials}.svg"
out_png = f"{group_dir_state}/fig_{model}_{trials}.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight")
print("Saved figure:", out_pdf, "and", out_svg)

plt.show()



### direction of interaction effect.
if model == 'M2_interact':
    sig = (df.query("model == 'M2_interact' and term == 'A_rep' and p_perm_t < 0.05").copy())
    n_pos = (sig["beta"] > 0).sum()
    n_neg = (sig["beta"] < 0).sum()
    
    print(f"Positive (stronger with later repeats): {n_pos}")
    print(f"Negative (weaker with later repeats):  {n_neg}")
    print(f"Fraction positive: {n_pos / (n_pos + n_neg):.2f}")
    
    n_total = n_pos + n_neg
    res_dir = binomtest(n_pos, n_total, p=0.5, alternative="greater")
    print("Directional binomial p-value:", res_dir.pvalue)
    
    
    for roi in rois:
        print(f"now for {roi}")
        sig = (df.query("model == 'M2_interact' and term == 'A_rep' and p_perm_F < 0.05").copy())
        n_pos = (sig[sig['roi'] == roi]["beta"] > 0).sum()
        n_neg = (sig[sig['roi'] == roi]["beta"] < 0).sum()
        
        print(f"Positive (stronger with later repeats): {n_pos}")
        print(f"Negative (weaker with later repeats):  {n_neg}")
        print(f"Fraction positive: {n_pos / (n_pos + n_neg):.2f}")
        
        n_total = n_pos + n_neg
        res_dir = binomtest(n_pos, n_total, p=0.5, alternative="greater")
        print("Directional binomial p-value:", res_dir.pvalue)
        

