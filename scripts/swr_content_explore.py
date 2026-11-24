#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Every exploratory ripple-content analysis. One entry point.

    python scripts/swr_content_explore.py                 # list them
    python scripts/swr_content_explore.py i11
    python scripts/swr_content_explore.py roles

⚠ NONE OF THIS IS IN THE REPORTED RESULT. That is `swr_content_main.py --step=phase`
(`mc/analyse/swr_phase_contrast.py`), which supports one claim: during ENCODING
hippocampal ripples carry the current square, this is not detectable once the
sequence is known, the phase difference survives matching ripple counts, and
the signal is NOT ripple-specific against a matched flank.

Everything here was run to find out whether ripples carry anything MORE than
that. With one exception they did not, and the exception failed a robustness
check -- the verdicts are in the table below, the numbers in
`data/final_results/ripple_analysis/RESULTS.md`.

The analyses live in `mc/analyse/swr_explore/`, one module each, rather than
merged into this file. They were written separately, have their own constants
(`SEED`, `N_PERM`, `ROI_SETS`, `MIN_RIPPLES` all differ between them) and their
own outputs; concatenating them into one namespace would let one analysis's
constant silently redefine another's. This file is the index and the dispatcher.

@author: Svenja Kuchenhoff
"""

import sys
import importlib

# name -> (module, question it answers, what it found)
ANALYSES = {
    "descriptives": (
        "content_descriptives",
        "spike and ripple budget, peri-ripple spike histograms",
        "95.7% of ripples are in-task with a known square"),
    "spike_rate": (
        "spike_rate_in_ripples",
        "is firing higher inside ripples than outside?",
        "yes but only ~4% (HC_ant 2.84 vs 2.65 Hz), far below rodent values"),
    "profile": (
        "ripple_profile",
        "the full 9-square profile by step distance, ripple vs flank",
        "a real spatial gradient, IDENTICAL in ripple and flank"),
    "i11": (
        "ripple_content_i11",
        "rewarded vs non-rewarded squares during exploration",
        "null (+0.241, p = 0.29), with stillness matched by triplets"),
    "roles": (
        "ripple_content_roles",
        "the 12-term role regression: goals, route, errors, adjacency",
        "only `on_route` in HC_mid (+0.530, p = 0.0087); see collinearity"),
    "collinearity": (
        "roles_collinearity",
        "are the role regressors collinear, and is the route effect stable?",
        "VIFs 1.3-2.3; stable to dropping any rival except `recent`"),
    "roles_timecourse": (
        "roles_timecourse",
        "descriptive: evidence by square role across the peri-ripple second",
        "flat; the regression effects are partial coefficients"),
    "pseudopop": (
        "pseudo_population",
        "pool cells across sessions that ran the SAME configuration",
        "signal is distributed over many cells, still not ripple-specific"),
    "pseudo_timecourse": (
        "pseudo_timecourse",
        "pseudo-population peri-ripple time course",
        "superseded by swr_content_main.py timecourse (session-level)"),
    "decoders": (
        "decoder_comparison",
        "can a 9-way location decoder beat chance on ANY signal?",
        "no -- HFB, theta, beta, ripple band and spikes all at chance"),
    "ieeg_decoder": (
        "ieeg_location_decoder",
        "the iEEG decoder gate for sequence analysis, with positive control",
        "gate CLOSED; run the positive control before believing any null"),
    "state": (
        "state_content",
        "do ripples carry STATE (A/B/C/D), and is it factorised from location?",
        "see output -- drift-controlled and time-matched variants reported"),
    "compositional": (
        "compositional",
        "ordered state structure across consecutive ripples; does HC lead mPFC?",
        "see output -- the two compositional tests this dataset can support"),
    "phase_splits": (
        "location_by_phase",
        "location in ripples split by encode/execute and by what the square is",
        "see output -- C0 and C1 per phase and subtype"),
    "goal_lag": (
        "goal_lag",
        "TEST b: do ripples carry the upcoming GOAL SEQUENCE (rewards ahead)?",
        "running -- coarse real-time plan lag, mPFC/mOFC/HC"),
    "warped_lag": (
        "warped_lag",
        "TEST c: the 12-bin future-lag code measured with ripple spikes only",
        "running -- template from all data, measurement from ripples"),
    "loading": (
        "loading",
        "TEST L1: is the plan MORE represented AFTER a ripple than before?",
        "running -- paired pre/post inside one subgoal; sham + current controls"),
    "discovery_lag": (
        "discovery_lag",
        "SK's model: rewards indexed by how many DISCOVERIES ago they were found",
        "running -- only already-known squares are tested"),
    "intention_ripple": (
        "intention_ripple",
        "lock to UNCOVERS: does pre-uncover mPFC intention predict ripples after?",
        "running -- SK's event-locked idea"),
    "consolidation": (
        "consolidation",
        "ripples as STORAGE: is the just-uncovered square represented after it?",
        "running -- split by whether a ripple followed the uncover"),
    "discovery_locked": (
        "discovery_locked",
        "does the newly-discovered-reward signal change AROUND a ripple?",
        "running -- time-resolved follow-up to `consolidation`"),
    "method_check": (
        "method_check",
        "VALIDATION: does the estimator recover HC current location OUTSIDE ripples?",
        "running -- if this fails, every null in this project is uninterpretable"),
    "preferred_lag": (
        "preferred_lag_firing",
        "do cells TAGGED lag-0 fire at their square? (per-lag encoding tags)",
        "running -- LOO preferred square, all cells, no significance filter"),
    "lag_group": (
        "lag_group_ripple",
        "COLLAPSED: are lag-0 cells more ripple-driven than future-lag cells?",
        "running -- no preferred location needed, so far better powered"),
    "storage_planning": (
        "storage_vs_planning",
        "do lag-0 cells fire at STORAGE and future-lag cells at PLANNING?",
        "running -- continuous lag strength, all cells"),
    "cross_task": (
        "cross_task",
        "WHY NEGATIVE? is the place template transferable across configurations?",
        "running -- tests SK's lag-divergence hypothesis directly"),
    "template_quality": (
        "template_quality",
        "is the mPFC new-reward effect carried by a transferring MINORITY?",
        "running -- per-cell transferability distribution + median split"),
    "future_shift": (
        "future_shift",
        "across a ripple, does activity move from NOW to the UPCOMING squares? "
        "(same-block references)",
        "NULL (mPFC -0.04, P = .86; HFB +0.07, P = .57); both fail the gate, "
        "HC passes and shows no shift"),
    "ripple_templates": (
        "ripple_templates",
        "is mPFC's 'now' code visible when its profile is built IN ripples?",
        "NULL in mPFC (-0.08, P = .81); HC gate passes; mOFC ripple-only "
        "template +0.52, P = .047 (exploratory)"),
    "path_content": (
        "path_content",
        "after a correct uncover, do HC ripples carry the path just walked "
        "or the path ahead? (T4, CHANGELOG 2026-10-29)",
        "HC first traversal PAST-FUT +0.44, P = .043 (storage side); flips to "
        "FUT +0.35, P = .006 in execution. ⚠ same-block templates are mostly "
        "execution maps for learning events (median 87% for learning "
        "ripples): the phase flip may be the template (CHANGELOG 2026-11-20)"),
    "discovery_cells": (
        "discovery_cells",
        "do lag-0 cells fire at the moment of discovery, reward-locked, and "
        "does that depend on depth / ripples? (N1, N2; CHANGELOG 2026-10-30)",
        "NULL for lag-0 (own p = .12, wrong sign); mPFC new-square +0.66 "
        "reward-locked but gone when stillness-matched (p = .47)"),
    "context_content": (
        "context_content",
        "D3: rewarded squares in ripples away from uncovers; D4: cortex <-> HC "
        "position-in-sequence direction, binding, code timeline "
        "(CHANGELOG 2026-11-15)",
        "D4a reversed; D4b null; D5 (before ripples) null. D4c timeline "
        "retracted (CHANGELOG 2026-11-19). ⚠ D3 uses same-block templates "
        "and is NOT location content; it reverses the reported phase "
        "contrast (CHANGELOG 2026-11-20, "
        "scripts/exploratory/location_estimator_reconcile.py)"),
    "reactivation": (
        "reactivation",
        "do learning ripples replay the co-firing of the block's later "
        "execution? (C1, Peyrache reactivation; CHANGELOG 2026-11-14 (b))",
        "NULL incl. the HC positive control (insensitive, not negative): mPFC "
        "specific matched +0.0006, P = .52"),
    "lag0_cells": (
        "lag0_cells",
        "are lag-0 cells ripple-locked, and do they carry the square at "
        "discovery? claim 2 a/b in lag-0 cells (F1, F3; CHANGELOG 2026-11-14)",
        "NULL: no lag-0 x ripple firing (|rho| <= .08), no lag-0 x discovered-"
        "square evidence; per-cell lag-0 r has ~0 split-half reliability (F2)"),
    "phase_timecourse": (
        "timecourse_by_phase",
        "peri-ripple location TIME COURSE, encode vs execute (descriptive)",
        "mHP encode cluster -100..+75 ms p=0.005; see swr_content_main for the "
        "reported contrast"),
    "location_timecourse": (
        "location_timecourse",
        "arrival-locked location time course",
        "⚠ role labels expire after ~0.44 s; only |t| < 0.25 s is meaningful"),
}


def _list():
    print(__doc__.split("@author")[0])
    print(f"{'analysis':20s} {'question':58s} finding")
    print("-" * 150)
    for name, (_, q, f) in ANALYSES.items():
        print(f"{name:20s} {q:58s} {f}")
    print("\nrun one with:  python scripts/swr_content_explore.py <analysis>")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "list"):
        _list()
        return
    name = argv[0]
    if name not in ANALYSES:
        print(f"unknown analysis {name!r}\n")
        _list()
        return
    mod = importlib.import_module("mc.analyse.swr_explore."
                                  + ANALYSES[name][0])
    print(f"=== {name}: {ANALYSES[name][1]} ===\n")
    mod.main()


if __name__ == "__main__":
    main()
