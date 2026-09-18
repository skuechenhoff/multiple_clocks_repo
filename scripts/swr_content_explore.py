#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Every exploratory ripple-content analysis. One entry point.

    python scripts/swr_content_explore.py                 # list them
    python scripts/swr_content_explore.py i11
    python scripts/swr_content_explore.py roles

⚠ NONE OF THIS IS IN THE REPORTED RESULT. That is `swr_content_main.py`, which
supports one claim: location information is present in and around hippocampal
ripples and is not ripple-specific. Everything here was run to find out whether
ripples carry anything MORE than that. With one exception they did not, and the
exception failed a robustness check -- the verdicts are in the table below and
the detail is in `data/final_results/ripple_analysis/`.

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
