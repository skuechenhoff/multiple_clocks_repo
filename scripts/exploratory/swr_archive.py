#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Every archived SWR analysis. One entry point.

    python scripts/exploratory/swr_archive.py              # list them
    python scripts/exploratory/swr_archive.py phase_sweep
    python scripts/exploratory/swr_archive.py roi_compare

⚠ NOTHING HERE IS IN THE REPORTED RESULT. That is `scripts/swr_main.py`.
Everything below was run to find out whether something was there; most of it
said no, and several reached conclusions that were later corrected. Read the
CHANGELOG entry for the date in a script's output directory before reusing
anything.

WHY KEEP THEM AT ALL
    Two reasons. Some are the EVIDENCE for a choice the reported analysis makes
    silently -- `phase_sweep` is why the phase boundary sits where it does, and
    why the valence x phase interaction is reported against the event's own
    baseline rather than the stillness-matched control. Others are the record
    of a question a reviewer is likely to ask, with the answer already
    computed.

The analyses stay in their own files rather than being merged: they have their
own constants (`SEED`, `N_PERM`, `MIN_RIPPLES`, `ROI_SETS` all differ) and one
namespace would let them redefine each other. This file is the index.

@author: Svenja Kuchenhoff
"""

import os
import sys
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# name -> (module, question it answered, what came of it)
ANALYSES = {
    "phase_sweep": (
        "swr_robustness_grid",
        "ripple rate under every phase boundary x grouping x reference "
        "(2 x 3 x 5, 51 tests)",
        "CHOSE the phase definition now used, and is the evidence that the "
        "valence x phase interaction does not survive the stillness control"),
    "phase_hfb": (
        "swr_hfb_phase_robustness",
        "the same question on the HFB branch",
        "negative: no stillness control is available there -- >=2 s still "
        "periods with >=15 clean ripples exist in only 17/53 sessions"),
    "roi_conditions": (
        "swr_hfb_roi_conditions",
        "coupling per ROI x condition x phase",
        "superseded by the GLM, which puts the conditions in one model"),
    "roi_compare": (
        "swr_hfb_roi_compare",
        "paired vs unpaired vs mixed-model ROI comparison; noise diagnostics",
        "a methods check, not a result"),
    "count_control": (
        "swr_hfb_ripple_count_control",
        "do unequal ripple counts across phases bias the phase contrasts?",
        "no bias, but a real power cost; no phase contrast is reported"),
    "event_latency": (
        "swr_hfb_event_latency",
        "when do events happen relative to their ripple?",
        "its headline was an artefact of event spacing -- CHANGELOG 2026-09-29 (f)"),
    "event_specificity": (
        "swr_hfb_event_specificity",
        "does the event response survive in ripple-free epochs?",
        "the rip/clean labels overlapped by 41%; superseded by the GLM"),
    "ripple_vs_event": (
        "swr_hfb_ripple_vs_event",
        "ripple-locking vs event-locking as a plane, sliding windows",
        "the clearest view of the ROI timing difference, but nothing from it "
        "is reported"),
}


def listing():
    print("\n  ⚠ ARCHIVED -- none of this is in the reported result.")
    print("    The reported result is: python scripts/swr_main.py\n")
    for name, (mod, q, outcome) in ANALYSES.items():
        print(f"  {name}")
        print(f"      module   scripts/exploratory/{mod}.py")
        print(f"      asked    {q}")
        print(f"      outcome  {outcome}")
    print("\n  python scripts/exploratory/swr_archive.py <name> [--arg=...]\n")


def _run_one(name, **kw):
    mod_name, q, outcome = ANALYSES[name]
    mod = importlib.import_module("scripts.exploratory." + mod_name)
    print(f"\n{'=' * 72}\n  ⚠ ARCHIVED: {name}\n  asked:   {q}\n"
          f"  outcome: {outcome}\n  module:  scripts/exploratory/{mod_name}.py\n"
          f"{'=' * 72}")
    return mod.run(**kw)


if __name__ == "__main__":
    try:
        import fire
    except ImportError:
        listing(); raise SystemExit
    verbs = {name: (lambda n=name: (lambda **kw: _run_one(n, **kw)))()
             for name in ANALYSES}
    verbs["list"] = listing
    if len(sys.argv) == 1:
        listing()
    else:
        fire.Fire(verbs)
