#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The reported SWR result. One entry point.

    python scripts/swr_main.py                  # list the steps
    python scripts/swr_main.py ripple_rate      # run one
    python scripts/swr_main.py all              # run them in order
    python scripts/swr_main.py order            # just print the order

★ EVERY NUMBER AND EVERY PANEL IN THE MANUSCRIPT COMES FROM A STEP BELOW.
Anything not listed here is exploratory, superseded, or infrastructure --
see `SCRIPTS_GUIDE_swr_ripple_rate.md`, `SCRIPTS_GUIDE_swr_content.md` and
`scripts/exploratory/README.md`.

WHY A DISPATCHER RATHER THAN ONE MERGED FILE
    The steps were written separately and have their own constants -- `SEED`,
    `MIN_RIPPLES`, `N_PERM`, `ROI_SETS` and the smoothing kernel all differ
    between them, deliberately. Concatenating them into one namespace would let
    one step's constant silently redefine another's, which is exactly the class
    of bug this branch has already been bitten by. This file is the index, the
    documented order, and the wiring; the analyses stay where they are.

WIRING
    Two steps consume the output directory of an earlier one. Rather than make
    the user paste paths, `all` resolves them to the NEWEST matching directory
    that is not marked `SUPERSEDED.md`, so a stale run cannot be picked up
    silently.

@author: Svenja Kuchenhoff
"""

import os
import sys
import glob
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io

# name -> what it is. `wires` maps a run() argument to a glob of directory
# names under the swr group folder; `all` fills it with the newest clean match.
STEPS = {
    "coverage": dict(
        module="swr_hfb_coverage_figure",
        produces="contact coverage; Euclidean distance to the nearest HC derivation",
        figure="Fig 4e",
        note="also gives the 27.9 / 33.6 / 63.1 mm numbers quoted in the results"),
    "ripple_rate": dict(
        module="swr_figure_phase_panels",
        produces="ripple rate by phase and feedback, both references, with clusters",
        figure="Fig 4f, 4g",
        note="THE ripple-rate result. encoding = rep < first correct repeat"),
    "stillness": dict(
        module="swr_stillness_by_phase",
        produces="post-uncover stillness by phase and feedback",
        figure="supplement",
        note="supporting: justifies the stillness-matched control"),
    "content": dict(
        module="swr_content_main",
        produces="location information in and around ripples",
        figure="Fig 4h",
        note="its own dispatcher; --step=phase is the reported analysis"),
    "hfb_roi": dict(
        module="swr_ripple_locked_hfb",
        produces="ripple-locked HFB per ROI, against the shifted null",
        figure="Fig 4i",
        note="medial frontal +0.0073; auditory and lateral temporal controls"),
    "hfb_conditions": dict(
        module="swr_ripple_hfb_conditions",
        produces="ripple-locked HFB by condition and phase",
        figure="Fig 4j",
        note="takes --boundary; defaults to the agreed definition"),
    "hfb_partition": dict(
        module="swr_hfb_roi_partition",
        produces="coupling per ROI with each derivation in exactly one ROI",
        figure="table",
        note="--exclude_medial=True gives the disjoint table the paper quotes"),
    "hfb_event": dict(
        module="swr_hfb_event_locked",
        produces="event-locked (not ripple-locked) HFB",
        figure="supplement",
        note="the 'no response to the uncovering itself' number"),
    "hfb_glm": dict(
        module="swr_hfb_glm",
        produces="FIR and single-regressor GLMs separating ripple from event",
        figure="supplement"),
    "hfb_contrasts": dict(
        module="swr_hfb_locking_contrasts",
        produces="the ROI DIFFERENCES, not two separate tests",
        figure="results text",
        wires={"region_dir": "ripple_locked_hfb_*",
               "cond_dir": "ripple_hfb_conditions_*"}),
    "hfb_panels": dict(
        module="swr_hfb_panels_2cm",
        produces="the 2 x 2 cm publication panels",
        figure="final figure assembly",
        wires={"region_dir": "ripple_locked_hfb_*",
               "cond_dir": "ripple_hfb_conditions_*",
               "partition_dir": "hfb_roi_partition_*"}),
}
# Dependencies run before their consumers; `all` uses this order.
ORDER = list(STEPS)


def _group_dir():
    return os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                        "group", "swr")


def _latest(pattern):
    """Newest directory matching `pattern` that is NOT marked superseded."""
    hits = [d for d in glob.glob(os.path.join(_group_dir(), pattern))
            if os.path.isdir(d)
            and not os.path.exists(os.path.join(d, "SUPERSEDED.md"))]
    return max(hits, key=os.path.getmtime) if hits else None


def _resolve(spec):
    """Fill a step's wired directory arguments, or explain what is missing."""
    out, missing = {}, []
    for arg, pattern in (spec.get("wires") or {}).items():
        d = _latest(pattern)
        (out.__setitem__(arg, d) if d else missing.append(f"{arg} ({pattern})"))
    return out, missing


def listing():
    print("\n  THE REPORTED SWR RESULT -- every manuscript number comes from "
          "a step here.\n")
    print(f"  {'step':16s} {'figure':12s} produces")
    print(f"  {'-' * 16} {'-' * 12} {'-' * 46}")
    for name, s in STEPS.items():
        print(f"  {name:16s} {s.get('figure', '-'):12s} {s['produces']}")
        if s.get("note"):
            print(f"  {'':16s} {'':12s} -- {s['note']}")
    print("\n  python scripts/swr_main.py <step> [--arg=...]   run one")
    print("  python scripts/swr_main.py all                  run them in order")
    print("  Exploratory work: python scripts/exploratory/swr_archive.py\n")


def order():
    print("\n  Order, and what each step needs:\n")
    for i, name in enumerate(ORDER, 1):
        s = STEPS[name]
        w = ", ".join((s.get("wires") or {}).keys()) or "-"
        print(f"   {i:2d}. {name:16s} needs: {w}")
    print()


def _run_one(name, **kw):
    spec = STEPS[name]
    mod = importlib.import_module("scripts." + spec["module"])
    wired, missing = _resolve(spec)
    if missing:
        raise SystemExit(f"  {name}: cannot resolve {', '.join(missing)}. "
                         f"Run the step that produces it first, or pass it "
                         f"explicitly.")
    wired.update(kw)
    print(f"\n{'=' * 72}\n  {name} -- {spec['produces']}\n"
          f"  module: scripts/{spec['module']}.py"
          + (f"\n  wired: {wired}" if wired else "") + f"\n{'=' * 72}")
    return mod.run(**wired)


def all(**kw):
    """Run every step in order. Per-step arguments are not forwarded."""
    done, failed = [], []
    for name in ORDER:
        try:
            _run_one(name)
            done.append(name)
        except SystemExit as e:
            print(e); failed.append((name, "unresolved input"))
        except Exception as e:
            print(f"  {name} FAILED: {type(e).__name__}: {e}")
            failed.append((name, f"{type(e).__name__}: {e}"))
    print(f"\n  {len(done)}/{len(ORDER)} steps completed: {', '.join(done)}")
    for name, why in failed:
        print(f"  ! {name}: {why}")
    return {"completed": done, "failed": failed}


if __name__ == "__main__":
    try:
        import fire
    except ImportError:
        listing(); raise SystemExit
    verbs = {name: (lambda n=name: (lambda **kw: _run_one(n, **kw)))()
             for name in STEPS}
    verbs.update({"list": listing, "order": order, "all": all})
    if len(sys.argv) == 1:
        listing()
    else:
        fire.Fire(verbs)
