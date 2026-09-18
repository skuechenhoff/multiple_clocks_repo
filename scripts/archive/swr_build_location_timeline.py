#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 0 of the ripple-content analysis: a real-time location timeline.

    python scripts/swr_build_location_timeline.py build
    python scripts/swr_build_location_timeline.py validate

`build` extracts one step table per session from `abcd_passed.mat` (see
`mc.analyse.swr_location` for why that is the only correct source) and caches it
under `derivatives/group/swr/location_timeline/`.

`validate` is the gate. Every uncover event in the SWR bundle has a known
location; the timeline must independently reproduce it. Anything below ~100%
means the clocks or the step indexing are wrong and nothing downstream is
trustworthy. For reference, integrating arrow presses from `press_categories.csv`
manages 62.9%.

@author: Svenja Kuchenhoff
"""

import sys
import numpy as np
import pandas as pd

import mc.analyse.swr_io as swr_io
import mc.analyse.swr_location as swl


def cmd_build(sessions=None):
    paths = swl.build(sessions)
    tot = 0
    for s, p in sorted(paths.items()):
        try:
            tot += len(pd.read_csv(p))
        except Exception:
            pass
    print(f"\n{len(paths)} session(s) cached, {tot:,} steps total")
    print(f"  -> {swl.cache_dir()}")


def cmd_validate(sessions=None):
    bundle = swr_io.default_bundle_dir() if hasattr(swr_io, "default_bundle_dir") \
        else None
    import os
    deriv = swr_io.derivatives_dir()
    unc = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "uncover.csv"))
    steps = swl.load(sessions)
    if not len(steps):
        print("no timeline cached -- run `build` first")
        return

    rows = []
    for s, u in unc.groupby("session"):
        st = steps[steps.session == s]
        if not len(st):
            continue
        pred = swl.location_at(st, u.t_s.to_numpy(float))
        true = u["loc"].to_numpy(float)
        ok = np.isfinite(pred)
        rows.append(dict(session=s, n=len(u), covered=int(ok.sum()),
                         match=int((pred[ok] == true[ok]).sum())))
    R = pd.DataFrame(rows)
    n, cov, mat = R.n.sum(), R.covered.sum(), R.match.sum()
    print(f"\nuncover events checked : {n:,}  in {len(R)} sessions")
    print(f"  inside the timeline   : {cov:,} ({cov/n:.1%})")
    print(f"  location reproduced   : {mat:,} ({mat/max(cov,1):.1%} of covered)")

    R["acc"] = R.match / R.covered.clip(lower=1)
    bad = R[R.acc < 0.99].sort_values("acc")
    if len(bad):
        print(f"\n{len(bad)} session(s) below 99%:")
        print(bad[["session", "n", "covered", "match", "acc"]].to_string(index=False))
    else:
        print("\nevery session at or above 99%.")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "build"
    sess = [int(x) for x in sys.argv[2:]] or None
    {"build": cmd_build, "validate": cmd_validate}[what](sess)
