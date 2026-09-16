#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What has finished, what failed, and what the numbers say -- across every stage.

`swr_check_inputs.py` answers "can I start?". This answers "did it work?", which
on a cluster is the question you actually have, and answering it with a pile of
`ls | wc -l` and `grep Traceback` misses the failures that do not raise.

Reports per stage: how many sessions finished, which are missing, and the one
or two numbers that say whether the output is sane. Reads the outputs
themselves, not the SLURM logs, so it is correct even if logs were rotated --
though it scans the logs too, for the failures that never wrote an output.

Usage:
    python scripts/swr_check_status.py
    python scripts/swr_check_status.py --analysis_name=swr_v2
    python scripts/swr_check_status.py --analysis_name=swr_v2 --verbose=True

@author: Svenja Kuchenhoff
"""

import os
import re
import sys
import glob
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

OK, WARN, BAD = "[  OK  ]", "[ WARN ]", "[ FAIL ]"


def _sessions_with(pattern, R):
    out = []
    for p in glob.glob(pattern):
        parts = p.split(os.sep)
        for x in parts:
            if x.startswith("s") and x[1:].isdigit():
                out.append(int(x[1:]))
                break
    return sorted(set(out))


def _fmt(missing, limit=12):
    if not missing:
        return ""
    s = ", ".join(f"s{m:02d}" for m in missing[:limit])
    return s + (f" ... (+{len(missing) - limit})" if len(missing) > limit else "")


def check(analysis_name="swr_v2", verbose=False, data_root=None,
          bundle_dir=None):
    R = data_root or swr_io.get_data_root()
    D = swr_io.derivatives_dir(R)
    G = os.path.join(D, "group", "swr")
    print(f"\ndata root: {R}\nanalysis : {analysis_name}")
    print("=" * 74)

    mf_p = os.path.join(G, "session_manifest.csv")
    if not os.path.isfile(mf_p):
        print(f"{BAD} no session_manifest.csv -- run swr_audit_sessions.py first")
        return None
    mf = pd.read_csv(mf_p)
    expect = sorted(int(s) for s in mf.loc[mf.n_raw_files > 0, "session"])
    print(f"sessions with raw data: {len(expect)}")

    # ---- stage 1: contacts -------------------------------------------------
    got = _sessions_with(os.path.join(D, "s*", "LFP", "bipolar_pairs_*.csv"), R)
    n_cx, n_stale = 0, []
    for s in got:
        f = os.path.join(swr_io.session_deriv_dir(s, R), "LFP",
                         f"bipolar_pairs_{s:02d}.csv")
        try:
            t = pd.read_csv(f)
        except Exception:
            continue
        if "role" not in t.columns:
            n_stale.append(s)
        else:
            n_cx += int((t.role == "hfb").sum())
    # NOT a ratio against `expect`: stage 1 reads channel lists, which can be
    # present for sessions whose raw LFP is not on this machine, so the count
    # legitimately exceeds the raw-data count and a ratio reads as >100%.
    print(f"\n{OK} stage 1  contacts      : {len(got)} sessions with pairs, "
          f"{n_cx} cortical derivations")
    if n_stale:
        print(f"{BAD}          {len(n_stale)} pre-cortical (no `role`): {_fmt(n_stale)}")
        print("           -> re-run swr_build_contacts.py")

    # ---- stage 2: extraction ----------------------------------------------
    got2 = _sessions_with(os.path.join(D, "s*", "LFP-clean", analysis_name,
                                       "continuous.npy"), R)
    skipped = _sessions_with(os.path.join(D, "s*", "LFP-clean", analysis_name,
                                          "SKIPPED.json"), R)
    miss = [s for s in got if s not in got2 and s not in skipped]
    tag = OK if not miss else WARN
    print(f"\n{tag} stage 2  continuous    : {len(got2)}/{len(got)} sessions"
          + (f", {len(skipped)} deliberate skips" if skipped else ""))
    if miss:
        print(f"           missing: {_fmt(miss)}")

    # ---- stage 3: detection ------------------------------------------------
    got3 = _sessions_with(os.path.join(D, "s*", "LFP-ripples", analysis_name,
                                       "ripple_events.csv"), R)
    rates, pads, near = [], set(), []
    for s in got3:
        d = os.path.join(swr_io.session_deriv_dir(s, R), "LFP-ripples", analysis_name)
        try:
            q = pd.read_csv(os.path.join(d, "channel_qc.csv"))
            good = q[~q.excluded.fillna(False)]
            rates.extend(good.rate_hz.dropna().tolist())
            if "frac_near_artifact" in good:
                near.extend(good.frac_near_artifact.dropna().tolist())
            st = json.load(open(os.path.join(d, "settings.json")))
            pads.add(st.get("artifact_pad_s"))
        except Exception:
            continue
    miss = [s for s in got2 if s not in got3]
    tag = OK if not miss else WARN
    print(f"\n{tag} stage 3  ripples       : {len(got3)}/{len(got2)} sessions")
    if miss:
        print(f"           missing: {_fmt(miss)}")
    if rates:
        r = np.array(rates)
        ok_rate = 0.05 <= np.median(r) <= 0.60
        print(f"  {OK if ok_rate else BAD}   rate median {np.median(r):.3f} Hz "
              f"(IQR {np.percentile(r,25):.3f}-{np.percentile(r,75):.3f}) "
              f"[Chen 0.17-0.24; 0.05-0.60 tolerated]")
        if near:
            print(f"           {100*np.median(near):.0f}% of events within 1 s of a "
                  f"crossing (the ones a 1 s pad would drop)")
        print(f"           artifact_pad_s used: {sorted(p for p in pads if p is not None)}")

    # ---- stage 3b: HFB -----------------------------------------------------
    got4 = _sessions_with(os.path.join(D, "s*", "LFP-hfb", analysis_name,
                                       "hfb.npz"), R)
    pos_ok, pos_tot, no_ctrl, lost_bands, n_mpfc, mb = 0, 0, [], 0, 0, 0.0
    for s in got4:
        d = os.path.join(swr_io.session_deriv_dir(s, R), "LFP-hfb", analysis_name)
        mb += os.path.getsize(os.path.join(d, "hfb.npz")) / 1e6
        try:
            q = pd.read_csv(os.path.join(d, "hfb_pairs.csv"))
            n_mpfc += int(q.roi_family.isin(["mPFC", "mOFC"]).sum())
            lost_bands += int((q.n_sub_bands < 8).sum())
            m = json.load(open(os.path.join(d, "meta.json")))
            peri = m.get("peri_ripple_qc")
            if not peri:
                no_ctrl.append(s)
                continue
            hc = [v for v in peri.values() if v.get("role") == "ripple"]
            pos_tot += len(hc)
            pos_ok += sum(1 for v in hc if v.get("peri_minus_flank", 0) > 0)
        except Exception:
            continue
    miss = [s for s in got2 if s not in got4]
    tag = OK if not miss else WARN
    print(f"\n{tag} stage 3b HFB           : {len(got4)}/{len(got2)} sessions, "
          f"{n_mpfc} mPFC/mOFC derivations, {mb/1000:.1f} GB")
    if miss:
        print(f"           missing: {_fmt(miss)}")
    if lost_bands:
        print(f"           {lost_bands} derivations lost sub-bands to the notch "
              f"(expected; n_sub_bands is a covariate)")
    if pos_tot:
        frac = pos_ok / pos_tot
        tag = OK if frac >= 0.8 else (WARN if frac >= 0.6 else BAD)
        print(f"  {tag}   POSITIVE CONTROL: {pos_ok}/{pos_tot} hippocampal "
              f"derivations show peri-ripple HFB > flanks ({100*frac:.0f}%)")
        if frac < 0.8:
            print("           ⛔ below 80% -- check the LFP/event clocks before "
                  "trusting ANY cortical number")
    if no_ctrl:
        print(f"           {len(no_ctrl)} session(s) have no positive control: "
              f"{_fmt(no_ctrl)}")
        print("           -> re-run stage 3b AFTER detection, to get it computed")

    # ---- bundle ------------------------------------------------------------
    # Look for a bundle belonging to THIS analysis first. Exporting swr_v2 to
    # `bundle_v2` (so the swr_v1 bundle survives) otherwise leaves this line
    # reporting the old bundle -- "0 with HFB, re-paddable NO" -- for a run that
    # is complete and fine.
    # Glob rather than guess names: bundles get renamed by hand
    # (`bundle_08.09.2026`, `bundle_old_from_cluster`, `bundle_v2`), so a fixed
    # candidate list finds nothing and reports "not exported yet" for a run that
    # is finished. Prefer one whose meta matches THIS analysis; otherwise show
    # the most recent and say it is a different analysis.
    bp, bname = None, None
    found = []
    for q in sorted(glob.glob(os.path.join(G, "*", "swr_bundle.pkl"))):
        try:
            import pickle as _pk
            an = _pk.load(open(q, "rb"))["meta"].get("analysis_name")
        except Exception:
            an = None
        found.append((q, os.path.basename(os.path.dirname(q)), an,
                      os.path.getmtime(q)))
    if bundle_dir:
        found = [f for f in found if f[1] == bundle_dir]
    match = [f for f in found if f[2] == analysis_name]
    pick = (sorted(match, key=lambda f: -f[3]) or
            sorted(found, key=lambda f: -f[3]))
    if pick:
        bp, bname = pick[0][0], pick[0][1]
    if len(found) > 1:
        print(f"\n           ({len(found)} bundles on disk: "
              + ", ".join(f"{f[1]}[{f[2]}]" for f in found) + ")")
    if bp:
        import pickle
        b = pickle.load(open(bp, "rb"))
        has_repad = ("dist_to_artifact_s" in b["ripples"].columns
                     and len(b.get("artifact_intervals", [])) > 0)
        stale = b["meta"].get("analysis_name") != analysis_name
        print(f"\n{WARN if stale else OK} bundle ({bname:<12s}): "
              f"{len(b['ripples'])} ripples, "
              f"{b['meta'].get('n_sessions')} sessions, "
              f"{len(b.get('hfb_index', []))} with HFB")
        if stale:
            print(f"           ⚠ this bundle is analysis_name="
                  f"{b['meta'].get('analysis_name')!r}, not {analysis_name!r} "
                  f"-- it has not been exported yet")
        print(f"           re-paddable on the laptop: "
              f"{'YES' if has_repad else 'NO -- rebuild with the current code'}")
    else:
        print(f"\n{WARN} bundle                 : not exported yet")

    # ---- SLURM failures that never wrote an output -------------------------
    errs = glob.glob(os.path.join(G, "slurm_logs", "*", "*.err"))
    bad = [e for e in errs if os.path.getsize(e) > 0
           and "Traceback" in open(e, errors="ignore").read()]
    # `slurm_logs/` accumulates every run ever. A traceback from three weeks ago
    # is not a failure of the run that just finished, and reporting it as one
    # sends you hunting a bug that was fixed long since. Anything older than the
    # newest stage output cannot have produced that output.
    newest_out = 0.0
    for pat in (("s*", "LFP-hfb", analysis_name, "hfb.npz"),
                ("s*", "LFP-ripples", analysis_name, "ripple_events.csv")):
        for f in glob.glob(os.path.join(D, *pat)):
            newest_out = max(newest_out, os.path.getmtime(f))
    fresh = [e for e in bad if os.path.getmtime(e) >= newest_out - 86400]
    old_ones = [e for e in bad if e not in fresh]

    tag = BAD if fresh else (OK if not bad else WARN)
    print(f"\n{tag} slurm .err with a Traceback: {len(bad)}/{len(errs)}"
          + (f"  ({len(fresh)} from this run, {len(old_ones)} older)"
             if bad else ""))
    if old_ones and not fresh:
        print(f"           all {len(old_ones)} predate the current outputs "
              f"-- earlier runs, not this one")

    def _cause(path):
        """The exception line, not the last line -- a traceback often ends in a
        pandas repr fragment like '[1 rows x 12 columns]', which says nothing."""
        lines = [l.rstrip() for l in open(path, errors="ignore").read().split("\n") if l.strip()]
        for l in reversed(lines):
            if re.match(r"^\s*[\w.]*(Error|Exception|Warning)\b", l.strip()):
                return l.strip()
        return lines[-1] if lines else ""

    show = (fresh or old_ones) if verbose else (fresh or old_ones)[:5]
    for e in show:
        job = os.path.basename(os.path.dirname(e))
        when = job.split("_")[0] if "_" in job else "?"
        mark = "NEW " if e in fresh else "old "
        print(f"           {mark}[{when}] {os.path.basename(e)}: {_cause(e)[:80]}")
    if len(fresh or old_ones) > 5 and not verbose:
        print("           (--verbose=True for all)")
    print("\n" + "=" * 74)
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(check)
    else:
        check()
