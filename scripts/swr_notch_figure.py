#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods figure: the adaptive notch, across recording sites.

Line noise in this dataset is a property of the recording centre, not of the
patient -- Baylor is essentially clean, UCLA is cleaned by the bipolar montage
alone, Utah is contaminated by orders of magnitude even after it. A blanket
notch would therefore remove real 120 Hz signal, which is the upper edge of the
ripple band, from two sites out of three for no benefit. This figure is the
evidence for that claim: before/after spectra for a spread of sessions, with the
measured peak-to-flank ratio printed at every harmonic so the reader can see
which decisions the threshold made and why.

Needs `notch_psd.npz`, written by `swr_extract_continuous.py`. Sessions
extracted before 2026-09-08 do not have it -- see `--list` for who does.

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_notch_figure.py list
    python scripts/swr_notch_figure.py build
    python scripts/swr_notch_figure.py build --sessions="[2,3,5,38]" --ncol=5

@author: Svenja Kuchenhoff
"""

import os
import sys
import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io

try:
    import fire
except ImportError:
    fire = None

ANALYSIS_NAME = "swr_v1"
# Enough panels to show the pattern, few enough to stay legible at 4 cm each.
DEFAULT_N = 10


def _available(analysis_name=ANALYSIS_NAME, data_root=None):
    R = data_root or swr_io.get_data_root()
    out = []
    for p in sorted(glob.glob(os.path.join(
            swr_io.derivatives_dir(R), "s*", "LFP-clean", analysis_name,
            "notch_psd.npz"))):
        try:
            z = np.load(p, allow_pickle=True)
            out.append({"session": int(z["session"]),
                        "recording_site": str(z["recording_site"]),
                        "path": p,
                        "n_applied": int(np.size(z["notch_applied_hz"])),
                        "max_ratio": float(np.nanmax(z["ratios_per_pair"]))
                                     if "ratios_per_pair" in z.files
                                     else (float(np.max(z["ratios"][:, 1]))
                                           if np.size(z["ratios"]) else np.nan)})
        except Exception as e:
            print(f"  [unreadable] {p}: {type(e).__name__}: {e}")
    return out


def list(analysis_name=ANALYSIS_NAME):
    """Which sessions have the before/after spectra, and what they show."""
    av = _available(analysis_name)
    if not av:
        print("No notch_psd.npz anywhere. Re-run swr_extract_continuous.py "
              "-- see the module docstring.")
        return None
    by_site = {}
    for a in av:
        by_site.setdefault(a["recording_site"], []).append(a)
    print(f"{len(av)} sessions with before/after spectra\n")
    for site, rows in sorted(by_site.items()):
        print(f"  {site:8s} {len(rows):3d}: "
              + ", ".join(f"s{r['session']:02d}"
                          f"({'notched' if r['n_applied'] else 'clean'})"
                          for r in sorted(rows, key=lambda r: r["session"])))
    return None


def _pick(av, n=DEFAULT_N, sessions=None):
    """A spread across sites, and within a site across notch outcomes.

    Taking the first n sessions would give ten Baylor panels that all say
    'none needed', which shows nothing. The figure has to contain both
    outcomes at more than one site to make its point.
    """
    if sessions is not None:
        want = [int(s) for s in sessions]
        return [a for s in want for a in av if a["session"] == s]
    by_site = {}
    for a in av:
        by_site.setdefault(a["recording_site"], []).append(a)
    picked, sites = [], sorted(by_site, key=lambda k: -len(by_site[k]))
    # round-robin over sites, alternating notched / not-notched within each
    pools = {}
    for site in sites:
        rows = sorted(by_site[site], key=lambda r: r["session"])
        pools[site] = ([r for r in rows if r["n_applied"]],
                       [r for r in rows if not r["n_applied"]])
    turn = 0
    while len(picked) < n and any(any(p) for p in pools.values()):
        for site in sites:
            hit, clean = pools[site]
            pool = (hit if (turn % 2 == 0 and hit) else (clean or hit))
            if not pool:
                continue
            picked.append(pool.pop(0))
            if len(picked) >= n:
                break
        turn += 1
    return picked


def build(sessions=None, n=DEFAULT_N, ncol=5, analysis_name=ANALYSIS_NAME,
          out_dir=None, panel_cm=(4.0, 3.4)):
    """Draw the gallery. PDF at print size -- 4 x 3 cm a panel, 9 pt."""
    import matplotlib
    matplotlib.use("Agg")
    import mc.plotting.ripple_figures as rfig

    R = swr_io.get_data_root()
    av = _available(analysis_name, R)
    if not av:
        print("No notch_psd.npz found -- re-run swr_extract_continuous.py.")
        return None
    picked = _pick(av, n=n, sessions=sessions)
    if not picked:
        print("Nothing selected."); return None

    entries = []
    for a in picked:
        z = np.load(a["path"], allow_pickle=True)
        want = ("freq", "psd_before", "psd_after", "ratios",
                "notch_applied_hz", "session", "recording_site",
                "ratios_per_pair", "n_notched", "n_pairs")
        entries.append({k: z[k] for k in want if k in z.files})

    out_dir = out_dir or os.path.join(swr_io.derivatives_dir(R), "group",
                                      "swr", "figures")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "notch_gallery")
    rfig.notch_gallery_figure(entries, out_stem=stem, ncol=ncol,
                              panel_cm=tuple(panel_cm))
    sites = {}
    for a in picked:
        sites[a["recording_site"]] = sites.get(a["recording_site"], 0) + 1
    print(f"{len(entries)} panels: " + ", ".join(f"{k} {v}" for k, v in sites.items()))
    print(f"  notched: {sum(1 for a in picked if a['n_applied'])}, "
          f"left alone: {sum(1 for a in picked if not a['n_applied'])}")
    print(f"-> {stem}.pdf / .png")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"list": list, "build": build})
    else:
        build()
