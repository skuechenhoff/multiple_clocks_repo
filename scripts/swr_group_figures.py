#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group figures for the manuscript, built ENTIRELY FROM THE BUNDLE.

`swr_qc_report.py figure` walks `derivatives/s*/` and therefore only ever pools
the sessions whose detection output is on the machine it runs on -- on a laptop
that is a handful. This script reads `bundle/` instead, so the same figures can
be built at full cohort scale wherever the bundle has been downloaded. The
bundle carries condensed waveform arrays for every session (see
`mc.analyse.swr_bundle.collect_figure_data`), which is exactly what these
figures need and a fraction of the size of the recordings.

Verbs:
    chen         pooled ripple-triggered average + TFR + one example (Chen Fig 2)
    attributes   rate / duration / peak frequency / amplitude, pooled
    contacts     hippocampal coverage (delegates to swr_contact_figure)
    artifact     contamination across every derivation
    all          every one of the above

Everything lands in `derivatives/group/swr/figures/`.

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_group_figures.py all
    python scripts/swr_group_figures.py attributes
    python scripts/swr_group_figures.py chen --bundle=<a downloaded bundle dir>

@author: Svenja Kuchenhoff
"""

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc.analyse.swr_io as swr_io

try:
    import fire
except ImportError:
    fire = None


def _paths(bundle=None, out_dir=None):
    R = swr_io.get_data_root()
    gdir = os.path.join(swr_io.derivatives_dir(R), "group", "swr")
    bundle = bundle or os.path.join(gdir, "bundle")
    out_dir = out_dir or os.path.join(gdir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    return gdir, bundle, out_dir


def _load(bundle):
    need = ["pairs.csv", "channel_qc.csv", "ripples.csv"]
    missing = [f for f in need if not os.path.isfile(os.path.join(bundle, f))]
    if missing:
        raise FileNotFoundError(
            f"{bundle} is missing {missing} -- point --bundle at a downloaded "
            "bundle directory, or run `swr_export.py bundle` first.")
    return {f.split(".")[0]: pd.read_csv(os.path.join(bundle, f)) for f in need}


def attributes(bundle=None, out_dir=None):
    """Rate, duration, peak frequency and amplitude across the whole study."""
    import matplotlib
    matplotlib.use("Agg")
    import mc.plotting.ripple_figures as rfig

    _, bundle, out_dir = _paths(bundle, out_dir)
    t = _load(bundle)
    qc = t["channel_qc"]
    inc = qc[~qc["excluded"].fillna(False).astype(bool)]
    stem = os.path.join(out_dir, "ripple_attributes")
    rfig.ripple_attributes_figure(
        t["ripples"], qc, out_stem=stem,
        title=f"{len(t['ripples']):,} ripples, {len(inc)} derivations, "
              f"{inc.session.nunique()} sessions")
    print(f"  {len(t['ripples']):,} events, {len(inc)} derivations")
    print(f"-> {stem}.pdf / .png")
    return None


def chen(bundle=None, out_dir=None, show_contacts=False):
    """Chen Fig 2a-b pooled across every session in the bundle.

    Averaged with ONE WEIGHT PER DERIVATION, not per event: a single contact
    with a high ripple rate would otherwise dominate the grand average, and the
    claim is about hippocampal contacts in general.

    `show_contacts` defaults to False: the coverage panel is drawn better, on a
    surface rather than a projection, by `contacts` below, and a manuscript
    figure should not carry it twice.
    """
    import matplotlib
    matplotlib.use("Agg")
    import mc.plotting.ripple_figures as rfig

    _, bundle, out_dir = _paths(bundle, out_dir)
    t = _load(bundle)
    npz = os.path.join(bundle, "swr_bundle_figures.npz")
    if not os.path.isfile(npz):
        raise FileNotFoundError(
            f"{npz} not found -- this figure needs the waveform arrays, which "
            "`swr_export.py bundle` writes alongside the tables.")
    z = np.load(npz, allow_pickle=True)
    sessions = sorted({int(m.group(1)) for k in z.files
                       if (m := re.match(r"^s(\d+)_", k))})

    means, tfrs, ns = [], [], []
    t_ms = None
    for s in sessions:
        tag = f"s{s:02d}"
        if f"{tag}_mean" not in z.files:
            continue
        means.append(np.asarray(z[f"{tag}_mean"], float))
        if f"{tag}_tfr_mean" in z.files:
            tfrs.append(np.asarray(z[f"{tag}_tfr_mean"], float))
        if f"{tag}_n_events" in z.files:
            ns.append(np.asarray(z[f"{tag}_n_events"], float))
        if t_ms is None and f"{tag}_t_ms" in z.files:
            t_ms = np.asarray(z[f"{tag}_t_ms"], float)
    if not means:
        print("nothing to pool"); return None
    means = np.concatenate(means)
    tfr = np.nanmean(np.stack(tfrs), axis=0) if tfrs else None

    # Which session's example to feature. The bundle keeps one per session, so
    # the choice here is between sessions: take the most FOCAL, i.e. the event
    # whose ripple-band envelope stands highest above its own surroundings.
    # Peak amplitude would just pick the noisiest contact.
    best, best_score = None, -np.inf
    for s in sessions:
        tag = f"s{s:02d}"
        if f"{tag}_ex_bp" not in z.files:
            continue
        bp = np.asarray(z[f"{tag}_ex_bp"], float)
        if bp.size < 10:
            continue
        env = np.abs(bp)
        c = len(env) // 2
        w = max(1, len(env) // 20)
        near = env[c - w:c + w].max()
        far = np.median(np.r_[env[:c - 4 * w], env[c + 4 * w:]]) + 1e-9
        score = near / far
        if score > best_score:
            best_score, best = score, tag
    ex_raw = np.asarray(z[f"{best}_ex_raw"], float) if best else None
    ex_bp = np.asarray(z[f"{best}_ex_bp"], float) if best else None
    ex_tfr = (np.asarray(z[f"{best}_ex_tfr"], float)
              if best and f"{best}_ex_tfr" in z.files else None)

    # coordinates for panel (a), from the analysed derivations
    p = t["pairs"].merge(t["channel_qc"][["session", "pair_id", "excluded"]],
                         on=["session", "pair_id"], how="left")
    p["excluded"] = p["excluded"].fillna(True).astype(bool)
    inc = p[~p.excluded].dropna(subset=["mni_x", "mni_y", "mni_z"])
    coords = inc[["mni_x", "mni_y", "mni_z"]].to_numpy(float)
    rois = list(inc.get("pair_roi_atlas", pd.Series(["HC_mid"] * len(inc))))

    stem = os.path.join(out_dir, "chen_fig2_group")
    rfig.chen_panels(
        means, tfr, t_ms, ex_raw, ex_tfr, coords=coords, rois=rois,
        out_stem=stem, n_contacts=len(means), ex_bp=ex_bp,
        show_contacts=show_contacts)
    # No suptitle: at 2.5 cm a panel it collides with the panel titles, and the
    # counts belong in the caption.
    print(f"  pooled {len(means)} derivations from {len(sessions)} sessions; "
          f"example from {best} (focality {best_score:.1f})")
    print(f"-> {stem}.pdf / .png")
    return None


def artifact(bundle=None, out_dir=None):
    """Contamination of every derivation in the study, against the 2/3 rule."""
    import matplotlib
    matplotlib.use("Agg")
    import mc.plotting.ripple_figures as rfig

    _, bundle, out_dir = _paths(bundle, out_dir)
    qc = _load(bundle)["channel_qc"]
    stem = os.path.join(out_dir, "contamination_group")
    # No long title: at 4 cm it overruns the axes. It belongs in the caption.
    rfig.contamination_group_figure(qc, out_stem=stem)
    print(f"-> {stem}.pdf / .png")
    return None


def contacts(bundle=None, out_dir=None, contact_scale=0.40):
    """Hippocampal coverage. Delegates, so there is only one implementation.

    `contact_scale` 0.40 (was 0.20) so the markers survive the panel being
    printed at ~2 cm a brain. Checked against merging: at 0.40 the densest
    cluster still resolves as separate spheres.
    """
    _, bundle, out_dir = _paths(bundle, out_dir)
    import swr_contact_figure as scf
    return scf.make_figure(group_dir=bundle, out_dir=out_dir, verbose=False,
                           contact_scale=contact_scale)


def all(bundle=None, out_dir=None):
    """Every group figure, in one command."""
    for fn in (contacts, chen, attributes, artifact):
        print(f"\n--- {fn.__name__} ---")
        try:
            fn(bundle=bundle, out_dir=out_dir)
        except Exception as e:
            print(f"  [{fn.__name__} failed: {type(e).__name__}: {e}]")
    _, _, out_dir = _paths(bundle, out_dir)
    print(f"\nAll group figures -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"chen": chen, "attributes": attributes,
                   "contacts": contacts, "artifact": artifact, "all": all})
    else:
        all()
