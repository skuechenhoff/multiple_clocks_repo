#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three follow-ups on the ROI x condition table, all about HOW the magnitudes are
being compared.

1. PAIRED vs PSEUDOPOPULATION
   `swr_hfb_roi_conditions.py` compares two ROIs with a PAIRED t over the
   sessions that have both. lOFC and mPFC co-occur in only 15 of 41 sessions
   (10 of 29 subjects), so that test throws away 6 lOFC and 20 mPFC sessions.
   Three estimators are reported side by side here:

     paired      the sessions with both -- controls for session and subject,
                 costs most of the data
     unpaired    every session with either -- Welch t, uses all of it, but a
                 patient with lOFC and a patient with mPFC differ in more than
                 the electrode
     mixed       every DERIVATION, with subject as a random intercept -- the
                 pseudopopulation version, and the one that uses the data at
                 the grain it was measured at without pretending derivations
                 within a patient are independent

   They answer slightly different questions and are reported together on
   purpose. Where they agree the conclusion does not rest on the choice.

2. IS lOFC's LARGER EFFECT JUST NOISE?
   HFB is z-scored per sub-band per channel on artifact-free samples before
   anything else (`swr_extract_hfb`), so a region with denser or larger raw
   high-frequency activity does NOT get a bigger number for that reason -- the
   scale is removed per derivation. What can still differ is the VARIANCE of
   the estimate, so this reports, per ROI: the spread across sessions, the
   residual noise in the non-peri band, and the ripples per derivation. That
   separates "bigger effect, fewer sessions" from "bigger effect, noisier".

3. IS MEDIAL FRONTAL COUPLING DRIVEN BY THE FIRST TRAVERSAL?
   All pairwise phase contrasts within ROI x condition, not just
   encoding-vs-after. A claim that an effect is carried by one phase needs that
   phase to beat the others -- numerically at the very least.

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_hfb_roi_compare.py run \\
        --cond_dir=<ripple_hfb_conditions_exclusive_...>

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import datetime
import itertools
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))   # repo root
import mc.analyse.swr_io as swr_io
import mc.plotting.ripple_figures as rfig

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))    # scripts/
# archived scripts import BOTH maintained siblings (in scripts/) and each
# other (here), so both directories have to be importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # exploratory/
from swr_hfb_locking_contrasts import pooled_cond_curves, NONPERI_S, COL, LABEL
from swr_hfb_roi_partition import _bh
from swr_hfb_roi_conditions import PHASES, COND_KEYS, PERI_MS, MEDIAL_GROUP

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "hfb_roi_compare"
MIN_N = 5


def _prep(cond_dir, partition_dir):
    """Index with a partition ROI and a unique derivation id on every row."""
    A = pd.read_csv(os.path.join(partition_dir, "derivation_assignment.csv"))
    part = A.set_index(["session", "cx_pair"])["roi_part"]
    z = np.load(os.path.join(cond_dir, "timecourses.npz"))
    ix = pd.read_csv(os.path.join(cond_dir, "timecourse_index.csv"))
    ix["is_real"] = ix.is_real.astype(bool)
    if "trace_row" not in ix.columns:
        ix["trace_row"] = np.arange(len(ix))
    fam = ix[ix.roi.isin(("mPFC", "mOFC"))].copy()
    fam["roi"] = pd.MultiIndex.from_arrays(
        [fam.session, fam.cx_pair]).map(part)
    mfr = ix[ix.roi == MEDIAL_GROUP].copy()
    ix = pd.concat([fam, mfr], ignore_index=True).dropna(subset=["roi"])
    # A pair label recurs across sessions of one patient, so the derivation id
    # has to carry the session too.
    ix["_der"] = ix.session.astype(str) + "|" + ix.cx_pair.astype(str)
    counts = pd.read_csv(os.path.join(cond_dir, "ripple_counts.csv"))
    return z["t_ms"], z["traces"], ix, counts


from swr_hfb_locking_contrasts import pooled_curves_by as _pooled_shared


def _pooled(t_ms, tr, sub, counts, conds, unit_col):
    """Kept as a thin wrapper; the implementation moved to
    `swr_hfb_locking_contrasts.pooled_curves_by` when this script was archived,
    so the maintained pipeline does not depend on an exploratory file.

    The imported version does `int(unit)` to look the ripple weight up by
    session, so it cannot group by a derivation id. Everything else here is the
    same arithmetic: count-weighted mean of the real traces minus the same of
    the null traces, then flank-normalised. The session for the weight is taken
    from the group, which is unambiguous because a derivation belongs to
    exactly one session.
    """
    return _pooled_shared(t_ms, tr, sub, counts, conds, unit_col)


def _values(t_ms, tr, ix, counts, roi, conds, unit):
    cur = _pooled(t_ms, tr, ix[ix.roi == roi], counts, conds, unit)
    peri = np.abs(t_ms) <= PERI_MS
    return {u: float(c[peri].mean()) for u, c in cur.items()}


def _flank_noise(t_ms, tr, ix, counts, roi, conds, unit):
    """SD of the per-unit curve in the non-peri band: the estimator's noise."""
    cur = _pooled(t_ms, tr, ix[ix.roi == roi], counts, conds, unit)
    fl = (np.abs(t_ms) >= NONPERI_S[0] * 1000) & \
         (np.abs(t_ms) < NONPERI_S[1] * 1000)
    if not cur:
        return np.nan
    return float(np.mean([np.std(c[fl], ddof=1) for c in cur.values()]))


def run(cond_dir=None, partition_dir=None, out_dir=None, save=True):
    from scipy import stats as st

    root = swr_io.derivatives_dir(swr_io.get_data_root())
    G = os.path.join(root, "group", "swr")
    if cond_dir is None:
        raise SystemExit("--cond_dir is required (the exclusive-boundary run)")
    partition_dir = partition_dir or sorted(
        os.path.join(G, x) for x in os.listdir(G)
        if x.startswith("hfb_roi_partition_"))[-1]

    t_ms, tr, ix, counts = _prep(cond_dir, partition_dir)
    der_sub = (ix[["_der", "subject", "session"]].drop_duplicates("_der")
                 .set_index("_der"))
    rois = [r for r in ix.roi.unique()
            if len(ix[ix.roi == r]._der.unique()) >= 10]
    rois = sorted(rois, key=lambda r: (r != MEDIAL_GROUP,
                                       -len(ix[ix.roi == r]._der.unique())))

    # ---- 2. what each ROI's estimate is made of -------------------------
    diag = []
    for roi in rois:
        g = ix[ix.roi == roi]
        sess = _values(t_ms, tr, ix, counts, roi,
                       PHASES["all"]["reward"], "session")
        der = _values(t_ms, tr, ix, counts, roi,
                      PHASES["all"]["reward"], "_der")
        a = np.array(list(sess.values()))
        cnt = counts[counts.session.isin(g.session.unique())]
        diag.append({
            "roi": roi,
            "n_sessions": int(g.session.nunique()),
            "n_subjects": int(g.subject.nunique()),
            "n_derivations": int(g._der.nunique()),
            "deriv_per_session": round(g._der.nunique() / g.session.nunique(), 2),
            "mean_correct": float(a.mean()),
            "sd_across_sessions": float(a.std(ddof=1)),
            "sem_across_sessions": float(a.std(ddof=1) / np.sqrt(len(a))),
            "flank_noise_session": _flank_noise(t_ms, tr, ix, counts, roi,
                                                PHASES["all"]["reward"],
                                                "session"),
            "flank_noise_derivation": _flank_noise(t_ms, tr, ix, counts, roi,
                                                   PHASES["all"]["reward"],
                                                   "_der"),
            "n_derivation_values": len(der),
        })
    D = pd.DataFrame(diag)

    # ---- 1. three estimators for every ROI pair x condition x phase -----
    rows = []
    pairs = [(a, b) for a, b in itertools.combinations(
        [r for r in rois if r != MEDIAL_GROUP], 2)]
    for (a, b), ph, k in itertools.product(pairs, PHASES, COND_KEYS):
        conds = PHASES[ph][k]
        sa = _values(t_ms, tr, ix, counts, a, conds, "session")
        sb = _values(t_ms, tr, ix, counts, b, conds, "session")
        da = _values(t_ms, tr, ix, counts, a, conds, "_der")
        db = _values(t_ms, tr, ix, counts, b, conds, "_der")
        if min(len(sa), len(sb)) < MIN_N:
            continue
        r = {"roi_a": a, "roi_b": b, "phase": ph, "condition": k,
             "n_sess_a": len(sa), "n_sess_b": len(sb),
             "n_der_a": len(da), "n_der_b": len(db),
             "mean_a": float(np.mean(list(sa.values()))),
             "mean_b": float(np.mean(list(sb.values())))}
        sh = sorted(set(sa) & set(sb))
        if len(sh) >= MIN_N:
            d = np.array([sa[u] - sb[u] for u in sh])
            t, p = st.ttest_1samp(d, 0.0)
            r.update({"n_paired": len(sh), "paired_diff": float(d.mean()),
                      "paired_p": float(p)})
        t, p = st.ttest_ind(list(sa.values()), list(sb.values()),
                            equal_var=False)
        r.update({"unpaired_diff": r["mean_a"] - r["mean_b"],
                  "unpaired_p": float(p)})
        r.update(_mixed(da, db, der_sub, a, b))
        rows.append(r)
    C = pd.DataFrame(rows)
    for col in ("paired_p", "unpaired_p", "mixed_p"):
        if col in C:
            C[col.replace("_p", "_q")] = _bh(C[col])

    # ---- 3. every pairwise phase contrast, within ROI x condition -------
    prows = []
    for roi, k in itertools.product(rois, COND_KEYS):
        for p1, p2 in itertools.combinations(
                [x for x in PHASES if x != "all"], 2):
            v1 = _values(t_ms, tr, ix, counts, roi, PHASES[p1][k], "session")
            v2 = _values(t_ms, tr, ix, counts, roi, PHASES[p2][k], "session")
            sh = sorted(set(v1) & set(v2))
            if len(sh) < MIN_N:
                continue
            d = np.array([v1[u] - v2[u] for u in sh])
            t, p = st.ttest_1samp(d, 0.0)
            prows.append({"roi": roi, "condition": k,
                          "phase_a": p1, "phase_b": p2, "n": len(sh),
                          "mean_a": float(np.mean([v1[u] for u in sh])),
                          "mean_b": float(np.mean([v2[u] for u in sh])),
                          "diff": float(d.mean()),
                          "sem": float(d.std(ddof=1) / np.sqrt(len(d))),
                          "p": float(p)})
    P = pd.DataFrame(prows)
    P["q"] = _bh(P["p"]) if len(P) else np.nan

    _report(D, C, P, rois)

    if not save:
        return None
    stamp = datetime.date.today().isoformat()
    out_dir = out_dir or os.path.join(G, f"{ANALYSIS_NAME}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    D.to_csv(os.path.join(out_dir, "roi_diagnostics.csv"), index=False)
    C.to_csv(os.path.join(out_dir, "roi_pair_three_estimators.csv"),
             index=False)
    P.to_csv(os.path.join(out_dir, "phase_contrasts_all_pairs.csv"),
             index=False)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"analysis": ANALYSIS_NAME, "created": stamp,
                   "cond_dir": cond_dir, "partition_dir": partition_dir,
                   "peri_ms": PERI_MS,
                   "hfb_normalisation": ("per sub-band Hilbert amplitude, log, "
                                         "z-scored per channel on "
                                         "artifact-free samples"),
                   "estimators": {
                       "paired": "sessions with both ROIs, one-sample t on diff",
                       "unpaired": "all sessions per ROI, Welch t",
                       "mixed": "all derivations, subject random intercept"},
                   "diagnostics": D.to_dict("records"),
                   "roi_pairs": C.to_dict("records"),
                   "phase_contrasts": P.to_dict("records")}, f, indent=2,
                  default=str)
    _figure(C, D, out_dir)
    print(f"\nsaved -> {out_dir}")
    return None


def _mixed(da, db, der_sub, a, b):
    """Derivation-level ROI difference with subject as a random intercept."""
    import statsmodels.formula.api as smf
    rows = ([{"y": v, "roi": 0, "subj": der_sub.subject.get(k)}
             for k, v in da.items()]
            + [{"y": v, "roi": 1, "subj": der_sub.subject.get(k)}
               for k, v in db.items()])
    d = pd.DataFrame(rows).dropna()
    if d.subj.nunique() < 3 or len(d) < 10:
        return {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm("y ~ roi", d, groups=d["subj"]).fit(reml=True)
        return {"mixed_diff": float(-m.params["roi"]),   # a minus b
                "mixed_p": float(m.pvalues["roi"]),
                "n_der_total": int(len(d)),
                "n_subj_total": int(d.subj.nunique())}
    except Exception as e:
        return {"mixed_note": f"{type(e).__name__}"}


def _report(D, C, P, rois):
    f4 = lambda v: "" if pd.isna(v) else f"{v:+.4f}"
    f3 = lambda v: "" if pd.isna(v) else f"{v:.3f}"
    print(f"\n{'='*104}\n2. WHAT EACH ROI's ESTIMATE IS MADE OF "
          f"(correct uncovers, all phases)\n{'='*104}")
    print(f"  {'ROI':<14s}{'sess':>5s}{'subj':>5s}{'der':>5s}{'der/sess':>9s}"
          f"{'mean':>10s}{'SD':>9s}{'SEM':>9s}{'flank noise':>13s}")
    for _, r in D.iterrows():
        print(f"  {r.roi:<14s}{r.n_sessions:>5d}{r.n_subjects:>5d}"
              f"{r.n_derivations:>5d}{r.deriv_per_session:>9.2f}"
              f"{f4(r.mean_correct):>10s}{r.sd_across_sessions:>9.4f}"
              f"{r['sem_across_sessions']:>9.4f}"
              f"{r.flank_noise_session:>13.4f}")
    print("  HFB is z-scored per channel before anything else, so a denser or "
          "larger\n  raw signal cannot inflate these numbers; only the SPREAD "
          "can differ.")

    print(f"\n{'='*104}\n1. THREE ESTIMATORS, lOFC vs mPFC, correct uncovers"
          f"\n{'='*104}")
    sel = C[(C.roi_a.isin(["lOFC", "mPFC"])) & (C.roi_b.isin(["lOFC", "mPFC"]))
            & (C.condition == "reward")]
    print(f"  {'phase':<17s}{'A - B':>10s}{'paired':>22s}{'unpaired':>22s}"
          f"{'mixed (per-deriv)':>24s}")
    for _, r in sel.iterrows():
        print(f"  {r.phase:<17s}{r.roi_a + '-' + r.roi_b:>10s}"
              f"{f4(r.get('paired_diff', np.nan))} p={f3(r.get('paired_p', np.nan))}"
              f" n={int(r.get('n_paired', 0)):<3d}"
              f"{f4(r.get('unpaired_diff', np.nan))} p={f3(r.get('unpaired_p', np.nan))}"
              f" n={r.n_sess_a}/{r.n_sess_b}"
              f"{f4(r.get('mixed_diff', np.nan))} p={f3(r.get('mixed_p', np.nan))}"
              f" n={int(r.get('n_der_total', 0))}d")
    print("  BH q over all ROI pairs x phases x conditions is in the csv; the "
          "smallest\n  q by estimator: "
          + ", ".join(f"{e} {C[e + '_q'].min():.3f}"
                      for e in ("paired", "unpaired", "mixed")
                      if e + "_q" in C and C[e + "_q"].notna().any()))

    print(f"\n{'='*104}\n3. IS THE EFFECT CARRIED BY ONE PHASE? all pairwise "
          f"phase contrasts\n{'='*104}")
    for roi in [MEDIAL_GROUP, "lOFC"]:
        g = P[(P.roi == roi) & (P.condition == "reward")]
        if not len(g):
            continue
        print(f"  --- {roi}, correct uncovers ---")
        print(f"  {'phase A':<17s}{'phase B':<17s}{'n':>4s}{'A':>10s}"
              f"{'B':>10s}{'A - B':>11s}{'p':>8s}{'q':>8s}")
        for _, r in g.sort_values("p").iterrows():
            print(f"  {r.phase_a:<17s}{r.phase_b:<17s}{r.n:>4d}"
                  f"{f4(r.mean_a):>10s}{f4(r.mean_b):>10s}"
                  f"{f4(r['diff']):>11s}{f3(r.p):>8s}{f3(r.q):>8s}")


def _figure(C, D, out_dir):
    """The three estimators side by side, for every ROI pair on reward."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = C[C.condition == "reward"].copy()
    if not len(g):
        return
    g["label"] = g.roi_a + " − " + g.roi_b + "\n" + g.phase.str.replace("_", " ")
    g = g.sort_values(["roi_a", "roi_b", "phase"])
    est = [("paired_diff", "paired_p", "paired (shared sessions)", "#448363"),
           ("unpaired_diff", "unpaired_p", "unpaired (all sessions)", "#DC673E"),
           ("mixed_diff", "mixed_p", "mixed (all derivations)", "#23677E")]
    with plt.rc_context(rfig._rc()):
        h = max(6.0, 0.30 * len(g) + 3.0)
        fig, ax = plt.subplots(figsize=(16 * rfig.CM, h * rfig.CM))
        y = np.arange(len(g))
        for i, (dcol, pcol, lab, col) in enumerate(est):
            if dcol not in g:
                continue
            off = (i - 1) * 0.26
            x = g[dcol].to_numpy(float)
            ax.scatter(x, y + off, s=14, c=col, label=lab, zorder=3,
                       linewidths=0)
            sig = g[pcol].to_numpy(float) < 0.05
            ax.scatter(x[sig], (y + off)[sig], s=46, facecolors="none",
                       edgecolors=col, linewidths=0.9, zorder=4)
        ax.axvline(0, color="#3d3d3d", lw=0.8, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(g.label, fontsize=rfig.FS_TICK - 2)
        ax.set_xlabel("difference in ripple-HFB coupling for CORRECT uncovers\n"
                      "(ringed = uncorrected p < 0.05; none survives BH)")
        ax.set_title("Same comparison, three estimators\n"
                     "they disagree where the ROIs rarely co-occur",
                     fontsize=rfig.FS_TITLE)
        ax.legend(frameon=False, fontsize=rfig.FS_TICK - 1, loc="best")
        ax.margins(y=0.01)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "three_estimators.pdf"))
        fig.savefig(os.path.join(out_dir, "three_estimators.png"), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
