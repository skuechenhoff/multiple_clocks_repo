#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is the first-D ripple increase (F5) stronger on CA-weighted than DG-weighted
hippocampal derivations?

Sakon et al. report a stronger ripple effect in CA1 than in dentate gyrus. F5 in
this project -- ripple rate rises 0.25-0.55 s after the FIRST uncovering of D --
is the closest analogue here, so the question is whether it carries the same
subfield gradient.

**Runs on the EXISTING bundle.** Subfield is a pure lookup on the derivation's
MNI coordinate, and `pairs` in the bundle already carries `mni_x/y/z`. Nothing
has to be recomputed on the cluster and no LFP is touched.

⚠ THREE LIMITS, all load-bearing. Read them before reading the result.

1. **This is CA vs DG, not CA1 vs CA3.** The Juelich "cornu ammonis" volume
   covers every CA field. Sakon's contrast is CA1 specifically; this can only
   approximate it. Do not write CA1 for this.
2. **A hard label would be a volume artefact.** Cornu ammonis dominates the
   hippocampus by size, so max-prob calls ~85% of derivations "CA". The
   analysis is therefore a CONTINUOUS moderator on
   `ca_dg_index = (P(CA) - P(DG)) / (P(CA) + P(DG))`. The median split is
   printed for interpretability only, and is not the test.
3. **The effect being moderated is itself exploratory.** F5 is one of twelve
   sliding-window cells, of which ~1.2 are expected significant by chance
   (SWR_SUMMARY F5 caveats). A moderator on a fragile effect is more fragile
   still. Treat a positive result as a reason to look, not as a finding.

Usage:
    conda activate env_multiple_clocks
    python scripts/swr_subfield_moderator.py run --bundle=<bundle dir>
    python scripts/swr_subfield_moderator.py run --bundle=<dir> --post=0.5

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.anatomy_atlas as anat_atlas
import mc.analyse.swr_sakon as sk

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "swr_subfield_moderator"
BASE_WIN = sk.BASE_WIN          # (-1.6, -1.1), Sakon's own-trial baseline
DEFAULT_POST = 0.5              # F5's surviving cluster is +0.25 to +0.55 s


def add_subfield(pairs):
    """`p_CA` / `p_DG` / `p_SUB` / `ca_dg_index` from the MNI coordinate."""
    pr = pairs.copy()
    xyz = pr[["mni_x", "mni_y", "mni_z"]].to_numpy(float)
    sub = anat_atlas.hippocampal_subfield_probability(xyz)
    for k, v in sub.items():
        pr[f"p_{k}"] = v
    tot = pr["p_CA"] + pr["p_DG"]
    pr["ca_dg_index"] = np.where(tot > 0, (pr["p_CA"] - pr["p_DG"]) / np.maximum(tot, 1e-9),
                                 np.nan)
    return pr


def _first_d_times(beh):
    """(first_D_times, later_D_times) per session, from the bundle's behaviour."""
    out = {}
    for sess, b in beh.groupby("session"):
        first, later = [], []
        for _, g in b.groupby("grid_no"):
            g = g.sort_values("rep_overall")
            t = g["t_D"].to_numpy(float)
            t = t[np.isfinite(t)]
            if not len(t):
                continue
            first.append(t[0])
            later.extend(t[1:])
        out[int(sess)] = (np.asarray(first), np.asarray(later))
    return out


def run(bundle=None, post=DEFAULT_POST, n_perm=10000, save=True, out_dir=None):
    b_dir = bundle or os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                                   "group", "swr", "bundle")
    with open(os.path.join(b_dir, "swr_bundle.pkl"), "rb") as f:
        B = pickle.load(f)
    rip, iv, pairs, beh = B["ripples"], B["intervals"], B["pairs"], B["behaviour"]
    if "session" not in pairs.columns:
        raise RuntimeError("bundle `pairs` has no session column")

    pairs = add_subfield(pairs)
    print(f"\n{len(pairs)} derivations, subfield from MNI coordinate")
    print(pairs[["p_CA", "p_DG", "p_SUB", "ca_dg_index"]].describe().round(2)
          .loc[["mean", "std", "min", "50%", "max"]].to_string())

    # PRECONDITION -- and it cuts both ways, so read both halves.
    #
    # The montage is almost entirely CA-weighted: 2 of 199 derivations have
    # P(DG) > P(CA). That is the contact-selection rule working AS INTENDED,
    # not a defect. `select_hpc_contacts` takes the single highest
    # P(hippocampus) contact per probe, i.e. the one deepest in the structure,
    # and cornu ammonis is what a deep hippocampal contact mostly sees. If
    # Sakon are right that the effect lives in CA1, this sample is already
    # concentrated where the effect should be strongest -- which is good for
    # detecting F5 at all.
    #
    # The cost is specific and limited: the CA-vs-DG CONTRAST has no DG arm.
    # This moderator therefore tests "more CA" against "slightly less CA", so a
    # null on it is uninformative about the Sakon gradient. It says nothing
    # about F5 itself, which is not a subfield contrast.
    n_dg = int((pairs.ca_dg_index < 0).sum())
    n_tot = int(pairs.ca_dg_index.notna().sum())
    print(f"\n  PRECONDITION: derivations where P(DG) > P(CA): {n_dg}/{n_tot}")
    if n_dg < 0.1 * max(n_tot, 1):
        print("  NOTE the montage is almost entirely CA-weighted. That is the "
              "selection rule\n       working as intended -- the deepest "
              "hippocampal contact per probe mostly sees\n       cornu ammonis, "
              "so the sample is already concentrated where Sakon put the\n"
              "       effect. The limitation is narrow: the CA-vs-DG contrast "
              "has no DG arm, so\n       a null on THIS moderator is "
              "uninformative about that gradient. It does not\n       bear on "
              "F5 itself.")
    res_pre = {"n_DG_dominant": n_dg, "n_derivations": n_tot,
               "moderator_can_test_CA_vs_DG": bool(n_dg >= 0.1 * max(n_tot, 1)),
               "interpretation": ("sample is CA-weighted by construction "
                                  "(deepest contact per probe); good for "
                                  "detecting a CA1 effect, but leaves the "
                                  "CA-vs-DG contrast without a DG arm")}

    dtimes = _first_d_times(beh)
    ev_win = (0.0, float(post))

    rows = []
    for _, p in pairs.iterrows():
        sess, pid = int(p.session), str(p.pair_id)
        if sess not in dtimes:
            continue
        t_r = rip.loc[rip.pair_id == pid, "t_peak_s"].to_numpy(float)
        ivs = iv.loc[iv.pair_id == pid, ["start_s", "stop_s"]].to_numpy(float)
        if not len(t_r) or not len(ivs):
            continue
        for cond, tt in (("first_D", dtimes[sess][0]), ("later_D", dtimes[sess][1])):
            tt = sk.dedup_events(tt)
            if not len(tt):
                continue
            r_ev, _, _ = sk.window_rate(tt, t_r, ivs, ev_win)
            r_bs, _, _ = sk.window_rate(tt, t_r, ivs, BASE_WIN)
            ok = np.isfinite(r_ev) & np.isfinite(r_bs)
            if ok.sum() < 3:
                continue
            rows.append({
                "session": sess, "pair_id": pid,
                "subject_key": p.get("subject_label", p.get("subject_key", sess)),
                "condition": cond, "n_events": int(ok.sum()),
                "rate_event": float(np.nanmean(r_ev[ok])),
                "rate_base": float(np.nanmean(r_bs[ok])),
                "delta": float(np.nanmean(r_ev[ok] - r_bs[ok])),
                "ca_dg_index": float(p.ca_dg_index),
                "p_CA": float(p.p_CA), "p_DG": float(p.p_DG),
                "pair_roi": p.get("pair_roi_atlas"),
            })
    d = pd.DataFrame(rows).dropna(subset=["ca_dg_index"])
    if not len(d):
        raise RuntimeError("no derivation yielded a usable window")

    res = {"analysis": ANALYSIS_NAME, "bundle": b_dir,
           "event_window_s": list(ev_win), "baseline_window_s": list(BASE_WIN),
           "created": datetime.now().isoformat(timespec="seconds"),
           "moderator": "ca_dg_index = (P(CA)-P(DG))/(P(CA)+P(DG)), Juelich prob-2mm",
           "caveat": ("CA covers all cornu ammonis fields -- this is NOT CA1 vs "
                      "CA3. F5 is itself exploratory; see the module docstring."),
           "precondition": res_pre}

    import statsmodels.formula.api as smf
    from scipy import stats as st
    for cond in ("first_D", "later_D"):
        g = d[d.condition == cond].copy()
        print(f"\n{'='*66}\n {cond}:  {len(g)} derivations, "
              f"{g.session.nunique()} sessions, {g.subject_key.nunique()} subjects")
        print(f"{'='*66}")
        eq2 = sk.fit_eq2(g, subject_col="subject_key", session_col="session",
                         n_perm_sign=n_perm)
        if "error" not in eq2:
            # NOT the canonical F5 test. fit_eq2 expects trial-level rows; it is
            # being given one row per derivation, so subjects with few
            # derivations fall below its 10-row floor and are dropped (13 of 42
            # subjects survive). Read it as a sanity check that the effect is
            # present at all, not as a reproduction of F5. The MODERATOR below
            # uses every derivation and is the actual test.
            print(f"  Eq.2 effect (event vs own baseline, DERIVATION-level -- "
                  f"see comment, not the canonical F5): "
                  f"mean t = {eq2['mean_t']:+.3f}, "
                  f"t({eq2['df']}) = {eq2['t']:+.3f}, "
                  f"p = {eq2['p']:.4f}, p_perm = {eq2['p_perm']:.4f}  "
                  f"[n = {eq2['n_subjects']} subjects]")
            res[f"{cond}_eq2"] = {k: v for k, v in eq2.items()
                                  if k not in ("per_subject", "perm")}
        # THE moderator test: does the per-derivation effect scale with CA/DG?
        # Subject as a random effect, because several derivations share one.
        try:
            m = smf.mixedlm("delta ~ ca_dg_index", g,
                            groups=g["subject_key"].astype(str)).fit(reml=True,
                                                                    method="nm")
            beta = float(m.params.get("ca_dg_index", np.nan))
            pval = float(m.pvalues.get("ca_dg_index", np.nan))
            print(f"  moderator  delta ~ ca_dg_index + (1|subject): "
                  f"beta = {beta:+.4f}, p = {pval:.4f}")
            res[f"{cond}_moderator"] = {"beta": beta, "p": pval,
                                        "n_derivations": int(len(g))}
        except Exception as e:
            print(f"  moderator model failed: {type(e).__name__}: {e}")
        # Median split, for reading only -- NOT the test.
        med = g.ca_dg_index.median()
        hi, lo = g[g.ca_dg_index >= med], g[g.ca_dg_index < med]
        tt = st.ttest_ind(hi.delta.dropna(), lo.delta.dropna(), equal_var=False)
        print(f"  [descriptive only] median split at {med:+.3f}: "
              f"CA-weighted {hi.delta.mean():+.4f} Hz (n={len(hi)}) vs "
              f"DG-weighted {lo.delta.mean():+.4f} Hz (n={len(lo)}), "
              f"Welch p = {tt.pvalue:.3f}")
        res[f"{cond}_median_split"] = {
            "threshold": float(med), "mean_delta_CA_weighted": float(hi.delta.mean()),
            "mean_delta_DG_weighted": float(lo.delta.mean()),
            "n_CA": int(len(hi)), "n_DG": int(len(lo)),
            "welch_p": float(tt.pvalue),
            "note": "descriptive only; the moderator model is the test"}

    if save:
        out_dir = out_dir or os.path.join(
            swr_io.derivatives_dir(swr_io.get_data_root()), "group", "swr",
            f"subfield_moderator_{datetime.now():%Y-%m-%d}")
        os.makedirs(out_dir, exist_ok=True)
        d.to_csv(os.path.join(out_dir, "per_derivation.csv"), index=False)
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"\nsaved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
