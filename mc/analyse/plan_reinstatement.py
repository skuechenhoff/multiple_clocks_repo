#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Within-session plan reinstatement: the cell analogue of the fMRI loading result.

    python scripts/cells_plan_reinstatement.py

The fMRI result: during the instruction phase, mPFC takes on the action-plan
geometry exactly when the final reward ('see D') is shown. The sEEG task has no
instruction phase -- rewards are DISCOVERED -- and the shared configurations put
8 different locations at every state, so the cross-layout plan RDM is constant
and cannot be fitted. This analysis drops the cross-layout RDM entirely.

QUESTION. When a reward is discovered, does the population already look like
the EXECUTION pattern of THIS layout at THIS square, more than like the execution
pattern of OTHER layouts at the same square?

    dot(i, b) = x_i · T_b(L) / n_cells
    dot ~ same_layout + |offset| + |offset|^2 + offset   (event-demeaned)
    Δ = coefficient on same_layout

x_i      z-scored spike counts in [t+0.05, t+1.0] s after test uncover i at L
T_b(L)   mean z-vector over EXECUTE correct uncovers at L in block b, for every
         block except the test event's own in which L is rewarded
offset   b - test block; regressed out because similarity drifts with block
         distance and repeats of one layout are spaced further apart

Location and event type are identical in both terms, so only the layout -- and
with it the future -- differs. The dot product is used rather than a
correlation because template noise does not bias its mean.

PREDICTION. At the first discovery of A, B, C, D the number of KNOWN future
rewards is 0, 1, 2, 3. A plan predicts Δ_D > Δ_A; anything constant across
states (state coding, reward response, location role) cancels in D - A.

⚠ At D the next goal (A) is known, at A-C it is not, and execution pauses after a
correct uncover are short (median 0.28 s), so the window includes heading to the
next goal. The trend over A, B, C is free of this.
⚠ At D the known set IS the plan, so plan and memory are not separable here.

Pre-registered in CHANGELOG.md, 2026-10-01.

@author: Svenja Kuchenhoff
"""

import os
import json
import datetime

import numpy as np
import pandas as pd
from scipy import stats

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc

SEED = 42
N_FLIP = 10000
WINDOW = (0.05, 1.00)        # s after the uncover
ROI_SETS = {"mPFC": ["mPFC"], "HC_all": ["HC_anterior", "HC_mid"],
            "mOFC": ["mOFC"], "PCC": ["PCC"]}
MIN_CELLS = 2                # also the minimum number of LFP derivations
LFP_SIGNALS = ["hfb", "theta", "beta", "ripple"]
LFP_ROI_SETS = ["MedialFrontal", "HPC", "Visual"]   # Visual = confound detector
MEDIAL_MAX_ABS_X = 20.0      # mm, as MEDIAL_MAX_ABS_X in swr_ripple_hfb_conditions
MIN_PREMOVE_S = 0.15         # premove: shortest usable window after WINDOW[0]
RIPPLE_MASK_S = 0.20         # ripmask: samples within this of a ripple peak
MIN_KEPT_SAMPLES = 5         # ripmask: a window needs this many samples left
SHIFT_RANGE_S = (2.0, 5.0)   # ripshift: |offset| of the shifted ripple train
MIN_TEMPLATE_EVENTS = 3      # execute uncovers needed to form one template
STATES = ["A", "B", "C", "D"]


def uncover_table(steps_s, occ):
    """One row per correct uncover: time, location, state, block, layout,
    phase, first discovery within its block, and run of its layout (0 = first)."""
    st = steps_s.sort_values("t_s").reset_index(drop=True)
    ph, _, _, _ = swc.phase_labels(st, occ)
    phase = np.full(len(st), "", dtype=object)
    phase[occ.step_idx.to_numpy()] = ph
    # ⚠ An uncover row's `t_s` is when the subject ARRIVED on the square; the
    # press -- and the reward, `t_A`..`t_D` to within 10 ms -- is the NEXT row's
    # `t_s`, a median 0.25-0.57 s later depending on session.
    t_press = np.append(st.t_s.to_numpy(float)[1:], np.nan)
    t_move = np.append(st.t_s.to_numpy(float)[2:], [np.nan, np.nan])
    m = (st.correct_uncover.to_numpy(float) == 1) & (phase != "") \
        & np.isfinite(t_press)
    ev = pd.DataFrame({"t_s": t_press[m],
                       "t_move": t_move[m],      # first move after the reward
                       "loc": st["loc"].to_numpy(float)[m].astype(int),
                       "state": st.state.to_numpy(float)[m].astype(int) - 1,
                       "grid_num": st.grid_num.to_numpy(float)[m],
                       "grid_id": st.grid_id.to_numpy(float)[m],
                       "phase": phase[m]})
    ev["first_discovery"] = ~ev.duplicated(["grid_num", "state"]) \
        & (ev.phase == "encode")
    first_block = ev.groupby("grid_id").grid_num.transform(
        lambda g: g.rank(method="dense") - 1)
    ev["run"] = first_block.astype(int)
    return ev.reset_index(drop=True)


def block_dots(Z, ev, test_idx):
    """Similarity of each test event to every other block's execute template
    at the same square. One row per (test event, template block).

    Z: (n_cells, n_events) z-scored counts, columns aligned with `ev`.
    """
    ex = (ev.phase == "execute").to_numpy()
    loc, gid, gnum = (ev[c].to_numpy() for c in ("loc", "grid_id", "grid_num"))
    rows = []
    for i in test_idx:
        pool = ex & (loc == loc[i]) & (gnum != gnum[i])
        for b in np.unique(gnum[pool]):
            m = pool & (gnum == b)
            if m.sum() >= MIN_TEMPLATE_EVENTS:
                rows.append((i, Z[:, i] @ Z[:, m].mean(axis=1) / Z.shape[0],
                             gid[m][0] == gid[i], b - gnum[i]))
    return pd.DataFrame(rows, columns=["event", "dot", "same", "offset"])


def same_layout_beta(D):
    """Δ = coefficient on 'same layout', with block distance regressed out.

    dot ~ same + |offset| + |offset|^2 + offset, every column demeaned within
    test event (event fixed effect, absorbs each event's overall scale).
    ⚠ Without the distance terms the estimator is biased negative: similarity
    falls with block distance (mPFC: +0.066 at 1 block, -0.044 at >12) and
    repeats of a layout are spaced further apart (mean 10.2 vs 8.1 blocks). The
    pre-registered, unadjusted version gave a NEGATIVE positive control
    (mPFC t = -3.93).
    """
    D = D[D.groupby("event").same.transform("nunique") == 2]
    if D.event.nunique() < 3:
        return np.nan
    a = D.offset.abs().to_numpy(float)
    X = np.column_stack([D.same.to_numpy(float), a, a ** 2,
                         D.offset.to_numpy(float), D["dot"].to_numpy()])
    X = X - pd.DataFrame(X).groupby(D.event.to_numpy()).transform("mean").to_numpy()
    beta, *_ = np.linalg.lstsq(X[:, :4], X[:, 4], rcond=None)
    return float(beta[0])


def signflip_t(values, n_flip=N_FLIP, seed=SEED):
    """One-sample t against 0, and one-sided P (> 0) from sign flips.

    Row 0 of the flip matrix is the identity, so the empirical t is computed by
    the same line as every null t.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    F = np.vstack([np.ones(len(v)),
                   rng.choice([-1.0, 1.0], size=(n_flip, len(v)))])
    X = F * v
    T = X.mean(axis=1) / (X.std(axis=1, ddof=1) / np.sqrt(len(v)))
    return dict(n=int(len(v)), mean=float(v.mean()),
                sem=float(stats.sem(v)), t=float(T[0]),
                p_one_sided=float((T[1:] >= T[0]).mean()),
                p_two_sided=float((np.abs(T[1:]) >= abs(T[0])).mean()))


def event_windows(ev, variant):
    """(t0, t1) per event. `premove` ends the window at the first move after
    the reward; events with less than MIN_PREMOVE_S left get NaN."""
    t0 = ev.t_s.to_numpy() + WINDOW[0]
    t1 = ev.t_s.to_numpy() + WINDOW[1]
    if variant == "premove":
        t1 = np.minimum(t1, ev.t_move.to_numpy())
        t1 = np.where(t1 - t0 >= MIN_PREMOVE_S, t1, np.nan)
    return t0, t1


def ripple_mask(t_rip, n, fs):
    """Boolean over samples: True within RIPPLE_MASK_S of a ripple peak."""
    m = np.zeros(n + 1, int)
    a = np.clip(np.round((t_rip - RIPPLE_MASK_S) * fs).astype(int), 0, n)
    b = np.clip(np.round((t_rip + RIPPLE_MASK_S) * fs).astype(int), 0, n)
    np.add.at(m, a, 1)
    np.add.at(m, b, -1)
    return np.cumsum(m)[:n] > 0


def spike_features(s, t0, t1, spk, roi):
    """{roi: (n_cells, n_events)} z-scored spike RATES in [t0, t1]."""
    out = {}
    for roiname, members in ROI_SETS.items():
        cells = [c for c in roi[(roi.session == s) & roi.roi.isin(members)].cell
                 if c < len(spk[s]["spikes"])]
        if len(cells) < MIN_CELLS:
            continue
        Z, = swc.zscore_cells([swc.window_counts(spk[s], cells, t0, t1)
                               / (t1 - t0)])
        out[roiname] = np.nan_to_num(Z)
    return out


def lfp_contacts(bundle):
    """Usable derivations with their analysis ROI.

    MedialFrontal = mPFC + mOFC within MEDIAL_MAX_ABS_X of the midline, as in
    the HFB ripple-locking analysis: 40% of 'mOFC' contacts sit beyond
    |x| = 25 mm, i.e. in lateral OFC, and must not be pooled in.
    """
    hp = bundle["hfb_pairs"]
    hp = hp[~hp.excluded.fillna(False)].copy()
    medial = hp.roi_family.isin(["mPFC", "mOFC"]) \
        & (hp.mni_x.abs() <= MEDIAL_MAX_ABS_X)
    hp["roi"] = np.where(medial, "MedialFrontal", hp.roi_family)
    return hp[hp.roi.isin(LFP_ROI_SETS)]


def lfp_features(t0, t1, store, signal, contacts, clean_iv, exclude=None):
    """{roi: (n_derivations, n_events)} z-scored mean `signal` in [t0, t1].

    A window touching an artifact interval is NaN for that derivation and
    becomes 0 after z-scoring, exactly as a silent cell does. `exclude` (bool
    per sample) drops samples from the mean -- the ripple mask; a window with
    fewer than MIN_KEPT_SAMPLES left is treated like an artifact window.
    """
    fs = store["out_fs"]
    X = store[signal]
    row = {p: i for i, p in enumerate(store["pair_ids"])}
    keep = np.ones(X.shape[1]) if exclude is None else (~exclude).astype(float)
    ck = np.concatenate([[0], np.cumsum(keep)])
    i0 = np.round(t0 * fs).astype(int)
    i1 = np.round(t1 * fs).astype(int)
    inb = (i0 >= 0) & (i1 < X.shape[1])
    i0, i1 = np.where(inb, i0, 0), np.where(inb, i1, 0)
    out = {}
    for roiname in LFP_ROI_SETS:
        feats = []
        for p in contacts[contacts.roi == roiname].pair_id:
            if p not in row:
                continue
            clean = np.zeros(X.shape[1] + 1)
            for a, b in clean_iv.get(p, []):
                clean[max(0, int(a * fs)):min(X.shape[1], int(b * fs))] = 1
            cc = np.concatenate([[0], np.cumsum(clean[:-1])])
            cx = np.concatenate([[0], np.cumsum(X[row[p]] * keep,
                                                dtype=np.float64)])
            n_kept = ck[i1] - ck[i0]
            v = (cx[i1] - cx[i0]) / np.maximum(n_kept, 1)
            ok = inb & ((cc[i1] - cc[i0]) == (i1 - i0)) \
                & (n_kept >= MIN_KEPT_SAMPLES)
            feats.append(np.where(ok, v, np.nan))
        if len(feats) < MIN_CELLS:
            continue
        F = np.array(feats)
        mu = np.nanmean(F, axis=1, keepdims=True)
        sd = np.nanstd(F, axis=1, keepdims=True)
        out[roiname] = np.nan_to_num((F - mu) / np.where(sd > 0, sd, np.inf))
    return out


def run(out_dir, signal="spikes", variant="full"):
    """`variant`: full | premove | ripmask | ripshift<k> (k seeds the shift)."""
    import pickle
    import mc.analyse.swr_bundle as sb
    b_dir = os.path.join(rrsa._derivatives(), "group", "swr", "bundle_v2")
    rip = pd.read_csv(os.path.join(b_dir, "ripples.csv"),
                      usecols=["session", "subject_key", "t_peak_s"])
    subject = dict(zip(rip.session, rip.subject_key))
    rng = np.random.default_rng(SEED + int(variant[8:] or 0)
                                if variant.startswith("ripshift") else SEED)
    sessions = sorted(subject)
    steps = swl.load(sessions)
    if signal == "spikes":
        spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
        roi = rrsa.cell_roi_table(sessions=sessions)
    else:
        with open(os.path.join(b_dir, "swr_bundle.pkl"), "rb") as f:
            B = pickle.load(f)
        contacts = lfp_contacts(B)
        hiv = B["hfb_intervals"]
        contacts.to_csv(os.path.join(out_dir, "contacts.csv"), index=False)

    rows = []
    for s in sessions:
        occ = swc.occupancy(steps, s)
        if not len(occ):
            continue
        ev = uncover_table(steps[steps.session == s], occ)
        t0, t1 = event_windows(ev, variant)
        ok = np.isfinite(t1)
        ev, t0, t1 = ev[ok].reset_index(drop=True), t0[ok], t1[ok]
        if signal == "spikes":
            if s not in spk:
                continue
            feats = spike_features(s, t0, t1, spk, roi)
        else:
            c_s = contacts[contacts.session == s]
            if not len(c_s):
                continue
            store = sb.load_hfb(b_dir, s, arrays=[signal])
            h = hiv[hiv.session == s]
            clean_iv = {p: g[["start_s", "stop_s"]].to_numpy(float)
                        for p, g in h.groupby("pair_id")}
            exclude = None
            if variant.startswith("rip"):
                t_rip = np.sort(rip[rip.session == s].t_peak_s.to_numpy(float))
                t_rip = t_rip[np.concatenate([[True], np.diff(t_rip) > 0.05])]
                if variant.startswith("ripshift"):
                    t_rip = t_rip + rng.uniform(*SHIFT_RANGE_S) \
                        * rng.choice([-1, 1])
                exclude = ripple_mask(t_rip, store[signal].shape[1],
                                      store["out_fs"])
            feats = lfp_features(t0, t1, store, signal, c_s, clean_iv,
                                 exclude)
        test = np.flatnonzero(ev.first_discovery | (ev.phase == "execute"))
        for roiname, Z in feats.items():
            D = block_dots(Z, ev, test)
            e = ev.iloc[D.event.to_numpy()].reset_index(drop=True)
            D = pd.concat([D, e[["state", "run", "first_discovery"]]], axis=1)
            rows.append(D.assign(session=s, subject=subject[s], roi=roiname,
                                 n_cells=Z.shape[0]))
        print(f"  s{s:02d}", flush=True)
    E = pd.concat(rows, ignore_index=True)
    E["condition"] = np.where(E.first_discovery, "discovery", "execute")
    E.to_csv(os.path.join(out_dir, "per_event_block.csv.gz"), index=False)
    return E


def session_table(E):
    """Per session x ROI: Δ per condition x state, and the contrasts."""
    out = []
    for (s, r), g in E.groupby(["session", "roi"]):
        rec = dict(session=s, roi=r, subject=g.subject.iloc[0],
                   n_cells=int(g.n_cells.iloc[0]))
        for cond in ("discovery", "execute"):
            for subset, gg in (("all", g), ("run0", g[g.run == 0])):
                h = gg[gg.condition == cond]
                for k, S in enumerate(STATES):
                    rec[f"{cond}_{subset}_{S}"] = same_layout_beta(
                        h[h.state == k])
                rec[f"{cond}_{subset}_mean"] = same_layout_beta(h)
        for subset in ("all", "run0"):
            m = np.array([rec[f"discovery_{subset}_{S}"] for S in STATES])
            rec[f"D_minus_A_{subset}"] = m[3] - m[0]
            rec[f"slope_ABC_{subset}"] = (np.polyfit([0, 1, 2], m[:3], 1)[0]
                                          if np.isfinite(m[:3]).all()
                                          else np.nan)
        out.append(rec)
    return pd.DataFrame(out)


TESTS = {
    "PRIMARY: discovery D - A": "D_minus_A_all",
    "positive control: execute Δ > 0": "execute_all_mean",
    "heading-free slope over A, B, C": "slope_ABC_all",
    "first run only: discovery D - A": "D_minus_A_run0",
    "first run only: slope over A, B, C": "slope_ABC_run0",
    "discovery Δ > 0 (all states)": "discovery_all_mean",
}


def group_tests(S):
    out = []
    for roiname in S.roi.unique():
        g = S[S.roi == roiname]
        for name, col in TESTS.items():
            res = signflip_t(g[col])
            subj = signflip_t(g.groupby("subject")[col].mean())
            out.append(dict(roi=roiname, test=name, column=col, **res,
                            subj_n=subj["n"], subj_t=subj["t"],
                            subj_p_one_sided=subj["p_one_sided"]))
        for cond in ("discovery", "execute"):
            for S_ in STATES:
                col = f"{cond}_all_{S_}"
                out.append(dict(roi=roiname, test=f"{cond} Δ at {S_}",
                                column=col, **signflip_t(g[col])))
    return pd.DataFrame(out)


def save_settings(out_dir, sessions_used, signal="spikes", variant="full"):
    json.dump(dict(signal=signal, variant=variant,
                   min_premove_s=MIN_PREMOVE_S, ripple_mask_s=RIPPLE_MASK_S,
                   min_kept_samples=MIN_KEPT_SAMPLES,
                   shift_range_s=SHIFT_RANGE_S, seed=SEED, n_signflip=N_FLIP, window_s=WINDOW,
                   event_time="press = next step row (reward time), not arrival",
                   roi_sets=ROI_SETS if signal == "spikes" else LFP_ROI_SETS,
                   medial_max_abs_x_mm=MEDIAL_MAX_ABS_X,
                   min_features=MIN_CELLS,
                   min_template_events=MIN_TEMPLATE_EVENTS,
                   template="per-block execute-phase correct uncovers, own block excluded; "
                            "Δ = coef on same-layout in dot ~ same + |offset| + "
                            "|offset|^2 + offset, event-demeaned",
                   test_events="first discovery of each state per block "
                               "(encode); execute uncovers = positive control",
                   inference="per-session values, one-sample t, sign-flip P",
                   sessions=sorted(int(s) for s in sessions_used),
                   preregistration="CHANGELOG.md 2026-10-01",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(out_dir, "settings.json"), "w"), indent=2)
