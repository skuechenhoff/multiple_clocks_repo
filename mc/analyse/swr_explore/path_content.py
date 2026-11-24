"""
PATH CONTENT: after a correct uncover, do hippocampal ripples carry the path
just walked (storage, reverse) or the path ahead (planning, forward)?

    python scripts/swr_content_explore.py path_content

Pre-registered in CHANGELOG.md, 2026-10-29 (T4). Test T4 of the storage vs
next-step planning set (`scripts/exploratory/swr_plan_vs_store.py` has T1-T3,
T5).

Everything is `future_shift`'s machinery, with one change: it scores the
ripple window ITSELF rather than the change across it.

    R(L)  = per-cell rate on square L over the OTHER traversals of the block
    E(L)  = x_w . R(L),  x_w = z-scored spike counts in [t - 125, t + 125] ms
    PAST - FUT = mean E over past1..3  -  mean E over fut1..3

past/fut are the previous / next three distinct squares of the block's visit
sequence, so on a reward square the past is the route that led to the reward
and the future the route away from it. Reward IDENTITIES are not used: in a
4-state loop the reward just found is also the goal four steps ahead.

THE CONTROL. Every still moment with the same window geometry and no ripple
within 0.25 s, in the same stratum (phase x reward square x time since arrival
x time to departure). For the post-uncover set, controls come from the same
post-uncover pauses, so a ripple is compared with a moment in the same
situation that had none. Ripple and control go through the same permutation
of the nine square labels.

PRIMARY: HC units, post-correct-uncover ripples, first traversal, PAST - FUT,
ripple - control. Storage/reverse predicts > 0, planning/forward < 0.
Gate: NOW in control moments > 0.

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
import mc.analyse.swr_explore.future_shift as fs

SEED = 42
N_FLIP = 10000
HALF_S = 0.125            # ripple window: peak +- this
FLANK_S = (0.25, 0.50)    # same-event reading: the ripple's own flanks
SUBSETS = ("post_correct", "all")
PHASES = ("first_traversal", "execute", "all")
QUANT = ("PAST-FUT", "PAST", "FUT", "now")
PRIMARY = ("HC_all", "post_correct", "first_traversal", "PAST-FUT")


def place(t, occ, t_avoid, pad):
    """`future_shift.place_events` for ONE centred window of +-HALF_S."""
    a, b = occ.start_s.to_numpy(), occ.stop_s.to_numpy()
    j = np.searchsorted(a, t, side="right") - 1
    jj = np.clip(j, 0, len(a) - 1)
    fits = (j >= 0) & (t - HALF_S >= a[jj]) & (t + HALF_S <= b[jj])
    av = np.sort(np.asarray(t_avoid, float))
    n_near = (np.searchsorted(av, t + HALF_S + pad, side="right")
              - np.searchsorted(av, t - HALF_S - pad, side="left"))
    return jj, fits, n_near, t - a[jj], b[jj] - t


def controls(occ):
    a = occ.start_s.to_numpy() + HALF_S
    b = occ.stop_s.to_numpy() - HALF_S
    out = [np.arange(x, y + 1e-9, fs.CTRL_STEP_S) for x, y in zip(a, b) if y >= x]
    return np.concatenate(out) if out else np.empty(0)


def in_pause(t, u, still):
    """True where t lies in (u, u + still] of some correct uncover."""
    if not len(u):
        return np.zeros(len(t), bool)
    o = np.argsort(u)
    u, e = u[o], (u + still)[o]
    k = np.searchsorted(u, t, side="left") - 1
    kk = np.clip(k, 0, None)
    return (k >= 0) & (t <= e[kk])


def session_events(steps, s, t_rip, d_rip, u, still):
    occ = swc.occupancy(steps, s)
    if not len(occ):
        return None
    RT = fs.role_table(steps[steps.session == s], occ)
    out = {"occ": occ, "roles": RT}
    for kind in ("ripple", "control"):
        if kind == "ripple":
            t = t_rip[d_rip <= fs.MAX_RIP_DUR_S]
            jj, fits, n_near, tsa, ttd = place(t, occ, t_rip, fs.OTHER_RIP_PAD_S)
            ok = fits & (n_near == 1)
        else:
            t = controls(occ)
            jj, fits, n_near, tsa, ttd = place(t, occ, t_rip, fs.CTRL_RIP_PAD_S)
            ok = fits & (n_near == 0)
        ok &= RT.phase.to_numpy()[jj] != ""
        ev = RT.iloc[jj[ok]].reset_index(drop=True)
        ev["t"] = t[ok]
        ev["stratum"] = fs.stratum(ev.phase, ev.is_rew, tsa[ok], ttd[ok])
        ev["post_correct"] = in_pause(t[ok], u, still)
        if kind == "ripple":
            # SAME-EVENT reading: both own flanks on the same square, and no
            # other ripple within them
            av = np.sort(t_rip)
            far = FLANK_S[1] + fs.OTHER_RIP_PAD_S
            n_f = (np.searchsorted(av, t[ok] + far, side="right")
                   - np.searchsorted(av, t[ok] - far, side="left"))
            ev["flank_ok"] = ((tsa[ok] >= FLANK_S[1]) & (ttd[ok] >= FLANK_S[1])
                              & (n_f == 1))
        out[kind] = ev
    return out


def weights(ev, V):
    w = fs.quantity_weights(ev, V)
    return {"PAST-FUT": (-w["J"][0], w["J"][1]), "PAST": w["PAST"],
            "FUT": w["FUT"], "now": w["now"]}


def roi_stats(ev, Xr, Xc, refs, P, Xr2=None, Xf=None):
    """Stillness-matched reading (Xr vs Xc, pre-registered) and, if given, the
    same-event reading (Xr2 vs the same ripple's own flanks Xf, z-scored
    together)."""
    rip, ctl = ev["ripple"], ev["control"]
    Er, Vr, _ = fs.evidence(Xr, rip, refs)
    Ec, Vc, _ = fs.evidence(Xc, ctl, refs)
    if Xf is not None:
        Er2, _, _ = fs.evidence(Xr2, rip, refs)
        Ef, _, _ = fs.evidence(Xf, rip, refs)
        hf = np.isfinite(Er2).all(1) & np.isfinite(Ef).all(1) \
            & rip.flank_ok.to_numpy()
        Er2, Ef = np.nan_to_num(Er2), np.nan_to_num(Ef)
    hr, hc = np.isfinite(Er).all(1), np.isfinite(Ec).all(1)
    Er, Ec = np.nan_to_num(Er), np.nan_to_num(Ec)
    wr, wc = weights(rip, Vr), weights(ctl, Vc)
    sr, sc = rip.stratum.to_numpy(), ctl.stratum.to_numpy()
    recs = {}
    for sub in SUBSETS:
        mr = rip.post_correct.to_numpy() if sub == "post_correct" \
            else np.ones(len(rip), bool)
        mc = ctl.post_correct.to_numpy() if sub == "post_correct" \
            else np.ones(len(ctl), bool)
        for q in QUANT:
            W_r, ok_r = wr[q]
            W_c, ok_c = wc[q]
            ok_c = ok_c & hc & mc
            Cs, cnt = fs.stratum_means(W_c, Ec, sc, ok_c, fs.N_STRATA, P)
            valid = ok_r & hr & mr & (cnt[sr] > 0)
            Rp = fs.permuted(W_r[valid], Er[valid][:, P])
            Cp = Cs[sr[valid]]
            ph = rip.phase.to_numpy()[valid]
            for p_ in PHASES:
                m = np.ones(len(Rp), bool) if p_ == "all" else ph == p_
                rec = dict(n_rip=int(m.sum()))
                if m.sum() >= fs.MIN_EVENTS:
                    rec.update(z_diff=float(fs._z((Rp[m] - Cp[m]).mean(0))),
                               z_rip=float(fs._z(Rp[m].mean(0))),
                               z_ctrl=float(fs._z(Cp[m].mean(0))))
                recs[(sub, q, p_)] = rec
            if Xf is not None:
                vf = ok_r & hf & mr
                Dp = fs.permuted(W_r[vf], (Er2[vf] - Ef[vf])[:, P])
                phf = rip.phase.to_numpy()[vf]
                for p_ in PHASES:
                    m = np.ones(len(Dp), bool) if p_ == "all" else phf == p_
                    recs[(sub, q, p_)]["n_rip_flank"] = int(m.sum())
                    if m.sum() >= fs.MIN_EVENTS:
                        recs[(sub, q, p_)]["z_ownflank"] = float(
                            fs._z(Dp[m].mean(0)))
        # gate: the current square in ordinary still moments of this subset
        W_c, ok_c = wc["now"]
        ok_c = ok_c & hc & mc
        if ok_c.sum() >= fs.MIN_EVENTS:
            Lv, _ = fs.stratum_means(W_c, Ec, np.zeros(len(ctl), int), ok_c,
                                     1, P)
            for q in QUANT:
                for p_ in PHASES:
                    recs[(sub, q, p_)]["gate_now_ctrl"] = float(fs._z(Lv[0]))
    return recs


def group(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size < 3:
        return dict(n=int(v.size))
    t, p = stats.ttest_1samp(v, 0.0)
    rng = np.random.default_rng(SEED)
    null = (rng.choice([-1.0, 1.0], size=(N_FLIP, v.size)) * v).mean(1)
    return dict(n=int(v.size), mean=float(v.mean()),
                sem=float(v.std(ddof=1) / np.sqrt(v.size)), t=float(t),
                p=float(p), p_perm=float((1 + np.sum(np.abs(null)
                                                     >= abs(v.mean())))
                                         / (1 + N_FLIP)))


def main():
    rng0 = np.random.default_rng(SEED)
    P = np.vstack([np.arange(9)] + [rng0.permutation(9)
                                    for _ in range(fs.N_PERM)])
    deriv = rrsa._derivatives()
    swr = os.path.join(deriv, "group", "swr")
    rip = pd.read_csv(os.path.join(swr, "bundle_v2", "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s",
                               "subject_key"])
    presses = pd.read_csv(os.path.join(swr, "press_categories.csv"))
    cu = presses[(presses.kind == "uncover") & (presses.valence == "correct")
                 & np.isfinite(presses.still_next_s)]
    subject = dict(zip(rip.session, rip.subject_key))
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)

    rows, counts = [], []
    for s in sessions:
        if s not in spk:
            continue
        t_rip, d_rip = fs._ripples(rip, s)
        c = cu[cu.session == s]
        ev = session_events(steps, s, t_rip, d_rip, c.t_s.to_numpy(float),
                            c.still_next_s.to_numpy(float))
        if ev is None or len(ev["ripple"]) < fs.MIN_EVENTS:
            continue
        counts.append(dict(session=s, n_ripple=len(ev["ripple"]),
                           n_ripple_post_correct=int(ev["ripple"]
                                                     .post_correct.sum()),
                           n_control=len(ev["control"]),
                           n_control_post_correct=int(ev["control"]
                                                      .post_correct.sum())))
        r = roi[roi.session == s]
        for name, members in fs.SPIKE_ROIS.items():
            cells = [c_ for c_ in r[r.roi.isin(members)].cell
                     if c_ < len(spk[s]["spikes"])]
            if len(cells) < fs.MIN_FEATURES:
                continue
            sums, win, min_expo = fs.spike_signal(spk[s], cells)
            S_, X_ = sums(ev["occ"])
            refs = fs.references(S_, X_, ev["occ"], ev["roles"], None,
                                 min_expo)
            tr, tc = ev["ripple"].t.to_numpy(), ev["control"].t.to_numpy()
            Xr, Xc = fs.zscore_features([win(tr - HALF_S, tr + HALF_S),
                                         win(tc - HALF_S, tc + HALF_S)])
            # same-event reading: ripple vs its own two flanks (mean count of
            # the two 250 ms windows), z-scored together
            Xr2, Xf = fs.zscore_features([
                win(tr - HALF_S, tr + HALF_S),
                (win(tr - FLANK_S[1], tr - FLANK_S[0])
                 + win(tr + FLANK_S[0], tr + FLANK_S[1])) / 2.0])
            for (sub, q, p_), rec in roi_stats(ev, Xr, Xc, refs, P,
                                               Xr2, Xf).items():
                rows.append(dict(session=s, subject=subject[s], roi=name,
                                 subset=sub, quantity=q, phase=p_,
                                 n_cells=len(cells), **rec))
        print(f"  s{s:02d}  ripples {len(ev['ripple'])} "
              f"(post-correct {int(ev['ripple'].post_correct.sum())})",
              flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(swr, f"plan_vs_store_{datetime.date.today()}",
                     "T4_path_content")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "per_session.csv"), index=False)
    pd.DataFrame(counts).to_csv(os.path.join(d, "event_counts.csv"),
                                index=False)
    S = []
    for (roi_, sub, q, p_), g in R.groupby(["roi", "subset", "quantity",
                                            "phase"]):
        for meas in ("z_diff", "z_rip", "z_ctrl", "gate_now_ctrl",
                     "z_ownflank"):
            if meas in g:
                S.append(dict(roi=roi_, subset=sub, quantity=q, phase=p_,
                              measure=meas, **group(g[meas].to_numpy(float))))
    S = pd.DataFrame(S)
    S.to_csv(os.path.join(d, "summary.csv"), index=False)
    prim = S[(S.roi == PRIMARY[0]) & (S.subset == PRIMARY[1])
             & (S.phase == PRIMARY[2]) & (S.quantity == PRIMARY[3])]
    print("\n" + prim.round(4).to_string(index=False))
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(dict(
            settings=dict(analysis="T4_path_content",
                          created=str(datetime.date.today()),
                          module="mc/analyse/swr_explore/path_content.py",
                          seed=SEED, n_perm=fs.N_PERM, n_flip=N_FLIP,
                          window_s=[-HALF_S, HALF_S],
                          control_ripple_pad_s=fs.CTRL_RIP_PAD_S,
                          other_ripple_pad_s=fs.OTHER_RIP_PAD_S,
                          max_ripple_duration_s=fs.MAX_RIP_DUR_S,
                          references="other traversals of the same block",
                          strata="phase x reward square x time since arrival "
                                 "x time to departure",
                          post_correct="ripples and controls both inside "
                                       "(uncover, next press] of a correct "
                                       "uncover",
                          primary=list(PRIMARY), unit="session",
                          preregistration="CHANGELOG.md 2026-10-29 T4"),
            summary=S.to_dict("records")), f, indent=2, default=str)
    phase_contrast(d)
    _figure(S, R, d)
    print(f"-> {d}")
    return d


def phase_contrast(d):
    """POST HOC (not pre-registered): first traversal - execution, paired over
    sessions, from the saved per-session table."""
    R = pd.read_csv(os.path.join(d, "per_session.csv"))
    rows = []
    for (roi_, sub, q), g in R.groupby(["roi", "subset", "quantity"]):
        for meas in ("z_diff", "z_ownflank"):
            if meas not in g:
                continue
            w = g.pivot_table(index="session", columns="phase", values=meas)
            if not {"first_traversal", "execute"} <= set(w.columns):
                continue
            v = (w.first_traversal - w.execute).dropna().to_numpy()
            rows.append(dict(roi=roi_, subset=sub, quantity=q, measure=meas,
                             contrast="first_traversal - execute", **group(v)))
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(d, "phase_contrast_posthoc.csv"), index=False)
    return C


def _figure(S, R, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CM = 1 / 2.54
    col = {"PAST-FUT": "#0e3d3a", "PAST": "#5b9b8d", "FUT": "#a7d9b2",
           "now": "#7eb1c4"}
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 3, figsize=(22 * CM, 7 * CM), sharey=True)
    for ax, roi_ in zip(axes, ("HC_all", "mPFC", "mOFC")):
        xt, xl = [], []
        for i, (sub, p_) in enumerate((("post_correct", "first_traversal"),
                                       ("post_correct", "execute"),
                                       ("all", "first_traversal"),
                                       ("all", "execute"))):
            for k, q in enumerate(QUANT):
                g = R[(R.roi == roi_) & (R.subset == sub) & (R.phase == p_)
                      & (R.quantity == q)]
                v = g.z_diff.dropna().to_numpy() if "z_diff" in g else []
                x = i * 5 + k
                if len(v):
                    ax.scatter(x + rng.uniform(-0.15, 0.15, len(v)), v, s=6,
                               color=col[q], alpha=0.5, lw=0)
                    ax.errorbar(x, v.mean(), v.std(ddof=1) / np.sqrt(len(v)),
                                fmt="o", ms=4.5, mfc=col[q], mec="#1a1a1a",
                                ecolor="#1a1a1a", capsize=2)
                    r = S[(S.roi == roi_) & (S.subset == sub) & (S.phase == p_)
                          & (S.quantity == q) & (S.measure == "z_diff")]
                    if len(r) and np.isfinite(r.p_perm.iloc[0]):
                        ax.text(x, 1.0, f"{r.p_perm.iloc[0]:.2f}", fontsize=5.5,
                                ha="center", rotation=90,
                                transform=ax.get_xaxis_transform())
            xt.append(i * 5 + 1.5)
            xl.append(f"{'post-correct' if sub == 'post_correct' else 'all'}\n"
                      f"{'first trav.' if p_ == 'first_traversal' else 'execution'}")
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_xticks(xt, xl, fontsize=7)
        ax.set_title(roi_.replace("HC_all", "hippocampus"), fontsize=10, pad=18)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("ripple − matched still moment\n(z vs label permutation)")
    hs = [plt.Line2D([], [], marker="o", ls="", mfc=col[q], mec="#1a1a1a",
                     label=q.replace("PAST-FUT", "PAST − FUT")) for q in QUANT]
    fig.legend(handles=hs, fontsize=7, frameon=False, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, -0.02))
    fig.text(0.01, -0.2,
             "Spike evidence in the ripple window (±125 ms) for the squares "
             "just walked (PAST, previous 3) and about to be walked (FUT, next "
             "3), against same-block references, minus\nstratum-matched still "
             "moments without a ripple (post-correct: both inside the pause "
             "after a correct uncover). Dots: sessions; marker: mean ± SEM; "
             "numbers: sign-flip p.", fontsize=7, color="0.3")
    fig.tight_layout()
    for e in (".png", ".pdf"):
        fig.savefig(os.path.join(d, "T4_path_content" + e), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
