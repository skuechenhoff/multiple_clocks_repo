"""
C1 REACTIVATION: do hippocampal ripples during LEARNING reactivate the
co-firing structure that medial frontal cells will show once the block is
EXECUTED?

    python scripts/swr_content_explore.py reactivation

Pre-registered in CHANGELOG.md, 2026-11-14 (b). No tuning curve, template map or
cell classification -- only which cells fire together (Peyrache et al. 2009,
reactivation strength).

Per session, every cell's spike counts in 250 ms bins over the task, z-scored
with session-wide statistics.
    C_b   template = cell x cell Pearson correlation of those bins over block
          b's EXECUTION (bins within 0.5 s of a ripple dropped), diagonal 0
    R(t)  = z(t)' C z(t) / (n (n - 1)),  z(t) = counts in [t - 125, t + 125] ms
Templates
    own     C_b of the block the moment belongs to
    other   mean of C_b' over blocks with a DIFFERENT configuration
    all     every execution bin of the session (general reactivation)
    specific = own - other: co-firing that belongs to THIS configuration's plan
Readings
    own      ripple minus its own flanks ([-0.75, -0.5] and [+0.5, +0.75] s)
    matched  ripple minus ripple-free moments of the same window (no ripple
             within 0.5 s, 50 ms grid) in the same stratum of time since the
             last press x time to the next one
Ripple sets: learning ripples of each block (PRIMARY), ripples in the pause
after a first discovery. Execution ripples are NOT tested: their window is the
one the template is built from, so the comparison would be circular (dropped
from the pre-registered secondaries for that reason).

PRIMARY: mPFC (sessions with >= 3 cells), learning ripples, specific, matched.

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import datetime

import numpy as np
import pandas as pd

import mc.analyse.ripple_rsa as rrsa
from mc.analyse.swr_explore.discovery_cells import load_inputs, group, FD_Q, \
    MAX_PAUSE_S

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "exploratory"))

SEED = 42
BIN_S = 0.25
HALF_S = 0.125
FLANK_S = (0.5, 0.75)
RIP_EXCL_S = 0.5
MIN_CELLS = 3
MIN_TPL_BINS = 40
MIN_RIP = 10
ROI_SETS = {"mPFC": ["mPFC"], "mOFC": ["mOFC"],
            "mPFC+mOFC": ["mPFC", "mOFC"], "HC_all": ["HC_anterior", "HC_mid"]}
TEMPLATES = ("own", "other", "all")


def counts(st, a, b):
    return np.searchsorted(st, b) - np.searchsorted(st, a)


def zwin(spk, t, mu, sd):
    """(n_events, n_cells) z-scored counts in [t - HALF, t + HALF]."""
    return np.column_stack([(counts(s, t - HALF_S, t + HALF_S) - m) / d
                            for s, m, d in zip(spk, mu, sd)])


def template(Z):
    if len(Z) < MIN_TPL_BINS:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(Z.T)
    C = np.nan_to_num(C)
    np.fill_diagonal(C, 0.0)
    return C


def R(Z, C):
    n = Z.shape[1]
    return np.einsum("ei,ij,ej->e", Z, C, Z) / (n * (n - 1))


def session_values(s, spk_s, cells, blocks, t_rip, presses):
    from swr_hfb_discovery_lock import strata, control_times, MIN_CTRL
    from swr_hfb_discovery_lock import _windows_mask
    spk = [np.sort(np.asarray(spk_s["spikes"][c], float)) for c in cells]
    t0, t1 = blocks.t_learn0.min(), blocks.t_end.max()
    edges = np.arange(t0, t1 + BIN_S, BIN_S)
    centres = edges[:-1] + BIN_S / 2
    Cnt = np.column_stack([np.histogram(s_, edges)[0] for s_ in spk]).astype(float)
    mu, sd = Cnt.mean(0), Cnt.std(0)
    if (sd == 0).any():
        keep = sd > 0
        spk = [x for x, k in zip(spk, keep) if k]
        Cnt, mu, sd = Cnt[:, keep], mu[keep], sd[keep]
    if len(spk) < MIN_CELLS:
        return None
    Zb = (Cnt - mu) / sd
    r_sorted = np.sort(t_rip)
    near = (np.searchsorted(r_sorted, centres + RIP_EXCL_S)
            - np.searchsorted(r_sorted, centres - RIP_EXCL_S)) > 0
    tp = np.sort(presses.t_s.to_numpy(float))
    tpl, cfg = {}, {}
    exe_all = np.zeros(len(centres), bool)
    for _, b in blocks.iterrows():
        m = (centres >= b.t_exe0) & (centres < b.t_end) & ~near
        exe_all |= m
        C = template(Zb[m])
        if C is not None:
            tpl[b.grid_no], cfg[b.grid_no] = C, b.cfg
    C_all = template(Zb[exe_all])
    fd = presses.query(FD_Q).dropna(subset=["still_next_s"])
    fd_t, fd_s = fd.t_s.to_numpy(float), fd.still_next_s.to_numpy(float)
    acc = {}
    for _, b in blocks.iterrows():
        g = b.grid_no
        others = [tpl[k] for k in tpl if cfg[k] != b.cfg]
        if g not in tpl or not others or C_all is None:
            continue
        Cs = {"own": tpl[g], "other": np.mean(others, 0), "all": C_all}
        learn = [(b.t_learn0, b.t_exe0)]
        sel = (fd_t >= b.t_learn0) & (fd_t < b.t_exe0)
        pause = [(t, t + min(x, MAX_PAUSE_S)) for t, x in zip(fd_t[sel],
                                                              fd_s[sel])]
        for set_name, period in (("learning", learn), ("post_FD", pause)):
            tr = t_rip[_windows_mask(t_rip, period)] if period else np.empty(0)
            if not len(tr):
                continue
            tc = control_times(period, t_rip)
            Zr = zwin(spk, tr, mu, sd)
            Zf1 = zwin(spk, tr - np.mean(FLANK_S), mu, sd)
            Zf2 = zwin(spk, tr + np.mean(FLANK_S), mu, sd)
            Zc = zwin(spk, tc, mu, sd) if len(tc) else None
            sr, sc = strata(tr, tp), strata(tc, tp) if len(tc) else None
            for k, C in Cs.items():
                rr = R(Zr, C)
                own_v = rr - 0.5 * (R(Zf1, C) + R(Zf2, C))
                a = acc.setdefault((set_name, k), dict(own=[], n_own=0,
                                                       matched=[], n_m=0))
                a["own"].append(own_v.sum())
                a["n_own"] += len(own_v)
                if Zc is not None:
                    rc = R(Zc, C)
                    for q in np.unique(sr):
                        cq = sc == q
                        if cq.sum() < MIN_CTRL:
                            continue
                        rq = sr == q
                        a["matched"].append((rr[rq] - rc[cq].mean()).sum())
                        a["n_m"] += int(rq.sum())
    out = {}
    for (set_name, k), a in acc.items():
        out[(set_name, k, "own")] = (sum(a["own"]) / a["n_own"]
                                     if a["n_own"] >= MIN_RIP else np.nan,
                                     a["n_own"])
        out[(set_name, k, "matched")] = (sum(a["matched"]) / a["n_m"]
                                         if a["n_m"] >= MIN_RIP else np.nan,
                                         a["n_m"])
    for set_name in ("learning", "post_FD"):
        for rd in ("own", "matched"):
            if (set_name, "own", rd) in out:
                v_own, n = out[(set_name, "own", rd)]
                v_oth, _ = out[(set_name, "other", rd)]
                out[(set_name, "specific", rd)] = (v_own - v_oth, n)
    return out


def main():
    from plan_code_vs_ripples import block_table
    deriv = rrsa._derivatives()
    d = os.path.join(deriv, "group", "swr",
                     f"reactivation_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    data, presses, _ = load_inputs()
    beh, rip = data["behaviour"], data["ripples"]
    sessions = sorted(presses.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    rows = []
    for s in sessions:
        if s not in spk:
            continue
        t_rip = np.unique(np.round(rip[rip.session == s].t_peak_s
                                   .to_numpy(float), 3))
        blocks = block_table(beh[beh.session == s], t_rip, np.empty(0))
        if len(blocks) < 3:
            continue
        P = presses[presses.session == s]
        for rname, members in ROI_SETS.items():
            cl = [c for c in roi[(roi.session == s) & roi.roi.isin(members)]
                  .cell.to_numpy() if c < len(spk[s]["spikes"])]
            if len(cl) < MIN_CELLS:
                continue
            v = session_values(s, spk[s], cl, blocks, t_rip, P)
            if not v:
                continue
            for (set_name, k, rd), (val, n) in v.items():
                rows.append(dict(session=int(s), roi=rname, n_cells=len(cl),
                                 ripples=set_name, template=k, reading=rd,
                                 value=val, n=n))
        print(f"  reactivation s{s:02d}", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(d, "per_session.csv"), index=False)
    res = {}
    for (rname, set_name, k, rd), g in T.groupby(["roi", "ripples",
                                                    "template", "reading"]):
        res[f"{rname} | {set_name} | {k} | {rd}"] = group(
            g.value.to_numpy(float))
    prim = res.get("mPFC | learning | specific | matched")
    print(f"\n  PRIMARY mPFC | learning | specific | matched: {prim}")
    for k_, v in res.items():
        print(f"  {k_:46s} n={v.get('n')} {v.get('mean', np.nan):+.5f} "
              f"p={v.get('p_perm', np.nan):.3f}")

    def js(o):
        return o.item() if hasattr(o, "item") else str(o)
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(dict(settings=dict(
            analysis="reactivation (C1)", created=str(datetime.date.today()),
            module="mc/analyse/swr_explore/reactivation.py", seed=SEED,
            bin_s=BIN_S, window_s=2 * HALF_S, flank_s=FLANK_S,
            ripple_exclusion_s=RIP_EXCL_S, min_cells=MIN_CELLS,
            min_template_bins=MIN_TPL_BINS, min_ripples=MIN_RIP,
            roi_sets=ROI_SETS,
            measure="Peyrache 2009 reactivation strength z'Cz/(n(n-1))",
            templates=dict(own="block's own execution co-firing",
                           other="mean over blocks of other configurations",
                           all="all execution bins of the session",
                           specific="own - other"),
            readings=dict(own="ripple - its own flanks",
                          matched="ripple - stratum-matched ripple-free "
                                  "moments of the same window"),
            deviation="execution ripples dropped (circular: template window)",
            preregistration="CHANGELOG.md 2026-11-14 (b) C1"),
            tests=res), f, indent=1, default=js)
    from mc.analyse.swr_explore.reactivation_figures import draw
    draw(d)
    print(f"-> {d}")
    return d
