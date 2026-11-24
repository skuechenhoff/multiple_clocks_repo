"""
Figures for `context_content` (D3, D4, D5), from its saved result.json.

Every panel shows session-level estimates as BARS from zero to the mean across
sessions, with the 95% t-interval as whiskers and the sign-flip p printed
above. Filled bar = p < .05, open bar = not. One colour per reading:

    still    level in still moments (no ripple within 0.5 s)
    ripple   level in ripples
    matched  ripple minus stratum-matched still moments   (baseline 1)
    own      ripple minus its own baseline window          (baseline 2)

    D3_square_content     which squares do HC ripples carry? current square,
                          route, rewarded squares
    D4c_code_levels       position (state) and location codes in still moments
                          and ripples, learning vs execution
    D4a_direction         which region's code comes first around a ripple
    D5_before_ripples     codes and their coupling BEFORE the ripple
    D4b_binding           location x position, together, in and out of ripples

    python -c "from mc.analyse.swr_explore.context_content_figures import draw; draw('<run folder>')"

@author: Svenja Kuchenhoff
"""

import os
import json
import textwrap

from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CM = 1 / 2.54
READ = {"still": ("still moments", "#7E6B8F"),
        "rip": ("ripples", "#5b9b8d"),
        "matched": ("ripple − matched still", "#a30d6c"),
        "own": ("ripple − own baseline", "#0e3d3a")}
SET_L = {"U_L": "uncover\nlearning", "U_E": "uncover\nexecution",
         "A_L": "away\nlearning", "A_E": "away\nexecution",
         "all_L": "all\nlearning", "all_E": "all\nexecution", "all": "all"}
SHORT = {"U_L": "unc.\nlearn", "U_E": "unc.\nexec", "A_L": "away\nlearn",
         "A_E": "away\nexec", "all_L": "all\nlearn", "all_E": "all\nexec",
         "all": "all", "L - E": "learn\n− exec"}
CODE_L = {"state": "position\n(full)", "sel": "position\n(selective)",
          "ramp": "position\n(ramp)", "loc": "location",
          "conj": "location x\nposition"}
Z_OF = {"still": "z_still", "rip": "z_rip", "matched": "z_matched",
        "own": "z_own"}
R4 = ("still", "rip", "matched", "own")


def _rc():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "pdf.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False})


def ptxt(p):
    return "p<.001" if p < .001 else f"p={p:.3f}".replace("0.", ".")


def ci(r):
    h = stats.t.ppf(0.975, r["n"] - 1) * r["sem"]
    return r["mean"] - h, r["mean"] + h


def bars(ax, R, groups, readings, key, labels, ylab, title, width=0.8):
    """groups: x categories; readings: reading names; key(g, rd) -> the
    result.json key. A missing key leaves a gap. Call ax.set_ylim first."""
    k = len(readings)
    w = width / k
    top = []
    for i, g in enumerate(groups):
        for j, rd in enumerate(readings):
            r = R.get(key(g, rd), {})
            if r.get("mean") is None or r.get("n", 0) < 3:
                continue
            x = i + (j - (k - 1) / 2) * w
            col = READ[rd][1]
            sig = r["p_perm"] < .05
            ax.bar(x, r["mean"], w * 0.9, color=col if sig else "white",
                   edgecolor=col, linewidth=1.0, zorder=2)
            lo, hi = ci(r)
            ax.plot([x, x], [lo, hi], color="0.15", lw=0.9, zorder=3)
            top.append((x, max(hi, 0), r["p_perm"], sig))
    ymin, ymax = ax.get_ylim()
    pad = 0.03 * (ymax - ymin)
    T = ymax + 0.45 * (ymax - ymin)
    for x, y, p, sig in top:
        # a whisker running off the axis is clipped; its label stays inside
        ax.text(x, min(y + pad, T - 0.32 * (T - ymin)), ptxt(p), rotation=90,
                ha="center", va="bottom", fontsize=7,
                color="0.1" if sig else "0.5")
    ax.set_ylim(ymin, T)
    ax.axhline(0, color="0.2", lw=0.8, zorder=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([labels.get(g, g) for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left")


def legend(fig, readings, y=0.0):
    h = [Patch(facecolor=READ[r][1], edgecolor=READ[r][1], label=READ[r][0])
         for r in readings]
    h += [Patch(facecolor="white", edgecolor="0.4", label="open: p ≥ .05")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=len(h), frameon=False, handlelength=1.2,
               columnspacing=1.0)


def caption(fig, text, y):
    w = int(fig.get_figwidth() * 2.54 / 0.15)
    fig.text(0.01, y, textwrap.fill(text, w), fontsize=7.5, va="top",
             color="0.2")


def save(fig, d, name):
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"{name}.{e}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(d, name + ".png"))


# ── D3: which squares do HC ripples carry ─────────────────────────────────
def draw_d3(d, R):
    Q = ("CURRENT", "ROUTE-OTHER", "REW-OTHER", "REW-ROUTE")
    QL = {"CURRENT": "current\nsquare", "ROUTE-OTHER": "route\n− other",
          "REW-OTHER": "reward\n− other", "REW-ROUTE": "reward\n− route"}
    zof = dict(Z_OF, still="z_ctrl")
    fig, axes = plt.subplots(2, 2, figsize=(18 * CM, 13 * CM), sharey=True,
                             gridspec_kw=dict(hspace=0.75, wspace=0.12))
    for ax, (st, ttl) in zip(axes.flat, (
            ("all_L", "a  learning, all ripples"),
            ("all_E", "b  execution, all ripples"),
            ("U_L", "c  learning, after an uncover"),
            ("A_E", "d  execution, away from uncovers"))):
        ax.set_ylim(-1.0, 2.0)
        bars(ax, R, Q, R4,
             lambda q, rd: f"D3 | HC_all | {q} | {st} | {zof[rd]}", QL,
             "HC evidence (z)" if ax in axes[:, 0] else "", ttl)
    legend(fig, R4, y=0.02)
    caption(fig, "Hippocampal cells (anterior + mid), T4 estimator: each "
            "cell's rate per square from the OTHER traversals of the same "
            "block, scored in ±125 ms around the ripple peak, z against 1000 "
            "square-label permutations, per session; bars = mean over "
            "sessions with 95% CI. current square = occupied square minus the "
            "other squares. route − other = non-rewarded squares on the "
            "block's execution route minus squares that are neither route nor "
            "reward; reward − other / reward − route = the configuration's "
            "reward squares minus those; all three compare squares at the "
            "SAME Manhattan distance (2–4) from the current square, so the "
            "spatial gradient cannot produce them. Learning = first traversal "
            "+ rest of encoding; uncover = in the pause after an uncover "
            "press (≤ 2 s); away = no uncover press within −2…+0.5 s. matched "
            "= ripple minus stratum-matched still moments (time since last / "
            "to next press) of the same context; own = ripple minus its own "
            "flanks (same square). p = sign-flip over sessions, uncorrected.",
            -0.06)
    save(fig, d, "D3_square_content")


# ── D4c: code levels ─────────────────────────────────────────────────────
def draw_d4c(d, R):
    regions = (("HC", "hippocampus"), ("mOFC", "mOFC"), ("mPFC", "mPFC"))
    codes = ("sel", "ramp", "loc")
    fig, axes = plt.subplots(2, 3, figsize=(18 * CM, 12 * CM), sharey=True,
                             gridspec_kw=dict(hspace=0.8, wspace=0.12))
    for col, (reg, ttl) in enumerate(regions):
        for row, (st, ph) in enumerate((("all_L", "learning"),
                                        ("all_E", "execution"))):
            ax = axes[row, col]
            ax.set_ylim(-1.0, 1.6)
            bars(ax, R, codes, R4,
                 lambda c, rd: f"D4c | {reg} | {c} | {st} | {Z_OF[rd]}",
                 CODE_L, f"{ph}\nevidence (z)" if col == 0 else "",
                 f"{'abcdef'[3 * row + col]}  {ttl}, {ph}")
    legend(fig, R4, y=0.02)
    caption(fig, "Evidence for the CURRENT position in the sequence (A–D) "
            "or the current square, in 250 ms windows: still moments (no "
            "ripple within 0.5 s) and ripples. Templates: each cell's rate "
            "per state (or square), leave-one-configuration-out, built "
            "within phase; position (selective) = the 4-state template minus "
            "its linear ramp across A–D; position (ramp) = that ramp. z vs "
            "all 23 state (100 square) label permutations, per session; bars "
            "= mean over sessions, 95% CI. This is NOT the published index "
            "(warped traces, goal progress and per-loop ramp removed, whole "
            "loop); see position_index_by_phase for that. p = sign-flip, "
            "uncorrected.", -0.06)
    save(fig, d, "D4c_code_levels")


# ── D4a: direction ───────────────────────────────────────────────────────
def draw_d4a(d, R):
    sets = ("U_L", "A_L", "U_E", "A_E", "all")
    fig, axes = plt.subplots(1, 3, figsize=(19 * CM, 7 * CM), sharey=True,
                             gridspec_kw=dict(wspace=0.12))
    for ax, (code, ttl) in zip(axes, (
            ("sel", "a  frontal position →\nHC position (selective)"),
            ("state", "b  frontal position →\nHC position (full)"),
            ("loc", "c  frontal position →\nHC location"))):
        ax.set_ylim(-0.12, 0.28)
        bars(ax, R, sets, ("matched", "own"),
             lambda s, rd: f"D4a | frontal -> HC {code} | {s} | {rd}", SHORT,
             "lead asymmetry, Fisher z\n(> 0: frontal first)"
             if ax is axes[0] else "", ttl)
    p = R.get("D4a | frontal -> HC state | U_L - A_E | own", {})
    prim = (f"Pre-registered primary (b, own): uncover-learning − "
            f"away-execution = {p['mean']:+.3f}, {ptxt(p['p_perm'])}; the "
            f"model predicted > 0. " if p else "")
    legend(fig, ("matched", "own"), y=-0.12)
    caption(fig, prim + "Per ripple: frontal (mPFC + mOFC) position evidence in "
            "the 450 ms BEFORE the ripple (−500…−50 ms) and HC evidence in "
            "the 450 ms AFTER (+50…+500 ms), and the reverse. Across the "
            "ripples of one session and context: asymmetry = z(ρ(frontal "
            "before, HC after)) − z(ρ(HC before, frontal after)). > 0: "
            "ripples preceded by a stronger frontal code are followed by a "
            "stronger HC code, more than the other way round. It is a "
            "lead-lag of code STRENGTH across ripples, not a timing of "
            "firing. own = minus the same with ripples paired at random; "
            "matched = minus the same computed around still moments. p = "
            "sign-flip over sessions, uncorrected.", -0.24)
    save(fig, d, "D4a_direction")


# ── D5: before the ripple ────────────────────────────────────────────────
def draw_d5(d, R):
    sets = ("all_L", "all_E", "U_L", "A_L", "U_E", "A_E")
    fig = plt.figure(figsize=(18 * CM, 19 * CM))
    gs = fig.add_gridspec(3, 2, hspace=0.75, wspace=0.42)
    axes = [[fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])],
            [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]]
    for ax, (reg, code, ttl) in zip(axes[0], (
            ("mOFC", "state", "a  mOFC position code, 450 ms BEFORE the ripple"),
            ("HC", "loc", "b  HC location code, 450 ms BEFORE the ripple"))):
        ax.set_ylim(-1.0, 1.6)
        bars(ax, R, sets, R4,
             lambda s, rd: f"D5 pre | {reg} | {code} | {s} | {Z_OF[rd]}",
             SHORT, "evidence (z)", ttl)
    pair = "HC loc x frontal state"
    ax = axes[1][0]
    ax.set_ylim(-0.12, 0.12)
    bars(ax, R, sets, ("matched", "own"),
         lambda s, rd: f"D5 pre | {pair} | {s} | dz_{rd}", SHORT,
         "coupling difference\n(Fisher z)",
         "c  HC location x frontal position:\nco-active BEFORE the ripple?")
    ax = axes[1][1]
    ax.set_ylim(-0.08, 0.08)
    lv = {"still": "level r_ctl_in", "rip": "level r_rip_in"}
    bars(ax, R, ("all_L", "all_E", "L - E"), ("still", "rip"),
         lambda s, rd: f"D4b | {pair} | {s} | {lv[rd]}",
         SHORT,
         "coupling level\n(Fisher z of partial r)",
         "d  the same coupling by phase\n(same window; in / out of ripples)")
    legend(fig, R4, y=0.02)
    caption(fig, "a, b: code evidence in the window −500…−50 ms before each "
            "ripple (and in the same window before still moments). own = "
            "minus the ripple's earlier window (−1000…−550 ms). c: across "
            "events, partial correlation of HC location evidence and frontal "
            "(mPFC + mOFC) position evidence in the pre-ripple window, given "
            "total activity; ripple − matched still moments, and ripple pre "
            "− ripple far-pre (own). d: the same coupling inside the ±125 ms "
            "window, as a level (Fisher z vs 0), in still moments and "
            "ripples, learning vs execution and their within-session "
            "difference. p = sign-flip, uncorrected.", -0.06)
    save(fig, d, "D5_before_ripples")


# ── D4b: binding ─────────────────────────────────────────────────────────
def draw_d4b(d, R):
    sets = ("all", "all_L", "all_E", "U_L", "A_L", "U_E", "A_E")
    fig, axes = plt.subplots(2, 1, figsize=(18 * CM, 13 * CM),
                             gridspec_kw=dict(hspace=0.75))
    ax = axes[0]
    ax.set_ylim(-1.0, 1.0)
    bars(ax, R, sets, R4, lambda s, rd: f"D4b-i | HC conj | {s} | {Z_OF[rd]}",
         SET_L, "conjunction\ninteraction (z)",
         "a  HC location x position conjunction, IN the ripple (±125 ms)")
    ax = axes[1]
    ax.set_ylim(-1.0, 1.0)
    bars(ax, R, sets, R4,
         lambda s, rd: f"D5 pre | HC | conj | {s} | {Z_OF[rd]}", SET_L,
         "conjunction\ninteraction (z)",
         "b  the same, BEFORE the ripple (−500…−50 ms)")
    legend(fig, R4, y=0.02)
    caption(fig, "Conjunction interaction = evidence for the true (square, "
            "position) cell − same square, other positions − same position, "
            "other squares + neither, over 36 leave-one-configuration-out "
            "templates; > 0 = the population carries the specific location x "
            "position combination beyond location and position separately; "
            "< 0 = less than their sum. z vs 200 permutations of the 36 "
            "labels, per session; bars = mean over sessions, 95% CI. own = "
            "minus the ripple's flanks (a) or its earlier window (b). p = "
            "sign-flip, uncorrected.", -0.06)
    save(fig, d, "D4b_binding")


def draw(d):
    _rc()
    R = json.load(open(os.path.join(d, "result.json")))["tests"]
    draw_d3(d, R)
    draw_d4c(d, R)
    draw_d4a(d, R)
    draw_d5(d, R)
    draw_d4b(d, R)
