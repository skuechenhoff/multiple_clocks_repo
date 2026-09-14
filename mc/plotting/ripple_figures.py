#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication figures for the hippocampal sharp-wave-ripple analysis.

The main entry point, `chen_figure`, reproduces the layout of Chen et al. 2025
(J Neurosci 45:e1502252025) Figure 2a-b, which is the field's standard evidence
that a ripple detector is finding ripples:

    a  hippocampal contacts on a template brain
    b  grand-average ripple-locked voltage + time-frequency decomposition,
       and the same for one example ripple, with the marginal amplitude
       spectrum drawn over the TFR

`sharp_wave_figure` is not in Chen -- it exists because "sharp-wave ripple"
names a two-part event and the sharp wave deserves to be looked at rather than
assumed. See the note on polarity in `sharp_wave_profiles`.

@author: Svenja Kuchenhoff
"""

import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import gridspec
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.fft import next_fast_len

import era_brewer

# ---------------------------------------------------------------- style ----
# These become subpanels of a half-A4 figure, so the whole thing is drawn at
# roughly the size it will be printed: ~18 cm wide, and font sizes are then
# real point sizes rather than something to be rescaled later.
CM = 1 / 2.54
FS_TITLE, FS_LABEL, FS_TICK = 11, 10, 9

PAL = era_brewer.era_brew("Showgirl2", n=7)
RIPPLE_C = "#B74C2D"        # ripple-band trace
RAW_C = "#3d3d3d"           # broadband trace
SW_C = "#448363"            # sharp wave
ROI_C = {"HC_anterior": "#23677E", "HC_mid": PAL[2], "HC": PAL[2],
         "HC_posterior": "#0e3d3a"}

RIPPLE_BAND = (80.0, 120.0)
SW_BAND_HZ = 20.0           # low-pass for the sharp wave; see sharp_wave_profiles
TFR_FREQS = np.arange(4.0, 176.0, 2.0)
TFR_BASELINE_S = (-1.5, -0.5)


def _rc():
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    }


# ------------------------------------------------------------- signals ----
def _bp(x, fs, lo, hi, order=4):
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def _lp(x, fs, hi, order=4):
    sos = butter(order, hi / (fs / 2), btype="low", output="sos")
    return sosfiltfilt(sos, x)


def trough_lock(events, ripple_band_signal):
    """Sample index of each event's largest negative ripple-band deflection.

    Detection marks each event at the peak of its RMS *envelope*, which is the
    right definition for counting events but the wrong one for averaging them:
    the envelope says nothing about the phase of the underlying oscillation, so
    a grand average built on it cancels almost completely. Measured on the
    development set, envelope-locking retains 7% of the single-event ripple
    amplitude in the average; locking to the trough retains 95%.

    This is a display choice for the ripple-triggered average only. Detection,
    event counts, rates and every statistic are untouched by it.
    """
    out = []
    for _, r in events.iterrows():
        a, b = int(r.start_sample), int(r.stop_sample)
        if b <= a:
            out.append(int(r.peak_sample))
        else:
            out.append(a + int(np.argmin(ripple_band_signal[a:b])))
    return np.asarray(out, int)


def _snips(x, peaks, half):
    keep = [p for p in peaks if p - half >= 0 and p + half < len(x)]
    if not keep:
        return np.zeros((0, 2 * half))
    return np.stack([x[p - half:p + half] for p in keep])


def ripple_tfr(raw, fs, peaks, freqs=TFR_FREQS, win_s=0.25,
               baseline_s=TFR_BASELINE_S, n_cycles=6.0):
    """Ripple-locked TFR as % change from a pre-event baseline.

    Chen's normalisation: the baseline is taken from -1.5 to -0.5 s around each
    event, not from the edges of the display window. That matters -- a ripple is
    a transient sitting on 1/f activity, and normalising to the window edges
    leaves the low frequencies looking suppressed simply because the window is
    short.

    Returns (tfr_pct, times_ms, freqs).
    """
    half = int(round(win_s * fs))
    b0, b1 = int(round(baseline_s[0] * fs)), int(round(baseline_s[1] * fs))
    peaks = np.asarray([p for p in peaks
                        if p + b0 >= 0 and p + half < len(raw)], int)
    tfr = np.full((len(freqs), 2 * half), np.nan)
    if not len(peaks):
        return tfr, np.linspace(-win_s, win_s, 2 * half) * 1000, freqs

    for j, f0 in enumerate(freqs):
        bw = max(2.0, f0 / n_cycles * 2.0)
        lo, hi = max(1.0, f0 - bw / 2), min(fs / 2 - 1.0, f0 + bw / 2)
        if hi <= lo:
            continue
        sos = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
        # hilbert FFTs at the signal's own length, which for a real recording is
        # an awkward number (2433051 for s02) and costs 1.08 s per frequency
        # against 0.14 s at the next fast length -- 8x, and this runs once per
        # frequency per derivation. The zero-padding perturbs only the edges:
        # measured on s02, median difference 8e-8 uV and interior max 1.1e-5 uV
        # against a 0.35 uV median envelope. Events near the edges are excluded
        # from the average anyway.
        amp = np.abs(hilbert(sosfiltfilt(sos, raw), N=next_fast_len(len(raw)))[:len(raw)])
        ev = _snips(amp, peaks, half)
        base = np.array([amp[p + b0:p + b1].mean() for p in peaks])
        ok = base > 0
        if ok.sum():
            tfr[j] = ((ev[ok] - base[ok, None]) / base[ok, None] * 100).mean(0)

    t_ms = np.linspace(-win_s, win_s, 2 * half) * 1000
    return tfr, t_ms, freqs


# --------------------------------------------------------- sharp wave ----
def sharp_wave_profiles(raw, fs, peaks, sw_hz=SW_BAND_HZ, win_s=0.25):
    """Ripple-locked low-frequency deflection, per contact.

    Two things make the sharp wave hard to see in this montage, and both are
    properties of the recording rather than of the detector:

    1. Its polarity depends on where the contact sits relative to the CA1
       pyramidal layer -- the dipole reverses across it. Averaging contacts
       without regard to sign therefore cancels a real signal.
    2. Bipolar re-referencing between neighbouring contacts is a spatial
       derivative, and the sharp wave is a spatially broad, low-frequency
       dipole. Much of it is common to both contacts and subtracts away, while
       the spatially focal ripple survives. This is why Chen's own grand
       average shows a ripple with no sharp wave under it.

    So this returns the per-contact profile and its sign, and leaves the
    interpretation to the caller. Nothing here flips anything.
    """
    half = int(round(win_s * fs))
    sw = _lp(raw, fs, sw_hz)
    sn = _snips(sw, peaks, half)
    if not len(sn):
        return None
    m = sn.mean(0)
    c, w = half, int(round(0.025 * fs))
    edge = int(round(0.05 * fs))
    flank = np.r_[m[:edge], m[-edge:]]
    return {
        "mean": m,
        "sem": sn.std(0) / np.sqrt(len(sn)),
        "n": len(sn),
        "deflection_uv": float(m[c - w:c + w].mean()),
        "flank_sd": float(flank.std()),
        "snr": float(m[c - w:c + w].mean() / flank.std()) if flank.std() else np.nan,
        "t_ms": np.linspace(-win_s, win_s, 2 * half) * 1000,
    }


# ------------------------------------------------------------- panels ----
def _panel_glassbrain(ax, coords, rois=None, title="Hippocampal contacts"):
    """Contacts on a glass brain. Falls back to a scatter if nilearn is absent."""
    coords = np.atleast_2d(np.asarray(coords, float))
    try:
        from nilearn import plotting as nlplot
        disp = nlplot.plot_glass_brain(
            None, display_mode="lyrz", axes=ax, black_bg=False,
            alpha=0.12, colorbar=False)
        if rois is None:
            disp.add_markers(coords, marker_color=ROI_C["HC_mid"],
                             marker_size=18)
        else:
            for roi in pd.unique(pd.Series(rois)):
                sel = np.asarray(rois) == roi
                disp.add_markers(coords[sel],
                                 marker_color=ROI_C.get(roi, PAL[2]),
                                 marker_size=18)
        ax.set_title(title, fontsize=FS_TITLE)
        return disp
    except Exception as e:                                # nilearn optional
        ax.scatter(coords[:, 0], coords[:, 2], s=14, c=ROI_C["HC_mid"])
        ax.set_xlabel("MNI x (mm)"); ax.set_ylabel("MNI z (mm)")
        ax.set_title(f"{title}\n(glass brain unavailable: {type(e).__name__})",
                     fontsize=FS_TICK)
        return None


def _panel_trace(ax, t_ms, y, color, ylabel, title=None, sem=None, xlim=250,
                 overlay=None, overlay_label=None):
    ax.plot(t_ms, y, color=color, lw=1.1)
    if overlay is not None:
        # The band-passed trace gets its OWN axis, in its own colour, rather
        # than being multiplied by a factor named in the legend. A ripple is
        # 1-2 uV against tens of uV of broadband LFP; a "(x5)" in small type is
        # easy to miss, and a reader who misses it reads the ripple as five
        # times its real amplitude.
        ax2 = ax.twinx()
        ax2.plot(t_ms, overlay, color=RIPPLE_C, lw=0.8)
        lim = 1.15 * np.nanmax(np.abs(overlay))
        ax2.set_ylim(-lim, lim)
        ax2.set_ylabel(overlay_label or r"80–120 Hz ($\mu$V)",
                       color=RIPPLE_C, fontsize=FS_LABEL - 1)
        ax2.tick_params(axis="y", colors=RIPPLE_C, labelsize=FS_TICK - 1)
        ax2.spines["right"].set_color(RIPPLE_C)
        ax2.spines["top"].set_visible(False)
    if sem is not None:
        ax.fill_between(t_ms, y - sem, y + sem, color=color, alpha=0.20, lw=0)
    ax.axvline(0, color="0.65", ls=":", lw=0.8)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    if title:
        ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(labelbottom=False)


def _panel_tfr(ax, tfr, t_ms, freqs, xlim=250, vmax=None, show_marginal=True,
               cbar=False):
    if vmax is None:
        vmax = np.nanpercentile(np.abs(tfr), 99) or 1.0
    im = ax.pcolormesh(t_ms, freqs, tfr, cmap="RdYlBu_r", shading="auto",
                       vmin=-vmax, vmax=vmax, rasterized=True)
    for f in RIPPLE_BAND:
        ax.axhline(f, color="0.25", ls=":", lw=0.6)
    if show_marginal:
        # Chen draw the marginal amplitude spectrum as a curve over the TFR:
        # mean % change across time for each frequency, mapped onto the x axis.
        marg = np.nanmean(tfr, axis=1)
        span = np.nanmax(np.abs(marg))
        if span:
            x = marg / span * (xlim * 0.42)
            ax.plot(x, freqs, color="k", lw=1.0)
            ax.axvline(0, color="k", lw=0.4, alpha=0.3)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(freqs[0], freqs[-1])
    ax.set_xlabel("Time from ripple (ms)", fontsize=FS_LABEL)
    ax.set_ylabel("Frequency (Hz)", fontsize=FS_LABEL)
    if cbar:
        cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
        cb.set_label("Amplitude (% change)", fontsize=FS_LABEL)
        cb.ax.tick_params(labelsize=FS_TICK)
    return im


# ------------------------------------------------------------- figure ----
def chen_figure(raw_by_pair, fs, events, coords=None, rois=None,
                out_stem=None, title="", example=None, win_s=0.25):
    """Chen et al. 2025 Figure 2a-b, reproduced.

    raw_by_pair : {pair_id: 1-D float array}  bipolar signal per derivation
    events      : DataFrame of ACCEPTED events with pair_id, start/stop/peak_sample
    coords      : (n, 3) MNI coordinates of the contacts, for panel a
    example     : (pair_id, peak_sample) to feature, or None to pick the
                  strongest event automatically
    """
    half = int(round(win_s * fs))
    ga_raw, ga_bp, all_peaks_by_pair = [], [], {}

    for pid, raw in raw_by_pair.items():
        e = events[events.pair_id == pid]
        if not len(e):
            continue
        rb = _bp(raw, fs, *RIPPLE_BAND)
        pk = trough_lock(e, rb)
        all_peaks_by_pair[pid] = pk
        ga_raw.append(_snips(raw, pk, half))
        ga_bp.append(_snips(rb, pk, half))

    if not ga_raw:
        print("  chen_figure: no events"); return None
    ga_raw = np.concatenate(ga_raw); ga_bp = np.concatenate(ga_bp)
    t_ms = np.linspace(-win_s, win_s, 2 * half) * 1000

    # grand-average TFR: pool events across derivations
    tfr_stack = []
    for pid, pk in all_peaks_by_pair.items():
        tf, _, fr = ripple_tfr(raw_by_pair[pid], fs, pk, win_s=win_s)
        if np.isfinite(tf).any():
            tfr_stack.append(tf)
    ga_tfr = np.nanmean(np.stack(tfr_stack), axis=0)

    # the example: strongest accepted event, unless one was named
    if example is None:
        best = events.sort_values("rms_peak_z", ascending=False).iloc[0]
        ex_pid = best.pair_id
        ex_pk = trough_lock(best.to_frame().T, _bp(raw_by_pair[ex_pid], fs,
                                                   *RIPPLE_BAND))[0]
    else:
        ex_pid, ex_pk = example
    ex_raw = raw_by_pair[ex_pid][ex_pk - half:ex_pk + half]
    ex_tfr, _, _ = ripple_tfr(raw_by_pair[ex_pid], fs, [ex_pk], win_s=win_s)

    return chen_panels(ga_raw, ga_tfr, t_ms, ex_raw, ex_tfr,
                       coords=coords, rois=rois, out_stem=out_stem, title=title,
                       ga_bp=ga_bp)


def chen_panels(ga_raw, ga_tfr, t_ms, ex_raw, ex_tfr, coords=None, rois=None,
                out_stem=None, title="", ga_bp=None, n_contacts=None,
                ex_bp=None, show_contacts=True, panel_cm=2.5):
    """Draw Chen Fig 2a-b from already-averaged data.

    ga_raw : (n_ripples, n_times) or (n_contacts, n_times) stack to average
    """
    # `show_contacts=False` drops panel (a) and reclaims its width. The
    # standalone `contact_coverage_3d_figure` renders the same coverage on a
    # surface rather than as a projection, so in any figure that carries both
    # this panel is a worse duplicate.
    ga_raw = np.atleast_2d(ga_raw)
    n_ev = len(ga_raw)
    # At 2.5 cm a panel there is room for two words. The n belongs in the
    # caption, where it is not competing with the data for width.
    lab = "Grand average"

    with plt.rc_context(_rc()):
        ncol = 3 if show_contacts else 2
        wr = [1.35, 1, 1] if show_contacts else [1, 1]
        # `panel_cm` is the size of ONE data subpanel; the figure is that plus
        # the margins the labels need, so the printed panel is the size asked
        # for rather than the size left over after tight_layout.
        n_data_col = ncol - (1 if show_contacts else 0)
        w_in = (n_data_col * panel_cm * 1.62 +
                (panel_cm * 1.35 if show_contacts else 0)) * CM
        h_in = (panel_cm * 1.72 * 2) * CM
        fig = plt.figure(figsize=(w_in, h_in))
        gs = gridspec.GridSpec(
            2, ncol, figure=fig, width_ratios=wr,
            height_ratios=[0.72, 1], hspace=0.14, wspace=0.46,
            left=0.055 if show_contacts else 0.155, right=0.97,
            top=0.86, bottom=0.13)
        c0 = 1 if show_contacts else 0

        # (a) contacts -----------------------------------------------------
        if show_contacts:
            ax_a = fig.add_subplot(gs[:, 0])
            if coords is not None and len(coords):
                _panel_glassbrain(ax_a, coords, rois)
            else:
                ax_a.axis("off")
            ax_a.text(-0.02, 1.10, "a", transform=ax_a.transAxes,
                      fontsize=FS_TITLE + 1, fontweight="bold", va="bottom")

        # (b) grand average -------------------------------------------------
        ax_b1 = fig.add_subplot(gs[0, c0])
        _panel_trace(ax_b1, t_ms, ga_raw.mean(0), RAW_C,
                     r"Voltage ($\mu$V)", lab,
                     sem=ga_raw.std(0) / np.sqrt(len(ga_raw)))
        ax_b1.text(-0.34, 1.16, "b" if show_contacts else "a",
                   transform=ax_b1.transAxes,
                   fontsize=FS_TITLE + 1, fontweight="bold", va="bottom")
        ax_b2 = fig.add_subplot(gs[1, c0])
        _panel_tfr(ax_b2, ga_tfr, t_ms, TFR_FREQS)

        # (c) example -------------------------------------------------------
        ax_c1 = fig.add_subplot(gs[0, c0 + 1])
        _panel_trace(ax_c1, t_ms, ex_raw, RAW_C, r"Voltage ($\mu$V)",
                     "Example ripple", overlay=ex_bp)
        ax_c2 = fig.add_subplot(gs[1, c0 + 1])
        _panel_tfr(ax_c2, ex_tfr, t_ms, TFR_FREQS, cbar=True)

        if title:
            fig.suptitle(title, fontsize=FS_TITLE, y=0.985)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf "
                  f"({n_ev} rows)")
    out = dict(n=n_ev, ga_raw_ptp=float(np.ptp(ga_raw.mean(0))))
    if ga_bp is not None and len(ga_bp):
        out["ga_bp_ptp"] = float(np.ptp(np.atleast_2d(ga_bp).mean(0)))
        out["single_bp_uv"] = float(np.abs(np.atleast_2d(ga_bp)).max(1).mean())
    return out


def sharp_wave_figure(raw_by_pair, fs, events, out_stem=None, title="",
                      win_s=0.25):
    """Per-contact ripple-locked low-frequency deflection.

    One line per derivation, with its own sign kept. If the sharp wave were
    being recovered, the lines would deflect together at t = 0; if the montage
    has cancelled it, they will not. Either way the figure states which.
    """
    rows, profiles = [], {}
    for pid, raw in raw_by_pair.items():
        e = events[events.pair_id == pid]
        if len(e) < 30:
            continue
        pk = trough_lock(e, _bp(raw, fs, *RIPPLE_BAND))
        pr = sharp_wave_profiles(raw, fs, pk, win_s=win_s)
        if pr is None:
            continue
        profiles[pid] = pr
        rows.append(dict(pair_id=pid, n=pr["n"],
                         deflection_uv=pr["deflection_uv"],
                         flank_sd=pr["flank_sd"], snr=pr["snr"]))
    if not profiles:
        return pd.DataFrame()
    tab = pd.DataFrame(rows)

    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 6.5 * CM),
                                 gridspec_kw=dict(width_ratios=[1.6, 1],
                                                  wspace=0.45))
        fig.subplots_adjust(top=0.74, bottom=0.20, left=0.10, right=0.97)
        ax = axes[0]
        for pid, pr in profiles.items():
            ax.plot(pr["t_ms"], pr["mean"], lw=1.0, alpha=0.85, label=pid)
        ax.axvline(0, color="0.65", ls=":", lw=0.8)
        ax.axhline(0, color="0.85", lw=0.6)
        ax.set_xlim(-win_s * 1000, win_s * 1000)
        ax.set_xlabel("Time from ripple (ms)")
        ax.set_ylabel(r"Low-frequency (< %g Hz) ($\mu$V)" % SW_BAND_HZ)
        ax.set_title("Slow deflection at ripple", fontsize=FS_TITLE - 1)
        if len(profiles) <= 6:
            ax.legend(frameon=False, fontsize=FS_TICK - 2, loc="best")

        ax = axes[1]
        col = [RIPPLE_C if v < 0 else SW_C for v in tab.deflection_uv]
        ax.barh(np.arange(len(tab)), tab.deflection_uv, color=col, height=0.7)
        ax.axvline(0, color="0.4", lw=0.8)
        ax.set_yticks(np.arange(len(tab)))
        ax.set_yticklabels(tab.pair_id, fontsize=FS_TICK - 2)
        ax.set_xlabel(r"Deflection at ripple ($\mu$V)")
        n_neg = int((tab.deflection_uv < 0).sum())
        ax.set_title(f"Polarity: {n_neg} negative / {len(tab) - n_neg} positive",
                     fontsize=FS_TITLE - 1)

        if title:
            fig.suptitle(title, fontsize=FS_TITLE, y=0.99)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return tab


def monopolar_sharpwave_figure(profiles, out_stem=None, title="", ncol=3):
    """Monopolar contacts vs their bipolar difference, ripple-locked.

    `profiles` maps "<pair> | <trace>" to a sharp_wave_profiles dict. One panel
    per pair, with contact A, contact B and A-B overlaid, so the question "does
    the subtraction remove a deflection that was there" is answered by looking
    at one panel rather than by comparing figures.
    """
    pairs = sorted({k.split(" | ")[0] for k in profiles})
    if not pairs:
        return None
    ncol = min(ncol, len(pairs))
    nrow = int(np.ceil(len(pairs) / ncol))
    style = {"contact A": dict(color="#7eb1c4", lw=1.1),
             "contact B": dict(color="#175e62", lw=1.1),
             "bipolar A-B": dict(color=RIPPLE_C, lw=1.4)}

    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                                 figsize=(6.2 * ncol * CM, 5.6 * nrow * CM))
        for ax in axes.ravel():
            ax.axis("off")
        for j, pid in enumerate(pairs):
            ax = axes[j // ncol][j % ncol]; ax.axis("on")
            for tag, st in style.items():
                pr = profiles.get(f"{pid} | {tag}")
                if pr is None:
                    continue
                ax.plot(pr["t_ms"], pr["mean"], label=tag, **st)
            ax.axvline(0, color="0.65", ls=":", lw=0.8)
            ax.axhline(0, color="0.88", lw=0.6)
            ax.set_title(pid, fontsize=FS_TICK)
            if j // ncol == nrow - 1:
                ax.set_xlabel("Time from ripple (ms)")
            if j % ncol == 0:
                ax.set_ylabel(r"< %g Hz ($\mu$V)" % SW_BAND_HZ)
        axes[0][0].legend(frameon=False, fontsize=FS_TICK - 2, loc="best")
        if title:
            fig.suptitle(title, fontsize=FS_TITLE)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return fig


CRIT_C = {"raw_amplitude": "#F15A29", "first_derivative": "#F7931E",
          "hf_rms_250": "#6B60AA", "broadband_1_60": "#23677E",
          "ied_janca": "#a30d6c"}
CRIT_SHORT = {"raw_amplitude": "amplitude", "first_derivative": "gradient",
              "hf_rms_250": ">250 Hz", "broadband_1_60": "1–60 Hz",
              "ied_janca": "IED"}


def artifact_figure(raw_by_pair, fs, masks, stats, out_stem=None, title="",
                    n_examples=4, win_s=1.0):
    """What the artifact rejection removes, and whether it should have.

    The criteria together discard 35-54% of a typical recording. That is a large
    fraction to delete on the strength of a percentage in a table, so this shows
    the removed data itself: which criterion fired, where in the recording, and
    what the excluded stretches actually look like next to kept ones at the same
    scale.

    masks : {pair_id: {"bad": bool array, "per": {criterion: bool array}}}
    stats : {pair_id: {criterion: fraction flagged}}
    """
    pairs = [p for p in raw_by_pair if p in masks]
    if not pairs:
        return None
    crits = [c for c in CRIT_C if any(c in stats[p] for p in pairs)]
    half = int(round(win_s * fs))

    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(18 * CM, 15 * CM))
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 0.7, 1.5],
                               hspace=0.95, wspace=0.28,
                               left=0.09, right=0.97, top=0.91, bottom=0.07)

        # (a) how much each criterion flags ---------------------------------
        ax = fig.add_subplot(gs[0, 0])
        w = 0.8 / max(1, len(pairs))
        for i, p in enumerate(pairs):
            v = [100 * stats[p].get(c, 0.0) for c in crits]
            ax.bar(np.arange(len(crits)) + i * w - 0.4 + w / 2, v, width=w,
                   color=[CRIT_C[c] for c in crits], edgecolor="w", linewidth=0.4)
        ax.set_xticks(np.arange(len(crits)))
        ax.set_xticklabels([CRIT_SHORT.get(c, c) for c in crits],
                           fontsize=FS_TICK - 1)
        ax.set_ylabel("% of recording flagged")
        ax.set_title("Each criterion alone", fontsize=FS_TITLE - 1)
        ax.legend(handles=[plt.Line2D([], [], marker="s", ls="", ms=5,
                                      color=CRIT_C[c], label=CRIT_SHORT.get(c, c))
                           for c in crits],
                  frameon=False, fontsize=FS_TICK - 2, ncol=2,
                  loc="upper right", handletextpad=0.3, columnspacing=0.8)

        # (b) and how much survives after union + padding ---------------------
        ax = fig.add_subplot(gs[0, 1])
        comb = [100 * stats[p].get("combined_after_pad", np.nan) for p in pairs]
        ax.bar(np.arange(len(pairs)), comb, color=RIPPLE_C, edgecolor="w")
        ax.axhline(100 * 2 / 3, color="0.3", ls="--", lw=0.9)
        ax.text(len(pairs) - 0.5, 100 * 2 / 3 + 2, "contact excluded above 2/3",
                ha="right", fontsize=FS_TICK - 1, color="0.3")
        ax.set_xticks(np.arange(len(pairs)))
        ax.set_xticklabels([p.split("-")[0] for p in pairs], rotation=20,
                           ha="right", fontsize=FS_TICK - 2)
        ax.set_ylabel("% removed"); ax.set_ylim(0, 100)
        ax.set_title("Union + 1 s padding", fontsize=FS_TITLE - 1)

        # (c) where in the recording ------------------------------------------
        ax = fig.add_subplot(gs[1, :])
        for i, p in enumerate(pairs):
            bad = masks[p]["bad"]
            t = np.arange(len(bad)) / fs / 60.0
            ax.fill_between(t, i, i + 0.8, where=bad, color=RIPPLE_C, lw=0)
            ax.text(-0.004 * t[-1], i + 0.4, p, ha="right", va="center",
                    fontsize=FS_TICK - 2)
        ax.set_ylim(-0.2, len(pairs)); ax.set_yticks([])
        ax.set_xlabel("Time in recording (min)")
        ax.set_title("When it was removed (red = excluded)", fontsize=FS_TITLE - 1)

        # (d) what the removed data looks like, vs kept ------------------------
        p0 = pairs[int(np.argmax([stats[p].get("combined_after_pad", 0)
                                  for p in pairs]))]
        x = raw_by_pair[p0]; bad = masks[p0]["bad"]; per = masks[p0]["per"]
        # rank removed stretches by how extreme they are
        lab_bad = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]) == 1)
        ends = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]) == -1)
        segs = [(a, b) for a, b in zip(lab_bad, ends)
                if a - half >= 0 and b + half < len(x)]
        segs.sort(key=lambda ab: -np.abs(x[ab[0]:ab[1]]).max() if ab[1] > ab[0] else 0)
        good = np.flatnonzero(~bad)
        rng = np.random.RandomState(42)

        inner = gridspec.GridSpecFromSubplotSpec(
            2, n_examples, subplot_spec=gs[2, :], hspace=0.55, wspace=0.25)
        ylims = []
        for j in range(n_examples):
            # removed
            ax = fig.add_subplot(inner[0, j])
            if j < len(segs):
                a, b = segs[j]
                c = (a + b) // 2
                seg = x[c - half:c + half]
                tt = (np.arange(len(seg)) - half) / fs
                ax.plot(tt, seg, color="0.35", lw=0.7)
                ax.axvspan((a - c) / fs, (b - c) / fs, color=RIPPLE_C, alpha=0.22, lw=0)
                # Which criteria fired, as colour marks: spelling them out
                # collides between panels once several fire at once, which on
                # an IED they usually do.
                fired = [k for k in CRIT_C if k in per and per[k][a:b].any()]
                for m, k in enumerate(fired):
                    ax.plot(0.06 + 0.10 * m, 1.06, "s", color=CRIT_C[k],
                            ms=4, transform=ax.transAxes, clip_on=False)
                if not fired:
                    ax.set_title("padding only", fontsize=FS_TICK - 2,
                                 color="0.5")
            ax.set_xticks([])
            ylims.append(ax.get_ylim())
            if j == 0:
                ax.set_ylabel("REMOVED\n" + r"($\mu$V)", fontsize=FS_TICK)
            # kept, matched window
            ax = fig.add_subplot(inner[1, j])
            if len(good) > 2 * half:
                c = int(good[rng.randint(half, len(good) - half)])
                if c - half >= 0 and c + half < len(x):
                    seg = x[c - half:c + half]
                    ax.plot((np.arange(len(seg)) - half) / fs, seg,
                            color=SW_C, lw=0.7)
            # same scale as the panel above: the point is the size difference
            if j < len(ylims):
                ax.set_ylim(ylims[j])
            ax.set_xlabel("s", fontsize=FS_TICK - 1)
            if j == 0:
                ax.set_ylabel("KEPT\n" + r"($\mu$V)", fontsize=FS_TICK)

        fig.suptitle(title or f"Artifact rejection — {p0}", fontsize=FS_TITLE)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return fig


def sharp_wave_examples(raw_by_pair, fs, events, sw_table, out_stem=None,
                        title="", n=6, win_s=0.35):
    """Single ripples on the contacts where a sharp wave actually survives.

    Picked from the derivation with the largest surviving low-frequency
    deflection (`sw_table`), then ranked within it by how large that event's own
    slow deflection is. This is deliberately a best-case display: it shows that
    the morphology exists in these data, not how typical it is. The typical case
    is the per-derivation table, where 4 of 9 show nothing.
    """
    if sw_table is None or not len(sw_table):
        return None
    best = sw_table.reindex(sw_table.deflection_uv.abs().sort_values(
        ascending=False).index)
    pid = None
    for _, r in best.iterrows():
        if r.pair_id in raw_by_pair:
            pid = r.pair_id; break
    if pid is None:
        return None

    raw = raw_by_pair[pid]
    rb = _bp(raw, fs, *RIPPLE_BAND)
    sw = _lp(raw, fs, SW_BAND_HZ)
    e = events[events.pair_id == pid]
    if not len(e):
        return None
    pk = trough_lock(e, rb)
    half = int(round(win_s * fs))
    ok = [(p, abs(sw[p])) for p in pk if p - half >= 0 and p + half < len(raw)]
    if not ok:
        return None
    ok.sort(key=lambda t: -t[1])
    ok = ok[:n]

    ncol = min(3, len(ok)); nrow = int(np.ceil(len(ok) / ncol))
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                                 figsize=(6.2 * ncol * CM, 5.2 * nrow * CM))
        for ax in axes.ravel():
            ax.axis("off")
        for j, (p, _) in enumerate(ok):
            ax = axes[j // ncol][j % ncol]; ax.axis("on")
            t = (np.arange(-half, half) / fs) * 1000
            ax.plot(t, raw[p - half:p + half], color="0.55", lw=0.6,
                    label="broadband")
            ax.plot(t, sw[p - half:p + half], color=SW_C, lw=1.6,
                    label=f"< {SW_BAND_HZ:g} Hz")
            # Ripple band on its own axis -- see `_panel_trace`. Here it matters
            # twice over: the panel exists to compare the timing of a ~1 uV
            # ripple against a ~30 uV sharp wave, and a hidden scale factor
            # makes the two look comparable in size when they are not.
            ax2 = ax.twinx()
            seg_rb = rb[p - half:p + half]
            ax2.plot(t, seg_rb, color=RIPPLE_C, lw=0.9)
            lim = 1.15 * np.abs(seg_rb).max()
            ax2.set_ylim(-lim, lim)
            ax2.tick_params(axis="y", colors=RIPPLE_C, labelsize=FS_TICK - 2)
            ax2.spines["right"].set_color(RIPPLE_C)
            ax2.spines["top"].set_visible(False)
            if j % ncol == ncol - 1:
                ax2.set_ylabel(r"80–120 Hz ($\mu$V)", color=RIPPLE_C,
                               fontsize=FS_TICK - 1)
            ax.axvline(0, color="0.65", ls=":", lw=0.8)
            ax.set_title(f"t = {p / fs:.1f} s", fontsize=FS_TICK)
            if j // ncol == nrow - 1:
                ax.set_xlabel("Time from ripple (ms)")
            if j % ncol == 0:
                ax.set_ylabel(r"$\mu$V")
        axes[0][0].legend(frameon=False, fontsize=FS_TICK - 2, loc="upper left")
        fig.suptitle(title or f"Ripples with a sharp wave — {pid}",
                     fontsize=FS_TITLE)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf ({pid})")
    return pid


STATE_COLORS = {"A": "#F15A29", "B": "#F7931E", "C": "#C7C6E2", "D": "#6B60AA"}
PHASE_COLORS = {"discovery": "#D7657F", "early": "#FCDDE3",
                "middle": "#D7657F", "later": "#5C1027", "late": "#5C1027"}
OBS_LINE_C = "#0e3d3a"


def _rate_by(counts, col):
    """Pooled rate per level: summed events over summed exposure.

    Not the mean of per-window rates -- that weights a 0.5 s window as heavily
    as a 5 s one and is dominated by the short ones.
    """
    g = counts.groupby(col).agg(n=("n_ripples", "sum"), e=("exposure_s", "sum"))
    g["rate_hz"] = g["n"] / g["e"].replace(0, np.nan)
    # subject-level spread, for an error bar that reflects the unit of inference
    per_s = (counts.groupby([col, "subject_key"])
             .agg(n=("n_ripples", "sum"), e=("exposure_s", "sum")))
    per_s["r"] = per_s["n"] / per_s["e"].replace(0, np.nan)
    # `sem` shadows DataFrame.sem, so every read of it must be by key
    g["sem"] = per_s.groupby(level=0)["r"].agg(
        lambda v: v.std() / max(np.sqrt(v.notna().sum()), 1))
    g["n_subjects"] = per_s.groupby(level=0)["r"].count()
    return g


# which column splits the conditions, per hypothesis. Shared with
# scripts/swr_hypotheses.py so the summary panel and the per-hypothesis
# distribution figures group the data the same way.
HYP_COND_COL = {"H1": "condition", "H2": "phase_after",
                "H4": ["feedback", "phase"], "H5": "phase3",
                "H6": ["state", "discovery"], "H7": "informative"}


def hypothesis_figure(results, out_stem=None, title="", ncol=4):
    """One panel per hypothesis: the condition rates, plus the permutation null.

    Bars are the pooled rate (events / artifact-free seconds), error bars the
    SEM across subjects, because the subject is the unit of inference. The
    histogram below is the circular-shift null with the observed value marked,
    since that -- not the GLM p -- is the primary inference.

    Wraps onto `ncol` columns so seven hypotheses still fit an A4 width.
    """
    keys = [k for k in ("H1", "H2", "H3", "H4", "H5", "H6", "H7")
            if k in results]
    if not keys:
        return None
    ncol = int(min(ncol, len(keys)))
    nblock = int(np.ceil(len(keys) / ncol))
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(2 * nblock, ncol, squeeze=False,
                                 figsize=(4.9 * ncol * CM, 11 * nblock * CM),
                                 gridspec_kw=dict(hspace=0.95, wspace=0.55))
        for ax in axes.ravel():
            ax.axis("off")
        for i, k in enumerate(keys):
            blk, j = divmod(i, ncol)
            r = results[k]
            c = r.get("counts")
            ax = axes[2 * blk][j]
            ax.axis("on")
            if k == "H3" and c is not None and len(c):
                ax.scatter(c.rate_hz, c.errors_per_repeat_after, s=9,
                           color=PHASE_COLORS["discovery"], alpha=0.6,
                           edgecolor="none")
                if c.rate_hz.notna().sum() > 2:
                    b = np.polyfit(c.rate_hz, c.errors_per_repeat_after, 1)
                    xs = np.linspace(c.rate_hz.min(), c.rate_hz.max(), 20)
                    ax.plot(xs, np.polyval(b, xs), color=OBS_LINE_C, lw=1.2)
                ax.set_xlabel("Ripple rate after\nfirst-D (Hz)")
                ax.set_ylabel("Errors per later repeat")
            elif c is not None and len(c):
                lab = _cond_series(c, HYP_COND_COL.get(k, "condition"))
                if lab is None:
                    lab = _cond_series(c, "condition")
                if lab is not None:
                    d = c.copy()
                    d["_cond"] = lab
                    g = _rate_by(d, "_cond")
                    cols = [_cond_color(ix, m) for m, ix in enumerate(g.index)]
                    ax.bar(np.arange(len(g)), g["rate_hz"], yerr=g["sem"],
                           color=cols, edgecolor="w", capsize=2.5,
                           error_kw=dict(lw=0.8))
                    ax.set_xticks(np.arange(len(g)))
                    ax.set_xticklabels([str(ix).replace("_", "\n")
                                        for ix in g.index],
                                       fontsize=FS_TICK - 2.5)
                    ax.set_ylabel("Ripple rate (Hz)")
            ax.set_title(k, fontsize=FS_TITLE)

            ax = axes[2 * blk + 1][j]
            ax.axis("on")
            null = r.get("null")
            if null is not None and len(np.ravel(null)):
                nn = np.ravel(np.asarray(null, float))
                nn = nn[np.isfinite(nn)]
                if nn.size:
                    ax.hist(nn, bins=30, color="0.8", edgecolor="w",
                            linewidth=0.3)
            obs = r.get("observed_log_rr", np.nan)
            if np.isfinite(obs):
                ax.axvline(obs, color=OBS_LINE_C, lw=1.6)
            ax.set_xlabel("log rate ratio" if k != "H3" else "coefficient")
            ax.set_ylabel("permutations")
            p_ = r.get("p_perm", np.nan)
            ax.set_title(f"p = {p_:.3f}" if np.isfinite(p_) else "p = n/a",
                         fontsize=FS_TICK)
        if title:
            fig.suptitle(title, fontsize=FS_TITLE, y=1.002)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return fig


# =============================================================================
# DESCRIPTIVE DISTRIBUTIONS -- how each test arises from the data
# =============================================================================
# A test statistic is one number; these panels are the data behind it. The point
# is to be able to see, for every contrast, (a) the spread across the unit of
# inference, (b) whether the conditions differ in the things that trivially move
# ripple rate -- window length, movement, artifact-free exposure -- and (c) where
# the observed value sits in its own permutation null.

# Feedback valence gets its own hue so that a crossed label like
# "error, discovery" is not drawn in the same colour as "correct, discovery".
# The substring fallback below matched on phase alone, which made every
# 2x2 feedback figure unreadable.
FEEDBACK_COLORS = {"correct": "#0E3D3A", "error": "#B03A5B"}


def _cond_color(level, i):
    lv = str(level)
    if lv in STATE_COLORS:
        return STATE_COLORS[lv]
    if lv in PHASE_COLORS:
        return PHASE_COLORS[lv]
    low = lv.lower()
    # crossed labels: valence sets the hue, phase sets the lightness
    val = next((v for v in FEEDBACK_COLORS if v in low), None)
    if val is not None:
        base = np.array(colors.to_rgb(FEEDBACK_COLORS[val]))
        if "first" in low:
            f = 0.0
        elif "learn" in low or "discovery" in low:
            f = 0.32
        else:
            f = 0.62
        return tuple(base + (np.array([1.0, 1.0, 1.0]) - base) * f)
    for key, c in PHASE_COLORS.items():
        if key in low:
            return c
    return PAL[i % len(PAL)]


def _cond_series(counts, condition_col):
    """The condition label per row, as one string column (handles 2-factor)."""
    if isinstance(condition_col, (list, tuple)):
        cols = [c for c in condition_col if c in counts.columns]
        if not cols:
            return None
        lab = counts[cols[0]].astype(str)
        for c in cols[1:]:
            lab = lab + "\n" + counts[c].astype(str)
        return lab
    if condition_col not in counts.columns:
        return None
    return counts[condition_col].astype(str)


def _strip_box(ax, groups, values, colors, ylabel, rng, paired=None):
    """Per-unit points with a box, and paired lines when the unit is shared.

    Every point that goes into the statistic is drawn -- nothing is trimmed or
    winsorised, so an outlier stays visible as an outlier.
    """
    pos = np.arange(len(groups))
    for i, g in enumerate(groups):
        v = np.asarray(values[g], float)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        ax.boxplot(v, positions=[i], widths=0.55, showfliers=False,
                   medianprops=dict(color="0.15", lw=1.2),
                   boxprops=dict(color="0.45", lw=0.8),
                   whiskerprops=dict(color="0.45", lw=0.8),
                   capprops=dict(color="0.45", lw=0.8))
        jitter = (rng.random(v.size) - 0.5) * 0.28
        ax.scatter(i + jitter, v, s=7, color=colors[i], alpha=0.65,
                   edgecolor="none", zorder=3)
    if paired is not None and len(groups) > 1:
        for _, row in paired.iterrows():
            y = [row.get(g, np.nan) for g in groups]
            if np.sum(np.isfinite(np.asarray(y, float))) < 2:
                continue
            ax.plot(pos, y, color="0.6", lw=0.4, alpha=0.5, zorder=1)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(g).replace("_", "\n") for g in groups],
                       fontsize=FS_TICK - 1)
    ax.set_ylabel(ylabel)


def condition_distributions(name, counts, condition_col, null=None,
                            observed=None, question="", out_stem=None,
                            seed=42):
    """Six panels showing how one hypothesis test arises from its data.

    counts : the window x derivation table the test was computed on
    null   : the circular-shift null, observed : the empirical statistic

    Panels: per-subject rate, per-derivation rate, window duration, movement,
    exposure and window count, and the permutation null.
    """
    rng = np.random.default_rng(seed)
    lab = _cond_series(counts, condition_col)
    if lab is None or not len(counts):
        return None
    c = counts.copy()
    c["_cond"] = lab
    groups = list(pd.unique(c["_cond"].sort_values()))
    colors = [_cond_color(g, i) for i, g in enumerate(groups)]

    # rate per unit: summed events over summed exposure, the same pooling the
    # statistic uses -- a mean of per-window rates would weight a 0.5 s window
    # like a 5 s one
    def _rate_per(unit):
        g = (c.groupby([unit, "_cond"])
             .agg(n=("n_ripples", "sum"), e=("exposure_s", "sum")))
        g["r"] = g["n"] / g["e"].replace(0, np.nan)
        return g["r"].unstack("_cond")

    per_subj = _rate_per("subject_key") if "subject_key" in c else None
    per_pair = _rate_per("pair_id") if "pair_id" in c else None

    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(2, 3, figsize=(19 * CM, 13 * CM))
        fig.subplots_adjust(hspace=0.55, wspace=0.38)

        ax = axes[0][0]
        if per_subj is not None:
            _strip_box(ax, groups, {g: per_subj[g].to_numpy() for g in groups
                                    if g in per_subj},
                       colors, "Ripple rate (Hz)", rng, paired=per_subj)
            ax.set_title(f"Per subject (n={per_subj.shape[0]})\nthe unit of inference",
                         fontsize=FS_TICK)

        ax = axes[0][1]
        if per_pair is not None:
            _strip_box(ax, groups, {g: per_pair[g].to_numpy() for g in groups
                                    if g in per_pair},
                       colors, "Ripple rate (Hz)", rng, paired=per_pair)
            ax.set_title(f"Per derivation (n={per_pair.shape[0]})",
                         fontsize=FS_TICK)

        ax = axes[0][2]
        if "duration_s" in c:
            d = c["duration_s"].to_numpy(float)
            d = d[np.isfinite(d)]
            # several designs lock a fixed-length window, so the "distribution"
            # is a constant and plotting it shows only float noise. Say so.
            if d.size and (d.max() - d.min()) < 1e-6:
                ax.text(0.5, 0.5, f"fixed by design\n{d[0]:.2f} s\nin every "
                                  "condition", ha="center", va="center",
                        transform=ax.transAxes, fontsize=FS_TICK)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            else:
                _strip_box(ax, groups,
                           {g: c.loc[c._cond == g, "duration_s"].to_numpy()
                            for g in groups}, colors, "Window duration (s)", rng)
            ax.set_title("Window length\n(confound: longer = more ripples)",
                         fontsize=FS_TICK)

        ax = axes[1][0]
        if "n_moves" in c:
            _strip_box(ax, groups,
                       {g: c.loc[c._cond == g, "n_moves"].to_numpy()
                        for g in groups}, colors, "Movement presses / window", rng)
            ax.set_title("Movement\n(confound: suppresses ripples)",
                         fontsize=FS_TICK)

        ax = axes[1][1]
        expo = c.groupby("_cond").agg(e=("exposure_s", "sum"),
                                      n=("n_ripples", "sum"),
                                      w=("exposure_s", "size")).reindex(groups)
        ax.bar(np.arange(len(groups)), expo["e"], color=colors, edgecolor="w")
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels([str(g).replace("_", "\n") for g in groups],
                           fontsize=FS_TICK - 1)
        ax.set_ylabel("Artifact-free exposure (s)")
        for i, (_, r) in enumerate(expo.iterrows()):
            ax.text(i, r["e"], f"{int(r['w'])} win\n{int(r['n'])} rip",
                    ha="center", va="bottom", fontsize=FS_TICK - 2.5)
        ax.margins(y=0.22)
        ax.set_title("Power per condition\n(unequal by design)", fontsize=FS_TICK)

        ax = axes[1][2]
        if null is not None and np.isfinite(np.asarray(null, float)).any():
            nn = np.asarray(null, float)
            nn = nn[np.isfinite(nn)]
            ax.hist(nn, bins=30, color="0.75", edgecolor="w", lw=0.4)
            if observed is not None and np.isfinite(observed):
                ax.axvline(observed, color=OBS_LINE_C, lw=1.6)
                p = (1 + np.sum(nn >= observed)) / (1 + nn.size)
                ax.text(0.97, 0.94, f"observed\np = {p:.4f}", ha="right",
                        va="top", transform=ax.transAxes, fontsize=FS_TICK - 1,
                        color=OBS_LINE_C)
            ax.set_xlabel("Statistic under circular shift")
            ax.set_ylabel("Permutations")
            ax.set_title(f"Null ({nn.size} shifts)", fontsize=FS_TICK)
        else:
            ax.axis("off")

        head = f"{name}: {question}" if question else name
        fig.suptitle(head, fontsize=FS_TITLE, y=1.005)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return fig


def regression_distributions(name, tab, x, y, null=None, observed=None,
                             question="", out_stem=None, seed=42):
    """The same idea for H3, which is a regression rather than a contrast."""
    rng = np.random.default_rng(seed)
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, 4, figsize=(22 * CM, 6.0 * CM))
        fig.subplots_adjust(wspace=0.42)

        ax = axes[0]
        ax.scatter(tab[x], tab[y], s=10, color=PHASE_COLORS["discovery"],
                   alpha=0.6, edgecolor="none")
        ok = tab[[x, y]].notna().all(axis=1)
        if ok.sum() > 2:
            b = np.polyfit(tab.loc[ok, x], tab.loc[ok, y], 1)
            xs = np.linspace(tab[x].min(), tab[x].max(), 20)
            ax.plot(xs, np.polyval(b, xs), color=OBS_LINE_C, lw=1.3)
        ax.set_xlabel("Ripple rate after first-D (Hz)")
        ax.set_ylabel("Errors after first-D")
        ax.set_title(f"{len(tab)} grid-derivations", fontsize=FS_TICK)

        for ax, col, lb in ((axes[1], x, "Ripple rate (Hz)"),
                            (axes[2], y, "Errors after first-D")):
            v = tab[col].to_numpy(float)
            v = v[np.isfinite(v)]
            ax.hist(v, bins=30, color="0.75", edgecolor="w", lw=0.4)
            ax.set_xlabel(lb); ax.set_ylabel("Grid-derivations")
            ax.set_title("Distribution", fontsize=FS_TICK)

        ax = axes[3]
        if null is not None and np.isfinite(np.asarray(null, float)).any():
            nn = np.asarray(null, float); nn = nn[np.isfinite(nn)]
            ax.hist(nn, bins=30, color="0.75", edgecolor="w", lw=0.4)
            if observed is not None and np.isfinite(observed):
                ax.axvline(observed, color=OBS_LINE_C, lw=1.6)
            ax.set_xlabel("beta under within-subject shuffle")
            ax.set_ylabel("Permutations")
            ax.set_title(f"Null ({nn.size} shuffles)", fontsize=FS_TICK)
        else:
            ax.axis("off")

        fig.suptitle(f"{name}: {question}" if question else name,
                     fontsize=FS_TITLE, y=1.02)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return fig


def effect_forest(name, counts, condition_col, hi, lo, out_stem=None,
                  question="", unit="session"):
    """Per-unit log rate ratio for a two-level contrast, ordered, with the pool.

    A single pooled number hides whether an effect is carried by every subject
    or by one. Each row is one session (or subject) with its own log rate ratio
    and a Poisson standard error; the vertical line is the pooled estimate over
    all events and exposure -- the quantity the test is actually computed on.
    """
    if condition_col not in counts.columns:
        return None
    c = counts[counts[condition_col].isin([hi, lo])]
    if not len(c):
        return None
    g = (c.groupby([unit, condition_col])
         .agg(n=("n_ripples", "sum"), e=("exposure_s", "sum")))
    n = g["n"].unstack(condition_col)
    e = g["e"].unstack(condition_col)
    if hi not in n or lo not in n:
        return None
    keep = (n[hi] > 0) & (n[lo] > 0) & (e[hi] > 0) & (e[lo] > 0)
    n, e = n[keep], e[keep]
    if not len(n):
        return None
    lrr = np.log((n[hi] / e[hi]) / (n[lo] / e[lo]))
    se = np.sqrt(1.0 / n[hi] + 1.0 / n[lo])          # Poisson delta method
    order = np.argsort(lrr.to_numpy())
    lrr, se = lrr.iloc[order], se.iloc[order]
    pooled = float(np.log((n[hi].sum() / e[hi].sum()) /
                          (n[lo].sum() / e[lo].sum())))

    with plt.rc_context(_rc()):
        h = max(6.0, 0.32 * len(lrr) + 3.0)
        fig, ax = plt.subplots(figsize=(9 * CM, h * CM))
        y = np.arange(len(lrr))
        ax.errorbar(lrr.to_numpy(), y, xerr=se.to_numpy(), fmt="o", ms=3.2,
                    lw=0.8, color="0.35", ecolor="0.7", zorder=2)
        ax.axvline(0, color="0.6", lw=0.8, ls=":")
        ax.axvline(pooled, color=OBS_LINE_C, lw=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels([f"s{int(i):02d}" if unit == "session" else str(i)
                            for i in lrr.index], fontsize=FS_TICK - 2.5)
        ax.set_xlabel(f"log rate ratio, {hi} vs {lo}")
        ax.set_ylabel(unit.capitalize())
        # annotate the pooled line itself rather than a legend box, which lands
        # on top of a data row in a plot this dense
        n_pos = int((lrr > 0).sum())
        ax.set_title(f"{name}: per-{unit} effect\n{question}\n"
                     f"pooled {pooled:+.3f}   ({n_pos}/{len(lrr)} {unit}s "
                     f"positive)", fontsize=FS_TICK)
        ax.margins(y=0.01)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return fig


def peth_figure(panels, out_stem=None, title="", ncol=2, ylabel="Ripple rate (Hz)",
                xlabel="Time from event (s)"):
    """Peri-event ripple rate over time, one axis per panel.

    `panels` is a list of (panel_title, {trace_label: (centres, mean, sem, n)}).
    Time course rather than a bar: a contrast between two windows collapses
    everything about *when* within the window the difference sits, and a flat
    PETH with a difference in mean means something quite different from a
    transient peak at the event.
    """
    if not panels:
        return None
    nrow = int(np.ceil(len(panels) / ncol))
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                                 figsize=(9.5 * ncol * CM, 6.6 * nrow * CM))
        fig.subplots_adjust(hspace=0.75, wspace=0.32)
        for ax in axes.ravel():
            ax.axis("off")
        for i, (ptitle, traces) in enumerate(panels):
            ax = axes[i // ncol][i % ncol]
            ax.axis("on")
            for j, (lab, (x, m, se, n)) in enumerate(traces.items()):
                col = _cond_color(lab, j)
                x, m = np.asarray(x, float), np.asarray(m, float)
                se = np.asarray(se, float)
                ax.plot(x, m, color=col, lw=1.3, label=f"{lab} (n={n})")
                ok = np.isfinite(m) & np.isfinite(se)
                ax.fill_between(x[ok], (m - se)[ok], (m + se)[ok], color=col,
                                alpha=0.22, lw=0)
            ax.axvline(0, color="0.5", lw=0.8, ls=":")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(ptitle, fontsize=FS_TICK)
            ax.legend(fontsize=FS_TICK - 2.5, frameon=False)
        if title:
            fig.suptitle(title, fontsize=FS_TITLE, y=1.005)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return fig


def ripple_property_figure(ripples, split=None, out_stem=None, title=""):
    """Distributions of the four ripple attributes, optionally split.

    Chen report rate, duration, amplitude and peak frequency separately so that
    an effect can be shown to be on *rate*. If a condition also shifts duration
    or frequency, the events themselves differ and "more ripples" is the wrong
    description.
    """
    cols = [(c, lab, u) for c, lab, u in (
        ("duration_s", "Duration", "s"),
        ("peak_freq_hz", "Peak frequency", "Hz"),
        ("amp_peak_uv", "Peak amplitude", "µV"),
        ("rms_peak_z", "Peak RMS", "SD")) if c in ripples.columns]
    if not cols:
        return None
    levels = ([None] if split is None or split not in ripples.columns
              else list(pd.unique(ripples[split].dropna())))
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, len(cols), figsize=(5.4 * len(cols) * CM,
                                                        6.0 * CM))
        fig.subplots_adjust(wspace=0.55, top=0.74)
        axes = np.atleast_1d(axes)
        for ax, (c, lab, u) in zip(axes, cols):
            v_all = ripples[c].to_numpy(float)
            v_all = v_all[np.isfinite(v_all)]
            if not v_all.size:
                continue
            lo, hi = np.percentile(v_all, [0.5, 99.5])
            bins = np.linspace(lo, hi, 40)
            for j, lv in enumerate(levels):
                v = (v_all if lv is None
                     else ripples.loc[ripples[split] == lv, c].to_numpy(float))
                v = v[np.isfinite(v)]
                if not v.size:
                    continue
                ax.hist(v, bins=bins, density=True, histtype="step", lw=1.3,
                        color=("0.35" if lv is None else _cond_color(lv, j)),
                        label=(None if lv is None else f"{lv} (n={v.size})"))
                ax.axvline(np.median(v), color=("0.35" if lv is None
                                                else _cond_color(lv, j)),
                           lw=0.8, ls=":")
            ax.set_xlabel(f"{lab} ({u})")
            ax.set_ylabel("Density")
            ax.set_title(f"median {np.median(v_all):.3g} {u}", fontsize=FS_TICK)
            if levels != [None]:
                ax.legend(fontsize=FS_TICK - 2.5, frameon=False)
        if title:
            fig.suptitle(title, fontsize=FS_TITLE, y=1.14)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return fig


def rate_by_ordinal(tab, x, out_stem=None, title="", xlabel="Repeat number",
                    split=None):
    """Ripple rate against an ordinal (repeat number, solve index).

    The learning curve view: if ripples support loading a plan, rate should
    fall as the route becomes automatic. A two-level contrast cannot show
    whether the change is monotonic or a step.
    """
    if x not in tab.columns:
        return None
    levels = [None] if split is None or split not in tab.columns else \
        list(pd.unique(tab[split].dropna()))
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(9 * CM, 6.5 * CM))
        for j, lv in enumerate(levels):
            d = tab if lv is None else tab[tab[split] == lv]
            g = (d.groupby([x, "subject_key"])
                 .agg(n=("n_ripples", "sum"), e=("exposure_s", "sum")))
            g["r"] = g["n"] / g["e"].replace(0, np.nan)
            m = g.groupby(level=0)["r"].mean()
            se = g.groupby(level=0)["r"].agg(
                lambda v: v.std() / max(np.sqrt(v.notna().sum()), 1))
            k = g.groupby(level=0)["r"].count()
            keep = k >= 5           # drop ordinals carried by <5 subjects
            col = "0.35" if lv is None else _cond_color(lv, j)
            ax.errorbar(m.index[keep], m[keep], yerr=se[keep], fmt="o-", ms=3,
                        lw=1.1, color=col, ecolor=col, capsize=2,
                        label=(None if lv is None else str(lv)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Ripple rate (Hz)")
        ax.set_title(title or "Rate across repeats", fontsize=FS_TICK)
        if levels != [None]:
            ax.legend(fontsize=FS_TICK - 2, frameon=False)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return fig


def rate_distribution_figure(channel_qc, out_stem=None, title=""):
    """Where the ripple rate sits, per derivation / session, against Chen.

    The QC table gives one row per bipolar derivation. Chen's reported range
    (0.17-0.24 Hz) is drawn as a band so an outlying session is visible as an
    outlier rather than as a number in a CSV.
    """
    need = {"rate_hz", "session"}
    if not need.issubset(channel_qc.columns):
        return None
    q = channel_qc[channel_qc.rate_hz.notna()]
    if not len(q):
        return None
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, 3, figsize=(19 * CM, 6.2 * CM))
        fig.subplots_adjust(wspace=0.36, top=0.74)

        ax = axes[0]
        ax.hist(q.rate_hz, bins=30, color=PAL[1], edgecolor="w", lw=0.4)
        ax.axvspan(0.17, 0.24, color=OBS_LINE_C, alpha=0.13, lw=0)
        ax.axvline(q.rate_hz.median(), color=OBS_LINE_C, lw=1.4)
        ax.set_xlabel("Ripple rate (Hz)")
        ax.set_ylabel("Derivations")
        ax.set_title(f"n={len(q)} derivations\nmedian {q.rate_hz.median():.3f} Hz "
                     "(band = Chen)", fontsize=FS_TICK)

        ax = axes[1]
        g = q.groupby("session").rate_hz.median().sort_values()
        ax.plot(np.arange(len(g)), g.to_numpy(), "o", ms=3.2, color=PAL[1])
        ax.axhspan(0.17, 0.24, color=OBS_LINE_C, alpha=0.13, lw=0)
        ax.set_xlabel("Session (ordered)")
        ax.set_ylabel("Median rate (Hz)")
        ax.set_title(f"{len(g)} sessions", fontsize=FS_TICK)

        ax = axes[2]
        if "clean_s" in q.columns:
            ax.scatter(q.clean_s / 60.0, q.rate_hz, s=10, color=PAL[1],
                       alpha=0.7, edgecolor="none")
            ax.set_xlabel("Artifact-free recording (min)")
            ax.set_ylabel("Ripple rate (Hz)")
            ax.set_title("Rate vs available clean time\n(should be flat)",
                         fontsize=FS_TICK)
        else:
            ax.axis("off")
        if title:
            fig.suptitle(title, fontsize=FS_TITLE, y=1.14)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return fig


ROI_UNIT_C = {"HC": "#23677E", "mPFC": PAL[1], "EC": PAL[0], "mOFC": PAL[4],
              "OFC": PAL[4], "PCC": PAL[3], "AMY": "#a30d6c", "other": "0.6"}


def ripple_triggered_units_figure(offsets, z_curves, unit_table,
                                  test_win=(0.0, 0.2), out_stem=None, title="",
                                  regions=("HC", "mPFC")):
    """Peri-ripple firing z per region, plus the per-unit distribution.

    Top row: mean +- SEM across units of the z time course, one axis per region.
    HC comes first deliberately -- it is the positive control.
    Bottom row: every unit's z in the pre-declared test window, so the group
    mean cannot hide being driven by a couple of units.
    """
    tcols = [c for c in z_curves.columns if c not in ("session", "unit")]
    Z = z_curves[tcols].to_numpy(float)
    offs = np.asarray(offsets, float)
    regs = [r for r in regions if (unit_table.region == r).any()]
    if not regs:
        return None
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(2, len(regs), squeeze=False,
                                 figsize=(8.0 * len(regs) * CM, 12 * CM),
                                 gridspec_kw=dict(hspace=0.75, wspace=0.4))
        for j, reg in enumerate(regs):
            sel = (unit_table.region == reg).to_numpy()
            col = ROI_UNIT_C.get(reg, PAL[j % len(PAL)])

            ax = axes[0][j]
            M = Z[sel]
            m = np.nanmean(M, axis=0)
            se = np.nanstd(M, axis=0) / max(np.sqrt(np.isfinite(M).any(axis=1).sum()), 1)
            ax.plot(offs, m, color=col, lw=1.4)
            ax.fill_between(offs, m - se, m + se, color=col, alpha=0.22, lw=0)
            ax.axhline(0, color="0.6", lw=0.8, ls=":")
            ax.axvline(0, color="0.5", lw=0.8, ls=":")
            ax.axvspan(*test_win, color=OBS_LINE_C, alpha=0.10, lw=0)
            ax.set_xlabel("Time from ripple peak (s)")
            ax.set_ylabel("Firing (z vs shifted null)")
            n_u = int(sel.sum())
            n_s = int(unit_table.loc[sel, "subject_key"].nunique())
            extra = "  (positive control)" if reg == "HC" else ""
            ax.set_title(f"{reg}: {n_u} units, {n_s} subjects{extra}",
                         fontsize=FS_TICK)

            ax = axes[1][j]
            v = unit_table.loc[sel, "z_test_window"].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size:
                ax.hist(v, bins=max(8, min(30, v.size // 2)), color=col,
                        edgecolor="w", lw=0.4)
                ax.axvline(0, color="0.6", lw=0.9, ls=":")
                ax.axvline(float(np.mean(v)), color=OBS_LINE_C, lw=1.5)
                ax.text(0.97, 0.93, f"mean {np.mean(v):+.2f}\n"
                                    f"{int((v > 0).sum())}/{v.size} > 0",
                        ha="right", va="top", transform=ax.transAxes,
                        fontsize=FS_TICK - 1, color=OBS_LINE_C)
            ax.set_xlabel(f"z, {test_win[0]*1000:.0f}-{test_win[1]*1000:.0f} ms")
            ax.set_ylabel("Units")
            ax.set_title("Per unit", fontsize=FS_TICK)
        if title:
            fig.suptitle(title, fontsize=FS_TITLE, y=1.005)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return fig


# What the two ends of the coefficient axis mean, per hypothesis, and the
# hypothesis itself. Without these a reader cannot tell which direction
# supports the claim -- the GLM codes the TEST condition as the reference, so
# its coefficient points the opposite way to the log rate ratio.
SWEEP_DIRECTION = {
    "H1": ("HYPOTHESIS: first-D > later-D", "later-D > first-D"),
    "H2": ("HYPOTHESIS: exploration > later repeats", "later > exploration"),
    "H5": ("HYPOTHESIS: plan > execute", "execute > plan"),
    "H3": ("HYPOTHESIS: more ripples -> fewer errors", "more ripples -> more errors"),
    "H4": ("error > correct", "correct > error"),
}
SWEEP_HYPOTHESIS = {
    "H1": "ripples elevated at the FIRST arrival at D vs later arrivals",
    "H2": "ripples elevated while still planning vs during execution",
    "H3": "ripple rate after first-D predicts FEWER later errors",
    "H5": "ripples elevated when planning vs executing",
}


def sweep_figure(tab, which, out_stem=None):
    """Every definitional variant as a coefficient with its CI, ordered as run.

    The point is to see whether a conclusion depends on a choice nobody
    defended. A column of overlapping intervals means the choice does not
    matter; a sign flip across variants means it decides the answer.
    """
    if not len(tab) or "coef" not in tab.columns:
        return None
    t = tab.dropna(subset=["coef"]).reset_index(drop=True)
    if not len(t):
        return None
    with plt.rc_context(_rc()):
        h = max(6.0, 0.55 * len(t) + 3.2)
        fig, ax = plt.subplots(figsize=(13 * CM, h * CM))
        y = np.arange(len(t))[::-1]
        sig = t.p_glm < 0.05 if "p_glm" in t.columns else np.zeros(len(t), bool)
        cols = [OBS_LINE_C if s else "0.55" for s in sig]
        ax.errorbar(t.coef, y, xerr=1.96 * t.se, fmt="none", ecolor="0.75", lw=1)
        ax.scatter(t.coef, y, s=28, c=cols, zorder=3)
        ax.axvline(0, color="0.5", lw=0.9, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels(t.variant, fontsize=FS_TICK - 2)
        ax.set_xlabel("GLM coefficient (95% CI)")
        # The single most confusing thing about this plot is that the GLM
        # coefficient runs OPPOSITE to the hypothesis: the model is coded with
        # the test condition as reference, so a positive coefficient means the
        # control condition had more ripples. Say so on the axis.
        hyp = SWEEP_DIRECTION.get(which, ("test > control", "control > test"))
        ax.annotate(f"<-- {hyp[0]}", xy=(0.02, -0.13), xycoords="axes fraction",
                    ha="left", fontsize=FS_TICK - 1, color=OBS_LINE_C)
        ax.annotate(f"{hyp[1]} -->", xy=(0.98, -0.13), xycoords="axes fraction",
                    ha="right", fontsize=FS_TICK - 1, color="0.45")
        ax.set_title(f"{which}: does the conclusion survive the definition?\n"
                     f"{SWEEP_HYPOTHESIS.get(which, '')}\n"
                     f"filled = p < 0.05 (exploratory, uncorrected, no permutation)",
                     fontsize=FS_TICK)
        for i, (_, r) in enumerate(t.iterrows()):
            if np.isfinite(r.get("p_glm", np.nan)):
                ax.text(1.02, y[i], f"p={r.p_glm:.3f}  n={int(r.n_rows)}",
                        transform=ax.get_yaxis_transform(), va="center",
                        fontsize=FS_TICK - 2.5, color="0.35")
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return fig


def window_definition_figure(counts_by_variant, which, out_stem=None,
                             max_variants=6):
    """What the windows in each variant actually look like.

    Window length and window count are the two things that decide how much
    evidence a condition contributes, and they are invisible in a coefficient
    table. One row per variant: the duration distribution per condition, and
    how many windows and ripples each condition has.
    """
    items = list(counts_by_variant.items())[:max_variants]
    if not items:
        return None
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(len(items), 2, squeeze=False,
                                 figsize=(17 * CM, 4.6 * len(items) * CM),
                                 gridspec_kw=dict(hspace=1.0, wspace=0.3))
        for i, (lab, c) in enumerate(items):
            col = next((k for k in ("condition", "phase_after", "phase3")
                        if k in c.columns), None)
            if col is None:
                continue
            groups = list(pd.unique(c[col].astype(str).sort_values()))
            ax = axes[i][0]
            for j, g in enumerate(groups):
                v = c.loc[c[col].astype(str) == g, "duration_s"].to_numpy(float)
                v = v[np.isfinite(v)]
                if not v.size:
                    continue
                ax.hist(v, bins=30, histtype="step", lw=1.3,
                        color=_cond_color(g, j), density=True,
                        label=f"{g}  med {np.median(v):.2f}s")
            ax.set_xlabel("Window duration (s)")
            ax.set_ylabel("Density")
            ax.set_title(lab, fontsize=FS_TICK - 1)
            ax.legend(fontsize=FS_TICK - 3, frameon=False)

            ax = axes[i][1]
            g2 = c.groupby(c[col].astype(str)).agg(
                w=("n_ripples", "size"), n=("n_ripples", "sum"),
                e=("exposure_s", "sum")).reindex(groups)
            xx = np.arange(len(groups))
            ax.bar(xx, g2["w"], color=[_cond_color(g, j)
                                       for j, g in enumerate(groups)],
                   edgecolor="w")
            ax.set_xticks(xx)
            ax.set_xticklabels([g.replace("_", "\n") for g in groups],
                               fontsize=FS_TICK - 2)
            ax.set_ylabel("Windows")
            for k, (_, r) in enumerate(g2.iterrows()):
                ax.text(k, r["w"], f"{int(r['n'])} rip\n{r['e']:.0f}s",
                        ha="center", va="bottom", fontsize=FS_TICK - 3)
            ax.margins(y=0.25)
            ax.set_title("Evidence per condition", fontsize=FS_TICK - 1)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return fig


def pvth_figure(centres, prof_by_subject, results, which, out_stem=None,
                pre_win=(-0.6, -0.1), base_win=(-1.6, -1.1), smooth=True,
                windows=None):
    """Peri-event time histogram in the Sakon & Kahana (2022) Fig. 2B style.

    Their conventions, kept deliberately: mean +- SEM across subjects; a dotted
    grey reference line to aid comparison between panels; the significant time
    range marked by a bar above the traces; the analysis and baseline windows
    shaded so the reader can see where the statistic was taken. Traces are
    triangle-smoothed for display only -- every number comes from the unsmoothed
    data.
    """
    import mc.analyse.swr_sakon as sk
    conds = sorted({c for _, c in prof_by_subject})
    if not conds:
        return None
    centres = np.asarray(centres, float)
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 6.6 * CM),
                                 gridspec_kw=dict(width_ratios=[2, 1]))
        fig.subplots_adjust(wspace=0.32, top=0.74)

        ax = axes[0]
        ax.axvspan(*base_win, color="0.85", lw=0, zorder=0)
        ax.axvspan(*pre_win, color=OBS_LINE_C, alpha=0.12, lw=0, zorder=0)
        ymax = -np.inf
        for j, cond in enumerate(conds):
            subs = sorted({s for s, c in prof_by_subject if c == cond})
            X = np.vstack([prof_by_subject[(s, cond)] for s in subs])
            m = np.nanmean(X, axis=0)
            se = np.nanstd(X, axis=0) / max(np.sqrt(len(subs)), 1)
            col = _cond_color(cond, j)
            mm, ss = (sk.triangle_smooth(m), sk.triangle_smooth(se)) if smooth else (m, se)
            ax.plot(centres, mm, color=col, lw=1.4, label=f"{cond} (n={len(subs)})")
            ax.fill_between(centres, mm - ss, mm + ss, color=col, alpha=0.22, lw=0)
            ymax = max(ymax, np.nanmax(mm + ss))
        # significant clusters, marked as a bar above the traces
        for j, cond in enumerate(conds):
            for c in results.get("clusters", {}).get(cond, []):
                if c["p"] < 0.05:
                    ax.plot([c["t_start_s"], c["t_stop_s"]],
                            [ymax * (1.04 + 0.05 * j)] * 2,
                            color=_cond_color(cond, j), lw=2.5,
                            solid_capstyle="butt")
        ax.axvline(0, color="0.35", lw=1.0)
        ax.set_xlabel("Time from event (s)")
        ax.set_ylabel("Ripple rate (events/s)")
        ax.set_title(f"{which}: peri-event ripple rate\n"
                     f"shaded = baseline (grey) and PRE (green) windows",
                     fontsize=FS_TICK)
        ax.legend(fontsize=FS_TICK - 2, frameon=False, loc="upper left")
        ax.margins(y=0.18)

        # before / during / after, each against the same baseline
        ax = axes[1]
        wnames = list(windows) if windows else ["_pre"]
        xs = np.arange(len(wnames))
        for j, cond in enumerate(conds):
            got = (results.get("eq2") or {}).get(cond) or {}
            m, lo, hi, stars = [], [], [], []
            for w in wnames:
                e2 = got.get(w) if isinstance(got.get(w), dict) else None
                if not e2 or "mean_t" not in e2:
                    m.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                    stars.append(False); continue
                per = pd.DataFrame(e2["per_subject"])["t"].to_numpy(float)
                per = per[np.isfinite(per)]
                m.append(per.mean())
                se = per.std(ddof=1) / max(np.sqrt(per.size), 1)
                lo.append(se); hi.append(se)
                stars.append(e2.get("p", 1) < 0.05)
            col = _cond_color(cond, j)
            off = (j - (len(conds) - 1) / 2) * 0.14
            ax.errorbar(xs + off, m, yerr=[lo, hi], fmt="o", ms=5, lw=1.4,
                        capsize=3, color=col, label=str(cond))
            for k, st in enumerate(stars):
                if st and np.isfinite(m[k]):
                    ax.text(xs[k] + off, m[k] + hi[k] + 0.04, "*", ha="center",
                            fontsize=FS_TITLE, color=col)
        ax.axhline(0, color="0.6", lw=0.9, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels([w.split("(")[0].strip() + "\n"
                            + ("(" + w.split("(")[1] if "(" in w else "")
                            for w in wnames], fontsize=FS_TICK - 2)
        ax.set_ylabel("Eq. 2 t score vs own baseline")
        ax.set_title("Before / during / after\npositive = above own baseline",
                     fontsize=FS_TICK)
        ax.legend(fontsize=FS_TICK - 2.5, frameon=False)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return fig


def sliding_cluster_figure(sw_by_cond, which, width_key, out_stem=None):
    """The sliding-window test and the permutation null that corrects it.

    Left: the t statistic at every window position, with the cluster-forming
    threshold drawn and any surviving cluster shaded. Nothing was chosen here --
    every position was evaluated, which is the point of the panel.
    Right: the null distribution of maximum cluster mass from the sign-flip
    permutation, with the observed cluster masses marked. That is what the
    p-value is, drawn rather than asserted.
    """
    conds = list(sw_by_cond)
    if not conds:
        return None
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(18 * CM, 6.4 * CM),
                                 gridspec_kw=dict(width_ratios=[1.85, 1]))
        fig.subplots_adjust(wspace=0.3, top=0.72)

        ax = axes[0]
        thr = None
        for j, cond in enumerate(conds):
            sw = sw_by_cond[cond]
            col = _cond_color(cond, j)
            ax.plot(sw["centres"], sw["t"], color=col, lw=1.5, label=str(cond))
            thr = sw.get("null", {}).get("threshold_t", thr)
            for c in sw["clusters"]:
                if c["p"] < 0.05:
                    ax.axvspan(c["t_start_s"], c["t_stop_s"], color=col,
                               alpha=0.16, lw=0)
                    ax.annotate(f"p = {c['p']:.3f}",
                                xy=(c["peak_at_s"], 0), xytext=(0, -26),
                                textcoords="offset points", ha="center",
                                fontsize=FS_TICK - 1.5, color=col)
        if thr:
            for sgn in (1, -1):
                ax.axhline(sgn * thr, color="0.6", lw=0.8, ls=":")
        ax.axhline(0, color="0.45", lw=0.9)
        ax.axvline(0, color="0.35", lw=1.1)
        ax.set_xlabel("Centre of the sliding window (s from event)")
        ax.set_ylabel("t vs own baseline")
        ax.set_title(f"{which}: every window position tested "
                     f"({width_key} wide)\ndotted = cluster-forming threshold; "
                     f"shaded = survives correction", fontsize=FS_TICK)
        ax.legend(fontsize=FS_TICK - 2, frameon=False)

        # The null of MAX cluster mass has a large atom at zero -- most
        # sign-flipped surrogates contain no suprathreshold run at all -- and a
        # long right tail. On a linear axis the zero spike flattens everything
        # else, so the tail (which is what the p-value reads off) is invisible.
        # Log counts, and the zero fraction stated rather than drawn.
        ax = axes[1]
        zero_txt = []
        for j, cond in enumerate(conds):
            n = sw_by_cond[cond].get("null", {})
            mass = np.asarray(n.get("null_mass", []), float)
            if not mass.size:
                continue
            col = _cond_color(cond, j)
            frac0 = float(np.mean(mass <= 0))
            zero_txt.append(f"{cond}: {frac0:.0%} of surrogates have no cluster")
            pos = mass[mass > 0]
            if pos.size:
                ax.hist(pos, bins=40, color=col, alpha=0.45, lw=0,
                        label=f"{cond} null (mass > 0)")
            for m in n.get("obs_mass", []):
                ax.axvline(m, color=col, lw=1.8)
        ax.set_yscale("log")
        ax.set_xlabel("Max cluster mass under sign-flipping")
        ax.set_ylabel("Permutations (log)")
        ax.set_title("Permutation null\nvertical lines = observed clusters",
                     fontsize=FS_TICK)
        ax.legend(fontsize=FS_TICK - 3, frameon=False, loc="upper right")
        if zero_txt:
            ax.text(0.02, -0.34, "\n".join(zero_txt), transform=ax.transAxes,
                    fontsize=FS_TICK - 3, color="0.45", va="top")
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png")
    return fig


# ------------------------------------------------- contact coverage ----
#
# Where the hippocampal derivations actually are. Three orthogonal glass-brain
# projections with the Harvard-Oxford hippocampus (the same 25 % probability
# map that selected the contacts) shaded behind them, so the figure shows the
# selection criterion and its result in one picture.

SITE_C = {"baylor": "#B74C2D", "utah": "#448363", "ucla": "#CCB178"}
SITE_LABEL = {"baylor": "Baylor", "utah": "Utah", "ucla": "UCLA"}
HC_SHADE = ("#dcdcdc", "#5a5a5a")     # light -> dark grey ramp for the HC MIP
HPC_PROB_SHOW = 25.0                  # matches contact_anatomy.HPC_PROB_MIN


def _hippocampus_prob_img():
    """Harvard-Oxford subcortical P(hippocampus) as a single 3-D image.

    Left and right volumes are combined with `max`, exactly as
    `anatomy_atlas.hippocampal_probability` does when it reads a contact, so
    the shading and the selection rule cannot drift apart.
    """
    import nibabel as nib
    from nilearn import datasets as nldatasets

    atlas = nldatasets.fetch_atlas_harvard_oxford("sub-prob-2mm")
    img = atlas.maps if hasattr(atlas.maps, "get_fdata") else nib.load(atlas.maps)
    names = [str(n) for n in atlas.labels]
    n_vol = img.shape[3] if img.ndim == 4 else 1
    off = 1 if len(names) == n_vol + 1 else 0        # drop the 'Background' entry
    idx = [i - off for i, n in enumerate(names) if "hippocampus" in n.lower()]
    idx = [i for i in idx if 0 <= i < n_vol]
    if not idx:
        raise RuntimeError("no hippocampus volume in the HO subcortical atlas")
    data = img.get_fdata()[..., idx].max(axis=-1)
    out = nib.Nifti1Image(data, img.affine)
    # A 2 mm atlas projected as a maximum-intensity silhouette has visibly
    # blocky edges at print size; a light smooth is cosmetic only and does not
    # touch the map that selects contacts.
    from nilearn.image import smooth_img
    return smooth_img(out, 2.0)


def contact_coverage_figure(contacts, out_stem=None, height_cm=3.5,
                            show_unselected=True, title=None):
    """Glass-brain coverage figure: every macro contact, hippocampal ones on top.

    `contacts` is the pooled macro-contact table (`macro_contacts_all.csv`):
    one row per recording channel, with `mni_x/y/z`, `resolved`, `is_hpc` and
    `recording_site`.

    Unselected resolved contacts are drawn small and pale to show the extent of
    the implantation; the selected hippocampal contacts are drawn on top,
    coloured by recording site, so it is visible at a glance that coverage is
    not carried by one centre.

    Sized for print at `height_cm` (default 3.5 cm) with 9 pt Arial, and
    written to `<out_stem>.pdf` (vector) and `<out_stem>.jpg` (300 dpi).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from nilearn import plotting as nlplot

    df = contacts.copy()
    xyz = ["mni_x", "mni_y", "mni_z"]
    df = df[df[xyz].notna().all(axis=1)]
    if "resolved" in df.columns:
        df = df[df["resolved"].fillna(False).astype(bool)]
    sel = df[df["is_hpc"].fillna(False).astype(bool)]
    rest = df[~df["is_hpc"].fillna(False).astype(bool)]

    # Three orthogonal projections; the widths below are the MNI bounding-box
    # aspect ratios, so the panels are not distorted relative to each other.
    # 3.05 is the summed MNI bounding-box aspect of the three projections, so
    # the panels keep their true proportions; the rest of the width is legend.
    brain_frac = 0.72
    h_in = height_cm * CM
    w_in = h_in * 3.05 / brain_frac

    # _rc() sets savefig.bbox="tight", which would grow the saved page past
    # `height_cm` to make room for the L/R annotations. The axes rects below
    # already fill the figure, so the page is saved exactly as sized.
    rc = dict(_rc()); rc["savefig.bbox"] = None
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(w_in, h_in))
        cmap = LinearSegmentedColormap.from_list("hc_shade", HC_SHADE)
        disp = nlplot.plot_glass_brain(
            _hippocampus_prob_img(), display_mode="ortho",
            figure=fig, axes=(0.0, 0.02, brain_frac, 0.94),
            cmap=cmap, vmin=0, vmax=100, threshold=HPC_PROB_SHOW,
            plot_abs=False, colorbar=False, black_bg=False, alpha=0.5,
            annotate=False)
        # nilearn's own L/R annotation ignores the rc font size and prints
        # far too large on a 3.5 cm panel.
        disp.annotate(size=FS_TICK)

        if show_unselected and len(rest):
            disp.add_markers(rest[xyz].to_numpy(float),
                             marker_color="#c8c8c8", marker_size=0.8,
                             alpha=0.35, edgecolors="none")
        for site, g in sel.groupby("recording_site"):
            disp.add_markers(g[xyz].to_numpy(float),
                             marker_color=SITE_C.get(str(site), "#888888"),
                             marker_size=6, alpha=0.95, edgecolors="none")

        handles = [Line2D([], [], marker="o", linestyle="none",
                          markersize=3.2, markeredgewidth=0,
                          color=SITE_C.get(s, "#888888"),
                          label=f"{SITE_LABEL.get(s, s)} ({(sel.recording_site == s).sum()})")
                   for s in ["baylor", "utah", "ucla"]
                   if (sel.recording_site == s).any()]
        if show_unselected and len(rest):
            handles.append(Line2D([], [], marker="o", linestyle="none",
                                  markersize=2.0, markeredgewidth=0,
                                  color="#bfbfbf",
                                  label=f"other contacts ({len(rest):,})"))
        # The threshold belongs in the caption, not the key: spelling it out
        # here is the one label that does not fit a 3.5 cm panel.
        handles.append(Line2D([], [], marker="s", linestyle="none",
                              markersize=3.2, markeredgewidth=0,
                              color=HC_SHADE[1], label="hippocampus"))

        leg_ax = fig.add_axes([brain_frac + 0.005, 0.0,
                               1.0 - brain_frac - 0.005, 1.0])
        leg_ax.axis("off")
        leg_ax.legend(handles=handles, loc="center left", frameon=False,
                      fontsize=FS_TICK, handletextpad=0.4,
                      labelspacing=0.45, borderpad=0.0, borderaxespad=0.0)
        if title:
            fig.suptitle(title, fontsize=FS_TITLE, y=1.04)

        if out_stem:
            # bbox_inches=None, not "tight": the axes rects already fill the
            # figure, and a tight box would grow it past `height_cm` to make
            # room for the L/R annotations, which is the one dimension that
            # has to be exact for a print panel.
            fig.savefig(out_stem + ".pdf")
            fig.savefig(out_stem + ".jpg", dpi=300)
        return fig


# ROI colours for the SWR montage figure. mPFC/mOFC/HC follow the project-wide
# convention in CLAUDE.md (Showgirl2 indices 1 and 4, HC teal). The control
# regions are deliberately GREY: they are there to be unremarkable, and a
# reader should be able to tell target from control without reading the key.
MONTAGE_C = {
    "HC_anterior": "#23677E", "HC_mid": PAL[2], "HC": PAL[2], "EC": PAL[0],
    "mPFC": PAL[1], "mOFC": PAL[4],
    "TemporalLateral": "#8C8C8C", "Auditory": "#4F4F4F", "Visual": "#B0A8C0",
    "Insula": "#D8C9A3", "PCC": PAL[3],
}
MONTAGE_ORDER = ["HC_anterior", "HC_mid", "mPFC", "mOFC",
                 "TemporalLateral", "Auditory", "Visual", "Insula", "PCC"]


def montage_figure(pairs, out_stem=None, height_cm=5.0, title=None,
                   show_unselected=None):
    """Where every derivation sits: hippocampus, mPFC/mOFC, and the controls.

    `pairs` is the pooled bipolar-pair table with `mni_x/y/z`, `pair_roi_atlas`
    and `role`. Markers are at the pair MIDPOINT, which is where a bipolar
    derivation is actually sensitive -- not at either contact.

    Derivations on the SAME SHAFT as the hippocampal contact are ringed in
    black. That subset is the tightest control this dataset offers: same
    electrode, same amplifier, same reference chain, same noise environment, so
    a difference against it cannot be a recording-quality difference. It is
    worth being able to see how much of the temporal control has that property
    (measured: 158 of 353 derivations).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from nilearn import plotting as nlplot

    xyz = ["mni_x", "mni_y", "mni_z"]
    df = pairs.copy()
    df = df[df[xyz].notna().all(axis=1)]
    roi = df["pair_roi_atlas"].astype(str)
    present = [r for r in MONTAGE_ORDER if (roi == r).any()]

    h_in = height_cm * CM
    brain_frac = 0.70
    w_in = h_in * 3.05 / brain_frac

    rc = dict(_rc()); rc["savefig.bbox"] = None
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(w_in, h_in))
        cmap = LinearSegmentedColormap.from_list("hc_shade", HC_SHADE)
        # savefig.bbox is None here (the panel size has to be exact), so a
        # suptitle above y=1 is simply cut off. Give it room inside the figure
        # instead of letting it hang over the edge.
        top = 0.88 if title else 0.94
        disp = nlplot.plot_glass_brain(
            _hippocampus_prob_img(), display_mode="ortho",
            figure=fig, axes=(0.0, 0.02, brain_frac, top),
            cmap=cmap, vmin=0, vmax=100, threshold=HPC_PROB_SHOW,
            plot_abs=False, colorbar=False, black_bg=False, alpha=0.45,
            annotate=False)
        disp.annotate(size=FS_TICK)

        same = df.get("same_probe_as_hpc")
        for r in present:
            g = df[roi == r]
            on_shaft = (g["same_probe_as_hpc"].fillna(False).astype(bool)
                        if same is not None else None)
            # Hippocampal rows are always on "their own" shaft; ringing them
            # would say nothing, so only cortical rows get the ring.
            if on_shaft is not None and r not in ("HC_anterior", "HC_mid", "HC"):
                plain, ringed = g[~on_shaft], g[on_shaft]
            else:
                plain, ringed = g, g.iloc[0:0]
            if len(plain):
                disp.add_markers(plain[xyz].to_numpy(float),
                                 marker_color=MONTAGE_C.get(r, "#888888"),
                                 marker_size=11, alpha=0.9, edgecolors="none")
            if len(ringed):
                disp.add_markers(ringed[xyz].to_numpy(float),
                                 marker_color=MONTAGE_C.get(r, "#888888"),
                                 marker_size=13, alpha=0.95,
                                 edgecolors="#111111", linewidths=0.45)

        handles = []
        for r in present:
            n = int((roi == r).sum())
            ns = int(df.loc[roi == r, "same_probe_as_hpc"].fillna(False).sum()) \
                if same is not None else 0
            lab = f"{r} ({n})"
            if ns and r not in ("HC_anterior", "HC_mid", "HC"):
                lab += f", {ns} on HC shaft"
            handles.append(Line2D([], [], marker="o", linestyle="none",
                                  markersize=3.4, markeredgewidth=0,
                                  color=MONTAGE_C.get(r, "#888888"), label=lab))
        handles.append(Line2D([], [], marker="o", linestyle="none",
                              markersize=3.4, markerfacecolor="none",
                              markeredgecolor="#111111", markeredgewidth=0.6,
                              color="none", label="on the hippocampal shaft"))

        leg_ax = fig.add_axes([brain_frac + 0.005, 0.0,
                               1.0 - brain_frac - 0.005, 1.0])
        leg_ax.axis("off")
        leg_ax.legend(handles=handles, loc="center left", frameon=False,
                      fontsize=FS_TICK, handletextpad=0.4,
                      labelspacing=0.42, borderpad=0.0, borderaxespad=0.0)
        if title:
            fig.suptitle(title, fontsize=FS_TITLE, y=0.985)
        if out_stem:
            fig.savefig(out_stem + ".pdf")
            fig.savefig(out_stem + ".jpg", dpi=300)
        return fig


# ============================================================ 3-D coverage ==
# A surface rendering, rather than the glass-brain projection above. The
# projection is honest about every contact but flattens the hippocampus into a
# smear; a translucent pial surface with the hippocampus rendered as a solid
# body puts each contact inside a structure the reader recognises.
FSAVERAGE_ENV = "SUBJECTS_DIR"
CORTEX_C = (0.78, 0.78, 0.78)     # pial surface, drawn nearly invisible
CORTEX_ALPHA = 0.085
HPC_BODY_C = "#8FC2D4"            # a light tint of the project HC teal
# The full-strength #23677E is too dark for this figure: black contacts sit
# ON the structure, and against it they disappeared. The tint keeps the hue
# the project uses for hippocampus while giving the markers something to
# read against.
HPC_BODY_ALPHA = 0.72
CONTACT_C = "#0d0d0d"
CONTACT_EXCLUDED_C = "#b9b9b9"

# MNI305 (= fsaverage surface RAS) -> MNI152. FreeSurfer's mni152reg matrix.
# Our coordinates are MNI152, so foci are pushed through its inverse before
# being planted on the fsaverage surface. The offset is only 1-2 mm, but it is
# a systematic one and free to remove.
_MNI305_TO_MNI152 = np.array([
    [0.9975, -0.0073, 0.0176, -0.0429],
    [0.0146, 1.0009, -0.0024, 1.5496],
    [-0.0130, -0.0093, 0.9971, 1.1840],
    [0.0, 0.0, 0.0, 1.0]])


def mni152_to_fsaverage(coords):
    """MNI152 mm -> fsaverage surface RAS (MNI305) mm."""
    xyz = np.asarray(coords, float).reshape(-1, 3)
    inv = np.linalg.inv(_MNI305_TO_MNI152)
    return (inv @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]


# Camera angles in mne.viz.Brain's convention, verified by rendering landmark
# foci at (0, 70, 0), (0, 0, 75) and (-70, 0, 0) rather than by guessing: the
# rolls below put anterior left in the lateral views and anterior up in the
# axial ones. Leaving roll unset gives a different orientation per call, which
# is why every entry names one.
BRAIN_VIEWS = {
    "left":     dict(azimuth=180, elevation=90,  roll=90),
    "right":    dict(azimuth=0,   elevation=90,  roll=270),
    "dorsal":   dict(azimuth=180, elevation=0,   roll=0),
    "ventral":  dict(azimuth=180, elevation=180, roll=180),
    "oblique":  dict(azimuth=215, elevation=65,  roll=90),
    "anterior": dict(azimuth=90,  elevation=90,  roll=0),
}


def _subjects_dir():
    """fsaverage's parent directory, fetching it if MNE has not already."""
    from mne.datasets import fetch_fsaverage
    return os.path.dirname(fetch_fsaverage(verbose=False))


HPC_PROB_MIN = 25.0        # the selection threshold, from contact_anatomy


def _hippocampus_mesh(prob_min=HPC_PROB_MIN):
    """The hippocampus AS THE ANALYSIS DEFINES IT, as a surface in MNI305.

    This draws the Harvard-Oxford subcortical *probability* map thresholded at
    the same value that selected the contacts, not FreeSurfer's `aseg`
    segmentation of fsaverage. The distinction is not cosmetic. The two
    disagree by a lot -- Harvard-Oxford at 25% is 15,296 mm3 against aseg's
    10,552 -- so a contact chosen at P = 30% is genuinely inside the criterion
    and genuinely outside `aseg`, and drawing `aseg` under it makes a correct
    selection look like a localisation error.

    A figure whose shaded region is not the region the text describes is
    misleading even when every number in the text is right.
    """
    import nibabel as nib
    import pyvista as pv
    from skimage.measure import marching_cubes
    import mc.analyse.anatomy_atlas as anat_atlas

    img, idx = anat_atlas._load_hc_prob()
    vol = img.get_fdata()[..., idx].max(-1)
    verts, faces, _, _ = marching_cubes(vol, level=float(prob_min))
    mni = nib.affines.apply_affine(img.affine, verts)
    return pv.PolyData(mni152_to_fsaverage(mni),
                       np.c_[np.full(len(faces), 3), faces].ravel())


def contacts_3d_views(included, excluded=None, hemispheres=None,
                      excluded_hemispheres=None, out_stem=None,
                      views=("left", "right", "dorsal"), size=(1600, 1200),
                      contact_scale=0.20, excluded_scale=0.18,
                      cortex_alpha=CORTEX_ALPHA, hpc_alpha=HPC_BODY_ALPHA,
                      hpc_color=HPC_BODY_C, contact_color=CONTACT_C,
                      excluded_color=CONTACT_EXCLUDED_C, distance=430,
                      hpc_source="atlas", prob_min=HPC_PROB_MIN):
    """Render hippocampal contacts inside a translucent fsaverage brain.

    `included` and `excluded` are (n, 3) arrays of MNI152 coordinates in mm.
    `hemispheres` (and `excluded_hemispheres`), if given, are matching arrays of
    'L'/'R': a lateral view then shows only the contacts on the side being
    looked at. Without that split, a left-lateral view draws right-hemisphere
    contacts 60 mm behind the left hippocampus and lets the reader believe they
    missed it. The axial views need no split -- the two sides are separated in
    the image already.

    `hpc_source` is 'atlas' (Harvard-Oxford at `prob_min`, the criterion that
    selected the contacts -- the default, and the honest one) or 'aseg'
    (FreeSurfer's segmentation of fsaverage, a smaller volume).

    Rendered under a PARALLEL projection. Under the default perspective camera
    the far hemisphere is magnified and displaced relative to the near one, so
    a contact sitting squarely inside the hippocampus can be drawn outside it.

    Returns {view_name: RGB screenshot}. With `out_stem`, each view is also
    written to `<out_stem>_<view>.png`.
    """
    import mne

    inc = np.asarray(included, float).reshape(-1, 3)
    exc = (np.asarray(excluded, float).reshape(-1, 3)
           if excluded is not None and len(excluded) else None)
    hemi = None if hemispheres is None else np.asarray(hemispheres, dtype=object)
    exc_hemi = (None if excluded_hemispheres is None
                else np.asarray(excluded_hemispheres, dtype=object))

    brain = mne.viz.Brain(
        "fsaverage", hemi="both", surf="pial", subjects_dir=_subjects_dir(),
        background="white", size=size, alpha=cortex_alpha, cortex=CORTEX_C,
        units="mm", offscreen=True)
    if hpc_source == "atlas":
        brain._renderer.plotter.add_mesh(
            _hippocampus_mesh(prob_min), color=hpc_color, opacity=hpc_alpha,
            smooth_shading=True)
    else:
        brain.add_volume_labels(
            aseg="aseg", labels=["Left-Hippocampus", "Right-Hippocampus"],
            colors=[hpc_color, hpc_color], alpha=hpc_alpha, smooth=0.0,
            fill_hole_size=1, legend=False)
    brain._renderer.plotter.enable_parallel_projection()

    shots = {}
    for name in views:
        side = {"left": "L", "right": "R"}.get(name)
        keep = slice(None) if (side is None or hemi is None) else (hemi == side)
        keep_e = (slice(None) if (side is None or exc_hemi is None)
                  else (exc_hemi == side))

        # Foci are re-added per view because the hemisphere subset changes; the
        # cortex and hippocampus meshes are built once and kept.
        added = []
        if exc is not None and len(exc[keep_e]):
            added.append(brain.add_foci(exc[keep_e], hemi="vol",
                                        color=excluded_color,
                                        scale_factor=excluded_scale, alpha=0.9))
        if len(inc[keep]):
            added.append(brain.add_foci(inc[keep], hemi="vol",
                                        color=contact_color,
                                        scale_factor=contact_scale))
        brain.show_view(distance=distance, **BRAIN_VIEWS[name])
        shots[name] = brain.screenshot()
        if out_stem:
            import imageio.v2 as imageio
            imageio.imwrite(f"{out_stem}_{name}.png", shots[name])
        for actor in added:
            try:
                brain._renderer.plotter.remove_actor(actor)
            except Exception:
                pass
    brain.close()
    return shots

def _trim_white(imgs, pad=10):
    """Crop each view to its content, then pad them to a common height.

    Cropping alone is not enough: matplotlib stretches every panel to the same
    axes height, so two views cropped to different heights would show the same
    brain at two different scales. Padding to a common height and letting the
    widths vary keeps the millimetres per pixel identical across panels while
    still removing the empty margin a renderer leaves at the sides.
    """
    out = []
    for im in imgs:
        mask = (im < 250).any(axis=2)
        if not mask.any():
            out.append(im)
            continue
        rows, cols = np.where(mask)
        r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, im.shape[0])
        c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, im.shape[1])
        out.append(im[r0:r1, c0:c1])

    h = max(im.shape[0] for im in out)
    padded = []
    for im in out:
        top = (h - im.shape[0]) // 2
        bot = h - im.shape[0] - top
        padded.append(np.pad(im, ((top, bot), (0, 0), (0, 0)),
                             constant_values=255))
    return padded


def contact_coverage_3d_figure(included, excluded=None, out_stem=None,
                               views=("left", "right", "dorsal"), width_cm=12.0,
                               view_labels=None, legend=True,
                               counts=None, save_views=True, **kw):
    """Publication panel: the 3-D views laid out side by side, with a key.

    A 3-D render is a raster whatever the container, so the PDF embeds the
    screenshot rather than pretending to be vector; it is written at 600 dpi so
    it survives being printed at a quarter of a page.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    shots = contacts_3d_views(included, excluded=excluded, views=views,
                              out_stem=out_stem if save_views else None, **kw)
    imgs = _trim_white([shots[v] for v in views])
    aspects = [im.shape[1] / im.shape[0] for im in imgs]

    # The axes are placed by hand rather than left to the default subplot
    # margins: an imshow axes keeps its image's aspect, so default margins
    # would leave a white band above and below the brains and shrink them for
    # no reason. Panel height follows from the width, since _trim_white has
    # already given every view the same pixel height.
    w_in = width_cm * CM
    title_in = 0.20 if view_labels else 0.0
    legend_in = 0.22 if legend else 0.0
    panel_h_in = w_in / sum(aspects)
    h_in = panel_h_in + title_in + legend_in

    rc = dict(_rc()); rc["savefig.bbox"] = "tight"
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            1, len(imgs), figsize=(w_in, h_in),
            gridspec_kw=dict(width_ratios=aspects, wspace=0.01,
                             left=0.0, right=1.0,
                             bottom=legend_in / h_in,
                             top=1.0 - title_in / h_in))
        axes = np.atleast_1d(axes)
        for ax, im, v in zip(axes, imgs, views):
            ax.imshow(im)
            ax.set_axis_off()
            if view_labels:
                ax.set_title(view_labels.get(v, v), fontsize=FS_TICK, pad=2)

        if legend:
            n_inc = len(included)
            n_exc = 0 if excluded is None else len(excluded)
            handles = [Line2D([], [], marker="o", linestyle="none",
                              markersize=3.4, markeredgewidth=0,
                              color=kw.get("contact_color", CONTACT_C),
                              label=f"analysed ({counts or n_inc})")]
            if n_exc:
                handles.append(Line2D(
                    [], [], marker="o", linestyle="none", markersize=3.0,
                    markeredgewidth=0,
                    color=kw.get("excluded_color", CONTACT_EXCLUDED_C),
                    label=f"not analysed ({n_exc})"))
            handles.append(Line2D([], [], marker="s", linestyle="none",
                                  markersize=3.4, markeredgewidth=0,
                                  color=kw.get("hpc_color", HPC_BODY_C),
                                  label="hippocampus"))
            fig.legend(handles=handles, loc="lower center", frameon=False,
                       ncol=len(handles), fontsize=FS_TICK,
                       handletextpad=0.35, columnspacing=1.4,
                       borderaxespad=0.1)

        if out_stem:
            fig.savefig(out_stem + ".pdf", dpi=600)
            fig.savefig(out_stem + ".png", dpi=600)
        return fig


# ==================================================== artifact, for print ==
# The per-session `artifact_figure` above is a diagnostic: dense, six panels,
# meant to be read at A4. These three are manuscript panels -- each makes one
# point, at the size it will be printed.

def ied_vs_ripple_figure(ied_raw, ripple_raw, fs, out_stem=None,
                         width_cm=3.5, height_cm=3.5, band=RIPPLE_BAND,
                         title=None, xlim_ms=200.0, lw=0.9):
    """A rejected discharge and an accepted ripple, on ONE voltage axis.

    The whole case for artifact rejection in an epilepsy cohort is that these
    two are separable, and a reader who will not read the five criteria will
    believe it or not on the strength of this panel. Both are drawn on a shared
    y-axis for exactly that reason: plotting each to its own scale is the
    standard way to make a 250 uV discharge and a 25 uV ripple look alike.

    `ied_raw` and `ripple_raw` are 1-D snippets, same length, centred on the
    event. The band-passed trace is drawn beneath each on its own (shared)
    axis, since the ripple band is ~1-2 uV against a raw swing of tens.
    """
    import matplotlib.pyplot as plt

    ied = np.asarray(ied_raw, float)
    rip = np.asarray(ripple_raw, float)
    n = min(len(ied), len(rip))
    ied, rip = ied[:n], rip[:n]
    t = (np.arange(n) - n / 2) / fs * 1000.0
    # Band-pass on the FULL snippet, then crop for display: filtering a window
    # that has already been cut rings at both ends.
    ied_bp, rip_bp = _bp(ied, fs, *band), _bp(rip, fs, *band)
    keep = np.abs(t) <= xlim_ms
    t, ied, rip = t[keep], ied[keep], rip[keep]
    ied_bp, rip_bp = ied_bp[keep], rip_bp[keep]

    def _nice(v):
        """A round tick value at or just below v -- 736 is not a tick."""
        if v <= 0:
            return 1.0
        mag = 10.0 ** np.floor(np.log10(v))
        for m in (10, 5, 2.5, 2, 1):
            if m * mag <= v:
                return m * mag
        return mag

    raw_lim = 1.10 * max(np.abs(ied).max(), np.abs(rip).max())
    bp_lim = 1.10 * max(np.abs(ied_bp).max(), np.abs(rip_bp).max())
    raw_tick, bp_tick = _nice(raw_lim / 1.4), _nice(bp_lim / 1.4)

    # At print size the panel is ~1.5 cm a cell, so everything optional goes:
    # two ticks an axis, short labels, and the condition named in two words.
    rc = dict(_rc())
    rc.update({"font.size": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
               "axes.labelsize": 9, "axes.titlesize": 9,
               "xtick.major.size": 2, "ytick.major.size": 2,
               "xtick.major.pad": 1.5, "ytick.major.pad": 1.5})
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(width_cm * CM, height_cm * CM),
                                 sharex=True, gridspec_kw=dict(
                                     height_ratios=[1.5, 1], hspace=0.24,
                                     wspace=0.22))
        for col, (x, xb, name, colour) in enumerate(
                [(ied, ied_bp, "discharge", CRIT_C["ied_janca"]),
                 (rip, rip_bp, "ripple", RIPPLE_C)]):
            ax = axes[0, col]
            ax.plot(t, x, color="0.3", lw=lw)
            ax.set_ylim(-raw_lim, raw_lim)
            ax.set_title(name, fontsize=9, color=colour, pad=2)
            ax.set_yticks([-raw_tick, 0, raw_tick])
            if col == 0:
                # "raw (uV)" rotated is longer than a 1.2 cm row is tall, so the
                # two row labels overlapped. The unit goes on the axis and the
                # row identity goes inside, where there is space.
                ax.set_ylabel(r"$\mu$V", labelpad=1)
                ax.text(0.02, 0.97, "broadband", transform=ax.transAxes,
                        ha="left", va="top", fontsize=7.5, color="0.45")
            else:
                ax.set_yticklabels([])

            ax = axes[1, col]
            ax.plot(t, xb, color=colour, lw=lw)
            ax.set_ylim(-bp_lim, bp_lim)
            ax.set_xlim(-xlim_ms, xlim_ms)
            # Only the left column carries x labels: at 3.5 cm the "200" of one
            # panel and the "-200" of the next collide.
            ax.set_xticks([-xlim_ms, 0, xlim_ms])
            ax.set_yticks([-bp_tick, 0, bp_tick])
            if col == 0:
                ax.set_xticklabels([f"{-xlim_ms:.0f}", "0", ""])
                ax.set_ylabel(r"$\mu$V", labelpad=1)
                ax.text(0.02, 0.97, f"{band[0]:g}–{band[1]:g} Hz",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=7.5, color="0.45")
                ax.set_xlabel("ms", labelpad=1)
            else:
                ax.set_xticklabels(["", "0", f"{xlim_ms:.0f}"])
                ax.set_yticklabels([])
                ax.set_xlabel("ms", labelpad=1)
        if title:
            fig.suptitle(title, fontsize=9, y=1.02)
        fig.subplots_adjust(left=0.22, right=0.995, top=0.87, bottom=0.20)
        if out_stem:
            fig.savefig(out_stem + ".pdf")
            fig.savefig(out_stem + ".png", dpi=400)
        return fig


def padding_figure(x, fs, per, bad, out_stem=None, width_cm=6.0,
                   height_cm=3.5, t0_s=0.0, title=None, overall_frac=None,
                   mark_ms=250.0, lw=0.45):
    """Where the rejected fraction comes from: crossings, then +-1 s padding.

    The equivalent of Chen's Supplemental Fig. S1, and the panel this analysis
    most needs. Each criterion flags well under 1% of samples; the union after
    dilation removes ~40%. Quoting the 40% without showing this invites the
    reading that a criterion is set too loosely, which is not what happened --
    the recording simply contains many separate discharges, and 1 s either side
    of each adds up.

    `per` is {criterion: bool array} BEFORE padding, `bad` the final mask.
    `overall_frac`, if given, is the whole derivation's rejected fraction and is
    printed beside the excerpt's own -- the caller picks a REPRESENTATIVE
    excerpt rather than the worst minute, and the two numbers agreeing is what
    shows the reader that it did.

    `mark_ms` widens each criterion crossing for DISPLAY ONLY. A crossing can be
    a handful of samples; across a minute of recording drawn at 6 cm that is a
    fraction of a pixel and simply does not render, which would make the panel
    argue the opposite of the truth. The widened marks are not the mask -- the
    mask is the bottom row, drawn from `bad` unmodified -- and the caption says
    so.
    """
    import matplotlib.pyplot as plt

    from scipy.ndimage import binary_dilation

    x = np.asarray(x, float)
    t = t0_s + np.arange(len(x)) / fs
    crits = [c for c in CRIT_C if c in per]
    grow = max(1, int(round(mark_ms * 1e-3 * fs / 2)))

    rc = dict(_rc())
    rc.update({"font.size": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
               "axes.labelsize": 9, "axes.titlesize": 9,
               "xtick.major.size": 2, "ytick.major.size": 2,
               "xtick.major.pad": 1.5, "ytick.major.pad": 1.5})
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(width_cm * CM, height_cm * CM))
        gs = gridspec.GridSpec(3, 1, figure=fig,
                               height_ratios=[1.7, 1.5, 0.40], hspace=0.30,
                               left=0.34, right=0.995, top=0.88, bottom=0.20)

        ax = fig.add_subplot(gs[0])
        ax.plot(t, x, color="0.35", lw=lw)
        ax.set_ylabel(r"$\mu$V", labelpad=1)
        ax.set_xlim(t[0], t[-1])
        ax.set_xticklabels([])
        ax.tick_params(bottom=False)
        # the title is set at the end, once the removed fraction is known

        # what each criterion flagged, one row each, unpadded
        ax = fig.add_subplot(gs[1])
        for i, c in enumerate(crits):
            m = binary_dilation(np.asarray(per[c], bool),
                                structure=np.ones(2 * grow + 1, bool))
            ax.fill_between(t, i + 0.10, i + 0.90, where=m, color=CRIT_C[c], lw=0)
            ax.text(t[0] - 0.015 * (t[-1] - t[0]), i + 0.5,
                    CRIT_SHORT.get(c, c), ha="right", va="center",
                    fontsize=7.5, color=CRIT_C[c])
        ax.set_ylim(-0.1, len(crits))
        ax.set_yticks([]); ax.set_xlim(t[0], t[-1]); ax.set_xticklabels([])
        # These stray tick marks read as a sixth, unlabelled criterion row.
        ax.tick_params(bottom=False, left=False)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)

        # the mask that is actually applied
        ax = fig.add_subplot(gs[2])
        ax.fill_between(t, 0, 1, where=np.asarray(bad, bool),
                        color=RIPPLE_C, lw=0)
        ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xlim(t[0], t[-1])
        ax.set_xlabel("Time in recording (s)", labelpad=1)
        ax.tick_params(left=False)
        ax.text(t[0] - 0.012 * (t[-1] - t[0]), 0.5, f"±{PAD_S_LABEL:g} s pad",
                ha="right", va="center", fontsize=8, color=RIPPLE_C)
        for sp in ("left",):
            ax.spines[sp].set_visible(False)
        frac = float(np.mean(np.asarray(bad, bool)))
        note = f"{100 * frac:.0f}% removed"
        if overall_frac is not None:
            note += f" (derivation {100 * overall_frac:.0f}%)"
        head = fig.axes[0]
        head.set_title((title + "  ·  " if title else "") + note,
                       fontsize=8, pad=2, color=RIPPLE_C if not title else None)

        if out_stem:
            fig.savefig(out_stem + ".pdf")
            fig.savefig(out_stem + ".png", dpi=400)
        return fig


PAD_S_LABEL = 1.0          # only for the axis label; the value lives in swr_artifact


def contamination_group_figure(channel_qc, out_stem=None, width_cm=4.0,
                               height_cm=3.5, max_contam=2.0 / 3.0,
                               title=None, histogram=False):
    """Contamination of every derivation in the study, against the 2/3 rule.

    `artifact_figure` panel (b) shows this for one session. The manuscript
    needs it for all of them at once: it is the only place a reader can see
    that the exclusions are a tail rather than a large arbitrary slice, and
    how close the retained derivations came to the line.
    """
    import matplotlib.pyplot as plt

    qc = channel_qc.copy()
    qc = qc[qc["contaminated_frac"].notna()]
    exc = qc["excluded"].fillna(False).astype(bool)
    v = qc["contaminated_frac"].to_numpy(float) * 100.0
    order = np.argsort(v)
    v, exc = v[order], exc.to_numpy()[order]

    rc = dict(_rc())
    rc.update({"font.size": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
               "axes.labelsize": 9, "axes.titlesize": 9,
               "xtick.major.size": 2, "ytick.major.size": 2,
               "xtick.major.pad": 1.5, "ytick.major.pad": 1.5})
    with plt.rc_context(rc):
        # At 4 cm the companion histogram is unreadable, and the ranked curve
        # already shows the distribution's shape; `histogram=True` restores it
        # for a larger version.
        if histogram:
            fig, (ax, axh) = plt.subplots(
                1, 2, figsize=(width_cm * CM, height_cm * CM),
                gridspec_kw=dict(width_ratios=[2.6, 1], wspace=0.30))
        else:
            fig, ax = plt.subplots(figsize=(width_cm * CM, height_cm * CM))
            axh = None

        ax.bar(np.arange(len(v)), v, width=1.0,
               color=np.where(exc, CRIT_C["ied_janca"], HPC_BODY_C),
               linewidth=0)
        ax.axhline(100 * max_contam, color="0.25", ls="--", lw=1.0)
        ax.text(0.02, 100 * max_contam + 2, "excluded above ⅔", fontsize=7.5,
                color="0.25", va="bottom", transform=ax.get_yaxis_transform())
        ax.set_xlabel(f"Derivation (n = {len(v)})", labelpad=1)
        ax.set_ylabel("% rejected", labelpad=1)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.set_xlim(-0.5, len(v) - 0.5)
        ax.set_xticks([0, len(v) - 1])
        ax.set_xticklabels(["1", f"{len(v)}"])
        if not histogram:
            ax.text(0.03, 0.97, f"{int(exc.sum())} excluded", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.5, color=CRIT_C["ied_janca"])

        if axh is None:
            if title:
                ax.set_title(title, fontsize=8, pad=2)
            fig.subplots_adjust(left=0.26, right=0.99, top=0.90, bottom=0.22)
            if out_stem:
                fig.savefig(out_stem + ".pdf")
                fig.savefig(out_stem + ".png", dpi=400)
            return fig

        axh.hist(v[~exc], bins=np.arange(0, 102, 5), orientation="horizontal",
                 color=HPC_BODY_C, edgecolor="w", linewidth=0.4)
        axh.hist(v[exc], bins=np.arange(0, 102, 5), orientation="horizontal",
                 color=CRIT_C["ied_janca"], edgecolor="w", linewidth=0.4)
        axh.axhline(100 * max_contam, color="0.25", ls="--", lw=0.9)
        axh.set_ylim(0, 100); axh.set_yticklabels([])
        axh.set_xlabel("Derivations")
        axh.text(0.95, 0.99,
                 f"{int(exc.sum())} excluded\n{int((~exc).sum())} analysed",
                 transform=axh.transAxes, ha="right", va="top",
                 fontsize=FS_TICK - 1)
        if title:
            fig.suptitle(title, fontsize=FS_TITLE, y=1.02)
        if out_stem:
            fig.savefig(out_stem + ".pdf", bbox_inches="tight")
            fig.savefig(out_stem + ".png", dpi=300, bbox_inches="tight")
        return fig


# ================================================= the adaptive notch, shown ==
NOTCH_BEFORE_C = "#b9b9b9"
NOTCH_AFTER_C = "#23677E"


def notch_gallery_figure(entries, out_stem=None, ncol=5, panel_cm=(4.0, 3.4),
                         band=RIPPLE_BAND, fmax=210.0, title=None):
    """A grid of before/after spectra: why the notch is adaptive, not blanket.

    Each panel is one session: every derivation's PSD before the notch in grey,
    after it in colour, with the ripple band shaded and the measured
    peak-to-flank ratio printed at each line harmonic. Sessions where nothing
    was notched are the point of the figure, not filler -- they are the evidence
    that 120 Hz, which sits on the upper edge of the ripple band, is only ever
    touched where a peak was actually measured.

    `entries` is a list of dicts from `notch_psd.npz`, each with freq,
    psd_before, psd_after, recording_site, session, ratios, ratio_threshold and
    notch_applied_hz. Ordering is the caller's business -- group by site.

    Panels are `panel_cm` (w, h), default 4 x 3 cm, at 9 pt: the size these are
    actually printed at, so nothing has to be rescaled afterwards.
    """
    import matplotlib.pyplot as plt

    n = len(entries)
    if not n:
        return None
    ncol = min(ncol, n)
    nrow = int(np.ceil(n / ncol))
    w_in = ncol * panel_cm[0] * CM
    h_in = nrow * panel_cm[1] * CM

    rc = dict(_rc())
    rc.update({"font.size": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
               "axes.labelsize": 9, "axes.titlesize": 9})
    with plt.rc_context(rc):
        fig, axes = plt.subplots(nrow, ncol, figsize=(w_in, h_in),
                                 squeeze=False, sharex=True)
        for ax in axes.ravel():
            ax.set_axis_off()

        for k, e in enumerate(entries):
            ax = axes.ravel()[k]
            ax.set_axis_on()
            f = np.asarray(e["freq"], float)
            keep = f <= fmax
            pb = np.asarray(e["psd_before"], float)[:, keep]
            pa = np.asarray(e["psd_after"], float)[:, keep]
            for i in range(len(pb)):
                ax.semilogy(f[keep], pb[i], color=NOTCH_BEFORE_C, lw=0.9)
            for i in range(len(pa)):
                ax.semilogy(f[keep], pa[i], color=NOTCH_AFTER_C, lw=0.7)
            ax.axvspan(band[0], band[1], color=RIPPLE_C, alpha=0.10, lw=0)

            # A notch cuts six orders of magnitude out of one bin. Left to
            # autoscale, that single bin sets the y-range and squashes the whole
            # spectrum -- including the line peak the panel exists to show -- into
            # a flat smear. Bound the axis by the SPECTRUM instead, away from the
            # harmonics, and let the notch run off the bottom: how deep it goes is
            # not the point, that it was applied is.
            core = np.ones(keep.sum(), bool)
            for f0 in (60.0, 120.0, 180.0):
                core &= np.abs(f[keep] - f0) > 2.5
            if core.any():
                lo = max(np.percentile(pa[:, core], 1) * 0.25, 1e-12)
                hi = max(pb.max(), pa.max()) * 3.0
                ax.set_ylim(lo, hi)

            applied = set(float(v) for v in np.atleast_1d(e["notch_applied_hz"]))
            ratios = np.asarray(e["ratios"], float).reshape(-1, 2)
            # the notch is decided per derivation, so the panel reports the
            # worst derivation's ratio, not a session average: a peak on one
            # contact is the thing the criterion is meant to catch
            per = e.get("ratios_per_pair")
            for k, (f0, r) in enumerate(ratios):
                if f0 > fmax:
                    continue
                if per is not None and np.size(per):
                    row = np.asarray(per, float)
                    if row.ndim == 2 and k < len(row):
                        vals = row[k][np.isfinite(row[k])]
                        if vals.size:
                            r = float(np.max(vals))
                hit = f0 in applied
                # No guide line at the harmonic: it would be drawn in the same
                # grey as the BEFORE spectrum, and the whole point of the panel
                # is the grey spike standing at that frequency. The x-ticks are
                # already at 60/120/180, which is guide enough.
                # The ratio is the number that decided it, so it is the number
                # printed -- a tick mark alone would not explain the decision.
                ax.text(f0, 1.02, f"{r:,.0f}" if r >= 10 else f"{r:.1f}",
                        transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=7,
                        color="#a30d6c" if hit else "0.55")

            site = str(e.get("recording_site", "")).title()
            sess = int(e.get("session", -1))
            n_notched = e.get("n_notched")
            n_pairs = int(e.get("n_pairs", 0) or 0)
            if applied and n_notched is not None and n_pairs:
                nn = {float(f): int(n) for f, n in
                      zip([60.0, 120.0, 180.0], np.atleast_1d(n_notched))}
                note = ", ".join(f"{v:.0f}" for v in sorted(applied)) + " Hz"
                k_ = max(nn.get(v, 0) for v in applied)
                note += f" ({k_}/{n_pairs})"
            elif applied:
                note = ", ".join(f"{v:.0f}" for v in sorted(applied)) + " Hz"
            else:
                note = "none needed"
            ax.set_title(f"s{sess:02d} · {site}\nnotch: {note}",
                         fontsize=8, pad=12, linespacing=1.25)
            ax.set_xlim(0, fmax)
            ax.set_xticks([0, 60, 120, 180])
            if k % ncol == 0:
                ax.set_ylabel(r"PSD ($\mu$V$^2$/Hz)", fontsize=8)
            if k >= n - ncol:
                ax.set_xlabel("Frequency (Hz)", fontsize=8)

        from matplotlib.lines import Line2D
        fig.legend(handles=[
            Line2D([], [], color=NOTCH_BEFORE_C, lw=1.2, label="before notch"),
            Line2D([], [], color=NOTCH_AFTER_C, lw=1.2, label="after notch"),
            Line2D([], [], color=RIPPLE_C, lw=4, alpha=0.25,
                   label=f"ripple band {band[0]:g}–{band[1]:g} Hz")],
            loc="lower center", ncol=3, frameon=False, fontsize=8,
            bbox_to_anchor=(0.5, 0.0))
        if title:
            fig.suptitle(title, fontsize=10, y=1.0)
        # Explicit, not tight_layout: a figure-level legend is invisible to it,
        # so it kept shrinking the axes to make room for a title it had already
        # accounted for. Panel geometry is fixed by `panel_cm` and must survive.
        legend_frac = 0.9 / (nrow * panel_cm[1])
        fig.subplots_adjust(left=0.085, right=0.995, top=0.80,
                            bottom=0.16 + legend_frac, wspace=0.42, hspace=0.75)
        if out_stem:
            fig.savefig(out_stem + ".pdf", bbox_inches="tight")
            fig.savefig(out_stem + ".png", dpi=300, bbox_inches="tight")
        return fig


# ============================================== ripple attributes, pooled ==
ATTR_SPECS = [
    ("rate_hz", "Ripple rate (Hz)", (0.17, 0.24), "per derivation", 1.0),
    ("duration_ms", "Duration (ms)", (50.0, 90.0), "per event", 1.0),
    ("peak_freq_hz", "Peak frequency (Hz)", RIPPLE_BAND, "per event", 1.0),
    ("amp_peak_uv", r"Amplitude ($\mu$V)", None, "per event", 1.0),
]


def ripple_attributes_figure(events, channel_qc, out_stem=None, width_cm=17.0,
                             height_cm=5.4, title=None, ref_label="Chen et al."):
    """The four attributes Chen report, pooled across the study.

    Rate is one value per DERIVATION; duration, frequency and amplitude are one
    value per EVENT. Mixing the two would let a single high-rate derivation
    dominate a distribution that is supposed to describe ripples, so each panel
    says which unit it is counting.

    Shaded bands are the published reference ranges. They are drawn to be
    checked against, not to be hit: the detector's thresholds were set by the
    1/f surrogate control, and agreement with the reference is a result rather
    than a target.
    """
    import matplotlib.pyplot as plt

    ev = events.copy()
    if "duration_s" in ev.columns and "duration_ms" not in ev.columns:
        ev["duration_ms"] = ev["duration_s"] * 1000.0
    qc = channel_qc[~channel_qc["excluded"].fillna(False).astype(bool)]

    data = {"rate_hz": qc["rate_hz"].dropna(),
            "duration_ms": ev["duration_ms"].dropna(),
            "peak_freq_hz": ev["peak_freq_hz"].dropna(),
            "amp_peak_uv": ev["amp_peak_uv"].dropna()}

    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, 4, figsize=(width_cm * CM, height_cm * CM))
        for ax, (key, label, ref, unit, _) in zip(axes, ATTR_SPECS):
            v = data[key]
            med = float(np.median(v))
            lo, hi = np.percentile(v, [25, 75])

            # Clip the display, never the statistics: the medians and IQRs above
            # are computed on everything, but a 500 ms duration ceiling and a
            # long amplitude tail leave most of the axis empty and the mode
            # unreadable.
            shown, clipped = v, 0
            if key in ("duration_ms", "amp_peak_uv"):
                cut = np.percentile(v, 99.5)
                shown = v[v <= cut]
                clipped = len(v) - len(shown)
            if key == "peak_freq_hz":
                # peak frequency is resolved on the 2.5 Hz spectral grid, so
                # arbitrary bins alias into a comb. Bin ON the grid.
                step = 2.5
                bins = np.arange(RIPPLE_BAND[0] - step / 2,
                                 RIPPLE_BAND[1] + step, step)
            else:
                bins = np.histogram_bin_edges(shown, bins=36)

            if ref is not None:
                ax.axvspan(ref[0], ref[1], color=RIPPLE_C, alpha=0.13, lw=0,
                           zorder=0)
            ax.hist(shown, bins=bins, color=HPC_BODY_C, edgecolor="w",
                    linewidth=0.3, zorder=2)
            ax.axvline(med, color=OBS_LINE_C, lw=1.2, zorder=3)
            ax.set_xlabel(label)
            ax.set_ylabel("Derivations" if unit == "per derivation" else "Events")
            ax.set_title(f"{med:.3g}  ({lo:.3g}–{hi:.3g})", fontsize=FS_TICK,
                         pad=4)
            note = f"n = {len(v):,}"
            if clipped:
                note += f"\n({clipped} beyond axis)"
            ax.text(0.97, 0.97, note, transform=ax.transAxes, ha="right",
                    va="top", fontsize=FS_TICK - 2, color="0.4",
                    linespacing=1.3)

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        fig.legend(handles=[
            Patch(facecolor=RIPPLE_C, alpha=0.25, label=f"{ref_label} range"),
            Line2D([], [], color=OBS_LINE_C, lw=1.4, label="median")],
            loc="lower center", ncol=2, frameon=False, fontsize=FS_TICK,
            bbox_to_anchor=(0.5, -0.06))
        if title:
            fig.suptitle(title, fontsize=FS_TITLE, y=1.03)
        fig.tight_layout(rect=(0, 0.06, 1, 1.0))
        if out_stem:
            fig.savefig(out_stem + ".pdf", bbox_inches="tight")
            fig.savefig(out_stem + ".png", dpi=300, bbox_inches="tight")
        return fig
