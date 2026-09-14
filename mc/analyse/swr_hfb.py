#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-frequency broadband (HFB) and band envelopes for every derivation.

The hippocampal branch of this pipeline asks *when* ripples happen. This module
exists for the other half of the question -- what the cortex does around them --
following He et al. 2026 (Nat Neurosci), who show that mPFC HFB power is
elevated around hippocampal ripples and that the mPFC representation is updated
across the ripple peak.

HFB is the standard proxy for local population spiking (Manning 2009, Ray &
Maunsell 2011): narrow sub-bands, Hilbert amplitude, **log-transformed and
z-scored per sub-band before averaging**. The log and the per-band z-score are
not cosmetic -- band power is strongly right-skewed and falls off as 1/f, so a
plain average across 70-150 Hz is dominated by the lowest sub-band and by
outliers. This is the same recipe `swr_artifact.criterion_broadband_power`
already uses for the 1-60 Hz criterion.

Two things this module does that a textbook HFB implementation does not:

1. **It drops sub-bands sitting on a notched line harmonic.** 120 Hz is inside
   any sensible HFB band, and this pipeline's notch is adaptive *per
   derivation* -- some derivations are notched at 120 Hz and some are not. A
   fixed band definition would therefore mean different things on different
   contacts of the same session. `notch_applied_pairs` in the extraction
   `meta.json` says which, so the drop is a lookup.

2. **It z-scores on artifact-free samples only.** Otherwise the few per cent of
   samples the artifact criteria flag -- which are by construction the largest
   excursions in the recording -- set the SD, and the z-scored HFB of a clean
   contact and a contaminated one are not on the same scale.

Output is continuous, at `OUT_FS` (100 Hz), and nothing is epoched here. That is
deliberate: epoching on the cluster would freeze the window choices, and the
whole point of bringing this home is that the windows are still open questions.

Stored as **float16** (`STORE_DTYPE`). These are z-scores in roughly [-6, 6],
where float16 resolves ~0.001 -- far finer than the quantity is meaningful to,
and it halves a store that would otherwise be several GB across the cohort and
would not survive being copied to a laptop. Cast to float32 on load before
doing arithmetic; float16 accumulates visible error over long sums.

@author: Svenja Kuchenhoff
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, resample_poly
from scipy.ndimage import uniform_filter1d

OUT_FS = 100.0               # 10 ms bins, as He et al. use for HFB
STORE_DTYPE = np.float16     # see module docstring: z-scores, so ample
HFB_BAND = (70.0, 150.0)
HFB_SUB_BW = 10.0            # sub-band width within HFB_BAND
NOTCH_GUARD_HZ = 5.0         # a sub-band this close to a notched line is dropped
LINE_HARMONICS = (60.0, 120.0, 180.0)

# Envelopes carried alongside HFB. Each is cheap (~1.5 MB/derivation/hour) and
# each opens a class of hypothesis that would otherwise need another cluster
# run: `ripple` for band-limited power without the detector's thresholds,
# `theta`/`theta_phase` for phase-amplitude coupling and ripple-theta timing.
# Add a band here rather than writing a new stage.
BANDS = {
    "ripple": (80.0, 120.0),
    "theta": (4.0, 8.0),
    "beta": (13.0, 30.0),
}
PHASE_BANDS = ("theta",)     # also stored as instantaneous phase, in radians


def _bandpass(x, fs, lo, hi, order=4):
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band', output='sos')
    return sosfiltfilt(sos, x)


def _decimate(y, fs, out_fs=OUT_FS):
    """Anti-aliased downsample to out_fs. The envelopes are smooth (their
    bandwidth is that of the sub-band, <= 10 Hz here), so 100 Hz is well above
    Nyquist for them."""
    from math import gcd
    g = gcd(int(round(fs)), int(round(out_fs)))
    return resample_poly(y, int(round(out_fs)) // g, int(round(fs)) // g, axis=-1)


def sub_bands(band=HFB_BAND, bw=HFB_SUB_BW, drop_hz=(), guard=NOTCH_GUARD_HZ):
    """Sub-band edges covering `band`, minus any overlapping a dropped line.

    Returns (kept, dropped) as lists of (lo, hi).
    """
    edges = np.arange(band[0], band[1] + 1e-9, bw)
    kept, dropped = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if any((lo - guard) <= f <= (hi + guard) for f in drop_hz):
            dropped.append((float(lo), float(hi)))
        else:
            kept.append((float(lo), float(hi)))
    return kept, dropped


def hfb_amplitude(x, fs, clean=None, band=HFB_BAND, bw=HFB_SUB_BW,
                  drop_hz=(), out_fs=OUT_FS):
    """Log, per-sub-band z-scored, sub-band-averaged HFB amplitude at `out_fs`.

    `clean` is the artifact-free boolean mask at the INPUT rate; the z-score
    statistics are taken over those samples only (see module docstring).
    Returns (hfb, info).
    """
    x = np.asarray(x, float)
    kept, dropped = sub_bands(band, bw, drop_hz)
    if not kept:
        raise ValueError(f"every sub-band of {band} was dropped by {drop_hz}")

    if clean is None:
        clean = np.ones(x.shape[-1], bool)
    clean = np.asarray(clean, bool)

    total = np.zeros(x.shape[-1])
    for lo, hi in kept:
        env = np.abs(hilbert(_bandpass(x, fs, lo, hi)))
        le = np.log(np.maximum(env, 1e-30))
        ref = le[clean] if clean.any() else le
        sd = float(np.std(ref))
        total += (le - float(np.mean(ref))) / (sd if sd > 0 else 1.0)
    total /= len(kept)

    info = {"band": list(band), "sub_bw": bw,
            "sub_bands_kept": kept, "sub_bands_dropped": dropped,
            "n_sub_bands": len(kept), "out_fs": out_fs,
            "z_scored_on": "artifact-free samples only",
            "clean_frac_used": float(clean.mean())}
    return _decimate(total, fs, out_fs).astype(STORE_DTYPE), info


def band_envelope(x, fs, lo, hi, clean=None, out_fs=OUT_FS, smooth_ms=20.0):
    """Log, z-scored Hilbert amplitude of one band, at `out_fs`.

    Same normalisation as `hfb_amplitude` so the arrays are comparable, with a
    short moving average first because a single narrow band's envelope is much
    noisier than an eight-band average.
    """
    x = np.asarray(x, float)
    env = np.abs(hilbert(_bandpass(x, fs, lo, hi)))
    w = max(1, int(round(smooth_ms * fs / 1000.0)))
    env = uniform_filter1d(env, w, mode='nearest')
    le = np.log(np.maximum(env, 1e-30))
    if clean is None:
        clean = np.ones(x.shape[-1], bool)
    clean = np.asarray(clean, bool)
    ref = le[clean] if clean.any() else le
    sd = float(np.std(ref))
    z = (le - float(np.mean(ref))) / (sd if sd > 0 else 1.0)
    return _decimate(z, fs, out_fs).astype(STORE_DTYPE)


def band_phase(x, fs, lo, hi, out_fs=OUT_FS):
    """Instantaneous phase of one band, in radians, at `out_fs`.

    Decimated on the unit circle (cos/sin separately) rather than on the
    wrapped angle -- decimating the angle itself filters across the +-pi
    discontinuity and produces garbage wherever the phase wraps.
    """
    a = hilbert(_bandpass(np.asarray(x, float), fs, lo, hi))
    c = _decimate(np.real(a) / np.abs(a), fs, out_fs)
    s = _decimate(np.imag(a) / np.abs(a), fs, out_fs)
    return np.arctan2(s, c).astype(STORE_DTYPE)


def notched_harmonics_for_pair(meta, pair_id, harmonics=LINE_HARMONICS):
    """Which line harmonics were notched on THIS derivation.

    `notch_applied_pairs` maps each harmonic to the list of pair_ids notched at
    it. Older extractions wrote `notch_applied_hz` as a ratio rather than a
    count and carry no per-pair list; there the safe reading is that any
    harmonic that was notched at all may have been notched here, so all
    reported harmonics are dropped. Conservative in the right direction: it
    removes a sub-band that might be clean, never keeps one that is notched.
    """
    per_pair = meta.get("notch_applied_pairs")
    if isinstance(per_pair, dict) and per_pair:
        out = [float(k) for k, v in per_pair.items()
               if isinstance(v, (list, tuple)) and pair_id in v]
        return sorted(out)
    applied = meta.get("notch_applied_hz") or {}
    return sorted(float(k) for k in applied)
