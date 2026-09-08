#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuous LFP preprocessing for the SWR pipeline.

Loads a whole session's hippocampal bipolar derivations onto the behavioural
clock: raw -> 1000 Hz -> bipolar -> notch, with the block structure resolved
from the recordings themselves.

Three decisions here differ from the previous pipeline, each for a measured
reason:

1. **Continuous, not per-snippet.** The detection threshold has to be estimated
   over the whole artifact-free session (see swr_detect). Cropping first makes
   the threshold snippet-dependent, which normalises away the very rate
   differences the analysis is about.

2. **`resample_poly`, not `resample`.** `scipy.signal.resample` is FFT-based and
   assumes periodicity, so it wraps the end of the recording into the
   beginning. Applied per snippet, as `preprocess_LFP.py:106` did, it rings at
   every crop boundary. Polyphase decimation uses a proper anti-alias FIR and
   does not wrap.

3. **1000 Hz, not 500.** 500 Hz makes Chen's >250 Hz artifact criterion and the
   120-200 Hz spectral rejection impossible.

Block alignment
---------------
The behavioural clock is **cumulative file duration**: block *k* starts at
`sum(durations of files 0..k-1)`. Verified on all 14 multi-block sessions with
local raw data -- every behavioural event lands inside its file.

Do NOT align by wall clock. The NSx header `TimeOrigin` is parseable, but it
records real elapsed time between recordings (for s18: 1943 s and 12459 s apart
versus file durations of 560 s and 4040 s). The behaviour was timestamped
against the concatenated recording, so wall-clock anchoring would displace
block 2+ by tens of minutes.

Because the clock is the concatenation, the resampled blocks laid end to end
ARE the session timeline: `sample = round(t_session * fs)`.

@author: Svenja Kuchenhoff
"""

import os
import re
import struct
import datetime

import numpy as np
from scipy.signal import resample_poly, iirnotch, sosfiltfilt, tf2sos

import mc.analyse.swr_io as swr_io


TARGET_FS = 1000.0
LINE_FREQS = (60.0, 120.0, 180.0)      # US sites; 120 sits on the ripple band edge
NOTCH_Q = 30.0
CHUNK_S = 600.0                        # per-block read chunk
CHUNK_PAD_S = 5.0                      # discarded either side to avoid edge effects


# =============================================================================
# NEURALYNX .ncs  (neo cannot read these files)
# =============================================================================
#
# neo's NeuralynxRawIO.parse_header() raises
#   TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'
# at neuralynxrawio.py:505 because global_t_start/global_t_stop are None.
# The format is simple enough to read directly, and doing so also exposes the
# per-record microsecond timestamps.

NCS_HEADER_BYTES = 16 * 1024
NCS_REC = np.dtype([('ts', '<u8'), ('ch', '<u4'), ('fs', '<u4'),
                    ('nvalid', '<u4'), ('samp', '<i2', 512)])


def ncs_header(path):
    """Parse the 16 KB ASCII header into a dict."""
    with open(path, 'rb') as f:
        raw = f.read(NCS_HEADER_BYTES).decode('latin-1', errors='replace')
    out = {}
    for line in raw.split('\n'):
        line = line.strip().strip('\x00').strip()
        if line.startswith('-'):
            parts = line[1:].split(None, 1)
            out[parts[0]] = parts[1].strip() if len(parts) > 1 else ''
    return out


def read_ncs(path, t0_s=None, t1_s=None):
    """Return (signal float32, fs, t_start_s). Optionally crop by seconds
    relative to the file's own first timestamp."""
    size = os.path.getsize(path)
    n_rec = (size - NCS_HEADER_BYTES) // NCS_REC.itemsize
    if n_rec <= 0:
        return np.zeros(0, np.float32), np.nan, np.nan

    recs = np.memmap(path, dtype=NCS_REC, mode='r',
                     offset=NCS_HEADER_BYTES, shape=(int(n_rec),))
    fs = float(recs['fs'][0])
    t_start = float(recs['ts'][0]) / 1e6

    lo, hi = 0, int(n_rec)
    if t0_s is not None:
        lo = max(0, int(np.floor(t0_s * fs / 512.0)))
    if t1_s is not None:
        hi = min(int(n_rec), int(np.ceil(t1_s * fs / 512.0)) + 1)
    if hi <= lo:
        return np.zeros(0, np.float32), fs, t_start

    block = recs[lo:hi]
    sig = block['samp'].reshape(-1).astype(np.float32)

    hdr = ncs_header(path)
    try:                                    # convert AD units to microvolts
        sig *= float(hdr.get('ADBitVolts', 1.0)) * 1e6
    except (TypeError, ValueError):
        pass
    return sig, fs, t_start + lo * 512.0 / fs


def nsx_time_origin(path):
    """NSx 2.2/2.3 TimeOrigin (Windows SYSTEMTIME at byte 294). Returns None
    for the BRSMPGRP variant. Recorded for provenance only -- it must NOT be
    used for block alignment (see module docstring)."""
    with open(path, 'rb') as f:
        head = f.read(320)
    if head[:8] != b'NEURALCD':
        return None
    y, mo, _dw, d, h, mi, s, ms = struct.unpack('<8H', head[294:310])
    try:
        return datetime.datetime(y, mo, d, h, mi, s, ms * 1000)
    except ValueError:
        return None


# =============================================================================
# BLOCK STRUCTURE
# =============================================================================

def _blackrock_reader(path, nsx):
    """BlackrockIO for one .nsX, tolerating a .nev that disagrees with it.

    neo cross-checks segment counts between the .nsx and its companion .nev and
    refuses to open the file when they differ -- s52 has 3 segments in the .nev
    and 2 in the .ns2, which cost the whole session. The .nev carries spike and
    event data that this pipeline never reads, so pointing the reader at a .nev
    that does not exist skips the check and the LFP loads normally.
    """
    import neo
    try:
        return neo.io.BlackrockIO(filename=path, nsx_to_load=int(nsx))
    except Exception:
        return neo.io.BlackrockIO(filename=path, nsx_to_load=int(nsx),
                                  nev_override=path + "__absent__")


def blackrock_block_info(path, nsx):
    """(duration_s, fs, n_channels, seg_index) of the real recording segment.

    Blackrock files carry a ~2 s stub segment alongside the recording; the
    longest segment is the real one. Picking it independently reproduces the
    YAML `segment` field wherever that field is populated.
    """
    reader = _blackrock_reader(path, nsx)
    best = (0.0, np.nan, 0, None)
    for bi in range(reader.block_count()):
        for si in range(reader.segment_count(bi)):
            try:
                seg = reader.read_segment(block_index=bi, seg_index=si, lazy=True)
            except Exception:
                continue
            for a in seg.analogsignals:
                fs = float(a.sampling_rate.magnitude)
                dur = float(a.shape[0]) / fs
                if dur > best[0]:
                    best = (dur, fs, int(a.shape[1]), si)
    return best


CLOCK_TOLERANCE_S = 5.0        # shared with swr_extract_continuous

# Sessions where the recording durations alone do NOT determine which files
# carry the task, resolved from evidence outside the timings. Every entry needs
# its reason recorded here -- these are the only places a human choice enters
# the block structure, and `resolve_run` refuses rather than guess without one.
RUN_OVERRIDE = {
    # s08 and s09 are the same patient (YEK) and their directories hold the
    # SAME four .ns3 files. The filenames pair them: EMU-030 blk-01 with
    # EMU-031 blk-02 (run-02) is one session, and the remaining two the other.
    # s09's behaviour fits files 1+2 with -2.0s of trailing truncation, which
    # leaves 3+4 for s08 -- and 3+4 is the only length-2 run that fits s08
    # anyway, so the two agree. A recording cannot host two sessions' tasks.
    9:  ((0, 1), "EMU-030 blk-01 + EMU-031 blk-02 are the matched pair; "
                 "the other two files carry s08"),
    # datafile202302_b001.ns2 is block 1 of one behavioural block; _b002 is a
    # separate run of the same patient. Duration alone cannot tell them apart
    # (both are long enough), the b00N suffix can.
    47: ((0,), "datafile202302_b001 is the task block; _b002 is a separate run"),
}


def _ncs_group_info(group):
    """(fs, duration_s, n_files) for one .ncs suffix group.

    Reading these off `group[0]` -- the alphabetically first file -- is wrong
    twice over at UCLA. 2 kHz macros and 32 kHz micro-wires share a group, so
    the first name decides the sampling rate; and header-only stubs sort early,
    so for s40 the probe was `A1.ncs`, which contains no data at all. That
    reported the task block as 1925 s at 32 kHz when the macro channels say
    2287 s at 2 kHz, and the session looked short of its behaviour.

    The MODAL sampling rate is taken instead -- macros always outnumber micros
    -- and the duration is the longest file at that rate.
    """
    rates, durs = [], []
    for f in group:
        n_rec = (os.path.getsize(f) - NCS_HEADER_BYTES) // NCS_REC.itemsize
        fs = float(ncs_header(f).get('SamplingFrequency', np.nan))
        if np.isfinite(fs) and fs > 0 and n_rec > 0:
            rates.append(fs)
            durs.append(n_rec * 512.0 / fs)
    if not rates:
        return np.nan, np.nan, 0
    rates = np.asarray(rates)
    modal = float(np.bincount(rates.astype(int)).argmax())
    at_modal = [d for r, d in zip(rates, durs) if r == modal]
    return modal, float(max(at_modal)), len(rates)


def _nsx_number(path, cfg):
    """Which .nsX stream to read, taken from the file on disk.

    `cfg['LFP_file_format']` is the string 'ncs' for the two UCLA sessions that
    turned out to be Blackrock (s50, s51). `discover_raw_files` notices and
    rebinds its own local `fmt`, but that never reaches here, so this used to
    do `int('ncs')` and raise -- losing both sessions to a config typo rather
    than to anything about the data.
    """
    fmt = (cfg or {}).get('LFP_file_format')
    if fmt in (2, 3, '2', '3'):
        return int(fmt)
    ext = os.path.splitext(path)[1]
    return int(ext[-1]) if ext[:-1] == '.ns' and ext[-1].isdigit() else 3


def resolve_run(durations, bt, tol=CLOCK_TOLERANCE_S, override=None):
    """Which contiguous run of recordings carries this session's behaviour.

    A session directory does not only hold that session's task. It can hold
    another day's session (s08 and s09 share four files, and s08's task is
    recordings 3+4), an intervening run of a different task (s21's recording 2
    is `task-T3`), an aborted start (s40's 107 s and 143 s groups, s47's second
    file) or a 4.3 s stub (s59's recording 3). Pairing `files[k]` with block
    `k` and refusing to run when the counts differ lost seven sessions to that.

    Every contiguous run is tried as the session timeline and the behaviour is
    tested against the run AS A WHOLE, on the cumulative-duration clock the
    module docstring establishes:

        head = min(beh_start)                          slack before the task
        tail = sum(durations in run) - max(beh_end)    slack after it

    Testing the run rather than block k against file k is also what recovers
    s19: its block 2 begins 12.3 s before file 2 does, which the per-file test
    calls a fatal misalignment, while on the concatenation the session fits
    with 17.4 s and 42.6 s to spare.

    NOTHING IS GUESSED. `head` is min(beh_start) whatever run is chosen, so the
    fit test only really asserts "the concatenation is long enough" -- a weak
    constraint that several runs can satisfy. Measured over the 53 sessions with
    known durations, exactly one run fits for 49 of them; ambiguity is real for
    s08, s09, s40 and s47 only. Rather than pick a winner on a margin, those are
    resolved by an explicit constraint or refused:

      1. runs that fit, within `tol`
      2. of those, prefer len(run) == n_behavioural_blocks -- one recording per
         block is what an uninterrupted acquisition produces. This alone makes
         s08 and s40 unique
      3. still more than one -> AMBIGUOUS. No run is returned. The caller must
         supply `override`, and the reason is recorded in RUN_OVERRIDE below

    Returns (run, head_s, tail_s, status) with status one of 'unique',
    'by_block_count', 'override', 'truncated', 'AMBIGUOUS'. 'truncated' means no
    run is long enough -- a short recording, not a mapping error, carried as a
    partial. 'AMBIGUOUS' returns an empty run: refusing is the honest outcome
    when the timings genuinely do not determine the answer.

    Verified against the 48 sessions that already read correctly -- the run is
    (0..n_blocks-1) for every one of them, so this is a no-op there.
    """
    d = np.asarray(durations, float)
    beh_start, beh_end = float(bt.beh_start_s.min()), float(bt.beh_end_s.max())
    nb = len(bt)

    fitting, best_any = [], None
    for a in range(len(d)):
        for b in range(a, len(d)):
            run = tuple(range(a, b + 1))
            if not np.isfinite(d[list(run)]).all():
                continue
            total = float(d[list(run)].sum())
            head, tail = beh_start, total - beh_end
            if best_any is None or min(head, tail) > best_any[0]:
                best_any = (min(head, tail), run, head, tail)
            if min(head, tail) > -tol:
                fitting.append((run, head, tail))

    if override is not None:
        run = tuple(override)
        total = float(d[list(run)].sum())
        return run, beh_start, total - beh_end, 'override'
    if not fitting:
        if best_any is None:
            return (), np.nan, np.nan, 'AMBIGUOUS'
        _, run, head, tail = best_any
        return run, head, tail, 'truncated'
    if len(fitting) == 1:
        run, head, tail = fitting[0]
        return run, head, tail, 'unique'

    by_count = [f for f in fitting if len(f[0]) == nb]
    if len(by_count) == 1:
        run, head, tail = by_count[0]
        return run, head, tail, 'by_block_count'
    return (), np.nan, np.nan, 'AMBIGUOUS'


def session_block_table(session, data_root=None):
    """The recordings carrying this session, with their session-clock offsets.

    One row per recording IN THE RESOLVED RUN (see `resolve_run`), not one per
    behavioural block -- those are not the same thing whenever the directory
    holds recordings that are not part of the task.

    offset_k = cumulative duration of the run's files 0..k-1 (module docstring).
    `head_margin_s` sits on the first row and `tail_margin_s` on the last, both
    session-level, so `np.nanmin` over either column still gives the worst
    margin and `swr_audit_sessions` needs no change.

    `rec_index` indexes back into `discover_raw_files`, so the caller loads the
    right files without re-deriving the mapping.
    """
    import pandas as pd
    data_root = data_root or swr_io.get_data_root()
    cfg = swr_io.session_config(session, data_root=data_root)
    files, kind, warn = swr_io.discover_raw_files(session, cfg, data_root=data_root)
    beh = swr_io.load_behaviour(session, data_root=data_root)
    bt = swr_io.block_table(beh)

    rows = []
    for k, f in enumerate(files):
        if kind == 'blackrock':
            try:
                dur, fs, nch, seg = blackrock_block_info(f, _nsx_number(f, cfg))
            except Exception as e:
                warn.append(f"{os.path.basename(f)}: {type(e).__name__}: {e}")
                dur, fs, nch, seg = np.nan, np.nan, 0, None
            name, t_origin = os.path.basename(f), nsx_time_origin(f)
        else:
            fs, dur, nch = _ncs_group_info(f)
            seg = None
            name = f"{len(f)} .ncs files"
            t_origin = ncs_header(f[0]).get('TimeCreated') if f else None
        rows.append({"rec_index": k, "file": name, "seg_index": seg,
                     "fs_raw": fs, "n_channels": nch, "duration_s": dur,
                     "time_origin": t_origin})

    full = pd.DataFrame(rows)
    if full.empty:
        return full, bt, warn

    ov = RUN_OVERRIDE.get(int(session))
    run, head, tail, status = resolve_run(full.duration_s.values, bt,
                                          override=ov[0] if ov else None)
    if status == 'AMBIGUOUS':
        warn.append(
            "block structure is AMBIGUOUS: more than one run of recordings is "
            "long enough to hold the behaviour and none has one recording per "
            "block. The durations do not determine the answer -- add an entry "
            "to swr_preproc.RUN_OVERRIDE with the evidence that does.")
        return full.iloc[0:0], bt, warn
    if not len(run):
        return full.iloc[0:0], bt, warn + ["no readable recording"]
    if status == 'override':
        warn.append(f"run {'+'.join(str(i + 1) for i in run)} set by "
                    f"RUN_OVERRIDE: {ov[1]}")
    elif run != tuple(range(len(bt))):
        warn.append(f"recordings {'+'.join(str(i + 1) for i in run)} carry the "
                    f"behaviour, not 1..{len(bt)} ({status})")

    df = full.iloc[list(run)].reset_index(drop=True)
    df.insert(0, "block", np.arange(1, len(df) + 1))
    df["offset_s"] = np.concatenate([[0.0], np.cumsum(df["duration_s"])[:-1]])
    for col in ("beh_start_s", "beh_end_s", "head_margin_s", "tail_margin_s"):
        df[col] = np.nan
    df.loc[0, "beh_start_s"] = float(bt.beh_start_s.min())
    df.loc[len(df) - 1, "beh_end_s"] = float(bt.beh_end_s.max())
    df.loc[0, "head_margin_s"] = head
    df.loc[len(df) - 1, "tail_margin_s"] = tail
    return df, bt, warn


# =============================================================================
# FILTERING
# =============================================================================

NOTCH_RATIO_THRESHOLD = 2.0    # notch a harmonic only if it exceeds this


def measure_line_noise(x, fs, freqs=LINE_FREQS, halfwidth=1.5, side=(4.0, 12.0)):
    """Peak-to-flank power ratio at each line harmonic. 1.0 = no peak.

    Measured across sites: Baylor is essentially clean (60 Hz ratio ~1.1),
    UCLA has strong referential line noise that bipolar removes (60.9 -> 1.14),
    and Utah is heavily contaminated even after bipolar (1.1e8 -> 5.5e4).
    """
    from scipy.signal import welch
    f, P = welch(np.asarray(x, float), fs=fs,
                 nperseg=min(4096, np.shape(x)[-1]), axis=-1)
    out = {}
    for f0 in freqs:
        if f0 >= fs / 2.0:
            continue
        inb = np.abs(f - f0) <= halfwidth
        sd = (np.abs(f - f0) >= side[0]) & (np.abs(f - f0) <= side[1])
        if not inb.any() or not sd.any():
            continue
        out[float(f0)] = float(np.max(P[..., inb], axis=-1).mean()
                               / max(np.median(P[..., sd]), 1e-30))
    return out


def notch_filter(x, fs, freqs=LINE_FREQS, q=NOTCH_Q,
                 ratio_threshold=NOTCH_RATIO_THRESHOLD):
    """Zero-phase narrow notches, applied ONLY where line noise is present.

    Returns (filtered, applied) where `applied` maps frequency -> measured
    ratio for the harmonics that were actually notched.

    Adaptive rather than blanket because 120 Hz sits on the upper edge of the
    80-120 Hz ripple band: notching it where there is no line noise discards
    real ripple-band signal for nothing. Baylor needs no notch at all, Utah
    needs it badly, UCLA is cleaned by the bipolar montage alone.
    """
    ratios = measure_line_noise(x, fs, freqs)
    y = np.asarray(x, dtype=np.float64)
    applied = {}
    for f0, r in ratios.items():
        if r < ratio_threshold:
            continue
        b, a = iirnotch(f0, q, fs)
        y = sosfiltfilt(tf2sos(b, a), y, axis=-1)
        applied[f0] = r
    return y.astype(np.float32), applied, ratios


def resample_to(x, fs_in, fs_out=TARGET_FS):
    """Polyphase decimation. Never scipy.signal.resample -- see docstring."""
    if abs(fs_in - fs_out) < 1e-6:
        return np.asarray(x, dtype=np.float32)
    from math import gcd
    g = gcd(int(round(fs_in)), int(round(fs_out)))
    up, down = int(round(fs_out)) // g, int(round(fs_in)) // g
    return resample_poly(np.asarray(x, dtype=np.float64), up, down,
                         axis=-1).astype(np.float32)


# =============================================================================
# CONTINUOUS LOADING
# =============================================================================

def _load_blackrock_channels(path, nsx, seg_index, ch_positions,
                             fs_raw, duration_s, verbose=False):
    """Load selected channels of one Blackrock block, resampled to TARGET_FS.

    Read in chunks with a discarded pad: a full block of 20 channels at 2 kHz
    over 4800 s is ~1.5 GB as float64 from neo. Only the channels we actually
    need are ever materialised -- `channel_indexes` is passed to neo.
    """
    reader = _blackrock_reader(path, nsx)
    seg = reader.read_segment(block_index=0, seg_index=int(seg_index), lazy=True)
    asig = max(seg.analogsignals, key=lambda a: a.shape[0])
    t0_file = float(asig.t_start.magnitude)

    out = [[] for _ in ch_positions]
    t = 0.0
    while t < duration_s:
        lo = max(0.0, t - CHUNK_PAD_S)
        hi = min(duration_s, t + CHUNK_S + CHUNK_PAD_S)
        chunk = asig.load(time_slice=(t0_file + lo, t0_file + hi),
                          channel_indexes=list(ch_positions))
        arr = np.asarray(chunk.magnitude, dtype=np.float32).T   # (ch, time)
        arr = resample_to(arr, fs_raw, TARGET_FS)

        pre = int(round((t - lo) * TARGET_FS))
        want = int(round((min(t + CHUNK_S, duration_s) - t) * TARGET_FS))
        for i in range(len(ch_positions)):
            out[i].append(arr[i, pre:pre + want])
        del chunk, arr
        t += CHUNK_S
        if verbose:
            print(f"      ...{min(t, duration_s):.0f}/{duration_s:.0f}s", end="\r")

    return np.stack([np.concatenate(c) for c in out], axis=0)


_NCS_BLOCK_SUFFIX = re.compile(r'_\d{4}$')


def _load_ncs_channels(files, stems_wanted, duration_s, verbose=False):
    """Load selected UCLA contacts of one .ncs block, resampled to TARGET_FS.

    Channels are matched on the electrode stem with the `_NNNN` RECORDING-BLOCK
    suffix removed, because the same electrode appears in every block under a
    different filename (`RMH1.ncs`, `RMH1_0001.ncs`, `RMH1_0002.ncs`) and the
    channel list is read from whichever block happens to be first on disk.

    This mattered for s40: its task is in block `_0002`, but `load_channel_list`
    names channels from block 1, so the requested `RMH1` never matched the
    available `RMH1_0002`. Every contact was then zero-filled and the session
    detected 0 ripples on 3 pairs with 100% of the recording marked
    artifact-free -- silently, because a flat trace crosses no threshold and
    triggers no artifact criterion.

    A channel that cannot be found is now a hard error. Zero-filling one side of
    a bipolar derivation does not produce a degraded ripple estimate, it
    produces a meaningless one, and it looks like a clean session.
    """
    by_stem = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        by_stem.setdefault(stem, f)
        by_stem.setdefault(_NCS_BLOCK_SUFFIX.sub("", stem), f)

    # Resolve every path BEFORE reading anything, so a missing channel is
    # reported as a missing channel rather than surfacing as whatever error the
    # first successful read happens to hit later.
    paths, missing = [], []
    for stem in stems_wanted:
        path = by_stem.get(stem) or by_stem.get(_NCS_BLOCK_SUFFIX.sub("", stem))
        (paths.append(path) if path is not None else missing.append(stem))
    if missing:
        raise RuntimeError(
            f"{len(missing)} requested channel(s) are not in this recording "
            f"block: {missing[:8]}{' ...' if len(missing) > 8 else ''}\n"
            f"  available stems (first 8): {sorted(by_stem)[:8]}\n"
            f"  Rebuild the session's contacts so the channel list comes from "
            f"the block that carries the behaviour.")

    rows = []
    for path in paths:
        sig, fs, _ = read_ncs(path)
        rows.append(resample_to(sig, fs, TARGET_FS))
    n = min(len(r) for r in rows)
    return np.stack([r[:n] for r in rows], axis=0)


def preprocess_session(session, pairs, data_root=None, verbose=True,
                       monopolar=False):
    """Build the continuous bipolar array for one session.

    Returns (signal, meta) where `signal` is (n_pairs, n_samples) float32 at
    TARGET_FS on the behavioural clock, so `sample = round(t_session * fs)`.

    `pairs` is the session's bipolar_pairs table; only the channels those pairs
    reference are ever read from disk.

    `monopolar=True` skips the bipolar subtraction and returns the individual
    contacts instead, one row per unique channel, with `meta['pair_ids']` giving
    the contact labels. This exists for the sharp-wave control: a bipolar
    derivation is a spatial derivative and removes the spatially broad
    low-frequency dipole that the sharp wave is, so the only way to ask whether
    a sharp wave is present at all is to look before that subtraction. Detection
    is never run on this -- it is a morphology check.
    """
    data_root = data_root or swr_io.get_data_root()
    cfg = swr_io.session_config(session, data_root=data_root)
    files, kind, _ = swr_io.discover_raw_files(session, cfg, data_root=data_root)
    blocks, bt, _ = session_block_table(session, data_root=data_root)
    if blocks.empty or not len(pairs):
        raise RuntimeError(f"s{session:02d}: no blocks or no pairs")

    if kind == 'blackrock':
        wanted = sorted({int(v) for v in
                         list(pairs.ns_pos_a) + list(pairs.ns_pos_b)})
        index_of = {v: i for i, v in enumerate(wanted)}
    else:
        wanted = sorted({str(v) for v in
                         list(pairs.ns_label_a) + list(pairs.ns_label_b)})
        index_of = {v: i for i, v in enumerate(wanted)}

    if verbose:
        print(f"  s{session:02d}: {len(blocks)} block(s), {len(pairs)} pairs, "
              f"{len(wanted)} unique channels")

    per_block = []
    for k, b in blocks.iterrows():
        if verbose:
            print(f"    block {int(b.block)}: {b.duration_s:.0f}s @ {b.fs_raw:.0f}Hz")
        # `rec_index` points into `files`, NOT the row order: the resolved run
        # may skip recordings that are not part of this session's task.
        f = files[int(b.rec_index)]
        if kind == 'blackrock':
            raw = _load_blackrock_channels(
                f, _nsx_number(f, cfg), b.seg_index, wanted,
                b.fs_raw, b.duration_s, verbose=verbose)
        else:
            raw = _load_ncs_channels(f, wanted, b.duration_s, verbose=verbose)
        per_block.append(raw)

    # Concatenating the blocks in order reproduces the behavioural clock,
    # because that clock IS the cumulative file duration.
    raw_all = np.concatenate(per_block, axis=1)
    del per_block

    if monopolar:
        sig = np.ascontiguousarray(raw_all, dtype=np.float32)
        row_ids = [str(w) for w in wanted]
    else:
        # Bipolar first (removes most common-mode line noise), then notch.
        sig = np.empty((len(pairs), raw_all.shape[1]), dtype=np.float32)
        for i, (_, p) in enumerate(pairs.iterrows()):
            ka = index_of[int(p.ns_pos_a) if kind == 'blackrock' else str(p.ns_label_a)]
            kb = index_of[int(p.ns_pos_b) if kind == 'blackrock' else str(p.ns_label_b)]
            sig[i] = raw_all[ka] - raw_all[kb]
        row_ids = list(pairs.pair_id)
    del raw_all

    sig, notch_applied, notch_ratios = notch_filter(sig, TARGET_FS)
    if verbose:
        print("    line-noise ratio (1.0 = none): "
              + ", ".join(f"{f:.0f}Hz={r:.2f}" for f, r in notch_ratios.items()))
        print(f"    notched: {[f'{f:.0f}Hz' for f in notch_applied] or 'none needed'}")

    meta = {
        "session": int(session),
        "fs": TARGET_FS,
        "n_pairs": int(sig.shape[0]),
        "monopolar": bool(monopolar),
        "n_samples": int(sig.shape[1]),
        "duration_s": float(sig.shape[1] / TARGET_FS),
        "recording_site": cfg.get('recording_site'),
        "reader": kind,
        "pair_ids": list(row_ids),
        "blocks": blocks.to_dict("records"),
        "clock": "behavioural = cumulative file duration; sample = t*fs",
        "notch_candidates_hz": list(LINE_FREQS),
        "notch_applied_hz": {f"{k:.0f}": v for k, v in notch_applied.items()},
        "line_noise_ratio": {f"{k:.0f}": v for k, v in notch_ratios.items()},
        "notch_ratio_threshold": NOTCH_RATIO_THRESHOLD,
        "resample": "scipy.signal.resample_poly",
    }
    return sig, meta
