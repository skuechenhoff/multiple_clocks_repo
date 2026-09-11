"""
Shared configuration for the techscan tSNR re-analysis.

Pilot EPI sequences acquired 2023 (3T + two 7T sessions) are re-compared to
justify the final 7T protocol used for the main fMRI dataset.

All settings that a result depends on live here, and get dumped into
config.json next to the results so every number is traceable.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
TECHSCAN = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data/techscan')
RAW = TECHSCAN / 'data'
DERIV = TECHSCAN / 'derivatives'
OUT = DERIV / 'group' / 'tsnr_20260910'
FSLDIR = Path('/Users/xpsy1114/fsl')
MNI2MM = FSLDIR / 'data/standard/MNI152_T1_2mm_brain.nii.gz'

# ---------------------------------------------------------------- runs
# (subject, run folder, session label, short label for figures)
# Session labels group runs that were acquired in one sitting -- only
# within-session comparisons are interpretable.
RUNS = [
    # --- 3T, slice-angle sweep, two base sequences, two participants -----
    ('sub-01', 'AB_sequ_transv',        '3T',    'AB 0deg'),
    ('sub-01', 'AB_sequ_plus30',        '3T',    'AB +30deg'),
    ('sub-01', 'AB_sequ_min30',         '3T',    'AB -30deg'),
    ('sub-01', 'Leip_sequ_transv',      '3T',    'Leip 0deg'),
    ('sub-01', 'Leip_sequ_plus30',      '3T',    'Leip +30deg'),
    ('sub-01', 'Leip_sequ_min30',       '3T',    'Leip -30deg'),
    ('sub-02', 'AB_sequ_transv',        '3T',    'AB 0deg'),
    ('sub-02', 'AB_sequ_plus30',        '3T',    'AB +30deg'),
    ('sub-02', 'AB_sequ_min30',         '3T',    'AB -30deg'),
    ('sub-02', 'Leip_sequ_transv',      '3T',    'Leip 0deg'),
    ('sub-02', 'Leip_sequ_plus30',      '3T',    'Leip +30deg'),
    ('sub-02', 'Leip_sequ_min30',       '3T',    'Leip -30deg'),
    # --- 7T session 1 (sub-02): voxel size / acceleration exploration ----
    ('sub-02', 'run1_T7_AB_rep',        '7T-s1', '2.0mm MB3/P2'),
    ('sub-02', 'run2_T7_2mm_3iPat_2MB', '7T-s1', '2.0mm MB2/P3'),
    ('sub-02', 'run3_T7_15mm_3iPat_3MB', '7T-s1', '1.5mm MB3/P3'),
    ('sub-02', 'run4_T7_15mm_2iPat_3MB', '7T-s1', '1.5mm MB3/P2'),
    ('sub-02', 'run5_T7_superfast',     '7T-s1', '1.5mm MB6/P2'),
    ('sub-02', 'run6_T7_12mm_3iPat_3MB', '7T-s1', '1.2mm MB3/P3'),
    # --- 7T session 2 (sub-03): acceleration sweep at fixed 2mm ----------
    ('sub-03', 'run1_3Trep_7T2',        '7T-s2', '2.0mm MB3/P2'),
    ('sub-03', 'run2_25mm_fast_7T2',    '7T-s2', '2.5mm MB3/P2'),
    ('sub-03', 'run3_mb4_fast_7T2',     '7T-s2', '2.0mm MB4/P2'),
    ('sub-03', 'run4_mb3ipat3_7T2',     '7T-s2', '2.0mm MB3/P3'),
    ('sub-03', 'run5_ipat4_7T2',        '7T-s2', '2.0mm MB2/P4'),   # FINAL
]

FINAL_RUN = ('sub-03', 'run5_ipat4_7T2')

# Always use preproc.feat: preproc3T.feat / preproc_wholeb.feat are re-runs of
# the same acquisition with different registration flags, not other sequences.
FEAT = 'preproc.feat'

# ---------------------------------------------------------------- ROIs
# The study's own masks, in the priority order the hypotheses give them:
#   1. mPFC   -- the action-plan result lives here
#   2. EC     -- the abstract task-structure result
#   3. HC
#   4. vOFC   -- wanted, but knowingly sacrificed to susceptibility dropout
#
# vOFC and mPFC come from the project mask library and are disjoint: vOFC sits
# at MNI z -30..-2 (directly above the frontal sinus, where dropout is worst),
# mPFC at z -10..+48 (dorsal to it, largely out of the sinus field). That
# split is the whole point -- the 2023 Harvard-Oxford "Frontal Medial Cortex"
# mask spanned both and averaged the two regimes into one uninterpretable
# number.
#
# EC and HC are atlas-defined (anatomically calibrated volumes). The Garvert
# MTL mask is carried alongside as the project's most precise MTL boundary,
# both whole and split into EC/HC by winner-take-all on the atlas probability
# maps, so the edge voxels can be checked against the atlas definition.
#
# NOTE ON ALIGNMENT: mask_PFC_LR_smoothed_resampled.nii.gz is stored with the
# opposite x-orientation to MNI152_T1_2mm (affine diag +2 vs -2). Masking by
# array index would mirror it left-right. Every mask is therefore resampled
# through its affine onto the reference grid, never indexed directly.
PROJECT_MASKS = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/masks')

ATLAS = {
    'HO_cort': FSLDIR / 'data/atlases/HarvardOxford/HarvardOxford-cort-prob-2mm.nii.gz',
    'HO_sub':  FSLDIR / 'data/atlases/HarvardOxford/HarvardOxford-sub-prob-2mm.nii.gz',
    'Juelich': FSLDIR / 'data/atlases/Juelich/Juelich-prob-2mm.nii.gz',
}

# label -> (source kind, spec, display name, priority rank)
#   ('file', filename)              binary mask from the project library
#   ('atlas', (atlas key, index))   probability map, thresholded at ROI_THR
#   ('garvert', which)              Garvert MTL, whole or split by atlas
ROIS = {
    'mPFC':   ('file',    'mask_PFC_LR_smoothed_resampled.nii.gz', 'mPFC', 1),
    'EC':     ('atlas',   ('Juelich', (18, 19)),      'Entorhinal ctx', 2),
    'HC':     ('atlas',   ('HO_sub',  (8, 18)),       'Hippocampus',    3),
    'vOFC':   ('file',    'vMPFC.nii.gz',             'vOFC',           4),
    # --- supplementary: the project's own MTL boundary -------------------
    'MTL_gv': ('garvert', 'whole', 'MTL (Garvert)',   5),
    'EC_gv':  ('garvert', 'EC',    'Entorhinal (Garvert)', 6),
    'HC_gv':  ('garvert', 'HC',    'Hippocampus (Garvert)', 7),
}

# ROIs carried in the main figures; the rest are robustness checks
PRIMARY_ROIS = ['mPFC', 'EC', 'HC', 'vOFC']

GARVERT_MASK = 'Garvert_MTL_elife_FigS2_1B.nii'

# Threshold on the atlas probability maps (percent). thr50 gives anatomically
# plausible volumes (bilateral hippocampus 8.4 cm3, entorhinal 7.3 cm3); the
# 2023 analysis used thr10, which is 2-3x too large and bleeds into ventricle
# and white matter.
ROI_THR = 50
ROI_THR_SENSITIVITY = [10, 25, 75]

# ---------------------------------------------------------------- tSNR
# Runs differ in length (83-442 volumes, 111-546 s). The experiment is limited
# by scanner time, not by volume count, so runs are matched on DURATION: the
# longest window every run has in common. Per-run N then varies, and the
# time-efficiency metric below is what actually answers "does a fixed-length
# task sample give better signal".
MATCH_SECONDS = 110.0

# A voxel counts as "usable" if its tSNR clears this. Used for the ROI
# coverage / dropout metric (% of ROI voxels above threshold).
TSNR_USABLE = 20.0

# FEAT preprocessing already applied (read back from each design.fsf, recorded
# here for the provenance dump): MCFLIRT, slice-timing correction (interleaved),
# fieldmap unwarping, BET, NO spatial smoothing, 100 s highpass.
HIGHPASS_S = 100.0

SEED = 42


# ---------------------------------------------------------------- comparisons
# The decision sequence, in the order the thesis walks through it. `step` is
# the position in that argument and `verdict` is what the step concluded, so
# the figures and the text cannot drift apart.
#
# `arms` is (subject, run, display label) in plotting order, so a panel's axis
# labels never depend on matching strings elsewhere. Within a comparison, tSNR
# is scored only on voxels every arm reached (common_coverage.py), so slab
# placement cannot masquerade as a sequence effect. Groups stay within one
# session except where noted, because shim, coil loading and participant would
# otherwise confound the parameter.
#
# ON MB vs iPAT. These are different axes and the sweep did not vary them
# independently -- total acceleration is MB x iPAT, which runs 6, 8, 9, 8
# across accel_7T_s2. That makes a raw four-arm comparison partly a comparison
# of total acceleration. `split_matched_7T_s2` fixes it: run3 and run5 have the
# SAME total acceleration of 8 and differ only in where it is spent, so it
# isolates the choice that actually had to be made.
COMPARISONS = {
    'field_3T_vs_7T': dict(
        step=1, title='Field strength',
        question='Is 7T worth it for the medial temporal lobe?',
        verdict='Yes for MTL, at a cost in orbitofrontal cortex. 3T is out.',
        note='same participant (sub-02), identical MB3/iPAT2, 2.0 mm, TE 20 ms; '
             'different sessions, which is unavoidable for a field comparison',
        vary='field strength',
        arms=[('sub-02', 'AB_sequ_transv', '3T'),
              ('sub-02', 'run1_T7_AB_rep', '7T')]),

    'voxel_7T_s2': dict(
        step=2, title='Voxel size',
        question='How small can the voxels be?',
        verdict='2.0 mm. Bigger returns little; smaller falls off a cliff.',
        note='clean single-parameter step: both MB3/iPAT2, only voxel size and '
             'the TE/TR it permits differ',
        vary='voxel size',
        arms=[('sub-03', 'run1_3Trep_7T2', '2.0 mm'),
              ('sub-03', 'run2_25mm_fast_7T2', '2.5 mm')]),

    'voxel_7T_s1': dict(
        step=2, title='Voxel size, going finer',
        question='What does going below 2 mm cost?',
        verdict='Too much. 1.5 mm loses ~30% of hippocampal tSNR, 1.2 mm ~66%.',
        note='2.0 and 1.5 mm are matched on MB3/iPAT2; the 1.2 mm arm also '
             'moves to iPAT3 and TE 23.2 ms, so it is not a clean single-'
             'parameter step. Slab placement varied across this session, so '
             'the common-coverage fractions are low -- treat as supporting.',
        vary='voxel size',
        arms=[('sub-02', 'run1_T7_AB_rep', '2.0 mm'),
              ('sub-02', 'run4_T7_15mm_2iPat_3MB', '1.5 mm'),
              ('sub-02', 'run6_T7_12mm_3iPat_3MB', '1.2 mm*')]),

    'angle_AB_3T': dict(
        step=3, title='Slice angle, accelerated sequence',
        question='How should the slab be tilted?',
        verdict='This sequence prefers -30 deg everywhere -- but +30 deg was '
                'adopted, on the Leip result and the literature.',
        note='within-subject, n=2 (sub-01, sub-02), one session each. Short '
             'readout (28.9 ms), so relatively insensitive to the '
             'susceptibility-gradient mechanism that makes tilt matter. Note '
             'this panel measures tSNR, not BOLD sensitivity, which is what '
             'the published tilt optima are defined on.',
        vary='slab tilt',
        arms=[('sub-01', 'AB_sequ_min30', '-30 deg'),
              ('sub-02', 'AB_sequ_min30', '-30 deg'),
              ('sub-01', 'AB_sequ_transv', 'transverse'),
              ('sub-02', 'AB_sequ_transv', 'transverse'),
              ('sub-01', 'AB_sequ_plus30', '+30 deg'),
              ('sub-02', 'AB_sequ_plus30', '+30 deg')]),

    'angle_Leip_3T': dict(
        step=3, title='Slice angle, unaccelerated sequence',
        question='Do entorhinal and orbitofrontal cortex want the same tilt?',
        verdict='No -- they want opposite tilts. EC prefers +30 deg, vOFC '
                'prefers -30 deg, replicated in both participants.',
        note='within-subject, n=2. Readout is 53.5 ms, 1.85x the accelerated '
             'sequence, so this arm is far more sensitive to susceptibility '
             'gradients along the phase-encode axis -- the mechanism that '
             'sets the optimal tilt. This is the arm that speaks to the '
             'published result.',
        vary='slab tilt',
        arms=[('sub-01', 'Leip_sequ_min30', '-30 deg'),
              ('sub-02', 'Leip_sequ_min30', '-30 deg'),
              ('sub-01', 'Leip_sequ_transv', 'transverse'),
              ('sub-02', 'Leip_sequ_transv', 'transverse'),
              ('sub-01', 'Leip_sequ_plus30', '+30 deg'),
              ('sub-02', 'Leip_sequ_plus30', '+30 deg')]),

    'multiband_7T_s1': dict(
        step=4, title='Multiband factor',
        question='How far can multiband be pushed?',
        verdict='Not past 3. MB6 costs 29-44% of tSNR everywhere.',
        note='cleanest single-parameter comparison in the dataset: same voxel '
             'size, slice count, iPAT and TE; only the multiband factor differs',
        vary='multiband factor',
        arms=[('sub-02', 'run4_T7_15mm_2iPat_3MB', 'MB3'),
              ('sub-02', 'run5_T7_superfast', 'MB6')]),

    'split_matched_7T_s2': dict(
        step=5, title='Where to spend the acceleration',
        question='Given a fixed total acceleration, in-plane or through-plane?',
        verdict='In-plane. Same total acceleration, but shorter TE, half the '
                'distortion and vOFC recovered.',
        note='THE decisive comparison: both arms have total acceleration '
             'MB x iPAT = 8, same session, same participant, same 2.0 mm '
             'voxel. Only the split differs, and with it TE and readout length.',
        vary='where the acceleration is spent',
        arms=[('sub-03', 'run3_mb4_fast_7T2', 'through-plane\nMB4 / iPAT2'),
              ('sub-03', 'run5_ipat4_7T2', 'in-plane\nMB2 / iPAT4')]),

    'accel_7T_s2': dict(
        step=5, title='The full acceleration sweep',
        question='All four settings piloted, for completeness.',
        verdict='Confounded by total acceleration; see split_matched_7T_s2.',
        note='total acceleration (MB x iPAT) is 6, 8, 9, 8 across these arms, '
             'so this is partly a comparison of how much acceleration, not '
             'only of how it was split. Shown for completeness.',
        vary='MB / iPAT (and the shortest TE each allows)',
        arms=[('sub-03', 'run1_3Trep_7T2', '2\nMB3'),
              ('sub-03', 'run3_mb4_fast_7T2', '2\nMB4'),
              ('sub-03', 'run4_mb3ipat3_7T2', '3\nMB3'),
              ('sub-03', 'run5_ipat4_7T2', '4\nMB2')]),

    'field_3T_vs_7T_final': dict(
        step=6, title='3T against the adopted protocol',
        question='Was the whole exercise worth it?',
        verdict='Yes. The adopted protocol beats 3T in all four ROIs.',
        note='different participants (sub-02 vs sub-03) and sessions, so this '
             'compares what was available at 3T against what was adopted, not '
             'a controlled field manipulation. Both 2.0 mm.',
        vary='field strength + protocol',
        arms=[('sub-02', 'AB_sequ_transv', '3T\nMB3/iPAT2'),
              ('sub-03', 'run5_ipat4_7T2', '7T final\nMB2/iPAT4')]),
}

for _g in COMPARISONS.values():
    _g['runs'] = [(s, r) for s, r, _ in _g['arms']]
    _g['order'] = list(dict.fromkeys(lab for _, _, lab in _g['arms']))

# Panels put a box around whichever arm corresponds to the adopted protocol.
# Verified from the image affines, not from notes: every 7T run, the adopted
# protocol included, was acquired at +30 deg (anterior edge of the slice
# tilted towards the feet). The 3T angle pilot's own tSNR winner was -30 deg
# for the accelerated sequence, so the adopted tilt is NOT that pilot's
# winner -- see the step 3 note.
# A box means "this arm IS the protocol that was adopted" -- nothing weaker.
# Earlier this set also held parameter values that were adopted while the arm
# itself was a pilot sequence (the +30 deg bar of a 3T pilot, the MB3 bar of a
# 1.5 mm pilot), which reads as "this sequence was chosen" and is wrong. Where
# a panel needs to mark an adopted parameter *value* instead, it does so with
# an explicit "adopted" label, not with this box.
ADOPTED_LABELS = {'7T final\nMB2/iPAT4', 'in-plane\nMB2 / iPAT4', '4\nMB2'}

# Slab tilt actually used, read back from the affines (deg; positive = anterior
# edge towards the feet, matching Weiskopf et al. 2006's convention), and the
# phase-encode direction. A>>P is Weiskopf's NEGATIVE PE polarity, which is the
# polarity his optimal-tilt signs have to be read against.
ADOPTED_TILT_DEG = +30
PHASE_ENCODE = 'A>>P'
