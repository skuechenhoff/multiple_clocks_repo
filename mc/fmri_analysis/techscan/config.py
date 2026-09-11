"""
Settings for the 7T EPI protocol-selection re-analysis.

Everything a result depends on lives here: which runs exist, which ROIs are
measured, and which comparisons the argument is built from. Nothing in this
package writes into the repository -- all output goes to OUTPUT_DIR under the
data tree.
"""
from pathlib import Path

# ----------------------------------------------------------------- paths
TECHSCAN_DIR = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data/techscan')
RAW_DIR = TECHSCAN_DIR / 'data'
DERIVATIVES_DIR = TECHSCAN_DIR / 'derivatives'
PROTOCOL_DIR = TECHSCAN_DIR / 'protocols'
OUTPUT_DIR = DERIVATIVES_DIR / 'group' / 'tsnr_20260910'

PROJECT_MASK_DIR = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/masks')
FSL_DIR = Path('/Users/xpsy1114/fsl')
MNI_2MM_BRAIN = FSL_DIR / 'data/standard/MNI152_T1_2mm_brain.nii.gz'
MNI_2MM_HEAD = FSL_DIR / 'data/standard/MNI152_T1_2mm.nii.gz'

# Always analyse preproc.feat. The preproc3T.feat and preproc_wholeb.feat
# directories are re-runs of the same acquisition with different registration
# flags, not different scans -- diffing their design.fsf shows they differ only
# in reginitial_highres_yn and the stats flags.
FEAT_DIR_NAME = 'preproc.feat'

# The protocol printout describing the sequence that was adopted.
FINAL_PROTOCOL_FILE = PROTOCOL_DIR / '2023_017 Music Box - Protocol v1.md'
FINAL_PROTOCOL_NAME = 'cmrr_mbep2d_bold_3Trep_2mm_mb2p4'

# ------------------------------------------------------------------ runs
# (subject, run folder, session, short label). Sessions group runs acquired in
# one sitting; only within-session comparisons are interpretable, because shim,
# coil loading and participant all change between sessions.
RUNS = [
    # 3T, slice-angle sweep, two base sequences, two participants
    ('sub-01', 'AB_sequ_transv',         '3T',    'AB 0deg'),
    ('sub-01', 'AB_sequ_plus30',         '3T',    'AB +30deg'),
    ('sub-01', 'AB_sequ_min30',          '3T',    'AB -30deg'),
    ('sub-01', 'Leip_sequ_transv',       '3T',    'Leip 0deg'),
    ('sub-01', 'Leip_sequ_plus30',       '3T',    'Leip +30deg'),
    ('sub-01', 'Leip_sequ_min30',        '3T',    'Leip -30deg'),
    ('sub-02', 'AB_sequ_transv',         '3T',    'AB 0deg'),
    ('sub-02', 'AB_sequ_plus30',         '3T',    'AB +30deg'),
    ('sub-02', 'AB_sequ_min30',          '3T',    'AB -30deg'),
    ('sub-02', 'Leip_sequ_transv',       '3T',    'Leip 0deg'),
    ('sub-02', 'Leip_sequ_plus30',       '3T',    'Leip +30deg'),
    ('sub-02', 'Leip_sequ_min30',        '3T',    'Leip -30deg'),
    # 7T session 1 (sub-02): voxel size and acceleration exploration
    ('sub-02', 'run1_T7_AB_rep',         '7T-s1', '2.0mm MB3/P2'),
    ('sub-02', 'run2_T7_2mm_3iPat_2MB',  '7T-s1', '2.0mm MB2/P3'),
    ('sub-02', 'run3_T7_15mm_3iPat_3MB', '7T-s1', '1.5mm MB3/P3'),
    ('sub-02', 'run4_T7_15mm_2iPat_3MB', '7T-s1', '1.5mm MB3/P2'),
    ('sub-02', 'run5_T7_superfast',      '7T-s1', '1.5mm MB6/P2'),
    ('sub-02', 'run6_T7_12mm_3iPat_3MB', '7T-s1', '1.2mm MB3/P3'),
    # 7T session 2 (sub-03): acceleration sweep at fixed 2 mm
    ('sub-03', 'run1_3Trep_7T2',         '7T-s2', '2.0mm MB3/P2'),
    ('sub-03', 'run2_25mm_fast_7T2',     '7T-s2', '2.5mm MB3/P2'),
    ('sub-03', 'run3_mb4_fast_7T2',      '7T-s2', '2.0mm MB4/P2'),
    ('sub-03', 'run4_mb3ipat3_7T2',      '7T-s2', '2.0mm MB3/P3'),
    ('sub-03', 'run5_ipat4_7T2',         '7T-s2', '2.0mm MB2/P4'),  # adopted
]

ADOPTED_RUN = ('sub-03', 'run5_ipat4_7T2')

# sub-03's BOLD JSON sidecars were lost at conversion. These come from the
# 24-08-2023 protocol printout, the only surviving record of MB/iPAT/flip for
# that session. Everything independently checkable -- TR, TE, slice count, FOV,
# base resolution, voxel size -- matches the NIfTI headers and the FEAT design
# files exactly, and inventory.py re-checks TE on every run.
#   protocol name: (field_T, TE_ms, flip_deg, multiband, ipat, bandwidth_hz_px)
PROTOCOL_PRINTOUT_SUB03 = {
    'cmrr_mbep2d_bold_3Trep_2mm_mb3p2':   (7, 19.6, 53, 3, 2, 2314),
    'cmrr_mbep2d_bold_3Trep_2.5mm_mb3p2': (7, 16.8, 53, 3, 2, 2236),
    'cmrr_mbep2d_bold_3Trep_2mm_mb4p2':   (7, 20.6, 53, 4, 2, 2204),
    'cmrr_mbep2d_bold_3Trep_2mm_mb3p3':   (7, 15.0, 53, 3, 3, 2436),
    'cmrr_mbep2d_bold_3Trep_2mm_mb2p4':   (7, 12.8, 53, 2, 4, 2436),
}

# ------------------------------------------------------------------ ROIs
# In the priority order the hypotheses give them:
#   1 mPFC   the action-plan result
#   2 EC     the abstract task-structure result
#   3 HC
#   4 mOFC   wanted, but knowingly sacrificed to susceptibility dropout
#
# mOFC and mPFC come from the project mask library and are disjoint: mOFC sits
# at MNI z -30..-2, directly above the frontal sinus where dropout is worst,
# mPFC at z -10..+48, dorsal to it. Keeping them separate is essential -- a
# single mask spanning both averages two opposite regimes into an
# uninterpretable number.
#
# ALIGNMENT: mask_PFC_LR_smoothed_resampled.nii.gz is stored with the opposite
# x-orientation to MNI152_T1_2mm (affine diagonal +2 rather than -2). Masking by
# array index would mirror it left-right, so every mask is resampled through its
# affine onto the reference grid and never indexed directly.
ATLAS_FILES = {
    'HO_cortical':    FSL_DIR / 'data/atlases/HarvardOxford/HarvardOxford-cort-prob-2mm.nii.gz',
    'HO_subcortical': FSL_DIR / 'data/atlases/HarvardOxford/HarvardOxford-sub-prob-2mm.nii.gz',
    'Juelich':        FSL_DIR / 'data/atlases/Juelich/Juelich-prob-2mm.nii.gz',
}

# roi -> (source_kind, spec, display name, priority)
#   'project_file'  filename in PROJECT_MASK_DIR, already binary
#   'atlas'         (atlas key, volume indices) thresholded at ATLAS_THRESHOLD
#   'garvert'       the project's MTL boundary, whole or split by atlas
ROI_DEFINITIONS = {
    'mPFC': ('project_file', 'mask_PFC_LR_smoothed_resampled.nii.gz', 'mPFC', 1),
    'EC':   ('atlas', ('Juelich', (18, 19)), 'Entorhinal ctx', 2),
    'HC':   ('atlas', ('HO_subcortical', (8, 18)), 'Hippocampus', 3),
    'mOFC': ('project_file', 'vMPFC.nii.gz', 'mOFC', 4),
    # supplementary: the project's own MTL boundary, whole and split
    'MTL_garvert': ('garvert', 'whole', 'MTL (Garvert)', 5),
    'EC_garvert':  ('garvert', 'EC', 'EC (Garvert)', 6),
    'HC_garvert':  ('garvert', 'HC', 'HC (Garvert)', 7),
}

PRIMARY_ROIS = ['mPFC', 'EC', 'HC', 'mOFC']
GARVERT_MTL_FILE = 'Garvert_MTL_elife_FigS2_1B.nii'

# Threshold on the atlas probability maps, in percent. 50% gives anatomically
# plausible volumes (bilateral hippocampus 8.4 cm3, entorhinal 7.3 cm3). The
# 2023 analysis used 10%, which is 2-3x too large and bleeds into ventricle and
# white matter.
ATLAS_THRESHOLD = 50

# --------------------------------------------------------------- analysis
# Runs differ in length (83-442 volumes, 111-546 s). The experiment is bounded
# by scanner time rather than volume count, so runs are matched on DURATION:
# the longest window every run shares. Volumes per run then vary with TR, which
# is why efficiency (tSNR / sqrt(TR)) is reported alongside raw tSNR.
MATCHED_WINDOW_SECONDS = 110.0

# A voxel counts as "usable" above this tSNR. Used only for the usable-fraction
# metric; the dropout metric uses FEAT's brain mask instead (see metrics.py).
USABLE_TSNR_THRESHOLD = 20.0

# Slab tilt actually used, verified from the image affines and the protocol
# printout. Positive means the anterior edge of the slice is tilted towards the
# feet, matching the convention in Weiskopf et al. (2006). A>>P is that paper's
# NEGATIVE phase-encode polarity, which is what their optimal tilt signs must be
# read against.
ADOPTED_TILT_DEG = 30
PHASE_ENCODE_DIRECTION = 'A>>P'
TILT_ANGLES_DEG = (-30, 0, 30)

# Sequences whose tilt sensitivity is weighed in metrics.tilt_penalties, with
# the parameters that scale each susceptibility mechanism.
TILT_REFERENCE_SEQUENCES = {
    'Leip 3T (2.5 mm, no iPAT)': dict(readout_s=53.46e-3, te_s=22.0e-3, slice_mm=2.5),
    'AB 3T (2.0 mm, iPAT2)':     dict(readout_s=28.89e-3, te_s=20.0e-3, slice_mm=2.0),
    'adopted 7T (iPAT4)':        dict(readout_s=14.18e-3, te_s=12.8e-3, slice_mm=2.0),
}

# Off-resonance at 3T in a strongly affected frontal voxel, scaled by field.
# Used only to express readout duration as an interpretable displacement in the
# sequence table; the figures use the measured fieldmap instead.
REFERENCE_OFFRESONANCE_HZ_AT_3T = 50.0

RANDOM_SEED = 42

# ----------------------------------------------------------- comparisons
# The decision chain, in the order the write-up walks through it. Each entry
# carries its step number, question, verdict and caveat, so figures and text
# cannot drift apart.
#
# `arms` is (subject, run, display label) in plotting order, so axis labels
# never depend on matching strings elsewhere. Within a comparison, tSNR is
# scored only on voxels every arm reached, so slab placement cannot masquerade
# as a sequence effect.
#
# ON MB vs iPAT: these are different axes and the sweep did not vary them
# independently. Total acceleration is MB x iPAT, which runs 6, 8, 9, 8 across
# `acceleration_sweep`, making a raw four-arm comparison partly a comparison of
# how much acceleration. `acceleration_split` fixes that: both its arms have
# total acceleration 8 and differ only in where it is spent.
COMPARISONS = {
    'field_strength': dict(
        step=1, title='Field strength',
        question='Is 7T worth it for the medial temporal lobe?',
        verdict='Yes for MTL, at a cost in orbitofrontal cortex. 3T is out.',
        note='same participant (sub-02), identical MB3/iPAT2, 2.0 mm, TE 20 ms; '
             'different sessions, unavoidable for a field comparison',
        varies='field strength',
        arms=[('sub-02', 'AB_sequ_transv', '3T'),
              ('sub-02', 'run1_T7_AB_rep', '7T')]),

    'voxel_size': dict(
        step=2, title='Voxel size',
        question='How small can the voxels be?',
        verdict='2.0 mm. Bigger returns little; smaller falls off a cliff.',
        note='clean single-parameter step: both MB3/iPAT2, only voxel size and '
             'the TE/TR it permits differ',
        varies='voxel size',
        arms=[('sub-03', 'run1_3Trep_7T2', '2.0 mm'),
              ('sub-03', 'run2_25mm_fast_7T2', '2.5 mm')]),

    'voxel_size_fine': dict(
        step=2, title='Voxel size, going finer',
        question='What does going below 2 mm cost?',
        verdict='Too much. 1.5 mm loses ~30% of hippocampal tSNR, 1.2 mm ~66%.',
        note='2.0 and 1.5 mm are matched on MB3/iPAT2 and TE 20 ms; the 1.2 mm '
             'arm also moves to iPAT3 and TE 23.2 ms, so it is not a clean '
             'single-parameter step. Slab placement varied across this session, '
             'so common-coverage fractions are low -- treat as supporting.',
        varies='voxel size',
        arms=[('sub-02', 'run1_T7_AB_rep', '2.0 mm'),
              ('sub-02', 'run4_T7_15mm_2iPat_3MB', '1.5 mm'),
              ('sub-02', 'run6_T7_12mm_3iPat_3MB', '1.2 mm*')]),

    'slice_angle_accelerated': dict(
        step=3, title='Slice angle, accelerated sequence',
        question='How should the slab be tilted?',
        verdict='This sequence prefers -30 deg in tSNR, but +30 deg was adopted '
                'on the field-gradient mechanism. Supplementary only.',
        note='within-subject, n=2. Readout 28.9 ms, so relatively insensitive '
             'to the susceptibility mechanism that makes tilt matter. Measures '
             'tSNR, not the BOLD sensitivity on which published tilt optima are '
             'defined.',
        varies='slab tilt',
        arms=[('sub-01', 'AB_sequ_min30', '-30 deg'),
              ('sub-02', 'AB_sequ_min30', '-30 deg'),
              ('sub-01', 'AB_sequ_transv', 'transverse'),
              ('sub-02', 'AB_sequ_transv', 'transverse'),
              ('sub-01', 'AB_sequ_plus30', '+30 deg'),
              ('sub-02', 'AB_sequ_plus30', '+30 deg')]),

    'slice_angle_unaccelerated': dict(
        step=3, title='Slice angle, unaccelerated sequence',
        question='Do entorhinal and orbitofrontal cortex want the same tilt?',
        verdict='EC prefers +30 deg, mOFC prefers -30 deg, in both participants.',
        note='within-subject, n=2. Readout 53.5 ms, 1.85x the accelerated '
             'sequence, so far more sensitive to susceptibility gradients along '
             'the phase-encode axis.',
        varies='slab tilt',
        arms=[('sub-01', 'Leip_sequ_min30', '-30 deg'),
              ('sub-02', 'Leip_sequ_min30', '-30 deg'),
              ('sub-01', 'Leip_sequ_transv', 'transverse'),
              ('sub-02', 'Leip_sequ_transv', 'transverse'),
              ('sub-01', 'Leip_sequ_plus30', '+30 deg'),
              ('sub-02', 'Leip_sequ_plus30', '+30 deg')]),

    'multiband': dict(
        step=4, title='Multiband factor',
        question='How far can through-plane acceleration be pushed?',
        verdict='Not past 3. MB6 costs 29-44% of tSNR everywhere.',
        note='cleanest single-parameter comparison in the dataset: same voxel '
             'size, slice count, iPAT and TE; only the multiband factor differs',
        varies='multiband factor',
        arms=[('sub-02', 'run4_T7_15mm_2iPat_3MB', 'MB3'),
              ('sub-02', 'run5_T7_superfast', 'MB6')]),

    'acceleration_split': dict(
        step=5, title='Where to spend the acceleration',
        question='Given a fixed total acceleration, in-plane or through-plane?',
        verdict='In-plane. Same total, shorter TE, half the distortion, mOFC '
                'recovered.',
        note='the decisive comparison: both arms have total acceleration '
             'MB x iPAT = 8, same session, same participant, same 2.0 mm voxel. '
             'Only the split differs, and with it TE and readout length.',
        varies='where the acceleration is spent',
        arms=[('sub-03', 'run3_mb4_fast_7T2', 'through-plane\nMB4 / iPAT2'),
              ('sub-03', 'run5_ipat4_7T2', 'in-plane\nMB2 / iPAT4')]),

    'acceleration_sweep': dict(
        step=5, title='The full acceleration sweep',
        question='All four settings piloted, for completeness.',
        verdict='Confounded by total acceleration; quote acceleration_split.',
        note='total acceleration (MB x iPAT) is 6, 8, 9, 8 across these arms, so '
             'this is partly a comparison of how much acceleration rather than '
             'only of how it was split. Supplementary.',
        varies='MB / iPAT and the shortest TE each allows',
        arms=[('sub-03', 'run1_3Trep_7T2', '2\nMB3'),
              ('sub-03', 'run3_mb4_fast_7T2', '2\nMB4'),
              ('sub-03', 'run4_mb3ipat3_7T2', '3\nMB3'),
              ('sub-03', 'run5_ipat4_7T2', '4\nMB2')]),

    'field_vs_adopted': dict(
        step=6, title='3T against the adopted protocol',
        question='Was the whole exercise worth it?',
        verdict='Yes. The adopted protocol beats 3T in all four ROIs.',
        note='different participants and sessions, so this compares what was '
             'available at 3T against what was adopted, not a controlled field '
             'manipulation. Both 2.0 mm.',
        varies='field strength and protocol',
        arms=[('sub-02', 'AB_sequ_transv', '3T\nMB3/iPAT2'),
              ('sub-03', 'run5_ipat4_7T2', '7T final\nMB2/iPAT4')]),
}

for _comparison in COMPARISONS.values():
    _comparison['runs'] = [(sub, run) for sub, run, _ in _comparison['arms']]
    _comparison['order'] = list(dict.fromkeys(
        label for _, _, label in _comparison['arms']))

# A box on a panel means "this arm IS the adopted protocol" and nothing weaker.
# Where a panel compares pilot sequences, the adopted parameter *value* is
# labelled instead, so a box never implies a pilot sequence was chosen.
ADOPTED_ARM_LABELS = {'7T final\nMB2/iPAT4', 'in-plane\nMB2 / iPAT4', '4\nMB2'}
