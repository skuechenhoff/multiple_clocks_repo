"""
Figure panels, one file each, sized for the thesis layout.

Plot panels are 5.4 x 3.9 cm, brain blocks 8.6 cm wide, nothing below 9 pt
Arial. At that size a panel carries one idea, so each one does. Output is
vector PDF with Arial embedded as TrueType (text stays selectable and editable
in Illustrator) plus a .jpg convenience copy; brain images are genuine rasters
embedded at 400 dpi.

Panels are drawn on a fixed canvas rather than a tight bounding box, so every
panel exports at exactly the same physical size regardless of how many lines
its tick labels take -- otherwise they refuse to line up side by side.

A BOX means "this arm IS the adopted protocol" and nothing weaker. Where a
panel compares pilot sequences, the adopted parameter *value* is labelled
instead, so a box never implies a pilot sequence was chosen.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from .config import (ADOPTED_ARM_LABELS, ADOPTED_TILT_DEG, COMPARISONS,
                     MNI_2MM_HEAD, OUTPUT_DIR, PRIMARY_ROIS, TILT_ANGLES_DEG,
                     TILT_REFERENCE_SEQUENCES, USABLE_TSNR_THRESHOLD)
from .metrics import tsnr_by_arm

CM = 1 / 2.54
PANEL_WIDTH, PANEL_HEIGHT = 5.4 * CM, 3.9 * CM
MAP_WIDTH, MAP_HEIGHT = 8.6 * CM, 7.0 * CM

# era_brewer "Showgirl2", n=7 -- the project's fixed ROI assignment
SHOWGIRL2 = ['#B74C2D', '#448363', '#CCB178', '#C1DCBF',
             '#DC673E', '#7BB594', '#C1DCBF']
ROI_COLOUR = {
    'mPFC': SHOWGIRL2[1],
    'EC':   SHOWGIRL2[0],
    'HC':   '#23677E',
    'mOFC': '#a30d6c',
    'MTL_garvert': '#23677E',
    'EC_garvert': SHOWGIRL2[0],
    'HC_garvert': '#23677E',
}
ROI_SHORT = {'mPFC': 'mPFC', 'EC': 'EC', 'HC': 'HC', 'mOFC': 'mOFC'}
ADOPTED_COLOUR = '#0e3d3a'   # the project's "observed value" dark green

# the four acceleration arms, short enough to fit a 5.4 cm panel at 9 pt
ACCELERATION_TICKS = ['2\nMB3', '2\nMB4', '3\nMB3', '4\nMB2']

FIGURE_DIR = OUTPUT_DIR / 'figures'
MNI_HEAD = None   # loaded lazily, it is only needed for the map panels

# which panel files show which comparison, for the lookup table
PANEL_INDEX = {
    'field_strength':            ['p_field_bars', 'p_maps_field'],
    'voxel_size':                ['p_voxel_curve'],
    'voxel_size_fine':           ['p_voxel_curve'],
    'slice_angle_accelerated':   ['p_angle_accelerated'],
    'slice_angle_unaccelerated': ['p_angle_unaccelerated'],
    'multiband':                 ['p_multiband'],
    'acceleration_split':        ['p_split_matched', 'p_maps_split',
                                  'p_distortion_maps'],
    'acceleration_sweep':        ['p_acceleration_change', 'p_displacement',
                                  'p_mofc_dropout'],
    'field_vs_adopted':          ['(text only)'],
}


def apply_style():
    matplotlib.rcParams.update({
        'font.family': 'Arial',
        'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'figure.titlesize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2, 'ytick.major.size': 2,
        'axes.labelpad': 2, 'xtick.major.pad': 2, 'ytick.major.pad': 2,
        'lines.linewidth': 1.2, 'lines.markersize': 3,
        'legend.frameon': False,
        # a tight bbox resizes the canvas around whatever labels are present,
        # so panels with two-line ticks come out taller than their neighbours
        'savefig.bbox': None,
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.02,
        'figure.constrained_layout.w_pad': 0.02,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def new_panel(width=None, height=None):
    return plt.subplots(figsize=(width or PANEL_WIDTH, height or PANEL_HEIGHT))


def save_panel(figure, name, formats=('pdf', 'jpg')):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in formats:
        path = FIGURE_DIR / f'{name}.{extension}'
        figure.savefig(path, dpi=400)
        written.append(str(path))
    plt.close(figure)
    return written


def outline_adopted_arm(axis, x_position, half_width=0.42):
    """Ring the arm that became the adopted protocol."""
    bottom, top = axis.get_ylim()
    axis.add_patch(plt.Rectangle(
        (x_position - half_width, bottom), 2 * half_width, top - bottom,
        fill=False, edgecolor=ADOPTED_COLOUR, lw=1.1, zorder=6, clip_on=False))


def grouped_roi_bars(axis, table, y_label):
    """One group of bars per ROI, one bar per arm, palest arm first. The bar
    belonging to the adopted protocol is outlined."""
    arms, rois = list(table.index), list(table.columns)
    positions = np.arange(len(rois))
    bar_width = 0.8 / len(arms)
    for index, arm in enumerate(arms):
        offset = (index - (len(arms) - 1) / 2) * bar_width
        shade = 0.35 + 0.65 * index / max(len(arms) - 1, 1)
        is_adopted = arm in ADOPTED_ARM_LABELS
        axis.bar(positions + offset, table.loc[arm], bar_width,
                 color=[ROI_COLOUR[roi] for roi in rois], alpha=shade,
                 edgecolor=ADOPTED_COLOUR if is_adopted else 'none',
                 linewidth=1.1 if is_adopted else 0, zorder=3)
    axis.set_xticks(positions)
    axis.set_xticklabels([ROI_SHORT[roi] for roi in rois])
    axis.set_ylabel(y_label)


def annotate_percent_change(axis, table, first_arm, second_arm):
    for index, roi in enumerate(table.columns):
        change = 100 * (table.loc[second_arm, roi] / table.loc[first_arm, roi] - 1)
        axis.text(index, table[roi].max() * 1.05, f'{change:+.0f}%',
                  ha='center', fontsize=9, color=ROI_COLOUR[roi])


# ------------------------------------------------------------ brain maps
def _mni_head():
    global MNI_HEAD
    if MNI_HEAD is None:
        MNI_HEAD = nib.load(MNI_2MM_HEAD).get_fdata()
    return MNI_HEAD


def _take_slice(volume, axis_index, slice_index):
    return [volume[slice_index], volume[:, slice_index],
            volume[:, :, slice_index]][axis_index]


def _mni_slice_index(mm, axis_index, reference_affine):
    return int(round((mm - reference_affine[axis_index, 3])
                     / reference_affine[axis_index, axis_index]))


def _roi_centroid_slice(roi, axis_index, hemisphere=None):
    """Every ROI here is bilateral, so a sagittal slice needs a hemisphere
    chosen first -- otherwise the centroid lands on the midline, where none of
    these structures are."""
    mask_image = nib.load(OUTPUT_DIR / 'rois' / 'MNI' / f'{roi}.nii.gz')
    mask = mask_image.get_fdata() > 0
    if hemisphere is not None:
        affine = mask_image.affine
        x_mm = affine[0, 0] * np.arange(mask.shape[0]) + affine[0, 3]
        keep = (x_mm < 0) if hemisphere == 'L' else (x_mm > 0)
        mask = mask & keep[:, None, None]
    return int(round(np.array(np.where(mask))[axis_index].mean()))


def map_block(runs, row_labels, name, title, overlay='tsnr', vmax=None,
              adopted_row=None):
    """Two rows (protocols) x two columns: sagittal through the left MTL,
    coronal through the frontal pair. Colour is laid over the MNI template in
    greyscale so a tilted, restricted slab reads as a slab rather than as a
    brain with pieces missing, and so ROI outlines have anatomy to sit on."""
    masks = {roi: nib.load(OUTPUT_DIR / 'rois' / 'MNI' / f'{roi}.nii.gz'
                           ).get_fdata() > 0 for roi in PRIMARY_ROIS}
    reference_affine = nib.load(
        OUTPUT_DIR / 'rois' / 'MNI' / 'HC.nii.gz').affine

    if overlay == 'tsnr':
        volumes = [nib.load(OUTPUT_DIR / 'tsnr' /
                            f'{sub}_{run}_tsnr_MNI.nii.gz').get_fdata()
                   for sub, run in runs]
        colourmap, colourbar_label = 'inferno', 'tSNR'
    else:
        volumes = [nib.load(OUTPUT_DIR / 'tsnr' /
                            f'displacement_{sub}_{run}_MNI.nii.gz').get_fdata()
                   for sub, run in runs]
        colourmap, colourbar_label = 'viridis', 'displacement (mm)'

    views = [(0, _roi_centroid_slice('HC', 0, hemisphere='L'), 'sagittal (MTL)'),
             (1, _mni_slice_index(38, 1, reference_affine), 'coronal (frontal)')]

    if vmax is None:
        # scale to the ROIs being argued about, not to the brightest cortex,
        # or the regions of interest wash out at the top of the ramp
        any_roi = np.zeros(volumes[0].shape, bool)
        for mask in masks.values():
            any_roi |= mask
        pooled = np.concatenate([v[any_roi & (v > 0)] for v in volumes])
        vmax = float(np.percentile(pooled, 95))

    figure, axes = plt.subplots(len(runs), 2,
                               figsize=(MAP_WIDTH, MAP_HEIGHT * len(runs) / 2),
                               constrained_layout=False)
    axes = np.atleast_2d(axes)
    head = _mni_head()
    for row, (volume, row_label) in enumerate(zip(volumes, row_labels)):
        for column, (axis_index, slice_index, view_name) in enumerate(views):
            axis = axes[row, column]
            background = np.rot90(_take_slice(head, axis_index, slice_index))
            axis.imshow(background, cmap='gray', vmin=0,
                        vmax=np.percentile(head[head > 0], 99.5),
                        interpolation='bilinear')
            overlay_slice = np.rot90(_take_slice(volume, axis_index, slice_index))
            axis.imshow(np.where(overlay_slice > 0, overlay_slice, np.nan),
                        cmap=colourmap, vmin=0, vmax=vmax,
                        interpolation='nearest', alpha=0.88)
            for roi, mask in masks.items():
                mask_slice = _take_slice(mask, axis_index, slice_index)
                if mask_slice.sum():
                    axis.contour(np.rot90(mask_slice), levels=[0.5],
                                 colors=ROI_COLOUR[roi], linewidths=0.8)
            axis.set_xticks([]); axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(view_name, fontsize=9, pad=2)
        axes[row, 0].set_ylabel(row_label, fontsize=9)
        if adopted_row is not None and row == adopted_row:
            for axis in axes[row]:
                for spine in axis.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(ADOPTED_COLOUR)
                    spine.set_linewidth(1.4)

    figure.subplots_adjust(wspace=.02, hspace=.04, right=.86)
    colourbar_axis = figure.add_axes([0.88, 0.15, 0.03, 0.7])
    mappable = plt.cm.ScalarMappable(cmap=colourmap,
                                     norm=plt.Normalize(0, vmax))
    colourbar = figure.colorbar(mappable, cax=colourbar_axis)
    colourbar.set_label(colourbar_label, fontsize=9)
    colourbar.ax.tick_params(labelsize=9)
    if title:
        figure.suptitle(title, fontsize=9, y=.99)
    return save_panel(figure, name)


def write_displacement_maps_in_mni(comparison='acceleration_split'):
    """Warp each arm's displacement map into MNI so map_block can render it."""
    import subprocess
    from .config import DERIVATIVES_DIR, FEAT_DIR_NAME
    from .metrics import displacement_map_mm
    readout = pd.read_csv(OUTPUT_DIR / 'tables' / 'readout_duration.csv'
                          ).set_index(['subject', 'run'])
    for subject, run, _ in COMPARISONS[comparison]['arms']:
        out = OUTPUT_DIR / 'tsnr' / f'displacement_{subject}_{run}_MNI.nii.gz'
        if out.exists():
            continue
        readout_s = float(readout.loc[(subject, run), 'readout_ms']) / 1e3
        session = COMPARISONS[comparison].get('session', '7T-s2')
        shift_mm, brain_mask, source_run = displacement_map_mm(
            subject, run, readout_s, session)
        feat_dir = DERIVATIVES_DIR / subject / source_run / 'func' / FEAT_DIR_NAME
        example_func = nib.load(feat_dir / 'reg' / 'example_func.nii.gz')
        native = OUTPUT_DIR / 'tsnr' / f'displacement_{subject}_{run}_native.nii.gz'
        nib.save(nib.Nifti1Image((shift_mm * brain_mask).astype(np.float32),
                                 example_func.affine), native)
        subprocess.run(
            ['applywarp', '-i', str(native),
             '-r', str(feat_dir / 'reg' / 'standard.nii.gz'),
             '-w', str(feat_dir / 'reg' / 'example_func2standard_warp.nii.gz'),
             '-o', str(out), '--interp=trilinear'],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


# ---------------------------------------------------------------- panels
def panel_field_bars(common):
    table = tsnr_by_arm(common, 'field_strength')
    figure, axis = new_panel()
    grouped_roi_bars(axis, table, 'tSNR')
    annotate_percent_change(axis, table, '3T', '7T')
    axis.set_ylim(0, table.values.max() * 1.22)
    return save_panel(figure, 'p_field_bars')


def panel_voxel_curve(common):
    """tSNR against voxel size, showing where the physiological-noise ceiling
    flattens the return on larger voxels."""
    figure, axis = new_panel()
    for comparison, linestyle, marker in [('voxel_size', '-', 'o'),
                                          ('voxel_size_fine', '--', 's')]:
        subset = common[(common.comparison == comparison)
                        & (common.roi.isin(PRIMARY_ROIS))]
        by_voxel = subset.groupby(['voxel_volume_mm3', 'roi'],
                                  as_index=False).tsnr_mean.mean()
        for roi in ['HC', 'mPFC']:
            points = by_voxel[by_voxel.roi == roi].sort_values('voxel_volume_mm3')
            axis.plot(points.voxel_volume_mm3, points.tsnr_mean, linestyle,
                      marker=marker, color=ROI_COLOUR[roi],
                      mfc='none' if linestyle == '--' else None)
    axis.set_xscale('log')
    axis.set_xticks([1.8, 3.4, 8.0, 15.8])
    axis.set_xticklabels(['1.2', '1.5', '2.0', '2.5'])
    axis.minorticks_off()
    axis.axvspan(7.2, 8.9, color=ADOPTED_COLOUR, alpha=.08, lw=0, zorder=0)
    axis.text(8.0, axis.get_ylim()[1] * 1.02, 'adopted', fontsize=9,
              color=ADOPTED_COLOUR, va='bottom', ha='center')
    axis.set_xlabel('voxel size (mm, isotropic)')
    axis.set_ylabel('tSNR')
    return save_panel(figure, 'p_voxel_curve')


def panel_angle(common, comparison, name):
    table = tsnr_by_arm(common, comparison)
    figure, axis = new_panel()
    grouped_roi_bars(axis, table, 'tSNR')
    axis.set_ylim(0, table.values.max() * 1.16)
    return save_panel(figure, name)


def panel_multiband(common):
    table = tsnr_by_arm(common, 'multiband')
    figure, axis = new_panel()
    grouped_roi_bars(axis, table, 'tSNR')
    annotate_percent_change(axis, table, 'MB3', 'MB6')
    axis.set_ylim(0, table.values.max() * 1.24)
    return save_panel(figure, 'p_multiband')


def panel_split_matched(common):
    """The decisive comparison: total acceleration held at 8, only the split
    between in-plane and through-plane differs."""
    table = tsnr_by_arm(common, 'acceleration_split')
    first, second = COMPARISONS['acceleration_split']['order']
    figure, axis = new_panel()
    grouped_roi_bars(axis, table, 'tSNR')
    annotate_percent_change(axis, table, first, second)
    axis.set_ylim(0, table.values.max() * 1.24)
    return save_panel(figure, 'p_split_matched')


def panel_acceleration_change(common):
    table = tsnr_by_arm(common, 'acceleration_sweep')
    relative = 100 * (table / table.iloc[0] - 1)
    positions = np.arange(len(table))
    figure, axis = new_panel()
    for roi in relative.columns:
        axis.plot(positions, relative[roi], '-o', color=ROI_COLOUR[roi])
        axis.annotate(ROI_SHORT[roi], (positions[-1], relative[roi].iloc[-1]),
                      xytext=(4, 0), textcoords='offset points', fontsize=9,
                      color=ROI_COLOUR[roi], va='center')
    axis.axhline(0, color='0.5', lw=.6, ls='--')
    axis.set_xticks(positions)
    axis.set_xticklabels(ACCELERATION_TICKS)
    axis.set_xlabel('iPAT / MB')
    axis.set_ylabel('tSNR change (%)')
    axis.set_xlim(-.3, len(table) + .55)
    outline_adopted_arm(axis, positions[-1], 0.34)
    return save_panel(figure, 'p_acceleration_change')


def panel_displacement(displacement):
    arms = COMPARISONS['acceleration_sweep']['arms']
    by_run = displacement.set_index(['subject', 'run'])
    values = np.array([by_run.loc[(sub, run), 'p99_shift_mm']
                       for sub, run, _ in arms])
    positions = np.arange(len(arms))
    colours = ['#B8B8B8'] * (len(arms) - 1) + [ADOPTED_COLOUR]
    figure, axis = new_panel()
    axis.bar(positions, values, color=colours, width=.62, edgecolor='none')
    for position, value in zip(positions, values):
        axis.text(position, value + .12, f'{value:.1f}', ha='center', fontsize=9)
    axis.set_xticks(positions)
    axis.set_xticklabels(ACCELERATION_TICKS)
    axis.set_xlabel('iPAT / MB')
    axis.set_ylabel('displacement (mm)')
    axis.set_ylim(0, values.max() * 1.25)
    outline_adopted_arm(axis, positions[-1])
    return save_panel(figure, 'p_displacement')


def panel_mofc_dropout(coverage):
    """The metric that actually moved: signal lost inside an acquired slab."""
    arms = COMPARISONS['acceleration_sweep']['arms']
    by_run = coverage.set_index(['subject', 'run', 'roi'])
    values = np.array([100 * by_run.loc[(sub, run, 'mOFC'), 'dropout_within_slab']
                       for sub, run, _ in arms])
    positions = np.arange(len(arms))
    colours = ['#B8B8B8'] * (len(arms) - 1) + [ADOPTED_COLOUR]
    figure, axis = new_panel()
    axis.bar(positions, values, color=colours, width=.62, edgecolor='none')
    for position, value in zip(positions, values):
        axis.text(position, value + .8, f'{value:.0f}', ha='center', fontsize=9)
    axis.set_xticks(positions)
    axis.set_xticklabels(ACCELERATION_TICKS)
    axis.set_xlabel('iPAT / MB')
    axis.set_ylabel('mOFC signal lost (%)')
    axis.set_ylim(0, max(values.max() * 1.22, 5))
    outline_adopted_arm(axis, positions[-1])
    return save_panel(figure, 'p_mofc_dropout')


def panel_tilt_prediction(penalties, roi):
    """Predicted signal retained against tilt, one line per sequence. This is
    the panel that carries the tilt argument: every sequence rises towards
    +30 deg, and the adopted one is both highest and flattest."""
    positions = np.arange(len(TILT_ANGLES_DEG))
    styles = [(list(TILT_REFERENCE_SEQUENCES)[0], '#C6C6C6', ':', 's'),
              (list(TILT_REFERENCE_SEQUENCES)[1], '#8A8A8A', '--', '^'),
              (list(TILT_REFERENCE_SEQUENCES)[2], ADOPTED_COLOUR, '-', 'o')]
    figure, axis = new_panel()
    for sequence_name, colour, linestyle, marker in styles:
        points = penalties[(penalties.roi == roi)
                           & (penalties.sequence == sequence_name)
                           ].set_index('tilt_deg').reindex(TILT_ANGLES_DEG)
        axis.plot(positions, 100 * points.net_signal, linestyle, marker=marker,
                  color=colour, lw=1.6 if 'adopted' in sequence_name else 1.1,
                  label=sequence_name.split(' (')[0])
    axis.set_xticks(positions)
    axis.set_xticklabels(['−30°', '0°', '+30°'])
    axis.set_xlabel('slab tilt')
    axis.set_ylabel('predicted signal kept')
    axis.set_ylim(60, 100)
    axis.set_yticks([60, 70, 80, 90, 100])
    axis.set_yticklabels(['60%', '70%', '80%', '90%', '100%'])
    # +30 deg is the adopted tilt, but none of these lines is a single acquired
    # run, so this is labelled rather than boxed
    adopted_position = list(TILT_ANGLES_DEG).index(ADOPTED_TILT_DEG)
    axis.axvline(adopted_position, color=ADOPTED_COLOUR, lw=.8, ls=':')
    axis.text(adopted_position, 61, 'adopted\ntilt', fontsize=9,
              color=ADOPTED_COLOUR, ha='center', va='bottom')
    axis.legend(fontsize=9, handlelength=1.4, labelspacing=.15, borderpad=.1,
                handletextpad=.4, loc='lower right')
    axis.set_title(roi, fontsize=9, pad=2)
    return save_panel(figure, f'p_tilt_prediction_{roi}')


def run(common, displacement, coverage, penalties, verbose=True):
    apply_style()
    written = []
    written += panel_field_bars(common)
    written += panel_voxel_curve(common)
    written += panel_angle(common, 'slice_angle_accelerated', 'p_angle_accelerated')
    written += panel_angle(common, 'slice_angle_unaccelerated', 'p_angle_unaccelerated')
    written += panel_multiband(common)
    written += panel_split_matched(common)
    written += panel_acceleration_change(common)
    written += panel_displacement(displacement)
    written += panel_mofc_dropout(coverage)
    for roi in ('EC', 'mOFC'):
        written += panel_tilt_prediction(penalties, roi)

    written += map_block(
        COMPARISONS['field_strength']['runs'], ['3T', '7T'], 'p_maps_field',
        '3T vs 7T, identical recipe (sub-02)')
    written += map_block(
        COMPARISONS['acceleration_split']['runs'],
        ['through-plane\nMB4 / iPAT2', 'in-plane\nMB2 / iPAT4'], 'p_maps_split',
        'Same total acceleration (MB x iPAT = 8), spent differently',
        adopted_row=1)
    write_displacement_maps_in_mni()
    written += map_block(
        COMPARISONS['acceleration_split']['runs'],
        ['through-plane\nMB4 / iPAT2\n27.8 ms readout',
         'in-plane\nMB2 / iPAT4\n14.2 ms readout'], 'p_distortion_maps',
        'Where signal is mis-placed, and by how far',
        overlay='displacement', vmax=6.0, adopted_row=1)

    if verbose:
        pdfs = [path for path in written if path.endswith('.pdf')]
        print(f'  wrote {len(pdfs)} vector panels to {FIGURE_DIR}')
    return written
