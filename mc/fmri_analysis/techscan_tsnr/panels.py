"""
Stage 4: individual figure panels, one file each.

Panels are sized for the thesis layout: 4.0 x 3.5 cm for a plot covering a
third of the text width, 8.6 cm wide for the 2x2 brain-map blocks, nothing
below 9 pt Arial. At that size a panel can carry one idea, so each one does.

Everything is plotted as computed. Numbers come from the native-space tSNR
maps restricted to each comparison's common-coverage mask; brain maps are
shown in MNI space so that slices correspond between protocols, which is
stated in the captions.

Run:  python -m mc.fmri_analysis.techscan_tsnr.panels
"""
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from .config import (OUT, DERIV, FEAT, COMPARISONS, PRIMARY_ROIS, MNI2MM,
                     TSNR_USABLE, ADOPTED_LABELS)
from . import plotstyle as ps

FIG = OUT / 'figures'

# the four acceleration arms, short enough to fit a 4 cm panel at 9 pt
ACCEL_TICKS = ['2\nMB3', '2\nMB4', '3\nMB3', '4\nMB2']
REF = nib.load(MNI2MM)

# Whole-head MNI template as the greyscale underlay, matching the existing
# Figure 1 family. Laying colour over anatomy rather than over black is what
# stops a tilted, restricted slab from reading as "half a brain is missing",
# and it gives the ROI outlines something to sit on.
UNDERLAY = nib.load(str(MNI2MM).replace('_brain', '')).get_fdata()


# ------------------------------------------------------------------ helpers
def draw_underlay(ax, axis, idx):
    sl = np.rot90(take_slice(UNDERLAY, axis, idx))
    ax.imshow(sl, cmap='gray', vmin=0,
              vmax=np.percentile(UNDERLAY[UNDERLAY > 0], 99.5),
              interpolation='bilinear')


def box_adopted(ax, x, half_width=0.5, colour=None):
    """Ring the arm that became the adopted protocol, so every panel says
    which one was chosen without needing the caption."""
    lo, hi = ax.get_ylim()
    ax.add_patch(plt.Rectangle((x - half_width, lo), 2 * half_width, hi - lo,
                               fill=False, edgecolor=colour or ps.FINAL_MARK,
                               lw=1.1, zorder=6, clip_on=False))

def load_common():
    return pd.read_csv(OUT / 'tables' / 'tsnr_roi_common.csv')


def arm_means(comp, metric='tsnr_mean'):
    """metric per arm x ROI, averaged over participants where a group has >1."""
    c = load_common()
    d = c[(c.comparison == comp) & (c.roi.isin(PRIMARY_ROIS))]
    piv = (d.groupby(['arm', 'roi'], as_index=False)[metric].mean()
             .pivot(index='arm', columns='roi', values=metric))
    return piv.reindex(COMPARISONS[comp]['order'])[PRIMARY_ROIS]


def mni_index(mm, axis):
    a = REF.affine
    return int(round((mm - a[axis, 3]) / a[axis, axis]))


def roi_centroid_slice(roi, axis, hemi=None):
    """Centre-of-mass slice for an ROI. Every ROI here is bilateral, so for a
    sagittal slice one hemisphere has to be chosen first -- otherwise the
    centroid lands on the midline, where none of these structures are."""
    m = nib.load(OUT / 'rois' / 'MNI' / f'{roi}.nii.gz').get_fdata() > 0
    if hemi is not None:
        x_mm = REF.affine[0, 0] * np.arange(m.shape[0]) + REF.affine[0, 3]
        keep = (x_mm < 0) if hemi == 'L' else (x_mm > 0)
        m = m & keep[:, None, None]
    idx = np.array(np.where(m))
    return int(round(idx[axis].mean()))


def take_slice(vol, axis, idx):
    return [vol[idx], vol[:, idx], vol[:, :, idx]][axis]


def grouped_bars(ax, piv, colours, ylabel):
    """One group of bars per ROI, one bar per arm. The bar belonging to the
    adopted protocol is outlined, in every panel, so a reader can always find
    the sequence that was chosen."""
    arms = list(piv.index)
    rois = list(piv.columns)
    x = np.arange(len(rois))
    w = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        off = (i - (len(arms) - 1) / 2) * w
        shade = 0.35 + 0.65 * i / max(len(arms) - 1, 1)
        adopted = arm in ADOPTED_LABELS
        ax.bar(x + off, piv.loc[arm], w,
               color=[colours[r] for r in rois], alpha=shade,
               edgecolor=ps.FINAL_MARK if adopted else 'none',
               linewidth=1.1 if adopted else 0, zorder=3, label=arm)
    ax.set_xticks(x)
    ax.set_xticklabels([ps.ROI_DISPLAY[r].replace('Entorhinal', 'EC')
                        .replace('Hippocampus', 'HC') for r in rois])
    ax.set_ylabel(ylabel)
    return x, w


# ------------------------------------------------------- brain map panels
def map_block(runs, labels, out_name, title, vmax=None, rois=None,
              overlay='tsnr', adopted_row=None):
    """Two rows (protocols) x two columns: sagittal through the left MTL,
    coronal through the frontal pair. Colour is laid over the MNI template in
    greyscale, so the acquired slab reads as a slab rather than as a brain with
    pieces missing, and so the ROI outlines have anatomy to sit on."""
    rois = rois or PRIMARY_ROIS
    masks = {r: nib.load(OUT / 'rois' / 'MNI' / f'{r}.nii.gz').get_fdata() > 0
             for r in rois}
    if overlay == 'tsnr':
        vols = [nib.load(OUT / 'tsnr' / f'{s}_{r}_tsnr_MNI.nii.gz').get_fdata()
                for s, r in runs]
        cmap, cbl = 'inferno', 'tSNR'
    else:
        vols = [nib.load(OUT / 'tsnr' / f'_shift_{s}_{r}_MNI.nii.gz').get_fdata()
                for s, r in runs]
        cmap, cbl = 'viridis', 'displacement (mm)'

    sag = roi_centroid_slice('HC', 0, hemi='L')
    cor = mni_index(38, 1)
    views = [(0, sag, 'sagittal (MTL)'), (1, cor, 'coronal (frontal)')]

    if vmax is None:
        roi_any = np.zeros(REF.shape, bool)
        for m in masks.values():
            roi_any |= m
        vals = np.concatenate([v[roi_any & (v > 0)] for v in vols])
        vmax = float(np.percentile(vals, 95))

    fig, ax = plt.subplots(len(runs), 2,
                           figsize=(ps.MAP_W, ps.MAP_H * len(runs) / 2),
                           constrained_layout=False)
    ax = np.atleast_2d(ax)
    for i, (vol, lab) in enumerate(zip(vols, labels)):
        for j, (axis, idx, vname) in enumerate(views):
            a = ax[i, j]
            draw_underlay(a, axis, idx)
            img = np.rot90(take_slice(vol, axis, idx))
            a.imshow(np.where(img > 0, img, np.nan), cmap=cmap,
                     vmin=0, vmax=vmax, interpolation='nearest', alpha=0.88)
            for r, m in masks.items():
                s = take_slice(m, axis, idx)
                if s.sum():
                    a.contour(np.rot90(s), levels=[0.5],
                              colors=ps.ROI_COLOUR[r], linewidths=0.8)
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            if i == 0:
                a.set_title(vname, fontsize=9, pad=2)
        ax[i, 0].set_ylabel(lab, fontsize=9)
        if adopted_row is not None and i == adopted_row:
            for a in ax[i]:
                for sp in a.spines.values():
                    sp.set_visible(True)
                    sp.set_edgecolor(ps.FINAL_MARK)
                    sp.set_linewidth(1.4)

    fig.subplots_adjust(wspace=.02, hspace=.04, right=.86)
    cax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(cbl, fontsize=9)
    cb.ax.tick_params(labelsize=9)
    if title:
        fig.suptitle(title, fontsize=9, y=.99)
    return ps.save(fig, FIG / out_name)


# --------------------------------------------------------------- panels
def panel_field_bars():
    piv = arm_means('field_3T_vs_7T')
    fig, ax = ps.panel_fig()
    grouped_bars(ax, piv, ps.ROI_COLOUR, 'tSNR')
    for i, r in enumerate(piv.columns):
        pc = 100 * (piv.iloc[1][r] / piv.iloc[0][r] - 1)
        ax.text(i, max(piv[r]) * 1.04, f'{pc:+.0f}%', ha='center', fontsize=9,
                color=ps.ROI_COLOUR[r])
    ax.set_ylim(0, piv.values.max() * 1.22)
    return ps.save(fig, FIG / 'p_field_bars')


def panel_accel_tsnr():
    piv = arm_means('accel_7T_s2')
    xs = np.arange(len(piv))
    fig, ax = ps.panel_fig()
    for r in piv.columns:
        ax.plot(xs, piv[r], '-o', color=ps.ROI_COLOUR[r], label=r)
    ax.set_xticks(xs)
    ax.set_xticklabels(ACCEL_TICKS)
    ax.set_xlabel('iPAT / MB')
    ax.set_ylabel('tSNR')
    ax.set_ylim(0, piv.values.max() * 1.15)
    box_adopted(ax, xs[-1], 0.42)
    return ps.save(fig, FIG / 'p_accel_tsnr')


def panel_accel_change():
    piv = arm_means('accel_7T_s2')
    rel = 100 * (piv / piv.iloc[0] - 1)
    xs = np.arange(len(piv))
    fig, ax = ps.panel_fig()
    for r in rel.columns:
        ax.plot(xs, rel[r], '-o', color=ps.ROI_COLOUR[r])
        ax.annotate(r, (xs[-1], rel[r].iloc[-1]), xytext=(4, 0),
                    textcoords='offset points', fontsize=9,
                    color=ps.ROI_COLOUR[r], va='center')
    ax.axhline(0, color='0.5', lw=.6, ls='--')
    ax.set_xticks(xs); ax.set_xticklabels(ACCEL_TICKS)
    ax.set_xlabel('iPAT / MB')
    ax.set_ylabel('tSNR change (%)')
    ax.set_xlim(-.3, len(piv) + .55)
    box_adopted(ax, xs[-1], 0.34)
    return ps.save(fig, FIG / 'p_accel_change')


def panel_accel_usable():
    piv = arm_means('accel_7T_s2', metric='usable_frac') * 100
    fig, ax = ps.panel_fig()
    grouped_bars(ax, piv, ps.ROI_COLOUR, f'voxels tSNR>{TSNR_USABLE:.0f} (%)')
    ax.set_ylim(0, 112)
    return ps.save(fig, FIG / 'p_accel_usable')


def panel_readout():
    """Measured, not modelled: FEAT's own voxel-shift maps."""
    sh = pd.read_csv(OUT / 'tables' / 'shiftmaps.csv').set_index(['sub', 'run'])
    keys = [(s, r) for s, r, _ in COMPARISONS['accel_7T_s2']['arms']]
    v = sh.loc[keys, 'p99_shift_mm'].values
    xs = np.arange(len(keys))
    cols = ['#B8B8B8'] * (len(keys) - 1) + [ps.FINAL_MARK]
    fig, ax = ps.panel_fig()
    ax.bar(xs, v, color=cols, width=.62, edgecolor='none')
    ax.set_xticks(xs); ax.set_xticklabels(ACCEL_TICKS)
    ax.set_xlabel('iPAT / MB')
    ax.set_ylabel('displacement (mm)')
    ax.set_ylim(0, v.max() * 1.25)
    for x, y in zip(xs, v):
        ax.text(x, y + .12, f'{y:.1f}', ha='center', fontsize=9)
    box_adopted(ax, xs[-1], 0.42)
    return ps.save(fig, FIG / 'p_distortion_measured')


def panel_vofc_dropout():
    """The metric that actually moved: signal lost inside an acquired slab."""
    cov = pd.read_csv(OUT / 'tables' / 'coverage_decomposed.csv'
                      ).set_index(['sub', 'run', 'roi'])
    keys = [(s, r) for s, r, _ in COMPARISONS['accel_7T_s2']['arms']]
    xs = np.arange(len(keys))
    cols = ['#B8B8B8'] * (len(keys) - 1) + [ps.FINAL_MARK]
    v = np.array([100 * cov.loc[(s, r, 'vOFC'), 'dropout_within_slab']
                  for s, r in keys])
    fig, ax = ps.panel_fig()
    ax.bar(xs, v, color=cols, width=.62, edgecolor='none')
    for x, y in zip(xs, v):
        ax.text(x, y + .8, f'{y:.0f}', ha='center', fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(ACCEL_TICKS)
    ax.set_xlabel('iPAT / MB')
    ax.set_ylabel('vOFC signal lost (%)')
    ax.set_ylim(0, max(v.max() * 1.22, 5))
    box_adopted(ax, xs[-1], 0.42)
    return ps.save(fig, FIG / 'p_vofc_dropout')


def panel_voxel_curve():
    """tSNR against voxel volume, showing where the physiological-noise
    ceiling flattens the return on bigger voxels."""
    c = load_common()
    fig, ax = ps.panel_fig()
    for comp, mk, ls in [('voxel_7T_s2', 'o', '-'), ('voxel_7T_s1', 's', '--')]:
        d = c[(c.comparison == comp) & (c.roi.isin(PRIMARY_ROIS))]
        g = d.groupby(['vox_mm3', 'roi'], as_index=False).tsnr_mean.mean()
        for r in ['HC', 'mPFC']:
            gg = g[g.roi == r].sort_values('vox_mm3')
            ax.plot(gg.vox_mm3, gg.tsnr_mean, ls, marker=mk,
                    color=ps.ROI_COLOUR[r], mfc='none' if comp.endswith('s1') else None)
    ax.axvline(8.0, color=ps.FINAL_MARK, lw=.8, ls=':')
    ax.text(8.0, ax.get_ylim()[1] * 1.02, 'adopted', fontsize=9,
            color=ps.FINAL_MARK, va='bottom', ha='center')
    ax.set_xlabel('voxel volume (mm$^3$)')
    ax.set_ylabel('tSNR')
    ax.set_xscale('log')
    ax.set_xticks([1.8, 3.4, 8.0, 15.8])
    ax.set_xticklabels(['1.2', '1.5', '2.0', '2.5'])
    ax.minorticks_off()
    ax.axvspan(7.2, 8.9, color=ps.FINAL_MARK, alpha=.08, lw=0, zorder=0)
    ax.set_xlabel('voxel size (mm, isotropic)')
    return ps.save(fig, FIG / 'p_voxel_curve')


def panel_angle(comp='angle_AB_3T', out='p_angle'):
    """Tilt panels are shown per sequence, because the two sequences disagree
    and the disagreement is the point: the longer-readout sequence resolves an
    EC-vs-vOFC dissociation that the short-readout one does not."""
    piv = arm_means(comp)
    fig, ax = ps.panel_fig()
    grouped_bars(ax, piv, ps.ROI_COLOUR, 'tSNR')
    ax.set_ylim(0, piv.values.max() * 1.16)
    return ps.save(fig, FIG / out)


def panel_angle_leip():
    return panel_angle('angle_Leip_3T', 'p_angle_leip')


def panel_angle_dissociation():
    """The comparison that matters: tSNR change from transverse, for the two
    regions that pull in opposite directions, in the sequence whose readout is
    long enough for the effect to show."""
    fig, ax = ps.panel_fig()
    order = ['-30 deg', 'transverse', '+30 deg']
    xs = np.arange(3)
    for comp, ls, mk in [('angle_Leip_3T', '-', 'o'), ('angle_AB_3T', '--', 's')]:
        piv = arm_means(comp).reindex(order)
        for roi in ['EC', 'vOFC']:
            rel = 100 * (piv[roi] / piv.loc['transverse', roi] - 1)
            ax.plot(xs, rel, ls, marker=mk, color=ps.ROI_COLOUR[roi],
                    mfc='none' if ls == '--' else None)
            if ls == '-':          # label only the long-readout pair
                ax.annotate(roi, (xs[-1], rel.iloc[-1]), xytext=(4, 0),
                            textcoords='offset points', va='center',
                            fontsize=9, color=ps.ROI_COLOUR[roi])
    ax.axhline(0, color='0.5', lw=.6, ls=':')
    ax.set_xticks(xs)
    ax.set_xticklabels(['\u221230\u00b0', '0\u00b0', '+30\u00b0'])
    ax.set_xlabel('slab tilt')
    ax.set_ylabel('tSNR vs transverse (%)')
    ax.set_xlim(-.25, 2.55)
    return ps.save(fig, FIG / 'p_angle_dissociation')


def panel_multiband():
    piv = arm_means('multiband_7T_s1')
    fig, ax = ps.panel_fig()
    grouped_bars(ax, piv, ps.ROI_COLOUR, 'tSNR')
    for i, r in enumerate(piv.columns):
        pc = 100 * (piv.iloc[1][r] / piv.iloc[0][r] - 1)
        ax.text(i, max(piv[r]) * 1.04, f'{pc:+.0f}%', ha='center', fontsize=9,
                color=ps.ROI_COLOUR[r])
    ax.set_ylim(0, piv.values.max() * 1.24)
    return ps.save(fig, FIG / 'p_multiband')


def prepare_shift_volumes(arms, session):
    """Write each arm's phase-encode displacement map into MNI, so the map
    renderer can treat it like any other overlay."""
    from .shift_and_coverage import shift_map_mm
    import subprocess
    dist = pd.read_csv(OUT / 'tables' / 'distortion.csv'
                       ).set_index(['sub', 'run'])
    for sub, run, _ in arms:
        out = OUT / 'tsnr' / f'_shift_{sub}_{run}_MNI.nii.gz'
        if out.exists():
            continue
        readout = float(dist.loc[(sub, run), 'total_readout_ms']) / 1e3
        mm, bmask, src = shift_map_mm(sub, run, readout, session)
        feat = DERIV / sub / src / 'func' / FEAT
        ref = nib.load(feat / 'reg' / 'example_func.nii.gz')
        nat = OUT / 'tsnr' / f'_shift_{sub}_{run}_nat.nii.gz'
        nib.save(nib.Nifti1Image((mm * bmask).astype(np.float32), ref.affine),
                 nat)
        subprocess.run(
            ['applywarp', '-i', str(nat),
             '-r', str(feat / 'reg' / 'standard.nii.gz'),
             '-w', str(feat / 'reg' / 'example_func2standard_warp.nii.gz'),
             '-o', str(out), '--interp=trilinear'],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def panel_distortion_maps():
    """What geometric distortion is, and where it lives.

    Displacement = measured off-resonance (Hz, from this participant's own B0
    fieldmap) x readout duration. The field is the same in both rows, so the
    only thing that differs is the readout the protocol runs."""
    arms = [COMPARISONS['split_matched_7T_s2']['arms'][0],
            COMPARISONS['split_matched_7T_s2']['arms'][1]]
    prepare_shift_volumes(arms, '7T-s2')
    return map_block([(s, r) for s, r, _ in arms],
                     ['through-plane\nMB4 / iPAT2\n27.8 ms readout',
                      'in-plane\nMB2 / iPAT4\n14.2 ms readout'],
                     'p_distortion_maps',
                     'Where signal is mis-placed, and by how far',
                     vmax=6.0, overlay='shift', adopted_row=1)


def panel_split_matched():
    """The decisive comparison: total acceleration held at 8, only the split
    between in-plane and through-plane differs."""
    piv = arm_means('split_matched_7T_s2')
    fig, ax = ps.panel_fig()
    grouped_bars(ax, piv, ps.ROI_COLOUR, 'tSNR')
    for i, r in enumerate(piv.columns):
        pc = 100 * (piv.iloc[1][r] / piv.iloc[0][r] - 1)
        ax.text(i, max(piv[r]) * 1.05, f'{pc:+.0f}%', ha='center', fontsize=9,
                color=ps.ROI_COLOUR[r])
    ax.set_ylim(0, piv.values.max() * 1.24)
    return ps.save(fig, FIG / 'p_split_matched')


def panel_split_maps():
    return map_block(
        [(s, r) for s, r, _ in COMPARISONS['split_matched_7T_s2']['arms']],
        ['through-plane\nMB4 / iPAT2', 'in-plane\nMB2 / iPAT4'],
        'p_maps_split',
        'Same total acceleration (MB x iPAT = 8), spent differently',
        adopted_row=1)


def panel_bold_sensitivity():
    """tSNR is not BOLD sensitivity. Signal change per unit BOLD scales as
    TE*exp(-TE/T2*), which peaks at TE = T2*. A short TE therefore wins only
    where T2* is short -- exactly the dropout regions -- and loses elsewhere.
    This is the honest counterweight to the short-TE argument."""
    t2 = np.linspace(5, 45, 300)
    tes = [(12.8, 'in-plane\nMB2/iPAT4', ps.FINAL_MARK, 2.0),
           (20.6, 'through-plane\nMB4/iPAT2', '#9A9A9A', 1.4)]
    fig, ax = ps.panel_fig()
    curves = {}
    for te, lab, col, lw in tes:
        s = te * np.exp(-te / t2)
        curves[te] = s
        ax.plot(t2, s, color=col, lw=lw, label=f'TE {te} ms')
    # the crossover: below this T2*, the shorter TE gives more BOLD contrast
    cross = t2[np.argmin(np.abs(curves[12.8] - curves[20.6]))]
    top = max(c.max() for c in curves.values()) * 1.28
    ax.set_ylim(0, top)
    ax.axvline(cross, color='0.5', lw=.7, ls=':')
    ax.annotate(f'{cross:.0f} ms', (cross, top * .012), xytext=(-3, 0),
                textcoords='offset points', ha='right', va='bottom',
                fontsize=9, color='0.35')
    # typical T2* at 7T; vOFC is shortened by the frontal sinus
    for x, nm in [(20, 'vOFC'), (33, 'cortex')]:
        ax.plot([x], [top * .045], marker='v', color='0.25', ms=4)
        ax.text(x, top * .105, nm, fontsize=9, ha='center', color='0.25')
    ax.set_xlabel('T2* (ms)')
    ax.set_ylabel('BOLD sensitivity')
    ax.set_xlim(5, 45)
    ax.legend(handlelength=.9, loc='upper left', borderpad=.1,
              labelspacing=.15, handletextpad=.4,
              bbox_to_anchor=(-.02, 1.04))
    return ps.save(fig, FIG / 'p_bold_sensitivity')


def main():
    ps.apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    made += panel_field_bars()
    made += panel_accel_tsnr()
    made += panel_accel_change()
    made += panel_accel_usable()
    made += panel_voxel_curve()
    made += panel_angle()
    made += panel_angle_leip()
    made += panel_angle_dissociation()
    made += panel_multiband()
    made += panel_readout()
    made += panel_vofc_dropout()
    made += panel_split_matched()
    made += panel_split_maps()
    made += panel_bold_sensitivity()
    made += panel_distortion_maps()
    # brain maps
    made += map_block([('sub-02', 'AB_sequ_transv'), ('sub-02', 'run1_T7_AB_rep')],
                      ['3T', '7T'], 'p_maps_field',
                      '3T vs 7T, identical recipe (sub-02)')
    made += map_block([('sub-03', 'run1_3Trep_7T2'), ('sub-03', 'run5_ipat4_7T2')],
                      ['iPAT2\nTE 19.6', 'iPAT4\nTE 12.8'], 'p_maps_accel',
                      'Least vs most in-plane acceleration (sub-03)')
    for m in made:
        print('wrote', m)


if __name__ == '__main__':
    main()
