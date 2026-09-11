"""
Figure conventions for the techscan tSNR panels.

Sized for A4 subpanels of 6-8 cm, Arial, 11 pt titles down to 8 pt tick labels,
so the panels stay legible at roughly an eighth of a page. ROI colours follow
the project-wide mapping (era_brewer "Showgirl2"); V1 is deliberately grey,
because it is the control region and should read as one.
"""
import matplotlib
import matplotlib.pyplot as plt

CM = 1 / 2.54

# era_brewer Showgirl2, n=7 -> the project's fixed ROI assignment
SHOWGIRL2 = ['#B74C2D', '#448363', '#CCB178', '#C1DCBF',
             '#DC673E', '#7BB594', '#C1DCBF']

ROI_COLOUR = {
    'mPFC':   SHOWGIRL2[1],   # index 1, per the project convention
    'EC':     SHOWGIRL2[0],   # index 0
    'HC':     '#23677E',      # the project's hippocampus teal
    'vOFC':   '#a30d6c',      # the project's lOFC magenta -- the other frontal ROI
    'MTL_gv': '#23677E',
    'EC_gv':  SHOWGIRL2[0],
    'HC_gv':  '#23677E',
}
ROI_DISPLAY = {'mPFC': 'mPFC', 'EC': 'Entorhinal', 'HC': 'Hippocampus',
               'vOFC': 'vOFC', 'MTL_gv': 'MTL (Garvert)',
               'EC_gv': 'EC (Garvert)', 'HC_gv': 'HC (Garvert)'}
# hypothesis priority: mPFC first, then EC, then HC; vOFC knowingly sacrificed
ROI_ORDER = ['mPFC', 'EC', 'HC', 'vOFC']

# dark green, the project's "observed value" marker, used here to flag the
# protocol that was actually adopted
FINAL_MARK = '#0e3d3a'


# Panel geometry. A single panel covering a third of an A4 text column is
# 4.0 x 3.5 cm; anything smaller shrinks from there. Nothing is allowed below
# 9 pt Arial, which at this size is the binding constraint on how much a panel
# can carry -- so panels stay single-purpose and labels stay short.
PANEL_W, PANEL_H = 5.4 * CM, 3.9 * CM
MAP_W, MAP_H = 8.6 * CM, 7.0 * CM     # 2x2 brain-map blocks, half text width


def panel_fig(w=None, h=None):
    """A plot panel on a fixed canvas, so every panel exports at exactly the
    same physical size regardless of how many lines its tick labels take."""
    return plt.subplots(figsize=(w or PANEL_W, h or PANEL_H))


def apply_style():
    matplotlib.rcParams.update({
        'font.family': 'Arial',
        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2,
        'ytick.major.size': 2,
        'axes.labelpad': 2,
        'xtick.major.pad': 2,
        'ytick.major.pad': 2,
        'lines.linewidth': 1.2,
        'lines.markersize': 3,
        'legend.frameon': False,
        # NOT 'tight': a tight bbox resizes the canvas around whatever
        # labels happen to be there, so panels with two-line tick labels
        # come out taller than their neighbours. A fixed canvas plus
        # constrained_layout keeps every panel the same physical size,
        # which is what matters when they are laid out side by side.
        'savefig.bbox': None,
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.02,
        'figure.constrained_layout.w_pad': 0.02,
        'savefig.dpi': 400,
        'pdf.fonttype': 42,     # keep text as text in vector output
        'ps.fonttype': 42,
    })


def region_of(roi):
    """ROIs are already bilateral; kept so callers need not care."""
    return roi


def save(fig, path_no_ext, formats=('pdf', 'jpg')):
    """Panels are saved individually, PDF first.

    In the PDF all text and vector artwork stay vector -- `pdf.fonttype = 42`
    embeds Arial as TrueType rather than converting glyphs to paths, so the
    labels remain selectable and editable in Illustrator. Brain images are
    genuine rasters and are embedded at 400 dpi. The .jpg alongside is only a
    convenience copy for quick viewing and for the review page."""
    out = []
    for f in formats:
        fig.savefig(f'{path_no_ext}.{f}', dpi=400)
        out.append(f'{path_no_ext}.{f}')
    plt.close(fig)
    return out
