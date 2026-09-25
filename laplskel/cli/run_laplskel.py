"""Command-line interface for Laplacian skeletonisation."""

import argparse
import sys

from laplskel.workflows import laplacian_skeletonisation

VALID_CONNECTIVITY = (6, 18, 26)
VALID_SOLVER = ('LU', 'CG', 'AMGCG')


def _positive_integer(value):
    """Parse a strictly positive integer argument."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError('must be a positive integer') from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return parsed


def _get_parser():
    """
    Parse command line inputs for this function.

    Returns
    -------
    parser.parse_args() : argparse dict

    """
    parser = argparse.ArgumentParser(
        description='Configurable EDT-Guided Laplacian Graph Contraction Pipeline.',
        add_help=False,
    )

    required = parser.add_argument_group('Required Arguments')
    required.add_argument(
        '--input',
        '-i',
        dest='nifti_path',
        type=str,
        required=True,
        help='Path pointing toward the input .nii or .nii.gz file volume.',
    )

    optional = parser.add_argument_group('Other Optional Arguments')
    optional.add_argument(
        '--output',
        '-o',
        dest='out_path',
        type=str,
        default=None,
        help='Path destination for the generated skeleton arrays.',
    )
    optional.add_argument(
        '--graphml',
        action='store_true',
        default=argparse.SUPPRESS,
        help=(
            'Write the converged graph as GraphML instead of coordinate and '
            'adjacency NPZ files.'
        ),
    )
    optional.add_argument(
        '--dev_contra_graph',
        action='store_true',
        help=(
            'Also write the contracted graph before thinning as '
            '<output_stem>_intermediate.graphml. Ignored with --alternating.'
        ),
    )
    optional.add_argument(
        '--alternating',
        action='store_true',
        help=(
            'Use the experimental fixed-schedule contraction/thinning workflow '
            'with original-foreground EDT ridge guidance.'
        ),
    )
    optional.add_argument(
        '--alter_init_thinning',
        action='store_true',
        help=(
            'With --alternating, build an initial branch reference and enable '
            'reference preservation, critical-node movement caps and containment. '
            'Without this flag, use default thinning rules and free graph fitting.'
        ),
    )
    optional.add_argument(
        '--contraction_steps',
        type=_positive_integer,
        default=5,
        help='Maximum contraction iterations per --alternating cycle [Default=5].',
    )
    optional.add_argument(
        '--use_edt',
        action='store_true',
        help=(
            'Use distance transform potential constraints on top of classic uniform '
            'retention mapping.'
        ),
    )
    optional.add_argument(
        '--use_anisotropic',
        action='store_true',
        help=(
            'Flag to enable Laplacian weighting, '
            'favoring cross-sectional contraction while reducing longitudinal contraction.'
            'Omitting this flag will disable anisotropic constraints and fall back to standard '
            'isotropic Laplacian matrix operations.'
        ),
    )
    optional.add_argument(
        '--enforce_containment',
        action='store_true',
        help=(
            'Project escaped nodes to the nearest foreground voxel before each '
            'contraction update and after its solve.'
        ),
    )
    optional.add_argument(
        '--beta_edt',
        type=float,
        default=1.0,
        help=(
            'Custom scaling weight assigned to modulate the EDT boundary energy '
            'constraints.'
        ),
    )
    optional.add_argument(
        '--w_L',
        type=float,
        default=0.5,
        help='Contraction weight scalar multiplier variable.',
    )
    optional.add_argument(
        '--w_H',
        dest='w_H_base',
        type=float,
        default=0.5,
        help='Baseline structural anchor retention weight variable.',
    )
    optional.add_argument(
        '--w_H_medial',
        dest='w_H_medial',
        type=float,
        default=1.0,
        help=(
            'Retention weight boost applied to nodes at or around inscribed-sphere '
            'centres, tightening the centreline onto the medial axis. Raised to the '
            'power of each node medialness score, so boundary nodes keep their '
            'baseline weight. 1.0 disables the boost [Default=1.0].'
        ),
    )
    optional.add_argument(
        '--tol',
        type=float,
        default=0.05,
        help='Maximum contraction movement in voxels, required for three stable steps.',
    )
    optional.add_argument(
        '--decimate_every',
        dest='decimate_every',
        type=int,
        default=200,
        help='Run default-workflow graph decimation every N steps [Default=200].',
    )
    optional.add_argument(
        '--dec_grid_size',
        dest='min_edge_length',
        type=float,
        default=0.01,
        help=(
            'Legacy edge-decimation threshold retained for compatibility; it does '
            'not affect default-workflow graph decimation.'
        ),
    )
    optional.add_argument(
        '--dev-elongation-ratio',
        '--dev_elongation_ratio',
        dest='dev_elongation_ratio',
        type=float,
        default=25.0,
        help=(
            'Developmental PCA threshold for projecting degree-five '
            'neighbourhoods onto chains. Cycles use a fixed cutoff of 25 '
            '[Default=25].'
        ),
    )
    optional.add_argument(
        '--dev_edge_thresh',
        type=float,
        default=0.05,
        help=(
            'Merge connected edges shorter than this fraction of the smallest '
            'voxel spacing [Default=0.05]. Zero disables optional short-edge '
            'merging; connected coincident nodes still merge.'
        ),
    )
    optional.add_argument(
        '--dev_peri_ratio',
        type=float,
        default=1.0,
        help=(
            'Cycle perimeter limit as a multiple of the smallest voxel-face '
            'perimeter [Default=1.0]. Larger values admit larger cycles, which '
            'must also have a PCA ratio below the fixed cutoff of 25.'
        ),
    )
    optional.add_argument(
        '--merge_tolerance',
        type=float,
        default=0.25,
        help=(
            'Maximum branch simplification error in voxels [Default=0.25]. '
            '0 retains reference sampling; every foreground tunnel is preserved '
            'at every setting using 26-connectivity. Higher values allow more aggressive simplification, '
            'subject to foreground and topology constraints.'
        ),
    )
    optional.add_argument(
        '--init_graph_adj',
        type=int,
        choices=VALID_CONNECTIVITY,
        default=26,
        help=(
            'Voxel-neighborhood connectivity for the initial graph (6, 18, or 26) '
            '[Default=26].'
        ),
    )
    optional.add_argument(
        '--local_pca_hops',
        type=_positive_integer,
        default=1,
        help=(
            'Graph hops used for local tangent PCA at nodes with at least two '
            'neighbors; terminal nodes use their incident edge [Default=1].'
        ),
    )
    optional.add_argument(
        '--separate_streams',
        action='store_true',
        help=(
            'Separate segmentation components into individual labels using '
            'scipy.ndimage.label and run contraction on each independently.'
        ),
    )
    optional.add_argument(
        '--label_connectivity',
        type=int,
        choices=VALID_CONNECTIVITY,
        default=26,
        help='Component labeling connectivity; use 26 for tunnel preservation [Default=26].',
    )
    optional.add_argument(
        '--solver',
        type=str,
        choices=VALID_SOLVER,
        default='CG',
        help=(
            'The solver to use to solve the linear system in computing the new '
            'coordinates system. LU uses SuperLU, a direct solver, CG uses Conjugate '
            'Gradient (iterative solver), better for memory performance on big data, '
            'AMGCG constructs an Algebraic Multigrid (AMG) preconditioner before '
            'running CG, far faster (but may be a bit more memory demanding than pure '
            'CG). Default is AMGCG.'
        ),
    )
    optional.add_argument(
        '--n_jobs',
        type=int,
        default=None,
        help=(
            'Number of parallel jobs. If not set or <=0, defaults to '
            '~30%% of available CPU cores.'
        ),
    )
    optional.add_argument(
        '--downsample',
        action='store_true',
        help='Unsupported with strict tunnel preservation; using this flag raises an error.',
    )
    optional.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Base random seed for reproducible downsampling [Default=42].',
    )
    optional.add_argument(
        '-h', '--help', action='help', help='Show this help message and exit'
    )
    return parser


def _main(argv=None):
    args = _get_parser().parse_args(argv)

    laplacian_skeletonisation(**vars(args))


if __name__ == '__main__':
    _main(sys.argv[1:])
