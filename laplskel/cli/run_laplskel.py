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
            'Apply a hard projection constraint to force nodes drifting out of the '
            'foreground mask onto the closest inner boundary shell surface voxel.'
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
        help='Convergence tolerance limit evaluated against mean vertex displacement.',
    )
    optional.add_argument(
        '--decimate_every',
        dest='decimate_every',
        type=int,
        default=1,
        help='Decimate nodes every N steps [Default=1].',
    )
    optional.add_argument(
        '--dec_grid_size',
        dest='min_edge_length',
        type=float,
        default=0.01,
        help=(
            'The Euclidean spatial threshold criteria below which two connected nodes '
            'undergo structural merging, expressed as a fraction of the isotropic '
            'voxel length.'
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
            'Number of graph hops used for each local tangent PCA neighborhood '
            '[Default=1].'
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
        default=6,
        help='Neighborhood connectivity structure for labeling (6, 18, or 26) [Default=6].',
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
        help='Downsample the original matrix to preserve RAM.',
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
