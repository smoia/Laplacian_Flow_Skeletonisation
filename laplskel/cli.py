"""Command-line interface for Laplacian skeletonisation."""

import argparse
import sys

from .pipeline import laplacian_skeletonisation


def get_parser():
    """
    Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured command-line parser.
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
        help='Path destination for the generated skeleton outputs.',
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
            'Flag to disable anisotropic constraints and fall back to standard '
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
        '--tol',
        type=float,
        default=0.05,
        help='Convergence tolerance limit evaluated against mean vertex displacement.',
    )
    optional.add_argument(
        '--decimate_every',
        dest='decimate_every',
        type=int,
        default=2,
        help='Decimate nodes every N steps [Default=2].',
    )
    optional.add_argument(
        '--dec_grid_size',
        dest='min_edge_length',
        type=float,
        default=0.5,
        help=(
            'The Euclidean spatial threshold criteria below which two connected nodes '
            'undergo structural merging, i.e. the isotropic voxel size of the grid used'
            ' for decimation.'
        ),
    )
    optional.add_argument(
        '--max_distance_adjmat',
        dest='max_distance',
        type=float,
        default=2.4999,
        help=(
            'Maximum distance to consider when computing the sparse adjacency matrix.'
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
        choices=[6, 18, 26],
        default=6,
        help='Neighborhood connectivity structure for labeling (6, 18, or 26) [Default=6].',
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


def main(argv=None):
    """Run the command-line skeletonisation pipeline."""
    args = get_parser().parse_args(argv)
    laplacian_skeletonisation(**vars(args))


if __name__ == '__main__':
    main(sys.argv[1:])
