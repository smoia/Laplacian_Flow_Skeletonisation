"""NIfTI input/output and skeleton rasterization."""

import os

import numpy as np
from nigsp import io
from scipy import sparse


def load_nifti_mask(nifti_path):
    """Load a three-dimensional NIfTI segmentation mask."""
    return io.load_nifti_get_mask(nifti_path, is_mask=True, ndim=3)


def coords_to_dense_3d(X, volume_shape):
    """
    Create and fill in a volume using coordinates of points.

    Parameters
    ----------
    X : ndarray of shape (N, 3)
        The 3D coordinates of the areas with content.
    volume_shape : tuple of int (D, H, W)
        The structural grid dimensions of the target 3D matrix.

    Returns
    -------
    dense_volume : ndarray of shape (D, H, W)
        A binary 3D array where 1 represents the skeleton path.
    """
    # 1. Initialize empty dense matrix
    dense_volume = np.zeros(volume_shape, dtype=bool)

    coords = np.rint(X).astype(np.int8)

    # 2. Fix coordinates on boundaries due to numpy's round-to-even
    for dim, bound in enumerate(volume_shape):
        coords[:, dim][coords[:, dim] == bound] = bound - 1

    # 3. Rasterize edges and nodes into the grid
    for i in coords:
        dense_volume[tuple(i)] = True

    return dense_volume


def get_output_path(nifti_path, out_path):
    """Return an explicit or input-derived skeleton output path."""
    if out_path:
        return out_path
    return f'{os.path.splitext(os.path.splitext(nifti_path)[0])[0]}_skel'


def save_skeleton(
    out_path,
    contracted_X,
    final_adj,
    volume_data,
    img,
    enforce_containment,
):
    """Save coordinate, adjacency, and dense NIfTI skeleton outputs."""
    print(f'\nSaving structural centerline data matrices to: {out_path}')
    np.savez_compressed(f'{out_path}_coords.npz', contracted_X=contracted_X)
    sparse.save_npz(f'{out_path}.npz', final_adj)
    nifti_skel = coords_to_dense_3d(contracted_X, volume_data.shape)

    # If enforce containment was used, assume no loss of tracts masking with original segmentation.
    if enforce_containment:
        nifti_skel = nifti_skel * volume_data

    io.export_nifti(nifti_skel, img, f'{out_path}.nii.gz')
    return nifti_skel
