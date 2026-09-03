"""Medial-axis scoring used to weight positional retention during contraction."""

import numpy as np
from scipy import ndimage


def compute_medialness(edt_volume, node_coords, threshold=0.8):
    """
    Score how close each graph vertex sits to an inscribed-sphere centre.

    A vertex is maximally medial when its Euclidean Distance Transform value equals
    the largest value found in its 3x3x3 voxel neighborhood, which marks the centre
    of a locally maximal inscribed sphere. The raw neighborhood ratio is
    contrast-stretched so that only vertices at or around such centres score above
    zero, leaving boundary vertices unscored.

    The 3x3x3 footprint is equivalent to a one-hop maximum over a 26-connected voxel
    graph, but is evaluated on the volume so the score stays independent of the
    connectivity chosen for the initial graph.

    Parameters
    ----------
    edt_volume : numpy.ndarray
        The 3D Euclidean Distance Transform of the binary segmentation, in voxel units.
    node_coords : numpy.ndarray
        An (N, 3) array holding the 3D coordinates of the graph vertices. Coordinates
        are rounded to the nearest voxel and clipped to the volume before sampling.
    threshold : float, optional
        Lower cut of the contrast stretch. Vertices whose neighborhood ratio falls at
        or below this value score 0, while a ratio of 1 scores 1. Must lie in [0, 1).
        Default is 0.8.

    Returns
    -------
    medialness : numpy.ndarray
        An (N,) float array of medialness scores bounded to [0, 1].

    Raises
    ------
    ValueError
        If `edt_volume` is not three-dimensional, if `node_coords` does not have
        shape (N, 3), or if `threshold` lies outside [0, 1).
    """
    edt_volume = np.asarray(edt_volume, dtype=float)
    if edt_volume.ndim != 3:
        raise ValueError('edt_volume must be a 3D array.')

    coordinates = np.asarray(node_coords, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError('node_coords must have shape (N, 3).')

    if not 0.0 <= float(threshold) < 1.0:
        raise ValueError('threshold must lie in [0, 1).')

    voxel_positions = np.clip(
        np.rint(coordinates).astype(np.intp), 0, np.asarray(edt_volume.shape) - 1
    )
    lookup = tuple(voxel_positions.T)

    # Largest inscribed-sphere radius available within each voxel's neighborhood
    neighborhood_max = ndimage.maximum_filter(
        edt_volume, size=3, mode='constant', cval=0.0
    )

    ratios = edt_volume[lookup] / np.maximum(neighborhood_max[lookup], 1e-12)

    return np.clip((ratios - threshold) / (1.0 - threshold), 0.0, 1.0)
