"""Parallel component processing and shared-volume lifecycle."""

import os
import tempfile
from contextlib import contextmanager

import numpy as np
from joblib import delayed
from scipy.spatial import KDTree
from tqdm_joblib import ParallelPbar

from .contraction import laplacian_graph_contraction_edt
from .graph import compute_sparse_adjacency_matrix


def _process_single_label(
    label_id,
    labeled_volume_mmap_path,
    lab_vol_shape,
    use_edt,
    use_anisotropic,
    enforce_containment,
    beta_edt,
    w_L,
    w_H_base,
    tol,
    max_distance,
    decimate_every,
    min_edge_length,
    num_features,
):
    """
    Worker function to process a single connected component label.

    Parameters
    ----------
    label_id : int
        ID of current label
    labeled_volume_mmap_path : path
        Path to memmap of labelled volume.
    lab_vol_shape : tuple
        Shape of labeled volume.
    use_edt : bool
        Enables boundary tracking potential constraints using Euclidean Distance Transforms.
    use_anisotropic : bool
        Enables anisotropic geometry handling to penalize internal longitudinal
        shortening vectors.
    enforce_containment : bool
        If True, applies a hard projection constraint to force nodes drifting out of the
        foreground mask onto the closest inner boundary shell surface voxel.
    beta_edt : float
        Scaling modulation weight assigned to boundary energy calculation properties.
    w_L : float
        Contraction weight step modifier targeting structural local geometric collapse.
        This should be alpha in Damseh 2021.
    w_H_base : float
        Baseline structural node anchor positional persistence value metric.
        This should be equivalent to beta in Damseh 2021.
    tol : float
        Convergence tolerance limit evaluated against mean vertex displacement.
        This should be the equivalent of gamma in Damseh 2021 (not sure).
    max_distance : float
        Maximum distance to consider when making the sparse adjacency matrix.
    decimate_every : int
        Frequency cadence interval defining how many contraction loop steps occur before
        triggering an edge-collapse decimation execution.
    min_edge_length : float
        The Euclidean spatial threshold criteria below which two connected nodes undergo
        structural merging, i.e. the isotropic voxel size of the grid used for
        decimation.
    num_features : int
        Number of extracted labels.

    Returns
    -------
    label_id : int
        ID of current label (For tracking)
    contracted_X : numpy.ndarray
        An (M, 3) matrix mapping the continuous 3D spatial points along the skeleton path.
    final_adj : scipy.sparse.csr_matrix
        The resulting graph sparse adjacency connectivity representation of shape (M, M).
    """
    labeled_volume = np.memmap(
        labeled_volume_mmap_path, dtype=np.int32, mode='r', shape=lab_vol_shape
    )
    X_init = np.argwhere(labeled_volume == label_id).astype(np.int16)
    tree = KDTree(X_init)

    # Skip small noise components
    if len(X_init) < 3:
        adj_sparse = compute_sparse_adjacency_matrix(tree, max_distance)

        return label_id, X_init, adj_sparse

    print(
        f'\n--- Processing Label {label_id}/{num_features} ({X_init.sum()} voxels) ---'
    )

    print('Computing proximity network coordinates...')
    adj_sparse = compute_sparse_adjacency_matrix(tree, max_distance)

    # Run contraction on this label's component mask
    label_X, label_adj = laplacian_graph_contraction_edt(
        X_init,
        adj_sparse,
        binary_segmentation=labeled_volume == label_id,
        use_edt=use_edt,
        use_anisotropic=use_anisotropic,
        enforce_containment=enforce_containment,
        beta_edt=beta_edt,
        w_L=w_L,
        w_H_base=w_H_base,
        tol=tol,
        decimate_every=decimate_every,
        min_edge_length=min_edge_length,
    )

    return label_id, label_X, label_adj


@contextmanager
def shared_labeled_volume(labeled_volume):
    """Expose a labeled volume to workers through a temporary memmap."""
    temp_dir = tempfile.mkdtemp()
    labeled_volume_mmap_path = os.path.join(temp_dir, 'labeled_vol.dat')
    lab_vol_shape = labeled_volume.shape
    fp = np.memmap(
        labeled_volume_mmap_path,
        dtype=np.int32,
        mode='w+',
        shape=lab_vol_shape,
    )
    fp[:] = labeled_volume[:]
    fp.flush()
    del fp

    try:
        yield labeled_volume_mmap_path, lab_vol_shape
    finally:
        print('Clean up temp files')
        if os.path.exists(labeled_volume_mmap_path):
            os.remove(labeled_volume_mmap_path)
        os.rmdir(temp_dir)


def process_components(
    labeled_volume_mmap_path,
    lab_vol_shape,
    num_features,
    use_edt,
    use_anisotropic,
    enforce_containment,
    beta_edt,
    w_L,
    w_H_base,
    tol,
    max_distance,
    decimate_every,
    min_edge_length,
    n_jobs,
):
    """Process labeled segmentation components in parallel."""
    total_cores = os.cpu_count() or 1
    if n_jobs is None or n_jobs <= 0:
        n_workers = max(1, int(np.floor(0.30 * total_cores)))
    else:
        n_workers = min(n_jobs, total_cores)

    print(
        f'Processing {num_features} components in parallel using {n_workers} worker(s) '
        f'on {total_cores} CPU cores detected.'
    )

    results = ParallelPbar('Skeletonising')(n_jobs=n_workers)(
        delayed(_process_single_label)(
            label_id,
            labeled_volume_mmap_path,
            lab_vol_shape,
            use_edt,
            use_anisotropic,
            enforce_containment,
            beta_edt,
            w_L,
            w_H_base,
            tol,
            max_distance,
            decimate_every,
            min_edge_length,
            num_features,
        )
        for label_id in range(1, num_features + 1)
    )

    return results
