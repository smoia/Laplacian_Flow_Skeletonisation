#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
from joblib import delayed
from nigsp import io
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg, spsolve
from scipy.spatial import KDTree
from tqdm_joblib import ParallelPbar


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


def compute_laplacian_matrix(
    X, adjacency_matrix, use_anisotropic=True, alpha_norm=1.5, alpha_tang=0.1
):
    """
    Compute the Graph Laplacian Matrix L = D - W.

    Supports toggling between
    -------------------------
    1. Standard (Isotropic) Laplacian: Purely reciprocal Euclidean distance weights.
    2. Anisotropic Laplacian: Multiplies distance affinity by directional alignment factors
       to penalize longitudinal shrinkage while favoring radial cross-sectional collapse.

    Parameters
    ----------
    X : numpy.ndarray
        An (N, 3) array containing the continuous 3D spatial coordinates of the
        graph vertices/nodes.
    adjacency_matrix : scipy.sparse.spmatrix
        A sparse binary adjacency matrix of shape (N, N) defining the structural
        connectivity profile between the vertices.
    use_anisotropic : bool, optional
        If True, modulates affinity weights using localized directional alignment vectors
        to encourage radial over longitudinal contraction. If False, defaults to classic
        isotropic Euclidean distance reciprocals. Default is True.
    alpha_norm : float, optional
        The scaling coefficient penalty assigned to normal (cross-sectional radial)
        displacement components when `use_anisotropic` is active. Default is 1.5.
    alpha_tang : float, optional
        The scaling coefficient penalty assigned to tangential (longitudinal direction)
        displacement components when `use_anisotropic` is active. Default is 0.1.

    Returns
    -------
    L : scipy.sparse.csr_matrix
        The calculated sparse Graph Laplacian Matrix of shape (N, N) governed
        by the equation L = D - W.
    """
    n_vertices = X.shape[0]

    # Get row and col indices from the sparse adjacency matrix
    rows, cols = adjacency_matrix.nonzero()

    # 1. Compute spatial difference vectors and Euclidean distances
    diffs = X[rows] - X[cols]
    distances = np.linalg.norm(diffs, axis=1)
    distances = np.maximum(distances, 1e-6)  # Prevent division by zero

    if use_anisotropic:
        # Estimate local structural tangents using local neighborhood PCA proxy
        tangents = np.zeros_like(X)

        indptr = adjacency_matrix.indptr
        indices = adjacency_matrix.indices

        for i in range(n_vertices):
            nbr_idx = indices[indptr[i] : indptr[i + 1]]
            if len(nbr_idx) > 2:
                pts = X[nbr_idx] - X[i]
                # Fast 3x3 SVD / Eigen decomposition without np.cov overhead
                _, _, vh = np.linalg.svd(pts, full_matrices=False)
                tangents[i] = vh[0]  # Principal direction
            else:
                tangents[i] = np.array([1.0, 0.0, 0.0])

        t_i = tangents[rows]
        dot_products = np.sum(diffs * t_i, axis=1)

        # Decompose into longitudinal and cross-sectional components
        tangential_comps = np.abs(dot_products)
        normal_comps = np.linalg.norm(diffs - (dot_products[:, None] * t_i), axis=1)

        # Scale the affinity weights using the anisotropy parameters
        aniso_mod = (alpha_norm * normal_comps) + (alpha_tang * tangential_comps)
        weights = aniso_mod / distances
    else:
        # Standard Isotropic Weights
        weights = 1.0 / distances

    # Assemble sparse operators
    W = sparse.csr_matrix((weights, (rows, cols)), shape=(n_vertices, n_vertices))

    # Build diagonal degree matrix D
    degree_values = np.array(W.sum(axis=1)).flatten()
    D = sparse.csr_matrix(
        (degree_values, (range(n_vertices), range(n_vertices))),
        shape=(n_vertices, n_vertices),
    )

    return D - W


def edge_collapse_decimation(X, adjacency_matrix, min_edge_length):
    """
    Perform structural decimation (E-collapse).

    Merges vertices connected by edges shorter than min_edge_length to maintain
    clean topology and prevent node crowding during graph contraction.

    Parameters
    ----------
    X : numpy.ndarray
        An (N, 3) float array containing the 3D spatial coordinates of the
        graph's vertices, where N is the number of vertices.
    adjacency_matrix : scipy.sparse.spmatrix
        A square, sparse adjacency matrix (e.g., CSR or COO format) of shape (N, N)
        representing the structural connectivity between nodes.
    min_edge_length : float
        The structural distance threshold. Any edge with a Euclidean length shorter
        than this value will be collapsed.

    Returns
    -------
    new_X : numpy.ndarray
        A (M, 3) float array containing the updated spatial coordinates of the remaining
        M unique vertices after simplification.
    new_adj : scipy.sparse.csr_matrix
        A simplified sparse CSR adjacency matrix of shape (M, M) with self-loops
        and duplicate edges removed.
    """
    n_vertices = X.shape[0]
    rows, cols = adjacency_matrix.nonzero()

    # Keep track of which vertices are mapped/merged to which
    vertex_map = np.arange(n_vertices)

    for u, v in zip(rows, cols):
        if u >= v:
            continue  # Only check each unique undirected edge once

        # Check if the edge is shorter than the allowed threshold
        dist = np.linalg.norm(X[u] - X[v])
        if dist < min_edge_length:
            root_u = vertex_map[u]
            root_v = vertex_map[v]
            if root_u != root_v:
                # Merge v into u: update positions to their average
                X[root_u] = (X[root_u] + X[root_v]) / 2.0
                vertex_map[vertex_map == root_v] = root_u

    # Remap unique remaining vertices
    unique_verts, inverse_indices = np.unique(vertex_map, return_inverse=True)
    new_X = X[unique_verts]

    # Rebuild the simplified adjacency matrix
    new_rows = inverse_indices[rows]
    new_cols = inverse_indices[cols]

    # Remove self-loops and duplicates
    valid_mask = new_rows != new_cols
    new_rows = new_rows[valid_mask]
    new_cols = new_cols[valid_mask]

    new_data = np.ones(len(new_rows), dtype=bool)
    new_adj = sparse.csr_matrix(
        (new_data, (new_rows, new_cols)), shape=(len(unique_verts), len(unique_verts))
    )

    return new_X, new_adj


def laplacian_graph_contraction_edt(
    X_init,
    adj_init,
    binary_segmentation=None,
    use_edt=True,
    use_anisotropic=True,
    enforce_containment=False,
    w_L=0.5,
    w_H_base=0.5,
    beta_edt=1.0,
    delta=0.5,
    max_iter=2000,
    tol=0.05,
    decimate_every=2,
    min_edge_length=0.5,
    alpha_norm=1.5,
    alpha_tang=0.1,
    solver='CG',
):
    """
    Carry out Laplacian Flow Dynamics.

    It uses optimization using 3D Euclidean Distance Transform (EDT) to dynamically
    scale retention forces and optionally enforces hard-voxel mask boundary containment.

    Parameters
    ----------
    X_init : numpy.ndarray
        Initial 3D coordinates of the graph vertices as an (N, 3) array.
    adj_init : scipy.sparse.csr_matrix
        Boolean sparse adjacency matrix representing initial network connectivity of shape (N, N).
    binary_segmentation : numpy.ndarray, optional
        The binary segmentation mask volume used to calculate the EDT profile. Default is None.
    use_edt : bool, optional
        If True, enables the Euclidean Distance Transform boundary potential constraint to prevent
        implosive collapse beyond true anatomy boundaries. Default is True.
    use_anisotropic : bool, optional
        If True, applies directionally weighted affinity rules prioritizing cross-sectional
        radial contraction over structural longitudinal shrinkage. Default is True.
    enforce_containment : bool, optional
        If True, applies a hard projection constraint to force nodes drifting out of the
        foreground mask onto the closest inner boundary shell surface voxel. Default is False.
    w_L : float, optional
        Contraction weight coefficient forcing nodes toward localized neighborhood geometric centers.
        Default is 0.5.
    w_H_base : float, optional
        The baseline structural positional anchor retention weight coefficient. Default is 0.5.
    beta_edt : float, optional
        Scaling parameter modulate exponent behavior of the EDT boundary attraction potential.
        Default is 1.0.
    delta : float, optional
        A smoothing stabilizer parameter added to the denominator to avoid division-by-zero errors
        at exact boundary contours. Default is 0.5.
    max_iter : int, optional
        Maximum allowed iteration steps for the contraction flow solver. Default is 2000.
    tol : float, optional
        Convergence tolerance limit evaluated against mean vertex displacement. Default is 1e-3.
    decimate_every : int, optional
        Frequency cadence interval defining how many contraction loop steps occur before triggering
        an edge-collapse decimation execution. Default is 2.
    min_edge_length : float, optional
        The Euclidean spatial threshold criteria below which two connected nodes undergo structural merging.
        Default is 0.5.
    alpha_norm : float, optional
        The normal/cross-sectional penalty parameter used during anisotropic calculation phases.
        Default is 1.5.
    alpha_tang : float, optional
        The tangential/longitudinal orientation penalty parameter used during anisotropic calculation phases.
        Default is 0.1.
    solver : ['LU', 'CG'], string, optional
        The solver to use to solve the linear system Ax = b. LU uses SuperLU, a direct
        solver, CG uses Conjugate Gradient (iterative solver), better for memory on big data.


    Returns
    -------
    X : numpy.ndarray
        Contracted centerline coordinates as an (M, 3) matrix.
    adj : scipy.sparse.csr_matrix
        Decimated skeleton topology graph connectivity representation of shape (M, M).
    """
    X = X_init.copy().astype(float)
    adj = adj_init.copy()

    # Conditional 3D EDT & Hard-Voxel Constraint Lookup Precomputation
    edt_volume = None
    closest_vessels_indices = None

    if (use_edt or enforce_containment) and binary_segmentation is not None:
        print('Computing 3D EDT Map and boundary projection lookup tensors...')
        # Inverse transform tells background voxels how far they are from the foreground target mask
        background_edt, nearest_indices = ndimage.distance_transform_edt(
            binary_segmentation == 0, return_indices=True
        )
        edt_volume = ndimage.distance_transform_edt(binary_segmentation)
        closest_vessels_indices = nearest_indices
        vol_shape = binary_segmentation.shape
    elif (use_edt or enforce_containment) and binary_segmentation is None:
        print(
            'Warning: No segmentation mask provided. Falling back to classic approach.'
        )
        use_edt = False
        enforce_containment = False

    edt_string = f' beta_edt (EDT scale factor)={beta_edt},' if use_edt else ''

    print(
        f'Starting contraction with {X.shape[0]} nodes \n\n'
        f'Params:\n'
        f' - w_L (\u03b1)={w_L}, w_H_base (\u03b2)={w_H_base}, tol (\u03b3)={tol},\n'
        f' -{edt_string} min_edge_length (decimation grid)={min_edge_length}\n\n'
        f'Options:\n'
        f' - Anisotropic={use_anisotropic}\n'
        f' - EDT={use_edt}\n'
        f' - Hard Containment={enforce_containment}\n'
        f' - Decimation step={decimate_every}\n'
    )

    for i in range(max_iter):
        n_vertices = X.shape[0]

        # 1. Compute chosen Laplacian variant
        L = compute_laplacian_matrix(
            X,
            adj,
            use_anisotropic=use_anisotropic,
            alpha_norm=alpha_norm,
            alpha_tang=alpha_tang,
        )
        L_squared = L.dot(L)

        # 2. Extract localized retention matrix mapping
        max_pull = ''

        if use_edt:
            ix = np.clip(np.round(X[:, 0]).astype(int), 0, vol_shape[0] - 1)
            iy = np.clip(np.round(X[:, 1]).astype(int), 0, vol_shape[1] - 1)
            iz = np.clip(np.round(X[:, 2]).astype(int), 0, vol_shape[2] - 1)
            node_distances = edt_volume[ix, iy, iz]
            w_H_per_node = w_H_base * np.exp(beta_edt / (node_distances + delta))
            W_H_sq = sparse.diags(w_H_per_node**2, format='csr')
            max_pull = f' - Max EDT w_H Pull: {np.max(w_H_per_node):.4f}'
        else:
            W_H_sq = sparse.eye(n_vertices, format='csr') * (w_H_base**2)

        # 3. Solve Implicit Update System equations
        if solver == 'LU':
            A = (w_L**2) * L_squared + W_H_sq
            B = W_H_sq.dot(X)

            X_next = np.zeros_like(X)
            for dim in range(3):
                X_next[:, dim] = spsolve(A, B[:, dim])
        elif solver == 'CG':
            A = (w_L**2) * L_squared + W_H_sq
            B = W_H_sq.dot(X)

            X_next = np.zeros_like(X)
            for dim in range(3):
                # Use CG with the previous coordinate array as a warm start (x0)
                # tol=1e-4 is plenty accurate for contraction steps
                sol, info = cg(A, B[:, dim], x0=X[:, dim], tol=1e-4, maxiter=500)
                X_next[:, dim] = sol

        # 4. Explicit Hard-Voxel Containment Constraint Projection
        if enforce_containment:
            # Re-discretize positions to evaluate mask containment state
            ix_next = np.clip(np.round(X_next[:, 0]).astype(int), 0, vol_shape[0] - 1)
            iy_next = np.clip(np.round(X_next[:, 1]).astype(int), 0, vol_shape[1] - 1)
            iz_next = np.clip(np.round(X_next[:, 2]).astype(int), 0, vol_shape[2] - 1)

            # Find points that fell outside the vessel grid (mask == 0)
            escaped_mask = binary_segmentation[ix_next, iy_next, iz_next] == 0
            escaped_count = np.sum(escaped_mask)

            if escaped_count > 0:
                # Extract precomputed closest coordinate index maps for escaped nodes
                proj_x = closest_vessels_indices[0][
                    ix_next[escaped_mask], iy_next[escaped_mask], iz_next[escaped_mask]
                ]
                proj_y = closest_vessels_indices[1][
                    ix_next[escaped_mask], iy_next[escaped_mask], iz_next[escaped_mask]
                ]
                proj_z = closest_vessels_indices[2][
                    ix_next[escaped_mask], iy_next[escaped_mask], iz_next[escaped_mask]
                ]

                # Project continuous coordinates onto the target boundary shell voxels
                X_next[escaped_mask] = np.stack(
                    [proj_x, proj_y, proj_z], axis=1
                ).astype(float)
                max_pull += f' [Projected: {escaped_count} escaped nodes]'

        displacement = np.mean(np.linalg.norm(X_next - X, axis=1))
        X = X_next

        print(
            f'Iter {i + 1}/{max_iter} - Remaining Nodes: {X.shape[0]} - '
            f'Error Drift: {displacement:.5f}{max_pull}'
        )

        if displacement < tol:
            print('Convergence criteria reached.')
            break

        if (i + 1) % decimate_every == 0:
            X, adj = edge_collapse_decimation(X, adj, min_edge_length)

    return X, adj


def compute_sparse_adjacency_matrix(tree, max_distance=2.4999):
    """
    Compute sparse adjacency matrix.

    Parameters
    ----------
    tree : scipy.spatial.KDTree
        The tree initialised from X_init
    max_distance : float, optional
        Max distance to consider in the tree to compute the adj matrix

    Returns
    -------
    adj_sparse : scipy.sparse.csr_matrix
        Sparse adjacency matrix
    """
    # Returns a sparse adjacency matrix directly for distances strictly within radius (0, 2.5)
    adj_sparse = tree.sparse_distance_matrix(tree, max_distance=max_distance).tocsr()
    # Remove self-loops (distance == 0 on diagonal)
    adj_sparse.setdiag(0)
    adj_sparse.eliminate_zeros()

    return (adj_sparse > 0).astype(bool)


def _process_single_label(
    label_id,
    segment,
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
    segment : np.ndarray
        One labelled segment from scipy's label.
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
    X_init = np.argwhere(segment).astype(float)
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
        binary_segmentation=segment,
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


def label_and_sort_by_size(binary_mask, label_connectivity=6):
    """
    Label connected components and re-order by size in reverse round-robin fashion.

    Parameters
    ----------
    binary_mask : np.ndarray
        Volume to label
    label_connectivity : 6, 18, 26, optional
        Connectivity profile to use to separate streams - 6, 18, or 26 edges.

    Returns
    -------
    labeled_volume
        Labeled volume in precise order
    num_features
        Number of features extracted from binary_mask

    Raises
    ------
    ValueError
        If label_connectivity is not a valid number
    """
    CONN = {6: 1, 18: 2, 26: 3}
    if label_connectivity not in CONN:
        raise ValueError(
            f'Label connectivity {label_connectivity} is not a valid option [6, 18, 26].'
        )

    labeled_volume, num_features = ndimage.label(
        binary_mask,
        structure=ndimage.generate_binary_structure(3, CONN[label_connectivity]),
    )

    print(
        f'Divided volume into {num_features} distinct label components '
        f'({label_connectivity}-connectivity).'
    )

    if num_features == 0:
        return labeled_volume, 0

    counts = np.bincount(labeled_volume.ravel())
    # Exclude background (index 0) and sort descending
    sorted_labels = np.argsort(counts[1:])[::-1] + 1

    mapping = np.zeros(num_features + 1, dtype=labeled_volume.dtype)
    mapping[sorted_labels] = np.arange(1, num_features + 1)

    return mapping[labeled_volume], num_features


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

    coords = np.rint(X).astype(np.int16)

    # 2. Fix coordinates on boundaries due to numpy's round-to-even
    for dim, bound in enumerate(volume_shape):
        coords[:, dim][coords[:, dim] == bound] = bound - 1

    # 3. Rasterize edges and nodes into the grid
    for i in coords:
        dense_volume[tuple(i)] = True

    return dense_volume


def laplacian_skeletonisation(
    nifti_path,
    out_path=None,
    use_edt=True,
    use_anisotropic=True,
    enforce_containment=False,
    beta_edt=1.0,
    w_L=0.5,
    w_H_base=0.5,
    tol=0.05,
    decimate_every=2,
    min_edge_length=0.5,
    downsample=False,
    seed=42,
    separate_streams=False,
    label_connectivity=6,
    max_distance=2.4999,
    n_jobs=None,
):
    """
    Load a NIfTI file volume image and perform geometric graph contraction skeletonisation.

    Parameters
    ----------
    nifti_path : str
        File system location path pointing directly toward the source input .nii or .nii.gz file.
    out_path : str, optional
        Output destination storage path base where resulting arrays and generated skeleton
        files will be written. Default is None (autogenerated from input file name).
    use_edt : bool, optional
        Enables boundary tracking potential constraints using Euclidean Distance Transforms.
        Default is True.
    use_anisotropic : bool, optional
        Enables anisotropic geometry handling to penalize internal longitudinal
        shortening vectors. Default is True.
    enforce_containment : bool, optional
        If True, applies a hard projection constraint to force nodes drifting out of the
        foreground mask onto the closest inner boundary shell surface voxel.
        Default is False.
    beta_edt : float, optional
        Scaling modulation weight assigned to boundary energy calculation properties.
        Default is 1.0.
    w_L : float, optional
        Contraction weight step modifier targeting structural local geometric collapse.
        This should be alpha in Damseh 2021. Default is 0.5.
    w_H_base : float, optional
        Baseline structural node anchor positional persistence value metric.
        This should be equivalent to beta in Damseh 2021. Default is 0.5.
    tol : float, optional
        Convergence tolerance limit evaluated against mean vertex displacement.
        This should be the equivalent of gamma in Damseh 2021 (not sure). Default is 0.05.
    decimate_every : int, optional
        Frequency cadence interval defining how many contraction loop steps occur before
        triggering an edge-collapse decimation execution. Default is 2.
    min_edge_length : float, optional
        The Euclidean spatial threshold criteria below which two connected nodes undergo
        structural merging, i.e. the isotropic voxel size of the grid used for
        decimation. Default is 0.5.
    downsample : bool, optional
        Flag setting whether point arrays containing high density are uniformly downsampled
        to stay within safe RAM footprints. Default is False.
    seed : int, optional
        Base random seed for reproducible downsampling (42 is default).
    separate_streams : bool, optional
        Process each "independent" vessel by itself (i.e. non-connected segment)
    label_connectivity : 6, 18, 26, optional
        Connectivity profile to use to separate streams - 6, 18, or 26 edges.
    max_distance : float
        Maximum distance to consider when making the sparse adjacency matrix.
    n_jobs : None, optional
        Number of parallel jobs. If not set or <=0, defaults to ~30%% of available CPU
        cores.

    Returns
    -------
    contracted_X : numpy.ndarray
        An (M, 3) matrix mapping the continuous 3D spatial points along the skeleton path.
    final_adj : scipy.sparse.csr_matrix
        The resulting graph sparse adjacency connectivity representation of shape (M, M).
    nifti_skel : numpy.ndarray
        A dense nifti matrix with the skeleton inside.

    Raises
    ------
    ValueError
        If the loaded structural NIfTI mask image is completely empty or lacks foreground elements.
    """
    print(f'Ingesting NIfTI image: {nifti_path}')
    _, volume_data, img = io.load_nifti_get_mask(nifti_path, is_mask=True, ndim=3)

    if not np.any(volume_data):
        raise ValueError('Provided segmentation volume lacks any foreground structure.')

    # Downsample points cloud initialization limits if necessary to guard RAM bounds
    if downsample and np.any(volume_data) > 200000:
        print(f'Volume contains {np.any(volume_data)} points. Downsampling.')
        vessel_voxels = np.argwhere(volume_data).astype(float)
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(len(vessel_voxels), 150000, replace=False)
        vessel_voxels = vessel_voxels[idx]
        volume_data = coords_to_dense_3d(vessel_voxels, volume_data.shape)

    # Process each component independently if separate_streams is True
    if separate_streams:
        labeled_volume, num_features = label_and_sort_by_size(
            volume_data, label_connectivity
        )
    else:
        labeled_volume, num_features = volume_data * 1, 1

    total_cores = os.cpu_count() or 1
    if n_jobs is None or n_jobs <= 0:
        n_workers = max(1, int(np.floor(0.30 * total_cores)))
    else:
        n_workers = min(n_jobs, total_cores)

    print(
        f'Processing {num_features} components in parallel using {n_workers} worker(s) '
        f'on {total_cores} CPU cores detected.'
    )

    tasks = []
    for label_id in range(1, num_features + 1):
        tasks.append((label_id, labeled_volume == label_id))

    results = ParallelPbar('Skeletonising')(
        n_jobs=n_workers, max_nbytes='1M', mmap_mode='r'
    )(
        delayed(_process_single_label)(
            label_id,
            segment,
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
        for label_id, segment in tasks
    )

    print('Reuniting results from parallel jobs.')

    if not results:
        raise ValueError('No valid components found for contraction.')

    # Merge coordinates and sparse block-diagonal adjacency matrices across all labels
    contracted_X = np.vstack([res[1] for res in results])
    final_adj = sparse.block_diag([res[2] for res in results], format='csr')

    out_path = (
        out_path
        if out_path
        else f'{os.path.splitext(os.path.splitext(nifti_path)[0])[0]}_skel'
    )

    print(f'\nSaving structural centerline data matrices to: {out_path}')
    np.savez_compressed(f'{out_path}_coords.npz', contracted_X=contracted_X)
    sparse.save_npz(f'{out_path}.npz', final_adj)
    nifti_skel = coords_to_dense_3d(contracted_X, volume_data.shape)

    # If enforce containment was used, assume no loss of tracts masking with original segmentation.
    if enforce_containment:
        nifti_skel = nifti_skel * volume_data

    io.export_nifti(nifti_skel, img, f'{out_path}.nii.gz')

    return contracted_X, final_adj, nifti_skel


def _main(argv=None):
    args = _get_parser().parse_args(argv)

    laplacian_skeletonisation(**vars(args))


if __name__ == '__main__':
    _main(sys.argv[1:])
