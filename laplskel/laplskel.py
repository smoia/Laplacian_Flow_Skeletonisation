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

VALID_CONNECTIVITY = (6, 18, 26)
VALID_SOLVER = ('LU', 'CG', 'AMGCG')


class UnionFind:
    """Disjoint-set data structure with path compression for O(1) edge collapses."""

    def __init__(self, n):
        self.parent = np.arange(n)

    def find(self, i):
        # Path compression: update parent pointers recursively
        path = []
        while self.parent[i] != i:
            path.append(i)
            i = self.parent[i]
        for node in path:
            self.parent[node] = i
        return i

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i
            return True, root_i, root_j
        return False, root_i, root_j


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
        default=1,
        help='Decimate nodes every N steps [Default=2].',
    )
    optional.add_argument(
        '--dec_grid_size',
        dest='min_edge_length',
        type=float,
        default=0.01,
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
        degrees = np.bincount(rows, minlength=n_vertices)

        # Compute neighbor means for all vertices via sparse matrix multiplication
        # shape: (N, 3)
        neighbor_sums = adjacency_matrix.dot(X)
        safe_degrees = np.maximum(degrees[:, None], 1)
        neighbor_means = neighbor_sums / safe_degrees

        # Compute neighbor deviations from neighbor means per edge
        # shape: (E, 3)
        devs = X[cols] - neighbor_means[rows]

        # Assemble (N, 3, 3) covariance tensor using vectorized bincount
        cov_tensor = np.zeros((n_vertices, 3, 3), dtype=X.dtype)

        # 6 unique terms in a symmetric 3x3 matrix
        pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        denom = np.maximum(degrees - 1, 1)[:, None]

        for r, c in pairs:
            # Aggregate outer products per vertex
            cov_val = np.bincount(
                rows, weights=devs[:, r] * devs[:, c], minlength=n_vertices
            )
            cov_tensor[:, r, c] = cov_val
            if r != c:
                cov_tensor[:, c, r] = cov_val  # Symmetric fill

        cov_tensor /= denom[:, :, None]

        # Vectorized eigenvalue decomposition on tensor of shape (N, 3, 3)
        # eigvecs has shape (N, 3, 3); last column is the principal eigenvector
        eigvals, eigvecs = np.linalg.eigh(cov_tensor)
        tangents = eigvecs[:, :, -1]

        # Fallback for isolated vertices/single neighbors: default to [1, 0, 0]
        fallback_mask = degrees <= 1
        if np.any(fallback_mask):
            tangents[fallback_mask] = np.array([1.0, 0.0, 0.0])

        # Compute anisotropic components per edge
        t_i = tangents[rows]
        dot_products = np.sum(diffs * t_i, axis=1)

        tangential_comps = np.abs(dot_products)
        normal_comps = np.linalg.norm(diffs - (dot_products[:, None] * t_i), axis=1)

        aniso_mod = (alpha_norm * normal_comps) + (alpha_tang * tangential_comps)
        weights = aniso_mod / distances
    else:
        # Standard Isotropic Weights
        weights = 1.0 / distances

    # Assemble sparse operators
    W = sparse.csr_matrix((weights, (rows, cols)), shape=(n_vertices, n_vertices))

    # Build diagonal degree matrix D
    degree_values = np.array(W.sum(axis=1)).flatten()
    D = sparse.diags(degree_values, format='csr')

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

    # Only process upper triangle of the symmetric matrix (unique undirected edges)
    edge_mask = rows < cols
    u_nodes = rows[edge_mask]
    v_nodes = cols[edge_mask]

    # Calculate Euclidean distances for all unique edges at once
    edge_dists = np.linalg.norm(X[u_nodes] - X[v_nodes], axis=1)

    # Filter edges that are shorter than the threshold
    collapse_mask = edge_dists < min_edge_length
    short_u = u_nodes[collapse_mask]
    short_v = v_nodes[collapse_mask]

    uf = UnionFind(n_vertices)

    # Track merged positions without mutating X during the loop
    # We maintain running coordinate sums and vertex counts for each root
    coord_sums = X.copy()
    node_counts = np.ones(n_vertices, dtype=int)

    for u, v in zip(short_u, short_v):
        merged, root_u, root_v = uf.union(u, v)
        if merged:
            # Accumulate positions into the new combined root
            coord_sums[root_u] += coord_sums[root_v]
            node_counts[root_u] += node_counts[root_v]

    # Resolve final root assignments for every vertex
    final_roots = np.array([uf.find(i) for i in range(n_vertices)])

    # Compute averaged coordinates for each root
    unique_roots, inverse_indices = np.unique(final_roots, return_inverse=True)
    new_X = coord_sums[unique_roots] / node_counts[unique_roots][:, None]

    # Rebuild the simplified adjacency matrix using remapped indices
    new_rows = inverse_indices[rows]
    new_cols = inverse_indices[cols]

    # Remove self-loops
    valid_mask = new_rows != new_cols
    new_rows = new_rows[valid_mask]
    new_cols = new_cols[valid_mask]

    new_data = np.ones(len(new_rows), dtype=bool)
    new_adj = sparse.csr_matrix(
        (new_data, (new_rows, new_cols)), shape=(len(unique_roots), len(unique_roots))
    )

    return new_X, new_adj


def laplacian_graph_contraction(
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
    decimate_every=1,
    min_edge_length=0.01,
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
        an edge-collapse decimation execution. Default is 1.
    min_edge_length : float, optional
        The Euclidean spatial threshold criteria below which two connected nodes undergo structural merging.
        Default is 0.01.
    alpha_norm : float, optional
        The normal/cross-sectional penalty parameter used during anisotropic calculation phases.
        Default is 1.5.
    alpha_tang : float, optional
        The tangential/longitudinal orientation penalty parameter used during anisotropic calculation phases.
        Default is 0.1.
    solver : ['LU', 'CG', 'AMGCG'], string, optional
        The solver to use to solve the linear system Ax = b. LU uses SuperLU, a direct
        solver, CG uses Conjugate Gradient (iterative solver), better for memory on big
        data, AMGCG constructs an Algebraic Multigrid (AMG) preconditioner before
        running CG, which makes it faster, but may require a tad more memory.
        Default is CG.

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
            # Find value of EDT_volume by trilinear interpolation of new coordinates.
            # map_coordinates expects shape (ndim, N), so pass X.T
            node_distances = ndimage.map_coordinates(
                edt_volume, X.T, order=1, mode='nearest'
            )

            # Prevent divide-by-zero/negative issues from interpolation near boundary
            node_distances = np.maximum(node_distances, 0.0)

            w_H_per_node = w_H_base * np.exp(beta_edt / (node_distances + delta))
            W_H_sq = sparse.diags(w_H_per_node**2, format='csr')
            max_pull = f' - Max EDT w_H Pull: {np.max(w_H_per_node):.4f}'
        else:
            W_H_sq = sparse.eye(n_vertices, format='csr') * (w_H_base**2)

        # 3. Solve Implicit Update System equations
        A = (w_L**2) * L_squared + W_H_sq
        B = W_H_sq.dot(X)

        # Select solver between LU, AMGCG, and CG, check solver only once entirely.
        check_solver = True

        if solver == 'AMGCG' and check_solver:
            # Prepare fallback to CG if AMGCG cannot run due to too many voxels.
            try:
                import pyamg
            except ImportError:
                print(
                    '!!! WARNING: AMGCG solver was selected, but pyAMG is not '
                    'installed. Switching solver to CG. !!!'
                )
                solver = 'CG'

            if A.indptr.dtype == np.int64 or A.indices.dtype == np.int64:
                max_idx = max(A.shape[0], A.nnz)
                if max_idx <= np.iinfo(np.int32).max:
                    A = A.copy()
                    A.indptr = A.indptr.astype(np.int32)
                    A.indices = A.indices.astype(np.int32)
                    print('!!! WARNING: downcasting A indexes to int32 to use pyAMG')
                else:
                    # NNZ or shape exceeds int32 max limit (pyAMG C++ extensions will fail)
                    print(
                        '!!! WARNING: AMGCG solver was selected, but A has too many '
                        'non-zero voxels or rows. Switching solver to CG for this '
                        'segment. !!!'
                    )
                    solver = 'CG'

        if solver == 'LU':
            X_next = np.zeros_like(X)
            for dim in range(3):
                X_next[:, dim] = spsolve(A, B[:, dim])

        elif solver == 'AMGCG':
            ml = pyamg.ruge_stuben_solver(A)
            M = ml.aspreconditioner(cycle='V')

            X_next = np.zeros_like(X)
            for dim in range(3):
                sol, info = cg(A, B[:, dim], x0=X[:, dim], M=M, rtol=1e-4, maxiter=500)
                X_next[:, dim] = sol

        elif solver == 'CG':
            X_next = np.zeros_like(X)
            for dim in range(3):
                # Use CG with the previous coordinate array as a warm start (x0)
                # tol=1e-4 is plenty accurate for contraction steps
                sol, info = cg(A, B[:, dim], x0=X[:, dim], rtol=1e-4, maxiter=500)
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
    cropped_label,
    offset_origin,
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
    solver,
):
    """
    Worker function to process a single connected component label.

    Parameters
    ----------
    label_id : int
        ID of current label
    cropped_label : np.ndarray
        Segmentation labelled with scipy's label and cropped with find_objects.
    offset_origin : list
        cropped offset.
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
    solver : ['LU', 'CG', 'AMGCG'], string, optional
        The solver to use to solve the linear system Ax = b. LU uses SuperLU, a direct
        solver, CG uses Conjugate Gradient (iterative solver), better for memory on big
        data, AMGCG constructs an Algebraic Multigrid (AMG) preconditioner before
        running CG, which makes it far faster, but may require a tad more memory.

    Returns
    -------
    label_id : int
        ID of current label (For tracking)
    contracted_X : numpy.ndarray
        An (M, 3) matrix mapping the continuous 3D spatial points along the skeleton path.
    final_adj : scipy.sparse.csr_matrix
        The resulting graph sparse adjacency connectivity representation of shape (M, M).
    """
    X_init_local = np.argwhere(cropped_label).astype(np.uint16)
    tree = KDTree(X_init_local)

    # Skip small noise components
    if len(X_init_local) <= 3:
        adj_sparse = compute_sparse_adjacency_matrix(tree, max_distance)

        X_init_global = X_init_local + np.array(offset_origin, dtype=np.float32)
        return label_id, X_init_global, adj_sparse

    print(
        f'\n--- Processing Label {label_id}/{num_features} ({X_init_local.sum()} voxels) ---'
    )

    print('Computing proximity network coordinates...')
    adj_sparse = compute_sparse_adjacency_matrix(tree, max_distance)

    # Run contraction on this label's component mask
    label_X_local, label_adj = laplacian_graph_contraction(
        X_init_local,
        adj_sparse,
        binary_segmentation=cropped_label,
        use_edt=use_edt,
        use_anisotropic=use_anisotropic,
        enforce_containment=enforce_containment,
        beta_edt=beta_edt,
        w_L=w_L,
        w_H_base=w_H_base,
        tol=tol,
        decimate_every=decimate_every,
        min_edge_length=min_edge_length,
        solver=solver,
    )

    label_X_global = label_X_local + np.array(offset_origin, dtype=np.float32)

    return label_id, label_X_global, label_adj


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

    sizes = ndimage.sum(
        binary_mask,
        labeled_volume,
        index=np.arange(1, num_features + 1, dtype=np.int32),
    )
    # Sort descending
    sorted_labels = np.argsort(sizes)[::-1] + 1

    mapping = np.zeros(num_features + 1, dtype=labeled_volume.dtype)
    mapping[sorted_labels] = np.arange(1, num_features + 1, dtype=labeled_volume.dtype)

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

    coords = np.rint(X).astype(np.int64)

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
    decimate_every=1,
    min_edge_length=0.01,
    downsample=False,
    seed=42,
    separate_streams=False,
    label_connectivity=6,
    solver='CG',
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
        triggering an edge-collapse decimation execution. Default is 1.
    min_edge_length : float, optional
        The Euclidean spatial threshold criteria below which two connected nodes undergo
        structural merging, i.e. the isotropic voxel size of the grid used for
        decimation. Default is 0.01.
    downsample : bool, optional
        Flag setting whether point arrays containing high density are uniformly downsampled
        to stay within safe RAM footprints. Default is False.
    seed : int, optional
        Base random seed for reproducible downsampling (42 is default).
    separate_streams : bool, optional
        Process each "independent" vessel by itself (i.e. non-connected segment)
    label_connectivity : 6, 18, 26, optional
        Connectivity profile to use to separate streams - 6, 18, or 26 edges.
    solver : ['LU', 'CG', 'AMGCG'], string, optional
        The solver to use to solve the linear system Ax = b. LU uses SuperLU, a direct
        solver, CG uses Conjugate Gradient (iterative solver), better for memory on big
        data, AMGCG constructs an Algebraic Multigrid (AMG) preconditioner before
        running CG, which makes it far faster, but may require a tad more memory.
        Default is CG.
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
        vessel_voxels = np.argwhere(volume_data).astype(np.uint16)
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

    slices_list = ndimage.find_objects(labeled_volume)

    tasks = []
    for label_id in range(1, num_features + 1):
        bbox_slice = slices_list[label_id - 1]

        if bbox_slice is None:
            continue

        # Extract cropped boolean mask for ONLY this label
        cropped_label = labeled_volume[bbox_slice] == label_id

        # Offset origin (min_x, min_y, min_z) used to map back to original volume
        offset_origin = (
            bbox_slice[0].start,
            bbox_slice[1].start,
            bbox_slice[2].start,
        )

        tasks.append(
            delayed(_process_single_label)(
                label_id,
                cropped_label,
                offset_origin,
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
                solver,
            )
        )

    # 2. Run workers in parallel
    results = ParallelPbar('Skeletonising')(n_jobs=n_workers, batch_size=1)(tasks)

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
