"""Graph construction, weighting, and decimation operations."""

import numpy as np
from scipy import sparse


def compute_laplacian_matrix(
    X,
    adjacency_matrix,
    use_anisotropic=True,
    alpha_norm=1.5,
    alpha_tang=0.1,
    local_pca_hops=1,
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
    local_pca_hops : int, optional
        Number of graph hops included in each vertex's local neighborhood when
        estimating anisotropic tangent directions. Default is 1.

    Returns
    -------
    L : scipy.sparse.csr_matrix
        The calculated sparse Graph Laplacian Matrix of shape (N, N) governed
        by the equation L = D - W.
    """
    if (
        isinstance(local_pca_hops, (bool, np.bool_))
        or not isinstance(local_pca_hops, (int, np.integer))
        or local_pca_hops <= 0
    ):
        raise ValueError('local_pca_hops must be a positive integer.')

    n_vertices = X.shape[0]

    # Get row and col indices from the sparse adjacency matrix
    rows, cols = adjacency_matrix.nonzero()

    # 1. Compute spatial difference vectors and Euclidean distances
    diffs = X[rows] - X[cols]
    distances = np.linalg.norm(diffs, axis=1)
    distances = np.maximum(distances, 1e-6)  # Prevent division by zero

    if use_anisotropic:
        pca_adjacency = _compute_n_hop_adjacency(
            adjacency_matrix, local_pca_hops
        )
        pca_rows, pca_cols = pca_adjacency.nonzero()

        # Estimate local structural tangents using local neighborhood PCA proxy
        degrees = np.bincount(pca_rows, minlength=n_vertices)

        # Compute neighbor means for all vertices via sparse matrix multiplication
        # shape: (N, 3)
        neighbor_sums = pca_adjacency.dot(X)
        safe_degrees = np.maximum(degrees[:, None], 1)
        neighbor_means = neighbor_sums / safe_degrees

        # Compute neighbor deviations from neighbor means per edge
        # shape: (E, 3)
        devs = X[pca_cols] - neighbor_means[pca_rows]

        # Assemble (N, 3, 3) covariance tensor using vectorized bincount
        cov_tensor = np.zeros((n_vertices, 3, 3), dtype=X.dtype)

        # 6 unique terms in a symmetric 3x3 matrix
        pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        denom = np.maximum(degrees - 1, 1)[:, None]

        for r, c in pairs:
            # Aggregate outer products per vertex
            cov_val = np.bincount(
                pca_rows,
                weights=devs[:, r] * devs[:, c],
                minlength=n_vertices,
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


def _compute_n_hop_adjacency(adjacency_matrix, hops):
    """Return boolean connectivity to every node reachable within ``hops`` edges."""
    adjacency = adjacency_matrix.astype(bool).tocsr()
    adjacency.setdiag(False)
    adjacency.eliminate_zeros()

    reached = adjacency.copy()
    frontier = adjacency
    for _ in range(1, hops):
        frontier = (frontier @ adjacency).astype(bool).tocsr()
        frontier.setdiag(False)
        frontier.eliminate_zeros()
        reached = (reached + frontier).astype(bool).tocsr()

    reached.setdiag(False)
    reached.eliminate_zeros()
    return reached


def compute_sparse_adjacency_matrix(tree, init_graph_adj=26):
    """
    Compute sparse adjacency matrix.

    Parameters
    ----------
    tree : scipy.spatial.KDTree
        The tree initialised from X_init
    init_graph_adj : {6, 18, 26}, int, optional
        Voxel-neighborhood connectivity used to construct graph edges. Default is 26.

    Returns
    -------
    adj_sparse : scipy.sparse.csr_matrix
        Sparse adjacency matrix
    """
    connectivity_radii = {
        6: 1.0,
        18: np.nextafter(np.sqrt(2.0), np.inf),
        26: np.nextafter(np.sqrt(3.0), np.inf),
    }
    if init_graph_adj not in connectivity_radii:
        raise ValueError('init_graph_adj must be one of 6, 18, or 26.')

    coordinates = np.asarray(tree.data)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError('Initial graph coordinates must have shape (N, 3).')
    if not np.all(np.isfinite(coordinates)) or not np.all(
        coordinates == np.rint(coordinates)
    ):
        raise ValueError(
            'Initial graph coordinates must lie on the integer voxel grid.'
        )

    # On an integer 3D voxel grid, radii 1, sqrt(2), and sqrt(3) correspond
    # exactly to face-, face-and-edge-, and full-corner connectivity.
    neighbor_radius = connectivity_radii[init_graph_adj]
    adj_sparse = tree.sparse_distance_matrix(tree, neighbor_radius).tocsr()
    # Remove self-loops (distance == 0 on diagonal)
    adj_sparse.setdiag(0)
    adj_sparse.eliminate_zeros()

    return (adj_sparse > 0).astype(bool)
