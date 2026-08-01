"""Graph construction, weighting, and decimation operations."""

import numpy as np
from scipy import sparse


class UnionFind:
    """Disjoint-set data structure with path compression."""

    def __init__(self, n_vertices):
        self.parent = np.arange(n_vertices)

    def find(self, vertex):
        path = []
        while self.parent[vertex] != vertex:
            path.append(vertex)
            vertex = self.parent[vertex]
        for path_vertex in path:
            self.parent[path_vertex] = vertex
        return vertex

    def union(self, first, second):
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False, first_root, second_root

        self.parent[second_root] = first_root
        return True, first_root, second_root


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
        neighbor_sums = adjacency_matrix.dot(X)
        neighbor_means = neighbor_sums / np.maximum(degrees[:, None], 1)
        deviations = X[cols] - neighbor_means[rows]

        covariance = np.zeros((n_vertices, 3, 3), dtype=X.dtype)
        covariance_terms = (
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 1),
            (1, 2),
            (2, 2),
        )
        for row_dim, col_dim in covariance_terms:
            values = np.bincount(
                rows,
                weights=deviations[:, row_dim] * deviations[:, col_dim],
                minlength=n_vertices,
            )
            covariance[:, row_dim, col_dim] = values
            if row_dim != col_dim:
                covariance[:, col_dim, row_dim] = values

        covariance /= np.maximum(degrees - 1, 1)[:, None, None]
        _, eigenvectors = np.linalg.eigh(covariance)
        tangents = eigenvectors[:, :, -1]

        fallback_mask = degrees <= 1
        if np.any(fallback_mask):
            tangents[fallback_mask] = np.array([1.0, 0.0, 0.0])

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

    edge_mask = rows < cols
    first_nodes = rows[edge_mask]
    second_nodes = cols[edge_mask]
    edge_distances = np.linalg.norm(X[first_nodes] - X[second_nodes], axis=1)
    short_edge_mask = edge_distances < min_edge_length

    union_find = UnionFind(n_vertices)
    coordinate_sums = X.copy()
    node_counts = np.ones(n_vertices, dtype=int)

    for first, second in zip(
        first_nodes[short_edge_mask], second_nodes[short_edge_mask]
    ):
        merged, first_root, second_root = union_find.union(first, second)
        if merged:
            coordinate_sums[first_root] += coordinate_sums[second_root]
            node_counts[first_root] += node_counts[second_root]

    final_roots = np.array(
        [union_find.find(vertex) for vertex in range(n_vertices)]
    )
    unique_roots, inverse_indices = np.unique(final_roots, return_inverse=True)
    new_X = coordinate_sums[unique_roots] / node_counts[unique_roots][:, None]

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
