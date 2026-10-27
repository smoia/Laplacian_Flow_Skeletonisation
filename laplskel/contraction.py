"""Laplacian graph contraction algorithm."""

import warnings

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg, spsolve
from scipy.spatial import cKDTree

from .graph import compute_laplacian_matrix
from .medial import compute_medialness
from .objects import UnionFind


def edge_collapse_decimation(
    X, adjacency_matrix, min_edge_length, medialness=None, w_H_medial=1.0
):
    """
    Perform structural decimation (E-collapse).

    Merge short edges to reduce crowding, bounding each complete cluster by
    min_edge_length. This intermediate operation does not certify foreground
    topology; the workflow's final foreground-guided refinement does that.

    When medialness scores are supplied, merged coordinates are averaged with
    medial weighting so the surviving vertex is drawn toward the most medial of its
    members rather than to their unweighted centroid, and the surviving vertex
    inherits the highest score among them.

    Parameters
    ----------
    X : numpy.ndarray
        An (N, 3) float array containing the 3D spatial coordinates of the
        graph's vertices, where N is the number of vertices.
    adjacency_matrix : scipy.sparse.spmatrix
        A square, sparse adjacency matrix (e.g., CSR or COO format) of shape (N, N)
        representing the structural connectivity between nodes.
    min_edge_length : float
        Maximum bounding-box diagonal of a merged cluster, in voxel units.
    medialness : numpy.ndarray, optional
        An (N,) array of medialness scores in [0, 1] used to weight merged coordinates.
        Default is None, which averages merged coordinates uniformly.
    w_H_medial : float, optional
        The medial retention boost whose profile also weights merged coordinates.
        A value of 1.0 reproduces uniform averaging exactly. Default is 1.0.

    Returns
    -------
    new_X : numpy.ndarray
        A (M, 3) float array containing the updated spatial coordinates of the remaining
        M unique vertices after simplification.
    new_adj : scipy.sparse.csr_matrix
        A simplified sparse CSR adjacency matrix of shape (M, M) with self-loops
        and duplicate edges removed.
    new_medialness : numpy.ndarray or None
        An (M,) array of inherited medialness scores, or None when `medialness`
        was not supplied.
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

    # Medial vertices carry more weight into a merge, so the surviving position sits
    # nearer the inscribed-sphere centre instead of the plain centroid. Unit boost
    # leaves every weight at one, which is exactly uniform averaging.
    if medialness is None:
        merge_weights = np.ones(n_vertices, dtype=float)
    else:
        merge_weights = float(w_H_medial) ** np.asarray(medialness, dtype=float)

    # Track merged positions without mutating X during the loop
    # We maintain running coordinate sums and vertex weights for each root
    coord_sums = X * merge_weights[:, None]
    weight_sums = merge_weights.copy()

    # Bound whole clusters, not just individual links in a potentially long chain.
    lower_bounds = X.copy()
    upper_bounds = X.copy()
    for edge_index in np.argsort(edge_dists[collapse_mask], kind='stable'):
        u, v = short_u[edge_index], short_v[edge_index]
        root_u, root_v = uf.find(u), uf.find(v)
        lower = np.minimum(lower_bounds[root_u], lower_bounds[root_v])
        upper = np.maximum(upper_bounds[root_u], upper_bounds[root_v])
        if np.linalg.norm(upper - lower) >= min_edge_length:
            continue
        merged, root_u, root_v = uf.union(u, v)
        if merged:
            # Accumulate positions into the new combined root
            coord_sums[root_u] += coord_sums[root_v]
            weight_sums[root_u] += weight_sums[root_v]
            lower_bounds[root_u] = lower
            upper_bounds[root_u] = upper

    # Resolve final root assignments for every vertex
    final_roots = np.array([uf.find(i) for i in range(n_vertices)])

    # Compute averaged coordinates for each root
    unique_roots, inverse_indices = np.unique(final_roots, return_inverse=True)
    new_X = coord_sums[unique_roots] / weight_sums[unique_roots][:, None]

    # Each surviving vertex keeps the highest medialness among its merged members
    new_medialness = None
    if medialness is not None:
        new_medialness = np.zeros(len(unique_roots), dtype=float)
        np.maximum.at(new_medialness, inverse_indices, medialness)

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

    return new_X, new_adj, new_medialness


def _project_escaped_nodes(points, foreground_tree):
    """Snap points outside closed foreground voxel cubes to nearest centres."""
    if not np.all(np.isfinite(points)):
        raise ValueError('Containment requires finite node coordinates.')

    # The nearest centre under the maximum-coordinate distance identifies cube
    # membership, including face, edge and corner contacts.
    cube_distances, _ = foreground_tree.query(points, p=np.inf, eps=0)
    escaped = cube_distances > 0.5 + 1e-9
    count = int(np.count_nonzero(escaped))
    if not count:
        return points, 0

    projected = points.copy()
    _, nearest = foreground_tree.query(points[escaped], p=2, eps=0)
    projected[escaped] = foreground_tree.data[nearest]
    return projected, count


def laplacian_graph_contraction(
    X_init,
    adj_init,
    binary_segmentation=None,
    use_edt=True,
    use_anisotropic=True,
    enforce_containment=False,
    w_L=0.5,
    w_H_base=0.5,
    w_H_medial=1.0,
    beta_edt=1.0,
    delta=0.5,
    max_iter=2000,
    tol=0.05,
    decimate_ratio=0.93,
    min_edge_length=0.01,
    alpha_norm=1.5,
    alpha_tang=0.1,
    local_pca_hops=1,
    solver='CG',
    provisional=False,
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
        If True, snaps nodes outside the continuous foreground voxel cubes to the
        nearest foreground voxel centre before and during contraction. Default is False.
    w_L : float, optional
        Contraction weight coefficient forcing nodes toward localized neighborhood geometric centers.
        Default is 0.5.
    w_H_base : float, optional
        The baseline structural positional anchor retention weight coefficient. Default is 0.5.
    w_H_medial : float, optional
        Multiplicative retention boost applied to nodes at or around inscribed-sphere
        centres, stacked on top of the baseline and EDT weights. The boost is raised to
        the power of each node's medialness score, so fully medial nodes receive the
        whole factor while boundary nodes receive none. A value of 1.0 disables the
        boost and reproduces the unweighted behaviour exactly. Must be >= 1.0.
        Default is 1.0.
    beta_edt : float, optional
        Scaling parameter modulate exponent behavior of the EDT boundary attraction potential.
        Default is 1.0.
    delta : float, optional
        A smoothing stabilizer parameter added to the denominator to avoid division-by-zero errors
        at exact boundary contours. Default is 0.5.
    max_iter : int, optional
        Maximum allowed iteration steps for the contraction flow solver. Default is 2000.
    tol : float, optional
        Maximum vertex displacement tolerance in voxel units. Three stable steps
        without structural changes are required. Default is 0.05.
    decimate_ratio : float, optional
        Adaptive decimation threshold in (0, 1]. Drift is the mean node displacement
        after containment projection. Edge-collapse decimation runs once three
        consecutive drift ratios (current / previous) lie in [decimate_ratio, 1],
        i.e. on the fourth valid drift. Failed solves and zero drift clear the ratio
        history, and every decimation attempt resets it. Small-step and final-iteration
        decimation are unchanged. Default is 0.93.
    min_edge_length : float, optional
        The Euclidean spatial threshold criteria below which two connected nodes undergo structural merging.
        A nonpositive value disables merging and the adaptive trigger. Default is 0.01.
    alpha_norm : float, optional
        The normal/cross-sectional penalty parameter used during anisotropic calculation phases.
        Default is 1.5.
    alpha_tang : float, optional
        The tangential/longitudinal orientation penalty parameter used during anisotropic calculation phases.
        Default is 0.1.
    local_pca_hops : int, optional
        Number of graph hops included in each node's neighborhood when estimating
        local tangent directions. Default is 1.
    solver : ['LU', 'CG', 'AMGCG'], string, optional
        The solver to use to solve the linear system Ax = b. LU uses SuperLU, a direct
        solver, CG uses Conjugate Gradient (iterative solver), better for memory on big
        data, AMGCG constructs an Algebraic Multigrid (AMG) preconditioner before
        running CG, which makes it faster, but may require a tad more memory.
        Default is CG.
    provisional : bool, optional
        If True, treat ``max_iter`` as an intentional bounded contraction stage and
        do not warn when that stage ends before geometric convergence. Default is
        False. This does not change the numerical update or convergence test.

    Returns
    -------
    X : numpy.ndarray
        Contracted centerline coordinates as an (M, 3) matrix.
    adj : scipy.sparse.csr_matrix
        Decimated skeleton topology graph connectivity representation of shape (M, M).

    Raises
    ------
    ValueError
        If `w_H_medial` is smaller than 1.0 or `decimate_ratio` is outside (0, 1].
    """
    if not np.isfinite(w_H_medial) or w_H_medial < 1.0:
        raise ValueError('w_H_medial must be finite and >= 1.0.')
    if solver not in ('CG', 'AMGCG', 'LU'):
        raise ValueError('solver must be CG, AMGCG, or LU.')
    if max_iter < 1:
        raise ValueError('max_iter must be positive.')
    if not np.isfinite(decimate_ratio) or not 0 < decimate_ratio <= 1:
        raise ValueError('decimate_ratio must be finite and in (0, 1].')

    X = X_init.copy().astype(float)
    adj = adj_init.copy()

    # Prepare containment and optional EDT guidance.
    edt_volume = None
    foreground_tree = None
    medialness = None

    needs_edt = use_edt or w_H_medial > 1.0

    if binary_segmentation is not None:
        if enforce_containment:
            foreground_centres = np.argwhere(binary_segmentation)
            if not len(foreground_centres):
                raise ValueError('Containment requires a nonempty foreground mask.')
            foreground_tree = cKDTree(foreground_centres)
            X, initial_count = _project_escaped_nodes(X, foreground_tree)
            if initial_count:
                print(f'Projected {initial_count} initial escaped nodes to foreground centres.')

        if needs_edt:
            print('Computing 3D EDT Map...')
            edt_volume = ndimage.distance_transform_edt(binary_segmentation)
        if w_H_medial > 1.0:
            medialness = compute_medialness(edt_volume, X)
    elif needs_edt or enforce_containment:
        print(
            'Warning: No segmentation mask provided. Falling back to classic approach.'
        )
        use_edt = False
        enforce_containment = False
        w_H_medial = 1.0

    edt_string = f' beta_edt (EDT scale factor)={beta_edt},' if use_edt else ''

    print(
        f'Starting contraction with {X.shape[0]} nodes \n\n'
        f'Params:\n'
        f' - w_L (\u03b1)={w_L}, w_H_base (\u03b2)={w_H_base}, tol (\u03b3)={tol},\n'
        f' -{edt_string} min_edge_length (collapse threshold)={min_edge_length}\n\n'
        f'Options:\n'
        f' - Anisotropic={use_anisotropic}\n'
        f' - EDT={use_edt}\n'
        f' - Medial Boost={w_H_medial if medialness is not None else False}\n'
        f' - Hard Containment={enforce_containment}\n'
        f' - Decimation ratio={decimate_ratio}\n'
    )

    stable_steps = 0
    previous_drift = None
    qualifying_ratios = []
    for i in range(max_iter):
        n_vertices = X.shape[0]

        # 1. Compute chosen Laplacian variant
        L = compute_laplacian_matrix(
            X,
            adj,
            use_anisotropic=use_anisotropic,
            alpha_norm=alpha_norm,
            alpha_tang=alpha_tang,
            local_pca_hops=local_pca_hops,
        )
        L_squared = L.T.dot(L)

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
        else:
            w_H_per_node = np.full(n_vertices, w_H_base, dtype=float)

        if medialness is not None:
            # Raising the boost to the medialness score leaves boundary nodes at their
            # baseline weight while holding inscribed-sphere centres in place.
            w_H_per_node = w_H_per_node * (w_H_medial**medialness)

        W_H_sq = sparse.diags(w_H_per_node**2, format='csr')

        if use_edt:
            max_pull = f' - Max EDT w_H Pull: {np.max(w_H_per_node):.4f}'
        elif medialness is not None:
            max_pull = f' - Max Medial w_H Pull: {np.max(w_H_per_node):.4f}'

        # 3. Solve Implicit Update System equations
        A = (w_L**2) * L_squared + W_H_sq
        # Solve for motion, so large anchors or coordinate offsets cannot make
        # CG accept the previous coordinates without taking an accurate step.
        B = -(w_L**2) * L.T.dot(L.dot(X - X.mean(axis=0)))
        step_solved = True

        # Select solver between LU, AMGCG, and CG.
        if solver == 'AMGCG':
            # Prepare fallback to CG if AMGCG cannot run due to too many voxels.
            try:
                import pyamg
            except ImportError:
                warnings.warn(
                    'PyAMG is unavailable; switching to CG.',
                    RuntimeWarning,
                    stacklevel=2,
                )
                solver = 'CG'

            if solver == 'AMGCG' and (
                A.indptr.dtype == np.int64 or A.indices.dtype == np.int64
            ):
                max_idx = max(A.shape[0], A.nnz)
                if max_idx <= np.iinfo(np.int32).max:
                    A = A.copy()
                    A.indptr = A.indptr.astype(np.int32)
                    A.indices = A.indices.astype(np.int32)
                else:
                    # NNZ or shape exceeds int32 max limit (pyAMG C++ extensions will fail)
                    warnings.warn(
                        'AMGCG does not support this matrix index range; switching '
                        'to CG.',
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    solver = 'CG'

        preconditioner = None
        if solver == 'AMGCG':
            try:
                hierarchy = pyamg.ruge_stuben_solver(A)
                preconditioner = hierarchy.aspreconditioner(cycle='V')
            except (MemoryError, RuntimeError, TypeError, ValueError) as error:
                warnings.warn(
                    f'AMGCG setup failed ({error}); switching to CG.',
                    RuntimeWarning,
                    stacklevel=2,
                )
                solver = 'CG'

        if solver == 'LU':
            X_next = np.zeros_like(X)
            for dim in range(3):
                X_next[:, dim] = X[:, dim] + spsolve(A, B[:, dim])

        elif solver == 'AMGCG':
            X_next = np.zeros_like(X)
            for dim in range(3):
                sol, info = cg(
                    A,
                    B[:, dim],
                    x0=np.zeros(n_vertices),
                    M=preconditioner,
                    rtol=1e-6,
                    maxiter=500,
                )
                if info != 0:
                    step_solved = False
                    warnings.warn(
                        f'CG did not converge on axis {dim} (info={info}); the '
                        'contraction step may be inaccurate.',
                        RuntimeWarning,
                        stacklevel=2,
                    )
                X_next[:, dim] = X[:, dim] + sol

        elif solver == 'CG':
            X_next = np.zeros_like(X)
            for dim in range(3):
                sol, info = cg(
                    A, B[:, dim], x0=np.zeros(n_vertices), rtol=1e-6, maxiter=500
                )
                if info != 0:
                    step_solved = False
                    warnings.warn(
                        f'CG did not converge on axis {dim} (info={info}); the '
                        'contraction step may be inaccurate.',
                        RuntimeWarning,
                        stacklevel=2,
                    )
                X_next[:, dim] = X[:, dim] + sol

        if not np.all(np.isfinite(X_next)):
            raise RuntimeError('Contraction produced non-finite coordinates.')

        # 4. Project nodes that have left the continuous foreground cubes.
        if enforce_containment:
            X_next, escaped_count = _project_escaped_nodes(X_next, foreground_tree)
            if escaped_count:
                max_pull += f' [Projected: {escaped_count} escaped nodes to centres]'

        movements = np.linalg.norm(X_next - X, axis=1)
        displacement = np.mean(movements)
        max_displacement = np.max(movements)
        X = X_next

        # Adaptive decimation waits for the mean drift to plateau. Neither side of a
        # ratio may come from a failed solve, and zero drift gives no ratio at all.
        ratio_string = ''
        if not step_solved or displacement == 0:
            previous_drift, qualifying_ratios = None, []
        else:
            if previous_drift is not None:
                ratio = displacement / previous_drift
                if decimate_ratio <= ratio <= 1:
                    qualifying_ratios.append(ratio)
                else:
                    qualifying_ratios = []
                ratio_string = (
                    f' - Drift Ratio: {ratio:.5f} (streak {len(qualifying_ratios)})'
                )
            previous_drift = displacement

        print(
            f'Iter {i + 1}/{max_iter} - Remaining Nodes: {X.shape[0]} - '
            f'Mean Drift: {displacement:.5f} - Max Drift: {max_displacement:.5f}'
            f'{ratio_string}{max_pull}'
        )

        small_step = max_displacement < tol and step_solved
        # Without a positive merge length no edge can collapse, so a plateau gives
        # adaptive decimation nothing to remove.
        triggers = {
            'adaptive': len(qualifying_ratios) >= 3 and min_edge_length > 0,
            'small-step': small_step,
            'final-iteration': i + 1 == max_iter,
        }
        if any(triggers.values()):
            reasons = ', '.join(name for name, hit in triggers.items() if hit)
            if triggers['adaptive']:
                reasons += ' - Ratios: ' + ', '.join(
                    f'{ratio:.5f}' for ratio in qualifying_ratios[-3:]
                )
            X, adj, medialness = edge_collapse_decimation(
                X, adj, min_edge_length, medialness=medialness, w_H_medial=w_H_medial
            )
            if enforce_containment:
                X, decimated_count = _project_escaped_nodes(X, foreground_tree)
                if decimated_count:
                    print(
                        f'Projected {decimated_count} escaped nodes after decimation '
                        'to foreground centres.'
                    )
            print(
                f'Decimation at iter {i + 1} [{reasons}] - Removed Nodes: '
                f'{n_vertices - X.shape[0]} - Remaining Nodes: {X.shape[0]}'
            )
            previous_drift, qualifying_ratios = None, []
        stable_steps = stable_steps + 1 if small_step and len(X) == n_vertices else 0
        if stable_steps >= 3:
            print('Contraction geometry stable.')
            break
    else:
        if provisional:
            return X, adj
        warnings.warn(
            'Contraction reached max_iter without stable geometry; these coordinates '
            'are provisional and require foreground-guided refinement.',
            RuntimeWarning,
            stacklevel=2,
        )

    return X, adj
