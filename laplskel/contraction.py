"""Laplacian graph contraction algorithm."""

import warnings

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg, spsolve

from .graph import compute_laplacian_matrix
from .medial import compute_medialness
from .objects import UnionFind
from .triangle_decimation import DistanceFluxSampler, triangle_collapse_decimation


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


def _project_escaped_nodes(coordinates, foreground, closest_indices):
    """Project nodes outside the rounded foreground onto its nearest voxel."""
    voxel_positions = np.rint(coordinates).astype(np.intp)
    volume_bounds = np.asarray(foreground.shape)
    out_of_bounds = np.any(
        (voxel_positions < 0) | (voxel_positions >= volume_bounds), axis=1
    )
    lookup_positions = np.clip(voxel_positions, 0, volume_bounds - 1)
    ix, iy, iz = lookup_positions.T
    escaped = out_of_bounds | (foreground[ix, iy, iz] == 0)
    escaped_count = int(np.sum(escaped))
    if escaped_count:
        projected = [
            indices[ix[escaped], iy[escaped], iz[escaped]]
            for indices in closest_indices
        ]
        coordinates[escaped] = np.stack(projected, axis=1).astype(float)
    return coordinates, escaped_count


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
    decimate_every=1,
    min_edge_length=0.01,
    alpha_norm=1.5,
    alpha_tang=0.1,
    local_pca_hops=1,
    solver='CG',
    provisional=False,
    decimation_mode='edge',
    voxel_spacing=(1.0, 1.0, 1.0),
    dev_elongation_ratio=25.0,
    dev_edge_thresh=0.05,
    dev_peri_ratio=1.0,
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
        If True, project escaped nodes onto the nearest foreground voxel before
        each update and again after its solve. Default is False.
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
    decimate_every : int, optional
        Contraction updates between scheduled decimation passes. Default is 1
        for direct callers; the default workflow supplies 200.
    min_edge_length : float, optional
        Legacy edge-mode merge threshold in voxel units. It does not control
        short-edge merging in triangle mode. Default is 0.01.
    alpha_norm : float, optional
        The normal/cross-sectional penalty parameter used during anisotropic calculation phases.
        Default is 1.5.
    alpha_tang : float, optional
        The tangential/longitudinal orientation penalty parameter used during anisotropic calculation phases.
        Default is 0.1.
    local_pca_hops : int, optional
        Number of graph hops included in each node's neighborhood when estimating
        local tangent directions. Degree-one nodes use their incident edge.
        Default is 1.
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
    decimation_mode : {'edge', 'triangle'}, optional
        Intermediate simplification method. The legacy edge mode remains the default
        for direct callers and the alternating workflow. The 'triangle' token now
        runs short-edge merging, compact-cycle collapse, and neighbourhood chain
        projection. Default is 'edge'.
    voxel_spacing : tuple of float, optional
        Physical voxel spacing used by triangle geometry and flux calculations.
    dev_elongation_ratio : float, optional
        PCA ratio at or above which degree-five neighbourhoods project to
        chains. Cycle collapse uses a separate fixed PCA cutoff of 25.
        Default is 25.
    dev_edge_thresh : float, optional
        Short-edge threshold as a fraction of minimum voxel spacing in triangle
        mode. Default is 0.05.
    dev_peri_ratio : float, optional
        Multiplier of the smallest voxel-face perimeter for cycle eligibility.
        Eligible cycles also require a PCA ratio below the fixed cutoff of 25.
        Default is 1.0.

    Returns
    -------
    X : numpy.ndarray
        Contracted centerline coordinates as an (M, 3) matrix.
    adj : scipy.sparse.csr_matrix
        Decimated skeleton topology graph connectivity representation of shape (M, M).

    Raises
    ------
    ValueError
        If `w_H_medial` is smaller than 1.0.
    """
    if not np.isfinite(w_H_medial) or w_H_medial < 1.0:
        raise ValueError('w_H_medial must be finite and >= 1.0.')
    if solver not in ('CG', 'AMGCG', 'LU'):
        raise ValueError('solver must be CG, AMGCG, or LU.')
    if max_iter < 1 or decimate_every < 1:
        raise ValueError('max_iter and decimate_every must be positive.')
    if decimation_mode not in ('edge', 'triangle'):
        raise ValueError("decimation_mode must be 'edge' or 'triangle'.")
    voxel_spacing = np.asarray(voxel_spacing, dtype=float)
    if (
        voxel_spacing.shape != (3,)
        or np.any(voxel_spacing <= 0)
        or np.any(~np.isfinite(voxel_spacing))
    ):
        raise ValueError('voxel_spacing must contain three positive finite values.')
    if not np.isfinite(dev_elongation_ratio) or dev_elongation_ratio <= 1.0:
        raise ValueError('dev_elongation_ratio must be finite and greater than 1.')
    if decimation_mode == 'triangle' and (
        not np.isfinite(dev_edge_thresh) or dev_edge_thresh < 0
    ):
        raise ValueError('dev_edge_thresh must be finite and nonnegative.')
    if decimation_mode == 'triangle' and (
        not np.isfinite(dev_peri_ratio) or dev_peri_ratio <= 0
    ):
        raise ValueError('dev_peri_ratio must be finite and positive.')
    if decimation_mode == 'triangle' and (
        binary_segmentation is None
        or np.asarray(binary_segmentation).ndim != 3
        or not np.any(binary_segmentation)
    ):
        raise ValueError('triangle decimation requires a nonempty 3D segmentation.')

    X = X_init.copy().astype(float)
    adj = adj_init.copy()

    # Conditional 3D EDT & Hard-Voxel Constraint Lookup Precomputation
    edt_volume = None
    closest_vessels_indices = None
    medialness = None
    flux_sampler = None

    needs_edt = use_edt or enforce_containment or w_H_medial > 1.0

    if needs_edt and binary_segmentation is not None:
        print('Computing 3D EDT Map and boundary projection lookup tensors...')
        edt_volume = ndimage.distance_transform_edt(binary_segmentation)
        if enforce_containment:
            # Inverse transform tells background voxels how far they are from the foreground target mask
            _, closest_vessels_indices = ndimage.distance_transform_edt(
                binary_segmentation == 0, return_indices=True
            )

        if w_H_medial > 1.0:
            medialness = compute_medialness(edt_volume, X)
    elif needs_edt and binary_segmentation is None:
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
        f' -{edt_string} min_edge_length (legacy edge threshold)={min_edge_length}\n\n'
        f'Options:\n'
        f' - Anisotropic={use_anisotropic}\n'
        f' - EDT={use_edt}\n'
        f' - Medial Boost={w_H_medial if medialness is not None else False}\n'
        f' - Hard Containment={enforce_containment}\n'
        f' - Decimation mode={decimation_mode}\n'
        f' - Decimation step={decimate_every}\n'
    )

    stable_steps = 0
    converged = False
    final_triangle_collapse = False
    for i in range(max_iter + 1):
        if i == max_iter and not final_triangle_collapse:
            break
        n_vertices = X.shape[0]
        iteration_start = X.copy()
        preprojected = 0
        preproject_max_displacement = 0.0
        if enforce_containment:
            X, preprojected = _project_escaped_nodes(
                X, binary_segmentation, closest_vessels_indices
            )
            preproject_max_displacement = np.max(
                np.linalg.norm(X - iteration_start, axis=1)
            )

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

        # 4. Explicit Hard-Voxel Containment Constraint Projection
        if enforce_containment:
            X_next, escaped_count = _project_escaped_nodes(
                X_next, binary_segmentation, closest_vessels_indices
            )
            if escaped_count:
                max_pull += f' [Projected: {escaped_count} escaped nodes]'

        movements = np.linalg.norm(X_next - iteration_start, axis=1)
        displacement = np.mean(movements)
        max_displacement = np.max(movements)
        X = X_next

        iteration_label = f'{i + 1}/{max_iter}'
        if i == max_iter:
            iteration_label += ' (post-collapse update)'
        preprojection_note = (
            f' [Pre-projected: {preprojected} escaped nodes]' if preprojected else ''
        )
        print(
            f'Iter {iteration_label} - Remaining Nodes: {X.shape[0]} - '
            f'Mean Drift: {displacement:.5f} - Max Drift: {max_displacement:.5f}'
            f'{max_pull}{preprojection_note}'
        )

        small_step = (
            max_displacement < tol
            and preproject_max_displacement < tol
            and step_solved
        )
        structural_change = False
        if decimation_mode == 'edge' and (
            (i + 1) % decimate_every == 0 or small_step or i + 1 == max_iter
        ):
            X, adj, medialness = edge_collapse_decimation(
                X, adj, min_edge_length, medialness=medialness, w_H_medial=w_H_medial
            )
            structural_change = len(X) != n_vertices
        elif (
            decimation_mode == 'triangle'
            and i < max_iter
            and (i + 1) % decimate_every == 0
        ):
            if flux_sampler is None:
                flux_sampler = DistanceFluxSampler(binary_segmentation, voxel_spacing)
            X, adj, medialness, structural_change = triangle_collapse_decimation(
                X,
                adj,
                binary_segmentation,
                voxel_spacing,
                elongation_ratio=dev_elongation_ratio,
                medialness=medialness,
                flux_sampler=flux_sampler,
                dev_edge_thresh=dev_edge_thresh,
                dev_peri_ratio=dev_peri_ratio,
            )
            if i + 1 == max_iter and structural_change:
                final_triangle_collapse = True
        if structural_change:
            stable_steps = 0
        else:
            stable_steps = (
                stable_steps + 1 if small_step and len(X) == n_vertices else 0
            )
        if stable_steps >= 3:
            print('Contraction geometry stable.')
            converged = True
            break
    if not converged:
        if provisional:
            return X, adj
        warnings.warn(
            'Contraction reached max_iter without stable geometry; these coordinates '
            'are provisional and require foreground-guided refinement.',
            RuntimeWarning,
            stacklevel=2,
        )

    return X, adj
