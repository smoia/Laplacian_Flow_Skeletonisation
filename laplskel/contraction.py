"""Laplacian graph contraction algorithm."""

import numpy as np
from scipy import ndimage, sparse

from .graph import compute_laplacian_matrix, edge_collapse_decimation
from .solvers import solve_cg, solve_lu


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
        _, nearest_indices = ndimage.distance_transform_edt(
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
            X_next = solve_lu(A, B)
        elif solver == 'CG':
            A = (w_L**2) * L_squared + W_H_sq
            B = W_H_sq.dot(X)
            X_next = solve_cg(A, B, X)

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
