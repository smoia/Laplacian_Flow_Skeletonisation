import os

import numpy as np
from joblib import delayed
from scipy.ndimage import find_objects
from scipy.spatial import KDTree
from tqdm_joblib import ParallelPbar

from .alternating import alternating_graph_skeletonisation
from .contraction import laplacian_graph_contraction
from .graph import compute_sparse_adjacency_matrix
from .refinement import refine_graph


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
    w_H_medial,
    tol,
    init_graph_adj,
    local_pca_hops,
    decimate_every,
    min_edge_length,
    num_features,
    solver,
    merge_tolerance=0.25,
    alternating=False,
    voxel_spacing=(1.0, 1.0, 1.0),
    contraction_steps=5,
    alter_init_thinning=False,
    dev_contra_graph=False,
    dev_elongation_ratio=25.0,
    dev_edge_thresh=0.05,
    dev_peri_ratio=1.0,
):
    """
    Worker function to process a single connected component label.

    Parameters
    ----------
    label_id : int
        ID of current label
    cropped_label : np.ndarray
        Boolean component mask with a one-voxel background halo.
    offset_origin : tuple
        Global-volume origin corresponding to local crop coordinate zero.
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
    w_H_medial : float
        Multiplicative retention boost applied to nodes at or around inscribed-sphere
        centres. A value of 1.0 disables the boost.
    tol : float
        Maximum contraction displacement in voxels for three stable steps.
    init_graph_adj : {6, 18, 26}, int
        Voxel-neighborhood connectivity used to construct the initial graph.
    local_pca_hops : int
        Number of graph hops included in each node's neighborhood when estimating
        local tangent directions.
    decimate_every : int
        Frequency cadence interval defining how many contraction loop steps occur before
        triggering an edge-collapse decimation execution.
    min_edge_length : float
        The Euclidean spatial threshold criteria below which two connected nodes undergo
        structural merging, expressed as a fraction of the isotropic voxel length.
    num_features : int
        Number of extracted labels.
    solver : ['LU', 'CG', 'AMGCG'], string, optional
        The solver to use to solve the linear system Ax = b. LU uses SuperLU, a direct
        solver, CG uses Conjugate Gradient (iterative solver), better for memory on big
        data, AMGCG constructs an Algebraic Multigrid (AMG) preconditioner before
        running CG, which makes it far faster, but may require a tad more memory.
    merge_tolerance : float, optional
        Maximum branch simplification error in voxels; tunnels are always preserved.
    alternating : bool, optional
        Route the component through the experimental alternating workflow.
    voxel_spacing : tuple of float, optional
        Physical voxel spacing used by EDT and geometric-distance calculations.
    contraction_steps : int, optional
        Maximum contraction iterations per alternating cycle. Default is 5.
    alter_init_thinning : bool, optional
        Enable initial branch references and constrained alternating geometry.
        Default False; ignored by the default workflow.
    dev_contra_graph : bool, optional
        Append pre-refinement coordinates and adjacency in the default workflow.
    dev_elongation_ratio : float, optional
        PCA ratio for neighbourhood projection. Cycle collapse uses a fixed
        cutoff of 25.
    dev_edge_thresh : float, optional
        Short-edge diameter fraction in the default workflow. Default is 0.05.
    dev_peri_ratio : float, optional
        Smallest voxel-face perimeter multiplier for compact cycles.
        Default is 1.0.

    Returns
    -------
    label_id : int
        ID of current label (For tracking)
    contracted_X : numpy.ndarray
        An (M, 3) matrix mapping the continuous 3D spatial points along the skeleton path.
    final_adj : scipy.sparse.csr_matrix
        The resulting graph sparse adjacency connectivity representation of shape (M, M).
    reference_voxels : numpy.ndarray
        Digital skeleton coordinates in the global volume.
    edge_paths : dict
        Fitted edge polylines in global coordinates, before sampling reduction.
    intermediate_X, intermediate_adj : numpy.ndarray, scipy.sparse.csr_matrix
        Appended only when dev_contra_graph is enabled in the default workflow.
        Global coordinates and adjacency before refinement; components with at
        most three voxels retain their initial graph because contraction is skipped.
    """
    X_init_local = np.argwhere(cropped_label).astype(np.uint16)
    intermediate = ()
    if dev_contra_graph and not alternating and len(X_init_local) <= 3:
        intermediate = (
            X_init_local + np.array(offset_origin, dtype=np.float32),
            compute_sparse_adjacency_matrix(KDTree(X_init_local), init_graph_adj),
        )
    # Skip small noise components
    if len(X_init_local) <= 3 and (not alternating or alter_init_thinning):
        refined, adj_sparse, voxels, paths = refine_graph(
            cropped_label, X_init_local, merge_tolerance, w_H_medial, return_paths=True
        )
        X_init_global = refined + np.array(offset_origin, dtype=np.float32)
        return (
            label_id, X_init_global, adj_sparse, voxels + offset_origin,
            {edge: path + offset_origin for edge, path in paths.items()},
        ) + intermediate

    print(
        f'\n--- Processing Label {label_id}/{num_features} ({X_init_local.sum()} voxels) ---'
    )

    if alternating:
        label_X_local, label_adj, voxels, paths = alternating_graph_skeletonisation(
            cropped_label,
            spacing=voxel_spacing,
            contraction_steps=contraction_steps,
            alter_init_thinning=alter_init_thinning,
            use_anisotropic=use_anisotropic,
            w_L=w_L,
            w_H_base=w_H_base,
            w_H_medial=w_H_medial,
            tol=tol,
            local_pca_hops=local_pca_hops,
            solver=solver,
            merge_tolerance=merge_tolerance,
        )
    else:
        # Run the established contraction followed by complete refinement.
        print(f'Computing {init_graph_adj}-connected voxel graph...')
        adj_sparse = compute_sparse_adjacency_matrix(
            KDTree(X_init_local), init_graph_adj
        )
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
            w_H_medial=w_H_medial,
            tol=tol,
            local_pca_hops=local_pca_hops,
            decimate_every=decimate_every,
            min_edge_length=min_edge_length,
            solver=solver,
            decimation_mode='triangle',
            voxel_spacing=voxel_spacing,
            dev_elongation_ratio=dev_elongation_ratio,
            dev_edge_thresh=dev_edge_thresh,
            dev_peri_ratio=dev_peri_ratio,
        )

        if dev_contra_graph:
            intermediate = (
                label_X_local + np.array(offset_origin, dtype=np.float32),
                label_adj.copy(),
            )

        label_X_local, label_adj, voxels, paths = refine_graph(
            cropped_label, label_X_local, merge_tolerance, w_H_medial, return_paths=True
        )
    label_X_global = label_X_local + np.array(offset_origin, dtype=np.float32)

    return (
        label_id, label_X_global, label_adj, voxels + offset_origin,
        {edge: path + offset_origin for edge, path in paths.items()},
    ) + intermediate


def _padded_component_crop(labeled_volume, label_id, bounding_box):
    """Return one tightly cropped component with a one-voxel background halo."""
    component = labeled_volume[bounding_box] == label_id
    padded_component = np.pad(component, 1, mode='constant', constant_values=False)
    offset_origin = tuple(axis.start - 1 for axis in bounding_box)
    return padded_component, offset_origin


def process_components(
    labeled_volume,
    num_features,
    use_edt,
    use_anisotropic,
    enforce_containment,
    beta_edt,
    w_L,
    w_H_base,
    w_H_medial,
    tol,
    init_graph_adj,
    local_pca_hops,
    decimate_every,
    min_edge_length,
    n_jobs,
    solver,
    merge_tolerance=0.25,
    alternating=False,
    voxel_spacing=(1.0, 1.0, 1.0),
    contraction_steps=5,
    alter_init_thinning=False,
    dev_contra_graph=False,
    dev_elongation_ratio=25.0,
    dev_edge_thresh=0.05,
    dev_peri_ratio=1.0,
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

    slices_list = find_objects(labeled_volume)

    tasks = []
    for label_id in range(1, num_features + 1):
        bbox_slice = slices_list[label_id - 1]

        if bbox_slice is None:
            continue

        cropped_label, offset_origin = _padded_component_crop(
            labeled_volume, label_id, bbox_slice
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
                w_H_medial,
                tol,
                init_graph_adj,
                local_pca_hops,
                decimate_every,
                min_edge_length,
                num_features,
                solver,
                merge_tolerance,
                alternating,
                voxel_spacing,
                contraction_steps,
                alter_init_thinning,
                dev_contra_graph,
                dev_elongation_ratio,
                dev_edge_thresh,
                dev_peri_ratio,
            )
        )

    results = ParallelPbar('Skeletonising')(n_jobs=n_workers, batch_size=1)(tasks)

    return results
