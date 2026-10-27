"""Experimental alternating geometric contraction and topological thinning."""

import warnings

import numpy as np
from scipy import ndimage, sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree, cKDTree

from .contraction import laplacian_graph_contraction
from .graph import compute_sparse_adjacency_matrix
from .medial import estimate_graph_tangents, transverse_edt_targets
from .refinement import (
    _simplify_paths,
    _thin_foreground,
    _tunnel_graph,
    branch_complex_signature,
    foreground_topology,
    thin_foreground_sweep,
)
from .utils import (
    _edge_voxels,
    _foreground_voxel,
    _nonincident_intersections,
    _sweep_contained,
)


def _adjacency_from_edges(n_vertices, edges):
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    if len(edges):
        rows = np.concatenate((edges[:, 0], edges[:, 1]))
        cols = np.concatenate((edges[:, 1], edges[:, 0]))
    else:
        rows = cols = np.array([], dtype=int)
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=(n_vertices, n_vertices),
    )


def _resample_edges(coordinates, edges, spacing):
    """Insert graph vertices at half-voxel physical arc-length intervals."""
    points = [point.copy() for point in np.asarray(coordinates, dtype=float)]
    sampled_edges = []
    step = 0.5 * float(np.min(spacing))
    for u, v in np.asarray(edges, dtype=int).reshape(-1, 2):
        length = np.linalg.norm((points[v] - points[u]) * spacing)
        pieces = max(1, int(np.ceil(length / step)))
        previous = int(u)
        for part in range(1, pieces):
            points.append(points[u] + part * (points[v] - points[u]) / pieces)
            current = len(points) - 1
            sampled_edges.append((previous, current))
            previous = current
        sampled_edges.append((previous, int(v)))
    return (
        np.asarray(points, dtype=float),
        np.asarray(sampled_edges, dtype=int).reshape(-1, 2),
    )


def _valid_graph_update(old, proposed, edges, mask, spacing):
    """Check containment, deformation sweep and unintended crossings."""
    try:
        for point in proposed:
            _foreground_voxel(point, mask.shape, mask)
        for u, v in edges:
            _edge_voxels(proposed[u], proposed[v], mask.shape, mask)
            if not _sweep_contained(old[u], old[v], proposed[u], proposed[v], mask):
                return False
    except ValueError:
        return False
    return not _nonincident_intersections(proposed, edges, spacing)


def _terminal_anchors(mask, edt, spacing):
    """Retain one medial voxel at each narrow end face of an elongated branch."""
    anchors = np.zeros_like(mask, dtype=bool)
    labels, count = ndimage.label(
        mask, structure=np.ones((3, 3, 3), dtype=bool)
    )
    for label in range(1, count + 1):
        component = labels == label
        points = np.argwhere(component)
        extents = np.ptp(points, axis=0) + 1
        if not any(
            extents[axis] >= 4 and extents[axis] >= 1.5 * min(
                extents[other] for other in range(3) if other != axis
            )
            for axis in range(3)
        ):
            continue
        if foreground_topology(component)[1]:
            continue
        for axis in range(3):
            if extents[axis] < 4 or extents[axis] < 1.5 * min(
                extents[other] for other in range(3) if other != axis
            ):
                continue
            faces = [
                points[points[:, axis] == value]
                for value in (np.min(points[:, axis]), np.max(points[:, axis]))
            ]
            narrow = min(len(face) for face in faces)
            for face in faces:
                if len(face) > 2 * narrow:
                    continue
                radii = edt[tuple(face.T)]
                best = face[np.isclose(radii, np.max(radii), rtol=0.0, atol=1e-12)]
                centre = np.mean(face, axis=0)
                choice = np.argmin(np.sum(((best - centre) * spacing) ** 2, axis=1))
                anchors[tuple(best[choice])] = True
    return anchors


def _skeleton_state(mask, guidance, anchors=None):
    voxels = _thin_foreground(mask, guidance, protected=anchors).astype(int)
    edges, _ = _tunnel_graph(voxels)
    return voxels, edges, branch_complex_signature(voxels, edges)


def _critical_branch_data(coordinates, edges, edt, spacing):
    """Return critical-node radii and shortest incident branch lengths."""
    neighbours = [set() for _ in coordinates]
    for u, v in edges:
        neighbours[int(u)].add(int(v))
        neighbours[int(v)].add(int(u))
    degrees = np.array([len(items) for items in neighbours])
    critical = np.flatnonzero(degrees != 2)
    radii = {}
    lengths = {}
    physical = coordinates * spacing
    for start in critical:
        branch_radii = []
        branch_lengths = []
        for neighbour in neighbours[start]:
            previous, current = int(start), int(neighbour)
            distance = np.linalg.norm(physical[current] - physical[previous])
            samples = [edt[tuple(coordinates[current].astype(int))]]
            while degrees[current] == 2:
                following = next(iter(neighbours[current] - {previous}))
                distance += np.linalg.norm(physical[following] - physical[current])
                previous, current = current, following
                samples.append(edt[tuple(coordinates[current].astype(int))])
            branch_lengths.append(float(distance))
            branch_radii.append(float(np.max(samples)))
        if branch_lengths:
            lengths[int(start)] = min(branch_lengths)
            if degrees[start] == 1:
                radii[int(start)] = branch_radii[0]
            else:
                radii[int(start)] = float(np.median(branch_radii))
        else:
            lengths[int(start)] = 0.0
            radii[int(start)] = float(edt[tuple(coordinates[start].astype(int))])
    return degrees, critical, radii, lengths


def _match_critical_nodes(reference, reference_edges, current, current_edges, spacing):
    """Match movable critical points to fixed references by degree and distance."""
    ref_degree = np.bincount(
        np.asarray(reference_edges).ravel(), minlength=len(reference)
    )
    ref_critical = np.flatnonzero(ref_degree != 2)
    cur_degree = np.bincount(
        np.asarray(current_edges).ravel(), minlength=len(current)
    )
    mapping = {}
    for degree in sorted(set(ref_degree[ref_critical])):
        ref_group = ref_critical[ref_degree[ref_critical] == degree]
        cur_group = np.flatnonzero((cur_degree == degree) & (cur_degree != 2))
        if len(ref_group) != len(cur_group):
            return None
        costs = np.linalg.norm(
            (current[cur_group, None] - reference[ref_group][None]) * spacing,
            axis=2,
        )
        rows, cols = linear_sum_assignment(costs)
        mapping.update(
            {int(cur_group[row]): int(ref_group[col]) for row, col in zip(rows, cols)}
        )
    return mapping


def _junction_targets(coordinates, adjacency, ridge_targets, tangents, spacing):
    """Fit each junction to the least-squares meeting of incident ridge lines."""
    targets = ridge_targets.copy()
    degree = adjacency.getnnz(axis=1)
    physical_ridges = ridge_targets * spacing
    for vertex in np.flatnonzero(degree >= 3):
        begin, end = adjacency.indptr[vertex : vertex + 2]
        neighbours = adjacency.indices[begin:end]
        system = np.zeros((3, 3), dtype=float)
        target = np.zeros(3, dtype=float)
        for neighbour in neighbours:
            direction = tangents[neighbour]
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            direction = direction / norm
            normal = np.eye(3) - np.outer(direction, direction)
            system += normal
            target += normal @ physical_ridges[neighbour]
        if np.linalg.matrix_rank(system) >= 2:
            targets[vertex] = np.linalg.lstsq(system, target, rcond=None)[0] / spacing
    return targets


def fit_branch_complex(
    reference_voxels,
    reference_edges,
    skeleton_voxels,
    skeleton_edges,
    original_mask,
    original_edt,
    spacing,
    merge_tolerance,
    max_iterations=20,
    preserve_reference=True,
):
    """Fit branches, optionally enforcing reference and foreground constraints."""
    coordinates, edges = _resample_edges(skeleton_voxels, skeleton_edges, spacing)
    if preserve_reference:
        try:
            for point in coordinates:
                _foreground_voxel(point, original_mask.shape, original_mask)
            for u, v in edges:
                _edge_voxels(
                    coordinates[u], coordinates[v], original_mask.shape, original_mask
                )
        except ValueError as error:
            raise RuntimeError(
                'No contained initial graph geometry is available.'
            ) from error
        if _nonincident_intersections(coordinates, edges, spacing):
            raise RuntimeError('Initial graph geometry contains an unintended crossing.')
    adjacency = _adjacency_from_edges(len(coordinates), edges)
    degree = adjacency.getnnz(axis=1)
    mapping = {}
    if preserve_reference:
        reference = np.asarray(reference_voxels, dtype=float)
        reference_edges = np.asarray(reference_edges, dtype=int).reshape(-1, 2)
        mapping = _match_critical_nodes(
            reference, reference_edges, coordinates, edges, spacing
        )
        if mapping is None:
            raise RuntimeError(
                'Final skeleton no longer matches the reference branch complex.'
            )
        anchors = coordinates.copy()
        critical_degree, _, critical_radii, critical_lengths = _critical_branch_data(
            np.asarray(skeleton_voxels, dtype=float), skeleton_edges, original_edt, spacing
        )
    stable = False
    rejected_updates = 0
    for _ in range(max_iterations):
        local_radii = ndimage.map_coordinates(
            original_edt, coordinates.T, order=1, mode='nearest'
        )
        window = max(2.0 * float(np.min(spacing)), float(np.median(local_radii)))
        tangents = estimate_graph_tangents(coordinates, adjacency, spacing, window)
        ridge = transverse_edt_targets(
            original_edt, original_mask, coordinates, tangents, spacing,
            enforce_containment=preserve_reference,
        )
        ridge = _junction_targets(coordinates, adjacency, ridge, tangents, spacing)
        neighbour_sum = adjacency.astype(float).dot(coordinates)
        neighbour_mean = neighbour_sum / np.maximum(degree[:, None], 1)
        if not preserve_reference:
            neighbour_mean[degree == 0] = coordinates[degree == 0]
        proposal = 0.4 * ridge + 0.6 * neighbour_mean
        if not np.all(np.isfinite(proposal)):
            raise RuntimeError('Graph fitting produced non-finite coordinates.')

        # Retain the longitudinal endpoint/branch cap while allowing transverse
        # displacement to the centre of an even-width EDT plateau.
        for current_index in mapping:
            reference_position = anchors[current_index]
            radius = critical_radii[current_index]
            branch_length = critical_lengths[current_index]
            coefficient = 0.25 if critical_degree[current_index] == 1 else 0.5
            cap = min(coefficient * radius, 0.2 * branch_length)
            displacement = (proposal[current_index] - reference_position) * spacing
            direction = tangents[current_index]
            norm = np.linalg.norm(direction)
            if norm:
                axial = float(displacement @ (direction / norm))
                displacement -= (axial - np.clip(axial, -cap, cap)) * direction / norm
                proposal[current_index] = reference_position + displacement / spacing
            elif cap == 0:
                proposal[current_index] = reference_position

        accepted = coordinates
        accepted_trial = False
        for fraction in (1.0, 0.5, 0.25, 0.125):
            trial = coordinates + fraction * (proposal - coordinates)
            if not preserve_reference or _valid_graph_update(
                coordinates, trial, edges, original_mask, spacing
            ):
                accepted = trial
                accepted_trial = True
                break
        if not accepted_trial:
            rejected_updates += 1
        movement = np.max(np.linalg.norm((accepted - coordinates) * spacing, axis=1))
        coordinates = accepted
        if movement < 0.05 * float(np.min(spacing)):
            stable = True
            break

    local_radii = ndimage.map_coordinates(
        original_edt, coordinates.T, order=1, mode='nearest'
    )
    direction_window = max(
        2.0 * float(np.min(spacing)), float(np.median(local_radii))
    )
    tangents = estimate_graph_tangents(
        coordinates, adjacency, spacing, direction_window
    )
    ridge = transverse_edt_targets(
        original_edt, original_mask, coordinates, tangents, spacing,
        enforce_containment=preserve_reference,
    )
    ridge = _junction_targets(coordinates, adjacency, ridge, tangents, spacing)
    ridge_distance = np.linalg.norm((coordinates - ridge) * spacing, axis=1)
    ridge_tolerance = np.maximum(
        0.5 * float(np.min(spacing)), 0.2 * local_radii
    )
    diagnostics = {
        'stable': stable,
        'fallback': bool(rejected_updates),
        'ridge_satisfied': bool(np.all(ridge_distance <= ridge_tolerance)),
        'mean_ridge_distance': float(np.mean(ridge_distance)),
        'max_ridge_distance': float(np.max(ridge_distance)),
    }
    coordinates, simplified_edges, paths = _simplify_paths(
        coordinates, edges,
        original_mask if preserve_reference else None, merge_tolerance
    )
    if preserve_reference:
        fitted_signature = branch_complex_signature(coordinates, simplified_edges)
        if fitted_signature != branch_complex_signature(reference, reference_edges):
            raise RuntimeError('Graph fitting changed the reference branch complex.')
        for point in coordinates:
            _foreground_voxel(point, original_mask.shape, original_mask)
        for path in paths.values():
            for start, end in zip(path[:-1], path[1:]):
                _edge_voxels(start, end, original_mask.shape, original_mask)
    return (
        coordinates,
        _adjacency_from_edges(len(coordinates), simplified_edges),
        paths,
        diagnostics,
    )


def alternating_graph_skeletonisation(
    mask,
    spacing=(1.0, 1.0, 1.0),
    contraction_steps=5,
    max_cycles=100,
    use_anisotropic=True,
    w_L=0.5,
    w_H_base=0.5,
    w_H_medial=1.0,
    tol=0.05,
    local_pca_hops=1,
    solver='CG',
    merge_tolerance=0.25,
    alter_init_thinning=False,
):
    """Alternate bounded contraction and one-layer topology-safe thinning.

    The original foreground and physical EDT stay fixed. Local thinning uses
    the default simple-voxel and tip-protection rules. With alter_init_thinning,
    an initial branch reference additionally constrains thinning and fitting,
    and hard containment is enabled. Without it, geometry is unconstrained and
    candidate skeletons use the current contraction/EDT guidance.
    """
    foreground = np.asarray(mask, dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    if foreground.ndim != 3 or not np.any(foreground):
        raise ValueError('Alternating skeletonisation requires a nonempty 3D mask.')
    if contraction_steps < 1 or max_cycles < 1:
        raise ValueError('contraction_steps and max_cycles must be positive.')
    if spacing.shape != (3,) or np.any(spacing <= 0) or np.any(~np.isfinite(spacing)):
        raise ValueError('spacing must contain three positive finite values.')
    if alter_init_thinning and foreground_topology(foreground)[2]:
        raise ValueError(
            'Foreground contains enclosed cavities; a curve graph cannot preserve them.'
        )

    original_edt = ndimage.distance_transform_edt(
        np.pad(foreground, 1), sampling=spacing
    )[1:-1, 1:-1, 1:-1]
    anchors = reference_voxels = reference_edges = reference_signature = None
    topology_guidance = None
    if alter_init_thinning:
        anchors = _terminal_anchors(foreground, original_edt, spacing)
        initial = np.argwhere(foreground).astype(float)
        topology_guidance = initial.copy()
        reference_voxels, reference_edges, reference_signature = _skeleton_state(
            foreground, topology_guidance, anchors
        )
        reference_mask = np.zeros_like(foreground)
        reference_mask[tuple(reference_voxels.T)] = True
        original_topology = foreground_topology(foreground)[:2]
        if foreground_topology(reference_mask)[:2] != original_topology:
            raise RuntimeError('Reference digital skeleton changed foreground topology.')
    working = foreground.copy()
    previous_voxels = None
    previous_guide = None
    last_skeleton_voxels = reference_voxels
    last_skeleton_edges = reference_edges
    outcome = 'budget-limited'
    stagnant_cycles = 0

    for cycle in range(max_cycles):
        voxels = np.argwhere(working).astype(float)
        adjacency = compute_sparse_adjacency_matrix(KDTree(voxels), 26)
        if previous_voxels is None:
            contraction_start = voxels
        else:
            correspondence = cKDTree(previous_voxels).query(voxels)[1]
            contraction_start = voxels + (
                previous_guide[correspondence] - previous_voxels[correspondence]
            )
        contracted, _ = laplacian_graph_contraction(
            contraction_start,
            adjacency,
            binary_segmentation=foreground,
            use_edt=False,
            use_anisotropic=use_anisotropic,
            enforce_containment=alter_init_thinning,
            w_L=w_L,
            w_H_base=w_H_base,
            w_H_medial=w_H_medial,
            max_iter=contraction_steps,
            tol=tol,
            # Zero merge length suppresses node merging and adaptive decimation.
            min_edge_length=0.0,
            local_pca_hops=local_pca_hops,
            solver=solver,
            provisional=True,
        )
        sampled_radii = original_edt[tuple(voxels.astype(int).T)]
        direction_window = max(
            2.0 * float(np.min(spacing)), float(np.median(sampled_radii))
        )
        tangents = estimate_graph_tangents(
            contracted, adjacency, spacing, direction_window
        )
        ridge_targets = transverse_edt_targets(
            original_edt, foreground, contracted, tangents, spacing,
            enforce_containment=alter_init_thinning,
        )
        guide = 0.5 * contracted + 0.5 * ridge_targets
        def preserves_reference(candidate):
            return (
                _skeleton_state(candidate, topology_guidance, anchors)[2]
                == reference_signature
            )

        proposed, deleted = thin_foreground_sweep(
            working,
            guide,
            original_edt,
            spacing,
            validator=preserves_reference if alter_init_thinning else None,
            protected=anchors,
        )
        if alter_init_thinning:
            proposed_voxels, proposed_edges, proposed_signature = _skeleton_state(
                proposed, topology_guidance, anchors
            )
        else:
            proposed_voxels = _thin_foreground(proposed, guide).astype(int)
            proposed_edges, _ = _tunnel_graph(proposed_voxels)
        if alter_init_thinning and proposed_signature != reference_signature:
            raise RuntimeError('A validated thinning sweep changed the branch complex.')
        if alter_init_thinning:
            candidate_mask = np.zeros_like(foreground)
            candidate_mask[tuple(proposed_voxels.T)] = True
            if foreground_topology(candidate_mask)[:2] != original_topology:
                raise RuntimeError(
                    'Candidate digital skeleton changed foreground topology.'
                )
        last_skeleton_voxels = proposed_voxels
        last_skeleton_edges = proposed_edges
        if deleted == 0:
            previous_voxels, previous_guide = voxels, guide
            if np.count_nonzero(working) == len(last_skeleton_voxels):
                outcome = 'digital-complete'
                break
            stagnant_cycles += 1
            if stagnant_cycles >= 5:
                outcome = 'stagnated'
                break
            print(
                f'Alternating cycle {cycle + 1}/{max_cycles}: no accepted '
                f'deletions ({stagnant_cycles}/5 stagnant cycles).'
            )
            continue
        working = proposed
        previous_voxels, previous_guide = voxels, guide
        stagnant_cycles = 0
        print(
            f'Alternating cycle {cycle + 1}/{max_cycles}: deleted {deleted} voxels; '
            f'{np.count_nonzero(working)} remain.'
        )
    else:
        warnings.warn(
            'Alternating skeletonisation reached its cycle budget; returning the '
            'last digital skeleton.',
            RuntimeWarning,
            stacklevel=2,
        )

    if previous_guide is None:
        previous_guide = np.argwhere(working).astype(float)
    skeleton_voxels = last_skeleton_voxels
    skeleton_edges = last_skeleton_edges
    if alter_init_thinning:
        final_signature = branch_complex_signature(skeleton_voxels, skeleton_edges)
        if final_signature != reference_signature:
            raise RuntimeError(
                'Alternating skeletonisation changed the reference branch complex.'
            )
    coordinates, adjacency, paths, diagnostics = fit_branch_complex(
        reference_voxels,
        reference_edges,
        skeleton_voxels,
        skeleton_edges,
        foreground,
        original_edt,
        spacing,
        merge_tolerance,
        preserve_reference=alter_init_thinning,
    )
    if alter_init_thinning:
        output_mask = np.zeros_like(foreground)
        output_mask[tuple(skeleton_voxels.T)] = True
        if foreground_topology(output_mask)[:2] != original_topology:
            raise RuntimeError('Digital skeleton changed foreground components or tunnels.')
    output_voxels = skeleton_voxels
    geometry_outcome = (
        'validated-fallback' if diagnostics['fallback']
        else 'stable' if diagnostics['stable'] and diagnostics['ridge_satisfied']
        else 'feature-limited' if diagnostics['stable'] else 'fit-budget-limited'
    )
    if outcome == 'digital-complete' and geometry_outcome == 'stable':
        outcome = 'converged'
    print(
        f'Alternating digital {outcome}; graph geometry {geometry_outcome}: '
        f'{len(coordinates)} graph nodes, '
        f'{adjacency.nnz // 2} edges; mean/max ridge distance '
        f'{diagnostics["mean_ridge_distance"]:.4g}/'
        f'{diagnostics["max_ridge_distance"]:.4g} physical units.'
    )
    return coordinates, adjacency, output_voxels, paths
