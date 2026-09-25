"""Three-stage graph decimation for the default contraction workflow."""

import numpy as np
from scipy import ndimage, sparse


_POSITION_TOLERANCE = 1e-6
_FLUX_TOLERANCE = 1e-6
_FLUX_BATCH_SIZE = 256
_COMPACT_CYCLE_RATIO = 25.0


def _sphere_directions():
    """Return 64 deterministic, approximately uniform, antipodal directions."""
    count = 32
    indices = np.arange(count, dtype=float)
    z = (indices + 0.5) / count
    angle = indices * np.pi * (3.0 - np.sqrt(5.0))
    radius = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    hemisphere = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    return np.vstack((hemisphere, -hemisphere))


SPHERE_DIRECTIONS = _sphere_directions()


class DistanceFluxSampler:
    """Cache a signed physical distance-gradient field for flux queries."""

    def __init__(self, foreground, spacing):
        foreground = np.asarray(foreground, dtype=bool)
        spacing = np.asarray(spacing, dtype=float)
        if foreground.ndim != 3 or not np.any(foreground):
            raise ValueError('triangle decimation requires a nonempty 3D segmentation.')
        if (
            spacing.shape != (3,)
            or np.any(spacing <= 0)
            or np.any(~np.isfinite(spacing))
        ):
            raise ValueError('voxel_spacing must contain three positive finite values.')
        self.foreground = foreground
        self.spacing = spacing
        self.radius = float(np.min(spacing))
        base = np.ceil(self.radius / spacing).astype(int) + 2
        self._before = base.copy()
        self._after = base.copy()
        self._gradient = None
        self._rebuild()

    def _rebuild(self):
        padding = tuple(
            (int(before), int(after))
            for before, after in zip(self._before, self._after)
        )
        padded = np.pad(
            self.foreground, padding, mode='constant', constant_values=False
        )
        inside = ndimage.distance_transform_edt(padded, sampling=self.spacing)
        outside = ndimage.distance_transform_edt(~padded, sampling=self.spacing)
        signed_distance = inside - outside
        self._gradient = np.asarray(
            np.gradient(signed_distance, *self.spacing, edge_order=2)
        )

    def _ensure_bounds(self, sample_voxels):
        minima = np.min(sample_voxels, axis=0)
        maxima = np.max(sample_voxels, axis=0)
        required_before = np.maximum(0, np.ceil(1.0 - minima).astype(int))
        required_after = np.maximum(
            0,
            np.ceil(maxima - np.asarray(self.foreground.shape) + 2.0).astype(int),
        )
        before = np.maximum(self._before, required_before)
        after = np.maximum(self._after, required_after)
        if np.array_equal(before, self._before) and np.array_equal(after, self._after):
            return
        self._before = before
        self._after = after
        self._rebuild()

    def score(self, physical_positions):
        """Estimate average outward flux at physical-coordinate positions."""
        positions = np.asarray(physical_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError('physical_positions must have shape (N, 3).')
        if len(positions) == 0:
            return np.empty(0, dtype=float)

        scores = np.empty(len(positions), dtype=float)
        offsets = self.radius * SPHERE_DIRECTIONS
        for start in range(0, len(positions), _FLUX_BATCH_SIZE):
            stop = min(start + _FLUX_BATCH_SIZE, len(positions))
            samples = positions[start:stop, None, :] + offsets[None, :, :]
            sample_voxels = samples.reshape(-1, 3) / self.spacing
            self._ensure_bounds(sample_voxels)
            padded_voxels = sample_voxels + self._before
            values = np.column_stack(
                [
                    ndimage.map_coordinates(
                        component,
                        padded_voxels.T,
                        order=1,
                        mode='constant',
                        cval=np.nan,
                        prefilter=False,
                    )
                    for component in self._gradient
                ]
            ).reshape(stop - start, len(SPHERE_DIRECTIONS), 3)
            if not np.all(np.isfinite(values)):
                raise RuntimeError('Flux sampling exceeded the padded distance field.')
            scores[start:stop] = np.mean(
                np.sum(values * SPHERE_DIRECTIONS[None, :, :], axis=2), axis=1
            )
        return scores


def _simple_adjacency(adjacency_matrix):
    """Return a symmetric, loop-free boolean adjacency matrix."""
    adjacency = adjacency_matrix.astype(bool).tocsr()
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(False)
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return adjacency


def _edges(adjacency):
    """Return unique undirected edges as ordered pairs."""
    rows, cols = sparse.triu(adjacency, k=1).nonzero()
    return [(int(u), int(v)) for u, v in zip(rows, cols)]


def _adjacency_from_edges(n_vertices, edges):
    """Build a simple undirected graph from edge pairs."""
    edges = sorted({tuple(sorted((u, v))) for u, v in edges if u != v})
    if not edges:
        return sparse.csr_matrix((n_vertices, n_vertices), dtype=bool)
    rows = [u for u, v in edges] + [v for u, v in edges]
    cols = [v for u, v in edges] + [u for u, v in edges]
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=(n_vertices, n_vertices),
    )


def enumerate_chordless_cycles(adjacency_matrix, max_vertices=5):
    """Enumerate unique chordless cycles with three to five vertices."""
    adjacency = _simple_adjacency(adjacency_matrix)
    neighbours = [
        adjacency.indices[adjacency.indptr[u] : adjacency.indptr[u + 1]]
        for u in range(adjacency.shape[0])
    ]
    neighbour_sets = [set(int(v) for v in row) for row in neighbours]
    cycles = []

    def visit(start, path):
        current = path[-1]
        if len(path) >= 3 and start in neighbour_sets[current]:
            if path[1] < current:
                cycles.append(tuple(path))
            return  # Any extension would make the closing edge a chord.
        if len(path) == max_vertices:
            return
        for neighbour in neighbours[current]:
            neighbour = int(neighbour)
            if neighbour <= start or neighbour in path:
                continue
            # Edges to earlier interior vertices would become chords. At path
            # length two, adjacency to start is the valid triangle closure.
            if any(neighbour in neighbour_sets[u] for u in path[1:-1]):
                continue
            visit(start, path + [neighbour])

    for start in range(adjacency.shape[0]):
        for neighbour in neighbours[start]:
            if neighbour > start:
                visit(start, [start, int(neighbour)])
    return cycles


def enumerate_triangles(adjacency_matrix):
    """Return each graph triangle once, preserving the existing helper API."""
    return [
        cycle for cycle in enumerate_chordless_cycles(adjacency_matrix)
        if len(cycle) == 3
    ]


def _circumcentre(points):
    """Return the circumcentre of three non-collinear physical points."""
    a, b, c = points
    normal = np.cross(b - a, c - a)
    system = np.vstack((2.0 * (b - a), 2.0 * (c - a), normal))
    target = np.array(
        [
            np.dot(b, b) - np.dot(a, a),
            np.dot(c, c) - np.dot(a, a),
            np.dot(normal, points.mean(axis=0)),
        ]
    )
    return np.linalg.solve(system, target)


def _compact_target(points, flux_sampler, tolerance):
    centroid = points.mean(axis=0)
    candidates = np.vstack((points, centroid))
    if len(points) == 3:
        candidates = np.vstack((candidates, _circumcentre(points)))
    scores = flux_sampler.score(candidates)
    tied = np.flatnonzero(scores <= np.min(scores) + _FLUX_TOLERANCE)
    distances = np.linalg.norm(candidates[tied] - centroid, axis=1)
    nearest = tied[distances <= np.min(distances) + tolerance]
    if len(nearest) == 1:
        return candidates[nearest[0]]
    order = np.lexsort(
        (candidates[nearest, 2], candidates[nearest, 1], candidates[nearest, 0])
    )
    return candidates[nearest[order[0]]]


def _remap_graph(coordinates, adjacency, labels, positions, medialness):
    """Merge labelled vertices and retain their external connections."""
    unique, inverse = np.unique(labels, return_inverse=True)
    new_coordinates = np.asarray(
        [positions.get(int(label), coordinates[label]) for label in unique]
    )
    remapped = [
        (int(inverse[u]), int(inverse[v])) for u, v in _edges(adjacency)
    ]
    new_adjacency = _adjacency_from_edges(len(unique), remapped)
    new_medialness = None
    if medialness is not None:
        new_medialness = np.zeros(len(unique), dtype=float)
        np.maximum.at(new_medialness, inverse, medialness)
    return new_coordinates, new_adjacency, new_medialness


def _merge_short_edges(
    coordinates, adjacency, spacing, threshold, medialness,
):
    """Merge coincident or short edges with a complete-group diameter bound."""
    n_vertices = len(coordinates)
    if n_vertices < 2:
        return coordinates, adjacency, medialness, 0
    coincidence_tolerance = _POSITION_TOLERANCE * float(np.min(spacing))

    def within_limit(value):
        return value <= coincidence_tolerance or value < threshold

    physical = coordinates * spacing
    candidates = []
    for u, v in _edges(adjacency):
        length = float(np.linalg.norm(physical[u] - physical[v]))
        if within_limit(length):
            candidates.append((length, u, v))
    candidates.sort()
    groups = {index: [index] for index in range(n_vertices)}
    labels = np.arange(n_vertices)
    coordinate_sums = coordinates.copy()
    member_counts = np.ones(n_vertices, dtype=int)
    for _, u, v in candidates:
        root_u, root_v = int(labels[u]), int(labels[v])
        if root_u == root_v:
            continue
        members = groups[root_u] + groups[root_v]
        group_points = physical[members]
        diameter = np.max(
            np.linalg.norm(
                group_points[:, None, :] - group_points[None, :, :], axis=2
            )
        )
        if not within_limit(diameter):
            continue
        root = min(root_u, root_v)
        other = max(root_u, root_v)
        groups[root] = members
        del groups[other]
        labels[members] = root
        coordinate_sums[root] += coordinate_sums[other]
        member_counts[root] += member_counts[other]
    positions = {
        root: coordinate_sums[root] / member_counts[root]
        for root, members in groups.items()
        if len(members) > 1
    }
    merged_groups = len(positions)
    if not merged_groups:
        return coordinates, adjacency, medialness, 0
    result = _remap_graph(coordinates, adjacency, labels, positions, medialness)
    return *result, merged_groups


def _pca(points, tolerance):
    """Return leading direction, elongation ratio and collinearity."""
    centre = points.mean(axis=0)
    centred = points - centre
    eigenvalues, eigenvectors = np.linalg.eigh(centred.T.dot(centred) / len(points))
    tangent = eigenvectors[:, -1]
    residual = centred - centred.dot(tangent)[:, None] * tangent
    collinear = np.max(np.linalg.norm(residual, axis=1)) <= tolerance
    ratio = (
        np.inf if collinear or eigenvalues[-2] <= 0
        else eigenvalues[-1] / eigenvalues[-2]
    )
    return tangent, ratio, collinear


def _collapse_cycles(
    coordinates, adjacency, spacing, peri_ratio,
    medialness, flux_sampler,
):
    """Collapse chordless cycles below the perimeter and fixed PCA limits."""
    cycles = enumerate_chordless_cycles(adjacency)
    by_length = {length: 0 for length in (3, 4, 5)}
    physical = coordinates * spacing
    limit = peri_ratio * 2.0 * np.sum(np.sort(spacing)[:2])
    candidates = []
    for cycle in cycles:
        by_length[len(cycle)] += 1
        points = physical[list(cycle)]
        perimeter = np.sum(np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1))
        if perimeter < limit:
            candidates.append((perimeter, cycle))
    candidates.sort(key=lambda item: (item[0], item[1]))

    tolerance = _POSITION_TOLERANCE * float(np.min(spacing))
    compact_candidates = []
    for perimeter, cycle in candidates:
        points = physical[list(cycle)]
        diameter = np.max(
            np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        )
        if diameter <= tolerance:
            compact_candidates.append((perimeter, cycle, True))
            continue
        _, ratio, collinear = _pca(points, tolerance)
        if not collinear and ratio < _COMPACT_CYCLE_RATIO:
            compact_candidates.append((perimeter, cycle, False))

    labels = np.arange(len(coordinates))
    positions = {}
    used = set()
    accepted = 0
    for _, cycle, coincident in compact_candidates:
        if used.intersection(cycle):
            continue
        points = physical[list(cycle)]
        if coincident:
            target = points.mean(axis=0)
        else:
            target = _compact_target(points, flux_sampler, tolerance)
        root = min(cycle)
        labels[list(cycle)] = root
        positions[root] = target / spacing
        used.update(cycle)
        accepted += 1

    if accepted:
        coordinates, adjacency, medialness = _remap_graph(
            coordinates, adjacency, labels, positions, medialness
        )
    return (
        coordinates, adjacency, medialness, by_length,
        len(candidates), len(compact_candidates), accepted,
    )


def _project_neighbourhoods(coordinates, adjacency, spacing, ratio_limit):
    """Project eligible degree-five neighbourhoods onto chains through their centres."""
    physical = coordinates * spacing
    tolerance = _POSITION_TOLERANCE * float(np.min(spacing))
    adjacency = _simple_adjacency(adjacency)
    candidates = []
    for centre in range(len(coordinates)):
        neighbours = adjacency.indices[
            adjacency.indptr[centre] : adjacency.indptr[centre + 1]
        ]
        if len(neighbours) < 5:
            continue
        vertices = tuple(sorted((centre, *map(int, neighbours))))
        points = physical[list(vertices)]
        if np.max(np.linalg.norm(points - points[0], axis=1)) <= tolerance:
            continue
        tangent, ratio, _ = _pca(points, tolerance)
        if ratio >= ratio_limit:
            candidates.append((ratio, centre, vertices, tangent))
    candidates.sort(key=lambda item: (-item[0], item[1]))

    used = set()
    positions = physical.copy()
    edges = set(_edges(adjacency))
    accepted = 0
    for _, centre, vertices, tangent in candidates:
        if used.intersection(vertices):
            continue
        pivot = int(np.argmax(np.abs(tangent)))
        if tangent[pivot] < 0:
            tangent = -tangent
        neighbours = [vertex for vertex in vertices if vertex != centre]
        axial = {
            vertex: float(np.dot(physical[vertex] - physical[centre], tangent))
            for vertex in vertices
        }
        ordered = sorted(vertices, key=lambda vertex: (axial[vertex], vertex))
        projected = {
            vertex: physical[centre] + axial[vertex] * tangent
            for vertex in neighbours
        }
        vertex_set = set(vertices)
        internal = {
            (u, v) for u, v in edges if u in vertex_set and v in vertex_set
        }
        chain = {
            tuple(sorted((u, v))) for u, v in zip(ordered[:-1], ordered[1:])
        }
        moved = any(
            np.linalg.norm(projected[vertex] - physical[vertex]) > tolerance
            for vertex in neighbours
        )
        if not moved and internal == chain:
            continue
        for vertex, point in projected.items():
            positions[vertex] = point
        edges.difference_update(internal)
        edges.update(chain)
        used.update(vertices)
        accepted += 1
    if accepted:
        return (
            positions / spacing,
            _adjacency_from_edges(len(coordinates), edges),
            accepted,
        )
    return coordinates, adjacency, 0


def triangle_collapse_decimation(
    X,
    adjacency_matrix,
    binary_segmentation,
    voxel_spacing,
    elongation_ratio=25.0,
    medialness=None,
    flux_sampler=None,
    dev_edge_thresh=0.05,
    dev_peri_ratio=1.0,
):
    """Merge short edges and compact cycles, then project elongated neighbourhoods.

    ``elongation_ratio`` controls neighbourhood projection only. Cycle collapse
    uses a fixed PCA ratio below 25 and the configured perimeter limit.
    """
    coordinates = np.asarray(X, dtype=float)
    spacing = np.asarray(voxel_spacing, dtype=float)
    if not np.isfinite(elongation_ratio) or elongation_ratio <= 1.0:
        raise ValueError('dev_elongation_ratio must be finite and greater than 1.')
    if not np.isfinite(dev_edge_thresh) or dev_edge_thresh < 0:
        raise ValueError('dev_edge_thresh must be finite and nonnegative.')
    if not np.isfinite(dev_peri_ratio) or dev_peri_ratio <= 0:
        raise ValueError('dev_peri_ratio must be finite and positive.')
    if (
        spacing.shape != (3,)
        or np.any(spacing <= 0)
        or np.any(~np.isfinite(spacing))
    ):
        raise ValueError('voxel_spacing must contain three positive finite values.')
    segmentation = np.asarray(binary_segmentation)
    if segmentation.ndim != 3 or not np.any(segmentation):
        raise ValueError('triangle decimation requires a nonempty 3D segmentation.')
    adjacency = _simple_adjacency(adjacency_matrix)
    minimum_spacing = float(np.min(spacing))
    short_limit = dev_edge_thresh * minimum_spacing
    coordinates, adjacency, medialness, merged_groups = _merge_short_edges(
        coordinates, adjacency, spacing, short_limit, medialness
    )
    if flux_sampler is None:
        flux_sampler = DistanceFluxSampler(binary_segmentation, spacing)
    (
        coordinates, adjacency, medialness, counts, eligible,
        compact, collapsed,
    ) = _collapse_cycles(
        coordinates, adjacency, spacing, dev_peri_ratio,
        medialness, flux_sampler,
    )
    coordinates, adjacency, projections = _project_neighbourhoods(
        coordinates, adjacency, spacing, elongation_ratio
    )
    print(
        f'Decimation pass: edge groups={merged_groups}, cycles={counts}, '
        f'perimeter eligible={eligible}, compact={compact}, '
        f'collapsed={collapsed}, '
        f'neighbourhood projections={projections}, '
        f'nodes={len(coordinates)}, edges={adjacency.nnz // 2}'
    )
    changed = bool(merged_groups or collapsed or projections)
    return coordinates, adjacency, medialness, changed
