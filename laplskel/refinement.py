"""Foreground-guided graph refinement with fixed 26/6 digital topology.

Sequential simple-point deletion preserves the foreground's topology. The
26-neighbour graph of the surviving voxels still has filled triangles; removing
those relations over GF(2) leaves exactly the foreground's independent tunnels,
with their correspondence to the mask intact (not just the same cycle count).

Simple-point criterion: Bertrand's T26(foreground) = T6(background) = 1;
T6 counts components in N18 that touch a face neighbour of the deleted voxel.
See https://www.dgtal.org/doc/2.0/moduleDigitalTopology.html.
"""

import hashlib
import heapq
from itertools import product

import numpy as np
from scipy import ndimage, sparse
from scipy.spatial import cKDTree

from .graph import compute_sparse_adjacency_matrix
from .medial import compute_medialness
from .objects import UnionFind


_FOREGROUND = ndimage.generate_binary_structure(3, 3)
_BACKGROUND = ndimage.generate_binary_structure(3, 1)
_N18 = ndimage.generate_binary_structure(3, 2)
_FACES = _BACKGROUND.copy()
_FACES[1, 1, 1] = False
_OFFSETS = np.array([p for p in product((-1, 0, 1), repeat=3) if p != (0, 0, 0)])


def foreground_topology(mask):
    """Return component, tunnel and cavity counts for closed foreground cubes."""
    mask = np.asarray(mask, dtype=bool)
    coordinates = np.argwhere(mask)
    components = ndimage.label(mask, _FOREGROUND)[1]
    cavities = ndimage.label(~np.pad(mask, 1), _BACKGROUND)[1] - 1
    if not len(coordinates):
        return 0, 0, 0

    def count_cells(offsets):
        points = np.concatenate([coordinates + offset for offset in offsets])
        return len(np.unique(points, axis=0))

    vertices = count_cells(list(product((0, 1), repeat=3)))
    edges = faces = 0
    for axis in range(3):
        # An edge varies along one axis; a face is perpendicular to one axis.
        edges += count_cells([p for p in product((0, 1), repeat=3) if p[axis] == 0])
        offset = np.zeros(3, dtype=int)
        offset[axis] = 1
        faces += count_cells([np.zeros(3, dtype=int), offset])
    euler = vertices - edges + faces - len(coordinates)
    return components, components + cavities - euler, cavities


def _is_simple(neighbourhood):
    """Test one deletion against the current, rather than a cached, neighbourhood."""
    foreground = neighbourhood.copy()
    foreground[1, 1, 1] = False
    if ndimage.label(foreground, _FOREGROUND)[1] != 1:
        return False
    background = ~neighbourhood & _N18
    background[1, 1, 1] = False
    labels, _ = ndimage.label(background, _BACKGROUND)
    touching = np.unique(labels[_FACES])
    return np.count_nonzero(touching) == 1


def thin_foreground_sweep(
    mask,
    guidance,
    reference_edt,
    spacing=(1.0, 1.0, 1.0),
    validator=None,
    validation_batch=64,
    protected=None,
):
    """Delete one frozen boundary layer using topology-safe sequential checks.

    Low-EDT boundary voxels are considered first. Within one EDT band, voxels
    farther from the current geometric guide are preferred. Newly exposed voxels
    wait for the next call, which makes contraction able to influence every layer.
    """
    foreground = np.asarray(mask, dtype=bool)
    guide = np.asarray(guidance, dtype=float)
    edt = np.asarray(reference_edt, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    if foreground.ndim != 3 or edt.shape != foreground.shape:
        raise ValueError('mask and reference_edt must be matching 3D arrays.')
    protected = np.zeros_like(foreground) if protected is None else np.asarray(
        protected, dtype=bool
    )
    if protected.shape != foreground.shape or np.any(protected & ~foreground):
        raise ValueError('protected voxels must be contained in the working mask.')
    if guide.ndim != 2 or guide.shape[1] != 3 or not len(guide):
        raise ValueError('guidance must have shape (N, 3) with at least one point.')
    if spacing.shape != (3,) or np.any(spacing <= 0) or np.any(~np.isfinite(spacing)):
        raise ValueError('spacing must contain three positive finite values.')

    boundary = foreground & ~ndimage.binary_erosion(
        foreground, structure=_FOREGROUND, border_value=0
    )
    candidates = np.argwhere(boundary)
    if not len(candidates):
        return foreground.copy(), 0
    guide_distance = cKDTree(guide * spacing).query(candidates * spacing)[0]
    band_width = float(np.min(spacing))
    edt_bands = np.floor(edt[tuple(candidates.T)] / band_width + 1e-12)
    order = np.lexsort(
        (
            candidates[:, 2],
            candidates[:, 1],
            candidates[:, 0],
            -guide_distance,
            edt_bands,
        )
    )

    if validation_batch < 1:
        raise ValueError('validation_batch must be positive.')
    work = np.pad(foreground, 1)
    protected_work = np.pad(protected, 1)
    deleted = 0

    def try_delete(candidate):
        position = tuple(candidate)
        if not work[position] or protected_work[position]:
            return False
        patch = work[tuple(slice(value - 1, value + 2) for value in candidate)]
        # Retaining the last neighbour of a tip explicitly preserves terminal stubs.
        if np.count_nonzero(patch) <= 2 or not _is_simple(patch):
            return False
        work[position] = False
        return True

    checkpoint = work.copy()
    batch = []
    for candidate in candidates[order] + 1:
        if try_delete(candidate):
            batch.append(candidate)
        if len(batch) < validation_batch:
            continue
        if validator is not None and not validator(work[1:-1, 1:-1, 1:-1]):
            work = checkpoint
        else:
            deleted += len(batch)
        checkpoint = work.copy()
        batch = []
    if batch:
        if validator is not None and not validator(work[1:-1, 1:-1, 1:-1]):
            work = checkpoint
        else:
            deleted += len(batch)
    return work[1:-1, 1:-1, 1:-1], deleted


def _thin_foreground(mask, guidance, protected=None):
    """Peel simple voxels from the boundary inward, retaining curve endpoints."""
    work = np.pad(np.asarray(mask, dtype=bool), 1)
    protected_work = np.pad(
        np.zeros_like(mask, dtype=bool) if protected is None else protected, 1
    )
    if protected_work.shape != work.shape or np.any(protected_work & ~work):
        raise ValueError('protected voxels must be contained in the working mask.')
    coordinates = np.argwhere(work)
    distance = ndimage.distance_transform_edt(work)
    guide_distance = cKDTree(guidance).query(coordinates - 1)[0]
    priorities = {
        tuple(p): (float(distance[tuple(p)]), -float(np.round(d, 12)))
        for p, d in zip(coordinates, guide_distance)
    }
    queue = [(priorities[tuple(p)], tuple(p)) for p in coordinates]
    heapq.heapify(queue)
    queued = set(priorities)
    while queue:
        _, p = heapq.heappop(queue)
        queued.remove(p)
        if not work[p] or protected_work[p]:
            continue
        patch = work[tuple(slice(v - 1, v + 2) for v in p)]
        # A curve tip may be simple but deleting it shortens a branch.
        if np.count_nonzero(patch) <= 2 or not _is_simple(patch):
            continue
        work[p] = False
        for neighbour in np.asarray(p) + _OFFSETS:
            q = tuple(neighbour)
            if work[q] and q not in queued:
                heapq.heappush(queue, (priorities[q], q))
                queued.add(q)
    return np.argwhere(work) - 1


def _insert_binary_row(row, basis):
    """Insert an independent GF(2) row, represented by a Python integer."""
    while row:
        pivot = row.bit_length() - 1
        if pivot not in basis:
            basis[pivot] = row
            return True
        row ^= basis[pivot]
    return False


def _tunnel_graph(coordinates):
    """Keep a forest plus independent cycles modulo filled voxel triangles.

    Pairwise touching axis-aligned voxel cubes have a common intersection.
    Consequently their nerve is the clique complex of the 26-neighbour graph.
    Its triangle boundaries generate all relations needed for first homology.
    Projection onto non-forest edges gives coordinates for the graph cycle space.
    Choosing a basis modulo these boundaries preserves each tunnel class.
    """
    adjacency = compute_sparse_adjacency_matrix(cKDTree(coordinates), 26)
    rows, cols = sparse.triu(adjacency, k=1).nonzero()
    edges = sorted(
        zip(rows.tolist(), cols.tolist()),
        key=lambda e: (np.sum((coordinates[e[0]] - coordinates[e[1]]) ** 2), e),
    )
    forest = UnionFind(len(coordinates))
    kept = []
    chords = {}
    for edge in edges:
        if forest.union(*edge)[0]:
            kept.append(edge)
        else:
            chords[edge] = len(chords)
    neighbours = [
        set(adjacency.indices[adjacency.indptr[i] : adjacency.indptr[i + 1]])
        for i in range(len(coordinates))
    ]
    boundaries = {}
    for u, v in edges:
        for w in sorted(neighbours[u] & neighbours[v]):
            if w <= v:
                continue
            row = 0
            for edge in ((u, v), (u, w), (v, w)):
                if edge in chords:
                    row ^= 1 << chords[edge]
            _insert_binary_row(row, boundaries)
    tunnels = len(chords) - len(boundaries)
    for edge, index in chords.items():
        if _insert_binary_row(1 << index, boundaries):
            kept.append(edge)
    return np.asarray(kept, dtype=int).reshape(-1, 2), tunnels


def branch_complex_signature(coordinates, edges):
    """Describe terminal branches, junction attachments, and independent cycles.

    Degree-two sampling is removed from the description. This is deliberately
    stricter than component and Betti-number checks: losing a terminal stub changes
    both the endpoint count and the branch attachment signature.
    """
    n_vertices = len(coordinates)
    neighbours = [set() for _ in range(n_vertices)]
    for u, v in np.asarray(edges, dtype=int).reshape(-1, 2):
        neighbours[int(u)].add(int(v))
        neighbours[int(v)].add(int(u))
    degrees = np.array([len(items) for items in neighbours], dtype=int)
    critical = set(np.flatnonzero(degrees != 2).tolist())
    visited = set()
    attachments = []
    core_edges = []
    closed_cycles = 0

    def edge_key(u, v):
        return (min(u, v), max(u, v))

    for start in sorted(critical):
        for neighbour in sorted(neighbours[start]):
            if edge_key(start, neighbour) in visited:
                continue
            visited.add(edge_key(start, neighbour))
            previous, current = start, neighbour
            while current not in critical:
                onward = sorted(neighbours[current] - {previous})
                if not onward:
                    break
                following = onward[0]
                visited.add(edge_key(current, following))
                previous, current = current, following
            attachments.append(
                tuple(sorted((int(degrees[start]), int(degrees[current]))))
            )
            core_edges.append((int(start), int(current)))

    for start in range(n_vertices):
        for neighbour in neighbours[start]:
            if edge_key(start, neighbour) in visited:
                continue
            closed_cycles += 1
            previous, current = start, neighbour
            visited.add(edge_key(start, neighbour))
            while current != start:
                onward = [item for item in neighbours[current] if item != previous]
                if not onward:
                    break
                following = onward[0]
                visited.add(edge_key(current, following))
                previous, current = current, following

    if n_vertices:
        rows = np.concatenate(
            ([u for u, v in edges], [v for u, v in edges])
        ) if len(edges) else np.array([], dtype=int)
        cols = np.concatenate(
            ([v for u, v in edges], [u for u, v in edges])
        ) if len(edges) else np.array([], dtype=int)
        adjacency = sparse.csr_matrix(
            (np.ones(len(rows), dtype=bool), (rows, cols)),
            shape=(n_vertices, n_vertices),
        )
        components = sparse.csgraph.connected_components(adjacency, directed=False)[0]
    else:
        components = 0
    tunnels = len(edges) - n_vertices + components
    colors = {vertex: str(int(degrees[vertex])) for vertex in critical}
    multiplicity = {
        (u, v): sum(
            (start == u and end == v) or (start == v and end == u)
            for start, end in core_edges
        )
        for u in critical
        for v in critical
    }
    for _ in range(len(critical)):
        colors = {
            vertex: hashlib.sha256(
                repr(
                    (
                        colors[vertex],
                        tuple(
                            sorted(
                                (colors[neighbour], multiplicity[vertex, neighbour])
                                for neighbour in critical
                                if multiplicity[vertex, neighbour]
                            )
                        ),
                    )
                ).encode()
            ).hexdigest()
            for vertex in critical
        }
    return (
        int(components),
        int(tunnels),
        tuple(sorted(degrees[degrees != 2].tolist())),
        tuple(sorted(attachments)),
        int(closed_cycles),
        tuple(sorted(colors.values())),
    )


def _filled_box(points, mask):
    """Certify a shortcut's entire swept region lies in solid foreground.

    Testing the bounding box is conservative: it may reject a safe shortcut but
    never accepts one that could cut across a tunnel or leave the foreground.
    """
    lower = np.floor(np.min(points, axis=0) + 0.5).astype(int)
    upper = np.ceil(np.max(points, axis=0) - 0.5).astype(int)
    upper = np.maximum(upper, lower)
    if np.any(lower < 0) or np.any(upper >= mask.shape):
        return False
    return bool(np.all(mask[tuple(slice(a, b + 1) for a, b in zip(lower, upper))]))


def _path_error(points, start, end):
    """Measure deviation of every original path sample from a proposed segment."""
    direction = end - start
    squared_length = np.dot(direction, direction)
    if squared_length == 0:
        return float(np.max(np.linalg.norm(points - start, axis=1)))
    fraction = np.clip((points - start) @ direction / squared_length, 0.0, 1.0)
    return float(
        np.max(np.linalg.norm(points - start - fraction[:, None] * direction, axis=1))
    )


def _fit_geometry(coordinates, edges, guidance, mask, medial_boost):
    """Fit each voxel's interior point between fixed edge contact points.

    Each edge gets a fixed midpoint shared by its two source voxel cubes. A
    fitted point stays inside its own cube, so the two half-edges and their
    deformation stay inside the foreground even at diagonal-only contacts.
    Junctions and endpoints stay fixed. For other points the weighted quadratic
    fit has a closed-form minimum, clipped to the voxel's convex bounds.
    """
    if not len(edges):
        return coordinates, edges
    degree = np.bincount(edges.ravel(), minlength=len(coordinates))
    contacts = coordinates[edges].mean(axis=1)
    sums = np.zeros_like(coordinates)
    np.add.at(sums, edges[:, 0], contacts)
    np.add.at(sums, edges[:, 1], contacts)
    nearest = cKDTree(guidance).query(coordinates)[1]
    targets = np.clip(guidance[nearest], coordinates - 0.49, coordinates + 0.49)
    edt = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1, 1:-1]
    reference_radius = ndimage.map_coordinates(edt, coordinates.T, order=1)
    target_radius = ndimage.map_coordinates(edt, targets.T, order=1)
    # An unconverged dense flow can drift towards a wall. Do not give that
    # displacement extra retention at a better-centred reference voxel.
    targets[target_radius < reference_radius] = coordinates[
        target_radius < reference_radius
    ]
    weights = medial_boost ** (2 * compute_medialness(edt, coordinates))
    fitted = (weights[:, None] * targets + sums) / (weights + degree)[:, None]
    fitted = np.clip(fitted, coordinates - 0.49, coordinates + 0.49)
    fitted[degree != 2] = coordinates[degree != 2]
    portals = np.arange(len(edges)) + len(coordinates)
    split_edges = np.vstack(
        (
            np.column_stack((edges[:, 0], portals)),
            np.column_stack((portals, edges[:, 1])),
        )
    )
    return np.vstack((fitted, contacts)), split_edges


def _simplify_paths(coordinates, edges, mask, tolerance):
    """Remove degree-two vertices; mask=None skips foreground shortcut checks."""
    neighbours = [set() for _ in coordinates]
    paths = {}
    for u, v in edges:
        neighbours[u].add(v)
        neighbours[v].add(u)
        key = tuple(sorted((u, v)))
        paths[key] = coordinates[list(key)].copy()
    active = np.ones(len(coordinates), dtype=bool)
    queue = list(range(len(coordinates))) if tolerance > 0 else []
    heapq.heapify(queue)
    queued = set(queue)
    while queue:
        vertex = heapq.heappop(queue)
        queued.remove(vertex)
        if not active[vertex] or len(neighbours[vertex]) != 2:
            continue
        u, v = sorted(neighbours[vertex])
        # Adding an already existing edge would destroy a real cycle.
        if v in neighbours[u]:
            continue
        first, second = tuple(sorted((u, vertex))), tuple(sorted((v, vertex)))
        incoming = paths[first] if u < vertex else paths[first][::-1]
        outgoing = paths[second] if vertex < v else paths[second][::-1]
        samples = np.vstack((incoming, outgoing[1:]))
        if _path_error(
            samples, coordinates[u], coordinates[v]
        ) > tolerance or (mask is not None and not _filled_box(samples, mask)):
            continue
        neighbours[u].remove(vertex)
        neighbours[v].remove(vertex)
        neighbours[u].add(v)
        neighbours[v].add(u)
        neighbours[vertex].clear()
        active[vertex] = False
        del paths[first], paths[second]
        paths[(u, v)] = samples
        for neighbour in (u, v):
            if neighbour not in queued:
                heapq.heappush(queue, neighbour)
                queued.add(neighbour)
    remap = np.cumsum(active) - 1
    edges = np.asarray(
        [(remap[u], remap[v]) for u, v in sorted(paths)], dtype=int
    ).reshape(-1, 2)
    paths = {(int(remap[u]), int(remap[v])): path for (u, v), path in paths.items()}
    return coordinates[active], edges, paths


def refine_graph(
    mask,
    contracted_coordinates,
    merge_tolerance=0.25,
    w_H_medial=1.0,
    return_paths=False,
):
    """Extract a graph with every foreground tunnel and simplify its geometry.

    Parameters
    ----------
    mask : (D, H, W) array
        Original foreground, interpreted with fixed 26-connectivity.
    contracted_coordinates : (N, 3) array
        Laplacian contraction result in the mask's voxel coordinate system.
        Guides which medial voxels survive; never determines tunnel removal.
    merge_tolerance : float, optional
        Maximum polyline deviation in voxel units during degree-two merging.
        Zero retains the reference sampling. No value permits tunnel removal.
    w_H_medial : float, optional
        Retention boost for the bounded final geometric fit. Default is 1.
    return_paths : bool, optional
        Also return reference voxels and the retained edge polylines for export.

    Returns
    -------
    coordinates : (M, 3) array
        Refined graph coordinates in voxel units.
    adjacency : sparse.csr_matrix
        Symmetric adjacency with no redundant independent cycles.
    reference_voxels : (K, 3) array, optional
        Topology-preserving digital skeleton, returned when return_paths is True.
    edge_paths : dict, optional
        Original fitted polylines keyed by sorted node-index pairs, returned
        when return_paths is True. Avoids topology changes from re-voxelization.
    """
    if not np.isfinite(merge_tolerance) or merge_tolerance < 0:
        raise ValueError('merge_tolerance must be finite and >= 0.')
    if not np.isfinite(w_H_medial) or w_H_medial < 1:
        raise ValueError('w_H_medial must be finite and >= 1.')
    mask = np.asarray(mask, dtype=bool)
    guidance = np.asarray(contracted_coordinates, dtype=float)
    if mask.ndim != 3 or not np.any(mask):
        raise ValueError('Refinement requires a nonempty 3D foreground mask.')
    if (
        guidance.ndim != 2
        or guidance.shape[1] != 3
        or not len(guidance)
        or not np.all(np.isfinite(guidance))
    ):
        raise ValueError(
            'Contracted coordinates must be a finite nonempty (N, 3) array.'
        )
    # A curve graph cannot represent an enclosed three-dimensional cavity.
    expected_components, expected_tunnels, cavities = foreground_topology(mask)
    if cavities:
        raise ValueError(
            'Foreground contains enclosed cavities; a curve graph cannot preserve their topology.'
        )
    coordinates = _thin_foreground(mask, guidance).astype(float)
    reference_voxels = coordinates.astype(int)
    edges, tunnels = _tunnel_graph(coordinates)
    coordinates, edges = _fit_geometry(coordinates, edges, guidance, mask, w_H_medial)
    coordinates, edges, paths = _simplify_paths(
        coordinates, edges, mask, merge_tolerance
    )
    if len(edges):
        rows = np.concatenate((edges[:, 0], edges[:, 1]))
        cols = np.concatenate((edges[:, 1], edges[:, 0]))
    else:
        rows = cols = np.array([], dtype=int)
    adjacency = sparse.csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=(len(coordinates), len(coordinates)),
    )
    components = sparse.csgraph.connected_components(adjacency, directed=False)[0]
    if (
        components != expected_components
        or tunnels != expected_tunnels
        or len(edges) - len(coordinates) + components != expected_tunnels
    ):
        raise RuntimeError('Refinement failed its foreground topology checks.')
    print(
        f'Refined graph: {len(coordinates)} nodes, {len(edges)} edges, '
        f'{components} components, {tunnels} independent tunnels preserved.'
    )
    if return_paths:
        return coordinates, adjacency, reference_voxels, paths
    return coordinates, adjacency
