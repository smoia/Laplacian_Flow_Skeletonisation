"""Medial-axis scoring and branch-wise EDT ridge guidance."""

import heapq

import numpy as np
from scipy import ndimage

from .utils import _foreground_voxel


def compute_medialness(edt_volume, node_coords, threshold=0.8):
    """
    Score how close each graph vertex sits to an inscribed-sphere centre.

    A vertex is maximally medial when its Euclidean Distance Transform value equals
    the largest value found in its 3x3x3 voxel neighborhood, which marks the centre
    of a locally maximal inscribed sphere. The raw neighborhood ratio is
    contrast-stretched so that only vertices at or around such centres score above
    zero, leaving boundary vertices unscored.

    The 3x3x3 footprint is equivalent to a one-hop maximum over a 26-connected voxel
    graph, but is evaluated on the volume so the score stays independent of the
    connectivity chosen for the initial graph.

    Parameters
    ----------
    edt_volume : numpy.ndarray
        The 3D Euclidean Distance Transform of the binary segmentation, in voxel units.
    node_coords : numpy.ndarray
        An (N, 3) array holding the 3D coordinates of the graph vertices. Coordinates
        are rounded to the nearest voxel and clipped to the volume before sampling.
    threshold : float, optional
        Lower cut of the contrast stretch. Vertices whose neighborhood ratio falls at
        or below this value score 0, while a ratio of 1 scores 1. Must lie in [0, 1).
        Default is 0.8.

    Returns
    -------
    medialness : numpy.ndarray
        An (N,) float array of medialness scores bounded to [0, 1].

    Raises
    ------
    ValueError
        If `edt_volume` is not three-dimensional, if `node_coords` does not have
        shape (N, 3), or if `threshold` lies outside [0, 1).
    """
    edt_volume = np.asarray(edt_volume, dtype=float)
    if edt_volume.ndim != 3:
        raise ValueError('edt_volume must be a 3D array.')

    coordinates = np.asarray(node_coords, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError('node_coords must have shape (N, 3).')

    if not 0.0 <= float(threshold) < 1.0:
        raise ValueError('threshold must lie in [0, 1).')

    voxel_positions = np.clip(
        np.rint(coordinates).astype(np.intp), 0, np.asarray(edt_volume.shape) - 1
    )
    lookup = tuple(voxel_positions.T)

    # Largest inscribed-sphere radius available within each voxel's neighborhood
    neighborhood_max = ndimage.maximum_filter(
        edt_volume, size=3, mode='constant', cval=0.0
    )

    ratios = edt_volume[lookup] / np.maximum(neighborhood_max[lookup], 1e-12)

    return np.clip((ratios - threshold) / (1.0 - threshold), 0.0, 1.0)


def estimate_graph_tangents(
    coordinates, adjacency, spacing=(1.0, 1.0, 1.0), window=None
):
    """Estimate branch directions by physical arc-length neighbourhood PCA.

    The neighbourhood follows graph edges, so nearby but unconnected branches do
    not contaminate one another. Directions are re-estimated whenever this helper
    is called; their sign is intentionally arbitrary.
    """
    points = np.asarray(coordinates, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('coordinates must have shape (N, 3).')
    if (
        spacing.shape != (3,)
        or np.any(~np.isfinite(spacing))
        or np.any(spacing <= 0)
    ):
        raise ValueError('spacing must contain three positive finite values.')
    if window is None:
        window = 2.0 * float(np.min(spacing))
    if not np.isfinite(window) or window <= 0:
        raise ValueError('window must be positive and finite.')

    graph = adjacency.tocsr()
    tangents = np.zeros_like(points)
    physical = points * spacing
    for start in range(len(points)):
        distances = {start: 0.0}
        queue = [(0.0, start)]
        members = []
        while queue:
            distance, vertex = heapq.heappop(queue)
            if distance != distances[vertex] or distance > window:
                continue
            members.append(vertex)
            begin, end = graph.indptr[vertex : vertex + 2]
            for neighbour in graph.indices[begin:end]:
                next_distance = distance + np.linalg.norm(
                    physical[vertex] - physical[neighbour]
                )
                if next_distance <= window and next_distance < distances.get(
                    int(neighbour), np.inf
                ):
                    distances[int(neighbour)] = float(next_distance)
                    heapq.heappush(queue, (float(next_distance), int(neighbour)))
        local = physical[members]
        if len(local) >= 2:
            covariance = (local - local.mean(axis=0)).T @ (local - local.mean(axis=0))
            tangents[start] = np.linalg.eigh(covariance)[1][:, -1]
        else:
            tangents[start, 0] = 1.0
    return tangents


def transverse_edt_targets(
    edt_volume,
    foreground,
    coordinates,
    tangents,
    spacing=(1.0, 1.0, 1.0),
    enforce_containment=True,
):
    """Find a nearby high-EDT target in each point's transverse plane.

    EDT is sampled from the original foreground. The search radius is the larger
    of two minimum voxel spacings and the EDT radius at the current point.
    Equal-valued local ridge voxels form a plateau. Its connected transverse part
    within the current foreground component is centred before the displacement
    is projected off the branch tangent. With enforce_containment=False,
    points may be outside the mask, targets may cross component boundaries,
    and projected targets are not forced back into foreground.
    """
    edt = np.asarray(edt_volume, dtype=float)
    mask = np.asarray(foreground, dtype=bool)
    points = np.asarray(coordinates, dtype=float)
    directions = np.asarray(tangents, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    if edt.shape != mask.shape or edt.ndim != 3:
        raise ValueError('edt_volume and foreground must be matching 3D arrays.')
    if points.shape != directions.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('coordinates and tangents must have matching shape (N, 3).')

    sampled_radius = ndimage.map_coordinates(
        edt, points.T, order=1, mode='nearest'
    )
    targets = points.copy()
    h = float(np.min(spacing))
    bounds = np.asarray(mask.shape)
    components = (
        ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))[0]
        if enforce_containment else None
    )
    for index, (point, tangent, radius) in enumerate(
        zip(points, directions, sampled_radius)
    ):
        component = (
            components[_foreground_voxel(point, mask.shape, mask)]
            if enforce_containment else None
        )
        search_radius = max(2.0 * h, float(radius))
        voxel_radius = np.ceil(search_radius / spacing).astype(int)
        centre = np.rint(point).astype(int)
        lower = np.clip(centre - voxel_radius, 0, bounds)
        upper = np.clip(centre + voxel_radius + 1, 0, bounds)
        region = tuple(slice(a, b) for a, b in zip(lower, upper))
        eligible_mask = mask[region]
        if enforce_containment:
            eligible_mask = eligible_mask & (components[region] == component)
        candidates = np.argwhere(eligible_mask) + lower
        if not len(candidates):
            continue
        offsets = (candidates - point) * spacing
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 0:
            axial = np.abs(offsets @ (tangent / tangent_norm))
        else:
            axial = np.zeros(len(candidates))
        radial = np.linalg.norm(offsets, axis=1)
        eligible = (axial <= 0.5 * h + 1e-12) & (
            radial <= search_radius + 1e-12
        )
        if not np.any(eligible):
            continue
        candidates = candidates[eligible]
        radial = radial[eligible]
        values = edt[tuple(candidates.T)]
        best_value = np.max(values)
        best = np.flatnonzero(np.isclose(values, best_value, rtol=0.0, atol=1e-12))
        tied = candidates[best]
        seed = int(np.argmin(radial[best]))
        connected = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            neighbours = np.flatnonzero(
                np.max(np.abs(tied - tied[current]), axis=1) <= 1
            )
            for neighbour in neighbours:
                neighbour = int(neighbour)
                if neighbour not in connected:
                    connected.add(neighbour)
                    frontier.append(neighbour)
        target = np.mean(tied[sorted(connected)], axis=0)
        if tangent_norm > 0:
            unit = tangent / tangent_norm
            displacement = (target - point) * spacing
            target = point + (displacement - unit * (displacement @ unit)) / spacing
        if enforce_containment:
            try:
                _foreground_voxel(target, mask.shape, mask)
            except ValueError:
                target = tied[seed].astype(float)
        targets[index] = target
    return targets
