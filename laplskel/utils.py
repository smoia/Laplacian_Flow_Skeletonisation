"""Geometry checks, output and skeleton rasterization."""

import json
import xml.etree.ElementTree as ET
from itertools import product

import numpy as np
from scipy import ndimage, sparse
from scipy.spatial import cKDTree


GRAPHML_NAMESPACE = 'http://graphml.graphdrawing.org/xmlns'
GRAPHML_XSI_NAMESPACE = 'http://www.w3.org/2001/XMLSchema-instance'


def coords_to_dense_3d(X, volume_shape, adjacency_matrix=None, binary_segmentation=None):
    """
    Create and fill in a volume using coordinates of points.

    Parameters
    ----------
    X : ndarray of shape (N, 3)
        The 3D coordinates of the areas with content.
    volume_shape : tuple of int (D, H, W)
        The structural grid dimensions of the target 3D matrix.
    adjacency_matrix : sparse matrix, optional
        Graph edges to rasterize along with nodes.
    binary_segmentation : array, optional
        Foreground used to resolve voxel-boundary ties and check containment.

    Returns
    -------
    dense_volume : ndarray of shape (D, H, W)
        A binary 3D array where 1 represents the skeleton path.
    """
    # 1. Initialize empty dense matrix
    dense_volume = np.zeros(volume_shape, dtype=bool)

    for point in X:
        dense_volume[_foreground_voxel(point, volume_shape, binary_segmentation)] = True
    if adjacency_matrix is not None:
        rows, cols = sparse.triu(adjacency_matrix, k=1).nonzero()
        for u, v in zip(rows, cols):
            for voxel in _edge_voxels(X[u], X[v], volume_shape, binary_segmentation):
                dense_volume[voxel] = True

    return dense_volume


def _clip_voxel(voxel, volume_shape):
    """Clip one integer voxel coordinate to the target volume bounds."""
    clipped = np.clip(
        np.asarray(voxel, dtype=int),
        0,
        np.asarray(volume_shape, dtype=int) - 1,
    )
    return tuple(int(value) for value in clipped)


def _foreground_voxel(point, volume_shape, mask=None):
    """Resolve a boundary contact to an incident foreground voxel."""
    voxel = _clip_voxel(np.rint(point), volume_shape)
    if mask is None:
        return voxel
    if mask[voxel] and np.max(np.abs(np.asarray(voxel) - point)) <= 0.5 + 1e-9:
        return voxel
    # Diagonal cells can meet exactly at a corner where round-to-even picks a
    # background cell. Only incident cells are allowed; never project an escape.
    for offset in product((-1, 0, 1), repeat=3):
        candidate = np.asarray(voxel) + offset
        if (
            np.all(candidate >= 0)
            and np.all(candidate < volume_shape)
            and np.max(np.abs(candidate - point)) <= 0.5 + 1e-9
            and mask[tuple(candidate)]
        ):
            return tuple(int(value) for value in candidate)
    raise ValueError('Graph geometry leaves the foreground mask.')


def _edge_voxels(start, end, volume_shape, binary_segmentation=None):
    """Traverse every voxel interval of a continuous graph edge, including ends."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    crossings = [0.0, 1.0]
    for axis in range(3):
        if delta[axis] == 0:
            continue
        low, high = sorted((start[axis], end[axis]))
        planes = np.arange(np.ceil(low - 0.5), np.floor(high - 0.5) + 1) + 0.5
        crossings.extend(((planes - start[axis]) / delta[axis]).tolist())
    crossings = np.unique(np.clip(crossings, 0.0, 1.0))
    fractions = [0.0] + ((crossings[:-1] + crossings[1:]) / 2).tolist() + [1.0]
    voxels = []
    for fraction in fractions:
        point = start + fraction * delta
        voxel = _foreground_voxel(point, volume_shape, binary_segmentation)
        if not voxels or voxels[-1] != voxel:
            voxels.append(voxel)
    return voxels


def _box_in_foreground(points, mask):
    """Conservatively certify a box covering a moving graph edge."""
    lower = np.ceil(np.min(points, axis=0) - 0.5 - 1e-10).astype(int)
    upper = np.floor(np.max(points, axis=0) + 0.5 + 1e-10).astype(int)
    if np.any(lower < 0) or np.any(upper >= mask.shape):
        return False
    return bool(np.all(mask[tuple(slice(a, b + 1) for a, b in zip(lower, upper))]))


def _triangle_enters_voxel(triangle, voxel):
    """Clip a swept triangle against the open interior of one voxel cube."""
    polygon = [point for point in triangle]
    epsilon = 1e-10
    for axis in range(3):
        for sign, boundary in (
            (1.0, voxel[axis] - 0.5 + epsilon),
            (-1.0, voxel[axis] + 0.5 - epsilon),
        ):
            if not polygon:
                return False
            clipped = []
            for start, end in zip(polygon, polygon[1:] + polygon[:1]):
                first = sign * (start[axis] - boundary)
                second = sign * (end[axis] - boundary)
                if first > 0:
                    clipped.append(start)
                if (first > 0) != (second > 0):
                    fraction = first / (first - second)
                    clipped.append(start + fraction * (end - start))
            polygon = clipped
    return bool(polygon)


def _sweep_contained(old_start, old_end, new_start, new_end, mask):
    """Certify the swept edge by recursively covering its triangles with voxels."""
    triangles = (
        np.asarray([old_start, old_end, new_start]),
        np.asarray([new_start, old_end, new_end]),
    )

    def contained(triangle, remaining):
        if _box_in_foreground(triangle, mask):
            return True
        lower = np.ceil(np.min(triangle, axis=0) - 0.5 - 1e-10).astype(int)
        upper = np.floor(np.max(triangle, axis=0) + 0.5 + 1e-10).astype(int)
        volume = int(np.prod(upper - lower + 1))
        if remaining == 0 or volume <= 8:
            probes = list(triangle)
            probes += [
                0.5 * (triangle[a] + triangle[b])
                for a, b in ((0, 1), (1, 2), (2, 0))
            ]
            probes.append(np.mean(triangle, axis=0))
            try:
                for point in probes:
                    _foreground_voxel(point, mask.shape, mask)
            except ValueError:
                return False
            for voxel in product(*(range(a, b + 1) for a, b in zip(lower, upper))):
                if (
                    (
                        any(
                            value < 0 or value >= bound
                            for value, bound in zip(voxel, mask.shape)
                        )
                        or not mask[voxel]
                    )
                    and _triangle_enters_voxel(triangle, voxel)
                ):
                    return False
            return True
        pairs = ((0, 1), (1, 2), (2, 0))
        a, b = max(
            pairs,
            key=lambda pair: np.linalg.norm(
                triangle[pair[0]] - triangle[pair[1]]
            ),
        )
        c = 3 - a - b
        middle = 0.5 * (triangle[a] + triangle[b])
        return (
            contained(np.asarray([triangle[a], middle, triangle[c]]), remaining - 1)
            and contained(
                np.asarray([middle, triangle[b], triangle[c]]), remaining - 1
            )
        )

    return all(contained(triangle, 10) for triangle in triangles)


def _nonincident_intersections(coordinates, edges, spacing):
    """Detect unrepresented crossings and overlap beyond shared endpoints."""
    physical = np.asarray(coordinates, dtype=float) * spacing
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    tolerance = 1e-8 * float(np.min(spacing))
    if not len(edges):
        return False
    midpoints = np.mean(physical[edges], axis=1)
    radii = 0.5 * np.linalg.norm(physical[edges[:, 0]] - physical[edges[:, 1]], axis=1)
    tree = cKDTree(midpoints)
    for index, (u, v) in enumerate(edges):
        p, q = physical[[u, v]]
        candidates = tree.query_ball_point(
            midpoints[index], radii[index] + np.max(radii) + tolerance
        )
        for other_index in candidates:
            if other_index <= index:
                continue
            a, b = edges[other_index]
            shared = {int(u), int(v)} & {int(a), int(b)}
            if shared:
                if len(shared) == 1:
                    junction = shared.pop()
                    first = physical[v if u == junction else u] - physical[junction]
                    second = physical[b if a == junction else a] - physical[junction]
                    if (
                        first @ second > 0
                        and np.linalg.norm(np.cross(first, second))
                        <= tolerance * max(
                            np.linalg.norm(first), np.linalg.norm(second)
                        )
                    ):
                        return True
                continue
            r, s = physical[[a, b]]
            if np.any(np.maximum(np.minimum(p, q), np.minimum(r, s)) >
                      np.minimum(np.maximum(p, q), np.maximum(r, s)) + tolerance):
                continue
            first, second = q - p, s - r
            difference = p - r
            aa, bb, cc = first @ first, first @ second, second @ second
            dd, ee = first @ difference, second @ difference
            determinant = aa * cc - bb * bb
            if determinant > tolerance ** 4:
                t = np.clip((bb * ee - cc * dd) / determinant, 0.0, 1.0)
            else:
                t = 0.0
            w = np.clip((bb * t + ee) / cc, 0.0, 1.0) if cc else 0.0
            t = np.clip((bb * w - dd) / aa, 0.0, 1.0) if aa else 0.0
            def endpoint_distance(point, start, end):
                direction = end - start
                length_squared = direction @ direction
                fraction = (
                    np.clip(((point - start) @ direction) / length_squared, 0.0, 1.0)
                    if length_squared else 0.0
                )
                return np.linalg.norm(point - start - fraction * direction)

            if min(
                np.linalg.norm(p + t * first - r - w * second),
                endpoint_distance(p, r, s),
                endpoint_distance(q, r, s),
                endpoint_distance(r, p, q),
                endpoint_distance(s, p, q),
            ) <= tolerance:
                return True
    return False


def _segment_contacts(start, end, first, second, tolerance):
    """Return contact points between segments, including collinear overlap."""
    direction = end - start
    other = second - first
    length = np.linalg.norm(direction)
    other_length = np.linalg.norm(other)
    if length <= tolerance:
        if other_length <= tolerance:
            return [start] if np.linalg.norm(start - first) <= tolerance else []
        fraction = np.clip((start - first) @ other / (other @ other), 0, 1)
        distance = np.linalg.norm(start - first - fraction * other)
        return [start] if distance <= tolerance else []
    if other_length <= tolerance:
        return _segment_contacts(first, second, start, end, tolerance)
    if (np.linalg.norm(np.cross(direction, other))
            <= tolerance * max(length, other_length)):
        if np.linalg.norm(np.cross(first - start, direction)) > tolerance * length:
            return []
        fractions = np.array([
            (first - start) @ direction, (second - start) @ direction
        ]) / (length ** 2)
        low, high = max(0., float(fractions.min())), min(1., float(fractions.max()))
        if low > high + tolerance / length:
            return []
        return [start + low * direction, start + high * direction]
    # The unconstrained closest points plus the four endpoint projections cover
    # every possible minimum on the closed parameter square.
    matrix = np.column_stack((direction, -other))
    fractions = np.linalg.lstsq(matrix, first - start, rcond=None)[0]
    candidates = [fractions] if np.all((fractions >= 0) & (fractions <= 1)) else []
    candidates += [
        (t, np.clip((start + t * direction - first) @ other / (other @ other), 0, 1))
        for t in (0., 1.)
    ]
    candidates += [
        (np.clip((first + t * other - start) @ direction / (direction @ direction),
                 0, 1), t)
        for t in (0., 1.)
    ]
    for t, u in candidates:
        point = start + t * direction
        if np.linalg.norm(point - first - u * other) <= tolerance:
            return [point]
    return []


def _segment_triangle_contacts(start, end, triangle, tolerance):
    """Intersect a segment with a closed triangle, including coplanar contact."""
    a, b, c = triangle
    first, second = b - a, c - a
    normal = np.cross(first, second)
    norm = np.linalg.norm(normal)
    scale = max(np.linalg.norm(first), np.linalg.norm(second), np.linalg.norm(c - b))
    if norm <= tolerance * scale:
        # A degenerate swept triangle is the segment between its extreme points.
        pairs = ((a, b), (a, c), (b, c))
        low, high = max(pairs, key=lambda pair: np.linalg.norm(pair[1] - pair[0]))
        return _segment_contacts(start, end, low, high, tolerance)
    normal /= norm
    distances = np.array([(start - a) @ normal, (end - a) @ normal])
    if np.all(distances > tolerance) or np.all(distances < -tolerance):
        return []
    # Barycentric coordinates give three affine half-space inequalities.
    basis = np.column_stack((first, second))
    inverse = np.linalg.pinv(basis)
    barycentric = np.asarray([inverse @ (point - a) for point in (start, end)])
    barycentric = np.column_stack((1 - barycentric.sum(axis=1), barycentric))
    slack = tolerance * np.linalg.norm(inverse, axis=1).max()
    if np.max(np.abs(distances)) > tolerance:
        fraction = distances[0] / (distances[0] - distances[1])
        if not 0 <= fraction <= 1:
            return []
        bary = barycentric[0] + fraction * (barycentric[1] - barycentric[0])
        return [start + fraction * (end - start)] if np.all(bary >= -slack) else []
    low, high = 0., 1.
    for initial, change in zip(barycentric[0], barycentric[1] - barycentric[0]):
        if abs(change) <= np.finfo(float).eps:
            if initial < -slack:
                return []
        elif change > 0:
            low = max(low, -initial / change)
        else:
            high = min(high, -initial / change)
    if low > high + tolerance / max(np.linalg.norm(end - start), tolerance):
        return []
    return [start + np.clip(t, 0, 1) * (end - start) for t in (low, high)]


def _valid_node_move(coordinates, edges, vertex, target, mask, spacing):
    """Certify a single-node deformation, including crossings during its sweep."""
    old = coordinates[vertex]
    incident = np.flatnonzero(np.any(edges == vertex, axis=1))
    neighbours = [int(v if u == vertex else u) for u, v in edges[incident]]
    physical = coordinates * spacing
    old_physical, new_physical = old * spacing, target * spacing
    tolerance = 1e-8 * float(np.min(spacing))
    try:
        _foreground_voxel(target, mask.shape, mask)
        for neighbour in neighbours:
            _edge_voxels(target, coordinates[neighbour], mask.shape, mask)
            if not _sweep_contained(old, coordinates[neighbour], target,
                                    coordinates[neighbour], mask):
                return False
            # The moving vertex must not pass through either incident endpoint.
            if _segment_contacts(old_physical, new_physical, physical[neighbour],
                                 physical[neighbour], tolerance):
                return False
    except ValueError:
        return False
    # Two incident edges may overlap while the vertex crosses their supporting
    # line beyond either neighbour. Crossing BETWEEN neighbours is safe.
    for index, first in enumerate(neighbours):
        for second in neighbours[index + 1:]:
            axis = physical[second] - physical[first]
            initial = np.cross(old_physical - physical[first], axis)
            change = np.cross(new_physical - old_physical, axis)
            fraction = (
                np.clip(-initial @ change / (change @ change), 0, 1)
                if change @ change else 0.
            )
            point = old_physical + fraction * (new_physical - old_physical)
            if (np.linalg.norm(initial + fraction * change)
                    <= tolerance * np.linalg.norm(axis)):
                along = (point - physical[first]) @ axis / (axis @ axis)
                if along <= 0 or along >= 1:
                    return False
            if change @ change == 0:
                along = (new_physical - physical[first]) @ axis / (axis @ axis)
                if (np.linalg.norm(initial) <= tolerance * np.linalg.norm(axis)
                        and not 0 < along < 1):
                    return False
    stationary = np.ones(len(edges), dtype=bool)
    stationary[incident] = False
    edge_points = physical[edges]
    for neighbour in neighbours:
        triangle = np.asarray([old_physical, physical[neighbour], new_physical])
        low, high = triangle.min(axis=0) - tolerance, triangle.max(axis=0) + tolerance
        candidates = np.flatnonzero(
            stationary & np.all(edge_points.max(axis=1) >= low, axis=1)
            & np.all(edge_points.min(axis=1) <= high, axis=1)
        )
        for edge_index in candidates:
            u, v = edges[edge_index]
            contacts = _segment_triangle_contacts(
                *physical[[u, v]], triangle, tolerance
            )
            if not contacts:
                continue
            # Only the represented common endpoint is an allowed contact.
            if neighbour not in (u, v) or any(
                np.linalg.norm(point - physical[neighbour]) > tolerance
                for point in contacts
            ):
                return False
    return True


def _graphml_tag(name):
    """Return a GraphML namespace-qualified XML tag."""
    return f'{{{GRAPHML_NAMESPACE}}}{name}'


def _add_graphml_data(element, key, value):
    """Append one GraphML data element."""
    data = ET.SubElement(element, _graphml_tag('data'), {'key': key})
    data.text = str(value)


def write_graphml(
    X,
    adjacency_matrix,
    component_labels,
    affine,
    volume_shape,
    output_path,
    binary_segmentation=None,
    edge_paths=None,
):
    """
    Write a contracted graph using the SkelHub Laplacian GraphML schema.

    Each edge stores an endpoint-inclusive 26-connected voxel run in
    ``centerline_voxels`` and the unrounded floating-point path in voxel
    coordinates in ``centerline_voxel_points``. If supplied, edge_paths retains
    the fitted polyline through degree-two simplification; otherwise the path
    contains the two node positions. Node radius is intentionally omitted.
    """
    X = np.asarray(X, dtype=float)
    component_labels = np.asarray(component_labels, dtype=int)
    affine = np.asarray(affine, dtype=float)
    volume_shape = tuple(int(bound) for bound in volume_shape)

    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError('X must have shape (N, 3).')
    if adjacency_matrix.shape != (X.shape[0], X.shape[0]):
        raise ValueError('adjacency_matrix shape must match the number of nodes.')
    if component_labels.shape != (X.shape[0],):
        raise ValueError('component_labels must contain one label per node.')
    if affine.shape != (4, 4):
        raise ValueError('affine must have shape (4, 4).')
    if len(volume_shape) != 3 or any(bound <= 0 for bound in volume_shape):
        raise ValueError('volume_shape must contain three positive dimensions.')

    ET.register_namespace('', GRAPHML_NAMESPACE)
    ET.register_namespace('xsi', GRAPHML_XSI_NAMESPACE)
    root = ET.Element(
        _graphml_tag('graphml'),
        {
            f'{{{GRAPHML_XSI_NAMESPACE}}}schemaLocation': (
                f'{GRAPHML_NAMESPACE} {GRAPHML_NAMESPACE}/1.0/graphml.xsd'
            )
        },
    )
    root.append(ET.Comment(' Created by laplskel '))

    key_definitions = (
        ('v_name', 'node', 'name', 'string'),
        ('v_laplacian_id', 'node', 'laplacian_id', 'double'),
        ('v_X', 'node', 'X', 'double'),
        ('v_Y', 'node', 'Y', 'double'),
        ('v_Z', 'node', 'Z', 'double'),
        ('v_voxel_pos', 'node', 'voxel_pos', 'string'),
        ('v_component_index', 'node', 'component_index', 'double'),
        ('v_component_label', 'node', 'component_label', 'double'),
        ('v_id', 'node', 'id', 'string'),
        ('e_laplacian_edge_id', 'edge', 'laplacian_edge_id', 'double'),
        (
            'e_source_laplacian_id',
            'edge',
            'source_laplacian_id',
            'double',
        ),
        (
            'e_target_laplacian_id',
            'edge',
            'target_laplacian_id',
            'double',
        ),
        ('e_centerline_voxels', 'edge', 'centerline_voxels', 'string'),
        ('e_centerline_voxel_points', 'edge', 'centerline_voxel_points', 'string'),
        (
            'e_num_centerline_voxels',
            'edge',
            'num_centerline_voxels',
            'double',
        ),
        ('e_component_index', 'edge', 'component_index', 'double'),
        ('e_component_label', 'edge', 'component_label', 'double'),
        (
            'e_component_edge_index',
            'edge',
            'component_edge_index',
            'double',
        ),
    )
    for key_id, scope, attribute_name, attribute_type in key_definitions:
        ET.SubElement(
            root,
            _graphml_tag('key'),
            {
                'id': key_id,
                'for': scope,
                'attr.name': attribute_name,
                'attr.type': attribute_type,
            },
        )

    graph = ET.SubElement(
        root,
        _graphml_tag('graph'),
        {'id': 'G', 'edgedefault': 'undirected'},
    )
    for node_id, (voxel_pos, component_label) in enumerate(
        zip(X, component_labels)
    ):
        node_name = f'n{node_id}'
        node = ET.SubElement(graph, _graphml_tag('node'), {'id': node_name})
        world_pos = (affine @ np.append(voxel_pos, 1.0))[:3]
        node_values = (
            ('v_name', node_id),
            ('v_laplacian_id', node_id),
            ('v_X', float(world_pos[0])),
            ('v_Y', float(world_pos[1])),
            ('v_Z', float(world_pos[2])),
            (
                'v_voxel_pos',
                json.dumps(voxel_pos.tolist(), separators=(',', ':')),
            ),
            ('v_component_index', int(component_label)),
            ('v_component_label', int(component_label)),
            ('v_id', node_name),
        )
        for key, value in node_values:
            _add_graphml_data(node, key, value)

    undirected_adjacency = adjacency_matrix.maximum(adjacency_matrix.T)
    upper_adjacency = sparse.triu(undirected_adjacency, k=1, format='coo')
    edge_order = np.lexsort((upper_adjacency.col, upper_adjacency.row))
    component_edge_indices = {}
    for edge_id, edge_position in enumerate(edge_order):
        source = int(upper_adjacency.row[edge_position])
        target = int(upper_adjacency.col[edge_position])
        component_label = int(component_labels[source])
        if component_label != int(component_labels[target]):
            raise ValueError('Graph edges cannot connect different components.')

        component_edge_index = component_edge_indices.get(component_label, 0)
        component_edge_indices[component_label] = component_edge_index + 1
        path = X[[source, target]] if edge_paths is None else edge_paths[(source, target)]
        centerline_voxels = []
        for start, end in zip(path[:-1], path[1:]):
            for voxel in _edge_voxels(start, end, volume_shape, binary_segmentation):
                if not centerline_voxels or centerline_voxels[-1] != voxel:
                    centerline_voxels.append(voxel)

        edge = ET.SubElement(
            graph,
            _graphml_tag('edge'),
            {'source': f'n{source}', 'target': f'n{target}'},
        )
        edge_values = (
            ('e_laplacian_edge_id', edge_id),
            ('e_source_laplacian_id', source),
            ('e_target_laplacian_id', target),
            (
                'e_centerline_voxels',
                json.dumps(centerline_voxels, separators=(',', ':')),
            ),
            (
                'e_centerline_voxel_points',
                json.dumps(np.asarray(path, dtype=float).tolist(), separators=(',', ':')),
            ),
            ('e_num_centerline_voxels', len(centerline_voxels)),
            ('e_component_index', component_label),
            ('e_component_label', component_label),
            ('e_component_edge_index', component_edge_index),
        )
        for key, value in edge_values:
            _add_graphml_data(edge, key, value)

    if hasattr(ET, 'indent'):
        ET.indent(root, space='  ')
    ET.ElementTree(root).write(
        output_path,
        encoding='utf-8',
        xml_declaration=True,
    )


def label_and_sort_by_size(binary_mask, label_connectivity=6):
    """
    Label connected components and re-order by size in reverse round-robin fashion.

    Parameters
    ----------
    binary_mask : np.ndarray
        Volume to label
    label_connectivity : 6, 18, 26, optional
        Connectivity profile to use to separate streams - 6, 18, or 26 edges.

    Returns
    -------
    labeled_volume
        Labeled volume in precise order
    num_features
        Number of features extracted from binary_mask

    Raises
    ------
    ValueError
        If label_connectivity is not a valid number
    """
    CONN = {6: 1, 18: 2, 26: 3}
    if label_connectivity not in CONN:
        raise ValueError(
            f'Label connectivity {label_connectivity} is not a valid option [6, 18, 26].'
        )

    labeled_volume, num_features = ndimage.label(
        binary_mask,
        structure=ndimage.generate_binary_structure(3, CONN[label_connectivity]),
    )

    print(
        f'Divided volume into {num_features} distinct label components '
        f'({label_connectivity}-connectivity).'
    )

    if num_features == 0:
        return labeled_volume, 0

    sizes = ndimage.sum(
        binary_mask,
        labeled_volume,
        index=np.arange(1, num_features + 1, dtype=np.int32),
    )
    # Sort descending
    sorted_labels = np.argsort(sizes)[::-1] + 1

    mapping = np.zeros(num_features + 1, dtype=labeled_volume.dtype)
    mapping[sorted_labels] = np.arange(1, num_features + 1, dtype=labeled_volume.dtype)

    return mapping[labeled_volume], num_features
