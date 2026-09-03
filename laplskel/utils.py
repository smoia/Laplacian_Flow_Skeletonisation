"""Output and skeleton rasterization."""

import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy import ndimage, sparse


GRAPHML_NAMESPACE = 'http://graphml.graphdrawing.org/xmlns'
GRAPHML_XSI_NAMESPACE = 'http://www.w3.org/2001/XMLSchema-instance'


def coords_to_dense_3d(X, volume_shape):
    """
    Create and fill in a volume using coordinates of points.

    Parameters
    ----------
    X : ndarray of shape (N, 3)
        The 3D coordinates of the areas with content.
    volume_shape : tuple of int (D, H, W)
        The structural grid dimensions of the target 3D matrix.

    Returns
    -------
    dense_volume : ndarray of shape (D, H, W)
        A binary 3D array where 1 represents the skeleton path.
    """
    # 1. Initialize empty dense matrix
    dense_volume = np.zeros(volume_shape, dtype=bool)

    coords = np.rint(X).astype(np.int8)

    # 2. Fix coordinates on boundaries due to numpy's round-to-even
    for dim, bound in enumerate(volume_shape):
        coords[:, dim][coords[:, dim] == bound] = bound - 1

    # 3. Rasterize edges and nodes into the grid
    for i in coords:
        dense_volume[tuple(i)] = True

    return dense_volume


def _clip_voxel(voxel, volume_shape):
    """Clip one integer voxel coordinate to the target volume bounds."""
    clipped = np.clip(
        np.asarray(voxel, dtype=int),
        0,
        np.asarray(volume_shape, dtype=int) - 1,
    )
    return tuple(int(value) for value in clipped)


def _edge_voxels(start, end, volume_shape):
    """Return an endpoint-inclusive 26-connected voxel run for one graph edge."""
    start = np.rint(np.asarray(start, dtype=float)).astype(int)
    end = np.rint(np.asarray(end, dtype=float)).astype(int)
    delta = end - start
    steps = int(np.max(np.abs(delta)))
    if steps == 0:
        return [_clip_voxel(start, volume_shape)]

    voxels = []
    for step in range(steps + 1):
        point = np.rint(start + delta * (step / steps)).astype(int)
        voxel = _clip_voxel(point, volume_shape)
        if not voxels or voxels[-1] != voxel:
            voxels.append(voxel)
    return voxels


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
):
    """
    Write a contracted graph using the SkelHub Laplacian GraphML schema.

    Each edge stores a rounded, clipped, endpoint-inclusive 26-connected voxel
    run in ``centerline_voxels``. Node radius is intentionally omitted.
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
        centerline_voxels = _edge_voxels(
            X[source],
            X[target],
            volume_shape,
        )

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
