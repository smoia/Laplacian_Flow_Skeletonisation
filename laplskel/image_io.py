"""NIfTI input/output and skeleton rasterization."""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
from nigsp import io
from scipy import sparse


GRAPHML_NAMESPACE = 'http://graphml.graphdrawing.org/xmlns'
GRAPHML_XSI_NAMESPACE = 'http://www.w3.org/2001/XMLSchema-instance'


def load_nifti_mask(nifti_path):
    """Load a three-dimensional NIfTI segmentation mask."""
    return io.load_nifti_get_mask(nifti_path, is_mask=True, ndim=3)


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


def get_output_path(nifti_path, out_path):
    """Return an explicit or input-derived skeleton output path."""
    if out_path:
        return out_path
    return f'{os.path.splitext(os.path.splitext(nifti_path)[0])[0]}_skel'


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

    ET.indent(root, space='  ')
    ET.ElementTree(root).write(
        output_path,
        encoding='utf-8',
        xml_declaration=True,
    )


def save_skeleton(
    out_path,
    contracted_X,
    final_adj,
    volume_data,
    img,
    enforce_containment,
    graphml=False,
    component_labels=None,
):
    """Save graph data and a dense NIfTI skeleton."""
    print(f'\nSaving structural centerline data to: {out_path}')
    if graphml:
        if component_labels is None:
            raise ValueError('component_labels are required for GraphML output.')
        write_graphml(
            contracted_X,
            final_adj,
            component_labels=component_labels,
            affine=img.affine,
            volume_shape=volume_data.shape,
            output_path=f'{out_path}.graphml',
        )
    else:
        np.savez_compressed(f'{out_path}_coords.npz', contracted_X=contracted_X)
        sparse.save_npz(f'{out_path}.npz', final_adj)
    nifti_skel = coords_to_dense_3d(contracted_X, volume_data.shape)

    # If enforce containment was used, assume no loss of tracts masking with original segmentation.
    if enforce_containment:
        nifti_skel = nifti_skel * volume_data

    io.export_nifti(nifti_skel, img, f'{out_path}.nii.gz')
    return nifti_skel
