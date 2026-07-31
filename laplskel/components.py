"""Connected-component labeling and ordering."""

import numpy as np
from scipy import ndimage


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

    counts = np.bincount(labeled_volume.ravel())
    # Exclude background (index 0) and sort descending
    sorted_labels = np.argsort(counts[1:])[::-1] + 1

    mapping = np.zeros(num_features + 1, dtype=labeled_volume.dtype)
    mapping[sorted_labels] = np.arange(1, num_features + 1)

    return mapping[labeled_volume], num_features
