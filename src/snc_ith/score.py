"""Three-dimensional extension of Li et al.'s ITHscore.

The score is computed from a positive-integer 3-D cluster label map. Zero is
background. Connectivity is 26-neighbourhood, as specified in the SNC study.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def calculate_3d_ithscore(
    label_map: np.ndarray,
    *,
    min_volume: int = 200,
    small_component_threshold: int = 2,
) -> float:
    """Return 3D-ITHscore in [0, 1] from a 3-D cluster label map.

    This preserves the reference package's small-object rule: for tumors with
    at most ``min_volume`` voxels, components of size <= ``small_component_threshold``
    are ignored; otherwise single-voxel components are ignored. Connected
    components are evaluated with full 26-connectivity.
    """
    labels = np.asarray(label_map)
    if labels.ndim != 3:
        raise ValueError(f"label_map must be 3-D; received shape {labels.shape}")
    if not np.issubdtype(labels.dtype, np.number) or not np.all(np.isfinite(labels)):
        raise ValueError("label_map must contain finite numeric values")
    if np.any(labels < 0) or np.any(labels != np.floor(labels)):
        raise ValueError("labels must be non-negative integers with zero as background")

    tumor_volume = int(np.count_nonzero(labels))
    if tumor_volume == 0:
        raise ValueError("label_map contains no tumor voxels")

    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    contribution = 0.0
    threshold = small_component_threshold if tumor_volume <= min_volume else 1

    for cluster_label in np.unique(labels):
        if cluster_label == 0:
            continue
        component_map, n_components = ndimage.label(labels == cluster_label, structure=structure)
        sizes = np.bincount(component_map.ravel())[1 : n_components + 1]
        retained = sizes[sizes > threshold]
        if retained.size:
            contribution += float(retained.max()) / float(retained.size)

    score = 1.0 - contribution / float(tumor_volume)
    # Floating-point guard; mathematically the expression is bounded.
    return float(np.clip(score, 0.0, 1.0))
