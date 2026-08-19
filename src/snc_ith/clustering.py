"""Patient-wise voxel clustering for dual-sequence MRI features."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import MinMaxScaler


def fuse_sequences(fs_t2_features: np.ndarray, ce_t1_features: np.ndarray) -> np.ndarray:
    """Concatenate aligned FS-T2WI and CE-T1WI voxel feature matrices."""
    a, b = np.asarray(fs_t2_features, float), np.asarray(ce_t1_features, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("Both feature matrices must be 2-D with identical voxel rows")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Voxel feature matrices contain NaN or Inf")
    return np.hstack([a, b])


def cluster_voxels(
    mask: np.ndarray,
    fused_features: np.ndarray,
    *,
    n_clusters: int = 3,
    random_state: int = 2024,
) -> tuple[np.ndarray, float]:
    """Min-max scale within patient, fit K-means, and return map plus CH index."""
    roi = np.asarray(mask).astype(bool)
    if roi.ndim != 3:
        raise ValueError("mask must be 3-D")
    x = np.asarray(fused_features, float)
    if x.shape[0] != int(roi.sum()):
        raise ValueError("Feature rows must follow np.argwhere(mask) order and equal ROI voxels")
    if x.shape[0] <= n_clusters:
        raise ValueError("Number of tumor voxels must exceed n_clusters")
    x_scaled = MinMaxScaler().fit_transform(x)
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    cluster_ids = model.fit_predict(x_scaled)
    label_map = np.zeros(roi.shape, dtype=np.int16)
    label_map[roi] = cluster_ids + 1
    ch = float(calinski_harabasz_score(x_scaled, cluster_ids))
    return label_map, ch
