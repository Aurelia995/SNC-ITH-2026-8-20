import numpy as np

from snc_ith.clustering import cluster_voxels, fuse_sequences


def test_dual_sequence_fusion_and_deterministic_clustering():
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    a = np.arange(54, dtype=float).reshape(27, 2)
    b = a[:, ::-1]
    fused = fuse_sequences(a, b)
    m1, ch1 = cluster_voxels(mask, fused, n_clusters=3, random_state=2024)
    m2, ch2 = cluster_voxels(mask, fused, n_clusters=3, random_state=2024)
    assert fused.shape == (27, 4)
    assert np.array_equal(m1, m2)
    assert ch1 == ch2
