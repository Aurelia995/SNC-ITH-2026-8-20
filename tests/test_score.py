import numpy as np
import pytest

from snc_ith.score import calculate_3d_ithscore


def test_compact_single_cluster_is_zero():
    x = np.ones((5, 5, 5), dtype=int)
    assert calculate_3d_ithscore(x) == pytest.approx(0.0)


def test_fragmentation_increases_score():
    compact = np.ones((5, 5, 5), dtype=int)
    fragmented = compact.copy()
    fragmented[2, :, :] = 0
    fragmented[:, 2, :] = 0
    fragmented[:, :, 2] = 0
    assert calculate_3d_ithscore(fragmented) > calculate_3d_ithscore(compact)


def test_label_permutation_invariant():
    x = np.zeros((6, 6, 6), dtype=int)
    x[:3] = 1
    x[3:] = 2
    y = np.where(x == 1, 2, np.where(x == 2, 1, 0))
    assert calculate_3d_ithscore(x) == pytest.approx(calculate_3d_ithscore(y))


def test_empty_map_rejected():
    with pytest.raises(ValueError):
        calculate_3d_ithscore(np.zeros((2, 2, 2), dtype=int))
