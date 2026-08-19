"""Voxel-centred 3-D local radiomics extraction for two registered MRI sequences.

The implementation follows the study's voxel-centred 5x5x5 sliding-cube method.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor


FEATURE_PREFIXES = (
    "original_firstorder_",
    "original_shape_",
    "original_glcm_",
    "original_gldm_",
    "original_glrlm_",
    "original_glszm_",
    "original_ngtdm_",
)


def _local_cube(shape: tuple[int, int, int], center: Iterable[int], radius: int = 2) -> tuple[slice, ...]:
    return tuple(slice(max(0, int(c) - radius), min(n, int(c) + radius + 1)) for c, n in zip(center, shape))


def extract_sequence_voxel_features(
    image: sitk.Image,
    mask: sitk.Image,
    *,
    window_size: int = 5,
    settings: dict | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Extract local features for every ROI voxel using a 5x5x5 sliding cube."""
    if window_size != 5 or window_size % 2 != 1:
        raise ValueError("The study protocol specifies a 5x5x5 voxel window")
    image_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    if image_arr.shape != mask_arr.shape:
        raise ValueError("Registered image and mask arrays must have identical shapes")
    coordinates = np.argwhere(mask_arr)
    if coordinates.size == 0:
        raise ValueError("Mask is empty")

    extractor = featureextractor.RadiomicsFeatureExtractor(**(settings or {}))
    rows: list[list[float]] = []
    names: list[str] | None = None
    radius = window_size // 2
    for center in coordinates:
        region = _local_cube(mask_arr.shape, center, radius)
        local_image_arr = image_arr[region]
        # Local shape and texture describe the tumor portion inside the cube.
        local_mask_arr = mask_arr[region].astype(np.uint8)
        if local_mask_arr.sum() < 2:
            raise RuntimeError(f"Insufficient local mask voxels at {center.tolist()}")
        local_image = sitk.GetImageFromArray(local_image_arr)
        local_mask = sitk.GetImageFromArray(local_mask_arr)
        result = extractor.execute(local_image, local_mask, label=1)
        selected = [(k, float(v)) for k, v in result.items() if k.startswith(FEATURE_PREFIXES)]
        current_names = [k for k, _ in selected]
        if names is None:
            names = current_names
        elif current_names != names:
            raise RuntimeError("PyRadiomics feature order changed between voxels")
        rows.append([v for _, v in selected])
    return np.asarray(rows, dtype=float), (names or []), coordinates


def extract_dual_sequence_voxel_features(
    fs_t2: sitk.Image,
    ce_t1: sitk.Image,
    mask: sitk.Image,
    **kwargs,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Extract both sequences and concatenate features for aligned voxel rows."""
    fs, fs_names, coords = extract_sequence_voxel_features(fs_t2, mask, **kwargs)
    ce, ce_names, ce_coords = extract_sequence_voxel_features(ce_t1, mask, **kwargs)
    if not np.array_equal(coords, ce_coords):
        raise RuntimeError("Sequence voxel ordering is not aligned")
    names = [f"FS-T2WI_{x}" for x in fs_names] + [f"CE-T1WI_{x}" for x in ce_names]
    return np.hstack([fs, ce]), names, coords
