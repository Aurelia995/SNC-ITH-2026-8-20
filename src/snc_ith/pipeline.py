"""End-to-end single-patient 3D-ITHscore pipeline."""

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .clustering import cluster_voxels
from .features import extract_dual_sequence_voxel_features
from .score import calculate_3d_ithscore


def run_patient(fs_t2_path: str, ce_t1_path: str, mask_path: str, output_map: str, *, k: int = 3) -> dict:
    fs_t2, ce_t1, mask = map(sitk.ReadImage, [fs_t2_path, ce_t1_path, mask_path])
    if fs_t2.GetSize() != ce_t1.GetSize() or fs_t2.GetSize() != mask.GetSize():
        raise ValueError("Inputs must already be registered/resampled to the same grid")
    features, names, _ = extract_dual_sequence_voxel_features(fs_t2, ce_t1, mask)
    label_map, ch = cluster_voxels(sitk.GetArrayFromImage(mask), features, n_clusters=k)
    score = calculate_3d_ithscore(label_map)
    out = sitk.GetImageFromArray(label_map)
    out.CopyInformation(mask)
    Path(output_map).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, output_map)
    return {"k": k, "ithscore_3d": score, "calinski_harabasz": ch, "n_features": len(names)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate dual-sequence MRI 3D-ITHscore")
    parser.add_argument("--fs-t2wi", required=True)
    parser.add_argument("--ce-t1wi", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--summary-json")
    args = parser.parse_args()
    result = run_patient(args.fs_t2wi, args.ce_t1wi, args.mask, args.label_map, k=args.k)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.summary_json:
        Path(args.summary_json).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
