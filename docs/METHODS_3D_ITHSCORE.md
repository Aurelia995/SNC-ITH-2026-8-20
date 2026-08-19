# 3D-ITHscore construction method

## Preprocessing contract

FS-T2WI and CE-T1WI must already be rigid/deformably registered as used in the
study, N4-corrected, resampled to 1x1x1 mm3, histogram-normalized, and aligned
with a binary whole-tumor mask. The code rejects
different array sizes; it does not silently register or resample inputs.

## Local radiomics and fusion

For every voxel inside the tumor, a 5x5x5 cube is constructed. PyRadiomics
original first-order, 3-D shape, GLCM, GLDM, GLRLM, GLSZM and NGTDM features are
extracted from the tumor portion within that cube, independently for FS-T2WI and
CE-T1WI. The two feature vectors are concatenated by voxel. Patient-wise min-max
scaling maps each fused feature to [0,1].

The analysis records all PyRadiomics feature names and their sequence prefixes
to preserve the complete voxel feature matrix used for clustering.

## Clustering

K-means is fitted within each patient on fused voxel features. The formal study
uses global K=3. `random_state=2024` and `n_init=20` make clustering
deterministic. The original K selection was image-driven: mean
Calinski-Harabasz index over K=2-8, followed by a non-outcome clinicopathologic
ablation. OS was not used. Later K=2-10 survival analyses are post-selection
sensitivity analyses and must not be used to reselect K.

## 3-D score

For cluster i, let n_i be the number of retained 26-connected components and
S_i,max the voxel volume of its largest retained component. Let S_total be total
tumor voxel volume. Then:

`3D-ITHscore = 1 - (1 / S_total) * sum_i(S_i,max / n_i)`.

The official package ignores components <=2 voxels when tumor volume <=200 and
otherwise ignores single-voxel components. This rule is retained, but expressed
in voxel counts because all images are resampled to isotropic 1 mm3.

## Quality controls

The implementation includes equation/topology tests, label
permutation invariance checks, deterministic clustering checks, finite-value
guards, image-grid consistency checks and explicit recording of the runtime
PyRadiomics feature names.
