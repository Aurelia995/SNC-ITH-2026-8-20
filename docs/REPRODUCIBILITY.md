# Reproducibility checklist

1. Record hashes of registered images, masks and tabular inputs.
2. Pin Python/R package versions and archive the PyRadiomics parameter YAML.
3. Confirm both MRI sequences and mask share size, origin, spacing and direction.
4. Confirm the installed PyRadiomics version returns the intended 104-feature
   manifest per sequence; do not select features by output position.
5. Run K=2-8 CH calculations without loading survival outcomes; retain K=3 for
   the formal pipeline.
6. Save label maps, CH values, feature names, K-means seed and software versions.
7. Fit preprocessing/model selection on training data only.
8. Treat the n=22 validation cohort as descriptive where prespecified.
9. Run `pytest` and archive the test log before a full rerun.

The reconstructed voxel extractor is computationally intensive and should be
executed on the study server. CI validates only topology and clustering logic.
