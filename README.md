# SNC-ITH-2026-8-20

Reproducible research code for 3D intratumoral heterogeneity (3D-ITHscore),
clinical/HCR/survival modelling, and reviewer-requested sensitivity analyses in
sinonasal carcinoma (SNC).

## Scope and validation status

The 3D-ITHscore implementation is a documented reconstruction made after the
original project script was lost. It follows the study Methods/Supplement and
the official MIT-licensed `ITHscore` 0.3.3 implementation by Li et al. The key
study adaptations are:

1. the entire registered 3-D tumor volume replaces the largest 2-D slice;
2. a 5x5x5 voxel-centred cube replaces the 5x5 pixel window;
3. FS-T2WI and CE-T1WI local feature vectors are concatenated before clustering;
4. connected components use 26-connectivity;
5. global K=3 is fixed from the prespecified image-driven CH analysis and
   non-outcome ablation analysis.

The scoring equation and topology implementation have unit tests. Full
patient-level voxel re-extraction was **not** rerun because it requires the
original registered images/masks and high-compute environment. Consequently,
this repository does not claim bit-for-bit equivalence with the deleted script.
No survival outcome is used to choose K or calibrate the reconstructed score.

## Repository map

- `src/snc_ith/`: reconstructed 3-D dual-sequence ITHscore pipeline.
- `scripts/clinical/`: clinical + semantic MRI LASSO-Cox analyses.
- `scripts/hcr/`: handcrafted radiomics/ICC/ComBat/mRMR/LASSO-Cox pipeline.
- `scripts/models/`: seven-model evaluation, bootstrap stability and KM analysis.
- `scripts/sensitivity/`: histology, K, volume and correlation analyses.
- `scripts/reproducibility/`: segmentation and semantic-reader agreement.
- `third_party/ITHscore_0.3.3/`: the reference scoring function and its MIT license.
- `docs/`: methods, provenance, data contract and reproducibility guidance.

## Installation and tests

```bash
python -m venv .venv
python -m pip install -e ".[test]"
pytest
```

PyRadiomics extraction is intentionally excluded from continuous integration;
tests use synthetic arrays and cover the score and deterministic clustering.

## Data availability and privacy

Patient-level CSV/XLSX data, MRI volumes, masks, fitted objects, SSH credentials,
and derived patient scores are intentionally excluded. Scripts retain command
line/path parameters or their original analysis constants to document the exact
workflow; users must provide authorized data locally. See `docs/DATA_CONTRACT.md`.

## Primary method source

Li J, Qiu Z, Zhang C, et al. *ITHscore: comprehensive quantification of
intra-tumor heterogeneity in NSCLC by multi-scale radiomic features.* European
Radiology. 2023;33:893-903. DOI: 10.1007/s00330-022-09055-0.

Official implementation: https://github.com/LiJiaqi96/ITHscore and
https://pypi.org/project/ITHscore/ (version 0.3.3).
