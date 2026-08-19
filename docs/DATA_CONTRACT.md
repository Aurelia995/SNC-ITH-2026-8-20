# Data contract

No study data are distributed. Expected inputs are local, authorized copies.

## Imaging pipeline

- registered FS-T2WI NIfTI
- registered CE-T1WI NIfTI
- binary whole-tumor mask on the same grid
- one record per de-identified patient ID

## Survival analyses

Scripts expect a cohort indicator, OS time in months, binary OS event, clinical
and MRI semantic variables described in their headers, 3D-ITHscore, and/or HCR
features. Training-only fitting and preprocessing must be preserved. Validation
data are transformed with training parameters and are not used for feature
selection, K selection or inferential model choice.

Never commit direct identifiers, dates, accession numbers, DICOM headers,
patient-level predictions, credentials or private server addresses.
