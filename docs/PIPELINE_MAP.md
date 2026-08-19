# Study code map

| Study component | Code |
|---|---|
| 3D dual-sequence ITHscore | `src/snc_ith/` |
| Clinical + MRI semantic LASSO-Cox | `scripts/clinical/clinical_semantic_lasso_cox.R` |
| SCC/ADC clinical subgroup | `scripts/clinical/histology_scc_adc_lasso_cox.R` |
| HCR extraction, ICC, filtering, ComBat, mRMR, LASSO-Cox | `scripts/hcr/` |
| Seven-model performance | `scripts/models/seven_model_evaluation.py` |
| Bootstrap/LRT/NRI/IDI/internal validation | `scripts/models/stability_validation.py` |
| Optimal-model KM stratification | `scripts/models/km_survival_analysis.R` |
| Histology-adjusted/subgroup analysis | `scripts/sensitivity/histology_*` |
| ITH-clinicopathologic correlation + FDR | `scripts/sensitivity/ithscore_clinical_correlation.py` |
| ITH-volume independence/increment | `scripts/sensitivity/ithscore_volume_analysis.py` |
| K=2-10 post-selection sensitivity | `scripts/sensitivity/k_sensitivity_analysis.py` |
| Formal K=3 replacement audit | `scripts/sensitivity/reexport_corrected_k_scores.py` |
| Dice and semantic inter-reader agreement | `scripts/reproducibility/` |

The scripts are preserved as analysis records. Several use study-specific path
defaults; review their top-level configuration before reuse. Inputs and outputs
are deliberately absent from this repository.
