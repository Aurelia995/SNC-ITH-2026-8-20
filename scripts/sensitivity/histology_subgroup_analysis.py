from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from scipy.stats import chi2
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

SEED = 20250308
rng = np.random.default_rng(SEED)
SRC = Path("/root/autodl-tmp/HCR_runs/seven_models_ith3_original_20260814")
CLIN = Path("/root/autodl-tmp/HCR_input/临床模型数据集.csv")
OUT = Path("/root/autodl-tmp/HCR_runs/seven_models_histology_20260814")
OUT.mkdir(parents=True, exist_ok=True)


def read_clinical() -> pd.DataFrame:
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(CLIN, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Cannot decode clinical CSV")


def norm_id(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.lower()
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"^c(?=\d+$)", "", regex=True)
            .str.replace(r"^0+(?=\d+$)", "", regex=True))


clin = read_clinical()
clin["id_key"] = norm_id(clin["patient_id"])
rename = {
    "TNM overall stage": "TNM_stage",
    "性别": "Sex",
    "年龄": "Age",
    "位置（鼻腔1，鼻窦2）": "Primary_site",
    "病理类型（1鳞癌；2腺癌；3未分化癌；4鼻窦SMARCB1缺失癌）": "Histology",
}
clin = clin.rename(columns=rename)
keep_clin = ["id_key", "TNM_stage", "Sex", "Age", "Primary_site", "Histology"]
clin = clin[keep_clin]


def load_analysis(name: str, cohort: str) -> pd.DataFrame:
    d = pd.read_csv(SRC / name)
    d["id_key"] = norm_id(d["patient_id"])
    d = d.merge(clin, on="id_key", how="left", validate="one_to_one")
    if d[keep_clin[1:]].isna().all(axis=1).any():
        raise RuntimeError(f"Clinical merge failed for {cohort}")
    d["Cohort"] = cohort
    d = d.rename(columns={"OS/m": "OS_time", "OS/label": "OS_event"})
    return d


train = load_analysis("analysis_dataset_training.csv", "Training")
valid = load_analysis("analysis_dataset_validation.csv", "Validation")
if len(train) != 129 or len(valid) != 22:
    raise RuntimeError("Unexpected cohort size")

hist_labels = {1: "Squamous cell carcinoma", 2: "Adenocarcinoma",
               3: "Undifferentiated carcinoma", 4: "SMARCB1-deficient carcinoma"}
stage_labels = {2: "Stage II", 3: "Stage III", 4: "Stage IVA", 5: "Stage IVB"}

dist = []
for cohort, d in [("Training", train), ("Validation", valid), ("All", pd.concat([train, valid]))]:
    for code in (1, 2, 3, 4):
        z = d[d.Histology == code]
        dist.append({"Cohort": cohort, "Histology_code": code, "Histology": hist_labels[code],
                     "N": len(z), "Events": int(z.OS_event.sum()),
                     "Percent": 100 * len(z) / len(d)})
pd.DataFrame(dist).to_csv(OUT / "Table_Histology_Distribution.csv", index=False)

# Task 1: forced-entry Cox model in training cohort. One missing stage is handled
# by complete-case primary analysis and training-mode imputation sensitivity.
forced_cols = ["OS_time", "OS_event", "TNM_stage", "Sex", "Age", "NLR",
               "Primary_site", "Histology", "ith_score"]
forced_raw = train[forced_cols].copy()
missing = forced_raw.isna().sum().rename("Missing_N").to_frame()
missing["Missing_percent"] = 100 * missing.Missing_N / len(forced_raw)
missing.to_csv(OUT / "Table_Forced_Cox_Missingness.csv")


def design_forced(raw: pd.DataFrame, impute_stage: bool = False):
    d = raw.copy()
    if impute_stage:
        mode = train.TNM_stage.mode(dropna=True).iloc[0]
        d["TNM_stage"] = d["TNM_stage"].fillna(mode)
    else:
        d = d.dropna()
    d["Age_per10y"] = d["Age"] / 10.0
    d["ITH_per0.1"] = d["ith_score"] * 10.0
    d["Sex_male"] = d["Sex"].astype(int)
    d["Site_sinus"] = (d["Primary_site"] == 2).astype(int)
    for code in (3, 4, 5):
        d[f"TNM_{code}"] = (d["TNM_stage"] == code).astype(int)
    for code in (2, 3, 4):
        d[f"Hist_{code}"] = (d["Histology"] == code).astype(int)
    xcols = ["TNM_3", "TNM_4", "TNM_5", "Sex_male", "Age_per10y", "NLR",
             "Site_sinus", "Hist_2", "Hist_3", "Hist_4", "ITH_per0.1"]
    return d[["OS_time", "OS_event"] + xcols], xcols


def fit_forced(raw: pd.DataFrame, impute: bool, analysis: str):
    d, xcols = design_forced(raw, impute)
    fit = CoxPHFitter().fit(d, "OS_time", "OS_event", show_progress=False)
    ph = proportional_hazard_test(fit, d, time_transform="rank").summary
    meta = {
        "TNM_3": ("TNM overall stage", "Stage III vs Stage II", "Clinical"),
        "TNM_4": ("TNM overall stage", "Stage IVA vs Stage II", "Clinical"),
        "TNM_5": ("TNM overall stage", "Stage IVB vs Stage II", "Clinical"),
        "Sex_male": ("Sex", "Male vs female", "Clinical"),
        "Age_per10y": ("Age", "Per 10-year increase", "Clinical"),
        "NLR": ("NLR", "Per 1-unit increase", "Clinical"),
        "Site_sinus": ("Primary site", "Sinonasal sinus vs nasal cavity", "Clinical"),
        "Hist_2": ("Histologic subtype", "Adenocarcinoma vs SCC", "Pathology"),
        "Hist_3": ("Histologic subtype", "Undifferentiated carcinoma vs SCC", "Pathology"),
        "Hist_4": ("Histologic subtype", "SMARCB1-deficient carcinoma vs SCC", "Pathology"),
        "ITH_per0.1": ("3D-ITHscore", "Per 0.1 increase", "Imaging"),
    }
    rows = []
    for term in xcols:
        s = fit.summary.loc[term]
        rows.append({"Analysis": analysis, "N": len(d), "Events": int(d.OS_event.sum()),
                     "Variable": meta[term][0], "Level_or_scale": meta[term][1],
                     "Group": meta[term][2], "Term": term, "Beta": s["coef"],
                     "HR": s["exp(coef)"], "CI_L95": s["exp(coef) lower 95%"],
                     "CI_U95": s["exp(coef) upper 95%"], "P": s["p"],
                     "PH_test_P": ph.loc[term, "p"]})
    out = pd.DataFrame(rows)
    # Likelihood-ratio global tests for multi-level variables.
    global_groups = {"TNM overall stage": ["TNM_3", "TNM_4", "TNM_5"],
                     "Histologic subtype": ["Hist_2", "Hist_3", "Hist_4"]}
    out["Global_P"] = np.nan
    for var, terms in global_groups.items():
        red_cols = [c for c in xcols if c not in terms]
        red = CoxPHFitter().fit(d[["OS_time", "OS_event"] + red_cols], "OS_time", "OS_event")
        lr = 2 * (fit.log_likelihood_ - red.log_likelihood_)
        gp = chi2.sf(lr, len(terms))
        out.loc[out.Variable == var, "Global_P"] = gp
    out["Significant"] = np.where(out.P < 0.05, "Yes", "No")
    return fit, d, out


forced_cc, forced_cc_data, forced_cc_table = fit_forced(forced_raw, False, "Complete-case primary")
forced_imp, forced_imp_data, forced_imp_table = fit_forced(forced_raw, True, "Mode-imputed sensitivity")
forced_table = pd.concat([forced_cc_table, forced_imp_table], ignore_index=True)
forced_table.to_csv(OUT / "Table_Forced_Entry_Cox.csv", index=False)

# Task 2: refit the fixed Clinical-ITH predictor set after removing rare histologies.
model_cols = ["NLR", "Ki67", "Margin", "Myxoid", "ith_score"]


def fit_clinical_ith(d: pd.DataFrame):
    z = d[["OS_time", "OS_event"] + model_cols].dropna().copy()
    fit = CoxPHFitter().fit(z, "OS_time", "OS_event")
    return fit, z


full_fit, full_train_model = fit_clinical_ith(train)
s_train = train[train.Histology.isin([1, 2])].copy()
s_valid = valid[valid.Histology.isin([1, 2])].copy()
sub_fit, sub_train_model = fit_clinical_ith(s_train)
if len(s_train) != 122 or len(s_valid) != 21:
    raise RuntimeError(f"Unexpected subgroup sizes: {len(s_train)}, {len(s_valid)}")


def coef_table(fit, label):
    s = fit.summary
    return pd.DataFrame({"Analysis": label, "Variable": s.index, "Beta": s["coef"].values,
                         "HR": s["exp(coef)"].values,
                         "CI_L95": s["exp(coef) lower 95%"].values,
                         "CI_U95": s["exp(coef) upper 95%"].values,
                         "P": s["p"].values})


coef_full = coef_table(full_fit, "Full cohort training")
coef_sub = coef_table(sub_fit, "SCC+ADC training")
coef_compare = coef_full.merge(coef_sub, on="Variable", suffixes=("_full", "_subgroup"))
coef_compare["Beta_difference_subgroup_minus_full"] = coef_compare.Beta_subgroup - coef_compare.Beta_full
coef_compare["Relative_absolute_beta_change_percent"] = (
    100 * np.abs(coef_compare.Beta_difference_subgroup_minus_full) /
    np.maximum(np.abs(coef_compare.Beta_full), 1e-12)
)
coef_compare["Direction_consistent"] = np.sign(coef_compare.Beta_full) == np.sign(coef_compare.Beta_subgroup)
coef_compare.to_csv(OUT / "Table_Coefficient_Stability.csv", index=False)
pd.concat([coef_full, coef_sub]).to_csv(OUT / "Table_Clinical_ITH_Coefficients.csv", index=False)


def surv_array(d):
    return Surv.from_arrays(d.OS_event.astype(bool).to_numpy(), d.OS_time.astype(float).to_numpy())


def harrell(d, risk):
    return concordance_index_censored(d.OS_event.astype(bool), d.OS_time.astype(float), risk)[0]


def bootstrap_harrell(d, risk, B=2000):
    vals = []
    n = len(d)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            vals.append(concordance_index_censored(d.OS_event.iloc[idx].astype(bool),
                                                    d.OS_time.iloc[idx].astype(float), risk[idx])[0])
        except Exception:
            pass
    return np.quantile(vals, [0.025, 0.975]), len(vals)


def evaluate(label, fit, tr, va):
    xtr, xva = tr[model_cols], va[model_cols]
    rtr = fit.predict_log_partial_hazard(xtr).to_numpy()
    rva = fit.predict_log_partial_hazard(xva).to_numpy()
    out = []
    for cohort, d, risk, tau in [("Training", tr, rtr, 60.0), ("Validation", va, rva, 24.0)]:
        ci, bs = bootstrap_harrell(d.reset_index(drop=True), risk)
        y_ref = surv_array(tr)
        y = surv_array(d)
        uno = concordance_index_ipcw(y_ref, y, risk, tau=tau)[0]
        times = np.array([12., 24., 36., 60.]) if cohort == "Training" else np.array([12., 24.])
        aucs, iauc = cumulative_dynamic_auc(y_ref, y, risk, times)
        surv_prob = fit.predict_survival_function(d[model_cols], times=times).T.to_numpy()
        ibs = integrated_brier_score(y_ref, y, surv_prob, times)
        row = {"Population": label, "Cohort": cohort, "N": len(d), "Events": int(d.OS_event.sum()),
               "Harrell_C": harrell(d, risk), "Harrell_C_L95": ci[0], "Harrell_C_U95": ci[1],
               "Harrell_bootstrap_successful": bs, "Uno_C": uno, "Uno_tau_months": tau,
               "iAUC": iauc, "IBS": ibs}
        for t, a in zip(times, aucs): row[f"AUC_{int(t)}m"] = a
        out.append(row)
    return out, rtr, rva


full_perf, _, _ = evaluate("Full cohort", full_fit, train, valid)
sub_perf, sub_risk_train, sub_risk_val = evaluate("SCC+ADC subgroup", sub_fit, s_train, s_valid)
pd.DataFrame(full_perf + sub_perf).to_csv(OUT / "Table_Performance_Full_vs_Subgroup.csv", index=False)

# X-tile-equivalent cutpoint: maximize training log-rank chi-square with >=20% per group.
def logrank_chisq(d, risk, cut):
    from lifelines.statistics import logrank_test
    high = risk > cut
    z = logrank_test(d.OS_time[~high], d.OS_time[high],
                     event_observed_A=d.OS_event[~high], event_observed_B=d.OS_event[high])
    return z.test_statistic, z.p_value


min_n = math.ceil(0.20 * len(s_train))
cuts = []
for cut in np.unique(sub_risk_train):
    nh = int((sub_risk_train > cut).sum()); nl = len(s_train) - nh
    if nh >= min_n and nl >= min_n:
        ch, p = logrank_chisq(s_train, sub_risk_train, cut)
        cuts.append({"Cutoff": cut, "N_low": nl, "N_high": nh, "Logrank_chisq": ch, "Logrank_P": p})
cut_df = pd.DataFrame(cuts).sort_values(["Logrank_chisq", "Cutoff"], ascending=[False, True])
best_cut = float(cut_df.iloc[0].Cutoff)
cut_df["Selected"] = np.where(cut_df.Cutoff == best_cut, "Yes", "No")
cut_df.to_csv(OUT / "Table_Subgroup_Cutpoint_Search.csv", index=False)


def score_frame(d, risk, cohort):
    return pd.DataFrame({"patient_id": d.patient_id.astype(str), "Cohort": cohort,
                         "OS_time": d.OS_time, "OS_event": d.OS_event.astype(int),
                         "Histology_code": d.Histology.astype(int), "Risk_score": risk,
                         "Risk_group": np.where(risk > best_cut, "High risk", "Low risk")})


score_train = score_frame(s_train, sub_risk_train, "Training")
score_val = score_frame(s_valid, sub_risk_val, "Validation")
score_all = pd.concat([score_train, score_val], ignore_index=True)
score_all["Cohort"] = "All subgroup"
score_train.to_csv(OUT / "Risk_scores_SCC_ADC_training.csv", index=False)
score_val.to_csv(OUT / "Risk_scores_SCC_ADC_validation.csv", index=False)
score_all.to_csv(OUT / "Risk_scores_SCC_ADC_all.csv", index=False)

summary = {
    "seed": SEED,
    "ith_source": "authoritative ith_3clusters",
    "forced_cox_primary": "training complete case",
    "full_training_N": len(train), "full_training_events": int(train.OS_event.sum()),
    "subgroup_training_N": len(s_train), "subgroup_training_events": int(s_train.OS_event.sum()),
    "subgroup_validation_N": len(s_valid), "subgroup_validation_events": int(s_valid.OS_event.sum()),
    "subgroup_cutoff": best_cut, "cutpoint_min_group_fraction": 0.20,
    "harrell_bootstrap": 2000,
}
(OUT / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
