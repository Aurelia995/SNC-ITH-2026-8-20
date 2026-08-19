from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from scipy import stats

SEED = 20250308
RNG = np.random.default_rng(SEED)
OUT = Path("/root/autodl-tmp/HCR_runs/ith_volume_20260819")
OUT.mkdir(parents=True, exist_ok=True)

RAD_TRAIN = Path("/root/autodl-tmp/HCR_input/radiomics-TRAIN.csv")
RAD_VAL = Path("/root/autodl-tmp/HCR_input/radiomics-TEST.csv")
ITH_TRAIN = Path("/root/autodl-tmp/HCR_input/ITH3_TRAIN.csv")
ITH_VAL = Path("/root/autodl-tmp/HCR_input/ITH3_TEST.csv")
CLIN = Path("/root/autodl-tmp/HCR_input/clinical_model_dataset_utf8.csv")
PRED_TRAIN = Path("/root/autodl-tmp/HCR_runs/seven_models_ith3_original_20260814/risk_predictions_training.csv")
PRED_VAL = Path("/root/autodl-tmp/HCR_runs/seven_models_ith3_original_20260814/risk_predictions_validation.csv")


def norm_id(x):
    s = str(x).strip().lower().replace(".0", "")
    if s.startswith("c"):
        return s[1:].lstrip("0")
    return s.lstrip("0")


def read_clinical():
    d = pd.read_csv(CLIN, encoding="gb18030")
    if len(d) != 151:
        raise RuntimeError(f"Clinical dataset must contain 151 rows; got {len(d)}")
    safe = ["patient_id", "dataset", "event", "time", "TNM", "T_stage", "N_stage",
            "Sex", "Age", "Smoking", "NLR", "PLR", "LMR", "Site", "Histology",
            "Ki67", "Size_gt5", "T2_heterogeneity", "Margin", "Myxoid", "Necrosis",
            "Septations", "Enhancement"]
    d.columns = safe
    d["id_norm"] = d.patient_id.map(norm_id)
    d["dataset"] = d.dataset.astype(str).str.lower().map({"train": "training", "training": "training", "test": "validation", "validation": "validation"})
    return d


def load_cohort(rad_path, ith_path, cohort, pred_path):
    r = pd.read_csv(rad_path)
    i = pd.read_csv(ith_path)
    c = read_clinical()
    p = pd.read_csv(pred_path)
    for x, col in [(r, "id"), (i, "id"), (p, "patient_id")]:
        x["id_norm"] = x[col].map(norm_id)
    c = c[c.dataset == cohort].copy()
    vol_cols = ["T1_original_shape_MeshVolume", "T2_original_shape_MeshVolume",
                "T1_original_shape_VoxelVolume", "T2_original_shape_VoxelVolume"]
    if any(x not in r for x in vol_cols):
        raise RuntimeError("Required unprocessed PyRadiomics shape volume fields are absent")
    if not np.allclose(r[vol_cols[0]], r[vol_cols[1]], rtol=0, atol=1e-10):
        raise RuntimeError("T1 and T2 MeshVolume values differ; mask/geometry must be audited")
    r = r[["id_norm"] + vol_cols].copy()
    i = i[["id_norm", "ith_3clusters"]].rename(columns={"ith_3clusters": "ITHscore_3D"})
    p = p[["id_norm", "Clinical-ITH"]].rename(columns={"Clinical-ITH": "existing_Clinical_ITH_lp"})
    d = c.merge(r, on="id_norm", how="inner", validate="one_to_one").merge(i, on="id_norm", how="inner", validate="one_to_one").merge(p, on="id_norm", how="inner", validate="one_to_one")
    expected = 129 if cohort == "training" else 22
    if len(d) != expected:
        raise RuntimeError(f"{cohort} merge expected {expected}, got {len(d)}")
    d["Volume_mm3"] = d[vol_cols[0]]
    if (d.Volume_mm3 <= 0).any():
        raise RuntimeError("Non-positive MeshVolume prevents log transformation")
    d["Volume_cm3"] = d.Volume_mm3 / 1000.0
    d["log_Volume"] = np.log(d.Volume_cm3)
    return d


train = load_cohort(RAD_TRAIN, ITH_TRAIN, "training", PRED_TRAIN)
val = load_cohort(RAD_VAL, ITH_VAL, "validation", PRED_VAL)
if int(train.event.sum()) != 68 or int(val.event.sum()) != 6:
    raise RuntimeError(f"Unexpected events: train={train.event.sum()}, validation={val.event.sum()}")

# Training-only imputation values reproduce the prior seven-model analysis.
impute = {"NLR": float(train.NLR.median()), "Ki67": float(train.Ki67.median()),
          "Margin": float(train.Margin.mode().iloc[0]), "Myxoid": float(train.Myxoid.mode().iloc[0])}
for d in (train, val):
    for k, v in impute.items():
        d[k] = d[k].fillna(v)

train.to_csv(OUT / "Analysis_dataset_training.csv", index=False)
val.to_csv(OUT / "Analysis_dataset_validation.csv", index=False)

qc = pd.DataFrame([
    {"Cohort": name, "N": len(d), "Events": int(d.event.sum()), "Missing_volume": int(d.Volume_mm3.isna().sum()),
     "Volume_source": "raw PyRadiomics T1_original_shape_MeshVolume", "Units": "mm3 (also reported cm3)",
     "T1_T2_mesh_exact_match": bool(np.allclose(d.T1_original_shape_MeshVolume, d.T2_original_shape_MeshVolume, rtol=0, atol=1e-10)),
     "Median_volume_cm3": d.Volume_cm3.median(), "IQR_volume_cm3": f"{d.Volume_cm3.quantile(.25):.3f}-{d.Volume_cm3.quantile(.75):.3f}"}
    for name, d in [("Training", train), ("Validation", val)]
])
qc.to_csv(OUT / "Table_Volume_Extraction_QC.csv", index=False)


def boot_spearman(x, y, B=2000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho, p = stats.spearmanr(x, y)
    vals = []
    for _ in range(B):
        ix = RNG.integers(0, len(x), len(x))
        if np.unique(x[ix]).size > 1 and np.unique(y[ix]).size > 1:
            vals.append(stats.spearmanr(x[ix], y[ix]).statistic)
    lo, hi = np.quantile(vals, [.025, .975])
    return len(x), rho, lo, hi, p, len(vals)


def partial_spearman(x, y, z):
    a = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    rx = stats.rankdata(a.x); ry = stats.rankdata(a.y); rz = stats.rankdata(a.z)
    X = np.column_stack([np.ones(len(a)), rz])
    ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]
    t = r * np.sqrt((len(a)-3) / max(1e-15, 1-r*r))
    p = 2 * stats.t.sf(abs(t), len(a)-3)
    return r, p


def boot_partial(x, y, z, B=2000):
    a = pd.DataFrame({"x": x, "y": y, "z": z}).dropna().reset_index(drop=True)
    r, p = partial_spearman(a.x, a.y, a.z)
    vals = []
    for _ in range(B):
        b = a.iloc[RNG.integers(0, len(a), len(a))]
        if b.x.nunique() > 1 and b.y.nunique() > 1 and b.z.nunique() > 1:
            vals.append(partial_spearman(b.x, b.y, b.z)[0])
    return len(a), r, *np.quantile(vals, [.025, .975]), p, len(vals)


corr_rows = []
for cohort, d in [("Training", train), ("Validation", val)]:
    for label, y in [("log(Volume_cm3)", d.log_Volume), ("T_stage", d.T_stage), ("TNM_overall", d.TNM)]:
        n, r, lo, hi, p, nb = boot_spearman(d.ITHscore_3D, y)
        corr_rows.append({"Cohort": cohort, "Analysis": "Spearman", "Variables": f"ITHscore_3D vs {label}", "Adjusted_for": "None", "N": n,
                          "Estimate": r, "CI95_low": lo, "CI95_high": hi, "P_value": p if cohort == "Training" else np.nan,
                          "Bootstrap_valid": nb, "Role": "Primary inference" if cohort == "Training" else "Descriptive only; no hypothesis test"})
    for label, y in [("T_stage", d.T_stage), ("TNM_overall", d.TNM)]:
        n, r, lo, hi, p, nb = boot_partial(d.ITHscore_3D, y, d.log_Volume)
        corr_rows.append({"Cohort": cohort, "Analysis": "Partial Spearman (rank residualization)", "Variables": f"ITHscore_3D vs {label}", "Adjusted_for": "log(Volume_cm3)", "N": n,
                          "Estimate": r, "CI95_low": lo, "CI95_high": hi, "P_value": p if cohort == "Training" else np.nan,
                          "Bootstrap_valid": nb, "Role": "Primary inference" if cohort == "Training" else "Descriptive only; no hypothesis test"})
corr = pd.DataFrame(corr_rows)
corr.to_csv(OUT / "Table_Correlation_PartialCorrelation.csv", index=False)


def fit_cox(d, cols, penalizer=1e-7):
    z = d[["time", "event"] + cols].dropna().copy()
    m = CoxPHFitter(penalizer=penalizer)
    m.fit(z, duration_col="time", event_col="event", show_progress=False)
    return m, z


# Analysis 2: ITH adjusted for tumor volume and ordinal T stage.
adj, adjdata = fit_cox(train, ["ITHscore_3D", "T_stage", "log_Volume"])
adjtab = adj.summary.reset_index().rename(columns={"covariate": "Variable", "coef": "Beta", "exp(coef)": "HR", "p": "P_value", "coef lower 95%": "Beta_CI_low", "coef upper 95%": "Beta_CI_high"})
adjtab["HR_CI_low"] = np.exp(adjtab.Beta_CI_low)
adjtab["HR_CI_high"] = np.exp(adjtab.Beta_CI_high)
adjtab["Scale_note"] = adjtab.Variable.map({"ITHscore_3D": "per 1.0 increase (also see per-0.1 row)", "T_stage": "per one ordinal category", "log_Volume": "per one natural-log cm3"})
ithrow = adjtab[adjtab.Variable == "ITHscore_3D"].iloc[0].copy()
ithrow["Variable"] = "ITHscore_3D_per_0.1"
ithrow["Beta"] = ithrow.Beta * .1
ithrow["HR"] = math.exp(ithrow.Beta)
ithrow["HR_CI_low"] = math.exp(float(adjtab.loc[adjtab.Variable == "ITHscore_3D", "Beta_CI_low"].iloc[0]) * .1)
ithrow["HR_CI_high"] = math.exp(float(adjtab.loc[adjtab.Variable == "ITHscore_3D", "Beta_CI_high"].iloc[0]) * .1)
ithrow["Scale_note"] = "per 0.1 increase"
adjtab = pd.concat([adjtab, ithrow.to_frame().T], ignore_index=True)
adjtab.to_csv(OUT / "Table_Adjusted_Cox_ITH_Tstage_Volume.csv", index=False)


clinical = ["NLR", "Ki67", "Margin", "Myxoid"]
specs = {
    "Volume-clinical": clinical + ["log_Volume"],
    "ITH-clinical": clinical + ["ITHscore_3D"],
    "Volume-ITH-clinical": clinical + ["log_Volume", "ITHscore_3D"],
}
models, risks_train, risks_val = {}, {}, {}
for name, cols in specs.items():
    m, _ = fit_cox(train, cols)
    models[name] = m
    risks_train[name] = m.predict_log_partial_hazard(train[cols]).to_numpy()
    risks_val[name] = m.predict_log_partial_hazard(val[cols]).to_numpy()

# Verify refitted ITH-clinical reproduces the existing model/ranking.
verify_corr = stats.pearsonr(risks_train["ITH-clinical"], train.existing_Clinical_ITH_lp).statistic
verify_max_rank_diff = abs(pd.Series(risks_train["ITH-clinical"]).rank().to_numpy() - pd.Series(train.existing_Clinical_ITH_lp).rank().to_numpy()).max()


def cidx(d, risk):
    return concordance_index(d.time, -np.asarray(risk), d.event)


point_c = {k: cidx(train, v) for k, v in risks_train.items()}
point_val = {k: cidx(val, v) for k, v in risks_val.items()}
boot_rows, failed = [], 0
for b in range(1000):
    ix = RNG.integers(0, len(train), len(train))
    db = train.iloc[ix].reset_index(drop=True)
    if db.event.sum() == 0:
        failed += 1; continue
    row = {"Iteration": b + 1}
    try:
        for name, cols in specs.items():
            m, _ = fit_cox(db, cols)
            lp = m.predict_log_partial_hazard(db[cols]).to_numpy()
            row[name] = cidx(db, lp)
        row["Delta_ITH_minus_Volume"] = row["ITH-clinical"] - row["Volume-clinical"]
        boot_rows.append(row)
    except Exception:
        failed += 1
boot = pd.DataFrame(boot_rows)
boot.to_csv(OUT / "Bootstrap_Model_Cindex_1000_iterations.csv", index=False)

perf = []
for name in specs:
    lo, hi = boot[name].quantile([.025, .975])
    perf.append({"Model": name, "Training_Cindex": point_c[name], "Training_bootstrap_CI95_low": lo, "Training_bootstrap_CI95_high": hi,
                 "Validation_Cindex_descriptive": point_val[name], "Validation_inference": "Not performed (n=22, events=6)"})
perf = pd.DataFrame(perf)
perf.to_csv(OUT / "Table_Model_Performance.csv", index=False)

delta_point = point_c["ITH-clinical"] - point_c["Volume-clinical"]
delta_lo, delta_hi = boot.Delta_ITH_minus_Volume.quantile([.025, .975])
simple = models["Volume-clinical"]
full = models["Volume-ITH-clinical"]
lrt_chi2 = 2 * (full.log_likelihood_ - simple.log_likelihood_)
lrt_p = stats.chi2.sf(lrt_chi2, 1)
lrt = pd.DataFrame([{
    "Comparison": "Volume-clinical vs Volume-ITH-clinical", "Added_predictor": "ITHscore_3D", "df": 1,
    "LRT_chi_square": lrt_chi2, "LRT_P_value": lrt_p,
    "Delta_C_ITHclinical_minus_Volumeclinical": delta_point,
    "Delta_C_bootstrap_CI95_low": delta_lo, "Delta_C_bootstrap_CI95_high": delta_hi,
    "Incremental_value_conclusion": "Yes: statistically significant by LRT" if lrt_p < .05 else "No statistically significant incremental value by LRT"
}])
lrt.to_csv(OUT / "Table_Incremental_Value_LRT_and_DeltaC.csv", index=False)

# Volume strata: choose tertiles if all event counts >=15; otherwise median split.
q1, q2 = train.Volume_cm3.quantile([1/3, 2/3])
train["Volume_stratum_tertile"] = pd.cut(train.Volume_cm3, [-np.inf, q1, q2, np.inf], labels=["Low", "Medium", "High"], include_lowest=True)
events_tert = train.groupby("Volume_stratum_tertile", observed=False).event.agg(["size", "sum"]).rename(columns={"size": "N", "sum": "Events"})
if (events_tert.Events < 15).any():
    med = train.Volume_cm3.median()
    train["Volume_stratum"] = pd.cut(train.Volume_cm3, [-np.inf, med, np.inf], labels=["Low", "High"], include_lowest=True)
    strat_method = "Training median split because at least one tertile had <15 events"
else:
    train["Volume_stratum"] = train.Volume_stratum_tertile
    strat_method = "Training tertiles; all strata had >=15 events"
events_tert.reset_index().to_csv(OUT / "Table_Tertile_Precheck.csv", index=False)

subrows = []
fig, axes = plt.subplots(1, train.Volume_stratum.nunique(), figsize=(5.3 * train.Volume_stratum.nunique(), 4.8), sharey=True)
if not isinstance(axes, np.ndarray): axes = np.array([axes])
for ax, (stratum, d) in zip(axes, train.groupby("Volume_stratum", observed=True)):
    cut = d.ITHscore_3D.median()
    d = d.copy(); d["ITH_group"] = np.where(d.ITHscore_3D > cut, "High ITH", "Low ITH")
    m, _ = fit_cox(d.assign(ITH_high=(d.ITH_group == "High ITH").astype(int)), ["ITH_high"])
    s = m.summary.loc["ITH_high"]
    lr = logrank_test(d.loc[d.ITH_group == "High ITH", "time"], d.loc[d.ITH_group == "Low ITH", "time"],
                      event_observed_A=d.loc[d.ITH_group == "High ITH", "event"], event_observed_B=d.loc[d.ITH_group == "Low ITH", "event"])
    subrows.append({"Volume_stratum": str(stratum), "N": len(d), "Events": int(d.event.sum()), "Within_stratum_ITH_median_cut": cut,
                    "HR_high_vs_low_ITH": s["exp(coef)"], "HR_CI95_low": s["exp(coef) lower 95%"], "HR_CI95_high": s["exp(coef) upper 95%"],
                    "Cox_P_value": s["p"], "Logrank_P_value": lr.p_value,
                    "Caution": "Events <15: descriptive only, avoid overinterpretation" if d.event.sum() < 15 else ""})
    for group, color in [("High ITH", "#E64B35"), ("Low ITH", "#4DBBD5")]:
        g = d[d.ITH_group == group]
        km = KaplanMeierFitter().fit(g.time, g.event, label=f"{group} (n={len(g)})")
        km.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2)
    ax.set_title(f"{stratum} volume\nEvents={int(d.event.sum())}, log-rank P={lr.p_value:.3g}")
    ax.set_xlim(0, 60); ax.set_ylim(0, 1.02); ax.set_xlabel("Time (months)")
axes[0].set_ylabel("Overall survival probability")
fig.suptitle("Training cohort: within-volume-stratum 3D-ITHscore risk groups", y=1.03)
fig.tight_layout(); fig.savefig(OUT / "Fig_KM_ITH_within_Volume_Strata.pdf", bbox_inches="tight"); fig.savefig(OUT / "Fig_KM_ITH_within_Volume_Strata.png", dpi=300, bbox_inches="tight"); plt.close(fig)
sub = pd.DataFrame(subrows)

# Interaction: continuous ITH (per 0.1) x categorical volume stratum, global 2-df LRT.
it = train[["time", "event", "ITHscore_3D", "Volume_stratum"]].copy()
it["ITH_std"] = (it.ITHscore_3D - it.ITHscore_3D.mean()) / it.ITHscore_3D.std(ddof=0)
dummies = pd.get_dummies(it.Volume_stratum, prefix="Volume", drop_first=True, dtype=float)
it = pd.concat([it.drop(columns=["Volume_stratum", "ITHscore_3D"]), dummies], axis=1)
main_cols = ["ITH_std"] + dummies.columns.tolist()
for c in dummies.columns: it[f"ITH_x_{c}"] = it["ITH_std"] * it[c]
int_cols = [c for c in it if c.startswith("ITH_x_")]
m0, _ = fit_cox(it, main_cols)
m1, _ = fit_cox(it, main_cols + int_cols)
int_chi = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
int_p = stats.chi2.sf(int_chi, len(int_cols))
sub["Interaction_global_LRT_P"] = int_p
sub["Stratification_method"] = strat_method
sub.to_csv(OUT / "Table_Volume_Stratified_ITH_Effects.csv", index=False)
pd.DataFrame([{"Test": "ITHscore_3D (continuous, standardized for numerical stability) x volume stratum global interaction", "Chi_square": int_chi, "df": len(int_cols), "P_value": int_p, "Reduced_log_likelihood": m0.log_likelihood_, "Full_log_likelihood": m1.log_likelihood_, "Method": strat_method}]).to_csv(OUT / "Table_Interaction_Test.csv", index=False)

# Forest plot.
fig, ax = plt.subplots(figsize=(6.8, 4.6))
y = np.arange(len(sub))[::-1]
hr = sub.HR_high_vs_low_ITH.to_numpy(float); lo = sub.HR_CI95_low.to_numpy(float); hi = sub.HR_CI95_high.to_numpy(float)
ax.errorbar(hr, y, xerr=[hr-lo, hi-hr], fmt="o", color="#3C5488", ecolor="#3C5488", capsize=4, markersize=7)
ax.axvline(1, color="gray", linestyle="--")
ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels([f"{s} volume (N={n}, events={e})" for s,n,e in zip(sub.Volume_stratum, sub.N, sub.Events)])
ax.set_xlabel("Hazard ratio: high vs low 3D-ITHscore (95% CI)")
ax.set_title(f"Training cohort volume-stratified effects\nGlobal interaction P={int_p:.3g}")
fig.tight_layout(); fig.savefig(OUT / "Fig_Forest_Volume_Stratified_ITH.pdf", bbox_inches="tight"); fig.savefig(OUT / "Fig_Forest_Volume_Stratified_ITH.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# Supporting figures.
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(6, 5))
sns.regplot(data=train, x="log_Volume", y="ITHscore_3D", lowess=True, scatter_kws={"s": 30, "alpha": .75}, line_kws={"color": "#E64B35"}, ax=ax)
rr = corr[(corr.Cohort == "Training") & (corr.Variables == "ITHscore_3D vs log(Volume_cm3)")].iloc[0]
ax.set_title(f"Training cohort: 3D-ITHscore vs log(volume)\nSpearman rho={rr.Estimate:.3f}, 95% CI {rr.CI95_low:.3f} to {rr.CI95_high:.3f}, P={rr.P_value:.3g}")
fig.tight_layout(); fig.savefig(OUT / "Fig_ITH_vs_LogVolume.pdf", bbox_inches="tight"); fig.savefig(OUT / "Fig_ITH_vs_LogVolume.png", dpi=300, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.8))
ax.errorbar(perf.Training_Cindex, np.arange(3), xerr=[perf.Training_Cindex-perf.Training_bootstrap_CI95_low, perf.Training_bootstrap_CI95_high-perf.Training_Cindex], fmt="o", color="#00A087", capsize=4)
ax.set_yticks(np.arange(3)); ax.set_yticklabels(perf.Model); ax.set_xlim(.45, 1); ax.set_xlabel("Training Harrell C-index (bootstrap 95% CI)"); ax.set_title("Volume and 3D-ITHscore clinical model comparison")
fig.tight_layout(); fig.savefig(OUT / "Fig_Model_Cindex_Comparison.pdf", bbox_inches="tight"); fig.savefig(OUT / "Fig_Model_Cindex_Comparison.png", dpi=300, bbox_inches="tight"); plt.close(fig)

report = f"""3D-ITHscore and tumor volume analysis report
================================================
Seed: {SEED}
Primary inference: training cohort only (N={len(train)}, events={int(train.event.sum())}).
Validation cohort: descriptive direction only (N={len(val)}, events={int(val.event.sum())}); no validation P values calculated.

Volume provenance
-----------------
Tumor volume was obtained exclusively from unprocessed PyRadiomics raw extraction tables:
T1_original_shape_MeshVolume (mm3), converted to cm3 by division by 1000, then natural-log transformed.
The T1 and T2 MeshVolume columns matched exactly for every patient. No ComBat-harmonized or z-standardized feature was used.

Correlation analyses
--------------------
{corr[corr.Cohort=='Training'].to_string(index=False)}

Adjusted Cox model (training)
-----------------------------
Model: OS ~ ITHscore_3D + ordinal T stage + log(volume cm3).
{adjtab[['Variable','HR','HR_CI_low','HR_CI_high','P_value','Scale_note']].to_string(index=False)}

Incremental value
-----------------
Clinical variables were exactly those used in the seven-model comparison: NLR, Ki67, Margin, and Myxoid.
Training-only imputation values: {json.dumps(impute)}.
The existing ITH-clinical model was reproducibly refitted from the same specification for paired bootstrap; its training LP Pearson correlation with the saved existing LP was {verify_corr:.12f}; maximum rank difference was {verify_max_rank_diff:.0f}.
{perf.to_string(index=False)}
Paired bootstrap delta C (ITH-clinical minus Volume-clinical): {delta_point:.4f} (95% CI {delta_lo:.4f} to {delta_hi:.4f}).
LRT adding 3D-ITHscore to Volume-clinical: chi-square={lrt_chi2:.4f}, df=1, P={lrt_p:.6g}.
Direct answer: {lrt.iloc[0].Incremental_value_conclusion}.

Volume-stratified analysis
--------------------------
Method: {strat_method}. Within each stratum, 3D-ITHscore was dichotomized at that stratum's own median.
The global interaction test used standardized continuous ITHscore by categorical volume stratum (standardization changes numerical conditioning, not the interaction test): P={int_p:.6g}.
{sub.to_string(index=False)}

Interpretive limits
-------------------
These analyses support association and incremental prognostic information, not causality or clinical utility. The validation cohort is small and used descriptively only. Data-driven within-stratum median cutoffs and subgroup estimates are sensitivity analyses and should not be treated as independently validated thresholds.
Bootstrap C-index intervals are standard case-resampling intervals from 1000 training-set refits. Correlation CIs use 2000 case-resampling bootstrap replicates.
"""
(OUT / "完整分析报告.txt").write_text(report, encoding="utf-8")

methods = f"""Methods for reviewer response
Tumor volume was extracted from the unprocessed PyRadiomics shape feature T1_original_shape_MeshVolume (mm3), before ComBat harmonization or z-score standardization; the identical T2-derived value was used as a cross-check. Volume was converted to cm3 and natural-log transformed. All inferential analyses were prespecified in the training cohort (n=129, 68 deaths), with the validation cohort (n=22, 6 deaths) restricted to descriptive assessment. Spearman correlations and 95% percentile confidence intervals (2,000 bootstrap resamples) quantified associations of 3D-ITHscore with log-volume, ordinal T stage, and ordinal overall TNM stage. Partial Spearman correlations were obtained by correlating residuals of ranked variables after adjustment for ranked log-volume. An adjusted Cox model included continuous 3D-ITHscore, ordinal T stage, and log-volume. Incremental value was assessed using models containing the same clinical predictors as the prior seven-model comparison (NLR, Ki-67, margin, and myxoid change), standard 1,000-resample bootstrap intervals for Harrell's C-index, a paired bootstrap interval for the C-index difference, and a one-degree-of-freedom likelihood-ratio test comparing Volume-clinical with Volume-ITH-clinical. Volume strata were selected from training-cohort tertiles if every stratum contained at least 15 deaths; otherwise a median split was used. Within each stratum, 3D-ITHscore was dichotomized at the stratum-specific median for Kaplan-Meier and univariable Cox analyses. Effect modification was evaluated by a global likelihood-ratio test for standardized continuous 3D-ITHscore-by-volume-stratum interaction; standardization was used only for numerical conditioning and does not alter the test. Validation-cohort P values were not calculated.
"""
(OUT / "Methods_for_Reviewer_Response.txt").write_text(methods, encoding="utf-8")

readme = pd.DataFrame([
    ["Table_Volume_Extraction_QC.csv", "Raw volume provenance and cohort QC"],
    ["Table_Correlation_PartialCorrelation.csv", "Spearman and volume-adjusted partial Spearman results"],
    ["Table_Adjusted_Cox_ITH_Tstage_Volume.csv", "Adjusted Cox model"],
    ["Table_Model_Performance.csv", "Three-model Harrell C-index comparison"],
    ["Table_Incremental_Value_LRT_and_DeltaC.csv", "Core LRT and paired delta-C evidence"],
    ["Bootstrap_Model_Cindex_1000_iterations.csv", "All valid training bootstrap iterations"],
    ["Table_Tertile_Precheck.csv", "Volume-stratum event-count decision"],
    ["Table_Volume_Stratified_ITH_Effects.csv", "Within-stratum HR and log-rank results"],
    ["Table_Interaction_Test.csv", "Global interaction test"],
    ["Fig_KM_ITH_within_Volume_Strata.pdf/png", "KM sensitivity analysis"],
    ["Fig_Forest_Volume_Stratified_ITH.pdf/png", "Stratified forest plot"],
    ["Fig_ITH_vs_LogVolume.pdf/png", "Volume-score association"],
    ["Fig_Model_Cindex_Comparison.pdf/png", "Model discrimination comparison"],
    ["完整分析报告.txt", "Full reproducible numerical report"],
    ["Methods_for_Reviewer_Response.txt", "Manuscript-ready methods paragraph"],
], columns=["File", "Description"])
readme.to_csv(OUT / "Output_File_Index.csv", index=False)

# Reproducibility metadata and MD5 after all outputs exist.
meta = {"seed": SEED, "training_N": len(train), "training_events": int(train.event.sum()), "validation_N": len(val), "validation_events": int(val.event.sum()),
        "bootstrap_correlation": 2000, "bootstrap_cindex": 1000, "bootstrap_failed_cindex": failed,
        "volume_raw_column": "T1_original_shape_MeshVolume", "volume_transform": "ln(mm3/1000)",
        "existing_Clinical_ITH_LP_Pearson_r": verify_corr, "existing_Clinical_ITH_max_rank_difference": float(verify_max_rank_diff)}
(OUT / "Reproducibility_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

manifest = []
for f in sorted(OUT.iterdir()):
    if f.is_file() and f.name != "MD5_manifest.csv":
        manifest.append({"File": f.name, "Bytes": f.stat().st_size, "MD5": hashlib.md5(f.read_bytes()).hexdigest()})
pd.DataFrame(manifest).to_csv(OUT / "MD5_manifest.csv", index=False)

print(report)
print(f"OUTPUT_DIR={OUT}")
