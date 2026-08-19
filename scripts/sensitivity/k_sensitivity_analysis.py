from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.multitest import multipletests

SEED = 2024
RNG = np.random.default_rng(SEED)
INFILE = Path("/root/autodl-tmp/HCR_input/all_patients_ith_K2_K10_corrected.csv")
OUT = Path("/root/autodl-tmp/HCR_runs/k_sensitivity_20260819")
OUT.mkdir(parents=True, exist_ok=True)
KS = list(range(2, 11))
KCOLS = [f"ith_{k}clusters" for k in KS]


def read_data():
    d = pd.read_csv(INFILE, encoding="gb18030")
    required = ["patient_id", "dataset", "OS/label", "OS/m"] + KCOLS
    missing = [c for c in required if c not in d]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    if len(d) != 151 or d.patient_id.duplicated().any():
        raise RuntimeError(f"Expected 151 unique patients; got rows={len(d)}, unique IDs={d.patient_id.nunique()}")
    d = d.rename(columns={"OS/label": "event", "OS/m": "time"})
    d["cohort"] = d.dataset.astype(str).str.lower().map({"train": "Training", "training": "Training", "test": "Validation", "validation": "Validation"})
    if d.cohort.isna().any():
        raise RuntimeError(f"Unrecognized dataset labels: {d.loc[d.cohort.isna(),'dataset'].unique()}")
    for c in KCOLS + ["event", "time"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


d = read_data()
train = d[d.cohort == "Training"].copy()
val = d[d.cohort == "Validation"].copy()
if (len(train), int(train.event.sum()), len(val), int(val.event.sum())) != (129, 68, 22, 6):
    raise RuntimeError(f"Unexpected cohort counts: train {len(train)}/{train.event.sum()} events; validation {len(val)}/{val.event.sum()} events")

missing_rows = []
for cohort, z in [("All", d), ("Training", train), ("Validation", val)]:
    for k, c in zip(KS, KCOLS):
        missing_rows.append({"Cohort": cohort, "K": k, "N_total": len(z), "N_missing_score": int(z[c].isna().sum()), "N_valid_score": int(z[c].notna().sum()),
                             "Handling": "Pairwise complete; no imputation"})
missing = pd.DataFrame(missing_rows)
missing.to_csv(OUT / "Table_SK0_Missingness_QC.csv", index=False)

# Part 1: full-cohort descriptive score stability.
rho = d[KCOLS].corr(method="spearman", min_periods=2)
rho.index = [f"K={k}" for k in KS]; rho.columns = [f"K={k}" for k in KS]
rho.to_csv(OUT / "Table_S-K1a_Spearman_Matrix.csv")

pairs = []
for i, k1 in enumerate(KS):
    for k2 in KS[i+1:]:
        x, y = d[f"ith_{k1}clusters"], d[f"ith_{k2}clusters"]
        ok = x.notna() & y.notna()
        r, p = stats.spearmanr(x[ok], y[ok])
        pairs.append({"K1": k1, "K2": k2, "N_pairwise": int(ok.sum()), "Spearman_rho": r, "P_value_descriptive": p,
                      "K3_comparison": k1 == 3 or k2 == 3, "Adjacent_K": k2 == k1 + 1})
pairs = pd.DataFrame(pairs)
pairs.to_csv(OUT / "Table_S-K1b_Spearman_Pairs.csv", index=False)

def kappa_strength(x):
    if x < .20: return "Slight/poor (<0.20)"
    if x <= .40: return "Fair/weak (0.21-0.40)"
    if x <= .60: return "Moderate (0.41-0.60)"
    if x <= .80: return "Substantial (0.61-0.80)"
    return "Almost perfect/very strong (>0.80)"

kappa_rows = []
base = train["ith_3clusters"]
base_cut = base.median(skipna=True)
for k in KS:
    if k == 3: continue
    s = train[f"ith_{k}clusters"]
    ok = base.notna() & s.notna()
    cut = s[ok].median()
    g3 = (base[ok] > base_cut).astype(int)
    gk = (s[ok] > cut).astype(int)
    kap = cohen_kappa_score(g3, gk)
    kappa_rows.append({"Comparison": f"K=3 vs K={k}", "K_other": k, "N_pairwise_training": int(ok.sum()), "K3_median": base_cut,
                       "K_other_median": cut, "Kappa": kap, "Agreement_strength": kappa_strength(kap),
                       "K3_high_N": int(g3.sum()), "K_other_high_N": int(gk.sum()), "Percent_agreement": (g3 == gk).mean() * 100})
kappa = pd.DataFrame(kappa_rows)
kappa.to_csv(OUT / "Table_S-K1c_Kappa_K3_vs_Other.csv", index=False)

offdiag = pairs.Spearman_rho
summary_stability = pd.DataFrame([{
    "All_36_pairs_min_rho": offdiag.min(), "All_36_pairs_median_rho": offdiag.median(), "All_36_pairs_max_rho": offdiag.max(),
    "K3_vs_others_min_rho": pairs.loc[pairs.K3_comparison, "Spearman_rho"].min(),
    "K3_vs_others_median_rho": pairs.loc[pairs.K3_comparison, "Spearman_rho"].median(),
    "K3_vs_others_max_rho": pairs.loc[pairs.K3_comparison, "Spearman_rho"].max(),
    "Adjacent_pairs_min_rho": pairs.loc[pairs.Adjacent_K, "Spearman_rho"].min(),
    "Adjacent_pairs_median_rho": pairs.loc[pairs.Adjacent_K, "Spearman_rho"].median(),
    "Adjacent_pairs_max_rho": pairs.loc[pairs.Adjacent_K, "Spearman_rho"].max(),
}])
summary_stability.to_csv(OUT / "Table_S-K1d_Stability_Summary.csv", index=False)

# Figure S-K1: Li et al.-style annotated heatmap; K=3 highlighted.
sns.set_theme(style="white", context="talk")
fig, ax = plt.subplots(figsize=(10.5, 9))
sns.heatmap(rho, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, square=True, linewidths=.6, cbar_kws={"label": "Spearman rho"}, ax=ax)
ax.set_title("Cross-K stability of 3D-ITHscore (all patients, N=151)", pad=18)
for tick in ax.get_xticklabels():
    if tick.get_text() == "K=3": tick.set_color("#D73027"); tick.set_fontweight("bold")
for tick in ax.get_yticklabels():
    if tick.get_text() == "K=3": tick.set_color("#D73027"); tick.set_fontweight("bold")
ax.add_patch(plt.Rectangle((1, 0), 1, 9, fill=False, edgecolor="#D73027", lw=2.5))
ax.add_patch(plt.Rectangle((0, 1), 9, 1, fill=False, edgecolor="#D73027", lw=2.5))
fig.tight_layout(); fig.savefig(OUT / "Figure_S-K1_Spearman_Heatmap.pdf", bbox_inches="tight"); fig.savefig(OUT / "Figure_S-K1_Spearman_Heatmap.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fit_cox(z, score_col, standardized=False):
    a = z[["time", "event", score_col]].dropna().copy()
    name = "score_z" if standardized else "score_raw"
    if standardized:
        sd = a[score_col].std(ddof=0)
        if not np.isfinite(sd) or sd == 0: raise RuntimeError(f"Zero SD for {score_col}")
        a[name] = (a[score_col] - a[score_col].mean()) / sd
    else:
        a[name] = a[score_col]
    m = CoxPHFitter(penalizer=1e-9).fit(a[["time", "event", name]], duration_col="time", event_col="event")
    return m, a, name


def cindex_from_model(m, a, name):
    lp = m.predict_log_partial_hazard(a[[name]]).to_numpy()
    return concordance_index(a.time, -lp, a.event), lp


cox_rows, boot_rows, km_rows = [], [], []
for k, c in zip(KS, KCOLS):
    m_raw, a_raw, n_raw = fit_cox(train, c, False)
    m_z, a_z, n_z = fit_cox(train, c, True)
    sr = m_raw.summary.loc[n_raw]; sz = m_z.summary.loc[n_z]
    cpt, lp = cindex_from_model(m_raw, a_raw, n_raw)
    boots = []
    for b in range(1000):
        ix = RNG.integers(0, len(a_raw), len(a_raw))
        bb = a_raw.iloc[ix]
        if bb.event.sum() == 0: continue
        try: boots.append(concordance_index(bb.time, -lp[ix], bb.event))
        except Exception: pass
    lo, hi = np.quantile(boots, [.025, .975])

    # Validation is descriptive only: same fitted direction/association metrics, no inferential claim.
    mv, av, nv = fit_cox(val, c, False)
    mvz, avz, nvz = fit_cox(val, c, True)
    svr = mv.summary.loc[nv]; svz = mvz.summary.loc[nvz]
    cv, _ = cindex_from_model(mv, av, nv)

    cox_rows.append({"K": k, "Training_N": len(a_raw), "Training_events": int(a_raw.event.sum()),
                     "HR_raw": sr["exp(coef)"], "HR_raw_CI95_low": sr["exp(coef) lower 95%"], "HR_raw_CI95_high": sr["exp(coef) upper 95%"], "Wald_P_raw": sr["p"],
                     "HR_per_1SD": sz["exp(coef)"], "HR_1SD_CI95_low": sz["exp(coef) lower 95%"], "HR_1SD_CI95_high": sz["exp(coef) upper 95%"], "Wald_P_1SD": sz["p"],
                     "Harrell_C": cpt, "Harrell_C_bootstrap_CI95_low": lo, "Harrell_C_bootstrap_CI95_high": hi, "C_bootstrap_valid": len(boots),
                     "Validation_N": len(av), "Validation_events": int(av.event.sum()), "Validation_HR_raw_descriptive": svr["exp(coef)"],
                     "Validation_HR_per_1SD_descriptive": svz["exp(coef)"], "Validation_Harrell_C_descriptive": cv,
                     "Validation_note": "Limited sample (n=22, 6 events); trend only, not significance inference"})
    for bidx, bv in enumerate(boots, 1): boot_rows.append({"K": k, "Iteration": bidx, "Harrell_C": bv})

    # Median split separately within each cohort/K.
    for cohort, z in [("Training", train), ("Validation", val)]:
        aa = z[["time", "event", c]].dropna().copy(); cut = aa[c].median(); aa["high"] = (aa[c] > cut).astype(int)
        if aa.high.nunique() < 2:
            km_rows.append({"K": k, "Cohort": cohort, "N": len(aa), "Events": int(aa.event.sum()), "Median_cut": cut, "High_N": int(aa.high.sum()), "Low_N": int((1-aa.high).sum()),
                            "HR_high_vs_low": np.nan, "HR_CI95_low": np.nan, "HR_CI95_high": np.nan, "Cox_P": np.nan, "Logrank_P": np.nan,
                            "Inference_role": "Primary inference" if cohort == "Training" else "Descriptive only", "Issue": "Median split yielded one group"})
            continue
        mm = CoxPHFitter(penalizer=1e-9).fit(aa[["time", "event", "high"]], duration_col="time", event_col="event")
        sm = mm.summary.loc["high"]
        lr = logrank_test(aa.loc[aa.high==1,"time"], aa.loc[aa.high==0,"time"], event_observed_A=aa.loc[aa.high==1,"event"], event_observed_B=aa.loc[aa.high==0,"event"])
        km_rows.append({"K": k, "Cohort": cohort, "N": len(aa), "Events": int(aa.event.sum()), "Median_cut": cut, "High_N": int(aa.high.sum()), "Low_N": int((1-aa.high).sum()),
                        "HR_high_vs_low": sm["exp(coef)"], "HR_CI95_low": sm["exp(coef) lower 95%"], "HR_CI95_high": sm["exp(coef) upper 95%"], "Cox_P": sm["p"], "Logrank_P": lr.p_value,
                        "Inference_role": "Primary inference" if cohort == "Training" else "Descriptive only; not used for significance inference", "Issue": ""})

cox = pd.DataFrame(cox_rows)
cox["Wald_P_FDR_BH"] = multipletests(cox.Wald_P_raw, method="fdr_bh")[1]
cox["FDR_significant"] = np.where(cox.Wald_P_FDR_BH < .05, "Yes", "No")
cox["Direction_training"] = np.where(cox.HR_raw > 1, "Higher score = higher hazard", "Higher score = lower hazard")
cox["Direction_validation"] = np.where(cox.Validation_HR_raw_descriptive > 1, "Higher score = higher hazard", "Higher score = lower hazard")
cox.to_csv(OUT / "Table_S-K2_Cox_Cindex_AllK.csv", index=False)
pd.DataFrame(boot_rows).to_csv(OUT / "Bootstrap_Cindex_1000_AllK.csv", index=False)
km = pd.DataFrame(km_rows)
km.loc[km.Cohort == "Training", "Logrank_P_FDR_BH"] = multipletests(km.loc[km.Cohort == "Training", "Logrank_P"], method="fdr_bh")[1]
km.to_csv(OUT / "Table_S-K3_KM_MedianSplit_AllK.csv", index=False)

# Figure S-K2: aligned panels for P and C; K=3 red, all K shown.
trkm = km[km.Cohort == "Training"].sort_values("K")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.3))
colors = ["#D73027" if k == 3 else "#3C5488" for k in KS]
ax1.plot(trkm.K, -np.log10(trkm.Logrank_P), color="#3C5488", lw=1.8, zorder=1)
ax1.scatter(trkm.K, -np.log10(trkm.Logrank_P), c=colors, s=[100 if k==3 else 55 for k in KS], zorder=2)
ax1.axhline(-np.log10(.05), ls="--", color="gray", label="P=0.05")
ax1.set_xticks(KS); ax1.set_xlabel("Number of clusters (K)"); ax1.set_ylabel("-log10(log-rank P)"); ax1.set_title("Median-split survival separation"); ax1.legend(frameon=False)
ax2.errorbar(cox.K, cox.Harrell_C, yerr=[cox.Harrell_C-cox.Harrell_C_bootstrap_CI95_low, cox.Harrell_C_bootstrap_CI95_high-cox.Harrell_C], fmt="none", ecolor="#3C5488", capsize=3, lw=1.5)
ax2.plot(cox.K, cox.Harrell_C, color="#3C5488", lw=1.8, zorder=1)
ax2.scatter(cox.K, cox.Harrell_C, c=colors, s=[100 if k==3 else 55 for k in KS], zorder=2)
ax2.set_xticks(KS); ax2.set_xlabel("Number of clusters (K)"); ax2.set_ylabel("Harrell C-index (95% bootstrap CI)"); ax2.set_title("Continuous-score discrimination")
for ax in (ax1, ax2): ax.annotate("K=3 (preselected)", xy=(3, (trkm.loc[trkm.K==3, '-logp'].iloc[0] if '-logp' in trkm else -np.log10(trkm.loc[trkm.K==3,'Logrank_P'].iloc[0])) if ax is ax1 else cox.loc[cox.K==3,'Harrell_C'].iloc[0]), xytext=(3.7, ax.get_ylim()[1]*.88 if ax is ax1 else ax.get_ylim()[1]-.015), color="#D73027", arrowprops={"arrowstyle":"->","color":"#D73027"}, fontsize=11)
fig.suptitle("Post-selection sensitivity analysis across K=2-10 (training cohort)", y=1.02)
fig.tight_layout(); fig.savefig(OUT / "Figure_S-K2_Pvalue_Cindex_Trend.pdf", bbox_inches="tight"); fig.savefig(OUT / "Figure_S-K2_Pvalue_Cindex_Trend.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# Compact all-K KM result: forest-style summary avoids nine separate KM figures.
fig, ax = plt.subplots(figsize=(8.5, 7))
yy = np.arange(len(trkm))[::-1]
hr = trkm.HR_high_vs_low.to_numpy(float); hlo = trkm.HR_CI95_low.to_numpy(float); hhi = trkm.HR_CI95_high.to_numpy(float)
for j, (k, x, l, h, y) in enumerate(zip(trkm.K, hr, hlo, hhi, yy)):
    col = "#D73027" if k == 3 else "#3C5488"
    ax.errorbar(x, y, xerr=[[x-l],[h-x]], fmt="*" if k==3 else "o", color=col, ecolor=col, capsize=3, markersize=11 if k==3 else 7)
ax.axvline(1, color="gray", ls="--"); ax.set_xscale("log"); ax.set_yticks(yy); ax.set_yticklabels([f"K={k}  (P={p:.2g})" for k,p in zip(trkm.K,trkm.Logrank_P)])
ax.set_xlabel("HR for high vs low 3D-ITHscore (95% CI)"); ax.set_title("Compact summary of nine median-split KM analyses\nTraining cohort; K=3 highlighted")
fig.tight_layout(); fig.savefig(OUT / "Figure_S-K3_Compact_KM_Summary.pdf", bbox_inches="tight"); fig.savefig(OUT / "Figure_S-K3_Compact_KM_Summary.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# One representative K=3 KM curve; all other K results remain fully tabulated and summarized above.
aa = train[["time", "event", "ith_3clusters"]].dropna().copy(); cut3 = aa.ith_3clusters.median(); aa["group"] = np.where(aa.ith_3clusters > cut3, "High ITH", "Low ITH")
fig, ax = plt.subplots(figsize=(7, 5.5))
for label, color in [("High ITH", "#E64B35"), ("Low ITH", "#4DBBD5")]:
    g=aa[aa.group==label]; KaplanMeierFitter().fit(g.time,g.event,label=f"{label} (n={len(g)})").plot_survival_function(ax=ax,ci_show=True,color=color,lw=2.2)
p3=trkm.loc[trkm.K==3,'Logrank_P'].iloc[0]; hr3=trkm.loc[trkm.K==3,'HR_high_vs_low'].iloc[0]; l3=trkm.loc[trkm.K==3,'HR_CI95_low'].iloc[0]; h3=trkm.loc[trkm.K==3,'HR_CI95_high'].iloc[0]
ax.set_xlim(0,60); ax.set_ylim(0,1.02); ax.set_xlabel("Time (months)"); ax.set_ylabel("Overall survival probability"); ax.set_title(f"Preselected K=3: training cohort\nHR={hr3:.2f} ({l3:.2f}-{h3:.2f}), log-rank P={p3:.3g}")
fig.tight_layout(); fig.savefig(OUT / "Figure_S-K4_K3_Representative_KM.pdf", bbox_inches="tight"); fig.savefig(OUT / "Figure_S-K4_K3_Representative_KM.png", dpi=300, bbox_inches="tight"); plt.close(fig)

direction_all_train = bool((cox.HR_raw > 1).all())
direction_all_val = bool((cox.Validation_HR_raw_descriptive > 1).all())
best_c_k = int(cox.loc[cox.Harrell_C.idxmax(), "K"])
best_c = float(cox.Harrell_C.max())
k3c = cox.loc[cox.K==3].iloc[0]
non_k3_note = (f"K={best_c_k} had the highest training C-index ({best_c:.3f}); this post-selection observation does not alter the prespecified K=3 model." if best_c_k != 3 else "K=3 had the highest training C-index; this was not used to select K.")

summary_cn = f"""K值稳健性与信息泄露排查分析总结
==================================

重要时间与逻辑顺序
------------------
正式K=3在任何OS分析之前已经确定。第一步，在仅使用体素级影像特征、完全不使用临床或生存结局的无监督分析中，通过Calinski-Harabasz指数在K=2-8范围内最大化确定K=3；第二步，在不使用生存结局的横断面病理分类关联消融分析中，K=3同样表现最佳（Supplementary Table S3）。本次K=2-10分析发生在K值确定之后，严格属于事后敏感性验证，其目的仅为评价评分与预后结论对K值变化的稳健性，不用于重新选择或优化K。

样本与缺失
----------
评分稳定性描述使用全队列151例。Cox和log-rank推断仅使用训练集129例、68个死亡事件。验证集22例、6个事件仅作方向性描述，不用于显著性推断。所有缺失评分均按pairwise complete处理，未插补。各K缺失详情见Table_SK0_Missingness_QC.csv。

评分跨K稳定性
-------------
全部36对Spearman相关系数范围为{offdiag.min():.3f}-{offdiag.max():.3f}，中位数{offdiag.median():.3f}。K=3与其他K的相关系数范围为{pairs.loc[pairs.K3_comparison,'Spearman_rho'].min():.3f}-{pairs.loc[pairs.K3_comparison,'Spearman_rho'].max():.3f}，中位数{pairs.loc[pairs.K3_comparison,'Spearman_rho'].median():.3f}。相邻K值相关系数范围为{pairs.loc[pairs.Adjacent_K,'Spearman_rho'].min():.3f}-{pairs.loc[pairs.Adjacent_K,'Spearman_rho'].max():.3f}，中位数{pairs.loc[pairs.Adjacent_K,'Spearman_rho'].median():.3f}。训练集K=3分组与其他K分组的Kappa范围为{kappa.Kappa.min():.3f}-{kappa.Kappa.max():.3f}。

训练集生存敏感性
----------------
K=2-10九种配置的连续评分HR方向{'全部一致（均为HR>1）' if direction_all_train else '并非全部一致，详见Table S-K2'}。K=3的标准化HR为{k3c.HR_per_1SD:.3f}（95%CI {k3c.HR_1SD_CI95_low:.3f}-{k3c.HR_1SD_CI95_high:.3f}），Harrell C-index为{k3c.Harrell_C:.3f}（95% bootstrap CI {k3c.Harrell_C_bootstrap_CI95_low:.3f}-{k3c.Harrell_C_bootstrap_CI95_high:.3f}）。九个Wald检验均完整列出并进行BH-FDR校正。{non_k3_note}

验证集描述
----------
验证集连续评分HR方向{'全部为HR>1' if direction_all_val else '存在方向不一致，详见Table S-K2'}；由于仅6个事件，所有验证结果仅用于趋势参考，不作显著性推断或模型选择。

结论
----
跨K的评分相关、风险分组一致性、连续Cox效应方向及KM分层结果共同用于评价敏感性。无论个别非K=3配置的点估计是否更高，均不得据此更改正式K=3，因为K=3已由结局盲法的无监督CH标准及非结局病理消融分析事先确定。本分析不构成新的K值优化过程，也不支持使用OS反向选择K。
"""
(OUT / "中文完整分析报告.txt").write_text(summary_cn, encoding="utf-8")

methods = f"""Methods and leakage-control statement
The choice of K=3 preceded all survival analyses. First, K=3 maximized the Calinski-Harabasz index over the prespecified image-only, unsupervised search range K=2-8, using voxel-level imaging features without clinical or outcome information. Second, a cross-sectional ablation analysis against non-outcome pathological categories (Supplementary Table S3) also favored K=3. The present OS analyses were conducted only after K had been fixed and were treated exclusively as post-selection sensitivity analyses, not as a procedure for selecting or optimizing K.

Score stability was described in all 151 patients using pairwise-complete Spearman correlations across K=2-10; no missing score was imputed. Following the strategy illustrated by Li et al., the full annotated correlation matrix was displayed as a heatmap. Training-cohort (n=129) median splits defined separately for each K were compared with the K=3 split using Cohen's kappa. Survival inference was restricted to the training cohort (68 deaths). For each K, univariable Cox models reported effects per raw score unit and per one standard deviation, with Wald tests and Benjamini-Hochberg correction across nine K values. Harrell's C-index confidence intervals used 1,000 case-resampling bootstrap replicates with seed 2024. Each K-specific median was used for Kaplan-Meier/log-rank analysis. The validation cohort (n=22, 6 deaths) was analyzed only descriptively and was not used for hypothesis testing, K selection, or model optimization.
"""
(OUT / "Methods_Leakage_Control_for_Reviewer.txt").write_text(methods, encoding="utf-8")

index = pd.DataFrame([
    ["Table_SK0_Missingness_QC.csv", "Missingness and actual N by K/cohort"],
    ["Table_S-K1a_Spearman_Matrix.csv", "Complete 9x9 Spearman matrix"],
    ["Table_S-K1b_Spearman_Pairs.csv", "All 36 pairwise results with K=3/adjacent flags"],
    ["Table_S-K1c_Kappa_K3_vs_Other.csv", "Training K=3 high/low agreement vs other K"],
    ["Table_S-K1d_Stability_Summary.csv", "Minimum/median/maximum rho summaries"],
    ["Table_S-K2_Cox_Cindex_AllK.csv", "All-K Cox, FDR, C-index and validation description"],
    ["Bootstrap_Cindex_1000_AllK.csv", "All 9,000 bootstrap C-index values"],
    ["Table_S-K3_KM_MedianSplit_AllK.csv", "All K-specific KM/log-rank and high-low Cox results"],
    ["Figure_S-K1_Spearman_Heatmap.pdf/png", "Annotated cross-K heatmap"],
    ["Figure_S-K2_Pvalue_Cindex_Trend.pdf/png", "All-K P/C-index sensitivity trend"],
    ["Figure_S-K3_Compact_KM_Summary.pdf/png", "Compact alternative to nine separate KM plots"],
    ["Figure_S-K4_K3_Representative_KM.pdf/png", "Representative prespecified K=3 KM curve"],
    ["中文完整分析报告.txt", "Numerical Chinese summary with leakage-control statement"],
    ["Methods_Leakage_Control_for_Reviewer.txt", "Reviewer-ready methods statement"],
    ["k_sensitivity_analysis.py", "Complete reproducible analysis code"],
], columns=["File", "Description"])
index.to_csv(OUT / "Output_File_Index.csv", index=False)

shutil.copy2(__file__, OUT / "k_sensitivity_analysis.py")
meta = {"seed": SEED, "all_N": len(d), "training_N": len(train), "training_events": int(train.event.sum()), "validation_N": len(val), "validation_events": int(val.event.sum()),
        "K_values": KS, "bootstrap_per_K": 1000, "survival_inference_cohort": "Training only", "validation_role": "Descriptive only",
        "K3_selection_timing": "Before OS analyses", "K3_selection_inputs": "Image-only unsupervised CH index, followed by non-outcome pathology ablation"}
(OUT / "Reproducibility_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

manifest=[]
for f in sorted(OUT.iterdir()):
    if f.is_file() and f.name != "MD5_manifest.csv": manifest.append({"File":f.name,"Bytes":f.stat().st_size,"MD5":hashlib.md5(f.read_bytes()).hexdigest()})
pd.DataFrame(manifest).to_csv(OUT / "MD5_manifest.csv", index=False)

print(summary_cn)
print("OUTPUT_DIR", OUT)
