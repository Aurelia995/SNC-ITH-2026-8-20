from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

SEED = 20250308
np.random.seed(SEED)
INPUT = Path("/root/autodl-tmp/HCR_input")
OUT = Path("/root/autodl-tmp/HCR_runs/ith_correlation_corrected_20260814")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    raise RuntimeError(f"Cannot decode {path}")


def norm_id(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.lower()
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"^c(?=\d+$)", "", regex=True)
            .str.replace(r"^0+(?=\d+$)", "", regex=True))


raw = read_csv_flexible(INPUT / "原始数据.csv")
ith_train = read_csv_flexible(INPUT / "ITH3_TRAIN.csv")
ith_test = read_csv_flexible(INPUT / "ITH3_TEST.csv")
ith = pd.concat([ith_train.assign(ith_source="training"),
                 ith_test.assign(ith_source="validation")], ignore_index=True)
raw["id_key"] = norm_id(raw["patient_id"])
ith["id_key"] = norm_id(ith["id"])
if raw.id_key.duplicated().any() or ith.id_key.duplicated().any():
    raise RuntimeError("Duplicate normalized patient IDs")
merged = raw.merge(ith[["id_key", "ith_3clusters", "ith_source"]],
                   on="id_key", how="inner", validate="one_to_one")
if len(merged) != 151 or merged.ith_3clusters.isna().any():
    raise RuntimeError(f"Expected 151 complete ITH matches, got {len(merged)}")
if not merged.ith_3clusters.between(0, 1).all():
    raise RuntimeError("ith_3clusters outside 0-1")

cols = {
    "Age": "年龄",
    "Gender": "性别",
    "Smoking": "吸烟",
    "NLR": "中/淋（NLR）",
    "PLR": "板/淋（PLR）",
    "LMR": "淋/单(LMR)",
    "Location": "位置（鼻腔1，鼻窦2）",
    "Histology": "病理类型（1鳞癌；2腺癌；3未分化癌；4鼻窦SMARCB1缺失癌）",
    "Ki67_status": "label Ki67",
    "TNM_overall": "overall stage",
    "Size_category": "Size>5cm",
    "T2_heterogeneity": "T2不均匀信号（50%）",
    "Margin": "边缘（1清楚）",
    "Myxoid": "粘液样变（10%）",
    "Necrosis": "坏死（10%）",
    "Septations": "分隔",
    "Enhancement": "增强（1轻中度，2明显）",
}
missing_source = [c for c in cols.values() if c not in merged.columns]
if missing_source:
    raise RuntimeError(f"Missing source columns: {missing_source}")

data = pd.DataFrame({"patient_id": merged.patient_id.astype(str),
                     "ITHscore_3D": pd.to_numeric(merged.ith_3clusters, errors="coerce")})
for new, old in cols.items():
    data[new] = pd.to_numeric(merged[old], errors="coerce")

# Recode only to the user-requested display scale; rank/group tests are invariant
# to these monotone/bijective recodings.
data["Location"] = data["Location"] - 1
data["Histology"] = data["Histology"] - 1
data["Enhancement"] = data["Enhancement"] - 1

specs = [
    ("Age", "Continuous", "spearman"),
    ("Gender", "Binary categorical", "mw"),
    ("Smoking", "Binary categorical", "mw"),
    ("NLR", "Continuous", "spearman"),
    ("PLR", "Continuous", "spearman"),
    ("LMR", "Continuous", "spearman"),
    ("Location", "Binary categorical", "mw"),
    ("Histology", "Nominal categorical (4 levels)", "kw"),
    ("Ki67_status", "Binary categorical", "mw"),
    ("TNM_overall", "Ordered categorical", "spearman"),
    ("Size_category", "Binary categorical", "mw"),
    ("Size_continuous", "Continuous", "unavailable"),
    ("T2_heterogeneity", "Binary categorical", "mw"),
    ("Margin", "Binary categorical", "mw"),
    ("Myxoid", "Binary categorical", "mw"),
    ("Necrosis", "Binary categorical", "mw"),
    ("Septations", "Binary categorical", "mw"),
    ("Enhancement", "Binary categorical", "mw"),
]

rows = []
for var, vtype, method in specs:
    if method == "unavailable":
        rows.append({"Variable": var, "Variable_type": vtype,
                     "Test_method": "Not tested - continuous size absent from source data",
                     "N_valid": 0, "Effect_size": np.nan, "Effect_size_type": "NA",
                     "Test_statistic": np.nan, "P_raw": np.nan,
                     "Data_status": "Unavailable in 原始数据.csv"})
        continue
    z = data[["ITHscore_3D", var]].dropna()
    y = z.ITHscore_3D.to_numpy(float)
    x = z[var].to_numpy(float)
    if method == "spearman":
        res = stats.spearmanr(x, y)
        effect = stat = float(res.statistic); p = float(res.pvalue)
        test_name, effect_name = "Spearman rank correlation", "rho"
    elif method == "mw":
        levels = np.sort(np.unique(x))
        if len(levels) != 2:
            raise RuntimeError(f"{var} expected 2 levels, got {levels}")
        y0, y1 = y[x == levels[0]], y[x == levels[1]]
        res = stats.mannwhitneyu(y0, y1, alternative="two-sided", method="asymptotic")
        stat = float(res.statistic); p = float(res.pvalue)
        effect = float(1 - 2 * stat / (len(y0) * len(y1)))
        test_name, effect_name = "Mann-Whitney U", "rank-biserial r (positive=higher in level 1)"
    else:
        levels = np.sort(np.unique(x))
        groups = [y[x == lev] for lev in levels]
        res = stats.kruskal(*groups)
        stat = float(res.statistic); p = float(res.pvalue)
        k, n = len(groups), len(y)
        effect = float(max(0.0, (stat - k + 1) / (n - k)))
        test_name, effect_name = "Kruskal-Wallis H", "eta-squared"
    rows.append({"Variable": var, "Variable_type": vtype, "Test_method": test_name,
                 "N_valid": len(z), "Effect_size": effect,
                 "Effect_size_type": effect_name, "Test_statistic": stat,
                 "P_raw": p, "Data_status": "Available"})

res = pd.DataFrame(rows)
available = res.P_raw.notna()
res["P_adjusted_FDR"] = np.nan
res.loc[available, "P_adjusted_FDR"] = multipletests(
    res.loc[available, "P_raw"], method="fdr_bh")[1]
res["Significance"] = np.where(res.P_adjusted_FDR < 0.05, "Yes", "No")
res.loc[~available, "Significance"] = "Not tested"
res.to_csv(OUT / "ITHscore_correlation_results.csv", index=False, encoding="utf-8-sig")
data.to_csv(OUT / "ITHscore_correlation_merged_data.csv", index=False, encoding="utf-8-sig")

sig = res.loc[res.Significance == "Yes", "Variable"].tolist()
(OUT / "ITHscore_significant_vars.txt").write_text(
    "\n".join(sig) + ("\n" if sig else ""), encoding="utf-8")

# QC and method audit.
missingness = []
for var, _, _ in specs:
    if var == "Size_continuous":
        missingness.append({"Variable": var, "N_total": 151, "N_valid": 0,
                            "Missing_N": 151, "Missing_percent": 100.0})
    else:
        nvalid = int(data[var].notna().sum())
        missingness.append({"Variable": var, "N_total": 151, "N_valid": nvalid,
                            "Missing_N": 151-nvalid, "Missing_percent": 100*(151-nvalid)/151})
pd.DataFrame(missingness).to_csv(OUT / "Table_Missingness.csv", index=False, encoding="utf-8-sig")

level_labels = {
    "Gender": {0:"Female",1:"Male"}, "Smoking": {0:"No",1:"Yes"},
    "Location": {0:"Nasal cavity",1:"Sinonasal sinus"},
    "Histology": {0:"SCC",1:"ADC",2:"Undifferentiated",3:"SMARCB1-deficient"},
    "Ki67_status": {0:"Low (<=50%)",1:"High (>50%)"},
    "TNM_overall": {2:"II",3:"III",4:"IVA",5:"IVB"},
    "Size_category": {0:"<=5 cm",1:">5 cm"},
    "T2_heterogeneity": {0:"Homogeneous",1:"Heterogeneous"},
    "Margin": {0:"Ill-defined",1:"Well-defined"},
    "Myxoid": {0:"Absent",1:"Present"}, "Necrosis": {0:"Absent",1:"Present"},
    "Septations": {0:"Absent",1:"Present"},
    "Enhancement": {0:"Mild/moderate",1:"Marked"},
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
palette = ["#4DBBD5", "#E64B35", "#00A087", "#3C5488"]
for var in sig:
    row = res.loc[res.Variable == var].iloc[0]
    z = data[["ITHscore_3D", var]].dropna().copy()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    if row.Test_method == "Spearman rank correlation" and row.Variable_type == "Continuous":
        sns.regplot(data=z, x=var, y="ITHscore_3D", lowess=True,
                    scatter_kws={"s":30,"alpha":0.72,"color":"#3C5488"},
                    line_kws={"lw":2.2,"color":"#E64B35"}, ax=ax)
        subtitle = f"Spearman rho={row.Effect_size:.3f}; FDR P={row.P_adjusted_FDR:.3g}"
    else:
        lab = level_labels.get(var, {})
        z["Group"] = z[var].map(lab).fillna(z[var].astype(str))
        order = [lab[k] for k in sorted(lab) if lab[k] in set(z.Group)] if lab else sorted(z.Group.unique())
        sns.violinplot(data=z, x="Group", y="ITHscore_3D", order=order, inner=None,
                       cut=0, palette=palette[:len(order)], alpha=.32, ax=ax)
        sns.boxplot(data=z, x="Group", y="ITHscore_3D", order=order, width=.28,
                    showfliers=False, boxprops={"facecolor":"white","alpha":.8}, ax=ax)
        sns.stripplot(data=z, x="Group", y="ITHscore_3D", order=order,
                      color="#333333", alpha=.55, size=3.2, jitter=.16, ax=ax)
        subtitle = f"{row.Effect_size_type}={row.Effect_size:.3f}; FDR P={row.P_adjusted_FDR:.3g}"
        ax.tick_params(axis="x", rotation=20)
    ax.set_title(f"3D-ITHscore vs {var}\n{subtitle}", weight="bold")
    ax.set_ylabel("3D-ITHscore (ith_3clusters)")
    fig.tight_layout()
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", var)
    fig.savefig(FIG / f"ITHscore_vs_{safe}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Combined Figure 3 candidate panel.
if sig:
    ncol = min(3, len(sig)); nrow = int(np.ceil(len(sig)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6*ncol, 4.8*nrow), squeeze=False)
    for ax, var in zip(axes.flat, sig):
        row = res.loc[res.Variable == var].iloc[0]
        z = data[["ITHscore_3D",var]].dropna().copy()
        if row.Test_method == "Spearman rank correlation" and row.Variable_type == "Continuous":
            sns.regplot(data=z,x=var,y="ITHscore_3D",lowess=True,
                        scatter_kws={"s":24,"alpha":.65,"color":"#3C5488"},
                        line_kws={"lw":2,"color":"#E64B35"},ax=ax)
        else:
            lab=level_labels.get(var,{}); z["Group"]=z[var].map(lab).fillna(z[var].astype(str))
            order=[lab[k] for k in sorted(lab) if lab[k] in set(z.Group)] if lab else sorted(z.Group.unique())
            sns.boxplot(data=z,x="Group",y="ITHscore_3D",order=order,palette=palette[:len(order)],ax=ax)
            sns.stripplot(data=z,x="Group",y="ITHscore_3D",order=order,color="#333",alpha=.45,size=2.6,jitter=.15,ax=ax)
            ax.tick_params(axis="x",rotation=20)
        ax.set_title(f"{var}\nFDR P={row.P_adjusted_FDR:.3g}",weight="bold")
        ax.set_ylabel("3D-ITHscore")
    for ax in axes.flat[len(sig):]: ax.axis("off")
    fig.suptitle("FDR-significant baseline correlates of 3D-ITHscore",fontsize=16,weight="bold",y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "Figure3_candidate_correlations.png",dpi=300,bbox_inches="tight")
    plt.close(fig)

summary = {
    "seed": SEED, "N_merged": len(merged), "ith_source": "ITH3_TRAIN/TEST ith_3clusters",
    "candidate_variables_listed": 18, "variables_tested": int(available.sum()),
    "variables_unavailable": ["Size_continuous"], "FDR_method": "Benjamini-Hochberg",
    "FDR_family_size": int(available.sum()), "FDR_threshold": 0.05,
    "significant_variables": sig,
}
(OUT / "analysis_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

report = [
    "=== Corrected 3D-ITHscore correlation analysis ===",
    f"N merged: {len(merged)}",
    "ITH source: corrected ith_3clusters (ITH3_TRAIN.csv + ITH3_TEST.csv)",
    "Candidate variables listed: 18",
    f"Variables statistically tested: {available.sum()}",
    "Unavailable variable: Size_continuous (not present as centimeters in 原始数据.csv; the available Size field is binary Size>5cm)",
    "FDR: Benjamini-Hochberg across the 17 valid tests",
    "FDR-significant variables: " + (", ".join(sig) if sig else "None"),
    "Excluded by design: treatment variables and all survival outcomes",
]
(OUT / "analysis_report.txt").write_text("\n".join(report)+"\n",encoding="utf-8")

# Save a copy of this running script and a manifest after orchestration copies it.
print(json.dumps(summary,ensure_ascii=False))
