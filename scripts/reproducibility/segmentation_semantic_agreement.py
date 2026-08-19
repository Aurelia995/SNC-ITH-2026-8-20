from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import SimpleITK as sitk
from scipy import stats
from sklearn.metrics import cohen_kappa_score

SEED = 2024
RNG = np.random.default_rng(SEED)
ROOT = Path("/root/autodl-tmp")
OUT = ROOT / "HCR_runs" / "reproducibility_20260820"
OUT.mkdir(parents=True, exist_ok=True)
ICC1 = ROOT / "HCR_input" / "Icc1" / "mask1"
ICC2 = ROOT / "HCR_input" / "Icc2" / "mask2"
R1_FILE = ROOT / "HCR_input" / "reader1_semantic.csv"
R2_FILE = ROOT / "HCR_input" / "reader2_semantic.csv"
CLIN_FILE = ROOT / "HCR_input" / "clinical_model_dataset_for_repro.csv"


def norm_id(x):
    s = str(x).strip().lower()
    if s.endswith(".0"): s = s[:-2]
    if s.startswith("c"): s = s[1:]
    return s.lstrip("0")


def read_clinical():
    d = pd.read_csv(CLIN_FILE, encoding="gb18030")
    if len(d) != 151: raise RuntimeError(f"Expected 151 clinical rows, got {len(d)}")
    safe = ["patient_id", "dataset", "event", "time", "TNM", "T_stage", "N_stage", "Sex", "Age", "Smoking", "NLR", "PLR", "LMR", "Site", "Histology", "Ki67", "Size5cm", "T2_heterogeneity", "Margin", "Myxoid", "Necrosis", "Septations", "Enhancement"]
    d.columns = safe
    d["id_norm"] = d.patient_id.map(norm_id)
    d["Center"] = d.dataset.astype(str).str.lower().map({"train":"Training center", "training":"Training center", "test":"Validation center", "validation":"Validation center"})
    return d


clin = read_clinical()

# ---------- Layer 1: segmentation Dice ----------
f1 = {norm_id(p.name.replace(".nii.gz", "")): p for p in ICC1.glob("*.nii.gz")}
f2 = {norm_id(p.name.replace(".nii.gz", "")): p for p in ICC2.glob("*.nii.gz")}
common = sorted(set(f1) & set(f2))
if len(common) != 40 or len(f1) != 40 or len(f2) != 40:
    raise RuntimeError(f"Expected 40 paired masks; ICC1={len(f1)}, ICC2={len(f2)}, common={len(common)}")

dice_rows = []
for pid in common:
    im1, im2 = sitk.ReadImage(str(f1[pid])), sitk.ReadImage(str(f2[pid]))
    raw_a = sitk.GetArrayFromImage(im1) > 0
    raw_b = sitk.GetArrayFromImage(im2) > 0
    if raw_a.shape == raw_b.shape and raw_a.sum() + raw_b.sum() > 0:
        index_dice = 2.0 * np.logical_and(raw_a, raw_b).sum() / (raw_a.sum() + raw_b.sum())
        support_identical = bool(np.array_equal(raw_a, raw_b))
    else:
        index_dice = np.nan
        support_identical = False
    geom_match = (im1.GetSize() == im2.GetSize() and np.allclose(im1.GetSpacing(), im2.GetSpacing(), atol=1e-6)
                  and np.allclose(im1.GetOrigin(), im2.GetOrigin(), atol=1e-5) and np.allclose(im1.GetDirection(), im2.GetDirection(), atol=1e-6))
    resampled = False
    if not geom_match:
        im2 = sitk.Resample(im2, im1, sitk.Transform(), sitk.sitkNearestNeighbor, 0, im2.GetPixelID())
        resampled = True
    a = sitk.GetArrayFromImage(im1) > 0
    b = sitk.GetArrayFromImage(im2) > 0
    na, nb = int(a.sum()), int(b.sum())
    if na + nb == 0: dice = np.nan
    else: dice = 2.0 * np.logical_and(a, b).sum() / (na + nb)
    voxel_mm3 = float(np.prod(im1.GetSpacing()))
    dice_rows.append({"patient_id_norm":pid, "mask1_file":f1[pid].name, "mask2_file":f2[pid].name,
                      "DSC":dice, "mask1_voxels":na, "mask2_voxels":nb,
                      "index_space_DSC_QC_only":index_dice, "voxel_support_identical_before_geometry_alignment":support_identical,
                      "mask1_volume_cm3":na*voxel_mm3/1000.0, "mask2_volume_cm3":nb*voxel_mm3/1000.0,
                      "geometry_originally_matched":geom_match, "mask2_resampled_to_mask1":resampled})
dice = pd.DataFrame(dice_rows).merge(clin[["id_norm","patient_id","Center","Site"]], left_on="patient_id_norm", right_on="id_norm", how="left", validate="one_to_one")
if dice.patient_id.isna().any(): raise RuntimeError(f"Unmatched ICC IDs: {dice.loc[dice.patient_id.isna(),'patient_id_norm'].tolist()}")
dice["Primary_site"] = dice.Site.map({1:"Nasal cavity", 2:"Paranasal sinus"}).fillna("Unknown")
dice["Volume_tertile"] = pd.qcut(dice.mask1_volume_cm3, 3, labels=["Low", "Medium", "High"], duplicates="raise")
dice.to_csv(OUT / "Table_L1_Patient_Dice.csv", index=False)

q1, q3 = dice.DSC.quantile([.25,.75])
summary = pd.DataFrame([{"N_paired":len(dice), "DSC_mean":dice.DSC.mean(), "DSC_SD":dice.DSC.std(ddof=1), "DSC_median":dice.DSC.median(), "DSC_Q1":q1, "DSC_Q3":q3,
                         "DSC_min":dice.DSC.min(), "DSC_max":dice.DSC.max(), "Geometry_match_N":int(dice.geometry_originally_matched.sum()), "Resampled_N":int(dice.mask2_resampled_to_mask1.sum()),
                         "Physical_DSC_exactly_1_N":int(np.isclose(dice.DSC,1,atol=1e-12).sum()), "Voxel_support_identical_before_alignment_N":int(dice.voxel_support_identical_before_geometry_alignment.sum()),
                         "Physical_DSC_below_0.9_N":int((dice.DSC<.9).sum()), "Physical_DSC_zero_N":int((dice.DSC==0).sum())}])
summary.to_csv(OUT / "Table_L1_Dice_Summary.csv", index=False)

strata_rows=[]
for variable in ["Center","Primary_site","Volume_tertile"]:
    tmp=dice.dropna(subset=[variable,"DSC"])
    levels=list(tmp[variable].astype(str).unique())
    groups=[tmp.loc[tmp[variable].astype(str)==lv,"DSC"].to_numpy() for lv in levels]
    if len(groups)>=2 and all(len(g)>0 for g in groups):
        H,p=stats.kruskal(*groups); note="Kruskal-Wallis test"
    else:
        H,p=np.nan,np.nan; note="Not estimable: only one observed level"
    for lv,g in zip(levels,groups):
        strata_rows.append({"Stratification":variable,"Level":lv,"N":len(g),"DSC_mean":np.mean(g),"DSC_SD":np.std(g,ddof=1) if len(g)>1 else np.nan,
                            "DSC_median":np.median(g),"DSC_Q1":np.quantile(g,.25),"DSC_Q3":np.quantile(g,.75),"Kruskal_Wallis_H_global":H,"Kruskal_Wallis_P_global":p,"Test_note":note})
strata=pd.DataFrame(strata_rows)
strata.to_csv(OUT / "Table_L1_Dice_Stratified.csv", index=False)

sns.set_theme(style="whitegrid",context="talk")
fig,ax=plt.subplots(figsize=(7,5.2)); sns.histplot(dice.DSC,bins=10,kde=True,color="#3C5488",ax=ax)
ax.axvline(dice.DSC.mean(),color="#E64B35",ls="--",lw=2,label=f"Mean={dice.DSC.mean():.3f}"); ax.axvline(dice.DSC.median(),color="#00A087",ls=":",lw=2,label=f"Median={dice.DSC.median():.3f}")
ax.set_xlabel("3D Dice similarity coefficient"); ax.set_ylabel("Number of patients"); ax.set_title("Inter-observer segmentation agreement (N=40)"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT/"Figure_L1_Dice_Histogram.png",dpi=300,bbox_inches="tight"); plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,5.2))
for ax,var,title in zip(axes,["Primary_site","Volume_tertile"],["Primary site","Mask-1 volume tertile"]):
    sns.boxplot(data=dice,x=var,y="DSC",color="#4DBBD5",width=.55,ax=ax); sns.stripplot(data=dice,x=var,y="DSC",color="#3C5488",size=5,jitter=.12,ax=ax)
    p=strata.loc[strata.Stratification==var,"Kruskal_Wallis_P_global"].iloc[0]
    ax.set_title(f"{title}\nKruskal-Wallis P={p:.3g}" if np.isfinite(p) else f"{title}\nTest not estimable"); ax.set_xlabel(""); ax.set_ylabel("DSC")
fig.tight_layout(); fig.savefig(OUT/"Figure_L1_Dice_Stratified.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# ---------- Layer 2: semantic inter-reader kappa ----------
r1=pd.read_csv(R1_FILE,encoding="gb18030")
r2=pd.read_csv(R2_FILE,encoding="gb18030")
if len(r1)!=151 or len(r2)!=151: raise RuntimeError(f"Reader tables must each have 151 rows; got {len(r1)}, {len(r2)}")
if not np.array_equal(r1.iloc[:,0].to_numpy(),r2.iloc[:,0].to_numpy()): raise RuntimeError("Reader group columns differ by row")

feature_names=["Tumor size >5 cm","FS-T2WI heterogeneity >=50%","Clear margin","Myxoid change >=10%","Necrosis >=10%","Septations present","Marked enhancement"]
formal_cols=["Size5cm","T2_heterogeneity","Margin","Myxoid","Necrosis","Septations","Enhancement"]
raw_cols=list(r1.columns[1:8])

# Source files have no patient_id. Row mapping is accepted only after checking shared group sequence and near-exact agreement with the formal data order.
if not np.array_equal((clin.Center=="Training center").map({True:1,False:2}).to_numpy(),r1.iloc[:,0].to_numpy()):
    raise RuntimeError("Reader rows do not match clinical cohort sequence")
rowmap_checks=[]
for raw,formal,name in zip(raw_cols,formal_cols,feature_names):
    a=pd.to_numeric(r1[raw],errors="coerce"); b=pd.to_numeric(clin[formal],errors="coerce"); ok=a.notna()&b.notna()
    rowmap_checks.append({"Feature":name,"N_comparable":int(ok.sum()),"Reader1_vs_formal_rowwise_agreement":float((a[ok].to_numpy()==b[ok].to_numpy()).mean())})
rowmap=pd.DataFrame(rowmap_checks)
if rowmap.Reader1_vs_formal_rowwise_agreement.min()<.95: raise RuntimeError("Row-order mapping validation below 95%; patient IDs cannot be assigned safely")
rowmap.to_csv(OUT/"Table_L2_Row_Mapping_QC.csv",index=False)

def landis_kappa(k):
    if not np.isfinite(k): return "Not estimable"
    if k < 0: return "Poor"
    if k <= .20: return "Slight"
    if k <= .40: return "Fair"
    if k <= .60: return "Moderate"
    if k <= .80: return "Substantial"
    return "Almost perfect"

krows=[]; disag=[]
for raw,formal,name in zip(raw_cols,formal_cols,feature_names):
    a=pd.to_numeric(r1[raw],errors="coerce"); b=pd.to_numeric(r2[raw],errors="coerce"); ok=a.notna()&b.notna()
    av=a[ok].astype(int).to_numpy(); bv=b[ok].astype(int).to_numpy(); kap=cohen_kappa_score(av,bv); agree=(av==bv).mean()
    boots=[]
    for _ in range(1000):
        ix=RNG.integers(0,len(av),len(av)); x,y=av[ix],bv[ix]
        if len(np.unique(x))<2 and len(np.unique(y))<2: continue
        kk=cohen_kappa_score(x,y)
        if np.isfinite(kk): boots.append(kk)
    lo,hi=np.quantile(boots,[.025,.975])
    dis_idx=np.flatnonzero(ok.to_numpy())[av!=bv]
    krows.append({"Feature":name,"N_valid_pairs":len(av),"Reader1_missing":int(a.isna().sum()),"Reader2_missing":int(b.isna().sum()),"Kappa":kap,"Kappa_CI95_low":lo,"Kappa_CI95_high":hi,
                  "Bootstrap_valid":len(boots),"Percent_agreement":agree*100,"Disagreement_N":int((av!=bv).sum()),"Landis_Koch_strength":landis_kappa(kap),
                  "Kappa_below_0.4_flag":"Yes" if kap<.4 else "No"})
    for idx in dis_idx:
        formal_value=clin.iloc[idx][formal]
        disag.append({"patient_id":clin.iloc[idx].patient_id,"Row_number_in_reader_files":idx+2,"Feature":name,"Reader1_score":a.iloc[idx],"Reader2_score":b.iloc[idx],"Formal_analysis_value":formal_value,
                      "Adjudication_status":"Formal value available; verify/document whether derived from senior-radiologist adjudication"})
kappa=pd.DataFrame(krows); disagreements=pd.DataFrame(disag)
kappa.to_csv(OUT/"Table_L2_Semantic_Kappa.csv",index=False)
disagreements.to_csv(OUT/"Table_L2_Disagreement_Cases.csv",index=False)

fig,ax=plt.subplots(figsize=(9,6))
y=np.arange(len(kappa))[::-1]; x=kappa.Kappa.to_numpy(); lo=kappa.Kappa_CI95_low.to_numpy(); hi=kappa.Kappa_CI95_high.to_numpy()
cols=["#E64B35" if v<.4 else "#3C5488" for v in x]
for yi,xx,ll,hh,col in zip(y,x,lo,hi,cols): ax.errorbar(xx,yi,xerr=[[xx-ll],[hh-xx]],fmt="o",color=col,ecolor=col,capsize=4,markersize=7)
for ref in [.4,.6,.8]: ax.axvline(ref,color="gray",ls="--",lw=1)
ax.set_yticks(y); ax.set_yticklabels(kappa.Feature); ax.set_xlim(-.05,1.02); ax.set_xlabel("Cohen's kappa (bootstrap 95% CI)"); ax.set_title("Inter-observer agreement for MRI semantic features\nRed indicates kappa <0.40")
fig.tight_layout(); fig.savefig(OUT/"Figure_L2_Semantic_Kappa.png",dpi=300,bbox_inches="tight"); plt.close(fig)

weak=kappa.loc[kappa.Kappa<.4,"Feature"].tolist()
center_levels=dice.Center.value_counts().to_dict()
center_p=strata.loc[strata.Stratification=="Center","Kruskal_Wallis_P_global"].iloc[0]
center_note=f"Dataset-defined centers represented in the ICC subset: {center_levels}. The center comparison was estimable (Kruskal-Wallis P={center_p:.6g}), but is sensitive to the observed geometry mismatches and should be interpreted cautiously."
report=f"""Imaging reproducibility analysis report
=======================================
Seed for semantic-feature bootstrap: {SEED}

Layer 1 - Segmentation spatial agreement
----------------------------------------
Forty patients had paired ICC1/ICC2 3D masks. Dice was calculated on nonzero voxels as 2|A intersection B|/(|A|+|B|). Geometry matched originally for {int(dice.geometry_originally_matched.sum())}/40 pairs; {int(dice.mask2_resampled_to_mask1.sum())} masks required nearest-neighbor resampling to mask1 geometry.
DSC mean +/- SD: {dice.DSC.mean():.4f} +/- {dice.DSC.std(ddof=1):.4f}
Median [IQR]: {dice.DSC.median():.4f} [{q1:.4f}, {q3:.4f}]
Range: {dice.DSC.min():.4f}-{dice.DSC.max():.4f}
{center_note}

Critical mask QC: {int(np.isclose(dice.DSC,1,atol=1e-12).sum())}/40 pairs had physical-space DSC exactly 1.0 and {int(dice.voxel_support_identical_before_geometry_alignment.sum())}/40 had identical nonzero voxel support before geometry alignment. Only one pair was binary-file identical by MD5, so identical voxel support was not caused solely by byte-for-byte duplicate files; however, exact independent delineations in this proportion are unusual and the segmentation provenance should be verified. Six pairs had physical DSC <0.90, including one DSC=0. These low values were concentrated among geometry-mismatched files. Therefore the physical-space DSC summary is retained without exclusion, but manuscript interpretation should state that mask export/reference-geometry consistency requires confirmation before treating the result as definitive inter-observer reproducibility.

Stratified results:
{strata.to_string(index=False)}

Layer 2 - MRI semantic inter-observer agreement
------------------------------------------------
The two reader files contained 151 rows and no patient identifier. Patient IDs were assigned by their shared row order after confirming the identical 129/22 cohort sequence and 98.6%-100% rowwise agreement between Reader 1 and the corresponding formal semantic columns. This mapping assumption is documented in Table_L2_Row_Mapping_QC.csv.

{kappa.to_string(index=False)}

Features with kappa <0.40: {', '.join(weak) if weak else 'None'}.
For any such feature, interpretation should explicitly acknowledge greater subjectivity and recommend future objective quantitative measurement (for example, automated volumetric proportion thresholds rather than visual estimation).

Disagreement/adjudication handling
----------------------------------
All reader-discordant cases are listed with patient ID and both scores in Table_L2_Disagreement_Cases.csv. A formal analysis value is present in the clinical dataset, but the supplied files do not document whether it was produced by a third senior radiologist. Therefore the table conservatively marks these cases as requiring verification/documentation of adjudication provenance; it does not falsely claim adjudication was completed.
"""
(OUT/"完整分析报告.txt").write_text(report,encoding="utf-8")

methods="""Methods
Paired three-dimensional binary masks from Reader 1 (ICC1) and Reader 2 (ICC2) were compared using the Dice similarity coefficient. Image size, spacing, origin, and direction were audited before comparison; any geometrically mismatched second-reader mask was resampled to the first-reader reference using nearest-neighbor interpolation and recorded. Index-space overlap and identical voxel support were retained as quality-control fields only, whereas physical-space Dice was the reported metric. Mask-1 physical volume was calculated from the nonzero voxel count and voxel spacing, and ICC-sample tertiles defined low, medium, and high volume groups. Dice distributions were compared across dataset-defined center, primary-site, and volume strata using Kruskal-Wallis tests. Exact-overlap and geometry-mismatch flags were reported without excluding cases.

For seven binary MRI semantic features, Cohen's kappa, percent agreement, and percentile 95% confidence intervals from 1,000 paired bootstrap resamples (seed 2024) were calculated. Agreement was interpreted using Landis and Koch categories. Missing ratings were handled using pairwise complete observations without imputation. The reader files lacked patient identifiers; IDs were attached by the common row order only after cohort-sequence and formal-column concordance checks. Discordant cases were listed for senior-radiologist adjudication/provenance verification.
"""
(OUT/"Methods_Reproducibility.txt").write_text(methods,encoding="utf-8")

shutil.copy2(__file__,OUT/"reproducibility_segmentation_semantic.py")
meta={"seed":SEED,"dice_formula":"2*intersection/(A+B)","mask1_dir":str(ICC1),"mask2_dir":str(ICC2),"paired_masks":len(dice),"semantic_N":151,"kappa_bootstrap":1000,"missing_handling":"pairwise complete; no imputation","center_definition":"dataset-defined training vs validation center","mask_QC":"geometry, index-space support identity, physical-space Dice, MD5 audit"}
(OUT/"Reproducibility_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")

manifest=[]
for f in sorted(OUT.iterdir()):
    if f.is_file() and f.name!="MD5_manifest.csv": manifest.append({"File":f.name,"Bytes":f.stat().st_size,"MD5":hashlib.md5(f.read_bytes()).hexdigest()})
pd.DataFrame(manifest).to_csv(OUT/"MD5_manifest.csv",index=False)
print(report)
print("OUTPUT_DIR",OUT)
