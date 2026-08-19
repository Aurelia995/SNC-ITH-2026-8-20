#!/usr/bin/env python3
from pathlib import Path
import hashlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from neuroCombat import neuroCombat
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SEED = 20250308
ROOT = Path("/root/autodl-tmp/HCR_runs/HCR_pure_revised_20260814")
OUT = ROOT / "results"
TRAIN = Path("/root/autodl-tmp/HCR_input/radiomics-TRAIN.csv")
SCANNER = Path("/root/autodl-tmp/HCR_input/clinical_SNC1.xlsx")
ICC = Path("/root/autodl-tmp/HCR_runs/HCR_existing_20260812/model/ICC_results_all.csv")


def nid(x):
    digits = re.sub(r"\D", "", str(x))
    return str(int(digits))


def correlation_prune(x, cutoff=0.9):
    c = x.corr(method="pearson").abs().fillna(0)
    np.fill_diagonal(c.values, 0)
    means = c.mean(axis=0).to_numpy()
    hits = np.argwhere(np.triu(c.to_numpy(), 1) > cutoff)
    removed = {int(j if means[j] > means[i] else i) for i, j in hits}
    return [f for i, f in enumerate(c.columns) if i not in removed]


def md5(path):
    h = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN)
    train["patient_id"] = train["id"].map(nid)
    excluded = {"id", "OS/label", "OS/m", "patient_id"}
    raw = [c for c in train.columns if c not in excluded and "_diagnostics_" not in c]
    icc = pd.read_csv(ICC)
    icc_keep = set(icc.loc[icc["ICC_A1"] > 0.8, "feature"])
    features = [c for c in raw if c in icc_keep]

    x = train[features].replace([np.inf, -np.inf], np.nan)
    x = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=features)
    variance_keep = x.var(ddof=1).loc[lambda s: s >= 0.01].index.tolist()
    x = x[variance_keep]
    correlation_keep = correlation_prune(x, 0.9)
    x_before = x[correlation_keep].copy()

    scanner = pd.read_excel(SCANNER).iloc[1:, [2, 3, 4]].copy()
    scanner.columns = ["patient_id", "scanner_t1", "scanner_t2"]
    scanner["patient_id"] = scanner["patient_id"].map(nid)
    meta = train[["patient_id"]].merge(scanner, on="patient_id", validate="one_to_one")
    batch = meta["scanner_t1"].astype(str).str.strip() + " | " + meta["scanner_t2"].astype(str).str.strip()
    counts = batch.value_counts()
    batch = batch.where(~batch.isin(counts[counts < 5].index), "Other")

    corrected = neuroCombat(
        dat=x_before.T,
        covars=pd.DataFrame({"batch": batch.to_numpy()}),
        batch_col="batch",
    )["data"].T
    x_after = pd.DataFrame(corrected, columns=correlation_keep)
    if not np.isfinite(x_after.to_numpy()).all():
        raise RuntimeError("ComBat produced NaN/Inf")

    # Use one pre-ComBat standardization and one pre-ComBat PCA basis so that
    # movement between panels reflects correction rather than refitting axes.
    common_scaler = StandardScaler().fit(x_before)
    z_before = common_scaler.transform(x_before)
    z_after = common_scaler.transform(x_after)
    pca = PCA(n_components=2, random_state=SEED).fit(z_before)
    scores_before = pca.transform(z_before)
    scores_after = pca.transform(z_after)

    sns.set_theme(style="whitegrid", context="notebook")
    palette = dict(zip(sorted(batch.unique()), sns.color_palette("colorblind", n_colors=batch.nunique())))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, scores, title in zip(axes, [scores_before, scores_after], ["Before ComBat", "After ComBat"]):
        for label in sorted(batch.unique()):
            mask = batch.to_numpy() == label
            ax.scatter(scores[mask, 0], scores[mask, 1], s=42, alpha=0.78,
                       color=palette[label], edgecolor="white", linewidth=0.4, label=label)
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% pre-ComBat variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% pre-ComBat variance)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Scanner batch", loc="lower center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle("ComBat batch-effect QC: PCA on training radiomics features", y=0.99)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    pca_path = OUT / "combat_before_after_PCA.png"
    fig.savefig(pca_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Heatmap the 30 features with the largest pre-ComBat between-batch mean SD.
    before_z = pd.DataFrame(z_before, columns=correlation_keep)
    after_z = pd.DataFrame(z_after, columns=correlation_keep)
    before_means = before_z.assign(batch=batch.to_numpy()).groupby("batch").mean()
    after_means = after_z.assign(batch=batch.to_numpy()).groupby("batch").mean()
    top = before_means.std(axis=0, ddof=0).nlargest(min(30, len(correlation_keep))).index
    heat = pd.concat({"Before ComBat": before_means[top], "After ComBat": after_means[top]}, axis=0)
    vmax = max(1.0, float(np.nanquantile(np.abs(heat.to_numpy()), 0.98)))
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), sharey=True)
    for ax, state, matrix in zip(axes, ["Before ComBat", "After ComBat"], [before_means[top], after_means[top]]):
        sns.heatmap(matrix.T, ax=ax, cmap="vlag", center=0, vmin=-vmax, vmax=vmax,
                    cbar_kws={"label": "Batch mean (pre-ComBat z scale)"})
        ax.set_title(state)
        ax.set_xlabel("Scanner batch")
        ax.set_ylabel("Radiomics feature" if state == "Before ComBat" else "")
    fig.suptitle("ComBat batch-effect QC: top 30 features by pre-correction batch variation", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    heat_path = OUT / "combat_batch_effect_heatmap.png"
    fig.savefig(heat_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics = pd.DataFrame({
        "metric": ["N_training", "N_batches", "N_features", "PC1_variance_ratio_pre", "PC2_variance_ratio_pre",
                   "mean_abs_batch_mean_before", "mean_abs_batch_mean_after"],
        "value": [len(x_before), batch.nunique(), len(correlation_keep), pca.explained_variance_ratio_[0],
                  pca.explained_variance_ratio_[1], np.abs(before_means.to_numpy()).mean(),
                  np.abs(after_means.to_numpy()).mean()],
    })
    metrics.to_csv(OUT / "combat_QC_metrics.csv", index=False)
    pd.DataFrame({"file": [pca_path.name, heat_path.name], "md5": [md5(pca_path), md5(heat_path)]}).to_csv(
        OUT / "combat_QC_png_md5.csv", index=False
    )
    print(metrics.to_string(index=False))
    print(f"Saved: {pca_path}")
    print(f"Saved: {heat_path}")


if __name__ == "__main__":
    main()
