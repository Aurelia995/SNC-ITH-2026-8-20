#!/usr/bin/env python3
import hashlib
import json
import math
import re
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from neuroCombat import neuroCombat
from scipy.stats import chi2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import brier_score, concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv

SEED = 20250308
RNG = np.random.default_rng(SEED)
B = 2000

BASE = Path("/root/autodl-tmp")
HCR_RUN = BASE / "HCR_runs/HCR_pure_revised_20260814"
OUT = BASE / "HCR_runs/seven_models_ith3_original_20260814"
CLIN = BASE / "HCR_input/临床模型数据集.csv"
ITH_TRAIN = BASE / "HCR_input/ITH3_TRAIN.csv"
ITH_VALIDATION = BASE / "HCR_input/ITH3_TEST.csv"
TRAIN_RAD = BASE / "HCR_input/radiomics-TRAIN.csv"
VAL_RAD = BASE / "HCR_input/radiomics-TEST.csv"
SCANNER = BASE / "HCR_input/clinical_SNC1.xlsx"
ICC = BASE / "HCR_runs/HCR_existing_20260812/model/ICC_results_all.csv"

MODEL_NAMES = [
    "Clinical", "HCR", "ITH", "Clinical-ITH", "HCR-ITH",
    "Clinical-HCR", "Clinical-ITH-HCR"
]
COLORS = {
    "Clinical": "#0072B2", "HCR": "#E69F00", "ITH": "#009E73",
    "Clinical-ITH": "#CC79A7", "HCR-ITH": "#D55E00",
    "Clinical-HCR": "#56B4E9", "Clinical-ITH-HCR": "#6A3D9A",
}
CLIN_COLS = {
    "NLR": "中/淋（NLR）", "Ki67": "ki-67", "Margin": "边缘（1清楚）",
    "Myxoid": "粘液样变（10%）"
}
CLIN_FIXED = {"NLR": 0.0884221000794938, "Ki67": 0.0154731468510556,
              "Margin": 0.168326295723267, "Myxoid": 0.058965241829482}


def nid(x):
    return str(int(re.sub(r"\D", "", str(x))))


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def correlation_prune(x, cutoff=0.9):
    c = x.corr(method="pearson").abs().fillna(0)
    np.fill_diagonal(c.values, 0)
    means = c.mean(axis=0).to_numpy()
    hits = np.argwhere(np.triu(c.to_numpy(), 1) > cutoff)
    removed = {int(j if means[j] > means[i] else i) for i, j in hits}
    return [f for i, f in enumerate(c.columns) if i not in removed]


def prepare_hcr():
    tr = pd.read_csv(TRAIN_RAD); va = pd.read_csv(VAL_RAD)
    tr["patient_id"] = tr.id.map(nid); va["patient_id"] = va.id.map(nid)
    excluded = {"id", "OS/label", "OS/m", "patient_id"}
    raw = [c for c in tr if c not in excluded and "_diagnostics_" not in c]
    icc = pd.read_csv(ICC)
    keep_icc = set(icc.loc[icc.ICC_A1 > 0.8, "feature"])
    cols = [c for c in raw if c in keep_icc]
    xtr = tr[cols].replace([np.inf, -np.inf], np.nan)
    xv = va[cols].replace([np.inf, -np.inf], np.nan)
    imp = SimpleImputer(strategy="median").fit(xtr)
    xtr = pd.DataFrame(imp.transform(xtr), columns=cols)
    xv = pd.DataFrame(imp.transform(xv), columns=cols)
    keep_var = xtr.var(ddof=1).loc[lambda s: s >= .01].index.tolist()
    xtr = xtr[keep_var]; xv = xv[keep_var]
    keep_cor = correlation_prune(xtr, .9)
    xtr = xtr[keep_cor]; xv = xv[keep_cor]
    scan = pd.read_excel(SCANNER).iloc[1:, [2, 3, 4]].copy()
    scan.columns = ["patient_id", "scanner_t1", "scanner_t2"]
    scan.patient_id = scan.patient_id.map(nid)
    meta = tr[["patient_id"]].merge(scan, on="patient_id", validate="one_to_one")
    batch = meta.scanner_t1.astype(str).str.strip() + " | " + meta.scanner_t2.astype(str).str.strip()
    vc = batch.value_counts(); batch = batch.where(~batch.isin(vc[vc < 5].index), "Other")
    combat = neuroCombat(dat=xtr.T, covars=pd.DataFrame({"batch": batch.values}), batch_col="batch")
    ctr = pd.DataFrame(combat["data"].T, columns=keep_cor)
    scaler = StandardScaler().fit(ctr)
    ztr = pd.DataFrame(scaler.transform(ctr), columns=keep_cor)
    zv = pd.DataFrame(scaler.transform(xv), columns=keep_cor)
    coef = pd.read_csv(HCR_RUN / "results/final_model_coefficients.csv")
    feature_col = "feature" if "feature" in coef else coef.columns[0]
    coef_col = "coefficient" if "coefficient" in coef else coef.columns[1]
    coef = coef.loc[coef[coef_col].abs() > 1e-10, [feature_col, coef_col]]
    features = coef[feature_col].tolist(); beta = coef[coef_col].to_numpy(float)
    outtr = ztr[features].copy(); outtr.insert(0, "patient_id", tr.patient_id)
    outv = zv[features].copy(); outv.insert(0, "patient_id", va.patient_id)
    outtr["HCR_fixed_lp"] = ztr[features].to_numpy() @ beta
    outv["HCR_fixed_lp"] = zv[features].to_numpy() @ beta
    # Reproducibility check against the saved final HCR run.
    saved_tr = pd.read_csv(HCR_RUN / "results/risk_scores_train.csv")
    saved_va = pd.read_csv(HCR_RUN / "results/risk_scores_validation.csv")
    score_col_tr = [c for c in saved_tr if "risk" in c.lower()][0]
    score_col_va = [c for c in saved_va if "risk" in c.lower()][0]
    d1 = np.max(np.abs(outtr.HCR_fixed_lp.to_numpy() - saved_tr[score_col_tr].to_numpy()))
    d2 = np.max(np.abs(outv.HCR_fixed_lp.to_numpy() - saved_va[score_col_va].to_numpy()))
    if max(d1, d2) > 1e-8:
        raise RuntimeError(f"HCR score reproduction mismatch: train={d1}, validation={d2}")
    return outtr, outv, features, beta, d1, d2


def prepare_data(htr, hva):
    c = pd.read_csv(CLIN, encoding="gb18030")
    itr = pd.read_csv(ITH_TRAIN, encoding="gb18030").rename(columns={"id": "patient_id"})
    iva = pd.read_csv(ITH_VALIDATION, encoding="gb18030").rename(columns={"id": "patient_id"})
    itr["dataset"] = "Train"; iva["dataset"] = "Test"
    i = pd.concat([itr, iva], ignore_index=True)
    c.patient_id = c.patient_id.map(nid); i.patient_id = i.patient_id.map(nid)
    if c.patient_id.duplicated().any() or i.patient_id.duplicated().any():
        raise RuntimeError("Duplicate patient IDs")
    # The two user-designated files are the authoritative 3D-ITHscore source.
    i = i.rename(columns={"ith_3clusters": "ith_score"})
    outcome = c[["patient_id", "OS/label", "OS/m"]].merge(
        i[["patient_id", "OS/label", "OS/m"]], on="patient_id", suffixes=("_clinical", "_ith"), validate="one_to_one"
    )
    if not ((outcome["OS/label_clinical"] == outcome["OS/label_ith"]).all() and
            (outcome["OS/m_clinical"] == outcome["OS/m_ith"]).all()):
        raise RuntimeError("ITH source outcomes differ from clinical outcomes")
    cols = ["patient_id", "dataset", "OS/label", "OS/m"] + list(CLIN_COLS.values())
    d = c[cols].merge(i[["patient_id", "dataset", "ith_score"]], on=["patient_id", "dataset"], validate="one_to_one")
    rename = {v: k for k, v in CLIN_COLS.items()}
    d = d.rename(columns=rename)
    tr = d.loc[d.dataset.eq("Train")].copy(); va = d.loc[d.dataset.eq("Test")].copy()
    tr = tr.merge(htr, on="patient_id", validate="one_to_one")
    va = va.merge(hva, on="patient_id", validate="one_to_one")
    if (len(tr), len(va), int(tr["OS/label"].sum()), int(va["OS/label"].sum())) != (129, 22, 68, 6):
        raise RuntimeError("Cohort hard check failed")
    impute = {}
    for col in ["NLR", "Ki67", "Margin", "Myxoid"]:
        if col in ["NLR", "Ki67"]:
            value = float(tr[col].median())
        else:
            value = float(tr[col].mode(dropna=True).iloc[0])
        impute[col] = value
        tr[col] = tr[col].fillna(value); va[col] = va[col].fillna(value)
    return tr.reset_index(drop=True), va.reset_index(drop=True), impute


def fit_cox(train, val, features):
    scaler = StandardScaler().fit(train[features])
    xtr = pd.DataFrame(scaler.transform(train[features]), columns=features)
    xv = pd.DataFrame(scaler.transform(val[features]), columns=features)
    fit = pd.concat([train[["OS/m", "OS/label"]].reset_index(drop=True), xtr], axis=1)
    cph = CoxPHFitter(penalizer=1e-7)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cph.fit(fit, duration_col="OS/m", event_col="OS/label", show_progress=False)
    return cph, cph.predict_log_partial_hazard(xtr).to_numpy(), cph.predict_log_partial_hazard(xv).to_numpy(), scaler, [str(x.message) for x in caught]


def baseline_survival(time, event, lp, grid):
    lp_original = np.asarray(lp, dtype=float)
    order = np.argsort(time)
    t = np.asarray(time)[order]; e = np.asarray(event)[order]; lp = lp_original[order]
    uniq = np.unique(t[e.astype(bool)])
    cum = 0.0; et = []; eh = []
    exp_lp = np.exp(np.clip(lp, -30, 30))
    for u in uniq:
        deaths = np.sum((t == u) & (e == 1))
        risk = exp_lp[t >= u].sum()
        cum += deaths / risk
        et.append(u); eh.append(cum)
    idx = np.searchsorted(np.asarray(et), np.asarray(grid), side="right") - 1
    h0 = np.where(idx >= 0, np.asarray(eh)[np.maximum(idx, 0)], 0.0)
    return np.exp(-np.outer(np.exp(np.clip(lp_original, -30, 30)), h0))


def harrell(time, event, risk):
    return float(concordance_index(time, -risk, event))


def bootstrap_cis(train_y, test_y, time, event, risks, tau, b=B):
    rows = []
    n = len(time)
    for name, risk in risks.items():
        h0 = harrell(time, event, risk); u0 = float(concordance_index_ipcw(train_y, test_y, risk, tau=tau)[0])
        hs = []; us = []
        for _ in range(b):
            ix = RNG.integers(0, n, n)
            if np.unique(event[ix]).size < 2: continue
            try:
                hs.append(harrell(time[ix], event[ix], risk[ix]))
                us.append(float(concordance_index_ipcw(train_y, test_y[ix], risk[ix], tau=tau)[0]))
            except Exception:
                pass
        hlo, hhi = np.quantile(hs, [.025, .975]); ulo, uhi = np.quantile(us, [.025, .975])
        rows.append({"Model": name, "Harrell_C": h0, "Harrell_C_L95": hlo, "Harrell_C_U95": hhi,
                     "Uno_C": u0, "Uno_C_L95": ulo, "Uno_C_U95": uhi, "Uno_tau_months": tau,
                     "Bootstrap_successful": min(len(hs), len(us))})
    return pd.DataFrame(rows)


def pairwise_bootstrap(time, event, risks, cohort, b=B):
    diffs = {(a, z): [] for j, a in enumerate(MODEL_NAMES) for z in MODEL_NAMES[j + 1:]}
    n = len(time); successful = 0
    for _ in range(b):
        ix = RNG.integers(0, n, n)
        if np.unique(event[ix]).size < 2: continue
        try:
            cs = {m: harrell(time[ix], event[ix], risks[m][ix]) for m in MODEL_NAMES}
        except Exception:
            continue
        successful += 1
        for pair in diffs: diffs[pair].append(cs[pair[0]] - cs[pair[1]])
    rows = []
    for (a, z), values in diffs.items():
        v = np.asarray(values); point = harrell(time, event, risks[a]) - harrell(time, event, risks[z])
        p = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))
        rows.append({"Cohort": cohort, "Model_A": a, "Model_B": z, "Delta_C_A_minus_B": point,
                     "Delta_L95": np.quantile(v, .025), "Delta_U95": np.quantile(v, .975),
                     "P_raw": p, "Bootstrap_successful": successful})
    tab = pd.DataFrame(rows)
    order = np.argsort(tab.P_raw.to_numpy()); ranked = tab.P_raw.to_numpy()[order] * len(tab) / np.arange(1, len(tab) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(len(tab)); adj[order] = np.minimum(ranked, 1)
    tab["P_FDR_BH"] = adj
    return tab


def km_risk(time, event, horizon):
    km = KaplanMeierFitter().fit(time, event_observed=event)
    return 1 - float(km.predict(horizon))


def calibration_plot(train, risks, horizons, output):
    with PdfPages(output) as pdf:
        for horizon in horizons:
            fig, ax = plt.subplots(figsize=(8.5, 7))
            for name in MODEL_NAMES:
                pred = 1 - baseline_survival(train["OS/m"], train["OS/label"], risks[name], [horizon])[:, 0]
                q = pd.qcut(pred, q=4, labels=False, duplicates="drop")
                points = []
                for g in sorted(pd.unique(q)):
                    mask = np.asarray(q == g)
                    points.append((pred[mask].mean(), km_risk(train.loc[mask, "OS/m"], train.loc[mask, "OS/label"], horizon)))
                points = np.asarray(points)
                ax.plot(points[:, 0], points[:, 1], marker="o", lw=1.8, color=COLORS[name], label=name)
            ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1, label="Ideal")
            ax.set(xlabel=f"Mean predicted {horizon}-month mortality", ylabel=f"Observed {horizon}-month mortality (KM)",
                   title=f"Training calibration at {horizon} months", xlim=(0, 1), ylim=(0, 1))
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
            fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def validation_calibration(val, train, risks_tr, risks_va, output):
    horizon = 24
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for name in MODEL_NAMES:
        pred = 1 - baseline_survival(train["OS/m"], train["OS/label"], risks_tr[name], [horizon])
        # Apply the training baseline hazard to validation linear predictors.
        # Recover H0 at horizon from a zero-LP pseudo-subject.
        s0 = baseline_survival(train["OS/m"], train["OS/label"], risks_tr[name], [horizon])
        h0 = -np.log(s0[:, 0]) / np.exp(np.clip(risks_tr[name], -30, 30))
        h0 = float(np.median(h0))
        pval = 1 - np.exp(-h0 * np.exp(np.clip(risks_va[name], -30, 30)))
        groups = pval > np.median(pval)
        points = []
        for g in [False, True]:
            mask = groups == g
            points.append((pval[mask].mean(), km_risk(val.loc[mask, "OS/m"], val.loc[mask, "OS/label"], horizon)))
        points = np.asarray(points)
        ax.plot(points[:, 0], points[:, 1], marker="o", lw=1.8, color=COLORS[name], label=name)
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1, label="Ideal")
    ax.set(xlabel="Mean predicted 24-month mortality", ylabel="Observed 24-month mortality (KM)",
           title="Validation calibration at 24 months (descriptive; two groups)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout(); fig.savefig(output, bbox_inches="tight"); plt.close(fig)


def net_benefit(event_by_horizon, predicted, thresholds):
    n = len(predicted); vals = []
    for pt in thresholds:
        positive = predicted >= pt
        tp = np.sum(positive & event_by_horizon); fp = np.sum(positive & ~event_by_horizon)
        vals.append(tp / n - fp / n * pt / (1 - pt))
    return np.asarray(vals)


def dca_plot(train, test, risks_tr, risks_te, horizon, output, validation=False):
    thresholds = np.linspace(.05, .75, 71)
    event_h = (test["OS/label"].to_numpy(bool) & (test["OS/m"].to_numpy() <= horizon))
    all_nb = event_h.mean() - (1 - event_h.mean()) * thresholds / (1 - thresholds)
    fig, ax = plt.subplots(figsize=(9.5, 7))
    rows = []
    for name in MODEL_NAMES:
        s0 = baseline_survival(train["OS/m"], train["OS/label"], risks_tr[name], [horizon])
        h0 = float(np.median(-np.log(s0[:, 0]) / np.exp(np.clip(risks_tr[name], -30, 30))))
        pred = 1 - np.exp(-h0 * np.exp(np.clip(risks_te[name], -30, 30)))
        nb = net_benefit(event_h, pred, thresholds)
        ax.plot(thresholds, nb, color=COLORS[name], lw=1.8, label=name)
        if validation:
            boots = []
            for _ in range(B):
                ix = RNG.integers(0, len(test), len(test))
                boots.append(net_benefit(event_h[ix], pred[ix], thresholds))
            lo, hi = np.quantile(np.asarray(boots), [.025, .975], axis=0)
            ax.fill_between(thresholds, lo, hi, color=COLORS[name], alpha=.08, linewidth=0)
        for t, v in zip(thresholds, nb): rows.append({"Cohort": "Validation" if validation else "Training", "Model": name, "Horizon": horizon, "Threshold": t, "Net_benefit": v})
    ax.plot(thresholds, np.zeros_like(thresholds), color="black", lw=1.2, label="Treat none")
    ax.plot(thresholds, all_nb, color="gray", ls="--", lw=1.2, label="Treat all")
    ax.set(xlabel="Threshold probability", ylabel="Net benefit",
           title=f"{'Validation' if validation else 'Training'} DCA at {horizon} months" + (" (bootstrap 95% bands)" if validation else ""))
    ax.set_ylim(max(-.2, np.nanquantile(all_nb, .05)), min(.8, max(.2, np.nanquantile(all_nb, .95) + .2)))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout(); fig.savefig(output, bbox_inches="tight"); plt.close(fig)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    htr, hva, hfeatures, hbeta, hdiff_tr, hdiff_va = prepare_hcr()
    tr, va, impute = prepare_data(htr, hva)
    tr.to_csv(OUT / "analysis_dataset_training.csv", index=False)
    va.to_csv(OUT / "analysis_dataset_validation.csv", index=False)
    pd.DataFrame([{"Variable": k, "Training_imputation_value": v} for k, v in impute.items()]).to_csv(OUT / "Table_Imputation.csv", index=False)

    # Fixed prior Clinical and fixed prior HCR models.
    clinical_tr = sum(CLIN_FIXED[k] * tr[k].to_numpy() for k in CLIN_FIXED)
    clinical_va = sum(CLIN_FIXED[k] * va[k].to_numpy() for k in CLIN_FIXED)
    hcr_tr = tr.HCR_fixed_lp.to_numpy(); hcr_va = va.HCR_fixed_lp.to_numpy()
    definitions = {
        "ITH": ["ith_score"],
        "Clinical-ITH": ["NLR", "Ki67", "Margin", "Myxoid", "ith_score"],
        "HCR-ITH": hfeatures + ["ith_score"],
        "Clinical-HCR": ["NLR", "Ki67", "Margin", "Myxoid"] + hfeatures,
        "Clinical-ITH-HCR": ["NLR", "Ki67", "Margin", "Myxoid"] + hfeatures + ["ith_score"],
    }
    risks_tr = {"Clinical": clinical_tr, "HCR": hcr_tr}; risks_va = {"Clinical": clinical_va, "HCR": hcr_va}
    fitted = {}; convergence = []
    for name, features in definitions.items():
        cph, rtr, rva, scaler, warns = fit_cox(tr, va, features)
        fitted[name] = {"model": cph, "scaler": scaler, "features": features}
        risks_tr[name] = rtr; risks_va[name] = rva
        convergence.append({"Model": name, "N_parameters": len(features), "Warnings": " | ".join(warns)})
    pd.DataFrame(convergence).to_csv(OUT / "model_fit_diagnostics.csv", index=False)
    joblib.dump({"models": fitted, "clinical_fixed_coefficients": CLIN_FIXED,
                 "hcr_features": hfeatures, "hcr_coefficients": hbeta, "imputation": impute,
                 "colors": COLORS}, OUT / "seven_models_objects.joblib")

    ytr = Surv.from_arrays(tr["OS/label"].astype(bool), tr["OS/m"].astype(float))
    yva = Surv.from_arrays(va["OS/label"].astype(bool), va["OS/m"].astype(float))
    ctrain = bootstrap_cis(ytr, ytr, tr["OS/m"].to_numpy(), tr["OS/label"].to_numpy(), risks_tr, tau=60).assign(Cohort="Training")
    cval = bootstrap_cis(ytr, yva, va["OS/m"].to_numpy(), va["OS/label"].to_numpy(), risks_va, tau=24).assign(Cohort="Validation")
    cindex = pd.concat([ctrain, cval], ignore_index=True)
    cindex.to_csv(OUT / "Table_Cindex.csv", index=False)

    # Time-dependent AUC at requested horizons and over restricted, estimable grids.
    auc_rows = []; curve_rows = []; iauc_rows = []
    for cohort, test, ytest, risks, horizons, grid in [
        ("Training", tr, ytr, risks_tr, [12, 24, 36, 60], np.linspace(12, 60, 97)),
        ("Validation", va, yva, risks_va, [12, 24, 36], np.linspace(12, 24, 49)),
    ]:
        for name in MODEL_NAMES:
            auc_h, mean_h = cumulative_dynamic_auc(ytr, ytest, risks[name], np.asarray(horizons, float))
            for t, a in zip(horizons, auc_h):
                auc_rows.append({"Cohort": cohort, "Model": name, "Time_months": t, "AUC": a,
                                 "Status": "Primary" if cohort == "Training" or t <= 24 else "Exploratory_3y"})
            auc_g, mean_auc = cumulative_dynamic_auc(ytr, ytest, risks[name], grid)
            for t, a in zip(grid, auc_g): curve_rows.append({"Cohort": cohort, "Model": name, "Time_months": t, "AUC": a})
            iauc_rows.append({"Cohort": cohort, "Model": name, "Integration_start_months": grid[0],
                              "Integration_end_months": grid[-1], "iAUC_survival_weighted": mean_auc,
                              "iAUC_time_average": np.trapz(auc_g, grid) / (grid[-1] - grid[0])})
    auc_detail = pd.DataFrame(auc_rows); auc_curve = pd.DataFrame(curve_rows); iauc = pd.DataFrame(iauc_rows)
    auc_detail.to_csv(OUT / "Table_TAUC_Details.csv", index=False)
    iauc.to_csv(OUT / "Table_TAUC_iAUC.csv", index=False)
    for cohort, fname in [("Training", "Fig_TAUC_Training.pdf"), ("Validation", "Fig_TAUC_Validation.pdf")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for name in MODEL_NAMES:
            q = auc_curve[(auc_curve.Cohort == cohort) & (auc_curve.Model == name)]
            ax.plot(q.Time_months, q.AUC, color=COLORS[name], lw=2, label=name)
        ax.axhline(.5, color="black", ls="--", lw=1)
        ax.set(xlabel="Time (months)", ylabel="Cumulative/dynamic AUC", title=f"Time-dependent AUC - {cohort} cohort", ylim=(0.3, 1.0))
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
        fig.tight_layout(); fig.savefig(OUT / fname, bbox_inches="tight"); plt.close(fig)

    # Brier/IBS, with survival probabilities based on training Breslow baseline.
    ibs_rows = []; brier_rows = []
    for cohort, test, ytest, risks_te, grid in [
        ("Training", tr, ytr, risks_tr, np.linspace(12, 60, 97)),
        ("Validation", va, yva, risks_va, np.linspace(12, 24, 49)),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for name in MODEL_NAMES:
            # Obtain H0 from training LP and apply it to either cohort.
            str0 = baseline_survival(tr["OS/m"], tr["OS/label"], risks_tr[name], grid)
            h0 = np.median(-np.log(np.clip(str0, 1e-15, 1)) / np.exp(np.clip(risks_tr[name], -30, 30))[:, None], axis=0)
            ste = np.exp(-np.outer(np.exp(np.clip(risks_te[name], -30, 30)), h0))
            _, bs = brier_score(ytr, ytest, ste, grid)
            ibs = integrated_brier_score(ytr, ytest, ste, grid)
            ibs_rows.append({"Cohort": cohort, "Model": name, "Start_months": grid[0], "End_months": grid[-1], "IBS": ibs})
            for t, v in zip(grid, bs): brier_rows.append({"Cohort": cohort, "Model": name, "Time_months": t, "Brier_score": v})
            ax.plot(grid, bs, color=COLORS[name], lw=2, label=f"{name} (IBS={ibs:.3f})")
        ax.set(xlabel="Time (months)", ylabel="IPCW Brier score", title=f"Prediction error - {cohort} cohort")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
        fig.tight_layout(); fig.savefig(OUT / f"Fig_IBS_{cohort}.pdf", bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(ibs_rows).to_csv(OUT / "Table_IBS_values.csv", index=False)
    pd.DataFrame(brier_rows).to_csv(OUT / "Table_Brier_Curves.csv", index=False)

    calibration_plot(tr, risks_tr, [12, 24, 36], OUT / "Fig_Calibration_Training_12m_24m_36m.pdf")
    # Also emit requested per-horizon files.
    for h in [12, 24, 36]: calibration_plot(tr, risks_tr, [h], OUT / f"Fig_Calibration_Training_{h}m.pdf")
    validation_calibration(va, tr, risks_tr, risks_va, OUT / "Fig_Calibration_Validation_Descriptive.pdf")
    dca_tr = dca_plot(tr, tr, risks_tr, risks_tr, 36, OUT / "Fig_DCA_Training_36m.pdf", validation=False)
    dca_va = dca_plot(tr, va, risks_tr, risks_va, 24, OUT / "Fig_DCA_Validation_24m_Bootstrap.pdf", validation=True)
    pd.concat([dca_tr, dca_va]).to_csv(OUT / "Table_DCA_NetBenefit.csv", index=False)

    pair = pd.concat([
        pairwise_bootstrap(tr["OS/m"].to_numpy(), tr["OS/label"].to_numpy(), risks_tr, "Training"),
        pairwise_bootstrap(va["OS/m"].to_numpy(), va["OS/label"].to_numpy(), risks_va, "Validation")
    ], ignore_index=True)
    pair.to_csv(OUT / "Table_Validation_Cindex_Pairwise_P.csv", index=False)

    # Formal nested-model LRTs use consistently refitted ordinary Cox models.
    nested = [
        ("Clinical", ["NLR", "Ki67", "Margin", "Myxoid"], "Clinical-ITH", definitions["Clinical-ITH"], "ITH"),
        ("HCR", hfeatures, "HCR-ITH", definitions["HCR-ITH"], "ITH"),
        ("Clinical-HCR", definitions["Clinical-HCR"], "Clinical-ITH-HCR", definitions["Clinical-ITH-HCR"], "ITH"),
        ("Clinical", ["NLR", "Ki67", "Margin", "Myxoid"], "Clinical-HCR", definitions["Clinical-HCR"], "HCR"),
        ("Clinical-ITH", definitions["Clinical-ITH"], "Clinical-ITH-HCR", definitions["Clinical-ITH-HCR"], "HCR"),
    ]
    lrt_rows = []
    for reduced_name, reduced_f, full_name, full_f, added in nested:
        red, *_ = fit_cox(tr, va, reduced_f); full, *_ = fit_cox(tr, va, full_f)
        stat = 2 * (full.log_likelihood_ - red.log_likelihood_); df = len(full_f) - len(reduced_f)
        lrt_rows.append({"Reduced_model": reduced_name, "Full_model": full_name, "Added_block": added,
                         "LR_chisq": stat, "df": df, "P": chi2.sf(stat, df),
                         "Note": "Training-only ordinary-Cox nested comparison; distinct from fixed penalized model performance"})
    pd.DataFrame(lrt_rows).to_csv(OUT / "Table_Incremental_Value_LRT.csv", index=False)

    # Prediction export and comprehensive table.
    for cohort, frame, risks in [("Training", tr, risks_tr), ("Validation", va, risks_va)]:
        pred = frame[["patient_id", "OS/m", "OS/label"]].copy()
        for name in MODEL_NAMES: pred[name] = risks[name]
        pred.to_csv(OUT / f"risk_predictions_{cohort.lower()}.csv", index=False)
    perf = cindex.merge(iauc[["Cohort", "Model", "iAUC_survival_weighted", "iAUC_time_average"]], on=["Cohort", "Model"])
    perf = perf.merge(pd.DataFrame(ibs_rows)[["Cohort", "Model", "IBS"]], on=["Cohort", "Model"])
    for t in [12, 24, 36, 60]:
        q = auc_detail.loc[auc_detail.Time_months.eq(t), ["Cohort", "Model", "AUC"]].rename(columns={"AUC": f"AUC_{t}m"})
        perf = perf.merge(q, on=["Cohort", "Model"], how="left")
    perf.to_csv(OUT / "Table_Performance_Summary.csv", index=False)

    model_rows = [
        {"Model": "Clinical", "Construction": "Previously selected lambda.min LASSO coefficients kept fixed", "Parameters": 4, "Features": "; ".join(CLIN_FIXED)},
        {"Model": "HCR", "Construction": "Final pure HCR LASSO coefficients kept fixed", "Parameters": len(hfeatures), "Features": "; ".join(hfeatures)},
    ]
    for name, feats in definitions.items():
        model_rows.append({"Model": name, "Construction": "Ordinary multivariable Cox fitted in training cohort", "Parameters": len(feats), "Features": "; ".join(feats)})
    pd.DataFrame(model_rows).to_csv(OUT / "Table_Model_Definitions.csv", index=False)

    val_pair = pair[pair.Cohort.eq("Validation")]
    report = f"""# Seven Cox models: comprehensive performance evaluation (authoritative ith_3clusters version)\n\n## Evidence hierarchy\nTraining cohort (N=129, 68 deaths) is the primary inferential cohort. Validation (N=22, 6 deaths) is exploratory because all performance and utility estimates are imprecise. Validation point estimates must not be used alone to claim superiority.\n\n## Reproducibility and leakage control\n- Random seed: {SEED}; paired bootstrap repetitions requested: {B}.\n- 3D-ITHscore was read exclusively from the user-designated TRAIN/TEST `ith_3clusters` columns.\n- Clinical and ITH records matched all 151 unique patient IDs; survival outcomes were identical.\n- Missing Clinical/semantic values were imputed using training-only medians/modes: {json.dumps(impute, ensure_ascii=False)}.\n- HCR preprocessing was reproduced from the final pure-HCR workflow. Maximum absolute score discrepancy versus saved scores: training={hdiff_tr:.3g}, validation={hdiff_va:.3g}.\n- The validation cohort was not used for model fitting, imputation, scaling, feature selection, or baseline-hazard estimation.\n\n## Metric windows\n- Training T-AUC/IBS/iAUC: 12-60 months; requested point AUCs at 12/24/36/60 months.\n- Validation primary T-AUC/IBS/iAUC: 12-24 months. The 36-month AUC is reported only as exploratory when computable; 60-month AUC is not extrapolated.\n- iAUC is not integrated from time zero or to the unstable last observation. Both survival-weighted cumulative/dynamic mean AUC and time-averaged AUC are supplied over the stated estimable window.\n\n## Calibration and DCA\n- Training calibration: four risk groups at 12/24/36 months.\n- Validation calibration: descriptive two-group display at 24 months.\n- Training DCA: 36 months. Validation DCA: 24 months with paired subject-level bootstrap 95% bands. Net benefit does not prove clinical utility in this small validation cohort.\n\n## Statistical inference\n- Harrell and Uno C-index confidence intervals use nonparametric bootstrap.\n- Pairwise C-index tests are paired bootstrap comparisons, with raw and BH-FDR-adjusted P values, separately by cohort.\n- Incremental-value likelihood-ratio tests are training-only and use consistently refitted ordinary nested Cox models. They are not computed by mixing fixed penalized and refitted likelihoods.\n\n## Principal limitation\nOnly six validation deaths make C-index differences, calibration, prediction error and DCA highly uncertain. This analysis is proof-of-concept/exploratory external validation and cannot establish robust generalizability.\n"""
    (OUT / "完整分析报告.md").write_text(report, encoding="utf-8")
    (OUT / "完整分析报告.txt").write_text(report, encoding="utf-8")
    summary = {"training_N": len(tr), "training_deaths": int(tr["OS/label"].sum()), "validation_N": len(va),
               "validation_deaths": int(va["OS/label"].sum()), "bootstrap": B, "ith_source": "user-designated TRAIN/TEST ith_3clusters",
               "hcr_features": hfeatures,
               "HCR_score_max_abs_diff_train": hdiff_tr, "HCR_score_max_abs_diff_validation": hdiff_va}
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != "output_md5_manifest.csv"]
    pd.DataFrame([{"file": p.name, "md5": md5(p), "bytes": p.stat().st_size} for p in sorted(files)]).to_csv(OUT / "output_md5_manifest.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
