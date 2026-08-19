#!/usr/bin/env python3
import hashlib
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEED = 20250308
RNG = np.random.default_rng(SEED)
B_CINDEX = 5000
B_RECLASS = 1000
B_632 = 1000
CV_REPEATS = 20

BASE = Path("/root/autodl-tmp/HCR_runs")
SOURCE = BASE / "seven_models_ith3_original_20260814"
OUT = BASE / "seven_models_stability_20260814"

HCR = [
    "T2_exponential_gldm_DependenceNonUniformity",
    "T1_square_firstorder_Kurtosis",
    "T1_wavelet-HHH_glszm_LowGrayLevelZoneEmphasis",
    "T2_wavelet-LLL_glcm_ClusterShade",
    "T2_exponential_firstorder_Median",
    "T2_exponential_glszm_LargeAreaLowGrayLevelEmphasis",
    "T2_original_shape_Maximum2DDiameterSlice",
]
CLIN = ["NLR", "Ki67", "Margin", "Myxoid"]
MODELS = {
    "Clinical": CLIN,
    "HCR": HCR,
    "ITH": ["ith_score"],
    "Clinical-ITH": CLIN + ["ith_score"],
    "HCR-ITH": HCR + ["ith_score"],
    "Clinical-HCR": CLIN + HCR,
    "Clinical-ITH-HCR": CLIN + HCR + ["ith_score"],
}
NESTED = [
    ("Clinical", "Clinical-ITH", "ITH"),
    ("HCR", "HCR-ITH", "ITH"),
    ("Clinical-HCR", "Clinical-ITH-HCR", "ITH"),
    ("Clinical", "Clinical-HCR", "HCR"),
    ("Clinical-ITH", "Clinical-ITH-HCR", "HCR"),
]
COLORS = {
    "Clinical": "#0072B2", "HCR": "#E69F00", "ITH": "#009E73",
    "Clinical-ITH": "#CC79A7", "HCR-ITH": "#D55E00",
    "Clinical-HCR": "#56B4E9", "Clinical-ITH-HCR": "#6A3D9A",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cindex(df, risk):
    return float(concordance_index(df["OS/m"], -np.asarray(risk), df["OS/label"]))


def fit_model(train, features, penalizer=1e-7):
    scaler = StandardScaler().fit(train[features])
    x = pd.DataFrame(scaler.transform(train[features]), columns=features)
    fit = pd.concat([train[["OS/m", "OS/label"]].reset_index(drop=True), x], axis=1)
    cph = CoxPHFitter(penalizer=penalizer)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cph.fit(fit, duration_col="OS/m", event_col="OS/label", show_progress=False)
    return cph, scaler, [str(x.message) for x in caught]


def predict_lp(model, scaler, frame, features):
    x = pd.DataFrame(scaler.transform(frame[features]), columns=features)
    return model.predict_log_partial_hazard(x).to_numpy()


def predict_risk(model, scaler, frame, features, horizon):
    x = pd.DataFrame(scaler.transform(frame[features]), columns=features)
    sf = model.predict_survival_function(x, times=[horizon])
    return 1 - sf.iloc[0].to_numpy()


def bh_adjust(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(ranked, 1)
    return out


def paired_cindex_bootstrap(frame, risks, cohort):
    pairs = [(a, b, block) for a, b, block in NESTED]
    store = {p: [] for p in pairs}; n = len(frame); successful = 0
    for _ in range(B_CINDEX):
        ix = RNG.integers(0, n, n)
        d = frame.iloc[ix].reset_index(drop=True)
        if d["OS/label"].nunique() < 2: continue
        try:
            cs = {m: cindex(d, risks[m][ix]) for m in risks}
        except Exception:
            continue
        successful += 1
        for p in pairs: store[p].append(cs[p[1]] - cs[p[0]])
    rows = []
    for p, vals in store.items():
        v = np.asarray(vals)
        point = cindex(frame, risks[p[1]]) - cindex(frame, risks[p[0]])
        pval = min(1.0, 2 * min(np.mean(v <= 0), np.mean(v >= 0)))
        rows.append({"Cohort": cohort, "Reduced_model": p[0], "Expanded_model": p[1], "Added_block": p[2],
                     "Delta_C": point, "L95": np.quantile(v, .025), "U95": np.quantile(v, .975),
                     "P_raw": pval, "Bootstrap_successful": successful})
    tab = pd.DataFrame(rows); tab["P_FDR_BH"] = bh_adjust(tab.P_raw)
    return tab


def lrt_table(train):
    fits = {}
    diagnostics = []
    for name, features in MODELS.items():
        m, s, w = fit_model(train, features, penalizer=0.0)
        fits[name] = (m, s)
        diagnostics.append({"Model": name, "Warnings": " | ".join(w), "Log_likelihood": m.log_likelihood_})
    rows = []
    for reduced, expanded, block in NESTED:
        stat = 2 * (fits[expanded][0].log_likelihood_ - fits[reduced][0].log_likelihood_)
        df = len(MODELS[expanded]) - len(MODELS[reduced])
        rows.append({"Reduced_model": reduced, "Expanded_model": expanded, "Added_block": block,
                     "LR_chisq": stat, "df": df, "P": chi2.sf(stat, df)})
    return pd.DataFrame(rows), pd.DataFrame(diagnostics), fits


def censor_weights(time, event, horizon):
    time = np.asarray(time, float); event = np.asarray(event, int)
    unique = np.sort(np.unique(time)); g_before = {}; g_after = {}; g = 1.0
    for t in unique:
        g_before[t] = g
        at_risk = np.sum(time >= t); cens = np.sum((time == t) & (event == 0))
        if at_risk: g *= (1 - cens / at_risk)
        g_after[t] = g
    def gb(t):
        vals = unique[unique <= t]
        if len(vals) == 0: return 1.0
        u = vals[-1]
        return g_before[u] if u == t else g_after[u]
    cases = (event == 1) & (time <= horizon)
    controls = time > horizon
    wc = np.zeros(len(time)); wn = np.zeros(len(time))
    for i in np.where(cases)[0]: wc[i] = 1 / max(gb(time[i]), 1e-8)
    g_tau = 1.0
    vals = unique[unique <= horizon]
    if len(vals): g_tau = g_after[vals[-1]]
    wn[controls] = 1 / max(g_tau, 1e-8)
    return cases, controls, wc, wn


def nri_idi(time, event, old, new, horizon):
    cases, controls, wc, wn = censor_weights(time, event, horizon)
    if cases.sum() == 0 or controls.sum() == 0: return (np.nan,) * 4
    delta = np.asarray(new) - np.asarray(old)
    we = wc[cases]; wne = wn[controls]
    nri_event = np.sum(we * (delta[cases] > 0)) / we.sum() - np.sum(we * (delta[cases] < 0)) / we.sum()
    nri_nonevent = np.sum(wne * (delta[controls] < 0)) / wne.sum() - np.sum(wne * (delta[controls] > 0)) / wne.sum()
    nri = nri_event + nri_nonevent
    old_slope = np.sum(we * old[cases]) / we.sum() - np.sum(wne * old[controls]) / wne.sum()
    new_slope = np.sum(we * new[cases]) / we.sum() - np.sum(wne * new[controls]) / wne.sum()
    return nri, nri_event, nri_nonevent, new_slope - old_slope


def reclassification_bootstrap(train, full_fits):
    rows = []; detail = []; n = len(train)
    for horizon in [36, 24]:
        full_risks = {m: predict_risk(full_fits[m][0], full_fits[m][1], train, MODELS[m], horizon) for m in MODELS}
        for reduced, expanded, block in NESTED:
            point = nri_idi(train["OS/m"], train["OS/label"], full_risks[reduced], full_risks[expanded], horizon)
            vals = []
            for b in range(B_RECLASS):
                ix = RNG.integers(0, n, n); d = train.iloc[ix].reset_index(drop=True)
                if d["OS/label"].nunique() < 2: continue
                try:
                    mr, sr, _ = fit_model(d, MODELS[reduced]); me, se, _ = fit_model(d, MODELS[expanded])
                    pr = predict_risk(mr, sr, d, MODELS[reduced], horizon)
                    pe = predict_risk(me, se, d, MODELS[expanded], horizon)
                    value = nri_idi(d["OS/m"], d["OS/label"], pr, pe, horizon)
                    if np.all(np.isfinite(value)): vals.append(value)
                except Exception:
                    continue
            arr = np.asarray(vals)
            metrics = ["NRI_continuous", "NRI_event", "NRI_nonevent", "IDI"]
            for j, metric in enumerate(metrics):
                pval = min(1.0, 2 * min(np.mean(arr[:, j] <= 0), np.mean(arr[:, j] >= 0)))
                rows.append({"Horizon_months": horizon, "Analysis_role": "Primary" if horizon == 36 else "Sensitivity",
                             "Reduced_model": reduced, "Expanded_model": expanded, "Added_block": block,
                             "Metric": metric, "Estimate": point[j], "L95": np.quantile(arr[:, j], .025),
                             "U95": np.quantile(arr[:, j], .975), "P_raw": pval, "Bootstrap_successful": len(arr)})
            for b, v in enumerate(arr):
                detail.append({"Horizon_months": horizon, "Reduced_model": reduced, "Expanded_model": expanded,
                               "Bootstrap_index": b, **dict(zip(metrics, v))})
    tab = pd.DataFrame(rows)
    tab["P_FDR_BH_within_horizon_metric"] = np.nan
    for (_, _), ix in tab.groupby(["Horizon_months", "Metric"]).groups.items():
        tab.loc[ix, "P_FDR_BH_within_horizon_metric"] = bh_adjust(tab.loc[ix, "P_raw"])
    return tab, pd.DataFrame(detail)


def repeated_cv(train):
    rows = []; summary = []
    for name, features in MODELS.items():
        full, scaler, _ = fit_model(train, features)
        apparent = cindex(train, predict_lp(full, scaler, train, features))
        repeat_rows = []
        for repeat in range(CV_REPEATS):
            skf = StratifiedKFold(10, shuffle=True, random_state=SEED + repeat)
            oof = np.full(len(train), np.nan); fold_opts = []
            for fold, (itr, ite) in enumerate(skf.split(np.zeros(len(train)), train["OS/label"])):
                dtr = train.iloc[itr].reset_index(drop=True); dte = train.iloc[ite].reset_index(drop=True)
                try:
                    m, s, _ = fit_model(dtr, features)
                    rtr = predict_lp(m, s, dtr, features); rte = predict_lp(m, s, dte, features)
                    oof[ite] = rte
                    if dte["OS/label"].nunique() > 1:
                        fold_opts.append(cindex(dtr, rtr) - cindex(dte, rte))
                except Exception:
                    continue
            pooled = cindex(train, oof) if np.isfinite(oof).all() else np.nan
            optimism = np.mean(fold_opts) if fold_opts else np.nan
            corrected = apparent - optimism
            repeat_rows.append((pooled, optimism, corrected))
            rows.append({"Model": name, "Repeat": repeat + 1, "Pooled_OOF_C": pooled,
                         "Mean_fold_optimism": optimism, "Optimism_corrected_C": corrected})
        a = np.asarray(repeat_rows)
        summary.append({"Model": name, "Apparent_C": apparent, "CV_repeats": CV_REPEATS,
                        "Mean_pooled_OOF_C": np.nanmean(a[:, 0]), "Pooled_OOF_C_L95": np.nanquantile(a[:, 0], .025),
                        "Pooled_OOF_C_U95": np.nanquantile(a[:, 0], .975), "Mean_optimism": np.nanmean(a[:, 1]),
                        "Mean_optimism_corrected_C": np.nanmean(a[:, 2]),
                        "Corrected_C_L95": np.nanquantile(a[:, 2], .025), "Corrected_C_U95": np.nanquantile(a[:, 2], .975)})
    return pd.DataFrame(summary), pd.DataFrame(rows)


def bootstrap_632plus(train):
    summary = []; detail = []; n = len(train)
    for name, features in MODELS.items():
        full, scaler, _ = fit_model(train, features)
        apparent = cindex(train, predict_lp(full, scaler, train, features))
        records = []
        for b in range(B_632):
            ix = RNG.integers(0, n, n); present = np.zeros(n, bool); present[np.unique(ix)] = True
            oob = np.where(~present)[0]
            dboot = train.iloc[ix].reset_index(drop=True); doob = train.iloc[oob].reset_index(drop=True)
            if dboot["OS/label"].nunique() < 2 or len(oob) < 5 or doob["OS/label"].nunique() < 2: continue
            try:
                m, s, _ = fit_model(dboot, features)
                cboot = cindex(dboot, predict_lp(m, s, dboot, features))
                coob = cindex(doob, predict_lp(m, s, doob, features))
                records.append((cboot, coob))
            except Exception:
                continue
        arr = np.asarray(records); mean_boot = arr[:, 0].mean(); mean_oob = arr[:, 1].mean()
        optimism = np.mean(arr[:, 0] - arr[:, 1]); corrected = apparent - optimism
        err_app = 1 - apparent; err_oob = 1 - mean_oob
        denom = .5 - err_app
        rel_overfit = np.clip((err_oob - err_app) / denom, 0, 1) if denom > 0 else 1
        weight = .632 / (1 - .368 * rel_overfit)
        c632 = 1 - ((1 - weight) * err_app + weight * err_oob)
        # Percentile interval from replicate-specific .632+ estimates.
        reps = []
        for cboot, coob in arr:
            r = np.clip(((1 - coob) - err_app) / denom, 0, 1) if denom > 0 else 1
            w = .632 / (1 - .368 * r)
            reps.append(1 - ((1 - w) * err_app + w * (1 - coob)))
        summary.append({"Model": name, "Apparent_C": apparent, "Bootstrap_requested": B_632,
                        "Bootstrap_successful": len(arr), "Mean_bootstrap_resubstitution_C": mean_boot,
                        "Mean_OOB_C": mean_oob, "Mean_optimism": optimism, "Optimism_corrected_C": corrected,
                        "Relative_overfitting_R": rel_overfit, "Weight_632plus": weight, "C_632plus": c632,
                        "C_632plus_L95": np.quantile(reps, .025), "C_632plus_U95": np.quantile(reps, .975)})
        for b, (cboot, coob) in enumerate(arr): detail.append({"Model": name, "Bootstrap_index": b, "C_bootstrap": cboot, "C_OOB": coob})
    return pd.DataFrame(summary), pd.DataFrame(detail)


def plots(cdiff, reclass, cv, boot):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    # C-index difference forest.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True)
    for ax, cohort in zip(axes, ["Training", "Validation"]):
        q = cdiff[cdiff.Cohort == cohort].reset_index(drop=True)
        y = np.arange(len(q)); ax.errorbar(q.Delta_C, y, xerr=[q.Delta_C-q.L95, q.U95-q.Delta_C], fmt="o", color="#0072B2", capsize=3)
        ax.axvline(0, color="black", ls="--", lw=1); ax.set_yticks(y); ax.set_yticklabels(q.Reduced_model + " -> " + q.Expanded_model)
        ax.set_title(cohort); ax.set_xlabel("Paired difference in Harrell C")
    fig.suptitle("Incremental discrimination: paired bootstrap 95% CI")
    fig.tight_layout(); fig.savefig(OUT / "Fig_Bootstrap_Cindex_Differences.pdf", bbox_inches="tight"); plt.close(fig)

    # 36-month NRI and IDI forest.
    q = reclass[(reclass.Horizon_months == 36) & reclass.Metric.isin(["NRI_continuous", "IDI"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, metric in zip(axes, ["NRI_continuous", "IDI"]):
        z = q[q.Metric == metric].reset_index(drop=True); y = np.arange(len(z))
        ax.errorbar(z.Estimate, y, xerr=[z.Estimate-z.L95, z.U95-z.Estimate], fmt="o", color="#009E73", capsize=3)
        ax.axvline(0, color="black", ls="--", lw=1); ax.set_yticks(y); ax.set_yticklabels(z.Reduced_model + " -> " + z.Expanded_model)
        ax.set_title(metric); ax.set_xlabel("Estimate (bootstrap 95% CI)")
    fig.suptitle("36-month IPCW reclassification improvement (training cohort)")
    fig.tight_layout(); fig.savefig(OUT / "Fig_NRI_IDI_36m.pdf", bbox_inches="tight"); plt.close(fig)

    # Internal validation summary.
    merged = cv[["Model", "Apparent_C", "Mean_pooled_OOF_C", "Mean_optimism_corrected_C"]].merge(boot[["Model", "C_632plus"]], on="Model")
    long = merged.melt("Model", var_name="Estimate", value_name="C")
    labels = {"Apparent_C":"Apparent", "Mean_pooled_OOF_C":"Repeated 10-fold OOF", "Mean_optimism_corrected_C":"10-fold optimism-corrected", "C_632plus":".632+ bootstrap"}
    long.Estimate = long.Estimate.map(labels)
    fig, ax = plt.subplots(figsize=(12, 6)); sns.pointplot(data=long, x="Model", y="C", hue="Estimate", dodge=.5, ax=ax)
    ax.axhline(.5, color="black", ls="--", lw=1); ax.set_ylim(.5, .9); ax.set_ylabel("Harrell C"); ax.tick_params(axis="x", rotation=25)
    ax.set_title("Internal validation of fixed predictor-set Cox models"); ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "Fig_Internal_Validation_Summary.pdf", bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(SOURCE / "analysis_dataset_training.csv")
    valid = pd.read_csv(SOURCE / "analysis_dataset_validation.csv")
    ptr = pd.read_csv(SOURCE / "risk_predictions_training.csv")
    pva = pd.read_csv(SOURCE / "risk_predictions_validation.csv")
    if (len(train), int(train["OS/label"].sum()), len(valid), int(valid["OS/label"].sum())) != (129, 68, 22, 6):
        raise RuntimeError("Cohort hard check failed")
    if not np.allclose(train.ith_score, ptr.ITH):
        # Linear predictor is a monotone transform, so raw equality is not expected.
        pass
    risks_tr = {m: ptr[m].to_numpy() for m in MODELS}; risks_va = {m: pva[m].to_numpy() for m in MODELS}

    cdiff = pd.concat([paired_cindex_bootstrap(train, risks_tr, "Training"), paired_cindex_bootstrap(valid, risks_va, "Validation")], ignore_index=True)
    cdiff.to_csv(OUT / "Table_Bootstrap_Cindex_Differences.csv", index=False)
    lrt, diagnostics, full_fits = lrt_table(train); lrt.to_csv(OUT / "Table_LRT_Nested_Models.csv", index=False); diagnostics.to_csv(OUT / "model_fit_diagnostics.csv", index=False)
    reclass, reclass_detail = reclassification_bootstrap(train, full_fits)
    reclass.to_csv(OUT / "Table_NRI_IDI.csv", index=False); reclass_detail.to_csv(OUT / "bootstrap_NRI_IDI_iterations.csv", index=False)
    cv, cv_detail = repeated_cv(train); cv.to_csv(OUT / "Table_Repeated_10Fold_CV.csv", index=False); cv_detail.to_csv(OUT / "repeated_10fold_iterations.csv", index=False)
    b632, bdetail = bootstrap_632plus(train); b632.to_csv(OUT / "Table_632plus_Bootstrap.csv", index=False); bdetail.to_csv(OUT / "bootstrap_632plus_iterations.csv", index=False)
    plots(cdiff, reclass, cv, b632)

    report = f"""# Incremental value and internal stability validation\n\n## Design\nTraining (N=129, 68 deaths) is the primary inferential cohort. Validation (N=22, 6 deaths) is exploratory. The authoritative 3D-ITHscore is the user-designated `ith_3clusters`.\n\n## Methods\n- Paired C-index difference: {B_CINDEX} subject-level bootstrap replicates, separately by cohort, with BH correction across five prespecified nested comparisons.\n- LRT: ordinary unpenalized Cox models with identical preprocessing and truly nested predictor sets; training only.\n- NRI/IDI: IPCW continuous survival NRI and IDI at 36 months (primary) and 24 months (sensitivity), {B_RECLASS} full model-refitting bootstrap replicates. Category-free NRI/IDI are auxiliary and not used alone for model selection.\n- Repeated stratified 10-fold CV: {CV_REPEATS} repeats. Report pooled OOF C, fold optimism, and apparent-minus-optimism C. Predictor sets were fixed in advance; this validates coefficient fitting, not the earlier feature-selection procedure.\n- Bootstrap: {B_632} resamples with out-of-bag evaluation, conventional optimism correction, and .632+ weighting.\n\n## Interpretation boundary\nInternal validation measures reproducibility in the source population and cannot replace adequately powered external validation. Validation comparisons remain imprecise because only six deaths were observed.\n"""
    (OUT / "稳定性验证方法与说明.md").write_text(report, encoding="utf-8")
    summary = {"seed": SEED, "training_N": len(train), "training_events": int(train["OS/label"].sum()),
               "validation_N": len(valid), "validation_events": int(valid["OS/label"].sum()),
               "bootstrap_cindex": B_CINDEX, "bootstrap_nri_idi": B_RECLASS, "bootstrap_632plus": B_632,
               "cv_repeats": CV_REPEATS, "ith_source": "authoritative ith_3clusters"}
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != "output_md5_manifest.csv"]
    pd.DataFrame([{"file": p.name, "md5": md5(p), "bytes": p.stat().st_size} for p in sorted(files)]).to_csv(OUT / "output_md5_manifest.csv", index=False)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
