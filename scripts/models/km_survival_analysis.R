options(stringsAsFactors = FALSE)
suppressPackageStartupMessages({
  library(survival)
  library(survminer)
  library(ggplot2)
})

set.seed(20250308)

source_dir <- "/root/autodl-tmp/HCR_runs/seven_models_ith3_original_20260814"
stability_dir <- "/root/autodl-tmp/HCR_runs/seven_models_stability_20260814"
out_dir <- "/root/autodl-tmp/HCR_runs/seven_models_survival_20260814"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

read_csv_checked <- function(path) {
  if (!file.exists(path)) stop("Missing input: ", path)
  read.csv(path, check.names = FALSE)
}

perf <- read_csv_checked(file.path(source_dir, "Table_Performance_Summary.csv"))
defs <- read_csv_checked(file.path(source_dir, "Table_Model_Definitions.csv"))
cv <- read_csv_checked(file.path(stability_dir, "Table_Repeated_10Fold_CV.csv"))
b632 <- read_csv_checked(file.path(stability_dir, "Table_632plus_Bootstrap.csv"))
pred_train <- read_csv_checked(file.path(source_dir, "risk_predictions_training.csv"))
pred_val <- read_csv_checked(file.path(source_dir, "risk_predictions_validation.csv"))

models <- c("Clinical", "HCR", "ITH", "Clinical-ITH", "HCR-ITH",
            "Clinical-HCR", "Clinical-ITH-HCR")
stopifnot(all(models %in% pred_train |> names()), all(models %in% pred_val |> names()))
stopifnot(nrow(pred_train) == 129, sum(pred_train[["OS/label"]]) == 68)
stopifnot(nrow(pred_val) == 22, sum(pred_val[["OS/label"]]) == 6)

train_perf <- perf[perf$Cohort == "Training", ]
val_perf <- perf[perf$Cohort == "Validation", ]
pick <- function(d, col) setNames(d[[col]], d$Model)[models]

selection <- data.frame(
  Model = models,
  Parameters = setNames(defs$Parameters, defs$Model)[models],
  Train_Harrell_C = pick(train_perf, "Harrell_C"),
  Train_Uno_C = pick(train_perf, "Uno_C"),
  Train_iAUC = pick(train_perf, "iAUC_survival_weighted"),
  Train_IBS = pick(train_perf, "IBS"),
  Repeated10fold_OOF_C = setNames(cv$Mean_pooled_OOF_C, cv$Model)[models],
  CV_optimism_corrected_C = setNames(cv$Mean_optimism_corrected_C, cv$Model)[models],
  Bootstrap_632plus_C = setNames(b632$C_632plus, b632$Model)[models],
  Validation_Harrell_C = pick(val_perf, "Harrell_C"),
  Validation_Uno_C = pick(val_perf, "Uno_C"),
  Validation_iAUC_12_24m = pick(val_perf, "iAUC_survival_weighted"),
  Validation_IBS_12_24m = pick(val_perf, "IBS")
)

# Prespecified selection hierarchy: internal validation is primary; external values
# are descriptive because validation has only six events. A model is competitive if
# both its repeated-CV OOF C and .632+ C are within 0.01 of their respective maxima.
selection$Within_0.01_best_OOF <-
  selection$Repeated10fold_OOF_C >= max(selection$Repeated10fold_OOF_C) - 0.01
selection$Within_0.01_best_632plus <-
  selection$Bootstrap_632plus_C >= max(selection$Bootstrap_632plus_C) - 0.01
selection$Primary_internal_candidate <-
  selection$Within_0.01_best_OOF & selection$Within_0.01_best_632plus
candidates <- selection[selection$Primary_internal_candidate, ]
best_model <- candidates$Model[which.min(candidates$Parameters)]
if (length(best_model) != 1L) stop("Selection rule did not yield one model")
selection$Selected <- ifelse(selection$Model == best_model, "Yes", "No")
write.csv(selection, file.path(out_dir, "Table_Model_Selection.csv"), row.names = FALSE)

# Use the training-fitted linear predictor already generated in the authoritative
# seven-model analysis; do not refit on validation or pooled data.
make_scores <- function(d, cohort) {
  data.frame(patient_id=d$patient_id, Cohort=cohort,
             OS_time=d[["OS/m"]], OS_event=d[["OS/label"]],
             Model=best_model, Risk_score=d[[best_model]])
}
scores_train <- make_scores(pred_train, "Training")
scores_val <- make_scores(pred_val, "Validation")

# X-tile principle: maximize the two-group log-rank chi-square in training only.
# Restrict both groups to >=20% of training subjects to avoid pathological tiny groups.
n <- nrow(scores_train)
min_n <- ceiling(0.20 * n)
candidate_cutoffs <- sort(unique(scores_train$Risk_score))
cut_rows <- lapply(candidate_cutoffs, function(cut) {
  grp <- ifelse(scores_train$Risk_score > cut, "High risk", "Low risk")
  tab <- table(grp)
  n_low <- ifelse("Low risk" %in% names(tab), tab[["Low risk"]], 0)
  n_high <- ifelse("High risk" %in% names(tab), tab[["High risk"]], 0)
  if (n_low < min_n || n_high < min_n) return(NULL)
  sd <- survdiff(Surv(OS_time, OS_event) ~ grp, data=scores_train, rho=0)
  chisq <- unname(sd$chisq)
  data.frame(Cutoff=cut, N_low=n_low, N_high=n_high,
             Logrank_chisq=chisq, Logrank_P=pchisq(chisq, 1, lower.tail=FALSE))
})
cut_search <- do.call(rbind, cut_rows)
if (is.null(cut_search) || nrow(cut_search) == 0) stop("No eligible cutpoint")
cut_search <- cut_search[order(-cut_search$Logrank_chisq, cut_search$Cutoff), ]
best_cut <- cut_search$Cutoff[1]
cut_search$Selected <- ifelse(cut_search$Cutoff == best_cut, "Yes", "No")
write.csv(cut_search, file.path(out_dir, "Table_Cutpoint_Search.csv"), row.names=FALSE)

apply_cut <- function(d) {
  d$Risk_group <- factor(ifelse(d$Risk_score > best_cut, "High risk", "Low risk"),
                         levels=c("Low risk", "High risk"))
  d
}
scores_train <- apply_cut(scores_train)
scores_val <- apply_cut(scores_val)
scores_all <- apply_cut(rbind(scores_train[, setdiff(names(scores_train), "Risk_group")],
                             scores_val[, setdiff(names(scores_val), "Risk_group")]))
scores_all$Cohort <- "All"
write.csv(scores_train, file.path(out_dir, "Risk_scores_training.csv"), row.names=FALSE)
write.csv(scores_val, file.path(out_dir, "Risk_scores_validation.csv"), row.names=FALSE)
write.csv(scores_all, file.path(out_dir, "Risk_scores_all_cohort.csv"), row.names=FALSE)

fmt_p <- function(p) ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))
km_stats <- function(d, cohort) {
  tab <- table(d$Risk_group)
  ev <- tapply(d$OS_event, d$Risk_group, sum)
  sf <- survfit(Surv(OS_time, OS_event) ~ Risk_group, data=d)
  med <- summary(sf)$table
  if (is.null(dim(med))) med <- matrix(med, nrow=1)
  median_low <- med[grep("Low risk", rownames(med)), "median"]
  median_high <- med[grep("High risk", rownames(med)), "median"]
  sd <- survdiff(Surv(OS_time, OS_event) ~ Risk_group, data=d, rho=0)
  lp <- pchisq(sd$chisq, 1, lower.tail=FALSE)
  fit <- coxph(Surv(OS_time, OS_event) ~ Risk_group, data=d, ties="efron")
  cs <- summary(fit)
  hr <- unname(cs$conf.int[1, "exp(coef)"])
  lo <- unname(cs$conf.int[1, "lower .95"])
  hi <- unname(cs$conf.int[1, "upper .95"])
  data.frame(Cohort=cohort, Model=best_model, Cutoff=best_cut,
             N=nrow(d), Events=sum(d$OS_event),
             N_low=unname(tab["Low risk"]), Events_low=unname(ev["Low risk"]),
             N_high=unname(tab["High risk"]), Events_high=unname(ev["High risk"]),
             Logrank_chisq=unname(sd$chisq), Logrank_P=lp,
             HR_high_vs_low=hr, HR_L95=lo, HR_U95=hi,
             Median_OS_low=median_low, Median_OS_high=median_high)
}
stats <- rbind(km_stats(scores_train, "Training"),
               km_stats(scores_val, "Validation"),
               km_stats(scores_all, "All cohort"))
write.csv(stats, file.path(out_dir, "Table_KM_Survival_Statistics.csv"), row.names=FALSE)

median_text <- function(x) ifelse(is.na(x), "NR", sprintf("%.1f mo", x))
plot_km <- function(d, cohort, filename) {
  st <- stats[stats$Cohort == cohort, ]
  fit <- survfit(Surv(OS_time, OS_event) ~ Risk_group, data=d)
  ann <- sprintf("Log-rank P %s\nHR = %.2f (95%% CI %.2f-%.2f)\nMedian OS: low %s; high %s",
                 ifelse(st$Logrank_P < 0.001, "< 0.001", paste0("= ", sprintf("%.3f", st$Logrank_P))),
                 st$HR_high_vs_low, st$HR_L95, st$HR_U95,
                 median_text(st$Median_OS_low), median_text(st$Median_OS_high))
  g <- ggsurvplot(
    fit, data=d, risk.table=TRUE, conf.int=TRUE, pval=FALSE,
    pval.method=FALSE, palette=c("#4DBBD5", "#E64B35"),
    legend.title="Risk group", legend.labs=c("Low risk", "High risk"),
    xlim=c(0,60), break.time.by=12, xlab="Time (months)",
    ylab="Overall survival probability", surv.median.line="hv",
    risk.table.height=0.25, risk.table.y.text.col=TRUE,
    risk.table.y.text=FALSE, ggtheme=theme_classic(base_size=12),
    tables.theme=theme_cleantable(base_size=11), censor.shape=3, censor.size=2.5
  )
  g$plot <- g$plot +
    ggtitle(paste0(best_model, " risk stratification - ", cohort)) +
    annotate("text", x=58, y=0.94, label=ann, hjust=1, vjust=1, size=3.6) +
    theme(plot.title=element_text(hjust=0.5, face="bold"),
          legend.position=c(0.18,0.18))
  pdf(file.path(out_dir, filename), width=8.2, height=7.2, useDingbats=FALSE)
  print(g, newpage=FALSE)
  dev.off()
}

plot_km(scores_train, "Training", "Fig_KM_Training.pdf")
plot_km(scores_val, "Validation", "Fig_KM_Validation.pdf")
plot_km(scores_all, "All cohort", "Fig_KM_All_Cohort.pdf")

writeLines(c(
  "=== Optimal model and KM survival analysis ===",
  paste0("Seed: 20250308"),
  paste0("Selected model: ", best_model),
  paste0("Selection rule: within 0.01 of best repeated 10-fold OOF C and best .632+ C; then fewest parameters."),
  paste0("Training-derived X-tile-style cutoff: ", format(best_cut, digits=12)),
  paste0("Eligible cutpoints: ", nrow(cut_search), "; minimum group size: ", min_n, " (20% of training N)."),
  "The cutoff was selected using training outcomes only and applied unchanged to validation and pooled cohorts.",
  "Validation results are descriptive because N=22 with 6 events.",
  "The training log-rank P value is cutpoint-selection optimistic and should not be treated as independent confirmation."
), file.path(out_dir, "analysis_report.txt"))

writeLines(c(
  "# 七模型最优模型筛选与KM风险分层方法说明",
  "",
  paste0("- 最优模型：", best_model),
  "- 选择原则：以训练集重复10折OOF C及.632+ Bootstrap C为主要内部验证指标；两项指标均距最优值不超过0.01者进入候选，然后选择参数更少的模型。验证集仅6个事件，其指标只作描述，不参与主导选择。",
  paste0("- 风险评分：直接使用此前训练集拟合模型产生的线性预测值，不在验证集或全队列重新拟合。"),
  paste0("- 截点：在训练集中搜索，要求每组至少占20%，最大化两组log-rank chi-square；最终截点=", format(best_cut, digits=12), "。"),
  "- 应用：训练集确定的同一截点原样用于验证集及全队列。",
  "- KM曲线：survival + survminer::ggsurvplot；显示95%CI、风险表、log-rank P、二分类Cox HR及中位OS，横轴限制0-60个月。",
  "- 局限：训练集log-rank P因结局驱动截点选择而偏乐观；全队列包含训练病例，也不是独立验证；验证集结果因样本量和事件数有限应谨慎解释。"
), file.path(out_dir, "方法与结果说明.md"))

files <- list.files(out_dir, full.names=TRUE)
md5 <- tools::md5sum(files)
manifest <- data.frame(file=basename(names(md5)), md5=unname(md5),
                       bytes=file.info(names(md5))$size)
write.csv(manifest, file.path(out_dir, "output_md5_manifest.csv"), row.names=FALSE)

cat(sprintf("DONE best_model=%s cutoff=%.12g files=%d\n", best_model, best_cut,
            length(list.files(out_dir))))
