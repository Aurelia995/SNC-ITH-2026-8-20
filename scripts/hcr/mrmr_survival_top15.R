#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2) stop('usage: mrmr_survival_top15.R input.csv output_dir')
suppressPackageStartupMessages({library(survival); library(mRMRe)})
set.seed(20250308)
d <- read.csv(args[1], check.names=FALSE)
X <- d[, setdiff(names(d), c('time','event')), drop=FALSE]
for (j in seq_along(X)) X[[j]] <- as.numeric(X[[j]])
mdat <- mRMR.data(data=data.frame(target=Surv(as.numeric(d$time), as.numeric(d$event)), X, check.names=FALSE))
fit <- mRMR.classic(data=mdat, target_indices=1, feature_count=min(15,ncol(X)))
idx <- solutions(fit)[[1]]; nm <- featureNames(mdat); selected <- nm[idx]; selected <- selected[selected!='target']
writeLines(head(selected,15), file.path(args[2],'mRMR_selected_15.txt'), useBytes=TRUE)
sc <- scores(fit)[[1]]
write.csv(data.frame(rank=seq_along(idx),feature=nm[idx],score=as.numeric(sc),check.names=FALSE),file.path(args[2],'mRMR_scores_selected.csv'),row.names=FALSE)
cat('mRMRe selected',length(selected),'features\n')
