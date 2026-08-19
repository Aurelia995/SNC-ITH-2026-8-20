options(stringsAsFactors = FALSE, warn = 1)
set.seed(20250308)
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(glmnet))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript clinical_model_analysis.R input.csv output_dir")
input_csv <- args[1]
outdir <- args[2]
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
log_con <- file(file.path(outdir, "analysis_console.log"), open = "wt")
sink(log_con, type = "output", split = TRUE)
sink(log_con, type = "message", append = TRUE)
on.exit({sink(type="message"); sink(type="output"); close(log_con)}, add=TRUE)
cat("EXP03/EXP04 reduced clinical + semantic MRI model analysis; no training-set internal validation\nSeed: 20250308\n")
cat("R:", R.version.string, "\n")
cat("survival:", as.character(packageVersion("survival")), " glmnet:", as.character(packageVersion("glmnet")), "\n")

raw <- read.csv(input_csv, check.names = FALSE, na.strings = c("", "NA"))
if (ncol(raw) != 23L) stop("Unexpected number of columns: ", ncol(raw))
original_names <- names(raw)
safe_names <- c("patient_id","dataset","event","time","stage","T_stage","N_stage","sex","age","smoking",
                "NLR","PLR","LMR","site","histology","Ki67","size_gt5",
                "t2_heterogeneity","clear_margin","myxoid_change","necrosis","septation","enhancement")
names(raw) <- safe_names
raw$dataset <- factor(raw$dataset, levels=c("Train","Test"))
raw$event <- as.integer(raw$event)
raw$time <- as.numeric(raw$time)
if (any(!raw$event %in% c(0L,1L)) || any(raw$time <= 0, na.rm=TRUE)) stop("Invalid survival outcome coding")
if (anyDuplicated(raw$patient_id)) stop("Duplicate patient_id detected")
train0 <- raw[raw$dataset == "Train", , drop=FALSE]
test0 <- raw[raw$dataset == "Test", , drop=FALSE]
cat("N/event Train:", nrow(train0), sum(train0$event), " Test:", nrow(test0), sum(test0$event), "\n")

force_vars <- c("stage","sex","age","NLR","site","histology","Ki67")
semantic_vars <- c("size_gt5","t2_heterogeneity","clear_margin","myxoid_change","necrosis","septation","enhancement")
candidate_vars <- c(force_vars, semantic_vars)
continuous_vars <- c("age","NLR","Ki67")
categorical_vars <- c("stage","sex","site","histology",semantic_vars)
variable_labels <- c(stage="TNM overall stage", sex="Sex", age="Age",
                     NLR="NLR", site="Primary site",
                     histology="Histologic subtype", Ki67="Ki-67", size_gt5="Tumor size >5 cm",
                     t2_heterogeneity="T2 heterogeneous signal (>=50%)", clear_margin="Clear margin",
                     myxoid_change="Myxoid change (>=10%)", necrosis="Necrosis (>=10%)",
                     septation="Septation", enhancement="Enhancement")
stage_levels <- c("II","III","IVA","IVB")
hist_levels <- c("Squamous cell carcinoma","Adenocarcinoma","Undifferentiated carcinoma","SMARCB1-deficient carcinoma")

missing_table <- do.call(rbind, lapply(c("Train","Test","Overall"), function(coh) {
  dd <- if (coh=="Train") train0 else if (coh=="Test") test0 else raw
  data.frame(cohort=coh, variable=names(dd), missing_n=sapply(dd, function(x)sum(is.na(x))),
             missing_pct=sapply(dd, function(x)mean(is.na(x))), row.names=NULL)
}))
write.csv(missing_table, file.path(outdir,"missingness_by_cohort.csv"), row.names=FALSE, fileEncoding="UTF-8")

reverse_km_median <- function(time,event) {
  fit <- survfit(Surv(time, 1-event) ~ 1)
  z <- summary(fit)$table
  as.numeric(z["median"])
}

cohort_qc <- do.call(rbind, lapply(list(Train=train0, Validation=test0), function(dd) {
  data.frame(cohort=ifelse(identical(dd,train0),"Training","Validation"), N=nrow(dd), deaths=sum(dd$event),
             censored=sum(dd$event==0), median_followup_months=reverse_km_median(dd$time,dd$event),
             candidate_parameters=18, EPV=sum(dd$event)/18)
}))
write.csv(cohort_qc, file.path(outdir,"Table_A_data_quality_events.csv"), row.names=FALSE)

mode_value <- function(x) {
  ux <- sort(unique(x[!is.na(x)])); ux[which.max(tabulate(match(x,ux)))]
}

fit_imputation_params <- function(dd) {
  list(median=setNames(sapply(continuous_vars, function(v)median(dd[[v]],na.rm=TRUE)),continuous_vars),
       mode=setNames(sapply(categorical_vars, function(v)mode_value(dd[[v]])),categorical_vars))
}
imp_params <- fit_imputation_params(train0)
write.csv(data.frame(variable=c(continuous_vars,categorical_vars),
                     statistic=c(rep("median",length(continuous_vars)),rep("mode",length(categorical_vars))),
                     value=c(imp_params$median,imp_params$mode)), file.path(outdir,"training_imputation_parameters.csv"), row.names=FALSE)

deterministic_impute <- function(dd, params) {
  out <- dd
  for(v in continuous_vars) out[[v]][is.na(out[[v]])] <- params$median[[v]]
  for(v in categorical_vars) out[[v]][is.na(out[[v]])] <- params$mode[[v]]
  out
}
train <- deterministic_impute(train0, imp_params)
test <- deterministic_impute(test0, imp_params)

prep_factors <- function(dd, stage_categorical=TRUE) {
  z <- dd
  if(stage_categorical) z$stage <- factor(z$stage, levels=2:5, labels=stage_levels) else z$stage <- as.numeric(z$stage)-2
  z$sex <- factor(z$sex, levels=c(0,1), labels=c("Female","Male"))
  z$smoking <- factor(z$smoking, levels=c(0,1), labels=c("No","Yes"))
  z$site <- factor(z$site, levels=c(1,2), labels=c("Nasal cavity","Paranasal sinus"))
  z$histology <- factor(z$histology, levels=1:4, labels=hist_levels)
  for(v in semantic_vars[semantic_vars != "enhancement"]) z[[v]] <- factor(z[[v]], levels=c(0,1), labels=c("No","Yes"))
  z$enhancement <- factor(z$enhancement, levels=c(1,2), labels=c("Mild/moderate","Marked"))
  z
}

# Chained stochastic predictive mean matching, baseline variables only; no outcome-derived predictors.
stochastic_impute_once <- function(dd, seed, n_iter=10L, k=5L) {
  set.seed(seed)
  out <- deterministic_impute(dd, fit_imputation_params(dd))
  miss <- lapply(candidate_vars, function(v)which(is.na(dd[[v]]))); names(miss) <- candidate_vars
  pred_vars <- candidate_vars
  for(iter in seq_len(n_iter)) for(v in candidate_vars) {
    idx_mis <- miss[[v]]; if(!length(idx_mis)) next
    idx_obs <- which(!is.na(dd[[v]]))
    others <- setdiff(pred_vars,v)
    dat <- prep_factors(out, stage_categorical=TRUE)
    if(v %in% continuous_vars) {
      f <- reformulate(others, response=v)
      fit <- try(lm(f, data=dat[idx_obs,,drop=FALSE]), silent=TRUE)
      if(inherits(fit,"try-error")) {out[[v]][idx_mis] <- sample(dd[[v]][idx_obs],length(idx_mis),replace=TRUE); next}
      beta <- coef(fit); V <- try(vcov(fit),silent=TRUE)
      if(!inherits(V,"try-error") && all(is.finite(V))) {
        beta <- as.numeric(MASS::mvrnorm(1, mu=beta, Sigma=V)); names(beta) <- names(coef(fit))
      }
      Xobs <- model.matrix(delete.response(terms(fit)), dat[idx_obs,,drop=FALSE])
      Xmis <- model.matrix(delete.response(terms(fit)), dat[idx_mis,,drop=FALSE])
      common <- intersect(colnames(Xobs),names(beta)); po <- as.numeric(Xobs[,common,drop=FALSE] %*% beta[common]); pm <- as.numeric(Xmis[,common,drop=FALSE] %*% beta[common])
      for(j in seq_along(idx_mis)) {near <- order(abs(po-pm[j]))[seq_len(min(k,length(po)))]; donor <- sample(near,1); out[[v]][idx_mis[j]] <- dd[[v]][idx_obs[donor]]}
    } else {
      vals <- dd[[v]][idx_obs]
      if(length(unique(vals))==2L) {
        f <- reformulate(others, response=v); fit <- try(glm(f,data=dat[idx_obs,,drop=FALSE],family=binomial()),silent=TRUE)
        if(inherits(fit,"try-error")) p <- rep(mean(vals==max(vals)),length(idx_mis)) else p <- pmin(pmax(predict(fit,newdata=dat[idx_mis,,drop=FALSE],type="response"),0.001),0.999)
        lev <- sort(unique(vals)); out[[v]][idx_mis] <- ifelse(runif(length(idx_mis))<p,lev[2],lev[1])
      } else out[[v]][idx_mis] <- sample(vals,length(idx_mis),replace=TRUE)
    }
  }
  out
}

mi_list <- lapply(seq_len(20), function(m) stochastic_impute_once(train0, 20250308+m))
saveRDS(mi_list, file.path(outdir,"multiple_imputation_datasets_m20.rds"))

force_formula_A <- Surv(time,event) ~ stage + sex + age + NLR + site + histology + Ki67
force_formula_B <- force_formula_A
fit_force <- function(dd, model=c("A","B")) {
  model <- match.arg(model); z <- prep_factors(dd, stage_categorical=(model=="A"))
  coxph(if(model=="A") force_formula_A else force_formula_B, data=z, ties="efron", x=TRUE, y=TRUE)
}
fits_A <- lapply(mi_list, fit_force, model="A")
fits_B <- lapply(mi_list, fit_force, model="B")

pool_cox <- function(fits, model_name) {
  common <- Reduce(intersect,lapply(fits,function(f)names(coef(f))))
  Bmat <- do.call(rbind,lapply(fits,function(f)coef(f)[common]))
  U <- lapply(fits,function(f)vcov(f)[common,common,drop=FALSE])
  qbar <- colMeans(Bmat); ubar <- Reduce(`+`,U)/length(U); b <- cov(Bmat)
  Tmat <- ubar + (1+1/length(U))*b
  se <- sqrt(diag(Tmat)); z <- qbar/se; p <- 2*pnorm(-abs(z))
  data.frame(model=model_name, term=common, beta=qbar, HR=exp(qbar), CI_low=exp(qbar-1.96*se),
             CI_high=exp(qbar+1.96*se), P=p, stringsAsFactors=FALSE,
             row.names=NULL, check.names=FALSE)
}
strip_cov <- function(x) x
poolA <- pool_cox(fits_A,"Model A: categorical overall TNM")
poolB <- pool_cox(fits_B,"Model B: ordinal overall TNM")

group_map <- function(terms, model) {
  out <- rep(NA_character_,length(terms))
  out[grepl("^stage",terms)] <- "stage"; out[grepl("^sex",terms)] <- "sex"; out[terms=="age"] <- "age"
  out[terms=="NLR"] <- "NLR"
  out[grepl("^site",terms)] <- "site"; out[grepl("^histology",terms)] <- "histology"; out[terms=="Ki67"] <- "Ki67"; out
  out[grepl("^size_gt5",terms)] <- "size_gt5"; out[grepl("^t2_heterogeneity",terms)] <- "t2_heterogeneity"
  out[grepl("^clear_margin",terms)] <- "clear_margin"; out[grepl("^myxoid_change",terms)] <- "myxoid_change"
  out[grepl("^necrosis",terms)] <- "necrosis"; out[grepl("^septation",terms)] <- "septation"; out[grepl("^enhancement",terms)] <- "enhancement"; out
}
global_tests <- function(pool_obj, fits, model_name) {
  terms <- pool_obj$term; groups <- group_map(terms,model_name); Bmat <- do.call(rbind,lapply(fits,function(f)coef(f)[terms])); U <- lapply(fits,function(f)vcov(f)[terms,terms,drop=FALSE])
  qbar <- colMeans(Bmat); Tmat <- Reduce(`+`,U)/length(U)+(1+1/length(U))*cov(Bmat)
  do.call(rbind,lapply(unique(groups),function(g){ix=which(groups==g); stat=as.numeric(t(qbar[ix])%*%qr.solve(Tmat[ix,ix,drop=FALSE],qbar[ix])); data.frame(variable=g,df=length(ix),global_P=pchisq(stat,length(ix),lower.tail=FALSE))}))
}
globA <- global_tests(poolA,fits_A,"A"); globB <- global_tests(poolB,fits_B,"B")

detA <- fit_force(train,"A"); detB <- fit_force(train,"B")
phA <- cox.zph(detA, transform="km")$table; phB <- cox.zph(detB, transform="km")$table
ph_term <- function(term, ph) {
  rn <- rownames(ph); base <- group_map(term,""); v <- base[1]; hit <- which(group_map(rn,"")==v); if(length(hit)) ph[hit[1],"p"] else NA_real_
}
make_force_table <- function(pool,glob,ph,model_label) {
  x <- strip_cov(pool); x$variable_key <- group_map(x$term,model_label); x$variable <- unname(variable_labels[x$variable_key]); x$level_reference <- x$term
  x$global_P <- glob$global_P[match(x$variable_key,glob$variable)]
  x$PH_test_P <- sapply(x$term,ph_term,ph=ph)
  x[,c("model","variable","level_reference","beta","HR","CI_low","CI_high","P","global_P","PH_test_P")]
}
force_table <- rbind(make_force_table(poolA,globA,phA,"A"),make_force_table(poolB,globB,phB,"B"))
write.csv(force_table,file.path(outdir,"Table_B_force_entry_cox.csv"),row.names=FALSE)
write.csv(rbind(transform(globA,model="Model A: categorical overall TNM"),transform(globB,model="Model B: ordinal overall TNM")),file.path(outdir,"force_entry_global_tests.csv"),row.names=FALSE)
saveRDS(list(fits_A=fits_A,fits_B=fits_B,pool_A=poolA,pool_B=poolB,global_A=globA,global_B=globB),file.path(outdir,"force_entry_MI_models.rds"))

indA <- globA$variable[globA$global_P<0.05]
indB <- globB$variable[globB$global_P<0.05]
independent_stable <- intersect(indA,indB)
independent_any <- union(indA,indB)
cat("Independent factors Model A:",paste(indA,collapse=", "),"\n")
cat("Independent factors Model B:",paste(indB,collapse=", "),"\n")

lasso_formula <- ~ stage + sex + age + NLR + site + histology + Ki67 + size_gt5 + t2_heterogeneity + clear_margin + myxoid_change + necrosis + septation + enhancement
build_design <- function(dd) {
  z <- prep_factors(dd,stage_categorical=TRUE)
  model.matrix(lasso_formula,z)[,-1,drop=FALSE]
}
x_train <- build_design(train); x_test <- build_design(test)
y_train <- Surv(train$time,train$event); y_test <- Surv(test$time,test$event)
if(!identical(colnames(x_train),colnames(x_test))) stop("Design column mismatch")
design_dictionary <- data.frame(design_column=colnames(x_train),variable_key=group_map(colnames(x_train),"A"),stringsAsFactors=FALSE)
write.csv(design_dictionary,file.path(outdir,"design_matrix_dictionary.csv"),row.names=FALSE)

make_foldid <- function(event,k=10L,seed=20250308) {
  set.seed(seed); id <- integer(length(event)); for(ev in c(0,1)){ix <- sample(which(event==ev)); id[ix] <- rep(seq_len(k),length.out=length(ix))}; id
}
foldid <- make_foldid(train$event)
write.csv(data.frame(patient_id=train$patient_id,event=train$event,foldid=foldid),file.path(outdir,"lasso_fold_assignments.csv"),row.names=FALSE)
cv_main <- cv.glmnet(x_train,y_train,family="cox",alpha=1,nfolds=10,foldid=foldid,type.measure="deviance",standardize=TRUE,keep=TRUE)
fit_main <- cv_main$glmnet.fit

coef_at <- function(cv,s) {
  b <- as.matrix(coef(cv,s=s)); data.frame(term=rownames(b),coefficient=as.numeric(b[,1]),stringsAsFactors=FALSE)
}
c1 <- coef_at(cv_main,"lambda.1se"); cmin <- coef_at(cv_main,"lambda.min")
selected_1se <- c1$term[c1$coefficient!=0]; selected_min <- cmin$term[cmin$coefficient!=0]
cat("lambda.1se:",cv_main$lambda.1se," lambda.min:",cv_main$lambda.min,"\n")
cat("Main LASSO 1se:",paste(selected_1se,collapse=", "),"\n")

suppressed <- character(0)
for(g in independent_any) {cols <- design_dictionary$design_column[design_dictionary$variable_key==g]; if(length(cols)&&all(c1$coefficient[match(cols,c1$term)]==0)) suppressed <- union(suppressed,g)}
penalty <- rep(1,ncol(x_train)); names(penalty) <- colnames(x_train)
if(length(suppressed)) penalty[design_dictionary$design_column[design_dictionary$variable_key %in% suppressed]] <- 0
cv_constrained <- NULL
if(length(suppressed)) {
  cv_constrained <- cv.glmnet(x_train,y_train,family="cox",alpha=1,nfolds=10,foldid=foldid,type.measure="deviance",standardize=TRUE,penalty.factor=penalty,keep=TRUE)
  cat("Clinically constrained factors:",paste(suppressed,collapse=", "),"\n")
}

cindex_harrell <- function(time,event,risk) {
  cc <- concordance(Surv(time,event)~risk,reverse=TRUE); c(C=unname(cc$concordance),SE=sqrt(unname(cc$var)))
}
km_censor <- function(time,event) {
  fit <- survfit(Surv(time,1-event)~1); list(time=fit$time,surv=fit$surv)
}
Gfun <- function(km,t,left=FALSE) {
  tt <- km$time; ss <- km$surv; sapply(t,function(z){ix <- if(left) which(tt<z) else which(tt<=z); if(!length(ix)) 1 else ss[max(ix)]})
}
uno_c <- function(train_time,train_event,test_time,test_event,risk,tau=NULL) {
  if(is.null(tau)) tau <- min(max(test_time),quantile(train_time,.9)); km <- km_censor(train_time,train_event); num<-0;den<-0
  for(i in which(test_event==1 & test_time<=tau)) {js <- which(test_time>test_time[i]); if(!length(js))next; w <- 1/max(Gfun(km,test_time[i],TRUE),1e-6)^2; den<-den+w*length(js); num<-num+w*sum(risk[i]>risk[js]+1e-12)+0.5*w*sum(abs(risk[i]-risk[js])<=1e-12)}
  num/den
}
auc_ipcw <- function(tr_t,tr_e,te_t,te_e,risk,horizon) {
  km<-km_censor(tr_t,tr_e); cases<-which(te_e==1 & te_t<=horizon); ctrls<-which(te_t>horizon); if(!length(cases)||!length(ctrls))return(NA_real_)
  wc<-1/pmax(Gfun(km,te_t[cases],TRUE),1e-6); wn<-rep(1/max(Gfun(km,horizon),1e-6),length(ctrls)); num<-0;den<-sum(wc)*sum(wn)
  for(i in seq_along(cases)){cmp<-risk[cases[i]]-risk[ctrls];num<-num+wc[i]*sum(wn*(cmp>1e-12)+0.5*wn*(abs(cmp)<=1e-12))};num/den
}
brier_ipcw <- function(tr_t,tr_e,te_t,te_e,survprob,horizon) {
  survprob <- as.numeric(survprob)
  km<-km_censor(tr_t,tr_e); a<-numeric(length(te_t)); ev<-te_e==1 & te_t<=horizon; alive<-te_t>horizon
  g_event <- as.numeric(Gfun(km,te_t[ev],TRUE))
  g_horizon <- as.numeric(Gfun(km,horizon))
  a[ev]<-(survprob[ev]^2)/pmax(g_event,1e-6); a[alive]<-((1-survprob[alive])^2)/max(g_horizon,1e-6);mean(a)
}
baseline_hazard <- function(time,event,lp) {
  et <- sort(unique(time[event==1])); H<-0; out<-numeric(length(et)); for(k in seq_along(et)){t<-et[k];d<-sum(event==1 & time==t);H<-H+d/sum(exp(lp[time>=t]));out[k]<-H};data.frame(time=et,H0=out)
}
surv_probs <- function(base,lp,times) {
  H <- sapply(times,function(t){ix<-which(base$time<=t);if(!length(ix))0 else base$H0[max(ix)]}); outer(exp(lp),H,function(r,h)exp(-h*r))
}
evaluate_model <- function(name,coefvec,xtr=x_train,xte=x_test,train_df=train,test_df=test) {
  b<-setNames(rep(0,ncol(xtr)),colnames(xtr)); b[names(coefvec)]<-coefvec; lptr<-as.numeric(xtr%*%b); lpte<-as.numeric(xte%*%b)
  base<-baseline_hazard(train_df$time,train_df$event,lptr); horizons<-c(12,24,36); grid<-sort(unique(c(seq(max(1,min(train_df$time)),min(36,quantile(train_df$time,.9)),length.out=60),horizons)))
  Str<-surv_probs(base,lptr,grid); Ste<-surv_probs(base,lpte,grid)
  ibs_tr<-trapz<-function(x,y)sum(diff(x)*(head(y,-1)+tail(y,-1))/2)/(max(x)-min(x)); bs_tr<-sapply(seq_along(grid),function(j)brier_ipcw(train_df$time,train_df$event,train_df$time,train_df$event,Str[,j],grid[j]));bs_te<-sapply(seq_along(grid),function(j)brier_ipcw(train_df$time,train_df$event,test_df$time,test_df$event,Ste[,j],grid[j]))
  ctr<-cindex_harrell(train_df$time,train_df$event,lptr);cte<-cindex_harrell(test_df$time,test_df$event,lpte)
  slope_tr<-try(coef(coxph(Surv(time,event)~lptr,data=train_df))[1],silent=TRUE); slope_te<-try(coef(coxph(Surv(time,event)~lpte,data=test_df))[1],silent=TRUE)
  data.frame(model=name,number_parameters=sum(b!=0),EPV=sum(train_df$event)/max(1,sum(b!=0)),training_Harrell_C=ctr["C"],training_C_low=max(0,ctr["C"]-1.96*ctr["SE"]),training_C_high=min(1,ctr["C"]+1.96*ctr["SE"]),optimism_corrected_C=NA_real_,Uno_C_train=uno_c(train_df$time,train_df$event,train_df$time,train_df$event,lptr),
             AUC_1y_train=auc_ipcw(train_df$time,train_df$event,train_df$time,train_df$event,lptr,12),AUC_2y_train=auc_ipcw(train_df$time,train_df$event,train_df$time,train_df$event,lptr,24),AUC_3y_train=auc_ipcw(train_df$time,train_df$event,train_df$time,train_df$event,lptr,36),IBS_train=trapz(grid,bs_tr),calibration_slope_train=if(inherits(slope_tr,"try-error"))NA else slope_tr,
             validation_Harrell_C=cte["C"],validation_C_low=max(0,cte["C"]-1.96*cte["SE"]),validation_C_high=min(1,cte["C"]+1.96*cte["SE"]),Uno_C_validation=uno_c(train_df$time,train_df$event,test_df$time,test_df$event,lpte),
             AUC_1y_validation=auc_ipcw(train_df$time,train_df$event,test_df$time,test_df$event,lpte,12),AUC_2y_validation=auc_ipcw(train_df$time,train_df$event,test_df$time,test_df$event,lpte,24),AUC_3y_validation=auc_ipcw(train_df$time,train_df$event,test_df$time,test_df$event,lpte,36),IBS_validation=trapz(grid,bs_te),calibration_slope_validation=if(inherits(slope_te,"try-error"))NA else slope_te,
             stringsAsFactors=FALSE)
}

get_nonzero <- function(cv,s) {z<-coef_at(cv,s);setNames(z$coefficient[z$coefficient!=0],z$term[z$coefficient!=0])}
models <- list("Main LASSO lambda.1se"=get_nonzero(cv_main,"lambda.1se"),"Main LASSO lambda.min"=get_nonzero(cv_main,"lambda.min"))
if(!is.null(cv_constrained)){models[["Clinically constrained LASSO lambda.1se"]]<-get_nonzero(cv_constrained,"lambda.1se");models[["Clinically constrained LASSO lambda.min"]]<-get_nonzero(cv_constrained,"lambda.min")}

perf<-do.call(rbind,lapply(names(models),function(n)evaluate_model(n,models[[n]])))
xB_train <- model.matrix(lasso_formula,prep_factors(train,stage_categorical=FALSE))[,-1,drop=FALSE]
xB_test <- model.matrix(lasso_formula,prep_factors(test,stage_categorical=FALSE))[,-1,drop=FALSE]
forceA_coef <- setNames(poolA$beta,poolA$term)
forceB_coef <- setNames(poolB$beta,poolB$term)
perf_force <- rbind(evaluate_model("Force-entry Cox Model A (categorical overall TNM)",forceA_coef,x_train,x_test,train,test),
                    evaluate_model("Force-entry Cox Model B (ordinal overall TNM)",forceB_coef,xB_train,xB_test,train,test))
perf <- rbind(perf_force,perf)
write.csv(perf,file.path(outdir,"Table_D_model_performance.csv"),row.names=FALSE)

tabC <- merge(c1,cmin,by="term",suffixes=c("_at_lambda.1se","_at_lambda.min"))
tabC$selection_frequency <- NA_real_
tabC$direction_consistency <- NA_real_
tabC$variable <- variable_labels[group_map(tabC$term,"A")];tabC$whether_clinically_forced<-group_map(tabC$term,"A")%in%suppressed
write.csv(tabC[,c("variable","term","coefficient_at_lambda.1se","coefficient_at_lambda.min","selection_frequency","direction_consistency","whether_clinically_forced")],file.path(outdir,"Table_C_LASSO_selection.csv"),row.names=FALSE)
write.csv(data.frame(item=c("bootstrap_optimism_correction","bootstrap_selection_stability"),performed=c(FALSE,FALSE),reason=c("Not requested: no training-set internal validation","Not requested: no training-set internal validation")),file.path(outdir,"internal_validation_status.csv"),row.names=FALSE)

coef_all <- do.call(rbind,lapply(names(models),function(n){
  tt <- names(models[[n]])
  data.frame(model=rep(n,length(tt)),term=tt,coefficient=as.numeric(models[[n]]),
             variable=unname(variable_labels[group_map(tt,"A")]),row.names=NULL)
}))
write.csv(coef_all,file.path(outdir,"lasso_nonzero_coefficients.csv"),row.names=FALSE)
lambda_table<-data.frame(model=c("Main LASSO",if(!is.null(cv_constrained))"Clinically constrained LASSO" else NULL),lambda_1se=c(cv_main$lambda.1se,if(!is.null(cv_constrained))cv_constrained$lambda.1se else NULL),lambda_min=c(cv_main$lambda.min,if(!is.null(cv_constrained))cv_constrained$lambda.min else NULL))
write.csv(lambda_table,file.path(outdir,"lasso_lambda_values.csv"),row.names=FALSE)

score_rows <- list()
for(n in names(models)){b<-setNames(rep(0,ncol(x_train)),colnames(x_train));b[names(models[[n]])]<-models[[n]];score_rows[[length(score_rows)+1]]<-data.frame(patient_id=c(train$patient_id,test$patient_id),dataset=c(rep("Train",nrow(train)),rep("Test",nrow(test))),event=c(train$event,test$event),time=c(train$time,test$time),model=n,linear_predictor=c(as.numeric(x_train%*%b),as.numeric(x_test%*%b)))}
scores<-do.call(rbind,score_rows);write.csv(scores,file.path(outdir,"clinical_model_linear_predictors.csv"),row.names=FALSE)
saveRDS(list(seed=20250308,candidates=candidate_vars,original_names=original_names,imputation_parameters=imp_params,design_columns=colnames(x_train),foldid=foldid,cv_main=cv_main,cv_constrained=cv_constrained,penalty_factor=penalty,coefficients=models,performance=perf),file.path(outdir,"clinical_model_objects.rds"))

# Complete-case sensitivity for force-entry models.
cc <- train0[complete.cases(train0[,force_vars]),,drop=FALSE]
ccA <- fit_force(cc,"A");ccB <- fit_force(cc,"B")
cc_table<-function(f,m){s<-summary(f);data.frame(model=rep(m,nrow(s$coefficients)),term=rownames(s$coefficients),beta=s$coefficients[,"coef"],HR=s$coefficients[,"exp(coef)"],CI_low=s$conf.int[,"lower .95"],CI_high=s$conf.int[,"upper .95"],P=s$coefficients[,"Pr(>|z|)"],N=rep(f$n,nrow(s$coefficients)),events=rep(f$nevent,nrow(s$coefficients)))}
write.csv(rbind(cc_table(ccA,"Model A categorical overall TNM complete-case"),cc_table(ccB,"Model B ordinal overall TNM complete-case")),file.path(outdir,"force_entry_complete_case_sensitivity.csv"),row.names=FALSE)

summary_list<-list(seed=20250308,train_N=nrow(train),train_events=sum(train$event),validation_N=nrow(test),validation_events=sum(test$event),force_entry_variables=force_vars,lasso_candidates=candidate_vars,semantic_MRI_variables=semantic_vars,excluded_variables=c("PLR","LMR","smoking","T_stage","N_stage"),TNM_overall_included=TRUE,training_internal_validation_performed=FALSE,independent_Model_A=indA,independent_Model_B=indB,staging_dependence=!setequal(indA,indB),lambda_1se=cv_main$lambda.1se,lambda_min=cv_main$lambda.min,main_selected_1se=selected_1se,main_selected_min=selected_min,constrained_required=length(suppressed)>0,constrained_factors=suppressed,models=names(models),treatment_variable_available=FALSE)
dput(summary_list,file=file.path(outdir,"analysis_summary_dput.txt"))
cat("Analysis complete\n")
