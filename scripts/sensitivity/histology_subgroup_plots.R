options(stringsAsFactors=FALSE)
suppressPackageStartupMessages({
  library(survival)
  library(survminer)
  library(ggplot2)
  library(forestplot)
  library(grid)
})

out_dir <- "/root/autodl-tmp/HCR_runs/seven_models_histology_20260814"

# ---------- Forest plot: complete-case forced-entry model ----------
cox_all <- read.csv(file.path(out_dir, "Table_Forced_Entry_Cox.csv"), check.names=FALSE)
cox <- cox_all[cox_all$Analysis == "Complete-case primary", ]
order_terms <- c("ITH_per0.1", "TNM_3", "TNM_4", "TNM_5", "Sex_male", "Age_per10y",
                 "NLR", "Site_sinus", "Hist_2", "Hist_3", "Hist_4")
cox <- cox[match(order_terms, cox$Term), ]

rows <- list(
  list(header=TRUE, group="Imaging", label="Imaging", level="", HR=NA, lo=NA, hi=NA, p=NA, sig=FALSE, ith=FALSE),
  list(header=FALSE, group="Imaging", label="3D-ITHscore", level="Per 0.1 increase", HR=cox$HR[1], lo=cox$CI_L95[1], hi=cox$CI_U95[1], p=cox$P[1], sig=cox$P[1]<0.05, ith=TRUE),
  list(header=TRUE, group="Clinical", label="Clinical", level="", HR=NA, lo=NA, hi=NA, p=NA, sig=FALSE, ith=FALSE),
  list(header=FALSE, group="Clinical", label="TNM overall stage", level="Stage III vs Stage II", HR=cox$HR[2], lo=cox$CI_L95[2], hi=cox$CI_U95[2], p=cox$P[2], sig=cox$P[2]<0.05, ith=FALSE),
  list(header=FALSE, group="Clinical", label="TNM overall stage", level="Stage IVA vs Stage II", HR=cox$HR[3], lo=cox$CI_L95[3], hi=cox$CI_U95[3], p=cox$P[3], sig=cox$P[3]<0.05, ith=FALSE),
  list(header=FALSE, group="Clinical", label="TNM overall stage", level="Stage IVB vs Stage II", HR=cox$HR[4], lo=cox$CI_L95[4], hi=cox$CI_U95[4], p=cox$P[4], sig=cox$P[4]<0.05, ith=FALSE),
  list(header=FALSE, group="Clinical", label="Sex", level="Male vs female", HR=cox$HR[5], lo=cox$CI_L95[5], hi=cox$CI_U95[5], p=cox$P[5], sig=cox$P[5]<0.05, ith=FALSE),
  list(header=FALSE, group="Clinical", label="Age", level="Per 10-year increase", HR=cox$HR[6], lo=cox$CI_L95[6], hi=cox$CI_U95[6], p=cox$P[6], sig=cox$P[6]<0.05, ith=FALSE),
  list(header=FALSE, group="Clinical", label="NLR", level="Per 1-unit increase", HR=cox$HR[7], lo=cox$CI_L95[7], hi=cox$CI_U95[7], p=cox$P[7], sig=cox$P[7]<0.05, ith=FALSE),
  list(header=FALSE, group="Clinical", label="Primary site", level="Sinus vs nasal cavity", HR=cox$HR[8], lo=cox$CI_L95[8], hi=cox$CI_U95[8], p=cox$P[8], sig=cox$P[8]<0.05, ith=FALSE),
  list(header=TRUE, group="Pathology", label="Pathology", level="", HR=NA, lo=NA, hi=NA, p=NA, sig=FALSE, ith=FALSE),
  list(header=FALSE, group="Pathology", label="Histologic subtype", level="Adenocarcinoma vs SCC", HR=cox$HR[9], lo=cox$CI_L95[9], hi=cox$CI_U95[9], p=cox$P[9], sig=cox$P[9]<0.05, ith=FALSE),
  list(header=FALSE, group="Pathology", label="Histologic subtype", level="Undifferentiated vs SCC", HR=cox$HR[10], lo=cox$CI_L95[10], hi=cox$CI_U95[10], p=cox$P[10], sig=cox$P[10]<0.05, ith=FALSE),
  list(header=FALSE, group="Pathology", label="Histologic subtype", level="SMARCB1-deficient vs SCC", HR=cox$HR[11], lo=cox$CI_L95[11], hi=cox$CI_U95[11], p=cox$P[11], sig=cox$P[11]<0.05, ith=FALSE)
)
df <- do.call(rbind, lapply(rows, as.data.frame))
fmtp <- function(p) ifelse(is.na(p), "", ifelse(p<0.001, "<0.001", sprintf("%.3f",p)))
df$label[df$sig & !df$header] <- paste0(df$label[df$sig & !df$header], " *")
df$label[df$ith] <- "3D-ITHscore * [highlighted]"
hrtxt <- ifelse(is.na(df$HR), "", sprintf("%.2f (%.2f-%.2f)", df$HR, df$lo, df$hi))
labels <- cbind(Variable=df$label, `Level / scale`=df$level, `HR (95% CI)`=hrtxt, P=fmtp(df$p))
boxcols <- ifelse(df$ith, "#E64B35", "#3C5488")
linecols <- boxcols
box_gp <- lapply(seq_len(nrow(df)), function(i) gpar(fill=boxcols[i], col=boxcols[i]))
line_gp <- lapply(seq_len(nrow(df)), function(i) gpar(col=linecols[i], lwd=ifelse(df$ith[i],2.2,1.2)))

pdf(file.path(out_dir, "Fig_Forced_Entry_Cox_Forest.pdf"), width=11.5, height=7.7, useDingbats=FALSE)
forestplot(labeltext=labels, mean=df$HR, lower=df$lo, upper=df$hi,
           is.summary=df$header, zero=1, xlog=TRUE, clip=c(0.15,60),
           xticks=c(0.25,0.5,1,2,5,10,25,50), graph.pos=3,
           boxsize=0.18, lineheight=unit(0.48,"cm"), colgap=unit(4,"mm"),
           col=fpColors(box="#3C5488", line="#3C5488", summary="#4DBBD5", zero="#777777"),
           shapes_gp=fpShapesGp(box=box_gp, lines=line_gp),
           txt_gp=fpTxtGp(label=gpar(cex=0.92), ticks=gpar(cex=0.85), xlab=gpar(cex=0.95),
                          title=gpar(cex=1.15, fontface="bold")),
           xlab="Hazard ratio (log scale)",
           title="Forced-entry multivariable Cox model (training cohort)",
           footnote="SCC is the histologic reference. 3D-ITHscore HR is per 0.1 increase. * P<0.05; red box/line highlights 3D-ITHscore.",
           new_page=FALSE)
dev.off()

# ---------- KM plots for the refitted SCC+ADC Clinical-ITH model ----------
tr <- read.csv(file.path(out_dir, "Risk_scores_SCC_ADC_training.csv"), check.names=FALSE)
va <- read.csv(file.path(out_dir, "Risk_scores_SCC_ADC_validation.csv"), check.names=FALSE)
al <- read.csv(file.path(out_dir, "Risk_scores_SCC_ADC_all.csv"), check.names=FALSE)
for (nm in c("tr","va","al")) {
  z <- get(nm); z$Risk_group <- factor(z$Risk_group, levels=c("Low risk","High risk")); assign(nm,z)
}

km_stats <- function(d, cohort) {
  sf <- survfit(Surv(OS_time,OS_event)~Risk_group,data=d)
  med <- summary(sf)$table
  if (is.null(dim(med))) med <- matrix(med,nrow=1)
  sd <- survdiff(Surv(OS_time,OS_event)~Risk_group,data=d)
  p <- pchisq(sd$chisq,1,lower.tail=FALSE)
  cf <- coxph(Surv(OS_time,OS_event)~Risk_group,data=d,ties="efron")
  cs <- summary(cf)$conf.int[1,]
  tab <- table(d$Risk_group); ev <- tapply(d$OS_event,d$Risk_group,sum)
  data.frame(Cohort=cohort,N=nrow(d),Events=sum(d$OS_event),
             N_low=unname(tab["Low risk"]),Events_low=unname(ev["Low risk"]),
             N_high=unname(tab["High risk"]),Events_high=unname(ev["High risk"]),
             Logrank_chisq=unname(sd$chisq),Logrank_P=p,
             HR_high_vs_low=unname(cs["exp(coef)"]),HR_L95=unname(cs["lower .95"]),HR_U95=unname(cs["upper .95"]),
             Median_OS_low=med[grep("Low risk",rownames(med)),"median"],
             Median_OS_high=med[grep("High risk",rownames(med)),"median"])
}
stats <- rbind(km_stats(tr,"Training"),km_stats(va,"Validation"),km_stats(al,"All subgroup"))
write.csv(stats,file.path(out_dir,"Table_Subgroup_KM_Statistics.csv"),row.names=FALSE)

medtxt <- function(x) ifelse(is.na(x),"NR",sprintf("%.1f mo",x))
plot_km <- function(d,cohort,file) {
  st <- stats[stats$Cohort==cohort,]
  ann <- sprintf("Log-rank P %s\nHR = %.2f (95%% CI %.2f-%.2f)\nMedian OS: low %s; high %s",
                 ifelse(st$Logrank_P<0.001,"< 0.001",paste0("= ",sprintf("%.3f",st$Logrank_P))),
                 st$HR_high_vs_low,st$HR_L95,st$HR_U95,medtxt(st$Median_OS_low),medtxt(st$Median_OS_high))
  sf <- survfit(Surv(OS_time,OS_event)~Risk_group,data=d)
  g <- ggsurvplot(sf,data=d,risk.table=TRUE,conf.int=TRUE,pval=FALSE,
                  palette=c("#4DBBD5","#E64B35"),legend.title="Risk group",
                  legend.labs=c("Low risk","High risk"),xlim=c(0,60),break.time.by=12,
                  xlab="Time (months)",ylab="Overall survival probability",
                  surv.median.line="hv",risk.table.height=0.25,risk.table.y.text.col=TRUE,
                  risk.table.y.text=FALSE,ggtheme=theme_classic(base_size=12),
                  tables.theme=theme_cleantable(base_size=11),censor.shape=3,censor.size=2.5)
  g$plot <- g$plot + ggtitle(paste0("SCC+ADC Clinical-ITH risk stratification - ",cohort)) +
    annotate("text",x=58,y=0.94,label=ann,hjust=1,vjust=1,size=3.5) +
    theme(plot.title=element_text(hjust=0.5,face="bold"),legend.position=c(0.18,0.18))
  pdf(file.path(out_dir,file),width=8.2,height=7.2,useDingbats=FALSE)
  print(g,newpage=FALSE)
  dev.off()
}
plot_km(tr,"Training","Fig_KM_SCC_ADC_Training.pdf")
plot_km(va,"Validation","Fig_KM_SCC_ADC_Validation.pdf")
plot_km(al,"All subgroup","Fig_KM_SCC_ADC_All.pdf")

cat("PLOTS_DONE\n")
