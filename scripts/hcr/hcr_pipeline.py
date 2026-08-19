#!/usr/bin/env python3
import argparse, hashlib, json, math, pickle, re, subprocess, warnings
from pathlib import Path
import joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, seaborn as sns
from lifelines.utils import concordance_index
from neuroCombat import neuroCombat
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv

SEED=20250308; RNG=np.random.default_rng(SEED)

def nid(x):
    s=re.sub(r'\D','',str(x)); return str(int(s))

def md5(p):
    h=hashlib.md5()
    with open(p,'rb') as f:
        while b:=f.read(1<<20): h.update(b)
    return h.hexdigest()

def icc_pass(icc_path):
    d=pd.read_csv(icc_path); keep=d.loc[d.ICC_A1>.8,'feature'].tolist()
    return d,keep

def correlation_prune(X,cutoff=.9):
    # caret::findCorrelation default fast strategy for p>=100: for each pair
    # above cutoff remove the member having the larger mean absolute correlation.
    c=X.corr(method='pearson').abs().fillna(0); np.fill_diagonal(c.values,0)
    means=c.mean(axis=0).to_numpy(); arr=c.to_numpy(); hit=np.argwhere(np.triu(arr,1)>cutoff)
    drop_idx=set()
    for i,j in hit:
        drop_idx.add(int(j if means[j]>means[i] else i))
    removed=[c.columns[i] for i in sorted(drop_idx)]
    active=[f for i,f in enumerate(c.columns) if i not in drop_idx]
    maxcorr={}
    for f in removed:
        candidates=c.loc[f,active]
        q=candidates.idxmax(); maxcorr[f]=(q,float(candidates[q]))
    return active,removed,maxcorr,c

def bootstrap_c(t,e,r,B=2000):
    vals=[]; n=len(t); base=concordance_index(t,-r,e)
    for _ in range(B):
        ix=RNG.integers(0,n,n)
        if np.unique(e[ix]).size<2: continue
        try: vals.append(concordance_index(t[ix],-r[ix],e[ix]))
        except Exception: pass
    return base,*np.quantile(vals,[.025,.975]),len(vals)

def stratified_folds(event,n=10):
    return list(StratifiedKFold(n,shuffle=True,random_state=SEED).split(np.zeros(len(event)),event))

def coxnet_cv(X,y,event,penalty,out):
    path=CoxnetSurvivalAnalysis(l1_ratio=1,penalty_factor=penalty,n_alphas=100,alpha_min_ratio=1e-4,max_iter=500000,tol=1e-7).fit(X,y)
    folds=stratified_folds(event); rows=[]
    for a in path.alphas_:
        ss=[]; nn=[]
        for tr,te in folds:
            try:
                m=CoxnetSurvivalAnalysis(l1_ratio=1,penalty_factor=penalty,alphas=[a],max_iter=500000,tol=1e-7).fit(X[tr],y[tr])
                ss.append(concordance_index(y[te]['time'],-m.predict(X[te]),y[te]['event']))
                nn.append(np.count_nonzero(abs(m.coef_[:,0])>1e-10))
            except Exception: ss.append(np.nan); nn.append(np.nan)
        rows.append((a,np.nanmean(ss),np.nanstd(ss,ddof=1)/np.sqrt(np.isfinite(ss).sum()),np.nanmean(nn)))
    tab=pd.DataFrame(rows,columns=['alpha','mean_cv_harrell_c','se','mean_nonzero']); tab.to_csv(out/'lasso_cv_results.csv',index=False)
    best=tab.loc[tab.mean_cv_harrell_c.idxmax()]; eligible=tab[tab.mean_cv_harrell_c>=best.mean_cv_harrell_c-best.se]
    target=eligible[(eligible.mean_nonzero>=5)&(eligible.mean_nonzero<=8)]
    if len(target): chosen=float(target.alpha.max()); rule='largest alpha within 1-SE and mean parameter count 5-8'
    else: chosen=float(eligible.alpha.max()); rule='largest alpha within 1-SE; no alpha in 1-SE set achieved 5-8 mean parameters'
    model=CoxnetSurvivalAnalysis(l1_ratio=1,penalty_factor=penalty,alphas=[chosen],max_iter=1000000,tol=1e-8,fit_baseline_model=True).fit(X,y)
    return path,tab,chosen,rule,model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train',required=True); ap.add_argument('--validation',required=True); ap.add_argument('--clinical',required=True); ap.add_argument('--scanner',required=True); ap.add_argument('--icc',required=True); ap.add_argument('--r-script',required=True); ap.add_argument('--out',required=True); ap.add_argument('--bootstrap',type=int,default=1000); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    tr=pd.read_csv(a.train); va=pd.read_csv(a.validation); tr['patient_id']=tr.id.map(nid); va['patient_id']=va.id.map(nid)
    outcomes=['id','OS/label','OS/m','patient_id']; rawcols=[c for c in tr.columns if c not in outcomes and '_diagnostics_' not in c]
    if rawcols != [c for c in va.columns if c not in outcomes and '_diagnostics_' not in c]: raise SystemExit('feature columns differ')
    icctab,icc_keep=icc_pass(a.icc); cols=[c for c in rawcols if c in set(icc_keep)]
    Xtr=tr[cols].replace([np.inf,-np.inf],np.nan); Xv=va[cols].replace([np.inf,-np.inf],np.nan)
    imp=SimpleImputer(strategy='median').fit(Xtr); Xtr=pd.DataFrame(imp.transform(Xtr),columns=cols); Xv=pd.DataFrame(imp.transform(Xv),columns=cols)
    var=Xtr.var(ddof=1); keepvar=var[var>=.01].index.tolist(); low=var[var<.01].index.tolist()
    pd.DataFrame({'feature':var.index,'training_variance':var.values,'retained':var.values>=.01}).to_csv(out/'variance_filter_results.csv',index=False)
    Xtr=Xtr[keepvar]; Xv=Xv[keepvar]
    keepcor,dropcor,pairs,cormat=correlation_prune(Xtr,.9)
    pd.DataFrame([{'removed_feature':f,'retained_correlated_feature':pairs[f][0],'absolute_r':pairs[f][1]} for f in dropcor]).to_csv(out/'correlation_filter_removed.csv',index=False)
    (out/'correlation_retained_features.txt').write_text('\n'.join(keepcor)+'\n')
    Xtr=Xtr[keepcor]; Xv=Xv[keepcor]
    scan=pd.read_excel(a.scanner).iloc[1:,[2,3,4]].copy(); scan.columns=['patient_id','scanner_t1','scanner_t2']; scan.patient_id=scan.patient_id.map(nid)
    meta=pd.concat([tr[['patient_id','OS/label','OS/m']].assign(cohort='Training'),va[['patient_id','OS/label','OS/m']].assign(cohort='Validation')]).merge(scan,on='patient_id',validate='one_to_one')
    mtr=meta.cohort.eq('Training'); batch=(meta.loc[mtr,'scanner_t1'].astype(str).str.strip()+' | '+meta.loc[mtr,'scanner_t2'].astype(str).str.strip())
    vc=batch.value_counts(); batch=batch.where(~batch.isin(vc[vc<5].index),'Other')
    bd=batch.value_counts().rename_axis('batch').reset_index(name='N'); bd['proportion']=bd.N/len(batch); bd.to_csv(out/'batch_distribution.csv',index=False)
    cb=neuroCombat(dat=Xtr.T,covars=pd.DataFrame({'batch':batch.values}),batch_col='batch'); Ctr=pd.DataFrame(cb['data'].T,columns=keepcor); Cv=Xv.copy()
    if not np.isfinite(Ctr).all().all(): raise SystemExit('ComBat produced nonfinite values')
    scaler=StandardScaler().fit(Ctr); Ztr=pd.DataFrame(scaler.transform(Ctr),columns=keepcor); Zv=pd.DataFrame(scaler.transform(Cv),columns=keepcor)
    # mRMRe survival Top 15 on training only.
    mdat=pd.concat([tr[['OS/m','OS/label']].rename(columns={'OS/m':'time','OS/label':'event'}).reset_index(drop=True),Ztr.reset_index(drop=True)],axis=1); mdat.to_csv(out/'mRMR_input_train.csv',index=False)
    rr=subprocess.run(['Rscript',a.r_script,str(out/'mRMR_input_train.csv'),str(out)],capture_output=True,text=True); (out/'mRMR_R.log').write_text(rr.stdout+'\nSTDERR\n'+rr.stderr)
    if rr.returncode: raise SystemExit('R mRMRe failed')
    rad15=[x.strip() for x in (out/'mRMR_selected_15.txt').read_text().splitlines() if x.strip()][:15]
    names=rad15; A=Ztr[rad15].to_numpy(); Av=Zv[rad15].to_numpy(); penalty=np.ones(len(rad15))
    event=tr['OS/label'].astype(int).to_numpy(); time=tr['OS/m'].astype(float).to_numpy(); ev=va['OS/label'].astype(int).to_numpy(); tv=va['OS/m'].astype(float).to_numpy(); y=Surv.from_arrays(event.astype(bool),time); yv=Surv.from_arrays(ev.astype(bool),tv)
    path,cvtab,alpha,rule,model=coxnet_cv(A,y,event,penalty,out); coef=model.coef_[:,0]; fullnz=abs(coef)>1e-10
    # Conditional bootstrap stability at the prespecified full-data alpha.
    selected=np.zeros((a.bootstrap,len(names)),bool); signs=np.zeros((a.bootstrap,len(names)),int); failures=[]
    for b in range(a.bootstrap):
        ix=RNG.integers(0,len(A),len(A))
        if np.unique(event[ix]).size<2: failures.append(b); continue
        try:
            bm=CoxnetSurvivalAnalysis(l1_ratio=1,penalty_factor=penalty,alphas=[alpha],max_iter=1000000,tol=1e-7).fit(A[ix],y[ix]); cc=bm.coef_[:,0]; selected[b]=abs(cc)>1e-10; signs[b]=np.sign(cc).astype(int)
        except Exception: failures.append(b)
    valid=np.ones(a.bootstrap,bool); valid[failures]=False; freq=selected[valid].mean(0); direction=[]
    for j in range(len(names)):
        s=signs[valid,j][selected[valid,j]]; direction.append(max((s>0).mean(),(s<0).mean()) if len(s) else np.nan)
    stab=pd.DataFrame({'feature':names,'full_model_coefficient':coef,'full_model_selected':fullnz,'bootstrap_selection_frequency':freq,'direction_consistency':direction})
    stab.to_csv(out/'bootstrap_stability_results.csv',index=False)
    final=model; final_names=names; fc=coef; nonzero=fullnz; final_alpha=alpha
    risk=final.predict(A); riskv=final.predict(Av); tc=bootstrap_c(time,event,risk); vcx=bootstrap_c(tv,ev,riskv)
    tau=min(60.,np.nextafter(time.max(),0),np.nextafter(tv.max(),0)); uno=float(concordance_index_ipcw(y,yv,riskv,tau=tau)[0])
    auc=[]
    for cohort,yr,yt,rs in [('Training',y,y,risk),('Validation',y,yv,riskv)]:
        for t in [12.,24.,36.,60.]:
            try: val=float(cumulative_dynamic_auc(yr,yt,rs,np.array([t]))[0][0]); note=''
            except Exception as e: val=np.nan; note=f'{type(e).__name__}: {e}'
            auc.append((cohort,t,val,note))
    auc=pd.DataFrame(auc,columns=['cohort','months','AUC','note']); auc.to_csv(out/'time_dependent_AUC.csv',index=False)
    pd.DataFrame({'patient_id':tr.patient_id,'risk_score':risk}).to_csv(out/'risk_scores_train.csv',index=False); pd.DataFrame({'patient_id':va.patient_id,'risk_score':riskv}).to_csv(out/'risk_scores_validation.csv',index=False)
    coeff=pd.DataFrame({'feature':final_names,'coefficient_standardized':fc,'nonzero':nonzero}); coeff.to_csv(out/'final_model_coefficients.csv',index=False)
    joblib.dump({'model':final,'features':final_names,'alpha':alpha,'radiomics_imputer':imp,'radiomics_variance_features':keepvar,'radiomics_correlation_features':keepcor,'radiomics_scaler':scaler,'combat':cb,'status':'pure HCR; exploratory validation'},out/'revised_HCR_model.joblib')
    # Plots
    fig,ax=plt.subplots(figsize=(9,6)); ax.plot(np.log10(path.alphas_),path.coef_.T,alpha=.6); ax.axvline(np.log10(alpha),c='red',ls='--'); ax.set(xlabel='log10(alpha)',ylabel='Coefficient'); fig.tight_layout(); fig.savefig(out/'lasso_coefficient_path.png',dpi=300); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6)); ss=stab.sort_values('bootstrap_selection_frequency'); ax.barh(ss.feature,ss.bootstrap_selection_frequency,color='#0072B2'); ax.axvline(.6,c='black',ls='--'); ax.set(xlabel='Bootstrap selection frequency',xlim=(0,1)); fig.tight_layout(); fig.savefig(out/'bootstrap_stability.png',dpi=300,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));
    for c,g in auc.groupby('cohort'): ax.plot(g.months,g.AUC,marker='o',label=c)
    ax.set(xlabel='Months',ylabel='Cumulative/dynamic AUC',ylim=(0,1)); ax.legend(); fig.tight_layout(); fig.savefig(out/'time_AUC.png',dpi=300); plt.close(fig)
    counts={'raw_features':len(rawcols),'icc_retained':len(cols),'variance_retained':len(keepvar),'low_variance_removed':len(low),'correlation_retained':len(keepcor),'correlation_removed':len(dropcor),'mrmr_top':len(rad15),'lasso_candidate_total':len(names),'final_nonzero_parameters':int(nonzero.sum())}
    pd.DataFrame(list(counts.items()),columns=['stage','N_features']).to_csv(out/'feature_reduction_flow.csv',index=False)
    epv=event.sum()/max(1,int(nonzero.sum())); target_met=bool(5<=int(nonzero.sum())<=8 and epv>=8)
    formula=' + '.join(f'{b:.12g}*{n}' for n,b,z in zip(final_names,fc,nonzero) if z)
    report=f'''# Pure HCR pipeline complete analysis report\n\n## Status\nThis is a pure radiomics-only HCR-Cox model. No clinical variables were forced into LASSO.\n\n## Data and leakage controls\n- Training: N={len(tr)}, deaths={event.sum()}; validation: N={len(va)}, deaths={ev.sum()}.\n- All variance, correlation, ComBat, scaling, mRMRe, alpha selection and bootstrap calculations were fit using training data only.\n- The validation cohort was not used for feature filtering or model selection.\n\n## Feature reduction\n- Raw non-diagnostic radiomics features: {len(rawcols)}.\n- ICC(A,1)>0.8: {len(cols)}.\n- Variance >=0.01: {len(keepvar)} retained; {len(low)} removed.\n- Correlation pruning |r|>0.9: {len(keepcor)} retained; {len(dropcor)} removed.\n- Survival mRMRe: Top {len(rad15)}.\n\n## LASSO-Cox and conditional stability\n- Alpha selection rule: {rule}.\n- Chosen alpha: {alpha:.12g}.\n- LASSO nonzero radiomics parameters: {int(fullnz.sum())}.\n- Bootstrap repetitions requested/successful: {a.bootstrap}/{int(valid.sum())}.\n- Bootstrap selection frequencies are descriptive and did not alter the final model.\n- EPV={epv:.2f}.\n- Prespecified target (5-8 parameters and EPV>=8): {'met' if target_met else 'not met; no threshold was changed post hoc'}.\n\n## Performance\n- Training Harrell C: {tc[0]:.3f} (95% CI {tc[1]:.3f}-{tc[2]:.3f}).\n- Validation Harrell C: {vcx[0]:.3f} (95% CI {vcx[1]:.3f}-{vcx[2]:.3f}).\n- Validation Uno C: {uno:.3f} at tau={tau:g} months.\n\n## Standardized risk formula\n`{formula}`\n\n## Interpretation\nValidation contains only six deaths, so all validation estimates are exploratory and imprecise. Feature selection frequencies are conditional on the ICC/variance/correlation/mRMRe preprocessing selected in the original training sample, not full-pipeline bootstrap frequencies. The requested 5-8 feature range was used only as a prespecified training-CV parsimony rule within the 1-SE set; validation results were never used for alpha selection.\n'''
    (out/'complete_analysis_report.md').write_text(report,encoding='utf-8'); (out/'complete_analysis_report.txt').write_text(report,encoding='utf-8')
    summary={'counts':counts,'alpha':float(alpha),'alpha_rule':rule,'bootstrap_successful':int(valid.sum()),'final_features':[n for n,z in zip(final_names,nonzero) if z],'EPV':float(epv),'target_met':bool(target_met),'train_harrell':[float(x) for x in tc[:3]],'validation_harrell':[float(x) for x in vcx[:3]],'validation_uno':float(uno),'tau':float(tau)}
    (out/'analysis_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    rows=[]
    for p in sorted(out.glob('*')):
        if p.is_file() and p.name!='output_md5_manifest.csv': rows.append((p.name,p.stat().st_size,md5(p)))
    pd.DataFrame(rows,columns=['file','bytes','md5']).to_csv(out/'output_md5_manifest.csv',index=False)
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__': main()
