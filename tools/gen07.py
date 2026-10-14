import sys; sys.path.insert(0,'/mnt/9035d428-606e-456a-8b00-3b244a71810b/cardml/tools')
from nbgen import NB
n=NB()
n.md("""# 07 — Ensemble of the five tuned models
Combines the saved out-of-fold (train) and test probabilities of Logistic Regression, KNN, Random Forest, XGBoost and LightGBM. Nothing is re-tuned on the test set: ensemble weights and thresholds come from **out-of-fold train probabilities only**; the test set is scored once.""")
n.code("""import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,roc_auc_score,average_precision_score,f1_score,precision_score,brier_score_loss,confusion_matrix,roc_curve)
pd.set_option('display.width',220,'display.max_columns',40); sns.set_theme(style='whitegrid'); SEED=42
ROOT=Path('..').resolve(); RES=ROOT/'results'; OUT=ROOT/'data'/'processed'
MODELS={'lr':'Logistic Regression','knn':'KNN','rf':'Random Forest','xgb':'XGBoost','lgbm':'LightGBM'}; TARGETS=['Cath','LAD','LCX','RCA']
df=pd.read_csv(OUT/'cad_clean.csv',keep_default_na=False); tr=df[df.split=='train']; te=df[df.split=='test']
Y_tr={t:tr[t].values for t in TARGETS}; Y_te={t:te[t].values for t in TARGETS}
oof={m:pd.read_csv(RES/f'{m}_oof_train_proba.csv') for m in MODELS}; tp={m:pd.read_csv(RES/f'{m}_test_proba.csv') for m in MODELS}
for m in MODELS: assert (oof[m].patient_idx.values==tr.patient_idx.values).all() and (tp[m].patient_idx.values==te.patient_idx.values).all()
print('loaded',list(MODELS))""")
n.md("## 1. Individual models on out-of-fold train probabilities")
n.code("""rows=[{'target':t,'model':MODELS[m],'oof_auc':roc_auc_score(Y_tr[t],oof[m][t]),'test_auc':roc_auc_score(Y_te[t],tp[m][t])} for t in TARGETS for m in MODELS]
ind=pd.DataFrame(rows); ind.pivot(index='model',columns='target',values='oof_auc').round(3).join(ind.pivot(index='model',columns='target',values='test_auc').round(3),rsuffix='_test')""")
n.md("## 2. Ensembles: simple mean, AUC-weighted mean, rank mean, logistic stacking")
n.code("""def thr_youden(y,p):
    f,t,th=roc_curve(y,p); return float(th[np.argmax(t-f)])
def thr_sens(y,p,s=0.95):
    f,t,th=roc_curve(y,p); return float(th[np.where(t>=s)[0][0]])
def metrics_at(y,p,thr):
    yh=(p>=thr).astype(int); tn,fp,fn,tp_=confusion_matrix(y,yh,labels=[0,1]).ravel(); se=tp_/(tp_+fn); sp=tn/(tn+fp)
    return dict(threshold=float(thr),accuracy=accuracy_score(y,yh),balanced_acc=(se+sp)/2,sensitivity=se,specificity=sp,precision=precision_score(y,yh,zero_division=0),
                f1=f1_score(y,yh),roc_auc=roc_auc_score(y,p),pr_auc=average_precision_score(y,p),brier=brier_score_loss(y,p))
def build(t):
    O=np.column_stack([oof[m][t] for m in MODELS]); T=np.column_stack([tp[m][t] for m in MODELS]); y=Y_tr[t]
    auc=np.array([roc_auc_score(y,O[:,i]) for i in range(O.shape[1])]); w=np.exp((auc-auc.max())*20); w/=w.sum()
    # rank mean (rank each model within its own set)
    rk=lambda A:np.column_stack([pd.Series(A[:,i]).rank(pct=True).values for i in range(A.shape[1])])
    lg=LogisticRegression(C=1.0,max_iter=2000); lo=lambda A:np.log(np.clip(A,1e-4,1-1e-4)/(1-np.clip(A,1e-4,1-1e-4)))
    oof_st=cross_val_predict(lg,lo(O),y,cv=StratifiedKFold(5,shuffle=True,random_state=SEED),method='predict_proba')[:,1]
    lg.fit(lo(O),y); te_st=lg.predict_proba(lo(T))[:,1]
    return {'mean':(O.mean(1),T.mean(1)),'auc-weighted':(O@w,T@w),'stack-logit':(oof_st,te_st)}, w
ens,weights={},{}
for t in TARGETS: ens[t],weights[t]=build(t)
pd.DataFrame(weights,index=list(MODELS)).round(3).rename_axis('AUC weights')""")
n.code("""rows=[]
for t in TARGETS:
    for name,(o,p) in ens[t].items():
        rows.append({'target':t,'ensemble':name,'oof_auc':roc_auc_score(Y_tr[t],o),'test_auc':roc_auc_score(Y_te[t],p)})
for t in TARGETS:
    best_single=ind[ind.target==t].sort_values('oof_auc').iloc[-1]
    rows.append({'target':t,'ensemble':'best single: '+best_single.model,'oof_auc':best_single.oof_auc,'test_auc':best_single.test_auc})
pd.DataFrame(rows).pivot(index='ensemble',columns='target',values=['oof_auc','test_auc']).round(3)""")
n.md("## 3. Choose the ensemble per target by OOF AUC, then score the test set once")
n.code("""final,chosen=[],{}
for t in TARGETS:
    name=max(ens[t],key=lambda k:roc_auc_score(Y_tr[t],ens[t][k][0])); chosen[t]=name; o,p=ens[t][name]
    for rule,thr in {'default_0.5':0.5,'youden':thr_youden(Y_tr[t],o),'sens>=0.95':thr_sens(Y_tr[t],o,0.95)}.items():
        final.append({'target':t,'strategy':'ensemble:'+name,'threshold_rule':rule,**metrics_at(Y_te[t],p,thr)})
final=pd.DataFrame(final); final.to_csv(RES/'ensemble_test_metrics.csv',index=False); print(chosen)
final[['target','strategy','threshold_rule','threshold','accuracy','balanced_acc','sensitivity','specificity','precision','f1','roc_auc','pr_auc','brier']].round(3)""")
n.md("## 4. Paper-style 3-way strategies with the ensemble probabilities")
n.code("""pL={t:ens[t][chosen[t]] for t in TARGETS}
thr={t:{'default_0.5':0.5,'youden':thr_youden(Y_tr[t],pL[t][0])} for t in TARGETS}
rows=[]
for rule in ('default_0.5','youden'):
    pred=np.zeros(len(te),int)
    for t in ['LAD','LCX','RCA']: pred|=(pL[t][1]>=thr[t][rule]).astype(int)
    y=Y_te['Cath']; tn,fp,fn,tp_=confusion_matrix(y,pred,labels=[0,1]).ravel()
    rows.append({'strategy':'ensemble OR-rule @'+rule,'accuracy':accuracy_score(y,pred),'sensitivity':tp_/(tp_+fn),'specificity':tn/(tn+fp),'roc_auc':roc_auc_score(y,1-np.prod([1-pL[t][1] for t in ['LAD','LCX','RCA']],axis=0))})
# stacking of vessel + CAD probabilities (OOF) into a final CAD model, paper Fig.1
Zo=np.column_stack([pL[t][0] for t in ['LAD','LCX','RCA','Cath']]); Zt=np.column_stack([pL[t][1] for t in ['LAD','LCX','RCA','Cath']])
lo=lambda A:np.log(np.clip(A,1e-4,1-1e-4)/(1-np.clip(A,1e-4,1-1e-4)))
fin=LogisticRegression(C=1.0,max_iter=2000); o2=cross_val_predict(fin,lo(Zo),Y_tr['Cath'],cv=StratifiedKFold(5,shuffle=True,random_state=SEED),method='predict_proba')[:,1]
fin.fit(lo(Zo),Y_tr['Cath']); p2=fin.predict_proba(lo(Zt))[:,1]
for rule,th in {'default_0.5':0.5,'youden':thr_youden(Y_tr['Cath'],o2),'sens>=0.95':thr_sens(Y_tr['Cath'],o2,0.95)}.items():
    m=metrics_at(Y_te['Cath'],p2,th); rows.append({'strategy':'ensemble stacked (paper FE) @'+rule,**{k:m[k] for k in ('accuracy','sensitivity','specificity','roc_auc')}})
for rule in ('default_0.5','youden','sens>=0.95'):
    m=final[(final.target=='Cath')&(final.threshold_rule==rule)].iloc[0]; rows.append({'strategy':'ensemble direct @'+rule,**{k:m[k] for k in ('accuracy','sensitivity','specificity','roc_auc')}})
paper=pd.DataFrame([{'strategy':'PAPER single (SVM)','accuracy':.850,'sensitivity':.894,'specificity':.768,'roc_auc':.81},{'strategy':'PAPER three-classifier (SVM)','accuracy':.810,'sensitivity':.814,'specificity':.798,'roc_auc':.87},
                    {'strategy':'PAPER feature engineering (SVM)','accuracy':.964,'sensitivity':1.0,'specificity':.881,'roc_auc':.92}])
cmp=pd.concat([paper,pd.DataFrame(rows)]).set_index('strategy'); cmp.round(3)""")
n.md("## 5. Cross-model leaderboard (held-out test, Youden threshold) and nested-CV leaderboard (all 303 patients)")
n.code("""lb=[]
for m,nm in MODELS.items():
    d=pd.read_csv(RES/f'{m}_test_metrics_direct.csv'); d=d[d.threshold_rule=='youden']; d.insert(0,'model',nm); lb.append(d)
e=final[final.threshold_rule=='youden'].copy(); e.insert(0,'model','ENSEMBLE'); lb.append(e)
lb=pd.concat(lb); lb[['model','target','accuracy','balanced_acc','sensitivity','specificity','precision','f1','roc_auc','pr_auc']].round(3).sort_values(['target','roc_auc'],ascending=[True,False])""")
n.code("""nest=pd.concat({MODELS[m]:pd.read_csv(RES/f'{m}_nested_cv.csv').set_index('target')[['accuracy','balanced_acc','sensitivity','specificity','precision','f1','roc_auc']] for m in MODELS})
nest.round(3)""")
n.code("""fig,axes=plt.subplots(1,4,figsize=(20,4))
for ax,t in zip(axes,TARGETS):
    d=nest.xs(t,level=1); d[['accuracy','sensitivity','specificity','roc_auc']].plot.bar(ax=ax); ax.set_title(f'{t}: nested 10-fold CV'); ax.set_ylim(0.2,1); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(RES/'figures'/'ensemble_nested_leaderboard.png',dpi=110); plt.show()""")
n.code("""nest.to_csv(RES/'nested_cv_leaderboard.csv'); lb.to_csv(RES/'test_leaderboard.csv',index=False); cmp.to_csv(RES/'paper_comparison_cath.csv'); print('saved')""")
n.save('/mnt/9035d428-606e-456a-8b00-3b244a71810b/cardml/notebooks/07_ensemble.ipynb')
