import sys; sys.path.insert(0,'/mnt/9035d428-606e-456a-8b00-3b244a71810b/cardml/tools')
from nbgen import NB
n=NB()
n.md("""# 01 — Data cleaning, EDA, feature engineering (Z-Alizadeh Sani, extended)
Reference: Alizadehsani et al., *Comput Methods Programs Biomed* 162 (2018) 119–127.

**Pipeline of this notebook:** audit → label-consistency check → cleaning → univariate/target/bivariate EDA → correlation & redundancy → feature engineering → leakage guard → stratified split → preprocessing pipeline (`fit` / `fit_transform` / `transform`) → class-imbalance study → save `cad_clean.csv`.""")
n.md("## 0. Setup")
n.code("""import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
pd.set_option('display.width',220,'display.max_columns',100,'display.max_rows',200)
sns.set_theme(style='whitegrid'); SEED=42""")
n.code("""ROOT = Path('..').resolve()
RAW  = ROOT/'extention+of+z+alizadeh+sani+dataset'/'extention of Z-Alizadeh sani dataset.xlsx'
OUT  = ROOT/'data'/'processed'; FIG = ROOT/'results'/'figures'
OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
TARGETS = ['Cath','LAD','LCX','RCA']
print(RAW.exists())""")
n.md("## 1. Read the whole workbook (both sheets)")
n.code("""xl = pd.ExcelFile(RAW)
print(xl.sheet_names)
sheets = {s: xl.parse(s) for s in xl.sheet_names}
for s,d in sheets.items(): print(f'{s!r}: shape={d.shape}')""")
n.code("""# Sheet1 has the extended 100-column schema but zero rows -> unusable
print('Sheet1 rows:', len(sheets['Sheet1']), '| columns:', sheets['Sheet1'].shape[1])
raw = sheets['Sheet 1 - Table 1'].copy()
raw.head()""")
n.code("raw.tail()")
n.code("""profile = pd.DataFrame({'dtype':raw.dtypes.astype(str),'n_unique':raw.nunique(),'n_missing':raw.isna().sum(),
                        'sample_values':[list(raw[c].dropna().unique()[:6]) for c in raw]})
profile""")
n.code("raw.describe().T")
n.code("raw.describe(include='object').T")
n.md("## 2. Data audit")
n.code("""audit = {'rows':len(raw),'cols':raw.shape[1],'missing_total':int(raw.isna().sum().sum()),
         'duplicate_rows':int(raw.duplicated().sum()),
         'constant_columns':[c for c in raw if raw[c].nunique()==1]}
audit""")
n.code("""# Range check against Table 1 of the paper
paper_ranges = {'Age':(30,86),'Weight':(48,120),'BMI':(18,41),'BP':(90,190),'PR':(50,110),'FBS':(62,400),'CR':(0.5,2.2),
 'TG':(37,1050),'LDL':(18,232),'HDL':(15,111),'BUN':(6,52),'ESR':(1,90),'HB':(8.9,17.6),'K':(3.0,6.6),'Na':(128,156),
 'WBC':(3700,18000),'Lymph':(7,60),'Neut':(32,89),'PLT':(25,742),'EF-TTE':(15,60),'Region RWMA':(0,4)}
rc = pd.DataFrame([{'feature':k,'paper_min':lo,'paper_max':hi,'data_min':raw[k].min(),'data_max':raw[k].max(),
                    'within_range':(raw[k].min()>=lo-0.5) and (raw[k].max()<=hi+0.5)} for k,(lo,hi) in paper_ranges.items()])
rc""")
n.code("""for c in raw.select_dtypes('object'): print(f'{c:22s}', raw[c].value_counts().to_dict())""")
n.md("""## 3. Label-consistency investigation
UCI / paper definition: *a patient has CAD if stenosis > 50 % in at least one of LAD, LCX or RCA.* So `Cath == CAD` must be equivalent to `LAD | LCX | RCA stenotic`. We test this before training.""")
n.code("pd.crosstab(raw['Cath'], [raw['LAD'],raw['LCX'],raw['RCA']])")
n.code("""raw['any_stenotic'] = ((raw.LAD=='Stenotic')|(raw.LCX=='Stenotic')|(raw.RCA=='Stenotic')).map({True:'CAD',False:'Normal'})
print(pd.crosstab(raw['any_stenotic'], raw['Cath'], rownames=['rule'], colnames=['Cath (supplied)']))
violations = raw[raw.any_stenotic!=raw.Cath]
violations[['Age','Sex','LAD','LCX','RCA','Cath','any_stenotic']]""")
n.md("""**Finding:** exactly one record (index 93) is labelled *Normal* although LAD is stenotic. The 40 patients with `Cath=CAD` and a normal LAD are consistent (LCX or RCA is stenotic).
**Decision (user-approved):** relabel `Cath → CAD` for that record, keep the supplied value in `Cath_original`, and flag it with `label_corrected`.""")
n.code("""df = raw.drop(columns='any_stenotic').copy()
df['Cath_original'] = df['Cath']
df['label_corrected'] = 0
idx = violations.index
df.loc[idx,'Cath'] = 'CAD'; df.loc[idx,'label_corrected'] = 1
print('rows corrected:', df.label_corrected.sum())
df.loc[idx,['Age','Sex','LAD','LCX','RCA','Cath_original','Cath','label_corrected']]""")
n.code("""chk = ((df.LAD=='Stenotic')|(df.LCX=='Stenotic')|(df.RCA=='Stenotic'))==(df.Cath=='CAD')
assert chk.all(); print('Cath == (LAD|LCX|RCA) for all', len(df), 'rows')""")
n.code("""# Vessel co-occurrence vs the paper's Appendix A (extended 500-sample dataset)
v = pd.DataFrame({t:(df[t]=='Stenotic').astype(int) for t in ['LAD','LCX','RCA']})
rules = []
for a in v.columns:
    for b in v.columns:
        if a!=b: rules.append({'rule':f'{a} => {b}','confidence':round(v.loc[v[a]==1,b].mean(),4)})
print(pd.DataFrame(rules))
print('paper (n=500): LAD=>LCX .542, LAD=>RCA .492, LCX=>LAD .807, LCX=>RCA .613, RCA=>LAD .763, RCA=>LCX .640')
print('all three stenotic: %.1f%% (paper 20.7%%)'%(100*(v.sum(axis=1)==3).mean()))""")
n.md("## 4. Cleaning")
n.code("""print(df.Sex.value_counts().to_dict())
df['Sex'] = df['Sex'].replace({'Fmale':'Female'})
print(df.Sex.value_counts().to_dict())""")
n.code("""# Exertional CP is constant (all N) -> no information
print(df['Exertional CP'].value_counts().to_dict())
df = df.drop(columns='Exertional CP')""")
n.code("""# Obesity is defined as BMI>25 in the paper; check the supplied flag and the BMI formula
print('max |BMI - Weight/Length^2| =', (df.Weight/(df.Length/100)**2 - df.BMI).abs().max())
mis = df[(df.Obesity.eq('Y')) != (df.BMI>25)]
print('Obesity/BMI mismatches:', len(mis)); mis[['Age','Weight','Length','BMI','Obesity']]""")
n.code("""# Y/N -> 1/0 for all two-level object columns (except Sex and the targets)
yn = [c for c in df.select_dtypes('object') if set(df[c].unique())<={'Y','N'}]
print(yn)
df[yn] = df[yn].replace({'Y':1,'N':0}).astype(int)
df['Sex'] = (df.Sex=='Male').astype(int)          # 1 = Male
for t in TARGETS+['Cath_original']:
    df[t] = df[t].map({'CAD':1,'Stenotic':1,'Normal':0}).astype(int)
df.dtypes.value_counts()""")
n.code("""# Chest-pain indicators: are they mutually exclusive?
cp = df[['Typical Chest Pain','Atypical','Nonanginal']]
print(cp.sum(axis=1).value_counts().to_dict(), '<- number of chest-pain flags per patient')
# smoking flags
print(pd.crosstab(df['Current Smoker'], df['EX-Smoker']))""")
n.code("""print('Cleaned shape:', df.shape)
df[TARGETS].apply(lambda s: s.value_counts()).T""")
n.md("## 5. Univariate EDA")
n.code("""num_cols = ['Age','Weight','Length','BMI','BP','PR','FBS','CR','TG','LDL','HDL','BUN','ESR','HB','K','Na','WBC','Lymph','Neut','PLT','EF-TTE']
ord_cols = ['Function Class','Region RWMA']
cat_cols = ['BBB','VHD']
bin_cols = [c for c in df.columns if df[c].nunique()==2 and c not in TARGETS+['Cath_original','label_corrected']]
print(len(num_cols),'numeric |',len(bin_cols),'binary |',cat_cols,'categorical |',ord_cols,'ordinal')""")
n.code("""fig,axes = plt.subplots(6,4,figsize=(18,20)); axes=axes.ravel()
for ax,c in zip(axes,num_cols):
    sns.histplot(df[c],kde=True,ax=ax,color='steelblue'); ax.set_title(c)
for ax in axes[len(num_cols):]: ax.axis('off')
plt.tight_layout(); plt.savefig(FIG/'01_numeric_hist.png',dpi=110); plt.show()""")
n.code("""fig,axes = plt.subplots(6,4,figsize=(18,18)); axes=axes.ravel()
for ax,c in zip(axes,num_cols):
    sns.boxplot(x=df[c],ax=ax,color='lightcoral'); ax.set_title(c)
for ax in axes[len(num_cols):]: ax.axis('off')
plt.tight_layout(); plt.savefig(FIG/'01_numeric_box.png',dpi=110); plt.show()""")
n.code("""sk = pd.DataFrame({'skew':df[num_cols].skew(),'kurtosis':df[num_cols].kurt()})
q1,q3 = df[num_cols].quantile(.25), df[num_cols].quantile(.75); iqr=q3-q1
sk['iqr_outliers'] = ((df[num_cols]<q1-1.5*iqr)|(df[num_cols]>q3+1.5*iqr)).sum()
sk['|z|>3'] = (np.abs(stats.zscore(df[num_cols]))>3).sum()
sk.sort_values('skew',key=abs,ascending=False)""")
n.md("Outliers are inside the clinically documented ranges of the paper's Table 1, so they are genuine measurements (e.g. TG up to 1050, FBS up to 400). They are **kept**; skew is handled by a `log1p` transform inside the pipeline.")
n.code("""fig,axes = plt.subplots(6,6,figsize=(20,16)); axes=axes.ravel()
for ax,c in zip(axes,bin_cols):
    df[c].value_counts().sort_index().plot.bar(ax=ax,color='seagreen'); ax.set_title(c,fontsize=9); ax.set_xlabel('')
for ax in axes[len(bin_cols):]: ax.axis('off')
plt.tight_layout(); plt.savefig(FIG/'01_binary_counts.png',dpi=110); plt.show()""")
n.code("""fig,axes = plt.subplots(1,4,figsize=(18,3.5))
for ax,c in zip(axes,['BBB','VHD','Function Class','Region RWMA']):
    sns.countplot(x=df[c],ax=ax,color='slateblue'); ax.set_title(c)
plt.tight_layout(); plt.show()""")
n.md("## 6. Target analysis")
n.code("""fig,axes = plt.subplots(1,4,figsize=(16,3.5))
for ax,t in zip(axes,TARGETS):
    vc=df[t].value_counts().sort_index(); ax.bar(['Normal','Positive'],vc.values,color=['#4c9','#e55']); ax.set_title(f'{t}: {vc[1]}/{vc.sum()} = {vc[1]/vc.sum():.1%}')
plt.tight_layout(); plt.savefig(FIG/'01_targets.png',dpi=110); plt.show()""")
n.code("""df['n_vessels'] = df[['LAD','LCX','RCA']].sum(axis=1)
print(df.n_vessels.value_counts().sort_index())
pat = df.groupby(['LAD','LCX','RCA']).size().rename('n').reset_index(); pat""")
n.md("## 7. Bivariate analysis — every feature vs every target")
n.code("""def cohen_d(a,b):
    s=np.sqrt(((len(a)-1)*a.var()+(len(b)-1)*b.var())/(len(a)+len(b)-2)); return (a.mean()-b.mean())/s if s>0 else 0
def cramers_v(x,y):
    ct=pd.crosstab(x,y); chi2=stats.chi2_contingency(ct,correction=False)[0]; n=ct.values.sum()
    return np.sqrt(chi2/(n*(min(ct.shape)-1)))
def assoc_table(data,target,numeric,categorical):
    rows=[]
    for c in numeric:
        a,b=data.loc[data[target]==1,c],data.loc[data[target]==0,c]
        rows.append({'feature':c,'kind':'numeric','effect':abs(cohen_d(a,b)),'p_value':stats.mannwhitneyu(a,b).pvalue})
    for c in categorical:
        ct=pd.crosstab(data[c],data[target])
        rows.append({'feature':c,'kind':'categorical','effect':cramers_v(data[c],data[target]),'p_value':stats.chi2_contingency(ct)[1]})
    return pd.DataFrame(rows).sort_values('effect',ascending=False).reset_index(drop=True)
cat_like = bin_cols+cat_cols+ord_cols
assoc = {t:assoc_table(df,t,num_cols,cat_like) for t in TARGETS}
assoc['Cath'].head(20)""")
n.code("assoc['LAD'].head(20)")
n.code("assoc['LCX'].head(20)")
n.code("assoc['RCA'].head(20)")
n.code("""# side-by-side top-10 per target (numeric effect = |Cohen d|, categorical effect = Cramér's V: indicative, not strictly comparable)
top = pd.concat({t:assoc[t].head(10)[['feature','effect']].reset_index(drop=True) for t in TARGETS},axis=1); top""")
n.code("""fig,axes = plt.subplots(2,4,figsize=(18,7))
for ax,c in zip(axes.ravel(),['Age','EF-TTE','BP','FBS','ESR','TG','K','PR']):
    sns.boxplot(data=df,x='Cath',y=c,ax=ax,palette=['#4c9','#e55']); ax.set_title(f'{c} vs Cath')
plt.tight_layout(); plt.savefig(FIG/'01_box_vs_cath.png',dpi=110); plt.show()""")
n.code("""fig,axes = plt.subplots(2,4,figsize=(18,7))
for ax,c in zip(axes.ravel(),['Typical Chest Pain','Atypical','Nonanginal','HTN','DM','Region RWMA','Function Class','Sex']):
    (df.groupby(c)['Cath'].mean()*100).plot.bar(ax=ax,color='indianred'); ax.set_ylabel('% CAD'); ax.set_title(c)
plt.tight_layout(); plt.show()""")
n.code("""# Age vs target by sex, per vessel
fig,axes = plt.subplots(1,3,figsize=(16,4))
for ax,t in zip(axes,['LAD','LCX','RCA']):
    sns.violinplot(data=df,x=t,y='Age',hue='Sex',split=True,ax=ax); ax.set_title(f'Age by {t} / sex (1=Male)')
plt.tight_layout(); plt.show()""")
n.md("## 8. Correlation and redundancy")
n.code("""plt.figure(figsize=(13,10))
sp = df[num_cols+ord_cols].corr(method='spearman')
sns.heatmap(sp,cmap='coolwarm',center=0,annot=False,square=True); plt.title('Spearman correlation (numeric + ordinal)')
plt.tight_layout(); plt.savefig(FIG/'01_spearman.png',dpi=110); plt.show()""")
n.code("""pairs = sp.where(np.triu(np.ones(sp.shape),1).astype(bool)).stack().rename('rho').reset_index()
pairs['abs']=pairs.rho.abs(); pairs.sort_values('abs',ascending=False).head(12)""")
n.code("""cv_cols = bin_cols+cat_cols
cm = pd.DataFrame(np.eye(len(cv_cols)),index=cv_cols,columns=cv_cols)
for i,a in enumerate(cv_cols):
    for b in cv_cols[i+1:]:
        try: cm.loc[a,b]=cm.loc[b,a]=cramers_v(df[a],df[b])
        except Exception: cm.loc[a,b]=cm.loc[b,a]=np.nan
plt.figure(figsize=(14,12)); sns.heatmap(cm,cmap='viridis',square=True); plt.title("Cramér's V (binary / categorical)")
plt.tight_layout(); plt.savefig(FIG/'01_cramers.png',dpi=110); plt.show()""")
n.code("""from statsmodels.stats.outliers_influence import variance_inflation_factor
X = StandardScaler().fit_transform(df[num_cols]); X = np.c_[np.ones(len(X)),X]
vif = pd.Series([variance_inflation_factor(X,i) for i in range(1,X.shape[1])],index=num_cols,name='VIF').sort_values(ascending=False)
vif.head(10)""")
n.md("`BMI` is an exact function of `Weight` and `Length` (VIF very high) → keep BMI, drop Weight/Length. `Neut` and `Lymph` are strongly (negatively) correlated, which motivates the neutrophil-to-lymphocyte ratio.")
n.md("## 9. Feature engineering")
n.code("""fe = df.copy()
# chest-pain type: the three indicators are mutually exclusive in this data, 30 patients have none
fe['CP_type'] = np.select([fe['Typical Chest Pain']==1, fe.Atypical==1, fe.Nonanginal==1],['Typical','Atypical','Nonanginal'],'NoCP')
fe['NLR']      = fe.Neut/fe.Lymph
fe['TG_HDL']   = fe.TG/fe.HDL
fe['LDL_HDL']  = fe.LDL/fe.HDL
fe['pulse_pressure_proxy'] = fe.BP/fe.PR
fe['risk_factor_count'] = fe[['DM','HTN','Current Smoker','FH','DLP','Obesity']].sum(axis=1)
ecg = ['Q Wave','St Elevation','St Depression','Tinversion','LVH','Poor R Progression']
fe['ecg_abnormal_count'] = fe[ecg].sum(axis=1) + (fe.BBB!='N').astype(int)
fe['age_group'] = pd.cut(fe.Age,[0,45,55,65,120],labels=['<=45','46-55','56-65','>65']).astype(str)
fe['symptom_score'] = fe['Typical Chest Pain']*2 + fe.Atypical + fe['Function Class'] + fe.Dyspnea
fe['ef_low'] = (fe['EF-TTE']<50).astype(int)
fe = fe.drop(columns=['Weight','Length'])
new = ['NLR','TG_HDL','LDL_HDL','pulse_pressure_proxy','risk_factor_count','ecg_abnormal_count','symptom_score','ef_low']
fe[new+['CP_type','age_group']].describe(include='all').T""")
n.code("""# do the engineered features carry signal? (compare with the best raw features)
print(assoc_table(fe,'Cath',new,['CP_type','age_group']).to_string())
print(assoc_table(fe,'LAD',new,['CP_type','age_group']).head(5).to_string())""")
n.code("""print('CP_type vs Cath (% CAD):'); print((fe.groupby('CP_type').Cath.agg(['mean','count'])).round(3))""")
n.md("## 10. Leakage guard")
n.code("""AUDIT = ['Cath_original','label_corrected','n_vessels']
ALL_LABELS = TARGETS+AUDIT
feature_df = fe.drop(columns=ALL_LABELS)
FEATURES = list(feature_df.columns)
assert not set(FEATURES)&set(ALL_LABELS), 'target leak!'
print(len(FEATURES),'input features; none of',ALL_LABELS,'is an input')""")
n.md("## 11. Stratified train / test split")
n.code("""# stratify on the vessel pattern LAD x LCX x RCA: it determines Cath and keeps all 4 target ratios close
strat = fe.LAD.astype(str)+fe.LCX.astype(str)+fe.RCA.astype(str)
print(strat.value_counts().to_dict())
idx_tr, idx_te = train_test_split(fe.index, test_size=0.20, stratify=strat, random_state=SEED)
fe['split'] = np.where(fe.index.isin(idx_te),'test','train')
print(fe.split.value_counts().to_dict())
pd.concat({t:fe.groupby('split')[t].mean().round(3) for t in TARGETS},axis=1)""")
n.md("## 12. Preprocessing pipeline: `fit`, `fit_transform`, `transform`")
n.code("""skew_cols = ['FBS','TG','ESR','WBC','PLT','CR','TG_HDL','NLR','LDL_HDL']            # log1p then scale
scale_cols = [c for c in num_cols if c in FEATURES and c not in skew_cols]+['pulse_pressure_proxy','risk_factor_count','ecg_abnormal_count','symptom_score']
ordinal_cols = ['VHD','Function Class','Region RWMA']
nominal_cols = ['BBB','CP_type','age_group']
binary_cols  = [c for c in FEATURES if c not in skew_cols+scale_cols+ordinal_cols+nominal_cols]
assert sorted(skew_cols+scale_cols+ordinal_cols+nominal_cols+binary_cols)==sorted(FEATURES)
print(len(skew_cols),len(scale_cols),len(ordinal_cols),len(nominal_cols),len(binary_cols))""")
n.code("""def build_preprocessor():
    return ColumnTransformer([
        ('skew', Pipeline([('log',FunctionTransformer(np.log1p,feature_names_out='one-to-one')),('sc',StandardScaler())]), skew_cols),
        ('num',  StandardScaler(), scale_cols),
        ('ord',  Pipeline([('enc',OrdinalEncoder(categories=[['N','mild','Moderate','Severe'],[0,1,2,3],[0,1,2,3,4]])),('sc',StandardScaler())]), ordinal_cols),
        ('nom',  OneHotEncoder(handle_unknown='ignore',sparse_output=False), nominal_cols),
        ('bin',  'passthrough', binary_cols)], verbose_feature_names_out=False)
X_train = fe.loc[fe.split=='train',FEATURES]; X_test = fe.loc[fe.split=='test',FEATURES]
prep = build_preprocessor()
prep.fit(X_train)                                  # learn means/stds/categories from TRAIN only
Xtr_t = prep.transform(X_train)                    # equals fit_transform(X_train)
Xte_t = prep.transform(X_test)                     # test is only transformed
print(Xtr_t.shape, Xte_t.shape)""")
n.code("""assert np.allclose(Xtr_t, build_preprocessor().fit_transform(X_train))      # fit().transform() == fit_transform()
names = prep.get_feature_names_out()
tr = pd.DataFrame(Xtr_t,columns=names); te = pd.DataFrame(Xte_t,columns=names)
chk = pd.DataFrame({'train_mean':tr[skew_cols+scale_cols].mean(),'train_std':tr[skew_cols+scale_cols].std(),
                    'test_mean':te[skew_cols+scale_cols].mean(),'test_std':te[skew_cols+scale_cols].std()}).round(3)
chk.head(12)""")
n.code("""assert tr[skew_cols+scale_cols].mean().abs().max()<1e-8
print('train means≈0, std≈1; test means deviate slightly -> test statistics were NOT used for fitting')
print('encoded feature count:',len(names)); list(names)""")
n.md("## 13. Class imbalance")
n.code("""imb = pd.DataFrame({t:{'positive':int(fe[t].sum()),'negative':int((1-fe[t]).sum()),
        'pos_rate':round(fe[t].mean(),3),'imbalance_ratio(neg/pos)':round((1-fe[t]).sum()/fe[t].sum(),2),
        'scale_pos_weight':round((1-fe.loc[fe.split=='train',t]).sum()/fe.loc[fe.split=='train',t].sum(),2)} for t in TARGETS}).T; imb""")
n.code("""# SMOTE demonstration on the TRAIN set only (inside model CV it lives in the pipeline)
for t in TARGETS:
    ytr = fe.loc[fe.split=='train',t]
    Xs,ys = SMOTE(random_state=SEED,k_neighbors=5).fit_resample(Xtr_t,ytr)
    print(f'{t}: before {dict(ytr.value_counts().sort_index())}  after SMOTE {dict(pd.Series(ys).value_counts().sort_index())}')""")
n.md("""**Class-handling strategy for modelling:** the imbalance is moderate (Cath 71/29, LAD 58/42, LCX 39/61, RCA 38/62), so no single treatment is assumed best. Each model notebook tunes *none* vs *class weights / scale_pos_weight* vs *SMOTE* as a pipeline hyper-parameter, and chooses the decision threshold from out-of-fold train probabilities.""")
n.md("## 14. Save outputs")
n.code("""clean = fe.reset_index().rename(columns={'index':'patient_idx'})
clean = clean[['patient_idx']+FEATURES+ALL_LABELS+['split']]
clean.to_csv(OUT/'cad_clean.csv',index=False)
trans_tr = tr.copy(); trans_tr[TARGETS]=fe.loc[fe.split=='train',TARGETS].values
trans_te = te.copy(); trans_te[TARGETS]=fe.loc[fe.split=='test',TARGETS].values
trans_tr.to_csv(OUT/'cad_train_transformed.csv',index=False); trans_te.to_csv(OUT/'cad_test_transformed.csv',index=False)
report = profile.copy(); report['paper_range_ok']=report.index.map(lambda c: dict(zip(rc.feature,rc.within_range)).get(c,np.nan))
report.to_csv(OUT/'data_quality_report.csv')
print(clean.shape, trans_tr.shape, trans_te.shape)""")
n.code("""chk = pd.read_csv(OUT/'cad_clean.csv')
assert len(chk)==303 and chk.label_corrected.sum()==1
assert ((chk.LAD|chk.LCX|chk.RCA)==chk.Cath).all()
assert not set(TARGETS)&set(FEATURES)
print('all assertions passed'); chk.head()""")
n.md("""## Summary
* 303 patients, no missing values, no duplicates; `Sheet1` (extended schema) is empty and ignored.
* One label inconsistency (row 93) fixed per the UCI/paper definition and logged.
* `Fmale` normalised, constant `Exertional CP` dropped, Y/N encoded, BMI kept / Weight+Length dropped.
* New features: `CP_type, NLR, TG_HDL, LDL_HDL, pulse_pressure_proxy, risk_factor_count, ecg_abnormal_count, symptom_score, ef_low, age_group`.
* Leakage guard in place; one frozen 80/20 split in `cad_clean.csv`; preprocessing is rebuilt inside each model pipeline.""")
n.save('/mnt/9035d428-606e-456a-8b00-3b244a71810b/cardml/notebooks/01_data_cleaning_eda_feature_engineering.ipynb')
