# Prompt for Antigravity — CAD multi-vessel prediction (Z-Alizadeh Sani)

Paste everything below into Antigravity, with the workspace set to this `cardml/` folder.

---

## Role
You are a senior ML engineer. Work inside this folder. Do **not** touch the raw data or `docs/`. Everything you write must run top-to-bottom in the `cardml` kernel (`.venv`, packages in `requirements.txt`).

## Inputs
- Data: `extention+of+z+alizadeh+sani+dataset/extention of Z-Alizadeh sani dataset.xlsx`
  - Sheet `Sheet 1 - Table 1`: 303 rows × 59 cols, no NaNs. Sheet `Sheet1` is empty (headers only), so ignore it and say so.
- Reference paper: `docs/1-s2.0-S0169260717314189-main.pdf` (Alizadehsani 2018). Brief: `docs/CardioVision_3D_Track_A_Analysis.docx`.
- Plan to follow: `/home/student/.claude/plans/now-create-a-ipynb-keen-toast.md`.

## Goal
Beat the paper's reported numbers (accuracy 96.4%, sensitivity 100%, specificity 88.1%, AUC 0.92 for its SVM feature-engineering method; single-classifier SVM 85.0 / 89.4 / 76.8 / 0.81) on **honest** held-out evaluation, for 4 targets: `Cath`, `LAD`, `LCX`, `RCA`.

## Hard rules (do not break)
1. `LAD`, `LCX`, `RCA`, `Cath` (and the audit columns `Cath_original`, `label_corrected`) are **never** inputs for any model, except the paper's stacked method, which may use *out-of-fold* predicted probabilities of the vessel models.
2. Row index 93 (Age 65, Male, LAD=Stenotic, LCX=Normal, RCA=Normal, Cath=Normal) violates the rule "CAD iff any vessel stenotic". Set `Cath=CAD`, keep `Cath_original`, set `label_corrected=1`, and print the audit table.
3. One stratified 80/20 split (seed 42, stratified on the LAD×LCX×RCA pattern), stored in the `split` column of `cad_clean.csv`. **The test set is touched once**, for the final report. All tuning, threshold selection and feature selection happen inside CV on the train rows.
4. Scaling, encoding, SMOTE and feature selection live inside the `Pipeline`, so the CV folds have no leakage.
5. Never report a number that came from tuning on the test set. If a result looks too good (>0.97 AUC on the vessel targets), look for leakage before reporting it.

## Deliverables
- `notebooks/01_data_cleaning_eda_feature_engineering.ipynb` → `data/processed/cad_clean.csv`, `cad_train_transformed.csv`, `cad_test_transformed.csv`, `data_quality_report.csv`.
- `notebooks/02_logistic_regression.ipynb`, `03_knn.ipynb`, `04_random_forest.ipynb`, `05_xgboost.ipynb`, `06_lightgbm.ipynb`.
- Many small cells, one step per cell, markdown headers on every section.
- Results saved to `results/` and fitted pipelines to `models/`.

## Notebook 01 must contain
Audit and range check vs the paper's Table 1 → label-consistency investigation → cleaning (Fmale→Female, Y/N→1/0, drop constant `Exertional CP`, flag the Obesity/BMI mismatch) → univariate EDA → target EDA → bivariate tests per target (Mann-Whitney + Cohen's d, χ² + Cramér's V) → correlation/VIF → feature engineering (`CP_type`, `NLR`, `TG_HDL`, `LDL_HDL`, `risk_factor_count`, `ecg_abnormal_count`, `age_group`; drop Weight and Length, keep BMI) → leakage guard → split → ColumnTransformer demo (`fit` / `fit_transform` on train, `transform` on test) → class-imbalance analysis (ratios, SMOTE demo, class weights) → save.

## Notebooks 02–06 must contain (same skeleton)
Load → pipeline (`prep → sampler → clf`) → Dummy and default baselines → depth/complexity validation curves → GridSearchCV then RandomizedSearchCV (wide, hundreds of combos) with RepeatedStratifiedKFold and multi-metric scoring (roc_auc, average_precision, balanced_accuracy, f1, recall) → CV fold sweep (k = 3, 5, 7, 10, repeated 5×3 / 10×3) → nested CV → threshold tuning from OOF train probabilities → final test metrics with bootstrap 95% CIs → confusion matrix / ROC / PR / calibration → **paper methods** (single classifier, OR rule of three vessel classifiers, stacked "feature engineering" method with OOF probabilities) → permutation importance → save.

## Ways to improve accuracy and every metric (try all, report what helps)
- Feature selection inside the CV (SelectKBest / L1 / SVM-weight like the paper, K in {10, 15, 20, 28, 35}).
- The stacked method with OOF probabilities of LAD/LCX/RCA/CAD models plus the top-K raw features, as in the paper.
- Class handling: none vs `class_weight` vs SMOTE vs `scale_pos_weight` as a tunable pipeline step.
- Threshold chosen for sensitivity ≥ 0.95 (the paper's clinical goal), and also for Youden J.
- Soft-voting / stacking ensemble across the 5 model families once the individual notebooks are done (a `07_ensemble.ipynb` is welcome).

## Report back
One table per target: model × {accuracy, balanced acc, sensitivity, specificity, precision, F1, ROC-AUC, PR-AUC, Brier} on the test set, next to the paper's Table 3. State clearly that the dataset has 303 rows (not the paper's 500), so the comparison is indicative, and list any step you could not verify.

## Before you finish
`jupyter nbconvert --to notebook --execute --inplace` every notebook in order, with zero errors. Assert in notebook 01: 303 rows, `label_corrected.sum()==1`, `Cath == (LAD|LCX|RCA)` for all rows, no target in the feature list.
