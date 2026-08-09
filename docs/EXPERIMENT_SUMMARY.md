# LendingClub Unified 500,000-Loan Experiment

Run ID: `run_20260806T154718Z`

## Design

- Completed-loan population after cleaning/filtering: 1,348,099
- Modelling sample: 500,000
- Sampling: proportional stratified random sampling by default and issue year
- Random state: 42
- Split shares: 64% train / 16% validation / 20% test
- Random split: main comparison
- Temporal split: out-of-time robustness check
- RQ1: ten-model comparison using Combined features.
- RQ2: Logistic Regression, LightGBM, and EBM across five feature groups.
- The outer data, validation, calibration, and evaluation protocol is common across estimators; imbalance controls use prespecified estimator-appropriate mechanisms rather than forcing class weights onto every model.
- RQ2 grouping definition: `v2_fico_and_credit_history_in_financial_credit`. FICO score and credit-history length are assigned to the Financial/Credit block; the Borrower block contains borrower and application characteristics.
- RQ3 SHAP sample: 5,000 randomly selected final-test observations.
- Compute backend: Hybrid: CUDA (NVIDIA GeForce RTX 5060 Laptop GPU) for MLP/FT-Transformer; CPU for statistical and classical/tree models
- Logistic predictive probabilities, main interpretation coefficients, and odds ratios are extracted from the same full-training fitted model.
- Predictive Logistic and Probit models use the complete training partition; no separate subsample supplies the main coefficient interpretation.
- MLP and FT-Transformer use the complete training partition; validation-based early stopping controls epochs, not sample size.
- The test set was used only for final evaluation; validation selected parameters, epochs, and thresholds.
- Hyperparameters, thresholds, and calibration maps were selected separately within the random and temporal designs using their respective validation partitions.
- Every model first produces its raw score or native probability. A common sigmoid map is then fitted on that model's validation predictions and applied unchanged to its final-test predictions. Main Brier scores and reliability curves therefore use a uniform validation-based calibration protocol; raw probability outputs are retained separately for audit and appendix comparison.

## Sample Representativeness

- Maximum categorical share difference: 0.000368
- Maximum absolute standardized mean difference: 0.001142
- Maximum numerical KS statistic: 0.000985
- Maximum default-by-issue-year stratum share difference: 0.000001

## RQ1: Random Combined-Feature Results

| Model | AUC | Gini | KS | AP | Brier | Precision | Recall | F1 | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.7267 | 0.4534 | 0.3297 | 0.3984 | 0.1424 | 0.3341 | 0.6549 | 0.4425 | 0.2113 |
| MLP Neural Network | 0.7264 | 0.4527 | 0.3291 | 0.3994 | 0.1424 | 0.3394 | 0.6312 | 0.4414 | 0.2176 |
| FT-Transformer | 0.7240 | 0.4480 | 0.3247 | 0.3960 | 0.1428 | 0.3337 | 0.6430 | 0.4394 | 0.2157 |
| Explainable Boosting Machine | 0.7231 | 0.4461 | 0.3235 | 0.3935 | 0.1430 | 0.3334 | 0.6419 | 0.4389 | 0.2136 |
| CatBoost | 0.7215 | 0.4431 | 0.3240 | 0.3910 | 0.1433 | 0.3288 | 0.6559 | 0.4381 | 0.2097 |
| Logistic Regression | 0.7190 | 0.4380 | 0.3169 | 0.3877 | 0.1437 | 0.3361 | 0.6202 | 0.4360 | 0.2193 |
| Linear SVM | 0.7189 | 0.4379 | 0.3177 | 0.3869 | 0.1437 | 0.3295 | 0.6400 | 0.4350 | 0.2118 |
| Probit Regression | 0.7189 | 0.4379 | 0.3169 | 0.3874 | 0.1437 | 0.3260 | 0.6514 | 0.4345 | 0.2104 |
| Random Forest | 0.7181 | 0.4362 | 0.3175 | 0.3862 | 0.1439 | 0.3433 | 0.5921 | 0.4346 | 0.2341 |
| XGBoost | 0.7168 | 0.4335 | 0.3180 | 0.3837 | 0.1441 | 0.3351 | 0.6205 | 0.4352 | 0.2188 |

## RQ2: Random Feature-Group Comparison

| Model | Feature Group | AUC | AP | Brier | F1 |
|---|---|---:|---:|---:|---:|
| Explainable Boosting Machine | Borrower + Financial | 0.7223 | 0.3926 | 0.1431 | 0.4382 |
| Explainable Boosting Machine | Borrower-only | 0.6010 | 0.2622 | 0.1567 | 0.3563 |
| Explainable Boosting Machine | Combined | 0.7231 | 0.3935 | 0.1430 | 0.4389 |
| Explainable Boosting Machine | Financial-only | 0.7167 | 0.3846 | 0.1440 | 0.4306 |
| Explainable Boosting Machine | Grade-only | 0.6815 | 0.3131 | 0.1488 | 0.4102 |
| LightGBM | Borrower + Financial | 0.7263 | 0.3974 | 0.1425 | 0.4412 |
| LightGBM | Borrower-only | 0.6016 | 0.2630 | 0.1566 | 0.3561 |
| LightGBM | Combined | 0.7267 | 0.3984 | 0.1424 | 0.4425 |
| LightGBM | Financial-only | 0.7210 | 0.3894 | 0.1434 | 0.4375 |
| LightGBM | Grade-only | 0.6815 | 0.3131 | 0.1488 | 0.4102 |
| Logistic Regression | Borrower + Financial | 0.7172 | 0.3859 | 0.1440 | 0.4346 |
| Logistic Regression | Borrower-only | 0.5994 | 0.2611 | 0.1568 | 0.3552 |
| Logistic Regression | Combined | 0.7190 | 0.3877 | 0.1437 | 0.4360 |
| Logistic Regression | Financial-only | 0.7114 | 0.3779 | 0.1449 | 0.4291 |
| Logistic Regression | Grade-only | 0.6815 | 0.3131 | 0.1488 | 0.4102 |

## RQ3: Random Interpretability

- Tree-based model selected for SHAP by validation AUC: LightGBM (validation AUC 0.7272; final-test AUC 0.7267).
- Logistic Regression coefficients and odds ratios are extracted directly from the same full-training fitted model used for prediction.
- SHAP summary, SHAP bar plot, and top-feature table use only the configured random sample from the final test set.
- EBM global importance and key-variable shape functions are in the split figures and tables directories.
- Dedicated EBM plots for int_rate, dti, loan_amnt, annual_inc, and fico_score use original cleaned feature units rather than standardised model-input values.

## RQ1: Temporal Combined-Feature Results

| Model | AUC | Gini | KS | AP | Brier | Precision | Recall | F1 | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP Neural Network | 0.7141 | 0.4281 | 0.3122 | 0.3967 | 0.1539 | 0.3305 | 0.7074 | 0.4505 | 0.2231 |
| LightGBM | 0.7102 | 0.4203 | 0.3079 | 0.3875 | 0.1549 | 0.3337 | 0.6911 | 0.4501 | 0.2317 |
| FT-Transformer | 0.7101 | 0.4203 | 0.3076 | 0.3900 | 0.1546 | 0.3448 | 0.6384 | 0.4477 | 0.2523 |
| Explainable Boosting Machine | 0.7096 | 0.4192 | 0.3048 | 0.3890 | 0.1549 | 0.3369 | 0.6660 | 0.4474 | 0.2335 |
| Logistic Regression | 0.7057 | 0.4114 | 0.3011 | 0.3859 | 0.1554 | 0.3399 | 0.6450 | 0.4452 | 0.2403 |
| Linear SVM | 0.7056 | 0.4113 | 0.3015 | 0.3853 | 0.1555 | 0.3305 | 0.6858 | 0.4461 | 0.2265 |
| Probit Regression | 0.7055 | 0.4111 | 0.3004 | 0.3852 | 0.1554 | 0.3375 | 0.6551 | 0.4455 | 0.2407 |
| CatBoost | 0.7050 | 0.4101 | 0.3007 | 0.3820 | 0.1558 | 0.3375 | 0.6575 | 0.4460 | 0.2443 |
| XGBoost | 0.7037 | 0.4074 | 0.2973 | 0.3806 | 0.1561 | 0.3280 | 0.6888 | 0.4444 | 0.2341 |
| Random Forest | 0.7030 | 0.4060 | 0.2976 | 0.3790 | 0.1559 | 0.3293 | 0.6814 | 0.4440 | 0.2383 |

## RQ2: Temporal Feature-Group Comparison

| Model | Feature Group | AUC | AP | Brier | F1 |
|---|---|---:|---:|---:|---:|
| Explainable Boosting Machine | Borrower + Financial | 0.7092 | 0.3883 | 0.1552 | 0.4470 |
| Explainable Boosting Machine | Borrower-only | 0.6126 | 0.2957 | 0.1660 | 0.3837 |
| Explainable Boosting Machine | Combined | 0.7096 | 0.3890 | 0.1549 | 0.4474 |
| Explainable Boosting Machine | Financial-only | 0.7006 | 0.3757 | 0.1570 | 0.4413 |
| Explainable Boosting Machine | Grade-only | 0.6720 | 0.3234 | 0.1597 | 0.4290 |
| LightGBM | Borrower + Financial | 0.7096 | 0.3867 | 0.1553 | 0.4490 |
| LightGBM | Borrower-only | 0.6131 | 0.2913 | 0.1660 | 0.3840 |
| LightGBM | Combined | 0.7102 | 0.3875 | 0.1549 | 0.4501 |
| LightGBM | Financial-only | 0.7027 | 0.3787 | 0.1564 | 0.4442 |
| LightGBM | Grade-only | 0.6720 | 0.3234 | 0.1597 | 0.4290 |
| Logistic Regression | Borrower + Financial | 0.7046 | 0.3844 | 0.1560 | 0.4440 |
| Logistic Regression | Borrower-only | 0.6124 | 0.2953 | 0.1660 | 0.3832 |
| Logistic Regression | Combined | 0.7057 | 0.3859 | 0.1554 | 0.4452 |
| Logistic Regression | Financial-only | 0.6956 | 0.3731 | 0.1576 | 0.4388 |
| Logistic Regression | Grade-only | 0.6720 | 0.3234 | 0.1597 | 0.4290 |

## RQ3: Temporal Interpretability

- Tree-based model selected for SHAP by validation AUC: LightGBM (validation AUC 0.7203; final-test AUC 0.7102).
- Logistic Regression coefficients and odds ratios are extracted directly from the same full-training fitted model used for prediction.
- SHAP summary, SHAP bar plot, and top-feature table use only the configured random sample from the final test set.
- EBM global importance and key-variable shape functions are in the split figures and tables directories.
- Dedicated EBM plots for int_rate, dti, loan_amnt, annual_inc, and fico_score use original cleaned feature units rather than standardised model-input values.

## RQ3: Integrated Interpretability Findings

### SHAP top-feature interpretation

The final-test SHAP sample identifies the following leading contributors to the best tree-based model:

| Rank | Feature | Mean absolute SHAP |
|---:|---|---:|
| 1 | int_rate | 0.3101 |
| 2 | term_months | 0.1899 |
| 3 | dti | 0.1306 |
| 4 | acc_open_past_24mths | 0.1297 |
| 5 | grade_A | 0.1212 |
| 6 | fico_score | 0.1043 |
| 7 | loan_amnt | 0.0941 |
| 8 | grade_B | 0.0789 |
| 9 | avg_cur_bal | 0.0722 |
| 10 | annual_inc | 0.0720 |

Higher interest rates, longer terms, higher DTI, and more recently opened accounts generally increase predicted risk, while higher FICO scores reduce it. These are predictive associations, not causal effects.

### EBM global importance and shape functions

| Rank | Feature or term | Importance |
|---:|---|---:|
| 1 | term_months | 0.2265 |
| 2 | loan_amnt | 0.1701 |
| 3 | acc_open_past_24mths | 0.1612 |
| 4 | grade_A | 0.1541 |
| 5 | dti | 0.1540 |
| 6 | total_acc | 0.1211 |
| 7 | int_rate | 0.1047 |
| 8 | grade_B | 0.1033 |
| 9 | total_il_high_credit_limit | 0.0995 |
| 10 | fico_score | 0.0918 |

The original-scale EBM plots show increasing risk contributions for DTI and loan amount over much of their ranges, and a decreasing contribution as FICO score rises. Interest-rate and annual-income functions are more locally non-linear and should not be interpreted causally.

### Logistic Regression coefficients and odds ratios

The coefficients and odds ratios below are extracted directly from the same full-training fitted Logistic model that generated the reported probabilities and performance metrics.

Numeric coefficients use the model's standardised, winsorised input scale; odds ratios therefore represent a one-standard-deviation increase in the cleaned feature, holding other variables constant.

| Feature | Coefficient | Odds ratio | Sign | Direction |
|---|---:|---:|---|---|
| int_rate | 0.1318 | 1.1409 | positive | Higher values are associated with higher default odds |
| fico_score | -0.1292 | 0.8788 | negative | Higher values are associated with lower default odds |
| dti | 0.1825 | 1.2002 | positive | Higher values are associated with higher default odds |
| annual_inc | -0.0706 | 0.9318 | negative | Higher values are associated with lower default odds |
| loan_amnt | -0.1308 | 0.8774 | negative | Higher values are associated with lower default odds |
| term_months | 0.3445 | 1.4113 | positive | Higher values are associated with higher default odds |
| revol_util | 0.0433 | 1.0442 | positive | Higher values are associated with higher default odds |
| credit_history_months | 0.0160 | 1.0162 | positive | Higher values are associated with higher default odds |

## Run-Level Model Comparison

The nominally highest random-split model was LightGBM (AUC 0.7267), compared with Logistic Regression at 0.7190, a small absolute AUC gain of 0.0077. Under the temporal split, the corresponding difference was 0.0084 (0.7141 versus 0.7057). Small differences between leading models should not be overinterpreted without paired uncertainty assessment.

All reported Brier scores and reliability curves use model-specific sigmoid maps fitted on validation predictions and applied unchanged to test predictions. Raw outputs are retained separately to document the effects of score scale and class weighting before calibration.

## Key Result Files

- `random_split/tables/main_model_comparison_random_combined.csv`
- `temporal_split/tables/model_metrics_temporal_combined.csv`
- `random_split/tables/feature_group_model_performance.csv` and the corresponding temporal table
- `random_split/figures/predictive_performance_by_feature_group.png` and the corresponding temporal figure
- `random_split/tables/ebm_global_importance.csv` and the corresponding temporal table
- `random_split/tables/logistic_key_coefficients_for_interpretation.csv` and the corresponding temporal table
- Sample-representativeness tables are retained in `common/tables`.

## Consistency

All bundled tables and figures derive from the same completed run. Logistic predictive performance and the main coefficient/odds-ratio interpretation refer to the same full-training fitted estimator.
