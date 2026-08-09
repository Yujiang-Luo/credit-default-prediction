# Results Guide

All files in `results/latest` come from final run `run_20260806T154718Z`.

## Common data and EDA outputs

`results/latest/common/figures` contains target distribution, default rates by grade and purpose, numerical distributions, the correlation heatmap and sample-representativeness diagnostics.

`results/latest/common/tables` contains variable definitions, preprocessing mappings, sample-versus-population diagnostics, target counts and model-complexity profiles.

## Random-split results

Key figures in `results/latest/random_split/figures`:

- `roc_curves_combined.png`
- `precision_recall_curves_combined.png`
- `gain_curves_combined.png`
- `calibration_curves_combined.png`
- `calibration_curves_combined_raw.png`
- `predictive_performance_by_feature_group.png`
- `best_tree_shap_summary_combined.png`
- `best_tree_shap_bar_combined.png`
- `ebm_global_feature_importance_combined.png`
- `ebm_shape_functions_original_scale_combined.png`

Key tables in `results/latest/random_split/tables` include the main model comparison, feature-group comparison, paired DeLong comparisons, calibrated and raw calibration bins, selected configurations, Logistic coefficients/odds ratios, SHAP summaries and EBM importance/shape-function coordinates.

## Temporal-split results

The matching files under `results/latest/temporal_split` report the out-of-time robustness assessment. Hyperparameters, calibration maps and thresholds were selected independently using the temporal validation partition.

## Audit records

`results/latest/audit` contains the run identifier, combined metrics, partition composition, selected configurations, reproducibility environment and the 105 run-audit checks.

## Intentionally excluded files

The GitHub package omits the cleaned 500,000-loan modelling sample and row-level calibrated/raw predictions. They remain in the complete desktop backup and can be regenerated from the source archive.

