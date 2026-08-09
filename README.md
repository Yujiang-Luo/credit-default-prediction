# LendingClub Credit-Default Prediction under a Unified Framework

This repository contains the code and reported outputs for a 500,000-loan LendingClub credit-default study. It compares statistical, conventional machine-learning, interpretable machine-learning and neural/tabular models under a common outer protocol for data, features, validation, calibration and evaluation. Model-internal training remains estimator-appropriate: imbalance controls are prespecified and applied only where the implementation supports them meaningfully.

## Research design

- Main benchmark: stratified random 64%/16%/20% train/validation/test split.
- Robustness assessment: month-based temporal split for later-cohort generalisation.
- RQ1: ten-model comparison on the Combined feature set.
- RQ2: five predefined information specifications evaluated with Logistic Regression, LightGBM and EBM.
- Unified framework: the outer comparison protocol is common across estimators, while model-native weighting or weighted-loss controls are used only where appropriate.
- RQ3: joint assessment of predictive performance, temporal robustness, probability reliability, interpretability and implementation/governance considerations.
- Probability protocol: a common sigmoid calibration map is fitted separately to each model's validation predictions and applied unchanged to its test predictions.

## Final run

- Run ID: `run_20260806T154718Z`
- Completed-loan population: 1,348,099
- Modelling sample: 500,000
- Random state: 42
- Audit: 105/105 checks passed
- Compute: GPU for MLP and modified FT-Transformer-style models; CPU for the remaining models.

The random-split ROC-AUC leaders were LightGBM (0.7267) and the MLP (0.7264), compared with 0.7190 for Logistic Regression. All models recorded lower discrimination in the temporal assessment.

## Repository structure

```text
src/                    Main end-to-end experimental pipeline
scripts/                Run-audit utility
data/                   Data acquisition and exclusion notes
docs/                   Final experiment summary and output guide
results/latest/         Figures, reporting tables and audit records
```

Large loan-level files are intentionally excluded from the GitHub version. The complete local backup retains them.

## Installation

Python 3.11 or later is recommended. Create an isolated environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA acceleration, install the PyTorch build appropriate for the local CUDA environment before running the full neural models.

The full reported experiment requires all packages listed in `requirements.txt`. The pipeline checks requested model dependencies before loading the loan archive and stops with a clear message if the environment is incomplete.

## Run the full experiment

From the repository root:

```powershell
python src/lendingclub_default_pipeline.py `
  --archive "C:\path\to\archive.zip" `
  --output-dir outputs\lendingclub_final `
  --sample-rows 500000 `
  --split both `
  --random-state 42 `
  --shap-sample-rows 5000
```

The pipeline creates a timestamped run folder containing common EDA outputs, random-split and temporal-split results, metadata, the code snapshot and an artefact manifest. The test partitions are reserved for final evaluation; validation partitions select hyperparameters, epochs, calibration maps and thresholds.

For a quicker environment check, add `--skip-tree-tuning --skip-ft-transformer --skip-mlp` and reduce `--sample-rows`. Such a run is not equivalent to the reported full experiment.

## Audit a completed run

```powershell
python scripts/audit_run.py outputs\lendingclub_final\run_YYYYMMDDTHHMMSSZ
```

## Results

The bundled `results/latest` directory contains all 31 final figures and the compact public result tables from the same latest run. Raw and uniformly calibrated probability outputs were retained in the complete local backup; large row-level prediction files are excluded here.

See [docs/RESULTS_GUIDE.md](docs/RESULTS_GUIDE.md) and [docs/EXPERIMENT_SUMMARY.md](docs/EXPERIMENT_SUMMARY.md) for details.

## Reproducibility and scope

The supplied results were generated on Windows 11 with Python 3.13.9. Exact package and hardware details are recorded in `results/latest/audit/reproducibility_environment.csv`. CatBoost was evaluated using the common processed representation rather than native ordered categorical encoding, and the modified FT-Transformer-style model uses the same common one-hot/scalar representation rather than canonical raw categorical tokenisation.

The analysis concerns accepted and completed LendingClub loans and therefore does not represent rejected applicants or unresolved loans. Interpretability outputs describe predictive associations and contributions, not causal effects.
