



# AD-Early-Prediction

Early prediction of Alzheimer's disease (AD) progression from longitudinal clinical data using gradient-boosted trees and lead-time analysis.

## Overview

This repository implements a machine-learning pipeline that predicts progression along the Alzheimer's disease continuum from longitudinal, structured clinical data collected through the National Alzheimer's Coordinating Center (NACC) Uniform Data Set (UDS v3).

Two binary classifiers are trained:

| Task | Starts | Progresses to | Positive-class definition |
|------|--------|---------------|---------------------------|
| CN → MCI | Cognitively normal (CN) | Mild cognitive impairment (MCI) | any visit beyond CN |
| MCI → AD | Mild cognitive impairment (MCI) | Alzheimer's dementia (AD) | any visit reaching AD |

Each subject is represented as a fixed-width, **visit-agnostic** feature vector (slopes, means, ranges, first/last values, lagged values, hearing × vision interactions, and visit-interval statistics) derived from their longitudinal record, so the model handles a variable number of visits per subject.

Key components:

- **XGBoost** classifiers with class re-weighting and native categorical handling.
- **Optuna** Bayesian hyperparameter search.
- Bootstrap confidence intervals for accuracy, sensitivity, specificity, F1, ROC-AUC, PPV, and NPV.
- **Lead-time analysis** that evaluates how early a model detects conversion by truncating each subject's visit sequence and masking the most recent control visits to estimate false-alarm rates.
- Two experiment designs: **non-imputed** (10 random train/test splits) and **MICE-imputed** (10 imputation variants × 10 bootstrap draws), with imputation fitted on the training split only to prevent leakage.

## Repository contents

| File | Description |
|------|-------------|
| `preprocessing.py` | Raw NACC investigator CSV → one-row-per-subject datasets (`run_pipeline`, `build_subject_df`), sentinel cleaning, hearing/vision composites, and leak-free MICE imputation (`fit_imputer` / `transform_imputer`). |
| `feature_engineering.py` | Visit-agnostic feature engineering (`create_delta_features`) and feature selection (`preprocess_data`). |
| `model.py` | XGBoost training with Optuna search (`train_best_model`, `train_best_model_from_split`) and bootstrap-CI reporting. |
| `leadtime.py` | Lead-time evaluation (`run_leadtime`, `analyze_run`, `run_grid`). |
| `visualization.py` | Plotting helpers: SHAP, ROC, precision-recall, confusion matrix, feature importance. |
| `cohort_analysis.ipynb` | Demographic, visit-interval, and missingness analyses. |
| `leadtime_analysis.ipynb` | Full lead-time experiment notebook (grid search, dynamic masking, Brier scores). |
| `example_usage.ipynb` | Minimal end-to-end example: train, evaluate, and lead-time analysis with a synthetic-data fallback. |

## Installation

Requires Python 3.11+.

```bash
git clone <repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For development (running the test suite), also install:

```bash
pip install -r requirements-dev.txt
```

## Quick start

The fastest way to see the pipeline run is [`example_usage.ipynb`](example_usage.ipynb). It trains both classifiers and runs a lead-time analysis. Because the NACC data are restricted (see Data Availability below), the notebook falls back to a small synthetic cohort with the correct schema when the dataset files are absent, so it runs out of the box.

The full workflow has three steps:

1. **Preprocess** — build subject-level datasets from a raw NACC investigator CSV:

   ```python
   from preprocessing import run_pipeline

   run_pipeline(
       source_csv="investigator_ftldlbd_nacc72.csv",
       dest_dir="datasets/Dataset_v2_1",
       min_visits=2, max_visits=10, min_age=50,
       lead_time_pct=0.05, do_impute=False,
   )
   ```

   This writes `pooled_CN.csv`, `pooled_MCI_AD.csv`, `lead_time_CN.csv`, and `lead_time_MCI_AD.csv` (plus reverters).

2. **Train** — fit and tune a classifier:

   ```python
   from model import train_best_model

   model, columns, summary = train_best_model(
       dataset, progression_type="CN", params=params, csv_path="scores.csv",
   )
   ```

3. **Lead-time** — score truncated visit sequences and measure detection lead time:

   ```python
   from leadtime import run_leadtime, analyze_run

   run_leadtime("datasets/Dataset_v2_1/lead_time_CN.csv", "leadtime_cache", model, "CN")
   ```

## Reproducing the paper results

The reported results use the full experimental budgets, which are intentionally larger than the quick-start example:

- **Hyperparameter search:** `n_trials=1000` per model.
- **Non-imputed experiments:** 10 random stratified train/test splits (seeds derived from a master seed of 42).
- **MICE-imputed experiments:** 10 imputation variants × 10 stratified-bootstrap lead-time draws; the MICE imputer is fitted on the training split only (`fit_imputer`) and applied to the test/holdout sets with `transform_imputer`.
- **Lead-time operating points:** CN → MCI uses threshold 0.25 with mask length 3; MCI → AD uses threshold 0.30 with mask length 2.

The full experiment loops are implemented in `leadtime_analysis.ipynb`.

## Tests

Run from the repository root:

```bash
pytest
```

## Data Availability Statement

The individual-level participant data used in this study were obtained from the National Alzheimer’s Coordinating Center (NACC). Because of the terms of the NACC Data Use Agreement (DUA) and human subjects protection regulations, the raw data records cannot be publicly distributed or uploaded to GitHub. 

The data are available to qualified researchers upon request directly from NACC. To replicate the findings of this study, researchers can request the data by submitting a formal data request through the [National Alzheimer’s Coordinating Center Data Request Portal](https://www.naccdata.org/data-request-process). 

**Dataset Specification:** 

* **Data Freeze / Version:** December 2025 Data Freeze
* **NACC Forms Used:** Uniform Data Set (UDS) v3, Neuropathology (NP) Form
