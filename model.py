import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier
import os
import joblib
import optuna
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
from preprocessing import create_target
from feature_engineering import create_delta_features, preprocess_data
from scipy import stats


# XGBoost hyperparameter keys that optuna_search / _fit_xgb understand
_XGB_HPARAM_KEYS = {
    'n_estimators', 'max_depth', 'learning_rate', 'subsample',
    'colsample_bytree', 'colsample_bylevel', 'colsample_bynode',
    'min_child_weight', 'gamma', 'reg_alpha', 'reg_lambda', 'max_delta_step',
}

def mean_ci(data, conf=0.95):
    """
    Compute mean and confidence interval using t-distribution.
    Returns: (mean, lower, upper, std)
    """
    n = len(data)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    sem = std / np.sqrt(n)
    ci = stats.t.ppf((1 + conf) / 2, df=n - 1) * sem
    return mean, mean - ci, mean + ci, std

def bootstrap_all_metrics_ci(
    y_true,
    y_pred,
    y_score,
    *,
    n_boot=1000,
    conf_level=0.95,
    random_state=42,
    max_attempts=10000,
    verbose=True,
):
    """Bootstrap CIs for accuracy, sensitivity, specificity, F1 (positive class), ROC AUC, PPV, and NPV.

    Keeps sampling until n_boot valid (both-class) resamples are collected, up to max_attempts.
    Returns a dict including a 'valid_samples' key showing how many attempts were needed:
      {
        'accuracy': (point, (lo, hi)),
        'sensitivity': (point, (lo, hi)),
        'specificity': (point, (lo, hi)),
        'f1': (point, (lo, hi)),
        'auc': (point, (lo, hi)),
        'ppv': (point, (lo, hi)),
        'npv': (point, (lo, hi)),
        'valid_samples': 'n_boot/total_attempts'  # e.g. '1000/1050'
      }
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    alpha = 1 - conf_level

    # Point estimates
    acc_pt = accuracy_score(y_true, y_pred)
    sens_pt = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spec_pt = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_pt = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    ppv_pt = precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    npv_pt = precision_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)
    auc_pt = np.nan
    if len(np.unique(y_true)) == 2:
        auc_pt = roc_auc_score(y_true, y_score)

    # Bootstrap arrays
    acc_b, sens_b, spec_b, f1_b = [], [], [], []
    ppv_b, npv_b = [], []
    auc_b = []
    n = len(y_true)
    total_attempts = 0
    skipped = 0

    # Keep sampling until we have n_boot valid samples (or hit max_attempts)
    while len(auc_b) < n_boot and total_attempts < max_attempts:
        total_attempts += 1
        idx = rng.choice(n, n, replace=True)
        bt_y = y_true[idx]

        # Skip if resample lacks both classes
        if len(np.unique(bt_y)) < 2:
            skipped += 1
            continue

        bt_pred = y_pred[idx]
        bt_score = y_score[idx]

        acc_b.append(accuracy_score(bt_y, bt_pred))
        sens_b.append(recall_score(bt_y, bt_pred, pos_label=1, zero_division=0))
        spec_b.append(recall_score(bt_y, bt_pred, pos_label=0, zero_division=0))
        f1_b.append(f1_score(bt_y, bt_pred, pos_label=1, zero_division=0))
        ppv_b.append(precision_score(bt_y, bt_pred, pos_label=1, average="binary", zero_division=0))
        npv_b.append(precision_score(bt_y, bt_pred, pos_label=0, average="binary", zero_division=0))

        try:
            auc_b.append(roc_auc_score(bt_y, bt_score))
        except ValueError:
            # Shouldn't happen if unique check passed, but guard anyway
            skipped += 1
            # Remove the metrics we just appended since AUC failed
            acc_b.pop()
            sens_b.pop()
            spec_b.pop()
            f1_b.pop()
            ppv_b.pop()
            npv_b.pop()

    valid_samples_str = f"{len(auc_b)}/{total_attempts}"

    if verbose:
        print(f"Bootstrap: valid={len(auc_b)}, total_attempts={total_attempts}, skipped={skipped} ({valid_samples_str})")

    # Helper to CI from list
    def ci_bounds(vals):
        arr = np.array(vals)
        return (
            float(np.percentile(arr, 100 * (alpha / 2))),
            float(np.percentile(arr, 100 * (1 - alpha / 2))),
        )

    results = {
        "accuracy": (acc_pt, ci_bounds(acc_b) if acc_b else (np.nan, np.nan)),
        "sensitivity": (sens_pt, ci_bounds(sens_b) if sens_b else (np.nan, np.nan)),
        "specificity": (spec_pt, ci_bounds(spec_b) if spec_b else (np.nan, np.nan)),
        "f1": (f1_pt, ci_bounds(f1_b) if f1_b else (np.nan, np.nan)),
        "auc": (auc_pt, ci_bounds(auc_b) if auc_b else (np.nan, np.nan)),
        "ppv": (ppv_pt, ci_bounds(ppv_b) if ppv_b else (np.nan, np.nan)),
        "npv": (npv_pt, ci_bounds(npv_b) if npv_b else (np.nan, np.nan)),
        "valid_samples": valid_samples_str,
    }

    return results

# Preprocess the data


def _fit_xgb(X_tr, X_te, y_tr, params, n_jobs=None, scale=False, random_state=42):
    """Fit (optional scaler +) XGBoost on training data; return predictions on test data.

    Parameters
    ----------
    params : dict
        Hyperparameter dict. All keys in _XGB_HPARAM_KEYS are forwarded to
        XGBClassifier; unrecognised keys are ignored.
    scale : bool, default False
        When True, apply StandardScaler to X_tr / X_te before fitting.

    Returns (model, imputer, scaler, y_pred, y_proba).
    imputer is always None (reserved for future re-addition).
    scaler  is None when scale=False.
    """
    if scale:
        scaler = StandardScaler()
        X_tr_proc = scaler.fit_transform(X_tr)
        X_te_proc = scaler.transform(X_te)
    else:
        scaler = None
        X_tr_proc = X_tr
        X_te_proc = X_te

    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    xgb_kwargs = {k: v for k, v in params.items() if k in _XGB_HPARAM_KEYS}

    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=spw,
        enable_categorical=True,
        n_jobs=n_jobs,
        random_state=random_state,
        **xgb_kwargs,
    )
    model.fit(X_tr_proc, y_tr)
    y_proba = model.predict_proba(X_te_proc)[:, 1]
    y_pred = model.predict(X_te_proc)
    return model, None, scaler, y_pred, y_proba


def _print_bootstrap_ci(metrics):
    """Print bootstrap 95% CI summary block to stdout."""
    acc_pt,  (acc_lo,  acc_hi)  = metrics["accuracy"]
    sens_pt, (sens_lo, sens_hi) = metrics["sensitivity"]
    spec_pt, (spec_lo, spec_hi) = metrics["specificity"]
    f1_pt,   (f1_lo,   f1_hi)   = metrics["f1"]
    auc_pt,  (auc_lo,  auc_hi)  = metrics["auc"]
    ppv_pt,  (ppv_lo,  ppv_hi)  = metrics["ppv"]
    npv_pt,  (npv_lo,  npv_hi)  = metrics["npv"]

    print(f"\nBootstrap 95% CI (n=1000, valid_samples={metrics['valid_samples']}):")
    print(f"- Accuracy: {acc_pt:.3f} (CI: {acc_lo:.3f}, {acc_hi:.3f}) range={acc_hi - acc_lo:.3f}")
    print(f"- Sensitivity: {sens_pt:.3f} (CI: {sens_lo:.3f}, {sens_hi:.3f}) range={sens_hi - sens_lo:.3f}")
    print(f"- Specificity: {spec_pt:.3f} (CI: {spec_lo:.3f}, {spec_hi:.3f}) range={spec_hi - spec_lo:.3f}")
    print(f"- F1 (positive class): {f1_pt:.3f} (CI: {f1_lo:.3f}, {f1_hi:.3f}) range={f1_hi - f1_lo:.3f}")
    if not np.isnan(auc_lo):
        print(f"- ROC AUC: {auc_pt:.3f} (CI: {auc_lo:.3f}, {auc_hi:.3f}) range={auc_hi - auc_lo:.3f}")
    else:
        print(f"- ROC AUC: {auc_pt:.3f} (CI unavailable; too few valid resamples)")
    print(f"- PPV: {ppv_pt:.3f} (CI: {ppv_lo:.3f}, {ppv_hi:.3f}) range={ppv_hi - ppv_lo:.3f}")
    print(f"- NPV: {npv_pt:.3f} (CI: {npv_lo:.3f}, {npv_hi:.3f}) range={npv_hi - npv_lo:.3f}")


def _write_bootstrap_ci(f, bm):
    """Write bootstrap 95% CI summary block to an open file handle."""
    f.write(f"Bootstrap 95% CI (n=1000, valid_samples={bm['valid_samples']}):\n")
    f.write(f"- Accuracy: {bm['accuracy'][0]:.3f} (CI: {bm['accuracy'][1][0]:.3f}, {bm['accuracy'][1][1]:.3f}) range={bm['accuracy'][1][1] - bm['accuracy'][1][0]:.3f}\n")
    f.write(f"- Sensitivity: {bm['sensitivity'][0]:.3f} (CI: {bm['sensitivity'][1][0]:.3f}, {bm['sensitivity'][1][1]:.3f}) range={bm['sensitivity'][1][1] - bm['sensitivity'][1][0]:.3f}\n")
    f.write(f"- Specificity: {bm['specificity'][0]:.3f} (CI: {bm['specificity'][1][0]:.3f}, {bm['specificity'][1][1]:.3f}) range={bm['specificity'][1][1] - bm['specificity'][1][0]:.3f}\n")
    f.write(f"- F1 (positive class): {bm['f1'][0]:.3f} (CI: {bm['f1'][1][0]:.3f}, {bm['f1'][1][1]:.3f}) range={bm['f1'][1][1] - bm['f1'][1][0]:.3f}\n")
    auc_lo, auc_hi = bm['auc'][1]
    if not np.isnan(auc_lo):
        f.write(f"- ROC AUC: {bm['auc'][0]:.3f} (CI: {auc_lo:.3f}, {auc_hi:.3f}) range={auc_hi - auc_lo:.3f}\n")
    else:
        f.write(f"- ROC AUC: {bm['auc'][0]:.3f} (CI unavailable; too few valid resamples)\n")
    f.write(f"- PPV: {bm['ppv'][0]:.3f} (CI: {bm['ppv'][1][0]:.3f}, {bm['ppv'][1][1]:.3f}) range={bm['ppv'][1][1] - bm['ppv'][1][0]:.3f}\n")
    f.write(f"- NPV: {bm['npv'][0]:.3f} (CI: {bm['npv'][1][0]:.3f}, {bm['npv'][1][1]:.3f}) range={bm['npv'][1][1] - bm['npv'][1][0]:.3f}\n")

# For the training of the best model with the best set of hyperparameters, prints out the whole performance report.
def build_model_final(X_train, X_test, y_train, y_test, model_dict, feature_names, charts_dir=None, random_state=42):

    from visualization import plot_shap_summary, plot_pr_curve

    model, imputer, scaler, y_pred, y_proba = _fit_xgb(X_train, X_test, y_train, model_dict, random_state=random_state)

    print("Classification Report:")
    cr_str = classification_report(y_test, y_pred)
    print(cr_str)

    # Base ROC AUC
    base_auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC AUC Score: {base_auc:.4f}")

    # Base PR-AUC
    base_avg_precision = average_precision_score(y_test, y_proba)
    print(f"PR-AUC (Average Precision): {base_avg_precision:.4f}")

    # Bootstrap CIs for all metrics
    metrics = bootstrap_all_metrics_ci(
        y_test,
        y_pred,
        y_proba,
        n_boot=1000,
        conf_level=0.95,
        random_state=42,
        verbose=True,
    )

    _print_bootstrap_ci(metrics)

    # SHAP analysis on training data
    shap_save = os.path.join(charts_dir, "shap_summary.png") if charts_dir else None
    # Keep the result explicitly dynamic: older visualization helpers annotate
    # this function as NoReturn even though they return (shap_values, plot).
    shap_result: object = plot_shap_summary(
        model, X_train, list(feature_names), save_path=shap_save
    )
    shap_values = shap_result[0] if isinstance(shap_result, tuple) else shap_result

    # PR curve on test data
    pr_save = os.path.join(charts_dir, "pr_curve.png") if charts_dir else None
    plot_pr_curve(y_test, y_proba, save_path=pr_save)

    summary = {
        "classification_report": cr_str,
        "base_auc": float(base_auc),
        "base_avg_precision": float(base_avg_precision),
        "bootstrap_metrics": metrics,
        "y_true": y_test,
        "y_proba": y_proba,
        "y_pred": y_pred,
        "feature_names": list(feature_names),
        "feature_importances": model.feature_importances_,
        "shap_values": shap_values,
        "X_train": X_train,
    }
    return model, feature_names, imputer, scaler, summary

# To train models for cross-validation. Returns only a CV score. Does not print out anything.
def build_model(X_train, X_test, y_train, y_test, model_dict, feature_names, xgb_n_jobs=None, objective_metric='auc'):

    _, _, _, y_pred, y_proba = _fit_xgb(X_train, X_test, y_train, model_dict, n_jobs=xgb_n_jobs)
    if objective_metric == 'avg_precision':
        return average_precision_score(y_test, y_proba)
    return roc_auc_score(y_test, y_proba)


def _suggest_param(trial, name, spec):
    """Translate a params-dict entry into an Optuna trial suggestion.

    spec conventions:
      scalar                    -> fixed value (no suggestion)
      (int_lo, int_hi)          -> trial.suggest_int
      (float_lo, float_hi)      -> trial.suggest_float
      (float_lo, float_hi, 'log') -> trial.suggest_float(..., log=True)
    """
    if not isinstance(spec, tuple):
        return spec
    if len(spec) == 3 and spec[2] == 'log':
        return trial.suggest_float(name, spec[0], spec[1], log=True)
    lo, hi = spec[0], spec[1]
    if isinstance(lo, int) and isinstance(hi, int):
        return trial.suggest_int(name, lo, hi)
    return trial.suggest_float(name, lo, hi)


def optuna_search(
    x, y, params, csv_path, feature_names,
    n_trials=100, n_jobs=1, objective_metric='auc', random_state=42
):
    """Bayesian hyperparameter search using Optuna (TPESampler).

    params dict convention:
      scalar value              -> fixed hyperparameter
      (int_lo, int_hi)          -> suggest_int search range
      (float_lo, float_hi)      -> suggest_float search range
      (float_lo, float_hi, 'log') -> suggest_float log-scale search range

    Supported hyperparameter keys (subset of _XGB_HPARAM_KEYS):
      n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
      colsample_bylevel, colsample_bynode, min_child_weight, gamma,
      reg_alpha, reg_lambda, max_delta_step

    Returns the best hyperparameter dict and saves per-trial scores to csv_path.
    """
    y_arr = np.array(y)

    # Determine feasible n_splits based on minority class count
    if np.unique(y_arr).size == 2:
        min_class = int(np.bincount(y_arr).min())
    else:
        min_class = len(y_arr)
    n_splits = max(2, min(5, min_class))
    if n_splits < 5:
        print(f"Note: Using StratifiedKFold with n_splits={n_splits} due to minority class size={min_class}")
    else:
        print(f"Using StratifiedKFold with n_splits={n_splits}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = [
        (x[tr], x[te], y_arr[tr], y_arr[te])
        for tr, te in skf.split(x, y_arr)
    ]

    trial_records = []

    def objective(trial):
        trial_params = {k: _suggest_param(trial, k, v) for k, v in params.items()}
        score_sum = sum(
            build_model(
                X_tr, X_te, y_tr, y_te,
                trial_params, feature_names,
                xgb_n_jobs=1,
                objective_metric=objective_metric,
            )
            for X_tr, X_te, y_tr, y_te in folds
        )
        return score_sum / n_splits

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state))

    print(f"Optuna search: {n_trials} trials, objective={objective_metric}, n_jobs={n_jobs}")
    with tqdm(total=n_trials, desc="Optuna trials", unit="trial") as pbar:
        def _cb(study, trial):
            record = {'trial': trial.number, 'score': trial.value}
            record.update(trial.params)
            trial_records.append(record)
            pbar.update(1)

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[_cb])

    # Save per-trial results
    dirn = os.path.dirname(csv_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    pd.DataFrame(trial_records).to_csv(csv_path, index=False)

    best = study.best_params
    # Fill in any fixed params that weren't part of the search
    for k, v in params.items():
        if not isinstance(v, tuple) and k not in best:
            best[k] = v
    return best

# Input an unprocessed dataset, progression type for preprocessing, and the parameter search space
# to get a best performing model with the best set of hyperparameters found from Bayesian optimization.
# Save per-trial scores to the input csv path.

def train_best_model(
    dataset, progression_type, params, csv_path,
    save_dir="saved_models", model_base_name=None,
    save_artifacts=True, n_jobs=1,
    n_trials=100, objective_metric='auc', random_state=42
):
    
    dataset = dataset.copy()

    # --- Step 1: Create target variable on the raw DataFrame (needed for stratified split) ---
    dataset['target'] = dataset['Progression'].apply(create_target, progression_type=progression_type)
    # TODO: make this random_state variable be adjustable by the user. 
   
    train_idx, test_idx = train_test_split(
        dataset.index, test_size=0.2, random_state=random_state, stratify=dataset['target']
    )
    df_train_raw = dataset.loc[train_idx].copy()
    df_test_raw = dataset.loc[test_idx].copy()

    # --- Step 2: Feature engineering + preprocessing on each split independently ---
    y_train_target = df_train_raw['target'].values
    y_test_target = df_test_raw['target'].values
    df_train_raw = df_train_raw.drop(columns=['target'])
    df_test_raw = df_test_raw.drop(columns=['target'])

    df_train_feat = create_delta_features(df_train_raw)
    df_test_feat = create_delta_features(df_test_raw)

    df_train_feat['target'] = y_train_target
    df_test_feat['target'] = y_test_target

    processed_train, _, _ = preprocess_data(df_train_feat, progression_type)
    processed_test, _, _ = preprocess_data(df_test_feat, progression_type)

    feature_names = [c for c in processed_train.columns if c != 'target']
    for col in feature_names:
        if col not in processed_test.columns:
            processed_test[col] = np.nan
    processed_test = processed_test[feature_names + ['target']]

    X_train = processed_train.drop(columns=['target']).values
    X_test = processed_test.drop(columns=['target']).values
    y_train = processed_train['target'].values
    y_test = processed_test['target'].values

    # --- Step 3: Bayesian hyperparameter search ---
    model_dict = optuna_search(
        X_train, y_train, params, csv_path, feature_names,
        n_trials=n_trials, n_jobs=n_jobs, objective_metric=objective_metric, random_state=random_state
    )

    # --- Step 4: Final model with full report ---
    charts_dir = save_dir if save_artifacts else None
    model, columns, imputer, scaler, summary = build_model_final(
        X_train, X_test, y_train, y_test, model_dict, feature_names,
        charts_dir=charts_dir, random_state=random_state
    )

    try:
        model.feature_names_in_ = np.array(columns)
    except Exception:
        pass

    if save_artifacts:
        os.makedirs(save_dir, exist_ok=True)
        base = model_base_name or os.path.splitext(os.path.basename(csv_path))[0]
        model_path = os.path.join(save_dir, f"{base}_model_{progression_type}.pkl")
        scaler_path = os.path.join(save_dir, f"{base}_scaler_{progression_type}.pkl")
        imputer_path = os.path.join(save_dir, f"{base}_imputer_{progression_type}.pkl")

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(imputer, imputer_path)

        print("\nSaved artifacts:")
        print(f"- Model:   {model_path}")
        print(f"- Scaler:  {scaler_path}")
        print(f"- Imputer: {imputer_path}")

        report_path = os.path.join(save_dir, f"{base}_report_{progression_type}.txt")
        with open(report_path, "w") as f:
            f.write(f"Dataset base: {base}\n")
            f.write(f"Progression type: {progression_type}\n")
            f.write(f"Objective metric: {objective_metric}\n")
            f.write(f"n_trials: {n_trials}\n")
            f.write(f"Random state: {random_state}\n")
            f.write("Best hyperparameters:\n")
            for k, v in model_dict.items():
                f.write(f"  {k}: {v}\n")
            f.write("\nClassification Report:\n")
            f.write(summary["classification_report"] + "\n")
            f.write(f"\nBase ROC AUC: {summary['base_auc']:.4f}\n")
            f.write(f"PR-AUC (Average Precision): {summary['base_avg_precision']:.4f}\n")
            bm = summary["bootstrap_metrics"]
            _write_bootstrap_ci(f, bm)
        print(f"- Report:  {report_path}")

    return model, columns, summary


def train_best_model_from_split(
    train_df, test_df, progression_type, params, csv_path,
    save_dir="saved_models", model_base_name=None,
    save_artifacts=True, n_jobs=1,
    n_trials=100, objective_metric='auc', random_state=42
):
    """
    Train the best model on already-split (and, for MICE experiments,
    already-imputed) train/test DataFrames.

    Unlike train_best_model, this does NOT perform its own train/test split: the
    caller is responsible for splitting — and, for MICE experiments, for fitting
    the imputer on the training set only — so no test-set information leaks into
    imputation. Both DataFrames must already contain a 'target' column.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    y_train = train_df['target'].values
    y_test = test_df['target'].values
    df_train_raw = train_df.drop(columns=['target'])
    df_test_raw = test_df.drop(columns=['target'])

    df_train_feat = create_delta_features(df_train_raw)
    df_test_feat = create_delta_features(df_test_raw)

    df_train_feat['target'] = y_train
    df_test_feat['target'] = y_test

    processed_train, _, _ = preprocess_data(df_train_feat, progression_type)
    processed_test, _, _ = preprocess_data(df_test_feat, progression_type)

    feature_names = [c for c in processed_train.columns if c != 'target']
    for col in feature_names:
        if col not in processed_test.columns:
            processed_test[col] = np.nan
    processed_test = processed_test[feature_names + ['target']]

    X_train = processed_train.drop(columns=['target']).values
    X_test = processed_test.drop(columns=['target']).values
    y_train = processed_train['target'].values
    y_test = processed_test['target'].values

    model_dict = optuna_search(
        X_train, y_train, params, csv_path, feature_names,
        n_trials=n_trials, n_jobs=n_jobs, objective_metric=objective_metric, random_state=random_state
    )

    charts_dir = save_dir if save_artifacts else None
    model, columns, imputer, scaler, summary = build_model_final(
        X_train, X_test, y_train, y_test, model_dict, feature_names,
        charts_dir=charts_dir, random_state=random_state
    )

    try:
        model.feature_names_in_ = np.array(columns)
    except Exception:
        pass

    if save_artifacts:
        os.makedirs(save_dir, exist_ok=True)
        base = model_base_name or os.path.splitext(os.path.basename(csv_path))[0]
        model_path = os.path.join(save_dir, f"{base}_model_{progression_type}.pkl")
        scaler_path = os.path.join(save_dir, f"{base}_scaler_{progression_type}.pkl")
        imputer_path = os.path.join(save_dir, f"{base}_imputer_{progression_type}.pkl")

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(imputer, imputer_path)

        print("\nSaved artifacts:")
        print(f"- Model:   {model_path}")
        print(f"- Scaler:  {scaler_path}")
        print(f"- Imputer: {imputer_path}")

        report_path = os.path.join(save_dir, f"{base}_report_{progression_type}.txt")
        with open(report_path, "w") as f:
            f.write(f"Dataset base: {base}\n")
            f.write(f"Progression type: {progression_type}\n")
            f.write(f"Objective metric: {objective_metric}\n")
            f.write(f"n_trials: {n_trials}\n")
            f.write(f"Random state: {random_state}\n")
            f.write("Best hyperparameters:\n")
            for k, v in model_dict.items():
                f.write(f"  {k}: {v}\n")
            f.write("\nClassification Report:\n")
            f.write(summary["classification_report"] + "\n")
            f.write(f"\nBase ROC AUC: {summary['base_auc']:.4f}\n")
            f.write(f"PR-AUC (Average Precision): {summary['base_avg_precision']:.4f}\n")
            bm = summary["bootstrap_metrics"]
            _write_bootstrap_ci(f, bm)
        print(f"- Report:  {report_path}")

    return model, columns, summary

